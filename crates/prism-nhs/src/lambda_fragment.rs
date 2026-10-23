//! Physical grand-canonical multisite lambda-fragment subsystem.
//!
//! This module defines the source-of-truth topology contract, one differentiable
//! soft-core pair Hamiltonian (energy, force and `dU/dlambda`), endpoint-only
//! observation semantics, and NCMC detailed-balance receipts.  The contract is
//! deliberately strict: a production lambda-fragment system is not admitted
//! without explicit periodic solvent, PME, complete rigid-water triplets and a
//! declared standard state.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
#[cfg(unix)]
use std::os::unix::io::{AsRawFd, FromRawFd};

use crate::input::PrismPrepTopology;
use prismcode::binary_runtime::opcode_id_for_name;
use prismcode::operand_ledger::RuntimeOrigin;
use prismcode::qnanl_artifact::QnanlArtifactFamily;
use prismcode::qnanl_artifact::{decode_packet_bytes, encode_bytes, QnanlEncodeOptions};
use std::path::{Path, PathBuf};

/// Canonical JSON ABI for 128-bit counter coordinates. `serde_json` can emit
/// a numeric `u128` but cannot losslessly deserialize that representation on
/// this production dependency surface. Persisted counters are therefore
/// decimal strings, with numeric forms rejected rather than accepted through
/// a value-dependent compatibility path.
mod u128_decimal_string {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(value: &u128, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&value.to_string())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<u128, D::Error>
    where
        D: Deserializer<'de>,
    {
        let encoded = String::deserialize(deserializer)?;
        if encoded.is_empty()
            || (encoded.len() > 1 && encoded.starts_with('0'))
            || !encoded.bytes().all(|byte| byte.is_ascii_digit())
        {
            return Err(serde::de::Error::custom(
                "u128 counter must be one canonical unsigned decimal string",
            ));
        }
        encoded
            .parse::<u128>()
            .map_err(|_| serde::de::Error::custom("u128 counter decimal string overflows"))
    }
}

/// Coulomb conversion factor for charges in elementary-charge units, distance
/// in Angstrom and energy in kcal/mol.
pub const COULOMB_KCAL_A_PER_MOL_E2: f64 = 332.063_713_299;

/// Canonical PrismCode opcodes assigned to the subsystem.
pub mod prismcode_opcodes {
    /// Native lambda-fragment pair force execution.
    pub const FORCE: &str = "MD.LAMBDA_FRAGMENT_FORCE";
    /// Native dU/dlambda accumulation.
    pub const DUDL: &str = "MD.LAMBDA_DUDL_ACCUMULATE";
    /// NCMC parameter perturbation.
    pub const NCMC_PERTURB: &str = "MD.LAMBDA_NCMC_PERTURB";
    /// NCMC dynamical propagation.
    pub const NCMC_PROPAGATE: &str = "MD.LAMBDA_NCMC_PROPAGATE";
    /// Metropolis accept/reject adjudication.
    pub const NCMC_ACCEPT: &str = "MD.LAMBDA_NCMC_ACCEPT";
    /// Physical endpoint observation.
    pub const ENDPOINT_OBSERVE: &str = "MD.LAMBDA_ENDPOINT_OBSERVE";
    /// Accepted chemical-state commit.
    pub const STATE_COMMIT: &str = "MD.LAMBDA_STATE_COMMIT";
    /// Rejected proposal rollback/no-change resolution.
    pub const STATE_ROLLBACK: &str = "MD.LAMBDA_STATE_ROLLBACK";
    /// Closed-interval proof for a physical no-probe lane.
    pub const NO_PROBE_ABSENCE: &str = "MD.LAMBDA_NO_PROBE_ABSENCE";
}

/// Complete topology extension for a lambda-fragment experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LambdaFragmentSystem {
    /// Schema discriminator.
    pub schema_kind: String,
    /// Schema version; version two carries exact endpoint bonded planes and
    /// constant-topology state-transition maps.
    pub schema_version: u32,
    /// Stable experiment identity, unique within an evidence lineage.
    pub experiment_id: String,
    /// Explicit periodic boundary contract.
    pub periodic_box: PeriodicBoxContract,
    /// Reciprocal-space electrostatics contract.
    pub pme: PmeContract,
    /// Rigid water/SETTLE contract.
    pub water: WaterContract,
    /// Every molecule in the topology, including receptor, ions and waters.
    pub molecules: Vec<MoleculeRecord>,
    /// Immutable fragment chemical identities.
    pub species: Vec<FragmentSpecies>,
    /// Stable spatial/molecular slots and their current alchemical state.
    pub slots: Vec<FragmentSlot>,
    /// Hamiltonian parameters shared by CPU and GPU evaluation.
    pub softcore: SoftcoreHamiltonian,
    /// Chemical-potential and standard-state calibration.
    pub chemical_potentials: Vec<ChemicalPotentialCalibration>,
    /// Detailed-balance switching protocol.
    pub ncmc: NcmcProtocol,
}

/// Evidence-bearing capability projection for the actually selected runtime.
/// A source file or allocated buffer is not by itself an executable capability.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LambdaRuntimeCapabilities {
    /// Selected engine binary identity from the live launch binding.
    pub selected_engine_sha256: Option<String>,
    /// Exact fused PTX identity whose loaded symbols execute this path.
    pub selected_fused_ptx_sha256: Option<String>,
    /// Prepared topology identity executed by this runtime instance.
    pub topology_sha256: Option<String>,
    /// Native soft-core force branch is present in the selected force kernel.
    pub native_softcore_force: bool,
    /// Native atom-owned dU/dlambda accumulation is graph-bound.
    pub native_dudl: bool,
    /// Occupied state-A/state-B parameter interpolation is native and complete.
    pub dual_endpoint_state_transform: bool,
    /// Identity/tautomer/protonation endpoint planes execute on the selected
    /// CUDA Hamiltonian rather than the legacy scalar parameter path.
    pub dual_endpoint_identity_transition_cuda: bool,
    /// Hard upper bound enforced by this selected engine build.
    pub max_simultaneous_identity_transition_slots: u32,
    /// Fixed-dimensional union/dummy-proton state transformations execute.
    pub constant_topology_union_state_transform: bool,
    /// Endpoint bonds, angles and torsions contribute analytic force/dUdl.
    pub endpoint_bonded_force_dudl: bool,
    /// Endpoint exact exceptions contribute analytic force/dUdl.
    pub endpoint_exception_force_dudl: bool,
    /// The selected force path executes the declared soft-core-LJ plus
    /// charge-linear Ewald/PME Hamiltonian without legacy Coulomb mixing.
    pub ewald_charge_linear_hamiltonian: bool,
    /// Periodic minimum-image displacement is used throughout the force field.
    pub periodic_minimum_image: bool,
    /// Position wrapping executes after constraints.
    pub periodic_box_wrapping: bool,
    /// Prepared periodic-cell contract hash bound by the selected runtime.
    pub periodic_cell_contract_sha256: Option<String>,
    /// PME direct, reciprocal, self and exclusion corrections execute inline.
    pub pme_complete: bool,
    /// PME assignment/interpolation executes OpenMM-compatible order five.
    pub pme_openmm_order5: bool,
    /// Independent same-parameter OpenMM U/F comparison receipt for the
    /// selected implementation and discretization.
    pub pme_openmm_reference_parity_receipt_sha256: Option<String>,
    /// Composed real+reciprocal+self+background+exception U/F/dUdl test passed.
    pub pme_total_hamiltonian_self_test: bool,
    /// Target-system composed-Hamiltonian validation receipt.  The OpenMM
    /// tolerance/grid heuristic alone cannot satisfy this gate.
    pub pme_target_system_validation_receipt_sha256: Option<String>,
    /// PME parameterization hash executed by this runtime instance.
    pub pme_parameterization_sha256: Option<String>,
    /// Complete rigid-water position constraints execute.
    pub rigid_water_positions: bool,
    /// Complete rigid-water velocity constraints execute.
    pub rigid_water_velocities: bool,
    /// Exact rigid-water model executed by SETTLE/RATTLE.
    pub rigid_water_model_id: Option<String>,
    /// Prepared water-model contract hash executed by SETTLE/RATTLE.
    pub rigid_water_parameterization_sha256: Option<String>,
    /// NCMC perturb/propagate/accept sequence executes against device state.
    pub ncmc_device_sequence: bool,
    /// Native accepted move receipts are schema v4; older receipts are rejected for
    /// production calibration.
    pub ncmc_receipt_schema_v6: bool,
    /// Exact raw proposal context is persisted inside the authenticated v4
    /// receipt, not reconstructed from a scalar log ratio.
    pub ncmc_raw_proposal_context_persisted: bool,
    /// Proposal selection, placement and orientation base measures are
    /// machine-validated by the v3 receipt.
    pub ncmc_proposal_measure_contract_validated: bool,
    /// Calibration rows are exported only from the verified journal loader.
    pub ncmc_loader_verified_calibration_join: bool,
    /// Hash of the immutable verified calibration-join artifact.
    pub ncmc_calibration_join_receipt_sha256: Option<String>,
    /// One device-owned NCMC transaction can switch the subject and declared
    /// counterion/co-ion partner as an atomic charge-neutral move.
    pub coupled_charge_neutral_ncmc: bool,
    /// Box/PME-bound finite-size correction terms are applied and recorded by
    /// device-owned NCMC adjudication for net-charge-changing moves.
    pub finite_size_corrected_charged_ncmc: bool,
    /// Typed PrismCode and QNANL events are emitted by that sequence.
    pub typed_event_dispatch: bool,
    /// Same-stream PME/SETTLE/NCMC composition receipt.
    pub same_stream_composition_receipt_sha256: Option<String>,
    /// Device U/F/dUdl, endpoint, rollback and graph-residency self-test receipt.
    pub runtime_self_test_receipt_sha256: Option<String>,
    /// Typed PrismCode/QNANL runtime dispatch receipt.
    pub typed_event_dispatch_receipt_sha256: Option<String>,
}

/// Named measurement receipts produced by the selected engine's live native-
/// physics self-test.  These are evidence identities, not source-presence
/// assertions: a capability may be promoted only when its corresponding
/// measurement receipt is present and hash-valid.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LambdaRuntimeSelfTestEvidence {
    pub device_uf_dudl_receipt_sha256: Option<String>,
    pub endpoint_transition_receipt_sha256: Option<String>,
    pub rollback_receipt_sha256: Option<String>,
    pub graph_residency_receipt_sha256: Option<String>,
    pub pme_order5_receipt_sha256: Option<String>,
    pub pme_openmm_reference_receipt_sha256: Option<String>,
    pub pme_target_system_receipt_sha256: Option<String>,
    pub ncmc_receipt_sha256: Option<String>,
    pub settle_receipt_sha256: Option<String>,
    pub typed_event_dispatch_receipt_sha256: Option<String>,
}

impl LambdaRuntimeSelfTestEvidence {
    fn validate(&self) -> Result<(), String> {
        for (label, digest) in [
            (
                "device U/F/dUdl measurement",
                self.device_uf_dudl_receipt_sha256.as_deref(),
            ),
            (
                "endpoint transition measurement",
                self.endpoint_transition_receipt_sha256.as_deref(),
            ),
            (
                "rollback measurement",
                self.rollback_receipt_sha256.as_deref(),
            ),
            (
                "graph residency measurement",
                self.graph_residency_receipt_sha256.as_deref(),
            ),
            (
                "PME order-five measurement",
                self.pme_order5_receipt_sha256.as_deref(),
            ),
            (
                "PME OpenMM reference measurement",
                self.pme_openmm_reference_receipt_sha256.as_deref(),
            ),
            (
                "PME target-system measurement",
                self.pme_target_system_receipt_sha256.as_deref(),
            ),
            ("NCMC measurement", self.ncmc_receipt_sha256.as_deref()),
            ("SETTLE measurement", self.settle_receipt_sha256.as_deref()),
            (
                "typed-event dispatch measurement",
                self.typed_event_dispatch_receipt_sha256.as_deref(),
            ),
        ] {
            if let Some(digest) = digest {
                validate_sha256(label, digest)?;
            }
        }
        Ok(())
    }
}

/// Durable output of a live selected-candidate native-physics self-test.
///
/// The measured projection deliberately carries no self-reference:
/// `runtime_self_test_receipt_sha256` must be absent here and is populated
/// only after this receipt validates and is consumed by the fused runtime.
/// This prevents a package-time capability document from manufacturing its
/// own evidence identity.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LambdaRuntimeSelfTestReceipt {
    pub schema_kind: String,
    pub schema_version: u32,
    pub selected_engine_path: PathBuf,
    pub selected_fused_ptx_path: PathBuf,
    pub topology_path: PathBuf,
    pub engine_sha256: String,
    pub fused_ptx_sha256: String,
    pub topology_sha256: String,
    pub measured_capabilities: LambdaRuntimeCapabilities,
    pub evidence: LambdaRuntimeSelfTestEvidence,
    pub receipt_sha256: String,
}

impl LambdaRuntimeSelfTestReceipt {
    fn recompute_sha256(&self) -> Result<String, String> {
        let mut projection = self.clone();
        projection.receipt_sha256.clear();
        let bytes = serde_json::to_vec(&projection).map_err(|error| error.to_string())?;
        let mut digest = Sha256::new();
        digest.update(b"PRISM_LAMBDA_RUNTIME_SELF_TEST_RECEIPT_V1\0");
        digest.update(bytes);
        Ok(format!("{:x}", digest.finalize()))
    }

    /// Seal a freshly measured receipt. Callers cannot supply a pre-existing
    /// digest: the content identity is derived only after every measured field
    /// and evidence reference has been populated.
    pub fn seal(mut self) -> Result<Self, String> {
        if !self.receipt_sha256.is_empty() {
            return Err("unsealed lambda runtime self-test must not carry a digest".into());
        }
        self.receipt_sha256 = self.recompute_sha256()?;
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_kind != "prismos_lambda_runtime_self_test_receipt"
            || self.schema_version != 1
            || self.receipt_sha256 != self.recompute_sha256()?
        {
            return Err("invalid lambda runtime self-test receipt".into());
        }
        for (label, digest) in [
            ("self-test engine", self.engine_sha256.as_str()),
            ("self-test fused PTX", self.fused_ptx_sha256.as_str()),
            ("self-test topology", self.topology_sha256.as_str()),
        ] {
            validate_sha256(label, digest)?;
        }
        self.evidence.validate()?;
        let measured = &self.measured_capabilities;
        if measured.selected_engine_sha256.as_deref() != Some(self.engine_sha256.as_str())
            || measured.selected_fused_ptx_sha256.as_deref() != Some(self.fused_ptx_sha256.as_str())
            || measured.topology_sha256.as_deref() != Some(self.topology_sha256.as_str())
            || measured.runtime_self_test_receipt_sha256.is_some()
        {
            return Err("self-test capability tuple is not its measured artifact tuple".into());
        }
        for (label, digest) in [
            (
                "periodic cell",
                measured.periodic_cell_contract_sha256.as_deref(),
            ),
            (
                "PME OpenMM parity",
                measured
                    .pme_openmm_reference_parity_receipt_sha256
                    .as_deref(),
            ),
            (
                "PME target system",
                measured
                    .pme_target_system_validation_receipt_sha256
                    .as_deref(),
            ),
            (
                "PME parameterization",
                measured.pme_parameterization_sha256.as_deref(),
            ),
            (
                "rigid-water parameterization",
                measured.rigid_water_parameterization_sha256.as_deref(),
            ),
            (
                "NCMC calibration join",
                measured.ncmc_calibration_join_receipt_sha256.as_deref(),
            ),
            (
                "same-stream composition",
                measured.same_stream_composition_receipt_sha256.as_deref(),
            ),
            (
                "typed-event dispatch",
                measured.typed_event_dispatch_receipt_sha256.as_deref(),
            ),
        ] {
            if let Some(digest) = digest {
                validate_sha256(label, digest)?;
            }
        }
        let requires = |condition: bool, evidence: Option<&str>, label: &str| {
            if condition && evidence.is_none() {
                Err(format!(
                    "measured {label} capability lacks its measurement receipt"
                ))
            } else {
                Ok(())
            }
        };
        requires(
            measured.native_softcore_force || measured.native_dudl,
            self.evidence.device_uf_dudl_receipt_sha256.as_deref(),
            "device U/F/dUdl",
        )?;
        requires(
            measured.dual_endpoint_state_transform
                || measured.dual_endpoint_identity_transition_cuda
                || measured.constant_topology_union_state_transform
                || measured.endpoint_bonded_force_dudl
                || measured.endpoint_exception_force_dudl,
            self.evidence.endpoint_transition_receipt_sha256.as_deref(),
            "endpoint transition",
        )?;
        requires(
            measured.ncmc_device_sequence,
            self.evidence.rollback_receipt_sha256.as_deref(),
            "rollback",
        )?;
        requires(
            measured.native_softcore_force || measured.pme_complete,
            self.evidence.graph_residency_receipt_sha256.as_deref(),
            "graph residency",
        )?;
        requires(
            measured.pme_openmm_order5,
            self.evidence.pme_order5_receipt_sha256.as_deref(),
            "PME order-five",
        )?;
        requires(
            measured
                .pme_openmm_reference_parity_receipt_sha256
                .is_some(),
            self.evidence.pme_openmm_reference_receipt_sha256.as_deref(),
            "PME OpenMM reference",
        )?;
        requires(
            measured.pme_total_hamiltonian_self_test
                || measured
                    .pme_target_system_validation_receipt_sha256
                    .is_some(),
            self.evidence.pme_target_system_receipt_sha256.as_deref(),
            "PME target-system Hamiltonian",
        )?;
        requires(
            measured.ncmc_device_sequence
                || measured.ncmc_receipt_schema_v6
                || measured.ncmc_raw_proposal_context_persisted
                || measured.ncmc_proposal_measure_contract_validated
                || measured.ncmc_loader_verified_calibration_join
                || measured.coupled_charge_neutral_ncmc
                || measured.finite_size_corrected_charged_ncmc,
            self.evidence.ncmc_receipt_sha256.as_deref(),
            "NCMC",
        )?;
        requires(
            measured.rigid_water_positions || measured.rigid_water_velocities,
            self.evidence.settle_receipt_sha256.as_deref(),
            "SETTLE",
        )?;
        requires(
            measured.typed_event_dispatch,
            self.evidence.typed_event_dispatch_receipt_sha256.as_deref(),
            "typed-event dispatch",
        )?;
        if measured.pme_openmm_reference_parity_receipt_sha256
            != self.evidence.pme_openmm_reference_receipt_sha256
            || measured.pme_target_system_validation_receipt_sha256
                != self.evidence.pme_target_system_receipt_sha256
            || measured.typed_event_dispatch_receipt_sha256
                != self.evidence.typed_event_dispatch_receipt_sha256
        {
            return Err("self-test capability evidence digest projection differs".into());
        }
        Ok(())
    }
}

impl LambdaRuntimeCapabilities {
    /// True only when the runtime can execute a production lambda experiment.
    pub fn production_ready(&self) -> bool {
        self.selected_engine_sha256.is_some()
            && self.selected_fused_ptx_sha256.is_some()
            && self.topology_sha256.is_some()
            && self.native_softcore_force
            && self.native_dudl
            && self.dual_endpoint_state_transform
            && self.dual_endpoint_identity_transition_cuda
            && self.max_simultaneous_identity_transition_slots >= 1
            && self.constant_topology_union_state_transform
            && self.endpoint_bonded_force_dudl
            && self.endpoint_exception_force_dudl
            && self.ewald_charge_linear_hamiltonian
            && self.periodic_minimum_image
            && self.periodic_box_wrapping
            && self.pme_complete
            && self.pme_openmm_order5
            && self.pme_openmm_reference_parity_receipt_sha256.is_some()
            && self.pme_total_hamiltonian_self_test
            && self.pme_target_system_validation_receipt_sha256.is_some()
            && self.pme_parameterization_sha256.is_some()
            && self.rigid_water_positions
            && self.rigid_water_velocities
            && self.ncmc_device_sequence
            && self.ncmc_receipt_schema_v6
            && self.ncmc_raw_proposal_context_persisted
            && self.ncmc_proposal_measure_contract_validated
            && self.ncmc_loader_verified_calibration_join
            && self.ncmc_calibration_join_receipt_sha256.is_some()
            && self.coupled_charge_neutral_ncmc
            && self.finite_size_corrected_charged_ncmc
            && self.typed_event_dispatch
            && self.same_stream_composition_receipt_sha256.is_some()
            && self.runtime_self_test_receipt_sha256.is_some()
            && self.typed_event_dispatch_receipt_sha256.is_some()
    }
}

/// Triclinic periodic box; vectors are rows in Angstrom.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PeriodicBoxContract {
    /// Three linearly independent box vectors.
    pub vectors_a: [[f64; 3]; 3],
    /// Must be true for production lambda-fragment execution.
    pub minimum_image_enabled: bool,
    /// Provenance hash of the box-producing source artifact.
    pub source_sha256: String,
}

impl PeriodicBoxContract {
    /// Signed cell volume in cubic Angstrom.
    pub fn signed_volume_a3(&self) -> f64 {
        let [a, b, c] = self.vectors_a;
        a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
    }

    /// Positive cell volume in cubic Angstrom.
    pub fn volume_a3(&self) -> f64 {
        self.signed_volume_a3().abs()
    }
}

/// PME parameters required to make charged-state changes thermodynamically
/// meaningful under periodic boundary conditions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PmeContract {
    /// Preparing implementation whose parameter semantics are reproduced.
    pub reference_implementation: String,
    /// Three-dimensional periodic electrostatic boundary convention.
    pub boundary_condition: String,
    /// Cardinal B-spline interpolation order.
    pub pme_interpolation_order: u32,
    /// Target relative force tolerance.  This selects parameters empirically;
    /// it is not an accuracy certificate.
    pub relative_force_tolerance: f64,
    /// Declared direct-space cutoff in Angstrom.
    pub real_space_cutoff_angstrom: f64,
    /// Ewald splitting parameter in inverse Angstrom.
    pub ewald_alpha_a_inv: f64,
    /// Human-readable, machine-checked alpha derivation.
    pub ewald_alpha_derivation: String,
    /// Unrounded OpenMM mesh lower bound in each dimension.
    pub pme_grid_dimensions_lower_bound: [usize; 3],
    /// Executed cuFFT-compatible grid dimensions.
    pub pme_grid_dimensions: [usize; 3],
    /// Exact mesh lower-bound formula identifier.
    pub pme_grid_formula: String,
    /// Grid rounding policy; no dimension may be rounded below its bound.
    pub pme_grid_rounding_policy: String,
    /// Runtime must bind the exact alpha/grid/order contract.
    pub runtime_exact_parameter_lock_required: bool,
    /// Production admission requires independent OpenMM U/F parity evidence.
    pub independent_openmm_force_energy_validation_required: bool,
    /// Hash of the PME parameterization/evidence artifact.
    pub pme_parameterization_sha256: String,
}

impl PmeContract {
    /// Recompute the preparation-layer 13-key parameterization hash.  Floating
    /// values are represented by exact IEEE-754 bit strings so Python/Rust
    /// formatting cannot alter identity.
    pub fn parameterization_digest_sha256(&self) -> Result<String, String> {
        let mut projection = std::collections::BTreeMap::new();
        projection.insert(
            "boundary_condition",
            serde_json::json!(self.boundary_condition),
        );
        projection.insert(
            "ewald_alpha_a_inv",
            serde_json::json!(format!("{:016x}", self.ewald_alpha_a_inv.to_bits())),
        );
        projection.insert(
            "ewald_alpha_derivation",
            serde_json::json!(self.ewald_alpha_derivation),
        );
        projection.insert(
            "independent_openmm_force_energy_validation_required",
            serde_json::json!(self.independent_openmm_force_energy_validation_required),
        );
        projection.insert(
            "pme_grid_dimensions",
            serde_json::json!(self.pme_grid_dimensions),
        );
        projection.insert(
            "pme_grid_dimensions_lower_bound",
            serde_json::json!(self.pme_grid_dimensions_lower_bound),
        );
        projection.insert("pme_grid_formula", serde_json::json!(self.pme_grid_formula));
        projection.insert(
            "pme_grid_rounding_policy",
            serde_json::json!(self.pme_grid_rounding_policy),
        );
        projection.insert(
            "pme_interpolation_order",
            serde_json::json!(self.pme_interpolation_order),
        );
        projection.insert(
            "real_space_cutoff_angstrom",
            serde_json::json!(format!(
                "{:016x}",
                self.real_space_cutoff_angstrom.to_bits()
            )),
        );
        projection.insert(
            "reference_implementation",
            serde_json::json!(self.reference_implementation),
        );
        projection.insert(
            "relative_force_tolerance",
            serde_json::json!(format!("{:016x}", self.relative_force_tolerance.to_bits())),
        );
        projection.insert(
            "runtime_exact_parameter_lock_required",
            serde_json::json!(self.runtime_exact_parameter_lock_required),
        );
        let encoded = serde_json::to_vec(&projection).map_err(|error| error.to_string())?;
        Ok(format!("{:x}", Sha256::digest(encoded)))
    }
}

/// Explicit rigid-water admission contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WaterContract {
    /// Model identity, e.g. `TIP3P` or `OPC3`.
    pub model: String,
    /// Analytical SETTLE position and velocity constraints are both required.
    pub settle_positions_enabled: bool,
    /// Analytical SETTLE/RATTLE velocity correction.
    pub settle_velocities_enabled: bool,
    /// O-H equilibrium distance in Angstrom.
    pub oh_distance_a: f64,
    /// H-O-H equilibrium angle in radians.
    pub hoh_angle_rad: f64,
    /// Hash of the force-field/water parameter source.
    pub parameterization_sha256: String,
}

/// Functional role of a molecule in the experiment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum MoleculeRole {
    /// Receptor or protein partner.
    Receptor,
    /// Explicit rigid water.
    Water,
    /// Counterion or salt ion.
    Ion,
    /// Alchemically controlled fragment molecule.
    LambdaFragment,
    /// Other explicitly parameterized non-fragment molecule.
    Other,
}

/// Stable molecule identity and complete atom membership.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MoleculeRecord {
    /// Stable molecule identifier.
    pub molecule_id: u64,
    /// Functional role.
    pub role: MoleculeRole,
    /// Atom indices, each owned by exactly one molecule.
    pub atom_indices: Vec<usize>,
    /// For water, exact `[oxygen, hydrogen1, hydrogen2]`; absent otherwise.
    pub water_triplet: Option<[usize; 3]>,
}

/// Immutable chemical species identity.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FragmentSpecies {
    /// Stable numeric species identifier.
    pub species_id: u32,
    /// Human-readable molecular name.
    pub name: String,
    /// Canonical isomeric SMILES.
    pub canonical_isomeric_smiles: String,
    /// InChIKey, including stereochemical/protonation identity.
    pub inchikey: String,
    /// Hash of the fully parameterized molecular template.
    pub parameter_template_sha256: String,
    /// Enumerated tautomer/protonation states with fixed atom mapping.
    pub chemical_states: Vec<FragmentChemicalState>,
    /// Explicit admitted constant-topology transitions between states.
    #[serde(default)]
    pub state_transitions: Vec<FragmentStateTransition>,
}

/// A tautomer/protonation state with explicit per-atom nonbonded parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FragmentChemicalState {
    /// Stable state identifier within a species.
    pub state_id: u32,
    /// State-specific canonical isomeric SMILES.
    pub state_canonical_isomeric_smiles: String,
    /// Explicit-hydrogen, atom-mapped SMILES used to construct the union.
    pub state_explicit_h_mapped_smiles: String,
    /// Fixed-H InChIKey for exact tautomer/protonation identity.
    pub state_inchikey_fixed_h: String,
    /// Canonical tautomer label.
    pub tautomer_id: String,
    /// Canonical protonation label.
    pub protonation_id: String,
    /// Formal molecular charge.
    pub formal_charge_e: i32,
    /// Explicit bound-proton count used for pH target-weight changes.
    pub proton_count: i32,
    /// Exact atom identity/mapping for this endpoint state.
    pub atom_mapping: Vec<FragmentStateAtomMapping>,
    /// Reference conformer in atom-mapping order, Angstrom.
    pub reference_conformer_positions_a: Vec<[f64; 3]>,
    /// Per-template-atom parameters in slot atom order.
    pub atom_parameters: Vec<FragmentAtomState>,
    /// Stable digest over state identity, charge and ordered atom parameters.
    pub parameter_digest_sha256: String,
    /// Relative intrinsic free energy at the calibration pH, kcal/mol.
    pub intrinsic_free_energy_kcal_mol: f64,
    /// True only when bonds, angles, proper/improper torsions and constraint
    /// topology are exactly invariant across the proposed state transition.
    pub intramolecular_parameters_invariant: bool,
    /// Hash of that complete invariant intramolecular parameter set.
    pub intramolecular_parameter_set_sha256: String,
    /// Complete endpoint bonded plane over union-template atom IDs.
    pub bonded_parameter_plane: FragmentBondedParameterPlane,
    /// Hash of the complete endpoint bonded plane.
    pub bonded_parameter_plane_sha256: String,
    /// Complete state-specific exact nonbonded exceptions within the fragment.
    pub exact_exception_parameters: Vec<FragmentExactExceptionState>,
}

/// State-specific exact 1-4/exception parameter keyed by template atom ids.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FragmentExactExceptionState {
    /// First template atom id.
    pub template_atom_i: u32,
    /// Second template atom id.
    pub template_atom_j: u32,
    /// Exact charge product in e^2.
    pub charge_product_e2: f64,
    /// Exact sigma in Angstrom.
    pub sigma_a: f64,
    /// Exact epsilon in kcal/mol.
    pub epsilon_kcal_mol: f64,
}

/// Exact atom identity in one physical endpoint state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FragmentStateAtomMapping {
    /// Stable union-template atom identity.
    pub template_atom_id: u32,
    /// Atomic number; one for explicit protons.
    pub atomic_number: u8,
    /// Canonical element symbol.
    pub element: String,
    /// Positive mapped-SMILES atom map number.
    pub atom_map_number: u32,
}

/// Harmonic bond parameter in kcal/mol and Angstrom.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FragmentBondParameter {
    pub template_atom_i: u32,
    pub template_atom_j: u32,
    pub length_a: f64,
    pub force_constant_kcal_mol_a2: f64,
}

/// Harmonic angle parameter in kcal/mol and radians.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FragmentAngleParameter {
    pub template_atom_i: u32,
    pub template_atom_j: u32,
    pub template_atom_k: u32,
    pub angle_rad: f64,
    pub force_constant_kcal_mol_rad2: f64,
}

/// Periodic proper or improper torsion parameter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FragmentTorsionParameter {
    pub template_atom_i: u32,
    pub template_atom_j: u32,
    pub template_atom_k: u32,
    pub template_atom_l: u32,
    pub periodicity: u32,
    pub phase_rad: f64,
    pub barrier_kcal_mol: f64,
    pub improper: bool,
}

/// Lambda-invariant holonomic constraint. Endpoint-dependent constraints are
/// forbidden because they do not define a reversible fixed-dimensional path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FragmentConstraintParameter {
    pub template_atom_i: u32,
    pub template_atom_j: u32,
    pub distance_a: f64,
}

/// Complete endpoint intramolecular Hamiltonian plane.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FragmentBondedParameterPlane {
    /// Exact endpoint energy convention.  The only admitted value uses
    /// AMBER-form bonds/angles (`k*delta^2`) and periodic torsions
    /// (`barrier*(1+cos(n*phi-phase))`).
    pub formula: String,
    pub bonds: Vec<FragmentBondParameter>,
    pub angles: Vec<FragmentAngleParameter>,
    pub proper_and_improper_torsions: Vec<FragmentTorsionParameter>,
    pub constraints: Vec<FragmentConstraintParameter>,
}

/// Atom map between two physical endpoint states.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FragmentEndpointAtomMap {
    pub atom_a: u32,
    pub atom_b: u32,
    pub element: String,
    pub map_confidence: f64,
}

/// One fixed-dimensional union atom and its endpoint presence masks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FragmentConstantTopologyUnionAtom {
    pub union_template_atom_id: u32,
    pub state_a_template_atom_id: Option<u32>,
    pub state_b_template_atom_id: Option<u32>,
    pub atomic_number: u8,
    pub element: String,
    pub present_in_state_a: bool,
    pub present_in_state_b: bool,
}

/// Explicit endpoint bond-graph change.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FragmentBondGraphChange {
    pub template_atom_i: u32,
    pub template_atom_j: u32,
    pub bond_order_state_a: Option<u8>,
    pub bond_order_state_b: Option<u8>,
}

/// Admitted constant-topology union/dummy-proton transition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FragmentStateTransition {
    pub transition_id: String,
    pub state_a_id: u32,
    pub state_b_id: u32,
    pub atom_map_a_to_b: Vec<FragmentEndpointAtomMap>,
    pub unmatched_a: Vec<u32>,
    pub unmatched_b: Vec<u32>,
    pub bond_graph_change_set: Vec<FragmentBondGraphChange>,
    pub constant_topology_union_atoms: Vec<FragmentConstantTopologyUnionAtom>,
    pub dummy_proton_required: bool,
    pub bonded_plane_invariant: bool,
    /// Must be true; endpoint-dependent constraints are not admitted.
    pub constraints_invariant: bool,
    /// Intrinsic reference free-energy offset B-A in kcal/mol.
    pub reference_free_energy_b_minus_a_kcal_mol: f64,
    /// Difference in bound proton count B-A for the pH target term.
    pub delta_bound_protons_b_minus_a: i32,
    /// Periodic electrostatics treatment for the exact A-to-B net-charge
    /// change.  A uniform PME background is deliberately not an admitted
    /// treatment because it does not remove finite-size free-energy bias.
    pub periodic_charge_treatment: PeriodicChargeTreatment,
    /// Exact capability string required of the selected runtime.
    pub required_runtime_capability: String,
    /// Hash over the complete transition with this field cleared.
    pub transition_sha256: String,
}

/// Chemical-state parameters for a single fragment atom.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FragmentAtomState {
    /// Stable template atom id.
    pub template_atom_id: u32,
    /// True for a physical endpoint atom, false for an inactive union dummy.
    pub endpoint_present: bool,
    /// Partial charge in elementary-charge units.
    pub charge_e: f64,
    /// Lennard-Jones sigma in Angstrom.
    pub sigma_a: f64,
    /// Lennard-Jones epsilon in kcal/mol.
    pub epsilon_kcal_mol: f64,
}

/// Stable fragment slot. Atoms are never reindexed when identity/state changes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FragmentSlot {
    /// Stable slot identity.
    pub slot_id: u32,
    /// Stable molecule identity registered in `molecules`.
    pub molecule_id: u64,
    /// Current species identity.
    pub species_id: u32,
    /// Current tautomer/protonation state identity.
    pub chemical_state_id: u32,
    /// Exact topology atom indices in template order.
    pub atom_indices: Vec<usize>,
    /// Alchemical coordinate. Only zero and one are observational endpoints.
    pub lambda: f64,
    /// Declared endpoint occupancy; must agree with lambda at endpoints.
    pub endpoint_occupancy: EndpointOccupancy,
    /// Physical meaning of the alchemical coordinate.
    pub path_kind: AlchemicalPathKind,
    /// Exact admitted state-transition identity for STATE_A_TO_STATE_B.
    pub state_transition_id: Option<String>,
    /// Exact lambda=0 endpoint identity.
    pub endpoint_a: NcmcEndpointIdentity,
    /// Exact lambda=1 endpoint identity.
    pub endpoint_b: NcmcEndpointIdentity,
}

/// Distinguishes decoupling from an occupied-to-occupied chemical transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AlchemicalPathKind {
    /// One endpoint is empty and the other is a coupled physical fragment.
    InsertDelete,
    /// Both endpoints are occupied but have distinct hash-bound parameters.
    StateAToStateB,
}

/// Physical endpoint interpretation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum EndpointOccupancy {
    /// Decoupled endpoint, lambda=0.
    Empty,
    /// Coupled physical endpoint, lambda=1.
    Occupied,
    /// Switching state; never eligible for a pocket observation.
    Switching,
}

/// Stable digest over both physical endpoint identities and the exact
/// parameter planes they name.  A force event carries this digest rather than
/// the mutable current-state label so an intermediate lambda observation
/// remains interpretable without assigning it a fictitious single molecule.
pub fn fragment_slot_endpoint_parameter_digest_sha256(
    system: &LambdaFragmentSystem,
    slot: &FragmentSlot,
) -> Result<String, String> {
    #[derive(Serialize)]
    struct EndpointProjection {
        occupancy: EndpointOccupancy,
        species_id: u32,
        chemical_state_id: u32,
        parameter_digest_sha256: Option<String>,
    }
    #[derive(Serialize)]
    struct SlotProjection<'a> {
        schema_kind: &'static str,
        schema_version: u32,
        slot_id: u32,
        path_kind: AlchemicalPathKind,
        state_transition_id: Option<&'a str>,
        endpoint_a: EndpointProjection,
        endpoint_b: EndpointProjection,
    }

    let state_digest = |endpoint: &NcmcEndpointIdentity| -> Result<Option<String>, String> {
        let state = system
            .species
            .iter()
            .find(|species| species.species_id == endpoint.species_id)
            .and_then(|species| {
                species
                    .chemical_states
                    .iter()
                    .find(|state| state.state_id == endpoint.chemical_state_id)
            });
        match (endpoint.occupancy, state) {
            (EndpointOccupancy::Occupied, Some(state)) => {
                validate_sha256(
                    "fragment endpoint parameter",
                    &state.parameter_digest_sha256,
                )?;
                Ok(Some(state.parameter_digest_sha256.clone()))
            }
            (EndpointOccupancy::Occupied, None) => Err(
                "occupied fragment endpoint lacks an exact chemical-state parameter plane".into(),
            ),
            (EndpointOccupancy::Empty, _) => Ok(None),
            (EndpointOccupancy::Switching, _) => {
                Err("fragment endpoint parameter digest cannot bind SWITCHING".into())
            }
        }
    };
    let projection = SlotProjection {
        schema_kind: "prismos_lambda_slot_endpoint_parameter_projection",
        schema_version: 1,
        slot_id: slot.slot_id,
        path_kind: slot.path_kind,
        state_transition_id: slot.state_transition_id.as_deref(),
        endpoint_a: EndpointProjection {
            occupancy: slot.endpoint_a.occupancy,
            species_id: slot.endpoint_a.species_id,
            chemical_state_id: slot.endpoint_a.chemical_state_id,
            parameter_digest_sha256: state_digest(&slot.endpoint_a)?,
        },
        endpoint_b: EndpointProjection {
            occupancy: slot.endpoint_b.occupancy,
            species_id: slot.endpoint_b.species_id,
            chemical_state_id: slot.endpoint_b.chemical_state_id,
            parameter_digest_sha256: state_digest(&slot.endpoint_b)?,
        },
    };
    let encoded = serde_json::to_vec(&projection).map_err(|error| error.to_string())?;
    Ok(format!("{:x}", Sha256::digest(encoded)))
}

/// Recompute the package/runtime binding between fixed prepared A/B endpoint
/// planes and the complete directed scheduler kernel. The projection includes
/// coupled charge partners because a subject edge is not reversible unless the
/// partner edge is bound to its own fixed slot endpoints as well.
pub fn lambda_endpoint_support_binding_sha256(
    system: &LambdaFragmentSystem,
    scheduler: &NcmcProposalSchedulerContract,
) -> Result<String, String> {
    #[derive(Serialize)]
    struct PreparedEndpointPair {
        slot_id: u32,
        endpoint_a: NcmcEndpointIdentity,
        endpoint_b: NcmcEndpointIdentity,
        path_kind: AlchemicalPathKind,
        endpoint_parameter_sha256: String,
        atom_index_set_sha256: String,
    }
    #[derive(Serialize)]
    struct DirectedSupport {
        support_id: String,
        reverse_support_id: String,
        slot_id: u32,
        move_kind: NcmcMoveKind,
        initial: NcmcEndpointIdentity,
        proposed: NcmcEndpointIdentity,
        coupled_charge_partner: Option<NcmcCoupledChargePartner>,
    }
    #[derive(Serialize)]
    struct Projection {
        schema_kind: &'static str,
        schema_version: u32,
        lambda_fragment_system_sha256: String,
        scheduler_contract_sha256: String,
        prepared_slot_count: usize,
        directed_support_count: usize,
        prepared_endpoint_pairs: Vec<PreparedEndpointPair>,
        directed_supports: Vec<DirectedSupport>,
    }

    scheduler.validate()?;
    let mut slots = system.slots.iter().collect::<Vec<_>>();
    slots.sort_by_key(|slot| slot.slot_id);
    let slot_ids = slots.iter().map(|slot| slot.slot_id).collect::<Vec<_>>();
    if slot_ids != scheduler.admitted_slot_ids {
        return Err("scheduler admitted slots differ from prepared endpoint slots".into());
    }
    let mut prepared = Vec::with_capacity(slots.len());
    for slot in &slots {
        let mut atom_digest = Sha256::new();
        atom_digest.update(b"PRISM_LAMBDA_SLOT_ATOM_INDEX_SET_V1\0");
        let mut atom_indices = slot.atom_indices.clone();
        atom_indices.sort_unstable();
        if atom_indices != slot.atom_indices
            || atom_indices.windows(2).any(|pair| pair[0] == pair[1])
        {
            return Err("prepared endpoint slot atom indices are not sorted and unique".into());
        }
        for atom in &atom_indices {
            atom_digest.update((*atom as u64).to_le_bytes());
        }
        prepared.push(PreparedEndpointPair {
            slot_id: slot.slot_id,
            endpoint_a: slot.endpoint_a.clone(),
            endpoint_b: slot.endpoint_b.clone(),
            path_kind: slot.path_kind,
            endpoint_parameter_sha256: fragment_slot_endpoint_parameter_digest_sha256(
                system, slot,
            )?,
            atom_index_set_sha256: format!("{:x}", atom_digest.finalize()),
        });
    }

    let slot_by_id = slots
        .iter()
        .map(|slot| (slot.slot_id, *slot))
        .collect::<BTreeMap<_, _>>();
    let mut directed = Vec::with_capacity(scheduler.proposal_supports.len());
    let mut edge_counts = BTreeMap::<u32, usize>::new();
    for support in &scheduler.proposal_supports {
        let slot = slot_by_id
            .get(&support.slot_id)
            .ok_or_else(|| "scheduler support names an absent prepared slot".to_string())?;
        let fixed_direction = (support.initial == slot.endpoint_a
            && support.proposed == slot.endpoint_b)
            || (support.initial == slot.endpoint_b && support.proposed == slot.endpoint_a);
        if !fixed_direction {
            return Err("scheduler support is not one fixed prepared A/B edge".into());
        }
        if let Some(partner) = &support.coupled_charge_partner {
            let partner_slot = slot_by_id.get(&partner.slot_id).ok_or_else(|| {
                "coupled scheduler support names an absent prepared partner slot".to_string()
            })?;
            let partner_fixed = (partner.initial == partner_slot.endpoint_a
                && partner.proposed == partner_slot.endpoint_b)
                || (partner.initial == partner_slot.endpoint_b
                    && partner.proposed == partner_slot.endpoint_a);
            if !partner_fixed {
                return Err("coupled scheduler partner is not one fixed prepared A/B edge".into());
            }
        }
        *edge_counts.entry(support.slot_id).or_default() += 1;
        directed.push(DirectedSupport {
            support_id: support.support_id.clone(),
            reverse_support_id: support.reverse_support_id.clone(),
            slot_id: support.slot_id,
            move_kind: support.move_kind,
            initial: support.initial.clone(),
            proposed: support.proposed.clone(),
            coupled_charge_partner: support.coupled_charge_partner.clone(),
        });
    }
    if slots
        .iter()
        .any(|slot| edge_counts.get(&slot.slot_id).copied() != Some(2))
    {
        return Err("each prepared endpoint slot must have exactly two directed supports".into());
    }
    directed.sort_by(|left, right| {
        (left.slot_id, left.support_id.as_str()).cmp(&(right.slot_id, right.support_id.as_str()))
    });
    let projection = Projection {
        schema_kind: "prismos_lambda_endpoint_support_binding",
        schema_version: 2,
        lambda_fragment_system_sha256: format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(system).map_err(|error| error.to_string())?)
        ),
        scheduler_contract_sha256: scheduler.contract_sha256.clone(),
        prepared_slot_count: prepared.len(),
        directed_support_count: directed.len(),
        prepared_endpoint_pairs: prepared,
        directed_supports: directed,
    };
    let mut digest = Sha256::new();
    digest.update(b"PRISM_LAMBDA_ENDPOINT_SUPPORT_BINDING_V2\0");
    digest.update(serde_json::to_vec(&projection).map_err(|error| error.to_string())?);
    Ok(format!("{:x}", digest.finalize()))
}

/// One-source differentiable pair Hamiltonian parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SoftcoreHamiltonian {
    /// Schema/formula name; production PME requires
    /// `PRISM_LAMBDA_SOFTCORE_LJ_EWALD_CHARGE_V2`. The legacy V1 formula is a
    /// distinct non-PME direct-space reference and may never be mixed with PME.
    pub formula: String,
    /// Exact fail-closed convention for net-cell-charge-changing paths.
    pub periodic_charge_convention: String,
    /// LJ soft-core alpha in units of sigma^6.
    pub alpha_lj: f64,
    /// Electrostatic soft-core alpha in square Angstrom.
    pub alpha_coulomb_a2: f64,
    /// Power on `(1-lambda)` in both soft-core denominators.
    pub softcore_power: f64,
    /// Power on lambda in interaction scaling.
    pub coupling_power: f64,
    /// Pair cutoff in Angstrom.
    pub cutoff_a: f64,
}

impl Default for SoftcoreHamiltonian {
    fn default() -> Self {
        Self {
            formula: "PRISM_LAMBDA_SOFTCORE_LJ_COULOMB_V1".to_string(),
            periodic_charge_convention:
                "NET_CHARGE_CHANGE_REQUIRES_COUPLED_NEUTRALIZATION_OR_BOX_PME_FINITE_SIZE_CORRECTION_V1"
                    .to_string(),
            alpha_lj: 0.5,
            alpha_coulomb_a2: 1.0,
            softcore_power: 2.0,
            coupling_power: 1.0,
            cutoff_a: 10.0,
        }
    }
}

impl SoftcoreHamiltonian {
    /// Stable identity of the machine-readable Hamiltonian.
    pub fn digest_sha256(&self) -> Result<String, String> {
        validate_hamiltonian(self)?;
        let encoded = serde_json::to_vec(self).map_err(|error| error.to_string())?;
        Ok(format!("{:x}", Sha256::digest(encoded)))
    }
}

/// Standard-state chemical potential for a species/state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChemicalPotentialCalibration {
    /// Species identity.
    pub species_id: u32,
    /// Chemical-state identity.
    pub chemical_state_id: u32,
    /// Calibration temperature.
    pub temperature_k: f64,
    /// Calibration pH.
    pub ph: f64,
    /// Standard concentration in mol/L.
    pub standard_concentration_molar: f64,
    /// Excess chemical potential in kcal/mol.
    pub excess_chemical_potential_kcal_mol: f64,
    /// Complete chemical potential used by the target distribution.
    pub total_chemical_potential_kcal_mol: f64,
    /// Machine-readable convention for the state-reference contribution.
    pub state_reference_convention: StateReferenceConvention,
    /// Exact intrinsic state free energy included once in `total_chemical_potential_kcal_mol`.
    pub included_intrinsic_state_free_energy_kcal_mol: f64,
    /// Periodic electrostatics treatment for insertion of this state from an
    /// empty slot.  Its correction is included exactly once in total mu.
    pub periodic_charge_treatment: PeriodicChargeTreatment,
    /// Exact finite-size correction included in total mu, kcal/mol.  This is
    /// zero for charge-neutral coupled paths and net-charge-invariant states.
    pub included_finite_size_correction_kcal_mol: f64,
    /// Method identifier (e.g. BAR/MBAR hydration calibration).
    pub method: String,
    /// Hash of calibration inputs and estimates.
    pub evidence_sha256: String,
    /// One-sigma uncertainty in kcal/mol.
    pub uncertainty_kcal_mol: f64,
}

/// Prevents an intrinsic tautomer/protonation reference from being omitted or
/// counted in both the switching Hamiltonian and target-state weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum StateReferenceConvention {
    /// The calibrated total chemical potential contains the intrinsic state
    /// reference; the mechanical NCMC Hamiltonian must not add it again.
    TotalMuIncludesIntrinsicStateFreeEnergy,
}

/// Physical treatment of a move that may change the periodic cell's net
/// charge.  No `UniformBackground` variant exists by design: PME's neutralizing
/// plasma makes the reciprocal sum finite but is not a finite-size free-energy
/// correction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "SCREAMING_SNAKE_CASE", deny_unknown_fields)]
pub enum PeriodicChargeTreatment {
    /// The move changes net cell charge by exactly zero.
    NetChargeInvariant { subject_charge_change_e: i32 },
    /// A simultaneously switched ion/co-ion path makes the combined move
    /// charge neutral at every physical endpoint.
    CoupledChargeNeutralPath {
        subject_charge_change_e: i32,
        partner_charge_change_e: i32,
        combined_charge_change_e: i32,
        coupled_path_id: String,
        partner_slot_id: u32,
        coupling_schedule_sha256: String,
        evidence_sha256: String,
    },
    /// An explicit analytical/numerical correction bound to the exact
    /// production box and PME convention.
    ExplicitFiniteSizeCorrection {
        subject_charge_change_e: i32,
        correction_model: String,
        correction_free_energy_kcal_mol: f64,
        correction_uncertainty_kcal_mol: f64,
        production_box_source_sha256: String,
        pme_parameterization_sha256: String,
        evidence_sha256: String,
    },
}

impl PeriodicChargeTreatment {
    fn reversed(&self) -> Self {
        match self {
            Self::NetChargeInvariant {
                subject_charge_change_e,
            } => Self::NetChargeInvariant {
                subject_charge_change_e: -*subject_charge_change_e,
            },
            Self::CoupledChargeNeutralPath {
                subject_charge_change_e,
                partner_charge_change_e,
                combined_charge_change_e,
                coupled_path_id,
                partner_slot_id,
                coupling_schedule_sha256,
                evidence_sha256,
            } => Self::CoupledChargeNeutralPath {
                subject_charge_change_e: -*subject_charge_change_e,
                partner_charge_change_e: -*partner_charge_change_e,
                combined_charge_change_e: -*combined_charge_change_e,
                coupled_path_id: coupled_path_id.clone(),
                partner_slot_id: *partner_slot_id,
                coupling_schedule_sha256: coupling_schedule_sha256.clone(),
                evidence_sha256: evidence_sha256.clone(),
            },
            Self::ExplicitFiniteSizeCorrection {
                subject_charge_change_e,
                correction_model,
                correction_free_energy_kcal_mol,
                correction_uncertainty_kcal_mol,
                production_box_source_sha256,
                pme_parameterization_sha256,
                evidence_sha256,
            } => Self::ExplicitFiniteSizeCorrection {
                subject_charge_change_e: -*subject_charge_change_e,
                correction_model: correction_model.clone(),
                correction_free_energy_kcal_mol: -*correction_free_energy_kcal_mol,
                correction_uncertainty_kcal_mol: *correction_uncertainty_kcal_mol,
                production_box_source_sha256: production_box_source_sha256.clone(),
                pme_parameterization_sha256: pme_parameterization_sha256.clone(),
                evidence_sha256: evidence_sha256.clone(),
            },
        }
    }

    fn validate_for_move(
        &self,
        expected_subject_charge_change_e: i32,
        expected_correction_kcal_mol: f64,
        periodic_box: &PeriodicBoxContract,
        pme: &PmeContract,
    ) -> Result<(), String> {
        if !expected_correction_kcal_mol.is_finite() {
            return Err("nonfinite periodic finite-size correction".into());
        }
        match self {
            Self::NetChargeInvariant {
                subject_charge_change_e,
            } => {
                if expected_subject_charge_change_e != 0
                    || *subject_charge_change_e != 0
                    || expected_correction_kcal_mol.abs() > 1.0e-12
                {
                    return Err("net-charge-changing move lacks physical charge treatment".into());
                }
            }
            Self::CoupledChargeNeutralPath {
                subject_charge_change_e,
                partner_charge_change_e,
                combined_charge_change_e,
                coupled_path_id,
                coupling_schedule_sha256,
                evidence_sha256,
                ..
            } => {
                if expected_subject_charge_change_e == 0
                    || *subject_charge_change_e != expected_subject_charge_change_e
                    || *partner_charge_change_e != -*subject_charge_change_e
                    || *combined_charge_change_e != 0
                    || coupled_path_id.trim().is_empty()
                    || expected_correction_kcal_mol.abs() > 1.0e-12
                {
                    return Err(
                        "coupled charge-neutral path does not conserve charge exactly".into(),
                    );
                }
                validate_sha256("charge coupling schedule", coupling_schedule_sha256)?;
                validate_sha256("charge-neutral path evidence", evidence_sha256)?;
            }
            Self::ExplicitFiniteSizeCorrection {
                subject_charge_change_e,
                correction_model,
                correction_free_energy_kcal_mol,
                correction_uncertainty_kcal_mol,
                production_box_source_sha256,
                pme_parameterization_sha256,
                evidence_sha256,
            } => {
                let admitted_model = matches!(
                    correction_model.as_str(),
                    "ROCKLIN_NET_CHARGE_SOLVATION_V1"
                        | "HUMMER_PRATT_GARCIA_V1"
                        | "NUMERIC_BOX_PME_TRANSFER_V1"
                );
                if expected_subject_charge_change_e == 0
                    || *subject_charge_change_e != expected_subject_charge_change_e
                    || !admitted_model
                    || !correction_free_energy_kcal_mol.is_finite()
                    || !correction_uncertainty_kcal_mol.is_finite()
                    || *correction_uncertainty_kcal_mol < 0.0
                    || (*correction_free_energy_kcal_mol - expected_correction_kcal_mol).abs()
                        > 1.0e-12
                    || production_box_source_sha256 != &periodic_box.source_sha256
                    || pme_parameterization_sha256 != &pme.pme_parameterization_sha256
                {
                    return Err(
                        "finite-size correction is not bound to move, production box and PME"
                            .into(),
                    );
                }
                validate_sha256("finite-size correction evidence", evidence_sha256)?;
            }
        }
        Ok(())
    }
}

impl ChemicalPotentialCalibration {
    /// Validate the complete calibrated thermodynamic input independently of
    /// a particular experiment's species registry.
    pub fn validate(&self) -> Result<(), String> {
        if !self.temperature_k.is_finite()
            || self.temperature_k <= 0.0
            || !self.ph.is_finite()
            || !self.standard_concentration_molar.is_finite()
            || self.standard_concentration_molar <= 0.0
            || !self.excess_chemical_potential_kcal_mol.is_finite()
            || !self.total_chemical_potential_kcal_mol.is_finite()
            || !self
                .included_intrinsic_state_free_energy_kcal_mol
                .is_finite()
            || !self.included_finite_size_correction_kcal_mol.is_finite()
            || self.method.trim().is_empty()
            || !self.uncertainty_kcal_mol.is_finite()
            || self.uncertainty_kcal_mol < 0.0
        {
            return Err("invalid chemical-potential calibration".into());
        }
        let expected_total = self.excess_chemical_potential_kcal_mol
            + self.included_intrinsic_state_free_energy_kcal_mol
            + self.included_finite_size_correction_kcal_mol;
        let scale = expected_total
            .abs()
            .max(self.total_chemical_potential_kcal_mol.abs())
            .max(1.0);
        if (self.total_chemical_potential_kcal_mol - expected_total).abs() > 1.0e-12 * scale {
            return Err(
                "chemical-potential total does not equal excess + intrinsic-state + finite-size correction"
                    .into(),
            );
        }
        match &self.periodic_charge_treatment {
            PeriodicChargeTreatment::NetChargeInvariant {
                subject_charge_change_e,
            } if *subject_charge_change_e == 0
                && self.included_finite_size_correction_kcal_mol.to_bits() == 0.0f64.to_bits() => {}
            PeriodicChargeTreatment::CoupledChargeNeutralPath {
                combined_charge_change_e,
                ..
            } if *combined_charge_change_e == 0
                && self.included_finite_size_correction_kcal_mol.to_bits() == 0.0f64.to_bits() => {}
            PeriodicChargeTreatment::ExplicitFiniteSizeCorrection {
                correction_free_energy_kcal_mol,
                ..
            } if (*correction_free_energy_kcal_mol
                - self.included_finite_size_correction_kcal_mol)
                .abs()
                <= 1.0e-12
                    * correction_free_energy_kcal_mol
                        .abs()
                        .max(self.included_finite_size_correction_kcal_mol.abs())
                        .max(1.0) => {}
            _ => {
                return Err(
                    "chemical-potential finite-size term disagrees with periodic charge treatment"
                        .into(),
                )
            }
        }
        validate_sha256("chemical-potential evidence", &self.evidence_sha256)
    }

    /// Stable identity over the exact calibration carried into a fiber.
    pub fn digest_sha256(&self) -> Result<String, String> {
        self.validate()?;
        let encoded = serde_json::to_vec(self).map_err(|error| error.to_string())?;
        Ok(format!("{:x}", Sha256::digest(encoded)))
    }
}

/// One exact operation in an ordered NCMC protocol.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "operation",
    rename_all = "SCREAMING_SNAKE_CASE",
    deny_unknown_fields
)]
pub enum NcmcProtocolOperation {
    /// Instantaneous Hamiltonian perturbation at fixed coordinates.
    Perturb { lambda_from: f64, lambda_to: f64 },
    /// Dynamical propagation at one fixed Hamiltonian endpoint/intermediate.
    Propagate { lambda: f64, integration_steps: u32 },
    /// Phase-space involution used by the explicitly declared reverse path.
    MomentumReverse,
}

impl NcmcProtocolOperation {
    fn reversed(&self) -> Self {
        match self {
            Self::Perturb {
                lambda_from,
                lambda_to,
            } => Self::Perturb {
                lambda_from: *lambda_to,
                lambda_to: *lambda_from,
            },
            Self::Propagate {
                lambda,
                integration_steps,
            } => Self::Propagate {
                lambda: *lambda,
                integration_steps: *integration_steps,
            },
            Self::MomentumReverse => Self::MomentumReverse,
        }
    }
}

/// How conditional integrator path probabilities enter Metropolis-Hastings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum NcmcIntegratorPathProbabilityConvention {
    /// Receipt records forward and reverse conditional log path probability.
    ExactConditionalPathProbability,
    /// Receipt records shadow work; the mapping is phase-volume preserving.
    ShadowWorkVolumePreserving,
    /// Receipt records shadow work and an explicit phase-space Jacobian.
    ShadowWorkWithJacobian,
}

/// Momentum handling required to pair the forward and reverse paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum NcmcMomentumReversalPolicy {
    /// Reverse the proposal path momenta and reverse momenta after rejection.
    ReversePathAndRejection,
}

/// Which hash-bound operation list was actually executed for a directed
/// endpoint proposal. A fixed-topology A-to-B support executes `FORWARD`;
/// its exact B-to-A reverse executes `REVERSE` and inverts the protocol
/// selection-probability contribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum NcmcProtocolDirection {
    Forward,
    Reverse,
}

/// NCMC schedule and detailed-balance requirements.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NcmcProtocol {
    /// Exact ordered forward perturb/propagate/involution operations.
    pub forward_operations: Vec<NcmcProtocolOperation>,
    /// Explicit reverse schedule. It must be the exact operation-wise reverse,
    /// not merely a mirrored lambda list.
    pub reverse_operations: Vec<NcmcProtocolOperation>,
    /// Probability with which this forward protocol is selected.
    pub forward_protocol_probability: f64,
    /// Probability with which its reverse protocol is selected.
    pub reverse_protocol_probability: f64,
    /// Integrator splitting string recorded in receipts.
    pub integrator_splitting: String,
    /// Mutually exclusive integrator path-probability accounting convention.
    pub integrator_path_probability_convention: NcmcIntegratorPathProbabilityConvention,
    /// Exact momentum involution/rejection policy.
    pub momentum_reversal_policy: NcmcMomentumReversalPolicy,
    /// The initial physical endpoint equilibrium density is evaluated.
    pub initial_endpoint_equilibrium_density_evaluated: bool,
    /// The proposed physical endpoint equilibrium density is evaluated.
    pub proposed_endpoint_equilibrium_density_evaluated: bool,
    /// Counter-based RNG algorithm identifier.
    pub rng_algorithm: String,
    /// Stable hash of every field above, including ordered operations.
    pub schedule_sha256: String,
}

/// Separates unbiased bulk work collection from target-lane grand-canonical
/// sampling.  The chemical potential being estimated is never fed back into
/// the calibration work Hamiltonian or acceptance target.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "SCREAMING_SNAKE_CASE", deny_unknown_fields)]
pub enum NcmcExecutionMode {
    /// Nonequilibrium bulk switching used to estimate a chemical potential.
    CalibrationWorkCollection {
        /// Immutable protocol/configuration authority for this work campaign.
        protocol_authority_sha256: String,
    },
    /// Production target-lane NCMC using an existing calibrated chemical
    /// potential and its evidence lineage.
    CalibratedTargetLane,
}

/// Measure from which a persistent dummy fragment's decoupled coordinates
/// are drawn. A fixed dummy slot is not a uniform-volume insertion proposal
/// unless the explicit support and mixing evidence below prove that claim.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "measure",
    rename_all = "SCREAMING_SNAKE_CASE",
    deny_unknown_fields
)]
pub enum NcmcPlacementMeasureAuthority {
    PersistentDecoupledCoordinateEnsemble {
        coordinate_ensemble_sha256: String,
        mixing_evidence_sha256: String,
        placement_density_formula: String,
        orientation_density_formula: String,
    },
    UniformAccessibleVolume {
        accessible_volume_a3: f64,
        support_region_sha256: String,
        coordinate_mixing_evidence_sha256: String,
        orientation_density_formula: String,
    },
}

/// Hash the exact native scheduler fields that define the persistent
/// decoupled-coordinate proposal measure. This digest is intentionally
/// distinct from the source branch authority and from the larger placement
/// evidence object: it proves which scheduler projection the runtime used.
pub fn ncmc_placement_projection_sha256(
    measure: &NcmcPlacementMeasureAuthority,
) -> Result<String, String> {
    #[derive(Serialize)]
    struct PersistentProjection<'a> {
        measure: &'static str,
        coordinate_ensemble_sha256: &'a str,
        mixing_evidence_sha256: &'a str,
        placement_density_formula: &'a str,
        orientation_density_formula: &'a str,
    }

    let projection = match measure {
        NcmcPlacementMeasureAuthority::PersistentDecoupledCoordinateEnsemble {
            coordinate_ensemble_sha256,
            mixing_evidence_sha256,
            placement_density_formula,
            orientation_density_formula,
        } => PersistentProjection {
            measure: "PERSISTENT_DECOUPLED_COORDINATE_ENSEMBLE",
            coordinate_ensemble_sha256,
            mixing_evidence_sha256,
            placement_density_formula,
            orientation_density_formula,
        },
        NcmcPlacementMeasureAuthority::UniformAccessibleVolume { .. } => {
            return Err(
                "persistent-slot runtime cannot project a uniform-volume scheduler measure".into(),
            )
        }
    };
    let mut digest = Sha256::new();
    digest.update(b"PRISM_DECOUPLED_FRAGMENT_NCMC_PLACEMENT_PROJECTION_V1\0");
    digest.update(serde_json::to_vec(&projection).map_err(|error| error.to_string())?);
    Ok(format!("{:x}", digest.finalize()))
}

#[derive(Serialize)]
struct DecoupledFragmentPromotionReceiptProjection<'a> {
    precursor_placement_measure_sha256: &'a str,
    chemical_potential_evidence_sha256: &'a str,
    cross_cell_unlock_sha256: &'a str,
    finite_size_admission_sha256: &'a str,
    seed_topology_member_sha256: &'a str,
    panel_scientific_sha256: &'a str,
    target_protocol_authority_sha256: &'a str,
}

/// Deterministically reconstruct the immutable target-lane promotion receipt
/// from the exact calibration, cross-cell, seed-member, panel and scheduler
/// authorities. The promoted placement points to this receipt; the receipt
/// deliberately does not point back to the promoted placement, avoiding a
/// digest cycle.
pub fn decoupled_fragment_promotion_receipt_sha256(
    precursor_placement_measure_sha256: &str,
    chemical_potential_evidence_sha256: &str,
    cross_cell_unlock_sha256: &str,
    finite_size_admission_sha256: &str,
    seed_topology_member_sha256: &str,
    panel_scientific_sha256: &str,
    target_protocol_authority_sha256: &str,
) -> Result<String, String> {
    for (label, digest) in [
        ("precursor placement", precursor_placement_measure_sha256),
        (
            "chemical-potential evidence",
            chemical_potential_evidence_sha256,
        ),
        ("terminal cross-cell unlock", cross_cell_unlock_sha256),
        ("finite-size admission", finite_size_admission_sha256),
        ("seed topology/cell member", seed_topology_member_sha256),
        ("panel scientific authority", panel_scientific_sha256),
        (
            "target protocol authority",
            target_protocol_authority_sha256,
        ),
    ] {
        validate_sha256(label, digest)?;
    }
    let projection = DecoupledFragmentPromotionReceiptProjection {
        precursor_placement_measure_sha256,
        chemical_potential_evidence_sha256,
        cross_cell_unlock_sha256,
        finite_size_admission_sha256,
        seed_topology_member_sha256,
        panel_scientific_sha256,
        target_protocol_authority_sha256,
    };
    let mut digest = Sha256::new();
    digest.update(b"PRISM_DECOUPLED_FRAGMENT_PRODUCTION_PROMOTION_RECEIPT_V1\0");
    digest.update(serde_json::to_vec(&projection).map_err(|error| error.to_string())?);
    Ok(format!("{:x}", digest.finalize()))
}

/// Content identity of every admitted target-lane chemical-potential record.
pub fn lambda_chemical_potential_evidence_sha256(
    system: &LambdaFragmentSystem,
) -> Result<String, String> {
    let mut calibrations = system.chemical_potentials.clone();
    calibrations.sort_by_key(|value| (value.species_id, value.chemical_state_id));
    if calibrations.is_empty()
        || calibrations.windows(2).any(|pair| {
            (pair[0].species_id, pair[0].chemical_state_id)
                >= (pair[1].species_id, pair[1].chemical_state_id)
        })
    {
        return Err("target lane lacks a unique chemical-potential calibration population".into());
    }
    for calibration in &calibrations {
        calibration.validate()?;
    }
    let mut digest = Sha256::new();
    digest.update(b"PRISM_LAMBDA_CHEMICAL_POTENTIAL_EVIDENCE_SET_V1\0");
    digest.update(serde_json::to_vec(&calibrations).map_err(|error| error.to_string())?);
    Ok(format!("{:x}", digest.finalize()))
}

/// Independent digest for the finite-size and charge-treatment portion of the
/// admitted chemical-potential population.
pub fn lambda_finite_size_admission_sha256(
    system: &LambdaFragmentSystem,
) -> Result<String, String> {
    #[derive(Serialize)]
    struct Projection<'a> {
        species_id: u32,
        chemical_state_id: u32,
        periodic_charge_treatment: &'a PeriodicChargeTreatment,
        included_finite_size_correction_kcal_mol: f64,
        evidence_sha256: &'a str,
    }
    let mut calibrations = system.chemical_potentials.iter().collect::<Vec<_>>();
    calibrations.sort_by_key(|value| (value.species_id, value.chemical_state_id));
    if calibrations.is_empty() {
        return Err("target lane lacks finite-size admission inputs".into());
    }
    let mut projection = Vec::with_capacity(calibrations.len());
    for calibration in calibrations {
        calibration.validate()?;
        projection.push(Projection {
            species_id: calibration.species_id,
            chemical_state_id: calibration.chemical_state_id,
            periodic_charge_treatment: &calibration.periodic_charge_treatment,
            included_finite_size_correction_kcal_mol: calibration
                .included_finite_size_correction_kcal_mol,
            evidence_sha256: &calibration.evidence_sha256,
        });
    }
    let mut digest = Sha256::new();
    digest.update(b"PRISM_LAMBDA_FINITE_SIZE_ADMISSION_SET_V1\0");
    digest.update(serde_json::to_vec(&projection).map_err(|error| error.to_string())?);
    Ok(format!("{:x}", digest.finalize()))
}

/// Exact topology/frozen-cell member admitted into a target lane.
pub fn lambda_seed_topology_member_sha256(
    topology_sha256: &str,
    frozen_cell_authority_sha256: &str,
    periodic_cell_sha256: &str,
    replica_seed: u64,
) -> Result<String, String> {
    #[derive(Serialize)]
    struct Projection<'a> {
        topology_sha256: &'a str,
        frozen_cell_authority_sha256: &'a str,
        periodic_cell_sha256: &'a str,
        replica_seed: u64,
    }
    validate_sha256("seed topology", topology_sha256)?;
    validate_sha256("seed frozen-cell authority", frozen_cell_authority_sha256)?;
    validate_sha256("seed periodic cell", periodic_cell_sha256)?;
    let projection = Projection {
        topology_sha256,
        frozen_cell_authority_sha256,
        periodic_cell_sha256,
        replica_seed,
    };
    let mut digest = Sha256::new();
    digest.update(b"PRISM_LAMBDA_SEED_TOPOLOGY_CELL_MEMBER_V1\0");
    digest.update(serde_json::to_vec(&projection).map_err(|error| error.to_string())?);
    Ok(format!("{:x}", digest.finalize()))
}

impl NcmcPlacementMeasureAuthority {
    fn validate(&self) -> Result<(), String> {
        match self {
            Self::PersistentDecoupledCoordinateEnsemble {
                coordinate_ensemble_sha256,
                mixing_evidence_sha256,
                placement_density_formula,
                orientation_density_formula,
            } => {
                validate_sha256("decoupled coordinate ensemble", coordinate_ensemble_sha256)?;
                validate_sha256(
                    "decoupled coordinate mixing evidence",
                    mixing_evidence_sha256,
                )?;
                if placement_density_formula.trim().is_empty()
                    || orientation_density_formula.trim().is_empty()
                {
                    return Err("persistent decoupled placement formulas are empty".into());
                }
            }
            Self::UniformAccessibleVolume {
                accessible_volume_a3,
                support_region_sha256,
                coordinate_mixing_evidence_sha256,
                orientation_density_formula,
            } => {
                if !accessible_volume_a3.is_finite() || *accessible_volume_a3 <= 0.0 {
                    return Err("uniform insertion support volume is invalid".into());
                }
                validate_sha256("uniform insertion support", support_region_sha256)?;
                validate_sha256(
                    "uniform insertion coordinate mixing evidence",
                    coordinate_mixing_evidence_sha256,
                )?;
                if orientation_density_formula.trim().is_empty() {
                    return Err("uniform insertion orientation formula is empty".into());
                }
            }
        }
        Ok(())
    }
}

/// Continuous-measure and thermodynamic terms authenticated by one proposal
/// support. Occupancy counts and discrete probabilities are computed from the
/// complete current endpoint state, never copied from a fixed move template.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NcmcProposalBalanceTemplate {
    pub temperature_k: f64,
    pub ph: f64,
    pub accessible_volume_a3: f64,
    pub forward_placement_density_a3_inv: f64,
    pub reverse_placement_density_a3_inv: f64,
    pub forward_orientation_density: f64,
    pub reverse_orientation_density: f64,
    pub initial_bias_weight: f64,
    pub proposed_bias_weight: f64,
    pub bias_epoch_sha256: String,
}

impl NcmcProposalBalanceTemplate {
    fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("temperature_k", self.temperature_k),
            ("ph", self.ph),
            ("accessible_volume_a3", self.accessible_volume_a3),
            (
                "forward_placement_density_a3_inv",
                self.forward_placement_density_a3_inv,
            ),
            (
                "reverse_placement_density_a3_inv",
                self.reverse_placement_density_a3_inv,
            ),
            (
                "forward_orientation_density",
                self.forward_orientation_density,
            ),
            (
                "reverse_orientation_density",
                self.reverse_orientation_density,
            ),
            ("initial_bias_weight", self.initial_bias_weight),
            ("proposed_bias_weight", self.proposed_bias_weight),
        ] {
            if !value.is_finite() || (name != "ph" && value <= 0.0) {
                return Err(format!("invalid NCMC support balance term {name}"));
            }
        }
        validate_sha256("NCMC support bias epoch", &self.bias_epoch_sha256)
    }

    fn reversed_matches(&self, reverse: &Self) -> bool {
        self.temperature_k.to_bits() == reverse.temperature_k.to_bits()
            && self.ph.to_bits() == reverse.ph.to_bits()
            && self.accessible_volume_a3.to_bits() == reverse.accessible_volume_a3.to_bits()
            && self.forward_placement_density_a3_inv.to_bits()
                == reverse.reverse_placement_density_a3_inv.to_bits()
            && self.reverse_placement_density_a3_inv.to_bits()
                == reverse.forward_placement_density_a3_inv.to_bits()
            && self.forward_orientation_density.to_bits()
                == reverse.reverse_orientation_density.to_bits()
            && self.reverse_orientation_density.to_bits()
                == reverse.forward_orientation_density.to_bits()
            && self.initial_bias_weight.to_bits() == reverse.proposed_bias_weight.to_bits()
            && self.proposed_bias_weight.to_bits() == reverse.initial_bias_weight.to_bits()
            && self.bias_epoch_sha256 == reverse.bias_epoch_sha256
    }

    fn materialize(
        &self,
        occupied_count_before: u64,
        occupied_count_after: u64,
        forward_selection_probability: f64,
        reverse_selection_probability: f64,
    ) -> NcmcBalanceContext {
        NcmcBalanceContext {
            temperature_k: self.temperature_k,
            ph: self.ph,
            accessible_volume_a3: self.accessible_volume_a3,
            occupied_count_before,
            occupied_count_after,
            forward_selection_probability,
            reverse_selection_probability,
            forward_placement_density_a3_inv: self.forward_placement_density_a3_inv,
            reverse_placement_density_a3_inv: self.reverse_placement_density_a3_inv,
            forward_orientation_density: self.forward_orientation_density,
            reverse_orientation_density: self.reverse_orientation_density,
            initial_bias_weight: self.initial_bias_weight,
            proposed_bias_weight: self.proposed_bias_weight,
            bias_epoch_sha256: self.bias_epoch_sha256.clone(),
        }
    }
}

/// One directed edge in the current-state-conditioned proposal kernel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NcmcProposalSupport {
    pub support_id: String,
    pub reverse_support_id: String,
    pub slot_id: u32,
    pub move_kind: NcmcMoveKind,
    pub initial: NcmcEndpointIdentity,
    pub proposed: NcmcEndpointIdentity,
    pub base_weight: f64,
    pub balance_template: NcmcProposalBalanceTemplate,
    pub periodic_charge_treatment: PeriodicChargeTreatment,
    pub coupled_charge_partner: Option<NcmcCoupledChargePartner>,
}

impl NcmcProposalSupport {
    fn validate(&self) -> Result<(), String> {
        if self.support_id.trim().is_empty()
            || self.reverse_support_id.trim().is_empty()
            || !self.base_weight.is_finite()
            || self.base_weight <= 0.0
            || self.initial == self.proposed
            || self.initial.occupancy == EndpointOccupancy::Switching
            || self.proposed.occupancy == EndpointOccupancy::Switching
        {
            return Err("invalid NCMC proposal support".into());
        }
        let kind_valid = match self.move_kind {
            NcmcMoveKind::Insertion => {
                self.initial.occupancy == EndpointOccupancy::Empty
                    && self.proposed.occupancy == EndpointOccupancy::Occupied
            }
            NcmcMoveKind::Deletion => {
                self.initial.occupancy == EndpointOccupancy::Occupied
                    && self.proposed.occupancy == EndpointOccupancy::Empty
            }
            NcmcMoveKind::Identity => {
                self.initial.occupancy == EndpointOccupancy::Occupied
                    && self.proposed.occupancy == EndpointOccupancy::Occupied
                    && self.initial.species_id != self.proposed.species_id
            }
            NcmcMoveKind::Tautomer | NcmcMoveKind::Protonation => {
                self.initial.occupancy == EndpointOccupancy::Occupied
                    && self.proposed.occupancy == EndpointOccupancy::Occupied
                    && self.initial.species_id == self.proposed.species_id
                    && self.initial.chemical_state_id != self.proposed.chemical_state_id
            }
        };
        if !kind_valid {
            return Err("NCMC proposal support transition disagrees with move kind".into());
        }
        self.balance_template.validate()?;
        match (
            &self.periodic_charge_treatment,
            &self.coupled_charge_partner,
        ) {
            (
                PeriodicChargeTreatment::CoupledChargeNeutralPath {
                    partner_slot_id,
                    partner_charge_change_e,
                    ..
                },
                Some(partner),
            ) if partner.slot_id == *partner_slot_id
                && partner.slot_id != self.slot_id
                && partner.charge_change_e == *partner_charge_change_e =>
            {
                partner.validate()?
            }
            (PeriodicChargeTreatment::CoupledChargeNeutralPath { .. }, _) => {
                return Err("NCMC support omits its exact charge partner".into())
            }
            (_, Some(_)) => return Err("uncoupled NCMC support carries a charge partner".into()),
            (_, None) => {}
        }
        Ok(())
    }

    fn eligible(&self, endpoints: &BTreeMap<u32, NcmcEndpointIdentity>) -> bool {
        endpoints.get(&self.slot_id) == Some(&self.initial)
            && self
                .coupled_charge_partner
                .as_ref()
                .map(|partner| endpoints.get(&partner.slot_id) == Some(&partner.initial))
                .unwrap_or(true)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NcmcScheduledOpportunity {
    pub move_ordinal: u64,
    pub step: u64,
    pub allowed_support_ids: Vec<String>,
    #[serde(with = "u128_decimal_string")]
    pub rng_counter_base: u128,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NcmcRngCounterLayout {
    #[serde(with = "u128_decimal_string")]
    pub selection_counter_offset: u128,
    #[serde(with = "u128_decimal_string")]
    pub placement_counter_start_offset: u128,
    #[serde(with = "u128_decimal_string")]
    pub placement_counter_count: u128,
    #[serde(with = "u128_decimal_string")]
    pub orientation_counter_start_offset: u128,
    #[serde(with = "u128_decimal_string")]
    pub orientation_counter_count: u128,
    #[serde(with = "u128_decimal_string")]
    pub momentum_counter_start_offset: u128,
    #[serde(with = "u128_decimal_string")]
    pub momentum_counter_count: u128,
    #[serde(with = "u128_decimal_string")]
    pub thermostat_counter_start_offset: u128,
    #[serde(with = "u128_decimal_string")]
    pub thermostat_counter_count: u128,
    #[serde(with = "u128_decimal_string")]
    pub acceptance_counter_offset: u128,
    #[serde(with = "u128_decimal_string")]
    pub counters_per_opportunity: u128,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NcmcRngCounterRange {
    #[serde(with = "u128_decimal_string")]
    pub start: u128,
    #[serde(with = "u128_decimal_string")]
    pub count: u128,
}

impl NcmcRngCounterRange {
    fn end_exclusive(&self) -> Result<u128, String> {
        self.start
            .checked_add(self.count)
            .ok_or_else(|| "NCMC RNG counter range overflow".to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NcmcMoveRngAllocation {
    #[serde(with = "u128_decimal_string")]
    pub opportunity_counter_base: u128,
    #[serde(with = "u128_decimal_string")]
    pub selection_counter: u128,
    pub placement: NcmcRngCounterRange,
    pub orientation: NcmcRngCounterRange,
    pub momenta: NcmcRngCounterRange,
    pub thermostat: NcmcRngCounterRange,
    #[serde(with = "u128_decimal_string")]
    pub acceptance_counter: u128,
}

impl NcmcMoveRngAllocation {
    fn validate_disjoint(&self) -> Result<(), String> {
        let mut ranges = vec![
            (
                self.selection_counter,
                self.selection_counter.checked_add(1),
            ),
            (self.placement.start, Some(self.placement.end_exclusive()?)),
            (
                self.orientation.start,
                Some(self.orientation.end_exclusive()?),
            ),
            (self.momenta.start, Some(self.momenta.end_exclusive()?)),
            (
                self.thermostat.start,
                Some(self.thermostat.end_exclusive()?),
            ),
            (
                self.acceptance_counter,
                self.acceptance_counter.checked_add(1),
            ),
        ];
        if self.placement.count == 0
            || self.orientation.count == 0
            || self.momenta.count == 0
            || self.thermostat.count == 0
            || ranges.iter().any(|(_, end)| end.is_none())
        {
            return Err("NCMC RNG allocation has an empty or overflowing range".into());
        }
        ranges.sort_unstable_by_key(|range| range.0);
        if ranges.windows(2).any(|pair| pair[0].1.unwrap() > pair[1].0) {
            return Err("NCMC RNG allocation subranges overlap".into());
        }
        Ok(())
    }
}

/// Semantic owner of one disjoint counter-RNG subrange.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum NcmcRngPurpose {
    Selection,
    Placement,
    Orientation,
    Momenta,
    Thermostat,
    Acceptance,
}

/// Exact physical reason a reserved counter range was not sampled. Reserved
/// counters remain irrevocably consumed by the opportunity and may never be
/// reused after either acceptance or rejection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum NcmcRngNonConsumptionReason {
    PersistentDecoupledCoordinatesRetained,
    StableSlotIdentityTransform,
    ExistingEquilibriumMomentaRetained,
    DeterministicGammaZeroThermostat,
}

/// Counter-range disposition. `CONSUMED` binds every deterministic draw to
/// its measure; `NOT_APPLICABLE` and `FORBIDDEN` are typed physical states,
/// not silent counter burning.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "disposition",
    rename_all = "SCREAMING_SNAKE_CASE",
    deny_unknown_fields
)]
pub enum NcmcRngRangeDisposition {
    Consumed {
        #[serde(with = "u128_decimal_string")]
        draw_count: u128,
        sampling_measure: String,
        deterministic_draws_sha256: String,
    },
    NotApplicable {
        physical_reason: NcmcRngNonConsumptionReason,
    },
    Forbidden {
        physical_reason: NcmcRngNonConsumptionReason,
    },
}

fn ncmc_rng_draws_sha256(
    rng_key: u64,
    purpose: NcmcRngPurpose,
    range: &NcmcRngCounterRange,
    sampling_measure: &str,
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"PRISM_NCMC_COUNTER_RANGE_DRAWS_V1\0");
    digest.update(rng_key.to_le_bytes());
    digest.update(serde_json::to_vec(&purpose).expect("serialize RNG purpose"));
    digest.update(range.start.to_le_bytes());
    digest.update(range.count.to_le_bytes());
    digest.update(sampling_measure.as_bytes());
    format!("{:x}", digest.finalize())
}

impl NcmcRngRangeDisposition {
    pub fn consumed(
        rng_key: u64,
        purpose: NcmcRngPurpose,
        range: &NcmcRngCounterRange,
        sampling_measure: impl Into<String>,
    ) -> Result<Self, String> {
        let sampling_measure = sampling_measure.into();
        if range.count == 0 || sampling_measure.trim().is_empty() {
            return Err("consumed NCMC RNG range is empty or lacks its measure".into());
        }
        Ok(Self::Consumed {
            draw_count: range.count,
            deterministic_draws_sha256: ncmc_rng_draws_sha256(
                rng_key,
                purpose,
                range,
                &sampling_measure,
            ),
            sampling_measure,
        })
    }

    fn validate(
        &self,
        rng_key: u64,
        purpose: NcmcRngPurpose,
        range: &NcmcRngCounterRange,
    ) -> Result<(), String> {
        match self {
            Self::Consumed {
                draw_count,
                sampling_measure,
                deterministic_draws_sha256,
            } => {
                let expected_measure = match purpose {
                    NcmcRngPurpose::Selection => "STATE_CONDITIONED_SUPPORT_CATEGORICAL_V2",
                    NcmcRngPurpose::Placement => "UNIFORM_ACCESSIBLE_VOLUME_CARTESIAN_V1",
                    NcmcRngPurpose::Orientation => "UNIFORM_SO3_QUATERNION_V1",
                    NcmcRngPurpose::Momenta => "MAXWELL_BOLTZMANN_CONSTRAINED_V1",
                    NcmcRngPurpose::Thermostat => "GAMMA_ZERO_FORBIDS_THERMOSTAT_DRAWS",
                    NcmcRngPurpose::Acceptance => "METROPOLIS_UNIFORM_OPEN01_V1",
                };
                if range.count == 0
                    || *draw_count != range.count
                    || sampling_measure != expected_measure
                    || *deterministic_draws_sha256
                        != ncmc_rng_draws_sha256(rng_key, purpose, range, sampling_measure)
                {
                    return Err("NCMC consumed RNG range does not close exactly".into());
                }
            }
            Self::NotApplicable { physical_reason } => {
                if *physical_reason == NcmcRngNonConsumptionReason::DeterministicGammaZeroThermostat
                {
                    return Err("gamma-zero thermostat counters must be FORBIDDEN".into());
                }
            }
            Self::Forbidden { physical_reason } => {
                if purpose != NcmcRngPurpose::Thermostat
                    || *physical_reason
                        != NcmcRngNonConsumptionReason::DeterministicGammaZeroThermostat
                {
                    return Err("only gamma-zero thermostat RNG may be FORBIDDEN".into());
                }
            }
        }
        Ok(())
    }
}

/// Receipt binding every proposal counter subrange to an exact physical use
/// or typed non-use. The reservation survives rollback and therefore closes
/// the no-counter-reuse invariant independently of device-state restoration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NcmcRngConsumptionReceipt {
    pub schema_kind: String,
    pub schema_version: u32,
    pub scheduler_contract_sha256: String,
    pub move_ordinal: u64,
    pub allocation: NcmcMoveRngAllocation,
    pub selection: NcmcRngRangeDisposition,
    pub placement: NcmcRngRangeDisposition,
    pub orientation: NcmcRngRangeDisposition,
    pub momenta: NcmcRngRangeDisposition,
    pub thermostat: NcmcRngRangeDisposition,
    pub acceptance: NcmcRngRangeDisposition,
    pub counters_irrevocably_reserved_after_rejection: bool,
    pub receipt_sha256: String,
}

impl NcmcRngConsumptionReceipt {
    fn recompute_sha256(&self) -> Result<String, String> {
        let mut projection = self.clone();
        projection.receipt_sha256.clear();
        let bytes = serde_json::to_vec(&projection).map_err(|error| error.to_string())?;
        let mut digest = Sha256::new();
        digest.update(b"PRISM_NCMC_RNG_CONSUMPTION_RECEIPT_V2\0");
        digest.update(bytes);
        Ok(format!("{:x}", digest.finalize()))
    }

    pub fn seal(
        proposal: &NcmcMoveProposal,
        placement: NcmcRngRangeDisposition,
        orientation: NcmcRngRangeDisposition,
        momenta: NcmcRngRangeDisposition,
        thermostat: NcmcRngRangeDisposition,
    ) -> Result<Self, String> {
        let selection_range = NcmcRngCounterRange {
            start: proposal.rng_allocation.selection_counter,
            count: 1,
        };
        let acceptance_range = NcmcRngCounterRange {
            start: proposal.rng_allocation.acceptance_counter,
            count: 1,
        };
        let mut receipt = Self {
            schema_kind: "prismos_ncmc_rng_consumption_receipt".into(),
            schema_version: 2,
            scheduler_contract_sha256: proposal.scheduler_contract_sha256.clone(),
            move_ordinal: proposal.move_id,
            allocation: proposal.rng_allocation.clone(),
            selection: NcmcRngRangeDisposition::consumed(
                proposal.rng_key,
                NcmcRngPurpose::Selection,
                &selection_range,
                "STATE_CONDITIONED_SUPPORT_CATEGORICAL_V2",
            )?,
            placement,
            orientation,
            momenta,
            thermostat,
            acceptance: NcmcRngRangeDisposition::consumed(
                proposal.rng_key,
                NcmcRngPurpose::Acceptance,
                &acceptance_range,
                "METROPOLIS_UNIFORM_OPEN01_V1",
            )?,
            counters_irrevocably_reserved_after_rejection: true,
            receipt_sha256: String::new(),
        };
        receipt.receipt_sha256 = receipt.recompute_sha256()?;
        receipt.validate_for_proposal(proposal)?;
        Ok(receipt)
    }

    pub fn validate_for_proposal(&self, proposal: &NcmcMoveProposal) -> Result<(), String> {
        if self.schema_kind != "prismos_ncmc_rng_consumption_receipt"
            || self.schema_version != 2
            || self.scheduler_contract_sha256 != proposal.scheduler_contract_sha256
            || self.move_ordinal != proposal.move_id
            || self.allocation != proposal.rng_allocation
            || !self.counters_irrevocably_reserved_after_rejection
        {
            return Err("NCMC RNG consumption receipt/proposal binding mismatch".into());
        }
        let selection_range = NcmcRngCounterRange {
            start: self.allocation.selection_counter,
            count: 1,
        };
        let acceptance_range = NcmcRngCounterRange {
            start: self.allocation.acceptance_counter,
            count: 1,
        };
        for (purpose, range, disposition) in [
            (NcmcRngPurpose::Selection, &selection_range, &self.selection),
            (
                NcmcRngPurpose::Placement,
                &self.allocation.placement,
                &self.placement,
            ),
            (
                NcmcRngPurpose::Orientation,
                &self.allocation.orientation,
                &self.orientation,
            ),
            (
                NcmcRngPurpose::Momenta,
                &self.allocation.momenta,
                &self.momenta,
            ),
            (
                NcmcRngPurpose::Thermostat,
                &self.allocation.thermostat,
                &self.thermostat,
            ),
            (
                NcmcRngPurpose::Acceptance,
                &acceptance_range,
                &self.acceptance,
            ),
        ] {
            disposition.validate(proposal.rng_key, purpose, range)?;
        }
        if !matches!(self.selection, NcmcRngRangeDisposition::Consumed { .. })
            || !matches!(self.acceptance, NcmcRngRangeDisposition::Consumed { .. })
            || !matches!(
                self.thermostat,
                NcmcRngRangeDisposition::Forbidden {
                    physical_reason: NcmcRngNonConsumptionReason::DeterministicGammaZeroThermostat
                }
            )
            || self.receipt_sha256 != self.recompute_sha256()?
        {
            return Err("NCMC RNG selection/acceptance/thermostat disposition is invalid".into());
        }
        let pair_matches = |reason: NcmcRngNonConsumptionReason| {
            matches!(
                self.placement,
                NcmcRngRangeDisposition::NotApplicable { physical_reason } if physical_reason == reason
            ) && matches!(
                self.orientation,
                NcmcRngRangeDisposition::NotApplicable { physical_reason } if physical_reason == reason
            )
        };
        let uniform_pair_consumed =
            matches!(self.placement, NcmcRngRangeDisposition::Consumed { .. })
                && matches!(self.orientation, NcmcRngRangeDisposition::Consumed { .. });
        let pose_valid = match proposal.move_kind {
            NcmcMoveKind::Insertion | NcmcMoveKind::Deletion => {
                pair_matches(NcmcRngNonConsumptionReason::PersistentDecoupledCoordinatesRetained)
                    || uniform_pair_consumed
            }
            NcmcMoveKind::Identity | NcmcMoveKind::Tautomer | NcmcMoveKind::Protonation => {
                pair_matches(NcmcRngNonConsumptionReason::StableSlotIdentityTransform)
            }
        };
        if !pose_valid
            || !matches!(
                self.momenta,
                NcmcRngRangeDisposition::NotApplicable {
                    physical_reason:
                        NcmcRngNonConsumptionReason::ExistingEquilibriumMomentaRetained
                }
            )
        {
            return Err("NCMC RNG disposition disagrees with move physics".into());
        }
        Ok(())
    }

    /// Bind pose RNG disposition to the scheduler's declared placement
    /// measure. This check requires the pre-run scheduler and is therefore
    /// repeated by the runtime emitter before durable typed emission.
    pub fn validate_for_scheduler(
        &self,
        proposal: &NcmcMoveProposal,
        scheduler: &NcmcProposalSchedulerContract,
    ) -> Result<(), String> {
        self.validate_for_proposal(proposal)?;
        scheduler.validate_proposal(proposal)?;
        match (&scheduler.placement_measure, proposal.move_kind) {
            (
                NcmcPlacementMeasureAuthority::PersistentDecoupledCoordinateEnsemble { .. },
                NcmcMoveKind::Insertion | NcmcMoveKind::Deletion,
            ) => {
                for disposition in [&self.placement, &self.orientation] {
                    if !matches!(
                        disposition,
                        NcmcRngRangeDisposition::NotApplicable {
                            physical_reason:
                                NcmcRngNonConsumptionReason::PersistentDecoupledCoordinatesRetained
                        }
                    ) {
                        return Err("persistent-decoupled scheduler consumed pose RNG".into());
                    }
                }
            }
            (
                NcmcPlacementMeasureAuthority::UniformAccessibleVolume { .. },
                NcmcMoveKind::Insertion | NcmcMoveKind::Deletion,
            ) => {
                if !matches!(self.placement, NcmcRngRangeDisposition::Consumed { .. })
                    || !matches!(self.orientation, NcmcRngRangeDisposition::Consumed { .. })
                {
                    return Err("uniform-volume scheduler lacks consumed pose RNG".into());
                }
            }
            (_, NcmcMoveKind::Identity | NcmcMoveKind::Tautomer | NcmcMoveKind::Protonation) => {}
        }
        Ok(())
    }
}

impl NcmcRngCounterLayout {
    fn allocation(&self, base: u128) -> Result<NcmcMoveRngAllocation, String> {
        let add = |offset: u128| {
            base.checked_add(offset)
                .ok_or_else(|| "NCMC RNG allocation overflow".to_string())
        };
        let allocation = NcmcMoveRngAllocation {
            opportunity_counter_base: base,
            selection_counter: add(self.selection_counter_offset)?,
            placement: NcmcRngCounterRange {
                start: add(self.placement_counter_start_offset)?,
                count: self.placement_counter_count,
            },
            orientation: NcmcRngCounterRange {
                start: add(self.orientation_counter_start_offset)?,
                count: self.orientation_counter_count,
            },
            momenta: NcmcRngCounterRange {
                start: add(self.momentum_counter_start_offset)?,
                count: self.momentum_counter_count,
            },
            thermostat: NcmcRngCounterRange {
                start: add(self.thermostat_counter_start_offset)?,
                count: self.thermostat_counter_count,
            },
            acceptance_counter: add(self.acceptance_counter_offset)?,
        };
        allocation.validate_disjoint()?;
        let end = base
            .checked_add(self.counters_per_opportunity)
            .ok_or_else(|| "NCMC RNG opportunity span overflow".to_string())?;
        let occupied_ends = [
            allocation.selection_counter.checked_add(1).unwrap(),
            allocation.placement.end_exclusive()?,
            allocation.orientation.end_exclusive()?,
            allocation.momenta.end_exclusive()?,
            allocation.thermostat.end_exclusive()?,
            allocation.acceptance_counter.checked_add(1).unwrap(),
        ];
        if self.counters_per_opportunity == 0 || occupied_ends.into_iter().any(|value| value > end)
        {
            return Err("NCMC RNG allocation exceeds its opportunity span".into());
        }
        Ok(allocation)
    }
}

/// Immutable proposal-scheduler authority consumed at runtime. It binds the
/// exact probability measure, cadence, equilibration/decorrelation rule and
/// disjoint counter-RNG allocation before any proposal can be generated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NcmcProposalSchedulerContract {
    pub schema_kind: String,
    pub schema_version: u32,
    pub scheduler_id: String,
    /// Complete endpoint-state map required at every decision.
    pub admitted_slot_ids: Vec<u32>,
    /// Complete directed proposal kernel, including explicit reverse supports.
    pub proposal_supports: Vec<NcmcProposalSupport>,
    /// Decision times and RNG reservations; outcomes remain state-conditioned.
    pub scheduled_opportunities: Vec<NcmcScheduledOpportunity>,
    pub placement_measure: NcmcPlacementMeasureAuthority,
    pub first_move_step: u64,
    pub move_interval_steps: u64,
    pub equilibrated_after_step: u64,
    pub minimum_decorrelation_steps: u64,
    pub maximum_move_count: u64,
    pub rng_algorithm: String,
    pub rng_domain: String,
    pub rng_key: u64,
    #[serde(with = "u128_decimal_string")]
    pub rng_counter_start: u128,
    pub rng_counter_layout: NcmcRngCounterLayout,
    pub bias_epoch_sha256: String,
    pub forward_proposal_density_formula: String,
    pub reverse_proposal_density_formula: String,
    pub ncmc_protocol_schedule_sha256: String,
    pub contract_sha256: String,
}

impl NcmcProposalSchedulerContract {
    fn recompute_sha256(&self) -> Result<String, String> {
        let mut projection = self.clone();
        projection.contract_sha256.clear();
        let bytes = serde_json::to_vec(&projection).map_err(|error| error.to_string())?;
        let mut digest = Sha256::new();
        digest.update(b"PRISM_NCMC_PROPOSAL_SCHEDULER_V3\0");
        digest.update(bytes);
        Ok(format!("{:x}", digest.finalize()))
    }

    fn support(&self, support_id: &str) -> Option<&NcmcProposalSupport> {
        self.proposal_supports
            .iter()
            .find(|support| support.support_id == support_id)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_kind != "prismos_ncmc_proposal_scheduler_contract"
            || self.schema_version != 3
            || self.scheduler_id.trim().is_empty()
            || self.admitted_slot_ids.is_empty()
            || self.proposal_supports.is_empty()
            || self.scheduled_opportunities.is_empty()
            || self.move_interval_steps == 0
            || self.minimum_decorrelation_steps < self.move_interval_steps
            || self.maximum_move_count == 0
            || self.rng_algorithm.trim().is_empty()
            || self.rng_domain.trim().is_empty()
            || self.forward_proposal_density_formula.trim().is_empty()
            || self.reverse_proposal_density_formula.trim().is_empty()
            || self.contract_sha256 != self.recompute_sha256()?
        {
            return Err("invalid NCMC proposal scheduler v3 contract".into());
        }
        self.placement_measure.validate()?;
        validate_sha256("NCMC scheduler bias epoch", &self.bias_epoch_sha256)?;
        validate_sha256(
            "NCMC scheduler protocol schedule",
            &self.ncmc_protocol_schedule_sha256,
        )?;
        if self
            .admitted_slot_ids
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        {
            return Err("NCMC admitted slot IDs must be unique and ascending".into());
        }
        let mut support_ids = BTreeSet::new();
        for support in &self.proposal_supports {
            support.validate()?;
            if !support_ids.insert(support.support_id.clone())
                || self
                    .admitted_slot_ids
                    .binary_search(&support.slot_id)
                    .is_err()
                || support
                    .coupled_charge_partner
                    .as_ref()
                    .map(|partner| {
                        self.admitted_slot_ids
                            .binary_search(&partner.slot_id)
                            .is_err()
                    })
                    .unwrap_or(false)
            {
                return Err("NCMC support identity or admitted-slot binding is invalid".into());
            }
            if support.balance_template.bias_epoch_sha256 != self.bias_epoch_sha256 {
                return Err("NCMC support bias epoch differs from scheduler epoch".into());
            }
        }
        for support in &self.proposal_supports {
            let reverse = self.support(&support.reverse_support_id).ok_or_else(|| {
                format!("NCMC support {} lacks named reverse", support.support_id)
            })?;
            let reverse_kind = match support.move_kind {
                NcmcMoveKind::Insertion => NcmcMoveKind::Deletion,
                NcmcMoveKind::Deletion => NcmcMoveKind::Insertion,
                value => value,
            };
            let expected_partner =
                support
                    .coupled_charge_partner
                    .as_ref()
                    .map(|partner| NcmcCoupledChargePartner {
                        slot_id: partner.slot_id,
                        initial: partner.proposed.clone(),
                        proposed: partner.initial.clone(),
                        charge_change_e: -partner.charge_change_e,
                    });
            if reverse.reverse_support_id != support.support_id
                || reverse.slot_id != support.slot_id
                || reverse.move_kind != reverse_kind
                || reverse.initial != support.proposed
                || reverse.proposed != support.initial
                || reverse.periodic_charge_treatment != support.periodic_charge_treatment.reversed()
                || reverse.coupled_charge_partner != expected_partner
                || !support
                    .balance_template
                    .reversed_matches(&reverse.balance_template)
            {
                return Err(format!(
                    "NCMC support {} and reverse {} are not exact involutions",
                    support.support_id, reverse.support_id
                ));
            }
        }
        if self.scheduled_opportunities.len() as u64 != self.maximum_move_count {
            return Err(
                "NCMC scheduled opportunity population differs from maximum move count".into(),
            );
        }
        self.rng_counter_layout.allocation(self.rng_counter_start)?;
        for (ordinal, opportunity) in self.scheduled_opportunities.iter().enumerate() {
            let ordinal = ordinal as u64;
            let expected_step = self
                .first_move_step
                .checked_add(
                    self.move_interval_steps
                        .checked_mul(ordinal)
                        .ok_or_else(|| "NCMC scheduled step overflow".to_string())?,
                )
                .ok_or_else(|| "NCMC scheduled step overflow".to_string())?;
            let expected_counter_base = self
                .rng_counter_start
                .checked_add(
                    self.rng_counter_layout
                        .counters_per_opportunity
                        .checked_mul(u128::from(ordinal))
                        .ok_or_else(|| "NCMC scheduled counter overflow".to_string())?,
                )
                .ok_or_else(|| "NCMC scheduled counter overflow".to_string())?;
            let mut allowed = BTreeSet::new();
            if opportunity.move_ordinal != ordinal
                || opportunity.step != expected_step
                || opportunity.rng_counter_base != expected_counter_base
                || opportunity.step < self.equilibrated_after_step
                || opportunity.allowed_support_ids.is_empty()
                || opportunity.allowed_support_ids.iter().any(|support_id| {
                    !allowed.insert(support_id.clone()) || !support_ids.contains(support_id)
                })
            {
                return Err("NCMC scheduled opportunity cadence/counter/support mismatch".into());
            }
            if opportunity.allowed_support_ids.iter().any(|support_id| {
                let support = self.support(support_id).unwrap();
                !allowed.contains(&support.reverse_support_id)
            }) {
                return Err("NCMC opportunity omits a named reverse support".into());
            }
            self.rng_counter_layout
                .allocation(opportunity.rng_counter_base)?;
        }
        self.rng_counter_start
            .checked_add(
                self.rng_counter_layout
                    .counters_per_opportunity
                    .checked_mul(u128::from(self.maximum_move_count))
                    .ok_or_else(|| "NCMC RNG allocation overflow".to_string())?,
            )
            .ok_or_else(|| "NCMC RNG allocation overflow".to_string())?;
        Ok(())
    }

    pub fn is_scheduled_step(&self, step: u64) -> bool {
        step >= self.first_move_step
            && step >= self.equilibrated_after_step
            && (step - self.first_move_step) % self.move_interval_steps == 0
            && (step - self.first_move_step) / self.move_interval_steps < self.maximum_move_count
    }

    /// Return the first immutable NCMC opportunity at or after `step`.
    /// Runtime chunk planners use this boundary to ensure ordinary MD can
    /// never advance across a scheduled transaction. The full scheduler is
    /// revalidated before the boundary is trusted.
    pub fn next_opportunity_at_or_after(
        &self,
        step: u64,
    ) -> Result<Option<&NcmcScheduledOpportunity>, String> {
        self.validate()?;
        Ok(self
            .scheduled_opportunities
            .iter()
            .find(|opportunity| opportunity.step >= step))
    }

    pub fn validate_proposal(&self, proposal: &NcmcMoveProposal) -> Result<(), String> {
        self.validate()?;
        proposal.validate()?;
        let support = self
            .support(&proposal.selected_support_id)
            .ok_or_else(|| "NCMC proposal selected support is absent".to_string())?;
        let opportunity = self
            .scheduled_opportunities
            .get(proposal.move_id as usize)
            .ok_or_else(|| "NCMC proposal move ordinal is outside schedule".to_string())?;
        if proposal.scheduler_contract_sha256 != self.contract_sha256
            || proposal.rng_key != self.rng_key
            || !opportunity
                .allowed_support_ids
                .contains(&support.support_id)
            || proposal.slot_id != support.slot_id
            || proposal.move_kind != support.move_kind
            || proposal.proposed_species_id != support.proposed.species_id
            || proposal.proposed_chemical_state_id != support.proposed.chemical_state_id
            || proposal.proposed_occupancy != support.proposed.occupancy
            || proposal.periodic_charge_treatment != support.periodic_charge_treatment
            || proposal.coupled_charge_partner != support.coupled_charge_partner
            || proposal.rng_allocation
                != self
                    .rng_counter_layout
                    .allocation(opportunity.rng_counter_base)?
            || proposal.rng_counter != proposal.rng_allocation.acceptance_counter
            || proposal.balance_context.forward_selection_probability <= 0.0
            || proposal.balance_context.reverse_selection_probability <= 0.0
        {
            return Err("NCMC proposal lacks scheduler v2 support".into());
        }
        Ok(())
    }

    pub fn validate_proposal_at_step(
        &self,
        step: u64,
        proposal: &NcmcMoveProposal,
        current_endpoints: &BTreeMap<u32, NcmcEndpointIdentity>,
    ) -> Result<(), String> {
        self.validate_proposal(proposal)?;
        if !self.is_scheduled_step(step) {
            return Err("NCMC proposal is outside scheduler cadence/equilibration".into());
        }
        let expected =
            self.proposal_for_step(step, current_endpoints, proposal.execution_mode.clone())?;
        if serde_json::to_vec(&expected).map_err(|error| error.to_string())?
            != serde_json::to_vec(proposal).map_err(|error| error.to_string())?
        {
            return Err("NCMC proposal differs from state-conditioned scheduler output".into());
        }
        Ok(())
    }

    /// Select one directed support from the current eligible population and
    /// compute the reverse probability from the proposed state's population.
    pub fn proposal_for_step(
        &self,
        step: u64,
        current_endpoints: &BTreeMap<u32, NcmcEndpointIdentity>,
        execution_mode: NcmcExecutionMode,
    ) -> Result<NcmcMoveProposal, String> {
        self.validate()?;
        execution_mode.validate()?;
        let matches = self
            .scheduled_opportunities
            .iter()
            .filter(|opportunity| opportunity.step == step)
            .collect::<Vec<_>>();
        if matches.len() != 1 {
            return Err(format!(
                "NCMC scheduler has {} opportunities at step {step}",
                matches.len()
            ));
        }
        let opportunity = matches[0];
        if current_endpoints.keys().copied().collect::<Vec<_>>() != self.admitted_slot_ids {
            return Err("NCMC scheduler received a partial or foreign endpoint-state map".into());
        }
        let eligible = opportunity
            .allowed_support_ids
            .iter()
            .map(|support_id| {
                self.support(support_id)
                    .ok_or_else(|| format!("NCMC opportunity support {support_id} is absent"))
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .filter(|support| support.eligible(current_endpoints))
            .collect::<Vec<_>>();
        let forward_weight_sum = eligible
            .iter()
            .map(|support| support.base_weight)
            .sum::<f64>();
        if eligible.is_empty() || !forward_weight_sum.is_finite() || forward_weight_sum <= 0.0 {
            return Err("NCMC opportunity has no positive eligible forward support".into());
        }
        let rng_allocation = self
            .rng_counter_layout
            .allocation(opportunity.rng_counter_base)?;
        let draw = counter_based_uniform_open01(self.rng_key, rng_allocation.selection_counter);
        let mut cumulative = 0.0;
        let mut selected = *eligible.last().unwrap();
        for support in &eligible {
            cumulative += support.base_weight / forward_weight_sum;
            if draw < cumulative {
                selected = support;
                break;
            }
        }
        let forward_probability = selected.base_weight / forward_weight_sum;
        let mut proposed_endpoints = current_endpoints.clone();
        proposed_endpoints.insert(selected.slot_id, selected.proposed.clone());
        if let Some(partner) = &selected.coupled_charge_partner {
            proposed_endpoints.insert(partner.slot_id, partner.proposed.clone());
        }
        let reverse_eligible = opportunity
            .allowed_support_ids
            .iter()
            .filter_map(|support_id| self.support(support_id))
            .filter(|support| support.eligible(&proposed_endpoints))
            .collect::<Vec<_>>();
        let reverse_weight_sum = reverse_eligible
            .iter()
            .map(|support| support.base_weight)
            .sum::<f64>();
        let reverse = reverse_eligible
            .iter()
            .find(|support| support.support_id == selected.reverse_support_id)
            .copied()
            .ok_or_else(|| {
                "NCMC selected move has zero reverse support in proposed state".to_string()
            })?;
        if !reverse_weight_sum.is_finite() || reverse_weight_sum <= 0.0 {
            return Err("NCMC proposed state has no positive reverse support population".into());
        }
        let reverse_probability = reverse.base_weight / reverse_weight_sum;
        if !forward_probability.is_finite()
            || !reverse_probability.is_finite()
            || forward_probability <= 0.0
            || reverse_probability <= 0.0
            || forward_probability > 1.0
            || reverse_probability > 1.0
        {
            return Err("NCMC normalized forward/reverse support probability is invalid".into());
        }
        let occupied_count_before = current_endpoints
            .values()
            .filter(|endpoint| endpoint.occupancy == EndpointOccupancy::Occupied)
            .count() as u64;
        let occupied_count_after = proposed_endpoints
            .values()
            .filter(|endpoint| endpoint.occupancy == EndpointOccupancy::Occupied)
            .count() as u64;
        let proposal = NcmcMoveProposal {
            move_id: opportunity.move_ordinal,
            slot_id: selected.slot_id,
            move_kind: selected.move_kind,
            execution_mode,
            scheduler_contract_sha256: self.contract_sha256.clone(),
            selected_support_id: selected.support_id.clone(),
            proposed_species_id: selected.proposed.species_id,
            proposed_chemical_state_id: selected.proposed.chemical_state_id,
            proposed_occupancy: selected.proposed.occupancy,
            balance_context: selected.balance_template.materialize(
                occupied_count_before,
                occupied_count_after,
                forward_probability,
                reverse_probability,
            ),
            periodic_charge_treatment: selected.periodic_charge_treatment.clone(),
            coupled_charge_partner: selected.coupled_charge_partner.clone(),
            rng_key: self.rng_key,
            rng_counter: rng_allocation.acceptance_counter,
            rng_allocation,
        };
        proposal.validate()?;
        Ok(proposal)
    }
}

/// Bound an ordinary-MD chunk so it cannot cross an immutable NCMC
/// opportunity, including non-divisible final chunks. A transaction scheduled
/// at the current frontier must already have resolved before any MD advances.
pub fn bound_ordinary_md_chunk_before_ncmc(
    scheduler: &NcmcProposalSchedulerContract,
    current_step: u64,
    requested_steps: u64,
    current_opportunity_resolved: bool,
) -> Result<u64, String> {
    scheduler.validate()?;
    if scheduler.is_scheduled_step(current_step) && !current_opportunity_resolved {
        return Err(
            "ordinary MD cannot advance over an unresolved current NCMC opportunity".into(),
        );
    }
    let search_step = if current_opportunity_resolved {
        current_step
            .checked_add(1)
            .ok_or_else(|| "NCMC planner frontier overflow".to_string())?
    } else {
        current_step
    };
    let bound = scheduler
        .next_opportunity_at_or_after(search_step)?
        .map(|opportunity| {
            opportunity
                .step
                .checked_sub(current_step)
                .ok_or_else(|| "NCMC opportunity precedes ordinary MD frontier".to_string())
        })
        .transpose()?
        .unwrap_or(requested_steps);
    Ok(requested_steps.min(bound))
}

impl NcmcExecutionMode {
    fn validate(&self) -> Result<(), String> {
        if let Self::CalibrationWorkCollection {
            protocol_authority_sha256,
        } = self
        {
            validate_sha256(
                "NCMC calibration-work protocol authority",
                protocol_authority_sha256,
            )?;
        }
        Ok(())
    }

    fn is_calibration_work_collection(&self) -> bool {
        matches!(self, Self::CalibrationWorkCollection { .. })
    }
}

/// One engine-owned NCMC proposal.  Proposal-density and target-density terms
/// are explicit inputs because placement, identity-selection and pH proposals
/// are experiment-specific; the engine never invents those probabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NcmcMoveProposal {
    /// Stable move id within the run lineage.
    pub move_id: u64,
    /// Slot to perturb.
    pub slot_id: u32,
    /// Thermodynamic move class.
    pub move_kind: NcmcMoveKind,
    /// Calibration-work or calibrated target-lane authority.
    pub execution_mode: NcmcExecutionMode,
    /// Exact pre-run scheduler authority that generated this proposal.
    pub scheduler_contract_sha256: String,
    /// Directed state-conditioned support selected by the scheduler.
    pub selected_support_id: String,
    /// Proposed species (equal to the current species unless identity changes).
    pub proposed_species_id: u32,
    /// Proposed tautomer/protonation state.
    pub proposed_chemical_state_id: u32,
    /// Proposed physical endpoint occupancy.
    pub proposed_occupancy: EndpointOccupancy,
    /// Thermodynamic and proposal-density inputs from which the engine derives
    /// the complete target/proposal ratios.
    pub balance_context: NcmcBalanceContext,
    /// Physical treatment of this directed move's net-cell-charge change.
    /// It is checked against the exact endpoint charges and calibration.
    pub periodic_charge_treatment: PeriodicChargeTreatment,
    /// Simultaneously switched ion/co-ion endpoint change. Required exactly
    /// for `COUPLED_CHARGE_NEUTRAL_PATH` and forbidden otherwise.
    pub coupled_charge_partner: Option<NcmcCoupledChargePartner>,
    /// Counter-based RNG key.
    pub rng_key: u64,
    /// Counter used exactly once by the accept/reject draw.
    #[serde(with = "u128_decimal_string")]
    pub rng_counter: u128,
    /// Disjoint counters reserved for every stochastic component of this move.
    pub rng_allocation: NcmcMoveRngAllocation,
}

impl NcmcMoveProposal {
    /// Fail closed before any device state is changed.
    pub fn validate(&self) -> Result<(), String> {
        self.execution_mode.validate()?;
        validate_sha256("NCMC proposal scheduler", &self.scheduler_contract_sha256)?;
        if self.selected_support_id.trim().is_empty()
            || matches!(self.proposed_occupancy, EndpointOccupancy::Switching)
            || self.rng_counter != self.rng_allocation.acceptance_counter
        {
            return Err("an NCMC proposal must terminate at a physical endpoint".into());
        }
        self.rng_allocation.validate_disjoint()?;
        self.balance_context.validate()?;
        match (
            &self.periodic_charge_treatment,
            &self.coupled_charge_partner,
        ) {
            (
                PeriodicChargeTreatment::CoupledChargeNeutralPath {
                    partner_slot_id,
                    partner_charge_change_e,
                    ..
                },
                Some(partner),
            ) if partner.slot_id == *partner_slot_id
                && partner.charge_change_e == *partner_charge_change_e =>
            {
                partner.validate()?;
            }
            (PeriodicChargeTreatment::CoupledChargeNeutralPath { .. }, _) => {
                return Err("coupled charge path lacks its exact partner move".into())
            }
            (_, Some(_)) => return Err("uncoupled NCMC move cannot carry a charge partner".into()),
            (_, None) => {}
        }
        Ok(())
    }
}

/// Exact simultaneous counterion/co-ion endpoint move used to preserve cell
/// charge during one NCMC proposal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NcmcCoupledChargePartner {
    pub slot_id: u32,
    pub initial: NcmcEndpointIdentity,
    pub proposed: NcmcEndpointIdentity,
    pub charge_change_e: i32,
}

impl NcmcCoupledChargePartner {
    fn validate(&self) -> Result<(), String> {
        if self.initial.occupancy == EndpointOccupancy::Switching
            || self.proposed.occupancy == EndpointOccupancy::Switching
            || self.initial == self.proposed
            || self.charge_change_e == 0
        {
            return Err("invalid coupled charge-partner endpoint move".into());
        }
        Ok(())
    }
}

/// Non-log, auditable inputs for detailed-balance ratio derivation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NcmcBalanceContext {
    /// Actual move temperature.
    pub temperature_k: f64,
    /// Actual move pH.
    pub ph: f64,
    /// Accessible insertion volume in A^3.
    pub accessible_volume_a3: f64,
    /// Occupied slots before the move.
    pub occupied_count_before: u64,
    /// Occupied slots after the proposed move.
    pub occupied_count_after: u64,
    /// Forward species/state selection probability.
    pub forward_selection_probability: f64,
    /// Reverse species/state selection probability.
    pub reverse_selection_probability: f64,
    /// Forward placement probability density in A^-3.
    pub forward_placement_density_a3_inv: f64,
    /// Reverse placement probability density in A^-3.
    pub reverse_placement_density_a3_inv: f64,
    /// Forward orientation probability density.
    pub forward_orientation_density: f64,
    /// Reverse orientation probability density.
    pub reverse_orientation_density: f64,
    /// Initial target-bias weight (one for an unbiased lane).
    pub initial_bias_weight: f64,
    /// Proposed target-bias weight (one for an unbiased lane).
    pub proposed_bias_weight: f64,
    /// Immutable adaptive/bias epoch identity.
    pub bias_epoch_sha256: String,
}

/// Declared base measures for the raw forward/reverse proposal factors in an
/// NCMC receipt.  This prevents a discrete selection probability from being
/// conflated with a translational or orientational probability density.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NcmcProposalMeasureContract {
    /// Schema discriminator.
    pub schema_kind: String,
    /// Schema version.
    pub schema_version: u32,
    /// Measure used by identity/state selection factors.
    pub selection_measure: String,
    /// Measure used by insertion/deletion placement factors.
    pub placement_measure: String,
    /// Measure used by orientation factors.
    pub orientation_measure: String,
    /// Exact convention used to compose the recorded log proposal ratio.
    pub ratio_convention: String,
}

impl NcmcProposalMeasureContract {
    /// Canonical measures used by the native balance derivation.
    pub fn canonical() -> Self {
        Self {
            schema_kind: "prismos_ncmc_proposal_measure_contract".into(),
            schema_version: 2,
            selection_measure: "DISCRETE_PROBABILITY".into(),
            placement_measure: "PROBABILITY_DENSITY_ANGSTROM^-3".into(),
            orientation_measure: "NORMALIZED_ORIENTATION_PROBABILITY_DENSITY".into(),
            ratio_convention: "LOG_REVERSE_OVER_FORWARD_PRODUCT".into(),
        }
    }

    fn validate(&self) -> Result<(), String> {
        if self != &Self::canonical() {
            return Err("NCMC proposal base-measure contract is not canonical".into());
        }
        Ok(())
    }
}

impl NcmcBalanceContext {
    fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("temperature_k", self.temperature_k),
            ("ph", self.ph),
            ("accessible_volume_a3", self.accessible_volume_a3),
            (
                "forward_selection_probability",
                self.forward_selection_probability,
            ),
            (
                "reverse_selection_probability",
                self.reverse_selection_probability,
            ),
            (
                "forward_placement_density_a3_inv",
                self.forward_placement_density_a3_inv,
            ),
            (
                "reverse_placement_density_a3_inv",
                self.reverse_placement_density_a3_inv,
            ),
            (
                "forward_orientation_density",
                self.forward_orientation_density,
            ),
            (
                "reverse_orientation_density",
                self.reverse_orientation_density,
            ),
            ("initial_bias_weight", self.initial_bias_weight),
            ("proposed_bias_weight", self.proposed_bias_weight),
        ] {
            if !value.is_finite() || (name != "ph" && value <= 0.0) {
                return Err(format!("invalid NCMC balance term {name}"));
            }
        }
        if self.forward_selection_probability > 1.0 || self.reverse_selection_probability > 1.0 {
            return Err("NCMC discrete selection probability exceeds one".into());
        }
        validate_sha256("bias epoch", &self.bias_epoch_sha256)
    }
}

/// Engine-derived, term-resolved detailed-balance ratios.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NcmcDerivedBalance {
    /// Inverse thermal energy beta in mol/kcal for unit reconstruction.
    pub beta_mol_per_kcal: f64,
    /// beta times calibrated chemical-potential change.
    pub beta_chemical_potential_term: f64,
    /// Portion of beta*delta(mu) arising from the intrinsic state reference
    /// already included in the calibrated total chemical potential.
    pub beta_intrinsic_state_reference_term_in_total_mu: f64,
    /// Remaining beta*delta(mu), excluding that state-reference portion.
    pub beta_nonreference_chemical_potential_term: f64,
    /// Portion of beta*delta(mu) arising from the box/PME-bound finite-size
    /// correction, included exactly once in calibrated total mu.
    pub beta_periodic_finite_size_correction_term_in_total_mu: f64,
    /// Remaining beta*delta(mu) after both intrinsic-state and periodic
    /// finite-size reference contributions are removed.
    pub beta_calibrated_mu_remainder_term: f64,
    /// Exact physical treatment proving that a net-charge-changing move is
    /// neutralized or corrected; a neutralizing plasma alone is inadmissible.
    pub periodic_charge_treatment: PeriodicChargeTreatment,
    /// Exact convention proving why mechanical switching work excludes the
    /// intrinsic state reference.
    pub state_reference_convention: StateReferenceConvention,
    /// pH/proton-count log-weight term.
    pub ph_log_weight_term: f64,
    /// Standard-state/access-volume Jacobian.
    pub standard_state_volume_term: f64,
    /// Occupancy combinatorial term.
    pub occupancy_combinatorics_term: f64,
    /// Target-bias log-weight difference.
    pub target_bias_term: f64,
    /// Complete target-density ratio.
    pub log_target_ratio: f64,
    /// Species/state selection proposal ratio.
    pub selection_proposal_term: f64,
    /// Placement proposal-density ratio.
    pub placement_proposal_term: f64,
    /// Orientation proposal-density ratio.
    pub orientation_proposal_term: f64,
    /// Complete reverse/forward proposal ratio.
    pub log_proposal_ratio: f64,
    /// Calibration/bias inputs hash.
    pub derivation_sha256: String,
}

impl NcmcDerivedBalance {
    fn validate_terms(&self) -> Result<(), String> {
        for (name, value) in [
            ("beta_mol_per_kcal", self.beta_mol_per_kcal),
            (
                "beta_chemical_potential_term",
                self.beta_chemical_potential_term,
            ),
            (
                "beta_intrinsic_state_reference_term_in_total_mu",
                self.beta_intrinsic_state_reference_term_in_total_mu,
            ),
            (
                "beta_nonreference_chemical_potential_term",
                self.beta_nonreference_chemical_potential_term,
            ),
            (
                "beta_periodic_finite_size_correction_term_in_total_mu",
                self.beta_periodic_finite_size_correction_term_in_total_mu,
            ),
            (
                "beta_calibrated_mu_remainder_term",
                self.beta_calibrated_mu_remainder_term,
            ),
            ("ph_log_weight_term", self.ph_log_weight_term),
            (
                "standard_state_volume_term",
                self.standard_state_volume_term,
            ),
            (
                "occupancy_combinatorics_term",
                self.occupancy_combinatorics_term,
            ),
            ("target_bias_term", self.target_bias_term),
            ("log_target_ratio", self.log_target_ratio),
            ("selection_proposal_term", self.selection_proposal_term),
            ("placement_proposal_term", self.placement_proposal_term),
            ("orientation_proposal_term", self.orientation_proposal_term),
            ("log_proposal_ratio", self.log_proposal_ratio),
        ] {
            if !value.is_finite() {
                return Err(format!("nonfinite derived NCMC term {name}"));
            }
        }
        if self.beta_mol_per_kcal <= 0.0 {
            return Err("NCMC beta must be positive".into());
        }
        if (self.beta_chemical_potential_term
            - self.beta_intrinsic_state_reference_term_in_total_mu
            - self.beta_nonreference_chemical_potential_term)
            .abs()
            > 1.0e-12
        {
            return Err(
                "intrinsic state reference is omitted or double-counted in delta(mu)".into(),
            );
        }
        if (self.beta_nonreference_chemical_potential_term
            - self.beta_periodic_finite_size_correction_term_in_total_mu
            - self.beta_calibrated_mu_remainder_term)
            .abs()
            > 1.0e-12
        {
            return Err(
                "periodic finite-size correction is omitted or double-counted in delta(mu)".into(),
            );
        }
        match &self.periodic_charge_treatment {
            PeriodicChargeTreatment::NetChargeInvariant {
                subject_charge_change_e,
            } => {
                if *subject_charge_change_e != 0
                    || self
                        .beta_periodic_finite_size_correction_term_in_total_mu
                        .abs()
                        > 1.0e-12
                {
                    return Err("invalid net-charge-invariant NCMC decomposition".into());
                }
            }
            PeriodicChargeTreatment::CoupledChargeNeutralPath {
                subject_charge_change_e,
                partner_charge_change_e,
                combined_charge_change_e,
                coupling_schedule_sha256,
                evidence_sha256,
                ..
            } => {
                if *subject_charge_change_e == 0
                    || *partner_charge_change_e != -*subject_charge_change_e
                    || *combined_charge_change_e != 0
                    || self
                        .beta_periodic_finite_size_correction_term_in_total_mu
                        .abs()
                        > 1.0e-12
                {
                    return Err("coupled NCMC charge path fails exact charge conservation".into());
                }
                validate_sha256("charge coupling schedule", coupling_schedule_sha256)?;
                validate_sha256("charge-neutral path evidence", evidence_sha256)?;
            }
            PeriodicChargeTreatment::ExplicitFiniteSizeCorrection {
                subject_charge_change_e,
                correction_model,
                correction_free_energy_kcal_mol,
                correction_uncertainty_kcal_mol,
                evidence_sha256,
                ..
            } => {
                let admitted_model = matches!(
                    correction_model.as_str(),
                    "ROCKLIN_NET_CHARGE_SOLVATION_V1"
                        | "HUMMER_PRATT_GARCIA_V1"
                        | "NUMERIC_BOX_PME_TRANSFER_V1"
                );
                if *subject_charge_change_e == 0
                    || !admitted_model
                    || !correction_free_energy_kcal_mol.is_finite()
                    || !correction_uncertainty_kcal_mol.is_finite()
                    || *correction_uncertainty_kcal_mol < 0.0
                    || (self.beta_mol_per_kcal * correction_free_energy_kcal_mol
                        - self.beta_periodic_finite_size_correction_term_in_total_mu)
                        .abs()
                        > 1.0e-12
                {
                    return Err("finite-size-corrected NCMC decomposition does not close".into());
                }
                validate_sha256("finite-size correction evidence", evidence_sha256)?;
            }
        }
        let target = self.beta_chemical_potential_term
            + self.ph_log_weight_term
            + self.standard_state_volume_term
            + self.occupancy_combinatorics_term
            + self.target_bias_term;
        let proposal = self.selection_proposal_term
            + self.placement_proposal_term
            + self.orientation_proposal_term;
        if (self.log_target_ratio - target).abs() > 1.0e-12
            || (self.log_proposal_ratio - proposal).abs() > 1.0e-12
        {
            return Err("derived NCMC totals do not close their term decomposition".into());
        }
        Ok(())
    }
}

/// Matched experimental lane for runtime and post-MD attribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum LambdaExperimentLaneKind {
    /// Matched dynamics with zero fragment slots and zero virtual probes.
    NoProbe,
    /// Explicit fragments selected for the pocket hypothesis.
    HypothesisFragments,
    /// Explicit size/shape-matched, chemically orthogonal controls.
    OrthogonalChemicalControls,
}

/// Hash-bound lane declaration. Virtual probes are forbidden because they do
/// not carry molecular identity, chemical potential or endpoint state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LambdaExperimentLaneContract {
    /// Schema discriminator.
    pub schema_kind: String,
    /// Schema version.
    pub schema_version: u32,
    /// Stable lane identity within the matched experiment.
    pub lane_id: String,
    /// Lane class.
    pub lane_kind: LambdaExperimentLaneKind,
    /// Exact species IDs admitted in this lane, sorted and unique.
    pub fragment_species_ids: Vec<u32>,
    /// Must be zero for every physical lane.
    pub virtual_probe_count: u64,
    /// Digest over every preceding field.
    pub contract_sha256: String,
}

impl LambdaExperimentLaneContract {
    /// Construct the canonical no-probe contract without fabricating a slot.
    pub fn no_probe(lane_id: impl Into<String>) -> Result<Self, String> {
        let mut value = Self {
            schema_kind: "prismos_lambda_experiment_lane_contract".into(),
            schema_version: 1,
            lane_id: lane_id.into(),
            lane_kind: LambdaExperimentLaneKind::NoProbe,
            fragment_species_ids: Vec::new(),
            virtual_probe_count: 0,
            contract_sha256: String::new(),
        };
        value.contract_sha256 = value.recompute_sha256()?;
        value.validate(None)?;
        Ok(value)
    }

    fn recompute_sha256(&self) -> Result<String, String> {
        let mut projection = self.clone();
        projection.contract_sha256.clear();
        let bytes = serde_json::to_vec(&projection).map_err(|error| error.to_string())?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }

    /// Verify lane identity against the actual fragment topology.
    pub fn validate(&self, system: Option<&LambdaFragmentSystem>) -> Result<(), String> {
        if self.schema_kind != "prismos_lambda_experiment_lane_contract"
            || self.schema_version != 1
            || self.lane_id.trim().is_empty()
            || self.virtual_probe_count != 0
            || self.contract_sha256 != self.recompute_sha256()?
        {
            return Err("invalid lambda experiment lane contract".into());
        }
        let mut species = self.fragment_species_ids.clone();
        species.sort_unstable();
        species.dedup();
        if species != self.fragment_species_ids {
            return Err("lane fragment species IDs must be sorted and unique".into());
        }
        match (self.lane_kind, system) {
            (LambdaExperimentLaneKind::NoProbe, None) if species.is_empty() => Ok(()),
            (LambdaExperimentLaneKind::NoProbe, _) => {
                Err("no-probe lane must have zero physical fragment slots/species".into())
            }
            (
                LambdaExperimentLaneKind::HypothesisFragments
                | LambdaExperimentLaneKind::OrthogonalChemicalControls,
                Some(system),
            ) => {
                let mut actual: Vec<u32> =
                    system.species.iter().map(|item| item.species_id).collect();
                actual.sort_unstable();
                actual.dedup();
                if species.is_empty() || species != actual || system.slots.is_empty() {
                    return Err(
                        "probe lane species/slots disagree with physical fragment topology".into(),
                    );
                }
                Ok(())
            }
            _ => Err("probe lane lacks a physical fragment topology".into()),
        }
    }
}

/// Authenticated source for a persistent decoupled fragment coordinate
/// ensemble. Nested generator/measure objects remain explicit JSON contracts
/// because their scientific implementations are producer-specific; the native
/// runtime nevertheless hash-binds their exact bytes and admits only an
/// authority whose quantitative mixing gates all pass.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecoupledFragmentPlacementMeasureAuthority {
    pub schema_kind: String,
    pub schema_version: u32,
    pub authority_id: String,
    pub status: String,
    pub calibration_sampling_admission_allowed: bool,
    pub production_admission_allowed: bool,
    pub precursor_placement_measure_sha256: Option<String>,
    pub promotion_receipt_sha256: Option<String>,
    pub chemical_potential_evidence_sha256: Option<String>,
    pub cross_cell_unlock_sha256: Option<String>,
    pub finite_size_admission_sha256: Option<String>,
    pub seed_topology_member_sha256: Option<String>,
    pub panel_scientific_sha256: Option<String>,
    pub target_protocol_authority_sha256: Option<String>,
    pub source_branch_authority_sha256: String,
    pub ncmc_placement_projection_sha256: String,
    pub generator: serde_json::Value,
    pub thermodynamic_authority_sha256: String,
    pub cell_authority_sha256: String,
    pub cell_id: String,
    pub periodic_cell_sha256: String,
    pub temperature_k: f64,
    pub species_id: u32,
    pub chemical_state_id: u32,
    pub parameter_template_sha256: String,
    pub endpoint_parameter_sha256: String,
    pub slot_count: u32,
    pub position_measure: serde_json::Value,
    pub orientation_measure: serde_json::Value,
    pub internal_measure: serde_json::Value,
    pub slot_selection_measure: serde_json::Value,
    pub ncmc_path_reversal: serde_json::Value,
    pub coordinate_ensemble_sha256: String,
    pub mixing_evidence_sha256: String,
    pub trajectory_artifacts: Vec<serde_json::Value>,
    pub com_fractional_uniformity_gate_pass: bool,
    pub orientation_haar_gate_pass: bool,
    pub per_torsion_mixing_ess_gate_pass: bool,
    pub slot_exchangeability_gate_pass: bool,
    pub all_proposal_densities_finite_nonzero: bool,
    pub forward_reverse_proposal_density_exact: bool,
    pub mixing_gate_pass: bool,
    pub placement_measure_sha256: String,
}

impl DecoupledFragmentPlacementMeasureAuthority {
    pub fn recompute_sha256(&self) -> Result<String, String> {
        let mut projection = self.clone();
        projection.placement_measure_sha256.clear();
        let mut digest = Sha256::new();
        digest.update(b"PRISM_DECOUPLED_FRAGMENT_PLACEMENT_MEASURE_AUTHORITY_V1\0");
        digest.update(serde_json::to_vec(&projection).map_err(|error| error.to_string())?);
        Ok(format!("{:x}", digest.finalize()))
    }

    pub fn validate(&self) -> Result<(), String> {
        let promotion_fields = [
            self.precursor_placement_measure_sha256.as_deref(),
            self.promotion_receipt_sha256.as_deref(),
            self.chemical_potential_evidence_sha256.as_deref(),
            self.cross_cell_unlock_sha256.as_deref(),
            self.finite_size_admission_sha256.as_deref(),
            self.seed_topology_member_sha256.as_deref(),
            self.panel_scientific_sha256.as_deref(),
            self.target_protocol_authority_sha256.as_deref(),
        ];
        if self.schema_kind != "prismos_decoupled_fragment_placement_measure_authority"
            || self.schema_version != 1
            || self.authority_id.trim().is_empty()
            || self.status != "IMPLEMENTATION_VERIFIED"
            || !self.calibration_sampling_admission_allowed
            || !self.temperature_k.is_finite()
            || self.temperature_k <= 0.0
            || self.cell_id.trim().is_empty()
            || self.slot_count == 0
            || !self.generator.is_object()
            || !self.position_measure.is_object()
            || !self.orientation_measure.is_object()
            || !self.internal_measure.is_object()
            || !self.slot_selection_measure.is_object()
            || !self.ncmc_path_reversal.is_object()
            || self.trajectory_artifacts.is_empty()
            || self
                .trajectory_artifacts
                .iter()
                .any(|artifact| !artifact.is_object())
            || !self.com_fractional_uniformity_gate_pass
            || !self.orientation_haar_gate_pass
            || !self.per_torsion_mixing_ess_gate_pass
            || !self.slot_exchangeability_gate_pass
            || !self.all_proposal_densities_finite_nonzero
            || !self.forward_reverse_proposal_density_exact
            || !self.mixing_gate_pass
            || (self.production_admission_allowed
                && promotion_fields.iter().any(|value| value.is_none()))
            || (!self.production_admission_allowed
                && promotion_fields.iter().any(|value| value.is_some()))
            || self.placement_measure_sha256 != self.recompute_sha256()?
        {
            return Err(
                "decoupled fragment placement authority is not production-admissible".into(),
            );
        }
        for (label, digest) in [
            (
                "source coordinate branch authority",
                &self.source_branch_authority_sha256,
            ),
            (
                "native NCMC placement projection",
                &self.ncmc_placement_projection_sha256,
            ),
            (
                "placement thermodynamic authority",
                &self.thermodynamic_authority_sha256,
            ),
            ("placement cell authority", &self.cell_authority_sha256),
            ("placement periodic cell", &self.periodic_cell_sha256),
            (
                "fragment parameter template",
                &self.parameter_template_sha256,
            ),
            (
                "fragment endpoint parameter",
                &self.endpoint_parameter_sha256,
            ),
            (
                "decoupled coordinate ensemble",
                &self.coordinate_ensemble_sha256,
            ),
            ("decoupled mixing evidence", &self.mixing_evidence_sha256),
        ] {
            validate_sha256(label, digest)?;
        }
        if self.production_admission_allowed {
            let precursor = self.precursor_placement_measure_sha256.as_deref().unwrap();
            let promotion = self.promotion_receipt_sha256.as_deref().unwrap();
            let mut precursor_projection = self.clone();
            precursor_projection.production_admission_allowed = false;
            precursor_projection.precursor_placement_measure_sha256 = None;
            precursor_projection.promotion_receipt_sha256 = None;
            precursor_projection.chemical_potential_evidence_sha256 = None;
            precursor_projection.cross_cell_unlock_sha256 = None;
            precursor_projection.finite_size_admission_sha256 = None;
            precursor_projection.seed_topology_member_sha256 = None;
            precursor_projection.panel_scientific_sha256 = None;
            precursor_projection.target_protocol_authority_sha256 = None;
            precursor_projection.placement_measure_sha256.clear();
            let expected_precursor = precursor_projection.recompute_sha256()?;
            let expected_promotion = decoupled_fragment_promotion_receipt_sha256(
                precursor,
                self.chemical_potential_evidence_sha256.as_deref().unwrap(),
                self.cross_cell_unlock_sha256.as_deref().unwrap(),
                self.finite_size_admission_sha256.as_deref().unwrap(),
                self.seed_topology_member_sha256.as_deref().unwrap(),
                self.panel_scientific_sha256.as_deref().unwrap(),
                self.target_protocol_authority_sha256.as_deref().unwrap(),
            )?;
            if precursor != expected_precursor
                || precursor == self.placement_measure_sha256
                || promotion != expected_promotion
            {
                return Err(
                    "target placement promotion receipt is cyclic or does not reconstruct".into(),
                );
            }
        }
        for artifact in &self.trajectory_artifacts {
            let object = artifact
                .as_object()
                .ok_or_else(|| "placement trajectory artifact is not an object".to_string())?;
            let path = object
                .get("path")
                .and_then(serde_json::Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .ok_or_else(|| "placement trajectory artifact lacks path".to_string())?;
            let _ = path;
            let digest = object
                .get("sha256")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| "placement trajectory artifact lacks sha256".to_string())?;
            validate_sha256("placement trajectory artifact", digest)?;
        }
        Ok(())
    }
}

/// Engine binding required for runtime-inline typed emission.  It contains
/// only pre-run identities and schedule state; no post-run observation may be
/// embedded in this request contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LambdaRuntimeBinding {
    /// Immutable run identity.
    pub run_id: String,
    /// Immutable replica identity.
    pub replica_id: String,
    /// Replica RNG seed.
    pub replica_seed: u64,
    /// Actual native stream identity.
    pub stream_id: u32,
    /// Exact per-stream seed.
    pub stream_seed: u64,
    /// Selected engine binary hash.
    pub engine_sha256: String,
    /// Exact input-topology hash.
    pub topology_sha256: String,
    /// Package-time proof that every directed scheduler support is exactly one
    /// of the prepared fixed A/B endpoint pairs. Absent only for no-probe
    /// lanes; fragment runtimes carry it into the resident union contract and
    /// every force/move event.
    pub endpoint_support_binding_sha256: Option<String>,
    /// Live-rehashed engine/topology/PTX capability authority. Runtime engine
    /// construction requires this field; post-run/manual emitter tests may
    /// leave it absent but cannot execute MD.
    pub runtime_capability_receipt: Option<LambdaRuntimeCapabilityReceipt>,
    /// Common-random-number contract that makes the environment atom prefix
    /// bit-identical across matched no-probe and fragment-bearing topologies.
    pub matched_velocity_initialization: LambdaMatchedVelocityInitializationContract,
    /// Full source artifact behind persistent decoupled coordinates. It is
    /// absent only for no-probe lanes and is independently reconciled against
    /// both the matched suffix lineage and scheduler placement measure.
    pub decoupled_fragment_placement_measure_authority:
        Option<DecoupledFragmentPlacementMeasureAuthority>,
    /// Package-selected terminal cross-cell unlock authority. Present only
    /// for a promoted target lane and matched exactly to the placement
    /// promotion receipt; calibration and no-probe lanes must omit it.
    pub target_cross_cell_unlock_sha256: Option<String>,
    /// Independent panel-science authority selected by the package. This is
    /// intentionally distinct from the lane identity/contract hash.
    pub panel_scientific_sha256: Option<String>,
    /// Matched lane declaration.
    pub lane: LambdaExperimentLaneContract,
    /// NCMC authority for physical fragment lanes; absent only for no-probe.
    pub ncmc_execution_mode: Option<NcmcExecutionMode>,
    /// Exact proposal-measure/cadence/RNG authority; absent only for no-probe.
    pub ncmc_scheduler: Option<NcmcProposalSchedulerContract>,
    /// Initial engine-owned protocol phase interval.
    pub initial_protocol_phase: crate::chemical_state_fiber::ProtocolPhaseBinding,
    /// Complete thermodynamic condition active for the initial phase.
    pub initial_thermodynamic_condition: crate::chemical_state_fiber::ThermodynamicCondition,
    /// Complete, content-addressed stream schedule. The first entry must equal
    /// the initial phase/condition above. Runtime phase transitions are
    /// admitted only from this immutable sequence.
    pub protocol_schedule: Vec<LambdaRuntimePhaseContext>,
    /// Existing, canonical run output root under which typed output is created.
    /// The emitter traverses this root with no-follow directory semantics.
    pub run_output_root: PathBuf,
    /// Directory owned by the active run, not a post-run recovery directory.
    pub typed_event_output_dir: PathBuf,
    /// Pre-run whole-campaign capacity/inode authority. The emitter refuses
    /// launch unless the complete maximum population fits with safety margin.
    pub typed_event_capacity: LambdaTypedEventCapacityContract,
}

/// Pre-run common-random-number and topology-membership authority for matched
/// branch initialization. Environment atoms are an exact prefix; every atom
/// after the prefix must belong to one explicitly decoupled fragment slot.
/// Coordinates are never sampled here: the environment prefix is transferred
/// byte-exactly from an admitted frozen-NVT checkpoint and the fragment suffix
/// is transferred byte-exactly from its authenticated coordinate branch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LambdaMatchedVelocityInitializationContract {
    pub schema_kind: String,
    pub schema_version: u32,
    pub environment_atom_count: u32,
    pub environment_atom_membership_sha256: String,
    pub environment_mass_sha256: String,
    pub environment_velocity_rng_domain: String,
    pub fragment_velocity_rng_domain: String,
    pub environment_position_transfer: String,
    pub environment_position_f64_checkpoint_sha256: String,
    pub f64_to_f32_projection_policy: String,
    pub environment_position_sha256: String,
    pub fragment_position_transfer: String,
    pub fragment_position_f64_checkpoint_sha256: Option<String>,
    pub fragment_position_sha256: Option<String>,
    pub fragment_coordinate_branch_authority_sha256: Option<String>,
    pub fragment_coordinate_ensemble_sha256: Option<String>,
    pub periodic_cell_contract_sha256: String,
    pub periodic_cell_f64_checkpoint_sha256: String,
    pub periodic_cell_f32_projection_sha256: String,
    pub npt_checkpoint_population_authority_sha256: String,
    pub npt_checkpoint_population_distinct_seed_count: u32,
    pub npt_seed_authority_sha256: String,
    pub npt_frozen_cell_authority_sha256: String,
    pub frozen_checkpoint_replica_seed: u64,
    pub contract_sha256: String,
}

impl LambdaMatchedVelocityInitializationContract {
    pub(crate) fn recompute_sha256(&self) -> Result<String, String> {
        let mut projection = self.clone();
        projection.contract_sha256.clear();
        let mut digest = Sha256::new();
        digest.update(b"PRISM_LAMBDA_MATCHED_VELOCITY_INITIALIZATION_CONTRACT_V3\0");
        digest.update(serde_json::to_vec(&projection).map_err(|error| error.to_string())?);
        Ok(format!("{:x}", digest.finalize()))
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_kind != "prismos_lambda_matched_velocity_initialization_contract"
            || self.schema_version != 3
            || self.environment_atom_count == 0
            || self.environment_position_transfer
                != "EXACT_AFTER_AUTHORIZED_F64_TO_F32_PROJECTION_NO_LATER_MUTATION"
            || self.f64_to_f32_projection_policy
                != "IEEE754_BINARY64_TO_BINARY32_ROUND_TO_NEAREST_TIES_TO_EVEN"
            || self.npt_checkpoint_population_distinct_seed_count < 5
            || self.contract_sha256 != self.recompute_sha256()?
        {
            return Err("invalid matched velocity initialization contract".into());
        }
        validate_sha256(
            "environment atom membership",
            &self.environment_atom_membership_sha256,
        )?;
        validate_sha256("environment mass", &self.environment_mass_sha256)?;
        validate_sha256(
            "frozen checkpoint environment position f64",
            &self.environment_position_f64_checkpoint_sha256,
        )?;
        validate_sha256("environment position", &self.environment_position_sha256)?;
        validate_sha256("periodic cell", &self.periodic_cell_contract_sha256)?;
        validate_sha256(
            "periodic cell f64 checkpoint",
            &self.periodic_cell_f64_checkpoint_sha256,
        )?;
        validate_sha256(
            "periodic cell f32 projection",
            &self.periodic_cell_f32_projection_sha256,
        )?;
        validate_sha256(
            "NPT checkpoint population authority",
            &self.npt_checkpoint_population_authority_sha256,
        )?;
        validate_sha256("NPT seed authority", &self.npt_seed_authority_sha256)?;
        validate_sha256(
            "NPT frozen cell authority",
            &self.npt_frozen_cell_authority_sha256,
        )?;
        for (value, label) in [
            (
                &self.environment_velocity_rng_domain,
                "environment velocity RNG domain",
            ),
            (
                &self.fragment_velocity_rng_domain,
                "fragment velocity RNG domain",
            ),
        ] {
            if value.trim().is_empty() {
                return Err(format!("{label} is empty"));
            }
        }
        if self.environment_velocity_rng_domain == self.fragment_velocity_rng_domain {
            return Err("environment and fragment RNG domains must be disjoint".into());
        }
        match (
            self.fragment_position_transfer.as_str(),
            &self.fragment_position_f64_checkpoint_sha256,
            &self.fragment_position_sha256,
            &self.fragment_coordinate_branch_authority_sha256,
            &self.fragment_coordinate_ensemble_sha256,
        ) {
            ("NOT_APPLICABLE_NO_PROBE", None, None, None, None) => {}
            (
                "EXACT_AUTHENTICATED_DECOUPLED_COORDINATE_BRANCH_BYTES",
                Some(position_f64_sha256),
                Some(position_sha256),
                Some(branch_sha256),
                Some(ensemble_sha256),
            ) => {
                validate_sha256("fragment position f64 checkpoint", position_f64_sha256)?;
                validate_sha256("fragment position", position_sha256)?;
                validate_sha256("fragment coordinate branch authority", branch_sha256)?;
                validate_sha256("fragment coordinate ensemble", ensemble_sha256)?;
            }
            _ => return Err("invalid fragment coordinate transfer authority".into()),
        }
        Ok(())
    }

    pub fn validate_for_topology(
        &self,
        masses: &[f32],
        positions: &[f32],
        periodic_cell_contract_sha256: Option<&str>,
        periodic_cell_vectors_f64: Option<[[f64; 3]; 3]>,
        npt_frozen_cell_contract: Option<&serde_json::Value>,
        system: Option<&LambdaFragmentSystem>,
    ) -> Result<(), String> {
        self.validate()?;
        let environment_atom_count = self.environment_atom_count as usize;
        if environment_atom_count > masses.len()
            || positions.len() != masses.len().saturating_mul(3)
            || self.environment_atom_membership_sha256
                != matched_environment_atom_membership_sha256(environment_atom_count)
            || self.environment_mass_sha256
                != matched_environment_mass_sha256(&masses[..environment_atom_count])
            || self.environment_position_sha256
                != matched_environment_position_sha256(
                    &positions[..environment_atom_count.saturating_mul(3)],
                )
            || periodic_cell_contract_sha256 != Some(self.periodic_cell_contract_sha256.as_str())
        {
            return Err(
                "matched frozen environment prefix, mass, position, or cell differs from topology"
                    .into(),
            );
        }
        let periodic_cell_vectors_f64 = periodic_cell_vectors_f64
            .ok_or_else(|| "matched topology lacks exact periodic-cell vectors".to_string())?;
        if self.periodic_cell_f64_checkpoint_sha256
            != matched_periodic_cell_f64_sha256(&periodic_cell_vectors_f64)
            || self.periodic_cell_f32_projection_sha256
                != matched_periodic_cell_f32_projection_sha256(&periodic_cell_vectors_f64)
        {
            return Err("matched topology periodic-cell f64/f32 lineage differs".into());
        }
        let frozen = npt_frozen_cell_contract
            .ok_or_else(|| "matched topology lacks frozen-NVT checkpoint authority".to_string())?;
        let frozen_field = |name: &str| frozen.get(name).and_then(serde_json::Value::as_str);
        if frozen_field("schema_kind") != Some("prismos_npt_frozen_nvt_topology_authority")
            || frozen
                .get("schema_version")
                .and_then(serde_json::Value::as_u64)
                != Some(2)
            || frozen_field("status") != Some("IMPLEMENTATION_VERIFIED")
            || frozen_field("coordinate_and_cell_transfer")
                != Some("EXACT_FROM_HASH_BOUND_STATE_XML")
            || frozen_field("positions_f64_sha256")
                != Some(self.environment_position_f64_checkpoint_sha256.as_str())
            || frozen_field("positions_f32_projection_sha256")
                != Some(self.environment_position_sha256.as_str())
            || frozen_field("f64_to_f32_projection_policy")
                != Some(self.f64_to_f32_projection_policy.as_str())
            || frozen_field("population_authority_sha256")
                != Some(self.npt_checkpoint_population_authority_sha256.as_str())
            || frozen_field("seed_authority_sha256")
                != Some(self.npt_seed_authority_sha256.as_str())
            || frozen_field("frozen_authority_sha256")
                != Some(self.npt_frozen_cell_authority_sha256.as_str())
            || frozen_field("periodic_cell_sha256")
                != Some(self.periodic_cell_contract_sha256.as_str())
            || frozen_field("periodic_cell_vectors_f64_sha256")
                != Some(self.periodic_cell_f64_checkpoint_sha256.as_str())
            || frozen_field("periodic_cell_vectors_f32_projection_sha256")
                != Some(self.periodic_cell_f32_projection_sha256.as_str())
            || frozen
                .get("replica_seed")
                .and_then(serde_json::Value::as_u64)
                != Some(self.frozen_checkpoint_replica_seed)
        {
            return Err(
                "matched topology frozen-NVT checkpoint authority differs from binding".into(),
            );
        }
        match system {
            None if environment_atom_count == masses.len()
                && self.fragment_position_f64_checkpoint_sha256.is_none()
                && self.fragment_position_sha256.is_none()
                && self.fragment_coordinate_branch_authority_sha256.is_none()
                && self.fragment_coordinate_ensemble_sha256.is_none() =>
            {
                Ok(())
            }
            None => Err("no-probe topology contains atoms outside the environment prefix".into()),
            Some(system) => {
                let mut fragment_atoms = system
                    .slots
                    .iter()
                    .flat_map(|slot| slot.atom_indices.iter().copied())
                    .collect::<Vec<_>>();
                fragment_atoms.sort_unstable();
                if fragment_atoms.windows(2).any(|pair| pair[0] == pair[1])
                    || fragment_atoms != (environment_atom_count..masses.len()).collect::<Vec<_>>()
                {
                    return Err(
                        "fragment slots must partition the exact suffix after the matched environment prefix"
                            .into(),
                    );
                }
                let fragment_positions = &positions[environment_atom_count * 3..];
                if frozen_field("fragment_positions_f64_sha256")
                    != self.fragment_position_f64_checkpoint_sha256.as_deref()
                    || frozen_field("fragment_positions_f32_projection_sha256")
                        != self.fragment_position_sha256.as_deref()
                    || self.fragment_position_sha256.as_deref()
                        != Some(matched_fragment_position_sha256(fragment_positions).as_str())
                    || self.fragment_coordinate_branch_authority_sha256.is_none()
                    || self.fragment_coordinate_ensemble_sha256.is_none()
                {
                    return Err(
                        "fragment suffix differs from authenticated coordinate branch".into(),
                    );
                }
                Ok(())
            }
        }
    }
}

pub fn matched_environment_atom_membership_sha256(environment_atom_count: usize) -> String {
    let mut digest = Sha256::new();
    digest.update(b"PRISM_MATCHED_ENVIRONMENT_ATOM_PREFIX_V1\0");
    for atom_index in 0..environment_atom_count as u64 {
        digest.update(atom_index.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

pub fn matched_environment_mass_sha256(masses: &[f32]) -> String {
    let mut digest = Sha256::new();
    digest.update(b"PRISM_MATCHED_ENVIRONMENT_MASS_PREFIX_F32_V1\0");
    for mass in masses {
        digest.update(mass.to_bits().to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

pub fn matched_environment_position_sha256(positions: &[f32]) -> String {
    matched_position_sha256(
        b"PRISM_MATCHED_ENVIRONMENT_POSITION_PREFIX_F32_V2\0",
        positions,
    )
}

pub fn matched_fragment_position_sha256(positions: &[f32]) -> String {
    matched_position_sha256(
        b"PRISM_MATCHED_FRAGMENT_POSITION_SUFFIX_F32_V2\0",
        positions,
    )
}

/// Hash the exact binary64 decoupled-fragment suffix before the authorized
/// IEEE-754 binary64-to-binary32 RNE projection. The source branch identity
/// remains a separate digest because it additionally binds slot/atom mapping
/// and producer authority.
pub fn matched_fragment_position_f64_sha256(positions: &[f64]) -> String {
    let mut digest = Sha256::new();
    digest.update(b"PRISM_MATCHED_FRAGMENT_POSITION_SUFFIX_F64_V1\0");
    for position in positions {
        digest.update(position.to_bits().to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

pub fn matched_periodic_cell_f64_sha256(vectors: &[[f64; 3]; 3]) -> String {
    let mut digest = Sha256::new();
    digest.update(b"PRISM_MATCHED_PERIODIC_CELL_VECTORS_F64_V1\0");
    for vector in vectors {
        for value in vector {
            digest.update(value.to_bits().to_le_bytes());
        }
    }
    format!("{:x}", digest.finalize())
}

pub fn matched_periodic_cell_f32_projection_sha256(vectors: &[[f64; 3]; 3]) -> String {
    let mut digest = Sha256::new();
    digest.update(b"PRISM_MATCHED_PERIODIC_CELL_VECTORS_RNE_F32_V1\0");
    for vector in vectors {
        for value in vector {
            digest.update((*value as f32).to_bits().to_le_bytes());
        }
    }
    format!("{:x}", digest.finalize())
}

fn matched_position_sha256(domain: &[u8], positions: &[f32]) -> String {
    let mut digest = Sha256::new();
    digest.update(domain);
    for position in positions {
        digest.update(position.to_bits().to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

/// Actual initialized bytes, emitted after replica reset and bound into the
/// terminal closeout receipt.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LambdaMatchedVelocityInitializationReceipt {
    pub schema_kind: String,
    pub schema_version: u32,
    pub stream_id: u16,
    pub replica_seed: u64,
    pub temperature_k: f32,
    pub contract_sha256: String,
    pub environment_atom_count: u32,
    pub fragment_atom_count: u32,
    pub environment_position_f64_checkpoint_sha256: String,
    pub f64_to_f32_projection_policy: String,
    pub periodic_cell_contract_sha256: String,
    pub periodic_cell_f64_checkpoint_sha256: String,
    pub periodic_cell_f32_projection_sha256: String,
    pub npt_checkpoint_population_authority_sha256: String,
    pub npt_seed_authority_sha256: String,
    pub npt_frozen_cell_authority_sha256: String,
    pub fragment_position_f64_checkpoint_sha256: Option<String>,
    pub fragment_position_sha256: Option<String>,
    pub fragment_coordinate_branch_authority_sha256: Option<String>,
    pub fragment_coordinate_ensemble_sha256: Option<String>,
    pub environment_velocity_sha256: String,
    pub full_velocity_sha256: String,
    pub environment_position_sha256: String,
    pub full_position_sha256: String,
    pub receipt_sha256: String,
}

impl LambdaMatchedVelocityInitializationReceipt {
    fn recompute_sha256(&self) -> Result<String, String> {
        let mut projection = self.clone();
        projection.receipt_sha256.clear();
        let mut digest = Sha256::new();
        digest.update(b"PRISM_LAMBDA_MATCHED_VELOCITY_INITIALIZATION_RECEIPT_V3\0");
        digest.update(serde_json::to_vec(&projection).map_err(|error| error.to_string())?);
        Ok(format!("{:x}", digest.finalize()))
    }

    pub fn seal(mut self) -> Result<Self, String> {
        self.receipt_sha256.clear();
        self.receipt_sha256 = self.recompute_sha256()?;
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_kind != "prismos_lambda_matched_velocity_initialization_receipt"
            || self.schema_version != 3
            || !self.temperature_k.is_finite()
            || self.temperature_k <= 0.0
            || self.environment_atom_count == 0
            || self.f64_to_f32_projection_policy
                != "IEEE754_BINARY64_TO_BINARY32_ROUND_TO_NEAREST_TIES_TO_EVEN"
            || self.receipt_sha256 != self.recompute_sha256()?
        {
            return Err("invalid matched velocity initialization receipt".into());
        }
        for (label, digest) in [
            ("matched velocity contract", &self.contract_sha256),
            (
                "frozen checkpoint environment position f64",
                &self.environment_position_f64_checkpoint_sha256,
            ),
            ("periodic cell", &self.periodic_cell_contract_sha256),
            (
                "periodic cell f64 checkpoint",
                &self.periodic_cell_f64_checkpoint_sha256,
            ),
            (
                "periodic cell f32 projection",
                &self.periodic_cell_f32_projection_sha256,
            ),
            (
                "NPT checkpoint population authority",
                &self.npt_checkpoint_population_authority_sha256,
            ),
            ("NPT seed authority", &self.npt_seed_authority_sha256),
            (
                "NPT frozen cell authority",
                &self.npt_frozen_cell_authority_sha256,
            ),
            ("environment velocity", &self.environment_velocity_sha256),
            ("full velocity", &self.full_velocity_sha256),
            ("environment position", &self.environment_position_sha256),
            ("full position", &self.full_position_sha256),
        ] {
            validate_sha256(label, digest)?;
        }
        if let Some(digest) = &self.fragment_coordinate_branch_authority_sha256 {
            validate_sha256("fragment coordinate branch authority", digest)?;
        }
        if let Some(digest) = &self.fragment_coordinate_ensemble_sha256 {
            validate_sha256("fragment coordinate ensemble", digest)?;
        }
        if let Some(digest) = &self.fragment_position_f64_checkpoint_sha256 {
            validate_sha256("fragment position f64 checkpoint", digest)?;
        }
        if let Some(digest) = &self.fragment_position_sha256 {
            validate_sha256("fragment position f32 projection", digest)?;
        }
        if (self.fragment_atom_count == 0)
            != self.fragment_coordinate_branch_authority_sha256.is_none()
            || (self.fragment_atom_count == 0) != self.fragment_coordinate_ensemble_sha256.is_none()
            || (self.fragment_atom_count == 0)
                != self.fragment_position_f64_checkpoint_sha256.is_none()
            || (self.fragment_atom_count == 0) != self.fragment_position_sha256.is_none()
        {
            return Err("matched initialization fragment branch/population mismatch".into());
        }
        Ok(())
    }
}

/// One immutable stream phase and its complete physical condition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LambdaRuntimePhaseContext {
    pub protocol_phase: crate::chemical_state_fiber::ProtocolPhaseBinding,
    pub thermodynamic_condition: crate::chemical_state_fiber::ThermodynamicCondition,
}

impl LambdaRuntimePhaseContext {
    fn validate_for_stream(&self, stream_id: u32) -> Result<(), String> {
        if u16::try_from(stream_id).ok() != Some(self.protocol_phase.stream_id)
            || self.protocol_phase.thermodynamic_condition_id
                != self.thermodynamic_condition.condition_id
        {
            return Err("lambda runtime phase schedule stream/condition mismatch".into());
        }
        self.thermodynamic_condition
            .validate()
            .map_err(|error| error.to_string())?;
        self.protocol_phase
            .validate_for(crate::chemical_state_fiber::GhostLineageKey {
                stream_id: self.protocol_phase.stream_id,
                frame_idx: 0,
                step_idx: self.protocol_phase.start_step,
                site_id: 0,
                record_index: 0,
            })
            .map_err(|error| error.to_string())
    }
}

/// Fail-closed admission for the current per-event packet/receipt/segment ABI.
/// This is an explicit bounded-capacity alternative to pretending that an
/// unbounded campaign can safely create four durable files per event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LambdaTypedEventCapacityContract {
    pub schema_kind: String,
    pub schema_version: u32,
    /// Hard maximum number of typed events for this stream.
    pub maximum_event_count: u64,
    /// Hard maximum total bytes emitted by packet, receipt, record and manifest.
    pub maximum_total_bytes: u64,
    /// Hard maximum durable bytes for one event's four-file transaction.
    pub maximum_event_bytes: u64,
    /// Pre-reserved inode budget for four durable files per event plus margin.
    pub reserved_inode_count: u64,
    /// Must be four for schema v1.
    pub durable_files_per_event: u32,
    /// Multiplicative free-space/inode safety margin, at least 1.25.
    pub safety_margin: f64,
    pub contract_sha256: String,
}

impl LambdaTypedEventCapacityContract {
    fn recompute_sha256(&self) -> Result<String, String> {
        let mut projection = self.clone();
        projection.contract_sha256.clear();
        let bytes = serde_json::to_vec(&projection).map_err(|error| error.to_string())?;
        let mut digest = Sha256::new();
        digest.update(b"PRISM_LAMBDA_TYPED_EVENT_CAPACITY_V1\0");
        digest.update(bytes);
        Ok(format!("{:x}", digest.finalize()))
    }

    fn validate(&self) -> Result<(), String> {
        let minimum_inodes = self
            .maximum_event_count
            .checked_mul(u64::from(self.durable_files_per_event))
            .and_then(|value| value.checked_add(16))
            .ok_or_else(|| "lambda typed-event inode budget overflow".to_string())?;
        let minimum_total_bytes = self
            .maximum_event_count
            .checked_mul(self.maximum_event_bytes)
            .ok_or_else(|| "lambda typed-event byte budget overflow".to_string())?;
        if self.schema_kind != "prismos_lambda_typed_event_capacity_contract"
            || self.schema_version != 1
            || self.maximum_event_count == 0
            || self.maximum_total_bytes == 0
            || self.maximum_event_bytes == 0
            || self.maximum_total_bytes < minimum_total_bytes
            || self.durable_files_per_event != 4
            || !self.safety_margin.is_finite()
            || self.safety_margin < 1.25
            || self.reserved_inode_count < minimum_inodes
            || self.contract_sha256 != self.recompute_sha256()?
        {
            return Err("invalid lambda typed-event capacity contract".into());
        }
        Ok(())
    }
}

/// Content-addressed authority for the exact selected executable, topology,
/// fused PTX and promoted runtime capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LambdaRuntimeCapabilityReceipt {
    pub schema_kind: String,
    pub schema_version: u32,
    pub selected_engine_path: PathBuf,
    pub selected_fused_ptx_path: PathBuf,
    pub topology_path: PathBuf,
    /// Separately persisted live self-test; package-time source/test paths are
    /// inadmissible capability evidence.
    pub runtime_self_test_receipt_path: PathBuf,
    /// Raw file SHA-256 of `runtime_self_test_receipt_path`.
    pub runtime_self_test_receipt_file_sha256: String,
    pub capabilities: LambdaRuntimeCapabilities,
    pub receipt_sha256: String,
}

impl LambdaRuntimeCapabilityReceipt {
    fn recompute_sha256(&self) -> Result<String, String> {
        let mut projection = self.clone();
        projection.receipt_sha256.clear();
        let bytes = serde_json::to_vec(&projection).map_err(|error| error.to_string())?;
        let mut digest = Sha256::new();
        digest.update(b"PRISM_LAMBDA_RUNTIME_CAPABILITY_RECEIPT_V1\0");
        digest.update(bytes);
        Ok(format!("{:x}", digest.finalize()))
    }

    fn read_live_regular_file(path: &Path, label: &str) -> Result<Vec<u8>, String> {
        if !path.is_absolute() {
            return Err(format!("{label} path is not absolute"));
        }
        let normalized = path
            .canonicalize()
            .map_err(|error| format!("canonicalize live {label} {}: {error}", path.display()))?;
        if normalized != path {
            return Err(format!("live {label} path is not canonical and normalized"));
        }
        let mut options = OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        options.custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
        let mut file = options.open(path).map_err(|error| {
            format!(
                "open live {label} {} without links: {error}",
                path.display()
            )
        })?;
        let metadata = file
            .metadata()
            .map_err(|error| format!("fstat live {label} {}: {error}", path.display()))?;
        if !metadata.is_file() {
            return Err(format!("live {label} is not a regular file"));
        }
        let mut bytes = Vec::with_capacity(usize::try_from(metadata.len()).unwrap_or(0));
        file.read_to_end(&mut bytes)
            .map_err(|error| format!("read live {label} {}: {error}", path.display()))?;
        Ok(bytes)
    }

    pub(crate) fn hash_live_regular_file(path: &Path, label: &str) -> Result<String, String> {
        Ok(format!(
            "{:x}",
            Sha256::digest(Self::read_live_regular_file(path, label)?)
        ))
    }

    /// Load and validate the separately persisted selected-candidate self-test
    /// using no-follow regular-file semantics.
    pub fn load_live_self_test(&self) -> Result<LambdaRuntimeSelfTestReceipt, String> {
        let bytes = Self::read_live_regular_file(
            &self.runtime_self_test_receipt_path,
            "lambda runtime self-test receipt",
        )?;
        let file_sha = format!("{:x}", Sha256::digest(&bytes));
        if file_sha != self.runtime_self_test_receipt_file_sha256 {
            return Err("lambda runtime self-test receipt file hash mismatch".into());
        }
        let receipt: LambdaRuntimeSelfTestReceipt =
            serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
        receipt.validate()?;
        if receipt.selected_engine_path != self.selected_engine_path
            || receipt.selected_fused_ptx_path != self.selected_fused_ptx_path
            || receipt.topology_path != self.topology_path
            || self
                .capabilities
                .runtime_self_test_receipt_sha256
                .as_deref()
                != Some(receipt.receipt_sha256.as_str())
        {
            return Err("lambda capability and self-test receipt identities differ".into());
        }
        Ok(receipt)
    }

    /// Rehash all selected artifacts at the launch boundary and bind them to
    /// the actual current executable. A signed claim or source-presence flag
    /// cannot satisfy this check.
    pub fn validate_live(&self) -> Result<(), String> {
        if self.schema_kind != "prismos_lambda_runtime_capability_receipt"
            || self.schema_version != 1
            || self.receipt_sha256 != self.recompute_sha256()?
        {
            return Err("invalid lambda runtime capability receipt".into());
        }
        let engine_sha = Self::hash_live_regular_file(&self.selected_engine_path, "engine")?;
        let ptx_sha = Self::hash_live_regular_file(&self.selected_fused_ptx_path, "fused PTX")?;
        let topology_sha = Self::hash_live_regular_file(&self.topology_path, "topology")?;
        let self_test = self.load_live_self_test()?;
        let current_exe = std::env::current_exe()
            .map_err(|error| format!("resolve current lambda executable: {error}"))?;
        let current_exe_sha = Self::hash_live_regular_file(&current_exe, "current executable")?;
        if current_exe_sha != engine_sha
            || self.capabilities.selected_engine_sha256.as_deref() != Some(engine_sha.as_str())
            || self.capabilities.selected_fused_ptx_sha256.as_deref() != Some(ptx_sha.as_str())
            || self.capabilities.topology_sha256.as_deref() != Some(topology_sha.as_str())
            || self_test.engine_sha256 != engine_sha
            || self_test.fused_ptx_sha256 != ptx_sha
            || self_test.topology_sha256 != topology_sha
        {
            return Err(
                "live engine/topology/PTX hashes differ from promoted lambda capability tuple"
                    .into(),
            );
        }
        Ok(())
    }
}

/// One content-addressed launch request containing every exact native stream
/// binding. The executable selects by stream and topology hash; it never
/// synthesizes run, phase, condition or output identities.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LambdaRuntimeAdmissionRequest {
    pub schema_kind: String,
    pub schema_version: u32,
    pub request_id: String,
    pub stream_bindings: Vec<LambdaRuntimeBinding>,
    pub request_sha256: String,
}

impl LambdaRuntimeAdmissionRequest {
    fn recompute_sha256(&self) -> Result<String, String> {
        let mut projection = self.clone();
        projection.request_sha256.clear();
        let bytes = serde_json::to_vec(&projection).map_err(|error| error.to_string())?;
        let mut digest = Sha256::new();
        digest.update(b"PRISM_LAMBDA_RUNTIME_ADMISSION_REQUEST_V1\0");
        digest.update(bytes);
        Ok(format!("{:x}", digest.finalize()))
    }

    pub fn validate_live(&self) -> Result<(), String> {
        if self.schema_kind != "prismos_lambda_runtime_admission_request"
            || self.schema_version != 1
            || self.request_id.trim().is_empty()
            || self.stream_bindings.is_empty()
            || self.request_sha256 != self.recompute_sha256()?
        {
            return Err("invalid lambda runtime admission request".into());
        }
        let mut identities = BTreeSet::new();
        for binding in &self.stream_bindings {
            binding.validate_pre_topology()?;
            let receipt = binding
                .runtime_capability_receipt
                .as_ref()
                .ok_or_else(|| "lambda launch binding lacks live capability receipt".to_string())?;
            receipt.validate_live()?;
            if binding.engine_sha256
                != receipt
                    .capabilities
                    .selected_engine_sha256
                    .as_deref()
                    .unwrap()
                || binding.topology_sha256
                    != receipt.capabilities.topology_sha256.as_deref().unwrap()
                || !identities.insert((binding.stream_id, binding.topology_sha256.clone()))
            {
                return Err(
                    "lambda launch binding aliases or differs from capability receipt".into(),
                );
            }
        }
        Ok(())
    }

    pub fn binding_for(
        &self,
        stream_id: u32,
        topology_sha256: &str,
    ) -> Result<LambdaRuntimeBinding, String> {
        let matches: Vec<_> = self
            .stream_bindings
            .iter()
            .filter(|binding| {
                binding.stream_id == stream_id && binding.topology_sha256 == topology_sha256
            })
            .collect();
        if matches.len() != 1 {
            return Err(format!(
                "lambda admission has {} bindings for stream {stream_id} topology {topology_sha256}",
                matches.len()
            ));
        }
        Ok(matches[0].clone())
    }

    pub fn selected_fused_ptx_path(&self) -> Result<&Path, String> {
        let mut paths = self.stream_bindings.iter().map(|binding| {
            binding
                .runtime_capability_receipt
                .as_ref()
                .map(|receipt| receipt.selected_fused_ptx_path.as_path())
                .ok_or_else(|| "lambda binding lacks capability receipt".to_string())
        });
        let first = paths
            .next()
            .ok_or_else(|| "lambda admission has no stream bindings".to_string())??;
        for path in paths {
            if path? != first {
                return Err("lambda admission bindings select different fused PTX files".into());
            }
        }
        Ok(first)
    }
}

pub fn load_lambda_runtime_admission_request(
    path: impl AsRef<Path>,
) -> Result<LambdaRuntimeAdmissionRequest, String> {
    let path = path.as_ref();
    if !path.is_absolute() || path.canonicalize().map_err(|error| error.to_string())? != path {
        return Err("lambda runtime admission path must be absolute and canonical".into());
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
    let mut file = options.open(path).map_err(|error| error.to_string())?;
    if !file
        .metadata()
        .map_err(|error| error.to_string())?
        .is_file()
    {
        return Err("lambda runtime admission is not a regular file".into());
    }
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    let request: LambdaRuntimeAdmissionRequest =
        serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
    request.validate_live()?;
    Ok(request)
}

fn validate_lambda_output_paths(run_root: &Path, output_dir: &Path) -> Result<(), String> {
    if !run_root.is_absolute() || !output_dir.is_absolute() || run_root == output_dir {
        return Err("lambda output root/path must be distinct absolute paths".into());
    }
    let canonical_root = run_root
        .canonicalize()
        .map_err(|error| format!("canonicalize lambda run output root: {error}"))?;
    if canonical_root != run_root {
        return Err("lambda run output root is not canonical".into());
    }
    let root_metadata = std::fs::symlink_metadata(run_root).map_err(|error| error.to_string())?;
    if root_metadata.file_type().is_symlink() || !root_metadata.is_dir() {
        return Err("lambda run output root is not a non-symlink directory".into());
    }
    let relative = output_dir
        .strip_prefix(run_root)
        .map_err(|_| "lambda typed-event output escapes admitted run root".to_string())?;
    if relative.as_os_str().is_empty()
        || !relative
            .components()
            .all(|component| matches!(component, std::path::Component::Normal(_)))
    {
        return Err("lambda typed-event output contains a traversal component".into());
    }
    let mut current = run_root.to_path_buf();
    for component in relative.components() {
        current.push(component.as_os_str());
        match std::fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => {
                return Err(format!(
                    "lambda typed-event output component is not a non-symlink directory: {}",
                    current.display()
                ))
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => break,
            Err(error) => return Err(error.to_string()),
        }
    }
    Ok(())
}

#[cfg(unix)]
fn secure_create_lambda_output_dir(run_root: &Path, output_dir: &Path) -> Result<(), String> {
    validate_lambda_output_paths(run_root, output_dir)?;
    let relative = output_dir
        .strip_prefix(run_root)
        .map_err(|_| "lambda typed-event output escapes run root".to_string())?;
    let mut current = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_DIRECTORY | libc::O_NOFOLLOW)
        .open(run_root)
        .map_err(|error| format!("open admitted lambda run root: {error}"))?;
    for component in relative.components() {
        let name = std::ffi::CString::new(component.as_os_str().as_bytes())
            .map_err(|_| "lambda output component contains NUL".to_string())?;
        let mkdir_rc = unsafe { libc::mkdirat(current.as_raw_fd(), name.as_ptr(), 0o750) };
        if mkdir_rc != 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() != std::io::ErrorKind::AlreadyExists {
                return Err(format!("create lambda output component: {error}"));
            }
        }
        let fd = unsafe {
            libc::openat(
                current.as_raw_fd(),
                name.as_ptr(),
                libc::O_RDONLY | libc::O_CLOEXEC | libc::O_DIRECTORY | libc::O_NOFOLLOW,
            )
        };
        if fd < 0 {
            return Err(format!(
                "open lambda output component without links: {}",
                std::io::Error::last_os_error()
            ));
        }
        current = unsafe { File::from_raw_fd(fd) };
    }
    current.sync_all().map_err(|error| error.to_string())?;
    let canonical_output = output_dir
        .canonicalize()
        .map_err(|error| format!("canonicalize created lambda output: {error}"))?;
    if canonical_output != output_dir || !canonical_output.starts_with(run_root) {
        return Err("created lambda typed-event output changed identity".into());
    }
    Ok(())
}

#[cfg(not(unix))]
fn secure_create_lambda_output_dir(run_root: &Path, output_dir: &Path) -> Result<(), String> {
    validate_lambda_output_paths(run_root, output_dir)?;
    std::fs::create_dir_all(output_dir).map_err(|error| error.to_string())?;
    if output_dir
        .canonicalize()
        .map_err(|error| error.to_string())?
        != output_dir
    {
        return Err("created lambda output is not canonical".into());
    }
    Ok(())
}

#[cfg(unix)]
fn lambda_output_capacity(output_dir: &Path) -> Result<(u64, u64), String> {
    let path = std::ffi::CString::new(output_dir.as_os_str().as_bytes())
        .map_err(|_| "lambda output path contains NUL".to_string())?;
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(path.as_ptr(), &mut stat) } != 0 {
        return Err(format!(
            "statvfs lambda output: {}",
            std::io::Error::last_os_error()
        ));
    }
    Ok((
        (stat.f_bavail as u64).saturating_mul(stat.f_frsize as u64),
        stat.f_favail as u64,
    ))
}

#[cfg(not(unix))]
fn lambda_output_capacity(_output_dir: &Path) -> Result<(u64, u64), String> {
    Err("lambda runtime capacity admission requires statvfs support".into())
}

fn enforce_lambda_output_capacity(
    output_dir: &Path,
    contract: &LambdaTypedEventCapacityContract,
) -> Result<(), String> {
    contract.validate()?;
    let (available_bytes, available_inodes) = lambda_output_capacity(output_dir)?;
    let required_bytes = (contract.maximum_total_bytes as f64 * contract.safety_margin).ceil();
    let required_inodes = (contract.reserved_inode_count as f64 * contract.safety_margin).ceil();
    if required_bytes > u64::MAX as f64
        || required_inodes > u64::MAX as f64
        || available_bytes < required_bytes as u64
        || available_inodes < required_inodes as u64
    {
        return Err(format!(
            "lambda whole-campaign capacity admission failed: bytes {available_bytes}/{} inodes {available_inodes}/{}",
            required_bytes as u64, required_inodes as u64
        ));
    }
    Ok(())
}

/// Receipt returned by the resident CUDA-graph propagation owner for one
/// NCMC propagation window.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NcmcResidentWindowReceipt {
    /// Number of physical MD steps executed by this window.
    pub logical_steps: u64,
    /// Host graph launches used to execute the window.
    pub host_graph_launches: u64,
    /// Must be zero: no legacy per-step replay dispatch is admissible.
    pub legacy_graph_replay_dispatch_count: u64,
    /// True only for a resident graph or device conditional loop.
    pub resident_execution: bool,
    /// Hash-closed deterministic capture schedule/kernel/allocation authority.
    pub capture_identity_sha256: String,
    /// Separate live graph topology witness; not sufficient as identity alone.
    pub graph_structure_sha256: String,
    /// True only for the gamma-zero, intervention-disabled NCMC graph.
    pub deterministic_reversible_propagation: bool,
    /// Hash binding the captured graph executable and mutable state buffers.
    pub graph_exec_sha256: String,
}

impl NcmcResidentWindowReceipt {
    /// Enforce the production graph-residency contract.
    pub fn validate(&self, expected_logical_steps: u64) -> Result<(), String> {
        if !self.resident_execution
            || !self.deterministic_reversible_propagation
            || self.logical_steps != expected_logical_steps
            || self.host_graph_launches == 0
            || self.legacy_graph_replay_dispatch_count != 0
        {
            return Err("NCMC propagation did not satisfy resident graph execution".into());
        }
        validate_sha256("NCMC capture identity", &self.capture_identity_sha256)?;
        validate_sha256("NCMC graph structure", &self.graph_structure_sha256)?;
        validate_sha256("NCMC graph executable", &self.graph_exec_sha256)
    }
}

/// One engine-materialized entry in the exact ordered NCMC work path.  The
/// trace deliberately retains energies rather than accepting a caller's
/// pre-reduced scalar: perturbation and shadow work are recomputed during
/// validation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NcmcOperationBoundaryLineage {
    pub before: NcmcBoundaryPopulationLineage,
    pub after: NcmcBoundaryPopulationLineage,
}

impl NcmcOperationBoundaryLineage {
    pub fn validate(
        &self,
        move_id: u64,
        protocol_direction: NcmcProtocolDirection,
        operation_index: u32,
        before_kind: NcmcForceBoundaryKind,
        after_kind: NcmcForceBoundaryKind,
    ) -> Result<(), String> {
        let expected_before = NcmcForceBoundaryIdentity::new(
            move_id,
            protocol_direction,
            operation_index,
            before_kind,
        )?;
        let expected_after = NcmcForceBoundaryIdentity::new(
            move_id,
            protocol_direction,
            operation_index,
            after_kind,
        )?;
        self.before.validate()?;
        self.after.validate()?;
        if self.before.boundary != expected_before || self.after.boundary != expected_after {
            return Err("NCMC operation boundary lineage is incomplete or misbound".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(
    tag = "entry_kind",
    rename_all = "SCREAMING_SNAKE_CASE",
    deny_unknown_fields
)]
pub enum NcmcWorkTraceEntry {
    Perturb {
        operation_index: u32,
        boundary_lineage: NcmcOperationBoundaryLineage,
        lambda_from: f64,
        lambda_to: f64,
        potential_before_kcal_mol: f64,
        potential_after_kcal_mol: f64,
        reduced_work: f64,
    },
    Propagate {
        operation_index: u32,
        boundary_lineage: NcmcOperationBoundaryLineage,
        lambda: f64,
        integration_steps: u32,
        hamiltonian_before_kcal_mol: f64,
        hamiltonian_after_kcal_mol: f64,
        constraint_impulse_work_kcal_mol: f64,
        reduced_shadow_work: f64,
        resident_window: NcmcResidentWindowReceipt,
    },
    MomentumReverse {
        operation_index: u32,
        boundary_lineage: NcmcOperationBoundaryLineage,
        kinetic_before_kcal_mol: f64,
        kinetic_after_kcal_mol: f64,
        exact_involution_applied: bool,
    },
}

/// Hash-closed work trace consumed by NCMC adjudication. The trace and the
/// admitted protocol must agree operation-for-operation; neither missing nor
/// extra resident propagation is permitted.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NcmcOrderedWorkTrace {
    pub schema_kind: String,
    pub schema_version: u32,
    pub move_id: u64,
    pub schedule_sha256: String,
    pub protocol_direction: NcmcProtocolDirection,
    pub beta_mol_per_kcal: f64,
    pub entries: Vec<NcmcWorkTraceEntry>,
    pub perturbation_work_reduced: f64,
    pub shadow_work_reduced: f64,
    pub initial_endpoint_equilibrium_density_evaluated: bool,
    pub proposed_endpoint_equilibrium_density_evaluated: bool,
    pub trace_sha256: String,
}

impl NcmcOrderedWorkTrace {
    fn recompute_sha256(&self) -> Result<String, String> {
        let mut projection = self.clone();
        projection.trace_sha256.clear();
        let bytes = serde_json::to_vec(&projection).map_err(|error| error.to_string())?;
        let mut digest = Sha256::new();
        digest.update(b"PRISM_NCMC_ORDERED_WORK_TRACE_V2\0");
        digest.update(bytes);
        Ok(format!("{:x}", digest.finalize()))
    }

    pub fn seal(
        protocol: &NcmcProtocol,
        move_id: u64,
        protocol_direction: NcmcProtocolDirection,
        beta_mol_per_kcal: f64,
        entries: Vec<NcmcWorkTraceEntry>,
        initial_endpoint_equilibrium_density_evaluated: bool,
        proposed_endpoint_equilibrium_density_evaluated: bool,
    ) -> Result<Self, String> {
        let mut perturbation_work_reduced = 0.0;
        let mut shadow_work_reduced = 0.0;
        for entry in &entries {
            match entry {
                NcmcWorkTraceEntry::Perturb { reduced_work, .. } => {
                    perturbation_work_reduced += reduced_work;
                }
                NcmcWorkTraceEntry::Propagate {
                    reduced_shadow_work,
                    ..
                } => shadow_work_reduced += reduced_shadow_work,
                NcmcWorkTraceEntry::MomentumReverse { .. } => {}
            }
        }
        let mut trace = Self {
            schema_kind: "prismos_ncmc_ordered_work_trace".into(),
            schema_version: 2,
            move_id,
            schedule_sha256: protocol.schedule_sha256.clone(),
            protocol_direction,
            beta_mol_per_kcal,
            entries,
            perturbation_work_reduced,
            shadow_work_reduced,
            initial_endpoint_equilibrium_density_evaluated,
            proposed_endpoint_equilibrium_density_evaluated,
            trace_sha256: String::new(),
        };
        trace.trace_sha256 = trace.recompute_sha256()?;
        trace.validate(protocol)?;
        Ok(trace)
    }

    pub fn validate(&self, protocol: &NcmcProtocol) -> Result<(), String> {
        validate_ncmc(protocol)?;
        let realized_operations = match self.protocol_direction {
            NcmcProtocolDirection::Forward => &protocol.forward_operations,
            NcmcProtocolDirection::Reverse => &protocol.reverse_operations,
        };
        if self.schema_kind != "prismos_ncmc_ordered_work_trace"
            || self.schema_version != 2
            || self.schedule_sha256 != protocol.schedule_sha256
            || !self.beta_mol_per_kcal.is_finite()
            || self.beta_mol_per_kcal <= 0.0
            || self.entries.len() != realized_operations.len()
            || !self.initial_endpoint_equilibrium_density_evaluated
            || !self.proposed_endpoint_equilibrium_density_evaluated
        {
            return Err("invalid or incomplete NCMC ordered work trace".into());
        }
        let close = |actual: f64, expected: f64| {
            actual.is_finite()
                && expected.is_finite()
                && (actual - expected).abs() <= 1.0e-10 * (1.0 + expected.abs())
        };
        let mut perturbation_sum = 0.0;
        let mut shadow_sum = 0.0;
        for (index, (entry, operation)) in self.entries.iter().zip(realized_operations).enumerate()
        {
            match (entry, operation) {
                (
                    NcmcWorkTraceEntry::Perturb {
                        operation_index,
                        boundary_lineage,
                        lambda_from,
                        lambda_to,
                        potential_before_kcal_mol,
                        potential_after_kcal_mol,
                        reduced_work,
                    },
                    NcmcProtocolOperation::Perturb {
                        lambda_from: expected_from,
                        lambda_to: expected_to,
                    },
                ) if *operation_index as usize == index
                    && boundary_lineage
                        .validate(
                            self.move_id,
                            self.protocol_direction,
                            *operation_index,
                            NcmcForceBoundaryKind::PerturbPre,
                            NcmcForceBoundaryKind::PerturbPost,
                        )
                        .is_ok()
                    && lambda_from.to_bits() == expected_from.to_bits()
                    && lambda_to.to_bits() == expected_to.to_bits()
                    && potential_before_kcal_mol.is_finite()
                    && potential_after_kcal_mol.is_finite()
                    && close(
                        *reduced_work,
                        self.beta_mol_per_kcal
                            * (potential_after_kcal_mol - potential_before_kcal_mol),
                    ) =>
                {
                    perturbation_sum += reduced_work
                }
                (
                    NcmcWorkTraceEntry::Propagate {
                        operation_index,
                        boundary_lineage,
                        lambda,
                        integration_steps,
                        hamiltonian_before_kcal_mol,
                        hamiltonian_after_kcal_mol,
                        constraint_impulse_work_kcal_mol,
                        reduced_shadow_work,
                        resident_window,
                    },
                    NcmcProtocolOperation::Propagate {
                        lambda: expected_lambda,
                        integration_steps: expected_steps,
                    },
                ) if *operation_index as usize == index
                    && boundary_lineage
                        .validate(
                            self.move_id,
                            self.protocol_direction,
                            *operation_index,
                            NcmcForceBoundaryKind::PropagatePre,
                            NcmcForceBoundaryKind::PropagatePost,
                        )
                        .is_ok()
                    && lambda.to_bits() == expected_lambda.to_bits()
                    && integration_steps == expected_steps
                    && hamiltonian_before_kcal_mol.is_finite()
                    && hamiltonian_after_kcal_mol.is_finite()
                    && constraint_impulse_work_kcal_mol.is_finite()
                    && close(
                        *reduced_shadow_work,
                        self.beta_mol_per_kcal
                            * (hamiltonian_after_kcal_mol - hamiltonian_before_kcal_mol),
                    ) =>
                {
                    resident_window.validate(u64::from(*integration_steps))?;
                    shadow_sum += reduced_shadow_work;
                }
                (
                    NcmcWorkTraceEntry::MomentumReverse {
                        operation_index,
                        boundary_lineage,
                        kinetic_before_kcal_mol,
                        kinetic_after_kcal_mol,
                        exact_involution_applied,
                    },
                    NcmcProtocolOperation::MomentumReverse,
                ) if *operation_index as usize == index
                    && boundary_lineage
                        .validate(
                            self.move_id,
                            self.protocol_direction,
                            *operation_index,
                            NcmcForceBoundaryKind::MomentumReversePre,
                            NcmcForceBoundaryKind::MomentumReversePost,
                        )
                        .is_ok()
                    && *exact_involution_applied
                    && close(*kinetic_after_kcal_mol, *kinetic_before_kcal_mol) => {}
                _ => {
                    return Err(format!(
                        "NCMC work trace entry {index} disagrees with admitted operation"
                    ));
                }
            }
        }
        if !close(self.perturbation_work_reduced, perturbation_sum)
            || !close(self.shadow_work_reduced, shadow_sum)
            || self.trace_sha256 != self.recompute_sha256()?
        {
            return Err("NCMC ordered work totals or digest do not close".into());
        }
        Ok(())
    }
}

/// Engine-owned result retaining the detailed-balance receipt and every
/// resident propagation proof used to construct it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NcmcExecutionResult {
    /// Metropolis-Hastings adjudication.
    pub move_receipt: NcmcMoveReceipt,
    /// One receipt per propagation segment, in execution order.
    pub resident_windows: Vec<NcmcResidentWindowReceipt>,
    /// True only when commit occurred after acceptance.
    pub state_committed: bool,
}

impl LambdaRuntimeBinding {
    /// Validate every immutable binding invariant available before the
    /// topology extension is decoded. Only lane species/slot membership is
    /// intentionally deferred to `validate(Some(system))`.
    pub fn validate_pre_topology(&self) -> Result<(), String> {
        if self.run_id.trim().is_empty() || self.replica_id.trim().is_empty() {
            return Err("lambda runtime run/replica identity is empty".into());
        }
        if u16::try_from(self.stream_id).is_err()
            || self.initial_protocol_phase.stream_id != self.stream_id as u16
            || self.initial_protocol_phase.thermodynamic_condition_id
                != self.initial_thermodynamic_condition.condition_id
        {
            return Err("lambda runtime stream/phase/condition identity mismatch".into());
        }
        validate_sha256("engine binary", &self.engine_sha256)?;
        validate_sha256("input topology", &self.topology_sha256)?;
        if let Some(binding) = &self.endpoint_support_binding_sha256 {
            validate_sha256("endpoint support binding", binding)?;
        }
        if let Some(unlock) = &self.target_cross_cell_unlock_sha256 {
            validate_sha256("target cross-cell unlock", unlock)?;
        }
        if let Some(panel) = &self.panel_scientific_sha256 {
            validate_sha256("panel scientific authority", panel)?;
        }
        self.matched_velocity_initialization.validate()?;
        if self
            .matched_velocity_initialization
            .frozen_checkpoint_replica_seed
            != self.replica_seed
        {
            return Err(
                "runtime replica seed differs from its admitted frozen-NVT checkpoint member"
                    .into(),
            );
        }
        self.initial_thermodynamic_condition
            .validate()
            .map_err(|error| error.to_string())?;
        self.initial_protocol_phase
            .validate_for(crate::chemical_state_fiber::GhostLineageKey {
                stream_id: self.stream_id as u16,
                frame_idx: 0,
                step_idx: self.initial_protocol_phase.start_step,
                site_id: 0,
                record_index: 0,
            })
            .map_err(|error| error.to_string())?;
        if self.protocol_schedule.is_empty() {
            return Err("lambda runtime protocol schedule is empty".into());
        }
        let first = &self.protocol_schedule[0];
        if first.protocol_phase != self.initial_protocol_phase
            || first.thermodynamic_condition != self.initial_thermodynamic_condition
        {
            return Err("lambda runtime initial phase differs from admitted schedule".into());
        }
        let mut prior: Option<&LambdaRuntimePhaseContext> = None;
        for context in &self.protocol_schedule {
            context.validate_for_stream(self.stream_id)?;
            if context.protocol_phase.phase_schedule_digest_sha256
                != self.initial_protocol_phase.phase_schedule_digest_sha256
                || context.protocol_phase.protocol_name != self.initial_protocol_phase.protocol_name
            {
                return Err(
                    "lambda runtime protocol schedule authority diverges across phases".into(),
                );
            }
            if let Some(previous) = prior {
                if context.protocol_phase.start_step != previous.protocol_phase.end_step_exclusive
                    || context.protocol_phase.phase_ordinal
                        != previous.protocol_phase.phase_ordinal.saturating_add(1)
                {
                    return Err(
                        "lambda runtime protocol phases must be contiguous and ordinal".into(),
                    );
                }
            }
            prior = Some(context);
        }
        if self.lane.schema_kind != "prismos_lambda_experiment_lane_contract"
            || self.lane.schema_version != 1
            || self.lane.lane_id.trim().is_empty()
            || self.lane.virtual_probe_count != 0
            || self.lane.contract_sha256 != self.lane.recompute_sha256()?
        {
            return Err("invalid lambda experiment lane contract".into());
        }
        let mut species = self.lane.fragment_species_ids.clone();
        species.sort_unstable();
        species.dedup();
        if species != self.lane.fragment_species_ids
            || (self.lane.lane_kind == LambdaExperimentLaneKind::NoProbe && !species.is_empty())
            || (self.lane.lane_kind != LambdaExperimentLaneKind::NoProbe && species.is_empty())
        {
            return Err("lambda runtime lane species declaration is invalid".into());
        }
        match (
            self.lane.lane_kind,
            &self.ncmc_execution_mode,
            &self.ncmc_scheduler,
            &self.endpoint_support_binding_sha256,
            &self.decoupled_fragment_placement_measure_authority,
            &self.target_cross_cell_unlock_sha256,
            &self.panel_scientific_sha256,
        ) {
            (LambdaExperimentLaneKind::NoProbe, None, None, None, None, None, None) => {}
            (LambdaExperimentLaneKind::NoProbe, _, _, _, _, _, _) => {
                return Err(
                    "no-probe runtime cannot declare NCMC mode/scheduler/endpoint support".into(),
                )
            }
            (_, Some(mode), Some(scheduler), Some(_), Some(authority), target_unlock, Some(_)) => {
                mode.validate()?;
                scheduler.validate()?;
                authority.validate()?;
                match mode {
                    NcmcExecutionMode::CalibrationWorkCollection { .. }
                        if authority.calibration_sampling_admission_allowed
                            && !authority.production_admission_allowed
                            && target_unlock.is_none() => {}
                    NcmcExecutionMode::CalibratedTargetLane
                        if authority.production_admission_allowed && target_unlock.is_some() => {}
                    NcmcExecutionMode::CalibrationWorkCollection { .. } => return Err(
                        "decoupled placement authority is not admitted for calibration sampling"
                            .into(),
                    ),
                    NcmcExecutionMode::CalibratedTargetLane => {
                        return Err(
                            "precalibration decoupled branch cannot admit a production target lane"
                                .into(),
                        )
                    }
                }
            }
            (_, _, _, _, _, _, _) => {
                return Err(
                    "fragment runtime lacks NCMC mode/scheduler/endpoint-support authority".into(),
                )
            }
        }
        validate_lambda_output_paths(&self.run_output_root, &self.typed_event_output_dir)?;
        self.typed_event_capacity.validate()?;
        self.phase_protocol_sha256()?;
        Ok(())
    }

    /// Validate binding lineage before the runtime-inline emitter is created.
    pub fn validate(&self, system: Option<&LambdaFragmentSystem>) -> Result<(), String> {
        self.validate_pre_topology()?;
        self.lane.validate(system)?;
        match (system, &self.ncmc_scheduler) {
            (Some(system), Some(scheduler))
                if scheduler.ncmc_protocol_schedule_sha256 == system.ncmc.schedule_sha256 =>
            {
                let expected_endpoint_support =
                    lambda_endpoint_support_binding_sha256(system, scheduler)?;
                if self.endpoint_support_binding_sha256.as_deref()
                    != Some(expected_endpoint_support.as_str())
                {
                    return Err(
                        "runtime endpoint-support binding differs from native prepared A/B and scheduler projection"
                            .into(),
                    );
                }
                match &scheduler.placement_measure {
                    NcmcPlacementMeasureAuthority::PersistentDecoupledCoordinateEnsemble {
                        coordinate_ensemble_sha256,
                        mixing_evidence_sha256,
                        ..
                    } => {
                        let authority = self
                            .decoupled_fragment_placement_measure_authority
                            .as_ref()
                            .ok_or_else(|| {
                                "persistent-decoupled runtime lacks placement authority".to_string()
                            })?;
                        authority.validate()?;
                        let species = system
                            .species
                            .iter()
                            .find(|species| species.species_id == authority.species_id)
                            .ok_or_else(|| {
                                "placement authority species is absent from topology".to_string()
                            })?;
                        let state = species
                            .chemical_states
                            .iter()
                            .find(|state| state.state_id == authority.chemical_state_id)
                            .ok_or_else(|| {
                                "placement authority chemical state is absent from topology"
                                    .to_string()
                            })?;
                        let expected_placement_projection =
                            ncmc_placement_projection_sha256(&scheduler.placement_measure)?;
                        if matches!(
                            self.ncmc_execution_mode.as_ref(),
                            Some(NcmcExecutionMode::CalibratedTargetLane)
                        ) {
                            let expected_mu =
                                lambda_chemical_potential_evidence_sha256(system)?;
                            let expected_finite_size =
                                lambda_finite_size_admission_sha256(system)?;
                            let expected_seed_member = lambda_seed_topology_member_sha256(
                                &self.topology_sha256,
                                &self
                                    .matched_velocity_initialization
                                    .npt_frozen_cell_authority_sha256,
                                &self
                                    .matched_velocity_initialization
                                    .periodic_cell_contract_sha256,
                                self.replica_seed,
                            )?;
                            if authority.chemical_potential_evidence_sha256.as_deref()
                                != Some(expected_mu.as_str())
                                || authority.finite_size_admission_sha256.as_deref()
                                    != Some(expected_finite_size.as_str())
                                || authority.seed_topology_member_sha256.as_deref()
                                    != Some(expected_seed_member.as_str())
                                || authority.cross_cell_unlock_sha256.as_deref()
                                    != self.target_cross_cell_unlock_sha256.as_deref()
                                || authority.panel_scientific_sha256.as_deref()
                                    != self.panel_scientific_sha256.as_deref()
                                || authority.target_protocol_authority_sha256.as_deref()
                                    != Some(scheduler.contract_sha256.as_str())
                            {
                                return Err(
                                    "target placement promotion differs from admitted mu, finite-size, seed, panel, or protocol authority"
                                        .into(),
                                );
                            }
                        }
                        if authority.coordinate_ensemble_sha256 != *coordinate_ensemble_sha256
                            || authority.mixing_evidence_sha256 != *mixing_evidence_sha256
                            || authority.source_branch_authority_sha256
                                != self
                                    .matched_velocity_initialization
                                    .fragment_coordinate_branch_authority_sha256
                                    .as_deref()
                                    .unwrap_or_default()
                            || authority.ncmc_placement_projection_sha256
                                != expected_placement_projection
                            || authority.coordinate_ensemble_sha256
                                != self
                                    .matched_velocity_initialization
                                    .fragment_coordinate_ensemble_sha256
                                    .as_deref()
                                    .unwrap_or_default()
                            || authority.periodic_cell_sha256
                                != self
                                    .matched_velocity_initialization
                                    .periodic_cell_contract_sha256
                            || authority.slot_count as usize != system.slots.len()
                            || authority.parameter_template_sha256
                                != species.parameter_template_sha256
                            || authority.endpoint_parameter_sha256
                                != state.parameter_digest_sha256
                            || authority.temperature_k.to_bits()
                                != self.initial_thermodynamic_condition.temperature_k.to_bits()
                        {
                            return Err(
                                "decoupled placement authority differs from scheduler, matched suffix, topology, or thermodynamic condition"
                                    .into(),
                            );
                        }
                    }
                    NcmcPlacementMeasureAuthority::UniformAccessibleVolume { .. } => {
                        return Err(
                            "native persistent-slot runtime does not implement uniform-volume insertion coordinates"
                                .into(),
                        )
                    }
                }
                Ok(())
            }
            (None, None)
                if self.lane.lane_kind == LambdaExperimentLaneKind::NoProbe
                    && self
                        .matched_velocity_initialization
                        .fragment_coordinate_branch_authority_sha256
                        .is_none()
                    && self
                        .matched_velocity_initialization
                        .fragment_coordinate_ensemble_sha256
                        .is_none()
                    && self
                        .decoupled_fragment_placement_measure_authority
                        .is_none()
                    && self.target_cross_cell_unlock_sha256.is_none()
                    && self.panel_scientific_sha256.is_none() =>
            {
                Ok(())
            }
            _ => Err("lambda runtime scheduler does not bind the topology NCMC protocol".into()),
        }
    }

    /// Canonical phase/condition digest shared with chemical-state fibers.
    pub fn phase_protocol_sha256(&self) -> Result<String, String> {
        crate::chemical_state_fiber::phase_protocol_binding_sha256(
            &self.initial_protocol_phase,
            &self.initial_thermodynamic_condition,
        )
        .map_err(|error| error.to_string())
    }

    /// Exact admitted phase for a runtime integration step. Gaps and out-of-
    /// schedule steps are hard failures rather than inferred labels.
    pub fn phase_context_for_step(&self, step: u64) -> Result<&LambdaRuntimePhaseContext, String> {
        let matches: Vec<_> = self
            .protocol_schedule
            .iter()
            .filter(|context| {
                step >= context.protocol_phase.start_step
                    && step < context.protocol_phase.end_step_exclusive
            })
            .collect();
        if matches.len() != 1 {
            return Err(format!(
                "lambda runtime step {step} has {} admitted protocol phases",
                matches.len()
            ));
        }
        Ok(matches[0])
    }
}

/// Energy/force/derivative result from one pair evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SoftcorePairEvaluation {
    /// Total potential in kcal/mol.
    pub potential_kcal_mol: f64,
    /// LJ contribution.
    pub lj_kcal_mol: f64,
    /// Coulomb contribution.
    pub coulomb_kcal_mol: f64,
    /// Force on atom i for displacement `r_i-r_j`, kcal/mol/Angstrom.
    pub force_on_i_kcal_mol_a: [f64; 3],
    /// Analytic derivative of total potential with respect to lambda.
    pub du_dlambda_kcal_mol: f64,
}

/// One endpoint's exact parameters for a fragment-environment pair or an
/// intrafragment exception. `present=false` is a constant-topology dummy and
/// therefore requires zero charge product and epsilon at admission.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FragmentPairEndpointParameters {
    pub present: bool,
    pub charge_product_e2: f64,
    pub sigma_a: f64,
    pub epsilon_kcal_mol: f64,
}

/// Energy-consistent evaluation of an occupied state-A/state-B pair path.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DualEndpointPairEvaluation {
    pub potential_kcal_mol: f64,
    pub endpoint_a_potential_kcal_mol: f64,
    pub endpoint_b_potential_kcal_mol: f64,
    pub force_on_i_kcal_mol_a: [f64; 3],
    pub du_dlambda_kcal_mol: f64,
}

/// Device-reduced slot observable sampled at a controlled NCMC boundary.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LambdaSlotObservable {
    /// Runtime slot index (stable `slot_id` is obtained from the topology).
    pub runtime_slot_index: u32,
    /// Slot-owned alchemical interaction energy in kcal/mol.
    pub potential_kcal_mol: f64,
    /// Analytic derivative in kcal/mol with respect to this slot lambda.
    pub du_dlambda_kcal_mol: f64,
}

/// Boundary telemetry for the complete PME composition. This deliberately
/// binds the fused post-correction observables rather than exposing the PME
/// object's last isolated scratch pass as if it were the physical result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LambdaPmeCompositionLineage {
    pub schema_kind: String,
    pub schema_version: u32,
    pub observable_source: String,
    pub combined_reciprocal_passes: u32,
    pub isolated_basis_passes: u32,
    pub per_slot_source_planes: Vec<Vec<i32>>,
    pub endpoint_occupancy_masks: Vec<i32>,
    pub last_isolated_scratch_plane_authoritative: bool,
    pub total_potential_f64_bits: u64,
    pub slot_energy_f64_bits: Vec<u64>,
    pub slot_dudlambda_f64_bits: Vec<u64>,
    pub composition_sha256: String,
}

/// Complete controlled-boundary observable used for exact perturbation and
/// reversible-integrator shadow-work bookkeeping.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LambdaBoundaryObservation {
    /// Device-reduced per-slot lambda observables.
    pub slots: Vec<LambdaSlotObservable>,
    /// Complete potential energy from the same force launch, kcal/mol.
    pub total_potential_kcal_mol: f64,
    /// Complete kinetic energy from the same device reduction, kcal/mol.
    pub total_kinetic_kcal_mol: f64,
    /// Exact post-composition PME lineage. `None` is valid only for a lane
    /// whose selected Hamiltonian has no lambda PME subsystem.
    pub pme_composition_lineage: Option<LambdaPmeCompositionLineage>,
    /// Ordered typed-emission receipt digests for every slot force event
    /// materialized at this exact boundary.  These are carried into the NCMC
    /// work trace so none of the intermediate U/dUdl science is detached from
    /// its byte lineage.
    pub force_population_lineage: Option<NcmcBoundaryPopulationLineage>,
}

/// Position of one exact force evaluation within a scheduled NCMC move.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum NcmcForceBoundaryKind {
    PerturbPre,
    PerturbPost,
    PropagatePre,
    PropagatePost,
    MomentumReversePre,
    MomentumReversePost,
}

/// Collision-free identity for the repeated force evaluations required by a
/// single NCMC opportunity.  `(stream, step, slot)` alone is intentionally not
/// unique because every perturbation/propagation needs before and after U/dUdl.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NcmcForceBoundaryIdentity {
    pub move_id: u64,
    pub protocol_direction: NcmcProtocolDirection,
    pub operation_index: u32,
    pub boundary_ordinal: u32,
    pub boundary_kind: NcmcForceBoundaryKind,
}

impl NcmcForceBoundaryIdentity {
    pub fn new(
        move_id: u64,
        protocol_direction: NcmcProtocolDirection,
        operation_index: u32,
        boundary_kind: NcmcForceBoundaryKind,
    ) -> Result<Self, String> {
        let boundary_ordinal = operation_index
            .checked_mul(2)
            .and_then(|value| {
                value.checked_add(match boundary_kind {
                    NcmcForceBoundaryKind::PerturbPre
                    | NcmcForceBoundaryKind::PropagatePre
                    | NcmcForceBoundaryKind::MomentumReversePre => 0,
                    NcmcForceBoundaryKind::PerturbPost
                    | NcmcForceBoundaryKind::PropagatePost
                    | NcmcForceBoundaryKind::MomentumReversePost => 1,
                })
            })
            .ok_or_else(|| "NCMC force-boundary ordinal overflow".to_string())?;
        Ok(Self {
            move_id,
            protocol_direction,
            operation_index,
            boundary_ordinal,
            boundary_kind,
        })
    }

    pub fn validate(&self) -> Result<(), String> {
        if *self
            != Self::new(
                self.move_id,
                self.protocol_direction,
                self.operation_index,
                self.boundary_kind,
            )?
        {
            return Err("NCMC force-boundary identity is not canonical".into());
        }
        Ok(())
    }
}

/// One slot member of a complete controlled-boundary force population.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NcmcBoundarySlotEventLineage {
    pub slot_id: u32,
    pub event_sequence: u64,
    pub raw_event_sha256: String,
    pub emission_receipt_sha256: String,
}

/// Ordered population proof for all slot force events produced by one force
/// evaluation.  The digest closes the exact slot population and byte lineage,
/// rather than merely binding the last event in the journal chain.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NcmcBoundaryPopulationLineage {
    pub schema_kind: String,
    pub schema_version: u32,
    pub boundary: NcmcForceBoundaryIdentity,
    pub slot_events: Vec<NcmcBoundarySlotEventLineage>,
    pub population_sha256: String,
}

impl NcmcBoundaryPopulationLineage {
    fn recompute_sha256(&self) -> Result<String, String> {
        let mut projection = self.clone();
        projection.population_sha256.clear();
        let mut digest = Sha256::new();
        digest.update(b"PRISM_NCMC_BOUNDARY_POPULATION_LINEAGE_V1\0");
        digest.update(serde_json::to_vec(&projection).map_err(|error| error.to_string())?);
        Ok(format!("{:x}", digest.finalize()))
    }

    pub fn seal(
        boundary: NcmcForceBoundaryIdentity,
        slot_events: Vec<NcmcBoundarySlotEventLineage>,
    ) -> Result<Self, String> {
        let mut value = Self {
            schema_kind: "prismos_ncmc_boundary_population_lineage".into(),
            schema_version: 1,
            boundary,
            slot_events,
            population_sha256: String::new(),
        };
        value.population_sha256 = value.recompute_sha256()?;
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<(), String> {
        self.boundary.validate()?;
        if self.schema_kind != "prismos_ncmc_boundary_population_lineage"
            || self.schema_version != 1
            || self.slot_events.is_empty()
            || self.population_sha256 != self.recompute_sha256()?
        {
            return Err("invalid NCMC boundary-population lineage".into());
        }
        let mut previous_slot = None;
        let mut previous_sequence = None;
        for event in &self.slot_events {
            validate_sha256("NCMC boundary raw event", &event.raw_event_sha256)?;
            validate_sha256(
                "NCMC boundary emission receipt",
                &event.emission_receipt_sha256,
            )?;
            if previous_slot.is_some_and(|slot| slot >= event.slot_id)
                || previous_sequence.is_some_and(|sequence| sequence >= event.event_sequence)
            {
                return Err("NCMC boundary slot population is not strictly ordered".into());
            }
            previous_slot = Some(event.slot_id);
            previous_sequence = Some(event.event_sequence);
        }
        Ok(())
    }
}

/// Typed force-plane event emitted by the native engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LambdaFragmentForceEvent {
    /// Schema discriminator.
    pub schema_kind: String,
    /// Schema version.
    pub schema_version: u32,
    /// Stream identity.
    pub stream_id: u32,
    /// MD step.
    pub step: u64,
    /// Strict total order within this lambda event stream.
    pub event_sequence: u64,
    /// Prior typed-emission receipt digest; absent only for sequence zero.
    pub parent_receipt_sha256: Option<String>,
    /// Exact NCMC move/operation/boundary identity.  Native lambda force
    /// observations are controlled NCMC boundaries and may repeat the same
    /// `(stream, step, slot)` without collision.
    pub ncmc_boundary: Option<NcmcForceBoundaryIdentity>,
    /// Slot identity.
    pub slot_id: u32,
    /// Species identity.
    pub species_id: u32,
    /// Chemical-state identity.
    pub chemical_state_id: u32,
    /// Current alchemical coordinate.
    pub lambda: f64,
    /// Alchemical path interpretation; partial state paths have no single
    /// chemical identity and must be interpreted through both endpoints.
    pub path_kind: AlchemicalPathKind,
    /// Exact lambda=0 identity.
    pub endpoint_a: NcmcEndpointIdentity,
    /// Exact lambda=1 identity.
    pub endpoint_b: NcmcEndpointIdentity,
    /// Slot contribution to potential energy.
    pub potential_kcal_mol: f64,
    /// Slot dU/dlambda.
    pub du_dlambda_kcal_mol: f64,
    /// Declared potential/dUdl source unit, exactly `kcal/mol`.
    pub energy_unit: String,
    /// Declared force source unit, exactly `kcal/mol/angstrom`.
    pub force_unit: String,
    /// Endpoint parameter identity.
    pub endpoint_parameter_sha256: String,
    /// Package-time scheduler-support to prepared-endpoint binding.
    pub endpoint_support_binding_sha256: String,
    /// Exact resident uploaded dual-endpoint union-plane contract used by this
    /// force evaluation, including path typing and every uploaded plane hash.
    pub dual_endpoint_union_contract_sha256: String,
    /// Hamiltonian identity.
    pub hamiltonian_sha256: String,
    /// Canonical phase/condition binding digest for this exact step.
    pub phase_protocol_sha256: String,
    /// Canonical PrismCode opcode.
    pub prismcode_opcode: String,
    /// Numeric PrismCode opcode bound by the canonical registry.
    pub prismcode_opcode_id: u16,
    /// Canonical QNANL family.
    pub qnanl_artifact_family: String,
}

impl LambdaFragmentForceEvent {
    /// Verify canonical executable dispatch identities and numeric data.
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_kind != "prismos_lambda_fragment_force_event"
            || self.schema_version != 3
            || !self.lambda.is_finite()
            || !(0.0..=1.0).contains(&self.lambda)
            || !self.potential_kcal_mol.is_finite()
            || !self.du_dlambda_kcal_mol.is_finite()
            || self.energy_unit != "kcal/mol"
            || self.force_unit != "kcal/mol/angstrom"
        {
            return Err("invalid lambda force event payload".into());
        }
        if let Some(boundary) = &self.ncmc_boundary {
            boundary.validate()?;
        }
        if matches!(self.endpoint_a.occupancy, EndpointOccupancy::Switching)
            || matches!(self.endpoint_b.occupancy, EndpointOccupancy::Switching)
        {
            return Err("lambda force event endpoints cannot be SWITCHING".into());
        }
        validate_dispatch_binding(
            &self.prismcode_opcode,
            self.prismcode_opcode_id,
            prismcode_opcodes::FORCE,
            &self.qnanl_artifact_family,
            QnanlArtifactFamily::LambdaFragmentForceEvent,
        )?;
        validate_sha256("endpoint parameter", &self.endpoint_parameter_sha256)?;
        validate_sha256(
            "endpoint support binding",
            &self.endpoint_support_binding_sha256,
        )?;
        validate_sha256(
            "dual endpoint union contract",
            &self.dual_endpoint_union_contract_sha256,
        )?;
        validate_sha256("Hamiltonian", &self.hamiltonian_sha256)?;
        validate_sha256("phase/condition binding", &self.phase_protocol_sha256)?;
        if let Some(parent) = &self.parent_receipt_sha256 {
            validate_sha256("parent receipt", parent)?;
        }
        Ok(())
    }
}

/// Hash-bound adapter receipt used when a fiber schema stores `dU/dlambda` in
/// kJ/mol while native force events remain in kcal/mol.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LambdaFiberUnitConversion {
    /// Force event raw digest before conversion.
    pub source_event_sha256: String,
    /// Exact source unit.
    pub source_unit: String,
    /// Exact target unit.
    pub target_unit: String,
    /// Exact CODATA-defined thermochemical conversion used here.
    pub factor: f64,
    /// Converted dU/dlambda.
    pub du_dlambda_kj_mol: f64,
}

impl LambdaFiberUnitConversion {
    /// Validate that no scalar was relabeled without the exact conversion.
    pub fn validate(&self) -> Result<(), String> {
        validate_sha256("source lambda force event", &self.source_event_sha256)?;
        if self.source_unit != "kcal/mol"
            || self.target_unit != "kJ/mol"
            || self.factor.to_bits() != 4.184f64.to_bits()
            || !self.du_dlambda_kj_mol.is_finite()
        {
            return Err("invalid lambda-to-fiber unit conversion receipt".into());
        }
        Ok(())
    }
}

/// Convert native force-event dU/dlambda to the fiber unit without silently
/// relabeling the scalar.
pub fn force_event_dudl_for_fiber(
    event: &LambdaFragmentForceEvent,
) -> Result<LambdaFiberUnitConversion, String> {
    event.validate()?;
    let source = serde_json::to_vec(event).map_err(|error| error.to_string())?;
    let conversion = LambdaFiberUnitConversion {
        source_event_sha256: format!("{:x}", Sha256::digest(source)),
        source_unit: "kcal/mol".into(),
        target_unit: "kJ/mol".into(),
        factor: 4.184,
        du_dlambda_kj_mol: event.du_dlambda_kcal_mol * 4.184,
    };
    conversion.validate()?;
    Ok(conversion)
}

/// Endpoint observation eligible to join a Ghost/lattice fiber.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LambdaFragmentEndpointObservation {
    /// Schema discriminator.
    pub schema_kind: String,
    /// Schema version.
    pub schema_version: u32,
    /// Stream identity.
    pub stream_id: u32,
    /// MD step.
    pub step: u64,
    /// Strict total order within this lambda event stream.
    pub event_sequence: u64,
    /// Prior typed-emission receipt digest.
    pub parent_receipt_sha256: String,
    /// Exact supporting physical-endpoint force event bytes.
    pub supporting_force_event_sha256: String,
    /// Typed emission receipt for the supporting force event.
    pub supporting_force_emission_receipt_sha256: String,
    /// NCMC boundary at which the supporting physical endpoint was measured.
    pub supporting_force_boundary: NcmcForceBoundaryIdentity,
    /// Slot identity.
    pub slot_id: u32,
    /// Species identity.
    pub species_id: u32,
    /// Chemical-state identity.
    pub chemical_state_id: u32,
    /// Exact endpoint, zero or one.
    pub lambda_endpoint: u8,
    /// Endpoint occupancy.
    pub occupancy: EndpointOccupancy,
    /// Alchemical path interpretation.
    pub path_kind: AlchemicalPathKind,
    /// Lambda=0 identity.
    pub endpoint_a: NcmcEndpointIdentity,
    /// Lambda=1 identity.
    pub endpoint_b: NcmcEndpointIdentity,
    /// Hash binding stream condition and protocol phase.
    pub phase_protocol_sha256: String,
    /// Optional Ghost record lineage digest assigned downstream.
    pub ghost_lineage_sha256: Option<String>,
    /// Canonical PrismCode opcode.
    pub prismcode_opcode: String,
    /// Numeric PrismCode opcode bound by the canonical registry.
    pub prismcode_opcode_id: u16,
    /// Canonical QNANL family.
    pub qnanl_artifact_family: String,
}

impl LambdaFragmentEndpointObservation {
    /// Fail closed if an intermediate lambda is presented as a physical event.
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_kind != "prismos_lambda_fragment_endpoint_observation"
            || self.schema_version != 2
            || !matches!(self.lambda_endpoint, 0 | 1)
        {
            return Err("invalid lambda endpoint observation contract".into());
        }
        validate_dispatch_binding(
            &self.prismcode_opcode,
            self.prismcode_opcode_id,
            prismcode_opcodes::ENDPOINT_OBSERVE,
            &self.qnanl_artifact_family,
            QnanlArtifactFamily::LambdaFragmentEndpointObservation,
        )?;
        if matches!(self.endpoint_a.occupancy, EndpointOccupancy::Switching)
            || matches!(self.endpoint_b.occupancy, EndpointOccupancy::Switching)
        {
            return Err("lambda observation endpoints cannot be SWITCHING".into());
        }
        let expected = if self.lambda_endpoint == 0 {
            self.endpoint_a.occupancy
        } else {
            self.endpoint_b.occupancy
        };
        if self.occupancy != expected {
            return Err("lambda endpoint and occupancy disagree".into());
        }
        validate_sha256("phase protocol digest", &self.phase_protocol_sha256)?;
        validate_sha256("parent receipt", &self.parent_receipt_sha256)?;
        validate_sha256(
            "supporting force event",
            &self.supporting_force_event_sha256,
        )?;
        validate_sha256(
            "supporting force emission receipt",
            &self.supporting_force_emission_receipt_sha256,
        )?;
        self.supporting_force_boundary.validate()?;
        if let Some(digest) = &self.ghost_lineage_sha256 {
            validate_sha256("Ghost lineage digest", digest)?;
        }
        Ok(())
    }
}

/// Evaluate energy, force and dU/dlambda from one shared formula.
pub fn evaluate_softcore_pair(
    displacement_a: [f64; 3],
    lambda: f64,
    charge_product_e2: f64,
    sigma_a: f64,
    epsilon_kcal_mol: f64,
    h: &SoftcoreHamiltonian,
) -> Result<SoftcorePairEvaluation, String> {
    validate_hamiltonian(h)?;
    if h.formula == "PRISM_LAMBDA_SOFTCORE_LJ_EWALD_CHARGE_V2"
        && charge_product_e2.to_bits() != 0.0f64.to_bits()
    {
        return Err(
            "V2 charge-linear Ewald electrostatics must use the PME real-space/reciprocal path"
                .into(),
        );
    }
    if !lambda.is_finite() || !(0.0..=1.0).contains(&lambda) {
        return Err("lambda must be finite and in [0,1]".to_string());
    }
    if !charge_product_e2.is_finite()
        || !sigma_a.is_finite()
        || !epsilon_kcal_mol.is_finite()
        || sigma_a <= 0.0
        || epsilon_kcal_mol < 0.0
    {
        return Err("invalid pair parameters".to_string());
    }
    let r2 = displacement_a.iter().map(|x| x * x).sum::<f64>();
    if !r2.is_finite() || r2 <= 0.0 {
        return Err("pair displacement must be finite and nonzero".to_string());
    }
    let r = r2.sqrt();
    if r > h.cutoff_a {
        return Ok(SoftcorePairEvaluation {
            potential_kcal_mol: 0.0,
            lj_kcal_mol: 0.0,
            coulomb_kcal_mol: 0.0,
            force_on_i_kcal_mol_a: [0.0; 3],
            du_dlambda_kcal_mol: 0.0,
        });
    }

    let one_minus = 1.0 - lambda;
    let scale = lambda.powf(h.coupling_power);
    let dscale = if lambda == 0.0 && h.coupling_power == 1.0 {
        1.0
    } else if lambda == 0.0 {
        0.0
    } else {
        h.coupling_power * lambda.powf(h.coupling_power - 1.0)
    };
    let soften = one_minus.powf(h.softcore_power);
    let dsoften = if one_minus == 0.0 && h.softcore_power == 1.0 {
        -1.0
    } else if one_minus == 0.0 {
        0.0
    } else {
        -h.softcore_power * one_minus.powf(h.softcore_power - 1.0)
    };

    let sigma2 = sigma_a * sigma_a;
    let sigma6 = sigma2 * sigma2 * sigma2;
    let r6 = r2 * r2 * r2;
    let sc6 = r6 + h.alpha_lj * soften * sigma6;
    let x = sigma6 / sc6;
    let lj_shape = x * x - x;
    let lj = 4.0 * epsilon_kcal_mol * scale * lj_shape;
    let dsc6_dlambda = h.alpha_lj * sigma6 * dsoften;
    let dx_dlambda = -sigma6 * dsc6_dlambda / (sc6 * sc6);
    let dlj_dlambda =
        4.0 * epsilon_kcal_mol * (dscale * lj_shape + scale * (2.0 * x - 1.0) * dx_dlambda);
    let dx_dr = -sigma6 * (6.0 * r.powi(5)) / (sc6 * sc6);
    let dlj_dr = 4.0 * epsilon_kcal_mol * scale * (2.0 * x - 1.0) * dx_dr;

    let sc2 = r2 + h.alpha_coulomb_a2 * soften;
    let inv_sc = sc2.sqrt().recip();
    let coulomb_prefactor = COULOMB_KCAL_A_PER_MOL_E2 * charge_product_e2;
    let coulomb = coulomb_prefactor * scale * inv_sc;
    let dsc2_dlambda = h.alpha_coulomb_a2 * dsoften;
    let dinv_sc_dlambda = -0.5 * dsc2_dlambda * inv_sc.powi(3);
    let dcoulomb_dlambda = coulomb_prefactor * (dscale * inv_sc + scale * dinv_sc_dlambda);
    let dcoulomb_dr = -coulomb_prefactor * scale * r * inv_sc.powi(3);

    let force_radial = -(dlj_dr + dcoulomb_dr);
    let inv_r = r.recip();
    let force = displacement_a.map(|component| force_radial * component * inv_r);
    Ok(SoftcorePairEvaluation {
        potential_kcal_mol: lj + coulomb,
        lj_kcal_mol: lj,
        coulomb_kcal_mol: coulomb,
        force_on_i_kcal_mol_a: force,
        du_dlambda_kcal_mol: dlj_dlambda + dcoulomb_dlambda,
    })
}

fn validate_pair_endpoint(endpoint: &FragmentPairEndpointParameters) -> Result<(), String> {
    if !endpoint.charge_product_e2.is_finite()
        || !endpoint.sigma_a.is_finite()
        || endpoint.sigma_a <= 0.0
        || !endpoint.epsilon_kcal_mol.is_finite()
        || endpoint.epsilon_kcal_mol < 0.0
        || (!endpoint.present
            && (endpoint.charge_product_e2.to_bits() != 0.0f64.to_bits()
                || endpoint.epsilon_kcal_mol.to_bits() != 0.0f64.to_bits()))
    {
        return Err("invalid physical/dummy pair endpoint parameters".into());
    }
    Ok(())
}

/// Evaluate one fixed-dimensional state-A/state-B pair Hamiltonian. Common
/// physical atoms use exact endpoint-energy interpolation; an appearing or
/// disappearing explicit proton uses the same soft-core Hamiltonian in the
/// appropriate direction. Energy, force, and dU/dlambda are derived from this
/// single path, so they cannot silently select different mixing rules.
pub fn evaluate_dual_endpoint_pair(
    displacement_a: [f64; 3],
    lambda: f64,
    endpoint_a: FragmentPairEndpointParameters,
    endpoint_b: FragmentPairEndpointParameters,
    hamiltonian: &SoftcoreHamiltonian,
) -> Result<DualEndpointPairEvaluation, String> {
    observation_eligibility(lambda)?;
    validate_pair_endpoint(&endpoint_a)?;
    validate_pair_endpoint(&endpoint_b)?;
    if !endpoint_a.present && !endpoint_b.present {
        return Err("pair is dummy at both state-transition endpoints".into());
    }
    let zero = SoftcorePairEvaluation {
        potential_kcal_mol: 0.0,
        lj_kcal_mol: 0.0,
        coulomb_kcal_mol: 0.0,
        force_on_i_kcal_mol_a: [0.0; 3],
        du_dlambda_kcal_mol: 0.0,
    };
    let physical = |parameters: FragmentPairEndpointParameters| {
        if parameters.present {
            evaluate_softcore_pair(
                displacement_a,
                1.0,
                parameters.charge_product_e2,
                parameters.sigma_a,
                parameters.epsilon_kcal_mol,
                hamiltonian,
            )
        } else {
            Ok(zero)
        }
    };
    let physical_a = physical(endpoint_a)?;
    let physical_b = physical(endpoint_b)?;
    let (potential, force, dudl) = match (endpoint_a.present, endpoint_b.present) {
        (true, true) => {
            let force = std::array::from_fn(|axis| {
                (1.0 - lambda) * physical_a.force_on_i_kcal_mol_a[axis]
                    + lambda * physical_b.force_on_i_kcal_mol_a[axis]
            });
            (
                (1.0 - lambda) * physical_a.potential_kcal_mol
                    + lambda * physical_b.potential_kcal_mol,
                force,
                physical_b.potential_kcal_mol - physical_a.potential_kcal_mol,
            )
        }
        (false, true) => {
            let value = evaluate_softcore_pair(
                displacement_a,
                lambda,
                endpoint_b.charge_product_e2,
                endpoint_b.sigma_a,
                endpoint_b.epsilon_kcal_mol,
                hamiltonian,
            )?;
            (
                value.potential_kcal_mol,
                value.force_on_i_kcal_mol_a,
                value.du_dlambda_kcal_mol,
            )
        }
        (true, false) => {
            let value = evaluate_softcore_pair(
                displacement_a,
                1.0 - lambda,
                endpoint_a.charge_product_e2,
                endpoint_a.sigma_a,
                endpoint_a.epsilon_kcal_mol,
                hamiltonian,
            )?;
            (
                value.potential_kcal_mol,
                value.force_on_i_kcal_mol_a,
                -value.du_dlambda_kcal_mol,
            )
        }
        (false, false) => unreachable!(),
    };
    Ok(DualEndpointPairEvaluation {
        potential_kcal_mol: potential,
        endpoint_a_potential_kcal_mol: physical_a.potential_kcal_mol,
        endpoint_b_potential_kcal_mol: physical_b.potential_kcal_mol,
        force_on_i_kcal_mol_a: force,
        du_dlambda_kcal_mol: dudl,
    })
}

/// Dynamic-charge real-space Ewald component used by production PME paths.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EwaldPairEvaluation {
    pub potential_kcal_mol: f64,
    pub force_on_i_kcal_mol_a: [f64; 3],
    pub du_dlambda_kcal_mol: f64,
}

/// Kernel inclusion mask required by the exact reciprocal correction. There
/// is deliberately no variant that also evaluates the ordinary erfc pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum EwaldExceptionRealspaceComposition {
    /// Excluded/scaled pairs are skipped in the ordinary erfc neighbor loop
    /// and evaluated exactly once by the exception loop.
    OrdinaryPairOmitted,
}

fn validate_ewald_pair_inputs(
    displacement_a: [f64; 3],
    beta_a_inv: f64,
) -> Result<(f64, [f64; 3]), String> {
    if !beta_a_inv.is_finite() || beta_a_inv <= 0.0 {
        return Err("Ewald beta must be finite and positive".into());
    }
    let r2 = displacement_a
        .iter()
        .map(|value| value * value)
        .sum::<f64>();
    if !r2.is_finite() || r2 <= 1.0e-16 {
        return Err("Ewald pair displacement is singular".into());
    }
    Ok((r2.sqrt(), displacement_a))
}

/// `q_i(lambda) q_j(lambda) erfc(beta*r)/r`, including the exact analytic
/// coordinate force and charge-path derivative. This is mutually exclusive
/// with the legacy unsplit `1/r` direct-space path.
pub fn evaluate_dynamic_charge_ewald_realspace_pair(
    displacement_a: [f64; 3],
    charge_i_e: f64,
    dcharge_i_dlambda_e: f64,
    charge_j_e: f64,
    dcharge_j_dlambda_e: f64,
    beta_a_inv: f64,
) -> Result<EwaldPairEvaluation, String> {
    for value in [
        charge_i_e,
        dcharge_i_dlambda_e,
        charge_j_e,
        dcharge_j_dlambda_e,
    ] {
        if !value.is_finite() {
            return Err("nonfinite dynamic charge in Ewald real-space pair".into());
        }
    }
    let (r, displacement) = validate_ewald_pair_inputs(displacement_a, beta_a_inv)?;
    let br = beta_a_inv * r;
    let erfc = statrs::function::erf::erfc(br);
    let inv_r = r.recip();
    let radial = erfc * inv_r;
    let q_product = charge_i_e * charge_j_e;
    let dq_product = dcharge_i_dlambda_e * charge_j_e + charge_i_e * dcharge_j_dlambda_e;
    let prefactor = COULOMB_KCAL_A_PER_MOL_E2;
    let potential = prefactor * q_product * radial;
    let du_dlambda = prefactor * dq_product * radial;
    let d_radial_dr = -2.0 * beta_a_inv / std::f64::consts::PI.sqrt() * (-br * br).exp() * inv_r
        - erfc * inv_r * inv_r;
    let radial_force = -prefactor * q_product * d_radial_dr;
    Ok(EwaldPairEvaluation {
        potential_kcal_mol: potential,
        force_on_i_kcal_mol_a: displacement.map(|value| radial_force * value * inv_r),
        du_dlambda_kcal_mol: du_dlambda,
    })
}

/// Exact reciprocal exclusion/scaled-1-4 correction. The reciprocal mesh
/// contains the regular current-charge pair; adding this term makes the final
/// pair equal `q_exception/r` (zero for a complete exclusion).
pub fn evaluate_dynamic_charge_ewald_exception_correction(
    displacement_a: [f64; 3],
    regular_charge_product_e2: f64,
    d_regular_charge_product_dlambda_e2: f64,
    exception_charge_product_e2: f64,
    d_exception_charge_product_dlambda_e2: f64,
    beta_a_inv: f64,
    composition: EwaldExceptionRealspaceComposition,
) -> Result<EwaldPairEvaluation, String> {
    if composition != EwaldExceptionRealspaceComposition::OrdinaryPairOmitted {
        return Err("unsupported Ewald exception real-space composition".into());
    }
    for value in [
        regular_charge_product_e2,
        d_regular_charge_product_dlambda_e2,
        exception_charge_product_e2,
        d_exception_charge_product_dlambda_e2,
    ] {
        if !value.is_finite() {
            return Err("nonfinite dynamic charge in Ewald exception correction".into());
        }
    }
    let (r, displacement) = validate_ewald_pair_inputs(displacement_a, beta_a_inv)?;
    let br = beta_a_inv * r;
    let erf = statrs::function::erf::erf(br);
    let inv_r = r.recip();
    let numerator = exception_charge_product_e2 - regular_charge_product_e2 * erf;
    let d_numerator =
        d_exception_charge_product_dlambda_e2 - d_regular_charge_product_dlambda_e2 * erf;
    let prefactor = COULOMB_KCAL_A_PER_MOL_E2;
    let potential = prefactor * numerator * inv_r;
    let du_dlambda = prefactor * d_numerator * inv_r;
    let d_erf_dr = 2.0 * beta_a_inv / std::f64::consts::PI.sqrt() * (-br * br).exp();
    let dpotential_dr =
        prefactor * (-regular_charge_product_e2 * d_erf_dr * inv_r - numerator * inv_r * inv_r);
    let radial_force = -dpotential_dr;
    Ok(EwaldPairEvaluation {
        potential_kcal_mol: potential,
        force_on_i_kcal_mol_a: displacement.map(|value| radial_force * value * inv_r),
        du_dlambda_kcal_mol: du_dlambda,
    })
}

/// Stable digest for an endpoint force-field parameter state. This permits
/// tautomer pairs such as 5-fluoro-2-hydroxypyridine/5-fluoro-2-pyridone to
/// share atom mapping while retaining distinct, hash-bound parameters.
pub fn fragment_state_parameter_digest_sha256(
    state: &FragmentChemicalState,
) -> Result<String, String> {
    let mut projection = state.clone();
    projection.parameter_digest_sha256.clear();
    let bytes = serde_json::to_vec(&projection).map_err(|error| error.to_string())?;
    let mut digest = Sha256::new();
    digest.update(b"PRISM_LAMBDA_FRAGMENT_ENDPOINT_PARAMETERS_V2\0");
    digest.update(bytes);
    Ok(format!("{:x}", digest.finalize()))
}

/// Stable digest for one endpoint bonded plane.
pub fn fragment_bonded_parameter_plane_sha256(
    plane: &FragmentBondedParameterPlane,
) -> Result<String, String> {
    let bytes = serde_json::to_vec(plane).map_err(|error| error.to_string())?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

/// Stable digest for a complete state transition.
pub fn fragment_state_transition_sha256(
    transition: &FragmentStateTransition,
) -> Result<String, String> {
    let mut projection = transition.clone();
    projection.transition_sha256.clear();
    let bytes = serde_json::to_vec(&projection).map_err(|error| error.to_string())?;
    let mut digest = Sha256::new();
    digest.update(b"PRISM_CONSTANT_TOPOLOGY_UNION_TRANSITION_V1\0");
    digest.update(bytes);
    Ok(format!("{:x}", digest.finalize()))
}

fn validate_bonded_plane(
    plane: &FragmentBondedParameterPlane,
    atom_ids: &HashSet<u32>,
) -> Result<(), String> {
    if plane.formula != "PRISM_ENDPOINT_BONDED_AMBER_V1" {
        return Err(format!(
            "unsupported endpoint bonded formula {}",
            plane.formula
        ));
    }
    let mut bonds = HashSet::new();
    for bond in &plane.bonds {
        let key = (
            bond.template_atom_i.min(bond.template_atom_j),
            bond.template_atom_i.max(bond.template_atom_j),
        );
        if key.0 == key.1
            || !bonds.insert(key)
            || !atom_ids.contains(&key.0)
            || !atom_ids.contains(&key.1)
            || !bond.length_a.is_finite()
            || bond.length_a <= 0.0
            || !bond.force_constant_kcal_mol_a2.is_finite()
            || bond.force_constant_kcal_mol_a2 < 0.0
        {
            return Err("invalid endpoint bond parameter".into());
        }
    }
    let mut angles = HashSet::new();
    for angle in &plane.angles {
        let key = (
            angle.template_atom_i,
            angle.template_atom_j,
            angle.template_atom_k,
        );
        if angle.template_atom_i == angle.template_atom_j
            || angle.template_atom_j == angle.template_atom_k
            || angle.template_atom_i == angle.template_atom_k
            || !angles.insert(key)
            || [
                angle.template_atom_i,
                angle.template_atom_j,
                angle.template_atom_k,
            ]
            .iter()
            .any(|id| !atom_ids.contains(id))
            || !angle.angle_rad.is_finite()
            || !(0.0..=std::f64::consts::PI).contains(&angle.angle_rad)
            || !angle.force_constant_kcal_mol_rad2.is_finite()
            || angle.force_constant_kcal_mol_rad2 < 0.0
        {
            return Err("invalid endpoint angle parameter".into());
        }
    }
    let mut torsions = HashSet::new();
    for torsion in &plane.proper_and_improper_torsions {
        let key = (
            torsion.template_atom_i,
            torsion.template_atom_j,
            torsion.template_atom_k,
            torsion.template_atom_l,
            torsion.periodicity,
            torsion.phase_rad.to_bits(),
            torsion.improper,
        );
        if torsion.periodicity == 0
            || !torsions.insert(key)
            || [
                torsion.template_atom_i,
                torsion.template_atom_j,
                torsion.template_atom_k,
                torsion.template_atom_l,
            ]
            .iter()
            .any(|id| !atom_ids.contains(id))
            || !torsion.phase_rad.is_finite()
            || !torsion.barrier_kcal_mol.is_finite()
            || torsion.barrier_kcal_mol < 0.0
        {
            return Err("invalid endpoint torsion parameter".into());
        }
    }
    let mut constraints = HashSet::new();
    for constraint in &plane.constraints {
        let key = (
            constraint.template_atom_i.min(constraint.template_atom_j),
            constraint.template_atom_i.max(constraint.template_atom_j),
        );
        if key.0 == key.1
            || !constraints.insert(key)
            || !atom_ids.contains(&key.0)
            || !atom_ids.contains(&key.1)
            || !constraint.distance_a.is_finite()
            || constraint.distance_a <= 0.0
        {
            return Err("invalid endpoint constraint parameter".into());
        }
    }
    Ok(())
}

fn validate_state_transition(
    transition: &FragmentStateTransition,
    state_a: &FragmentChemicalState,
    state_b: &FragmentChemicalState,
) -> Result<(), String> {
    if transition.transition_id.trim().is_empty()
        || transition.state_a_id != state_a.state_id
        || transition.state_b_id != state_b.state_id
        || transition.state_a_id == transition.state_b_id
        || !transition.constraints_invariant
        || !transition
            .reference_free_energy_b_minus_a_kcal_mol
            .is_finite()
        || transition.delta_bound_protons_b_minus_a != state_b.proton_count - state_a.proton_count
        || (transition.reference_free_energy_b_minus_a_kcal_mol
            - (state_b.intrinsic_free_energy_kcal_mol - state_a.intrinsic_free_energy_kcal_mol))
            .abs()
            > 1.0e-12
        || transition.required_runtime_capability != "CONSTANT_TOPOLOGY_UNION_DUMMY_PROTON_V1"
        || transition.transition_sha256 != fragment_state_transition_sha256(transition)?
    {
        return Err("invalid constant-topology state transition".into());
    }
    if state_a.bonded_parameter_plane.constraints != state_b.bonded_parameter_plane.constraints {
        return Err("state transition has lambda-dependent constraints".into());
    }
    let mut union_ids = HashSet::new();
    let state_a_atoms: HashMap<u32, &FragmentAtomState> = state_a
        .atom_parameters
        .iter()
        .map(|atom| (atom.template_atom_id, atom))
        .collect();
    let state_b_atoms: HashMap<u32, &FragmentAtomState> = state_b
        .atom_parameters
        .iter()
        .map(|atom| (atom.template_atom_id, atom))
        .collect();
    for atom in &transition.constant_topology_union_atoms {
        if !union_ids.insert(atom.union_template_atom_id)
            || atom.present_in_state_a
                != state_a_atoms
                    .get(&atom.union_template_atom_id)
                    .map(|value| value.endpoint_present)
                    .unwrap_or(false)
            || atom.present_in_state_b
                != state_b_atoms
                    .get(&atom.union_template_atom_id)
                    .map(|value| value.endpoint_present)
                    .unwrap_or(false)
        {
            return Err("union atom presence mask disagrees with endpoint parameters".into());
        }
        if atom.present_in_state_a != atom.present_in_state_b && atom.atomic_number != 1 {
            return Err("only explicit protons may enter/leave a protonation union path".into());
        }
    }
    if union_ids.len() != state_a.atom_parameters.len()
        || union_ids.len() != state_b.atom_parameters.len()
        || state_a_atoms.keys().any(|id| !union_ids.contains(id))
        || state_b_atoms.keys().any(|id| !union_ids.contains(id))
    {
        return Err("endpoint atom planes do not cover the exact constant-topology union".into());
    }
    let dummy_required = transition
        .constant_topology_union_atoms
        .iter()
        .any(|atom| atom.present_in_state_a != atom.present_in_state_b);
    if transition.dummy_proton_required != dummy_required {
        return Err("dummy-proton declaration disagrees with union presence masks".into());
    }
    let bonded_invariant =
        state_a.bonded_parameter_plane_sha256 == state_b.bonded_parameter_plane_sha256;
    if transition.bonded_plane_invariant != bonded_invariant {
        return Err("bonded-plane invariance declaration disagrees with endpoint digests".into());
    }
    let mut map_a = HashSet::new();
    let mut map_b = HashSet::new();
    for map in &transition.atom_map_a_to_b {
        if !map.map_confidence.is_finite()
            || !(0.0..=1.0).contains(&map.map_confidence)
            || !map_a.insert(map.atom_a)
            || !map_b.insert(map.atom_b)
            || !union_ids.contains(&map.atom_a)
            || !union_ids.contains(&map.atom_b)
            || map.element.trim().is_empty()
        {
            return Err("invalid endpoint atom map".into());
        }
    }
    let expected_common: HashSet<u32> = transition
        .constant_topology_union_atoms
        .iter()
        .filter(|atom| atom.present_in_state_a && atom.present_in_state_b)
        .map(|atom| atom.union_template_atom_id)
        .collect();
    if map_a != expected_common || map_b != expected_common {
        return Err("endpoint atom map does not exactly cover common physical atoms".into());
    }
    let unmatched_a: HashSet<u32> = transition.unmatched_a.iter().copied().collect();
    let unmatched_b: HashSet<u32> = transition.unmatched_b.iter().copied().collect();
    if unmatched_a.len() != transition.unmatched_a.len()
        || unmatched_b.len() != transition.unmatched_b.len()
        || unmatched_a
            != transition
                .constant_topology_union_atoms
                .iter()
                .filter(|atom| atom.present_in_state_a && !atom.present_in_state_b)
                .map(|atom| atom.union_template_atom_id)
                .collect::<HashSet<_>>()
        || unmatched_b
            != transition
                .constant_topology_union_atoms
                .iter()
                .filter(|atom| !atom.present_in_state_a && atom.present_in_state_b)
                .map(|atom| atom.union_template_atom_id)
                .collect::<HashSet<_>>()
    {
        return Err("unmatched endpoint atoms disagree with union presence masks".into());
    }
    let mut graph_changes = HashSet::new();
    for change in &transition.bond_graph_change_set {
        let key = (
            change.template_atom_i.min(change.template_atom_j),
            change.template_atom_i.max(change.template_atom_j),
        );
        if key.0 == key.1
            || !graph_changes.insert(key)
            || !union_ids.contains(&key.0)
            || !union_ids.contains(&key.1)
            || change.bond_order_state_a == change.bond_order_state_b
            || change
                .bond_order_state_a
                .into_iter()
                .chain(change.bond_order_state_b)
                .any(|order| order == 0 || order > 3)
        {
            return Err("invalid endpoint bond-graph change".into());
        }
    }
    Ok(())
}

fn validate_hamiltonian(h: &SoftcoreHamiltonian) -> Result<(), String> {
    if h.formula != "PRISM_LAMBDA_SOFTCORE_LJ_COULOMB_V1"
        && h.formula != "PRISM_LAMBDA_SOFTCORE_LJ_EWALD_CHARGE_V2"
    {
        return Err(format!(
            "unsupported lambda Hamiltonian formula {}",
            h.formula
        ));
    }
    if h.periodic_charge_convention
        != "NET_CHARGE_CHANGE_REQUIRES_COUPLED_NEUTRALIZATION_OR_BOX_PME_FINITE_SIZE_CORRECTION_V1"
    {
        return Err("unsupported periodic net-charge treatment convention".into());
    }
    for (name, value) in [
        ("alpha_lj", h.alpha_lj),
        ("alpha_coulomb_a2", h.alpha_coulomb_a2),
        ("softcore_power", h.softcore_power),
        ("coupling_power", h.coupling_power),
        ("cutoff_a", h.cutoff_a),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(format!("{name} must be finite and positive"));
        }
    }
    Ok(())
}

/// Whether a state may contribute to a physical pocket observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ObservationEligibility {
    /// Physical lambda=0 or lambda=1 endpoint.
    PhysicalEndpoint,
    /// Intermediate lambda; useful for sampling/work only.
    SwitchingOnly,
}

/// Classify observation eligibility without rounding an intermediate state to
/// a physical endpoint.
pub fn observation_eligibility(lambda: f64) -> Result<ObservationEligibility, String> {
    if !lambda.is_finite() || !(0.0..=1.0).contains(&lambda) {
        return Err("lambda must be finite and in [0,1]".to_string());
    }
    if lambda.to_bits() == 0.0f64.to_bits() || lambda.to_bits() == 1.0f64.to_bits() {
        Ok(ObservationEligibility::PhysicalEndpoint)
    } else {
        Ok(ObservationEligibility::SwitchingOnly)
    }
}

/// Triclinic minimum-image displacement. This CPU reference is used by
/// topology/water validation and translational-invariance tests. The CUDA
/// force path must agree with it before the production solvent gate can pass.
pub fn minimum_image_displacement(
    displacement_a: [f64; 3],
    periodic_box: &PeriodicBoxContract,
) -> Result<[f64; 3], String> {
    let m = periodic_box.vectors_a;
    let det = periodic_box.signed_volume_a3();
    if !det.is_finite() || det.abs() <= 1.0e-12 {
        return Err("periodic box is singular".into());
    }
    // Coordinates are row-vector coefficients: d = f0*a + f1*b + f2*c.
    let inverse = [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det,
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det,
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det,
        ],
        [
            (m[2][1] * m[0][2] - m[2][2] * m[0][1]) / det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det,
            (m[2][0] * m[0][1] - m[2][1] * m[0][0]) / det,
        ],
        [
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det,
        ],
    ];
    let mut fractional = [0.0; 3];
    for (i, value) in fractional.iter_mut().enumerate() {
        *value = inverse[i][0] * displacement_a[0]
            + inverse[i][1] * displacement_a[1]
            + inverse[i][2] * displacement_a[2];
        *value -= value.round();
    }
    Ok([
        fractional[0] * m[0][0] + fractional[1] * m[1][0] + fractional[2] * m[2][0],
        fractional[0] * m[0][1] + fractional[1] * m[1][1] + fractional[2] * m[2][1],
        fractional[0] * m[0][2] + fractional[1] * m[1][2] + fractional[2] * m[2][2],
    ])
}

#[derive(Clone)]
struct CoordinateJet {
    value: f64,
    derivative: Vec<f64>,
}

impl CoordinateJet {
    fn constant(value: f64, dimension: usize) -> Self {
        Self {
            value,
            derivative: vec![0.0; dimension],
        }
    }

    fn variable(value: f64, dimension: usize, axis: usize) -> Self {
        let mut derivative = vec![0.0; dimension];
        derivative[axis] = 1.0;
        Self { value, derivative }
    }

    fn add(&self, rhs: &Self) -> Self {
        Self {
            value: self.value + rhs.value,
            derivative: self
                .derivative
                .iter()
                .zip(&rhs.derivative)
                .map(|(a, b)| a + b)
                .collect(),
        }
    }

    fn sub(&self, rhs: &Self) -> Self {
        Self {
            value: self.value - rhs.value,
            derivative: self
                .derivative
                .iter()
                .zip(&rhs.derivative)
                .map(|(a, b)| a - b)
                .collect(),
        }
    }

    fn mul(&self, rhs: &Self) -> Self {
        Self {
            value: self.value * rhs.value,
            derivative: self
                .derivative
                .iter()
                .zip(&rhs.derivative)
                .map(|(a, b)| a * rhs.value + self.value * b)
                .collect(),
        }
    }

    fn scale(&self, factor: f64) -> Self {
        Self {
            value: self.value * factor,
            derivative: self.derivative.iter().map(|value| value * factor).collect(),
        }
    }

    fn sqrt(&self) -> Result<Self, String> {
        if !self.value.is_finite() || self.value <= 0.0 {
            return Err("singular endpoint bonded geometry".into());
        }
        let value = self.value.sqrt();
        Ok(Self {
            value,
            derivative: self
                .derivative
                .iter()
                .map(|item| item / (2.0 * value))
                .collect(),
        })
    }

    fn cos(&self) -> Self {
        Self {
            value: self.value.cos(),
            derivative: self
                .derivative
                .iter()
                .map(|item| -self.value.sin() * item)
                .collect(),
        }
    }

    fn atan2(y: &Self, x: &Self) -> Result<Self, String> {
        let denominator = x.value * x.value + y.value * y.value;
        if !denominator.is_finite() || denominator <= 1.0e-28 {
            return Err("singular endpoint angle/torsion geometry".into());
        }
        Ok(Self {
            value: y.value.atan2(x.value),
            derivative: x
                .derivative
                .iter()
                .zip(&y.derivative)
                .map(|(dx, dy)| (x.value * dy - y.value * dx) / denominator)
                .collect(),
        })
    }
}

fn jet_dot(a: &[CoordinateJet; 3], b: &[CoordinateJet; 3]) -> CoordinateJet {
    a[0].mul(&b[0]).add(&a[1].mul(&b[1])).add(&a[2].mul(&b[2]))
}

fn jet_cross(a: &[CoordinateJet; 3], b: &[CoordinateJet; 3]) -> [CoordinateJet; 3] {
    [
        a[1].mul(&b[2]).sub(&a[2].mul(&b[1])),
        a[2].mul(&b[0]).sub(&a[0].mul(&b[2])),
        a[0].mul(&b[1]).sub(&a[1].mul(&b[0])),
    ]
}

fn jet_displacement(
    atom_i: usize,
    atom_j: usize,
    positions_a: &[[f64; 3]],
    dimension: usize,
    periodic_box: Option<&PeriodicBoxContract>,
) -> Result<[CoordinateJet; 3], String> {
    let mut displacement: [CoordinateJet; 3] = std::array::from_fn(|axis| {
        CoordinateJet::variable(positions_a[atom_i][axis], dimension, atom_i * 3 + axis).sub(
            &CoordinateJet::variable(positions_a[atom_j][axis], dimension, atom_j * 3 + axis),
        )
    });
    if let Some(periodic_box) = periodic_box {
        let raw = displacement.each_ref().map(|value| value.value);
        let wrapped = minimum_image_displacement(raw, periodic_box)?;
        for axis in 0..3 {
            // The selected lattice image is locally constant. Add only its
            // translation, preserving exact coordinate derivatives.
            displacement[axis].value += wrapped[axis] - raw[axis];
        }
    }
    Ok(displacement)
}

/// Exact analytic endpoint bonded evaluation. Gradients are obtained by
/// forward-mode differentiation of the same AMBER-form energy expression;
/// finite differences are used only by tests as an independent oracle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EndpointBondedPlaneEvaluation {
    pub potential_kcal_mol: f64,
    pub forces_kcal_mol_a: Vec<[f64; 3]>,
}

/// Mechanical endpoint interpolation for one admitted state transition.
/// The reference/pH terms are carried explicitly but remain in the detailed-
/// balance target ratio, preventing double-counting them as both NCMC work and
/// reservoir/state weight.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateTransitionBondedEvaluation {
    pub potential_kcal_mol: f64,
    pub endpoint_a_potential_kcal_mol: f64,
    pub endpoint_b_potential_kcal_mol: f64,
    pub forces_kcal_mol_a: Vec<[f64; 3]>,
    pub du_dlambda_kcal_mol: f64,
    pub reference_free_energy_b_minus_a_kcal_mol: f64,
    pub delta_bound_protons_b_minus_a: i32,
}

fn bonded_angle_coordinate(ij: &[CoordinateJet;3], kj: &[CoordinateJet;3]) -> Result<CoordinateJet,String> {
    let cross=jet_cross(ij,kj);
    let sine_norm=jet_dot(&cross,&cross).sqrt()?;
    CoordinateJet::atan2(&sine_norm,&jet_dot(ij,kj))
}
fn bonded_torsion_coordinate(b1: &[CoordinateJet;3], b2: &[CoordinateJet;3], b3: &[CoordinateJet;3]) -> Result<CoordinateJet,String> {
        let n1 = jet_cross(&b1, &b2);
        let n2 = jet_cross(&b2, &b3);
        let b2_norm = jet_dot(&b2, &b2).sqrt()?;
        let cross_normals = jet_cross(&n1, &n2);
        let y_numerator = jet_dot(&cross_normals, &b2);
        let inv_b2 = CoordinateJet {
            value: b2_norm.value.recip(),
            derivative: b2_norm
                .derivative
                .iter()
                .map(|value| -value / (b2_norm.value * b2_norm.value))
                .collect(),
        };
        let y = y_numerator.mul(&inv_b2);
        CoordinateJet::atan2(&y, &jet_dot(&n1, &n2))
}

/// Coordinate-only replay of the same signed bonded geometry used by the
/// native endpoint force evaluator. Zero-dimensional jets reuse its exact
/// formula without evaluating energies or allocating force derivatives.
pub fn recorded_bonded_coordinate_observables(
    positions: &[[f64;3]], topology: &crate::input::PrismPrepTopology,
) -> anyhow::Result<serde_json::Value> {
    if positions.len()!=topology.n_atoms || positions.iter().flatten().any(|v|!v.is_finite()) {
        anyhow::bail!("recorded bonded coordinate population invalid");
    }
    let displacement=|i:usize,j:usize| -> anyhow::Result<[CoordinateJet;3]> {
        let a=positions.get(i).ok_or_else(||anyhow::anyhow!("bonded atom index out of range"))?;
        let b=positions.get(j).ok_or_else(||anyhow::anyhow!("bonded atom index out of range"))?;
        Ok(std::array::from_fn(|k|CoordinateJet::constant(a[k]-b[k],0)))
    };
    let mut unavailable=Vec::new();
    let mut bonds=Vec::with_capacity(topology.bonds.len());
    let mut angles=Vec::with_capacity(topology.angles.len());
    let mut torsions=Vec::with_capacity(topology.dihedrals.len());
    for (id,bond) in topology.bonds.iter().enumerate() {
        let d=displacement(bond.i,bond.j)?;
        match jet_dot(&d,&d).sqrt() {Ok(v)=>bonds.push(Some(v.value)),Err(reason)=>{bonds.push(None);unavailable.push(serde_json::json!({"kind":"bond","term_id":id,"reason":reason}));}}
    }
    for (id,a) in topology.angles.iter().enumerate() {
        match bonded_angle_coordinate(&displacement(a.i,a.j)?,&displacement(a.k_idx,a.j)?) {Ok(v)=>angles.push(Some(v.value)),Err(reason)=>{angles.push(None);unavailable.push(serde_json::json!({"kind":"angle","term_id":id,"reason":reason}));}}
    }
    for (id,d) in topology.dihedrals.iter().enumerate() {
        match bonded_torsion_coordinate(&displacement(d.j,d.i)?,&displacement(d.k_idx,d.j)?,&displacement(d.l,d.k_idx)?) {Ok(v)=>torsions.push(Some(v.value)),Err(reason)=>{torsions.push(None);unavailable.push(serde_json::json!({"kind":"dihedral","term_id":id,"reason":reason}));}}
    }
    Ok(serde_json::json!({"bond_lengths_angstrom":bonds,"angles_radians":angles,"signed_dihedrals_radians":torsions,"unavailable_geometry":unavailable,"native_method":"lambda_fragment endpoint bonded coordinate expressions; dimension-zero jets"}))
}

pub fn evaluate_endpoint_bonded_plane(
    positions_a: &[[f64; 3]],
    ordered_template_atom_ids: &[u32],
    plane: &FragmentBondedParameterPlane,
    periodic_box: Option<&PeriodicBoxContract>,
) -> Result<EndpointBondedPlaneEvaluation, String> {
    if positions_a.len() != ordered_template_atom_ids.len() || positions_a.is_empty() {
        return Err("endpoint bonded coordinate/atom-order mismatch".into());
    }
    if positions_a
        .iter()
        .flatten()
        .any(|coordinate| !coordinate.is_finite())
    {
        return Err("nonfinite endpoint bonded coordinate".into());
    }
    let atom_ids: HashSet<u32> = ordered_template_atom_ids.iter().copied().collect();
    if atom_ids.len() != ordered_template_atom_ids.len() {
        return Err("duplicate endpoint bonded atom identity".into());
    }
    validate_bonded_plane(plane, &atom_ids)?;
    let atom_index: HashMap<u32, usize> = ordered_template_atom_ids
        .iter()
        .copied()
        .enumerate()
        .map(|(index, id)| (id, index))
        .collect();
    let dimension = positions_a.len() * 3;
    let mut energy = CoordinateJet::constant(0.0, dimension);
    let index = |id: u32| -> Result<usize, String> {
        atom_index
            .get(&id)
            .copied()
            .ok_or_else(|| format!("bonded plane references absent atom {id}"))
    };
    for bond in &plane.bonds {
        let displacement = jet_displacement(
            index(bond.template_atom_i)?,
            index(bond.template_atom_j)?,
            positions_a,
            dimension,
            periodic_box,
        )?;
        let distance = jet_dot(&displacement, &displacement).sqrt()?;
        let delta = distance.sub(&CoordinateJet::constant(bond.length_a, dimension));
        energy = energy.add(&delta.mul(&delta).scale(bond.force_constant_kcal_mol_a2));
    }
    for angle in &plane.angles {
        let ij = jet_displacement(
            index(angle.template_atom_i)?,
            index(angle.template_atom_j)?,
            positions_a,
            dimension,
            periodic_box,
        )?;
        let kj = jet_displacement(
            index(angle.template_atom_k)?,
            index(angle.template_atom_j)?,
            positions_a,
            dimension,
            periodic_box,
        )?;
        let theta = bonded_angle_coordinate(&ij, &kj)?;
        let delta = theta.sub(&CoordinateJet::constant(angle.angle_rad, dimension));
        energy = energy.add(&delta.mul(&delta).scale(angle.force_constant_kcal_mol_rad2));
    }
    for torsion in &plane.proper_and_improper_torsions {
        let b1 = jet_displacement(
            index(torsion.template_atom_j)?,
            index(torsion.template_atom_i)?,
            positions_a,
            dimension,
            periodic_box,
        )?;
        let b2 = jet_displacement(
            index(torsion.template_atom_k)?,
            index(torsion.template_atom_j)?,
            positions_a,
            dimension,
            periodic_box,
        )?;
        let b3 = jet_displacement(
            index(torsion.template_atom_l)?,
            index(torsion.template_atom_k)?,
            positions_a,
            dimension,
            periodic_box,
        )?;
        let phi = bonded_torsion_coordinate(&b1, &b2, &b3)?;
        let argument = phi
            .scale(torsion.periodicity as f64)
            .sub(&CoordinateJet::constant(torsion.phase_rad, dimension));
        energy = energy.add(
            &argument
                .cos()
                .add(&CoordinateJet::constant(1.0, dimension))
                .scale(torsion.barrier_kcal_mol),
        );
    }
    let forces_kcal_mol_a = energy
        .derivative
        .chunks_exact(3)
        .map(|gradient| [-gradient[0], -gradient[1], -gradient[2]])
        .collect();
    Ok(EndpointBondedPlaneEvaluation {
        potential_kcal_mol: energy.value,
        forces_kcal_mol_a,
    })
}

pub fn evaluate_state_transition_bonded_hamiltonian(
    positions_a: &[[f64; 3]],
    ordered_template_atom_ids: &[u32],
    lambda: f64,
    state_a: &FragmentChemicalState,
    state_b: &FragmentChemicalState,
    transition: &FragmentStateTransition,
    periodic_box: Option<&PeriodicBoxContract>,
) -> Result<StateTransitionBondedEvaluation, String> {
    observation_eligibility(lambda)?;
    validate_state_transition(transition, state_a, state_b)?;
    let endpoint_a = evaluate_endpoint_bonded_plane(
        positions_a,
        ordered_template_atom_ids,
        &state_a.bonded_parameter_plane,
        periodic_box,
    )?;
    let endpoint_b = evaluate_endpoint_bonded_plane(
        positions_a,
        ordered_template_atom_ids,
        &state_b.bonded_parameter_plane,
        periodic_box,
    )?;
    let potential =
        (1.0 - lambda) * endpoint_a.potential_kcal_mol + lambda * endpoint_b.potential_kcal_mol;
    let forces_kcal_mol_a = endpoint_a
        .forces_kcal_mol_a
        .iter()
        .zip(&endpoint_b.forces_kcal_mol_a)
        .map(|(a, b)| std::array::from_fn(|axis| (1.0 - lambda) * a[axis] + lambda * b[axis]))
        .collect();
    Ok(StateTransitionBondedEvaluation {
        potential_kcal_mol: potential,
        endpoint_a_potential_kcal_mol: endpoint_a.potential_kcal_mol,
        endpoint_b_potential_kcal_mol: endpoint_b.potential_kcal_mol,
        forces_kcal_mol_a,
        du_dlambda_kcal_mol: endpoint_b.potential_kcal_mol - endpoint_a.potential_kcal_mol,
        reference_free_energy_b_minus_a_kcal_mol: transition
            .reference_free_energy_b_minus_a_kcal_mol,
        delta_bound_protons_b_minus_a: transition.delta_bound_protons_b_minus_a,
    })
}

/// Per-realized-path integrator probability accounting.  The variants are
/// mutually exclusive so shadow work cannot be double-counted with an exact
/// conditional path-probability ratio.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(
    tag = "accounting",
    rename_all = "SCREAMING_SNAKE_CASE",
    deny_unknown_fields
)]
pub enum NcmcIntegratorPathAccounting {
    ExactConditionalPathProbability {
        forward_log_path_probability: f64,
        reverse_log_path_probability: f64,
        log_abs_phase_space_jacobian: f64,
    },
    ShadowWorkVolumePreserving {
        shadow_work_reduced: f64,
    },
    ShadowWorkWithJacobian {
        shadow_work_reduced: f64,
        log_abs_phase_space_jacobian: f64,
    },
}

impl NcmcIntegratorPathAccounting {
    fn log_acceptance_contribution(&self) -> Result<f64, String> {
        let value = match self {
            Self::ExactConditionalPathProbability {
                forward_log_path_probability,
                reverse_log_path_probability,
                log_abs_phase_space_jacobian,
            } => {
                if !forward_log_path_probability.is_finite()
                    || !reverse_log_path_probability.is_finite()
                    || !log_abs_phase_space_jacobian.is_finite()
                {
                    return Err("nonfinite exact NCMC path-probability accounting".into());
                }
                reverse_log_path_probability - forward_log_path_probability
                    + log_abs_phase_space_jacobian
            }
            Self::ShadowWorkVolumePreserving {
                shadow_work_reduced,
            } => {
                if !shadow_work_reduced.is_finite() {
                    return Err("nonfinite NCMC shadow work".into());
                }
                -*shadow_work_reduced
            }
            Self::ShadowWorkWithJacobian {
                shadow_work_reduced,
                log_abs_phase_space_jacobian,
            } => {
                if !shadow_work_reduced.is_finite() || !log_abs_phase_space_jacobian.is_finite() {
                    return Err("nonfinite NCMC shadow-work/Jacobian accounting".into());
                }
                -*shadow_work_reduced + *log_abs_phase_space_jacobian
            }
        };
        Ok(value)
    }

    fn convention(&self) -> NcmcIntegratorPathProbabilityConvention {
        match self {
            Self::ExactConditionalPathProbability { .. } => {
                NcmcIntegratorPathProbabilityConvention::ExactConditionalPathProbability
            }
            Self::ShadowWorkVolumePreserving { .. } => {
                NcmcIntegratorPathProbabilityConvention::ShadowWorkVolumePreserving
            }
            Self::ShadowWorkWithJacobian { .. } => {
                NcmcIntegratorPathProbabilityConvention::ShadowWorkWithJacobian
            }
        }
    }
}

/// Complete NCMC move accounting. Reduced quantities are dimensionless.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NcmcMoveReceipt {
    /// Schema discriminator.
    pub schema_kind: String,
    /// Schema version.
    pub schema_version: u32,
    /// Native stream identity.
    pub stream_id: u32,
    /// Exact integration step at adjudication.
    pub step: u64,
    /// Stable move identity.
    pub move_id: u64,
    /// Strict total order within this lambda event stream.
    pub event_sequence: u64,
    /// Prior typed-emission receipt digest.
    pub parent_receipt_sha256: String,
    /// Slot being changed.
    pub slot_id: u32,
    /// Thermodynamic move class.
    pub move_kind: NcmcMoveKind,
    /// Calibration-work collection or calibrated production target-lane mode.
    pub execution_mode: NcmcExecutionMode,
    /// Exact pre-run scheduler authority that generated the move.
    pub scheduler_contract_sha256: String,
    /// Directed state-conditioned support selected by the scheduler.
    pub selected_support_id: String,
    /// Initial species/state/occupancy identity.
    pub initial: NcmcEndpointIdentity,
    /// Proposed species/state/occupancy identity.
    pub proposed: NcmcEndpointIdentity,
    /// Accumulated protocol work beta*W.
    pub protocol_work_reduced: f64,
    /// Hash-bound ordered forward/reverse operation schedule.
    pub ncmc_schedule_sha256: String,
    /// Full admitted directed protocol required for independent loader
    /// validation of operation order, lambdas and resident window lengths.
    pub ncmc_protocol: NcmcProtocol,
    /// Exact directed operation list executed for this proposal.
    pub protocol_direction: NcmcProtocolDirection,
    /// Probability of selecting the executed forward protocol.
    pub forward_protocol_probability: f64,
    /// Probability of selecting the exact reverse protocol.
    pub reverse_protocol_probability: f64,
    /// `ln(p_reverse_protocol/p_forward_protocol)`.
    pub log_protocol_probability_ratio: f64,
    /// Conditional integrator path action, shadow work and Jacobian as
    /// appropriate for the declared mutually exclusive convention.
    pub integrator_path_accounting: NcmcIntegratorPathAccounting,
    /// Derived integrator conditional-path/shadow-work/Jacobian contribution.
    pub integrator_log_acceptance_contribution: f64,
    /// True only when the declared reverse/rejection momentum involution was
    /// actually applied by the device transaction.
    pub momentum_reversal_applied: bool,
    /// Both physical endpoint equilibrium target densities were evaluated.
    pub endpoint_equilibrium_densities_evaluated: bool,
    /// Log target-density ratio excluding NCMC work. This includes chemical
    /// potential, standard-state/Jacobian, pH and combinatorial terms.
    pub log_target_ratio: f64,
    /// log(q(reverse)/q(forward)), including placement and identity choices.
    pub log_proposal_ratio: f64,
    /// Exact raw target/proposal inputs authenticated by the move receipt.
    /// Production calibration may not reconstruct these from a scalar ratio.
    pub proposal_context: NcmcBalanceContext,
    /// Base measures and units for the raw proposal factors above.
    pub proposal_measure_contract: NcmcProposalMeasureContract,
    /// Term-resolved engine derivation of both ratios.
    pub derived_balance: NcmcDerivedBalance,
    /// Exact simultaneous counterion/co-ion move, when one was used.
    pub coupled_charge_partner: Option<NcmcCoupledChargePartner>,
    /// Hamiltonian digest proving the exact soft-core/Ewald/charge-treatment
    /// convention under which the move was executed.
    pub hamiltonian_sha256: String,
    /// Package-time scheduler-support to fixed endpoint-pair binding admitted
    /// by the runtime that executed this move.
    pub endpoint_support_binding_sha256: String,
    /// Digest of the exact resident uploaded dual-endpoint union contract used
    /// by every force boundary in this move.
    pub dual_endpoint_union_contract_sha256: String,
    /// Box contract used by any periodic finite-size correction.
    pub periodic_box_source_sha256: String,
    /// PME parameterization used by any periodic finite-size correction.
    pub pme_parameterization_sha256: String,
    /// Computed Metropolis log acceptance, `min(0, ...)`.
    pub log_acceptance: f64,
    /// Uniform variate used for accept/reject.
    pub uniform_random: f64,
    /// Recorded decision.
    pub accepted: bool,
    /// Counter-based RNG key/counter for deterministic replay.
    pub rng_key: u64,
    /// Counter-based RNG counter.
    #[serde(with = "u128_decimal_string")]
    pub rng_counter: u128,
    /// Complete disjoint RNG subrange allocation for reproducible replay.
    pub rng_allocation: NcmcMoveRngAllocation,
    /// Typed physical use/non-use for every reserved subrange. Reservation is
    /// irrevocable after rejection and therefore cannot be rewound with the
    /// device phase-space checkpoint.
    pub rng_consumption: NcmcRngConsumptionReceipt,
    /// Full operation-level work and force-boundary byte lineage.  Persisting
    /// the object, not only its digest, makes adjudication reconstructable.
    pub ordered_work_trace: NcmcOrderedWorkTrace,
    /// Hash of the ordered per-step work trace.
    pub work_trace_sha256: String,
    /// Canonical phase/condition binding digest for this exact step.
    pub phase_protocol_sha256: String,
    /// Canonical PrismCode opcode for accept/reject adjudication.
    pub prismcode_opcode: String,
    /// Numeric PrismCode opcode bound by the canonical registry.
    pub prismcode_opcode_id: u16,
    /// Canonical QNANL family.
    pub qnanl_artifact_family: String,
}

/// Grand-canonical and chemical-state move classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum NcmcMoveKind {
    /// Empty to occupied.
    Insertion,
    /// Occupied to empty.
    Deletion,
    /// Species replacement with stable slot/molecule identity.
    Identity,
    /// Tautomer state change.
    Tautomer,
    /// Protonation state change.
    Protonation,
}

/// Endpoint state in an NCMC receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NcmcEndpointIdentity {
    /// Species identity.
    pub species_id: u32,
    /// Chemical state identity.
    pub chemical_state_id: u32,
    /// Physical endpoint occupancy.
    pub occupancy: EndpointOccupancy,
}

impl NcmcMoveReceipt {
    fn proposal_projection(&self) -> NcmcMoveProposal {
        NcmcMoveProposal {
            move_id: self.move_id,
            slot_id: self.slot_id,
            move_kind: self.move_kind,
            proposed_species_id: self.proposed.species_id,
            proposed_chemical_state_id: self.proposed.chemical_state_id,
            proposed_occupancy: self.proposed.occupancy,
            execution_mode: self.execution_mode.clone(),
            scheduler_contract_sha256: self.scheduler_contract_sha256.clone(),
            selected_support_id: self.selected_support_id.clone(),
            balance_context: self.proposal_context.clone(),
            periodic_charge_treatment: self.derived_balance.periodic_charge_treatment.clone(),
            coupled_charge_partner: self.coupled_charge_partner.clone(),
            rng_key: self.rng_key,
            rng_counter: self.rng_counter,
            rng_allocation: self.rng_allocation.clone(),
        }
    }

    /// Recompute and verify detailed-balance acceptance bookkeeping.
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_kind != "prismos_lambda_fragment_ncmc_move_receipt"
            || self.schema_version != 6
        {
            return Err("invalid typed NCMC receipt contract".into());
        }
        validate_dispatch_binding(
            &self.prismcode_opcode,
            self.prismcode_opcode_id,
            prismcode_opcodes::NCMC_ACCEPT,
            &self.qnanl_artifact_family,
            QnanlArtifactFamily::LambdaFragmentNcmcMoveReceipt,
        )?;
        self.execution_mode.validate()?;
        validate_sha256("NCMC receipt scheduler", &self.scheduler_contract_sha256)?;
        if self.selected_support_id.trim().is_empty()
            || self.rng_counter != self.rng_allocation.acceptance_counter
        {
            return Err("NCMC receipt lacks scheduler-support or RNG-allocation identity".into());
        }
        self.rng_allocation.validate_disjoint()?;
        self.rng_consumption
            .validate_for_proposal(&self.proposal_projection())?;
        for (name, value) in [
            ("protocol_work_reduced", self.protocol_work_reduced),
            (
                "forward_protocol_probability",
                self.forward_protocol_probability,
            ),
            (
                "reverse_protocol_probability",
                self.reverse_protocol_probability,
            ),
            (
                "log_protocol_probability_ratio",
                self.log_protocol_probability_ratio,
            ),
            (
                "integrator_log_acceptance_contribution",
                self.integrator_log_acceptance_contribution,
            ),
            ("log_target_ratio", self.log_target_ratio),
            ("log_proposal_ratio", self.log_proposal_ratio),
            ("log_acceptance", self.log_acceptance),
            ("uniform_random", self.uniform_random),
        ] {
            if !value.is_finite() {
                return Err(format!("{name} must be finite"));
            }
        }
        if !(0.0..1.0).contains(&self.uniform_random) {
            return Err("uniform_random must lie in [0,1)".to_string());
        }
        if self.uniform_random.to_bits()
            != counter_based_uniform_open01(self.rng_key, self.rng_counter).to_bits()
        {
            return Err("NCMC receipt random variate differs from its key/counter".into());
        }
        if !(0.0..=1.0).contains(&self.forward_protocol_probability)
            || self.forward_protocol_probability == 0.0
            || !(0.0..=1.0).contains(&self.reverse_protocol_probability)
            || self.reverse_protocol_probability == 0.0
            || !self.momentum_reversal_applied
            || !self.endpoint_equilibrium_densities_evaluated
        {
            return Err("NCMC protocol selection/momentum/endpoint evidence is incomplete".into());
        }
        let expected_protocol_ratio = match self.protocol_direction {
            NcmcProtocolDirection::Forward => {
                (self.reverse_protocol_probability / self.forward_protocol_probability).ln()
            }
            NcmcProtocolDirection::Reverse => {
                (self.forward_protocol_probability / self.reverse_protocol_probability).ln()
            }
        };
        if (self.log_protocol_probability_ratio - expected_protocol_ratio).abs() > 1.0e-12 {
            return Err("NCMC protocol selection probability ratio mismatch".into());
        }
        let integrator_contribution = self
            .integrator_path_accounting
            .log_acceptance_contribution()?;
        if (self.integrator_log_acceptance_contribution - integrator_contribution).abs() > 1.0e-12 {
            return Err("NCMC integrator path-accounting contribution mismatch".into());
        }
        validate_sha256("NCMC schedule", &self.ncmc_schedule_sha256)?;
        validate_ncmc(&self.ncmc_protocol)?;
        if self.ncmc_protocol.schedule_sha256 != self.ncmc_schedule_sha256 {
            return Err("NCMC receipt protocol object differs from schedule digest".into());
        }
        validate_sha256("work_trace_sha256", &self.work_trace_sha256)?;
        if self.ordered_work_trace.move_id != self.move_id
            || self.ordered_work_trace.schedule_sha256 != self.ncmc_schedule_sha256
            || self.ordered_work_trace.protocol_direction != self.protocol_direction
            || self.ordered_work_trace.trace_sha256 != self.work_trace_sha256
            || (self.ordered_work_trace.perturbation_work_reduced - self.protocol_work_reduced)
                .abs()
                > 1.0e-12
        {
            return Err("NCMC receipt work-trace identity mismatch".into());
        }
        self.ordered_work_trace.validate(&self.ncmc_protocol)?;
        match &self.integrator_path_accounting {
            NcmcIntegratorPathAccounting::ShadowWorkVolumePreserving {
                shadow_work_reduced,
            }
            | NcmcIntegratorPathAccounting::ShadowWorkWithJacobian {
                shadow_work_reduced,
                ..
            } if (*shadow_work_reduced - self.ordered_work_trace.shadow_work_reduced).abs()
                <= 1.0e-12 => {}
            NcmcIntegratorPathAccounting::ExactConditionalPathProbability { .. } => {}
            _ => return Err("NCMC receipt shadow work differs from ordered trace".into()),
        }
        validate_sha256("lambda Hamiltonian", &self.hamiltonian_sha256)?;
        validate_sha256(
            "endpoint support binding",
            &self.endpoint_support_binding_sha256,
        )?;
        validate_sha256(
            "dual endpoint union contract",
            &self.dual_endpoint_union_contract_sha256,
        )?;
        validate_sha256("periodic box source", &self.periodic_box_source_sha256)?;
        validate_sha256("PME parameterization", &self.pme_parameterization_sha256)?;
        validate_sha256("phase/condition binding", &self.phase_protocol_sha256)?;
        validate_sha256(
            "balance derivation",
            &self.derived_balance.derivation_sha256,
        )?;
        self.derived_balance.validate_terms()?;
        match (
            &self.derived_balance.periodic_charge_treatment,
            &self.coupled_charge_partner,
        ) {
            (
                PeriodicChargeTreatment::CoupledChargeNeutralPath {
                    partner_slot_id,
                    partner_charge_change_e,
                    ..
                },
                Some(partner),
            ) if partner.slot_id == *partner_slot_id
                && partner.charge_change_e == *partner_charge_change_e =>
            {
                partner.validate()?;
            }
            (PeriodicChargeTreatment::CoupledChargeNeutralPath { .. }, _) => {
                return Err("NCMC receipt omits coupled charge-partner move".into())
            }
            (_, Some(_)) => return Err("uncoupled NCMC receipt carries a charge partner".into()),
            (_, None) => {}
        }
        if let PeriodicChargeTreatment::ExplicitFiniteSizeCorrection {
            production_box_source_sha256,
            pme_parameterization_sha256,
            ..
        } = &self.derived_balance.periodic_charge_treatment
        {
            if production_box_source_sha256 != &self.periodic_box_source_sha256
                || pme_parameterization_sha256 != &self.pme_parameterization_sha256
            {
                return Err(
                    "NCMC finite-size correction is not bound to receipt box and PME".into(),
                );
            }
        }
        if (self.log_target_ratio - self.derived_balance.log_target_ratio).abs() > 1.0e-12
            || (self.log_proposal_ratio - self.derived_balance.log_proposal_ratio).abs() > 1.0e-12
        {
            return Err("NCMC scalar ratios do not match derived balance terms".into());
        }
        if self.execution_mode.is_calibration_work_collection()
            && (self.derived_balance.beta_chemical_potential_term.to_bits() != 0.0f64.to_bits()
                || self
                    .derived_balance
                    .beta_intrinsic_state_reference_term_in_total_mu
                    .to_bits()
                    != 0.0f64.to_bits()
                || self
                    .derived_balance
                    .beta_periodic_finite_size_correction_term_in_total_mu
                    .to_bits()
                    != 0.0f64.to_bits()
                || self.derived_balance.ph_log_weight_term.to_bits() != 0.0f64.to_bits()
                || self.derived_balance.standard_state_volume_term.to_bits() != 0.0f64.to_bits()
                || self.derived_balance.occupancy_combinatorics_term.to_bits() != 0.0f64.to_bits()
                || self.derived_balance.target_bias_term.to_bits() != 0.0f64.to_bits()
                || self.log_target_ratio.to_bits() != 0.0f64.to_bits())
        {
            return Err("calibration-work NCMC receipt contains circular target bias".into());
        }
        self.proposal_context.validate()?;
        self.proposal_measure_contract.validate()?;
        let selection = (self.proposal_context.reverse_selection_probability
            / self.proposal_context.forward_selection_probability)
            .ln();
        let placement = (self.proposal_context.reverse_placement_density_a3_inv
            / self.proposal_context.forward_placement_density_a3_inv)
            .ln();
        let orientation = (self.proposal_context.reverse_orientation_density
            / self.proposal_context.forward_orientation_density)
            .ln();
        let expected_beta =
            1.0 / (0.001_987_204_258_640_83_f64 * self.proposal_context.temperature_k);
        if (selection - self.derived_balance.selection_proposal_term).abs() > 1.0e-12
            || (placement - self.derived_balance.placement_proposal_term).abs() > 1.0e-12
            || (orientation - self.derived_balance.orientation_proposal_term).abs() > 1.0e-12
            || (selection + placement + orientation - self.log_proposal_ratio).abs() > 1.0e-12
            || (expected_beta - self.derived_balance.beta_mol_per_kcal).abs() > 1.0e-12
        {
            return Err(
                "NCMC raw proposal context does not reproduce the authenticated balance".into(),
            );
        }
        validate_sha256("parent receipt", &self.parent_receipt_sha256)?;
        if matches!(self.initial.occupancy, EndpointOccupancy::Switching)
            || matches!(self.proposed.occupancy, EndpointOccupancy::Switching)
        {
            return Err("NCMC receipt endpoints cannot be SWITCHING".to_string());
        }
        let raw = -self.protocol_work_reduced
            + self.log_target_ratio
            + self.log_proposal_ratio
            + self.log_protocol_probability_ratio
            + self.integrator_log_acceptance_contribution;
        let expected = raw.min(0.0);
        if (self.log_acceptance - expected).abs() > 1.0e-12 {
            return Err(format!(
                "NCMC log acceptance mismatch: recorded={} expected={}",
                self.log_acceptance, expected
            ));
        }
        let expected_decision = self.uniform_random.ln() < expected;
        if self.accepted != expected_decision {
            return Err("NCMC accept/reject decision mismatches recorded random variate".into());
        }
        Ok(())
    }

    /// Stable hash binding all scientifically relevant receipt fields.
    pub fn digest_sha256(&self) -> Result<String, String> {
        self.validate()?;
        let bytes = serde_json::to_vec(self).map_err(|e| e.to_string())?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }
}

/// Deterministic counter-based uniform variate in `[0,1)`.  This is a
/// stateless SplitMix64 bijection over the complete key/counter tuple; replay
/// requires no mutable host RNG state.
pub fn counter_based_uniform_open01(key: u64, counter: u128) -> f64 {
    fn mix(mut z: u64) -> u64 {
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    let lo = counter as u64;
    let hi = (counter >> 64) as u64;
    let bits = mix(key ^ mix(lo) ^ mix(hi.rotate_left(29)));
    // Use the top 53 bits and add half a bin so neither endpoint is possible.
    (((bits >> 11) as f64) + 0.5) * (1.0 / 9_007_199_254_740_992.0)
}

/// Resolve a state-conditioned support onto the fixed slot's A/B orientation.
/// This is the sole authority for selecting the forward versus reverse
/// operation list and prevents a B-to-A deletion/state move from being
/// integrated with the A-to-B schedule by accident.
pub fn ncmc_protocol_direction_for_move(
    system: &LambdaFragmentSystem,
    proposal: &NcmcMoveProposal,
    initial: &NcmcEndpointIdentity,
) -> Result<NcmcProtocolDirection, String> {
    let slot = system
        .slots
        .iter()
        .find(|slot| slot.slot_id == proposal.slot_id)
        .ok_or_else(|| "NCMC proposal slot is absent from the fragment system".to_string())?;
    let proposed = NcmcEndpointIdentity {
        species_id: proposal.proposed_species_id,
        chemical_state_id: proposal.proposed_chemical_state_id,
        occupancy: proposal.proposed_occupancy,
    };
    if initial == &slot.endpoint_a && proposed == slot.endpoint_b {
        Ok(NcmcProtocolDirection::Forward)
    } else if initial == &slot.endpoint_b && proposed == slot.endpoint_a {
        Ok(NcmcProtocolDirection::Reverse)
    } else {
        Err("NCMC directed support does not match either fixed slot endpoint path".into())
    }
}

/// Construct a validated acceptance receipt from exact accumulated reduced
/// work.  The caller supplies the ordered work-trace hash produced at the
/// device/host move boundary.
pub fn adjudicate_ncmc_move(
    system: &LambdaFragmentSystem,
    proposal: &NcmcMoveProposal,
    balance: &NcmcDerivedBalance,
    stream_id: u32,
    step: u64,
    phase_protocol_sha256: String,
    event_sequence: u64,
    parent_receipt_sha256: String,
    endpoint_support_binding_sha256: String,
    dual_endpoint_union_contract_sha256: String,
    initial: NcmcEndpointIdentity,
    protocol_work_reduced: f64,
    integrator_path_accounting: NcmcIntegratorPathAccounting,
    momentum_reversal_applied: bool,
    endpoint_equilibrium_densities_evaluated: bool,
    ordered_work_trace: NcmcOrderedWorkTrace,
    rng_consumption: NcmcRngConsumptionReceipt,
) -> Result<NcmcMoveReceipt, String> {
    proposal.validate()?;
    validate_ncmc(&system.ncmc)?;
    validate_sha256("parent receipt", &parent_receipt_sha256)?;
    validate_sha256("phase/condition binding", &phase_protocol_sha256)?;
    validate_sha256("endpoint support binding", &endpoint_support_binding_sha256)?;
    validate_sha256(
        "dual endpoint union contract",
        &dual_endpoint_union_contract_sha256,
    )?;
    ordered_work_trace.validate(&system.ncmc)?;
    if !protocol_work_reduced.is_finite() {
        return Err("NCMC protocol work must be finite".into());
    }
    rng_consumption.validate_for_proposal(proposal)?;
    if integrator_path_accounting.convention() != system.ncmc.integrator_path_probability_convention
    {
        return Err("NCMC realized path accounting disagrees with protocol convention".into());
    }
    let integrator_contribution = integrator_path_accounting.log_acceptance_contribution()?;
    let protocol_direction = ncmc_protocol_direction_for_move(system, proposal, &initial)?;
    let log_protocol_probability_ratio = match protocol_direction {
        NcmcProtocolDirection::Forward => (system.ncmc.reverse_protocol_probability
            / system.ncmc.forward_protocol_probability)
            .ln(),
        NcmcProtocolDirection::Reverse => (system.ncmc.forward_protocol_probability
            / system.ncmc.reverse_protocol_probability)
            .ln(),
    };
    let uniform_random = counter_based_uniform_open01(proposal.rng_key, proposal.rng_counter);
    let raw = -protocol_work_reduced
        + balance.log_target_ratio
        + balance.log_proposal_ratio
        + log_protocol_probability_ratio
        + integrator_contribution;
    let log_acceptance = raw.min(0.0);
    let receipt = NcmcMoveReceipt {
        schema_kind: "prismos_lambda_fragment_ncmc_move_receipt".into(),
        schema_version: 6,
        stream_id,
        step,
        move_id: proposal.move_id,
        event_sequence,
        parent_receipt_sha256,
        slot_id: proposal.slot_id,
        move_kind: proposal.move_kind,
        execution_mode: proposal.execution_mode.clone(),
        scheduler_contract_sha256: proposal.scheduler_contract_sha256.clone(),
        selected_support_id: proposal.selected_support_id.clone(),
        initial,
        proposed: NcmcEndpointIdentity {
            species_id: proposal.proposed_species_id,
            chemical_state_id: proposal.proposed_chemical_state_id,
            occupancy: proposal.proposed_occupancy,
        },
        protocol_work_reduced,
        ncmc_schedule_sha256: system.ncmc.schedule_sha256.clone(),
        ncmc_protocol: system.ncmc.clone(),
        protocol_direction,
        forward_protocol_probability: system.ncmc.forward_protocol_probability,
        reverse_protocol_probability: system.ncmc.reverse_protocol_probability,
        log_protocol_probability_ratio,
        integrator_path_accounting,
        integrator_log_acceptance_contribution: integrator_contribution,
        momentum_reversal_applied,
        endpoint_equilibrium_densities_evaluated,
        log_target_ratio: balance.log_target_ratio,
        log_proposal_ratio: balance.log_proposal_ratio,
        proposal_context: proposal.balance_context.clone(),
        proposal_measure_contract: NcmcProposalMeasureContract::canonical(),
        derived_balance: balance.clone(),
        coupled_charge_partner: proposal.coupled_charge_partner.clone(),
        hamiltonian_sha256: system.softcore.digest_sha256()?,
        endpoint_support_binding_sha256,
        dual_endpoint_union_contract_sha256,
        periodic_box_source_sha256: system.periodic_box.source_sha256.clone(),
        pme_parameterization_sha256: system.pme.pme_parameterization_sha256.clone(),
        log_acceptance,
        uniform_random,
        accepted: uniform_random.ln() < log_acceptance,
        rng_key: proposal.rng_key,
        rng_counter: proposal.rng_counter,
        rng_allocation: proposal.rng_allocation.clone(),
        rng_consumption,
        work_trace_sha256: ordered_work_trace.trace_sha256.clone(),
        ordered_work_trace,
        phase_protocol_sha256,
        prismcode_opcode: prismcode_opcodes::NCMC_ACCEPT.into(),
        prismcode_opcode_id: opcode_id_for_name(prismcode_opcodes::NCMC_ACCEPT)
            .ok_or_else(|| "PrismCode NCMC accept opcode missing".to_string())?,
        qnanl_artifact_family: QnanlArtifactFamily::LambdaFragmentNcmcMoveReceipt
            .as_str()
            .into(),
    };
    receipt.validate()?;
    Ok(receipt)
}

/// Derive all detailed-balance target/proposal terms from calibrated system
/// data and non-log proposal densities.
pub fn derive_ncmc_balance(
    system: &LambdaFragmentSystem,
    proposal: &NcmcMoveProposal,
    initial: &NcmcEndpointIdentity,
) -> Result<NcmcDerivedBalance, String> {
    proposal.validate()?;
    let context = &proposal.balance_context;
    const KB_KCAL_MOL_K: f64 = 0.001_987_204_258_640_83;
    const MOLAR_STANDARD_VOLUME_A3: f64 = 1660.539_067_173_846_6;
    let beta = 1.0 / (KB_KCAL_MOL_K * context.temperature_k);
    let find_state = |species_id: u32, state_id: u32| {
        system
            .species
            .iter()
            .find(|item| item.species_id == species_id)
            .and_then(|item| {
                item.chemical_states
                    .iter()
                    .find(|state| state.state_id == state_id)
            })
            .ok_or_else(|| format!("absent lambda chemical state {species_id}:{state_id}"))
    };
    let find_calibration = |species_id: u32, state_id: u32| {
        system
            .chemical_potentials
            .iter()
            .find(|calibration| {
                calibration.species_id == species_id
                    && calibration.chemical_state_id == state_id
                    && (calibration.temperature_k - context.temperature_k).abs() <= 1.0e-6
                    && calibration.state_reference_convention
                        == StateReferenceConvention::TotalMuIncludesIntrinsicStateFreeEnergy
            })
            .ok_or_else(|| {
                format!(
                    "no chemical-potential calibration at {} K for {}:{}",
                    context.temperature_k, species_id, state_id
                )
            })
    };
    let proposed_state = find_state(
        proposal.proposed_species_id,
        proposal.proposed_chemical_state_id,
    )?;
    if proposal.execution_mode.is_calibration_work_collection() {
        if !system.chemical_potentials.is_empty() {
            return Err(
                "calibration-work collection must not consume chemical-potential calibrations"
                    .into(),
            );
        }
        if context.initial_bias_weight.to_bits() != 1.0f64.to_bits()
            || context.proposed_bias_weight.to_bits() != 1.0f64.to_bits()
        {
            return Err("calibration-work collection forbids target-bias weights".into());
        }
        let expected_charge_change_e = match proposal.move_kind {
            NcmcMoveKind::Insertion => {
                if initial.occupancy != EndpointOccupancy::Empty
                    || proposal.proposed_occupancy != EndpointOccupancy::Occupied
                {
                    return Err("calibration-work insertion must switch EMPTY to OCCUPIED".into());
                }
                proposed_state.formal_charge_e
            }
            NcmcMoveKind::Deletion => {
                if initial.occupancy != EndpointOccupancy::Occupied
                    || proposal.proposed_occupancy != EndpointOccupancy::Empty
                {
                    return Err("calibration-work deletion must switch OCCUPIED to EMPTY".into());
                }
                -find_state(initial.species_id, initial.chemical_state_id)?.formal_charge_e
            }
            _ => {
                return Err(
                    "calibration-work collection admits only bulk insertion/deletion switching"
                        .into(),
                )
            }
        };
        if matches!(
            proposal.periodic_charge_treatment,
            PeriodicChargeTreatment::ExplicitFiniteSizeCorrection { .. }
        ) {
            return Err(
                "calibration-work collection forbids a pre-calibrated finite-size free-energy correction"
                    .into(),
            );
        }
        proposal.periodic_charge_treatment.validate_for_move(
            expected_charge_change_e,
            0.0,
            &system.periodic_box,
            &system.pme,
        )?;
        match (
            &proposal.periodic_charge_treatment,
            &proposal.coupled_charge_partner,
        ) {
            (
                PeriodicChargeTreatment::CoupledChargeNeutralPath {
                    partner_charge_change_e,
                    ..
                },
                Some(partner),
            ) if partner.charge_change_e == *partner_charge_change_e => {
                partner.validate()?;
                if partner.slot_id == proposal.slot_id {
                    return Err("NCMC charge partner cannot be the subject slot".into());
                }
                let partner_slot = system
                    .slots
                    .iter()
                    .find(|slot| slot.slot_id == partner.slot_id)
                    .ok_or_else(|| "coupled charge-partner slot is absent".to_string())?;
                let endpoint_pair_matches = (partner.initial == partner_slot.endpoint_a
                    && partner.proposed == partner_slot.endpoint_b)
                    || (partner.initial == partner_slot.endpoint_b
                        && partner.proposed == partner_slot.endpoint_a);
                if !endpoint_pair_matches || partner.charge_change_e != -expected_charge_change_e {
                    return Err(
                        "calibration-work coupled partner does not close the endpoint charge path"
                            .into(),
                    );
                }
            }
            (PeriodicChargeTreatment::CoupledChargeNeutralPath { .. }, _) => {
                return Err("calibration-work coupled charge path lacks its partner move".into())
            }
            (_, Some(_)) => {
                return Err("calibration-work uncoupled move carries a charge partner".into())
            }
            (_, None) => {}
        }
        let selection_proposal_term =
            (context.reverse_selection_probability / context.forward_selection_probability).ln();
        let placement_proposal_term = (context.reverse_placement_density_a3_inv
            / context.forward_placement_density_a3_inv)
            .ln();
        let orientation_proposal_term =
            (context.reverse_orientation_density / context.forward_orientation_density).ln();
        let log_proposal_ratio =
            selection_proposal_term + placement_proposal_term + orientation_proposal_term;
        let mut digest = Sha256::new();
        digest.update(b"PRISM_NCMC_DERIVED_BALANCE_V2\0");
        digest.update(
            serde_json::to_vec(&proposal.execution_mode).map_err(|error| error.to_string())?,
        );
        digest.update(serde_json::to_vec(context).map_err(|error| error.to_string())?);
        digest.update(initial.species_id.to_le_bytes());
        digest.update(initial.chemical_state_id.to_le_bytes());
        digest.update(proposal.proposed_species_id.to_le_bytes());
        digest.update(proposal.proposed_chemical_state_id.to_le_bytes());
        digest.update(
            serde_json::to_vec(&proposal.periodic_charge_treatment)
                .map_err(|error| error.to_string())?,
        );
        let balance = NcmcDerivedBalance {
            beta_mol_per_kcal: beta,
            beta_chemical_potential_term: 0.0,
            beta_intrinsic_state_reference_term_in_total_mu: 0.0,
            beta_nonreference_chemical_potential_term: 0.0,
            beta_periodic_finite_size_correction_term_in_total_mu: 0.0,
            beta_calibrated_mu_remainder_term: 0.0,
            periodic_charge_treatment: proposal.periodic_charge_treatment.clone(),
            state_reference_convention:
                StateReferenceConvention::TotalMuIncludesIntrinsicStateFreeEnergy,
            ph_log_weight_term: 0.0,
            standard_state_volume_term: 0.0,
            occupancy_combinatorics_term: 0.0,
            target_bias_term: 0.0,
            log_target_ratio: 0.0,
            selection_proposal_term,
            placement_proposal_term,
            orientation_proposal_term,
            log_proposal_ratio,
            derivation_sha256: format!("{:x}", digest.finalize()),
        };
        balance.validate_terms()?;
        return Ok(balance);
    }
    let proposed_cal = find_calibration(
        proposal.proposed_species_id,
        proposal.proposed_chemical_state_id,
    )?;
    let (
        delta_mu,
        delta_intrinsic_state_reference,
        delta_periodic_finite_size_correction,
        delta_ph_weight,
        standard_state_volume_term,
        occupancy_term,
        expected_charge_change_e,
        expected_charge_treatment,
    ) = match proposal.move_kind {
        NcmcMoveKind::Insertion => {
            let standard_volume =
                MOLAR_STANDARD_VOLUME_A3 / proposed_cal.standard_concentration_molar;
            (
                proposed_cal.total_chemical_potential_kcal_mol,
                proposed_cal.included_intrinsic_state_free_energy_kcal_mol,
                proposed_cal.included_finite_size_correction_kcal_mol,
                -(10.0f64.ln())
                    * proposed_state.proton_count as f64
                    * (context.ph - proposed_cal.ph),
                (context.accessible_volume_a3 / standard_volume).ln(),
                // Persistent fixed slots are labeled coordinates. Their
                // exact state-conditioned selection probabilities already
                // carry the eligible empty/occupied population ratio; adding
                // a second N! term here would double count combinatorics.
                0.0,
                proposed_state.formal_charge_e,
                proposed_cal.periodic_charge_treatment.clone(),
            )
        }
        NcmcMoveKind::Deletion => {
            let state = find_state(initial.species_id, initial.chemical_state_id)?;
            let calibration = find_calibration(initial.species_id, initial.chemical_state_id)?;
            let standard_volume =
                MOLAR_STANDARD_VOLUME_A3 / calibration.standard_concentration_molar;
            (
                -calibration.total_chemical_potential_kcal_mol,
                -calibration.included_intrinsic_state_free_energy_kcal_mol,
                -calibration.included_finite_size_correction_kcal_mol,
                (10.0f64.ln()) * state.proton_count as f64 * (context.ph - calibration.ph),
                (standard_volume / context.accessible_volume_a3).ln(),
                0.0,
                -state.formal_charge_e,
                calibration.periodic_charge_treatment.reversed(),
            )
        }
        NcmcMoveKind::Identity | NcmcMoveKind::Tautomer | NcmcMoveKind::Protonation => {
            let initial_state = find_state(initial.species_id, initial.chemical_state_id)?;
            let initial_cal = find_calibration(initial.species_id, initial.chemical_state_id)?;
            let declared_transition_treatment =
                if matches!(
                    proposal.move_kind,
                    NcmcMoveKind::Tautomer | NcmcMoveKind::Protonation
                ) {
                    if initial.species_id != proposal.proposed_species_id {
                        return Err("tautomer/protonation move cannot change species".into());
                    }
                    let species = system
                        .species
                        .iter()
                        .find(|item| item.species_id == initial.species_id)
                        .ok_or_else(|| "state-transform species absent".to_string())?;
                    let forward = species.state_transitions.iter().find(|transition| {
                        transition.state_a_id == initial.chemical_state_id
                            && transition.state_b_id == proposal.proposed_chemical_state_id
                    });
                    let reverse = species.state_transitions.iter().find(|transition| {
                        transition.state_b_id == initial.chemical_state_id
                            && transition.state_a_id == proposal.proposed_chemical_state_id
                    });
                    match (forward, reverse) {
                        (Some(transition), None) => transition.periodic_charge_treatment.clone(),
                        (None, Some(transition)) => transition.periodic_charge_treatment.reversed(),
                        _ => return Err(
                            "tautomer/protonation move lacks one exact directed state transition"
                                .into(),
                        ),
                    }
                } else {
                    proposal.periodic_charge_treatment.clone()
                };
            (
                proposed_cal.total_chemical_potential_kcal_mol
                    - initial_cal.total_chemical_potential_kcal_mol,
                proposed_cal.included_intrinsic_state_free_energy_kcal_mol
                    - initial_cal.included_intrinsic_state_free_energy_kcal_mol,
                proposed_cal.included_finite_size_correction_kcal_mol
                    - initial_cal.included_finite_size_correction_kcal_mol,
                -(10.0f64.ln())
                    * (proposed_state.proton_count as f64 * (context.ph - proposed_cal.ph)
                        - initial_state.proton_count as f64 * (context.ph - initial_cal.ph)),
                0.0,
                0.0,
                proposed_state.formal_charge_e - initial_state.formal_charge_e,
                declared_transition_treatment,
            )
        }
    };
    if proposal.periodic_charge_treatment != expected_charge_treatment {
        return Err(
            "NCMC charge treatment does not match the directed calibrated endpoint path".into(),
        );
    }
    proposal.periodic_charge_treatment.validate_for_move(
        expected_charge_change_e,
        delta_periodic_finite_size_correction,
        &system.periodic_box,
        &system.pme,
    )?;
    if let Some(partner_move) = &proposal.coupled_charge_partner {
        if partner_move.slot_id == proposal.slot_id {
            return Err("NCMC charge partner cannot be the subject slot".into());
        }
        let partner_slot = system
            .slots
            .iter()
            .find(|slot| slot.slot_id == partner_move.slot_id)
            .ok_or_else(|| "coupled charge-partner slot is absent".to_string())?;
        let endpoint_pair_matches = (partner_move.initial == partner_slot.endpoint_a
            && partner_move.proposed == partner_slot.endpoint_b)
            || (partner_move.initial == partner_slot.endpoint_b
                && partner_move.proposed == partner_slot.endpoint_a);
        if !endpoint_pair_matches {
            return Err(
                "coupled charge-partner move does not follow its declared slot path".into(),
            );
        }
        let endpoint_charge = |endpoint: &NcmcEndpointIdentity| -> Result<i32, String> {
            match endpoint.occupancy {
                EndpointOccupancy::Empty => Ok(0),
                EndpointOccupancy::Occupied => Ok(find_state(
                    endpoint.species_id,
                    endpoint.chemical_state_id,
                )?
                .formal_charge_e),
                EndpointOccupancy::Switching => {
                    Err("coupled charge-partner endpoint is nonphysical".into())
                }
            }
        };
        let measured_partner_delta =
            endpoint_charge(&partner_move.proposed)? - endpoint_charge(&partner_move.initial)?;
        if measured_partner_delta != partner_move.charge_change_e
            || measured_partner_delta != -expected_charge_change_e
        {
            return Err("coupled charge-partner endpoint charges do not close to zero".into());
        }
    }
    let beta_chemical_potential_term = beta * delta_mu;
    let beta_intrinsic_state_reference_term_in_total_mu = beta * delta_intrinsic_state_reference;
    let beta_nonreference_chemical_potential_term =
        beta_chemical_potential_term - beta_intrinsic_state_reference_term_in_total_mu;
    let beta_periodic_finite_size_correction_term_in_total_mu =
        beta * delta_periodic_finite_size_correction;
    let beta_calibrated_mu_remainder_term = beta_nonreference_chemical_potential_term
        - beta_periodic_finite_size_correction_term_in_total_mu;
    let target_bias_term = (context.proposed_bias_weight / context.initial_bias_weight).ln();
    let log_target_ratio = beta_chemical_potential_term
        + delta_ph_weight
        + standard_state_volume_term
        + occupancy_term
        + target_bias_term;
    let selection_proposal_term =
        (context.reverse_selection_probability / context.forward_selection_probability).ln();
    let placement_proposal_term =
        (context.reverse_placement_density_a3_inv / context.forward_placement_density_a3_inv).ln();
    let orientation_proposal_term =
        (context.reverse_orientation_density / context.forward_orientation_density).ln();
    let log_proposal_ratio =
        selection_proposal_term + placement_proposal_term + orientation_proposal_term;
    let mut digest = Sha256::new();
    digest.update(b"PRISM_NCMC_DERIVED_BALANCE_V2\0");
    digest.update(serde_json::to_vec(&proposal.execution_mode).map_err(|error| error.to_string())?);
    digest.update(serde_json::to_vec(context).map_err(|error| error.to_string())?);
    digest.update(initial.species_id.to_le_bytes());
    digest.update(initial.chemical_state_id.to_le_bytes());
    digest.update(proposal.proposed_species_id.to_le_bytes());
    digest.update(proposal.proposed_chemical_state_id.to_le_bytes());
    digest.update(proposed_cal.evidence_sha256.as_bytes());
    digest.update(
        serde_json::to_vec(&proposal.periodic_charge_treatment)
            .map_err(|error| error.to_string())?,
    );
    let derivation_sha256 = format!("{:x}", digest.finalize());
    let balance = NcmcDerivedBalance {
        beta_chemical_potential_term,
        beta_mol_per_kcal: beta,
        beta_intrinsic_state_reference_term_in_total_mu,
        beta_nonreference_chemical_potential_term,
        beta_periodic_finite_size_correction_term_in_total_mu,
        beta_calibrated_mu_remainder_term,
        periodic_charge_treatment: proposal.periodic_charge_treatment.clone(),
        state_reference_convention:
            StateReferenceConvention::TotalMuIncludesIntrinsicStateFreeEnergy,
        ph_log_weight_term: delta_ph_weight,
        standard_state_volume_term,
        occupancy_combinatorics_term: occupancy_term,
        target_bias_term,
        log_target_ratio,
        selection_proposal_term,
        placement_proposal_term,
        orientation_proposal_term,
        log_proposal_ratio,
        derivation_sha256,
    };
    balance.validate_terms()?;
    Ok(balance)
}

/// Accepted endpoint state commit. Rejected moves must not emit this event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LambdaFragmentStateCommit {
    /// Schema discriminator.
    pub schema_kind: String,
    /// Schema version.
    pub schema_version: u32,
    /// Native stream identity.
    pub stream_id: u32,
    /// Exact integration step of state mutation.
    pub step: u64,
    /// Move receipt digest authorizing this commit.
    pub ncmc_move_receipt_sha256: String,
    /// Strict total order within this lambda event stream.
    pub event_sequence: u64,
    /// Prior typed-emission receipt digest (normally the NCMC receipt emission).
    pub parent_receipt_sha256: String,
    /// Slot identity.
    pub slot_id: u32,
    /// New species identity.
    pub species_id: u32,
    /// New chemical state.
    pub chemical_state_id: u32,
    /// New endpoint occupancy.
    pub occupancy: EndpointOccupancy,
    /// Canonical phase/condition binding digest for this exact step.
    pub phase_protocol_sha256: String,
    /// Canonical PrismCode opcode.
    pub prismcode_opcode: String,
    /// Numeric PrismCode opcode bound by the canonical registry.
    pub prismcode_opcode_id: u16,
    /// Canonical QNANL family.
    pub qnanl_artifact_family: String,
}

impl LambdaFragmentStateCommit {
    /// Verify typed dispatch and source-receipt binding.
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_kind != "prismos_lambda_fragment_state_commit"
            || self.schema_version != 1
            || matches!(self.occupancy, EndpointOccupancy::Switching)
        {
            return Err("invalid lambda state commit contract".into());
        }
        validate_dispatch_binding(
            &self.prismcode_opcode,
            self.prismcode_opcode_id,
            prismcode_opcodes::STATE_COMMIT,
            &self.qnanl_artifact_family,
            QnanlArtifactFamily::LambdaFragmentStateCommit,
        )?;
        validate_sha256("NCMC move receipt", &self.ncmc_move_receipt_sha256)?;
        validate_sha256("parent receipt", &self.parent_receipt_sha256)?;
        validate_sha256("phase/condition binding", &self.phase_protocol_sha256)
    }
}

/// Explicit rollback/no-change authority for a rejected NCMC proposal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LambdaFragmentStateRollback {
    /// Schema discriminator.
    pub schema_kind: String,
    /// Schema version.
    pub schema_version: u32,
    /// Native stream identity.
    pub stream_id: u32,
    /// Exact integration step of resolution.
    pub step: u64,
    /// Rejected move receipt digest authorizing rollback.
    pub ncmc_move_receipt_sha256: String,
    /// Strict total order within this lambda event stream.
    pub event_sequence: u64,
    /// Prior typed-emission receipt digest.
    pub parent_receipt_sha256: String,
    /// Slot restored without mutation.
    pub slot_id: u32,
    /// Exact initial state restored after rejection.
    pub restored: NcmcEndpointIdentity,
    /// True only after the device state equals the initial endpoint.
    pub no_state_change_verified: bool,
    /// Canonical phase/condition binding digest for this exact step.
    pub phase_protocol_sha256: String,
    /// Canonical PrismCode opcode.
    pub prismcode_opcode: String,
    /// Numeric PrismCode opcode.
    pub prismcode_opcode_id: u16,
    /// Canonical QNANL family.
    pub qnanl_artifact_family: String,
}

impl LambdaFragmentStateRollback {
    /// Verify rejected-move restoration and typed dispatch.
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_kind != "prismos_lambda_fragment_state_rollback"
            || self.schema_version != 1
            || !self.no_state_change_verified
            || matches!(self.restored.occupancy, EndpointOccupancy::Switching)
        {
            return Err("invalid lambda state rollback contract".into());
        }
        validate_dispatch_binding(
            &self.prismcode_opcode,
            self.prismcode_opcode_id,
            prismcode_opcodes::STATE_ROLLBACK,
            &self.qnanl_artifact_family,
            QnanlArtifactFamily::LambdaFragmentStateRollback,
        )?;
        validate_sha256("NCMC move receipt", &self.ncmc_move_receipt_sha256)?;
        validate_sha256("parent receipt", &self.parent_receipt_sha256)?;
        validate_sha256("phase/condition binding", &self.phase_protocol_sha256)
    }
}

/// Closed-interval proof that a no-probe lane contained no lambda slots,
/// virtual probes or typed lambda force/move events.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoProbeAbsenceReceipt {
    /// Schema discriminator.
    pub schema_kind: String,
    /// Schema version.
    pub schema_version: u32,
    /// Native stream identity.
    pub stream_id: u32,
    /// Inclusive first integration step covered.
    pub start_step: u64,
    /// Exclusive end of the covered interval.
    pub end_step_exclusive: u64,
    /// Strict total order within this lambda event stream.
    pub event_sequence: u64,
    /// Prior typed-emission receipt digest, absent only at sequence zero.
    pub parent_receipt_sha256: Option<String>,
    /// Hash-bound no-probe lane identity.
    pub lane_contract_sha256: String,
    /// Must be exactly zero.
    pub lambda_slot_count: u64,
    /// Must be exactly zero.
    pub virtual_probe_count: u64,
    /// Count of force, endpoint, NCMC and state-resolution events in interval.
    pub lambda_event_count: u64,
    /// Canonical phase/condition binding digest for this interval.
    pub phase_protocol_sha256: String,
    /// Canonical PrismCode opcode.
    pub prismcode_opcode: String,
    /// Numeric PrismCode opcode.
    pub prismcode_opcode_id: u16,
    /// Canonical QNANL family.
    pub qnanl_artifact_family: String,
}

impl NoProbeAbsenceReceipt {
    /// Verify a closed physical absence interval without fabricating a slot.
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_kind != "prismos_lambda_no_probe_absence_receipt"
            || self.schema_version != 1
            || self.start_step >= self.end_step_exclusive
            || self.lambda_slot_count != 0
            || self.virtual_probe_count != 0
            || self.lambda_event_count != 0
        {
            return Err("invalid no-probe absence receipt".into());
        }
        validate_dispatch_binding(
            &self.prismcode_opcode,
            self.prismcode_opcode_id,
            prismcode_opcodes::NO_PROBE_ABSENCE,
            &self.qnanl_artifact_family,
            QnanlArtifactFamily::LambdaNoProbeAbsence,
        )?;
        validate_sha256("lane contract", &self.lane_contract_sha256)?;
        validate_sha256("phase/condition binding", &self.phase_protocol_sha256)?;
        if let Some(parent) = &self.parent_receipt_sha256 {
            validate_sha256("parent receipt", parent)?;
        }
        Ok(())
    }
}

/// Exact-byte QNANL emission receipt for one typed lambda event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LambdaTypedEmissionReceipt {
    /// Schema discriminator.
    pub schema_kind: String,
    /// Schema version.
    pub schema_version: u32,
    /// PASS only after byte-exact decode equality.
    pub status: String,
    /// Event sequence.
    pub event_sequence: u64,
    /// Parent receipt digest.
    pub parent_receipt_sha256: Option<String>,
    /// Canonical opcode name.
    pub prismcode_opcode: String,
    /// Canonical opcode id.
    pub prismcode_opcode_id: u16,
    /// QNANL family name.
    pub qnanl_artifact_family: String,
    /// QNANL family code.
    pub qnanl_artifact_family_code: u16,
    /// Honest lifecycle origin assigned before encoding.
    pub runtime_origin: String,
    /// Raw canonical event SHA256.
    pub raw_sha256: String,
    /// Packet SHA256.
    pub packet_sha256: String,
    /// Packet path.
    pub packet_path: PathBuf,
    /// QNANL receipt path.
    pub qnanl_receipt_path: PathBuf,
    /// Exact packet decode equals canonical event bytes.
    pub exact_byte_roundtrip: bool,
    /// Digest of this receipt's ordered binding fields.
    pub receipt_sha256: String,
}

/// Exact post-MD lookup key for one physical fragment slot observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LambdaJournalKey {
    /// Native stream identity.
    pub stream_id: u16,
    /// Exact integration step.
    pub step_idx: u64,
    /// Stable physical fragment slot identity.
    pub slot_id: u32,
}

/// Event class stored in an immutable lambda journal segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum LambdaJournalEventKind {
    /// Force, energy and dU/dlambda state.
    Force,
    /// NCMC accept/reject adjudication.
    NcmcMove,
    /// Physical endpoint observation.
    Endpoint,
    /// Accepted state mutation.
    StateCommit,
    /// Rejected proposal rollback/no change.
    StateRollback,
    /// Closed no-probe interval.
    NoProbeAbsence,
}

/// One immutable journal record. The raw scientific event remains the decoded
/// QNANL payload; this record binds it to exact stream/step/phase and segment
/// lineage without duplicating or rewriting that payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LambdaJournalRecord {
    /// Schema discriminator.
    pub schema_kind: String,
    /// Schema version.
    pub schema_version: u32,
    /// Segment sequence within this stream.
    pub segment_sequence: u64,
    /// Exact lookup key; absent only for a no-probe interval receipt.
    pub key: Option<LambdaJournalKey>,
    /// Event class.
    pub event_kind: LambdaJournalEventKind,
    /// Event sequence within this stream.
    pub event_sequence: u64,
    /// Parent typed-emission receipt.
    pub parent_receipt_sha256: Option<String>,
    /// Typed-emission receipt for this event.
    pub emission_receipt_sha256: String,
    /// SHA-256 of exact decoded event bytes, including trailing newline.
    pub raw_event_sha256: String,
    /// Exact decoded byte count.
    pub raw_event_bytes: u64,
    /// SHA-256 of the QNANL packet.
    pub packet_sha256: String,
    /// Segment-directory-relative QNANL path.
    pub packet_file: String,
    /// QNANL family.
    pub qnanl_artifact_family: String,
    /// Exact phase and complete condition binding.
    pub phase_protocol_sha256: String,
    /// Exact calibration for the active species/state, when applicable.
    pub chemical_potential_calibration: Option<ChemicalPotentialCalibration>,
    /// Digest of this record with this field cleared.
    pub journal_record_sha256: String,
}

impl LambdaJournalRecord {
    fn recompute_sha256(&self) -> Result<String, String> {
        let mut projection = self.clone();
        projection.journal_record_sha256.clear();
        let bytes = serde_json::to_vec(&projection).map_err(|error| error.to_string())?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }

    fn validate(&self) -> Result<(), String> {
        if self.schema_kind != "prismos_lambda_event_journal_record"
            || self.schema_version != 1
            || self.raw_event_bytes == 0
            || self.journal_record_sha256 != self.recompute_sha256()?
            || Path::new(&self.packet_file).is_absolute()
            || self.packet_file.contains("..")
        {
            return Err("invalid lambda journal record".into());
        }
        for (label, digest) in [
            ("emission receipt", &self.emission_receipt_sha256),
            ("raw event", &self.raw_event_sha256),
            ("QNANL packet", &self.packet_sha256),
            ("phase/condition binding", &self.phase_protocol_sha256),
        ] {
            validate_sha256(label, digest)?;
        }
        if let Some(parent) = &self.parent_receipt_sha256 {
            validate_sha256("parent emission receipt", parent)?;
        }
        if let Some(calibration) = &self.chemical_potential_calibration {
            calibration.validate()?;
        }
        match (self.event_kind, self.key) {
            (LambdaJournalEventKind::NoProbeAbsence, None) => Ok(()),
            (LambdaJournalEventKind::NoProbeAbsence, Some(_)) => {
                Err("no-probe absence must use interval lineage, not a fake slot key".into())
            }
            (_, Some(_)) => Ok(()),
            (_, None) => Err("fragment journal event lacks stream/step/slot key".into()),
        }
    }
}

/// Immutable per-segment runtime context copied from pre-run binding and the
/// actual phase active when the event was produced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LambdaJournalSegmentContext {
    /// Run identity.
    pub run_id: String,
    /// Replica identity.
    pub replica_id: String,
    /// Replica seed.
    pub replica_seed: u64,
    /// Native stream identity.
    pub stream_id: u16,
    /// Stream seed.
    pub stream_seed: u64,
    /// Engine binary digest.
    pub engine_sha256: String,
    /// Topology digest.
    pub topology_sha256: String,
    /// Matched lane declaration.
    pub lane: LambdaExperimentLaneContract,
    /// Calibration-work or calibrated target-lane authority; absent only for
    /// a physical no-probe lane.
    pub ncmc_execution_mode: Option<NcmcExecutionMode>,
    /// Exact proposal scheduler carried from launch admission.
    pub ncmc_scheduler: Option<NcmcProposalSchedulerContract>,
    /// Fragment system digest; absent only for no-probe.
    pub lambda_fragment_system_sha256: Option<String>,
    /// Actual protocol phase covering the event.
    pub protocol_phase: crate::chemical_state_fiber::ProtocolPhaseBinding,
    /// Actual thermodynamic condition covering the event.
    pub thermodynamic_condition: crate::chemical_state_fiber::ThermodynamicCondition,
    /// Canonical digest of phase plus complete condition.
    pub phase_protocol_sha256: String,
}

/// Exactly-once byte location within one immutable journal segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LambdaJournalByteLineage {
    /// Segment output-root-relative path.
    pub segment_file: String,
    /// Byte offset of the complete JSON record.
    pub byte_offset: u64,
    /// Byte length of the complete JSON record.
    pub byte_length: u64,
    /// SHA-256 of exact record bytes.
    pub record_bytes_sha256: String,
}

/// One immutable segment manifest. Each segment currently owns one event,
/// making crash recovery and exactly-once reconciliation unambiguous while
/// retaining a versioned segmented ABI.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LambdaJournalSegmentManifest {
    /// Schema discriminator.
    pub schema_kind: String,
    /// Schema version.
    pub schema_version: u32,
    /// Segment sequence.
    pub segment_sequence: u64,
    /// Prior segment-manifest digest in this stream.
    pub previous_segment_manifest_sha256: Option<String>,
    /// Runtime context.
    pub context: LambdaJournalSegmentContext,
    /// Exact event journal record.
    pub record: LambdaJournalRecord,
    /// Byte lineage for the record.
    pub byte_lineage: LambdaJournalByteLineage,
    /// Segment file digest.
    pub segment_sha256: String,
    /// Segment byte count.
    pub segment_bytes: u64,
    /// Digest of this manifest with this field cleared.
    pub manifest_sha256: String,
}

impl LambdaJournalSegmentManifest {
    fn recompute_sha256(&self) -> Result<String, String> {
        let mut projection = self.clone();
        projection.manifest_sha256.clear();
        let bytes = serde_json::to_vec(&projection).map_err(|error| error.to_string())?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }

    fn validate(&self) -> Result<(), String> {
        if self.schema_kind != "prismos_lambda_event_journal_segment_manifest"
            || self.schema_version != 1
            || self.record.segment_sequence != self.segment_sequence
            || self.segment_bytes == 0
            || self.byte_lineage.byte_offset != 0
            || self.byte_lineage.byte_length != self.segment_bytes
            || self.manifest_sha256 != self.recompute_sha256()?
        {
            return Err("invalid lambda journal segment manifest".into());
        }
        self.record.validate()?;
        validate_sha256("lambda journal segment", &self.segment_sha256)?;
        validate_sha256(
            "lambda journal record bytes",
            &self.byte_lineage.record_bytes_sha256,
        )?;
        if let Some(previous) = &self.previous_segment_manifest_sha256 {
            validate_sha256("previous lambda segment manifest", previous)?;
        }
        Ok(())
    }
}

/// Fully decoded event represented by one journal record.
#[derive(Debug, Clone)]
pub enum LambdaDecodedJournalEvent {
    /// Force/dUdl state.
    Force(LambdaFragmentForceEvent),
    /// NCMC move decision.
    Ncmc(NcmcMoveReceipt),
    /// Physical endpoint.
    Endpoint(LambdaFragmentEndpointObservation),
    /// Accepted commit.
    Commit(LambdaFragmentStateCommit),
    /// Rejected rollback.
    Rollback(LambdaFragmentStateRollback),
    /// No-probe interval.
    NoProbeAbsence(NoProbeAbsenceReceipt),
}

/// Complete exact-key observation assembled from hash-chained event segments.
#[derive(Debug, Clone)]
pub struct LambdaJournalObservation {
    /// Exact stream/step/slot key.
    pub key: LambdaJournalKey,
    /// Matched lane.
    pub lane: LambdaExperimentLaneContract,
    /// Every ordered force/dUdl event sampled at this exact key. NCMC requires
    /// repeated PRE/POST boundaries and none may be overwritten.
    pub force_events: Vec<LambdaFragmentForceEvent>,
    /// Endpoint event, only at exact lambda zero or one.
    pub endpoint_event: Option<LambdaFragmentEndpointObservation>,
    /// Exact chemical-potential calibration for the active species/state.
    pub chemical_potential_calibration: Option<ChemicalPotentialCalibration>,
    /// NCMC move receipt, if a move resolved at this step.
    pub ncmc_move_receipt: Option<NcmcMoveReceipt>,
    /// Accepted state commit authority.
    pub accepted_state_commit: Option<LambdaFragmentStateCommit>,
    /// Explicit rejected rollback/no-change authority.
    pub rollback_no_change: Option<LambdaFragmentStateRollback>,
    /// Ordered byte lineage for every event contributing to this key.
    pub event_lineage: Vec<LambdaJournalLoadedLineage>,
}

/// Loader-preserved segment and packet byte lineage.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LambdaJournalLoadedLineage {
    /// Event class.
    pub event_kind: LambdaJournalEventKind,
    /// Event sequence.
    pub event_sequence: u64,
    /// Parent emission receipt.
    pub parent_receipt_sha256: Option<String>,
    /// Emission receipt.
    pub emission_receipt_sha256: String,
    /// Raw event digest.
    pub raw_event_sha256: String,
    /// Segment manifest digest.
    pub segment_manifest_sha256: String,
    /// Segment digest.
    pub segment_sha256: String,
    /// Segment relative path.
    pub segment_file: String,
    /// Exact record offset.
    pub byte_offset: u64,
    /// Exact record length.
    pub byte_length: u64,
}

impl LambdaJournalObservation {
    /// Resolve the exact force event explicitly named by endpoint schema v2.
    pub fn supporting_force_event(&self) -> Result<Option<&LambdaFragmentForceEvent>, String> {
        let Some(endpoint) = &self.endpoint_event else {
            return Ok(None);
        };
        let lineage = self
            .event_lineage
            .iter()
            .find(|lineage| {
                lineage.event_kind == LambdaJournalEventKind::Force
                    && lineage.emission_receipt_sha256
                        == endpoint.supporting_force_emission_receipt_sha256
                    && lineage.raw_event_sha256 == endpoint.supporting_force_event_sha256
            })
            .ok_or_else(|| "endpoint supporting force byte lineage is absent".to_string())?;
        let matches: Vec<_> = self
            .force_events
            .iter()
            .filter(|force| {
                force.event_sequence == lineage.event_sequence
                    && force.ncmc_boundary.as_ref() == Some(&endpoint.supporting_force_boundary)
            })
            .collect();
        if matches.len() != 1 {
            return Err("endpoint supporting force identity is not unique".into());
        }
        Ok(Some(matches[0]))
    }
}

/// Verified post-MD journal loader result.
#[derive(Debug)]
pub struct LambdaEventJournalLoad {
    /// Number of manifest-declared events.
    pub produced: u64,
    /// Number with exact immutable segment bytes.
    pub persisted: u64,
    /// Number with byte-exact QNANL decode and typed validation.
    pub decoded: u64,
    /// Number represented in keyed observations or absence intervals.
    pub indexed: u64,
    /// Atomically committed three-event NCMC resolution batches.
    pub committed_resolution_batch_count: u64,
    /// Always zero on a successful load; any staging residue fails closed.
    pub partial_resolution_chain_count: u64,
    /// Exact stream/step/slot lookup.
    pub observations: BTreeMap<LambdaJournalKey, LambdaJournalObservation>,
    /// Explicit no-probe interval receipts.
    pub no_probe_absence_receipts: Vec<(NoProbeAbsenceReceipt, LambdaJournalLoadedLineage)>,
    /// Terminal segment manifest per stream.
    pub stream_terminal_manifest_sha256: BTreeMap<u16, String>,
    /// Terminal verified context per stream.
    pub stream_contexts: BTreeMap<u16, LambdaJournalSegmentContext>,
    /// Every distinct, ordered protocol-phase context observed per stream.
    pub stream_phase_contexts: BTreeMap<u16, Vec<LambdaJournalSegmentContext>>,
    /// Canonical digest over counts, terminal manifests, and phase contexts.
    pub population_receipt_sha256: String,
}

/// One exact immutable journal object bound into terminal runtime closeout.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LambdaRuntimeCloseoutArtifact {
    pub event_sequence: u64,
    pub event_kind: LambdaJournalEventKind,
    pub emission_receipt_sha256: String,
    pub raw_event_sha256: String,
    pub packet_sha256: String,
    pub journal_record_sha256: String,
    pub segment_sha256: String,
    pub segment_manifest_sha256: String,
}

/// Unconditional per-stream terminal authority. This receipt is emitted for
/// MD-only, no-probe, calibration and target lanes alike; native Path-B is not
/// required to establish journal conservation or terminal phase completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LambdaRuntimeCloseoutReceipt {
    pub schema_kind: String,
    pub schema_version: u32,
    pub run_id: String,
    pub replica_id: String,
    pub stream_id: u16,
    pub lane_contract_sha256: String,
    pub terminal_step_exclusive: u64,
    pub terminal_protocol_phase: crate::chemical_state_fiber::ProtocolPhaseBinding,
    pub terminal_thermodynamic_condition: crate::chemical_state_fiber::ThermodynamicCondition,
    pub terminal_phase_protocol_sha256: String,
    pub produced: u64,
    pub persisted: u64,
    pub decoded: u64,
    pub indexed: u64,
    pub first_event_sequence: Option<u64>,
    pub last_event_sequence: Option<u64>,
    pub first_emission_receipt_sha256: Option<String>,
    pub last_emission_receipt_sha256: Option<String>,
    pub terminal_segment_manifest_sha256: Option<String>,
    pub committed_resolution_batch_count: u64,
    pub partial_resolution_chain_count: u64,
    pub scheduled_ncmc_opportunity_count: u64,
    pub matched_velocity_initialization_receipt_sha256: String,
    pub environment_velocity_sha256: String,
    pub full_velocity_sha256: String,
    pub environment_position_sha256: String,
    pub full_position_sha256: String,
    pub journal_population_receipt_sha256: String,
    pub artifacts: Vec<LambdaRuntimeCloseoutArtifact>,
    pub artifact_population_sha256: String,
    pub completion_status: String,
    pub receipt_sha256: String,
}

impl LambdaRuntimeCloseoutReceipt {
    fn recompute_sha256(&self) -> Result<String, String> {
        let mut projection = self.clone();
        projection.receipt_sha256.clear();
        let mut digest = Sha256::new();
        digest.update(b"PRISM_LAMBDA_RUNTIME_CLOSEOUT_RECEIPT_V1\0");
        digest.update(serde_json::to_vec(&projection).map_err(|error| error.to_string())?);
        Ok(format!("{:x}", digest.finalize()))
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_kind != "prismos_lambda_runtime_closeout_receipt"
            || self.schema_version != 1
            || self.completion_status != "COMPLETE"
            || self.produced != self.persisted
            || self.produced != self.decoded
            || self.produced != self.indexed
            || self.produced != self.artifacts.len() as u64
            || self.partial_resolution_chain_count != 0
            || self.committed_resolution_batch_count != self.scheduled_ncmc_opportunity_count
            || self.terminal_step_exclusive != self.terminal_protocol_phase.end_step_exclusive
            || self.terminal_protocol_phase.stream_id != self.stream_id
            || self.terminal_protocol_phase.thermodynamic_condition_id
                != self.terminal_thermodynamic_condition.condition_id
            || self.receipt_sha256 != self.recompute_sha256()?
        {
            return Err("invalid or incomplete lambda runtime closeout receipt".into());
        }
        for (label, digest) in [
            ("lane contract", &self.lane_contract_sha256),
            (
                "terminal phase/condition",
                &self.terminal_phase_protocol_sha256,
            ),
            (
                "journal population",
                &self.journal_population_receipt_sha256,
            ),
            ("artifact population", &self.artifact_population_sha256),
            (
                "matched velocity initialization receipt",
                &self.matched_velocity_initialization_receipt_sha256,
            ),
            ("environment velocity", &self.environment_velocity_sha256),
            ("full velocity", &self.full_velocity_sha256),
            ("environment position", &self.environment_position_sha256),
            ("full position", &self.full_position_sha256),
        ] {
            validate_sha256(label, digest)?;
        }
        self.terminal_thermodynamic_condition
            .validate()
            .map_err(|error| error.to_string())?;
        self.terminal_protocol_phase
            .validate_for(crate::chemical_state_fiber::GhostLineageKey {
                stream_id: self.stream_id,
                frame_idx: 0,
                step_idx: self.terminal_protocol_phase.start_step,
                site_id: 0,
                record_index: 0,
            })
            .map_err(|error| error.to_string())?;
        let expected_phase = crate::chemical_state_fiber::phase_protocol_binding_sha256(
            &self.terminal_protocol_phase,
            &self.terminal_thermodynamic_condition,
        )
        .map_err(|error| error.to_string())?;
        if expected_phase != self.terminal_phase_protocol_sha256 {
            return Err("lambda closeout terminal phase digest mismatch".into());
        }
        let mut population = Sha256::new();
        population.update(b"PRISM_LAMBDA_RUNTIME_CLOSEOUT_ARTIFACT_POPULATION_V1\0");
        for (index, artifact) in self.artifacts.iter().enumerate() {
            if artifact.event_sequence != index as u64 {
                return Err("lambda closeout event sequence is not contiguous".into());
            }
            for (label, digest) in [
                ("emission receipt", &artifact.emission_receipt_sha256),
                ("raw event", &artifact.raw_event_sha256),
                ("packet", &artifact.packet_sha256),
                ("journal record", &artifact.journal_record_sha256),
                ("segment", &artifact.segment_sha256),
                ("segment manifest", &artifact.segment_manifest_sha256),
            ] {
                validate_sha256(label, digest)?;
            }
            population.update(serde_json::to_vec(artifact).map_err(|error| error.to_string())?);
        }
        if format!("{:x}", population.finalize()) != self.artifact_population_sha256 {
            return Err("lambda closeout artifact population digest mismatch".into());
        }
        match self.artifacts.as_slice() {
            [] => {
                if self.first_event_sequence.is_some()
                    || self.last_event_sequence.is_some()
                    || self.first_emission_receipt_sha256.is_some()
                    || self.last_emission_receipt_sha256.is_some()
                    || self.terminal_segment_manifest_sha256.is_some()
                {
                    return Err("zero-event lambda closeout carries fabricated event heads".into());
                }
            }
            artifacts => {
                let first = &artifacts[0];
                let last = &artifacts[artifacts.len() - 1];
                if self.first_event_sequence != Some(first.event_sequence)
                    || self.last_event_sequence != Some(last.event_sequence)
                    || self.first_emission_receipt_sha256.as_deref()
                        != Some(first.emission_receipt_sha256.as_str())
                    || self.last_emission_receipt_sha256.as_deref()
                        != Some(last.emission_receipt_sha256.as_str())
                    || self.terminal_segment_manifest_sha256.as_deref()
                        != Some(last.segment_manifest_sha256.as_str())
                {
                    return Err("lambda closeout first/last event authority mismatch".into());
                }
            }
        }
        Ok(())
    }
}

trait LambdaTypedEvent: Serialize {
    fn validate_event(&self) -> Result<(), String>;
    fn event_sequence(&self) -> u64;
    fn parent_receipt(&self) -> Option<&str>;
    fn opcode(&self) -> &'static str;
    fn family(&self) -> QnanlArtifactFamily;
    fn stream_id(&self) -> u32;
    fn step_or_interval_start(&self) -> u64;
    fn slot_id(&self) -> Option<u32>;
    fn phase_protocol_sha256(&self) -> &str;
    fn calibration_identity(&self) -> Option<(u32, u32)>;
}

impl LambdaTypedEvent for LambdaFragmentForceEvent {
    fn validate_event(&self) -> Result<(), String> {
        self.validate()
    }
    fn event_sequence(&self) -> u64 {
        self.event_sequence
    }
    fn parent_receipt(&self) -> Option<&str> {
        self.parent_receipt_sha256.as_deref()
    }
    fn opcode(&self) -> &'static str {
        prismcode_opcodes::FORCE
    }
    fn family(&self) -> QnanlArtifactFamily {
        QnanlArtifactFamily::LambdaFragmentForceEvent
    }
    fn stream_id(&self) -> u32 {
        self.stream_id
    }
    fn step_or_interval_start(&self) -> u64 {
        self.step
    }
    fn slot_id(&self) -> Option<u32> {
        Some(self.slot_id)
    }
    fn phase_protocol_sha256(&self) -> &str {
        &self.phase_protocol_sha256
    }
    fn calibration_identity(&self) -> Option<(u32, u32)> {
        Some((self.species_id, self.chemical_state_id))
    }
}

impl LambdaTypedEvent for NcmcMoveReceipt {
    fn validate_event(&self) -> Result<(), String> {
        self.validate()
    }
    fn event_sequence(&self) -> u64 {
        self.event_sequence
    }
    fn parent_receipt(&self) -> Option<&str> {
        Some(&self.parent_receipt_sha256)
    }
    fn opcode(&self) -> &'static str {
        prismcode_opcodes::NCMC_ACCEPT
    }
    fn family(&self) -> QnanlArtifactFamily {
        QnanlArtifactFamily::LambdaFragmentNcmcMoveReceipt
    }
    fn stream_id(&self) -> u32 {
        self.stream_id
    }
    fn step_or_interval_start(&self) -> u64 {
        self.step
    }
    fn slot_id(&self) -> Option<u32> {
        Some(self.slot_id)
    }
    fn phase_protocol_sha256(&self) -> &str {
        &self.phase_protocol_sha256
    }
    fn calibration_identity(&self) -> Option<(u32, u32)> {
        Some((self.proposed.species_id, self.proposed.chemical_state_id))
    }
}

impl LambdaTypedEvent for LambdaFragmentEndpointObservation {
    fn validate_event(&self) -> Result<(), String> {
        self.validate()
    }
    fn event_sequence(&self) -> u64 {
        self.event_sequence
    }
    fn parent_receipt(&self) -> Option<&str> {
        Some(&self.parent_receipt_sha256)
    }
    fn opcode(&self) -> &'static str {
        prismcode_opcodes::ENDPOINT_OBSERVE
    }
    fn family(&self) -> QnanlArtifactFamily {
        QnanlArtifactFamily::LambdaFragmentEndpointObservation
    }
    fn stream_id(&self) -> u32 {
        self.stream_id
    }
    fn step_or_interval_start(&self) -> u64 {
        self.step
    }
    fn slot_id(&self) -> Option<u32> {
        Some(self.slot_id)
    }
    fn phase_protocol_sha256(&self) -> &str {
        &self.phase_protocol_sha256
    }
    fn calibration_identity(&self) -> Option<(u32, u32)> {
        Some((self.species_id, self.chemical_state_id))
    }
}

impl LambdaTypedEvent for LambdaFragmentStateCommit {
    fn validate_event(&self) -> Result<(), String> {
        self.validate()
    }
    fn event_sequence(&self) -> u64 {
        self.event_sequence
    }
    fn parent_receipt(&self) -> Option<&str> {
        Some(&self.parent_receipt_sha256)
    }
    fn opcode(&self) -> &'static str {
        prismcode_opcodes::STATE_COMMIT
    }
    fn family(&self) -> QnanlArtifactFamily {
        QnanlArtifactFamily::LambdaFragmentStateCommit
    }
    fn stream_id(&self) -> u32 {
        self.stream_id
    }
    fn step_or_interval_start(&self) -> u64 {
        self.step
    }
    fn slot_id(&self) -> Option<u32> {
        Some(self.slot_id)
    }
    fn phase_protocol_sha256(&self) -> &str {
        &self.phase_protocol_sha256
    }
    fn calibration_identity(&self) -> Option<(u32, u32)> {
        Some((self.species_id, self.chemical_state_id))
    }
}

impl LambdaTypedEvent for LambdaFragmentStateRollback {
    fn validate_event(&self) -> Result<(), String> {
        self.validate()
    }
    fn event_sequence(&self) -> u64 {
        self.event_sequence
    }
    fn parent_receipt(&self) -> Option<&str> {
        Some(&self.parent_receipt_sha256)
    }
    fn opcode(&self) -> &'static str {
        prismcode_opcodes::STATE_ROLLBACK
    }
    fn family(&self) -> QnanlArtifactFamily {
        QnanlArtifactFamily::LambdaFragmentStateRollback
    }
    fn stream_id(&self) -> u32 {
        self.stream_id
    }
    fn step_or_interval_start(&self) -> u64 {
        self.step
    }
    fn slot_id(&self) -> Option<u32> {
        Some(self.slot_id)
    }
    fn phase_protocol_sha256(&self) -> &str {
        &self.phase_protocol_sha256
    }
    fn calibration_identity(&self) -> Option<(u32, u32)> {
        Some((self.restored.species_id, self.restored.chemical_state_id))
    }
}

impl LambdaTypedEvent for NoProbeAbsenceReceipt {
    fn validate_event(&self) -> Result<(), String> {
        self.validate()
    }
    fn event_sequence(&self) -> u64 {
        self.event_sequence
    }
    fn parent_receipt(&self) -> Option<&str> {
        self.parent_receipt_sha256.as_deref()
    }
    fn opcode(&self) -> &'static str {
        prismcode_opcodes::NO_PROBE_ABSENCE
    }
    fn family(&self) -> QnanlArtifactFamily {
        QnanlArtifactFamily::LambdaNoProbeAbsence
    }
    fn stream_id(&self) -> u32 {
        self.stream_id
    }
    fn step_or_interval_start(&self) -> u64 {
        self.start_step
    }
    fn slot_id(&self) -> Option<u32> {
        None
    }
    fn phase_protocol_sha256(&self) -> &str {
        &self.phase_protocol_sha256
    }
    fn calibration_identity(&self) -> Option<(u32, u32)> {
        None
    }
}

/// Stateful emitter that enforces event ordering and parent-receipt lineage.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct LambdaTypedEventEmitter {
    output_dir: PathBuf,
    logical_output_dir_override: Option<PathBuf>,
    runtime_origin: RuntimeOrigin,
    next_sequence: u64,
    last_receipt_sha256: Option<String>,
    next_segment_sequence: u64,
    last_segment_manifest_sha256: Option<String>,
    emitted_total_bytes: u64,
    closed_no_probe_phase_digests: BTreeSet<String>,
    runtime_binding: Option<LambdaRuntimeBinding>,
    lambda_system: Option<LambdaFragmentSystem>,
    active_protocol_phase: Option<crate::chemical_state_fiber::ProtocolPhaseBinding>,
    active_thermodynamic_condition: Option<crate::chemical_state_fiber::ThermodynamicCondition>,
    matched_velocity_initialization_receipt: Option<LambdaMatchedVelocityInitializationReceipt>,
}

/// Execute one runtime-inline typed force emission for the WP4 bootstrap.
///
/// This is intentionally an engine-owned producer rather than a deserializer:
/// callers provide the already validated pre-CUDA binding and live numeric
/// U/dUdl values, while this function derives every event identity from the
/// prepared system, emits QNANL bytes, and accepts only an exact runtime-inline
/// decode round trip.  A test fixture or caller-authored capability document
/// cannot instantiate the returned receipt.
pub fn emit_wp4_runtime_inline_force_self_test(
    binding: LambdaRuntimeBinding,
    system: &LambdaFragmentSystem,
    potential_kcal_mol: f64,
    dudlambda_kcal_mol: f64,
) -> Result<LambdaTypedEmissionReceipt, String> {
    binding.validate(Some(system))?;
    let slot = system
        .slots
        .first()
        .ok_or_else(|| "WP4 typed-event self-test requires a physical lambda slot".to_string())?;
    let endpoint_parameter_sha256 = fragment_slot_endpoint_parameter_digest_sha256(system, slot)?;
    let endpoint_support_binding_sha256 = binding
        .endpoint_support_binding_sha256
        .clone()
        .ok_or_else(|| "WP4 typed-event self-test lacks endpoint support binding".to_string())?;
    let phase_protocol_sha256 = binding.phase_protocol_sha256()?;
    let mut union_digest = Sha256::new();
    union_digest.update(b"PRISM_WP4_LIVE_DUAL_ENDPOINT_UNION_PROJECTION_V1\0");
    union_digest.update(serde_json::to_vec(system).map_err(|error| error.to_string())?);
    union_digest.update(endpoint_support_binding_sha256.as_bytes());
    let event = LambdaFragmentForceEvent {
        schema_kind: "prismos_lambda_fragment_force_event".into(),
        schema_version: 3,
        stream_id: binding.stream_id,
        step: binding.initial_protocol_phase.start_step,
        event_sequence: 0,
        parent_receipt_sha256: None,
        ncmc_boundary: None,
        slot_id: slot.slot_id,
        species_id: slot.species_id,
        chemical_state_id: slot.chemical_state_id,
        lambda: slot.lambda,
        path_kind: slot.path_kind,
        endpoint_a: slot.endpoint_a.clone(),
        endpoint_b: slot.endpoint_b.clone(),
        potential_kcal_mol,
        du_dlambda_kcal_mol: dudlambda_kcal_mol,
        energy_unit: "kcal/mol".into(),
        force_unit: "kcal/mol/angstrom".into(),
        endpoint_parameter_sha256,
        endpoint_support_binding_sha256,
        dual_endpoint_union_contract_sha256: format!("{:x}", union_digest.finalize()),
        hamiltonian_sha256: system.softcore.digest_sha256()?,
        phase_protocol_sha256,
        prismcode_opcode: prismcode_opcodes::FORCE.into(),
        prismcode_opcode_id: opcode_id_for_name(prismcode_opcodes::FORCE)
            .ok_or_else(|| "PrismCode lambda force opcode is absent".to_string())?,
        qnanl_artifact_family: QnanlArtifactFamily::LambdaFragmentForceEvent
            .as_str()
            .into(),
    };
    let mut emitter = LambdaTypedEventEmitter::new_runtime_inline(binding, Some(system.clone()))?;
    let receipt = emitter.emit_force(&event)?;
    if receipt.status != "PASS"
        || receipt.runtime_origin != RuntimeOrigin::RuntimeInline.as_str()
        || !receipt.exact_byte_roundtrip
    {
        return Err("WP4 runtime-inline typed-event self-test did not close".into());
    }
    Ok(receipt)
}

#[derive(Clone)]
pub(crate) struct LambdaNcmcResolutionBatchTransaction {
    original_output_dir: PathBuf,
    staging_dir: PathBuf,
    committed_dir: PathBuf,
    next_sequence: u64,
    last_receipt_sha256: Option<String>,
    next_segment_sequence: u64,
    last_segment_manifest_sha256: Option<String>,
    emitted_total_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LambdaNcmcResolutionBatchCommitManifest {
    schema_kind: String,
    schema_version: u32,
    stream_id: u32,
    step: u64,
    move_id: u64,
    first_event_sequence: u64,
    event_count: u64,
    ncmc_move_receipt_sha256: String,
    terminal_emission_receipt_sha256: String,
    members: Vec<LambdaNcmcResolutionBatchMember>,
    atomic_directory_publish: bool,
    manifest_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LambdaNcmcResolutionBatchMember {
    event_sequence: u64,
    event_kind: LambdaJournalEventKind,
    segment_manifest_sha256: String,
    journal_record_sha256: String,
    emission_receipt_sha256: String,
}

impl LambdaNcmcResolutionBatchCommitManifest {
    fn recompute_sha256(&self) -> Result<String, String> {
        let mut projection = self.clone();
        projection.manifest_sha256.clear();
        let mut digest = Sha256::new();
        digest.update(b"PRISM_LAMBDA_NCMC_RESOLUTION_BATCH_COMMIT_V1\0");
        digest.update(serde_json::to_vec(&projection).map_err(|error| error.to_string())?);
        Ok(format!("{:x}", digest.finalize()))
    }

    fn validate(&self) -> Result<(), String> {
        if self.schema_kind != "prismos_lambda_ncmc_resolution_batch_commit"
            || self.schema_version != 1
            || self.event_count != 3
            || self.members.len() != 3
            || !self.atomic_directory_publish
            || self.manifest_sha256 != self.recompute_sha256()?
        {
            return Err("invalid atomic NCMC resolution batch manifest".into());
        }
        validate_sha256("NCMC move object", &self.ncmc_move_receipt_sha256)?;
        validate_sha256(
            "NCMC terminal emission receipt",
            &self.terminal_emission_receipt_sha256,
        )?;
        for (offset, member) in self.members.iter().enumerate() {
            if member.event_sequence != self.first_event_sequence + offset as u64 {
                return Err("NCMC batch member sequence is not contiguous".into());
            }
            validate_sha256("NCMC member manifest", &member.segment_manifest_sha256)?;
            validate_sha256("NCMC member record", &member.journal_record_sha256)?;
            validate_sha256("NCMC member emission", &member.emission_receipt_sha256)?;
        }
        if self.members[0].event_kind != LambdaJournalEventKind::NcmcMove
            || self.members[1].event_kind != LambdaJournalEventKind::Endpoint
            || !matches!(
                self.members[2].event_kind,
                LambdaJournalEventKind::StateCommit | LambdaJournalEventKind::StateRollback
            )
            || self.members[2].emission_receipt_sha256 != self.terminal_emission_receipt_sha256
        {
            return Err(
                "NCMC batch member event classes are not decision/endpoint/resolution".into(),
            );
        }
        Ok(())
    }
}

impl LambdaTypedEventEmitter {
    /// Construct an honest post-run/manual emitter for validation and recovery.
    /// Such artifacts never count as inline engine ownership.
    pub fn new_post_run_manual(output_dir: impl Into<PathBuf>) -> Self {
        Self {
            output_dir: output_dir.into(),
            logical_output_dir_override: None,
            runtime_origin: RuntimeOrigin::PostRunManual,
            next_sequence: 0,
            last_receipt_sha256: None,
            next_segment_sequence: 0,
            last_segment_manifest_sha256: None,
            emitted_total_bytes: 0,
            closed_no_probe_phase_digests: BTreeSet::new(),
            runtime_binding: None,
            lambda_system: None,
            active_protocol_phase: None,
            active_thermodynamic_condition: None,
            matched_velocity_initialization_receipt: None,
        }
    }

    /// Runtime-inline constructor is crate-private so external post-processors
    /// cannot self-label output as engine-owned.
    pub(crate) fn new_runtime_inline(
        binding: LambdaRuntimeBinding,
        lambda_system: Option<LambdaFragmentSystem>,
    ) -> Result<Self, String> {
        binding.validate(lambda_system.as_ref())?;
        let output_dir = binding.typed_event_output_dir.clone();
        secure_create_lambda_output_dir(&binding.run_output_root, &output_dir)?;
        if std::fs::read_dir(&output_dir)
            .map_err(|error| error.to_string())?
            .next()
            .is_some()
        {
            return Err(
                "lambda runtime output root is nonempty: same-run emitter recovery is forbidden without an atomically bound physical phase-space and scheduler checkpoint; preserve this run and restart the seed in a new unique run root"
                    .into(),
            );
        }
        enforce_lambda_output_capacity(&output_dir, &binding.typed_event_capacity)?;
        let active_protocol_phase = Some(binding.initial_protocol_phase.clone());
        let active_thermodynamic_condition = Some(binding.initial_thermodynamic_condition.clone());
        Ok(Self {
            output_dir,
            logical_output_dir_override: None,
            runtime_origin: RuntimeOrigin::RuntimeInline,
            next_sequence: 0,
            last_receipt_sha256: None,
            next_segment_sequence: 0,
            last_segment_manifest_sha256: None,
            emitted_total_bytes: 0,
            closed_no_probe_phase_digests: BTreeSet::new(),
            runtime_binding: Some(binding),
            lambda_system,
            active_protocol_phase,
            active_thermodynamic_condition,
            matched_velocity_initialization_receipt: None,
        })
    }

    /// Bind the exact initialized environment/full phase-space prefixes to
    /// this runtime stream before any terminal closeout can pass.
    pub(crate) fn bind_matched_velocity_initialization(
        &mut self,
        receipt: LambdaMatchedVelocityInitializationReceipt,
    ) -> Result<(), String> {
        receipt.validate()?;
        let binding = self
            .runtime_binding
            .as_ref()
            .ok_or_else(|| "manual emitter cannot bind matched initialization".to_string())?;
        if receipt.stream_id != binding.stream_id as u16
            || receipt.replica_seed != binding.replica_seed
            || receipt.contract_sha256 != binding.matched_velocity_initialization.contract_sha256
            || receipt.environment_atom_count
                != binding
                    .matched_velocity_initialization
                    .environment_atom_count
            || receipt.environment_position_f64_checkpoint_sha256
                != binding
                    .matched_velocity_initialization
                    .environment_position_f64_checkpoint_sha256
            || receipt.f64_to_f32_projection_policy
                != binding
                    .matched_velocity_initialization
                    .f64_to_f32_projection_policy
            || receipt.periodic_cell_contract_sha256
                != binding
                    .matched_velocity_initialization
                    .periodic_cell_contract_sha256
            || receipt.periodic_cell_f64_checkpoint_sha256
                != binding
                    .matched_velocity_initialization
                    .periodic_cell_f64_checkpoint_sha256
            || receipt.periodic_cell_f32_projection_sha256
                != binding
                    .matched_velocity_initialization
                    .periodic_cell_f32_projection_sha256
            || receipt.npt_checkpoint_population_authority_sha256
                != binding
                    .matched_velocity_initialization
                    .npt_checkpoint_population_authority_sha256
            || receipt.npt_seed_authority_sha256
                != binding
                    .matched_velocity_initialization
                    .npt_seed_authority_sha256
            || receipt.npt_frozen_cell_authority_sha256
                != binding
                    .matched_velocity_initialization
                    .npt_frozen_cell_authority_sha256
            || receipt.fragment_coordinate_branch_authority_sha256
                != binding
                    .matched_velocity_initialization
                    .fragment_coordinate_branch_authority_sha256
            || receipt.fragment_coordinate_ensemble_sha256
                != binding
                    .matched_velocity_initialization
                    .fragment_coordinate_ensemble_sha256
            || receipt.fragment_position_f64_checkpoint_sha256
                != binding
                    .matched_velocity_initialization
                    .fragment_position_f64_checkpoint_sha256
            || receipt.fragment_position_sha256
                != binding
                    .matched_velocity_initialization
                    .fragment_position_sha256
            || receipt.environment_position_sha256
                != binding
                    .matched_velocity_initialization
                    .environment_position_sha256
            || (receipt.fragment_atom_count == 0)
                != binding
                    .matched_velocity_initialization
                    .fragment_position_sha256
                    .is_none()
        {
            return Err("matched initialization receipt differs from runtime admission".into());
        }
        if let Some(existing) = &self.matched_velocity_initialization_receipt {
            if existing.receipt_sha256 != receipt.receipt_sha256 {
                return Err("matched initialization changed within one runtime stream".into());
            }
            return Ok(());
        }
        let path = self
            .output_dir
            .join("prismos_lambda_matched_velocity_initialization_receipt.json");
        let mut bytes = serde_json::to_vec_pretty(&receipt).map_err(|error| error.to_string())?;
        bytes.push(b'\n');
        atomic_write_synced(&path, &bytes)?;
        self.matched_velocity_initialization_receipt = Some(receipt);
        Ok(())
    }

    /// Bind the next real protocol phase/condition before any event in that
    /// interval is emitted. This cannot rewrite prior segment context.
    pub fn bind_protocol_phase(
        &mut self,
        phase: crate::chemical_state_fiber::ProtocolPhaseBinding,
        condition: crate::chemical_state_fiber::ThermodynamicCondition,
    ) -> Result<(), String> {
        let binding = self
            .runtime_binding
            .as_ref()
            .ok_or_else(|| "manual emitter cannot claim runtime phase ownership".to_string())?;
        if phase.stream_id != binding.stream_id as u16
            || phase.thermodynamic_condition_id != condition.condition_id
        {
            return Err("lambda phase transition stream/condition mismatch".into());
        }
        condition.validate().map_err(|error| error.to_string())?;
        phase
            .validate_for(crate::chemical_state_fiber::GhostLineageKey {
                stream_id: phase.stream_id,
                frame_idx: 0,
                step_idx: phase.start_step,
                site_id: 0,
                record_index: 0,
            })
            .map_err(|error| error.to_string())?;
        self.active_protocol_phase = Some(phase);
        self.active_thermodynamic_condition = Some(condition);
        Ok(())
    }

    /// Bind the unique, pre-admitted phase covering `step`. This is the only
    /// production transition path; callers cannot synthesize phase metadata.
    pub(crate) fn bind_protocol_phase_for_step(&mut self, step: u64) -> Result<(), String> {
        let context = self
            .runtime_binding
            .as_ref()
            .ok_or_else(|| "manual emitter cannot bind an admitted runtime phase".to_string())?
            .phase_context_for_step(step)?
            .clone();
        if self.active_protocol_phase.as_ref() != Some(&context.protocol_phase)
            || self.active_thermodynamic_condition.as_ref()
                != Some(&context.thermodynamic_condition)
        {
            self.bind_protocol_phase(context.protocol_phase, context.thermodynamic_condition)?;
        }
        Ok(())
    }

    /// Close the active no-probe interval exactly once before a phase
    /// transition or terminal runtime closeout. This is an explicit engine
    /// operation and is never synthesized from Drop/teardown.
    pub(crate) fn close_active_no_probe_interval(
        &mut self,
    ) -> Result<Option<LambdaTypedEmissionReceipt>, String> {
        let binding = self
            .runtime_binding
            .as_ref()
            .ok_or_else(|| "manual emitter cannot close a runtime absence interval".to_string())?;
        if binding.lane.lane_kind != LambdaExperimentLaneKind::NoProbe {
            return Ok(None);
        }
        let phase = self
            .active_protocol_phase
            .as_ref()
            .ok_or_else(|| "no-probe emitter lacks active phase".to_string())?
            .clone();
        let digest = self.active_phase_protocol_sha256()?;
        if self.closed_no_probe_phase_digests.contains(&digest) {
            return Ok(None);
        }
        let event = NoProbeAbsenceReceipt {
            schema_kind: "prismos_lambda_no_probe_absence_receipt".into(),
            schema_version: 1,
            stream_id: binding.stream_id,
            start_step: phase.start_step,
            end_step_exclusive: phase.end_step_exclusive,
            event_sequence: self.next_sequence,
            parent_receipt_sha256: self.last_receipt_sha256.clone(),
            lane_contract_sha256: binding.lane.contract_sha256.clone(),
            lambda_slot_count: 0,
            virtual_probe_count: 0,
            lambda_event_count: 0,
            phase_protocol_sha256: digest.clone(),
            prismcode_opcode: prismcode_opcodes::NO_PROBE_ABSENCE.into(),
            prismcode_opcode_id: opcode_id_for_name(prismcode_opcodes::NO_PROBE_ABSENCE)
                .ok_or_else(|| "PrismCode no-probe absence opcode missing".to_string())?,
            qnanl_artifact_family: QnanlArtifactFamily::LambdaNoProbeAbsence.as_str().into(),
        };
        let receipt = self.emit_no_probe_absence(&event)?;
        self.closed_no_probe_phase_digests.insert(digest);
        Ok(Some(receipt))
    }

    /// Close the admitted terminal phase and durably publish an unconditional
    /// conservation receipt. The receipt is built by reloading exact segment
    /// and QNANL bytes, so runtime counters alone can never certify success.
    pub(crate) fn close_runtime_and_emit_receipt(
        &mut self,
        completed_end_step_exclusive: u64,
    ) -> Result<LambdaRuntimeCloseoutReceipt, String> {
        if self.logical_output_dir_override.is_some() {
            return Err("lambda closeout cannot run inside an NCMC staging transaction".into());
        }
        self.close_active_no_probe_interval()?;
        let binding = self
            .runtime_binding
            .as_ref()
            .ok_or_else(|| "manual emitter cannot publish runtime closeout".to_string())?
            .clone();
        let terminal = binding
            .protocol_schedule
            .last()
            .ok_or_else(|| "lambda runtime has no admitted terminal phase".to_string())?
            .clone();
        if completed_end_step_exclusive != terminal.protocol_phase.end_step_exclusive
            || self.active_protocol_phase.as_ref() != Some(&terminal.protocol_phase)
            || self.active_thermodynamic_condition.as_ref()
                != Some(&terminal.thermodynamic_condition)
        {
            return Err(
                "lambda runtime closeout is not bound to the admitted terminal phase".into(),
            );
        }
        let closeout_path = self
            .output_dir
            .join("prismos_lambda_runtime_closeout_receipt.json");
        if closeout_path.exists() {
            let existing: LambdaRuntimeCloseoutReceipt = serde_json::from_slice(
                &read_regular_nofollow_under_root(&self.output_dir, &closeout_path)?,
            )
            .map_err(|error| error.to_string())?;
            existing.validate()?;
            if existing.run_id != binding.run_id
                || existing.replica_id != binding.replica_id
                || existing.stream_id != binding.stream_id as u16
                || existing.terminal_step_exclusive != completed_end_step_exclusive
            {
                return Err("existing lambda runtime closeout belongs to another run".into());
            }
            return Ok(existing);
        }

        let mut manifest_paths = Vec::new();
        collect_lambda_journal_manifests(&self.output_dir, &self.output_dir, &mut manifest_paths)?;
        let loaded = if manifest_paths.is_empty() {
            None
        } else {
            Some(load_verified_lambda_event_journals(&self.output_dir)?)
        };
        let (produced, persisted, decoded, indexed, committed, partial, population_sha256) =
            if let Some(loaded) = &loaded {
                (
                    loaded.produced,
                    loaded.persisted,
                    loaded.decoded,
                    loaded.indexed,
                    loaded.committed_resolution_batch_count,
                    loaded.partial_resolution_chain_count,
                    loaded.population_receipt_sha256.clone(),
                )
            } else {
                let zero_population = serde_json::json!({
                    "schema_kind": "prismos_lambda_journal_population_receipt",
                    "schema_version": 2,
                    "produced": 0,
                    "persisted": 0,
                    "decoded": 0,
                    "indexed": 0,
                    "committed_resolution_batch_count": 0,
                    "partial_resolution_chain_count": 0,
                    "stream_terminal_manifest_sha256": BTreeMap::<u16, String>::new(),
                    "stream_phase_contexts": BTreeMap::<u16, Vec<LambdaJournalSegmentContext>>::new(),
                });
                let bytes =
                    serde_json::to_vec(&zero_population).map_err(|error| error.to_string())?;
                (0, 0, 0, 0, 0, 0, format!("{:x}", Sha256::digest(bytes)))
            };
        if produced != self.next_sequence {
            return Err(format!(
                "lambda closeout producer/loader count mismatch: emitter={} loaded={produced}",
                self.next_sequence
            ));
        }
        let scheduled_ncmc_opportunity_count =
            binding.ncmc_scheduler.as_ref().map_or(0, |scheduler| {
                scheduler.scheduled_opportunities.len() as u64
            });
        if committed != scheduled_ncmc_opportunity_count {
            return Err(format!(
                "lambda closeout NCMC conservation failed: scheduled={scheduled_ncmc_opportunity_count} committed={committed}"
            ));
        }

        let mut manifests = Vec::with_capacity(manifest_paths.len());
        for path in manifest_paths {
            let manifest: LambdaJournalSegmentManifest =
                serde_json::from_slice(&read_regular_nofollow_under_root(&self.output_dir, &path)?)
                    .map_err(|error| error.to_string())?;
            manifest.validate()?;
            if manifest.context.stream_id != binding.stream_id as u16 {
                return Err("lambda closeout discovered a foreign stream manifest".into());
            }
            manifests.push(manifest);
        }
        manifests.sort_by_key(|manifest| manifest.record.event_sequence);
        let artifacts = manifests
            .iter()
            .map(|manifest| LambdaRuntimeCloseoutArtifact {
                event_sequence: manifest.record.event_sequence,
                event_kind: manifest.record.event_kind,
                emission_receipt_sha256: manifest.record.emission_receipt_sha256.clone(),
                raw_event_sha256: manifest.record.raw_event_sha256.clone(),
                packet_sha256: manifest.record.packet_sha256.clone(),
                journal_record_sha256: manifest.record.journal_record_sha256.clone(),
                segment_sha256: manifest.segment_sha256.clone(),
                segment_manifest_sha256: manifest.manifest_sha256.clone(),
            })
            .collect::<Vec<_>>();
        let mut artifact_population = Sha256::new();
        artifact_population.update(b"PRISM_LAMBDA_RUNTIME_CLOSEOUT_ARTIFACT_POPULATION_V1\0");
        for artifact in &artifacts {
            artifact_population
                .update(serde_json::to_vec(artifact).map_err(|error| error.to_string())?);
        }
        let terminal_phase_protocol_sha256 =
            crate::chemical_state_fiber::phase_protocol_binding_sha256(
                &terminal.protocol_phase,
                &terminal.thermodynamic_condition,
            )
            .map_err(|error| error.to_string())?;
        let matched_initialization = self
            .matched_velocity_initialization_receipt
            .as_ref()
            .ok_or_else(|| {
                "lambda runtime closeout lacks matched velocity initialization receipt".to_string()
            })?;
        matched_initialization.validate()?;
        let mut receipt = LambdaRuntimeCloseoutReceipt {
            schema_kind: "prismos_lambda_runtime_closeout_receipt".into(),
            schema_version: 1,
            run_id: binding.run_id,
            replica_id: binding.replica_id,
            stream_id: binding.stream_id as u16,
            lane_contract_sha256: binding.lane.contract_sha256,
            terminal_step_exclusive: completed_end_step_exclusive,
            terminal_protocol_phase: terminal.protocol_phase,
            terminal_thermodynamic_condition: terminal.thermodynamic_condition,
            terminal_phase_protocol_sha256,
            produced,
            persisted,
            decoded,
            indexed,
            first_event_sequence: artifacts.first().map(|value| value.event_sequence),
            last_event_sequence: artifacts.last().map(|value| value.event_sequence),
            first_emission_receipt_sha256: artifacts
                .first()
                .map(|value| value.emission_receipt_sha256.clone()),
            last_emission_receipt_sha256: artifacts
                .last()
                .map(|value| value.emission_receipt_sha256.clone()),
            terminal_segment_manifest_sha256: artifacts
                .last()
                .map(|value| value.segment_manifest_sha256.clone()),
            committed_resolution_batch_count: committed,
            partial_resolution_chain_count: partial,
            scheduled_ncmc_opportunity_count,
            matched_velocity_initialization_receipt_sha256: matched_initialization
                .receipt_sha256
                .clone(),
            environment_velocity_sha256: matched_initialization.environment_velocity_sha256.clone(),
            full_velocity_sha256: matched_initialization.full_velocity_sha256.clone(),
            environment_position_sha256: matched_initialization.environment_position_sha256.clone(),
            full_position_sha256: matched_initialization.full_position_sha256.clone(),
            journal_population_receipt_sha256: population_sha256,
            artifacts,
            artifact_population_sha256: format!("{:x}", artifact_population.finalize()),
            completion_status: "COMPLETE".into(),
            receipt_sha256: String::new(),
        };
        receipt.receipt_sha256 = receipt.recompute_sha256()?;
        receipt.validate()?;
        let mut bytes = serde_json::to_vec_pretty(&receipt).map_err(|error| error.to_string())?;
        bytes.push(b'\n');
        atomic_write_synced(&closeout_path, &bytes)?;
        Ok(receipt)
    }

    pub(crate) fn active_phase_end_step_exclusive(&self) -> Option<u64> {
        self.active_protocol_phase
            .as_ref()
            .map(|phase| phase.end_step_exclusive)
    }

    /// Next sequence that must be used by the engine-owned event producer.
    pub(crate) fn next_sequence(&self) -> u64 {
        self.next_sequence
    }

    /// Parent receipt required by the next event.
    pub(crate) fn last_receipt_sha256(&self) -> Option<&str> {
        self.last_receipt_sha256.as_deref()
    }

    /// Digest of the engine-owned phase/condition interval currently
    /// governing emission.  Event producers must obtain it here rather than
    /// retaining the constructor's initial phase after a protocol transition.
    pub(crate) fn active_phase_protocol_sha256(&self) -> Result<String, String> {
        let phase = self
            .active_protocol_phase
            .as_ref()
            .ok_or_else(|| "lambda emitter lacks an active protocol phase".to_string())?;
        let condition = self
            .active_thermodynamic_condition
            .as_ref()
            .ok_or_else(|| "lambda emitter lacks an active thermodynamic condition".to_string())?;
        crate::chemical_state_fiber::phase_protocol_binding_sha256(phase, condition)
            .map_err(|error| error.to_string())
    }

    pub(crate) fn begin_ncmc_resolution_batch(
        &mut self,
        stream_id: u32,
        step: u64,
        move_id: u64,
    ) -> Result<LambdaNcmcResolutionBatchTransaction, String> {
        if self.runtime_binding.is_none() || self.logical_output_dir_override.is_some() {
            return Err("NCMC resolution batch requires one idle runtime emitter".into());
        }
        let original_output_dir = self.output_dir.clone();
        let leaf =
            format!("ncmc_resolution_stream_{stream_id:04}_step_{step:012}_move_{move_id:012}");
        let committed_dir = original_output_dir.join(&leaf);
        let staging_dir = original_output_dir.join(format!(".lambda_resolution_stage_{leaf}"));
        if committed_dir.exists() {
            return Err("NCMC resolution move key is already atomically committed".into());
        }
        if staging_dir.exists() {
            return Err("NCMC resolution has a prior incomplete staging transaction".into());
        }
        let binding = self.runtime_binding.as_ref().unwrap();
        secure_create_lambda_output_dir(&binding.run_output_root, &staging_dir)?;
        let transaction = LambdaNcmcResolutionBatchTransaction {
            original_output_dir,
            staging_dir: staging_dir.clone(),
            committed_dir: committed_dir.clone(),
            next_sequence: self.next_sequence,
            last_receipt_sha256: self.last_receipt_sha256.clone(),
            next_segment_sequence: self.next_segment_sequence,
            last_segment_manifest_sha256: self.last_segment_manifest_sha256.clone(),
            emitted_total_bytes: self.emitted_total_bytes,
        };
        self.output_dir = staging_dir;
        self.logical_output_dir_override = Some(committed_dir);
        Ok(transaction)
    }

    pub(crate) fn abort_ncmc_resolution_batch(
        &mut self,
        transaction: LambdaNcmcResolutionBatchTransaction,
    ) -> Result<(), String> {
        self.output_dir = transaction.original_output_dir;
        self.logical_output_dir_override = None;
        self.next_sequence = transaction.next_sequence;
        self.last_receipt_sha256 = transaction.last_receipt_sha256;
        self.next_segment_sequence = transaction.next_segment_sequence;
        self.last_segment_manifest_sha256 = transaction.last_segment_manifest_sha256;
        self.emitted_total_bytes = transaction.emitted_total_bytes;
        if transaction.staging_dir.exists() {
            std::fs::remove_dir_all(&transaction.staging_dir)
                .map_err(|error| format!("remove failed NCMC staging transaction: {error}"))?;
        }
        Ok(())
    }

    pub(crate) fn commit_ncmc_resolution_batch(
        &mut self,
        transaction: LambdaNcmcResolutionBatchTransaction,
        stream_id: u32,
        step: u64,
        move_id: u64,
        ncmc_move_receipt_sha256: String,
        terminal_emission_receipt_sha256: String,
    ) -> Result<(), String> {
        self.commit_ncmc_resolution_batch_impl(
            transaction,
            stream_id,
            step,
            move_id,
            ncmc_move_receipt_sha256,
            terminal_emission_receipt_sha256,
            false,
        )
    }

    #[cfg(test)]
    fn commit_ncmc_resolution_batch_inject_after_rename(
        &mut self,
        transaction: LambdaNcmcResolutionBatchTransaction,
        stream_id: u32,
        step: u64,
        move_id: u64,
        ncmc_move_receipt_sha256: String,
        terminal_emission_receipt_sha256: String,
    ) -> Result<(), String> {
        self.commit_ncmc_resolution_batch_impl(
            transaction,
            stream_id,
            step,
            move_id,
            ncmc_move_receipt_sha256,
            terminal_emission_receipt_sha256,
            true,
        )
    }

    fn commit_ncmc_resolution_batch_impl(
        &mut self,
        transaction: LambdaNcmcResolutionBatchTransaction,
        stream_id: u32,
        step: u64,
        move_id: u64,
        ncmc_move_receipt_sha256: String,
        terminal_emission_receipt_sha256: String,
        inject_failure_after_rename: bool,
    ) -> Result<(), String> {
        validate_sha256("NCMC move object", &ncmc_move_receipt_sha256)?;
        validate_sha256(
            "NCMC terminal emission receipt",
            &terminal_emission_receipt_sha256,
        )?;
        if self.output_dir != transaction.staging_dir
            || self.logical_output_dir_override.as_ref() != Some(&transaction.committed_dir)
            || self.next_sequence != transaction.next_sequence.saturating_add(3)
        {
            return Err(
                "NCMC resolution batch does not contain exactly three ordered events".into(),
            );
        }
        let mut members = Vec::new();
        for entry in std::fs::read_dir(&transaction.staging_dir)
            .map_err(|error| format!("list NCMC staging members: {error}"))?
        {
            let entry = entry.map_err(|error| error.to_string())?;
            let kind = entry.file_type().map_err(|error| error.to_string())?;
            let name = entry.file_name().to_string_lossy().to_string();
            if kind.is_symlink() {
                return Err("NCMC staging directory contains a symlink".into());
            }
            if kind.is_file()
                && name.starts_with("lambda_journal_stream_")
                && name.ends_with(".manifest.json")
            {
                let segment: LambdaJournalSegmentManifest = serde_json::from_slice(
                    &std::fs::read(entry.path()).map_err(|error| error.to_string())?,
                )
                .map_err(|error| error.to_string())?;
                segment.validate()?;
                members.push(LambdaNcmcResolutionBatchMember {
                    event_sequence: segment.record.event_sequence,
                    event_kind: segment.record.event_kind,
                    segment_manifest_sha256: segment.manifest_sha256,
                    journal_record_sha256: segment.record.journal_record_sha256,
                    emission_receipt_sha256: segment.record.emission_receipt_sha256,
                });
            }
        }
        members.sort_by_key(|member| member.event_sequence);
        if members.len() != 3 {
            return Err("NCMC resolution staging does not contain three member manifests".into());
        }
        let mut manifest = LambdaNcmcResolutionBatchCommitManifest {
            schema_kind: "prismos_lambda_ncmc_resolution_batch_commit".into(),
            schema_version: 1,
            stream_id,
            step,
            move_id,
            first_event_sequence: transaction.next_sequence,
            event_count: 3,
            ncmc_move_receipt_sha256,
            terminal_emission_receipt_sha256,
            members,
            atomic_directory_publish: true,
            manifest_sha256: String::new(),
        };
        manifest.manifest_sha256 = manifest.recompute_sha256()?;
        manifest.validate()?;
        let manifest_bytes =
            serde_json::to_vec_pretty(&manifest).map_err(|error| error.to_string())?;
        let projected = self
            .emitted_total_bytes
            .checked_add(manifest_bytes.len() as u64)
            .ok_or_else(|| "NCMC resolution batch byte count overflow".to_string())?;
        let binding = self.runtime_binding.as_ref().unwrap();
        if projected > binding.typed_event_capacity.maximum_total_bytes {
            return Err("NCMC resolution batch commit exceeds campaign capacity".into());
        }
        atomic_write_synced(
            &transaction
                .staging_dir
                .join("NCMC_RESOLUTION_BATCH_COMMIT.json"),
            &manifest_bytes,
        )?;
        File::open(&transaction.staging_dir)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| format!("fsync NCMC staging directory: {error}"))?;
        let parent_directory = File::open(&transaction.original_output_dir)
            .map_err(|error| format!("open NCMC resolution parent: {error}"))?;
        std::fs::rename(&transaction.staging_dir, &transaction.committed_dir)
            .map_err(|error| format!("atomically publish NCMC resolution directory: {error}"))?;
        if inject_failure_after_rename {
            std::fs::rename(&transaction.committed_dir, &transaction.staging_dir)
                .map_err(|error| format!("rollback injected post-rename NCMC failure: {error}"))?;
            return Err("injected crash after NCMC directory rename before parent fsync".into());
        }
        if let Err(error) = parent_directory.sync_all() {
            std::fs::rename(&transaction.committed_dir, &transaction.staging_dir).map_err(
                |rollback_error| {
                    format!(
                        "fsync NCMC resolution parent failed ({error}) and publish rollback failed ({rollback_error})"
                    )
                },
            )?;
            return Err(format!("fsync NCMC resolution parent: {error}"));
        }
        self.output_dir = transaction.original_output_dir;
        self.logical_output_dir_override = None;
        self.emitted_total_bytes = projected;
        Ok(())
    }

    /// Emit a force/dUdl event.
    pub fn emit_force(
        &mut self,
        event: &LambdaFragmentForceEvent,
    ) -> Result<LambdaTypedEmissionReceipt, String> {
        self.emit(event)
    }

    /// Emit a detailed-balance NCMC move receipt.
    pub fn emit_ncmc(
        &mut self,
        event: &NcmcMoveReceipt,
    ) -> Result<LambdaTypedEmissionReceipt, String> {
        if let Some(binding) = &self.runtime_binding {
            let scheduler = binding
                .ncmc_scheduler
                .as_ref()
                .ok_or_else(|| "runtime NCMC emission lacks scheduler authority".to_string())?;
            event
                .rng_consumption
                .validate_for_scheduler(&event.proposal_projection(), scheduler)?;
        }
        self.emit(event)
    }

    /// Emit an endpoint-only observation.
    pub fn emit_endpoint(
        &mut self,
        event: &LambdaFragmentEndpointObservation,
    ) -> Result<LambdaTypedEmissionReceipt, String> {
        self.emit(event)
    }

    /// Emit an accepted state commit.
    pub fn emit_state_commit(
        &mut self,
        event: &LambdaFragmentStateCommit,
    ) -> Result<LambdaTypedEmissionReceipt, String> {
        self.emit(event)
    }

    /// Emit an explicit rejected-move rollback/no-change authority.
    pub fn emit_state_rollback(
        &mut self,
        event: &LambdaFragmentStateRollback,
    ) -> Result<LambdaTypedEmissionReceipt, String> {
        self.emit(event)
    }

    /// Emit a closed no-probe interval. The runtime binding must itself prove
    /// a physical no-probe lane with no fragment system.
    pub fn emit_no_probe_absence(
        &mut self,
        event: &NoProbeAbsenceReceipt,
    ) -> Result<LambdaTypedEmissionReceipt, String> {
        let binding = self
            .runtime_binding
            .as_ref()
            .ok_or_else(|| "manual emitter cannot prove no-probe runtime absence".to_string())?;
        if binding.lane.lane_kind != LambdaExperimentLaneKind::NoProbe
            || self.lambda_system.is_some()
            || event.lane_contract_sha256 != binding.lane.contract_sha256
        {
            return Err("no-probe receipt disagrees with runtime lane/topology".into());
        }
        let active_phase = self
            .active_protocol_phase
            .as_ref()
            .ok_or_else(|| "no-probe emitter lacks an active protocol phase".to_string())?;
        if event.start_step != active_phase.start_step
            || event.end_step_exclusive != active_phase.end_step_exclusive
        {
            return Err("no-probe absence must close the complete active phase".into());
        }
        let phase_digest = event.phase_protocol_sha256.clone();
        let receipt = self.emit(event)?;
        self.closed_no_probe_phase_digests.insert(phase_digest);
        Ok(receipt)
    }

    fn emit<T: LambdaTypedEvent>(
        &mut self,
        event: &T,
    ) -> Result<LambdaTypedEmissionReceipt, String> {
        event.validate_event()?;
        if event.event_sequence() != self.next_sequence {
            return Err(format!(
                "lambda event sequence gap: got {} expected {}",
                event.event_sequence(),
                self.next_sequence
            ));
        }
        if event.parent_receipt() != self.last_receipt_sha256.as_deref() {
            return Err("lambda event parent receipt does not match prior emission".into());
        }
        if let Some(binding) = &self.runtime_binding {
            if self.next_sequence >= binding.typed_event_capacity.maximum_event_count {
                return Err(
                    "lambda typed-event population exceeded admitted campaign bound".into(),
                );
            }
        }
        if let Some(binding) = &self.runtime_binding {
            let phase = self
                .active_protocol_phase
                .as_ref()
                .ok_or_else(|| "runtime lambda emitter lacks active protocol phase".to_string())?;
            let condition = self
                .active_thermodynamic_condition
                .as_ref()
                .ok_or_else(|| {
                    "runtime lambda emitter lacks thermodynamic condition".to_string()
                })?;
            let expected_phase =
                crate::chemical_state_fiber::phase_protocol_binding_sha256(phase, condition)
                    .map_err(|error| error.to_string())?;
            let step = event.step_or_interval_start();
            if event.stream_id() != binding.stream_id
                || event.phase_protocol_sha256() != expected_phase
                || step < phase.start_step
                || step >= phase.end_step_exclusive
            {
                return Err(
                    "lambda event does not match active stream/phase/condition interval".into(),
                );
            }
        }
        if self.runtime_binding.is_some() {
            // Re-open/re-validate every transition boundary so a replaced
            // output path cannot silently redirect scientific evidence.
            let binding = self.runtime_binding.as_ref().unwrap();
            secure_create_lambda_output_dir(&binding.run_output_root, &self.output_dir)?;
        } else {
            std::fs::create_dir_all(&self.output_dir).map_err(|error| error.to_string())?;
        }
        let mut raw = serde_json::to_vec(event).map_err(|error| error.to_string())?;
        raw.push(b'\n');
        let stem = format!(
            "lambda_{:012}_{}",
            event.event_sequence(),
            event.family().as_str().to_ascii_lowercase()
        );
        let packet_storage_path = self.output_dir.join(format!("{stem}.prismqnanl"));
        let qnanl_receipt_storage_path = self.output_dir.join(format!("{stem}.receipt.json"));
        let logical_output_dir = self
            .logical_output_dir_override
            .as_ref()
            .unwrap_or(&self.output_dir);
        let packet_path = logical_output_dir.join(format!("{stem}.prismqnanl"));
        let qnanl_receipt_path = logical_output_dir.join(format!("{stem}.receipt.json"));
        if packet_storage_path.exists() || qnanl_receipt_storage_path.exists() {
            return Err("lambda typed event would overwrite an existing packet/receipt".into());
        }
        let raw_sha_for_temp = format!("{:x}", Sha256::digest(&raw));
        let encode_dir = self.output_dir.join(format!(
            ".lambda_encode_{:012}_{}",
            event.event_sequence(),
            &raw_sha_for_temp[..16]
        ));
        std::fs::create_dir(&encode_dir)
            .map_err(|error| format!("create-new lambda encode transaction directory: {error}"))?;
        let encode_packet_path = encode_dir.join("packet.prismqnanl");
        let encode_receipt_path = encode_dir.join("receipt.json");
        let options = QnanlEncodeOptions::new(
            &stem,
            event.family(),
            None,
            self.runtime_origin,
            "LambdaTypedEventEmitter::emit",
        );
        let mut qnanl_receipt =
            encode_bytes(&raw, &encode_packet_path, &encode_receipt_path, options)
                .map_err(|error| error.to_string())?;
        let packet = std::fs::read(&encode_packet_path).map_err(|error| error.to_string())?;
        let decoded = decode_packet_bytes(&packet).map_err(|error| error.to_string())?;
        let exact = decoded == raw
            && qnanl_receipt.roundtrip_verified
            && qnanl_receipt.codec_roundtrip_verified;
        if !exact {
            return Err("lambda typed QNANL exact-byte roundtrip failed".into());
        }
        qnanl_receipt.packet_path = packet_path.display().to_string();
        qnanl_receipt.receipt_path = qnanl_receipt_path.display().to_string();
        let qnanl_receipt_bytes = qnanl_receipt.to_json_pretty().into_bytes();
        let raw_sha256 = format!("{:x}", Sha256::digest(&raw));
        let packet_sha256 = format!("{:x}", Sha256::digest(&packet));
        let opcode_id = opcode_id_for_name(event.opcode())
            .ok_or_else(|| format!("PrismCode registry lacks {}", event.opcode()))?;
        let mut receipt_digest = Sha256::new();
        receipt_digest.update(b"PRISM_LAMBDA_TYPED_EMISSION_RECEIPT_V1\0");
        receipt_digest.update(event.event_sequence().to_le_bytes());
        if let Some(parent) = event.parent_receipt() {
            receipt_digest.update(parent.as_bytes());
        }
        receipt_digest.update(event.opcode().as_bytes());
        receipt_digest.update(opcode_id.to_le_bytes());
        receipt_digest.update((event.family() as u16).to_le_bytes());
        receipt_digest.update(raw_sha256.as_bytes());
        receipt_digest.update(packet_sha256.as_bytes());
        let receipt_sha256 = format!("{:x}", receipt_digest.finalize());
        let receipt = LambdaTypedEmissionReceipt {
            schema_kind: "prismos_lambda_typed_emission_receipt".into(),
            schema_version: 1,
            status: "PASS".into(),
            event_sequence: event.event_sequence(),
            parent_receipt_sha256: event.parent_receipt().map(str::to_string),
            prismcode_opcode: event.opcode().into(),
            prismcode_opcode_id: opcode_id,
            qnanl_artifact_family: event.family().as_str().into(),
            qnanl_artifact_family_code: event.family() as u16,
            runtime_origin: self.runtime_origin.as_str().into(),
            raw_sha256,
            packet_sha256,
            packet_path: packet_path.clone(),
            qnanl_receipt_path: qnanl_receipt_path.clone(),
            exact_byte_roundtrip: exact,
            receipt_sha256: receipt_sha256.clone(),
        };
        atomic_write_synced(&packet_storage_path, &packet)?;
        atomic_write_synced(&qnanl_receipt_storage_path, &qnanl_receipt_bytes)?;
        let mut durable_event_bytes = (packet.len() + qnanl_receipt_bytes.len()) as u64;
        let prior_segment_sequence = self.next_segment_sequence;
        let prior_manifest = self.last_segment_manifest_sha256.clone();
        let journal_paths_and_bytes = if self.runtime_binding.is_some() {
            let result = self.persist_runtime_journal_segment(event, &receipt, &raw, &packet)?;
            durable_event_bytes = durable_event_bytes
                .checked_add(result.2)
                .ok_or_else(|| "lambda durable event byte count overflow".to_string())?;
            Some(result)
        } else {
            None
        };
        if let Some(binding) = &self.runtime_binding {
            let projected = self
                .emitted_total_bytes
                .checked_add(durable_event_bytes)
                .ok_or_else(|| "lambda campaign byte count overflow".to_string())?;
            if durable_event_bytes > binding.typed_event_capacity.maximum_event_bytes
                || projected > binding.typed_event_capacity.maximum_total_bytes
            {
                for path in [&packet_storage_path, &qnanl_receipt_storage_path] {
                    let _ = std::fs::remove_file(path);
                }
                if let Some((segment_path, manifest_path, _)) = &journal_paths_and_bytes {
                    let _ = std::fs::remove_file(segment_path);
                    let _ = std::fs::remove_file(manifest_path);
                }
                self.next_segment_sequence = prior_segment_sequence;
                self.last_segment_manifest_sha256 = prior_manifest;
                return Err("lambda event exceeded admitted durable byte capacity".into());
            }
            self.emitted_total_bytes = projected;
        }
        for path in [&encode_packet_path, &encode_receipt_path] {
            let _ = std::fs::remove_file(path);
        }
        let _ = std::fs::remove_dir(&encode_dir);
        self.last_receipt_sha256 = Some(receipt_sha256);
        self.next_sequence += 1;
        Ok(receipt)
    }

    fn persist_runtime_journal_segment<T: LambdaTypedEvent>(
        &mut self,
        event: &T,
        receipt: &LambdaTypedEmissionReceipt,
        raw: &[u8],
        packet: &[u8],
    ) -> Result<(PathBuf, PathBuf, u64), String> {
        let binding = self.runtime_binding.as_ref().unwrap();
        let phase = self.active_protocol_phase.as_ref().unwrap();
        let condition = self.active_thermodynamic_condition.as_ref().unwrap();
        let stream_id = u16::try_from(event.stream_id())
            .map_err(|_| "lambda journal stream ID exceeds u16".to_string())?;
        let event_kind = match event.family() {
            QnanlArtifactFamily::LambdaFragmentForceEvent => LambdaJournalEventKind::Force,
            QnanlArtifactFamily::LambdaFragmentNcmcMoveReceipt => LambdaJournalEventKind::NcmcMove,
            QnanlArtifactFamily::LambdaFragmentEndpointObservation => {
                LambdaJournalEventKind::Endpoint
            }
            QnanlArtifactFamily::LambdaFragmentStateCommit => LambdaJournalEventKind::StateCommit,
            QnanlArtifactFamily::LambdaFragmentStateRollback => {
                LambdaJournalEventKind::StateRollback
            }
            QnanlArtifactFamily::LambdaNoProbeAbsence => LambdaJournalEventKind::NoProbeAbsence,
            _ => return Err("unsupported lambda journal QNANL family".into()),
        };
        let key = event.slot_id().map(|slot_id| LambdaJournalKey {
            stream_id,
            step_idx: event.step_or_interval_start(),
            slot_id,
        });
        let calibration = event.calibration_identity().and_then(|(species, state)| {
            self.lambda_system.as_ref().and_then(|system| {
                system
                    .chemical_potentials
                    .iter()
                    .find(|value| {
                        value.species_id == species
                            && value.chemical_state_id == state
                            && value.temperature_k.to_bits() == condition.temperature_k.to_bits()
                    })
                    .cloned()
            })
        });
        let calibration_work_collection = binding
            .ncmc_execution_mode
            .as_ref()
            .map(NcmcExecutionMode::is_calibration_work_collection)
            .unwrap_or(false);
        if event.calibration_identity().is_some()
            && calibration.is_none()
            && !calibration_work_collection
        {
            return Err(
                "lambda journal event lacks exact active chemical-potential calibration".into(),
            );
        }
        let packet_file = receipt
            .packet_path
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or_else(|| "lambda packet path lacks a normalized filename".to_string())?
            .to_string();
        let mut record = LambdaJournalRecord {
            schema_kind: "prismos_lambda_event_journal_record".into(),
            schema_version: 1,
            segment_sequence: self.next_segment_sequence,
            key,
            event_kind,
            event_sequence: event.event_sequence(),
            parent_receipt_sha256: event.parent_receipt().map(str::to_string),
            emission_receipt_sha256: receipt.receipt_sha256.clone(),
            raw_event_sha256: format!("{:x}", Sha256::digest(raw)),
            raw_event_bytes: raw.len() as u64,
            packet_sha256: format!("{:x}", Sha256::digest(packet)),
            packet_file,
            qnanl_artifact_family: event.family().as_str().into(),
            phase_protocol_sha256: event.phase_protocol_sha256().into(),
            chemical_potential_calibration: calibration,
            journal_record_sha256: String::new(),
        };
        record.journal_record_sha256 = record.recompute_sha256()?;
        record.validate()?;
        let mut segment_bytes = serde_json::to_vec(&record).map_err(|error| error.to_string())?;
        segment_bytes.push(b'\n');
        let segment_sha256 = format!("{:x}", Sha256::digest(&segment_bytes));
        let record_bytes_sha256 = segment_sha256.clone();
        let segment_stem = format!(
            "lambda_journal_stream_{:04}_segment_{:012}",
            stream_id, self.next_segment_sequence
        );
        let segment_file_name = format!("{segment_stem}.jsonl");
        let manifest_file_name = format!("{segment_stem}.manifest.json");
        let system_sha256 = self
            .lambda_system
            .as_ref()
            .map(|system| {
                serde_json::to_vec(system)
                    .map(|bytes| format!("{:x}", Sha256::digest(bytes)))
                    .map_err(|error| error.to_string())
            })
            .transpose()?;
        let context = LambdaJournalSegmentContext {
            run_id: binding.run_id.clone(),
            replica_id: binding.replica_id.clone(),
            replica_seed: binding.replica_seed,
            stream_id,
            stream_seed: binding.stream_seed,
            engine_sha256: binding.engine_sha256.clone(),
            topology_sha256: binding.topology_sha256.clone(),
            lane: binding.lane.clone(),
            ncmc_execution_mode: binding.ncmc_execution_mode.clone(),
            ncmc_scheduler: binding.ncmc_scheduler.clone(),
            lambda_fragment_system_sha256: system_sha256,
            protocol_phase: phase.clone(),
            thermodynamic_condition: condition.clone(),
            phase_protocol_sha256: event.phase_protocol_sha256().into(),
        };
        let mut manifest = LambdaJournalSegmentManifest {
            schema_kind: "prismos_lambda_event_journal_segment_manifest".into(),
            schema_version: 1,
            segment_sequence: self.next_segment_sequence,
            previous_segment_manifest_sha256: self.last_segment_manifest_sha256.clone(),
            context,
            record,
            byte_lineage: LambdaJournalByteLineage {
                segment_file: segment_file_name.clone(),
                byte_offset: 0,
                byte_length: segment_bytes.len() as u64,
                record_bytes_sha256,
            },
            segment_sha256,
            segment_bytes: segment_bytes.len() as u64,
            manifest_sha256: String::new(),
        };
        manifest.manifest_sha256 = manifest.recompute_sha256()?;
        manifest.validate()?;
        std::fs::create_dir_all(&self.output_dir).map_err(|error| error.to_string())?;
        let segment_path = self.output_dir.join(&segment_file_name);
        let manifest_path = self.output_dir.join(&manifest_file_name);
        atomic_write_synced(&segment_path, &segment_bytes)?;
        let mut manifest_bytes =
            serde_json::to_vec_pretty(&manifest).map_err(|error| error.to_string())?;
        manifest_bytes.push(b'\n');
        atomic_write_synced(&manifest_path, &manifest_bytes)?;
        self.last_segment_manifest_sha256 = Some(manifest.manifest_sha256);
        self.next_segment_sequence += 1;
        Ok((
            segment_path,
            manifest_path,
            (segment_bytes.len() + manifest_bytes.len()) as u64,
        ))
    }
}

fn collect_lambda_journal_manifests(
    run_root: &Path,
    directory: &Path,
    output: &mut Vec<PathBuf>,
) -> Result<(), String> {
    for entry in std::fs::read_dir(directory).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let metadata = entry.file_type().map_err(|error| error.to_string())?;
        if metadata.is_symlink() {
            return Err(format!(
                "lambda journal loader refuses symlink {}",
                entry.path().display()
            ));
        }
        if metadata.is_dir() {
            let directory_name = entry.file_name().to_string_lossy().to_string();
            if directory_name.starts_with(".lambda_resolution_stage_") {
                return Err(format!(
                    "partial NCMC resolution staging directory requires out-of-band recovery classification: {}",
                    entry.path().display()
                ));
            }
            if directory_name.starts_with("ncmc_resolution_") {
                output.extend(validate_ncmc_resolution_directory(run_root, &entry.path())?);
            } else {
                collect_lambda_journal_manifests(run_root, &entry.path(), output)?;
            }
        } else if metadata.is_file()
            && entry
                .file_name()
                .to_string_lossy()
                .ends_with(".manifest.json")
            && entry
                .file_name()
                .to_string_lossy()
                .starts_with("lambda_journal_stream_")
        {
            output.push(entry.path());
        }
    }
    Ok(())
}

fn validate_ncmc_resolution_directory(
    run_root: &Path,
    directory: &Path,
) -> Result<Vec<PathBuf>, String> {
    let commit_path = directory.join("NCMC_RESOLUTION_BATCH_COMMIT.json");
    let metadata = std::fs::symlink_metadata(&commit_path).map_err(|error| {
        format!(
            "committed-looking NCMC resolution directory lacks commit authority {}: {error}",
            commit_path.display()
        )
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err("NCMC resolution commit authority is not a regular file".into());
    }
    let manifest: LambdaNcmcResolutionBatchCommitManifest =
        serde_json::from_slice(&read_regular_nofollow_under_root(run_root, &commit_path)?)
            .map_err(|error| error.to_string())?;
    manifest.validate()?;
    let expected_leaf = format!(
        "ncmc_resolution_stream_{:04}_step_{:012}_move_{:012}",
        manifest.stream_id, manifest.step, manifest.move_id
    );
    if directory.file_name().and_then(|value| value.to_str()) != Some(expected_leaf.as_str()) {
        return Err("NCMC resolution directory name differs from commit authority".into());
    }
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(directory).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let kind = entry.file_type().map_err(|error| error.to_string())?;
        if kind.is_symlink() {
            return Err("NCMC resolution directory contains a symlink".into());
        }
        let name = entry.file_name().to_string_lossy().to_string();
        if kind.is_file()
            && name.starts_with("lambda_journal_stream_")
            && name.ends_with(".manifest.json")
        {
            paths.push(entry.path());
        } else if kind.is_dir() {
            return Err("NCMC resolution directory contains an unauthorized subdirectory".into());
        }
    }
    if paths.len() != manifest.event_count as usize {
        return Err("NCMC resolution commit does not authorize exactly three segments".into());
    }
    let mut decoded = Vec::with_capacity(paths.len());
    for path in &paths {
        let segment: LambdaJournalSegmentManifest =
            serde_json::from_slice(&read_regular_nofollow_under_root(run_root, path)?)
                .map_err(|error| error.to_string())?;
        segment.validate()?;
        decoded.push(segment);
    }
    decoded.sort_by_key(|segment| segment.record.event_sequence);
    paths.sort_by_key(|path| {
        decoded
            .iter()
            .position(|segment| {
                path.file_name()
                    .and_then(|value| value.to_str())
                    .is_some_and(|name| {
                        name.contains(&format!("segment_{:012}", segment.segment_sequence))
                    })
            })
            .unwrap_or(usize::MAX)
    });
    let expected_kinds = [
        LambdaJournalEventKind::NcmcMove,
        LambdaJournalEventKind::Endpoint,
        decoded[2].record.event_kind,
    ];
    if !matches!(
        expected_kinds[2],
        LambdaJournalEventKind::StateCommit | LambdaJournalEventKind::StateRollback
    ) {
        return Err("NCMC resolution terminal event is neither commit nor rollback".into());
    }
    for (offset, (segment, expected_kind)) in decoded.iter().zip(expected_kinds).enumerate() {
        let expected_sequence = manifest
            .first_event_sequence
            .checked_add(offset as u64)
            .ok_or_else(|| "NCMC resolution sequence overflow".to_string())?;
        if segment.record.event_sequence != expected_sequence
            || segment.record.event_kind != expected_kind
            || segment.record.key.map(|key| u32::from(key.stream_id)) != Some(manifest.stream_id)
            || segment.record.key.map(|key| key.step_idx) != Some(manifest.step)
        {
            return Err("NCMC resolution segment membership/order differs from commit".into());
        }
        let member = &manifest.members[offset];
        if member.event_sequence != segment.record.event_sequence
            || member.event_kind != segment.record.event_kind
            || member.segment_manifest_sha256 != segment.manifest_sha256
            || member.journal_record_sha256 != segment.record.journal_record_sha256
            || member.emission_receipt_sha256 != segment.record.emission_receipt_sha256
        {
            return Err("NCMC resolution content-addressed member differs from commit".into());
        }
    }
    if decoded[2].record.emission_receipt_sha256 != manifest.terminal_emission_receipt_sha256 {
        return Err("NCMC resolution terminal receipt differs from commit authority".into());
    }
    let ncmc_packet = read_regular_nofollow_under_root(
        run_root,
        &directory.join(&decoded[0].record.packet_file),
    )?;
    let raw = decode_packet_bytes(&ncmc_packet).map_err(|error| error.to_string())?;
    let receipt: NcmcMoveReceipt =
        serde_json::from_slice(&raw).map_err(|error| error.to_string())?;
    if receipt.digest_sha256()? != manifest.ncmc_move_receipt_sha256
        || receipt.move_id != manifest.move_id
    {
        return Err("NCMC resolution move object differs from commit authority".into());
    }
    Ok(paths)
}

fn normalized_leaf_name(value: &str) -> Result<(), String> {
    let path = Path::new(value);
    let mut components = path.components();
    match (components.next(), components.next()) {
        (Some(std::path::Component::Normal(_)), None) => Ok(()),
        _ => Err(format!(
            "lambda journal artifact is not a normalized leaf: {value}"
        )),
    }
}

/// Read a production-authority artifact only after proving it is a regular,
/// non-symlink file canonically beneath the run root. O_NOFOLLOW closes the
/// final-component replacement race between metadata inspection and open.
fn read_regular_nofollow_under_root(run_root: &Path, path: &Path) -> Result<Vec<u8>, String> {
    let canonical_root = run_root.canonicalize().map_err(|error| error.to_string())?;
    let symlink_metadata = std::fs::symlink_metadata(path).map_err(|error| error.to_string())?;
    if symlink_metadata.file_type().is_symlink() || !symlink_metadata.is_file() {
        return Err(format!(
            "lambda journal authority is not a regular non-symlink file: {}",
            path.display()
        ));
    }
    let canonical_path = path.canonicalize().map_err(|error| error.to_string())?;
    if !canonical_path.starts_with(&canonical_root) {
        return Err(format!(
            "lambda journal authority escapes run root: {}",
            path.display()
        ));
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
    let mut file = options.open(path).map_err(|error| error.to_string())?;
    let opened_metadata = file.metadata().map_err(|error| error.to_string())?;
    if !opened_metadata.is_file() {
        return Err("opened lambda journal authority is not a regular file".into());
    }
    let mut bytes = Vec::with_capacity(opened_metadata.len() as usize);
    file.read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    Ok(bytes)
}

fn decode_lambda_journal_event(
    record: &LambdaJournalRecord,
    raw: &[u8],
) -> Result<LambdaDecodedJournalEvent, String> {
    macro_rules! decode {
        ($ty:ty, $variant:ident) => {{
            let value: $ty = serde_json::from_slice(raw).map_err(|error| error.to_string())?;
            value.validate()?;
            LambdaDecodedJournalEvent::$variant(value)
        }};
    }
    Ok(match record.event_kind {
        LambdaJournalEventKind::Force => decode!(LambdaFragmentForceEvent, Force),
        LambdaJournalEventKind::NcmcMove => decode!(NcmcMoveReceipt, Ncmc),
        LambdaJournalEventKind::Endpoint => {
            decode!(LambdaFragmentEndpointObservation, Endpoint)
        }
        LambdaJournalEventKind::StateCommit => decode!(LambdaFragmentStateCommit, Commit),
        LambdaJournalEventKind::StateRollback => {
            decode!(LambdaFragmentStateRollback, Rollback)
        }
        LambdaJournalEventKind::NoProbeAbsence => {
            decode!(NoProbeAbsenceReceipt, NoProbeAbsence)
        }
    })
}

fn decoded_event_identity(
    event: &LambdaDecodedJournalEvent,
) -> (u32, u64, Option<u32>, u64, Option<&str>, &str) {
    match event {
        LambdaDecodedJournalEvent::Force(value) => (
            value.stream_id,
            value.step,
            Some(value.slot_id),
            value.event_sequence,
            value.parent_receipt_sha256.as_deref(),
            &value.phase_protocol_sha256,
        ),
        LambdaDecodedJournalEvent::Ncmc(value) => (
            value.stream_id,
            value.step,
            Some(value.slot_id),
            value.event_sequence,
            Some(&value.parent_receipt_sha256),
            &value.phase_protocol_sha256,
        ),
        LambdaDecodedJournalEvent::Endpoint(value) => (
            value.stream_id,
            value.step,
            Some(value.slot_id),
            value.event_sequence,
            Some(&value.parent_receipt_sha256),
            &value.phase_protocol_sha256,
        ),
        LambdaDecodedJournalEvent::Commit(value) => (
            value.stream_id,
            value.step,
            Some(value.slot_id),
            value.event_sequence,
            Some(&value.parent_receipt_sha256),
            &value.phase_protocol_sha256,
        ),
        LambdaDecodedJournalEvent::Rollback(value) => (
            value.stream_id,
            value.step,
            Some(value.slot_id),
            value.event_sequence,
            Some(&value.parent_receipt_sha256),
            &value.phase_protocol_sha256,
        ),
        LambdaDecodedJournalEvent::NoProbeAbsence(value) => (
            value.stream_id,
            value.start_step,
            None,
            value.event_sequence,
            value.parent_receipt_sha256.as_deref(),
            &value.phase_protocol_sha256,
        ),
    }
}

/// Load all immutable lambda journals below a completed run root. Every count
/// is conserved and every packet is exact-byte decoded before indexing.
pub fn load_verified_lambda_event_journals(
    run_root: &Path,
) -> Result<LambdaEventJournalLoad, String> {
    if !run_root.is_dir() {
        return Err(format!(
            "lambda journal run root is absent: {}",
            run_root.display()
        ));
    }
    let mut manifest_paths = Vec::new();
    collect_lambda_journal_manifests(run_root, run_root, &mut manifest_paths)?;
    if manifest_paths.is_empty() {
        return Err("no lambda event journal manifests found".into());
    }
    let mut manifests_by_stream: BTreeMap<u16, Vec<(PathBuf, LambdaJournalSegmentManifest)>> =
        BTreeMap::new();
    for path in manifest_paths {
        let bytes = read_regular_nofollow_under_root(run_root, &path)?;
        let manifest: LambdaJournalSegmentManifest =
            serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
        manifest.validate()?;
        manifests_by_stream
            .entry(manifest.context.stream_id)
            .or_default()
            .push((path, manifest));
    }

    let mut output = LambdaEventJournalLoad {
        produced: 0,
        persisted: 0,
        decoded: 0,
        indexed: 0,
        committed_resolution_batch_count: 0,
        partial_resolution_chain_count: 0,
        observations: BTreeMap::new(),
        no_probe_absence_receipts: Vec::new(),
        stream_terminal_manifest_sha256: BTreeMap::new(),
        stream_contexts: BTreeMap::new(),
        stream_phase_contexts: BTreeMap::new(),
        population_receipt_sha256: String::new(),
    };

    for (stream_id, manifests) in &mut manifests_by_stream {
        manifests.sort_by_key(|(_, manifest)| manifest.segment_sequence);
        let mut expected_segment = 0u64;
        let mut expected_event = 0u64;
        let mut prior_manifest: Option<String> = None;
        let mut prior_receipt: Option<String> = None;
        let mut invariant_context: Option<(&str, &str, u64, u64, &str, &str, &str, Option<&str>)> =
            None;
        let mut terminal_context: Option<LambdaJournalSegmentContext> = None;
        for (manifest_path, manifest) in manifests.iter() {
            output.produced += 1;
            if manifest.segment_sequence != expected_segment
                || manifest.previous_segment_manifest_sha256 != prior_manifest
                || manifest.record.event_sequence != expected_event
                || manifest.record.parent_receipt_sha256 != prior_receipt
            {
                return Err(format!(
                    "lambda journal stream {} has segment/event/parent gap at segment {}",
                    stream_id, manifest.segment_sequence
                ));
            }
            let current_invariants = (
                manifest.context.run_id.as_str(),
                manifest.context.replica_id.as_str(),
                manifest.context.replica_seed,
                manifest.context.stream_seed,
                manifest.context.engine_sha256.as_str(),
                manifest.context.topology_sha256.as_str(),
                manifest.context.lane.contract_sha256.as_str(),
                manifest
                    .context
                    .ncmc_scheduler
                    .as_ref()
                    .map(|scheduler| scheduler.contract_sha256.as_str()),
            );
            if let Some(expected) = invariant_context {
                if expected != current_invariants {
                    return Err("lambda journal immutable stream context changed mid-run".into());
                }
            } else {
                invariant_context = Some(current_invariants);
            }
            validate_sha256("journal engine", &manifest.context.engine_sha256)?;
            validate_sha256("journal topology", &manifest.context.topology_sha256)?;
            let recomputed_phase = crate::chemical_state_fiber::phase_protocol_binding_sha256(
                &manifest.context.protocol_phase,
                &manifest.context.thermodynamic_condition,
            )
            .map_err(|error| error.to_string())?;
            if recomputed_phase != manifest.context.phase_protocol_sha256
                || recomputed_phase != manifest.record.phase_protocol_sha256
                || manifest.context.protocol_phase.stream_id != *stream_id
                || manifest.context.protocol_phase.thermodynamic_condition_id
                    != manifest.context.thermodynamic_condition.condition_id
            {
                return Err("lambda journal phase/condition binding mismatch".into());
            }
            let phases = output.stream_phase_contexts.entry(*stream_id).or_default();
            if let Some(existing) = phases.iter().find(|context| {
                context.phase_protocol_sha256 == manifest.context.phase_protocol_sha256
            }) {
                if existing != &manifest.context {
                    return Err("lambda phase digest aliases different segment contexts".into());
                }
            } else {
                if phases.iter().any(|context| {
                    context.protocol_phase.start_step
                        < manifest.context.protocol_phase.end_step_exclusive
                        && manifest.context.protocol_phase.start_step
                            < context.protocol_phase.end_step_exclusive
                }) {
                    return Err("lambda stream has overlapping protocol-phase contexts".into());
                }
                phases.push(manifest.context.clone());
                phases.sort_by_key(|context| context.protocol_phase.start_step);
            }
            match manifest.context.lane.lane_kind {
                LambdaExperimentLaneKind::NoProbe => {
                    manifest.context.lane.validate(None)?;
                    if manifest.context.lambda_fragment_system_sha256.is_some()
                        || manifest.context.ncmc_scheduler.is_some()
                    {
                        return Err("no-probe journal carries fragment/NCMC authority".into());
                    }
                }
                _ => {
                    manifest
                        .context
                        .ncmc_scheduler
                        .as_ref()
                        .ok_or_else(|| "probe journal lacks NCMC scheduler".to_string())?
                        .validate()?;
                    validate_sha256(
                        "lambda fragment system",
                        manifest
                            .context
                            .lambda_fragment_system_sha256
                            .as_deref()
                            .ok_or_else(|| {
                                "probe journal lacks fragment system digest".to_string()
                            })?,
                    )?;
                }
            }

            let parent = manifest_path
                .parent()
                .ok_or_else(|| "lambda manifest lacks parent directory".to_string())?;
            let segment_path = parent.join(&manifest.byte_lineage.segment_file);
            normalized_leaf_name(&manifest.byte_lineage.segment_file)?;
            let segment_bytes = read_regular_nofollow_under_root(run_root, &segment_path)?;
            if segment_bytes.len() as u64 != manifest.segment_bytes
                || format!("{:x}", Sha256::digest(&segment_bytes)) != manifest.segment_sha256
                || format!("{:x}", Sha256::digest(&segment_bytes))
                    != manifest.byte_lineage.record_bytes_sha256
            {
                return Err("lambda journal segment byte lineage mismatch".into());
            }
            let record: LambdaJournalRecord =
                serde_json::from_slice(&segment_bytes).map_err(|error| error.to_string())?;
            if record.journal_record_sha256 != manifest.record.journal_record_sha256
                || record.emission_receipt_sha256 != manifest.record.emission_receipt_sha256
            {
                return Err("lambda journal segment record differs from manifest".into());
            }
            output.persisted += 1;

            let packet_path = parent.join(&record.packet_file);
            normalized_leaf_name(&record.packet_file)?;
            let packet = read_regular_nofollow_under_root(run_root, &packet_path)?;
            if format!("{:x}", Sha256::digest(&packet)) != record.packet_sha256 {
                return Err("lambda journal QNANL packet digest mismatch".into());
            }
            let raw = decode_packet_bytes(&packet).map_err(|error| error.to_string())?;
            if raw.len() as u64 != record.raw_event_bytes
                || format!("{:x}", Sha256::digest(&raw)) != record.raw_event_sha256
            {
                return Err("lambda journal decoded event byte lineage mismatch".into());
            }
            let event = decode_lambda_journal_event(&record, &raw)?;
            let decoded_family = match &event {
                LambdaDecodedJournalEvent::Force(_) => {
                    QnanlArtifactFamily::LambdaFragmentForceEvent
                }
                LambdaDecodedJournalEvent::Ncmc(_) => {
                    QnanlArtifactFamily::LambdaFragmentNcmcMoveReceipt
                }
                LambdaDecodedJournalEvent::Endpoint(_) => {
                    QnanlArtifactFamily::LambdaFragmentEndpointObservation
                }
                LambdaDecodedJournalEvent::Commit(_) => {
                    QnanlArtifactFamily::LambdaFragmentStateCommit
                }
                LambdaDecodedJournalEvent::Rollback(_) => {
                    QnanlArtifactFamily::LambdaFragmentStateRollback
                }
                LambdaDecodedJournalEvent::NoProbeAbsence(_) => {
                    QnanlArtifactFamily::LambdaNoProbeAbsence
                }
            };
            if record.qnanl_artifact_family != decoded_family.as_str() {
                return Err("journal record QNANL family differs from decoded event".into());
            }
            let (event_stream, event_step, event_slot, event_sequence, event_parent, event_phase) =
                decoded_event_identity(&event);
            if event_stream != u32::from(*stream_id)
                || event_sequence != record.event_sequence
                || event_parent != record.parent_receipt_sha256.as_deref()
                || event_phase != record.phase_protocol_sha256
                || record.key.map(|key| key.step_idx) != event_slot.map(|_| event_step)
                || record.key.map(|key| key.slot_id) != event_slot
            {
                return Err("decoded lambda event identity differs from journal record".into());
            }
            if event_step < manifest.context.protocol_phase.start_step
                || event_step >= manifest.context.protocol_phase.end_step_exclusive
            {
                return Err("decoded lambda event lies outside its protocol phase".into());
            }
            output.decoded += 1;

            let relative_segment = segment_path
                .strip_prefix(run_root)
                .map_err(|_| "lambda segment is outside run root".to_string())?
                .to_string_lossy()
                .to_string();
            let lineage = LambdaJournalLoadedLineage {
                event_kind: record.event_kind,
                event_sequence: record.event_sequence,
                parent_receipt_sha256: record.parent_receipt_sha256.clone(),
                emission_receipt_sha256: record.emission_receipt_sha256.clone(),
                raw_event_sha256: record.raw_event_sha256.clone(),
                segment_manifest_sha256: manifest.manifest_sha256.clone(),
                segment_sha256: manifest.segment_sha256.clone(),
                segment_file: relative_segment,
                byte_offset: manifest.byte_lineage.byte_offset,
                byte_length: manifest.byte_lineage.byte_length,
            };
            match event {
                LambdaDecodedJournalEvent::NoProbeAbsence(receipt) => {
                    if manifest.context.lane.lane_kind != LambdaExperimentLaneKind::NoProbe
                        || receipt.lane_contract_sha256 != manifest.context.lane.contract_sha256
                    {
                        return Err("no-probe absence receipt lane mismatch".into());
                    }
                    if receipt.end_step_exclusive
                        > manifest.context.protocol_phase.end_step_exclusive
                    {
                        return Err(
                            "no-probe absence interval crosses its bound protocol phase".into()
                        );
                    }
                    output.no_probe_absence_receipts.push((receipt, lineage));
                }
                event => {
                    let key = record.key.unwrap();
                    let observation = output.observations.entry(key).or_insert_with(|| {
                        LambdaJournalObservation {
                            key,
                            lane: manifest.context.lane.clone(),
                            force_events: Vec::new(),
                            endpoint_event: None,
                            chemical_potential_calibration: None,
                            ncmc_move_receipt: None,
                            accepted_state_commit: None,
                            rollback_no_change: None,
                            event_lineage: Vec::new(),
                        }
                    });
                    if observation.lane.contract_sha256 != manifest.context.lane.contract_sha256 {
                        return Err("lambda observation lane changed for an exact key".into());
                    }
                    if let Some(calibration) = &record.chemical_potential_calibration {
                        if let Some(existing) = &observation.chemical_potential_calibration {
                            if existing != calibration {
                                return Err("lambda calibration changed for an exact key".into());
                            }
                        } else {
                            observation.chemical_potential_calibration = Some(calibration.clone());
                        }
                    }
                    match event {
                        LambdaDecodedJournalEvent::Force(value) => {
                            let duplicate = observation.force_events.iter().any(|existing| {
                                existing.event_sequence == value.event_sequence
                                    || (existing.ncmc_boundary.is_some()
                                        && existing.ncmc_boundary == value.ncmc_boundary)
                            });
                            if duplicate {
                                return Err(
                                    "duplicate NCMC force boundary for lambda journal key".into()
                                );
                            }
                            observation.force_events.push(value);
                        }
                        LambdaDecodedJournalEvent::Ncmc(value) => {
                            if observation.ncmc_move_receipt.replace(value).is_some() {
                                return Err("duplicate NCMC event for lambda journal key".into());
                            }
                        }
                        LambdaDecodedJournalEvent::Endpoint(value) => {
                            if observation.endpoint_event.replace(value).is_some() {
                                return Err(
                                    "duplicate endpoint event for lambda journal key".into()
                                );
                            }
                        }
                        LambdaDecodedJournalEvent::Commit(value) => {
                            if observation.accepted_state_commit.replace(value).is_some() {
                                return Err("duplicate commit event for lambda journal key".into());
                            }
                        }
                        LambdaDecodedJournalEvent::Rollback(value) => {
                            if observation.rollback_no_change.replace(value).is_some() {
                                return Err(
                                    "duplicate rollback event for lambda journal key".into()
                                );
                            }
                        }
                        LambdaDecodedJournalEvent::NoProbeAbsence(_) => unreachable!(),
                    }
                    observation.event_lineage.push(lineage);
                }
            }
            output.indexed += 1;
            prior_manifest = Some(manifest.manifest_sha256.clone());
            prior_receipt = Some(record.emission_receipt_sha256.clone());
            expected_segment += 1;
            expected_event += 1;
            terminal_context = Some(manifest.context.clone());
        }
        output.stream_terminal_manifest_sha256.insert(
            *stream_id,
            prior_manifest.ok_or_else(|| "empty lambda journal stream".to_string())?,
        );
        output.stream_contexts.insert(
            *stream_id,
            terminal_context.ok_or_else(|| "empty lambda journal stream context".to_string())?,
        );
    }

    let mut admitted_move_keys = BTreeSet::new();
    for observation in output.observations.values() {
        if let Some(move_receipt) = &observation.ncmc_move_receipt {
            let resolution_lineage: Vec<_> = observation
                .event_lineage
                .iter()
                .filter(|lineage| {
                    matches!(
                        lineage.event_kind,
                        LambdaJournalEventKind::NcmcMove
                            | LambdaJournalEventKind::Endpoint
                            | LambdaJournalEventKind::StateCommit
                            | LambdaJournalEventKind::StateRollback
                    )
                })
                .collect();
            if resolution_lineage.len() != 3 {
                return Err("NCMC resolution is not one exact three-event chain".into());
            }
            let resolution_parent = Path::new(&resolution_lineage[0].segment_file)
                .parent()
                .and_then(|path| path.file_name())
                .and_then(|name| name.to_str())
                .ok_or_else(|| "NCMC resolution segment lacks atomic directory".to_string())?;
            if !resolution_parent.starts_with("ncmc_resolution_")
                || resolution_lineage.iter().any(|lineage| {
                    Path::new(&lineage.segment_file)
                        .parent()
                        .and_then(|path| path.file_name())
                        .and_then(|name| name.to_str())
                        != Some(resolution_parent)
                })
            {
                return Err("NCMC resolution events are outside one committed batch".into());
            }
            let stream_id = u16::try_from(move_receipt.stream_id)
                .map_err(|_| "NCMC stream identity exceeds journal key width".to_string())?;
            let move_key = (stream_id, move_receipt.step, move_receipt.move_id);
            if !admitted_move_keys.insert(move_key) {
                return Err("duplicate NCMC move identity in loaded journal".into());
            }
            let mut populations = Vec::new();
            for entry in &move_receipt.ordered_work_trace.entries {
                let lineage = match entry {
                    NcmcWorkTraceEntry::Perturb {
                        boundary_lineage, ..
                    }
                    | NcmcWorkTraceEntry::Propagate {
                        boundary_lineage, ..
                    }
                    | NcmcWorkTraceEntry::MomentumReverse {
                        boundary_lineage, ..
                    } => boundary_lineage,
                };
                populations.push(&lineage.before);
                populations.push(&lineage.after);
            }
            let first_slots: Vec<u32> = populations
                .first()
                .ok_or_else(|| "NCMC trace has no force boundaries".to_string())?
                .slot_events
                .iter()
                .map(|event| event.slot_id)
                .collect();
            let mut expected_population_count = 0usize;
            let mut terminal_force_sequence = None;
            for (ordinal, population) in populations.iter().enumerate() {
                population.validate()?;
                let slots: Vec<u32> = population
                    .slot_events
                    .iter()
                    .map(|event| event.slot_id)
                    .collect();
                if population.boundary.move_id != move_receipt.move_id
                    || population.boundary.protocol_direction != move_receipt.protocol_direction
                    || population.boundary.boundary_ordinal as usize != ordinal
                    || slots != first_slots
                {
                    return Err(
                        "NCMC trace boundary order or admitted slot population changed".into(),
                    );
                }
                for expected in &population.slot_events {
                    expected_population_count += 1;
                    terminal_force_sequence = Some(expected.event_sequence);
                    let key = LambdaJournalKey {
                        stream_id,
                        step_idx: move_receipt.step,
                        slot_id: expected.slot_id,
                    };
                    let slot_observation = output
                        .observations
                        .get(&key)
                        .ok_or_else(|| "NCMC trace names an absent force slot key".to_string())?;
                    let matches: Vec<_> = slot_observation
                        .force_events
                        .iter()
                        .filter(|force| {
                            force.event_sequence == expected.event_sequence
                                && force.ncmc_boundary.as_ref() == Some(&population.boundary)
                        })
                        .collect();
                    if matches.len() != 1 {
                        return Err("NCMC trace force row is missing or duplicated".into());
                    }
                    let lineage_matches: Vec<_> = slot_observation
                        .event_lineage
                        .iter()
                        .filter(|lineage| {
                            lineage.event_kind == LambdaJournalEventKind::Force
                                && lineage.event_sequence == expected.event_sequence
                                && lineage.raw_event_sha256 == expected.raw_event_sha256
                                && lineage.emission_receipt_sha256
                                    == expected.emission_receipt_sha256
                        })
                        .collect();
                    if lineage_matches.len() != 1 {
                        return Err("NCMC trace force byte lineage is not bijective".into());
                    }
                }
            }
            let actual_population_count = output
                .observations
                .values()
                .filter(|candidate| {
                    candidate.key.stream_id == stream_id
                        && candidate.key.step_idx == move_receipt.step
                })
                .flat_map(|candidate| &candidate.force_events)
                .filter(|force| {
                    force
                        .ncmc_boundary
                        .as_ref()
                        .is_some_and(|boundary| boundary.move_id == move_receipt.move_id)
                })
                .count();
            if actual_population_count != expected_population_count
                || terminal_force_sequence
                    .is_none_or(|sequence| sequence >= move_receipt.event_sequence)
            {
                return Err(
                    "NCMC force population has missing/extra rows or decision precedes force closure"
                        .into(),
                );
            }
            let move_sha256 = move_receipt.digest_sha256()?;
            match move_receipt.accepted {
                true => {
                    let commit = observation.accepted_state_commit.as_ref().ok_or_else(|| {
                        "accepted NCMC move lacks exact-key state commit".to_string()
                    })?;
                    if observation.rollback_no_change.is_some()
                        || commit.ncmc_move_receipt_sha256 != move_sha256
                    {
                        return Err("accepted NCMC move has conflicting resolution".into());
                    }
                }
                false => {
                    let rollback = observation.rollback_no_change.as_ref().ok_or_else(|| {
                        "rejected NCMC move lacks exact-key rollback authority".to_string()
                    })?;
                    if observation.accepted_state_commit.is_some()
                        || rollback.ncmc_move_receipt_sha256 != move_sha256
                        || rollback.restored != move_receipt.initial
                    {
                        return Err("rejected NCMC move has conflicting resolution".into());
                    }
                }
            }
        } else if observation.accepted_state_commit.is_some()
            || observation.rollback_no_change.is_some()
        {
            return Err("lambda state resolution lacks an NCMC move at the exact key".into());
        }
        if let Some(endpoint) = &observation.endpoint_event {
            if let Some(force) = observation.supporting_force_event()? {
                if endpoint.lambda_endpoint as f64 != force.lambda
                    || endpoint.slot_id != force.slot_id
                    || endpoint.species_id != force.species_id
                    || endpoint.chemical_state_id != force.chemical_state_id
                {
                    return Err("lambda endpoint and force event disagree at exact key".into());
                }
                let move_receipt = observation
                    .ncmc_move_receipt
                    .as_ref()
                    .ok_or_else(|| "endpoint supporting force lacks NCMC receipt".to_string())?;
                if endpoint.supporting_force_boundary.move_id != move_receipt.move_id
                    || endpoint.supporting_force_boundary.protocol_direction
                        != move_receipt.protocol_direction
                {
                    return Err("endpoint supporting force does not belong to its NCMC move".into());
                }
            }
        }
    }
    output.committed_resolution_batch_count = admitted_move_keys.len() as u64;
    for observation in output.observations.values() {
        for force in &observation.force_events {
            if let Some(boundary) = &force.ncmc_boundary {
                let key = (
                    observation.key.stream_id,
                    observation.key.step_idx,
                    boundary.move_id,
                );
                if !admitted_move_keys.contains(&key) {
                    return Err("NCMC force boundary has no committed move receipt".into());
                }
            }
        }
    }
    output
        .no_probe_absence_receipts
        .sort_by_key(|(receipt, _)| {
            (
                receipt.stream_id,
                receipt.start_step,
                receipt.end_step_exclusive,
            )
        });
    for pair in output.no_probe_absence_receipts.windows(2) {
        if pair[0].0.stream_id == pair[1].0.stream_id
            && pair[0].0.end_step_exclusive > pair[1].0.start_step
        {
            return Err("overlapping no-probe absence receipts".into());
        }
    }
    if output.produced != output.persisted
        || output.produced != output.decoded
        || output.produced != output.indexed
    {
        return Err(format!(
            "lambda journal conservation failed: produced={} persisted={} decoded={} indexed={}",
            output.produced, output.persisted, output.decoded, output.indexed
        ));
    }
    let population = serde_json::json!({
        "schema_kind": "prismos_lambda_journal_population_receipt",
        "schema_version": 2,
        "produced": output.produced,
        "persisted": output.persisted,
        "decoded": output.decoded,
        "indexed": output.indexed,
        "committed_resolution_batch_count": output.committed_resolution_batch_count,
        "partial_resolution_chain_count": output.partial_resolution_chain_count,
        "stream_terminal_manifest_sha256": output.stream_terminal_manifest_sha256,
        "stream_phase_contexts": output.stream_phase_contexts,
    });
    let population_bytes = serde_json::to_vec(&population).map_err(|error| error.to_string())?;
    output.population_receipt_sha256 = format!("{:x}", Sha256::digest(population_bytes));
    Ok(output)
}

impl LambdaEventJournalLoad {
    /// Exact verified stream phase/condition covering a Ghost step.
    pub fn phase_context_at(
        &self,
        stream_id: u16,
        step_idx: u64,
    ) -> Option<&LambdaJournalSegmentContext> {
        self.stream_phase_contexts
            .get(&stream_id)?
            .iter()
            .find(|context| {
                step_idx >= context.protocol_phase.start_step
                    && step_idx < context.protocol_phase.end_step_exclusive
            })
    }

    /// Locate the explicit absence authority covering a no-probe stream/step.
    pub fn no_probe_absence_at(
        &self,
        stream_id: u16,
        step_idx: u64,
    ) -> Option<&NoProbeAbsenceReceipt> {
        self.no_probe_absence_receipts
            .iter()
            .find(|(receipt, _)| {
                receipt.stream_id == u32::from(stream_id)
                    && step_idx >= receipt.start_step
                    && step_idx < receipt.end_step_exclusive
            })
            .map(|(receipt, _)| receipt)
    }
}

/// Fail-closed, loader-verified projection consumed by native Path-B.  It is
/// deliberately constructed from a verified journal population rather than
/// from naked force/event JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LambdaPathBObservationAdapter {
    pub schema_kind: String,
    pub schema_version: u32,
    pub lane: crate::chemical_state_fiber::ControlLane,
    pub fragment_panel_sha256: String,
    pub fragment_observations: Vec<crate::chemical_state_fiber::FragmentStateObservation>,
    pub lambda_population_sha256: String,
    /// Hash of the population-conserving post-persistence Ghost/endpoint join
    /// materialization from which this projection was derived.
    pub lambda_ghost_join_materialization_sha256: String,
    /// Exact immutable joins admitted to physical endpoint-state fibers.
    pub evaluated_lambda_ghost_joins: Vec<crate::chemical_state_fiber::LambdaGhostEndpointJoin>,
    /// Every retained row that failed one or more scientific admission gates.
    /// These rows remain provenance evidence but never enter pocket claims.
    pub not_evaluable_lambda_ghost_joins: Vec<crate::chemical_state_fiber::LambdaGhostEndpointJoin>,
    pub no_probe_absence_receipt_sha256: Option<String>,
    pub chemical_potential_calibration_population_sha256: Option<String>,
    pub adapter_sha256: String,
}

impl LambdaPathBObservationAdapter {
    fn recompute_sha256(&self) -> Result<String, String> {
        let mut projection = self.clone();
        projection.adapter_sha256.clear();
        let encoded = serde_json::to_vec(&projection).map_err(|error| error.to_string())?;
        let mut digest = Sha256::new();
        digest.update(b"PRISM_LAMBDA_PATHB_OBSERVATION_ADAPTER_V2\0");
        digest.update(encoded);
        Ok(format!("{:x}", digest.finalize()))
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_kind != "prismos_lambda_pathb_observation_adapter"
            || self.schema_version != 2
            || self.adapter_sha256 != self.recompute_sha256()?
        {
            return Err("invalid lambda Path-B observation adapter".into());
        }
        validate_sha256("fragment panel", &self.fragment_panel_sha256)?;
        validate_sha256("lambda journal population", &self.lambda_population_sha256)?;
        validate_sha256(
            "lambda/Ghost join materialization",
            &self.lambda_ghost_join_materialization_sha256,
        )?;
        let mut evaluated_ghost_hashes = BTreeSet::new();
        let mut retained_ghost_hashes = BTreeSet::new();
        let mut join_hashes = BTreeSet::new();
        for join in &self.evaluated_lambda_ghost_joins {
            join.verify().map_err(|error| error.to_string())?;
            if join.status != crate::chemical_state_fiber::LambdaGhostEndpointJoinStatus::Evaluated
                || join.not_evaluable_reason.is_some()
            {
                return Err("evaluated lambda/Ghost partition contains a rejected row".into());
            }
            if !join_hashes.insert(join.join_sha256.clone()) {
                return Err("lambda Path-B adapter duplicates an immutable join row".into());
            }
            evaluated_ghost_hashes.insert(
                join.ghost_lineage_sha256
                    .clone()
                    .ok_or_else(|| "evaluated lambda/Ghost join lacks Ghost lineage".to_string())?,
            );
        }
        for join in &self.not_evaluable_lambda_ghost_joins {
            join.verify().map_err(|error| error.to_string())?;
            if join.status
                != crate::chemical_state_fiber::LambdaGhostEndpointJoinStatus::NotEvaluable
                || join.not_evaluable_reason.is_none()
            {
                return Err("retained lambda/Ghost partition contains an admitted row".into());
            }
            if !join_hashes.insert(join.join_sha256.clone()) {
                return Err("lambda Path-B adapter duplicates an immutable join row".into());
            }
            if let Some(lineage) = &join.ghost_lineage_sha256 {
                retained_ghost_hashes.insert(lineage.clone());
            }
        }
        let mut ghost_hashes = BTreeSet::new();
        for observation in &self.fragment_observations {
            validate_sha256(
                "fragment observation Ghost",
                &observation.ghost_lineage_sha256,
            )?;
            if !ghost_hashes.insert(observation.ghost_lineage_sha256.clone()) {
                return Err("lambda Path-B adapter duplicates a Ghost occurrence".into());
            }
            observation
                .fragment
                .validate_for_lane(self.lane)
                .map_err(|error| error.to_string())?;
        }
        match self.lane {
            crate::chemical_state_fiber::ControlLane::NoProbe => {
                validate_sha256(
                    "no-probe absence population",
                    self.no_probe_absence_receipt_sha256
                        .as_deref()
                        .ok_or_else(|| {
                            "no-probe Path-B adapter lacks absence authority".to_string()
                        })?,
                )?;
                if self
                    .chemical_potential_calibration_population_sha256
                    .is_some()
                {
                    return Err("no-probe Path-B adapter carries calibration evidence".into());
                }
                if !self.evaluated_lambda_ghost_joins.is_empty()
                    || ghost_hashes != retained_ghost_hashes
                {
                    return Err(
                        "no-probe projection must use exactly the retained absence-backed Ghost population"
                            .into(),
                    );
                }
            }
            _ => {
                validate_sha256(
                    "chemical-potential calibration population",
                    self.chemical_potential_calibration_population_sha256
                        .as_deref()
                        .ok_or_else(|| {
                            "probe Path-B adapter lacks calibration authority".to_string()
                        })?,
                )?;
                if self.no_probe_absence_receipt_sha256.is_some() {
                    return Err("probe Path-B adapter carries no-probe absence authority".into());
                }
                if ghost_hashes != evaluated_ghost_hashes {
                    return Err(
                        "probe projection differs from the evaluated immutable join population"
                            .into(),
                    );
                }
            }
        }
        Ok(())
    }
}

fn fragment_identity_for_endpoint(
    system: &LambdaFragmentSystem,
    endpoint: &NcmcEndpointIdentity,
) -> Result<crate::chemical_state_fiber::FragmentChemicalIdentity, String> {
    let species = system
        .species
        .iter()
        .find(|species| species.species_id == endpoint.species_id)
        .ok_or_else(|| format!("fragment species {} absent", endpoint.species_id))?;
    let state = species
        .chemical_states
        .iter()
        .find(|state| state.state_id == endpoint.chemical_state_id)
        .ok_or_else(|| {
            format!(
                "fragment chemical state {}:{} absent",
                endpoint.species_id, endpoint.chemical_state_id
            )
        })?;
    let formal_charge = i16::try_from(state.formal_charge_e)
        .map_err(|_| "fragment formal charge exceeds fiber schema".to_string())?;
    let identity = crate::chemical_state_fiber::FragmentChemicalIdentity {
        species_id: species.name.clone(),
        native_species_id: species.species_id,
        native_chemical_state_id: state.state_id,
        canonical_isomeric_smiles: state.state_canonical_isomeric_smiles.clone(),
        parameter_digest_sha256: state.parameter_digest_sha256.clone(),
        tautomer_id: state.tautomer_id.clone(),
        protonation_state_id: state.protonation_id.clone(),
        formal_charge,
    };
    identity.validate().map_err(|error| error.to_string())?;
    Ok(identity)
}

/// Project the immutable post-persistence Ghost/endpoint join population into
/// native Path-B fragment state. Probe lanes consume only `Evaluated` rows;
/// every `NotEvaluable` row remains carried by the adapter as provenance and
/// never enters pocket claims. No-probe rows are absence-backed controls and
/// therefore consume retained Ghost nodes rather than inventing endpoints.
pub fn adapt_verified_lambda_journals_for_pathb(
    loaded: &LambdaEventJournalLoad,
    joins: &crate::chemical_state_fiber::LambdaGhostEndpointJoinMaterialization,
    lane: crate::chemical_state_fiber::ControlLane,
    fragment_panel_sha256: &str,
    system: Option<&LambdaFragmentSystem>,
) -> Result<LambdaPathBObservationAdapter, String> {
    validate_sha256("fragment panel", fragment_panel_sha256)?;
    validate_sha256(
        "lambda journal population",
        &loaded.population_receipt_sha256,
    )?;
    joins.verify().map_err(|error| error.to_string())?;
    if joins.lambda_population_receipt_sha256 != loaded.population_receipt_sha256 {
        return Err("lambda/Ghost join names a different verified journal population".into());
    }
    let evaluated_lambda_ghost_joins = joins
        .rows
        .iter()
        .filter(|row| {
            row.status == crate::chemical_state_fiber::LambdaGhostEndpointJoinStatus::Evaluated
        })
        .cloned()
        .collect::<Vec<_>>();
    let not_evaluable_lambda_ghost_joins = joins
        .rows
        .iter()
        .filter(|row| {
            row.status == crate::chemical_state_fiber::LambdaGhostEndpointJoinStatus::NotEvaluable
        })
        .cloned()
        .collect::<Vec<_>>();
    if evaluated_lambda_ghost_joins.len() as u64 != joins.evaluated_join_count
        || not_evaluable_lambda_ghost_joins.len() as u64 != joins.not_evaluable_count
    {
        return Err("lambda/Ghost join status partitions do not close".into());
    }
    let expected_lambda_lane = match lane {
        crate::chemical_state_fiber::ControlLane::NoProbe => LambdaExperimentLaneKind::NoProbe,
        crate::chemical_state_fiber::ControlLane::HypothesisFragment => {
            LambdaExperimentLaneKind::HypothesisFragments
        }
        crate::chemical_state_fiber::ControlLane::OrthogonalChemicalControl => {
            LambdaExperimentLaneKind::OrthogonalChemicalControls
        }
    };
    let ghost_rows = match lane {
        crate::chemical_state_fiber::ControlLane::NoProbe => {
            if !evaluated_lambda_ghost_joins.is_empty() {
                return Err("no-probe lane contains physical endpoint joins".into());
            }
            &not_evaluable_lambda_ghost_joins
        }
        _ => &evaluated_lambda_ghost_joins,
    };
    let ghosts = ghost_rows
        .iter()
        .filter_map(|row| row.ghost_node.as_ref().map(|node| &node.ghost))
        .collect::<Vec<_>>();
    let mut fragment_observations = Vec::with_capacity(ghosts.len());
    let mut calibration_digests = BTreeSet::new();
    let mut absence_digests = BTreeSet::new();
    let mut seen_ghosts = BTreeSet::new();
    if ghosts.is_empty() {
        if loaded.stream_contexts.is_empty() {
            return Err("zero-event lambda Path-B adapter has no declared journal streams".into());
        }
        for (&stream_id, terminal_context) in &loaded.stream_contexts {
            if terminal_context.lane.lane_kind != expected_lambda_lane {
                return Err("zero-event lambda Path-B stream lane differs from request".into());
            }
            let phases = loaded
                .stream_phase_contexts
                .get(&stream_id)
                .ok_or_else(|| "zero-event lambda stream lacks phase declarations".to_string())?;
            match lane {
                crate::chemical_state_fiber::ControlLane::NoProbe => {
                    if system.is_some() || terminal_context.ncmc_execution_mode.is_some() {
                        return Err("zero-event no-probe authority carries fragment state".into());
                    }
                    for phase in phases {
                        let (absence, lineage) = loaded
                            .no_probe_absence_receipts
                            .iter()
                            .find(|(receipt, _)| {
                                receipt.stream_id == u32::from(stream_id)
                                    && receipt.start_step <= phase.protocol_phase.start_step
                                    && receipt.end_step_exclusive
                                        >= phase.protocol_phase.end_step_exclusive
                                    && receipt.phase_protocol_sha256
                                        == phase.phase_protocol_sha256
                            })
                            .ok_or_else(|| {
                                format!(
                                    "zero-event no-probe stream {stream_id} lacks full phase absence coverage"
                                )
                            })?;
                        absence_digests.insert(format!(
                            "{:x}",
                            Sha256::digest(
                                serde_json::to_vec(absence).map_err(|error| error.to_string())?
                            )
                        ));
                        absence_digests.insert(lineage.emission_receipt_sha256.clone());
                    }
                }
                _ => {
                    let _ = system.ok_or_else(|| {
                        "zero-event probe authority lacks fragment topology".to_string()
                    })?;
                    if terminal_context.ncmc_execution_mode
                        != Some(NcmcExecutionMode::CalibratedTargetLane)
                    {
                        return Err(
                            "zero-event probe authority is not calibrated target NCMC".into()
                        );
                    }
                    for observation in loaded
                        .observations
                        .values()
                        .filter(|observation| observation.key.stream_id == stream_id)
                    {
                        if let Some(calibration) = &observation.chemical_potential_calibration {
                            calibration_digests.insert(calibration.digest_sha256()?);
                        }
                    }
                }
            }
        }
        if lane != crate::chemical_state_fiber::ControlLane::NoProbe
            && calibration_digests.is_empty()
        {
            return Err(
                "zero-event probe authority has no loader-verified calibration population".into(),
            );
        }
    }
    for ghost in ghosts {
        ghost.validate().map_err(|error| error.to_string())?;
        let ghost_lineage_sha256 = ghost.lineage_sha256().map_err(|error| error.to_string())?;
        if !seen_ghosts.insert(ghost_lineage_sha256.clone()) {
            return Err("duplicate Ghost occurrence supplied to lambda Path-B adapter".into());
        }
        match lane {
            crate::chemical_state_fiber::ControlLane::NoProbe => {
                if system.is_some() {
                    return Err("no-probe Path-B adapter received a fragment system".into());
                }
                let join = not_evaluable_lambda_ghost_joins
                    .iter()
                    .find(|row| {
                        row.ghost_lineage_sha256.as_deref() == Some(ghost_lineage_sha256.as_str())
                    })
                    .ok_or_else(|| {
                        "no-probe Ghost lacks immutable retained join lineage".to_string()
                    })?;
                if join.not_evaluable_reason
                    != Some(
                        crate::chemical_state_fiber::LambdaGhostEndpointNotEvaluableReason::UnmatchedGhostNode,
                    )
                {
                    return Err(
                        "no-probe projection received a non-absence join rejection class".into(),
                    );
                }
                let joined_node = join
                    .ghost_node
                    .as_ref()
                    .ok_or_else(|| "no-probe retained join lacks Ghost node".to_string())?;
                let (absence, lineage) = loaded
                    .no_probe_absence_receipts
                    .iter()
                    .find(|(receipt, _)| {
                        receipt.stream_id == u32::from(ghost.key.stream_id)
                            && ghost.key.step_idx >= receipt.start_step
                            && ghost.key.step_idx < receipt.end_step_exclusive
                    })
                    .ok_or_else(|| {
                        format!(
                            "Ghost {}:{} lacks typed no-probe absence coverage",
                            ghost.key.stream_id, ghost.key.step_idx
                        )
                    })?;
                let context = loaded
                    .phase_context_at(ghost.key.stream_id, ghost.key.step_idx)
                    .ok_or_else(|| "no-probe Ghost lacks journal phase context".to_string())?;
                if context.lane.lane_kind != expected_lambda_lane
                    || absence.phase_protocol_sha256 != context.phase_protocol_sha256
                    || joined_node.control_lane != crate::chemical_state_fiber::ControlLane::NoProbe
                    || joined_node.protocol_phase != context.protocol_phase
                    || joined_node.thermodynamic_condition != context.thermodynamic_condition
                {
                    return Err(
                        "no-probe Ghost, immutable join, absence, phase or condition differ".into(),
                    );
                }
                absence_digests.insert(format!(
                    "{:x}",
                    Sha256::digest(serde_json::to_vec(absence).map_err(|error| error.to_string())?)
                ));
                absence_digests.insert(lineage.emission_receipt_sha256.clone());
                let mut hamiltonian_digest = Sha256::new();
                hamiltonian_digest.update(b"PRISM_NO_PROBE_HAMILTONIAN_AUTHORITY_V1\0");
                hamiltonian_digest.update(ghost.engine_sha256.as_bytes());
                hamiltonian_digest.update(ghost.topology_sha256.as_bytes());
                hamiltonian_digest.update(context.phase_protocol_sha256.as_bytes());
                let fragment = crate::chemical_state_fiber::FragmentState {
                    identity: None,
                    endpoint_a_identity: None,
                    endpoint_b_identity: None,
                    occupancy: crate::chemical_state_fiber::FragmentOccupancy::Absent,
                    lambda: 0.0,
                    du_dlambda_kj_mol: 0.0,
                    dudl_unit_conversion: None,
                    chemical_potential_kj_mol: 0.0,
                    hamiltonian_sha256: format!("{:x}", hamiltonian_digest.finalize()),
                    source_force_event: None,
                    source_endpoint_observation: None,
                    chemical_potential_calibration: None,
                };
                fragment
                    .validate_for_lane(lane)
                    .map_err(|error| error.to_string())?;
                fragment_observations.push(crate::chemical_state_fiber::FragmentStateObservation {
                    ghost_key: ghost.key,
                    ghost_lineage_sha256,
                    fragment,
                });
            }
            _ => {
                let system = system.ok_or_else(|| {
                    "probe Path-B adapter lacks physical fragment topology".to_string()
                })?;
                let join_candidates = evaluated_lambda_ghost_joins
                    .iter()
                    .filter(|row| {
                        row.ghost_lineage_sha256.as_deref() == Some(ghost_lineage_sha256.as_str())
                    })
                    .collect::<Vec<_>>();
                if join_candidates.len() != 1 {
                    return Err(format!(
                        "Ghost {}:{} has {} evaluated immutable endpoint joins; expected one",
                        ghost.key.stream_id,
                        ghost.key.step_idx,
                        join_candidates.len()
                    ));
                }
                let join = join_candidates[0];
                let key = join
                    .lambda_key
                    .ok_or_else(|| "evaluated lambda/Ghost join lacks journal key".to_string())?;
                let observation = loaded.observations.get(&key).ok_or_else(|| {
                    "evaluated lambda/Ghost join lacks source observation".to_string()
                })?;
                if observation.lane.lane_kind != expected_lambda_lane {
                    return Err("probe Ghost and lambda journal lane differ".into());
                }
                let context = loaded
                    .phase_context_at(ghost.key.stream_id, ghost.key.step_idx)
                    .ok_or_else(|| "probe Ghost lacks journal phase context".to_string())?;
                if context.ncmc_execution_mode != Some(NcmcExecutionMode::CalibratedTargetLane) {
                    return Err(
                        "calibration-work lambda states cannot support Path-B pocket claims".into(),
                    );
                }
                let force = join
                    .supporting_force
                    .as_ref()
                    .ok_or_else(|| "evaluated Ghost join lacks native force event".to_string())?;
                let endpoint = join
                    .endpoint
                    .as_ref()
                    .ok_or_else(|| "evaluated Ghost join lacks endpoint event".to_string())?;
                if observation.endpoint_event.as_ref() != Some(endpoint)
                    || !observation.force_events.iter().any(|event| event == force)
                {
                    return Err(
                        "evaluated immutable join differs from its verified journal observation"
                            .into(),
                    );
                }
                let calibration = observation
                    .chemical_potential_calibration
                    .as_ref()
                    .ok_or_else(|| "probe Ghost join lacks active calibrated mu".to_string())?;
                if force.lambda.to_bits() != 0.0f64.to_bits()
                    && force.lambda.to_bits() != 1.0f64.to_bits()
                {
                    return Err("partial-lambda state is sampling-only, not a pocket claim".into());
                }
                let endpoint_a_identity =
                    fragment_identity_for_endpoint(system, &force.endpoint_a)?;
                let endpoint_b_identity =
                    fragment_identity_for_endpoint(system, &force.endpoint_b)?;
                let active_endpoint = if force.lambda.to_bits() == 0.0f64.to_bits() {
                    &force.endpoint_a
                } else {
                    &force.endpoint_b
                };
                let identity = (active_endpoint.occupancy == EndpointOccupancy::Occupied)
                    .then(|| fragment_identity_for_endpoint(system, active_endpoint))
                    .transpose()?;
                let slot = system
                    .slots
                    .iter()
                    .find(|slot| slot.slot_id == force.slot_id)
                    .ok_or_else(|| "probe force event names absent fragment slot".to_string())?;
                let occupancy = if identity.is_some() {
                    crate::chemical_state_fiber::FragmentOccupancy::Present {
                        molecule_instance_id: format!("molecule-{}", slot.molecule_id),
                        fragment_slot_id: slot.slot_id,
                    }
                } else {
                    crate::chemical_state_fiber::FragmentOccupancy::Absent
                };
                let conversion = force_event_dudl_for_fiber(force)?;
                let selected_identity = identity.as_ref();
                if selected_identity
                    .map(|value| (value.native_species_id, value.native_chemical_state_id))
                    != Some((calibration.species_id, calibration.chemical_state_id))
                {
                    return Err(
                        "probe endpoint calibration does not name active physical identity".into(),
                    );
                }
                calibration_digests.insert(calibration.digest_sha256()?);
                let fragment = crate::chemical_state_fiber::FragmentState {
                    identity,
                    endpoint_a_identity: Some(endpoint_a_identity),
                    endpoint_b_identity: Some(endpoint_b_identity),
                    occupancy,
                    lambda: force.lambda,
                    du_dlambda_kj_mol: conversion.du_dlambda_kj_mol,
                    dudl_unit_conversion: Some(conversion),
                    chemical_potential_kj_mol: calibration.total_chemical_potential_kcal_mol
                        * 4.184,
                    hamiltonian_sha256: force.hamiltonian_sha256.clone(),
                    source_force_event: Some(force.clone()),
                    source_endpoint_observation: Some(endpoint.clone()),
                    chemical_potential_calibration: Some(calibration.clone()),
                };
                fragment
                    .validate_for_lane(lane)
                    .map_err(|error| error.to_string())?;
                fragment_observations.push(crate::chemical_state_fiber::FragmentStateObservation {
                    ghost_key: ghost.key,
                    ghost_lineage_sha256,
                    fragment,
                });
            }
        }
    }
    let population_digest = |domain: &[u8], values: &BTreeSet<String>| -> Option<String> {
        if values.is_empty() {
            return None;
        }
        let mut digest = Sha256::new();
        digest.update(domain);
        digest.update(loaded.population_receipt_sha256.as_bytes());
        digest.update(fragment_panel_sha256.as_bytes());
        for value in values {
            digest.update(value.as_bytes());
        }
        Some(format!("{:x}", digest.finalize()))
    };
    let mut adapter = LambdaPathBObservationAdapter {
        schema_kind: "prismos_lambda_pathb_observation_adapter".into(),
        schema_version: 2,
        lane,
        fragment_panel_sha256: fragment_panel_sha256.into(),
        fragment_observations,
        lambda_population_sha256: loaded.population_receipt_sha256.clone(),
        lambda_ghost_join_materialization_sha256: joins.materialization_sha256.clone(),
        evaluated_lambda_ghost_joins,
        not_evaluable_lambda_ghost_joins,
        no_probe_absence_receipt_sha256: population_digest(
            b"PRISM_NO_PROBE_ABSENCE_POPULATION_V1\0",
            &absence_digests,
        ),
        chemical_potential_calibration_population_sha256: population_digest(
            b"PRISM_CHEMICAL_POTENTIAL_CALIBRATION_POPULATION_V1\0",
            &calibration_digests,
        ),
        adapter_sha256: String::new(),
    };
    adapter.adapter_sha256 = adapter.recompute_sha256()?;
    adapter.validate()?;
    Ok(adapter)
}

/// One calibration-ready NCMC row constructed only after exact-byte QNANL,
/// segment-chain, event-chain and state-resolution verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LambdaCalibrationJoinRow {
    pub schema_kind: String,
    pub schema_version: u32,
    pub key: LambdaJournalKey,
    pub lane: LambdaExperimentLaneContract,
    pub protocol_context: LambdaJournalSegmentContext,
    pub join_mode: NcmcExecutionMode,
    pub chemical_potential_calibration: Option<ChemicalPotentialCalibration>,
    pub chemical_potential_calibration_sha256: Option<String>,
    pub ncmc_move_receipt: NcmcMoveReceipt,
    pub ncmc_move_receipt_sha256: String,
    pub log_target_ratio: f64,
    pub log_acceptance: f64,
    pub uniform_random: f64,
    pub forward_protocol_probability: f64,
    pub reverse_protocol_probability: f64,
    pub forward_selection_probability: f64,
    pub reverse_selection_probability: f64,
    pub forward_placement_density_a3_inv: f64,
    pub reverse_placement_density_a3_inv: f64,
    pub forward_orientation_density: f64,
    pub reverse_orientation_density: f64,
    pub proposal_measure_contract: NcmcProposalMeasureContract,
    pub accepted_state_commit: Option<LambdaFragmentStateCommit>,
    pub rollback_no_change: Option<LambdaFragmentStateRollback>,
    pub complete_exact_key_event_lineage: Vec<LambdaJournalLoadedLineage>,
    pub source_journal_population_receipt_sha256: String,
    pub row_sha256: String,
}

impl LambdaCalibrationJoinRow {
    fn recompute_sha256(&self) -> Result<String, String> {
        let mut projection = self.clone();
        projection.row_sha256.clear();
        let encoded = serde_json::to_vec(&projection).map_err(|error| error.to_string())?;
        let mut digest = Sha256::new();
        digest.update(b"PRISM_LAMBDA_CALIBRATION_JOIN_ROW_V2\0");
        digest.update(encoded);
        Ok(format!("{:x}", digest.finalize()))
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_kind != "prismos_lambda_calibration_join_row"
            || self.schema_version != 2
            || self.row_sha256 != self.recompute_sha256()?
            || self.key.stream_id != self.protocol_context.stream_id
            || self.key.step_idx < self.protocol_context.protocol_phase.start_step
            || self.key.step_idx >= self.protocol_context.protocol_phase.end_step_exclusive
            || self.ncmc_move_receipt.stream_id != u32::from(self.key.stream_id)
            || self.ncmc_move_receipt.step != self.key.step_idx
            || self.ncmc_move_receipt.slot_id != self.key.slot_id
            || self.ncmc_move_receipt.phase_protocol_sha256
                != self.protocol_context.phase_protocol_sha256
            || self.lane.contract_sha256 != self.protocol_context.lane.contract_sha256
            || self.protocol_context.ncmc_execution_mode.as_ref() != Some(&self.join_mode)
            || self.ncmc_move_receipt.execution_mode != self.join_mode
        {
            return Err("invalid lambda calibration join identity".into());
        }
        self.join_mode.validate()?;
        self.ncmc_move_receipt.validate()?;
        self.proposal_measure_contract.validate()?;
        validate_sha256(
            "source lambda journal population",
            &self.source_journal_population_receipt_sha256,
        )?;
        match (
            &self.join_mode,
            &self.chemical_potential_calibration,
            &self.chemical_potential_calibration_sha256,
        ) {
            (NcmcExecutionMode::CalibrationWorkCollection { .. }, None, None) => {
                if self.log_target_ratio.to_bits() != 0.0f64.to_bits()
                    || self
                        .ncmc_move_receipt
                        .derived_balance
                        .beta_chemical_potential_term
                        .to_bits()
                        != 0.0f64.to_bits()
                {
                    return Err(
                        "pre-calibration join contains circular chemical-potential target terms"
                            .into(),
                    );
                }
            }
            (
                NcmcExecutionMode::CalibratedTargetLane,
                Some(calibration),
                Some(calibration_sha256),
            ) => {
                calibration.validate()?;
                if calibration_sha256 != &calibration.digest_sha256()? {
                    return Err("calibrated NCMC join calibration digest mismatch".into());
                }
            }
            (NcmcExecutionMode::CalibrationWorkCollection { .. }, _, _) => {
                return Err("pre-calibration join must not contain a calibrated mu".into())
            }
            (NcmcExecutionMode::CalibratedTargetLane, _, _) => {
                return Err("production target NCMC join lacks calibrated mu".into())
            }
        }
        if self.ncmc_move_receipt_sha256 != self.ncmc_move_receipt.digest_sha256()?
            || self.log_target_ratio.to_bits() != self.ncmc_move_receipt.log_target_ratio.to_bits()
            || self.log_acceptance.to_bits() != self.ncmc_move_receipt.log_acceptance.to_bits()
            || self.uniform_random.to_bits() != self.ncmc_move_receipt.uniform_random.to_bits()
            || self.forward_protocol_probability.to_bits()
                != self
                    .ncmc_move_receipt
                    .forward_protocol_probability
                    .to_bits()
            || self.reverse_protocol_probability.to_bits()
                != self
                    .ncmc_move_receipt
                    .reverse_protocol_probability
                    .to_bits()
            || self.forward_selection_probability.to_bits()
                != self
                    .ncmc_move_receipt
                    .proposal_context
                    .forward_selection_probability
                    .to_bits()
            || self.reverse_selection_probability.to_bits()
                != self
                    .ncmc_move_receipt
                    .proposal_context
                    .reverse_selection_probability
                    .to_bits()
            || self.forward_placement_density_a3_inv.to_bits()
                != self
                    .ncmc_move_receipt
                    .proposal_context
                    .forward_placement_density_a3_inv
                    .to_bits()
            || self.reverse_placement_density_a3_inv.to_bits()
                != self
                    .ncmc_move_receipt
                    .proposal_context
                    .reverse_placement_density_a3_inv
                    .to_bits()
            || self.forward_orientation_density.to_bits()
                != self
                    .ncmc_move_receipt
                    .proposal_context
                    .forward_orientation_density
                    .to_bits()
            || self.reverse_orientation_density.to_bits()
                != self
                    .ncmc_move_receipt
                    .proposal_context
                    .reverse_orientation_density
                    .to_bits()
            || self.proposal_measure_contract != self.ncmc_move_receipt.proposal_measure_contract
        {
            return Err("lambda calibration join scalar projection is not lossless".into());
        }
        let ncmc_lineage = self
            .complete_exact_key_event_lineage
            .iter()
            .filter(|lineage| lineage.event_kind == LambdaJournalEventKind::NcmcMove)
            .count();
        let commit_lineage = self
            .complete_exact_key_event_lineage
            .iter()
            .filter(|lineage| lineage.event_kind == LambdaJournalEventKind::StateCommit)
            .count();
        let rollback_lineage = self
            .complete_exact_key_event_lineage
            .iter()
            .filter(|lineage| lineage.event_kind == LambdaJournalEventKind::StateRollback)
            .count();
        if ncmc_lineage != 1
            || (self.ncmc_move_receipt.accepted
                && (self.accepted_state_commit.is_none()
                    || self.rollback_no_change.is_some()
                    || commit_lineage != 1
                    || rollback_lineage != 0))
            || (!self.ncmc_move_receipt.accepted
                && (self.rollback_no_change.is_none()
                    || self.accepted_state_commit.is_some()
                    || rollback_lineage != 1
                    || commit_lineage != 0))
        {
            return Err(
                "lambda calibration join lacks exactly one authenticated resolution".into(),
            );
        }
        for lineage in &self.complete_exact_key_event_lineage {
            for (label, digest) in [
                ("emission receipt", &lineage.emission_receipt_sha256),
                ("raw event", &lineage.raw_event_sha256),
                ("segment manifest", &lineage.segment_manifest_sha256),
                ("segment", &lineage.segment_sha256),
            ] {
                validate_sha256(label, digest)?;
            }
            if lineage.byte_length == 0 || Path::new(&lineage.segment_file).is_absolute() {
                return Err("invalid lambda calibration join byte lineage".into());
            }
        }
        Ok(())
    }
}

/// Immutable collection of loader-verified calibration rows.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LambdaCalibrationJoinArtifact {
    pub schema_kind: String,
    pub schema_version: u32,
    pub source_journal_population_receipt_sha256: String,
    pub source_produced: u64,
    pub source_persisted: u64,
    pub source_decoded: u64,
    pub source_indexed: u64,
    pub stream_terminal_manifest_sha256: BTreeMap<u16, String>,
    pub rows: Vec<LambdaCalibrationJoinRow>,
    pub artifact_sha256: String,
}

impl LambdaCalibrationJoinArtifact {
    fn recompute_sha256(&self) -> Result<String, String> {
        let mut projection = self.clone();
        projection.artifact_sha256.clear();
        let encoded = serde_json::to_vec(&projection).map_err(|error| error.to_string())?;
        let mut digest = Sha256::new();
        digest.update(b"PRISM_LAMBDA_CALIBRATION_JOIN_ARTIFACT_V2\0");
        digest.update(encoded);
        Ok(format!("{:x}", digest.finalize()))
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_kind != "prismos_lambda_calibration_join_artifact"
            || self.schema_version != 2
            || self.source_produced != self.source_persisted
            || self.source_produced != self.source_decoded
            || self.source_produced != self.source_indexed
            || self.artifact_sha256 != self.recompute_sha256()?
        {
            return Err("invalid lambda calibration join artifact".into());
        }
        validate_sha256(
            "source lambda journal population",
            &self.source_journal_population_receipt_sha256,
        )?;
        for digest in self.stream_terminal_manifest_sha256.values() {
            validate_sha256("lambda stream terminal manifest", digest)?;
        }
        let mut keys = BTreeSet::new();
        for row in &self.rows {
            row.validate()?;
            if row.source_journal_population_receipt_sha256
                != self.source_journal_population_receipt_sha256
                || !keys.insert(row.key)
            {
                return Err("duplicate or cross-population lambda calibration join row".into());
            }
        }
        Ok(())
    }
}

/// Export calibration rows from the verified journal loader.  There is no API
/// accepting a naked NCMC receipt, preventing unauthenticated work-journal
/// ingestion from being promoted to calibration evidence.
pub fn export_verified_lambda_calibration_joins(
    run_root: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
) -> Result<LambdaCalibrationJoinArtifact, String> {
    let loaded = load_verified_lambda_event_journals(run_root.as_ref())?;
    let mut rows = Vec::new();
    for observation in loaded.observations.values() {
        let Some(receipt) = observation.ncmc_move_receipt.as_ref() else {
            continue;
        };
        let protocol_context = loaded
            .phase_context_at(observation.key.stream_id, observation.key.step_idx)
            .ok_or_else(|| "NCMC calibration join lacks verified phase context".to_string())?
            .clone();
        let mut row = LambdaCalibrationJoinRow {
            schema_kind: "prismos_lambda_calibration_join_row".into(),
            schema_version: 2,
            key: observation.key,
            lane: observation.lane.clone(),
            protocol_context,
            join_mode: receipt.execution_mode.clone(),
            chemical_potential_calibration: observation.chemical_potential_calibration.clone(),
            chemical_potential_calibration_sha256: observation
                .chemical_potential_calibration
                .as_ref()
                .map(ChemicalPotentialCalibration::digest_sha256)
                .transpose()?,
            ncmc_move_receipt: receipt.clone(),
            ncmc_move_receipt_sha256: receipt.digest_sha256()?,
            log_target_ratio: receipt.log_target_ratio,
            log_acceptance: receipt.log_acceptance,
            uniform_random: receipt.uniform_random,
            forward_protocol_probability: receipt.forward_protocol_probability,
            reverse_protocol_probability: receipt.reverse_protocol_probability,
            forward_selection_probability: receipt.proposal_context.forward_selection_probability,
            reverse_selection_probability: receipt.proposal_context.reverse_selection_probability,
            forward_placement_density_a3_inv: receipt
                .proposal_context
                .forward_placement_density_a3_inv,
            reverse_placement_density_a3_inv: receipt
                .proposal_context
                .reverse_placement_density_a3_inv,
            forward_orientation_density: receipt.proposal_context.forward_orientation_density,
            reverse_orientation_density: receipt.proposal_context.reverse_orientation_density,
            proposal_measure_contract: receipt.proposal_measure_contract.clone(),
            accepted_state_commit: observation.accepted_state_commit.clone(),
            rollback_no_change: observation.rollback_no_change.clone(),
            complete_exact_key_event_lineage: observation.event_lineage.clone(),
            source_journal_population_receipt_sha256: loaded.population_receipt_sha256.clone(),
            row_sha256: String::new(),
        };
        row.row_sha256 = row.recompute_sha256()?;
        row.validate()?;
        rows.push(row);
    }
    rows.sort_by_key(|row| row.key);
    let mut artifact = LambdaCalibrationJoinArtifact {
        schema_kind: "prismos_lambda_calibration_join_artifact".into(),
        schema_version: 2,
        source_journal_population_receipt_sha256: loaded.population_receipt_sha256,
        source_produced: loaded.produced,
        source_persisted: loaded.persisted,
        source_decoded: loaded.decoded,
        source_indexed: loaded.indexed,
        stream_terminal_manifest_sha256: loaded.stream_terminal_manifest_sha256,
        rows,
        artifact_sha256: String::new(),
    };
    artifact.artifact_sha256 = artifact.recompute_sha256()?;
    artifact.validate()?;
    let encoded = serde_json::to_vec_pretty(&artifact).map_err(|error| error.to_string())?;
    atomic_write_synced(output_path.as_ref(), &encoded)?;
    Ok(artifact)
}

/// Decode and revalidate a persisted calibration join artifact.
pub fn load_verified_lambda_calibration_join_artifact(
    path: impl AsRef<Path>,
) -> Result<LambdaCalibrationJoinArtifact, String> {
    let bytes = std::fs::read(path.as_ref()).map_err(|error| error.to_string())?;
    let artifact: LambdaCalibrationJoinArtifact =
        serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
    artifact.validate()?;
    Ok(artifact)
}

/// Validate the complete production topology and runtime capability contract.
/// `pme_runtime_available` and `settle_runtime_available` must describe the
/// actual selected engine, not merely source files present in the repository.
pub fn validate_production_admission(
    topology: &PrismPrepTopology,
    system: &LambdaFragmentSystem,
    runtime: &LambdaRuntimeCapabilities,
) -> Result<(), String> {
    if system.schema_kind != "prismos_lambda_fragment_system" || system.schema_version != 2 {
        return Err("unsupported lambda-fragment topology schema".to_string());
    }
    if system.experiment_id.trim().is_empty() {
        return Err("lambda-fragment experiment_id is empty".to_string());
    }
    for (name, digest) in [
        ("selected engine", runtime.selected_engine_sha256.as_deref()),
        (
            "selected fused PTX",
            runtime.selected_fused_ptx_sha256.as_deref(),
        ),
        ("executed topology", runtime.topology_sha256.as_deref()),
        (
            "same-stream composition receipt",
            runtime.same_stream_composition_receipt_sha256.as_deref(),
        ),
        (
            "runtime self-test receipt",
            runtime.runtime_self_test_receipt_sha256.as_deref(),
        ),
        (
            "typed event dispatch receipt",
            runtime.typed_event_dispatch_receipt_sha256.as_deref(),
        ),
    ] {
        validate_sha256(
            name,
            digest.ok_or_else(|| format!("selected runtime lacks {name} identity"))?,
        )?;
    }
    validate_sha256(
        "periodic_box.source_sha256",
        &system.periodic_box.source_sha256,
    )?;
    if !system.periodic_box.minimum_image_enabled
        || !system.periodic_box.volume_a3().is_finite()
        || system.periodic_box.volume_a3() <= 1.0
    {
        return Err("lambda-fragment production requires a nonsingular periodic box".into());
    }
    validate_hamiltonian(&system.softcore)?;
    if system.softcore.formula != "PRISM_LAMBDA_SOFTCORE_LJ_EWALD_CHARGE_V2" {
        return Err(
            "production PME lambda execution requires the charge-linear Ewald V2 Hamiltonian"
                .into(),
        );
    }
    validate_pme(&system.pme, &system.periodic_box)?;
    validate_water(&system.water)?;
    if !runtime.native_softcore_force || !runtime.native_dudl {
        return Err("selected runtime lacks native soft-core force or dU/dlambda".into());
    }
    if system
        .slots
        .iter()
        .any(|slot| slot.path_kind == AlchemicalPathKind::StateAToStateB)
        && (!runtime.dual_endpoint_state_transform
            || !runtime.constant_topology_union_state_transform
            || !runtime.endpoint_bonded_force_dudl
            || !runtime.endpoint_exception_force_dudl)
    {
        return Err(
            "selected runtime lacks a validated occupied state-A/state-B Hamiltonian".into(),
        );
    }
    if !runtime.ewald_charge_linear_hamiltonian {
        return Err(
            "selected runtime lacks the declared charge-linear Ewald lambda Hamiltonian".into(),
        );
    }
    if !runtime.periodic_minimum_image || !runtime.periodic_box_wrapping {
        return Err(
            "selected runtime lacks complete periodic minimum-image/wrapping execution".into(),
        );
    }
    if !runtime.pme_complete {
        return Err(
            "selected fused engine has no verified PME reciprocal-space execution path".into(),
        );
    }
    if !runtime.pme_openmm_order5
        || runtime.pme_openmm_reference_parity_receipt_sha256.is_none()
        || !runtime.pme_total_hamiltonian_self_test
        || runtime
            .pme_target_system_validation_receipt_sha256
            .is_none()
    {
        return Err(
            "selected PME runtime lacks order-5, independent OpenMM parity, and target-system total U/F/dUdl closure evidence".into(),
        );
    }
    validate_sha256(
        "PME OpenMM reference parity receipt",
        runtime
            .pme_openmm_reference_parity_receipt_sha256
            .as_deref()
            .ok_or_else(|| "missing PME OpenMM reference parity receipt".to_string())?,
    )?;
    validate_sha256(
        "PME target-system validation receipt",
        runtime
            .pme_target_system_validation_receipt_sha256
            .as_deref()
            .ok_or_else(|| "missing PME target-system validation receipt".to_string())?,
    )?;
    if runtime.pme_parameterization_sha256.as_deref()
        != Some(system.pme.pme_parameterization_sha256.as_str())
    {
        return Err("selected PME runtime is not hash-bound to the prepared contract".into());
    }
    if !runtime.rigid_water_positions || !runtime.rigid_water_velocities {
        return Err("selected fused engine has no verified SETTLE position+velocity path".into());
    }
    if !runtime.ncmc_device_sequence
        || !runtime.ncmc_receipt_schema_v6
        || !runtime.ncmc_raw_proposal_context_persisted
        || !runtime.ncmc_proposal_measure_contract_validated
        || !runtime.ncmc_loader_verified_calibration_join
        || runtime.ncmc_calibration_join_receipt_sha256.is_none()
        || !runtime.typed_event_dispatch
    {
        return Err(
            "selected runtime lacks device-owned NCMC v4 raw-context persistence, verified calibration join, or typed event execution"
                .into(),
        );
    }
    validate_sha256(
        "NCMC calibration join receipt",
        runtime
            .ncmc_calibration_join_receipt_sha256
            .as_deref()
            .ok_or_else(|| "missing NCMC calibration join receipt".to_string())?,
    )?;
    let treatments = system
        .chemical_potentials
        .iter()
        .map(|calibration| &calibration.periodic_charge_treatment)
        .chain(system.species.iter().flat_map(|species| {
            species
                .state_transitions
                .iter()
                .map(|transition| &transition.periodic_charge_treatment)
        }));
    let (mut requires_coupled_charge_path, mut requires_finite_size_correction) = (false, false);
    for treatment in treatments {
        requires_coupled_charge_path |= matches!(
            treatment,
            PeriodicChargeTreatment::CoupledChargeNeutralPath { .. }
        );
        requires_finite_size_correction |= matches!(
            treatment,
            PeriodicChargeTreatment::ExplicitFiniteSizeCorrection { .. }
        );
    }
    if requires_coupled_charge_path && !runtime.coupled_charge_neutral_ncmc {
        return Err(
            "selected runtime lacks atomic device-owned coupled charge-neutral NCMC".into(),
        );
    }
    if requires_finite_size_correction && !runtime.finite_size_corrected_charged_ncmc {
        return Err(
            "selected runtime lacks box/PME-bound finite-size-corrected charged NCMC".into(),
        );
    }

    let mut molecule_ids = HashSet::new();
    let mut atom_owners = vec![None; topology.n_atoms];
    let mut fragment_molecules = HashMap::new();
    let mut declared_water_oxygens = BTreeSet::new();
    for molecule in &system.molecules {
        if !molecule_ids.insert(molecule.molecule_id) || molecule.atom_indices.is_empty() {
            return Err(format!(
                "duplicate or empty molecule {}",
                molecule.molecule_id
            ));
        }
        let mut local = HashSet::new();
        for &atom in &molecule.atom_indices {
            if atom >= topology.n_atoms || !local.insert(atom) || atom_owners[atom].is_some() {
                return Err(format!("invalid or multiply-owned molecule atom {atom}"));
            }
            atom_owners[atom] = Some(molecule.molecule_id);
        }
        match molecule.role {
            MoleculeRole::Water => {
                let triplet = molecule.water_triplet.ok_or_else(|| {
                    format!(
                        "water molecule {} lacks O-H-H triplet",
                        molecule.molecule_id
                    )
                })?;
                if molecule.atom_indices.len() != 3
                    || BTreeSet::from(triplet)
                        != molecule
                            .atom_indices
                            .iter()
                            .copied()
                            .collect::<BTreeSet<_>>()
                {
                    return Err(format!(
                        "water molecule {} is not one exact triplet",
                        molecule.molecule_id
                    ));
                }
                if topology.elements[triplet[0]].trim() != "O"
                    || topology.elements[triplet[1]].trim() != "H"
                    || topology.elements[triplet[2]].trim() != "H"
                {
                    return Err(format!(
                        "water molecule {} triplet is not ordered O-H-H",
                        molecule.molecule_id
                    ));
                }
                let position = |atom: usize| {
                    [
                        topology.positions[atom * 3] as f64,
                        topology.positions[atom * 3 + 1] as f64,
                        topology.positions[atom * 3 + 2] as f64,
                    ]
                };
                let o = position(triplet[0]);
                let h1 = position(triplet[1]);
                let h2 = position(triplet[2]);
                let oh1 = minimum_image_displacement(
                    [h1[0] - o[0], h1[1] - o[1], h1[2] - o[2]],
                    &system.periodic_box,
                )?;
                let oh2 = minimum_image_displacement(
                    [h2[0] - o[0], h2[1] - o[1], h2[2] - o[2]],
                    &system.periodic_box,
                )?;
                let n1 = oh1.iter().map(|x| x * x).sum::<f64>().sqrt();
                let n2 = oh2.iter().map(|x| x * x).sum::<f64>().sqrt();
                let cosine = oh1.iter().zip(oh2).map(|(x, y)| x * y).sum::<f64>() / (n1 * n2);
                let angle = cosine.clamp(-1.0, 1.0).acos();
                if (n1 - system.water.oh_distance_a).abs() > 5.0e-3
                    || (n2 - system.water.oh_distance_a).abs() > 5.0e-3
                    || (angle - system.water.hoh_angle_rad).abs() > 5.0e-3
                {
                    return Err(format!(
                        "water molecule {} does not match declared rigid geometry",
                        molecule.molecule_id
                    ));
                }
                declared_water_oxygens.insert(triplet[0]);
            }
            MoleculeRole::LambdaFragment => {
                if molecule.water_triplet.is_some() {
                    return Err("lambda-fragment molecule cannot declare a water triplet".into());
                }
                fragment_molecules.insert(molecule.molecule_id, molecule.atom_indices.clone());
            }
            _ if molecule.water_triplet.is_some() => {
                return Err("only WATER molecules may declare water_triplet".into())
            }
            _ => {}
        }
    }
    if atom_owners.iter().any(Option::is_none) {
        return Err("molecule registry does not own every topology atom exactly once".into());
    }
    if declared_water_oxygens
        != topology
            .water_oxygens
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
    {
        return Err("water_oxygens does not equal the complete molecule-triplet oxygen set".into());
    }

    let mut species = HashMap::new();
    for item in &system.species {
        if item.name.trim().is_empty()
            || item.canonical_isomeric_smiles.trim().is_empty()
            || item.inchikey.trim().is_empty()
            || item.chemical_states.is_empty()
            || species.insert(item.species_id, item).is_some()
        {
            return Err(format!(
                "invalid or duplicate fragment species {}",
                item.species_id
            ));
        }
        validate_sha256("parameter_template_sha256", &item.parameter_template_sha256)?;
        let mut state_ids = HashSet::new();
        for state in &item.chemical_states {
            if !state_ids.insert(state.state_id)
                || state.atom_parameters.is_empty()
                || state.state_canonical_isomeric_smiles.trim().is_empty()
                || state.state_explicit_h_mapped_smiles.trim().is_empty()
                || state.state_inchikey_fixed_h.trim().is_empty()
                || state.tautomer_id.trim().is_empty()
                || state.protonation_id.trim().is_empty()
                || !state.intrinsic_free_energy_kcal_mol.is_finite()
            {
                return Err(format!(
                    "invalid chemical state {}:{}",
                    item.species_id, state.state_id
                ));
            }
            validate_sha256(
                "intramolecular parameter set",
                &state.intramolecular_parameter_set_sha256,
            )?;
            if state.atom_mapping.len() != state.atom_parameters.len()
                || state.reference_conformer_positions_a.len() != state.atom_parameters.len()
            {
                return Err(format!(
                    "state identity/coordinate planes do not match union atom count in {}:{}",
                    item.species_id, state.state_id
                ));
            }
            let mut mapped_ids = HashSet::new();
            let mut mapped_numbers = HashSet::new();
            for mapping in &state.atom_mapping {
                if !mapped_ids.insert(mapping.template_atom_id)
                    || mapping.atomic_number == 0
                    || mapping.element.trim().is_empty()
                    || mapping.atom_map_number == 0
                    || !mapped_numbers.insert(mapping.atom_map_number)
                {
                    return Err(format!(
                        "invalid mapped atom identity in {}:{}",
                        item.species_id, state.state_id
                    ));
                }
            }
            if state
                .reference_conformer_positions_a
                .iter()
                .flatten()
                .any(|coordinate| !coordinate.is_finite())
            {
                return Err(format!(
                    "nonfinite reference conformer in {}:{}",
                    item.species_id, state.state_id
                ));
            }
            let mut exact_pairs = HashSet::new();
            for pair in &state.exact_exception_parameters {
                let key = if pair.template_atom_i < pair.template_atom_j {
                    (pair.template_atom_i, pair.template_atom_j)
                } else {
                    (pair.template_atom_j, pair.template_atom_i)
                };
                if pair.template_atom_i == pair.template_atom_j
                    || !exact_pairs.insert(key)
                    || !pair.charge_product_e2.is_finite()
                    || !pair.sigma_a.is_finite()
                    || pair.sigma_a <= 0.0
                    || !pair.epsilon_kcal_mol.is_finite()
                    || pair.epsilon_kcal_mol < 0.0
                {
                    return Err(format!(
                        "invalid exact exception in {}:{}",
                        item.species_id, state.state_id
                    ));
                }
            }
            let mut atom_ids = HashSet::new();
            let charge_sum = state.atom_parameters.iter().try_fold(0.0, |sum, atom| {
                if !atom_ids.insert(atom.template_atom_id)
                    || !atom.charge_e.is_finite()
                    || !atom.sigma_a.is_finite()
                    || !atom.epsilon_kcal_mol.is_finite()
                    || atom.sigma_a <= 0.0
                    || atom.epsilon_kcal_mol < 0.0
                    || (!atom.endpoint_present
                        && (atom.charge_e.to_bits() != 0.0f64.to_bits()
                            || atom.epsilon_kcal_mol.to_bits() != 0.0f64.to_bits()))
                {
                    Err(format!(
                        "invalid atom state in {}:{}",
                        item.species_id, state.state_id
                    ))
                } else {
                    Ok(sum + atom.charge_e)
                }
            })?;
            if atom_ids != mapped_ids {
                return Err(format!(
                    "mapped atom identities disagree with endpoint parameter plane in {}:{}",
                    item.species_id, state.state_id
                ));
            }
            validate_sha256(
                "endpoint bonded parameter plane",
                &state.bonded_parameter_plane_sha256,
            )?;
            validate_bonded_plane(&state.bonded_parameter_plane, &atom_ids)?;
            if state.bonded_parameter_plane_sha256
                != fragment_bonded_parameter_plane_sha256(&state.bonded_parameter_plane)?
            {
                return Err(format!(
                    "endpoint bonded-plane digest mismatch for {}:{}",
                    item.species_id, state.state_id
                ));
            }
            if state.exact_exception_parameters.iter().any(|pair| {
                !atom_ids.contains(&pair.template_atom_i)
                    || !atom_ids.contains(&pair.template_atom_j)
            }) {
                return Err(format!(
                    "exact exception references absent template atom in {}:{}",
                    item.species_id, state.state_id
                ));
            }
            if (charge_sum - state.formal_charge_e as f64).abs() > 5.0e-4 {
                return Err(format!(
                    "partial charges do not sum to formal charge for {}:{}",
                    item.species_id, state.state_id
                ));
            }
            validate_sha256("endpoint parameter digest", &state.parameter_digest_sha256)?;
            let actual = fragment_state_parameter_digest_sha256(state)?;
            if actual != state.parameter_digest_sha256 {
                return Err(format!(
                    "endpoint parameter digest mismatch for {}:{}",
                    item.species_id, state.state_id
                ));
            }
        }
        let states_by_id: HashMap<u32, &FragmentChemicalState> = item
            .chemical_states
            .iter()
            .map(|state| (state.state_id, state))
            .collect();
        let mut transition_ids = HashSet::new();
        for transition in &item.state_transitions {
            if !transition_ids.insert(transition.transition_id.as_str()) {
                return Err(format!(
                    "duplicate state transition {} in species {}",
                    transition.transition_id, item.species_id
                ));
            }
            let state_a = states_by_id.get(&transition.state_a_id).ok_or_else(|| {
                format!(
                    "transition {} references absent state A",
                    transition.transition_id
                )
            })?;
            let state_b = states_by_id.get(&transition.state_b_id).ok_or_else(|| {
                format!(
                    "transition {} references absent state B",
                    transition.transition_id
                )
            })?;
            validate_state_transition(transition, state_a, state_b)?;
        }
    }

    let mut slot_ids = HashSet::new();
    let mut slot_molecules = HashSet::new();
    for slot in &system.slots {
        if !slot_ids.insert(slot.slot_id) || !slot_molecules.insert(slot.molecule_id) {
            return Err(format!("duplicate slot or slot molecule {}", slot.slot_id));
        }
        let registered_atoms = fragment_molecules.get(&slot.molecule_id).ok_or_else(|| {
            format!(
                "slot {} molecule is not registered as LAMBDA_FRAGMENT",
                slot.slot_id
            )
        })?;
        if registered_atoms != &slot.atom_indices {
            return Err(format!(
                "slot {} atom order differs from molecule registry",
                slot.slot_id
            ));
        }
        let sp = species.get(&slot.species_id).ok_or_else(|| {
            format!(
                "slot {} references absent species {}",
                slot.slot_id, slot.species_id
            )
        })?;
        let state = sp
            .chemical_states
            .iter()
            .find(|s| s.state_id == slot.chemical_state_id)
            .ok_or_else(|| format!("slot {} references absent chemical state", slot.slot_id))?;
        if state.atom_parameters.len() != slot.atom_indices.len() {
            return Err(format!(
                "slot {} state/template atom count mismatch",
                slot.slot_id
            ));
        }
        let endpoint_state =
            |endpoint: &NcmcEndpointIdentity| -> Result<&FragmentChemicalState, String> {
                let endpoint_species = species
                    .get(&endpoint.species_id)
                    .ok_or_else(|| format!("slot {} endpoint species absent", slot.slot_id))?;
                endpoint_species
                    .chemical_states
                    .iter()
                    .find(|candidate| candidate.state_id == endpoint.chemical_state_id)
                    .ok_or_else(|| format!("slot {} endpoint state absent", slot.slot_id))
            };
        let state_a = endpoint_state(&slot.endpoint_a)?;
        let state_b = endpoint_state(&slot.endpoint_b)?;
        if state_a.atom_parameters.len() != slot.atom_indices.len()
            || state_b.atom_parameters.len() != slot.atom_indices.len()
        {
            return Err(format!(
                "slot {} endpoint atom mapping mismatch",
                slot.slot_id
            ));
        }
        match slot.path_kind {
            AlchemicalPathKind::InsertDelete => {
                let occupancies = (slot.endpoint_a.occupancy, slot.endpoint_b.occupancy);
                if occupancies != (EndpointOccupancy::Empty, EndpointOccupancy::Occupied)
                    || slot.state_transition_id.is_some()
                    || slot.endpoint_a.species_id != slot.endpoint_b.species_id
                    || slot.endpoint_a.chemical_state_id != slot.endpoint_b.chemical_state_id
                {
                    return Err(format!(
                        "slot {} INSERT_DELETE path must be one identity, canonical EMPTY(A) to OCCUPIED(B), with no state transition",
                        slot.slot_id
                    ));
                }
            }
            AlchemicalPathKind::StateAToStateB => {
                if slot.endpoint_a.occupancy != EndpointOccupancy::Occupied
                    || slot.endpoint_b.occupancy != EndpointOccupancy::Occupied
                    || (slot.endpoint_a.species_id == slot.endpoint_b.species_id
                        && slot.endpoint_a.chemical_state_id == slot.endpoint_b.chemical_state_id)
                    || slot.endpoint_a.species_id != slot.endpoint_b.species_id
                {
                    return Err(format!(
                        "slot {} STATE_A_TO_STATE_B requires two distinct occupied endpoints",
                        slot.slot_id
                    ));
                }
                let transition_id = slot.state_transition_id.as_deref().ok_or_else(|| {
                    format!(
                        "slot {} state transform lacks transition identity",
                        slot.slot_id
                    )
                })?;
                let transition = sp
                    .state_transitions
                    .iter()
                    .find(|candidate| candidate.transition_id == transition_id)
                    .ok_or_else(|| {
                        format!(
                            "slot {} references absent state transition {}",
                            slot.slot_id, transition_id
                        )
                    })?;
                validate_state_transition(transition, state_a, state_b)?;
            }
        }
        match observation_eligibility(slot.lambda)? {
            ObservationEligibility::PhysicalEndpoint
                if slot.lambda.to_bits() == 0.0f64.to_bits()
                    && (slot.endpoint_occupancy != slot.endpoint_a.occupancy
                        || slot.species_id != slot.endpoint_a.species_id
                        || slot.chemical_state_id != slot.endpoint_a.chemical_state_id) =>
            {
                return Err(format!(
                    "slot {} lambda=0 occupancy mismatches endpoint A",
                    slot.slot_id
                ))
            }
            ObservationEligibility::PhysicalEndpoint
                if slot.lambda.to_bits() == 1.0f64.to_bits()
                    && (slot.endpoint_occupancy != slot.endpoint_b.occupancy
                        || slot.species_id != slot.endpoint_b.species_id
                        || slot.chemical_state_id != slot.endpoint_b.chemical_state_id) =>
            {
                return Err(format!(
                    "slot {} lambda=1 occupancy mismatches endpoint B",
                    slot.slot_id
                ))
            }
            ObservationEligibility::SwitchingOnly
                if slot.endpoint_occupancy != EndpointOccupancy::Switching =>
            {
                return Err(format!(
                    "slot {} intermediate lambda must be SWITCHING",
                    slot.slot_id
                ))
            }
            _ => {}
        }
    }
    if slot_molecules.len() != fragment_molecules.len() {
        return Err("every LAMBDA_FRAGMENT molecule must have exactly one slot".into());
    }

    validate_ncmc(&system.ncmc)?;
    let mut calibrations = HashSet::new();
    for calibration in &system.chemical_potentials {
        let calibrated_species = species.get(&calibration.species_id).ok_or_else(|| {
            "chemical-potential calibration references absent species".to_string()
        })?;
        let calibrated_state = calibrated_species
            .chemical_states
            .iter()
            .find(|state| state.state_id == calibration.chemical_state_id)
            .ok_or_else(|| "chemical-potential calibration references absent state".to_string())?;
        if !calibrations.insert((calibration.species_id, calibration.chemical_state_id)) {
            return Err("invalid or duplicate chemical-potential calibration".into());
        }
        calibration.validate()?;
        if calibration.state_reference_convention
            != StateReferenceConvention::TotalMuIncludesIntrinsicStateFreeEnergy
            || (calibration.included_intrinsic_state_free_energy_kcal_mol
                - calibrated_state.intrinsic_free_energy_kcal_mol)
                .abs()
                > 1.0e-12
        {
            return Err(format!(
                "chemical-potential state-reference convention mismatch for {}:{}",
                calibration.species_id, calibration.chemical_state_id
            ));
        }
        calibration.periodic_charge_treatment.validate_for_move(
            calibrated_state.formal_charge_e,
            calibration.included_finite_size_correction_kcal_mol,
            &system.periodic_box,
            &system.pme,
        )?;
    }
    for slot in &system.slots {
        if !calibrations.contains(&(slot.species_id, slot.chemical_state_id)) {
            return Err(format!(
                "slot {} lacks a chemical-potential calibration",
                slot.slot_id
            ));
        }
    }
    let calibration_by_state: HashMap<(u32, u32), &ChemicalPotentialCalibration> = system
        .chemical_potentials
        .iter()
        .map(|calibration| {
            (
                (calibration.species_id, calibration.chemical_state_id),
                calibration,
            )
        })
        .collect();
    for item in &system.species {
        let states_by_id: HashMap<u32, &FragmentChemicalState> = item
            .chemical_states
            .iter()
            .map(|state| (state.state_id, state))
            .collect();
        for transition in &item.state_transitions {
            let state_a = states_by_id[&transition.state_a_id];
            let state_b = states_by_id[&transition.state_b_id];
            let cal_a = calibration_by_state
                .get(&(item.species_id, state_a.state_id))
                .ok_or_else(|| "state transition lacks endpoint-A calibration".to_string())?;
            let cal_b = calibration_by_state
                .get(&(item.species_id, state_b.state_id))
                .ok_or_else(|| "state transition lacks endpoint-B calibration".to_string())?;
            transition.periodic_charge_treatment.validate_for_move(
                state_b.formal_charge_e - state_a.formal_charge_e,
                cal_b.included_finite_size_correction_kcal_mol
                    - cal_a.included_finite_size_correction_kcal_mol,
                &system.periodic_box,
                &system.pme,
            )?;
        }
    }
    Ok(())
}

pub(crate) fn validate_pme(pme: &PmeContract, periodic_box: &PeriodicBoxContract) -> Result<(), String> {
    const ALPHA_DERIVATION: &str =
        "sqrt(-ln(2*relative_force_tolerance))/real_space_cutoff_angstrom";
    const GRID_FORMULA: &str =
        "ceil(2*ewald_alpha*reduced_cell_diagonal/(3*relative_force_tolerance^(1/5)));min=6";
    const GRID_ROUNDING: &str = "PRISM_CUFFT_GOOD_SIZE_TABLE_V1_ROUND_UP_NO_DOWNROUND";
    if pme.reference_implementation != "OpenMM NonbondedForce PME"
        || pme.boundary_condition != "ORTHORHOMBIC_TINFOIL_3D"
        || !pme.ewald_alpha_a_inv.is_finite()
        || pme.ewald_alpha_a_inv <= 0.0
        || pme.pme_grid_dimensions.iter().any(|&n| n < 6)
        || pme.pme_grid_dimensions_lower_bound.iter().any(|&n| n < 6)
        || pme.pme_interpolation_order != 5
        || !pme.real_space_cutoff_angstrom.is_finite()
        || pme.real_space_cutoff_angstrom <= 0.0
        || !pme.relative_force_tolerance.is_finite()
        || !(0.0..=1.0e-3).contains(&pme.relative_force_tolerance)
        || pme.ewald_alpha_derivation != ALPHA_DERIVATION
        || pme.pme_grid_formula != GRID_FORMULA
        || pme.pme_grid_rounding_policy != GRID_ROUNDING
        || !pme.runtime_exact_parameter_lock_required
        || !pme.independent_openmm_force_energy_validation_required
    {
        return Err("PME contract is incomplete or outside production bounds".into());
    }
    let derived_beta =
        (-(2.0 * pme.relative_force_tolerance).ln()).sqrt() / pme.real_space_cutoff_angstrom;
    if (pme.ewald_alpha_a_inv - derived_beta).abs() > 1.0e-8 {
        return Err(
            "PME beta does not close OpenMM alpha=sqrt(-ln(2*tolerance))/real_space_cutoff".into(),
        );
    }
    let [a, b, c] = periodic_box.vectors_a;
    let off_diagonal = [a[1], a[2], b[0], b[2], c[0], c[1]];
    if off_diagonal.iter().any(|value| value.abs() > 1.0e-10)
        || [a[0], b[1], c[2]].iter().any(|length| *length <= 0.0)
    {
        return Err("current production PME runtime requires a positive orthorhombic cell".into());
    }
    let denominator = 3.0 * pme.relative_force_tolerance.powf(0.2);
    let expected_lower_bound = [a[0], b[1], c[2]].map(|length| {
        ((2.0 * pme.ewald_alpha_a_inv * length / denominator).ceil() as usize).max(6)
    });
    if pme.pme_grid_dimensions_lower_bound != expected_lower_bound
        || pme
            .pme_grid_dimensions
            .iter()
            .zip(expected_lower_bound)
            .any(|(actual, minimum)| *actual < minimum)
        || pme
            .pme_grid_dimensions
            .iter()
            .any(|dimension| !is_cufft_good_size(*dimension))
    {
        return Err(
            "PME mesh dimensions do not close the OpenMM lower bound and no-downround policy"
                .into(),
        );
    }
    validate_sha256("PME parameterization", &pme.pme_parameterization_sha256)?;
    if pme.parameterization_digest_sha256()? != pme.pme_parameterization_sha256 {
        return Err("PME 13-key canonical parameterization hash mismatch".into());
    }
    Ok(())
}

fn is_cufft_good_size(mut dimension: usize) -> bool {
    if dimension < 1 {
        return false;
    }
    for factor in [2, 3, 5] {
        while dimension % factor == 0 {
            dimension /= factor;
        }
    }
    dimension == 1
}

fn validate_water(water: &WaterContract) -> Result<(), String> {
    if water.model.trim().is_empty()
        || !water.settle_positions_enabled
        || !water.settle_velocities_enabled
        || !water.oh_distance_a.is_finite()
        || water.oh_distance_a <= 0.0
        || !water.hoh_angle_rad.is_finite()
        || !(0.0..std::f64::consts::PI).contains(&water.hoh_angle_rad)
    {
        return Err("rigid-water SETTLE contract is incomplete".into());
    }
    validate_sha256("water parameterization", &water.parameterization_sha256)
}

fn validate_ncmc(ncmc: &NcmcProtocol) -> Result<(), String> {
    if ncmc.forward_operations.is_empty()
        || ncmc.reverse_operations.len() != ncmc.forward_operations.len()
        || ncmc.integrator_splitting.trim().is_empty()
        || ncmc.rng_algorithm.trim().is_empty()
        || !ncmc.forward_protocol_probability.is_finite()
        || !(0.0..=1.0).contains(&ncmc.forward_protocol_probability)
        || ncmc.forward_protocol_probability == 0.0
        || !ncmc.reverse_protocol_probability.is_finite()
        || !(0.0..=1.0).contains(&ncmc.reverse_protocol_probability)
        || ncmc.reverse_protocol_probability == 0.0
        || !ncmc.initial_endpoint_equilibrium_density_evaluated
        || !ncmc.proposed_endpoint_equilibrium_density_evaluated
    {
        return Err("NCMC protocol is incomplete".into());
    }
    let mut current_lambda = 0.0;
    let mut perturbation_count = 0usize;
    for (index, operation) in ncmc.forward_operations.iter().enumerate() {
        match operation {
            NcmcProtocolOperation::Perturb {
                lambda_from,
                lambda_to,
            } => {
                if !lambda_from.is_finite()
                    || !lambda_to.is_finite()
                    || !(0.0..=1.0).contains(lambda_from)
                    || !(0.0..=1.0).contains(lambda_to)
                    || (*lambda_from - current_lambda).abs() > 1.0e-12
                    || lambda_to <= lambda_from
                {
                    return Err(format!("invalid ordered NCMC perturbation at {index}"));
                }
                current_lambda = *lambda_to;
                perturbation_count += 1;
            }
            NcmcProtocolOperation::Propagate {
                lambda,
                integration_steps,
            } => {
                if !lambda.is_finite()
                    || !(0.0..=1.0).contains(lambda)
                    || (*lambda - current_lambda).abs() > 1.0e-12
                    || *integration_steps == 0
                {
                    return Err(format!("invalid ordered NCMC propagation at {index}"));
                }
            }
            NcmcProtocolOperation::MomentumReverse => {}
        }
    }
    if perturbation_count == 0 || (current_lambda - 1.0).abs() > 1.0e-12 {
        return Err("NCMC forward protocol does not connect physical endpoints 0 to 1".into());
    }
    let expected_reverse: Vec<_> = ncmc
        .forward_operations
        .iter()
        .rev()
        .map(NcmcProtocolOperation::reversed)
        .collect();
    if ncmc.reverse_operations != expected_reverse {
        return Err("NCMC reverse protocol is not the exact ordered path reversal".into());
    }
    validate_sha256("NCMC schedule", &ncmc.schedule_sha256)?;
    if ncmc.schedule_sha256 != ncmc_schedule_sha256(ncmc)? {
        return Err("NCMC schedule hash does not close exact ordered operations".into());
    }
    Ok(())
}

/// Canonical schedule digest excluding the digest field itself.
pub fn ncmc_schedule_sha256(ncmc: &NcmcProtocol) -> Result<String, String> {
    // A struct, rather than a generic JSON map, makes field order part of the
    // cross-language digest contract.  Python mirrors this exact declaration.
    #[derive(Serialize)]
    struct NcmcScheduleDigestProjection<'a> {
        forward_operations: &'a [NcmcProtocolOperation],
        reverse_operations: &'a [NcmcProtocolOperation],
        forward_protocol_probability: f64,
        reverse_protocol_probability: f64,
        integrator_splitting: &'a str,
        integrator_path_probability_convention: NcmcIntegratorPathProbabilityConvention,
        momentum_reversal_policy: NcmcMomentumReversalPolicy,
        initial_endpoint_equilibrium_density_evaluated: bool,
        proposed_endpoint_equilibrium_density_evaluated: bool,
        rng_algorithm: &'a str,
    }
    let projection = NcmcScheduleDigestProjection {
        forward_operations: &ncmc.forward_operations,
        reverse_operations: &ncmc.reverse_operations,
        forward_protocol_probability: ncmc.forward_protocol_probability,
        reverse_protocol_probability: ncmc.reverse_protocol_probability,
        integrator_splitting: &ncmc.integrator_splitting,
        integrator_path_probability_convention: ncmc.integrator_path_probability_convention,
        momentum_reversal_policy: ncmc.momentum_reversal_policy,
        initial_endpoint_equilibrium_density_evaluated: ncmc
            .initial_endpoint_equilibrium_density_evaluated,
        proposed_endpoint_equilibrium_density_evaluated: ncmc
            .proposed_endpoint_equilibrium_density_evaluated,
        rng_algorithm: &ncmc.rng_algorithm,
    };
    let encoded = serde_json::to_vec(&projection).map_err(|error| error.to_string())?;
    Ok(format!("{:x}", Sha256::digest(encoded)))
}

fn validate_sha256(label: &str, value: &str) -> Result<(), String> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("{label} must be a 64-character SHA256"));
    }
    Ok(())
}

/// Durably publish immutable bytes without overwriting an existing artifact.
/// A hard-link commit gives create-new semantics; directory fsync makes the
/// name durable before the event sequence advances.
fn atomic_write_synced(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| "durable lambda artifact path lacks parent".to_string())?;
    std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    if path.exists() {
        let existing = std::fs::read(path).map_err(|error| error.to_string())?;
        return if existing == bytes {
            Ok(())
        } else {
            Err(format!(
                "refusing to overwrite conflicting immutable lambda artifact {}",
                path.display()
            ))
        };
    }
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| "durable lambda artifact has invalid filename".to_string())?;
    let temp = parent.join(format!(
        ".{file_name}.tmp.{}.{}",
        std::process::id(),
        format!("{:x}", Sha256::digest(bytes))
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)
        .map_err(|error| error.to_string())?;
    file.write_all(bytes).map_err(|error| error.to_string())?;
    file.sync_all().map_err(|error| error.to_string())?;
    drop(file);
    match std::fs::hard_link(&temp, path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let existing = std::fs::read(path).map_err(|read_error| read_error.to_string())?;
            if existing != bytes {
                let _ = std::fs::remove_file(&temp);
                return Err(format!(
                    "immutable lambda artifact race produced conflicting bytes at {}",
                    path.display()
                ));
            }
        }
        Err(error) => {
            let _ = std::fs::remove_file(&temp);
            return Err(error.to_string());
        }
    }
    std::fs::remove_file(&temp).map_err(|error| error.to_string())?;
    File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| error.to_string())?;
    Ok(())
}

fn validate_dispatch_binding(
    opcode_name: &str,
    opcode_id: u16,
    expected_opcode_name: &str,
    qnanl_name: &str,
    expected_family: QnanlArtifactFamily,
) -> Result<(), String> {
    let canonical_opcode_id = opcode_id_for_name(expected_opcode_name)
        .ok_or_else(|| format!("PrismCode registry lacks {expected_opcode_name}"))?;
    if opcode_name != expected_opcode_name || opcode_id != canonical_opcode_id {
        return Err(format!(
            "PrismCode opcode binding mismatch: {}:{} expected {}:{}",
            opcode_name, opcode_id, expected_opcode_name, canonical_opcode_id
        ));
    }
    if qnanl_name != expected_family.as_str() {
        return Err(format!(
            "QNANL family binding mismatch: {} expected {}",
            qnanl_name,
            expected_family.as_str()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
    struct CanonicalU128Probe {
        #[serde(with = "u128_decimal_string")]
        value: u128,
    }

    #[test]
    fn persisted_u128_decimal_abi_is_canonical_and_adversarially_closed() {
        let maximum = CanonicalU128Probe { value: u128::MAX };
        let encoded = serde_json::to_string(&maximum).unwrap();
        assert_eq!(encoded, format!(r#"{{"value":"{}"}}"#, u128::MAX));
        assert_eq!(
            serde_json::from_str::<CanonicalU128Probe>(&encoded).unwrap(),
            maximum
        );
        for invalid in [
            r#"{"value":1}"#,
            r#"{"value":"01"}"#,
            r#"{"value":"+1"}"#,
            r#"{"value":"-1"}"#,
            r#"{"value":" 1"}"#,
            r#"{"value":"1 "}"#,
            r#"{"value":""}"#,
            r#"{"value":"340282366920938463463374607431768211456"}"#,
        ] {
            assert!(
                serde_json::from_str::<CanonicalU128Probe>(invalid).is_err(),
                "accepted noncanonical persisted u128: {invalid}"
            );
        }

        let scheduler = test_scheduler_contract("e".repeat(64));
        let scheduler_json = serde_json::to_value(&scheduler).unwrap();
        assert_eq!(scheduler_json["schema_version"], 3);
        assert!(scheduler_json["rng_counter_start"].is_string());
        assert!(scheduler_json["scheduled_opportunities"][0]["rng_counter_base"].is_string());
        for field in [
            "selection_counter_offset",
            "placement_counter_start_offset",
            "placement_counter_count",
            "orientation_counter_start_offset",
            "orientation_counter_count",
            "momentum_counter_start_offset",
            "momentum_counter_count",
            "thermostat_counter_start_offset",
            "thermostat_counter_count",
            "acceptance_counter_offset",
            "counters_per_opportunity",
        ] {
            assert!(scheduler_json["rng_counter_layout"][field].is_string());
        }
        let endpoints = BTreeMap::from([(
            9,
            NcmcEndpointIdentity {
                species_id: 7,
                chemical_state_id: 0,
                occupancy: EndpointOccupancy::Empty,
            },
        )]);
        let proposal = scheduler
            .proposal_for_step(18, &endpoints, NcmcExecutionMode::CalibratedTargetLane)
            .unwrap();
        let proposal_json = serde_json::to_value(&proposal).unwrap();
        assert!(proposal_json["rng_counter"].is_string());
        for field in [
            "opportunity_counter_base",
            "selection_counter",
            "acceptance_counter",
        ] {
            assert!(proposal_json["rng_allocation"][field].is_string());
        }
        for field in ["placement", "orientation", "momenta", "thermostat"] {
            assert!(proposal_json["rng_allocation"][field]["start"].is_string());
            assert!(proposal_json["rng_allocation"][field]["count"].is_string());
        }
        let consumption = test_rng_consumption(&proposal);
        let consumption_json = serde_json::to_value(&consumption).unwrap();
        assert_eq!(consumption_json["schema_version"], 2);
        assert!(consumption_json["selection"]["draw_count"].is_string());
        assert!(consumption_json["acceptance"]["draw_count"].is_string());
        assert!(consumption_json["allocation"]["acceptance_counter"].is_string());
        scheduler.validate().unwrap();
        consumption
            .validate_for_scheduler(&proposal, &scheduler)
            .unwrap();
    }

    fn test_capacity_contract() -> LambdaTypedEventCapacityContract {
        let mut contract = LambdaTypedEventCapacityContract {
            schema_kind: "prismos_lambda_typed_event_capacity_contract".into(),
            schema_version: 1,
            maximum_event_count: 128,
            maximum_total_bytes: 128 * 1_048_576,
            maximum_event_bytes: 1_048_576,
            reserved_inode_count: 1024,
            durable_files_per_event: 4,
            safety_margin: 1.25,
            contract_sha256: String::new(),
        };
        contract.contract_sha256 = contract.recompute_sha256().unwrap();
        contract
    }

    fn test_resident_dual_endpoint_union_contract_sha256(system: &LambdaFragmentSystem) -> String {
        let mut digest = Sha256::new();
        digest.update(b"PRISM_TEST_RESIDENT_DUAL_ENDPOINT_UNION_CONTRACT_V1\0");
        digest.update(serde_json::to_vec(system).unwrap());
        format!("{:x}", digest.finalize())
    }

    fn test_placement_authority(
        system: &LambdaFragmentSystem,
        scheduler: &NcmcProposalSchedulerContract,
        matched: &LambdaMatchedVelocityInitializationContract,
        production_admission_allowed: bool,
    ) -> DecoupledFragmentPlacementMeasureAuthority {
        let (coordinate_ensemble_sha256, mixing_evidence_sha256) =
            match &scheduler.placement_measure {
                NcmcPlacementMeasureAuthority::PersistentDecoupledCoordinateEnsemble {
                    coordinate_ensemble_sha256,
                    mixing_evidence_sha256,
                    ..
                } => (
                    coordinate_ensemble_sha256.clone(),
                    mixing_evidence_sha256.clone(),
                ),
                NcmcPlacementMeasureAuthority::UniformAccessibleVolume { .. } => {
                    panic!("test runtime requires persistent coordinates")
                }
            };
        let species = &system.species[0];
        let state = &species.chemical_states[0];
        let mut authority = DecoupledFragmentPlacementMeasureAuthority {
            schema_kind: "prismos_decoupled_fragment_placement_measure_authority".into(),
            schema_version: 1,
            authority_id: "test-persistent-decoupled-placement".into(),
            status: "IMPLEMENTATION_VERIFIED".into(),
            calibration_sampling_admission_allowed: true,
            production_admission_allowed,
            precursor_placement_measure_sha256: None,
            promotion_receipt_sha256: None,
            chemical_potential_evidence_sha256: None,
            cross_cell_unlock_sha256: None,
            finite_size_admission_sha256: None,
            seed_topology_member_sha256: None,
            panel_scientific_sha256: None,
            target_protocol_authority_sha256: None,
            source_branch_authority_sha256: matched
                .fragment_coordinate_branch_authority_sha256
                .clone()
                .unwrap(),
            ncmc_placement_projection_sha256: ncmc_placement_projection_sha256(
                &scheduler.placement_measure,
            )
            .unwrap(),
            generator: serde_json::json!({"algorithm":"TEST_COUNTER_BASED_BRANCH_V1"}),
            thermodynamic_authority_sha256: "3".repeat(64),
            cell_authority_sha256: "4".repeat(64),
            cell_id: "test-cell".into(),
            periodic_cell_sha256: matched.periodic_cell_contract_sha256.clone(),
            temperature_k: 300.0,
            species_id: species.species_id,
            chemical_state_id: state.state_id,
            parameter_template_sha256: species.parameter_template_sha256.clone(),
            endpoint_parameter_sha256: state.parameter_digest_sha256.clone(),
            slot_count: system.slots.len() as u32,
            position_measure: serde_json::json!({"measure":"FRACTIONAL_CELL_UNIFORM"}),
            orientation_measure: serde_json::json!({"measure":"HAAR_SO3"}),
            internal_measure: serde_json::json!({"measure":"AUTHENTICATED_ENSEMBLE"}),
            slot_selection_measure: serde_json::json!({"measure":"DISCRETE_UNIFORM"}),
            ncmc_path_reversal: serde_json::json!({"rule":"EXACT_REVERSE"}),
            coordinate_ensemble_sha256,
            mixing_evidence_sha256,
            trajectory_artifacts: vec![serde_json::json!({
                "path":"test://decoupled-coordinate-ensemble",
                "sha256":"5".repeat(64)
            })],
            com_fractional_uniformity_gate_pass: true,
            orientation_haar_gate_pass: true,
            per_torsion_mixing_ess_gate_pass: true,
            slot_exchangeability_gate_pass: true,
            all_proposal_densities_finite_nonzero: true,
            forward_reverse_proposal_density_exact: true,
            mixing_gate_pass: true,
            placement_measure_sha256: String::new(),
        };
        assert!(
            !production_admission_allowed,
            "test helper requires explicit promotion construction"
        );
        authority.placement_measure_sha256 = authority.recompute_sha256().unwrap();
        authority
    }

    fn promote_test_placement_authority(
        mut authority: DecoupledFragmentPlacementMeasureAuthority,
        system: &LambdaFragmentSystem,
        scheduler: &NcmcProposalSchedulerContract,
        matched: &LambdaMatchedVelocityInitializationContract,
        topology_sha256: &str,
        replica_seed: u64,
        cross_cell_unlock_sha256: &str,
        panel_scientific_sha256: &str,
    ) -> DecoupledFragmentPlacementMeasureAuthority {
        authority.validate().unwrap();
        let precursor_placement_measure_sha256 = authority.placement_measure_sha256.clone();
        let chemical_potential_evidence_sha256 =
            lambda_chemical_potential_evidence_sha256(system).unwrap();
        let finite_size_admission_sha256 = lambda_finite_size_admission_sha256(system).unwrap();
        let seed_topology_member_sha256 = lambda_seed_topology_member_sha256(
            topology_sha256,
            &matched.npt_frozen_cell_authority_sha256,
            &matched.periodic_cell_contract_sha256,
            replica_seed,
        )
        .unwrap();
        let target_protocol_authority_sha256 = scheduler.contract_sha256.clone();
        let promotion_receipt_sha256 = decoupled_fragment_promotion_receipt_sha256(
            &precursor_placement_measure_sha256,
            &chemical_potential_evidence_sha256,
            cross_cell_unlock_sha256,
            &finite_size_admission_sha256,
            &seed_topology_member_sha256,
            panel_scientific_sha256,
            &target_protocol_authority_sha256,
        )
        .unwrap();
        authority.production_admission_allowed = true;
        authority.precursor_placement_measure_sha256 = Some(precursor_placement_measure_sha256);
        authority.promotion_receipt_sha256 = Some(promotion_receipt_sha256);
        authority.chemical_potential_evidence_sha256 = Some(chemical_potential_evidence_sha256);
        authority.cross_cell_unlock_sha256 = Some(cross_cell_unlock_sha256.into());
        authority.finite_size_admission_sha256 = Some(finite_size_admission_sha256);
        authority.seed_topology_member_sha256 = Some(seed_topology_member_sha256);
        authority.panel_scientific_sha256 = Some(panel_scientific_sha256.into());
        authority.target_protocol_authority_sha256 = Some(target_protocol_authority_sha256);
        authority.placement_measure_sha256 = authority.recompute_sha256().unwrap();
        authority.validate().unwrap();
        authority
    }

    fn test_matched_velocity_contract(
        fragment_lane: bool,
    ) -> LambdaMatchedVelocityInitializationContract {
        let masses = [12.0f32, 1.0, 16.0, 14.0];
        let periodic_cell_vectors = [[20.0f64, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]];
        let mut contract = LambdaMatchedVelocityInitializationContract {
            schema_kind: "prismos_lambda_matched_velocity_initialization_contract".into(),
            schema_version: 3,
            environment_atom_count: masses.len() as u32,
            environment_atom_membership_sha256: matched_environment_atom_membership_sha256(
                masses.len(),
            ),
            environment_mass_sha256: matched_environment_mass_sha256(&masses),
            environment_velocity_rng_domain: "prism.matched.environment.velocity.v1".into(),
            fragment_velocity_rng_domain: "prism.matched.fragment.velocity.v1".into(),
            environment_position_transfer:
                "EXACT_AFTER_AUTHORIZED_F64_TO_F32_PROJECTION_NO_LATER_MUTATION".into(),
            environment_position_f64_checkpoint_sha256: "6".repeat(64),
            f64_to_f32_projection_policy:
                "IEEE754_BINARY64_TO_BINARY32_ROUND_TO_NEAREST_TIES_TO_EVEN".into(),
            environment_position_sha256: "c".repeat(64),
            fragment_position_transfer: if fragment_lane {
                "EXACT_AUTHENTICATED_DECOUPLED_COORDINATE_BRANCH_BYTES".into()
            } else {
                "NOT_APPLICABLE_NO_PROBE".into()
            },
            fragment_position_f64_checkpoint_sha256: fragment_lane.then(|| "e".repeat(64)),
            fragment_position_sha256: fragment_lane.then(|| "7".repeat(64)),
            fragment_coordinate_branch_authority_sha256: fragment_lane.then(|| "f".repeat(64)),
            fragment_coordinate_ensemble_sha256: fragment_lane.then(|| "a".repeat(64)),
            periodic_cell_contract_sha256: "8".repeat(64),
            periodic_cell_f64_checkpoint_sha256: matched_periodic_cell_f64_sha256(
                &periodic_cell_vectors,
            ),
            periodic_cell_f32_projection_sha256: matched_periodic_cell_f32_projection_sha256(
                &periodic_cell_vectors,
            ),
            npt_checkpoint_population_authority_sha256: "9".repeat(64),
            npt_checkpoint_population_distinct_seed_count: 5,
            npt_seed_authority_sha256: "1".repeat(64),
            npt_frozen_cell_authority_sha256: "2".repeat(64),
            frozen_checkpoint_replica_seed: 42,
            contract_sha256: String::new(),
        };
        contract.contract_sha256 = contract.recompute_sha256().unwrap();
        contract
    }

    fn test_matched_velocity_receipt(
        contract: &LambdaMatchedVelocityInitializationContract,
        stream_id: u16,
        replica_seed: u64,
    ) -> LambdaMatchedVelocityInitializationReceipt {
        LambdaMatchedVelocityInitializationReceipt {
            schema_kind: "prismos_lambda_matched_velocity_initialization_receipt".into(),
            schema_version: 3,
            stream_id,
            replica_seed,
            temperature_k: 300.0,
            contract_sha256: contract.contract_sha256.clone(),
            environment_atom_count: contract.environment_atom_count,
            fragment_atom_count: u32::from(
                contract
                    .fragment_coordinate_branch_authority_sha256
                    .is_some(),
            ),
            environment_position_f64_checkpoint_sha256: contract
                .environment_position_f64_checkpoint_sha256
                .clone(),
            f64_to_f32_projection_policy: contract.f64_to_f32_projection_policy.clone(),
            periodic_cell_contract_sha256: contract.periodic_cell_contract_sha256.clone(),
            periodic_cell_f64_checkpoint_sha256: contract
                .periodic_cell_f64_checkpoint_sha256
                .clone(),
            periodic_cell_f32_projection_sha256: contract
                .periodic_cell_f32_projection_sha256
                .clone(),
            npt_checkpoint_population_authority_sha256: contract
                .npt_checkpoint_population_authority_sha256
                .clone(),
            npt_seed_authority_sha256: contract.npt_seed_authority_sha256.clone(),
            npt_frozen_cell_authority_sha256: contract.npt_frozen_cell_authority_sha256.clone(),
            fragment_coordinate_branch_authority_sha256: contract
                .fragment_coordinate_branch_authority_sha256
                .clone(),
            fragment_position_f64_checkpoint_sha256: contract
                .fragment_position_f64_checkpoint_sha256
                .clone(),
            fragment_position_sha256: contract.fragment_position_sha256.clone(),
            fragment_coordinate_ensemble_sha256: contract
                .fragment_coordinate_ensemble_sha256
                .clone(),
            environment_velocity_sha256: "a".repeat(64),
            full_velocity_sha256: "b".repeat(64),
            environment_position_sha256: contract.environment_position_sha256.clone(),
            full_position_sha256: "d".repeat(64),
            receipt_sha256: String::new(),
        }
        .seal()
        .unwrap()
    }

    fn test_rng_allocation() -> NcmcMoveRngAllocation {
        NcmcMoveRngAllocation {
            opportunity_counter_base: 40,
            selection_counter: 40,
            placement: NcmcRngCounterRange {
                start: 41,
                count: 2,
            },
            orientation: NcmcRngCounterRange {
                start: 43,
                count: 2,
            },
            momenta: NcmcRngCounterRange {
                start: 45,
                count: 2,
            },
            thermostat: NcmcRngCounterRange {
                start: 47,
                count: 2,
            },
            acceptance_counter: 49,
        }
    }

    fn test_rng_consumption_raw(
        rng_key: u64,
        move_ordinal: u64,
        scheduler_contract_sha256: String,
        move_kind: NcmcMoveKind,
    ) -> NcmcRngConsumptionReceipt {
        let allocation = test_rng_allocation();
        let selection_range = NcmcRngCounterRange {
            start: allocation.selection_counter,
            count: 1,
        };
        let acceptance_range = NcmcRngCounterRange {
            start: allocation.acceptance_counter,
            count: 1,
        };
        let mut receipt = NcmcRngConsumptionReceipt {
            schema_kind: "prismos_ncmc_rng_consumption_receipt".into(),
            schema_version: 2,
            scheduler_contract_sha256,
            move_ordinal,
            allocation,
            selection: NcmcRngRangeDisposition::consumed(
                rng_key,
                NcmcRngPurpose::Selection,
                &selection_range,
                "STATE_CONDITIONED_SUPPORT_CATEGORICAL_V2",
            )
            .unwrap(),
            placement: NcmcRngRangeDisposition::NotApplicable {
                physical_reason: if matches!(
                    move_kind,
                    NcmcMoveKind::Insertion | NcmcMoveKind::Deletion
                ) {
                    NcmcRngNonConsumptionReason::PersistentDecoupledCoordinatesRetained
                } else {
                    NcmcRngNonConsumptionReason::StableSlotIdentityTransform
                },
            },
            orientation: NcmcRngRangeDisposition::NotApplicable {
                physical_reason: if matches!(
                    move_kind,
                    NcmcMoveKind::Insertion | NcmcMoveKind::Deletion
                ) {
                    NcmcRngNonConsumptionReason::PersistentDecoupledCoordinatesRetained
                } else {
                    NcmcRngNonConsumptionReason::StableSlotIdentityTransform
                },
            },
            momenta: NcmcRngRangeDisposition::NotApplicable {
                physical_reason: NcmcRngNonConsumptionReason::ExistingEquilibriumMomentaRetained,
            },
            thermostat: NcmcRngRangeDisposition::Forbidden {
                physical_reason: NcmcRngNonConsumptionReason::DeterministicGammaZeroThermostat,
            },
            acceptance: NcmcRngRangeDisposition::consumed(
                rng_key,
                NcmcRngPurpose::Acceptance,
                &acceptance_range,
                "METROPOLIS_UNIFORM_OPEN01_V1",
            )
            .unwrap(),
            counters_irrevocably_reserved_after_rejection: true,
            receipt_sha256: String::new(),
        };
        receipt.receipt_sha256 = receipt.recompute_sha256().unwrap();
        receipt
    }

    fn test_rng_consumption(proposal: &NcmcMoveProposal) -> NcmcRngConsumptionReceipt {
        NcmcRngConsumptionReceipt::seal(
            proposal,
            NcmcRngRangeDisposition::NotApplicable {
                physical_reason:
                    NcmcRngNonConsumptionReason::PersistentDecoupledCoordinatesRetained,
            },
            NcmcRngRangeDisposition::NotApplicable {
                physical_reason:
                    NcmcRngNonConsumptionReason::PersistentDecoupledCoordinatesRetained,
            },
            NcmcRngRangeDisposition::NotApplicable {
                physical_reason: NcmcRngNonConsumptionReason::ExistingEquilibriumMomentaRetained,
            },
            NcmcRngRangeDisposition::Forbidden {
                physical_reason: NcmcRngNonConsumptionReason::DeterministicGammaZeroThermostat,
            },
        )
        .unwrap()
    }

    fn test_scheduler_contract(schedule_sha256: String) -> NcmcProposalSchedulerContract {
        let empty = NcmcEndpointIdentity {
            species_id: 7,
            chemical_state_id: 0,
            occupancy: EndpointOccupancy::Empty,
        };
        let occupied = NcmcEndpointIdentity {
            species_id: 7,
            chemical_state_id: 0,
            occupancy: EndpointOccupancy::Occupied,
        };
        let balance_template = NcmcProposalBalanceTemplate {
            temperature_k: 300.0,
            ph: 7.0,
            accessible_volume_a3: 8000.0,
            forward_placement_density_a3_inv: 1.25e-4,
            reverse_placement_density_a3_inv: 1.25e-4,
            forward_orientation_density: 0.079_577_471_545_947_67,
            reverse_orientation_density: 0.079_577_471_545_947_67,
            initial_bias_weight: 1.0,
            proposed_bias_weight: 1.0,
            bias_epoch_sha256: "9".repeat(64),
        };
        let mut contract = NcmcProposalSchedulerContract {
            schema_kind: "prismos_ncmc_proposal_scheduler_contract".into(),
            schema_version: 3,
            scheduler_id: "test-state-conditioned-v3".into(),
            admitted_slot_ids: vec![9],
            proposal_supports: vec![
                NcmcProposalSupport {
                    support_id: "slot9-empty-to-occupied".into(),
                    reverse_support_id: "slot9-occupied-to-empty".into(),
                    slot_id: 9,
                    move_kind: NcmcMoveKind::Insertion,
                    initial: empty.clone(),
                    proposed: occupied.clone(),
                    base_weight: 1.0,
                    balance_template: balance_template.clone(),
                    periodic_charge_treatment: PeriodicChargeTreatment::NetChargeInvariant {
                        subject_charge_change_e: 0,
                    },
                    coupled_charge_partner: None,
                },
                NcmcProposalSupport {
                    support_id: "slot9-occupied-to-empty".into(),
                    reverse_support_id: "slot9-empty-to-occupied".into(),
                    slot_id: 9,
                    move_kind: NcmcMoveKind::Deletion,
                    initial: occupied,
                    proposed: empty,
                    base_weight: 1.0,
                    balance_template,
                    periodic_charge_treatment: PeriodicChargeTreatment::NetChargeInvariant {
                        subject_charge_change_e: 0,
                    },
                    coupled_charge_partner: None,
                },
            ],
            scheduled_opportunities: vec![NcmcScheduledOpportunity {
                move_ordinal: 0,
                step: 18,
                allowed_support_ids: vec![
                    "slot9-empty-to-occupied".into(),
                    "slot9-occupied-to-empty".into(),
                ],
                rng_counter_base: 8,
            }],
            placement_measure:
                NcmcPlacementMeasureAuthority::PersistentDecoupledCoordinateEnsemble {
                    coordinate_ensemble_sha256: "a".repeat(64),
                    mixing_evidence_sha256: "b".repeat(64),
                    placement_density_formula: "q_decoupled(x|slot,epoch)".into(),
                    orientation_density_formula: "q_decoupled(R|slot,epoch)".into(),
                },
            first_move_step: 18,
            move_interval_steps: 10,
            equilibrated_after_step: 18,
            minimum_decorrelation_steps: 10,
            maximum_move_count: 1,
            rng_algorithm: "SPLITMIX64_COUNTER_V1".into(),
            rng_domain: "test.lambda.ncmc".into(),
            rng_key: 7,
            rng_counter_start: 8,
            rng_counter_layout: NcmcRngCounterLayout {
                selection_counter_offset: 0,
                placement_counter_start_offset: 1,
                placement_counter_count: 4,
                orientation_counter_start_offset: 5,
                orientation_counter_count: 4,
                momentum_counter_start_offset: 9,
                momentum_counter_count: 16,
                thermostat_counter_start_offset: 25,
                thermostat_counter_count: 16,
                acceptance_counter_offset: 41,
                counters_per_opportunity: 42,
            },
            bias_epoch_sha256: "9".repeat(64),
            forward_proposal_density_formula: "q_fwd=move*selection*placement*orientation".into(),
            reverse_proposal_density_formula: "q_rev=move*selection*placement*orientation".into(),
            ncmc_protocol_schedule_sha256: schedule_sha256,
            contract_sha256: String::new(),
        };
        contract.contract_sha256 = contract.recompute_sha256().unwrap();
        contract
    }

    #[test]
    fn ncmc_scheduler_v2_is_state_conditioned_and_has_exact_reverse_support() {
        let mut scheduler = test_scheduler_contract("e".repeat(64));
        let empty = BTreeMap::from([(
            9,
            NcmcEndpointIdentity {
                species_id: 7,
                chemical_state_id: 0,
                occupancy: EndpointOccupancy::Empty,
            },
        )]);
        let occupied = BTreeMap::from([(
            9,
            NcmcEndpointIdentity {
                species_id: 7,
                chemical_state_id: 0,
                occupancy: EndpointOccupancy::Occupied,
            },
        )]);
        let insertion = scheduler
            .proposal_for_step(18, &empty, NcmcExecutionMode::CalibratedTargetLane)
            .unwrap();
        let deletion = scheduler
            .proposal_for_step(18, &occupied, NcmcExecutionMode::CalibratedTargetLane)
            .unwrap();
        assert_eq!(insertion.move_kind, NcmcMoveKind::Insertion);
        assert_eq!(deletion.move_kind, NcmcMoveKind::Deletion);
        assert_eq!(insertion.balance_context.forward_selection_probability, 1.0);
        assert_eq!(insertion.balance_context.reverse_selection_probability, 1.0);
        scheduler
            .validate_proposal_at_step(18, &insertion, &empty)
            .unwrap();
        assert_eq!(
            scheduler
                .next_opportunity_at_or_after(0)
                .unwrap()
                .unwrap()
                .step,
            18
        );
        assert_eq!(
            scheduler
                .next_opportunity_at_or_after(18)
                .unwrap()
                .unwrap()
                .move_ordinal,
            0
        );
        assert!(scheduler
            .next_opportunity_at_or_after(19)
            .unwrap()
            .is_none());

        scheduler.scheduled_opportunities[0].allowed_support_ids =
            vec!["slot9-empty-to-occupied".into()];
        scheduler.contract_sha256 = scheduler.recompute_sha256().unwrap();
        assert!(scheduler.validate().is_err());
    }

    #[test]
    fn ncmc_scheduler_v2_rejects_zero_eligibility_and_rng_overlap() {
        let scheduler = test_scheduler_contract("e".repeat(64));
        let foreign_state = BTreeMap::from([(
            9,
            NcmcEndpointIdentity {
                species_id: 99,
                chemical_state_id: 3,
                occupancy: EndpointOccupancy::Occupied,
            },
        )]);
        assert!(scheduler
            .proposal_for_step(18, &foreign_state, NcmcExecutionMode::CalibratedTargetLane,)
            .is_err());

        let mut overlap = scheduler;
        overlap.rng_counter_layout.acceptance_counter_offset = 1;
        overlap.contract_sha256 = overlap.recompute_sha256().unwrap();
        assert!(overlap.validate().is_err());
    }

    #[test]
    fn rejected_move_cannot_reuse_any_next_opportunity_counter_subrange() {
        let mut scheduler = test_scheduler_contract("e".repeat(64));
        scheduler.maximum_move_count = 2;
        let mut second = scheduler.scheduled_opportunities[0].clone();
        second.move_ordinal = 1;
        second.step = 28;
        second.rng_counter_base =
            scheduler.rng_counter_start + scheduler.rng_counter_layout.counters_per_opportunity;
        scheduler.scheduled_opportunities.push(second);
        scheduler.contract_sha256 = scheduler.recompute_sha256().unwrap();
        scheduler.validate().unwrap();

        // Simulate rejection: the physical endpoint remains EMPTY, but the
        // next deterministic opportunity must advance to a disjoint counter
        // span instead of rewinding proposal counters with device phase state.
        let endpoints = BTreeMap::from([(
            9,
            NcmcEndpointIdentity {
                species_id: 7,
                chemical_state_id: 0,
                occupancy: EndpointOccupancy::Empty,
            },
        )]);
        let first = scheduler
            .proposal_for_step(18, &endpoints, NcmcExecutionMode::CalibratedTargetLane)
            .unwrap();
        let next = scheduler
            .proposal_for_step(28, &endpoints, NcmcExecutionMode::CalibratedTargetLane)
            .unwrap();
        let first_consumption = test_rng_consumption(&first);
        let next_consumption = test_rng_consumption(&next);
        first_consumption.validate_for_proposal(&first).unwrap();
        next_consumption.validate_for_proposal(&next).unwrap();
        assert!(first_consumption.counters_irrevocably_reserved_after_rejection);

        let intervals = |allocation: &NcmcMoveRngAllocation| {
            vec![
                (
                    allocation.selection_counter,
                    allocation.selection_counter + 1,
                ),
                (
                    allocation.placement.start,
                    allocation.placement.end_exclusive().unwrap(),
                ),
                (
                    allocation.orientation.start,
                    allocation.orientation.end_exclusive().unwrap(),
                ),
                (
                    allocation.momenta.start,
                    allocation.momenta.end_exclusive().unwrap(),
                ),
                (
                    allocation.thermostat.start,
                    allocation.thermostat.end_exclusive().unwrap(),
                ),
                (
                    allocation.acceptance_counter,
                    allocation.acceptance_counter + 1,
                ),
            ]
        };
        for first_range in intervals(&first.rng_allocation) {
            for next_range in intervals(&next.rng_allocation) {
                assert!(
                    first_range.1 <= next_range.0 || next_range.1 <= first_range.0,
                    "rejected opportunity counter range was reused"
                );
            }
        }
    }

    #[test]
    fn python_parameterizer_v2_fixture_deserializes_and_digest_closes() {
        let species: FragmentSpecies = serde_json::from_str(include_str!(
            "../tests/fixtures/rust_fragment_species_v2.json"
        ))
        .unwrap();
        assert_eq!(species.species_id, 9001);
        assert_eq!(species.chemical_states.len(), 1);
        let state = &species.chemical_states[0];
        assert_eq!(
            fragment_state_parameter_digest_sha256(state).unwrap(),
            state.parameter_digest_sha256
        );
        assert_eq!(
            fragment_bonded_parameter_plane_sha256(&state.bonded_parameter_plane).unwrap(),
            state.bonded_parameter_plane_sha256
        );
        assert!(species.state_transitions.is_empty());
    }

    fn charge_test_box() -> PeriodicBoxContract {
        PeriodicBoxContract {
            vectors_a: [[40.0, 0.0, 0.0], [0.0, 40.0, 0.0], [0.0, 0.0, 40.0]],
            minimum_image_enabled: true,
            source_sha256: "a".repeat(64),
        }
    }

    fn charge_test_pme() -> PmeContract {
        PmeContract {
            reference_implementation: "OpenMM NonbondedForce PME".into(),
            boundary_condition: "ORTHORHOMBIC_TINFOIL_3D".into(),
            pme_interpolation_order: 5,
            relative_force_tolerance: 1.0e-5,
            real_space_cutoff_angstrom: 10.0,
            ewald_alpha_a_inv: (-(2.0e-5_f64).ln()).sqrt() / 10.0,
            ewald_alpha_derivation:
                "sqrt(-ln(2*relative_force_tolerance))/real_space_cutoff_angstrom".into(),
            pme_grid_dimensions_lower_bound: [88, 88, 88],
            pme_grid_dimensions: [90, 90, 90],
            pme_grid_formula:
                "ceil(2*ewald_alpha*reduced_cell_diagonal/(3*relative_force_tolerance^(1/5)));min=6"
                    .into(),
            pme_grid_rounding_policy: "PRISM_CUFFT_GOOD_SIZE_TABLE_V1_ROUND_UP_NO_DOWNROUND".into(),
            runtime_exact_parameter_lock_required: true,
            independent_openmm_force_energy_validation_required: true,
            pme_parameterization_sha256: "b".repeat(64),
        }
    }

    fn ncmc_test_protocol() -> NcmcProtocol {
        // Deliberately begin with a perturbation and end with propagation.
        // Detailed balance requires the exact reverse path and its probability;
        // it does not impose an artificial propagation operation at both ends.
        let forward_operations = vec![
            NcmcProtocolOperation::Perturb {
                lambda_from: 0.0,
                lambda_to: 0.25,
            },
            NcmcProtocolOperation::Propagate {
                lambda: 0.25,
                integration_steps: 4,
            },
            NcmcProtocolOperation::Perturb {
                lambda_from: 0.25,
                lambda_to: 1.0,
            },
            NcmcProtocolOperation::Propagate {
                lambda: 1.0,
                integration_steps: 8,
            },
        ];
        let reverse_operations = forward_operations
            .iter()
            .rev()
            .map(NcmcProtocolOperation::reversed)
            .collect();
        let mut protocol = NcmcProtocol {
            forward_operations,
            reverse_operations,
            forward_protocol_probability: 0.375,
            reverse_protocol_probability: 0.625,
            integrator_splitting: "BAOAB-EXACT-OU".into(),
            integrator_path_probability_convention:
                NcmcIntegratorPathProbabilityConvention::ExactConditionalPathProbability,
            momentum_reversal_policy: NcmcMomentumReversalPolicy::ReversePathAndRejection,
            initial_endpoint_equilibrium_density_evaluated: true,
            proposed_endpoint_equilibrium_density_evaluated: true,
            rng_algorithm: "PHILOX4X32_10".into(),
            schedule_sha256: "0".repeat(64),
        };
        protocol.schedule_sha256 = ncmc_schedule_sha256(&protocol).unwrap();
        protocol
    }

    #[test]
    fn ncmc_protocol_accepts_perturb_first_and_closes_exact_reverse_path() {
        let protocol = ncmc_test_protocol();
        assert!(matches!(
            protocol.forward_operations.first(),
            Some(NcmcProtocolOperation::Perturb { .. })
        ));
        assert!(matches!(
            protocol.reverse_operations.first(),
            Some(NcmcProtocolOperation::Propagate { lambda: 1.0, .. })
        ));
        validate_ncmc(&protocol).unwrap();
    }

    #[test]
    fn ncmc_protocol_rejects_nonreversed_operation_schedule() {
        let mut protocol = ncmc_test_protocol();
        protocol.reverse_operations.swap(0, 1);
        protocol.schedule_sha256 = ncmc_schedule_sha256(&protocol).unwrap();
        assert!(validate_ncmc(&protocol)
            .unwrap_err()
            .contains("exact ordered path reversal"));
    }

    #[test]
    fn ncmc_protocol_rejects_zero_protocol_selection_probability() {
        let mut protocol = ncmc_test_protocol();
        protocol.reverse_protocol_probability = 0.0;
        protocol.schedule_sha256 = ncmc_schedule_sha256(&protocol).unwrap();
        assert!(validate_ncmc(&protocol)
            .unwrap_err()
            .contains("protocol is incomplete"));
    }

    #[test]
    fn ncmc_protocol_rejects_tampered_schedule_digest() {
        let mut protocol = ncmc_test_protocol();
        protocol.forward_protocol_probability = 0.5;
        // Keep the previously computed digest so the mutation must be caught.
        assert!(validate_ncmc(&protocol)
            .unwrap_err()
            .contains("schedule hash does not close"));
    }

    #[test]
    fn ordinary_md_chunk_planner_never_crosses_nondivisible_ncmc_boundaries() {
        let mut scheduler = test_scheduler_contract(ncmc_test_protocol().schedule_sha256);
        let mut second = scheduler.scheduled_opportunities[0].clone();
        second.move_ordinal = 1;
        second.step = scheduler.first_move_step + scheduler.move_interval_steps;
        second.rng_counter_base += scheduler.rng_counter_layout.counters_per_opportunity;
        scheduler.scheduled_opportunities.push(second);
        scheduler.maximum_move_count = 2;
        scheduler.contract_sha256 = scheduler.recompute_sha256().unwrap();
        scheduler.validate().unwrap();
        assert_eq!(
            bound_ordinary_md_chunk_before_ncmc(&scheduler, 0, 1_000, false).unwrap(),
            scheduler.first_move_step
        );
        assert_eq!(
            bound_ordinary_md_chunk_before_ncmc(
                &scheduler,
                scheduler.first_move_step - 1,
                7,
                false,
            )
            .unwrap(),
            1
        );
        assert!(bound_ordinary_md_chunk_before_ncmc(
            &scheduler,
            scheduler.first_move_step,
            7,
            false,
        )
        .is_err());
        assert_eq!(
            bound_ordinary_md_chunk_before_ncmc(
                &scheduler,
                scheduler.first_move_step,
                scheduler.move_interval_steps + 13,
                true,
            )
            .unwrap(),
            scheduler.move_interval_steps
        );
        let after_last = scheduler.scheduled_opportunities.last().unwrap().step + 1;
        assert_eq!(
            bound_ordinary_md_chunk_before_ncmc(&scheduler, after_last, 37, false).unwrap(),
            37
        );
    }

    fn resident_window_for_test(logical_steps: u64) -> NcmcResidentWindowReceipt {
        NcmcResidentWindowReceipt {
            logical_steps,
            host_graph_launches: 1,
            legacy_graph_replay_dispatch_count: 0,
            resident_execution: true,
            capture_identity_sha256: "1".repeat(64),
            graph_structure_sha256: "2".repeat(64),
            deterministic_reversible_propagation: true,
            graph_exec_sha256: "3".repeat(64),
        }
    }

    fn boundary_population_for_test(
        move_id: u64,
        direction: NcmcProtocolDirection,
        operation_index: u32,
        kind: NcmcForceBoundaryKind,
        slot_id: u32,
    ) -> NcmcBoundaryPopulationLineage {
        let boundary =
            NcmcForceBoundaryIdentity::new(move_id, direction, operation_index, kind).unwrap();
        NcmcBoundaryPopulationLineage::seal(
            boundary.clone(),
            vec![NcmcBoundarySlotEventLineage {
                slot_id,
                event_sequence: u64::from(boundary.boundary_ordinal),
                raw_event_sha256: "a".repeat(64),
                emission_receipt_sha256: "b".repeat(64),
            }],
        )
        .unwrap()
    }

    fn operation_lineage_for_test(
        move_id: u64,
        direction: NcmcProtocolDirection,
        operation_index: u32,
        before: NcmcForceBoundaryKind,
        after: NcmcForceBoundaryKind,
        slot_id: u32,
    ) -> NcmcOperationBoundaryLineage {
        NcmcOperationBoundaryLineage {
            before: boundary_population_for_test(
                move_id,
                direction,
                operation_index,
                before,
                slot_id,
            ),
            after: boundary_population_for_test(
                move_id,
                direction,
                operation_index,
                after,
                slot_id,
            ),
        }
    }

    fn zero_work_trace_for_test(
        protocol: &NcmcProtocol,
        move_id: u64,
        direction: NcmcProtocolDirection,
        slot_id: u32,
    ) -> NcmcOrderedWorkTrace {
        let operations = match direction {
            NcmcProtocolDirection::Forward => &protocol.forward_operations,
            NcmcProtocolDirection::Reverse => &protocol.reverse_operations,
        };
        let entries = operations
            .iter()
            .enumerate()
            .map(|(index, operation)| match operation {
                NcmcProtocolOperation::Perturb {
                    lambda_from,
                    lambda_to,
                } => NcmcWorkTraceEntry::Perturb {
                    operation_index: index as u32,
                    boundary_lineage: operation_lineage_for_test(
                        move_id,
                        direction,
                        index as u32,
                        NcmcForceBoundaryKind::PerturbPre,
                        NcmcForceBoundaryKind::PerturbPost,
                        slot_id,
                    ),
                    lambda_from: *lambda_from,
                    lambda_to: *lambda_to,
                    potential_before_kcal_mol: 0.0,
                    potential_after_kcal_mol: 0.0,
                    reduced_work: 0.0,
                },
                NcmcProtocolOperation::Propagate {
                    lambda,
                    integration_steps,
                } => NcmcWorkTraceEntry::Propagate {
                    operation_index: index as u32,
                    boundary_lineage: operation_lineage_for_test(
                        move_id,
                        direction,
                        index as u32,
                        NcmcForceBoundaryKind::PropagatePre,
                        NcmcForceBoundaryKind::PropagatePost,
                        slot_id,
                    ),
                    lambda: *lambda,
                    integration_steps: *integration_steps,
                    hamiltonian_before_kcal_mol: 0.0,
                    hamiltonian_after_kcal_mol: 0.0,
                    constraint_impulse_work_kcal_mol: 0.0,
                    reduced_shadow_work: 0.0,
                    resident_window: resident_window_for_test(u64::from(*integration_steps)),
                },
                NcmcProtocolOperation::MomentumReverse => NcmcWorkTraceEntry::MomentumReverse {
                    operation_index: index as u32,
                    boundary_lineage: operation_lineage_for_test(
                        move_id,
                        direction,
                        index as u32,
                        NcmcForceBoundaryKind::MomentumReversePre,
                        NcmcForceBoundaryKind::MomentumReversePost,
                        slot_id,
                    ),
                    kinetic_before_kcal_mol: 0.0,
                    kinetic_after_kcal_mol: 0.0,
                    exact_involution_applied: true,
                },
            })
            .collect();
        NcmcOrderedWorkTrace::seal(protocol, move_id, direction, 1.0, entries, true, true).unwrap()
    }

    fn emit_zero_work_trace_for_test(
        emitter: &mut LambdaTypedEventEmitter,
        system: &LambdaFragmentSystem,
        move_id: u64,
        direction: NcmcProtocolDirection,
        stream_id: u32,
        step: u64,
        phase_protocol_sha256: &str,
    ) -> NcmcOrderedWorkTrace {
        let slot = &system.slots[0];
        let endpoint_support_binding_sha256 = lambda_endpoint_support_binding_sha256(
            system,
            emitter
                .runtime_binding
                .as_ref()
                .and_then(|binding| binding.ncmc_scheduler.as_ref())
                .expect("runtime test emitter scheduler"),
        )
        .unwrap();
        let dual_endpoint_union_contract_sha256 =
            test_resident_dual_endpoint_union_contract_sha256(system);
        let operations = match direction {
            NcmcProtocolDirection::Forward => &system.ncmc.forward_operations,
            NcmcProtocolDirection::Reverse => &system.ncmc.reverse_operations,
        };
        let mut current_lambda = match operations.first().unwrap() {
            NcmcProtocolOperation::Perturb { lambda_from, .. } => *lambda_from,
            NcmcProtocolOperation::Propagate { lambda, .. } => *lambda,
            NcmcProtocolOperation::MomentumReverse => slot.lambda,
        };
        let mut entries = Vec::new();
        for (operation_index, operation) in operations.iter().enumerate() {
            let (pre_kind, post_kind, post_lambda) = match operation {
                NcmcProtocolOperation::Perturb { lambda_to, .. } => (
                    NcmcForceBoundaryKind::PerturbPre,
                    NcmcForceBoundaryKind::PerturbPost,
                    *lambda_to,
                ),
                NcmcProtocolOperation::Propagate { lambda, .. } => (
                    NcmcForceBoundaryKind::PropagatePre,
                    NcmcForceBoundaryKind::PropagatePost,
                    *lambda,
                ),
                NcmcProtocolOperation::MomentumReverse => (
                    NcmcForceBoundaryKind::MomentumReversePre,
                    NcmcForceBoundaryKind::MomentumReversePost,
                    current_lambda,
                ),
            };
            let emit_boundary = |emitter: &mut LambdaTypedEventEmitter,
                                 kind: NcmcForceBoundaryKind,
                                 lambda: f64|
             -> NcmcBoundaryPopulationLineage {
                let boundary = NcmcForceBoundaryIdentity::new(
                    move_id,
                    direction,
                    operation_index as u32,
                    kind,
                )
                .unwrap();
                let event = LambdaFragmentForceEvent {
                    schema_kind: "prismos_lambda_fragment_force_event".into(),
                    schema_version: 3,
                    stream_id,
                    step,
                    event_sequence: emitter.next_sequence(),
                    parent_receipt_sha256: emitter.last_receipt_sha256().map(str::to_string),
                    ncmc_boundary: Some(boundary.clone()),
                    slot_id: slot.slot_id,
                    species_id: slot.species_id,
                    chemical_state_id: slot.chemical_state_id,
                    lambda,
                    path_kind: slot.path_kind,
                    endpoint_a: slot.endpoint_a.clone(),
                    endpoint_b: slot.endpoint_b.clone(),
                    potential_kcal_mol: 0.0,
                    du_dlambda_kcal_mol: 0.0,
                    energy_unit: "kcal/mol".into(),
                    force_unit: "kcal/mol/angstrom".into(),
                    endpoint_parameter_sha256: "a".repeat(64),
                    endpoint_support_binding_sha256: endpoint_support_binding_sha256.clone(),
                    dual_endpoint_union_contract_sha256: dual_endpoint_union_contract_sha256
                        .clone(),
                    hamiltonian_sha256: system.softcore.digest_sha256().unwrap(),
                    phase_protocol_sha256: phase_protocol_sha256.into(),
                    prismcode_opcode: prismcode_opcodes::FORCE.into(),
                    prismcode_opcode_id: opcode_id_for_name(prismcode_opcodes::FORCE).unwrap(),
                    qnanl_artifact_family: QnanlArtifactFamily::LambdaFragmentForceEvent
                        .as_str()
                        .into(),
                };
                let emission = emitter.emit_force(&event).unwrap();
                NcmcBoundaryPopulationLineage::seal(
                    boundary,
                    vec![NcmcBoundarySlotEventLineage {
                        slot_id: slot.slot_id,
                        event_sequence: event.event_sequence,
                        raw_event_sha256: emission.raw_sha256,
                        emission_receipt_sha256: emission.receipt_sha256,
                    }],
                )
                .unwrap()
            };
            let before = emit_boundary(emitter, pre_kind, current_lambda);
            let after = emit_boundary(emitter, post_kind, post_lambda);
            let boundary_lineage = NcmcOperationBoundaryLineage { before, after };
            entries.push(match operation {
                NcmcProtocolOperation::Perturb {
                    lambda_from,
                    lambda_to,
                } => NcmcWorkTraceEntry::Perturb {
                    operation_index: operation_index as u32,
                    boundary_lineage,
                    lambda_from: *lambda_from,
                    lambda_to: *lambda_to,
                    potential_before_kcal_mol: 0.0,
                    potential_after_kcal_mol: 0.0,
                    reduced_work: 0.0,
                },
                NcmcProtocolOperation::Propagate {
                    lambda,
                    integration_steps,
                } => NcmcWorkTraceEntry::Propagate {
                    operation_index: operation_index as u32,
                    boundary_lineage,
                    lambda: *lambda,
                    integration_steps: *integration_steps,
                    hamiltonian_before_kcal_mol: 0.0,
                    hamiltonian_after_kcal_mol: 0.0,
                    constraint_impulse_work_kcal_mol: 0.0,
                    reduced_shadow_work: 0.0,
                    resident_window: resident_window_for_test(u64::from(*integration_steps)),
                },
                NcmcProtocolOperation::MomentumReverse => NcmcWorkTraceEntry::MomentumReverse {
                    operation_index: operation_index as u32,
                    boundary_lineage,
                    kinetic_before_kcal_mol: 0.0,
                    kinetic_after_kcal_mol: 0.0,
                    exact_involution_applied: true,
                },
            });
            current_lambda = post_lambda;
        }
        NcmcOrderedWorkTrace::seal(&system.ncmc, move_id, direction, 1.0, entries, true, true)
            .unwrap()
    }

    #[test]
    fn ordered_work_trace_recomputes_every_operation_and_rejects_scalar_tamper() {
        let protocol = ncmc_test_protocol();
        let move_id = 8;
        let entries = vec![
            NcmcWorkTraceEntry::Perturb {
                operation_index: 0,
                boundary_lineage: operation_lineage_for_test(
                    move_id,
                    NcmcProtocolDirection::Forward,
                    0,
                    NcmcForceBoundaryKind::PerturbPre,
                    NcmcForceBoundaryKind::PerturbPost,
                    2,
                ),
                lambda_from: 0.0,
                lambda_to: 0.25,
                potential_before_kcal_mol: 1.0,
                potential_after_kcal_mol: 2.0,
                reduced_work: 2.0,
            },
            NcmcWorkTraceEntry::Propagate {
                operation_index: 1,
                boundary_lineage: operation_lineage_for_test(
                    move_id,
                    NcmcProtocolDirection::Forward,
                    1,
                    NcmcForceBoundaryKind::PropagatePre,
                    NcmcForceBoundaryKind::PropagatePost,
                    2,
                ),
                lambda: 0.25,
                integration_steps: 4,
                hamiltonian_before_kcal_mol: 3.0,
                hamiltonian_after_kcal_mol: 3.5,
                constraint_impulse_work_kcal_mol: 0.0,
                reduced_shadow_work: 1.0,
                resident_window: resident_window_for_test(4),
            },
            NcmcWorkTraceEntry::Perturb {
                operation_index: 2,
                boundary_lineage: operation_lineage_for_test(
                    move_id,
                    NcmcProtocolDirection::Forward,
                    2,
                    NcmcForceBoundaryKind::PerturbPre,
                    NcmcForceBoundaryKind::PerturbPost,
                    2,
                ),
                lambda_from: 0.25,
                lambda_to: 1.0,
                potential_before_kcal_mol: 2.0,
                potential_after_kcal_mol: 4.0,
                reduced_work: 4.0,
            },
            NcmcWorkTraceEntry::Propagate {
                operation_index: 3,
                boundary_lineage: operation_lineage_for_test(
                    move_id,
                    NcmcProtocolDirection::Forward,
                    3,
                    NcmcForceBoundaryKind::PropagatePre,
                    NcmcForceBoundaryKind::PropagatePost,
                    2,
                ),
                lambda: 1.0,
                integration_steps: 8,
                hamiltonian_before_kcal_mol: 5.0,
                hamiltonian_after_kcal_mol: 4.0,
                constraint_impulse_work_kcal_mol: 0.0,
                reduced_shadow_work: -2.0,
                resident_window: resident_window_for_test(8),
            },
        ];
        let trace = NcmcOrderedWorkTrace::seal(
            &protocol,
            move_id,
            NcmcProtocolDirection::Forward,
            2.0,
            entries,
            true,
            true,
        )
        .unwrap();
        assert_eq!(trace.perturbation_work_reduced, 6.0);
        assert_eq!(trace.shadow_work_reduced, -1.0);
        assert_eq!(trace.schema_version, 2);
        assert_eq!(trace.protocol_direction, NcmcProtocolDirection::Forward);
        trace.validate(&protocol).unwrap();

        let mut scalar_tamper = trace.clone();
        scalar_tamper.perturbation_work_reduced += 0.25;
        assert!(scalar_tamper.validate(&protocol).is_err());

        let mut dispatch_tamper = trace;
        if let NcmcWorkTraceEntry::Propagate {
            resident_window, ..
        } = &mut dispatch_tamper.entries[1]
        {
            resident_window.legacy_graph_replay_dispatch_count = 1;
        }
        assert!(dispatch_tamper.validate(&protocol).is_err());
    }

    #[test]
    fn ncmc_schedule_digest_matches_python_four_step_oracle() {
        let mut forward_operations = Vec::new();
        for index in 0..4 {
            let lambda_from = index as f64 / 4.0;
            let lambda_to = (index + 1) as f64 / 4.0;
            forward_operations.push(NcmcProtocolOperation::Perturb {
                lambda_from,
                lambda_to,
            });
            if index + 1 < 4 {
                forward_operations.push(NcmcProtocolOperation::Propagate {
                    lambda: lambda_to,
                    integration_steps: 2,
                });
            }
        }
        let reverse_operations = forward_operations
            .iter()
            .rev()
            .map(NcmcProtocolOperation::reversed)
            .collect();
        let protocol = NcmcProtocol {
            forward_operations,
            reverse_operations,
            forward_protocol_probability: 1.0,
            reverse_protocol_probability: 1.0,
            integrator_splitting: "V R O R V".into(),
            integrator_path_probability_convention:
                NcmcIntegratorPathProbabilityConvention::ShadowWorkVolumePreserving,
            momentum_reversal_policy: NcmcMomentumReversalPolicy::ReversePathAndRejection,
            initial_endpoint_equilibrium_density_evaluated: true,
            proposed_endpoint_equilibrium_density_evaluated: true,
            rng_algorithm: "PHILOX_COUNTER_BASED".into(),
            schedule_sha256: "0".repeat(64),
        };
        assert_eq!(
            ncmc_schedule_sha256(&protocol).unwrap(),
            "fe0269ef38b46c06fcc658f2c75be85c662dcf972300de3861972ee7457422b6"
        );
    }

    #[test]
    fn pme_parameterization_digest_matches_preparation_projection() {
        let pme = PmeContract {
            reference_implementation: "OpenMM NonbondedForce PME".into(),
            boundary_condition: "ORTHORHOMBIC_TINFOIL_3D".into(),
            pme_interpolation_order: 5,
            relative_force_tolerance: 1.0e-5,
            real_space_cutoff_angstrom: 10.0,
            ewald_alpha_a_inv: 0.328_934_313_874_522_43,
            ewald_alpha_derivation:
                "sqrt(-ln(2*relative_force_tolerance))/real_space_cutoff_angstrom".into(),
            pme_grid_dimensions_lower_bound: [257, 257, 257],
            pme_grid_dimensions: [270, 270, 270],
            pme_grid_formula:
                "ceil(2*ewald_alpha*reduced_cell_diagonal/(3*relative_force_tolerance^(1/5)));min=6"
                    .into(),
            pme_grid_rounding_policy: "PRISM_CUFFT_GOOD_SIZE_TABLE_V1_ROUND_UP_NO_DOWNROUND".into(),
            runtime_exact_parameter_lock_required: true,
            independent_openmm_force_energy_validation_required: true,
            pme_parameterization_sha256:
                "5429d29f9bf775e1025f3e51e10a230c9b8ca3b3c8f8672bc2e8101d30b179fb".into(),
        };
        assert_eq!(
            pme.parameterization_digest_sha256().unwrap(),
            pme.pme_parameterization_sha256
        );
    }

    #[test]
    fn charge_neutral_coupled_path_closes_and_reverses_exactly() {
        let forward = PeriodicChargeTreatment::CoupledChargeNeutralPath {
            subject_charge_change_e: 1,
            partner_charge_change_e: -1,
            combined_charge_change_e: 0,
            coupled_path_id: "fragment-plus-coion".into(),
            partner_slot_id: 41,
            coupling_schedule_sha256: "c".repeat(64),
            evidence_sha256: "d".repeat(64),
        };
        forward
            .validate_for_move(1, 0.0, &charge_test_box(), &charge_test_pme())
            .unwrap();
        let reverse = forward.reversed();
        reverse
            .validate_for_move(-1, 0.0, &charge_test_box(), &charge_test_pme())
            .unwrap();
        assert_eq!(reverse.reversed(), forward);
    }

    #[test]
    fn box_pme_finite_size_correction_closes_and_background_only_fails() {
        let correction = PeriodicChargeTreatment::ExplicitFiniteSizeCorrection {
            subject_charge_change_e: 1,
            correction_model: "ROCKLIN_NET_CHARGE_SOLVATION_V1".into(),
            correction_free_energy_kcal_mol: 0.73,
            correction_uncertainty_kcal_mol: 0.04,
            production_box_source_sha256: "a".repeat(64),
            pme_parameterization_sha256: "b".repeat(64),
            evidence_sha256: "e".repeat(64),
        };
        correction
            .validate_for_move(1, 0.73, &charge_test_box(), &charge_test_pme())
            .unwrap();
        correction
            .reversed()
            .validate_for_move(-1, -0.73, &charge_test_box(), &charge_test_pme())
            .unwrap();

        let background_only = PeriodicChargeTreatment::ExplicitFiniteSizeCorrection {
            subject_charge_change_e: 1,
            correction_model: "UNIFORM_BACKGROUND_ONLY".into(),
            correction_free_energy_kcal_mol: 0.0,
            correction_uncertainty_kcal_mol: 0.0,
            production_box_source_sha256: "a".repeat(64),
            pme_parameterization_sha256: "b".repeat(64),
            evidence_sha256: "e".repeat(64),
        };
        assert!(background_only
            .validate_for_move(1, 0.0, &charge_test_box(), &charge_test_pme())
            .unwrap_err()
            .contains("production box and PME"));
    }

    fn h() -> SoftcoreHamiltonian {
        SoftcoreHamiltonian::default()
    }

    fn bonded_plane(
        bond_length_a: f64,
        angle_rad: f64,
        torsion_phase_rad: f64,
    ) -> FragmentBondedParameterPlane {
        FragmentBondedParameterPlane {
            formula: "PRISM_ENDPOINT_BONDED_AMBER_V1".into(),
            bonds: vec![FragmentBondParameter {
                template_atom_i: 1,
                template_atom_j: 2,
                length_a: bond_length_a,
                force_constant_kcal_mol_a2: 110.0,
            }],
            angles: vec![FragmentAngleParameter {
                template_atom_i: 1,
                template_atom_j: 2,
                template_atom_k: 3,
                angle_rad,
                force_constant_kcal_mol_rad2: 38.0,
            }],
            proper_and_improper_torsions: vec![FragmentTorsionParameter {
                template_atom_i: 1,
                template_atom_j: 2,
                template_atom_k: 3,
                template_atom_l: 4,
                periodicity: 2,
                phase_rad: torsion_phase_rad,
                barrier_kcal_mol: 1.7,
                improper: false,
            }],
            constraints: Vec::new(),
        }
    }

    fn chemical_state(
        state_id: u32,
        proton_count: i32,
        intrinsic: f64,
        plane: FragmentBondedParameterPlane,
    ) -> FragmentChemicalState {
        let bonded_parameter_plane_sha256 = fragment_bonded_parameter_plane_sha256(&plane).unwrap();
        let mut state = FragmentChemicalState {
            state_id,
            state_canonical_isomeric_smiles: format!("state-{state_id}"),
            state_explicit_h_mapped_smiles: format!("[C:1][C:2][N:3][H:4]-{state_id}"),
            state_inchikey_fixed_h: format!("FIXEDH-STATE-{state_id}"),
            tautomer_id: format!("tautomer-{state_id}"),
            protonation_id: format!("prot-{state_id}"),
            formal_charge_e: 0,
            proton_count,
            atom_mapping: (1..=4)
                .map(|id| FragmentStateAtomMapping {
                    template_atom_id: id,
                    atomic_number: if id == 4 { 1 } else { 6 },
                    element: if id == 4 { "H" } else { "C" }.into(),
                    atom_map_number: id,
                })
                .collect(),
            reference_conformer_positions_a: vec![
                [0.0, 0.0, 0.0],
                [1.4, 0.1, 0.0],
                [2.2, 1.2, 0.2],
                [3.0, 1.1, 1.3],
            ],
            atom_parameters: (1..=4)
                .map(|id| FragmentAtomState {
                    template_atom_id: id,
                    endpoint_present: true,
                    charge_e: 0.0,
                    sigma_a: if id == 4 { 1.0 } else { 3.2 },
                    epsilon_kcal_mol: if id == 4 { 0.02 } else { 0.1 },
                })
                .collect(),
            parameter_digest_sha256: String::new(),
            intrinsic_free_energy_kcal_mol: intrinsic,
            intramolecular_parameters_invariant: false,
            intramolecular_parameter_set_sha256: "a".repeat(64),
            bonded_parameter_plane: plane,
            bonded_parameter_plane_sha256,
            exact_exception_parameters: Vec::new(),
        };
        state.parameter_digest_sha256 = fragment_state_parameter_digest_sha256(&state).unwrap();
        state
    }

    fn transition(
        state_a: &FragmentChemicalState,
        state_b: &FragmentChemicalState,
    ) -> FragmentStateTransition {
        let mut value = FragmentStateTransition {
            transition_id: "state-a-to-b".into(),
            state_a_id: state_a.state_id,
            state_b_id: state_b.state_id,
            atom_map_a_to_b: (1..=4)
                .map(|id| FragmentEndpointAtomMap {
                    atom_a: id,
                    atom_b: id,
                    element: if id == 4 { "H" } else { "C" }.into(),
                    map_confidence: 1.0,
                })
                .collect(),
            unmatched_a: Vec::new(),
            unmatched_b: Vec::new(),
            bond_graph_change_set: Vec::new(),
            constant_topology_union_atoms: (1..=4)
                .map(|id| FragmentConstantTopologyUnionAtom {
                    union_template_atom_id: id,
                    state_a_template_atom_id: Some(id),
                    state_b_template_atom_id: Some(id),
                    atomic_number: if id == 4 { 1 } else { 6 },
                    element: if id == 4 { "H" } else { "C" }.into(),
                    present_in_state_a: true,
                    present_in_state_b: true,
                })
                .collect(),
            dummy_proton_required: false,
            bonded_plane_invariant: state_a.bonded_parameter_plane_sha256
                == state_b.bonded_parameter_plane_sha256,
            constraints_invariant: true,
            reference_free_energy_b_minus_a_kcal_mol: state_b.intrinsic_free_energy_kcal_mol
                - state_a.intrinsic_free_energy_kcal_mol,
            delta_bound_protons_b_minus_a: state_b.proton_count - state_a.proton_count,
            periodic_charge_treatment: PeriodicChargeTreatment::NetChargeInvariant {
                subject_charge_change_e: 0,
            },
            required_runtime_capability: "CONSTANT_TOPOLOGY_UNION_DUMMY_PROTON_V1".into(),
            transition_sha256: String::new(),
        };
        value.transition_sha256 = fragment_state_transition_sha256(&value).unwrap();
        value
    }

    fn test_balance(target: f64, proposal: f64) -> NcmcDerivedBalance {
        NcmcDerivedBalance {
            beta_mol_per_kcal: 1.0,
            beta_chemical_potential_term: target,
            beta_intrinsic_state_reference_term_in_total_mu: 0.0,
            beta_nonreference_chemical_potential_term: target,
            beta_periodic_finite_size_correction_term_in_total_mu: 0.0,
            beta_calibrated_mu_remainder_term: target,
            periodic_charge_treatment: PeriodicChargeTreatment::NetChargeInvariant {
                subject_charge_change_e: 0,
            },
            state_reference_convention:
                StateReferenceConvention::TotalMuIncludesIntrinsicStateFreeEnergy,
            ph_log_weight_term: 0.0,
            standard_state_volume_term: 0.0,
            occupancy_combinatorics_term: 0.0,
            target_bias_term: 0.0,
            log_target_ratio: target,
            selection_proposal_term: proposal,
            placement_proposal_term: 0.0,
            orientation_proposal_term: 0.0,
            log_proposal_ratio: proposal,
            derivation_sha256: "d".repeat(64),
        }
    }

    fn test_proposal_context(log_proposal_ratio: f64) -> NcmcBalanceContext {
        let forward_selection_probability = 0.37;
        NcmcBalanceContext {
            // Makes beta exactly one under the production kcal/mol/K value.
            temperature_k: 1.0 / 0.001_987_204_258_640_83_f64,
            ph: 7.0,
            accessible_volume_a3: 1.0,
            occupied_count_before: 1,
            occupied_count_after: 1,
            forward_selection_probability,
            reverse_selection_probability: forward_selection_probability * log_proposal_ratio.exp(),
            forward_placement_density_a3_inv: 1.0,
            reverse_placement_density_a3_inv: 1.0,
            forward_orientation_density: 1.0,
            reverse_orientation_density: 1.0,
            initial_bias_weight: 1.0,
            proposed_bias_weight: 1.0,
            bias_epoch_sha256: "f".repeat(64),
        }
    }

    fn precalibration_test_system() -> LambdaFragmentSystem {
        let state = chemical_state(0, 0, 0.0, bonded_plane(1.42, 1.91, 0.37));
        let forward_operations = vec![
            NcmcProtocolOperation::Perturb {
                lambda_from: 0.0,
                lambda_to: 1.0,
            },
            NcmcProtocolOperation::Propagate {
                lambda: 1.0,
                integration_steps: 1,
            },
        ];
        let reverse_operations = forward_operations
            .iter()
            .rev()
            .map(NcmcProtocolOperation::reversed)
            .collect();
        let mut ncmc = NcmcProtocol {
            forward_operations,
            reverse_operations,
            forward_protocol_probability: 0.5,
            reverse_protocol_probability: 0.5,
            integrator_splitting: "BAOAB".into(),
            integrator_path_probability_convention:
                NcmcIntegratorPathProbabilityConvention::ShadowWorkVolumePreserving,
            momentum_reversal_policy: NcmcMomentumReversalPolicy::ReversePathAndRejection,
            initial_endpoint_equilibrium_density_evaluated: true,
            proposed_endpoint_equilibrium_density_evaluated: true,
            rng_algorithm: "PHILOX".into(),
            schedule_sha256: String::new(),
        };
        ncmc.schedule_sha256 = ncmc_schedule_sha256(&ncmc).unwrap();
        LambdaFragmentSystem {
            schema_kind: "prismos_lambda_fragment_system".into(),
            schema_version: 2,
            experiment_id: "rust-precalibration-fixture".into(),
            periodic_box: PeriodicBoxContract {
                vectors_a: [[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]],
                minimum_image_enabled: true,
                source_sha256: "a".repeat(64),
            },
            pme: PmeContract {
                reference_implementation: "OpenMM NonbondedForce PME".into(),
                boundary_condition: "ORTHORHOMBIC_TINFOIL_3D".into(),
                pme_interpolation_order: 5,
                relative_force_tolerance: 1.0e-5,
                real_space_cutoff_angstrom: 9.0,
                ewald_alpha_a_inv: (-(2.0e-5_f64).ln()).sqrt() / 9.0,
                ewald_alpha_derivation:
                    "sqrt(-ln(2*relative_force_tolerance))/real_space_cutoff_angstrom".into(),
                pme_grid_dimensions_lower_bound: [49, 49, 49],
                pme_grid_dimensions: [50, 50, 50],
                pme_grid_formula: "fixture-grid".into(),
                pme_grid_rounding_policy: "PRISM_CUFFT_GOOD_SIZE_TABLE_V1_ROUND_UP_NO_DOWNROUND"
                    .into(),
                runtime_exact_parameter_lock_required: true,
                independent_openmm_force_energy_validation_required: true,
                pme_parameterization_sha256: "b".repeat(64),
            },
            water: WaterContract {
                model: "TIP3P-FB".into(),
                settle_positions_enabled: true,
                settle_velocities_enabled: true,
                oh_distance_a: 0.9572,
                hoh_angle_rad: 1.824_218_134,
                parameterization_sha256: "c".repeat(64),
            },
            molecules: Vec::new(),
            species: vec![FragmentSpecies {
                species_id: 7,
                name: "rust-neutral-fragment".into(),
                canonical_isomeric_smiles: "C".into(),
                inchikey: "VNWKTOKETHGBQD-UHFFFAOYSA-N".into(),
                parameter_template_sha256: "d".repeat(64),
                chemical_states: vec![state],
                state_transitions: Vec::new(),
            }],
            slots: vec![FragmentSlot {
                slot_id: 9,
                molecule_id: 1009,
                species_id: 7,
                chemical_state_id: 0,
                atom_indices: vec![0, 1, 2, 3],
                lambda: 0.0,
                endpoint_occupancy: EndpointOccupancy::Empty,
                path_kind: AlchemicalPathKind::InsertDelete,
                state_transition_id: None,
                endpoint_a: NcmcEndpointIdentity {
                    species_id: 7,
                    chemical_state_id: 0,
                    occupancy: EndpointOccupancy::Empty,
                },
                endpoint_b: NcmcEndpointIdentity {
                    species_id: 7,
                    chemical_state_id: 0,
                    occupancy: EndpointOccupancy::Occupied,
                },
            }],
            softcore: h(),
            chemical_potentials: Vec::new(),
            ncmc,
        }
    }

    #[test]
    fn analytic_force_matches_central_difference() {
        let displacement = [3.17, -0.43, 0.81];
        let lambda = 0.37;
        let e = evaluate_softcore_pair(displacement, lambda, -0.21, 3.4, 0.12, &h()).unwrap();
        let delta = 1.0e-6;
        for axis in 0..3 {
            let mut plus = displacement;
            let mut minus = displacement;
            plus[axis] += delta;
            minus[axis] -= delta;
            let up = evaluate_softcore_pair(plus, lambda, -0.21, 3.4, 0.12, &h())
                .unwrap()
                .potential_kcal_mol;
            let um = evaluate_softcore_pair(minus, lambda, -0.21, 3.4, 0.12, &h())
                .unwrap()
                .potential_kcal_mol;
            let numerical_force = -(up - um) / (2.0 * delta);
            assert!((numerical_force - e.force_on_i_kcal_mol_a[axis]).abs() < 2.0e-7);
        }
    }

    #[test]
    fn analytic_du_dlambda_matches_central_difference() {
        let displacement = [3.17, -0.43, 0.81];
        let lambda = 0.37;
        let e = evaluate_softcore_pair(displacement, lambda, -0.21, 3.4, 0.12, &h()).unwrap();
        let delta = 1.0e-6;
        let up = evaluate_softcore_pair(displacement, lambda + delta, -0.21, 3.4, 0.12, &h())
            .unwrap()
            .potential_kcal_mol;
        let um = evaluate_softcore_pair(displacement, lambda - delta, -0.21, 3.4, 0.12, &h())
            .unwrap()
            .potential_kcal_mol;
        assert!(((up - um) / (2.0 * delta) - e.du_dlambda_kcal_mol).abs() < 2.0e-7);
    }

    #[test]
    fn endpoint_bond_angle_torsion_forces_match_coordinate_differences() {
        let plane = bonded_plane(1.42, 1.91, 0.37);
        let atom_ids = [1, 2, 3, 4];
        let positions = [
            [0.0, 0.0, 0.0],
            [1.47, 0.12, -0.05],
            [2.16, 1.29, 0.26],
            [2.91, 1.03, 1.41],
        ];
        let analytic = evaluate_endpoint_bonded_plane(&positions, &atom_ids, &plane, None).unwrap();
        let delta = 2.0e-6;
        for atom in 0..positions.len() {
            for axis in 0..3 {
                let mut plus = positions;
                let mut minus = positions;
                plus[atom][axis] += delta;
                minus[atom][axis] -= delta;
                let up = evaluate_endpoint_bonded_plane(&plus, &atom_ids, &plane, None)
                    .unwrap()
                    .potential_kcal_mol;
                let um = evaluate_endpoint_bonded_plane(&minus, &atom_ids, &plane, None)
                    .unwrap()
                    .potential_kcal_mol;
                let numerical_force = -(up - um) / (2.0 * delta);
                assert!(
                    (numerical_force - analytic.forces_kcal_mol_a[atom][axis]).abs() < 2.0e-5,
                    "atom={atom} axis={axis} analytic={} numerical={numerical_force}",
                    analytic.forces_kcal_mol_a[atom][axis]
                );
            }
        }
    }

    #[test]
    fn state_transition_bonded_endpoints_and_dudl_are_exact() {
        let state_a = chemical_state(0, 0, 0.3, bonded_plane(1.42, 1.91, 0.37));
        let state_b = chemical_state(1, 1, 1.1, bonded_plane(1.48, 2.03, -0.24));
        let transition = transition(&state_a, &state_b);
        let atom_ids = [1, 2, 3, 4];
        let positions = [
            [0.0, 0.0, 0.0],
            [1.47, 0.12, -0.05],
            [2.16, 1.29, 0.26],
            [2.91, 1.03, 1.41],
        ];
        let endpoint_a = evaluate_endpoint_bonded_plane(
            &positions,
            &atom_ids,
            &state_a.bonded_parameter_plane,
            None,
        )
        .unwrap();
        let endpoint_b = evaluate_endpoint_bonded_plane(
            &positions,
            &atom_ids,
            &state_b.bonded_parameter_plane,
            None,
        )
        .unwrap();
        let at_zero = evaluate_state_transition_bonded_hamiltonian(
            &positions,
            &atom_ids,
            0.0,
            &state_a,
            &state_b,
            &transition,
            None,
        )
        .unwrap();
        let at_one = evaluate_state_transition_bonded_hamiltonian(
            &positions,
            &atom_ids,
            1.0,
            &state_a,
            &state_b,
            &transition,
            None,
        )
        .unwrap();
        assert_eq!(
            at_zero.potential_kcal_mol.to_bits(),
            endpoint_a.potential_kcal_mol.to_bits()
        );
        assert_eq!(
            at_one.potential_kcal_mol.to_bits(),
            endpoint_b.potential_kcal_mol.to_bits()
        );
        let lambda = 0.41;
        let delta = 1.0e-7;
        let value = evaluate_state_transition_bonded_hamiltonian(
            &positions,
            &atom_ids,
            lambda,
            &state_a,
            &state_b,
            &transition,
            None,
        )
        .unwrap();
        let up = evaluate_state_transition_bonded_hamiltonian(
            &positions,
            &atom_ids,
            lambda + delta,
            &state_a,
            &state_b,
            &transition,
            None,
        )
        .unwrap()
        .potential_kcal_mol;
        let um = evaluate_state_transition_bonded_hamiltonian(
            &positions,
            &atom_ids,
            lambda - delta,
            &state_a,
            &state_b,
            &transition,
            None,
        )
        .unwrap()
        .potential_kcal_mol;
        assert!(((up - um) / (2.0 * delta) - value.du_dlambda_kcal_mol).abs() < 1.0e-8);
        assert_eq!(value.reference_free_energy_b_minus_a_kcal_mol, 0.8);
        assert_eq!(value.delta_bound_protons_b_minus_a, 1);
    }

    #[test]
    fn dual_endpoint_pair_is_endpoint_exact_and_dummy_softcore_is_differentiable() {
        let a = FragmentPairEndpointParameters {
            present: true,
            charge_product_e2: -0.11,
            sigma_a: 3.1,
            epsilon_kcal_mol: 0.08,
        };
        let b = FragmentPairEndpointParameters {
            present: true,
            charge_product_e2: 0.07,
            sigma_a: 3.4,
            epsilon_kcal_mol: 0.13,
        };
        let displacement = [3.7, -0.2, 0.5];
        let endpoint_a = evaluate_softcore_pair(
            displacement,
            1.0,
            a.charge_product_e2,
            a.sigma_a,
            a.epsilon_kcal_mol,
            &h(),
        )
        .unwrap();
        let endpoint_b = evaluate_softcore_pair(
            displacement,
            1.0,
            b.charge_product_e2,
            b.sigma_a,
            b.epsilon_kcal_mol,
            &h(),
        )
        .unwrap();
        assert_eq!(
            evaluate_dual_endpoint_pair(displacement, 0.0, a, b, &h())
                .unwrap()
                .potential_kcal_mol
                .to_bits(),
            endpoint_a.potential_kcal_mol.to_bits()
        );
        assert_eq!(
            evaluate_dual_endpoint_pair(displacement, 1.0, a, b, &h())
                .unwrap()
                .potential_kcal_mol
                .to_bits(),
            endpoint_b.potential_kcal_mol.to_bits()
        );
        let dummy = FragmentPairEndpointParameters {
            present: false,
            charge_product_e2: 0.0,
            sigma_a: 1.0,
            epsilon_kcal_mol: 0.0,
        };
        let lambda = 0.43;
        let delta = 1.0e-6;
        let value = evaluate_dual_endpoint_pair(displacement, lambda, dummy, b, &h()).unwrap();
        let up = evaluate_dual_endpoint_pair(displacement, lambda + delta, dummy, b, &h())
            .unwrap()
            .potential_kcal_mol;
        let um = evaluate_dual_endpoint_pair(displacement, lambda - delta, dummy, b, &h())
            .unwrap()
            .potential_kcal_mol;
        assert!(((up - um) / (2.0 * delta) - value.du_dlambda_kcal_mol).abs() < 2.0e-7);
    }

    #[test]
    fn dynamic_charge_ewald_realspace_force_and_dudl_are_analytic() {
        let displacement = [3.27, -0.41, 0.73];
        let beta = 0.31;
        let lambda = 0.38;
        let qi0 = -0.42;
        let qj0 = 0.31;
        let dqi = 0.17;
        let dqj = -0.09;
        let evaluate = |coordinates: [f64; 3], l: f64| {
            evaluate_dynamic_charge_ewald_realspace_pair(
                coordinates,
                qi0 + l * dqi,
                dqi,
                qj0 + l * dqj,
                dqj,
                beta,
            )
            .unwrap()
        };
        let analytic = evaluate(displacement, lambda);
        let delta = 1.0e-6;
        let up = evaluate(displacement, lambda + delta).potential_kcal_mol;
        let um = evaluate(displacement, lambda - delta).potential_kcal_mol;
        assert!(((up - um) / (2.0 * delta) - analytic.du_dlambda_kcal_mol).abs() < 2.0e-8);
        for axis in 0..3 {
            let mut plus = displacement;
            let mut minus = displacement;
            plus[axis] += delta;
            minus[axis] -= delta;
            let numerical_force = -(evaluate(plus, lambda).potential_kcal_mol
                - evaluate(minus, lambda).potential_kcal_mol)
                / (2.0 * delta);
            assert!((numerical_force - analytic.force_on_i_kcal_mol_a[axis]).abs() < 2.0e-8);
        }
    }

    #[test]
    fn ewald_real_and_exception_corrections_close_exact_pair_targets() {
        let displacement = [4.1, 0.3, -0.7];
        let r = displacement
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        let beta = 0.29;
        let q_regular = -0.21;
        let dq_regular = 0.08;
        let q_exception = -0.07;
        let dq_exception = 0.03;
        let br = beta * r;
        let reciprocal_pair =
            COULOMB_KCAL_A_PER_MOL_E2 * q_regular * statrs::function::erf::erf(br) / r;
        let reciprocal_pair_dudl =
            COULOMB_KCAL_A_PER_MOL_E2 * dq_regular * statrs::function::erf::erf(br) / r;
        let ordinary = evaluate_dynamic_charge_ewald_realspace_pair(
            displacement,
            q_regular,
            dq_regular,
            1.0,
            0.0,
            beta,
        )
        .unwrap();
        assert!(
            (ordinary.potential_kcal_mol + reciprocal_pair
                - COULOMB_KCAL_A_PER_MOL_E2 * q_regular / r)
                .abs()
                < 1.0e-12
        );
        assert!(
            (ordinary.du_dlambda_kcal_mol + reciprocal_pair_dudl
                - COULOMB_KCAL_A_PER_MOL_E2 * dq_regular / r)
                .abs()
                < 1.0e-12
        );
        let correction = evaluate_dynamic_charge_ewald_exception_correction(
            displacement,
            q_regular,
            dq_regular,
            q_exception,
            dq_exception,
            beta,
            EwaldExceptionRealspaceComposition::OrdinaryPairOmitted,
        )
        .unwrap();
        assert!(
            (correction.potential_kcal_mol + reciprocal_pair
                - COULOMB_KCAL_A_PER_MOL_E2 * q_exception / r)
                .abs()
                < 1.0e-12
        );
        assert!(
            (correction.du_dlambda_kcal_mol + reciprocal_pair_dudl
                - COULOMB_KCAL_A_PER_MOL_E2 * dq_exception / r)
                .abs()
                < 1.0e-12
        );
        let excluded = evaluate_dynamic_charge_ewald_exception_correction(
            displacement,
            q_regular,
            dq_regular,
            0.0,
            0.0,
            beta,
            EwaldExceptionRealspaceComposition::OrdinaryPairOmitted,
        )
        .unwrap();
        assert!((excluded.potential_kcal_mol + reciprocal_pair).abs() < 1.0e-12);
        assert!((excluded.du_dlambda_kcal_mol + reciprocal_pair_dudl).abs() < 1.0e-12);
    }

    #[test]
    fn ph_and_intrinsic_reference_terms_close_forward_reverse_exactly_once() {
        let state_a = chemical_state(0, 0, 0.3, bonded_plane(1.42, 1.91, 0.37));
        let state_b = chemical_state(1, 1, 1.1, bonded_plane(1.48, 2.03, -0.24));
        let forward_operations = vec![
            NcmcProtocolOperation::Perturb {
                lambda_from: 0.0,
                lambda_to: 0.5,
            },
            NcmcProtocolOperation::Propagate {
                lambda: 0.5,
                integration_steps: 1,
            },
            NcmcProtocolOperation::Perturb {
                lambda_from: 0.5,
                lambda_to: 1.0,
            },
            NcmcProtocolOperation::Propagate {
                lambda: 1.0,
                integration_steps: 1,
            },
        ];
        let reverse_operations = forward_operations
            .iter()
            .rev()
            .map(NcmcProtocolOperation::reversed)
            .collect();
        let mut ncmc = NcmcProtocol {
            forward_operations,
            reverse_operations,
            forward_protocol_probability: 0.5,
            reverse_protocol_probability: 0.5,
            integrator_splitting: "BAOAB".into(),
            integrator_path_probability_convention:
                NcmcIntegratorPathProbabilityConvention::ShadowWorkVolumePreserving,
            momentum_reversal_policy: NcmcMomentumReversalPolicy::ReversePathAndRejection,
            initial_endpoint_equilibrium_density_evaluated: true,
            proposed_endpoint_equilibrium_density_evaluated: true,
            rng_algorithm: "PHILOX".into(),
            schedule_sha256: "0".repeat(64),
        };
        ncmc.schedule_sha256 = ncmc_schedule_sha256(&ncmc).unwrap();
        let system = LambdaFragmentSystem {
            schema_kind: "prismos_lambda_fragment_system".into(),
            schema_version: 2,
            experiment_id: "balance-test".into(),
            periodic_box: PeriodicBoxContract {
                vectors_a: [[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]],
                minimum_image_enabled: true,
                source_sha256: "a".repeat(64),
            },
            pme: PmeContract {
                reference_implementation: "OpenMM NonbondedForce PME".into(),
                boundary_condition: "ORTHORHOMBIC_TINFOIL_3D".into(),
                pme_interpolation_order: 5,
                relative_force_tolerance: 1.0e-5,
                real_space_cutoff_angstrom: 9.0,
                ewald_alpha_a_inv: (-(2.0e-5_f64).ln()).sqrt() / 9.0,
                ewald_alpha_derivation:
                    "sqrt(-ln(2*relative_force_tolerance))/real_space_cutoff_angstrom".into(),
                pme_grid_dimensions_lower_bound: [49, 49, 49],
                pme_grid_dimensions: [50, 50, 50],
                pme_grid_formula: "ceil(2*ewald_alpha*reduced_cell_diagonal/(3*relative_force_tolerance^(1/5)));min=6".into(),
                pme_grid_rounding_policy:
                    "PRISM_CUFFT_GOOD_SIZE_TABLE_V1_ROUND_UP_NO_DOWNROUND".into(),
                runtime_exact_parameter_lock_required: true,
                independent_openmm_force_energy_validation_required: true,
                pme_parameterization_sha256: "b".repeat(64),
            },
            water: WaterContract {
                model: "TIP3P".into(),
                settle_positions_enabled: true,
                settle_velocities_enabled: true,
                oh_distance_a: 0.9572,
                hoh_angle_rad: 1.824_218_134,
                parameterization_sha256: "c".repeat(64),
            },
            molecules: vec![MoleculeRecord {
                molecule_id: 1,
                role: MoleculeRole::LambdaFragment,
                atom_indices: vec![0, 1, 2, 3],
                water_triplet: None,
            }],
            species: vec![FragmentSpecies {
                species_id: 7,
                name: "reference-pair".into(),
                canonical_isomeric_smiles: "C".into(),
                inchikey: "TEST".into(),
                parameter_template_sha256: "d".repeat(64),
                chemical_states: vec![state_a.clone(), state_b.clone()],
                state_transitions: vec![transition(&state_a, &state_b)],
            }],
            slots: vec![FragmentSlot {
                slot_id: 1,
                molecule_id: 1,
                species_id: 7,
                chemical_state_id: 0,
                atom_indices: vec![0, 1, 2, 3],
                lambda: 0.0,
                endpoint_occupancy: EndpointOccupancy::Occupied,
                path_kind: AlchemicalPathKind::StateAToStateB,
                state_transition_id: Some("state-a-to-b".into()),
                endpoint_a: NcmcEndpointIdentity {
                    species_id: 7,
                    chemical_state_id: 0,
                    occupancy: EndpointOccupancy::Occupied,
                },
                endpoint_b: NcmcEndpointIdentity {
                    species_id: 7,
                    chemical_state_id: 1,
                    occupancy: EndpointOccupancy::Occupied,
                },
            }],
            softcore: h(),
            chemical_potentials: vec![
                ChemicalPotentialCalibration {
                    species_id: 7,
                    chemical_state_id: 0,
                    temperature_k: 300.0,
                    ph: 7.0,
                    standard_concentration_molar: 1.0,
                    excess_chemical_potential_kcal_mol: 1.7,
                    total_chemical_potential_kcal_mol: 2.0,
                    state_reference_convention:
                        StateReferenceConvention::TotalMuIncludesIntrinsicStateFreeEnergy,
                    included_intrinsic_state_free_energy_kcal_mol: 0.3,
                    periodic_charge_treatment: PeriodicChargeTreatment::NetChargeInvariant {
                        subject_charge_change_e: 0,
                    },
                    included_finite_size_correction_kcal_mol: 0.0,
                    method: "TEST_BAR".into(),
                    evidence_sha256: "e".repeat(64),
                    uncertainty_kcal_mol: 0.0,
                },
                ChemicalPotentialCalibration {
                    species_id: 7,
                    chemical_state_id: 1,
                    temperature_k: 300.0,
                    ph: 7.0,
                    standard_concentration_molar: 1.0,
                    excess_chemical_potential_kcal_mol: 2.4,
                    total_chemical_potential_kcal_mol: 3.5,
                    state_reference_convention:
                        StateReferenceConvention::TotalMuIncludesIntrinsicStateFreeEnergy,
                    included_intrinsic_state_free_energy_kcal_mol: 1.1,
                    periodic_charge_treatment: PeriodicChargeTreatment::NetChargeInvariant {
                        subject_charge_change_e: 0,
                    },
                    included_finite_size_correction_kcal_mol: 0.0,
                    method: "TEST_BAR".into(),
                    evidence_sha256: "f".repeat(64),
                    uncertainty_kcal_mol: 0.0,
                },
            ],
            ncmc,
        };
        let context = NcmcBalanceContext {
            temperature_k: 300.0,
            ph: 6.5,
            accessible_volume_a3: 1000.0,
            occupied_count_before: 1,
            occupied_count_after: 1,
            forward_selection_probability: 0.5,
            reverse_selection_probability: 0.5,
            forward_placement_density_a3_inv: 1.0,
            reverse_placement_density_a3_inv: 1.0,
            forward_orientation_density: 1.0,
            reverse_orientation_density: 1.0,
            initial_bias_weight: 1.0,
            proposed_bias_weight: 1.0,
            bias_epoch_sha256: "9".repeat(64),
        };
        let initial_a = NcmcEndpointIdentity {
            species_id: 7,
            chemical_state_id: 0,
            occupancy: EndpointOccupancy::Occupied,
        };
        let initial_b = NcmcEndpointIdentity {
            species_id: 7,
            chemical_state_id: 1,
            occupancy: EndpointOccupancy::Occupied,
        };
        let proposal = |move_id, proposed_state| NcmcMoveProposal {
            move_id,
            slot_id: 1,
            move_kind: NcmcMoveKind::Protonation,
            execution_mode: NcmcExecutionMode::CalibratedTargetLane,
            scheduler_contract_sha256: "6".repeat(64),
            selected_support_id: format!("test-protonation-{move_id}"),
            proposed_species_id: 7,
            proposed_chemical_state_id: proposed_state,
            proposed_occupancy: EndpointOccupancy::Occupied,
            balance_context: context.clone(),
            periodic_charge_treatment: PeriodicChargeTreatment::NetChargeInvariant {
                subject_charge_change_e: 0,
            },
            coupled_charge_partner: None,
            rng_key: 1,
            rng_counter: test_rng_allocation().acceptance_counter,
            rng_allocation: test_rng_allocation(),
        };
        let forward = derive_ncmc_balance(&system, &proposal(1, 1), &initial_a).unwrap();
        let reverse = derive_ncmc_balance(&system, &proposal(2, 0), &initial_b).unwrap();
        forward.validate_terms().unwrap();
        reverse.validate_terms().unwrap();
        for (a, b) in [
            (
                forward.beta_chemical_potential_term,
                reverse.beta_chemical_potential_term,
            ),
            (
                forward.beta_intrinsic_state_reference_term_in_total_mu,
                reverse.beta_intrinsic_state_reference_term_in_total_mu,
            ),
            (
                forward.beta_nonreference_chemical_potential_term,
                reverse.beta_nonreference_chemical_potential_term,
            ),
            (forward.ph_log_weight_term, reverse.ph_log_weight_term),
            (forward.log_target_ratio, reverse.log_target_ratio),
        ] {
            assert!((a + b).abs() < 1.0e-12, "forward={a} reverse={b}");
        }
        let beta = 1.0 / (0.001_987_204_258_640_83 * 300.0);
        assert!(
            (forward.beta_intrinsic_state_reference_term_in_total_mu - beta * 0.8).abs() < 1.0e-12
        );
        assert!((forward.ph_log_weight_term - 0.5 * 10.0f64.ln()).abs() < 1.0e-12);

        // Bulk BAR work collection is a distinct distribution: it must not
        // consume the chemical potential it is intended to estimate.
        let mut precalibration_system = system.clone();
        precalibration_system.chemical_potentials.clear();
        let precalibration_slot = &mut precalibration_system.slots[0];
        precalibration_slot.path_kind = AlchemicalPathKind::InsertDelete;
        precalibration_slot.state_transition_id = None;
        precalibration_slot.chemical_state_id = 0;
        precalibration_slot.endpoint_occupancy = EndpointOccupancy::Empty;
        precalibration_slot.endpoint_a = NcmcEndpointIdentity {
            species_id: 7,
            chemical_state_id: 0,
            occupancy: EndpointOccupancy::Empty,
        };
        precalibration_slot.endpoint_b = NcmcEndpointIdentity {
            species_id: 7,
            chemical_state_id: 0,
            occupancy: EndpointOccupancy::Occupied,
        };
        let mut precal_context = context.clone();
        precal_context.occupied_count_before = 0;
        precal_context.occupied_count_after = 1;
        let precal_proposal = NcmcMoveProposal {
            move_id: 3,
            slot_id: 1,
            move_kind: NcmcMoveKind::Insertion,
            execution_mode: NcmcExecutionMode::CalibrationWorkCollection {
                protocol_authority_sha256: "8".repeat(64),
            },
            scheduler_contract_sha256: "6".repeat(64),
            selected_support_id: "test-calibration-insertion".into(),
            proposed_species_id: 7,
            proposed_chemical_state_id: 0,
            proposed_occupancy: EndpointOccupancy::Occupied,
            balance_context: precal_context,
            periodic_charge_treatment: PeriodicChargeTreatment::NetChargeInvariant {
                subject_charge_change_e: 0,
            },
            coupled_charge_partner: None,
            rng_key: 3,
            rng_counter: test_rng_allocation().acceptance_counter,
            rng_allocation: test_rng_allocation(),
        };
        let empty = NcmcEndpointIdentity {
            species_id: 7,
            chemical_state_id: 0,
            occupancy: EndpointOccupancy::Empty,
        };
        let precal_balance =
            derive_ncmc_balance(&precalibration_system, &precal_proposal, &empty).unwrap();
        for value in [
            precal_balance.beta_chemical_potential_term,
            precal_balance.beta_intrinsic_state_reference_term_in_total_mu,
            precal_balance.beta_periodic_finite_size_correction_term_in_total_mu,
            precal_balance.ph_log_weight_term,
            precal_balance.standard_state_volume_term,
            precal_balance.occupancy_combinatorics_term,
            precal_balance.target_bias_term,
            precal_balance.log_target_ratio,
        ] {
            assert_eq!(value.to_bits(), 0.0f64.to_bits());
        }
        let precal_receipt = adjudicate_ncmc_move(
            &precalibration_system,
            &precal_proposal,
            &precal_balance,
            0,
            10,
            "7".repeat(64),
            1,
            "6".repeat(64),
            format!(
                "{:x}",
                Sha256::digest(serde_json::to_vec(&precal_proposal).unwrap())
            ),
            test_resident_dual_endpoint_union_contract_sha256(&precalibration_system),
            empty,
            0.0,
            NcmcIntegratorPathAccounting::ShadowWorkVolumePreserving {
                shadow_work_reduced: 0.0,
            },
            true,
            true,
            zero_work_trace_for_test(
                &precalibration_system.ncmc,
                precal_proposal.move_id,
                NcmcProtocolDirection::Forward,
                precal_proposal.slot_id,
            ),
            test_rng_consumption(&precal_proposal),
        )
        .unwrap();
        precal_receipt.validate().unwrap();

        let mut illegal_target = precal_proposal.clone();
        illegal_target.execution_mode = NcmcExecutionMode::CalibratedTargetLane;
        assert!(derive_ncmc_balance(
            &precalibration_system,
            &illegal_target,
            &NcmcEndpointIdentity {
                species_id: 7,
                chemical_state_id: 0,
                occupancy: EndpointOccupancy::Empty,
            }
        )
        .unwrap_err()
        .contains("no chemical-potential calibration"));
        assert!(derive_ncmc_balance(
            &system,
            &precal_proposal,
            &NcmcEndpointIdentity {
                species_id: 7,
                chemical_state_id: 0,
                occupancy: EndpointOccupancy::Empty,
            }
        )
        .unwrap_err()
        .contains("must not consume chemical-potential"));
    }

    #[test]
    fn endpoints_are_physical_and_intermediates_are_not_observations() {
        assert_eq!(
            observation_eligibility(0.0).unwrap(),
            ObservationEligibility::PhysicalEndpoint
        );
        assert_eq!(
            observation_eligibility(1.0).unwrap(),
            ObservationEligibility::PhysicalEndpoint
        );
        assert_eq!(
            observation_eligibility(0.5).unwrap(),
            ObservationEligibility::SwitchingOnly
        );
        assert_eq!(
            observation_eligibility(1.0e-13).unwrap(),
            ObservationEligibility::SwitchingOnly
        );
        assert_eq!(
            observation_eligibility(1.0 - 1.0e-13).unwrap(),
            ObservationEligibility::SwitchingOnly
        );
        let off = evaluate_softcore_pair([3.0, 0.0, 0.0], 0.0, -0.2, 3.0, 0.1, &h()).unwrap();
        assert_eq!(off.potential_kcal_mol, 0.0);
        assert_eq!(off.force_on_i_kcal_mol_a, [0.0; 3]);
        let physical = evaluate_softcore_pair([3.0, 0.0, 0.0], 1.0, 0.0, 3.0, 0.1, &h()).unwrap();
        let sr6 = 1.0f64;
        assert!((physical.lj_kcal_mol - 4.0 * 0.1 * (sr6 * sr6 - sr6)).abs() < 1e-14);
    }

    #[test]
    fn chemical_potential_total_decomposition_is_closed_for_neutral_and_charged_states() {
        let neutral = ChemicalPotentialCalibration {
            species_id: 1,
            chemical_state_id: 0,
            temperature_k: 300.0,
            ph: 7.0,
            standard_concentration_molar: 1.0,
            excess_chemical_potential_kcal_mol: -2.0,
            total_chemical_potential_kcal_mol: -1.7,
            state_reference_convention:
                StateReferenceConvention::TotalMuIncludesIntrinsicStateFreeEnergy,
            included_intrinsic_state_free_energy_kcal_mol: 0.3,
            periodic_charge_treatment: PeriodicChargeTreatment::NetChargeInvariant {
                subject_charge_change_e: 0,
            },
            included_finite_size_correction_kcal_mol: 0.0,
            method: "BAR_EXPLICIT_SOLVENT".into(),
            evidence_sha256: "1".repeat(64),
            uncertainty_kcal_mol: 0.05,
        };
        neutral.validate().unwrap();

        let mut charged = ChemicalPotentialCalibration {
            species_id: 2,
            chemical_state_id: 1,
            temperature_k: 300.0,
            ph: 7.0,
            standard_concentration_molar: 1.0,
            excess_chemical_potential_kcal_mol: -3.25,
            total_chemical_potential_kcal_mol: -2.65,
            state_reference_convention:
                StateReferenceConvention::TotalMuIncludesIntrinsicStateFreeEnergy,
            included_intrinsic_state_free_energy_kcal_mol: 0.4,
            periodic_charge_treatment: PeriodicChargeTreatment::ExplicitFiniteSizeCorrection {
                subject_charge_change_e: 1,
                correction_model: "ROCKLIN_NET_CHARGE_SOLVATION_V1".into(),
                correction_free_energy_kcal_mol: 0.2,
                correction_uncertainty_kcal_mol: 0.01,
                production_box_source_sha256: "2".repeat(64),
                pme_parameterization_sha256: "3".repeat(64),
                evidence_sha256: "4".repeat(64),
            },
            included_finite_size_correction_kcal_mol: 0.2,
            method: "MBAR_EXPLICIT_SOLVENT_CHARGED".into(),
            evidence_sha256: "5".repeat(64),
            uncertainty_kcal_mol: 0.08,
        };
        charged.validate().unwrap();
        charged.total_chemical_potential_kcal_mol += 0.1;
        assert!(charged
            .validate()
            .unwrap_err()
            .contains("total does not equal"));
        charged.total_chemical_potential_kcal_mol -= 0.1;
        charged.included_finite_size_correction_kcal_mol = 0.1;
        assert!(charged.validate().is_err());
    }

    #[test]
    fn ncmc_receipt_enforces_detailed_balance_bookkeeping() {
        let protocol = ncmc_test_protocol();
        let trace = zero_work_trace_for_test(&protocol, 7, NcmcProtocolDirection::Forward, 2);
        let raw: f64 = 0.4 - 0.2;
        let mut receipt = NcmcMoveReceipt {
            schema_kind: "prismos_lambda_fragment_ncmc_move_receipt".into(),
            schema_version: 6,
            stream_id: 0,
            step: 0,
            move_id: 7,
            event_sequence: 1,
            parent_receipt_sha256: "b".repeat(64),
            slot_id: 2,
            move_kind: NcmcMoveKind::Insertion,
            execution_mode: NcmcExecutionMode::CalibratedTargetLane,
            scheduler_contract_sha256: "6".repeat(64),
            selected_support_id: "test-receipt-insertion".into(),
            initial: NcmcEndpointIdentity {
                species_id: 1,
                chemical_state_id: 3,
                occupancy: EndpointOccupancy::Empty,
            },
            proposed: NcmcEndpointIdentity {
                species_id: 1,
                chemical_state_id: 3,
                occupancy: EndpointOccupancy::Occupied,
            },
            protocol_work_reduced: 0.0,
            ncmc_schedule_sha256: protocol.schedule_sha256.clone(),
            ncmc_protocol: protocol,
            protocol_direction: NcmcProtocolDirection::Forward,
            forward_protocol_probability: 0.5,
            reverse_protocol_probability: 0.5,
            log_protocol_probability_ratio: 0.0,
            integrator_path_accounting: NcmcIntegratorPathAccounting::ShadowWorkVolumePreserving {
                shadow_work_reduced: 0.0,
            },
            integrator_log_acceptance_contribution: 0.0,
            momentum_reversal_applied: true,
            endpoint_equilibrium_densities_evaluated: true,
            log_target_ratio: 0.4,
            log_proposal_ratio: -0.2,
            proposal_context: test_proposal_context(-0.2),
            proposal_measure_contract: NcmcProposalMeasureContract::canonical(),
            derived_balance: test_balance(0.4, -0.2),
            coupled_charge_partner: None,
            hamiltonian_sha256: h().digest_sha256().unwrap(),
            endpoint_support_binding_sha256: "e".repeat(64),
            dual_endpoint_union_contract_sha256: "f".repeat(64),
            periodic_box_source_sha256: "c".repeat(64),
            pme_parameterization_sha256: "d".repeat(64),
            log_acceptance: raw.min(0.0),
            uniform_random: counter_based_uniform_open01(
                19,
                test_rng_allocation().acceptance_counter,
            ),
            accepted: counter_based_uniform_open01(19, test_rng_allocation().acceptance_counter)
                .ln()
                < raw.min(0.0),
            rng_key: 19,
            rng_counter: test_rng_allocation().acceptance_counter,
            rng_allocation: test_rng_allocation(),
            rng_consumption: test_rng_consumption_raw(
                19,
                7,
                "6".repeat(64),
                NcmcMoveKind::Insertion,
            ),
            work_trace_sha256: trace.trace_sha256.clone(),
            ordered_work_trace: trace,
            phase_protocol_sha256: "c".repeat(64),
            prismcode_opcode: prismcode_opcodes::NCMC_ACCEPT.into(),
            prismcode_opcode_id: opcode_id_for_name(prismcode_opcodes::NCMC_ACCEPT).unwrap(),
            qnanl_artifact_family: QnanlArtifactFamily::LambdaFragmentNcmcMoveReceipt
                .as_str()
                .into(),
        };
        receipt.validate().unwrap();
        assert_ne!(receipt.proposal_context.forward_selection_probability, 1.0);
        assert_ne!(receipt.proposal_context.reverse_selection_probability, 1.0);
        assert!(
            ((receipt.proposal_context.reverse_selection_probability
                / receipt.proposal_context.forward_selection_probability)
                .ln()
                - receipt.derived_balance.selection_proposal_term)
                .abs()
                < 1.0e-15
        );
        let persisted = serde_json::to_value(&receipt).unwrap();
        assert_eq!(persisted["schema_version"], 6);
        assert!(persisted["rng_counter"].is_string());
        assert!(persisted["rng_allocation"]["selection_counter"].is_string());
        assert!(persisted["rng_consumption"]["allocation"]["acceptance_counter"].is_string());
        let mut legacy_v2 = receipt.clone();
        legacy_v2.schema_version = 2;
        assert!(legacy_v2.validate().is_err());
        let mut tampered_context = receipt.clone();
        tampered_context
            .proposal_context
            .reverse_selection_probability *= 0.5;
        assert!(tampered_context
            .validate()
            .unwrap_err()
            .contains("raw proposal context"));
        let mut tampered_scalar = receipt.clone();
        tampered_scalar.derived_balance.selection_proposal_term -= 0.125;
        assert!(tampered_scalar.validate().is_err());
        receipt.accepted = !receipt.accepted;
        assert!(receipt.validate().unwrap_err().contains("decision"));
    }

    #[test]
    fn forward_reverse_balance_residual_closes_for_every_move_class() {
        for kind in [
            NcmcMoveKind::Insertion,
            NcmcMoveKind::Deletion,
            NcmcMoveKind::Identity,
            NcmcMoveKind::Tautomer,
            NcmcMoveKind::Protonation,
        ] {
            let work: f64 = 0.37;
            let shadow: f64 = -0.04;
            let target: f64 = 0.23;
            let proposal: f64 = -0.11;
            let forward_log_ratio = -(work + shadow) + target + proposal;
            let reverse_log_ratio = -(-work - shadow) - target - proposal;
            assert!(
                (forward_log_ratio + reverse_log_ratio).abs() < 1.0e-15,
                "{kind:?}"
            );
        }
    }

    #[test]
    fn enumerable_two_slot_ncmc_matches_analytic_stationary_measure_without_double_counting() {
        const M: usize = 2;
        let activity = 1.7f64;
        let insertion_weight = [0.3f64, 0.7];
        let deletion_weight = [0.6f64, 0.4];
        let state_count = 1usize << M;
        let unnormalized = (0..state_count)
            .map(|state| activity.powi((state as u32).count_ones() as i32))
            .collect::<Vec<_>>();
        let normalization = unnormalized.iter().sum::<f64>();
        let stationary = unnormalized
            .iter()
            .map(|weight| weight / normalization)
            .collect::<Vec<_>>();
        let proposal_probability = |state: usize, slot: usize| {
            let mut total = 0.0;
            for candidate in 0..M {
                total += if state & (1 << candidate) == 0 {
                    insertion_weight[candidate]
                } else {
                    deletion_weight[candidate]
                };
            }
            let weight = if state & (1 << slot) == 0 {
                insertion_weight[slot]
            } else {
                deletion_weight[slot]
            };
            weight / total
        };
        let mut transition = vec![vec![0.0f64; state_count]; state_count];
        let mut wrong_transition = transition.clone();
        for state in 0..state_count {
            let mut row_sum = 0.0;
            let mut wrong_row_sum = 0.0;
            for slot in 0..M {
                let proposed = state ^ (1 << slot);
                let q_forward = proposal_probability(state, slot);
                let q_reverse = proposal_probability(proposed, slot);
                let log_target = (unnormalized[proposed] / unnormalized[state]).ln();
                let log_proposal = (q_reverse / q_forward).ln();
                let acceptance = (log_target + log_proposal).min(0.0).exp();
                transition[state][proposed] = q_forward * acceptance;
                row_sum += transition[state][proposed];

                let occupied_before = (state as u32).count_ones() as f64;
                let occupied_after = (proposed as u32).count_ones() as f64;
                let duplicated_combinatoric = if occupied_after > occupied_before {
                    ((M as f64 - occupied_before) / occupied_after).ln()
                } else {
                    (occupied_before / (M as f64 - occupied_after)).ln()
                };
                let wrong_acceptance = (log_target + log_proposal + duplicated_combinatoric)
                    .min(0.0)
                    .exp();
                wrong_transition[state][proposed] = q_forward * wrong_acceptance;
                wrong_row_sum += wrong_transition[state][proposed];
            }
            transition[state][state] = 1.0 - row_sum;
            wrong_transition[state][state] = 1.0 - wrong_row_sum;
        }
        let mut max_residual = 0.0f64;
        let mut wrong_max_residual = 0.0f64;
        for x in 0..state_count {
            assert!((transition[x].iter().sum::<f64>() - 1.0).abs() < 1.0e-14);
            for y in 0..state_count {
                max_residual = max_residual.max(
                    (stationary[x] * transition[x][y] - stationary[y] * transition[y][x]).abs(),
                );
                wrong_max_residual = wrong_max_residual.max(
                    (stationary[x] * wrong_transition[x][y]
                        - stationary[y] * wrong_transition[y][x])
                        .abs(),
                );
            }
        }
        assert!(max_residual < 1.0e-14);
        assert!(wrong_max_residual > 1.0e-4);
        let propagated = (0..state_count)
            .map(|destination| {
                (0..state_count)
                    .map(|source| stationary[source] * transition[source][destination])
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        for state in 0..state_count {
            assert!((propagated[state] - stationary[state]).abs() < 1.0e-14);
        }
        let aggregated = [stationary[0], stationary[1] + stationary[2], stationary[3]];
        let analytic = [1.0, 2.0 * activity, activity * activity]
            .map(|weight| weight / (1.0 + activity).powi(2));
        for occupancy in 0..=M {
            assert!((aggregated[occupancy] - analytic[occupancy]).abs() < 1.0e-14);
        }
        let pass_receipt = serde_json::json!({
            "schema_kind": "prismos_enumerable_fixed_slot_ncmc_detailed_balance_receipt",
            "schema_version": 1,
            "slot_count": M,
            "target_measure": "PI(mask) proportional to activity^popcount(mask); persistent slots are labeled and exact q_reverse/q_forward carries eligible-support asymmetry",
            "insertion_weights": insertion_weight,
            "deletion_weights": deletion_weight,
            "max_pairwise_detailed_balance_residual": max_residual,
            "double_counting_control_residual": wrong_max_residual,
            "status": "PASS"
        });
        let receipt_sha256 = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&pass_receipt).unwrap())
        );
        validate_sha256("enumerable NCMC PASS receipt", &receipt_sha256).unwrap();
    }

    #[test]
    fn triclinic_minimum_image_is_lattice_translation_invariant() {
        let periodic_box = PeriodicBoxContract {
            vectors_a: [[20.0, 0.0, 0.0], [3.0, 18.0, 0.0], [1.0, 2.0, 16.0]],
            minimum_image_enabled: true,
            source_sha256: "b".repeat(64),
        };
        let d = [2.1, -4.2, 5.7];
        let translated = [
            d[0] + periodic_box.vectors_a[0][0] - 2.0 * periodic_box.vectors_a[2][0],
            d[1] + periodic_box.vectors_a[0][1] - 2.0 * periodic_box.vectors_a[2][1],
            d[2] + periodic_box.vectors_a[0][2] - 2.0 * periodic_box.vectors_a[2][2],
        ];
        let a = minimum_image_displacement(d, &periodic_box).unwrap();
        let b = minimum_image_displacement(translated, &periodic_box).unwrap();
        for axis in 0..3 {
            assert!((a[axis] - b[axis]).abs() < 1.0e-12);
        }
    }

    #[test]
    fn lambda_one_matches_unsmoothed_lj_coulomb_endpoint() {
        let displacement = [3.2, 0.4, -0.2];
        let charge_product = -0.18;
        let sigma: f64 = 3.1;
        let epsilon = 0.13;
        let result =
            evaluate_softcore_pair(displacement, 1.0, charge_product, sigma, epsilon, &h())
                .unwrap();
        let r2 = displacement.iter().map(|x| x * x).sum::<f64>();
        let r = r2.sqrt();
        let sr6 = (sigma / r).powi(6);
        let expected =
            4.0 * epsilon * (sr6 * sr6 - sr6) + COULOMB_KCAL_A_PER_MOL_E2 * charge_product / r;
        assert!((result.potential_kcal_mol - expected).abs() < 1.0e-12);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn cuda_energy_force_and_dudl_match_cpu_reference() {
        use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
        use cudarc::nvrtc::Ptx;

        let context = CudaContext::new(0).expect("CUDA context");
        let stream = context.default_stream();
        let ptx_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/ptx/nhs_amber_fused.ptx")
            .canonicalize()
            .expect("canonical rebuilt fused PTX path");
        let module = context
            .load_module(Ptx::from_file(ptx_path))
            .expect("load rebuilt fused PTX");
        let function = module
            .load_function("nhs_lambda_fragment_pair_reference")
            .expect("lambda validation kernel");
        let mut d_potential = stream.alloc_zeros::<f64>(1).unwrap();
        let mut d_force = stream.alloc_zeros::<f32>(3).unwrap();
        let mut d_dudl = stream.alloc_zeros::<f64>(1).unwrap();
        let displacement = [3.17f32, -0.43, 0.81];
        let lambda = 0.37f32;
        let charge_product = -0.21f32;
        let sigma = 3.4f32;
        let epsilon = 0.12f32;
        let params = h();
        unsafe {
            stream
                .launch_builder(&function)
                .arg(&displacement[0])
                .arg(&displacement[1])
                .arg(&displacement[2])
                .arg(&lambda)
                .arg(&charge_product)
                .arg(&sigma)
                .arg(&epsilon)
                .arg(&(params.alpha_lj as f32))
                .arg(&(params.alpha_coulomb_a2 as f32))
                .arg(&(params.softcore_power as f32))
                .arg(&(params.coupling_power as f32))
                .arg(&(params.cutoff_a as f32))
                .arg(&mut d_potential)
                .arg(&mut d_force)
                .arg(&mut d_dudl)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
                .unwrap();
        }
        stream.synchronize().unwrap();
        let mut gpu_potential = [0.0f64; 1];
        let mut gpu_force = [0.0f32; 3];
        let mut gpu_dudl = [0.0f64; 1];
        stream
            .memcpy_dtoh(&d_potential, &mut gpu_potential)
            .unwrap();
        stream.memcpy_dtoh(&d_force, &mut gpu_force).unwrap();
        stream.memcpy_dtoh(&d_dudl, &mut gpu_dudl).unwrap();
        let cpu = evaluate_softcore_pair(
            displacement.map(f64::from),
            f64::from(lambda),
            f64::from(charge_product),
            f64::from(sigma),
            f64::from(epsilon),
            &params,
        )
        .unwrap();
        let potential_abs_error = (gpu_potential[0] - cpu.potential_kcal_mol).abs();
        let potential_relative_error =
            potential_abs_error / cpu.potential_kcal_mol.abs().max(f64::MIN_POSITIVE);
        assert!(
            potential_abs_error < 5.0e-6 && potential_relative_error < 5.0e-7,
            "GPU U={} CPU U={} abs_error={} relative_error={}",
            gpu_potential[0],
            cpu.potential_kcal_mol,
            potential_abs_error,
            potential_relative_error
        );
        assert!(
            (gpu_dudl[0] - cpu.du_dlambda_kcal_mol).abs() < 2.0e-5,
            "GPU dUdl={} CPU dUdl={}",
            gpu_dudl[0],
            cpu.du_dlambda_kcal_mol
        );
        for axis in 0..3 {
            assert!((f64::from(gpu_force[axis]) - cpu.force_on_i_kcal_mol_a[axis]).abs() < 2.0e-5);
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn cuda_v2_ewald_realspace_and_exception_match_cpu_reference() {
        use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
        use cudarc::nvrtc::Ptx;

        let context = CudaContext::new(0).expect("CUDA context");
        let stream = context.default_stream();
        let ptx_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/ptx/nhs_amber_fused.ptx")
            .canonicalize()
            .expect("canonical rebuilt fused PTX path");
        let module = context
            .load_module(Ptx::from_file(ptx_path))
            .expect("load rebuilt fused PTX");
        let function = module
            .load_function("nhs_lambda_fragment_ewald_pair_reference_v2")
            .expect("V2 Ewald validation kernel");

        let displacement = [23.17f32, -0.43, 0.81];
        let box_lengths = [40.0f32, 40.0, 40.0];
        let qi = 0.41f32;
        let dqi = -0.13f32;
        let qj = -0.37f32;
        let dqj = 0.09f32;
        let exception_q = -0.052f32;
        let d_exception_q = 0.017f32;
        let beta = 0.31f32;

        for mode in [0i32, 1i32] {
            let mut d_potential = stream.alloc_zeros::<f64>(1).unwrap();
            let mut d_force = stream.alloc_zeros::<f32>(3).unwrap();
            let mut d_dudl = stream.alloc_zeros::<f64>(1).unwrap();
            unsafe {
                stream
                    .launch_builder(&function)
                    .arg(&displacement[0])
                    .arg(&displacement[1])
                    .arg(&displacement[2])
                    .arg(&box_lengths[0])
                    .arg(&box_lengths[1])
                    .arg(&box_lengths[2])
                    .arg(&qi)
                    .arg(&dqi)
                    .arg(&qj)
                    .arg(&dqj)
                    .arg(&exception_q)
                    .arg(&d_exception_q)
                    .arg(&beta)
                    .arg(&mode)
                    .arg(&mut d_potential)
                    .arg(&mut d_force)
                    .arg(&mut d_dudl)
                    .launch(LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
            stream.synchronize().unwrap();
            let mut gpu_potential = [0.0f64; 1];
            let mut gpu_force = [0.0f32; 3];
            let mut gpu_dudl = [0.0f64; 1];
            stream
                .memcpy_dtoh(&d_potential, &mut gpu_potential)
                .unwrap();
            stream.memcpy_dtoh(&d_force, &mut gpu_force).unwrap();
            stream.memcpy_dtoh(&d_dudl, &mut gpu_dudl).unwrap();

            let minimum_image = [-16.83f64, -0.43, 0.81];
            let cpu = if mode == 0 {
                evaluate_dynamic_charge_ewald_realspace_pair(
                    minimum_image,
                    f64::from(qi),
                    f64::from(dqi),
                    f64::from(qj),
                    f64::from(dqj),
                    f64::from(beta),
                )
                .unwrap()
            } else {
                evaluate_dynamic_charge_ewald_exception_correction(
                    minimum_image,
                    f64::from(qi * qj),
                    f64::from(dqi * qj + qi * dqj),
                    f64::from(exception_q),
                    f64::from(d_exception_q),
                    f64::from(beta),
                    EwaldExceptionRealspaceComposition::OrdinaryPairOmitted,
                )
                .unwrap()
            };
            assert!((gpu_potential[0] - cpu.potential_kcal_mol).abs() < 5.0e-6);
            assert!((gpu_dudl[0] - cpu.du_dlambda_kcal_mol).abs() < 5.0e-6);
            for axis in 0..3 {
                assert!(
                    (f64::from(gpu_force[axis]) - cpu.force_on_i_kcal_mol_a[axis]).abs() < 5.0e-6
                );
            }
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn cuda_v2_softcore_lj_matches_cpu_reference() {
        use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
        use cudarc::nvrtc::Ptx;

        let context = CudaContext::new(0).expect("CUDA context");
        let stream = context.default_stream();
        let ptx_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/ptx/nhs_amber_fused.ptx")
            .canonicalize()
            .expect("canonical rebuilt fused PTX path");
        let module = context
            .load_module(Ptx::from_file(ptx_path))
            .expect("load rebuilt fused PTX");
        let function = module
            .load_function("nhs_lambda_fragment_softcore_lj_reference_v2")
            .expect("V2 softcore-LJ validation kernel");
        let displacement = [3.17f32, -0.43, 0.81];
        let lambda = 0.37f32;
        let sigma = 3.4f32;
        let epsilon = 0.12f32;
        let mut params = h();
        params.formula = "PRISM_LAMBDA_SOFTCORE_LJ_EWALD_CHARGE_V2".into();
        let mut d_potential = stream.alloc_zeros::<f64>(1).unwrap();
        let mut d_force = stream.alloc_zeros::<f32>(3).unwrap();
        let mut d_dudl = stream.alloc_zeros::<f64>(1).unwrap();
        unsafe {
            stream
                .launch_builder(&function)
                .arg(&displacement[0])
                .arg(&displacement[1])
                .arg(&displacement[2])
                .arg(&lambda)
                .arg(&sigma)
                .arg(&epsilon)
                .arg(&(params.alpha_lj as f32))
                .arg(&(params.softcore_power as f32))
                .arg(&(params.coupling_power as f32))
                .arg(&(params.cutoff_a as f32))
                .arg(&mut d_potential)
                .arg(&mut d_force)
                .arg(&mut d_dudl)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
                .unwrap();
        }
        stream.synchronize().unwrap();
        let mut gpu_potential = [0.0f64; 1];
        let mut gpu_force = [0.0f32; 3];
        let mut gpu_dudl = [0.0f64; 1];
        stream
            .memcpy_dtoh(&d_potential, &mut gpu_potential)
            .unwrap();
        stream.memcpy_dtoh(&d_force, &mut gpu_force).unwrap();
        stream.memcpy_dtoh(&d_dudl, &mut gpu_dudl).unwrap();
        let cpu = evaluate_softcore_pair(
            displacement.map(f64::from),
            f64::from(lambda),
            0.0,
            f64::from(sigma),
            f64::from(epsilon),
            &params,
        )
        .unwrap();
        assert!((gpu_potential[0] - cpu.potential_kcal_mol).abs() < 5.0e-6);
        assert!((gpu_dudl[0] - cpu.du_dlambda_kcal_mol).abs() < 2.0e-5);
        for axis in 0..3 {
            assert!((f64::from(gpu_force[axis]) - cpu.force_on_i_kcal_mol_a[axis]).abs() < 2.0e-5);
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn cuda_cross_slot_two_partial_lj_is_energy_force_and_per_slot_dudl_exact() {
        use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
        use cudarc::nvrtc::Ptx;

        #[derive(Clone, Copy)]
        struct Endpoint {
            present: bool,
            sigma: f32,
            epsilon: f32,
        }

        let context = CudaContext::new(0).expect("CUDA context");
        let stream = context.default_stream();
        let ptx_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/ptx/nhs_amber_fused.ptx")
            .canonicalize()
            .expect("canonical rebuilt fused PTX path");
        let module = context
            .load_module(Ptx::from_file(ptx_path))
            .expect("load rebuilt fused PTX");
        let function = module
            .load_function("nhs_lambda_dual_cross_slot_lj_reference_v4")
            .expect("cross-slot endpoint-aware LJ validation kernel");
        let params = h();
        let run_gpu = |displacement: [f32; 3],
                       lambda_i: f32,
                       ia: Endpoint,
                       ib: Endpoint,
                       lambda_j: f32,
                       ja: Endpoint,
                       jb: Endpoint| {
            let mut d_potential = stream.alloc_zeros::<f64>(1).unwrap();
            let mut d_force = stream.alloc_zeros::<f32>(3).unwrap();
            let mut d_dudl_i = stream.alloc_zeros::<f64>(1).unwrap();
            let mut d_dudl_j = stream.alloc_zeros::<f64>(1).unwrap();
            let present_ia = i32::from(ia.present);
            let present_ib = i32::from(ib.present);
            let present_ja = i32::from(ja.present);
            let present_jb = i32::from(jb.present);
            unsafe {
                stream
                    .launch_builder(&function)
                    .arg(&displacement[0])
                    .arg(&displacement[1])
                    .arg(&displacement[2])
                    .arg(&lambda_i)
                    .arg(&present_ia)
                    .arg(&ia.sigma)
                    .arg(&ia.epsilon)
                    .arg(&present_ib)
                    .arg(&ib.sigma)
                    .arg(&ib.epsilon)
                    .arg(&lambda_j)
                    .arg(&present_ja)
                    .arg(&ja.sigma)
                    .arg(&ja.epsilon)
                    .arg(&present_jb)
                    .arg(&jb.sigma)
                    .arg(&jb.epsilon)
                    .arg(&(params.alpha_lj as f32))
                    .arg(&(params.softcore_power as f32))
                    .arg(&(params.coupling_power as f32))
                    .arg(&(params.cutoff_a as f32))
                    .arg(&mut d_potential)
                    .arg(&mut d_force)
                    .arg(&mut d_dudl_i)
                    .arg(&mut d_dudl_j)
                    .launch(LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .unwrap();
            }
            stream.synchronize().unwrap();
            let mut potential = [0.0f64; 1];
            let mut force = [0.0f32; 3];
            let mut dudl_i = [0.0f64; 1];
            let mut dudl_j = [0.0f64; 1];
            stream.memcpy_dtoh(&d_potential, &mut potential).unwrap();
            stream.memcpy_dtoh(&d_force, &mut force).unwrap();
            stream.memcpy_dtoh(&d_dudl_i, &mut dudl_i).unwrap();
            stream.memcpy_dtoh(&d_dudl_j, &mut dudl_j).unwrap();
            (potential[0], force, dudl_i[0], dudl_j[0])
        };
        let cpu = |displacement: [f32; 3],
                   lambda_i: f32,
                   ia: Endpoint,
                   ib: Endpoint,
                   lambda_j: f32,
                   ja: Endpoint,
                   jb: Endpoint| {
            let identity_i = ia.present && ib.present;
            let identity_j = ja.present && jb.present;
            let li = f64::from(lambda_i);
            let lj = f64::from(lambda_j);
            let ci = if identity_i {
                1.0
            } else if ib.present {
                li
            } else {
                1.0 - li
            };
            let cj = if identity_j {
                1.0
            } else if jb.present {
                lj
            } else {
                1.0 - lj
            };
            let dci = if identity_i {
                0.0
            } else if ib.present {
                1.0
            } else {
                -1.0
            };
            let dcj = if identity_j {
                0.0
            } else if jb.present {
                1.0
            } else {
                -1.0
            };
            let wi = if identity_i {
                [1.0 - li, li]
            } else {
                [f64::from(ia.present), f64::from(ib.present)]
            };
            let wj = if identity_j {
                [1.0 - lj, lj]
            } else {
                [f64::from(ja.present), f64::from(jb.present)]
            };
            let dwi = if identity_i { [-1.0, 1.0] } else { [0.0, 0.0] };
            let dwj = if identity_j { [-1.0, 1.0] } else { [0.0, 0.0] };
            let endpoints_i = [ia, ib];
            let endpoints_j = [ja, jb];
            let mut potential = 0.0;
            let mut force = [0.0; 3];
            let mut dudl_i = 0.0;
            let mut dudl_j = 0.0;
            for a in 0..2 {
                if !endpoints_i[a].present || wi[a] == 0.0 {
                    continue;
                }
                for b in 0..2 {
                    if !endpoints_j[b].present || wj[b] == 0.0 {
                        continue;
                    }
                    let component = evaluate_softcore_pair(
                        displacement.map(f64::from),
                        ci * cj,
                        0.0,
                        0.5 * (f64::from(endpoints_i[a].sigma) + f64::from(endpoints_j[b].sigma)),
                        (f64::from(endpoints_i[a].epsilon) * f64::from(endpoints_j[b].epsilon))
                            .sqrt(),
                        &params,
                    )
                    .unwrap();
                    let weight = wi[a] * wj[b];
                    potential += weight * component.potential_kcal_mol;
                    for axis in 0..3 {
                        force[axis] += weight * component.force_on_i_kcal_mol_a[axis];
                    }
                    dudl_i += dwi[a] * wj[b] * component.potential_kcal_mol
                        + weight * component.du_dlambda_kcal_mol * dci * cj;
                    dudl_j += wi[a] * dwj[b] * component.potential_kcal_mol
                        + weight * component.du_dlambda_kcal_mol * ci * dcj;
                }
            }
            (potential, force, dudl_i, dudl_j)
        };

        let absent = Endpoint {
            present: false,
            sigma: 1.0,
            epsilon: 0.0,
        };
        let scenarios = [
            (
                0.37f32,
                absent,
                Endpoint {
                    present: true,
                    sigma: 3.1,
                    epsilon: 0.08,
                },
                0.61f32,
                absent,
                Endpoint {
                    present: true,
                    sigma: 3.7,
                    epsilon: 0.14,
                },
            ),
            (
                0.43,
                Endpoint {
                    present: true,
                    sigma: 3.2,
                    epsilon: 0.09,
                },
                absent,
                0.58,
                absent,
                Endpoint {
                    present: true,
                    sigma: 3.6,
                    epsilon: 0.15,
                },
            ),
            (
                0.29,
                Endpoint {
                    present: true,
                    sigma: 2.9,
                    epsilon: 0.07,
                },
                Endpoint {
                    present: true,
                    sigma: 3.4,
                    epsilon: 0.11,
                },
                0.72,
                Endpoint {
                    present: true,
                    sigma: 3.0,
                    epsilon: 0.10,
                },
                Endpoint {
                    present: true,
                    sigma: 3.8,
                    epsilon: 0.16,
                },
            ),
        ];
        let displacement = [3.73f32, -0.47, 0.82];
        for (lambda_i, ia, ib, lambda_j, ja, jb) in scenarios {
            let gpu = run_gpu(displacement, lambda_i, ia, ib, lambda_j, ja, jb);
            let reference = cpu(displacement, lambda_i, ia, ib, lambda_j, ja, jb);
            assert!(
                (gpu.0 - reference.0).abs() < 8.0e-6,
                "cross-slot U mismatch"
            );
            assert!(
                (gpu.2 - reference.2).abs() < 4.0e-5,
                "slot-i dU/dlambda mismatch"
            );
            assert!(
                (gpu.3 - reference.3).abs() < 4.0e-5,
                "slot-j dU/dlambda mismatch"
            );
            for axis in 0..3 {
                assert!(
                    (f64::from(gpu.1[axis]) - reference.1[axis]).abs() < 3.0e-5,
                    "cross-slot force mismatch axis={axis}"
                );
            }
        }
    }

    #[test]
    fn typed_lambda_events_qnanl_roundtrip_with_ordered_lineage() {
        let output = std::env::temp_dir().join(format!(
            "prism_lambda_typed_events_{}_{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = std::fs::remove_dir_all(&output);
        let mut emitter = LambdaTypedEventEmitter::new_post_run_manual(&output);
        let force = LambdaFragmentForceEvent {
            schema_kind: "prismos_lambda_fragment_force_event".into(),
            schema_version: 3,
            stream_id: 3,
            step: 17,
            event_sequence: 0,
            parent_receipt_sha256: None,
            ncmc_boundary: None,
            slot_id: 9,
            species_id: 2,
            chemical_state_id: 4,
            lambda: 1.0,
            path_kind: AlchemicalPathKind::InsertDelete,
            endpoint_a: NcmcEndpointIdentity {
                species_id: 2,
                chemical_state_id: 4,
                occupancy: EndpointOccupancy::Empty,
            },
            endpoint_b: NcmcEndpointIdentity {
                species_id: 2,
                chemical_state_id: 4,
                occupancy: EndpointOccupancy::Occupied,
            },
            potential_kcal_mol: -1.7,
            du_dlambda_kcal_mol: 0.3,
            energy_unit: "kcal/mol".into(),
            force_unit: "kcal/mol/angstrom".into(),
            endpoint_parameter_sha256: "a".repeat(64),
            endpoint_support_binding_sha256: "e".repeat(64),
            dual_endpoint_union_contract_sha256: "f".repeat(64),
            hamiltonian_sha256: h().digest_sha256().unwrap(),
            phase_protocol_sha256: "d".repeat(64),
            prismcode_opcode: prismcode_opcodes::FORCE.into(),
            prismcode_opcode_id: opcode_id_for_name(prismcode_opcodes::FORCE).unwrap(),
            qnanl_artifact_family: QnanlArtifactFamily::LambdaFragmentForceEvent
                .as_str()
                .into(),
        };
        let force_receipt = emitter.emit_force(&force).unwrap();
        assert!(force_receipt.exact_byte_roundtrip);
        assert_eq!(force_receipt.runtime_origin, "POST_RUN_MANUAL");
        let fiber_units = force_event_dudl_for_fiber(&force).unwrap();
        assert_eq!(fiber_units.source_unit, "kcal/mol");
        assert_eq!(fiber_units.target_unit, "kJ/mol");
        assert_eq!(fiber_units.factor.to_bits(), 4.184f64.to_bits());
        assert!((fiber_units.du_dlambda_kj_mol - 1.2552).abs() < 1.0e-12);

        let protocol = ncmc_test_protocol();
        let trace = zero_work_trace_for_test(&protocol, 18, NcmcProtocolDirection::Forward, 9);
        let raw_log_ratio: f64 = 0.1 - 0.03;
        let ncmc = NcmcMoveReceipt {
            schema_kind: "prismos_lambda_fragment_ncmc_move_receipt".into(),
            schema_version: 6,
            stream_id: 3,
            step: 18,
            move_id: 18,
            event_sequence: 1,
            parent_receipt_sha256: force_receipt.receipt_sha256.clone(),
            slot_id: 9,
            move_kind: NcmcMoveKind::Tautomer,
            execution_mode: NcmcExecutionMode::CalibratedTargetLane,
            scheduler_contract_sha256: "6".repeat(64),
            selected_support_id: "test-fiber-tautomer".into(),
            initial: NcmcEndpointIdentity {
                species_id: 2,
                chemical_state_id: 3,
                occupancy: EndpointOccupancy::Occupied,
            },
            proposed: NcmcEndpointIdentity {
                species_id: 2,
                chemical_state_id: 4,
                occupancy: EndpointOccupancy::Occupied,
            },
            protocol_work_reduced: 0.0,
            ncmc_schedule_sha256: protocol.schedule_sha256.clone(),
            ncmc_protocol: protocol,
            protocol_direction: NcmcProtocolDirection::Forward,
            forward_protocol_probability: 0.5,
            reverse_protocol_probability: 0.5,
            log_protocol_probability_ratio: 0.0,
            integrator_path_accounting: NcmcIntegratorPathAccounting::ShadowWorkVolumePreserving {
                shadow_work_reduced: 0.0,
            },
            integrator_log_acceptance_contribution: 0.0,
            momentum_reversal_applied: true,
            endpoint_equilibrium_densities_evaluated: true,
            log_target_ratio: 0.1,
            log_proposal_ratio: -0.03,
            proposal_context: test_proposal_context(-0.03),
            proposal_measure_contract: NcmcProposalMeasureContract::canonical(),
            derived_balance: test_balance(0.1, -0.03),
            coupled_charge_partner: None,
            hamiltonian_sha256: h().digest_sha256().unwrap(),
            endpoint_support_binding_sha256: force.endpoint_support_binding_sha256.clone(),
            dual_endpoint_union_contract_sha256: force.dual_endpoint_union_contract_sha256.clone(),
            periodic_box_source_sha256: "c".repeat(64),
            pme_parameterization_sha256: "d".repeat(64),
            log_acceptance: raw_log_ratio.min(0.0),
            uniform_random: counter_based_uniform_open01(
                7,
                test_rng_allocation().acceptance_counter,
            ),
            accepted: counter_based_uniform_open01(7, test_rng_allocation().acceptance_counter)
                .ln()
                < raw_log_ratio.min(0.0),
            rng_key: 7,
            rng_counter: test_rng_allocation().acceptance_counter,
            rng_allocation: test_rng_allocation(),
            rng_consumption: test_rng_consumption_raw(
                7,
                18,
                "6".repeat(64),
                NcmcMoveKind::Tautomer,
            ),
            work_trace_sha256: trace.trace_sha256.clone(),
            ordered_work_trace: trace,
            phase_protocol_sha256: "d".repeat(64),
            prismcode_opcode: prismcode_opcodes::NCMC_ACCEPT.into(),
            prismcode_opcode_id: opcode_id_for_name(prismcode_opcodes::NCMC_ACCEPT).unwrap(),
            qnanl_artifact_family: QnanlArtifactFamily::LambdaFragmentNcmcMoveReceipt
                .as_str()
                .into(),
        };
        let ncmc_object_digest = ncmc.digest_sha256().unwrap();
        let ncmc_receipt = emitter.emit_ncmc(&ncmc).unwrap();

        let endpoint = LambdaFragmentEndpointObservation {
            schema_kind: "prismos_lambda_fragment_endpoint_observation".into(),
            schema_version: 2,
            stream_id: 3,
            step: 18,
            event_sequence: 2,
            parent_receipt_sha256: ncmc_receipt.receipt_sha256.clone(),
            supporting_force_event_sha256: force_receipt.raw_sha256.clone(),
            supporting_force_emission_receipt_sha256: force_receipt.receipt_sha256.clone(),
            supporting_force_boundary: NcmcForceBoundaryIdentity::new(
                18,
                NcmcProtocolDirection::Forward,
                0,
                NcmcForceBoundaryKind::PerturbPre,
            )
            .unwrap(),
            slot_id: 9,
            species_id: 2,
            chemical_state_id: 4,
            lambda_endpoint: 1,
            occupancy: EndpointOccupancy::Occupied,
            path_kind: AlchemicalPathKind::StateAToStateB,
            endpoint_a: NcmcEndpointIdentity {
                species_id: 2,
                chemical_state_id: 3,
                occupancy: EndpointOccupancy::Occupied,
            },
            endpoint_b: NcmcEndpointIdentity {
                species_id: 2,
                chemical_state_id: 4,
                occupancy: EndpointOccupancy::Occupied,
            },
            phase_protocol_sha256: "d".repeat(64),
            ghost_lineage_sha256: Some("e".repeat(64)),
            prismcode_opcode: prismcode_opcodes::ENDPOINT_OBSERVE.into(),
            prismcode_opcode_id: opcode_id_for_name(prismcode_opcodes::ENDPOINT_OBSERVE).unwrap(),
            qnanl_artifact_family: QnanlArtifactFamily::LambdaFragmentEndpointObservation
                .as_str()
                .into(),
        };
        let endpoint_receipt = emitter.emit_endpoint(&endpoint).unwrap();

        let commit = LambdaFragmentStateCommit {
            schema_kind: "prismos_lambda_fragment_state_commit".into(),
            schema_version: 1,
            stream_id: 3,
            step: 18,
            ncmc_move_receipt_sha256: ncmc_object_digest,
            event_sequence: 3,
            parent_receipt_sha256: endpoint_receipt.receipt_sha256.clone(),
            slot_id: 9,
            species_id: 2,
            chemical_state_id: 4,
            occupancy: EndpointOccupancy::Occupied,
            phase_protocol_sha256: "d".repeat(64),
            prismcode_opcode: prismcode_opcodes::STATE_COMMIT.into(),
            prismcode_opcode_id: opcode_id_for_name(prismcode_opcodes::STATE_COMMIT).unwrap(),
            qnanl_artifact_family: QnanlArtifactFamily::LambdaFragmentStateCommit
                .as_str()
                .into(),
        };
        let commit_receipt = emitter.emit_state_commit(&commit).unwrap();
        assert_eq!(commit_receipt.event_sequence, 3);
        assert_eq!(
            commit_receipt.parent_receipt_sha256,
            Some(endpoint_receipt.receipt_sha256)
        );
        for entry in std::fs::read_dir(&output).unwrap() {
            assert!(entry.unwrap().metadata().unwrap().len() > 0);
        }
        let _ = std::fs::remove_dir_all(&output);
    }

    #[test]
    fn endpoint_support_v2_exports_cross_language_fixture() {
        let system = precalibration_test_system();
        let scheduler = test_scheduler_contract(system.ncmc.schedule_sha256.clone());
        let expected_binding_sha256 =
            lambda_endpoint_support_binding_sha256(&system, &scheduler).unwrap();
        let fixture = serde_json::json!({
            "schema_kind": "prismos_lambda_endpoint_support_cross_language_fixture",
            "schema_version": 1,
            "binding_schema_kind": "prismos_lambda_endpoint_support_binding",
            "binding_schema_version": 2,
            "binding_domain": "PRISM_LAMBDA_ENDPOINT_SUPPORT_BINDING_V2\\0",
            "atom_index_set_domain": "PRISM_LAMBDA_SLOT_ATOM_INDEX_SET_V1\\0",
            "lambda_fragment_system": system,
            "ncmc_proposal_scheduler": scheduler,
            "expected_binding_sha256": expected_binding_sha256,
        });
        let decoded: serde_json::Value =
            serde_json::from_slice(&serde_json::to_vec(&fixture).unwrap()).unwrap();
        assert_eq!(decoded, fixture);
        if let Some(path) = std::env::var_os("PRISM_ENDPOINT_SUPPORT_FIXTURE_OUTPUT") {
            let path = PathBuf::from(path);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(path, serde_json::to_vec_pretty(&fixture).unwrap()).unwrap();
        }
    }

    #[test]
    fn promotion_v1_exports_cross_language_parity_fixture() {
        #[derive(Serialize)]
        struct SeedProjection<'a> {
            topology_sha256: &'a str,
            frozen_cell_authority_sha256: &'a str,
            periodic_cell_sha256: &'a str,
            replica_seed: u64,
        }

        #[derive(Serialize)]
        struct FiniteSizeProjection<'a> {
            species_id: u32,
            chemical_state_id: u32,
            periodic_charge_treatment: &'a PeriodicChargeTreatment,
            included_finite_size_correction_kcal_mol: f64,
            evidence_sha256: &'a str,
        }

        let mut system = precalibration_test_system();
        system.experiment_id = "rust-promotion-parity-v1".into();
        system.chemical_potentials = vec![ChemicalPotentialCalibration {
            species_id: 7,
            chemical_state_id: 0,
            temperature_k: 300.0,
            ph: 7.0,
            standard_concentration_molar: 1.0,
            excess_chemical_potential_kcal_mol: -1.25,
            total_chemical_potential_kcal_mol: -1.25,
            state_reference_convention:
                StateReferenceConvention::TotalMuIncludesIntrinsicStateFreeEnergy,
            included_intrinsic_state_free_energy_kcal_mol: 0.0,
            periodic_charge_treatment: PeriodicChargeTreatment::NetChargeInvariant {
                subject_charge_change_e: 0,
            },
            included_finite_size_correction_kcal_mol: 0.0,
            method: "BAR_EXPLICIT_SOLVENT_PARITY_FIXTURE".into(),
            evidence_sha256: "e".repeat(64),
            uncertainty_kcal_mol: 0.05,
        }];
        system.chemical_potentials[0].validate().unwrap();

        let system_json = serde_json::to_vec(&system).unwrap();
        let system_sha256 = format!("{:x}", Sha256::digest(&system_json));
        let chemical_potential_evidence_sha256 =
            lambda_chemical_potential_evidence_sha256(&system).unwrap();
        let finite_size_admission_sha256 = lambda_finite_size_admission_sha256(&system).unwrap();
        let chemical_potential_projection_json =
            serde_json::to_string(&system.chemical_potentials).unwrap();
        let finite_size_projection_json = serde_json::to_string(&vec![FiniteSizeProjection {
            species_id: system.chemical_potentials[0].species_id,
            chemical_state_id: system.chemical_potentials[0].chemical_state_id,
            periodic_charge_treatment: &system.chemical_potentials[0].periodic_charge_treatment,
            included_finite_size_correction_kcal_mol: system.chemical_potentials[0]
                .included_finite_size_correction_kcal_mol,
            evidence_sha256: &system.chemical_potentials[0].evidence_sha256,
        }])
        .unwrap();

        let topology_sha256 = "1".repeat(64);
        let frozen_cell_authority_sha256 = "2".repeat(64);
        let periodic_cell_sha256 = "3".repeat(64);
        let replica_seed = 18_446_744_073_709_551_557_u64;
        let seed_projection = SeedProjection {
            topology_sha256: &topology_sha256,
            frozen_cell_authority_sha256: &frozen_cell_authority_sha256,
            periodic_cell_sha256: &periodic_cell_sha256,
            replica_seed,
        };
        let seed_topology_member_sha256 = lambda_seed_topology_member_sha256(
            &topology_sha256,
            &frozen_cell_authority_sha256,
            &periodic_cell_sha256,
            replica_seed,
        )
        .unwrap();

        let precursor_placement_measure_sha256 = "4".repeat(64);
        let cross_cell_unlock_sha256 = "5".repeat(64);
        let panel_scientific_sha256 = "6".repeat(64);
        let target_protocol_authority_sha256 = "7".repeat(64);
        let promotion_projection = DecoupledFragmentPromotionReceiptProjection {
            precursor_placement_measure_sha256: &precursor_placement_measure_sha256,
            chemical_potential_evidence_sha256: &chemical_potential_evidence_sha256,
            cross_cell_unlock_sha256: &cross_cell_unlock_sha256,
            finite_size_admission_sha256: &finite_size_admission_sha256,
            seed_topology_member_sha256: &seed_topology_member_sha256,
            panel_scientific_sha256: &panel_scientific_sha256,
            target_protocol_authority_sha256: &target_protocol_authority_sha256,
        };
        let promotion_receipt_sha256 = decoupled_fragment_promotion_receipt_sha256(
            &precursor_placement_measure_sha256,
            &chemical_potential_evidence_sha256,
            &cross_cell_unlock_sha256,
            &finite_size_admission_sha256,
            &seed_topology_member_sha256,
            &panel_scientific_sha256,
            &target_protocol_authority_sha256,
        )
        .unwrap();

        let ryu_values = [
            0.0_f64,
            -0.0_f64,
            f64::from_bits(1),
            f64::MIN_POSITIVE,
            0.1_f64,
            1.0e-6_f64,
            1.0e21_f64,
            f64::from_bits(0x3fefffffffffffff),
            f64::from_bits(0x3ff0000000000001),
            f64::MAX,
        ];
        let ryu_float_cases = ryu_values
            .into_iter()
            .map(|value| {
                let spelling = serde_json::to_string(&value).unwrap();
                let mut digest = Sha256::new();
                digest.update(b"PRISM_RYU_F64_SPELLING_V1\0");
                digest.update(value.to_bits().to_le_bytes());
                digest.update(spelling.as_bytes());
                serde_json::json!({
                    "bits_hex": format!("{:016x}", value.to_bits()),
                    "serialized_json_number": spelling,
                    "spelling_sha256": format!("{:x}", digest.finalize()),
                })
            })
            .collect::<Vec<_>>();

        let fixture = serde_json::json!({
            "schema_kind": "prismos_lambda_promotion_cross_language_parity_fixture",
            "schema_version": 1,
            "domains": {
                "chemical_potential_evidence": "PRISM_LAMBDA_CHEMICAL_POTENTIAL_EVIDENCE_SET_V1\0",
                "finite_size_admission": "PRISM_LAMBDA_FINITE_SIZE_ADMISSION_SET_V1\0",
                "seed_topology_member": "PRISM_LAMBDA_SEED_TOPOLOGY_CELL_MEMBER_V1\0",
                "production_promotion_receipt": "PRISM_DECOUPLED_FRAGMENT_PRODUCTION_PROMOTION_RECEIPT_V1\0",
                "ryu_f64_spelling": "PRISM_RYU_F64_SPELLING_V1\0"
            },
            "lambda_fragment_system": system,
            "lambda_fragment_system_json_utf8": String::from_utf8(system_json).unwrap(),
            "expected_lambda_fragment_system_sha256": system_sha256,
            "chemical_potential_projection_json_utf8": chemical_potential_projection_json,
            "expected_chemical_potential_evidence_sha256": chemical_potential_evidence_sha256,
            "finite_size_projection_json_utf8": finite_size_projection_json,
            "expected_finite_size_admission_sha256": finite_size_admission_sha256,
            "seed_member": {
                "projection_field_order": [
                    "topology_sha256",
                    "frozen_cell_authority_sha256",
                    "periodic_cell_sha256",
                    "replica_seed"
                ],
                "projection": seed_projection,
                "projection_json_utf8": serde_json::to_string(&seed_projection).unwrap(),
                "expected_sha256": seed_topology_member_sha256
            },
            "promotion": {
                "projection_field_order": [
                    "precursor_placement_measure_sha256",
                    "chemical_potential_evidence_sha256",
                    "cross_cell_unlock_sha256",
                    "finite_size_admission_sha256",
                    "seed_topology_member_sha256",
                    "panel_scientific_sha256",
                    "target_protocol_authority_sha256"
                ],
                "projection": promotion_projection,
                "projection_json_utf8": serde_json::to_string(&promotion_projection).unwrap(),
                "expected_sha256": promotion_receipt_sha256
            },
            "ryu_float_cases": ryu_float_cases
        });
        let decoded: serde_json::Value =
            serde_json::from_slice(&serde_json::to_vec(&fixture).unwrap()).unwrap();
        assert_eq!(decoded, fixture);
        let committed: serde_json::Value = serde_json::from_str(include_str!(
            "../../../scripts/tests/fixtures/rust_lambda_promotion_parity_v1.json"
        ))
        .unwrap();
        assert_eq!(committed, fixture, "committed Rust parity fixture is stale");
        if let Some(path) = std::env::var_os("PRISM_PROMOTION_PARITY_FIXTURE_OUTPUT") {
            let path = PathBuf::from(path);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(path, serde_json::to_vec_pretty(&fixture).unwrap()).unwrap();
        }
    }

    #[test]
    fn rust_precalibration_v6_journal_exports_lossless_join_fixture() {
        let output = std::env::temp_dir().join(format!(
            "prism_lambda_precalibration_journal_{}_{}",
            std::process::id(),
            counter_based_uniform_open01(31, 41).to_bits()
        ));
        let _ = std::fs::remove_dir_all(&output);
        let system = precalibration_test_system();
        let mut lane = LambdaExperimentLaneContract {
            schema_kind: "prismos_lambda_experiment_lane_contract".into(),
            schema_version: 1,
            lane_id: "rust-precalibration-work".into(),
            lane_kind: LambdaExperimentLaneKind::HypothesisFragments,
            fragment_species_ids: vec![7],
            virtual_probe_count: 0,
            contract_sha256: String::new(),
        };
        lane.contract_sha256 = lane.recompute_sha256().unwrap();
        lane.validate(Some(&system)).unwrap();
        let mut scheduler = test_scheduler_contract(system.ncmc.schedule_sha256.clone());
        scheduler.scheduler_id = "CALIBRATION_WORK::rust-ncmc-v6-fixture".into();
        scheduler.contract_sha256 = scheduler.recompute_sha256().unwrap();
        scheduler.validate().unwrap();
        let execution_mode = NcmcExecutionMode::CalibrationWorkCollection {
            protocol_authority_sha256: scheduler.contract_sha256.clone(),
        };
        let phase = crate::chemical_state_fiber::ProtocolPhaseBinding {
            stream_id: 3,
            thermodynamic_condition_id: "bulk-calibration-300K".into(),
            phase_ordinal: 0,
            phase_name: "bulk-switching-work".into(),
            start_step: 0,
            end_step_exclusive: 100,
            protocol_name: "rust-ncmc-v6-fixture".into(),
            phase_schedule_digest_sha256: "1".repeat(64),
            protocol_state_hash: "2".repeat(64),
        };
        let condition = crate::chemical_state_fiber::ThermodynamicCondition {
            condition_id: "bulk-calibration-300K".into(),
            temperature_k: 300.0,
            pressure_bar: 1.0,
            ionic_strength_molar: 0.15,
            solvent_model: "TIP3P-FB".into(),
            periodic_box_vectors_a: system.periodic_box.vectors_a,
            pme_parameter_sha256: system.pme.pme_parameterization_sha256.clone(),
        };
        let matched_velocity_initialization = test_matched_velocity_contract(true);
        let endpoint_support_binding_sha256 =
            lambda_endpoint_support_binding_sha256(&system, &scheduler).unwrap();
        let dual_endpoint_union_contract_sha256 =
            test_resident_dual_endpoint_union_contract_sha256(&system);
        let placement_authority =
            test_placement_authority(&system, &scheduler, &matched_velocity_initialization, false);
        let binding = LambdaRuntimeBinding {
            run_id: "rust-ncmc-v6-calibration-fixture".into(),
            replica_id: "rep-00".into(),
            replica_seed: 42,
            stream_id: 3,
            stream_seed: 45,
            engine_sha256: "4".repeat(64),
            topology_sha256: "5".repeat(64),
            endpoint_support_binding_sha256: Some(endpoint_support_binding_sha256.clone()),
            runtime_capability_receipt: None,
            matched_velocity_initialization,
            decoupled_fragment_placement_measure_authority: Some(placement_authority),
            target_cross_cell_unlock_sha256: None,
            panel_scientific_sha256: Some("7".repeat(64)),
            lane,
            ncmc_execution_mode: Some(execution_mode.clone()),
            ncmc_scheduler: Some(scheduler.clone()),
            initial_protocol_phase: phase.clone(),
            initial_thermodynamic_condition: condition.clone(),
            protocol_schedule: vec![LambdaRuntimePhaseContext {
                protocol_phase: phase,
                thermodynamic_condition: condition,
            }],
            run_output_root: output.parent().unwrap().to_path_buf(),
            typed_event_output_dir: output.clone(),
            typed_event_capacity: test_capacity_contract(),
        };
        binding.validate_pre_topology().unwrap();
        binding.validate(Some(&system)).unwrap();
        let precursor_authority = binding
            .decoupled_fragment_placement_measure_authority
            .as_ref()
            .unwrap()
            .clone();
        let mut bool_only_promotion = precursor_authority.clone();
        bool_only_promotion.production_admission_allowed = true;
        bool_only_promotion.placement_measure_sha256 =
            bool_only_promotion.recompute_sha256().unwrap();
        assert!(bool_only_promotion.validate().is_err());

        let mut target_system = system.clone();
        target_system
            .chemical_potentials
            .push(ChemicalPotentialCalibration {
                species_id: 7,
                chemical_state_id: 0,
                temperature_k: 300.0,
                ph: 7.0,
                standard_concentration_molar: 1.0,
                excess_chemical_potential_kcal_mol: -1.0,
                total_chemical_potential_kcal_mol: -1.0,
                state_reference_convention:
                    StateReferenceConvention::TotalMuIncludesIntrinsicStateFreeEnergy,
                included_intrinsic_state_free_energy_kcal_mol: 0.0,
                periodic_charge_treatment: PeriodicChargeTreatment::NetChargeInvariant {
                    subject_charge_change_e: 0,
                },
                included_finite_size_correction_kcal_mol: 0.0,
                method: "TEST_BAR_EXPLICIT_SOLVENT".into(),
                evidence_sha256: "9".repeat(64),
                uncertainty_kcal_mol: 0.05,
            });
        let cross_cell_unlock_sha256 = "8".repeat(64);
        let panel_scientific_sha256 = binding.panel_scientific_sha256.as_deref().unwrap();
        let promoted_authority = promote_test_placement_authority(
            precursor_authority,
            &target_system,
            &scheduler,
            &binding.matched_velocity_initialization,
            &binding.topology_sha256,
            binding.replica_seed,
            &cross_cell_unlock_sha256,
            panel_scientific_sha256,
        );
        let mut target_binding = binding.clone();
        target_binding.ncmc_execution_mode = Some(NcmcExecutionMode::CalibratedTargetLane);
        target_binding.endpoint_support_binding_sha256 =
            Some(lambda_endpoint_support_binding_sha256(&target_system, &scheduler).unwrap());
        target_binding.target_cross_cell_unlock_sha256 = Some(cross_cell_unlock_sha256.clone());
        target_binding.decoupled_fragment_placement_measure_authority = Some(promoted_authority);
        target_binding.validate(Some(&target_system)).unwrap();
        let mut wrong_unlock = target_binding.clone();
        wrong_unlock.target_cross_cell_unlock_sha256 = Some("0".repeat(64));
        assert!(wrong_unlock.validate(Some(&target_system)).is_err());
        let mut wrong_panel = target_binding;
        wrong_panel.panel_scientific_sha256 = Some("1".repeat(64));
        assert!(wrong_panel.validate(Some(&target_system)).is_err());

        let mut corrupted = binding.clone();
        corrupted.endpoint_support_binding_sha256 = Some("0".repeat(64));
        assert!(corrupted.validate_pre_topology().is_ok());
        assert!(corrupted.validate(Some(&system)).is_err());
        let mut corrupted = binding.clone();
        corrupted.initial_protocol_phase.stream_id = 9;
        assert!(corrupted.validate_pre_topology().is_err());
        let mut corrupted = binding.clone();
        corrupted.initial_protocol_phase.thermodynamic_condition_id = "wrong".into();
        assert!(corrupted.validate_pre_topology().is_err());
        let mut corrupted = binding.clone();
        corrupted.engine_sha256 = "not-a-sha".into();
        assert!(corrupted.validate_pre_topology().is_err());
        let mut corrupted = binding.clone();
        corrupted.topology_sha256 = "not-a-sha".into();
        assert!(corrupted.validate_pre_topology().is_err());
        let mut corrupted = binding.clone();
        corrupted.lane.contract_sha256 = "0".repeat(64);
        assert!(corrupted.validate_pre_topology().is_err());
        let mut corrupted = binding.clone();
        corrupted.ncmc_execution_mode = None;
        assert!(corrupted.validate_pre_topology().is_err());
        let mut corrupted = binding.clone();
        corrupted.typed_event_output_dir = PathBuf::new();
        assert!(corrupted.validate_pre_topology().is_err());
        let phase_digest = binding.phase_protocol_sha256().unwrap();
        let restart_binding = binding.clone();
        let matched_receipt =
            test_matched_velocity_receipt(&binding.matched_velocity_initialization, 3, 42);
        let mut emitter =
            LambdaTypedEventEmitter::new_runtime_inline(binding, Some(system.clone())).unwrap();
        emitter
            .bind_matched_velocity_initialization(matched_receipt)
            .unwrap();
        let initial = system.slots[0].endpoint_a.clone();
        let current_endpoints = BTreeMap::from([(system.slots[0].slot_id, initial.clone())]);
        let proposal = scheduler
            .proposal_for_step(18, &current_endpoints, execution_mode)
            .unwrap();
        scheduler
            .validate_proposal_at_step(18, &proposal, &current_endpoints)
            .unwrap();
        let balance = derive_ncmc_balance(&system, &proposal, &initial).unwrap();
        let trace = emit_zero_work_trace_for_test(
            &mut emitter,
            &system,
            proposal.move_id,
            NcmcProtocolDirection::Forward,
            3,
            18,
            &phase_digest,
        );
        let ncmc = adjudicate_ncmc_move(
            &system,
            &proposal,
            &balance,
            3,
            18,
            phase_digest.clone(),
            emitter.next_sequence(),
            emitter.last_receipt_sha256().unwrap().to_string(),
            endpoint_support_binding_sha256,
            dual_endpoint_union_contract_sha256,
            initial,
            0.0,
            NcmcIntegratorPathAccounting::ShadowWorkVolumePreserving {
                shadow_work_reduced: 0.0,
            },
            true,
            true,
            trace,
            test_rng_consumption(&proposal),
        )
        .unwrap();
        let ncmc_json = serde_json::to_vec(&ncmc).unwrap();
        serde_json::from_slice::<NcmcMoveReceipt>(&ncmc_json).unwrap();
        assert!(ncmc.accepted);
        let ncmc_object_sha256 = ncmc.digest_sha256().unwrap();
        let supporting_population = match ncmc.ordered_work_trace.entries.last().unwrap() {
            NcmcWorkTraceEntry::Perturb {
                boundary_lineage, ..
            }
            | NcmcWorkTraceEntry::Propagate {
                boundary_lineage, ..
            }
            | NcmcWorkTraceEntry::MomentumReverse {
                boundary_lineage, ..
            } => &boundary_lineage.after,
        };
        let supporting_force = &supporting_population.slot_events[0];

        // Crash injection after member 1: no committed-looking directory and
        // the emitter head must return exactly to the force-trace frontier.
        let crash_one = emitter
            .begin_ncmc_resolution_batch(3, 18, ncmc.move_id)
            .unwrap();
        let ncmc_emission = emitter.emit_ncmc(&ncmc).unwrap();
        emitter.abort_ncmc_resolution_batch(crash_one).unwrap();
        assert_eq!(emitter.next_sequence(), ncmc.event_sequence);
        assert_eq!(
            emitter.last_receipt_sha256(),
            Some(ncmc.parent_receipt_sha256.as_str())
        );
        let endpoint = LambdaFragmentEndpointObservation {
            schema_kind: "prismos_lambda_fragment_endpoint_observation".into(),
            schema_version: 2,
            stream_id: 3,
            step: 18,
            event_sequence: ncmc.event_sequence + 1,
            parent_receipt_sha256: ncmc_emission.receipt_sha256.clone(),
            supporting_force_event_sha256: supporting_force.raw_event_sha256.clone(),
            supporting_force_emission_receipt_sha256: supporting_force
                .emission_receipt_sha256
                .clone(),
            supporting_force_boundary: supporting_population.boundary.clone(),
            slot_id: 9,
            species_id: proposal.proposed_species_id,
            chemical_state_id: proposal.proposed_chemical_state_id,
            lambda_endpoint: 1,
            occupancy: proposal.proposed_occupancy,
            path_kind: system.slots[0].path_kind,
            endpoint_a: system.slots[0].endpoint_a.clone(),
            endpoint_b: system.slots[0].endpoint_b.clone(),
            phase_protocol_sha256: phase_digest.clone(),
            ghost_lineage_sha256: None,
            prismcode_opcode: prismcode_opcodes::ENDPOINT_OBSERVE.into(),
            prismcode_opcode_id: opcode_id_for_name(prismcode_opcodes::ENDPOINT_OBSERVE).unwrap(),
            qnanl_artifact_family: QnanlArtifactFamily::LambdaFragmentEndpointObservation
                .as_str()
                .into(),
        };

        // Crash injection after member 2.
        let crash_two = emitter
            .begin_ncmc_resolution_batch(3, 18, ncmc.move_id)
            .unwrap();
        assert_eq!(
            emitter.emit_ncmc(&ncmc).unwrap().receipt_sha256,
            ncmc_emission.receipt_sha256
        );
        let endpoint_emission = emitter.emit_endpoint(&endpoint).unwrap();
        emitter.abort_ncmc_resolution_batch(crash_two).unwrap();
        assert_eq!(emitter.next_sequence(), ncmc.event_sequence);
        let commit = LambdaFragmentStateCommit {
            schema_kind: "prismos_lambda_fragment_state_commit".into(),
            schema_version: 1,
            stream_id: 3,
            step: 18,
            ncmc_move_receipt_sha256: ncmc_object_sha256.clone(),
            event_sequence: ncmc.event_sequence + 2,
            parent_receipt_sha256: endpoint_emission.receipt_sha256.clone(),
            slot_id: 9,
            species_id: 7,
            chemical_state_id: 0,
            occupancy: EndpointOccupancy::Occupied,
            phase_protocol_sha256: phase_digest,
            prismcode_opcode: prismcode_opcodes::STATE_COMMIT.into(),
            prismcode_opcode_id: opcode_id_for_name(prismcode_opcodes::STATE_COMMIT).unwrap(),
            qnanl_artifact_family: QnanlArtifactFamily::LambdaFragmentStateCommit
                .as_str()
                .into(),
        };

        // Crash injection after all three staged members but before publish.
        let crash_three = emitter
            .begin_ncmc_resolution_batch(3, 18, ncmc.move_id)
            .unwrap();
        emitter.emit_ncmc(&ncmc).unwrap();
        emitter.emit_endpoint(&endpoint).unwrap();
        let crash_terminal = emitter.emit_state_commit(&commit).unwrap();
        let injected_error = emitter
            .commit_ncmc_resolution_batch_inject_after_rename(
                crash_three.clone(),
                3,
                18,
                ncmc.move_id,
                ncmc_object_sha256.clone(),
                crash_terminal.receipt_sha256,
            )
            .unwrap_err();
        assert!(injected_error.contains("injected crash"));
        emitter.abort_ncmc_resolution_batch(crash_three).unwrap();
        assert_eq!(emitter.next_sequence(), ncmc.event_sequence);

        let resolution_transaction = emitter
            .begin_ncmc_resolution_batch(3, 18, ncmc.move_id)
            .unwrap();
        emitter.emit_ncmc(&ncmc).unwrap();
        emitter.emit_endpoint(&endpoint).unwrap();
        let terminal_emission = emitter.emit_state_commit(&commit).unwrap();
        emitter
            .commit_ncmc_resolution_batch(
                resolution_transaction,
                3,
                18,
                ncmc.move_id,
                ncmc_object_sha256,
                terminal_emission.receipt_sha256,
            )
            .unwrap();
        let closeout = emitter.close_runtime_and_emit_receipt(100).unwrap();
        closeout.validate().unwrap();
        assert_eq!(
            (
                closeout.produced,
                closeout.persisted,
                closeout.decoded,
                closeout.indexed,
                closeout.committed_resolution_batch_count,
                closeout.scheduled_ncmc_opportunity_count,
            ),
            (
                emitter.next_sequence(),
                emitter.next_sequence(),
                emitter.next_sequence(),
                emitter.next_sequence(),
                1,
                1,
            )
        );
        drop(emitter);
        let restart_error =
            LambdaTypedEventEmitter::new_runtime_inline(restart_binding, Some(system.clone()))
                .err()
                .expect("same-run journal restart must fail closed without a physical checkpoint");
        assert!(restart_error.contains("same-run emitter recovery is forbidden"));
        let fixture_override = std::env::var_os("PRISM_NCMC_FIXTURE_OUTPUT").map(PathBuf::from);
        let fixture_path = fixture_override
            .clone()
            .unwrap_or_else(|| output.join("rust_ncmc_v6_calibration_join.json"));
        let artifact = export_verified_lambda_calibration_joins(&output, &fixture_path).unwrap();
        artifact.validate().unwrap();
        assert_eq!(artifact.rows.len(), 1);
        assert!(artifact.rows[0].chemical_potential_calibration.is_none());
        assert_eq!(
            artifact.rows[0].log_target_ratio.to_bits(),
            0.0f64.to_bits()
        );
        if fixture_override.is_none() {
            let _ = std::fs::remove_dir_all(&output);
        }
    }

    #[test]
    fn runtime_no_probe_journal_roundtrips_context_and_population_receipt() {
        let output = std::env::temp_dir().join(format!(
            "prism_lambda_no_probe_journal_{}_{}",
            std::process::id(),
            counter_based_uniform_open01(11, 19).to_bits()
        ));
        let _ = std::fs::remove_dir_all(&output);
        let lane = LambdaExperimentLaneContract::no_probe("no-probe-control").unwrap();
        let phase = crate::chemical_state_fiber::ProtocolPhaseBinding {
            stream_id: 2,
            thermodynamic_condition_id: "condition-0".into(),
            phase_ordinal: 0,
            phase_name: "apo-baseline".into(),
            start_step: 100,
            end_step_exclusive: 200,
            protocol_name: "matched-lane-v1".into(),
            phase_schedule_digest_sha256: "1".repeat(64),
            protocol_state_hash: "2".repeat(64),
        };
        let condition = crate::chemical_state_fiber::ThermodynamicCondition {
            condition_id: "condition-0".into(),
            temperature_k: 300.0,
            pressure_bar: 1.0,
            ionic_strength_molar: 0.15,
            solvent_model: "TIP3P".into(),
            periodic_box_vectors_a: [[40.0, 0.0, 0.0], [0.0, 40.0, 0.0], [0.0, 0.0, 40.0]],
            pme_parameter_sha256: "3".repeat(64),
        };
        let binding = LambdaRuntimeBinding {
            run_id: "run-no-probe".into(),
            replica_id: "rep-00".into(),
            replica_seed: 42,
            stream_id: 2,
            stream_seed: 44,
            engine_sha256: "4".repeat(64),
            topology_sha256: "5".repeat(64),
            endpoint_support_binding_sha256: None,
            runtime_capability_receipt: None,
            matched_velocity_initialization: test_matched_velocity_contract(false),
            decoupled_fragment_placement_measure_authority: None,
            target_cross_cell_unlock_sha256: None,
            panel_scientific_sha256: None,
            lane: lane.clone(),
            ncmc_execution_mode: None,
            ncmc_scheduler: None,
            initial_protocol_phase: phase.clone(),
            initial_thermodynamic_condition: condition.clone(),
            protocol_schedule: vec![LambdaRuntimePhaseContext {
                protocol_phase: phase.clone(),
                thermodynamic_condition: condition.clone(),
            }],
            run_output_root: output.parent().unwrap().to_path_buf(),
            typed_event_output_dir: output.clone(),
            typed_event_capacity: test_capacity_contract(),
        };
        let phase_digest = binding.phase_protocol_sha256().unwrap();
        let matched_receipt =
            test_matched_velocity_receipt(&binding.matched_velocity_initialization, 2, 42);
        let mut emitter = LambdaTypedEventEmitter::new_runtime_inline(binding, None).unwrap();
        emitter
            .bind_matched_velocity_initialization(matched_receipt)
            .unwrap();
        let event = NoProbeAbsenceReceipt {
            schema_kind: "prismos_lambda_no_probe_absence_receipt".into(),
            schema_version: 1,
            stream_id: 2,
            start_step: 100,
            end_step_exclusive: 200,
            event_sequence: 0,
            parent_receipt_sha256: None,
            lane_contract_sha256: lane.contract_sha256,
            lambda_slot_count: 0,
            virtual_probe_count: 0,
            lambda_event_count: 0,
            phase_protocol_sha256: phase_digest,
            prismcode_opcode: prismcode_opcodes::NO_PROBE_ABSENCE.into(),
            prismcode_opcode_id: opcode_id_for_name(prismcode_opcodes::NO_PROBE_ABSENCE).unwrap(),
            qnanl_artifact_family: QnanlArtifactFamily::LambdaNoProbeAbsence.as_str().into(),
        };
        emitter.emit_no_probe_absence(&event).unwrap();
        let closeout = emitter.close_runtime_and_emit_receipt(200).unwrap();
        closeout.validate().unwrap();
        assert_eq!(
            (
                closeout.produced,
                closeout.persisted,
                closeout.decoded,
                closeout.indexed,
            ),
            (1, 1, 1, 1)
        );
        let loaded = load_verified_lambda_event_journals(&output).unwrap();
        assert_eq!(
            (
                loaded.produced,
                loaded.persisted,
                loaded.decoded,
                loaded.indexed
            ),
            (1, 1, 1, 1)
        );
        assert_eq!(loaded.stream_contexts[&2].protocol_phase, phase);
        assert_eq!(
            loaded
                .phase_context_at(2, 150)
                .unwrap()
                .thermodynamic_condition,
            condition
        );
        assert!(loaded.phase_context_at(2, 200).is_none());
        assert!(loaded.no_probe_absence_at(2, 150).is_some());
        let make_ghost = |
            step_idx: u64,
            protocol_phase: crate::chemical_state_fiber::ProtocolPhaseBinding,
            thermodynamic_condition: crate::chemical_state_fiber::ThermodynamicCondition,
        | {
            crate::chemical_state_fiber::GhostFiberObservation {
                site_key: "site-no-probe".into(),
                control_lane: crate::chemical_state_fiber::ControlLane::NoProbe,
                ghost: crate::chemical_state_fiber::GhostByteLineage {
                    key: crate::chemical_state_fiber::GhostLineageKey {
                        stream_id: 2,
                        frame_idx: step_idx,
                        step_idx,
                        site_id: 7,
                        record_index: 0,
                    },
                    run_id: "run-no-probe".into(),
                    replica_id: "rep-00".into(),
                    replica_seed: 42,
                    stream_seed: 44,
                    topology_sha256: "5".repeat(64),
                    record_abi_sha256: "6".repeat(64),
                    record_source_sha256: "7".repeat(64),
                    segment_context_digest_sha256: "8".repeat(64),
                    segment_manifest_sha256: "9".repeat(64),
                    segment_sequence: 0,
                    segment_sha256: "a".repeat(64),
                    segment_file: format!("ghost_{step_idx}.bin"),
                    byte_offset: 0,
                    byte_length: 64,
                    record_sha256: format!("{:064x}", step_idx),
                    lattice_node_index: 0,
                    engine_sha256: "4".repeat(64),
                    condition_id: thermodynamic_condition.condition_id.clone(),
                    protocol_name: protocol_phase.protocol_name.clone(),
                    protocol_phase: protocol_phase.phase_ordinal,
                    phase_schedule_digest_sha256: protocol_phase
                        .phase_schedule_digest_sha256
                        .clone(),
                    protocol_state_hash: protocol_phase.protocol_state_hash.clone(),
                },
                protocol_phase,
                thermodynamic_condition,
                structural_response: crate::chemical_state_fiber::StructuralResponse {
                    open_state: crate::chemical_state_fiber::OpenPocketEstimand {
                        probability: 0.0,
                        definition_sha256: "a".repeat(64),
                        pathb_model_sha256: "b".repeat(64),
                        lag_steps: 1,
                        open_state_ids: vec!["open".into()],
                    },
                    pocket_volume_a3: None,
                    pocket_depth_a: None,
                    solvent_accessible_area_a2: None,
                    delta_volume_a3: None,
                    delta_hydration: None,
                    fragment_contact_fraction: None,
                },
                uncertainty: crate::chemical_state_fiber::SamplingUncertainty {
                    effective_sample_size: 1.0,
                    integrated_autocorrelation_time_frames: 0.0,
                    block_count: 1,
                    standard_error: 0.0,
                    confidence_interval_lower: 0.0,
                    confidence_interval_upper: 0.0,
                    replica_seed: 42,
                    replica_id: "rep-00".into(),
                },
                evidence_artifact_sha256: BTreeMap::new(),
            }
        };
        let ghost = make_ghost(150, phase.clone(), condition.clone());
        let joins =
            crate::chemical_state_fiber::materialize_lambda_ghost_endpoint_joins(&[ghost], &loaded)
                .unwrap();
        assert_eq!(joins.evaluated_join_count, 0);
        assert_eq!(joins.not_evaluable_count, 1);
        let adapter = adapt_verified_lambda_journals_for_pathb(
            &loaded,
            &joins,
            crate::chemical_state_fiber::ControlLane::NoProbe,
            &"c".repeat(64),
            None,
        )
        .unwrap();
        adapter.validate().unwrap();
        assert_eq!(adapter.fragment_observations.len(), 1);
        assert_eq!(adapter.not_evaluable_lambda_ghost_joins.len(), 1);

        let mut wrong_population = joins.clone();
        wrong_population.lambda_population_receipt_sha256 = "d".repeat(64);
        wrong_population.materialization_sha256.clear();
        wrong_population.materialization_sha256 =
            crate::chemical_state_fiber::canonical_sha256(&wrong_population).unwrap();
        wrong_population.verify().unwrap();
        assert!(adapt_verified_lambda_journals_for_pathb(
            &loaded,
            &wrong_population,
            crate::chemical_state_fiber::ControlLane::NoProbe,
            &"c".repeat(64),
            None,
        )
        .unwrap_err()
        .contains("different verified journal population"));

        let outside_phase = crate::chemical_state_fiber::ProtocolPhaseBinding {
            start_step: 200,
            end_step_exclusive: 300,
            phase_ordinal: 1,
            phase_name: "uncovered-phase".into(),
            ..phase.clone()
        };
        let unmatched = make_ghost(250, outside_phase, condition.clone());
        let unmatched_joins = crate::chemical_state_fiber::materialize_lambda_ghost_endpoint_joins(
            &[unmatched],
            &loaded,
        )
        .unwrap();
        assert!(adapt_verified_lambda_journals_for_pathb(
            &loaded,
            &unmatched_joins,
            crate::chemical_state_fiber::ControlLane::NoProbe,
            &"c".repeat(64),
            None,
        )
        .unwrap_err()
        .contains("lacks typed no-probe absence coverage"));
        validate_sha256(
            "lambda journal population",
            &loaded.population_receipt_sha256,
        )
        .unwrap();
        let _ = std::fs::remove_dir_all(&output);
    }
}
