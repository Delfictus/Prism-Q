//! Content-addressed chemical-state fibers for native Ghost/Path-B evidence.
//!
//! A fiber is an exact join, never a proximity join.  It binds one Ghost record
//! (including segment and byte offsets) to the fragment Hamiltonian state, the
//! engine-owned phase schedule, a structural response, a native Path-B state,
//! and block/seed uncertainty.  The body is serialized deterministically and
//! SHA-256 addressed, so the same typed inputs reconstruct the same fiber even
//! when input collections arrive in a different order.

/// Full saved-data mechanistic measurements bound to exact fiber identities.
pub mod mechanistic;

use crate::lambda_fragment::{
    force_event_dudl_for_fiber, AlchemicalPathKind, ChemicalPotentialCalibration,
    EndpointOccupancy as LambdaEndpointOccupancy, LambdaEventJournalLoad,
    LambdaExperimentLaneContract, LambdaExperimentLaneKind, LambdaFiberUnitConversion,
    LambdaFragmentEndpointObservation, LambdaFragmentForceEvent, LambdaJournalEventKind,
    LambdaJournalKey, LambdaJournalLoadedLineage, LambdaJournalObservation,
    NcmcForceBoundaryIdentity,
};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

/// Schema identifier for serialized fiber records.
pub const CHEMICAL_STATE_FIBER_SCHEMA: &str = "prism_chemical_state_fiber";
/// Current fiber schema version.
pub const CHEMICAL_STATE_FIBER_VERSION: &str = "4.0.0";
/// Canonical PrismCode opcode for `PATHB.CHEMICAL_STATE_FIBER_MATERIALIZED`.
pub use prismcode::binary_runtime::opcode::PATHB_CHEMICAL_STATE_FIBER_MATERIALIZED as PRISMCODE_OPCODE_CHEMICAL_STATE_FIBER_MATERIALIZED;
/// Canonical PrismCode semantic row for fiber materialization.
pub const EVENT_CHEMICAL_STATE_FIBER_MATERIALIZED: &str = "PATHB.CHEMICAL_STATE_FIBER_MATERIALIZED";
/// Canonical QNANL artifact-family code for chemical-state fibers.
pub const QNANL_ARTIFACT_CODE_CHEMICAL_STATE_FIBER: u16 =
    prismcode::qnanl_artifact::QnanlArtifactFamily::ChemicalStateFiber as u16;
/// Canonical QNANL artifact family for chemical-state fibers.
pub const QNANL_FAMILY_CHEMICAL_STATE_FIBER: &str =
    prismcode::qnanl_artifact::QnanlArtifactFamily::ChemicalStateFiber.as_str();

/// Experimental lane represented by a fiber.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlLane {
    /// Matched dynamics without a fragment molecule.
    NoProbe,
    /// Fragment selected to stimulate the hypothesized pocket chemistry.
    HypothesisFragment,
    /// Size/shape-matched but chemically orthogonal fragment control.
    OrthogonalChemicalControl,
}

/// Whether a lambda state can support a physical pocket claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EndpointEligibility {
    /// Physical Hamiltonian endpoint; eligible after all other gates pass.
    PhysicalEndpoint,
    /// Intermediate lambda state used for sampling only.
    SamplingOnly,
}

/// Stable key identifying one Ghost record before content hashing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct GhostLineageKey {
    /// Engine stream identifier.
    pub stream_id: u16,
    /// Engine frame identifier.
    pub frame_idx: u64,
    /// Integration step identifier.
    pub step_idx: u64,
    /// Ghost site identifier within the stream.
    pub site_id: u32,
    /// Ordinal record index within the segment.
    pub record_index: u64,
}

/// Exactly-once byte lineage for one Ghost record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GhostByteLineage {
    /// Stable record coordinates.
    pub key: GhostLineageKey,
    /// Immutable run identity from the segment context.
    pub run_id: String,
    /// Immutable replica identity from the segment context.
    pub replica_id: String,
    /// Independent replica seed that produced the record.
    pub replica_seed: u64,
    /// Exact per-stream seed that produced the record.
    pub stream_seed: u64,
    /// SHA-256 of the exact source topology.
    pub topology_sha256: String,
    /// SHA-256 of the record ABI declaration.
    pub record_abi_sha256: String,
    /// SHA-256 of the record producer source.
    pub record_source_sha256: String,
    /// SHA-256 of the complete immutable segment context.
    pub segment_context_digest_sha256: String,
    /// SHA-256 of the segment manifest that owns this record.
    pub segment_manifest_sha256: String,
    /// Monotonic segment sequence number.
    pub segment_sequence: u64,
    /// SHA-256 of the immutable segment file.
    pub segment_sha256: String,
    /// Stable output-root-relative segment artifact path.
    pub segment_file: String,
    /// Byte offset of this complete record in the segment.
    pub byte_offset: u64,
    /// Byte length of this record.
    pub byte_length: u32,
    /// SHA-256 of the exact record bytes.
    pub record_sha256: String,
    /// Lattice node index produced from this record.
    pub lattice_node_index: u64,
    /// Content-addressed engine build identity.
    pub engine_sha256: String,
    /// Engine-owned thermodynamic condition identity.
    pub condition_id: String,
    /// Engine-owned protocol identity.
    pub protocol_name: String,
    /// Exact resolved protocol phase ordinal.
    pub protocol_phase: u16,
    /// Digest of the exact phase schedule.
    pub phase_schedule_digest_sha256: String,
    /// Hash of the final engine protocol state.
    pub protocol_state_hash: String,
}

impl GhostByteLineage {
    /// Validate hashes and byte-range arithmetic.
    pub fn validate(&self) -> Result<()> {
        require_sha256(&self.segment_manifest_sha256, "segment_manifest_sha256")?;
        require_sha256(&self.segment_sha256, "segment_sha256")?;
        require_sha256(&self.record_sha256, "record_sha256")?;
        require_sha256(&self.engine_sha256, "engine_sha256")?;
        for (value, field) in [
            (&self.topology_sha256, "topology_sha256"),
            (&self.record_abi_sha256, "record_abi_sha256"),
            (&self.record_source_sha256, "record_source_sha256"),
            (
                &self.segment_context_digest_sha256,
                "segment_context_digest_sha256",
            ),
            (
                &self.phase_schedule_digest_sha256,
                "phase_schedule_digest_sha256",
            ),
            (&self.protocol_state_hash, "protocol_state_hash"),
        ] {
            require_sha256(value, field)?;
        }
        for (value, field) in [
            (&self.run_id, "run_id"),
            (&self.replica_id, "replica_id"),
            (&self.segment_file, "segment_file"),
            (&self.condition_id, "condition_id"),
            (&self.protocol_name, "protocol_name"),
        ] {
            require_nonempty(value, field)?;
        }
        let segment_path = std::path::Path::new(&self.segment_file);
        if segment_path.is_absolute()
            || segment_path.components().any(|component| {
                matches!(
                    component,
                    std::path::Component::CurDir | std::path::Component::ParentDir
                )
            })
        {
            bail!("segment_file must be a normalized output-root-relative path");
        }
        if self.byte_length == 0 {
            bail!("Ghost record byte_length must be non-zero");
        }
        self.byte_offset
            .checked_add(u64::from(self.byte_length))
            .context("Ghost record byte range overflow")?;
        Ok(())
    }

    /// Canonical global identity of this exact record occurrence, including
    /// run/replica/seed, segment byte location, engine/topology and protocol
    /// context. Raw record bytes alone are not globally unique across matched
    /// independent runs.
    pub fn lineage_sha256(&self) -> Result<String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

/// Explicit molecular identity at one lambda state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FragmentChemicalIdentity {
    /// Stable species identifier from the fragment parameter manifest.
    pub species_id: String,
    /// Exact native numeric species identity used by the lambda kernel.
    pub native_species_id: u32,
    /// Exact native tautomer/protonation-state identity.
    pub native_chemical_state_id: u32,
    /// Canonical isomeric SMILES used to define molecular identity.
    pub canonical_isomeric_smiles: String,
    /// Content hash of atom ordering, bonded terms, charges, and endpoints.
    pub parameter_digest_sha256: String,
    /// Explicit tautomer identifier.
    pub tautomer_id: String,
    /// Explicit protonation microstate identifier.
    pub protonation_state_id: String,
    /// Formal molecular charge.
    pub formal_charge: i16,
}

impl FragmentChemicalIdentity {
    /// Validate that identity fields are explicit and hash-bound.
    pub fn validate(&self) -> Result<()> {
        require_nonempty(&self.species_id, "species_id")?;
        require_nonempty(&self.canonical_isomeric_smiles, "canonical_isomeric_smiles")?;
        require_sha256(&self.parameter_digest_sha256, "parameter_digest_sha256")?;
        require_nonempty(&self.tautomer_id, "tautomer_id")?;
        require_nonempty(&self.protonation_state_id, "protonation_state_id")?;
        Ok(())
    }
}

/// Molecular occupancy at a physical or sampling state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum FragmentOccupancy {
    /// No fragment molecule is present.
    Absent,
    /// One explicit molecule instance occupies a fragment slot.
    Present {
        /// Stable molecule instance identifier.
        molecule_instance_id: String,
        /// Stable slot identifier in the periodic topology.
        fragment_slot_id: u32,
    },
}

/// Hamiltonian state for a fragment observation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FragmentState {
    /// Physical identity at an occupied lambda endpoint. It is absent for the
    /// no-probe lane, a decoupled endpoint, and partial switching states.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub identity: Option<FragmentChemicalIdentity>,
    /// Explicit lambda=0 identity (even when that endpoint is decoupled).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint_a_identity: Option<FragmentChemicalIdentity>,
    /// Explicit lambda=1 identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint_b_identity: Option<FragmentChemicalIdentity>,
    /// Explicit occupancy state.
    pub occupancy: FragmentOccupancy,
    /// Soft-core coupling coordinate in `[0, 1]`.
    pub lambda: f64,
    /// Analytic derivative of the declared potential with respect to lambda.
    pub du_dlambda_kj_mol: f64,
    /// Hash-bound unit-conversion receipt from the native lambda force event.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dudl_unit_conversion: Option<LambdaFiberUnitConversion>,
    /// Chemical potential used for this species/microstate.
    pub chemical_potential_kj_mol: f64,
    /// Content hash of the Hamiltonian declaration used at this step.
    pub hamiltonian_sha256: String,
    /// Complete native force event from which energy and dU/dlambda derive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_force_event: Option<LambdaFragmentForceEvent>,
    /// Exact physical endpoint observation. Absent only for no-probe and
    /// intermediate switching states.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_endpoint_observation: Option<LambdaFragmentEndpointObservation>,
    /// Explicit standard-state chemical-potential calibration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chemical_potential_calibration: Option<ChemicalPotentialCalibration>,
}

impl FragmentState {
    /// Validate physical identity and numeric state for a lane.
    pub fn validate_for_lane(&self, lane: ControlLane) -> Result<()> {
        finite_in_range(self.lambda, 0.0, 1.0, "lambda")?;
        require_finite(self.du_dlambda_kj_mol, "du_dlambda_kj_mol")?;
        require_finite(self.chemical_potential_kj_mol, "chemical_potential_kj_mol")?;
        require_sha256(&self.hamiltonian_sha256, "hamiltonian_sha256")?;
        match lane {
            ControlLane::NoProbe => {
                if self.identity.is_some()
                    || self.endpoint_a_identity.is_some()
                    || self.endpoint_b_identity.is_some()
                    || self.occupancy != FragmentOccupancy::Absent
                {
                    bail!("no_probe fibers must have absent identity and occupancy");
                }
                if self.lambda != 0.0 {
                    bail!("no_probe lambda must be exactly 0");
                }
                if self.du_dlambda_kj_mol != 0.0 || self.dudl_unit_conversion.is_some() {
                    bail!("no_probe fibers cannot carry a lambda-force derivative");
                }
                if self.chemical_potential_kj_mol != 0.0
                    || self.source_force_event.is_some()
                    || self.source_endpoint_observation.is_some()
                    || self.chemical_potential_calibration.is_some()
                {
                    bail!("no_probe fibers cannot carry lambda or chemical-potential evidence");
                }
            }
            ControlLane::HypothesisFragment | ControlLane::OrthogonalChemicalControl => {
                let endpoint_a_identity = self
                    .endpoint_a_identity
                    .as_ref()
                    .context("probe lane requires explicit lambda=0 chemical identity")?;
                endpoint_a_identity.validate()?;
                let endpoint_b_identity = self
                    .endpoint_b_identity
                    .as_ref()
                    .context("probe lane requires explicit lambda=1 chemical identity")?;
                endpoint_b_identity.validate()?;
                let conversion = self
                    .dudl_unit_conversion
                    .as_ref()
                    .context("probe fiber requires a native force-event dU/dlambda receipt")?;
                conversion
                    .validate()
                    .map_err(|error| anyhow::anyhow!(error))?;
                if conversion.du_dlambda_kj_mol.to_bits() != self.du_dlambda_kj_mol.to_bits() {
                    bail!("fiber dU/dlambda differs from the hash-bound native conversion");
                }
                let force = self
                    .source_force_event
                    .as_ref()
                    .context("probe fiber requires the complete native force event")?;
                force.validate().map_err(anyhow::Error::msg)?;
                let recomputed = force_event_dudl_for_fiber(force).map_err(anyhow::Error::msg)?;
                if &recomputed != conversion
                    || force.hamiltonian_sha256 != self.hamiltonian_sha256
                    || force.lambda.to_bits() != self.lambda.to_bits()
                    || force.endpoint_a.species_id != endpoint_a_identity.native_species_id
                    || force.endpoint_a.chemical_state_id
                        != endpoint_a_identity.native_chemical_state_id
                    || force.endpoint_b.species_id != endpoint_b_identity.native_species_id
                    || force.endpoint_b.chemical_state_id
                        != endpoint_b_identity.native_chemical_state_id
                {
                    bail!("fiber lambda event, identities, Hamiltonian, or dU/dlambda disagree");
                }
                let calibration = self
                    .chemical_potential_calibration
                    .as_ref()
                    .context("probe fiber requires chemical-potential calibration")?;
                calibration.validate().map_err(anyhow::Error::msg)?;
                let selected_identity = if self.lambda == 0.0 {
                    endpoint_a_identity
                } else {
                    endpoint_b_identity
                };
                if calibration.species_id != selected_identity.native_species_id
                    || calibration.chemical_state_id != selected_identity.native_chemical_state_id
                    || (calibration.total_chemical_potential_kcal_mol * 4.184).to_bits()
                        != self.chemical_potential_kj_mol.to_bits()
                {
                    bail!("fiber chemical potential is not the exact calibrated microstate value");
                }
                self.validate_endpoint_semantics(force, endpoint_a_identity, endpoint_b_identity)?;
            }
        }
        Ok(())
    }

    fn validate_endpoint_semantics(
        &self,
        force: &LambdaFragmentForceEvent,
        endpoint_a_identity: &FragmentChemicalIdentity,
        endpoint_b_identity: &FragmentChemicalIdentity,
    ) -> Result<()> {
        let expected_active = if self.lambda == 0.0 {
            (force.endpoint_a.occupancy == LambdaEndpointOccupancy::Occupied)
                .then_some(endpoint_a_identity)
        } else if self.lambda == 1.0 {
            (force.endpoint_b.occupancy == LambdaEndpointOccupancy::Occupied)
                .then_some(endpoint_b_identity)
        } else {
            None
        };
        if self.identity.as_ref() != expected_active {
            bail!("fiber active identity does not match its physical lambda endpoint");
        }
        match (&self.occupancy, expected_active, self.lambda) {
            (FragmentOccupancy::Absent, None, 0.0)
                if force.path_kind == AlchemicalPathKind::InsertDelete => {}
            (
                FragmentOccupancy::Present {
                    molecule_instance_id,
                    fragment_slot_id,
                },
                Some(_),
                _,
            ) if *fragment_slot_id == force.slot_id => {
                require_nonempty(molecule_instance_id, "molecule_instance_id")?;
            }
            (
                FragmentOccupancy::Present {
                    molecule_instance_id,
                    fragment_slot_id,
                },
                None,
                lambda,
            ) if (0.0..1.0).contains(&lambda) && *fragment_slot_id == force.slot_id => {
                require_nonempty(molecule_instance_id, "molecule_instance_id")?;
            }
            _ => bail!("fiber occupancy does not match the native alchemical path endpoint"),
        }
        let at_endpoint = self.lambda == 0.0 || self.lambda == 1.0;
        match (&self.source_endpoint_observation, at_endpoint) {
            (None, false) => Ok(()),
            (Some(endpoint), true) => {
                endpoint.validate().map_err(anyhow::Error::msg)?;
                let expected_ordinal = if self.lambda == 0.0 { 0 } else { 1 };
                let expected_occupancy = if self.lambda == 0.0 {
                    force.endpoint_a.occupancy
                } else {
                    force.endpoint_b.occupancy
                };
                if endpoint.lambda_endpoint != expected_ordinal
                    || endpoint.occupancy != expected_occupancy
                    || endpoint.slot_id != force.slot_id
                    || endpoint.species_id != force.species_id
                    || endpoint.chemical_state_id != force.chemical_state_id
                    || endpoint.path_kind != force.path_kind
                    || endpoint.endpoint_a.species_id != force.endpoint_a.species_id
                    || endpoint.endpoint_a.chemical_state_id != force.endpoint_a.chemical_state_id
                    || endpoint.endpoint_a.occupancy != force.endpoint_a.occupancy
                    || endpoint.endpoint_b.species_id != force.endpoint_b.species_id
                    || endpoint.endpoint_b.chemical_state_id != force.endpoint_b.chemical_state_id
                    || endpoint.endpoint_b.occupancy != force.endpoint_b.occupancy
                    || force.ncmc_boundary.as_ref() != Some(&endpoint.supporting_force_boundary)
                {
                    bail!("lambda endpoint observation disagrees with force-plane event");
                }
                let mut force_bytes =
                    serde_json::to_vec(force).context("serialize supporting lambda force event")?;
                force_bytes.push(b'\n');
                let exact_force_sha256 = format!("{:x}", Sha256::digest(force_bytes));
                if endpoint.supporting_force_event_sha256 != exact_force_sha256 {
                    bail!("lambda endpoint does not name its exact supporting force bytes");
                }
                Ok(())
            }
            _ => bail!("physical endpoints require exactly one native endpoint observation"),
        }
    }

    fn validate_runtime_lineage(
        &self,
        ghost: &GhostByteLineage,
        phase: &ProtocolPhaseBinding,
        condition: &ThermodynamicCondition,
        lane: ControlLane,
    ) -> Result<()> {
        if lane == ControlLane::NoProbe {
            return Ok(());
        }
        let force = self.source_force_event.as_ref().unwrap();
        if u16::try_from(force.stream_id).ok() != Some(ghost.key.stream_id)
            || force.step != ghost.key.step_idx
        {
            bail!("lambda force event does not match exact Ghost stream/step");
        }
        let calibration = self.chemical_potential_calibration.as_ref().unwrap();
        if calibration.temperature_k.to_bits() != condition.temperature_k.to_bits() {
            bail!("chemical-potential calibration temperature differs from fiber condition");
        }
        if let Some(endpoint) = &self.source_endpoint_observation {
            let ghost_lineage_sha256 = ghost.lineage_sha256()?;
            if u16::try_from(endpoint.stream_id).ok() != Some(ghost.key.stream_id)
                || endpoint.step != ghost.key.step_idx
                || endpoint.ghost_lineage_sha256.as_deref() != Some(ghost_lineage_sha256.as_str())
                || endpoint.phase_protocol_sha256
                    != phase_protocol_binding_sha256(phase, condition)?
            {
                bail!("lambda endpoint observation lacks exact Ghost/phase/condition lineage");
            }
        }
        Ok(())
    }

    /// Classify lambda states without rounding intermediate values to endpoints.
    pub fn endpoint_eligibility(&self, lane: ControlLane) -> EndpointEligibility {
        if lane == ControlLane::NoProbe || self.lambda == 0.0 || self.lambda == 1.0 {
            EndpointEligibility::PhysicalEndpoint
        } else {
            EndpointEligibility::SamplingOnly
        }
    }

    /// Validate the complete fragment state against its exact Ghost,
    /// protocol-phase and thermodynamic-condition context. This is used by
    /// immutable pre-Path-B lane evidence before a neutral state assignment
    /// exists; it performs the same physical lineage checks as a full fiber.
    pub fn validate_for_ghost_context(
        &self,
        lane: ControlLane,
        ghost: &GhostByteLineage,
        phase: &ProtocolPhaseBinding,
        condition: &ThermodynamicCondition,
    ) -> Result<()> {
        self.validate_for_lane(lane)?;
        self.validate_runtime_lineage(ghost, phase, condition, lane)
    }
}

/// Engine-owned mapping from stream/step to protocol phase and condition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProtocolPhaseBinding {
    /// Engine stream identifier.
    pub stream_id: u16,
    /// Stable condition identifier.
    pub thermodynamic_condition_id: String,
    /// Protocol phase ordinal.
    pub phase_ordinal: u16,
    /// Operator-facing phase name.
    pub phase_name: String,
    /// Inclusive first integration step in the phase.
    pub start_step: u64,
    /// Exclusive step after the phase.
    pub end_step_exclusive: u64,
    /// Engine-owned protocol identity.
    pub protocol_name: String,
    /// SHA-256 of the engine-owned per-stream schedule.
    pub phase_schedule_digest_sha256: String,
    /// Hash of the final engine protocol state.
    pub protocol_state_hash: String,
}

impl ProtocolPhaseBinding {
    /// Validate exact stream and step membership.
    pub fn validate_for(&self, key: GhostLineageKey) -> Result<()> {
        require_nonempty(
            &self.thermodynamic_condition_id,
            "thermodynamic_condition_id",
        )?;
        require_nonempty(&self.phase_name, "phase_name")?;
        require_nonempty(&self.protocol_name, "protocol_name")?;
        require_sha256(
            &self.phase_schedule_digest_sha256,
            "phase_schedule_digest_sha256",
        )?;
        require_sha256(&self.protocol_state_hash, "protocol_state_hash")?;
        if self.stream_id != key.stream_id {
            bail!("phase binding stream does not match Ghost lineage");
        }
        if self.start_step >= self.end_step_exclusive
            || key.step_idx < self.start_step
            || key.step_idx >= self.end_step_exclusive
        {
            bail!("Ghost step is outside its declared protocol phase interval");
        }
        Ok(())
    }
}

/// Thermodynamic and periodic-solvent state for a fiber.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThermodynamicCondition {
    /// Stable condition identifier referenced by the phase binding.
    pub condition_id: String,
    /// Temperature in kelvin.
    pub temperature_k: f64,
    /// Pressure in bar.
    pub pressure_bar: f64,
    /// Ionic strength in molar units.
    pub ionic_strength_molar: f64,
    /// Explicit solvent model identifier.
    pub solvent_model: String,
    /// Periodic box vectors in angstrom.
    pub periodic_box_vectors_a: [[f64; 3]; 3],
    /// Particle-mesh Ewald parameter digest.
    pub pme_parameter_sha256: String,
}

impl ThermodynamicCondition {
    /// Validate finite physical state and a non-degenerate periodic box.
    pub fn validate(&self) -> Result<()> {
        require_nonempty(&self.condition_id, "condition_id")?;
        finite_in_range(self.temperature_k, 0.0, 10_000.0, "temperature_k")?;
        finite_in_range(self.pressure_bar, 0.0, 1_000_000.0, "pressure_bar")?;
        finite_in_range(
            self.ionic_strength_molar,
            0.0,
            100.0,
            "ionic_strength_molar",
        )?;
        require_nonempty(&self.solvent_model, "solvent_model")?;
        require_sha256(&self.pme_parameter_sha256, "pme_parameter_sha256")?;
        for row in self.periodic_box_vectors_a {
            for value in row {
                require_finite(value, "periodic_box_vectors_a")?;
            }
        }
        let [a, b, c] = self.periodic_box_vectors_a;
        let determinant = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]);
        if determinant.abs() <= 1.0e-9 {
            bail!("periodic box vectors are degenerate");
        }
        Ok(())
    }

    /// Canonical matched-lane physical state. Lane/species-specific condition
    /// labels are intentionally excluded; the exact temperature, pressure,
    /// ionic strength, solvent, periodic box, PME and physical phase interval
    /// remain required to match.
    pub fn matched_physical_state_sha256(&self, phase: &ProtocolPhaseBinding) -> Result<String> {
        self.validate()?;
        if phase.thermodynamic_condition_id != self.condition_id {
            bail!("phase and thermodynamic condition identities disagree");
        }
        canonical_sha256(&serde_json::json!({
            "temperature_k": self.temperature_k,
            "pressure_bar": self.pressure_bar,
            "ionic_strength_molar": self.ionic_strength_molar,
            "solvent_model": self.solvent_model,
            "periodic_box_vectors_a": self.periodic_box_vectors_a,
            "pme_parameter_sha256": self.pme_parameter_sha256,
            "phase_ordinal": phase.phase_ordinal,
            "phase_name": phase.phase_name,
            "phase_start_step": phase.start_step,
            "phase_end_step_exclusive": phase.end_step_exclusive,
        }))
    }
}

/// One optional physical measurement with explicit estimator provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhysicalMeasurement {
    /// Measured value in the field-specific units.
    pub value: f64,
    /// Stable estimator/method identifier.
    pub method_id: String,
    /// Content hash of estimator definition and parameters.
    pub estimator_sha256: String,
    /// Content hash of the coordinate/field source actually measured.
    pub source_artifact_sha256: String,
}

impl PhysicalMeasurement {
    fn validate(&self, field: &str, nonnegative: bool) -> Result<()> {
        require_finite(self.value, field)?;
        if nonnegative && self.value < 0.0 {
            bail!("{field} must be nonnegative");
        }
        require_nonempty(&self.method_id, "measurement method_id")?;
        require_sha256(&self.estimator_sha256, "estimator_sha256")?;
        require_sha256(&self.source_artifact_sha256, "source_artifact_sha256")?;
        Ok(())
    }
}

/// Path-B-defined common open-state probability estimand.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenPocketEstimand {
    /// Probability of membership in the predeclared open-state set.
    pub probability: f64,
    /// Content hash of the common estimand definition.
    pub definition_sha256: String,
    /// Native Path-B model defining state memberships.
    pub pathb_model_sha256: String,
    /// Exact Markov lag in integration steps.
    pub lag_steps: u64,
    /// Predeclared open-state identities, sorted and unique.
    pub open_state_ids: Vec<String>,
}

impl OpenPocketEstimand {
    fn validate_for(&self, pathb: &PathBProjection) -> Result<()> {
        finite_in_range(self.probability, 0.0, 1.0, "open-state probability")?;
        require_sha256(&self.definition_sha256, "open estimand definition_sha256")?;
        require_sha256(&self.pathb_model_sha256, "open estimand pathb_model_sha256")?;
        if self.pathb_model_sha256 != pathb.model_sha256 || self.lag_steps != pathb.lag_steps {
            bail!("open-state estimand model/lag is not the attached Path-B projection");
        }
        if self.open_state_ids.is_empty()
            || self.open_state_ids.iter().any(|id| id.trim().is_empty())
            || self
                .open_state_ids
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            bail!("open-state identities must be sorted, unique, and non-empty");
        }
        Ok(())
    }
}

/// Structural response measured for a Ghost/fragment state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuralResponse {
    /// Common Path-B endpoint estimand.
    pub open_state: OpenPocketEstimand,
    /// Coordinate-derived cavity volume in cubic angstrom, when measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pocket_volume_a3: Option<PhysicalMeasurement>,
    /// Coordinate-derived cavity depth in angstrom, when measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pocket_depth_a: Option<PhysicalMeasurement>,
    /// Coordinate-derived solvent-accessible area, when measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub solvent_accessible_area_a2: Option<PhysicalMeasurement>,
    /// Matched-baseline cavity-volume change, when measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delta_volume_a3: Option<PhysicalMeasurement>,
    /// Matched-baseline hydration change, when measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delta_hydration: Option<PhysicalMeasurement>,
    /// Fragment-contact fraction, when atom-contact evidence exists.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fragment_contact_fraction: Option<PhysicalMeasurement>,
}

impl StructuralResponse {
    /// Validate only measurements actually available and bind the common
    /// open-state estimand to the native Path-B model and lag.
    pub fn validate_for_pathb(&self, pathb: &PathBProjection) -> Result<()> {
        self.open_state.validate_for(pathb)?;
        for (measurement, field, nonnegative) in [
            (&self.pocket_volume_a3, "pocket_volume_a3", true),
            (&self.pocket_depth_a, "pocket_depth_a", true),
            (
                &self.solvent_accessible_area_a2,
                "solvent_accessible_area_a2",
                true,
            ),
            (&self.delta_volume_a3, "delta_volume_a3", false),
            (&self.delta_hydration, "delta_hydration", false),
            (
                &self.fragment_contact_fraction,
                "fragment_contact_fraction",
                true,
            ),
        ] {
            if let Some(measurement) = measurement {
                measurement.validate(field, nonnegative)?;
                if field == "fragment_contact_fraction" && measurement.value > 1.0 {
                    bail!("fragment_contact_fraction must be in [0, 1]");
                }
            }
        }
        Ok(())
    }
}

/// Native Path-B projection attached to a Ghost record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PathBProjection {
    /// Content hash of the native Path-B model.
    pub model_sha256: String,
    /// Physical lane whose reversible operator supplied transition kinetics.
    pub control_lane: ControlLane,
    /// Dominant metastable-state identifier.
    pub metastable_state_id: String,
    /// Probability assigned to the dominant state.
    pub state_membership_probability: f64,
    /// Optional source state for a directed transition.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transition_from_state_id: Option<String>,
    /// Optional destination state for a directed transition.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transition_to_state_id: Option<String>,
    /// Transition probability at the declared lag, present only when an exact
    /// predecessor exists for this trajectory and lag.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transition_probability: Option<f64>,
    /// Forward committor for a hash-bound target basin, when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forward_committor: Option<f64>,
    /// Path-B lag in integration steps.
    pub lag_steps: u64,
    /// Exact Ghost record hash used for the state assignment.
    pub source_ghost_record_sha256: String,
    /// Canonical hash of the complete Ghost byte-lineage occurrence.
    pub source_ghost_lineage_sha256: String,
}

impl PathBProjection {
    /// Validate normalized probability fields and source linkage.
    pub fn validate_for(&self, lineage: &GhostByteLineage) -> Result<()> {
        require_sha256(&self.model_sha256, "model_sha256")?;
        require_nonempty(&self.metastable_state_id, "metastable_state_id")?;
        finite_in_range(
            self.state_membership_probability,
            0.0,
            1.0,
            "state_membership_probability",
        )?;
        if let Some(probability) = self.transition_probability {
            finite_in_range(probability, 0.0, 1.0, "transition_probability")?;
        }
        if let Some(committor) = self.forward_committor {
            finite_in_range(committor, 0.0, 1.0, "forward_committor")?;
        }
        if self.lag_steps == 0 {
            bail!("Path-B lag_steps must be non-zero");
        }
        require_sha256(
            &self.source_ghost_record_sha256,
            "source_ghost_record_sha256",
        )?;
        require_sha256(
            &self.source_ghost_lineage_sha256,
            "source_ghost_lineage_sha256",
        )?;
        if self.source_ghost_record_sha256 != lineage.record_sha256 {
            bail!("Path-B source record hash does not match Ghost lineage");
        }
        if self.source_ghost_lineage_sha256 != lineage.lineage_sha256()? {
            bail!("Path-B source lineage hash does not match exact Ghost occurrence");
        }
        match (
            &self.transition_from_state_id,
            &self.transition_to_state_id,
            self.transition_probability,
        ) {
            (Some(from), Some(to), Some(_)) => {
                require_nonempty(from, "transition_from_state_id")?;
                require_nonempty(to, "transition_to_state_id")?;
            }
            (None, None, None) => {}
            _ => bail!(
                "Path-B transition endpoints and probability must be all present or all absent"
            ),
        }
        Ok(())
    }
}

/// Autocorrelation-aware uncertainty attached to one fiber aggregate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SamplingUncertainty {
    /// Effective independent sample count.
    pub effective_sample_size: f64,
    /// Integrated autocorrelation time in frames.
    pub integrated_autocorrelation_time_frames: f64,
    /// Number of non-overlapping blocks used by the estimator.
    pub block_count: u32,
    /// Standard error of the common open-pocket estimand.
    pub standard_error: f64,
    /// Two-sided confidence interval lower bound.
    pub confidence_interval_lower: f64,
    /// Two-sided confidence interval upper bound.
    pub confidence_interval_upper: f64,
    /// Independent replica seed.
    pub replica_seed: u64,
    /// Replica identifier within the seed.
    pub replica_id: String,
}

impl SamplingUncertainty {
    /// Validate the uncertainty and recurrence dimensions.
    pub fn validate(&self) -> Result<()> {
        finite_in_range(
            self.effective_sample_size,
            1.0,
            f64::MAX,
            "effective_sample_size",
        )?;
        finite_in_range(
            self.integrated_autocorrelation_time_frames,
            0.0,
            f64::MAX,
            "integrated_autocorrelation_time_frames",
        )?;
        if self.block_count == 0 {
            bail!("block_count must be non-zero");
        }
        finite_in_range(self.standard_error, 0.0, f64::MAX, "standard_error")?;
        require_finite(self.confidence_interval_lower, "confidence_interval_lower")?;
        require_finite(self.confidence_interval_upper, "confidence_interval_upper")?;
        if self.confidence_interval_lower > self.confidence_interval_upper {
            bail!("confidence interval is reversed");
        }
        require_nonempty(&self.replica_id, "replica_id")?;
        Ok(())
    }
}

/// Hashable payload of a chemical-state fiber.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChemicalStateFiberBody {
    /// Target/site identity shared by all matched lanes.
    pub site_key: String,
    /// Experimental control lane.
    pub control_lane: ControlLane,
    /// Exact Ghost byte lineage.
    pub ghost: GhostByteLineage,
    /// Fragment Hamiltonian state.
    pub fragment: FragmentState,
    /// Engine-owned phase binding.
    pub protocol_phase: ProtocolPhaseBinding,
    /// Thermodynamic/solvent condition.
    pub thermodynamic_condition: ThermodynamicCondition,
    /// Structural response.
    pub structural_response: StructuralResponse,
    /// Native Path-B projection.
    pub pathb: PathBProjection,
    /// Autocorrelation-aware uncertainty and seed identity.
    pub uncertainty: SamplingUncertainty,
    /// Additional typed, content-addressed evidence artifacts.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub evidence_artifact_sha256: BTreeMap<String, String>,
}

impl ChemicalStateFiberBody {
    /// Validate all dimensions and cross-field bindings.
    pub fn validate(&self) -> Result<()> {
        require_nonempty(&self.site_key, "site_key")?;
        self.ghost.validate()?;
        self.fragment.validate_for_lane(self.control_lane)?;
        self.protocol_phase.validate_for(self.ghost.key)?;
        self.thermodynamic_condition.validate()?;
        if self.protocol_phase.thermodynamic_condition_id
            != self.thermodynamic_condition.condition_id
            || self.protocol_phase.thermodynamic_condition_id != self.ghost.condition_id
            || self.protocol_phase.protocol_name != self.ghost.protocol_name
            || self.protocol_phase.phase_ordinal != self.ghost.protocol_phase
            || self.protocol_phase.phase_schedule_digest_sha256
                != self.ghost.phase_schedule_digest_sha256
            || self.protocol_phase.protocol_state_hash != self.ghost.protocol_state_hash
        {
            bail!("fiber phase/condition is not the engine-owned Ghost binding");
        }
        self.fragment.validate_runtime_lineage(
            &self.ghost,
            &self.protocol_phase,
            &self.thermodynamic_condition,
            self.control_lane,
        )?;
        self.pathb.validate_for(&self.ghost)?;
        if self.pathb.control_lane != self.control_lane {
            bail!("fiber Path-B operator lane differs from its physical control lane");
        }
        self.structural_response.validate_for_pathb(&self.pathb)?;
        self.uncertainty.validate()?;
        for (name, digest) in &self.evidence_artifact_sha256 {
            require_nonempty(name, "evidence artifact name")?;
            require_sha256(digest, "evidence artifact sha256")?;
        }
        Ok(())
    }
}

/// Canonical hash used by the native lambda endpoint event to bind the exact
/// stream phase and complete thermodynamic condition represented by a fiber.
pub fn phase_protocol_binding_sha256(
    phase: &ProtocolPhaseBinding,
    condition: &ThermodynamicCondition,
) -> Result<String> {
    canonical_sha256(&(phase, condition))
}

/// Canonical, content-addressed chemical-state fiber.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChemicalStateFiber {
    /// Schema identifier.
    pub schema_kind: String,
    /// Schema version.
    pub schema_version: String,
    /// SHA-256 of the canonical serialized body.
    pub fiber_sha256: String,
    /// Whether this fiber is at a physical endpoint.
    pub endpoint_eligibility: EndpointEligibility,
    /// Complete fiber body.
    pub body: ChemicalStateFiberBody,
}

impl ChemicalStateFiber {
    /// Materialize a validated content-addressed fiber.
    pub fn materialize(body: ChemicalStateFiberBody) -> Result<Self> {
        body.validate()?;
        let fiber_sha256 = canonical_sha256(&body)?;
        let endpoint_eligibility = body.fragment.endpoint_eligibility(body.control_lane);
        Ok(Self {
            schema_kind: CHEMICAL_STATE_FIBER_SCHEMA.to_string(),
            schema_version: CHEMICAL_STATE_FIBER_VERSION.to_string(),
            fiber_sha256,
            endpoint_eligibility,
            body,
        })
    }

    /// Recompute and verify schema and content identity.
    pub fn verify(&self) -> Result<()> {
        if self.schema_kind != CHEMICAL_STATE_FIBER_SCHEMA
            || self.schema_version != CHEMICAL_STATE_FIBER_VERSION
        {
            bail!("unsupported chemical-state fiber schema");
        }
        self.body.validate()?;
        let expected = canonical_sha256(&self.body)?;
        if self.fiber_sha256 != expected {
            bail!("chemical-state fiber content hash mismatch");
        }
        if self.endpoint_eligibility
            != self
                .body
                .fragment
                .endpoint_eligibility(self.body.control_lane)
        {
            bail!("fiber endpoint eligibility does not match lambda state");
        }
        Ok(())
    }
}

/// Ghost-side observation awaiting exact fragment and Path-B joins.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GhostFiberObservation {
    /// Target/site identity.
    pub site_key: String,
    /// Control lane.
    pub control_lane: ControlLane,
    /// Ghost lineage.
    pub ghost: GhostByteLineage,
    /// Engine-owned protocol phase.
    pub protocol_phase: ProtocolPhaseBinding,
    /// Thermodynamic condition.
    pub thermodynamic_condition: ThermodynamicCondition,
    /// Structural response.
    pub structural_response: StructuralResponse,
    /// Uncertainty and seed dimensions.
    pub uncertainty: SamplingUncertainty,
    /// Additional evidence artifact hashes.
    #[serde(default)]
    pub evidence_artifact_sha256: BTreeMap<String, String>,
}

/// Schema identifier for the immutable post-persistence Ghost/lambda join.
pub const LAMBDA_GHOST_ENDPOINT_JOIN_SCHEMA: &str = "prism_lambda_ghost_endpoint_join";
/// Schema version for the immutable post-persistence Ghost/lambda join.
pub const LAMBDA_GHOST_ENDPOINT_JOIN_VERSION: u32 = 1;

/// Whether an exact post-persistence Ghost/lambda relationship is admissible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum LambdaGhostEndpointJoinStatus {
    /// All exact coordinate, endpoint, force, byte, phase, condition and lane
    /// gates passed. This does not itself establish a pocket claim.
    Evaluated,
    /// The source evidence is retained but cannot enter endpoint-state fibers.
    NotEvaluable,
}

/// Typed reason an immutable input could not enter an endpoint-state fiber.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum LambdaGhostEndpointNotEvaluableReason {
    /// Ghost bytes or their segment/manifest lineage failed validation.
    GhostByteLineageInvalid,
    /// A valid Ghost node had no lambda observation at the exact stream/step.
    UnmatchedGhostNode,
    /// A lambda endpoint had no Ghost node at the exact stream/step.
    UnmatchedLambdaEndpoint,
    /// A partial-lambda state is sampling evidence, not physical-endpoint evidence.
    PartialLambdaSamplingState,
    /// A physical endpoint force population lacked an endpoint observation.
    MissingEndpointObservation,
    /// The endpoint event lacked one exact persisted byte-lineage row.
    MissingEndpointByteLineage,
    /// The endpoint lacked its exact supporting physical force event.
    MissingSupportingForceEvent,
    /// The supporting force lacked one exact persisted byte-lineage row.
    MissingSupportingForceByteLineage,
    /// No unique persisted phase/condition context covered the exact step.
    MissingProtocolContext,
    /// Lambda and Ghost lane identities differ.
    LaneMismatch,
    /// Lambda and Ghost phase identities or intervals differ.
    ProtocolPhaseMismatch,
    /// Lambda and Ghost complete thermodynamic conditions differ.
    ThermodynamicConditionMismatch,
    /// A pre-existing endpoint Ghost digest contradicts the exact joined record.
    GhostEndpointLineageMismatch,
}

/// Exact Ghost node and its engine-owned experimental context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LambdaGhostJoinGhostNode {
    /// Target/site identity retained from the lattice materializer.
    pub site_key: String,
    /// Matched experimental lane assigned to this Ghost stream.
    pub control_lane: ControlLane,
    /// Exactly-once Ghost segment and byte lineage.
    pub ghost: GhostByteLineage,
    /// Engine-owned phase covering this node.
    pub protocol_phase: ProtocolPhaseBinding,
    /// Complete physical condition covering this node.
    pub thermodynamic_condition: ThermodynamicCondition,
}

impl From<&GhostFiberObservation> for LambdaGhostJoinGhostNode {
    fn from(value: &GhostFiberObservation) -> Self {
        Self {
            site_key: value.site_key.clone(),
            control_lane: value.control_lane,
            ghost: value.ghost.clone(),
            protocol_phase: value.protocol_phase.clone(),
            thermodynamic_condition: value.thermodynamic_condition.clone(),
        }
    }
}

/// One immutable exact-coordinate join or one retained non-evaluable input.
/// No frame tolerance, nearest-frame search, interpolation or cross-stream
/// association is permitted by this schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LambdaGhostEndpointJoin {
    /// Schema discriminator.
    pub schema_kind: String,
    /// Schema version.
    pub schema_version: u32,
    /// Constant algorithm identity: `EXACT_SAME_STREAM_SAME_STEP`.
    pub join_method: String,
    /// Scientific admission status.
    pub status: LambdaGhostEndpointJoinStatus,
    /// Typed non-evaluable reason, absent only for evaluated rows.
    pub not_evaluable_reason: Option<LambdaGhostEndpointNotEvaluableReason>,
    /// Exact Ghost node, absent only for an unmatched/partial lambda input.
    pub ghost_node: Option<LambdaGhostJoinGhostNode>,
    /// Exact malformed Ghost input retained without admitting its invalid byte
    /// lineage as scientific evidence.
    pub retained_invalid_ghost_node: Option<LambdaGhostJoinGhostNode>,
    /// Canonical digest of the retained malformed Ghost input.
    pub retained_invalid_ghost_input_sha256: Option<String>,
    /// Canonical complete Ghost occurrence digest.
    pub ghost_lineage_sha256: Option<String>,
    /// Exact Ghost record-byte digest (not a derived node feature digest).
    pub ghost_record_sha256: Option<String>,
    /// Exact owning Ghost segment-manifest digest.
    pub ghost_segment_manifest_sha256: Option<String>,
    /// Exact immutable Ghost segment digest.
    pub ghost_segment_sha256: Option<String>,
    /// Exact record byte offset within the immutable Ghost segment.
    pub ghost_byte_offset: Option<u64>,
    /// Exact record byte length within the immutable Ghost segment.
    pub ghost_byte_length: Option<u32>,
    /// Lambda stream/step/slot coordinate, absent only for a Ghost-only row.
    pub lambda_key: Option<LambdaJournalKey>,
    /// Complete immutable matched-lane contract.
    pub lambda_lane: Option<LambdaExperimentLaneContract>,
    /// Exact physical endpoint observation.
    pub endpoint: Option<LambdaFragmentEndpointObservation>,
    /// Endpoint event byte/segment/emission-receipt lineage.
    pub endpoint_lineage: Option<LambdaJournalLoadedLineage>,
    /// Exact supporting physical force event named by the endpoint.
    pub supporting_force: Option<LambdaFragmentForceEvent>,
    /// Supporting force byte/segment/emission-receipt lineage.
    pub supporting_force_lineage: Option<LambdaJournalLoadedLineage>,
    /// Ordered force states retained when no physical endpoint observation is
    /// available. Partial-lambda rows remain sampling evidence only.
    pub sampling_force_events: Vec<LambdaFragmentForceEvent>,
    /// Exact persisted byte lineage bijective with `sampling_force_events`.
    pub sampling_force_lineage: Vec<LambdaJournalLoadedLineage>,
    /// Collision-free NCMC move/operation/boundary identity.
    pub supporting_force_boundary: Option<NcmcForceBoundaryIdentity>,
    /// Persisted phase/condition binding digest at this lambda step.
    pub phase_protocol_sha256: Option<String>,
    /// Persisted engine-owned phase at this lambda step.
    pub protocol_phase: Option<ProtocolPhaseBinding>,
    /// Persisted complete thermodynamic condition at this lambda step.
    pub thermodynamic_condition: Option<ThermodynamicCondition>,
    /// SHA-256 of every preceding field with this field cleared.
    pub join_sha256: String,
}

impl LambdaGhostEndpointJoin {
    fn seal(mut self) -> Result<Self> {
        self.join_sha256.clear();
        self.join_sha256 = canonical_sha256(&self)?;
        self.verify()?;
        Ok(self)
    }

    /// Verify exact immutable coordinates, receipts, byte lineage and context.
    pub fn verify(&self) -> Result<()> {
        if self.schema_kind != LAMBDA_GHOST_ENDPOINT_JOIN_SCHEMA
            || self.schema_version != LAMBDA_GHOST_ENDPOINT_JOIN_VERSION
            || self.join_method != "EXACT_SAME_STREAM_SAME_STEP"
        {
            bail!("invalid LambdaGhostEndpointJoin schema or join method");
        }
        match (self.status, self.not_evaluable_reason) {
            (LambdaGhostEndpointJoinStatus::Evaluated, None) => {}
            (LambdaGhostEndpointJoinStatus::NotEvaluable, Some(_)) => {}
            _ => bail!("lambda/Ghost join status and reason disagree"),
        }
        match (&self.lambda_key, &self.lambda_lane) {
            (Some(_), Some(lane)) => validate_join_lane_contract(lane)?,
            (None, None) => {}
            _ => bail!("lambda coordinate and lane contract must co-exist"),
        }
        if let Some(node) = &self.ghost_node {
            node.ghost.validate()?;
            let lineage = node.ghost.lineage_sha256()?;
            if self.ghost_lineage_sha256.as_deref() != Some(lineage.as_str())
                || self.ghost_record_sha256.as_deref() != Some(node.ghost.record_sha256.as_str())
                || self.ghost_segment_manifest_sha256.as_deref()
                    != Some(node.ghost.segment_manifest_sha256.as_str())
                || self.ghost_segment_sha256.as_deref() != Some(node.ghost.segment_sha256.as_str())
                || self.ghost_byte_offset != Some(node.ghost.byte_offset)
                || self.ghost_byte_length != Some(node.ghost.byte_length)
            {
                bail!("explicit Ghost byte lineage differs from joined Ghost node");
            }
        } else if self.ghost_lineage_sha256.is_some()
            || self.ghost_record_sha256.is_some()
            || self.ghost_segment_manifest_sha256.is_some()
            || self.ghost_segment_sha256.is_some()
            || self.ghost_byte_offset.is_some()
            || self.ghost_byte_length.is_some()
        {
            bail!("Ghost byte fields exist without a Ghost node");
        }
        match (
            &self.retained_invalid_ghost_node,
            &self.retained_invalid_ghost_input_sha256,
        ) {
            (Some(node), Some(digest))
                if self.status == LambdaGhostEndpointJoinStatus::NotEvaluable
                    && self.not_evaluable_reason
                        == Some(LambdaGhostEndpointNotEvaluableReason::GhostByteLineageInvalid)
                    && self.ghost_node.is_none() =>
            {
                require_sha256(digest, "retained invalid Ghost input")?;
                if canonical_sha256(node)? != *digest {
                    bail!("retained invalid Ghost input digest mismatch");
                }
            }
            (None, None) => {}
            _ => bail!("retained invalid Ghost fields are incomplete or ineligible"),
        }
        if let Some(endpoint) = &self.endpoint {
            endpoint.validate().map_err(anyhow::Error::msg)?;
            let key = self
                .lambda_key
                .context("endpoint join lacks lambda coordinate")?;
            if u16::try_from(endpoint.stream_id).ok() != Some(key.stream_id)
                || endpoint.step != key.step_idx
                || endpoint.slot_id != key.slot_id
            {
                bail!("endpoint identity differs from lambda join coordinate");
            }
            if let Some(lineage) = &self.endpoint_lineage {
                validate_loaded_lambda_lineage(lineage)?;
                if lineage.event_kind != LambdaJournalEventKind::Endpoint
                    || lineage.event_sequence != endpoint.event_sequence
                    || lineage.raw_event_sha256 != canonical_event_line_sha256(endpoint)?
                    || lineage.parent_receipt_sha256.as_deref()
                        != Some(endpoint.parent_receipt_sha256.as_str())
                {
                    bail!("endpoint byte lineage names a different event");
                }
            } else if self.not_evaluable_reason
                != Some(LambdaGhostEndpointNotEvaluableReason::MissingEndpointByteLineage)
            {
                bail!("endpoint join lacks endpoint byte lineage");
            }
        } else if self.endpoint_lineage.is_some() {
            bail!("endpoint byte lineage exists without an endpoint event");
        }
        if let Some(force) = &self.supporting_force {
            force.validate().map_err(anyhow::Error::msg)?;
            if let Some(lineage) = &self.supporting_force_lineage {
                validate_loaded_lambda_lineage(lineage)?;
                if lineage.event_kind != LambdaJournalEventKind::Force
                    || lineage.event_sequence != force.event_sequence
                    || lineage.raw_event_sha256 != canonical_event_line_sha256(force)?
                    || lineage.parent_receipt_sha256.as_deref()
                        != force.parent_receipt_sha256.as_deref()
                    || self.supporting_force_boundary.as_ref() != force.ncmc_boundary.as_ref()
                {
                    bail!("supporting force lineage or boundary differs from force event");
                }
                if let Some(endpoint) = &self.endpoint {
                    if lineage.raw_event_sha256 != endpoint.supporting_force_event_sha256
                        || lineage.emission_receipt_sha256
                            != endpoint.supporting_force_emission_receipt_sha256
                        || force.ncmc_boundary.as_ref() != Some(&endpoint.supporting_force_boundary)
                    {
                        bail!("endpoint does not name the exact supporting force bytes/receipt");
                    }
                }
            } else if self.not_evaluable_reason
                != Some(LambdaGhostEndpointNotEvaluableReason::MissingSupportingForceByteLineage)
            {
                bail!("supporting force lacks byte lineage");
            }
        } else if self.supporting_force_lineage.is_some()
            || self.supporting_force_boundary.is_some()
        {
            bail!("supporting-force lineage exists without a supporting force");
        }
        if self.endpoint.is_some()
            && (!self.sampling_force_events.is_empty() || !self.sampling_force_lineage.is_empty())
        {
            bail!("endpoint join cannot also carry sampling-only force population");
        }
        if self.sampling_force_events.len() != self.sampling_force_lineage.len()
            && self.not_evaluable_reason
                != Some(LambdaGhostEndpointNotEvaluableReason::MissingSupportingForceByteLineage)
        {
            bail!("sampling force population and byte lineage are not bijective");
        }
        let mut sampling_sequences = BTreeSet::new();
        for force in &self.sampling_force_events {
            force.validate().map_err(anyhow::Error::msg)?;
            let lineage_matches: Vec<_> = self
                .sampling_force_lineage
                .iter()
                .filter(|lineage| lineage.event_sequence == force.event_sequence)
                .collect();
            if !sampling_sequences.insert(force.event_sequence) || lineage_matches.len() > 1 {
                bail!("sampling force population does not match exact persisted bytes");
            }
            if let Some(lineage) = lineage_matches.first() {
                if lineage.event_kind != LambdaJournalEventKind::Force
                    || lineage.raw_event_sha256 != canonical_event_line_sha256(force)?
                    || lineage.parent_receipt_sha256.as_deref()
                        != force.parent_receipt_sha256.as_deref()
                {
                    bail!("sampling force population does not match exact persisted bytes");
                }
                validate_loaded_lambda_lineage(lineage)?;
            } else if self.not_evaluable_reason
                != Some(LambdaGhostEndpointNotEvaluableReason::MissingSupportingForceByteLineage)
            {
                bail!("sampling force event lacks persisted byte lineage");
            }
            let key = self
                .lambda_key
                .context("sampling force row lacks lambda coordinate")?;
            if u16::try_from(force.stream_id).ok() != Some(key.stream_id)
                || force.step != key.step_idx
                || force.slot_id != key.slot_id
            {
                bail!("sampling force differs from lambda join coordinate");
            }
        }
        if self
            .sampling_force_lineage
            .iter()
            .any(|lineage| !sampling_sequences.contains(&lineage.event_sequence))
        {
            bail!("sampling lineage contains an extra force event");
        }
        if self.not_evaluable_reason
            == Some(LambdaGhostEndpointNotEvaluableReason::PartialLambdaSamplingState)
            && (self.sampling_force_events.is_empty()
                || self
                    .sampling_force_events
                    .iter()
                    .any(|force| force.lambda == 0.0 || force.lambda == 1.0))
        {
            bail!("partial-lambda row contains no partial state or a physical endpoint");
        }
        if self.status == LambdaGhostEndpointJoinStatus::Evaluated {
            let node = self
                .ghost_node
                .as_ref()
                .context("evaluated join lacks Ghost node")?;
            let endpoint = self
                .endpoint
                .as_ref()
                .context("evaluated join lacks endpoint")?;
            let force = self
                .supporting_force
                .as_ref()
                .context("evaluated join lacks supporting force")?;
            let lane = self
                .lambda_lane
                .as_ref()
                .context("evaluated join lacks lane contract")?;
            validate_join_lane_contract(lane)?;
            let phase = self
                .protocol_phase
                .as_ref()
                .context("evaluated join lacks protocol phase")?;
            let condition = self
                .thermodynamic_condition
                .as_ref()
                .context("evaluated join lacks thermodynamic condition")?;
            if endpoint.lambda_endpoint as f64 != force.lambda
                || !matches!(force.lambda, 0.0 | 1.0)
                || node.ghost.key.stream_id != self.lambda_key.unwrap().stream_id
                || node.ghost.key.step_idx != self.lambda_key.unwrap().step_idx
                || control_lane_from_lambda(lane.lane_kind) != node.control_lane
                || phase != &node.protocol_phase
                || condition != &node.thermodynamic_condition
            {
                bail!("evaluated join violates endpoint, coordinate, lane or context equality");
            }
            let expected_phase = phase_protocol_binding_sha256(phase, condition)?;
            if self.phase_protocol_sha256.as_deref() != Some(expected_phase.as_str())
                || endpoint.phase_protocol_sha256 != expected_phase
                || force.phase_protocol_sha256 != expected_phase
            {
                bail!("evaluated join phase/condition digest mismatch");
            }
            if let Some(existing) = &endpoint.ghost_lineage_sha256 {
                if self.ghost_lineage_sha256.as_ref() != Some(existing) {
                    bail!("endpoint pre-bound Ghost lineage contradicts exact join");
                }
            }
        }
        let mut projection = self.clone();
        projection.join_sha256.clear();
        if self.join_sha256 != canonical_sha256(&projection)? {
            bail!("LambdaGhostEndpointJoin content hash mismatch");
        }
        Ok(())
    }
}

/// Population-conserving post-persistence join result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LambdaGhostEndpointJoinMaterialization {
    /// Schema discriminator.
    pub schema_kind: String,
    /// Schema version.
    pub schema_version: u32,
    /// Ghost node population presented to the join.
    pub ghost_input_count: u64,
    /// Lambda exact-key population presented by the verified loader.
    pub lambda_observation_input_count: u64,
    /// Loader population conservation authority.
    pub lambda_population_receipt_sha256: String,
    /// Number of evaluated exact endpoint joins.
    pub evaluated_join_count: u64,
    /// Number of retained typed non-evaluable rows.
    pub not_evaluable_count: u64,
    /// Deterministically ordered evaluated and retained rows.
    pub rows: Vec<LambdaGhostEndpointJoin>,
    /// Digest over every preceding field with this field cleared.
    pub materialization_sha256: String,
}

impl LambdaGhostEndpointJoinMaterialization {
    /// Verify every row and population count.
    pub fn verify(&self) -> Result<()> {
        if self.schema_kind != "prism_lambda_ghost_endpoint_join_materialization"
            || self.schema_version != 1
        {
            bail!("invalid lambda/Ghost join materialization schema");
        }
        require_sha256(
            &self.lambda_population_receipt_sha256,
            "lambda population receipt",
        )?;
        let evaluated = self
            .rows
            .iter()
            .filter(|row| row.status == LambdaGhostEndpointJoinStatus::Evaluated)
            .count() as u64;
        let not_evaluable = self.rows.len() as u64 - evaluated;
        if evaluated != self.evaluated_join_count || not_evaluable != self.not_evaluable_count {
            bail!("lambda/Ghost join materialization counts do not close");
        }
        let mut identities = BTreeSet::new();
        let mut represented_ghosts = BTreeSet::new();
        let mut represented_lambda_keys = BTreeSet::new();
        for row in &self.rows {
            row.verify()?;
            if !identities.insert(row.join_sha256.clone()) {
                bail!("duplicate immutable lambda/Ghost join row");
            }
            if let Some(lineage) = &row.ghost_lineage_sha256 {
                represented_ghosts.insert(lineage.clone());
            } else if let Some(digest) = &row.retained_invalid_ghost_input_sha256 {
                represented_ghosts.insert(digest.clone());
            }
            if let Some(key) = row.lambda_key {
                represented_lambda_keys.insert(key);
            }
        }
        if represented_ghosts.len() as u64 != self.ghost_input_count
            || represented_lambda_keys.len() as u64 != self.lambda_observation_input_count
        {
            bail!("lambda/Ghost join failed input-population representation conservation");
        }
        let mut projection = self.clone();
        projection.materialization_sha256.clear();
        if self.materialization_sha256 != canonical_sha256(&projection)? {
            bail!("lambda/Ghost join materialization hash mismatch");
        }
        Ok(())
    }
}

fn control_lane_from_lambda(lane: LambdaExperimentLaneKind) -> ControlLane {
    match lane {
        LambdaExperimentLaneKind::NoProbe => ControlLane::NoProbe,
        LambdaExperimentLaneKind::HypothesisFragments => ControlLane::HypothesisFragment,
        LambdaExperimentLaneKind::OrthogonalChemicalControls => {
            ControlLane::OrthogonalChemicalControl
        }
    }
}

fn endpoint_lineage<'a>(
    observation: &'a LambdaJournalObservation,
    kind: LambdaJournalEventKind,
    sequence: u64,
) -> Result<Option<&'a LambdaJournalLoadedLineage>> {
    let matches: Vec<_> = observation
        .event_lineage
        .iter()
        .filter(|lineage| lineage.event_kind == kind && lineage.event_sequence == sequence)
        .collect();
    match matches.as_slice() {
        [] => Ok(None),
        [lineage] => Ok(Some(*lineage)),
        _ => bail!("duplicate lambda event byte lineage for one exact event"),
    }
}

fn canonical_event_line_sha256<T: Serialize>(event: &T) -> Result<String> {
    let mut bytes = serde_json::to_vec(event).context("serialize lambda event bytes")?;
    bytes.push(b'\n');
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn validate_loaded_lambda_lineage(lineage: &LambdaJournalLoadedLineage) -> Result<()> {
    for (value, label) in [
        (&lineage.emission_receipt_sha256, "lambda emission receipt"),
        (&lineage.raw_event_sha256, "lambda raw event"),
        (&lineage.segment_manifest_sha256, "lambda segment manifest"),
        (&lineage.segment_sha256, "lambda segment"),
    ] {
        require_sha256(value, label)?;
    }
    if let Some(parent) = &lineage.parent_receipt_sha256 {
        require_sha256(parent, "lambda parent receipt")?;
    }
    let segment = std::path::Path::new(&lineage.segment_file);
    if lineage.byte_length == 0
        || segment.is_absolute()
        || segment.components().any(|component| {
            matches!(
                component,
                std::path::Component::CurDir | std::path::Component::ParentDir
            )
        })
        || lineage
            .byte_offset
            .checked_add(lineage.byte_length)
            .is_none()
    {
        bail!("invalid persisted lambda byte lineage");
    }
    Ok(())
}

fn validate_join_lane_contract(lane: &LambdaExperimentLaneContract) -> Result<()> {
    require_nonempty(&lane.lane_id, "lambda lane id")?;
    require_sha256(&lane.contract_sha256, "lambda lane contract")?;
    let mut projection = lane.clone();
    projection.contract_sha256.clear();
    if lane.schema_kind != "prismos_lambda_experiment_lane_contract"
        || lane.schema_version != 1
        || lane.virtual_probe_count != 0
        || lane.contract_sha256 != canonical_sha256(&projection)?
    {
        bail!("invalid lambda lane contract identity");
    }
    let mut species = lane.fragment_species_ids.clone();
    species.sort_unstable();
    species.dedup();
    if species != lane.fragment_species_ids
        || (lane.lane_kind == LambdaExperimentLaneKind::NoProbe && !species.is_empty())
        || (lane.lane_kind != LambdaExperimentLaneKind::NoProbe && species.is_empty())
    {
        bail!("lambda lane species population is not canonical");
    }
    Ok(())
}

fn make_lambda_ghost_join(
    status: LambdaGhostEndpointJoinStatus,
    reason: Option<LambdaGhostEndpointNotEvaluableReason>,
    ghost: Option<&LambdaGhostJoinGhostNode>,
    key: Option<LambdaJournalKey>,
    observation: Option<&LambdaJournalObservation>,
    endpoint: Option<&LambdaFragmentEndpointObservation>,
    endpoint_lineage: Option<&LambdaJournalLoadedLineage>,
    force: Option<&LambdaFragmentForceEvent>,
    force_lineage: Option<&LambdaJournalLoadedLineage>,
    sampling_force_events: &[LambdaFragmentForceEvent],
    sampling_force_lineage: &[LambdaJournalLoadedLineage],
    phase: Option<&ProtocolPhaseBinding>,
    condition: Option<&ThermodynamicCondition>,
) -> Result<LambdaGhostEndpointJoin> {
    let (ghost_node, ghost_lineage_sha256, record, manifest, segment, offset, length) =
        if let Some(ghost) = ghost {
            let lineage = ghost.ghost.lineage_sha256()?;
            (
                Some(ghost.clone()),
                Some(lineage),
                Some(ghost.ghost.record_sha256.clone()),
                Some(ghost.ghost.segment_manifest_sha256.clone()),
                Some(ghost.ghost.segment_sha256.clone()),
                Some(ghost.ghost.byte_offset),
                Some(ghost.ghost.byte_length),
            )
        } else {
            (None, None, None, None, None, None, None)
        };
    LambdaGhostEndpointJoin {
        schema_kind: LAMBDA_GHOST_ENDPOINT_JOIN_SCHEMA.to_string(),
        schema_version: LAMBDA_GHOST_ENDPOINT_JOIN_VERSION,
        join_method: "EXACT_SAME_STREAM_SAME_STEP".to_string(),
        status,
        not_evaluable_reason: reason,
        ghost_node,
        retained_invalid_ghost_node: None,
        retained_invalid_ghost_input_sha256: None,
        ghost_lineage_sha256,
        ghost_record_sha256: record,
        ghost_segment_manifest_sha256: manifest,
        ghost_segment_sha256: segment,
        ghost_byte_offset: offset,
        ghost_byte_length: length,
        lambda_key: key,
        lambda_lane: observation.map(|value| value.lane.clone()),
        endpoint: endpoint.cloned(),
        endpoint_lineage: endpoint_lineage.cloned(),
        supporting_force: force.cloned(),
        supporting_force_lineage: force_lineage.cloned(),
        sampling_force_events: sampling_force_events.to_vec(),
        sampling_force_lineage: sampling_force_lineage.to_vec(),
        supporting_force_boundary: force.and_then(|value| value.ncmc_boundary.clone()),
        phase_protocol_sha256: endpoint
            .map(|value| value.phase_protocol_sha256.clone())
            .or_else(|| force.map(|value| value.phase_protocol_sha256.clone())),
        protocol_phase: phase.cloned(),
        thermodynamic_condition: condition.cloned(),
        join_sha256: String::new(),
    }
    .seal()
}

/// Join verified post-persistence lambda endpoint journals to lossless Ghost
/// nodes. Matching is exact `(stream_id, step_idx)` only. Every valid endpoint
/// joins every distinct lattice site/node at that exact coordinate; slot
/// identity remains explicit. No nearest-frame fallback exists.
pub fn materialize_lambda_ghost_endpoint_joins(
    ghost_observations: &[GhostFiberObservation],
    lambda: &LambdaEventJournalLoad,
) -> Result<LambdaGhostEndpointJoinMaterialization> {
    let ghost_nodes = ghost_observations
        .iter()
        .map(LambdaGhostJoinGhostNode::from)
        .collect::<Vec<_>>();
    materialize_lambda_ghost_endpoint_joins_from_nodes(&ghost_nodes, lambda)
}

/// Core pre-Path-B join entry point. It accepts only exact Ghost node/context
/// state, so callers never fabricate an open-state response before Path-B has
/// fitted a model and defined the open basin.
pub fn materialize_lambda_ghost_endpoint_joins_from_nodes(
    ghost_nodes: &[LambdaGhostJoinGhostNode],
    lambda: &LambdaEventJournalLoad,
) -> Result<LambdaGhostEndpointJoinMaterialization> {
    if lambda.produced != lambda.persisted
        || lambda.produced != lambda.decoded
        || lambda.produced != lambda.indexed
    {
        bail!("lambda journal population violates produced=persisted=decoded=indexed");
    }
    require_sha256(
        &lambda.population_receipt_sha256,
        "lambda population receipt",
    )?;

    let mut rows = Vec::new();
    let mut valid_ghosts = Vec::new();
    let mut ghost_identity = BTreeSet::new();
    for ghost in ghost_nodes {
        match ghost.ghost.lineage_sha256() {
            Ok(lineage) => {
                if !ghost_identity.insert(lineage) {
                    bail!("duplicate exact Ghost lineage in endpoint join input");
                }
                valid_ghosts.push(ghost);
            }
            Err(_) => {
                // Preserve the input in typed form without fabricating byte
                // coordinates or a content digest that failed validation.
                let node = ghost.clone();
                let invalid = LambdaGhostEndpointJoin {
                    schema_kind: LAMBDA_GHOST_ENDPOINT_JOIN_SCHEMA.to_string(),
                    schema_version: LAMBDA_GHOST_ENDPOINT_JOIN_VERSION,
                    join_method: "EXACT_SAME_STREAM_SAME_STEP".to_string(),
                    status: LambdaGhostEndpointJoinStatus::NotEvaluable,
                    not_evaluable_reason: Some(
                        LambdaGhostEndpointNotEvaluableReason::GhostByteLineageInvalid,
                    ),
                    ghost_node: None,
                    retained_invalid_ghost_input_sha256: Some(canonical_sha256(&node)?),
                    retained_invalid_ghost_node: Some(node.clone()),
                    ghost_lineage_sha256: None,
                    ghost_record_sha256: None,
                    ghost_segment_manifest_sha256: None,
                    ghost_segment_sha256: None,
                    ghost_byte_offset: None,
                    ghost_byte_length: None,
                    lambda_key: None,
                    lambda_lane: None,
                    endpoint: None,
                    endpoint_lineage: None,
                    supporting_force: None,
                    supporting_force_lineage: None,
                    sampling_force_events: Vec::new(),
                    sampling_force_lineage: Vec::new(),
                    supporting_force_boundary: None,
                    phase_protocol_sha256: None,
                    protocol_phase: Some(node.protocol_phase.clone()),
                    thermodynamic_condition: Some(node.thermodynamic_condition.clone()),
                    join_sha256: String::new(),
                };
                rows.push(invalid.seal()?);
            }
        }
    }

    let mut matched_ghost_lineages = BTreeSet::new();
    for (key, observation) in &lambda.observations {
        if observation.key != *key {
            bail!("lambda observation map key differs from observation key");
        }
        let coordinate_ghosts: Vec<_> = valid_ghosts
            .iter()
            .copied()
            .filter(|ghost| {
                ghost.ghost.key.stream_id == key.stream_id
                    && ghost.ghost.key.step_idx == key.step_idx
            })
            .collect();
        let step_contexts: Vec<_> = lambda
            .stream_phase_contexts
            .get(&key.stream_id)
            .into_iter()
            .flatten()
            .filter(|context| {
                key.step_idx >= context.protocol_phase.start_step
                    && key.step_idx < context.protocol_phase.end_step_exclusive
            })
            .collect();
        if step_contexts.len() > 1 {
            bail!("multiple lambda protocol contexts cover one exact stream/step");
        }
        let context = step_contexts.first().copied();
        let Some(endpoint) = observation.endpoint_event.as_ref() else {
            let mut sampling_lineage = Vec::with_capacity(observation.force_events.len());
            let mut missing_sampling_lineage = false;
            for force in &observation.force_events {
                match endpoint_lineage(
                    observation,
                    LambdaJournalEventKind::Force,
                    force.event_sequence,
                )? {
                    Some(lineage) => sampling_lineage.push(lineage.clone()),
                    None => missing_sampling_lineage = true,
                }
            }
            let reason = if missing_sampling_lineage {
                LambdaGhostEndpointNotEvaluableReason::MissingSupportingForceByteLineage
            } else if observation.force_events.is_empty() {
                LambdaGhostEndpointNotEvaluableReason::MissingSupportingForceEvent
            } else if observation
                .force_events
                .iter()
                .all(|force| force.lambda != 0.0 && force.lambda != 1.0)
            {
                LambdaGhostEndpointNotEvaluableReason::PartialLambdaSamplingState
            } else {
                LambdaGhostEndpointNotEvaluableReason::MissingEndpointObservation
            };
            rows.push(make_lambda_ghost_join(
                LambdaGhostEndpointJoinStatus::NotEvaluable,
                Some(reason),
                None,
                Some(*key),
                Some(observation),
                None,
                None,
                None,
                None,
                &observation.force_events,
                &sampling_lineage,
                context.map(|value| &value.protocol_phase),
                context.map(|value| &value.thermodynamic_condition),
            )?);
            continue;
        };
        endpoint.validate().map_err(anyhow::Error::msg)?;
        let endpoint_bytes = endpoint_lineage(
            observation,
            LambdaJournalEventKind::Endpoint,
            endpoint.event_sequence,
        )?;
        let supporting_force_matches: Vec<_> = observation
            .force_events
            .iter()
            .filter(|force| {
                force.ncmc_boundary.as_ref() == Some(&endpoint.supporting_force_boundary)
                    && force.slot_id == endpoint.slot_id
                    && force.stream_id == endpoint.stream_id
                    && force.step == endpoint.step
            })
            .collect();
        if supporting_force_matches.len() > 1 {
            bail!("endpoint supporting force boundary is duplicated");
        }
        let supporting_force = supporting_force_matches.first().copied();
        let force_bytes = if let Some(force) = supporting_force {
            let lineage = endpoint_lineage(
                observation,
                LambdaJournalEventKind::Force,
                force.event_sequence,
            )?;
            if let Some(lineage) = lineage {
                if lineage.raw_event_sha256 != endpoint.supporting_force_event_sha256
                    || lineage.emission_receipt_sha256
                        != endpoint.supporting_force_emission_receipt_sha256
                {
                    bail!("endpoint supporting-force receipt contradicts persisted byte lineage");
                }
            }
            lineage
        } else {
            None
        };
        let base_reason = if endpoint_bytes.is_none() {
            Some(LambdaGhostEndpointNotEvaluableReason::MissingEndpointByteLineage)
        } else if supporting_force.is_none() {
            Some(LambdaGhostEndpointNotEvaluableReason::MissingSupportingForceEvent)
        } else if force_bytes.is_none() {
            Some(LambdaGhostEndpointNotEvaluableReason::MissingSupportingForceByteLineage)
        } else if context.is_none() {
            Some(LambdaGhostEndpointNotEvaluableReason::MissingProtocolContext)
        } else if context.unwrap().phase_protocol_sha256 != endpoint.phase_protocol_sha256 {
            Some(LambdaGhostEndpointNotEvaluableReason::ProtocolPhaseMismatch)
        } else {
            None
        };

        if coordinate_ghosts.is_empty() {
            rows.push(make_lambda_ghost_join(
                LambdaGhostEndpointJoinStatus::NotEvaluable,
                Some(
                    base_reason
                        .unwrap_or(LambdaGhostEndpointNotEvaluableReason::UnmatchedLambdaEndpoint),
                ),
                None,
                Some(*key),
                Some(observation),
                Some(endpoint),
                endpoint_bytes,
                supporting_force,
                force_bytes,
                &[],
                &[],
                context.map(|value| &value.protocol_phase),
                context.map(|value| &value.thermodynamic_condition),
            )?);
            continue;
        }

        for ghost in coordinate_ghosts {
            let ghost_lineage = ghost.ghost.lineage_sha256()?;
            matched_ghost_lineages.insert(ghost_lineage.clone());
            let reason = base_reason.or_else(|| {
                let context = context.unwrap();
                if control_lane_from_lambda(observation.lane.lane_kind) != ghost.control_lane {
                    Some(LambdaGhostEndpointNotEvaluableReason::LaneMismatch)
                } else if context.protocol_phase != ghost.protocol_phase {
                    Some(LambdaGhostEndpointNotEvaluableReason::ProtocolPhaseMismatch)
                } else if context.thermodynamic_condition != ghost.thermodynamic_condition {
                    Some(LambdaGhostEndpointNotEvaluableReason::ThermodynamicConditionMismatch)
                } else if endpoint
                    .ghost_lineage_sha256
                    .as_ref()
                    .is_some_and(|bound| bound != &ghost_lineage)
                {
                    Some(LambdaGhostEndpointNotEvaluableReason::GhostEndpointLineageMismatch)
                } else {
                    None
                }
            });
            rows.push(make_lambda_ghost_join(
                if reason.is_none() {
                    LambdaGhostEndpointJoinStatus::Evaluated
                } else {
                    LambdaGhostEndpointJoinStatus::NotEvaluable
                },
                reason,
                Some(ghost),
                Some(*key),
                Some(observation),
                Some(endpoint),
                endpoint_bytes,
                supporting_force,
                force_bytes,
                &[],
                &[],
                context.map(|value| &value.protocol_phase),
                context.map(|value| &value.thermodynamic_condition),
            )?);
        }
    }

    for ghost in valid_ghosts {
        let lineage = ghost.ghost.lineage_sha256()?;
        if !matched_ghost_lineages.contains(&lineage) {
            rows.push(make_lambda_ghost_join(
                LambdaGhostEndpointJoinStatus::NotEvaluable,
                Some(LambdaGhostEndpointNotEvaluableReason::UnmatchedGhostNode),
                Some(ghost),
                None,
                None,
                None,
                None,
                None,
                None,
                &[],
                &[],
                Some(&ghost.protocol_phase),
                Some(&ghost.thermodynamic_condition),
            )?);
        }
    }
    rows.sort_by(|a, b| {
        (
            a.ghost_node.as_ref().map(|node| node.ghost.key),
            a.lambda_key,
            a.status as u8,
            a.not_evaluable_reason,
            &a.join_sha256,
        )
            .cmp(&(
                b.ghost_node.as_ref().map(|node| node.ghost.key),
                b.lambda_key,
                b.status as u8,
                b.not_evaluable_reason,
                &b.join_sha256,
            ))
    });
    let evaluated_join_count = rows
        .iter()
        .filter(|row| row.status == LambdaGhostEndpointJoinStatus::Evaluated)
        .count() as u64;
    let mut output = LambdaGhostEndpointJoinMaterialization {
        schema_kind: "prism_lambda_ghost_endpoint_join_materialization".to_string(),
        schema_version: 1,
        ghost_input_count: ghost_nodes.len() as u64,
        lambda_observation_input_count: lambda.observations.len() as u64,
        lambda_population_receipt_sha256: lambda.population_receipt_sha256.clone(),
        evaluated_join_count,
        not_evaluable_count: rows.len() as u64 - evaluated_join_count,
        rows,
        materialization_sha256: String::new(),
    };
    output.materialization_sha256 = canonical_sha256(&output)?;
    output.verify()?;
    Ok(output)
}

/// Fragment state keyed to one Ghost record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FragmentStateObservation {
    /// Ghost record key.
    pub ghost_key: GhostLineageKey,
    /// Canonical complete Ghost occurrence SHA-256. The local key may collide
    /// across matched independent runs and is never sufficient by itself.
    pub ghost_lineage_sha256: String,
    /// Fragment state at that exact record.
    pub fragment: FragmentState,
}

/// Native Path-B projection keyed by exact Ghost record hash.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PathBProjectionObservation {
    /// Canonical complete Ghost occurrence SHA-256.
    pub ghost_lineage_sha256: String,
    /// Native Path-B projection.
    pub pathb: PathBProjection,
}

/// Deterministically join exact Ghost, fragment, and Path-B observations.
pub fn assemble_fibers(
    ghost_observations: Vec<GhostFiberObservation>,
    fragment_observations: Vec<FragmentStateObservation>,
    pathb_observations: Vec<PathBProjectionObservation>,
) -> Result<Vec<ChemicalStateFiber>> {
    let mut fragments = BTreeMap::new();
    for observation in fragment_observations {
        require_sha256(
            &observation.ghost_lineage_sha256,
            "fragment Ghost lineage_sha256",
        )?;
        if fragments
            .insert(
                observation.ghost_lineage_sha256,
                (observation.ghost_key, observation.fragment),
            )
            .is_some()
        {
            bail!("duplicate fragment state for Ghost key");
        }
    }
    let mut pathb = BTreeMap::new();
    for observation in pathb_observations {
        require_sha256(&observation.ghost_lineage_sha256, "ghost_lineage_sha256")?;
        if pathb
            .insert(observation.ghost_lineage_sha256, observation.pathb)
            .is_some()
        {
            bail!("duplicate Path-B projection for exact Ghost occurrence");
        }
    }

    let mut ghosts = BTreeMap::new();
    for observation in ghost_observations {
        let lineage_sha256 = observation.ghost.lineage_sha256()?;
        if ghosts.insert(lineage_sha256, observation).is_some() {
            bail!("duplicate global Ghost observation lineage");
        }
    }

    let expected = ghosts.len();
    let mut fibers = Vec::with_capacity(expected);
    let mut used_pathb = BTreeSet::new();
    for (lineage_hash, ghost_observation) in ghosts {
        let (fragment_key, fragment) = fragments
            .remove(&lineage_hash)
            .with_context(|| format!("missing fragment state for Ghost lineage {lineage_hash}"))?;
        if fragment_key != ghost_observation.ghost.key {
            bail!("fragment local Ghost key differs from its global lineage");
        }
        let pathb_projection = pathb
            .get(&lineage_hash)
            .with_context(|| format!("missing Path-B state for Ghost lineage {lineage_hash}"))?
            .clone();
        used_pathb.insert(lineage_hash);
        fibers.push(ChemicalStateFiber::materialize(ChemicalStateFiberBody {
            site_key: ghost_observation.site_key,
            control_lane: ghost_observation.control_lane,
            ghost: ghost_observation.ghost,
            fragment,
            protocol_phase: ghost_observation.protocol_phase,
            thermodynamic_condition: ghost_observation.thermodynamic_condition,
            structural_response: ghost_observation.structural_response,
            pathb: pathb_projection,
            uncertainty: ghost_observation.uncertainty,
            evidence_artifact_sha256: ghost_observation.evidence_artifact_sha256,
        })?);
    }
    if !fragments.is_empty() {
        bail!("unmatched fragment states remain after fiber reconstruction");
    }
    if used_pathb.len() != pathb.len() {
        bail!("unmatched Path-B projections remain after fiber reconstruction");
    }
    if fibers.len() != expected {
        bail!("fiber reconstruction violated population conservation");
    }
    Ok(fibers)
}

/// Canonical SHA-256 of a serializable value.
pub fn canonical_sha256<T: Serialize>(value: &T) -> Result<String> {
    let bytes = serde_json::to_vec(value).context("serialize canonical scientific record")?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

/// Validate a lowercase or uppercase hexadecimal SHA-256 string.
pub(crate) fn require_sha256(value: &str, field: &str) -> Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{field} must be a 64-character SHA-256 hex digest");
    }
    Ok(())
}

fn require_nonempty(value: &str, field: &str) -> Result<()> {
    if value.trim().is_empty() {
        bail!("{field} must be non-empty");
    }
    Ok(())
}

fn require_finite(value: f64, field: &str) -> Result<()> {
    if !value.is_finite() {
        bail!("{field} must be finite");
    }
    Ok(())
}

fn finite_in_range(value: f64, minimum: f64, maximum: f64, field: &str) -> Result<()> {
    require_finite(value, field)?;
    if value < minimum || value > maximum {
        bail!("{field} is outside [{minimum}, {maximum}]");
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::lambda_fragment::{
        prismcode_opcodes, LambdaJournalSegmentContext, NcmcEndpointIdentity,
        NcmcForceBoundaryKind, NcmcProtocolDirection, PeriodicChargeTreatment,
        StateReferenceConvention,
    };
    use prismcode::binary_runtime::opcode_id_for_name;
    use prismcode::qnanl_artifact::QnanlArtifactFamily;

    fn digest(byte: u8) -> String {
        format!("{byte:02x}").repeat(32)
    }

    pub(crate) fn body(
        lane: ControlLane,
        lambda: f64,
        step: u64,
        seed: u64,
    ) -> ChemicalStateFiberBody {
        let stream_id = lane as u16;
        let record_sha = digest((step % 250) as u8 + 4);
        let protocol_phase = ProtocolPhaseBinding {
            stream_id,
            thermodynamic_condition_id: "T300_P1".to_string(),
            phase_ordinal: 1,
            phase_name: "production".to_string(),
            start_step: 0,
            end_step_exclusive: 10_000,
            protocol_name: "matched-production".to_string(),
            phase_schedule_digest_sha256: digest(6),
            protocol_state_hash: digest(245),
        };
        let thermodynamic_condition = ThermodynamicCondition {
            condition_id: "T300_P1".to_string(),
            temperature_k: 300.0,
            pressure_bar: 1.0,
            ionic_strength_molar: 0.15,
            solvent_model: "TIP3P".to_string(),
            periodic_box_vectors_a: [[80.0, 0.0, 0.0], [0.0, 80.0, 0.0], [0.0, 0.0, 80.0]],
            pme_parameter_sha256: digest(7),
        };
        let mut fragment = if lane == ControlLane::NoProbe {
            FragmentState {
                identity: None,
                endpoint_a_identity: None,
                endpoint_b_identity: None,
                occupancy: FragmentOccupancy::Absent,
                lambda: 0.0,
                du_dlambda_kj_mol: 0.0,
                dudl_unit_conversion: None,
                chemical_potential_kj_mol: 0.0,
                hamiltonian_sha256: digest(5),
                source_force_event: None,
                source_endpoint_observation: None,
                chemical_potential_calibration: None,
            }
        } else {
            let species_id = 10 + u32::from(stream_id);
            let identity = FragmentChemicalIdentity {
                species_id: format!("species-{lane:?}"),
                native_species_id: species_id,
                native_chemical_state_id: 1,
                canonical_isomeric_smiles: "c1ccccc1O".to_string(),
                parameter_digest_sha256: digest(1 + stream_id as u8),
                tautomer_id: "t0".to_string(),
                protonation_state_id: "p0".to_string(),
                formal_charge: 0,
            };
            let endpoint_a = NcmcEndpointIdentity {
                species_id,
                chemical_state_id: 1,
                occupancy: LambdaEndpointOccupancy::Empty,
            };
            let endpoint_b = NcmcEndpointIdentity {
                species_id,
                chemical_state_id: 1,
                occupancy: LambdaEndpointOccupancy::Occupied,
            };
            let force_boundary = NcmcForceBoundaryIdentity::new(
                step,
                NcmcProtocolDirection::Forward,
                0,
                NcmcForceBoundaryKind::PerturbPost,
            )
            .unwrap();
            let force = LambdaFragmentForceEvent {
                schema_kind: "prismos_lambda_fragment_force_event".to_string(),
                schema_version: 3,
                stream_id: u32::from(stream_id),
                step,
                event_sequence: 1,
                parent_receipt_sha256: Some(digest(240)),
                ncmc_boundary: Some(force_boundary.clone()),
                slot_id: 0,
                species_id,
                chemical_state_id: 1,
                lambda,
                path_kind: AlchemicalPathKind::InsertDelete,
                endpoint_a: endpoint_a.clone(),
                endpoint_b: endpoint_b.clone(),
                potential_kcal_mol: 0.5,
                du_dlambda_kcal_mol: 1.25 / 4.184,
                energy_unit: "kcal/mol".to_string(),
                force_unit: "kcal/mol/angstrom".to_string(),
                endpoint_parameter_sha256: digest(4),
                endpoint_support_binding_sha256: digest(6),
                dual_endpoint_union_contract_sha256: digest(7),
                hamiltonian_sha256: digest(5),
                phase_protocol_sha256: phase_protocol_binding_sha256(
                    &protocol_phase,
                    &thermodynamic_condition,
                )
                .unwrap(),
                prismcode_opcode: prismcode_opcodes::FORCE.to_string(),
                prismcode_opcode_id: opcode_id_for_name(prismcode_opcodes::FORCE).unwrap(),
                qnanl_artifact_family: QnanlArtifactFamily::LambdaFragmentForceEvent
                    .as_str()
                    .to_string(),
            };
            let conversion = force_event_dudl_for_fiber(&force).unwrap();
            let mut exact_force_bytes = serde_json::to_vec(&force).unwrap();
            exact_force_bytes.push(b'\n');
            let exact_force_sha256 = format!("{:x}", Sha256::digest(exact_force_bytes));
            let source_endpoint_observation = if lambda == 0.0 || lambda == 1.0 {
                let (lambda_endpoint, occupancy) = if lambda == 0.0 {
                    (0, LambdaEndpointOccupancy::Empty)
                } else {
                    (1, LambdaEndpointOccupancy::Occupied)
                };
                Some(LambdaFragmentEndpointObservation {
                    schema_kind: "prismos_lambda_fragment_endpoint_observation".to_string(),
                    schema_version: 2,
                    stream_id: u32::from(stream_id),
                    step,
                    event_sequence: 2,
                    parent_receipt_sha256: digest(241),
                    supporting_force_event_sha256: exact_force_sha256,
                    supporting_force_emission_receipt_sha256: digest(243),
                    supporting_force_boundary: force_boundary,
                    slot_id: 0,
                    species_id,
                    chemical_state_id: 1,
                    lambda_endpoint,
                    occupancy,
                    path_kind: AlchemicalPathKind::InsertDelete,
                    endpoint_a: endpoint_a.clone(),
                    endpoint_b: endpoint_b.clone(),
                    phase_protocol_sha256: phase_protocol_binding_sha256(
                        &protocol_phase,
                        &thermodynamic_condition,
                    )
                    .unwrap(),
                    ghost_lineage_sha256: None,
                    prismcode_opcode: prismcode_opcodes::ENDPOINT_OBSERVE.to_string(),
                    prismcode_opcode_id: opcode_id_for_name(prismcode_opcodes::ENDPOINT_OBSERVE)
                        .unwrap(),
                    qnanl_artifact_family: QnanlArtifactFamily::LambdaFragmentEndpointObservation
                        .as_str()
                        .to_string(),
                })
            } else {
                None
            };
            let calibration = ChemicalPotentialCalibration {
                species_id,
                chemical_state_id: 1,
                temperature_k: 300.0,
                ph: 7.4,
                standard_concentration_molar: 1.0,
                excess_chemical_potential_kcal_mol: -0.25,
                total_chemical_potential_kcal_mol: -0.25,
                state_reference_convention:
                    StateReferenceConvention::TotalMuIncludesIntrinsicStateFreeEnergy,
                included_intrinsic_state_free_energy_kcal_mol: 0.0,
                periodic_charge_treatment: PeriodicChargeTreatment::NetChargeInvariant {
                    subject_charge_change_e: 0,
                },
                included_finite_size_correction_kcal_mol: 0.0,
                method: "BAR_explicit_solvent".to_string(),
                evidence_sha256: digest(242),
                uncertainty_kcal_mol: 0.05,
            };
            FragmentState {
                identity: (lambda == 1.0).then_some(identity.clone()),
                endpoint_a_identity: Some(identity.clone()),
                endpoint_b_identity: Some(identity),
                occupancy: if lambda == 0.0 {
                    FragmentOccupancy::Absent
                } else {
                    FragmentOccupancy::Present {
                        molecule_instance_id: "mol-0".to_string(),
                        fragment_slot_id: 0,
                    }
                },
                lambda,
                du_dlambda_kj_mol: conversion.du_dlambda_kj_mol,
                dudl_unit_conversion: Some(conversion),
                chemical_potential_kj_mol: calibration.total_chemical_potential_kcal_mol * 4.184,
                hamiltonian_sha256: digest(5),
                source_force_event: Some(force),
                source_endpoint_observation,
                chemical_potential_calibration: Some(calibration),
            }
        };
        let ghost = GhostByteLineage {
            key: GhostLineageKey {
                stream_id,
                frame_idx: step,
                step_idx: step,
                site_id: 2,
                record_index: step,
            },
            run_id: "run-test".to_string(),
            replica_id: format!("rep-{seed}"),
            replica_seed: seed,
            stream_seed: seed + u64::from(stream_id),
            topology_sha256: digest(249),
            record_abi_sha256: digest(248),
            record_source_sha256: digest(247),
            segment_context_digest_sha256: digest(246),
            segment_manifest_sha256: digest(2),
            segment_sequence: 1,
            segment_sha256: digest(3),
            segment_file: format!("ghost/stream_{:02}/segment_1.bin", lane as u16),
            byte_offset: step * 4096,
            byte_length: 4096,
            record_sha256: record_sha.clone(),
            lattice_node_index: step,
            engine_sha256: digest(254),
            condition_id: "T300_P1".to_string(),
            protocol_name: "matched-production".to_string(),
            protocol_phase: 1,
            phase_schedule_digest_sha256: digest(6),
            protocol_state_hash: digest(245),
        };
        let ghost_lineage_sha256 = ghost.lineage_sha256().unwrap();
        if let Some(endpoint) = fragment.source_endpoint_observation.as_mut() {
            endpoint.ghost_lineage_sha256 = Some(ghost_lineage_sha256.clone());
        }
        ChemicalStateFiberBody {
            site_key: "MYC_MAX_SITE_2".to_string(),
            control_lane: lane,
            ghost,
            fragment,
            protocol_phase,
            thermodynamic_condition,
            structural_response: StructuralResponse {
                open_state: OpenPocketEstimand {
                    probability: 0.4,
                    definition_sha256: digest(8),
                    pathb_model_sha256: digest(9),
                    lag_steps: 50,
                    open_state_ids: vec!["open".to_string()],
                },
                pocket_volume_a3: None,
                pocket_depth_a: None,
                solvent_accessible_area_a2: None,
                delta_volume_a3: None,
                delta_hydration: None,
                fragment_contact_fraction: None,
            },
            pathb: PathBProjection {
                model_sha256: digest(9),
                control_lane: lane,
                metastable_state_id: "open".to_string(),
                state_membership_probability: 0.91,
                transition_from_state_id: Some("closed".to_string()),
                transition_to_state_id: Some("open".to_string()),
                transition_probability: Some(0.2),
                forward_committor: None,
                lag_steps: 50,
                source_ghost_record_sha256: record_sha,
                source_ghost_lineage_sha256: ghost_lineage_sha256,
            },
            uncertainty: SamplingUncertainty {
                effective_sample_size: 120.0,
                integrated_autocorrelation_time_frames: 2.5,
                block_count: 12,
                standard_error: 0.03,
                confidence_interval_lower: 0.34,
                confidence_interval_upper: 0.46,
                replica_seed: seed,
                replica_id: format!("rep-{seed}"),
            },
            evidence_artifact_sha256: BTreeMap::from([("qnanl".to_string(), digest(10))]),
        }
    }

    pub(crate) fn ghost_observation(body: &ChemicalStateFiberBody) -> GhostFiberObservation {
        GhostFiberObservation {
            site_key: body.site_key.clone(),
            control_lane: body.control_lane,
            ghost: body.ghost.clone(),
            protocol_phase: body.protocol_phase.clone(),
            thermodynamic_condition: body.thermodynamic_condition.clone(),
            structural_response: body.structural_response.clone(),
            uncertainty: body.uncertainty.clone(),
            evidence_artifact_sha256: body.evidence_artifact_sha256.clone(),
        }
    }

    fn loaded_lineage(
        kind: LambdaJournalEventKind,
        sequence: u64,
        parent_receipt_sha256: Option<String>,
        raw_event_sha256: String,
        emission_receipt_sha256: String,
    ) -> LambdaJournalLoadedLineage {
        LambdaJournalLoadedLineage {
            event_kind: kind,
            event_sequence: sequence,
            parent_receipt_sha256,
            emission_receipt_sha256,
            raw_event_sha256,
            segment_manifest_sha256: digest(sequence as u8 + 20),
            segment_sha256: digest(sequence as u8 + 30),
            segment_file: format!("lambda/segment_{sequence}.json"),
            byte_offset: 0,
            byte_length: 1024,
        }
    }

    pub(crate) fn lambda_load(body: &ChemicalStateFiberBody) -> LambdaEventJournalLoad {
        let force = body.fragment.source_force_event.clone().unwrap();
        let endpoint = body.fragment.source_endpoint_observation.clone();
        let mut lane = LambdaExperimentLaneContract {
            schema_kind: "prismos_lambda_experiment_lane_contract".to_string(),
            schema_version: 1,
            lane_id: format!("lane-{:?}", body.control_lane),
            lane_kind: match body.control_lane {
                ControlLane::NoProbe => LambdaExperimentLaneKind::NoProbe,
                ControlLane::HypothesisFragment => LambdaExperimentLaneKind::HypothesisFragments,
                ControlLane::OrthogonalChemicalControl => {
                    LambdaExperimentLaneKind::OrthogonalChemicalControls
                }
            },
            fragment_species_ids: vec![force.species_id],
            virtual_probe_count: 0,
            contract_sha256: String::new(),
        };
        lane.contract_sha256 = canonical_sha256(&lane).unwrap();
        let key = LambdaJournalKey {
            stream_id: body.ghost.key.stream_id,
            step_idx: body.ghost.key.step_idx,
            slot_id: force.slot_id,
        };
        let force_lineage = loaded_lineage(
            LambdaJournalEventKind::Force,
            force.event_sequence,
            force.parent_receipt_sha256.clone(),
            endpoint
                .as_ref()
                .map(|value| value.supporting_force_event_sha256.clone())
                .unwrap_or_else(|| {
                    let mut bytes = serde_json::to_vec(&force).unwrap();
                    bytes.push(b'\n');
                    format!("{:x}", Sha256::digest(bytes))
                }),
            endpoint
                .as_ref()
                .map(|value| value.supporting_force_emission_receipt_sha256.clone())
                .unwrap_or_else(|| digest(243)),
        );
        let mut event_lineage = vec![force_lineage];
        if let Some(endpoint) = &endpoint {
            let mut endpoint_bytes = serde_json::to_vec(endpoint).unwrap();
            endpoint_bytes.push(b'\n');
            event_lineage.push(loaded_lineage(
                LambdaJournalEventKind::Endpoint,
                endpoint.event_sequence,
                Some(endpoint.parent_receipt_sha256.clone()),
                format!("{:x}", Sha256::digest(endpoint_bytes)),
                digest(244),
            ));
        }
        let observation = LambdaJournalObservation {
            key,
            lane: lane.clone(),
            force_events: vec![force],
            endpoint_event: endpoint,
            chemical_potential_calibration: body.fragment.chemical_potential_calibration.clone(),
            ncmc_move_receipt: None,
            accepted_state_commit: None,
            rollback_no_change: None,
            event_lineage,
        };
        let context = LambdaJournalSegmentContext {
            run_id: body.ghost.run_id.clone(),
            replica_id: body.ghost.replica_id.clone(),
            replica_seed: body.ghost.replica_seed,
            stream_id: body.ghost.key.stream_id,
            stream_seed: body.ghost.stream_seed,
            engine_sha256: body.ghost.engine_sha256.clone(),
            topology_sha256: body.ghost.topology_sha256.clone(),
            lane,
            ncmc_execution_mode: None,
            ncmc_scheduler: None,
            lambda_fragment_system_sha256: Some(digest(232)),
            protocol_phase: body.protocol_phase.clone(),
            thermodynamic_condition: body.thermodynamic_condition.clone(),
            phase_protocol_sha256: phase_protocol_binding_sha256(
                &body.protocol_phase,
                &body.thermodynamic_condition,
            )
            .unwrap(),
        };
        let event_count = observation.event_lineage.len() as u64;
        LambdaEventJournalLoad {
            produced: event_count,
            persisted: event_count,
            decoded: event_count,
            indexed: event_count,
            committed_resolution_batch_count: 0,
            partial_resolution_chain_count: 0,
            observations: BTreeMap::from([(key, observation)]),
            no_probe_absence_receipts: Vec::new(),
            stream_terminal_manifest_sha256: BTreeMap::from([(
                body.ghost.key.stream_id,
                digest(231),
            )]),
            stream_contexts: BTreeMap::from([(body.ghost.key.stream_id, context.clone())]),
            stream_phase_contexts: BTreeMap::from([(body.ghost.key.stream_id, vec![context])]),
            population_receipt_sha256: digest(230),
        }
    }

    #[test]
    fn post_persistence_join_binds_exact_ghost_endpoint_force_and_context() {
        let body = body(ControlLane::HypothesisFragment, 1.0, 200, 51);
        let ghost = ghost_observation(&body);
        let load = lambda_load(&body);
        let result = materialize_lambda_ghost_endpoint_joins(&[ghost], &load).unwrap();
        assert_eq!(result.evaluated_join_count, 1);
        assert_eq!(result.not_evaluable_count, 0);
        let row = &result.rows[0];
        assert_eq!(row.status, LambdaGhostEndpointJoinStatus::Evaluated);
        assert_eq!(
            row.ghost_record_sha256.as_deref(),
            Some(body.ghost.record_sha256.as_str())
        );
        assert_eq!(row.ghost_byte_offset, Some(body.ghost.byte_offset));
        assert_eq!(row.ghost_byte_length, Some(body.ghost.byte_length));
        assert_eq!(
            row.supporting_force_boundary,
            body.fragment
                .source_force_event
                .as_ref()
                .unwrap()
                .ncmc_boundary
        );
        result.verify().unwrap();
    }

    #[test]
    fn post_persistence_join_never_uses_nearest_frame_inference() {
        let body = body(ControlLane::HypothesisFragment, 1.0, 210, 52);
        let mut ghost = ghost_observation(&body);
        ghost.ghost.key.step_idx += 1;
        ghost.ghost.key.frame_idx += 1;
        let load = lambda_load(&body);
        let result = materialize_lambda_ghost_endpoint_joins(&[ghost], &load).unwrap();
        assert_eq!(result.evaluated_join_count, 0);
        assert_eq!(result.not_evaluable_count, 2);
        let reasons: BTreeSet<_> = result
            .rows
            .iter()
            .filter_map(|row| row.not_evaluable_reason)
            .collect();
        assert!(reasons.contains(&LambdaGhostEndpointNotEvaluableReason::UnmatchedGhostNode));
        assert!(reasons.contains(&LambdaGhostEndpointNotEvaluableReason::UnmatchedLambdaEndpoint));
    }

    #[test]
    fn post_persistence_join_rejects_duplicate_exact_inputs() {
        let body = body(ControlLane::HypothesisFragment, 1.0, 220, 53);
        let ghost = ghost_observation(&body);
        let load = lambda_load(&body);
        assert!(materialize_lambda_ghost_endpoint_joins(&[ghost.clone(), ghost], &load).is_err());

        let mut duplicate_lineage = lambda_load(&body);
        let observation = duplicate_lineage.observations.values_mut().next().unwrap();
        observation.event_lineage.push(
            observation
                .event_lineage
                .iter()
                .find(|lineage| lineage.event_kind == LambdaJournalEventKind::Endpoint)
                .unwrap()
                .clone(),
        );
        assert!(materialize_lambda_ghost_endpoint_joins(
            &[ghost_observation(&body)],
            &duplicate_lineage
        )
        .is_err());
    }

    #[test]
    fn post_persistence_join_preserves_missing_coordinate_and_lineage() {
        let body = body(ControlLane::HypothesisFragment, 1.0, 230, 54);
        let ghost = ghost_observation(&body);
        let mut missing_endpoint_lineage = lambda_load(&body);
        missing_endpoint_lineage
            .observations
            .values_mut()
            .next()
            .unwrap()
            .event_lineage
            .retain(|lineage| lineage.event_kind != LambdaJournalEventKind::Endpoint);
        let result =
            materialize_lambda_ghost_endpoint_joins(&[ghost.clone()], &missing_endpoint_lineage)
                .unwrap();
        assert_eq!(result.evaluated_join_count, 0);
        assert_eq!(
            result.rows[0].not_evaluable_reason,
            Some(LambdaGhostEndpointNotEvaluableReason::MissingEndpointByteLineage)
        );

        let mut missing_force_lineage = lambda_load(&body);
        missing_force_lineage
            .observations
            .values_mut()
            .next()
            .unwrap()
            .event_lineage
            .retain(|lineage| lineage.event_kind != LambdaJournalEventKind::Force);
        let result =
            materialize_lambda_ghost_endpoint_joins(&[ghost], &missing_force_lineage).unwrap();
        assert_eq!(
            result.rows[0].not_evaluable_reason,
            Some(LambdaGhostEndpointNotEvaluableReason::MissingSupportingForceByteLineage)
        );

        let mut invalid_ghost = ghost_observation(&body);
        invalid_ghost.ghost.byte_length = 0;
        let result =
            materialize_lambda_ghost_endpoint_joins(&[invalid_ghost], &lambda_load(&body)).unwrap();
        let retained = result
            .rows
            .iter()
            .find(|row| {
                row.not_evaluable_reason
                    == Some(LambdaGhostEndpointNotEvaluableReason::GhostByteLineageInvalid)
            })
            .unwrap();
        assert!(retained.retained_invalid_ghost_node.is_some());
        assert!(retained.retained_invalid_ghost_input_sha256.is_some());
    }

    #[test]
    fn post_persistence_join_rejects_partial_lambda_for_endpoint_claims() {
        let body = body(ControlLane::HypothesisFragment, 0.5, 240, 55);
        let ghost = ghost_observation(&body);
        let load = lambda_load(&body);
        let result = materialize_lambda_ghost_endpoint_joins(&[ghost], &load).unwrap();
        assert_eq!(result.evaluated_join_count, 0);
        assert!(result.rows.iter().any(|row| {
            row.not_evaluable_reason
                == Some(LambdaGhostEndpointNotEvaluableReason::PartialLambdaSamplingState)
        }));
    }

    #[test]
    fn deterministic_content_identity_and_tamper_detection() {
        let a =
            ChemicalStateFiber::materialize(body(ControlLane::HypothesisFragment, 1.0, 100, 42))
                .unwrap();
        let b =
            ChemicalStateFiber::materialize(body(ControlLane::HypothesisFragment, 1.0, 100, 42))
                .unwrap();
        assert_eq!(a.fiber_sha256, b.fiber_sha256);
        a.verify().unwrap();
        let mut tampered = a;
        tampered.body.structural_response.open_state.probability = 0.5;
        assert!(tampered.verify().is_err());
    }

    #[test]
    fn partial_lambda_is_sampling_only() {
        let fiber =
            ChemicalStateFiber::materialize(body(ControlLane::HypothesisFragment, 0.5, 101, 43))
                .unwrap();
        assert_eq!(
            fiber.endpoint_eligibility,
            EndpointEligibility::SamplingOnly
        );
    }

    #[test]
    fn insertion_and_deletion_endpoints_preserve_species_and_occupancy_semantics() {
        let mut absent = body(ControlLane::HypothesisFragment, 0.0, 102, 44);
        absent.fragment.occupancy = FragmentOccupancy::Absent;
        let lambda_zero = ChemicalStateFiber::materialize(absent).unwrap();
        assert!(lambda_zero.body.fragment.identity.is_none());
        assert!(lambda_zero.body.fragment.endpoint_a_identity.is_some());
        assert!(lambda_zero.body.fragment.endpoint_b_identity.is_some());
        assert_eq!(
            lambda_zero.endpoint_eligibility,
            EndpointEligibility::PhysicalEndpoint
        );

        let coupled =
            ChemicalStateFiber::materialize(body(ControlLane::HypothesisFragment, 1.0, 103, 44))
                .unwrap();
        assert!(matches!(
            coupled.body.fragment.occupancy,
            FragmentOccupancy::Present { .. }
        ));

        let mut invalid_absent_partial = body(ControlLane::HypothesisFragment, 0.5, 104, 44);
        invalid_absent_partial.fragment.occupancy = FragmentOccupancy::Absent;
        assert!(ChemicalStateFiber::materialize(invalid_absent_partial).is_err());

        let mut invalid_present_zero = body(ControlLane::HypothesisFragment, 0.0, 105, 44);
        invalid_present_zero.fragment.occupancy = FragmentOccupancy::Present {
            molecule_instance_id: "mol-0".to_string(),
            fragment_slot_id: 0,
        };
        invalid_present_zero.fragment.identity =
            invalid_present_zero.fragment.endpoint_a_identity.clone();
        assert!(ChemicalStateFiber::materialize(invalid_present_zero).is_err());
    }

    #[test]
    fn exact_join_reconstructs_independent_of_input_order() {
        let bodies = vec![
            body(ControlLane::NoProbe, 0.0, 100, 1),
            body(ControlLane::HypothesisFragment, 1.0, 101, 1),
        ];
        let make_inputs = || {
            let ghosts = bodies
                .iter()
                .map(|body| GhostFiberObservation {
                    site_key: body.site_key.clone(),
                    control_lane: body.control_lane,
                    ghost: body.ghost.clone(),
                    protocol_phase: body.protocol_phase.clone(),
                    thermodynamic_condition: body.thermodynamic_condition.clone(),
                    structural_response: body.structural_response.clone(),
                    uncertainty: body.uncertainty.clone(),
                    evidence_artifact_sha256: body.evidence_artifact_sha256.clone(),
                })
                .collect::<Vec<_>>();
            let fragments = bodies
                .iter()
                .map(|body| FragmentStateObservation {
                    ghost_key: body.ghost.key,
                    ghost_lineage_sha256: body.ghost.lineage_sha256().unwrap(),
                    fragment: body.fragment.clone(),
                })
                .collect::<Vec<_>>();
            let pathb = bodies
                .iter()
                .map(|body| PathBProjectionObservation {
                    ghost_lineage_sha256: body.ghost.lineage_sha256().unwrap(),
                    pathb: body.pathb.clone(),
                })
                .collect::<Vec<_>>();
            (ghosts, fragments, pathb)
        };
        let (g, f, p) = make_inputs();
        let first = assemble_fibers(g, f, p).unwrap();
        let (mut g, mut f, mut p) = make_inputs();
        g.reverse();
        f.reverse();
        p.reverse();
        let second = assemble_fibers(g, f, p).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), bodies.len());
    }

    #[test]
    fn rejects_phase_or_pathb_lineage_mismatch() {
        let mut bad_phase = body(ControlLane::NoProbe, 0.0, 100, 1);
        bad_phase.protocol_phase.stream_id = 99;
        assert!(ChemicalStateFiber::materialize(bad_phase).is_err());

        let mut bad_pathb = body(ControlLane::NoProbe, 0.0, 100, 1);
        bad_pathb.pathb.source_ghost_record_sha256 = digest(99);
        assert!(ChemicalStateFiber::materialize(bad_pathb).is_err());
    }
}
