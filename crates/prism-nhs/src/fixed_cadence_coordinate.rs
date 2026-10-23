//! Lossless fixed-cadence coordinate evidence for Ghost discovery and confirmation.
//!
//! Discovery persists the complete signed protein-analysis population at a
//! predeclared cadence.  Confirmation additionally persists the complete
//! full-system source frame and a PBC-aware water/ion/fragment shell produced
//! by an exhaustive scan.  The strict loader reconstructs the shell from the
//! full frame, so an omitted local molecule cannot pass conservation.

use crate::ghost_coordinate_segment::ExactLocalShellAtom;
use crate::input::PrismPrepTopology;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

const SCHEMA_VERSION: u32 = 3;
const HEADER_BYTES: usize = 512;
const MAGIC: [u8; 8] = *b"PGFCS003";
const ZERO_SHA256: &str = "0000000000000000000000000000000000000000000000000000000000000000";
const SHELL_ROW_BYTES: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixedCadenceCoordinateMode {
    WholeProteinDiscovery,
    CandidateConfirmation,
}

/// Explicit spatial boundary used by the body-fixed transform.  A periodic
/// trajectory is always unwrapped with the full triclinic cell.  Euclidean
/// displacement is legal only when a separately hash-bound nonperiodic
/// topology contract declares it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BodyFixedBoundaryMode {
    PeriodicTriclinicMinimumImage,
    DeclaredNonperiodicEuclidean,
}

/// One topology-exact, chain-aware atom in the signed body-frame authority.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BodyFixedAnchorAtom {
    pub atom_index: usize,
    pub analysis_residue_id: u32,
    pub chain_id: String,
    pub pdb_residue_id: i64,
    pub residue_name: String,
    pub atom_name: String,
    pub reference_position_angstrom: [f64; 3],
}

/// Pre-run authority for candidate tracking in a moving molecular frame.
///
/// The first three atoms define origin, positive-x, and the x/y plane. More
/// anchors are permitted and enter the rigid-fit residual. Numerical limits
/// are supplied by the signed authority; this implementation supplies none.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BodyFixedTrackerAuthority {
    pub schema_kind: String,
    pub schema_version: String,
    pub authority_sha256: String,
    pub reference_structure_sha256: String,
    pub reference_coordinate_frame_sha256: String,
    pub chain_mapping_receipt_sha256: String,
    pub topology_sha256: String,
    pub spatial_boundary_contract_sha256: String,
    pub boundary_mode: BodyFixedBoundaryMode,
    pub unwrap_semantics: String,
    pub frame_construction_semantics: String,
    pub anchor_atoms: Vec<BodyFixedAnchorAtom>,
    pub reference_candidate_centroid_angstrom: [f64; 3],
    pub maximum_anchor_fit_rmsd_angstrom: f64,
    pub maximum_anchor_fit_residual_angstrom: f64,
}

impl BodyFixedTrackerAuthority {
    fn body_sha256(&self) -> Result<String> {
        let mut body = self.clone();
        body.authority_sha256.clear();
        canonical_sha256(&body)
    }

    pub fn materialize(mut self) -> Result<Self> {
        if !self.authority_sha256.is_empty() {
            bail!("body-fixed authority materialization requires an empty digest");
        }
        self.authority_sha256 = self.body_sha256()?;
        self.verify()?;
        Ok(self)
    }

    pub fn verify(&self) -> Result<()> {
        if self.schema_kind != "prism_body_fixed_tracker_authority"
            || self.schema_version != "1.0.0"
            || self.unwrap_semantics
                != match self.boundary_mode {
                    BodyFixedBoundaryMode::PeriodicTriclinicMinimumImage => {
                        "anchor_zero_triclinic_minimum_image_v1"
                    }
                    BodyFixedBoundaryMode::DeclaredNonperiodicEuclidean => {
                        "declared_nonperiodic_no_unwrap_v1"
                    }
                }
            || self.frame_construction_semantics
                != "ordered_three_anchor_right_handed_rigid_frame_v1"
            || self.anchor_atoms.len() < 3
            || self
                .reference_candidate_centroid_angstrom
                .iter()
                .any(|value| !value.is_finite())
            || !self.maximum_anchor_fit_rmsd_angstrom.is_finite()
            || self.maximum_anchor_fit_rmsd_angstrom <= 0.0
            || !self.maximum_anchor_fit_residual_angstrom.is_finite()
            || self.maximum_anchor_fit_residual_angstrom <= 0.0
        {
            bail!("invalid body-fixed tracker authority");
        }
        for (value, field) in [
            (&self.authority_sha256, "body-fixed authority"),
            (
                &self.reference_structure_sha256,
                "body-fixed reference structure",
            ),
            (
                &self.reference_coordinate_frame_sha256,
                "body-fixed reference coordinate frame",
            ),
            (
                &self.chain_mapping_receipt_sha256,
                "body-fixed chain mapping receipt",
            ),
            (&self.topology_sha256, "body-fixed topology"),
            (
                &self.spatial_boundary_contract_sha256,
                "body-fixed spatial boundary contract",
            ),
        ] {
            require_sha256(value, field)?;
        }
        let mut atom_indices = BTreeSet::new();
        for atom in &self.anchor_atoms {
            if !atom_indices.insert(atom.atom_index)
                || atom.chain_id.trim().is_empty()
                || atom.residue_name.trim().is_empty()
                || atom.atom_name.trim().is_empty()
                || atom
                    .reference_position_angstrom
                    .iter()
                    .any(|value| !value.is_finite())
            {
                bail!("body-fixed anchor population is invalid or duplicate");
            }
        }
        body_frame_basis([
            self.anchor_atoms[0].reference_position_angstrom,
            self.anchor_atoms[1].reference_position_angstrom,
            self.anchor_atoms[2].reference_position_angstrom,
        ])
        .context("body-fixed reference anchors are degenerate")?;
        if self.body_sha256()? != self.authority_sha256 {
            bail!("body-fixed tracker authority hash mismatch");
        }
        Ok(())
    }
}

/// Per-frame rigid transform and deformation residual derived from the exact
/// synchronized coordinates. The centroid is therefore never a static
/// laboratory-frame input.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BodyFixedFrameTransform {
    pub schema_kind: String,
    pub schema_version: String,
    pub transform_sha256: String,
    pub authority_sha256: String,
    pub boundary_mode: BodyFixedBoundaryMode,
    pub reference_to_frame_rotation: [[f64; 3]; 3],
    pub reference_to_frame_translation_angstrom: [f64; 3],
    pub transformed_centroid_angstrom: [f64; 3],
    pub anchor_fit_rmsd_angstrom: f64,
    pub anchor_fit_max_residual_angstrom: f64,
    pub anchor_count: usize,
}

impl BodyFixedFrameTransform {
    fn body_sha256(&self) -> Result<String> {
        let mut body = self.clone();
        body.transform_sha256.clear();
        canonical_sha256(&body)
    }

    fn verify(&self, authority: &BodyFixedTrackerAuthority) -> Result<()> {
        if self.schema_kind != "prism_body_fixed_frame_transform"
            || self.schema_version != "1.0.0"
            || self.authority_sha256 != authority.authority_sha256
            || self.boundary_mode != authority.boundary_mode
            || self.anchor_count != authority.anchor_atoms.len()
            || self
                .reference_to_frame_rotation
                .iter()
                .flatten()
                .chain(&self.reference_to_frame_translation_angstrom)
                .chain(&self.transformed_centroid_angstrom)
                .any(|value| !value.is_finite())
            || !self.anchor_fit_rmsd_angstrom.is_finite()
            || !self.anchor_fit_max_residual_angstrom.is_finite()
            || self.anchor_fit_rmsd_angstrom < 0.0
            || self.anchor_fit_max_residual_angstrom < 0.0
            || self.anchor_fit_rmsd_angstrom > authority.maximum_anchor_fit_rmsd_angstrom
            || self.anchor_fit_max_residual_angstrom
                > authority.maximum_anchor_fit_residual_angstrom
        {
            bail!("invalid or out-of-authority body-fixed frame transform");
        }
        require_sha256(&self.transform_sha256, "body-fixed frame transform")?;
        let rotation = self.reference_to_frame_rotation;
        for left in 0..3 {
            for right in 0..3 {
                let observed = (0..3)
                    .map(|axis| rotation[axis][left] * rotation[axis][right])
                    .sum::<f64>();
                let expected = if left == right { 1.0 } else { 0.0 };
                if (observed - expected).abs() > 1.0e-10 {
                    bail!("body-fixed transform rotation is not orthonormal");
                }
            }
        }
        if self.body_sha256()? != self.transform_sha256 {
            bail!("body-fixed frame transform hash mismatch");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixedCadenceCoordinateContext {
    pub schema_kind: String,
    pub schema_version: u32,
    pub run_id: String,
    pub replica_id: String,
    pub replica_seed: u64,
    pub stream_seed: u64,
    pub stream_id: u16,
    pub topology_sha256: String,
    pub selected_engine_sha256: String,
    /// Complete content-addressed request admitted before CUDA/MD. This
    /// prevents a coordinate population from being replayed under a different
    /// Path-B estimand.
    pub native_pathb_runtime_request_sha256: String,
    /// Digest of the signed timebase context and of its underlying integrator
    /// authority artifact. Neither is inferred from the scalar timestep.
    pub integration_timebase_context_sha256: String,
    pub integration_timebase_authority_sha256: String,
    pub integration_timestep_femtoseconds: f64,
    /// Exact predeclared lag grid copied from the admitted request. Every lag
    /// must be an integer multiple of the persisted coordinate cadence.
    pub candidate_lag_steps: Vec<u64>,
    pub analysis_residue_contract_sha256: String,
    pub atom_analysis_residue_ids_sha256: String,
    pub molecule_identity_table_sha256: String,
    pub periodic_cell_contract_sha256: String,
    pub thermodynamic_condition_id: String,
    pub protocol_id: String,
    pub protocol_state_hash: String,
    pub phase_schedule_digest_sha256: String,
    pub mode: FixedCadenceCoordinateMode,
    pub cadence_steps: u64,
    pub start_step: u64,
    pub end_step_exclusive: u64,
    pub candidate_tracker_sha256: Option<String>,
    pub local_shell_cutoff_angstrom: Option<f64>,
    pub body_fixed_tracker_authority: Option<BodyFixedTrackerAuthority>,
}

impl FixedCadenceCoordinateContext {
    pub fn verify(&self) -> Result<()> {
        if self.schema_kind != "prism_fixed_cadence_coordinate_context"
            || self.schema_version != SCHEMA_VERSION
            || self.run_id.trim().is_empty()
            || self.replica_id.trim().is_empty()
            || self.protocol_id.trim().is_empty()
            || self.thermodynamic_condition_id.trim().is_empty()
            || self.cadence_steps == 0
            || !self.integration_timestep_femtoseconds.is_finite()
            || self.integration_timestep_femtoseconds <= 0.0
            || self.candidate_lag_steps.is_empty()
            || self
                .candidate_lag_steps
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || self
                .candidate_lag_steps
                .iter()
                .any(|lag| *lag == 0 || *lag % self.cadence_steps != 0)
            || self.end_step_exclusive <= self.start_step
            || (self.end_step_exclusive - self.start_step) % self.cadence_steps != 0
        {
            bail!("invalid fixed-cadence coordinate context");
        }
        for (value, field) in [
            (&self.topology_sha256, "topology_sha256"),
            (&self.selected_engine_sha256, "selected_engine_sha256"),
            (
                &self.native_pathb_runtime_request_sha256,
                "native_pathb_runtime_request_sha256",
            ),
            (
                &self.integration_timebase_context_sha256,
                "integration_timebase_context_sha256",
            ),
            (
                &self.integration_timebase_authority_sha256,
                "integration_timebase_authority_sha256",
            ),
            (
                &self.analysis_residue_contract_sha256,
                "analysis_residue_contract_sha256",
            ),
            (
                &self.atom_analysis_residue_ids_sha256,
                "atom_analysis_residue_ids_sha256",
            ),
            (
                &self.molecule_identity_table_sha256,
                "molecule_identity_table_sha256",
            ),
            (
                &self.periodic_cell_contract_sha256,
                "periodic_cell_contract_sha256",
            ),
            (&self.protocol_state_hash, "protocol_state_hash"),
            (
                &self.phase_schedule_digest_sha256,
                "phase_schedule_digest_sha256",
            ),
        ] {
            require_sha256(value, field)?;
        }
        match self.mode {
            FixedCadenceCoordinateMode::WholeProteinDiscovery => {
                if self.candidate_tracker_sha256.is_some()
                    || self.local_shell_cutoff_angstrom.is_some()
                    || self.body_fixed_tracker_authority.is_some()
                {
                    bail!("discovery cadence context cannot carry a confirmation tracker");
                }
            }
            FixedCadenceCoordinateMode::CandidateConfirmation => {
                require_sha256(
                    self.candidate_tracker_sha256
                        .as_deref()
                        .context("confirmation context lacks candidate tracker")?,
                    "candidate_tracker_sha256",
                )?;
                let cutoff = self
                    .local_shell_cutoff_angstrom
                    .context("confirmation context lacks local-shell cutoff")?;
                if !cutoff.is_finite() || cutoff <= 0.0 {
                    bail!("invalid confirmation local-shell cutoff");
                }
                let authority = self
                    .body_fixed_tracker_authority
                    .as_ref()
                    .context("confirmation context lacks body-fixed tracker authority")?;
                authority.verify()?;
                if authority.topology_sha256 != self.topology_sha256
                    || authority.spatial_boundary_contract_sha256
                        != self.periodic_cell_contract_sha256
                {
                    bail!("body-fixed authority differs from cadence topology/boundary");
                }
            }
        }
        Ok(())
    }

    pub fn digest_sha256(&self) -> Result<String> {
        self.verify()?;
        canonical_sha256(self)
    }

    pub fn scheduled_frame_count(&self) -> u64 {
        (self.end_step_exclusive - self.start_step) / self.cadence_steps
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixedCadenceFrameEntry {
    pub segment_ordinal: u64,
    pub file: String,
    pub file_sha256: String,
    pub file_bytes: u64,
    pub previous_file_sha256: String,
    pub context_digest_sha256: String,
    pub stream_id: u16,
    pub step_idx: u64,
    pub protocol_phase: u8,
    pub thermodynamic_condition_id: String,
    pub protocol_state_hash: String,
    pub phase_schedule_digest_sha256: String,
    pub source_full_frame_sha256: String,
    pub analysis_coordinate_sha256: String,
    pub persisted_full_frame_sha256: Option<String>,
    pub local_shell_sha256: Option<String>,
    pub hydration_shell_sha256: Option<String>,
    pub fragment_shell_sha256: Option<String>,
    pub analysis_atom_count: u64,
    pub full_atom_count: u64,
    pub local_shell_atom_count: u64,
    pub excluded_population_scanned: u64,
    pub tracked_centroid_angstrom: Option<[f32; 3]>,
    pub body_fixed_transform: Option<BodyFixedFrameTransform>,
    pub fsync_pass: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixedCadenceCoordinateManifest {
    pub schema_kind: String,
    pub schema_version: u32,
    pub context: FixedCadenceCoordinateContext,
    pub context_digest_sha256: String,
    pub scheduled_frame_count: u64,
    pub persisted_frame_count: u64,
    pub decoded_frame_count: u64,
    pub indexed_frame_count: u64,
    pub frames: Vec<FixedCadenceFrameEntry>,
    pub completed: bool,
    pub scheduled_equals_persisted_equals_decoded_equals_indexed: bool,
}

#[derive(Clone, Debug)]
pub struct FixedCadenceCoordinateInput {
    pub step_idx: u64,
    pub protocol_phase: u8,
    pub positions_angstrom: Vec<f32>,
    /// Legacy caller-supplied laboratory centroid. This must always be `None`;
    /// confirmation centroids are derived from the signed body-frame authority.
    pub tracked_centroid_angstrom: Option<[f32; 3]>,
}

#[derive(Clone, Debug)]
pub struct VerifiedFixedCadenceFrame {
    pub entry: FixedCadenceFrameEntry,
    pub analysis_positions_angstrom: Vec<f32>,
    pub full_positions_angstrom: Option<Vec<f32>>,
    pub local_shell: Vec<ExactLocalShellAtom>,
}

#[derive(Clone, Debug)]
pub struct FixedCadenceCoordinateLoad {
    pub manifest: FixedCadenceCoordinateManifest,
    pub frames: BTreeMap<u64, VerifiedFixedCadenceFrame>,
}

#[derive(Clone, Debug)]
pub struct FixedCadenceCoordinatePopulationLoad {
    pub streams: BTreeMap<u16, FixedCadenceCoordinateLoad>,
    pub requested_stream_ids: Vec<u16>,
    pub total_scheduled_frame_count: u64,
    pub total_persisted_frame_count: u64,
    pub total_decoded_frame_count: u64,
    pub total_indexed_frame_count: u64,
    pub population_sha256: String,
}

#[derive(Serialize)]
struct FixedCadencePopulationDigest<'a> {
    schema_kind: &'static str,
    schema_version: u32,
    requested_stream_ids: &'a [u16],
    total_scheduled_frame_count: u64,
    total_persisted_frame_count: u64,
    total_decoded_frame_count: u64,
    total_indexed_frame_count: u64,
    streams: &'a [FixedCadencePopulationDigestRow],
}

#[derive(Serialize)]
struct FixedCadencePopulationDigestRow {
    stream_id: u16,
    manifest_sha256: String,
    context_digest_sha256: String,
    scheduled_frame_count: u64,
    persisted_frame_count: u64,
    decoded_frame_count: u64,
    indexed_frame_count: u64,
}

#[derive(Clone)]
struct AtomRole {
    analysis_id: i32,
    molecule_id: u64,
    role: String,
}

struct TopologyView {
    n_atoms: usize,
    analysis_ids: Vec<i32>,
    roles: Vec<AtomRole>,
    box_matrix: Option<[[f64; 3]; 3]>,
    inverse_box: Option<[[f64; 3]; 3]>,
}

impl TopologyView {
    fn from_topology(topology: &PrismPrepTopology) -> Result<Self> {
        let analysis_ids = topology
            .atom_analysis_residue_ids
            .as_ref()
            .context("fixed-cadence coordinates require signed atom-analysis ids")?
            .clone();
        if analysis_ids.len() != topology.n_atoms {
            bail!("atom-analysis id population differs from topology atoms");
        }
        let table = topology
            .molecule_identity_table
            .as_ref()
            .context("fixed-cadence coordinates require molecule identity table")?;
        let mut roles = vec![None; topology.n_atoms];
        for molecule in &table.molecules {
            let role = match molecule.kind.to_ascii_uppercase().as_str() {
                "BIOMOLECULE" | "PROTEIN" | "RECEPTOR" | "POLYMER" => "RECEPTOR",
                "WATER" => "WATER",
                "ION" => "ION",
                "LAMBDA_FRAGMENT" | "FRAGMENT" => "LAMBDA_FRAGMENT",
                _ => "OTHER",
            };
            for &atom in &molecule.atom_indices {
                if atom >= topology.n_atoms || roles[atom].is_some() {
                    bail!("molecule identity table is not an exact atom partition");
                }
                roles[atom] = Some(AtomRole {
                    analysis_id: analysis_ids[atom],
                    molecule_id: molecule.molecule_id,
                    role: role.to_string(),
                });
            }
        }
        let roles = roles
            .into_iter()
            .enumerate()
            .map(|(atom, role)| role.with_context(|| format!("molecule role missing atom {atom}")))
            .collect::<Result<Vec<_>>>()?;
        let (box_matrix, inverse_box) = match topology.periodic_cell.as_ref() {
            Some(cell) if cell.periodic => {
                let rows = cell
                    .vectors_angstrom
                    .context("periodic fixed-cadence topology lacks box vectors")?;
                let matrix = [
                    [rows[0][0], rows[1][0], rows[2][0]],
                    [rows[0][1], rows[1][1], rows[2][1]],
                    [rows[0][2], rows[1][2], rows[2][2]],
                ];
                let inverse = invert_3x3(matrix).context("periodic box is singular")?;
                (Some(matrix), Some(inverse))
            }
            _ => (None, None),
        };
        Ok(Self {
            n_atoms: topology.n_atoms,
            analysis_ids,
            roles,
            box_matrix,
            inverse_box,
        })
    }
}

pub struct FixedCadenceCoordinateWriter {
    root: PathBuf,
    manifest_path: PathBuf,
    topology: TopologyView,
    manifest: FixedCadenceCoordinateManifest,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixedCadenceContinuationV1 {
    pub output: crate::checkpoint_output::OutputDirectoryBoundary,
    pub manifest: FixedCadenceCoordinateManifest,
}
impl FixedCadenceCoordinateWriter {
    pub fn checkpoint(&self, run_root: &Path) -> Result<FixedCadenceContinuationV1> {
        let bytes = read_regular_nofollow(&self.manifest_path, None)?;
        if bytes != canonical_json_bytes(&self.manifest)? {
            bail!("fixed-cadence live manifest differs from durable boundary");
        }
        Ok(FixedCadenceContinuationV1 {
            output: crate::checkpoint_output::capture_directory(
                run_root,
                &self.root,
                &["manifest.json"],
            )?,
            manifest: self.manifest.clone(),
        })
    }
    pub fn reopen_checkpoint(
        run_root: &Path,
        state: &FixedCadenceContinuationV1,
        topology: &PrismPrepTopology,
        expected: &FixedCadenceCoordinateContext,
    ) -> Result<Self> {
        validate_topology_binding(topology, expected)?;
        if &state.manifest.context != expected
            || state.manifest.context_digest_sha256 != expected.digest_sha256()?
        {
            bail!("fixed-cadence checkpoint context differs");
        }
        let root = crate::checkpoint_output::restore_directory(run_root, &state.output)?;
        let manifest_path = root.join("manifest.json");
        let bytes = read_regular_nofollow(&manifest_path, None)?;
        if bytes != canonical_json_bytes(&state.manifest)? {
            bail!("fixed-cadence restored manifest differs");
        }
        let view = TopologyView::from_topology(topology)?;
        let manifest = &state.manifest;
        if manifest.schema_kind != "prism_fixed_cadence_coordinate_manifest"
            || manifest.schema_version != SCHEMA_VERSION
            || manifest.scheduled_frame_count != expected.scheduled_frame_count()
            || manifest.persisted_frame_count != manifest.frames.len() as u64
            || manifest.persisted_frame_count > manifest.scheduled_frame_count
        {
            bail!("fixed-cadence checkpoint population mismatch");
        }
        if manifest.completed {
            load_verified_fixed_cadence_coordinates(&root, topology, expected)?;
        } else if manifest.decoded_frame_count != 0
            || manifest.indexed_frame_count != 0
            || manifest.scheduled_equals_persisted_equals_decoded_equals_indexed
        {
            bail!("incomplete fixed-cadence checkpoint claims terminal conservation");
        }
        let mut previous = ZERO_SHA256.to_string();
        for (index, entry) in manifest.frames.iter().enumerate() {
            let step = expected
                .start_step
                .checked_add(
                    (index as u64)
                        .checked_mul(expected.cadence_steps)
                        .context("cadence overflow")?,
                )
                .context("cadence overflow")?;
            if entry.file != format!("frame_{index:08}.ghostcadence")
                || entry.segment_ordinal != index as u64
                || entry.step_idx != step
                || entry.previous_file_sha256 != previous
                || entry.stream_id != expected.stream_id
                || entry.context_digest_sha256 != manifest.context_digest_sha256
                || entry.thermodynamic_condition_id != expected.thermodynamic_condition_id
                || entry.protocol_state_hash != expected.protocol_state_hash
                || entry.phase_schedule_digest_sha256 != expected.phase_schedule_digest_sha256
                || !entry.fsync_pass
            {
                bail!("fixed-cadence checkpoint frame lineage mismatch");
            }
            let bytes = read_regular_nofollow(&root.join(&entry.file), Some(entry.file_bytes))?;
            if sha256_bytes(&bytes) != entry.file_sha256 {
                bail!("fixed-cadence committed frame changed");
            }
            decode_and_verify_frame(&bytes, entry, expected, &view)?;
            previous = entry.file_sha256.clone();
        }
        Ok(Self {
            root,
            manifest_path,
            topology: view,
            manifest: state.manifest.clone(),
        })
    }

    pub fn create(
        output_dir: &Path,
        topo_stem: &str,
        topology: &PrismPrepTopology,
        context: FixedCadenceCoordinateContext,
    ) -> Result<Self> {
        validate_stem(topo_stem)?;
        validate_topology_binding(topology, &context)?;
        let topology_view = TopologyView::from_topology(topology)?;
        let root = fixed_cadence_coordinate_root(output_dir, topo_stem, context.stream_id);
        std::fs::create_dir_all(&root)?;
        fsync_directory(output_dir)?;
        let manifest_path = root.join("manifest.json");
        if manifest_path.exists() {
            bail!("fixed-cadence coordinate manifest already exists");
        }
        let context_digest_sha256 = context.digest_sha256()?;
        let manifest = FixedCadenceCoordinateManifest {
            schema_kind: "prism_fixed_cadence_coordinate_manifest".to_string(),
            schema_version: SCHEMA_VERSION,
            scheduled_frame_count: context.scheduled_frame_count(),
            persisted_frame_count: 0,
            decoded_frame_count: 0,
            indexed_frame_count: 0,
            context,
            context_digest_sha256,
            frames: Vec::new(),
            completed: false,
            scheduled_equals_persisted_equals_decoded_equals_indexed: false,
        };
        write_manifest_atomic(&root, &manifest_path, &manifest)?;
        Ok(Self {
            root,
            manifest_path,
            topology: topology_view,
            manifest,
        })
    }

    pub fn append(&mut self, input: FixedCadenceCoordinateInput) -> Result<()> {
        if self.manifest.completed {
            bail!("cannot append to completed fixed-cadence coordinates");
        }
        let ordinal = self.manifest.frames.len() as u64;
        let expected_step = self
            .manifest
            .context
            .start_step
            .checked_add(
                ordinal
                    .checked_mul(self.manifest.context.cadence_steps)
                    .context("fixed-cadence schedule overflow")?,
            )
            .context("fixed-cadence schedule overflow")?;
        if input.step_idx != expected_step
            || input.step_idx >= self.manifest.context.end_step_exclusive
            || input.protocol_phase > 4
            || input.positions_angstrom.len() != self.topology.n_atoms * 3
            || input
                .positions_angstrom
                .iter()
                .any(|value| !value.is_finite())
        {
            bail!("off-cadence, out-of-range, or incomplete coordinate frame");
        }
        let confirmation =
            self.manifest.context.mode == FixedCadenceCoordinateMode::CandidateConfirmation;
        if input.tracked_centroid_angstrom.is_some() {
            bail!("static laboratory-frame tracker centroids are prohibited");
        }
        let source_bytes = f32_bytes(&input.positions_angstrom);
        let source_full_frame_sha256 = sha256_bytes(&source_bytes);
        let analysis_positions = compact_analysis_positions(
            self.topology.n_atoms,
            &self.topology.analysis_ids,
            &input.positions_angstrom,
        )?;
        let analysis_bytes = f32_bytes(&analysis_positions);
        let analysis_coordinate_sha256 = sha256_bytes(&analysis_bytes);
        let body_fixed_transform = if confirmation {
            Some(compute_body_fixed_transform(
                &self.topology,
                &input.positions_angstrom,
                self.manifest
                    .context
                    .body_fixed_tracker_authority
                    .as_ref()
                    .context("confirmation context lost body-fixed authority")?,
            )?)
        } else {
            None
        };
        let tracked_centroid_angstrom = body_fixed_transform.as_ref().map(|transform| {
            transform
                .transformed_centroid_angstrom
                .map(|value| value as f32)
        });
        if tracked_centroid_angstrom
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
        {
            bail!("body-fixed transformed centroid exceeds persisted precision");
        }
        let (full_bytes, shell, excluded_population_scanned) = if confirmation {
            let centroid = tracked_centroid_angstrom.unwrap();
            let shell = exact_local_shell(
                &self.topology,
                &input.positions_angstrom,
                centroid,
                self.manifest.context.local_shell_cutoff_angstrom.unwrap(),
            )?;
            (
                source_bytes,
                shell,
                self.topology
                    .roles
                    .iter()
                    .filter(|role| role.analysis_id < 0)
                    .count() as u64,
            )
        } else {
            (Vec::new(), Vec::new(), 0)
        };
        let shell_bytes = encode_shell(&shell)?;
        let local_shell_sha256 = confirmation.then(|| sha256_bytes(&shell_bytes));
        let hydration_shell_sha256 = confirmation
            .then(|| shell_subset_sha256(&shell, |role| matches!(role, "WATER" | "ION")));
        let fragment_shell_sha256 =
            confirmation.then(|| shell_subset_sha256(&shell, |role| role == "LAMBDA_FRAGMENT"));
        let persisted_full_frame_sha256 = confirmation.then(|| sha256_bytes(&full_bytes));
        if persisted_full_frame_sha256
            .as_deref()
            .is_some_and(|hash| hash != source_full_frame_sha256)
        {
            bail!("persisted full frame differs from synchronized source frame");
        }

        let analysis_offset = HEADER_BYTES as u64;
        let full_offset = analysis_offset + analysis_bytes.len() as u64;
        let shell_offset = full_offset + full_bytes.len() as u64;
        let previous_file_sha256 = self
            .manifest
            .frames
            .last()
            .map(|entry| entry.file_sha256.clone())
            .unwrap_or_else(|| ZERO_SHA256.to_string());
        let mut bytes = vec![0u8; HEADER_BYTES];
        bytes.extend_from_slice(&analysis_bytes);
        bytes.extend_from_slice(&full_bytes);
        bytes.extend_from_slice(&shell_bytes);
        encode_header(
            &mut bytes[..HEADER_BYTES],
            HeaderFields {
                context_sha256: &self.manifest.context_digest_sha256,
                previous_sha256: &previous_file_sha256,
                stream_id: self.manifest.context.stream_id,
                protocol_phase: input.protocol_phase,
                mode: self.manifest.context.mode,
                step_idx: input.step_idx,
                analysis_atom_count: (analysis_positions.len() / 3) as u64,
                full_atom_count: if confirmation {
                    self.topology.n_atoms as u64
                } else {
                    0
                },
                shell_atom_count: shell.len() as u64,
                excluded_population_scanned,
                analysis_offset,
                analysis_bytes: analysis_bytes.len() as u64,
                full_offset,
                full_bytes: full_bytes.len() as u64,
                shell_offset,
                shell_bytes: shell_bytes.len() as u64,
                source_full_frame_sha256: &source_full_frame_sha256,
                analysis_coordinate_sha256: &analysis_coordinate_sha256,
                persisted_full_frame_sha256: persisted_full_frame_sha256.as_deref(),
                local_shell_sha256: local_shell_sha256.as_deref(),
                hydration_shell_sha256: hydration_shell_sha256.as_deref(),
                fragment_shell_sha256: fragment_shell_sha256.as_deref(),
                tracked_centroid_angstrom,
            },
        )?;
        let file_name = format!("frame_{ordinal:08}.ghostcadence");
        let final_path = self.root.join(&file_name);
        write_segment_atomic(&self.root, &final_path, &bytes)?;
        let file_sha256 = sha256_bytes(&bytes);
        let entry = FixedCadenceFrameEntry {
            segment_ordinal: ordinal,
            file: file_name,
            file_sha256,
            file_bytes: bytes.len() as u64,
            previous_file_sha256,
            context_digest_sha256: self.manifest.context_digest_sha256.clone(),
            stream_id: self.manifest.context.stream_id,
            step_idx: input.step_idx,
            protocol_phase: input.protocol_phase,
            thermodynamic_condition_id: self.manifest.context.thermodynamic_condition_id.clone(),
            protocol_state_hash: self.manifest.context.protocol_state_hash.clone(),
            phase_schedule_digest_sha256: self
                .manifest
                .context
                .phase_schedule_digest_sha256
                .clone(),
            source_full_frame_sha256,
            analysis_coordinate_sha256,
            persisted_full_frame_sha256,
            local_shell_sha256,
            hydration_shell_sha256,
            fragment_shell_sha256,
            analysis_atom_count: (analysis_positions.len() / 3) as u64,
            full_atom_count: if confirmation {
                self.topology.n_atoms as u64
            } else {
                0
            },
            local_shell_atom_count: shell.len() as u64,
            excluded_population_scanned,
            tracked_centroid_angstrom,
            body_fixed_transform,
            fsync_pass: true,
        };
        self.manifest.frames.push(entry);
        self.manifest.persisted_frame_count = self.manifest.frames.len() as u64;
        write_manifest_atomic(&self.root, &self.manifest_path, &self.manifest)
    }

    pub fn complete(&mut self) -> Result<()> {
        if self.manifest.persisted_frame_count != self.manifest.scheduled_frame_count {
            bail!(
                "fixed-cadence capture incomplete: scheduled={} persisted={}",
                self.manifest.scheduled_frame_count,
                self.manifest.persisted_frame_count,
            );
        }
        self.manifest.decoded_frame_count = self.manifest.persisted_frame_count;
        self.manifest.indexed_frame_count = self.manifest.persisted_frame_count;
        self.manifest.completed = true;
        self.manifest
            .scheduled_equals_persisted_equals_decoded_equals_indexed = true;
        write_manifest_atomic(&self.root, &self.manifest_path, &self.manifest)?;
        fsync_directory(&self.root)
    }

    pub fn manifest(&self) -> &FixedCadenceCoordinateManifest {
        &self.manifest
    }
}

pub fn load_verified_fixed_cadence_coordinates(
    root: &Path,
    topology: &PrismPrepTopology,
    expected_context: &FixedCadenceCoordinateContext,
) -> Result<FixedCadenceCoordinateLoad> {
    validate_topology_binding(topology, expected_context)?;
    let manifest_path = root.join("manifest.json");
    let manifest_bytes = read_regular_nofollow(&manifest_path, None)?;
    let manifest: FixedCadenceCoordinateManifest = serde_json::from_slice(&manifest_bytes)?;
    if &manifest.context != expected_context
        || manifest.schema_kind != "prism_fixed_cadence_coordinate_manifest"
        || manifest.schema_version != SCHEMA_VERSION
        || manifest.context_digest_sha256 != expected_context.digest_sha256()?
        || !manifest.completed
        || !manifest.scheduled_equals_persisted_equals_decoded_equals_indexed
        || manifest.scheduled_frame_count != expected_context.scheduled_frame_count()
        || manifest.persisted_frame_count != manifest.scheduled_frame_count
        || manifest.decoded_frame_count != manifest.scheduled_frame_count
        || manifest.indexed_frame_count != manifest.scheduled_frame_count
        || manifest.frames.len() as u64 != manifest.scheduled_frame_count
    {
        bail!("fixed-cadence manifest conservation or context failure");
    }
    if canonical_json_bytes(&manifest)? != manifest_bytes {
        bail!("fixed-cadence manifest bytes are not canonical producer bytes");
    }
    let topology_view = TopologyView::from_topology(topology)?;
    let mut frames = BTreeMap::new();
    let mut previous_sha = ZERO_SHA256.to_string();
    for (ordinal, entry) in manifest.frames.iter().enumerate() {
        let expected_step =
            expected_context.start_step + ordinal as u64 * expected_context.cadence_steps;
        if entry.segment_ordinal != ordinal as u64
            || entry.step_idx != expected_step
            || entry.stream_id != expected_context.stream_id
            || entry.previous_file_sha256 != previous_sha
            || entry.context_digest_sha256 != manifest.context_digest_sha256
            || entry.thermodynamic_condition_id != expected_context.thermodynamic_condition_id
            || entry.protocol_state_hash != expected_context.protocol_state_hash
            || entry.phase_schedule_digest_sha256 != expected_context.phase_schedule_digest_sha256
            || !entry.fsync_pass
        {
            bail!("fixed-cadence frame ordering or lineage mismatch");
        }
        let bytes = read_regular_nofollow(&root.join(&entry.file), Some(entry.file_bytes))?;
        if sha256_bytes(&bytes) != entry.file_sha256 {
            bail!("fixed-cadence frame file SHA mismatch");
        }
        let decoded = decode_and_verify_frame(&bytes, entry, expected_context, &topology_view)?;
        if frames.insert(entry.step_idx, decoded).is_some() {
            bail!("duplicate fixed-cadence stream/step frame");
        }
        previous_sha = entry.file_sha256.clone();
    }
    if frames.len() as u64 != manifest.scheduled_frame_count {
        bail!("fixed-cadence decoded/indexed population mismatch");
    }
    Ok(FixedCadenceCoordinateLoad { manifest, frames })
}

/// Load the exact candidate-confirmation coordinate population for a run root.
///
/// Only direct child directories emitted by `fixed_cadence_coordinate_root`
/// are considered. Every matching directory must be a real directory, its
/// manifest must be a regular non-symlink file, the requested stream set must
/// match exactly, and every stream is reloaded through the strict per-stream
/// conservation loader before the population digest is computed.
pub fn load_verified_candidate_confirmation_population(
    run_root: &Path,
    topology: &PrismPrepTopology,
    expected_stream_ids: &BTreeSet<u16>,
) -> Result<FixedCadenceCoordinatePopulationLoad> {
    load_verified_fixed_cadence_population(
        run_root,
        topology,
        expected_stream_ids,
        FixedCadenceCoordinateMode::CandidateConfirmation,
    )
}

/// Load an exact whole-protein discovery coordinate population. This path is
/// mode-strict: a confirmation stream, candidate tracker, local shell, or
/// body-frame authority in the same run root fails the entire population.
pub fn load_verified_whole_protein_discovery_population(
    run_root: &Path,
    topology: &PrismPrepTopology,
    expected_stream_ids: &BTreeSet<u16>,
) -> Result<FixedCadenceCoordinatePopulationLoad> {
    load_verified_fixed_cadence_population(
        run_root,
        topology,
        expected_stream_ids,
        FixedCadenceCoordinateMode::WholeProteinDiscovery,
    )
}

fn load_verified_fixed_cadence_population(
    run_root: &Path,
    topology: &PrismPrepTopology,
    expected_stream_ids: &BTreeSet<u16>,
    expected_mode: FixedCadenceCoordinateMode,
) -> Result<FixedCadenceCoordinatePopulationLoad> {
    if expected_stream_ids.is_empty() {
        bail!("fixed-cadence population requires a nonempty stream set");
    }
    let root_metadata = std::fs::symlink_metadata(run_root)?;
    if root_metadata.file_type().is_symlink() || !root_metadata.is_dir() {
        bail!("fixed-cadence run root is not a real directory");
    }
    let canonical_root = std::fs::canonicalize(run_root)?;
    let mut discovered = Vec::<(u16, PathBuf, FixedCadenceCoordinateContext, String)>::new();
    for entry in std::fs::read_dir(&canonical_root)? {
        let entry = entry?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow::anyhow!("non-UTF8 fixed-cadence directory name"))?;
        if !name.ends_with("_fixed_cadence_coordinates") {
            continue;
        }
        let path = entry.path();
        let metadata = std::fs::symlink_metadata(&path)?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            bail!("matching fixed-cadence population entry is not a real directory");
        }
        let canonical_path = std::fs::canonicalize(&path)?;
        if canonical_path.parent() != Some(canonical_root.as_path()) {
            bail!("fixed-cadence population directory escapes the run root");
        }
        let manifest_path = canonical_path.join("manifest.json");
        let manifest_bytes = read_regular_nofollow(&manifest_path, None)?;
        let manifest: FixedCadenceCoordinateManifest = serde_json::from_slice(&manifest_bytes)?;
        if canonical_json_bytes(&manifest)? != manifest_bytes {
            bail!("fixed-cadence population manifest is not canonical producer bytes");
        }
        if manifest.context.mode != expected_mode {
            bail!("fixed-cadence population contains a stream from the wrong mode");
        }
        if expected_mode == FixedCadenceCoordinateMode::WholeProteinDiscovery
            && (manifest.context.candidate_tracker_sha256.is_some()
                || manifest.context.local_shell_cutoff_angstrom.is_some()
                || manifest.context.body_fixed_tracker_authority.is_some())
        {
            bail!("whole-protein discovery population contains candidate authority");
        }
        let stream_id = manifest.context.stream_id;
        discovered.push((
            stream_id,
            canonical_path,
            manifest.context,
            sha256_bytes(&manifest_bytes),
        ));
    }
    if discovered.is_empty() {
        bail!("fixed-cadence coordinate population is empty");
    }
    discovered.sort_by_key(|row| row.0);
    let mut streams = BTreeMap::new();
    let mut digest_rows = Vec::with_capacity(discovered.len());
    let mut observed = BTreeSet::new();
    let mut total_scheduled = 0u64;
    let mut total_persisted = 0u64;
    let mut total_decoded = 0u64;
    let mut total_indexed = 0u64;
    let mut shared_design_identities = BTreeSet::new();
    for (stream_id, root, context, manifest_sha256) in discovered {
        if !observed.insert(stream_id) {
            bail!("duplicate stream in fixed-cadence coordinate population");
        }
        shared_design_identities.insert(canonical_sha256(&serde_json::json!({
            "schema_kind": "prism_fixed_cadence_shared_design_identity",
            "schema_version": SCHEMA_VERSION,
            "run_id": &context.run_id,
            "replica_id": &context.replica_id,
            "replica_seed": context.replica_seed,
            "mode": context.mode,
            "topology_sha256": &context.topology_sha256,
            "selected_engine_sha256": &context.selected_engine_sha256,
            "native_pathb_runtime_request_sha256": &context.native_pathb_runtime_request_sha256,
            "integration_timebase_context_sha256": &context.integration_timebase_context_sha256,
            "integration_timebase_authority_sha256": &context.integration_timebase_authority_sha256,
            "integration_timestep_femtoseconds": context.integration_timestep_femtoseconds,
            "candidate_lag_steps": &context.candidate_lag_steps,
            "cadence_steps": context.cadence_steps,
            "analysis_residue_contract_sha256": &context.analysis_residue_contract_sha256,
            "atom_analysis_residue_ids_sha256": &context.atom_analysis_residue_ids_sha256,
            "molecule_identity_table_sha256": &context.molecule_identity_table_sha256,
            "periodic_cell_contract_sha256": &context.periodic_cell_contract_sha256,
            "candidate_tracker_sha256": &context.candidate_tracker_sha256,
            "local_shell_cutoff_angstrom": context.local_shell_cutoff_angstrom,
            "body_fixed_tracker_authority_sha256": context
                .body_fixed_tracker_authority
                .as_ref()
                .map(|authority| authority.authority_sha256.as_str()),
        }))?);
        let loaded = load_verified_fixed_cadence_coordinates(&root, topology, &context)?;
        let manifest = &loaded.manifest;
        total_scheduled = total_scheduled
            .checked_add(manifest.scheduled_frame_count)
            .context("fixed-cadence scheduled population overflow")?;
        total_persisted = total_persisted
            .checked_add(manifest.persisted_frame_count)
            .context("fixed-cadence persisted population overflow")?;
        total_decoded = total_decoded
            .checked_add(manifest.decoded_frame_count)
            .context("fixed-cadence decoded population overflow")?;
        total_indexed = total_indexed
            .checked_add(manifest.indexed_frame_count)
            .context("fixed-cadence indexed population overflow")?;
        digest_rows.push(FixedCadencePopulationDigestRow {
            stream_id,
            manifest_sha256,
            context_digest_sha256: manifest.context_digest_sha256.clone(),
            scheduled_frame_count: manifest.scheduled_frame_count,
            persisted_frame_count: manifest.persisted_frame_count,
            decoded_frame_count: manifest.decoded_frame_count,
            indexed_frame_count: manifest.indexed_frame_count,
        });
        streams.insert(stream_id, loaded);
    }
    if observed != *expected_stream_ids
        || shared_design_identities.len() != 1
        || total_scheduled != total_persisted
        || total_scheduled != total_decoded
        || total_scheduled != total_indexed
    {
        bail!("fixed-cadence stream/design identity or population conservation failed");
    }
    let requested_stream_ids = expected_stream_ids.iter().copied().collect::<Vec<_>>();
    let population_sha256 = canonical_sha256(&FixedCadencePopulationDigest {
        schema_kind: "prism_fixed_cadence_coordinate_population",
        schema_version: SCHEMA_VERSION,
        requested_stream_ids: &requested_stream_ids,
        total_scheduled_frame_count: total_scheduled,
        total_persisted_frame_count: total_persisted,
        total_decoded_frame_count: total_decoded,
        total_indexed_frame_count: total_indexed,
        streams: &digest_rows,
    })?;
    Ok(FixedCadenceCoordinatePopulationLoad {
        streams,
        requested_stream_ids,
        total_scheduled_frame_count: total_scheduled,
        total_persisted_frame_count: total_persisted,
        total_decoded_frame_count: total_decoded,
        total_indexed_frame_count: total_indexed,
        population_sha256,
    })
}

pub fn fixed_cadence_coordinate_root(
    output_dir: &Path,
    topo_stem: &str,
    stream_id: u16,
) -> PathBuf {
    output_dir.join(format!(
        "{topo_stem}_stream{stream_id:02}_fixed_cadence_coordinates"
    ))
}

struct HeaderFields<'a> {
    context_sha256: &'a str,
    previous_sha256: &'a str,
    stream_id: u16,
    protocol_phase: u8,
    mode: FixedCadenceCoordinateMode,
    step_idx: u64,
    analysis_atom_count: u64,
    full_atom_count: u64,
    shell_atom_count: u64,
    excluded_population_scanned: u64,
    analysis_offset: u64,
    analysis_bytes: u64,
    full_offset: u64,
    full_bytes: u64,
    shell_offset: u64,
    shell_bytes: u64,
    source_full_frame_sha256: &'a str,
    analysis_coordinate_sha256: &'a str,
    persisted_full_frame_sha256: Option<&'a str>,
    local_shell_sha256: Option<&'a str>,
    hydration_shell_sha256: Option<&'a str>,
    fragment_shell_sha256: Option<&'a str>,
    tracked_centroid_angstrom: Option<[f32; 3]>,
}

fn encode_header(header: &mut [u8], fields: HeaderFields<'_>) -> Result<()> {
    if header.len() != HEADER_BYTES {
        bail!("fixed-cadence header width mismatch");
    }
    header[0..8].copy_from_slice(&MAGIC);
    put_u32(header, 8, SCHEMA_VERSION);
    put_u32(header, 12, HEADER_BYTES as u32);
    put_sha256(header, 16, fields.context_sha256)?;
    put_sha256(header, 48, fields.previous_sha256)?;
    header[80..82].copy_from_slice(&fields.stream_id.to_le_bytes());
    header[82] = fields.protocol_phase;
    header[83] = match fields.mode {
        FixedCadenceCoordinateMode::WholeProteinDiscovery => 1,
        FixedCadenceCoordinateMode::CandidateConfirmation => 2,
    };
    put_u64(header, 88, fields.step_idx);
    put_u64(header, 96, fields.analysis_atom_count);
    put_u64(header, 104, fields.full_atom_count);
    put_u64(header, 112, fields.shell_atom_count);
    put_u64(header, 120, fields.excluded_population_scanned);
    put_u64(header, 128, fields.analysis_offset);
    put_u64(header, 136, fields.analysis_bytes);
    put_u64(header, 144, fields.full_offset);
    put_u64(header, 152, fields.full_bytes);
    put_u64(header, 160, fields.shell_offset);
    put_u64(header, 168, fields.shell_bytes);
    put_sha256(header, 176, fields.source_full_frame_sha256)?;
    put_sha256(header, 208, fields.analysis_coordinate_sha256)?;
    put_optional_sha256(header, 240, fields.persisted_full_frame_sha256)?;
    put_optional_sha256(header, 272, fields.local_shell_sha256)?;
    put_optional_sha256(header, 304, fields.hydration_shell_sha256)?;
    put_optional_sha256(header, 336, fields.fragment_shell_sha256)?;
    if let Some(centroid) = fields.tracked_centroid_angstrom {
        header[368] = 1;
        for (axis, value) in centroid.into_iter().enumerate() {
            header[372 + axis * 4..376 + axis * 4].copy_from_slice(&value.to_le_bytes());
        }
    }
    Ok(())
}

fn decode_and_verify_frame(
    bytes: &[u8],
    entry: &FixedCadenceFrameEntry,
    context: &FixedCadenceCoordinateContext,
    topology: &TopologyView,
) -> Result<VerifiedFixedCadenceFrame> {
    if bytes.len() < HEADER_BYTES
        || bytes[0..8] != MAGIC
        || read_u32(bytes, 8)? != SCHEMA_VERSION
        || read_u32(bytes, 12)? as usize != HEADER_BYTES
        || read_sha256(bytes, 16)? != context.digest_sha256()?
        || read_sha256(bytes, 48)? != entry.previous_file_sha256
        || u16::from_le_bytes(bytes[80..82].try_into()?) != entry.stream_id
        || bytes[82] != entry.protocol_phase
        || read_u64(bytes, 88)? != entry.step_idx
    {
        bail!("fixed-cadence frame header identity mismatch");
    }
    let expected_mode = match context.mode {
        FixedCadenceCoordinateMode::WholeProteinDiscovery => 1,
        FixedCadenceCoordinateMode::CandidateConfirmation => 2,
    };
    if bytes[83] != expected_mode {
        bail!("fixed-cadence frame mode mismatch");
    }
    let analysis_atom_count = read_u64(bytes, 96)?;
    let full_atom_count = read_u64(bytes, 104)?;
    let shell_atom_count = read_u64(bytes, 112)?;
    let excluded_population_scanned = read_u64(bytes, 120)?;
    let analysis = byte_range(bytes, read_u64(bytes, 128)?, read_u64(bytes, 136)?)?;
    let full = byte_range(bytes, read_u64(bytes, 144)?, read_u64(bytes, 152)?)?;
    let shell_bytes = byte_range(bytes, read_u64(bytes, 160)?, read_u64(bytes, 168)?)?;
    if analysis_atom_count != entry.analysis_atom_count
        || full_atom_count != entry.full_atom_count
        || shell_atom_count != entry.local_shell_atom_count
        || excluded_population_scanned != entry.excluded_population_scanned
        || sha256_bytes(analysis) != entry.analysis_coordinate_sha256
        || read_sha256(bytes, 176)? != entry.source_full_frame_sha256
        || read_sha256(bytes, 208)? != entry.analysis_coordinate_sha256
        || read_optional_sha256(bytes, 240)? != entry.persisted_full_frame_sha256
        || read_optional_sha256(bytes, 272)? != entry.local_shell_sha256
        || read_optional_sha256(bytes, 304)? != entry.hydration_shell_sha256
        || read_optional_sha256(bytes, 336)? != entry.fragment_shell_sha256
    {
        bail!("fixed-cadence frame payload descriptor mismatch");
    }
    let analysis_positions = decode_f32(analysis)?;
    if analysis_positions.len() != analysis_atom_count as usize * 3 {
        bail!("fixed-cadence analysis frame shape mismatch");
    }
    let local_shell = decode_shell(shell_bytes)?;
    if local_shell.len() as u64 != shell_atom_count {
        bail!("fixed-cadence shell count mismatch");
    }
    let full_positions = if context.mode == FixedCadenceCoordinateMode::CandidateConfirmation {
        let positions = decode_f32(full)?;
        let full_sha256 = sha256_bytes(full);
        if positions.len() != topology.n_atoms * 3
            || full_atom_count != topology.n_atoms as u64
            || full_sha256 != entry.source_full_frame_sha256
            || entry.persisted_full_frame_sha256.as_deref() != Some(full_sha256.as_str())
        {
            bail!("confirmation frame does not retain its complete source population");
        }
        let centroid = entry
            .tracked_centroid_angstrom
            .context("confirmation frame lacks tracked centroid")?;
        let authority = context
            .body_fixed_tracker_authority
            .as_ref()
            .context("confirmation context lacks body-fixed authority")?;
        let expected_transform = compute_body_fixed_transform(topology, &positions, authority)?;
        expected_transform.verify(authority)?;
        if entry.body_fixed_transform.as_ref() != Some(&expected_transform)
            || centroid
                != expected_transform
                    .transformed_centroid_angstrom
                    .map(|value| value as f32)
        {
            bail!("confirmation centroid is not the exact body-fixed transform result");
        }
        let expected_shell = exact_local_shell(
            topology,
            &positions,
            centroid,
            context.local_shell_cutoff_angstrom.unwrap(),
        )?;
        let expected_excluded = topology
            .roles
            .iter()
            .filter(|role| role.analysis_id < 0)
            .count() as u64;
        let shell_sha256 = sha256_bytes(shell_bytes);
        let hydration_sha256 =
            shell_subset_sha256(&local_shell, |role| matches!(role, "WATER" | "ION"));
        let fragment_sha256 = shell_subset_sha256(&local_shell, |role| role == "LAMBDA_FRAGMENT");
        if expected_shell != local_shell
            || excluded_population_scanned != expected_excluded
            || entry.local_shell_sha256.as_deref() != Some(shell_sha256.as_str())
            || entry.hydration_shell_sha256.as_deref() != Some(hydration_sha256.as_str())
            || entry.fragment_shell_sha256.as_deref() != Some(fragment_sha256.as_str())
        {
            bail!("confirmation shell is not a complete full-population reconstruction");
        }
        Some(positions)
    } else {
        if !full.is_empty()
            || !shell_bytes.is_empty()
            || full_atom_count != 0
            || shell_atom_count != 0
            || excluded_population_scanned != 0
            || entry.tracked_centroid_angstrom.is_some()
            || entry.body_fixed_transform.is_some()
            || entry.persisted_full_frame_sha256.is_some()
            || entry.local_shell_sha256.is_some()
            || entry.hydration_shell_sha256.is_some()
            || entry.fragment_shell_sha256.is_some()
        {
            bail!("discovery frame crossed the confirmation evidence boundary");
        }
        None
    };
    Ok(VerifiedFixedCadenceFrame {
        entry: entry.clone(),
        analysis_positions_angstrom: analysis_positions,
        full_positions_angstrom: full_positions,
        local_shell,
    })
}

fn validate_topology_binding(
    topology: &PrismPrepTopology,
    context: &FixedCadenceCoordinateContext,
) -> Result<()> {
    context.verify()?;
    topology
        .validate_analysis_residue_axis()
        .map_err(anyhow::Error::msg)?;
    let analysis = topology.analysis_residue_contract.as_ref().unwrap();
    let molecules = topology.molecule_identity_table.as_ref().unwrap();
    let periodic = topology.periodic_cell.as_ref().unwrap();
    if analysis.contract_sha256 != context.analysis_residue_contract_sha256
        || analysis.atom_analysis_residue_ids_sha256 != context.atom_analysis_residue_ids_sha256
        || molecules.table_sha256 != context.molecule_identity_table_sha256
        || periodic.contract_sha256.as_deref()
            != Some(context.periodic_cell_contract_sha256.as_str())
    {
        bail!("fixed-cadence context differs from prepared topology");
    }
    if let Some(authority) = &context.body_fixed_tracker_authority {
        let periodic_matches = match authority.boundary_mode {
            BodyFixedBoundaryMode::PeriodicTriclinicMinimumImage => {
                periodic.periodic && periodic.vectors_angstrom.is_some()
            }
            BodyFixedBoundaryMode::DeclaredNonperiodicEuclidean => {
                !periodic.periodic && periodic.vectors_angstrom.is_none()
            }
        };
        if !periodic_matches || authority.chain_mapping_receipt_sha256 != analysis.contract_sha256 {
            bail!("body-fixed boundary/chain authority differs from prepared topology");
        }
        let by_analysis_id = analysis
            .analysis_residues
            .iter()
            .map(|row| (row.analysis_residue_id, row))
            .collect::<BTreeMap<_, _>>();
        for anchor in &authority.anchor_atoms {
            let atom = anchor.atom_index;
            if atom >= topology.n_atoms
                || topology.atom_names.get(atom) != Some(&anchor.atom_name)
                || topology.chain_ids.get(atom) != Some(&anchor.chain_id)
                || topology.residue_names.get(atom) != Some(&anchor.residue_name)
                || topology
                    .atom_analysis_residue_ids
                    .as_ref()
                    .and_then(|ids| ids.get(atom))
                    .and_then(|id| u32::try_from(*id).ok())
                    != Some(anchor.analysis_residue_id)
            {
                bail!("body-fixed anchor atom differs from prepared topology");
            }
            let row = by_analysis_id
                .get(&(anchor.analysis_residue_id as usize))
                .context("body-fixed anchor is absent from signed analysis mapping")?;
            if row.chain_id != anchor.chain_id
                || row.pdb_residue_id != anchor.pdb_residue_id
                || row.residue_name != anchor.residue_name
            {
                bail!("body-fixed anchor chain/residue identity is not signed");
            }
        }
    }
    Ok(())
}

fn compact_analysis_positions(
    n_atoms: usize,
    analysis_ids: &[i32],
    positions: &[f32],
) -> Result<Vec<f32>> {
    if positions.len() != n_atoms * 3 || analysis_ids.len() != n_atoms {
        bail!("analysis compaction source is incomplete");
    }
    let mut compact = Vec::new();
    for (atom, analysis_id) in analysis_ids.iter().enumerate() {
        if *analysis_id >= 0 {
            compact.extend_from_slice(&positions[atom * 3..atom * 3 + 3]);
        }
    }
    Ok(compact)
}

fn compute_body_fixed_transform(
    topology: &TopologyView,
    positions: &[f32],
    authority: &BodyFixedTrackerAuthority,
) -> Result<BodyFixedFrameTransform> {
    authority.verify()?;
    if positions.len() != topology.n_atoms * 3 {
        bail!("body-fixed transform source frame is incomplete");
    }
    match (
        authority.boundary_mode,
        topology.box_matrix,
        topology.inverse_box,
    ) {
        (BodyFixedBoundaryMode::PeriodicTriclinicMinimumImage, Some(_), Some(_))
        | (BodyFixedBoundaryMode::DeclaredNonperiodicEuclidean, None, None) => {}
        _ => bail!("body-fixed transform boundary differs from topology"),
    }
    let raw = authority
        .anchor_atoms
        .iter()
        .map(|anchor| {
            if anchor.atom_index >= topology.n_atoms {
                bail!("body-fixed anchor index escapes topology");
            }
            Ok([
                f64::from(positions[3 * anchor.atom_index]),
                f64::from(positions[3 * anchor.atom_index + 1]),
                f64::from(positions[3 * anchor.atom_index + 2]),
            ])
        })
        .collect::<Result<Vec<_>>>()?;
    let current_origin = raw[0];
    let current = raw
        .iter()
        .map(|point| match (topology.box_matrix, topology.inverse_box) {
            (Some(matrix), Some(inverse)) => Ok(add(
                current_origin,
                minimum_image(sub(*point, current_origin), matrix, inverse)?,
            )),
            (None, None) => Ok(*point),
            _ => bail!("body-fixed transform has an incomplete periodic metric"),
        })
        .collect::<Result<Vec<_>>>()?;
    let reference = authority
        .anchor_atoms
        .iter()
        .map(|anchor| anchor.reference_position_angstrom)
        .collect::<Vec<_>>();
    let fit = measure_body_frame_fit(
        &reference,
        &current,
        authority.reference_candidate_centroid_angstrom,
    )?;
    let mut transform = BodyFixedFrameTransform {
        schema_kind: "prism_body_fixed_frame_transform".to_string(),
        schema_version: "1.0.0".to_string(),
        transform_sha256: String::new(),
        authority_sha256: authority.authority_sha256.clone(),
        boundary_mode: authority.boundary_mode,
        reference_to_frame_rotation: fit.rotation,
        reference_to_frame_translation_angstrom: fit.translation,
        transformed_centroid_angstrom: fit.centroid,
        anchor_fit_rmsd_angstrom: fit.rmsd,
        anchor_fit_max_residual_angstrom: fit.max_residual,
        anchor_count: authority.anchor_atoms.len(),
    };
    transform.transform_sha256 = transform.body_sha256()?;
    transform.verify(authority)?;
    Ok(transform)
}

/// Measured rigid transport shared by prospective tracking and retrospective
/// coordinate analysis. Residuals are observations, not admission thresholds.
#[derive(Clone, Debug, Serialize)]
pub struct MeasuredBodyFrameFit {
    pub rotation: [[f64; 3]; 3],
    pub translation: [f64; 3],
    pub centroid: [f64; 3],
    pub rmsd: f64,
    pub max_residual: f64,
}
impl MeasuredBodyFrameFit {
    /// Map a current-frame point into the declared reference molecular frame.
    pub fn point_in_reference(&self, point: [f64; 3]) -> [f64; 3] {
        mat_vec(transpose(self.rotation), sub(point, self.translation))
    }
}

/// Reuse the native ordered-anchor body frame. The caller must supply points
/// under an explicit boundary convention; this function never unwraps them.
pub fn measure_body_frame_fit(
    reference: &[[f64; 3]],
    current: &[[f64; 3]],
    centroid: [f64; 3],
) -> Result<MeasuredBodyFrameFit> {
    if reference.len() < 3
        || reference.len() != current.len()
        || reference
            .iter()
            .chain(current)
            .flatten()
            .chain(centroid.iter())
            .any(|x| !x.is_finite())
    {
        bail!("incomplete or nonfinite body-frame fit population");
    }
    let (_, reference_basis) = body_frame_basis([reference[0], reference[1], reference[2]])
        .context("body-fixed reference anchors are degenerate")?;
    let (_, current_basis) = body_frame_basis([current[0], current[1], current[2]])
        .context("body-fixed frame anchors are degenerate")?;
    let rotation = mat_mul(current_basis, transpose(reference_basis));
    let translation = sub(current[0], mat_vec(rotation, reference[0]));
    let transformed_centroid = add(mat_vec(rotation, centroid), translation);
    let residuals = reference
        .iter()
        .zip(current)
        .map(|(reference_point, current_point)| {
            let fitted = add(mat_vec(rotation, *reference_point), translation);
            dot(sub(*current_point, fitted), sub(*current_point, fitted)).sqrt()
        })
        .collect::<Vec<_>>();
    let rmsd =
        (residuals.iter().map(|value| value * value).sum::<f64>() / residuals.len() as f64).sqrt();
    let max_residual = residuals.iter().copied().fold(0.0, f64::max);
    Ok(MeasuredBodyFrameFit {
        rotation,
        translation,
        centroid: transformed_centroid,
        rmsd,
        max_residual,
    })
}

fn body_frame_basis(points: [[f64; 3]; 3]) -> Result<([f64; 3], [[f64; 3]; 3])> {
    if points.iter().flatten().any(|value| !value.is_finite()) {
        bail!("body-frame anchors contain nonfinite coordinates");
    }
    let x = normalize(sub(points[1], points[0]))
        .context("body-frame origin/x anchors are coincident")?;
    let plane = sub(points[2], points[0]);
    let z = normalize(cross(x, plane)).context("body-frame anchors are collinear")?;
    let y = normalize(cross(z, x)).context("body-frame basis is degenerate")?;
    let basis = [[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]];
    Ok((points[0], basis))
}

fn normalize(vector: [f64; 3]) -> Option<[f64; 3]> {
    let norm = dot(vector, vector).sqrt();
    (norm.is_finite() && norm > 1.0e-12).then(|| vector.map(|value| value / norm))
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn transpose(matrix: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|row| std::array::from_fn(|column| matrix[column][row]))
}

fn mat_mul(left: [[f64; 3]; 3], right: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            (0..3)
                .map(|axis| left[row][axis] * right[axis][column])
                .sum()
        })
    })
}

fn add(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| left[axis] + right[axis])
}

fn exact_local_shell(
    topology: &TopologyView,
    positions: &[f32],
    centroid: [f32; 3],
    cutoff: f64,
) -> Result<Vec<ExactLocalShellAtom>> {
    if positions.len() != topology.n_atoms * 3 || !cutoff.is_finite() || cutoff <= 0.0 {
        bail!("local-shell source frame or cutoff is invalid");
    }
    let center = [centroid[0] as f64, centroid[1] as f64, centroid[2] as f64];
    let mut shell = Vec::new();
    for atom in 0..topology.n_atoms {
        let role = &topology.roles[atom];
        if role.analysis_id >= 0 {
            continue;
        }
        let point = [
            positions[3 * atom] as f64,
            positions[3 * atom + 1] as f64,
            positions[3 * atom + 2] as f64,
        ];
        let delta = match (topology.box_matrix, topology.inverse_box) {
            (Some(matrix), Some(inverse)) => minimum_image(sub(point, center), matrix, inverse)?,
            (None, None) => sub(point, center),
            _ => bail!("incomplete fixed-cadence spatial boundary"),
        };
        if dot(delta, delta) <= cutoff * cutoff {
            shell.push(ExactLocalShellAtom {
                atom_index: atom.try_into().context("shell atom index exceeds u32")?,
                molecule_id: role.molecule_id,
                role: role.role.clone(),
                position_angstrom: [
                    positions[3 * atom],
                    positions[3 * atom + 1],
                    positions[3 * atom + 2],
                ],
            });
        }
    }
    Ok(shell)
}

fn shell_subset_sha256<F>(shell: &[ExactLocalShellAtom], mut include: F) -> String
where
    F: FnMut(&str) -> bool,
{
    let selected = shell
        .iter()
        .filter(|atom| include(&atom.role))
        .cloned()
        .collect::<Vec<_>>();
    sha256_bytes(&encode_shell(&selected).expect("validated shell role"))
}

fn encode_shell(shell: &[ExactLocalShellAtom]) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(shell.len() * SHELL_ROW_BYTES);
    for atom in shell {
        let role = match atom.role.as_str() {
            "WATER" => 1u8,
            "ION" => 2,
            "LAMBDA_FRAGMENT" => 3,
            "OTHER" => 4,
            other => bail!("unsupported shell role {other}"),
        };
        bytes.extend_from_slice(&atom.atom_index.to_le_bytes());
        bytes.extend_from_slice(&atom.molecule_id.to_le_bytes());
        bytes.push(role);
        bytes.extend_from_slice(&[0u8; 3]);
        for value in atom.position_angstrom {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&[0u8; 4]);
    }
    Ok(bytes)
}

fn decode_shell(bytes: &[u8]) -> Result<Vec<ExactLocalShellAtom>> {
    if bytes.len() % SHELL_ROW_BYTES != 0 {
        bail!("truncated fixed-cadence shell payload");
    }
    bytes
        .chunks_exact(SHELL_ROW_BYTES)
        .map(|row| {
            if row[13..16].iter().any(|value| *value != 0)
                || row[28..32].iter().any(|value| *value != 0)
            {
                bail!("nonzero fixed-cadence shell reserved bytes");
            }
            let role = match row[12] {
                1 => "WATER",
                2 => "ION",
                3 => "LAMBDA_FRAGMENT",
                4 => "OTHER",
                _ => bail!("invalid fixed-cadence shell role"),
            };
            Ok(ExactLocalShellAtom {
                atom_index: u32::from_le_bytes(row[0..4].try_into()?),
                molecule_id: u64::from_le_bytes(row[4..12].try_into()?),
                role: role.to_string(),
                position_angstrom: [
                    f32::from_le_bytes(row[16..20].try_into()?),
                    f32::from_le_bytes(row[20..24].try_into()?),
                    f32::from_le_bytes(row[24..28].try_into()?),
                ],
            })
        })
        .collect()
}

fn decode_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        bail!("truncated fixed-cadence float payload");
    }
    let values = bytes
        .chunks_exact(4)
        .map(|row| f32::from_le_bytes(row.try_into().unwrap()))
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        bail!("nonfinite fixed-cadence coordinate");
    }
    Ok(values)
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn write_segment_atomic(root: &Path, final_path: &Path, bytes: &[u8]) -> Result<()> {
    if final_path.exists() {
        bail!("fixed-cadence segment target already exists");
    }
    let name = final_path
        .file_name()
        .and_then(|name| name.to_str())
        .context("invalid fixed-cadence segment filename")?;
    let partial_path = root.join(format!(".{name}.partial"));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(&partial_path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    std::fs::rename(&partial_path, final_path)?;
    fsync_directory(root)
}

fn write_manifest_atomic(
    root: &Path,
    manifest_path: &Path,
    manifest: &FixedCadenceCoordinateManifest,
) -> Result<()> {
    let bytes = canonical_json_bytes(manifest)?;
    let partial_path = root.join(".manifest.json.partial");
    if partial_path.exists() {
        bail!("fixed-cadence manifest recovery blocker exists");
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(&partial_path)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    std::fs::rename(&partial_path, manifest_path)?;
    fsync_directory(root)
}

fn canonical_json_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn canonical_sha256<T: Serialize>(value: &T) -> Result<String> {
    Ok(sha256_bytes(&serde_json::to_vec(value)?))
}

fn read_regular_nofollow(path: &Path, expected_bytes: Option<u64>) -> Result<Vec<u8>> {
    let metadata = std::fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        bail!("fixed-cadence artifact is not a regular non-symlink file");
    }
    if expected_bytes.is_some_and(|expected| metadata.len() != expected) {
        bail!("fixed-cadence artifact byte length mismatch before open");
    }
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)?;
    let opened = file.metadata()?;
    if !opened.file_type().is_file()
        || expected_bytes.is_some_and(|expected| opened.len() != expected)
    {
        bail!("opened fixed-cadence artifact identity/size mismatch");
    }
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    if bytes.len() as u64 != opened.len() {
        bail!("fixed-cadence artifact changed while reading");
    }
    Ok(bytes)
}

fn fsync_directory(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}

fn validate_stem(stem: &str) -> Result<()> {
    if stem.is_empty()
        || !stem
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    {
        bail!("invalid fixed-cadence topology stem");
    }
    Ok(())
}

fn require_sha256(value: &str, field: &str) -> Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("invalid {field}");
    }
    Ok(())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    encode_hex(&Sha256::digest(bytes))
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(
        bytes
            .get(offset..offset + 4)
            .context("truncated fixed-cadence u32")?
            .try_into()?,
    ))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    Ok(u64::from_le_bytes(
        bytes
            .get(offset..offset + 8)
            .context("truncated fixed-cadence u64")?
            .try_into()?,
    ))
}

fn put_sha256(bytes: &mut [u8], offset: usize, value: &str) -> Result<()> {
    require_sha256(value, "header SHA256")?;
    let decoded = decode_hex(value)?;
    bytes[offset..offset + 32].copy_from_slice(&decoded);
    Ok(())
}

fn put_optional_sha256(bytes: &mut [u8], offset: usize, value: Option<&str>) -> Result<()> {
    put_sha256(bytes, offset, value.unwrap_or(ZERO_SHA256))
}

fn read_sha256(bytes: &[u8], offset: usize) -> Result<String> {
    Ok(encode_hex(
        bytes
            .get(offset..offset + 32)
            .context("truncated fixed-cadence SHA256")?,
    ))
}

fn encode_hex(bytes: &[u8]) -> String {
    let mut value = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut value, "{byte:02x}").expect("writing to String cannot fail");
    }
    value
}

fn decode_hex(value: &str) -> Result<Vec<u8>> {
    if value.len() % 2 != 0 {
        bail!("odd-width hexadecimal input");
    }
    (0..value.len())
        .step_by(2)
        .map(|offset| {
            u8::from_str_radix(&value[offset..offset + 2], 16).context("invalid hexadecimal input")
        })
        .collect()
}

fn read_optional_sha256(bytes: &[u8], offset: usize) -> Result<Option<String>> {
    let value = read_sha256(bytes, offset)?;
    Ok((value != ZERO_SHA256).then_some(value))
}

fn byte_range(bytes: &[u8], offset: u64, length: u64) -> Result<&[u8]> {
    let start: usize = offset.try_into()?;
    let len: usize = length.try_into()?;
    let end = start
        .checked_add(len)
        .context("fixed-cadence byte range overflow")?;
    bytes
        .get(start..end)
        .context("fixed-cadence byte range escapes segment")
}

fn invert_3x3(m: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let d = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if !d.is_finite() || d.abs() <= 1.0e-12 {
        return None;
    }
    Some([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / d,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / d,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / d,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / d,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / d,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / d,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / d,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / d,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / d,
        ],
    ])
}

fn minimum_image(
    delta: [f64; 3],
    matrix: [[f64; 3]; 3],
    inverse: [[f64; 3]; 3],
) -> Result<[f64; 3]> {
    let fractional = mat_vec(inverse, delta);
    if fractional.iter().any(|value| !value.is_finite()) {
        bail!("triclinic minimum-image displacement is nonfinite");
    }
    let initial_translation = fractional.map(f64::round);
    let initial = mat_vec(matrix, sub(fractional, initial_translation));
    let initial_norm = dot(initial, initial).sqrt();
    let inverse_frobenius = inverse
        .iter()
        .flatten()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let bound = inverse_frobenius * initial_norm;
    if !bound.is_finite() {
        bail!("triclinic minimum-image search bound is nonfinite");
    }
    let ranges = fractional.map(|value| -> Result<(i64, i64)> {
        let padding = f64::EPSILON * (1.0 + value.abs() + bound);
        let lower = (value - bound - padding).ceil();
        let upper = (value + bound + padding).floor();
        if lower < i64::MIN as f64 || upper > i64::MAX as f64 || lower > upper {
            bail!("triclinic minimum-image lattice range is invalid");
        }
        Ok((lower as i64, upper as i64))
    });
    let [x_range, y_range, z_range] = ranges;
    let (x_range, y_range, z_range) = (x_range?, y_range?, z_range?);
    let mut best = initial;
    let mut best_norm2 = dot(initial, initial);
    let mut best_translation = initial_translation.map(|value| value as i64);
    for nx in x_range.0..=x_range.1 {
        for ny in y_range.0..=y_range.1 {
            for nz in z_range.0..=z_range.1 {
                let translation = [nx, ny, nz];
                let candidate = mat_vec(
                    matrix,
                    std::array::from_fn(|axis| fractional[axis] - translation[axis] as f64),
                );
                let norm2 = dot(candidate, candidate);
                if norm2 < best_norm2
                    || (norm2.to_bits() == best_norm2.to_bits() && translation < best_translation)
                {
                    best = candidate;
                    best_norm2 = norm2;
                    best_translation = translation;
                }
            }
        }
    }
    Ok(best)
}

fn mat_vec(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn recorded_body_fit_preserves_rigid_geometry_and_reports_deformation() {
        let reference = [[0., 0., 0.], [2., 0., 0.], [0., 2., 0.], [0., 0., 2.]];
        let current = reference.map(|p| [-p[1] + 11., p[0] - 7., p[2] + 3.]);
        let fit = measure_body_frame_fit(&reference, &current, [1., 1., 1.]).unwrap();
        assert_eq!(fit.centroid, [10., -6., 4.]);
        assert!(fit.rmsd < 1e-12);
        for (a, b) in current.iter().zip(reference) {
            assert!(fit
                .point_in_reference(*a)
                .iter()
                .zip(b)
                .all(|(x, y)| (x - y).abs() < 1e-12));
        }
        let mut deformed = current;
        deformed[3][2] += 2.;
        let fit = measure_body_frame_fit(&reference, &deformed, [1., 1., 1.]).unwrap();
        assert_eq!(fit.max_residual, 2.);
        assert_eq!(fit.rmsd, 1.);
        assert!(measure_body_frame_fit(&reference[..2], &current[..2], [0.; 3]).is_err());
        assert!(measure_body_frame_fit(&[[0.; 3]; 3], &[[0.; 3]; 3], [0.; 3]).is_err());
    }

    fn digest(seed: u8) -> String {
        format!("{seed:02x}").repeat(32)
    }

    fn fixture_topology() -> PrismPrepTopology {
        let fragments = [
            serde_json::json!({
            "schema_kind": "prismos_openmm_topology",
            "schema_version": "3.0.0",
            "source_pdb": "fixed-cadence-fixture.pdb",
            "n_atoms": 6,
            "n_residues": 3,
            "n_chains": 3,
            "positions": [0.0,0.0,0.0, 1.0,0.0,0.0, 0.0,1.0,0.0, 3.0,0.0,0.0, 4.0,0.0,0.0, 5.0,0.0,0.0],
            "masses": [12.0,12.0,14.0,16.0,1.0,23.0],
            "elements": ["C","C","N","O","H","Na"],
            "atom_names": ["CA","CB","N","O","H1","NA"],
            "residue_names": ["ALA","ALA","ALA","HOH","HOH","NA"],
            "residue_ids": [0,0,0,1,1,2],
            "chain_ids": ["A","A","A","W","W","I"],
            }),
            serde_json::json!({
            "ca_indices": [0],
            "charges": [0.0,0.0,0.0,-0.8,0.4,1.0],
            "lj_params": [
                {"sigma":3.0,"epsilon":0.1}, {"sigma":3.0,"epsilon":0.1},
                {"sigma":3.0,"epsilon":0.1}, {"sigma":3.0,"epsilon":0.1},
                {"sigma":3.0,"epsilon":0.1}, {"sigma":3.0,"epsilon":0.1}
            ],
            "gb_radii": [1.5,1.5,1.5,1.5,1.5,1.5],
            "bonds": [], "angles": [], "dihedrals": [],
            "water_oxygens": [3], "h_clusters": [], "exclusions": [],
            "residues": [], "residue_to_atom_indices": {}, "pdb_atom_serial_to_topo_index": {},
            "total_physical_residue_count": 3,
            "n_analysis_residues": 1,
            "atom_physical_residue_ids": [0,0,0,1,1,2],
            "atom_analysis_residue_ids": [0,0,0,-1,-1,-1],
            }),
            serde_json::json!({
            "analysis_residue_contract": {
                "schema_kind":"prismos_analysis_residue_mapping","schema_version":1,
                "analysis_authority":"KCC_GHOST_PROTEIN_RESIDUE_AXIS",
                "excluded_atom_sentinel":-1,"n_atoms":6,
                "total_physical_residue_count":3,"n_analysis_residues":1,
                "physical_residue_roles":["RECEPTOR","WATER","ION"],
                "physical_to_analysis_residue_ids":[0,-1,-1],
                "analysis_residues":[{
                    "analysis_residue_id":0,"physical_residue_id":0,"chain_id":"A",
                    "pdb_residue_id":1,"residue_name":"ALA","role":"RECEPTOR"
                }],
                "atom_analysis_residue_ids_sha256":digest(2),
                "physical_molecule_table_sha256":digest(10),
                "lambda_fragment_system_sha256":null,
                "role_counts":{"RECEPTOR":1,"WATER":1,"ION":1},
                "excluded_roles":["WATER","ION","LAMBDA_FRAGMENT","OTHER"],
                "must_recompute_after_fragment_materialization":true,
                "contract_sha256":digest(5)
            },
            "periodic_cell": {
                "periodic":true,"vectors_angstrom":[[20.0,0.0,0.0],[0.0,20.0,0.0],[0.0,0.0,20.0]],
                "volume_angstrom3":8000.0,"orthorhombic":true,"minimum_image_required":true,
                "coordinate_wrapping_required":true,"reciprocal_method":"PME",
                "real_space_cutoff_angstrom":8.0,"contract_sha256":digest(6)
            },
            }),
            serde_json::json!({
            "molecule_identity_table": {
                "schema_kind":"prismos_molecule_identity_table","schema_version":1,
                "n_molecules":3,"n_atoms":6,
                "molecules":[
                    {"molecule_id":0,"atom_indices":[0,1,2],"residue_indices":[0],"residue_names":["ALA"],"chain_ids":["A"],"kind":"BIOMOLECULE","identity":"MYC","component_sha256":digest(7)},
                    {"molecule_id":1,"atom_indices":[3,4],"residue_indices":[1],"residue_names":["HOH"],"chain_ids":["W"],"kind":"WATER","identity":"TIP3P-FB","component_sha256":digest(8)},
                    {"molecule_id":2,"atom_indices":[5],"residue_indices":[2],"residue_names":["NA"],"chain_ids":["I"],"kind":"ION","identity":"NA","component_sha256":digest(9)}
                ],"table_sha256":digest(10)
            }
            }),
        ];
        let mut combined = serde_json::Map::new();
        for fragment in fragments {
            combined.extend(fragment.as_object().unwrap().clone());
        }
        let atom_analysis_ids = serde_json::json!([0, 0, 0, -1, -1, -1]);
        let atom_analysis_sha = crate::input::python_canonical_sha256(&atom_analysis_ids).unwrap();
        let analysis_contract = combined
            .get_mut("analysis_residue_contract")
            .unwrap()
            .as_object_mut()
            .unwrap();
        analysis_contract.insert(
            "atom_analysis_residue_ids_sha256".into(),
            atom_analysis_sha.into(),
        );
        analysis_contract.remove("contract_sha256");
        let contract_sha = crate::input::python_canonical_sha256(&serde_json::Value::Object(
            analysis_contract.clone(),
        ))
        .unwrap();
        analysis_contract.insert("contract_sha256".into(), contract_sha.into());
        serde_json::from_value(serde_json::Value::Object(combined)).unwrap()
    }

    fn context(
        mode: FixedCadenceCoordinateMode,
        topology: &PrismPrepTopology,
    ) -> FixedCadenceCoordinateContext {
        context_for_stream(mode, topology, 0)
    }

    fn context_for_stream(
        mode: FixedCadenceCoordinateMode,
        topology: &PrismPrepTopology,
        stream_id: u16,
    ) -> FixedCadenceCoordinateContext {
        let analysis = topology.analysis_residue_contract.as_ref().unwrap();
        let molecules = topology.molecule_identity_table.as_ref().unwrap();
        let body_fixed_tracker_authority =
            (mode == FixedCadenceCoordinateMode::CandidateConfirmation).then(|| {
                BodyFixedTrackerAuthority {
                    schema_kind: "prism_body_fixed_tracker_authority".into(),
                    schema_version: "1.0.0".into(),
                    authority_sha256: String::new(),
                    reference_structure_sha256: digest(21),
                    reference_coordinate_frame_sha256: digest(22),
                    chain_mapping_receipt_sha256: analysis.contract_sha256.clone(),
                    topology_sha256: digest(11),
                    spatial_boundary_contract_sha256: digest(6),
                    boundary_mode: BodyFixedBoundaryMode::PeriodicTriclinicMinimumImage,
                    unwrap_semantics: "anchor_zero_triclinic_minimum_image_v1".into(),
                    frame_construction_semantics:
                        "ordered_three_anchor_right_handed_rigid_frame_v1".into(),
                    anchor_atoms: [
                        (0, "CA", [0.0, 0.0, 0.0]),
                        (1, "CB", [1.0, 0.0, 0.0]),
                        (2, "N", [0.0, 1.0, 0.0]),
                    ]
                    .into_iter()
                    .map(|(atom_index, atom_name, reference_position_angstrom)| {
                        BodyFixedAnchorAtom {
                            atom_index,
                            analysis_residue_id: 0,
                            chain_id: "A".into(),
                            pdb_residue_id: 1,
                            residue_name: "ALA".into(),
                            atom_name: atom_name.into(),
                            reference_position_angstrom,
                        }
                    })
                    .collect(),
                    reference_candidate_centroid_angstrom: [0.0, 0.0, 0.0],
                    maximum_anchor_fit_rmsd_angstrom: 0.01,
                    maximum_anchor_fit_residual_angstrom: 0.01,
                }
                .materialize()
                .unwrap()
            });
        FixedCadenceCoordinateContext {
            schema_kind: "prism_fixed_cadence_coordinate_context".into(),
            schema_version: 3,
            run_id: "run".into(),
            replica_id: "rep_00".into(),
            replica_seed: 42,
            stream_seed: 7 + u64::from(stream_id),
            stream_id,
            topology_sha256: digest(11),
            selected_engine_sha256: digest(12),
            native_pathb_runtime_request_sha256: digest(16),
            integration_timebase_context_sha256: digest(17),
            integration_timebase_authority_sha256: digest(18),
            integration_timestep_femtoseconds: 2.0,
            candidate_lag_steps: vec![10, 20, 30],
            analysis_residue_contract_sha256: analysis.contract_sha256.clone(),
            atom_analysis_residue_ids_sha256: analysis.atom_analysis_residue_ids_sha256.clone(),
            molecule_identity_table_sha256: molecules.table_sha256.clone(),
            periodic_cell_contract_sha256: digest(6),
            thermodynamic_condition_id: "condition".into(),
            protocol_id: "thermal".into(),
            protocol_state_hash: digest(13),
            phase_schedule_digest_sha256: digest(14),
            mode,
            cadence_steps: 10,
            start_step: 10,
            end_step_exclusive: 40,
            candidate_tracker_sha256: (mode == FixedCadenceCoordinateMode::CandidateConfirmation)
                .then(|| digest(15)),
            local_shell_cutoff_angstrom: (mode
                == FixedCadenceCoordinateMode::CandidateConfirmation)
                .then_some(5.0),
            body_fixed_tracker_authority,
        }
    }

    #[test]
    fn context_fail_closes_on_request_timebase_or_lag_cadence_drift() {
        let topology = fixture_topology();
        let baseline = context(FixedCadenceCoordinateMode::WholeProteinDiscovery, &topology);
        baseline.verify().unwrap();

        let mut request = baseline.clone();
        request.native_pathb_runtime_request_sha256 = "not-a-digest".into();
        assert!(request.verify().is_err());

        let mut timebase = baseline.clone();
        timebase.integration_timestep_femtoseconds = f64::NAN;
        assert!(timebase.verify().is_err());

        let mut authority = baseline.clone();
        authority.integration_timebase_authority_sha256 = digest(19);
        authority.integration_timebase_context_sha256.clear();
        assert!(authority.verify().is_err());

        let mut off_cadence = baseline;
        off_cadence.candidate_lag_steps = vec![15, 20];
        assert!(off_cadence.verify().is_err());
    }

    #[test]
    fn discovery_and_confirmation_close_exact_counts_and_shells() {
        for mode in [
            FixedCadenceCoordinateMode::WholeProteinDiscovery,
            FixedCadenceCoordinateMode::CandidateConfirmation,
        ] {
            let tmp = TempDir::new().unwrap();
            let topology = fixture_topology();
            let ctx = context(mode, &topology);
            let mut writer =
                FixedCadenceCoordinateWriter::create(tmp.path(), "mycmax", &topology, ctx.clone())
                    .unwrap();
            for step in [10, 20, 30] {
                writer
                    .append(FixedCadenceCoordinateInput {
                        step_idx: step,
                        protocol_phase: 1,
                        positions_angstrom: topology.positions.clone(),
                        tracked_centroid_angstrom: None,
                    })
                    .unwrap();
            }
            writer.complete().unwrap();
            let root = fixed_cadence_coordinate_root(tmp.path(), "mycmax", 0);
            let loaded = load_verified_fixed_cadence_coordinates(&root, &topology, &ctx).unwrap();
            assert_eq!(loaded.frames.len(), 3);
            assert!(
                loaded
                    .manifest
                    .scheduled_equals_persisted_equals_decoded_equals_indexed
            );
            if mode == FixedCadenceCoordinateMode::CandidateConfirmation {
                assert!(loaded
                    .frames
                    .values()
                    .all(|frame| frame.full_positions_angstrom.is_some()));
                assert!(loaded
                    .frames
                    .values()
                    .all(|frame| frame.local_shell.len() == 3));
            } else {
                assert!(loaded
                    .frames
                    .values()
                    .all(|frame| frame.full_positions_angstrom.is_none()));
                assert!(loaded
                    .frames
                    .values()
                    .all(|frame| frame.local_shell.is_empty()));
            }
        }
    }

    #[test]
    fn gap_and_tamper_fail_closed() {
        let topology = fixture_topology();
        let ctx = context(FixedCadenceCoordinateMode::CandidateConfirmation, &topology);
        let gap = TempDir::new().unwrap();
        let mut writer =
            FixedCadenceCoordinateWriter::create(gap.path(), "mycmax", &topology, ctx.clone())
                .unwrap();
        assert!(writer
            .append(FixedCadenceCoordinateInput {
                step_idx: 20,
                protocol_phase: 1,
                positions_angstrom: topology.positions.clone(),
                tracked_centroid_angstrom: None,
            })
            .is_err());

        let tamper = TempDir::new().unwrap();
        let mut writer =
            FixedCadenceCoordinateWriter::create(tamper.path(), "mycmax", &topology, ctx.clone())
                .unwrap();
        for step in [10, 20, 30] {
            writer
                .append(FixedCadenceCoordinateInput {
                    step_idx: step,
                    protocol_phase: 1,
                    positions_angstrom: topology.positions.clone(),
                    tracked_centroid_angstrom: None,
                })
                .unwrap();
        }
        writer.complete().unwrap();
        let root = fixed_cadence_coordinate_root(tamper.path(), "mycmax", 0);
        let path = root.join("frame_00000001.ghostcadence");
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[HEADER_BYTES + 3] ^= 1;
        std::fs::write(&path, bytes).unwrap();
        assert!(load_verified_fixed_cadence_coordinates(&root, &topology, &ctx).is_err());
    }

    #[test]
    fn candidate_confirmation_population_is_exact_and_fail_closed() {
        let topology = fixture_topology();
        let complete_stream = |root: &Path, stem: &str, stream_id: u16| {
            let ctx = context_for_stream(
                FixedCadenceCoordinateMode::CandidateConfirmation,
                &topology,
                stream_id,
            );
            let mut writer =
                FixedCadenceCoordinateWriter::create(root, stem, &topology, ctx).unwrap();
            for step in [10, 20, 30] {
                writer
                    .append(FixedCadenceCoordinateInput {
                        step_idx: step,
                        protocol_phase: 1,
                        positions_angstrom: topology.positions.clone(),
                        tracked_centroid_angstrom: None,
                    })
                    .unwrap();
            }
            writer.complete().unwrap();
        };

        let population = TempDir::new().unwrap();
        complete_stream(population.path(), "mycmax", 0);
        complete_stream(population.path(), "mycmax", 1);
        let expected = BTreeSet::from([0u16, 1u16]);
        let loaded = load_verified_candidate_confirmation_population(
            population.path(),
            &topology,
            &expected,
        )
        .unwrap();
        assert_eq!(
            loaded.streams.keys().copied().collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(loaded.total_scheduled_frame_count, 6);
        assert_eq!(
            loaded.total_scheduled_frame_count,
            loaded.total_persisted_frame_count
        );
        assert_eq!(
            loaded.total_scheduled_frame_count,
            loaded.total_decoded_frame_count
        );
        assert_eq!(
            loaded.total_scheduled_frame_count,
            loaded.total_indexed_frame_count
        );
        require_sha256(&loaded.population_sha256, "population_sha256").unwrap();
        assert!(load_verified_candidate_confirmation_population(
            population.path(),
            &topology,
            &BTreeSet::from([0u16]),
        )
        .is_err());

        let duplicate = TempDir::new().unwrap();
        complete_stream(duplicate.path(), "mycmax_a", 0);
        complete_stream(duplicate.path(), "mycmax_b", 0);
        assert!(load_verified_candidate_confirmation_population(
            duplicate.path(),
            &topology,
            &BTreeSet::from([0u16]),
        )
        .is_err());

        let discovery = TempDir::new().unwrap();
        let discovery_context =
            context(FixedCadenceCoordinateMode::WholeProteinDiscovery, &topology);
        let mut writer = FixedCadenceCoordinateWriter::create(
            discovery.path(),
            "mycmax",
            &topology,
            discovery_context,
        )
        .unwrap();
        for step in [10, 20, 30] {
            writer
                .append(FixedCadenceCoordinateInput {
                    step_idx: step,
                    protocol_phase: 1,
                    positions_angstrom: topology.positions.clone(),
                    tracked_centroid_angstrom: None,
                })
                .unwrap();
        }
        writer.complete().unwrap();
        assert!(load_verified_candidate_confirmation_population(
            discovery.path(),
            &topology,
            &BTreeSet::from([0u16]),
        )
        .is_err());

        let complete_discovery_stream =
            |root: &Path, stem: &str, stream_id: u16, request_sha256: String, replica_seed: u64| {
                let mut ctx = context_for_stream(
                    FixedCadenceCoordinateMode::WholeProteinDiscovery,
                    &topology,
                    stream_id,
                );
                ctx.native_pathb_runtime_request_sha256 = request_sha256;
                ctx.replica_seed = replica_seed;
                let mut writer =
                    FixedCadenceCoordinateWriter::create(root, stem, &topology, ctx).unwrap();
                for step in [10, 20, 30] {
                    writer
                        .append(FixedCadenceCoordinateInput {
                            step_idx: step,
                            protocol_phase: 1,
                            positions_angstrom: topology.positions.clone(),
                            tracked_centroid_angstrom: None,
                        })
                        .unwrap();
                }
                writer.complete().unwrap();
            };

        let discovery_population = TempDir::new().unwrap();
        complete_discovery_stream(discovery_population.path(), "mycmax", 0, digest(16), 42);
        complete_discovery_stream(discovery_population.path(), "mycmax", 1, digest(16), 42);
        let discovery_loaded = load_verified_whole_protein_discovery_population(
            discovery_population.path(),
            &topology,
            &expected,
        )
        .unwrap();
        assert_eq!(discovery_loaded.total_scheduled_frame_count, 6);
        assert!(load_verified_whole_protein_discovery_population(
            discovery_population.path(),
            &topology,
            &BTreeSet::from([0u16]),
        )
        .is_err());
        assert!(load_verified_candidate_confirmation_population(
            discovery_population.path(),
            &topology,
            &expected,
        )
        .is_err());

        let mixed_mode = TempDir::new().unwrap();
        complete_discovery_stream(mixed_mode.path(), "discovery", 0, digest(16), 42);
        complete_stream(mixed_mode.path(), "confirmation", 1);
        assert!(load_verified_whole_protein_discovery_population(
            mixed_mode.path(),
            &topology,
            &expected,
        )
        .is_err());

        let mixed_identity = TempDir::new().unwrap();
        complete_discovery_stream(mixed_identity.path(), "mycmax", 0, digest(16), 42);
        complete_discovery_stream(mixed_identity.path(), "mycmax", 1, digest(19), 42);
        assert!(load_verified_whole_protein_discovery_population(
            mixed_identity.path(),
            &topology,
            &expected,
        )
        .is_err());

        let mixed_replica = TempDir::new().unwrap();
        complete_discovery_stream(mixed_replica.path(), "mycmax", 0, digest(16), 42);
        complete_discovery_stream(mixed_replica.path(), "mycmax", 1, digest(16), 43);
        assert!(load_verified_whole_protein_discovery_population(
            mixed_replica.path(),
            &topology,
            &expected,
        )
        .is_err());
    }

    #[test]
    fn body_fixed_transform_tracks_wrapped_rotation_and_rejects_static_centroid() {
        let topology = fixture_topology();
        let ctx = context(FixedCadenceCoordinateMode::CandidateConfirmation, &topology);
        let static_input = TempDir::new().unwrap();
        let mut writer = FixedCadenceCoordinateWriter::create(
            static_input.path(),
            "mycmax",
            &topology,
            ctx.clone(),
        )
        .unwrap();
        assert!(writer
            .append(FixedCadenceCoordinateInput {
                step_idx: 10,
                protocol_phase: 1,
                positions_angstrom: topology.positions.clone(),
                tracked_centroid_angstrom: Some([0.0, 0.0, 0.0]),
            })
            .is_err());

        let transformed = TempDir::new().unwrap();
        let mut writer =
            FixedCadenceCoordinateWriter::create(transformed.path(), "mycmax", &topology, ctx)
                .unwrap();
        let mut positions = topology.positions.clone();
        // 90-degree rotation about z plus translation. Atom 1 is wrapped
        // across y=20, forcing full periodic unwrapping around anchor zero.
        positions[0..9].copy_from_slice(&[19.5, 19.5, 0.0, 19.5, 0.5, 0.0, 18.5, 19.5, 0.0]);
        writer
            .append(FixedCadenceCoordinateInput {
                step_idx: 10,
                protocol_phase: 1,
                positions_angstrom: positions,
                tracked_centroid_angstrom: None,
            })
            .unwrap();
        let entry = &writer.manifest().frames[0];
        let transform = entry.body_fixed_transform.as_ref().unwrap();
        assert_eq!(entry.tracked_centroid_angstrom, Some([19.5, 19.5, 0.0]));
        assert!(transform.anchor_fit_rmsd_angstrom < 1.0e-12);
        assert!((transform.reference_to_frame_rotation[0][1] + 1.0).abs() < 1.0e-12);
        assert!((transform.reference_to_frame_rotation[1][0] - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn body_fixed_transform_rejects_degenerate_and_excess_residual_frames() {
        let topology = fixture_topology();
        let ctx = context(FixedCadenceCoordinateMode::CandidateConfirmation, &topology);
        for anchors in [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ] {
            let tmp = TempDir::new().unwrap();
            let mut writer =
                FixedCadenceCoordinateWriter::create(tmp.path(), "mycmax", &topology, ctx.clone())
                    .unwrap();
            let mut positions = topology.positions.clone();
            positions[0..9].copy_from_slice(&anchors);
            assert!(writer
                .append(FixedCadenceCoordinateInput {
                    step_idx: 10,
                    protocol_phase: 1,
                    positions_angstrom: positions,
                    tracked_centroid_angstrom: None,
                })
                .is_err());
        }
    }

    #[test]
    fn body_fixed_authority_rejects_degenerate_reference_anchors() {
        let topology = fixture_topology();
        let mut ctx = context(FixedCadenceCoordinateMode::CandidateConfirmation, &topology);
        let authority = ctx.body_fixed_tracker_authority.as_mut().unwrap();
        authority.anchor_atoms[2].reference_position_angstrom = [2.0, 0.0, 0.0];
        authority.authority_sha256.clear();
        assert!(authority.clone().materialize().is_err());
        assert!(ctx.verify().is_err());
    }

    #[test]
    fn triclinic_minimum_image_solves_nearest_lattice_not_axis_rounding() {
        let matrix = [[10.0, 9.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 10.0]];
        let inverse = invert_3x3(matrix).unwrap();
        let cartesian = mat_vec(matrix, [0.49, 0.49, 0.0]);
        let image = minimum_image(cartesian, matrix, inverse).unwrap();
        assert!((image[0] - 0.31).abs() < 1.0e-12);
        assert!((image[1] + 0.51).abs() < 1.0e-12);
        // Independent axis rounding would retain [9.31, 0.49, 0].
        assert!(dot(image, image).sqrt() < 1.0);
    }

    #[test]
    fn checkpoint_reopen_preserves_later_frame_and_replays_exact_cadence() {
        let tmp = TempDir::new().unwrap();
        let topology = fixture_topology();
        let ctx = context(FixedCadenceCoordinateMode::WholeProteinDiscovery, &topology);
        let mut writer =
            FixedCadenceCoordinateWriter::create(tmp.path(), "target", &topology, ctx.clone())
                .unwrap();
        let frame = |step| FixedCadenceCoordinateInput {
            step_idx: step,
            protocol_phase: 1,
            positions_angstrom: topology.positions.clone(),
            tracked_centroid_angstrom: None,
        };
        writer.append(frame(10)).unwrap();
        let boundary = writer.checkpoint(tmp.path()).unwrap();
        writer.append(frame(20)).unwrap();
        drop(writer);
        let mut restored =
            FixedCadenceCoordinateWriter::reopen_checkpoint(tmp.path(), &boundary, &topology, &ctx)
                .unwrap();
        assert_eq!(restored.manifest.persisted_frame_count, 1);
        assert!(restored.root.join(".continuation_tails").is_dir());
        restored.append(frame(20)).unwrap();
        restored.append(frame(30)).unwrap();
        restored.complete().unwrap();
        assert_eq!(
            load_verified_fixed_cadence_coordinates(&restored.root, &topology, &ctx)
                .unwrap()
                .frames
                .len(),
            3
        );
    }
}
