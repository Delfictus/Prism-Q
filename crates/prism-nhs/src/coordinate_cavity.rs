//! Exact V2 trajectory joins and coordinate-derived local cavity observables.
//!
//! This module never interpolates frames and never treats a Ghost AABB as a
//! cavity. It parses the lossless resident/V2 stream bytes, verifies the audit
//! counts and rolling hash, joins by exact `(stream_id, step_idx)`, uses the
//! prepared molecule/atom/residue map, and evaluates a declared multi-grid
//! solvent-probe geometry. Raw coordinates are transient and are not copied
//! into the emitted sidecar.

use crate::chemical_state_fiber::{canonical_sha256, require_sha256, PhysicalMeasurement};
use crate::ghost_coordinate_segment::{GhostCoordinateContext, VerifiedGhostCoordinate};
use crate::input::PrismPrepTopology;
use crate::pathb_evidence_emitter::ExactCoordinateFrameBinding;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs::OpenOptions;
use std::io::Read;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

pub mod recorded_roi;
pub mod recorded_interface;
pub mod saved_ghost;

pub const COORDINATE_CAVITY_SCHEMA: &str = "prism_coordinate_cavity_sidecar";
pub const COORDINATE_CAVITY_VERSION: &str = "2.0.0";
pub const PHYSICAL_CAVITY_LINING_SCHEMA: &str = "prism_physical_cavity_lining_evidence";
pub const PHYSICAL_CAVITY_LINING_VERSION: &str = "1.0.0";

#[derive(Debug, Clone)]
struct IndexedFrame {
    ordinal: u64,
    byte_offset: u64,
    byte_length: u64,
    frame_sha256: String,
    positions: Vec<f32>,
}

/// Fully verified exact-step index of one lossless V2 trajectory stream.
#[derive(Debug, Clone)]
pub struct LosslessV2TrajectoryIndex {
    stream_id: u16,
    trajectory_sha256: String,
    audit_sha256: String,
    atom_count: usize,
    frames: BTreeMap<u64, IndexedFrame>,
}

/// Declared local cavity estimator parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoordinateCavityConfig {
    pub grid_spacings_a: Vec<f64>,
    pub grid_phase_offsets: Vec<f64>,
    pub local_half_width_a: f64,
    pub solvent_probe_radius_a: f64,
    pub ray_horizon_a: f64,
    pub minimum_blocked_axial_directions: u8,
    pub require_opposed_wall_pair: bool,
}

impl Default for CoordinateCavityConfig {
    fn default() -> Self {
        Self {
            grid_spacings_a: vec![0.75, 0.50],
            grid_phase_offsets: vec![0.0, 0.5],
            local_half_width_a: 6.0,
            solvent_probe_radius_a: 1.4,
            ray_horizon_a: 8.0,
            minimum_blocked_axial_directions: 3,
            require_opposed_wall_pair: true,
        }
    }
}

impl CoordinateCavityConfig {
    /// Validate the complete estimator declaration before MD starts.
    pub fn validate(&self) -> Result<()> {
        if self.grid_spacings_a.len() < 2
            || self
                .grid_spacings_a
                .iter()
                .any(|v| !v.is_finite() || *v <= 0.0)
            || self.grid_phase_offsets.len() < 2
            || self
                .grid_phase_offsets
                .iter()
                .any(|v| !v.is_finite() || !(0.0..1.0).contains(v))
            || !self.local_half_width_a.is_finite()
            || self.local_half_width_a <= 0.0
            || !self.solvent_probe_radius_a.is_finite()
            || self.solvent_probe_radius_a <= 0.0
            || !self.ray_horizon_a.is_finite()
            || self.ray_horizon_a <= 0.0
            || !(1..=6).contains(&self.minimum_blocked_axial_directions)
        {
            bail!("invalid multi-resolution coordinate cavity estimator configuration");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CavityGridEstimate {
    pub spacing_a: f64,
    pub phase_offset_fraction: f64,
    pub cavity_voxels: u64,
    pub volume_a3: f64,
}

/// Spatial metric actually used for atom-to-face attribution.  Periodic
/// distance is evaluated with the complete triclinic cell, never fractional
/// coordinate Euclidean distance.  The nonperiodic branch is legal only when
/// the prepared topology carries a separately hash-bound declaration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhysicalCavitySpatialMetricMode {
    PeriodicTriclinicMinimumImage,
    DeclaredNonperiodicEuclidean,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhysicalCavitySpatialMetricAuthority {
    pub schema_kind: String,
    pub schema_version: String,
    pub authority_sha256: String,
    pub periodic_cell_contract_sha256: String,
    pub mode: PhysicalCavitySpatialMetricMode,
    pub box_vectors_angstrom: Option<[[f64; 3]; 3]>,
    pub nonperiodic_reason: Option<String>,
}

impl PhysicalCavitySpatialMetricAuthority {
    fn body_sha256(&self) -> Result<String> {
        let mut body = self.clone();
        body.authority_sha256.clear();
        canonical_sha256(&body)
    }

    fn materialize(mut self) -> Result<Self> {
        if !self.authority_sha256.is_empty() {
            bail!("spatial metric materialization requires an empty digest");
        }
        self.authority_sha256 = self.body_sha256()?;
        self.verify()?;
        Ok(self)
    }

    pub fn verify(&self) -> Result<()> {
        if self.schema_kind != "prism_physical_cavity_spatial_metric_authority"
            || self.schema_version != "1.0.0"
        {
            bail!("invalid physical-cavity spatial metric authority schema");
        }
        require_sha256(&self.authority_sha256, "physical-cavity metric authority")?;
        require_sha256(
            &self.periodic_cell_contract_sha256,
            "physical-cavity periodic-cell contract",
        )?;
        match self.mode {
            PhysicalCavitySpatialMetricMode::PeriodicTriclinicMinimumImage => {
                let vectors = self
                    .box_vectors_angstrom
                    .context("periodic physical-cavity metric lacks triclinic vectors")?;
                if self.nonperiodic_reason.is_some()
                    || vectors.iter().flatten().any(|value| !value.is_finite())
                    || invert_3x3(row_vectors_to_matrix(vectors)).is_none()
                {
                    bail!("invalid periodic triclinic physical-cavity metric");
                }
            }
            PhysicalCavitySpatialMetricMode::DeclaredNonperiodicEuclidean => {
                if self.box_vectors_angstrom.is_some()
                    || self
                        .nonperiodic_reason
                        .as_deref()
                        .is_none_or(|reason| reason.trim().is_empty())
                {
                    bail!("nonperiodic physical-cavity metric lacks explicit authority");
                }
            }
        }
        if self.body_sha256()? != self.authority_sha256 {
            bail!("physical-cavity spatial metric authority hash mismatch");
        }
        Ok(())
    }
}

/// Exact identity of a receptor atom admitted by the prepared topology.  The
/// dense analysis residue is preserved alongside physical/PDB identity so no
/// RichSpike or contact-residue proxy is needed downstream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhysicalCavityLiningAtomIdentity {
    pub atom_identity_sha256: String,
    pub atom_index: usize,
    pub atom_name: String,
    pub element: String,
    pub physical_residue_id: usize,
    pub analysis_residue_id: u32,
    pub pdb_residue_id: i64,
    pub residue_name: String,
    pub chain_id: String,
    pub molecule_id: u64,
    pub molecule_identity: String,
    pub molecule_component_sha256: String,
}

impl PhysicalCavityLiningAtomIdentity {
    fn body_sha256(&self) -> Result<String> {
        let mut body = self.clone();
        body.atom_identity_sha256.clear();
        canonical_sha256(&body)
    }

    fn materialize(mut self) -> Result<Self> {
        if !self.atom_identity_sha256.is_empty() {
            bail!("physical lining atom materialization requires an empty digest");
        }
        self.atom_identity_sha256 = self.body_sha256()?;
        self.verify()?;
        Ok(self)
    }

    pub fn verify(&self) -> Result<()> {
        require_sha256(&self.atom_identity_sha256, "physical lining atom identity")?;
        require_sha256(
            &self.molecule_component_sha256,
            "physical lining molecule component",
        )?;
        if self.atom_name.trim().is_empty()
            || self.element.trim().is_empty()
            || self.residue_name.trim().is_empty()
            || self.chain_id.trim().is_empty()
            || self.molecule_identity.trim().is_empty()
            || self.body_sha256()? != self.atom_identity_sha256
        {
            bail!("invalid physical cavity lining atom identity");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhysicalCavityBoundaryKind {
    ReceptorExcludedVolume,
    EstimatorMask,
    LocalWindow,
}

/// One oriented face on one selected cavity voxel.  Every selected-component
/// boundary face is retained; no top-k or residue truncation is permitted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhysicalCavityBoundaryFaceAttribution {
    pub face_record_sha256: String,
    pub grid_source_sha256: String,
    pub voxel_linear_index: u64,
    pub voxel_xyz: [u64; 3],
    pub axis: u8,
    pub orientation: i8,
    pub boundary_kind: PhysicalCavityBoundaryKind,
    pub face_center_relative_angstrom: [f64; 3],
    pub face_area_a2: f64,
    pub nearest_atom_identity_sha256: String,
    pub nearest_atom_index: usize,
    pub nearest_atom_center_distance_a: f64,
    pub nearest_solvent_excluded_surface_distance_a: f64,
}

impl PhysicalCavityBoundaryFaceAttribution {
    fn body_sha256(&self) -> Result<String> {
        let mut body = self.clone();
        body.face_record_sha256.clear();
        canonical_sha256(&body)
    }

    fn materialize(mut self) -> Result<Self> {
        if !self.face_record_sha256.is_empty() {
            bail!("physical lining face materialization requires an empty digest");
        }
        self.face_record_sha256 = self.body_sha256()?;
        self.verify()?;
        Ok(self)
    }

    pub fn verify(&self) -> Result<()> {
        require_sha256(&self.face_record_sha256, "physical lining face record")?;
        require_sha256(&self.grid_source_sha256, "physical lining grid source")?;
        require_sha256(
            &self.nearest_atom_identity_sha256,
            "physical lining nearest atom",
        )?;
        if self.axis > 2
            || !matches!(self.orientation, -1 | 1)
            || self
                .face_center_relative_angstrom
                .iter()
                .any(|value| !value.is_finite())
            || !self.face_area_a2.is_finite()
            || self.face_area_a2 <= 0.0
            || !self.nearest_atom_center_distance_a.is_finite()
            || self.nearest_atom_center_distance_a < 0.0
            || !self.nearest_solvent_excluded_surface_distance_a.is_finite()
            || self.body_sha256()? != self.face_record_sha256
        {
            bail!("invalid physical cavity boundary-face attribution");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhysicalCavityGridAtomAttribution {
    pub atom_identity_sha256: String,
    pub atom_index: usize,
    pub boundary_face_count: u64,
    pub receptor_excluded_volume_face_count: u64,
    pub estimator_mask_face_count: u64,
    pub local_window_face_count: u64,
    pub boundary_area_a2: f64,
    pub minimum_surface_distance_a: f64,
    pub maximum_surface_distance_a: f64,
}

/// Exact byte lineage and complete atom attribution for one grid/phase pair.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhysicalCavityGridLiningEvidence {
    pub grid_record_sha256: String,
    pub grid_source_sha256: String,
    pub grid_ordinal: u64,
    pub spacing_a: f64,
    pub phase_offset_fraction: f64,
    pub dimension: u64,
    pub origin_relative_angstrom: f64,
    pub voxel_count: u64,
    pub packed_bit_semantics: String,
    pub occupied_grid_bytes: u64,
    pub occupied_grid_sha256: String,
    pub occupied_grid_packed_lsb0: Vec<u8>,
    pub candidate_grid_bytes: u64,
    pub candidate_grid_sha256: String,
    pub candidate_grid_packed_lsb0: Vec<u8>,
    pub selected_component_grid_bytes: u64,
    pub selected_component_grid_sha256: String,
    pub selected_component_grid_packed_lsb0: Vec<u8>,
    pub selected_cavity_voxel_count: u64,
    pub selected_cavity_volume_a3: f64,
    pub boundary_face_count: u64,
    pub receptor_excluded_volume_face_count: u64,
    pub estimator_mask_face_count: u64,
    pub local_window_face_count: u64,
    pub boundary_area_a2: f64,
    pub boundary_faces_sha256: String,
    pub boundary_faces: Vec<PhysicalCavityBoundaryFaceAttribution>,
    pub per_atom_attributions: Vec<PhysicalCavityGridAtomAttribution>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhysicalCavityMultiGridConservation {
    pub expected_grid_count: u64,
    pub persisted_grid_count: u64,
    pub selected_cavity_voxel_count_from_grids: u64,
    pub selected_cavity_voxel_count_conserved: u64,
    pub boundary_face_count_from_grids: u64,
    pub boundary_face_count_from_records: u64,
    pub boundary_face_count_from_atom_attributions: u64,
    pub unique_lining_atom_count: u64,
    pub unique_lining_analysis_residue_count: u64,
    pub grids_with_observed_cavity: u64,
    pub grids_with_observed_no_cavity: u64,
    pub uncertainty_input_sha256: String,
    pub conservation_pass: bool,
}

/// Lossless typed physical-lining evidence produced at the grid evaluator.
/// An empty selected component is preserved as observed absence; missing
/// topology, coordinates, or metric authority returns an error instead.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhysicalCavityLiningEvidence {
    pub schema_kind: String,
    pub schema_version: String,
    pub evidence_sha256: String,
    pub ghost_record_sha256: String,
    pub coordinate_frame_sha256: String,
    pub topology_sha256: String,
    pub atom_residue_mapping_sha256: String,
    pub analysis_residue_contract_sha256: String,
    pub molecule_identity_table_sha256: String,
    pub estimator_sha256: String,
    pub spatial_metric: PhysicalCavitySpatialMetricAuthority,
    pub atom_identities: Vec<PhysicalCavityLiningAtomIdentity>,
    pub grids: Vec<PhysicalCavityGridLiningEvidence>,
    pub conservation: PhysicalCavityMultiGridConservation,
    pub complete_without_truncation: bool,
}

impl PhysicalCavityLiningEvidence {
    fn body_sha256(&self) -> Result<String> {
        let mut body = self.clone();
        body.evidence_sha256.clear();
        canonical_sha256(&body)
    }

    fn materialize(mut self) -> Result<Self> {
        if !self.evidence_sha256.is_empty() {
            bail!("physical lining evidence materialization requires an empty digest");
        }
        self.evidence_sha256 = self.body_sha256()?;
        self.verify()?;
        Ok(self)
    }

    pub fn verify(&self) -> Result<()> {
        if self.schema_kind != PHYSICAL_CAVITY_LINING_SCHEMA
            || self.schema_version != PHYSICAL_CAVITY_LINING_VERSION
            || !self.complete_without_truncation
        {
            bail!("invalid or truncated physical cavity lining evidence");
        }
        for (value, field) in [
            (&self.evidence_sha256, "physical lining evidence"),
            (&self.ghost_record_sha256, "physical lining Ghost record"),
            (
                &self.coordinate_frame_sha256,
                "physical lining coordinate frame",
            ),
            (&self.topology_sha256, "physical lining topology"),
            (
                &self.atom_residue_mapping_sha256,
                "physical lining atom-residue mapping",
            ),
            (
                &self.analysis_residue_contract_sha256,
                "physical lining analysis residue contract",
            ),
            (
                &self.molecule_identity_table_sha256,
                "physical lining molecule table",
            ),
            (&self.estimator_sha256, "physical lining estimator"),
            (
                &self.conservation.uncertainty_input_sha256,
                "physical lining uncertainty input",
            ),
        ] {
            require_sha256(value, field)?;
        }
        self.spatial_metric.verify()?;
        if self.conservation.expected_grid_count != self.grids.len() as u64
            || self.conservation.persisted_grid_count != self.grids.len() as u64
        {
            bail!("physical lining grid population is incomplete");
        }
        let mut atom_hashes = BTreeMap::new();
        let mut atom_indices = BTreeSet::new();
        for atom in &self.atom_identities {
            atom.verify()?;
            if atom_hashes
                .insert(atom.atom_identity_sha256.clone(), atom.atom_index)
                .is_some()
                || !atom_indices.insert(atom.atom_index)
            {
                bail!("physical lining atom identity population is duplicate");
            }
        }
        let mut total_voxels = 0u64;
        let mut total_faces = 0u64;
        let mut total_atom_faces = 0u64;
        let mut observed = 0u64;
        let mut empty = 0u64;
        let mut seen_atoms = BTreeSet::new();
        let mut seen_residues = BTreeSet::new();
        for (ordinal, grid) in self.grids.iter().enumerate() {
            let expected_grid_source = canonical_sha256(&GridSourceDigest {
                schema_kind: "prism_physical_cavity_grid_source",
                schema_version: "1.0.0",
                ghost_record_sha256: &self.ghost_record_sha256,
                coordinate_frame_sha256: &self.coordinate_frame_sha256,
                topology_sha256: &self.topology_sha256,
                atom_residue_mapping_sha256: &self.atom_residue_mapping_sha256,
                estimator_sha256: &self.estimator_sha256,
                spatial_metric_authority_sha256: &self.spatial_metric.authority_sha256,
                grid_ordinal: ordinal as u64,
                spacing_a: grid.spacing_a,
                phase_offset_fraction: grid.phase_offset_fraction,
                dimension: grid.dimension,
                origin_relative_angstrom: grid.origin_relative_angstrom,
                occupied_grid_sha256: &grid.occupied_grid_sha256,
                candidate_grid_sha256: &grid.candidate_grid_sha256,
                selected_component_grid_sha256: &grid.selected_component_grid_sha256,
            })?;
            if expected_grid_source != grid.grid_source_sha256 {
                bail!("physical lining grid is not bound to evidence source bytes");
            }
            verify_physical_grid_lining(grid, ordinal as u64, &atom_hashes)?;
            total_voxels = total_voxels
                .checked_add(grid.selected_cavity_voxel_count)
                .context("physical lining voxel conservation overflow")?;
            total_faces = total_faces
                .checked_add(grid.boundary_face_count)
                .context("physical lining face conservation overflow")?;
            for attribution in &grid.per_atom_attributions {
                total_atom_faces = total_atom_faces
                    .checked_add(attribution.boundary_face_count)
                    .context("physical lining atom-face conservation overflow")?;
                seen_atoms.insert(attribution.atom_identity_sha256.clone());
                let atom_index = atom_hashes[&attribution.atom_identity_sha256];
                let atom = self
                    .atom_identities
                    .iter()
                    .find(|atom| atom.atom_index == atom_index)
                    .context("physical lining atom attribution lost identity")?;
                seen_residues.insert((
                    atom.chain_id.clone(),
                    atom.analysis_residue_id,
                    atom.pdb_residue_id,
                ));
            }
            if grid.selected_cavity_voxel_count == 0 {
                empty += 1;
            } else {
                observed += 1;
            }
        }
        let uncertainty = self
            .grids
            .iter()
            .map(|grid| {
                (
                    grid.grid_ordinal,
                    grid.spacing_a,
                    grid.phase_offset_fraction,
                    grid.selected_cavity_volume_a3,
                    grid.boundary_area_a2,
                    grid.grid_record_sha256.as_str(),
                )
            })
            .collect::<Vec<_>>();
        if canonical_sha256(&uncertainty)? != self.conservation.uncertainty_input_sha256
            || total_voxels != self.conservation.selected_cavity_voxel_count_from_grids
            || total_voxels != self.conservation.selected_cavity_voxel_count_conserved
            || total_faces != self.conservation.boundary_face_count_from_grids
            || total_faces != self.conservation.boundary_face_count_from_records
            || total_atom_faces != self.conservation.boundary_face_count_from_atom_attributions
            || total_faces != total_atom_faces
            || seen_atoms.len() as u64 != self.conservation.unique_lining_atom_count
            || seen_residues.len() as u64 != self.conservation.unique_lining_analysis_residue_count
            || observed != self.conservation.grids_with_observed_cavity
            || empty != self.conservation.grids_with_observed_no_cavity
            || observed + empty != self.grids.len() as u64
            || !self.conservation.conservation_pass
            || self.body_sha256()? != self.evidence_sha256
        {
            bail!("physical cavity lining conservation or content hash mismatch");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoordinateCavitySidecarBody {
    pub ghost_record_sha256: String,
    pub stream_id: u16,
    pub step_idx: u64,
    pub coordinate_frame_sha256: String,
    pub trajectory_artifact_sha256: String,
    pub trajectory_audit_sha256: String,
    pub topology_sha256: String,
    pub atom_residue_mapping_sha256: String,
    pub contact_residue_ids: Vec<u32>,
    pub estimator_config: CoordinateCavityConfig,
    pub estimator_sha256: String,
    pub grid_estimates: Vec<CavityGridEstimate>,
    pub mean_volume_a3: f64,
    pub discretization_standard_error_a3: f64,
    pub discretization_min_a3: f64,
    pub discretization_max_a3: f64,
    pub physical_lining: PhysicalCavityLiningEvidence,
    pub raw_coordinates_embedded: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoordinateCavitySidecar {
    pub schema_kind: String,
    pub schema_version: String,
    pub sidecar_sha256: String,
    pub body: CoordinateCavitySidecarBody,
}

impl CoordinateCavitySidecar {
    /// Verify exact content identity and recompute every derived grid summary.
    pub fn verify(&self) -> Result<()> {
        if self.schema_kind != COORDINATE_CAVITY_SCHEMA
            || self.schema_version != COORDINATE_CAVITY_VERSION
            || self.body.raw_coordinates_embedded
        {
            bail!("invalid coordinate cavity sidecar schema or raw-coordinate boundary");
        }
        for (value, field) in [
            (&self.sidecar_sha256, "sidecar_sha256"),
            (&self.body.ghost_record_sha256, "ghost_record_sha256"),
            (
                &self.body.coordinate_frame_sha256,
                "coordinate_frame_sha256",
            ),
            (
                &self.body.trajectory_artifact_sha256,
                "trajectory_artifact_sha256",
            ),
            (
                &self.body.trajectory_audit_sha256,
                "trajectory_audit_sha256",
            ),
            (&self.body.topology_sha256, "topology_sha256"),
            (
                &self.body.atom_residue_mapping_sha256,
                "atom_residue_mapping_sha256",
            ),
            (&self.body.estimator_sha256, "estimator_sha256"),
        ] {
            require_sha256(value, field)?;
        }
        self.body.estimator_config.validate()?;
        if canonical_sha256(&self.body.estimator_config)? != self.body.estimator_sha256 {
            bail!("coordinate cavity estimator configuration hash mismatch");
        }
        if self.body.contact_residue_ids.is_empty()
            || self
                .body
                .contact_residue_ids
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            bail!("coordinate cavity residue population is empty, duplicate, or unsorted");
        }
        let expected_grids = self.body.estimator_config.grid_spacings_a.len()
            * self.body.estimator_config.grid_phase_offsets.len();
        if self.body.grid_estimates.len() != expected_grids {
            bail!("coordinate cavity grid population is incomplete");
        }
        let mut grid_index = 0usize;
        for spacing in &self.body.estimator_config.grid_spacings_a {
            for phase in &self.body.estimator_config.grid_phase_offsets {
                let estimate = &self.body.grid_estimates[grid_index];
                grid_index += 1;
                let expected_volume = estimate.cavity_voxels as f64 * spacing.powi(3);
                if estimate.spacing_a.to_bits() != spacing.to_bits()
                    || estimate.phase_offset_fraction.to_bits() != phase.to_bits()
                    || !estimate.volume_a3.is_finite()
                    || estimate.volume_a3 < 0.0
                    || (estimate.volume_a3 - expected_volume).abs() > 1.0e-9
                {
                    bail!("coordinate cavity grid estimate is invalid or out of order");
                }
            }
        }
        let volumes = self
            .body
            .grid_estimates
            .iter()
            .map(|estimate| estimate.volume_a3)
            .collect::<Vec<_>>();
        let mean = volumes.iter().sum::<f64>() / volumes.len() as f64;
        let variance = volumes
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / (volumes.len() - 1) as f64;
        let standard_error = (variance / volumes.len() as f64).sqrt();
        let min = volumes.iter().copied().fold(f64::INFINITY, f64::min);
        let max = volumes.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if self.body.mean_volume_a3.to_bits() != mean.to_bits()
            || self.body.discretization_standard_error_a3.to_bits() != standard_error.to_bits()
            || self.body.discretization_min_a3.to_bits() != min.to_bits()
            || self.body.discretization_max_a3.to_bits() != max.to_bits()
            || canonical_sha256(&self.body)? != self.sidecar_sha256
        {
            bail!("coordinate cavity aggregate or content hash mismatch");
        }
        self.body.physical_lining.verify()?;
        if self.body.physical_lining.ghost_record_sha256 != self.body.ghost_record_sha256
            || self.body.physical_lining.coordinate_frame_sha256
                != self.body.coordinate_frame_sha256
            || self.body.physical_lining.topology_sha256 != self.body.topology_sha256
            || self.body.physical_lining.atom_residue_mapping_sha256
                != self.body.atom_residue_mapping_sha256
            || self.body.physical_lining.estimator_sha256 != self.body.estimator_sha256
            || self.body.physical_lining.grids.len() != self.body.grid_estimates.len()
            || self
                .body
                .grid_estimates
                .iter()
                .zip(&self.body.physical_lining.grids)
                .any(|(estimate, grid)| {
                    estimate.spacing_a.to_bits() != grid.spacing_a.to_bits()
                        || estimate.phase_offset_fraction.to_bits()
                            != grid.phase_offset_fraction.to_bits()
                        || estimate.cavity_voxels != grid.selected_cavity_voxel_count
                        || estimate.volume_a3.to_bits() != grid.selected_cavity_volume_a3.to_bits()
                })
        {
            bail!("physical cavity lining differs from its coordinate sidecar");
        }
        Ok(())
    }
}

/// Exact coordinate result used by the Path-B orchestration input.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoordinateCavityObservation {
    pub coordinate_frame: ExactCoordinateFrameBinding,
    pub pocket_volume_a3: PhysicalMeasurement,
    pub sidecar: CoordinateCavitySidecar,
    pub physical_lining: PhysicalCavityLiningEvidence,
}

/// Parse and verify one lossless streamed V2 trajectory and its audit sidecar.
pub fn load_lossless_v2_trajectory(
    trajectory_path: &Path,
    audit_path: &Path,
    stream_id: u16,
    expected_atom_count: usize,
) -> Result<LosslessV2TrajectoryIndex> {
    if expected_atom_count == 0 {
        bail!("expected atom count must be non-zero");
    }
    let bytes = read_regular_nofollow(trajectory_path, "trajectory")?;
    let audit_bytes = read_regular_nofollow(audit_path, "trajectory audit")?;
    let audit: serde_json::Value = serde_json::from_slice(&audit_bytes)?;
    let producer = audit_u64(&audit, "producer_frames_enqueued")?;
    let written = audit_u64(&audit, "frames_written")?;
    let mismatches = audit_u64(&audit, "hash_mismatches")?;
    let expected_floats = audit
        .get("expected_floats_per_frame")
        .and_then(serde_json::Value::as_u64)
        .context("trajectory audit lacks expected_floats_per_frame")?;
    let audit_stream_id = audit_u64(&audit, "stream_id")?;
    let audit_bytes_written = audit_u64(&audit, "bytes_written")?;
    let lossless_required = audit
        .get("lossless_required")
        .and_then(serde_json::Value::as_bool)
        .context("trajectory audit lacks lossless_required")?;
    let audit_frames = audit
        .get("frames")
        .and_then(serde_json::Value::as_array)
        .context("trajectory audit lacks per-frame lineage")?;
    if producer == 0
        || producer != written
        || mismatches != 0
        || audit_stream_id != u64::from(stream_id)
        || audit_bytes_written != bytes.len() as u64
        || !lossless_required
        || audit_frames.len() as u64 != written
        || expected_floats != (expected_atom_count * 3) as u64
    {
        bail!("lossless V2 trajectory audit failed count/hash/shape contract");
    }
    let mut cursor = 0usize;
    let frame_count = read_u64(&bytes, &mut cursor)?;
    if frame_count != written {
        bail!("V2 trajectory header count differs from audit");
    }
    let mut rolling_hash = FNV_OFFSET;
    let mut frames = BTreeMap::new();
    let mut previous_step = None;
    for ordinal in 0..frame_count {
        let start = cursor;
        let step = read_u64(&bytes, &mut cursor)?;
        if previous_step.is_some_and(|previous| step <= previous) {
            bail!("V2 trajectory steps are not strictly increasing");
        }
        previous_step = Some(step);
        let n_floats = read_u32(&bytes, &mut cursor)? as usize;
        if n_floats != expected_atom_count * 3 {
            bail!("V2 trajectory frame shape differs from topology");
        }
        let payload_bytes = n_floats
            .checked_mul(4)
            .context("trajectory payload length overflow")?;
        let end = cursor
            .checked_add(payload_bytes)
            .context("trajectory frame byte range overflow")?;
        let payload = bytes
            .get(cursor..end)
            .context("truncated V2 trajectory frame")?;
        let positions = payload
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        if positions.iter().any(|value| !value.is_finite()) {
            bail!("V2 trajectory contains nonfinite coordinate");
        }
        let record = &bytes[start..end];
        let frame_hash64 = fnv_update(FNV_OFFSET, record);
        let audit_frame = &audit_frames[ordinal as usize];
        let audit_ordinal = audit_u64(audit_frame, "frame_index")?;
        let audit_step = audit_u64(audit_frame, "step")?;
        let audit_n_floats = audit_u64(audit_frame, "n_floats")?;
        let producer_hash = audit_frame
            .get("producer_hash64")
            .and_then(serde_json::Value::as_str)
            .context("trajectory frame audit lacks producer_hash64")?;
        let writer_hash = audit_frame
            .get("writer_hash64")
            .and_then(serde_json::Value::as_str)
            .context("trajectory frame audit lacks writer_hash64")?;
        let expected_hash = format!("{frame_hash64:016x}");
        if audit_ordinal != ordinal
            || audit_step != step
            || audit_n_floats != n_floats as u64
            || producer_hash != expected_hash
            || writer_hash != expected_hash
        {
            bail!("V2 trajectory frame disagrees with per-frame producer/writer audit");
        }
        rolling_hash = fnv_update(rolling_hash, &frame_hash64.to_le_bytes());
        let indexed = IndexedFrame {
            ordinal,
            byte_offset: start as u64,
            byte_length: (end - start) as u64,
            frame_sha256: sha256_bytes(record),
            positions,
        };
        if frames.insert(step, indexed).is_some() {
            bail!("V2 trajectory contains duplicate integration step");
        }
        cursor = end;
    }
    if cursor != bytes.len() {
        bail!("V2 trajectory contains trailing bytes");
    }
    let first_step = frames.first_key_value().map(|(step, _)| *step);
    let last_step = frames.last_key_value().map(|(step, _)| *step);
    let audit_first_step = audit.get("first_step").and_then(serde_json::Value::as_u64);
    let audit_last_step = audit.get("last_step").and_then(serde_json::Value::as_u64);
    if audit_first_step != first_step || audit_last_step != last_step {
        bail!("V2 trajectory first/last step differs from audit");
    }
    let audit_rolling = audit
        .get("rolling_hash64")
        .and_then(serde_json::Value::as_str)
        .context("trajectory audit lacks rolling_hash64")?
        .trim_start_matches("fnv1a64:");
    if audit_rolling != format!("{rolling_hash:016x}") {
        bail!("V2 trajectory rolling hash differs from audit");
    }
    Ok(LosslessV2TrajectoryIndex {
        stream_id,
        trajectory_sha256: sha256_bytes(&bytes),
        audit_sha256: sha256_bytes(&audit_bytes),
        atom_count: expected_atom_count,
        frames,
    })
}

fn read_regular_nofollow(path: &Path, label: &str) -> Result<Vec<u8>> {
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("inspect {label} {}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        bail!("{label} must be a regular non-symlink file");
    }
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)
        .with_context(|| format!("open {label} {}", path.display()))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    Ok(bytes)
}

/// Compute an exact-frame, PBC-aware, multi-grid cavity sidecar.
pub fn compute_coordinate_cavity(
    trajectory: &LosslessV2TrajectoryIndex,
    step_idx: u64,
    ghost_record_sha256: &str,
    centroid_a: [f64; 3],
    contact_residue_ids: &[u32],
    topology: &PrismPrepTopology,
    topology_sha256: &str,
    config: &CoordinateCavityConfig,
) -> Result<CoordinateCavityObservation> {
    require_sha256(ghost_record_sha256, "ghost_record_sha256")?;
    require_sha256(topology_sha256, "topology_sha256")?;
    config.validate()?;
    if topology.n_atoms != trajectory.atom_count
        || topology.lj_params.len() != topology.n_atoms
        || topology.residue_ids.len() != topology.n_atoms
    {
        bail!("trajectory/topology atom or parameter population mismatch");
    }
    let frame = trajectory
        .frames
        .get(&step_idx)
        .context("exact V2 stream/step coordinate frame is absent; interpolation is forbidden")?;
    let metric = PhysicalSpatialMetric::from_topology(topology)?;
    let identity_map = receptor_atom_identity_map(topology)?;
    let mapping_sha256 = topology
        .analysis_residue_contract
        .as_ref()
        .unwrap()
        .atom_analysis_residue_ids_sha256
        .clone();
    let contact_set = contact_residue_ids.iter().copied().collect::<BTreeSet<_>>();
    if contact_set.is_empty() {
        bail!("coordinate cavity requires exact Ghost residue membership");
    }
    let receptor_atoms = receptor_atom_indices(topology)?;
    if !receptor_atoms.iter().any(|index| {
        u32::try_from(topology.residue_ids[*index])
            .ok()
            .is_some_and(|residue| contact_set.contains(&residue))
    }) {
        bail!("Ghost residue membership does not map to receptor atoms");
    }
    let mut local_atoms = Vec::new();
    for atom_index in receptor_atoms {
        let position = [
            frame.positions[3 * atom_index] as f64,
            frame.positions[3 * atom_index + 1] as f64,
            frame.positions[3 * atom_index + 2] as f64,
        ];
        let relative = metric.displacement(position, centroid_a);
        let sigma = topology.lj_params[atom_index].sigma;
        if !sigma.is_finite() || sigma <= 0.0 {
            bail!("receptor atom has invalid exact LJ sigma");
        }
        let vdw_radius = 0.5 * 2.0f64.powf(1.0 / 6.0) * sigma;
        local_atoms.push(local_receptor_atom(
            &identity_map,
            atom_index,
            relative,
            vdw_radius + config.solvent_probe_radius_a,
        )?);
    }
    if local_atoms.is_empty() {
        bail!("no receptor atoms enter the coordinate cavity window");
    }
    let estimator_sha256 = canonical_sha256(config)?;
    let mut grid_estimates = Vec::new();
    let mut lining_grids = Vec::new();
    let source = PhysicalGridSourceBinding {
        region_frame_observation: false,
        ghost_record_sha256,
        coordinate_frame_sha256: &frame.frame_sha256,
        topology_sha256,
        atom_residue_mapping_sha256: &mapping_sha256,
        estimator_sha256: &estimator_sha256,
    };
    for spacing in &config.grid_spacings_a {
        for phase in &config.grid_phase_offsets {
            let evaluated = evaluate_grid_with_physical_lining(
                &local_atoms,
                *spacing,
                *phase,
                lining_grids.len() as u64,
                config,
                &metric,
                &source,
            )?;
            grid_estimates.push(evaluated.estimate);
            lining_grids.push(evaluated.lining);
        }
    }
    let physical_lining = materialize_physical_lining(
        ghost_record_sha256,
        &frame.frame_sha256,
        topology_sha256,
        &mapping_sha256,
        &topology
            .analysis_residue_contract
            .as_ref()
            .unwrap()
            .contract_sha256,
        &topology
            .molecule_identity_table
            .as_ref()
            .unwrap()
            .table_sha256,
        &estimator_sha256,
        &metric,
        &local_atoms,
        lining_grids,
    )?;
    let volumes = grid_estimates
        .iter()
        .map(|estimate| estimate.volume_a3)
        .collect::<Vec<_>>();
    let mean = volumes.iter().sum::<f64>() / volumes.len() as f64;
    let variance = volumes
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (volumes.len() - 1) as f64;
    let standard_error = (variance / volumes.len() as f64).sqrt();
    let min = volumes.iter().copied().fold(f64::INFINITY, f64::min);
    let max = volumes.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut residues = contact_set.into_iter().collect::<Vec<_>>();
    residues.sort_unstable();
    let body = CoordinateCavitySidecarBody {
        ghost_record_sha256: ghost_record_sha256.to_string(),
        stream_id: trajectory.stream_id,
        step_idx,
        coordinate_frame_sha256: frame.frame_sha256.clone(),
        trajectory_artifact_sha256: trajectory.trajectory_sha256.clone(),
        trajectory_audit_sha256: trajectory.audit_sha256.clone(),
        topology_sha256: topology_sha256.to_string(),
        atom_residue_mapping_sha256: mapping_sha256.clone(),
        contact_residue_ids: residues,
        estimator_config: config.clone(),
        estimator_sha256: estimator_sha256.clone(),
        grid_estimates,
        mean_volume_a3: mean,
        discretization_standard_error_a3: standard_error,
        discretization_min_a3: min,
        discretization_max_a3: max,
        physical_lining: physical_lining.clone(),
        raw_coordinates_embedded: false,
    };
    let sidecar_sha256 = canonical_sha256(&body)?;
    let sidecar = CoordinateCavitySidecar {
        schema_kind: COORDINATE_CAVITY_SCHEMA.to_string(),
        schema_version: COORDINATE_CAVITY_VERSION.to_string(),
        sidecar_sha256: sidecar_sha256.clone(),
        body,
    };
    sidecar.verify()?;
    let coordinate_frame = ExactCoordinateFrameBinding {
        stream_id: trajectory.stream_id,
        step_idx,
        trajectory_frame_ordinal: frame.ordinal,
        trajectory_byte_offset: frame.byte_offset,
        trajectory_byte_length: frame.byte_length,
        coordinate_frame_sha256: frame.frame_sha256.clone(),
        trajectory_artifact_sha256: trajectory.trajectory_sha256.clone(),
        trajectory_audit_sha256: trajectory.audit_sha256.clone(),
        topology_sha256: topology_sha256.to_string(),
        atom_residue_mapping_sha256: mapping_sha256,
        cavity_sidecar_record_sha256: sidecar_sha256.clone(),
        atom_count: trajectory.atom_count as u64,
        source_contract: "lossless_v2_exact_stream_step_no_interpolation".to_string(),
    };
    let pocket_volume_a3 = PhysicalMeasurement {
        value: mean,
        method_id: "prism_multiresolution_coordinate_ligsite_v1".to_string(),
        estimator_sha256,
        source_artifact_sha256: sidecar_sha256,
    };
    Ok(CoordinateCavityObservation {
        coordinate_frame,
        pocket_volume_a3,
        sidecar,
        physical_lining,
    })
}

/// Compute the production cavity observable from the exact signed-analysis
/// coordinate row already cross-joined to Ghost byte/lattice lineage. No
/// sparse trajectory interval or interpolation path is consulted.
pub fn compute_coordinate_cavity_from_ghost_join(
    coordinate: &VerifiedGhostCoordinate,
    context: &GhostCoordinateContext,
    centroid_a: [f64; 3],
    contact_residue_ids: &[u32],
    topology: &PrismPrepTopology,
    config: &CoordinateCavityConfig,
) -> Result<CoordinateCavityObservation> {
    context.verify()?;
    config.validate()?;
    if coordinate.entry.key.stream_id != context.stream_id
        || coordinate.entry.thermodynamic_condition_id != context.thermodynamic_condition_id
        || coordinate.entry.protocol_state_hash != context.protocol_state_hash
        || coordinate.entry.phase_schedule_digest_sha256 != context.phase_schedule_digest_sha256
        || coordinate.positions_angstrom.len() as u64 != context.analysis_atom_count * 3
    {
        bail!("exact Ghost coordinate row differs from its authenticated stream context");
    }
    for digest in [
        &coordinate.manifest_sha256,
        &coordinate.segment_file_sha256,
        &coordinate.entry.coordinate_frame_sha256,
        &coordinate.entry.key.ghost_record_sha256,
        &context.topology_sha256,
        &context.atom_analysis_residue_ids_sha256,
    ] {
        require_sha256(digest, "Ghost coordinate lineage SHA256")?;
    }
    let analysis_ids = topology
        .atom_analysis_residue_ids
        .as_ref()
        .context("coordinate cavity requires signed atom-analysis mapping")?;
    if analysis_ids.len() != topology.n_atoms || topology.lj_params.len() != topology.n_atoms {
        bail!("signed analysis mapping or exact LJ population differs from topology atoms");
    }
    let receptor_atoms = receptor_atom_indices(topology)?
        .into_iter()
        .collect::<BTreeSet<_>>();
    let analysis_atoms = analysis_ids
        .iter()
        .enumerate()
        .filter_map(|(atom, analysis_id)| (*analysis_id >= 0).then_some((atom, *analysis_id)))
        .collect::<Vec<_>>();
    if analysis_atoms.len() as u64 != context.analysis_atom_count {
        bail!("signed analysis atom population differs from coordinate context");
    }
    let metric = PhysicalSpatialMetric::from_topology(topology)?;
    let identity_map = receptor_atom_identity_map(topology)?;
    let contact_set = contact_residue_ids.iter().copied().collect::<BTreeSet<_>>();
    if contact_set.is_empty()
        || !analysis_atoms.iter().any(|(_, analysis_id)| {
            u32::try_from(*analysis_id)
                .ok()
                .is_some_and(|id| contact_set.contains(&id))
        })
    {
        bail!("Ghost cluster membership does not map to signed receptor analysis residues");
    }
    let mut local_atoms = Vec::new();
    for (compact_index, (atom_index, _)) in analysis_atoms.iter().enumerate() {
        if !receptor_atoms.contains(atom_index) {
            bail!("signed analysis coordinate population contains non-receptor atoms");
        }
        let position = [
            coordinate.positions_angstrom[3 * compact_index] as f64,
            coordinate.positions_angstrom[3 * compact_index + 1] as f64,
            coordinate.positions_angstrom[3 * compact_index + 2] as f64,
        ];
        let relative = metric.displacement(position, centroid_a);
        let sigma = topology.lj_params[*atom_index].sigma;
        if !sigma.is_finite() || sigma <= 0.0 {
            bail!("signed receptor atom has invalid exact LJ sigma");
        }
        let vdw_radius = 0.5 * 2.0f64.powf(1.0 / 6.0) * sigma;
        local_atoms.push(local_receptor_atom(
            &identity_map,
            *atom_index,
            relative,
            vdw_radius + config.solvent_probe_radius_a,
        )?);
    }
    if local_atoms.is_empty() {
        bail!("no signed receptor atoms enter the coordinate cavity window");
    }
    let estimator_sha256 = canonical_sha256(config)?;
    let mut grid_estimates = Vec::new();
    let mut lining_grids = Vec::new();
    let source = PhysicalGridSourceBinding {
        region_frame_observation: false,
        ghost_record_sha256: &coordinate.entry.key.ghost_record_sha256,
        coordinate_frame_sha256: &coordinate.entry.coordinate_frame_sha256,
        topology_sha256: &context.topology_sha256,
        atom_residue_mapping_sha256: &context.atom_analysis_residue_ids_sha256,
        estimator_sha256: &estimator_sha256,
    };
    for spacing in &config.grid_spacings_a {
        for phase in &config.grid_phase_offsets {
            let evaluated = evaluate_grid_with_physical_lining(
                &local_atoms,
                *spacing,
                *phase,
                lining_grids.len() as u64,
                config,
                &metric,
                &source,
            )?;
            grid_estimates.push(evaluated.estimate);
            lining_grids.push(evaluated.lining);
        }
    }
    let physical_lining = materialize_physical_lining(
        &coordinate.entry.key.ghost_record_sha256,
        &coordinate.entry.coordinate_frame_sha256,
        &context.topology_sha256,
        &context.atom_analysis_residue_ids_sha256,
        &topology
            .analysis_residue_contract
            .as_ref()
            .unwrap()
            .contract_sha256,
        &topology
            .molecule_identity_table
            .as_ref()
            .unwrap()
            .table_sha256,
        &estimator_sha256,
        &metric,
        &local_atoms,
        lining_grids,
    )?;
    let volumes = grid_estimates
        .iter()
        .map(|estimate| estimate.volume_a3)
        .collect::<Vec<_>>();
    let mean = volumes.iter().sum::<f64>() / volumes.len() as f64;
    let variance = volumes
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (volumes.len() - 1) as f64;
    let standard_error = (variance / volumes.len() as f64).sqrt();
    let min = volumes.iter().copied().fold(f64::INFINITY, f64::min);
    let max = volumes.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut residues = contact_set.into_iter().collect::<Vec<_>>();
    residues.sort_unstable();
    let body = CoordinateCavitySidecarBody {
        ghost_record_sha256: coordinate.entry.key.ghost_record_sha256.clone(),
        stream_id: coordinate.entry.key.stream_id,
        step_idx: coordinate.entry.key.step_idx,
        coordinate_frame_sha256: coordinate.entry.coordinate_frame_sha256.clone(),
        trajectory_artifact_sha256: coordinate.segment_file_sha256.clone(),
        trajectory_audit_sha256: coordinate.manifest_sha256.clone(),
        topology_sha256: context.topology_sha256.clone(),
        atom_residue_mapping_sha256: context.atom_analysis_residue_ids_sha256.clone(),
        contact_residue_ids: residues,
        estimator_config: config.clone(),
        estimator_sha256: estimator_sha256.clone(),
        grid_estimates,
        mean_volume_a3: mean,
        discretization_standard_error_a3: standard_error,
        discretization_min_a3: min,
        discretization_max_a3: max,
        physical_lining: physical_lining.clone(),
        raw_coordinates_embedded: false,
    };
    let sidecar_sha256 = canonical_sha256(&body)?;
    let sidecar = CoordinateCavitySidecar {
        schema_kind: COORDINATE_CAVITY_SCHEMA.to_string(),
        schema_version: COORDINATE_CAVITY_VERSION.to_string(),
        sidecar_sha256: sidecar_sha256.clone(),
        body,
    };
    sidecar.verify()?;
    let coordinate_frame = ExactCoordinateFrameBinding {
        stream_id: coordinate.entry.key.stream_id,
        step_idx: coordinate.entry.key.step_idx,
        trajectory_frame_ordinal: coordinate.entry.coordinate_frame_ordinal,
        trajectory_byte_offset: coordinate.entry.coordinate_frame_byte_offset,
        trajectory_byte_length: coordinate.entry.coordinate_frame_byte_length,
        coordinate_frame_sha256: coordinate.entry.coordinate_frame_sha256.clone(),
        trajectory_artifact_sha256: coordinate.segment_file_sha256.clone(),
        trajectory_audit_sha256: coordinate.manifest_sha256.clone(),
        topology_sha256: context.topology_sha256.clone(),
        atom_residue_mapping_sha256: context.atom_analysis_residue_ids_sha256.clone(),
        cavity_sidecar_record_sha256: sidecar_sha256.clone(),
        atom_count: context.analysis_atom_count,
        source_contract: "ghost_coordinate_segment_exact_join_v1".to_string(),
    };
    let pocket_volume_a3 = PhysicalMeasurement {
        value: mean,
        method_id: "prism_multiresolution_coordinate_ligsite_v1".to_string(),
        estimator_sha256,
        source_artifact_sha256: sidecar_sha256,
    };
    Ok(CoordinateCavityObservation {
        coordinate_frame,
        pocket_volume_a3,
        sidecar,
        physical_lining,
    })
}

/// Compute the same production cavity observable from one verified
/// fixed-cadence confirmation frame. The observation authority is either the
/// exact emitted Ghost record SHA at that step or the typed no-emission
/// receipt SHA; therefore negative time-at-risk rows remain byte attributable
/// without fabricating a Ghost record.
pub fn compute_coordinate_cavity_from_fixed_cadence(
    frame: &crate::fixed_cadence_coordinate::VerifiedFixedCadenceFrame,
    manifest: &crate::fixed_cadence_coordinate::FixedCadenceCoordinateManifest,
    observation_authority_sha256: &str,
    contact_residue_ids: &[u32],
    topology: &PrismPrepTopology,
    config: &CoordinateCavityConfig,
) -> Result<CoordinateCavityObservation> {
    require_sha256(
        observation_authority_sha256,
        "fixed-cadence observation authority",
    )?;
    manifest.context.verify()?;
    config.validate()?;
    if manifest.context.mode
        != crate::fixed_cadence_coordinate::FixedCadenceCoordinateMode::CandidateConfirmation
        || frame.entry.stream_id != manifest.context.stream_id
        || frame.entry.thermodynamic_condition_id != manifest.context.thermodynamic_condition_id
        || frame.entry.protocol_state_hash != manifest.context.protocol_state_hash
        || frame.entry.phase_schedule_digest_sha256 != manifest.context.phase_schedule_digest_sha256
        || frame.entry.tracked_centroid_angstrom.is_none()
        || frame.analysis_positions_angstrom.len() as u64 != frame.entry.analysis_atom_count * 3
        || frame.full_positions_angstrom.is_none()
    {
        bail!("fixed-cadence cavity frame differs from its confirmation manifest");
    }
    let analysis_ids = topology
        .atom_analysis_residue_ids
        .as_ref()
        .context("fixed-cadence cavity requires signed atom-analysis mapping")?;
    if analysis_ids.len() != topology.n_atoms || topology.lj_params.len() != topology.n_atoms {
        bail!("signed analysis mapping or exact LJ population differs from topology atoms");
    }
    let receptor_atoms = receptor_atom_indices(topology)?
        .into_iter()
        .collect::<BTreeSet<_>>();
    let analysis_atoms = analysis_ids
        .iter()
        .enumerate()
        .filter_map(|(atom, analysis_id)| (*analysis_id >= 0).then_some((atom, *analysis_id)))
        .collect::<Vec<_>>();
    if analysis_atoms.len() as u64 != frame.entry.analysis_atom_count {
        bail!("fixed-cadence signed analysis atom population differs from frame");
    }
    let metric = PhysicalSpatialMetric::from_topology(topology)?;
    let identity_map = receptor_atom_identity_map(topology)?;
    let analysis_contract = topology.analysis_residue_contract.as_ref().unwrap();
    let molecule_table = topology.molecule_identity_table.as_ref().unwrap();
    if analysis_contract.contract_sha256 != manifest.context.analysis_residue_contract_sha256
        || analysis_contract.atom_analysis_residue_ids_sha256
            != manifest.context.atom_analysis_residue_ids_sha256
        || molecule_table.table_sha256 != manifest.context.molecule_identity_table_sha256
        || metric.authority.periodic_cell_contract_sha256
            != manifest.context.periodic_cell_contract_sha256
    {
        bail!("fixed-cadence physical lining topology identity differs from manifest");
    }
    let contact_set = contact_residue_ids.iter().copied().collect::<BTreeSet<_>>();
    if contact_set.is_empty()
        || !analysis_atoms.iter().any(|(_, analysis_id)| {
            u32::try_from(*analysis_id)
                .ok()
                .is_some_and(|id| contact_set.contains(&id))
        })
    {
        bail!("confirmation tracker residues do not map to signed receptor atoms");
    }
    let centroid = frame.entry.tracked_centroid_angstrom.unwrap();
    let centroid_a = [
        f64::from(centroid[0]),
        f64::from(centroid[1]),
        f64::from(centroid[2]),
    ];
    let mut local_atoms = Vec::new();
    for (compact_index, (atom_index, _)) in analysis_atoms.iter().enumerate() {
        if !receptor_atoms.contains(atom_index) {
            bail!("fixed-cadence signed analysis population contains non-receptor atoms");
        }
        let position = [
            f64::from(frame.analysis_positions_angstrom[3 * compact_index]),
            f64::from(frame.analysis_positions_angstrom[3 * compact_index + 1]),
            f64::from(frame.analysis_positions_angstrom[3 * compact_index + 2]),
        ];
        let relative = metric.displacement(position, centroid_a);
        let sigma = topology.lj_params[*atom_index].sigma;
        if !sigma.is_finite() || sigma <= 0.0 {
            bail!("fixed-cadence receptor atom has invalid exact LJ sigma");
        }
        let vdw_radius = 0.5 * 2.0f64.powf(1.0 / 6.0) * sigma;
        local_atoms.push(local_receptor_atom(
            &identity_map,
            *atom_index,
            relative,
            vdw_radius + config.solvent_probe_radius_a,
        )?);
    }
    if local_atoms.is_empty() {
        bail!("no signed receptor atoms enter the fixed-cadence cavity window");
    }
    let estimator_sha256 = canonical_sha256(config)?;
    let mut grid_estimates = Vec::new();
    let mut lining_grids = Vec::new();
    let source = PhysicalGridSourceBinding {
        region_frame_observation: false,
        ghost_record_sha256: observation_authority_sha256,
        coordinate_frame_sha256: &frame.entry.analysis_coordinate_sha256,
        topology_sha256: &manifest.context.topology_sha256,
        atom_residue_mapping_sha256: &manifest.context.atom_analysis_residue_ids_sha256,
        estimator_sha256: &estimator_sha256,
    };
    for spacing in &config.grid_spacings_a {
        for phase in &config.grid_phase_offsets {
            let evaluated = evaluate_grid_with_physical_lining(
                &local_atoms,
                *spacing,
                *phase,
                lining_grids.len() as u64,
                config,
                &metric,
                &source,
            )?;
            grid_estimates.push(evaluated.estimate);
            lining_grids.push(evaluated.lining);
        }
    }
    let physical_lining = materialize_physical_lining(
        observation_authority_sha256,
        &frame.entry.analysis_coordinate_sha256,
        &manifest.context.topology_sha256,
        &manifest.context.atom_analysis_residue_ids_sha256,
        &analysis_contract.contract_sha256,
        &molecule_table.table_sha256,
        &estimator_sha256,
        &metric,
        &local_atoms,
        lining_grids,
    )?;
    let volumes = grid_estimates
        .iter()
        .map(|estimate| estimate.volume_a3)
        .collect::<Vec<_>>();
    let mean = volumes.iter().sum::<f64>() / volumes.len() as f64;
    let variance = volumes
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (volumes.len() - 1) as f64;
    let standard_error = (variance / volumes.len() as f64).sqrt();
    let min = volumes.iter().copied().fold(f64::INFINITY, f64::min);
    let max = volumes.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut residues = contact_set.into_iter().collect::<Vec<_>>();
    residues.sort_unstable();
    let manifest_sha256 = canonical_sha256(manifest)?;
    let body = CoordinateCavitySidecarBody {
        ghost_record_sha256: observation_authority_sha256.to_string(),
        stream_id: frame.entry.stream_id,
        step_idx: frame.entry.step_idx,
        coordinate_frame_sha256: frame.entry.analysis_coordinate_sha256.clone(),
        trajectory_artifact_sha256: frame.entry.file_sha256.clone(),
        trajectory_audit_sha256: manifest_sha256.clone(),
        topology_sha256: manifest.context.topology_sha256.clone(),
        atom_residue_mapping_sha256: manifest.context.atom_analysis_residue_ids_sha256.clone(),
        contact_residue_ids: residues,
        estimator_config: config.clone(),
        estimator_sha256: estimator_sha256.clone(),
        grid_estimates,
        mean_volume_a3: mean,
        discretization_standard_error_a3: standard_error,
        discretization_min_a3: min,
        discretization_max_a3: max,
        physical_lining: physical_lining.clone(),
        raw_coordinates_embedded: false,
    };
    let sidecar_sha256 = canonical_sha256(&body)?;
    let sidecar = CoordinateCavitySidecar {
        schema_kind: COORDINATE_CAVITY_SCHEMA.to_string(),
        schema_version: COORDINATE_CAVITY_VERSION.to_string(),
        sidecar_sha256: sidecar_sha256.clone(),
        body,
    };
    sidecar.verify()?;
    let coordinate_frame = ExactCoordinateFrameBinding {
        stream_id: frame.entry.stream_id,
        step_idx: frame.entry.step_idx,
        trajectory_frame_ordinal: frame.entry.segment_ordinal,
        trajectory_byte_offset: 0,
        trajectory_byte_length: frame.entry.file_bytes,
        coordinate_frame_sha256: frame.entry.analysis_coordinate_sha256.clone(),
        trajectory_artifact_sha256: frame.entry.file_sha256.clone(),
        trajectory_audit_sha256: manifest_sha256,
        topology_sha256: manifest.context.topology_sha256.clone(),
        atom_residue_mapping_sha256: manifest.context.atom_analysis_residue_ids_sha256.clone(),
        cavity_sidecar_record_sha256: sidecar_sha256.clone(),
        atom_count: frame.entry.analysis_atom_count,
        source_contract: "fixed_cadence_candidate_confirmation_v1".to_string(),
    };
    let pocket_volume_a3 = PhysicalMeasurement {
        value: mean,
        method_id: "prism_multiresolution_coordinate_ligsite_v1".to_string(),
        estimator_sha256,
        source_artifact_sha256: sidecar_sha256,
    };
    Ok(CoordinateCavityObservation {
        coordinate_frame,
        pocket_volume_a3,
        sidecar,
        physical_lining,
    })
}

fn receptor_atom_indices(topology: &PrismPrepTopology) -> Result<Vec<usize>> {
    let table = topology
        .molecule_identity_table
        .as_ref()
        .context("coordinate cavity requires exact molecule identity table")?;
    let mut atoms = Vec::new();
    for molecule in &table.molecules {
        let kind = molecule.kind.to_ascii_lowercase();
        if kind.contains("biomolecule")
            || kind.contains("protein")
            || kind.contains("receptor")
            || kind.contains("polymer")
        {
            atoms.extend(molecule.atom_indices.iter().copied());
        }
    }
    atoms.sort_unstable();
    atoms.dedup();
    if atoms.is_empty() || atoms.iter().any(|index| *index >= topology.n_atoms) {
        bail!("molecule identity table has no valid receptor atom partition");
    }
    Ok(atoms)
}

#[derive(Clone)]
struct LocalReceptorAtom {
    position_relative_angstrom: [f64; 3],
    solvent_excluded_radius_angstrom: f64,
    identity: PhysicalCavityLiningAtomIdentity,
}

/// Reuse the native spatial authority for recorded measurement fibers.
/// The caller retains the original topology pin and any declared legacy
/// nonperiodic normalization; no periodic box or PME data are manufactured.
pub(crate) fn physical_spatial_metric_authority(
    topology: &PrismPrepTopology,
) -> Result<PhysicalCavitySpatialMetricAuthority> {
    Ok(PhysicalSpatialMetric::from_topology(topology)?.authority)
}

struct PhysicalSpatialMetric {
    authority: PhysicalCavitySpatialMetricAuthority,
    box_matrix: Option<[[f64; 3]; 3]>,
    inverse_box: Option<[[f64; 3]; 3]>,
}

impl PhysicalSpatialMetric {
    fn from_topology(topology: &PrismPrepTopology) -> Result<Self> {
        let cell = topology
            .periodic_cell
            .as_ref()
            .context("physical cavity lining requires a spatial boundary contract")?;
        let contract_sha256 = cell
            .contract_sha256
            .as_deref()
            .context("physical cavity lining boundary contract lacks SHA256")?;
        require_sha256(
            contract_sha256,
            "physical cavity lining periodic-cell contract",
        )?;
        if cell.periodic {
            let row_vectors = cell
                .vectors_angstrom
                .context("periodic physical cavity lining lacks triclinic vectors")?;
            if cell.minimum_image_required != Some(true)
                || cell.coordinate_wrapping_required == Some(false)
                || row_vectors.iter().flatten().any(|value| !value.is_finite())
            {
                bail!("periodic physical cavity lining lacks minimum-image authority");
            }
            let box_matrix = row_vectors_to_matrix(row_vectors);
            let inverse_box = invert_3x3(box_matrix).context("periodic box is singular")?;
            let authority = PhysicalCavitySpatialMetricAuthority {
                schema_kind: "prism_physical_cavity_spatial_metric_authority".to_string(),
                schema_version: "1.0.0".to_string(),
                authority_sha256: String::new(),
                periodic_cell_contract_sha256: contract_sha256.to_string(),
                mode: PhysicalCavitySpatialMetricMode::PeriodicTriclinicMinimumImage,
                box_vectors_angstrom: Some(row_vectors),
                nonperiodic_reason: None,
            }
            .materialize()?;
            Ok(Self {
                authority,
                box_matrix: Some(box_matrix),
                inverse_box: Some(inverse_box),
            })
        } else {
            if cell.vectors_angstrom.is_some()
                || cell.minimum_image_required == Some(true)
                || cell
                    .reason
                    .as_deref()
                    .is_none_or(|reason| reason.trim().is_empty())
            {
                bail!("nonperiodic physical cavity lining lacks explicit Euclidean authority");
            }
            let authority = PhysicalCavitySpatialMetricAuthority {
                schema_kind: "prism_physical_cavity_spatial_metric_authority".to_string(),
                schema_version: "1.0.0".to_string(),
                authority_sha256: String::new(),
                periodic_cell_contract_sha256: contract_sha256.to_string(),
                mode: PhysicalCavitySpatialMetricMode::DeclaredNonperiodicEuclidean,
                box_vectors_angstrom: None,
                nonperiodic_reason: cell.reason.clone(),
            }
            .materialize()?;
            Ok(Self {
                authority,
                box_matrix: None,
                inverse_box: None,
            })
        }
    }

    fn displacement(&self, left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
        let delta = sub(left, right);
        match (self.box_matrix, self.inverse_box) {
            (Some(box_matrix), Some(inverse_box)) => minimum_image(delta, box_matrix, inverse_box),
            (None, None) => delta,
            _ => unreachable!("verified metric stores box matrix and inverse together"),
        }
    }

    fn require_unique_periodic_local_window(
        &self,
        origin: f64,
        terminal: f64,
        maximum_atom_radius: f64,
    ) -> Result<()> {
        let Some(inverse) = self.inverse_box else {
            return Ok(());
        };
        let minimum_cell_height = inverse
            .iter()
            .map(|reciprocal_row| 1.0 / norm2(*reciprocal_row).sqrt())
            .fold(f64::INFINITY, f64::min);
        let maximum_grid_radius = 3.0f64.sqrt() * origin.abs().max(terminal.abs());
        if !minimum_cell_height.is_finite()
            || minimum_cell_height <= 0.0
            || !maximum_atom_radius.is_finite()
            || maximum_atom_radius <= 0.0
            || 2.0 * (maximum_grid_radius + maximum_atom_radius) >= minimum_cell_height
        {
            bail!(
                "triclinic cell cannot uniquely support the declared local cavity window; periodic image expansion is required"
            );
        }
        Ok(())
    }
}

fn row_vectors_to_matrix(row_vectors: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [row_vectors[0][0], row_vectors[1][0], row_vectors[2][0]],
        [row_vectors[0][1], row_vectors[1][1], row_vectors[2][1]],
        [row_vectors[0][2], row_vectors[1][2], row_vectors[2][2]],
    ]
}

pub(crate) fn receptor_atom_identity_map(
    topology: &PrismPrepTopology,
) -> Result<BTreeMap<usize, PhysicalCavityLiningAtomIdentity>> {
    topology
        .validate_analysis_residue_axis()
        .map_err(anyhow::Error::msg)?;
    if topology.elements.len() != topology.n_atoms
        || topology.atom_names.len() != topology.n_atoms
        || topology.residue_names.len() != topology.n_atoms
        || topology.residue_ids.len() != topology.n_atoms
        || topology.chain_ids.len() != topology.n_atoms
        || topology.lj_params.len() != topology.n_atoms
    {
        bail!("physical cavity lining topology atom identity population is incomplete");
    }
    let analysis_ids = topology
        .atom_analysis_residue_ids
        .as_ref()
        .context("physical cavity lining lacks signed analysis atom mapping")?;
    let analysis_contract = topology
        .analysis_residue_contract
        .as_ref()
        .context("physical cavity lining lacks analysis residue contract")?;
    let molecule_table = topology
        .molecule_identity_table
        .as_ref()
        .context("physical cavity lining lacks molecule identity table")?;
    require_sha256(
        &analysis_contract.contract_sha256,
        "physical cavity lining analysis residue contract",
    )?;
    require_sha256(
        &molecule_table.table_sha256,
        "physical cavity lining molecule table",
    )?;
    let analysis_rows = analysis_contract
        .analysis_residues
        .iter()
        .map(|row| (row.analysis_residue_id, row))
        .collect::<BTreeMap<_, _>>();
    let mut atom_molecule = vec![None; topology.n_atoms];
    for molecule in &molecule_table.molecules {
        require_sha256(
            &molecule.component_sha256,
            "physical cavity lining molecule component",
        )?;
        for atom_index in &molecule.atom_indices {
            if *atom_index >= topology.n_atoms
                || atom_molecule[*atom_index].replace(molecule).is_some()
            {
                bail!("physical cavity lining molecule partition is overlapping or invalid");
            }
        }
    }
    if atom_molecule.iter().any(Option::is_none) {
        bail!("physical cavity lining molecule partition does not cover every atom");
    }
    let receptor_indices = receptor_atom_indices(topology)?;
    let signed_receptor_indices = analysis_ids
        .iter()
        .enumerate()
        .filter_map(|(atom_index, analysis_id)| (*analysis_id >= 0).then_some(atom_index))
        .collect::<Vec<_>>();
    if receptor_indices != signed_receptor_indices {
        bail!("molecule table receptor partition differs from signed analysis population");
    }
    let mut identities = BTreeMap::new();
    for atom_index in receptor_indices {
        let analysis_id = u32::try_from(analysis_ids[atom_index])
            .context("receptor atom is excluded from signed analysis identity")?;
        let row = analysis_rows
            .get(&(analysis_id as usize))
            .context("receptor atom analysis identity lacks signed residue row")?;
        let molecule = atom_molecule[atom_index].unwrap();
        if row.physical_residue_id != topology.residue_ids[atom_index]
            || row.chain_id != topology.chain_ids[atom_index]
            || row.residue_name != topology.residue_names[atom_index]
        {
            bail!("physical cavity lining atom/residue/chain identity is inconsistent");
        }
        let identity = PhysicalCavityLiningAtomIdentity {
            atom_identity_sha256: String::new(),
            atom_index,
            atom_name: topology.atom_names[atom_index].clone(),
            element: topology.elements[atom_index].clone(),
            physical_residue_id: topology.residue_ids[atom_index],
            analysis_residue_id: analysis_id,
            pdb_residue_id: row.pdb_residue_id,
            residue_name: topology.residue_names[atom_index].clone(),
            chain_id: topology.chain_ids[atom_index].clone(),
            molecule_id: molecule.molecule_id,
            molecule_identity: molecule.identity.clone(),
            molecule_component_sha256: molecule.component_sha256.clone(),
        }
        .materialize()?;
        identities.insert(atom_index, identity);
    }
    if identities.is_empty() {
        bail!("physical cavity lining has no admitted receptor atoms");
    }
    Ok(identities)
}

fn local_receptor_atom(
    identity_map: &BTreeMap<usize, PhysicalCavityLiningAtomIdentity>,
    atom_index: usize,
    position_relative_angstrom: [f64; 3],
    solvent_excluded_radius_angstrom: f64,
) -> Result<LocalReceptorAtom> {
    if position_relative_angstrom
        .iter()
        .any(|value| !value.is_finite())
        || !solvent_excluded_radius_angstrom.is_finite()
        || solvent_excluded_radius_angstrom <= 0.0
    {
        bail!("physical cavity lining local receptor geometry is invalid");
    }
    Ok(LocalReceptorAtom {
        position_relative_angstrom,
        solvent_excluded_radius_angstrom,
        identity: identity_map
            .get(&atom_index)
            .context("local receptor atom lacks admitted topology identity")?
            .clone(),
    })
}

#[derive(Serialize)]
struct GridSourceDigest<'a> {
    schema_kind: &'static str,
    schema_version: &'static str,
    ghost_record_sha256: &'a str,
    coordinate_frame_sha256: &'a str,
    topology_sha256: &'a str,
    atom_residue_mapping_sha256: &'a str,
    estimator_sha256: &'a str,
    spatial_metric_authority_sha256: &'a str,
    grid_ordinal: u64,
    spacing_a: f64,
    phase_offset_fraction: f64,
    dimension: u64,
    origin_relative_angstrom: f64,
    occupied_grid_sha256: &'a str,
    candidate_grid_sha256: &'a str,
    selected_component_grid_sha256: &'a str,
}

struct PhysicalGridSourceBinding<'a> {
    region_frame_observation: bool,
    ghost_record_sha256: &'a str,
    coordinate_frame_sha256: &'a str,
    topology_sha256: &'a str,
    atom_residue_mapping_sha256: &'a str,
    estimator_sha256: &'a str,
}

struct EvaluatedCavityGrid {
    estimate: CavityGridEstimate,
    lining: PhysicalCavityGridLiningEvidence,
}

fn evaluate_grid(
    atoms: &[([f64; 3], f64)],
    spacing: f64,
    phase: f64,
    config: &CoordinateCavityConfig,
) -> Result<CavityGridEstimate> {
    let local_atoms = atoms
        .iter()
        .enumerate()
        .map(|(atom_index, (position, radius))| {
            let identity = PhysicalCavityLiningAtomIdentity {
                atom_identity_sha256: String::new(),
                atom_index,
                atom_name: format!("TEST{atom_index}"),
                element: "C".to_string(),
                physical_residue_id: atom_index,
                analysis_residue_id: atom_index as u32,
                pdb_residue_id: atom_index as i64,
                residue_name: "TST".to_string(),
                chain_id: "T".to_string(),
                molecule_id: 0,
                molecule_identity: "in-module-grid-geometry-test".to_string(),
                molecule_component_sha256: format!("{:064x}", atom_index + 1),
            }
            .materialize()?;
            Ok(LocalReceptorAtom {
                position_relative_angstrom: *position,
                solvent_excluded_radius_angstrom: *radius,
                identity,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let metric = PhysicalSpatialMetric {
        authority: PhysicalCavitySpatialMetricAuthority {
            schema_kind: "prism_physical_cavity_spatial_metric_authority".to_string(),
            schema_version: "1.0.0".to_string(),
            authority_sha256: String::new(),
            periodic_cell_contract_sha256: "f".repeat(64),
            mode: PhysicalCavitySpatialMetricMode::DeclaredNonperiodicEuclidean,
            box_vectors_angstrom: None,
            nonperiodic_reason: Some("in-module-grid-geometry-test".to_string()),
        }
        .materialize()?,
        box_matrix: None,
        inverse_box: None,
    };
    let digest = "e".repeat(64);
    Ok(evaluate_grid_with_physical_lining(
        &local_atoms,
        spacing,
        phase,
        0,
        config,
        &metric,
        &PhysicalGridSourceBinding {
        region_frame_observation: false,
            ghost_record_sha256: &digest,
            coordinate_frame_sha256: &digest,
            topology_sha256: &digest,
            atom_residue_mapping_sha256: &digest,
            estimator_sha256: &digest,
        },
    )?
    .estimate)
}

fn evaluate_grid_with_physical_lining(
    atoms: &[LocalReceptorAtom],
    spacing: f64,
    phase: f64,
    grid_ordinal: u64,
    config: &CoordinateCavityConfig,
    metric: &PhysicalSpatialMetric,
    source: &PhysicalGridSourceBinding<'_>,
) -> Result<EvaluatedCavityGrid> {
    if atoms.is_empty() {
        bail!("physical cavity grid has no receptor atoms");
    }
    let width = 2.0 * config.local_half_width_a;
    let dimension = (width / spacing).ceil();
    if !dimension.is_finite() || dimension > (usize::MAX - 1) as f64 {
        bail!("physical cavity grid dimension exceeds addressable range");
    }
    let n = dimension as usize + 1;
    let size = n
        .checked_mul(n)
        .and_then(|value| value.checked_mul(n))
        .context("cavity grid size overflow")?;
    let origin = -config.local_half_width_a + phase * spacing;
    let terminal = origin + (n - 1) as f64 * spacing;
    let maximum_atom_radius = atoms
        .iter()
        .map(|atom| atom.solvent_excluded_radius_angstrom)
        .fold(f64::NEG_INFINITY, f64::max);
    metric.require_unique_periodic_local_window(origin, terminal, maximum_atom_radius)?;
    let index = |x: usize, y: usize, z: usize| (x * n + y) * n + z;
    let mut occupied = Vec::new();
    occupied
        .try_reserve_exact(size)
        .context("physical cavity occupied-grid allocation failed")?;
    occupied.resize(size, false);
    for atom in atoms {
        let center = atom.position_relative_angstrom;
        let radius = atom.solvent_excluded_radius_angstrom;
        if (0..3).any(|axis| center[axis] + radius < origin || center[axis] - radius > terminal) {
            continue;
        }
        let lower = (0..3)
            .map(|axis| {
                (((center[axis] - radius - origin) / spacing).floor() as isize)
                    .clamp(0, n as isize - 1)
            })
            .collect::<Vec<_>>();
        let upper = (0..3)
            .map(|axis| {
                (((center[axis] + radius - origin) / spacing).ceil() as isize)
                    .clamp(0, n as isize - 1)
            })
            .collect::<Vec<_>>();
        for x in lower[0]..=upper[0] {
            for y in lower[1]..=upper[1] {
                for z in lower[2]..=upper[2] {
                    let point = [
                        origin + x as f64 * spacing,
                        origin + y as f64 * spacing,
                        origin + z as f64 * spacing,
                    ];
                    if norm2(metric.displacement(point, center)) <= radius * radius {
                        occupied[index(x as usize, y as usize, z as usize)] = true;
                    }
                }
            }
        }
    }
    let horizon = (config.ray_horizon_a / spacing).ceil() as isize;
    let directions = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ];
    let mut candidate = Vec::new();
    candidate
        .try_reserve_exact(size)
        .context("physical cavity candidate-grid allocation failed")?;
    candidate.resize(size, false);
    for x in 0..n {
        for y in 0..n {
            for z in 0..n {
                let here = index(x, y, z);
                if occupied[here] {
                    continue;
                }
                let hits = directions.map(|(dx, dy, dz)| {
                    (1..=horizon).any(|distance| {
                        let nx = x as isize + dx * distance;
                        let ny = y as isize + dy * distance;
                        let nz = z as isize + dz * distance;
                        nx >= 0
                            && ny >= 0
                            && nz >= 0
                            && nx < n as isize
                            && ny < n as isize
                            && nz < n as isize
                            && occupied[index(nx as usize, ny as usize, nz as usize)]
                    })
                });
                let opposed = (hits[0] && hits[1]) || (hits[2] && hits[3]) || (hits[4] && hits[5]);
                candidate[here] = hits.iter().filter(|hit| **hit).count()
                    >= config.minimum_blocked_axial_directions as usize
                    && (!config.require_opposed_wall_pair || opposed);
            }
        }
    }
    // Keep only the candidate component nearest the Ghost centroid, preventing
    // unrelated local voids from inflating the site observable.
    let center_coord = ((0.0 - origin) / spacing).round() as isize;
    let mut seed = None;
    let mut best_distance = f64::INFINITY;
    for x in 0..n {
        for y in 0..n {
            for z in 0..n {
                if candidate[index(x, y, z)] {
                    let d2 = (x as isize - center_coord).pow(2)
                        + (y as isize - center_coord).pow(2)
                        + (z as isize - center_coord).pow(2);
                    if (d2 as f64) < best_distance {
                        best_distance = d2 as f64;
                        seed = Some((x, y, z));
                    }
                }
            }
        }
    }
    let mut selected = Vec::new();
    selected
        .try_reserve_exact(size)
        .context("physical cavity selected-grid allocation failed")?;
    selected.resize(size, false);
    let mut count = 0u64;
    if let Some(seed) = seed {
        let mut visited = Vec::new();
        visited
            .try_reserve_exact(size)
            .context("physical cavity flood-fill allocation failed")?;
        visited.resize(size, false);
        let mut queue = VecDeque::new();
        queue
            .try_reserve(size)
            .context("physical cavity flood-fill queue allocation failed")?;
        queue.push_back(seed);
        visited[index(seed.0, seed.1, seed.2)] = true;
        while let Some((x, y, z)) = queue.pop_front() {
            count = count
                .checked_add(1)
                .context("physical cavity selected voxel count overflow")?;
            selected[index(x, y, z)] = true;
            for (dx, dy, dz) in directions {
                let nx = x as isize + dx;
                let ny = y as isize + dy;
                let nz = z as isize + dz;
                if nx < 0
                    || ny < 0
                    || nz < 0
                    || nx >= n as isize
                    || ny >= n as isize
                    || nz >= n as isize
                {
                    continue;
                }
                let next = index(nx as usize, ny as usize, nz as usize);
                if candidate[next] && !visited[next] {
                    visited[next] = true;
                    queue.push_back((nx as usize, ny as usize, nz as usize));
                }
            }
        }
    }

    let occupied_bytes = pack_grid_bits(&occupied)?;
    let candidate_bytes = pack_grid_bits(&candidate)?;
    let selected_bytes = pack_grid_bits(&selected)?;
    let occupied_sha256 = sha256_bytes(&occupied_bytes);
    let candidate_sha256 = sha256_bytes(&candidate_bytes);
    let selected_sha256 = sha256_bytes(&selected_bytes);
    let grid_digest = GridSourceDigest {
        schema_kind: "prism_physical_cavity_grid_source",
        schema_version: "1.0.0",
        ghost_record_sha256: source.ghost_record_sha256,
        coordinate_frame_sha256: source.coordinate_frame_sha256,
        topology_sha256: source.topology_sha256,
        atom_residue_mapping_sha256: source.atom_residue_mapping_sha256,
        estimator_sha256: source.estimator_sha256,
        spatial_metric_authority_sha256: &metric.authority.authority_sha256,
        grid_ordinal,
        spacing_a: spacing,
        phase_offset_fraction: phase,
        dimension: n as u64,
        origin_relative_angstrom: origin,
        occupied_grid_sha256: &occupied_sha256,
        candidate_grid_sha256: &candidate_sha256,
        selected_component_grid_sha256: &selected_sha256,
    };
    let grid_source_sha256 = if source.region_frame_observation {
        let mut typed = serde_json::to_value(&grid_digest)?;
        let map = typed.as_object_mut().unwrap();
        let identity = map.remove("ghost_record_sha256").unwrap();
        map.insert("materializer_region_frame_sha256".into(), identity);
        map.insert("schema_kind".into(), "prism_materializer_region_frame_cavity_grid_source".into());
        canonical_sha256(&typed)?
    } else {
        canonical_sha256(&grid_digest)?
    };
    let face_directions = [
        (0u8, 1i8, 1isize, 0isize, 0isize),
        (0, -1, -1, 0, 0),
        (1, 1, 0, 1, 0),
        (1, -1, 0, -1, 0),
        (2, 1, 0, 0, 1),
        (2, -1, 0, 0, -1),
    ];
    let maximum_face_count = count
        .checked_mul(6)
        .and_then(|count| usize::try_from(count).ok())
        .context("physical cavity boundary-face capacity overflow")?;
    let mut faces = Vec::new();
    faces
        .try_reserve_exact(maximum_face_count)
        .context("physical cavity boundary-face allocation failed")?;
    for x in 0..n {
        for y in 0..n {
            for z in 0..n {
                let linear = index(x, y, z);
                if !selected[linear] {
                    continue;
                }
                let voxel_center = [
                    origin + x as f64 * spacing,
                    origin + y as f64 * spacing,
                    origin + z as f64 * spacing,
                ];
                for (axis, orientation, dx, dy, dz) in face_directions {
                    let nx = x as isize + dx;
                    let ny = y as isize + dy;
                    let nz = z as isize + dz;
                    let within = nx >= 0
                        && ny >= 0
                        && nz >= 0
                        && nx < n as isize
                        && ny < n as isize
                        && nz < n as isize;
                    if within && selected[index(nx as usize, ny as usize, nz as usize)] {
                        continue;
                    }
                    let boundary_kind = if !within {
                        PhysicalCavityBoundaryKind::LocalWindow
                    } else if occupied[index(nx as usize, ny as usize, nz as usize)] {
                        PhysicalCavityBoundaryKind::ReceptorExcludedVolume
                    } else {
                        PhysicalCavityBoundaryKind::EstimatorMask
                    };
                    let mut face_center = voxel_center;
                    face_center[axis as usize] += f64::from(orientation) * 0.5 * spacing;
                    let (nearest, center_distance, surface_distance) = atoms
                        .iter()
                        .map(|atom| {
                            let center_distance = norm2(
                                metric.displacement(face_center, atom.position_relative_angstrom),
                            )
                            .sqrt();
                            (
                                atom,
                                center_distance,
                                center_distance - atom.solvent_excluded_radius_angstrom,
                            )
                        })
                        .min_by(|left, right| {
                            left.1.total_cmp(&right.1).then_with(|| {
                                left.0.identity.atom_index.cmp(&right.0.identity.atom_index)
                            })
                        })
                        .context("physical cavity boundary face lacks receptor attribution")?;
                    faces.push(
                        PhysicalCavityBoundaryFaceAttribution {
                            face_record_sha256: String::new(),
                            grid_source_sha256: grid_source_sha256.clone(),
                            voxel_linear_index: linear as u64,
                            voxel_xyz: [x as u64, y as u64, z as u64],
                            axis,
                            orientation,
                            boundary_kind,
                            face_center_relative_angstrom: face_center,
                            face_area_a2: spacing.powi(2),
                            nearest_atom_identity_sha256: nearest
                                .identity
                                .atom_identity_sha256
                                .clone(),
                            nearest_atom_index: nearest.identity.atom_index,
                            nearest_atom_center_distance_a: center_distance,
                            nearest_solvent_excluded_surface_distance_a: surface_distance,
                        }
                        .materialize()?,
                    );
                }
            }
        }
    }
    let boundary_faces_sha256 = canonical_sha256(&faces)?;
    let mut aggregate = BTreeMap::<usize, (String, u64, u64, u64, u64, f64, f64, f64)>::new();
    for face in &faces {
        let entry = aggregate.entry(face.nearest_atom_index).or_insert_with(|| {
            (
                face.nearest_atom_identity_sha256.clone(),
                0,
                0,
                0,
                0,
                0.0,
                f64::INFINITY,
                f64::NEG_INFINITY,
            )
        });
        entry.1 = entry
            .1
            .checked_add(1)
            .context("physical cavity per-atom face count overflow")?;
        match face.boundary_kind {
            PhysicalCavityBoundaryKind::ReceptorExcludedVolume => entry.2 += 1,
            PhysicalCavityBoundaryKind::EstimatorMask => entry.3 += 1,
            PhysicalCavityBoundaryKind::LocalWindow => entry.4 += 1,
        }
        entry.5 += face.face_area_a2;
        entry.6 = entry
            .6
            .min(face.nearest_solvent_excluded_surface_distance_a);
        entry.7 = entry
            .7
            .max(face.nearest_solvent_excluded_surface_distance_a);
    }
    let per_atom_attributions = aggregate
        .into_iter()
        .map(
            |(
                atom_index,
                (
                    atom_identity_sha256,
                    face_count,
                    receptor_count,
                    estimator_count,
                    window_count,
                    area,
                    min,
                    max,
                ),
            )| {
                PhysicalCavityGridAtomAttribution {
                    atom_identity_sha256,
                    atom_index,
                    boundary_face_count: face_count,
                    receptor_excluded_volume_face_count: receptor_count,
                    estimator_mask_face_count: estimator_count,
                    local_window_face_count: window_count,
                    boundary_area_a2: area,
                    minimum_surface_distance_a: min,
                    maximum_surface_distance_a: max,
                }
            },
        )
        .collect::<Vec<_>>();
    let receptor_faces = faces
        .iter()
        .filter(|face| face.boundary_kind == PhysicalCavityBoundaryKind::ReceptorExcludedVolume)
        .count() as u64;
    let estimator_faces = faces
        .iter()
        .filter(|face| face.boundary_kind == PhysicalCavityBoundaryKind::EstimatorMask)
        .count() as u64;
    let window_faces = faces
        .iter()
        .filter(|face| face.boundary_kind == PhysicalCavityBoundaryKind::LocalWindow)
        .count() as u64;
    let estimate = CavityGridEstimate {
        spacing_a: spacing,
        phase_offset_fraction: phase,
        cavity_voxels: count,
        volume_a3: count as f64 * spacing.powi(3),
    };
    let mut lining = PhysicalCavityGridLiningEvidence {
        grid_record_sha256: String::new(),
        grid_source_sha256,
        grid_ordinal,
        spacing_a: spacing,
        phase_offset_fraction: phase,
        dimension: n as u64,
        origin_relative_angstrom: origin,
        voxel_count: size as u64,
        packed_bit_semantics: "linear_index_xyz_z_fastest_lsb0_v1".to_string(),
        occupied_grid_bytes: occupied_bytes.len() as u64,
        occupied_grid_sha256: occupied_sha256,
        occupied_grid_packed_lsb0: occupied_bytes,
        candidate_grid_bytes: candidate_bytes.len() as u64,
        candidate_grid_sha256: candidate_sha256,
        candidate_grid_packed_lsb0: candidate_bytes,
        selected_component_grid_bytes: selected_bytes.len() as u64,
        selected_component_grid_sha256: selected_sha256,
        selected_component_grid_packed_lsb0: selected_bytes,
        selected_cavity_voxel_count: count,
        selected_cavity_volume_a3: estimate.volume_a3,
        boundary_face_count: faces.len() as u64,
        receptor_excluded_volume_face_count: receptor_faces,
        estimator_mask_face_count: estimator_faces,
        local_window_face_count: window_faces,
        boundary_area_a2: faces.len() as f64 * spacing.powi(2),
        boundary_faces_sha256,
        boundary_faces: faces,
        per_atom_attributions,
    };
    lining.grid_record_sha256 = physical_grid_record_sha256(&lining)?;
    let atom_hashes = atoms
        .iter()
        .map(|atom| {
            (
                atom.identity.atom_identity_sha256.clone(),
                atom.identity.atom_index,
            )
        })
        .collect::<BTreeMap<_, _>>();
    verify_physical_grid_lining(&lining, grid_ordinal, &atom_hashes)?;
    Ok(EvaluatedCavityGrid { estimate, lining })
}

fn pack_grid_bits(values: &[bool]) -> Result<Vec<u8>> {
    let byte_count = values.len().div_ceil(8);
    let mut packed = Vec::new();
    packed
        .try_reserve_exact(byte_count)
        .context("physical cavity packed-grid allocation failed")?;
    packed.resize(byte_count, 0u8);
    for (index, value) in values.iter().enumerate() {
        if *value {
            packed[index / 8] |= 1 << (index % 8);
        }
    }
    Ok(packed)
}

fn physical_grid_record_sha256(grid: &PhysicalCavityGridLiningEvidence) -> Result<String> {
    let mut body = grid.clone();
    body.grid_record_sha256.clear();
    canonical_sha256(&body)
}

pub(crate) fn verify_physical_grid_lining(
    grid: &PhysicalCavityGridLiningEvidence,
    expected_ordinal: u64,
    atom_hashes: &BTreeMap<String, usize>,
) -> Result<()> {
    for (value, field) in [
        (&grid.grid_record_sha256, "physical lining grid record"),
        (&grid.grid_source_sha256, "physical lining grid source"),
        (&grid.occupied_grid_sha256, "physical lining occupied grid"),
        (
            &grid.candidate_grid_sha256,
            "physical lining candidate grid",
        ),
        (
            &grid.selected_component_grid_sha256,
            "physical lining selected grid",
        ),
        (
            &grid.boundary_faces_sha256,
            "physical lining boundary faces",
        ),
    ] {
        require_sha256(value, field)?;
    }
    let expected_voxels = grid
        .dimension
        .checked_mul(grid.dimension)
        .and_then(|value| value.checked_mul(grid.dimension))
        .context("physical lining grid dimension overflow")?;
    let expected_bytes = expected_voxels.div_ceil(8);
    if grid.grid_ordinal != expected_ordinal
        || !grid.spacing_a.is_finite()
        || grid.spacing_a <= 0.0
        || !grid.phase_offset_fraction.is_finite()
        || !(0.0..1.0).contains(&grid.phase_offset_fraction)
        || !grid.origin_relative_angstrom.is_finite()
        || grid.voxel_count != expected_voxels
        || grid.occupied_grid_bytes != expected_bytes
        || grid.occupied_grid_packed_lsb0.len() as u64 != expected_bytes
        || sha256_bytes(&grid.occupied_grid_packed_lsb0) != grid.occupied_grid_sha256
        || grid.candidate_grid_bytes != expected_bytes
        || grid.candidate_grid_packed_lsb0.len() as u64 != expected_bytes
        || sha256_bytes(&grid.candidate_grid_packed_lsb0) != grid.candidate_grid_sha256
        || grid.selected_component_grid_bytes != expected_bytes
        || grid.selected_component_grid_packed_lsb0.len() as u64 != expected_bytes
        || sha256_bytes(&grid.selected_component_grid_packed_lsb0)
            != grid.selected_component_grid_sha256
        || !packed_grid_padding_is_zero(&grid.occupied_grid_packed_lsb0, expected_voxels)
        || !packed_grid_padding_is_zero(&grid.candidate_grid_packed_lsb0, expected_voxels)
        || !packed_grid_padding_is_zero(&grid.selected_component_grid_packed_lsb0, expected_voxels)
        || grid.packed_bit_semantics != "linear_index_xyz_z_fastest_lsb0_v1"
        || grid.selected_cavity_volume_a3.to_bits()
            != (grid.selected_cavity_voxel_count as f64 * grid.spacing_a.powi(3)).to_bits()
        || grid.boundary_face_count != grid.boundary_faces.len() as u64
        || grid.boundary_area_a2.to_bits()
            != (grid.boundary_face_count as f64 * grid.spacing_a.powi(2)).to_bits()
        || grid.receptor_excluded_volume_face_count
            + grid.estimator_mask_face_count
            + grid.local_window_face_count
            != grid.boundary_face_count
    {
        bail!("invalid physical cavity grid lining dimensions or conservation");
    }
    let selected_count = (0..expected_voxels)
        .filter(|index| grid_bit(&grid.selected_component_grid_packed_lsb0, *index))
        .count() as u64;
    for linear in 0..expected_voxels {
        let occupied = grid_bit(&grid.occupied_grid_packed_lsb0, linear);
        let candidate = grid_bit(&grid.candidate_grid_packed_lsb0, linear);
        let selected = grid_bit(&grid.selected_component_grid_packed_lsb0, linear);
        if (occupied && candidate) || (selected && !candidate) {
            bail!("physical cavity packed grids violate occupancy/component semantics");
        }
    }
    if selected_count != grid.selected_cavity_voxel_count {
        bail!("physical cavity selected packed-grid count is not conserved");
    }
    let directions = [
        (0u8, 1i8, 1i64, 0i64, 0i64),
        (0, -1, -1, 0, 0),
        (1, 1, 0, 1, 0),
        (1, -1, 0, -1, 0),
        (2, 1, 0, 0, 1),
        (2, -1, 0, 0, -1),
    ];
    let mut expected_boundaries = BTreeMap::new();
    for linear in 0..expected_voxels {
        if !grid_bit(&grid.selected_component_grid_packed_lsb0, linear) {
            continue;
        }
        let plane = grid.dimension * grid.dimension;
        let x = linear / plane;
        let remainder = linear % plane;
        let y = remainder / grid.dimension;
        let z = remainder % grid.dimension;
        for (axis, orientation, dx, dy, dz) in directions {
            let nx = x as i64 + dx;
            let ny = y as i64 + dy;
            let nz = z as i64 + dz;
            let within = nx >= 0
                && ny >= 0
                && nz >= 0
                && nx < grid.dimension as i64
                && ny < grid.dimension as i64
                && nz < grid.dimension as i64;
            let neighbor = within
                .then(|| (nx as u64 * grid.dimension + ny as u64) * grid.dimension + nz as u64);
            if neighbor.is_some_and(|neighbor| {
                grid_bit(&grid.selected_component_grid_packed_lsb0, neighbor)
            }) {
                continue;
            }
            let kind = match neighbor {
                None => PhysicalCavityBoundaryKind::LocalWindow,
                Some(neighbor) if grid_bit(&grid.occupied_grid_packed_lsb0, neighbor) => {
                    PhysicalCavityBoundaryKind::ReceptorExcludedVolume
                }
                Some(_) => PhysicalCavityBoundaryKind::EstimatorMask,
            };
            expected_boundaries.insert((linear, axis, orientation), kind);
        }
    }
    let mut keys = BTreeSet::new();
    let mut counts = BTreeMap::<usize, (String, u64, u64, u64, u64, f64, f64, f64)>::new();
    let mut kinds = [0u64; 3];
    for face in &grid.boundary_faces {
        face.verify()?;
        let mapped_index = atom_hashes
            .get(&face.nearest_atom_identity_sha256)
            .context("physical lining face references unknown atom identity")?;
        if face.grid_source_sha256 != grid.grid_source_sha256
            || *mapped_index != face.nearest_atom_index
            || face.voxel_linear_index >= grid.voxel_count
            || face.voxel_xyz.iter().any(|value| *value >= grid.dimension)
            || face.voxel_linear_index
                != (face.voxel_xyz[0] * grid.dimension + face.voxel_xyz[1]) * grid.dimension
                    + face.voxel_xyz[2]
            || face.face_area_a2.to_bits() != grid.spacing_a.powi(2).to_bits()
            || face.face_center_relative_angstrom
                != [
                    grid.origin_relative_angstrom
                        + face.voxel_xyz[0] as f64 * grid.spacing_a
                        + if face.axis == 0 {
                            f64::from(face.orientation) * 0.5 * grid.spacing_a
                        } else {
                            0.0
                        },
                    grid.origin_relative_angstrom
                        + face.voxel_xyz[1] as f64 * grid.spacing_a
                        + if face.axis == 1 {
                            f64::from(face.orientation) * 0.5 * grid.spacing_a
                        } else {
                            0.0
                        },
                    grid.origin_relative_angstrom
                        + face.voxel_xyz[2] as f64 * grid.spacing_a
                        + if face.axis == 2 {
                            f64::from(face.orientation) * 0.5 * grid.spacing_a
                        } else {
                            0.0
                        },
                ]
            || expected_boundaries.get(&(face.voxel_linear_index, face.axis, face.orientation))
                != Some(&face.boundary_kind)
            || !keys.insert((face.voxel_linear_index, face.axis, face.orientation))
        {
            bail!("physical lining boundary face identity or ordering is invalid");
        }
        kinds[match face.boundary_kind {
            PhysicalCavityBoundaryKind::ReceptorExcludedVolume => 0,
            PhysicalCavityBoundaryKind::EstimatorMask => 1,
            PhysicalCavityBoundaryKind::LocalWindow => 2,
        }] += 1;
        let entry = counts.entry(face.nearest_atom_index).or_insert_with(|| {
            (
                face.nearest_atom_identity_sha256.clone(),
                0,
                0,
                0,
                0,
                0.0,
                f64::INFINITY,
                f64::NEG_INFINITY,
            )
        });
        entry.1 = entry
            .1
            .checked_add(1)
            .context("physical lining verification face count overflow")?;
        match face.boundary_kind {
            PhysicalCavityBoundaryKind::ReceptorExcludedVolume => entry.2 += 1,
            PhysicalCavityBoundaryKind::EstimatorMask => entry.3 += 1,
            PhysicalCavityBoundaryKind::LocalWindow => entry.4 += 1,
        }
        entry.5 += face.face_area_a2;
        entry.6 = entry
            .6
            .min(face.nearest_solvent_excluded_surface_distance_a);
        entry.7 = entry
            .7
            .max(face.nearest_solvent_excluded_surface_distance_a);
    }
    let expected_attributions = counts
        .into_iter()
        .map(
            |(
                atom_index,
                (
                    atom_identity_sha256,
                    count,
                    receptor_count,
                    estimator_count,
                    window_count,
                    area,
                    min,
                    max,
                ),
            )| {
                PhysicalCavityGridAtomAttribution {
                    atom_identity_sha256,
                    atom_index,
                    boundary_face_count: count,
                    receptor_excluded_volume_face_count: receptor_count,
                    estimator_mask_face_count: estimator_count,
                    local_window_face_count: window_count,
                    boundary_area_a2: area,
                    minimum_surface_distance_a: min,
                    maximum_surface_distance_a: max,
                }
            },
        )
        .collect::<Vec<_>>();
    if kinds
        != [
            grid.receptor_excluded_volume_face_count,
            grid.estimator_mask_face_count,
            grid.local_window_face_count,
        ]
        || keys.len() != expected_boundaries.len()
        || expected_attributions != grid.per_atom_attributions
        || canonical_sha256(&grid.boundary_faces)? != grid.boundary_faces_sha256
        || physical_grid_record_sha256(grid)? != grid.grid_record_sha256
    {
        bail!("physical lining boundary-face attribution conservation failed");
    }
    Ok(())
}

fn grid_bit(bytes: &[u8], linear_index: u64) -> bool {
    bytes
        .get((linear_index / 8) as usize)
        .is_some_and(|byte| byte & (1 << (linear_index % 8)) != 0)
}

fn packed_grid_padding_is_zero(bytes: &[u8], voxel_count: u64) -> bool {
    let remainder = voxel_count % 8;
    remainder == 0
        || bytes.last().is_some_and(|last| {
            let used_mask = (1u16 << remainder) as u8 - 1;
            last & !used_mask == 0
        })
}

fn materialize_physical_lining(
    ghost_record_sha256: &str,
    coordinate_frame_sha256: &str,
    topology_sha256: &str,
    atom_residue_mapping_sha256: &str,
    analysis_residue_contract_sha256: &str,
    molecule_identity_table_sha256: &str,
    estimator_sha256: &str,
    metric: &PhysicalSpatialMetric,
    local_atoms: &[LocalReceptorAtom],
    grids: Vec<PhysicalCavityGridLiningEvidence>,
) -> Result<PhysicalCavityLiningEvidence> {
    for digest in [
        ghost_record_sha256,
        coordinate_frame_sha256,
        topology_sha256,
        atom_residue_mapping_sha256,
        analysis_residue_contract_sha256,
        molecule_identity_table_sha256,
        estimator_sha256,
    ] {
        require_sha256(digest, "physical cavity lining source SHA256")?;
    }
    let referenced = grids
        .iter()
        .flat_map(|grid| {
            grid.per_atom_attributions
                .iter()
                .map(|attribution| attribution.atom_index)
        })
        .collect::<BTreeSet<_>>();
    let identity_by_index = local_atoms
        .iter()
        .map(|atom| (atom.identity.atom_index, atom.identity.clone()))
        .collect::<BTreeMap<_, _>>();
    let atom_identities = referenced
        .iter()
        .map(|atom_index| {
            identity_by_index
                .get(atom_index)
                .cloned()
                .context("physical lining referenced atom is absent from local identity map")
        })
        .collect::<Result<Vec<_>>>()?;
    let total_voxels = grids.iter().try_fold(0u64, |total, grid| {
        total
            .checked_add(grid.selected_cavity_voxel_count)
            .context("physical lining multigrid voxel overflow")
    })?;
    let total_faces = grids.iter().try_fold(0u64, |total, grid| {
        total
            .checked_add(grid.boundary_face_count)
            .context("physical lining multigrid face overflow")
    })?;
    let total_atom_faces = grids
        .iter()
        .flat_map(|grid| &grid.per_atom_attributions)
        .try_fold(0u64, |total, attribution| {
            total
                .checked_add(attribution.boundary_face_count)
                .context("physical lining multigrid atom-face overflow")
        })?;
    let unique_residues = atom_identities
        .iter()
        .map(|atom| {
            (
                atom.chain_id.clone(),
                atom.analysis_residue_id,
                atom.pdb_residue_id,
            )
        })
        .collect::<BTreeSet<_>>()
        .len() as u64;
    let uncertainty = grids
        .iter()
        .map(|grid| {
            (
                grid.grid_ordinal,
                grid.spacing_a,
                grid.phase_offset_fraction,
                grid.selected_cavity_volume_a3,
                grid.boundary_area_a2,
                grid.grid_record_sha256.as_str(),
            )
        })
        .collect::<Vec<_>>();
    let observed = grids
        .iter()
        .filter(|grid| grid.selected_cavity_voxel_count > 0)
        .count() as u64;
    PhysicalCavityLiningEvidence {
        schema_kind: PHYSICAL_CAVITY_LINING_SCHEMA.to_string(),
        schema_version: PHYSICAL_CAVITY_LINING_VERSION.to_string(),
        evidence_sha256: String::new(),
        ghost_record_sha256: ghost_record_sha256.to_string(),
        coordinate_frame_sha256: coordinate_frame_sha256.to_string(),
        topology_sha256: topology_sha256.to_string(),
        atom_residue_mapping_sha256: atom_residue_mapping_sha256.to_string(),
        analysis_residue_contract_sha256: analysis_residue_contract_sha256.to_string(),
        molecule_identity_table_sha256: molecule_identity_table_sha256.to_string(),
        estimator_sha256: estimator_sha256.to_string(),
        spatial_metric: metric.authority.clone(),
        atom_identities,
        conservation: PhysicalCavityMultiGridConservation {
            expected_grid_count: grids.len() as u64,
            persisted_grid_count: grids.len() as u64,
            selected_cavity_voxel_count_from_grids: total_voxels,
            selected_cavity_voxel_count_conserved: total_voxels,
            boundary_face_count_from_grids: total_faces,
            boundary_face_count_from_records: total_faces,
            boundary_face_count_from_atom_attributions: total_atom_faces,
            unique_lining_atom_count: referenced.len() as u64,
            unique_lining_analysis_residue_count: unique_residues,
            grids_with_observed_cavity: observed,
            grids_with_observed_no_cavity: grids.len() as u64 - observed,
            uncertainty_input_sha256: canonical_sha256(&uncertainty)?,
            conservation_pass: total_faces == total_atom_faces,
        },
        grids,
        complete_without_truncation: true,
    }
    .materialize()
}

fn invert_3x3(matrix: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let determinant = matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
    if !determinant.is_finite() || determinant.abs() <= 1.0e-12 {
        return None;
    }
    Some([
        [
            (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) / determinant,
            (matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2]) / determinant,
            (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]) / determinant,
        ],
        [
            (matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2]) / determinant,
            (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]) / determinant,
            (matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2]) / determinant,
        ],
        [
            (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]) / determinant,
            (matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1]) / determinant,
            (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) / determinant,
        ],
    ])
}

fn minimum_image(delta: [f64; 3], box_vectors: [[f64; 3]; 3], inverse: [[f64; 3]; 3]) -> [f64; 3] {
    let mut fractional = mat_vec(inverse, delta);
    for value in &mut fractional {
        *value -= value.round();
    }
    mat_vec(box_vectors, fractional)
}

fn mat_vec(matrix: [[f64; 3]; 3], vector: [f64; 3]) -> [f64; 3] {
    [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    ]
}

fn sub(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn norm2(value: [f64; 3]) -> f64 {
    value.iter().map(|component| component * component).sum()
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv_update(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn read_u64(bytes: &[u8], cursor: &mut usize) -> Result<u64> {
    let end = cursor.checked_add(8).context("u64 cursor overflow")?;
    let value = u64::from_le_bytes(
        bytes
            .get(*cursor..end)
            .context("truncated u64")?
            .try_into()?,
    );
    *cursor = end;
    Ok(value)
}

fn read_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32> {
    let end = cursor.checked_add(4).context("u32 cursor overflow")?;
    let value = u32::from_le_bytes(
        bytes
            .get(*cursor..end)
            .context("truncated u32")?
            .try_into()?,
    );
    *cursor = end;
    Ok(value)
}

fn audit_u64(value: &serde_json::Value, key: &str) -> Result<u64> {
    value
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .with_context(|| format!("trajectory audit lacks {key}"))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lossless_v2_index_requires_exact_steps_and_verified_rolling_hash() {
        let temp = tempfile::tempdir().unwrap();
        let trajectory_path = temp.path().join("stream00.bin");
        let audit_path = temp.path().join("stream00.bin.audit.json");
        let frames = [
            (10u64, vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0]),
            (20u64, vec![6.0f32, 7.0, 8.0, 9.0, 10.0, 11.0]),
        ];
        let mut bytes = (frames.len() as u64).to_le_bytes().to_vec();
        let mut rolling = FNV_OFFSET;
        let mut frame_audit = Vec::new();
        for (ordinal, (step, positions)) in frames.iter().enumerate() {
            let start = bytes.len();
            bytes.extend_from_slice(&step.to_le_bytes());
            bytes.extend_from_slice(&(positions.len() as u32).to_le_bytes());
            for position in positions {
                bytes.extend_from_slice(&position.to_le_bytes());
            }
            let frame_hash = fnv_update(FNV_OFFSET, &bytes[start..]);
            rolling = fnv_update(rolling, &frame_hash.to_le_bytes());
            frame_audit.push(serde_json::json!({
                "frame_index": ordinal,
                "step": step,
                "n_floats": positions.len(),
                "producer_hash64": format!("{frame_hash:016x}"),
                "writer_hash64": format!("{frame_hash:016x}"),
            }));
        }
        std::fs::write(&trajectory_path, &bytes).unwrap();
        std::fs::write(
            &audit_path,
            serde_json::to_vec(&serde_json::json!({
                "stream_id": 0,
                "lossless_required": true,
                "producer_frames_enqueued": 2,
                "frames_written": 2,
                "bytes_written": bytes.len(),
                "hash_mismatches": 0,
                "expected_floats_per_frame": 6,
                "rolling_hash64": format!("{rolling:016x}"),
                "first_step": 10,
                "last_step": 20,
                "frames": frame_audit,
            }))
            .unwrap(),
        )
        .unwrap();
        let index = load_lossless_v2_trajectory(&trajectory_path, &audit_path, 0, 2).unwrap();
        assert!(index.frames.contains_key(&10));
        assert!(index.frames.contains_key(&20));
        assert!(!index.frames.contains_key(&15));
        let mut corrupted_audit: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&audit_path).unwrap()).unwrap();
        corrupted_audit["rolling_hash64"] = serde_json::json!("0000000000000000");
        std::fs::write(&audit_path, serde_json::to_vec(&corrupted_audit).unwrap()).unwrap();
        assert!(load_lossless_v2_trajectory(&trajectory_path, &audit_path, 0, 2).is_err());
    }

    #[test]
    fn multi_grid_estimator_reports_resolution_population() {
        let atoms = vec![
            ([-2.5, 0.0, 0.0], 1.8),
            ([2.5, 0.0, 0.0], 1.8),
            ([0.0, -2.5, 0.0], 1.8),
            ([0.0, 2.5, 0.0], 1.8),
        ];
        let config = CoordinateCavityConfig {
            grid_spacings_a: vec![1.0, 0.5],
            grid_phase_offsets: vec![0.0, 0.5],
            local_half_width_a: 4.0,
            solvent_probe_radius_a: 1.4,
            ray_horizon_a: 6.0,
            minimum_blocked_axial_directions: 3,
            require_opposed_wall_pair: true,
        };
        config.validate().unwrap();
        let estimates = config
            .grid_spacings_a
            .iter()
            .flat_map(|spacing| {
                config
                    .grid_phase_offsets
                    .iter()
                    .map(|phase| evaluate_grid(&atoms, *spacing, *phase, &config).unwrap())
            })
            .collect::<Vec<_>>();
        assert_eq!(estimates.len(), 4);
        assert!(estimates
            .iter()
            .all(|estimate| estimate.volume_a3.is_finite()));
    }

    fn test_lining_atom(
        atom_index: usize,
        position_relative_angstrom: [f64; 3],
        radius: f64,
    ) -> LocalReceptorAtom {
        LocalReceptorAtom {
            position_relative_angstrom,
            solvent_excluded_radius_angstrom: radius,
            identity: PhysicalCavityLiningAtomIdentity {
                atom_identity_sha256: String::new(),
                atom_index,
                atom_name: format!("C{atom_index}"),
                element: "C".to_string(),
                physical_residue_id: atom_index,
                analysis_residue_id: atom_index as u32,
                pdb_residue_id: 900 + atom_index as i64,
                residue_name: "ALA".to_string(),
                chain_id: if atom_index % 2 == 0 { "A" } else { "B" }.to_string(),
                molecule_id: (atom_index % 2) as u64,
                molecule_identity: if atom_index % 2 == 0 { "MYC" } else { "MAX" }.to_string(),
                molecule_component_sha256: format!("{:064x}", atom_index + 1),
            }
            .materialize()
            .unwrap(),
        }
    }

    fn test_nonperiodic_metric() -> PhysicalSpatialMetric {
        PhysicalSpatialMetric {
            authority: PhysicalCavitySpatialMetricAuthority {
                schema_kind: "prism_physical_cavity_spatial_metric_authority".to_string(),
                schema_version: "1.0.0".to_string(),
                authority_sha256: String::new(),
                periodic_cell_contract_sha256: "a".repeat(64),
                mode: PhysicalCavitySpatialMetricMode::DeclaredNonperiodicEuclidean,
                box_vectors_angstrom: None,
                nonperiodic_reason: Some("declared unit-test nonperiodic boundary".to_string()),
            }
            .materialize()
            .unwrap(),
            box_matrix: None,
            inverse_box: None,
        }
    }

    #[test]
    fn physical_lining_preserves_every_boundary_face_and_exact_atom_identity() {
        let atoms = vec![
            test_lining_atom(0, [-2.5, 0.0, 0.0], 1.8),
            test_lining_atom(1, [2.5, 0.0, 0.0], 1.8),
            test_lining_atom(2, [0.0, -2.5, 0.0], 1.8),
            test_lining_atom(3, [0.0, 2.5, 0.0], 1.8),
        ];
        let config = CoordinateCavityConfig {
            grid_spacings_a: vec![1.0, 0.5],
            grid_phase_offsets: vec![0.0, 0.5],
            local_half_width_a: 4.0,
            solvent_probe_radius_a: 1.4,
            ray_horizon_a: 6.0,
            minimum_blocked_axial_directions: 3,
            require_opposed_wall_pair: true,
        };
        let metric = test_nonperiodic_metric();
        let ghost = "b".repeat(64);
        let coordinate = "c".repeat(64);
        let topology = "d".repeat(64);
        let mapping = "e".repeat(64);
        let estimator = canonical_sha256(&config).unwrap();
        let source = PhysicalGridSourceBinding {
        region_frame_observation: false,
            ghost_record_sha256: &ghost,
            coordinate_frame_sha256: &coordinate,
            topology_sha256: &topology,
            atom_residue_mapping_sha256: &mapping,
            estimator_sha256: &estimator,
        };
        let evaluated =
            evaluate_grid_with_physical_lining(&atoms, 1.0, 0.0, 0, &config, &metric, &source)
                .unwrap();
        let grid = evaluated.lining;
        assert!(grid.selected_cavity_voxel_count > 0);
        assert!(grid.boundary_face_count > 0);
        assert_eq!(grid.boundary_face_count, grid.boundary_faces.len() as u64);
        assert_eq!(
            grid.boundary_face_count,
            grid.per_atom_attributions
                .iter()
                .map(|row| row.boundary_face_count)
                .sum::<u64>()
        );
        let exact_identities = atoms
            .iter()
            .map(|atom| {
                (
                    atom.identity.atom_identity_sha256.clone(),
                    atom.identity.atom_index,
                )
            })
            .collect::<BTreeMap<_, _>>();
        verify_physical_grid_lining(&grid, 0, &exact_identities).unwrap();
        assert!(grid.boundary_faces.iter().all(|face| {
            exact_identities.get(&face.nearest_atom_identity_sha256)
                == Some(&face.nearest_atom_index)
        }));
        let evidence = materialize_physical_lining(
            &ghost,
            &coordinate,
            &topology,
            &mapping,
            &"6".repeat(64),
            &"7".repeat(64),
            &estimator,
            &metric,
            &atoms,
            vec![grid.clone()],
        )
        .unwrap();
        evidence.verify().unwrap();
        assert_eq!(evidence.conservation.persisted_grid_count, 1);
        assert_eq!(
            evidence.conservation.boundary_face_count_from_grids,
            grid.boundary_face_count
        );
        assert!(evidence.complete_without_truncation);
        let mut rebound = evidence.clone();
        rebound.ghost_record_sha256 = "8".repeat(64);
        assert!(rebound.verify().is_err());
        let mut truncated = grid.clone();
        truncated.boundary_faces.pop();
        assert!(verify_physical_grid_lining(&truncated, 0, &exact_identities).is_err());
        let mut altered_bits = grid.clone();
        altered_bits.selected_component_grid_packed_lsb0[0] ^= 1;
        assert!(verify_physical_grid_lining(&altered_bits, 0, &exact_identities).is_err());
    }

    #[test]
    fn grid_identity_changes_with_ghost_or_coordinate_authority_not_geometry_bytes() {
        let atoms = vec![
            test_lining_atom(0, [-2.5, 0.0, 0.0], 1.8),
            test_lining_atom(1, [2.5, 0.0, 0.0], 1.8),
            test_lining_atom(2, [0.0, -2.5, 0.0], 1.8),
            test_lining_atom(3, [0.0, 2.5, 0.0], 1.8),
        ];
        let config = CoordinateCavityConfig::default();
        let metric = test_nonperiodic_metric();
        let coordinate = "c".repeat(64);
        let topology = "d".repeat(64);
        let mapping = "e".repeat(64);
        let estimator = canonical_sha256(&config).unwrap();
        let run = |ghost: &str| {
            evaluate_grid_with_physical_lining(
                &atoms,
                0.75,
                0.0,
                0,
                &config,
                &metric,
                &PhysicalGridSourceBinding {
        region_frame_observation: false,
                    ghost_record_sha256: ghost,
                    coordinate_frame_sha256: &coordinate,
                    topology_sha256: &topology,
                    atom_residue_mapping_sha256: &mapping,
                    estimator_sha256: &estimator,
                },
            )
            .unwrap()
            .lining
        };
        let first = run(&"1".repeat(64));
        let repeat = run(&"1".repeat(64));
        let changed = run(&"2".repeat(64));
        assert_eq!(first, repeat);
        assert_eq!(
            first.selected_component_grid_sha256,
            changed.selected_component_grid_sha256
        );
        assert_ne!(first.grid_source_sha256, changed.grid_source_sha256);
        assert_ne!(first.grid_record_sha256, changed.grid_record_sha256);
        assert_ne!(
            first.boundary_faces[0].face_record_sha256,
            changed.boundary_faces[0].face_record_sha256
        );
    }

    #[test]
    fn triclinic_minimum_image_differs_from_fractional_euclidean_distance() {
        let row_vectors = [[10.0, 0.0, 0.0], [2.0, 10.0, 0.0], [0.0, 0.0, 10.0]];
        let matrix = row_vectors_to_matrix(row_vectors);
        let metric = PhysicalSpatialMetric {
            authority: PhysicalCavitySpatialMetricAuthority {
                schema_kind: "prism_physical_cavity_spatial_metric_authority".to_string(),
                schema_version: "1.0.0".to_string(),
                authority_sha256: String::new(),
                periodic_cell_contract_sha256: "f".repeat(64),
                mode: PhysicalCavitySpatialMetricMode::PeriodicTriclinicMinimumImage,
                box_vectors_angstrom: Some(row_vectors),
                nonperiodic_reason: None,
            }
            .materialize()
            .unwrap(),
            box_matrix: Some(matrix),
            inverse_box: Some(invert_3x3(matrix).unwrap()),
        };
        let left = mat_vec(matrix, [0.49, 0.49, 0.0]);
        let right = mat_vec(matrix, [-0.49, -0.49, 0.0]);
        let delta = metric.displacement(left, right);
        assert!((delta[0] + 0.24).abs() < 1.0e-12);
        assert!((delta[1] + 0.20).abs() < 1.0e-12);
        assert!(norm2(delta).sqrt() < 0.32);
        assert!(norm2(sub(left, right)).sqrt() > 10.0);
    }

    #[test]
    fn observed_empty_grid_is_complete_but_unbounded_grid_fails_before_allocation() {
        let atoms = vec![test_lining_atom(0, [0.0, 0.0, 0.0], 100.0)];
        let config = CoordinateCavityConfig::default();
        let metric = test_nonperiodic_metric();
        let digest = "9".repeat(64);
        let source = PhysicalGridSourceBinding {
        region_frame_observation: false,
            ghost_record_sha256: &digest,
            coordinate_frame_sha256: &digest,
            topology_sha256: &digest,
            atom_residue_mapping_sha256: &digest,
            estimator_sha256: &digest,
        };
        let empty =
            evaluate_grid_with_physical_lining(&atoms, 0.75, 0.0, 0, &config, &metric, &source)
                .unwrap()
                .lining;
        assert_eq!(empty.selected_cavity_voxel_count, 0);
        assert!(empty.boundary_faces.is_empty());
        assert!(empty.per_atom_attributions.is_empty());
        let atom_hashes = atoms
            .iter()
            .map(|atom| {
                (
                    atom.identity.atom_identity_sha256.clone(),
                    atom.identity.atom_index,
                )
            })
            .collect::<BTreeMap<_, _>>();
        verify_physical_grid_lining(&empty, 0, &atom_hashes).unwrap();

        let tiny_spacing = f64::MIN_POSITIVE;
        assert!(evaluate_grid_with_physical_lining(
            &atoms,
            tiny_spacing,
            0.0,
            0,
            &config,
            &metric,
            &source,
        )
        .is_err());
    }

    #[test]
    fn spatial_metric_authority_rejects_undeclared_boundary_modes() {
        let missing_vectors = PhysicalCavitySpatialMetricAuthority {
            schema_kind: "prism_physical_cavity_spatial_metric_authority".to_string(),
            schema_version: "1.0.0".to_string(),
            authority_sha256: "1".repeat(64),
            periodic_cell_contract_sha256: "2".repeat(64),
            mode: PhysicalCavitySpatialMetricMode::PeriodicTriclinicMinimumImage,
            box_vectors_angstrom: None,
            nonperiodic_reason: None,
        };
        assert!(missing_vectors.verify().is_err());
        let undeclared_nonperiodic = PhysicalCavitySpatialMetricAuthority {
            schema_kind: "prism_physical_cavity_spatial_metric_authority".to_string(),
            schema_version: "1.0.0".to_string(),
            authority_sha256: "1".repeat(64),
            periodic_cell_contract_sha256: "2".repeat(64),
            mode: PhysicalCavitySpatialMetricMode::DeclaredNonperiodicEuclidean,
            box_vectors_angstrom: None,
            nonperiodic_reason: None,
        };
        assert!(undeclared_nonperiodic.verify().is_err());
    }
}
