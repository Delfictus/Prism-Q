//! Deterministic cross-run Ghost-component correspondence.
//!
//! Raw `site_id` and component ordinals are deliberately absent. Identity is
//! inferred from signed chain/canonical-residue contact distributions and a
//! periodic-aware spatial distribution. Every input is emitted exactly once
//! as recurrent, ambiguous, or unmatched evidence.

use crate::chemical_state_fiber::{canonical_sha256, require_sha256, ControlLane};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const PHYSICAL_LINING_OCCURRENCE_SCHEMA: &str = "prism_physical_cavity_lining_occurrence_evidence";
const PHYSICAL_LINING_OCCURRENCE_VERSION: &str = "2.0.0";

/// Explicit classification authority for multi-grid cavity observations.
/// Neither threshold is supplied by runtime defaults.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CavityBinaryLabelAuthority {
    pub schema_kind: String,
    pub schema_version: String,
    pub authority_sha256: String,
    pub source_design_authority_sha256: String,
    pub closed_max_volume_a3: f64,
    pub open_min_volume_a3: f64,
}

impl CavityBinaryLabelAuthority {
    fn body_sha256(&self) -> Result<String> {
        let mut body = self.clone();
        body.authority_sha256.clear();
        canonical_sha256(&body)
    }

    pub fn materialize(mut self) -> Result<Self> {
        if !self.authority_sha256.is_empty() {
            bail!("cavity label authority materialization requires empty digest");
        }
        self.authority_sha256 = self.body_sha256()?;
        self.verify()?;
        Ok(self)
    }

    pub fn verify(&self) -> Result<()> {
        if self.schema_kind != "prism_cavity_binary_label_authority"
            || self.schema_version != "1.0.0"
            || !self.closed_max_volume_a3.is_finite()
            || !self.open_min_volume_a3.is_finite()
            || self.closed_max_volume_a3 < 0.0
            || self.open_min_volume_a3 <= self.closed_max_volume_a3
        {
            bail!("invalid cavity binary-label authority");
        }
        require_sha256(&self.authority_sha256, "cavity label authority")?;
        require_sha256(
            &self.source_design_authority_sha256,
            "cavity label source design authority",
        )?;
        if self.body_sha256()? != self.authority_sha256 {
            bail!("cavity label authority hash mismatch");
        }
        Ok(())
    }
}

/// Binary interpretation of a continuous multi-grid observation. Grid
/// disagreement is never collapsed through an average.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CavityBinaryLabel {
    Closed,
    Open,
    IndeterminateGridThresholdStraddle,
    IndeterminateOpenClosedBand,
}

/// Complete multi-grid continuous cavity measurement plus its conservative
/// threshold interpretation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MultiGridCavityMeasurement {
    pub estimator_sha256: String,
    pub label_authority_sha256: String,
    pub grid_volume_a3: Vec<f64>,
    pub mean_volume_a3: f64,
    pub discretization_standard_error_a3: f64,
    pub discretization_min_a3: f64,
    pub discretization_max_a3: f64,
    pub binary_label: CavityBinaryLabel,
}

impl MultiGridCavityMeasurement {
    pub fn materialize(
        estimator_sha256: String,
        grid_volume_a3: Vec<f64>,
        authority: &CavityBinaryLabelAuthority,
    ) -> Result<Self> {
        authority.verify()?;
        require_sha256(&estimator_sha256, "multi-grid cavity estimator")?;
        if grid_volume_a3.len() < 2
            || grid_volume_a3
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            bail!("multi-grid cavity population is incomplete or invalid");
        }
        let mean = grid_volume_a3.iter().sum::<f64>() / grid_volume_a3.len() as f64;
        let variance = grid_volume_a3
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / (grid_volume_a3.len() - 1) as f64;
        let standard_error = (variance / grid_volume_a3.len() as f64).sqrt();
        let min = grid_volume_a3.iter().copied().fold(f64::INFINITY, f64::min);
        let max = grid_volume_a3
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let per_grid = grid_volume_a3
            .iter()
            .map(|volume| {
                if *volume <= authority.closed_max_volume_a3 {
                    0u8
                } else if *volume >= authority.open_min_volume_a3 {
                    2u8
                } else {
                    1u8
                }
            })
            .collect::<BTreeSet<_>>();
        let binary_label = if per_grid == BTreeSet::from([0]) {
            CavityBinaryLabel::Closed
        } else if per_grid == BTreeSet::from([2]) {
            CavityBinaryLabel::Open
        } else if per_grid == BTreeSet::from([1]) {
            CavityBinaryLabel::IndeterminateOpenClosedBand
        } else {
            CavityBinaryLabel::IndeterminateGridThresholdStraddle
        };
        Ok(Self {
            estimator_sha256,
            label_authority_sha256: authority.authority_sha256.clone(),
            grid_volume_a3,
            mean_volume_a3: mean,
            discretization_standard_error_a3: standard_error,
            discretization_min_a3: min,
            discretization_max_a3: max,
            binary_label,
        })
    }

    pub fn verify(&self, authority: &CavityBinaryLabelAuthority) -> Result<()> {
        let rebuilt = Self::materialize(
            self.estimator_sha256.clone(),
            self.grid_volume_a3.clone(),
            authority,
        )?;
        if &rebuilt != self {
            bail!("multi-grid cavity measurement is not content exact");
        }
        Ok(())
    }
}

/// Exact occurrence binding around the coordinate-cavity producer's complete
/// physical-lining bytes. The producer evidence is embedded without
/// translation; this wrapper adds only the globally unique Ghost occurrence,
/// lattice index, and sidecar identity. Binary open/closed thresholds are a
/// confirmation-protocol authority and are deliberately absent from unbiased
/// whole-lattice discovery evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhysicalLiningOccurrenceEvidence {
    pub schema_kind: String,
    pub schema_version: String,
    pub occurrence_sha256: String,
    pub ghost_lineage_sha256: String,
    pub lattice_node_index: u64,
    pub cavity_sidecar_sha256: String,
    pub physical_lining: crate::coordinate_cavity::PhysicalCavityLiningEvidence,
}

impl PhysicalLiningOccurrenceEvidence {
    fn body_sha256(&self) -> Result<String> {
        let mut body = self.clone();
        body.occurrence_sha256.clear();
        canonical_sha256(&body)
    }

    pub fn materialize(
        lineage: &crate::chemical_state_fiber::GhostByteLineage,
        cavity_sidecar_sha256: String,
        physical_lining: crate::coordinate_cavity::PhysicalCavityLiningEvidence,
    ) -> Result<Self> {
        lineage.validate()?;
        physical_lining.verify()?;
        require_sha256(&cavity_sidecar_sha256, "physical lining cavity sidecar")?;
        if physical_lining.ghost_record_sha256 != lineage.record_sha256 {
            bail!("coordinate-cavity lining bytes differ from Ghost record lineage");
        }
        let mut occurrence = Self {
            schema_kind: PHYSICAL_LINING_OCCURRENCE_SCHEMA.to_string(),
            schema_version: PHYSICAL_LINING_OCCURRENCE_VERSION.to_string(),
            occurrence_sha256: String::new(),
            ghost_lineage_sha256: lineage.lineage_sha256()?,
            lattice_node_index: lineage.lattice_node_index,
            cavity_sidecar_sha256,
            physical_lining,
        };
        occurrence.occurrence_sha256 = occurrence.body_sha256()?;
        occurrence.verify()?;
        Ok(occurrence)
    }

    pub fn verify(&self) -> Result<()> {
        if self.schema_kind != PHYSICAL_LINING_OCCURRENCE_SCHEMA
            || self.schema_version != PHYSICAL_LINING_OCCURRENCE_VERSION
        {
            bail!("invalid physical cavity-lining occurrence schema");
        }
        for (value, field) in [
            (&self.occurrence_sha256, "physical cavity-lining occurrence"),
            (&self.ghost_lineage_sha256, "physical lining Ghost lineage"),
            (
                &self.cavity_sidecar_sha256,
                "physical lining cavity sidecar",
            ),
        ] {
            require_sha256(value, field)?;
        }
        self.physical_lining.verify()?;
        if self.body_sha256()? != self.occurrence_sha256 {
            bail!("physical cavity-lining occurrence hash mismatch");
        }
        Ok(())
    }
}

/* Retired v1 residue-summary evidence deliberately has no compatibility
 * adapter. Production correspondence consumes exact coordinate-cavity atom
 * and boundary-face bytes through `PhysicalLiningOccurrenceEvidence`. */
/*
impl PhysicalLiningOccurrenceEvidence {
    pub fn _retired_v1_marker(&self) {
        let _ = self;
    }
}
*/

/// Construct the complete descriptor population for one exact native-lattice
/// projection. Residue contacts come only from authenticated RichSpike byte
/// lineage; component geometry comes only from the corresponding native
/// lattice nodes. Raw `site_id` and component ordinals are never identities.
#[cfg(feature = "gpu")]
pub fn build_native_component_descriptors(
    topology: &crate::input::PrismPrepTopology,
    full_nodes: &[crate::ghost_phase_lattice::GhostPhaseLatticeNode],
    full_record_lineage: &[crate::ghost_phase_lattice::GhostRecordByteLineage],
    selected_lineages: &[crate::chemical_state_fiber::GhostByteLineage],
    connectivity: &crate::pathb_chemical_state::NativePathBConnectivityAuthority,
    control_lane: ControlLane,
) -> Result<Vec<NativeComponentDescriptor>> {
    connectivity.verify_for(selected_lineages)?;
    if full_nodes.len() != full_record_lineage.len() {
        bail!("native descriptor source node/byte-lineage populations differ");
    }
    let analysis = topology
        .analysis_residue_contract
        .as_ref()
        .context("native component descriptors require signed analysis-residue mapping")?;
    require_sha256(&analysis.contract_sha256, "analysis residue contract")?;
    let canonical_residues = analysis
        .analysis_residues
        .iter()
        .map(|row| {
            (
                u32::try_from(row.analysis_residue_id),
                (
                    row.chain_id.clone(),
                    format!("{}:{}", row.pdb_residue_id, row.residue_name),
                ),
            )
        })
        .map(|(id, value)| Ok((id.context("analysis residue id exceeds u32")?, value)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    if canonical_residues.len() != analysis.analysis_residues.len() {
        bail!("signed analysis-residue mapping contains duplicate ids");
    }
    let selected = selected_lineages
        .iter()
        .map(|lineage| {
            let hash = lineage.lineage_sha256()?;
            let index = usize::try_from(lineage.lattice_node_index)
                .context("lattice node index exceeds host width")?;
            let node = full_nodes
                .get(index)
                .context("selected lineage points outside full lattice")?;
            let record = full_record_lineage
                .get(index)
                .context("selected lineage lacks byte-lineage row")?;
            if node.stream_id != lineage.key.stream_id
                || node.step_idx != lineage.key.step_idx
                || node.site_id != lineage.key.site_id
                || record.record_sha256 != lineage.record_sha256
            {
                bail!("selected component descriptor lineage differs from full lattice");
            }
            Ok((hash, (lineage, node, record)))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    if selected.len() != selected_lineages.len() {
        bail!("selected component descriptor lineage contains duplicate identities");
    }
    let (coordinate_mode, box_vectors, inverse_box, box_sha256) = if let Some(vectors) = topology
        .periodic_cell
        .as_ref()
        .filter(|cell| cell.periodic)
        .and_then(|cell| cell.vectors_angstrom)
    {
        let matrix = [
            [vectors[0][0], vectors[1][0], vectors[2][0]],
            [vectors[0][1], vectors[1][1], vectors[2][1]],
            [vectors[0][2], vectors[1][2], vectors[2][2]],
        ];
        (
            SpatialCoordinateMode::FractionalPeriodic,
            Some(matrix),
            Some(invert_3x3(matrix).context("component periodic box is singular")?),
            Some(canonical_sha256(&vectors)?),
        )
    } else {
        (SpatialCoordinateMode::CartesianAngstrom, None, None, None)
    };
    let mut descriptors = Vec::with_capacity(connectivity.components.len());
    for component in &connectivity.components {
        let mut weighted_contacts: BTreeMap<(String, String), u64> = BTreeMap::new();
        let mut total_contact_weight = 0u64;
        let mut positions = Vec::with_capacity(component.node_ghost_lineage_sha256.len());
        let mut run_units = BTreeSet::new();
        for node_hash in &component.node_ghost_lineage_sha256 {
            let (lineage, node, record) = selected
                .get(node_hash)
                .context("exact component names an absent selected lineage")?;
            run_units.insert((
                lineage.run_id.clone(),
                lineage.replica_id.clone(),
                lineage.replica_seed,
            ));
            let cartesian = [
                f64::from(node.centroid_xyz[0]),
                f64::from(node.centroid_xyz[1]),
                f64::from(node.centroid_xyz[2]),
            ];
            positions.push(match inverse_box {
                Some(inverse) => {
                    let fractional = mat_vec(inverse, cartesian);
                    fractional.map(|value| value.rem_euclid(1.0))
                }
                None => cartesian,
            });
            for contact in &record
                .cluster_residue_membership
                .cluster_residue_spike_counts
            {
                let (chain, canonical) =
                    canonical_residues
                        .get(&contact.residue_id)
                        .with_context(|| {
                            format!(
                                "Ghost residue {} is absent from signed analysis mapping",
                                contact.residue_id
                            )
                        })?;
                let weight = u64::from(contact.spike_count);
                *weighted_contacts
                    .entry((chain.clone(), canonical.clone()))
                    .or_default() += weight;
                total_contact_weight = total_contact_weight
                    .checked_add(weight)
                    .context("component contact-weight overflow")?;
            }
        }
        if run_units.len() != 1 || total_contact_weight == 0 {
            bail!("one native component descriptor must have one run unit and contact mass");
        }
        let (run_id, replica_id, replica_seed) = run_units.into_iter().next().unwrap();
        let signed_contact_distribution = weighted_contacts
            .into_iter()
            .map(
                |((chain_id, canonical_residue_id), weight)| CanonicalResidueContactProbability {
                    chain_id,
                    canonical_residue_id,
                    chain_mapping_receipt_sha256: analysis.contract_sha256.clone(),
                    probability: weight as f64 / total_contact_weight as f64,
                },
            )
            .collect::<Vec<_>>();
        let spatial_distribution =
            spatial_distribution(coordinate_mode, &positions, box_vectors, box_sha256.clone())?;
        descriptors.push(
            NativeComponentDescriptor {
                descriptor_sha256: String::new(),
                source_component_sha256: component.component_sha256.clone(),
                run_id,
                replica_id,
                replica_seed,
                control_lane,
                signed_contact_distribution,
                spatial_distribution,
            }
            .materialize()?,
        );
    }
    descriptors.sort_by(|left, right| left.descriptor_sha256.cmp(&right.descriptor_sha256));
    Ok(descriptors)
}

/// Provenance binding between one cross-run descriptor and the complete
/// coordinate/grid lining evidence used to construct its residue mass.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhysicalLiningComponentBinding {
    pub source_component_sha256: String,
    pub descriptor_sha256: String,
    pub physical_lining_evidence_sha256: Vec<String>,
    pub physical_lining_population_sha256: String,
}

/// Complete physically lined descriptor population. Consumers must retain
/// this authority alongside `descriptors`; extracting descriptors and
/// discarding bindings loses the evidence class.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhysicalLiningDescriptorPopulation {
    pub schema_kind: String,
    pub schema_version: String,
    pub population_sha256: String,
    pub spatial_boundary_contract_sha256: String,
    pub physical_lining_evidence: Vec<PhysicalLiningOccurrenceEvidence>,
    pub descriptors: Vec<NativeComponentDescriptor>,
    pub bindings: Vec<PhysicalLiningComponentBinding>,
}

impl PhysicalLiningDescriptorPopulation {
    fn body_sha256(&self) -> Result<String> {
        let mut body = self.clone();
        body.population_sha256.clear();
        canonical_sha256(&body)
    }

    pub fn verify(&self) -> Result<()> {
        if self.schema_kind != "prism_physical_lining_descriptor_population"
            || self.schema_version != "2.0.0"
            || self.descriptors.len() != self.bindings.len()
        {
            bail!("invalid physical-lining descriptor population");
        }
        for (value, field) in [
            (
                &self.population_sha256,
                "physical-lining descriptor population",
            ),
            (
                &self.spatial_boundary_contract_sha256,
                "physical-lining spatial boundary contract",
            ),
        ] {
            require_sha256(value, field)?;
        }
        if self
            .physical_lining_evidence
            .windows(2)
            .any(|pair| pair[0].ghost_lineage_sha256 >= pair[1].ghost_lineage_sha256)
            || self
                .descriptors
                .windows(2)
                .any(|pair| pair[0].descriptor_sha256 >= pair[1].descriptor_sha256)
        {
            bail!("physical-lining population is duplicate or noncanonical order");
        }
        let evidence_hashes = self
            .physical_lining_evidence
            .iter()
            .map(|row| {
                row.verify()?;
                if row
                    .physical_lining
                    .spatial_metric
                    .periodic_cell_contract_sha256
                    != self.spatial_boundary_contract_sha256
                {
                    bail!("physical-lining evidence boundary mismatch");
                }
                Ok(row.occurrence_sha256.clone())
            })
            .collect::<Result<BTreeSet<_>>>()?;
        if evidence_hashes.len() != self.physical_lining_evidence.len() {
            bail!("physical-lining population contains duplicate evidence");
        }
        let by_source = self
            .descriptors
            .iter()
            .map(|descriptor| {
                descriptor.verify()?;
                Ok((
                    descriptor.source_component_sha256.as_str(),
                    descriptor.descriptor_sha256.as_str(),
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        if by_source.len() != self.descriptors.len() {
            bail!("physical-lining descriptors contain duplicate source components");
        }
        let mut previous = None;
        let mut bound_evidence = BTreeSet::new();
        for binding in &self.bindings {
            if previous
                .as_ref()
                .is_some_and(|prior: &String| prior >= &binding.source_component_sha256)
                || by_source
                    .get(binding.source_component_sha256.as_str())
                    .copied()
                    != Some(binding.descriptor_sha256.as_str())
                || binding.physical_lining_evidence_sha256.is_empty()
                || binding
                    .physical_lining_evidence_sha256
                    .windows(2)
                    .any(|pair| pair[0] >= pair[1])
            {
                bail!("physical-lining component binding is incomplete or unsorted");
            }
            require_sha256(
                &binding.physical_lining_population_sha256,
                "component physical-lining population",
            )?;
            for hash in &binding.physical_lining_evidence_sha256 {
                require_sha256(hash, "component physical-lining evidence")?;
                if !bound_evidence.insert(hash.clone()) {
                    bail!("physical-lining evidence is bound to multiple components");
                }
            }
            if canonical_sha256(&binding.physical_lining_evidence_sha256)?
                != binding.physical_lining_population_sha256
            {
                bail!("component physical-lining population hash mismatch");
            }
            previous = Some(binding.source_component_sha256.clone());
        }
        if bound_evidence != evidence_hashes {
            bail!("physical-lining component bindings do not conserve evidence");
        }
        if self.body_sha256()? != self.population_sha256 {
            bail!("physical-lining descriptor population hash mismatch");
        }
        Ok(())
    }
}

/// Construct component descriptors from exact physical cavity/grid lining,
/// never from RichSpike cluster membership. The legacy builder remains
/// available solely for selection evidence; production correspondence should
/// consume this population and preserve its bindings.
#[cfg(any(feature = "gpu", feature = "cpu-analysis"))]
pub fn build_native_component_descriptors_with_physical_lining(
    topology: &crate::input::PrismPrepTopology,
    full_nodes: &[crate::ghost_phase_lattice::GhostPhaseLatticeNode],
    full_record_lineage: &[crate::ghost_phase_lattice::GhostRecordByteLineage],
    selected_lineages: &[crate::chemical_state_fiber::GhostByteLineage],
    connectivity: &crate::pathb_chemical_state::NativePathBConnectivityAuthority,
    control_lane: ControlLane,
    physical_lining: Vec<PhysicalLiningOccurrenceEvidence>,
) -> Result<PhysicalLiningDescriptorPopulation> {
    connectivity.verify_for(selected_lineages)?;
    if full_nodes.len() != full_record_lineage.len() {
        bail!("physical-lining full node/record populations differ");
    }
    let periodic = topology
        .periodic_cell
        .as_ref()
        .context("physical-lining correspondence requires spatial boundary contract")?;
    let boundary_sha256 = periodic
        .contract_sha256
        .as_ref()
        .context("physical-lining spatial boundary is not hash-bound")?;
    if periodic.periodic {
        if periodic.vectors_angstrom.is_none() {
            bail!("periodic physical-lining boundary lacks triclinic vectors");
        }
    } else if periodic.vectors_angstrom.is_some()
        || periodic.reason.as_deref().is_none_or(str::is_empty)
    {
        bail!("nonperiodic Euclidean correspondence lacks explicit declaration");
    }
    let analysis = topology
        .analysis_residue_contract
        .as_ref()
        .context("physical-lining descriptors require signed analysis mapping")?;
    let canonical_residues = analysis
        .analysis_residues
        .iter()
        .map(|row| {
            (
                u32::try_from(row.analysis_residue_id),
                (
                    row.chain_id.as_str(),
                    format!("{}:{}", row.pdb_residue_id, row.residue_name),
                ),
            )
        })
        .map(|(id, value)| Ok((id.context("analysis residue id exceeds u32")?, value)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    if canonical_residues.len() != analysis.analysis_residues.len() {
        bail!("signed analysis-residue mapping contains duplicate ids");
    }
    let molecule_table_sha256 = &topology
        .molecule_identity_table
        .as_ref()
        .context("physical-lining descriptors require molecule identity table")?
        .table_sha256;
    require_sha256(
        molecule_table_sha256,
        "physical-lining molecule identity table",
    )?;
    let selected = selected_lineages
        .iter()
        .map(|lineage| Ok((lineage.lineage_sha256()?, lineage)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    if selected.len() != selected_lineages.len() {
        bail!("physical-lining selected lineage contains duplicate identities");
    }
    let evidence = physical_lining
        .into_iter()
        .map(|row| {
            row.verify()?;
            if row
                .physical_lining
                .spatial_metric
                .periodic_cell_contract_sha256
                != *boundary_sha256
                || row.physical_lining.analysis_residue_contract_sha256 != analysis.contract_sha256
                || row.physical_lining.molecule_identity_table_sha256 != *molecule_table_sha256
            {
                bail!("physical-lining evidence differs from topology boundary");
            }
            let lineage = selected
                .get(&row.ghost_lineage_sha256)
                .context("physical-lining evidence names an unselected Ghost lineage")?;
            let node_index = usize::try_from(row.lattice_node_index)?;
            let node = full_nodes
                .get(node_index)
                .context("physical-lining occurrence points outside native lattice")?;
            let record = full_record_lineage
                .get(node_index)
                .context("physical-lining occurrence lacks Ghost record lineage")?;
            if row.physical_lining.ghost_record_sha256 != lineage.record_sha256
                || row.lattice_node_index != lineage.lattice_node_index
                || row.physical_lining.topology_sha256 != lineage.topology_sha256
                || record.record_sha256 != lineage.record_sha256
                || node.stream_id != lineage.key.stream_id
                || node.step_idx != lineage.key.step_idx
                || node.site_id != lineage.key.site_id
            {
                bail!("physical-lining evidence differs from Ghost byte lineage");
            }
            for atom in &row.physical_lining.atom_identities {
                let (chain, canonical) = canonical_residues
                    .get(&atom.analysis_residue_id)
                    .context("physical-lining residue is absent from signed mapping")?;
                if atom.chain_id != *chain
                    || format!("{}:{}", atom.pdb_residue_id, atom.residue_name) != *canonical
                    || atom.atom_index >= topology.n_atoms
                    || topology
                        .atom_analysis_residue_ids
                        .as_ref()
                        .and_then(|ids| ids.get(atom.atom_index))
                        .and_then(|id| u32::try_from(*id).ok())
                        != Some(atom.analysis_residue_id)
                {
                    bail!("physical-lining atom does not match signed topology identity");
                }
            }
            Ok((row.ghost_lineage_sha256.clone(), row))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    if evidence.len() != selected.len()
        || evidence.keys().collect::<BTreeSet<_>>() != selected.keys().collect::<BTreeSet<_>>()
    {
        bail!("physical-lining evidence does not exactly cover selected Ghost lineage");
    }
    let (coordinate_mode, box_vectors, inverse_box, box_vectors_sha256) =
        if let Some(vectors) = periodic.vectors_angstrom.filter(|_| periodic.periodic) {
            let matrix = [
                [vectors[0][0], vectors[1][0], vectors[2][0]],
                [vectors[0][1], vectors[1][1], vectors[2][1]],
                [vectors[0][2], vectors[1][2], vectors[2][2]],
            ];
            (
                SpatialCoordinateMode::FractionalPeriodic,
                Some(matrix),
                Some(invert_3x3(matrix).context("component periodic box is singular")?),
                Some(canonical_sha256(&vectors)?),
            )
        } else {
            (SpatialCoordinateMode::CartesianAngstrom, None, None, None)
        };
    let mut descriptors = Vec::with_capacity(connectivity.components.len());
    for component in &connectivity.components {
        let mut run_units = BTreeSet::new();
        let mut positions = Vec::with_capacity(component.node_ghost_lineage_sha256.len());
        for lineage_sha256 in &component.node_ghost_lineage_sha256 {
            let lineage = selected
                .get(lineage_sha256)
                .context("component names an absent selected Ghost lineage")?;
            let node = full_nodes
                .get(usize::try_from(lineage.lattice_node_index)?)
                .context("component Ghost lineage points outside full lattice")?;
            run_units.insert((
                lineage.run_id.clone(),
                lineage.replica_id.clone(),
                lineage.replica_seed,
            ));
            let cartesian = [
                f64::from(node.centroid_xyz[0]),
                f64::from(node.centroid_xyz[1]),
                f64::from(node.centroid_xyz[2]),
            ];
            positions.push(match inverse_box {
                Some(inverse) => mat_vec(inverse, cartesian).map(|value| value.rem_euclid(1.0)),
                None => cartesian,
            });
        }
        if run_units.len() != 1 {
            bail!("one physical-lining component descriptor must have one run unit");
        }
        let (run_id, replica_id, replica_seed) = run_units.into_iter().next().unwrap();
        descriptors.push(NativeComponentDescriptor {
            descriptor_sha256: String::new(),
            source_component_sha256: component.component_sha256.clone(),
            run_id,
            replica_id,
            replica_seed,
            control_lane,
            signed_contact_distribution: Vec::new(),
            spatial_distribution: spatial_distribution(
                coordinate_mode,
                &positions,
                box_vectors,
                box_vectors_sha256.clone(),
            )?,
        });
    }
    let descriptor_indices = descriptors
        .iter()
        .enumerate()
        .map(|(index, descriptor)| (descriptor.source_component_sha256.clone(), index))
        .collect::<BTreeMap<_, _>>();
    if descriptor_indices.len() != descriptors.len() {
        bail!("native components contain duplicate source identities");
    }
    let mut bindings = Vec::with_capacity(connectivity.components.len());
    for component in &connectivity.components {
        let mut residue_weight = BTreeMap::<(String, String), f64>::new();
        let mut evidence_sha256 = Vec::new();
        for lineage_sha256 in &component.node_ghost_lineage_sha256 {
            let row = evidence
                .get(lineage_sha256)
                .context("component lacks physical cavity-lining evidence")?;
            evidence_sha256.push(row.occurrence_sha256.clone());
            let atom_by_identity = row
                .physical_lining
                .atom_identities
                .iter()
                .map(|atom| (atom.atom_identity_sha256.as_str(), atom))
                .collect::<BTreeMap<_, _>>();
            if atom_by_identity.len() != row.physical_lining.atom_identities.len() {
                bail!("physical-lining atom identities are not unique");
            }
            let grid_count = row.physical_lining.grids.len();
            if grid_count == 0 {
                bail!("physical-lining occurrence has no grid evidence");
            }
            let mut occurrence_residue_area = BTreeMap::<(String, String), f64>::new();
            for grid in &row.physical_lining.grids {
                for attribution in &grid.per_atom_attributions {
                    let atom = atom_by_identity
                        .get(attribution.atom_identity_sha256.as_str())
                        .context("grid attribution lacks exact atom identity bytes")?;
                    let (chain, canonical) = canonical_residues
                        .get(&atom.analysis_residue_id)
                        .context("grid attribution atom lacks signed analysis residue")?;
                    if attribution.atom_index != atom.atom_index || atom.chain_id != *chain {
                        bail!("grid attribution changes atom/chain identity");
                    }
                    *occurrence_residue_area
                        .entry(((*chain).to_string(), canonical.clone()))
                        .or_default() += attribution.boundary_area_a2;
                }
            }
            for (residue, area_sum) in occurrence_residue_area {
                *residue_weight.entry(residue).or_default() += area_sum / grid_count as f64;
            }
        }
        evidence_sha256.sort();
        if evidence_sha256.windows(2).any(|pair| pair[0] == pair[1]) {
            bail!("component repeats physical cavity-lining evidence");
        }
        let total = residue_weight.values().sum::<f64>();
        if !total.is_finite() || total <= 0.0 {
            bail!("component physical cavity-lining mass is empty");
        }
        let index = *descriptor_indices
            .get(&component.component_sha256)
            .context("component descriptor is absent")?;
        let source_component_sha256 = descriptors[index].source_component_sha256.clone();
        descriptors[index].descriptor_sha256.clear();
        descriptors[index].signed_contact_distribution = residue_weight
            .into_iter()
            .map(
                |((chain_id, canonical_residue_id), weight)| CanonicalResidueContactProbability {
                    chain_id,
                    canonical_residue_id,
                    chain_mapping_receipt_sha256: analysis.contract_sha256.clone(),
                    probability: weight / total,
                },
            )
            .collect();
        descriptors[index] = descriptors[index].clone().materialize()?;
        bindings.push(PhysicalLiningComponentBinding {
            source_component_sha256,
            descriptor_sha256: descriptors[index].descriptor_sha256.clone(),
            physical_lining_population_sha256: canonical_sha256(&evidence_sha256)?,
            physical_lining_evidence_sha256: evidence_sha256,
        });
    }
    descriptors.sort_by(|left, right| left.descriptor_sha256.cmp(&right.descriptor_sha256));
    bindings.sort_by(|left, right| {
        left.source_component_sha256
            .cmp(&right.source_component_sha256)
    });
    let mut population = PhysicalLiningDescriptorPopulation {
        schema_kind: "prism_physical_lining_descriptor_population".into(),
        schema_version: "2.0.0".into(),
        population_sha256: String::new(),
        spatial_boundary_contract_sha256: boundary_sha256.clone(),
        physical_lining_evidence: evidence.into_values().collect(),
        descriptors,
        bindings,
    };
    population.population_sha256 = population.body_sha256()?;
    population.verify()?;
    Ok(population)
}

fn mat_vec(matrix: [[f64; 3]; 3], vector: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|row| {
        matrix[row][0] * vector[0] + matrix[row][1] * vector[1] + matrix[row][2] * vector[2]
    })
}

fn invert_3x3(matrix: [[f64; 3]; 3]) -> Result<[[f64; 3]; 3]> {
    let determinant = matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
    if !determinant.is_finite() || determinant.abs() <= 1.0e-12 {
        bail!("non-invertible periodic box");
    }
    let inverse = [
        [
            matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1],
            matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2],
            matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1],
        ],
        [
            matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2],
            matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0],
            matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2],
        ],
        [
            matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0],
            matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1],
            matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0],
        ],
    ];
    Ok(inverse.map(|row| row.map(|value| value / determinant)))
}

fn metric_tensor(matrix: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|left| {
        std::array::from_fn(|right| {
            (0..3)
                .map(|axis| matrix[axis][left] * matrix[axis][right])
                .sum()
        })
    })
}

fn triclinic_minimum_image_translation(
    fractional: [f64; 3],
    metric: [[f64; 3]; 3],
) -> Result<[i64; 3]> {
    if fractional.iter().any(|value| !value.is_finite()) {
        bail!("triclinic fractional displacement is nonfinite");
    }
    let inverse_metric = invert_3x3(metric).context("triclinic metric is singular")?;
    let quadratic = |delta: [f64; 3]| -> f64 {
        (0..3)
            .map(|left| {
                (0..3)
                    .map(|right| delta[left] * metric[left][right] * delta[right])
                    .sum::<f64>()
            })
            .sum()
    };
    let initial_translation = fractional.map(f64::round);
    let initial_delta = std::array::from_fn(|axis| fractional[axis] - initial_translation[axis]);
    let initial_norm2 = quadratic(initial_delta).max(0.0);
    // ||f-n|| <= ||H^-1||_2 ||H(f-n)||.  sqrt(trace(G^-1))
    // is a rigorous upper bound for ||H^-1||_2 and therefore yields a
    // complete, data-derived enumeration rather than a fixed neighbor shell.
    let inverse_metric_trace = inverse_metric[0][0] + inverse_metric[1][1] + inverse_metric[2][2];
    if !inverse_metric_trace.is_finite() || inverse_metric_trace <= 0.0 {
        bail!("triclinic inverse metric trace is invalid");
    }
    let bound = inverse_metric_trace.sqrt() * initial_norm2.sqrt();
    let ranges = fractional.map(|value| -> Result<(i64, i64)> {
        let padding = f64::EPSILON * (1.0 + value.abs() + bound);
        let lower = (value - bound - padding).ceil();
        let upper = (value + bound + padding).floor();
        if lower < i64::MIN as f64 || upper > i64::MAX as f64 || lower > upper {
            bail!("triclinic closest-lattice range is invalid");
        }
        Ok((lower as i64, upper as i64))
    });
    let [x_range, y_range, z_range] = ranges;
    let (x_range, y_range, z_range) = (x_range?, y_range?, z_range?);
    let mut best = initial_translation.map(|value| value as i64);
    let mut best_norm2 = initial_norm2;
    for nx in x_range.0..=x_range.1 {
        for ny in y_range.0..=y_range.1 {
            for nz in z_range.0..=z_range.1 {
                let translation = [nx, ny, nz];
                let delta = std::array::from_fn(|axis| fractional[axis] - translation[axis] as f64);
                let norm2 = quadratic(delta).max(0.0);
                if norm2 < best_norm2
                    || (norm2.to_bits() == best_norm2.to_bits() && translation < best)
                {
                    best = translation;
                    best_norm2 = norm2;
                }
            }
        }
    }
    Ok(best)
}

fn triclinic_minimum_image_cartesian(
    fractional: [f64; 3],
    matrix: [[f64; 3]; 3],
) -> Result<[f64; 3]> {
    let translation = triclinic_minimum_image_translation(fractional, metric_tensor(matrix))?;
    Ok(mat_vec(
        matrix,
        std::array::from_fn(|axis| fractional[axis] - translation[axis] as f64),
    ))
}

fn triclinic_minimum_image_norm(fractional: [f64; 3], metric_upper: [f64; 6]) -> Result<f64> {
    let [gxx, gxy, gxz, gyy, gyz, gzz] = metric_upper;
    let metric = [[gxx, gxy, gxz], [gxy, gyy, gyz], [gxz, gyz, gzz]];
    let translation = triclinic_minimum_image_translation(fractional, metric)?;
    let delta: [f64; 3] = std::array::from_fn(|axis| fractional[axis] - translation[axis] as f64);
    let norm2 = (0..3)
        .map(|left| {
            (0..3)
                .map(|right| delta[left] * metric[left][right] * delta[right])
                .sum::<f64>()
        })
        .sum::<f64>();
    Ok(norm2.max(0.0).sqrt())
}

fn spatial_distribution(
    mode: SpatialCoordinateMode,
    positions: &[[f64; 3]],
    box_vectors: Option<[[f64; 3]; 3]>,
    box_sha256: Option<String>,
) -> Result<PeriodicAwareSpatialDistribution> {
    if positions.is_empty() {
        bail!("component spatial distribution has no nodes");
    }
    let mean = match mode {
        SpatialCoordinateMode::CartesianAngstrom => std::array::from_fn(|axis| {
            positions.iter().map(|point| point[axis]).sum::<f64>() / positions.len() as f64
        }),
        SpatialCoordinateMode::FractionalPeriodic => std::array::from_fn(|axis| {
            let (sin_sum, cos_sum) = positions.iter().fold((0.0, 0.0), |(sine, cosine), point| {
                let angle = std::f64::consts::TAU * point[axis];
                (sine + angle.sin(), cosine + angle.cos())
            });
            (sin_sum.atan2(cos_sum) / std::f64::consts::TAU).rem_euclid(1.0)
        }),
    };
    let coordinate_displacements = positions
        .iter()
        .map(|point| {
            std::array::from_fn(|axis| {
                let delta = point[axis] - mean[axis];
                match mode {
                    SpatialCoordinateMode::CartesianAngstrom => delta,
                    SpatialCoordinateMode::FractionalPeriodic => delta,
                }
            })
        })
        .collect::<Vec<[f64; 3]>>();
    let displacements = coordinate_displacements
        .iter()
        .map(|delta| match (mode, box_vectors) {
            (SpatialCoordinateMode::FractionalPeriodic, Some(matrix)) => {
                triclinic_minimum_image_cartesian(*delta, matrix)
            }
            (SpatialCoordinateMode::CartesianAngstrom, None) => Ok(*delta),
            _ => bail!("component spatial distribution boundary is incomplete"),
        })
        .collect::<Result<Vec<_>>>()?;
    let denominator = positions.len() as f64;
    let covariance_upper = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)].map(|(left, right)| {
        displacements
            .iter()
            .map(|delta| delta[left] * delta[right])
            .sum::<f64>()
            / denominator
    });
    let mut radii = displacements
        .iter()
        .map(|delta| (delta[0].powi(2) + delta[1].powi(2) + delta[2].powi(2)).sqrt())
        .collect::<Vec<_>>();
    radii.sort_by(f64::total_cmp);
    let quantile = |fraction: f64| -> f64 {
        let index = ((radii.len() - 1) as f64 * fraction).round() as usize;
        radii[index]
    };
    Ok(PeriodicAwareSpatialDistribution {
        coordinate_mode: mode,
        estimator_semantics: match mode {
            SpatialCoordinateMode::CartesianAngstrom => {
                "cartesian_mean_covariance_radial_quantiles_v1"
            }
            SpatialCoordinateMode::FractionalPeriodic => {
                "fractional_circular_mean_triclinic_cartesian_moments_v2"
            }
        }
        .to_string(),
        mean_xyz: mean,
        covariance_upper,
        radial_quantiles_25_50_75: [quantile(0.25), quantile(0.50), quantile(0.75)],
        periodic_box_vectors_sha256: box_sha256,
        periodic_box_metric_tensor_upper: box_vectors.map(|matrix| {
            let g = |left: usize, right: usize| {
                (0..3)
                    .map(|axis| matrix[axis][left] * matrix[axis][right])
                    .sum::<f64>()
            };
            [g(0, 0), g(0, 1), g(0, 2), g(1, 1), g(1, 2), g(2, 2)]
        }),
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalResidueContactProbability {
    pub chain_id: String,
    pub canonical_residue_id: String,
    pub chain_mapping_receipt_sha256: String,
    pub probability: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpatialCoordinateMode {
    CartesianAngstrom,
    FractionalPeriodic,
}

/// Gaussian-moment and radial-quantile summary produced from the complete
/// component node population. Fractional-periodic means must be circular
/// means and covariance must use minimum-image displacements.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PeriodicAwareSpatialDistribution {
    pub coordinate_mode: SpatialCoordinateMode,
    pub estimator_semantics: String,
    pub mean_xyz: [f64; 3],
    /// Symmetric covariance `[xx, xy, xz, yy, yz, zz]`.
    pub covariance_upper: [f64; 6],
    pub radial_quantiles_25_50_75: [f64; 3],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub periodic_box_vectors_sha256: Option<String>,
    /// Triclinic fractional-coordinate metric `[gxx,gxy,gxz,gyy,gyz,gzz]`
    /// where `G = H^T H`. This is required to convert minimum-image
    /// fractional displacements into physical Angstrom distances.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub periodic_box_metric_tensor_upper: Option<[f64; 6]>,
}

impl PeriodicAwareSpatialDistribution {
    fn verify(&self) -> Result<()> {
        let expected_semantics = match self.coordinate_mode {
            SpatialCoordinateMode::CartesianAngstrom => {
                "cartesian_mean_covariance_radial_quantiles_v1"
            }
            SpatialCoordinateMode::FractionalPeriodic => {
                "fractional_circular_mean_triclinic_cartesian_moments_v2"
            }
        };
        if self.estimator_semantics != expected_semantics
            || self
                .mean_xyz
                .iter()
                .chain(&self.covariance_upper)
                .chain(&self.radial_quantiles_25_50_75)
                .any(|value| !value.is_finite())
            || self.covariance_upper[0] < 0.0
            || self.covariance_upper[3] < 0.0
            || self.covariance_upper[5] < 0.0
            || self
                .radial_quantiles_25_50_75
                .windows(2)
                .any(|pair| pair[0] > pair[1])
            || self.radial_quantiles_25_50_75[0] < 0.0
        {
            bail!("invalid periodic-aware component spatial distribution");
        }
        match self.coordinate_mode {
            SpatialCoordinateMode::CartesianAngstrom => {
                if self.periodic_box_vectors_sha256.is_some() {
                    bail!("Cartesian component distribution carries a periodic box");
                }
                if self.periodic_box_metric_tensor_upper.is_some() {
                    bail!("Cartesian component distribution carries a periodic metric");
                }
            }
            SpatialCoordinateMode::FractionalPeriodic => {
                if self
                    .mean_xyz
                    .iter()
                    .any(|value| !(0.0..1.0).contains(value))
                {
                    bail!("fractional circular mean is outside [0,1)");
                }
                require_sha256(
                    self.periodic_box_vectors_sha256
                        .as_deref()
                        .context("periodic component distribution lacks box vectors")?,
                    "periodic box vectors",
                )?;
                let metric = self
                    .periodic_box_metric_tensor_upper
                    .context("periodic component distribution lacks triclinic metric")?;
                if metric.iter().any(|value| !value.is_finite())
                    || metric[0] <= 0.0
                    || metric[3] <= 0.0
                    || metric[5] <= 0.0
                {
                    bail!("periodic component metric is non-finite or non-positive");
                }
                let determinant = metric[0] * (metric[3] * metric[5] - metric[4] * metric[4])
                    - metric[1] * (metric[1] * metric[5] - metric[4] * metric[2])
                    + metric[2] * (metric[1] * metric[4] - metric[3] * metric[2]);
                if determinant <= 1.0e-12 {
                    bail!("periodic component metric is not positive definite");
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeComponentDescriptor {
    pub descriptor_sha256: String,
    pub source_component_sha256: String,
    pub run_id: String,
    pub replica_id: String,
    pub replica_seed: u64,
    pub control_lane: ControlLane,
    pub signed_contact_distribution: Vec<CanonicalResidueContactProbability>,
    pub spatial_distribution: PeriodicAwareSpatialDistribution,
}

#[derive(Serialize)]
struct NativeComponentDescriptorBody<'a> {
    source_component_sha256: &'a str,
    run_id: &'a str,
    replica_id: &'a str,
    replica_seed: u64,
    control_lane: ControlLane,
    signed_contact_distribution: &'a [CanonicalResidueContactProbability],
    spatial_distribution: &'a PeriodicAwareSpatialDistribution,
}

impl NativeComponentDescriptor {
    fn body(&self) -> NativeComponentDescriptorBody<'_> {
        NativeComponentDescriptorBody {
            source_component_sha256: &self.source_component_sha256,
            run_id: &self.run_id,
            replica_id: &self.replica_id,
            replica_seed: self.replica_seed,
            control_lane: self.control_lane,
            signed_contact_distribution: &self.signed_contact_distribution,
            spatial_distribution: &self.spatial_distribution,
        }
    }

    pub fn materialize(mut self) -> Result<Self> {
        if !self.descriptor_sha256.is_empty() {
            bail!("component descriptor materialization requires an empty digest");
        }
        self.descriptor_sha256 = canonical_sha256(&self.body())?;
        self.verify()?;
        Ok(self)
    }

    pub fn verify(&self) -> Result<()> {
        require_sha256(&self.descriptor_sha256, "component descriptor")?;
        require_sha256(&self.source_component_sha256, "source component")?;
        if self.run_id.trim().is_empty()
            || self.replica_id.trim().is_empty()
            || self.signed_contact_distribution.is_empty()
        {
            bail!("component descriptor identity/contact population is empty");
        }
        self.spatial_distribution.verify()?;
        let mut previous = None;
        let mut total = 0.0;
        for contact in &self.signed_contact_distribution {
            if contact.chain_id.trim().is_empty()
                || contact.canonical_residue_id.trim().is_empty()
                || !contact.probability.is_finite()
                || contact.probability <= 0.0
            {
                bail!("component contact distribution contains an invalid row");
            }
            require_sha256(
                &contact.chain_mapping_receipt_sha256,
                "component chain mapping receipt",
            )?;
            let key = (
                contact.chain_id.as_str(),
                contact.canonical_residue_id.as_str(),
                contact.chain_mapping_receipt_sha256.as_str(),
            );
            if previous.is_some_and(|prior| prior >= key) {
                bail!("component contact distribution is duplicate or unsorted");
            }
            previous = Some(key);
            total += contact.probability;
        }
        if (total - 1.0).abs() > 1.0e-9 || canonical_sha256(&self.body())? != self.descriptor_sha256
        {
            bail!("component descriptor probability/hash conservation failed");
        }
        Ok(())
    }

    fn run_unit(&self) -> (&str, &str, u64, ControlLane) {
        (
            &self.run_id,
            &self.replica_id,
            self.replica_seed,
            self.control_lane,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComponentCorrespondencePolicy {
    pub policy_sha256: String,
    pub residue_js_weight: f64,
    pub spatial_weight: f64,
    pub maximum_assignment_cost: f64,
    pub ambiguity_margin: f64,
}

impl ComponentCorrespondencePolicy {
    pub fn materialize(mut self) -> Result<Self> {
        if !self.policy_sha256.is_empty() {
            bail!("correspondence policy materialization requires an empty digest");
        }
        self.policy_sha256 = canonical_sha256(&(
            self.residue_js_weight,
            self.spatial_weight,
            self.maximum_assignment_cost,
            self.ambiguity_margin,
        ))?;
        self.verify()?;
        Ok(self)
    }

    pub fn verify(&self) -> Result<()> {
        require_sha256(&self.policy_sha256, "correspondence policy")?;
        if [
            self.residue_js_weight,
            self.spatial_weight,
            self.maximum_assignment_cost,
            self.ambiguity_margin,
        ]
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
            || (self.residue_js_weight + self.spatial_weight - 1.0).abs() > 1.0e-12
            || self.maximum_assignment_cost <= 0.0
            || self.ambiguity_margin >= self.maximum_assignment_cost
            || canonical_sha256(&(
                self.residue_js_weight,
                self.spatial_weight,
                self.maximum_assignment_cost,
                self.ambiguity_margin,
            ))? != self.policy_sha256
        {
            bail!("invalid component correspondence policy");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum ComponentCorrespondenceStatus {
    Recurrent {
        correspondence_sha256: String,
        anchor_descriptor_sha256: String,
        anchor: bool,
        assignment_cost: f64,
    },
    Ambiguous {
        candidate_correspondence_sha256: Vec<String>,
        candidate_cost: Vec<f64>,
    },
    Unmatched {
        reason_code: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComponentCorrespondenceAssignment {
    pub descriptor_sha256: String,
    pub status: ComponentCorrespondenceStatus,
}

/// Actual topology-backed canonical residue namespace for one physical lane.
/// Original contact receipts and descriptor hashes are retained in the ledger.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LaneContactMappingAuthority {
    pub topology_path: std::path::PathBuf,
    pub topology_sha256: String,
    pub chain_mapping_receipt_sha256: String,
    pub atom_analysis_residue_ids_sha256: String,
    pub canonical_residues: BTreeSet<(String, String)>,
}

impl LaneContactMappingAuthority {
    pub fn from_topology_path(path: &std::path::Path, expected_sha256: &str) -> Result<Self> {
        use std::io::Read;
        use std::os::unix::fs::OpenOptionsExt;
        use sha2::{Digest, Sha256};
        require_sha256(expected_sha256, "contact mapping topology")?;
        if !path.is_absolute() || path.canonicalize()? != path {
            bail!("contact mapping topology path is not canonical");
        }
        let mut file = std::fs::OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW).open(path)?;
        if !file.metadata()?.is_file() {
            bail!("contact mapping topology is not a regular file");
        }
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        if format!("{:x}", Sha256::digest(&bytes)) != expected_sha256 {
            bail!("contact mapping topology hash differs from lane declaration");
        }
        let topology: crate::input::PrismPrepTopology = serde_json::from_slice(&bytes)?;
        topology.validate().map_err(anyhow::Error::msg)?;
        topology.validate_analysis_residue_axis().map_err(anyhow::Error::msg)?;
        let mapping = topology.analysis_residue_contract.as_ref().context("contact mapping lacks analysis residue authority")?;
        let canonical_residues = mapping.analysis_residues.iter().map(|row| {
            (row.chain_id.clone(), format!("{}:{}", row.pdb_residue_id, row.residue_name))
        }).collect::<BTreeSet<_>>();
        if canonical_residues.len() != mapping.analysis_residues.len() || canonical_residues.is_empty() {
            bail!("contact mapping has ambiguous or empty canonical residue identity");
        }
        Ok(Self { topology_path: path.to_path_buf(), topology_sha256: expected_sha256.into(),
            chain_mapping_receipt_sha256: mapping.contract_sha256.clone(),
            atom_analysis_residue_ids_sha256: mapping.atom_analysis_residue_ids_sha256.clone(), canonical_residues })
    }

    fn verify(&self) -> Result<()> {
        if *self != Self::from_topology_path(&self.topology_path, &self.topology_sha256)? {
            bail!("contact mapping authority differs from actual topology");
        }
        Ok(())
    }

    fn verify_descriptor(&self, descriptor: &NativeComponentDescriptor) -> Result<()> {
        for row in &descriptor.signed_contact_distribution {
            if row.chain_mapping_receipt_sha256 != self.chain_mapping_receipt_sha256
                || !self.canonical_residues.contains(&(row.chain_id.clone(), row.canonical_residue_id.clone())) {
                bail!("component contact is not a member of its verified lane mapping");
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComponentCorrespondenceLedger {
    pub schema_kind: String,
    pub schema_version: String,
    pub ledger_sha256: String,
    pub policy: ComponentCorrespondencePolicy,
    pub descriptor_population_sha256: String,
    pub descriptors: Vec<NativeComponentDescriptor>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub contact_mappings_by_lane: BTreeMap<ControlLane, LaneContactMappingAuthority>,
    pub assignments: Vec<ComponentCorrespondenceAssignment>,
    pub recurrent_correspondence_count: usize,
    pub ambiguous_count: usize,
    pub unmatched_count: usize,
}

fn contact_map(descriptor: &NativeComponentDescriptor, verified_canonical_namespace: bool) -> BTreeMap<(String, String, String), f64> {
    descriptor
        .signed_contact_distribution
        .iter()
        .map(|row| {
            (
                (
                    row.chain_id.clone(),
                    row.canonical_residue_id.clone(),
                    if verified_canonical_namespace { String::new() } else { row.chain_mapping_receipt_sha256.clone() },
                ),
                row.probability,
            )
        })
        .collect()
}

fn component_cost(
    left: &NativeComponentDescriptor,
    right: &NativeComponentDescriptor,
    policy: &ComponentCorrespondencePolicy,
) -> Result<f64> {
    component_cost_in_namespace(left, right, policy, false)
}

fn component_cost_in_namespace(
    left: &NativeComponentDescriptor,
    right: &NativeComponentDescriptor,
    policy: &ComponentCorrespondencePolicy,
    verified_canonical_namespace: bool,
) -> Result<f64> {
    if left.run_unit() == right.run_unit() {
        bail!("correspondence cost cannot compare components from one run unit");
    }
    if left.spatial_distribution.coordinate_mode != right.spatial_distribution.coordinate_mode
        || left.spatial_distribution.periodic_box_vectors_sha256
            != right.spatial_distribution.periodic_box_vectors_sha256
    {
        bail!("component spatial frames are not comparable");
    }
    let left_contacts = contact_map(left, verified_canonical_namespace);
    let right_contacts = contact_map(right, verified_canonical_namespace);
    let keys = left_contacts
        .keys()
        .chain(right_contacts.keys())
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut js = 0.0;
    for key in keys {
        let p = left_contacts.get(&key).copied().unwrap_or(0.0);
        let q = right_contacts.get(&key).copied().unwrap_or(0.0);
        let mean = 0.5 * (p + q);
        if p > 0.0 {
            js += 0.5 * p * (p / mean).ln();
        }
        if q > 0.0 {
            js += 0.5 * q * (q / mean).ln();
        }
    }
    let residue_distance = (js / std::f64::consts::LN_2).clamp(0.0, 1.0);
    let delta = (0..3)
        .map(|axis| {
            left.spatial_distribution.mean_xyz[axis] - right.spatial_distribution.mean_xyz[axis]
        })
        .collect::<Vec<_>>();
    let mean_distance = match left.spatial_distribution.coordinate_mode {
        SpatialCoordinateMode::CartesianAngstrom => {
            delta.iter().map(|value| value * value).sum::<f64>().sqrt()
        }
        SpatialCoordinateMode::FractionalPeriodic => {
            let left_metric = left
                .spatial_distribution
                .periodic_box_metric_tensor_upper
                .context("left periodic descriptor lacks triclinic metric")?;
            let right_metric = right
                .spatial_distribution
                .periodic_box_metric_tensor_upper
                .context("right periodic descriptor lacks triclinic metric")?;
            if left_metric != right_metric {
                bail!("component periodic metric tensors differ");
            }
            triclinic_minimum_image_norm([delta[0], delta[1], delta[2]], left_metric)?
        }
    };
    let covariance_distance = left
        .spatial_distribution
        .covariance_upper
        .iter()
        .zip(&right.spatial_distribution.covariance_upper)
        .map(|(a, b)| (a - b).abs() / (a.abs() + b.abs() + 1.0e-12))
        .sum::<f64>()
        / 6.0;
    let quantile_distance = left
        .spatial_distribution
        .radial_quantiles_25_50_75
        .iter()
        .zip(&right.spatial_distribution.radial_quantiles_25_50_75)
        .map(|(a, b)| (a - b).abs() / (a.abs() + b.abs() + 1.0e-12))
        .sum::<f64>()
        / 3.0;
    let spatial_distance =
        ((mean_distance / (1.0 + mean_distance)) + covariance_distance + quantile_distance) / 3.0;
    Ok(policy.residue_js_weight * residue_distance + policy.spatial_weight * spatial_distance)
}

fn correspondence_id(policy_sha256: &str, anchor_descriptor_sha256: &str) -> Result<String> {
    canonical_sha256(&(
        "prism_component_correspondence_v1",
        policy_sha256,
        anchor_descriptor_sha256,
    ))
}

/// Deterministic reference-anchored one-to-one correspondence. Each earliest
/// remaining run unit becomes an anchor stratum. Near-ties on either side are
/// preserved as ambiguity; they are never silently tie-broken into recurrence.
pub fn establish_component_correspondence(
    descriptors: Vec<NativeComponentDescriptor>,
    policy: ComponentCorrespondencePolicy,
) -> Result<ComponentCorrespondenceLedger> {
    establish_component_correspondence_with_lane_mappings(descriptors, policy, BTreeMap::new())
}

/// Compare canonical residues only after verifying every original receipt and
/// residue against the descriptor's predeclared, actual topology-local mapping.
pub fn establish_component_correspondence_with_lane_mappings(
    mut descriptors: Vec<NativeComponentDescriptor>,
    policy: ComponentCorrespondencePolicy,
    contact_mappings_by_lane: BTreeMap<ControlLane, LaneContactMappingAuthority>,
) -> Result<ComponentCorrespondenceLedger> {
    for mapping in contact_mappings_by_lane.values() { mapping.verify()?; }
    if !contact_mappings_by_lane.is_empty() {
        if contact_mappings_by_lane.keys().copied().collect::<BTreeSet<_>>() != BTreeSet::from([
            ControlLane::NoProbe, ControlLane::HypothesisFragment, ControlLane::OrthogonalChemicalControl]) {
            bail!("correspondence requires all three declared contact mapping lanes");
        }
        for descriptor in &descriptors {
            contact_mappings_by_lane.get(&descriptor.control_lane)
                .context("descriptor lane has no contact mapping authority")?.verify_descriptor(descriptor)?;
        }
    }
    policy.verify()?;
    descriptors.sort_by(|left, right| left.descriptor_sha256.cmp(&right.descriptor_sha256));
    for descriptor in &descriptors {
        descriptor.verify()?;
    }
    if descriptors.is_empty()
        || descriptors
            .windows(2)
            .any(|pair| pair[0].descriptor_sha256 == pair[1].descriptor_sha256)
    {
        bail!("component correspondence descriptor population is empty or duplicate");
    }
    let descriptor_population_sha256 = canonical_sha256(&descriptors)?;
    let mut remaining = descriptors
        .iter()
        .map(|row| row.descriptor_sha256.clone())
        .collect::<BTreeSet<_>>();
    let by_hash = descriptors
        .iter()
        .map(|row| (row.descriptor_sha256.clone(), row))
        .collect::<BTreeMap<_, _>>();
    let mut statuses = BTreeMap::new();
    while !remaining.is_empty() {
        let anchor_unit = remaining
            .iter()
            .map(|hash| by_hash[hash].run_unit())
            .min()
            .unwrap();
        let anchors = remaining
            .iter()
            .filter(|hash| by_hash[*hash].run_unit() == anchor_unit)
            .cloned()
            .collect::<Vec<_>>();
        let anchor_correspondence = anchors
            .iter()
            .map(|hash| {
                Ok((
                    hash.clone(),
                    correspondence_id(&policy.policy_sha256, hash)?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        let mut recurrent_members: BTreeMap<String, Vec<(String, f64)>> = anchors
            .iter()
            .map(|hash| (hash.clone(), Vec::new()))
            .collect();
        let target_units = remaining
            .iter()
            .map(|hash| by_hash[hash].run_unit())
            .filter(|unit| *unit != anchor_unit)
            .collect::<BTreeSet<_>>();
        for target_unit in target_units {
            let targets = remaining
                .iter()
                .filter(|hash| by_hash[*hash].run_unit() == target_unit)
                .cloned()
                .collect::<Vec<_>>();
            let mut costs = BTreeMap::<(String, String), f64>::new();
            for anchor in &anchors {
                for target in &targets {
                    if let Ok(cost) = component_cost_in_namespace(by_hash[anchor], by_hash[target], &policy, !contact_mappings_by_lane.is_empty()) {
                        costs.insert((anchor.clone(), target.clone()), cost);
                    }
                }
            }
            for target in targets {
                let mut target_choices = anchors
                    .iter()
                    .filter_map(|anchor| {
                        costs
                            .get(&(anchor.clone(), target.clone()))
                            .copied()
                            .filter(|cost| *cost <= policy.maximum_assignment_cost)
                            .map(|cost| (anchor.clone(), cost))
                    })
                    .collect::<Vec<_>>();
                target_choices.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
                let Some((best_anchor, best_cost)) = target_choices.first().cloned() else {
                    continue;
                };
                let target_ambiguous = target_choices
                    .get(1)
                    .is_some_and(|(_, cost)| *cost - best_cost <= policy.ambiguity_margin);
                let mut anchor_choices =
                    targets_for_anchor(&costs, &best_anchor, &remaining, &by_hash)
                        .into_iter()
                        .filter(|(_, cost)| *cost <= policy.maximum_assignment_cost)
                        .collect::<Vec<_>>();
                anchor_choices.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
                let anchor_ambiguous = anchor_choices.get(1).is_some_and(|(_, cost)| {
                    *cost - anchor_choices[0].1 <= policy.ambiguity_margin
                });
                if target_ambiguous || anchor_ambiguous || anchor_choices[0].0 != target {
                    let viable = target_choices
                        .iter()
                        .take_while(|(_, cost)| *cost - best_cost <= policy.ambiguity_margin)
                        .cloned()
                        .collect::<Vec<_>>();
                    statuses.insert(
                        target.clone(),
                        ComponentCorrespondenceStatus::Ambiguous {
                            candidate_correspondence_sha256: viable
                                .iter()
                                .map(|(anchor, _)| anchor_correspondence[anchor].clone())
                                .collect(),
                            candidate_cost: viable.iter().map(|(_, cost)| *cost).collect(),
                        },
                    );
                    remaining.remove(&target);
                } else {
                    recurrent_members
                        .get_mut(&best_anchor)
                        .unwrap()
                        .push((target.clone(), best_cost));
                    remaining.remove(&target);
                }
            }
        }
        for anchor in anchors {
            remaining.remove(&anchor);
            let members = recurrent_members.remove(&anchor).unwrap();
            if members.is_empty() {
                statuses.insert(
                    anchor,
                    ComponentCorrespondenceStatus::Unmatched {
                        reason_code: "no_unambiguous_cross_run_match_within_threshold".into(),
                    },
                );
            } else {
                let correspondence_sha256 = anchor_correspondence[&anchor].clone();
                statuses.insert(
                    anchor.clone(),
                    ComponentCorrespondenceStatus::Recurrent {
                        correspondence_sha256: correspondence_sha256.clone(),
                        anchor_descriptor_sha256: anchor.clone(),
                        anchor: true,
                        assignment_cost: 0.0,
                    },
                );
                for (member, cost) in members {
                    statuses.insert(
                        member,
                        ComponentCorrespondenceStatus::Recurrent {
                            correspondence_sha256: correspondence_sha256.clone(),
                            anchor_descriptor_sha256: anchor.clone(),
                            anchor: false,
                            assignment_cost: cost,
                        },
                    );
                }
            }
        }
    }
    let assignments = descriptors
        .iter()
        .map(|descriptor| {
            Ok(ComponentCorrespondenceAssignment {
                descriptor_sha256: descriptor.descriptor_sha256.clone(),
                status: statuses
                    .remove(&descriptor.descriptor_sha256)
                    .context("component correspondence omitted a descriptor")?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let recurrent_correspondence_count = assignments
        .iter()
        .filter_map(|row| match &row.status {
            ComponentCorrespondenceStatus::Recurrent {
                correspondence_sha256,
                ..
            } => Some(correspondence_sha256.clone()),
            _ => None,
        })
        .collect::<BTreeSet<_>>()
        .len();
    let ambiguous_count = assignments
        .iter()
        .filter(|row| matches!(row.status, ComponentCorrespondenceStatus::Ambiguous { .. }))
        .count();
    let unmatched_count = assignments
        .iter()
        .filter(|row| matches!(row.status, ComponentCorrespondenceStatus::Unmatched { .. }))
        .count();
    let legacy_content_sha256 = canonical_sha256(&(
        &policy,
        descriptor_population_sha256.as_str(),
        &descriptors,
        &assignments,
        recurrent_correspondence_count,
        ambiguous_count,
        unmatched_count,
    ))?;
    let ledger_sha256 = if contact_mappings_by_lane.is_empty() {
        legacy_content_sha256
    } else {
        canonical_sha256(&("verified_lane_contact_namespace_v2", legacy_content_sha256, &contact_mappings_by_lane))?
    };
    Ok(ComponentCorrespondenceLedger {
        schema_kind: "prism_component_correspondence_ledger".into(),
        schema_version: if contact_mappings_by_lane.is_empty() { "1.0.0" } else { "2.0.0" }.into(),
        ledger_sha256,
        policy,
        descriptor_population_sha256,
        contact_mappings_by_lane,
        descriptors,
        assignments,
        recurrent_correspondence_count,
        ambiguous_count,
        unmatched_count,
    })
}

fn targets_for_anchor(
    costs: &BTreeMap<(String, String), f64>,
    anchor: &str,
    remaining: &BTreeSet<String>,
    by_hash: &BTreeMap<String, &NativeComponentDescriptor>,
) -> Vec<(String, f64)> {
    let target_unit = costs
        .keys()
        .find(|(candidate_anchor, _)| candidate_anchor == anchor)
        .map(|(_, target)| by_hash[target].run_unit());
    costs
        .iter()
        .filter_map(|((candidate_anchor, target), cost)| {
            (candidate_anchor == anchor
                && remaining.contains(target)
                && target_unit.is_some_and(|unit| by_hash[target].run_unit() == unit))
            .then_some((target.clone(), *cost))
        })
        .collect()
}

impl ComponentCorrespondenceLedger {
    pub fn verify(&self) -> Result<()> {
        if self.schema_kind != "prism_component_correspondence_ledger"
            || !matches!(self.schema_version.as_str(), "1.0.0" | "2.0.0")
            || (self.schema_version == "1.0.0") != self.contact_mappings_by_lane.is_empty()
        {
            bail!("unsupported component correspondence ledger schema");
        }
        let rebuilt =
            establish_component_correspondence_with_lane_mappings(self.descriptors.clone(), self.policy.clone(), self.contact_mappings_by_lane.clone())?;
        if &rebuilt != self {
            bail!("component correspondence ledger is not deterministic/content exact");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: u8) -> String {
        format!("{byte:02x}").repeat(32)
    }

    fn policy() -> ComponentCorrespondencePolicy {
        ComponentCorrespondencePolicy {
            policy_sha256: String::new(),
            residue_js_weight: 0.7,
            spatial_weight: 0.3,
            maximum_assignment_cost: 0.25,
            ambiguity_margin: 0.01,
        }
        .materialize()
        .unwrap()
    }

    fn descriptor(source: u8, run: &str, seed: u64, mean_x: f64) -> NativeComponentDescriptor {
        NativeComponentDescriptor {
            descriptor_sha256: String::new(),
            source_component_sha256: digest(source),
            run_id: run.into(),
            replica_id: format!("rep_{seed:02}"),
            replica_seed: seed,
            control_lane: ControlLane::NoProbe,
            signed_contact_distribution: vec![CanonicalResidueContactProbability {
                chain_id: "MYC".into(),
                canonical_residue_id: "R902".into(),
                chain_mapping_receipt_sha256: digest(9),
                probability: 1.0,
            }],
            spatial_distribution: PeriodicAwareSpatialDistribution {
                coordinate_mode: SpatialCoordinateMode::FractionalPeriodic,
                estimator_semantics: "fractional_circular_mean_triclinic_cartesian_moments_v2"
                    .into(),
                mean_xyz: [mean_x, 0.5, 0.5],
                covariance_upper: [0.01, 0.0, 0.0, 0.01, 0.0, 0.01],
                radial_quantiles_25_50_75: [0.05, 0.1, 0.15],
                periodic_box_vectors_sha256: Some(digest(8)),
                periodic_box_metric_tensor_upper: Some([1600.0, 0.0, 0.0, 1600.0, 0.0, 1600.0]),
            },
        }
        .materialize()
        .unwrap()
    }

    #[test]
    fn correspondence_is_permutation_invariant_and_periodic_aware() {
        let descriptors = vec![
            descriptor(1, "run_a", 1, 0.99),
            descriptor(2, "run_b", 2, 0.01),
            descriptor(3, "run_c", 3, 0.98),
        ];
        let forward = establish_component_correspondence(descriptors.clone(), policy()).unwrap();
        let reverse =
            establish_component_correspondence(descriptors.into_iter().rev().collect(), policy())
                .unwrap();
        assert_eq!(forward, reverse);
        assert_eq!(forward.recurrent_correspondence_count, 1);
        assert_eq!(forward.ambiguous_count, 0);
        assert_eq!(forward.unmatched_count, 0);
        forward.verify().unwrap();
    }

    #[test]
    fn near_tie_is_preserved_as_ambiguous_not_hash_tiebroken() {
        let descriptors = vec![
            descriptor(1, "run_a", 1, 0.50),
            descriptor(2, "run_a", 1, 0.50),
            descriptor(3, "run_b", 2, 0.50),
        ];
        let ledger = establish_component_correspondence(descriptors, policy()).unwrap();
        assert_eq!(ledger.ambiguous_count, 1);
        let ambiguous = ledger
            .assignments
            .iter()
            .find_map(|row| match &row.status {
                ComponentCorrespondenceStatus::Ambiguous {
                    candidate_correspondence_sha256,
                    ..
                } => Some(candidate_correspondence_sha256),
                _ => None,
            })
            .unwrap();
        assert_eq!(ambiguous.len(), 2);
        assert_eq!(ledger.unmatched_count, 2);
    }

    #[test]
    fn periodic_cost_uses_physical_triclinic_metric_not_fractional_euclidean() {
        let mut left = descriptor(1, "run_a", 1, 0.10);
        let mut right = descriptor(2, "run_b", 2, 0.00);
        // H columns are (40,0,0), (36,4,0), (0,0,40). A fractional
        // displacement (0.1,-0.1,0) is only sqrt(0.02) in fractional units
        // but sqrt(0.32) Angstrom under G=H^T H.
        let metric = [1600.0, 1440.0, 0.0, 1312.0, 0.0, 1600.0];
        for row in [&mut left, &mut right] {
            row.descriptor_sha256.clear();
            row.spatial_distribution.periodic_box_metric_tensor_upper = Some(metric);
            *row = row.clone().materialize().unwrap();
        }
        right.descriptor_sha256.clear();
        right.spatial_distribution.mean_xyz[1] = 0.60;
        right = right.materialize().unwrap();
        let physical = component_cost(
            &left,
            &right,
            &ComponentCorrespondencePolicy {
                policy_sha256: String::new(),
                residue_js_weight: 0.0,
                spatial_weight: 1.0,
                maximum_assignment_cost: 0.9,
                ambiguity_margin: 0.01,
            }
            .materialize()
            .unwrap(),
        )
        .unwrap();
        let physical_distance = 0.32f64.sqrt();
        let expected = (physical_distance / (1.0 + physical_distance)) / 3.0;
        assert!((physical - expected).abs() < 1.0e-12);
        let fractional_distance = 0.02f64.sqrt();
        let forbidden = (fractional_distance / (1.0 + fractional_distance)) / 3.0;
        assert!((physical - forbidden).abs() > 0.05);
    }

    #[test]
    fn triclinic_metric_uses_exact_nearest_lattice_image() {
        let matrix = [[10.0, 9.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 10.0]];
        let metric_matrix = metric_tensor(matrix);
        let metric = [
            metric_matrix[0][0],
            metric_matrix[0][1],
            metric_matrix[0][2],
            metric_matrix[1][1],
            metric_matrix[1][2],
            metric_matrix[2][2],
        ];
        let observed = triclinic_minimum_image_norm([0.49, 0.49, 0.0], metric).unwrap();
        let expected = (0.31f64.powi(2) + 0.51f64.powi(2)).sqrt();
        assert!((observed - expected).abs() < 1.0e-12);
        let forbidden_axis_rounding = (9.31f64.powi(2) + 0.49f64.powi(2)).sqrt();
        assert!(observed < forbidden_axis_rounding / 10.0);
    }

    fn cavity_label_authority() -> CavityBinaryLabelAuthority {
        CavityBinaryLabelAuthority {
            schema_kind: "prism_cavity_binary_label_authority".into(),
            schema_version: "1.0.0".into(),
            authority_sha256: String::new(),
            source_design_authority_sha256: digest(31),
            closed_max_volume_a3: 100.0,
            open_min_volume_a3: 200.0,
        }
        .materialize()
        .unwrap()
    }

    #[test]
    fn multi_grid_threshold_straddle_is_indeterminate_and_continuous() {
        let authority = cavity_label_authority();
        let measurement = MultiGridCavityMeasurement::materialize(
            digest(32),
            vec![90.0, 210.0, 220.0, 80.0],
            &authority,
        )
        .unwrap();
        assert_eq!(
            measurement.binary_label,
            CavityBinaryLabel::IndeterminateGridThresholdStraddle
        );
        assert_eq!(measurement.mean_volume_a3, 150.0);
        assert_eq!(measurement.discretization_min_a3, 80.0);
        assert_eq!(measurement.discretization_max_a3, 220.0);
        measurement.verify(&authority).unwrap();
    }

    #[test]
    fn multi_grid_band_is_indeterminate_without_erasing_measurements() {
        let authority = cavity_label_authority();
        let measurement = MultiGridCavityMeasurement::materialize(
            digest(33),
            vec![120.0, 130.0, 140.0],
            &authority,
        )
        .unwrap();
        assert_eq!(
            measurement.binary_label,
            CavityBinaryLabel::IndeterminateOpenClosedBand
        );
        assert_eq!(measurement.grid_volume_a3, vec![120.0, 130.0, 140.0]);
    }
    #[test]
    fn legacy_correspondence_encoding_and_digest_are_preserved() {
        let ledger = establish_component_correspondence(vec![descriptor(101,"run-a",1,0.1), descriptor(102,"run-b",2,0.1)], policy()).unwrap();
        let raw = serde_json::to_value(&ledger).unwrap();
        assert!(raw.get("contact_mappings_by_lane").is_none());
        assert_eq!(ledger.ledger_sha256, canonical_sha256(&(
            &ledger.policy, ledger.descriptor_population_sha256.as_str(), &ledger.descriptors, &ledger.assignments,
            ledger.recurrent_correspondence_count, ledger.ambiguous_count, ledger.unmatched_count)).unwrap());
        let decoded: ComponentCorrespondenceLedger = serde_json::from_value(raw.clone()).unwrap();
        assert_eq!(serde_json::to_value(&decoded).unwrap(), raw);
        decoded.verify().unwrap();
    }

    #[test]
    fn verified_contact_namespace_preserves_original_receipts_and_rejects_unknown_contacts() {
        let left = descriptor(111,"run-a",1,0.1);
        let mut right = descriptor(112,"run-b",2,0.1);
        right.signed_contact_distribution[0].chain_mapping_receipt_sha256 = digest(20);
        right.descriptor_sha256.clear();
        let right = right.materialize().unwrap();
        let original_hash = right.descriptor_sha256.clone();
        let mapping = LaneContactMappingAuthority {
            topology_path: "/not-a-campaign-input".into(), topology_sha256: digest(21),
            chain_mapping_receipt_sha256: digest(20), atom_analysis_residue_ids_sha256: digest(22),
            canonical_residues: BTreeSet::from([("MYC".into(), "R902".into())]),
        };
        mapping.verify_descriptor(&right).unwrap();
        assert!(mapping.verify_descriptor(&left).is_err());
        assert!(component_cost(&left, &right, &policy()).unwrap() > 0.0);
        assert_eq!(component_cost_in_namespace(&left, &right, &policy(), true).unwrap(), 0.0);
        assert_eq!(right.descriptor_sha256, original_hash);
        let mut bad = right.clone();
        bad.signed_contact_distribution[0].canonical_residue_id = "R903".into();
        assert!(mapping.verify_descriptor(&bad).is_err());
    }

}
