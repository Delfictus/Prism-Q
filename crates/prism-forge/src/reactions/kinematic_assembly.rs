use crate::core::synthon::{AttachmentPoint, MolecularStateError, ScaffoldState3D, Synthon3D};
use crate::reactions::kinematics::DihedralConstraint;
use crate::reactions::reaction_registry::{AssemblyPlan, ReactionRule as RegistryReactionRule};
use anyhow::{anyhow, Context, Result};
use arrow_array::{
    Array, Float64Array, Int32Array, Int64Array, LargeStringArray, RecordBatch, StringArray,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::Deserialize;
use std::collections::HashMap;
#[cfg(any(test, debug_assertions))]
use std::collections::VecDeque;
use std::fs::File;
use std::path::Path;
use thiserror::Error;

const EPSILON: f32 = 1.0e-6;

#[derive(Debug, Clone, Deserialize)]
struct ConformerAtomRow {
    atomic_num: u8,
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoadedSynthon {
    pub synthon_id: String,
    pub canonical_smiles: String,
    pub reaction_rule_id: String,
    pub reaction_role: String,
    pub reaction_tags: Vec<String>,
    pub reaction_sites: Vec<SynthonReactionSite>,
    pub leaving_group_sites: Vec<SynthonLeavingGroupSite>,
    pub synthon: Synthon3D,
    pub atomic_numbers: Vec<u8>,
    pub leaving_group_formal_charge: f32,
}

impl LoadedSynthon {
    #[must_use]
    pub fn attachment(&self) -> &AttachmentPoint {
        &self.synthon.attachment_points[0]
    }

    #[must_use]
    pub fn has_role(&self, reaction_id: &str, role_name: &str) -> bool {
        self.reaction_tags
            .iter()
            .any(|tag| tag == &format!("{reaction_id}:{role_name}"))
            || (self.reaction_rule_id == reaction_id && self.reaction_role == role_name)
    }

    #[must_use]
    pub fn reaction_site(
        &self,
        reaction_id: &str,
        role_name: &str,
    ) -> Option<&SynthonReactionSite> {
        self.reaction_sites
            .iter()
            .find(|site| site.reaction_id == reaction_id && site.role_name == role_name)
    }

    #[must_use]
    pub fn leaving_groups_for(&self, reaction_id: &str, role_name: &str) -> Vec<usize> {
        self.leaving_group_sites
            .iter()
            .find(|site| site.reaction_id == reaction_id && site.role_name == role_name)
            .map(|site| site.leaving_group_atom_indices.clone())
            .unwrap_or_default()
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct SynthonReactionSite {
    #[serde(default)]
    pub atom_map_to_atom_idx: HashMap<String, usize>,
    #[serde(default)]
    pub multi_match_enumeration_required: bool,
    pub reaction_id: String,
    pub reactive_atom_idx: usize,
    pub reference_atom_idx: Option<usize>,
    pub role_name: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct SynthonLeavingGroupSite {
    #[serde(default)]
    pub leaving_group_atom_indices: Vec<usize>,
    pub reaction_id: String,
    pub role_name: String,
}

#[derive(Debug, Default)]
pub struct SynthonLibrary {
    synthons: HashMap<String, LoadedSynthon>,
    by_rule_role: HashMap<(String, String), Vec<String>>,
}

impl SynthonLibrary {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, synthon: LoadedSynthon) {
        let mut indexed_any = false;
        for tag in &synthon.reaction_tags {
            let Some((reaction_id, role_name)) = tag.split_once(':') else {
                continue;
            };
            self.insert_role_index(reaction_id, role_name, &synthon.synthon_id);
            indexed_any = true;
        }
        if !indexed_any {
            self.insert_role_index(
                &synthon.reaction_rule_id,
                &synthon.reaction_role,
                &synthon.synthon_id,
            );
        }
        self.synthons.insert(synthon.synthon_id.clone(), synthon);
    }

    fn insert_role_index(&mut self, reaction_id: &str, role_name: &str, synthon_id: &str) {
        let ids = self
            .by_rule_role
            .entry((reaction_id.to_owned(), role_name.to_owned()))
            .or_default();
        if !ids.iter().any(|value| value == synthon_id) {
            ids.push(synthon_id.to_owned());
        }
    }

    #[must_use]
    pub fn get(&self, synthon_id: &str) -> Option<&LoadedSynthon> {
        self.synthons.get(synthon_id)
    }

    #[must_use]
    pub fn compatible_ids(&self, rule_id: &str, role: &str) -> &[String] {
        self.by_rule_role
            .get(&(rule_id.to_owned(), role.to_owned()))
            .map_or(&[], Vec::as_slice)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.synthons.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.synthons.is_empty()
    }

    pub fn ids(&self) -> impl Iterator<Item = &str> {
        self.synthons.keys().map(String::as_str)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AssemblyReactionRule {
    pub rule_id: String,
    pub synthon_a_role: String,
    pub synthon_b_role: String,
    pub bond_length_a: f32,
    pub dihedral_omega_rad: DihedralConstraint,
}

impl AssemblyReactionRule {
    #[must_use]
    pub fn amide_coupling() -> Self {
        Self {
            rule_id: "RXN_AMIDE_COUPLING".to_owned(),
            synthon_a_role: "acid".to_owned(),
            synthon_b_role: "amine".to_owned(),
            bond_length_a: 1.33,
            dihedral_omega_rad: DihedralConstraint::Fixed(core::f32::consts::PI),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Product3D {
    pub product_id: String,
    pub smiles: String,
    pub coordinates: Vec<f32>,
    pub charges: Vec<f32>,
    pub bonds: Vec<(usize, usize)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ZMatrixAssemblyRule {
    pub scaffold_reference_atom_1: usize,
    pub scaffold_reference_atom_2: usize,
    pub bond_length_a: f32,
    pub bond_angle_deg: f32,
    pub dihedral_deg: f32,
    pub hybridization_model: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ZMatrixAssemblyReport {
    pub coordinates: Vec<f32>,
    pub sampled_dihedral_deg: f32,
    pub bond_length_a: f32,
    pub bond_angle_deg: f32,
    pub hybridization_model: String,
    pub z_matrix_active: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SmartsAssemblyMetadata {
    pub reaction_id: String,
    pub scaffold_role: String,
    pub synthon_role: String,
    pub scaffold_reactive_atom_idx: usize,
    pub synthon_reactive_atom_idx: usize,
    pub removed_leaving_group_atom_indices: Vec<usize>,
    pub product_bond_length_a: f32,
    pub product_bond_angle_deg: f32,
    pub selected_dihedral_deg: f32,
    pub assembly_mode: String,
    pub z_matrix_active: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SmartsZMatrixProduct {
    pub state: ScaffoldState3D,
    pub metadata: SmartsAssemblyMetadata,
    pub scaffold_index_map: Vec<Option<usize>>,
    pub synthon_index_map: Vec<Option<usize>>,
}

impl Product3D {
    #[must_use]
    pub fn atom_count(&self) -> usize {
        self.charges.len()
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum KinematicAssemblyError {
    #[error("{0}")]
    InvalidState(#[from] MolecularStateError),
    #[error("synthon id {0} was not present in the in-memory library")]
    MissingSynthon(String),
    #[error("synthon {synthon_id} is incompatible with {rule_id}:{expected_role}")]
    IncompatibleSynthon {
        synthon_id: String,
        rule_id: String,
        expected_role: String,
    },
    #[error("reaction bond length must be positive and finite, got {0}")]
    InvalidBondLength(f32),
    #[error("{label} vector is zero length")]
    ZeroVector { label: &'static str },
    #[error("assembly generated a non-finite coordinate")]
    NonFiniteCoordinate,
    #[error("invalid SMARTS assembly plan: {0}")]
    InvalidReactionPlan(String),
}

pub fn assemble_product_from_library(
    library: &SynthonLibrary,
    rule: &AssemblyReactionRule,
    synthon_a_id: &str,
    synthon_b_id: &str,
) -> Result<Product3D, KinematicAssemblyError> {
    let synthon_a = library
        .get(synthon_a_id)
        .ok_or_else(|| KinematicAssemblyError::MissingSynthon(synthon_a_id.to_owned()))?;
    let synthon_b = library
        .get(synthon_b_id)
        .ok_or_else(|| KinematicAssemblyError::MissingSynthon(synthon_b_id.to_owned()))?;
    assemble_pair(rule, synthon_a, synthon_b)
}

pub fn assemble_pair(
    rule: &AssemblyReactionRule,
    synthon_a: &LoadedSynthon,
    synthon_b: &LoadedSynthon,
) -> Result<Product3D, KinematicAssemblyError> {
    validate_participant(rule, synthon_a, &rule.synthon_a_role)?;
    validate_participant(rule, synthon_b, &rule.synthon_b_role)?;
    if !rule.bond_length_a.is_finite() || rule.bond_length_a <= 0.0 {
        return Err(KinematicAssemblyError::InvalidBondLength(
            rule.bond_length_a,
        ));
    }

    let a_attachment = synthon_a.attachment();
    let b_attachment = synthon_b.attachment();
    let a_anchor = synthon_a.synthon.atom_xyz(a_attachment.atom_index);
    let b_anchor = synthon_b.synthon.atom_xyz(b_attachment.atom_index);
    let a_exit = normalize(a_attachment.attachment_vector, "synthon A exit vector")?;
    let b_attach = normalize(
        b_attachment.attachment_vector,
        "synthon B attachment vector",
    )?;

    let rotation_to_exit = rotation_between(b_attach, mul(a_exit, -1.0))?;
    let rotated_b_anchor = mat_vec(rotation_to_exit, b_anchor);
    let ideal_b_anchor = add(a_anchor, mul(a_exit, rule.bond_length_a));
    let translation = sub(ideal_b_anchor, rotated_b_anchor);

    let mut transformed_b = Vec::with_capacity(synthon_b.synthon.coordinates.len());
    for atom_index in 0..synthon_b.synthon.atom_count() {
        let rotated = mat_vec(rotation_to_exit, synthon_b.synthon.atom_xyz(atom_index));
        transformed_b.extend_from_slice(&add(rotated, translation));
    }

    if let DihedralConstraint::Fixed(target_rad) = rule.dihedral_omega_rad {
        if let (Some(a_ref), Some(b_ref)) = (
            a_attachment.dihedral_reference_atom_index,
            b_attachment.dihedral_reference_atom_index,
        ) {
            let current = dihedral(
                synthon_a.synthon.atom_xyz(a_ref),
                a_anchor,
                ideal_b_anchor,
                read_xyz(&transformed_b, b_ref),
            )?;
            rotate_in_place(
                &mut transformed_b,
                ideal_b_anchor,
                a_exit,
                target_rad - current,
            )?;
        } else {
            rotate_in_place(&mut transformed_b, ideal_b_anchor, a_exit, target_rad)?;
        }
    }

    let mut product_coordinates =
        Vec::with_capacity(synthon_a.synthon.coordinates.len() + transformed_b.len());
    let mut product_charges =
        Vec::with_capacity(synthon_a.synthon.charges.len() + synthon_b.synthon.charges.len());
    let a_map = append_atoms(
        &synthon_a.synthon.coordinates,
        &synthon_a.synthon.charges,
        a_attachment.leaving_group_atom_index,
        &mut product_coordinates,
        &mut product_charges,
    );
    let b_map = append_atoms(
        &transformed_b,
        &synthon_b.synthon.charges,
        b_attachment.leaving_group_atom_index,
        &mut product_coordinates,
        &mut product_charges,
    );

    if let Some(product_idx) = a_map[a_attachment.atom_index] {
        product_charges[product_idx] -= synthon_a.leaving_group_formal_charge;
    }
    if let Some(product_idx) = b_map[b_attachment.atom_index] {
        product_charges[product_idx] -= synthon_b.leaving_group_formal_charge;
    }
    if product_coordinates.iter().any(|value| !value.is_finite()) {
        return Err(KinematicAssemblyError::NonFiniteCoordinate);
    }

    Ok(Product3D {
        product_id: format!(
            "{}__{}__{}",
            rule.rule_id, synthon_a.synthon_id, synthon_b.synthon_id
        ),
        smiles: format!(
            "{}.{}>>{}",
            synthon_a.canonical_smiles, synthon_b.canonical_smiles, rule.rule_id
        ),
        coordinates: product_coordinates,
        charges: product_charges,
        bonds: Vec::new(),
    })
}

#[allow(clippy::too_many_arguments)]
pub fn zmatrix_attach_fragment(
    scaffold_exit_atom: [f32; 3],
    scaffold_reference_atom_1: [f32; 3],
    scaffold_reference_atom_2: [f32; 3],
    fragment_attachment_atom: [f32; 3],
    fragment_reference_atom: [f32; 3],
    fragment_coordinates: &[f32],
    bond_length_a: f32,
    bond_angle_deg: f32,
    dihedral_deg: f32,
    hybridization_model: &str,
) -> Result<ZMatrixAssemblyReport, KinematicAssemblyError> {
    if !bond_length_a.is_finite() || bond_length_a <= 0.0 {
        return Err(KinematicAssemblyError::InvalidBondLength(bond_length_a));
    }
    let target_attachment = internal_coordinate_to_cartesian(
        scaffold_reference_atom_2,
        scaffold_reference_atom_1,
        scaffold_exit_atom,
        bond_length_a,
        bond_angle_deg.to_radians(),
        dihedral_deg.to_radians(),
    )?;
    let target_bond_axis = normalize(
        sub(target_attachment, scaffold_exit_atom),
        "z-matrix target bond axis",
    )?;
    let fragment_axis = normalize(
        sub(fragment_reference_atom, fragment_attachment_atom),
        "fragment attachment-reference axis",
    )?;
    let rotation = rotation_between(fragment_axis, target_bond_axis)?;
    let rotated_attachment = mat_vec(rotation, fragment_attachment_atom);
    let translation = sub(target_attachment, rotated_attachment);
    let mut transformed = Vec::with_capacity(fragment_coordinates.len());
    for atom_index in 0..(fragment_coordinates.len() / 3) {
        let rotated = mat_vec(rotation, read_xyz(fragment_coordinates, atom_index));
        transformed.extend_from_slice(&add(rotated, translation));
    }
    rotate_in_place(
        &mut transformed,
        target_attachment,
        target_bond_axis,
        dihedral_deg.to_radians(),
    )?;
    Ok(ZMatrixAssemblyReport {
        coordinates: transformed,
        sampled_dihedral_deg: dihedral_deg,
        bond_length_a,
        bond_angle_deg,
        hybridization_model: hybridization_model.to_owned(),
        z_matrix_active: true,
    })
}

pub fn execute_zmatrix_reaction(
    scaffold: &ScaffoldState3D,
    synthon: &Synthon3D,
    scaffold_attachment_idx: usize,
    synthon_attachment_idx: usize,
    rule: &ZMatrixAssemblyRule,
) -> Result<ScaffoldState3D, KinematicAssemblyError> {
    let scaffold_attachment = scaffold
        .attachment_points
        .get(scaffold_attachment_idx)
        .ok_or(KinematicAssemblyError::MissingSynthon(format!(
            "scaffold_attachment_{scaffold_attachment_idx}"
        )))?;
    let synthon_attachment = synthon
        .attachment_points
        .get(synthon_attachment_idx)
        .ok_or(KinematicAssemblyError::MissingSynthon(format!(
            "synthon_attachment_{synthon_attachment_idx}"
        )))?;
    let fragment_reference_atom = synthon_attachment
        .dihedral_reference_atom_index
        .unwrap_or_else(|| first_non_attachment_atom(synthon, synthon_attachment.atom_index));
    let report = zmatrix_attach_fragment(
        scaffold.atom_xyz(scaffold_attachment.atom_index),
        scaffold.atom_xyz(rule.scaffold_reference_atom_1),
        scaffold.atom_xyz(rule.scaffold_reference_atom_2),
        synthon.atom_xyz(synthon_attachment.atom_index),
        synthon.atom_xyz(fragment_reference_atom),
        &synthon.coordinates,
        rule.bond_length_a,
        rule.bond_angle_deg,
        rule.dihedral_deg,
        &rule.hybridization_model,
    )?;
    let scaffold_skip = scaffold_attachment.leaving_group_atom_index;
    let synthon_skip = synthon_attachment.leaving_group_atom_index;
    let mut product_coordinates =
        Vec::with_capacity(scaffold.coordinates.len() + report.coordinates.len());
    let mut product_charges = Vec::with_capacity(scaffold.charges.len() + synthon.charges.len());
    let scaffold_index_map = append_atoms(
        &scaffold.coordinates,
        &scaffold.charges,
        scaffold_skip,
        &mut product_coordinates,
        &mut product_charges,
    );
    let synthon_index_map = append_atoms(
        &report.coordinates,
        &synthon.charges,
        synthon_skip,
        &mut product_coordinates,
        &mut product_charges,
    );
    if product_coordinates.iter().any(|value| !value.is_finite()) {
        return Err(KinematicAssemblyError::NonFiniteCoordinate);
    }
    let mut product_bonds = remap_bonds(&scaffold.bonds, &scaffold_index_map);
    for bond in remap_bonds(&synthon.bonds, &synthon_index_map) {
        push_unique_bond(&mut product_bonds, bond.0, bond.1);
    }
    if let (Some(lhs), Some(rhs)) = (
        scaffold_index_map
            .get(scaffold_attachment.atom_index)
            .copied()
            .flatten(),
        synthon_index_map
            .get(synthon_attachment.atom_index)
            .copied()
            .flatten(),
    ) {
        push_unique_bond(&mut product_bonds, lhs, rhs);
    }
    #[cfg(debug_assertions)]
    assert_topology_integrity(
        scaffold,
        synthon,
        &scaffold_index_map,
        &synthon_index_map,
        &product_bonds,
        product_charges.len(),
        scaffold_attachment.atom_index,
        synthon_attachment.atom_index,
        scaffold_skip,
        synthon_skip,
    );
    ScaffoldState3D::new_with_bonds(
        product_coordinates,
        product_charges,
        Vec::new(),
        product_bonds,
    )
    .map_err(Into::into)
}

pub fn execute_smarts_zmatrix_reaction(
    scaffold: &ScaffoldState3D,
    synthon: &Synthon3D,
    plan: &AssemblyPlan,
    reaction_rule: &RegistryReactionRule,
) -> Result<SmartsZMatrixProduct, KinematicAssemblyError> {
    if plan.reaction_id != reaction_rule.reaction_id {
        return Err(KinematicAssemblyError::InvalidReactionPlan(format!(
            "plan reaction_id {} does not match rule {}",
            plan.reaction_id, reaction_rule.reaction_id
        )));
    }
    validate_atom_index(
        scaffold.atom_count(),
        plan.scaffold_reactive_atom_idx,
        "scaffold reactive atom",
    )?;
    validate_atom_index(
        scaffold.atom_count(),
        plan.scaffold_reference_atom_1,
        "scaffold reference atom 1",
    )?;
    validate_atom_index(
        scaffold.atom_count(),
        plan.scaffold_reference_atom_2,
        "scaffold reference atom 2",
    )?;
    validate_atom_index(
        synthon.atom_count(),
        plan.synthon_reactive_atom_idx,
        "synthon reactive atom",
    )?;
    validate_atom_index(
        synthon.atom_count(),
        plan.synthon_reference_atom_idx,
        "synthon reference atom",
    )?;
    if !reaction_rule.product_bond.ideal_bond_length_a.is_finite()
        || reaction_rule.product_bond.ideal_bond_length_a <= 0.0
    {
        return Err(KinematicAssemblyError::InvalidBondLength(
            reaction_rule.product_bond.ideal_bond_length_a,
        ));
    }
    let report = zmatrix_attach_fragment(
        scaffold.atom_xyz(plan.scaffold_reactive_atom_idx),
        scaffold.atom_xyz(plan.scaffold_reference_atom_1),
        scaffold.atom_xyz(plan.scaffold_reference_atom_2),
        synthon.atom_xyz(plan.synthon_reactive_atom_idx),
        synthon.atom_xyz(plan.synthon_reference_atom_idx),
        &synthon.coordinates,
        reaction_rule.product_bond.ideal_bond_length_a,
        reaction_rule.product_bond.ideal_bond_angle_deg,
        plan.selected_dihedral_deg,
        "smarts_zmatrix",
    )?;
    let scaffold_skip = plan
        .scaffold_leaving_group_atom_indices
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    let synthon_skip = plan
        .synthon_leaving_group_atom_indices
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    let mut product_coordinates =
        Vec::with_capacity(scaffold.coordinates.len() + report.coordinates.len());
    let mut product_charges = Vec::with_capacity(scaffold.charges.len() + synthon.charges.len());
    let scaffold_index_map = append_atoms_skipping(
        &scaffold.coordinates,
        &scaffold.charges,
        &scaffold_skip,
        &mut product_coordinates,
        &mut product_charges,
    );
    let synthon_index_map = append_atoms_skipping(
        &report.coordinates,
        &synthon.charges,
        &synthon_skip,
        &mut product_coordinates,
        &mut product_charges,
    );
    if product_coordinates.iter().any(|value| !value.is_finite()) {
        return Err(KinematicAssemblyError::NonFiniteCoordinate);
    }
    let scaffold_product_idx = scaffold_index_map
        .get(plan.scaffold_reactive_atom_idx)
        .copied()
        .flatten()
        .ok_or_else(|| {
            KinematicAssemblyError::InvalidReactionPlan(
                "scaffold reactive atom was removed as a leaving group".to_owned(),
            )
        })?;
    let synthon_product_idx = synthon_index_map
        .get(plan.synthon_reactive_atom_idx)
        .copied()
        .flatten()
        .ok_or_else(|| {
            KinematicAssemblyError::InvalidReactionPlan(
                "synthon reactive atom was removed as a leaving group".to_owned(),
            )
        })?;
    let mut product_bonds = remap_bonds(&scaffold.bonds, &scaffold_index_map);
    for bond in remap_bonds(&synthon.bonds, &synthon_index_map) {
        push_unique_bond(&mut product_bonds, bond.0, bond.1);
    }
    push_unique_bond(
        &mut product_bonds,
        scaffold_product_idx,
        synthon_product_idx,
    );
    #[cfg(debug_assertions)]
    assert_topology_integrity(
        scaffold,
        synthon,
        &scaffold_index_map,
        &synthon_index_map,
        &product_bonds,
        product_charges.len(),
        plan.scaffold_reactive_atom_idx,
        plan.synthon_reactive_atom_idx,
        plan.scaffold_leaving_group_atom_indices.iter().copied(),
        plan.synthon_leaving_group_atom_indices.iter().copied(),
    );
    let mut removed = plan.scaffold_leaving_group_atom_indices.clone();
    removed.extend(plan.synthon_leaving_group_atom_indices.iter().copied());
    let metadata = SmartsAssemblyMetadata {
        reaction_id: reaction_rule.reaction_id.clone(),
        scaffold_role: plan.scaffold_role.clone(),
        synthon_role: plan.synthon_role.clone(),
        scaffold_reactive_atom_idx: plan.scaffold_reactive_atom_idx,
        synthon_reactive_atom_idx: plan.synthon_reactive_atom_idx,
        removed_leaving_group_atom_indices: removed,
        product_bond_length_a: reaction_rule.product_bond.ideal_bond_length_a,
        product_bond_angle_deg: reaction_rule.product_bond.ideal_bond_angle_deg,
        selected_dihedral_deg: plan.selected_dihedral_deg,
        assembly_mode: "smarts_zmatrix".to_owned(),
        z_matrix_active: true,
    };
    let state = ScaffoldState3D::new_with_bonds(
        product_coordinates,
        product_charges,
        Vec::new(),
        product_bonds,
    )?;
    Ok(SmartsZMatrixProduct {
        state,
        metadata,
        scaffold_index_map,
        synthon_index_map,
    })
}

fn remap_bonds(bonds: &[(usize, usize)], index_map: &[Option<usize>]) -> Vec<(usize, usize)> {
    let mut product_bonds = Vec::new();
    for (lhs, rhs) in bonds {
        let Some(new_lhs) = index_map.get(*lhs).copied().flatten() else {
            continue;
        };
        let Some(new_rhs) = index_map.get(*rhs).copied().flatten() else {
            continue;
        };
        push_unique_bond(&mut product_bonds, new_lhs, new_rhs);
    }
    product_bonds
}

fn push_unique_bond(bonds: &mut Vec<(usize, usize)>, lhs: usize, rhs: usize) {
    if lhs == rhs {
        return;
    }
    let bond = if lhs < rhs { (lhs, rhs) } else { (rhs, lhs) };
    if !bonds.contains(&bond) {
        bonds.push(bond);
    }
}

#[cfg(debug_assertions)]
#[allow(clippy::too_many_arguments)]
fn assert_topology_integrity<I, J>(
    scaffold: &ScaffoldState3D,
    synthon: &Synthon3D,
    scaffold_index_map: &[Option<usize>],
    synthon_index_map: &[Option<usize>],
    product_bonds: &[(usize, usize)],
    product_atom_count: usize,
    scaffold_reactive_atom_idx: usize,
    synthon_reactive_atom_idx: usize,
    scaffold_leaving_groups: I,
    synthon_leaving_groups: J,
) where
    I: IntoIterator<Item = usize>,
    J: IntoIterator<Item = usize>,
{
    let scaffold_leaving_groups: Vec<usize> = scaffold_leaving_groups.into_iter().collect();
    let synthon_leaving_groups: Vec<usize> = synthon_leaving_groups.into_iter().collect();
    for atom_index in &scaffold_leaving_groups {
        assert!(
            scaffold_index_map
                .get(*atom_index)
                .copied()
                .flatten()
                .is_none(),
            "scaffold leaving-group atom {atom_index} survived topology remap"
        );
    }
    for atom_index in &synthon_leaving_groups {
        assert!(
            synthon_index_map
                .get(*atom_index)
                .copied()
                .flatten()
                .is_none(),
            "synthon leaving-group atom {atom_index} survived topology remap"
        );
    }

    let retained_scaffold_atoms = scaffold_index_map
        .iter()
        .filter(|value| value.is_some())
        .count();
    let retained_synthon_atoms = synthon_index_map
        .iter()
        .filter(|value| value.is_some())
        .count();
    assert_eq!(
        product_atom_count,
        retained_scaffold_atoms + retained_synthon_atoms,
        "product atom count does not match retained scaffold + retained synthon atoms"
    );
    assert_eq!(
        product_atom_count,
        scaffold.atom_count() + synthon.atom_count()
            - scaffold_leaving_groups.len()
            - synthon_leaving_groups.len(),
        "product atom count does not match leaving-group deletion arithmetic"
    );

    let scaffold_product_idx = scaffold_index_map
        .get(scaffold_reactive_atom_idx)
        .copied()
        .flatten()
        .expect("scaffold reactive atom was removed before topology assertion");
    let synthon_product_idx = synthon_index_map
        .get(synthon_reactive_atom_idx)
        .copied()
        .flatten()
        .expect("synthon reactive atom was removed before topology assertion");
    let expected_bond = if scaffold_product_idx < synthon_product_idx {
        (scaffold_product_idx, synthon_product_idx)
    } else {
        (synthon_product_idx, scaffold_product_idx)
    };
    assert!(
        product_bonds.contains(&expected_bond),
        "new covalent reaction bond {expected_bond:?} missing from product topology"
    );

    let synthon_topology_complete = retained_synthon_atoms <= 1 || !synthon.bonds.is_empty();
    let scaffold_topology_complete = retained_scaffold_atoms <= 1 || !scaffold.bonds.is_empty();
    if scaffold_topology_complete && synthon_topology_complete {
        assert_eq!(
            count_connected_components(product_atom_count, product_bonds),
            1,
            "assembled product topology is disconnected"
        );
    }
}

#[cfg(any(test, debug_assertions))]
fn count_connected_components(atom_count: usize, bonds: &[(usize, usize)]) -> usize {
    if atom_count == 0 {
        return 0;
    }
    let mut adjacency = vec![Vec::new(); atom_count];
    for (lhs, rhs) in bonds {
        if *lhs >= atom_count || *rhs >= atom_count || lhs == rhs {
            continue;
        }
        adjacency[*lhs].push(*rhs);
        adjacency[*rhs].push(*lhs);
    }
    let mut visited = vec![false; atom_count];
    let mut components = 0;
    for atom_index in 0..atom_count {
        if visited[atom_index] {
            continue;
        }
        components += 1;
        let mut queue = VecDeque::from([atom_index]);
        visited[atom_index] = true;
        while let Some(current) = queue.pop_front() {
            for neighbor in &adjacency[current] {
                if !visited[*neighbor] {
                    visited[*neighbor] = true;
                    queue.push_back(*neighbor);
                }
            }
        }
    }
    components
}

fn validate_atom_index(
    atom_count: usize,
    atom_index: usize,
    label: &str,
) -> Result<(), KinematicAssemblyError> {
    if atom_index >= atom_count {
        return Err(KinematicAssemblyError::InvalidReactionPlan(format!(
            "{label} index {atom_index} out of bounds for atom_count {atom_count}"
        )));
    }
    Ok(())
}

fn first_non_attachment_atom(synthon: &Synthon3D, attachment_atom: usize) -> usize {
    (0..synthon.atom_count())
        .find(|atom_index| *atom_index != attachment_atom)
        .unwrap_or(attachment_atom)
}

fn internal_coordinate_to_cartesian(
    atom_a: [f32; 3],
    atom_b: [f32; 3],
    atom_c: [f32; 3],
    bond_length_a: f32,
    bond_angle_rad: f32,
    dihedral_rad: f32,
) -> Result<[f32; 3], KinematicAssemblyError> {
    let bc = normalize(sub(atom_c, atom_b), "z-matrix BC axis")?;
    let ba = normalize(sub(atom_a, atom_b), "z-matrix BA axis")?;
    let normal = normalize(cross(ba, bc), "z-matrix plane normal")?;
    let binormal = normalize(cross(normal, bc), "z-matrix binormal")?;
    let term_axis = mul(bc, -bond_angle_rad.cos());
    let term_binormal = mul(binormal, bond_angle_rad.sin() * dihedral_rad.cos());
    let term_normal = mul(normal, bond_angle_rad.sin() * dihedral_rad.sin());
    Ok(add(
        atom_c,
        mul(
            add(add(term_axis, term_binormal), term_normal),
            bond_length_a,
        ),
    ))
}

fn validate_participant(
    rule: &AssemblyReactionRule,
    synthon: &LoadedSynthon,
    expected_role: &str,
) -> Result<(), KinematicAssemblyError> {
    if synthon.reaction_rule_id != rule.rule_id || synthon.reaction_role != expected_role {
        return Err(KinematicAssemblyError::IncompatibleSynthon {
            synthon_id: synthon.synthon_id.clone(),
            rule_id: rule.rule_id.clone(),
            expected_role: expected_role.to_owned(),
        });
    }
    Ok(())
}

fn append_atoms(
    coordinates: &[f32],
    charges: &[f32],
    skip_atom: Option<usize>,
    product_coordinates: &mut Vec<f32>,
    product_charges: &mut Vec<f32>,
) -> Vec<Option<usize>> {
    let mut index_map = vec![None; charges.len()];
    for (atom_index, charge) in charges.iter().copied().enumerate() {
        if skip_atom == Some(atom_index) {
            continue;
        }
        let product_index = product_charges.len();
        index_map[atom_index] = Some(product_index);
        let offset = atom_index * 3;
        product_coordinates.extend_from_slice(&coordinates[offset..offset + 3]);
        product_charges.push(charge);
    }
    index_map
}

fn append_atoms_skipping(
    coordinates: &[f32],
    charges: &[f32],
    skip_atoms: &std::collections::HashSet<usize>,
    product_coordinates: &mut Vec<f32>,
    product_charges: &mut Vec<f32>,
) -> Vec<Option<usize>> {
    let mut index_map = vec![None; charges.len()];
    for (atom_index, charge) in charges.iter().copied().enumerate() {
        if skip_atoms.contains(&atom_index) {
            continue;
        }
        let product_index = product_charges.len();
        index_map[atom_index] = Some(product_index);
        let offset = atom_index * 3;
        product_coordinates.extend_from_slice(&coordinates[offset..offset + 3]);
        product_charges.push(charge);
    }
    index_map
}

pub fn load_synthon_library_from_parquet(path: &Path) -> Result<SynthonLibrary> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("read parquet metadata {}", path.display()))?
        .build()?;
    let mut library = SynthonLibrary::new();
    for batch in &mut reader {
        append_synthon_batch(&mut library, &batch?)?;
    }
    Ok(library)
}

fn append_synthon_batch(library: &mut SynthonLibrary, batch: &RecordBatch) -> Result<()> {
    if batch.column_by_name("anchor_id").is_some() && batch.column_by_name("synthon_id").is_none() {
        return append_calibration_anchor_batch(library, batch);
    }
    for row in 0..batch.num_rows() {
        if batch.column_by_name("ingest_status").is_some()
            && string_value(batch, "ingest_status", row)? != "ok"
        {
            continue;
        }
        let (coordinates, atomic_numbers) =
            parse_conformer_atoms(&string_value(batch, "conformer_atoms_json", row)?)?;
        let charges = normalize_charge_count(
            parse_f32_list(&string_value(batch, "partial_charges_json", row)?)?,
            atomic_numbers.len(),
        );
        let attachment_vector = parse_vec3(&string_value(batch, "attachment_vector_json", row)?)?;
        let attachment = AttachmentPoint::new(
            usize_value(batch, "attachment_atom_idx", row)?,
            optional_usize_value(batch, "leaving_group_atom_idx", row)?,
            attachment_vector,
            optional_usize_value(batch, "dihedral_reference_atom_idx", row)?,
        );
        let synthon = Synthon3D::new(coordinates, charges, vec![attachment])?;
        let reaction_tags = parse_reaction_tags(&string_value(batch, "reaction_tags_json", row)?)?;
        let reaction_sites =
            parse_reaction_sites(&string_value(batch, "reaction_match_atoms_json", row)?)?;
        let leaving_group_sites =
            parse_leaving_group_sites(&string_value(batch, "leaving_group_atoms_json", row)?)?;
        library.insert(LoadedSynthon {
            synthon_id: string_value(batch, "synthon_id", row)?,
            canonical_smiles: string_value(batch, "canonical_smiles", row)?,
            reaction_rule_id: string_value(batch, "reaction_rule_id", row)?,
            reaction_role: string_value(batch, "reaction_role", row)?,
            reaction_tags,
            reaction_sites,
            leaving_group_sites,
            synthon,
            atomic_numbers,
            leaving_group_formal_charge: f32_value(batch, "leaving_group_formal_charge", row)?,
        });
    }
    Ok(())
}

fn normalize_charge_count(mut charges: Vec<f32>, atom_count: usize) -> Vec<f32> {
    if charges.len() > atom_count {
        charges.truncate(atom_count);
    }
    if charges.len() < atom_count {
        charges.resize(atom_count, 0.0);
    }
    charges
}

fn append_calibration_anchor_batch(
    library: &mut SynthonLibrary,
    batch: &RecordBatch,
) -> Result<()> {
    for row in 0..batch.num_rows() {
        if batch.column_by_name("generation_status").is_some()
            && string_value(batch, "generation_status", row)? != "ok"
        {
            continue;
        }
        let (coordinates, atomic_numbers) =
            parse_conformer_atoms(&string_value(batch, "conformer_atoms_json", row)?)?;
        let charges = normalize_charge_count(
            parse_f32_list(&string_value(batch, "partial_charges_json", row)?)?,
            atomic_numbers.len(),
        );
        let attachment_atom = first_heavy_atom_index(&atomic_numbers).unwrap_or(0);
        let reference_atom =
            first_reference_atom_index(&atomic_numbers, attachment_atom).unwrap_or(attachment_atom);
        let attachment_vector = if reference_atom == attachment_atom {
            [1.0, 0.0, 0.0]
        } else {
            normalize(
                sub(
                    read_xyz(&coordinates, reference_atom),
                    read_xyz(&coordinates, attachment_atom),
                ),
                "calibration anchor attachment vector",
            )
            .map_err(|err| anyhow!("{err}"))?
        };
        let attachment = AttachmentPoint::new(
            attachment_atom,
            None,
            attachment_vector,
            (reference_atom != attachment_atom).then_some(reference_atom),
        );
        let synthon = Synthon3D::new(coordinates, charges, vec![attachment])?;
        library.insert(LoadedSynthon {
            synthon_id: string_value(batch, "anchor_id", row)?,
            canonical_smiles: string_value(batch, "canonical_smiles", row)?,
            reaction_rule_id: "RXN_AMIDE_COUPLING".to_owned(),
            reaction_role: "acid".to_owned(),
            reaction_tags: vec!["RXN_AMIDE_COUPLING:acid".to_owned()],
            reaction_sites: Vec::new(),
            leaving_group_sites: Vec::new(),
            synthon,
            atomic_numbers,
            leaving_group_formal_charge: 0.0,
        });
    }
    Ok(())
}

fn first_heavy_atom_index(atomic_numbers: &[u8]) -> Option<usize> {
    atomic_numbers
        .iter()
        .position(|atomic_number| *atomic_number > 1)
}

fn first_reference_atom_index(atomic_numbers: &[u8], attachment_atom: usize) -> Option<usize> {
    atomic_numbers
        .iter()
        .enumerate()
        .find_map(|(atom_index, atomic_number)| {
            (*atomic_number > 1 && atom_index != attachment_atom).then_some(atom_index)
        })
}

fn parse_conformer_atoms(value: &str) -> Result<(Vec<f32>, Vec<u8>)> {
    let rows: Vec<ConformerAtomRow> = serde_json::from_str(value)?;
    let mut coordinates = Vec::with_capacity(rows.len() * 3);
    let mut atomic_numbers = Vec::with_capacity(rows.len());
    for row in rows {
        atomic_numbers.push(row.atomic_num);
        coordinates.extend_from_slice(&[row.x, row.y, row.z]);
    }
    Ok((coordinates, atomic_numbers))
}

fn parse_f32_list(value: &str) -> Result<Vec<f32>> {
    Ok(serde_json::from_str(value)?)
}

fn parse_vec3(value: &str) -> Result<[f32; 3]> {
    let values: Vec<f32> = serde_json::from_str(value)?;
    if values.len() != 3 {
        return Err(anyhow!(
            "attachment vector must contain exactly three floats"
        ));
    }
    Ok([values[0], values[1], values[2]])
}

fn parse_reaction_tags(value: &str) -> Result<Vec<String>> {
    Ok(serde_json::from_str(value)?)
}

fn parse_reaction_sites(value: &str) -> Result<Vec<SynthonReactionSite>> {
    Ok(serde_json::from_str(value)?)
}

fn parse_leaving_group_sites(value: &str) -> Result<Vec<SynthonLeavingGroupSite>> {
    Ok(serde_json::from_str(value)?)
}

fn string_value(batch: &RecordBatch, column: &str, row: usize) -> Result<String> {
    let array = batch
        .column_by_name(column)
        .ok_or_else(|| anyhow!("missing column {column}"))?;
    if let Some(strings) = array.as_any().downcast_ref::<StringArray>() {
        return Ok(strings.value(row).to_owned());
    }
    if let Some(strings) = array.as_any().downcast_ref::<LargeStringArray>() {
        return Ok(strings.value(row).to_owned());
    }
    Err(anyhow!("column {column} was not a string array"))
}

fn usize_value(batch: &RecordBatch, column: &str, row: usize) -> Result<usize> {
    let value = i64_value(batch, column, row)?;
    usize::try_from(value).with_context(|| format!("column {column} row {row} was negative"))
}

fn optional_usize_value(batch: &RecordBatch, column: &str, row: usize) -> Result<Option<usize>> {
    let array = batch
        .column_by_name(column)
        .ok_or_else(|| anyhow!("missing column {column}"))?;
    if array.is_null(row) {
        return Ok(None);
    }
    usize_value(batch, column, row).map(Some)
}

fn i64_value(batch: &RecordBatch, column: &str, row: usize) -> Result<i64> {
    let array = batch
        .column_by_name(column)
        .ok_or_else(|| anyhow!("missing column {column}"))?;
    if let Some(values) = array.as_any().downcast_ref::<Int64Array>() {
        return Ok(values.value(row));
    }
    if let Some(values) = array.as_any().downcast_ref::<Int32Array>() {
        return Ok(i64::from(values.value(row)));
    }
    Err(anyhow!("column {column} was not an integer array"))
}

fn f32_value(batch: &RecordBatch, column: &str, row: usize) -> Result<f32> {
    let array = batch
        .column_by_name(column)
        .ok_or_else(|| anyhow!("missing column {column}"))?;
    if let Some(values) = array.as_any().downcast_ref::<Float64Array>() {
        return Ok(values.value(row) as f32);
    }
    Err(anyhow!("column {column} was not a Float64 array"))
}

fn rotate_in_place(
    coordinates: &mut [f32],
    origin: [f32; 3],
    axis: [f32; 3],
    angle_rad: f32,
) -> Result<(), KinematicAssemblyError> {
    let rotation = axis_angle(axis, angle_rad)?;
    for xyz in coordinates.chunks_exact_mut(3) {
        let relative = sub([xyz[0], xyz[1], xyz[2]], origin);
        xyz.copy_from_slice(&add(mat_vec(rotation, relative), origin));
    }
    Ok(())
}

fn read_xyz(coordinates: &[f32], atom_index: usize) -> [f32; 3] {
    let offset = atom_index * 3;
    [
        coordinates[offset],
        coordinates[offset + 1],
        coordinates[offset + 2],
    ]
}

fn normalize(value: [f32; 3], label: &'static str) -> Result<[f32; 3], KinematicAssemblyError> {
    let norm = dot(value, value).sqrt();
    if norm <= EPSILON || !norm.is_finite() {
        return Err(KinematicAssemblyError::ZeroVector { label });
    }
    Ok(mul(value, 1.0 / norm))
}

fn rotation_between(from: [f32; 3], to: [f32; 3]) -> Result<[[f32; 3]; 3], KinematicAssemblyError> {
    let from = normalize(from, "rotation source vector")?;
    let to = normalize(to, "rotation target vector")?;
    let cross_value = cross(from, to);
    let sin_theta = dot(cross_value, cross_value).sqrt();
    let cos_theta = dot(from, to).clamp(-1.0, 1.0);
    if sin_theta <= EPSILON {
        if cos_theta > 0.0 {
            return Ok(identity());
        }
        let fallback = if from[0].abs() < 0.9 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        return axis_angle(
            normalize(cross(from, fallback), "anti-parallel rotation axis")?,
            core::f32::consts::PI,
        );
    }
    axis_angle(
        mul(cross_value, 1.0 / sin_theta),
        sin_theta.atan2(cos_theta),
    )
}

fn axis_angle(axis: [f32; 3], angle_rad: f32) -> Result<[[f32; 3]; 3], KinematicAssemblyError> {
    let axis = normalize(axis, "rotation axis")?;
    let c = angle_rad.cos();
    let s = angle_rad.sin();
    let one_c = 1.0 - c;
    let [x, y, z] = axis;
    Ok([
        [
            c + x * x * one_c,
            x * y * one_c - z * s,
            x * z * one_c + y * s,
        ],
        [
            y * x * one_c + z * s,
            c + y * y * one_c,
            y * z * one_c - x * s,
        ],
        [
            z * x * one_c - y * s,
            z * y * one_c + x * s,
            c + z * z * one_c,
        ],
    ])
}

fn dihedral(
    p0: [f32; 3],
    p1: [f32; 3],
    p2: [f32; 3],
    p3: [f32; 3],
) -> Result<f32, KinematicAssemblyError> {
    let b0 = normalize(sub(p0, p1), "dihedral b0")?;
    let b1 = normalize(sub(p2, p1), "dihedral b1")?;
    let b2 = normalize(sub(p3, p2), "dihedral b2")?;
    let v = sub(b0, mul(b1, dot(b0, b1)));
    let w = sub(b2, mul(b1, dot(b2, b1)));
    Ok(dot(cross(b1, v), w).atan2(dot(v, w)))
}

fn identity() -> [[f32; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

fn mat_vec(matrix: [[f32; 3]; 3], value: [f32; 3]) -> [f32; 3] {
    [
        matrix[0][0] * value[0] + matrix[0][1] * value[1] + matrix[0][2] * value[2],
        matrix[1][0] * value[0] + matrix[1][1] * value[1] + matrix[1][2] * value[2],
        matrix[2][0] * value[0] + matrix[2][1] * value[1] + matrix[2][2] * value[2],
    ]
}

fn add(lhs: [f32; 3], rhs: [f32; 3]) -> [f32; 3] {
    [lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]]
}

fn sub(lhs: [f32; 3], rhs: [f32; 3]) -> [f32; 3] {
    [lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]]
}

fn mul(value: [f32; 3], scale: f32) -> [f32; 3] {
    [value[0] * scale, value[1] * scale, value[2] * scale]
}

fn dot(lhs: [f32; 3], rhs: [f32; 3]) -> f32 {
    lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]
}

fn cross(lhs: [f32; 3], rhs: [f32; 3]) -> [f32; 3] {
    [
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reactions::reaction_registry::{
        ProductBondSpec, ReactantRole, ReactionGuards, ReactionProvenance, TorsionPolicy,
    };

    #[test]
    fn topology_roundtrip_smarts_zmatrix_deletes_leaving_groups_and_connects_product() -> Result<()>
    {
        let scaffold = ScaffoldState3D::new_with_bonds(
            vec![
                0.0, 1.0, 0.0, // reference
                0.0, 0.0, 0.0, // reactive
                -1.0, 0.0, 0.0, // leaving/reference
            ],
            vec![0.0, 0.0, 0.0],
            Vec::new(),
            vec![(0, 1), (1, 2)],
        )?;
        let synthon = Synthon3D::new_with_bonds(
            vec![
                0.0, 0.0, 0.0, // reactive
                1.0, 0.0, 0.0, // retained reference
                0.0, -1.0, 0.0, // leaving
            ],
            vec![0.0, 0.0, 0.0],
            Vec::new(),
            vec![(0, 1), (0, 2)],
        )?;
        let plan = AssemblyPlan {
            reaction_id: "RXN_TEST".to_owned(),
            scaffold_role: "scaffold".to_owned(),
            synthon_role: "synthon".to_owned(),
            scaffold_reactive_atom_idx: 1,
            synthon_reactive_atom_idx: 0,
            scaffold_reference_atom_1: 0,
            scaffold_reference_atom_2: 2,
            synthon_reference_atom_idx: 1,
            scaffold_leaving_group_atom_indices: vec![2],
            synthon_leaving_group_atom_indices: vec![2],
            selected_dihedral_deg: 60.0,
        };
        let rule = RegistryReactionRule {
            reaction_id: "RXN_TEST".to_owned(),
            reaction_name: "test".to_owned(),
            reaction_class: "unit".to_owned(),
            version: 1,
            enabled: true,
            epistemic_status: "PROJECTED_REACTION_GRAMMAR".to_owned(),
            smarts: "[C:1].[N:2]>>[C:1][N:2]".to_owned(),
            reactant_roles: HashMap::from([
                (
                    "scaffold".to_owned(),
                    ReactantRole {
                        required_smarts: "[C:1]".to_owned(),
                        reactive_atom_map: 1,
                        leaving_group_atom_maps: vec![99],
                        bond_vector_reference: vec![1],
                    },
                ),
                (
                    "synthon".to_owned(),
                    ReactantRole {
                        required_smarts: "[N:2]".to_owned(),
                        reactive_atom_map: 2,
                        leaving_group_atom_maps: vec![98],
                        bond_vector_reference: vec![2],
                    },
                ),
            ]),
            product_bond: ProductBondSpec {
                atom_map_a: 1,
                atom_map_b: 2,
                bond_order: 1,
                ideal_bond_length_a: 1.5,
                ideal_bond_angle_deg: 109.5,
                torsion_policy: TorsionPolicy {
                    mode: "discrete_grid".to_owned(),
                    dihedral_deg: vec![0.0, 60.0],
                },
            },
            guards: ReactionGuards {
                max_product_heavy_atoms: 64,
                allowed_formal_charge_range: [-2, 2],
                reject_radicals: true,
                reject_unmapped_reactive_atoms: true,
                require_single_match_or_enumerate_matches: "enumerate".to_owned(),
            },
            provenance: ReactionProvenance {
                source: "unit".to_owned(),
                notes: "topology roundtrip".to_owned(),
            },
        };

        let product = execute_smarts_zmatrix_reaction(&scaffold, &synthon, &plan, &rule)?;

        assert_eq!(product.state.atom_count(), 4);
        assert_eq!(product.scaffold_index_map[2], None);
        assert_eq!(product.synthon_index_map[2], None);
        assert!(product.state.bonds.contains(&(1, 2)));
        assert_eq!(
            count_connected_components(product.state.atom_count(), &product.state.bonds),
            1
        );
        assert_eq!(product.metadata.assembly_mode, "smarts_zmatrix");
        assert!(product.metadata.z_matrix_active);
        Ok(())
    }

    #[test]
    fn guard_leaving_group_deletion_and_atom_count_remain_active() -> Result<()> {
        topology_roundtrip_smarts_zmatrix_deletes_leaving_groups_and_connects_product()
    }
}
