use crate::core::synthon::{AttachmentPoint, MolecularStateError, ScaffoldState3D, Synthon3D};
use thiserror::Error;

const EPSILON: f32 = 1.0e-6;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DihedralConstraint {
    Fixed(f32),
    FreeRotation,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReactionKinematicRule {
    pub bond_length_a: f32,
    pub dihedral_omega_rad: DihedralConstraint,
}

#[derive(Debug, Error, PartialEq)]
pub enum ReactionKinematicError {
    #[error("{0}")]
    InvalidState(#[from] MolecularStateError),
    #[error("scaffold attachment index {index} is outside attachment point count {count}")]
    ScaffoldAttachmentOutOfRange { index: usize, count: usize },
    #[error("synthon attachment index {index} is outside attachment point count {count}")]
    SynthonAttachmentOutOfRange { index: usize, count: usize },
    #[error("reaction bond length must be positive and finite, got {0}")]
    InvalidBondLength(f32),
    #[error("{label} vector is zero length")]
    ZeroVector { label: &'static str },
    #[error("reaction generated a non-finite coordinate")]
    NonFiniteCoordinate,
}

pub fn execute_3d_reaction(
    scaffold: &ScaffoldState3D,
    synthon: &Synthon3D,
    scaffold_attachment_idx: usize,
    synthon_attachment_idx: usize,
    rule: ReactionKinematicRule,
) -> Result<ScaffoldState3D, ReactionKinematicError> {
    if !rule.bond_length_a.is_finite() || rule.bond_length_a <= 0.0 {
        return Err(ReactionKinematicError::InvalidBondLength(
            rule.bond_length_a,
        ));
    }
    let scaffold_attachment = scaffold
        .attachment_points
        .get(scaffold_attachment_idx)
        .ok_or(ReactionKinematicError::ScaffoldAttachmentOutOfRange {
            index: scaffold_attachment_idx,
            count: scaffold.attachment_points.len(),
        })?;
    let synthon_attachment = synthon
        .attachment_points
        .get(synthon_attachment_idx)
        .ok_or(ReactionKinematicError::SynthonAttachmentOutOfRange {
            index: synthon_attachment_idx,
            count: synthon.attachment_points.len(),
        })?;

    let scaffold_anchor = scaffold.atom_xyz(scaffold_attachment.atom_index);
    let scaffold_exit = normalize(
        scaffold_attachment.attachment_vector,
        "scaffold exit vector",
    )?;
    let synthon_vector = normalize(
        synthon_attachment.attachment_vector,
        "synthon attachment vector",
    )?;
    let rotation_to_exit = rotation_between(synthon_vector, mul(scaffold_exit, -1.0))?;

    let synthon_anchor = synthon.atom_xyz(synthon_attachment.atom_index);
    let transformed_anchor = mat_vec(rotation_to_exit, synthon_anchor);
    let desired_anchor = add(scaffold_anchor, mul(scaffold_exit, rule.bond_length_a));
    let translation = sub(desired_anchor, transformed_anchor);

    let mut transformed = Vec::with_capacity(synthon.coordinates.len());
    for atom_index in 0..synthon.atom_count() {
        let rotated = mat_vec(rotation_to_exit, synthon.atom_xyz(atom_index));
        let xyz = add(rotated, translation);
        transformed.extend_from_slice(&xyz);
    }

    if let DihedralConstraint::Fixed(target_rad) = rule.dihedral_omega_rad {
        if let (Some(scaffold_reference_idx), Some(synthon_reference_idx)) = (
            scaffold_attachment.dihedral_reference_atom_index,
            synthon_attachment.dihedral_reference_atom_index,
        ) {
            let scaffold_reference = scaffold.atom_xyz(scaffold_reference_idx);
            let synthon_anchor_after = read_xyz(&transformed, synthon_attachment.atom_index);
            let synthon_reference_after = read_xyz(&transformed, synthon_reference_idx);
            let current = dihedral(
                scaffold_reference,
                scaffold_anchor,
                synthon_anchor_after,
                synthon_reference_after,
            )?;
            let delta = target_rad - current;
            rotate_synthon_in_place(&mut transformed, desired_anchor, scaffold_exit, delta)?;
        } else {
            rotate_synthon_in_place(&mut transformed, desired_anchor, scaffold_exit, target_rad)?;
        }
    }

    let scaffold_skip = scaffold_attachment.leaving_group_atom_index;
    let synthon_skip = synthon_attachment.leaving_group_atom_index;
    let mut product_coordinates =
        Vec::with_capacity(scaffold.coordinates.len() + transformed.len());
    let mut product_charges = Vec::with_capacity(scaffold.charges.len() + synthon.charges.len());
    append_atoms(
        &scaffold.coordinates,
        &scaffold.charges,
        scaffold_skip,
        &mut product_coordinates,
        &mut product_charges,
    );
    append_atoms(
        &transformed,
        &synthon.charges,
        synthon_skip,
        &mut product_coordinates,
        &mut product_charges,
    );
    if product_coordinates.iter().any(|value| !value.is_finite()) {
        return Err(ReactionKinematicError::NonFiniteCoordinate);
    }
    ScaffoldState3D::new(product_coordinates, product_charges, Vec::new()).map_err(Into::into)
}

fn append_atoms(
    coordinates: &[f32],
    charges: &[f32],
    skip_atom: Option<usize>,
    product_coordinates: &mut Vec<f32>,
    product_charges: &mut Vec<f32>,
) {
    for (atom_index, charge) in charges.iter().copied().enumerate() {
        if skip_atom == Some(atom_index) {
            continue;
        }
        let offset = atom_index * 3;
        product_coordinates.extend_from_slice(&coordinates[offset..offset + 3]);
        product_charges.push(charge);
    }
}

fn rotate_synthon_in_place(
    coordinates: &mut [f32],
    origin: [f32; 3],
    axis: [f32; 3],
    angle_rad: f32,
) -> Result<(), ReactionKinematicError> {
    let rotation = axis_angle(axis, angle_rad)?;
    for xyz in coordinates.chunks_exact_mut(3) {
        let relative = sub([xyz[0], xyz[1], xyz[2]], origin);
        let rotated = add(mat_vec(rotation, relative), origin);
        xyz.copy_from_slice(&rotated);
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

fn normalize(value: [f32; 3], label: &'static str) -> Result<[f32; 3], ReactionKinematicError> {
    let norm = dot(value, value).sqrt();
    if norm <= EPSILON || !norm.is_finite() {
        return Err(ReactionKinematicError::ZeroVector { label });
    }
    Ok(mul(value, 1.0 / norm))
}

fn rotation_between(from: [f32; 3], to: [f32; 3]) -> Result<[[f32; 3]; 3], ReactionKinematicError> {
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
            normalize(cross(from, fallback), "anti-parallel axis")?,
            core::f32::consts::PI,
        );
    }
    axis_angle(
        mul(cross_value, 1.0 / sin_theta),
        sin_theta.atan2(cos_theta),
    )
}

fn axis_angle(axis: [f32; 3], angle_rad: f32) -> Result<[[f32; 3]; 3], ReactionKinematicError> {
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
) -> Result<f32, ReactionKinematicError> {
    let b0 = normalize(sub(p0, p1), "dihedral b0")?;
    let b1 = normalize(sub(p2, p1), "dihedral b1")?;
    let b2 = normalize(sub(p3, p2), "dihedral b2")?;
    let v = sub(b0, mul(b1, dot(b0, b1)));
    let w = sub(b2, mul(b1, dot(b2, b1)));
    let x = dot(v, w);
    let y = dot(cross(b1, v), w);
    Ok(y.atan2(x))
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

#[must_use]
pub fn attachment_point_from_vectors(
    atom_index: usize,
    leaving_group_atom_index: Option<usize>,
    vector: [f32; 3],
    dihedral_reference_atom_index: Option<usize>,
) -> AttachmentPoint {
    AttachmentPoint::new(
        atom_index,
        leaving_group_atom_index,
        vector,
        dihedral_reference_atom_index,
    )
}
