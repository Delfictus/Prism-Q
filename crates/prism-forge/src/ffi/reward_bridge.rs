use crate::{HitCount, VoxelIdx};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

const VARIANCE_STABLE_OCCUPIED: i8 = 1;
const VARIANCE_THERMALLY_ACTIVATED: i8 = 2;
const VARIANCE_THERMALLY_DESTABILIZED: i8 = 3;

#[pyfunction]
#[pyo3(signature = (
    coordinates,
    charges,
    origin_xyz,
    spacing_a,
    grid_dim,
    field_voxel_indices,
    field_variance_codes,
    beta_f=1.0,
    beta_s=1.0
))]
#[allow(clippy::too_many_arguments)]
pub fn compute_thermodynamic_reward_3d(
    coordinates: PyReadonlyArray2<'_, f32>,
    charges: PyReadonlyArray1<'_, f32>,
    origin_xyz: PyReadonlyArray1<'_, f32>,
    spacing_a: f32,
    grid_dim: u32,
    field_voxel_indices: PyReadonlyArray1<'_, u32>,
    field_variance_codes: PyReadonlyArray1<'_, i8>,
    beta_f: f64,
    beta_s: f64,
) -> PyResult<f64> {
    if !spacing_a.is_finite() || spacing_a <= 0.0 {
        return Err(PyValueError::new_err(
            "spacing_a must be positive and finite",
        ));
    }
    if grid_dim == 0 {
        return Err(PyValueError::new_err("grid_dim must be positive"));
    }
    if !beta_f.is_finite() || !beta_s.is_finite() {
        return Err(PyValueError::new_err("beta_f and beta_s must be finite"));
    }
    let coordinates = coordinates.as_slice().map_err(|_| {
        PyValueError::new_err("coordinates must be contiguous float32 NumPy storage")
    })?;
    if coordinates.len() % 3 != 0 {
        return Err(PyValueError::new_err("coordinates must have shape [N, 3]"));
    }
    let atom_count = coordinates.len() / 3;
    let charges = charges
        .as_slice()
        .map_err(|_| PyValueError::new_err("charges must be contiguous float32 NumPy storage"))?;
    if charges.len() != atom_count {
        return Err(PyValueError::new_err(
            "charges length must match coordinate atom count",
        ));
    }
    let origin = origin_xyz.as_slice().map_err(|_| {
        PyValueError::new_err("origin_xyz must be contiguous float32 NumPy storage")
    })?;
    if origin.len() != 3 {
        return Err(PyValueError::new_err(
            "origin_xyz must contain exactly 3 floats",
        ));
    }
    let field_voxels = field_voxel_indices.as_slice().map_err(|_| {
        PyValueError::new_err("field_voxel_indices must be contiguous uint32 NumPy storage")
    })?;
    let field_codes = field_variance_codes.as_slice().map_err(|_| {
        PyValueError::new_err("field_variance_codes must be contiguous int8 NumPy storage")
    })?;
    if field_voxels.len() != field_codes.len() {
        return Err(PyValueError::new_err(
            "field_voxel_indices and field_variance_codes must have matching lengths",
        ));
    }

    let field_lookup: HashMap<u32, i8> = field_voxels
        .iter()
        .copied()
        .zip(field_codes.iter().copied())
        .collect();

    let (pi_clash, pi_complement, mapped_atoms) = coordinates
        .par_chunks_exact(3)
        .enumerate()
        .filter_map(|(atom_index, xyz)| {
            coordinate_to_voxel(
                [xyz[0], xyz[1], xyz[2]],
                [origin[0], origin[1], origin[2]],
                spacing_a,
                grid_dim,
            )
            .map(|voxel_idx| (atom_index, voxel_idx))
        })
        .map(|(atom_index, voxel_idx)| {
            let voxel = VoxelIdx::new(voxel_idx);
            let charge_weight = 1.0_f64 + f64::from(charges[atom_index].abs()).min(2.0) * 0.05;
            match field_lookup.get(&voxel.get()).copied().unwrap_or_default() {
                VARIANCE_STABLE_OCCUPIED | VARIANCE_THERMALLY_DESTABILIZED => {
                    (charge_weight, 0.0_f64, 1_u32)
                }
                VARIANCE_THERMALLY_ACTIVATED => (0.0_f64, charge_weight, 1_u32),
                _ => (0.0_f64, 0.0_f64, 1_u32),
            }
        })
        .reduce(
            || (0.0_f64, 0.0_f64, 0_u32),
            |lhs, rhs| (lhs.0 + rhs.0, lhs.1 + rhs.1, lhs.2 + rhs.2),
        );

    let mapped = HitCount::new(mapped_atoms);
    let normalization = f64::from(mapped.get().max(1));
    let clash = pi_clash / normalization;
    let complement = pi_complement / normalization;
    let multiplier = (beta_f * clash).exp() * (-beta_s * complement).exp();
    if !multiplier.is_finite() || multiplier <= 0.0 {
        return Err(PyValueError::new_err(
            "thermodynamic reward was not finite and positive",
        ));
    }
    Ok(multiplier)
}

fn coordinate_to_voxel(
    coordinate: [f32; 3],
    origin: [f32; 3],
    spacing_a: f32,
    grid_dim: u32,
) -> Option<u32> {
    let mut index = [0_u32; 3];
    for axis in 0..3 {
        let raw = ((coordinate[axis] - origin[axis]) / spacing_a).floor();
        if !raw.is_finite() || raw < 0.0 || raw >= grid_dim as f32 {
            return None;
        }
        index[axis] = raw as u32;
    }
    Some(index[2] * grid_dim * grid_dim + index[1] * grid_dim + index[0])
}
