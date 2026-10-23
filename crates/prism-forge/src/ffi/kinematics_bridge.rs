use crate::core::synthon::{AttachmentPoint, ScaffoldState3D, Synthon3D};
use crate::reactions::kinematics::{
    execute_3d_reaction as execute_native_3d_reaction, DihedralConstraint, ReactionKinematicRule,
};
use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type CoordinateChargeArrays = (Py<PyArray2<f32>>, Py<PyArray1<f32>>);

#[pyfunction]
#[pyo3(signature = (
    scaffold_coordinates,
    scaffold_charges,
    synthon_coordinates,
    synthon_charges,
    scaffold_atom_index,
    synthon_atom_index,
    scaffold_exit_vector,
    synthon_attachment_vector,
    bond_length_a,
    scaffold_leaving_group_atom_index=None,
    synthon_leaving_group_atom_index=None,
    scaffold_dihedral_reference_atom_index=None,
    synthon_dihedral_reference_atom_index=None,
    dihedral_omega_rad=None
))]
#[allow(clippy::too_many_arguments)]
pub fn execute_3d_reaction<'py>(
    py: Python<'py>,
    scaffold_coordinates: PyReadonlyArray2<'_, f32>,
    scaffold_charges: PyReadonlyArray1<'_, f32>,
    synthon_coordinates: PyReadonlyArray2<'_, f32>,
    synthon_charges: PyReadonlyArray1<'_, f32>,
    scaffold_atom_index: usize,
    synthon_atom_index: usize,
    scaffold_exit_vector: PyReadonlyArray1<'_, f32>,
    synthon_attachment_vector: PyReadonlyArray1<'_, f32>,
    bond_length_a: f32,
    scaffold_leaving_group_atom_index: Option<usize>,
    synthon_leaving_group_atom_index: Option<usize>,
    scaffold_dihedral_reference_atom_index: Option<usize>,
    synthon_dihedral_reference_atom_index: Option<usize>,
    dihedral_omega_rad: Option<f32>,
) -> PyResult<CoordinateChargeArrays> {
    let scaffold_coordinates = flatten_coordinates(scaffold_coordinates, "scaffold_coordinates")?;
    let scaffold_charges = flat_vector(scaffold_charges, "scaffold_charges")?;
    let synthon_coordinates = flatten_coordinates(synthon_coordinates, "synthon_coordinates")?;
    let synthon_charges = flat_vector(synthon_charges, "synthon_charges")?;
    let scaffold_vector = vector3(scaffold_exit_vector, "scaffold_exit_vector")?;
    let synthon_vector = vector3(synthon_attachment_vector, "synthon_attachment_vector")?;

    let scaffold = ScaffoldState3D::new(
        scaffold_coordinates,
        scaffold_charges,
        vec![AttachmentPoint::new(
            scaffold_atom_index,
            scaffold_leaving_group_atom_index,
            scaffold_vector,
            scaffold_dihedral_reference_atom_index,
        )],
    )
    .map_err(to_value_error)?;
    let synthon = Synthon3D::new(
        synthon_coordinates,
        synthon_charges,
        vec![AttachmentPoint::new(
            synthon_atom_index,
            synthon_leaving_group_atom_index,
            synthon_vector,
            synthon_dihedral_reference_atom_index,
        )],
    )
    .map_err(to_value_error)?;
    let rule = ReactionKinematicRule {
        bond_length_a,
        dihedral_omega_rad: dihedral_omega_rad
            .map_or(DihedralConstraint::FreeRotation, DihedralConstraint::Fixed),
    };
    let product =
        execute_native_3d_reaction(&scaffold, &synthon, 0, 0, rule).map_err(to_value_error)?;
    let atom_count = product.charges.len();
    let coordinates = Array2::from_shape_vec((atom_count, 3), product.coordinates)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let charges = Array1::from_vec(product.charges);
    Ok((
        coordinates.into_pyarray(py).to_owned(),
        charges.into_pyarray(py).to_owned(),
    ))
}

fn flatten_coordinates(coordinates: PyReadonlyArray2<'_, f32>, label: &str) -> PyResult<Vec<f32>> {
    let shape = coordinates.shape();
    if shape.len() != 2 || shape[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "{label} must have shape [N, 3]"
        )));
    }
    let slice = coordinates.as_slice().map_err(|_| {
        PyValueError::new_err(format!("{label} must be contiguous float32 NumPy storage"))
    })?;
    Ok(slice.to_vec())
}

fn flat_vector(vector: PyReadonlyArray1<'_, f32>, label: &str) -> PyResult<Vec<f32>> {
    vector.as_slice().map(|slice| slice.to_vec()).map_err(|_| {
        PyValueError::new_err(format!("{label} must be contiguous float32 NumPy storage"))
    })
}

fn vector3(vector: PyReadonlyArray1<'_, f32>, label: &str) -> PyResult<[f32; 3]> {
    let slice = vector.as_slice().map_err(|_| {
        PyValueError::new_err(format!("{label} must be contiguous float32 NumPy storage"))
    })?;
    if slice.len() != 3 {
        return Err(PyValueError::new_err(format!(
            "{label} must contain exactly 3 floats"
        )));
    }
    Ok([slice[0], slice[1], slice[2]])
}

fn to_value_error(error: impl core::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}
