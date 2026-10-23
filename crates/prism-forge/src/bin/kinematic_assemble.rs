use anyhow::{anyhow, Context, Result};
use prism_forge::reactions::kinematic_assembly::zmatrix_attach_fragment;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
struct AssemblyBatchRequest {
    requests: Vec<AssemblyRequest>,
}

#[derive(Debug, Deserialize)]
struct AssemblyRequest {
    trajectory_id: String,
    scaffold_coordinates: Vec<[f32; 3]>,
    scaffold_bonds: Vec<[usize; 2]>,
    scaffold_exit_atom_index: usize,
    scaffold_exit_atom: [f32; 3],
    scaffold_reference_atom_1: [f32; 3],
    scaffold_reference_atom_2: [f32; 3],
    fragment_coordinates: Vec<[f32; 3]>,
    fragment_bonds: Vec<[usize; 2]>,
    bond_length_a: f32,
    bond_angle_deg: f32,
    dihedral_deg: f32,
    hybridization_model: String,
}

#[derive(Debug, Serialize)]
struct AssemblyBatchResponse {
    responses: Vec<AssemblyResponse>,
}

#[derive(Debug, Serialize)]
struct AssemblyResponse {
    trajectory_id: String,
    coordinates: Vec<[f32; 3]>,
    product_coordinates: Vec<[f32; 3]>,
    product_bonds: Vec<[usize; 2]>,
    sampled_dihedral_deg: f32,
    bond_length_a: f32,
    bond_angle_deg: f32,
    hybridization_model: String,
    assembly_mode: String,
    z_matrix_active: bool,
}

fn main() -> Result<()> {
    let (input, output) = parse_args()?;
    let raw = fs::read_to_string(&input)
        .with_context(|| format!("read kinematic assembly request {}", input.display()))?;
    let request: AssemblyBatchRequest =
        serde_json::from_str(&raw).context("parse kinematic assembly JSON request")?;
    let mut responses = Vec::with_capacity(request.requests.len());
    for item in request.requests {
        responses.push(assemble_one(item)?);
    }
    let encoded = serde_json::to_string(&AssemblyBatchResponse { responses })
        .context("encode kinematic assembly JSON response")?;
    fs::write(&output, encoded)
        .with_context(|| format!("write kinematic assembly response {}", output.display()))?;
    Ok(())
}

fn parse_args() -> Result<(PathBuf, PathBuf)> {
    let mut args = env::args().skip(1);
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" => {
                input = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| anyhow!("--input requires a path"))?,
                ));
            }
            "--output" => {
                output = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| anyhow!("--output requires a path"))?,
                ));
            }
            "--help" | "-h" => {
                println!("Usage: kinematic_assemble --input request.json --output response.json");
                std::process::exit(0);
            }
            other => return Err(anyhow!("unknown argument: {other}")),
        }
    }
    Ok((
        input.ok_or_else(|| anyhow!("missing --input"))?,
        output.ok_or_else(|| anyhow!("missing --output"))?,
    ))
}

fn assemble_one(request: AssemblyRequest) -> Result<AssemblyResponse> {
    if request.scaffold_coordinates.is_empty() {
        return Err(anyhow!(
            "trajectory {} has no scaffold coordinates",
            request.trajectory_id
        ));
    }
    if request.scaffold_exit_atom_index >= request.scaffold_coordinates.len() {
        return Err(anyhow!(
            "trajectory {} has scaffold_exit_atom_index outside scaffold coordinates",
            request.trajectory_id
        ));
    }
    if request.fragment_coordinates.is_empty() {
        return Err(anyhow!(
            "trajectory {} has no fragment coordinates",
            request.trajectory_id
        ));
    }
    let fragment_attachment = request.fragment_coordinates[0];
    let fragment_reference = if request.fragment_coordinates.len() > 1 {
        request.fragment_coordinates[1]
    } else {
        [
            fragment_attachment[0] + 1.0,
            fragment_attachment[1],
            fragment_attachment[2],
        ]
    };
    let flattened = request
        .fragment_coordinates
        .iter()
        .flat_map(|xyz| xyz.iter().copied())
        .collect::<Vec<_>>();
    let report = zmatrix_attach_fragment(
        request.scaffold_exit_atom,
        request.scaffold_reference_atom_1,
        request.scaffold_reference_atom_2,
        fragment_attachment,
        fragment_reference,
        &flattened,
        request.bond_length_a,
        request.bond_angle_deg,
        request.dihedral_deg,
        &request.hybridization_model,
    )
    .with_context(|| format!("z-matrix assembly failed for {}", request.trajectory_id))?;
    let coordinates = report
        .coordinates
        .chunks_exact(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect::<Vec<_>>();
    let mut product_coordinates = request.scaffold_coordinates.clone();
    product_coordinates.extend(coordinates.iter().copied());
    let product_bonds = product_bonds(
        request.scaffold_coordinates.len(),
        request.scaffold_exit_atom_index,
        &request.scaffold_bonds,
        &request.fragment_bonds,
    );
    Ok(AssemblyResponse {
        trajectory_id: request.trajectory_id,
        coordinates,
        product_coordinates,
        product_bonds,
        sampled_dihedral_deg: report.sampled_dihedral_deg,
        bond_length_a: report.bond_length_a,
        bond_angle_deg: report.bond_angle_deg,
        hybridization_model: report.hybridization_model,
        assembly_mode: "rust_zmatrix_subprocess".to_owned(),
        z_matrix_active: report.z_matrix_active,
    })
}

fn product_bonds(
    scaffold_atoms: usize,
    scaffold_exit_atom_index: usize,
    scaffold_bonds: &[[usize; 2]],
    fragment_bonds: &[[usize; 2]],
) -> Vec<[usize; 2]> {
    let mut bonds = Vec::new();
    for &[lhs, rhs] in scaffold_bonds {
        push_unique_bond(&mut bonds, lhs, rhs);
    }
    for &[lhs, rhs] in fragment_bonds {
        push_unique_bond(&mut bonds, scaffold_atoms + lhs, scaffold_atoms + rhs);
    }
    push_unique_bond(&mut bonds, scaffold_exit_atom_index, scaffold_atoms);
    bonds
}

fn push_unique_bond(bonds: &mut Vec<[usize; 2]>, lhs: usize, rhs: usize) {
    if lhs == rhs {
        return;
    }
    let bond = if lhs < rhs { [lhs, rhs] } else { [rhs, lhs] };
    if !bonds.contains(&bond) {
        bonds.push(bond);
    }
}
