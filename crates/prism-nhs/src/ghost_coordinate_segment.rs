//! Exact, lossless coordinate lineage for durable Ghost records.
//!
//! The interval trajectory is not an admissible source for this contract: a
//! Ghost record may be produced at a step that was not selected by the movie
//! cadence. Instead, the runtime compacts the complete signed protein-analysis
//! population in topology-atom order for every unique `(stream_id, step_idx)`
//! present in a synchronized Ghost drain, persists it in a content-addressed
//! segment, and indexes every Ghost record to that exact frame. Multiple sites
//! at the same step share one frame without losing record-level byte lineage.
//!
//! Water, ion and lambda-fragment coordinates remain a separate exact local-
//! shell payload produced by a complete excluded-population scan. Both planes
//! are SHA-bound to the same full source frame. This eliminates nearest-frame
//! interpolation, AABB geometry, fixed-ring truncation and non-protein residue-
//! axis leakage without duplicating bulk solvent in every analysis frame.

use crate::ghost_segment::{GhostSegmentContext, GhostSegmentEntry, GHOST_SEGMENT_RECORD_BYTES};
use crate::ghost_tile::{
    decode_ghost_v3_cluster_residue_membership, GHOST_FRAME_SCHEMA_V3, GHOST_V2_OFFSET_CENTROID,
    GHOST_V2_OFFSET_SCHEMA_VERSION, GHOST_V2_OFFSET_STEP_IDX,
};
use crate::input::{
    AnalysisResidueContract, MoleculeIdentityRow, MoleculeIdentityTable, PeriodicCellContract,
    PrismPrepTopology,
};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Component, Path, PathBuf};

pub const GHOST_COORDINATE_SCHEMA_VERSION: u32 = 2;
pub const GHOST_COORDINATE_SEGMENT_MAGIC: [u8; 8] = *b"PGCSEG01";
pub const GHOST_COORDINATE_HEADER_BYTES: usize = 4096;
const FRAME_MAGIC: [u8; 8] = *b"PGCFRM01";
const FRAME_HEADER_BYTES: usize = 128;
const JOIN_MAGIC: [u8; 8] = *b"PGCJON01";
const ZERO_SHA256: &str = "0000000000000000000000000000000000000000000000000000000000000000";
const SHELL_ENTRY_BYTES: usize = 32;
const LEGACY_TOPOLOGY_AUTHORITY_ENV: &str = "PRISMOS_GHOST_TOPOLOGY_AUTHORITY";
const LEGACY_TOPOLOGY_POLICY_ENV: &str = "PRISMOS_GHOST_TOPOLOGY_POLICY";
const LEGACY_SOURCE_PRMTOP_ENV: &str = "PRISMOS_GHOST_SOURCE_PRMTOP";

#[cfg(feature = "v2_ignition")]
const COMPILED_GHOST_POLICY_SHA256: &str = env!("PRISM_GHOST_TOPOLOGY_POLICY_SHA256");
#[cfg(feature = "v2_ignition")]
const COMPILED_GHOST_POLICY_BUILD_PATH: &str = env!("PRISM_GHOST_TOPOLOGY_POLICY_BUILD_PATH");

fn compiled_ghost_policy_binding() -> (Option<&'static str>, Option<&'static str>) {
    #[cfg(feature = "v2_ignition")]
    {
        return (
            Some(COMPILED_GHOST_POLICY_SHA256),
            Some(COMPILED_GHOST_POLICY_BUILD_PATH),
        );
    }
    #[cfg(not(feature = "v2_ignition"))]
    {
        (None, None)
    }
}

fn require_compiled_ghost_policy(policy_artifact_sha256: &str) -> Result<()> {
    let (compiled_policy_sha256, compiled_policy_build_path) = compiled_ghost_policy_binding();
    require_ghost_policy_binding(
        policy_artifact_sha256,
        compiled_policy_sha256,
        compiled_policy_build_path,
    )
}

fn require_ghost_policy_binding(
    policy_artifact_sha256: &str,
    compiled_policy_sha256: Option<&str>,
    compiled_policy_build_path: Option<&str>,
) -> Result<()> {
    match (compiled_policy_sha256, compiled_policy_build_path) {
        (None, None) => Ok(()),
        (Some(compiled_sha256), Some(build_path)) => {
            require_sha256(compiled_sha256, "compiled_ghost_policy_sha256")?;
            if !Path::new(build_path).is_absolute() {
                bail!("compiled Ghost topology policy source path is not absolute");
            }
            if policy_artifact_sha256 != compiled_sha256 {
                bail!(
                    "runtime Ghost topology policy differs from compiled v2 policy digest: compiled={} runtime={}",
                    compiled_sha256,
                    policy_artifact_sha256
                );
            }
            Ok(())
        }
        _ => bail!("compiled Ghost topology policy binding is incomplete"),
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyTopologyIdentityPolicy {
    sha256: String,
    size_bytes: u64,
    n_atoms: usize,
    n_residues: usize,
    n_chains: usize,
    chain_ids: Vec<String>,
    generated_prmtop_sha256: String,
    source_prmtop_artifact: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyParameterTopologyPolicy {
    sha256: String,
    size_bytes: u64,
    amber_pointer_format: String,
    natom: usize,
    nres: usize,
    ifbox: i64,
    box_dimensions_required_absent: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyProjectionPolicy {
    analysis_authority: String,
    spatial_distance_mode: String,
    spatial_boundary_reason: String,
    protein_residue_names: Vec<String>,
    expected_molecule_count: usize,
    expected_molecule_kind: String,
    expected_molecule_identity: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyGhostTopologyPolicy {
    schema_kind: String,
    schema_version: u32,
    policy_id: String,
    topology: LegacyTopologyIdentityPolicy,
    source_parameter_topology: LegacyParameterTopologyPolicy,
    projection: LegacyProjectionPolicy,
    authority_producer_sha256: String,
    policy_contract_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyGhostTopologyAuthority {
    schema_kind: String,
    schema_version: u32,
    policy_sha256: String,
    policy_contract_sha256: String,
    topology_sha256: String,
    topology_size_bytes: u64,
    topology_atom_count: usize,
    topology_physical_residue_count: usize,
    topology_analysis_residue_count: usize,
    source_parameter_topology_sha256: String,
    source_parameter_topology_size_bytes: u64,
    source_parameter_topology_natom: usize,
    source_parameter_topology_nres: usize,
    source_parameter_topology_ifbox: i64,
    source_parameter_topology_box_dimensions_present: bool,
    atom_physical_residue_ids: Vec<usize>,
    atom_analysis_residue_ids: Vec<i32>,
    analysis_residue_contract: AnalysisResidueContract,
    molecule_identity_table: MoleculeIdentityTable,
    spatial_boundary: PeriodicCellContract,
    spatial_distance_mode: String,
    atom_order_sha256: String,
    producer_source_sha256: String,
    authority_contract_sha256: String,
}

#[derive(Debug)]
struct ResolvedGhostTopology {
    topology: PrismPrepTopology,
    topology_sha256: String,
    topology_size_bytes: u64,
    projection_mode: String,
    authority_contract_sha256: String,
    authority_artifact_sha256: String,
    authority_source_sha256: String,
    atom_order_sha256: String,
    compiled_policy_sha256: Option<String>,
    compiled_policy_build_path: Option<String>,
    policy_artifact_sha256: String,
    policy_contract_sha256: String,
    source_parameter_topology_sha256: String,
    source_parameter_topology_size_bytes: u64,
    source_parameter_topology_natom: usize,
    source_parameter_topology_nres: usize,
    source_parameter_topology_ifbox: i64,
    source_parameter_topology_box_dimensions_present: bool,
}

/// Deterministic, pre-CUDA admission result for an exact legacy Ghost input.
///
/// Successful construction means that topology, policy, sidecar, parameter
/// topology, residue axis, connected components, atom order and boundary all
/// passed the same fail-closed validators used by the production coordinate
/// writer. Rejections are returned as errors and must not launch the workload.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GhostTopologyAdmission {
    pub schema_kind: String,
    pub schema_version: u32,
    pub status: String,
    pub cuda_context_created: bool,
    pub selected_engine_path: String,
    pub selected_engine_sha256: String,
    pub selected_engine_size_bytes: u64,
    pub topology_projection_mode: String,
    pub topology_sha256: String,
    pub topology_size_bytes: u64,
    pub authority_artifact_sha256: String,
    pub authority_contract_sha256: String,
    pub authority_source_sha256: String,
    pub compiled_policy_sha256: Option<String>,
    pub compiled_policy_build_path: Option<String>,
    pub policy_artifact_sha256: String,
    pub policy_contract_sha256: String,
    pub source_parameter_topology_sha256: String,
    pub source_parameter_topology_size_bytes: u64,
    pub source_parameter_topology_natom: usize,
    pub source_parameter_topology_nres: usize,
    pub source_parameter_topology_ifbox: i64,
    pub source_parameter_topology_box_dimensions_present: bool,
    pub physical_residue_count: usize,
    pub analysis_residue_count: usize,
    pub molecule_count: usize,
    pub analysis_residue_contract_sha256: String,
    pub molecule_identity_table_sha256: String,
    pub spatial_boundary_contract_sha256: String,
    pub atom_order_sha256: String,
    pub spatial_distance_mode: String,
}

#[derive(Clone, Copy, Debug)]
struct AmberPointerContract {
    pointer_count: usize,
    natom: usize,
    nres: usize,
    ifbox: i64,
    box_dimensions_present: bool,
}

#[derive(Clone, Copy)]
enum SpatialDistance {
    PeriodicMinimumImage {
        matrix: [[f64; 3]; 3],
        inverse: [[f64; 3]; 3],
    },
    NonperiodicDirect,
}

fn canonical_sha256_without_field<T: Serialize>(value: &T, field: &str) -> Result<String> {
    let mut value = serde_json::to_value(value)?;
    value
        .as_object_mut()
        .context("digest-bound authority is not a JSON object")?
        .remove(field);
    crate::input::python_canonical_sha256(&value).map_err(anyhow::Error::msg)
}

fn native_atom_order_sha256(topology: &PrismPrepTopology) -> Result<String> {
    let molecules = topology
        .molecule_identity_table
        .as_ref()
        .context("coordinate lineage requires molecule_identity_table")?;
    crate::input::python_canonical_sha256(&serde_json::json!({
        "n_atoms": topology.n_atoms,
        "elements": topology.elements,
        "atom_names": topology.atom_names,
        "residue_names": topology.residue_names,
        "atom_physical_residue_ids": topology.atom_physical_residue_ids,
        "atom_analysis_residue_ids": topology.atom_analysis_residue_ids,
        "chain_ids": topology.chain_ids,
        "molecule_identity_table_sha256": molecules.table_sha256,
    }))
    .map_err(anyhow::Error::msg)
}

fn validate_spatial_boundary(topology: &PrismPrepTopology) -> Result<&'static str> {
    let boundary = topology
        .periodic_cell
        .as_ref()
        .context("coordinate lineage requires a spatial boundary contract")?;
    require_sha256(
        boundary
            .contract_sha256
            .as_deref()
            .context("coordinate lineage requires a spatial-boundary SHA256")?,
        "spatial_boundary_contract_sha256",
    )?;
    if boundary.periodic
        && boundary.minimum_image_required == Some(true)
        && boundary.vectors_angstrom.is_some()
    {
        return Ok("PERIODIC_TRICLINIC_MINIMUM_IMAGE");
    }
    if !boundary.periodic
        && boundary.minimum_image_required == Some(false)
        && boundary.coordinate_wrapping_required == Some(false)
        && boundary.vectors_angstrom.is_none()
        && boundary.reciprocal_method.as_deref() == Some("NONE_NONPERIODIC_DIRECT_DISTANCE")
    {
        return Ok("NONPERIODIC_DIRECT_DISTANCE");
    }
    bail!("coordinate lineage spatial boundary is incomplete or inconsistent")
}

/// Older v3 producers represented an open-boundary topology only as
/// `{periodic:false, reason:"no_periodic_cell"}`.  The Ghost coordinate
/// lineage contract needs the corresponding explicit Euclidean semantics and
/// a content hash.  Normalize that exact legacy declaration in memory; source
/// topology bytes remain untouched and every later context check resolves the
/// same derived boundary deterministically.
pub(crate) fn normalize_legacy_nonperiodic_boundary(
    topology: &mut PrismPrepTopology,
) -> Result<bool> {
    let Some(boundary) = topology.periodic_cell.as_mut() else {
        return Ok(false);
    };
    let legacy_open_boundary = !boundary.periodic
        && boundary.reason.as_deref() == Some("no_periodic_cell")
        && boundary.vectors_angstrom.is_none()
        && boundary.minimum_image_required.is_none()
        && boundary.coordinate_wrapping_required.is_none()
        && boundary.reciprocal_method.is_none()
        && boundary.contract_sha256.is_none();
    if !legacy_open_boundary {
        return Ok(false);
    }
    boundary.minimum_image_required = Some(false);
    boundary.coordinate_wrapping_required = Some(false);
    boundary.reciprocal_method = Some("NONE_NONPERIODIC_DIRECT_DISTANCE".to_string());
    boundary.contract_sha256 = Some(canonical_sha256_without_field(boundary, "contract_sha256")?);
    Ok(true)
}

fn resolve_ghost_topology(
    topology: &PrismPrepTopology,
    topology_sha256: &str,
) -> Result<ResolvedGhostTopology> {
    require_sha256(topology_sha256, "topology_sha256")?;
    let mut native_topology = topology.clone();
    let normalized_legacy_nonperiodic =
        normalize_legacy_nonperiodic_boundary(&mut native_topology)?;
    if native_topology.validate_analysis_residue_axis().is_ok() {
        validate_spatial_boundary(&native_topology)?;
        let analysis = native_topology.analysis_residue_contract.as_ref().unwrap();
        let molecules = native_topology.molecule_identity_table.as_ref().unwrap();
        let boundary = native_topology.periodic_cell.as_ref().unwrap();
        let projection_contract = crate::input::python_canonical_sha256(&serde_json::json!({
            "topology_sha256": topology_sha256,
            "analysis_residue_contract_sha256": analysis.contract_sha256,
            "molecule_identity_table_sha256": molecules.table_sha256,
            "spatial_boundary_contract_sha256": boundary.contract_sha256,
        }))
        .map_err(anyhow::Error::msg)?;
        let (compiled_policy_sha256, compiled_policy_build_path) = compiled_ghost_policy_binding();
        return Ok(ResolvedGhostTopology {
            topology: native_topology.clone(),
            topology_sha256: topology_sha256.into(),
            topology_size_bytes: 0,
            projection_mode: "NATIVE_TOPOLOGY_V3".into(),
            authority_contract_sha256: projection_contract,
            authority_artifact_sha256: topology_sha256.into(),
            authority_source_sha256: topology_sha256.into(),
            atom_order_sha256: native_atom_order_sha256(&native_topology)?,
            compiled_policy_sha256: compiled_policy_sha256.map(str::to_owned),
            compiled_policy_build_path: compiled_policy_build_path.map(str::to_owned),
            policy_artifact_sha256: topology_sha256.into(),
            policy_contract_sha256: topology_sha256.into(),
            source_parameter_topology_sha256: topology
                .generated_prmtop_sha256
                .clone()
                .unwrap_or_else(|| topology_sha256.into()),
            source_parameter_topology_size_bytes: 0,
            source_parameter_topology_natom: native_topology.n_atoms,
            source_parameter_topology_nres: native_topology.n_residues,
            source_parameter_topology_ifbox: if native_topology
                .periodic_cell
                .as_ref()
                .is_some_and(|cell| cell.periodic)
            {
                1
            } else {
                0
            },
            source_parameter_topology_box_dimensions_present: native_topology
                .periodic_cell
                .as_ref()
                .is_some_and(|cell| cell.periodic),
        })
        .map(|mut resolved| {
            if normalized_legacy_nonperiodic {
                resolved.projection_mode =
                    "NATIVE_TOPOLOGY_V3_LEGACY_OPEN_BOUNDARY_NORMALIZED".into();
            }
            resolved
        });
    }

    let authority_path =
        required_absolute_env_path(LEGACY_TOPOLOGY_AUTHORITY_ENV, "legacy topology authority")?;
    let policy_path =
        required_absolute_env_path(LEGACY_TOPOLOGY_POLICY_ENV, "legacy topology policy")?;
    let source_prmtop_path = required_absolute_env_path(
        LEGACY_SOURCE_PRMTOP_ENV,
        "legacy source AMBER parameter topology",
    )?;
    let authority_bytes = read_regular_nofollow(&authority_path).with_context(|| {
        format!(
            "read explicit legacy Ghost topology authority {}",
            authority_path.display()
        )
    })?;
    let policy_bytes = read_regular_nofollow(&policy_path).with_context(|| {
        format!(
            "read explicit legacy Ghost topology policy {}",
            policy_path.display()
        )
    })?;
    let source_prmtop_bytes = read_regular_nofollow(&source_prmtop_path).with_context(|| {
        format!(
            "read exact legacy source AMBER parameter topology {}",
            source_prmtop_path.display()
        )
    })?;
    resolve_legacy_ghost_topology_authority(
        topology,
        topology_sha256,
        &authority_bytes,
        &policy_bytes,
        &source_prmtop_bytes,
    )
}

fn required_absolute_env_path(name: &str, label: &str) -> Result<PathBuf> {
    let path = std::env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .with_context(|| {
            format!("legacy topology requires {name}; implicit {label} is forbidden")
        })?;
    if !path.is_absolute() {
        bail!("{label} path from {name} must be absolute");
    }
    Ok(path)
}

fn parse_amber_pointer_contract(bytes: &[u8]) -> Result<AmberPointerContract> {
    if !bytes.is_ascii() {
        bail!("source AMBER parameter topology is not ASCII");
    }
    let text = std::str::from_utf8(bytes)?;
    let lines: Vec<&str> = text.lines().collect();
    let pointer_indices: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter_map(|(index, line)| (*line == "%FLAG POINTERS").then_some(index))
        .collect();
    if pointer_indices.len() != 1 {
        bail!("source AMBER parameter topology must contain exactly one POINTERS flag");
    }
    let start = pointer_indices[0];
    if lines.get(start + 1) != Some(&"%FORMAT(10I8)") {
        bail!("source AMBER POINTERS must use exact %FORMAT(10I8)");
    }
    let mut pointers = Vec::<i64>::new();
    for line in lines.iter().skip(start + 2) {
        if line.starts_with("%FLAG ") {
            break;
        }
        if line.starts_with('%') || line.len() % 8 != 0 {
            bail!("source AMBER POINTERS contains a malformed fixed-width I8 row");
        }
        for field in line.as_bytes().chunks_exact(8) {
            let token = std::str::from_utf8(field)?.trim();
            pointers.push(
                token
                    .parse::<i64>()
                    .context("source AMBER POINTERS contains a non-integer I8 field")?,
            );
        }
    }
    if pointers.len() < 31 {
        bail!("source AMBER POINTERS contains fewer than 31 required values");
    }
    let natom: usize = pointers[0]
        .try_into()
        .context("source AMBER NATOM is negative or exceeds usize")?;
    let nres: usize = pointers[11]
        .try_into()
        .context("source AMBER NRES is negative or exceeds usize")?;
    Ok(AmberPointerContract {
        pointer_count: pointers.len(),
        natom,
        nres,
        ifbox: pointers[27],
        box_dimensions_present: lines.iter().any(|line| *line == "%FLAG BOX_DIMENSIONS"),
    })
}

fn decode_legacy_topology_policy(
    policy_bytes: &[u8],
) -> Result<(LegacyGhostTopologyPolicy, String)> {
    let policy_artifact_sha256 = sha256_bytes(policy_bytes);
    require_compiled_ghost_policy(&policy_artifact_sha256)?;
    let policy: LegacyGhostTopologyPolicy = serde_json::from_slice(policy_bytes)
        .context("decode explicit legacy Ghost topology policy")?;
    if policy.schema_kind != "prismos_legacy_ghost_topology_policy"
        || policy.schema_version != 1
        || policy.policy_id.trim().is_empty()
        || policy.topology.size_bytes == 0
        || policy.topology.n_atoms == 0
        || policy.topology.n_residues == 0
        || policy.topology.n_chains == 0
        || policy.topology.chain_ids.is_empty()
        || policy.topology.source_prmtop_artifact.trim().is_empty()
        || policy.source_parameter_topology.size_bytes == 0
        || policy.source_parameter_topology.amber_pointer_format != "AMBER_PRMTOP_POINTERS_10I8"
        || policy.source_parameter_topology.ifbox != 0
        || !policy
            .source_parameter_topology
            .box_dimensions_required_absent
        || policy.projection.analysis_authority != "KCC_GHOST_PROTEIN_RESIDUE_AXIS"
        || policy.projection.spatial_distance_mode != "NONPERIODIC_DIRECT_DISTANCE"
        || policy.projection.spatial_boundary_reason.trim().is_empty()
        || policy.projection.protein_residue_names.is_empty()
        || policy.projection.expected_molecule_count == 0
        || policy.projection.expected_molecule_kind.trim().is_empty()
        || policy
            .projection
            .expected_molecule_identity
            .trim()
            .is_empty()
        || policy.topology.generated_prmtop_sha256 != policy.source_parameter_topology.sha256
    {
        bail!("explicit legacy Ghost topology policy is inadmissible");
    }
    for (name, digest) in [
        ("policy_contract_sha256", &policy.policy_contract_sha256),
        (
            "authority_producer_sha256",
            &policy.authority_producer_sha256,
        ),
        ("policy_topology_sha256", &policy.topology.sha256),
        (
            "policy_generated_prmtop_sha256",
            &policy.topology.generated_prmtop_sha256,
        ),
        (
            "policy_source_parameter_topology_sha256",
            &policy.source_parameter_topology.sha256,
        ),
    ] {
        require_sha256(digest, name)?;
    }
    if canonical_sha256_without_field(&policy, "policy_contract_sha256")?
        != policy.policy_contract_sha256
    {
        bail!("explicit legacy Ghost topology policy contract digest mismatch");
    }
    Ok((policy, policy_artifact_sha256))
}

fn canonical_molecule_table(topology: &PrismPrepTopology) -> Result<MoleculeIdentityTable> {
    let mut adjacency = vec![Vec::<usize>::new(); topology.n_atoms];
    let mut observed_edges = BTreeSet::<(usize, usize)>::new();
    for (ordinal, bond) in topology.bonds.iter().enumerate() {
        if bond.i >= topology.n_atoms || bond.j >= topology.n_atoms || bond.i == bond.j {
            bail!("legacy bond {ordinal} has invalid atom endpoints");
        }
        let edge = if bond.i < bond.j {
            (bond.i, bond.j)
        } else {
            (bond.j, bond.i)
        };
        if !observed_edges.insert(edge) {
            bail!("legacy bond table repeats edge {edge:?}");
        }
        adjacency[bond.i].push(bond.j);
        adjacency[bond.j].push(bond.i);
    }
    if observed_edges.is_empty() {
        bail!("legacy topology lacks an explicit nonempty bond table");
    }

    let water_names = ["HOH", "WAT", "TIP3"];
    let ion_elements = ["Na", "K", "Cl", "Mg", "Ca", "Zn"];
    let mut visited = vec![false; topology.n_atoms];
    let mut molecules = Vec::<MoleculeIdentityRow>::new();
    for start in 0..topology.n_atoms {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let mut stack = vec![start];
        let mut atom_indices = Vec::<usize>::new();
        while let Some(atom) = stack.pop() {
            atom_indices.push(atom);
            for &neighbor in &adjacency[atom] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    stack.push(neighbor);
                }
            }
        }
        atom_indices.sort_unstable();
        let residue_indices: Vec<usize> = atom_indices
            .iter()
            .map(|&atom| topology.residue_ids[atom])
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let residue_names: Vec<String> = atom_indices
            .iter()
            .map(|&atom| topology.residue_names[atom].clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let chain_ids: Vec<String> = atom_indices
            .iter()
            .map(|&atom| topology.chain_ids[atom].clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let (kind, identity) = if !residue_names.is_empty()
            && residue_names
                .iter()
                .all(|name| water_names.contains(&name.as_str()))
        {
            ("WATER".to_string(), "TIP3P-FB".to_string())
        } else if atom_indices.len() == 1
            && ion_elements.contains(&topology.elements[atom_indices[0]].as_str())
        {
            (
                "ION".to_string(),
                residue_names
                    .first()
                    .cloned()
                    .context("single-atom ion has no residue identity")?,
            )
        } else {
            (
                "BIOMOLECULE".to_string(),
                if chain_ids.is_empty() {
                    "UNLABELED".to_string()
                } else {
                    chain_ids.join("+")
                },
            )
        };
        let lineage = serde_json::json!({
            "atom_indices": &atom_indices,
            "residue_indices": &residue_indices,
            "residue_names": &residue_names,
            "chain_ids": &chain_ids,
            "kind": &kind,
            "identity": &identity,
        });
        molecules.push(MoleculeIdentityRow {
            molecule_id: molecules.len() as u64,
            atom_indices,
            residue_indices,
            residue_names,
            chain_ids,
            kind,
            identity,
            component_sha256: crate::input::python_canonical_sha256(&lineage)
                .map_err(anyhow::Error::msg)?,
        });
    }
    let covered: Vec<usize> = molecules
        .iter()
        .flat_map(|molecule| molecule.atom_indices.iter().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if covered != (0..topology.n_atoms).collect::<Vec<_>>()
        || molecules
            .iter()
            .map(|molecule| molecule.atom_indices.len())
            .sum::<usize>()
            != topology.n_atoms
    {
        bail!("legacy molecule connected-component table is not an exact atom partition");
    }
    let mut table = MoleculeIdentityTable {
        schema_kind: "prismos_molecule_identity_table".into(),
        schema_version: 1,
        n_molecules: molecules.len(),
        n_atoms: topology.n_atoms,
        molecules,
        table_sha256: String::new(),
    };
    table.table_sha256 = canonical_sha256_without_field(&table, "table_sha256")?;
    Ok(table)
}

fn resolve_legacy_ghost_topology_authority(
    topology: &PrismPrepTopology,
    topology_sha256: &str,
    authority_bytes: &[u8],
    policy_bytes: &[u8],
    source_prmtop_bytes: &[u8],
) -> Result<ResolvedGhostTopology> {
    require_sha256(topology_sha256, "topology_sha256")?;
    topology
        .validate()
        .map_err(anyhow::Error::msg)
        .context("validate exact legacy Ghost topology before projection")?;
    let authority_artifact_sha256 = sha256_bytes(authority_bytes);
    let authority: LegacyGhostTopologyAuthority = serde_json::from_slice(authority_bytes)
        .context("decode explicit legacy Ghost topology authority")?;
    let (policy, policy_artifact_sha256) = decode_legacy_topology_policy(policy_bytes)?;
    let source_prmtop_sha256 = sha256_bytes(source_prmtop_bytes);
    let amber = parse_amber_pointer_contract(source_prmtop_bytes)?;
    if authority.schema_kind != "prismos_legacy_ghost_topology_authority"
        || authority.schema_version != 2
        || authority.policy_sha256 != policy_artifact_sha256
        || authority.policy_contract_sha256 != policy.policy_contract_sha256
        || authority.topology_sha256 != topology_sha256
        || authority.topology_sha256 != policy.topology.sha256
        || authority.topology_size_bytes != policy.topology.size_bytes
        || authority.topology_atom_count != topology.n_atoms
        || authority.topology_atom_count != policy.topology.n_atoms
        || authority.topology_physical_residue_count != topology.n_residues
        || authority.topology_physical_residue_count != policy.topology.n_residues
        || authority.topology_analysis_residue_count == 0
        || authority.topology_analysis_residue_count > topology.n_residues
        || topology.n_chains != policy.topology.n_chains
        || topology.generated_prmtop_sha256.as_deref()
            != Some(policy.topology.generated_prmtop_sha256.as_str())
        || authority.source_parameter_topology_sha256 != source_prmtop_sha256
        || authority.source_parameter_topology_sha256 != policy.source_parameter_topology.sha256
        || authority.source_parameter_topology_size_bytes != source_prmtop_bytes.len() as u64
        || authority.source_parameter_topology_size_bytes
            != policy.source_parameter_topology.size_bytes
        || authority.source_parameter_topology_natom != amber.natom
        || authority.source_parameter_topology_natom != policy.source_parameter_topology.natom
        || authority.source_parameter_topology_nres != amber.nres
        || authority.source_parameter_topology_nres != policy.source_parameter_topology.nres
        || authority.source_parameter_topology_ifbox != amber.ifbox
        || authority.source_parameter_topology_ifbox != policy.source_parameter_topology.ifbox
        || authority.source_parameter_topology_box_dimensions_present
            != amber.box_dimensions_present
        || authority.source_parameter_topology_box_dimensions_present
        || amber.pointer_count < 31
        || authority.atom_physical_residue_ids != topology.residue_ids
        || authority.atom_analysis_residue_ids.len() != topology.n_atoms
        || authority.spatial_distance_mode != policy.projection.spatial_distance_mode
        || authority.producer_source_sha256 != policy.authority_producer_sha256
    {
        bail!("explicit legacy Ghost topology authority identity or shape mismatch");
    }
    let observed_chain_ids: Vec<String> = topology
        .chain_ids
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if observed_chain_ids != policy.topology.chain_ids {
        bail!("legacy Ghost topology chain identity differs from policy");
    }
    let canonical_protein_names: Vec<String> = policy
        .projection
        .protein_residue_names
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if canonical_protein_names != policy.projection.protein_residue_names
        || topology
            .residue_names
            .iter()
            .any(|name| canonical_protein_names.binary_search(name).is_err())
        || authority.spatial_boundary.reason.as_deref()
            != Some(policy.projection.spatial_boundary_reason.as_str())
        || authority.spatial_boundary.periodic
        || authority.spatial_boundary.vectors_angstrom.is_some()
        || authority.spatial_boundary.volume_angstrom3.is_some()
        || authority.spatial_boundary.orthorhombic.is_some()
        || authority.spatial_boundary.minimum_image_required != Some(false)
        || authority.spatial_boundary.coordinate_wrapping_required != Some(false)
        || authority.spatial_boundary.reciprocal_method.as_deref()
            != Some("NONE_NONPERIODIC_DIRECT_DISTANCE")
        || authority
            .spatial_boundary
            .real_space_cutoff_angstrom
            .is_some()
    {
        bail!("legacy Ghost topology protein/boundary identity differs from policy");
    }
    for (name, value) in [
        (
            "authority_contract_sha256",
            &authority.authority_contract_sha256,
        ),
        ("policy_sha256", &authority.policy_sha256),
        ("policy_contract_sha256", &authority.policy_contract_sha256),
        (
            "source_parameter_topology_sha256",
            &authority.source_parameter_topology_sha256,
        ),
        ("producer_source_sha256", &authority.producer_source_sha256),
        ("atom_order_sha256", &authority.atom_order_sha256),
    ] {
        require_sha256(value, name)?;
    }
    if canonical_sha256_without_field(&authority, "authority_contract_sha256")?
        != authority.authority_contract_sha256
    {
        bail!("explicit legacy Ghost topology authority contract digest mismatch");
    }
    if canonical_sha256_without_field(&authority.analysis_residue_contract, "contract_sha256")?
        != authority.analysis_residue_contract.contract_sha256
    {
        bail!("explicit legacy Ghost analysis-residue contract digest mismatch");
    }
    if canonical_sha256_without_field(&authority.molecule_identity_table, "table_sha256")?
        != authority.molecule_identity_table.table_sha256
    {
        bail!("explicit legacy Ghost molecule-table digest mismatch");
    }
    let canonical_table = canonical_molecule_table(topology)?;
    if serde_json::to_value(&canonical_table)?
        != serde_json::to_value(&authority.molecule_identity_table)?
        || canonical_table.n_molecules != policy.projection.expected_molecule_count
        || canonical_table.molecules.iter().any(|molecule| {
            molecule.kind != policy.projection.expected_molecule_kind
                || molecule.identity != policy.projection.expected_molecule_identity
        })
    {
        bail!("explicit legacy Ghost molecule table differs from canonical bond components");
    }
    if canonical_sha256_without_field(&authority.spatial_boundary, "contract_sha256")?
        != authority
            .spatial_boundary
            .contract_sha256
            .as_deref()
            .context("legacy spatial boundary is not hash-bound")?
    {
        bail!("explicit legacy Ghost spatial-boundary digest mismatch");
    }

    let mut effective = topology.clone();
    effective.total_physical_residue_count = Some(authority.topology_physical_residue_count);
    effective.n_analysis_residues = Some(authority.topology_analysis_residue_count);
    effective.atom_physical_residue_ids = Some(authority.atom_physical_residue_ids.clone());
    effective.atom_analysis_residue_ids = Some(authority.atom_analysis_residue_ids.clone());
    effective.analysis_residue_contract = Some(authority.analysis_residue_contract.clone());
    effective.molecule_identity_table = Some(authority.molecule_identity_table.clone());
    effective.periodic_cell = Some(authority.spatial_boundary.clone());
    effective
        .validate_analysis_residue_axis()
        .map_err(anyhow::Error::msg)
        .context("validate source-bound legacy Ghost residue projection")?;
    if validate_spatial_boundary(&effective)? != "NONPERIODIC_DIRECT_DISTANCE"
        || native_atom_order_sha256(&effective)? != authority.atom_order_sha256
    {
        bail!("explicit legacy Ghost topology projection differs from declared authority");
    }
    let (compiled_policy_sha256, compiled_policy_build_path) = compiled_ghost_policy_binding();
    Ok(ResolvedGhostTopology {
        topology: effective,
        topology_sha256: topology_sha256.into(),
        topology_size_bytes: authority.topology_size_bytes,
        projection_mode: "HASH_BOUND_LEGACY_SIDECAR".into(),
        authority_contract_sha256: authority.authority_contract_sha256,
        authority_artifact_sha256,
        authority_source_sha256: authority.producer_source_sha256,
        atom_order_sha256: authority.atom_order_sha256,
        compiled_policy_sha256: compiled_policy_sha256.map(str::to_owned),
        compiled_policy_build_path: compiled_policy_build_path.map(str::to_owned),
        policy_artifact_sha256,
        policy_contract_sha256: policy.policy_contract_sha256,
        source_parameter_topology_sha256: source_prmtop_sha256,
        source_parameter_topology_size_bytes: source_prmtop_bytes.len() as u64,
        source_parameter_topology_natom: amber.natom,
        source_parameter_topology_nres: amber.nres,
        source_parameter_topology_ifbox: amber.ifbox,
        source_parameter_topology_box_dimensions_present: amber.box_dimensions_present,
    })
}

fn ghost_topology_admission(resolved: &ResolvedGhostTopology) -> Result<GhostTopologyAdmission> {
    let topology = &resolved.topology;
    let analysis = topology
        .analysis_residue_contract
        .as_ref()
        .context("admitted Ghost topology lacks analysis residue contract")?;
    let molecules = topology
        .molecule_identity_table
        .as_ref()
        .context("admitted Ghost topology lacks molecule identity table")?;
    let boundary = topology
        .periodic_cell
        .as_ref()
        .context("admitted Ghost topology lacks spatial boundary contract")?;
    let selected_engine_path =
        std::env::current_exe().context("resolve selected Ghost preflight executable")?;
    if !selected_engine_path.is_absolute() {
        bail!("selected Ghost preflight executable path is not absolute");
    }
    let selected_engine_bytes =
        read_regular_nofollow(&selected_engine_path).with_context(|| {
            format!(
                "read selected Ghost preflight executable {}",
                selected_engine_path.display()
            )
        })?;
    Ok(GhostTopologyAdmission {
        schema_kind: "prismos_ghost_topology_admission".into(),
        schema_version: 1,
        status: "VERIFIED".into(),
        cuda_context_created: false,
        selected_engine_path: selected_engine_path.to_string_lossy().into_owned(),
        selected_engine_sha256: sha256_bytes(&selected_engine_bytes),
        selected_engine_size_bytes: selected_engine_bytes.len() as u64,
        topology_projection_mode: resolved.projection_mode.clone(),
        topology_sha256: resolved.topology_sha256.clone(),
        topology_size_bytes: resolved.topology_size_bytes,
        authority_artifact_sha256: resolved.authority_artifact_sha256.clone(),
        authority_contract_sha256: resolved.authority_contract_sha256.clone(),
        authority_source_sha256: resolved.authority_source_sha256.clone(),
        compiled_policy_sha256: resolved.compiled_policy_sha256.clone(),
        compiled_policy_build_path: resolved.compiled_policy_build_path.clone(),
        policy_artifact_sha256: resolved.policy_artifact_sha256.clone(),
        policy_contract_sha256: resolved.policy_contract_sha256.clone(),
        source_parameter_topology_sha256: resolved.source_parameter_topology_sha256.clone(),
        source_parameter_topology_size_bytes: resolved.source_parameter_topology_size_bytes,
        source_parameter_topology_natom: resolved.source_parameter_topology_natom,
        source_parameter_topology_nres: resolved.source_parameter_topology_nres,
        source_parameter_topology_ifbox: resolved.source_parameter_topology_ifbox,
        source_parameter_topology_box_dimensions_present: resolved
            .source_parameter_topology_box_dimensions_present,
        physical_residue_count: topology.n_residues,
        analysis_residue_count: topology.n_analysis_residues.unwrap_or_default(),
        molecule_count: molecules.n_molecules,
        analysis_residue_contract_sha256: analysis.contract_sha256.clone(),
        molecule_identity_table_sha256: molecules.table_sha256.clone(),
        spatial_boundary_contract_sha256: boundary
            .contract_sha256
            .clone()
            .context("admitted Ghost topology boundary is not hash-bound")?,
        atom_order_sha256: resolved.atom_order_sha256.clone(),
        spatial_distance_mode: validate_spatial_boundary(topology)?.into(),
    })
}

/// Validate exact path identities before any CUDA context or paid workload is
/// created. Every input is read as a regular nofollow file and independently
/// compared with the digest selected by the sealed execution identity pin.
#[allow(clippy::too_many_arguments)]
pub fn preflight_legacy_ghost_topology_paths(
    topology_path: &Path,
    expected_topology_sha256: &str,
    authority_path: &Path,
    expected_authority_sha256: &str,
    policy_path: &Path,
    expected_policy_sha256: &str,
    source_prmtop_path: &Path,
    expected_source_prmtop_sha256: &str,
) -> Result<GhostTopologyAdmission> {
    for (label, path) in [
        ("topology", topology_path),
        ("authority", authority_path),
        ("policy", policy_path),
        ("source AMBER parameter topology", source_prmtop_path),
    ] {
        if !path.is_absolute() {
            bail!("legacy Ghost {label} path must be absolute");
        }
    }
    for (label, digest) in [
        ("expected_topology_sha256", expected_topology_sha256),
        ("expected_authority_sha256", expected_authority_sha256),
        ("expected_policy_sha256", expected_policy_sha256),
        (
            "expected_source_prmtop_sha256",
            expected_source_prmtop_sha256,
        ),
    ] {
        require_sha256(digest, label)?;
    }
    let topology_bytes = read_regular_nofollow(topology_path)
        .with_context(|| format!("read legacy Ghost topology {}", topology_path.display()))?;
    let authority_bytes = read_regular_nofollow(authority_path)
        .with_context(|| format!("read legacy Ghost authority {}", authority_path.display()))?;
    let policy_bytes = read_regular_nofollow(policy_path)
        .with_context(|| format!("read legacy Ghost policy {}", policy_path.display()))?;
    let source_prmtop_bytes = read_regular_nofollow(source_prmtop_path).with_context(|| {
        format!(
            "read legacy source AMBER parameter topology {}",
            source_prmtop_path.display()
        )
    })?;
    let topology_sha256 = sha256_bytes(&topology_bytes);
    if topology_sha256 != expected_topology_sha256
        || sha256_bytes(&authority_bytes) != expected_authority_sha256
        || sha256_bytes(&policy_bytes) != expected_policy_sha256
        || sha256_bytes(&source_prmtop_bytes) != expected_source_prmtop_sha256
    {
        bail!("legacy Ghost preflight artifact digest differs from sealed execution identity");
    }
    let topology: PrismPrepTopology = serde_json::from_slice(&topology_bytes)
        .context("decode exact legacy Ghost topology for preflight")?;
    let resolved = resolve_legacy_ghost_topology_authority(
        &topology,
        &topology_sha256,
        &authority_bytes,
        &policy_bytes,
        &source_prmtop_bytes,
    )?;
    if resolved.topology_size_bytes != topology_bytes.len() as u64 {
        bail!("legacy Ghost topology byte length differs from policy-bound authority");
    }
    ghost_topology_admission(&resolved)
}

/// Immutable identity and scientific population of one stream's coordinate
/// chain.  The complete context is held by the atomic manifest and its digest
/// is embedded in every binary segment header.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GhostCoordinateContext {
    pub schema_kind: String,
    pub schema_version: u32,
    pub run_id: String,
    pub replica_id: String,
    pub replica_seed: u64,
    pub stream_seed: u64,
    pub stream_id: u16,
    pub topology_sha256: String,
    pub selected_engine_sha256: String,
    pub ghost_record_abi_sha256: String,
    pub ghost_record_source_sha256: String,
    pub ghost_segment_context_digest_sha256: String,
    pub thermodynamic_condition_id: String,
    pub protocol_id: String,
    pub protocol_state_hash: String,
    pub phase_schedule_digest_sha256: String,
    pub atom_count: u64,
    pub analysis_residue_count: u32,
    pub analysis_atom_count: u64,
    pub atom_analysis_residue_ids_sha256: String,
    pub analysis_residue_contract_sha256: String,
    pub molecule_identity_table_sha256: String,
    pub spatial_boundary_contract_sha256: String,
    pub spatial_distance_mode: String,
    pub topology_projection_mode: String,
    pub topology_projection_authority_sha256: String,
    pub topology_projection_artifact_sha256: String,
    pub topology_projection_source_sha256: String,
    pub coordinate_population: String,
    pub coordinate_scalar: String,
    pub coordinate_unit: String,
    pub atom_order_contract: String,
    pub atom_order_sha256: String,
    pub local_shell_cutoff_angstrom: f64,
    pub producer_source_sha256: String,
}

impl GhostCoordinateContext {
    /// Build and validate the coordinate context from the exact prepared
    /// topology and already-authenticated Ghost producer context.
    pub fn from_topology(
        topology: &PrismPrepTopology,
        topology_sha256: &str,
        ghost: &GhostSegmentContext,
        local_shell_cutoff_angstrom: f64,
    ) -> Result<Self> {
        ghost.verify()?;
        require_sha256(topology_sha256, "topology_sha256")?;
        if topology_sha256 != ghost.topology_sha256 {
            bail!("coordinate/Ghost topology SHA256 mismatch");
        }
        if !local_shell_cutoff_angstrom.is_finite() || local_shell_cutoff_angstrom <= 0.0 {
            bail!("local shell cutoff must be finite and positive");
        }
        let resolved = resolve_ghost_topology(topology, topology_sha256)?;
        let topology = &resolved.topology;
        let analysis = topology
            .analysis_residue_contract
            .as_ref()
            .context("coordinate lineage requires analysis_residue_contract")?;
        let analysis_ids = topology
            .atom_analysis_residue_ids
            .as_ref()
            .context("coordinate lineage requires atom_analysis_residue_ids")?;
        let molecules = topology
            .molecule_identity_table
            .as_ref()
            .context("coordinate lineage requires molecule_identity_table")?;
        let boundary = topology
            .periodic_cell
            .as_ref()
            .context("coordinate lineage requires a spatial boundary contract")?;
        let boundary_sha = boundary
            .contract_sha256
            .as_ref()
            .context("coordinate lineage requires spatial-boundary SHA256")?;
        let spatial_distance_mode = validate_spatial_boundary(topology)?.to_string();
        let context = Self {
            schema_kind: "prism_ghost_coordinate_context".into(),
            schema_version: GHOST_COORDINATE_SCHEMA_VERSION,
            run_id: ghost.run_id.clone(),
            replica_id: ghost.replica_id.clone(),
            replica_seed: ghost.replica_seed,
            stream_seed: ghost.stream_seed,
            stream_id: ghost.stream_id,
            topology_sha256: topology_sha256.into(),
            selected_engine_sha256: ghost.selected_engine_sha256.clone(),
            ghost_record_abi_sha256: ghost.record_abi_sha256.clone(),
            ghost_record_source_sha256: ghost.record_source_sha256.clone(),
            ghost_segment_context_digest_sha256: ghost.digest_sha256()?,
            thermodynamic_condition_id: ghost.thermodynamic_condition_id.clone(),
            protocol_id: ghost.protocol_id.clone(),
            protocol_state_hash: ghost.protocol_state_hash.clone(),
            phase_schedule_digest_sha256: ghost.phase_schedule_digest_sha256.clone(),
            atom_count: topology.n_atoms as u64,
            analysis_residue_count: topology.n_analysis_residues.unwrap_or_default() as u32,
            analysis_atom_count: analysis_ids.iter().filter(|value| **value >= 0).count() as u64,
            atom_analysis_residue_ids_sha256: analysis.atom_analysis_residue_ids_sha256.clone(),
            analysis_residue_contract_sha256: analysis.contract_sha256.clone(),
            molecule_identity_table_sha256: molecules.table_sha256.clone(),
            spatial_boundary_contract_sha256: boundary_sha.clone(),
            spatial_distance_mode,
            topology_projection_mode: resolved.projection_mode,
            topology_projection_authority_sha256: resolved.authority_contract_sha256,
            topology_projection_artifact_sha256: resolved.authority_artifact_sha256,
            topology_projection_source_sha256: resolved.authority_source_sha256,
            coordinate_population: "SIGNED_ANALYSIS_ATOMS_ORDERED_BY_TOPOLOGY_INDEX".into(),
            coordinate_scalar: "IEEE754_BINARY32_LE".into(),
            coordinate_unit: "angstrom".into(),
            atom_order_contract: "topology_atom_index_where_atom_analysis_residue_id_ge_zero"
                .into(),
            atom_order_sha256: resolved.atom_order_sha256,
            local_shell_cutoff_angstrom,
            producer_source_sha256: coordinate_producer_source_sha256(),
        };
        context.verify()?;
        Ok(context)
    }

    pub fn verify(&self) -> Result<()> {
        if self.schema_kind != "prism_ghost_coordinate_context"
            || self.schema_version != GHOST_COORDINATE_SCHEMA_VERSION
            || self.run_id.trim().is_empty()
            || self.replica_id.trim().is_empty()
            || self.thermodynamic_condition_id.trim().is_empty()
            || self.protocol_id.trim().is_empty()
            || self.atom_count == 0
            || self.analysis_residue_count == 0
            || self.analysis_atom_count == 0
            || self.analysis_atom_count > self.atom_count
            || self.coordinate_population != "SIGNED_ANALYSIS_ATOMS_ORDERED_BY_TOPOLOGY_INDEX"
            || self.coordinate_scalar != "IEEE754_BINARY32_LE"
            || self.coordinate_unit != "angstrom"
            || self.atom_order_contract
                != "topology_atom_index_where_atom_analysis_residue_id_ge_zero"
            || !matches!(
                self.spatial_distance_mode.as_str(),
                "PERIODIC_TRICLINIC_MINIMUM_IMAGE" | "NONPERIODIC_DIRECT_DISTANCE"
            )
            || !matches!(
                self.topology_projection_mode.as_str(),
                "NATIVE_TOPOLOGY_V3"
                    | "NATIVE_TOPOLOGY_V3_LEGACY_OPEN_BOUNDARY_NORMALIZED"
                    | "HASH_BOUND_LEGACY_SIDECAR"
            )
            || !self.local_shell_cutoff_angstrom.is_finite()
            || self.local_shell_cutoff_angstrom <= 0.0
        {
            bail!("invalid Ghost coordinate context header or population");
        }
        for (name, value) in [
            ("topology_sha256", &self.topology_sha256),
            ("selected_engine_sha256", &self.selected_engine_sha256),
            ("ghost_record_abi_sha256", &self.ghost_record_abi_sha256),
            (
                "ghost_record_source_sha256",
                &self.ghost_record_source_sha256,
            ),
            (
                "ghost_segment_context_digest_sha256",
                &self.ghost_segment_context_digest_sha256,
            ),
            ("protocol_state_hash", &self.protocol_state_hash),
            (
                "phase_schedule_digest_sha256",
                &self.phase_schedule_digest_sha256,
            ),
            (
                "atom_analysis_residue_ids_sha256",
                &self.atom_analysis_residue_ids_sha256,
            ),
            (
                "analysis_residue_contract_sha256",
                &self.analysis_residue_contract_sha256,
            ),
            (
                "molecule_identity_table_sha256",
                &self.molecule_identity_table_sha256,
            ),
            (
                "spatial_boundary_contract_sha256",
                &self.spatial_boundary_contract_sha256,
            ),
            (
                "topology_projection_authority_sha256",
                &self.topology_projection_authority_sha256,
            ),
            (
                "topology_projection_artifact_sha256",
                &self.topology_projection_artifact_sha256,
            ),
            (
                "topology_projection_source_sha256",
                &self.topology_projection_source_sha256,
            ),
            ("atom_order_sha256", &self.atom_order_sha256),
            ("producer_source_sha256", &self.producer_source_sha256),
        ] {
            require_sha256(value, name)?;
        }
        Ok(())
    }

    pub fn digest_sha256(&self) -> Result<String> {
        self.verify()?;
        canonical_sha256(self)
    }
}

/// Exact identity used by downstream Path-B/fiber joins.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GhostCoordinateKey {
    pub stream_id: u16,
    pub step_idx: u64,
    pub ghost_site_id: u32,
    pub ghost_record_sha256: String,
}

/// One local water/ion/fragment coordinate in the separately indexed shell.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactLocalShellAtom {
    pub atom_index: u32,
    pub molecule_id: u64,
    pub role: String,
    pub position_angstrom: [f32; 3],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GhostCoordinateFrameEntry {
    pub step_idx: u64,
    pub frame_ordinal: u64,
    pub atom_count: u64,
    pub byte_offset: u64,
    pub byte_length: u64,
    pub coordinate_payload_sha256: String,
    pub frame_record_sha256: String,
    pub source_full_frame_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GhostCoordinateJoinEntry {
    pub key: GhostCoordinateKey,
    pub stream_record_ordinal: u64,
    pub frame_idx: u64,
    pub ghost_segment_ordinal: u64,
    pub ghost_segment_file: String,
    pub ghost_segment_file_sha256: String,
    pub ghost_payload_byte_offset: u64,
    pub ghost_byte_length: u32,
    pub ghost_centroid_angstrom: [f32; 3],
    pub cluster_residue_membership_sha256: String,
    pub thermodynamic_condition_id: String,
    pub protocol_phase: u8,
    pub protocol_state_hash: String,
    pub phase_schedule_digest_sha256: String,
    pub coordinate_frame_ordinal: u64,
    pub coordinate_frame_byte_offset: u64,
    pub coordinate_frame_byte_length: u64,
    pub coordinate_frame_sha256: String,
    pub local_shell_status: String,
    pub local_shell_cutoff_angstrom: f64,
    pub local_shell_atom_count: u64,
    pub local_shell_byte_offset: u64,
    pub local_shell_byte_length: u64,
    pub local_shell_payload_sha256: String,
    pub excluded_population_scanned: u64,
    pub join_byte_offset: u64,
    pub join_byte_length: u64,
    pub join_record_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GhostCoordinateSegmentEntry {
    pub segment_ordinal: u64,
    pub first_join_ordinal: u64,
    pub frame_count: u64,
    pub join_count: u64,
    pub file: String,
    pub file_sha256: String,
    pub previous_file_sha256: String,
    pub context_digest_sha256: String,
    pub ghost_segment_ordinal: u64,
    pub ghost_segment_file_sha256: String,
    pub payload_bytes: u64,
    pub fsync_pass: bool,
    pub frames: Vec<GhostCoordinateFrameEntry>,
    pub joins: Vec<GhostCoordinateJoinEntry>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GhostCoordinateManifest {
    pub schema_kind: String,
    pub schema_version: u32,
    pub context: GhostCoordinateContext,
    pub context_digest_sha256: String,
    pub requested_ghost_records: u64,
    pub persisted_coordinate_frames: u64,
    pub persisted_ghost_joins: u64,
    pub segments: Vec<GhostCoordinateSegmentEntry>,
    pub completed: bool,
    pub requested_equals_persisted: bool,
}

impl GhostCoordinateManifest {
    fn empty(context: GhostCoordinateContext) -> Result<Self> {
        let context_digest_sha256 = context.digest_sha256()?;
        Ok(Self {
            schema_kind: "prism_ghost_coordinate_manifest".into(),
            schema_version: GHOST_COORDINATE_SCHEMA_VERSION,
            context,
            context_digest_sha256,
            requested_ghost_records: 0,
            persisted_coordinate_frames: 0,
            persisted_ghost_joins: 0,
            segments: Vec::new(),
            completed: false,
            requested_equals_persisted: true,
        })
    }
}

/// Capacity proof for a declared campaign upper bound.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GhostCoordinateCapacityPlan {
    pub analysis_atom_count: u64,
    pub streams: u64,
    pub maximum_ghost_producing_steps_per_stream: u64,
    pub analysis_frame_bytes: u64,
    pub analysis_frame_campaign_bytes: u64,
    pub margin_numerator: u64,
    pub margin_denominator: u64,
    pub required_disk_bytes: u64,
    pub available_disk_bytes: u64,
    pub required_gpu_staging_bytes: u64,
    pub available_gpu_bytes: u64,
    pub disk_pass: bool,
    pub gpu_pass: bool,
}

impl GhostCoordinateCapacityPlan {
    pub fn require_pass(&self) -> Result<()> {
        if !self.disk_pass || !self.gpu_pass {
            bail!(
                "Ghost coordinate preflight failed: required_disk={} available_disk={} required_gpu={} available_gpu={}",
                self.required_disk_bytes,
                self.available_disk_bytes,
                self.required_gpu_staging_bytes,
                self.available_gpu_bytes,
            );
        }
        Ok(())
    }
}

/// Conservative exact analysis-coordinate capacity admission. The 5/4 disk multiplier
/// reserves segment headers, per-site shell/index payloads and atomic-manifest
/// churn without using a fixed ring as scientific storage.
pub fn preflight_coordinate_capacity(
    output_dir: &Path,
    analysis_atom_count: u64,
    streams: u64,
    maximum_ghost_producing_steps_per_stream: u64,
    available_gpu_bytes: u64,
) -> Result<GhostCoordinateCapacityPlan> {
    if analysis_atom_count == 0 || streams == 0 || maximum_ghost_producing_steps_per_stream == 0 {
        bail!("coordinate capacity inputs must be non-zero");
    }
    let analysis_frame_bytes = analysis_atom_count
        .checked_mul(3)
        .and_then(|value| value.checked_mul(4))
        .context("analysis coordinate frame byte overflow")?;
    let analysis_frame_campaign_bytes = analysis_frame_bytes
        .checked_mul(streams)
        .and_then(|value| value.checked_mul(maximum_ghost_producing_steps_per_stream))
        .context("coordinate campaign byte overflow")?;
    let required_disk_bytes = analysis_frame_campaign_bytes
        .checked_mul(5)
        .context("coordinate disk margin overflow")?
        / 4;
    let available_disk_bytes = available_bytes(output_dir)?;
    let required_gpu_staging_bytes = analysis_frame_bytes
        .checked_mul(streams)
        .context("coordinate GPU staging overflow")?;
    Ok(GhostCoordinateCapacityPlan {
        analysis_atom_count,
        streams,
        maximum_ghost_producing_steps_per_stream,
        analysis_frame_bytes,
        analysis_frame_campaign_bytes,
        margin_numerator: 5,
        margin_denominator: 4,
        required_disk_bytes,
        available_disk_bytes,
        required_gpu_staging_bytes,
        available_gpu_bytes,
        disk_pass: available_disk_bytes >= required_disk_bytes,
        gpu_pass: available_gpu_bytes >= required_gpu_staging_bytes,
    })
}

/// One exact full-system source frame supplied at a synchronized producer
/// boundary. Only the signed analysis-atom population is persisted in the
/// coordinate frame; the full source is scanned once to materialize separate
/// exact water/ion/fragment shells and is SHA-bound in the frame header.
#[derive(Clone, Debug)]
pub struct ExactGhostCoordinateFrame {
    pub step_idx: u64,
    pub positions_angstrom: Vec<f32>,
}

/// Minimal immutable topology projection retained by each stream writer.
/// Keeping a complete 155,979-atom topology clone per stream would also clone
/// bonds, angles, dihedrals and exact nonbonded tables that coordinate
/// persistence never consumes. This projection retains only the signed
/// analysis map, molecule roles and PBC matrices required for exact compaction
/// and local-shell scans.
struct GhostCoordinateTopologyView {
    n_atoms: usize,
    analysis_ids: Vec<i32>,
    roles: Vec<AtomRole>,
    spatial_distance: SpatialDistance,
}

impl GhostCoordinateTopologyView {
    fn from_topology(topology: &PrismPrepTopology, topology_sha256: &str) -> Result<Self> {
        let resolved = resolve_ghost_topology(topology, topology_sha256)?;
        let topology = &resolved.topology;
        Ok(Self {
            n_atoms: topology.n_atoms,
            analysis_ids: topology
                .atom_analysis_residue_ids
                .as_ref()
                .context("coordinate writer requires signed atom analysis map")?
                .clone(),
            roles: molecule_role_map(topology)?,
            spatial_distance: spatial_distance(topology)?,
        })
    }
}

/// Durable single-writer coordinate sink for one Ghost stream.
pub struct GhostCoordinateSegmentWriter {
    root: PathBuf,
    manifest_path: PathBuf,
    topology: GhostCoordinateTopologyView,
    topo_stem: String,
    manifest: GhostCoordinateManifest,
    seen_keys: BTreeSet<GhostCoordinateKey>,
    seen_steps: BTreeSet<u64>,
    persistence_work: GhostCoordinatePersistenceWork,
}

/// Actual metadata bytes serialized and durably written by this writer instance.
/// Segment payload bytes are unchanged and are deliberately accounted separately.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GhostCoordinatePersistenceWork {
    pub journal_entries_written: u64,
    pub journal_bytes_written: u64,
    pub checkpoint_writes: u64,
    pub checkpoint_bytes_written: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GhostCoordinateContinuationV1 {
    pub output: crate::checkpoint_output::OutputDirectoryBoundary,
    pub topo_stem: String,
    pub manifest: GhostCoordinateManifest,
    pub persistence_work: GhostCoordinatePersistenceWork,
}
impl GhostCoordinateSegmentWriter {
    pub fn checkpoint(&self, run_root: &Path) -> Result<GhostCoordinateContinuationV1> {
        let manifest = load_coordinate_manifest_with_journal(&self.root)?;
        if manifest != self.manifest {
            bail!("Ghost coordinate live manifest differs from durable boundary");
        }
        Ok(GhostCoordinateContinuationV1 {
            output: crate::checkpoint_output::capture_directory(
                run_root,
                &self.root,
                &["manifest.json"],
            )?,
            topo_stem: self.topo_stem.clone(),
            manifest,
            persistence_work: self.persistence_work.clone(),
        })
    }
    pub fn reopen_checkpoint(
        run_root: &Path,
        state: &GhostCoordinateContinuationV1,
        topology: &PrismPrepTopology,
        expected: &GhostCoordinateContext,
    ) -> Result<Self> {
        validate_stem(&state.topo_stem)?;
        validate_topology_against_context(topology, expected)?;
        if &state.manifest.context != expected {
            bail!("Ghost coordinate checkpoint context differs");
        }
        let root = crate::checkpoint_output::restore_directory(run_root, &state.output)?;
        let manifest = load_coordinate_manifest_with_journal(&root)?;
        if manifest != state.manifest {
            bail!("Ghost coordinate restored manifest differs");
        }
        let loaded = verify_manifest_segments(&root, &manifest, topology)?;
        let seen_keys = loaded.index.keys().cloned().collect();
        let seen_steps = loaded.frames.keys().map(|(_, step)| *step).collect();
        Ok(Self {
            manifest_path: root.join("manifest.json"),
            root,
            topology: GhostCoordinateTopologyView::from_topology(
                topology,
                &expected.topology_sha256,
            )?,
            topo_stem: state.topo_stem.clone(),
            manifest,
            seen_keys,
            seen_steps,
            persistence_work: state.persistence_work.clone(),
        })
    }

    pub fn create(
        output_dir: &Path,
        topo_stem: &str,
        topology: &PrismPrepTopology,
        context: GhostCoordinateContext,
    ) -> Result<Self> {
        validate_stem(topo_stem)?;
        validate_topology_against_context(topology, &context)?;
        let root = coordinate_segment_root(output_dir, topo_stem, context.stream_id);
        std::fs::create_dir_all(&root)?;
        fsync_directory(output_dir)?;
        let manifest_path = root.join("manifest.json");
        if manifest_path.exists() {
            bail!(
                "Ghost coordinate manifest already exists: {}",
                manifest_path.display()
            );
        }
        let topology_view =
            GhostCoordinateTopologyView::from_topology(topology, &context.topology_sha256)?;
        let manifest = GhostCoordinateManifest::empty(context)?;
        let checkpoint_bytes = write_manifest_atomic(&root, &manifest_path, &manifest)?;
        ensure_index_journal_directory(&root)?;
        Ok(Self {
            root,
            manifest_path,
            topology: topology_view,
            topo_stem: topo_stem.into(),
            manifest,
            seen_keys: BTreeSet::new(),
            seen_steps: BTreeSet::new(),
            persistence_work: GhostCoordinatePersistenceWork {
                checkpoint_writes: 1,
                checkpoint_bytes_written: checkpoint_bytes,
                ..Default::default()
            },
        })
    }

    /// Resume a verified incomplete chain.  Orphan `.partial` files are never
    /// trusted or deleted; their presence is a typed recovery blocker.
    pub fn resume(
        output_dir: &Path,
        topo_stem: &str,
        topology: &PrismPrepTopology,
        context: GhostCoordinateContext,
    ) -> Result<Self> {
        validate_stem(topo_stem)?;
        validate_topology_against_context(topology, &context)?;
        let root = coordinate_segment_root(output_dir, topo_stem, context.stream_id);
        let manifest_path = root.join("manifest.json");
        reject_partial_files(&root)?;
        ensure_index_journal_directory(&root)?;
        reject_partial_files(&root.join(INDEX_JOURNAL_DIRECTORY))?;
        let mut manifest = load_coordinate_manifest_with_journal(&root)?;
        if manifest.context != context
            || manifest.context_digest_sha256 != context.digest_sha256()?
        {
            bail!("Ghost coordinate resume context mismatch");
        }
        if manifest.completed {
            bail!("cannot resume completed Ghost coordinate stream");
        }
        reject_partial_files(&root)?;
        let mut persistence_work = GhostCoordinatePersistenceWork::default();
        recover_orphan_segments(&root, &mut manifest, &mut persistence_work)?;
        let loaded = verify_manifest_segments(&root, &manifest, topology)?;
        let seen_keys = loaded.index.keys().cloned().collect();
        let seen_steps = loaded.frames.keys().map(|(_, step)| *step).collect();
        Ok(Self {
            root,
            manifest_path,
            topology: GhostCoordinateTopologyView::from_topology(
                topology,
                &context.topology_sha256,
            )?,
            topo_stem: topo_stem.into(),
            manifest,
            seen_keys,
            seen_steps,
            persistence_work,
        })
    }

    /// Persist all coordinate frames and record joins for one already-durable
    /// Ghost segment. Every record step must have an exact full frame.
    pub fn append_ghost_segment<F>(
        &mut self,
        ghost_entry: &GhostSegmentEntry,
        ghost_payload: &[u8],
        frames: &[ExactGhostCoordinateFrame],
        mut phase_for_step: F,
    ) -> Result<()>
    where
        F: FnMut(u64) -> Option<u8>,
    {
        if self.manifest.completed {
            bail!("cannot append to completed Ghost coordinate stream");
        }
        if ghost_payload.len() != ghost_entry.payload_bytes as usize
            || ghost_payload.len() % GHOST_SEGMENT_RECORD_BYTES != 0
            || sha256_bytes(ghost_payload) != ghost_entry.payload_sha256
            || ghost_entry.record_count != (ghost_payload.len() / GHOST_SEGMENT_RECORD_BYTES) as u64
        {
            bail!("Ghost coordinate append payload/segment identity mismatch");
        }
        if ghost_entry.context_digest_sha256
            != self.manifest.context.ghost_segment_context_digest_sha256
        {
            bail!("Ghost coordinate append context digest mismatch");
        }
        let mut frame_map = BTreeMap::<u64, &ExactGhostCoordinateFrame>::new();
        for frame in frames {
            validate_full_frame_population(self.topology.n_atoms, frame)?;
            if frame_map.insert(frame.step_idx, frame).is_some() {
                bail!(
                    "duplicate exact coordinate frame for step {}",
                    frame.step_idx
                );
            }
        }
        let mut records = ghost_payload
            .chunks_exact(GHOST_SEGMENT_RECORD_BYTES)
            .enumerate()
            .map(|(local, record)| decode_record_request(record, local as u64))
            .collect::<Result<Vec<_>>>()?;
        for record in &mut records {
            record.key.stream_id = self.manifest.context.stream_id;
        }
        let unique_record_keys = records
            .iter()
            .map(|record| record.key.clone())
            .collect::<BTreeSet<_>>();
        if unique_record_keys.len() != records.len() {
            bail!("duplicate Ghost coordinate key within durable segment");
        }
        let requested_steps = records
            .iter()
            .map(|record| record.step_idx)
            .collect::<BTreeSet<_>>();
        if requested_steps != frame_map.keys().copied().collect::<BTreeSet<_>>() {
            bail!(
                "exact Ghost coordinate population mismatch: record_steps={:?} supplied_steps={:?}; interpolation is forbidden",
                requested_steps,
                frame_map.keys().collect::<Vec<_>>()
            );
        }
        for record in &records {
            if self.seen_keys.contains(&record.key) {
                bail!("duplicate Ghost coordinate key: {:?}", record.key);
            }
        }
        for step in &requested_steps {
            if self.seen_steps.contains(step) {
                bail!(
                    "duplicate Ghost coordinate step {} across durable segments",
                    step
                );
            }
        }

        let segment_ordinal = self.manifest.segments.len() as u64;
        if ghost_entry.segment_ordinal != segment_ordinal {
            bail!(
                "Ghost/coordinate segment ordinal gap: ghost={} coordinate={}",
                ghost_entry.segment_ordinal,
                segment_ordinal
            );
        }
        let previous_file_sha256 = self
            .manifest
            .segments
            .last()
            .map(|entry| entry.file_sha256.clone())
            .unwrap_or_else(|| ZERO_SHA256.into());
        let file_name = format!("segment_{segment_ordinal:08}.ghostcoord");
        let final_path = self.root.join(&file_name);
        let partial_path = self.root.join(format!(".{file_name}.partial"));
        if final_path.exists() || partial_path.exists() {
            bail!("duplicate Ghost coordinate segment target");
        }

        let mut bytes = vec![0u8; GHOST_COORDINATE_HEADER_BYTES];
        let mut frame_entries = Vec::new();
        let mut frame_by_step = BTreeMap::<u64, GhostCoordinateFrameEntry>::new();
        for (frame_ordinal, (step, frame)) in frame_map.iter().enumerate() {
            let byte_offset = bytes.len() as u64;
            let analysis_positions = compact_analysis_positions_from_map(
                self.topology.n_atoms,
                &self.topology.analysis_ids,
                &frame.positions_angstrom,
            )?;
            let coordinate_payload = f32_bytes(&analysis_positions);
            let coordinate_payload_sha256 = sha256_bytes(&coordinate_payload);
            let mut header = vec![0u8; FRAME_HEADER_BYTES];
            header[0..8].copy_from_slice(&FRAME_MAGIC);
            put_u32(&mut header, 8, GHOST_COORDINATE_SCHEMA_VERSION);
            put_u32(&mut header, 12, FRAME_HEADER_BYTES as u32);
            put_u64(&mut header, 16, *step);
            put_u64(&mut header, 24, self.manifest.context.analysis_atom_count);
            put_u64(&mut header, 32, coordinate_payload.len() as u64);
            put_sha256(&mut header, 40, &coordinate_payload_sha256)?;
            let source_full_frame_sha256 = sha256_bytes(&f32_bytes(&frame.positions_angstrom));
            put_sha256(&mut header, 72, &source_full_frame_sha256)?;
            bytes.extend_from_slice(&header);
            bytes.extend_from_slice(&coordinate_payload);
            let byte_length = (FRAME_HEADER_BYTES + coordinate_payload.len()) as u64;
            let frame_record_sha256 =
                sha256_bytes(&bytes[byte_offset as usize..(byte_offset + byte_length) as usize]);
            let entry = GhostCoordinateFrameEntry {
                step_idx: *step,
                frame_ordinal: frame_ordinal as u64,
                atom_count: self.manifest.context.analysis_atom_count,
                byte_offset,
                byte_length,
                coordinate_payload_sha256,
                frame_record_sha256,
                source_full_frame_sha256,
            };
            frame_by_step.insert(*step, entry.clone());
            frame_entries.push(entry);
        }

        let mut join_entries = Vec::new();
        for record in &records {
            let frame = frame_map[&record.step_idx];
            let frame_entry = &frame_by_step[&record.step_idx];
            let shell = exact_local_shell_from_population(
                self.topology.n_atoms,
                &self.topology.roles,
                &frame.positions_angstrom,
                record.centroid,
                self.manifest.context.local_shell_cutoff_angstrom,
                self.topology.spatial_distance,
            )?;
            let shell_bytes = encode_shell(&shell)?;
            let shell_sha256 = sha256_bytes(&shell_bytes);
            let metadata = PersistedJoinMetadata {
                schema_version: GHOST_COORDINATE_SCHEMA_VERSION,
                key: record.key.clone(),
                stream_record_ordinal: ghost_entry.first_record_ordinal + record.local_index,
                frame_idx: record.frame_idx,
                ghost_segment_ordinal: ghost_entry.segment_ordinal,
                ghost_segment_file: format!(
                    "{}_stream{:02}_ghost_segments/{}",
                    self.topo_stem, self.manifest.context.stream_id, ghost_entry.file
                ),
                ghost_segment_file_sha256: ghost_entry.file_sha256.clone(),
                ghost_payload_byte_offset: ghost_entry.payload_offset
                    + record.local_index * GHOST_SEGMENT_RECORD_BYTES as u64,
                ghost_byte_length: GHOST_SEGMENT_RECORD_BYTES as u32,
                ghost_centroid_angstrom: record.centroid,
                cluster_residue_membership_sha256: record.membership_sha256.clone(),
                thermodynamic_condition_id: self
                    .manifest
                    .context
                    .thermodynamic_condition_id
                    .clone(),
                protocol_phase: phase_for_step(record.step_idx).with_context(|| {
                    format!(
                        "EVIDENCE_NOT_FOUND: protocol phase for Ghost step {}",
                        record.step_idx
                    )
                })?,
                protocol_state_hash: self.manifest.context.protocol_state_hash.clone(),
                phase_schedule_digest_sha256: self
                    .manifest
                    .context
                    .phase_schedule_digest_sha256
                    .clone(),
                coordinate_frame_ordinal: frame_entry.frame_ordinal,
                coordinate_frame_byte_offset: frame_entry.byte_offset,
                coordinate_frame_byte_length: frame_entry.byte_length,
                coordinate_frame_sha256: frame_entry.frame_record_sha256.clone(),
                local_shell_status: "COMPLETE_EXACT_HASH_BOUND_FULL_POPULATION_SCAN".into(),
                local_shell_cutoff_angstrom: self.manifest.context.local_shell_cutoff_angstrom,
                local_shell_atom_count: shell.len() as u64,
                local_shell_payload_sha256: shell_sha256.clone(),
                excluded_population_scanned: self
                    .topology
                    .roles
                    .iter()
                    .filter(|role| role.analysis_id < 0)
                    .count() as u64,
            };
            let metadata_bytes = serde_json::to_vec(&metadata)?;
            let join_byte_offset = bytes.len() as u64;
            bytes.extend_from_slice(&JOIN_MAGIC);
            bytes.extend_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
            bytes.extend_from_slice(&(shell_bytes.len() as u64).to_le_bytes());
            bytes.extend_from_slice(&metadata_bytes);
            let local_shell_byte_offset = bytes.len() as u64;
            bytes.extend_from_slice(&shell_bytes);
            let join_byte_length = bytes.len() as u64 - join_byte_offset;
            let join_record_sha256 = sha256_bytes(
                &bytes[join_byte_offset as usize..(join_byte_offset + join_byte_length) as usize],
            );
            join_entries.push(GhostCoordinateJoinEntry {
                key: metadata.key,
                stream_record_ordinal: metadata.stream_record_ordinal,
                frame_idx: metadata.frame_idx,
                ghost_segment_ordinal: metadata.ghost_segment_ordinal,
                ghost_segment_file: metadata.ghost_segment_file,
                ghost_segment_file_sha256: metadata.ghost_segment_file_sha256,
                ghost_payload_byte_offset: metadata.ghost_payload_byte_offset,
                ghost_byte_length: metadata.ghost_byte_length,
                ghost_centroid_angstrom: metadata.ghost_centroid_angstrom,
                cluster_residue_membership_sha256: metadata.cluster_residue_membership_sha256,
                thermodynamic_condition_id: metadata.thermodynamic_condition_id,
                protocol_phase: metadata.protocol_phase,
                protocol_state_hash: metadata.protocol_state_hash,
                phase_schedule_digest_sha256: metadata.phase_schedule_digest_sha256,
                coordinate_frame_ordinal: metadata.coordinate_frame_ordinal,
                coordinate_frame_byte_offset: metadata.coordinate_frame_byte_offset,
                coordinate_frame_byte_length: metadata.coordinate_frame_byte_length,
                coordinate_frame_sha256: metadata.coordinate_frame_sha256,
                local_shell_status: metadata.local_shell_status,
                local_shell_cutoff_angstrom: metadata.local_shell_cutoff_angstrom,
                local_shell_atom_count: metadata.local_shell_atom_count,
                local_shell_byte_offset,
                local_shell_byte_length: shell_bytes.len() as u64,
                local_shell_payload_sha256: shell_sha256,
                excluded_population_scanned: metadata.excluded_population_scanned,
                join_byte_offset,
                join_byte_length,
                join_record_sha256,
            });
        }

        let segment_payload_bytes = (bytes.len() - GHOST_COORDINATE_HEADER_BYTES) as u64;
        encode_segment_header(
            &mut bytes[..GHOST_COORDINATE_HEADER_BYTES],
            segment_ordinal,
            frame_entries.len() as u64,
            join_entries.len() as u64,
            segment_payload_bytes,
            &self.manifest.context_digest_sha256,
            &previous_file_sha256,
            &ghost_entry.file_sha256,
        )?;
        ensure_disk_for_write(&self.root, bytes.len() as u64)?;
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
            .open(&partial_path)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&partial_path, &final_path)?;
        fsync_directory(&self.root)?;
        let file_sha256 = sha256_file(&final_path)?;
        let entry = GhostCoordinateSegmentEntry {
            segment_ordinal,
            first_join_ordinal: self.manifest.persisted_ghost_joins,
            frame_count: frame_entries.len() as u64,
            join_count: join_entries.len() as u64,
            file: file_name,
            file_sha256,
            previous_file_sha256,
            context_digest_sha256: self.manifest.context_digest_sha256.clone(),
            ghost_segment_ordinal: ghost_entry.segment_ordinal,
            ghost_segment_file_sha256: ghost_entry.file_sha256.clone(),
            payload_bytes: segment_payload_bytes,
            fsync_pass: true,
            frames: frame_entries,
            joins: join_entries,
        };
        self.manifest.requested_ghost_records += ghost_entry.record_count;
        self.manifest.persisted_coordinate_frames += entry.frame_count;
        self.manifest.persisted_ghost_joins += entry.join_count;
        self.manifest.segments.push(entry.clone());
        self.manifest.requested_equals_persisted =
            self.manifest.requested_ghost_records == self.manifest.persisted_ghost_joins;
        let journal_bytes = write_index_journal_entry(&self.root, &entry)?;
        self.persistence_work.journal_entries_written += 1;
        self.persistence_work.journal_bytes_written += journal_bytes;
        for join in entry.joins {
            self.seen_keys.insert(join.key);
        }
        for frame in entry.frames {
            self.seen_steps.insert(frame.step_idx);
        }
        Ok(())
    }

    pub fn complete(&mut self, ghost_produced_records: u64) -> Result<()> {
        if ghost_produced_records != self.manifest.requested_ghost_records
            || self.manifest.requested_ghost_records != self.manifest.persisted_ghost_joins
        {
            bail!(
                "Ghost coordinate conservation failed: ghost_produced={} requested={} persisted={}",
                ghost_produced_records,
                self.manifest.requested_ghost_records,
                self.manifest.persisted_ghost_joins
            );
        }
        self.manifest.completed = true;
        self.manifest.requested_equals_persisted = true;
        let checkpoint_bytes =
            write_manifest_atomic(&self.root, &self.manifest_path, &self.manifest)?;
        self.persistence_work.checkpoint_writes += 1;
        self.persistence_work.checkpoint_bytes_written += checkpoint_bytes;
        fsync_directory(&self.root)
    }

    pub fn manifest(&self) -> &GhostCoordinateManifest {
        &self.manifest
    }

    pub fn persistence_work(&self) -> &GhostCoordinatePersistenceWork {
        &self.persistence_work
    }
}

#[derive(Clone, Debug)]
struct DecodedRecordRequest {
    local_index: u64,
    key: GhostCoordinateKey,
    step_idx: u64,
    frame_idx: u64,
    centroid: [f32; 3],
    membership_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PersistedJoinMetadata {
    schema_version: u32,
    key: GhostCoordinateKey,
    stream_record_ordinal: u64,
    frame_idx: u64,
    ghost_segment_ordinal: u64,
    ghost_segment_file: String,
    ghost_segment_file_sha256: String,
    ghost_payload_byte_offset: u64,
    ghost_byte_length: u32,
    ghost_centroid_angstrom: [f32; 3],
    cluster_residue_membership_sha256: String,
    thermodynamic_condition_id: String,
    protocol_phase: u8,
    protocol_state_hash: String,
    phase_schedule_digest_sha256: String,
    coordinate_frame_ordinal: u64,
    coordinate_frame_byte_offset: u64,
    coordinate_frame_byte_length: u64,
    coordinate_frame_sha256: String,
    local_shell_status: String,
    local_shell_cutoff_angstrom: f64,
    local_shell_atom_count: u64,
    local_shell_payload_sha256: String,
    excluded_population_scanned: u64,
}

#[derive(Clone, Debug)]
struct AtomRole {
    analysis_id: i32,
    molecule_id: u64,
    role: String,
}

/// Fully decoded and cross-verified coordinate row.
#[derive(Clone, Debug)]
pub struct VerifiedGhostCoordinate {
    pub manifest_sha256: String,
    pub segment_file: String,
    pub segment_file_sha256: String,
    pub entry: GhostCoordinateJoinEntry,
    pub positions_angstrom: Vec<f32>,
    pub local_shell: Vec<ExactLocalShellAtom>,
}

/// Aggregate loader output. `equality_pass` is true only after requested,
/// persisted, decoded and indexed record populations are identical.
#[derive(Clone, Debug)]
pub struct GhostCoordinateSegmentLoad {
    pub contexts: BTreeMap<u16, GhostCoordinateContext>,
    pub frames: BTreeMap<(u16, u64), Vec<f32>>,
    pub index: BTreeMap<GhostCoordinateKey, VerifiedGhostCoordinate>,
    pub requested_records: u64,
    pub persisted_records: u64,
    pub decoded_records: u64,
    pub indexed_records: u64,
    pub equality_pass: bool,
}

/// End-to-end Ghost byte/lattice/coordinate conservation receipt.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GhostCoordinateJoinConservation {
    pub ghost_produced_records: u64,
    pub ghost_persisted_records: u64,
    pub ghost_decoded_records: u64,
    pub ghost_lattice_nodes: u64,
    pub coordinate_requested_records: u64,
    pub coordinate_persisted_records: u64,
    pub coordinate_decoded_records: u64,
    pub coordinate_indexed_records: u64,
    pub exact_joined_records: u64,
    pub equality_pass: bool,
    pub receipt_sha256: String,
}

/// Cross-verified coordinate population in the same identity domain consumed
/// by Native Lattice, Path-B and fibers.
#[derive(Clone, Debug)]
pub struct GhostCoordinateLineageJoin {
    pub by_key: BTreeMap<GhostCoordinateKey, VerifiedGhostCoordinate>,
    pub conservation: GhostCoordinateJoinConservation,
}

/// Join exact coordinate rows to the authenticated Ghost byte lineage and
/// lattice projection. Missing, extra, cross-stream, phase-mismatched or
/// byte-offset-mismatched rows fail as `EVIDENCE_NOT_FOUND`; no uncovered node
/// can reach a cavity or Path-B claim.
pub fn join_ghost_lattice_coordinates(
    coordinates: &GhostCoordinateSegmentLoad,
    ghost: &crate::ghost_phase_lattice::GhostSegmentLatticeLoad,
) -> Result<GhostCoordinateLineageJoin> {
    if !coordinates.equality_pass || !ghost.equality_pass {
        bail!("EVIDENCE_NOT_FOUND: Ghost or coordinate conservation is not closed");
    }
    let mut joined = BTreeMap::new();
    for lineage in &ghost.record_lineage {
        let key = GhostCoordinateKey {
            stream_id: lineage.stream_id,
            step_idx: lineage.step_idx,
            ghost_site_id: lineage.site_id,
            ghost_record_sha256: lineage.record_sha256.clone(),
        };
        let row = coordinates.index.get(&key).with_context(|| {
            format!(
                "EVIDENCE_NOT_FOUND: exact coordinate row for stream={} step={} site={} record={}",
                key.stream_id, key.step_idx, key.ghost_site_id, key.ghost_record_sha256,
            )
        })?;
        let context = coordinates
            .contexts
            .get(&lineage.stream_id)
            .context("EVIDENCE_NOT_FOUND: coordinate stream context")?;
        let membership_sha256 = canonical_sha256(&lineage.cluster_residue_membership)?;
        if context.run_id != lineage.run_id
            || context.replica_id != lineage.replica_id
            || context.replica_seed != lineage.replica_seed
            || context.stream_seed != lineage.stream_seed
            || context.topology_sha256 != lineage.topology_sha256
            || context.selected_engine_sha256 != lineage.selected_engine_sha256
            || context.ghost_record_abi_sha256 != lineage.record_abi_sha256
            || context.ghost_record_source_sha256 != lineage.record_source_sha256
            || context.ghost_segment_context_digest_sha256 != lineage.segment_context_digest_sha256
            || context.thermodynamic_condition_id != lineage.condition_id
            || context.protocol_state_hash != lineage.protocol_state_hash
            || context.phase_schedule_digest_sha256 != lineage.phase_schedule_digest_sha256
            || row.entry.stream_record_ordinal != lineage.stream_record_ordinal
            || row.entry.ghost_segment_ordinal != lineage.segment_ordinal
            || row.entry.ghost_segment_file != lineage.segment_file
            || row.entry.ghost_segment_file_sha256 != lineage.segment_file_sha256
            || row.entry.ghost_payload_byte_offset != lineage.payload_byte_offset
            || row.entry.ghost_byte_length != lineage.byte_length
            || row.entry.frame_idx != lineage.frame_idx
            || row.entry.protocol_phase != lineage.protocol_phase
            || row.entry.cluster_residue_membership_sha256 != membership_sha256
            || row.positions_angstrom.len() as u64 != context.analysis_atom_count * 3
        {
            bail!(
                "Ghost coordinate row differs from authenticated byte/lattice lineage: {:?}",
                key
            );
        }
        if joined.insert(key.clone(), row.clone()).is_some() {
            bail!("duplicate exact Ghost coordinate join: {:?}", key);
        }
    }
    if joined.len() != coordinates.index.len() {
        bail!(
            "Ghost coordinate index contains {} unconsumed rows",
            coordinates.index.len() - joined.len()
        );
    }
    let mut conservation = GhostCoordinateJoinConservation {
        ghost_produced_records: ghost.produced_records,
        ghost_persisted_records: ghost.persisted_records,
        ghost_decoded_records: ghost.decoded_records,
        ghost_lattice_nodes: ghost.lattice_node_lineage_count,
        coordinate_requested_records: coordinates.requested_records,
        coordinate_persisted_records: coordinates.persisted_records,
        coordinate_decoded_records: coordinates.decoded_records,
        coordinate_indexed_records: coordinates.indexed_records,
        exact_joined_records: joined.len() as u64,
        equality_pass: false,
        receipt_sha256: ZERO_SHA256.into(),
    };
    let counts = [
        conservation.ghost_produced_records,
        conservation.ghost_persisted_records,
        conservation.ghost_decoded_records,
        conservation.ghost_lattice_nodes,
        conservation.coordinate_requested_records,
        conservation.coordinate_persisted_records,
        conservation.coordinate_decoded_records,
        conservation.coordinate_indexed_records,
        conservation.exact_joined_records,
    ];
    conservation.equality_pass = counts.windows(2).all(|pair| pair[0] == pair[1]);
    if !conservation.equality_pass {
        bail!("Ghost produced=persisted=decoded=lattice=coordinate=joined conservation failed");
    }
    conservation.receipt_sha256 = canonical_sha256(&serde_json::json!({
        "schema_kind": "prism_ghost_coordinate_join_conservation",
        "schema_version": 1,
        "counts": counts,
        "equality_pass": true,
    }))?;
    Ok(GhostCoordinateLineageJoin {
        by_key: joined,
        conservation,
    })
}

/// Discover and fully verify every coordinate stream below a run root.
pub fn load_verified_ghost_coordinate_segments(
    run_root: &Path,
    topology: &PrismPrepTopology,
) -> Result<GhostCoordinateSegmentLoad> {
    let mut manifests = Vec::new();
    for entry in std::fs::read_dir(run_root)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if entry.file_type()?.is_dir() && name.ends_with("_ghost_coordinate_segments") {
            let path = entry.path().join("manifest.json");
            let bytes = read_regular_nofollow(&path)?;
            let manifest: GhostCoordinateManifest = serde_json::from_slice(&bytes)?;
            manifests.push((entry.path(), bytes, manifest));
        }
    }
    manifests.sort_by_key(|(_, _, manifest)| manifest.context.stream_id);
    if manifests.is_empty() {
        bail!(
            "EVIDENCE_NOT_FOUND: no Ghost coordinate segment manifests in {}",
            run_root.display()
        );
    }
    let mut result = GhostCoordinateSegmentLoad {
        contexts: BTreeMap::new(),
        frames: BTreeMap::new(),
        index: BTreeMap::new(),
        requested_records: 0,
        persisted_records: 0,
        decoded_records: 0,
        indexed_records: 0,
        equality_pass: false,
    };
    for (root, manifest_bytes, manifest) in manifests {
        if !manifest.completed || !manifest.requested_equals_persisted {
            bail!(
                "EVIDENCE_NOT_FOUND: incomplete Ghost coordinate stream {}",
                manifest.context.stream_id
            );
        }
        validate_topology_against_context(topology, &manifest.context)?;
        let loaded = verify_manifest_segments(&root, &manifest, topology)?;
        let stream_id = manifest.context.stream_id;
        if result
            .contexts
            .insert(stream_id, manifest.context.clone())
            .is_some()
        {
            bail!("duplicate Ghost coordinate stream {}", stream_id);
        }
        for ((_, step), positions) in loaded.frames {
            if result.frames.insert((stream_id, step), positions).is_some() {
                bail!(
                    "duplicate Ghost coordinate frame stream={} step={}",
                    stream_id,
                    step
                );
            }
        }
        // The immutable source manifest binds every row in this stream.
        // Hash its exact bytes once; preserve identical per-row source identity.
        let manifest_sha256 = sha256_bytes(&manifest_bytes);
        for (key, mut row) in loaded.index {
            row.manifest_sha256 = manifest_sha256.clone();
            if result.index.insert(key.clone(), row).is_some() {
                bail!("duplicate Ghost coordinate index key: {:?}", key);
            }
        }
        result.requested_records += manifest.requested_ghost_records;
        result.persisted_records += manifest.persisted_ghost_joins;
        result.decoded_records += loaded.decoded_records;
    }
    result.indexed_records = result.index.len() as u64;
    result.equality_pass = result.requested_records == result.persisted_records
        && result.persisted_records == result.decoded_records
        && result.decoded_records == result.indexed_records;
    if !result.equality_pass {
        bail!(
            "Ghost coordinate conservation failed: requested={} persisted={} decoded={} indexed={}",
            result.requested_records,
            result.persisted_records,
            result.decoded_records,
            result.indexed_records
        );
    }
    Ok(result)
}

struct VerifiedManifestLoad {
    frames: BTreeMap<(u16, u64), Vec<f32>>,
    index: BTreeMap<GhostCoordinateKey, VerifiedGhostCoordinate>,
    decoded_records: u64,
}

fn verify_manifest_segments(
    root: &Path,
    manifest: &GhostCoordinateManifest,
    topology: &PrismPrepTopology,
) -> Result<VerifiedManifestLoad> {
    manifest.context.verify()?;
    let resolved = resolve_ghost_topology(topology, &manifest.context.topology_sha256)?;
    let effective_topology = &resolved.topology;
    if manifest.schema_kind != "prism_ghost_coordinate_manifest"
        || manifest.schema_version != GHOST_COORDINATE_SCHEMA_VERSION
        || manifest.context_digest_sha256 != manifest.context.digest_sha256()?
    {
        bail!("invalid Ghost coordinate manifest header/context");
    }
    let mut frames = BTreeMap::new();
    let mut index = BTreeMap::new();
    let mut previous = ZERO_SHA256.to_string();
    let mut expected_join_ordinal = 0u64;
    let mut decoded_records = 0u64;
    for (ordinal, segment) in manifest.segments.iter().enumerate() {
        if segment.segment_ordinal != ordinal as u64
            || segment.first_join_ordinal != expected_join_ordinal
            || segment.previous_file_sha256 != previous
            || segment.context_digest_sha256 != manifest.context_digest_sha256
            || !segment.fsync_pass
            || segment.frame_count != segment.frames.len() as u64
            || segment.join_count != segment.joins.len() as u64
        {
            bail!("Ghost coordinate manifest segment ordinal/chain/count failure");
        }
        validate_basename(&segment.file)?;
        let path = root.join(&segment.file);
        let bytes = read_regular_nofollow(&path)?;
        if sha256_bytes(&bytes) != segment.file_sha256
            || bytes.len() < GHOST_COORDINATE_HEADER_BYTES
            || segment.payload_bytes != (bytes.len() - GHOST_COORDINATE_HEADER_BYTES) as u64
        {
            bail!("Ghost coordinate segment content hash/length mismatch");
        }
        verify_segment_header(&bytes[..GHOST_COORDINATE_HEADER_BYTES], segment)?;
        let mut segment_frames = BTreeMap::<u64, Vec<f32>>::new();
        for frame in &segment.frames {
            let range = checked_range(frame.byte_offset, frame.byte_length, bytes.len())?;
            let record = &bytes[range];
            if sha256_bytes(record) != frame.frame_record_sha256
                || record.len() < FRAME_HEADER_BYTES
                || record[..8] != FRAME_MAGIC
                || read_u32(record, 8)? != GHOST_COORDINATE_SCHEMA_VERSION
                || read_u32(record, 12)? as usize != FRAME_HEADER_BYTES
                || read_u64(record, 16)? != frame.step_idx
                || read_u64(record, 24)? != manifest.context.analysis_atom_count
            {
                bail!("Ghost coordinate frame header/hash mismatch");
            }
            let payload_len = read_u64(record, 32)? as usize;
            if payload_len != manifest.context.analysis_atom_count as usize * 3 * 4
                || FRAME_HEADER_BYTES + payload_len != record.len()
            {
                bail!("Ghost coordinate frame payload shape mismatch");
            }
            let payload = &record[FRAME_HEADER_BYTES..];
            if sha256_bytes(payload) != frame.coordinate_payload_sha256
                || read_sha256(record, 40)? != frame.coordinate_payload_sha256
                || read_sha256(record, 72)? != frame.source_full_frame_sha256
            {
                bail!("Ghost coordinate frame payload/source hash mismatch");
            }
            let positions = decode_f32(payload)?;
            if positions.iter().any(|value| !value.is_finite()) {
                bail!("Ghost coordinate frame contains nonfinite value");
            }
            if segment_frames
                .insert(frame.step_idx, positions.clone())
                .is_some()
                || frames
                    .insert((manifest.context.stream_id, frame.step_idx), positions)
                    .is_some()
            {
                bail!("duplicate Ghost coordinate frame step");
            }
        }
        let role_map = molecule_role_map(effective_topology)?;
        let spatial_distance = spatial_distance(effective_topology)?;
        for join in &segment.joins {
            let range = checked_range(join.join_byte_offset, join.join_byte_length, bytes.len())?;
            let record = &bytes[range];
            if sha256_bytes(record) != join.join_record_sha256 || record[..8] != JOIN_MAGIC {
                bail!("Ghost coordinate join hash/magic mismatch");
            }
            let metadata_len = read_u32(record, 8)? as usize;
            let shell_len = read_u64(record, 12)? as usize;
            let metadata_end = 20usize
                .checked_add(metadata_len)
                .context("join metadata overflow")?;
            let shell_end = metadata_end
                .checked_add(shell_len)
                .context("join shell overflow")?;
            if shell_end != record.len() {
                bail!("Ghost coordinate join byte shape mismatch");
            }
            let metadata: PersistedJoinMetadata =
                serde_json::from_slice(&record[20..metadata_end])?;
            verify_join_metadata(&metadata, join, &manifest.context)?;
            let shell_bytes = &record[metadata_end..shell_end];
            if sha256_bytes(shell_bytes) != join.local_shell_payload_sha256
                || shell_len as u64 != join.local_shell_byte_length
                || join.local_shell_byte_offset != join.join_byte_offset + metadata_end as u64
            {
                bail!("Ghost coordinate local-shell byte lineage mismatch");
            }
            let shell = decode_shell(shell_bytes)?;
            if shell.len() as u64 != join.local_shell_atom_count {
                bail!("Ghost coordinate local-shell count mismatch");
            }
            let positions = segment_frames
                .get(&join.key.step_idx)
                .with_context(|| format!("EVIDENCE_NOT_FOUND: exact frame for {:?}", join.key))?;
            verify_persisted_local_shell(
                effective_topology.n_atoms,
                &role_map,
                &shell,
                join,
                spatial_distance,
            )?;
            require_sha256(&join.key.ghost_record_sha256, "ghost_record_sha256")?;
            let row = VerifiedGhostCoordinate {
                manifest_sha256: String::new(),
                segment_file: segment.file.clone(),
                segment_file_sha256: segment.file_sha256.clone(),
                entry: join.clone(),
                positions_angstrom: positions.clone(),
                local_shell: shell,
            };
            if index.insert(join.key.clone(), row).is_some() {
                bail!("duplicate Ghost coordinate join key");
            }
            decoded_records += 1;
        }
        expected_join_ordinal += segment.join_count;
        previous = segment.file_sha256.clone();
    }
    if manifest.persisted_coordinate_frames != frames.len() as u64
        || manifest.persisted_ghost_joins != decoded_records
        || manifest.requested_ghost_records != manifest.persisted_ghost_joins
    {
        bail!("Ghost coordinate manifest/decoded conservation mismatch");
    }
    Ok(VerifiedManifestLoad {
        frames,
        index,
        decoded_records,
    })
}

fn verify_join_metadata(
    metadata: &PersistedJoinMetadata,
    entry: &GhostCoordinateJoinEntry,
    context: &GhostCoordinateContext,
) -> Result<()> {
    if metadata.schema_version != GHOST_COORDINATE_SCHEMA_VERSION
        || metadata.key != entry.key
        || metadata.stream_record_ordinal != entry.stream_record_ordinal
        || metadata.frame_idx != entry.frame_idx
        || metadata.ghost_segment_ordinal != entry.ghost_segment_ordinal
        || metadata.ghost_segment_file != entry.ghost_segment_file
        || metadata.ghost_segment_file_sha256 != entry.ghost_segment_file_sha256
        || metadata.ghost_payload_byte_offset != entry.ghost_payload_byte_offset
        || metadata.ghost_byte_length != entry.ghost_byte_length
        || metadata.ghost_centroid_angstrom != entry.ghost_centroid_angstrom
        || metadata.cluster_residue_membership_sha256 != entry.cluster_residue_membership_sha256
        || metadata.thermodynamic_condition_id != context.thermodynamic_condition_id
        || metadata.thermodynamic_condition_id != entry.thermodynamic_condition_id
        || metadata.protocol_state_hash != context.protocol_state_hash
        || metadata.protocol_state_hash != entry.protocol_state_hash
        || metadata.phase_schedule_digest_sha256 != context.phase_schedule_digest_sha256
        || metadata.phase_schedule_digest_sha256 != entry.phase_schedule_digest_sha256
        || metadata.coordinate_frame_ordinal != entry.coordinate_frame_ordinal
        || metadata.coordinate_frame_byte_offset != entry.coordinate_frame_byte_offset
        || metadata.coordinate_frame_byte_length != entry.coordinate_frame_byte_length
        || metadata.coordinate_frame_sha256 != entry.coordinate_frame_sha256
        || metadata.local_shell_status != "COMPLETE_EXACT_HASH_BOUND_FULL_POPULATION_SCAN"
        || metadata.local_shell_status != entry.local_shell_status
        || metadata.local_shell_atom_count != entry.local_shell_atom_count
        || metadata.local_shell_payload_sha256 != entry.local_shell_payload_sha256
        || metadata.excluded_population_scanned != entry.excluded_population_scanned
    {
        bail!("Ghost coordinate persisted join metadata/index mismatch");
    }
    Ok(())
}

fn decode_record_request(record: &[u8], local_index: u64) -> Result<DecodedRecordRequest> {
    if record.len() != GHOST_SEGMENT_RECORD_BYTES
        || read_u32(record, GHOST_V2_OFFSET_SCHEMA_VERSION)? != GHOST_FRAME_SCHEMA_V3
    {
        bail!("Ghost coordinate producer requires exact v3 record");
    }
    let membership =
        decode_ghost_v3_cluster_residue_membership(record).map_err(anyhow::Error::msg)?;
    let membership_sha256 = canonical_sha256(&membership)?;
    let frame_idx = read_u64(record, 0)?;
    let site_id = read_u32(record, 8)?;
    let step_idx = read_u64(record, GHOST_V2_OFFSET_STEP_IDX)?;
    let centroid = [
        read_f32(record, GHOST_V2_OFFSET_CENTROID)?,
        read_f32(record, GHOST_V2_OFFSET_CENTROID + 4)?,
        read_f32(record, GHOST_V2_OFFSET_CENTROID + 8)?,
    ];
    if centroid.iter().any(|value| !value.is_finite()) {
        bail!("Ghost coordinate record centroid is nonfinite");
    }
    let record_sha256 = sha256_bytes(record);
    Ok(DecodedRecordRequest {
        local_index,
        key: GhostCoordinateKey {
            stream_id: 0, // replaced by context at append call below
            step_idx,
            ghost_site_id: site_id,
            ghost_record_sha256: record_sha256,
        },
        step_idx,
        frame_idx,
        centroid,
        membership_sha256,
    })
}

fn validate_topology_against_context(
    topology: &PrismPrepTopology,
    context: &GhostCoordinateContext,
) -> Result<()> {
    context.verify()?;
    let resolved = resolve_ghost_topology(topology, &context.topology_sha256)?;
    let topology = &resolved.topology;
    let analysis = topology.analysis_residue_contract.as_ref().unwrap();
    let molecules = topology.molecule_identity_table.as_ref().unwrap();
    let boundary = topology.periodic_cell.as_ref().unwrap();
    let analysis_ids = topology.atom_analysis_residue_ids.as_ref().unwrap();
    if topology.n_atoms as u64 != context.atom_count
        || topology.n_analysis_residues.unwrap_or_default() as u32 != context.analysis_residue_count
        || analysis_ids.iter().filter(|value| **value >= 0).count() as u64
            != context.analysis_atom_count
        || analysis.atom_analysis_residue_ids_sha256 != context.atom_analysis_residue_ids_sha256
        || analysis.contract_sha256 != context.analysis_residue_contract_sha256
        || molecules.table_sha256 != context.molecule_identity_table_sha256
        || boundary.contract_sha256.as_deref()
            != Some(context.spatial_boundary_contract_sha256.as_str())
        || validate_spatial_boundary(topology)? != context.spatial_distance_mode
        || resolved.projection_mode != context.topology_projection_mode
        || resolved.authority_contract_sha256 != context.topology_projection_authority_sha256
        || resolved.authority_artifact_sha256 != context.topology_projection_artifact_sha256
        || resolved.authority_source_sha256 != context.topology_projection_source_sha256
        || resolved.atom_order_sha256 != context.atom_order_sha256
    {
        bail!("prepared topology differs from Ghost coordinate context");
    }
    Ok(())
}

fn validate_full_frame(
    topology: &PrismPrepTopology,
    frame: &ExactGhostCoordinateFrame,
) -> Result<()> {
    validate_full_frame_population(topology.n_atoms, frame)
}

fn validate_full_frame_population(n_atoms: usize, frame: &ExactGhostCoordinateFrame) -> Result<()> {
    if frame.positions_angstrom.len() != n_atoms * 3
        || frame
            .positions_angstrom
            .iter()
            .any(|value| !value.is_finite())
    {
        bail!("exact Ghost coordinate frame is incomplete or nonfinite");
    }
    Ok(())
}

fn compact_analysis_positions(topology: &PrismPrepTopology, positions: &[f32]) -> Result<Vec<f32>> {
    let analysis = topology
        .atom_analysis_residue_ids
        .as_ref()
        .context("analysis compaction requires signed atom map")?;
    compact_analysis_positions_from_map(topology.n_atoms, analysis, positions)
}

fn compact_analysis_positions_from_map(
    n_atoms: usize,
    analysis: &[i32],
    positions: &[f32],
) -> Result<Vec<f32>> {
    if positions.len() != n_atoms * 3 || analysis.len() != n_atoms {
        bail!("analysis compaction source is not a complete full-system frame");
    }
    let mut compact = Vec::with_capacity(analysis.iter().filter(|value| **value >= 0).count() * 3);
    for (atom, analysis_id) in analysis.iter().enumerate() {
        if *analysis_id >= 0 {
            compact.extend_from_slice(&positions[3 * atom..3 * atom + 3]);
        }
    }
    if compact.iter().any(|value| !value.is_finite()) {
        bail!("analysis coordinate compaction produced nonfinite value");
    }
    Ok(compact)
}

fn molecule_role_map(topology: &PrismPrepTopology) -> Result<Vec<AtomRole>> {
    let analysis = topology
        .atom_analysis_residue_ids
        .as_ref()
        .context("missing analysis map")?;
    let table = topology
        .molecule_identity_table
        .as_ref()
        .context("missing molecule table")?;
    let mut roles = vec![None::<AtomRole>; topology.n_atoms];
    for molecule in &table.molecules {
        let role = match molecule.kind.to_ascii_uppercase().as_str() {
            "WATER" => "WATER",
            "ION" => "ION",
            "LAMBDA_FRAGMENT" | "FRAGMENT" => "LAMBDA_FRAGMENT",
            "BIOMOLECULE" | "PROTEIN" | "RECEPTOR" | "POLYMER" => "RECEPTOR",
            _ => "OTHER",
        };
        for &atom in &molecule.atom_indices {
            if atom >= topology.n_atoms || roles[atom].is_some() {
                bail!("molecule identity table atom partition is incomplete or duplicate");
            }
            roles[atom] = Some(AtomRole {
                analysis_id: analysis[atom],
                molecule_id: molecule.molecule_id,
                role: role.into(),
            });
        }
    }
    roles
        .into_iter()
        .enumerate()
        .map(|(atom, role)| role.with_context(|| format!("molecule role missing atom {atom}")))
        .collect()
}

fn exact_local_shell(
    topology: &PrismPrepTopology,
    roles: &[AtomRole],
    positions: &[f32],
    centroid: [f32; 3],
    cutoff: f64,
    spatial_distance: SpatialDistance,
) -> Result<Vec<ExactLocalShellAtom>> {
    exact_local_shell_from_population(
        topology.n_atoms,
        roles,
        positions,
        centroid,
        cutoff,
        spatial_distance,
    )
}

fn exact_local_shell_from_population(
    n_atoms: usize,
    roles: &[AtomRole],
    positions: &[f32],
    centroid: [f32; 3],
    cutoff: f64,
    spatial_distance: SpatialDistance,
) -> Result<Vec<ExactLocalShellAtom>> {
    if positions.len() != n_atoms * 3 || roles.len() != n_atoms {
        bail!("local-shell source population is incomplete");
    }
    let center = [centroid[0] as f64, centroid[1] as f64, centroid[2] as f64];
    let cutoff2 = cutoff * cutoff;
    let mut shell = Vec::new();
    for atom in 0..n_atoms {
        if roles[atom].analysis_id >= 0 {
            continue;
        }
        let point = [
            positions[3 * atom] as f64,
            positions[3 * atom + 1] as f64,
            positions[3 * atom + 2] as f64,
        ];
        let delta = spatial_displacement(sub(point, center), spatial_distance);
        if dot(delta, delta) <= cutoff2 {
            shell.push(ExactLocalShellAtom {
                atom_index: atom.try_into().context("shell atom index exceeds u32")?,
                molecule_id: roles[atom].molecule_id,
                role: roles[atom].role.clone(),
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

fn verify_persisted_local_shell(
    n_atoms: usize,
    roles: &[AtomRole],
    shell: &[ExactLocalShellAtom],
    join: &GhostCoordinateJoinEntry,
    spatial_distance: SpatialDistance,
) -> Result<()> {
    let expected_excluded = roles.iter().filter(|role| role.analysis_id < 0).count() as u64;
    if join.local_shell_status != "COMPLETE_EXACT_HASH_BOUND_FULL_POPULATION_SCAN"
        || join.excluded_population_scanned != expected_excluded
    {
        bail!("EVIDENCE_NOT_FOUND: local shell lacks complete excluded-population scan");
    }
    let center = [
        join.ghost_centroid_angstrom[0] as f64,
        join.ghost_centroid_angstrom[1] as f64,
        join.ghost_centroid_angstrom[2] as f64,
    ];
    let cutoff2 = join.local_shell_cutoff_angstrom.powi(2);
    let mut observed = BTreeSet::new();
    for atom in shell {
        let index = atom.atom_index as usize;
        if index >= n_atoms
            || roles[index].analysis_id >= 0
            || roles[index].molecule_id != atom.molecule_id
            || roles[index].role != atom.role
            || !observed.insert(index)
        {
            bail!("local-shell atom identity is duplicate or differs from topology");
        }
        let point = [
            atom.position_angstrom[0] as f64,
            atom.position_angstrom[1] as f64,
            atom.position_angstrom[2] as f64,
        ];
        let delta = spatial_displacement(sub(point, center), spatial_distance);
        if dot(delta, delta) > cutoff2 + 1.0e-8 {
            bail!("local-shell atom lies outside declared PBC cutoff");
        }
    }
    Ok(())
}

fn encode_shell(shell: &[ExactLocalShellAtom]) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(shell.len() * SHELL_ENTRY_BYTES);
    for atom in shell {
        let role = match atom.role.as_str() {
            "WATER" => 1u8,
            "ION" => 2,
            "LAMBDA_FRAGMENT" => 3,
            "OTHER" => 4,
            other => bail!("unsupported local-shell role {other}"),
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
    if bytes.len() % SHELL_ENTRY_BYTES != 0 {
        bail!("local-shell payload is truncated");
    }
    let mut shell = Vec::new();
    for row in bytes.chunks_exact(SHELL_ENTRY_BYTES) {
        if row[13..16].iter().any(|value| *value != 0)
            || row[28..32].iter().any(|value| *value != 0)
        {
            bail!("local-shell reserved bytes are nonzero");
        }
        let role = match row[12] {
            1 => "WATER",
            2 => "ION",
            3 => "LAMBDA_FRAGMENT",
            4 => "OTHER",
            other => bail!("invalid local-shell role code {other}"),
        };
        let position = [
            f32::from_le_bytes(row[16..20].try_into()?),
            f32::from_le_bytes(row[20..24].try_into()?),
            f32::from_le_bytes(row[24..28].try_into()?),
        ];
        if position.iter().any(|value| !value.is_finite()) {
            bail!("local-shell coordinate is nonfinite");
        }
        shell.push(ExactLocalShellAtom {
            atom_index: u32::from_le_bytes(row[0..4].try_into()?),
            molecule_id: u64::from_le_bytes(row[4..12].try_into()?),
            role: role.into(),
            position_angstrom: position,
        });
    }
    Ok(shell)
}

fn spatial_distance(topology: &PrismPrepTopology) -> Result<SpatialDistance> {
    let boundary = topology
        .periodic_cell
        .as_ref()
        .context("exact Ghost coordinates require a spatial boundary")?;
    let mode = validate_spatial_boundary(topology)?;
    if mode == "NONPERIODIC_DIRECT_DISTANCE" {
        return Ok(SpatialDistance::NonperiodicDirect);
    }
    let rows = boundary
        .vectors_angstrom
        .context("periodic Ghost coordinates require cell vectors")?;
    let matrix = [
        [rows[0][0], rows[1][0], rows[2][0]],
        [rows[0][1], rows[1][1], rows[2][1]],
        [rows[0][2], rows[1][2], rows[2][2]],
    ];
    let inverse = invert_3x3(matrix).context("periodic cell is singular")?;
    Ok(SpatialDistance::PeriodicMinimumImage { matrix, inverse })
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

fn minimum_image(delta: [f64; 3], matrix: [[f64; 3]; 3], inverse: [[f64; 3]; 3]) -> [f64; 3] {
    let mut frac = mat_vec(inverse, delta);
    for value in &mut frac {
        *value -= value.round();
    }
    mat_vec(matrix, frac)
}

fn spatial_displacement(delta: [f64; 3], spatial_distance: SpatialDistance) -> [f64; 3] {
    match spatial_distance {
        SpatialDistance::PeriodicMinimumImage { matrix, inverse } => {
            minimum_image(delta, matrix, inverse)
        }
        SpatialDistance::NonperiodicDirect => delta,
    }
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

fn encode_segment_header(
    header: &mut [u8],
    segment_ordinal: u64,
    frame_count: u64,
    join_count: u64,
    payload_bytes: u64,
    context_sha: &str,
    previous_sha: &str,
    ghost_segment_sha: &str,
) -> Result<()> {
    if header.len() != GHOST_COORDINATE_HEADER_BYTES {
        bail!("invalid coordinate segment header width");
    }
    header.fill(0);
    header[0..8].copy_from_slice(&GHOST_COORDINATE_SEGMENT_MAGIC);
    put_u32(header, 8, GHOST_COORDINATE_SCHEMA_VERSION);
    put_u32(header, 12, GHOST_COORDINATE_HEADER_BYTES as u32);
    put_u64(header, 16, segment_ordinal);
    put_u64(header, 24, frame_count);
    put_u64(header, 32, join_count);
    put_u64(header, 40, payload_bytes);
    put_sha256(header, 48, context_sha)?;
    put_sha256(header, 80, previous_sha)?;
    put_sha256(header, 112, ghost_segment_sha)
}

fn verify_segment_header(header: &[u8], entry: &GhostCoordinateSegmentEntry) -> Result<()> {
    if header.len() != GHOST_COORDINATE_HEADER_BYTES
        || header[..8] != GHOST_COORDINATE_SEGMENT_MAGIC
        || read_u32(header, 8)? != GHOST_COORDINATE_SCHEMA_VERSION
        || read_u32(header, 12)? as usize != GHOST_COORDINATE_HEADER_BYTES
        || read_u64(header, 16)? != entry.segment_ordinal
        || read_u64(header, 24)? != entry.frame_count
        || read_u64(header, 32)? != entry.join_count
        || read_u64(header, 40)? != entry.payload_bytes
        || read_sha256(header, 48)? != entry.context_digest_sha256
        || read_sha256(header, 80)? != entry.previous_file_sha256
        || read_sha256(header, 112)? != entry.ghost_segment_file_sha256
        || header[144..].iter().any(|value| *value != 0)
    {
        bail!("Ghost coordinate segment header mismatch");
    }
    Ok(())
}

pub fn coordinate_segment_root(output_dir: &Path, topo_stem: &str, stream_id: u16) -> PathBuf {
    output_dir.join(format!(
        "{topo_stem}_stream{stream_id:02}_ghost_coordinate_segments"
    ))
}

// Each renamed/fsynced index record is an immutable journal append. The
// full manifest remains a checkpoint until complete(); this prevents O(N^2)
// serialization of all prior frame/join metadata during N runtime chunks.
const INDEX_JOURNAL_DIRECTORY: &str = "index_journal";

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CoordinateIndexJournalEntry {
    schema_kind: String,
    schema_version: u32,
    context_digest_sha256: String,
    entry_sha256: String,
    entry: GhostCoordinateSegmentEntry,
}

fn ensure_index_journal_directory(root: &Path) -> Result<()> {
    let path = root.join(INDEX_JOURNAL_DIRECTORY);
    match std::fs::symlink_metadata(&path) {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => Ok(()),
        Ok(_) => bail!("coordinate index journal is not a real directory"),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            std::fs::create_dir(&path)?;
            fsync_directory(root)
        }
        Err(error) => Err(error.into()),
    }
}

fn write_index_journal_entry(root: &Path, entry: &GhostCoordinateSegmentEntry) -> Result<u64> {
    let journal = root.join(INDEX_JOURNAL_DIRECTORY);
    let path = journal.join(format!("segment_{:08}.json", entry.segment_ordinal));
    let partial = journal.join(format!(
        ".segment_{:08}.json.partial",
        entry.segment_ordinal
    ));
    if path.exists() || partial.exists() {
        bail!("coordinate journal entry already exists or is incomplete; recovery required");
    }
    let record = CoordinateIndexJournalEntry {
        schema_kind: "prism_ghost_coordinate_index_entry".into(),
        schema_version: 1,
        context_digest_sha256: entry.context_digest_sha256.clone(),
        entry_sha256: sha256_bytes(&serde_json::to_vec(entry)?),
        entry: entry.clone(),
    };
    let bytes = serde_json::to_vec(&record)?;
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(&partial)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(&partial, &path)?;
    fsync_directory(&journal)?;
    Ok(bytes.len() as u64)
}

/// Read a committed metadata snapshot, including entries newer than the
/// checkpoint. `.partial` entries are uncommitted and never consumed. A reader
/// racing an append receives a committed prefix, never a completed claim.
/// Full coordinate/hash verification is performed by the public live loader.
fn load_coordinate_manifest_with_journal(root: &Path) -> Result<GhostCoordinateManifest> {
    let mut manifest: GhostCoordinateManifest =
        serde_json::from_slice(&read_regular_nofollow(&root.join("manifest.json"))?)?;
    manifest.context.verify()?;
    if manifest.context_digest_sha256 != manifest.context.digest_sha256()? {
        bail!("coordinate checkpoint context digest mismatch");
    }
    let journal = root.join(INDEX_JOURNAL_DIRECTORY);
    if !journal.exists() {
        return Ok(manifest);
    }
    let metadata = std::fs::symlink_metadata(&journal)?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        bail!("coordinate index journal is not a real directory");
    }
    let mut paths = BTreeMap::new();
    for file in std::fs::read_dir(&journal)? {
        let file = file?;
        let name = file.file_name().to_string_lossy().into_owned();
        if name.ends_with(".partial") {
            continue;
        }
        let ordinal = name
            .strip_prefix("segment_")
            .and_then(|s| s.strip_suffix(".json"))
            .and_then(|s| s.parse::<u64>().ok())
            .context("unrecognized coordinate journal file")?;
        if name != format!("segment_{ordinal:08}.json")
            || paths.insert(ordinal, file.path()).is_some()
        {
            bail!("noncanonical or duplicate coordinate journal ordinal");
        }
    }
    // A checkpoint may contain a legacy prefix without journal rows. Every
    // journal row that does exist is checked, including already-covered rows.
    for (ordinal, path) in paths {
        let record: CoordinateIndexJournalEntry =
            serde_json::from_slice(&read_regular_nofollow(&path)?)?;
        let entry = record.entry;
        if record.schema_kind != "prism_ghost_coordinate_index_entry"
            || record.schema_version != 1
            || record.context_digest_sha256 != manifest.context_digest_sha256
            || entry.context_digest_sha256 != manifest.context_digest_sha256
            || entry.segment_ordinal != ordinal
            || record.entry_sha256 != sha256_bytes(&serde_json::to_vec(&entry)?)
        {
            bail!("coordinate index journal integrity/context/ordinal mismatch");
        }
        if (ordinal as usize) < manifest.segments.len() {
            if manifest.segments[ordinal as usize] != entry {
                bail!("coordinate journal conflicts with checkpoint entry");
            }
            continue;
        }
        if manifest.completed
            || ordinal as usize != manifest.segments.len()
            || entry.first_join_ordinal != manifest.persisted_ghost_joins
            || entry.previous_file_sha256
                != manifest
                    .segments
                    .last()
                    .map(|s| s.file_sha256.as_str())
                    .unwrap_or(ZERO_SHA256)
            || entry.frame_count != entry.frames.len() as u64
            || entry.join_count != entry.joins.len() as u64
            || !entry.fsync_pass
        {
            bail!("coordinate journal segment ordinal/chain/count failure");
        }
        manifest.requested_ghost_records += entry.join_count;
        manifest.persisted_ghost_joins += entry.join_count;
        manifest.persisted_coordinate_frames += entry.frame_count;
        manifest.segments.push(entry);
        manifest.requested_equals_persisted =
            manifest.requested_ghost_records == manifest.persisted_ghost_joins;
    }
    Ok(manifest)
}

/// Verify and return a live committed-prefix snapshot. The segment count and
/// last ordinal bound this snapshot; a later append is outside its scope.
/// `manifest.json` alone is a checkpoint while `completed == false`.
/// Crash recovery remains a writer operation; this reader never changes files.
pub fn load_verified_live_ghost_coordinate_manifest(
    root: &Path,
    topology: &PrismPrepTopology,
) -> Result<GhostCoordinateManifest> {
    let manifest = load_coordinate_manifest_with_journal(root)?;
    validate_topology_against_context(topology, &manifest.context)?;
    verify_manifest_segments(root, &manifest, topology)?;
    Ok(manifest)
}

fn write_manifest_atomic(
    root: &Path,
    path: &Path,
    manifest: &GhostCoordinateManifest,
) -> Result<u64> {
    let partial = root.join(".manifest.json.partial");
    if partial.exists() {
        bail!("coordinate manifest partial exists; recovery required");
    }
    let bytes = serde_json::to_vec(manifest)?;
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(&partial)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(&partial, path)?;
    fsync_directory(root)?;
    Ok(bytes.len() as u64)
}

fn reject_partial_files(root: &Path) -> Result<()> {
    for entry in std::fs::read_dir(root)? {
        let name = entry?.file_name().to_string_lossy().into_owned();
        if name.ends_with(".partial") {
            bail!("EVIDENCE_NOT_FOUND: incomplete coordinate artifact {name}");
        }
    }
    Ok(())
}

/// Recover the fsync+rename/manifest-update crash window. Every orphan final
/// segment is self-describing: frame headers and join metadata carry enough
/// information to reconstruct the exact manifest rows and re-verify all
/// hashes before the atomic manifest advances.
fn recover_orphan_segments(
    root: &Path,
    manifest: &mut GhostCoordinateManifest,
    work: &mut GhostCoordinatePersistenceWork,
) -> Result<()> {
    loop {
        let ordinal = manifest.segments.len() as u64;
        let file_name = format!("segment_{ordinal:08}.ghostcoord");
        let path = root.join(&file_name);
        if !path.exists() {
            break;
        }
        let bytes = read_regular_nofollow(&path)?;
        if bytes.len() < GHOST_COORDINATE_HEADER_BYTES
            || bytes[..8] != GHOST_COORDINATE_SEGMENT_MAGIC
            || read_u32(&bytes, 8)? != GHOST_COORDINATE_SCHEMA_VERSION
            || read_u64(&bytes, 16)? != ordinal
            || read_sha256(&bytes, 48)? != manifest.context_digest_sha256
        {
            bail!("orphan Ghost coordinate segment has invalid header/context");
        }
        let frame_count = read_u64(&bytes, 24)?;
        let join_count = read_u64(&bytes, 32)?;
        let payload_bytes = read_u64(&bytes, 40)?;
        if payload_bytes != (bytes.len() - GHOST_COORDINATE_HEADER_BYTES) as u64 {
            bail!("orphan Ghost coordinate segment payload length mismatch");
        }
        let previous_file_sha256 = read_sha256(&bytes, 80)?;
        let expected_previous = manifest
            .segments
            .last()
            .map(|entry| entry.file_sha256.as_str())
            .unwrap_or(ZERO_SHA256);
        if previous_file_sha256 != expected_previous {
            bail!("orphan Ghost coordinate segment breaks SHA chain");
        }
        let ghost_segment_file_sha256 = read_sha256(&bytes, 112)?;
        let mut cursor = GHOST_COORDINATE_HEADER_BYTES;
        let mut frames = Vec::new();
        for frame_ordinal in 0..frame_count {
            let start = cursor;
            if bytes.get(start..start + 8) != Some(FRAME_MAGIC.as_slice())
                || read_u32(&bytes, start + 8)? != GHOST_COORDINATE_SCHEMA_VERSION
                || read_u32(&bytes, start + 12)? as usize != FRAME_HEADER_BYTES
            {
                bail!("orphan Ghost coordinate frame header is invalid");
            }
            let step_idx = read_u64(&bytes, start + 16)?;
            let atom_count = read_u64(&bytes, start + 24)?;
            let coordinate_bytes = read_u64(&bytes, start + 32)? as usize;
            let end = start
                .checked_add(FRAME_HEADER_BYTES)
                .and_then(|value| value.checked_add(coordinate_bytes))
                .context("orphan coordinate frame byte overflow")?;
            let record = bytes
                .get(start..end)
                .context("orphan coordinate frame truncated")?;
            let payload = &record[FRAME_HEADER_BYTES..];
            let coordinate_payload_sha256 = read_sha256(record, 40)?;
            if sha256_bytes(payload) != coordinate_payload_sha256 {
                bail!("orphan coordinate frame payload hash mismatch");
            }
            frames.push(GhostCoordinateFrameEntry {
                step_idx,
                frame_ordinal,
                atom_count,
                byte_offset: start as u64,
                byte_length: record.len() as u64,
                coordinate_payload_sha256,
                frame_record_sha256: sha256_bytes(record),
                source_full_frame_sha256: read_sha256(record, 72)?,
            });
            cursor = end;
        }
        let mut joins = Vec::new();
        for _ in 0..join_count {
            let start = cursor;
            if bytes.get(start..start + 8) != Some(JOIN_MAGIC.as_slice()) {
                bail!("orphan Ghost coordinate join magic is invalid");
            }
            let metadata_len = read_u32(&bytes, start + 8)? as usize;
            let shell_len = read_u64(&bytes, start + 12)? as usize;
            let metadata_start = start + 20;
            let metadata_end = metadata_start
                .checked_add(metadata_len)
                .context("orphan join metadata overflow")?;
            let end = metadata_end
                .checked_add(shell_len)
                .context("orphan join shell overflow")?;
            let record = bytes
                .get(start..end)
                .context("orphan coordinate join truncated")?;
            let metadata: PersistedJoinMetadata = serde_json::from_slice(
                bytes
                    .get(metadata_start..metadata_end)
                    .context("orphan coordinate metadata truncated")?,
            )?;
            let shell = bytes
                .get(metadata_end..end)
                .context("orphan coordinate shell truncated")?;
            if sha256_bytes(shell) != metadata.local_shell_payload_sha256
                || metadata.ghost_segment_file_sha256 != ghost_segment_file_sha256
            {
                bail!("orphan coordinate join shell/Ghost hash mismatch");
            }
            joins.push(GhostCoordinateJoinEntry {
                key: metadata.key,
                stream_record_ordinal: metadata.stream_record_ordinal,
                frame_idx: metadata.frame_idx,
                ghost_segment_ordinal: metadata.ghost_segment_ordinal,
                ghost_segment_file: metadata.ghost_segment_file,
                ghost_segment_file_sha256: metadata.ghost_segment_file_sha256,
                ghost_payload_byte_offset: metadata.ghost_payload_byte_offset,
                ghost_byte_length: metadata.ghost_byte_length,
                ghost_centroid_angstrom: metadata.ghost_centroid_angstrom,
                cluster_residue_membership_sha256: metadata.cluster_residue_membership_sha256,
                thermodynamic_condition_id: metadata.thermodynamic_condition_id,
                protocol_phase: metadata.protocol_phase,
                protocol_state_hash: metadata.protocol_state_hash,
                phase_schedule_digest_sha256: metadata.phase_schedule_digest_sha256,
                coordinate_frame_ordinal: metadata.coordinate_frame_ordinal,
                coordinate_frame_byte_offset: metadata.coordinate_frame_byte_offset,
                coordinate_frame_byte_length: metadata.coordinate_frame_byte_length,
                coordinate_frame_sha256: metadata.coordinate_frame_sha256,
                local_shell_status: metadata.local_shell_status,
                local_shell_cutoff_angstrom: metadata.local_shell_cutoff_angstrom,
                local_shell_atom_count: metadata.local_shell_atom_count,
                local_shell_byte_offset: metadata_end as u64,
                local_shell_byte_length: shell_len as u64,
                local_shell_payload_sha256: metadata.local_shell_payload_sha256,
                excluded_population_scanned: metadata.excluded_population_scanned,
                join_byte_offset: start as u64,
                join_byte_length: record.len() as u64,
                join_record_sha256: sha256_bytes(record),
            });
            cursor = end;
        }
        if cursor != bytes.len() || joins.is_empty() {
            bail!("orphan Ghost coordinate segment has trailing bytes or no joins");
        }
        let ghost_segment_ordinal = joins[0].ghost_segment_ordinal;
        if joins
            .iter()
            .any(|join| join.ghost_segment_ordinal != ghost_segment_ordinal)
        {
            bail!("orphan coordinate segment spans multiple Ghost segments");
        }
        let file_sha256 = sha256_bytes(&bytes);
        let entry = GhostCoordinateSegmentEntry {
            segment_ordinal: ordinal,
            first_join_ordinal: manifest.persisted_ghost_joins,
            frame_count,
            join_count,
            file: file_name,
            file_sha256,
            previous_file_sha256,
            context_digest_sha256: manifest.context_digest_sha256.clone(),
            ghost_segment_ordinal,
            ghost_segment_file_sha256,
            payload_bytes,
            fsync_pass: true,
            frames,
            joins,
        };
        manifest.requested_ghost_records += join_count;
        manifest.persisted_coordinate_frames += frame_count;
        manifest.persisted_ghost_joins += join_count;
        let bytes = write_index_journal_entry(root, &entry)?;
        work.journal_entries_written += 1;
        work.journal_bytes_written += bytes;
        manifest.segments.push(entry);
        manifest.requested_equals_persisted =
            manifest.requested_ghost_records == manifest.persisted_ghost_joins;
    }
    let durable_segment_count = std::fs::read_dir(root)?
        .filter_map(std::result::Result::ok)
        .filter(|entry| entry.file_name().to_string_lossy().ends_with(".ghostcoord"))
        .count();
    if durable_segment_count != manifest.segments.len() {
        bail!("Ghost coordinate segment ordinal gap or duplicate file population");
    }
    Ok(())
}

fn ensure_disk_for_write(path: &Path, bytes: u64) -> Result<()> {
    let available = available_bytes(path)?;
    let required = bytes
        .checked_mul(5)
        .context("per-segment disk margin overflow")?
        / 4;
    if available < required {
        bail!(
            "insufficient disk for lossless coordinate segment: required={} available={}",
            required,
            available
        );
    }
    Ok(())
}

fn available_bytes(path: &Path) -> Result<u64> {
    let c_path = std::ffi::CString::new(path.as_os_str().as_encoded_bytes())?;
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c_path.as_ptr(), &mut stat) } != 0 {
        return Err(std::io::Error::last_os_error()).context("statvfs coordinate output");
    }
    Ok((stat.f_bavail as u64).saturating_mul(stat.f_frsize as u64))
}

fn fsync_directory(path: &Path) -> Result<()> {
    let dir = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_DIRECTORY | libc::O_NOFOLLOW)
        .open(path)?;
    dir.sync_all()?;
    Ok(())
}

fn read_regular_nofollow(path: &Path) -> Result<Vec<u8>> {
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)?;
    let before = file.metadata()?;
    if !before.is_file() {
        bail!("coordinate artifact must be a regular non-symlink file");
    }
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    let after = file.metadata()?;
    let before_identity = (
        before.dev(),
        before.ino(),
        before.len(),
        before.mtime(),
        before.mtime_nsec(),
        before.ctime(),
        before.ctime_nsec(),
    );
    let after_identity = (
        after.dev(),
        after.ino(),
        after.len(),
        after.mtime(),
        after.mtime_nsec(),
        after.ctime(),
        after.ctime_nsec(),
    );
    if before_identity != after_identity || bytes.len() as u64 != before.len() {
        bail!("coordinate artifact changed during nofollow read");
    }
    Ok(bytes)
}

fn validate_stem(stem: &str) -> Result<()> {
    if stem.is_empty()
        || Path::new(stem).components().count() != 1
        || !Path::new(stem)
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
    {
        bail!("invalid topology stem");
    }
    Ok(())
}

fn validate_basename(name: &str) -> Result<()> {
    validate_stem(name)
}

fn checked_range(offset: u64, length: u64, total: usize) -> Result<std::ops::Range<usize>> {
    let start: usize = offset.try_into().context("byte offset exceeds usize")?;
    let end: usize = offset
        .checked_add(length)
        .context("byte range overflow")?
        .try_into()
        .context("byte end exceeds usize")?;
    if start < GHOST_COORDINATE_HEADER_BYTES || end > total || start > end {
        bail!("coordinate byte range outside segment");
    }
    Ok(start..end)
}

fn canonical_sha256<T: Serialize + ?Sized>(value: &T) -> Result<String> {
    Ok(sha256_bytes(&serde_json::to_vec(value)?))
}

fn coordinate_producer_source_sha256() -> String {
    let mut h = Sha256::new();
    h.update(b"prism-ghost-coordinate-producer-v1\0");
    h.update(include_bytes!("ghost_coordinate_segment.rs"));
    format!("{:x}", h.finalize())
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn decode_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        bail!("f32 coordinate payload is truncated");
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
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
            .context("truncated u32")?
            .try_into()?,
    ))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    Ok(u64::from_le_bytes(
        bytes
            .get(offset..offset + 8)
            .context("truncated u64")?
            .try_into()?,
    ))
}

fn read_f32(bytes: &[u8], offset: usize) -> Result<f32> {
    Ok(f32::from_le_bytes(
        bytes
            .get(offset..offset + 4)
            .context("truncated f32")?
            .try_into()?,
    ))
}

fn put_sha256(bytes: &mut [u8], offset: usize, value: &str) -> Result<()> {
    require_sha256(value, "binary SHA256 field")?;
    let raw = hex_decode(value)?;
    bytes[offset..offset + 32].copy_from_slice(&raw);
    Ok(())
}

fn read_sha256(bytes: &[u8], offset: usize) -> Result<String> {
    Ok(bytes
        .get(offset..offset + 32)
        .context("truncated SHA256")?
        .iter()
        .map(|value| format!("{value:02x}"))
        .collect())
}

fn require_sha256(value: &str, name: &str) -> Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{name} is not SHA256");
    }
    Ok(())
}

fn hex_decode(value: &str) -> Result<Vec<u8>> {
    (0..value.len())
        .step_by(2)
        .map(|offset| u8::from_str_radix(&value[offset..offset + 2], 16).map_err(Into::into))
        .collect()
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_file(path: &Path) -> Result<String> {
    Ok(sha256_bytes(&read_regular_nofollow(path)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ghost_segment::GhostSegmentContext;
    use crate::ghost_tile::{
        ghost_record_abi_sha256, ghost_record_source_sha256, GHOST_V3_CLUSTER_RESIDUE_CAPACITY,
        GHOST_V3_CLUSTER_RESIDUE_ENTRY_BYTES, GHOST_V3_GEOMETRY_CLUSTER_RESIDUE_MEMBERSHIP_EXACT,
        GHOST_V3_GEOMETRY_COORDINATE_SIDECAR_REQUIRED, GHOST_V3_MEMBERSHIP_SCHEMA_VERSION,
        GHOST_V3_OFFSET_CLUSTER_RESIDUE_CAPACITY, GHOST_V3_OFFSET_CLUSTER_RESIDUE_COUNT,
        GHOST_V3_OFFSET_CLUSTER_RESIDUE_ENTRIES, GHOST_V3_OFFSET_CLUSTER_RESIDUE_ENTRY_BYTES,
        GHOST_V3_OFFSET_CLUSTER_RESIDUE_OVERFLOW, GHOST_V3_OFFSET_GEOMETRY_AVAILABILITY_FLAGS,
        GHOST_V3_OFFSET_MEMBERSHIP_SCHEMA_VERSION, GHOST_V3_OFFSET_SOURCE_SPIKE_COUNT,
        GHOST_V3_OFFSET_UNRESOLVED_RESIDUE_SPIKE_COUNT,
    };

    fn test_topology() -> PrismPrepTopology {
        let molecule_table_sha256 = "77".repeat(32);
        let atom_analysis_ids = serde_json::json!([0, -1]);
        let atom_analysis_residue_ids_sha256 =
            crate::input::python_canonical_sha256(&atom_analysis_ids).unwrap();
        let mut analysis_contract = serde_json::json!({
            "schema_kind": "prismos_analysis_residue_mapping",
            "schema_version": 1,
            "analysis_authority": "KCC_GHOST_PROTEIN_RESIDUE_AXIS",
            "excluded_atom_sentinel": -1,
            "n_atoms": 2,
            "total_physical_residue_count": 2,
            "n_analysis_residues": 1,
            "physical_residue_roles": ["RECEPTOR", "WATER"],
            "physical_to_analysis_residue_ids": [0, -1],
            "analysis_residues": [{
                "analysis_residue_id": 0,
                "physical_residue_id": 0,
                "chain_id": "A",
                "pdb_residue_id": 1,
                "residue_name": "ALA",
                "role": "RECEPTOR"
            }],
            "atom_analysis_residue_ids_sha256": atom_analysis_residue_ids_sha256,
            "physical_molecule_table_sha256": molecule_table_sha256,
            "lambda_fragment_system_sha256": null,
            "role_counts": {"RECEPTOR": 1, "WATER": 1},
            "excluded_roles": ["WATER", "ION", "LAMBDA_FRAGMENT", "OTHER"],
            "must_recompute_after_fragment_materialization": true
        });
        let contract_sha256 = crate::input::python_canonical_sha256(&analysis_contract).unwrap();
        analysis_contract
            .as_object_mut()
            .unwrap()
            .insert("contract_sha256".into(), contract_sha256.into());
        serde_json::from_value(serde_json::json!({
            "schema_kind": "prismos_topology",
            "schema_version": 3,
            "source_pdb": "synthetic-production-schema.pdb",
            "n_atoms": 2,
            "n_residues": 2,
            "total_physical_residue_count": 2,
            "n_analysis_residues": 1,
            "n_chains": 1,
            "positions": [1.0, 1.0, 1.0, 1.5, 1.0, 1.0],
            "elements": ["C", "O"],
            "atom_names": ["CA", "O"],
            "residue_names": ["ALA", "HOH"],
            "residue_ids": [0, 1],
            "atom_physical_residue_ids": [0, 1],
            "atom_analysis_residue_ids": atom_analysis_ids,
            "analysis_residue_contract": analysis_contract,
            "chain_ids": ["A", "W"],
            "charges": [0.0, 0.0],
            "masses": [12.0, 16.0],
            "ca_indices": [0],
            "bonds": [],
            "angles": [],
            "dihedrals": [],
            "lj_params": [
                {"sigma": 3.4, "epsilon": 0.1},
                {"sigma": 3.1, "epsilon": 0.15}
            ],
            "exclusions": [[], []],
            "molecule_identity_table": {
                "schema_kind": "prismos_molecule_identity_table",
                "schema_version": 1,
                "n_molecules": 2,
                "n_atoms": 2,
                "molecules": [
                    {
                        "molecule_id": 0,
                        "atom_indices": [0],
                        "residue_indices": [0],
                        "residue_names": ["ALA"],
                        "chain_ids": ["A"],
                        "kind": "BIOMOLECULE",
                        "identity": "receptor",
                        "component_sha256": "88".repeat(32)
                    },
                    {
                        "molecule_id": 1,
                        "atom_indices": [1],
                        "residue_indices": [1],
                        "residue_names": ["HOH"],
                        "chain_ids": ["W"],
                        "kind": "WATER",
                        "identity": "water_0",
                        "component_sha256": "99".repeat(32)
                    }
                ],
                "table_sha256": molecule_table_sha256
            },
            "periodic_cell": {
                "periodic": true,
                "vectors_angstrom": [[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]],
                "volume_angstrom3": 8000.0,
                "orthorhombic": true,
                "minimum_image_required": true,
                "coordinate_wrapping_required": true,
                "reciprocal_method": "PME",
                "real_space_cutoff_angstrom": 9.0,
                "contract_sha256": "aa".repeat(32)
            }
        }))
        .unwrap()
    }

    fn test_ghost_context(stream_id: u16) -> GhostSegmentContext {
        GhostSegmentContext {
            schema_kind: "prism_ghost_segment_context".into(),
            schema_version: 1,
            run_id: "run_exact_coordinate_test".into(),
            replica_id: "rep_00".into(),
            replica_seed: 42,
            stream_seed: 42 + stream_id as u64,
            stream_id,
            topology_sha256: "11".repeat(32),
            selected_engine_sha256: "22".repeat(32),
            record_abi_sha256: ghost_record_abi_sha256(),
            record_source_sha256: ghost_record_source_sha256(),
            thermodynamic_condition_id: format!("condition_{stream_id}"),
            protocol_id: "thermal_shock_v1".into(),
            protocol_state_hash: "33".repeat(32),
            phase_schedule_digest_sha256: "44".repeat(32),
        }
    }

    fn coordinate_context(
        topology: &PrismPrepTopology,
        ghost: &GhostSegmentContext,
    ) -> GhostCoordinateContext {
        GhostCoordinateContext::from_topology(topology, &ghost.topology_sha256, ghost, 3.0).unwrap()
    }

    fn production_record(step: u64, frame_idx: u64, site_id: u32) -> Vec<u8> {
        let mut record = vec![0u8; GHOST_SEGMENT_RECORD_BYTES];
        record[0..8].copy_from_slice(&frame_idx.to_le_bytes());
        record[8..12].copy_from_slice(&site_id.to_le_bytes());
        record[GHOST_V2_OFFSET_SCHEMA_VERSION..GHOST_V2_OFFSET_SCHEMA_VERSION + 4]
            .copy_from_slice(&GHOST_FRAME_SCHEMA_V3.to_le_bytes());
        record[GHOST_V2_OFFSET_STEP_IDX..GHOST_V2_OFFSET_STEP_IDX + 8]
            .copy_from_slice(&step.to_le_bytes());
        for (axis, value) in [1.0f32, 1.0, 1.0].into_iter().enumerate() {
            let offset = GHOST_V2_OFFSET_CENTROID + axis * 4;
            record[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        record[GHOST_V3_OFFSET_MEMBERSHIP_SCHEMA_VERSION
            ..GHOST_V3_OFFSET_MEMBERSHIP_SCHEMA_VERSION + 4]
            .copy_from_slice(&GHOST_V3_MEMBERSHIP_SCHEMA_VERSION.to_le_bytes());
        record[GHOST_V3_OFFSET_CLUSTER_RESIDUE_COUNT..GHOST_V3_OFFSET_CLUSTER_RESIDUE_COUNT + 4]
            .copy_from_slice(&1u32.to_le_bytes());
        record[GHOST_V3_OFFSET_CLUSTER_RESIDUE_CAPACITY
            ..GHOST_V3_OFFSET_CLUSTER_RESIDUE_CAPACITY + 4]
            .copy_from_slice(&(GHOST_V3_CLUSTER_RESIDUE_CAPACITY as u32).to_le_bytes());
        record[GHOST_V3_OFFSET_CLUSTER_RESIDUE_OVERFLOW
            ..GHOST_V3_OFFSET_CLUSTER_RESIDUE_OVERFLOW + 4]
            .copy_from_slice(&0u32.to_le_bytes());
        record[GHOST_V3_OFFSET_SOURCE_SPIKE_COUNT..GHOST_V3_OFFSET_SOURCE_SPIKE_COUNT + 4]
            .copy_from_slice(&1u32.to_le_bytes());
        record[GHOST_V3_OFFSET_UNRESOLVED_RESIDUE_SPIKE_COUNT
            ..GHOST_V3_OFFSET_UNRESOLVED_RESIDUE_SPIKE_COUNT + 4]
            .copy_from_slice(&0u32.to_le_bytes());
        record[GHOST_V3_OFFSET_CLUSTER_RESIDUE_ENTRY_BYTES
            ..GHOST_V3_OFFSET_CLUSTER_RESIDUE_ENTRY_BYTES + 4]
            .copy_from_slice(&(GHOST_V3_CLUSTER_RESIDUE_ENTRY_BYTES as u32).to_le_bytes());
        record[GHOST_V3_OFFSET_GEOMETRY_AVAILABILITY_FLAGS
            ..GHOST_V3_OFFSET_GEOMETRY_AVAILABILITY_FLAGS + 4]
            .copy_from_slice(
                &(GHOST_V3_GEOMETRY_CLUSTER_RESIDUE_MEMBERSHIP_EXACT
                    | GHOST_V3_GEOMETRY_COORDINATE_SIDECAR_REQUIRED)
                    .to_le_bytes(),
            );
        record
            [GHOST_V3_OFFSET_CLUSTER_RESIDUE_ENTRIES..GHOST_V3_OFFSET_CLUSTER_RESIDUE_ENTRIES + 4]
            .copy_from_slice(&0u32.to_le_bytes());
        record[GHOST_V3_OFFSET_CLUSTER_RESIDUE_ENTRIES + 4
            ..GHOST_V3_OFFSET_CLUSTER_RESIDUE_ENTRIES + 8]
            .copy_from_slice(&1u32.to_le_bytes());
        record
    }

    fn ghost_entry(
        context: &GhostSegmentContext,
        payload: &[u8],
        segment_ordinal: u64,
        first_record_ordinal: u64,
    ) -> GhostSegmentEntry {
        GhostSegmentEntry {
            segment_ordinal,
            first_record_ordinal,
            record_count: (payload.len() / GHOST_SEGMENT_RECORD_BYTES) as u64,
            producer_drain_record_count: (payload.len() / GHOST_SEGMENT_RECORD_BYTES) as u64,
            record_schema_version: GHOST_FRAME_SCHEMA_V3,
            payload_offset: 4096,
            payload_bytes: payload.len() as u64,
            file: format!("segment_{segment_ordinal:08}.ghostseg"),
            payload_sha256: sha256_bytes(payload),
            file_sha256: format!("{:064x}", segment_ordinal + 1),
            previous_file_sha256: if segment_ordinal == 0 {
                ZERO_SHA256.into()
            } else {
                format!("{:064x}", segment_ordinal)
            },
            context_digest_sha256: context.digest_sha256().unwrap(),
            fsync_pass: true,
        }
    }

    fn exact_frame(step: u64) -> ExactGhostCoordinateFrame {
        ExactGhostCoordinateFrame {
            step_idx: step,
            positions_angstrom: vec![1.0, 1.0, 1.0, 1.5, 1.0, 1.0],
        }
    }

    #[test]
    fn capacity_plan_uses_exact_analysis_population_and_margin() {
        let dir = tempfile::tempdir().unwrap();
        let plan =
            preflight_coordinate_capacity(dir.path(), 2_790, 8, 10_000, 16_000_000_000).unwrap();
        assert_eq!(plan.analysis_frame_bytes, 33_480);
        assert_eq!(plan.analysis_frame_campaign_bytes, 2_678_400_000);
        assert_eq!(plan.required_disk_bytes, 3_348_000_000);
        assert!(plan.gpu_pass);
    }

    #[test]
    fn shell_binary_roundtrip_is_exact() {
        let shell = vec![ExactLocalShellAtom {
            atom_index: 42,
            molecule_id: 7,
            role: "LAMBDA_FRAGMENT".into(),
            position_angstrom: [1.25, -2.5, 3.75],
        }];
        let bytes = encode_shell(&shell).unwrap();
        assert_eq!(bytes.len(), SHELL_ENTRY_BYTES);
        assert_eq!(decode_shell(&bytes).unwrap(), shell);
        let mut corrupted = bytes;
        corrupted[31] = 1;
        assert!(decode_shell(&corrupted).is_err());
    }

    #[test]
    fn exact_record_step_frame_step_join_closes_conservation() {
        let tmp = tempfile::tempdir().unwrap();
        let topology = test_topology();
        topology.validate_analysis_residue_axis().unwrap();
        let ghost = test_ghost_context(3);
        let context = coordinate_context(&topology, &ghost);
        let mut payload = production_record(1000, 7, 11);
        payload.extend_from_slice(&production_record(1000, 7, 12));
        let entry = ghost_entry(&ghost, &payload, 0, 0);
        let mut writer =
            GhostCoordinateSegmentWriter::create(tmp.path(), "mycmax", &topology, context).unwrap();
        writer
            .append_ghost_segment(&entry, &payload, &[exact_frame(1000)], |_| Some(2))
            .unwrap();
        writer.complete(2).unwrap();

        let loaded = load_verified_ghost_coordinate_segments(tmp.path(), &topology).unwrap();
        assert!(loaded.equality_pass);
        assert_eq!(loaded.requested_records, 2);
        assert_eq!(loaded.persisted_records, 2);
        assert_eq!(loaded.decoded_records, 2);
        assert_eq!(loaded.indexed_records, 2);
        assert_eq!(loaded.frames[&(3, 1000)], vec![1.0, 1.0, 1.0]);
        assert!(loaded.index.values().all(|row| {
            row.entry.protocol_phase == 2
                && row.entry.excluded_population_scanned == 1
                && row.local_shell.len() == 1
                && row.local_shell[0].atom_index == 1
                && row.local_shell[0].role == "WATER"
        }));
    }

    #[test]
    fn mismatched_record_and_frame_steps_are_rejected_before_persistence() {
        let tmp = tempfile::tempdir().unwrap();
        let topology = test_topology();
        let ghost = test_ghost_context(0);
        let payload = production_record(1000, 7, 11);
        let entry = ghost_entry(&ghost, &payload, 0, 0);
        let mut writer = GhostCoordinateSegmentWriter::create(
            tmp.path(),
            "mycmax",
            &topology,
            coordinate_context(&topology, &ghost),
        )
        .unwrap();
        let error = writer
            .append_ghost_segment(&entry, &payload, &[exact_frame(999)], |_| Some(1))
            .unwrap_err()
            .to_string();
        assert!(error.contains("interpolation is forbidden"));
        assert!(writer.manifest().segments.is_empty());
        assert_eq!(writer.manifest().persisted_ghost_joins, 0);
    }

    #[test]
    fn duplicate_keys_and_segment_ordinal_gaps_are_rejected() {
        let topology = test_topology();
        let ghost = test_ghost_context(0);

        let duplicate_tmp = tempfile::tempdir().unwrap();
        let record = production_record(1000, 7, 11);
        let mut duplicate_payload = record.clone();
        duplicate_payload.extend_from_slice(&record);
        let duplicate_entry = ghost_entry(&ghost, &duplicate_payload, 0, 0);
        let mut duplicate_writer = GhostCoordinateSegmentWriter::create(
            duplicate_tmp.path(),
            "mycmax",
            &topology,
            coordinate_context(&topology, &ghost),
        )
        .unwrap();
        assert!(duplicate_writer
            .append_ghost_segment(
                &duplicate_entry,
                &duplicate_payload,
                &[exact_frame(1000)],
                |_| Some(1),
            )
            .unwrap_err()
            .to_string()
            .contains("duplicate Ghost coordinate key"));

        let gap_tmp = tempfile::tempdir().unwrap();
        let payload = production_record(1000, 7, 11);
        let gap_entry = ghost_entry(&ghost, &payload, 1, 0);
        let mut gap_writer = GhostCoordinateSegmentWriter::create(
            gap_tmp.path(),
            "mycmax",
            &topology,
            coordinate_context(&topology, &ghost),
        )
        .unwrap();
        assert!(gap_writer
            .append_ghost_segment(&gap_entry, &payload, &[exact_frame(1000)], |_| Some(1))
            .unwrap_err()
            .to_string()
            .contains("segment ordinal gap"));
    }

    #[test]
    fn restart_resume_preserves_hash_chain_and_exactly_once_population() {
        let tmp = tempfile::tempdir().unwrap();
        let topology = test_topology();
        let ghost = test_ghost_context(2);
        let context = coordinate_context(&topology, &ghost);
        let payload0 = production_record(1000, 7, 11);
        let entry0 = ghost_entry(&ghost, &payload0, 0, 0);
        {
            let mut writer = GhostCoordinateSegmentWriter::create(
                tmp.path(),
                "mycmax",
                &topology,
                context.clone(),
            )
            .unwrap();
            writer
                .append_ghost_segment(&entry0, &payload0, &[exact_frame(1000)], |_| Some(1))
                .unwrap();
        }
        let payload1 = production_record(2000, 8, 12);
        let entry1 = ghost_entry(&ghost, &payload1, 1, 1);
        let mut resumed =
            GhostCoordinateSegmentWriter::resume(tmp.path(), "mycmax", &topology, context).unwrap();
        resumed
            .append_ghost_segment(&entry1, &payload1, &[exact_frame(2000)], |_| Some(2))
            .unwrap();
        resumed.complete(2).unwrap();
        let loaded = load_verified_ghost_coordinate_segments(tmp.path(), &topology).unwrap();
        assert!(loaded.equality_pass);
        assert_eq!(loaded.indexed_records, 2);
        assert_eq!(loaded.frames.len(), 2);
    }

    #[test]
    fn corruption_and_hash_chain_tamper_fail_closed() {
        let tmp = tempfile::tempdir().unwrap();
        let topology = test_topology();
        let ghost = test_ghost_context(0);
        let payload = production_record(1000, 7, 11);
        let entry = ghost_entry(&ghost, &payload, 0, 0);
        let mut writer = GhostCoordinateSegmentWriter::create(
            tmp.path(),
            "mycmax",
            &topology,
            coordinate_context(&topology, &ghost),
        )
        .unwrap();
        writer
            .append_ghost_segment(&entry, &payload, &[exact_frame(1000)], |_| Some(1))
            .unwrap();
        writer.complete(1).unwrap();
        let root = coordinate_segment_root(tmp.path(), "mycmax", 0);
        let segment_path = root.join("segment_00000000.ghostcoord");
        let mut bytes = std::fs::read(&segment_path).unwrap();
        bytes[GHOST_COORDINATE_HEADER_BYTES + FRAME_HEADER_BYTES] ^= 0x01;
        std::fs::write(&segment_path, bytes).unwrap();
        assert!(
            load_verified_ghost_coordinate_segments(tmp.path(), &topology)
                .unwrap_err()
                .to_string()
                .contains("content hash/length mismatch")
        );

        let chain_tmp = tempfile::tempdir().unwrap();
        let mut chain_writer = GhostCoordinateSegmentWriter::create(
            chain_tmp.path(),
            "mycmax",
            &topology,
            coordinate_context(&topology, &ghost),
        )
        .unwrap();
        chain_writer
            .append_ghost_segment(&entry, &payload, &[exact_frame(1000)], |_| Some(1))
            .unwrap();
        let payload1 = production_record(2000, 8, 12);
        let entry1 = ghost_entry(&ghost, &payload1, 1, 1);
        chain_writer
            .append_ghost_segment(&entry1, &payload1, &[exact_frame(2000)], |_| Some(2))
            .unwrap();
        chain_writer.complete(2).unwrap();
        let chain_root = coordinate_segment_root(chain_tmp.path(), "mycmax", 0);
        let manifest_path = chain_root.join("manifest.json");
        let mut manifest: GhostCoordinateManifest =
            serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
        manifest.segments[1].previous_file_sha256 = "ff".repeat(32);
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        assert!(
            load_verified_ghost_coordinate_segments(chain_tmp.path(), &topology)
                .unwrap_err()
                .to_string()
                .contains("ordinal/chain/count failure")
        );
    }

    fn append_test_coordinate_chunk(
        writer: &mut GhostCoordinateSegmentWriter,
        ghost: &GhostSegmentContext,
        ordinal: u64,
    ) {
        let step = (ordinal + 1) * 10;
        let payload = production_record(step, ordinal, 11);
        let entry = ghost_entry(ghost, &payload, ordinal, ordinal);
        writer
            .append_ghost_segment(&entry, &payload, &[exact_frame(step)], |_| Some(1))
            .unwrap();
    }

    #[test]
    fn persistence_journal_live_prefix_resume_and_final_equality() {
        let tmp = tempfile::tempdir().unwrap();
        let topology = test_topology();
        let ghost = test_ghost_context(0);
        let context = coordinate_context(&topology, &ghost);
        let mut writer =
            GhostCoordinateSegmentWriter::create(tmp.path(), "mycmax", &topology, context.clone())
                .unwrap();
        let root = coordinate_segment_root(tmp.path(), "mycmax", 0);
        let checkpoint_before = std::fs::read(root.join("manifest.json")).unwrap();
        let mut legacy_cumulative_serialized_bytes = checkpoint_before.len() as u64;
        for ordinal in 0..32 {
            append_test_coordinate_chunk(&mut writer, &ghost, ordinal);
            legacy_cumulative_serialized_bytes +=
                serde_json::to_vec_pretty(writer.manifest()).unwrap().len() as u64;
            assert_eq!(
                std::fs::read(root.join("manifest.json")).unwrap(),
                checkpoint_before,
                "runtime chunks must not rewrite cumulative manifest"
            );
            let live = load_verified_live_ghost_coordinate_manifest(&root, &topology).unwrap();
            assert_eq!(&live, writer.manifest());
            assert!(!live.completed);
            assert_eq!(live.persisted_ghost_joins, ordinal + 1);
        }
        let before_restart = writer.manifest().clone();
        let metadata_work = writer.persistence_work().clone();
        assert_eq!(metadata_work.checkpoint_writes, 1);
        assert_eq!(metadata_work.journal_entries_written, 32);
        drop(writer);
        let mut writer =
            GhostCoordinateSegmentWriter::resume(tmp.path(), "mycmax", &topology, context).unwrap();
        assert_eq!(writer.manifest(), &before_restart);
        append_test_coordinate_chunk(&mut writer, &ghost, 32);
        writer.complete(33).unwrap();
        let disk: GhostCoordinateManifest =
            serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(&disk, writer.manifest());
        let loaded = load_verified_ghost_coordinate_segments(tmp.path(), &topology).unwrap();
        assert!(loaded.equality_pass);
        assert_eq!(loaded.indexed_records, 33);
        let actual_metadata_bytes = metadata_work.journal_bytes_written
            + metadata_work.checkpoint_bytes_written
            + writer.persistence_work().journal_bytes_written
            + writer.persistence_work().checkpoint_bytes_written;
        // This counts actual serialized/written metadata, not inferred GPU speed.
        assert!(legacy_cumulative_serialized_bytes > actual_metadata_bytes * 8);
        eprintln!(
            "{}",
            serde_json::json!({"measurement":"ghost_coordinate_metadata_serialized_bytes",
            "prior_32_cumulative_pretty_manifest_bytes":legacy_cumulative_serialized_bytes,
            "new_33_journal_plus_initial_and_final_checkpoint_bytes":actual_metadata_bytes,
            "coordinate_and_join_payloads_unchanged":true,"durable_chunks":33,
            "runtime_gpu_throughput_claim":false})
        );
    }

    #[test]
    fn persistence_orphan_after_segment_rename_recovers_exact_index() {
        let tmp = tempfile::tempdir().unwrap();
        let topology = test_topology();
        let ghost = test_ghost_context(0);
        let context = coordinate_context(&topology, &ghost);
        let mut writer =
            GhostCoordinateSegmentWriter::create(tmp.path(), "mycmax", &topology, context.clone())
                .unwrap();
        append_test_coordinate_chunk(&mut writer, &ghost, 0);
        let expected = writer.manifest().clone();
        drop(writer);
        let root = coordinate_segment_root(tmp.path(), "mycmax", 0);
        // Simulate the durable segment rename preceding journal commit.
        std::fs::remove_file(
            root.join(INDEX_JOURNAL_DIRECTORY)
                .join("segment_00000000.json"),
        )
        .unwrap();
        let mut recovered =
            GhostCoordinateSegmentWriter::resume(tmp.path(), "mycmax", &topology, context).unwrap();
        assert_eq!(recovered.manifest(), &expected);
        assert_eq!(recovered.persistence_work().journal_entries_written, 1);
        recovered.complete(1).unwrap();
        assert!(
            load_verified_ghost_coordinate_segments(tmp.path(), &topology)
                .unwrap()
                .equality_pass
        );
    }

    #[test]
    fn persistence_legacy_checkpoint_prefix_and_covered_entry_conflict() {
        let tmp = tempfile::tempdir().unwrap();
        let topology = test_topology();
        let ghost = test_ghost_context(0);
        let context = coordinate_context(&topology, &ghost);
        let mut writer =
            GhostCoordinateSegmentWriter::create(tmp.path(), "mycmax", &topology, context.clone())
                .unwrap();
        append_test_coordinate_chunk(&mut writer, &ghost, 0);
        let checkpoint = writer.manifest().clone();
        let root = coordinate_segment_root(tmp.path(), "mycmax", 0);
        drop(writer);
        // A legacy incomplete manifest carries its complete prefix directly.
        write_manifest_atomic(&root, &root.join("manifest.json"), &checkpoint).unwrap();
        std::fs::remove_dir_all(root.join(INDEX_JOURNAL_DIRECTORY)).unwrap();
        let mut resumed =
            GhostCoordinateSegmentWriter::resume(tmp.path(), "mycmax", &topology, context).unwrap();
        assert_eq!(resumed.manifest(), &checkpoint);
        append_test_coordinate_chunk(&mut resumed, &ghost, 1);
        resumed.complete(2).unwrap();
        let journal = root
            .join(INDEX_JOURNAL_DIRECTORY)
            .join("segment_00000001.json");
        let mut row: CoordinateIndexJournalEntry =
            serde_json::from_slice(&std::fs::read(&journal).unwrap()).unwrap();
        row.entry.joins[0].protocol_phase = 3;
        // Even a rehashed journal may not contradict its completed checkpoint.
        row.entry_sha256 = sha256_bytes(&serde_json::to_vec(&row.entry).unwrap());
        std::fs::write(&journal, serde_json::to_vec(&row).unwrap()).unwrap();
        assert!(
            load_verified_live_ghost_coordinate_manifest(&root, &topology)
                .unwrap_err()
                .to_string()
                .contains("conflicts with checkpoint")
        );
        // Completed legacy consumer authority remains the complete manifest.
        assert!(
            load_verified_ghost_coordinate_segments(tmp.path(), &topology)
                .unwrap()
                .equality_pass
        );
    }

    #[test]
    fn persistence_journal_truncation_tamper_gaps_and_partials_fail_closed() {
        for failure in ["truncated", "tampered", "ordinal_gap", "partial"] {
            let tmp = tempfile::tempdir().unwrap();
            let topology = test_topology();
            let ghost = test_ghost_context(0);
            let context = coordinate_context(&topology, &ghost);
            let mut writer = GhostCoordinateSegmentWriter::create(
                tmp.path(),
                "mycmax",
                &topology,
                context.clone(),
            )
            .unwrap();
            append_test_coordinate_chunk(&mut writer, &ghost, 0);
            append_test_coordinate_chunk(&mut writer, &ghost, 1);
            drop(writer);
            let root = coordinate_segment_root(tmp.path(), "mycmax", 0);
            let journal = root.join(INDEX_JOURNAL_DIRECTORY);
            let first = journal.join("segment_00000000.json");
            match failure {
                "truncated" => {
                    let bytes = std::fs::read(&first).unwrap();
                    std::fs::write(&first, &bytes[..bytes.len() / 2]).unwrap();
                }
                "tampered" => {
                    let mut row: CoordinateIndexJournalEntry =
                        serde_json::from_slice(&std::fs::read(&first).unwrap()).unwrap();
                    row.entry.joins[0].protocol_phase = 3;
                    std::fs::write(&first, serde_json::to_vec(&row).unwrap()).unwrap();
                }
                "ordinal_gap" => {
                    std::fs::remove_file(&first).unwrap();
                }
                "partial" => {
                    std::fs::write(
                        journal.join(".segment_00000002.json.partial"),
                        b"incomplete",
                    )
                    .unwrap();
                    let live =
                        load_verified_live_ghost_coordinate_manifest(&root, &topology).unwrap();
                    assert_eq!(live.persisted_ghost_joins, 2);
                    assert!(!live.completed);
                }
                _ => unreachable!(),
            }
            assert!(
                GhostCoordinateSegmentWriter::resume(tmp.path(), "mycmax", &topology, context)
                    .is_err(),
                "{failure}"
            );
        }
    }

    #[test]
    fn local_shell_complete_scan_and_tamper_detection_are_fail_closed() {
        let topology = test_topology();
        let roles = molecule_role_map(&topology).unwrap();
        let spatial_distance = spatial_distance(&topology).unwrap();
        let positions = vec![1.0, 1.0, 1.0, 1.5, 1.0, 1.0];
        let shell = exact_local_shell(
            &topology,
            &roles,
            &positions,
            [1.0, 1.0, 1.0],
            3.0,
            spatial_distance,
        )
        .unwrap();
        assert_eq!(roles.iter().filter(|role| role.analysis_id < 0).count(), 1);
        assert_eq!(shell.len(), 1);
        let mut encoded = encode_shell(&shell).unwrap();
        encoded[12] = 0xff;
        assert!(decode_shell(&encoded)
            .unwrap_err()
            .to_string()
            .contains("invalid local-shell role code"));

        let mut incomplete_join = GhostCoordinateJoinEntry {
            key: GhostCoordinateKey {
                stream_id: 0,
                step_idx: 1,
                ghost_site_id: 1,
                ghost_record_sha256: "11".repeat(32),
            },
            stream_record_ordinal: 0,
            frame_idx: 0,
            ghost_segment_ordinal: 0,
            ghost_segment_file: "segment_00000000.ghostseg".into(),
            ghost_segment_file_sha256: "22".repeat(32),
            ghost_payload_byte_offset: 4096,
            ghost_byte_length: GHOST_SEGMENT_RECORD_BYTES as u32,
            ghost_centroid_angstrom: [1.0, 1.0, 1.0],
            cluster_residue_membership_sha256: "33".repeat(32),
            thermodynamic_condition_id: "condition_0".into(),
            protocol_phase: 1,
            protocol_state_hash: "44".repeat(32),
            phase_schedule_digest_sha256: "55".repeat(32),
            coordinate_frame_ordinal: 0,
            coordinate_frame_byte_offset: 0,
            coordinate_frame_byte_length: 0,
            coordinate_frame_sha256: "66".repeat(32),
            local_shell_status: "COMPLETE_EXACT_HASH_BOUND_FULL_POPULATION_SCAN".into(),
            local_shell_cutoff_angstrom: 3.0,
            local_shell_atom_count: 1,
            local_shell_byte_offset: 0,
            local_shell_byte_length: SHELL_ENTRY_BYTES as u64,
            local_shell_payload_sha256: sha256_bytes(&encode_shell(&shell).unwrap()),
            excluded_population_scanned: 0,
            join_byte_offset: 0,
            join_byte_length: 0,
            join_record_sha256: "77".repeat(32),
        };
        assert!(verify_persisted_local_shell(
            topology.n_atoms,
            &roles,
            &shell,
            &incomplete_join,
            spatial_distance,
        )
        .unwrap_err()
        .to_string()
        .contains("complete excluded-population scan"));
        incomplete_join.excluded_population_scanned = 1;
        assert!(verify_persisted_local_shell(
            topology.n_atoms,
            &roles,
            &shell,
            &incomplete_join,
            spatial_distance,
        )
        .is_ok());
    }

    fn amber_prmtop_fixture(natom: i64, nres: i64, ifbox: i64, box_flag: bool) -> Vec<u8> {
        let mut pointers = vec![0i64; 31];
        pointers[0] = natom;
        pointers[11] = nres;
        pointers[27] = ifbox;
        let mut text =
            String::from("%VERSION VERSION_STAMP = V0001.000\n%FLAG POINTERS\n%FORMAT(10I8)\n");
        for row in pointers.chunks(10) {
            for value in row {
                text.push_str(&format!("{value:8}"));
            }
            text.push('\n');
        }
        if box_flag {
            text.push_str("%FLAG BOX_DIMENSIONS\n%FORMAT(5E16.8)\n");
        } else {
            text.push_str("%FLAG ATOM_NAME\n%FORMAT(20a4)\n");
        }
        text.into_bytes()
    }

    fn legacy_authority_fixture() -> (
        PrismPrepTopology,
        LegacyGhostTopologyAuthority,
        LegacyGhostTopologyPolicy,
        Vec<u8>,
        String,
    ) {
        let mut effective = test_topology();
        effective.residue_names = vec!["ALA".into(), "GLY".into()];
        effective.chain_ids = vec!["A".into(), "A".into()];
        effective.elements = vec!["C".into(), "C".into()];
        effective.atom_names = vec!["CA".into(), "CA".into()];
        effective.bonds = vec![crate::input::BondParam {
            i: 0,
            j: 1,
            r0: 1.5,
            k: 100.0,
        }];
        let source_prmtop = amber_prmtop_fixture(2, 2, 0, false);
        let source_prmtop_sha256 = sha256_bytes(&source_prmtop);
        effective.generated_prmtop_sha256 = Some(source_prmtop_sha256.clone());
        let mut boundary = PeriodicCellContract {
            periodic: false,
            reason: Some("LEGACY_RECEPTOR_ONLY_NONPERIODIC_TOPOLOGY_AMBER_IFBOX_0".into()),
            vectors_angstrom: None,
            volume_angstrom3: None,
            orthorhombic: None,
            minimum_image_required: Some(false),
            coordinate_wrapping_required: Some(false),
            reciprocal_method: Some("NONE_NONPERIODIC_DIRECT_DISTANCE".into()),
            real_space_cutoff_angstrom: None,
            contract_sha256: None,
            ..Default::default()
        };
        boundary.contract_sha256 =
            Some(canonical_sha256_without_field(&boundary, "contract_sha256").unwrap());
        effective.periodic_cell = Some(boundary.clone());
        let table = canonical_molecule_table(&effective).unwrap();
        let table_sha256 = table.table_sha256.clone();
        effective.molecule_identity_table = Some(table.clone());
        effective.total_physical_residue_count = Some(2);
        effective.n_analysis_residues = Some(2);
        effective.atom_physical_residue_ids = Some(vec![0, 1]);
        effective.atom_analysis_residue_ids = Some(vec![0, 1]);
        let mut role_counts = BTreeMap::new();
        role_counts.insert("RECEPTOR".to_string(), 2);
        let mut analysis = AnalysisResidueContract {
            schema_kind: "prismos_analysis_residue_mapping".into(),
            schema_version: 1,
            analysis_authority: "KCC_GHOST_PROTEIN_RESIDUE_AXIS".into(),
            excluded_atom_sentinel: -1,
            n_atoms: 2,
            total_physical_residue_count: 2,
            n_analysis_residues: 2,
            physical_residue_roles: vec!["RECEPTOR".into(), "RECEPTOR".into()],
            physical_to_analysis_residue_ids: vec![0, 1],
            analysis_residues: vec![
                crate::input::AnalysisResidueRecord {
                    analysis_residue_id: 0,
                    physical_residue_id: 0,
                    chain_id: "A".into(),
                    pdb_residue_id: 1,
                    insertion_code: String::new(),
                    residue_name: "ALA".into(),
                    role: "RECEPTOR".into(),
                },
                crate::input::AnalysisResidueRecord {
                    analysis_residue_id: 1,
                    physical_residue_id: 1,
                    chain_id: "A".into(),
                    pdb_residue_id: 2,
                    insertion_code: String::new(),
                    residue_name: "GLY".into(),
                    role: "RECEPTOR".into(),
                },
            ],
            atom_analysis_residue_ids_sha256: crate::input::python_canonical_sha256(
                &serde_json::json!([0, 1]),
            )
            .unwrap(),
            physical_molecule_table_sha256: table_sha256,
            lambda_fragment_system_sha256: None,
            role_counts,
            excluded_roles: vec![
                "WATER".into(),
                "ION".into(),
                "LAMBDA_FRAGMENT".into(),
                "OTHER".into(),
            ],
            must_recompute_after_fragment_materialization: true,
            contract_sha256: String::new(),
        };
        analysis.contract_sha256 =
            canonical_sha256_without_field(&analysis, "contract_sha256").unwrap();
        effective.analysis_residue_contract = Some(analysis.clone());
        let topology_sha256 = "ab".repeat(32);
        let producer_sha256 = "cd".repeat(32);
        let mut policy = LegacyGhostTopologyPolicy {
            schema_kind: "prismos_legacy_ghost_topology_policy".into(),
            schema_version: 1,
            policy_id: "TEST_LEGACY_GHOST_TOPOLOGY".into(),
            topology: LegacyTopologyIdentityPolicy {
                sha256: topology_sha256.clone(),
                size_bytes: 1234,
                n_atoms: 2,
                n_residues: 2,
                n_chains: 1,
                chain_ids: vec!["A".into()],
                generated_prmtop_sha256: source_prmtop_sha256.clone(),
                source_prmtop_artifact: "synthetic.prmtop".into(),
            },
            source_parameter_topology: LegacyParameterTopologyPolicy {
                sha256: source_prmtop_sha256.clone(),
                size_bytes: source_prmtop.len() as u64,
                amber_pointer_format: "AMBER_PRMTOP_POINTERS_10I8".into(),
                natom: 2,
                nres: 2,
                ifbox: 0,
                box_dimensions_required_absent: true,
            },
            projection: LegacyProjectionPolicy {
                analysis_authority: "KCC_GHOST_PROTEIN_RESIDUE_AXIS".into(),
                spatial_distance_mode: "NONPERIODIC_DIRECT_DISTANCE".into(),
                spatial_boundary_reason: "LEGACY_RECEPTOR_ONLY_NONPERIODIC_TOPOLOGY_AMBER_IFBOX_0"
                    .into(),
                protein_residue_names: vec!["ALA".into(), "GLY".into()],
                expected_molecule_count: 1,
                expected_molecule_kind: "BIOMOLECULE".into(),
                expected_molecule_identity: "A".into(),
            },
            authority_producer_sha256: producer_sha256.clone(),
            policy_contract_sha256: String::new(),
        };
        policy.policy_contract_sha256 =
            canonical_sha256_without_field(&policy, "policy_contract_sha256").unwrap();
        let policy_sha256 = sha256_bytes(&serde_json::to_vec(&policy).unwrap());
        let mut authority = LegacyGhostTopologyAuthority {
            schema_kind: "prismos_legacy_ghost_topology_authority".into(),
            schema_version: 2,
            policy_sha256,
            policy_contract_sha256: policy.policy_contract_sha256.clone(),
            topology_sha256: topology_sha256.clone(),
            topology_size_bytes: 1234,
            topology_atom_count: effective.n_atoms,
            topology_physical_residue_count: effective.n_residues,
            topology_analysis_residue_count: effective.n_analysis_residues.unwrap(),
            source_parameter_topology_sha256: source_prmtop_sha256,
            source_parameter_topology_size_bytes: source_prmtop.len() as u64,
            source_parameter_topology_natom: 2,
            source_parameter_topology_nres: 2,
            source_parameter_topology_ifbox: 0,
            source_parameter_topology_box_dimensions_present: false,
            atom_physical_residue_ids: effective
                .atom_physical_residue_ids
                .as_ref()
                .unwrap()
                .clone(),
            atom_analysis_residue_ids: effective
                .atom_analysis_residue_ids
                .as_ref()
                .unwrap()
                .clone(),
            analysis_residue_contract: effective
                .analysis_residue_contract
                .as_ref()
                .unwrap()
                .clone(),
            molecule_identity_table: effective.molecule_identity_table.as_ref().unwrap().clone(),
            spatial_boundary: boundary,
            spatial_distance_mode: "NONPERIODIC_DIRECT_DISTANCE".into(),
            atom_order_sha256: native_atom_order_sha256(&effective).unwrap(),
            producer_source_sha256: producer_sha256,
            authority_contract_sha256: String::new(),
        };
        authority.authority_contract_sha256 =
            canonical_sha256_without_field(&authority, "authority_contract_sha256").unwrap();
        let mut legacy = effective;
        legacy.total_physical_residue_count = None;
        legacy.n_analysis_residues = None;
        legacy.atom_physical_residue_ids = None;
        legacy.atom_analysis_residue_ids = None;
        legacy.analysis_residue_contract = None;
        legacy.molecule_identity_table = None;
        legacy.periodic_cell = None;
        (legacy, authority, policy, source_prmtop, topology_sha256)
    }

    fn resolve_fixture(
        topology: &PrismPrepTopology,
        topology_sha256: &str,
        authority: &LegacyGhostTopologyAuthority,
        policy: &LegacyGhostTopologyPolicy,
        source_prmtop: &[u8],
    ) -> Result<ResolvedGhostTopology> {
        resolve_legacy_ghost_topology_authority(
            topology,
            topology_sha256,
            &serde_json::to_vec(authority).unwrap(),
            &serde_json::to_vec(policy).unwrap(),
            source_prmtop,
        )
    }

    fn reseal_authority(authority: &mut LegacyGhostTopologyAuthority) {
        authority.authority_contract_sha256 = String::new();
        authority.authority_contract_sha256 =
            canonical_sha256_without_field(authority, "authority_contract_sha256").unwrap();
    }

    /// Under a policy-pinned (`v2_ignition`) build the resolver admits exactly
    /// one policy artifact: the file `build.rs` hashed at compile time and
    /// exposed through `PRISM_GHOST_TOPOLOGY_POLICY_BUILD_PATH`. Read that same
    /// file here so the compiled and runtime digests are compared on the
    /// genuine pinned bytes; a synthetic policy can never satisfy the pin.
    #[cfg(feature = "v2_ignition")]
    fn compiled_policy_fixture() -> (Vec<u8>, String) {
        let build_path = Path::new(COMPILED_GHOST_POLICY_BUILD_PATH);
        assert!(build_path.is_absolute());
        require_sha256(COMPILED_GHOST_POLICY_SHA256, "compiled_ghost_policy_sha256").unwrap();
        let bytes = read_regular_nofollow(build_path).unwrap();
        let sha256 = sha256_bytes(&bytes);
        assert_eq!(sha256, COMPILED_GHOST_POLICY_SHA256);
        (bytes, sha256)
    }

    /// Unpinned builds admit the hash-bound synthetic authority and refuse
    /// every mutation. Policy-pinned builds cannot admit a synthetic policy at
    /// all: the compiled digest gate refuses it before any authority
    /// cross-check runs, while the pinned policy file itself passes the gate
    /// and is then held to the same authority identity checks.
    #[test]
    fn legacy_nonperiodic_authority_is_explicit_hash_bound_and_fail_closed() {
        let (legacy, authority, policy, source_prmtop, topology_sha256) =
            legacy_authority_fixture();
        #[cfg(feature = "v2_ignition")]
        {
            let (compiled_bytes, compiled_sha256) = compiled_policy_fixture();
            let synthetic_sha256 = sha256_bytes(&serde_json::to_vec(&policy).unwrap());
            assert_ne!(synthetic_sha256, compiled_sha256);
            require_compiled_ghost_policy(&compiled_sha256).unwrap();
            assert!(require_compiled_ghost_policy(&synthetic_sha256)
                .unwrap_err()
                .to_string()
                .contains("differs from compiled v2 policy digest"));

            let refusal = resolve_fixture(
                &legacy,
                &topology_sha256,
                &authority,
                &policy,
                &source_prmtop,
            )
            .unwrap_err()
            .to_string();
            assert!(refusal.contains("differs from compiled v2 policy digest"));
            assert!(refusal.contains(&format!("compiled={compiled_sha256}")));
            assert!(refusal.contains(&format!("runtime={synthetic_sha256}")));

            let (compiled_policy, admitted_sha256) =
                decode_legacy_topology_policy(&compiled_bytes).unwrap();
            assert_eq!(admitted_sha256, compiled_sha256);
            assert_eq!(
                compiled_policy.schema_kind,
                "prismos_legacy_ghost_topology_policy"
            );
            assert_eq!(
                canonical_sha256_without_field(&compiled_policy, "policy_contract_sha256").unwrap(),
                compiled_policy.policy_contract_sha256
            );

            let past_gate = resolve_legacy_ghost_topology_authority(
                &legacy,
                &topology_sha256,
                &serde_json::to_vec(&authority).unwrap(),
                &compiled_bytes,
                &source_prmtop,
            )
            .unwrap_err()
            .to_string();
            assert!(!past_gate.contains("differs from compiled v2 policy digest"));
            assert!(past_gate.contains("identity or shape mismatch"));
        }
        #[cfg(not(feature = "v2_ignition"))]
        {
            let resolved = resolve_fixture(
                &legacy,
                &topology_sha256,
                &authority,
                &policy,
                &source_prmtop,
            )
            .unwrap();
            assert_eq!(resolved.projection_mode, "HASH_BOUND_LEGACY_SIDECAR");
            assert_eq!(
                validate_spatial_boundary(&resolved.topology).unwrap(),
                "NONPERIODIC_DIRECT_DISTANCE"
            );

            assert!(resolve_fixture(
                &legacy,
                &"cd".repeat(32),
                &authority,
                &policy,
                &source_prmtop,
            )
            .unwrap_err()
            .to_string()
            .contains("identity or shape mismatch"));

            let mut bad_contract = authority.clone();
            bad_contract.authority_contract_sha256 = "ef".repeat(32);
            assert!(resolve_fixture(
                &legacy,
                &topology_sha256,
                &bad_contract,
                &policy,
                &source_prmtop,
            )
            .unwrap_err()
            .to_string()
            .contains("digest mismatch"));

            let mut bad_map = authority.clone();
            bad_map.atom_analysis_residue_ids[0] = -1;
            reseal_authority(&mut bad_map);
            assert!(
                resolve_fixture(&legacy, &topology_sha256, &bad_map, &policy, &source_prmtop,)
                    .is_err()
            );

            let mut bad_boundary = authority.clone();
            bad_boundary.spatial_distance_mode = "PERIODIC_TRICLINIC_MINIMUM_IMAGE".into();
            reseal_authority(&mut bad_boundary);
            assert!(resolve_fixture(
                &legacy,
                &topology_sha256,
                &bad_boundary,
                &policy,
                &source_prmtop,
            )
            .unwrap_err()
            .to_string()
            .contains("identity or shape mismatch"));

            let mut bad_producer = authority.clone();
            bad_producer.producer_source_sha256 = "01".repeat(32);
            reseal_authority(&mut bad_producer);
            assert!(resolve_fixture(
                &legacy,
                &topology_sha256,
                &bad_producer,
                &policy,
                &source_prmtop,
            )
            .unwrap_err()
            .to_string()
            .contains("identity or shape mismatch"));

            let mut nonprotein = legacy;
            nonprotein.residue_names[0] = "LIG".into();
            assert!(resolve_fixture(
                &nonprotein,
                &topology_sha256,
                &authority,
                &policy,
                &source_prmtop,
            )
            .is_err());

            let periodic_prmtop = amber_prmtop_fixture(2, 2, 1, true);
            assert!(resolve_fixture(
                &nonprotein,
                &topology_sha256,
                &authority,
                &policy,
                &periodic_prmtop,
            )
            .is_err());
        }
    }

    #[test]
    fn legacy_compiled_policy_binding_is_fail_closed() {
        let compiled_sha256 = "ab".repeat(32);
        require_ghost_policy_binding(
            &compiled_sha256,
            Some(&compiled_sha256),
            Some("/sealed/source/GHOST_TOPOLOGY_POLICY.json"),
        )
        .unwrap();

        let mismatch = require_ghost_policy_binding(
            &"cd".repeat(32),
            Some(&compiled_sha256),
            Some("/sealed/source/GHOST_TOPOLOGY_POLICY.json"),
        )
        .unwrap_err()
        .to_string();
        assert!(mismatch.contains("differs from compiled v2 policy digest"));

        let relative_path = require_ghost_policy_binding(
            &compiled_sha256,
            Some(&compiled_sha256),
            Some("relative/GHOST_TOPOLOGY_POLICY.json"),
        )
        .unwrap_err()
        .to_string();
        assert!(relative_path.contains("source path is not absolute"));

        let incomplete =
            require_ghost_policy_binding(&compiled_sha256, Some(&compiled_sha256), None)
                .unwrap_err()
                .to_string();
        assert!(incomplete.contains("binding is incomplete"));
    }

    #[test]
    fn legacy_path_preflight_is_nofollow_and_returns_serializable_admission() {
        let (legacy, mut authority, mut policy, source_prmtop, _) = legacy_authority_fixture();
        let topology_bytes = serde_json::to_vec(&legacy).unwrap();
        let topology_sha256 = sha256_bytes(&topology_bytes);
        policy.topology.sha256 = topology_sha256.clone();
        policy.topology.size_bytes = topology_bytes.len() as u64;
        policy.policy_contract_sha256 = String::new();
        policy.policy_contract_sha256 =
            canonical_sha256_without_field(&policy, "policy_contract_sha256").unwrap();
        let policy_bytes = serde_json::to_vec(&policy).unwrap();
        let policy_sha256 = sha256_bytes(&policy_bytes);
        authority.topology_sha256 = topology_sha256.clone();
        authority.topology_size_bytes = topology_bytes.len() as u64;
        authority.policy_sha256 = policy_sha256.clone();
        authority.policy_contract_sha256 = policy.policy_contract_sha256.clone();
        reseal_authority(&mut authority);
        let authority_bytes = serde_json::to_vec(&authority).unwrap();
        let authority_sha256 = sha256_bytes(&authority_bytes);
        let source_prmtop_sha256 = sha256_bytes(&source_prmtop);

        let root = tempfile::tempdir().unwrap();
        let topology_path = root.path().join("topology.json");
        let authority_path = root.path().join("authority.json");
        let policy_path = root.path().join("policy.json");
        let source_prmtop_path = root.path().join("source.prmtop");
        std::fs::write(&topology_path, &topology_bytes).unwrap();
        std::fs::write(&authority_path, &authority_bytes).unwrap();
        std::fs::write(&policy_path, &policy_bytes).unwrap();
        std::fs::write(&source_prmtop_path, &source_prmtop).unwrap();

        #[cfg(not(feature = "v2_ignition"))]
        {
            let admission = preflight_legacy_ghost_topology_paths(
                &topology_path,
                &topology_sha256,
                &authority_path,
                &authority_sha256,
                &policy_path,
                &policy_sha256,
                &source_prmtop_path,
                &source_prmtop_sha256,
            )
            .unwrap();
            assert_eq!(admission.status, "VERIFIED");
            assert!(!admission.cuda_context_created);
            assert!(Path::new(&admission.selected_engine_path).is_absolute());
            require_sha256(&admission.selected_engine_sha256, "selected_engine_sha256").unwrap();
            assert!(admission.selected_engine_size_bytes > 0);
            assert_eq!(admission.topology_sha256, topology_sha256);
            assert_eq!(admission.compiled_policy_sha256, None);
            assert_eq!(admission.compiled_policy_build_path, None);
            assert_eq!(admission.source_parameter_topology_ifbox, 0);
            assert!(!admission.source_parameter_topology_box_dimensions_present);
            assert_eq!(admission.molecule_count, 1);
            serde_json::to_vec(&admission).unwrap();
        }
        #[cfg(feature = "v2_ignition")]
        {
            // A policy-pinned build never issues an admission for a policy
            // other than the compiled one: the synthetic policy is refused at
            // the digest gate before any authority cross-check runs.
            let (compiled_bytes, compiled_sha256) = compiled_policy_fixture();
            let refusal = preflight_legacy_ghost_topology_paths(
                &topology_path,
                &topology_sha256,
                &authority_path,
                &authority_sha256,
                &policy_path,
                &policy_sha256,
                &source_prmtop_path,
                &source_prmtop_sha256,
            )
            .unwrap_err()
            .to_string();
            assert!(refusal.contains("differs from compiled v2 policy digest"));
            assert!(refusal.contains(&format!("compiled={compiled_sha256}")));
            assert!(refusal.contains(&format!("runtime={policy_sha256}")));

            // The genuine pinned bytes pass the gate on content, not on path;
            // the synthetic sidecars are then refused on identity, so no
            // admission exists for artifacts the pinned policy does not bind.
            std::fs::write(&policy_path, &compiled_bytes).unwrap();
            let past_gate = preflight_legacy_ghost_topology_paths(
                &topology_path,
                &topology_sha256,
                &authority_path,
                &authority_sha256,
                &policy_path,
                &compiled_sha256,
                &source_prmtop_path,
                &source_prmtop_sha256,
            )
            .unwrap_err()
            .to_string();
            assert!(!past_gate.contains("differs from compiled v2 policy digest"));
            assert!(past_gate.contains("identity or shape mismatch"));

            // The sealed execution identity is enforced before the policy is
            // decoded, so a stale expected digest never reaches the gate.
            let sealed = preflight_legacy_ghost_topology_paths(
                &topology_path,
                &topology_sha256,
                &authority_path,
                &authority_sha256,
                &policy_path,
                &policy_sha256,
                &source_prmtop_path,
                &source_prmtop_sha256,
            )
            .unwrap_err()
            .to_string();
            assert!(sealed.contains("differs from sealed execution identity"));
        }

        let authority_link = root.path().join("authority-link.json");
        std::os::unix::fs::symlink(&authority_path, &authority_link).unwrap();
        let error = preflight_legacy_ghost_topology_paths(
            &topology_path,
            &topology_sha256,
            &authority_link,
            &authority_sha256,
            &policy_path,
            &policy_sha256,
            &source_prmtop_path,
            &source_prmtop_sha256,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("read legacy Ghost authority"));
    }

    #[test]
    fn checkpoint_reopen_restores_coordinate_keys_journals_and_work_counters() {
        let tmp = tempfile::tempdir().unwrap();
        let topology = test_topology();
        let ghost = test_ghost_context(0);
        let context = coordinate_context(&topology, &ghost);
        let mut writer =
            GhostCoordinateSegmentWriter::create(tmp.path(), "target", &topology, context.clone())
                .unwrap();
        let p0 = production_record(1000, 0, 11);
        let e0 = ghost_entry(&ghost, &p0, 0, 0);
        writer
            .append_ghost_segment(&e0, &p0, &[exact_frame(1000)], |_| Some(2))
            .unwrap();
        let boundary = writer.checkpoint(tmp.path()).unwrap();
        let p1 = production_record(1001, 1, 11);
        let e1 = ghost_entry(&ghost, &p1, 1, 1);
        writer
            .append_ghost_segment(&e1, &p1, &[exact_frame(1001)], |_| Some(2))
            .unwrap();
        drop(writer);
        let mut restored = GhostCoordinateSegmentWriter::reopen_checkpoint(
            tmp.path(),
            &boundary,
            &topology,
            &context,
        )
        .unwrap();
        assert_eq!(restored.manifest.persisted_coordinate_frames, 1);
        assert_eq!(restored.persistence_work, boundary.persistence_work);
        assert!(restored.root.join(".continuation_tails").is_dir());
        restored
            .append_ghost_segment(&e1, &p1, &[exact_frame(1001)], |_| Some(2))
            .unwrap();
        restored.complete(2).unwrap();
        assert_eq!(
            load_verified_ghost_coordinate_segments(tmp.path(), &topology)
                .unwrap()
                .persisted_records,
            2
        );
    }
}
