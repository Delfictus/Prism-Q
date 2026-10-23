//! Explicit import of a prepared harmonic convention for an unchanged input.
//! The authority is a separately prepared, hash-bound provider export with
//! exactly the same ordered atom and harmonic populations. Historical producer
//! verification remains unknown; no physical arrays are imported.
use super::{hash, serr, PinnedFile};
use crate::input::{HarmonicEnergyConvention, PrismPrepTopology};
use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};

const ATOM_IDENTITY: &[&str] = &[
    "n_atoms", "n_residues", "n_chains", "elements", "atom_names",
    "residue_names", "residue_ids", "chain_ids", "ca_indices",
    "residue_to_atom_indices", "pdb_atom_serial_to_topo_index",
];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Resource {
    resource: String,
    file: PinnedFile,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Binding {
    schema: String,
    target_topology: PinnedFile,
    authority_topology: PinnedFile,
    preparation_receipt: PinnedFile,
    preparation_start: PinnedFile,
    preparation_execution: PinnedFile,
    producer: PinnedFile,
    source_pdb: PinnedFile,
    forcefield_resources: Vec<Resource>,
    bonds_f64_rows_sha256: String,
    angles_f64_rows_sha256: String,
    historical_producer_verification: String,
}

fn harmonic_rows(topology: &PrismPrepTopology, angles: bool) -> Vec<u8> {
    let mut bytes = if angles { b"angles\0".to_vec() } else { b"bonds\0".to_vec() };
    let count = if angles { topology.angles.len() } else { topology.bonds.len() };
    bytes.extend_from_slice(&(count as u64).to_le_bytes());
    for i in 0..count {
        bytes.extend_from_slice(&(i as u64).to_le_bytes());
        let (indices, values) = if angles {
            let a = &topology.angles[i];
            (vec![a.i, a.j, a.k_idx], [a.theta0, a.force_k])
        } else {
            let b = &topology.bonds[i];
            (vec![b.i, b.j], [b.r0, b.k])
        };
        for index in indices { bytes.extend_from_slice(&(index as u64).to_le_bytes()); }
        for value in values { bytes.extend_from_slice(&value.to_bits().to_le_bytes()); }
    }
    bytes
}

fn output_matches(outputs: &Value, pin: &PinnedFile, length: usize) -> bool {
    outputs.as_array().is_some_and(|rows| rows.iter().any(|row| {
        row["path"].as_str() == pin.path.to_str()
            && row["sha256"].as_str() == Some(pin.sha256.as_str())
            && row["bytes"].as_u64() == Some(length as u64)
    }))
}

pub(super) fn bind(
    topology: &mut PrismPrepTopology,
    target: &Value,
    target_pin: &PinnedFile,
    binding_bytes: &[u8],
) -> Result<Value> {
    let b: Binding = serde_json::from_slice(binding_bytes)?;
    ensure!(b.schema == "prism.harmonic_source_binding.v1"
        && b.historical_producer_verification == "UNKNOWN",
        "unsupported harmonic binding scope");
    ensure!(topology.harmonic_parameter_contract.is_none(),
        "harmonic binding cannot replace an embedded contract");
    ensure!(b.target_topology.path == target_pin.path
        && b.target_topology.sha256 == target_pin.sha256,
        "harmonic binding targets a different input");
    let target_bytes = b.target_topology.read()?;
    ensure!(serde_json::from_slice::<Value>(&target_bytes)? == *target,
        "harmonic binding target changed after input resolution");
    let authority_bytes = b.authority_topology.read()?;
    let authority_json: Value = serde_json::from_slice(&authority_bytes)?;
    let authority: PrismPrepTopology = serde_json::from_slice(&authority_bytes)?;
    let lowering = serr(authority.lower_harmonic_parameters(HarmonicEnergyConvention::AmberNoHalfEnergy))?;
    for key in ATOM_IDENTITY {
        ensure!(target.get(*key).is_some() && target.get(*key) == authority_json.get(*key),
            "harmonic authority atom identity differs at {key}");
    }
    for (angles, expected) in [(false, &b.bonds_f64_rows_sha256), (true, &b.angles_f64_rows_sha256)] {
        let target_rows = harmonic_rows(topology, angles);
        ensure!(target_rows == harmonic_rows(&authority, angles)
            && hash(&target_rows) == *expected,
            "harmonic authority ordered {} index/f64 population differs", if angles { "angle" } else { "bond" });
    }
    let receipt_bytes = b.preparation_receipt.read()?;
    let receipt: Value = serde_json::from_slice(&receipt_bytes)?;
    ensure!(receipt["schema_kind"] == "prismos_topology_preparation_receipt"
        && receipt["schema_version"] == 2 && receipt["status"] == "PASS",
        "harmonic authority preparation receipt is not accepted");
    ensure!(receipt["artifacts"].as_array().is_some_and(|rows| rows.iter().any(|row| {
        row["role"] == "topology" && row["path"].as_str() == b.authority_topology.path.to_str()
            && row["sha256"].as_str() == Some(b.authority_topology.sha256.as_str())
            && row["size_bytes"].as_u64() == Some(authority_bytes.len() as u64)
    })), "harmonic authority preparation receipt targets different bytes");
    let execution: Value = serde_json::from_slice(&b.preparation_execution.read()?)?;
    let start: Value = serde_json::from_slice(&b.preparation_start.read()?)?;
    ensure!(execution["event"] == "CHILD_FINISHED" && execution["exit_code"] == 0
        && execution["pid"].as_u64().is_some_and(|pid| pid > 0)
        && execution["stage"].as_u64().is_some()
        && execution["started_utc"].as_str().is_some_and(|s| !s.is_empty())
        && start["event"] == "CHILD_STARTED" && start["pid"] == execution["pid"]
        && start["stage"] == execution["stage"] && start["started_utc"] == execution["started_utc"]
        && output_matches(&execution["outputs"], &b.authority_topology, authority_bytes.len())
        && output_matches(&execution["outputs"], &b.preparation_receipt, receipt_bytes.len()),
        "harmonic authority actual preparation execution does not bind its outputs");
    b.producer.read()?;
    b.source_pdb.read()?;
    let provenance = &authority_json["topology_provenance"];
    ensure!(provenance["producer"].as_str() == b.producer.path.to_str()
        && provenance["producer_sha256"].as_str() == Some(b.producer.sha256.as_str())
        && provenance["source_pdb"].as_str() == b.source_pdb.path.to_str()
        && provenance["source_pdb_sha256"].as_str() == Some(b.source_pdb.sha256.as_str())
        && receipt["topology_provenance_sha256"] == provenance["contract_sha256"]
        && receipt["scientific_parameter_topology_sha256"]
            == authority_json["scientific_parameter_topology_contract"]["scientific_parameter_topology_sha256"]
        && start["argv"].as_array().is_some_and(|argv| {
            argv.windows(3).any(|args| args[0].as_str() == b.producer.path.to_str()
                && args[1] == provenance["source_pdb"]
                && args[2].as_str() == b.authority_topology.path.to_str())
        }), "harmonic authority producer/actual command lineage differs");
    let resources = provenance["forcefield_resources"]["resources"].as_array()
        .context("harmonic authority forcefield resource closure absent")?;
    ensure!(!resources.is_empty() && resources.len() == b.forcefield_resources.len(),
        "harmonic authority forcefield population mismatch");
    let root = std::path::Path::new(provenance["forcefield_resources"]["openmm_data_root"].as_str()
        .context("harmonic authority forcefield root absent")?);
    let mut names = std::collections::BTreeSet::new();
    for (actual, bound) in resources.iter().zip(&b.forcefield_resources) {
        ensure!(names.insert(bound.resource.as_str())
            && actual["resource"].as_str() == Some(bound.resource.as_str())
            && root.join(&bound.resource) == bound.file.path
            && actual["sha256"].as_str() == Some(bound.file.sha256.as_str())
            && actual["size_bytes"].as_u64() == Some(bound.file.read()?.len() as u64),
            "harmonic authority forcefield resource identity differs");
    }
    let contract = authority.harmonic_parameter_contract.clone()
        .context("harmonic authority lacks source contract")?;
    let version = provenance["openmm_version"].as_str().context("harmonic provider version absent")?;
    ensure!(contract.source_provider == format!("OpenMM/{version}:HarmonicBondForce+HarmonicAngleForce")
        && contract.bond_convention == HarmonicEnergyConvention::OpenmmFullKHalfEnergy
        && contract.angle_convention == HarmonicEnergyConvention::OpenmmFullKHalfEnergy,
        "harmonic binding requires the explicit prepared OpenMM full-k/half-energy contract");
    // Validate every precondition on a private candidate. Failure leaves the
    // caller's topology unmodified, including its absent contract.
    let mut candidate = topology.clone();
    candidate.harmonic_parameter_contract = Some(contract.clone());
    let target_lowering = serr(candidate.lower_harmonic_parameters(HarmonicEnergyConvention::AmberNoHalfEnergy))?;
    ensure!(target_lowering.accounting["parsed_raw_f64_sha256"] == lowering.accounting["parsed_raw_f64_sha256"]
        && target_lowering.accounting["lowered_coefficient_f32_sha256"] == lowering.accounting["lowered_coefficient_f32_sha256"],
        "harmonic target/provider lowering differs");
    topology.harmonic_parameter_contract = Some(contract);
    Ok(json!({
        "schema":"prism.harmonic_source_binding.execution.v1",
        "binding_sha256":hash(binding_bytes),
        "binding":serde_json::from_slice::<Value>(binding_bytes)?,
        "historical_producer_verification":"UNKNOWN",
        "current_semantics":"exact ordered atom and harmonic index/f64 equality with retained provider export",
        "original_topology_bytes_mutated":false,
        "imported_fields":["harmonic_parameter_contract"],
        "physical_arrays_imported":[],
        "lowering":target_lowering.accounting,
    }))
}
