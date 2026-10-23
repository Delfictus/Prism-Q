//! Read complete native checkpoint payloads and compare two accepted endpoints.
//! No CUDA context, propagation, checkpoint mutation, or relaxed tolerance.
use anyhow::{ensure, Context, Result};
use prism_nhs::scientific_checkpoint::CheckpointStore;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, io::Write, path::PathBuf};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().collect();
    ensure!(args.len() == 4, "usage: compare_native_checkpoints REFERENCE_STORE RESUMED_STORE NEW_JSON");
    let left = fs::canonicalize(PathBuf::from(&args[1]))?;
    let right = fs::canonicalize(PathBuf::from(&args[2]))?;
    let output = PathBuf::from(&args[3]);
    ensure!(left != right && !output.exists(), "two distinct native stores and a fresh output required");
    for root in [&left, &right] {
        for name in ["HEAD", "LOCK"] { ensure!(root.join(name).is_file(), "native scientific store absent"); }
        for name in ["objects", "manifests", "encoded_objects"] { ensure!(root.join(name).is_dir(), "native store layout absent"); }
    }
    let heads_before = [fs::read(left.join("HEAD"))?, fs::read(right.join("HEAD"))?];
    let stores = [CheckpointStore::create(&left, 64 << 30)?, CheckpointStore::create(&right, 64 << 30)?];
    let (lh, li) = stores[0].identity_at_head()?.context("reference HEAD absent")?;
    let (rh, ri) = stores[1].identity_at_head()?.context("resumed HEAD absent")?;
    let mut left_identity = li.clone(); left_identity.run_id = "<independent-process-run>".into();
    let mut right_identity = ri.clone(); right_identity.run_id = left_identity.run_id.clone();
    ensure!(left_identity == right_identity, "model/build/input/protocol/ABI identity differs");
    let indexes = [stores[0].validate_generation(&lh, &li)?, stores[1].validate_generation(&rh, &ri)?];
    indexes[0].closure().validate()?; indexes[1].closure().validate()?;
    ensure!(indexes[0].closure() == indexes[1].closure(), "complete native owner/field contract differs");
    ensure!(li.stream_count == 8 && indexes[0].boundaries().len() == 8, "original eight-owner population required");
    let mut boundary_rows = Vec::new();
    let mut physical_boundaries_equal = true;
    for (a, b) in indexes[0].boundaries().iter().zip(indexes[1].boundaries()) {
        let mut normalized_a = a.clone(); let mut normalized_b = b.clone();
        normalized_a.committed_output_sequence = 0; normalized_b.committed_output_sequence = 0;
        let equal = normalized_a == normalized_b;
        physical_boundaries_equal &= equal;
        boundary_rows.push(json!({"reference":a,"resumed":b,"physical_boundary_equal":equal,
            "output_sequence_comparison_role":"durable publication provenance; separately retained, not a physical clock"}));
    }
    let required_names = ["d_positions", "d_velocities", "d_forces", "d_rng_states", "d_protocol_state",
        "d_observer_physical_clock_state", "d_observer_physical_time_records", "temp_protocol", "gamma_base"];
    let mut required_seen: BTreeMap<String, u32> = required_names.iter().map(|s| (s.to_string(), 0)).collect();
    let mut required_equal = true; let mut all_equal = true;
    let mut rows = Vec::new(); let mut compared_bytes = 0u64;
    for contract in &indexes[0].closure().fields {
        let a = stores[0].read_generation_field(&indexes[0], &contract.owner, &contract.name, 64 << 30)?;
        let b = stores[1].read_generation_field(&indexes[1], &contract.owner, &contract.name, 64 << 30)?;
        let equal = a.bytes == b.bytes;
        all_equal &= equal;
        let required = required_seen.get_mut(&contract.name);
        if let Some(count) = required { *count += 1; required_equal &= equal && a.bytes.is_some(); }
        let ah = a.bytes.as_ref().map(|v| format!("{:x}", Sha256::digest(v)));
        let bh = b.bytes.as_ref().map(|v| format!("{:x}", Sha256::digest(v)));
        compared_bytes += a.bytes.as_ref().map_or(0, |v| v.len() as u64);
        rows.push(json!({"owner":contract.owner,"name":contract.name,"encoding":contract.encoding,
            "reference_bytes":a.bytes.as_ref().map(|v|v.len()),"resumed_bytes":b.bytes.as_ref().map(|v|v.len()),
            "reference_sha256":ah,"resumed_sha256":bh,"byte_equal":equal,
            "required_physical_field":required_names.contains(&contract.name.as_str())}));
    }
    ensure!(required_seen.values().all(|n|*n == 8), "required physical field population is not all eight owners");
    ensure!(fs::read(left.join("HEAD"))? == heads_before[0] && fs::read(right.join("HEAD"))? == heads_before[1],
        "source HEAD changed during complete comparison");
    let pass = all_equal && required_equal && physical_boundaries_equal;
    let result = json!({"schema":"prism.native_checkpoint_complete_comparison.v1",
        "status":if pass{"IMPLEMENTATION_VERIFIED"}else{"IMPLEMENTATION_PARTIAL"},
        "scope":"complete actual native checkpoint payload comparison at matched physical endpoint",
        "gpu_execution":false,"md_executed":false,"source_HEADs_unchanged":true,
        "reference_store":left,"resumed_store":right,"reference_head":lh,"resumed_head":rh,
        "reference_identity":li,"resumed_identity":ri,"boundaries":boundary_rows,
        "physical_boundaries_bitwise_equal":physical_boundaries_equal,
        "required_physical_fields_bitwise_equal":required_equal,"required_field_population":required_seen,
        "all_captured_field_bytes_equal":all_equal,"field_count":rows.len(),"payload_bytes_compared":compared_bytes,
        "fields":rows,"restart_equivalence_qualified":pass,
        "comparison_policy":"zero numerical tolerance; every captured payload checked. Any unmatched payload prevents full restart qualification until its exact semantics are resolved; run_id and output publication sequence are retained operational identities."});
    let mut file = fs::OpenOptions::new().write(true).create_new(true).open(&output)?;
    serde_json::to_writer_pretty(&mut file, &result)?; file.write_all(b"\n")?; file.sync_all()?;
    if !pass { std::process::exit(4); }
    Ok(())
}
