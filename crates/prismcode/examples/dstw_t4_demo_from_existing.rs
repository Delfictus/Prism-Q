//! DSTW-T4 POST_RUN_MANUAL demonstration exerciser (UNCOMMITTED demo harness).
//!
//! Drives the REAL `prismcode::dstw_state_graph::emit_dstw_t4_state_graph` code
//! path over an EXISTING completed engine run dir (output/5VEX_transition), then
//! POST-LABELS every emitted artifact runtime_origin=POST_RUN_MANUAL because THIS
//! exerciser is NOT the live prismos md-run. The PRISMOS_NATIVE
//! (RUNTIME_SUBPROCESS_ORCHESTRATED_BY_PRISMOS) vertices come from the GO-gated
//! live run; this only PROVES the machinery emits/seals/decodes byte-exact.
//!
//! No engine launch. Reads only existing artifacts. Run:
//!   CARGO_TARGET_DIR=/tmp/prism-cargo-target \
//!   cargo run --release -p prismcode --example dstw_t4_demo_from_existing

use prismcode::dstw_state_graph::{emit_dstw_t4_state_graph, DstwStateGraphInputs};
use prismcode::operand_ledger::{RuntimeOrigin, TSV_COLUMNS};
use prismcode::qnanl_artifact::{
    decode_packet_bytes, encode_bytes, QnanlArtifactFamily, QnanlEncodeOptions,
};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn sha256_hex(path: &Path) -> String {
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(path).expect("read for sha");
    let mut h = Sha256::new();
    h.update(&bytes);
    hex(&h.finalize())
}
fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn symlink_in(staging: &Path, real: &Path) {
    let dest = staging.join(real.file_name().unwrap());
    let _ = std::fs::remove_file(&dest);
    std::os::unix::fs::symlink(real, &dest).expect("symlink");
}

fn main() {
    // Roots come from the environment; there are no compiled-in host paths.
    let repo = PathBuf::from(std::env::var_os("PRISM_LEGACY_REPO_ROOT").expect("PRISM_LEGACY_REPO_ROOT must name the legacy Prism4D-bio checkout holding output/5VEX_transition"));
    let run_dir = repo.join("output/5VEX_transition");
    let topology = repo.join(
        "runs/glp1r_379_385_path_a_md_20260615T073919Z/01_receptor_prep/5VEX/03_topology/5VEX_chain_B_engine.topology.json",
    );
    let demo_root = PathBuf::from(std::env::var_os("PRISM_DSTW_DEMO_ROOT").expect("PRISM_DSTW_DEMO_ROOT must name the DSTW_T4 demo output root")).join("demo_from_existing");
    let staging = demo_root.join("_native_label_staging");
    std::fs::create_dir_all(&staging).unwrap();
    std::fs::create_dir_all(&demo_root).unwrap();

    // --- ground real engine artifacts so post_md tensors VALUE-verify -----------
    // first sorted *_stream0_spikes.bin + *_stream0_adaptive_dt.bin + binding_sites
    let pick = |suffix: &str| -> PathBuf {
        let mut v: Vec<PathBuf> = std::fs::read_dir(&run_dir)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.file_name().unwrap().to_str().unwrap().ends_with(suffix))
            .collect();
        v.sort();
        v.into_iter()
            .next()
            .unwrap_or_else(|| panic!("no {suffix} in run dir"))
    };
    let real_spikes = pick("_spikes.bin");
    let real_adaptive = pick("_adaptive_dt.bin");
    let real_binding = pick(".binding_sites.json");
    symlink_in(&staging, &real_spikes);
    symlink_in(&staging, &real_adaptive);
    symlink_in(&staging, &real_binding);

    // real opcode_counts from the existing prismcode_md_native_receipt.json
    let receipt_path = pick("_prismcode_md_native_receipt.json");
    let receipt_json: Value =
        serde_json::from_slice(&std::fs::read(&receipt_path).unwrap()).unwrap();
    let mut opcode_counts: BTreeMap<String, u64> = BTreeMap::new();
    if let Some(map) = receipt_json
        .get("opcode_counts")
        .and_then(|v| v.as_object())
    {
        for (k, v) in map {
            opcode_counts.insert(k.clone(), v.as_u64().unwrap_or(0));
        }
    }

    let inputs = DstwStateGraphInputs {
        output_dir: staging.clone(),
        topology_path: topology.clone(),
        target_id: "5VEX:chain_B".to_string(),
        ligand_mode: "apo".to_string(),
        detection_mode: "md_only_evidence".to_string(),
        site_ranker: "phase-manifold".to_string(),
        md_only_evidence: true,
        engine_path: repo.join("target/release/nhs_rt_full"),
        operator_version: "643bef00".to_string(),
        command_argv: vec![
            "nhs_rt_full".into(),
            "--md-only-evidence".into(),
            "--site-ranker".into(),
            "phase-manifold".into(),
            "--path-a-production-profile".into(),
        ],
        holo_topology: None,
        opcode_counts,
        dense_tensor_runtime_origin: RuntimeOrigin::PostRunDerived,
    };

    // --- DRIVE THE REAL EMITTER (native labels into staging) --------------------
    let report = emit_dstw_t4_state_graph(&inputs).expect("emit_dstw_t4_state_graph");
    assert!(
        report.all_decode_proofs_pass,
        "native staging decode proofs must pass"
    );

    // --- POST-LABEL every artifact POST_RUN_MANUAL into demo_from_existing ------
    let exerciser_note = "POST_RUN_MANUAL: produced by dstw_t4_demo_from_existing exerciser over \
        an EXISTING completed engine run (output/5VEX_transition). This is NOT the live prismos \
        md-run; it PROVES the dstw_state_graph machinery (emit -> QNANL encode_bytes -> byte-exact \
        decode). PRISMOS_NATIVE (RUNTIME_SUBPROCESS_ORCHESTRATED_BY_PRISMOS) emission is GO-gated.";

    let mut demo_records: Vec<Value> = Vec::new();
    let mut vertices = 0usize;
    let mut transitions = 0usize;

    for art in &report.emitted {
        let raw = std::fs::read(&art.json_path).unwrap();
        let mut v: Value = serde_json::from_slice(&raw).unwrap();
        let native_origin = art.runtime_origin.clone();
        let was_native = RuntimeOrigin::from_label(&native_origin)
            .map(|o| o.is_prismos_native())
            .unwrap_or(false);
        let obj = v.as_object_mut().unwrap();
        // Never falsely claim native: relabel would-be-native legs to POST_RUN_MANUAL;
        // BLOCKED / EXTERNAL_DERIVED stay (already honestly non-native).
        let effective_origin = if was_native {
            obj.insert("runtime_origin".into(), json!("POST_RUN_MANUAL"));
            obj.insert("runtime_native".into(), json!(false));
            "POST_RUN_MANUAL"
        } else {
            native_origin.as_str()
        };
        obj.insert("demonstration".into(), json!(true));
        obj.insert("exerciser_runtime_origin".into(), json!("POST_RUN_MANUAL"));
        obj.insert(
            "native_emission_origin_if_live_run".into(),
            json!(native_origin),
        );
        obj.insert("exerciser_note".into(), json!(exerciser_note));

        let basename = art
            .json_path
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        let out_json = demo_root.join(&basename);
        let pretty = serde_json::to_string_pretty(&v).unwrap();
        let relabeled = pretty.into_bytes();
        std::fs::write(&out_json, &relabeled).unwrap();

        // re-seal the relabeled bytes POST_RUN_MANUAL
        let packet = PathBuf::from(format!("{}.prismqnanl", out_json.display()));
        let receipt = PathBuf::from(format!("{}.prismqnanl.receipt.json", out_json.display()));
        let rec = encode_bytes(
            &relabeled,
            &packet,
            &receipt,
            QnanlEncodeOptions::new(
                basename.clone(),
                QnanlArtifactFamily::JsonTelemetry,
                Some(out_json.clone()),
                RuntimeOrigin::PostRunManual,
                "dstw_t4_demo_from_existing::post_run_reseal",
            ),
        )
        .expect("encode_bytes");
        std::fs::write(&receipt, rec.to_json_pretty()).unwrap();

        // executed byte-exact decode proof over the RELABELED bytes
        let decoded = decode_packet_bytes(&std::fs::read(&packet).unwrap()).expect("decode");
        let byte_exact = decoded == relabeled;
        assert!(byte_exact, "decode byte-exact FAILED for {basename}");
        let proof = json!({
            "schema_kind": "dstw_t4_decode_proof",
            "artifact": basename,
            "runtime_origin": "POST_RUN_MANUAL",
            "runtime_native": false,
            "native_emission_origin_if_live_run": native_origin,
            "packet_path": packet.display().to_string(),
            "raw_bytes": relabeled.len(),
            "decoded_bytes": decoded.len(),
            "raw_hash_blake3": rec.raw_hash_blake3,
            "decode_byte_exact": byte_exact
        });
        std::fs::write(
            demo_root.join(format!("{basename}.decode_proof.json")),
            serde_json::to_string_pretty(&proof).unwrap(),
        )
        .unwrap();

        if art.kind.contains("transition") {
            transitions += 1;
        } else if art.kind.contains("vertex") {
            vertices += 1;
        }
        demo_records.push(json!({
            "artifact": basename,
            "kind": art.kind,
            "effective_runtime_origin": effective_origin,
            "native_emission_origin_if_live_run": native_origin,
            "decode_byte_exact": byte_exact
        }));
    }

    // --- relabel operand ledger (runtime_origin column) -> demo -----------------
    let led_raw = std::fs::read_to_string(&report.operand_ledger_path).unwrap();
    let ro_idx = TSV_COLUMNS
        .iter()
        .position(|c| *c == "runtime_origin")
        .unwrap();
    let rn_idx = TSV_COLUMNS.iter().position(|c| *c == "runtime_native");
    let mut out_lines: Vec<String> = Vec::new();
    let mut seen_header = false;
    for line in led_raw.lines() {
        if line.starts_with('#') {
            out_lines.push(line.to_string());
            continue;
        }
        if !seen_header {
            seen_header = true; // column header row
            out_lines.push(line.to_string());
            continue;
        }
        let mut cells: Vec<String> = line.split('\t').map(|s| s.to_string()).collect();
        if let Some(cell) = cells.get_mut(ro_idx) {
            if cell == "RUNTIME_SUBPROCESS_ORCHESTRATED_BY_PRISMOS" || cell == "RUNTIME_INLINE" {
                *cell = "POST_RUN_MANUAL".to_string();
                if let Some(ri) = rn_idx {
                    if let Some(nc) = cells.get_mut(ri) {
                        *nc = "false".to_string();
                    }
                }
            }
        }
        out_lines.push(cells.join("\t"));
    }
    std::fs::write(
        demo_root.join("dstw_t4_operand_ledger.tsv"),
        out_lines.join("\n") + "\n",
    )
    .unwrap();

    // --- VALUE-verify a sample operand: post_md spike sha256 == real file -------
    let post_md: Value = serde_json::from_slice(
        &std::fs::read(demo_root.join("post_md_state_vertex.json")).unwrap(),
    )
    .unwrap();
    let claimed_sha = post_md["spike_sensor_tensor"]["value"]["sha256"]
        .as_str()
        .unwrap_or("")
        .to_string();
    let real_sha = sha256_hex(&real_spikes);
    let value_verified = claimed_sha == real_sha && !claimed_sha.is_empty();
    assert!(
        value_verified,
        "spike operand VALUE-verify FAILED: claimed {claimed_sha} real {real_sha}"
    );

    // post_md packet decode byte-exact (already asserted; re-prove explicitly)
    let pm_packet = demo_root.join("post_md_state_vertex.json.prismqnanl");
    let pm_decoded = decode_packet_bytes(&std::fs::read(&pm_packet).unwrap()).unwrap();
    let pm_raw = std::fs::read(demo_root.join("post_md_state_vertex.json")).unwrap();
    let qnanl_decode_proof = pm_decoded == pm_raw;

    let manifest = json!({
        "schema_kind": "dstw_t4_demo_from_existing_manifest",
        "exerciser_runtime_origin": "POST_RUN_MANUAL",
        "is_prismos_native": false,
        "note": exerciser_note,
        "source_run_dir": run_dir.display().to_string(),
        "topology": topology.display().to_string(),
        "topology_sha256": sha256_hex(&topology),
        "grounded_engine_artifacts": {
            "spikes_bin": real_spikes.display().to_string(),
            "spikes_sha256": real_sha,
            "adaptive_dt_bin": real_adaptive.display().to_string(),
            "binding_sites_json": real_binding.display().to_string(),
            "opcode_counts_source": receipt_path.display().to_string()
        },
        "vertices_emitted": vertices,
        "transitions_emitted": transitions,
        "operand_records": report.operand_record_count,
        "sample_operand_value_verified": {
            "field": "post_md_state_vertex.spike_sensor_tensor.value.sha256",
            "claimed_sha256": claimed_sha,
            "real_file_sha256": real_sha,
            "match": value_verified
        },
        "qnanl_decode_proof_post_md_byte_exact": qnanl_decode_proof,
        "all_decode_proofs_pass": true,
        "artifacts": demo_records
    });
    std::fs::write(
        demo_root.join("demo_manifest.json"),
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();

    println!("DSTW_T4_DEMO_OK vertices={vertices} transitions={transitions} operand_records={} value_verified={value_verified} qnanl_decode_proof={qnanl_decode_proof}", report.operand_record_count);
    println!("demo_dir={}", demo_root.display());
}
