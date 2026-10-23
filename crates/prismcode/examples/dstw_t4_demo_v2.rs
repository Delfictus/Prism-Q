//! DSTW-T4 Phase 2C POST_RUN_DERIVED demonstration exerciser (committed demo harness).
//!
//! Phase 2C extends the Phase 2B `dstw_t4_demo_from_existing` exerciser. It drives the
//! REAL `prismcode::dstw_state_graph::emit_dstw_t4_state_graph` (which now also emits the
//! nine PRISM-native merge-readiness ledgers, the POST_RUN_DERIVED dense-tensor semantic
//! summaries, and the falsifiable transitions/ValidationState) over an EXISTING completed
//! engine run dir, then:
//!
//!   1. Runs a COUNTERFACTUAL CONTROL across two DISTINCT existing streams of the run
//!      (stream0 vs stream1) and records whether the post-MD state tensors actually DIFFER.
//!   2. Re-labels EVERY would-be-native artifact `runtime_origin=POST_RUN_DERIVED` (NOT
//!      native — this is NOT the live `prismos md-run`; the orchestrated NATIVE legs are
//!      GO-gated) and re-seals into `DSTW_T4/demo_v2/`, attaching a PROCESS-PROVEN
//!      `runtime_provenance` block to every record (run_pid + packaged-binary sha256 +
//!      emitted-artifact sha256), recording `native_emission_origin_if_live_run`.
//!   3. Re-derives the nine merge-readiness ledgers over the demo_v2 copies via the public
//!      `emit_prism_native_merge_manifest` (orphan check + process-proven RUNTIME_EMISSION
//!      ledger over the demo copies).
//!   4. Exercises the lifecycle-orchestrated `RuntimeQnanlStreamWriter` over the relabeled
//!      vertex/transition bytes (decode-proof byte-exact) into `_orchestrated_writer_proof/`,
//!      honestly recording that its writer-stamped label is the would-be-live native label.
//!
//! Everything under `demo_v2/` is `runtime_origin=POST_RUN_DERIVED` (or the honest
//! non-native origin BLOCKED / EXTERNAL_DERIVED the emitter already stamped). NO engine
//! launch. Reads only existing artifacts. Run:
//!   CARGO_TARGET_DIR=/tmp/prism-cargo-target \
//!   cargo run --release -p prismcode --example dstw_t4_demo_v2

use prismcode::dstw_state_graph::{
    emit_dstw_t4_state_graph, DstwEmittedArtifact, DstwStateGraphInputs, DstwStateGraphReport,
};
use prismcode::merge_manifest::emit_prism_native_merge_manifest;
use prismcode::operand_ledger::{OperandLedger, RuntimeOrigin, TSV_COLUMNS};
use prismcode::qnanl_artifact::{
    decode_packet_bytes, encode_bytes, QnanlArtifactFamily, QnanlEncodeOptions,
    RuntimeQnanlChunkContext, RuntimeQnanlStreamWriter,
};
use prismcode::runtime_provenance::RuntimeProvenance;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn sha256_hex(path: &Path) -> String {
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(path).expect("read for sha");
    let mut h = Sha256::new();
    h.update(&bytes);
    h.finalize().iter().map(|x| format!("{x:02x}")).collect()
}

/// First file in `dir` whose name CONTAINS `needle`, sorted for determinism.
fn find_containing(dir: &Path, needle: &str) -> Option<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.contains(needle))
                .unwrap_or(false)
        })
        .collect();
    v.sort();
    v.into_iter().next()
}

fn symlink_in(staging: &Path, real: &Path, as_name: &str) {
    let dest = staging.join(as_name);
    let _ = std::fs::remove_file(&dest);
    std::os::unix::fs::symlink(real, &dest).expect("symlink");
}

fn read_opcode_counts(receipt_path: &Path) -> BTreeMap<String, u64> {
    let mut m = BTreeMap::new();
    if let Ok(bytes) = std::fs::read(receipt_path) {
        if let Ok(j) = serde_json::from_slice::<Value>(&bytes) {
            if let Some(map) = j.get("opcode_counts").and_then(|v| v.as_object()) {
                for (k, v) in map {
                    m.insert(k.clone(), v.as_u64().unwrap_or(0));
                }
            }
        }
    }
    m
}

/// Stage one stream's real engine artifacts into `staging` and drive the REAL emitter.
/// `stream_spike` is the EXACT spikes filename for the chosen stream (counterfactual leg).
fn stage_and_emit(
    run_dir: &Path,
    topology: &Path,
    engine_path: &Path,
    staging: &Path,
    stream_spike: &Path,
    stream_adaptive: &Path,
    binding: &Path,
    receipt: &Path,
) -> DstwStateGraphReport {
    std::fs::create_dir_all(staging).unwrap();
    // Re-name the staged spikes/adaptive to the generic stream0 name the emitter probes
    // (find_artifact picks the first sorted `_spikes.bin`); only ONE is present in staging.
    symlink_in(
        staging,
        stream_spike,
        "5VEX_chain_B_engine_stream0_spikes.bin",
    );
    symlink_in(
        staging,
        stream_adaptive,
        "5VEX_chain_B_engine_stream0_adaptive_dt.bin",
    );
    symlink_in(staging, binding, "5VEX_chain_B_engine.binding_sites.json");

    let inputs = DstwStateGraphInputs {
        output_dir: staging.to_path_buf(),
        topology_path: topology.to_path_buf(),
        target_id: "5VEX:chain_B".to_string(),
        ligand_mode: "apo".to_string(),
        detection_mode: "md_only_evidence".to_string(),
        site_ranker: "phase-manifold".to_string(),
        md_only_evidence: true,
        engine_path: engine_path.to_path_buf(),
        operator_version: "e065f65e".to_string(),
        command_argv: vec![
            "nhs_rt_full".into(),
            "--md-only-evidence".into(),
            "--site-ranker".into(),
            "phase-manifold".into(),
            "--path-a-production-profile".into(),
        ],
        holo_topology: None,
        opcode_counts: read_opcode_counts(receipt),
        dense_tensor_runtime_origin: RuntimeOrigin::PostRunDerived,
    };
    let _ = run_dir; // run_dir is the source of the staged real artifacts (documented in manifest)
    emit_dstw_t4_state_graph(&inputs).expect("emit_dstw_t4_state_graph")
}

/// Read `post_md_state_vertex.json -> spike_sensor_tensor.value.sha256` from a staging dir.
fn post_md_spike_sha(staging: &Path) -> String {
    let v: Value =
        serde_json::from_slice(&std::fs::read(staging.join("post_md_state_vertex.json")).unwrap())
            .unwrap();
    v["spike_sensor_tensor"]["value"]["sha256"]
        .as_str()
        .unwrap_or("")
        .to_string()
}

fn main() {
    // Roots come from the environment; there are no compiled-in host paths.
    let repo = PathBuf::from(std::env::var_os("PRISM_LEGACY_REPO_ROOT").expect("PRISM_LEGACY_REPO_ROOT must name the legacy Prism4D-bio checkout holding output/5VEX_transition"));
    let run_dir = repo.join("output/5VEX_transition");
    let topology = repo.join(
        "runs/glp1r_379_385_path_a_md_20260615T073919Z/01_receptor_prep/5VEX/03_topology/5VEX_chain_B_engine.topology.json",
    );
    let engine_path = repo.join("target/release/nhs_rt_full");
    let demo_root = PathBuf::from(std::env::var_os("PRISM_DSTW_DEMO_ROOT").expect("PRISM_DSTW_DEMO_ROOT must name the DSTW_T4 demo output root")).join("demo_v2");
    let _ = std::fs::remove_dir_all(&demo_root);
    std::fs::create_dir_all(&demo_root).unwrap();
    let stage_a = demo_root.join("_staging_stream0");
    let stage_b = demo_root.join("_staging_stream1");

    // ---- pick the two DISTINCT counterfactual streams (exact filenames) --------
    let spikes_a = find_containing(&run_dir, "_stream0_spikes.bin").expect("stream0 spikes");
    let spikes_b = find_containing(&run_dir, "_stream1_spikes.bin").expect("stream1 spikes");
    let adaptive_a =
        find_containing(&run_dir, "_stream0_adaptive_dt.bin").expect("stream0 adaptive_dt");
    let adaptive_b =
        find_containing(&run_dir, "_stream1_adaptive_dt.bin").expect("stream1 adaptive_dt");
    let binding = find_containing(&run_dir, ".binding_sites.json").expect("binding_sites");
    let receipt_a =
        find_containing(&run_dir, "_stream00_prismcode_md_native_receipt.json").expect("receipt00");
    let receipt_b =
        find_containing(&run_dir, "_stream01_prismcode_md_native_receipt.json").expect("receipt01");

    // ---- COUNTERFACTUAL CONTROL: emit over BOTH streams, compare state tensors ---
    let report_a = stage_and_emit(
        &run_dir,
        &topology,
        &engine_path,
        &stage_a,
        &spikes_a,
        &adaptive_a,
        &binding,
        &receipt_a,
    );
    let report_b = stage_and_emit(
        &run_dir,
        &topology,
        &engine_path,
        &stage_b,
        &spikes_b,
        &adaptive_b,
        &binding,
        &receipt_b,
    );
    let cf_sha_a = post_md_spike_sha(&stage_a);
    let cf_sha_b = post_md_spike_sha(&stage_b);
    let real_sha_a = sha256_hex(&spikes_a);
    let real_sha_b = sha256_hex(&spikes_b);
    let tensors_differ = cf_sha_a != cf_sha_b && !cf_sha_a.is_empty() && !cf_sha_b.is_empty();
    // The state tensor sha256 must equal the real per-stream spikes.bin (VALUE-verified).
    assert_eq!(
        cf_sha_a, real_sha_a,
        "stream0 state tensor must VALUE-verify"
    );
    assert_eq!(
        cf_sha_b, real_sha_b,
        "stream1 state tensor must VALUE-verify"
    );
    let counterfactual = json!({
        "schema_kind": "dstw_t4_counterfactual_control",
        "control_kind": "MULTI_STREAM_CONTRAST_SAME_RUN",
        "leg_a": {"stream": "stream0", "spikes": spikes_a.display().to_string(),
                  "state_tensor_sha256": cf_sha_a, "real_file_sha256": real_sha_a, "value_verified": cf_sha_a==real_sha_a},
        "leg_b": {"stream": "stream1", "spikes": spikes_b.display().to_string(),
                  "state_tensor_sha256": cf_sha_b, "real_file_sha256": real_sha_b, "value_verified": cf_sha_b==real_sha_b},
        "state_tensors_differ": tensors_differ,
        "control_meaningful": tensors_differ,
        "verdict": if tensors_differ { "MEANINGFUL_TENSORS_CHANGE_ACROSS_DISTINCT_STREAMS" } else { "DEGENERATE_TENSORS_IDENTICAL" },
        "ligand_contrast_counterfactual": "PENDING_ON_LIVE_DETECTION_OR_LIGAND_CONTRAST_RUN",
        "ligand_contrast_reason": "only apo md-only-evidence streams of a single run exist on disk; an apo-vs-holo (ligand) state contrast requires a live GO-gated detection/ligand-contrast prismos md-run. The multi-stream contrast PROVES the state-tensor pipeline is input-sensitive (distinct streams -> distinct sha256), but the scientific ligand counterfactual is not yet demonstrable POST_RUN."
    });
    std::fs::write(
        demo_root.join("COUNTERFACTUAL_CONTROL.json"),
        serde_json::to_string_pretty(&counterfactual).unwrap(),
    )
    .unwrap();

    // ---- PROCESS-PROVEN provenance for THIS exerciser process ------------------
    let provenance = RuntimeProvenance::capture(
        &demo_root,
        vec!["dstw_t4_demo_v2".into(), "--post-run-derived".into()],
    );
    std::fs::write(
        demo_root.join("runtime_provenance.json"),
        provenance.to_pretty_json(),
    )
    .unwrap();

    let exerciser_note = "POST_RUN_DERIVED: produced by the dstw_t4_demo_v2 exerciser over an \
        EXISTING completed engine run (output/5VEX_transition, stream0). This is NOT the live \
        prismos md-run; it PROVES the dstw_state_graph + merge-manifest machinery (emit -> QNANL \
        encode_bytes -> byte-exact decode -> 9 merge ledgers). PRISMOS_NATIVE \
        (RUNTIME_SUBPROCESS_ORCHESTRATED_BY_PRISMOS / RUNTIME_INLINE) emission is GO-gated.";

    // ---- relabel run A's emitted bundle POST_RUN_DERIVED into demo_v2 -----------
    let mut demo_emitted: Vec<DstwEmittedArtifact> = Vec::new();
    let mut demo_records: Vec<Value> = Vec::new();
    let mut all_provenance_present = true;
    for art in &report_a.emitted {
        let raw = std::fs::read(&art.json_path).unwrap();
        let mut v: Value = serde_json::from_slice(&raw).unwrap();
        let native_origin = art.runtime_origin.clone();
        let was_native = RuntimeOrigin::from_label(&native_origin)
            .map(|o| o.is_prismos_native())
            .unwrap_or(false);
        // PROCESS-PROVEN provenance block, computed over the staging source artifact.
        let prov_block = provenance
            .artifact_record(
                &art.json_path,
                "dstw_t4_demo_v2_relabel_from_staging",
                Vec::new(),
            )
            .to_json();
        if prov_block
            .get("run_pid")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            == 0
        {
            all_provenance_present = false;
        }
        let obj = v.as_object_mut().unwrap();
        let effective_origin = if was_native {
            obj.insert("runtime_origin".into(), json!("POST_RUN_DERIVED"));
            obj.insert("runtime_native".into(), json!(false));
            "POST_RUN_DERIVED"
        } else {
            native_origin.as_str()
        };
        obj.insert("demonstration".into(), json!(true));
        obj.insert("exerciser_runtime_origin".into(), json!("POST_RUN_DERIVED"));
        obj.insert(
            "native_emission_origin_if_live_run".into(),
            json!(native_origin),
        );
        obj.insert("exerciser_note".into(), json!(exerciser_note));
        obj.insert("runtime_provenance".into(), prov_block);

        let basename = art
            .json_path
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        let out_json = demo_root.join(&basename);
        let relabeled = serde_json::to_string_pretty(&v).unwrap().into_bytes();
        std::fs::write(&out_json, &relabeled).unwrap();

        // re-seal the relabeled bytes POST_RUN_DERIVED (SECTION_ZSTD body + NANL-Q4 seal)
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
                RuntimeOrigin::PostRunDerived,
                "dstw_t4_demo_v2::post_run_derived_reseal",
            ),
        )
        .expect("encode_bytes")
        .with_runtime_provenance(provenance.artifact_record(
            &out_json,
            "dstw_t4_demo_v2_seal",
            Vec::new(),
        ));
        std::fs::write(&receipt, rec.to_json_pretty()).unwrap();

        // executed byte-exact decode proof over the RELABELED bytes
        let decoded = decode_packet_bytes(&std::fs::read(&packet).unwrap()).expect("decode");
        let byte_exact = decoded == relabeled;
        assert!(byte_exact, "decode byte-exact FAILED for {basename}");
        std::fs::write(
            demo_root.join(format!("{basename}.decode_proof.json")),
            serde_json::to_string_pretty(&json!({
                "schema_kind": "dstw_t4_decode_proof",
                "artifact": basename,
                "runtime_origin": "POST_RUN_DERIVED",
                "runtime_native": false,
                "native_emission_origin_if_live_run": native_origin,
                "packet_path": packet.display().to_string(),
                "raw_bytes": relabeled.len(),
                "decoded_bytes": decoded.len(),
                "raw_hash_blake3": rec.raw_hash_blake3,
                "decode_byte_exact": byte_exact
            }))
            .unwrap(),
        )
        .unwrap();

        demo_emitted.push(DstwEmittedArtifact {
            kind: art.kind.clone(),
            json_path: out_json.clone(),
            qnanl_packet_path: packet,
            decode_proof_pass: byte_exact,
            runtime_origin: effective_origin.to_string(),
        });
        demo_records.push(json!({
            "artifact": basename,
            "kind": art.kind,
            "effective_runtime_origin": effective_origin,
            "native_emission_origin_if_live_run": native_origin,
            "decode_byte_exact": byte_exact
        }));
    }

    // ---- relabel the operand ledger + coverage (native -> POST_RUN_DERIVED) -----
    let relabel_tsv = |src: &Path, dst: &Path| {
        let raw = std::fs::read_to_string(src).unwrap();
        let ro_idx = TSV_COLUMNS.iter().position(|c| *c == "runtime_origin");
        let rn_idx = TSV_COLUMNS.iter().position(|c| *c == "runtime_native");
        let mut out: Vec<String> = Vec::new();
        let mut seen_header = false;
        for line in raw.lines() {
            if line.starts_with('#') {
                out.push(line.to_string());
                continue;
            }
            if !seen_header {
                seen_header = true;
                out.push(line.to_string());
                continue;
            }
            let mut cells: Vec<String> = line.split('\t').map(|s| s.to_string()).collect();
            if let Some(i) = ro_idx {
                if let Some(c) = cells.get_mut(i) {
                    if c == "RUNTIME_SUBPROCESS_ORCHESTRATED_BY_PRISMOS" || c == "RUNTIME_INLINE" {
                        *c = "POST_RUN_DERIVED".to_string();
                        if let Some(ri) = rn_idx {
                            if let Some(nc) = cells.get_mut(ri) {
                                *nc = "false".to_string();
                            }
                        }
                    }
                }
            }
            out.push(cells.join("\t"));
        }
        std::fs::write(dst, out.join("\n") + "\n").unwrap();
    };
    relabel_tsv(
        &report_a.operand_ledger_path,
        &demo_root.join("dstw_t4_operand_ledger.tsv"),
    );

    // ---- re-derive the NINE merge-readiness ledgers over the demo_v2 copies -----
    // process-proven by THIS exerciser process; orphan check over the demo copies.
    let empty_ledger = OperandLedger::new();
    let manifest = emit_prism_native_merge_manifest(
        &demo_emitted,
        &empty_ledger,
        &provenance,
        "5VEX:chain_B",
        &demo_root,
    )
    .expect("emit_prism_native_merge_manifest");
    // Authentic coverage: overwrite the empty re-emit with the relabeled real coverage.
    relabel_tsv(
        &stage_a.join("OPCODE_OPERAND_COVERAGE_LEDGER.tsv"),
        &demo_root.join("OPCODE_OPERAND_COVERAGE_LEDGER.tsv"),
    );

    // ---- ORCHESTRATED-WRITER QNANL proof (decode-proof byte-exact) -------------
    // Exercise the lifecycle-orchestrated RuntimeQnanlStreamWriter over the relabeled
    // vertex/transition bytes. Its writer-stamped label is the would-be-LIVE native
    // RUNTIME_SUBPROCESS_ORCHESTRATED_BY_PRISMOS; here the EXERCISER drove it POST_RUN,
    // so the demo-level origin remains POST_RUN_DERIVED (recorded in the proof note).
    let ow_dir = demo_root.join("_orchestrated_writer_proof");
    std::fs::create_dir_all(&ow_dir).unwrap();
    let mut writer = RuntimeQnanlStreamWriter::new(
        &ow_dir,
        "dstw_t4_state_stream",
        Some(sha256_hex(&topology)),
        Some(engine_path.display().to_string()),
    )
    .with_parent_state_vertex(
        report_a
            .emitted
            .iter()
            .find(|a| a.kind == "post_md_state_vertex")
            .and_then(|a| {
                serde_json::from_slice::<Value>(&std::fs::read(&a.json_path).ok()?)
                    .ok()?
                    .get("vertex_id")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            }),
        Some("post_md_state_vertex id unreadable".to_string()),
    );
    let mut ow_chunks = 0usize;
    let mut ow_all_decode = true;
    for kind in ["post_md_state_vertex", "md_state_transition"] {
        if let Some(art) = demo_emitted.iter().find(|a| a.kind == kind) {
            let raw = std::fs::read(&art.json_path).unwrap();
            let prov =
                provenance.artifact_record(&art.json_path, "orchestrated_writer_proof", Vec::new());
            let rcpt = writer
                .push_chunk_with_context(
                    &raw,
                    RuntimeQnanlChunkContext::new()
                        .with_source_label(format!("demo_v2:{kind}"))
                        .with_runtime_provenance(prov),
                )
                .expect("orchestrated push_chunk");
            ow_all_decode &= rcpt.roundtrip_verified;
            ow_chunks += 1;
        }
    }
    let ow_lineage = writer.finalize().expect("orchestrated finalize");

    // ---- TOKENIZER manifest (matched_fields gate is tokenizer-owned, run after) -
    std::fs::write(
        demo_root.join("tok_manifest.json"),
        serde_json::to_string(&json!([{
            "source_id": "dstw_t4_demo_v2_5vex",
            "kind": "DstwStateGraph",
            "dstw_dir": demo_root.display().to_string()
        }]))
        .unwrap(),
    )
    .unwrap();

    // ---- HARD VERIFICATION -----------------------------------------------------
    // orphan check: read the state-graph orphan_check record from the demo copies.
    let sg = std::fs::read_to_string(demo_root.join("PRISM_NATIVE_STATE_GRAPH.jsonl")).unwrap();
    let orphan_rec: Value = serde_json::from_str(sg.lines().last().unwrap()).unwrap();
    let orphan_count = orphan_rec["orphan_count"].as_u64().unwrap_or(u64::MAX);
    let total_nodes = orphan_rec["total_artifacts"].as_u64().unwrap_or(0);
    let mapped = total_nodes.saturating_sub(orphan_count);
    assert_eq!(orphan_count, 0, "demo_v2 must have zero orphans");
    assert_eq!(manifest.orphan_count() as u64, orphan_count);

    // operand value_proof: demo post_md spike sha256 == real stream0 spikes.bin
    let demo_post_sha = post_md_spike_sha(&demo_root);
    let operand_value_verified = demo_post_sha == real_sha_a && !demo_post_sha.is_empty();
    assert!(
        operand_value_verified,
        "post_md spike operand VALUE-verify failed"
    );

    // QNANL decode byte-exact (re-proven explicitly for post_md packet)
    let pm_raw = std::fs::read(demo_root.join("post_md_state_vertex.json")).unwrap();
    let pm_packet = std::fs::read(demo_root.join("post_md_state_vertex.json.prismqnanl")).unwrap();
    let qnanl_decode_byte_exact = decode_packet_bytes(&pm_packet).unwrap() == pm_raw;
    assert!(qnanl_decode_byte_exact);

    // runtime provenance present + non-null on EVERY demo record JSON
    for art in &demo_emitted {
        let v: Value = serde_json::from_slice(&std::fs::read(&art.json_path).unwrap()).unwrap();
        let rp = &v["runtime_provenance"];
        assert!(
            rp.is_object() && rp["run_pid"].as_u64().unwrap_or(0) > 0,
            "runtime_provenance missing/non-process-proven on {}",
            art.kind
        );
    }

    let manifest_doc = json!({
        "schema_kind": "dstw_t4_demo_v2_manifest",
        "exerciser_runtime_origin": "POST_RUN_DERIVED",
        "is_prismos_native": false,
        "note": exerciser_note,
        "source_run_dir": run_dir.display().to_string(),
        "topology": topology.display().to_string(),
        "topology_sha256": sha256_hex(&topology),
        "counterfactual_control": counterfactual,
        "process_proven_provenance": {
            "run_pid": provenance.pid,
            "packaged_binary_path": provenance.packaged_binary_path,
            "packaged_binary_sha256": provenance.packaged_binary_sha256,
            "all_records_provenance_present": all_provenance_present
        },
        "artifacts_relabeled": demo_records.len(),
        "dense_tensor_summary_count": report_a.dense_tensor_summary_count,
        "merge_ledgers_emitted": manifest.output_paths.len(),
        "merge_ledger_paths": manifest.output_paths.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
        "orphan_check": {"total": total_nodes, "mapped": mapped, "orphaned": orphan_count},
        "merge_readiness_counts": {
            "training_ready_prismos_native": manifest.training_ready_native_count,
            "native_candidate_blocked_by_pending_gate": manifest.native_candidate_blocked_count,
            "derived_or_post_run_separated": manifest.derived_or_post_run_count
        },
        "operand_value_proof": {
            "field": "post_md_state_vertex.spike_sensor_tensor.value.sha256",
            "claimed_sha256": demo_post_sha,
            "real_file_sha256": real_sha_a,
            "match": operand_value_verified
        },
        "qnanl_decode_post_md_byte_exact": qnanl_decode_byte_exact,
        "orchestrated_writer_proof": {
            "writer_version": prismcode::qnanl_artifact::RUNTIME_QNANL_STREAM_WRITER_VERSION,
            "chunks_sealed": ow_chunks,
            "all_decode_byte_exact": ow_all_decode,
            "stream_lineage": ow_lineage.display().to_string(),
            "writer_stamped_label": "RUNTIME_SUBPROCESS_ORCHESTRATED_BY_PRISMOS",
            "demo_level_origin": "POST_RUN_DERIVED",
            "note": "orchestrated-writer machinery proven (chunk seal + byte-exact decode); its native label is the would-be-LIVE origin, the exerciser drove it POST_RUN."
        },
        "report_b_counterfactual_leg": {
            "stream1_emitted_artifacts": report_b.emitted.len(),
            "stream1_merge_orphans": report_b.merge_manifest_orphan_count
        },
        "all_decode_proofs_pass": report_a.all_decode_proofs_pass,
        "artifacts": demo_records
    });
    std::fs::write(
        demo_root.join("demo_v2_manifest.json"),
        serde_json::to_string_pretty(&manifest_doc).unwrap(),
    )
    .unwrap();

    println!(
        "DSTW_T4_DEMO_V2_OK relabeled={} ledgers={} orphans={} mapped={} cf_differ={} operand_verified={} qnanl_decode={} ow_chunks={} ow_decode={} prov_present={}",
        demo_records.len(),
        manifest.output_paths.len(),
        orphan_count,
        mapped,
        tensors_differ,
        operand_value_verified,
        qnanl_decode_byte_exact,
        ow_chunks,
        ow_all_decode,
        all_provenance_present
    );
    println!("demo_dir={}", demo_root.display());
}
