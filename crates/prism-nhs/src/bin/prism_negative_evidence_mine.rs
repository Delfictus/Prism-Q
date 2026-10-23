use anyhow::{Context, Result};
use clap::Parser;
use prism_nhs::negative_scientific_evidence::{
    normalize_negative_scientific_evidence_jsonl, sha256_file, NegativeEvidenceInputReceipt,
    NegativeScientificEvidenceMiningStats,
};
use serde_json::json;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "prism-negative-evidence-mine",
    about = "Normalize pre-merge adversarial evidence into NegativeScientificEvidenceEvent records"
)]
struct Args {
    /// Existing pre-merge negative evidence JSONL to normalize.
    #[arg(long = "input-jsonl", required = true)]
    input_jsonl: Vec<PathBuf>,

    /// Existing pre-merge corpus/SUT/chronology/causal evidence roots audited by the closure.
    #[arg(long = "evidence-root")]
    evidence_roots: Vec<PathBuf>,

    /// Output directory.
    #[arg(long)]
    out: PathBuf,

    /// Optional cap for regression fixtures. Omit for full corpus mining.
    #[arg(long)]
    max_records: Option<u64>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    std::fs::create_dir_all(&args.out).with_context(|| format!("create {}", args.out.display()))?;

    let events_path = args.out.join("negative_scientific_evidence_events.jsonl");
    let mut writer = BufWriter::new(
        File::create(&events_path).with_context(|| format!("create {}", events_path.display()))?,
    );

    let mut aggregate = NegativeScientificEvidenceMiningStats::default();
    let mut input_receipts = Vec::new();
    for input in &args.input_jsonl {
        let bytes = input
            .metadata()
            .with_context(|| format!("stat {}", input.display()))?
            .len();
        let input_sha = sha256_file(input)?;
        input_receipts.push(NegativeEvidenceInputReceipt {
            path: input.clone(),
            sha256: input_sha,
            bytes,
        });
        let reader =
            BufReader::new(File::open(input).with_context(|| format!("open {}", input.display()))?);
        let remaining_cap = args
            .max_records
            .map(|max| max.saturating_sub(aggregate.emitted_records));
        let stats = normalize_negative_scientific_evidence_jsonl(
            reader,
            input,
            &mut writer,
            remaining_cap,
        )?;
        merge_stats(&mut aggregate, stats);
        if args
            .max_records
            .map(|max| aggregate.emitted_records >= max)
            .unwrap_or(false)
        {
            break;
        }
    }
    drop(writer);

    let event_sha = sha256_file(&events_path)?;
    let tokenizer_manifest_path = args.out.join("tokenizer_manifest.json");
    std::fs::write(
        &tokenizer_manifest_path,
        serde_json::to_vec_pretty(&json!([
            {
                "family": "NegativeScientificEvidenceEvent",
                "jsonl_records_path": events_path.display().to_string(),
                "source_id": "negative_scientific_evidence_events"
            }
        ]))?,
    )?;

    let input_parent_audits: Vec<_> = args
        .input_jsonl
        .iter()
        .filter_map(|path| path.parent())
        .map(|parent| {
            json!({
                "input_parent": parent.display().to_string(),
                "recoverable_negative_evidence_audit": parent.join("RECOVERABLE_NEGATIVE_EVIDENCE_AUDIT.json").display().to_string(),
                "negative_evidence_ledger": parent.join("NegativeEvidenceLedger.json").display().to_string(),
                "science_quality_gate_negative_extension": parent.join("SCIENCE_QUALITY_GATE_NEGATIVE_EXTENSION.json").display().to_string(),
            })
        })
        .collect();

    let ledger = json!({
        "schema_kind": "negative_scientific_evidence_premerge_mining_ledger",
        "schema_version": 1,
        "implementation_status": if aggregate.required_lineage_complete() { "IMPLEMENTATION_VERIFIED" } else { "IMPLEMENTATION_PARTIAL" },
        "runtime_observer_required": false,
        "runtime_observer_artifacts_consumed": [],
        "input_jsonl": input_receipts,
        "evidence_roots": args.evidence_roots.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
        "input_parent_audits": input_parent_audits,
        "output_events_path": events_path.display().to_string(),
        "output_events_sha256": event_sha,
        "tokenizer_manifest_path": tokenizer_manifest_path.display().to_string(),
        "input_records": aggregate.input_records,
        "emitted_records": aggregate.emitted_records,
        "parse_error_count": aggregate.parse_error_count,
        "required_lineage_complete": aggregate.required_lineage_complete(),
        "missing_prismos_opcode_count": aggregate.missing_prismos_opcode_count,
        "missing_prismcode_instruction_count": aggregate.missing_prismcode_instruction_count,
        "missing_qnanl_context_count": aggregate.missing_qnanl_context_count,
        "missing_tokenizer_family_count": aggregate.missing_tokenizer_family_count,
        "missing_provenance_count": aggregate.missing_provenance_count,
        "missing_source_hash_count": aggregate.missing_source_hash_count,
        "missing_chronology_refs_count": aggregate.missing_chronology_refs_count,
        "missing_uncertainty_tensor_refs_count": aggregate.missing_uncertainty_tensor_refs_count,
        "type_counts": aggregate.type_counts,
        "rejection_class_counts": aggregate.rejection_class_counts,
        "unique_source_hashes": aggregate.source_hashes.len(),
    });
    let ledger_path = args
        .out
        .join("NEGATIVE_SCIENTIFIC_EVIDENCE_MINING_LEDGER.json");
    std::fs::write(&ledger_path, serde_json::to_vec_pretty(&ledger)?)?;

    let report = format!(
        "# Negative Scientific Evidence Mining Report\n\nimplementation_status: {}\n\nrecords_mined: {}\n\nrequired_lineage_complete: {}\n\nruntime_observer_required: false\n\noutput_events: `{}`\n\ntokenizer_manifest: `{}`\n",
        ledger["implementation_status"].as_str().unwrap_or("IMPLEMENTATION_PARTIAL"),
        aggregate.emitted_records,
        aggregate.required_lineage_complete(),
        events_path.display(),
        tokenizer_manifest_path.display(),
    );
    std::fs::write(
        args.out
            .join("NEGATIVE_SCIENTIFIC_EVIDENCE_MINING_REPORT.md"),
        report,
    )?;

    println!(
        "{} records -> {}",
        aggregate.emitted_records,
        events_path.display()
    );
    Ok(())
}

fn merge_stats(
    aggregate: &mut NegativeScientificEvidenceMiningStats,
    stats: NegativeScientificEvidenceMiningStats,
) {
    aggregate.input_records += stats.input_records;
    aggregate.emitted_records += stats.emitted_records;
    aggregate.parse_error_count += stats.parse_error_count;
    aggregate.missing_prismos_opcode_count += stats.missing_prismos_opcode_count;
    aggregate.missing_prismcode_instruction_count += stats.missing_prismcode_instruction_count;
    aggregate.missing_qnanl_context_count += stats.missing_qnanl_context_count;
    aggregate.missing_tokenizer_family_count += stats.missing_tokenizer_family_count;
    aggregate.missing_provenance_count += stats.missing_provenance_count;
    aggregate.missing_source_hash_count += stats.missing_source_hash_count;
    aggregate.missing_chronology_refs_count += stats.missing_chronology_refs_count;
    aggregate.missing_uncertainty_tensor_refs_count += stats.missing_uncertainty_tensor_refs_count;
    merge_counts(&mut aggregate.type_counts, stats.type_counts);
    merge_counts(
        &mut aggregate.rejection_class_counts,
        stats.rejection_class_counts,
    );
    aggregate.source_hashes.extend(stats.source_hashes);
}

fn merge_counts(dst: &mut BTreeMap<String, u64>, src: BTreeMap<String, u64>) {
    for (key, value) in src {
        *dst.entry(key).or_default() += value;
    }
}
