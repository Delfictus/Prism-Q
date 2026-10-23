use anyhow::{Context, Result};
use clap::Parser;
use prism_nhs::tsh_consensus_gate::{run_consensus_gate, sha256_file, ConsensusGateConfig};
use serde_json::json;
use sha2::Digest;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(
    name = "prism-tsh-consensus-gate",
    about = "Materialize TSH falsification consensus objects from Stage-II runtime artifacts and canonical negative evidence"
)]
struct Args {
    /// Evaluate complete saved mechanistic tensors independently of TSH admission.
    #[arg(long, conflicts_with_all = ["measured_pair_manifest", "tsh_runtime_dir", "negative_evidence_jsonl"])]
    mechanistic_pair_manifest: Option<PathBuf>,
    /// Run native Stage II on source-pinned paired measured frames, without MD.
    #[arg(long, conflicts_with_all = ["tsh_runtime_dir", "negative_evidence_jsonl"])]
    measured_pair_manifest: Option<PathBuf>,
    /// Directory containing TSH Stage-II runtime artifacts.
    #[arg(long, required_unless_present_any = ["measured_pair_manifest", "mechanistic_pair_manifest"])]
    tsh_runtime_dir: Option<PathBuf>,
    /// V6 canonical NegativeScientificEvidenceEvent JSONL.
    #[arg(long, required_unless_present_any = ["measured_pair_manifest", "mechanistic_pair_manifest"])]
    negative_evidence_jsonl: Option<PathBuf>,
    /// Output directory for the consensus gate bundle.
    #[arg(long)]
    output_dir: PathBuf,
    /// Minimum seed count required for persistent transition promotion.
    #[arg(long, default_value_t = 2)]
    min_consensus_seeds: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(manifest) = &args.mechanistic_pair_manifest {
        let summary = prism_nhs::chemical_state_fiber::mechanistic::evaluate_mechanistic_pair(
            manifest,
            &args.output_dir,
        )?;
        println!("{}", serde_json::to_string_pretty(&summary)?);
        return Ok(());
    }
    if let Some(manifest) = &args.measured_pair_manifest {
        let summary = prism_nhs::tsh_stage_ii::observe_measured_pair(manifest, &args.output_dir)?;
        println!("{}", serde_json::to_string_pretty(&summary)?);
        return Ok(());
    }
    let mut config = ConsensusGateConfig::new(
        args.tsh_runtime_dir
            .clone()
            .context("missing TSH runtime directory")?,
        args.negative_evidence_jsonl
            .clone()
            .context("missing negative evidence")?,
        args.output_dir.clone(),
    );
    config.min_consensus_seeds = args.min_consensus_seeds;
    std::fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("create {}", args.output_dir.display()))?;

    write_preflight_files(&args)?;
    let summary = run_consensus_gate(&config)?;
    write_initial_reports(&args, &summary)?;

    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn write_preflight_files(args: &Args) -> Result<()> {
    let directive = format!(
        "directive_id: TSH_FALSIFICATION_CONSENSUS_GATE_OPERATIONALIZATION_V1\n\
         status_target: IMPLEMENTATION_VERIFIED_OR_PARTIAL\n\
         stage_iii_steering: NOT_IMPLEMENTED\n\
         force_mutation: FORBIDDEN\n\
         dt_mutation: FORBIDDEN\n\
         sampling_bias: FORBIDDEN\n\
         trajectory_mutation: FORBIDDEN\n\
         new_md_execution: FORBIDDEN_UNLESS_EXISTING_CORPUS_EXHAUSTED\n\
         objective: evidence_adjudicated_transition_science_substrate\n"
    );
    std::fs::write(args.output_dir.join("DIRECTIVE_ONTOLOGY.yaml"), directive)?;

    let tsh_dir = args
        .tsh_runtime_dir
        .as_ref()
        .context("missing TSH runtime directory")?;
    let negative = args
        .negative_evidence_jsonl
        .as_ref()
        .context("missing negative evidence")?;
    let tsh_hash = hash_optional_dir_manifest(tsh_dir)?;
    let neg_hash = sha256_file(negative)?;
    let reconciliation = format!(
        "source_of_truth:\n\
         tsh_runtime_dir: {}\n\
         tsh_runtime_dir_hash: {}\n\
         canonical_negative_evidence_jsonl: {}\n\
         canonical_negative_evidence_sha256: {}\n\
         runtime_scope: TSH_STAGE_II_OBSERVATIONAL_ONLY\n\
         falsification_scope: V6_CANONICAL_NEGATIVE_EVIDENCE_BUNDLE\n\
         stage_iii_steering: NOT_MERGED\n",
        tsh_dir.display(),
        tsh_hash,
        negative.display(),
        neg_hash
    );
    std::fs::write(
        args.output_dir.join("SOURCE_OF_TRUTH_RECONCILIATION.yaml"),
        reconciliation,
    )?;
    Ok(())
}

fn write_initial_reports(
    args: &Args,
    summary: &prism_nhs::tsh_consensus_gate::ConsensusGateRunSummary,
) -> Result<()> {
    let regression = json!({
        "schema_kind": "tsh_falsification_consensus_regression_audit",
        "schema_version": 1,
        "status": "PENDING_VALIDATION_COMMANDS",
        "materialization_status": summary.status,
        "new_md_executed": false,
        "stage_iii_steering_merged": false,
        "force_mutation": false,
        "dt_mutation": false,
        "sampling_bias": false,
        "trajectory_mutation": false
    });
    std::fs::write(
        args.output_dir.join("REGRESSION_AUDIT.json"),
        serde_json::to_vec_pretty(&regression)?,
    )?;
    let report = format!(
        "# TSH Falsification Consensus Gate\n\n\
         Status: {}\n\n\
         Motifs: {}\n\
         Canonical negative events: {}\n\
         Consensus events: {}\n\
         Falsification profiles: {}\n\
         PersistentTransitionObjects: {}\n\
         Rejected candidates: {}\n\n\
         Stage III steering: NOT_MERGED\n\
         Runtime mutation: NONE\n",
        summary.status,
        summary.motif_count,
        summary.negative_event_count,
        summary.consensus_event_count,
        summary.falsification_profile_count,
        summary.persistent_transition_object_count,
        summary.rejected_candidate_count
    );
    std::fs::write(args.output_dir.join("FINAL_CLOSURE_REPORT.md"), report)?;
    Ok(())
}

fn hash_optional_dir_manifest(dir: &PathBuf) -> Result<String> {
    let mut entries = Vec::new();
    for name in [
        "thermodynamic_salience_events.jsonl",
        "transition_motif_candidates.jsonl",
        "transition_motifs.jsonl",
        "transition_consensus_events.jsonl",
        "transition_evidence_ledger.jsonl",
        "resident_transition_cache.json",
        "tsh_runtime_non_interference_ledger.json",
        "tsh_stage_iii_compatibility_manifest.json",
    ] {
        let path = dir.join(name);
        if path.exists() {
            entries.push(format!("{name}:{}", sha256_file(&path)?));
        }
    }
    let mut hasher = sha2::Sha256::new();
    for entry in entries {
        sha2::Digest::update(&mut hasher, entry.as_bytes());
    }
    Ok(format!("{:x}", sha2::Digest::finalize(hasher)))
}
