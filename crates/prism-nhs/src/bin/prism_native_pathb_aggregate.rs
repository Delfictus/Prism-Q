//! Production cross-run Native Path-B aggregation entry point.
//!
//! This command is intentionally separate from `nhs_rt_full`: one MD runtime
//! produces exactly one physical lane/run/seed evidence unit, while this
//! command admits the immutable 3-lane x >=5-seed manifest and performs the
//! single no-clobber Path-B/fiber/control/registry commit.

use anyhow::{bail, Context, Result};
use clap::Parser;
use prismcode::runtime_provenance::RuntimeProvenance;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Component, Path, PathBuf};

#[derive(Debug, Parser)]
#[command(name = "prism-native-pathb-aggregate")]
#[command(
    about = "Aggregate immutable matched-lane evidence into Native Path-B fibers and the append-only champion registry"
)]
struct Args {
    /// Absolute normalized path to the immutable cross-run manifest.
    #[arg(long)]
    manifest: PathBuf,

    /// Absolute normalized, no-clobber output root for this aggregation.
    #[arg(long)]
    output: PathBuf,
}

fn validate_output_root(path: &Path) -> Result<()> {
    if !path.is_absolute()
        || path
            .components()
            .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
    {
        bail!("aggregation output must be an absolute lexically normalized path");
    }
    if path.exists() {
        let metadata = std::fs::symlink_metadata(path)
            .with_context(|| format!("inspect aggregation output {}", path.display()))?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            bail!("existing aggregation output must be a non-symlink directory");
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Serialize)]
struct ExecutionReceiptBody {
    aggregation_manifest_path: String,
    aggregation_manifest_sha256: String,
    aggregation_manifest_file_sha256: String,
    output_root: String,
    executable_path: String,
    executable_sha256: String,
    argv: Vec<String>,
    pathb_model_sha256: String,
    component_correspondence_sha256: String,
    open_basin_binding_sha256: String,
    fiber_count: usize,
    fragment_assignment_ledger_sha256: String,
    candidate_claim_evaluation_sha256: String,
    candidate_claim_evaluation_status:
        prism_nhs::pathb_evidence_emitter::CandidateClaimEvaluationStatus,
    matched_control_result_sha256: String,
    candidate_sha256: Option<String>,
    durable_registry_path: String,
    durable_registry_sha256: Option<String>,
    registry_snapshot_path: Option<String>,
    registry_snapshot_sha256: Option<String>,
    typed_artifacts: Vec<prism_nhs::pathb_evidence_emitter::TypedEvidenceArtifact>,
    emission_sha256: String,
    legacy_binding_site_json_inputs: u64,
}

#[derive(Debug, Serialize)]
struct ExecutionReceipt {
    schema_kind: &'static str,
    schema_version: u32,
    status: &'static str,
    receipt_sha256: String,
    body: ExecutionReceiptBody,
}

fn sha256_regular_nofollow(path: &Path) -> Result<String> {
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("inspect regular input {}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        bail!(
            "hash input is not a regular non-symlink file: {}",
            path.display()
        );
    }
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)
        .with_context(|| format!("open regular input {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn commit_execution_receipt(path: &Path, body: ExecutionReceiptBody) -> Result<ExecutionReceipt> {
    let body_bytes = serde_json::to_vec(&body)?;
    let receipt = ExecutionReceipt {
        schema_kind: "prism_native_pathb_aggregate_execution_receipt",
        schema_version: 1,
        status: "PASS",
        receipt_sha256: format!("{:x}", Sha256::digest(&body_bytes)),
        body,
    };
    let mut bytes = serde_json::to_vec_pretty(&receipt)?;
    bytes.push(b'\n');
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)
        .with_context(|| format!("create-new execution receipt {}", path.display()))?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    let parent = path.parent().context("execution receipt has no parent")?;
    std::fs::File::open(parent)?.sync_all()?;
    let readback = std::fs::read(path)?;
    if readback != bytes
        || format!("{:x}", Sha256::digest(serde_json::to_vec(&receipt.body)?))
            != receipt.receipt_sha256
    {
        bail!("committed execution receipt failed exact readback/hash verification");
    }
    Ok(receipt)
}

#[cfg(feature = "gpu")]
fn run(args: Args) -> Result<()> {
    validate_output_root(&args.output)?;
    let receipt_path = args
        .output
        .join("prism_native_pathb_aggregate_execution_receipt.json");
    let receipt_packet_path = PathBuf::from(format!("{}.prismqnanl", receipt_path.display()));
    let receipt_qnanl_receipt_path =
        PathBuf::from(format!("{}.receipt.json", receipt_packet_path.display()));
    if receipt_path.exists() || receipt_packet_path.exists() || receipt_qnanl_receipt_path.exists()
    {
        bail!("aggregation execution receipt targets already exist; no-clobber admission failed");
    }
    let argv = std::env::args_os()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    let provenance = RuntimeProvenance::capture(&args.output, argv.clone());
    let manifest_sha256 = sha256_regular_nofollow(&args.manifest)?;
    let executable_path = std::env::current_exe()?.canonicalize()?;
    let executable_sha256 = sha256_regular_nofollow(&executable_path)?;
    let loaded = prism_nhs::pathb_evidence_emitter::load_pathb_cross_run_aggregation_manifest(
        &args.manifest,
    )
    .context("admit immutable Native Path-B cross-run manifest")?;
    let emission = prism_nhs::pathb_evidence_emitter::aggregate_cross_run_native_pathb_evidence(
        &args.output,
        loaded,
        &provenance,
    )
    .context("execute no-clobber Native Path-B cross-run aggregation")?;
    if manifest_sha256 != sha256_regular_nofollow(&args.manifest)? {
        bail!("aggregation manifest changed during execution");
    }
    let receipt = commit_execution_receipt(
        &receipt_path,
        ExecutionReceiptBody {
            aggregation_manifest_path: args.manifest.display().to_string(),
            aggregation_manifest_sha256: emission.aggregation_manifest_sha256.clone(),
            aggregation_manifest_file_sha256: manifest_sha256,
            output_root: args.output.display().to_string(),
            executable_path: executable_path.display().to_string(),
            executable_sha256,
            argv,
            pathb_model_sha256: emission.pathb.model.model_sha256.clone(),
            component_correspondence_sha256: emission
                .component_correspondence
                .ledger_sha256
                .clone(),
            open_basin_binding_sha256: emission.open_basin.binding_sha256.clone(),
            fiber_count: emission.fiber_count,
            fragment_assignment_ledger_sha256: emission.fragment_assignment.ledger_sha256.clone(),
            candidate_claim_evaluation_sha256: emission.claim_evaluation.evaluation_sha256.clone(),
            candidate_claim_evaluation_status: emission.claim_evaluation.status,
            matched_control_result_sha256: emission.matched_control.result_sha256.clone(),
            candidate_sha256: emission
                .candidate
                .as_ref()
                .map(|candidate| candidate.candidate_sha256.clone()),
            durable_registry_path: emission.durable_registry_path.clone(),
            durable_registry_sha256: emission.durable_registry_sha256.clone(),
            registry_snapshot_path: emission.registry_snapshot_path.clone(),
            registry_snapshot_sha256: emission.registry_snapshot_sha256.clone(),
            typed_artifacts: emission.artifacts.clone(),
            emission_sha256: emission.emission_sha256.clone(),
            legacy_binding_site_json_inputs: 0,
        },
    )?;
    let typed_receipt =
        prism_nhs::pathb_evidence_emitter::emit_native_pathb_aggregate_execution_receipt_qnanl(
            &receipt_path,
            &provenance,
        )?;
    let summary = serde_json::json!({
        "status": "PASS",
        "execution_receipt_path": receipt_path,
        "execution_receipt_sha256": receipt.receipt_sha256,
        "execution_receipt_qnanl_packet": typed_receipt.packet_path,
        "execution_receipt_qnanl_packet_sha256": typed_receipt.packet_sha256,
        "execution_receipt_qnanl_receipt": typed_receipt.receipt_path,
        "execution_receipt_qnanl_receipt_sha256": typed_receipt.receipt_sha256,
        "exact_byte_roundtrip_pass": typed_receipt.exact_byte_roundtrip_pass,
    });
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn run(_args: Args) -> Result<()> {
    bail!("prism-native-pathb-aggregate requires the gpu feature")
}

fn main() -> Result<()> {
    run(Args::parse())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn command_shape_requires_manifest_and_output() {
        let parsed = Args::try_parse_from([
            "prism-native-pathb-aggregate",
            "--manifest",
            "/evidence/campaign.json",
            "--output",
            "/evidence/aggregate-01",
        ])
        .unwrap();
        assert_eq!(parsed.manifest, PathBuf::from("/evidence/campaign.json"));
        assert_eq!(parsed.output, PathBuf::from("/evidence/aggregate-01"));
        assert!(Args::try_parse_from(["prism-native-pathb-aggregate"]).is_err());
    }

    #[test]
    fn output_root_rejects_relative_and_parent_traversal() {
        assert!(validate_output_root(Path::new("relative/output")).is_err());
        assert!(validate_output_root(Path::new("/tmp/../escape")).is_err());
    }

    #[test]
    fn execution_receipt_is_create_new_fsynced_and_hash_bound() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("execution.json");
        let body = ExecutionReceiptBody {
            aggregation_manifest_path: "/evidence/campaign.json".into(),
            aggregation_manifest_sha256: "01".repeat(32),
            aggregation_manifest_file_sha256: "02".repeat(32),
            output_root: root.path().display().to_string(),
            executable_path: "/opt/prismos/bin/prism-native-pathb-aggregate".into(),
            executable_sha256: "03".repeat(32),
            argv: vec!["prism-native-pathb-aggregate".into()],
            pathb_model_sha256: "04".repeat(32),
            component_correspondence_sha256: "05".repeat(32),
            open_basin_binding_sha256: "06".repeat(32),
            fiber_count: 15,
            matched_control_result_sha256: "07".repeat(32),
            fragment_assignment_ledger_sha256: "08".repeat(32),
            candidate_claim_evaluation_sha256: "09".repeat(32),
            candidate_claim_evaluation_status:
                prism_nhs::pathb_evidence_emitter::CandidateClaimEvaluationStatus::Eligible,
            candidate_sha256: Some("0a".repeat(32)),
            durable_registry_path: "/evidence/champions.jsonl".into(),
            durable_registry_sha256: Some("0b".repeat(32)),
            registry_snapshot_path: Some("/evidence/snapshot.json".into()),
            registry_snapshot_sha256: Some("0c".repeat(32)),
            typed_artifacts: Vec::new(),
            emission_sha256: "0d".repeat(32),
            legacy_binding_site_json_inputs: 0,
        };
        let expected = format!("{:x}", Sha256::digest(serde_json::to_vec(&body).unwrap()));
        let receipt = commit_execution_receipt(&path, body.clone()).unwrap();
        assert_eq!(receipt.receipt_sha256, expected);
        assert_eq!(std::fs::read(&path).unwrap().last(), Some(&b'\n'));
        assert!(commit_execution_receipt(&path, body).is_err());
    }
}
