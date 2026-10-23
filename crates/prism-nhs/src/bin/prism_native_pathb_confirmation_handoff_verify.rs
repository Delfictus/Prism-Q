//! Read-only verifier for a committed Native Path-B confirmation handoff.
//!
//! The command emits one compact canonical JSON line to stdout and performs no
//! filesystem writes.  It deliberately calls the native committed-root loader;
//! downstream campaign tooling can therefore bind this verifier output instead
//! of reimplementing the authority graph or trusting legacy candidate JSON.

use anyhow::{bail, Context, Result};
use clap::Parser;
use prism_nhs::pathb_discovery::{
    load_committed_confirmation_handoff, CONFIRMATION_HANDOFF_COMMIT_FILE,
    CONFIRMATION_HANDOFF_FILE,
};
use prismcode::runtime_provenance::RuntimeProvenance;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Component, Path, PathBuf};

#[derive(Debug, Parser)]
#[command(name = "prism-native-pathb-confirmation-handoff-verify")]
#[command(about = "Read-only verification of a committed Path-B confirmation handoff root")]
struct Args {
    /// Absolute normalized committed confirmation-handoff root.
    #[arg(long)]
    handoff_root: PathBuf,
}

#[derive(Debug, Serialize)]
struct VerificationSummaryBody {
    verification_contract: &'static str,
    verification_mode: &'static str,
    handoff_root: String,
    handoff_root_commit_file_sha256: String,
    handoff_root_raw_file_sha256: String,
    handoff_sha256: String,
    handoff_transaction_commit_sha256: String,
    handoff_ordered_membership_sha256: String,
    discovery_transaction_commit_sha256: String,
    discovery_population_sha256: String,
    discovery_manifest_sha256: String,
    correspondence_ledger_sha256: String,
    selected_candidate_id_sha256: String,
    selected_candidate_authority_sha256: String,
    explicit_selection_authority_sha256: String,
    operator_selection_authority_sha256: String,
    tracker_design_authority_sha256: String,
    supplied_tracker_sha256: String,
    operator_design_authority_sha256: String,
    disjoint_confirmation_seed_authority_sha256: String,
    discovery_replica_seeds: Vec<u64>,
    confirmation_replica_seeds: Vec<u64>,
    qnanl_artifact_family: String,
    qnanl_raw_sha256: String,
    qnanl_packet_sha256: String,
    qnanl_receipt_sha256: String,
    qnanl_exact_byte_roundtrip_pass: bool,
    verifier_executable_path: String,
    verifier_executable_sha256: String,
    runtime_provenance_sha256: String,
    runtime_provenance: RuntimeProvenance,
    double_read_stability_pass: bool,
    legacy_candidate_json_inputs: u64,
}

#[derive(Debug, Serialize)]
struct VerificationSummary {
    schema_kind: &'static str,
    schema_version: &'static str,
    status: &'static str,
    summary_body_sha256: String,
    body: VerificationSummaryBody,
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn validate_absolute_normalized_directory(path: &Path) -> Result<()> {
    if !path.is_absolute()
        || path
            .components()
            .any(|part| matches!(part, Component::CurDir | Component::ParentDir))
    {
        bail!("confirmation-handoff root must be absolute and lexically normalized");
    }
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("inspect confirmation-handoff root {}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        bail!("confirmation-handoff root must be a non-symlink directory");
    }
    let canonical = path
        .canonicalize()
        .with_context(|| format!("canonicalize confirmation-handoff root {}", path.display()))?;
    if canonical != path {
        bail!("confirmation-handoff root or an ancestor resolves through a non-canonical path");
    }
    Ok(())
}

fn sha256_regular_nofollow(path: &Path) -> Result<String> {
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("inspect immutable verifier input {}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        bail!("verifier input must be a non-symlink regular file");
    }
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)
        .with_context(|| format!("open immutable verifier input {}", path.display()))?;
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

fn run(args: Args) -> Result<()> {
    validate_absolute_normalized_directory(&args.handoff_root)?;
    let commit_path = args.handoff_root.join(CONFIRMATION_HANDOFF_COMMIT_FILE);
    let raw_path = args.handoff_root.join(CONFIRMATION_HANDOFF_FILE);
    let commit_file_sha256_before = sha256_regular_nofollow(&commit_path)?;
    let raw_file_sha256_before = sha256_regular_nofollow(&raw_path)?;

    let first = load_committed_confirmation_handoff(&args.handoff_root)
        .context("first native committed confirmation-handoff verification")?;

    let argv = std::env::args_os()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    let runtime_provenance = RuntimeProvenance::capture(&args.handoff_root, argv);
    let executable_path = std::env::current_exe()
        .context("resolve verifier executable")?
        .canonicalize()
        .context("canonicalize verifier executable")?;
    let executable_sha256_before = sha256_regular_nofollow(&executable_path)?;
    if runtime_provenance.packaged_binary_path.as_deref()
        != Some(executable_path.to_string_lossy().as_ref())
        || runtime_provenance.packaged_binary_sha256.as_deref()
            != Some(executable_sha256_before.as_str())
    {
        bail!("runtime provenance does not bind the executing verifier bytes");
    }

    let second = load_committed_confirmation_handoff(&args.handoff_root)
        .context("second native committed confirmation-handoff verification")?;
    let commit_file_sha256_after = sha256_regular_nofollow(&commit_path)?;
    let raw_file_sha256_after = sha256_regular_nofollow(&raw_path)?;
    let executable_sha256_after = sha256_regular_nofollow(&executable_path)?;
    if first != second
        || commit_file_sha256_before != commit_file_sha256_after
        || raw_file_sha256_before != raw_file_sha256_after
        || executable_sha256_before != executable_sha256_after
    {
        bail!("handoff root or verifier executable changed during read-only verification");
    }

    let (handoff, commit) = second;
    let typed = &commit.body.handoff_typed_artifact;
    let body = VerificationSummaryBody {
        verification_contract:
            "native_load_committed_confirmation_handoff_double_read_exact_bytes_v1",
        verification_mode: "read_only_no_legacy_candidate_json",
        handoff_root: args.handoff_root.display().to_string(),
        handoff_root_commit_file_sha256: commit_file_sha256_after,
        handoff_root_raw_file_sha256: raw_file_sha256_after,
        handoff_sha256: handoff.handoff_sha256.clone(),
        handoff_transaction_commit_sha256: commit.commit_sha256.clone(),
        handoff_ordered_membership_sha256: commit.body.ordered_membership_sha256.clone(),
        discovery_transaction_commit_sha256: handoff
            .body
            .discovery_transaction_commit_sha256
            .clone(),
        discovery_population_sha256: handoff.body.discovery_population_sha256.clone(),
        discovery_manifest_sha256: handoff.body.discovery_manifest_sha256.clone(),
        correspondence_ledger_sha256: handoff.body.correspondence_ledger_sha256.clone(),
        selected_candidate_id_sha256: handoff.body.selected_candidate_id_sha256.clone(),
        selected_candidate_authority_sha256: handoff
            .body
            .selected_candidate_authority_sha256
            .clone(),
        explicit_selection_authority_sha256: handoff
            .body
            .explicit_selection_authority_sha256
            .clone(),
        operator_selection_authority_sha256: handoff
            .body
            .explicit_selection_authority
            .body
            .operator_selection_authority_sha256
            .clone(),
        tracker_design_authority_sha256: handoff.body.tracker_design_authority_sha256.clone(),
        supplied_tracker_sha256: handoff.body.supplied_tracker_sha256.clone(),
        operator_design_authority_sha256: handoff.body.operator_design_authority_sha256.clone(),
        disjoint_confirmation_seed_authority_sha256: handoff
            .body
            .disjoint_confirmation_seed_authority_sha256
            .clone(),
        discovery_replica_seeds: handoff.body.discovery_replica_seeds.clone(),
        confirmation_replica_seeds: handoff.body.confirmation_replica_seeds.clone(),
        qnanl_artifact_family: typed.artifact_family.clone(),
        qnanl_raw_sha256: typed.raw_sha256.clone(),
        qnanl_packet_sha256: typed.packet_sha256.clone(),
        qnanl_receipt_sha256: typed.receipt_sha256.clone(),
        qnanl_exact_byte_roundtrip_pass: typed.exact_byte_roundtrip_pass,
        verifier_executable_path: executable_path.display().to_string(),
        verifier_executable_sha256: executable_sha256_after,
        runtime_provenance_sha256: sha256_bytes(&serde_json::to_vec(&runtime_provenance)?),
        runtime_provenance,
        double_read_stability_pass: true,
        legacy_candidate_json_inputs: 0,
    };
    let summary = VerificationSummary {
        schema_kind: "prism_native_pathb_confirmation_handoff_verification_summary",
        schema_version: "1.0.0",
        status: "PASS",
        summary_body_sha256: sha256_bytes(&serde_json::to_vec(&body)?),
        body,
    };
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer(&mut stdout, &summary)?;
    stdout.write_all(b"\n")?;
    Ok(())
}

fn main() -> Result<()> {
    run(Args::parse())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_is_read_only_and_accepts_only_a_committed_root() {
        assert!(Args::try_parse_from([
            "prism-native-pathb-confirmation-handoff-verify",
            "--handoff-root",
            "/evidence/confirmation-handoff",
        ])
        .is_ok());
        assert!(Args::try_parse_from(["prism-native-pathb-confirmation-handoff-verify"]).is_err());
        assert!(Args::try_parse_from([
            "prism-native-pathb-confirmation-handoff-verify",
            "--handoff-root",
            "/evidence/confirmation-handoff",
            "--output",
            "/evidence/summary.json",
        ])
        .is_err());
        assert!(validate_absolute_normalized_directory(Path::new("relative/handoff")).is_err());
    }
}
