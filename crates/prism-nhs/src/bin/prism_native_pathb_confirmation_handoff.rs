//! Fail-atomic post-discovery confirmation handoff.
//!
//! This executable cannot aggregate or select discovery candidates. It accepts
//! an operator-authored explicit selection only after the source discovery
//! transaction is committed and independently verifies tracker-v2 and disjoint
//! confirmation-seed authorities.

use anyhow::{bail, Context, Result};
use clap::Parser;
use prismcode::runtime_provenance::RuntimeProvenance;
use sha2::{Digest, Sha256};
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Component, Path, PathBuf};

#[derive(Debug, Parser)]
#[command(name = "prism-native-pathb-confirmation-handoff")]
#[command(about = "Materialize an explicit post-commit Path-B confirmation handoff")]
struct Args {
    #[arg(long)]
    discovery_transaction: PathBuf,
    #[arg(long)]
    selection_authority: PathBuf,
    #[arg(long)]
    tracker_design_authority: PathBuf,
    #[arg(long)]
    confirmation_seed_authority: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

fn validate_absolute(path: &Path) -> Result<()> {
    if !path.is_absolute()
        || path
            .components()
            .any(|part| matches!(part, Component::CurDir | Component::ParentDir))
    {
        bail!("all confirmation-handoff paths must be absolute and lexically normalized");
    }
    Ok(())
}

fn sha256_regular_nofollow(path: &Path) -> Result<String> {
    validate_absolute(path)?;
    let metadata = std::fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        bail!("confirmation-handoff authority must be a non-symlink regular file");
    }
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)?;
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

fn main() -> Result<()> {
    let args = Args::parse();
    for path in [
        &args.discovery_transaction,
        &args.selection_authority,
        &args.tracker_design_authority,
        &args.confirmation_seed_authority,
        &args.output,
    ] {
        validate_absolute(path)?;
    }
    let input_hashes = [
        sha256_regular_nofollow(&args.selection_authority)?,
        sha256_regular_nofollow(&args.tracker_design_authority)?,
        sha256_regular_nofollow(&args.confirmation_seed_authority)?,
    ];
    let selection = prism_nhs::pathb_discovery::load_explicit_candidate_selection_authority(
        &args.selection_authority,
    )
    .context("load explicit post-commit candidate selection")?;
    let tracker = prism_nhs::pathb_discovery::load_unbound_tracker_design_authority(
        &args.tracker_design_authority,
    )
    .context("load tracker design authority")?;
    let seeds = prism_nhs::pathb_discovery::load_unbound_confirmation_seed_authority(
        &args.confirmation_seed_authority,
    )
    .context("load disjoint confirmation seed authority")?;
    let argv = std::env::args_os()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    let provenance = RuntimeProvenance::capture(&args.output, argv);
    let emission = prism_nhs::pathb_discovery::materialize_confirmation_handoff(
        &args.discovery_transaction,
        &args.output,
        selection,
        tracker,
        seeds,
        &provenance,
    )
    .context("materialize fail-atomic post-commit confirmation handoff")?;
    let observed_hashes = [
        sha256_regular_nofollow(&args.selection_authority)?,
        sha256_regular_nofollow(&args.tracker_design_authority)?,
        sha256_regular_nofollow(&args.confirmation_seed_authority)?,
    ];
    if observed_hashes != input_hashes {
        bail!("confirmation-handoff authority changed during execution");
    }
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer_pretty(
        &mut stdout,
        &serde_json::json!({
            "status": "PASS",
            "schema_kind": "prism_native_pathb_confirmation_handoff_execution_summary",
            "handoff_sha256": emission.handoff.handoff_sha256,
            "transaction_commit_sha256": emission.commit.commit_sha256,
            "discovery_transaction_commit_sha256": emission.handoff.body.discovery_transaction_commit_sha256,
            "discovery_population_sha256": emission.handoff.body.discovery_population_sha256,
            "selected_candidate_id_sha256": emission.handoff.body.selected_candidate_id_sha256,
            "selected_candidate_authority_sha256": emission.handoff.body.selected_candidate_authority_sha256,
            "explicit_selection_authority_sha256": emission.handoff.body.explicit_selection_authority_sha256,
            "tracker_design_authority_sha256": emission.handoff.body.tracker_design_authority_sha256,
            "disjoint_confirmation_seed_authority_sha256": emission.handoff.body.disjoint_confirmation_seed_authority_sha256,
            "output_root": emission.output_root,
        }),
    )?;
    stdout.write_all(b"\n")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_requires_every_post_commit_authority() {
        assert!(Args::try_parse_from([
            "prism-native-pathb-confirmation-handoff",
            "--discovery-transaction",
            "/evidence/discovery",
            "--selection-authority",
            "/evidence/selection.json",
            "--tracker-design-authority",
            "/evidence/tracker.json",
            "--confirmation-seed-authority",
            "/evidence/seeds.json",
            "--output",
            "/evidence/handoff",
        ])
        .is_ok());
        assert!(Args::try_parse_from(["prism-native-pathb-confirmation-handoff"]).is_err());
        assert!(validate_absolute(Path::new("relative/handoff")).is_err());
    }
}
