//! Production no-probe whole-lattice discovery aggregation entry point.

use anyhow::{bail, Context, Result};
use clap::Parser;
use prismcode::runtime_provenance::RuntimeProvenance;
use sha2::{Digest, Sha256};
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Component, Path, PathBuf};

#[derive(Debug, Parser)]
#[command(name = "prism-native-pathb-discovery")]
#[command(
    about = "Aggregate committed no-probe whole-lattice lanes into immutable unbiased discovery candidates"
)]
struct Args {
    #[arg(long)]
    manifest: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

fn validate_absolute(path: &Path) -> Result<()> {
    if !path.is_absolute()
        || path
            .components()
            .any(|part| matches!(part, Component::CurDir | Component::ParentDir))
    {
        bail!("all discovery CLI paths must be absolute and lexically normalized");
    }
    Ok(())
}

fn sha256_regular_nofollow(path: &Path) -> Result<String> {
    validate_absolute(path)?;
    let metadata = std::fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        bail!("discovery input must be a non-symlink regular file");
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
    validate_absolute(&args.manifest)?;
    validate_absolute(&args.output)?;
    let manifest_file_sha256 = sha256_regular_nofollow(&args.manifest)?;
    let manifest = prism_nhs::pathb_discovery::load_discovery_manifest(&args.manifest)
        .context("load canonical no-probe discovery manifest")?;
    let loaded = prism_nhs::pathb_discovery::load_discovery_population(manifest)
        .context("authenticate committed no-probe lane population")?;
    let argv = std::env::args_os()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    let provenance = RuntimeProvenance::capture(&args.output, argv);
    let emission =
        prism_nhs::pathb_discovery::aggregate_no_probe_discovery(&args.output, loaded, &provenance)
            .context("materialize fail-atomic no-probe discovery population")?;
    if manifest_file_sha256 != sha256_regular_nofollow(&args.manifest)? {
        bail!("discovery manifest changed during execution");
    }
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer_pretty(
        &mut stdout,
        &serde_json::json!({
            "status": "PASS",
            "schema_kind": "prism_native_pathb_discovery_execution_summary",
            "population_sha256": emission.population.population_sha256,
            "candidate_authority_count": emission.authorities.len(),
            "transaction_commit_sha256": emission.commit.commit_sha256,
            "output_root": emission.output_root,
            "legacy_binding_site_json_inputs": 0,
            "probe_chemistry_candidate_inputs": 0,
            "ranking_applied": false,
            "suppression_applied": false,
        }),
    )?;
    stdout.write_all(b"\n")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_requires_absolute_manifest_and_output() {
        let parsed = Args::try_parse_from([
            "prism-native-pathb-discovery",
            "--manifest",
            "/evidence/discovery.json",
            "--output",
            "/evidence/discovery-output",
        ])
        .unwrap();
        assert!(validate_absolute(&parsed.manifest).is_ok());
        assert!(Args::try_parse_from(["prism-native-pathb-discovery"]).is_err());
        assert!(Args::try_parse_from([
            "prism-native-pathb-discovery",
            "--manifest",
            "/evidence/discovery.json",
            "--output",
            "/evidence/discovery-output",
            "--tracker-design-authority",
            "/evidence/tracker.json",
        ])
        .is_err());
        assert!(validate_absolute(Path::new("relative/output")).is_err());
        assert!(validate_absolute(Path::new("/tmp/../escape")).is_err());
    }
}
