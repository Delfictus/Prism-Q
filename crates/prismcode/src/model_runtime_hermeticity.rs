use crate::binary_runtime::{PcError, PcResult};
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct CanonicalArtifactSnapshot {
    pub root: PathBuf,
    pub hash: String,
    pub entries: Vec<Value>,
}

pub fn snapshot_canonical_artifacts(root: &Path) -> PcResult<CanonicalArtifactSnapshot> {
    let root = fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    let mut entries = Vec::new();
    collect(&root, &root, &mut entries)?;
    entries.sort_by(|a, b| {
        a.get("path")
            .and_then(Value::as_str)
            .unwrap_or("")
            .cmp(b.get("path").and_then(Value::as_str).unwrap_or(""))
    });
    let mut hasher = blake3::Hasher::new();
    for entry in &entries {
        hasher.update(
            entry
                .get("path")
                .and_then(Value::as_str)
                .unwrap_or("")
                .as_bytes(),
        );
        hasher.update(
            entry
                .get("hash_blake3")
                .and_then(Value::as_str)
                .unwrap_or("")
                .as_bytes(),
        );
    }
    Ok(CanonicalArtifactSnapshot {
        root,
        hash: hasher.finalize().to_hex().to_string(),
        entries,
    })
}

pub fn hermeticity_report(
    before: &CanonicalArtifactSnapshot,
    after: &CanonicalArtifactSnapshot,
    runtime_tmp: &Path,
    runtime_binary_path: &Path,
) -> Value {
    let runtime_tmp = fs::canonicalize(runtime_tmp).unwrap_or_else(|_| runtime_tmp.to_path_buf());
    let runtime_binary_path =
        fs::canonicalize(runtime_binary_path).unwrap_or_else(|_| runtime_binary_path.to_path_buf());
    let bounded_tmp =
        runtime_tmp.starts_with("/mnt/storage/NAS") || runtime_tmp.starts_with("/mnt/prism4d-nas");
    let portable_tmp = portable_non_nas_tmp(&runtime_tmp);
    json!({
        "status": if before.hash == after.hash && (bounded_tmp || portable_tmp) && runtime_binary_path.is_absolute() { "PASS" } else { "FAIL" },
        "primary_artifact_root": before.root.display().to_string(),
        "primary_artifact_hash_before": before.hash,
        "primary_artifact_hash_after": after.hash,
        "primary_artifacts_unchanged": before.hash == after.hash,
        "canonical_artifact_count": before.entries.len(),
        "runtime_temp_root": runtime_tmp.display().to_string(),
        "runtime_temp_under_bounded_nas": bounded_tmp,
        "runtime_temp_portable_non_nas": portable_tmp,
        "runtime_process_launched_by_absolute_path": runtime_binary_path.is_absolute(),
        "source_tree_used": false,
        "cargo_used": false,
    })
}

fn portable_non_nas_tmp(path: &Path) -> bool {
    crate::host_roots::portable_non_nas_tmp(path)
}

fn collect(root: &Path, dir: &Path, entries: &mut Vec<Value>) -> PcResult<()> {
    if !dir.exists() {
        return Ok(());
    }
    for item in fs::read_dir(dir)? {
        let item = item?;
        let path = item.path();
        if path.is_dir() {
            collect(root, &path, entries)?;
        } else if path.is_file() && is_canonical_artifact(&path) {
            let real = fs::canonicalize(&path)?;
            if !real.starts_with(root) {
                return Err(PcError::Security(format!(
                    "model runtime canonical artifact path escape: {}",
                    real.display()
                )));
            }
            let bytes = fs::read(&real)?;
            let rel = real
                .strip_prefix(root)
                .map_err(|_| PcError::Security("model runtime path escape".to_string()))?;
            entries.push(json!({
                "path": rel.display().to_string(),
                "hash_blake3": blake3::hash(&bytes).to_hex().to_string(),
                "byte_size": bytes.len(),
            }));
        }
    }
    Ok(())
}

fn is_canonical_artifact(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    name.ends_with(".pcode")
        || name.ends_with(".pcout")
        || name.ends_with(".pseal")
        || name.ends_with(".nanlb")
        || name.ends_with(".nanlq4")
        || name.ends_with(".nrf")
        || name.ends_with(".drf")
}
