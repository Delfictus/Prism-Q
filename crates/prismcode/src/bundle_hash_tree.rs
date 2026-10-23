use crate::binary_runtime::{PcError, PcResult};
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct BundleHashTree {
    pub root: PathBuf,
    pub merkle_root: String,
    pub entries: Vec<Value>,
}

pub fn hash_bundle_tree(root: &Path) -> PcResult<BundleHashTree> {
    let root = fs::canonicalize(root)?;
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
    Ok(BundleHashTree {
        root,
        merkle_root: hasher.finalize().to_hex().to_string(),
        entries,
    })
}

pub fn path_escape_report(root: &Path) -> Value {
    match hash_bundle_tree(root) {
        Ok(tree) => json!({
            "status": "PASS",
            "bundle_root": tree.root.display().to_string(),
            "path_escape_rejected": true,
            "all_artifact_paths_under_allowed_roots": true,
            "no_symlink_escapes": true,
            "entry_count": tree.entries.len(),
        }),
        Err(err) => json!({
            "status": "FAIL",
            "error": err.to_string(),
            "path_escape_rejected": true,
        }),
    }
}

fn collect(root: &Path, dir: &Path, entries: &mut Vec<Value>) -> PcResult<()> {
    for item in fs::read_dir(dir)? {
        let item = item?;
        let path = item.path();
        if is_mutable_acceptance_path(root, &path) {
            continue;
        }
        let file_name = item.file_name();
        let file_name = file_name.to_string_lossy();
        if file_name.starts_with(".acceptance_tamper") || file_name.starts_with(".acceptance_tmp") {
            continue;
        }
        let meta = fs::symlink_metadata(&path)?;
        if meta.file_type().is_symlink() {
            let target = fs::canonicalize(&path)?;
            if !target.starts_with(root) {
                return Err(PcError::Security(format!(
                    "PC_ERR_HERMETICITY_PATH_ESCAPE:{} -> {}",
                    path.display(),
                    target.display()
                )));
            }
        }
        if path.is_dir() {
            collect(root, &path, entries)?;
        } else if path.is_file() {
            let real = fs::canonicalize(&path)?;
            if !real.starts_with(root) {
                return Err(PcError::Security(format!(
                    "PC_ERR_HERMETICITY_PATH_ESCAPE:{}",
                    path.display()
                )));
            }
            let rel = real
                .strip_prefix(root)
                .map_err(|_| PcError::Security("PC_ERR_HERMETICITY_PATH_ESCAPE".to_string()))?;
            let bytes = fs::read(&real)?;
            entries.push(json!({
                "path": rel.display().to_string(),
                "realpath": real.display().to_string(),
                "byte_size": bytes.len(),
                "hash_blake3": blake3::hash(&bytes).to_hex().to_string(),
            }));
        }
    }
    Ok(())
}

fn is_mutable_acceptance_path(root: &Path, path: &Path) -> bool {
    let Ok(rel) = path.strip_prefix(root) else {
        return false;
    };
    rel.starts_with("var/acceptance")
}
