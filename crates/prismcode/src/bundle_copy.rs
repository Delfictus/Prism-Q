use crate::binary_runtime::{PcError, PcResult};
use std::fs;
use std::path::Path;

pub fn copy_bundle_for_tamper(src: &Path, dst: &Path) -> PcResult<()> {
    let src = fs::canonicalize(src)?;
    if dst.exists() {
        fs::remove_dir_all(dst)?;
    }
    fs::create_dir_all(dst)?;
    copy_dir(&src, &src, dst)
}

fn copy_dir(root: &Path, src: &Path, dst: &Path) -> PcResult<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let src_path = entry.path();
        let name = entry.file_name();
        let name_string = name.to_string_lossy();
        if name_string.starts_with(".acceptance_tamper")
            || name_string.starts_with(".acceptance_tmp")
        {
            continue;
        }
        let dst_path = dst.join(&name);
        let meta = fs::symlink_metadata(&src_path)?;
        if meta.file_type().is_symlink() {
            let target = fs::canonicalize(&src_path)?;
            if !target.starts_with(root) {
                return Err(PcError::Security(format!(
                    "PC_ERR_HERMETICITY_PATH_ESCAPE:{} -> {}",
                    src_path.display(),
                    target.display()
                )));
            }
            if target.is_dir() {
                copy_dir(root, &target, &dst_path)?;
            } else {
                fs::copy(&target, &dst_path)?;
            }
        } else if src_path.is_dir() {
            copy_dir(root, &src_path, &dst_path)?;
        } else if src_path.is_file() {
            copy_file(root, &src_path, &dst_path)?;
        }
    }
    Ok(())
}

fn copy_file(root: &Path, src: &Path, dst: &Path) -> PcResult<()> {
    if is_model_artifact(root, src) {
        match fs::hard_link(src, dst) {
            Ok(()) => return Ok(()),
            Err(err) if hardlink_fallback_allowed(&err) => {}
            Err(err) => return Err(PcError::from(err)),
        }
    }
    fs::copy(src, dst)?;
    Ok(())
}

fn is_model_artifact(root: &Path, path: &Path) -> bool {
    path.strip_prefix(root)
        .map(|rel| rel.starts_with("models"))
        .unwrap_or(false)
}

fn hardlink_fallback_allowed(err: &std::io::Error) -> bool {
    matches!(
        err.kind(),
        std::io::ErrorKind::AlreadyExists
            | std::io::ErrorKind::CrossesDevices
            | std::io::ErrorKind::Unsupported
            | std::io::ErrorKind::PermissionDenied
    )
}
