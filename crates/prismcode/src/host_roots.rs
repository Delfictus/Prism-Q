//! Host-relative roots for hermeticity and portability policy checks.
//!
//! Policy checks need the operator's per-user cache, toolchain and development
//! source trees, and the bounded NAS work lane. None of these are compiled in:
//! they are derived from the environment at call time. When a required
//! variable is unset, empty or relative the caller fails closed.

use std::path::{Path, PathBuf};

/// `$HOME` as an absolute path, or `None` when unset, empty or relative.
pub fn home() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
}

/// `$HOME/<rel>` for each entry of `rels`; `None` when `$HOME` is unusable.
pub fn under_home(rels: &[&str]) -> Option<Vec<String>> {
    let home = home()?;
    Some(
        rels.iter()
            .map(|rel| home.join(rel).to_string_lossy().into_owned())
            .collect(),
    )
}

/// Development source trees that portable artifacts must never reference.
///
/// `PRISM_DEV_SOURCE_ROOTS` (a `:`-separated list of absolute paths) overrides
/// the default `$HOME/Desktop/Prism4D-bio` and `$HOME/Desktop/PrismOS`.
/// `None` when neither source yields an absolute root.
pub fn dev_source_roots() -> Option<Vec<String>> {
    if let Some(value) = std::env::var_os("PRISM_DEV_SOURCE_ROOTS") {
        let roots: Vec<String> = std::env::split_paths(&value)
            .filter(|path| path.is_absolute())
            .map(|path| path.to_string_lossy().into_owned())
            .collect();
        return if roots.is_empty() { None } else { Some(roots) };
    }
    under_home(&["Desktop/Prism4D-bio", "Desktop/PrismOS"])
}

/// Per-user roots (relative to `$HOME`) that hermetic runtime temp space and
/// portable artifacts must never touch.
pub const HOME_FORBIDDEN_RELATIVE: [&str; 5] =
    [".cache", ".huggingface", ".ollama", ".cargo", ".rustup"];

/// The bounded NAS work lane, `PRISM_BOUNDED_NAS`, when set to an absolute path.
pub fn bounded_nas_work_root() -> Option<PathBuf> {
    std::env::var_os("PRISM_BOUNDED_NAS")
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
}

/// True when `path` is a portable `/tmp/prism_*` runtime temp root that names
/// no NAS mount, per-user cache or toolchain root, or development source tree.
/// Fails closed (`false`) when `$HOME` or the source roots cannot be resolved.
pub fn portable_non_nas_tmp(path: &Path) -> bool {
    let text = path.to_string_lossy();
    if !text.starts_with("/tmp/prism_")
        || text.starts_with("/mnt/storage")
        || text.starts_with("/mnt/prism4d-nas")
    {
        return false;
    }
    let (Some(mut forbidden), Some(dev)) = (under_home(&HOME_FORBIDDEN_RELATIVE), dev_source_roots())
    else {
        return false;
    };
    forbidden.extend(dev);
    !forbidden.iter().any(|root| text.contains(root.as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roots_are_absolute_and_never_compiled_in() {
        if let Some(roots) = dev_source_roots() {
            assert!(roots.iter().all(|root| root.starts_with('/')));
        }
        if let Some(roots) = under_home(&HOME_FORBIDDEN_RELATIVE) {
            assert_eq!(roots.len(), HOME_FORBIDDEN_RELATIVE.len());
        }
    }
}
