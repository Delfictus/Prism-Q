//! Single PTX authority for every runtime PTX load (P0.5, 2026-10-23).
//!
//! Every loader asks this module for the PTX directory instead of probing
//! cwd-relative candidates. Probing `target/ptx`, `../../target/ptx` or
//! `crates/prism-gpu/src/kernels/*.ptx` from the current directory could bind
//! stale PTX (for example the July 2026 in-tree copies) to a fresh binary
//! without any error. The authority is resolved once per process, in this order:
//!
//! 1. `PRISM4D_PTX_DIR`, then `PRISM_PTX_DIR`. If one is set, it must name an
//!    existing directory, or resolution fails closed. If both are set, they must
//!    name the same directory.
//! 2. Executable-relative layouts, never the current directory:
//!    `<exe_dir>/ptx` (immutable engine store `/mnt/storage/prism-engines/<sha8>/`),
//!    `<exe_dir>/../ptx` (cargo `<target>/release/<bin>` next to `<target>/ptx`),
//!    `<exe_dir>/../assets/ptx` (release bundle).
//! 3. The governed build output of this exact build (`OUT_DIR/ptx`, baked in
//!    at compile time by build.rs as `PRISM_GPU_GOVERNED_PTX_DIR`).
//!
//! Resolution returns the directory, how it was chosen, and its
//! `kernel_set_sha256`. That hash uses the immutable engine store definition:
//! sha256 over the UTF-8 `sha256sum` lines (`"<hex>  <name>\n"`) of every
//! `*.ptx` file in the directory, sorted by file name. Sidecar `.sha256` files
//! are excluded. A file missing from the authority is an error and never
//! triggers a search elsewhere.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use sha2::{Digest, Sha256};

/// Compile-time governed PTX output directory of this build (OUT_DIR/ptx).
/// `None` when prism-gpu was built without the `cuda` feature.
pub const GOVERNED_BUILD_PTX_DIR: Option<&str> = option_env!("PRISM_GPU_GOVERNED_PTX_DIR");
/// `nvcc --version` release line recorded by build.rs for this build.
pub const BUILD_NVCC_VERSION: Option<&str> = option_env!("PRISM_GPU_NVCC_VERSION");
/// nvcc path pinned for this build (PRISM_GPU_NVCC, required in release).
pub const BUILD_NVCC_PATH: Option<&str> = option_env!("PRISM_GPU_NVCC_PATH");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PtxAuthoritySource {
    EnvPrism4dPtxDir,
    EnvPrismPtxDir,
    ExecutableSiblingPtx,
    ExecutableParentPtx,
    ExecutableBundleAssetsPtx,
    GovernedBuildOutDir,
}

impl PtxAuthoritySource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::EnvPrism4dPtxDir => "env:PRISM4D_PTX_DIR",
            Self::EnvPrismPtxDir => "env:PRISM_PTX_DIR",
            Self::ExecutableSiblingPtx => "exe_dir/ptx",
            Self::ExecutableParentPtx => "exe_dir/../ptx",
            Self::ExecutableBundleAssetsPtx => "exe_dir/../assets/ptx",
            Self::GovernedBuildOutDir => "build:OUT_DIR/ptx",
        }
    }
}

#[derive(Debug, Clone)]
pub struct PtxAuthority {
    /// Canonicalized PTX directory.
    pub dir: PathBuf,
    pub source: PtxAuthoritySource,
    /// sha256 over sorted `sha256sum` lines of every `*.ptx` in `dir`.
    pub kernel_set_sha256: String,
    pub ptx_count: usize,
}

impl PtxAuthority {
    /// Path of `file_name` inside the authority. Fails closed if absent.
    pub fn file(&self, file_name: &str) -> Result<PathBuf, PtxAuthorityError> {
        if file_name.contains('/') || file_name.contains('\\') || file_name == ".." {
            return Err(PtxAuthorityError::InvalidName(file_name.to_string()));
        }
        let path = self.dir.join(file_name);
        if path.is_file() {
            Ok(path)
        } else {
            Err(PtxAuthorityError::MissingFile {
                file: file_name.to_string(),
                dir: self.dir.clone(),
                authority_source: self.source.as_str(),
                kernel_set_sha256: self.kernel_set_sha256.clone(),
            })
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PtxAuthorityError {
    #[error("{var} is set but empty")]
    EmptyEnv { var: &'static str },
    #[error("{var}={value} is not an existing directory (fail closed; no fallback)")]
    EnvNotADirectory { var: &'static str, value: String },
    #[error("PRISM4D_PTX_DIR={a} and PRISM_PTX_DIR={b} name different directories")]
    EnvConflict { a: String, b: String },
    #[error("no PTX authority: set PRISM4D_PTX_DIR/PRISM_PTX_DIR or install ptx/ beside the executable (governed build dir {0} is absent)")]
    NotFound(String),
    #[error("PTX file {file} is not in the PTX authority {dir} ({authority_source}, kernel_set_sha256={kernel_set_sha256})")]
    MissingFile {
        file: String,
        dir: PathBuf,
        authority_source: &'static str,
        kernel_set_sha256: String,
    },
    #[error("invalid PTX file name {0:?}")]
    InvalidName(String),
    #[error("PTX authority I/O error on {path}: {error}")]
    Io { path: PathBuf, error: String },
}

static AUTHORITY: OnceLock<Result<PtxAuthority, String>> = OnceLock::new();

/// Resolve (once per process) and return the PTX authority.
pub fn resolve() -> Result<&'static PtxAuthority, PtxAuthorityError> {
    let cached = AUTHORITY.get_or_init(|| match resolve_uncached() {
        Ok(authority) => {
            log::info!(
                "PTX authority: {} ({}), {} PTX, kernel_set_sha256={}",
                authority.dir.display(),
                authority.source.as_str(),
                authority.ptx_count,
                authority.kernel_set_sha256
            );
            Ok(authority)
        }
        Err(error) => Err(error.to_string()),
    });
    match cached {
        Ok(authority) => Ok(authority),
        Err(message) => Err(PtxAuthorityError::NotFound(message.clone())),
    }
}

/// Resolve the authority and return the path of one PTX file in it.
pub fn ptx_file(file_name: &str) -> Result<PathBuf, PtxAuthorityError> {
    resolve()?.file(file_name)
}

/// Uncached resolution (exposed for tests and receipts).
pub fn resolve_uncached() -> Result<PtxAuthority, PtxAuthorityError> {
    let env_a = read_env_dir("PRISM4D_PTX_DIR")?;
    let env_b = read_env_dir("PRISM_PTX_DIR")?;
    match (&env_a, &env_b) {
        (Some(a), Some(b)) if a != b => {
            return Err(PtxAuthorityError::EnvConflict {
                a: a.display().to_string(),
                b: b.display().to_string(),
            })
        }
        (Some(a), _) => return build(a.clone(), PtxAuthoritySource::EnvPrism4dPtxDir),
        (None, Some(b)) => return build(b.clone(), PtxAuthoritySource::EnvPrismPtxDir),
        (None, None) => {}
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            for (candidate, source) in [
                (exe_dir.join("ptx"), PtxAuthoritySource::ExecutableSiblingPtx),
                (exe_dir.join("../ptx"), PtxAuthoritySource::ExecutableParentPtx),
                (
                    exe_dir.join("../assets/ptx"),
                    PtxAuthoritySource::ExecutableBundleAssetsPtx,
                ),
            ] {
                if candidate.is_dir() && contains_ptx(&candidate) {
                    return build(candidate, source);
                }
            }
        }
    }
    if let Some(governed) = GOVERNED_BUILD_PTX_DIR.map(PathBuf::from) {
        if governed.is_dir() {
            return build(governed, PtxAuthoritySource::GovernedBuildOutDir);
        }
    }
    Err(PtxAuthorityError::NotFound(
        GOVERNED_BUILD_PTX_DIR.unwrap_or("<not built with cuda>").to_string(),
    ))
}

fn read_env_dir(var: &'static str) -> Result<Option<PathBuf>, PtxAuthorityError> {
    let Ok(raw) = std::env::var(var) else {
        return Ok(None);
    };
    let value = raw.trim();
    if value.is_empty() {
        return Err(PtxAuthorityError::EmptyEnv { var });
    }
    let path = PathBuf::from(value);
    if !path.is_dir() {
        return Err(PtxAuthorityError::EnvNotADirectory {
            var,
            value: value.to_string(),
        });
    }
    Ok(Some(path.canonicalize().map_err(|e| PtxAuthorityError::Io {
        path: path.clone(),
        error: e.to_string(),
    })?))
}

fn contains_ptx(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .any(|e| e.path().extension().and_then(|x| x.to_str()) == Some("ptx"))
        })
        .unwrap_or(false)
}

fn build(dir: PathBuf, source: PtxAuthoritySource) -> Result<PtxAuthority, PtxAuthorityError> {
    let dir = dir.canonicalize().map_err(|e| PtxAuthorityError::Io {
        path: dir.clone(),
        error: e.to_string(),
    })?;
    let (kernel_set_sha256, ptx_count) = kernel_set_sha256(&dir)?;
    Ok(PtxAuthority {
        dir,
        source,
        kernel_set_sha256,
        ptx_count,
    })
}

/// sha256 over `"<sha256hex>  <name>\n"` lines of every `*.ptx` in `dir`, sorted by name.
pub fn kernel_set_sha256(dir: &Path) -> Result<(String, usize), PtxAuthorityError> {
    let io = |e: std::io::Error| PtxAuthorityError::Io {
        path: dir.to_path_buf(),
        error: e.to_string(),
    };
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .map_err(io)?
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            (name.ends_with(".ptx") && e.path().is_file()).then_some(name)
        })
        .collect();
    names.sort();
    let mut set = Sha256::new();
    for name in &names {
        let bytes = std::fs::read(dir.join(name)).map_err(|e| PtxAuthorityError::Io {
            path: dir.join(name),
            error: e.to_string(),
        })?;
        let file_hex = hex_lower(&Sha256::digest(&bytes));
        set.update(format!("{file_hex}  {name}\n").as_bytes());
    }
    Ok((hex_lower(&set.finalize()), names.len()))
}

fn hex_lower(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_set_matches_store_definition() {
        let dir = std::env::temp_dir().join(format!("ptx_authority_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("b.ptx"), b"B").unwrap();
        std::fs::write(dir.join("a.ptx"), b"A").unwrap();
        std::fs::write(dir.join("a.ptx.sha256"), b"ignored").unwrap();
        let (hash, count) = kernel_set_sha256(&dir).unwrap();
        let a = hex_lower(&Sha256::digest(b"A"));
        let b = hex_lower(&Sha256::digest(b"B"));
        let expected = hex_lower(&Sha256::digest(format!("{a}  a.ptx\n{b}  b.ptx\n").as_bytes()));
        assert_eq!(count, 2);
        assert_eq!(hash, expected);
        let auth = build(dir.clone(), PtxAuthoritySource::EnvPrismPtxDir).unwrap();
        assert!(auth.file("a.ptx").is_ok());
        assert!(matches!(auth.file("c.ptx"), Err(PtxAuthorityError::MissingFile { .. })));
        assert!(matches!(auth.file("../a.ptx"), Err(PtxAuthorityError::InvalidName(_))));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
