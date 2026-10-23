//! Build script for prism-gpu
//!
//! Compiles CUDA kernels to PTX for runtime loading.
//!
//! CUDA COMPILATION:
//! - Target architecture: configurable by `PRISM_GPU_CUDA_ARCH`
//!   (falls back to `PRISMCODE_CUDA_ARCH`, then `sm_120`)
//! - Optimization level: -O3 (maximum performance)
//! - PTX output: target/ptx/<kernel_name>.ptx
//! - PrismOS beta bare-metal acceptance uses this to build a complete
//!   hardware-matched PTX bundle for the target GPU, rather than shipping a
//!   workstation-only PTX target to every machine.
//!
//! DEPENDENCIES:
//! - CUDA Toolkit 12.6 installed at /usr/local/cuda-12.6
//! - nvcc compiler in PATH or CUDA_HOME set
//!
//! SECURITY:
//! - PTX modules signed with SHA-256 for verification
//! - Signatures stored in target/ptx/<kernel_name>.ptx.sha256
//!
//! REFERENCE: PRISM GPU Plan §4.3 (PTX Compilation)

use std::collections::HashSet;
use std::env;
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::process::Command;

const GOVERNED_PTX_SCHEMA_KIND: &str = "prismos_successor_source_owned_ptx_membership";
const GOVERNED_PTX_SCHEMA_VERSION: u32 = 1;
const GOVERNED_PTX_MEMBER_COUNT: usize = 95;
/// SHA-256 of zero bytes. A governed PTX sidecar carrying this digest is a
/// false attestation and must never be produced or accepted.
const EMPTY_SHA256_HEX: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
/// PTX modules open with `//` comment lines followed by the mandatory
/// `.version` and `.target` directives; both must appear within this prefix.
const PTX_HEADER_SCAN_BYTES: usize = 8192;
const COOPERATIVE_PTX_NAMES: [&str; 3] = [
    "twin_coupling_persistent.ptx",
    "twin_persistent.ptx",
    "twin_persistent_physics.ptx",
];

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct GovernedPtxManifest {
    schema_kind: String,
    schema_version: u32,
    contract_member_count: usize,
    members: Vec<GovernedPtxMember>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct GovernedPtxMember {
    ptx: String,
    source: String,
    kind: String,
}

fn main() {
    println!("cargo:rerun-if-changed=../prism-nhs/src/cuda/prism_governor_measurement.cuh");
    println!("cargo:rerun-if-changed=../prism-nhs/src/cuda/prism_asc_decision.cuh");
    println!("cargo:rerun-if-changed=src/kernels/");
    println!("cargo:rerun-if-changed=GOVERNED_PTX_MEMBERS.json");
    println!("cargo:rerun-if-env-changed=PRISM_GPU_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=PRISMCODE_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=PRISM_GPU_NVCC");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    // Check if CUDA feature is enabled
    let cuda_enabled = env::var("CARGO_FEATURE_CUDA").is_ok();
    if !cuda_enabled {
        println!("cargo:warning=CUDA feature not enabled, skipping PTX compilation");
        return;
    }

    let cuda_arch = cuda_arch();
    let cuda_include = cuda_include_dir();
    println!("cargo:info=Using CUDA target arch: {}", cuda_arch);
    println!("cargo:rustc-env=PRISM_GPU_CUDA_TARGET_ARCH={}", cuda_arch);

    // Link cuFFT library (part of CUDA toolkit, not an external dependency).
    // Keep this configurable so PrismOS can build an appliance PTX bundle
    // against the CUDA toolkit that matches the target system instead of
    // silently binding to whichever /usr/local/cuda happens to be newest.
    if let Some(cuda_lib_dir) = cuda_lib_dir() {
        println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());
    } else {
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    }
    println!("cargo:rustc-link-lib=cufft");

    // Locate nvcc compiler
    let nvcc = find_nvcc().expect("nvcc not found. Ensure CUDA toolkit is installed.");

    println!("cargo:info=Using nvcc: {}", nvcc);

    // Create output directory
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_dir = out_dir.join("ptx");
    std::fs::create_dir_all(&ptx_dir).expect("Failed to create PTX output directory");

    // Also create target/ptx for easier access.
    //
    // CRITICAL: cargo runs build scripts with CWD = the build script's crate
    // directory, NOT the workspace root. The previous version used a relative
    // path `target/ptx`, which (since this build.rs lives in
    // `crates/prism-gpu/`) resolved to `crates/prism-gpu/target/ptx/` rather
    // than the workspace `target/ptx/`. Meanwhile `find_twin_ptx` (in
    // `crates/prism-nhs/src/twin_kernels.rs`) loads PTX from the workspace
    // `target/ptx/` first. The result was that fresh PTX builds went into
    // a directory the engine never reads, and the vendored Apr-9 bundle
    // sitting in workspace `target/ptx/` "won" against every fresh build —
    // silently. Stage 2 was the symptom that exposed this: my kernel
    // changes built cleanly but produced zero observable effect because the
    // engine was loading the stale vendored kernel.
    //
    // Fix: anchor `target_ptx_dir` to `CARGO_MANIFEST_DIR/../../target/ptx`
    // so it always lands at the workspace root regardless of where cargo
    // runs the build script from.
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let target_ptx_dir = manifest_dir.join("../../target/ptx");
    std::fs::create_dir_all(&target_ptx_dir).expect("Failed to create target/ptx directory");

    let governed = load_governed_ptx_manifest(&manifest_dir);
    let declared: HashSet<String> = governed
        .members
        .iter()
        .map(|member| member.ptx.clone())
        .collect();

    for member in &governed.members {
        let output = ptx_dir.join(&member.ptx);
        let target_output = target_ptx_dir.join(&member.ptx);
        match member.kind.as_str() {
            "standard" => compile_kernel(
                &nvcc,
                &cuda_arch,
                &cuda_include,
                &member.source,
                &output,
                &target_output,
            ),
            "cooperative" => compile_cooperative_kernel(
                &nvcc,
                &cuda_arch,
                &cuda_include,
                &member.source,
                &output,
                &target_output,
            ),
            _ => unreachable!("governed PTX kind was validated before compilation"),
        }
    }

    // Both destinations persist between Cargo invocations. Remove stale rows,
    // then fail closed unless each directory is exactly the governed 95.
    prune_unexpected_ptx(&ptx_dir, &declared);
    prune_unexpected_ptx(&target_ptx_dir, &declared);
    assert_exact_ptx_set(&ptx_dir, &declared);
    assert_exact_ptx_set(&target_ptx_dir, &declared);
    println!("cargo:info=PTX compilation completed successfully");
}

/// Finds nvcc compiler in PATH or CUDA_HOME
fn find_nvcc() -> Option<String> {
    for var in ["PRISM_GPU_NVCC", "NVCC"] {
        if let Ok(nvcc) = env::var(var) {
            let path = PathBuf::from(&nvcc);
            if path.exists() || Command::new(&nvcc).arg("--version").output().is_ok() {
                return Some(nvcc);
            }
        }
    }

    // Check CUDA_HOME environment variable
    if let Ok(cuda_home) = env::var("CUDA_HOME") {
        let nvcc_path = PathBuf::from(cuda_home).join("bin").join("nvcc");
        if nvcc_path.exists() {
            return Some(nvcc_path.to_string_lossy().to_string());
        }
    }

    // Check common CUDA installation paths
    let common_paths = vec![
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-12.6/bin/nvcc",
        "/usr/local/cuda-12/bin/nvcc",
        "/opt/cuda/bin/nvcc",
    ];

    for path in common_paths {
        if PathBuf::from(path).exists() {
            return Some(path.to_string());
        }
    }

    // Check if nvcc is in PATH
    if Command::new("nvcc").arg("--version").output().is_ok() {
        return Some("nvcc".to_string());
    }

    None
}

fn cuda_arch() -> String {
    env::var("PRISM_GPU_CUDA_ARCH")
        .or_else(|_| env::var("PRISMCODE_CUDA_ARCH"))
        .unwrap_or_else(|_| "sm_120".to_string())
}

fn cuda_include_dir() -> PathBuf {
    if let Ok(cuda_home) = env::var("CUDA_HOME") {
        let include = PathBuf::from(cuda_home).join("include");
        if include.exists() {
            return include;
        }
    }

    for path in [
        "/usr/lib/cuda/include",
        "/usr/include",
        "/usr/local/cuda/include",
        "/usr/local/cuda-12/include",
        "/usr/local/cuda-12.6/include",
    ] {
        let include = PathBuf::from(path);
        if include.join("cuda.h").exists() {
            return include;
        }
    }

    PathBuf::from("/usr/local/cuda/include")
}

fn cuda_lib_dir() -> Option<PathBuf> {
    if let Ok(cuda_home) = env::var("CUDA_HOME") {
        for suffix in ["lib64", "lib", "targets/x86_64-linux/lib"] {
            let lib_dir = PathBuf::from(&cuda_home).join(suffix);
            if lib_dir.join("libcufft.so").exists() {
                return Some(lib_dir);
            }
        }
    }

    for path in [
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/usr/local/cuda-12/lib64",
        "/usr/local/cuda-12.6/lib64",
        "/usr/local/cuda-13.2/targets/x86_64-linux/lib",
    ] {
        let lib_dir = PathBuf::from(path);
        if lib_dir.join("libcufft.so").exists() {
            return Some(lib_dir);
        }
    }

    None
}

/// Compiles a CUDA kernel to PTX
///
/// # Arguments
/// * `nvcc` - Path to nvcc compiler
/// * `source` - Path to .cu source file
/// * `output` - Path to output .ptx file (in OUT_DIR)
/// * `target_output` - Path to copy .ptx file (in target/ptx)
fn compile_kernel(
    nvcc: &str,
    cuda_arch: &str,
    cuda_include: &Path,
    source: &str,
    output: &PathBuf,
    target_output: &PathBuf,
) {
    println!("cargo:info=Compiling {} -> {}", source, output.display());
    prepare_fresh_ptx_output(output);

    // Match the actual native Biprism diagnostic compiler contract. Preserve
    // the existing language flags for every other governed module.
    let language_flags: &[&str] = if source == "src/kernels/nhs_biprism_native.cu" {
        &["-std=c++17", "-lineinfo"]
    } else {
        &["--expt-relaxed-constexpr"]
    };

    let status = Command::new(nvcc)
        .arg("--ptx") // Generate PTX (not binary)
        .arg("-o")
        .arg(output)
        .arg(source)
        // Target architecture. --ptx mode only supports one architecture,
        // so PrismOS package builds must compile the complete kernel bundle
        // for the target machine, e.g. PRISM_GPU_CUDA_ARCH=sm_86 for the
        // RTX 3060 laptop.
        .arg(format!("-arch={}", cuda_arch))
        // Optimization flags
        .arg("-O3") // Maximum optimization
        // ── Phase 0 (Bug C fix): --use_fast_math intentionally REMOVED ──
        //
        // --use_fast_math implies:
        //   • --ftz=true       (flush denormals to zero)
        //   • --prec-div=false (approximate division)
        //   • --prec-sqrt=false (approximate sqrt)
        //   • --fmad=true      (fused multiply-add)
        //
        // These flags caused PTX codegen drift between freshly-built and
        // vendored PTX bundles. Specifically, `protocol_director.ptx` from
        // a fresh build with --use_fast_math used `div.approx.ftz.f32` /
        // `sub.ftz.f32` / `fma.rn.ftz.f32` instead of the IEEE-correct
        // `div.rn.f32` / `sub.f32` / `fma.rn.f32` opcodes used in the
        // vendored bundle. When the engine computed small phase deltas
        // near zero, FTZ flushed them to zero → division produced inf or
        // NaN → integer cast → out-of-bounds index → SIGSEGV in the
        // Director kernel.
        //
        // The bootstrap commit (24c6cdb1) worked around this by vendoring
        // a known-good PTX bundle in vendor/working_ptx_2026-04-10/ and
        // having scripts/prism-validate-and-run.sh seed target/ptx/ from
        // it when freshly-built protocol_director.ptx was missing.
        //
        // Removing --use_fast_math here makes fresh builds match the
        // vendored bundle's IEEE semantics. The vendored bundle becomes
        // unnecessary and can be retired in a follow-up commit once this
        // change is validated against the bootstrap signature gate.
        //
        // Performance impact: minimal. PRISM-4D's hot kernels (Physics,
        // multi_lif) are dominated by memory bandwidth, not transcendental
        // throughput. The IEEE-correct division and sub instructions cost
        // ~5% more cycles than their fast-math counterparts but the
        // bottleneck is L2/HBM, not the ALU.
        //
        // If a future kernel genuinely needs fast-math semantics for
        // performance, it should use the --use_fast_math equivalent
        // intrinsics directly (`__fdividef`, `__fadd_rz`, etc.) on a
        // per-call-site basis rather than enabling it globally.
        .arg("--restrict") // Enable restrict keyword optimization
        // Include paths
        .arg(format!("-I{}", cuda_include.display()))
        .arg("-Isrc/kernels") // For shared headers (protocol_state.cuh)
        .arg("-include")
        .arg("stdint.h")
        // Warning flags
        .arg("-Xptxas=-v") // Verbose PTX assembly info (shows register usage)
        .args(language_flags)
        .status()
        .expect("Failed to execute nvcc");

    if !status.success() {
        panic!("nvcc compilation failed for {}", source);
    }

    // Validate the emitted module, mirror it into target/ptx as a fresh inode,
    // and sign both surfaces so each is independently byte-verifiable.
    install_governed_ptx(source, output, target_output);

    println!("cargo:info=PTX compiled: {}", target_output.display());
}

/// Compiles a CUDA cooperative kernel to PTX (requires -rdc=true for grid.sync())
fn compile_cooperative_kernel(
    nvcc: &str,
    cuda_arch: &str,
    cuda_include: &Path,
    source: &str,
    output: &PathBuf,
    target_output: &PathBuf,
) {
    println!(
        "cargo:info=Compiling (cooperative) {} -> {}",
        source,
        output.display()
    );
    prepare_fresh_ptx_output(output);

    let status = Command::new(nvcc)
        .arg("--ptx")
        .arg("-o")
        .arg(output)
        .arg(source)
        .arg(format!("-arch={}", cuda_arch))
        .arg("-rdc=true") // relocatable device code for grid.sync()
        .arg("-O3")
        // --use_fast_math intentionally REMOVED — see compile_kernel
        // above for the full rationale (Phase 0 / Bug C fix). Same fix
        // applies here for cooperative kernels (twin_persistent_physics
        // and friends): IEEE-correct codegen prevents the PTX drift
        // that caused SIGSEGV when small phase deltas got flushed to
        // zero by FTZ semantics.
        .arg("--restrict")
        .arg(format!("-I{}", cuda_include.display()))
        .arg("-Isrc/kernels") // For shared headers (protocol_state.cuh)
        .arg("-include")
        .arg("stdint.h")
        .arg("-Xptxas=-v")
        .arg("--expt-relaxed-constexpr")
        .status()
        .expect("Failed to execute nvcc");

    if !status.success() {
        panic!("nvcc cooperative compilation failed for {}", source);
    }

    install_governed_ptx(source, output, target_output);
    println!(
        "cargo:info=PTX compiled (cooperative): {}",
        target_output.display()
    );
}

fn load_governed_ptx_manifest(manifest_dir: &Path) -> GovernedPtxManifest {
    let governed: GovernedPtxManifest =
        serde_json::from_str(include_str!("GOVERNED_PTX_MEMBERS.json"))
            .expect("invalid governed PTX membership JSON");

    assert_eq!(
        governed.schema_kind, GOVERNED_PTX_SCHEMA_KIND,
        "governed PTX schema_kind mismatch"
    );
    assert_eq!(
        governed.schema_version, GOVERNED_PTX_SCHEMA_VERSION,
        "governed PTX schema_version mismatch"
    );
    assert_eq!(
        governed.contract_member_count, GOVERNED_PTX_MEMBER_COUNT,
        "governed PTX contract_member_count mismatch"
    );
    assert_eq!(
        governed.members.len(),
        GOVERNED_PTX_MEMBER_COUNT,
        "governed PTX member list must contain exactly 95 rows"
    );

    let canonical_manifest_dir = manifest_dir
        .canonicalize()
        .expect("failed to canonicalize prism-gpu manifest directory");
    let kernel_root = canonical_manifest_dir.join("src/kernels");
    let mut ptx_names = HashSet::with_capacity(GOVERNED_PTX_MEMBER_COUNT);
    let mut source_paths = HashSet::with_capacity(GOVERNED_PTX_MEMBER_COUNT);
    let mut cooperative_names = HashSet::new();
    let mut previous_ptx: Option<&str> = None;

    for member in &governed.members {
        let ptx_path = Path::new(&member.ptx);
        assert!(
            ptx_path.components().count() == 1
                && matches!(ptx_path.components().next(), Some(Component::Normal(_)))
                && ptx_path.extension().and_then(|value| value.to_str()) == Some("ptx"),
            "governed PTX name must be a single .ptx basename: {}",
            member.ptx
        );
        assert!(
            ptx_names.insert(member.ptx.clone()),
            "duplicate governed PTX name: {}",
            member.ptx
        );
        if let Some(previous) = previous_ptx {
            assert!(
                previous < member.ptx.as_str(),
                "governed PTX rows must be strictly sorted: {} then {}",
                previous,
                member.ptx
            );
        }
        previous_ptx = Some(&member.ptx);

        let source_rel = Path::new(&member.source);
        assert!(
            !source_rel.as_os_str().is_empty()
                && source_rel
                    .components()
                    .all(|component| matches!(component, Component::Normal(_)))
                && source_rel.extension().and_then(|value| value.to_str()) == Some("cu")
                && source_rel.starts_with("src/kernels"),
            "governed CUDA source must be a relative .cu path under src/kernels: {}",
            member.source
        );
        let source_path = canonical_manifest_dir.join(source_rel);
        let canonical_source = source_path.canonicalize().unwrap_or_else(|error| {
            panic!(
                "governed CUDA source does not resolve: {}: {}",
                source_path.display(),
                error
            )
        });
        assert!(
            canonical_source.starts_with(&kernel_root) && canonical_source.is_file(),
            "governed CUDA source escapes src/kernels or is not a file: {}",
            canonical_source.display()
        );
        assert!(
            source_paths.insert(canonical_source),
            "duplicate governed CUDA source: {}",
            member.source
        );

        match member.kind.as_str() {
            "standard" => {}
            "cooperative" => {
                cooperative_names.insert(member.ptx.as_str());
            }
            other => panic!("unsupported governed PTX kind {} for {}", other, member.ptx),
        }
    }

    let expected_cooperative: HashSet<&str> = COOPERATIVE_PTX_NAMES.into_iter().collect();
    assert_eq!(
        cooperative_names, expected_cooperative,
        "governed cooperative PTX membership mismatch"
    );
    assert_eq!(
        governed
            .members
            .iter()
            .filter(|member| member.kind == "standard")
            .count(),
        92,
        "governed PTX manifest must contain exactly 92 standard kernels"
    );

    governed
}

fn prune_unexpected_ptx(dir: &Path, expected: &HashSet<String>) {
    for entry in std::fs::read_dir(dir).expect("failed to read PTX output directory") {
        let entry = entry.expect("failed to read PTX output entry");
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if name.contains(".ptx.tmp.") || name.contains(".ptx.sha256.tmp.") {
            // Leftover from an interrupted install_bytes_fresh(); never governed.
            std::fs::remove_file(&path).unwrap_or_else(|error| {
                panic!(
                    "failed to prune PTX temp file {}: {}",
                    path.display(),
                    error
                )
            });
            println!(
                "cargo:warning=Pruned interrupted PTX temp file: {}",
                path.display()
            );
            continue;
        }
        let ptx_name = if let Some(base) = name.strip_suffix(".sha256") {
            if !base.ends_with(".ptx") {
                continue;
            }
            base
        } else if name.ends_with(".ptx") {
            name
        } else {
            continue;
        };
        if !expected.contains(ptx_name) {
            std::fs::remove_file(&path).unwrap_or_else(|error| {
                panic!("failed to prune stale PTX {}: {}", path.display(), error)
            });
            println!(
                "cargo:warning=Pruned stale PTX artifact: {}",
                path.display()
            );
        }
    }
}

fn assert_exact_ptx_set(dir: &Path, expected: &HashSet<String>) {
    let mut actual = HashSet::new();
    for entry in std::fs::read_dir(dir).expect("failed to read PTX output directory") {
        let entry = entry.expect("failed to read PTX output entry");
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if !name.ends_with(".ptx") {
            continue;
        }
        assert!(
            entry
                .file_type()
                .expect("failed to stat PTX output")
                .is_file(),
            "governed PTX output is not a regular file: {}",
            path.display()
        );
        // Fail closed on empty, headerless, unsigned, or mis-signed modules.
        // Before this gate existed, 16 governed kernels shipped as signed
        // 0-byte files (digest e3b0c442...) and the set check still passed.
        let bytes = read_valid_ptx(&path, "governed PTX set gate");
        let digest = sha256_hex(&bytes);
        let sidecar = sidecar_path(&path);
        let signed = std::fs::read_to_string(&sidecar).unwrap_or_else(|error| {
            panic!(
                "governed PTX has no signature sidecar: {}: {}",
                sidecar.display(),
                error
            )
        });
        assert_eq!(
            signed.trim(),
            digest,
            "governed PTX signature sidecar does not match file bytes: {}",
            sidecar.display()
        );
        assert_unshared(&path);
        actual.insert(name.to_string());
    }
    assert_eq!(
        &actual,
        expected,
        "PTX directory does not exactly match governed membership: {}",
        dir.display()
    );
}

/// Unlinks any pre-existing module, sidecar, or temp file at `output` before
/// nvcc runs, so nvcc always creates a fresh, unshared inode.
///
/// ROOT CAUSE (2026-09-21, 16 governed kernels emitted as 0 bytes): nvcc
/// writes `-o <path>` in place (O_TRUNC on the existing inode). For those 16
/// kernels, OUT_DIR/ptx/<k>.ptx, workspace target/ptx/<k>.ptx and 14-34
/// external copies had been hard-linked onto ONE inode. nvcc rewrote that
/// inode with valid PTX (exit 0), then `std::fs::copy(output, target_output)`
/// opened the destination with O_TRUNC — the same inode — truncating the
/// module to 0 bytes before copying 0 bytes, and the signer then attested
/// the empty digest. Unlinking first breaks the share on the nvcc side;
/// `install_bytes_fresh` never opens a destination for writing at all.
fn prepare_fresh_ptx_output(output: &Path) {
    remove_if_exists(output);
    remove_if_exists(&sidecar_path(output));
}

fn remove_if_exists(path: &Path) {
    match std::fs::remove_file(path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => panic!(
            "failed to remove stale PTX artifact {}: {}",
            path.display(),
            error
        ),
    }
}

fn sidecar_path(ptx_path: &Path) -> PathBuf {
    ptx_path.with_extension("ptx.sha256")
}

fn temp_path(path: &Path) -> PathBuf {
    let mut name = path
        .file_name()
        .expect("PTX artifact path has no file name")
        .to_os_string();
    name.push(format!(".tmp.{}", std::process::id()));
    path.with_file_name(name)
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(bytes))
}

/// Reads a governed PTX module and rejects anything that is not a non-empty
/// regular file opening with the mandatory `.version` / `.target` header.
fn read_valid_ptx(path: &Path, context: &str) -> Vec<u8> {
    let metadata = std::fs::symlink_metadata(path).unwrap_or_else(|error| {
        panic!(
            "{}: governed PTX output missing: {}: {}",
            context,
            path.display(),
            error
        )
    });
    assert!(
        metadata.is_file(),
        "{}: governed PTX output is not a regular file: {}",
        context,
        path.display()
    );
    let bytes = std::fs::read(path).unwrap_or_else(|error| {
        panic!(
            "{}: failed to read governed PTX output {}: {}",
            context,
            path.display(),
            error
        )
    });
    assert!(
        !bytes.is_empty(),
        "{}: governed PTX output is empty (0 bytes): {}",
        context,
        path.display()
    );
    assert_ptx_header(&bytes, path, context);
    bytes
}

fn assert_ptx_header(bytes: &[u8], path: &Path, context: &str) {
    let head = String::from_utf8_lossy(&bytes[..bytes.len().min(PTX_HEADER_SCAN_BYTES)]);
    let mut first_directive_is_version = false;
    let mut saw_target = false;
    for line in head.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("//") {
            continue;
        }
        if !first_directive_is_version {
            first_directive_is_version = line.starts_with(".version ");
            if !first_directive_is_version {
                break;
            }
            continue;
        }
        if line.starts_with(".target ") {
            saw_target = true;
            break;
        }
    }
    assert!(
        first_directive_is_version && saw_target,
        "{}: governed PTX output lacks a PTX header (.version then .target within the first {} bytes): {} ({} bytes)",
        context,
        PTX_HEADER_SCAN_BYTES,
        path.display(),
        bytes.len()
    );
}

#[cfg(unix)]
fn assert_unshared(path: &Path) {
    use std::os::unix::fs::MetadataExt;
    let nlink = std::fs::metadata(path)
        .unwrap_or_else(|error| panic!("failed to stat {}: {}", path.display(), error))
        .nlink();
    assert_eq!(
        nlink,
        1,
        "governed PTX artifact is hard-linked ({} links); refusing to attest a shared inode: {}",
        nlink,
        path.display()
    );
}

#[cfg(not(unix))]
fn assert_unshared(_path: &Path) {}

/// Installs `bytes` at `dest` as a fresh inode: write a sibling temp file,
/// fsync, then rename over `dest`. rename() replaces only the directory
/// entry, so a pre-existing hard link of `dest` keeps its old inode and
/// content untouched, and `dest` is never opened with O_TRUNC.
fn install_bytes_fresh(dest: &Path, bytes: &[u8]) {
    let tmp = temp_path(dest);
    remove_if_exists(&tmp);
    {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
            .unwrap_or_else(|error| panic!("failed to create {}: {}", tmp.display(), error));
        file.write_all(bytes)
            .unwrap_or_else(|error| panic!("failed to write {}: {}", tmp.display(), error));
        file.sync_all()
            .unwrap_or_else(|error| panic!("failed to sync {}: {}", tmp.display(), error));
    }
    std::fs::rename(&tmp, dest).unwrap_or_else(|error| {
        panic!(
            "failed to rename {} -> {}: {}",
            tmp.display(),
            dest.display(),
            error
        )
    });
}

/// Validates the module nvcc emitted at `output`, mirrors it to
/// `target_output` as a fresh inode, verifies the mirror byte-for-byte, and
/// signs both surfaces. Every step fails closed.
fn install_governed_ptx(source: &str, output: &Path, target_output: &Path) {
    let bytes = read_valid_ptx(output, &format!("nvcc output for {}", source));
    let digest = sha256_hex(&bytes);
    assert_ne!(
        digest, EMPTY_SHA256_HEX,
        "refusing to sign the empty digest for {}",
        source
    );
    assert_unshared(output);

    install_bytes_fresh(target_output, &bytes);
    let mirrored = read_valid_ptx(target_output, &format!("target/ptx mirror of {}", source));
    assert_eq!(
        sha256_hex(&mirrored),
        digest,
        "target/ptx mirror diverges from OUT_DIR PTX for {}: {}",
        source,
        target_output.display()
    );
    assert_unshared(target_output);

    write_ptx_signature(output, &digest);
    write_ptx_signature(target_output, &digest);
}

/// Writes the SHA-256 signature sidecar (bare lowercase hex, no newline —
/// the format every existing consumer reads) beside `ptx_path`, as a fresh
/// inode so a hard-linked stale sidecar is never rewritten in place.
fn write_ptx_signature(ptx_path: &Path, digest_hex: &str) {
    let sig_path = sidecar_path(ptx_path);
    install_bytes_fresh(&sig_path, digest_hex.as_bytes());
    println!("cargo:info=PTX signature: {}", sig_path.display());
}
