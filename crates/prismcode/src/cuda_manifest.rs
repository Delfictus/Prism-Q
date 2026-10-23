use crate::binary_runtime::{PcError, PcResult};
use crate::manifest_signature;
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

const GOVERNED_PTX_SCHEMA_KIND: &str = "prismos_successor_source_owned_ptx_membership";
const GOVERNED_PTX_SCHEMA_VERSION: u32 = 1;
const GOVERNED_PTX_MEMBER_COUNT: usize = 95;
const COOPERATIVE_PTX_NAMES: [&str; 3] = [
    "twin_coupling_persistent.ptx",
    "twin_persistent.ptx",
    "twin_persistent_physics.ptx",
];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GovernedPtxManifest {
    schema_kind: String,
    schema_version: u32,
    contract_member_count: usize,
    members: Vec<GovernedPtxMember>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GovernedPtxMember {
    ptx: String,
    source: String,
    kind: String,
}

struct GovernedPtxEvidence {
    source_manifest_path: PathBuf,
    source_manifest_sha256: String,
    per_kernel_sha256: BTreeMap<String, String>,
    ptx_hashes: Vec<serde_json::Value>,
    fused_sha256: String,
}

pub fn trust_manifest_path() -> PathBuf {
    std::env::var_os("PRISMCODE_CUDA_TRUST_MANIFEST")
        .map(PathBuf::from)
        .or_else(bundled_trust_manifest_path)
        .or_else(|| {
            if std::env::var_os("PRISM_HERMETIC").as_deref() == Some(std::ffi::OsStr::new("1")) {
                std::env::var_os("PRISM_ACCEPTANCE_TMP")
                    .map(PathBuf::from)
                    .map(|root| root.join("cuda/prismcode_cuda_manifest.json"))
            } else {
                None
            }
        })
        .unwrap_or_else(|| PathBuf::from("build/cuda/prismcode_cuda_manifest.json"))
}

pub fn read_trust_manifest() -> PcResult<serde_json::Value> {
    Ok(serde_json::from_slice(&fs::read(trust_manifest_path())?)?)
}

pub fn manifest_is_signed(manifest: &serde_json::Value) -> bool {
    manifest
        .get("cuda_kernel_signed")
        .and_then(|value| value.as_bool())
        == Some(true)
        && manifest
            .get("signature_status")
            .and_then(|value| value.as_str())
            == Some("SIGNED")
        && manifest
            .get("kernel_hashes")
            .and_then(|value| value.as_array())
            .map(|rows| !rows.is_empty())
            .unwrap_or(false)
}

pub fn trust_manifest_sidecar_signature_verified(manifest_path: &Path) -> bool {
    trust_manifest_signature_path(manifest_path)
        .and_then(|sig| manifest_signature::verify_artifact_report(manifest_path, &sig).ok())
        .and_then(|report| {
            let status_ok = report.get("status").and_then(|value| value.as_str()) == Some("PASS");
            let signature_ok = report
                .get("signature_status")
                .and_then(|value| value.as_str())
                == Some("SIGNATURE_VERIFIED");
            (status_ok && signature_ok).then_some(())
        })
        .is_some()
}

pub fn manifest_is_dev_hash_bound(manifest: &serde_json::Value) -> bool {
    manifest
        .get("signature_status")
        .and_then(|value| value.as_str())
        == Some("DEV_HASH_BOUND_UNSIGNED")
        && manifest
            .get("kernel_source_hashes")
            .and_then(|value| value.as_array())
            .map(|rows| !rows.is_empty())
            .unwrap_or(false)
        && manifest
            .get("host_library_hash")
            .and_then(|value| value.as_str())
            .map(|value| !value.trim().is_empty())
            .unwrap_or(false)
}

pub fn trust_manifest_hash_hex() -> PcResult<String> {
    hash_file_hex(&trust_manifest_path())
}

pub fn generated_unsigned_manifest() -> PcResult<serde_json::Value> {
    let governed_ptx = governed_ptx_evidence()?;
    let mut kernel_source_hashes = Vec::new();
    for path in cuda_manifest_paths()? {
        if path.exists() {
            let canonical_path = canonical_recorded_path(&path, "QNANL CUDA source")?;
            kernel_source_hashes.push(json!({
                "path": path_string(&canonical_path),
                "blake3": hash_file_hex(&canonical_path)?,
            }));
        }
    }
    let host_library_path = std::env::var("PRISMCODE_CUDA_STATIC_LIB")
        .ok()
        .map(PathBuf::from)
        .filter(|path| path.exists())
        .map(|path| canonical_recorded_path(&path, "PrismCode CUDA static library"))
        .transpose()?;
    let host_library_hash = match host_library_path.as_ref() {
        Some(path) => hash_file_hex(path)?,
        None => "EVIDENCE_NOT_FOUND".to_string(),
    };
    let host_library_artifacts = host_library_path
        .as_ref()
        .map(|path| json!([{"path": path_string(path), "blake3": host_library_hash}]))
        .unwrap_or_else(|| json!([]));
    let toolchain = toolchain_report();
    let nvcc_path =
        command_stdout("which", &["nvcc"]).unwrap_or_else(|| "EVIDENCE_NOT_FOUND".to_string());
    reject_old_root_text(&nvcc_path, "nvcc_path")?;
    let build_timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);

    let target_arch = option_env!("PRISMCODE_CUDA_TARGET_ARCHES")
        .map(str::to_string)
        .or_else(|| std::env::var("PRISMCODE_CUDA_ARCHES").ok())
        .or_else(|| std::env::var("PRISMCODE_CUDA_ARCH").ok())
        .unwrap_or_else(|| "EVIDENCE_NOT_FOUND".to_string());

    let manifest = json!({
        "manifest_schema": "prismcode.cuda.kernel_trust.v0.1",
        "release_target": "PRISMCODE v0.1-alpha -> rc1 A1 CUDA Runtime M1",
        "cuda_driver_version": serde_json::Value::Null,
        "cuda_runtime_version": serde_json::Value::Null,
        "cuda_toolkit_version": toolchain["nvcc"]["stdout"].clone(),
        "nvcc_path": nvcc_path,
        "nvcc_version": toolchain["nvcc"]["stdout"].clone(),
        "gpu_name": toolchain["nvidia_smi"]["stdout"].clone(),
        "gpu_uuid": serde_json::Value::Null,
        "compute_capability": toolchain["nvidia_smi"]["stdout"].clone(),
        "target_arch": target_arch,
        "cuda_abi_version": "0x00010000",
        "cuda_isa_version": "0x00010000",
        "required_runtime_surfaces": [
            "link_probe",
            "runtime_status",
            "m1_selftest",
            "tensor.copy.parity",
            "tensor.reduce.parity",
            "event.integrate",
            "nanl.bin_pack",
            "nanl.q4_validate",
            "drf.fire"
        ],
        "kernel_artifacts": host_library_artifacts.clone(),
        "kernel_source_hashes": kernel_source_hashes.clone(),
        "kernel_hashes": kernel_source_hashes,
        "ptx_hashes": governed_ptx.ptx_hashes,
        "governed_ptx_membership": {
            "source_owned_manifest": {
                "path": path_string(&governed_ptx.source_manifest_path),
                "sha256": governed_ptx.source_manifest_sha256,
            },
            "count": GOVERNED_PTX_MEMBER_COUNT,
            "per_kernel_sha256": governed_ptx.per_kernel_sha256,
            "fused_sha256": governed_ptx.fused_sha256,
        },
        "cubin_or_shared_library_hashes": host_library_artifacts,
        "host_library_hash": host_library_hash,
        "build_timestamp": build_timestamp,
        "psk_mode": "DEV_PSK_ACTIVE",
        "cuda_kernel_signed": false,
        "signature_status": "DEV_HASH_BOUND_UNSIGNED",
    });
    reject_old_root_text(
        &serde_json::to_string(&manifest)?,
        "generated CUDA trust manifest",
    )?;
    Ok(manifest)
}

pub fn write_generated_unsigned_manifest() -> PcResult<serde_json::Value> {
    if bundled_trust_manifest_path().is_some() && trust_manifest_path().exists() {
        let manifest = read_trust_manifest()?;
        validate_existing_governed_ptx_evidence(&manifest)?;
        return Ok(manifest);
    }
    let manifest = generated_unsigned_manifest()?;
    let path = trust_manifest_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&path, serde_json::to_vec_pretty(&manifest)?)?;
    Ok(manifest)
}

pub fn toolchain_report() -> serde_json::Value {
    let nvcc = command_report("nvcc", &["--version"]);
    let nvidia_smi = json!({"command":"nvidia-smi", "status":"NOT_OBSERVED_PASSIVE_METADATA",
        "available":null, "success":null, "stdout":null});
    json!({
        "nvcc": nvcc,
        "nvidia_smi": nvidia_smi,
        "cuda_toolchain_available": command_success(&nvcc),
        "cuda_hardware_probe_available": null,
    })
}

fn governed_ptx_evidence() -> PcResult<GovernedPtxEvidence> {
    let prismcode_dir = canonical_recorded_path(
        Path::new(env!("CARGO_MANIFEST_DIR")),
        "prismcode crate root",
    )?;
    let crates_dir = prismcode_dir.parent().ok_or_else(|| {
        evidence_not_found(format!(
            "prismcode crate root has no parent: {}",
            prismcode_dir.display()
        ))
    })?;
    let prism_gpu_root =
        canonical_recorded_path(&crates_dir.join("prism-gpu"), "prism-gpu crate root")?;
    let source_manifest_path = canonical_recorded_path(
        &prism_gpu_root.join("GOVERNED_PTX_MEMBERS.json"),
        "governed PTX source manifest",
    )?;
    let source_manifest_bytes = fs::read(&source_manifest_path).map_err(|error| {
        evidence_not_found(format!(
            "cannot read governed PTX source manifest {}: {}",
            source_manifest_path.display(),
            error
        ))
    })?;
    let governed: GovernedPtxManifest =
        serde_json::from_slice(&source_manifest_bytes).map_err(|error| {
            evidence_not_found(format!(
                "cannot parse governed PTX source manifest {}: {}",
                source_manifest_path.display(),
                error
            ))
        })?;
    if governed.schema_kind != GOVERNED_PTX_SCHEMA_KIND
        || governed.schema_version != GOVERNED_PTX_SCHEMA_VERSION
        || governed.contract_member_count != GOVERNED_PTX_MEMBER_COUNT
        || governed.members.len() != GOVERNED_PTX_MEMBER_COUNT
    {
        return Err(evidence_not_found(format!(
            "governed PTX manifest schema/count mismatch: kind={} version={} declared={} rows={}",
            governed.schema_kind,
            governed.schema_version,
            governed.contract_member_count,
            governed.members.len()
        )));
    }

    let kernel_root =
        canonical_recorded_path(&prism_gpu_root.join("src/kernels"), "prism-gpu kernel root")?;
    let mut declared_names = BTreeSet::new();
    let mut canonical_sources = BTreeSet::new();
    let mut cooperative_names = BTreeSet::new();
    let mut previous_ptx: Option<&str> = None;

    for member in &governed.members {
        let ptx_path = Path::new(&member.ptx);
        if ptx_path.components().count() != 1
            || !matches!(ptx_path.components().next(), Some(Component::Normal(_)))
            || ptx_path.extension().and_then(|value| value.to_str()) != Some("ptx")
        {
            return Err(evidence_not_found(format!(
                "invalid governed PTX basename: {}",
                member.ptx
            )));
        }
        if !declared_names.insert(member.ptx.clone()) {
            return Err(evidence_not_found(format!(
                "duplicate governed PTX basename: {}",
                member.ptx
            )));
        }
        if previous_ptx.is_some_and(|previous| previous >= member.ptx.as_str()) {
            return Err(evidence_not_found(format!(
                "governed PTX rows are not strictly sorted at {}",
                member.ptx
            )));
        }
        previous_ptx = Some(&member.ptx);

        let source_rel = Path::new(&member.source);
        if source_rel.as_os_str().is_empty()
            || !source_rel
                .components()
                .all(|component| matches!(component, Component::Normal(_)))
            || source_rel.extension().and_then(|value| value.to_str()) != Some("cu")
            || !source_rel.starts_with("src/kernels")
        {
            return Err(evidence_not_found(format!(
                "invalid governed CUDA source path: {}",
                member.source
            )));
        }
        let canonical_source =
            canonical_recorded_path(&prism_gpu_root.join(source_rel), "governed CUDA source")?;
        if !canonical_source.starts_with(&kernel_root) || !canonical_source.is_file() {
            return Err(evidence_not_found(format!(
                "governed CUDA source escaped kernel root or is not a file: {}",
                canonical_source.display()
            )));
        }
        if !canonical_sources.insert(canonical_source) {
            return Err(evidence_not_found(format!(
                "duplicate governed CUDA source: {}",
                member.source
            )));
        }

        match member.kind.as_str() {
            "standard" => {}
            "cooperative" => {
                cooperative_names.insert(member.ptx.as_str());
            }
            other => {
                return Err(evidence_not_found(format!(
                    "unsupported governed PTX kind {} for {}",
                    other, member.ptx
                )));
            }
        }
    }

    let expected_cooperative: BTreeSet<&str> = COOPERATIVE_PTX_NAMES.into_iter().collect();
    if cooperative_names != expected_cooperative
        || governed
            .members
            .iter()
            .filter(|member| member.kind == "standard")
            .count()
            != 92
    {
        return Err(evidence_not_found(
            "governed PTX standard/cooperative membership mismatch",
        ));
    }

    let target_ptx_dir = canonical_recorded_path(
        &prismcode_dir.join("../../target/ptx"),
        "workspace governed PTX mirror",
    )?;
    let mut actual_names = BTreeSet::new();
    for entry in fs::read_dir(&target_ptx_dir).map_err(|error| {
        evidence_not_found(format!(
            "cannot read governed PTX mirror {}: {}",
            target_ptx_dir.display(),
            error
        ))
    })? {
        let entry = entry.map_err(|error| {
            evidence_not_found(format!("cannot read governed PTX mirror row: {}", error))
        })?;
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("ptx") {
            continue;
        }
        let file_type = entry.file_type().map_err(|error| {
            evidence_not_found(format!(
                "cannot stat PTX mirror row {}: {}",
                path.display(),
                error
            ))
        })?;
        if !file_type.is_file() || file_type.is_symlink() {
            return Err(evidence_not_found(format!(
                "governed PTX mirror row is not a regular file: {}",
                path.display()
            )));
        }
        let name = entry.file_name().into_string().map_err(|_| {
            evidence_not_found(format!(
                "governed PTX mirror has a non-UTF-8 name: {}",
                path.display()
            ))
        })?;
        actual_names.insert(name);
    }
    if actual_names != declared_names {
        let extra: Vec<_> = actual_names.difference(&declared_names).cloned().collect();
        let missing: Vec<_> = declared_names.difference(&actual_names).cloned().collect();
        return Err(evidence_not_found(format!(
            "workspace PTX mirror differs from governed membership: extra={:?} missing={:?}",
            extra, missing
        )));
    }

    let mut per_kernel_sha256 = BTreeMap::new();
    let mut ptx_hashes = Vec::with_capacity(GOVERNED_PTX_MEMBER_COUNT);
    for name in &declared_names {
        let path = target_ptx_dir.join(name);
        let metadata = fs::symlink_metadata(&path).map_err(|error| {
            evidence_not_found(format!(
                "cannot stat governed PTX {}: {}",
                path.display(),
                error
            ))
        })?;
        if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
            return Err(evidence_not_found(format!(
                "governed PTX is not a regular file: {}",
                path.display()
            )));
        }
        let canonical_path = canonical_recorded_path(&path, "governed PTX mirror row")?;
        if !canonical_path.starts_with(&target_ptx_dir) {
            return Err(evidence_not_found(format!(
                "governed PTX escaped workspace mirror: {}",
                canonical_path.display()
            )));
        }
        let sha256 = sha256_file_hex(&canonical_path)?;
        let blake3 = hash_file_hex(&canonical_path)?;
        per_kernel_sha256.insert(name.clone(), sha256.clone());
        ptx_hashes.push(json!({
            "name": name,
            "path": path_string(&canonical_path),
            "sha256": sha256,
            "blake3": blake3,
            "bytes": metadata.len(),
            "family": if name == "nhs_amber_fused.ptx" {
                "MD_FORCE_KERNEL_PTX"
            } else {
                "GOVERNED_PTX"
            },
            "authority_role": "SOURCE_OWNED_BUILD_OUTPUT",
            "runtime_selectable": false,
        }));
    }

    let fused_sha256 = canonical_membership_digest(&per_kernel_sha256);
    Ok(GovernedPtxEvidence {
        source_manifest_path,
        source_manifest_sha256: format!("{:x}", Sha256::digest(&source_manifest_bytes)),
        per_kernel_sha256,
        ptx_hashes,
        fused_sha256,
    })
}

pub fn validate_existing_governed_ptx_evidence(manifest: &serde_json::Value) -> PcResult<()> {
    reject_old_root_text(
        &serde_json::to_string(manifest)?,
        "existing CUDA trust manifest",
    )?;
    let current = governed_ptx_evidence()?;
    let block = manifest
        .get("governed_ptx_membership")
        .ok_or_else(|| evidence_not_found("existing trust manifest lacks governed membership"))?;
    if block.get("count").and_then(serde_json::Value::as_u64)
        != Some(GOVERNED_PTX_MEMBER_COUNT as u64)
        || block
            .get("fused_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(current.fused_sha256.as_str())
        || block
            .pointer("/source_owned_manifest/path")
            .and_then(serde_json::Value::as_str)
            != Some(path_string(&current.source_manifest_path).as_str())
        || block
            .pointer("/source_owned_manifest/sha256")
            .and_then(serde_json::Value::as_str)
            != Some(current.source_manifest_sha256.as_str())
    {
        return Err(evidence_not_found(
            "existing trust manifest governed membership identity mismatch",
        ));
    }

    let recorded_map = block
        .get("per_kernel_sha256")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            evidence_not_found("existing trust manifest lacks per_kernel_sha256 object")
        })?;
    if recorded_map.len() != GOVERNED_PTX_MEMBER_COUNT
        || current.per_kernel_sha256.iter().any(|(name, sha256)| {
            recorded_map.get(name).and_then(serde_json::Value::as_str) != Some(sha256.as_str())
        })
    {
        return Err(evidence_not_found(
            "existing trust manifest per-kernel SHA-256 map mismatch",
        ));
    }

    let ptx_rows = manifest
        .get("ptx_hashes")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| evidence_not_found("existing trust manifest lacks ptx_hashes array"))?;
    let mut row_map = BTreeMap::new();
    for row in ptx_rows {
        let name = row
            .get("name")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| evidence_not_found("existing ptx_hashes row lacks name"))?;
        let sha256 = row
            .get("sha256")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| evidence_not_found("existing ptx_hashes row lacks sha256"))?;
        if row_map
            .insert(name.to_string(), sha256.to_string())
            .is_some()
        {
            return Err(evidence_not_found(format!(
                "existing ptx_hashes contains duplicate name {}",
                name
            )));
        }
    }
    if row_map != current.per_kernel_sha256 {
        return Err(evidence_not_found(
            "existing ptx_hashes differs from governed per-kernel SHA-256 map",
        ));
    }
    Ok(())
}

fn canonical_membership_digest(per_kernel_sha256: &BTreeMap<String, String>) -> String {
    let mut digest = Sha256::new();
    for (name, sha256) in per_kernel_sha256 {
        digest.update(name.as_bytes());
        digest.update(b":");
        digest.update(sha256.as_bytes());
        digest.update(b"\n");
    }
    format!("{:x}", digest.finalize())
}

fn sha256_file_hex(path: &Path) -> PcResult<String> {
    let bytes = fs::read(path).map_err(|error| {
        evidence_not_found(format!(
            "cannot read {} for SHA-256: {}",
            path.display(),
            error
        ))
    })?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn evidence_not_found(message: impl Into<String>) -> PcError {
    PcError::Provenance(format!("EVIDENCE_NOT_FOUND:{}", message.into()))
}

/// Text that identifies the retired Prism4D-bio development tree. The match is
/// HOME-independent (`/Desktop/Prism4D-bio`) and also covers the operator's
/// configured development roots, so it never depends on the caller's `$HOME`.
fn old_prism_root_needles() -> Vec<String> {
    let mut needles = vec!["/Desktop/Prism4D-bio".to_string()];
    if let Some(roots) = crate::host_roots::dev_source_roots() {
        needles.extend(roots.into_iter().filter(|root| root.ends_with("Prism4D-bio")));
    }
    needles
}

fn reject_old_root_text(value: &str, label: &str) -> PcResult<()> {
    if let Some(old_root) = old_prism_root_needles()
        .into_iter()
        .find(|needle| value.contains(needle.as_str()))
    {
        return Err(evidence_not_found(format!(
            "{} resolves under forbidden old root {}",
            label, old_root
        )));
    }
    Ok(())
}

fn canonical_recorded_path(path: &Path, label: &str) -> PcResult<PathBuf> {
    let canonical = path.canonicalize().map_err(|error| {
        evidence_not_found(format!(
            "cannot canonicalize {} {}: {}",
            label,
            path.display(),
            error
        ))
    })?;
    reject_old_root_text(&path_string(&canonical), label)?;
    Ok(canonical)
}

fn cuda_manifest_paths() -> PcResult<Vec<PathBuf>> {
    let prismcode_dir = canonical_recorded_path(
        Path::new(env!("CARGO_MANIFEST_DIR")),
        "prismcode crate root",
    )?;
    let cuda_root = canonical_recorded_path(
        &prismcode_dir.join("../../cuda"),
        "successor CUDA source root",
    )?;
    let paths = [
        "include/prismcode_cuda.h",
        "src/prismcode_cuda.cu",
        "src/prismcode_cuda_runtime.cu",
        "src/nanl_pack.cu",
        "src/event_integrate.cu",
        "src/drf_fire.cu",
        "src/drf_resonate.cu",
        "src/tensor_ops.cu",
        "src/thermo_reduce.cu",
        "kernels/prismcode_cuda_device.cuh",
        "kernels/nanl_pack.cu",
        "kernels/event_integrate.cu",
        "kernels/drf_fire.cu",
        "kernels/drf_resonate.cu",
        "kernels/tensor_ops.cu",
        "kernels/tensor_reduce.cu",
        "kernels/nanl_bin_pack.cu",
        "kernels/nanl_q4_validate.cu",
        "kernels/thermo_reduce.cu",
    ]
    .into_iter()
    .map(|relative| cuda_root.join(relative))
    .collect::<Vec<_>>();
    for path in paths.iter().filter(|path| path.exists()) {
        let canonical_path = canonical_recorded_path(path, "QNANL CUDA source")?;
        if !canonical_path.starts_with(&cuda_root) {
            return Err(evidence_not_found(format!(
                "QNANL CUDA source escaped successor CUDA root: {}",
                canonical_path.display()
            )));
        }
    }
    Ok(paths)
}

fn bundled_trust_manifest_path() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(root) = std::env::var_os("PRISM_BUNDLE_ROOT").map(PathBuf::from) {
        candidates.push(root.join("release/prismcode_cuda_manifest.json"));
        candidates.push(root.join("release/cuda_manifest.json"));
        candidates.push(root.join("share/prismcode/cuda/prismcode_cuda_manifest.json"));
        candidates.push(root.join("share/prismcode/cuda/cuda_manifest.json"));
    }
    candidates.push(PathBuf::from(
        "/opt/prismos/share/prismcode/cuda/prismcode_cuda_manifest.json",
    ));
    candidates.push(PathBuf::from(
        "/opt/prismos/share/prismcode/cuda/cuda_manifest.json",
    ));
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join("release/prismcode_cuda_manifest.json"));
        candidates.push(cwd.join("release/cuda_manifest.json"));
    }
    candidates.into_iter().find(|path| path.exists())
}

fn trust_manifest_signature_path(manifest_path: &Path) -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("PRISMCODE_CUDA_TRUST_MANIFEST_SIG").map(PathBuf::from) {
        return path.exists().then_some(path);
    }

    let sibling_sig = manifest_path
        .parent()
        .map(|parent| parent.join("cuda_manifest.sig"));
    if let Some(sig) = sibling_sig.filter(|path| path.exists()) {
        return Some(sig);
    }

    let build_sig = PathBuf::from("build/security/cuda_manifest.sig");
    build_sig.exists().then_some(build_sig)
}

fn hash_file_hex(path: &Path) -> PcResult<String> {
    let bytes = fs::read(path)?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

fn command_report(command: &str, args: &[&str]) -> serde_json::Value {
    match Command::new(command).args(args).output() {
        Ok(output) => json!({
            "command": command,
            "available": true,
            "exit_code": output.status.code(),
            "success": output.status.success(),
            "stdout": String::from_utf8_lossy(&output.stdout).trim(),
            "stderr": String::from_utf8_lossy(&output.stderr).trim(),
        }),
        Err(err) => json!({
            "command": command,
            "available": false,
            "success": false,
            "error": err.to_string(),
        }),
    }
}

fn command_stdout(command: &str, args: &[&str]) -> Option<String> {
    Command::new(command)
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|value| !value.is_empty())
}

fn command_success(value: &serde_json::Value) -> bool {
    value
        .get("success")
        .and_then(|success| success.as_bool())
        .unwrap_or(false)
}

fn path_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

pub fn manifest_error_status(err: &PcError) -> &'static str {
    match err {
        PcError::Io(_) => "TRUST_MANIFEST_NOT_FOUND",
        _ => "TRUST_MANIFEST_INVALID",
    }
}
