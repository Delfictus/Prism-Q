use crate::binary_runtime::{PcError, PcResult};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fs;
use std::path::{Path, PathBuf};

pub const CUDA_OPCODE_COVERAGE_PATH: &str = "build/cuda/cuda_opcode_coverage.json";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CudaOpcodeEvidence {
    pub opcode: String,
    pub status: String,
    pub ffi_error_code: i32,
    pub cuda_status: String,
    pub evidence_hash_blake3: String,
    pub details: serde_json::Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CudaOpcodeCoverageReport {
    pub status: String,
    pub required_opcode_count: usize,
    pub passed_required_opcode_count: usize,
    pub cuda_opcode_coverage: serde_json::Value,
    pub opcodes: Vec<CudaOpcodeEvidence>,
    pub manifest_signature_status_accepted_for_blocker_01: String,
}

pub fn opcode_evidence(
    opcode: &str,
    ffi_error_code: i32,
    cuda_status: String,
    passed: bool,
    details: serde_json::Value,
) -> CudaOpcodeEvidence {
    let evidence_hash_blake3 = blake3::hash(
        serde_json::to_vec(&json!({
            "opcode": opcode,
            "ffi_error_code": ffi_error_code,
            "cuda_status": cuda_status,
            "passed": passed,
            "details": details,
        }))
        .unwrap_or_default()
        .as_slice(),
    )
    .to_hex()
    .to_string();
    CudaOpcodeEvidence {
        opcode: opcode.to_string(),
        status: if passed { "PASS" } else { "FAIL" }.to_string(),
        ffi_error_code,
        cuda_status,
        evidence_hash_blake3,
        details,
    }
}

pub fn coverage_report(opcodes: Vec<CudaOpcodeEvidence>) -> CudaOpcodeCoverageReport {
    let required = [
        "TENSOR.COPY",
        "TENSOR.REDUCE",
        "EVENT.INTEGRATE",
        "NANL.BIN_PACK",
        "NANL.Q4_VALIDATE",
        "DRF.FIRE",
    ];
    let passed_required = required
        .iter()
        .filter(|required_opcode| {
            opcodes
                .iter()
                .any(|row| row.opcode == **required_opcode && row.status.as_str() == "PASS")
        })
        .count();
    let status = if passed_required == required.len() {
        "PASS"
    } else {
        "FAIL"
    };
    let mut coverage = serde_json::Map::new();
    for required_opcode in required {
        let value = opcodes
            .iter()
            .find(|row| row.opcode == required_opcode)
            .map(|row| row.status.clone())
            .unwrap_or_else(|| "FAIL".to_string());
        coverage.insert(required_opcode.to_string(), json!(value));
    }
    coverage.insert(
        "DRF.RESONATE".to_string(),
        json!("NOT_REQUIRED_FOR_BLOCKER_01"),
    );
    CudaOpcodeCoverageReport {
        status: status.to_string(),
        required_opcode_count: required.len(),
        passed_required_opcode_count: passed_required,
        cuda_opcode_coverage: serde_json::Value::Object(coverage),
        opcodes,
        manifest_signature_status_accepted_for_blocker_01: "DEV_HASH_BOUND_UNSIGNED".to_string(),
    }
}

pub fn write_coverage_report(path: &Path, report: &CudaOpcodeCoverageReport) -> PcResult<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(report)?)?;
    Ok(())
}

pub fn write_default_coverage_report(report: &CudaOpcodeCoverageReport) -> PcResult<()> {
    write_coverage_report(&default_coverage_path(), report)
}

pub fn default_coverage_path() -> PathBuf {
    if let Some(path) = std::env::var_os("PRISMCODE_CUDA_OPCODE_COVERAGE") {
        return PathBuf::from(path);
    }
    if let Some(dir) = std::env::var_os("PRISMCODE_CUDA_OUTPUT_DIR") {
        return PathBuf::from(dir).join("cuda_opcode_coverage.json");
    }
    if std::env::var_os("PRISM_HERMETIC").as_deref() == Some(std::ffi::OsStr::new("1")) {
        if let Some(tmp) = std::env::var_os("PRISM_ACCEPTANCE_TMP") {
            return PathBuf::from(tmp).join("cuda/cuda_opcode_coverage.json");
        }
    }
    PathBuf::from(CUDA_OPCODE_COVERAGE_PATH)
}

pub fn ensure_coverage_pass(report: &CudaOpcodeCoverageReport) -> PcResult<()> {
    if report.status == "PASS" {
        Ok(())
    } else {
        Err(PcError::Gpu(format!(
            "CUDA required opcode coverage failed: {}/{} required opcodes passed",
            report.passed_required_opcode_count, report.required_opcode_count
        )))
    }
}
