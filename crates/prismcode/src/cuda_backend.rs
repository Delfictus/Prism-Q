use crate::binary_runtime::{PcError, PcResult};
use crate::cuda_ffi;
use crate::cuda_manifest;
use crate::cuda_planner;
use crate::cuda_operation::CudaOperation;
use serde_json::json;

const STATUS_FAIL_CLOSED_UNLINKED: &str = "FAIL_CLOSED_UNLINKED";
const STATUS_FAIL_CLOSED_RUNTIME_UNAVAILABLE: &str = "FAIL_CLOSED_RUNTIME_UNAVAILABLE";
const STATUS_FAIL_CLOSED_SELFTEST_FAILED: &str = "FAIL_CLOSED_SELFTEST_FAILED";
const STATUS_FAIL_CLOSED_UNSIGNED_KERNELS: &str = "FAIL_CLOSED_UNSIGNED_KERNELS";
const STATUS_LINKED_SIGNED_SELFTEST_READY: &str = "LINKED_SIGNED_SELFTEST_READY";
const STATUS_LINKED_DEV_HASH_BOUND_SELFTEST_READY: &str = "LINKED_DEV_HASH_BOUND_SELFTEST_READY";

pub fn cuda_runtime_report() -> serde_json::Value {
    cuda_report(false)
}

// The device-observing branch uses the existing process-local FFI operation
// mutex. Actual CUDA runtime and self-test results determine device readiness.
fn cuda_report(observe_device: bool) -> serde_json::Value {
    let manifest_before_generation = cuda_manifest::read_trust_manifest();
    let generated_manifest = if manifest_before_generation.is_ok() {
        json!({
            "status": "SKIPPED_EXISTING_TRUST_MANIFEST",
            "manifest_path": cuda_manifest::trust_manifest_path().display().to_string(),
        })
    } else {
        json!({
            "status": "NOT_GENERATED_BY_PASSIVE_METADATA",
            "manifest_path": cuda_manifest::trust_manifest_path().display().to_string(),
        })
    };
    let (abi, isa, probe) = cuda_ffi::link_probe();
    let linked = cuda_ffi::link_probe_matches();
    let (runtime_ffi_error, runtime_status) = if observe_device {
        cuda_ffi::runtime_status()
    } else {
        (-1, cuda_ffi::PrismcodeCudaRuntimeStatus::default())
    };
    let runtime_ready = linked
        && runtime_ffi_error == 0
        && runtime_status.runtime_state == cuda_ffi::CUDA_RUNTIME_LINKED_READY;
    let (selftest_ffi_error, selftest) = if runtime_ready {
        cuda_ffi::m1_selftest()
    } else {
        cuda_ffi::skipped_selftest()
    };
    let selftest_ready = runtime_ready
        && selftest_ffi_error == 0
        && selftest.status == cuda_ffi::CUDA_SELFTEST_PASS
        && selftest.tensor_copy_parity == 1
        && selftest.mismatch_count == 0;

    let manifest_path = cuda_manifest::trust_manifest_path();
    let manifest_result = cuda_manifest::read_trust_manifest();
    let sidecar_signature_verified = manifest_result.is_ok()
        && cuda_manifest::trust_manifest_sidecar_signature_verified(&manifest_path);
    let (kernel_trust, kernel_signed, dev_hash_bound, manifest_hash) =
        match manifest_result.as_ref() {
            Ok(manifest) if cuda_manifest::manifest_is_signed(manifest) => (
                "SIGNED_TRUST_MANIFEST_PRESENT",
                true,
                false,
                cuda_manifest::trust_manifest_hash_hex().ok(),
            ),
            Ok(manifest)
                if cuda_manifest::manifest_is_dev_hash_bound(manifest)
                    && sidecar_signature_verified =>
            {
                (
                    "SIGNED_TRUST_MANIFEST_PRESENT",
                    true,
                    false,
                    cuda_manifest::trust_manifest_hash_hex().ok(),
                )
            }
            Ok(manifest) if cuda_manifest::manifest_is_dev_hash_bound(manifest) => (
                "DEV_HASH_BOUND_UNSIGNED",
                false,
                true,
                cuda_manifest::trust_manifest_hash_hex().ok(),
            ),
            Ok(_) => ("UNSIGNED_OR_INCOMPLETE_MANIFEST", false, false, None),
            Err(err) => (
                cuda_manifest::manifest_error_status(err),
                false,
                false,
                None,
            ),
        };

    let execution_status = if !linked {
        STATUS_FAIL_CLOSED_UNLINKED
    } else if !observe_device {
        "NOT_OBSERVED_PASSIVE_METADATA"
    } else if !runtime_ready {
        STATUS_FAIL_CLOSED_RUNTIME_UNAVAILABLE
    } else if !selftest_ready {
        STATUS_FAIL_CLOSED_SELFTEST_FAILED
    } else if kernel_signed {
        STATUS_LINKED_SIGNED_SELFTEST_READY
    } else if dev_hash_bound {
        STATUS_LINKED_DEV_HASH_BOUND_SELFTEST_READY
    } else {
        STATUS_FAIL_CLOSED_UNSIGNED_KERNELS
    };

    let mut report = json!({
        "cuda_build_feature_enabled": cuda_ffi::cuda_feature_enabled(),
        "cuda_device_observation_performed": observe_device,
        "cuda_backend_runtime_linked": linked,
        "cuda_link_probe": if linked { format!("0x{probe:08x}") } else { "UNLINKED".to_string() },
        "cuda_abi_version": if linked { format!("0x{abi:08x}") } else { "UNLINKED".to_string() },
        "cuda_isa_version": if linked { format!("0x{isa:08x}") } else { "UNLINKED".to_string() },
        "cuda_kernel_trust": kernel_trust,
        "cuda_kernel_signed": kernel_signed,
        "cuda_kernel_dev_hash_bound": dev_hash_bound,
        "cuda_kernel_manifest_signature_verified": sidecar_signature_verified,
        "cuda_kernel_manifest": manifest_path.display().to_string(),
        "cuda_execution_status": execution_status,
        "cuda_runtime_status": cuda_ffi::runtime_status_json(runtime_ffi_error, &runtime_status),
        "cuda_m1_selftest": cuda_ffi::selftest_json(selftest_ffi_error, &selftest),
        "cuda_toolchain": cuda_manifest::toolchain_report(),
        "cuda_generated_unsigned_manifest": generated_manifest,
    });

    if let Some(hash) = manifest_hash {
        report["cuda_kernel_manifest_hash_blake3"] = json!(hash);
    }

    report
}

pub fn cuda_selftest_report() -> serde_json::Value {
    match CudaOperation::begin() {
        Ok(operation) => {
            let mut report = diagnostic_selftest_report_from_runtime(&cuda_report(true));
            report["cuda_operation"] = operation.provenance.clone();
            report
        }
        Err(error) => {
            let mut report = diagnostic_selftest_report_from_runtime(&cuda_runtime_report());
            report["cuda_execution_status"] = json!("FAIL_CLOSED_CUDA_OPERATION");
            report["cuda_operation_error"] = json!(error.to_string());
            report
        }
    }
}

/// Classify an already observed runtime report without touching CUDA. Keeping
/// this transformation pure prevents metadata-contract tests from initializing
/// a device or launching a self-test.
pub(crate) fn diagnostic_selftest_report_from_runtime(report: &serde_json::Value) -> serde_json::Value {
    json!({
        "authority_classification": "DIAGNOSTIC_SELF_TEST_NOT_NATIVE_PACKET_EXECUTION",
        "native_packet_execution": false,
        "cuda_execution_status": report["cuda_execution_status"].clone(),
        "cuda_device_observation_performed": report["cuda_device_observation_performed"].clone(),
        "cuda_backend_runtime_linked": report["cuda_backend_runtime_linked"].clone(),
        "cuda_runtime_status": report["cuda_runtime_status"].clone(),
        "cuda_m1_selftest": report["cuda_m1_selftest"].clone(),
        "cuda_kernel_trust": report["cuda_kernel_trust"].clone(),
    })
}

pub fn cuda_execution_preflight() -> PcResult<serde_json::Value> {
    if !cuda_ffi::link_probe_matches() {
        return Err(PcError::Gpu("PRISMCODE CUDA fail-closed: runtime is not linked into this binary".into()));
    }
    let operation = CudaOperation::begin()?;
    cuda_execution_preflight_in_operation(&operation)
}

fn cuda_execution_preflight_in_operation(operation: &CudaOperation) -> PcResult<serde_json::Value> {
    let mut report = cuda_report(true);
    report["cuda_operation"] = operation.provenance.clone();
    let status = report
        .get("cuda_execution_status")
        .and_then(|value| value.as_str())
        .unwrap_or(STATUS_FAIL_CLOSED_UNLINKED);

    match status {
        STATUS_LINKED_SIGNED_SELFTEST_READY | STATUS_LINKED_DEV_HASH_BOUND_SELFTEST_READY => {
            Ok(report)
        }
        STATUS_FAIL_CLOSED_UNSIGNED_KERNELS => Err(PcError::Security(
            "PRISMCODE CUDA runtime is linked and self-tested, but kernel trust manifest is missing or unsigned"
                .to_string(),
        )),
        STATUS_FAIL_CLOSED_UNLINKED => Err(PcError::Gpu(
            "PRISMCODE CUDA fail-closed: runtime is not linked into this binary".to_string(),
        )),
        STATUS_FAIL_CLOSED_RUNTIME_UNAVAILABLE => Err(PcError::Gpu(
            "PRISMCODE CUDA fail-closed: linked runtime cannot reach a usable CUDA device"
                .to_string(),
        )),
        STATUS_FAIL_CLOSED_SELFTEST_FAILED => Err(PcError::Gpu(
            "PRISMCODE CUDA fail-closed: M1 TENSOR.COPY selftest did not pass".to_string(),
        )),
        other => Err(PcError::Gpu(format!(
            "PRISMCODE CUDA fail-closed: unexpected execution status {other}"
        ))),
    }
}

pub fn execute_required_opcode_suite(
    instruction_count: u64,
    state_count: u64,
    event_count: u64,
    nanl_count: u64,
) -> PcResult<serde_json::Value> {
    let operation = CudaOperation::begin()?;
    cuda_execution_preflight_in_operation(&operation)?;

    let tensor_count = state_count.max(8).min(4096) as usize;
    let tensor_input = vec![1.0f32; tensor_count];
    let (copy_err, tensor_copy) = cuda_ffi::execute_tensor_copy(&tensor_input);
    let tensor_copy_pass = copy_err == 0 && tensor_copy == tensor_input;

    let (reduce_err, reduce_sum) = cuda_ffi::execute_tensor_reduce(&tensor_input);
    let reduce_expected = tensor_input.len() as f32;
    let tensor_reduce_pass = reduce_err == 0 && (reduce_sum - reduce_expected).abs() <= 0.001;

    let required_instruction_count = instruction_count.max(1).min(4096);
    let row_count = state_count.max(1).min(4096);
    let (event_err, integrated_events, event_hash) =
        cuda_ffi::execute_event_integrate(required_instruction_count, row_count);
    let event_integrate_pass =
        event_err == 0 && integrated_events == required_instruction_count && event_hash != 0;

    let token_count = nanl_count.max(8).min(4096) as usize;
    let tokens: Vec<u64> = (0..token_count as u64)
        .map(|index| (index & 0x3) | (((index % 17) + 1) << 2))
        .collect();
    let (nanl_err, packed) = cuda_ffi::execute_nanl_bin_pack(&tokens);
    let nanl_bin_pack_pass =
        nanl_err == 0 && packed.len() == tokens.len() && packed.iter().any(|value| *value != 0);

    let q4_len = event_count.max(8).min(4096) as usize;
    let q4_bytes: Vec<u8> = (0..q4_len)
        .map(|index| ((index as u8) << 2) ^ 0b0101_1010)
        .collect();
    let (q4_err, q4_pass, q4_hash) = cuda_ffi::execute_nanl_q4_validate(&q4_bytes);
    let nanl_q4_validate_pass = q4_err == 0 && q4_pass && q4_hash != 0;

    let resonance = vec![0.10f32, 0.90, 1.10, 0.40, 2.00, 0.75, 0.76, 0.01];
    let (drf_err, intensities, fired) = cuda_ffi::execute_drf_fire(&resonance, 0.75);
    let drf_fire_pass = drf_err == 0
        && fired == 5
        && intensities.len() == resonance.len()
        && intensities[1] > 0.0
        && intensities[2] > 0.0
        && intensities[4] > 0.0
        && intensities[5] > 0.0
        && intensities[6] > 0.0;

    let opcodes = vec![
        cuda_planner::opcode_evidence(
            "TENSOR.COPY",
            copy_err,
            cuda_ffi::status_string(copy_err),
            tensor_copy_pass,
            json!({
                "element_count": tensor_count,
                "parity": tensor_copy_pass,
            }),
        ),
        cuda_planner::opcode_evidence(
            "TENSOR.REDUCE",
            reduce_err,
            cuda_ffi::status_string(reduce_err),
            tensor_reduce_pass,
            json!({
                "element_count": tensor_count,
                "gpu_sum": reduce_sum,
                "expected_sum": reduce_expected,
            }),
        ),
        cuda_planner::opcode_evidence(
            "EVENT.INTEGRATE",
            event_err,
            cuda_ffi::status_string(event_err),
            event_integrate_pass,
            json!({
                "instruction_count": required_instruction_count,
                "row_count": row_count,
                "integrated_events": integrated_events,
                "event_hash": format!("0x{event_hash:016x}"),
            }),
        ),
        cuda_planner::opcode_evidence(
            "NANL.BIN_PACK",
            nanl_err,
            cuda_ffi::status_string(nanl_err),
            nanl_bin_pack_pass,
            json!({
                "token_count": token_count,
                "nonzero_output": packed.iter().any(|value| *value != 0),
            }),
        ),
        cuda_planner::opcode_evidence(
            "NANL.Q4_VALIDATE",
            q4_err,
            cuda_ffi::status_string(q4_err),
            nanl_q4_validate_pass,
            json!({
                "byte_count": q4_len,
                "constraint_validation": q4_pass,
                "q4_hash": format!("0x{q4_hash:016x}"),
            }),
        ),
        cuda_planner::opcode_evidence(
            "DRF.FIRE",
            drf_err,
            cuda_ffi::status_string(drf_err),
            drf_fire_pass,
            json!({
                "state_count": resonance.len(),
                "threshold": 0.75,
                "fired": fired,
            }),
        ),
    ];
    let coverage = cuda_planner::coverage_report(opcodes);
    cuda_planner::write_default_coverage_report(&coverage)?;
    cuda_planner::ensure_coverage_pass(&coverage)?;
    Ok(serde_json::to_value(coverage)?)
}
