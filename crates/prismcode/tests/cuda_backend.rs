#[cfg(not(feature = "cuda"))]
use prismcode::binary_runtime::PcError;
use prismcode::cuda_backend::{
    cuda_execution_preflight, cuda_runtime_report, cuda_selftest_report,
};
use prismcode::cuda_ffi::{PrismcodeCudaRuntimeStatus, PrismcodeCudaSelfTestResult};
use prismcode::cuda_manifest::{generated_unsigned_manifest, manifest_is_signed};

#[test]
fn cuda_ffi_struct_sizes_match_c_abi() {
    assert_eq!(std::mem::size_of::<PrismcodeCudaRuntimeStatus>(), 184);
    assert_eq!(std::mem::size_of::<PrismcodeCudaSelfTestResult>(), 48);
}

#[test]
#[cfg(not(feature = "cuda"))]
fn cuda_report_fail_closes_when_runtime_is_not_linked() {
    let report = cuda_runtime_report();
    assert_eq!(report["cuda_build_feature_enabled"].as_bool(), Some(false));
    assert_eq!(report["cuda_backend_runtime_linked"].as_bool(), Some(false));
    assert_eq!(
        report["cuda_execution_status"].as_str(),
        Some("FAIL_CLOSED_UNLINKED")
    );
    assert_eq!(
        report["cuda_m1_selftest"]["status"].as_str(),
        Some("SKIPPED")
    );
    assert_eq!(
        report["cuda_m1_selftest"]["tensor_copy_parity"].as_bool(),
        Some(false)
    );
}

#[test]
#[cfg(not(feature = "cuda"))]
fn cuda_preflight_rejects_unlinked_runtime_without_cpu_fallback() {
    let err = cuda_execution_preflight().expect_err("unlinked CUDA must fail closed");
    match err {
        PcError::Gpu(message) => assert!(message.contains("fail-closed")),
        other => panic!("expected GPU fail-closed error, got {other:?}"),
    }
}

#[test]
fn generated_cuda_manifest_is_unsigned_until_explicitly_signed() {
    let manifest = generated_unsigned_manifest().expect("manifest generation must hash CUDA files");
    assert_eq!(
        manifest["manifest_schema"].as_str(),
        Some("prismcode.cuda.kernel_trust.v0.1")
    );
    assert_eq!(manifest["cuda_kernel_signed"].as_bool(), Some(false));
    assert_eq!(
        manifest["signature_status"].as_str(),
        Some("DEV_HASH_BOUND_UNSIGNED")
    );
    assert!(!manifest_is_signed(&manifest));
    assert!(manifest["kernel_hashes"]
        .as_array()
        .map(|rows| rows.iter().any(|row| row["path"]
            .as_str()
            .map(|path| path.ends_with("cuda/kernels/tensor_ops.cu"))
            .unwrap_or(false)))
        .unwrap_or(false));
}

#[test]
#[cfg(not(feature = "cuda"))]
fn direct_selftest_reports_unlinked_cuda_without_requesting_a_lease() {
    let runtime = cuda_runtime_report();
    let selftest = cuda_selftest_report();
    assert_eq!(runtime["cuda_device_observation_performed"], false);
    assert_eq!(runtime["cuda_toolchain"]["nvidia_smi"]["status"], "NOT_OBSERVED_PASSIVE_METADATA");
    assert_eq!(selftest["cuda_execution_status"], "FAIL_CLOSED_UNLINKED");
    assert_eq!(selftest["cuda_operation"]["gpu_lease_required"], false);
    assert_eq!(selftest["cuda_operation"]["arbiter_invoked"], false);
    assert_eq!(selftest["cuda_operation"]["process_id"], std::process::id());
    assert!(selftest.get("diagnostic_admission").is_none());
    assert!(selftest["cuda_operation"].get("lease_id").is_none());
    assert!(selftest.get("cuda_m1_selftest").is_some());
}

#[test]
fn passive_hardware_metadata_does_not_claim_gpu_observation() {
    let report = prismcode::binary_runtime::hardware_report().unwrap();
    assert!(report["cuda_hardware_visible"].is_null());
    assert_eq!(report["cuda_hardware_observation_status"], "NOT_OBSERVED_PASSIVE_METADATA");
    assert_eq!(report["cuda_runtime_contract"]["cuda_device_observation_performed"], false);
    assert_eq!(report["cuda_runtime_contract"]["cuda_toolchain"]["nvidia_smi"]["status"], "NOT_OBSERVED_PASSIVE_METADATA");
    assert_ne!(report["cuda_status"], "CUDA_HARDWARE_NOT_VISIBLE");
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_feature_preflight_only_passes_after_link_selftest_and_trust() {
    // This is a live CUDA integration test. Passive metadata cannot predict
    // its result; check the actually observed runtime response instead.
    let result = cuda_execution_preflight();
    match result {
        Ok(report) => {
            assert!(matches!(report["cuda_execution_status"].as_str(),
                Some("LINKED_SIGNED_SELFTEST_READY" | "LINKED_DEV_HASH_BOUND_SELFTEST_READY")));
            assert_eq!(report["cuda_device_observation_performed"], true);
            assert_eq!(report["cuda_m1_selftest"]["status"], "PASS");
            assert_eq!(report["cuda_operation"]["gpu_lease_required"], false);
            assert_eq!(report["cuda_operation"]["arbiter_invoked"], false);
        }
        Err(error) => {
            let message = error.to_string();
            assert!(message.contains("runtime is not linked")
                || message.contains("cannot reach a usable CUDA device")
                || message.contains("selftest did not pass")
                || message.contains("kernel trust manifest is missing or unsigned"),
                "unexpected direct CUDA error: {message}");
        }
    }
}
