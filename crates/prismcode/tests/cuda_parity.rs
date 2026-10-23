#![cfg(feature = "cuda")]

use prismcode::binary_runtime::{
    compile_reference_json, execute_packet, verify_replay, Backend, PcError,
};
use prismcode::cuda_backend::{cuda_execution_preflight, execute_required_opcode_suite};

fn reference_json_path() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/reference/reference.json")
}

#[test]
fn explicit_cuda_executes_required_opcodes_without_cpu_fallback() {
    cuda_execution_preflight().expect("CUDA Blocker 01 requires linked dev hash-bound CUDA");
    let suite = execute_required_opcode_suite(24, 3, 2, 206)
        .expect("required CUDA opcode suite must execute");
    assert_eq!(suite["status"].as_str(), Some("PASS"));
    assert_eq!(
        suite["cuda_opcode_coverage"]["TENSOR.COPY"].as_str(),
        Some("PASS")
    );
    assert_eq!(
        suite["cuda_opcode_coverage"]["TENSOR.REDUCE"].as_str(),
        Some("PASS")
    );
    assert_eq!(
        suite["cuda_opcode_coverage"]["EVENT.INTEGRATE"].as_str(),
        Some("PASS")
    );
    assert_eq!(
        suite["cuda_opcode_coverage"]["NANL.BIN_PACK"].as_str(),
        Some("PASS")
    );
    assert_eq!(
        suite["cuda_opcode_coverage"]["NANL.Q4_VALIDATE"].as_str(),
        Some("PASS")
    );
    assert_eq!(
        suite["cuda_opcode_coverage"]["DRF.FIRE"].as_str(),
        Some("PASS")
    );
}

#[ignore = "GPU_RUNTIME_REQUIRED: needs real GPU + built PTX; run on the RTX workstation via `cargo test -- --ignored`"]
#[test]
fn cpu_cuda_execution_reports_match_for_reference_packet() {
    cuda_execution_preflight().expect("CUDA Blocker 01 requires linked dev hash-bound CUDA");
    let tmp = tempfile::tempdir().unwrap();
    let pcode = tmp.path().join("reference.pcode");
    let cpu_pcout = tmp.path().join("reference.cpu.pcout");
    let cuda_pcout = tmp.path().join("reference.cuda.pcout");
    let cpu_replay = tmp.path().join("reference.cpu.replay.json");
    let cuda_replay = tmp.path().join("reference.cuda.replay.json");

    compile_reference_json(&reference_json_path(), &pcode).expect("reference corpus must compile");
    let cpu = execute_packet(&pcode, Backend::Cpu, &cpu_pcout).expect("CPU execution must pass");
    let cuda =
        execute_packet(&pcode, Backend::Cuda, &cuda_pcout).expect("CUDA execution must pass");

    assert_eq!(cpu.backend, "cpu");
    assert_eq!(cuda.backend, "cuda");
    assert_eq!(cpu.input_packet_hash, cuda.input_packet_hash);
    assert_eq!(cpu.output_state_hash, cuda.output_state_hash);
    assert_eq!(cpu.instruction_count, cuda.instruction_count);
    assert_eq!(cpu.event_count, cuda.event_count);
    assert!(cuda.cuda_execution.is_some());
    assert!(cpu_pcout.exists());
    assert!(cuda_pcout.exists());

    let cpu_replay = verify_replay(&cpu_pcout, Some(&cpu_replay)).expect("CPU replay must pass");
    let cuda_replay =
        verify_replay(&cuda_pcout, Some(&cuda_replay)).expect("CUDA replay must pass");
    assert_eq!(cpu_replay.status, "PASS");
    assert_eq!(cuda_replay.status, "PASS");
    assert_eq!(cpu_replay.output_state_hash, cuda_replay.output_state_hash);
}

#[ignore = "GPU_RUNTIME_REQUIRED: needs real GPU + built PTX; run on the RTX workstation via `cargo test -- --ignored`"]
#[test]
fn explicit_cuda_error_is_gpu_or_security_and_never_cpu_backend() {
    let tmp = tempfile::tempdir().unwrap();
    let pcode = tmp.path().join("reference.pcode");
    let cuda_pcout = tmp.path().join("reference.cuda.pcout");
    compile_reference_json(&reference_json_path(), &pcode).expect("reference corpus must compile");
    match execute_packet(&pcode, Backend::Cuda, &cuda_pcout) {
        Ok(report) => {
            assert_eq!(report.backend, "cuda");
            assert!(cuda_pcout.exists());
        }
        Err(PcError::Gpu(_)) | Err(PcError::Security(_)) => {
            assert!(!cuda_pcout.exists());
        }
        Err(other) => panic!("unexpected CUDA error class: {other:?}"),
    }
}
