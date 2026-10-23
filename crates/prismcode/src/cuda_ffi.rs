use serde_json::json;
use std::ffi::CStr;
use std::os::raw::c_char;
#[cfg(feature = "cuda")]
type CudaStreamPtr = *mut std::ffi::c_void;

pub const CUDA_ABI_V0_1: u32 = 0x0001_0000;
pub const CUDA_ISA_V0_1: u32 = 0x0001_0000;
pub const CUDA_LINK_SENTINEL: u32 = 0x5043_5531;

pub const CUDA_RUNTIME_LINKED_READY: u32 = 1;
pub const CUDA_RUNTIME_NO_DEVICE: u32 = 2;
pub const CUDA_RUNTIME_DRIVER_OR_RUNTIME_ERROR: u32 = 3;

pub const CUDA_SELFTEST_PASS: u32 = 1;
pub const CUDA_SELFTEST_FAIL: u32 = 2;
pub const CUDA_SELFTEST_SKIPPED: u32 = 3;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PrismcodeCudaRuntimeStatus {
    pub abi_version: u32,
    pub isa_version: u32,
    pub link_sentinel: u32,
    pub runtime_state: u32,
    pub runtime_version: i32,
    pub driver_version: i32,
    pub device_count: i32,
    pub active_device: i32,
    pub compute_major: i32,
    pub compute_minor: i32,
    pub total_global_mem: u64,
    pub error_code: u32,
    pub reserved0: u32,
    pub device_name: [c_char; 128],
}

impl Default for PrismcodeCudaRuntimeStatus {
    fn default() -> Self {
        Self {
            abi_version: 0,
            isa_version: 0,
            link_sentinel: 0,
            runtime_state: 0,
            runtime_version: -1,
            driver_version: -1,
            device_count: 0,
            active_device: -1,
            compute_major: -1,
            compute_minor: -1,
            total_global_mem: 0,
            error_code: 0,
            reserved0: 0,
            device_name: [0; 128],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PrismcodeCudaSelfTestResult {
    pub status: u32,
    pub error_code: u32,
    pub element_count: u64,
    pub max_abs_error: f32,
    pub tensor_copy_parity: u32,
    pub mismatch_count: u64,
    pub sentinel_hash: u64,
    pub reserved0: u32,
    pub reserved1: u32,
}

#[cfg(feature = "cuda")]
extern "C" {
    fn prismcode_cuda_abi_version() -> u32;
    fn prismcode_cuda_isa_version() -> u32;
    fn prismcode_cuda_link_probe() -> u32;
    fn prismcode_cuda_runtime_status(out_status: *mut PrismcodeCudaRuntimeStatus) -> i32;
    fn prismcode_cuda_m1_selftest(
        out_result: *mut PrismcodeCudaSelfTestResult,
        stream: CudaStreamPtr,
    ) -> i32;
    fn pc_cuda_get_device_count(out_count: *mut i32) -> i32;
    fn pc_cuda_get_device_info(out_status: *mut PrismcodeCudaRuntimeStatus) -> i32;
    fn pc_cuda_selftest(out_result: *mut PrismcodeCudaSelfTestResult) -> i32;
    fn pc_cuda_execute_tensor_copy(h_in: *const f32, h_out: *mut f32, count: u64) -> i32;
    fn pc_cuda_execute_tensor_reduce(h_in: *const f32, count: u64, h_sum_out: *mut f32) -> i32;
    fn pc_cuda_execute_event_integrate(
        instruction_count: u64,
        row_count: u64,
        out_event_count: *mut u64,
        out_hash: *mut u64,
    ) -> i32;
    fn pc_cuda_execute_nanl_bin_pack(h_tokens: *const u64, h_packed: *mut u64, count: u64) -> i32;
    fn pc_cuda_execute_nanl_q4_validate(
        h_q4: *const u8,
        len: u64,
        out_pass: *mut u32,
        out_hash: *mut u64,
    ) -> i32;
    fn pc_cuda_execute_drf_fire(
        h_resonance: *const f32,
        h_intensity: *mut f32,
        count: u64,
        threshold: f32,
        out_fired: *mut u64,
    ) -> i32;
    fn pc_cuda_status_string(code: i32) -> *const c_char;
}

pub fn cuda_feature_enabled() -> bool {
    cfg!(feature = "cuda")
}

#[cfg(feature = "cuda")]
pub fn link_probe() -> (u32, u32, u32) {
    unsafe {
        (
            prismcode_cuda_abi_version(),
            prismcode_cuda_isa_version(),
            prismcode_cuda_link_probe(),
        )
    }
}

#[cfg(not(feature = "cuda"))]
pub fn link_probe() -> (u32, u32, u32) {
    (0, 0, 0)
}

pub fn link_probe_matches() -> bool {
    let (abi, isa, probe) = link_probe();
    abi == CUDA_ABI_V0_1 && isa == CUDA_ISA_V0_1 && probe == CUDA_LINK_SENTINEL
}

#[cfg(feature = "cuda")]
pub(crate) fn runtime_status() -> (i32, PrismcodeCudaRuntimeStatus) {
    let mut status = PrismcodeCudaRuntimeStatus::default();
    let ffi_error = unsafe { prismcode_cuda_runtime_status(&mut status) };
    (ffi_error, status)
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn runtime_status() -> (i32, PrismcodeCudaRuntimeStatus) {
    (-1, PrismcodeCudaRuntimeStatus::default())
}

#[cfg(feature = "cuda")]
pub(crate) fn m1_selftest() -> (i32, PrismcodeCudaSelfTestResult) {
    let mut result = PrismcodeCudaSelfTestResult::default();
    let ffi_error = unsafe { prismcode_cuda_m1_selftest(&mut result, std::ptr::null_mut()) };
    (ffi_error, result)
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn m1_selftest() -> (i32, PrismcodeCudaSelfTestResult) {
    skipped_selftest()
}

pub fn skipped_selftest() -> (i32, PrismcodeCudaSelfTestResult) {
    (
        -1,
        PrismcodeCudaSelfTestResult {
            status: CUDA_SELFTEST_SKIPPED,
            ..PrismcodeCudaSelfTestResult::default()
        },
    )
}

#[cfg(feature = "cuda")]
pub(crate) fn pc_device_count() -> (i32, i32) {
    let mut count = 0;
    let ffi_error = unsafe { pc_cuda_get_device_count(&mut count) };
    (ffi_error, count)
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn pc_device_count() -> (i32, i32) {
    (-1, 0)
}

#[cfg(feature = "cuda")]
pub(crate) fn pc_device_info() -> (i32, PrismcodeCudaRuntimeStatus) {
    let mut status = PrismcodeCudaRuntimeStatus::default();
    let ffi_error = unsafe { pc_cuda_get_device_info(&mut status) };
    (ffi_error, status)
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn pc_device_info() -> (i32, PrismcodeCudaRuntimeStatus) {
    (-1, PrismcodeCudaRuntimeStatus::default())
}

#[cfg(feature = "cuda")]
pub(crate) fn pc_selftest() -> (i32, PrismcodeCudaSelfTestResult) {
    let mut result = PrismcodeCudaSelfTestResult::default();
    let ffi_error = unsafe { pc_cuda_selftest(&mut result) };
    (ffi_error, result)
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn pc_selftest() -> (i32, PrismcodeCudaSelfTestResult) {
    skipped_selftest()
}

#[cfg(feature = "cuda")]
pub(crate) fn execute_tensor_copy(input: &[f32]) -> (i32, Vec<f32>) {
    let mut out = vec![0.0f32; input.len()];
    let ffi_error = unsafe {
        pc_cuda_execute_tensor_copy(input.as_ptr(), out.as_mut_ptr(), input.len() as u64)
    };
    (ffi_error, out)
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn execute_tensor_copy(input: &[f32]) -> (i32, Vec<f32>) {
    (-1, vec![0.0; input.len()])
}

#[cfg(feature = "cuda")]
pub(crate) fn execute_tensor_reduce(input: &[f32]) -> (i32, f32) {
    let mut sum = 0.0f32;
    let ffi_error =
        unsafe { pc_cuda_execute_tensor_reduce(input.as_ptr(), input.len() as u64, &mut sum) };
    (ffi_error, sum)
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn execute_tensor_reduce(_input: &[f32]) -> (i32, f32) {
    (-1, 0.0)
}

#[cfg(feature = "cuda")]
pub(crate) fn execute_event_integrate(instruction_count: u64, row_count: u64) -> (i32, u64, u64) {
    let mut event_count = 0u64;
    let mut hash = 0u64;
    let ffi_error = unsafe {
        pc_cuda_execute_event_integrate(instruction_count, row_count, &mut event_count, &mut hash)
    };
    (ffi_error, event_count, hash)
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn execute_event_integrate(_instruction_count: u64, _row_count: u64) -> (i32, u64, u64) {
    (-1, 0, 0)
}

#[cfg(feature = "cuda")]
pub(crate) fn execute_nanl_bin_pack(tokens: &[u64]) -> (i32, Vec<u64>) {
    let mut out = vec![0u64; tokens.len()];
    let ffi_error = unsafe {
        pc_cuda_execute_nanl_bin_pack(tokens.as_ptr(), out.as_mut_ptr(), tokens.len() as u64)
    };
    (ffi_error, out)
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn execute_nanl_bin_pack(tokens: &[u64]) -> (i32, Vec<u64>) {
    (-1, vec![0; tokens.len()])
}

#[cfg(feature = "cuda")]
pub(crate) fn execute_nanl_q4_validate(bytes: &[u8]) -> (i32, bool, u64) {
    let mut pass = 0u32;
    let mut hash = 0u64;
    let ffi_error = unsafe {
        pc_cuda_execute_nanl_q4_validate(bytes.as_ptr(), bytes.len() as u64, &mut pass, &mut hash)
    };
    (ffi_error, pass == 1, hash)
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn execute_nanl_q4_validate(_bytes: &[u8]) -> (i32, bool, u64) {
    (-1, false, 0)
}

#[cfg(feature = "cuda")]
pub(crate) fn execute_drf_fire(resonance: &[f32], threshold: f32) -> (i32, Vec<f32>, u64) {
    let mut out = vec![0.0f32; resonance.len()];
    let mut fired = 0u64;
    let ffi_error = unsafe {
        pc_cuda_execute_drf_fire(
            resonance.as_ptr(),
            out.as_mut_ptr(),
            resonance.len() as u64,
            threshold,
            &mut fired,
        )
    };
    (ffi_error, out, fired)
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn execute_drf_fire(resonance: &[f32], _threshold: f32) -> (i32, Vec<f32>, u64) {
    (-1, vec![0.0; resonance.len()], 0)
}

#[cfg(feature = "cuda")]
pub fn status_string(code: i32) -> String {
    let ptr = unsafe { pc_cuda_status_string(code) };
    if ptr.is_null() {
        return format!("CUDA_ERROR_{code}");
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

#[cfg(not(feature = "cuda"))]
pub fn status_string(code: i32) -> String {
    format!("CUDA_UNLINKED_{code}")
}

pub fn runtime_state_label(state: u32) -> &'static str {
    match state {
        CUDA_RUNTIME_LINKED_READY => "LINKED_READY",
        CUDA_RUNTIME_NO_DEVICE => "NO_DEVICE",
        CUDA_RUNTIME_DRIVER_OR_RUNTIME_ERROR => "DRIVER_OR_RUNTIME_ERROR",
        _ => "UNKNOWN",
    }
}

pub fn selftest_status_label(status: u32) -> &'static str {
    match status {
        CUDA_SELFTEST_PASS => "PASS",
        CUDA_SELFTEST_FAIL => "FAIL",
        CUDA_SELFTEST_SKIPPED => "SKIPPED",
        _ => "UNKNOWN",
    }
}

pub fn runtime_status_json(
    ffi_error: i32,
    status: &PrismcodeCudaRuntimeStatus,
) -> serde_json::Value {
    json!({
        "ffi_error_code": ffi_error,
        "ffi_error_name": status_string(ffi_error),
        "abi_version": format!("0x{:08x}", status.abi_version),
        "isa_version": format!("0x{:08x}", status.isa_version),
        "link_sentinel": format!("0x{:08x}", status.link_sentinel),
        "runtime_state": runtime_state_label(status.runtime_state),
        "runtime_version": version_number(status.runtime_version),
        "driver_version": version_number(status.driver_version),
        "device_count": status.device_count,
        "active_device": status.active_device,
        "compute_capability": compute_capability(status.compute_major, status.compute_minor),
        "total_global_mem_bytes": status.total_global_mem,
        "device_name": device_name(&status.device_name),
        "cuda_error_code": status.error_code,
        "cuda_error_name": status_string(status.error_code as i32),
    })
}

pub fn selftest_json(ffi_error: i32, result: &PrismcodeCudaSelfTestResult) -> serde_json::Value {
    json!({
        "ffi_error_code": ffi_error,
        "ffi_error_name": status_string(ffi_error),
        "status": selftest_status_label(result.status),
        "cuda_error_code": result.error_code,
        "cuda_error_name": status_string(result.error_code as i32),
        "element_count": result.element_count,
        "tensor_copy_parity": result.tensor_copy_parity == 1,
        "mismatch_count": result.mismatch_count,
        "max_abs_error": result.max_abs_error,
        "sentinel_hash": format!("0x{:016x}", result.sentinel_hash),
    })
}

fn version_number(value: i32) -> serde_json::Value {
    if value < 0 {
        serde_json::Value::String("EVIDENCE_NOT_FOUND".to_string())
    } else {
        json!(value)
    }
}

fn compute_capability(major: i32, minor: i32) -> String {
    if major < 0 || minor < 0 {
        "EVIDENCE_NOT_FOUND".to_string()
    } else {
        format!("{major}.{minor}")
    }
}

fn device_name(name: &[c_char; 128]) -> String {
    if name[0] == 0 {
        return "EVIDENCE_NOT_FOUND".to_string();
    }
    unsafe { CStr::from_ptr(name.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}
