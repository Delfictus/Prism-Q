//! cuFFT FFI bindings for PME electrostatics
//!
//! These are minimal bindings to NVIDIA's cuFFT library for 3D FFT operations
//! used in Particle Mesh Ewald calculations.

use std::ffi::c_int;
use std::os::raw::c_void;

/// cuFFT plan handle
pub type CufftHandle = c_int;

/// cuFFT complex number (single precision)
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct CufftComplex {
    pub x: f32, // Real part
    pub y: f32, // Imaginary part
}

// Safety: CufftComplex is a POD type
unsafe impl cudarc::driver::DeviceRepr for CufftComplex {}
unsafe impl cudarc::driver::ValidAsZeroBits for CufftComplex {}

/// cuFFT result codes
pub type CufftResult = c_int;

pub const CUFFT_SUCCESS: CufftResult = 0;
pub const CUFFT_INVALID_PLAN: CufftResult = 1;
pub const CUFFT_ALLOC_FAILED: CufftResult = 2;
pub const CUFFT_INVALID_TYPE: CufftResult = 3;
pub const CUFFT_INVALID_VALUE: CufftResult = 4;
pub const CUFFT_INTERNAL_ERROR: CufftResult = 5;
pub const CUFFT_EXEC_FAILED: CufftResult = 6;
pub const CUFFT_SETUP_FAILED: CufftResult = 7;
pub const CUFFT_INVALID_SIZE: CufftResult = 8;
pub const CUFFT_UNALIGNED_DATA: CufftResult = 9;

/// cuFFT transform type for R2C (Real-to-Complex)
pub const CUFFT_R2C: c_int = 0x2a;

/// cuFFT transform type for C2R (Complex-to-Real)
pub const CUFFT_C2R: c_int = 0x2c;

#[link(name = "cufft")]
extern "C" {
    /// Create a 3D FFT plan
    pub fn cufftPlan3d(
        plan: *mut CufftHandle,
        nx: c_int,
        ny: c_int,
        nz: c_int,
        type_: c_int,
    ) -> CufftResult;

    /// Destroy an FFT plan
    pub fn cufftDestroy(plan: CufftHandle) -> CufftResult;

    /// Execute Real-to-Complex FFT
    pub fn cufftExecR2C(
        plan: CufftHandle,
        idata: *mut f32,
        odata: *mut CufftComplex,
    ) -> CufftResult;

    /// Execute Complex-to-Real FFT
    pub fn cufftExecC2R(
        plan: CufftHandle,
        idata: *mut CufftComplex,
        odata: *mut f32,
    ) -> CufftResult;

    /// Set the CUDA stream for FFT execution
    pub fn cufftSetStream(plan: CufftHandle, stream: *mut c_void) -> CufftResult;
}

/// Convert cuFFT error code to human-readable string
pub fn cufft_error_string(result: CufftResult) -> &'static str {
    match result {
        CUFFT_SUCCESS => "CUFFT_SUCCESS",
        CUFFT_INVALID_PLAN => "CUFFT_INVALID_PLAN",
        CUFFT_ALLOC_FAILED => "CUFFT_ALLOC_FAILED",
        CUFFT_INVALID_TYPE => "CUFFT_INVALID_TYPE",
        CUFFT_INVALID_VALUE => "CUFFT_INVALID_VALUE",
        CUFFT_INTERNAL_ERROR => "CUFFT_INTERNAL_ERROR",
        CUFFT_EXEC_FAILED => "CUFFT_EXEC_FAILED",
        CUFFT_SETUP_FAILED => "CUFFT_SETUP_FAILED",
        CUFFT_INVALID_SIZE => "CUFFT_INVALID_SIZE",
        CUFFT_UNALIGNED_DATA => "CUFFT_UNALIGNED_DATA",
        _ => "CUFFT_UNKNOWN_ERROR",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_string() {
        assert_eq!(cufft_error_string(CUFFT_SUCCESS), "CUFFT_SUCCESS");
        assert_eq!(cufft_error_string(CUFFT_INVALID_PLAN), "CUFFT_INVALID_PLAN");
    }
}
