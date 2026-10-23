// [STAGE-2-FFI] OptiX FFI Bindings
//
// Low-level unsafe FFI bindings to NVIDIA OptiX 9.1.0 ray tracing API.
//
// This crate provides direct access to the OptiX C API. All functions are unsafe.
// For a safe Rust wrapper, use the `prism-optix` crate instead.
//
// ## Architecture
//
// OptiX is a header-only SDK with runtime functionality provided by the NVIDIA driver.
// This crate:
// 1. Generates Rust FFI bindings from OptiX headers using bindgen (build.rs)
// 2. Links against CUDA runtime libraries (libcuda.so, libcudart.so)
// 3. OptiX functions are loaded dynamically from the driver at runtime
//
// ## Requirements
//
// - **GPU**: NVIDIA RTX GPU (Turing, Ampere, Ada, or Blackwell architecture)
// - **Driver**: R590 or later (for OptiX 9.1.0)
// - **CUDA**: CUDA toolkit installed (for header dependencies)
//
// ## Usage
//
// ```rust,no_run
// use optix_sys::*;
//
// unsafe {
//     // Initialize OptiX
//     optixInit();
//
//     // Create OptiX context
//     let mut context: OptixDeviceContext = std::ptr::null_mut();
//     optixDeviceContextCreate(
//         cu_context,
//         &options,
//         &mut context,
//     );
//     // ...
// }
// ```
//
// ## Safety
//
// All OptiX functions are `unsafe` because:
// - They interact with GPU hardware and driver
// - They require correct CUDA context management
// - They involve raw pointers and manual memory management
// - Improper use can cause GPU faults or driver crashes
//
// **NEVER** call OptiX functions from safe Rust code. Always use through the
// `prism-optix` safe wrapper crate.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(deref_nullptr)] // bindgen generates some nullptr derefs in layout tests

// Include the generated FFI bindings from build.rs
include!(concat!(env!("OUT_DIR"), "/optix_bindings.rs"));

// ============================================================================
// Version Constants
// ============================================================================

/// OptiX SDK version: 9.1.0
pub const OPTIX_VERSION_MAJOR: u32 = 9;
pub const OPTIX_VERSION_MINOR: u32 = 1;
pub const OPTIX_VERSION_MICRO: u32 = 0;

/// Combined version number (9_01_00)
pub const OPTIX_VERSION: u32 =
    OPTIX_VERSION_MAJOR * 10_000 + OPTIX_VERSION_MINOR * 100 + OPTIX_VERSION_MICRO;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optix_version_constants() {
        assert_eq!(OPTIX_VERSION_MAJOR, 9);
        assert_eq!(OPTIX_VERSION_MINOR, 1);
        assert_eq!(OPTIX_VERSION_MICRO, 0);
        assert_eq!(OPTIX_VERSION, 90100);
    }

    #[test]
    fn test_optix_types_exist() {
        // Verify key OptiX types are generated
        // (These will fail to compile if bindgen didn't generate them)

        let _result: OptixResult = OptixResult::OPTIX_SUCCESS;

        // These are type aliases/pointers, just verify they exist
        let _ctx: OptixDeviceContext = std::ptr::null_mut();
        let _module: OptixModule = std::ptr::null_mut();
        let _pipeline: OptixPipeline = std::ptr::null_mut();
    }

    #[test]
    fn test_optix_result_codes() {
        // Verify OptiX result codes are accessible
        assert_ne!(
            OptixResult::OPTIX_SUCCESS as u32,
            OptixResult::OPTIX_ERROR_INVALID_VALUE as u32
        );
    }
}
