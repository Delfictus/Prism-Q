// [STAGE-2-OPTIX] Safe Rust Wrapper for NVIDIA OptiX 9.1.0
//
// Provides safe, ergonomic abstractions over the unsafe optix-sys FFI bindings.
//
// ## Features
//
// - **RAII Resource Management**: Automatic cleanup of OptiX resources
// - **Error Handling**: Result<T, E> pattern with detailed error types
// - **Safe Abstractions**: No raw pointers in public API
// - **CUDA Integration**: Seamless integration with cudarc for CUDA context management
//
// ## Architecture
//
// This crate wraps the low-level optix-sys FFI bindings with safe Rust types:
//
// - `OptixContext`: Device context with RAII cleanup
// - `OptixError`: Comprehensive error types with conversion from OptixResult
// - Future: BVH acceleration structures, ray tracing pipelines, modules
//
// ## Requirements
//
// - **GPU**: NVIDIA RTX GPU (Turing, Ampere, Ada, or Blackwell architecture)
// - **Driver**: R590 or later (for OptiX 9.1.0)
// - **CUDA**: CUDA context must be current when creating OptiX context
//
// ## Example
//
// ```no_run
// use prism_optix::{OptixContext, Result};
// use cudarc::driver::CudaDevice;
//
// fn main() -> Result<()> {
//     // Initialize OptiX API (once per process)
//     OptixContext::init()?;
//
//     // Create CUDA device and context
//     let cuda_device = CudaDevice::new(0)?;
//
//     // Create OptiX context (RAII - automatically cleaned up)
//     let optix_ctx = OptixContext::new(
//         cuda_device.cu_primary_ctx(),
//         true  // validation enabled for development
//     )?;
//
//     // Use OptiX context for ray tracing...
//
//     Ok(())
//     // OptiX context automatically destroyed here
// }
// ```
//
// ## Safety
//
// This crate provides safe abstractions over unsafe FFI. All unsafe code is
// contained within this crate and carefully reviewed. Public API is entirely safe.
//
// ## Performance
//
// - **Module Caching**: Enabled by default for fast startup
// - **Validation Mode**: Disable in production for maximum performance
// - **Zero-Cost Abstractions**: Thin wrappers with no runtime overhead

#![warn(missing_docs)]
#![warn(clippy::all)]

// Re-export optix-sys types that are safe to use
pub use optix_sys::{
    OptixDeviceContext, OptixModule, OptixPipeline, OptixResult, OPTIX_VERSION,
    OPTIX_VERSION_MAJOR, OPTIX_VERSION_MICRO, OPTIX_VERSION_MINOR,
};

// Internal modules
mod accel;
mod context;
mod context_impl;
mod error;
mod loader;
mod module;
mod pipeline;

// Public exports
pub use accel::{AccelStructure, BvhBuildFlags, BvhInputType};
pub use context::OptixContext;
pub use error::{check_optix, OptixError, Result};
pub use module::{
    DebugLevel, Module, ModuleCompileOptions, OptimizationLevel, PipelineCompileOptions,
};
pub use pipeline::{
    aligned_sbt_record_size, Pipeline, PipelineLinkOptions, ProgramGroup, ShaderBindingTable,
    SBT_RECORD_HEADER_SIZE,
};

/// OptiX version information
pub mod version {
    use super::*;

    /// Get OptiX version as (major, minor, micro)
    pub fn version() -> (u32, u32, u32) {
        (
            OPTIX_VERSION_MAJOR,
            OPTIX_VERSION_MINOR,
            OPTIX_VERSION_MICRO,
        )
    }

    /// Get OptiX version as a combined integer (e.g., 90100 for 9.1.0)
    pub fn version_number() -> u32 {
        OPTIX_VERSION
    }

    /// Get OptiX version as a string (e.g., "9.1.0")
    pub fn version_string() -> String {
        format!(
            "{}.{}.{}",
            OPTIX_VERSION_MAJOR, OPTIX_VERSION_MINOR, OPTIX_VERSION_MICRO
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_info() {
        assert_eq!(version::version(), (9, 1, 0));
        assert_eq!(version::version_number(), 90100);
        assert_eq!(version::version_string(), "9.1.0");
    }
}
