// [STAGE-2-OPTIX] OptiX Device Context (RAII wrapper)
//
// Safe RAII wrapper for OptixDeviceContext with automatic cleanup.
// Manages OptiX context lifecycle and provides safe API access.

use cudarc::driver::sys::CUcontext;
use optix_sys::*;

/// Safe wrapper for OptixDeviceContext with RAII semantics
///
/// This struct ensures proper cleanup of OptiX resources when dropped.
/// Only one OptiX context should exist per CUDA context.
pub struct OptixContext {
    /// Raw OptiX device context (handle)
    pub(crate) handle: OptixDeviceContext,

    /// CUDA context this OptiX context is bound to
    pub(crate) cuda_context: CUcontext,

    /// Validation mode (affects performance vs safety trade-off)
    pub(crate) validation_enabled: bool,
}

impl OptixContext {
    /// Create OptixContext from existing handle (for advanced use)
    ///
    /// # Safety
    ///
    /// The handle must be a valid OptiX device context.
    /// This OptixContext will take ownership and destroy it on drop.
    pub unsafe fn from_handle(
        handle: OptixDeviceContext,
        cuda_context: CUcontext,
        validation_enabled: bool,
    ) -> Self {
        Self {
            handle,
            cuda_context,
            validation_enabled,
        }
    }

    /// Get the raw OptiX device context handle
    ///
    /// # Safety
    ///
    /// The returned handle is only valid for the lifetime of this OptixContext.
    /// Do not use after this OptixContext is dropped.
    pub fn handle(&self) -> OptixDeviceContext {
        self.handle
    }

    /// Get the CUDA context this OptiX context is bound to
    pub fn cuda_context(&self) -> CUcontext {
        self.cuda_context
    }

    /// Check if validation mode is enabled
    pub fn validation_enabled(&self) -> bool {
        self.validation_enabled
    }

    // NOTE: Full implementation in context_impl.rs:
    // - init()
    // - new()
    // - set_cache_location()
    // - set_cache_enabled()
}

// NOTE: Drop implementation in context_impl.rs

// OptixContext is Send (can be moved between threads)
// but not Sync (cannot be shared between threads without synchronization)
unsafe impl Send for OptixContext {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optix_version() {
        assert_eq!(OPTIX_VERSION, 90100);
    }
}
