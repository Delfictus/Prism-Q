// [STAGE-2-OPTIX] OptiX Context Implementation with Function Table
//
// Full implementation of OptixContext with dynamic function loading.

use crate::error::{check_optix, Result};
use crate::loader::OptixApi;
use cudarc::driver::sys::CUcontext;
use optix_sys::*;
use std::ffi::CString;
use std::ptr;

impl crate::context::OptixContext {
    /// Initialize OptiX API (must be called once before creating contexts)
    ///
    /// # Safety
    ///
    /// Must be called before any other OptiX function.
    /// Should only be called once per process.
    pub fn init() -> Result<()> {
        let api = OptixApi::get()?;
        unsafe {
            check_optix(api.init(), "OptiX initialization")?;
        }
        log::info!("OptiX API initialized (version {})", OPTIX_VERSION);
        Ok(())
    }

    /// Create a new OptiX device context
    ///
    /// # Arguments
    ///
    /// * `cuda_context` - CUDA context to bind to (must be current)
    /// * `validation_enabled` - Enable validation mode (slower but catches errors)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use prism_optix::OptixContext;
    /// use cudarc::driver::CudaDevice;
    ///
    /// // Initialize OptiX once
    /// OptixContext::init().unwrap();
    ///
    /// // Create CUDA device
    /// let cuda_dev = CudaDevice::new(0).unwrap();
    ///
    /// // Create OptiX context
    /// let optix_ctx = OptixContext::new(
    ///     cuda_dev.cu_primary_ctx(),
    ///     true  // validation enabled
    /// ).unwrap();
    /// ```
    pub fn new(cuda_context: CUcontext, validation_enabled: bool) -> Result<Self> {
        let api = OptixApi::get()?;

        let mut options = OptixDeviceContextOptions {
            logCallbackFunction: Some(log_callback),
            logCallbackData: ptr::null_mut(),
            logCallbackLevel: if validation_enabled { 4 } else { 2 },
            validationMode: if validation_enabled {
                OptixDeviceContextValidationMode::OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL
            } else {
                OptixDeviceContextValidationMode::OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF
            },
        };

        let mut handle: OptixDeviceContext = ptr::null_mut();

        unsafe {
            // Cast CUcontext from cudarc to optix-sys (same ABI)
            let cuda_ctx_ptr = cuda_context as *mut optix_sys::CUctx_st;
            check_optix(
                api.device_context_create(cuda_ctx_ptr, &mut options, &mut handle),
                "OptiX device context creation",
            )?;
        }

        log::info!("Created OptiX context (validation: {})", validation_enabled);

        Ok(Self {
            handle,
            cuda_context,
            validation_enabled,
        })
    }

    /// Set cache location for OptiX module caching
    pub fn set_cache_location(&self, cache_dir: &str) -> Result<()> {
        let api = OptixApi::get()?;
        let c_str = CString::new(cache_dir).map_err(|_| {
            crate::error::OptixError::InvalidValue("Invalid cache directory path".to_string())
        })?;

        unsafe {
            check_optix(
                api.device_context_set_cache_location(self.handle, c_str.as_ptr()),
                "Set OptiX cache location",
            )?;
        }
        log::info!("OptiX cache location set to: {}", cache_dir);
        Ok(())
    }

    /// Enable or disable OptiX cache
    pub fn set_cache_enabled(&self, enabled: bool) -> Result<()> {
        let api = OptixApi::get()?;

        unsafe {
            check_optix(
                api.device_context_set_cache_enabled(self.handle, if enabled { 1 } else { 0 }),
                "Set OptiX cache enabled",
            )?;
        }
        log::info!("OptiX cache enabled: {}", enabled);
        Ok(())
    }
}

impl Drop for crate::context::OptixContext {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            if let Ok(api) = OptixApi::get() {
                unsafe {
                    if let Err(e) = check_optix(
                        api.device_context_destroy(self.handle),
                        "OptiX context destruction",
                    ) {
                        log::error!("Failed to destroy OptiX context: {}", e);
                    } else {
                        log::debug!("OptiX context destroyed");
                    }
                }
            }
        }
    }
}

// OptiX log callback
unsafe extern "C" fn log_callback(
    level: ::std::os::raw::c_uint,
    tag: *const ::std::os::raw::c_char,
    message: *const ::std::os::raw::c_char,
    _cbdata: *mut ::std::os::raw::c_void,
) {
    use std::ffi::CStr;

    let tag_str = if !tag.is_null() {
        CStr::from_ptr(tag).to_string_lossy()
    } else {
        "".into()
    };

    let msg_str = if !message.is_null() {
        CStr::from_ptr(message).to_string_lossy()
    } else {
        "".into()
    };

    match level {
        0 | 1 | 2 => log::error!("[OptiX] [{}] {}", tag_str, msg_str),
        3 => log::warn!("[OptiX] [{}] {}", tag_str, msg_str),
        4 => log::info!("[OptiX] [{}] {}", tag_str, msg_str),
        _ => log::debug!("[OptiX] [{}] {}", tag_str, msg_str),
    }
}
