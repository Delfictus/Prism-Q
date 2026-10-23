// [STAGE-2-OPTIX] OptiX Error Types
//
// Safe error handling for OptiX operations using thiserror.
// Converts OptixResult codes to Rust Result<T, E> pattern.

use optix_sys::{OptixResult, OPTIX_VERSION};
use thiserror::Error;

/// OptiX error type with detailed error information
#[derive(Error, Debug)]
pub enum OptixError {
    /// OptiX initialization failed
    #[error("OptiX initialization failed: {0}")]
    InitializationFailed(String),

    /// Invalid OptiX context
    #[error("Invalid OptiX context: {0}")]
    InvalidContext(String),

    /// Invalid value passed to OptiX API
    #[error("Invalid value: {0}")]
    InvalidValue(String),

    /// Out of memory (GPU or host)
    #[error("Out of memory: {0}")]
    OutOfMemory(String),

    /// CUDA error occurred
    #[error("CUDA error: {0}")]
    CudaError(String),

    /// OptiX version mismatch
    #[error("Version mismatch: expected {expected}, got {actual}")]
    VersionMismatch { expected: u32, actual: u32 },

    /// Internal OptiX error
    #[error("Internal OptiX error: {0}")]
    InternalError(String),

    /// OptiX function not implemented
    #[error("Function not implemented: {0}")]
    NotImplemented(String),

    /// Pipeline error (compilation, validation)
    #[error("Pipeline error: {0}")]
    PipelineError(String),

    /// Acceleration structure error
    #[error("Acceleration structure error: {0}")]
    AccelError(String),

    /// Invalid operation
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    /// GPU memory allocation failed
    #[error("GPU memory allocation failed: {0}")]
    AllocationFailed(String),

    /// Module creation/compilation error
    #[error("Module error: {0}")]
    ModuleError(String),

    /// Unknown OptiX error
    #[error("Unknown OptiX error (code {code}): {message}")]
    Unknown { code: u32, message: String },
}

/// Implement conversion from OptixResult
impl From<OptixResult> for OptixError {
    fn from(result: OptixResult) -> Self {
        OptixError::from_result(result, OptixError::optix_error_name(result))
    }
}

impl OptixError {
    /// Create error from OptixResult code with context message
    pub fn from_result(result: OptixResult, context: &str) -> Self {
        match result {
            OptixResult::OPTIX_ERROR_INVALID_DEVICE_CONTEXT => {
                OptixError::InvalidContext(context.to_string())
            }
            OptixResult::OPTIX_ERROR_INVALID_VALUE => OptixError::InvalidValue(context.to_string()),
            OptixResult::OPTIX_ERROR_HOST_OUT_OF_MEMORY
            | OptixResult::OPTIX_ERROR_DEVICE_OUT_OF_MEMORY => {
                OptixError::OutOfMemory(context.to_string())
            }
            OptixResult::OPTIX_ERROR_CUDA_NOT_INITIALIZED => {
                OptixError::CudaError(format!("CUDA not initialized: {}", context))
            }
            OptixResult::OPTIX_ERROR_CUDA_ERROR => OptixError::CudaError(context.to_string()),
            OptixResult::OPTIX_ERROR_INTERNAL_ERROR => {
                OptixError::InternalError(context.to_string())
            }
            OptixResult::OPTIX_ERROR_NOT_SUPPORTED => {
                OptixError::NotImplemented(context.to_string())
            }
            OptixResult::OPTIX_ERROR_UNSUPPORTED_ABI_VERSION => OptixError::VersionMismatch {
                expected: OPTIX_VERSION,
                actual: 0,
            },
            OptixResult::OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH => {
                OptixError::InitializationFailed(format!(
                    "Function table size mismatch: {}",
                    context
                ))
            }
            OptixResult::OPTIX_ERROR_INVALID_ENTRY_FUNCTION_OPTIONS => {
                OptixError::PipelineError(format!("Invalid entry function options: {}", context))
            }
            OptixResult::OPTIX_ERROR_INVALID_LAUNCH_PARAMETER => {
                OptixError::PipelineError(format!("Invalid launch parameter: {}", context))
            }
            OptixResult::OPTIX_ERROR_NOT_COMPATIBLE => OptixError::AccelError(format!(
                "Acceleration structure not compatible: {}",
                context
            )),
            _ => OptixError::Unknown {
                code: result as u32,
                message: context.to_string(),
            },
        }
    }

    /// Get the OptiX error name string
    pub fn optix_error_name(result: OptixResult) -> &'static str {
        match result {
            OptixResult::OPTIX_SUCCESS => "OPTIX_SUCCESS",
            OptixResult::OPTIX_ERROR_INVALID_VALUE => "OPTIX_ERROR_INVALID_VALUE",
            OptixResult::OPTIX_ERROR_HOST_OUT_OF_MEMORY => "OPTIX_ERROR_HOST_OUT_OF_MEMORY",
            OptixResult::OPTIX_ERROR_INVALID_OPERATION => "OPTIX_ERROR_INVALID_OPERATION",
            OptixResult::OPTIX_ERROR_FILE_IO_ERROR => "OPTIX_ERROR_FILE_IO_ERROR",
            OptixResult::OPTIX_ERROR_INVALID_FILE_FORMAT => "OPTIX_ERROR_INVALID_FILE_FORMAT",
            OptixResult::OPTIX_ERROR_DISK_CACHE_INVALID_PATH => {
                "OPTIX_ERROR_DISK_CACHE_INVALID_PATH"
            }
            OptixResult::OPTIX_ERROR_DISK_CACHE_PERMISSION_ERROR => {
                "OPTIX_ERROR_DISK_CACHE_PERMISSION_ERROR"
            }
            OptixResult::OPTIX_ERROR_DISK_CACHE_DATABASE_ERROR => {
                "OPTIX_ERROR_DISK_CACHE_DATABASE_ERROR"
            }
            OptixResult::OPTIX_ERROR_DISK_CACHE_INVALID_DATA => {
                "OPTIX_ERROR_DISK_CACHE_INVALID_DATA"
            }
            OptixResult::OPTIX_ERROR_LAUNCH_FAILURE => "OPTIX_ERROR_LAUNCH_FAILURE",
            OptixResult::OPTIX_ERROR_INVALID_DEVICE_CONTEXT => "OPTIX_ERROR_INVALID_DEVICE_CONTEXT",
            OptixResult::OPTIX_ERROR_CUDA_NOT_INITIALIZED => "OPTIX_ERROR_CUDA_NOT_INITIALIZED",
            OptixResult::OPTIX_ERROR_VALIDATION_FAILURE => "OPTIX_ERROR_VALIDATION_FAILURE",
            OptixResult::OPTIX_ERROR_INVALID_INPUT => "OPTIX_ERROR_INVALID_INPUT",
            OptixResult::OPTIX_ERROR_INVALID_LAUNCH_PARAMETER => {
                "OPTIX_ERROR_INVALID_LAUNCH_PARAMETER"
            }
            OptixResult::OPTIX_ERROR_INVALID_PAYLOAD_ACCESS => "OPTIX_ERROR_INVALID_PAYLOAD_ACCESS",
            OptixResult::OPTIX_ERROR_INVALID_ATTRIBUTE_ACCESS => {
                "OPTIX_ERROR_INVALID_ATTRIBUTE_ACCESS"
            }
            OptixResult::OPTIX_ERROR_INVALID_FUNCTION_USE => "OPTIX_ERROR_INVALID_FUNCTION_USE",
            OptixResult::OPTIX_ERROR_INVALID_FUNCTION_ARGUMENTS => {
                "OPTIX_ERROR_INVALID_FUNCTION_ARGUMENTS"
            }
            OptixResult::OPTIX_ERROR_PIPELINE_OUT_OF_CONSTANT_MEMORY => {
                "OPTIX_ERROR_PIPELINE_OUT_OF_CONSTANT_MEMORY"
            }
            OptixResult::OPTIX_ERROR_PIPELINE_LINK_ERROR => "OPTIX_ERROR_PIPELINE_LINK_ERROR",
            OptixResult::OPTIX_ERROR_ILLEGAL_DURING_TASK_EXECUTE => {
                "OPTIX_ERROR_ILLEGAL_DURING_TASK_EXECUTE"
            }
            OptixResult::OPTIX_ERROR_CREATION_CANCELED => "OPTIX_ERROR_CREATION_CANCELED",
            OptixResult::OPTIX_ERROR_INTERNAL_COMPILER_ERROR => {
                "OPTIX_ERROR_INTERNAL_COMPILER_ERROR"
            }
            OptixResult::OPTIX_ERROR_DENOISER_MODEL_NOT_SET => "OPTIX_ERROR_DENOISER_MODEL_NOT_SET",
            OptixResult::OPTIX_ERROR_DENOISER_NOT_INITIALIZED => {
                "OPTIX_ERROR_DENOISER_NOT_INITIALIZED"
            }
            OptixResult::OPTIX_ERROR_NOT_COMPATIBLE => "OPTIX_ERROR_NOT_COMPATIBLE",
            OptixResult::OPTIX_ERROR_PAYLOAD_TYPE_MISMATCH => "OPTIX_ERROR_PAYLOAD_TYPE_MISMATCH",
            OptixResult::OPTIX_ERROR_PAYLOAD_TYPE_RESOLUTION_FAILED => {
                "OPTIX_ERROR_PAYLOAD_TYPE_RESOLUTION_FAILED"
            }
            OptixResult::OPTIX_ERROR_PAYLOAD_TYPE_ID_INVALID => {
                "OPTIX_ERROR_PAYLOAD_TYPE_ID_INVALID"
            }
            OptixResult::OPTIX_ERROR_NOT_SUPPORTED => "OPTIX_ERROR_NOT_SUPPORTED",
            OptixResult::OPTIX_ERROR_UNSUPPORTED_ABI_VERSION => {
                "OPTIX_ERROR_UNSUPPORTED_ABI_VERSION"
            }
            OptixResult::OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH => {
                "OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH"
            }
            OptixResult::OPTIX_ERROR_INVALID_ENTRY_FUNCTION_OPTIONS => {
                "OPTIX_ERROR_INVALID_ENTRY_FUNCTION_OPTIONS"
            }
            OptixResult::OPTIX_ERROR_LIBRARY_NOT_FOUND => "OPTIX_ERROR_LIBRARY_NOT_FOUND",
            OptixResult::OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND => "OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND",
            OptixResult::OPTIX_ERROR_LIBRARY_UNLOAD_FAILURE => "OPTIX_ERROR_LIBRARY_UNLOAD_FAILURE",
            OptixResult::OPTIX_ERROR_DEVICE_OUT_OF_MEMORY => "OPTIX_ERROR_DEVICE_OUT_OF_MEMORY",
            OptixResult::OPTIX_ERROR_INVALID_POINTER => "OPTIX_ERROR_INVALID_POINTER",
            OptixResult::OPTIX_ERROR_SYMBOL_NOT_FOUND => "OPTIX_ERROR_SYMBOL_NOT_FOUND",
            OptixResult::OPTIX_ERROR_CUDA_ERROR => "OPTIX_ERROR_CUDA_ERROR",
            OptixResult::OPTIX_ERROR_INTERNAL_ERROR => "OPTIX_ERROR_INTERNAL_ERROR",
            OptixResult::OPTIX_ERROR_UNKNOWN => "OPTIX_ERROR_UNKNOWN",
            _ => "OPTIX_ERROR_UNKNOWN",
        }
    }
}

/// Result type alias for OptiX operations
pub type Result<T> = std::result::Result<T, OptixError>;

/// Check OptiX result and convert to Result<(), OptixError>
#[inline]
pub fn check_optix(result: OptixResult, context: &str) -> Result<()> {
    if result == OptixResult::OPTIX_SUCCESS {
        Ok(())
    } else {
        log::error!(
            "OptiX error: {} - {} ({})",
            OptixError::optix_error_name(result),
            context,
            result as u32
        );
        Err(OptixError::from_result(result, context))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_conversion() {
        let err = OptixError::from_result(OptixResult::OPTIX_ERROR_INVALID_VALUE, "test context");
        assert!(matches!(err, OptixError::InvalidValue(_)));
    }

    #[test]
    fn test_check_optix_success() {
        let result = check_optix(OptixResult::OPTIX_SUCCESS, "test");
        assert!(result.is_ok());
    }

    #[test]
    fn test_check_optix_failure() {
        let result = check_optix(OptixResult::OPTIX_ERROR_INVALID_VALUE, "test");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_name() {
        assert_eq!(
            OptixError::optix_error_name(OptixResult::OPTIX_SUCCESS),
            "OPTIX_SUCCESS"
        );
        assert_eq!(
            OptixError::optix_error_name(OptixResult::OPTIX_ERROR_INVALID_VALUE),
            "OPTIX_ERROR_INVALID_VALUE"
        );
    }
}
