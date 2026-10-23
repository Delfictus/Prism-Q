// [STAGE-2-OPTIX] OptiX Module Management
//
// Provides safe abstractions for loading OptiX IR/PTX modules.

use crate::context::OptixContext;
use crate::error::{check_optix, OptixError, Result};
use crate::loader::OptixApi;
use optix_sys::*;
use std::ffi::CString;
use std::path::Path;
use std::ptr;

/// Compile options for OptiX modules
#[derive(Debug, Clone)]
pub struct ModuleCompileOptions {
    /// Maximum number of registers per thread
    pub max_register_count: u32,
    /// Optimization level (0-3)
    pub opt_level: OptimizationLevel,
    /// Debug level
    pub debug_level: DebugLevel,
}

/// Optimization level for module compilation
#[derive(Debug, Clone, Copy, Default)]
pub enum OptimizationLevel {
    /// No optimization
    Level0,
    /// Basic optimization
    Level1,
    /// Standard optimization
    Level2,
    /// Maximum optimization
    #[default]
    Level3,
}

impl From<OptimizationLevel> for OptixCompileOptimizationLevel {
    fn from(level: OptimizationLevel) -> Self {
        match level {
            OptimizationLevel::Level0 => {
                OptixCompileOptimizationLevel::OPTIX_COMPILE_OPTIMIZATION_LEVEL_0
            }
            OptimizationLevel::Level1 => {
                OptixCompileOptimizationLevel::OPTIX_COMPILE_OPTIMIZATION_LEVEL_1
            }
            OptimizationLevel::Level2 => {
                OptixCompileOptimizationLevel::OPTIX_COMPILE_OPTIMIZATION_LEVEL_2
            }
            OptimizationLevel::Level3 => {
                OptixCompileOptimizationLevel::OPTIX_COMPILE_OPTIMIZATION_LEVEL_3
            }
        }
    }
}

/// Debug level for module compilation
#[derive(Debug, Clone, Copy, Default)]
pub enum DebugLevel {
    /// No debug info
    #[default]
    None,
    /// Minimal debug info
    Minimal,
    /// Moderate debug info
    Moderate,
    /// Full debug info
    Full,
}

impl From<DebugLevel> for OptixCompileDebugLevel {
    fn from(level: DebugLevel) -> Self {
        match level {
            DebugLevel::None => OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_NONE,
            DebugLevel::Minimal => OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL,
            DebugLevel::Moderate => OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_MODERATE,
            DebugLevel::Full => OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_FULL,
        }
    }
}

impl Default for ModuleCompileOptions {
    fn default() -> Self {
        Self {
            max_register_count: OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
            opt_level: OptimizationLevel::Level3,
            debug_level: DebugLevel::None,
        }
    }
}

/// Pipeline compile options (shared across all modules in a pipeline)
#[derive(Debug, Clone)]
pub struct PipelineCompileOptions {
    /// Whether to use motion blur
    pub uses_motion_blur: bool,
    /// Traversable graph flags
    pub traversable_graph_flags: u32,
    /// Number of payload values
    pub num_payload_values: u32,
    /// Number of attribute values
    pub num_attribute_values: u32,
    /// Exception flags
    pub exception_flags: u32,
    /// Pipeline launch params variable name
    pub pipeline_launch_params_variable_name: String,
    /// Primitive type flags
    pub use_primitive_type_flags: u32,
}

impl Default for PipelineCompileOptions {
    fn default() -> Self {
        Self {
            uses_motion_blur: false,
            traversable_graph_flags:
                OptixTraversableGraphFlags::OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS as u32,
            num_payload_values: 2,
            num_attribute_values: 2,
            exception_flags: OptixExceptionFlags::OPTIX_EXCEPTION_FLAG_NONE as u32,
            pipeline_launch_params_variable_name: "params".to_string(),
            use_primitive_type_flags: OptixPrimitiveTypeFlags::OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM
                as u32
                | OptixPrimitiveTypeFlags::OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE as u32,
        }
    }
}

/// An OptiX module containing compiled shader code
pub struct Module {
    handle: OptixModule,
    _context: OptixDeviceContext,
}

impl Module {
    /// Create a module from OptiX IR file
    pub fn from_optix_ir(
        ctx: &OptixContext,
        path: impl AsRef<Path>,
        module_options: &ModuleCompileOptions,
        pipeline_options: &PipelineCompileOptions,
    ) -> Result<Self> {
        let path = path.as_ref();
        let ir_data = std::fs::read(path).map_err(|e| {
            OptixError::ModuleError(format!(
                "Failed to read OptiX IR file {}: {}",
                path.display(),
                e
            ))
        })?;

        Self::from_optix_ir_bytes(ctx, &ir_data, module_options, pipeline_options)
    }

    /// Create a module from OptiX IR bytes
    pub fn from_optix_ir_bytes(
        ctx: &OptixContext,
        ir_data: &[u8],
        module_options: &ModuleCompileOptions,
        pipeline_options: &PipelineCompileOptions,
    ) -> Result<Self> {
        let api = OptixApi::get()?;

        // Prepare module compile options
        let module_compile_options = OptixModuleCompileOptions {
            maxRegisterCount: module_options.max_register_count as i32,
            optLevel: module_options.opt_level.into(),
            debugLevel: module_options.debug_level.into(),
            boundValues: ptr::null(),
            numBoundValues: 0,
            numPayloadTypes: 0,
            payloadTypes: ptr::null(),
            baseModule: ptr::null_mut(),
        };

        // Prepare pipeline compile options
        let launch_params_name = CString::new(
            pipeline_options
                .pipeline_launch_params_variable_name
                .as_str(),
        )
        .map_err(|_| OptixError::ModuleError("Invalid launch params name".to_string()))?;

        let pipeline_compile_options = OptixPipelineCompileOptions {
            usesMotionBlur: if pipeline_options.uses_motion_blur {
                1
            } else {
                0
            },
            traversableGraphFlags: pipeline_options.traversable_graph_flags,
            numPayloadValues: pipeline_options.num_payload_values as i32,
            numAttributeValues: pipeline_options.num_attribute_values as i32,
            exceptionFlags: pipeline_options.exception_flags,
            pipelineLaunchParamsVariableName: launch_params_name.as_ptr(),
            pipelineLaunchParamsSizeInBytes: 0, // 0 = no size validation
            usesPrimitiveTypeFlags: pipeline_options.use_primitive_type_flags,
            allowOpacityMicromaps: 0,
            allowClusteredGeometry: 0,
        };

        // Log buffer for compilation messages (larger buffer for Blackwell)
        let mut log_buffer = vec![0u8; 16384];
        let mut log_size = log_buffer.len();

        let mut module: OptixModule = ptr::null_mut();

        let result = unsafe {
            api.module_create(
                ctx.handle(),
                &module_compile_options,
                &pipeline_compile_options,
                ir_data.as_ptr() as *const i8,
                ir_data.len(),
                log_buffer.as_mut_ptr() as *mut i8,
                &mut log_size,
                &mut module,
            )
        };

        // Log any compilation messages (clamp log_size to buffer size)
        let actual_log_size = log_size.min(log_buffer.len());
        if actual_log_size > 1 {
            let log_msg = String::from_utf8_lossy(&log_buffer[..actual_log_size.saturating_sub(1)]);
            log::debug!("OptiX module compilation log: {}", log_msg);
        }

        check_optix(result, "module creation")?;

        if module.is_null() {
            return Err(OptixError::ModuleError(
                "Module creation returned null".to_string(),
            ));
        }

        Ok(Self {
            handle: module,
            _context: ctx.handle(),
        })
    }

    /// Get the raw OptiX module handle
    pub fn handle(&self) -> OptixModule {
        self.handle
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        if let Ok(api) = OptixApi::get() {
            unsafe {
                let _ = api.module_destroy(self.handle);
            }
        }
    }
}

// Note: Built-in sphere intersection modules require optixBuiltinISModuleGet
// which needs to be added to the loader. For now, use custom RT shaders instead.
