// [STAGE-2-OPTIX] OptiX Pipeline and Program Group Management
//
// Provides safe abstractions for OptiX ray tracing pipelines.

use crate::context::OptixContext;
use crate::error::{check_optix, OptixError, Result};
use crate::loader::OptixApi;
use crate::module::{Module, PipelineCompileOptions};
use optix_sys::*;
use std::ffi::CString;
use std::ptr;

/// Program group kind
#[derive(Debug, Clone, Copy)]
pub enum ProgramGroupKind {
    /// Ray generation program
    RayGen,
    /// Miss program
    Miss,
    /// Hit group (closest hit, any hit, intersection)
    HitGroup,
    /// Callable program
    Callable,
    /// Exception program
    Exception,
}

/// Description of a program within a module
#[derive(Debug, Clone)]
pub struct ProgramDesc {
    /// Entry function name in the module
    pub entry_function: String,
}

impl ProgramDesc {
    /// Create a new program description
    pub fn new(entry_function: impl Into<String>) -> Self {
        Self {
            entry_function: entry_function.into(),
        }
    }
}

/// Hit group program description
#[derive(Debug, Clone, Default)]
pub struct HitGroupDesc {
    /// Closest hit program (optional)
    pub closest_hit: Option<ProgramDesc>,
    /// Any hit program (optional)
    pub any_hit: Option<ProgramDesc>,
    /// Intersection program (optional, for custom primitives)
    pub intersection: Option<ProgramDesc>,
}

/// An OptiX program group
pub struct ProgramGroup {
    handle: OptixProgramGroup,
    _context: OptixDeviceContext,
}

impl ProgramGroup {
    /// Create a ray generation program group
    pub fn create_raygen(
        ctx: &OptixContext,
        module: &Module,
        entry_function: &str,
    ) -> Result<Self> {
        let api = OptixApi::get()?;

        let entry_fn = CString::new(entry_function)
            .map_err(|_| OptixError::PipelineError("Invalid entry function name".to_string()))?;

        let desc = OptixProgramGroupDesc {
            kind: OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
            __bindgen_anon_1: OptixProgramGroupDesc__bindgen_ty_1 {
                raygen: OptixProgramGroupSingleModule {
                    module: module.handle(),
                    entryFunctionName: entry_fn.as_ptr(),
                },
            },
            flags: 0,
        };

        Self::create_single(ctx, &desc)
    }

    /// Create a miss program group
    pub fn create_miss(ctx: &OptixContext, module: &Module, entry_function: &str) -> Result<Self> {
        let api = OptixApi::get()?;

        let entry_fn = CString::new(entry_function)
            .map_err(|_| OptixError::PipelineError("Invalid entry function name".to_string()))?;

        let desc = OptixProgramGroupDesc {
            kind: OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_MISS,
            __bindgen_anon_1: OptixProgramGroupDesc__bindgen_ty_1 {
                miss: OptixProgramGroupSingleModule {
                    module: module.handle(),
                    entryFunctionName: entry_fn.as_ptr(),
                },
            },
            flags: 0,
        };

        Self::create_single(ctx, &desc)
    }

    /// Create a hit group program group
    pub fn create_hitgroup(
        ctx: &OptixContext,
        ch_module: Option<&Module>,
        ch_entry: Option<&str>,
        ah_module: Option<&Module>,
        ah_entry: Option<&str>,
        is_module: Option<&Module>,
        is_entry: Option<&str>,
    ) -> Result<Self> {
        let ch_entry_cstr = ch_entry.map(|s| CString::new(s).ok()).flatten();
        let ah_entry_cstr = ah_entry.map(|s| CString::new(s).ok()).flatten();
        let is_entry_cstr = is_entry.map(|s| CString::new(s).ok()).flatten();

        let desc = OptixProgramGroupDesc {
            kind: OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
            __bindgen_anon_1: OptixProgramGroupDesc__bindgen_ty_1 {
                hitgroup: OptixProgramGroupHitgroup {
                    moduleCH: ch_module.map(|m| m.handle()).unwrap_or(ptr::null_mut()),
                    entryFunctionNameCH: ch_entry_cstr
                        .as_ref()
                        .map(|s| s.as_ptr())
                        .unwrap_or(ptr::null()),
                    moduleAH: ah_module.map(|m| m.handle()).unwrap_or(ptr::null_mut()),
                    entryFunctionNameAH: ah_entry_cstr
                        .as_ref()
                        .map(|s| s.as_ptr())
                        .unwrap_or(ptr::null()),
                    moduleIS: is_module.map(|m| m.handle()).unwrap_or(ptr::null_mut()),
                    entryFunctionNameIS: is_entry_cstr
                        .as_ref()
                        .map(|s| s.as_ptr())
                        .unwrap_or(ptr::null()),
                },
            },
            flags: 0,
        };

        Self::create_single(ctx, &desc)
    }

    fn create_single(ctx: &OptixContext, desc: &OptixProgramGroupDesc) -> Result<Self> {
        let api = OptixApi::get()?;

        let options = OptixProgramGroupOptions {
            payloadType: ptr::null(),
        };

        let mut log_buffer = vec![0u8; 16384];
        let mut log_size = log_buffer.len();
        let mut program_group: OptixProgramGroup = ptr::null_mut();

        let result = unsafe {
            api.program_group_create(
                ctx.handle(),
                desc,
                1,
                &options,
                log_buffer.as_mut_ptr() as *mut i8,
                &mut log_size,
                &mut program_group,
            )
        };

        let actual_log_size = log_size.min(log_buffer.len());
        if actual_log_size > 1 {
            let log_msg = String::from_utf8_lossy(&log_buffer[..actual_log_size.saturating_sub(1)]);
            log::debug!("OptiX program group creation log: {}", log_msg);
        }

        check_optix(result, "OptiX operation")?;

        if program_group.is_null() {
            return Err(OptixError::PipelineError(
                "Program group creation returned null".to_string(),
            ));
        }

        Ok(Self {
            handle: program_group,
            _context: ctx.handle(),
        })
    }

    /// Get the raw program group handle
    pub fn handle(&self) -> OptixProgramGroup {
        self.handle
    }

    /// Pack SBT record header for this program group
    pub fn pack_header(&self, record: &mut [u8]) -> Result<()> {
        if record.len() < SBT_RECORD_HEADER_SIZE {
            return Err(OptixError::PipelineError(format!(
                "SBT record buffer too small: {} < {}",
                record.len(),
                SBT_RECORD_HEADER_SIZE
            )));
        }

        let api = OptixApi::get()?;
        let result =
            unsafe { api.sbt_record_pack_header(self.handle, record.as_mut_ptr() as *mut _) };
        check_optix(result, "OptiX operation")
    }
}

impl Drop for ProgramGroup {
    fn drop(&mut self) {
        if let Ok(api) = OptixApi::get() {
            unsafe {
                let _ = api.program_group_destroy(self.handle);
            }
        }
    }
}

/// Pipeline link options
#[derive(Debug, Clone)]
pub struct PipelineLinkOptions {
    /// Maximum trace recursion depth
    pub max_trace_depth: u32,
}

impl Default for PipelineLinkOptions {
    fn default() -> Self {
        Self { max_trace_depth: 2 }
    }
}

/// An OptiX ray tracing pipeline
pub struct Pipeline {
    handle: OptixPipeline,
    _context: OptixDeviceContext,
}

impl Pipeline {
    /// Create a pipeline from program groups
    pub fn create(
        ctx: &OptixContext,
        compile_options: &PipelineCompileOptions,
        link_options: &PipelineLinkOptions,
        program_groups: &[&ProgramGroup],
    ) -> Result<Self> {
        let api = OptixApi::get()?;

        let launch_params_name = CString::new(
            compile_options
                .pipeline_launch_params_variable_name
                .as_str(),
        )
        .map_err(|_| OptixError::PipelineError("Invalid launch params name".to_string()))?;

        let pipeline_compile_options = OptixPipelineCompileOptions {
            usesMotionBlur: if compile_options.uses_motion_blur {
                1
            } else {
                0
            },
            traversableGraphFlags: compile_options.traversable_graph_flags,
            numPayloadValues: compile_options.num_payload_values as i32,
            numAttributeValues: compile_options.num_attribute_values as i32,
            exceptionFlags: compile_options.exception_flags,
            pipelineLaunchParamsVariableName: launch_params_name.as_ptr(),
            pipelineLaunchParamsSizeInBytes: 0, // 0 = no size validation
            usesPrimitiveTypeFlags: compile_options.use_primitive_type_flags,
            allowOpacityMicromaps: 0,
            allowClusteredGeometry: 0,
        };

        let pipeline_link_options = OptixPipelineLinkOptions {
            maxTraceDepth: link_options.max_trace_depth,
            maxContinuationCallableDepth: 0,
            maxDirectCallableDepthFromState: 0,
            maxDirectCallableDepthFromTraversal: 0,
            maxTraversableGraphDepth: 1, // default for single GAS
        };

        let handles: Vec<OptixProgramGroup> = program_groups.iter().map(|pg| pg.handle()).collect();

        let mut log_buffer = vec![0u8; 16384];
        let mut log_size = log_buffer.len();
        let mut pipeline: OptixPipeline = ptr::null_mut();

        let result = unsafe {
            api.pipeline_create(
                ctx.handle(),
                &pipeline_compile_options,
                &pipeline_link_options,
                handles.as_ptr(),
                handles.len() as u32,
                log_buffer.as_mut_ptr() as *mut i8,
                &mut log_size,
                &mut pipeline,
            )
        };

        let actual_log_size = log_size.min(log_buffer.len());
        if actual_log_size > 1 {
            let log_msg = String::from_utf8_lossy(&log_buffer[..actual_log_size.saturating_sub(1)]);
            log::debug!("OptiX pipeline creation log: {}", log_msg);
        }

        check_optix(result, "OptiX operation")?;

        if pipeline.is_null() {
            return Err(OptixError::PipelineError(
                "Pipeline creation returned null".to_string(),
            ));
        }

        Ok(Self {
            handle: pipeline,
            _context: ctx.handle(),
        })
    }

    /// Set stack sizes for the pipeline
    pub fn set_stack_size(
        &self,
        direct_callable_depth: u32,
        continuation_callable_depth: u32,
        max_traversable_depth: u32,
        max_cc_depth: u32,
    ) -> Result<()> {
        let api = OptixApi::get()?;
        let result = unsafe {
            api.pipeline_set_stack_size(
                self.handle,
                direct_callable_depth,
                continuation_callable_depth,
                max_traversable_depth,
                max_cc_depth,
            )
        };
        check_optix(result, "OptiX operation")
    }

    /// Get the raw pipeline handle
    pub fn handle(&self) -> OptixPipeline {
        self.handle
    }

    /// Launch the pipeline
    pub fn launch(
        &self,
        stream: CUstream,
        params_ptr: CUdeviceptr,
        params_size: usize,
        sbt: &ShaderBindingTable,
        width: u32,
        height: u32,
        depth: u32,
    ) -> Result<()> {
        let api = OptixApi::get()?;

        let optix_sbt = sbt.to_optix_sbt();

        let result = unsafe {
            api.launch(
                self.handle,
                stream,
                params_ptr,
                params_size,
                &optix_sbt,
                width,
                height,
                depth,
            )
        };

        check_optix(result, "OptiX operation")
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        if let Ok(api) = OptixApi::get() {
            unsafe {
                let _ = api.pipeline_destroy(self.handle);
            }
        }
    }
}

/// Shader Binding Table (SBT) for pipeline launch
pub struct ShaderBindingTable {
    /// Ray generation record device pointer
    pub raygen_record: CUdeviceptr,
    /// Exception record device pointer (optional)
    pub exception_record: CUdeviceptr,
    /// Miss records device pointer
    pub miss_record_base: CUdeviceptr,
    /// Miss record stride in bytes
    pub miss_record_stride: u32,
    /// Number of miss records
    pub miss_record_count: u32,
    /// Hit group records device pointer
    pub hitgroup_record_base: CUdeviceptr,
    /// Hit group record stride in bytes
    pub hitgroup_record_stride: u32,
    /// Number of hit group records
    pub hitgroup_record_count: u32,
    /// Callable records device pointer (optional)
    pub callable_record_base: CUdeviceptr,
    /// Callable record stride in bytes
    pub callable_record_stride: u32,
    /// Number of callable records
    pub callable_record_count: u32,
}

impl ShaderBindingTable {
    /// Create a new shader binding table
    pub fn new() -> Self {
        Self {
            raygen_record: 0,
            exception_record: 0,
            miss_record_base: 0,
            miss_record_stride: 0,
            miss_record_count: 0,
            hitgroup_record_base: 0,
            hitgroup_record_stride: 0,
            hitgroup_record_count: 0,
            callable_record_base: 0,
            callable_record_stride: 0,
            callable_record_count: 0,
        }
    }

    /// Convert to OptiX SBT structure
    fn to_optix_sbt(&self) -> OptixShaderBindingTable {
        OptixShaderBindingTable {
            raygenRecord: self.raygen_record,
            exceptionRecord: self.exception_record,
            missRecordBase: self.miss_record_base,
            missRecordStrideInBytes: self.miss_record_stride,
            missRecordCount: self.miss_record_count,
            hitgroupRecordBase: self.hitgroup_record_base,
            hitgroupRecordStrideInBytes: self.hitgroup_record_stride,
            hitgroupRecordCount: self.hitgroup_record_count,
            callablesRecordBase: self.callable_record_base,
            callablesRecordStrideInBytes: self.callable_record_stride,
            callablesRecordCount: self.callable_record_count,
        }
    }
}

impl Default for ShaderBindingTable {
    fn default() -> Self {
        Self::new()
    }
}

/// SBT record header size in bytes (32 bytes as defined in optix_types.h)
pub const SBT_RECORD_HEADER_SIZE: usize = 32;

/// Calculate aligned SBT record size
pub fn aligned_sbt_record_size(data_size: usize) -> usize {
    let size = SBT_RECORD_HEADER_SIZE + data_size;
    // Align to OPTIX_SBT_RECORD_ALIGNMENT
    let alignment = OPTIX_SBT_RECORD_ALIGNMENT as usize;
    (size + alignment - 1) & !(alignment - 1)
}
