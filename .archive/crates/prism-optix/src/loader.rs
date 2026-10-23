// [STAGE-2-OPTIX] OptiX Dynamic Function Loader
//
// Loads OptiX functions dynamically using optixQueryFunctionTable.
// This is the correct mechanism for OptiX 7.0+ which only exports
// optixQueryFunctionTable from the driver library.

use crate::error::{OptixError, Result};
use optix_sys::*;
use std::sync::OnceLock;

/// Global OptiX function table (initialized once)
static OPTIX_API: OnceLock<OptixApi> = OnceLock::new();

/// OptiX API wrapper around the function table
pub struct OptixApi {
    table: OptixFunctionTable,
}

// Function pointer type for optixQueryFunctionTable
// Note: The ACTUAL signature from the driver is:
//   OptixResult optixQueryFunctionTable(
//       int abiId,
//       unsigned int numOptions,
//       OptixQueryFunctionTableOptions* optionKeys,
//       const void** optionValues,
//       void* functionTable,
//       size_t sizeOfTable  // <-- passed by VALUE, not pointer!
//   )
type OptixQueryFunctionTableFn = unsafe extern "C" fn(
    abiId: ::std::os::raw::c_int,
    numOptions: ::std::os::raw::c_uint,
    optionKeys: *mut OptixQueryFunctionTableOptions,
    optionValues: *mut *const ::std::os::raw::c_void,
    functionTable: *mut ::std::os::raw::c_void,
    sizeOfTable: usize, // Passed by value!
) -> OptixResult;

impl OptixApi {
    /// Load OptiX functions using the function table mechanism
    unsafe fn load() -> Result<Self> {
        #[cfg(target_os = "linux")]
        let lib_name = "libnvoptix.so.1";
        #[cfg(target_os = "windows")]
        let lib_name = "nvoptix.dll";

        let lib = libloading::Library::new(lib_name).map_err(|e| {
            OptixError::InitializationFailed(format!("Failed to load OptiX library: {}", e))
        })?;

        // Load the single exported function
        let query_fn: libloading::Symbol<OptixQueryFunctionTableFn> = lib
            .get(b"optixQueryFunctionTable\0")
            .map_err(|e| {
                OptixError::InitializationFailed(format!(
                    "optixQueryFunctionTable not found: {}. Make sure NVIDIA driver R590+ is installed.",
                    e
                ))
            })?;

        // Try multiple ABI versions in order of preference
        // ABI 118 = OptiX 9.1.0 (our bindings)
        // ABI 117 = OptiX 9.0.0
        // ABI 100-116 = OptiX 8.x and earlier
        let abi_versions: &[i32] = &[
            OPTIX_ABI_VERSION as i32, // Try our bindings version first (118)
            119,                      // Try newer (in case Blackwell has a newer version)
            120,
            117, // Try OptiX 9.0
            116, // Try OptiX 8.x
            115,
            100, // OptiX 8.0 baseline
            86,  // OptiX 7.7
            85,  // OptiX 7.6
            80,  // OptiX 7.5
            78,  // OptiX 7.4
            55,  // OptiX 7.3
            47,  // OptiX 7.2
            41,  // OptiX 7.1
            36,  // OptiX 7.0
        ];

        let mut working_abi: Option<i32> = None;
        let mut final_table_buffer: Option<Vec<u8>> = None;
        let our_table_size = std::mem::size_of::<OptixFunctionTable>();

        log::info!("OptiX: Our function table size: {} bytes", our_table_size);

        // The optixQueryFunctionTable signature takes sizeOfTable BY VALUE
        // If our table size doesn't match what the driver expects for a given ABI,
        // it returns SIZE_MISMATCH. We try different ABIs until one accepts our size.
        for &abi in abi_versions {
            let mut table_buffer = vec![0u8; our_table_size];

            // Pass our table size and see if this ABI accepts it
            let result = query_fn(
                abi,
                0,                    // numOptions
                std::ptr::null_mut(), // optionKeys
                std::ptr::null_mut(), // optionValues
                table_buffer.as_mut_ptr() as *mut ::std::os::raw::c_void,
                our_table_size, // sizeOfTable (by value)
            );

            match result {
                OptixResult::OPTIX_SUCCESS => {
                    working_abi = Some(abi);
                    final_table_buffer = Some(table_buffer);
                    log::info!(
                        "OptiX: ABI version {} accepted our table size ({} bytes)",
                        abi,
                        our_table_size
                    );
                    break;
                }
                OptixResult::OPTIX_ERROR_UNSUPPORTED_ABI_VERSION => {
                    log::debug!("OptiX: ABI {} is not recognized by driver", abi);
                }
                OptixResult::OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH => {
                    log::debug!(
                        "OptiX: ABI {} expects different table size (ours: {} bytes)",
                        abi,
                        our_table_size
                    );
                }
                _ => {
                    log::debug!("OptiX: ABI version {} failed: {:?}", abi, result);
                }
            }
        }

        let working_abi = working_abi.ok_or_else(|| {
            OptixError::InitializationFailed(format!(
                "No compatible OptiX ABI version found. Tried: {:?}. \
                Our table size: {} bytes. \
                Your driver may be too old or too new for these bindings.",
                abi_versions, our_table_size
            ))
        })?;

        let table_buffer =
            final_table_buffer.expect("Table buffer should be set when ABI is found");

        // Copy the bytes to our struct
        let mut table = std::mem::MaybeUninit::<OptixFunctionTable>::uninit();
        std::ptr::copy_nonoverlapping(
            table_buffer.as_ptr(),
            table.as_mut_ptr() as *mut u8,
            our_table_size,
        );

        let table = table.assume_init();

        // Verify critical functions are present
        if table.optixDeviceContextCreate.is_none() {
            return Err(OptixError::InitializationFailed(format!(
                "Function table missing optixDeviceContextCreate. \
                ABI {} may not be compatible with our bindings ({})",
                working_abi, OPTIX_ABI_VERSION
            )));
        }

        // Leak the library to keep it loaded for the program lifetime
        std::mem::forget(lib);

        log::info!(
            "OptiX function table loaded successfully (ABI version {}, table_size={} bytes)",
            working_abi,
            our_table_size
        );

        Ok(Self { table })
    }

    /// Get the global OptiX API (initializes on first call)
    pub fn get() -> Result<&'static OptixApi> {
        OPTIX_API.get_or_init(|| unsafe { Self::load().expect("Failed to load OptiX API") });
        OPTIX_API.get().ok_or_else(|| {
            OptixError::InitializationFailed("OptiX API not initialized".to_string())
        })
    }

    // Accessor methods for function pointers

    /// Initialize OptiX (no-op in function table mode, just returns success)
    pub fn init(&self) -> OptixResult {
        OptixResult::OPTIX_SUCCESS
    }

    /// Create a device context
    pub unsafe fn device_context_create(
        &self,
        from_context: CUcontext,
        options: *const OptixDeviceContextOptions,
        context: *mut OptixDeviceContext,
    ) -> OptixResult {
        if let Some(f) = self.table.optixDeviceContextCreate {
            f(from_context, options, context)
        } else {
            OptixResult::OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH
        }
    }

    /// Destroy a device context
    pub unsafe fn device_context_destroy(&self, context: OptixDeviceContext) -> OptixResult {
        if let Some(f) = self.table.optixDeviceContextDestroy {
            f(context)
        } else {
            OptixResult::OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH
        }
    }

    /// Set cache enabled
    pub unsafe fn device_context_set_cache_enabled(
        &self,
        context: OptixDeviceContext,
        enabled: i32,
    ) -> OptixResult {
        if let Some(f) = self.table.optixDeviceContextSetCacheEnabled {
            f(context, enabled)
        } else {
            OptixResult::OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH
        }
    }

    /// Set cache location
    pub unsafe fn device_context_set_cache_location(
        &self,
        context: OptixDeviceContext,
        location: *const i8,
    ) -> OptixResult {
        if let Some(f) = self.table.optixDeviceContextSetCacheLocation {
            f(context, location)
        } else {
            OptixResult::OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH
        }
    }

    /// Compute acceleration structure memory usage
    pub unsafe fn accel_compute_memory_usage(
        &self,
        context: OptixDeviceContext,
        accel_options: *const OptixAccelBuildOptions,
        build_inputs: *const OptixBuildInput,
        num_build_inputs: u32,
        buffer_sizes: *mut OptixAccelBufferSizes,
    ) -> OptixResult {
        if let Some(f) = self.table.optixAccelComputeMemoryUsage {
            f(
                context,
                accel_options,
                build_inputs,
                num_build_inputs,
                buffer_sizes,
            )
        } else {
            OptixResult::OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH
        }
    }

    /// Build acceleration structure
    pub unsafe fn accel_build(
        &self,
        context: OptixDeviceContext,
        stream: CUstream,
        accel_options: *const OptixAccelBuildOptions,
        build_inputs: *const OptixBuildInput,
        num_build_inputs: u32,
        temp_buffer: CUdeviceptr,
        temp_buffer_size: usize,
        output_buffer: CUdeviceptr,
        output_buffer_size: usize,
        output_handle: *mut OptixTraversableHandle,
        emitted_properties: *const OptixAccelEmitDesc,
        num_emitted_properties: u32,
    ) -> OptixResult {
        if let Some(f) = self.table.optixAccelBuild {
            f(
                context,
                stream,
                accel_options,
                build_inputs,
                num_build_inputs,
                temp_buffer,
                temp_buffer_size,
                output_buffer,
                output_buffer_size,
                output_handle,
                emitted_properties,
                num_emitted_properties,
            )
        } else {
            OptixResult::OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH
        }
    }

    /// Create module from PTX/OptiX IR
    pub unsafe fn module_create(
        &self,
        context: OptixDeviceContext,
        module_compile_options: *const OptixModuleCompileOptions,
        pipeline_compile_options: *const OptixPipelineCompileOptions,
        input: *const i8,
        input_size: usize,
        log_string: *mut i8,
        log_string_size: *mut usize,
        module: *mut OptixModule,
    ) -> OptixResult {
        if let Some(f) = self.table.optixModuleCreate {
            f(
                context,
                module_compile_options,
                pipeline_compile_options,
                input,
                input_size,
                log_string,
                log_string_size,
                module,
            )
        } else {
            OptixResult::OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH
        }
    }

    /// Destroy module
    pub unsafe fn module_destroy(&self, module: OptixModule) -> OptixResult {
        if let Some(f) = self.table.optixModuleDestroy {
            f(module)
        } else {
            OptixResult::OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH
        }
    }

    /// Create program group
    pub unsafe fn program_group_create(
        &self,
        context: OptixDeviceContext,
        program_descriptions: *const OptixProgramGroupDesc,
        num_program_groups: u32,
        options: *const OptixProgramGroupOptions,
        log_string: *mut i8,
        log_string_size: *mut usize,
        program_groups: *mut OptixProgramGroup,
    ) -> OptixResult {
        if let Some(f) = self.table.optixProgramGroupCreate {
            f(
                context,
                program_descriptions,
                num_program_groups,
                options,
                log_string,
                log_string_size,
                program_groups,
            )
        } else {
            OptixResult::OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH
        }
    }

    /// Destroy program group
    pub unsafe fn program_group_destroy(&self, program_group: OptixProgramGroup) -> OptixResult {
        if let Some(f) = self.table.optixProgramGroupDestroy {
            f(program_group)
        } else {
            OptixResult::OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH
        }
    }

    /// Create pipeline
    pub unsafe fn pipeline_create(
        &self,
        context: OptixDeviceContext,
        pipeline_compile_options: *const OptixPipelineCompileOptions,
        pipeline_link_options: *const OptixPipelineLinkOptions,
        program_groups: *const OptixProgramGroup,
        num_program_groups: u32,
        log_string: *mut i8,
        log_string_size: *mut usize,
        pipeline: *mut OptixPipeline,
    ) -> OptixResult {
        if let Some(f) = self.table.optixPipelineCreate {
            f(
                context,
                pipeline_compile_options,
                pipeline_link_options,
                program_groups,
                num_program_groups,
                log_string,
                log_string_size,
                pipeline,
            )
        } else {
            OptixResult::OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH
        }
    }

    /// Destroy pipeline
    pub unsafe fn pipeline_destroy(&self, pipeline: OptixPipeline) -> OptixResult {
        if let Some(f) = self.table.optixPipelineDestroy {
            f(pipeline)
        } else {
            OptixResult::OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH
        }
    }

    /// Set pipeline stack size
    pub unsafe fn pipeline_set_stack_size(
        &self,
        pipeline: OptixPipeline,
        direct_callable_stack_size_from_traversal: u32,
        direct_callable_stack_size_from_state: u32,
        continuation_stack_size: u32,
        max_traversable_graph_depth: u32,
    ) -> OptixResult {
        if let Some(f) = self.table.optixPipelineSetStackSize {
            f(
                pipeline,
                direct_callable_stack_size_from_traversal,
                direct_callable_stack_size_from_state,
                continuation_stack_size,
                max_traversable_graph_depth,
            )
        } else {
            OptixResult::OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH
        }
    }

    /// Pack SBT record header
    pub unsafe fn sbt_record_pack_header(
        &self,
        program_group: OptixProgramGroup,
        sbt_record_header_host_pointer: *mut ::std::os::raw::c_void,
    ) -> OptixResult {
        if let Some(f) = self.table.optixSbtRecordPackHeader {
            f(program_group, sbt_record_header_host_pointer)
        } else {
            OptixResult::OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH
        }
    }

    /// Launch pipeline
    pub unsafe fn launch(
        &self,
        pipeline: OptixPipeline,
        stream: CUstream,
        pipeline_params: CUdeviceptr,
        pipeline_params_size: usize,
        sbt: *const OptixShaderBindingTable,
        width: u32,
        height: u32,
        depth: u32,
    ) -> OptixResult {
        if let Some(f) = self.table.optixLaunch {
            f(
                pipeline,
                stream,
                pipeline_params,
                pipeline_params_size,
                sbt,
                width,
                height,
                depth,
            )
        } else {
            OptixResult::OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires OptiX driver
    fn test_load_optix_api() {
        let api = OptixApi::get();
        assert!(api.is_ok(), "Failed to load OptiX API");
    }
}
