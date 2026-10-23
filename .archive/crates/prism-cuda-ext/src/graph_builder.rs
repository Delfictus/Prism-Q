//! CUDA Graph Body Builder — populates the conditional WHILE loop with kernel nodes.
//!
//! This module adds kernel nodes to the body graph of the TwinCouplingGraph's
//! WHILE conditional node. Each iteration of the loop executes:
//!
//!   1. compute_compact_grid_size (A) — device-sizes the compact kernel
//!   2. device_compact_and_push (A) — compact A's spikes, dual-write ring+exhaust
//!   3. compute_compact_grid_size (B) — device-sizes for B
//!   4. device_compact_and_push (B) — compact B's spikes
//!   5. ring_buffer_read_and_adapt (B→A) — adapt A's thresholds from B's evidence
//!   6. ring_buffer_read_and_adapt (A→B) — adapt B's thresholds from A's evidence
//!   7. ring_buffer_threshold_recovery (A+B) — periodic recovery
//!   8. update_prev_spike_count (A+B) — bookkeeping
//!   9. decrement_step_counter — condition variable for WHILE loop exit
//!
//! All kernel function handles are obtained via cuModuleGetFunction on raw
//! CUmodule handles, since cudarc's CudaFunction doesn't expose the raw
//! CUfunction needed by cuGraphAddKernelNode.

use anyhow::{Context, Result};
use cudarc::driver::sys;
use std::ffi::CString;
use std::ptr;

/// Check CUDA result and return anyhow error.
pub fn check(result: sys::CUresult, ctx: &str) -> Result<()> {
    if result == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        anyhow::bail!("CUDA error in {}: {:?}", ctx, result)
    }
}

/// Load a CUfunction from a PTX file by name.
/// Returns the raw CUfunction handle for use in graph kernel nodes.
pub unsafe fn load_function_raw(
    ctx: sys::CUcontext,
    ptx_path: &str,
    function_name: &str,
) -> Result<(sys::CUmodule, sys::CUfunction)> {
    // Load the PTX module
    let ptx_data =
        std::fs::read(ptx_path).with_context(|| format!("Failed to read PTX: {}", ptx_path))?;

    // Ensure null-terminated
    let mut ptx_cstr = ptx_data.clone();
    if !ptx_cstr.ends_with(&[0]) {
        ptx_cstr.push(0);
    }

    let mut module: sys::CUmodule = ptr::null_mut();
    check(
        sys::cuModuleLoadData(&mut module, ptx_cstr.as_ptr() as *const _),
        &format!("cuModuleLoadData({})", ptx_path),
    )?;

    // Get the function
    let func_name = CString::new(function_name)
        .with_context(|| format!("Invalid function name: {}", function_name))?;

    let mut func: sys::CUfunction = ptr::null_mut();
    check(
        sys::cuModuleGetFunction(&mut func, module, func_name.as_ptr()),
        &format!("cuModuleGetFunction({})", function_name),
    )?;

    Ok((module, func))
}

/// Add a kernel node to a CUDA graph.
///
/// `dependencies` are the nodes that must complete before this node starts.
/// `kernel_params` is an array of pointers to the kernel arguments.
///
/// Returns the new node handle.
pub unsafe fn add_kernel_node(
    graph: sys::CUgraph,
    dependencies: &[sys::CUgraphNode],
    func: sys::CUfunction,
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
    shared_mem: u32,
    kernel_params: &mut [*mut std::ffi::c_void],
) -> Result<sys::CUgraphNode> {
    // Use CUDA_KERNEL_NODE_PARAMS_v2 (available with cuda-12050 feature)
    let params = sys::CUDA_KERNEL_NODE_PARAMS_v2_st {
        func,
        gridDimX: grid_dim.0,
        gridDimY: grid_dim.1,
        gridDimZ: grid_dim.2,
        blockDimX: block_dim.0,
        blockDimY: block_dim.1,
        blockDimZ: block_dim.2,
        sharedMemBytes: shared_mem,
        kernelParams: kernel_params.as_mut_ptr(),
        extra: ptr::null_mut(),
        kern: ptr::null_mut(), // not using CUkernel API
        ctx: ptr::null_mut(),  // use current context
    };

    let mut node: sys::CUgraphNode = ptr::null_mut();
    let dep_ptr = if dependencies.is_empty() {
        ptr::null()
    } else {
        dependencies.as_ptr()
    };
    let n_deps = dependencies.len();

    check(
        sys::cuGraphAddKernelNode_v2(&mut node, graph, dep_ptr, n_deps, &params),
        "cuGraphAddKernelNode_v2",
    )?;

    Ok(node)
}

/// Metadata for graph body population.
/// Contains all the device pointers needed by the coupling kernels.
pub struct GraphBodyConfig {
    // Compact kernel params for stream A
    pub spike_buf_a: sys::CUdeviceptr,
    pub spike_count_a: sys::CUdeviceptr,
    pub prev_count_a: sys::CUdeviceptr,
    pub grid_dim_a: sys::CUdeviceptr,

    // Compact kernel params for stream B
    pub spike_buf_b: sys::CUdeviceptr,
    pub spike_count_b: sys::CUdeviceptr,
    pub prev_count_b: sys::CUdeviceptr,
    pub grid_dim_b: sys::CUdeviceptr,

    // Ring buffers
    pub ring_a_buffer: sys::CUdeviceptr,
    pub ring_a_head: sys::CUdeviceptr,
    pub ring_a_overflow: sys::CUdeviceptr,
    pub ring_b_buffer: sys::CUdeviceptr,
    pub ring_b_head: sys::CUdeviceptr,
    pub ring_b_overflow: sys::CUdeviceptr,
    pub ring_capacity: u32,

    // Exhaust buffer (mapped host RAM)
    pub exhaust_buffer: sys::CUdeviceptr,
    pub exhaust_head: sys::CUdeviceptr,
    pub exhaust_capacity: u32,
    pub exhaust_enabled: i32,

    // Threshold buffers
    pub thresh_a: sys::CUdeviceptr,
    pub base_a: sys::CUdeviceptr,
    pub thresh_b: sys::CUdeviceptr,
    pub base_b: sys::CUdeviceptr,

    // Grid geometry
    pub n_voxels_x: i32,
    pub n_voxels_y: i32,
    pub n_voxels_z: i32,
    pub grid_origin_x: f32,
    pub grid_origin_y: f32,
    pub grid_origin_z: f32,
    pub voxel_size: f32,

    // Coupling params
    pub sensitivity_boost: f32,
    pub max_reduction: f32,
    pub decay_constant: f32,
    pub recovery_rate: f32,

    // Step counter for WHILE condition
    pub step_counter: sys::CUdeviceptr,
}

// NOTE: The actual body population (calling add_kernel_node for each
// of the 9 kernel nodes with proper dependencies) requires the
// GraphBodyConfig to be filled with real device pointers from the
// physics engines. This wiring happens in coupled_md.rs when the
// graph is constructed for a specific protein run.
//
// The functions above (load_function_raw, add_kernel_node) provide
// the building blocks. The orchestration code in coupled_md.rs will:
//   1. Load device_compact.ptx and ring_buffer.ptx via load_function_raw
//   2. Create GraphBodyConfig from engine GPU state
//   3. Call add_kernel_node for each step in the sequence
//   4. Set up dependencies between nodes (serial chain)
//   5. The WHILE conditional node wraps this body graph
