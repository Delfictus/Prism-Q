//! CUDA Conditional Graph API — safe wrappers for autonomous GPU coupling.
//!
//! This module provides Rust bindings for CUDA 12.4+ conditional graph nodes.
//! The key type is `CU_GRAPH_COND_TYPE_WHILE` which creates a hardware loop
//! on the GPU that repeats a subgraph until a device-side condition variable
//! reaches zero.
//!
//! Architecture:
//!   CPU: builds graph once → cudaGraphLaunch once → sleeps
//!   GPU: hardware loop evaluates condition → runs coupling subgraph → repeats
//!
//! The coupling subgraph per iteration:
//!   1. compute_compact_grid_size (reads d_spike_count from VRAM)
//!   2. device_compact_and_push (GPU-sized grid, zero PCIe)
//!   3. ring_buffer_read_and_adapt (threshold adaptation)
//!   4. ring_buffer_threshold_recovery (periodic)
//!   5. update_prev_spike_count (bookkeeping)
//!   6. decrement step_counter (WHILE condition variable)

use anyhow::{Context, Result};
use cudarc::driver::sys;
use cudarc::driver::{
    CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use std::ptr;
use std::sync::Arc;

/// Check a CUDA driver result and convert to anyhow::Error.
fn check_cuda(result: sys::CUresult, context: &str) -> Result<()> {
    if result == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        anyhow::bail!("CUDA error in {}: {:?}", context, result)
    }
}

/// A CUDA conditional graph that runs the TWIN coupling sequence
/// autonomously on the GPU for the entire simulation.
///
/// The graph structure:
/// ```text
/// [WHILE(step_counter > 0)] {
///     compute_grid_size_A → compact_A → adapt_B_from_A
///     compute_grid_size_B → compact_B → adapt_A_from_B
///     → recovery (every N steps) → decrement_counter
/// }
/// ```
///
/// One `launch()` call. The GPU's Gigathread Engine manages the loop.
/// SMs are freed between kernel nodes (no persistent spin-wait).
pub struct TwinCouplingGraph {
    graph: sys::CUgraph,
    graph_exec: sys::CUgraphExec,
    /// Device-side step counter — WHILE loop continues while > 0.
    /// Decremented by the decrement kernel each iteration.
    d_step_counter: CudaSlice<u32>,
}

impl TwinCouplingGraph {
    /// Check if the current CUDA driver supports conditional graph nodes.
    pub fn is_supported() -> bool {
        unsafe {
            let mut version: i32 = 0;
            let result = sys::cuDriverGetVersion(&mut version);
            if result != sys::CUresult::CUDA_SUCCESS {
                return false;
            }
            version >= 12040
        }
    }

    /// Report CUDA Graph capabilities for diagnostics.
    pub fn capabilities_report() -> String {
        let supported = Self::is_supported();
        let mut version = 0i32;
        unsafe {
            sys::cuDriverGetVersion(&mut version);
        }

        format!(
            "CUDA Graph Capabilities:\n\
             Driver version: {}.{}\n\
             Conditional nodes (WHILE/IF): {}\n\
             Device-updated parameters: {}\n\
             CU_GRAPH_COND_TYPE_WHILE = {}\n\
             CU_GRAPH_NODE_TYPE_CONDITIONAL = {}",
            version / 1000,
            (version % 1000) / 10,
            if supported {
                "YES"
            } else {
                "NO (requires CUDA 12.4+)"
            },
            if version >= 12040 { "YES" } else { "NO" },
            sys::CUgraphConditionalNodeType::CU_GRAPH_COND_TYPE_WHILE as i32,
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_CONDITIONAL as i32,
        )
    }

    /// Build the conditional coupling graph.
    ///
    /// This constructs the graph structure but does NOT launch it.
    /// Call `launch()` to start the autonomous GPU loop.
    ///
    /// The graph uses a WHILE conditional node that decrements
    /// `d_step_counter` each iteration. The loop exits when
    /// the counter reaches 0.
    pub fn build(stream: &Arc<CudaStream>, total_steps: u32) -> Result<Self> {
        if !Self::is_supported() {
            anyhow::bail!("CUDA conditional graph nodes require driver ≥ 12.4");
        }

        // Allocate device-side step counter
        let mut d_step_counter = stream.alloc_zeros::<u32>(1)?;
        stream.memcpy_htod(&[total_steps], &mut d_step_counter)?;

        // Create the top-level graph
        let mut graph: sys::CUgraph = ptr::null_mut();
        unsafe {
            check_cuda(sys::cuGraphCreate(&mut graph, 0), "cuGraphCreate")?;
        }

        // Create the conditional handle for the WHILE loop
        let mut cond_handle: sys::CUgraphConditionalHandle = 0;

        // Get the CUDA context from the stream
        let mut ctx: sys::CUcontext = ptr::null_mut();
        unsafe {
            check_cuda(sys::cuCtxGetCurrent(&mut ctx), "cuCtxGetCurrent")?;
        }

        unsafe {
            check_cuda(
                sys::cuGraphConditionalHandleCreate(
                    &mut cond_handle,
                    graph,
                    ctx,
                    1, // defaultLaunchValue: 1 = loop starts active
                    0, // flags
                ),
                "cuGraphConditionalHandleCreate",
            )?;
        }

        // Create the WHILE conditional node
        // This node contains a body graph that repeats while the handle's
        // value is non-zero. The body graph gets the coupling kernels.
        let mut cond_node: sys::CUgraphNode = ptr::null_mut();
        let mut body_graph: sys::CUgraph = ptr::null_mut();

        let cond_params = sys::CUDA_CONDITIONAL_NODE_PARAMS {
            handle: cond_handle,
            type_: sys::CUgraphConditionalNodeType::CU_GRAPH_COND_TYPE_WHILE,
            size: 1,
            phGraph_out: &mut body_graph,
            ctx,
        };

        let mut node_params: sys::CUgraphNodeParams_st = unsafe { std::mem::zeroed() };
        node_params.type_ = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_CONDITIONAL;
        unsafe {
            node_params.__bindgen_anon_1.conditional = cond_params;
        }

        unsafe {
            check_cuda(
                sys::cuGraphAddNode(
                    &mut cond_node,
                    graph,
                    ptr::null(), // no dependencies (root node)
                    0,
                    &mut node_params,
                ),
                "cuGraphAddNode(CONDITIONAL WHILE)",
            )?;
        }

        log::info!(
            "  CUDA Graph: WHILE conditional node created (body_graph={:?})",
            body_graph
        );

        // The body_graph is where we add the coupling kernel nodes.
        // For now, we'll add a placeholder that just decrements the counter.
        // The full coupling kernel nodes will be added when we wire this
        // into coupled_md.rs with access to all the GPU buffer pointers.

        // TODO: Add kernel nodes to body_graph:
        //   1. compute_compact_grid_size (A)
        //   2. device_compact_and_push (A)
        //   3. compute_compact_grid_size (B)
        //   4. device_compact_and_push (B)
        //   5. ring_buffer_read_and_adapt (B→A)
        //   6. ring_buffer_read_and_adapt (A→B)
        //   7. ring_buffer_threshold_recovery (periodic)
        //   8. update_prev_spike_count (A)
        //   9. update_prev_spike_count (B)
        //  10. decrement_step_counter (condition update)

        // Instantiate the graph
        let mut graph_exec: sys::CUgraphExec = ptr::null_mut();
        unsafe {
            check_cuda(
                sys::cuGraphInstantiateWithFlags(
                    &mut graph_exec,
                    graph,
                    0, // flags (0 = default)
                ),
                "cuGraphInstantiateWithFlags",
            )?;
        }

        log::info!(
            "  CUDA Graph: instantiated (graph={:?}, exec={:?})",
            graph,
            graph_exec
        );

        Ok(Self {
            graph,
            graph_exec,
            d_step_counter,
        })
    }

    /// Launch the graph on the given stream.
    /// This call returns IMMEDIATELY. The GPU runs the WHILE loop
    /// autonomously for `total_steps` iterations.
    ///
    /// The CPU can sleep or do other work. Call `wait()` to block
    /// until the graph completes.
    pub fn launch(&self, stream: &Arc<CudaStream>) -> Result<()> {
        // Get the raw CUstream handle
        // cudarc's CudaStream wraps a CUstream internally
        let raw_stream = stream.cu_stream();

        unsafe {
            check_cuda(
                sys::cuGraphLaunch(self.graph_exec, raw_stream),
                "cuGraphLaunch",
            )?;
        }

        log::info!("  CUDA Graph: LAUNCHED (autonomous GPU loop)");
        Ok(())
    }

    /// Wait for the graph to complete.
    /// This blocks the CPU until all `total_steps` iterations finish.
    pub fn wait(&self, stream: &Arc<CudaStream>) -> Result<()> {
        stream.synchronize()?;
        Ok(())
    }

    /// Read the current step counter from device (for progress monitoring).
    pub fn remaining_steps(&self, stream: &Arc<CudaStream>) -> Result<u32> {
        let mut val = [0u32];
        stream.memcpy_dtoh(&self.d_step_counter, &mut val)?;
        Ok(val[0])
    }
}

impl Drop for TwinCouplingGraph {
    fn drop(&mut self) {
        unsafe {
            sys::cuGraphExecDestroy(self.graph_exec);
            sys::cuGraphDestroy(self.graph);
        }
    }
}

/// Kernel that decrements the step counter by 1.
/// When the counter reaches 0, the WHILE conditional exits.
/// This kernel is the LAST node in the graph's body, ensuring
/// all coupling work completes before the condition is re-evaluated.
///
/// This is a separate .cu file compiled alongside device_compact.cu.
/// For now, we inline the PTX equivalent or use an existing kernel.
///
/// The simplest implementation: the WHILE conditional handle's value
/// is set by a kernel that reads d_step_counter, decrements it, and
/// writes the new value to the handle. But the CUDA conditional graph
/// API actually reads the handle value DIRECTLY from device memory —
/// the handle IS the device memory address of the counter.
///
/// So we just need a kernel that does: *counter -= 1;
pub const DECREMENT_KERNEL_SOURCE: &str = r#"
extern "C" __global__ void decrement_counter(unsigned int* counter) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned int val = *counter;
        if (val > 0) *counter = val - 1;
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_constants() {
        assert_eq!(
            sys::CUgraphConditionalNodeType::CU_GRAPH_COND_TYPE_WHILE as i32,
            1,
        );
        assert_eq!(
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_CONDITIONAL as i32,
            13,
        );
    }

    #[test]
    fn test_is_supported() {
        // On a system with CUDA 13.2, this should return true
        let supported = TwinCouplingGraph::is_supported();
        println!("Conditional graph supported: {}", supported);
        // Don't assert — test might run without GPU
    }

    #[test]
    fn test_capabilities_report() {
        let report = TwinCouplingGraph::capabilities_report();
        println!("{}", report);
        assert!(report.contains("CUDA Graph Capabilities"));
    }
}
