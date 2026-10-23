//! Replayable Coupling Graph — captured coupling kernel sequence.
//!
//! Architecture (Option 2):
//!   Host loop manages physics (all flags: --fast, --hysteresis, --prism-therm,
//!   --spike-percentile, --fused-steps, --hmr, --adaptive-dt, --multi-stream 8).
//!   Graph replays coupling (compact + adapt + recover) with zero host kernel launches.
//!
//! ```text
//! Host loop (for step in 0..outer_steps):
//!     for each engine: engine.step()   ← host-launched, protocol-aware
//!     coupling_graph.launch()          ← ONE graph launch replaying all coupling
//! ```
//!
//! The graph is captured ONCE by running the coupling kernel sequence on
//! the exchange stream while it's in capture mode. Then each step,
//! cuGraphLaunch replays it with zero host overhead.
//!
//! For multi-engine (2×N), the graph captures compact + push for ALL N
//! engines in each group, plus the cross-group threshold adaptation.

use anyhow::{Context, Result};
use cudarc::driver::sys;
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use std::sync::Arc;

use crate::graph_builder::check;

/// A captured coupling graph that can be replayed each simulation step.
///
/// Contains the instantiated graph exec handle. Call `launch()` each step
/// instead of individually launching compact + adapt + recover kernels.
pub struct CouplingReplayGraph {
    cu_graph: sys::CUgraph,
    cu_graph_exec: sys::CUgraphExec,
    raw_stream: sys::CUstream,
}

impl CouplingReplayGraph {
    /// Capture the coupling kernel sequence by running it once on a capturing stream.
    ///
    /// Steps:
    ///   1. Begin capture on exchange stream
    ///   2. Launch ALL coupling kernels (compact A/B, adapt, recover)
    ///      using the normal cudarc launch_builder API
    ///   3. End capture → get graph
    ///   4. Instantiate graph exec
    ///
    /// After construction, call `launch()` each step to replay.
    ///
    /// The `run_coupling_step` closure should launch all coupling kernels
    /// on the exchange stream using the normal launch_builder API.
    /// During capture, these launches are recorded, not executed.
    pub fn capture<F>(stream: &Arc<CudaStream>, run_coupling_step: F) -> Result<Self>
    where
        F: FnOnce() -> Result<()>,
    {
        let raw_stream = stream.cu_stream();

        // Begin capture
        unsafe {
            check(
                sys::cuStreamBeginCapture_v2(
                    raw_stream,
                    // RELAXED mode: allows cross-stream dependencies (the coupling
                    // kernels read buffers modified by physics engines on other streams).
                    // GLOBAL mode would require ALL streams to be captured simultaneously.
                    sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
                ),
                "cuStreamBeginCapture (coupling)",
            )?;
        }

        // Run the coupling kernels — recorded, not executed
        let step_result = run_coupling_step();

        // If the coupling step failed, abort the capture to restore the stream
        if let Err(e) = step_result {
            log::warn!("  Coupling step failed during capture: {}", e);
            // Abort capture to restore stream to normal mode
            let mut dummy_graph: sys::CUgraph = std::ptr::null_mut();
            unsafe {
                let _ = sys::cuStreamEndCapture(raw_stream, &mut dummy_graph);
                if !dummy_graph.is_null() {
                    sys::cuGraphDestroy(dummy_graph);
                }
            }
            anyhow::bail!("Coupling step failed during capture: {}", e);
        }

        // End capture
        let mut cu_graph: sys::CUgraph = std::ptr::null_mut();
        unsafe {
            check(
                sys::cuStreamEndCapture(raw_stream, &mut cu_graph),
                "cuStreamEndCapture (coupling)",
            )?;
        }

        if cu_graph.is_null() {
            anyhow::bail!("Coupling graph capture produced null graph — no kernels launched?");
        }

        // Count nodes
        let mut n_nodes = 0usize;
        unsafe {
            sys::cuGraphGetNodes(cu_graph, std::ptr::null_mut(), &mut n_nodes);
        }

        // Instantiate
        let mut cu_graph_exec: sys::CUgraphExec = std::ptr::null_mut();
        unsafe {
            check(
                sys::cuGraphInstantiateWithFlags(&mut cu_graph_exec, cu_graph, 0),
                "cuGraphInstantiateWithFlags (coupling)",
            )?;
        }

        log::info!(
            "  Coupling graph: captured {} kernel nodes, instantiated for replay",
            n_nodes
        );

        Ok(Self {
            cu_graph,
            cu_graph_exec,
            raw_stream,
        })
    }

    /// Replay the captured coupling sequence.
    /// This is a SINGLE cuGraphLaunch call that replays all coupling
    /// kernels (compact, adapt, recover) with zero host overhead.
    pub fn launch(&self) -> Result<()> {
        unsafe {
            check(
                sys::cuGraphLaunch(self.cu_graph_exec, self.raw_stream),
                "cuGraphLaunch (coupling)",
            )?;
        }
        Ok(())
    }

    /// Number of kernel nodes in the graph.
    pub fn node_count(&self) -> usize {
        let mut n = 0usize;
        unsafe {
            sys::cuGraphGetNodes(self.cu_graph, std::ptr::null_mut(), &mut n);
        }
        n
    }
}

impl Drop for CouplingReplayGraph {
    fn drop(&mut self) {
        unsafe {
            if !self.cu_graph_exec.is_null() {
                sys::cuGraphExecDestroy(self.cu_graph_exec);
            }
            if !self.cu_graph.is_null() {
                sys::cuGraphDestroy(self.cu_graph);
            }
        }
    }
}
