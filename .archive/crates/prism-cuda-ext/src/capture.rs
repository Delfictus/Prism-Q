//! Stream Capture — record physics kernel launches as a CUDA graph.
//!
//! Uses cuStreamBeginCapture / cuStreamEndCapture to capture the exact
//! kernel sequence launched by the engine's step() method, including all
//! 78 parameters of nhs_amber_fused_step, multi_lif, and LADD kernels.
//!
//! The captured graph becomes a child node inside the WHILE conditional
//! graph body. No engine code is modified — the capture is transparent.
//!
//! This is the same technique used by GROMACS and OpenMM for graph-based
//! MD execution on NVIDIA GPUs.

use anyhow::Result;
use cudarc::driver::sys;
use cudarc::driver::CudaStream;
use std::sync::Arc;

use crate::graph_builder::check;

/// Captured physics graph from one engine step.
///
/// Contains the raw CUgraph extracted from stream capture.
/// This graph can be added as a child node inside a larger graph.
pub struct CapturedPhysicsGraph {
    /// The captured graph (owned — will be destroyed on drop)
    pub(crate) cu_graph: sys::CUgraph,
}

impl CapturedPhysicsGraph {
    /// Begin stream capture on the given stream.
    ///
    /// After calling this, ALL kernel launches on this stream are
    /// recorded as graph nodes instead of being executed. The
    /// kernels do NOT actually run during capture.
    ///
    /// Call `end_capture()` after running one engine step.
    pub fn begin_capture(stream: &Arc<CudaStream>) -> Result<()> {
        let raw_stream = stream.cu_stream();
        unsafe {
            check(
                sys::cuStreamBeginCapture_v2(
                    raw_stream,
                    sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
                ),
                "cuStreamBeginCapture",
            )?;
        }
        log::debug!("  Stream capture: STARTED");
        Ok(())
    }

    /// End stream capture and return the captured graph.
    ///
    /// The returned graph contains all kernel nodes that were
    /// "launched" (actually recorded) since begin_capture().
    pub fn end_capture(stream: &Arc<CudaStream>) -> Result<Self> {
        let raw_stream = stream.cu_stream();
        let mut cu_graph: sys::CUgraph = std::ptr::null_mut();

        unsafe {
            check(
                sys::cuStreamEndCapture(raw_stream, &mut cu_graph),
                "cuStreamEndCapture",
            )?;
        }

        if cu_graph.is_null() {
            anyhow::bail!("Stream capture produced null graph — no kernels were launched?");
        }

        // Count nodes for diagnostics
        let mut n_nodes = 0usize;
        unsafe {
            sys::cuGraphGetNodes(cu_graph, std::ptr::null_mut(), &mut n_nodes);
        }

        log::info!(
            "  Stream capture: COMPLETED ({} kernel nodes captured)",
            n_nodes
        );
        Ok(Self { cu_graph })
    }

    /// Get the raw CUgraph handle for adding as a child node.
    pub fn raw(&self) -> sys::CUgraph {
        self.cu_graph
    }

    /// Add this captured graph as a child graph node inside a parent graph.
    ///
    /// This is how you embed the physics kernel sequence inside
    /// the WHILE conditional graph body.
    pub fn add_as_child_node(
        &self,
        parent_graph: sys::CUgraph,
        dependencies: &[sys::CUgraphNode],
    ) -> Result<sys::CUgraphNode> {
        let mut node: sys::CUgraphNode = std::ptr::null_mut();
        let dep_ptr = if dependencies.is_empty() {
            std::ptr::null()
        } else {
            dependencies.as_ptr()
        };

        unsafe {
            check(
                sys::cuGraphAddChildGraphNode(
                    &mut node,
                    parent_graph,
                    dep_ptr,
                    dependencies.len(),
                    self.cu_graph,
                ),
                "cuGraphAddChildGraphNode(physics)",
            )?;
        }

        log::info!("  Physics graph added as child node in WHILE body");
        Ok(node)
    }
}

impl Drop for CapturedPhysicsGraph {
    fn drop(&mut self) {
        if !self.cu_graph.is_null() {
            unsafe {
                sys::cuGraphDestroy(self.cu_graph);
            }
        }
    }
}

/// Capture one full TWIN step (both engines + coupling) as a graph.
///
/// This is the master capture function that records:
///   1. Engine A step (fused + multi_lif + LADD)
///   2. Engine B step (fused + multi_lif + LADD)
///   3. Device compact A + B (dual-write ring + exhaust)
///   4. Ring buffer adapt (threshold coupling)
///   5. Threshold recovery
///
/// The captured graph is then wrapped in a WHILE conditional node
/// for autonomous execution.
///
/// IMPORTANT: During capture, the engine's step() does NOT actually
/// execute the kernels. It just records them. So the first "step" of
/// the simulation is effectively a recording pass, not a physics pass.
/// The actual simulation starts when the graph is launched.
///
/// Usage:
/// ```rust,ignore
/// // 1. Begin capture
/// CapturedPhysicsGraph::begin_capture(&stream)?;
///
/// // 2. Run one iteration (recorded, not executed)
/// engine_a.step()?;    // 78-param fused + multi_lif + LADD captured
/// engine_b.step()?;    // same for B
/// // coupling kernels would go here too
///
/// // 3. End capture
/// let physics_graph = CapturedPhysicsGraph::end_capture(&stream)?;
///
/// // 4. Add as child of WHILE body
/// let node = physics_graph.add_as_child_node(body_graph, &[])?;
/// ```
pub fn capture_one_twin_step_usage_note() {
    // This is a documentation function. The actual capture happens
    // in coupled_md.rs where the engine instances are available.
}
