//! CUDA 13.x `cuGraphAddNode` — unified graph-builder Rust bindings.
//!
//! Begun 2026-05-03 — operator directive "migration to cuGraphAddNode for
//! a monolithic splice as planned so we don't continue to have
//! childgraphIF failures bc we are using the wrong name conventions".
//!
//! ## Why this module exists
//!
//! cudarc 0.18.2 does not expose CUDA 13.x's `cuGraphAddNode` symbol
//! (only `cuGraphInstantiate` is bound; the unified
//! `cuGraphAddNode_v2` / `_v3` family is missing). Two Rust call
//! sites still drive graph construction through the legacy per-type
//! adders:
//!
//! | Call site                                | Legacy adder                  |
//! |------------------------------------------|-------------------------------|
//! | `graph_capture::PrismGraphTemplate::splice_child_template` | `cuGraphAddChildGraphNode` |
//! | `captured_pipeline::add_burst_marker_memset_node`           | `cuGraphAddMemsetNode`     |
//!
//! The legacy adders mishandle CUDA 13.x conditional-handle metadata
//! attached to body subgraphs (the source of the operator's reported
//! "ChildGraph IF" failures), so we route Rust call sites through C++
//! FFI helpers (`prism_graph_add_*_v3_ffi` in
//! `src/cuda/graph_node.cu`) that internally invoke the unified
//! `cudaGraphAddNode`.
//!
//! ## Migration status
//!
//! Phase 1 (this commit):
//!   - C++ FFI scaffold (`src/cuda/graph_node.cu`) — DONE
//!   - Rust extern + safe wrappers (this file) — DONE
//!   - Pilot migration: `graph_capture::splice_child_template`     ← in progress
//!
//! Phase 2 (follow-up):
//!   - Migrate `captured_pipeline::add_burst_marker_memset_node`
//!   - Audit comment-only references to legacy adders for accuracy
//!   - Retire `cuGraphAddChildGraphNode` / `cuGraphAddMemsetNode`
//!     extern declarations once both sites are migrated and a
//!     captured-graph run validates without ChildGraph-IF errors.

use cudarc::driver::sys::{CUgraph, CUgraphNode};
use std::ffi::c_void;

#[link(name = "graph_node", kind = "static")]
extern "C" {
    fn prism_graph_add_child_node_v3_ffi(
        parent_graph: CUgraph,
        p_dependencies: *const CUgraphNode,
        num_deps: usize,
        child_template: CUgraph,
        p_out_node: *mut CUgraphNode,
    ) -> i32;

    fn prism_graph_add_memset_node_v3_ffi(
        parent_graph: CUgraph,
        p_dependencies: *const CUgraphNode,
        num_deps: usize,
        dst: *mut c_void,
        pitch: usize,
        value_u32: u32,
        element_size: u32,
        width: usize,
        height: usize,
        p_out_node: *mut CUgraphNode,
    ) -> i32;

    /// TIER 8 — preflight legality check.  Returns 0 on successful
    /// inspection; nonzero cudaError_t on inspection failure.  On
    /// success, the node-count outputs are populated.  Any nonzero
    /// conditional / alloc / free count means the child graph is
    /// **NOT** safe to splice via `cudaGraphAddNode(GRAPH)`.
    fn prism_graph_is_splice_legal_ffi(
        child_graph: CUgraph,
        p_out_total_nodes: *mut usize,
        p_out_conditional_nodes: *mut usize,
        p_out_alloc_nodes: *mut usize,
        p_out_free_nodes: *mut usize,
    ) -> i32;

    // TIER 8 — WHILE FFI scaffold (WHILE-FFI-SCAFFOLD-001, 2026-05-04).
    //
    // Mirrors the G26 SWITCH FFI surface in adjudicator.cu:1119/1137
    // but specialised to `cudaGraphCondTypeWhile` and `size = 1`. See
    // the C++ side in `crates/prism-nhs/src/cuda/graph_node.cu` for
    // the implementation, including the `CUDART_VERSION >= 12040`
    // gate.
    //
    // FFI added but NOT WIRED into any production graph topology this
    // commit — R6 lane will install the bounded post-T7 deferred-drain
    // WHILE via `captured_pipeline.rs` in a follow-up commit.

    /// **TIER 8 — Create a parent-owned WHILE conditional handle.**
    ///
    /// Wraps `cudaGraphConditionalHandleCreate` with
    /// `flags = cudaGraphCondAssignDefault` so `default_value` is
    /// applied at every graph launch.  The CUDA conditional handle is
    /// `typedef unsigned long long` so it round-trips through `u64`.
    fn prism_graph_create_while_handle_ffi(
        parent_graph: CUgraph,
        default_value: u32,
        p_out_handle: *mut u64,
    ) -> i32;

    /// **TIER 8 — Add a parent-owned WHILE conditional node.**
    ///
    /// Wraps `cudaGraphAddNode` with `nodeParams.type =
    /// cudaGraphNodeTypeConditional`,
    /// `nodeParams.conditional.type = cudaGraphCondTypeWhile`, and
    /// `nodeParams.conditional.size = 1` (the only allowed value for
    /// WHILE per `driver_types.h:3572`).  After the add, the body
    /// subgraph (`phGraph_out[0]`) is copied out via
    /// `p_out_body_subgraph` so the caller can populate it.
    fn prism_graph_add_while_node_ffi(
        parent_graph: CUgraph,
        p_dependencies: *const CUgraphNode,
        num_deps: usize,
        handle_v: u64,
        p_out_conditional_node: *mut CUgraphNode,
        p_out_body_subgraph: *mut CUgraph,
    ) -> i32;
}

/// **TIER 8 (2026-05-03)** — Splice-legality preflight result.
///
/// `is_legal() == true` ⇒ child graph is splice-legal under PRISM's
/// GRAPH-SPLICE-001 invariant.  Conditional, allocation, and free
/// nodes must be lifted to the parent/control graph.
#[derive(Debug, Clone, Copy)]
pub struct SpliceLegalityReport {
    pub total_nodes: usize,
    pub conditional_count: usize,
    pub allocation_count: usize,
    pub free_count: usize,
}

impl SpliceLegalityReport {
    /// Cheap predicate: this child graph is safe to splice.
    pub fn is_legal(&self) -> bool {
        self.conditional_count == 0 && self.allocation_count == 0 && self.free_count == 0
    }
}

/// **TIER 8 (2026-05-03)** — Splice-attempt error taxonomy.
///
/// Distinguishes the three failure modes a `cudaGraphAddNode(GRAPH)`
/// invocation can hit so the orchestrator can react with policy:
///   - `Cuda(rc)` — generic CUDA driver failure during the splice
///     itself (e.g., out-of-memory, invalid handle).
///   - `NullNode` — defensive guard; CUDA reported success but
///     returned a null handle (driver bug or contract violation).
///   - `Illegal(report)` — the preflight check found conditional
///     nodes in the child template; the splice was NEVER attempted.
///     This is the **PRISM-specific** error that fires before CUDA
///     can return 801 on the actual splice.
#[derive(Debug)]
pub enum SpliceError {
    /// Inspection or splice itself returned a CUDA driver error.
    Cuda(i32),
    /// Splice succeeded but yielded a null node handle.
    NullNode,
    /// Preflight legality check rejected the child template — it
    /// contains conditional nodes that CUDA 13.x cannot splice.
    Illegal(SpliceLegalityReport),
}

impl std::fmt::Display for SpliceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpliceError::Cuda(rc) => write!(f, "splice cuda error: rc={}", rc),
            SpliceError::NullNode => write!(f, "splice succeeded but returned null node handle"),
            SpliceError::Illegal(r) => write!(
                f,
                "splice ILLEGAL — child graph contains conditional={} alloc={} free={} \
                 node(s) of {} total; CUDA 13.x graph-splice policy rejects \
                 cudaGraphAddNode(GRAPH) children with conditional/allocation/free \
                 nodes. TIER 8 fix: lift control/allocation nodes to the parent graph.",
                r.conditional_count, r.allocation_count, r.free_count, r.total_nodes
            ),
        }
    }
}

impl std::error::Error for SpliceError {}

/// **TIER 8 (2026-05-03)** — Inspect `child_graph` for splice
/// legality without actually attempting a splice.
///
/// # Safety
/// `child_graph` must be a valid live `CUgraph` handle.
pub unsafe fn splice_legality_check(child_graph: CUgraph) -> Result<SpliceLegalityReport, i32> {
    let mut total: usize = 0;
    let mut cond: usize = 0;
    let mut alloc: usize = 0;
    let mut free: usize = 0;
    let rc = prism_graph_is_splice_legal_ffi(
        child_graph,
        &mut total as *mut _,
        &mut cond as *mut _,
        &mut alloc as *mut _,
        &mut free as *mut _,
    );
    if rc != 0 {
        return Err(rc);
    }
    Ok(SpliceLegalityReport {
        total_nodes: total,
        conditional_count: cond,
        allocation_count: alloc,
        free_count: free,
    })
}

/// Splice a complete child template into `parent_graph` as one
/// monolithic GRAPH-type node (CUDA 13.x `cudaGraphAddNode` with
/// `nodeParams.type = cudaGraphNodeTypeGraph`).
///
/// **TIER 8 (2026-05-03)** — runs `splice_legality_check` BEFORE
/// the splice attempt.  If the child template contains any
/// conditional / allocation / free nodes, returns
/// `Err(SpliceError::Illegal(report))`
/// without ever calling `cudaGraphAddNode`.  This is the PRISM-
/// specific gate that fires before CUDA can return 801
/// (cudaErrorNotSupported) on the splice itself.
///
/// Returns the new node handle on success.  The child template is
/// **copied** by CUDA — subsequent mutations to `child_template` do
/// not propagate to the spliced instance.
///
/// # Errors
///
/// - `SpliceError::Illegal(report)` — preflight rejected the splice
///   (see [`SpliceLegalityReport`]).  No splice was attempted.
/// - `SpliceError::Cuda(rc)` — CUDA returned a non-success
///   `cudaError_t` from either inspection or the splice itself.
/// - `SpliceError::NullNode` — splice reported success but returned
///   a null handle (defensive).
///
/// # Safety
///
/// Caller must ensure:
/// - `parent_graph` and `child_template` are valid live `CUgraph` handles.
/// - All `CUgraphNode` entries in `deps` belong to `parent_graph`.
/// - `parent_graph` is not currently being captured by an active
///   `cuStreamBeginCapture` window on a stream other than the build
///   thread (THREAD_LOCAL capture mode keeps build-thread captures
///   safe — see `captured_pipeline.rs:1381` rationale).
pub unsafe fn add_child_graph_node_v3(
    parent_graph: CUgraph,
    deps: &[CUgraphNode],
    child_template: CUgraph,
) -> Result<CUgraphNode, SpliceError> {
    // TIER 8 preflight — fail PRISM-side BEFORE CUDA returns 801.
    let report = splice_legality_check(child_template).map_err(SpliceError::Cuda)?;
    if !report.is_legal() {
        return Err(SpliceError::Illegal(report));
    }

    // Splice via the unified CUDA 13.x cudaGraphAddNode (TIER 7).
    let mut out_node: CUgraphNode = std::ptr::null_mut();
    let p_deps = if deps.is_empty() {
        std::ptr::null()
    } else {
        deps.as_ptr()
    };
    let rc = prism_graph_add_child_node_v3_ffi(
        parent_graph,
        p_deps,
        deps.len(),
        child_template,
        &mut out_node as *mut _,
    );
    if rc != 0 {
        return Err(SpliceError::Cuda(rc));
    }
    if out_node.is_null() {
        return Err(SpliceError::NullNode);
    }
    Ok(out_node)
}

/// Add a MEMSET node via CUDA 13.x `cudaGraphAddNode`.
///
/// `pitch` is bytes per row; for tight 1-D MEMSETs use
/// `element_size * width`. `value_u32` is truncated to the low
/// `element_size` bytes per CUDA semantics. `element_size` must be
/// 1, 2, or 4 (CUDA-imposed).
///
/// # Safety
///
/// Same invariants as `add_child_graph_node_v3` for the parent graph
/// + deps. `dst` must be a valid CUdeviceptr cast to `*mut c_void`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn add_memset_node_v3(
    parent_graph: CUgraph,
    deps: &[CUgraphNode],
    dst: *mut c_void,
    pitch: usize,
    value_u32: u32,
    element_size: u32,
    width: usize,
    height: usize,
) -> Result<CUgraphNode, i32> {
    let mut out_node: CUgraphNode = std::ptr::null_mut();
    let p_deps = if deps.is_empty() {
        std::ptr::null()
    } else {
        deps.as_ptr()
    };
    let rc = prism_graph_add_memset_node_v3_ffi(
        parent_graph,
        p_deps,
        deps.len(),
        dst,
        pitch,
        value_u32,
        element_size,
        width,
        height,
        &mut out_node as *mut _,
    );
    if rc != 0 {
        return Err(rc);
    }
    if out_node.is_null() {
        return Err(-1);
    }
    Ok(out_node)
}

// ═══════════════════════════════════════════════════════════════════════
// TIER 8 — WHILE FFI scaffold (WHILE-FFI-SCAFFOLD-001, 2026-05-04).
//
// Safe wrappers over `prism_graph_create_while_handle_ffi` and
// `prism_graph_add_while_node_ffi` mirroring the G26 SWITCH path
// shape.  Reuse the existing `SpliceError` taxonomy for the node-add
// path (the same three failure classes apply: CUDA driver error, null
// handle, illegal — though "illegal" cannot fire for parent-owned
// WHILE nodes since there is no preflight on the parent graph itself).
// ═══════════════════════════════════════════════════════════════════════

/// **TIER 8 (2026-05-04)** — Create a parent-owned WHILE conditional
/// handle bound to `parent_graph`.
///
/// `default_value` controls whether the body runs at least once at
/// instantiation:
/// - `1` — enter body once by default; the predicate-bridge kernel
///   inside the body uses `cudaGraphSetConditional(handle, 0)` to
///   exit.
/// - `0` — body NEVER runs unless the predicate-bridge sets the
///   handle to a non-zero value first; safe default for "no-op WHILE
///   if predicate never fires".
///
/// Returns the conditional handle as a `u64` (the CUDA
/// `cudaGraphConditionalHandle` typedef is `unsigned long long`).
///
/// # Errors
/// Returns the underlying `cudaError_t` cast to `i32` on failure.  In
/// particular, returns `cudaErrorNotSupported` (901) when the build
/// targets a CUDA toolkit older than 12.4.
///
/// # Safety
/// `parent_graph` must be a valid live `CUgraph` not currently being
/// captured.  WHILE is parent-owned only — calling this for a child
/// template handle is a TIER 8 invariant violation; the runtime
/// catches the consequence at splice time via
/// `splice_legality_check`, but callers MUST NOT pass a child template
/// handle here.
pub unsafe fn create_while_handle(
    parent_graph: CUgraph,
    default_value: u32,
) -> Result<u64, SpliceError> {
    let mut handle: u64 = 0;
    let rc =
        prism_graph_create_while_handle_ffi(parent_graph, default_value, &mut handle as *mut _);
    if rc != 0 {
        return Err(SpliceError::Cuda(rc));
    }
    if handle == 0 {
        return Err(SpliceError::NullNode);
    }
    Ok(handle)
}

/// **TIER 8 (2026-05-04)** — Add a parent-owned WHILE conditional
/// node to `parent_graph`.
///
/// Returns `(while_cond_node, body_subgraph)`.  The body subgraph is
/// CUDA-owned with the same lifetime as the conditional node itself
/// (per `driver_types.h:3574`).  The caller is responsible for
/// populating the body — typically the predicate-bridge kernel + a
/// drain pump — BEFORE the parent graph is instantiated.
///
/// # Errors
/// - `SpliceError::Cuda(rc)` — CUDA returned a non-success
///   `cudaError_t` from `cudaGraphAddNode`.  In particular,
///   `cudaErrorNotSupported` (901) indicates a pre-CUDA-12.4 toolkit.
/// - `SpliceError::NullNode` — CUDA reported success but returned a
///   null conditional-node or body-subgraph handle (defensive).
///
/// # Safety
/// All `CUgraphNode` entries in `deps` must belong to `parent_graph`.
/// `parent_graph` must not be inside an active stream-capture window
/// on a foreign thread. The `handle` must belong to this graph or the
/// top-level owner of this CUDA-owned conditional body (CUDA 13
/// cuda_runtime_api.h: conditional handles may bind nodes in graph children).
/// Never copy a graph containing conditional nodes through a GRAPH child node.
pub unsafe fn add_while_node_v3(
    parent_graph: CUgraph,
    deps: &[CUgraphNode],
    handle: u64,
) -> Result<(CUgraphNode, CUgraph), SpliceError> {
    if handle == 0 {
        return Err(SpliceError::NullNode);
    }
    let mut out_node: CUgraphNode = std::ptr::null_mut();
    let mut out_body: CUgraph = std::ptr::null_mut();
    let p_deps = if deps.is_empty() {
        std::ptr::null()
    } else {
        deps.as_ptr()
    };
    let rc = prism_graph_add_while_node_ffi(
        parent_graph,
        p_deps,
        deps.len(),
        handle,
        &mut out_node as *mut _,
        &mut out_body as *mut _,
    );
    if rc != 0 {
        return Err(SpliceError::Cuda(rc));
    }
    if out_node.is_null() || out_body.is_null() {
        return Err(SpliceError::NullNode);
    }
    Ok((out_node, out_body))
}

// Reuse the existing native four-body SWITCH factory. It creates no gearbox
// kernels; callers populate each returned body with their native templates.
#[cfg(feature = "v2_ignition")]
extern "C" {
    fn prism_gearbox_wire_with_handle_ffi(graph: CUgraph, predicate: CUgraphNode,
        handle: u64, node: *mut CUgraphNode, bodies: *mut CUgraph) -> i32;
}
#[cfg(feature = "v2_ignition")]
pub(crate) unsafe fn add_switch4_node(graph: CUgraph, predicate: CUgraphNode, handle: u64)
    -> Result<(CUgraphNode, [CUgraph;4]), i32> {
    let mut node = std::ptr::null_mut();
    let mut bodies = [std::ptr::null_mut();4];
    let rc = prism_gearbox_wire_with_handle_ffi(graph,predicate,handle,&mut node,bodies.as_mut_ptr());
    if rc != 0 { return Err(rc); }
    if node.is_null() || bodies.iter().any(|body| body.is_null()) { return Err(1); }
    Ok((node,bodies))
}
