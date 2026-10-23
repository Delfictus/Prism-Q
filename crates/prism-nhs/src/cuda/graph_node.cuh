// ═══════════════════════════════════════════════════════════════════════
// PRISM-4D / Graph Node Builder — CUDA 13.x cuGraphAddNode FFI Wrappers
// ═══════════════════════════════════════════════════════════════════════
//
// Begun 2026-05-03 — operator directive "migration to cuGraphAddNode for a
// monolithic splice ... so we don't continue to have childgraphIF failures
// bc we are using the wrong name conventions".
//
// Background
// ----------
// CUDA 13.x consolidates the per-type graph-builder API surface
// (`cuGraphAddKernelNode`, `cuGraphAddMemcpyNode`, `cuGraphAddMemsetNode`,
// `cuGraphAddChildGraphNode`, …) into a single unified call:
//
//   cudaError_t cudaGraphAddNode(
//       cudaGraphNode_t       *pGraphNode,
//       cudaGraph_t            graph,
//       const cudaGraphNode_t *pDependencies,
//       const cudaGraphEdgeData *dependencyData,   // NEW in 13.x
//       size_t                 numDependencies,
//       cudaGraphNodeParams   *nodeParams);
//
// where `nodeParams.type` selects which arm of the union is read.
//
// The Rust side of this codebase still uses the legacy per-type adders
// in two call sites:
//   1. graph_capture.rs:161 — `cuGraphAddChildGraphNode` for the
//      monolithic child-template splice (the actual "monolithic
//      splice" the operator referenced).
//   2. captured_pipeline.rs:2908 — `cuGraphAddMemsetNode` for the
//      conditional-body burst-marker MEMSET.
//
// The mismatch causes ChildGraph IF (CONDITIONAL) failures when the
// child template was built with the CUDA 13.x API but spliced with
// the legacy CUDA 11.x adder — naming conventions for the conditional
// handle metadata diverge between the two API generations.
//
// This header exposes thin host-side FFI wrappers that internally
// invoke the unified `cudaGraphAddNode` so all Rust call sites
// converge on one consistent CUDA 13.x naming convention.

#ifndef PRISM_GRAPH_NODE_CUH
#define PRISM_GRAPH_NODE_CUH

#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// ─────────────────────────────────────────────────────────────────────
// prism_graph_add_child_node_v3_ffi — monolithic child-graph splice
// ─────────────────────────────────────────────────────────────────────
//
// Replaces `cuGraphAddChildGraphNode(pGraphNode, parent, deps, n_deps,
// child)` with the CUDA 13.x unified `cudaGraphAddNode` call where
// `nodeParams.type = cudaGraphNodeTypeGraph` and
// `nodeParams.graph.graph = child_template`.
//
// This is the "monolithic splice" path — one call adds the entire
// child template as a single node, with all its internal nodes/edges
// preserved (CUDA copies the template, not by reference; future
// mutations of `child_template` do NOT propagate to the spliced
// instance).
//
// Args:
//   parent_graph     — destination CUgraph the splice node lands in
//   pDependencies    — array of CUgraphNode* dependencies (may be null)
//   numDependencies  — count (0 if pDependencies null)
//   child_template   — CUgraph being spliced as a single node
//   pOutNode         — destination for the new splice node handle
//
// Returns: cudaError_t cast to int. 0 on success.
int prism_graph_add_child_node_v3_ffi(
    cudaGraph_t            parent_graph,
    const cudaGraphNode_t *pDependencies,
    size_t                 numDependencies,
    cudaGraph_t            child_template,
    cudaGraphNode_t       *pOutNode);

// ─────────────────────────────────────────────────────────────────────
// prism_graph_add_memset_node_v3_ffi — unified MEMSET node adder
// ─────────────────────────────────────────────────────────────────────
//
// Replaces `cuGraphAddMemsetNode(pGraphNode, parent, deps, n_deps,
// memset_params, ctx)` with the CUDA 13.x unified call. The legacy
// adder takes a trailing CUcontext arg; the v3 form derives context
// from the graph itself.
//
// Currently emits a `width × height` MEMSET of `value` (truncated to
// `elementSize` bytes per element, max 4) into `dst`. The
// `pitch` arg matches the legacy semantics.
//
// Args:
//   parent_graph     — destination CUgraph the MEMSET node lands in
//   pDependencies    — array of CUgraphNode* dependencies (may be null)
//   numDependencies  — count (0 if pDependencies null)
//   dst              — destination CUdeviceptr
//   pitch            — bytes per row (use elementSize*width for tight)
//   value_u32        — fill value (only the low elementSize bytes used)
//   element_size     — 1, 2, or 4 (CUDA constraint)
//   width            — elements per row
//   height           — number of rows (1 for 1-D MEMSET)
//   pOutNode         — destination for the new MEMSET node handle
//
// Returns: cudaError_t cast to int. 0 on success.
int prism_graph_add_memset_node_v3_ffi(
    cudaGraph_t            parent_graph,
    const cudaGraphNode_t *pDependencies,
    size_t                 numDependencies,
    void                  *dst,
    size_t                 pitch,
    uint32_t               value_u32,
    unsigned int           element_size,
    size_t                 width,
    size_t                 height,
    cudaGraphNode_t       *pOutNode);

// ─────────────────────────────────────────────────────────────────────
// prism_graph_is_splice_legal_ffi — preflight legality check
// ─────────────────────────────────────────────────────────────────────
//
// TIER 8 — operator directive 2026-05-03: "Add a hard preflight
// legality check before any splice attempt.  Fail with a PRISM-
// specific error before CUDA returns 801."
//
// Walks every node in `child_graph` via cudaGraphGetNodes +
// cudaGraphNodeGetType, counts conditional, allocation, and free nodes.
// PRISM GRAPH-SPLICE-001 forbids all three node classes in spliceable
// child templates.  This helper enumerates violations BEFORE the splice
// attempt so the orchestrator can emit a typed PRISM error instead of
// catching a raw cudaErrorNotSupported (801).
//
// Args:
//   child_graph             — CUgraph to inspect (typically the V2
//                             captured-pipeline raw graph that an
//                             orchestrator is about to splice into a
//                             parent template)
//   pOutTotalNodes          — receives the total node count; null OK
//   pOutConditionalNodes    — receives the conditional node count; null OK
//   pOutAllocNodes          — receives the mem-alloc node count; null OK
//   pOutFreeNodes           — receives the mem-free node count; null OK
//
// Returns: cudaError_t cast to int.
//   0 — inspection succeeded.  Caller examines *pOutConditionalNodes:
//       0 ⇒ legal to splice; >0 ⇒ illegal.
//   nonzero — inspection itself failed (e.g., null graph, OOM during
//             enumeration); caller must abort.
int prism_graph_is_splice_legal_ffi(
    cudaGraph_t            child_graph,
    size_t                *pOutTotalNodes,
    size_t                *pOutConditionalNodes,
    size_t                *pOutAllocNodes,
    size_t                *pOutFreeNodes);

// ─────────────────────────────────────────────────────────────────────
// prism_graph_create_while_handle_ffi — parent-owned WHILE handle
// ─────────────────────────────────────────────────────────────────────
//
// TIER 8 — operator directive (WHILE-FFI-SCAFFOLD-001, 2026-05-04):
// "WHILE bounded micro-loop scaffold, parent-owned, size = 1, default
// value semantically driven by predicate-bridge kernel."
//
// Mirrors `prism_gearbox_create_handle_ffi` (adjudicator.cu:1119) which
// drives the SWITCH path; specialised for the WHILE invariant
// `cudaGraphCondTypeWhile = 1` (driver_types.h:3560) and `size = 1`
// (driver_types.h:3572 — "Allowed values are 1 for cudaGraphCondTypeWhile").
//
// `default_value` controls whether the body runs at least once at
// instantiation:
//   `1` — enter body once by default; predicate-bridge kernel
//         falsifies via cudaGraphSetConditional to exit.
//   `0` — body NEVER runs unless predicate-bridge sets it true; this
//         is the safe default for "no-op WHILE" (loop terminates
//         immediately if predicate never set).
//
// CUDA 12.4 floor (CUDART_VERSION >= 12040). Below that, the helper
// returns `cudaErrorNotSupported` per existing scaffold convention.
//
// Args:
//   parent_graph     — parent CUgraph that will host the WHILE node;
//                      handle is bound to this graph at creation
//   default_value    — initial conditional value at launch (see above)
//   pOutHandle       — receives the conditional handle as a u64 (the
//                      CUDA `cudaGraphConditionalHandle` typedef is
//                      `unsigned long long`)
//
// Returns: cudaError_t cast to int. 0 on success.
int prism_graph_create_while_handle_ffi(
    cudaGraph_t   parent_graph,
    uint32_t      default_value,
    uint64_t     *pOutHandle);

// ─────────────────────────────────────────────────────────────────────
// prism_graph_add_while_node_ffi — parent-owned WHILE conditional node
// ─────────────────────────────────────────────────────────────────────
//
// TIER 8 — adds a single CONDITIONAL node of type
// `cudaGraphCondTypeWhile` with `size = 1` to `parent_graph`, downstream
// of `pDependencies`. After successful add, the body subgraph (the
// single WHILE body) is returned via `pOutBodySubgraph` so the caller
// (R6 lane / `captured_pipeline.rs`) can populate the body with the
// predicate-bridge kernel + drain pump.
//
// Per CUDA 13.x semantics (driver_types.h:3574-3592): the
// `cudaConditionalNodeParams.phGraph_out` array is CUDA-OWNED and
// populated DURING the `cudaGraphAddNode` call. Its lifetime equals
// the lifetime of the conditional node itself. We copy
// `phGraph_out[0]` into the caller-provided slot so callers do not
// need to keep the `nodeParams` struct alive.
//
// Mirrors `prism_gearbox_wire_with_handle_ffi` (adjudicator.cu:1137);
// specialised to `size = 1` (the SWITCH variant uses `size = 4`).
//
// Args:
//   parent_graph        — destination parent CUgraph; MUST be the same
//                         graph the handle was created against
//   pDependencies       — array of CUgraphNode* dependencies (may be
//                         null when num_deps = 0)
//   numDependencies     — count (0 if pDependencies null)
//   handle_v            — handle returned by
//                         prism_graph_create_while_handle_ffi
//   pOutConditionalNode — receives the new WHILE conditional node
//   pOutBodySubgraph    — receives the (single) body subgraph
//                         (`phGraph_out[0]`) for caller population
//
// Returns: cudaError_t cast to int. 0 on success.
int prism_graph_add_while_node_ffi(
    cudaGraph_t            parent_graph,
    const cudaGraphNode_t *pDependencies,
    size_t                 numDependencies,
    uint64_t               handle_v,
    cudaGraphNode_t       *pOutConditionalNode,
    cudaGraph_t           *pOutBodySubgraph);

#ifdef __cplusplus
}
#endif

#endif // PRISM_GRAPH_NODE_CUH
