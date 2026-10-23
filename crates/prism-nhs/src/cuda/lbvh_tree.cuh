// ═══════════════════════════════════════════════════════════════════════
// PRISM-4D / LBVH Phase 2 — Karras Radix Tree + Bottom-Up AABB Reduce
// ═══════════════════════════════════════════════════════════════════════
//
// Per the Progressive Automation mandate §1 (operator directive
// 2026-04-29). Linear Bounding Volume Hierarchy via Karras 2012,
// upgraded for Blackwell sm_120 with:
//
//   - 64-bit COMPOSITE sort keys [30-bit Morton | 32-bit residue_id |
//     2 high bits unused]. Identity tie-breaker per Mandate §M2:
//     two spikes at the same voxel hash to distinct keys via their
//     residue_id discriminator. Eliminates the zero-length-split
//     infinite-loop in the radix tree builder under high-density
//     Morton collisions.
//
//   - 64-byte cache-line-aligned LBVHNode for line-aligned LDG.128
//     atomic-flag access during the bottom-up reduce.
//
//   - Last-arrival atomic-flag bottom-up AABB reduce (§1.2). Each
//     internal node has a u32 atomic counter; first child to arrive
//     increments-and-exits; second child sees the prior `1` value
//     returned by `atomicAdd`, fences for child-AABB visibility,
//     merges children, and continues up.
//
// **Layout contract (FFI-stable)**:
//
//   sizeof(LBVHNode)  == 64 bytes  (static_assert below)
//   alignof(LBVHNode) == 64 bytes  (static_assert below)
//
// **Child encoding** (Karras convention):
//
//   In a tree of `N` leaves, there are `N-1` internal nodes
//   (indices 0..N-2) and `N` leaves (indices N-1..2N-2).
//   `LBVHNode::left_child` / `right_child` carry the unified
//   index: < N-1 ⇒ internal node; >= N-1 ⇒ leaf at (child - (N-1)).
//
// **Root convention**: internal node 0 is always the root. Its
// `parent_idx == -1`.
//
// **AABB sentinel** (pre-reduce): aabb_min = (+FLT_MAX)³,
// aabb_max = (-FLT_MAX)³. The first child's contribution
// monotonically pulls the bounds inward.
//
// Compilation: nvcc -arch=sm_120 -O3 --use_fast_math --restrict
//              --expt-relaxed-constexpr -std=c++17 -c
//
// ═══════════════════════════════════════════════════════════════════════

#ifndef PRISM_NHS_LBVH_TREE_CUH
#define PRISM_NHS_LBVH_TREE_CUH

#include <cstdint>
#include <cuda_runtime.h>

namespace prism_nhs { namespace lbvh_tree {

// ─────────────────────────────────────────────────────────────────────
// LBVHNode — 64-byte cache-line-aligned tree node.
//
// Layout (offsets):
//   0   parent_idx     (i32)
//   4   left_child     (i32)  — unified index per Karras encoding
//   8   right_child    (i32)
//   12  atomic_flag    (u32)  — last-arrival counter (0 → 1 → 2)
//   16  aabb_min       (float4) — 4th comp is unused padding
//   32  aabb_max       (float4)
//   48  metadata       (u64)  — reserved for Morton/causal coupling
//   56  _pad           (u8[8])
//   64  (end)
// ─────────────────────────────────────────────────────────────────────
struct alignas(64) LBVHNode {
    int32_t  parent_idx;
    int32_t  left_child;
    int32_t  right_child;
    uint32_t atomic_flag;
    float4   aabb_min;
    float4   aabb_max;
    uint64_t metadata;
    uint8_t  _pad[8];
};
static_assert(sizeof(LBVHNode) == 64, "LBVHNode size drift: not 64 B");
static_assert(alignof(LBVHNode) == 64, "LBVHNode alignment drift: not 64 B");

// ─────────────────────────────────────────────────────────────────────
// Sentinel constants.
// ─────────────────────────────────────────────────────────────────────
constexpr int32_t  LBVH_PARENT_NONE   = -1;
constexpr int32_t  LBVH_CHILD_MISSING = -1;  // distinct from "leaf id 0"

// ─────────────────────────────────────────────────────────────────────
// Composite sort-key helpers (CPU + GPU bit-equivalent via
// __host__ __device__).
//
// Layout (MSB → LSB):
//   bits 63..62 : reserved (currently 0)
//   bits 61..32 : 30-bit Morton code
//   bits 31..0  : 32-bit residue_id (signed-as-unsigned)
//
// Two spikes with identical Morton AND identical residue_id share a
// key — by construction this is a structural duplicate (same voxel +
// same residue) and the radix tree handles it via stable-sort
// semantics. Two spikes with same Morton + DIFFERENT residue_id
// hash to distinct keys, eliminating the zero-length-split case.
// ─────────────────────────────────────────────────────────────────────
#ifdef __CUDACC__
  #define PRISM_LBVH_HD __host__ __device__ __forceinline__
#else
  #define PRISM_LBVH_HD inline
#endif

PRISM_LBVH_HD uint64_t prism_lbvh_compose_sort_key(uint32_t morton30, int32_t residue_id) {
    const uint64_t m = static_cast<uint64_t>(morton30 & 0x3FFFFFFFu);  // mask high 2 bits
    const uint64_t r = static_cast<uint64_t>(static_cast<uint32_t>(residue_id));
    return (m << 32) | r;
}

PRISM_LBVH_HD uint32_t prism_lbvh_extract_morton(uint64_t key) {
    return static_cast<uint32_t>((key >> 32) & 0x3FFFFFFFull);
}

PRISM_LBVH_HD uint32_t prism_lbvh_extract_residue_raw(uint64_t key) {
    return static_cast<uint32_t>(key & 0xFFFFFFFFull);
}

// ─────────────────────────────────────────────────────────────────────
// extern "C" host orchestration.
// ─────────────────────────────────────────────────────────────────────

extern "C" {

/// Sentinel `0xBABE` — pinned by Rust-side link-probe test.
uint32_t prism_lbvh_tree_link_probe(void);

/// Pre-condition step: initialize internal_nodes to a fresh state
/// before tree build. Sets parent_idx = -1, atomic_flag = 0, AABB
/// sentinels (+FLT_MAX min, -FLT_MAX max). Single-threaded launch
/// per node count.
cudaError_t prism_lbvh_init_internal_nodes(
    LBVHNode*    d_internal_nodes,
    uint32_t     n_internal,
    cudaStream_t stream
);

/// Build the Karras 2012 radix tree from a SORTED array of 64-bit
/// composite keys. Writes parent_idx + left_child + right_child for
/// every internal node and the parent index for every leaf into
/// `d_leaf_parents`. Single-pass kernel; uses `__clzll` on XOR'd
/// adjacent keys to find common-prefix length δ and binary-searches
/// for the split point.
///
/// Caller is responsible for setting `d_internal_nodes[0].parent_idx
/// = -1` (the root has no parent); the kernel writes parent_idx for
/// every NON-root internal node and every leaf.
cudaError_t prism_lbvh_karras_build(
    const uint64_t* d_sorted_keys,
    uint32_t        n_leaves,
    LBVHNode*       d_internal_nodes,    // size n_leaves - 1
    int32_t*        d_leaf_parents,      // size n_leaves
    cudaStream_t    stream
);

/// Bottom-up AABB reduce. Each leaf seeds its leaf-AABB from
/// `d_leaf_positions` (planar [N][3]) and walks up via the
/// last-arrival atomic-flag pattern. The second thread to arrive at
/// a parent merges the two child AABBs and propagates up.
///
/// **Pre-condition**: `d_internal_nodes[*].atomic_flag = 0` and
/// `aabb_min/max` set to sentinels (+FLT_MAX / -FLT_MAX) by
/// `prism_lbvh_init_internal_nodes`. The Karras builder must have
/// populated parent_idx + left_child + right_child for every
/// internal node and leaf_parents for every leaf.
///
/// **Post-condition**: every internal node's aabb_min / aabb_max
/// is the tight bbox over the leaves in its subtree. Root's AABB
/// is the bbox over every leaf.
cudaError_t prism_lbvh_aabb_reduce(
    LBVHNode*       d_internal_nodes,
    uint32_t        n_internal,
    const int32_t*  d_leaf_parents,
    const float*    d_leaf_positions,    // planar [n_leaves][3]
    uint32_t        n_leaves,
    cudaStream_t    stream
);

}  // extern "C"

}}  // namespace prism_nhs::lbvh_tree

#endif  // PRISM_NHS_LBVH_TREE_CUH
