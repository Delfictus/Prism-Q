// ═══════════════════════════════════════════════════════════════════════
// PRISM-4D / LBVH Phase 2 — Karras Radix Tree Builder + AABB Reduce
// ═══════════════════════════════════════════════════════════════════════
//
// See `lbvh_tree.cuh` for layout + caller contract.
//
// **Karras 2012 algorithm summary**:
//
//   For N sorted composite keys, build N-1 internal nodes:
//
//   For thread `i` in [0, N-2]:
//     1. Compute direction `d`: +1 if δ(i, i+1) > δ(i, i-1) else -1.
//     2. Compute `delta_min = δ(i, i - d)` (delta toward the
//        prefix-shorter side).
//     3. Find upper bound on length `l_max` by doubling
//        until `δ(i, i + l_max * d) <= delta_min`.
//     4. Binary search for actual length `l` (the maximum t
//        such that `δ(i, i + (l + t) * d) > delta_min`).
//     5. Compute `j = i + l * d` (the other end of the range).
//     6. Find split point γ via binary search on δ within
//        the range [i, j], looking for the maximum `s` such
//        that `δ(i, i + s * d) > δ(i, j)`.
//     7. Translate γ to a child index, with leaf-encoding via
//        `(child >= n_internal) ? leaf : internal`.
//
// **AABB reduce algorithm**:
//
//   For each leaf in parallel:
//     - Initialize the LEAF's contribution from `leaf_positions[leaf]`.
//     - Walk up via `leaf_parents[leaf] → internal[node].parent_idx`.
//     - At each internal node: atomicAdd(&atomic_flag, 1). If old
//       value is 0 (first arrival), exit. If old value is 1 (second
//       arrival), __threadfence(), read the two children's AABBs,
//       merge, write back, continue up.
//
// Compilation: nvcc -arch=sm_120 -O3 --use_fast_math --restrict
//              --expt-relaxed-constexpr -std=c++17 -Xcompiler -fPIC -c
// ═══════════════════════════════════════════════════════════════════════

#include "lbvh_tree.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>

namespace prism_nhs { namespace lbvh_tree {

// ─────────────────────────────────────────────────────────────────────
// δ — common-prefix length on 64-bit composite keys via __clzll on XOR.
//
// Returns -1 for out-of-bounds j (acts as a "less than any in-bound
// δ" sentinel; the binary searches treat it as a hard lower bound).
// ─────────────────────────────────────────────────────────────────────
__device__ __forceinline__ int prism_delta(
    const uint64_t* __restrict__ keys,
    int n_keys,
    int i,
    int j
) {
    if (j < 0 || j >= n_keys) return -1;
    return __clzll(keys[i] ^ keys[j]);
}

__device__ __forceinline__ int prism_sign(int x) {
    return (x > 0) - (x < 0);
}

// ─────────────────────────────────────────────────────────────────────
// __global__ kernels
// ─────────────────────────────────────────────────────────────────────

/// One thread per internal node. Initializes parent_idx = -1,
/// atomic_flag = 0, aabb_min = +FLT_MAX, aabb_max = -FLT_MAX.
__global__ void prism_lbvh_init_internal_nodes_kernel(
    LBVHNode* __restrict__ d_internal_nodes,
    uint32_t               n_internal
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_internal) return;

    LBVHNode& node = d_internal_nodes[tid];
    node.parent_idx = LBVH_PARENT_NONE;
    node.left_child = LBVH_CHILD_MISSING;
    node.right_child = LBVH_CHILD_MISSING;
    node.atomic_flag = 0u;
    node.aabb_min = make_float4( FLT_MAX,  FLT_MAX,  FLT_MAX, 0.0f);
    node.aabb_max = make_float4(-FLT_MAX, -FLT_MAX, -FLT_MAX, 0.0f);
    node.metadata = 0ull;
}

/// One thread per internal node. Karras 2012 single-pass tree
/// construction. Writes parent_idx for non-root internals and for
/// every leaf (via d_leaf_parents).
__global__ void prism_lbvh_karras_build_kernel(
    const uint64_t* __restrict__ d_sorted_keys,
    uint32_t                     n_leaves,
    LBVHNode* __restrict__       d_internal_nodes,
    int32_t* __restrict__        d_leaf_parents
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_internal = static_cast<int>(n_leaves) - 1;
    if (i >= n_internal) return;
    const int n_keys = static_cast<int>(n_leaves);

    // Step 1: direction.
    const int delta_left  = prism_delta(d_sorted_keys, n_keys, i, i - 1);
    const int delta_right = prism_delta(d_sorted_keys, n_keys, i, i + 1);
    const int d = prism_sign(delta_right - delta_left);

    // Step 2: delta_min (toward prefix-shorter side).
    const int delta_min = (d > 0) ? delta_left : delta_right;

    // Step 3: upper bound on length via doubling.
    int l_max = 2;
    while (prism_delta(d_sorted_keys, n_keys, i, i + l_max * d) > delta_min) {
        l_max *= 2;
    }

    // Step 4: binary search for length.
    int l = 0;
    for (int t = l_max / 2; t > 0; t /= 2) {
        if (prism_delta(d_sorted_keys, n_keys, i, i + (l + t) * d) > delta_min) {
            l += t;
        }
    }

    // Step 5: other end.
    const int j = i + l * d;

    // Step 6: binary search for split.
    const int delta_node = prism_delta(d_sorted_keys, n_keys, i, j);
    int s = 0;
    int t = l;
    do {
        t = (t + 1) / 2;
        if (prism_delta(d_sorted_keys, n_keys, i, i + (s + t) * d) > delta_node) {
            s += t;
        }
    } while (t > 1);

    // Step 7: split point γ in unified-index space.
    //   γ = i + s*d + min(d, 0)
    // For d=+1: γ = i + s.    For d=-1: γ = i - s - 1.
    const int gamma = i + s * d + ((d < 0) ? -1 : 0);

    // Determine left and right children with leaf encoding.
    // In the Karras scheme: leaf k is unified-index (n_internal + k).
    const int min_ij = (i < j) ? i : j;
    const int max_ij = (i > j) ? i : j;

    const int left_unified = (min_ij == gamma)
        ? (n_internal + gamma)        // leaf at gamma
        : gamma;                       // internal at gamma

    const int right_unified = (max_ij == (gamma + 1))
        ? (n_internal + gamma + 1)    // leaf at gamma+1
        : (gamma + 1);                 // internal at gamma+1

    LBVHNode& self = d_internal_nodes[i];
    self.left_child = left_unified;
    self.right_child = right_unified;

    // Set children's parent pointer.
    if (left_unified >= n_internal) {
        // Left is a leaf.
        d_leaf_parents[left_unified - n_internal] = i;
    } else {
        d_internal_nodes[left_unified].parent_idx = i;
    }
    if (right_unified >= n_internal) {
        d_leaf_parents[right_unified - n_internal] = i;
    } else {
        d_internal_nodes[right_unified].parent_idx = i;
    }
}

/// Bottom-up AABB reduce via last-arrival atomic flag.
///
/// One thread per leaf. Each thread:
///   1. Reads its leaf position (3 floats from d_leaf_positions).
///   2. Looks up its parent via d_leaf_parents[leaf].
///   3. atomicAdd(&parent.atomic_flag, 1). If old == 0 (first
///      arrival): exit (the leaf's contribution is read directly
///      by the second-arrival thread via the leaf-positions array).
///   4. If old == 1 (second arrival): __threadfence(), then merge
///      both children's AABBs into the parent. Walk up to the
///      grandparent and repeat.
///
/// Children that are LEAVES contribute directly via leaf_positions
/// (looked up by leaf id = unified - n_internal). Children that are
/// INTERNAL nodes contribute via their already-merged aabb_min/max.
__global__ void prism_lbvh_aabb_reduce_kernel(
    LBVHNode* __restrict__       d_internal_nodes,
    uint32_t                     n_internal,
    const int32_t* __restrict__  d_leaf_parents,
    const float* __restrict__    d_leaf_positions,
    uint32_t                     n_leaves
) {
    const uint32_t leaf_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf_id >= n_leaves) return;

    int current = d_leaf_parents[leaf_id];

    while (current != LBVH_PARENT_NONE) {
        // Atomic increment; old == 0 → first arrival, exit.
        const uint32_t old = atomicAdd(&d_internal_nodes[current].atomic_flag, 1u);
        if (old == 0u) {
            return;  // first child has arrived; second child will do the merge
        }

        // We are the second-arrival thread. Ensure the first child's
        // contribution is visible.
        __threadfence();

        // Merge children's AABBs into self.
        LBVHNode& self = d_internal_nodes[current];
        const int left = self.left_child;
        const int right = self.right_child;

        float4 mn_l, mx_l, mn_r, mx_r;
        if (left >= static_cast<int>(n_internal)) {
            // Left is a leaf.
            const int leaf_idx = left - static_cast<int>(n_internal);
            const float lx = d_leaf_positions[3 * leaf_idx + 0];
            const float ly = d_leaf_positions[3 * leaf_idx + 1];
            const float lz = d_leaf_positions[3 * leaf_idx + 2];
            mn_l = make_float4(lx, ly, lz, 0.0f);
            mx_l = make_float4(lx, ly, lz, 0.0f);
        } else {
            mn_l = d_internal_nodes[left].aabb_min;
            mx_l = d_internal_nodes[left].aabb_max;
        }
        if (right >= static_cast<int>(n_internal)) {
            const int leaf_idx = right - static_cast<int>(n_internal);
            const float rx = d_leaf_positions[3 * leaf_idx + 0];
            const float ry = d_leaf_positions[3 * leaf_idx + 1];
            const float rz = d_leaf_positions[3 * leaf_idx + 2];
            mn_r = make_float4(rx, ry, rz, 0.0f);
            mx_r = make_float4(rx, ry, rz, 0.0f);
        } else {
            mn_r = d_internal_nodes[right].aabb_min;
            mx_r = d_internal_nodes[right].aabb_max;
        }

        self.aabb_min = make_float4(
            fminf(mn_l.x, mn_r.x),
            fminf(mn_l.y, mn_r.y),
            fminf(mn_l.z, mn_r.z),
            0.0f
        );
        self.aabb_max = make_float4(
            fmaxf(mx_l.x, mx_r.x),
            fmaxf(mx_l.y, mx_r.y),
            fmaxf(mx_l.z, mx_r.z),
            0.0f
        );

        // Memory barrier so the parent (above us) sees our writes.
        __threadfence();

        // Walk up.
        current = self.parent_idx;
    }
}

// ─────────────────────────────────────────────────────────────────────
// extern "C" host orchestration
// ─────────────────────────────────────────────────────────────────────

extern "C" {

uint32_t prism_lbvh_tree_link_probe(void) {
    return 0xBABEu;
}

cudaError_t prism_lbvh_init_internal_nodes(
    LBVHNode*    d_internal_nodes,
    uint32_t     n_internal,
    cudaStream_t stream
) {
    if (n_internal == 0u) return cudaSuccess;
    if (d_internal_nodes == nullptr) return cudaErrorInvalidValue;

    constexpr uint32_t TPB = 256u;
    const uint32_t blocks = (n_internal + TPB - 1u) / TPB;
    prism_lbvh_init_internal_nodes_kernel<<<blocks, TPB, 0, stream>>>(
        d_internal_nodes, n_internal
    );
    return cudaGetLastError();
}

cudaError_t prism_lbvh_karras_build(
    const uint64_t* d_sorted_keys,
    uint32_t        n_leaves,
    LBVHNode*       d_internal_nodes,
    int32_t*        d_leaf_parents,
    cudaStream_t    stream
) {
    if (n_leaves < 2u) {
        // Karras needs at least 2 leaves to have any internal nodes.
        return cudaSuccess;
    }
    if (d_sorted_keys == nullptr || d_internal_nodes == nullptr || d_leaf_parents == nullptr) {
        return cudaErrorInvalidValue;
    }

    const uint32_t n_internal = n_leaves - 1u;
    constexpr uint32_t TPB = 256u;
    const uint32_t blocks = (n_internal + TPB - 1u) / TPB;
    prism_lbvh_karras_build_kernel<<<blocks, TPB, 0, stream>>>(
        d_sorted_keys, n_leaves, d_internal_nodes, d_leaf_parents
    );
    return cudaGetLastError();
}

cudaError_t prism_lbvh_aabb_reduce(
    LBVHNode*      d_internal_nodes,
    uint32_t       n_internal,
    const int32_t* d_leaf_parents,
    const float*   d_leaf_positions,
    uint32_t       n_leaves,
    cudaStream_t   stream
) {
    if (n_leaves < 2u || n_internal == 0u) {
        return cudaSuccess;
    }
    if (d_internal_nodes == nullptr || d_leaf_parents == nullptr || d_leaf_positions == nullptr) {
        return cudaErrorInvalidValue;
    }

    constexpr uint32_t TPB = 256u;
    const uint32_t blocks = (n_leaves + TPB - 1u) / TPB;
    prism_lbvh_aabb_reduce_kernel<<<blocks, TPB, 0, stream>>>(
        d_internal_nodes, n_internal,
        d_leaf_parents, d_leaf_positions, n_leaves
    );
    return cudaGetLastError();
}

}  // extern "C"

}}  // namespace prism_nhs::lbvh_tree
