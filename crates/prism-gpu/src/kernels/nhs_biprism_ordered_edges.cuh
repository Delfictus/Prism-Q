#pragma once
#include <stddef.h>
#include <stdint.h>

// Per-invocation traversal scratch, not physical state. Each row contains the
// original edge ordinals in increasing order, retaining both endpoint direction
// and the exact sequence of floating-point terms of the linear edge scan.
constexpr uint32_t BP_ORDERED_EDGES_MAGIC = 0x31454942u; // BIE1
constexpr uint32_t BP_ORDERED_EDGES_VERSION = 1u;
constexpr uint32_t BP_ORDERED_EDGES_CORE = 192u;
constexpr uint32_t BP_ORDERED_EDGES_MAX = 1536u;
constexpr uint32_t BP_ORDERED_EDGES_CAPACITY = 2u * BP_ORDERED_EDGES_MAX;
constexpr uint32_t BP_ORDERED_EDGES_INVALID = 0u;
constexpr uint32_t BP_ORDERED_EDGES_BUILDING = 1u;
constexpr uint32_t BP_ORDERED_EDGES_READY = 2u;

struct alignas(128) BiprismOrderedEdgesFfi {
    uint32_t magic;
    uint32_t version;
    uint32_t leaf;
    uint32_t num_edges;
    uint64_t tile_address;
    uint32_t accepted_substeps_at_entry;
    uint32_t publication_state;
    uint32_t incident_count;
    uint32_t reserved[7];
    uint16_t offsets[BP_ORDERED_EDGES_CORE + 1u];
    uint16_t edge_index[BP_ORDERED_EDGES_CAPACITY];
    uint8_t padding[62];
};
static_assert(sizeof(BiprismOrderedEdgesFfi) == 6656, "ordered edge scratch extent");
static_assert(alignof(BiprismOrderedEdgesFfi) == 128, "ordered edge scratch alignment");
static_assert(offsetof(BiprismOrderedEdgesFfi, offsets) == 64, "ordered row offsets");
static_assert(offsetof(BiprismOrderedEdgesFfi, edge_index) == 450, "ordered edge ordinals");

#if defined(__CUDACC__)
#define BP_ORDERED_HD __host__ __device__ __forceinline__
#else
#define BP_ORDERED_HD inline
#endif

struct BiprismLinearEdgeTraversal {
    uint32_t num_edges;
    BP_ORDERED_HD uint32_t begin(uint32_t) const { return 0u; }
    BP_ORDERED_HD uint32_t end(uint32_t) const { return num_edges; }
    BP_ORDERED_HD uint32_t edge(uint32_t cursor) const { return cursor; }
};
struct BiprismIncidentEdgeTraversal {
    const BiprismOrderedEdgesFfi* index;
    BP_ORDERED_HD uint32_t begin(uint32_t vertex) const { return index->offsets[vertex]; }
    BP_ORDERED_HD uint32_t end(uint32_t vertex) const { return index->offsets[vertex + 1u]; }
    BP_ORDERED_HD uint32_t edge(uint32_t cursor) const { return index->edge_index[cursor]; }
};

// These three integer-only phases are also used by the exact host controls.
// The CUDA entry below provides their actual cooperative publication order.
BP_ORDERED_HD uint32_t biprism_ordered_edge_count_row(
    const uint8_t* edge_u, const uint8_t* edge_v, uint32_t stride,
    uint32_t num_edges, uint32_t vertex
) {
    uint32_t count = 0u;
    for (uint32_t e = 0u; e < num_edges; ++e)
        count += edge_u[e * stride] == vertex || edge_v[e * stride] == vertex;
    return count;
}

BP_ORDERED_HD bool biprism_ordered_edge_prefix(BiprismOrderedEdgesFfi& index) {
    uint32_t sum = 0u;
    index.offsets[0] = 0u;
    for (uint32_t vertex = 0u; vertex < BP_ORDERED_EDGES_CORE; ++vertex) {
        const uint32_t count = index.offsets[vertex + 1u];
        if (count > index.num_edges || sum + count > BP_ORDERED_EDGES_CAPACITY)
            return false;
        sum += count;
        index.offsets[vertex + 1u] = static_cast<uint16_t>(sum);
    }
    index.incident_count = sum;
    return true;
}

BP_ORDERED_HD bool biprism_ordered_edge_fill_row(
    BiprismOrderedEdgesFfi& index, const uint8_t* edge_u,
    const uint8_t* edge_v, uint32_t stride, uint32_t vertex
) {
    if (vertex >= BP_ORDERED_EDGES_CORE) return false;
    uint32_t cursor = index.offsets[vertex];
    const uint32_t end = index.offsets[vertex + 1u];
    if (cursor > end || end > BP_ORDERED_EDGES_CAPACITY) return false;
    for (uint32_t e = 0u; e < index.num_edges; ++e) {
        if (edge_u[e * stride] == vertex || edge_v[e * stride] == vertex) {
            if (cursor == end) return false;
            index.edge_index[cursor++] = static_cast<uint16_t>(e);
        }
    }
    return cursor == end;
}

BP_ORDERED_HD bool biprism_ordered_edge_row_matches(
    const BiprismOrderedEdgesFfi& index, const uint8_t* edge_u,
    const uint8_t* edge_v, uint32_t stride, uint32_t vertex
) {
    if (vertex >= BP_ORDERED_EDGES_CORE || index.num_edges > BP_ORDERED_EDGES_MAX)
        return false;
    uint32_t cursor = index.offsets[vertex];
    const uint32_t end = index.offsets[vertex + 1u];
    if (cursor > end || end > BP_ORDERED_EDGES_CAPACITY) return false;
    for (uint32_t e = 0u; e < index.num_edges; ++e) {
        if (edge_u[e * stride] == vertex || edge_v[e * stride] == vertex) {
            if (cursor == end || index.edge_index[cursor++] != e) return false;
        }
    }
    return cursor == end;
}

BP_ORDERED_HD bool biprism_ordered_edges_identity_matches(
    const BiprismOrderedEdgesFfi& index, uint32_t leaf, uint64_t tile_address,
    uint32_t num_edges, uint32_t accepted_substeps_at_entry
) {
    if (index.magic != BP_ORDERED_EDGES_MAGIC || index.version != BP_ORDERED_EDGES_VERSION
        || index.publication_state != BP_ORDERED_EDGES_READY || index.leaf != leaf
        || index.tile_address != tile_address || index.num_edges != num_edges
        || index.accepted_substeps_at_entry != accepted_substeps_at_entry
        || num_edges == 0u || num_edges > BP_ORDERED_EDGES_MAX
        || index.offsets[0] != 0u || index.incident_count > BP_ORDERED_EDGES_CAPACITY
        || index.offsets[BP_ORDERED_EDGES_CORE] != index.incident_count) return false;
    for (uint32_t i = 0u; i < 7u; ++i) if (index.reserved[i] != 0u) return false;
    return true;
}

#if defined(__CUDACC__)
// Exactly one 256-thread CTA owns one index. Invoke unconditionally once after
// the current invocation's topology guard and before any indexed consumer.
// No prior publication can qualify: state is invalidated before validation,
// counts and fill are rebuilt from the same stable current endpoint arrays,
// and READY is published only after every row writer has completed.
__device__ __forceinline__ bool biprism_build_ordered_edges_256(
    BiprismOrderedEdgesFfi* index, const uint8_t* edge_u,
    const uint8_t* edge_v, uint32_t stride, uint32_t num_edges,
    uint32_t leaf, uint64_t tile_address, uint32_t accepted_substeps_at_entry
) {
    const uint32_t tid = threadIdx.x;
    const bool pointer_ok = index && (reinterpret_cast<uintptr_t>(index) & 127u) == 0u;
    if (!__syncthreads_and(pointer_ok)) return false;
    if (tid == 0u) index->publication_state = BP_ORDERED_EDGES_INVALID;
    __syncthreads();
    bool valid = blockDim.x == 256u && blockDim.y == 1u && blockDim.z == 1u
        && edge_u && edge_v && stride > 0u && num_edges > 0u
        && num_edges <= BP_ORDERED_EDGES_MAX && leaf < 8u && tile_address != 0u;
    if (!__syncthreads_and(valid)) return false;
    for (uint32_t e = tid; e < num_edges; e += 256u)
        valid = valid && edge_u[e * stride] != edge_v[e * stride];
    if (!__syncthreads_and(valid)) return false;
    if (tid == 0u) {
        index->magic = BP_ORDERED_EDGES_MAGIC;
        index->version = BP_ORDERED_EDGES_VERSION;
        index->leaf = leaf;
        index->num_edges = num_edges;
        index->tile_address = tile_address;
        index->accepted_substeps_at_entry = accepted_substeps_at_entry;
        index->incident_count = 0u;
        index->publication_state = BP_ORDERED_EDGES_BUILDING;
        for (uint32_t i = 0u; i < 7u; ++i) index->reserved[i] = 0u;
    }
    if (tid < BP_ORDERED_EDGES_CORE)
        index->offsets[tid + 1u] = static_cast<uint16_t>(
            biprism_ordered_edge_count_row(edge_u, edge_v, stride, num_edges, tid));
    __syncthreads();
    if (tid == 0u && !biprism_ordered_edge_prefix(*index))
        index->publication_state = BP_ORDERED_EDGES_INVALID;
    __syncthreads();
    if (index->publication_state != BP_ORDERED_EDGES_BUILDING) return false;
    valid = tid >= BP_ORDERED_EDGES_CORE
        || biprism_ordered_edge_fill_row(*index, edge_u, edge_v, stride, tid);
    if (!__syncthreads_and(valid)) {
        if (tid == 0u) index->publication_state = BP_ORDERED_EDGES_INVALID;
        __syncthreads();
        return false;
    }
    if (tid == 0u) index->publication_state = BP_ORDERED_EDGES_READY;
    __syncthreads();
    return true;
}
#endif

#undef BP_ORDERED_HD
