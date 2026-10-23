/**
 * SDST Internal Types and GPU Helpers
 * Not exposed via public API.
 */

#ifndef SDST_INTERNAL_H
#define SDST_INTERNAL_H

#include "sdst_api.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>   /* fprintf, stderr, printf */
#include <stdlib.h>  /* malloc, free, calloc */

/* ============================================================
 * CUDA error checking
 * ============================================================ */

#define SDST_CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        return SDST_ERROR_CUDA; \
    } \
} while(0)

#define SDST_CUDA_CHECK_KERNEL() do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "Kernel launch error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        return SDST_ERROR_CUDA; \
    } \
} while(0)

/* ============================================================
 * Morton code encoding/decoding for 128³ grid
 * Uses 7 bits per axis, 21 bits total, fits in u32
 * ============================================================ */

/**
 * Spread bits for 3D Morton encoding: insert 2 zero bits between each bit.
 * Input: 7-bit value (0-127), representing one grid axis.
 * Output: 21-bit value with each input bit at position 3*i.
 *
 * NOTE: The original magic-number approach (0x070007 etc.) was WRONG for
 * values >= 8 — it dropped bits at positions 3-6. This loop-based
 * implementation is verified correct for all 7-bit inputs (GATE 0 / Issue A).
 * The compiler unrolls the 7-iteration loop to branchless code on SM120.
 */
__device__ __host__ __forceinline__
uint32_t morton_spread(uint32_t v) {
    v &= 0x7F; /* 7 bits */
    uint32_t result = 0;
#ifdef __CUDA_ARCH__
    #pragma unroll
#endif
    for (uint32_t i = 0; i < 7; i++) {
        result |= ((v >> i) & 1u) << (3u * i);
    }
    return result;
}

/**
 * Compact bits for 3D Morton decoding: extract every 3rd bit.
 * Inverse of morton_spread.
 */
__device__ __host__ __forceinline__
uint32_t morton_compact(uint32_t v) {
    v &= 0x249249; /* keep only bits at positions 0,3,6,9,12,15,18 */
    uint32_t result = 0;
#ifdef __CUDA_ARCH__
    #pragma unroll
#endif
    for (uint32_t i = 0; i < 7; i++) {
        result |= ((v >> (3u * i)) & 1u) << i;
    }
    return result;
}

/** Encode (x,y,z) -> Morton code */
__device__ __host__ __forceinline__
MortonCode morton_encode(uint32_t x, uint32_t y, uint32_t z) {
    return morton_spread(x) | (morton_spread(y) << 1) | (morton_spread(z) << 2);
}

/** Decode Morton code -> (x,y,z) */
__device__ __host__ __forceinline__
void morton_decode(MortonCode code, uint32_t* x, uint32_t* y, uint32_t* z) {
    *x = morton_compact(code);
    *y = morton_compact(code >> 1);
    *z = morton_compact(code >> 2);
}

/** Convert grid coordinates to Angstrom position */
__device__ __host__ __forceinline__
void grid_to_angstrom(uint32_t gx, uint32_t gy, uint32_t gz,
                      float spacing,
                      float* ax, float* ay, float* az) {
    *ax = (float)gx * spacing;
    *ay = (float)gy * spacing;
    *az = (float)gz * spacing;
}

/** Euclidean distance between two Morton-encoded voxels (in grid units) */
__device__ __forceinline__
float morton_distance(MortonCode a, MortonCode b) {
    uint32_t ax, ay, az, bx, by, bz;
    morton_decode(a, &ax, &ay, &az);
    morton_decode(b, &bx, &by, &bz);
    float dx = (float)ax - (float)bx;
    float dy = (float)ay - (float)by;
    float dz = (float)az - (float)bz;
    return sqrtf(dx*dx + dy*dy + dz*dz);
}

/* ============================================================
 * f16 conversion helpers
 * ============================================================ */

__device__ __host__ __forceinline__
uint16_t float_to_f16_bits(float v) {
    __half h = __float2half(v);
    return *(uint16_t*)&h;
}

__device__ __host__ __forceinline__
float f16_bits_to_float(uint16_t bits) {
    __half h = *(__half*)&bits;
    return __half2float(h);
}

/* ============================================================
 * Hash table constants
 * ============================================================ */

#define SDST_EMPTY_KEY      0xFFFFFFFF
#define SDST_EMPTY_VALUE    0xFFFFFFFF
#define SDST_OCCUPIED_BIT   0x80000000
#define SDST_KEY_MASK       0x7FFFFFFF

/** Open-addressing hash function (Murmur3 finalizer) */
__device__ __host__ __forceinline__
uint32_t sdst_hash(uint32_t key, uint32_t capacity) {
    key ^= key >> 16;
    key *= 0x85ebca6b;
    key ^= key >> 13;
    key *= 0xc2b2ae35;
    key ^= key >> 16;
    return key & (capacity - 1); /* capacity must be power of 2 */
}

/* ============================================================
 * GPU Union-Find for avalanche detection
 * ============================================================ */

/** Atomic path-compressed find */
__device__ __forceinline__
uint32_t uf_find(uint32_t* parent, uint32_t x) {
    uint32_t root = x;
    while (parent[root] != root) {
        root = parent[root];
    }
    /* Path compression */
    while (parent[x] != root) {
        uint32_t next = parent[x];
        atomicCAS(&parent[x], next, root);
        x = next;
    }
    return root;
}

/** Atomic union by index (lower index wins) */
__device__ __forceinline__
void uf_union(uint32_t* parent, uint32_t a, uint32_t b) {
    while (true) {
        uint32_t ra = uf_find(parent, a);
        uint32_t rb = uf_find(parent, b);
        if (ra == rb) return;
        if (ra > rb) { uint32_t t = ra; ra = rb; rb = t; }
        if (atomicCAS(&parent[rb], rb, ra) == rb) return;
    }
}

/* ============================================================
 * Forward declarations for internal functions used across .cu files
 * Must use extern "C" to match the definitions in sdst_wct.cu / sdst_tcl.cu
 * ============================================================ */
extern "C" SdstError sdst_process_wavefronts(SdstHandle handle, uint32_t base_idx, uint32_t n_events, void* stream);
extern "C" SdstError sdst_compute_tcl_flags(SdstHandle handle, uint32_t base_idx, uint32_t n_events, void* stream);

/* ============================================================
 * Internal context structure
 * ============================================================ */

struct SdstContext {
    SdstConfig config;

    /* Device memory */
    HashEntry*      d_hash_table;       /* Hash table slots */
    SpikeEvent*     d_event_buffer;     /* All spike events */
    uint32_t*       d_event_count;      /* Atomic counter for events */
    uint32_t*       d_avalanche_parent; /* Union-find parent array */
    uint32_t*       d_avalanche_size;   /* Per-root avalanche sizes */
    uint32_t*       d_wavefront_count;  /* Atomic counter for wavefronts */
    uint32_t*       d_phase_boundaries; /* Device copy of phase_boundaries[6] */

    /* Per-voxel chain heads: maps Morton code to most recent event index */
    uint32_t*       d_voxel_chain;      /* [hash_table_capacity] */
    uint32_t*       d_event_chain_next; /* [max_spike_events] next ptr per event */

    /* Wavefront state */
    WavefrontStats* d_wavefront_stats;  /* [max_wavefronts] */
    uint32_t*       d_voxel_last_wavefront; /* Most recent wavefront per voxel */
    uint32_t*       d_voxel_last_time;      /* Timestamp of last spike per voxel */

    /* Temporal index: per-timestep event ranges for O(1) time queries */
    uint32_t*       d_time_index_start; /* [max_timesteps] start index */
    uint32_t*       d_time_index_count; /* [max_timesteps] count */
    uint32_t        max_timesteps;

    /* Query scratch buffers (per-stream) */
    SpikeEvent**    d_query_buffers;    /* [num_streams] */
    uint32_t*       d_query_counts;     /* [num_streams] */
    uint32_t        query_buffer_size;

    /* CUDA streams */
    cudaStream_t*   streams;
    uint32_t        num_streams;

    /* Host-side state */
    uint32_t        h_event_count;
    uint32_t        h_wavefront_count;
};

/* ============================================================
 * Kernel launch helpers
 * ============================================================ */

#define SDST_BLOCK_SIZE 256
#define SDST_GRID_SIZE(n) (((n) + SDST_BLOCK_SIZE - 1) / SDST_BLOCK_SIZE)

/* Maximum events per query result */
#define SDST_QUERY_BUFFER_SIZE (1 << 20) /* 1M events */

#endif /* SDST_INTERNAL_H */
