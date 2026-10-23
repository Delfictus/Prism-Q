/**
 * SDST Core: Hash table, spike insertion, parent detection, avalanche clustering.
 *
 * This is the foundation of the PRISM-Therm data structure. Every spike event
 * from the NHS engine enters through sdst_insert_spikes() or sdst_insert_raw(),
 * which:
 *   1. Morton-encodes the voxel coordinate
 *   2. Inserts into the open-addressing hash table
 *   3. Detects causal parent via spatial-temporal proximity
 *   4. Assigns avalanche membership via GPU union-find
 *   5. Chains events per-voxel for temporal queries
 */

#include "../include/sdst_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Thrust for GPU radix sort (used by sorted parent detection) */
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

/* ============================================================
 * Error string
 * ============================================================ */

extern "C"
const char* sdst_error_string(SdstError err) {
    switch (err) {
        case SDST_SUCCESS:                  return "Success";
        case SDST_ERROR_CUDA:               return "CUDA error";
        case SDST_ERROR_OOM:                return "Out of GPU memory";
        case SDST_ERROR_INVALID_PARAM:      return "Invalid parameter";
        case SDST_ERROR_TABLE_FULL:         return "Hash table full";
        case SDST_ERROR_NOT_FOUND:          return "Not found";
        case SDST_ERROR_WAVEFRONT_OVERFLOW: return "Wavefront table overflow";
        case SDST_ERROR_STREAM_INVALID:     return "Invalid CUDA stream";
        default:                            return "Unknown error";
    }
}

/* ============================================================
 * Default config
 * ============================================================ */

extern "C"
SdstConfig sdst_default_config(void) {
    SdstConfig cfg;
    memset(&cfg, 0, sizeof(cfg));

    cfg.grid_nx = 128;
    cfg.grid_ny = 128;
    cfg.grid_nz = 128;
    cfg.grid_spacing = 0.75f;

    cfg.hash_table_capacity = 1 << 22; /* 4M slots, ~2x 128³ */
    cfg.max_spike_events = 2000000;

    cfg.max_wavefronts = 65536;
    cfg.wavefront_merge_dist = 2.25f;  /* 3 voxels * 0.75Å */
    cfg.wavefront_max_dt = 5;          /* 5 timestep window */

    cfg.avalanche_spatial_cutoff = 7.5f;  /* 10 voxels * 0.75Å — wider net for sparse NHS spikes */
    cfg.avalanche_max_gap = 50;           /* NHS fires ~1 spike per 100-1000 steps/voxel; 3 caught nothing */

    /* Default 5-phase hysteresis: 14K/6K/15K/6K/14K steps */
    cfg.phase_boundaries[0] = 0;
    cfg.phase_boundaries[1] = 14000;
    cfg.phase_boundaries[2] = 20000;
    cfg.phase_boundaries[3] = 35000;
    cfg.phase_boundaries[4] = 41000;
    cfg.phase_boundaries[5] = 55000;

    cfg.ccns_soc_threshold = 1.5f;
    cfg.ccns_barrier_threshold = 2.0f;

    cfg.num_streams = 4;
    cfg.device_id = 0;

    return cfg;
}

/* ============================================================
 * Lifecycle: create / destroy / reset
 * ============================================================ */

extern "C"
SdstError sdst_create(const SdstConfig* config, SdstHandle* out_handle) {
    if (!config || !out_handle) return SDST_ERROR_INVALID_PARAM;

    SDST_CUDA_CHECK(cudaSetDevice(config->device_id));

    SdstContext* ctx = (SdstContext*)calloc(1, sizeof(SdstContext));
    if (!ctx) return SDST_ERROR_OOM;
    ctx->config = *config;

    uint32_t cap = config->hash_table_capacity;
    uint32_t max_ev = config->max_spike_events;
    uint32_t max_wf = config->max_wavefronts;
    uint32_t max_ts = config->phase_boundaries[5] + 1;
    ctx->max_timesteps = max_ts;

    /* Hash table */
    SDST_CUDA_CHECK(cudaMalloc(&ctx->d_hash_table, cap * sizeof(HashEntry)));
    SDST_CUDA_CHECK(cudaMemset(ctx->d_hash_table, 0xFF, cap * sizeof(HashEntry)));

    /* Event buffer */
    SDST_CUDA_CHECK(cudaMalloc(&ctx->d_event_buffer, max_ev * sizeof(SpikeEvent)));
    SDST_CUDA_CHECK(cudaMalloc(&ctx->d_event_count, sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemset(ctx->d_event_count, 0, sizeof(uint32_t)));

    /* Per-voxel chain */
    SDST_CUDA_CHECK(cudaMalloc(&ctx->d_voxel_chain, cap * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemset(ctx->d_voxel_chain, 0xFF, cap * sizeof(uint32_t)));

    /* Event chain (linked list next pointers) */
    SDST_CUDA_CHECK(cudaMalloc(&ctx->d_event_chain_next, max_ev * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemset(ctx->d_event_chain_next, 0xFF, max_ev * sizeof(uint32_t)));

    /* Avalanche union-find */
    SDST_CUDA_CHECK(cudaMalloc(&ctx->d_avalanche_parent, max_ev * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMalloc(&ctx->d_avalanche_size, max_ev * sizeof(uint32_t)));

    /* Wavefront tracking */
    SDST_CUDA_CHECK(cudaMalloc(&ctx->d_wavefront_count, sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemset(ctx->d_wavefront_count, 0, sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMalloc(&ctx->d_wavefront_stats, max_wf * sizeof(WavefrontStats)));
    SDST_CUDA_CHECK(cudaMalloc(&ctx->d_voxel_last_wavefront, cap * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemset(ctx->d_voxel_last_wavefront, 0xFF, cap * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMalloc(&ctx->d_voxel_last_time, cap * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemset(ctx->d_voxel_last_time, 0, cap * sizeof(uint32_t)));

    /* Phase boundaries (device copy for kernel access) */
    SDST_CUDA_CHECK(cudaMalloc(&ctx->d_phase_boundaries, 6 * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemcpy(ctx->d_phase_boundaries, config->phase_boundaries,
                               6 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    /* Temporal index */
    SDST_CUDA_CHECK(cudaMalloc(&ctx->d_time_index_start, max_ts * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMalloc(&ctx->d_time_index_count, max_ts * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemset(ctx->d_time_index_start, 0xFF, max_ts * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemset(ctx->d_time_index_count, 0, max_ts * sizeof(uint32_t)));

    /* Query buffers (one per stream) */
    ctx->query_buffer_size = SDST_QUERY_BUFFER_SIZE;
    ctx->num_streams = config->num_streams;
    ctx->d_query_buffers = (SpikeEvent**)calloc(ctx->num_streams, sizeof(SpikeEvent*));
    ctx->d_query_counts = NULL;
    SDST_CUDA_CHECK(cudaMalloc(&ctx->d_query_counts, ctx->num_streams * sizeof(uint32_t)));
    for (uint32_t i = 0; i < ctx->num_streams; i++) {
        SDST_CUDA_CHECK(cudaMalloc(&ctx->d_query_buffers[i],
                                   SDST_QUERY_BUFFER_SIZE * sizeof(SpikeEvent)));
    }

    /* CUDA streams */
    ctx->streams = (cudaStream_t*)calloc(ctx->num_streams, sizeof(cudaStream_t));
    for (uint32_t i = 0; i < ctx->num_streams; i++) {
        SDST_CUDA_CHECK(cudaStreamCreate(&ctx->streams[i]));
    }

    ctx->h_event_count = 0;
    ctx->h_wavefront_count = 0;

    *out_handle = ctx;
    return SDST_SUCCESS;
}

extern "C"
SdstError sdst_destroy(SdstHandle handle) {
    if (!handle) return SDST_ERROR_INVALID_PARAM;
    SdstContext* ctx = handle;

    cudaFree(ctx->d_hash_table);
    cudaFree(ctx->d_event_buffer);
    cudaFree(ctx->d_event_count);
    cudaFree(ctx->d_voxel_chain);
    cudaFree(ctx->d_event_chain_next);
    cudaFree(ctx->d_avalanche_parent);
    cudaFree(ctx->d_avalanche_size);
    cudaFree(ctx->d_wavefront_count);
    cudaFree(ctx->d_wavefront_stats);
    cudaFree(ctx->d_voxel_last_wavefront);
    cudaFree(ctx->d_voxel_last_time);
    cudaFree(ctx->d_phase_boundaries);
    cudaFree(ctx->d_time_index_start);
    cudaFree(ctx->d_time_index_count);
    cudaFree(ctx->d_query_counts);

    for (uint32_t i = 0; i < ctx->num_streams; i++) {
        cudaFree(ctx->d_query_buffers[i]);
        cudaStreamDestroy(ctx->streams[i]);
    }
    free(ctx->d_query_buffers);
    free(ctx->streams);
    free(ctx);

    return SDST_SUCCESS;
}

extern "C"
SdstError sdst_reset(SdstHandle handle) {
    if (!handle) return SDST_ERROR_INVALID_PARAM;
    SdstContext* ctx = handle;

    uint32_t cap = ctx->config.hash_table_capacity;
    uint32_t max_ev = ctx->config.max_spike_events;

    SDST_CUDA_CHECK(cudaMemset(ctx->d_hash_table, 0xFF, cap * sizeof(HashEntry)));
    SDST_CUDA_CHECK(cudaMemset(ctx->d_event_count, 0, sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemset(ctx->d_voxel_chain, 0xFF, cap * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemset(ctx->d_event_chain_next, 0xFF, max_ev * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemset(ctx->d_wavefront_count, 0, sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemset(ctx->d_voxel_last_wavefront, 0xFF, cap * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemset(ctx->d_voxel_last_time, 0, cap * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemset(ctx->d_time_index_start, 0xFF, ctx->max_timesteps * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemset(ctx->d_time_index_count, 0, ctx->max_timesteps * sizeof(uint32_t)));

    ctx->h_event_count = 0;
    ctx->h_wavefront_count = 0;

    return SDST_SUCCESS;
}

extern "C"
SdstError sdst_event_count(SdstHandle handle, uint32_t* out_count) {
    if (!handle || !out_count) return SDST_ERROR_INVALID_PARAM;
    SDST_CUDA_CHECK(cudaMemcpy(out_count, handle->d_event_count,
                               sizeof(uint32_t), cudaMemcpyDeviceToHost));
    handle->h_event_count = *out_count;
    return SDST_SUCCESS;
}

extern "C"
SdstError sdst_memory_usage(SdstHandle handle, size_t* out_bytes) {
    if (!handle || !out_bytes) return SDST_ERROR_INVALID_PARAM;
    SdstContext* ctx = handle;
    uint32_t cap = ctx->config.hash_table_capacity;
    uint32_t max_ev = ctx->config.max_spike_events;
    uint32_t max_wf = ctx->config.max_wavefronts;

    *out_bytes = 0;
    *out_bytes += cap * sizeof(HashEntry);                  /* hash table */
    *out_bytes += max_ev * sizeof(SpikeEvent);              /* event buffer */
    *out_bytes += cap * sizeof(uint32_t);                   /* voxel chain heads */
    *out_bytes += max_ev * sizeof(uint32_t);                /* event chain next */
    *out_bytes += max_ev * sizeof(uint32_t) * 2;            /* union-find */
    *out_bytes += max_wf * sizeof(WavefrontStats);          /* wavefront stats */
    *out_bytes += cap * sizeof(uint32_t) * 2;               /* voxel last wavefront/time */
    *out_bytes += ctx->max_timesteps * sizeof(uint32_t) * 2; /* temporal index */
    *out_bytes += ctx->num_streams * SDST_QUERY_BUFFER_SIZE * sizeof(SpikeEvent);

    return SDST_SUCCESS;
}

/* ============================================================
 * Kernel: Hash table insert (open addressing, linear probing)
 * ============================================================ */

__global__
void kernel_hash_insert(
    HashEntry*      hash_table,
    uint32_t        capacity,
    uint32_t*       voxel_chain,
    uint32_t*       event_chain_next,
    SpikeEvent*     event_buffer,
    uint32_t*       event_count_ptr,
    const SpikeEvent* new_events,
    uint32_t        n_events,
    uint32_t        base_event_idx,
    uint32_t*       voxel_last_time  /* Track most recent timestamp per voxel slot */
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_events) return;

    SpikeEvent ev = new_events[tid];
    uint32_t event_idx = base_event_idx + tid;
    MortonCode key = ev.voxel;

    /* Store event in buffer */
    event_buffer[event_idx] = ev;

    /* Insert into hash table with linear probing */
    uint32_t slot = sdst_hash(key, capacity);
    for (uint32_t probe = 0; probe < capacity; probe++) {
        uint32_t idx = (slot + probe) & (capacity - 1);
        uint32_t old_key = atomicCAS(&hash_table[idx].key, SDST_EMPTY_KEY, key);

        if (old_key == SDST_EMPTY_KEY || old_key == key) {
            /* Slot acquired or already ours - update voxel chain */
            uint32_t old_head = atomicExch(&voxel_chain[idx], event_idx);
            event_chain_next[event_idx] = old_head;
            hash_table[idx].head_idx = event_idx;
            /* Track latest timestamp at this hash slot for temporal filtering */
            atomicMax(&voxel_last_time[idx], ev.timestamp);
            return;
        }
    }
    /* Table full - should not happen with proper sizing */
}

/* ============================================================
 * Kernel: Parent spike detection via spatial-temporal proximity
 *
 * For each new spike, find the most recent spike in neighboring
 * voxels within the avalanche spatial cutoff and time gap.
 * That spike is the causal parent.
 * ============================================================ */

__global__
void kernel_detect_parents(
    SpikeEvent*     event_buffer,
    const HashEntry* hash_table,
    const uint32_t* voxel_chain,
    const uint32_t* event_chain_next,
    uint32_t        capacity,
    uint32_t        base_event_idx,
    uint32_t        n_events,
    float           spatial_cutoff_grid, /* in grid units */
    uint32_t        max_time_gap,
    uint32_t        grid_nx, uint32_t grid_ny, uint32_t grid_nz,
    const uint32_t* voxel_last_time  /* Latest timestamp per hash slot for temporal skip */
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_events) return;

    uint32_t my_idx = base_event_idx + tid;
    SpikeEvent* my_ev = &event_buffer[my_idx];
    MortonCode my_morton = my_ev->voxel;
    uint32_t my_time = my_ev->timestamp;

    /* Earliest acceptable parent timestamp */
    uint32_t time_floor = (my_time > max_time_gap) ? (my_time - max_time_gap) : 0;

    uint32_t mx, my_y, mz;
    morton_decode(my_morton, &mx, &my_y, &mz);

    /* Search radius in grid cells */
    int radius = (int)(spatial_cutoff_grid + 0.5f);
    uint32_t best_parent = 0; /* 0 = no parent (spontaneous) */
    uint32_t best_time = 0;
    float best_dist = spatial_cutoff_grid + 1.0f;

    /* Scan neighboring voxels */
    for (int dz = -radius; dz <= radius; dz++) {
        int nz = (int)mz + dz;
        if (nz < 0 || nz >= (int)grid_nz) continue;
        for (int dy = -radius; dy <= radius; dy++) {
            int ny = (int)my_y + dy;
            if (ny < 0 || ny >= (int)grid_ny) continue;
            for (int dx = -radius; dx <= radius; dx++) {
                int nx = (int)mx + dx;
                if (nx < 0 || nx >= (int)grid_nx) continue;
                if (dx == 0 && dy == 0 && dz == 0) continue;

                float dist = sqrtf((float)(dx*dx + dy*dy + dz*dz));
                if (dist > spatial_cutoff_grid) continue;

                MortonCode neighbor = morton_encode(nx, ny, nz);
                uint32_t slot = sdst_hash(neighbor, capacity);

                /* Probe for this neighbor in hash table */
                for (uint32_t probe = 0; probe < 32; probe++) {
                    uint32_t idx = (slot + probe) & (capacity - 1);
                    uint32_t stored_key = hash_table[idx].key;
                    if (stored_key == SDST_EMPTY_KEY) break;
                    if (stored_key != neighbor) continue;

                    /* Temporal skip: if the newest event at this voxel is older
                     * than our time window, no event here can be a valid parent.
                     * This turns O(chain_length) into O(1) for stale voxels. */
                    uint32_t last_t = voxel_last_time[idx];
                    if (last_t < time_floor) break;

                    /* Walk event chain for this voxel.
                     * Walk limit raised from 16→512; temporal batching keeps
                     * chains short so this limit is a safety bound, not a
                     * correctness constraint. */
                    uint32_t chain_idx = voxel_chain[idx];
                    uint32_t walk = 0;
                    while (chain_idx != SDST_EMPTY_VALUE && walk < 512) {
                        SpikeEvent* candidate = &event_buffer[chain_idx];
                        uint32_t ct = candidate->timestamp;
                        if (ct < my_time && ct >= time_floor) {
                            /* Candidate parent: closer in time wins,
                               ties broken by spatial distance */
                            if (ct > best_time ||
                                (ct == best_time && dist < best_dist)) {
                                best_parent = chain_idx + 1; /* +1 so 0 = none */
                                best_time = ct;
                                best_dist = dist;
                            }
                        }
                        chain_idx = event_chain_next[chain_idx];
                        walk++;
                    }
                    break;
                }
            }
        }
    }

    my_ev->parent_spike = best_parent;
}

/* ============================================================
 * Kernel: Avalanche assignment via union-find
 *
 * Each spike with a parent is unioned with its parent.
 * Avalanche ID = root of the union-find tree.
 * ============================================================ */

/* ============================================================
 * Avalanche clustering via iterative pointer jumping
 *
 * Replaces atomic-heavy union-find with a lock-free, read-mostly
 * algorithm. Each event follows its parent chain to find the root
 * (the spontaneous event at the top). Pointer jumping doubles the
 * jump distance each iteration, converging in O(log D) iterations
 * where D is max chain depth. Each iteration: 13.2M threads, each
 * doing 2 reads + 1 write with NO atomics. ~1ms per iteration,
 * ~30ms total. Union-find took hours due to atomicCAS contention.
 * ============================================================ */

/** Initialize avalanche labels: each event points to its parent.
 *  Roots (spontaneous spikes with no parent) point to themselves. */
__global__
void kernel_avalanche_label_init(
    const SpikeEvent* event_buffer,
    uint32_t*         labels,
    uint32_t          base_idx,
    uint32_t          n_events
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_events) return;
    uint32_t idx = base_idx + tid;
    uint32_t parent = event_buffer[idx].parent_spike;
    labels[idx] = (parent > 0) ? (parent - 1) : idx;
}

/** Pointer jump: each label jumps to its grandparent.
 *  After K iterations, chains of length 2^K collapse to roots.
 *  No atomics needed — each thread writes only to its own index. */
__global__
void kernel_avalanche_jump(
    uint32_t*       labels,
    uint32_t        base_idx,
    uint32_t        n_events
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_events) return;
    uint32_t idx = base_idx + tid;
    uint32_t cur = labels[idx];
    uint32_t next = labels[cur];
    if (cur != next) {
        labels[idx] = next;
    }
}

/** Write final avalanche ID from converged labels to events. */
__global__
void kernel_avalanche_finalize(
    SpikeEvent*       event_buffer,
    const uint32_t*   labels,
    uint32_t          base_idx,
    uint32_t          n_events
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_events) return;
    uint32_t idx = base_idx + tid;
    event_buffer[idx].avalanche_id = labels[idx];
}

/* ============================================================
 * Kernel: Temporal index update
 * ============================================================ */

__global__
void kernel_update_time_index(
    uint32_t*       time_index_start,
    uint32_t*       time_index_count,
    const SpikeEvent* new_events,
    uint32_t        n_events,
    uint32_t        base_event_idx
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_events) return;

    uint32_t ts = new_events[tid].timestamp;
    /* Atomically set start index (keep minimum) */
    atomicMin(&time_index_start[ts], base_event_idx + tid);
    atomicAdd(&time_index_count[ts], 1);
}

/* ============================================================
 * Kernel: Convert raw inputs to SpikeEvents with Morton encoding
 * ============================================================ */

__global__
void kernel_encode_raw_inputs(
    const SpikeInput* inputs,
    SpikeEvent*       events,
    uint32_t          n_events
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_events) return;

    SpikeInput in = inputs[tid];
    SpikeEvent ev;
    memset(&ev, 0, sizeof(SpikeEvent));

    ev.voxel = morton_encode(in.voxel_x, in.voxel_y, in.voxel_z);
    ev.timestamp = in.timestamp;
    ev.amplitude = float_to_f16_bits(in.amplitude);
    ev.local_temp = float_to_f16_bits(in.local_temp);
    ev.energy_gradient = float_to_f16_bits(in.energy_gradient);
    ev.solvent_exposure = float_to_f16_bits(in.solvent_exposure);
    ev.phase_id = in.phase_id;
    ev.parent_spike = 0;
    ev.avalanche_id = 0;
    ev.wavefront_id = 0;
    ev.tcl_flags = 0;

    events[tid] = ev;
}

/* ============================================================
 * Kernel: Convert NHS raw spike buffer (92 bytes/event) to SpikeEvent
 *
 * NHS GpuSpikeEvent layout (92 bytes, matching Rust fused_engine.rs):
 *   offset  0: timestep         (i32)
 *   offset  4: voxel_idx        (i32)  — linear index: vx + vy*128 + vz*128²
 *   offset  8: pos_x,y,z        (3×f32, unused here)
 *   offset 20: intensity         (f32)
 *   offset 24: nearby_residues   (8×i32, unused here)
 *   offset 56: n_residues        (i32, unused)
 *   offset 60: spike_source      (i32, unused)
 *   offset 64: wavelength_nm     (f32, unused)
 *   offset 68: aromatic_type     (i32, unused)
 *   offset 72: aromatic_res_id   (i32, unused)
 *   offset 76: water_density     (f32)
 *   offset 80: vibrational_energy(f32, unused)
 *   offset 84: n_nearby_excited  (i32, unused)
 *   offset 88: wd_change         (f32)
 *
 * Computes phase_id, local_temp, energy_gradient, solvent_exposure
 * from protocol parameters — same logic as Rust convert_spike().
 * ============================================================ */

__global__
void kernel_convert_nhs_to_sdst(
    const uint8_t*  nhs_buffer,      /* NHS spike events (sorted by timestamp) */
    SpikeEvent*     events_out,      /* Output: SDST SpikeEvent format */
    uint32_t        n_events,
    uint32_t        nhs_stride,      /* sizeof(GpuSpikeEvent) = 92 */
    uint32_t        grid_dim,        /* 128 */
    float           start_temp,      /* Protocol cold temperature (K) */
    float           end_temp,        /* Protocol warm temperature (K) */
    uint32_t        cold_hold,       /* Phase boundary: cold hold steps */
    uint32_t        ramp_up,         /* Phase boundary: ramp up steps */
    uint32_t        warm_hold,       /* Phase boundary: warm hold steps */
    uint32_t        ramp_down        /* Phase boundary: ramp down steps */
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_events) return;

    /* Parse NHS 92-byte format */
    const uint8_t* base = nhs_buffer + (size_t)tid * nhs_stride;
    int32_t timestep    = *(const int32_t*)(base + 0);
    int32_t voxel_idx   = *(const int32_t*)(base + 4);
    float   intensity   = *(const float*)(base + 20);
    float   water_dens  = *(const float*)(base + 76);
    float   wd_change   = *(const float*)(base + 88);

    /* Clamp to valid ranges */
    uint32_t ts = (uint32_t)max(timestep, 0);
    uint32_t idx = (uint32_t)max(voxel_idx, 0);
    uint32_t dim_max = grid_dim - 1;

    /* Decode linear voxel index to (x, y, z) grid coords */
    uint32_t vx = idx % grid_dim;
    uint32_t vy = (idx / grid_dim) % grid_dim;
    uint32_t vz = idx / (grid_dim * grid_dim);
    vx = min(vx, dim_max);
    vy = min(vy, dim_max);
    vz = min(vz, dim_max);

    /* Phase boundaries (cumulative) */
    uint32_t p1 = cold_hold;
    uint32_t p2 = p1 + ramp_up;
    uint32_t p3 = p2 + warm_hold;
    uint32_t p4 = p3 + ramp_down;

    /* Phase ID */
    uint8_t phase_id;
    if      (ts < p1) phase_id = 0;
    else if (ts < p2) phase_id = 1;
    else if (ts < p3) phase_id = 2;
    else if (ts < p4) phase_id = 3;
    else              phase_id = 4;

    /* Temperature interpolation (matches Rust temp_at()) */
    float t_protocol;
    if (ts < p1) {
        t_protocol = start_temp;
    } else if (ts < p2) {
        float frac = (float)(ts - p1) / fmaxf((float)ramp_up, 1.0f);
        t_protocol = start_temp + frac * (end_temp - start_temp);
    } else if (ts < p3) {
        t_protocol = end_temp;
    } else if (ts < p4) {
        float frac = (float)(ts - p3) / fmaxf((float)ramp_down, 1.0f);
        t_protocol = end_temp - frac * (end_temp - start_temp);
    } else {
        t_protocol = start_temp;
    }

    /* Spatially modulated local temperature:
     * Buried voxels (WD≈0) see 1.5× protocol temp;
     * solvated voxels (WD≈1) see ~1× protocol temp. */
    float wd_clamped = fminf(fmaxf(water_dens, 0.0f), 1.0f);
    float local_temp = t_protocol * (1.0f + 0.5f * (1.0f - wd_clamped));

    /* Build output SpikeEvent */
    SpikeEvent ev;
    memset(&ev, 0, sizeof(SpikeEvent));
    ev.voxel            = morton_encode(vx, vy, vz);
    ev.timestamp        = ts;
    ev.amplitude        = float_to_f16_bits(intensity);
    ev.local_temp       = float_to_f16_bits(local_temp);
    ev.energy_gradient  = float_to_f16_bits(wd_change);
    ev.solvent_exposure = float_to_f16_bits(water_dens);
    ev.phase_id         = phase_id;
    ev.parent_spike     = 0;
    ev.avalanche_id     = 0;
    ev.wavefront_id     = 0;
    ev.tcl_flags        = 0;

    events_out[tid] = ev;
}

/* ============================================================
 * Public: Insert spike events
 * ============================================================ */

extern "C"
SdstError sdst_insert_spikes(
    SdstHandle handle,
    SpikeEvent* d_events,
    uint32_t count,
    void* stream
) {
    if (!handle || !d_events || count == 0) return SDST_ERROR_INVALID_PARAM;
    SdstContext* ctx = handle;
    cudaStream_t s = stream ? (cudaStream_t)stream : 0;

    /* Get current event count (base index for this batch) */
    uint32_t base_idx;
    SDST_CUDA_CHECK(cudaMemcpyAsync(&base_idx, ctx->d_event_count,
                                     sizeof(uint32_t), cudaMemcpyDeviceToHost, s));
    SDST_CUDA_CHECK(cudaStreamSynchronize(s));

    if (base_idx + count > ctx->config.max_spike_events) {
        return SDST_ERROR_TABLE_FULL;
    }

    /* Atomically reserve event slots */
    /* We use a temporary host-side approach; in production, use atomicAdd on device */
    uint32_t new_count = base_idx + count;
    SDST_CUDA_CHECK(cudaMemcpyAsync(ctx->d_event_count, &new_count,
                                     sizeof(uint32_t), cudaMemcpyHostToDevice, s));

    dim3 grid(SDST_GRID_SIZE(count));
    dim3 block(SDST_BLOCK_SIZE);

    float spatial_cutoff_grid = ctx->config.avalanche_spatial_cutoff / ctx->config.grid_spacing;

    /* 1. Insert into hash table and build voxel chains */
    kernel_hash_insert<<<grid, block, 0, s>>>(
        ctx->d_hash_table, ctx->config.hash_table_capacity,
        ctx->d_voxel_chain, ctx->d_event_chain_next,
        ctx->d_event_buffer, ctx->d_event_count,
        d_events, count, base_idx,
        ctx->d_voxel_last_time
    );
    SDST_CUDA_CHECK_KERNEL();

    /* 2. Detect causal parents via spatial-temporal proximity */
    kernel_detect_parents<<<grid, block, 0, s>>>(
        ctx->d_event_buffer, ctx->d_hash_table,
        ctx->d_voxel_chain, ctx->d_event_chain_next,
        ctx->config.hash_table_capacity,
        base_idx, count,
        spatial_cutoff_grid, ctx->config.avalanche_max_gap,
        ctx->config.grid_nx, ctx->config.grid_ny, ctx->config.grid_nz,
        ctx->d_voxel_last_time
    );
    SDST_CUDA_CHECK_KERNEL();

    /* 3. Avalanche clustering via pointer jumping (no atomics) */
    kernel_avalanche_label_init<<<grid, block, 0, s>>>(
        ctx->d_event_buffer, ctx->d_avalanche_parent, base_idx, count
    );
    SDST_CUDA_CHECK_KERNEL();
    for (int iter = 0; iter < 30; iter++) {
        kernel_avalanche_jump<<<grid, block, 0, s>>>(
            ctx->d_avalanche_parent, base_idx, count
        );
        SDST_CUDA_CHECK_KERNEL();
    }
    kernel_avalanche_finalize<<<grid, block, 0, s>>>(
        ctx->d_event_buffer, ctx->d_avalanche_parent, base_idx, count
    );
    SDST_CUDA_CHECK_KERNEL();

    /* 4. Update temporal index */
    kernel_update_time_index<<<grid, block, 0, s>>>(
        ctx->d_time_index_start, ctx->d_time_index_count,
        d_events, count, base_idx
    );
    SDST_CUDA_CHECK_KERNEL();

    /* 5. Compute TCL (thermodynamic context layer) flags */
    SdstError tcl_err = sdst_compute_tcl_flags(handle, base_idx, count, stream);
    if (tcl_err != SDST_SUCCESS) return tcl_err;

    /* 6. Wavefront coherence tracking */
    SdstError wf_err = sdst_process_wavefronts(handle, base_idx, count, stream);
    if (wf_err != SDST_SUCCESS) return wf_err;

    ctx->h_event_count = new_count;

    return SDST_SUCCESS;
}

extern "C"
SdstError sdst_insert_raw(
    SdstHandle handle,
    const SpikeInput* d_inputs,
    uint32_t count,
    void* stream
) {
    if (!handle || !d_inputs || count == 0) return SDST_ERROR_INVALID_PARAM;
    SdstContext* ctx = handle;
    cudaStream_t s = stream ? (cudaStream_t)stream : 0;

    /* Allocate temporary buffer for encoded events */
    SpikeEvent* d_encoded;
    SDST_CUDA_CHECK(cudaMallocAsync(&d_encoded, count * sizeof(SpikeEvent), s));

    dim3 grid(SDST_GRID_SIZE(count));
    dim3 block(SDST_BLOCK_SIZE);

    kernel_encode_raw_inputs<<<grid, block, 0, s>>>(d_inputs, d_encoded, count);
    SDST_CUDA_CHECK_KERNEL();

    SdstError err = sdst_insert_spikes(handle, d_encoded, count, stream);

    cudaFreeAsync(d_encoded, s);
    return err;
}

/* ============================================================
 * Kernel: Temporal-sweep parent detection
 *
 * Replaces the O(N × 9261 × chain_length) hash-probe approach
 * with O(N × W × S) temporal sweep, where W = avg timesteps
 * checked before early exit (~25) and S = events per timestep
 * (~240). Total: ~6000 sequential reads vs 27K random reads.
 *
 * Algorithm: for each event at voxel V, time T:
 *   Sweep timesteps T-1, T-2, ..., T-max_gap:
 *     Linear scan all events at that timestep (contiguous in memory)
 *     For each: compute grid-distance, check spatial cutoff
 *     Early exit once any parent found (nearest-in-time wins)
 *
 * Memory access: L2-friendly — timestep segments are contiguous
 * and small (240 events × 36 bytes = 8.6 KB). Warps processing
 * events at similar timestamps share L2-cached segments.
 * ============================================================ */

__global__
void kernel_detect_parents_sweep(
    SpikeEvent*       event_buffer,       /* All events, timestamp-sorted */
    const uint32_t*   time_index_start,   /* [max_timesteps] first event idx */
    const uint32_t*   time_index_count,   /* [max_timesteps] event count */
    uint32_t          n_events,
    uint32_t          base_event_idx,
    float             spatial_cutoff_sq,  /* (cutoff_in_grid_units)² */
    uint32_t          max_time_gap,
    uint32_t          max_timesteps
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_events) return;

    uint32_t my_idx = base_event_idx + tid;
    SpikeEvent my_ev = event_buffer[my_idx]; /* local copy — register-resident */
    uint32_t my_time = my_ev.timestamp;

    uint32_t mx, my_y, mz;
    morton_decode(my_ev.voxel, &mx, &my_y, &mz);
    int imx = (int)mx, imy = (int)my_y, imz = (int)mz;

    uint32_t time_floor = (my_time > max_time_gap) ? (my_time - max_time_gap) : 0;

    uint32_t best_parent = 0;   /* 0 = no parent (spontaneous) */
    float    best_dist_sq = spatial_cutoff_sq + 1.0f;

    /* Sweep backwards: newest potential parent first.
     * Early exit on first timestep with a spatial match —
     * that timestep contains the nearest-in-time parent. */
    for (uint32_t ts = my_time; ts > time_floor; ) {
        ts--;
        if (ts >= max_timesteps) continue;

        uint32_t seg_start = time_index_start[ts];
        if (seg_start == 0xFFFFFFFF) continue; /* no events at this timestep */
        uint32_t seg_count = time_index_count[ts];

        /* Linear scan of contiguous timestep segment.
         * ~240 events avg, 36 bytes each = 8.6 KB — fits in L2. */
        bool found_at_ts = false;
        for (uint32_t j = seg_start; j < seg_start + seg_count; j++) {
            SpikeEvent cand = event_buffer[j];
            uint32_t cx, cy, cz;
            morton_decode(cand.voxel, &cx, &cy, &cz);
            int dx = imx - (int)cx;
            int dy = imy - (int)cy;
            int dz = imz - (int)cz;
            float dist_sq = (float)(dx*dx + dy*dy + dz*dz);

            if (dist_sq <= spatial_cutoff_sq) {
                /* Within spatial cutoff — candidate parent.
                 * Since we sweep newest-first, all candidates at this
                 * timestep share the same time. Pick nearest spatially. */
                if (dist_sq < best_dist_sq) {
                    best_parent = j + 1;   /* +1 encoding: 0 = none */
                    best_dist_sq = dist_sq;
                    found_at_ts = true;
                }
            }
        }

        /* Nearest-in-time parent guaranteed at this timestep.
         * No older timestep can produce a closer-in-time match. */
        if (found_at_ts) break;
    }

    event_buffer[my_idx].parent_spike = best_parent;
}

/* ============================================================
 * Sorted parent detection — O(log n) per voxel via contiguous arrays
 *
 * Strategy: sort events by (morton_code, timestamp), build a dense
 * per-voxel lookup table, then binary search for temporal window.
 * Replaces linked-list chain walks (random pointer chasing) with
 * coalesced sorted array access.
 *
 * Memory layout after sort:
 *   d_sort_keys[N]:  uint64_t = (morton << 32) | timestamp
 *   d_sort_vals[N]:  uint32_t = original event index
 *   d_vlut_start[2^21]: start index in sorted array per morton code
 *   d_vlut_count[2^21]: event count per morton code
 * ============================================================ */

#define SDST_MORTON_BITS  21
#define SDST_VLUT_SIZE    (1u << SDST_MORTON_BITS) /* 2M entries = 16 MB */

/** Build composite sort keys: (morton << 32) | timestamp */
__global__
void kernel_build_sort_keys(
    const SpikeEvent* events,
    uint64_t*         sort_keys,
    uint32_t*         sort_values,
    uint32_t          n_events,
    uint32_t          base_idx
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_events) return;
    uint32_t idx = base_idx + tid;
    sort_keys[tid] = ((uint64_t)events[idx].voxel << 32) | (uint64_t)events[idx].timestamp;
    sort_values[tid] = idx;
}

/** Detect segment boundaries in sorted keys to build per-voxel lookup.
 *  Sets voxel_start[morton] for the FIRST event at each voxel. */
__global__
void kernel_vlut_set_starts(
    const uint64_t* sort_keys,
    uint32_t        n_events,
    uint32_t*       voxel_start
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_events) return;
    uint32_t my_morton = (uint32_t)(sort_keys[tid] >> 32);
    if (tid == 0 || (uint32_t)(sort_keys[tid - 1] >> 32) != my_morton) {
        voxel_start[my_morton] = tid;
    }
}

/** Compute segment lengths from starts (runs after kernel_vlut_set_starts). */
__global__
void kernel_vlut_set_counts(
    const uint64_t* sort_keys,
    uint32_t        n_events,
    const uint32_t* voxel_start,
    uint32_t*       voxel_count
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_events) return;
    uint32_t my_morton = (uint32_t)(sort_keys[tid] >> 32);
    if (tid == n_events - 1 || (uint32_t)(sort_keys[tid + 1] >> 32) != my_morton) {
        voxel_count[my_morton] = tid - voxel_start[my_morton] + 1;
    }
}

/** Parent detection via sorted per-voxel segments + binary search.
 *
 * For each event: enumerate 343 neighbor morton codes within radius=3,
 * O(1) lookup in dense voxel table, binary search for temporal window,
 * linear scan of ~20 candidates. No linked-list pointer chasing. */
__global__
void kernel_detect_parents_sorted(
    SpikeEvent*       event_buffer,
    const uint64_t*   sort_keys,
    const uint32_t*   sort_values,
    const uint32_t*   vlut_start,
    const uint32_t*   vlut_count,
    uint32_t          n_events,
    uint32_t          base_idx,
    float             spatial_cutoff_grid,
    uint32_t          max_time_gap,
    uint32_t          grid_nx, uint32_t grid_ny, uint32_t grid_nz
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_events) return;

    uint32_t my_idx = base_idx + tid;
    SpikeEvent* my_ev = &event_buffer[my_idx];
    uint32_t my_time = my_ev->timestamp;
    MortonCode my_morton = my_ev->voxel;

    uint32_t time_floor = (my_time > max_time_gap) ? (my_time - max_time_gap) : 0;

    uint32_t mx, my_y, mz;
    morton_decode(my_morton, &mx, &my_y, &mz);

    int radius = (int)(spatial_cutoff_grid + 0.5f);
    uint32_t best_parent = 0;
    uint32_t best_time = 0;
    float best_dist = spatial_cutoff_grid + 1.0f;

    for (int dz = -radius; dz <= radius; dz++) {
        int nz = (int)mz + dz;
        if (nz < 0 || nz >= (int)grid_nz) continue;
        for (int dy = -radius; dy <= radius; dy++) {
            int ny = (int)my_y + dy;
            if (ny < 0 || ny >= (int)grid_ny) continue;
            for (int dx = -radius; dx <= radius; dx++) {
                int nx = (int)mx + dx;
                if (nx < 0 || nx >= (int)grid_nx) continue;
                if (dx == 0 && dy == 0 && dz == 0) continue;

                float dist = sqrtf((float)(dx*dx + dy*dy + dz*dz));
                if (dist > spatial_cutoff_grid) continue;

                MortonCode neighbor = morton_encode(nx, ny, nz);

                /* O(1) dense lookup — no hash probing */
                uint32_t seg_count = vlut_count[neighbor];
                if (seg_count == 0) continue;
                uint32_t seg_start = vlut_start[neighbor];

                /* Binary search for UPPER bound (my_time) within this
                 * voxel's sorted segment. The most recent event with
                 * ts < my_time is at position (lo - 1).
                 * O(log n) + O(1) — no linear scan needed. */
                uint64_t target_hi = ((uint64_t)neighbor << 32) | (uint64_t)my_time;
                uint32_t lo = seg_start, hi = seg_start + seg_count;
                while (lo < hi) {
                    uint32_t mid = (lo + hi) >> 1;
                    if (sort_keys[mid] < target_hi)
                        lo = mid + 1;
                    else
                        hi = mid;
                }

                /* lo = first event with ts >= my_time.
                 * (lo - 1) = most recent event with ts < my_time.
                 * Check it's within the time window [time_floor, my_time). */
                if (lo > seg_start) {
                    uint32_t j = lo - 1;
                    uint32_t ts = (uint32_t)(sort_keys[j] & 0xFFFFFFFF);
                    if (ts >= time_floor) {
                        if (ts > best_time ||
                            (ts == best_time && dist < best_dist)) {
                            best_parent = sort_values[j] + 1;
                            best_time = ts;
                            best_dist = dist;
                        }
                    }
                }
            }
        }
    }

    my_ev->parent_spike = best_parent;
}

/* ============================================================
 * Public: Insert from NHS raw spike buffer (GPU-native path)
 *
 * Architecture: Sorted-Index Insertion Pipeline
 *
 * Eliminates the CPU conversion round-trip. Takes a HOST pointer to
 * the accumulated GpuSpikeEvent[] array (sorted by timestep on CPU),
 * uploads to GPU, converts via kernel, then runs a decomposed
 * pipeline:
 *
 *   1. Convert NHS → SpikeEvent         (GPU kernel, single pass)
 *   2. Hash insert (all events)          (builds hash table + chains
 *      for downstream analysis queries)
 *   3. Build temporal index              (for time-range queries)
 *   4. Sorted parent detection           (Thrust radix sort by
 *      (morton, timestamp), dense VLUT, binary search within
 *      per-voxel segments — O(N × 179 × log(n_per_voxel)))
 *   5. Avalanche clustering              (union-find kernels)
 *   6. TCL + wavefront coherence         (analysis functions)
 *
 * Expected SDST phase time for 13M events: ~2-5 seconds.
 * ============================================================ */

extern "C"
SdstError sdst_insert_from_nhs_buffer(
    SdstHandle handle,
    const void* h_nhs_events,   /* HOST pointer to sorted GpuSpikeEvent[] */
    uint32_t    count,
    uint32_t    nhs_stride,     /* sizeof(GpuSpikeEvent) = 92 */
    float       start_temp,     /* Protocol cold temperature (K) */
    float       end_temp,       /* Protocol warm temperature (K) */
    uint32_t    cold_hold,      /* Protocol: cold_hold_steps */
    uint32_t    ramp_up,        /* Protocol: ramp_steps */
    uint32_t    warm_hold,      /* Protocol: warm_hold_steps */
    uint32_t    ramp_down,      /* Protocol: ramp_down_steps */
    void*       stream
) {
    if (!handle || !h_nhs_events || count == 0) return SDST_ERROR_INVALID_PARAM;
    SdstContext* ctx = handle;
    cudaStream_t s = stream ? (cudaStream_t)stream : 0;

    if (count > ctx->config.max_spike_events) {
        return SDST_ERROR_TABLE_FULL;
    }

    /* Timing markers for performance diagnostics */
    cudaEvent_t t_start, t_s1, t_s2, t_s3, t_s4a, t_s4b, t_s4c, t_s4d, t_s5, t_s6;
    cudaEventCreate(&t_start); cudaEventCreate(&t_s1); cudaEventCreate(&t_s2);
    cudaEventCreate(&t_s3); cudaEventCreate(&t_s4a); cudaEventCreate(&t_s4b);
    cudaEventCreate(&t_s4c); cudaEventCreate(&t_s4d); cudaEventCreate(&t_s5);
    cudaEventCreate(&t_s6);
    cudaEventRecord(t_start, s);

    /* ── Stage 1: Bulk upload + convert NHS → SpikeEvent ──────── */

    size_t nhs_total_bytes = (size_t)count * nhs_stride;
    uint8_t* d_nhs;
    SDST_CUDA_CHECK(cudaMallocAsync(&d_nhs, nhs_total_bytes, s));
    SDST_CUDA_CHECK(cudaMemcpyAsync(d_nhs, h_nhs_events, nhs_total_bytes,
                                     cudaMemcpyHostToDevice, s));

    SpikeEvent* d_converted;
    SDST_CUDA_CHECK(cudaMallocAsync(&d_converted, (size_t)count * sizeof(SpikeEvent), s));

    {
        dim3 grid(SDST_GRID_SIZE(count));
        dim3 block(SDST_BLOCK_SIZE);
        kernel_convert_nhs_to_sdst<<<grid, block, 0, s>>>(
            d_nhs, d_converted, count, nhs_stride,
            ctx->config.grid_nx,
            start_temp, end_temp,
            cold_hold, ramp_up, warm_hold, ramp_down
        );
        SDST_CUDA_CHECK_KERNEL();
    }

    cudaFreeAsync(d_nhs, s);
    cudaEventRecord(t_s1, s);

    /* ── Stage 2: Hash insert (copies to event_buffer, builds hash table) ── */

    uint32_t base_idx = 0; /* NHS bulk path always starts from 0 */
    uint32_t new_count = count;
    SDST_CUDA_CHECK(cudaMemcpyAsync(ctx->d_event_count, &new_count,
                                     sizeof(uint32_t), cudaMemcpyHostToDevice, s));

    {
        dim3 grid(SDST_GRID_SIZE(count));
        dim3 block(SDST_BLOCK_SIZE);
        kernel_hash_insert<<<grid, block, 0, s>>>(
            ctx->d_hash_table, ctx->config.hash_table_capacity,
            ctx->d_voxel_chain, ctx->d_event_chain_next,
            ctx->d_event_buffer, ctx->d_event_count,
            d_converted, count, base_idx,
            ctx->d_voxel_last_time
        );
        SDST_CUDA_CHECK_KERNEL();
    }

    cudaFreeAsync(d_converted, s);
    cudaEventRecord(t_s2, s);

    /* ── Stage 3: Build temporal index (BEFORE parent detection) ── */

    {
        dim3 grid(SDST_GRID_SIZE(count));
        dim3 block(SDST_BLOCK_SIZE);
        kernel_update_time_index<<<grid, block, 0, s>>>(
            ctx->d_time_index_start, ctx->d_time_index_count,
            ctx->d_event_buffer, count, base_idx
        );
        SDST_CUDA_CHECK_KERNEL();
    }

    cudaEventRecord(t_s3, s);

    /* ── Stage 4: Sorted parent detection (radius=3) ──────────── */
    /* Thrust radix sort by (morton, timestamp) + dense per-voxel
     * lookup table + binary search.  Replaces linked-list chain
     * walks with contiguous sorted array reads — cache-friendly
     * and O(log n) per voxel instead of O(chain_length).       */

    {
        float cutoff_grid = 3.0f; /* radius=3: optimal speed/accuracy for current LIF */

        /* Allocate sort buffers */
        uint64_t* d_sort_keys;
        uint32_t* d_sort_vals;
        SDST_CUDA_CHECK(cudaMallocAsync(&d_sort_keys, (size_t)count * sizeof(uint64_t), s));
        SDST_CUDA_CHECK(cudaMallocAsync(&d_sort_vals, (size_t)count * sizeof(uint32_t), s));

        /* Build composite keys: (morton << 32) | timestamp */
        {
            dim3 grid(SDST_GRID_SIZE(count));
            dim3 block(SDST_BLOCK_SIZE);
            kernel_build_sort_keys<<<grid, block, 0, s>>>(
                ctx->d_event_buffer, d_sort_keys, d_sort_vals, count, base_idx
            );
            SDST_CUDA_CHECK_KERNEL();
        }

        /* Radix sort by key — groups events by voxel, sorted by timestamp within */
        SDST_CUDA_CHECK(cudaStreamSynchronize(s));
        cudaEventRecord(t_s4a, s);
        {
            thrust::device_ptr<uint64_t> k_ptr(d_sort_keys);
            thrust::device_ptr<uint32_t> v_ptr(d_sort_vals);
            thrust::sort_by_key(thrust::cuda::par.on(s), k_ptr, k_ptr + count, v_ptr);
        }
        SDST_CUDA_CHECK(cudaStreamSynchronize(s));
        cudaEventRecord(t_s4b, s);

        /* Build per-voxel lookup table (VLUT) */
        uint32_t* d_vlut_start;
        uint32_t* d_vlut_count;
        SDST_CUDA_CHECK(cudaMallocAsync(&d_vlut_start, SDST_VLUT_SIZE * sizeof(uint32_t), s));
        SDST_CUDA_CHECK(cudaMallocAsync(&d_vlut_count, SDST_VLUT_SIZE * sizeof(uint32_t), s));
        SDST_CUDA_CHECK(cudaMemsetAsync(d_vlut_start, 0xFF, SDST_VLUT_SIZE * sizeof(uint32_t), s));
        SDST_CUDA_CHECK(cudaMemsetAsync(d_vlut_count, 0,    SDST_VLUT_SIZE * sizeof(uint32_t), s));

        {
            dim3 grid(SDST_GRID_SIZE(count));
            dim3 block(SDST_BLOCK_SIZE);
            kernel_vlut_set_starts<<<grid, block, 0, s>>>(
                d_sort_keys, count, d_vlut_start
            );
            SDST_CUDA_CHECK_KERNEL();
        }
        SDST_CUDA_CHECK(cudaStreamSynchronize(s));
        {
            dim3 grid(SDST_GRID_SIZE(count));
            dim3 block(SDST_BLOCK_SIZE);
            kernel_vlut_set_counts<<<grid, block, 0, s>>>(
                d_sort_keys, count, d_vlut_start, d_vlut_count
            );
            SDST_CUDA_CHECK_KERNEL();
        }

        cudaEventRecord(t_s4c, s);

        /* Parent detection via sorted segments */
        {
            dim3 grid(SDST_GRID_SIZE(count));
            dim3 block(SDST_BLOCK_SIZE);
            kernel_detect_parents_sorted<<<grid, block, 0, s>>>(
                ctx->d_event_buffer,
                d_sort_keys, d_sort_vals,
                d_vlut_start, d_vlut_count,
                count, base_idx,
                cutoff_grid,
                ctx->config.avalanche_max_gap,
                ctx->config.grid_nx, ctx->config.grid_ny, ctx->config.grid_nz
            );
            SDST_CUDA_CHECK_KERNEL();
        }

        cudaEventRecord(t_s4d, s);

        /* Cleanup temporary sorted index */
        cudaFreeAsync(d_sort_keys, s);
        cudaFreeAsync(d_sort_vals, s);
        cudaFreeAsync(d_vlut_start, s);
        cudaFreeAsync(d_vlut_count, s);
    }

    cudaEventRecord(t_s5, s);

    /* ── Stage 5: Avalanche clustering (pointer jumping) ──────── */
    /* Lock-free, no atomics. Iterative pointer doubling converges
     * in O(log D) iterations. Each iteration: 1 kernel launch,
     * 13.2M threads × 2 reads + 1 write. ~30ms total vs hours. */

    {
        dim3 grid(SDST_GRID_SIZE(count));
        dim3 block(SDST_BLOCK_SIZE);

        kernel_avalanche_label_init<<<grid, block, 0, s>>>(
            ctx->d_event_buffer, ctx->d_avalanche_parent, base_idx, count
        );
        SDST_CUDA_CHECK_KERNEL();

        for (int iter = 0; iter < 30; iter++) {
            kernel_avalanche_jump<<<grid, block, 0, s>>>(
                ctx->d_avalanche_parent, base_idx, count
            );
            SDST_CUDA_CHECK_KERNEL();
        }

        kernel_avalanche_finalize<<<grid, block, 0, s>>>(
            ctx->d_event_buffer, ctx->d_avalanche_parent, base_idx, count
        );
        SDST_CUDA_CHECK_KERNEL();
    }

    /* ── Stage 6: TCL + Wavefront coherence ───────────────────── */

    SdstError tcl_err = sdst_compute_tcl_flags(handle, base_idx, count, stream);
    if (tcl_err != SDST_SUCCESS) return tcl_err;

    SdstError wf_err = sdst_process_wavefronts(handle, base_idx, count, stream);
    if (wf_err != SDST_SUCCESS) return wf_err;

    /* ── Finalize + timing report ─────────────────────────────── */

    cudaEventRecord(t_s6, s);
    ctx->h_event_count = new_count;
    SDST_CUDA_CHECK(cudaStreamSynchronize(s));

    /* Print per-stage timing */
    {
        float ms_s1, ms_s2, ms_s3, ms_sort_keys, ms_sort, ms_vlut, ms_parents, ms_s5, ms_s6;
        cudaEventElapsedTime(&ms_s1, t_start, t_s1);
        cudaEventElapsedTime(&ms_s2, t_s1, t_s2);
        cudaEventElapsedTime(&ms_s3, t_s2, t_s3);
        cudaEventElapsedTime(&ms_sort_keys, t_s3, t_s4a);
        cudaEventElapsedTime(&ms_sort, t_s4a, t_s4b);
        cudaEventElapsedTime(&ms_vlut, t_s4b, t_s4c);
        cudaEventElapsedTime(&ms_parents, t_s4c, t_s4d);
        cudaEventElapsedTime(&ms_s5, t_s4d, t_s5);
        cudaEventElapsedTime(&ms_s6, t_s5, t_s6);
        float total;
        cudaEventElapsedTime(&total, t_start, t_s6);
        fprintf(stderr, "[SDST] Pipeline timing for %u events:\n", count);
        fprintf(stderr, "  S1 upload+convert:  %8.1f ms\n", ms_s1);
        fprintf(stderr, "  S2 hash insert:     %8.1f ms\n", ms_s2);
        fprintf(stderr, "  S3 temporal index:  %8.1f ms\n", ms_s3);
        fprintf(stderr, "  S4a sort keys:      %8.1f ms\n", ms_sort_keys);
        fprintf(stderr, "  S4b thrust sort:    %8.1f ms\n", ms_sort);
        fprintf(stderr, "  S4c VLUT build:     %8.1f ms\n", ms_vlut);
        fprintf(stderr, "  S4d parent detect:  %8.1f ms\n", ms_parents);
        fprintf(stderr, "  S5 avalanche:       %8.1f ms\n", ms_s5);
        fprintf(stderr, "  S6 TCL+wavefronts:  %8.1f ms\n", ms_s6);
        fprintf(stderr, "  TOTAL:              %8.1f ms\n", total);
    }

    cudaEventDestroy(t_start); cudaEventDestroy(t_s1); cudaEventDestroy(t_s2);
    cudaEventDestroy(t_s3); cudaEventDestroy(t_s4a); cudaEventDestroy(t_s4b);
    cudaEventDestroy(t_s4c); cudaEventDestroy(t_s4d); cudaEventDestroy(t_s5);
    cudaEventDestroy(t_s6);

    return SDST_SUCCESS;
}

/* ============================================================
 * Debug: print stats
 * ============================================================ */

extern "C"
SdstError sdst_print_stats(SdstHandle handle) {
    if (!handle) return SDST_ERROR_INVALID_PARAM;
    SdstContext* ctx = handle;

    uint32_t ev_count, wf_count;
    SDST_CUDA_CHECK(cudaMemcpy(&ev_count, ctx->d_event_count,
                               sizeof(uint32_t), cudaMemcpyDeviceToHost));
    SDST_CUDA_CHECK(cudaMemcpy(&wf_count, ctx->d_wavefront_count,
                               sizeof(uint32_t), cudaMemcpyDeviceToHost));

    size_t mem;
    sdst_memory_usage(handle, &mem);

    printf("=== SDST Statistics ===\n");
    printf("Events:     %u / %u (%.1f%%)\n",
           ev_count, ctx->config.max_spike_events,
           100.0f * ev_count / ctx->config.max_spike_events);
    printf("Wavefronts: %u / %u\n", wf_count, ctx->config.max_wavefronts);
    printf("Hash table: %u slots\n", ctx->config.hash_table_capacity);
    printf("GPU memory: %.1f MB\n", (float)mem / (1024 * 1024));
    printf("Grid:       %u x %u x %u @ %.2f Å\n",
           ctx->config.grid_nx, ctx->config.grid_ny, ctx->config.grid_nz,
           ctx->config.grid_spacing);
    printf("=======================\n");

    return SDST_SUCCESS;
}
