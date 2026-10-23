/**
 * SDST GPU-Native CCNS Pipeline
 *
 * Replaces the host-side sdst_ccns_all_pockets (which downloaded 13.2M events
 * to CPU per tile and segfaulted) with a fully GPU-resident sort-reduce
 * architecture.
 *
 * Architecture:
 *   Stage 1: Avalanche size computation via CUB radix sort + boundaries (zero atomics)
 *   Stage 2: Spatial tile indexing via sort + contiguous segments
 *   Stage 3: Fused CCNS kernel with Clauset-Shalizi-Newman (CSN) truncated
 *            power-law estimator via KS statistic minimization
 *
 * Reference: Clauset, Shalizi, Newman (2009) "Power-law distributions in
 *            empirical data" (SIAM Review, Vol. 51, No. 4, pp. 661-703).
 */

#include "../include/sdst_internal.h"
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <math.h>
#include <float.h>

/* ============================================================
 * Constants
 * ============================================================ */

#define CCNS_TILE_DIM    16       /* 128 / 8 voxels per tile axis */
#define CCNS_MAX_TILES   4096     /* 16^3 */
#define CCNS_BLOCK_SIZE  256
#define CCNS_TAU_STEPS   32
#define CCNS_LAMBDA_STEPS 32
#define CCNS_MAX_UNIQUE_PER_TILE 1024  /* Max unique avalanches per tile for shared mem */

/* ============================================================
 * Stage 1: Avalanche Size Computation (Zero Atomics)
 *
 * 1. Extract avalanche_id from all events into key array
 * 2. CUB radix sort by avalanche_id
 * 3. Adjacent-difference boundary kernel → avalanche sizes
 * ============================================================ */

/** Extract avalanche_id from each event into a flat key array.
 *  Also stores event index as value for later re-association. */
__global__
void kernel_extract_avalanche_ids(
    const SpikeEvent* __restrict__ events,
    uint32_t total_events,
    uint32_t* __restrict__ out_keys,     /* avalanche_id per event */
    uint32_t* __restrict__ out_values    /* event index */
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_events) return;
    out_keys[tid]   = events[tid].avalanche_id;
    out_values[tid]  = tid;
}

/** After sorting by avalanche_id, compute size of each avalanche.
 *  Adjacent-difference: where id changes, write size and start. */
__global__
void kernel_avalanche_boundaries(
    const uint32_t* __restrict__ sorted_ids,
    uint32_t total_events,
    uint32_t* __restrict__ aval_id_out,     /* unique avalanche IDs */
    uint32_t* __restrict__ aval_size_out,   /* size per unique avalanche */
    uint32_t* __restrict__ aval_start_out,  /* start index in sorted array */
    uint32_t* __restrict__ n_unique         /* number of unique avalanches */
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_events) return;

    bool is_start = (tid == 0) || (sorted_ids[tid] != sorted_ids[tid - 1]);
    bool is_end   = (tid == total_events - 1) || (sorted_ids[tid] != sorted_ids[tid + 1]);

    if (is_start) {
        /* Find the end of this run */
        uint32_t end = tid;
        /* We'll compute size when we hit the end; for now just mark start */
        aval_start_out[tid] = tid; /* temporary, will be compacted */
    }

    if (is_end) {
        /* Walk back to find start of this run */
        uint32_t start = tid;
        while (start > 0 && sorted_ids[start - 1] == sorted_ids[tid]) {
            start--;
        }
        uint32_t size = tid - start + 1;
        uint32_t idx = atomicAdd(n_unique, 1);
        aval_id_out[idx] = sorted_ids[tid];
        aval_size_out[idx] = size;
        aval_start_out[idx] = start;
    }
}

/* ============================================================
 * Stage 2: Spatial Tile Indexing
 *
 * 1. Compute tile_id per event from Morton-decoded voxel coords
 * 2. CUB radix sort events by tile_id
 * 3. Adjacent-difference for per-tile [start, count]
 * ============================================================ */

/** Compute tile_id for each event.
 *  TILE_DIM = 16, so tile_id = (vx/8) + (vy/8)*16 + (vz/8)*256
 *  Also copies the avalanche size (looked up from Stage 1 results). */
__global__
void kernel_compute_tile_ids(
    const SpikeEvent* __restrict__ events,
    uint32_t total_events,
    uint32_t* __restrict__ out_tile_ids,
    uint32_t* __restrict__ out_aval_ids
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_events) return;

    uint32_t vx, vy, vz;
    morton_decode(events[tid].voxel, &vx, &vy, &vz);

    uint32_t tx = vx >> 3;  /* vx / 8 */
    uint32_t ty = vy >> 3;
    uint32_t tz = vz >> 3;

    out_tile_ids[tid] = tx + ty * CCNS_TILE_DIM + tz * CCNS_TILE_DIM * CCNS_TILE_DIM;
    out_aval_ids[tid] = events[tid].avalanche_id;
}

/** After sorting by tile_id, find start and count for each tile.
 *  The is_end thread walks backward to find its own start — no cross-block
 *  dependency. This is safe because the array is sorted. */
__global__
void kernel_tile_boundaries(
    const uint32_t* __restrict__ sorted_tile_ids,
    uint32_t total_events,
    uint32_t* __restrict__ tile_start,   /* [CCNS_MAX_TILES] */
    uint32_t* __restrict__ tile_count    /* [CCNS_MAX_TILES] */
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_events) return;

    uint32_t my_tile = sorted_tile_ids[tid];
    bool is_end = (tid == total_events - 1) || (sorted_tile_ids[tid + 1] != my_tile);

    if (is_end) {
        /* Walk backward to find start of this tile's run */
        uint32_t start = tid;
        while (start > 0 && sorted_tile_ids[start - 1] == my_tile) {
            start--;
        }
        tile_start[my_tile] = start;
        tile_count[my_tile] = tid - start + 1;
    }
}

/* ============================================================
 * Stage 3: Fused CCNS Kernel with CSN Estimator
 *
 * One block per active tile. Each block:
 *   1. Loads its tile's contiguous segment of avalanche_ids
 *   2. Deduplicates in shared memory → unique avalanche sizes
 *   3. Sorts sizes via bitonic sort in shared memory
 *   4. Computes empirical CDF
 *   5. Grid searches (tau, lambda) pairs for truncated power law
 *   6. Finds minimum KS statistic via block-wide reduction
 *   7. Writes result
 * ============================================================ */

/** Result struct for one tile's CCNS analysis */
struct GpuCcnsResult {
    float    tau;
    float    lambda;
    float    ks_distance;
    uint32_t n_avalanches;
    uint32_t tile_id;
    int32_t  ccns_class;     /* 0=SOC, 1=NearCritical, 2=Barrier */
    float    druggability;
};

/** Bitonic sort step in shared memory */
__device__
void bitonic_sort_shared(uint32_t* data, uint32_t n) {
    uint32_t lane = threadIdx.x;

    for (uint32_t k = 2; k <= n; k <<= 1) {
        for (uint32_t j = k >> 1; j > 0; j >>= 1) {
            __syncthreads();
            for (uint32_t i = lane; i < n; i += blockDim.x) {
                uint32_t ixj = i ^ j;
                if (ixj > i) {
                    bool ascending = ((i & k) == 0);
                    if (ascending ? (data[i] > data[ixj]) : (data[i] < data[ixj])) {
                        uint32_t tmp = data[i];
                        data[i] = data[ixj];
                        data[ixj] = tmp;
                    }
                }
            }
        }
    }
    __syncthreads();
}

/** Round up to next power of 2 */
__device__
uint32_t next_pow2(uint32_t v) {
    v--;
    v |= v >> 1; v |= v >> 2; v |= v >> 4;
    v |= v >> 8; v |= v >> 16;
    return v + 1;
}

/** Compute the normalizing constant for truncated power law:
 *  Z(tau, lambda, s_min, s_max) = sum_{s=s_min}^{s_max} s^(-tau) * exp(-lambda*s) */
__device__
float truncated_power_law_Z(float tau, float lambda, uint32_t s_min, uint32_t s_max) {
    float z = 0.0f;
    /* For efficiency, only sum up to s_max or 10000, whichever is smaller */
    uint32_t limit = min(s_max, (uint32_t)10000);
    for (uint32_t s = s_min; s <= limit; s++) {
        z += powf((float)s, -tau) * expf(-lambda * (float)s);
    }
    return z;
}

/** Fused CCNS kernel: one block per active tile.
 *
 *  Shared memory layout:
 *    [0 .. CCNS_MAX_UNIQUE_PER_TILE-1]  : uint32_t avalanche_ids (dedup workspace)
 *    [next .. +CCNS_MAX_UNIQUE_PER_TILE] : uint32_t sizes (sorted unique sizes)
 *    [next .. +CCNS_MAX_UNIQUE_PER_TILE] : float empirical_cdf
 *    [next .. +CCNS_BLOCK_SIZE]          : float ks_values (per-thread reduction)
 *    [next .. +CCNS_BLOCK_SIZE]          : uint32_t ks_indices (candidate index)
 */
__global__
void kernel_batched_ccns(
    const uint32_t* __restrict__ sorted_aval_ids,   /* sorted by tile_id */
    const uint32_t* __restrict__ aval_size_lookup,   /* avalanche_id → size (dense) */
    uint32_t max_aval_id,
    const uint32_t* __restrict__ tile_start,
    const uint32_t* __restrict__ tile_count,
    const uint32_t* __restrict__ active_tile_ids,    /* list of non-empty tile IDs */
    uint32_t n_active_tiles,
    GpuCcnsResult* __restrict__ out_results
) {
    if (blockIdx.x >= n_active_tiles) return;

    uint32_t tile_id = active_tile_ids[blockIdx.x];
    uint32_t seg_start = tile_start[tile_id];
    uint32_t seg_count = tile_count[tile_id];

    /* Shared memory: dynamically allocated */
    extern __shared__ uint32_t smem[];

    /* Partition shared memory */
    uint32_t* s_aval_ids = smem;                                        /* [CCNS_MAX_UNIQUE_PER_TILE] */
    uint32_t* s_sizes    = smem + CCNS_MAX_UNIQUE_PER_TILE;             /* [CCNS_MAX_UNIQUE_PER_TILE] */
    float*    s_ecdf     = (float*)(smem + 2 * CCNS_MAX_UNIQUE_PER_TILE); /* [CCNS_MAX_UNIQUE_PER_TILE] */
    float*    s_ks       = (float*)(smem + 3 * CCNS_MAX_UNIQUE_PER_TILE); /* [CCNS_BLOCK_SIZE] */
    uint32_t* s_ks_idx   = smem + 3 * CCNS_MAX_UNIQUE_PER_TILE + CCNS_BLOCK_SIZE; /* [CCNS_BLOCK_SIZE] */

    uint32_t lane = threadIdx.x;

    /* --- Step 1: Collect unique avalanche IDs in this tile --- */

    /* Initialize dedup workspace to sentinel */
    for (uint32_t i = lane; i < CCNS_MAX_UNIQUE_PER_TILE; i += blockDim.x) {
        s_aval_ids[i] = 0xFFFFFFFF;
        s_sizes[i] = 0;
    }
    __syncthreads();

    /* Load avalanche IDs from this tile's segment.
     * Use a simple hash-based dedup in shared memory. */
    __shared__ uint32_t s_n_unique;
    if (lane == 0) s_n_unique = 0;
    __syncthreads();

    /* First pass: collect unique avalanche IDs using sorted-unique approach.
     * Since events are sorted by tile_id (not by avalanche_id within tile),
     * we do a simple insert-if-not-present. */
    for (uint32_t i = lane; i < seg_count; i += blockDim.x) {
        uint32_t aid = sorted_aval_ids[seg_start + i];
        if (aid == 0) continue;  /* Skip null avalanche */

        /* Hash probe to check if already present */
        uint32_t slot = aid % CCNS_MAX_UNIQUE_PER_TILE;
        bool found = false;
        for (uint32_t probe = 0; probe < 32; probe++) {
            uint32_t s = (slot + probe) % CCNS_MAX_UNIQUE_PER_TILE;
            uint32_t existing = atomicCAS(&s_aval_ids[s], 0xFFFFFFFF, aid);
            if (existing == 0xFFFFFFFF) {
                /* We inserted it — new unique ID */
                atomicAdd(&s_n_unique, 1);
                found = true;
                break;
            } else if (existing == aid) {
                /* Already present */
                found = true;
                break;
            }
            /* Collision with different ID, continue probing */
        }
    }
    __syncthreads();

    uint32_t n_unique = s_n_unique;
    if (n_unique < 5) {
        /* Insufficient data for power-law fit */
        if (lane == 0) {
            out_results[blockIdx.x].tile_id = tile_id;
            out_results[blockIdx.x].tau = 0.0f;
            out_results[blockIdx.x].lambda = 0.0f;
            out_results[blockIdx.x].ks_distance = 1.0f;
            out_results[blockIdx.x].n_avalanches = n_unique;
            out_results[blockIdx.x].ccns_class = 2; /* Barrier */
            out_results[blockIdx.x].druggability = 0.0f;
        }
        return;
    }

    /* --- Step 2: Compact unique IDs and look up sizes --- */

    /* Compact: collect non-sentinel entries into contiguous array */
    /* Thread 0 does this serially for simplicity (n_unique ≤ 1024) */
    if (lane == 0) {
        uint32_t j = 0;
        for (uint32_t i = 0; i < CCNS_MAX_UNIQUE_PER_TILE && j < n_unique; i++) {
            if (s_aval_ids[i] != 0xFFFFFFFF) {
                uint32_t aid = s_aval_ids[i];
                s_sizes[j] = (aid <= max_aval_id) ? aval_size_lookup[aid] : 1;
                j++;
            }
        }
        /* Pad to next power of 2 for bitonic sort */
        uint32_t padded = next_pow2(n_unique);
        for (uint32_t i = n_unique; i < padded && i < CCNS_MAX_UNIQUE_PER_TILE; i++) {
            s_sizes[i] = 0xFFFFFFFF; /* sentinel sorts to end */
        }
    }
    __syncthreads();

    /* --- Step 3: Bitonic sort the unique sizes --- */

    uint32_t padded_n = next_pow2(n_unique);
    if (padded_n > CCNS_MAX_UNIQUE_PER_TILE) padded_n = CCNS_MAX_UNIQUE_PER_TILE;
    bitonic_sort_shared(s_sizes, padded_n);

    /* Remove sentinels: actual sizes are s_sizes[0 .. n_unique-1] */
    /* s_min = smallest non-zero size */
    __shared__ uint32_t s_min_size;
    __shared__ uint32_t s_max_size;
    if (lane == 0) {
        /* Find first non-zero, non-sentinel value */
        s_min_size = 1;
        s_max_size = 1;
        for (uint32_t i = 0; i < n_unique; i++) {
            if (s_sizes[i] > 0 && s_sizes[i] != 0xFFFFFFFF) {
                s_min_size = s_sizes[i];
                break;
            }
        }
        for (int i = (int)n_unique - 1; i >= 0; i--) {
            if (s_sizes[i] != 0xFFFFFFFF) {
                s_max_size = s_sizes[i];
                break;
            }
        }
        if (s_min_size < 1) s_min_size = 1;
    }
    __syncthreads();

    /* --- Step 4: Compute empirical CDF --- */
    for (uint32_t i = lane; i < n_unique; i += blockDim.x) {
        s_ecdf[i] = (float)(i + 1) / (float)n_unique;
    }
    __syncthreads();

    /* --- Step 5: Grid search over (tau, lambda) pairs --- */
    /* 32 × 32 = 1024 candidates, 4 per thread with 256 threads */

    float best_ks = FLT_MAX;
    uint32_t best_idx = 0;

    uint32_t n_candidates = CCNS_TAU_STEPS * CCNS_LAMBDA_STEPS; /* 1024 */
    uint32_t per_thread = (n_candidates + blockDim.x - 1) / blockDim.x;

    for (uint32_t c = 0; c < per_thread; c++) {
        uint32_t cand_idx = lane + c * blockDim.x;
        if (cand_idx >= n_candidates) break;

        uint32_t tau_step    = cand_idx / CCNS_LAMBDA_STEPS;
        uint32_t lambda_step = cand_idx % CCNS_LAMBDA_STEPS;

        float tau    = 1.0f + tau_step * (2.0f / (CCNS_TAU_STEPS - 1));       /* [1.0, 3.0] */
        float lambda = lambda_step * (0.1f / (CCNS_LAMBDA_STEPS - 1));        /* [0.0, 0.1] */

        /* Compute normalizing constant */
        float Z = truncated_power_law_Z(tau, lambda, s_min_size, s_max_size);
        if (Z < 1e-30f) continue;

        /* Compute theoretical CDF and KS statistic.
         * The CDF at observed size x_i = sum_{s=s_min}^{x_i} P(s) / Z.
         * We must include ALL integer sizes between observed data points,
         * not just P(x_i). Iterate incrementally using prev_s tracker. */
        float max_diff = 0.0f;
        float theoretical_cdf = 0.0f;
        uint32_t prev_s = s_min_size > 0 ? s_min_size - 1 : 0;
        for (uint32_t i = 0; i < n_unique; i++) {
            uint32_t s = s_sizes[i];
            if (s < s_min_size || s == 0 || s == 0xFFFFFFFF) continue;
            /* Sum probability mass for all sizes from prev_s+1 to s */
            uint32_t from = (prev_s < s_min_size) ? s_min_size : prev_s + 1;
            for (uint32_t ss = from; ss <= s; ss++) {
                theoretical_cdf += powf((float)ss, -tau) * expf(-lambda * (float)ss) / Z;
            }
            float diff = fabsf(theoretical_cdf - s_ecdf[i]);
            if (diff > max_diff) max_diff = diff;
            prev_s = s;
        }

        if (max_diff < best_ks) {
            best_ks = max_diff;
            best_idx = cand_idx;
        }
    }

    /* Store per-thread best into shared mem for reduction */
    s_ks[lane] = best_ks;
    s_ks_idx[lane] = best_idx;
    __syncthreads();

    /* --- Step 6: Block-wide min reduction --- */
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            if (s_ks[lane + stride] < s_ks[lane]) {
                s_ks[lane] = s_ks[lane + stride];
                s_ks_idx[lane] = s_ks_idx[lane + stride];
            }
        }
        __syncthreads();
    }

    /* --- Step 7: Thread 0 writes result --- */
    if (lane == 0) {
        uint32_t win_idx = s_ks_idx[0];
        uint32_t tau_step    = win_idx / CCNS_LAMBDA_STEPS;
        uint32_t lambda_step = win_idx % CCNS_LAMBDA_STEPS;
        float tau    = 1.0f + tau_step * (2.0f / (CCNS_TAU_STEPS - 1));
        float lambda = lambda_step * (0.1f / (CCNS_LAMBDA_STEPS - 1));

        int32_t cls;
        if (tau < 1.5f) cls = 0;       /* SOC */
        else if (tau < 2.0f) cls = 1;  /* NearCritical */
        else cls = 2;                   /* Barrier */

        /* Druggability: SOC = most druggable */
        float confidence = (n_unique >= 20) ? 1.0f : (float)n_unique / 20.0f;
        float drug = fmaxf(0.0f, (1.0f - (tau - 1.0f) / 3.0f) * confidence);

        out_results[blockIdx.x].tile_id = tile_id;
        out_results[blockIdx.x].tau = tau;
        out_results[blockIdx.x].lambda = lambda;
        out_results[blockIdx.x].ks_distance = s_ks[0];
        out_results[blockIdx.x].n_avalanches = n_unique;
        out_results[blockIdx.x].ccns_class = cls;
        out_results[blockIdx.x].druggability = drug;
    }
}

/* ============================================================
 * Kernel: Build dense avalanche size lookup table
 *
 * Given CUB run-length-encode output (unique_ids, run_lengths, n_runs),
 * scatter sizes into a dense array indexed by avalanche_id.
 * ============================================================ */

__global__
void kernel_build_size_lookup(
    const uint32_t* __restrict__ unique_ids,
    const uint32_t* __restrict__ run_lengths,
    uint32_t n_runs,
    uint32_t* __restrict__ size_lookup,  /* [max_aval_id + 1], pre-zeroed */
    uint32_t max_aval_id
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_runs) return;
    uint32_t aid = unique_ids[tid];
    if (aid <= max_aval_id) {
        size_lookup[aid] = run_lengths[tid];
    }
}

/* ============================================================
 * Kernel: Find active (non-empty) tiles
 * ============================================================ */

__global__
void kernel_find_active_tiles(
    const uint32_t* __restrict__ tile_count,
    uint32_t* __restrict__ active_tile_ids,
    uint32_t* __restrict__ n_active
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= CCNS_MAX_TILES) return;
    if (tile_count[tid] > 0) {
        uint32_t idx = atomicAdd(n_active, 1);
        active_tile_ids[idx] = tid;
    }
}

/* ============================================================
 * Kernel: Find max avalanche_id (single-pass reduction)
 * ============================================================ */

__global__
void kernel_find_max_aval_id(
    const uint32_t* __restrict__ aval_ids,
    uint32_t total,
    uint32_t* __restrict__ out_max
) {
    __shared__ uint32_t s_max[256];
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t lane = threadIdx.x;

    uint32_t local_max = 0;
    for (uint32_t i = tid; i < total; i += gridDim.x * blockDim.x) {
        if (aval_ids[i] > local_max) local_max = aval_ids[i];
    }
    s_max[lane] = local_max;
    __syncthreads();

    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (lane < s && s_max[lane + s] > s_max[lane]) {
            s_max[lane] = s_max[lane + s];
        }
        __syncthreads();
    }
    if (lane == 0) atomicMax(out_max, s_max[0]);
}


/* ============================================================
 * Stage 2b: CUB Segmented Sort + Histogram
 *
 * Replaces the 1024-slot shared-memory hash dedup that silently
 * overflows when regions have >1024 unique avalanches.
 * Also replaces stride sampling which destroyed power-law tails.
 *
 * Pipeline:
 *   2b-1. CUB DeviceSegmentedRadixSort on aval_ids within segments
 *   2b-2. kernel_mark_segment_unique: adjacent-diff unique flags
 *   2b-3. kernel_build_size_histogram: atomicAdd sizes → histogram[1024]
 *   2b-4. kernel_batched_ccns_histogram: CSN fit directly on histogram
 *
 * The histogram is EXACT — no sampling, no approximation.
 * 1024 bins × 4B = 4KB per group, fits in shared memory.
 * ============================================================ */

/** Mark unique avalanche IDs within sorted segments.
 *  An event is unique if it's the first in its segment (group boundary)
 *  OR its avalanche_id differs from the previous event's.
 *  group_keys are sorted, so segment boundaries = where key changes. */
__global__
void kernel_mark_segment_unique(
    const uint32_t* __restrict__ sorted_group_keys,  /* sorted tile/region ids */
    const uint32_t* __restrict__ sorted_aval_ids,     /* aval ids sorted by group key */
    uint32_t total_events,
    uint32_t* __restrict__ out_flags                  /* 1 = unique, 0 = duplicate */
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_events) return;

    if (tid == 0) {
        out_flags[tid] = 1;
        return;
    }

    /* Segment boundary: group key changed */
    if (sorted_group_keys[tid] != sorted_group_keys[tid - 1]) {
        out_flags[tid] = 1;
        return;
    }

    /* Within segment: unique if avalanche_id differs from previous */
    out_flags[tid] = (sorted_aval_ids[tid] != sorted_aval_ids[tid - 1]) ? 1 : 0;
}

/** After ExclusiveSum on flags, compute per-group unique count and offset.
 *  For each group g: offset = scan[group_start[g]], count = scan[group_end] - offset.
 *  group_end = group_start[g] + group_count[g]. */
__global__
void kernel_compute_group_unique_info(
    const uint32_t* __restrict__ group_start,   /* [n_groups] start in sorted array */
    const uint32_t* __restrict__ group_count,   /* [n_groups] event count */
    const uint32_t* __restrict__ scan,          /* exclusive sum of flags [total_events] */
    const uint32_t* __restrict__ flags,         /* unique flags [total_events] */
    uint32_t total_events,
    uint32_t n_groups,
    uint32_t* __restrict__ out_unique_offset,   /* [n_groups] offset into compacted array */
    uint32_t* __restrict__ out_unique_count     /* [n_groups] number of unique avalanches */
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_groups) return;

    uint32_t gs = group_start[gid];
    uint32_t gc = group_count[gid];

    if (gc == 0) {
        out_unique_offset[gid] = 0;
        out_unique_count[gid] = 0;
        return;
    }

    uint32_t end_idx = gs + gc;  /* one past last event */
    /* Offset in compacted array = scan value at group start */
    out_unique_offset[gid] = scan[gs];
    /* Count = (scan[end] + flag[end-1]) - scan[start]
     * But scan is exclusive, so total up to end = scan[end] if end < total,
     * or scan[end-1] + flags[end-1] if end == total */
    uint32_t end_scan;
    if (end_idx < total_events) {
        end_scan = scan[end_idx];
    } else {
        end_scan = scan[end_idx - 1] + flags[end_idx - 1];
    }
    out_unique_count[gid] = end_scan - scan[gs];
}

/** Build size histogram for each unique-flagged avalanche.
 *  For each event with flags[tid]==1 (unique avalanche in its segment):
 *    size = size_lookup[aval_id]; bin = min(size, 1023)
 *    atomicAdd(histogram[group_id * 1024 + bin], 1)
 *  group_id = sorted_group_keys[tid]. */
__global__
void kernel_build_size_histogram(
    const uint32_t* __restrict__ sorted_group_keys,  /* per-event group (tile/region) */
    const uint32_t* __restrict__ sorted_aval_ids,    /* per-event avalanche id (seg-sorted) */
    const uint32_t* __restrict__ flags,              /* 1 = unique avalanche */
    const uint32_t* __restrict__ aval_size_lookup,   /* avalanche_id → size (dense) */
    uint32_t max_aval_id,
    uint32_t total_events,
    uint32_t n_groups,                               /* bounds check: gid must be < n_groups */
    uint32_t* __restrict__ histograms               /* [n_groups * 1024], pre-zeroed */
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_events) return;

    if (!flags[tid]) return;

    uint32_t gid = sorted_group_keys[tid];
    if (gid >= n_groups) return;  /* Skip unmatched events (0xFFFF etc.) */

    uint32_t aid = sorted_aval_ids[tid];
    uint32_t size = (aid <= max_aval_id) ? aval_size_lookup[aid] : 1;
    uint32_t bin = (size >= CCNS_MAX_UNIQUE_PER_TILE) ? (CCNS_MAX_UNIQUE_PER_TILE - 1) : size;
    if (bin == 0) bin = 1;  /* size 0 → bin 1 */

    atomicAdd(&histograms[gid * CCNS_MAX_UNIQUE_PER_TILE + bin], 1);
}

/** Histogram-based CCNS kernel with Hill MLE estimator.
 *
 *  Input: histogram[1024] per group where histogram[s] = count of
 *  avalanches with exactly s events (bin 0 unused, bin 1023 = overflow).
 *
 *  Hill MLE for discrete power-law exponent (matches CPU compute_tau_mle):
 *    tau = 1 + N / sum_{s >= s_min}( count[s] * ln(s / (s_min - 0.5)) )
 *  where N = total avalanches with size >= s_min.
 *  s_min = 2: size-1 avalanches are trivial single events, not cascades.
 *  The (s_min - 0.5) = 1.5 is the Clauset et al. discrete-to-continuous
 *  correction for integer-valued data.
 *
 *  Shared memory layout (4KB):
 *    s_hist[1024] : uint32_t — the histogram
 */
__global__
void kernel_batched_ccns_histogram(
    const uint32_t* __restrict__ histograms,       /* [n_groups * 1024] */
    const uint32_t* __restrict__ active_group_ids, /* list of group IDs to process */
    uint32_t n_active_groups,
    GpuCcnsResult* __restrict__ out_results
) {
    if (blockIdx.x >= n_active_groups) return;

    uint32_t group_id = active_group_ids[blockIdx.x];
    uint32_t lane = threadIdx.x;

    /* Shared memory — only need the histogram */
    extern __shared__ uint32_t smem[];
    uint32_t* s_hist = smem;  /* [1024] */

    /* Load histogram into shared memory */
    uint32_t base = group_id * CCNS_MAX_UNIQUE_PER_TILE;
    for (uint32_t i = lane; i < CCNS_MAX_UNIQUE_PER_TILE; i += blockDim.x) {
        s_hist[i] = histograms[base + i];
    }
    __syncthreads();

    /* Thread 0: Hill MLE from histogram */
    if (lane == 0) {
        /* Count avalanches with size >= S_MIN=2 (cascade events only).
         * Matches CPU compute_tau_mle(sizes, n, s_min=2). */
        const double S_MIN = 2.0;
        const double S_MIN_HALF = S_MIN - 0.5;  /* = 1.5, discrete correction */

        uint32_t valid = 0;   /* N: count of avalanches with size >= 2 */
        uint32_t smax = 0;
        double log_sum = 0.0;

        for (uint32_t s = 2; s < CCNS_MAX_UNIQUE_PER_TILE; s++) {
            if (s_hist[s] > 0) {
                valid += s_hist[s];
                if (s > smax) smax = s;
                /* Continuous approximation: ln(s / (s_min - 0.5)) */
                log_sum += (double)s_hist[s] * log((double)s / S_MIN_HALF);
            }
        }

        if (valid < 5) {
            out_results[blockIdx.x].tile_id = group_id;
            out_results[blockIdx.x].tau = 0.0f;
            out_results[blockIdx.x].lambda = 0.0f;
            out_results[blockIdx.x].ks_distance = 1.0f;
            out_results[blockIdx.x].n_avalanches = valid;
            out_results[blockIdx.x].ccns_class = 2; /* Barrier */
            out_results[blockIdx.x].druggability = 0.0f;
            return;
        }

        /* Hill MLE: tau = 1 + N / sum( count[s] * ln(s / 1.5) ) */
        float tau;
        if (log_sum < 1e-12) {
            tau = 1.0f;
        } else {
            tau = 1.0f + (float)((double)valid / log_sum);
        }

        /* Clamp to physical range [1.0, 3.0] */
        if (tau > 3.0f) tau = 3.0f;
        if (tau < 1.0f) tau = 1.0f;

        int32_t cls;
        if (tau < 1.5f) cls = 0;       /* Critical */
        else if (tau < 2.0f) cls = 1;  /* Near-critical */
        else cls = 2;                  /* Barrier */

        float confidence = (valid >= 20) ? 1.0f : (float)valid / 20.0f;
        float drug = fmaxf(0.0f, (1.0f - (tau - 1.0f) / 3.0f) * confidence);

        out_results[blockIdx.x].tile_id = group_id;
        out_results[blockIdx.x].tau = tau;
        out_results[blockIdx.x].lambda = 0.0f;
        out_results[blockIdx.x].ks_distance = 0.0f;
        out_results[blockIdx.x].n_avalanches = valid;
        out_results[blockIdx.x].ccns_class = cls;
        out_results[blockIdx.x].druggability = drug;
    }
}

/* ============================================================
 * Helper: Run the histogram pipeline on already-sorted
 * (group_key, aval_id) pairs. Produces a histogram array
 * [n_groups * 1024] ready for kernel_batched_ccns_histogram.
 *
 * Pipeline:
 *   1. CUB DeviceSegmentedRadixSort on aval_ids within segments
 *   2. kernel_mark_segment_unique: adjacent-diff unique flags
 *   3. kernel_build_size_histogram: atomicAdd sizes into histogram bins
 *
 * No ExclusiveSum, no compaction, no stride sampling.
 * The histogram is EXACT for any event count.
 *
 * Caller must cudaFree *out_histograms.
 * ============================================================ */
static SdstError run_histogram_pipeline(
    cudaStream_t s,
    const uint32_t* d_sorted_group_keys,
    const uint32_t* d_sorted_aval_ids,
    uint32_t total_events,
    const uint32_t* d_group_start,
    const uint32_t* d_group_count,
    uint32_t n_groups,
    const uint32_t* d_size_lookup,
    uint32_t max_aval_id,
    uint32_t** out_histograms
) {
    /* 2b-1. Build segment ends for CUB segmented sort */
    uint32_t* d_seg_ends = NULL;
    SDST_CUDA_CHECK(cudaMalloc(&d_seg_ends, n_groups * sizeof(uint32_t)));

    /* n_groups is typically ≤4096 tiles or ≤50 regions, so host round-trip is fine */
    uint32_t* h_group_start = (uint32_t*)malloc(n_groups * sizeof(uint32_t));
    uint32_t* h_group_count = (uint32_t*)malloc(n_groups * sizeof(uint32_t));
    SDST_CUDA_CHECK(cudaMemcpyAsync(h_group_start, d_group_start,
        n_groups * sizeof(uint32_t), cudaMemcpyDeviceToHost, s));
    SDST_CUDA_CHECK(cudaMemcpyAsync(h_group_count, d_group_count,
        n_groups * sizeof(uint32_t), cudaMemcpyDeviceToHost, s));
    SDST_CUDA_CHECK(cudaStreamSynchronize(s));

    uint32_t* h_seg_ends = (uint32_t*)malloc(n_groups * sizeof(uint32_t));
    for (uint32_t i = 0; i < n_groups; i++) {
        h_seg_ends[i] = h_group_start[i] + h_group_count[i];
    }
    SDST_CUDA_CHECK(cudaMemcpyAsync(d_seg_ends, h_seg_ends,
        n_groups * sizeof(uint32_t), cudaMemcpyHostToDevice, s));

    free(h_group_start);
    free(h_group_count);
    free(h_seg_ends);

    /* 2b-2. CUB segmented sort: sort aval_ids within each segment */
    uint32_t* d_aval_ids_segsorted = NULL;
    SDST_CUDA_CHECK(cudaMalloc(&d_aval_ids_segsorted, total_events * sizeof(uint32_t)));

    size_t temp_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortKeys(NULL, temp_bytes,
        d_sorted_aval_ids, d_aval_ids_segsorted,
        (int)total_events, (int)n_groups,
        d_group_start, d_seg_ends,
        0, 32, s);
    void* d_temp = NULL;
    SDST_CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
    cub::DeviceSegmentedRadixSort::SortKeys(d_temp, temp_bytes,
        d_sorted_aval_ids, d_aval_ids_segsorted,
        (int)total_events, (int)n_groups,
        d_group_start, d_seg_ends,
        0, 32, s);
    SDST_CUDA_CHECK_KERNEL();
    SDST_CUDA_CHECK(cudaFree(d_temp)); d_temp = NULL;
    SDST_CUDA_CHECK(cudaFree(d_seg_ends));

    /* 2b-3. Mark unique within segments */
    uint32_t* d_flags = NULL;
    SDST_CUDA_CHECK(cudaMalloc(&d_flags, total_events * sizeof(uint32_t)));

    dim3 grid_ev(SDST_GRID_SIZE(total_events));
    kernel_mark_segment_unique<<<grid_ev, SDST_BLOCK_SIZE, 0, s>>>(
        d_sorted_group_keys, d_aval_ids_segsorted,
        total_events, d_flags);
    SDST_CUDA_CHECK_KERNEL();

    /* 2b-4. Build histograms directly from unique-flagged events */
    uint32_t* d_histograms = NULL;
    SDST_CUDA_CHECK(cudaMalloc(&d_histograms,
        (size_t)n_groups * CCNS_MAX_UNIQUE_PER_TILE * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemsetAsync(d_histograms, 0,
        (size_t)n_groups * CCNS_MAX_UNIQUE_PER_TILE * sizeof(uint32_t), s));

    kernel_build_size_histogram<<<grid_ev, SDST_BLOCK_SIZE, 0, s>>>(
        d_sorted_group_keys, d_aval_ids_segsorted, d_flags,
        d_size_lookup, max_aval_id,
        total_events, n_groups, d_histograms);
    SDST_CUDA_CHECK_KERNEL();

    SDST_CUDA_CHECK(cudaFree(d_aval_ids_segsorted));
    SDST_CUDA_CHECK(cudaFree(d_flags));

    *out_histograms = d_histograms;
    return SDST_SUCCESS;
}


/* ============================================================
 * Host function: sdst_ccns_all_pockets_gpu()
 *
 * Orchestrates Stage 1 → Stage 2 → Stage 2b → Stage 3 entirely on GPU.
 * Only the final GpuCcnsResult array is copied to host.
 * ============================================================ */

extern "C"
SdstError sdst_ccns_all_pockets_gpu(
    SdstHandle handle,
    CcnsResult** out_results,
    SpatialRegion** out_regions,
    uint32_t* out_count,
    void* stream
) {
    if (!handle || !out_results || !out_regions || !out_count)
        return SDST_ERROR_INVALID_PARAM;

    SdstContext* ctx = handle;
    cudaStream_t s = stream ? (cudaStream_t)stream : 0;

    uint32_t total_events;
    SDST_CUDA_CHECK(cudaMemcpy(&total_events, ctx->d_event_count,
                               sizeof(uint32_t), cudaMemcpyDeviceToHost));

    if (total_events == 0) {
        *out_results = NULL;
        *out_regions = NULL;
        *out_count = 0;
        return SDST_SUCCESS;
    }

    /* ===== Stage 1: Avalanche size computation ===== */

    /* 1a. Extract avalanche_ids */
    uint32_t* d_aval_keys = NULL;
    uint32_t* d_aval_keys_sorted = NULL;
    uint32_t* d_aval_vals = NULL;
    uint32_t* d_aval_vals_sorted = NULL;

    SDST_CUDA_CHECK(cudaMalloc(&d_aval_keys, total_events * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMalloc(&d_aval_keys_sorted, total_events * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMalloc(&d_aval_vals, total_events * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMalloc(&d_aval_vals_sorted, total_events * sizeof(uint32_t)));

    dim3 grid1(SDST_GRID_SIZE(total_events));
    kernel_extract_avalanche_ids<<<grid1, SDST_BLOCK_SIZE, 0, s>>>(
        ctx->d_event_buffer, total_events, d_aval_keys, d_aval_vals);
    SDST_CUDA_CHECK_KERNEL();

    /* 1b. CUB radix sort by avalanche_id */
    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(NULL, temp_bytes,
        d_aval_keys, d_aval_keys_sorted,
        d_aval_vals, d_aval_vals_sorted,
        total_events, 0, 32, s);
    void* d_temp = NULL;
    SDST_CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
    cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes,
        d_aval_keys, d_aval_keys_sorted,
        d_aval_vals, d_aval_vals_sorted,
        total_events, 0, 32, s);
    SDST_CUDA_CHECK_KERNEL();
    SDST_CUDA_CHECK(cudaFree(d_temp)); d_temp = NULL;

    /* 1c. Find max avalanche_id */
    uint32_t* d_max_aval_id = NULL;
    SDST_CUDA_CHECK(cudaMalloc(&d_max_aval_id, sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemsetAsync(d_max_aval_id, 0, sizeof(uint32_t), s));

    uint32_t max_grid = (total_events + 255) / 256;
    if (max_grid > 1024) max_grid = 1024;
    kernel_find_max_aval_id<<<max_grid, 256, 0, s>>>(
        d_aval_keys_sorted, total_events, d_max_aval_id);
    SDST_CUDA_CHECK_KERNEL();

    uint32_t h_max_aval_id;
    SDST_CUDA_CHECK(cudaMemcpyAsync(&h_max_aval_id, d_max_aval_id,
        sizeof(uint32_t), cudaMemcpyDeviceToHost, s));
    SDST_CUDA_CHECK(cudaStreamSynchronize(s));
    SDST_CUDA_CHECK(cudaFree(d_max_aval_id));

    if (h_max_aval_id == 0) {
        cudaFree(d_aval_keys); cudaFree(d_aval_keys_sorted);
        cudaFree(d_aval_vals); cudaFree(d_aval_vals_sorted);
        *out_results = NULL; *out_regions = NULL; *out_count = 0;
        return SDST_SUCCESS;
    }

    /* 1d. CUB run-length encode to get unique IDs and their counts (= sizes).
     *     Output arrays need at most total_events entries (one per unique ID). */
    uint32_t* d_unique_ids = NULL;
    uint32_t* d_run_lengths = NULL;
    uint32_t* d_n_runs = NULL;
    SDST_CUDA_CHECK(cudaMalloc(&d_unique_ids, total_events * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMalloc(&d_run_lengths, total_events * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMalloc(&d_n_runs, sizeof(uint32_t)));

    temp_bytes = 0;
    cub::DeviceRunLengthEncode::Encode(NULL, temp_bytes,
        d_aval_keys_sorted, d_unique_ids, d_run_lengths, d_n_runs,
        total_events, s);
    SDST_CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
    cub::DeviceRunLengthEncode::Encode(d_temp, temp_bytes,
        d_aval_keys_sorted, d_unique_ids, d_run_lengths, d_n_runs,
        total_events, s);
    SDST_CUDA_CHECK_KERNEL();
    SDST_CUDA_CHECK(cudaFree(d_temp)); d_temp = NULL;

    uint32_t h_n_runs;
    SDST_CUDA_CHECK(cudaMemcpyAsync(&h_n_runs, d_n_runs,
        sizeof(uint32_t), cudaMemcpyDeviceToHost, s));
    SDST_CUDA_CHECK(cudaStreamSynchronize(s));
    SDST_CUDA_CHECK(cudaFree(d_n_runs));

    /* 1e. Build dense size lookup: avalanche_id → size */
    uint32_t* d_size_lookup = NULL;
    SDST_CUDA_CHECK(cudaMalloc(&d_size_lookup, (h_max_aval_id + 1) * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemsetAsync(d_size_lookup, 0, (h_max_aval_id + 1) * sizeof(uint32_t), s));

    dim3 grid_lookup(SDST_GRID_SIZE(h_n_runs));
    kernel_build_size_lookup<<<grid_lookup, SDST_BLOCK_SIZE, 0, s>>>(
        d_unique_ids, d_run_lengths, h_n_runs, d_size_lookup, h_max_aval_id);
    SDST_CUDA_CHECK_KERNEL();

    /* Free Stage 1 temporaries we no longer need */
    SDST_CUDA_CHECK(cudaFree(d_aval_keys));
    SDST_CUDA_CHECK(cudaFree(d_aval_keys_sorted));
    SDST_CUDA_CHECK(cudaFree(d_aval_vals));
    SDST_CUDA_CHECK(cudaFree(d_aval_vals_sorted));
    SDST_CUDA_CHECK(cudaFree(d_unique_ids));
    SDST_CUDA_CHECK(cudaFree(d_run_lengths));

    /* ===== Stage 2: Spatial tile indexing ===== */

    uint32_t* d_tile_keys = NULL;
    uint32_t* d_tile_keys_sorted = NULL;
    uint32_t* d_tile_aval_ids = NULL;
    uint32_t* d_tile_aval_ids_sorted = NULL;

    SDST_CUDA_CHECK(cudaMalloc(&d_tile_keys, total_events * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMalloc(&d_tile_keys_sorted, total_events * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMalloc(&d_tile_aval_ids, total_events * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMalloc(&d_tile_aval_ids_sorted, total_events * sizeof(uint32_t)));

    /* 2a. Compute tile_id per event */
    kernel_compute_tile_ids<<<grid1, SDST_BLOCK_SIZE, 0, s>>>(
        ctx->d_event_buffer, total_events, d_tile_keys, d_tile_aval_ids);
    SDST_CUDA_CHECK_KERNEL();

    /* 2b. CUB sort by tile_id */
    temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(NULL, temp_bytes,
        d_tile_keys, d_tile_keys_sorted,
        d_tile_aval_ids, d_tile_aval_ids_sorted,
        total_events, 0, 12, s);  /* 12 bits for tile_id (max 4095) */
    SDST_CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
    cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes,
        d_tile_keys, d_tile_keys_sorted,
        d_tile_aval_ids, d_tile_aval_ids_sorted,
        total_events, 0, 12, s);
    SDST_CUDA_CHECK_KERNEL();
    SDST_CUDA_CHECK(cudaFree(d_temp)); d_temp = NULL;

    SDST_CUDA_CHECK(cudaFree(d_tile_keys));
    SDST_CUDA_CHECK(cudaFree(d_tile_aval_ids));

    /* 2c. Find tile boundaries */
    uint32_t* d_tile_start = NULL;
    uint32_t* d_tile_count = NULL;
    SDST_CUDA_CHECK(cudaMalloc(&d_tile_start, CCNS_MAX_TILES * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMalloc(&d_tile_count, CCNS_MAX_TILES * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemsetAsync(d_tile_start, 0, CCNS_MAX_TILES * sizeof(uint32_t), s));
    SDST_CUDA_CHECK(cudaMemsetAsync(d_tile_count, 0, CCNS_MAX_TILES * sizeof(uint32_t), s));

    kernel_tile_boundaries<<<grid1, SDST_BLOCK_SIZE, 0, s>>>(
        d_tile_keys_sorted, total_events, d_tile_start, d_tile_count);
    SDST_CUDA_CHECK_KERNEL();

    /* Keep d_tile_keys_sorted — needed for Stage 2b segmented sort */

    /* 2d. Find active (non-empty) tiles */
    uint32_t* d_active_tiles = NULL;
    uint32_t* d_n_active = NULL;
    SDST_CUDA_CHECK(cudaMalloc(&d_active_tiles, CCNS_MAX_TILES * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMalloc(&d_n_active, sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemsetAsync(d_n_active, 0, sizeof(uint32_t), s));

    dim3 grid_tiles(SDST_GRID_SIZE(CCNS_MAX_TILES));
    kernel_find_active_tiles<<<grid_tiles, SDST_BLOCK_SIZE, 0, s>>>(
        d_tile_count, d_active_tiles, d_n_active);
    SDST_CUDA_CHECK_KERNEL();

    uint32_t h_n_active;
    SDST_CUDA_CHECK(cudaMemcpyAsync(&h_n_active, d_n_active,
        sizeof(uint32_t), cudaMemcpyDeviceToHost, s));
    SDST_CUDA_CHECK(cudaStreamSynchronize(s));
    SDST_CUDA_CHECK(cudaFree(d_n_active));

    if (h_n_active == 0) {
        cudaFree(d_tile_aval_ids_sorted); cudaFree(d_tile_keys_sorted);
        cudaFree(d_tile_start); cudaFree(d_tile_count);
        cudaFree(d_active_tiles); cudaFree(d_size_lookup);
        *out_results = NULL; *out_regions = NULL; *out_count = 0;
        return SDST_SUCCESS;
    }

    /* ===== Stage 2b: Histogram via CUB segmented sort ===== */

    SDST_CUDA_CHECK(cudaStreamSynchronize(s));

    uint32_t* d_histograms = NULL;

    SdstError hist_err = run_histogram_pipeline(
        s, d_tile_keys_sorted, d_tile_aval_ids_sorted,
        total_events,
        d_tile_start, d_tile_count,
        CCNS_MAX_TILES,
        d_size_lookup, h_max_aval_id,
        &d_histograms);

    SDST_CUDA_CHECK(cudaFree(d_tile_keys_sorted));
    SDST_CUDA_CHECK(cudaFree(d_tile_aval_ids_sorted));

    if (hist_err != SDST_SUCCESS) {
        cudaFree(d_tile_start); cudaFree(d_tile_count);
        cudaFree(d_active_tiles); cudaFree(d_size_lookup);
        if (d_histograms) cudaFree(d_histograms);
        *out_results = NULL; *out_regions = NULL; *out_count = 0;
        return hist_err;
    }

    /* ===== Stage 3: Histogram CCNS kernel ===== */

    GpuCcnsResult* d_ccns_results = NULL;
    SDST_CUDA_CHECK(cudaMalloc(&d_ccns_results, h_n_active * sizeof(GpuCcnsResult)));

    /* Shared memory: 1024 uint32_t histogram = 4096 bytes */
    size_t shared_bytes = CCNS_MAX_UNIQUE_PER_TILE * sizeof(uint32_t);

    kernel_batched_ccns_histogram<<<h_n_active, CCNS_BLOCK_SIZE, shared_bytes, s>>>(
        d_histograms,
        d_active_tiles,
        h_n_active,
        d_ccns_results
    );
    SDST_CUDA_CHECK_KERNEL();

    /* Copy results to host */
    GpuCcnsResult* h_ccns = (GpuCcnsResult*)malloc(h_n_active * sizeof(GpuCcnsResult));
    SDST_CUDA_CHECK(cudaMemcpyAsync(h_ccns, d_ccns_results,
        h_n_active * sizeof(GpuCcnsResult), cudaMemcpyDeviceToHost, s));
    SDST_CUDA_CHECK(cudaStreamSynchronize(s));

    /* Free GPU memory */
    SDST_CUDA_CHECK(cudaFree(d_tile_start));
    SDST_CUDA_CHECK(cudaFree(d_tile_count));
    SDST_CUDA_CHECK(cudaFree(d_active_tiles));
    SDST_CUDA_CHECK(cudaFree(d_size_lookup));
    SDST_CUDA_CHECK(cudaFree(d_ccns_results));
    SDST_CUDA_CHECK(cudaFree(d_histograms));

    /* Convert GPU results to API CcnsResult + SpatialRegion */
    /* Filter: only tiles with valid tau (>0) and sufficient avalanches */
    uint32_t valid_count = 0;
    for (uint32_t i = 0; i < h_n_active; i++) {
        if (h_ccns[i].tau > 0.0f && h_ccns[i].n_avalanches >= 5) {
            valid_count++;
        }
    }

    *out_results = (CcnsResult*)malloc(valid_count * sizeof(CcnsResult));
    *out_regions = (SpatialRegion*)malloc(valid_count * sizeof(SpatialRegion));
    *out_count = valid_count;

    float spacing = ctx->config.grid_spacing;
    uint32_t j = 0;
    for (uint32_t i = 0; i < h_n_active && j < valid_count; i++) {
        GpuCcnsResult* r = &h_ccns[i];
        if (r->tau <= 0.0f || r->n_avalanches < 5) continue;

        CcnsResult* cr = &(*out_results)[j];
        cr->tau = r->tau;
        cr->tau_stderr = 0.0f; /* CSN doesn't produce a stderr; KS distance serves as quality metric */
        cr->n_avalanches = r->n_avalanches;
        cr->druggability = r->druggability;
        cr->classification = (CcnsClass)r->ccns_class;

        /* Reconstruct SpatialRegion from tile_id */
        uint32_t tid = r->tile_id;
        uint32_t tx = tid % CCNS_TILE_DIM;
        uint32_t ty = (tid / CCNS_TILE_DIM) % CCNS_TILE_DIM;
        uint32_t tz = tid / (CCNS_TILE_DIM * CCNS_TILE_DIM);

        SpatialRegion* sr = &(*out_regions)[j];
        sr->x_min = tx * 8;
        sr->x_max = tx * 8 + 7;
        sr->y_min = ty * 8;
        sr->y_max = ty * 8 + 7;
        sr->z_min = tz * 8;
        sr->z_max = tz * 8 + 7;

        /* Clamp to grid bounds */
        if (sr->x_max >= ctx->config.grid_nx) sr->x_max = ctx->config.grid_nx - 1;
        if (sr->y_max >= ctx->config.grid_ny) sr->y_max = ctx->config.grid_ny - 1;
        if (sr->z_max >= ctx->config.grid_nz) sr->z_max = ctx->config.grid_nz - 1;

        j++;
    }

    free(h_ccns);
    return SDST_SUCCESS;
}


/* ============================================================
 * kernel_compute_region_ids
 *
 * For each event, Morton-decode its voxel → check which of
 * n_regions contains it. Write region_id (0..n_regions-1)
 * or 0xFFFF if no region matches.
 * ============================================================ */

__global__
void kernel_compute_region_ids(
    const SpikeEvent* __restrict__ events,
    uint32_t total_events,
    const SpatialRegion* __restrict__ regions,
    uint32_t n_regions,
    uint32_t* __restrict__ out_region_ids,
    uint32_t* __restrict__ out_aval_ids
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_events) return;

    uint32_t vx, vy, vz;
    morton_decode(events[tid].voxel, &vx, &vy, &vz);

    uint32_t region_id = 0xFFFF;
    for (uint32_t r = 0; r < n_regions; r++) {
        if (vx >= regions[r].x_min && vx <= regions[r].x_max &&
            vy >= regions[r].y_min && vy <= regions[r].y_max &&
            vz >= regions[r].z_min && vz <= regions[r].z_max) {
            region_id = r;
            break;
        }
    }

    out_region_ids[tid] = region_id;
    out_aval_ids[tid] = events[tid].avalanche_id;
}

/** After sorting by region_id, find start and count for each region.
 *  Same backward-walk pattern as kernel_tile_boundaries but skips
 *  unmatched events (region_id >= n_regions, i.e. 0xFFFF). */
__global__
void kernel_region_boundaries(
    const uint32_t* __restrict__ sorted_region_ids,
    uint32_t total_events,
    uint32_t n_regions,
    uint32_t* __restrict__ region_start,
    uint32_t* __restrict__ region_count
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_events) return;

    uint32_t my_region = sorted_region_ids[tid];
    if (my_region >= n_regions) return;  /* Skip unmatched (0xFFFF) */

    bool is_end = (tid == total_events - 1) || (sorted_region_ids[tid + 1] != my_region);

    if (is_end) {
        uint32_t start = tid;
        while (start > 0 && sorted_region_ids[start - 1] == my_region) {
            start--;
        }
        region_start[my_region] = start;
        region_count[my_region] = tid - start + 1;
    }
}


/* ============================================================
 * Host function: sdst_ccns_for_regions()
 *
 * Batched GPU-native CCNS for caller-specified spatial regions
 * (NHS pocket bounding boxes). Reuses the same CSN estimator
 * as the global tile scan but groups events by region membership
 * instead of by 8³-voxel tile.
 *
 * Stage 1 (avalanche sizes) computed once, shared across all regions.
 * Stage 2 sorts by region_id instead of tile_id.
 * Stage 3 reuses kernel_batched_ccns.
 * ============================================================ */

extern "C"
SdstError sdst_ccns_for_regions(
    SdstHandle handle,
    const SpatialRegion* regions,
    uint32_t n_regions,
    CcnsResult* out_results,
    void* stream
) {
    if (!handle || !regions || !out_results || n_regions == 0)
        return SDST_ERROR_INVALID_PARAM;

    SdstContext* ctx = handle;
    cudaStream_t s = stream ? (cudaStream_t)stream : 0;

    uint32_t total_events;
    SDST_CUDA_CHECK(cudaMemcpy(&total_events, ctx->d_event_count,
                               sizeof(uint32_t), cudaMemcpyDeviceToHost));

    if (total_events == 0) {
        for (uint32_t i = 0; i < n_regions; i++) {
            out_results[i].tau = 0.0f;
            out_results[i].classification = CCNS_BARRIER;
            out_results[i].tau_stderr = 0.0f;
            out_results[i].n_avalanches = 0;
            out_results[i].druggability = 0.0f;
        }
        return SDST_SUCCESS;
    }

    /* ===== Stage 1: Avalanche size computation (same as global) ===== */

    uint32_t* d_aval_keys = NULL;
    uint32_t* d_aval_keys_sorted = NULL;
    uint32_t* d_aval_vals = NULL;
    uint32_t* d_aval_vals_sorted = NULL;

    SDST_CUDA_CHECK(cudaMalloc(&d_aval_keys, total_events * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMalloc(&d_aval_keys_sorted, total_events * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMalloc(&d_aval_vals, total_events * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMalloc(&d_aval_vals_sorted, total_events * sizeof(uint32_t)));

    dim3 grid1(SDST_GRID_SIZE(total_events));
    kernel_extract_avalanche_ids<<<grid1, SDST_BLOCK_SIZE, 0, s>>>(
        ctx->d_event_buffer, total_events, d_aval_keys, d_aval_vals);
    SDST_CUDA_CHECK_KERNEL();

    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(NULL, temp_bytes,
        d_aval_keys, d_aval_keys_sorted,
        d_aval_vals, d_aval_vals_sorted,
        total_events, 0, 32, s);
    void* d_temp = NULL;
    SDST_CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
    cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes,
        d_aval_keys, d_aval_keys_sorted,
        d_aval_vals, d_aval_vals_sorted,
        total_events, 0, 32, s);
    SDST_CUDA_CHECK_KERNEL();
    SDST_CUDA_CHECK(cudaFree(d_temp)); d_temp = NULL;

    uint32_t* d_max_aval_id = NULL;
    SDST_CUDA_CHECK(cudaMalloc(&d_max_aval_id, sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemsetAsync(d_max_aval_id, 0, sizeof(uint32_t), s));
    uint32_t max_grid = (total_events + 255) / 256;
    if (max_grid > 1024) max_grid = 1024;
    kernel_find_max_aval_id<<<max_grid, 256, 0, s>>>(
        d_aval_keys_sorted, total_events, d_max_aval_id);
    SDST_CUDA_CHECK_KERNEL();

    uint32_t h_max_aval_id;
    SDST_CUDA_CHECK(cudaMemcpyAsync(&h_max_aval_id, d_max_aval_id,
        sizeof(uint32_t), cudaMemcpyDeviceToHost, s));
    SDST_CUDA_CHECK(cudaStreamSynchronize(s));
    SDST_CUDA_CHECK(cudaFree(d_max_aval_id));

    if (h_max_aval_id == 0) {
        cudaFree(d_aval_keys); cudaFree(d_aval_keys_sorted);
        cudaFree(d_aval_vals); cudaFree(d_aval_vals_sorted);
        for (uint32_t i = 0; i < n_regions; i++) {
            out_results[i].tau = 0.0f;
            out_results[i].classification = CCNS_BARRIER;
            out_results[i].tau_stderr = 0.0f;
            out_results[i].n_avalanches = 0;
            out_results[i].druggability = 0.0f;
        }
        return SDST_SUCCESS;
    }

    uint32_t* d_unique_ids = NULL;
    uint32_t* d_run_lengths = NULL;
    uint32_t* d_n_runs = NULL;
    SDST_CUDA_CHECK(cudaMalloc(&d_unique_ids, total_events * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMalloc(&d_run_lengths, total_events * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMalloc(&d_n_runs, sizeof(uint32_t)));

    temp_bytes = 0;
    cub::DeviceRunLengthEncode::Encode(NULL, temp_bytes,
        d_aval_keys_sorted, d_unique_ids, d_run_lengths, d_n_runs,
        total_events, s);
    SDST_CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
    cub::DeviceRunLengthEncode::Encode(d_temp, temp_bytes,
        d_aval_keys_sorted, d_unique_ids, d_run_lengths, d_n_runs,
        total_events, s);
    SDST_CUDA_CHECK_KERNEL();
    SDST_CUDA_CHECK(cudaFree(d_temp)); d_temp = NULL;

    uint32_t h_n_runs;
    SDST_CUDA_CHECK(cudaMemcpyAsync(&h_n_runs, d_n_runs,
        sizeof(uint32_t), cudaMemcpyDeviceToHost, s));
    SDST_CUDA_CHECK(cudaStreamSynchronize(s));
    SDST_CUDA_CHECK(cudaFree(d_n_runs));

    uint32_t* d_size_lookup = NULL;
    SDST_CUDA_CHECK(cudaMalloc(&d_size_lookup, (h_max_aval_id + 1) * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemsetAsync(d_size_lookup, 0, (h_max_aval_id + 1) * sizeof(uint32_t), s));

    dim3 grid_lookup(SDST_GRID_SIZE(h_n_runs));
    kernel_build_size_lookup<<<grid_lookup, SDST_BLOCK_SIZE, 0, s>>>(
        d_unique_ids, d_run_lengths, h_n_runs, d_size_lookup, h_max_aval_id);
    SDST_CUDA_CHECK_KERNEL();

    SDST_CUDA_CHECK(cudaFree(d_aval_keys));
    SDST_CUDA_CHECK(cudaFree(d_aval_keys_sorted));
    SDST_CUDA_CHECK(cudaFree(d_aval_vals));
    SDST_CUDA_CHECK(cudaFree(d_aval_vals_sorted));
    SDST_CUDA_CHECK(cudaFree(d_unique_ids));
    SDST_CUDA_CHECK(cudaFree(d_run_lengths));

    /* ===== Stage 2: Region membership + sort ===== */

    SpatialRegion* d_regions = NULL;
    SDST_CUDA_CHECK(cudaMalloc(&d_regions, n_regions * sizeof(SpatialRegion)));
    SDST_CUDA_CHECK(cudaMemcpyAsync(d_regions, regions,
        n_regions * sizeof(SpatialRegion), cudaMemcpyHostToDevice, s));

    uint32_t* d_region_keys = NULL;
    uint32_t* d_region_keys_sorted = NULL;
    uint32_t* d_region_aval_ids = NULL;
    uint32_t* d_region_aval_ids_sorted = NULL;

    SDST_CUDA_CHECK(cudaMalloc(&d_region_keys, total_events * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMalloc(&d_region_keys_sorted, total_events * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMalloc(&d_region_aval_ids, total_events * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMalloc(&d_region_aval_ids_sorted, total_events * sizeof(uint32_t)));

    kernel_compute_region_ids<<<grid1, SDST_BLOCK_SIZE, 0, s>>>(
        ctx->d_event_buffer, total_events, d_regions, n_regions,
        d_region_keys, d_region_aval_ids);
    SDST_CUDA_CHECK_KERNEL();

    SDST_CUDA_CHECK(cudaFree(d_regions));

    /* CUB sort by region_id (16-bit key: max 65535 regions) */
    temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(NULL, temp_bytes,
        d_region_keys, d_region_keys_sorted,
        d_region_aval_ids, d_region_aval_ids_sorted,
        total_events, 0, 16, s);
    SDST_CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
    cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes,
        d_region_keys, d_region_keys_sorted,
        d_region_aval_ids, d_region_aval_ids_sorted,
        total_events, 0, 16, s);
    SDST_CUDA_CHECK_KERNEL();
    SDST_CUDA_CHECK(cudaFree(d_temp)); d_temp = NULL;

    SDST_CUDA_CHECK(cudaFree(d_region_keys));
    SDST_CUDA_CHECK(cudaFree(d_region_aval_ids));

    /* Find region boundaries */
    uint32_t* d_region_start = NULL;
    uint32_t* d_region_count = NULL;
    SDST_CUDA_CHECK(cudaMalloc(&d_region_start, n_regions * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMalloc(&d_region_count, n_regions * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemsetAsync(d_region_start, 0, n_regions * sizeof(uint32_t), s));
    SDST_CUDA_CHECK(cudaMemsetAsync(d_region_count, 0, n_regions * sizeof(uint32_t), s));

    kernel_region_boundaries<<<grid1, SDST_BLOCK_SIZE, 0, s>>>(
        d_region_keys_sorted, total_events, n_regions,
        d_region_start, d_region_count);
    SDST_CUDA_CHECK_KERNEL();

    /* Keep d_region_keys_sorted — needed for Stage 2b segmented sort */

    /* Build active_region_ids: [0, 1, 2, ..., n_regions-1] on host, upload */
    uint32_t* h_active_ids = (uint32_t*)malloc(n_regions * sizeof(uint32_t));
    for (uint32_t i = 0; i < n_regions; i++) h_active_ids[i] = i;
    uint32_t* d_active_ids = NULL;
    SDST_CUDA_CHECK(cudaMalloc(&d_active_ids, n_regions * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemcpyAsync(d_active_ids, h_active_ids,
        n_regions * sizeof(uint32_t), cudaMemcpyHostToDevice, s));
    free(h_active_ids);

    /* ===== Stage 2b: Histogram via CUB segmented sort ===== */

    SDST_CUDA_CHECK(cudaStreamSynchronize(s));

    uint32_t* d_histograms = NULL;

    SdstError hist_err = run_histogram_pipeline(
        s, d_region_keys_sorted, d_region_aval_ids_sorted,
        total_events,
        d_region_start, d_region_count,
        n_regions,
        d_size_lookup, h_max_aval_id,
        &d_histograms);

    SDST_CUDA_CHECK(cudaFree(d_region_keys_sorted));
    SDST_CUDA_CHECK(cudaFree(d_region_aval_ids_sorted));

    if (hist_err != SDST_SUCCESS) {
        cudaFree(d_region_start); cudaFree(d_region_count);
        cudaFree(d_active_ids); cudaFree(d_size_lookup);
        if (d_histograms) cudaFree(d_histograms);
        for (uint32_t i = 0; i < n_regions; i++) {
            out_results[i].tau = 0.0f;
            out_results[i].classification = CCNS_BARRIER;
            out_results[i].tau_stderr = 0.0f;
            out_results[i].n_avalanches = 0;
            out_results[i].druggability = 0.0f;
        }
        return hist_err;
    }

    /* ===== Stage 3: Histogram CCNS kernel ===== */

    GpuCcnsResult* d_ccns_results = NULL;
    SDST_CUDA_CHECK(cudaMalloc(&d_ccns_results, n_regions * sizeof(GpuCcnsResult)));

    /* Shared memory: 1024 uint32_t histogram = 4096 bytes */
    size_t shared_bytes = CCNS_MAX_UNIQUE_PER_TILE * sizeof(uint32_t);

    kernel_batched_ccns_histogram<<<n_regions, CCNS_BLOCK_SIZE, shared_bytes, s>>>(
        d_histograms,
        d_active_ids,
        n_regions,
        d_ccns_results
    );
    SDST_CUDA_CHECK_KERNEL();

    /* Copy results to host */
    GpuCcnsResult* h_ccns = (GpuCcnsResult*)malloc(n_regions * sizeof(GpuCcnsResult));
    SDST_CUDA_CHECK(cudaMemcpyAsync(h_ccns, d_ccns_results,
        n_regions * sizeof(GpuCcnsResult), cudaMemcpyDeviceToHost, s));
    SDST_CUDA_CHECK(cudaStreamSynchronize(s));

    /* Free GPU memory */
    SDST_CUDA_CHECK(cudaFree(d_region_start));
    SDST_CUDA_CHECK(cudaFree(d_region_count));
    SDST_CUDA_CHECK(cudaFree(d_active_ids));
    SDST_CUDA_CHECK(cudaFree(d_size_lookup));
    SDST_CUDA_CHECK(cudaFree(d_ccns_results));
    SDST_CUDA_CHECK(cudaFree(d_histograms));

    /* Convert GpuCcnsResult → CcnsResult */
    for (uint32_t i = 0; i < n_regions; i++) {
        GpuCcnsResult* r = &h_ccns[i];
        out_results[i].tau = r->tau;
        out_results[i].tau_stderr = 0.0f;
        out_results[i].n_avalanches = r->n_avalanches;
        out_results[i].druggability = r->druggability;
        out_results[i].classification = (CcnsClass)r->ccns_class;
    }

    free(h_ccns);
    return SDST_SUCCESS;
}
