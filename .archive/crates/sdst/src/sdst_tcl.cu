/**
 * SDST Thermodynamic Context Layer (TCL)
 *
 * Captures local thermodynamic state at each spike event.
 * Enables hysteresis asymmetry detection - the mechanism that
 * distinguishes real cryptic sites from thermal noise.
 *
 * Key insight: TCL data is stored PER SPIKE EVENT, not per voxel per frame.
 * This is O(total_spikes) not O(grid × frames), giving ~250x memory savings.
 */

#include "../include/sdst_internal.h"
#include <float.h>

/* ============================================================
 * Kernel: Compute TCL flags for spike events
 *
 * Flags indicate thermodynamic significance:
 *   bit 0: is_transition    - spike occurs near a phase boundary
 *   bit 1: is_boundary      - voxel is at protein-solvent boundary
 *   bit 2: high_gradient    - energy gradient above threshold
 *   bit 3: cooling_spike    - occurs during cooling phase (3 or 4)
 *   bit 4: heating_spike    - occurs during heating phase (0 or 1)
 *   bit 5: peak_temp        - occurs at temperature peak (phase 2)
 *   bit 6: hysteresis_candidate - different behavior heating vs cooling
 * ============================================================ */

__global__
void kernel_compute_tcl_flags(
    SpikeEvent*     event_buffer,
    uint32_t        base_idx,
    uint32_t        n_events,
    const uint32_t* phase_boundaries, /* [6] */
    float           gradient_threshold
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_events) return;

    uint32_t idx = base_idx + tid;
    SpikeEvent* ev = &event_buffer[idx];
    uint8_t flags = 0;

    uint32_t ts = ev->timestamp;
    PhaseId phase = ev->phase_id;
    float grad = f16_bits_to_float(ev->energy_gradient);

    /* Check proximity to phase boundaries (within 500 steps) */
    for (int i = 1; i <= 5; i++) {
        uint32_t boundary = phase_boundaries[i];
        if (ts > boundary - 500 && ts < boundary + 500) {
            flags |= (1 << 0); /* is_transition */
            break;
        }
    }

    /* Solvent boundary: SASA proxy > 0.3 indicates surface exposure */
    float sasa = f16_bits_to_float(ev->solvent_exposure);
    if (sasa > 0.3f) {
        flags |= (1 << 1); /* is_boundary */
    }

    /* High energy gradient */
    if (grad > gradient_threshold) {
        flags |= (1 << 2); /* high_gradient */
    }

    /* Phase classification */
    if (phase == 3 || phase == 4) {
        flags |= (1 << 3); /* cooling_spike */
    }
    if (phase == 0 || phase == 1) {
        flags |= (1 << 4); /* heating_spike */
    }
    if (phase == 2) {
        flags |= (1 << 5); /* peak_temp */
    }

    ev->tcl_flags = flags;
}

/* ============================================================
 * Host: Launch TCL flag computation for a batch of new events
 *
 * Called from sdst_insert_spikes after temporal index update.
 * Uses the device-side phase_boundaries array stored in SdstContext.
 * ============================================================ */

extern "C"
SdstError sdst_compute_tcl_flags(
    SdstHandle handle,
    uint32_t base_idx,
    uint32_t n_events,
    void* stream
) {
    if (!handle) return SDST_ERROR_INVALID_PARAM;
    SdstContext* ctx = handle;
    cudaStream_t s = stream ? (cudaStream_t)stream : 0;

    /* gradient_threshold: flag spikes with |∇E| > 0.5 (f16 ~= moderate energy barrier) */
    float gradient_threshold = 0.5f;

    dim3 grid(SDST_GRID_SIZE(n_events));
    dim3 block(SDST_BLOCK_SIZE);

    kernel_compute_tcl_flags<<<grid, block, 0, s>>>(
        ctx->d_event_buffer,
        base_idx,
        n_events,
        ctx->d_phase_boundaries,
        gradient_threshold
    );
    SDST_CUDA_CHECK_KERNEL();

    return SDST_SUCCESS;
}

/* ============================================================
 * Kernel: Hysteresis asymmetry detection for a spatial region
 *
 * Compares heating phase activity (phases 0,1) vs cooling (3,4).
 * Asymmetry indicates a cryptic site with conformational memory.
 * ============================================================ */

__global__
void kernel_hysteresis_region(
    const SpikeEvent* event_buffer,
    uint32_t          total_events,
    SpatialRegion     region,
    float             grid_spacing,
    /* Output: atomically accumulated */
    uint32_t*         heating_count,
    uint32_t*         cooling_count,
    float*            heating_amp_sum,
    float*            cooling_amp_sum,
    uint32_t*         heating_avalanche_sizes, /* histogram bins */
    uint32_t*         cooling_avalanche_sizes,
    uint32_t          n_hist_bins
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_events) return;

    SpikeEvent ev = event_buffer[tid];

    /* Decode Morton and check spatial region */
    uint32_t vx, vy, vz;
    morton_decode(ev.voxel, &vx, &vy, &vz);

    if (vx < region.x_min || vx > region.x_max) return;
    if (vy < region.y_min || vy > region.y_max) return;
    if (vz < region.z_min || vz > region.z_max) return;

    float amp = f16_bits_to_float(ev.amplitude);
    PhaseId phase = ev.phase_id;

    if (phase == 0 || phase == 1) {
        /* Heating phase */
        atomicAdd(heating_count, 1);
        atomicAdd(heating_amp_sum, amp);
    } else if (phase == 3 || phase == 4) {
        /* Cooling phase */
        atomicAdd(cooling_count, 1);
        atomicAdd(cooling_amp_sum, amp);
    }
    /* Phase 2 (peak) excluded from asymmetry calculation */
}

/* ============================================================
 * Host: Compute hysteresis result for a region
 * ============================================================ */

extern "C"
SdstError sdst_hysteresis_region(
    SdstHandle handle,
    const SpatialRegion* region,
    float asymmetry_threshold,
    HysteresisResult* out_result,
    void* stream
) {
    if (!handle || !region || !out_result) return SDST_ERROR_INVALID_PARAM;
    SdstContext* ctx = handle;
    cudaStream_t s = stream ? (cudaStream_t)stream : 0;

    uint32_t total_events;
    SDST_CUDA_CHECK(cudaMemcpy(&total_events, ctx->d_event_count,
                               sizeof(uint32_t), cudaMemcpyDeviceToHost));

    if (total_events == 0) {
        memset(out_result, 0, sizeof(HysteresisResult));
        return SDST_SUCCESS;
    }

    /* Allocate device accumulators */
    uint32_t *d_hcount, *d_ccount;
    float *d_hamp, *d_camp;
    SDST_CUDA_CHECK(cudaMallocAsync(&d_hcount, sizeof(uint32_t), s));
    SDST_CUDA_CHECK(cudaMallocAsync(&d_ccount, sizeof(uint32_t), s));
    SDST_CUDA_CHECK(cudaMallocAsync(&d_hamp, sizeof(float), s));
    SDST_CUDA_CHECK(cudaMallocAsync(&d_camp, sizeof(float), s));
    SDST_CUDA_CHECK(cudaMemsetAsync(d_hcount, 0, sizeof(uint32_t), s));
    SDST_CUDA_CHECK(cudaMemsetAsync(d_ccount, 0, sizeof(uint32_t), s));
    SDST_CUDA_CHECK(cudaMemsetAsync(d_hamp, 0, sizeof(float), s));
    SDST_CUDA_CHECK(cudaMemsetAsync(d_camp, 0, sizeof(float), s));

    dim3 grid(SDST_GRID_SIZE(total_events));
    dim3 block(SDST_BLOCK_SIZE);

    kernel_hysteresis_region<<<grid, block, 0, s>>>(
        ctx->d_event_buffer, total_events, *region, ctx->config.grid_spacing,
        d_hcount, d_ccount, d_hamp, d_camp,
        NULL, NULL, 0 /* avalanche histograms - TODO */
    );
    SDST_CUDA_CHECK_KERNEL();

    /* Copy results back */
    uint32_t h_hcount, h_ccount;
    float h_hamp, h_camp;
    SDST_CUDA_CHECK(cudaMemcpyAsync(&h_hcount, d_hcount, sizeof(uint32_t), cudaMemcpyDeviceToHost, s));
    SDST_CUDA_CHECK(cudaMemcpyAsync(&h_ccount, d_ccount, sizeof(uint32_t), cudaMemcpyDeviceToHost, s));
    SDST_CUDA_CHECK(cudaMemcpyAsync(&h_hamp, d_hamp, sizeof(float), cudaMemcpyDeviceToHost, s));
    SDST_CUDA_CHECK(cudaMemcpyAsync(&h_camp, d_camp, sizeof(float), cudaMemcpyDeviceToHost, s));
    SDST_CUDA_CHECK(cudaStreamSynchronize(s));

    /* Compute asymmetry metrics */
    float total = (float)(h_hcount + h_ccount);
    float heating_steps = (float)(ctx->config.phase_boundaries[2] - ctx->config.phase_boundaries[0]);
    float cooling_steps = (float)(ctx->config.phase_boundaries[5] - ctx->config.phase_boundaries[3]);

    out_result->heating_spike_count = h_hcount;
    out_result->cooling_spike_count = h_ccount;
    out_result->heating_spike_rate = (heating_steps > 0) ? (float)h_hcount / heating_steps : 0;
    out_result->cooling_spike_rate = (cooling_steps > 0) ? (float)h_ccount / cooling_steps : 0;

    if (total > 0) {
        out_result->asymmetry_score = fabsf((float)h_hcount - (float)h_ccount) / total;
    } else {
        out_result->asymmetry_score = 0;
    }

    /* Compute avalanche_size_ratio and wavefront_coherence_ratio from events */
    {
        SpikeEvent* h_events = (SpikeEvent*)malloc(total_events * sizeof(SpikeEvent));
        SDST_CUDA_CHECK(cudaMemcpy(h_events, ctx->d_event_buffer,
                                   total_events * sizeof(SpikeEvent),
                                   cudaMemcpyDeviceToHost));

        /* Find max avalanche_id among region events */
        uint32_t max_aid = 0;
        for (uint32_t i = 0; i < total_events; i++) {
            uint32_t vx, vy, vz;
            morton_decode(h_events[i].voxel, &vx, &vy, &vz);
            if (vx < region->x_min || vx > region->x_max) continue;
            if (vy < region->y_min || vy > region->y_max) continue;
            if (vz < region->z_min || vz > region->z_max) continue;
            if (h_events[i].avalanche_id > max_aid) max_aid = h_events[i].avalanche_id;
        }

        if (max_aid > 0) {
            /* Per-avalanche: size, phase, wavefront coherence sum */
            uint32_t* a_sizes = (uint32_t*)calloc(max_aid + 1, sizeof(uint32_t));
            PhaseId*  a_phase = (PhaseId*)calloc(max_aid + 1, sizeof(PhaseId));
            bool*     a_seen  = (bool*)calloc(max_aid + 1, sizeof(bool));
            float*    a_wf_sum = (float*)calloc(max_aid + 1, sizeof(float));

            for (uint32_t i = 0; i < total_events; i++) {
                uint32_t vx, vy, vz;
                morton_decode(h_events[i].voxel, &vx, &vy, &vz);
                if (vx < region->x_min || vx > region->x_max) continue;
                if (vy < region->y_min || vy > region->y_max) continue;
                if (vz < region->z_min || vz > region->z_max) continue;

                AvalancheId aid = h_events[i].avalanche_id;
                if (aid == 0) continue;
                a_sizes[aid]++;
                if (!a_seen[aid]) {
                    a_phase[aid] = h_events[i].phase_id;
                    a_seen[aid] = true;
                }
                a_wf_sum[aid] += f16_bits_to_float(h_events[i].amplitude);
            }

            /* Accumulate mean sizes and coherence for heating vs cooling */
            float heat_size_sum = 0, cool_size_sum = 0;
            float heat_wf_sum = 0, cool_wf_sum = 0;
            uint32_t n_heat_av = 0, n_cool_av = 0;

            for (uint32_t i = 1; i <= max_aid; i++) {
                if (!a_seen[i] || a_sizes[i] <= 1) continue;
                PhaseId ph = a_phase[i];
                float mean_amp = a_wf_sum[i] / (float)a_sizes[i];
                if (ph == 0 || ph == 1) {
                    heat_size_sum += (float)a_sizes[i];
                    heat_wf_sum += mean_amp;
                    n_heat_av++;
                } else if (ph == 3 || ph == 4) {
                    cool_size_sum += (float)a_sizes[i];
                    cool_wf_sum += mean_amp;
                    n_cool_av++;
                }
            }

            float mean_heat_size = (n_heat_av > 0) ? heat_size_sum / n_heat_av : 0;
            float mean_cool_size = (n_cool_av > 0) ? cool_size_sum / n_cool_av : 0;
            float mean_heat_wf   = (n_heat_av > 0) ? heat_wf_sum / n_heat_av : 0;
            float mean_cool_wf   = (n_cool_av > 0) ? cool_wf_sum / n_cool_av : 0;

            out_result->avalanche_size_ratio =
                (mean_cool_size > 0) ? mean_heat_size / mean_cool_size : 1.0f;
            out_result->wavefront_coherence_ratio =
                (mean_cool_wf > 0) ? mean_heat_wf / mean_cool_wf : 1.0f;

            free(a_sizes); free(a_phase); free(a_seen); free(a_wf_sum);
        } else {
            out_result->avalanche_size_ratio = 1.0f;
            out_result->wavefront_coherence_ratio = 1.0f;
        }
        free(h_events);
    }

    out_result->is_hysteretic = (out_result->asymmetry_score > asymmetry_threshold);

    cudaFreeAsync(d_hcount, s);
    cudaFreeAsync(d_ccount, s);
    cudaFreeAsync(d_hamp, s);
    cudaFreeAsync(d_camp, s);

    return SDST_SUCCESS;
}

/* ============================================================
 * Kernel: Full-grid hysteresis scan
 *
 * Tiles the grid into blocks and computes asymmetry per block.
 * High-asymmetry blocks are candidate cryptic sites.
 * ============================================================ */

#define SCAN_TILE_SIZE 8 /* 8³ voxel tiles = 6Å cubes at 0.75Å */

__global__
void kernel_hysteresis_scan_tiles(
    const SpikeEvent* event_buffer,
    uint32_t          total_events,
    uint32_t          grid_nx, uint32_t grid_ny, uint32_t grid_nz,
    uint32_t          tile_size,
    /* Per-tile outputs */
    float*            tile_asymmetry,  /* [n_tiles] */
    uint32_t*         tile_heating,    /* [n_tiles] */
    uint32_t*         tile_cooling,    /* [n_tiles] */
    uint32_t          tiles_x, uint32_t tiles_y, uint32_t tiles_z
) {
    uint32_t eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= total_events) return;

    SpikeEvent ev = event_buffer[eid];
    uint32_t vx, vy, vz;
    morton_decode(ev.voxel, &vx, &vy, &vz);

    /* Determine which tile this event belongs to */
    uint32_t tx = vx / tile_size;
    uint32_t ty = vy / tile_size;
    uint32_t tz = vz / tile_size;

    if (tx >= tiles_x || ty >= tiles_y || tz >= tiles_z) return;

    uint32_t tile_idx = tx + ty * tiles_x + tz * tiles_x * tiles_y;
    PhaseId phase = ev.phase_id;

    if (phase == 0 || phase == 1) {
        atomicAdd(&tile_heating[tile_idx], 1);
    } else if (phase == 3 || phase == 4) {
        atomicAdd(&tile_cooling[tile_idx], 1);
    }
}

__global__
void kernel_compute_tile_asymmetry(
    float*    tile_asymmetry,
    uint32_t* tile_heating,
    uint32_t* tile_cooling,
    uint32_t  n_tiles,
    float     min_spike_count /* minimum spikes to be considered */
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_tiles) return;

    uint32_t h = tile_heating[tid];
    uint32_t c = tile_cooling[tid];
    float total = (float)(h + c);

    if (total < min_spike_count) {
        tile_asymmetry[tid] = 0.0f;
        return;
    }

    tile_asymmetry[tid] = fabsf((float)h - (float)c) / total;
}

extern "C"
SdstError sdst_hysteresis_scan(
    SdstHandle handle,
    float asymmetry_threshold,
    HysteresisResult** out_results,
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

    uint32_t nx = ctx->config.grid_nx;
    uint32_t ny = ctx->config.grid_ny;
    uint32_t nz = ctx->config.grid_nz;
    uint32_t ts = SCAN_TILE_SIZE;
    uint32_t tiles_x = (nx + ts - 1) / ts;
    uint32_t tiles_y = (ny + ts - 1) / ts;
    uint32_t tiles_z = (nz + ts - 1) / ts;
    uint32_t n_tiles = tiles_x * tiles_y * tiles_z;

    /* Allocate tile buffers */
    float *d_asym;
    uint32_t *d_theat, *d_tcool;
    SDST_CUDA_CHECK(cudaMallocAsync(&d_asym, n_tiles * sizeof(float), s));
    SDST_CUDA_CHECK(cudaMallocAsync(&d_theat, n_tiles * sizeof(uint32_t), s));
    SDST_CUDA_CHECK(cudaMallocAsync(&d_tcool, n_tiles * sizeof(uint32_t), s));
    SDST_CUDA_CHECK(cudaMemsetAsync(d_theat, 0, n_tiles * sizeof(uint32_t), s));
    SDST_CUDA_CHECK(cudaMemsetAsync(d_tcool, 0, n_tiles * sizeof(uint32_t), s));

    /* Pass 1: Accumulate per-tile spike counts */
    dim3 grid1(SDST_GRID_SIZE(total_events));
    dim3 block1(SDST_BLOCK_SIZE);
    kernel_hysteresis_scan_tiles<<<grid1, block1, 0, s>>>(
        ctx->d_event_buffer, total_events,
        nx, ny, nz, ts,
        d_asym, d_theat, d_tcool,
        tiles_x, tiles_y, tiles_z
    );
    SDST_CUDA_CHECK_KERNEL();

    /* Pass 2: Compute asymmetry per tile */
    dim3 grid2(SDST_GRID_SIZE(n_tiles));
    kernel_compute_tile_asymmetry<<<grid2, block1, 0, s>>>(
        d_asym, d_theat, d_tcool, n_tiles,
        10.0f /* minimum 10 spikes in tile to count */
    );
    SDST_CUDA_CHECK_KERNEL();

    /* Copy to host and filter */
    float* h_asym = (float*)malloc(n_tiles * sizeof(float));
    uint32_t* h_heat = (uint32_t*)malloc(n_tiles * sizeof(uint32_t));
    uint32_t* h_cool = (uint32_t*)malloc(n_tiles * sizeof(uint32_t));
    SDST_CUDA_CHECK(cudaMemcpyAsync(h_asym, d_asym, n_tiles * sizeof(float),
                                     cudaMemcpyDeviceToHost, s));
    SDST_CUDA_CHECK(cudaMemcpyAsync(h_heat, d_theat, n_tiles * sizeof(uint32_t),
                                     cudaMemcpyDeviceToHost, s));
    SDST_CUDA_CHECK(cudaMemcpyAsync(h_cool, d_tcool, n_tiles * sizeof(uint32_t),
                                     cudaMemcpyDeviceToHost, s));
    SDST_CUDA_CHECK(cudaStreamSynchronize(s));

    /* Count hits */
    uint32_t hit_count = 0;
    for (uint32_t i = 0; i < n_tiles; i++) {
        if (h_asym[i] > asymmetry_threshold) hit_count++;
    }

    /* Build per-hit-tile avalanche size & coherence ratios from event data */
    /* Map hit tiles to compact indices for accumulation */
    int32_t* tile_to_hit = (int32_t*)malloc(n_tiles * sizeof(int32_t));
    for (uint32_t i = 0; i < n_tiles; i++) tile_to_hit[i] = -1;
    {
        uint32_t idx = 0;
        for (uint32_t i = 0; i < n_tiles; i++) {
            if (h_asym[i] > asymmetry_threshold) tile_to_hit[i] = idx++;
        }
    }

    /* Per-hit-tile avalanche accumulators */
    float* hit_heat_size_sum  = (float*)calloc(hit_count, sizeof(float));
    float* hit_cool_size_sum  = (float*)calloc(hit_count, sizeof(float));
    float* hit_heat_amp_sum   = (float*)calloc(hit_count, sizeof(float));
    float* hit_cool_amp_sum   = (float*)calloc(hit_count, sizeof(float));
    uint32_t* hit_n_heat_av   = (uint32_t*)calloc(hit_count, sizeof(uint32_t));
    uint32_t* hit_n_cool_av   = (uint32_t*)calloc(hit_count, sizeof(uint32_t));

    if (hit_count > 0 && total_events > 0) {
        /* Copy events to host for avalanche analysis */
        SpikeEvent* h_events = (SpikeEvent*)malloc(total_events * sizeof(SpikeEvent));
        SDST_CUDA_CHECK(cudaMemcpy(h_events, ctx->d_event_buffer,
                                   total_events * sizeof(SpikeEvent),
                                   cudaMemcpyDeviceToHost));

        /* Find max avalanche_id */
        uint32_t max_aid = 0;
        for (uint32_t i = 0; i < total_events; i++) {
            if (h_events[i].avalanche_id > max_aid)
                max_aid = h_events[i].avalanche_id;
        }

        if (max_aid > 0) {
            /* Per-avalanche: size, phase, tile, amplitude sum */
            uint32_t* a_sizes  = (uint32_t*)calloc(max_aid + 1, sizeof(uint32_t));
            PhaseId*  a_phase  = (PhaseId*)calloc(max_aid + 1, sizeof(PhaseId));
            int32_t*  a_tile   = (int32_t*)malloc((max_aid + 1) * sizeof(int32_t));
            float*    a_amp    = (float*)calloc(max_aid + 1, sizeof(float));
            bool*     a_seen   = (bool*)calloc(max_aid + 1, sizeof(bool));
            for (uint32_t i = 0; i <= max_aid; i++) a_tile[i] = -1;

            for (uint32_t i = 0; i < total_events; i++) {
                AvalancheId aid = h_events[i].avalanche_id;
                if (aid == 0) continue;
                uint32_t vx, vy, vz;
                morton_decode(h_events[i].voxel, &vx, &vy, &vz);
                uint32_t txi = vx / ts, tyi = vy / ts, tzi = vz / ts;
                if (txi >= tiles_x || tyi >= tiles_y || tzi >= tiles_z) continue;
                uint32_t tidx = txi + tyi * tiles_x + tzi * tiles_x * tiles_y;

                a_sizes[aid]++;
                a_amp[aid] += f16_bits_to_float(h_events[i].amplitude);
                if (!a_seen[aid]) {
                    a_phase[aid] = h_events[i].phase_id;
                    a_tile[aid] = (int32_t)tidx;
                    a_seen[aid] = true;
                }
            }

            /* Distribute avalanche stats to hit tiles */
            for (uint32_t i = 1; i <= max_aid; i++) {
                if (!a_seen[i] || a_sizes[i] <= 1 || a_tile[i] < 0) continue;
                int32_t hit_idx = tile_to_hit[(uint32_t)a_tile[i]];
                if (hit_idx < 0) continue;

                float mean_amp = a_amp[i] / (float)a_sizes[i];
                PhaseId ph = a_phase[i];
                if (ph == 0 || ph == 1) {
                    hit_heat_size_sum[hit_idx] += (float)a_sizes[i];
                    hit_heat_amp_sum[hit_idx]  += mean_amp;
                    hit_n_heat_av[hit_idx]++;
                } else if (ph == 3 || ph == 4) {
                    hit_cool_size_sum[hit_idx] += (float)a_sizes[i];
                    hit_cool_amp_sum[hit_idx]  += mean_amp;
                    hit_n_cool_av[hit_idx]++;
                }
            }

            free(a_sizes); free(a_phase); free(a_tile); free(a_amp); free(a_seen);
        }
        free(h_events);
    }

    /* Build result arrays */
    *out_results = (HysteresisResult*)malloc(hit_count * sizeof(HysteresisResult));
    *out_regions = (SpatialRegion*)malloc(hit_count * sizeof(SpatialRegion));
    *out_count = hit_count;

    float heating_steps = (float)(ctx->config.phase_boundaries[2] - ctx->config.phase_boundaries[0]);
    float cooling_steps = (float)(ctx->config.phase_boundaries[5] - ctx->config.phase_boundaries[3]);

    uint32_t out_idx = 0;
    for (uint32_t i = 0; i < n_tiles; i++) {
        if (h_asym[i] <= asymmetry_threshold) continue;

        uint32_t tz_i = i / (tiles_x * tiles_y);
        uint32_t ty_i = (i - tz_i * tiles_x * tiles_y) / tiles_x;
        uint32_t tx_i = i - tz_i * tiles_x * tiles_y - ty_i * tiles_x;

        SpatialRegion* reg = &(*out_regions)[out_idx];
        reg->x_min = tx_i * ts; reg->x_max = (tx_i + 1) * ts - 1;
        reg->y_min = ty_i * ts; reg->y_max = (ty_i + 1) * ts - 1;
        reg->z_min = tz_i * ts; reg->z_max = (tz_i + 1) * ts - 1;

        /* Clamp to grid */
        if (reg->x_max >= nx) reg->x_max = nx - 1;
        if (reg->y_max >= ny) reg->y_max = ny - 1;
        if (reg->z_max >= nz) reg->z_max = nz - 1;

        HysteresisResult* res = &(*out_results)[out_idx];
        res->heating_spike_count = h_heat[i];
        res->cooling_spike_count = h_cool[i];
        res->heating_spike_rate = (heating_steps > 0) ? (float)h_heat[i] / heating_steps : 0;
        res->cooling_spike_rate = (cooling_steps > 0) ? (float)h_cool[i] / cooling_steps : 0;
        res->asymmetry_score = h_asym[i];

        /* Compute ratios from accumulated avalanche data */
        float mhs = (hit_n_heat_av[out_idx] > 0) ?
                    hit_heat_size_sum[out_idx] / hit_n_heat_av[out_idx] : 0;
        float mcs = (hit_n_cool_av[out_idx] > 0) ?
                    hit_cool_size_sum[out_idx] / hit_n_cool_av[out_idx] : 0;
        float mha = (hit_n_heat_av[out_idx] > 0) ?
                    hit_heat_amp_sum[out_idx] / hit_n_heat_av[out_idx] : 0;
        float mca = (hit_n_cool_av[out_idx] > 0) ?
                    hit_cool_amp_sum[out_idx] / hit_n_cool_av[out_idx] : 0;

        res->avalanche_size_ratio = (mcs > 0) ? mhs / mcs : 1.0f;
        res->wavefront_coherence_ratio = (mca > 0) ? mha / mca : 1.0f;
        res->is_hysteretic = true;

        out_idx++;
    }

    free(tile_to_hit);
    free(hit_heat_size_sum); free(hit_cool_size_sum);
    free(hit_heat_amp_sum);  free(hit_cool_amp_sum);
    free(hit_n_heat_av);     free(hit_n_cool_av);
    free(h_asym); free(h_heat); free(h_cool);
    cudaFreeAsync(d_asym, s);
    cudaFreeAsync(d_theat, s);
    cudaFreeAsync(d_tcool, s);

    return SDST_SUCCESS;
}
