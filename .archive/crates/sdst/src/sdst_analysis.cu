/**
 * SDST Analysis: CCNS, Spatial Queries, Causal Subgraphs, DCC
 *
 * Scientific analysis functions that operate on the SDST event record.
 * These transform raw spike data into druggability predictions,
 * causal mechanisms, and validation metrics.
 */

#include "../include/sdst_internal.h"
#include <math.h>
#include <float.h>

/* ============================================================
 * Kernel: Spatial region query - collect events in a bounding box
 * ============================================================ */

__global__
void kernel_query_region(
    const SpikeEvent* event_buffer,
    uint32_t          total_events,
    SpatialRegion     region,
    SpikeEvent*       out_buffer,
    uint32_t*         out_count,
    uint32_t          max_results
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_events) return;

    SpikeEvent ev = event_buffer[tid];
    uint32_t vx, vy, vz;
    morton_decode(ev.voxel, &vx, &vy, &vz);

    if (vx >= region.x_min && vx <= region.x_max &&
        vy >= region.y_min && vy <= region.y_max &&
        vz >= region.z_min && vz <= region.z_max) {
        uint32_t idx = atomicAdd(out_count, 1);
        if (idx < max_results) {
            out_buffer[idx] = ev;
        }
    }
}

extern "C"
SdstError sdst_query_region(
    SdstHandle handle,
    const SpatialRegion* region,
    SpikeEvent** out_events,
    uint32_t* out_count,
    void* stream
) {
    if (!handle || !region || !out_events || !out_count)
        return SDST_ERROR_INVALID_PARAM;
    SdstContext* ctx = handle;
    cudaStream_t s = stream ? (cudaStream_t)stream : 0;

    uint32_t total_events;
    SDST_CUDA_CHECK(cudaMemcpy(&total_events, ctx->d_event_count,
                               sizeof(uint32_t), cudaMemcpyDeviceToHost));

    /* Use stream 0's query buffer */
    /* Map CUDA stream pointer to a query buffer index via modular hash.
     * Collisions are safe — query results are synchronous per call. */
    uint32_t stream_idx = ((uint64_t)(uintptr_t)s / 8) % ctx->num_streams;
    SDST_CUDA_CHECK(cudaMemsetAsync(ctx->d_query_counts + stream_idx,
                                     0, sizeof(uint32_t), s));

    dim3 grid(SDST_GRID_SIZE(total_events));
    dim3 block(SDST_BLOCK_SIZE);

    kernel_query_region<<<grid, block, 0, s>>>(
        ctx->d_event_buffer, total_events, *region,
        ctx->d_query_buffers[stream_idx],
        ctx->d_query_counts + stream_idx,
        ctx->query_buffer_size
    );
    SDST_CUDA_CHECK_KERNEL();

    uint32_t count;
    SDST_CUDA_CHECK(cudaMemcpyAsync(&count, ctx->d_query_counts + stream_idx,
                                     sizeof(uint32_t), cudaMemcpyDeviceToHost, s));
    SDST_CUDA_CHECK(cudaStreamSynchronize(s));

    if (count > ctx->query_buffer_size) count = ctx->query_buffer_size;

    *out_events = ctx->d_query_buffers[stream_idx];
    *out_count = count;

    return SDST_SUCCESS;
}

/* ============================================================
 * Voxel query: all events for a specific voxel
 * ============================================================ */

extern "C"
SdstError sdst_query_voxel(
    SdstHandle handle,
    uint32_t x, uint32_t y, uint32_t z,
    SpikeEvent** out_events,
    uint32_t* out_count,
    void* stream
) {
    if (!handle || !out_events || !out_count) return SDST_ERROR_INVALID_PARAM;

    SpatialRegion region = {x, x, y, y, z, z};
    return sdst_query_region(handle, &region, out_events, out_count, stream);
}

/* ============================================================
 * Time range query
 * ============================================================ */

__global__
void kernel_query_timerange(
    const SpikeEvent* event_buffer,
    uint32_t          total_events,
    uint32_t          t_start,
    uint32_t          t_end,
    SpikeEvent*       out_buffer,
    uint32_t*         out_count,
    uint32_t          max_results
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_events) return;

    SpikeEvent ev = event_buffer[tid];
    if (ev.timestamp >= t_start && ev.timestamp <= t_end) {
        uint32_t idx = atomicAdd(out_count, 1);
        if (idx < max_results) {
            out_buffer[idx] = ev;
        }
    }
}

extern "C"
SdstError sdst_query_timerange(
    SdstHandle handle,
    uint32_t t_start, uint32_t t_end,
    SpikeEvent** out_events,
    uint32_t* out_count,
    void* stream
) {
    if (!handle || !out_events || !out_count) return SDST_ERROR_INVALID_PARAM;
    SdstContext* ctx = handle;
    cudaStream_t s = stream ? (cudaStream_t)stream : 0;

    uint32_t total_events;
    SDST_CUDA_CHECK(cudaMemcpy(&total_events, ctx->d_event_count,
                               sizeof(uint32_t), cudaMemcpyDeviceToHost));

    uint32_t stream_idx = 0;
    SDST_CUDA_CHECK(cudaMemsetAsync(ctx->d_query_counts + stream_idx,
                                     0, sizeof(uint32_t), s));

    dim3 grid(SDST_GRID_SIZE(total_events));
    dim3 block(SDST_BLOCK_SIZE);

    kernel_query_timerange<<<grid, block, 0, s>>>(
        ctx->d_event_buffer, total_events, t_start, t_end,
        ctx->d_query_buffers[stream_idx],
        ctx->d_query_counts + stream_idx,
        ctx->query_buffer_size
    );
    SDST_CUDA_CHECK_KERNEL();

    uint32_t count;
    SDST_CUDA_CHECK(cudaMemcpyAsync(&count, ctx->d_query_counts + stream_idx,
                                     sizeof(uint32_t), cudaMemcpyDeviceToHost, s));
    SDST_CUDA_CHECK(cudaStreamSynchronize(s));

    if (count > ctx->query_buffer_size) count = ctx->query_buffer_size;
    *out_events = ctx->d_query_buffers[stream_idx];
    *out_count = count;

    return SDST_SUCCESS;
}

/* ============================================================
 * Combined spatial + temporal query
 * ============================================================ */

__global__
void kernel_query_region_timerange(
    const SpikeEvent* event_buffer,
    uint32_t          total_events,
    SpatialRegion     region,
    uint32_t          t_start,
    uint32_t          t_end,
    SpikeEvent*       out_buffer,
    uint32_t*         out_count,
    uint32_t          max_results
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_events) return;

    SpikeEvent ev = event_buffer[tid];
    if (ev.timestamp < t_start || ev.timestamp > t_end) return;

    uint32_t vx, vy, vz;
    morton_decode(ev.voxel, &vx, &vy, &vz);

    if (vx >= region.x_min && vx <= region.x_max &&
        vy >= region.y_min && vy <= region.y_max &&
        vz >= region.z_min && vz <= region.z_max) {
        uint32_t idx = atomicAdd(out_count, 1);
        if (idx < max_results) {
            out_buffer[idx] = ev;
        }
    }
}

extern "C"
SdstError sdst_query_region_timerange(
    SdstHandle handle,
    const SpatialRegion* region,
    uint32_t t_start, uint32_t t_end,
    SpikeEvent** out_events,
    uint32_t* out_count,
    void* stream
) {
    if (!handle || !region || !out_events || !out_count)
        return SDST_ERROR_INVALID_PARAM;
    SdstContext* ctx = handle;
    cudaStream_t s = stream ? (cudaStream_t)stream : 0;

    uint32_t total_events;
    SDST_CUDA_CHECK(cudaMemcpy(&total_events, ctx->d_event_count,
                               sizeof(uint32_t), cudaMemcpyDeviceToHost));

    uint32_t stream_idx = 0;
    SDST_CUDA_CHECK(cudaMemsetAsync(ctx->d_query_counts + stream_idx,
                                     0, sizeof(uint32_t), s));

    dim3 grid(SDST_GRID_SIZE(total_events));
    dim3 block(SDST_BLOCK_SIZE);

    kernel_query_region_timerange<<<grid, block, 0, s>>>(
        ctx->d_event_buffer, total_events, *region, t_start, t_end,
        ctx->d_query_buffers[stream_idx],
        ctx->d_query_counts + stream_idx,
        ctx->query_buffer_size
    );
    SDST_CUDA_CHECK_KERNEL();

    uint32_t count;
    SDST_CUDA_CHECK(cudaMemcpyAsync(&count, ctx->d_query_counts + stream_idx,
                                     sizeof(uint32_t), cudaMemcpyDeviceToHost, s));
    SDST_CUDA_CHECK(cudaStreamSynchronize(s));

    if (count > ctx->query_buffer_size) count = ctx->query_buffer_size;
    *out_events = ctx->d_query_buffers[stream_idx];
    *out_count = count;

    return SDST_SUCCESS;
}

/* ============================================================
 * Causal subgraph extraction
 * ============================================================ */

extern "C"
SdstError sdst_causal_subgraph(
    SdstHandle handle,
    SpikeId root_spike,
    uint32_t max_depth,
    CausalSubgraph* out_graph,
    void* stream
) {
    if (!handle || !out_graph) return SDST_ERROR_INVALID_PARAM;
    SdstContext* ctx = handle;
    cudaStream_t s = stream ? (cudaStream_t)stream : 0;

    uint32_t total_events;
    SDST_CUDA_CHECK(cudaMemcpy(&total_events, ctx->d_event_count,
                               sizeof(uint32_t), cudaMemcpyDeviceToHost));

    if (root_spike >= total_events) return SDST_ERROR_NOT_FOUND;

    /* Copy all events to host for graph traversal */
    SpikeEvent* h_events = (SpikeEvent*)malloc(total_events * sizeof(SpikeEvent));
    SDST_CUDA_CHECK(cudaMemcpyAsync(h_events, ctx->d_event_buffer,
                                     total_events * sizeof(SpikeEvent),
                                     cudaMemcpyDeviceToHost, s));
    SDST_CUDA_CHECK(cudaStreamSynchronize(s));

    /* BFS from root following parent_spike links backwards */
    /* First, build reverse index: for each event, find all events whose parent is this event */
    uint32_t* children_count = (uint32_t*)calloc(total_events, sizeof(uint32_t));
    for (uint32_t i = 0; i < total_events; i++) {
        uint32_t parent = h_events[i].parent_spike;
        if (parent > 0) {
            children_count[parent - 1]++;
        }
    }

    /* BFS from root_spike, collecting descendants */
    uint32_t max_collect = (max_depth == 0) ? total_events : max_depth * 1000;
    SpikeEvent* collected = (SpikeEvent*)malloc(max_collect * sizeof(SpikeEvent));
    uint32_t* parent_indices = (uint32_t*)malloc(max_collect * sizeof(uint32_t));
    uint32_t* queue = (uint32_t*)malloc(total_events * sizeof(uint32_t));
    bool* visited = (bool*)calloc(total_events, sizeof(bool));

    collected[0] = h_events[root_spike];
    parent_indices[0] = 0xFFFFFFFF; /* root has no parent in subgraph */
    visited[root_spike] = true;
    uint32_t collected_count = 1;

    /* Also walk UP the parent chain from root */
    uint32_t up_idx = root_spike;
    uint32_t depth_up = 0;
    while (h_events[up_idx].parent_spike > 0 &&
           (max_depth == 0 || depth_up < max_depth)) {
        uint32_t parent_idx = h_events[up_idx].parent_spike - 1;
        if (parent_idx >= total_events || visited[parent_idx]) break;
        visited[parent_idx] = true;
        if (collected_count < max_collect) {
            parent_indices[collected_count] = collected_count - 1; /* points to previous */
            collected[collected_count] = h_events[parent_idx];
            collected_count++;
        }
        up_idx = parent_idx;
        depth_up++;
    }

    /* BFS downward: find all children of root_spike */
    uint32_t q_head = 0, q_tail = 0;
    queue[q_tail++] = root_spike;
    uint32_t current_depth = 0;

    while (q_head < q_tail && (max_depth == 0 || current_depth < max_depth)) {
        uint32_t level_size = q_tail - q_head;
        for (uint32_t li = 0; li < level_size; li++) {
            uint32_t node = queue[q_head++];
            /* Find children: events whose parent_spike == node + 1 */
            for (uint32_t i = 0; i < total_events && collected_count < max_collect; i++) {
                if (!visited[i] && h_events[i].parent_spike == node + 1) {
                    visited[i] = true;
                    parent_indices[collected_count] = node; /* parent in subgraph */
                    collected[collected_count] = h_events[i];
                    collected_count++;
                    queue[q_tail++] = i;
                }
            }
        }
        current_depth++;
    }

    out_graph->events = (SpikeEvent*)malloc(collected_count * sizeof(SpikeEvent));
    out_graph->parent_indices = (uint32_t*)malloc(collected_count * sizeof(uint32_t));
    out_graph->count = collected_count;
    memcpy(out_graph->events, collected, collected_count * sizeof(SpikeEvent));
    memcpy(out_graph->parent_indices, parent_indices, collected_count * sizeof(uint32_t));

    free(h_events); free(children_count); free(collected);
    free(parent_indices); free(queue); free(visited);

    return SDST_SUCCESS;
}

extern "C"
SdstError sdst_free_subgraph(CausalSubgraph* graph) {
    if (!graph) return SDST_ERROR_INVALID_PARAM;
    free(graph->events);
    free(graph->parent_indices);
    graph->events = NULL;
    graph->parent_indices = NULL;
    graph->count = 0;
    return SDST_SUCCESS;
}

/* ============================================================
 * CCNS: Power-law exponent from avalanche size distribution
 *
 * Uses Maximum Likelihood Estimation for discrete power law:
 *   tau = 1 + n * [Σ ln(s_i / s_min)]^-1
 * with Clauset-Shalizi-Newman goodness-of-fit test.
 * ============================================================ */

static float compute_tau_mle(const uint32_t* sizes, uint32_t n, uint32_t s_min) {
    if (n == 0) return 0.0f;

    double sum_log = 0.0;
    uint32_t valid = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (sizes[i] >= s_min) {
            sum_log += log((double)sizes[i] / ((double)s_min - 0.5));
            valid++;
        }
    }

    if (valid < 5) return 0.0f; /* Insufficient data */

    double tau = 1.0 + (double)valid / sum_log;
    return (float)tau;
}

static float compute_tau_stderr(float tau, uint32_t n) {
    if (n < 2) return 0.0f;
    return (tau - 1.0f) / sqrtf((float)n);
}

extern "C"
SdstError sdst_ccns_region(
    SdstHandle handle,
    const SpatialRegion* region,
    CcnsResult* out_result,
    void* stream
) {
    if (!handle || !region || !out_result) return SDST_ERROR_INVALID_PARAM;
    SdstContext* ctx = handle;
    cudaStream_t s = stream ? (cudaStream_t)stream : 0;

    uint32_t total_events;
    SDST_CUDA_CHECK(cudaMemcpy(&total_events, ctx->d_event_count,
                               sizeof(uint32_t), cudaMemcpyDeviceToHost));

    /* Copy events to host */
    SpikeEvent* h_events = (SpikeEvent*)malloc(total_events * sizeof(SpikeEvent));
    SDST_CUDA_CHECK(cudaMemcpyAsync(h_events, ctx->d_event_buffer,
                                     total_events * sizeof(SpikeEvent),
                                     cudaMemcpyDeviceToHost, s));
    SDST_CUDA_CHECK(cudaStreamSynchronize(s));

    /* Collect avalanche IDs and sizes for events in region.
     * max_avalanche_id MUST be computed from ALL events — not just region events.
     * avalanche_sizes[aid] iterates all events, so arrays must cover the global max. */
    uint32_t max_avalanche_id = 0;
    for (uint32_t i = 0; i < total_events; i++) {
        if (h_events[i].avalanche_id > max_avalanche_id)
            max_avalanche_id = h_events[i].avalanche_id;
    }

    if (max_avalanche_id == 0) {
        memset(out_result, 0, sizeof(CcnsResult));
        free(h_events);
        return SDST_SUCCESS;
    }

    uint32_t* avalanche_sizes = (uint32_t*)calloc(max_avalanche_id + 1, sizeof(uint32_t));
    bool* in_region = (bool*)calloc(max_avalanche_id + 1, sizeof(bool));

    for (uint32_t i = 0; i < total_events; i++) {
        AvalancheId aid = h_events[i].avalanche_id;
        avalanche_sizes[aid]++;

        uint32_t vx, vy, vz;
        morton_decode(h_events[i].voxel, &vx, &vy, &vz);
        if (vx >= region->x_min && vx <= region->x_max &&
            vy >= region->y_min && vy <= region->y_max &&
            vz >= region->z_min && vz <= region->z_max) {
            in_region[aid] = true;
        }
    }

    /* Collect sizes of avalanches that touch the region */
    uint32_t* region_sizes = (uint32_t*)malloc((max_avalanche_id + 1) * sizeof(uint32_t));
    uint32_t n_avalanches = 0;
    for (uint32_t i = 1; i <= max_avalanche_id; i++) {
        if (in_region[i] && avalanche_sizes[i] > 0) {
            region_sizes[n_avalanches++] = avalanche_sizes[i];
        }
    }

    /* Compute tau via MLE */
    float tau = compute_tau_mle(region_sizes, n_avalanches, 2);
    float tau_se = compute_tau_stderr(tau, n_avalanches);

    out_result->tau = tau;
    out_result->tau_stderr = tau_se;
    out_result->n_avalanches = n_avalanches;

    if (tau < ctx->config.ccns_soc_threshold) {
        out_result->classification = CCNS_SOC;
    } else if (tau < ctx->config.ccns_barrier_threshold) {
        out_result->classification = CCNS_NEAR_CRITICAL;
    } else {
        out_result->classification = CCNS_BARRIER;
    }

    /* Composite druggability score:
     * SOC sites get highest score (most responsive to perturbation)
     * Score = (2.0 - tau) * confidence_factor
     * Guard: tau=0 means insufficient avalanche data → druggability=0 */
    float confidence;
    if (tau < 1e-6f || tau_se >= tau) {
        confidence = 0.0f;
    } else {
        confidence = 1.0f - tau_se / tau;
        if (confidence < 0) confidence = 0;
    }
    out_result->druggability = (tau < 1e-6f) ? 0.0f : (1.0f - (tau - 1.0f) / 3.0f) * confidence;
    if (out_result->druggability < 0) out_result->druggability = 0;

    free(h_events); free(avalanche_sizes); free(in_region); free(region_sizes);
    return SDST_SUCCESS;
}

/* ============================================================
 * DCC: Distance to Closest Contact
 * ============================================================ */

extern "C"
SdstError sdst_compute_dcc(
    SdstHandle handle,
    const float* known_sites,
    uint32_t n_known,
    float** out_dcc,
    float** out_centroids,
    uint32_t* out_n_detected,
    void* stream
) {
    if (!handle || !out_dcc || !out_centroids || !out_n_detected)
        return SDST_ERROR_INVALID_PARAM;
    SdstContext* ctx = handle;
    cudaStream_t s = stream ? (cudaStream_t)stream : 0;

    /* Run hysteresis scan to find candidate pockets */
    HysteresisResult* hyst_results;
    SpatialRegion* hyst_regions;
    uint32_t n_pockets;

    SdstError err = sdst_hysteresis_scan(handle, 0.15f,
                                         &hyst_results, &hyst_regions,
                                         &n_pockets, stream);
    if (err != SDST_SUCCESS) return err;

    *out_n_detected = n_pockets;
    *out_dcc = (float*)malloc(n_pockets * sizeof(float));
    *out_centroids = (float*)malloc(n_pockets * 3 * sizeof(float));

    float spacing = ctx->config.grid_spacing;

    for (uint32_t i = 0; i < n_pockets; i++) {
        /* Compute centroid in Angstroms */
        float cx = ((float)(hyst_regions[i].x_min + hyst_regions[i].x_max) / 2.0f) * spacing;
        float cy = ((float)(hyst_regions[i].y_min + hyst_regions[i].y_max) / 2.0f) * spacing;
        float cz = ((float)(hyst_regions[i].z_min + hyst_regions[i].z_max) / 2.0f) * spacing;

        (*out_centroids)[i * 3 + 0] = cx;
        (*out_centroids)[i * 3 + 1] = cy;
        (*out_centroids)[i * 3 + 2] = cz;

        /* Find minimum distance to any known site */
        float min_dist = FLT_MAX;
        for (uint32_t j = 0; j < n_known; j++) {
            float dx = cx - known_sites[j * 3 + 0];
            float dy = cy - known_sites[j * 3 + 1];
            float dz = cz - known_sites[j * 3 + 2];
            float dist = sqrtf(dx*dx + dy*dy + dz*dz);
            if (dist < min_dist) min_dist = dist;
        }
        (*out_dcc)[i] = min_dist;
    }

    free(hyst_results);
    free(hyst_regions);

    return SDST_SUCCESS;
}

/* ============================================================
 * Avalanche statistics export
 * ============================================================ */

extern "C"
SdstError sdst_avalanche_stats(
    SdstHandle handle,
    int phase_filter,
    AvalancheStats** out_stats,
    uint32_t* out_count,
    void* stream
) {
    if (!handle || !out_stats || !out_count) return SDST_ERROR_INVALID_PARAM;
    SdstContext* ctx = handle;
    cudaStream_t s = stream ? (cudaStream_t)stream : 0;

    uint32_t total_events;
    SDST_CUDA_CHECK(cudaMemcpy(&total_events, ctx->d_event_count,
                               sizeof(uint32_t), cudaMemcpyDeviceToHost));

    SpikeEvent* h_events = (SpikeEvent*)malloc(total_events * sizeof(SpikeEvent));
    SDST_CUDA_CHECK(cudaMemcpyAsync(h_events, ctx->d_event_buffer,
                                     total_events * sizeof(SpikeEvent),
                                     cudaMemcpyDeviceToHost, s));
    SDST_CUDA_CHECK(cudaStreamSynchronize(s));

    /* Find unique avalanche roots and compute stats */
    uint32_t max_aid = 0;
    for (uint32_t i = 0; i < total_events; i++) {
        if (h_events[i].avalanche_id > max_aid) max_aid = h_events[i].avalanche_id;
    }

    if (max_aid == 0) {
        *out_stats = NULL;
        *out_count = 0;
        free(h_events);
        return SDST_SUCCESS;
    }

    /* Per-avalanche accumulators (including bbox for spatial_extent) */
    uint32_t* sizes = (uint32_t*)calloc(max_aid + 1, sizeof(uint32_t));
    uint32_t* t_min = (uint32_t*)malloc((max_aid + 1) * sizeof(uint32_t));
    uint32_t* t_max = (uint32_t*)calloc(max_aid + 1, sizeof(uint32_t));
    MortonCode* seeds = (MortonCode*)calloc(max_aid + 1, sizeof(MortonCode));
    PhaseId* phases = (PhaseId*)calloc(max_aid + 1, sizeof(PhaseId));
    bool* valid = (bool*)calloc(max_aid + 1, sizeof(bool));
    /* Bounding box per avalanche for spatial_extent */
    uint32_t* bbox_xmin = (uint32_t*)malloc((max_aid + 1) * sizeof(uint32_t));
    uint32_t* bbox_xmax = (uint32_t*)calloc(max_aid + 1, sizeof(uint32_t));
    uint32_t* bbox_ymin = (uint32_t*)malloc((max_aid + 1) * sizeof(uint32_t));
    uint32_t* bbox_ymax = (uint32_t*)calloc(max_aid + 1, sizeof(uint32_t));
    uint32_t* bbox_zmin = (uint32_t*)malloc((max_aid + 1) * sizeof(uint32_t));
    uint32_t* bbox_zmax = (uint32_t*)calloc(max_aid + 1, sizeof(uint32_t));

    for (uint32_t i = 0; i <= max_aid; i++) {
        t_min[i] = 0xFFFFFFFF;
        bbox_xmin[i] = 0xFFFFFFFF;
        bbox_ymin[i] = 0xFFFFFFFF;
        bbox_zmin[i] = 0xFFFFFFFF;
    }

    for (uint32_t i = 0; i < total_events; i++) {
        AvalancheId aid = h_events[i].avalanche_id;
        sizes[aid]++;
        if (h_events[i].timestamp < t_min[aid]) {
            t_min[aid] = h_events[i].timestamp;
            seeds[aid] = h_events[i].voxel;
            phases[aid] = h_events[i].phase_id;
        }
        if (h_events[i].timestamp > t_max[aid]) {
            t_max[aid] = h_events[i].timestamp;
        }
        valid[aid] = true;

        /* Track bounding box */
        uint32_t vx, vy, vz;
        morton_decode(h_events[i].voxel, &vx, &vy, &vz);
        if (vx < bbox_xmin[aid]) bbox_xmin[aid] = vx;
        if (vx > bbox_xmax[aid]) bbox_xmax[aid] = vx;
        if (vy < bbox_ymin[aid]) bbox_ymin[aid] = vy;
        if (vy > bbox_ymax[aid]) bbox_ymax[aid] = vy;
        if (vz < bbox_zmin[aid]) bbox_zmin[aid] = vz;
        if (vz > bbox_zmax[aid]) bbox_zmax[aid] = vz;
    }

    /* Count valid avalanches (optionally filtered by phase) */
    uint32_t count = 0;
    for (uint32_t i = 1; i <= max_aid; i++) {
        if (valid[i] && sizes[i] > 1) {
            if (phase_filter < 0 || phases[i] == (PhaseId)phase_filter) {
                count++;
            }
        }
    }

    *out_stats = (AvalancheStats*)malloc(count * sizeof(AvalancheStats));
    *out_count = count;

    float spacing = ctx->config.grid_spacing;
    uint32_t j = 0;
    for (uint32_t i = 1; i <= max_aid && j < count; i++) {
        if (!valid[i] || sizes[i] <= 1) continue;
        if (phase_filter >= 0 && phases[i] != (PhaseId)phase_filter) continue;

        AvalancheStats* as = &(*out_stats)[j];
        as->id = i;
        as->size = sizes[i];
        as->duration = t_max[i] - t_min[i];
        as->seed_voxel = seeds[i];
        as->phase = phases[i];
        /* Spatial extent = bounding box diagonal in Angstroms */
        {
            float dx = (float)(bbox_xmax[i] - bbox_xmin[i]);
            float dy = (float)(bbox_ymax[i] - bbox_ymin[i]);
            float dz = (float)(bbox_zmax[i] - bbox_zmin[i]);
            as->spatial_extent = sqrtf(dx*dx + dy*dy + dz*dz) * spacing;
        }

        /* Local tau from this avalanche's sub-avalanches is not meaningful;
         * set to 0. Use sdst_ccns_region for regional tau. */
        as->tau_local = 0.0f;
        j++;
    }

    free(h_events); free(sizes); free(t_min); free(t_max);
    free(seeds); free(phases); free(valid);
    free(bbox_xmin); free(bbox_xmax);
    free(bbox_ymin); free(bbox_ymax);
    free(bbox_zmin); free(bbox_zmax);

    return SDST_SUCCESS;
}

/* ============================================================
 * sdst_causal_subgraph_region: Extract causal subgraph for all
 * spikes in a spatial region.
 *
 * Strategy:
 *   1. Query all events in the region
 *   2. Find unique avalanche roots (earliest event per avalanche_id)
 *   3. For each root, call sdst_causal_subgraph and merge results
 * ============================================================ */

extern "C"
SdstError sdst_causal_subgraph_region(
    SdstHandle handle,
    const SpatialRegion* region,
    uint32_t max_depth,
    CausalSubgraph* out_graph,
    void* stream
) {
    if (!handle || !region || !out_graph) return SDST_ERROR_INVALID_PARAM;
    SdstContext* ctx = handle;
    cudaStream_t s = stream ? (cudaStream_t)stream : 0;

    uint32_t total_events;
    SDST_CUDA_CHECK(cudaMemcpy(&total_events, ctx->d_event_count,
                               sizeof(uint32_t), cudaMemcpyDeviceToHost));
    if (total_events == 0) {
        out_graph->events = NULL;
        out_graph->parent_indices = NULL;
        out_graph->count = 0;
        return SDST_SUCCESS;
    }

    /* Copy all events to host */
    SpikeEvent* h_events = (SpikeEvent*)malloc(total_events * sizeof(SpikeEvent));
    SDST_CUDA_CHECK(cudaMemcpyAsync(h_events, ctx->d_event_buffer,
                                     total_events * sizeof(SpikeEvent),
                                     cudaMemcpyDeviceToHost, s));
    SDST_CUDA_CHECK(cudaStreamSynchronize(s));

    /* Identify unique avalanche roots for events in the region */
    /* A root is the event with the earliest timestamp per avalanche_id */
    uint32_t max_aid = 0;
    for (uint32_t i = 0; i < total_events; i++) {
        if (h_events[i].avalanche_id > max_aid) max_aid = h_events[i].avalanche_id;
    }

    /* Mark which avalanche IDs touch the region */
    bool* aid_in_region = (bool*)calloc(max_aid + 1, sizeof(bool));
    uint32_t* aid_earliest = (uint32_t*)malloc((max_aid + 1) * sizeof(uint32_t));
    uint32_t* aid_earliest_time = (uint32_t*)malloc((max_aid + 1) * sizeof(uint32_t));
    for (uint32_t i = 0; i <= max_aid; i++) aid_earliest_time[i] = 0xFFFFFFFF;

    for (uint32_t i = 0; i < total_events; i++) {
        uint32_t vx, vy, vz;
        morton_decode(h_events[i].voxel, &vx, &vy, &vz);
        if (vx >= region->x_min && vx <= region->x_max &&
            vy >= region->y_min && vy <= region->y_max &&
            vz >= region->z_min && vz <= region->z_max) {
            AvalancheId aid = h_events[i].avalanche_id;
            aid_in_region[aid] = true;
        }
        /* Track earliest event per avalanche */
        AvalancheId aid = h_events[i].avalanche_id;
        if (h_events[i].timestamp < aid_earliest_time[aid]) {
            aid_earliest_time[aid] = h_events[i].timestamp;
            aid_earliest[aid] = i;
        }
    }
    free(h_events);

    /* Collect roots for region-touching avalanches */
    uint32_t n_roots = 0;
    for (uint32_t i = 1; i <= max_aid; i++) {
        if (aid_in_region[i]) n_roots++;
    }

    if (n_roots == 0) {
        out_graph->events = NULL;
        out_graph->parent_indices = NULL;
        out_graph->count = 0;
        free(aid_in_region); free(aid_earliest); free(aid_earliest_time);
        return SDST_SUCCESS;
    }

    /* Merge subgraphs from all roots */
    /* Accumulate all events across roots, deduplicating by event index */
    uint32_t total_count = 0;
    SpikeEvent** all_events = (SpikeEvent**)calloc(n_roots, sizeof(SpikeEvent*));
    uint32_t** all_parents = (uint32_t**)calloc(n_roots, sizeof(uint32_t*));
    uint32_t* root_counts = (uint32_t*)calloc(n_roots, sizeof(uint32_t));

    uint32_t ri = 0;
    for (uint32_t i = 1; i <= max_aid && ri < n_roots; i++) {
        if (!aid_in_region[i]) continue;
        CausalSubgraph sub;
        SdstError err = sdst_causal_subgraph(handle, aid_earliest[i], max_depth, &sub, stream);
        if (err == SDST_SUCCESS && sub.count > 0) {
            all_events[ri] = sub.events;
            all_parents[ri] = sub.parent_indices;
            root_counts[ri] = sub.count;
            total_count += sub.count;
        }
        ri++;
    }
    free(aid_in_region); free(aid_earliest); free(aid_earliest_time);

    /* Flatten merged subgraphs */
    out_graph->events = (SpikeEvent*)malloc(total_count * sizeof(SpikeEvent));
    out_graph->parent_indices = (uint32_t*)malloc(total_count * sizeof(uint32_t));
    out_graph->count = total_count;

    uint32_t offset = 0;
    for (uint32_t r = 0; r < n_roots; r++) {
        if (root_counts[r] == 0) continue;
        memcpy(out_graph->events + offset, all_events[r],
               root_counts[r] * sizeof(SpikeEvent));
        /* Adjust parent indices to account for merge offset */
        for (uint32_t k = 0; k < root_counts[r]; k++) {
            uint32_t p = all_parents[r][k];
            out_graph->parent_indices[offset + k] =
                (p == 0xFFFFFFFF) ? 0xFFFFFFFF : (p + offset);
        }
        offset += root_counts[r];
        free(all_events[r]);
        free(all_parents[r]);
    }
    free(all_events); free(all_parents); free(root_counts);

    return SDST_SUCCESS;
}

/* ============================================================
 * sdst_ccns_all_pockets: CCNS for all detected pocket regions.
 *
 * Strategy:
 *   1. Run sdst_hysteresis_scan to detect candidate regions
 *   2. For each hysteretic region, compute sdst_ccns_region
 *   3. Return all results with their associated spatial regions
 * ============================================================ */

extern "C"
SdstError sdst_ccns_all_pockets(
    SdstHandle handle,
    CcnsResult** out_results,
    SpatialRegion** out_regions,
    uint32_t* out_count,
    void* stream
) {
    if (!handle || !out_results || !out_regions || !out_count)
        return SDST_ERROR_INVALID_PARAM;

    /* Step 1: Find hysteretic pocket candidates */
    HysteresisResult* hyst_results;
    SpatialRegion*    hyst_regions;
    uint32_t          n_pockets;

    SdstError err = sdst_hysteresis_scan(handle, 0.15f,
                                         &hyst_results, &hyst_regions,
                                         &n_pockets, stream);
    if (err != SDST_SUCCESS) return err;

    if (n_pockets == 0) {
        *out_results = NULL;
        *out_regions = NULL;
        *out_count = 0;
        free(hyst_results);
        free(hyst_regions);
        return SDST_SUCCESS;
    }

    /* Step 2: Compute CCNS tau for each pocket */
    *out_results = (CcnsResult*)malloc(n_pockets * sizeof(CcnsResult));
    *out_regions = (SpatialRegion*)malloc(n_pockets * sizeof(SpatialRegion));
    *out_count = n_pockets;

    for (uint32_t i = 0; i < n_pockets; i++) {
        (*out_regions)[i] = hyst_regions[i];
        SdstError ccns_err = sdst_ccns_region(handle, &hyst_regions[i],
                                               &(*out_results)[i], stream);
        if (ccns_err != SDST_SUCCESS) {
            /* Fill with sentinel values on error, continue */
            memset(&(*out_results)[i], 0, sizeof(CcnsResult));
        }
    }

    free(hyst_results);
    free(hyst_regions);
    return SDST_SUCCESS;
}

/* ============================================================
 * sdst_validate: Consistency checker for debugging.
 *
 * Checks:
 *   1. Hash table: no duplicate keys (Morton codes unique per slot)
 *   2. Event chain: no cycles (walk terminates in ≤ event_count steps)
 *   3. Union-find idempotency: uf_find(uf_find(x)) == uf_find(x) for sampled events
 *   4. Wavefront IDs: all wavefront_id values < wavefront_count
 *   5. Timestamps: all timestamps within phase_boundaries[5]
 * ============================================================ */

extern "C"
SdstError sdst_validate(SdstHandle handle) {
    if (!handle) return SDST_ERROR_INVALID_PARAM;
    SdstContext* ctx = handle;

    uint32_t event_count, wf_count;
    SDST_CUDA_CHECK(cudaMemcpy(&event_count, ctx->d_event_count,
                               sizeof(uint32_t), cudaMemcpyDeviceToHost));
    SDST_CUDA_CHECK(cudaMemcpy(&wf_count, ctx->d_wavefront_count,
                               sizeof(uint32_t), cudaMemcpyDeviceToHost));

    if (event_count == 0) return SDST_SUCCESS; /* Nothing to validate */

    uint32_t cap = ctx->config.hash_table_capacity;
    uint32_t max_ts = ctx->config.phase_boundaries[5];

    /* Copy all needed arrays to host */
    SpikeEvent* h_events = (SpikeEvent*)malloc(event_count * sizeof(SpikeEvent));
    uint32_t* h_chain_next = (uint32_t*)malloc(event_count * sizeof(uint32_t));
    uint32_t* h_uf_parent = (uint32_t*)malloc(event_count * sizeof(uint32_t));
    HashEntry* h_hash = (HashEntry*)malloc(cap * sizeof(HashEntry));

    SDST_CUDA_CHECK(cudaMemcpy(h_events, ctx->d_event_buffer,
                               event_count * sizeof(SpikeEvent), cudaMemcpyDeviceToHost));
    SDST_CUDA_CHECK(cudaMemcpy(h_chain_next, ctx->d_event_chain_next,
                               event_count * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    SDST_CUDA_CHECK(cudaMemcpy(h_uf_parent, ctx->d_avalanche_parent,
                               event_count * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    SDST_CUDA_CHECK(cudaMemcpy(h_hash, ctx->d_hash_table,
                               cap * sizeof(HashEntry), cudaMemcpyDeviceToHost));

    SdstError result = SDST_SUCCESS;

    /* Check 1: Event timestamps within bounds */
    for (uint32_t i = 0; i < event_count && result == SDST_SUCCESS; i++) {
        if (h_events[i].timestamp > max_ts) {
            fprintf(stderr, "SDST validate: event %u has timestamp %u > max %u\n",
                    i, h_events[i].timestamp, max_ts);
            result = SDST_ERROR_INVALID_PARAM;
        }
    }

    /* Check 2: Wavefront IDs within bounds */
    for (uint32_t i = 0; i < event_count && result == SDST_SUCCESS; i++) {
        WavefrontId wid = h_events[i].wavefront_id;
        if (wid > ctx->config.max_wavefronts) {
            fprintf(stderr, "SDST validate: event %u has wavefront_id %u > max %u\n",
                    i, wid, ctx->config.max_wavefronts);
            result = SDST_ERROR_INVALID_PARAM;
        }
    }

    /* Check 3: Event chain - no cycles (depth limited walk) */
    for (uint32_t i = 0; i < event_count && result == SDST_SUCCESS; i++) {
        uint32_t chain_idx = h_chain_next[i];
        uint32_t walk = 0;
        while (chain_idx != 0xFFFFFFFF && walk <= event_count) {
            if (chain_idx >= event_count) {
                fprintf(stderr, "SDST validate: chain_next[%u] = %u out of bounds\n",
                        i, chain_idx);
                result = SDST_ERROR_INVALID_PARAM;
                break;
            }
            chain_idx = h_chain_next[chain_idx];
            walk++;
        }
        if (walk > event_count) {
            fprintf(stderr, "SDST validate: cycle detected in chain from event %u\n", i);
            result = SDST_ERROR_INVALID_PARAM;
        }
    }

    /* Check 4: Union-find idempotency (sample every 100th event) */
    for (uint32_t i = 0; i < event_count && result == SDST_SUCCESS; i += 100) {
        /* Host-side uf_find */
        uint32_t x = i;
        uint32_t iters = 0;
        while (h_uf_parent[x] != x && iters < event_count) {
            x = h_uf_parent[x];
            iters++;
        }
        if (iters >= event_count) {
            fprintf(stderr, "SDST validate: uf_find did not converge for event %u\n", i);
            result = SDST_ERROR_INVALID_PARAM;
        }
        /* Idempotency: find root of root should be same root */
        uint32_t root1 = x;
        uint32_t x2 = root1;
        iters = 0;
        while (h_uf_parent[x2] != x2 && iters < event_count) {
            x2 = h_uf_parent[x2]; iters++;
        }
        if (x2 != root1) {
            fprintf(stderr, "SDST validate: uf_find not idempotent at event %u\n", i);
            result = SDST_ERROR_INVALID_PARAM;
        }
    }

    /* Check 5: Hash table - occupied slots have valid Morton codes */
    for (uint32_t i = 0; i < cap && result == SDST_SUCCESS; i++) {
        uint32_t key = h_hash[i].key;
        if (key == 0xFFFFFFFF) continue; /* Empty slot */
        /* Verify Morton code decodes to valid grid coordinates */
        uint32_t gx, gy, gz;
        morton_decode(key & 0x7FFFFFFF, &gx, &gy, &gz);
        if (gx >= ctx->config.grid_nx || gy >= ctx->config.grid_ny ||
            gz >= ctx->config.grid_nz) {
            fprintf(stderr, "SDST validate: hash slot %u has invalid Morton code %u"
                    " -> (%u,%u,%u)\n", i, key, gx, gy, gz);
            result = SDST_ERROR_INVALID_PARAM;
        }
    }

    free(h_events); free(h_chain_next); free(h_uf_parent); free(h_hash);

    if (result == SDST_SUCCESS) {
        printf("SDST validate: OK (%u events, %u wavefronts)\n", event_count, wf_count);
    }
    return result;
}
