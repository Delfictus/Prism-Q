/** Regression fixtures for native implementation contracts, not campaign evidence.
 * Includes the actual estimator to test its private guards without reimplementing it.
 * Link this executable to the same rebuilt libsdst used by offline replay. */
#include "../src/sdst_tide.cu"
#include <assert.h>
#include <vector>
#include <stdio.h>

static void check_te_receipts() {
    uint8_t src[40] = {}, tgt[40] = {};
    for (unsigned i = 0; i < 40; ++i) { src[i] = (i % 4) < 2; tgt[i] = i % 2; }
    TideMeasurementSupportV1 support = {};
    float baseline = compute_transfer_entropy(src, tgt, 40);
    float recorded = compute_transfer_entropy(src, tgt, 40, &support);
    assert(memcmp(&baseline, &recorded, sizeof(float)) == 0);
    assert(support.unavailable_flags == 0 && support.source_active_bins == 20 && support.target_active_bins == 20);
    unsigned pairs = 0; for (auto n : support.joint_counts) pairs += n;
    assert(pairs == 39);
    TideMeasurementSupportV1 window_support = {};
    float v0 = compute_fisher_info(src, tgt, 40, 10);
    float v1 = compute_fisher_info(src, tgt, 40, 10, &window_support);
    assert(memcmp(&v0, &v1, sizeof(float)) == 0);
    assert(window_support.variance_total_windows == 4 && window_support.variance_valid_windows == 4);
    memset(src, 0, sizeof(src));
    memset(tgt, 1, sizeof(tgt));
    support = {};
    assert(isnan(compute_transfer_entropy(src, tgt, 40, &support)));
    assert(support.unavailable_flags == (2u | 16u));
    assert(support.source_active_bins == 0 && support.target_active_bins == 40);
    memset(src, 1, sizeof(src)); memset(tgt, 0, sizeof(tgt));
    support = {};
    assert(isnan(compute_transfer_entropy(src, tgt, 40, &support)));
    assert(support.unavailable_flags == (4u | 8u));
    // Exact inclusive guard boundaries are observable, not silently rounded.
    memset(src, 0, sizeof(src)); for (unsigned i = 0; i < 2; i++) src[i] = 1;
    for (unsigned i = 0; i < 40; i++) tgt[i] = i % 2;
    support = {};
    assert(isnan(compute_transfer_entropy(src, tgt, 40, &support)) && support.unavailable_flags == 2);
    memset(src, 1, sizeof(src)); src[0] = src[1] = 0;
    support = {};
    assert(isnan(compute_transfer_entropy(src, tgt, 40, &support)) && support.unavailable_flags == 4);
    support = {};
    assert(isnan(compute_transfer_entropy(src, tgt, 2, &support)) && support.unavailable_flags == 1);
    assert(sizeof(TideDecomposition) == 24 && sizeof(TideMeasurementSupportV1) == 88);
    printf("PASS TIDE guard reasons, boundary occupancy, support counts, and bit-identical estimator values\n");
}

static void check_shuffled_parent_inheritance() {
    SdstConfig config = sdst_default_config();
    config.grid_nx = config.grid_ny = config.grid_nz = 16;
    config.hash_table_capacity = 64;
    config.max_spike_events = config.max_wavefronts = 32;
    config.num_streams = 1;
    config.grid_spacing = 0.75f;
    config.wavefront_merge_dist = 2.25f;
    config.wavefront_max_dt = 5.0f;
    config.phase_boundaries[5] = 100;
    SdstHandle handle = NULL;
    assert(sdst_create(&config, &handle) == SDST_SUCCESS);
    // Shuffled memory: youngest first; index 0 -> 1 -> 2. Root 2 is last.
    // Event 3 fails dt only; event 4 fails distance only; event 5 is equal-time.
    SpikeEvent events[7] = {};
    events[0].voxel = morton_encode(4, 2, 2); events[0].timestamp = 16; events[0].parent_spike = 2;
    events[1].voxel = morton_encode(3, 2, 2); events[1].timestamp = 12; events[1].parent_spike = 3;
    events[2].voxel = morton_encode(2, 2, 2); events[2].timestamp = 10;
    events[3].voxel = morton_encode(3, 2, 2); events[3].timestamp = 22; events[3].parent_spike = 1;
    events[4].voxel = morton_encode(9, 2, 2); events[4].timestamp = 17; events[4].parent_spike = 1;
    events[5].voxel = morton_encode(3, 3, 2); events[5].timestamp = 16; events[5].parent_spike = 1;
    for (unsigned repeat = 0; repeat < 10; ++repeat) {
        assert(sdst_reset(handle) == SDST_SUCCESS);
        assert(cudaMemcpy(handle->d_event_buffer, events, 6 * sizeof(SpikeEvent), cudaMemcpyHostToDevice) == cudaSuccess);
        assert(sdst_process_wavefronts(handle, 0, 6, NULL) == SDST_SUCCESS);
        SpikeEvent got[7] = {};
        assert(cudaMemcpy(got, handle->d_event_buffer, 6 * sizeof(SpikeEvent), cudaMemcpyDeviceToHost) == cudaSuccess);
        assert(got[0].wavefront_id && got[0].wavefront_id == got[1].wavefront_id && got[1].wavefront_id == got[2].wavefront_id);
        assert(got[3].wavefront_id != got[0].wavefront_id && got[4].wavefront_id != got[0].wavefront_id && got[5].wavefront_id != got[0].wavefront_id);
        uint32_t count = 0;
        assert(cudaMemcpy(&count, handle->d_wavefront_count, sizeof(count), cudaMemcpyDeviceToHost) == cudaSuccess);
        assert(count == 4);
        WavefrontStats stats[32];
        assert(cudaMemcpy(stats, handle->d_wavefront_stats, sizeof(stats), cudaMemcpyDeviceToHost) == cudaSuccess);
        assert(stats[got[0].wavefront_id - 1].spike_count == 3);
        assert(fabs(stats[got[0].wavefront_id - 1].spatial_extent - 1.5f) < 1e-6f);
        // A later batch must inherit an already assigned parent without new roots.
        events[6].voxel = morton_encode(4, 3, 2); events[6].timestamp = 18; events[6].parent_spike = 1;
        assert(cudaMemcpy(handle->d_event_buffer + 6, events + 6, sizeof(SpikeEvent), cudaMemcpyHostToDevice) == cudaSuccess);
        assert(sdst_process_wavefronts(handle, 6, 1, NULL) == SDST_SUCCESS);
        assert(cudaMemcpy(got + 6, handle->d_event_buffer + 6, sizeof(SpikeEvent), cudaMemcpyDeviceToHost) == cudaSuccess);
        assert(got[6].wavefront_id == got[0].wavefront_id);
        assert(cudaMemcpy(stats, handle->d_wavefront_stats, sizeof(stats), cudaMemcpyDeviceToHost) == cudaSuccess);
        assert(stats[got[0].wavefront_id - 1].spike_count == 4);
    }
    assert(sdst_destroy(handle) == SDST_SUCCESS);
    printf("PASS shuffled parent chain, threshold exclusions, exact root stats, cross-batch inheritance (10 repeats)\n");
}

int main() {
    check_te_receipts();
    check_shuffled_parent_inheritance();
    return 0;
}
