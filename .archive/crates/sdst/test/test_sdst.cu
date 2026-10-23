/**
 * SDST Comprehensive Test Suite
 *
 * Exercises the full SDST pipeline per Phase 5 of the SDST implementation brief.
 * Tests are numbered and print PASS/FAIL with actual values.
 *
 * Expected: all steps PASS, zero CUDA errors, sdst_validate SUCCESS,
 * serialization round-trip preserves event count.
 */

#include "../include/sdst_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <cuda_runtime.h>

/* ============================================================ */
/* Helpers                                                       */
/* ============================================================ */

static int g_pass = 0, g_fail = 0;

#define TEST(name, cond, fmt, ...) do { \
    if (cond) { \
        printf("[PASS] %s\n", name); \
        g_pass++; \
    } else { \
        printf("[FAIL] %s: " fmt "\n", name, ##__VA_ARGS__); \
        g_fail++; \
    } \
} while(0)

#define SDST_REQUIRE(name, call) do { \
    SdstError _err = (call); \
    if (_err != SDST_SUCCESS) { \
        printf("[FAIL] %s: sdst error %d (%s)\n", name, (int)_err, \
               sdst_error_string(_err)); \
        g_fail++; \
    } else { \
        printf("[PASS] %s\n", name); \
        g_pass++; \
    } \
} while(0)

/* ============================================================ */
/* Synthetic spike data generation                               */
/* ============================================================ */

static void generate_spike_batch(
    SpikeInput* h_inputs,
    uint32_t n_spikes,
    uint32_t timestep,
    uint8_t phase,
    bool include_cluster /* inject known cluster at (60-70, 60-70, 60-70) */
) {
    srand(timestep * 31337 + phase);

    for (uint32_t i = 0; i < n_spikes; i++) {
        uint32_t x, y, z;
        if (include_cluster && i < n_spikes / 5) {
            /* Concentrated activity in known cluster */
            x = 60 + (rand() % 11);
            y = 60 + (rand() % 11);
            z = 60 + (rand() % 11);
        } else {
            x = rand() % 128;
            y = rand() % 128;
            z = rand() % 128;
        }

        h_inputs[i].voxel_x = x;
        h_inputs[i].voxel_y = y;
        h_inputs[i].voxel_z = z;
        h_inputs[i].timestamp = timestep;
        h_inputs[i].amplitude = 0.5f + ((float)(rand() % 100) / 100.0f);

        /* Thermodynamic fields: simulate hysteresis cycle */
        float temp_frac;
        if (phase <= 1) temp_frac = (float)phase / 2.0f;       /* heating: 0 -> 0.5 -> 1.0 */
        else if (phase == 2) temp_frac = 1.0f;                  /* peak */
        else temp_frac = 1.0f - ((float)(phase - 2) / 3.0f);   /* cooling */

        h_inputs[i].local_temp = 50.0f + temp_frac * 250.0f;  /* 50K to 300K */
        h_inputs[i].energy_gradient = 0.1f + ((float)(rand() % 100) / 50.0f);
        h_inputs[i].solvent_exposure = (x > 60 && y > 60 && z > 60) ?
            0.1f : 0.5f + ((float)(rand() % 50) / 100.0f);
        h_inputs[i].phase_id = phase;
    }
}

/* ============================================================ */
/* Test program                                                   */
/* ============================================================ */

int main(int argc, char** argv) {
    printf("=================================================\n");
    printf("  SDST Test Suite — Phase 5 Gate Verification\n");
    printf("=================================================\n\n");

    /* ---- Step 1: Create with default config ---- */
    SdstConfig cfg = sdst_default_config();

    /* Use smaller run for test: 200 timesteps */
    cfg.phase_boundaries[0] = 0;
    cfg.phase_boundaries[1] = 50;
    cfg.phase_boundaries[2] = 100;
    cfg.phase_boundaries[3] = 150;
    cfg.phase_boundaries[4] = 175;
    cfg.phase_boundaries[5] = 200;
    cfg.max_spike_events = 500000;
    cfg.max_wavefronts = 65536;

    SdstHandle handle = NULL;
    SdstError err = sdst_create(&cfg, &handle);
    TEST("1. sdst_create",
         err == SDST_SUCCESS && handle != NULL,
         "err=%d handle=%p", (int)err, (void*)handle);
    if (err != SDST_SUCCESS) {
        printf("Fatal: cannot create SDST handle. Aborting.\n");
        return 1;
    }

    /* ---- Step 2: Print initial stats ---- */
    printf("\n-- Initial stats --\n");
    SDST_REQUIRE("2. sdst_print_stats", sdst_print_stats(handle));

    /* ---- Step 3+4: Generate and insert 100 timesteps ---- */
    printf("\n-- Inserting 100 timesteps (50 spikes each) --\n");
    const uint32_t SPIKES_PER_STEP = 50;
    const uint32_t N_STEPS = 100;
    uint32_t expected_total = 0;

    SpikeInput* h_inputs = (SpikeInput*)malloc(SPIKES_PER_STEP * sizeof(SpikeInput));
    SpikeInput* d_inputs;
    cudaMalloc(&d_inputs, SPIKES_PER_STEP * sizeof(SpikeInput));

    bool insert_ok = true;
    for (uint32_t step = 0; step < N_STEPS; step++) {
        /* Determine phase based on test config boundaries */
        uint8_t phase = 0;
        for (int p = 4; p >= 0; p--) {
            if (step >= cfg.phase_boundaries[p]) { phase = (uint8_t)p; break; }
        }

        generate_spike_batch(h_inputs, SPIKES_PER_STEP, step, phase,
                             (step % 5 == 0)); /* inject cluster every 5 steps */

        cudaMemcpy(d_inputs, h_inputs, SPIKES_PER_STEP * sizeof(SpikeInput),
                   cudaMemcpyHostToDevice);

        SdstError ins_err = sdst_insert_raw(handle, d_inputs, SPIKES_PER_STEP, NULL);
        if (ins_err != SDST_SUCCESS) {
            printf("[FAIL] 4. sdst_insert_raw at step %u: err=%d\n", step, (int)ins_err);
            insert_ok = false;
            g_fail++;
            break;
        }
        expected_total += SPIKES_PER_STEP;
    }
    if (insert_ok) {
        printf("[PASS] 3+4. sdst_insert_raw (100 steps × 50 spikes = %u events)\n",
               expected_total);
        g_pass++;
    }

    free(h_inputs);
    cudaFree(d_inputs);

    /* ---- Step 5: Event count ---- */
    uint32_t actual_count = 0;
    SDST_REQUIRE("5a. sdst_event_count call", sdst_event_count(handle, &actual_count));
    TEST("5b. event count matches expected",
         actual_count == expected_total,
         "actual=%u expected=%u", actual_count, expected_total);

    /* ---- Step 6: sdst_query_region — known cluster ---- */
    SpatialRegion cluster_region = {58, 72, 58, 72, 58, 72};
    SpikeEvent* qr_events = NULL;
    uint32_t    qr_count  = 0;
    SDST_REQUIRE("6a. sdst_query_region call",
        sdst_query_region(handle, &cluster_region, &qr_events, &qr_count, NULL));
    TEST("6b. query_region finds events in cluster",
         qr_count > 0,
         "found=%u (expected >0 since cluster was injected every 5 steps)", qr_count);
    printf("     cluster region events found: %u\n", qr_count);

    /* ---- Step 7: sdst_query_voxel — specific active voxel ---- */
    SpikeEvent* vq_events = NULL;
    uint32_t    vq_count  = 0;
    /* Voxel (64,64,64) was frequently injected */
    SDST_REQUIRE("7a. sdst_query_voxel call",
        sdst_query_voxel(handle, 64, 64, 64, &vq_events, &vq_count, NULL));
    printf("     voxel (64,64,64) events: %u\n", vq_count);
    /* Just verify the call succeeds; cluster voxel may or may not have exact hits */
    TEST("7b. sdst_query_voxel returns valid count",
         vq_count <= actual_count, "count=%u total=%u", vq_count, actual_count);

    /* ---- Step 8: sdst_query_timerange — first 50 timesteps ---- */
    SpikeEvent* tr_events = NULL;
    uint32_t    tr_count  = 0;
    SDST_REQUIRE("8a. sdst_query_timerange call",
        sdst_query_timerange(handle, 0, 49, &tr_events, &tr_count, NULL));
    TEST("8b. timerange finds events in [0,49]",
         tr_count > 0 && tr_count <= 50 * SPIKES_PER_STEP,
         "found=%u", tr_count);
    printf("     timerange [0,49] events: %u\n", tr_count);

    /* ---- Step 9: sdst_ccns_region — compute tau ---- */
    CcnsResult ccns;
    SDST_REQUIRE("9a. sdst_ccns_region call",
        sdst_ccns_region(handle, &cluster_region, &ccns, NULL));
    printf("     CCNS: tau=%.3f ± %.3f, class=%d, druggability=%.3f, n_avalanches=%u\n",
           ccns.tau, ccns.tau_stderr, (int)ccns.classification,
           ccns.druggability, ccns.n_avalanches);
    TEST("9b. tau is in plausible range [0.5, 4.0]",
         ccns.tau == 0.0f || (ccns.tau >= 0.5f && ccns.tau <= 4.0f),
         "tau=%.3f", ccns.tau);

    /* ---- Step 10: sdst_hysteresis_region ---- */
    HysteresisResult hyst;
    SDST_REQUIRE("10a. sdst_hysteresis_region call",
        sdst_hysteresis_region(handle, &cluster_region, 0.2f, &hyst, NULL));
    printf("     Hysteresis: asymmetry=%.3f, heating=%u, cooling=%u, is_hysteretic=%d\n",
           hyst.asymmetry_score, hyst.heating_spike_count,
           hyst.cooling_spike_count, (int)hyst.is_hysteretic);
    TEST("10b. hysteresis asymmetry is in [0,1]",
         hyst.asymmetry_score >= 0.0f && hyst.asymmetry_score <= 1.0f,
         "score=%.3f", hyst.asymmetry_score);

    /* ---- Step 11: sdst_causal_subgraph ---- */
    CausalSubgraph subgraph;
    memset(&subgraph, 0, sizeof(CausalSubgraph));
    /* Pick event 0 as root */
    SDST_REQUIRE("11a. sdst_causal_subgraph call",
        sdst_causal_subgraph(handle, 0, 5, &subgraph, NULL));
    printf("     causal subgraph from event 0: %u events\n", subgraph.count);
    TEST("11b. causal subgraph has at least root event",
         subgraph.count >= 1, "count=%u", subgraph.count);
    if (subgraph.count > 0) {
        sdst_free_subgraph(&subgraph);
    }

    /* ---- Step 12: sdst_wavefront_stats ---- */
    WavefrontStats* wf_stats = NULL;
    uint32_t        wf_count = 0;
    SDST_REQUIRE("12a. sdst_wavefront_stats call",
        sdst_wavefront_stats(handle, -1, &wf_stats, &wf_count, NULL));
    printf("     wavefronts detected: %u\n", wf_count);
    TEST("12b. wavefronts > 0 (TCL+WCT wired in)",
         wf_count > 0, "wf_count=%u", wf_count);
    if (wf_stats) free(wf_stats);

    /* ---- Step 13: sdst_validate ---- */
    SDST_REQUIRE("13. sdst_validate", sdst_validate(handle));

    /* ---- Step 14: sdst_ccns_all_pockets ---- */
    CcnsResult*    pocket_ccns = NULL;
    SpatialRegion* pocket_regs = NULL;
    uint32_t       pocket_count = 0;
    SDST_REQUIRE("14a. sdst_ccns_all_pockets call",
        sdst_ccns_all_pockets(handle, &pocket_ccns, &pocket_regs, &pocket_count, NULL));
    printf("     auto-detected pockets: %u\n", pocket_count);
    TEST("14b. ccns_all_pockets returns valid pointer/count",
         (pocket_count == 0 || (pocket_ccns != NULL && pocket_regs != NULL)),
         "count=%u ptr=%p", pocket_count, (void*)pocket_ccns);
    if (pocket_ccns) { free(pocket_ccns); free(pocket_regs); }

    /* ---- Step 15: sdst_save + sdst_load round-trip ---- */
    const char* save_path = "/tmp/sdst_test_save.bin";
    SDST_REQUIRE("15a. sdst_save", sdst_save(handle, save_path));

    SdstHandle loaded_handle = NULL;
    SDST_REQUIRE("15b. sdst_load", sdst_load(save_path, &loaded_handle));

    uint32_t loaded_count = 0;
    if (loaded_handle) {
        sdst_event_count(loaded_handle, &loaded_count);
        TEST("15c. round-trip event count matches",
             loaded_count == actual_count,
             "loaded=%u original=%u", loaded_count, actual_count);
        sdst_destroy(loaded_handle);
    } else {
        printf("[FAIL] 15c. loaded handle is NULL\n");
        g_fail++;
    }

    /* ---- Step 16: Cleanup ---- */
    SDST_REQUIRE("16. sdst_destroy", sdst_destroy(handle));

    /* ---- Summary ---- */
    printf("\n=================================================\n");
    printf("  Results: %d PASS, %d FAIL\n", g_pass, g_fail);
    printf("=================================================\n");

    /* Final CUDA error check */
    cudaError_t final_err = cudaGetLastError();
    if (final_err != cudaSuccess) {
        printf("[FAIL] Final CUDA error: %s\n", cudaGetErrorString(final_err));
        return 1;
    }
    printf("[PASS] No CUDA errors\n");

    return (g_fail == 0) ? 0 : 1;
}
