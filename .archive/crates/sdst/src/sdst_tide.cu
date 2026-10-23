/**
 * SDST TIDE: Transfer entropy-Integrated Decomposed Energetics
 *
 * Plan C analytical layer. Computes causal ΔG decomposition by:
 *   1. Mapping spike events to residue identities
 *   2. Computing transfer entropy between residue spike trains and pocket activity
 *   3. Weighting energy contributions by causal influence
 *   4. Computing Fisher information (leverage) and KL divergence (reorganization cost)
 *
 * This runs on SDST event data - no additional simulation needed.
 */

#include "../include/sdst_internal.h"
#include <math.h>

/* ============================================================
 * Transfer Entropy computation
 *
 * TE(X→Y) = Σ p(y_{t+1}, y_t^k, x_t^l) * log[ p(y_{t+1}|y_t^k, x_t^l) / p(y_{t+1}|y_t^k) ]
 *
 * For spike trains, we use a binned approach:
 *   - Discretize time into bins of width Δt
 *   - Count spikes per bin per residue (source X)
 *   - Count spikes per bin in target pocket (target Y)
 *   - Compute conditional probabilities from co-occurrence counts
 *
 * History length k = l = 1 (Markov order 1) for computational tractability.
 * ============================================================ */

#define TE_BIN_WIDTH 100    /* timesteps per bin */
#define TE_MAX_BINS  1024   /* max time bins */

/**
 * Compute transfer entropy from source residue spike train to target pocket spike train.
 *
 * Uses binary spike trains (spike present/absent per bin) with k=l=1.
 *
 * Honest TE contract (post producer-repair-causal-truthing):
 *   - n_bins < 3: not computable → NaN
 *   - source or target saturated (all 0s or all 1s): TE is undefined
 *     because the marginal entropy collapses to 0; conditioning on a
 *     constant signal yields a constant log-ratio of 0. Returning 0.0f
 *     here was the root cause of "TE = 0.0 everywhere": with binary
 *     binning at TE_BIN_WIDTH=100 and ~89M events / ~1024 bins, pocket
 *     and active-residue trains saturate to all-1s ⇒ TE collapses to
 *     0 for every site×residue pair. NaN is the on-device contract for
 *     "not honestly computable"; consumer (sdst_bridge.rs / serde JSON)
 *     maps to None / null.
 *   - finite TE in [0, ∞): computed normally.
 */
static float compute_transfer_entropy(
    const uint8_t* source_train, /* [n_bins] binary: 1 = spike in this bin */
    const uint8_t* target_train, /* [n_bins] binary */
    uint32_t n_bins,
    TideMeasurementSupportV1* support = NULL
) {
    if (support) {
        support->n_bins = n_bins;
        support->unavailable_flags = n_bins < 3 ? 1u : 0u;
        for (uint32_t t = 0; t < n_bins; t++) {
            support->source_active_bins += source_train[t] != 0;
            support->target_active_bins += target_train[t] != 0;
        }
        for (uint32_t t = 0; t + 1 < n_bins; t++) {
            support->joint_counts[(target_train[t + 1] << 2) |
                                  (target_train[t] << 1) | source_train[t]]++;
        }
    }
    if (n_bins < 3) return nanf("");

    /* Saturation guard: if either train's occupancy is outside
     * (SAT_LO, SAT_HI), TE collapses (Markov-1 conditional ≈ marginal,
     * log-ratio → 0).  The partial-saturation case is the dominant
     * failure mode in PRISM-Therm: with TE_BIN_WIDTH=100 and ~89M
     * events / ~1024 bins, active residue and pocket trains push to
     * ~100% occupancy and TE drops to literal 0.0 across the board.
     * NaN is the on-device contract for "not honestly computable".
     *
     * Bounds (5%/95%) are conservative: well outside the regime where
     * Bernoulli-pair Markov-1 TE is statistically resolvable, and well
     * inside the regime where saturation is deterministic. */
    uint32_t source_active = 0, target_active = 0;
    for (uint32_t t = 0; t < n_bins; t++) {
        if (source_train[t]) source_active++;
        if (target_train[t]) target_active++;
    }
    const double SAT_LO = 0.05;
    const double SAT_HI = 0.95;
    double src_occ = (double)source_active / (double)n_bins;
    double tgt_occ = (double)target_active / (double)n_bins;
    if (support) {
        if (src_occ <= SAT_LO) support->unavailable_flags |= 2u;
        if (src_occ >= SAT_HI) support->unavailable_flags |= 4u;
        if (tgt_occ <= SAT_LO) support->unavailable_flags |= 8u;
        if (tgt_occ >= SAT_HI) support->unavailable_flags |= 16u;
    }
    if (src_occ <= SAT_LO || src_occ >= SAT_HI ||
        tgt_occ <= SAT_LO || tgt_occ >= SAT_HI) {
        return nanf("");
    }

    /* Count joint and marginal probabilities for:
     *   p(y_{t+1}, y_t, x_t)  - 2x2x2 = 8 entries
     *   p(y_{t+1}, y_t)       - 2x2 = 4 entries
     *   p(y_t, x_t)           - 2x2 = 4 entries
     *   p(y_t)                - 2 entries
     */
    uint32_t count_yy1x[2][2][2] = {{{0}}}; /* [y_{t+1}][y_t][x_t] */
    uint32_t count_yy1[2][2] = {{0}};       /* [y_{t+1}][y_t] */
    uint32_t count_yx[2][2] = {{0}};        /* [y_t][x_t] */
    uint32_t total = 0;

    for (uint32_t t = 0; t < n_bins - 1; t++) {
        uint8_t yt  = target_train[t];
        uint8_t yt1 = target_train[t + 1];
        uint8_t xt  = source_train[t];

        count_yy1x[yt1][yt][xt]++;
        count_yy1[yt1][yt]++;
        count_yx[yt][xt]++;
        total++;
    }

    if (total == 0) return nanf("");

    double te = 0.0;
    double inv_total = 1.0 / (double)total;

    for (int yt1 = 0; yt1 < 2; yt1++) {
        for (int yt = 0; yt < 2; yt++) {
            for (int xt = 0; xt < 2; xt++) {
                uint32_t n_joint = count_yy1x[yt1][yt][xt];
                if (n_joint == 0) continue;

                double p_joint = (double)n_joint * inv_total;

                /* p(y_{t+1} | y_t, x_t) = count(y_{t+1}, y_t, x_t) / count(y_t, x_t) */
                uint32_t n_yx = count_yx[yt][xt];
                if (n_yx == 0) continue;
                double p_cond_full = (double)n_joint / (double)n_yx;

                /* p(y_{t+1} | y_t) = count(y_{t+1}, y_t) / Σ_x count(y_t, x) */
                uint32_t n_yt = count_yx[yt][0] + count_yx[yt][1];
                if (n_yt == 0) continue;
                double p_cond_marginal = (double)count_yy1[yt1][yt] / (double)n_yt;

                if (p_cond_marginal > 0 && p_cond_full > 0) {
                    te += p_joint * log2(p_cond_full / p_cond_marginal);
                }
            }
        }
    }

    return (float)fmax(te, 0.0); /* TE is non-negative */
}

/**
 * Compute Fisher information for a residue's spike train.
 * Fisher info = Var[∂log p / ∂θ] ≈ sensitivity of pocket response to residue activity.
 *
 * Approximation: variance of the local transfer entropy across time windows.
 * High Fisher info = small changes in residue activity cause large changes in pocket response.
 * These are the leverage points for drug design.
 */
static float compute_fisher_info(
    const uint8_t* source_train,
    const uint8_t* target_train,
    uint32_t n_bins,
    uint32_t window_size, /* bins per window */
    TideMeasurementSupportV1* support = NULL
) {
    if (support) {
        support->variance_window_bins = window_size;
        support->variance_total_windows = n_bins / window_size;
    }
    if (n_bins < window_size * 2) return nanf("");

    uint32_t n_windows = n_bins / window_size;
    if (n_windows < 2) return nanf("");

    float* window_te = (float*)calloc(n_windows, sizeof(float));

    /* If any per-window TE is NaN (degenerate), Fisher info is also undefined.
     * Counting valid windows lets us distinguish "no signal" from "no data". */
    uint32_t valid_windows = 0;
    for (uint32_t w = 0; w < n_windows; w++) {
        uint32_t start = w * window_size;
        uint32_t end = start + window_size;
        if (end > n_bins) end = n_bins;
        uint32_t len = end - start;

        window_te[w] = compute_transfer_entropy(
            source_train + start,
            target_train + start,
            len
        );
        if (!isnan(window_te[w])) valid_windows++;
    }

    if (support) support->variance_valid_windows = valid_windows;
    if (valid_windows < 2) {
        free(window_te);
        return nanf("");
    }

    /* Compute variance over valid (non-NaN) windows only. */
    double mean = 0;
    for (uint32_t w = 0; w < n_windows; w++) {
        if (!isnan(window_te[w])) mean += window_te[w];
    }
    mean /= (double)valid_windows;

    double var = 0;
    for (uint32_t w = 0; w < n_windows; w++) {
        if (isnan(window_te[w])) continue;
        double diff = window_te[w] - mean;
        var += diff * diff;
    }
    var /= (double)(valid_windows - 1);

    free(window_te);
    return (float)var;
}

/**
 * Compute KL divergence between a residue's spike distribution in heating vs cooling.
 * This measures the conformational reorganization cost:
 *   D_KL(P_heating || P_cooling)
 *
 * High KL = the residue behaves very differently during heating vs cooling = high reorganization.
 */
static float compute_kl_divergence(
    const uint8_t* train,
    const PhaseId* phase_per_bin,
    uint32_t n_bins
) {
    /* Count spike rates in heating (phases 0,1) vs cooling (3,4) */
    uint32_t heat_spikes = 0, heat_bins = 0;
    uint32_t cool_spikes = 0, cool_bins = 0;

    for (uint32_t t = 0; t < n_bins; t++) {
        PhaseId p = phase_per_bin[t];
        if (p == 0 || p == 1) {
            heat_bins++;
            if (train[t]) heat_spikes++;
        } else if (p == 3 || p == 4) {
            cool_bins++;
            if (train[t]) cool_spikes++;
        }
    }

    if (heat_bins == 0 || cool_bins == 0) return 0.0f;

    /* Spike probability in each regime */
    double p_heat = (double)heat_spikes / (double)heat_bins;
    double p_cool = (double)cool_spikes / (double)cool_bins;

    /* Clamp to avoid log(0) */
    double eps = 1e-10;
    p_heat = fmax(p_heat, eps);
    p_cool = fmax(p_cool, eps);
    double q_heat = 1.0 - p_heat;
    double q_cool = 1.0 - p_cool;
    q_heat = fmax(q_heat, eps);
    q_cool = fmax(q_cool, eps);

    /* KL(heating || cooling) for Bernoulli */
    double kl = p_heat * log(p_heat / p_cool) + q_heat * log(q_heat / q_cool);

    return (float)fmax(kl, 0.0);
}

/* ============================================================
 * Public: TIDE decomposition
 * ============================================================ */

extern "C"
SdstError sdst_tide_decomposition_with_support_v1(
    SdstHandle handle,
    const SpatialRegion* pocket_region,
    const uint32_t* h_residue_map,   /* Host pointer, linear-voxel-indexed.
                                      * Size = grid_nx × grid_ny × grid_nz.
                                      * Index = x + y*grid_nx + z*grid_nx*grid_ny.
                                      * UINT32_MAX for empty (no-residue) voxels. */
    uint32_t n_residues,
    TideDecomposition** out_decomp,
    TideMeasurementSupportV1** out_support,
    uint32_t* out_count,
    void* stream
) {
    if (!handle || !pocket_region || !h_residue_map || !out_decomp || !out_count)
        return SDST_ERROR_INVALID_PARAM;
    *out_decomp = NULL;
    *out_count = 0;
    if (out_support) *out_support = NULL;
    SdstContext* ctx = handle;
    cudaStream_t s = stream ? (cudaStream_t)stream : 0;

    uint32_t grid_nx = ctx->config.grid_nx;
    uint32_t grid_ny = ctx->config.grid_ny;
    uint32_t grid_nz = ctx->config.grid_nz;

    uint32_t total_events;
    SDST_CUDA_CHECK(cudaMemcpy(&total_events, ctx->d_event_count,
                               sizeof(uint32_t), cudaMemcpyDeviceToHost));

    if (total_events == 0) {
        *out_decomp = NULL;
        *out_count = 0;
        return SDST_SUCCESS;
    }

    /* Copy events to host */
    SpikeEvent* h_events = (SpikeEvent*)malloc(total_events * sizeof(SpikeEvent));
    SDST_CUDA_CHECK(cudaMemcpyAsync(h_events, ctx->d_event_buffer,
                                     total_events * sizeof(SpikeEvent),
                                     cudaMemcpyDeviceToHost, s));
    SDST_CUDA_CHECK(cudaStreamSynchronize(s));

    /* Determine time binning */
    uint32_t max_ts = ctx->config.phase_boundaries[5];
    uint32_t n_bins = (max_ts + TE_BIN_WIDTH - 1) / TE_BIN_WIDTH;
    if (n_bins > TE_MAX_BINS) n_bins = TE_MAX_BINS;

    /* Build pocket spike train (target Y) */
    uint8_t* pocket_train = (uint8_t*)calloc(n_bins, sizeof(uint8_t));
    PhaseId* phase_per_bin = (PhaseId*)calloc(n_bins, sizeof(PhaseId));

    /* Assign phase per bin */
    for (uint32_t b = 0; b < n_bins; b++) {
        uint32_t t = b * TE_BIN_WIDTH;
        for (int p = 4; p >= 0; p--) {
            if (t >= ctx->config.phase_boundaries[p]) {
                phase_per_bin[b] = (PhaseId)p;
                break;
            }
        }
    }

    uint32_t target_binned_spikes = 0, target_outside_bin_range = 0;
    /* Fill pocket train */
    for (uint32_t i = 0; i < total_events; i++) {
        uint32_t vx, vy, vz;
        morton_decode(h_events[i].voxel, &vx, &vy, &vz);
        if (vx >= pocket_region->x_min && vx <= pocket_region->x_max &&
            vy >= pocket_region->y_min && vy <= pocket_region->y_max &&
            vz >= pocket_region->z_min && vz <= pocket_region->z_max) {
            uint32_t bin = h_events[i].timestamp / TE_BIN_WIDTH;
            if (bin < n_bins) {
                pocket_train[bin] = 1;
                target_binned_spikes++;
            } else target_outside_bin_range++;
        }
    }

    /* Build per-residue spike trains and accumulate energy in a single O(N) pass.
     * Previous implementation did hash-table probe per event (O(N×32)) plus
     * a second O(n_residues × N × 32) pass for energy — replaced with
     * dense linear-voxel-indexed lookup: Morton→(x,y,z)→linear→residue_id. */
    uint8_t** residue_trains = (uint8_t**)calloc(n_residues, sizeof(uint8_t*));
    uint32_t* residue_spike_counts = (uint32_t*)calloc(n_residues, sizeof(uint32_t));
    uint32_t* residue_outside_bin_range = (uint32_t*)calloc(n_residues, sizeof(uint32_t));
    uint32_t* residue_causal_counts = (uint32_t*)calloc(n_residues, sizeof(uint32_t));
    float*    residue_energy_sum = (float*)calloc(n_residues, sizeof(float));

    for (uint32_t r = 0; r < n_residues; r++) {
        residue_trains[r] = (uint8_t*)calloc(n_bins, sizeof(uint8_t));
    }

    /* Single-pass: map events → residues via dense linear-voxel-indexed map.
     * SPATIAL FILTER: Only process events within an expanded pocket region
     * (pocket + 2 voxel margin) so that residue spike trains are pocket-local.
     * Without this filter, globally-active residues dominate TE for every pocket. */
    uint32_t pr_x_min = pocket_region->x_min > 2 ? pocket_region->x_min - 2 : 0;
    uint32_t pr_x_max = pocket_region->x_max + 2 < grid_nx ? pocket_region->x_max + 2 : grid_nx - 1;
    uint32_t pr_y_min = pocket_region->y_min > 2 ? pocket_region->y_min - 2 : 0;
    uint32_t pr_y_max = pocket_region->y_max + 2 < grid_ny ? pocket_region->y_max + 2 : grid_ny - 1;
    uint32_t pr_z_min = pocket_region->z_min > 2 ? pocket_region->z_min - 2 : 0;
    uint32_t pr_z_max = pocket_region->z_max + 2 < grid_nz ? pocket_region->z_max + 2 : grid_nz - 1;

    for (uint32_t i = 0; i < total_events; i++) {
        MortonCode mc = h_events[i].voxel;
        uint32_t vx, vy, vz;
        morton_decode(mc, &vx, &vy, &vz);

        if (vx >= grid_nx || vy >= grid_ny || vz >= grid_nz) continue;

        /* Skip events outside expanded pocket region */
        if (vx < pr_x_min || vx > pr_x_max ||
            vy < pr_y_min || vy > pr_y_max ||
            vz < pr_z_min || vz > pr_z_max) continue;

        uint32_t linear = vx + vy * grid_nx + vz * grid_nx * grid_ny;
        uint32_t res_id = h_residue_map[linear];

        if (res_id == UINT32_MAX || res_id >= n_residues) continue;

        uint32_t bin = h_events[i].timestamp / TE_BIN_WIDTH;
        if (bin < n_bins) {
            residue_trains[res_id][bin] = 1;
            residue_spike_counts[res_id]++;
        } else residue_outside_bin_range[res_id]++;

        /* Accumulate energy gradient for mean computation */
        residue_energy_sum[res_id] += f16_bits_to_float(h_events[i].energy_gradient);

        /* Check strict pocket membership for causal count */
        if (vx >= pocket_region->x_min && vx <= pocket_region->x_max &&
            vy >= pocket_region->y_min && vy <= pocket_region->y_max &&
            vz >= pocket_region->z_min && vz <= pocket_region->z_max) {
            residue_causal_counts[res_id]++;
        }
    }

    /* Compute TIDE metrics for each active residue */
    *out_decomp = (TideDecomposition*)malloc(n_residues * sizeof(TideDecomposition));
    if (out_support) {
        *out_support = (TideMeasurementSupportV1*)calloc(n_residues, sizeof(TideMeasurementSupportV1));
    }
    if (!*out_decomp || (out_support && !*out_support)) {
        free(*out_decomp); *out_decomp = NULL;
        if (out_support) { free(*out_support); *out_support = NULL; }
        for (uint32_t r = 0; r < n_residues; r++) free(residue_trains[r]);
        free(residue_trains); free(residue_spike_counts);
        free(residue_causal_counts); free(residue_outside_bin_range);
        free(residue_energy_sum); free(pocket_train); free(phase_per_bin); free(h_events);
        return SDST_ERROR_OOM;
    }
    uint32_t active_count = 0;

    for (uint32_t r = 0; r < n_residues; r++) {
        if (residue_spike_counts[r] == 0) continue;

        TideDecomposition* td = &(*out_decomp)[active_count];
        td->residue_id = r;
        td->n_causal_spikes = residue_causal_counts[r];
        TideMeasurementSupportV1* support = out_support ? &(*out_support)[active_count] : NULL;
        if (support) {
            support->residue_id = r;
            support->bin_width_steps = TE_BIN_WIDTH;
            support->max_bins = TE_MAX_BINS;
            support->source_binned_spikes = residue_spike_counts[r];
            support->target_binned_spikes = target_binned_spikes;
            support->source_outside_bin_range = residue_outside_bin_range[r];
            support->target_outside_bin_range = target_outside_bin_range;
        }

        /* Transfer entropy: causal influence of this residue on pocket */
        td->transfer_entropy = compute_transfer_entropy(
            residue_trains[r], pocket_train, n_bins, support
        );
        if (support && !isfinite(td->transfer_entropy) && support->unavailable_flags == 0)
            support->unavailable_flags |= 32u;

        /* Fisher information: sensitivity / leverage */
        td->fisher_info = compute_fisher_info(
            residue_trains[r], pocket_train, n_bins, 10, support
        );

        /* KL divergence: conformational reorganization cost */
        td->kl_divergence = compute_kl_divergence(
            residue_trains[r], phase_per_bin, n_bins
        );

        /* Causal ΔG: TE-weighted free energy contribution.
         * Formula: -TE × log(1 + n_causal_spikes) × RT (kcal/mol at 300K).
         * TE measures information flow (bits), log(1+n) compresses the huge
         * dynamic range of spike counts (10 → 100K+), RT converts to energy.
         * Range: roughly -2 to 0 kcal/mol for active TRIGGER residues.
         *
         * NaN propagation: if TE is undefined (saturated trains) we cannot
         * honestly compute ΔG. Previously this multiplied through to -0.0,
         * which masquerades as "computed and small" — now NaN. Consumer
         * (sdst_bridge → JSON) maps to None / null. */
        float RT = 0.596f;  /* kcal/mol at 300K */
        if (isnan(td->transfer_entropy)) {
            td->causal_dG = nanf("");
        } else {
            td->causal_dG = -td->transfer_entropy
                             * logf(1.0f + (float)residue_causal_counts[r])
                             * RT;
        }

        active_count++;
    }

    *out_count = active_count;

    /* Cleanup */
    for (uint32_t r = 0; r < n_residues; r++) free(residue_trains[r]);
    free(residue_trains);
    free(residue_spike_counts);
    free(residue_causal_counts);
    free(residue_outside_bin_range);
    free(residue_energy_sum);
    free(pocket_train);
    free(phase_per_bin);
    free(h_events);

    return SDST_SUCCESS;
}

/* Preserve the original ABI and exact estimator path for existing clients. */
extern "C"
SdstError sdst_tide_decomposition(
    SdstHandle handle, const SpatialRegion* pocket_region,
    const uint32_t* h_residue_map, uint32_t n_residues,
    TideDecomposition** out_decomp, uint32_t* out_count, void* stream
) {
    return sdst_tide_decomposition_with_support_v1(handle, pocket_region,
        h_residue_map, n_residues, out_decomp, NULL, out_count, stream);
}
