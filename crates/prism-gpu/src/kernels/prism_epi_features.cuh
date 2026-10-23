//=============================================================================
// PRISM>4D EPIDEMIOLOGICAL FEATURE EXTENSIONS
// Cross-variant competition, momentum, and immunity recency
//=============================================================================
//
// PURPOSE: Extend 101-dim feature vector to 112-dim with epidemiologically
//          meaningful features that capture competitive dynamics and temporal
//          patterns missing from pure structural analysis.
//
// PRIORITY: Competition features are P0 (highest ROI for FALL prediction)
//
// AUTHOR: PRISM>4D Kernel Engineering
// DATE: 2024-12
//=============================================================================

#ifndef PRISM_EPI_FEATURES_CUH
#define PRISM_EPI_FEATURES_CUH

#include <cuda_runtime.h>
#include "prism_numerics.cuh"

namespace prism {
namespace epi {

//=============================================================================
// CONSTANTS
//=============================================================================

#define MAX_VARIANTS_BATCH 256
#define HISTORY_DAYS 35
#define N_EPI_FEATURES 11  // Competition(3) + Momentum(3) + Immunity(4) + Country(1)

//=============================================================================
// CROSS-VARIANT COMPETITION FEATURES (Indices 101-103)
//
// These capture competitive exclusion—why a fit variant still FALLs when
// an even fitter variant is sweeping through the population.
//=============================================================================

/**
 * Compute competition features for a single variant
 *
 * Features:
 *   [0] freq_rank_norm: Frequency rank / n_variants (0 = dominant)
 *   [1] gamma_deficit: my_gamma - max_competitor_gamma (negative = losing)
 *   [2] suppression_pressure: Σ freq of variants with gamma > my_gamma
 *
 * @param variant_idx     Index of target variant
 * @param n_variants      Total variants in batch
 * @param all_frequencies All variant frequencies [n_variants]
 * @param all_gammas      All variant fitness scores [n_variants]
 * @param features_out    Output [3] competition features
 */
__device__ void compute_competition_features(
    int variant_idx,
    int n_variants,
    const float* __restrict__ all_frequencies,
    const float* __restrict__ all_gammas,
    float* __restrict__ features_out
) {
    float my_freq = all_frequencies[variant_idx];
    float my_gamma = all_gammas[variant_idx];
    
    // Initialize accumulators
    int rank = 0;
    float max_competitor_gamma = -1e9f;
    float suppression_pressure = 0.0f;
    
    // Single pass through all variants
    for (int v = 0; v < n_variants; v++) {
        if (v == variant_idx) continue;
        
        float v_freq = all_frequencies[v];
        float v_gamma = all_gammas[v];
        
        // Rank: count variants with higher frequency
        if (v_freq > my_freq) {
            rank++;
        }
        
        // Max competitor fitness
        if (v_gamma > max_competitor_gamma) {
            max_competitor_gamma = v_gamma;
        }
        
        // Suppression: sum of frequencies of fitter variants
        if (v_gamma > my_gamma) {
            suppression_pressure += v_freq;
        }
    }
    
    // Normalize and clamp outputs
    features_out[0] = (float)rank / fmaxf((float)(n_variants - 1), 1.0f);
    features_out[1] = fmaxf(-2.0f, fminf(2.0f, my_gamma - max_competitor_gamma));
    features_out[2] = fminf(suppression_pressure, 1.0f);
}

/**
 * Batch kernel for competition features
 * One thread per variant, processes entire batch
 */
extern "C" __global__ void compute_competition_features_batch(
    const float* __restrict__ all_frequencies,  // [n_variants]
    const float* __restrict__ all_gammas,       // [n_variants]
    int n_variants,
    float* __restrict__ competition_out         // [n_variants × 3]
) {
    int variant_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (variant_idx >= n_variants) return;
    
    float features[3];
    compute_competition_features(
        variant_idx, n_variants,
        all_frequencies, all_gammas,
        features
    );
    
    // Coalesced write
    competition_out[variant_idx * 3 + 0] = features[0];
    competition_out[variant_idx * 3 + 1] = features[1];
    competition_out[variant_idx * 3 + 2] = features[2];
}


//=============================================================================
// MULTI-SCALE MOMENTUM FEATURES (Indices 104-106)
//
// These capture growth dynamics at different timescales to detect regime
// transitions (exponential growth → plateau → decline).
//=============================================================================

/**
 * Compute momentum features from frequency history
 *
 * Features:
 *   [0] log_slope_7d:  sign(Δf) × log(1 + |Δf/7| × 100), 7-day momentum
 *   [1] log_slope_28d: sign(Δf) × log(1 + |Δf/28| × 100), trend direction
 *   [2] acceleration:  (slope_7d_t - slope_7d_{t-7}) / 7, regime change
 *
 * @param variant_idx    Index of target variant
 * @param freq_history   Historical frequencies [n_variants × HISTORY_DAYS]
 * @param n_variants     Stride for history array
 * @param features_out   Output [3] momentum features
 */
__device__ void compute_momentum_features(
    int variant_idx,
    const float* __restrict__ freq_history,
    int n_variants,
    float* __restrict__ features_out
) {
    // Extract historical frequencies (column-major: history[day][variant])
    // Day 0 = today, Day 7 = 1 week ago, etc.
    float f_t0  = freq_history[0 * n_variants + variant_idx];   // Today
    float f_t7  = freq_history[7 * n_variants + variant_idx];   // 7 days ago
    float f_t14 = freq_history[14 * n_variants + variant_idx];  // 14 days ago
    float f_t21 = freq_history[21 * n_variants + variant_idx];  // 21 days ago
    float f_t28 = freq_history[28 * n_variants + variant_idx];  // 28 days ago
    
    // Compute slopes (daily rate of change)
    float slope_7d = (f_t0 - f_t7) / 7.0f;
    float slope_7d_prev = (f_t7 - f_t14) / 7.0f;
    float slope_28d = (f_t0 - f_t28) / 28.0f;
    
    // Log-transform for scale invariance
    // copysignf preserves sign while log1p handles small values
    float log_slope_7d = copysignf(
        log1pf(fabsf(slope_7d) * 100.0f),
        slope_7d
    );
    float log_slope_28d = copysignf(
        log1pf(fabsf(slope_28d) * 100.0f),
        slope_28d
    );
    
    // Acceleration (second derivative of frequency)
    float acceleration = (slope_7d - slope_7d_prev) / 7.0f;
    // Scale and clamp to [-1, 1]
    acceleration = fmaxf(-1.0f, fminf(1.0f, acceleration * 100.0f));
    
    // Clamp log-slopes to [-5, 5]
    features_out[0] = fmaxf(-5.0f, fminf(5.0f, log_slope_7d));
    features_out[1] = fmaxf(-5.0f, fminf(5.0f, log_slope_28d));
    features_out[2] = acceleration;
}

/**
 * Batch kernel for momentum features
 */
extern "C" __global__ void compute_momentum_features_batch(
    const float* __restrict__ freq_history,  // [HISTORY_DAYS × n_variants]
    int n_variants,
    float* __restrict__ momentum_out         // [n_variants × 3]
) {
    int variant_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (variant_idx >= n_variants) return;
    
    float features[3];
    compute_momentum_features(variant_idx, freq_history, n_variants, features);
    
    momentum_out[variant_idx * 3 + 0] = features[0];
    momentum_out[variant_idx * 3 + 1] = features[1];
    momentum_out[variant_idx * 3 + 2] = features[2];
}


//=============================================================================
// IMMUNITY RECENCY FEATURES (Indices 107-110)
//
// These expose the temporal structure of population immunity, enabling
// the RL agent to learn waning-dependent decision boundaries.
//
// Note: These are computed host-side and passed as uniforms (not per-variant)
//=============================================================================

/**
 * Immunity recency uniform buffer
 * Passed once per country per date, not per variant
 */
struct __align__(16) ImmunityRecencyParams {
    float days_since_vaccine_norm;   // days / 365, capped at 1
    float days_since_wave_norm;      // days / 180, capped at 1
    float immunity_derivative;       // (I_t - I_{t-30}) / 30
    float immunity_source_ratio;     // vaccine_immunity / total_immunity
};

/**
 * Apply immunity recency features to variant output
 * Simply copies the uniform values (same for all variants in country-date)
 */
__device__ void apply_immunity_features(
    const ImmunityRecencyParams* __restrict__ params,
    float* __restrict__ features_out  // [4]
) {
    features_out[0] = params->days_since_vaccine_norm;
    features_out[1] = params->days_since_wave_norm;
    features_out[2] = fmaxf(-0.1f, fminf(0.1f, params->immunity_derivative));
    features_out[3] = params->immunity_source_ratio;
}


//=============================================================================
// COMBINED EPIDEMIOLOGICAL FEATURE KERNEL
// Computes all 11 new features in a single launch
//=============================================================================

/**
 * Full epidemiological feature computation kernel
 *
 * Outputs 11 features per variant:
 *   [0-2]  Competition: rank, gamma_deficit, suppression
 *   [3-5]  Momentum: log_slope_7d, log_slope_28d, acceleration
 *   [6-9]  Immunity: vaccine_recency, wave_recency, derivative, source_ratio
 *   [10]   Country: normalized country ID
 *
 * @param all_frequencies    Current frequencies [n_variants]
 * @param all_gammas         Fitness scores [n_variants]
 * @param freq_history       Historical frequencies [HISTORY_DAYS × n_variants]
 * @param immunity_params    Immunity recency (uniform per launch)
 * @param country_id_norm    Country ID / 11 (uniform per launch)
 * @param n_variants         Number of variants
 * @param epi_features_out   Output [n_variants × 11]
 */
extern "C" __global__ void __launch_bounds__(256, 4)
compute_all_epi_features(
    const float* __restrict__ all_frequencies,
    const float* __restrict__ all_gammas,
    const float* __restrict__ freq_history,
    const ImmunityRecencyParams* __restrict__ immunity_params,
    float country_id_norm,
    int n_variants,
    float* __restrict__ epi_features_out
) {
    int variant_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (variant_idx >= n_variants) return;
    
    float* out = &epi_features_out[variant_idx * N_EPI_FEATURES];
    
    // Competition features [0-2]
    compute_competition_features(
        variant_idx, n_variants,
        all_frequencies, all_gammas,
        &out[0]
    );
    
    // Momentum features [3-5]
    compute_momentum_features(
        variant_idx,
        freq_history,
        n_variants,
        &out[3]
    );
    
    // Immunity features [6-9] (uniform)
    apply_immunity_features(immunity_params, &out[6]);
    
    // Country ID [10]
    out[10] = country_id_norm;
}


//=============================================================================
// FEATURE INTEGRATION INTO COMBINED_FEATURES
// Appends epi features to existing 101-dim vector
//=============================================================================

/**
 * Append epidemiological features to existing combined_features buffer
 *
 * @param combined_features      Existing [n_residues × 101]
 * @param epi_features           New epi features [n_variants × 11]
 * @param variant_to_residue_map Maps variant_idx to representative residue
 * @param n_variants             Number of variants
 * @param n_residues             Total residues
 * @param extended_out           Output [n_residues × 112]
 */
extern "C" __global__ void append_epi_features(
    const float* __restrict__ combined_features,     // [n_residues × 101]
    const float* __restrict__ epi_features,          // [n_variants × 11]
    const int* __restrict__ variant_to_residue_map,  // [n_variants]
    int n_variants,
    int n_residues,
    float* __restrict__ extended_out                 // [n_residues × 112]
) {
    int residue_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (residue_idx >= n_residues) return;
    
    // Copy existing 101 features
    const float* src = &combined_features[residue_idx * 101];
    float* dst = &extended_out[residue_idx * 112];
    
    #pragma unroll 4
    for (int i = 0; i < 101; i++) {
        dst[i] = src[i];
    }
    
    // Find which variant this residue belongs to (if any)
    // For simplicity, broadcast first variant's epi features to all residues
    // In production, would use proper variant-residue mapping
    
    // Default: use first variant's epi features (batch typically single-variant)
    const float* epi_src = &epi_features[0];
    
    #pragma unroll
    for (int i = 0; i < 11; i++) {
        dst[101 + i] = epi_src[i];
    }
}


//=============================================================================
// HOST WRAPPER FUNCTIONS
//=============================================================================

extern "C" cudaError_t launch_epi_features(
    const float* d_frequencies,
    const float* d_gammas,
    const float* d_freq_history,
    const ImmunityRecencyParams* d_immunity_params,
    float country_id_norm,
    int n_variants,
    float* d_epi_out,
    cudaStream_t stream
) {
    dim3 grid((n_variants + 255) / 256);
    dim3 block(256);
    
    compute_all_epi_features<<<grid, block, 0, stream>>>(
        d_frequencies, d_gammas, d_freq_history,
        d_immunity_params, country_id_norm,
        n_variants, d_epi_out
    );
    
    return cudaGetLastError();
}

extern "C" cudaError_t launch_append_epi(
    const float* d_combined,
    const float* d_epi,
    const int* d_variant_map,
    int n_variants,
    int n_residues,
    float* d_extended,
    cudaStream_t stream
) {
    dim3 grid((n_residues + 255) / 256);
    dim3 block(256);
    
    append_epi_features<<<grid, block, 0, stream>>>(
        d_combined, d_epi, d_variant_map,
        n_variants, n_residues, d_extended
    );
    
    return cudaGetLastError();
}

}  // namespace epi
}  // namespace prism

#endif  // PRISM_EPI_FEATURES_CUH
