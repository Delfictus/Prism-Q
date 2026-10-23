//=============================================================================
// GAMMA ENVELOPE REDUCTION KERNEL (FIXED VERSION)
// GPU-Accelerated 75-PK Envelope Computation for VASIL Exact Metric
//=============================================================================
//
// Purpose: Parallel reduction of 75 PK immunity values → (min, max, mean) gamma
//
// CRITICAL FIX: weighted_avg must be computed PER-PK to match VASIL methodology!
//   - OLD (BUGGY): weighted_avg = single value computed from MEAN immunity
//   - NEW (CORRECT): weighted_avg_75pk = 75 values, one per PK combination
//
// This fix reduces envelope spread by ~2x and enables correct RISE/FALL decisions.
//
// Input:  immunity_75pk[n_samples × 75]  - All 75 PK immunity values per sample
//         weighted_avg_75pk[n_samples × 75] - Per-PK weighted avg susceptibility
// Output: gamma_min[n_samples]  - Minimum gamma across 75 PKs
//         gamma_max[n_samples]  - Maximum gamma across 75 PKs
//         gamma_mean[n_samples] - Mean gamma across 75 PKs
//
// Algorithm:
//   For each sample (variant_y, date):
//     For each PK ∈ {0..74}:
//       gamma[pk] = S_y[pk] / weighted_avg_S[pk] - 1.0
//     min = reduce_min(gamma[0..74])
//     max = reduce_max(gamma[0..74])
//     mean = reduce_sum(gamma[0..74]) / 75.0
//
// Grid Launch: (n_samples + 255) / 256 blocks × 256 threads
//
//=============================================================================

#include <cuda_runtime.h>
#include <cfloat>

#define N_PK_COMBINATIONS 75

//=============================================================================
// KERNEL: Compute Gamma Envelope (Min, Max, Mean) for 75 PK Scenarios
// FIXED: Now takes per-PK weighted_avg for correct envelope computation
//=============================================================================

extern "C" __global__ void compute_gamma_envelopes_batch(
    const double* __restrict__ d_immunity_75pk,       // [n_samples × 75] immunity values
    const double* __restrict__ d_weighted_avg_75pk,   // [n_samples × 75] PER-PK weighted avg (FIXED!)
    double* __restrict__ d_gamma_min,                 // [n_samples] OUTPUT: min gamma
    double* __restrict__ d_gamma_max,                 // [n_samples] OUTPUT: max gamma
    double* __restrict__ d_gamma_mean,                // [n_samples] OUTPUT: mean gamma
    double population,                                // Population size for susceptibility calculation
    int n_samples
) {
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (sample_idx >= n_samples) return;

    // Pointer to this sample's 75 immunity values
    const double* immunity_75 = &d_immunity_75pk[sample_idx * 75];
    
    // Pointer to this sample's 75 weighted_avg values (FIXED: per-PK!)
    const double* weighted_avg_75 = &d_weighted_avg_75pk[sample_idx * 75];

    // Initialize reduction accumulators
    double min_gamma = DBL_MAX;
    double max_gamma = -DBL_MAX;
    double sum_gamma = 0.0;
    int valid_count = 0;

    // Parallel reduction over 75 PK values (unrolled for performance)
    #pragma unroll 8
    for (int pk = 0; pk < N_PK_COMBINATIONS; pk++) {
        // FIXED: Use per-PK weighted_avg!
        double weighted_avg = __ldg(&weighted_avg_75[pk]);
        
        // Skip if denominator is too small
        if (weighted_avg < 1e-9) continue;

        // VASIL formula: γ = E[S_y] / weighted_avg_S - 1
        // where E[S_y] = Population - immunity (susceptibility)
        double immunity = __ldg(&immunity_75[pk]);
        double susceptibility = population - immunity;
        susceptibility = fmax(0.0, susceptibility);  // Clamp non-negative
        
        double gamma = (susceptibility / weighted_avg) - 1.0;

        // Update min/max
        min_gamma = fmin(min_gamma, gamma);
        max_gamma = fmax(max_gamma, gamma);

        // Accumulate for mean
        sum_gamma += gamma;
        valid_count++;
    }

    // Handle edge case of no valid PK combinations
    if (valid_count == 0) {
        d_gamma_min[sample_idx] = 0.0;
        d_gamma_max[sample_idx] = 0.0;
        d_gamma_mean[sample_idx] = 0.0;
        return;
    }

    // Compute mean
    double mean_gamma = sum_gamma / (double)valid_count;

    // Write outputs
    d_gamma_min[sample_idx] = min_gamma;
    d_gamma_max[sample_idx] = max_gamma;
    d_gamma_mean[sample_idx] = mean_gamma;
}

//=============================================================================
// KERNEL: Classify Envelope Decision (Rising/Falling/Undecided)
//=============================================================================

// Envelope decision codes
#define ENVELOPE_RISING 1
#define ENVELOPE_FALLING -1
#define ENVELOPE_UNDECIDED 0

extern "C" __global__ void classify_gamma_envelopes_batch(
    const double* __restrict__ d_gamma_min,   // [n_samples]
    const double* __restrict__ d_gamma_max,   // [n_samples]
    int* __restrict__ d_decision,             // [n_samples] OUTPUT: decision code
    int n_samples
) {
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (sample_idx >= n_samples) return;

    double min_val = __ldg(&d_gamma_min[sample_idx]);
    double max_val = __ldg(&d_gamma_max[sample_idx]);

    // VASIL decision rule from Extended Data Fig 6a:
    // - If entire envelope is positive → Rising
    // - If entire envelope is negative → Falling
    // - If envelope crosses zero → Undecided (EXCLUDE from accuracy)

    int decision;

    if (max_val < 0.0) {
        // Entire envelope negative
        decision = ENVELOPE_FALLING;
    } else if (min_val > 0.0) {
        // Entire envelope positive
        decision = ENVELOPE_RISING;
    } else {
        // Envelope crosses zero → cannot decide
        decision = ENVELOPE_UNDECIDED;
    }

    d_decision[sample_idx] = decision;
}

//=============================================================================
// KERNEL: Compute Weighted Average Susceptibility (FIXED: PER-PK)
//=============================================================================
//
// Purpose: For each (variant_y, date_t, pk), compute:
//          weighted_avg_S[pk] = Σ(freq_x * S_x[pk]) / Σ(freq_x)
//          where S_x[pk] = population - immunity_x[pk]
//
// CRITICAL FIX: Now computes 75 values per sample (one per PK)!
//   - OLD (BUGGY): Single weighted_avg computed from MEAN immunity
//   - NEW (CORRECT): 75 weighted_avg values, each using SAME PK as numerator
//
// This ensures gamma[pk] = S_y[pk] / weighted_avg[pk] - 1 uses consistent PK.
//
// Input:  d_immunity_75pk[75 × n_variants × n_days] - All 75 PK immunity values
//         d_frequencies[n_variants × max_history_days] - Variant frequencies
//         population - Total population size
//
// Output: d_weighted_avg_75pk[n_variants × n_days × 75] - Per-PK weighted avg!
//
// Grid Launch: (n_samples × 75 + 255) / 256 blocks × 256 threads
//   where n_samples = n_variants × n_eval_days
//=============================================================================

extern "C" __global__ void compute_weighted_avg_susceptibility(
    const double* __restrict__ d_immunity_75pk,  // [75 × n_variants × n_days] immunity values
    const float* __restrict__ d_frequencies,     // [n_variants × max_history_days] frequencies
    double* __restrict__ d_weighted_avg_75pk,    // [n_variants × n_days × 75] OUTPUT (FIXED!)
    double population,
    int n_variants,
    int n_eval_days,
    int max_history_days,
    int eval_start_offset
) {
    // Each thread handles one (variant_y, date_t, pk) triplet
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_samples = n_variants * n_eval_days;
    
    if (global_idx >= n_samples * N_PK_COMBINATIONS) return;
    
    // Decode global index → (sample_idx, pk)
    int sample_idx = global_idx / N_PK_COMBINATIONS;
    int pk = global_idx % N_PK_COMBINATIONS;
    
    // Decode sample index → (y_idx, t_idx)
    int y_idx = sample_idx / n_eval_days;
    int t_idx = sample_idx % n_eval_days;
    
    // Actual calendar date index (including pre-evaluation history)
    int date_idx = t_idx + eval_start_offset;
    
    if (date_idx >= max_history_days) {
        d_weighted_avg_75pk[sample_idx * 75 + pk] = population * 0.5;
        return;
    }
    
    // Compute weighted average across all active variants at this date FOR THIS PK
    double weighted_sum = 0.0;
    double total_freq = 0.0;
    
    for (int x_idx = 0; x_idx < n_variants; x_idx++) {
        // Get frequency of variant x at this date
        int freq_offset = x_idx * max_history_days + date_idx;
        float freq_x = __ldg(&d_frequencies[freq_offset]);
        
        if (freq_x < 1e-9) continue;  // Skip inactive variants
        
        // Get immunity for variant x at this date FOR THIS SPECIFIC PK
        // immunity_75pk layout: [pk][variant][day]
        int immunity_offset = (pk * n_variants * n_eval_days) + (x_idx * n_eval_days) + t_idx;
        double immunity_x = __ldg(&d_immunity_75pk[immunity_offset]);
        
        // Susceptibility = population - immunity (for THIS PK)
        double susceptibility_x = population - immunity_x;
        susceptibility_x = fmax(0.0, susceptibility_x);  // Clamp to non-negative
        
        // Accumulate weighted sum
        weighted_sum += (double)freq_x * susceptibility_x;
        total_freq += (double)freq_x;
    }
    
    // Compute weighted average
    double weighted_avg_s;
    if (total_freq > 1e-9 && weighted_sum > 1e-9) {
        weighted_avg_s = weighted_sum / total_freq;
    } else {
        // No active variants → fallback
        weighted_avg_s = population * 0.5;
    }
    
    // Write output for this sample and PK
    d_weighted_avg_75pk[sample_idx * 75 + pk] = weighted_avg_s;
}

//=============================================================================
// NOTE: Host wrapper functions removed - not needed for PTX compilation
// Rust FFI calls kernels directly using unmangled names
// This matches the pattern in prism_immunity_accurate.cu
//=============================================================================
