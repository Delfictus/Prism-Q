/**
 * POLYCENTRIC IMMUNITY FIELD KERNEL
 *
 * Implements interference-based fitness computation where 10 epitope classes
 * act as immune pressure centers creating wave-like propagation patterns.
 *
 * Theory: Γ(x) = |Σᵢ Aᵢ · e^(i·φᵢ) · K(x, cᵢ)|²
 * - Constructive interference → variant RISES
 * - Destructive interference → variant FALLS
 */

#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// CONSTANTS
// ============================================================================

#define N_EPITOPE_CENTERS 10
#define N_PK_SCENARIOS 75
#define FEATURE_DIM 136
#define FRACTAL_ALPHA 1.5f  // Fractal dimension for kernel decay
#define MAX_FRACTAL_DEPTH 4

// Epitope class names (for reference):
// 0: Class 1 (RBD-A), 1: Class 2 (RBD-B), 2: Class 3 (RBD-C), 3: Class 4 (RBD-D)
// 4: Class 5 (S309), 5: Class 6 (CR3022), 6: NTD-1, 7: NTD-2, 8: NTD-3, 9: S2

// Pre-computed epitope center coordinates in 136-dim feature space
// These are the centroids of each epitope class's structural signature
__constant__ float c_epitope_centers[N_EPITOPE_CENTERS * FEATURE_DIM];

// Cross-reactivity matrix (10x10) - how much immunity to epitope i protects against j
__constant__ float c_cross_reactivity[N_EPITOPE_CENTERS * N_EPITOPE_CENTERS];

// PK parameters for 75 scenarios (t_max, t_half pairs)
__constant__ float c_pk_tmax[N_PK_SCENARIOS];
__constant__ float c_pk_thalf[N_PK_SCENARIOS];

// Wave propagation speed (how fast immune pressure spreads in feature space)
__constant__ float c_wave_speed = 0.1f;

// Damping coefficient (antibody decay effect on wave amplitude)
__constant__ float c_wave_damping = 0.05f;


// ============================================================================
// DEVICE FUNCTIONS
// ============================================================================

/**
 * Fractal distance kernel - NOT Gaussian!
 * K(r) = 1 / (1 + r^α) where α = fractal dimension
 *
 * This creates scale-invariant influence decay, matching the self-similar
 * nature of immune escape mutations.
 */
__device__ float fractal_kernel(float dist_sq, float alpha) {
    return 1.0f / (1.0f + powf(dist_sq, alpha / 2.0f));
}


/**
 * Compute squared distance from feature vector to epitope center
 */
__device__ float distance_to_center_sq(
    const float* features,    // 136-dim feature vector
    int center_idx            // Which epitope center (0-9)
) {
    float dist_sq = 0.0f;
    int offset = center_idx * FEATURE_DIM;

    #pragma unroll 8
    for (int d = 0; d < FEATURE_DIM; d++) {
        float diff = features[d] - c_epitope_centers[offset + d];
        dist_sq += diff * diff;
    }

    return dist_sq;
}


/**
 * Compute antibody concentration at time t using PK model
 * c(t) = (e^(-ke·t) - e^(-ka·t)) / (e^(-ke·tmax) - e^(-ka·tmax))
 */
__device__ float pk_concentration(float t, int pk_idx) {
    float tmax = c_pk_tmax[pk_idx];
    float thalf = c_pk_thalf[pk_idx];

    float ke = 0.693147f / thalf;  // ln(2) / thalf
    float ka = logf(ke * tmax / (ke * tmax - 0.693147f));

    float num = expf(-ke * t) - expf(-ka * t);
    float denom = expf(-ke * tmax) - expf(-ka * tmax);

    return fmaxf(0.0f, num / (denom + 1e-8f));
}


/**
 * CORE FUNCTION: Compute interference field at a point in feature space
 *
 * This is the heart of the polycentric model. Each epitope center acts as
 * a source of immune pressure waves. The waves interfere constructively
 * or destructively based on:
 *   - Distance to center (kernel amplitude)
 *   - Escape score (wave amplitude)
 *   - PK phase (temporal offset)
 *
 * Returns: |Σᵢ Aᵢ · e^(i·φᵢ) · K(x, cᵢ)|² (interference intensity)
 */
__device__ float compute_interference_field(
    const float* features,        // 136-dim structural features
    const float* escape_10d,      // Per-epitope escape scores [10]
    const float* pk_immunity,     // Current immunity level per PK scenario [75]
    int pk_idx,                   // Which PK scenario to use
    float time_since_infection    // Days since last major infection wave
) {
    float field_real = 0.0f;
    float field_imag = 0.0f;

    for (int c = 0; c < N_EPITOPE_CENTERS; c++) {
        // 1. Distance to this epitope center
        float dist_sq = distance_to_center_sq(features, c);

        // 2. Fractal kernel (scale-invariant decay)
        float kernel = fractal_kernel(dist_sq, FRACTAL_ALPHA);

        // 3. Wave amplitude = escape score × cross-reactivity sum
        float amplitude = escape_10d[c];

        // Modulate by cross-reactivity (how much other epitopes shield this one)
        float cross_shield = 0.0f;
        for (int j = 0; j < N_EPITOPE_CENTERS; j++) {
            if (j != c) {
                cross_shield += c_cross_reactivity[c * N_EPITOPE_CENTERS + j]
                              * (1.0f - escape_10d[j]);
            }
        }
        amplitude *= (1.0f - 0.5f * cross_shield);  // Partial shielding

        // 4. Phase from PK model (temporal offset of immune pressure)
        // High immunity → phase near 0 (wave arriving)
        // Low immunity → phase near π (wave passed)
        float immunity_level = pk_immunity[pk_idx];
        float phase = (1.0f - immunity_level) * 3.14159f;

        // 5. Additional phase from time since infection (wave propagation)
        phase += c_wave_speed * time_since_infection * sqrtf(dist_sq);

        // 6. Damping from antibody decay
        float damped_amplitude = amplitude * expf(-c_wave_damping * time_since_infection);

        // 7. Superposition (complex addition)
        field_real += damped_amplitude * kernel * cosf(phase);
        field_imag += damped_amplitude * kernel * sinf(phase);
    }

    // Return interference intensity (|field|²)
    return field_real * field_real + field_imag * field_imag;
}


/**
 * Compute interference field across ALL 75 PK scenarios
 * Returns the envelope (max, min, mean) for robust prediction
 */
__device__ void compute_interference_envelope(
    const float* features,
    const float* escape_10d,
    const float* pk_immunity_75,  // [75] immunity levels
    float time_since_infection,
    float* out_max,
    float* out_min,
    float* out_mean
) {
    float sum = 0.0f;
    float max_val = -1e9f;
    float min_val = 1e9f;

    for (int pk = 0; pk < N_PK_SCENARIOS; pk++) {
        float field = compute_interference_field(
            features, escape_10d, pk_immunity_75, pk, time_since_infection
        );

        sum += field;
        max_val = fmaxf(max_val, field);
        min_val = fminf(min_val, field);
    }

    *out_max = max_val;
    *out_min = min_val;
    *out_mean = sum / N_PK_SCENARIOS;
}


/**
 * Compute wave propagation features for Stage 11
 * These replace the static competition metrics with dynamic wave physics
 */
__device__ void compute_wave_features(
    const float* features,
    const float* escape_10d,
    const float* pk_immunity_75,
    const float* freq_history_7d,   // Last 7 days of frequency
    float current_freq,
    float time_since_infection,
    float* out_wave_features        // [6] output features
) {
    // F0: Wave amplitude (interference intensity)
    float field_max, field_min, field_mean;
    compute_interference_envelope(
        features, escape_10d, pk_immunity_75, time_since_infection,
        &field_max, &field_min, &field_mean
    );
    out_wave_features[0] = field_mean;

    // F1: Standing wave ratio (max/min) - high = strong interference pattern
    out_wave_features[1] = (field_min > 1e-6f) ? (field_max / field_min) : 10.0f;

    // F2: Phase velocity estimate (from frequency trajectory)
    float freq_diff = freq_history_7d[6] - freq_history_7d[0];  // 7-day change
    out_wave_features[2] = freq_diff / (7.0f * current_freq + 1e-6f);

    // F3: Wavefront distance (min distance to any epitope center)
    float min_dist = 1e9f;
    for (int c = 0; c < N_EPITOPE_CENTERS; c++) {
        float d = sqrtf(distance_to_center_sq(features, c));
        min_dist = fminf(min_dist, d);
    }
    out_wave_features[3] = min_dist;

    // F4: Constructive interference score (real part magnitude)
    // Positive = constructive (RISE), Negative = destructive (FALL)
    float real_sum = 0.0f;
    for (int c = 0; c < N_EPITOPE_CENTERS; c++) {
        float dist_sq = distance_to_center_sq(features, c);
        float kernel = fractal_kernel(dist_sq, FRACTAL_ALPHA);
        float phase = (1.0f - pk_immunity_75[37]) * 3.14159f;  // Use median PK
        real_sum += escape_10d[c] * kernel * cosf(phase);
    }
    out_wave_features[4] = real_sum;

    // F5: Field gradient magnitude (rate of change across feature space)
    // Approximated by variance across epitope contributions
    float var_sum = 0.0f;
    for (int c = 0; c < N_EPITOPE_CENTERS; c++) {
        float contrib = escape_10d[c] * fractal_kernel(
            distance_to_center_sq(features, c), FRACTAL_ALPHA
        );
        var_sum += (contrib - field_mean) * (contrib - field_mean);
    }
    out_wave_features[5] = sqrtf(var_sum / N_EPITOPE_CENTERS);
}


// ============================================================================
// MAIN KERNEL
// ============================================================================

/**
 * Batch kernel: Process multiple structures with polycentric immunity field
 *
 * Replaces Stage 9-10 in mega_fused_batch.cu with interference-based computation
 */
extern "C" __global__ void polycentric_immunity_kernel(
    // Input: Structure features (from Stages 1-8)
    const float* __restrict__ features_packed,      // [total_residues × 136]
    const int* __restrict__ residue_offsets,        // [n_structures]
    const int* __restrict__ n_residues_per_struct,  // [n_structures]

    // Input: Escape data
    const float* __restrict__ escape_10d_packed,    // [n_structures × 10]

    // Input: PK immunity data
    const float* __restrict__ pk_immunity_packed,   // [n_structures × 75]

    // Input: Temporal data
    const float* __restrict__ time_since_infection, // [n_structures]
    const float* __restrict__ freq_history_packed,  // [n_structures × 7]
    const float* __restrict__ current_freq,         // [n_structures]

    // Output: Enhanced features (Stage 9-10 replacement)
    float* __restrict__ output_features,            // [n_structures × 22]
    // 22 = 10 (epitope escape) + 6 (wave features) + 6 (envelope stats)

    // Dimensions
    int n_structures
) {
    int struct_idx = blockIdx.x;
    if (struct_idx >= n_structures) return;

    int tid = threadIdx.x;

    // Get structure bounds
    int res_offset = residue_offsets[struct_idx];
    int n_res = n_residues_per_struct[struct_idx];

    // Shared memory for structure-level aggregation
    __shared__ float s_mean_features[FEATURE_DIM];
    __shared__ float s_escape_10d[N_EPITOPE_CENTERS];
    __shared__ float s_pk_immunity[N_PK_SCENARIOS];
    __shared__ float s_freq_history[7];

    // Load escape and PK data (thread 0)
    if (tid == 0) {
        for (int i = 0; i < N_EPITOPE_CENTERS; i++) {
            s_escape_10d[i] = escape_10d_packed[struct_idx * N_EPITOPE_CENTERS + i];
        }
        for (int i = 0; i < N_PK_SCENARIOS; i++) {
            s_pk_immunity[i] = pk_immunity_packed[struct_idx * N_PK_SCENARIOS + i];
        }
        for (int i = 0; i < 7; i++) {
            s_freq_history[i] = freq_history_packed[struct_idx * 7 + i];
        }
    }

    // Compute mean features across residues (parallel reduction)
    for (int d = tid; d < FEATURE_DIM; d += blockDim.x) {
        float sum = 0.0f;
        for (int r = 0; r < n_res; r++) {
            sum += features_packed[(res_offset + r) * FEATURE_DIM + d];
        }
        s_mean_features[d] = sum / n_res;
    }
    __syncthreads();

    // Thread 0 computes final output
    if (tid == 0) {
        float t_inf = time_since_infection[struct_idx];
        float freq = current_freq[struct_idx];

        // Output offset
        int out_offset = struct_idx * 22;

        // Copy escape scores (features 0-9)
        for (int i = 0; i < N_EPITOPE_CENTERS; i++) {
            output_features[out_offset + i] = s_escape_10d[i];
        }

        // Compute wave features (features 10-15)
        float wave_features[6];
        compute_wave_features(
            s_mean_features, s_escape_10d, s_pk_immunity,
            s_freq_history, freq, t_inf, wave_features
        );
        for (int i = 0; i < 6; i++) {
            output_features[out_offset + 10 + i] = wave_features[i];
        }

        // Compute envelope statistics (features 16-21)
        float env_max, env_min, env_mean;
        compute_interference_envelope(
            s_mean_features, s_escape_10d, s_pk_immunity, t_inf,
            &env_max, &env_min, &env_mean
        );
        output_features[out_offset + 16] = env_max;
        output_features[out_offset + 17] = env_min;
        output_features[out_offset + 18] = env_mean;
        output_features[out_offset + 19] = env_max - env_min;  // Range
        output_features[out_offset + 20] = (env_max + env_min) / 2.0f;  // Midpoint
        output_features[out_offset + 21] = (env_mean - env_min) / (env_max - env_min + 1e-6f);  // Skew
    }
}


/**
 * Kernel to initialize epitope centers from training data
 * Run ONCE at startup to compute centroids
 */
extern "C" __global__ void init_epitope_centers(
    const float* __restrict__ all_features,     // [n_samples × 136]
    const int* __restrict__ epitope_labels,     // [n_samples] which epitope (0-9)
    const int* __restrict__ samples_per_epitope,// [10] count per class
    int n_samples,
    float* __restrict__ out_centers             // [10 × 136]
) {
    int epitope = blockIdx.x;
    int dim = threadIdx.x;

    if (epitope >= N_EPITOPE_CENTERS || dim >= FEATURE_DIM) return;

    float sum = 0.0f;
    int count = 0;

    for (int s = 0; s < n_samples; s++) {
        if (epitope_labels[s] == epitope) {
            sum += all_features[s * FEATURE_DIM + dim];
            count++;
        }
    }

    out_centers[epitope * FEATURE_DIM + dim] = (count > 0) ? (sum / count) : 0.0f;
}
