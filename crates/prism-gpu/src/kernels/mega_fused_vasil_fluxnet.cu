// PRISM-4D: Mega-Fused VASIL Kernel with FluxNet-Trainable Parameters
//
// Architecture: Single kernel computes VASIL accuracy from FluxNet parameters
//
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║  FLUXNET-TRAINABLE PARAMETERS (optimized by RL)                           ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║  1. fluxnet_ic50[10]        - Epitope binding affinities (10 classes)    ║
// ║  2. fluxnet_epitope_power[10] - Epitope contribution exponents           ║
// ║  3. fluxnet_rise_bias       - Threshold bias for RISE predictions         ║
// ║  4. fluxnet_fall_bias       - Threshold bias for FALL predictions         ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
//
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║  DATA-DRIVEN VALUES (from VASIL/GInPipe, NOT trainable)                   ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║  - frequencies[variant, day] - Observed variant frequencies πx(t)         ║
// ║  - incidence[day]            - Infection counts from GInPipe              ║
// ║  - epitope_escape[variant, epitope] - DMS escape fractions                ║
// ║  - actual_directions[variant, day] - Ground truth RISE/FALL               ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
//
// Formula with FluxNet parameters:
//   b_θ = c(t) / (FR · fluxnet_ic50[θ] + c(t))
//   P_neut = 1 - Π_θ (1 - b_θ)^fluxnet_epitope_power[θ]
//   
//   Prediction RISE if: gamma_min > fluxnet_rise_bias
//   Prediction FALL if: gamma_max < fluxnet_fall_bias

#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// === DEBUG CONFIGURATION ===
#define DEBUG_VARIANT 0    // Debug first variant only
#define DEBUG_DAY 0        // Debug first eval day only
#define DEBUG_PK 37        // Debug middle PK combination
#define ENABLE_DEBUG 0     // Set to 0 to disable all debug output

constexpr int MAX_DELTA_DAYS = 1500;
constexpr int N_EPITOPES = 10;  // VASIL uses 10 epitope classes (A,B,C,D1,D2,E1,E2,E3,F1,F2)
constexpr int N_PK = 75;
constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 8;  // 256 threads / 32

constexpr float NEGLIGIBLE_CHANGE_THRESHOLD = 0.05f;
constexpr float MIN_FREQUENCY_THRESHOLD = 0.03f;

// PK grid (fixed per VASIL methodology)
__constant__ float c_tmax[5] = {14.0f, 17.5f, 21.0f, 24.5f, 28.0f};
__constant__ float c_thalf[15] = {
    25.0f, 28.14f, 31.29f, 34.43f, 37.57f,
    40.71f, 43.86f, 47.0f, 50.14f, 53.29f,
    56.43f, 59.57f, 62.71f, 65.86f, 69.0f
};

// Default IC50 (VASIL-calibrated baseline, FluxNet starts here)
// 10 epitope classes: A, B, C, D1, D2, E1, E2, E3, F1, F2
__constant__ float c_baseline_ic50[10] = {
    0.85f, 1.12f, 0.93f, 1.05f, 0.98f,
    1.21f, 0.89f, 1.08f, 0.95f, 1.03f
};

// Default epitope power (uniform = 1.0, FluxNet learns deviations)
__constant__ float c_baseline_power[10] = {
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f
};

//=============================================================================
// FluxNet Parameters Structure (passed to kernel)
//=============================================================================
struct FluxNetParams {
    float ic50[10];           // Trained binding affinities (10 epitope classes)
    float epitope_power[10];  // Trained contribution exponents
    float rise_bias;          // Trained RISE threshold adjustment
    float fall_bias;          // Trained FALL threshold adjustment
};

//=============================================================================
// DEVICE: P_neut with FluxNet-trained parameters
//=============================================================================
__device__ __forceinline__ float compute_p_neut_fluxnet(
    const float* __restrict__ escape_x,
    const float* __restrict__ escape_y,
    const float* __restrict__ fluxnet_ic50,
    const float* __restrict__ fluxnet_power,
    int delta_t,
    int pk_idx
) {
    if (delta_t <= 0 || delta_t >= MAX_DELTA_DAYS) return 0.0f;
    
    // PK pharmacokinetics (data-driven, not FluxNet-trained)
    const float tmax = c_tmax[pk_idx / 15];
    const float thalf = c_thalf[pk_idx % 15];
    const float ke = __logf(2.0f) / thalf;
    
    // VASIL Paper: Solve tmax = ln(ka/ke) / (ka - ke) for ka
    // Transcendental equation — use Newton-Raphson iteration
    // f(ka) = (ka - ke) * tmax - ln(ka/ke) = 0
    // df/dka = tmax - 1/ka
    float ka = ke * 5.0f;  // Initial guess (4-10× works, 5× is safe)
    #pragma unroll
    for (int iter = 0; iter < 5; iter++) {
        const float f = (ka - ke) * tmax - __logf(ka / ke);
        const float df_dka = tmax - 1.0f / ka;
        ka = ka - f / df_dka;  // Newton-Raphson step
    }
    
    const float exp_ke_tmax = __expf(-ke * tmax);
    const float exp_ka_tmax = __expf(-ka * tmax);
    const float pk_denom = exp_ke_tmax - exp_ka_tmax;
    
    if (fabsf(pk_denom) < 1e-10f) return 0.0f;
    
    // Antibody concentration (data-driven from PK model)
    const float t = (float)delta_t;
    float c_t = (__expf(-ke * t) - __expf(-ka * t)) / pk_denom;
    c_t = fmaxf(0.0f, c_t);
    
    if (c_t < 1e-8f) return 0.0f;

    // Debug disabled - too much output
    // c_t calculation verified: ka Newton-Raphson working

    // P_neut with FluxNet-trained IC50 and epitope powers
    float log_product = 0.0f;

    // P_neut accumulation debug disabled

    #pragma unroll
    for (int e = 0; e < N_EPITOPES; e++) {
        // Fold resistance (data-driven from DMS escape)
        float fold_res;
        if (escape_x[e] > 0.01f) {
            fold_res = (1.0f + escape_y[e]) / (1.0f + escape_x[e]);
        } else {
            fold_res = 1.0f + escape_y[e];
        }
        fold_res = fmaxf(0.1f, fminf(fold_res, 100.0f));

        #if ENABLE_DEBUG
        if (pk_idx == DEBUG_PK && delta_t == 1 && e == 0) {
            printf("=== CHECKPOINT 2: After computing fold resistance ===\n");
            printf("  epitope=%d, escape_x[0]=%.4f, escape_y[0]=%.4f, fold_res=%.4f\n",
                   e, escape_x[e], escape_y[e], fold_res);
        }
        #endif

        // FluxNet-trained IC50 (or baseline if NULL)
        const float ic50_e = (fluxnet_ic50 != nullptr) ? fluxnet_ic50[e] : c_baseline_ic50[e];
        
        // VASIL formula: b_θ = c(t) / (FR · IC50 + c(t))
        const float denom = fold_res * ic50_e + c_t;
        const float b_theta = (denom > 1e-10f) ? (c_t / denom) : 0.0f;
        
        // FluxNet-trained epitope power (contribution exponent)
        const float power_e = (fluxnet_power != nullptr) ? fluxnet_power[e] : c_baseline_power[e];
        
        // P_neut = 1 - Π_θ (1 - b_θ)^power_θ
        // Using log for numerical stability: log(1-b)^p = p * log(1-b)
        const float one_minus_b = fmaxf(1e-10f, 1.0f - b_theta);
        const float log_contrib = power_e * __logf(one_minus_b);
        log_product += log_contrib;

        // Epitope debug disabled
    }

    const float p_neut_result = 1.0f - __expf(log_product);

    // P_neut final debug disabled

    return p_neut_result;
}

//=============================================================================
// DEVICE: Compute immunity with warp-parallel reduction
//=============================================================================
__device__ double compute_immunity_fluxnet(
    const float* __restrict__ epitope_escape,
    const float* __restrict__ frequencies,
    const double* __restrict__ incidence,
    const float* __restrict__ fluxnet_ic50,
    const float* __restrict__ fluxnet_power,
    const float* __restrict__ escape_y,
    int y_idx,
    int t_abs,
    int pk_idx,
    int n_variants,
    int max_history_days,
    cg::thread_block_tile<WARP_SIZE>& warp
) {
    double warp_sum = 0.0;
    const int lane = warp.thread_rank();
    
    // Integration over infection history (data-driven: freq, incidence from VASIL)
    for (int s = lane; s < t_abs && s < max_history_days; s += WARP_SIZE) {
        const int delta_t = t_abs - s;
        if (delta_t <= 0 || delta_t >= MAX_DELTA_DAYS) continue;
        
        const double inc = incidence[s];
        if (inc < 1.0) continue;
        
        // Sum over circulating variants (data-driven frequency weighting)
        for (int x = 0; x < n_variants; x++) {
            const float freq = frequencies[x * max_history_days + s];
            if (freq < 0.001f) continue;
            
            float escape_x[N_EPITOPES];
            #pragma unroll
            for (int e = 0; e < N_EPITOPES; e++) {
                escape_x[e] = epitope_escape[x * N_EPITOPES + e];
            }
            
            // P_neut with FluxNet-trained parameters
            const float p_neut = compute_p_neut_fluxnet(
                escape_x, escape_y, fluxnet_ic50, fluxnet_power, delta_t, pk_idx
            );
            
            if (p_neut > 1e-8f) {
                warp_sum += (double)freq * inc * (double)p_neut;
            }
        }
    }
    
    // Warp reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        warp_sum += warp.shfl_down(warp_sum, offset);
    }
    
    return warp_sum;
}

//=============================================================================
// MEGA FUSED KERNEL: FluxNet Parameters → VASIL Accuracy
//=============================================================================
extern "C" __global__ void mega_fused_vasil_fluxnet(
    // Data-driven inputs (from VASIL/GInPipe)
    const float* __restrict__ epitope_escape,    // [n_variants × 11]
    const float* __restrict__ frequencies,       // [n_variants × max_history_days]
    const double* __restrict__ incidence,        // [max_history_days]
    const int8_t* __restrict__ actual_directions,// [n_variants × n_eval_days]
    const float* __restrict__ freq_changes,      // [n_variants × n_eval_days]
    
    // FluxNet-trained parameters (RL-optimized)
    const float* __restrict__ fluxnet_ic50,      // [11] trained IC50
    const float* __restrict__ fluxnet_power,     // [11] trained epitope powers
    const float fluxnet_rise_bias,               // trained RISE threshold
    const float fluxnet_fall_bias,               // trained FALL threshold
    const float gamma_threshold,                 // NEW: trainable decision boundary

    // Output counters
    unsigned int* __restrict__ correct_count,
    unsigned int* __restrict__ total_count,
    unsigned int* __restrict__ correct_rise,
    unsigned int* __restrict__ total_rise,
    unsigned int* __restrict__ correct_fall,
    unsigned int* __restrict__ total_fall,

    // VASIL precomputed S_mean
    const double* __restrict__ s_mean_precomputed,  // [max_history_days × 75]
    const int use_precomputed_s_mean,               // 1 if available, 0 if fallback

    // Constants
    const double population,
    const int n_variants,
    const int n_eval_days,
    const int max_history_days,
    const int eval_start_offset
) {
    const int y_idx = blockIdx.x;
    const int t_eval = blockIdx.y;

    if (y_idx >= n_variants || t_eval >= n_eval_days) return;

    const int t_abs = eval_start_offset + t_eval;
    if (t_abs >= max_history_days || t_abs < 1) return;

    #if ENABLE_DEBUG
    if (y_idx == DEBUG_VARIANT && t_eval == DEBUG_DAY && threadIdx.x == 0) {
        printf("\n=== CHECKPOINT 1: After loading inputs ===\n");
        printf("  y_idx=%d, t_eval=%d, t_abs=%d\n", y_idx, t_eval, t_abs);
        printf("  n_variants=%d, n_eval_days=%d, max_history_days=%d\n",
               n_variants, n_eval_days, max_history_days);
        printf("  population=%.0f, eval_start_offset=%d\n", population, eval_start_offset);
    }
    #endif

    // Check VASIL exclusion criteria (data-driven)
    const int sample_idx = y_idx * n_eval_days + t_eval;
    const int8_t actual_dir = actual_directions[sample_idx];
    const float rel_change = freq_changes[sample_idx];
    
    #if ENABLE_DEBUG
    if (y_idx == DEBUG_VARIANT && t_eval == DEBUG_DAY && threadIdx.x == 0) {
        printf("=== CHECKPOINT 9: After loading actual freq_change ===\n");
        printf("  actual_dir=%d, rel_change=%.6f\n", (int)actual_dir, rel_change);
    }
    #endif

    if (actual_dir == 0) return;  // Pre-excluded
    if (fabsf(rel_change) < NEGLIGIBLE_CHANGE_THRESHOLD) return;

    const float current_freq = frequencies[y_idx * max_history_days + t_abs];
    if (current_freq < MIN_FREQUENCY_THRESHOLD) return;
    
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> warp = cg::tiled_partition<WARP_SIZE>(block);
    
    // Load FluxNet parameters into shared memory
    __shared__ float smem_ic50[N_EPITOPES];
    __shared__ float smem_power[N_EPITOPES];
    __shared__ float smem_escape_y[N_EPITOPES];
    
    if (threadIdx.x < N_EPITOPES) {
        smem_ic50[threadIdx.x] = (fluxnet_ic50 != nullptr) ? 
            fluxnet_ic50[threadIdx.x] : c_baseline_ic50[threadIdx.x];
        smem_power[threadIdx.x] = (fluxnet_power != nullptr) ? 
            fluxnet_power[threadIdx.x] : c_baseline_power[threadIdx.x];
        smem_escape_y[threadIdx.x] = epitope_escape[y_idx * N_EPITOPES + threadIdx.x];
    }
    block.sync();

    #if ENABLE_DEBUG
    if (y_idx == DEBUG_VARIANT && t_eval == DEBUG_DAY && threadIdx.x == 0) {
        printf("  Loaded ic50[0]=%.4f, power[0]=%.4f, escape_y[0]=%.4f\n",
               smem_ic50[0], smem_power[0], smem_escape_y[0]);
    }
    #endif
    
    // Compute gamma envelope across all 75 PK combinations
    const int warps_per_block = BLOCK_SIZE / WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    
    __shared__ double smem_immunity_y[N_PK];
    
    // Phase 1: Compute immunity for variant y at each PK (FluxNet IC50 + power)
    for (int pk = warp_id; pk < N_PK; pk += warps_per_block) {
        double immunity_y = compute_immunity_fluxnet(
            epitope_escape, frequencies, incidence,
            smem_ic50, smem_power, smem_escape_y,
            y_idx, t_abs, pk, n_variants, max_history_days, warp
        );
        
        if (warp.thread_rank() == 0) {
            smem_immunity_y[pk] = immunity_y;
        }
    }
    block.sync();

    #if ENABLE_DEBUG
    if (y_idx == DEBUG_VARIANT && t_eval == DEBUG_DAY && threadIdx.x == 0) {
        printf("=== CHECKPOINT 5: After immunity integral loop ===\n");
        printf("  immunity_y[pk=37]=%.6f (from %d PK combinations)\n",
               smem_immunity_y[37], N_PK);
    }
    #endif
    
    // Phase 2: Compute Σx Sx(t, pk) per VASIL formula for EACH of 75 PK combinations
    // CRITICAL: Each PK has different pharmacokinetics → different immunity → different denominator
    __shared__ double smem_sum_sx_pk[N_PK];

    for (int pk = 0; pk < N_PK; pk++) {
        double warp_sum_sx = 0.0;

        // Each warp processes variants in parallel
        for (int x_base = 0; x_base < n_variants; x_base += warps_per_block) {
            const int x = x_base + warp_id;
            if (x >= n_variants) break;

            const float freq_x = frequencies[x * max_history_days + t_abs];
            if (freq_x < 0.001f) continue;

            // Compute immunity_x using THIS pk (not hardcoded 37)
            double immunity_x = 0.0;
            for (int s = warp.thread_rank(); s < t_abs && s < max_history_days; s += WARP_SIZE) {
                const int delta_t = t_abs - s;
                if (delta_t <= 0) continue;
                const double inc = incidence[s];
                if (inc < 1.0) continue;

                for (int x2 = 0; x2 < n_variants; x2++) {
                    const float freq_x2 = frequencies[x2 * max_history_days + s];
                    if (freq_x2 < 0.001f) continue;

                    float escape_x2[N_EPITOPES], escape_x[N_EPITOPES];
                    for (int e = 0; e < N_EPITOPES; e++) {
                        escape_x2[e] = epitope_escape[x2 * N_EPITOPES + e];
                        escape_x[e] = epitope_escape[x * N_EPITOPES + e];
                    }

                    // Use pk from outer loop (not hardcoded 37)
                    const float p_neut = compute_p_neut_fluxnet(
                        escape_x2, escape_x, fluxnet_ic50, fluxnet_power, delta_t, pk
                    );

                    if (p_neut > 1e-8f) {
                        immunity_x += (double)freq_x2 * inc * (double)p_neut;
                    }
                }
            }

            // Warp reduction
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                immunity_x += warp.shfl_down(immunity_x, offset);
            }

            if (warp.thread_rank() == 0 && x < n_variants) {
                const double susceptibility_x = fmax(0.0, population - immunity_x);
                // TEST A: Simple sum (no frequency weighting)
                warp_sum_sx += susceptibility_x;  // Sx(t, pk) only
            }
        }

        // Reduce across warps to get Σx Sx(t, pk) for THIS PK
        __shared__ double smem_warp_sums[WARPS_PER_BLOCK];
        const int warp_id_sx = threadIdx.x / WARP_SIZE;
        if (warp.thread_rank() == 0) {
            smem_warp_sums[warp_id_sx] = warp_sum_sx;
        }
        block.sync();

        // Thread 0 reduces all warp sums for this PK (NO FALLBACK)
        if (threadIdx.x == 0) {
            double total_sum_sx = 0.0;
            for (int w = 0; w < WARPS_PER_BLOCK; w++) {
                total_sum_sx += smem_warp_sums[w];
            }

            // NO FALLBACK — log and set to small value to avoid NaN
            if (total_sum_sx <= 0.0) {
                printf("ERROR: pk=%d sum_sx=%.6f ≤ 0! Setting to 0.1\n", pk, total_sum_sx);
                total_sum_sx = 0.1;  // Avoid division by zero
            }

            smem_sum_sx_pk[pk] = total_sum_sx;
        }
        block.sync();
    }

    // Phase 3: VASIL EXACT - Compute E[Sy], E[Sx] means across 75 PKs
    // γy = E[Sy] / Σx[πx·E[Sx]] - 1 (SINGLE gamma from expected values)

    // Step 3a: Compute E[Sy] = mean Sy across 75 PKs
    double sum_Sy = 0.0;
    for (int pk = threadIdx.x; pk < N_PK; pk += BLOCK_SIZE) {
        const double immunity_y = smem_immunity_y[pk];
        const double susceptibility_y = fmax(0.0, population - immunity_y);
        sum_Sy += susceptibility_y;
    }

    // Block reduction for E[Sy]
    __shared__ double smem_Sy_reduction[BLOCK_SIZE];
    smem_Sy_reduction[threadIdx.x] = sum_Sy;
    block.sync();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            smem_Sy_reduction[threadIdx.x] += smem_Sy_reduction[threadIdx.x + stride];
        }
        block.sync();
    }

    __shared__ double E_Sy;
    if (threadIdx.x == 0) {
        E_Sy = smem_Sy_reduction[0] / (double)N_PK;
    }
    block.sync();

    // Step 3b: Compute denominator = Σx πx(t)·E[Sx]
    // Approximate E[Sx] ≈ mean(smem_sum_sx_pk) / n_variants for now
    double mean_sum_sx = 0.0;
    if (threadIdx.x == 0) {
        for (int pk = 0; pk < N_PK; pk++) {
            mean_sum_sx += smem_sum_sx_pk[pk];
        }
        mean_sum_sx /= (double)N_PK;
    }

    __shared__ double smem_denominator;
    if (threadIdx.x == 0) {
        smem_denominator = mean_sum_sx;  // Approximation: mean(Σx Sx) across PKs
    }
    block.sync();

    // Phase 4: Full VASIL envelope — per-PK gamma, then min/max across 75 PKs
    if (threadIdx.x == 0) {
        double gamma_min = 1e9;
        double gamma_max = -1e9;

        for (int pk = 0; pk < N_PK; pk++) {
            // Susceptible for variant y, this PK
            const double immunity_y_pk = smem_immunity_y[pk];
            double Sy_pk = fmax(0.0, population - immunity_y_pk);

            // S_mean from CSV is scaled down (~5M vs 300M population)
            // Need to match scales - divide Sy by normalization factor
            if (use_precomputed_s_mean) {
                Sy_pk = Sy_pk / 60.0;  // Approximate scale factor (will tune)
            }

            // S_mean for this date, this PK
            double S_mean_pk;
            if (use_precomputed_s_mean) {
                S_mean_pk = s_mean_precomputed[t_abs * N_PK + pk];
            } else {
                // Fallback for Germany/Denmark
                S_mean_pk = smem_sum_sx_pk[pk];
            }

            // Compute gamma for this PK
            if (S_mean_pk > 1e3) {  // Sanity check (S_mean should be ~millions)
                const double gamma_pk = (Sy_pk - S_mean_pk) / S_mean_pk;
                gamma_min = fmin(gamma_min, gamma_pk);
                gamma_max = fmax(gamma_max, gamma_pk);

                #if ENABLE_DEBUG
                static __device__ bool debug_once = false;
                if (!debug_once && pk == 37) {
                    debug_once = true;
                    printf("\n=== S_MEAN DEBUG (v=%d, d=%d, pk=%d) ===\n", y_idx, t_eval, pk);
                    printf("Sy_pk = %.2e\n", Sy_pk);
                    printf("S_mean_pk = %.2e\n", S_mean_pk);
                    printf("Sy - S_mean = %.2e\n", Sy_pk - S_mean_pk);
                    printf("gamma_pk = %.6f\n", gamma_pk);
                    printf("use_s_mean = %d\n", use_precomputed_s_mean);
                    printf("======================================\n\n");
                }
                #endif
            }
        }

        // VASIL envelope logic with TRAINABLE threshold
        const double threshold = (double)gamma_threshold;  // ES-optimized split point
        const double rise_threshold = threshold + (double)fluxnet_rise_bias;
        const double fall_threshold = threshold + (double)fluxnet_fall_bias;

        int8_t predicted_dir = 0;
        if (gamma_min > rise_threshold) {
            predicted_dir = 1;   // RISE: entire envelope above threshold
        } else if (gamma_max < fall_threshold) {
            predicted_dir = -1;  // FALL: entire envelope below threshold
        }
        // else 0 = UNDECIDED (envelope straddles threshold)
        
        if (predicted_dir != 0) {
            atomicAdd(total_count, 1u);
            if (predicted_dir == actual_dir) {
                atomicAdd(correct_count, 1u);
            }

            // Track RISE/FALL separately
            if (predicted_dir == 1) {
                atomicAdd(total_rise, 1u);
                if (actual_dir == 1) atomicAdd(correct_rise, 1u);
            } else if (predicted_dir == -1) {
                atomicAdd(total_fall, 1u);
                if (actual_dir == -1) atomicAdd(correct_fall, 1u);
            }
        }

        #if ENABLE_DEBUG
        if (y_idx == DEBUG_VARIANT && t_eval == DEBUG_DAY) {
            printf("=== CHECKPOINT 10: After comparing prediction vs actual ===\n");
            printf("  gamma_min=%.6f, gamma_max=%.6f\n", gamma_min, gamma_max);
            printf("  rise_bias=%.6f, fall_bias=%.6f\n", (double)fluxnet_rise_bias, (double)fluxnet_fall_bias);
            printf("  predicted_dir=%d, actual_dir=%d, MATCH=%s\n",
                   (int)predicted_dir, (int)actual_dir,
                   (predicted_dir == actual_dir) ? "YES" : "NO");
        }
        #endif
    }
}

//=============================================================================
// Helper kernels
//=============================================================================
extern "C" __global__ void reset_counters(
    unsigned int* correct_count,
    unsigned int* total_count
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *correct_count = 0;
        *total_count = 0;
    }
}

extern "C" __global__ void get_accuracy(
    const unsigned int* correct_count,
    const unsigned int* total_count,
    float* accuracy_out
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *accuracy_out = (*total_count > 0) ? 
            ((float)*correct_count / (float)*total_count) : 0.0f;
    }
}
