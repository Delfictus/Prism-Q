// PRISM-4D: Mega-Fused VASIL Benchmark Kernel
//
// Single kernel that computes VASIL accuracy directly from IC50 parameters.
// Designed for FluxNet RL optimization - maximum throughput.
//
// VASIL Methodology (Nature 2024, Extended Data Fig 6a):
//   1. Compute γy(t) = E[Sy(t)] / weighted_avg(E[Sx(t)]) - 1
//   2. Envelope: [min(γ), max(γ)] across 75 PK combinations
//   3. Prediction: envelope all positive → RISE, all negative → FALL, else UNDECIDED
//   4. Accuracy: correct predictions / total included predictions
//
// Exclusion criteria (VASIL-exact):
//   - Relative frequency change < 5% → exclude
//   - Frequency < 3% → exclude
//   - Envelope crosses zero → exclude (undecided)
//
// Architecture: Each thread block handles one (lineage, day) pair
// Output: Atomic accumulation of correct/total counts → single accuracy value

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

constexpr int MAX_DELTA_DAYS = 1500;
constexpr int N_EPITOPES = 11;
constexpr int N_PK = 75;
constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;

// Thresholds (VASIL-exact)
constexpr float NEGLIGIBLE_CHANGE_THRESHOLD = 0.05f;
constexpr float MIN_FREQUENCY_THRESHOLD = 0.03f;

__constant__ float c_tmax[5] = {14.0f, 17.5f, 21.0f, 24.5f, 28.0f};
__constant__ float c_thalf[15] = {
    25.0f, 28.14f, 31.29f, 34.43f, 37.57f,
    40.71f, 43.86f, 47.0f, 50.14f, 53.29f,
    56.43f, 59.57f, 62.71f, 65.86f, 69.0f
};

__constant__ float c_default_ic50[11] = {
    0.85f, 1.12f, 0.93f, 1.05f, 0.98f,
    1.21f, 0.89f, 1.08f, 0.95f, 1.03f, 1.00f
};

//=============================================================================
// DEVICE: P_neut with IC50 (inlined for fusion)
//=============================================================================
__device__ __forceinline__ float compute_p_neut_fused(
    const float* __restrict__ escape_x,
    const float* __restrict__ escape_y,
    const float* __restrict__ ic50,
    int delta_t,
    int pk_idx
) {
    if (delta_t <= 0 || delta_t >= MAX_DELTA_DAYS) return 0.0f;
    
    const float tmax = c_tmax[pk_idx / 15];
    const float thalf = c_thalf[pk_idx % 15];
    const float ke = __logf(2.0f) / thalf;
    
    float ka;
    const float ke_tmax = ke * tmax;
    if (ke_tmax > __logf(2.0f) + 0.01f) {
        ka = __logf(ke_tmax / (ke_tmax - __logf(2.0f)));
    } else {
        ka = ke * 2.0f;
    }
    
    const float exp_ke_tmax = __expf(-ke * tmax);
    const float exp_ka_tmax = __expf(-ka * tmax);
    const float pk_denom = exp_ke_tmax - exp_ka_tmax;
    
    if (fabsf(pk_denom) < 1e-10f) return 0.0f;
    
    const float t = (float)delta_t;
    float c_t = (__expf(-ke * t) - __expf(-ka * t)) / pk_denom;
    c_t = fmaxf(0.0f, c_t);
    
    if (c_t < 1e-8f) return 0.0f;
    
    float product = 1.0f;
    #pragma unroll
    for (int e = 0; e < N_EPITOPES; e++) {
        float fold_res;
        if (escape_x[e] > 0.01f) {
            fold_res = (1.0f + escape_y[e]) / (1.0f + escape_x[e]);
        } else {
            fold_res = 1.0f + escape_y[e];
        }
        fold_res = fmaxf(0.1f, fminf(fold_res, 100.0f));
        
        const float ic50_e = ic50[e];
        const float denom = fold_res * ic50_e + c_t;
        const float b_theta = (denom > 1e-10f) ? (c_t / denom) : 0.0f;
        product *= (1.0f - b_theta);
    }
    
    return 1.0f - product;
}

//=============================================================================
// DEVICE: Compute immunity for single (variant, day, pk) - warp-parallel
//=============================================================================
__device__ double compute_immunity_warp(
    const float* __restrict__ epitope_escape,
    const float* __restrict__ frequencies,
    const double* __restrict__ incidence,
    const float* __restrict__ ic50,
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
    const int n_work = n_variants * t_abs;
    
    for (int xs_idx = lane; xs_idx < n_work; xs_idx += WARP_SIZE) {
        const int x = xs_idx / t_abs;
        const int s = xs_idx % t_abs;
        
        if (s >= max_history_days) continue;
        
        const int delta_t = t_abs - s;
        if (delta_t <= 0 || delta_t >= MAX_DELTA_DAYS) continue;
        
        const float freq = frequencies[x * max_history_days + s];
        if (freq < 0.001f) continue;
        
        const double inc = incidence[s];
        if (inc < 1.0) continue;
        
        // Load escape_x into registers
        float escape_x[N_EPITOPES];
        #pragma unroll
        for (int e = 0; e < N_EPITOPES; e++) {
            escape_x[e] = epitope_escape[x * N_EPITOPES + e];
        }
        
        const float p_neut = compute_p_neut_fused(escape_x, escape_y, ic50, delta_t, pk_idx);
        
        if (p_neut > 1e-8f) {
            warp_sum += (double)freq * inc * (double)p_neut;
        }
    }
    
    // Warp reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        warp_sum += warp.shfl_down(warp_sum, offset);
    }
    
    return warp_sum;
}

//=============================================================================
// MEGA FUSED KERNEL: IC50 → Accuracy (single kernel!)
//=============================================================================
extern "C" __global__ void mega_fused_vasil_accuracy(
    const float* __restrict__ epitope_escape,    // [n_variants × 11]
    const float* __restrict__ frequencies,       // [n_variants × max_history_days]
    const double* __restrict__ incidence,        // [max_history_days]
    const float* __restrict__ ic50,              // [11] - FluxNet tunable!
    const int8_t* __restrict__ actual_directions,// [n_variants × n_eval_days] (-1=FALL, 0=exclude, 1=RISE)
    const float* __restrict__ freq_changes,      // [n_variants × n_eval_days] relative changes
    unsigned int* __restrict__ correct_count,    // Atomic counter
    unsigned int* __restrict__ total_count,      // Atomic counter
    const double population,
    const int n_variants,
    const int n_eval_days,
    const int max_history_days,
    const int eval_start_offset
) {
    // Each block handles one (lineage, day) pair
    const int y_idx = blockIdx.x;
    const int t_eval = blockIdx.y;
    
    if (y_idx >= n_variants || t_eval >= n_eval_days) return;
    
    const int t_abs = eval_start_offset + t_eval;
    if (t_abs >= max_history_days || t_abs < 1) return;
    
    // Check if this (lineage, day) should be included
    const int sample_idx = y_idx * n_eval_days + t_eval;
    const int8_t actual_dir = actual_directions[sample_idx];
    const float rel_change = freq_changes[sample_idx];
    
    // VASIL exclusion: actual_dir == 0 means already excluded (low freq or negligible change)
    if (actual_dir == 0) return;
    
    // Additional check: relative change threshold
    if (fabsf(rel_change) < NEGLIGIBLE_CHANGE_THRESHOLD) return;
    
    // Current frequency check
    const float current_freq = frequencies[y_idx * max_history_days + t_abs];
    if (current_freq < MIN_FREQUENCY_THRESHOLD) return;
    
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> warp = cg::tiled_partition<WARP_SIZE>(block);
    
    // Load IC50 and escape_y into shared memory
    __shared__ float smem_ic50[N_EPITOPES];
    __shared__ float smem_escape_y[N_EPITOPES];
    
    if (threadIdx.x < N_EPITOPES) {
        smem_ic50[threadIdx.x] = (ic50 != nullptr) ? ic50[threadIdx.x] : c_default_ic50[threadIdx.x];
        smem_escape_y[threadIdx.x] = epitope_escape[y_idx * N_EPITOPES + threadIdx.x];
    }
    block.sync();
    
    // Compute immunity for all 75 PK combinations
    // Each warp handles a subset of PK indices
    const int warps_per_block = BLOCK_SIZE / WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    
    __shared__ double smem_immunity_y[N_PK];
    __shared__ double smem_weighted_avg[N_PK];
    
    // Phase 1: Compute immunity for variant y across all PK
    for (int pk = warp_id; pk < N_PK; pk += warps_per_block) {
        double immunity_y = compute_immunity_warp(
            epitope_escape, frequencies, incidence, smem_ic50, smem_escape_y,
            y_idx, t_abs, pk, n_variants, max_history_days, warp
        );
        
        if (warp.thread_rank() == 0) {
            smem_immunity_y[pk] = immunity_y;
        }
    }
    block.sync();
    
    // Phase 2: Compute weighted average susceptibility (use middle PK = 37)
    // Only first warp does this
    if (warp_id == 0) {
        double weighted_sum = 0.0;
        double freq_sum = 0.0;
        
        for (int x = warp.thread_rank(); x < n_variants; x += WARP_SIZE) {
            const float freq = frequencies[x * max_history_days + t_abs];
            if (freq >= 0.001f) {
                // Compute immunity for variant x at middle PK
                float escape_x[N_EPITOPES];
                #pragma unroll
                for (int e = 0; e < N_EPITOPES; e++) {
                    escape_x[e] = epitope_escape[x * N_EPITOPES + e];
                }
                
                double immunity_x = 0.0;
                // Simplified: use single representative history point
                for (int s = 0; s < t_abs && s < max_history_days; s++) {
                    const float f = frequencies[x * max_history_days + s];
                    if (f < 0.001f) continue;
                    const double inc = incidence[s];
                    if (inc < 1.0) continue;
                    const int delta_t = t_abs - s;
                    if (delta_t <= 0 || delta_t >= MAX_DELTA_DAYS) continue;
                    
                    const float p_neut = compute_p_neut_fused(escape_x, smem_escape_y, smem_ic50, delta_t, 37);
                    immunity_x += (double)f * inc * (double)p_neut;
                }
                
                const double susceptibility_x = fmax(0.0, population - immunity_x);
                weighted_sum += (double)freq * susceptibility_x;
                freq_sum += (double)freq;
            }
        }
        
        // Warp reduction
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            weighted_sum += warp.shfl_down(weighted_sum, offset);
            freq_sum += warp.shfl_down(freq_sum, offset);
        }
        
        if (warp.thread_rank() == 0) {
            const double weighted_avg = (freq_sum > 0.0) ? (weighted_sum / freq_sum) : (population * 0.5);
            for (int pk = 0; pk < N_PK; pk++) {
                smem_weighted_avg[pk] = weighted_avg;
            }
        }
    }
    block.sync();
    
    // Phase 3: Compute gamma envelope
    __shared__ double smem_gamma_min;
    __shared__ double smem_gamma_max;
    
    if (threadIdx.x == 0) {
        smem_gamma_min = 1e10;
        smem_gamma_max = -1e10;
    }
    block.sync();
    
    // Each thread handles subset of PK indices
    double local_min = 1e10;
    double local_max = -1e10;
    
    for (int pk = threadIdx.x; pk < N_PK; pk += BLOCK_SIZE) {
        const double immunity_y = smem_immunity_y[pk];
        const double weighted_avg = smem_weighted_avg[pk];
        const double susceptibility_y = fmax(0.0, population - immunity_y);
        
        double gamma;
        if (weighted_avg > 0.1) {
            gamma = (susceptibility_y / weighted_avg) - 1.0;
        } else {
            gamma = 0.0;
        }
        
        local_min = fmin(local_min, gamma);
        local_max = fmax(local_max, gamma);
    }
    
    // Block reduction for min/max
    __shared__ double smem_local_min[BLOCK_SIZE];
    __shared__ double smem_local_max[BLOCK_SIZE];
    smem_local_min[threadIdx.x] = local_min;
    smem_local_max[threadIdx.x] = local_max;
    block.sync();
    
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            smem_local_min[threadIdx.x] = fmin(smem_local_min[threadIdx.x], smem_local_min[threadIdx.x + stride]);
            smem_local_max[threadIdx.x] = fmax(smem_local_max[threadIdx.x], smem_local_max[threadIdx.x + stride]);
        }
        block.sync();
    }
    
    // Phase 4: Make prediction and compare
    if (threadIdx.x == 0) {
        const double gamma_min = smem_local_min[0];
        const double gamma_max = smem_local_max[0];
        
        int8_t predicted_dir;
        if (gamma_min > 0.0 && gamma_max > 0.0) {
            predicted_dir = 1;  // RISE
        } else if (gamma_min < 0.0 && gamma_max < 0.0) {
            predicted_dir = -1; // FALL
        } else {
            predicted_dir = 0;  // UNDECIDED - exclude
        }
        
        // Only count if prediction is decided
        if (predicted_dir != 0) {
            atomicAdd(total_count, 1u);
            if (predicted_dir == actual_dir) {
                atomicAdd(correct_count, 1u);
            }
        }
    }
}

//=============================================================================
// KERNEL: Reset counters (call before mega_fused_vasil_accuracy)
//=============================================================================
extern "C" __global__ void reset_accuracy_counters(
    unsigned int* correct_count,
    unsigned int* total_count
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *correct_count = 0;
        *total_count = 0;
    }
}

//=============================================================================
// KERNEL: Compute final accuracy (call after mega_fused_vasil_accuracy)
//=============================================================================
extern "C" __global__ void compute_final_accuracy(
    const unsigned int* correct_count,
    const unsigned int* total_count,
    float* accuracy_out
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const unsigned int correct = *correct_count;
        const unsigned int total = *total_count;
        *accuracy_out = (total > 0) ? ((float)correct / (float)total) : 0.0f;
    }
}
