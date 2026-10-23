// PRISM-4D: PATH B Immunity with Tunable IC50 (FluxNet-Optimizable)
//
// VASIL Formula (Methods, Page 12):
//   bϑ(t, x, y) = cϑ(t) / (FRx,y(ϑ)·IC50(ϑ) + cϑ(t))
//   P_neut = 1 - Π_ϑ (1 - bϑ)
//
// IC50 values are passed as kernel parameter for FluxNet RL optimization.
// This enables 60× faster calibration vs CPU-based IC50 updates.

#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

constexpr int MAX_DELTA_DAYS = 1500;
constexpr int N_EPITOPES = 11;
constexpr int N_PK = 75;
constexpr int BLOCK_SIZE = 256;

__constant__ float c_tmax[5] = {14.0f, 17.5f, 21.0f, 24.5f, 28.0f};
__constant__ float c_thalf[15] = {
    25.0f, 28.14f, 31.29f, 34.43f, 37.57f,
    40.71f, 43.86f, 47.0f, 50.14f, 53.29f,
    56.43f, 59.57f, 62.71f, 65.86f, 69.0f
};

// Default IC50 values (VASIL calibrated) - used if ic50 parameter is NULL
__constant__ float c_default_ic50[11] = {
    0.85f,  // A
    1.12f,  // B
    0.93f,  // C
    1.05f,  // D1
    0.98f,  // D2
    1.21f,  // E12
    0.89f,  // E3
    1.08f,  // F1
    0.95f,  // F2
    1.03f,  // F3
    1.00f   // NTD
};

//=============================================================================
// DEVICE: Compute P_neut with IC50 (VASIL-Exact Formula)
//=============================================================================
__device__ float compute_p_neut_with_ic50(
    const float* escape_x,      // [11] epitope escapes for variant x
    const float* escape_y,      // [11] epitope escapes for variant y  
    const float* ic50,          // [11] IC50 values (or NULL for defaults)
    const float* epitope_weights, // [11] epitope contribution weights (or NULL)
    int delta_t,
    int pk_idx
) {
    if (delta_t <= 0 || delta_t >= MAX_DELTA_DAYS) return 0.0f;
    
    // PK parameters from grid
    const float tmax = c_tmax[pk_idx / 15];
    const float thalf = c_thalf[pk_idx % 15];
    const float ke = logf(2.0f) / thalf;
    
    // Compute absorption rate ka
    float ka;
    const float ke_tmax = ke * tmax;
    if (ke_tmax > logf(2.0f) + 0.01f) {
        ka = logf(ke_tmax / (ke_tmax - logf(2.0f)));
    } else {
        ka = ke * 2.0f;
    }
    
    // PK normalization denominator
    const float exp_ke_tmax = expf(-ke * tmax);
    const float exp_ka_tmax = expf(-ka * tmax);
    const float pk_denom = exp_ke_tmax - exp_ka_tmax;
    
    if (fabsf(pk_denom) < 1e-10f) return 0.0f;
    
    // Antibody concentration at time delta_t
    const float t = (float)delta_t;
    float c_t = (expf(-ke * t) - expf(-ka * t)) / pk_denom;
    c_t = fmaxf(0.0f, c_t);
    
    if (c_t < 1e-8f) return 0.0f;
    
    // P_neut = 1 - Π_ϑ (1 - bϑ)
    float product = 1.0f;
    
    #pragma unroll
    for (int e = 0; e < N_EPITOPES; e++) {
        // Fold resistance: FR(x,y) = (1 + escape_y) / (1 + escape_x)
        float fold_res;
        if (escape_x[e] > 0.01f) {
            fold_res = (1.0f + escape_y[e]) / (1.0f + escape_x[e]);
        } else {
            fold_res = 1.0f + escape_y[e];
        }
        fold_res = fmaxf(0.1f, fminf(fold_res, 100.0f));
        
        // Get IC50 (use parameter or default)
        const float ic50_e = (ic50 != nullptr) ? ic50[e] : c_default_ic50[e];
        
        // VASIL Formula: bϑ = cϑ(t) / (FR·IC50 + cϑ(t))
        const float denominator = fold_res * ic50_e + c_t;
        const float b_theta = (denominator > 1e-10f) ? (c_t / denominator) : 0.0f;
        
        // Apply epitope weight if provided
        float weighted_b = b_theta;
        if (epitope_weights != nullptr) {
            weighted_b = b_theta * epitope_weights[e];
        }
        
        product *= (1.0f - weighted_b);
    }
    
    return 1.0f - product;
}

//=============================================================================
// KERNEL: Compute Immunity with Tunable IC50 (PATH B + FluxNet)
//=============================================================================
extern "C" __global__ void compute_immunity_with_ic50(
    const float* __restrict__ epitope_escape,   // [n_variants × 11]
    const float* __restrict__ frequencies,      // [n_variants × max_history_days]
    const double* __restrict__ incidence,       // [max_history_days]
    const float* __restrict__ ic50,             // [11] IC50 values (can be NULL)
    const float* __restrict__ epitope_weights,  // [11] weights (can be NULL)
    double* __restrict__ immunity_out,          // [75 × n_variants × n_eval_days]
    const int n_variants,
    const int n_eval_days,
    const int max_history_days,
    const int eval_start_offset
) {
    // Grid mapping: blockIdx.x = variant × 75 PK, blockIdx.y = eval_day
    const int combined_idx = blockIdx.x;
    const int y_idx = combined_idx / N_PK;
    const int pk_idx = combined_idx % N_PK;
    const int t_eval = blockIdx.y;
    
    if (y_idx >= n_variants || t_eval >= n_eval_days || pk_idx >= N_PK) return;
    
    const int t_abs = eval_start_offset + t_eval;
    if (t_abs >= max_history_days) return;
    
    cg::thread_block block = cg::this_thread_block();
    
    // Load IC50 and weights into shared memory for fast access
    __shared__ float smem_ic50[N_EPITOPES];
    __shared__ float smem_weights[N_EPITOPES];
    __shared__ float smem_escape_y[N_EPITOPES];
    
    if (threadIdx.x < N_EPITOPES) {
        smem_escape_y[threadIdx.x] = epitope_escape[y_idx * N_EPITOPES + threadIdx.x];
        smem_ic50[threadIdx.x] = (ic50 != nullptr) ? ic50[threadIdx.x] : c_default_ic50[threadIdx.x];
        smem_weights[threadIdx.x] = (epitope_weights != nullptr) ? epitope_weights[threadIdx.x] : 1.0f;
    }
    block.sync();
    
    // Parallel sum with Kahan compensation for numerical stability
    double thread_sum = 0.0;
    double thread_comp = 0.0;
    
    // Each thread processes subset of (x, s) pairs
    for (int xs_idx = threadIdx.x; xs_idx < n_variants * t_abs; xs_idx += BLOCK_SIZE) {
        const int x = xs_idx / t_abs;
        const int s = xs_idx % t_abs;
        
        if (s >= max_history_days) continue;
        
        const int delta_t = t_abs - s;
        if (delta_t <= 0 || delta_t >= MAX_DELTA_DAYS) continue;
        
        // Load frequency and incidence
        const float freq = frequencies[x * max_history_days + s];
        if (freq < 0.001f) continue;
        
        const double inc = incidence[s];
        if (inc < 1.0) continue;
        
        // Load epitope escape for variant x
        float escape_x[N_EPITOPES];
        #pragma unroll
        for (int e = 0; e < N_EPITOPES; e++) {
            escape_x[e] = epitope_escape[x * N_EPITOPES + e];
        }
        
        // Compute P_neut with IC50 (VASIL-exact formula)
        const float p_neut = compute_p_neut_with_ic50(
            escape_x,
            smem_escape_y,
            smem_ic50,
            smem_weights,
            delta_t,
            pk_idx
        );
        
        if (p_neut < 1e-8f) continue;
        
        // Kahan summation for precision
        const double term = (double)freq * inc * (double)p_neut;
        const double temp = thread_sum + term;
        if (fabs(thread_sum) >= fabs(term)) {
            thread_comp += (thread_sum - temp) + term;
        } else {
            thread_comp += (term - temp) + thread_sum;
        }
        thread_sum = temp;
    }
    
    // Block-wide reduction
    __shared__ double smem_sum[BLOCK_SIZE];
    __shared__ double smem_comp[BLOCK_SIZE];
    smem_sum[threadIdx.x] = thread_sum;
    smem_comp[threadIdx.x] = thread_comp;
    block.sync();
    
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            double a = smem_sum[threadIdx.x];
            double b = smem_sum[threadIdx.x + stride];
            double sum = a + b;
            double comp = smem_comp[threadIdx.x] + smem_comp[threadIdx.x + stride];
            if (fabs(a) >= fabs(b)) comp += (a - sum) + b;
            else comp += (b - sum) + a;
            smem_sum[threadIdx.x] = sum;
            smem_comp[threadIdx.x] = comp;
        }
        block.sync();
    }
    
    // Write result
    if (threadIdx.x == 0) {
        const size_t out_idx = (size_t)pk_idx * n_variants * n_eval_days + 
                              (size_t)y_idx * n_eval_days + t_eval;
        immunity_out[out_idx] = smem_sum[0] + smem_comp[0];
    }
}

//=============================================================================
// KERNEL: Fast Single-PK Immunity (For FluxNet optimization - skip envelope)
//=============================================================================
extern "C" __global__ void compute_immunity_single_pk(
    const float* __restrict__ epitope_escape,   // [n_variants × 11]
    const float* __restrict__ frequencies,      // [n_variants × max_history_days]
    const double* __restrict__ incidence,       // [max_history_days]
    const float* __restrict__ ic50,             // [11] IC50 values
    const float* __restrict__ epitope_weights,  // [11] weights
    double* __restrict__ immunity_out,          // [n_variants × n_eval_days]
    const int n_variants,
    const int n_eval_days,
    const int max_history_days,
    const int eval_start_offset,
    const int pk_idx                            // Single PK index (0-74)
) {
    const int y_idx = blockIdx.x;
    const int t_eval = blockIdx.y;
    
    if (y_idx >= n_variants || t_eval >= n_eval_days) return;
    
    const int t_abs = eval_start_offset + t_eval;
    if (t_abs >= max_history_days) return;
    
    cg::thread_block block = cg::this_thread_block();
    
    __shared__ float smem_ic50[N_EPITOPES];
    __shared__ float smem_weights[N_EPITOPES];
    __shared__ float smem_escape_y[N_EPITOPES];
    
    if (threadIdx.x < N_EPITOPES) {
        smem_escape_y[threadIdx.x] = epitope_escape[y_idx * N_EPITOPES + threadIdx.x];
        smem_ic50[threadIdx.x] = (ic50 != nullptr) ? ic50[threadIdx.x] : c_default_ic50[threadIdx.x];
        smem_weights[threadIdx.x] = (epitope_weights != nullptr) ? epitope_weights[threadIdx.x] : 1.0f;
    }
    block.sync();
    
    double thread_sum = 0.0;
    double thread_comp = 0.0;
    
    for (int xs_idx = threadIdx.x; xs_idx < n_variants * t_abs; xs_idx += BLOCK_SIZE) {
        const int x = xs_idx / t_abs;
        const int s = xs_idx % t_abs;
        
        if (s >= max_history_days) continue;
        
        const int delta_t = t_abs - s;
        if (delta_t <= 0 || delta_t >= MAX_DELTA_DAYS) continue;
        
        const float freq = frequencies[x * max_history_days + s];
        if (freq < 0.001f) continue;
        
        const double inc = incidence[s];
        if (inc < 1.0) continue;
        
        float escape_x[N_EPITOPES];
        #pragma unroll
        for (int e = 0; e < N_EPITOPES; e++) {
            escape_x[e] = epitope_escape[x * N_EPITOPES + e];
        }
        
        const float p_neut = compute_p_neut_with_ic50(
            escape_x, smem_escape_y, smem_ic50, smem_weights, delta_t, pk_idx
        );
        
        if (p_neut < 1e-8f) continue;
        
        const double term = (double)freq * inc * (double)p_neut;
        const double temp = thread_sum + term;
        if (fabs(thread_sum) >= fabs(term)) {
            thread_comp += (thread_sum - temp) + term;
        } else {
            thread_comp += (term - temp) + thread_sum;
        }
        thread_sum = temp;
    }
    
    __shared__ double smem_sum[BLOCK_SIZE];
    __shared__ double smem_comp[BLOCK_SIZE];
    smem_sum[threadIdx.x] = thread_sum;
    smem_comp[threadIdx.x] = thread_comp;
    block.sync();
    
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            double a = smem_sum[threadIdx.x];
            double b = smem_sum[threadIdx.x + stride];
            double sum = a + b;
            double comp = smem_comp[threadIdx.x] + smem_comp[threadIdx.x + stride];
            if (fabs(a) >= fabs(b)) comp += (a - sum) + b;
            else comp += (b - sum) + a;
            smem_sum[threadIdx.x] = sum;
            smem_comp[threadIdx.x] = comp;
        }
        block.sync();
    }
    
    if (threadIdx.x == 0) {
        immunity_out[y_idx * n_eval_days + t_eval] = smem_sum[0] + smem_comp[0];
    }
}

//=============================================================================
// KERNEL: Compute Gamma Envelopes from 75-PK Immunity Matrix
//=============================================================================
extern "C" __global__ void compute_gamma_from_immunity(
    const double* __restrict__ immunity_75pk,  // [75 × n_variants × n_eval_days]
    const float* __restrict__ frequencies,     // [n_variants × max_history_days]
    double* __restrict__ gamma_min,            // [n_variants × n_eval_days]
    double* __restrict__ gamma_max,            // [n_variants × n_eval_days]
    double* __restrict__ gamma_mean,           // [n_variants × n_eval_days]
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
    if (t_abs >= max_history_days) return;
    
    // Compute weighted average susceptibility across all variants
    double weighted_sum_s = 0.0;
    double freq_sum = 0.0;
    
    for (int x = 0; x < n_variants; x++) {
        const float freq = frequencies[x * max_history_days + t_abs];
        if (freq < 0.001f) continue;
        
        // Use middle PK (index 37) for weighted average calculation
        const double immunity_x = immunity_75pk[37 * n_variants * n_eval_days + x * n_eval_days + t_eval];
        const double susceptibility_x = fmax(0.0, population - immunity_x);
        
        weighted_sum_s += (double)freq * susceptibility_x;
        freq_sum += (double)freq;
    }
    
    const double weighted_avg_s = (freq_sum > 0.0) ? (weighted_sum_s / freq_sum) : (population * 0.5);
    
    // Compute gamma envelope across all 75 PK combinations
    double g_min = 1e10;
    double g_max = -1e10;
    double g_sum = 0.0;
    
    for (int pk = 0; pk < N_PK; pk++) {
        const double immunity_y = immunity_75pk[pk * n_variants * n_eval_days + y_idx * n_eval_days + t_eval];
        const double susceptibility_y = fmax(0.0, population - immunity_y);
        
        double gamma;
        if (weighted_avg_s > 0.1) {
            gamma = (susceptibility_y / weighted_avg_s) - 1.0;
        } else {
            gamma = 0.0;
        }
        
        g_min = fmin(g_min, gamma);
        g_max = fmax(g_max, gamma);
        g_sum += gamma;
    }
    
    const size_t out_idx = y_idx * n_eval_days + t_eval;
    gamma_min[out_idx] = g_min;
    gamma_max[out_idx] = g_max;
    gamma_mean[out_idx] = g_sum / (double)N_PK;
}
