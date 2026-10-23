// PRISM-4D: On-the-Fly P_neut Computation (Memory-Efficient)
// 
// Instead of pre-computing P_neut for all time deltas (20 GB),
// compute it on-the-fly only when needed (ZERO extra memory).
//
// This is the CORRECT approach for large variant counts.

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

//=============================================================================
// DEVICE FUNCTION: Compute P_neut for specific (x, y, delta_t, pk)
//=============================================================================
__device__ float compute_p_neut_onthefly(
    const float* escape_x,  // [11] epitope escapes for variant x
    const float* escape_y,  // [11] epitope escapes for variant y
    int delta_t,
    int pk_idx
) {
    if (delta_t <= 0 || delta_t >= MAX_DELTA_DAYS) return 0.0f;
    
    // Get PK parameters
    const float tmax = c_tmax[pk_idx / 15];
    const float thalf = c_thalf[pk_idx % 15];
    const float ke = logf(2.0f) / thalf;
    
    float ka;
    const float ke_tmax = ke * tmax;
    if (ke_tmax > logf(2.0f) + 0.01f) {
        ka = logf(ke_tmax / (ke_tmax - logf(2.0f)));
    } else {
        ka = ke * 2.0f;
    }
    
    const float exp_ke_tmax = expf(-ke * tmax);
    const float exp_ka_tmax = expf(-ka * tmax);
    const float pk_denom = exp_ke_tmax - exp_ka_tmax;
    
    if (fabsf(pk_denom) < 1e-10f) return 0.0f;
    
    // Compute antibody concentration at time delta_t
    const float t = (float)delta_t;
    float c_t = (expf(-ke * t) - expf(-ka * t)) / pk_denom;
    c_t = fmaxf(0.0f, c_t);
    
    if (c_t < 1e-8f) return 0.0f;
    
    // Compute fold resistance for each epitope
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
        
        const float b_theta = c_t / (fold_res + c_t);
        product *= (1.0f - b_theta);
    }
    
    return 1.0f - product;
}

//=============================================================================
// KERNEL: Compute Immunity (On-the-Fly P_neut)
//=============================================================================
extern "C" __global__ void compute_immunity_onthefly(
    const float* __restrict__ epitope_escape,  // [n_variants × 11]
    const float* __restrict__ frequencies,     // [n_variants × max_history_days]
    const double* __restrict__ incidence,      // [max_history_days]
    double* __restrict__ immunity_out,         // [75 × n_variants × n_eval_days]
    const int n_variants,
    const int n_eval_days,
    const int max_history_days,
    const int eval_start_offset
) {
    // Decode collapsed grid: blockIdx.x encodes (variant × 75 PK)
    const int combined_idx = blockIdx.x;
    const int y_idx = combined_idx / N_PK;
    const int pk_idx = combined_idx % N_PK;
    const int t_eval = blockIdx.y;
    
    if (y_idx >= n_variants || t_eval >= n_eval_days || pk_idx >= N_PK) return;
    
    const int t_abs = eval_start_offset + t_eval;
    if (t_abs >= max_history_days) return;
    
    cg::thread_block block = cg::this_thread_block();
    
    // Load epitope escape for variant y (once per thread block)
    __shared__ float smem_escape_y[N_EPITOPES];
    if (threadIdx.x < N_EPITOPES) {
        smem_escape_y[threadIdx.x] = epitope_escape[y_idx * N_EPITOPES + threadIdx.x];
    }
    block.sync();
    
    // Parallel sum with Kahan compensation
    double thread_sum = 0.0;
    double thread_comp = 0.0;
    
    // Each thread processes a subset of (x, s) pairs
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
        
        // Compute P_neut on-the-fly
        const float p_neut = compute_p_neut_onthefly(
            escape_x,
            smem_escape_y,
            delta_t,
            pk_idx
        );
        
        if (p_neut < 1e-8f) continue;
        
        // Kahan summation
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
