// PRISM-4D: Epitope-Based P_neut Computation (PATH A)
//
// Computes neutralization probability using 11-dimensional epitope distance:
//   P_neut(x,y) = exp(-d²(x,y) / (2σ²))
//   where d²(x,y) = Σ w_e × (escape_x[e] - escape_y[e])²
//
// This is simpler than PATH B (no PK dimension) and more accurate
// (captures antigenic distance properly).

#include <cuda_runtime.h>

constexpr int N_EPITOPES = 11;  // 10 RBD + 1 NTD

//=============================================================================
// KERNEL: Compute Epitope-Based P_neut Matrix
//=============================================================================
// Grid: (n_variants, n_variants, 1)
// Block: (1, 1, 1) - one thread per (x,y) pair
//
// This kernel is MUCH simpler than PATH B's on-the-fly kernel because:
// - No time dimension (P_neut is time-independent in epitope space)
// - No PK dimension (replaced by calibrated epitope weights)
// - No reduction (each thread computes one output value)
//=============================================================================
extern "C" __global__ void compute_epitope_p_neut(
    const float* __restrict__ epitope_escape,  // [n_variants × 11]
    float* __restrict__ p_neut_out,             // [n_variants × n_variants]
    const float* __restrict__ epitope_weights,  // [11] - calibrated weights
    const float sigma,                           // Gaussian bandwidth
    const int n_variants
) {
    const int x_idx = blockIdx.x;
    const int y_idx = blockIdx.y;
    
    if (x_idx >= n_variants || y_idx >= n_variants) return;
    
    // Load epitope escape for variant x
    float escape_x[N_EPITOPES];
    #pragma unroll
    for (int e = 0; e < N_EPITOPES; e++) {
        escape_x[e] = epitope_escape[x_idx * N_EPITOPES + e];
    }
    
    // Load epitope escape for variant y
    float escape_y[N_EPITOPES];
    #pragma unroll
    for (int e = 0; e < N_EPITOPES; e++) {
        escape_y[e] = epitope_escape[y_idx * N_EPITOPES + e];
    }
    
    // Compute weighted squared distance
    float d_squared = 0.0f;
    #pragma unroll
    for (int e = 0; e < N_EPITOPES; e++) {
        const float diff = escape_x[e] - escape_y[e];
        const float weight = epitope_weights[e];
        d_squared += weight * diff * diff;
    }
    
    // Compute Gaussian kernel: exp(-d² / 2σ²)
    const float sigma_sq = sigma * sigma;
    const float p_neut = expf(-d_squared / (2.0f * sigma_sq));
    
    // Write result
    const int out_idx = x_idx * n_variants + y_idx;
    p_neut_out[out_idx] = p_neut;
}

//=============================================================================
// KERNEL: Compute Immunity Using Precomputed P_neut (PATH A)
//=============================================================================
// Grid: (n_variants, n_eval_days, 1)
// Block: (256, 1, 1)
//
// This replaces PATH B's on-the-fly kernel. Uses precomputed P_neut matrix.
//=============================================================================
extern "C" __global__ void compute_immunity_from_epitope_p_neut(
    const float* __restrict__ p_neut_matrix,   // [n_variants × n_variants]
    const float* __restrict__ frequencies,     // [n_variants × max_history_days]
    const double* __restrict__ incidence,      // [max_history_days]
    double* __restrict__ immunity_out,         // [n_variants × n_eval_days]
    const int n_variants,
    const int n_eval_days,
    const int max_history_days,
    const int eval_start_offset
) {
    const int y_idx = blockIdx.x;    // Target variant
    const int t_eval = blockIdx.y;   // Evaluation day
    
    if (y_idx >= n_variants || t_eval >= n_eval_days) return;
    
    const int t_abs = eval_start_offset + t_eval;
    if (t_abs >= max_history_days) return;
    
    // CRITICAL FIX: Dynamic shared memory based on actual block size
    extern __shared__ double smem[];
    
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    double thread_sum = 0.0;
    
    // Process past variants (stride by block size)
    // OPTIMIZATION: Limit history to avoid TDR timeout (max 1000 days)
    const int history_limit = min(t_abs, 1000);
    
    for (int x = tid; x < n_variants; x += block_size) {
        // OPTIMIZATION: Vectorized P_neut load (cache-friendly)
        const float p_neut = p_neut_matrix[x * n_variants + y_idx];
        
        // Skip if P_neut is negligible
        if (p_neut < 1e-8f) continue;
        
        // Sum over history days (optimized: start from recent history)
        #pragma unroll 4
        for (int s = 0; s <= history_limit; s++) {
            const float freq = frequencies[x * max_history_days + s];
            if (freq < 0.001f) continue;
            
            const double inc = incidence[s];
            if (inc < 1.0) continue;
            
            // Accumulate contribution
            thread_sum += (double)freq * inc * (double)p_neut;
        }
    }
    
    // Block-wide reduction with dynamic shared memory
    smem[tid] = thread_sum;
    __syncthreads();
    
    // Parallel reduction (supports any block size)
    for (int stride = block_size / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    
    // Warp-level reduction (no sync needed)
    if (tid < 32) {
        volatile double* vsmem = smem;
        if (block_size >= 64) vsmem[tid] += vsmem[tid + 32];
        if (block_size >= 32) vsmem[tid] += vsmem[tid + 16];
        if (block_size >= 16) vsmem[tid] += vsmem[tid + 8];
        if (block_size >= 8)  vsmem[tid] += vsmem[tid + 4];
        if (block_size >= 4)  vsmem[tid] += vsmem[tid + 2];
        if (block_size >= 2)  vsmem[tid] += vsmem[tid + 1];
    }
    
    // Write result
    if (tid == 0) {
        const int out_idx = y_idx * n_eval_days + t_eval;
        immunity_out[out_idx] = smem[0];
    }
}

//=============================================================================
// KERNEL: Correlation Metric for Calibration
//=============================================================================
// Computes Pearson correlation between predicted P_neut and reference P_neut
// Used during parameter calibration (Nelder-Mead optimization)
//
// Grid: (1, 1, 1)
// Block: (256, 1, 1)
//=============================================================================
extern "C" __global__ void compute_p_neut_correlation(
    const float* __restrict__ p_neut_predicted, // [n_samples]
    const float* __restrict__ p_neut_reference, // [n_samples]
    double* __restrict__ correlation_out,        // [1]
    const int n_samples
) {
    __shared__ double smem_sum_x[256];
    __shared__ double smem_sum_y[256];
    __shared__ double smem_sum_xy[256];
    __shared__ double smem_sum_xx[256];
    __shared__ double smem_sum_yy[256];
    
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_xy = 0.0;
    double sum_xx = 0.0;
    double sum_yy = 0.0;
    
    // Accumulate statistics (each thread processes multiple samples)
    for (int i = tid; i < n_samples; i += block_size) {
        const double x = (double)p_neut_predicted[i];
        const double y = (double)p_neut_reference[i];
        
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_xx += x * x;
        sum_yy += y * y;
    }
    
    // Store to shared memory
    smem_sum_x[tid] = sum_x;
    smem_sum_y[tid] = sum_y;
    smem_sum_xy[tid] = sum_xy;
    smem_sum_xx[tid] = sum_xx;
    smem_sum_yy[tid] = sum_yy;
    __syncthreads();
    
    // Reduction
    for (int stride = block_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            smem_sum_x[tid] += smem_sum_x[tid + stride];
            smem_sum_y[tid] += smem_sum_y[tid + stride];
            smem_sum_xy[tid] += smem_sum_xy[tid + stride];
            smem_sum_xx[tid] += smem_sum_xx[tid + stride];
            smem_sum_yy[tid] += smem_sum_yy[tid + stride];
        }
        __syncthreads();
    }
    
    // Compute correlation
    if (tid == 0) {
        const double n = (double)n_samples;
        const double mean_x = smem_sum_x[0] / n;
        const double mean_y = smem_sum_y[0] / n;
        
        const double cov = (smem_sum_xy[0] / n) - (mean_x * mean_y);
        const double var_x = (smem_sum_xx[0] / n) - (mean_x * mean_x);
        const double var_y = (smem_sum_yy[0] / n) - (mean_y * mean_y);
        
        const double std_x = sqrt(var_x);
        const double std_y = sqrt(var_y);
        
        double corr = 0.0;
        if (std_x > 1e-12 && std_y > 1e-12) {
            corr = cov / (std_x * std_y);
        }
        
        correlation_out[0] = corr;
    }
}
