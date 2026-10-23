// PRISM-4D: Parallel 75-PK GPU Immunity Computation
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

constexpr int MAX_DELTA_DAYS = 1500;
constexpr int N_EPITOPES = 11;
constexpr int N_PK = 75;
constexpr int BLOCK_SIZE = 256;
constexpr int TILE_S = 32;
constexpr int TILE_X = 32;

__constant__ float c_tmax[5] = {14.0f, 17.5f, 21.0f, 24.5f, 28.0f};
__constant__ float c_thalf[15] = {
    25.0f, 28.14f, 31.29f, 34.43f, 37.57f,
    40.71f, 43.86f, 47.0f, 50.14f, 53.29f,
    56.43f, 59.57f, 62.71f, 65.86f, 69.0f
};

extern "C" __global__ void build_p_neut_tables_all_pk(
    const float* __restrict__ epitope_escape,
    float* __restrict__ p_neut_tables,
    const int n_variants
) {
    const int x_idx = blockIdx.x;
    const int y_idx = blockIdx.y;
    const int pk_idx = blockIdx.z;
    
    if (x_idx >= n_variants || y_idx >= n_variants || pk_idx >= N_PK) return;
    
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
    
    float escape_x[N_EPITOPES], escape_y[N_EPITOPES];
    #pragma unroll
    for (int e = 0; e < N_EPITOPES; e++) {
        escape_x[e] = epitope_escape[x_idx * N_EPITOPES + e];
        escape_y[e] = epitope_escape[y_idx * N_EPITOPES + e];
    }
    
    float fold_resistance[N_EPITOPES];
    #pragma unroll
    for (int e = 0; e < N_EPITOPES; e++) {
        if (escape_x[e] > 0.01f) {
            fold_resistance[e] = (1.0f + escape_y[e]) / (1.0f + escape_x[e]);
        } else {
            fold_resistance[e] = 1.0f + escape_y[e];
        }
        fold_resistance[e] = fmaxf(0.1f, fminf(fold_resistance[e], 100.0f));
    }
    
    const size_t pk_stride = (size_t)MAX_DELTA_DAYS * n_variants * n_variants;
    
    for (int delta_t = threadIdx.x; delta_t < MAX_DELTA_DAYS; delta_t += BLOCK_SIZE) {
        float p_neut = 0.0f;
        
        if (delta_t > 0 && fabsf(pk_denom) > 1e-10f) {
            const float t = (float)delta_t;
            float c_t = (expf(-ke * t) - expf(-ka * t)) / pk_denom;
            c_t = fmaxf(0.0f, c_t);
            
            if (c_t > 1e-8f) {
                float product = 1.0f;
                #pragma unroll
                for (int e = 0; e < N_EPITOPES; e++) {
                    const float b_theta = c_t / (fold_resistance[e] + c_t);
                    product *= (1.0f - b_theta);
                }
                p_neut = 1.0f - product;
            }
        }
        
        const size_t idx = pk_idx * pk_stride + 
                          (size_t)delta_t * n_variants * n_variants +
                          (size_t)x_idx * n_variants + y_idx;
        p_neut_tables[idx] = p_neut;
    }
}

extern "C" __global__ void compute_immunity_all_pk(
    const float* __restrict__ frequencies,
    const double* __restrict__ incidence,
    const float* __restrict__ p_neut_tables,
    double* __restrict__ immunity_out,
    const int n_variants,
    const int n_eval_days,
    const int max_history_days,
    const int eval_start_offset
) {
    __shared__ float smem_freq[TILE_X][TILE_S + 1];
    __shared__ double smem_inc[TILE_S];
    
    const int y_idx = blockIdx.x;
    const int t_eval = blockIdx.y;
    const int pk_idx = blockIdx.z;
    
    if (y_idx >= n_variants || t_eval >= n_eval_days || pk_idx >= N_PK) return;
    
    const int t_abs = eval_start_offset + t_eval;
    cg::thread_block block = cg::this_thread_block();
    
    const size_t pk_stride = (size_t)MAX_DELTA_DAYS * n_variants * n_variants;
    const size_t pk_offset = pk_idx * pk_stride;
    
    double thread_sum = 0.0;
    double thread_comp = 0.0;
    
    for (int s_tile = 0; s_tile < t_abs; s_tile += TILE_S) {
        if (threadIdx.x < TILE_S) {
            const int s = s_tile + threadIdx.x;
            smem_inc[threadIdx.x] = (s < t_abs && s < max_history_days) ? incidence[s] : 0.0;
        }
        
        for (int x_tile = 0; x_tile < n_variants; x_tile += TILE_X) {
            for (int load_idx = threadIdx.x; load_idx < TILE_X * TILE_S; load_idx += BLOCK_SIZE) {
                const int local_x = load_idx / TILE_S;
                const int local_s = load_idx % TILE_S;
                const int x = x_tile + local_x;
                const int s = s_tile + local_s;
                
                if (x < n_variants && s < t_abs && s < max_history_days) {
                    smem_freq[local_x][local_s] = frequencies[x * max_history_days + s];
                } else {
                    smem_freq[local_x][local_s] = 0.0f;
                }
            }
            block.sync();
            
            for (int local_idx = threadIdx.x; local_idx < TILE_S * TILE_X; local_idx += BLOCK_SIZE) {
                const int local_s = local_idx / TILE_X;
                const int local_x = local_idx % TILE_X;
                const int s = s_tile + local_s;
                const int x = x_tile + local_x;
                
                if (s >= t_abs || x >= n_variants) continue;
                
                const int delta_t = t_abs - s;
                if (delta_t <= 0 || delta_t >= MAX_DELTA_DAYS) continue;
                
                const float freq = smem_freq[local_x][local_s];
                if (freq < 0.001f) continue;
                
                const double inc = smem_inc[local_s];
                if (inc < 1.0) continue;
                
                const size_t p_idx = pk_offset + 
                                    (size_t)delta_t * n_variants * n_variants +
                                    (size_t)x * n_variants + y_idx;
                const float p_neut = p_neut_tables[p_idx];
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
            block.sync();
        }
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
        const size_t out_idx = (size_t)pk_idx * n_variants * n_eval_days + 
                               y_idx * n_eval_days + t_eval;
        immunity_out[out_idx] = smem_sum[0] + smem_comp[0];
    }
}

extern "C" __global__ void compute_gamma_envelope(
    const double* __restrict__ immunity_all,
    const float* __restrict__ frequencies_eval,
    double* __restrict__ gamma_min,
    double* __restrict__ gamma_max,
    double* __restrict__ gamma_mean,
    const double population,
    const int n_variants,
    const int n_eval_days
) {
    const int y_idx = blockIdx.x;
    const int t_idx = blockIdx.y;
    
    if (y_idx >= n_variants || t_idx >= n_eval_days) return;
    
    const size_t imm_stride = (size_t)n_variants * n_eval_days;
    
    double local_min = 1e30;
    double local_max = -1e30;
    double local_sum = 0.0;
    int count = 0;
    
    for (int pk = threadIdx.x; pk < N_PK; pk += BLOCK_SIZE) {
        const double e_imm_y = immunity_all[pk * imm_stride + y_idx * n_eval_days + t_idx];
        const double e_s_y = population - e_imm_y;
        
        double weighted_sum = 0.0;
        double freq_sum = 0.0;
        
        for (int x = 0; x < n_variants; x++) {
            const float freq_x = frequencies_eval[x * n_eval_days + t_idx];
            if (freq_x >= 0.01f) {
                const double e_imm_x = immunity_all[pk * imm_stride + x * n_eval_days + t_idx];
                const double e_s_x = population - e_imm_x;
                weighted_sum += freq_x * e_s_x;
                freq_sum += freq_x;
            }
        }
        
        double gamma = 0.0;
        if (freq_sum > 0.01 && weighted_sum > 1.0) {
            gamma = (e_s_y / (weighted_sum / freq_sum)) - 1.0;
        }
        
        local_min = fmin(local_min, gamma);
        local_max = fmax(local_max, gamma);
        local_sum += gamma;
        count++;
    }
    
    __shared__ double s_min[BLOCK_SIZE];
    __shared__ double s_max[BLOCK_SIZE];
    __shared__ double s_sum[BLOCK_SIZE];
    __shared__ int s_cnt[BLOCK_SIZE];
    
    s_min[threadIdx.x] = local_min;
    s_max[threadIdx.x] = local_max;
    s_sum[threadIdx.x] = local_sum;
    s_cnt[threadIdx.x] = count;
    __syncthreads();
    
    for (int s = BLOCK_SIZE/2; s > 0; s /= 2) {
        if (threadIdx.x < s) {
            s_min[threadIdx.x] = fmin(s_min[threadIdx.x], s_min[threadIdx.x + s]);
            s_max[threadIdx.x] = fmax(s_max[threadIdx.x], s_max[threadIdx.x + s]);
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            s_cnt[threadIdx.x] += s_cnt[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        const size_t idx = y_idx * n_eval_days + t_idx;
        gamma_min[idx] = s_min[0];
        gamma_max[idx] = s_max[0];
        gamma_mean[idx] = (s_cnt[0] > 0) ? s_sum[0] / s_cnt[0] : 0.0;
    }
}

extern "C" __global__ void compute_predictions(
    const double* gamma_min,
    const double* gamma_max,
    int* predictions,
    const int n_total
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_total) return;
    
    const double g_min = gamma_min[idx];
    const double g_max = gamma_max[idx];
    
    if (g_min > 0.0 && g_max > 0.0) predictions[idx] = 1;
    else if (g_min < 0.0 && g_max < 0.0) predictions[idx] = -1;
    else predictions[idx] = 0;
}
