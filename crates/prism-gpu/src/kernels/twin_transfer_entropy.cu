// ═══════════════════════════════════════════════════════════════════════
// PRISM-TWIN: GPU Binned Transfer Entropy
//
// Computes TE(A→B) and TE(B→A) for all residue pairs using the same
// time-binned FP16 spike matrices as the Tensor Core CCF kernel.
//
// Algorithm: Binned TE (not KSG). For discretized spike data (above/below
// median per time bin), the conditional probability tables are 2×2×2
// (8 entries), fit entirely in registers. O(n_bins) per pair.
//
// Input:  Same [n_res_padded × n_bins_padded] FP16 matrices as tensor_ccf.cu
// Output: [n_res × n_res] TE(A→B) matrix
//         [n_res × n_res] TE(B→A) matrix
//
// Grid:  (ceil(n_res/16), ceil(n_res/16)), Block: (16, 16)
// Each thread computes TE for one (source, target) residue pair.
//
// Design notes:
// - Uses mean (not median) as threshold for binarization — computing
//   exact median would require sorting, which is O(n_bins log n_bins)
//   per residue. Mean is O(1) from pre-computed row sums and works well
//   for approximately symmetric spike distributions.
// - Lag parameter controls temporal offset for causal inference.
//   Default lag=1 (immediate causation). Higher lags detect slower propagation.
// - TE can be negative from finite-sample bias; clamped to max(0, TE).
//
// Performance: ~1ms for 300 residues, 350 bins (same order as CCF).
// ═══════════════════════════════════════════════════════════════════════

#include <cuda_fp16.h>
#include <math.h>

// ─────────────────────────────────────────────────────────────────────
// KERNEL 1: Compute per-row mean for binarization threshold
// ─────────────────────────────────────────────────────────────────────

extern "C" __global__ void twin_compute_row_means(
    const half* __restrict__ spike_matrix,   // [n_res_padded × n_bins_padded]
    float* __restrict__ means,               // [n_res] output
    int n_res,
    int n_bins,           // actual number of time bins (not padded)
    int n_bins_padded     // padded dimension (multiple of 16)
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_res) return;

    float sum = 0.0f;
    for (int t = 0; t < n_bins; t++) {
        sum += __half2float(spike_matrix[row * n_bins_padded + t]);
    }
    means[row] = sum / fmaxf((float)n_bins, 1.0f);
}

// ─────────────────────────────────────────────────────────────────────
// KERNEL 2: Binned Transfer Entropy — full n_res × n_res matrix
//
// TE(src→tgt) = Σ p(src_past, tgt_past, tgt_future) ×
//               log2( p(tgt_future | tgt_past, src_past) /
//                     p(tgt_future | tgt_past) )
//
// Discretized to binary (above/below mean), so the conditional
// probability tables have 2×2×2 = 8 entries, all in registers.
// ─────────────────────────────────────────────────────────────────────

extern "C" __global__ void twin_binned_te(
    const half* __restrict__ spike_matrix_a,  // [n_res_padded × n_bins_padded]
    const half* __restrict__ spike_matrix_b,  // [n_res_padded × n_bins_padded]
    float* __restrict__ te_a_to_b,            // [n_res × n_res] output: TE(A→B)
    float* __restrict__ te_b_to_a,            // [n_res × n_res] output: TE(B→A)
    const float* __restrict__ mean_a,         // [n_res] per-row mean for binarization
    const float* __restrict__ mean_b,         // [n_res] per-row mean
    int n_res,
    int n_res_padded,
    int n_bins,           // actual time bins (not padded)
    int n_bins_padded,
    int lag               // temporal lag for causal inference (default: 1)
) {
    int src = blockIdx.x * blockDim.x + threadIdx.x;
    int tgt = blockIdx.y * blockDim.y + threadIdx.y;
    if (src >= n_res || tgt >= n_res) return;

    int out_idx = src * n_res + tgt;

    // Self-TE is undefined
    if (src == tgt) {
        te_a_to_b[out_idx] = 0.0f;
        te_b_to_a[out_idx] = 0.0f;
        return;
    }

    // Binarization thresholds
    float thresh_src_a = mean_a[src];
    float thresh_tgt_b = mean_b[tgt];
    float thresh_src_b = mean_b[src];
    float thresh_tgt_a = mean_a[tgt];

    // ── TE(A_src → B_tgt) ──
    // "Does knowing A_src's past reduce uncertainty about B_tgt's future?"
    {
        // Count conditional frequencies: counts[src_past][tgt_past][tgt_future]
        int counts[2][2][2];
        for (int i = 0; i < 8; i++) ((int*)counts)[i] = 0;
        int total = 0;

        for (int t = lag; t < n_bins - 1; t++) {
            float src_val = __half2float(spike_matrix_a[src * n_bins_padded + (t - lag)]);
            float tgt_past_val = __half2float(spike_matrix_b[tgt * n_bins_padded + (t - lag)]);
            float tgt_fut_val = __half2float(spike_matrix_b[tgt * n_bins_padded + (t + 1)]);

            int sp = (src_val > thresh_src_a) ? 1 : 0;
            int tp = (tgt_past_val > thresh_tgt_b) ? 1 : 0;
            int tf = (tgt_fut_val > thresh_tgt_b) ? 1 : 0;

            counts[sp][tp][tf]++;
            total++;
        }

        float te = 0.0f;
        if (total >= 10) {
            float inv_total = 1.0f / (float)total;
            for (int sp = 0; sp < 2; sp++) {
                for (int tp = 0; tp < 2; tp++) {
                    // Marginals
                    int n_tp = counts[0][tp][0] + counts[0][tp][1] +
                               counts[1][tp][0] + counts[1][tp][1];
                    if (n_tp == 0) continue;

                    for (int tf = 0; tf < 2; tf++) {
                        int n_joint = counts[sp][tp][tf];
                        if (n_joint == 0) continue;

                        int n_sp_tp = counts[sp][tp][0] + counts[sp][tp][1];
                        int n_tp_tf = counts[0][tp][tf] + counts[1][tp][tf];
                        if (n_sp_tp == 0 || n_tp_tf == 0) continue;

                        float p_joint = (float)n_joint * inv_total;
                        float p_cond_full = (float)n_joint / (float)n_sp_tp;
                        float p_cond_marginal = (float)n_tp_tf / (float)n_tp;

                        // Guard against log2(0)
                        float ratio = p_cond_full / fmaxf(p_cond_marginal, 1e-10f);
                        te += p_joint * log2f(fmaxf(ratio, 1e-10f));
                    }
                }
            }
        }
        te_a_to_b[out_idx] = fmaxf(0.0f, te);  // Clamp negative (finite-sample bias)
    }

    // ── TE(B_src → A_tgt) ──
    // Symmetric computation with swapped roles
    {
        int counts[2][2][2];
        for (int i = 0; i < 8; i++) ((int*)counts)[i] = 0;
        int total = 0;

        for (int t = lag; t < n_bins - 1; t++) {
            float src_val = __half2float(spike_matrix_b[src * n_bins_padded + (t - lag)]);
            float tgt_past_val = __half2float(spike_matrix_a[tgt * n_bins_padded + (t - lag)]);
            float tgt_fut_val = __half2float(spike_matrix_a[tgt * n_bins_padded + (t + 1)]);

            int sp = (src_val > thresh_src_b) ? 1 : 0;
            int tp = (tgt_past_val > thresh_tgt_a) ? 1 : 0;
            int tf = (tgt_fut_val > thresh_tgt_a) ? 1 : 0;

            counts[sp][tp][tf]++;
            total++;
        }

        float te = 0.0f;
        if (total >= 10) {
            float inv_total = 1.0f / (float)total;
            for (int sp = 0; sp < 2; sp++) {
                for (int tp = 0; tp < 2; tp++) {
                    int n_tp = counts[0][tp][0] + counts[0][tp][1] +
                               counts[1][tp][0] + counts[1][tp][1];
                    if (n_tp == 0) continue;

                    for (int tf = 0; tf < 2; tf++) {
                        int n_joint = counts[sp][tp][tf];
                        if (n_joint == 0) continue;

                        int n_sp_tp = counts[sp][tp][0] + counts[sp][tp][1];
                        int n_tp_tf = counts[0][tp][tf] + counts[1][tp][tf];
                        if (n_sp_tp == 0 || n_tp_tf == 0) continue;

                        float p_joint = (float)n_joint * inv_total;
                        float p_cond_full = (float)n_joint / (float)n_sp_tp;
                        float p_cond_marginal = (float)n_tp_tf / (float)n_tp;

                        float ratio = p_cond_full / fmaxf(p_cond_marginal, 1e-10f);
                        te += p_joint * log2f(fmaxf(ratio, 1e-10f));
                    }
                }
            }
        }
        te_b_to_a[out_idx] = fmaxf(0.0f, te);
    }
}

// ─────────────────────────────────────────────────────────────────────
// KERNEL 3: Compute per-residue mutual information from TE matrices
//
// MI(i) = sum over all j of (TE(A_i→B_j) + TE(B_j→A_i)) / 2
// This measures how much information residue i carries about the
// cross-stream dynamics overall.
// ─────────────────────────────────────────────────────────────────────

extern "C" __global__ void twin_compute_mutual_info(
    const float* __restrict__ te_a_to_b,     // [n_res × n_res]
    const float* __restrict__ te_b_to_a,     // [n_res × n_res]
    float* __restrict__ mutual_info,          // [n_res] output
    float* __restrict__ causal_flow,          // [n_res] output: net TE direction
    int n_res
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_res) return;

    float mi_sum = 0.0f;
    float te_out = 0.0f;  // TE flowing OUT from residue i (A→B)
    float te_in = 0.0f;   // TE flowing IN to residue i (B→A)

    for (int j = 0; j < n_res; j++) {
        if (j == i) continue;
        float te_ab = te_a_to_b[i * n_res + j];
        float te_ba = te_b_to_a[j * n_res + i];
        mi_sum += (te_ab + te_ba) * 0.5f;
        te_out += te_ab;
        te_in += te_ba;
    }

    mutual_info[i] = mi_sum;
    causal_flow[i] = te_out - te_in;  // Positive = net information source
}
