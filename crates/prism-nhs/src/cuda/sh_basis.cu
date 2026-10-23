// ═══════════════════════════════════════════════════════════════════════
// PRISM-4D / RECT-3.1 — Spherical Harmonics Y_lm (CUDA implementation)
// ═══════════════════════════════════════════════════════════════════════
//
// See `sh_basis.cuh` for layout + caller contract.
// ═══════════════════════════════════════════════════════════════════════

#include "sh_basis.cuh"

#include <cuda_runtime.h>
#include <cstdint>

namespace prism_nhs { namespace sh_basis {

// Storage for the K_LM table. Lives in this translation unit only;
// other .cu files (e.g. RECT-3.1.b's so3_project_manifold_kernel)
// receive K_LM via a const-pointer kernel argument set by the host
// orchestrator after `prism_sh_basis_init` populates this symbol.
__device__ float K_LM[N_COEFFS];

// ─────────────────────────────────────────────────────────────────────
// Init kernel: populates K_LM via the canonical formula.
//   K_lm (m=0)  = sqrt((2l+1) / (4π))
//   K_lm (m≠0)  = sqrt((2l+1) / (2π) * (l-|m|)! / (l+|m|)!)
//
// Single-thread launch; runs once per process.
// ─────────────────────────────────────────────────────────────────────
__global__ void prism_sh_init_constants_kernel(void) {
    if (threadIdx.x != 0u || blockIdx.x != 0u) return;

    constexpr float TWO_PI_F  = 6.28318530717959f;
    constexpr float FOUR_PI_F = 12.5663706143592f;

    // Per-(l, |m|) factorial ratio (l-|m|)! / (l+|m|)! for l in [0,5],
    // |m| in [0, l]. We unroll the small loop manually because |m|
    // ranges 0..5 and we want a single straight-line execution.
    for (int l = 0; l <= LMAX; ++l) {
        for (int m = -l; m <= l; ++m) {
            const int idx = l * (l + 1) + m;
            const int am = (m < 0) ? -m : m;

            // Factorial ratio (l-am)! / (l+am)!  for am in [0, l].
            // Compute as 1 / Π_{i=l-am+1..l+am} i.
            float denom = 1.0f;
            for (int i = l - am + 1; i <= l + am; ++i) {
                denom *= static_cast<float>(i);
            }
            const float fact_ratio = 1.0f / denom;

            float k;
            if (m == 0) {
                k = sqrtf(static_cast<float>(2 * l + 1) / FOUR_PI_F);
            } else {
                k = sqrtf(static_cast<float>(2 * l + 1) / TWO_PI_F * fact_ratio);
            }
            K_LM[idx] = k;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Test-driver kernel: one thread per (theta, phi) point. Calls the
// straight-line evaluator; writes 36 floats per point.
// ─────────────────────────────────────────────────────────────────────
__global__ void prism_sh_eval_kernel(
    const float* __restrict__ d_theta_phi,
    uint32_t                  n_points,
    float* __restrict__       d_Y_out
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_points) return;

    const float theta = d_theta_phi[2u * tid + 0u];
    const float phi   = d_theta_phi[2u * tid + 1u];

    float Y[N_COEFFS];
    prism_sh_eval_lmax5(theta, phi, K_LM, Y);

    #pragma unroll
    for (int i = 0; i < N_COEFFS; ++i) {
        d_Y_out[N_COEFFS * tid + i] = Y[i];
    }
}

// ─────────────────────────────────────────────────────────────────────
// extern "C" host orchestration
// ─────────────────────────────────────────────────────────────────────

extern "C" {

uint32_t prism_sh_basis_link_probe(void) {
    return 0x5BACu;
}

cudaError_t prism_sh_basis_init(cudaStream_t stream) {
    prism_sh_init_constants_kernel<<<1, 1, 0, stream>>>();
    return cudaGetLastError();
}

cudaError_t prism_sh_eval_run(
    const float* d_theta_phi,
    uint32_t     n_points,
    float*       d_Y_out,
    cudaStream_t stream
) {
    if (n_points == 0u) return cudaSuccess;
    if (d_theta_phi == nullptr || d_Y_out == nullptr) {
        return cudaErrorInvalidValue;
    }

    constexpr uint32_t TPB = 128u;
    const uint32_t blocks = (n_points + TPB - 1u) / TPB;
    prism_sh_eval_kernel<<<blocks, TPB, 0, stream>>>(
        d_theta_phi, n_points, d_Y_out
    );
    return cudaGetLastError();
}

cudaError_t prism_sh_basis_get_k_lm_dev_ptr(const float** out_dev_ptr) {
    if (out_dev_ptr == nullptr) return cudaErrorInvalidValue;
    void* sym = nullptr;
    const cudaError_t rc = cudaGetSymbolAddress(&sym, K_LM);
    if (rc != cudaSuccess) return rc;
    *out_dev_ptr = static_cast<const float*>(sym);
    return cudaSuccess;
}

}  // extern "C"

}}  // namespace prism_nhs::sh_basis

// Exact module state transfer at an all-worker quiescent boundary.
// Caller owns the context, holds every producer and synchronizes before entry.
extern "C" size_t prism_checkpoint_sh_basis_bytes() { return sizeof(prism_nhs::sh_basis::K_LM); }
extern "C" int prism_checkpoint_sh_basis_capture(unsigned char* bytes, size_t length, cudaStream_t stream) {
    if (!bytes || length != prism_checkpoint_sh_basis_bytes()) return (int)cudaErrorInvalidValue;
    size_t offset = 0; cudaError_t rc;
    rc = cudaMemcpyFromSymbolAsync(bytes + offset, prism_nhs::sh_basis::K_LM, sizeof(prism_nhs::sh_basis::K_LM), 0, cudaMemcpyDeviceToHost, stream);
    if (rc != cudaSuccess) return (int)rc;
    offset += sizeof(prism_nhs::sh_basis::K_LM);
    return (int)cudaStreamSynchronize(stream);
}
extern "C" int prism_checkpoint_sh_basis_restore(const unsigned char* bytes, size_t length, cudaStream_t stream) {
    if (!bytes || length != prism_checkpoint_sh_basis_bytes()) return (int)cudaErrorInvalidValue;
    size_t offset = 0; cudaError_t rc;
    rc = cudaMemcpyToSymbolAsync(prism_nhs::sh_basis::K_LM, bytes + offset, sizeof(prism_nhs::sh_basis::K_LM), 0, cudaMemcpyHostToDevice, stream);
    if (rc != cudaSuccess) return (int)rc;
    offset += sizeof(prism_nhs::sh_basis::K_LM);
    return (int)cudaStreamSynchronize(stream);
}
