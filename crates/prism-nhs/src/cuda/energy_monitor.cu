// ═══════════════════════════════════════════════════════════════════════════
// PRISM-4D / M1.2.17 — Hamiltonian Auditor implementation (f64 → f64)
// ═══════════════════════════════════════════════════════════════════════════

#include "energy_monitor.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>  // Amendment 3.10 — kernel-level printf triage

// CUB device-wide reduction.  CUDA 13.x ships CUB under
// /usr/local/cuda/include/cccl/cub; build.rs adds that include path.
#include <cub/cub.cuh>

// ─── Window update kernel ─────────────────────────────────────────────────
//
// Single-thread <<<1,1>>>.  Reads *d_pe_scalar (the CUB-reduce f64
// output) and rolls the energy window:
//   window->prev = window->cur
//   window->cur  = *d_pe_scalar
// ALSO writes the same scalar into *d_adj_pe_target — the
// `adj.potential_energy` field (offset 112) so the SFA reads the
// latest V_t directly from the FFI struct without an extra memcpy.

extern "C"
__global__ void prism_energy_monitor_window_update_kernel(
    EnergyWindow*  __restrict__ window,
    const double*  __restrict__ d_pe_scalar,
    double*        __restrict__ d_adj_pe_target
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (window == nullptr || d_pe_scalar == nullptr) return;

    const double new_cur = *d_pe_scalar;
    window->prev = window->cur;
    window->cur  = new_cur;
    if (d_adj_pe_target != nullptr) {
        *d_adj_pe_target = new_cur;
    }
}

// ─── Host-side launchers ──────────────────────────────────────────────────

extern "C"
int prism_energy_monitor_temp_storage_bytes(
    uint32_t n,
    size_t*  out_temp_bytes)
{
    if (out_temp_bytes == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    size_t bytes = 0;
    double*  d_in_dummy   = nullptr;
    double*  d_out_dummy  = nullptr;
    cudaError_t err = cub::DeviceReduce::Sum(
        /*d_temp_storage=*/      nullptr,
        /*temp_storage_bytes=*/  bytes,
        /*d_in=*/                d_in_dummy,
        /*d_out=*/               d_out_dummy,
        /*num_items=*/           static_cast<int>(n));
    if (err != cudaSuccess) return static_cast<int>(err);

    *out_temp_bytes = bytes;
    return static_cast<int>(cudaSuccess);
}

extern "C"
int prism_energy_monitor_launch_reduce(
    const double*   d_pe_components,
    uint32_t        n,
    void*           d_temp_storage,
    size_t          temp_storage_bytes,
    double*         d_pe_scalar,
    EnergyWindow*   d_energy_window,
    double*         d_adj_pe_target,
    void*           stream)
{
    if (d_pe_components == nullptr || d_pe_scalar == nullptr ||
        d_energy_window == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaStream_t s = static_cast<cudaStream_t>(stream);

    // CUB DeviceReduce::Sum on f64 → f64 scalar.  Captured into the
    // graph as a series of cub::DeviceReduce::Sum-internal nodes;
    // CUDA 12.4+ supports CUB device-wide algorithm capture.
    cudaError_t err = cub::DeviceReduce::Sum(
        d_temp_storage,
        temp_storage_bytes,
        d_pe_components,
        d_pe_scalar,
        static_cast<int>(n),
        s);
    if (err != cudaSuccess) return static_cast<int>(err);

    // Window roll + adj.potential_energy write in a single 1-thread kernel.
    prism_energy_monitor_window_update_kernel<<<1, 1, 0, s>>>(
        d_energy_window, d_pe_scalar, d_adj_pe_target);
    return static_cast<int>(cudaGetLastError());
}
