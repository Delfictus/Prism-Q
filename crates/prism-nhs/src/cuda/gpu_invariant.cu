// ═══════════════════════════════════════════════════════════════════════
// PRISM-4D / Rectification Phase 1 — Hard-Trap Invariant (CUDA impl)
// ═══════════════════════════════════════════════════════════════════════
//
// See `gpu_invariant.cuh` for rationale + caller contract.
//
// Compilation: nvcc -arch=sm_120 -O3 --use_fast_math --restrict
//              --expt-relaxed-constexpr -std=c++17 -Xcompiler -fPIC -c
// ═══════════════════════════════════════════════════════════════════════

#include "gpu_invariant.cuh"

#include <cuda_runtime.h>
#include <cstdint>

namespace prism_nhs { namespace gpu_invariant {

// ═══════════════════════════════════════════════════════════════════════
// __global__ kernel: audit_mass_conservation
//
// Single-thread launch (`<<<1, 1, 0, stream>>>`). One thread reads
// the two u64 device-side counters, sums them, and hard-asserts
// equality with `expected_total_input_spikes`. On violation the warp
// traps; the host's stream-synchronize observes a non-success error.
// ═══════════════════════════════════════════════════════════════════════

__global__ void audit_mass_conservation_kernel(
    const uint64_t* __restrict__ d_total_attributed,
    const uint64_t* __restrict__ d_background_count,
    unsigned long long expected_total_input_spikes
) {
    if (threadIdx.x != 0u || blockIdx.x != 0u) return;

    // Volatile cast on the dereferences so the compiler can't fuse
    // the loads with prior atomic ops in surprising ways. We want
    // the read to happen here, after every prior stream-ordered
    // mutation has retired.
    const unsigned long long attributed =
        *reinterpret_cast<const volatile unsigned long long*>(d_total_attributed);
    const unsigned long long background =
        *reinterpret_cast<const volatile unsigned long long*>(d_background_count);

    const unsigned long long actual = attributed + background;
    gpu_hard_assert_eq_u64(actual, expected_total_input_spikes);
}

// ═══════════════════════════════════════════════════════════════════════
// extern "C" host orchestration
// ═══════════════════════════════════════════════════════════════════════

extern "C" {

uint32_t prism_gpu_invariant_link_probe(void) {
    // Sentinel: 0xA55E47 = "ASSERT" (rough). Rust-side test pin.
    return 0xA55E47u;
}

cudaError_t prism_audit_mass_conservation(
    const uint64_t* d_total_attributed,
    const uint64_t* d_background_count,
    uint64_t expected_total_input_spikes,
    cudaStream_t stream
) {
    if (d_total_attributed == nullptr || d_background_count == nullptr) {
        return cudaErrorInvalidValue;
    }
    audit_mass_conservation_kernel<<<1, 1, 0, stream>>>(
        d_total_attributed,
        d_background_count,
        static_cast<unsigned long long>(expected_total_input_spikes)
    );
    return cudaGetLastError();
}

}  // extern "C"

}}  // namespace prism_nhs::gpu_invariant
