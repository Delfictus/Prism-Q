// ═══════════════════════════════════════════════════════════════════════
// PRISM-4D / Continuous Learning — RichSpike (CUDA implementation)
// ═══════════════════════════════════════════════════════════════════════
//
// Phase 1 of CLA: link-probe shim only. The RichSpike struct + helpers
// are header-only in `rich_spike.cuh`; this .cu exists to anchor the
// FFI link probe + future kernels (CLA-1b: Morton encoder upgrade
// to consume RichSpike, CLA-2: compression kernel for telemetry).
//
// Compilation: nvcc -arch=sm_120 -O3 --use_fast_math --restrict
//              --expt-relaxed-constexpr -std=c++17 -Xcompiler -fPIC -c
// ═══════════════════════════════════════════════════════════════════════

#include "rich_spike.cuh"

namespace prism_nhs { namespace rich_spike {

extern "C" {

uint32_t prism_rich_spike_link_probe(void) {
    // Sentinel 0x6164 ("ad" — for "rich [a]d[ata]"). Pinned by
    // Rust-side test.
    return 0x6164u;
}

}  // extern "C"

}}  // namespace prism_nhs::rich_spike
