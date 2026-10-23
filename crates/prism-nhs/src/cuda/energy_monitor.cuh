// ═══════════════════════════════════════════════════════════════════════════
// PRISM-4D / B.3.2 — Hamiltonian Auditor (CUB-based potential-energy reduce)
// ═══════════════════════════════════════════════════════════════════════════
//
// Operator directive 2026-05-02 §1 ("Hamiltonian Auditor"):
//
//   "We require V to gate the gear-shift logic and prevent kinetic
//    explosions.  Utilize cub::DeviceReduce::Sum to perform a single-
//    pass reduction of the d_potential_energy_components buffer."
//
// Wave B.3.2 surface — the SCAFFOLD landed.  The CUB reduce kernel
// below reads from a `float* d_pe_components` (per-atom PE buffer
// owned by NhsAmberFusedEngine) and writes a single f64 to
// `d_pe_scalar`.  A second kernel rolls the energy window:
// E_prev := E_now; E_now := *d_pe_scalar.
//
// **Honest scaffold note**: the AMBER force kernels in
// crates/prism-gpu/src/kernels/nhs_amber_fused.cu currently write
// FORCES only (not per-atom PE).  The d_pe_components buffer is
// allocated and zero-filled each step; the reduce returns 0.0f64
// until the AMBER kernels are extended to also accumulate per-atom
// U(r) contributions.  That AMBER-side wiring is a separate Wave
// (M1.2.17 — "Hamiltonian Recovery").
//
// The B.3.2 commit lands the FFI surface, the captured-graph node,
// the energy-window scratch, and the SFA stability-fuse hook so that
// when M1.2.17 wires the AMBER PE outputs, the stability gate fires
// without further integration work.
//
// Compilation: nvcc -arch=sm_120 -O3 --use_fast_math --restrict
//              --expt-relaxed-constexpr -std=c++17 -Xcompiler -fPIC -c
//              -I/usr/local/cuda/include/cccl
// ═══════════════════════════════════════════════════════════════════════════

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/// B.3.2 — Energy-window scratch buffer.  16 bytes (2 × f64), 16-byte
/// aligned.  The captured-graph energy-monitor sequence updates this
/// at the end of every launch:
///
///   prev := cur;            (E_n   becomes E_{n-1})
///   cur  := *d_pe_scalar;   (latest CUB-reduced PE)
///
/// The SFA kernel (B.3.2-B) reads both values to compute the
/// rolling-window drift |cur - prev| / |prev| and routes Gear 3
/// (Abort) when drift > 0.01.
struct EnergyWindow {
    double prev;     // E_{n-1}
    double cur;      // E_n
};

static_assert(sizeof(EnergyWindow) == 16,
              "EnergyWindow MUST be 16 bytes.");

/// M1.2.17 — Returns the temp-storage size required by the CUB
/// DeviceReduce::Sum operation on `n` f64 inputs (the per-atom PE
/// components buffer).  Caller allocates a device buffer of this size
/// and passes it to `prism_energy_monitor_launch_reduce`.
///
/// Returns cudaError_t cast to int; on success writes the size into
/// *out_temp_bytes.  Required-temp-storage is (currently) ~ a few KB
/// for any practical n; this query is cheap and runs once at build.
int prism_energy_monitor_temp_storage_bytes(
    uint32_t n,
    size_t*  out_temp_bytes);

/// M1.2.17 — Records two captured graph kernel nodes onto `stream`:
///
///   1. CUB DeviceReduce::Sum reducing `d_pe_components` (n f64)
///      into `*d_pe_scalar` (single f64) — Hamiltonian Auditor.
///   2. Single-thread `prism_energy_monitor_window_update_kernel` —
///      copies window->cur into window->prev, then writes
///      window->cur := *d_pe_scalar.  ALSO writes the same scalar
///      into `*d_adj_pe_target` (= &adj.d_potential_energy at
///      offset 112) so the SFA stability-fuse logic reads the
///      latest V_t directly from the FFI struct.
///
/// `d_temp_storage` is the device buffer sized via
/// `prism_energy_monitor_temp_storage_bytes`.
int prism_energy_monitor_launch_reduce(
    const double*  d_pe_components,
    uint32_t       n,
    void*          d_temp_storage,
    size_t         temp_storage_bytes,
    double*        d_pe_scalar,
    EnergyWindow*  d_energy_window,
    double*        d_adj_pe_target,
    void*          stream);

#ifdef __cplusplus
}
#endif
