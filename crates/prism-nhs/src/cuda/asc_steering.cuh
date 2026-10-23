// ═══════════════════════════════════════════════════════════════════════════
// PRISM-4D / ASC Steering — Vectorized Repulsion Kernel declarations
//
// asc_inject_repulsion_v4_kernel: applies F_i = α × ΔKL × (pos_i − c)
// for each atom. SM120+ uses Blackwell-native atom.global.add.v4.f32 on
// full aligned quads; SM70-SM89 uses explicit PTX red.global.add.f32 lane
// reductions with the same accumulation semantics.
//
// Gate G22_ATOMIC_v4_VERIFIED: ptxas --verbose audit required before
// production deployment (see Phase 2 validation in ZSTR roadmap).
//
// Compilation: nvcc -arch=sm_120 -O3 --use_fast_math --restrict
//              --expt-relaxed-constexpr -std=c++17 -c
// ═══════════════════════════════════════════════════════════════════════════

#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace prism_nhs { namespace asc {

/// Vectorized ASC repulsion force injection.
///
/// Adds F_i = alpha_gain * delta_kl * (pos_i - centroid) to d_forces
/// for each atom. The force buffer layout is AoS: for n_atoms atoms
/// d_forces stores [fx0,fy0,fz0, fx1,fy1,fz1, ...] as n_atoms*3 f32.
/// SM120+ full quads are reinterpreted as float4 for vector atomics; all
/// other lanes use bounded flat-lane reduction so tail values are never
/// read or written past n_atoms*3.
///
/// @param d_forces     VRAM force accumulator (n_atoms * 3 × f32, AoS).
///                     MUST be at least f32-aligned; SM120 vector quads also
///                     require the underlying allocation to satisfy float4
///                     alignment for full-quad vector atomic execution.
/// @param d_pos        VRAM atom positions (n_atoms * 3 × f32, same AoS).
/// @param centroid     Pocket centroid (Å).
/// @param alpha_gain   ASC steering gain coefficient.
/// @param delta_kl     SURP score (KL divergence from adjudicator).
/// @param n_atoms      Total atom count.
///
/// Launch geometry: grid_x = ceil(n_atoms * 3 / 4), block = 128.
/// Gate: G22_ATOMIC_v4_VERIFIED on SM120; PrismOS laptop SM86 must instead
/// disassemble to red.global.add.f32 lane reductions.
extern "C"
__global__ void asc_inject_repulsion_v4_kernel(
    float4* __restrict__       d_forces,
    const float4* __restrict__ d_pos,
    float3                     centroid,
    float                      alpha_gain,
    float                      delta_kl,
    uint32_t                   n_atoms
);

/// ABI-preserving selected-support owner entry points. The device header stays
/// PrismAscDeviceV1/736 B; mode 3 assigns semantics to its reserved fields.
extern "C" int prism_asc_create_selected_v2(
    uint64_t* out,
    uint32_t source,
    uint64_t generation,
    uint32_t atoms,
    const uint64_t* topology,
    float alpha,
    uint64_t support_identity,
    const uint8_t* membership_host,
    uint32_t selected_count,
    void* stream);

extern "C" int prism_asc_compose_force_selected_v2(
    uint64_t ptr, uint32_t n, uint32_t phase, const float* x, float* f,
    double* pe, double* work, void* stream);

extern "C" int prism_asc_seal_step_selected_v2(uint64_t ptr, void* stream);

extern "C" int prism_asc_publish_decision_selected_v2(
    uint64_t ptr, const void* adj, void* stream);

}} // namespace prism_nhs::asc
