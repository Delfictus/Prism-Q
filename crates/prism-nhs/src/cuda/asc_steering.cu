// ═══════════════════════════════════════════════════════════════════════════
// PRISM-4D / ASC Steering — Vectorized Repulsion Kernel (impl)
//
// Implements asc_inject_repulsion_v4_kernel using hardware-specific force
// accumulation:
//   - SM120+ uses Blackwell atom.global.add.v4.f32 for full aligned quads.
//   - SM70-SM89 uses PTX red.global.add.f32, a no-return global reduction
//     instruction, for each valid force lane.
//
// This is not a generic C++ atomicAdd fallback. The PrismOS laptop beta target
// (RTX 3060 Laptop, sm_86) cannot assemble atom.global.add.v4.f32, so the
// native Ampere path is expressed as explicit PTX reductions with bounds checks.
//
// ── Design Notes ────────────────────────────────────────────────────────────
//
// Force buffer layout: AoS, [fx0,fy0,fz0, fx1,fy1,fz1, ...], n_atoms*3 f32.
// Reinterpreted as float4*: each float4 quad covers 4 consecutive f32 values
// which span atom boundaries.  The centroid pattern cycles with period 3:
//   quad offset 0 mod 3 → [cx, cy, cz, cx]
//   quad offset 1 mod 3 → [cy, cz, cx, cy]
//   quad offset 2 mod 3 → [cz, cx, cy, cz]
// This correctly applies F_i = α * ΔKL * (pos_i - c_i) per f32 component.
//
// ── Gate G22_ATOMIC_v4_VERIFIED ─────────────────────────────────────────────
// ptxas audit required:
//   nvcc -arch=sm_120 --ptx asc_steering.cu -o /tmp/asc.ptx
//   grep "atom.global.add.v4.f32" /tmp/asc.ptx
// If the target architecture does not support that vector atomic form, the
// PTX reduction path must be present and audited in disassembly. This is
// required for the PrismOS laptop beta target and is not a missing kernel
// surface.
//
// Compilation: nvcc -arch=sm_120 -O3 --use_fast_math --restrict
//              --expt-relaxed-constexpr -std=c++17 -Xcompiler -fPIC -c
// ═══════════════════════════════════════════════════════════════════════════

#include "asc_steering.cuh"
#include "prism_asc_decision.cuh"
#include "adjudicator.cuh"
#include "../../../prism-gpu/src/kernels/nhs_biprism_coordinate_precision.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace prism_nhs { namespace asc {

// Centroid component at a given flat f32 index (cycles x→y→z→x→...).
__device__ __forceinline__
float centroid_at(const float3 c, uint32_t flat_idx) {
    switch (flat_idx % 3u) {
        case 0u: return c.x;
        case 1u: return c.y;
        default: return c.z;
    }
}

__device__ __forceinline__
void asc_reduce_add_f32(float* addr, float value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    // Fire-and-forget global reduction. ASC does not need the pre-addition
    // value, so this avoids the return dependency of atomicAdd while preserving
    // inter-kernel accumulation semantics.
    asm volatile(
        "red.global.add.f32 [%0], %1;"
        :
        : "l"((unsigned long long)(uintptr_t)addr), "f"(value)
        : "memory"
    );
#else
    atomicAdd(addr, value);
#endif
}

extern "C"
__global__ void asc_inject_repulsion_v4_kernel(
    float4* __restrict__       d_forces,
    const float4* __restrict__ d_pos,
    float3                     centroid,
    float                      alpha_gain,
    float                      delta_kl,
    uint32_t                   n_atoms
) {
    const uint32_t n_floats = n_atoms * 3u;
    const uint32_t n_quads  = (n_floats + 3u) >> 2u;  // ceil(n_floats / 4)
    const uint32_t quad_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (quad_idx >= n_quads) return;

    const float scale = alpha_gain * delta_kl;

    // Base flat-f32 index for this quad.
    const uint32_t base = quad_idx << 2u;  // quad_idx * 4

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    // ── Blackwell vectorized atomic force accumulation ───────────────────
    // Blackwell sm_120: atom.global.add.v4.f32 applies four concurrent
    // 32-bit float atomicAdds in a single instruction, matching the
    // 128-bit memory-access width of the L2 cache line.
    //
    // G22 gate: if ptxas disassembly does NOT show atom.global.add.v4.f32,
    // the four-atomicAdd fallback below is the correct equivalence.
    //
    if (base + 3u < n_floats) {
        // Load 4 atom-position floats (may span two atoms), but only on full
        // quads. Tail quads are handled by the bounded reduction path below.
        float4 pos_quad = __ldg(&d_pos[quad_idx]);

        // Compute repulsion delta per f32 component.
        // centroid_at() resolves the correct X/Y/Z for each flat index.
        float4 force_delta;
        force_delta.x = scale * (pos_quad.x - centroid_at(centroid, base + 0u));
        force_delta.y = scale * (pos_quad.y - centroid_at(centroid, base + 1u));
        force_delta.z = scale * (pos_quad.z - centroid_at(centroid, base + 2u));
        force_delta.w = scale * (pos_quad.w - centroid_at(centroid, base + 3u));

        // Output registers capture the PRE-addition values (standard atom
        // semantics); we discard them via separate temporaries.
        float old_x, old_y, old_z, old_w;
        asm volatile(
            "atom.global.add.v4.f32 {%0,%1,%2,%3}, [%4], {%5,%6,%7,%8};"
            : "=f"(old_x), "=f"(old_y), "=f"(old_z), "=f"(old_w)
            : "l"((unsigned long long)(uintptr_t)(&d_forces[quad_idx])),
              "f"(force_delta.x), "f"(force_delta.y),
              "f"(force_delta.z), "f"(force_delta.w)
            : "memory"
        );
        // old_x/y/z/w discarded — only the side-effect (in-place VRAM add) matters.
        (void)old_x; (void)old_y; (void)old_z; (void)old_w;
        return;
    }
#endif

    // ── SM86 / tail path: bounded PTX global reductions ──────────────────
    // Work directly on the flat AoS lanes so the final quad never reads or
    // writes past n_atoms*3.
    const float* pos_f = reinterpret_cast<const float*>(d_pos);
    float* force_f = reinterpret_cast<float*>(d_forces);

    if (base + 0u < n_floats) {
        const float v = scale * (__ldg(pos_f + base + 0u) - centroid_at(centroid, base + 0u));
        asc_reduce_add_f32(force_f + base + 0u, v);
    }
    if (base + 1u < n_floats) {
        const float v = scale * (__ldg(pos_f + base + 1u) - centroid_at(centroid, base + 1u));
        asc_reduce_add_f32(force_f + base + 1u, v);
    }
    if (base + 2u < n_floats) {
        const float v = scale * (__ldg(pos_f + base + 2u) - centroid_at(centroid, base + 2u));
        asc_reduce_add_f32(force_f + base + 2u, v);
    }
    if (base + 3u < n_floats) {
        const float v = scale * (__ldg(pos_f + base + 3u) - centroid_at(centroid, base + 3u));
        asc_reduce_add_f32(force_f + base + 3u, v);
    }
}

__global__ void prism_asc_init_selected_v2(
    PrismAscDeviceV1* d, PrismAscDeviceV1 initial, uint8_t* mask) {
    if (threadIdx.x || blockIdx.x) return;
    *d = initial;
    d->rows = reinterpret_cast<PrismAscAtomV1*>(d + 1);
    prism_asc_initialize_work_tail(d);
    d->reserved3 = reinterpret_cast<uint64_t>(mask);
}

__global__ void prism_asc_selected_centroid_v2(
    PrismAscDeviceV1* d, const float* positions) {
    if (threadIdx.x || blockIdx.x) return;
    if (!d || !positions || d->mode != ASC_SELECTED_STRUCTURAL_SUPPORT ||
        d->reserved == 0 || d->reserved3 == 0) {
        if (d) d->status = ASC_BAD_DECISION;
        asm volatile("trap;");
        return;
    }
    double sum[3] = {0.0, 0.0, 0.0};
    uint32_t count = 0;
    for (uint32_t i = 0; i < d->n_atoms; ++i) {
        if (!prism_asc_atom_selected(*d, i)) continue;
        for (int j = 0; j < 3; ++j) sum[j] += static_cast<double>(positions[3 * i + j]);
        ++count;
    }
    if (count != d->reserved) {
        d->status = ASC_BAD_DECISION;
        asm volatile("trap;");
        return;
    }
    for (int j = 0; j < 3; ++j) {
        // This is decision input. The active center remains immutable until a
        // completed publication is promoted at the next physical begin.
        d->pending.center[j] = static_cast<float>(sum[j] / static_cast<double>(count));
        if (!isfinite(d->pending.center[j])) {
            d->status = ASC_NONFINITE;
            asm volatile("trap;");
            return;
        }
    }
}

__global__ void prism_asc_compose_selected_v2(
    PrismAscDeviceV1* d, uint32_t phase, const float* x, float* f,
    double* pe, double* work) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d->n_atoms) return;
    prism_asc_compose(d, i, d->n_atoms, phase, x + 3 * i, f + 3 * i,
                      pe ? pe + i : nullptr, work ? work + i : nullptr);
}

// Explicit precision profile. Its preflight finishes on the same stream before
// any force row is changed; a bad descriptor/value latches the actual physical
// leaf's fault sink and prevents this phase from being mistaken for completion.
__global__ void prism_asc_coordinates_preflight_v1(
    PrismAscDeviceV1* d,uint32_t n,const float* x,
    const BiprismCoordinateOwnerFfi* owners,uint32_t leaf,uint32_t* fault,
    uint32_t expected_mode) {
    if(threadIdx.x||blockIdx.x)return;
    if(!fault)return; // The host entrypoint rejects a missing sink.
    if(*fault)return;
    bool valid=d&&owners&&!(reinterpret_cast<uintptr_t>(owners)&15u)&&leaf<8u&&x&&n&&d->n_atoms==n&&d->mode==expected_mode;
    if(valid){const auto& o=owners[leaf];
        valid=bp_coord_owner_valid(&o,leaf,n,o.generation,BP_COORD_NATIVE_ANGSTROM,
            BP_COORD_CURRENT_INITIALIZED,x);
        for(uint32_t i=0;valid&&i<n;++i)
            valid=bp_coord_finite(bp_coord_load(o,BiprismCoordinatePlane::Current,i));
    }
    if(!valid){if(d)d->status=ASC_BAD_DECISION;atomicOr(fault,0x10000000u);}
}

__global__ void prism_asc_selected_centroid_compensated_v2(
    PrismAscDeviceV1* d,const BiprismCoordinateOwnerFfi* owners,
    uint32_t leaf,uint32_t* fault) {
    if(threadIdx.x||blockIdx.x||*fault)return;
    if(d->mode!=ASC_SELECTED_STRUCTURAL_SUPPORT||!d->reserved||!d->reserved3){
        d->status=ASC_BAD_DECISION;atomicOr(fault,0x10000000u);return;
    }
    const auto& o=owners[leaf];double sx=0.,sy=0.,sz=0.;uint32_t count=0;
    for(uint32_t i=0;i<d->n_atoms;++i){
        if(!prism_asc_atom_selected(*d,i))continue;
        const auto p=bp_coord_value(bp_coord_load(o,BiprismCoordinatePlane::Current,i));
        sx+=p.x;sy+=p.y;sz+=p.z;++count;
    }
    if(!count||count!=d->reserved){d->status=ASC_BAD_DECISION;atomicOr(fault,0x10000000u);return;}
    const float cx=float(sx/double(count)),cy=float(sy/double(count)),cz=float(sz/double(count));
    if(!isfinite(cx)||!isfinite(cy)||!isfinite(cz)){
        d->status=ASC_BAD_DECISION;atomicOr(fault,0x10000000u);return;
    }
    // The controller's published center remains an explicitly FP32 quantity.
    // It is never used as a replacement for a physical coordinate low plane.
    d->pending.center[0]=cx;d->pending.center[1]=cy;d->pending.center[2]=cz;
}

__global__ void prism_asc_compose_compensated_v1(
    PrismAscDeviceV1* d,uint32_t phase,float* f,double* pe,double* work,
    const BiprismCoordinateOwnerFfi* owners,uint32_t leaf,const uint32_t* fault) {
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(*fault||i>=d->n_atoms)return;
    const auto p=bp_coord_value(bp_coord_load(owners[leaf],BiprismCoordinatePlane::Current,i));
    const double x[3]={p.x,p.y,p.z};
    prism_asc_compose_f64(d,i,d->n_atoms,phase,x,f+3u*i,pe+i,work+i);
}

__global__ void prism_asc_seal_selected_v2(PrismAscDeviceV1* d) {
    if (threadIdx.x || blockIdx.x) return;
    if (!d || d->mode != ASC_SELECTED_STRUCTURAL_SUPPORT || d->reserved == 0) {
        if (d) d->status = ASC_BAD_DECISION;
        asm volatile("trap;");
        return;
    }
    uint32_t selected = 0;
    for (uint32_t i = 0; i < d->n_atoms; ++i) {
        if (!prism_asc_atom_selected(*d, i)) continue;
        ++selected;
        const PrismAscAtomV1& r = d->rows[i];
        bool covered = true;
        for (int k = 0; k < 4; ++k)
            covered &= r.epoch[k] == d->physical_epoch &&
                       r.decision_epoch[k] == d->active.producer_epoch;
        if (!covered) { ++d->coverage_failures; continue; }
        d->control_work += r.control_work;
        for (int p = 0; p < 2; ++p) {
            PrismAscPhaseV1& s = d->phases[p];
            ++s.composed_atoms;
            ++s.consumed_atoms;
            bool nonzero_force = false, nonzero_impulse = false;
            for (int j = 0; j < 3; ++j) {
                s.force[j] += r.force[p][j];
                s.torque[j] += r.torque[p][j];
                s.impulse[j] += r.impulse[p][j];
                s.velocity_delta[j] += r.velocity_delta[p][j];
                nonzero_force |= r.force[p][j] != 0.0;
                nonzero_impulse |= r.impulse[p][j] != 0.0;
            }
            s.nonzero_force_atoms += nonzero_force;
            s.nonzero_impulse_atoms += nonzero_impulse;
            s.potential += r.potential[p];
            s.differential_kick_energy += r.differential_kick_energy[p];
        }
    }
    bool finite = selected == d->reserved && isfinite(d->control_work);
    for (int p = 0; p < 2; ++p) {
        const PrismAscPhaseV1& s = d->phases[p];
        finite &= isfinite(s.potential) && isfinite(s.differential_kick_energy);
        for (int j = 0; j < 3; ++j)
            finite &= isfinite(s.force[j]) && isfinite(s.torque[j]) &&
                      isfinite(s.impulse[j]) && isfinite(s.velocity_delta[j]);
    }
    if (d->coverage_failures || !finite) {
        d->status = d->coverage_failures ? ASC_INCOMPLETE_COVERAGE : ASC_NONFINITE;
        asm volatile("trap;");
        return;
    }
    prism_asc_commit_control_work(d);
    d->completed_epoch = d->physical_epoch;
    ++d->composed_steps;
    ++d->consumed_steps;
    d->status = d->active.producer_epoch ? ASC_COMPLETE : ASC_NO_PRIOR_DECISION;
}

__global__ void prism_asc_publish_selected_v2(
    PrismAscDeviceV1* d, const InterferometricAdjudicatorFfi* adj) {
    if (threadIdx.x || blockIdx.x) return;
    if (!d || !adj || d->mode != ASC_SELECTED_STRUCTURAL_SUPPORT ||
        d->physical_epoch == 0 || d->completed_epoch != d->physical_epoch ||
        d->published_epoch >= d->physical_epoch) {
        if (d) d->status = ASC_BAD_DECISION;
        asm volatile("trap;");
        return;
    }
    PrismAscControlV1 c = {};
    c.source = d->source;
    c.mode = d->mode;
    c.generation = d->generation;
    c.producer_epoch = d->completed_epoch;
    c.producer_frame = d->completed_epoch;
    c.support_identity = d->reserved2;
    c.status = ASC_COMPLETE;
    for (int j = 0; j < 4; ++j) c.topology[j] = d->topology[j];
    for (int j = 0; j < 3; ++j) c.center[j] = d->pending.center[j];
    if (adj->adjudication_code > PRISM_ADJ_VIOLATION || !isfinite(adj->current_divergence)) {
        d->status = ASC_BAD_DECISION;
        asm volatile("trap;");
        return;
    }
    if (adj->adjudication_code == PRISM_ADJ_CONSTRUCT) {
        c.valid = 1;
        c.strength = d->alpha * adj->current_divergence;
    }
    if (!prism_asc_control_matches(c, *d, d->completed_epoch)) {
        d->status = ASC_BAD_DECISION;
        asm volatile("trap;");
        return;
    }
    d->pending = c;
    d->published_epoch = c.producer_epoch;
    ++d->decisions_produced;
}

extern "C" int prism_asc_create_selected_v2(
    uint64_t* out, uint32_t source, uint64_t generation, uint32_t atoms,
    const uint64_t* topology, float alpha, uint64_t support_identity,
    const uint8_t* membership_host, uint32_t selected_count, void* stream) {
    if (!out || !topology || !membership_host || !stream || !atoms || !generation ||
        !support_identity || !isfinite(alpha) || selected_count == 0 || selected_count > atoms)
        return static_cast<int>(cudaErrorInvalidValue);
    *out = 0;
    uint32_t measured_selected = 0;
    for (uint32_t i = 0; i < atoms; ++i) {
        if (membership_host[i] > 1)
            return static_cast<int>(cudaErrorInvalidValue);
        measured_selected += membership_host[i];
    }
    if (measured_selected != selected_count)
        return static_cast<int>(cudaErrorInvalidValue);
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    cudaStreamCaptureStatus capture;
    cudaError_t rc = cudaStreamIsCapturing(cuda_stream, &capture);
    if (rc != cudaSuccess) return static_cast<int>(rc);
    if (capture != cudaStreamCaptureStatusNone)
        return static_cast<int>(cudaErrorStreamCaptureUnsupported);
    const size_t fixed = sizeof(PrismAscDeviceV1) + sizeof(PrismAscWorkCumulativeV1);
    if (atoms > (SIZE_MAX - fixed) / sizeof(PrismAscAtomV1))
        return static_cast<int>(cudaErrorInvalidValue);
    const size_t rows_bytes = static_cast<size_t>(atoms) * sizeof(PrismAscAtomV1);
    if (atoms > SIZE_MAX - fixed - rows_bytes)
        return static_cast<int>(cudaErrorInvalidValue);
    const size_t bytes = fixed + rows_bytes + atoms;
    PrismAscDeviceV1* d = nullptr;
    rc = cudaMalloc(reinterpret_cast<void**>(&d), bytes);
    if (rc != cudaSuccess) return static_cast<int>(rc);
    uint8_t* mask = reinterpret_cast<uint8_t*>(d + 1) + rows_bytes + sizeof(PrismAscWorkCumulativeV1);
    rc = cudaMemsetAsync(d, 0, bytes, cuda_stream);
    if (rc == cudaSuccess)
        rc = cudaMemcpyAsync(mask, membership_host, atoms, cudaMemcpyHostToDevice, cuda_stream);
    if (rc != cudaSuccess) { cudaFree(d); return static_cast<int>(rc); }
    PrismAscDeviceV1 initial = {};
    initial.abi = 1;
    initial.byte_size = sizeof(initial);
    initial.source = source;
    initial.n_atoms = atoms;
    initial.generation = generation;
    initial.alpha = alpha;
    initial.mode = ASC_SELECTED_STRUCTURAL_SUPPORT;
    initial.reserved = selected_count;
    initial.reserved2 = support_identity;
    for (int j = 0; j < 4; ++j) initial.topology[j] = topology[j];
    prism_asc_init_selected_v2<<<1, 1, 0, cuda_stream>>>(d, initial, mask);
    rc = cudaGetLastError();
    if (rc != cudaSuccess) { cudaFree(d); return static_cast<int>(rc); }
    *out = reinterpret_cast<uint64_t>(d);
    return static_cast<int>(cudaSuccess);
}

extern "C" int prism_asc_compose_force_selected_v2(
    uint64_t ptr, uint32_t n, uint32_t phase, const float* x, float* f,
    double* pe, double* work, void* stream) {
    if (!ptr || !n || phase > 1 || !x || !f || !pe || !work || !stream)
        return static_cast<int>(cudaErrorInvalidValue);
    auto* d = reinterpret_cast<PrismAscDeviceV1*>(ptr);
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    prism_asc_selected_centroid_v2<<<1, 1, 0, cuda_stream>>>(d, x);
    prism_asc_compose_selected_v2<<<(n + 255) / 256, 256, 0, cuda_stream>>>(
        d, phase, x, f, pe, work);
    return static_cast<int>(cudaGetLastError());
}

static int prism_asc_compose_force_precision(
    uint64_t ptr,uint32_t n,uint32_t phase,const float* x,float* f,
    double* pe,double* work,void* stream,const BiprismCoordinateOwnerFfi* owners,
    uint32_t leaf,uint32_t* fault,uint32_t mode) {
    if(!ptr||!n||phase>1u||!x||!f||!pe||!work||!stream||!owners||leaf>=8u||!fault)
        return static_cast<int>(cudaErrorInvalidValue);
    auto* d=reinterpret_cast<PrismAscDeviceV1*>(ptr);
    const auto s=static_cast<cudaStream_t>(stream);
    prism_asc_coordinates_preflight_v1<<<1,1,0,s>>>(d,n,x,owners,leaf,fault,mode);
    cudaError_t rc=cudaGetLastError();if(rc!=cudaSuccess)return static_cast<int>(rc);
    if(mode==ASC_SELECTED_STRUCTURAL_SUPPORT){
        prism_asc_selected_centroid_compensated_v2<<<1,1,0,s>>>(d,owners,leaf,fault);
        rc=cudaGetLastError();if(rc!=cudaSuccess)return static_cast<int>(rc);
    }
    prism_asc_compose_compensated_v1<<<(n+255u)/256u,256,0,s>>>(d,phase,f,pe,work,owners,leaf,fault);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int prism_asc_compose_force_compensated_v1(
    uint64_t ptr,uint32_t n,uint32_t phase,const float* x,float* f,
    double* pe,double* work,void* stream,const BiprismCoordinateOwnerFfi* owners,
    uint32_t leaf,uint32_t* fault) {
    return prism_asc_compose_force_precision(ptr,n,phase,x,f,pe,work,stream,owners,leaf,fault,ASC_LEGACY_WHOLE_SYSTEM);
}
extern "C" int prism_asc_compose_force_selected_compensated_v2(
    uint64_t ptr,uint32_t n,uint32_t phase,const float* x,float* f,
    double* pe,double* work,void* stream,const BiprismCoordinateOwnerFfi* owners,
    uint32_t leaf,uint32_t* fault) {
    return prism_asc_compose_force_precision(ptr,n,phase,x,f,pe,work,stream,owners,leaf,fault,ASC_SELECTED_STRUCTURAL_SUPPORT);
}

extern "C" int prism_asc_seal_step_selected_v2(uint64_t ptr, void* stream) {
    if (!ptr || !stream) return static_cast<int>(cudaErrorInvalidValue);
    prism_asc_seal_selected_v2<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<PrismAscDeviceV1*>(ptr));
    return static_cast<int>(cudaGetLastError());
}

extern "C" int prism_asc_publish_decision_selected_v2(
    uint64_t ptr, const void* adj, void* stream) {
    if (!ptr || !adj || !stream) return static_cast<int>(cudaErrorInvalidValue);
    prism_asc_publish_selected_v2<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<PrismAscDeviceV1*>(ptr),
        static_cast<const InterferometricAdjudicatorFfi*>(adj));
    return static_cast<int>(cudaGetLastError());
}

}} // namespace prism_nhs::asc
