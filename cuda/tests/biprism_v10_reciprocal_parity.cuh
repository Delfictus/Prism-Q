// Native differential control for the canonical reciprocal-connection cache.
// Include this after the actual v10 header, or use the canonical include below.
// The host entry point launches diagnostic kernels only; the caller owns GPU
// admission, the exact build/source receipt, and the actual KRAS preparation.
//
// Coverage is pointwise FP64 helper equivalence and the original signed-FMA
// deposit. It does not establish a coupled trajectory, cache lifetime safety,
// or a nonadiabatic benchmark. Every actual edge is used; no edge subsampling.
// The fixed-state H control uses actual onsite/edge/ring helpers and a declared
// deterministic row gather, not the scheduling of production atomic assembly.
#pragma once
#include "../../crates/prism-gpu/src/kernels/nhs_biprism_variational_so3g_v10.cuh"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace biprism_reciprocal_parity {

using Hermitian = BiprismElectronicHermitian<double>;
using Spinor = BiprismElectronicSpinor<double>;
using Vector = BiprismElectronicVector<double>;
constexpr uint32_t kLeaves = 8u;
constexpr uint32_t kBroadControls = 4096u;
static_assert(TILE_CORE_VERTICES == 192 && TILE_TOTAL_VERTICES == 256,
              "This diagnostic must retain the actual 192+64 population");

struct VertexRecord {
    uint32_t leaf, local_vertex, atom, status;
    float x[3], reference[3];
    float amber_charge, raw_omega, phi, reserved;
    Spinor psi;
    double charge;
    Hermitian energy_h;
    // Version 2 appends the authoritative coordinate lows. The preceding
    // 128-byte prefix is retained; a high plane alone is not the input state.
    float x_low[3], reference_low[3];
    // Version3 retains the full potential that generated H. phi above remains
    // the explicitly checked compatibility projection in the legacy prefix.
    double full_phi;
};
struct EdgeRecord {
    uint32_t leaf, local_edge, u, v, atom_u, atom_v, status, reserved;
    float live_weight;
    uint32_t reserved_weight;
    Vector kappa;
    BiprismElectronicEdgeProvider<double> provider;
    BiprismElectronicEdgeDiscrete<double> discrete;
};
// family 0: fixed-state actual tile H/spinor/weight, all four dynamic masks.
// family 1: finite control H/unnormalized psi_bar, actual edge kappa.
// family 2: finite control H/unnormalized psi_bar and explicit stress kappa.
struct CaseRecord {
    uint32_t leaf, local_edge, family, variant;
    uint32_t dynamic_mask, actual_dynamic_mask, status, reserved;
    Vector kappa;
    Hermitian hu, hv;
    Spinor u_bar, v_bar;
    double mass_u, mass_v;
    Vector accumulator;
};
struct ForceRecord {
    Vector pair, deposit_u, deposit_v;
    uint32_t status, reserved;
};
struct CacheRecord {
    double zu, zv;
    uint32_t status, reserved;
};
struct GatherRecord {
    Vector force;
    uint32_t status, incident_edges;
};
struct Metadata {
    char magic[8];
    uint32_t version, leaves, vertices_per_leaf, edge_count;
    uint32_t actual_cases, broad_cases, total_cases, vertex_bytes;
    uint32_t edge_bytes, case_bytes, force_bytes, cache_bytes;
    double kappa_foliated;
    uint64_t reserved;
};
// Separate coordinate-force caller differential. Common inputs retain the
// actual LUT class and represented geometry/charge/density. Each tested scalar
// call reevaluates the original provider; no cached or copied gradient formula.
struct CoordinateCase {
    uint32_t leaf, local_edge, family, variant, element_u, element_v, status, reserved;
    Vector xu, xv;
    double charge_u, charge_v;
    BiprismElectronicDensity<double> density;
};
template<class R> struct CoordinateResult {
    R force[3];
    uint32_t status[3];
};
struct CoordinateMetadata {
    char magic[8];
    uint32_t version, actual_cases, broad_cases, total_cases;
    uint32_t input_bytes, float_output_bytes, double_output_bytes, reserved;
};
static_assert(sizeof(VertexRecord) == 160, "raw compensated vertex and potential record ABI");
static_assert(offsetof(VertexRecord,x_low) == 128, "coordinate low suffix ABI");
static_assert(offsetof(VertexRecord,full_phi) == 152, "full potential suffix ABI");
static_assert(sizeof(EdgeRecord) == 272, "raw edge/provider record ABI");
static_assert(sizeof(CaseRecord) == 224, "raw differential input ABI");
static_assert(sizeof(ForceRecord) == 80, "raw differential force ABI");
static_assert(sizeof(CacheRecord) == 24, "raw global-cache ABI");
static_assert(sizeof(GatherRecord) == 32, "raw production gather output ABI");
static_assert(sizeof(Metadata) == 72, "raw metadata ABI");
static_assert(sizeof(CoordinateCase) == 128, "coordinate caller input ABI");
static_assert(sizeof(CoordinateResult<float>) == 24, "FP32 coordinate output ABI");
static_assert(sizeof(CoordinateResult<double>) == 40, "FP64 coordinate output ABI");
static_assert(sizeof(CoordinateMetadata) == 40, "coordinate metadata ABI");
static_assert(std::is_trivially_copyable<CaseRecord>::value, "raw input POD");

__device__ __forceinline__ bool finite_vector(Vector x) {
    return isfinite(x.x) && isfinite(x.y) && isfinite(x.z);
}
__device__ __forceinline__ bool finite_spinor(Spinor x) {
    return isfinite(x.s0.re) && isfinite(x.s0.im)
        && isfinite(x.s1.re) && isfinite(x.s1.im);
}
__device__ __forceinline__ bool finite_h(Hermitian x) {
    return isfinite(x.h00) && isfinite(x.h11)
        && isfinite(x.h01_re) && isfinite(x.h01_im);
}

__device__ __forceinline__ BiprismCoordinateExpansion3 vertex_position(const VertexRecord& r) {
    return {{r.x[0],r.x_low[0]},{r.x[1],r.x_low[1]},{r.x[2],r.x_low[2]}};
}
__device__ __forceinline__ Vector vertex_value(const VertexRecord& r) {
    // Independent record routing, rather than the production shared accessor.
    return {__dadd_rn(double(r.x[0]),double(r.x_low[0])),
            __dadd_rn(double(r.x[1]),double(r.x_low[1])),
            __dadd_rn(double(r.x[2]),double(r.x_low[2]))};
}

__global__ void prepare_vertices(const FoliatedStreamPtrTableFfi* streams,
    const ContactShellSimplicialStateFfi* tiles, const float* amber,
    uint32_t atom_count, VertexRecord* vertices) {
    const uint32_t leaf = blockIdx.x, u = threadIdx.x;
    if (leaf >= kLeaves || u >= TILE_TOTAL_VERTICES) return;
    VertexRecord r{};
    r.leaf = leaf; r.local_vertex = u;
    const auto& tile = tiles[leaf];
    r.atom = UNPACK_GLOBAL_ATOM_ID(tile.global_atom_idx[u]);
    if (r.atom >= atom_count || !streams->d_coords_xyz[leaf] || !streams->d_coordinate_owners) {
        vertices[leaf * TILE_TOTAL_VERTICES + u] = r;
        return;
    }
    const auto& owner=streams->d_coordinate_owners[leaf];
    if(!bp_coord_owner_valid(&owner,leaf,atom_count,owner.generation,BP_COORD_ATOMIC_BOHR,
        BP_COORD_ALL_INITIALIZED,streams->d_coords_xyz[leaf])) {
        vertices[leaf*TILE_TOTAL_VERTICES+u]=r;return;
    }
    const auto x=bp_coord_load(owner,BiprismCoordinatePlane::Current,r.atom);
    const auto ref=bp_coord_load(owner,BiprismCoordinatePlane::MeshReference,r.atom);
    r.x[0]=x.x.hi;r.x[1]=x.y.hi;r.x[2]=x.z.hi;
    r.x_low[0]=x.x.lo;r.x_low[1]=x.y.lo;r.x_low[2]=x.z.lo;
    r.reference[0]=ref.x.hi;r.reference[1]=ref.y.hi;r.reference[2]=ref.z.hi;
    r.reference_low[0]=ref.x.lo;r.reference_low[1]=ref.y.lo;r.reference_low[2]=ref.z.lo;
    r.amber_charge = amber[r.atom]; r.raw_omega = tile.star0_vol[u];
    r.phi = tile.phi_dec[u];
    if(!streams->d_potential_owners){vertices[leaf*TILE_TOTAL_VERTICES+u]=r;return;}
    const auto& potential=streams->d_potential_owners[leaf];
    if(!bp_potential_owner_valid(&potential,leaf,tiles+leaf,&owner)
        ||!bp_potential_projection_value_valid(potential,u,r.phi)){
        vertices[leaf*TILE_TOTAL_VERTICES+u]=r;return;
    }
    r.full_phi=bp_potential_input_phi(potential)[u];
    // Fixed halos have the same fixed electronic state used by v10.
    if(!streams->d_spinor_owners){vertices[leaf*TILE_TOTAL_VERTICES+u]=r;return;}
    r.psi=u<TILE_CORE_VERTICES?streams->d_spinor_owners[leaf].accepted[u]:Spinor{{1.,0.},{0.,0.}};
    r.charge = u < TILE_CORE_VERTICES
        ? bp_ed_endpoint_data(r.psi, r.psi, double(r.amber_charge)).q_bar
        : double(r.amber_charge);
    r.status = finite_spinor(r.psi) && isfinite(r.charge)
        && isfinite(r.raw_omega) && r.raw_omega > 0.f && isfinite(r.phi) && isfinite(r.full_phi)
        && bp_coord_finite(x) && bp_coord_finite(ref)
        && ref.x.hi==tile.ref_pos_q[u].x && ref.y.hi==tile.ref_pos_q[u].y
        && ref.z.hi==tile.ref_pos_q[u].z;
    vertices[leaf * TILE_TOTAL_VERTICES + u] = r;
}

__global__ void prepare_edges(const FoliatedStreamPtrTableFfi* streams,const ContactShellSimplicialStateFfi* tiles,
    const uint32_t* offsets, const VertexRecord* vertices, EdgeRecord* edges,
    BiprismV10NumericalScratchFfi* numerical) {
    // The actual live-weight accessor consumes the dedicated full coordinate
    // plane. Bind and populate it before the first accessor read.
    extern __shared__ __align__(128) unsigned char scratch_bytes[];
    auto* scratch = reinterpret_cast<VariationalClusterDsmemV10*>(scratch_bytes);
    const uint32_t leaf = blockIdx.x, tid = threadIdx.x;
    if (leaf >= kLeaves) return;
    const auto* tile = tiles + leaf;
    const auto* vertex = vertices + leaf * TILE_TOTAL_VERTICES;
    scratch->x_mid[tid] = {vertex[tid].x_low[0], vertex[tid].x_low[1], vertex[tid].x_low[2]};
    numerical[leaf].nuclear_position_live[tid]=vertex_position(vertex[tid]);
    if(tid==0u){scratch->invocation.tile=const_cast<ContactShellSimplicialStateFfi*>(tile);
        scratch->invocation.numerical=numerical+leaf;
        scratch->invocation.derivative_coordinate_phase=0u;
        scratch->invocation.collective.streams=streams;scratch->invocation.collective.leaf=leaf;}
    const uint32_t count = offsets[leaf + 1u] - offsets[leaf];
    #pragma unroll 1
    for (uint32_t e = tid; e < count; e += 256u) {
        scratch->edge_u8[e][0] = tile->edge_u8[e][0];
        scratch->edge_u8[e][1] = tile->edge_u8[e][1];
    }
    __syncthreads();
    const BiprismLiveDecWeightsV10 live_weight{scratch, tile};
    #pragma unroll 1
    for (uint32_t e = tid; e < count; e += 256u) {
        EdgeRecord r{};
        r.leaf = leaf; r.local_edge = e;
        r.u = tile->edge_u8[e][0]; r.v = tile->edge_u8[e][1];
        const auto& u = vertex[r.u]; const auto& v = vertex[r.v];
        r.atom_u = u.atom; r.atom_v = v.atom;
        bool ok = u.status == 1u && v.status == 1u && r.u != r.v;
        if (ok) {
            r.live_weight = live_weight(e);
            ok = isfinite(r.live_weight) && r.live_weight > 0.f
                && bp_ed_evaluate_edge(
                    bp_v10_element(UNPACK_ELEM_CLASS(tile->global_atom_idx[r.u])),
                    bp_v10_element(UNPACK_ELEM_CLASS(tile->global_atom_idx[r.v])),
                    vertex_value(u),vertex_value(v),
                    u.charge, v.charge, r.provider);
        }
        if (ok) {
            r.kappa = bp_v10_kappa(r.provider, r.live_weight);
            r.discrete = bp_ed_edge_discrete(r.provider.E0, r.provider.E1,
                r.provider.V01, r.provider.S, r.provider.gamma,
                bp_ed_endpoint_data(u.psi, u.psi, 0.).rho_bar);
            ok = finite_vector(r.kappa) && finite_h(r.discrete.owner_h)
                && isfinite(r.discrete.charge_derivative_u)
                && isfinite(r.discrete.charge_derivative_v);
        }
        r.status = ok ? 1u : 0u;
        edges[offsets[leaf] + e] = r;
    }
}

__global__ void prepare_energy_h(const ContactShellSimplicialStateFfi* tiles,
    const uint32_t* offsets, const EdgeRecord* edges, double foliation,
    VertexRecord* vertices) {
    const uint32_t leaf = blockIdx.x, u = threadIdx.x;
    if (leaf >= kLeaves || u >= TILE_TOTAL_VERTICES) return;
    // Do not read another thread's whole mutable record: neighbor inputs here
    // are immutable psi/raw_omega fields, while this kernel writes only H/status.
    auto* target = vertices + leaf * TILE_TOTAL_VERTICES + u;
    if (u >= TILE_CORE_VERTICES) { target->energy_h = {}; return; }
    const auto element = bp_v10_element(UNPACK_ELEM_CLASS(tiles[leaf].global_atom_idx[u]));
    Hermitian h = bp_ed_onsite_dec_hamiltonian(double(target->raw_omega),
        element.h_s_valence, element.h_p_excited, target->full_phi);
    const auto& prev = vertices[((leaf + 7u) & 7u) * TILE_TOTAL_VERTICES + u];
    const auto& next = vertices[((leaf + 1u) & 7u) * TILE_TOTAL_VERTICES + u];
    const double sp = bp_v10_population(prev.psi), si = bp_v10_population(target->psi),
                 sn = bp_v10_population(next.psi);
    const auto ring = biprism_foliated::endpoint_ledger(foliation,
        double(prev.raw_omega), double(target->raw_omega), double(next.raw_omega),
        sp, sp, si, si, sn, sn);
    h.h00 += ring.numerator; h.h11 -= ring.numerator;
    bool ok = target->status == 1u;
    #pragma unroll 1
    for (uint32_t e = offsets[leaf]; e < offsets[leaf + 1u]; ++e) {
        const auto& p = edges[e];
        if (p.u != u && p.v != u) continue;
        ok = ok && p.status == 1u;
        if (p.u == u) {
            h.h00 += p.discrete.owner_h.h00;
            h.h11 += p.discrete.owner_h.h11 + double(.15f) * p.discrete.charge_derivative_u;
            h.h01_re += p.discrete.owner_h.h01_re;
        }
        if (p.v == u) h.h11 += double(.15f) * p.discrete.charge_derivative_v;
    }
    target->energy_h = h;
    target->status = ok && finite_h(h) ? 1u : 0u;
}

__device__ __forceinline__ uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30u)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27u)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31u);
}
__device__ __forceinline__ double finite_scalar(uint64_t key, int exponent) {
    const uint64_t b = mix(key);
    const uint64_t bits = 0x3ff0000000000000ULL | (b & 0x000fffffffffffffULL);
    const double magnitude = __longlong_as_double(static_cast<long long>(bits));
    return ldexp((b >> 63u) ? -magnitude : magnitude, exponent);
}
__device__ __forceinline__ Spinor represented_control_spinor(uint64_t key, int exponent) {
    // Each endpoint component is first represented in FP32, exactly as in the
    // candidate bank; the midpoint formed below is deliberately not normalized.
    return {{double(float(finite_scalar(key + 0u, exponent))),
             double(float(finite_scalar(key + 1u, exponent)))},
            {double(float(finite_scalar(key + 2u, exponent))),
             double(float(finite_scalar(key + 3u, exponent)))}};
}
__device__ __forceinline__ Spinor negate_spinor(Spinor p) {
    return {{-p.s0.re, -p.s0.im}, {-p.s1.re, -p.s1.im}};
}
__device__ __forceinline__ Hermitian control_h(uint64_t key, int exponent) {
    return {finite_scalar(key, exponent), finite_scalar(key + 1u, exponent),
            finite_scalar(key + 2u, exponent), finite_scalar(key + 3u, exponent)};
}

__global__ void prepare_cases(const VertexRecord* vertices, const EdgeRecord* edges,
    uint32_t edge_count, uint32_t total_cases, CaseRecord* cases) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_cases) return;
    const uint32_t actual_cases = 4u * edge_count;
    const bool actual = i < actual_cases;
    const uint32_t control = actual ? i / 4u : (i - actual_cases) / 4u;
    const uint32_t edge_id = actual ? control : uint32_t(mix(control) % edge_count);
    const auto& edge = edges[edge_id];
    const auto& u = vertices[edge.leaf * TILE_TOTAL_VERTICES + edge.u];
    const auto& v = vertices[edge.leaf * TILE_TOTAL_VERTICES + edge.v];
    CaseRecord r{};
    r.leaf = edge.leaf; r.local_edge = edge.local_edge; r.variant = control;
    r.dynamic_mask = i & 3u;
    r.actual_dynamic_mask = (edge.u < TILE_CORE_VERTICES ? 1u : 0u)
                         | (edge.v < TILE_CORE_VERTICES ? 2u : 0u);
    r.kappa = edge.kappa; r.hu = u.energy_h; r.hv = v.energy_h;
    r.u_bar = bp_ed_endpoint_data(u.psi, u.psi, 0.).psi_bar;
    r.v_bar = bp_ed_endpoint_data(v.psi, v.psi, 0.).psi_bar;
    r.mass_u = double(fmaxf(u.raw_omega, 1e-4f));
    r.mass_v = double(fmaxf(v.raw_omega, 1e-4f));
    r.accumulator = {finite_scalar(i + 41u, -3), finite_scalar(i + 42u, 7),
                     finite_scalar(i + 43u, -27)};
    if (!actual) {
        const uint64_t seed = 0xa13e8f27d63bc941ULL + uint64_t(control) * 131u;
        const int he = int(mix(seed) % 401u) - 200;
        const int pe = int(mix(seed + 1u) % 201u) - 100;
        r.family = (control & 1u) ? 2u : 1u;
        r.hu = control_h(seed + 2u, he);
        r.hv = control_h(seed + 6u, -he);
        const auto u0 = represented_control_spinor(seed + 10u, pe);
        auto u1 = represented_control_spinor(seed + 14u, pe);
        const auto v0 = represented_control_spinor(seed + 18u, -pe);
        auto v1 = represented_control_spinor(seed + 22u, -pe);
        if ((control & 15u) == 2u) u1 = negate_spinor(u0);
        if ((control & 15u) == 3u) v1 = negate_spinor(v0);
        r.u_bar = bp_ed_endpoint_data(u0, u1, 0.).psi_bar;
        r.v_bar = bp_ed_endpoint_data(v0, v1, 0.).psi_bar;
        r.mass_u = fmax(double(1e-4f), fabs(finite_scalar(seed + 26u, int(control % 73u) - 13)));
        r.mass_v = fmax(double(1e-4f), fabs(finite_scalar(seed + 27u, int((control * 17u) % 73u) - 13)));
        switch (control & 15u) {
            case 0u: // Exact endpoint cancellation with nontrivial H/psi.
                r.hv = r.hu; r.v_bar = r.u_bar; r.mass_v = r.mass_u; break;
            case 1u: // Adjacent-H cancellation, unequal last-bit numerator.
                r.hv = r.hu; r.hv.h11 = nextafter(r.hu.h11, CUDART_INF);
                r.v_bar = r.u_bar; r.mass_v = r.mass_u; break;
            case 4u: r.hu.h01_im = 0.; r.hv.h01_im = -0.; break;
            case 5u: r.hu = {r.hu.h00, r.hu.h00, 0., 0.}; break;
            case 6u: r.u_bar.s0 = {0., -0.}; break;
            case 7u: r.v_bar.s1 = {-0., 0.}; break;
            case 8u: r.mass_u = double(1e-4f); r.mass_v = nextafter(r.mass_u, CUDART_INF); break;
            case 9u: r.mass_u = 1.; r.mass_v = 1.; break;
            default: break;
        }
        if (r.family == 2u) {
            const int ke = int(mix(seed + 30u) % 401u) - 200;
            r.kappa = {finite_scalar(seed + 31u, ke), finite_scalar(seed + 32u, -ke),
                       finite_scalar(seed + 33u, ke / 2)};
            if ((control & 15u) == 11u) r.kappa = {0., -0., 0.};
            if ((control & 15u) == 13u) r.kappa.x = ldexp(1., -1000);
            if ((control & 15u) == 15u) r.kappa.z = -ldexp(1., -1000);
        }
    }
    const bool finite = finite_h(r.hu) && finite_h(r.hv)
        && finite_spinor(r.u_bar) && finite_spinor(r.v_bar)
        && finite_vector(r.kappa) && finite_vector(r.accumulator)
        && isfinite(r.mass_u) && r.mass_u > 0.
        && isfinite(r.mass_v) && r.mass_v > 0.;
    r.status = edge.status == 1u && u.status == 1u && v.status == 1u && finite ? 1u : 0u;
    cases[i] = r;
}

__device__ __forceinline__ Vector signed_deposit(double sign, Vector pair, Vector seed) {
    return {__fma_rn(sign, pair.x, seed.x), __fma_rn(sign, pair.y, seed.y),
            __fma_rn(sign, pair.z, seed.z)};
}

__global__ void original_force(const CaseRecord* cases, uint32_t count, ForceRecord* out) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto r = cases[i]; ForceRecord result{};
    if (r.status == 1u) {
        // This is the unchanged canonical helper, not a copied oracle formula.
        result.pair = bp_ed_reciprocal_connection_force<double>(r.kappa,
            r.hu, r.u_bar, r.mass_u, (r.dynamic_mask & 1u) != 0u,
            r.hv, r.v_bar, r.mass_v, (r.dynamic_mask & 2u) != 0u);
        result.deposit_u = signed_deposit(1., result.pair, r.accumulator);
        result.deposit_v = signed_deposit(-1., result.pair, r.accumulator);
        result.status = finite_vector(result.pair) && finite_vector(result.deposit_u)
                     && finite_vector(result.deposit_v) ? 1u : 0u;
    }
    out[i] = result;
}

__global__ void publish_pairing_cache(const CaseRecord* cases, uint32_t count,
    CacheRecord* cache) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto r = cases[i];
    // Volatile global writes plus a distinct consumer launch ensure the exact
    // FP64 z values actually cross the same storage/rounding boundary as C2.
    volatile CacheRecord* p = cache + i;
    double zu = 0., zv = 0.;
    if (r.status == 1u) {
        if (r.dynamic_mask & 1u)
            zu = bp_ed_connection_energy_pairing(r.hu, r.u_bar) / r.mass_u;
        if (r.dynamic_mask & 2u)
            zv = bp_ed_connection_energy_pairing(r.hv, r.v_bar) / r.mass_v;
    }
    p->zu = zu; p->zv = zv;
    p->status = r.status == 1u && isfinite(zu) && isfinite(zv) ? 1u : 0u;
    p->reserved = 0u;
}

__global__ void cached_force(const CaseRecord* cases, const CacheRecord* cache,
    uint32_t count, ForceRecord* out) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto r = cases[i]; ForceRecord result{};
    const volatile CacheRecord* p = cache + i;
    const uint32_t status = p->status;
    const double zu = p->zu, zv = p->zv;
    if (r.status == 1u && status == 1u) {
        // This is the actual current main's contraction, not a test clone.
        result.pair = bp_v10_reciprocal_from_pairings(r.kappa, zu, zv);
        result.deposit_u = signed_deposit(1., result.pair, r.accumulator);
        result.deposit_v = signed_deposit(-1., result.pair, r.accumulator);
        result.status = finite_vector(result.pair) && finite_vector(result.deposit_u)
                     && finite_vector(result.deposit_v) ? 1u : 0u;
    }
    out[i] = result;
}

__global__ void prepare_coordinate_cases(const ContactShellSimplicialStateFfi* tiles,
    const VertexRecord* vertices, const EdgeRecord* edges, uint32_t edge_count,
    uint32_t count, CoordinateCase* cases) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const uint32_t control = i < edge_count ? i : i - edge_count;
    const auto& edge = edges[i < edge_count ? i : uint32_t(mix(control) % edge_count)];
    const auto& u = vertices[edge.leaf * TILE_TOTAL_VERTICES + edge.u];
    const auto& v = vertices[edge.leaf * TILE_TOTAL_VERTICES + edge.v];
    CoordinateCase r{};
    r.leaf = edge.leaf; r.local_edge = edge.local_edge; r.variant = control;
    r.element_u = UNPACK_ELEM_CLASS(tiles[edge.leaf].global_atom_idx[edge.u]);
    r.element_v = UNPACK_ELEM_CLASS(tiles[edge.leaf].global_atom_idx[edge.v]);
    r.xu = vertex_value(u);r.xv = vertex_value(v);
    r.charge_u = u.charge; r.charge_v = v.charge;
    r.density = bp_ed_endpoint_data(u.psi, u.psi, 0.).rho_bar;
    r.status = edge.status == 1u && u.status == 1u && v.status == 1u;
    if (i >= edge_count) {
        r.family = 1u;
        r.element_u = control & 7u; r.element_v = (control >> 3u) & 7u;
        // Every LUT pair sees all 64 variants. The last four intentionally
        // exercise provider refusal, rather than silently skipping failures.
        const uint32_t kind = control >> 6u;
        const uint64_t seed = 0x523a9be817c6d0f4ULL + control * 113u;
        const double distances[16] = {0., 1e-5, 1e-3, .01, .03, .06, .1, .3,
            1., 3., 10., 30., 100., 1e4, 0x1p60, 0x1p100};
        const double distance = distances[kind & 15u];
        r.xu = {0., 0., 0.};
        r.xv = {distance, .25 * distance, -.125 * distance};
        r.charge_u = double(float(finite_scalar(seed, int(kind % 7u) - 3)));
        r.charge_v = double(float(finite_scalar(seed + 1u, int(kind % 7u) - 3)));
        const auto p0 = represented_control_spinor(seed + 2u, int(kind % 17u) - 8);
        const auto p1 = represented_control_spinor(seed + 6u, int(kind % 17u) - 8);
        r.density = bp_ed_endpoint_data(p0, p1, 0.).rho_bar;
        if (kind == 60u) r.charge_u = CUDART_INF;
        if (kind == 61u) r.charge_v = CUDART_NAN;
        if (kind == 62u) r.xv.x = 1e300;
        if (kind == 63u) r.xv.y = CUDART_NAN;
        r.status = 1u;
    }
    cases[i] = r;
}

template<class R> __device__ __forceinline__ bool coordinate_provider(
    const CoordinateCase* r, BiprismElectronicEdgeProvider<R>& p) {
    const auto u = bp_v10_element(r->element_u), v = bp_v10_element(r->element_v);
    const BiprismElectronicElementParams<R> a = {R(u.zeta_slater), R(u.h_s_valence),
        R(u.h_p_excited), R(u.hubbard_eta)};
    const BiprismElectronicElementParams<R> b = {R(v.zeta_slater), R(v.h_s_valence),
        R(v.h_p_excited), R(v.hubbard_eta)};
    return r->status == 1u && bp_ed_evaluate_edge(a, b,
        BiprismElectronicVector<R>{R(r->xu.x), R(r->xu.y), R(r->xu.z)},
        BiprismElectronicVector<R>{R(r->xv.x), R(r->xv.y), R(r->xv.z)},
        R(r->charge_u), R(r->charge_v), p);
}
template<class R> __device__ __forceinline__ BiprismElectronicDensity<R>
coordinate_density(const CoordinateCase* r) {
    const volatile BiprismElectronicDensity<double>* p = &r->density;
    return {R(p->r00), R(p->r11), R(p->r01_re), R(p->r01_im)};
}
template<class R> __global__ void original_coordinate_force(
    const CoordinateCase* cases, uint32_t count, CoordinateResult<R>* output) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    CoordinateResult<R> out{{R(0x1.234p5), R(-0x1.234p5), R(0x1.567p8)}, {0u, 0u, 0u}};
    BiprismElectronicEdgeProvider<R> p;
    if (coordinate_provider(cases + i, p)) {
        asm volatile("" ::: "memory");
        const auto f = bp_ed_edge_coordinate_force(p.E0, p.E1, p.V01, p.S,
            coordinate_density<R>(cases + i), p.grad_E0_u, p.grad_E1_u,
            p.grad_V01_u, p.grad_S_u);
        out.force[0] = f.x; out.force[1] = f.y; out.force[2] = f.z;
        out.status[0] = out.status[1] = out.status[2] = 1u + uint32_t(R(1)-p.S*p.S <= R(1e-3f));
    }
    output[i] = out;
}
template<unsigned Component, class R> __global__ void actual_coordinate_component(
    const CoordinateCase* cases, uint32_t count, CoordinateResult<R>* output) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const R sentinels[3] = {R(0x1.234p5), R(-0x1.234p5), R(0x1.567p8)};
    R value = sentinels[Component]; uint32_t status = 0u;
    // Fresh full-provider evaluation in every component invocation. Only the
    // actual main adapter selects the output; this test has no force formula.
    BiprismElectronicEdgeProvider<R> p;
    if (coordinate_provider(cases + i, p)) {
        asm volatile("" ::: "memory");
        value = bp_v10_original_force_component<Component>(p, coordinate_density<R>(cases + i));
        status = 1u + uint32_t(R(1)-p.S*p.S <= R(1e-3f));
    }
    volatile CoordinateResult<R>* out = output + i;
    out->force[Component] = value; out->status[Component] = status;
}

__global__ void actual_shared_coordinate_callers(
    const FoliatedStreamPtrTableFfi* streams,const ContactShellSimplicialStateFfi* tiles, const float* amber,
    const uint32_t* offsets, const VertexRecord* vertices,
    BiprismV10NumericalScratchFfi* numerical, uint32_t refused_control,
    CoordinateResult<double>* original, CoordinateResult<double>* component_out) {
    extern __shared__ __align__(128) unsigned char bytes[];
    auto* s = reinterpret_cast<VariationalClusterDsmemV10*>(bytes);
    const uint32_t leaf = blockIdx.x, tid = threadIdx.x;
    if (leaf >= kLeaves) return;
    const auto* vertex = vertices + leaf * TILE_TOTAL_VERTICES;
    s->x_mid[tid] = {vertex[tid].x_low[0], vertex[tid].x_low[1], vertex[tid].x_low[2]};
    numerical[leaf].nuclear_position_live[tid]=vertex_position(vertex[tid]);
    if (refused_control) {s->x_mid[tid].x = CUDART_NAN_F;numerical[leaf].nuclear_position_live[tid].x.lo=CUDART_NAN_F;}
    if (tid < TILE_CORE_VERTICES) {
        s->q_live[tid] = vertex[tid].charge;

        streams->d_spinor_owners[leaf].substep_start[tid] = vertex[tid].psi;
        streams->d_spinor_owners[leaf].candidate[tid] = vertex[tid].psi;
    }
    if (tid == 0u) {
        s->invocation.tile = const_cast<ContactShellSimplicialStateFfi*>(tiles + leaf);
        s->invocation.amber = amber;
        s->invocation.numerical = numerical + leaf;
        s->invocation.derivative_coordinate_phase=0u;
        s->invocation.collective.streams=streams;s->invocation.collective.leaf=leaf;
    }
    __syncthreads();
    const uint32_t count = offsets[leaf + 1u] - offsets[leaf];
    #pragma unroll 1
    for (uint32_t e = tid; e < count; e += 256u) {
        const uint32_t u = tiles[leaf].edge_u8[e][0], v = tiles[leaf].edge_u8[e][1];
        CoordinateResult<double> a{{0x1.234p5, -0x1.234p5, 0x1.567p8}, {0u, 0u, 0u}}, b = a;
        BiprismElectronicEdgeProvider<double> p;
        Vector xu=vertex_value(vertex[u]),xv=vertex_value(vertex[v]);
        if(refused_control){xu.x=CUDART_NAN;xv.x=CUDART_NAN;}
        if (bp_ed_evaluate_edge(bp_v10_element(UNPACK_ELEM_CLASS(tiles[leaf].global_atom_idx[u])),
            bp_v10_element(UNPACK_ELEM_CLASS(tiles[leaf].global_atom_idx[v])),
            xu,xv,
            bp_v10_charge(s, tiles + leaf, amber, u), bp_v10_charge(s, tiles + leaf, amber, v), p)) {
            asm volatile("" ::: "memory");
            const auto f = bp_ed_edge_coordinate_force(p.E0, p.E1, p.V01, p.S,
                bp_v10_density_mean(s, u, true), p.grad_E0_u, p.grad_E1_u, p.grad_V01_u, p.grad_S_u);
            a.force[0] = f.x; a.force[1] = f.y; a.force[2] = f.z;
            a.status[0] = a.status[1] = a.status[2] = 1u;
        }
        // Actual production store wrapper, including its refused-provider
        // no-write contract. Read the same real own-lane global slots C2 uses.
        numerical[leaf].edge_force_x[tid] = b.force[0];
        numerical[leaf].edge_force_y[tid] = b.force[1];
        numerical[leaf].edge_force_z[tid] = b.force[2];
        asm volatile("" ::: "memory");
        b.status[0] = bp_v10_store_original_force_component<0u>(s, u, v);
        asm volatile("" ::: "memory");
        b.status[1] = bp_v10_store_original_force_component<1u>(s, u, v);
        asm volatile("" ::: "memory");
        b.status[2] = bp_v10_store_original_force_component<2u>(s, u, v);
        asm volatile("" ::: "memory");
        const volatile auto* stored = numerical + leaf;
        b.force[0] = stored->edge_force_x[tid];
        b.force[1] = stored->edge_force_y[tid];
        b.force[2] = stored->edge_force_z[tid];
        original[offsets[leaf] + e] = a; component_out[offsets[leaf] + e] = b;
    }
}

__global__ void publish_gather_pairings(const VertexRecord* vertices,
    BiprismV10NumericalScratchFfi* numerical, double* retained_pairings) {
    const uint32_t leaf=blockIdx.x,tid=threadIdx.x;
    if(leaf>=kLeaves||tid>=TILE_CORE_VERTICES)return;
    const auto& v=vertices[leaf*TILE_TOTAL_VERTICES+tid];
    const auto pm=bp_ed_endpoint_data(v.psi,v.psi,0.).psi_bar;
    const double z=bp_ed_connection_energy_pairing(v.energy_h,pm)/double(fmaxf(v.raw_omega,1e-4f));
    volatile double* p=&numerical[leaf].reciprocal_connection_z[tid];
    *p=z;retained_pairings[leaf*TILE_CORE_VERTICES+tid]=z;
}

__global__ void original_incident_gather(const uint32_t* offsets,
    const VertexRecord* vertices,const EdgeRecord* edges,GatherRecord* output) {
    const uint32_t leaf=blockIdx.x,tid=threadIdx.x;
    if(leaf>=kLeaves||tid>=TILE_CORE_VERTICES)return;
    GatherRecord r{};r.status=1u;
    for(uint32_t e=offsets[leaf];e<offsets[leaf+1u];++e){
        const auto& edge=edges[e];
        if(edge.u!=tid&&edge.v!=tid)continue;
        const auto& u=vertices[leaf*TILE_TOTAL_VERTICES+edge.u];
        const auto& v=vertices[leaf*TILE_TOTAL_VERTICES+edge.v];
        const auto pair=bp_ed_reciprocal_connection_force<double>(edge.kappa,
            u.energy_h,bp_ed_endpoint_data(u.psi,u.psi,0.).psi_bar,
            double(fmaxf(u.raw_omega,1e-4f)),edge.u<TILE_CORE_VERTICES,
            v.energy_h,bp_ed_endpoint_data(v.psi,v.psi,0.).psi_bar,
            double(fmaxf(v.raw_omega,1e-4f)),edge.v<TILE_CORE_VERTICES);
        r.force=signed_deposit(edge.u==tid?1.:-1.,pair,r.force);
        r.status=uint32_t(r.status&&edge.status==1u&&finite_vector(pair));
        ++r.incident_edges;
    }
    r.status=uint32_t(r.status&&finite_vector(r.force));
    output[leaf*TILE_CORE_VERTICES+tid]=r;
}

__global__ void actual_incident_gather(const FoliatedStreamPtrTableFfi* streams,const ContactShellSimplicialStateFfi* tiles,
    const float* amber,const uint32_t* offsets,const VertexRecord* vertices,
    const EdgeRecord* edges,BiprismV10NumericalScratchFfi* numerical,
    FoliatedCouplingLeafFfi* banks,GatherRecord* output) {
    extern __shared__ __align__(128) unsigned char bytes[];
    auto* s=reinterpret_cast<VariationalClusterDsmemV10*>(bytes);
    const uint32_t leaf=blockIdx.x,tid=threadIdx.x;
    if(leaf>=kLeaves)return;
    const auto& vertex=vertices[leaf*TILE_TOTAL_VERTICES+tid];
    s->x_mid[tid]={vertex.x_low[0],vertex.x_low[1],vertex.x_low[2]};
    numerical[leaf].nuclear_position_live[tid]=vertex_position(vertex);
    if(tid<TILE_CORE_VERTICES){
        s->q_live[tid]=vertex.charge;
        for(uint32_t axis=0;axis<3u;++axis)numerical[leaf].reciprocal_force[axis][tid]=0.;
    }
    const uint32_t count=offsets[leaf+1u]-offsets[leaf];
    for(uint32_t e=tid;e<count;e+=256u)banks[leaf].live_weights[e]=edges[offsets[leaf]+e].live_weight;
    if(tid==0u){
        s->invocation.tile=const_cast<ContactShellSimplicialStateFfi*>(tiles+leaf);
        s->invocation.amber=amber;s->invocation.numerical=numerical+leaf;
        s->invocation.derivative_coordinate_phase=0u;
        s->invocation.bank=banks+leaf;
        s->invocation.collective.streams=streams;s->invocation.collective.leaf=leaf;
    }
    __syncthreads();
    if(tid<TILE_CORE_VERTICES){
        GatherRecord r{};r.status=1u;
        #pragma unroll 1
        for(uint32_t e=0u;e<count;++e){
            const uint32_t u=tiles[leaf].edge_u8[e][0],v=tiles[leaf].edge_u8[e][1];
            if(u!=tid&&v!=tid)continue;
            const bool ok=bp_v10_accumulate_reciprocal_edge(s,e,u,v);
            r.status=uint32_t(ok&&r.status);++r.incident_edges;
        }
        asm volatile("" ::: "memory");
        const volatile auto* stored=numerical+leaf;
        r.force={stored->reciprocal_force[0][tid],stored->reciprocal_force[1][tid],stored->reciprocal_force[2][tid]};
        r.status=uint32_t(r.status&&finite_vector(r.force));
        output[leaf*TILE_CORE_VERTICES+tid]=r;
    }
}

__global__ void actual_berry_callers(const FoliatedStreamPtrTableFfi* streams,
    const ContactShellSimplicialStateFfi* tiles,const float* amber,
    const uint32_t* offsets,const VertexRecord* vertices,uint32_t variant,float h,
    BiprismV10NumericalScratchFfi* numerical,float3* retained_velocities,
    CoordinateResult<double>* original,CoordinateResult<double>* from_state) {
    extern __shared__ __align__(128) unsigned char bytes[];
    auto* s=reinterpret_cast<VariationalClusterDsmemV10*>(bytes);
    const uint32_t leaf=blockIdx.x,tid=threadIdx.x;
    if(leaf>=kLeaves)return;
    const auto* vertex=vertices+leaf*TILE_TOTAL_VERTICES;
    s->x_mid[tid]={vertex[tid].x_low[0],vertex[tid].x_low[1],vertex[tid].x_low[2]};
    numerical[leaf].nuclear_position_live[tid]=vertex_position(vertex[tid]);
    float3 velocity{};
    if(variant==0u){
        const float* p=streams->d_vels_xyz[leaf]+3u*vertex[tid].atom;
        velocity={p[0],p[1],p[2]};
    }else{
        const uint64_t seed=0x24af78e31b906dc5ULL+uint64_t(leaf*256u+tid)*7u;
        const int exponent=variant==1u?-12:-5;
        velocity={float(finite_scalar(seed,exponent)),float(finite_scalar(seed+1u,exponent)),
            float(finite_scalar(seed+2u,exponent))};
    }
    if(tid>=TILE_CORE_VERTICES)velocity={0.f,0.f,0.f}; // production fixed-halo boundary
    s->v_mid[tid]=velocity;retained_velocities[leaf*TILE_TOTAL_VERTICES+tid]=velocity;
    if(tid<TILE_CORE_VERTICES)bp_ns_store3d(numerical[leaf].nuclear_trial_velocity,tid,{double(velocity.x),double(velocity.y),double(velocity.z)});
    if(tid<TILE_CORE_VERTICES)s->q_live[tid]=vertex[tid].charge;
    if(tid==0u){
        s->invocation.tile=const_cast<ContactShellSimplicialStateFfi*>(tiles+leaf);
        s->invocation.amber=amber;bp_v10_publish_h(s,double(h));s->invocation.numerical=numerical+leaf;
        s->invocation.derivative_coordinate_phase=0u;
        s->invocation.collective.streams=streams;s->invocation.collective.leaf=leaf;
    }
    __syncthreads();
    const uint32_t count=offsets[leaf+1u]-offsets[leaf];
    #pragma unroll 1
    for(uint32_t edge=tid;edge<count;edge+=256u){
        const uint32_t u=tiles[leaf].edge_u8[edge][0],v=tiles[leaf].edge_u8[edge][1];
        CoordinateResult<double> a{},b{};
        {
            const auto vu=bp_v10_reload_vec3(&s->v_mid[u]),vv=bp_v10_reload_vec3(&s->v_mid[v]);
            const auto e=bp_coord_difference(vertex_position(vertex[v]),vertex_position(vertex[u]));
            const BiprismVec64 vr={double(vv.x)-double(vu.x),double(vv.y)-double(vu.y),double(vv.z)-double(vu.z)};
            const BiprismVec64 de={double(h)*vr.x,double(h)*vr.y,double(h)*vr.z};
            const uint32_t eu=UNPACK_ELEM_CLASS(tiles[leaf].global_atom_idx[u]),ev=UNPACK_ELEM_CLASS(tiles[leaf].global_atom_idx[v]);
            // New precision profile: compare the actual state wrapper with
            // the direct compensated-node calls and the original sum order.
            // This is not an equivalence claim to legacy FP32 geometry.
            const auto fa=bp_v10_berry_node_compensated(eu,ev,vertex[u].charge,vertex[v].charge,e,de,vr,h,-0.3872983346f,0.27777778f);
            const auto fb=bp_v10_berry_node_compensated(eu,ev,vertex[u].charge,vertex[v].charge,e,de,vr,h,0.f,0.44444444f);
            const auto fd=bp_v10_berry_node_compensated(eu,ev,vertex[u].charge,vertex[v].charge,e,de,vr,h,0.3872983346f,0.27777778f);
            const BiprismVec64 f={fa.x+fb.x+fd.x,fa.y+fb.y+fd.y,fa.z+fb.z+fd.z};
            a.force[0]=f.x;a.force[1]=f.y;a.force[2]=f.z;
            a.status[0]=a.status[1]=a.status[2]=1u;
        }
        {
            const auto f=bp_v10_berry_chord_from_state(s,u,v);
            b.force[0]=f.x;b.force[1]=f.y;b.force[2]=f.z;
            b.status[0]=b.status[1]=b.status[2]=1u;
        }
        original[offsets[leaf]+edge]=a;from_state[offsets[leaf]+edge]=b;
    }
}

inline void cuda_check(cudaError_t e, const char* operation) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("reciprocal parity ") + operation + ": " + cudaGetErrorString(e));
}
template<class T> struct Buffer {
    T* p = nullptr;
    size_t size;
    explicit Buffer(size_t n) : size(n) {
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&p), std::max(size_t(1), n) * sizeof(T)), "allocation");
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() {
        if (p) {
            const auto e = cudaFree(p);
            if (e != cudaSuccess) std::fprintf(stderr, "reciprocal parity release: %s\n", cudaGetErrorString(e));
        }
    }
    std::vector<T> read() const {
        std::vector<T> result(size);
        cuda_check(cudaMemcpy(result.data(), p, size * sizeof(T), cudaMemcpyDeviceToHost), "readback");
        return result;
    }
};
template<class T> inline void retain(const std::string& path, const std::vector<T>& values) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(values.data()), std::streamsize(values.size() * sizeof(T)));
    f.close();
    if (!f) throw std::runtime_error("reciprocal parity raw evidence write: " + path);
}
inline uint64_t bits(double value) {
    uint64_t b; std::memcpy(&b, &value, sizeof(b)); return b;
}
inline uint64_t ordered_bits(double value) {
    const uint64_t b = bits(value);
    return b & 0x8000000000000000ULL ? ~b : b | 0x8000000000000000ULL;
}
inline uint64_t ulp_distance(double a, double b) {
    const auto x = ordered_bits(a), y = ordered_bits(b);
    return x > y ? x - y : y - x;
}
inline double component(Vector v, unsigned axis) {
    return axis == 0u ? v.x : (axis == 1u ? v.y : v.z);
}
template<class R> inline uint64_t coordinate_bits(R value) {
    uint64_t out = 0u; std::memcpy(&out, &value, sizeof(R)); return out;
}
template<class R> inline uint64_t coordinate_ulp(R a, R b) {
    constexpr uint64_t sign = uint64_t(1) << (sizeof(R)*8u-1u);
    constexpr uint64_t mask = sizeof(R)==8u ? UINT64_MAX : uint64_t(UINT32_MAX);
    const uint64_t x = coordinate_bits(a), y = coordinate_bits(b);
    const uint64_t ox = x & sign ? (~x & mask) : (x | sign);
    const uint64_t oy = y & sign ? (~y & mask) : (y | sign);
    return ox > oy ? ox-oy : oy-ox;
}
template<class R> inline bool compare_coordinate_results(const char* label,
    const std::vector<CoordinateResult<R>>& reference,
    const std::vector<CoordinateResult<R>>& actual, uint32_t actual_cases,
    bool require_broad_coverage,bool require_nonzero=true) {
    if (reference.size()!=actual.size() || actual_cases>reference.size())
        throw std::runtime_error("coordinate parity output extent");
    uint64_t mismatches=0u,status_mismatches=0u,invalid_actual=0u,nonfinite=0u;
    uint64_t refused=0u,clamped=0u,unclamped=0u,nonzero=0u,max_ulp=0u;
    bool printed=false;
    for(size_t i=0u;i<reference.size();++i)for(unsigned c=0u;c<3u;++c){
        const auto& a=reference[i];const auto& b=actual[i];
        status_mismatches+=a.status[c]!=b.status[c];
        invalid_actual+=i<actual_cases && (a.status[c]==0u || b.status[c]==0u);
        if(a.status[c]==0u)++refused;
        else if(a.status[c]==1u)++unclamped;
        else if(a.status[c]==2u)++clamped;
        else ++status_mismatches;
        nonfinite+=!std::isfinite(a.force[c]) || !std::isfinite(b.force[c]);
        nonzero+=a.status[c]!=0u && a.force[c]!=R(0);
        const bool different=coordinate_bits(a.force[c])!=coordinate_bits(b.force[c]);
        mismatches+=different;
        if(different&&std::isfinite(a.force[c])&&std::isfinite(b.force[c]))
            max_ulp=std::max(max_ulp,coordinate_ulp(a.force[c],b.force[c]));
        if(!printed&&(different||a.status[c]!=b.status[c])){
            std::printf("coordinate_parity_first_mismatch path=%s precision=%zu case=%zu axis=%u status=%u/%u original=%a actual=%a original_bits=%016llx actual_bits=%016llx\n",
                label,sizeof(R)*8u,i,c,a.status[c],b.status[c],double(a.force[c]),double(b.force[c]),
                (unsigned long long)coordinate_bits(a.force[c]),(unsigned long long)coordinate_bits(b.force[c]));
            printed=true;
        }
    }
    const bool coverage=(!require_nonzero||nonzero>0u)&&unclamped>0u&&(!require_broad_coverage||(clamped>0u&&refused>0u));
    const bool pass=coverage&&mismatches==0u&&status_mismatches==0u&&invalid_actual==0u&&nonfinite==0u;
    std::printf("coordinate_parity path=%s precision=%zu actual_edges=%u total_cases=%zu components=%zu value_mismatches=%llu status_mismatches=%llu max_ulp=%llu invalid_actual=%llu nonfinite=%llu clamped_components=%llu unclamped_components=%llu refused_components=%llu nonzero_components=%llu coverage=%u pass=%u\n",
        label,sizeof(R)*8u,actual_cases,reference.size(),3u*reference.size(),
        (unsigned long long)mismatches,(unsigned long long)status_mismatches,(unsigned long long)max_ulp,
        (unsigned long long)invalid_actual,(unsigned long long)nonfinite,(unsigned long long)clamped,
        (unsigned long long)unclamped,(unsigned long long)refused,(unsigned long long)nonzero,
        unsigned(coverage),unsigned(pass));
    return pass;
}

inline bool run_coordinate_callers(const FoliatedStreamPtrTableFfi* streams,const ContactShellSimplicialStateFfi* tiles,
    const float* amber,const uint32_t* offsets,const VertexRecord* vertices,
    const EdgeRecord* edges,uint32_t edge_count,const std::string& prefix) {
    const uint32_t count=edge_count+kBroadControls,blocks=(count+255u)/256u;
    Buffer<CoordinateCase> cases(count);
    Buffer<BiprismV10NumericalScratchFfi> scratch(kLeaves);
    Buffer<CoordinateResult<float>> of(count),cf(count);
    Buffer<CoordinateResult<double>> od(count),cd(count),shared_o(edge_count),shared_c(edge_count);
    prepare_coordinate_cases<<<blocks,256>>>(tiles,vertices,edges,edge_count,count,cases.p);
    cuda_check(cudaGetLastError(),"coordinate case preparation");
    original_coordinate_force<float><<<blocks,256>>>(cases.p,count,of.p);
    cuda_check(cudaGetLastError(),"original FP32 coordinate force");
    original_coordinate_force<double><<<blocks,256>>>(cases.p,count,od.p);
    cuda_check(cudaGetLastError(),"original FP64 coordinate force");
    actual_coordinate_component<0u,float><<<blocks,256>>>(cases.p,count,cf.p);
    cuda_check(cudaGetLastError(),"actual FP32 coordinate x caller");
    actual_coordinate_component<1u,float><<<blocks,256>>>(cases.p,count,cf.p);
    cuda_check(cudaGetLastError(),"actual FP32 coordinate y caller");
    actual_coordinate_component<2u,float><<<blocks,256>>>(cases.p,count,cf.p);
    cuda_check(cudaGetLastError(),"actual FP32 coordinate z caller");
    actual_coordinate_component<0u,double><<<blocks,256>>>(cases.p,count,cd.p);
    cuda_check(cudaGetLastError(),"actual FP64 coordinate x caller");
    actual_coordinate_component<1u,double><<<blocks,256>>>(cases.p,count,cd.p);
    cuda_check(cudaGetLastError(),"actual FP64 coordinate y caller");
    actual_coordinate_component<2u,double><<<blocks,256>>>(cases.p,count,cd.p);
    cuda_check(cudaGetLastError(),"actual FP64 coordinate z caller");
    actual_shared_coordinate_callers<<<kLeaves,256,sizeof(VariationalClusterDsmemV10)>>>(
        streams,tiles,amber,offsets,vertices,scratch.p,0u,shared_o.p,shared_c.p);
    cuda_check(cudaGetLastError(),"actual FP64 production global-store callers");
    cuda_check(cudaDeviceSynchronize(),"coordinate caller parity completion");
    const auto inputs=cases.read();const auto a=of.read(),b=cf.read();
    const auto d=od.read(),e=cd.read(),so=shared_o.read(),sc=shared_c.read();
    const std::string stem=prefix+".coordinate_parity";
    const CoordinateMetadata meta{{'B','P','C','O','O','R','2','\0'},2u,edge_count,kBroadControls,count,
        sizeof(CoordinateCase),sizeof(CoordinateResult<float>),sizeof(CoordinateResult<double>),0u};
    retain(stem+".meta.bin",std::vector<CoordinateMetadata>{meta});
    retain(stem+".cases.bin",inputs);retain(stem+".f32.original.bin",a);retain(stem+".f32.component.bin",b);
    retain(stem+".f64.original.bin",d);retain(stem+".f64.component.bin",e);
    retain(stem+".shared.original.bin",so);retain(stem+".shared.component.bin",sc);
    bool coverage=true;uint64_t pair_counts[8][8]={};
    for(uint32_t i=0u;i<count;++i){const auto& c=inputs[i];
        coverage=coverage&&c.status==1u&&c.element_u<8u&&c.element_v<8u;
        if(i<edge_count)coverage=coverage&&c.family==0u;
        else if(c.element_u<8u&&c.element_v<8u){++pair_counts[c.element_u][c.element_v];coverage=coverage&&c.family==1u;}
    }
    for(unsigned u=0;u<8u;++u)for(unsigned v=0;v<8u;++v)coverage=coverage&&pair_counts[u][v]==64u;
    const bool pf=compare_coordinate_results("actual_adapter",a,b,edge_count,true);
    const bool pd=compare_coordinate_results("actual_adapter",d,e,edge_count,true);
    const bool ps=compare_coordinate_results("actual_shared_store_wrapper",so,sc,edge_count,false);
    // The shared control uses the same actual edge/charge/density inputs as
    // the generic FP64 controls. Check that routing independently as well.
    bool routing=true;
    for(uint32_t i=0;i<edge_count;++i)for(unsigned c=0;c<3u;++c)
        routing=routing&&d[i].status[c]!=0u&&so[i].status[c]!=0u
            &&coordinate_bits(d[i].force[c])==coordinate_bits(so[i].force[c]);
    // Separate deliberate nonfinite-geometry control: every provider must
    // refuse and the actual store wrapper must preserve all three sentinels.
    actual_shared_coordinate_callers<<<kLeaves,256,sizeof(VariationalClusterDsmemV10)>>>(
        streams,tiles,amber,offsets,vertices,scratch.p,1u,shared_o.p,shared_c.p);
    cuda_check(cudaGetLastError(),"actual store refused-provider control");
    cuda_check(cudaDeviceSynchronize(),"store refusal control completion");
    const auto ro=shared_o.read(),rc=shared_c.read();
    retain(stem+".shared_refusal.geometry_mode.u32le",std::vector<uint32_t>{1u});
    retain(stem+".shared_refusal.original.bin",ro);retain(stem+".shared_refusal.store.bin",rc);
    const double sentinel[3]={0x1.234p5,-0x1.234p5,0x1.567p8};
    uint64_t refusal_errors=0u;
    for(uint32_t i=0u;i<edge_count;++i)for(unsigned c=0u;c<3u;++c)
        refusal_errors+=ro[i].status[c]!=0u||rc[i].status[c]!=0u
            ||bits(ro[i].force[c])!=bits(sentinel[c])||bits(rc[i].force[c])!=bits(sentinel[c]);
    std::printf("coordinate_parity_population all_actual_edges=%u all64_LUT_pairs=%u generic_shared_input_routing=%u refused_store_components=%llu refusal_errors=%llu pass=%u\n",
        edge_count,unsigned(coverage),unsigned(routing),(unsigned long long)(3ull*edge_count),
        (unsigned long long)refusal_errors,unsigned(coverage&&routing&&pf&&pd&&ps&&refusal_errors==0u));
    return coverage&&routing&&pf&&pd&&ps&&refusal_errors==0u;
}

inline bool run_incident_gather(const FoliatedStreamPtrTableFfi* streams,const ContactShellSimplicialStateFfi* tiles,
    const float* amber,const uint32_t* offsets,const VertexRecord* vertices,
    const EdgeRecord* edges,const std::string& prefix) {
    constexpr uint32_t count=kLeaves*TILE_CORE_VERTICES;
    Buffer<BiprismV10NumericalScratchFfi> scratch(kLeaves);
    Buffer<FoliatedCouplingLeafFfi> banks(kLeaves);
    Buffer<double> pairings(count);
    Buffer<GatherRecord> original(count),actual(count);
    original_incident_gather<<<kLeaves,256>>>(offsets,vertices,edges,original.p);
    cuda_check(cudaGetLastError(),"original full-helper incident gather");
    publish_gather_pairings<<<kLeaves,256>>>(vertices,scratch.p,pairings.p);
    cuda_check(cudaGetLastError(),"production-gather global pairing publication");
    actual_incident_gather<<<kLeaves,256,sizeof(VariationalClusterDsmemV10)>>>(
        streams,tiles,amber,offsets,vertices,edges,scratch.p,banks.p,actual.p);
    cuda_check(cudaGetLastError(),"actual noinline incident-edge updater");
    cuda_check(cudaDeviceSynchronize(),"actual incident gather parity completion");
    const auto z=pairings.read();const auto a=original.read(),b=actual.read();
    const std::string stem=prefix+".reciprocal_parity.incident_gather";
    retain(stem+".pairings.f64le",z);retain(stem+".original.bin",a);retain(stem+".actual.bin",b);
    uint64_t mismatch=0u,invalid=0u,nonzero=0u,incidences=0u,max_ulp=0u;
    bool printed=false;
    for(uint32_t i=0u;i<count;++i){
        invalid+=!std::isfinite(z[i])||a[i].status!=1u||b[i].status!=1u
            ||a[i].incident_edges==0u||a[i].incident_edges!=b[i].incident_edges;
        incidences+=a[i].incident_edges;
        for(unsigned c=0u;c<3u;++c){
            const double x=component(a[i].force,c),y=component(b[i].force,c);
            invalid+=!std::isfinite(x)||!std::isfinite(y);nonzero+=x!=0.;
            if(bits(x)!=bits(y)){
                ++mismatch;max_ulp=std::max(max_ulp,ulp_distance(x,y));
                if(!printed){std::printf("incident_gather_first_mismatch leaf=%u vertex=%u axis=%u original=%a actual=%a\n",
                    i/TILE_CORE_VERTICES,i%TILE_CORE_VERTICES,c,x,y);printed=true;}
            }
        }
    }
    const bool pass=invalid==0u&&mismatch==0u&&nonzero>0u;
    std::printf("incident_gather_parity actual_core_vertices=%u components=%u ordered_incidence_visits=%llu invalid=%llu value_mismatches=%llu max_ulp=%llu nonzero=%llu global_cache_boundary=separate_kernel pass=%u\n",
        count,3u*count,(unsigned long long)incidences,(unsigned long long)invalid,
        (unsigned long long)mismatch,(unsigned long long)max_ulp,(unsigned long long)nonzero,unsigned(pass));
    return pass;
}

// Independent CUDA libdevice reference lives only in this diagnostic entry;
// its slow-path stack is never linked into the production v10 call graph.
__global__ void magnetic_phase_control(const double* x,double* reference,double* actual,uint32_t count){
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    reference[i]=x[i]+sin(x[i]);
    actual[i]=biprism_magnetic_f64::phase_plus_sine(x[i]);
}
inline bool run_magnetic_phase_control(const std::string& prefix){
    std::vector<double> input{0.,-0.};
    for(int e=-1074;e<=1023;++e){
        const double x=std::ldexp(1.+double(uint32_t(e+1074)*7919u%65536u)/65536.,e);
        input.push_back(x);input.push_back(-x);
    }
    for(int i=-2048;i<=2048;++i){
        const double x=double(i)*0x1.921fb54442d18p+0;
        input.push_back(std::nextafter(x,-INFINITY));input.push_back(x);
        input.push_back(std::nextafter(x,INFINITY));
    }
    Buffer<double> x(input.size()),reference(input.size()),actual(input.size());
    cuda_check(cudaMemcpy(x.p,input.data(),input.size()*sizeof(double),cudaMemcpyHostToDevice),"phase primitive input");
    magnetic_phase_control<<<(input.size()+255u)/256u,256>>>(x.p,reference.p,actual.p,uint32_t(input.size()));
    cuda_check(cudaGetLastError(),"phase primitive launch");cuda_check(cudaDeviceSynchronize(),"phase primitive completion");
    const auto r=reference.read(),a=actual.read();uint64_t worst=0;bool pass=true;
    for(size_t i=0;i<input.size();++i){
        const auto ulp=ulp_distance(r[i],a[i]);worst=std::max(worst,ulp);
        pass=pass&&std::isfinite(a[i])&&ulp<=3u;
    }
    retain(prefix+".magnetic_phase.input.f64le",input);retain(prefix+".magnetic_phase.libdevice.f64le",r);
    retain(prefix+".magnetic_phase.actual.f64le",a);
    std::printf("magnetic_phase_arithmetic cases=%zu max_ULP=%llu gate_ULP=3 independent_libdevice=1 trajectory_acceptance=0 pass=%u\n",input.size(),(unsigned long long)worst,unsigned(pass));
    return pass;
}

inline bool run_berry_callers(const FoliatedStreamPtrTableFfi* streams,
    const ContactShellSimplicialStateFfi* tiles,const float* amber,const uint32_t* offsets,
    const VertexRecord* vertices,uint32_t edge_count,const std::string& prefix,
    float h=float(0.0005*41341.37333517015)*.25f) {
    // Berry and electronic components share bytes in C2. The per-lane stride
    // must match the consumed double slot; a packed float[256] is a race with
    // another lane's still-live electronic component, even if pure Berry
    // numerical parity happens to pass.
    static_assert(sizeof(((BiprismV10NumericalScratchFfi*)nullptr)->edge_berry_x[0])==sizeof(double),
        "Berry cache must retain the eight-byte electronic lane ownership");
    static_assert(sizeof(((BiprismV10NumericalScratchFfi*)nullptr)->edge_berry_y[0])==sizeof(double),
        "Berry y lane ownership");
    static_assert(sizeof(((BiprismV10NumericalScratchFfi*)nullptr)->edge_berry_z[0])==sizeof(double),
        "Berry z lane ownership");
    Buffer<BiprismV10NumericalScratchFfi> scratch(kLeaves);
    Buffer<float3> velocities(kLeaves*TILE_TOTAL_VERTICES);
    Buffer<CoordinateResult<double>> original(edge_count),state(edge_count);
    if(!std::isfinite(h)||!(h>0.f))throw std::runtime_error("Berry caller substep domain");
    retain(prefix+".berry_parity.h.f32le",std::vector<float>{h});
    bool pass=run_magnetic_phase_control(prefix);
    for(uint32_t variant=0u;variant<3u;++variant){
        actual_berry_callers<<<kLeaves,256,sizeof(VariationalClusterDsmemV10)>>>(streams,tiles,amber,
            offsets,vertices,variant,h,scratch.p,velocities.p,original.p,state.p);
        cuda_check(cudaGetLastError(),"actual Berry state-wrapper launch");
        cuda_check(cudaDeviceSynchronize(),"Berry caller parity completion");
        const auto v=velocities.read();const auto a=original.read(),b=state.read();
        const std::string stem=prefix+".berry_parity.variant"+std::to_string(variant);
        retain(stem+".velocities.bin",v);retain(stem+".original.bin",a);retain(stem+".state.bin",b);
        const std::string label="actual_Berry_state_wrapper_variant"+std::to_string(variant);
        const bool result=compare_coordinate_results(label.c_str(),a,b,edge_count,false,variant!=0u);
        pass=pass&&result;
    }
    std::printf("berry_parity field_nodes_and_sum=FP64 phase_sine=FP64_MPFR_bounded geometry=compensated_FP64 direct_node_vs_state_wrapper all_actual_edges=%u variants=3 actual_velocity_variant=0 declared_velocity_controls=2 h=%a all_components_required_zero_ULP=1 pass=%u\n",
        edge_count,double(h),unsigned(pass));
    return pass;
}

// Call once on test-owned, actually prepared AU tiles, after Phase2 and before
// the positive main launch. This function never modifies scientific input.
// All outputs are retained even if a finite/bitwise/coverage gate fails.
inline bool run(const FoliatedStreamPtrTableFfi* device_streams,
    const ContactShellSimplicialStateFfi* device_tiles, const float* device_amber,
    uint32_t atom_count, double kappa_foliated, const std::string& output_prefix,
    float h=float(0.0005*41341.37333517015)*.25f) {
    if (!device_streams || !device_tiles || !device_amber || atom_count == 0u
        || !std::isfinite(kappa_foliated)||!std::isfinite(h)||!(h>0.f))
        throw std::runtime_error("reciprocal parity input domain");
    std::vector<ContactShellSimplicialStateFfi> tiles(kLeaves);
    FoliatedStreamPtrTableFfi streams{};
    cuda_check(cudaMemcpy(tiles.data(), device_tiles, sizeof(tiles[0]) * tiles.size(), cudaMemcpyDeviceToHost), "tile identity");
    cuda_check(cudaMemcpy(&streams, device_streams, sizeof(streams), cudaMemcpyDeviceToHost), "stream identity");
    // Parity owns its work planes; never overwrite the production candidate or
    // accepted banks while exercising the actual consumers.
    Buffer<BiprismSpinorOwnerFfi> private_spinors(kLeaves);Buffer<FoliatedStreamPtrTableFfi> private_streams(1);
    if(!streams.d_spinor_owners)throw std::runtime_error("parity spinor owner absent");
    cuda_check(cudaMemcpy(private_spinors.p,streams.d_spinor_owners,kLeaves*sizeof(BiprismSpinorOwnerFfi),cudaMemcpyDeviceToDevice),"private spinor copy");
    streams.d_spinor_owners=private_spinors.p;
    cuda_check(cudaMemcpy(private_streams.p,&streams,sizeof(streams),cudaMemcpyHostToDevice),"private table");
    device_streams=private_streams.p;
    if(!streams.d_coordinate_owners||(reinterpret_cast<uintptr_t>(streams.d_coordinate_owners)&15u)!=0u
        ||streams.reserved_coordinate_alignment!=0u)
        throw std::runtime_error("reciprocal parity requires coordinate authority");
    std::vector<BiprismCoordinateOwnerFfi> coordinate_owners(kLeaves);
    cuda_check(cudaMemcpy(coordinate_owners.data(),streams.d_coordinate_owners,
        kLeaves*sizeof(BiprismCoordinateOwnerFfi),cudaMemcpyDeviceToHost),"coordinate owner identity");
    std::vector<uint32_t> offsets(kLeaves + 1u, 0u);
    uint64_t actual_masks[4] = {};
    for (uint32_t leaf = 0; leaf < kLeaves; ++leaf) {
        const auto& t = tiles[leaf];
        if(!bp_coord_owner_valid(&coordinate_owners[leaf],leaf,atom_count,coordinate_owners[0].generation,
            BP_COORD_ATOMIC_BOHR,BP_COORD_ALL_INITIALIZED,streams.d_coords_xyz[leaf]))
            throw std::runtime_error("reciprocal parity coordinate identity/generation/frame");
        if (!streams.d_coords_xyz[leaf] || !streams.d_vels_xyz[leaf]
            || t.num_active_edges == 0u || t.num_active_edges > TILE_MAX_EDGES)
            throw std::runtime_error("reciprocal parity requires all eight nonempty actual tiles");
        offsets[leaf + 1u] = offsets[leaf] + t.num_active_edges;
        for (uint32_t u = 0; u < TILE_TOTAL_VERTICES; ++u) {
            const auto a = UNPACK_GLOBAL_ATOM_ID(t.global_atom_idx[u]);
            if (a >= atom_count) throw std::runtime_error("reciprocal parity atom extent");
            if (u < TILE_CORE_VERTICES && a != UNPACK_GLOBAL_ATOM_ID(tiles[0].global_atom_idx[u]))
                throw std::runtime_error("reciprocal parity ring requires actual matched core identities");
        }
        for (uint32_t e = 0; e < t.num_active_edges; ++e) {
            const uint32_t u = t.edge_u8[e][0], v = t.edge_u8[e][1];
            if (u == v) throw std::runtime_error("reciprocal parity degenerate edge identity");
            ++actual_masks[(u < TILE_CORE_VERTICES ? 1u : 0u) | (v < TILE_CORE_VERTICES ? 2u : 0u)];
        }
    }
    const uint32_t edge_count = offsets.back(), actual_cases = 4u * edge_count;
    const uint32_t broad_cases = 4u * kBroadControls, count = actual_cases + broad_cases;
    Buffer<uint32_t> d_offsets(offsets.size());
    cuda_check(cudaMemcpy(d_offsets.p, offsets.data(), offsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice), "offset upload");
    Buffer<VertexRecord> d_vertices(kLeaves * TILE_TOTAL_VERTICES);
    Buffer<EdgeRecord> d_edges(edge_count);
    Buffer<BiprismV10NumericalScratchFfi> d_preparation_numerical(kLeaves);
    Buffer<CaseRecord> d_cases(count);
    Buffer<CacheRecord> d_cache(count);
    Buffer<ForceRecord> d_original(count), d_cached(count);
    const unsigned blocks = (count + 255u) / 256u;
    prepare_vertices<<<kLeaves, 256>>>(device_streams, device_tiles, device_amber, atom_count, d_vertices.p);
    cuda_check(cudaGetLastError(), "actual vertex preparation launch");
    prepare_edges<<<kLeaves, 256, sizeof(VariationalClusterDsmemV10)>>>(device_streams,device_tiles, d_offsets.p, d_vertices.p, d_edges.p,d_preparation_numerical.p);
    cuda_check(cudaGetLastError(), "actual provider/live-weight preparation launch");
    prepare_energy_h<<<kLeaves, 256>>>(device_tiles, d_offsets.p, d_edges.p, kappa_foliated, d_vertices.p);
    cuda_check(cudaGetLastError(), "fixed-state energy-H launch");
    prepare_cases<<<blocks, 256>>>(d_vertices.p, d_edges.p, edge_count, count, d_cases.p);
    cuda_check(cudaGetLastError(), "case preparation launch");
    original_force<<<blocks, 256>>>(d_cases.p, count, d_original.p);
    cuda_check(cudaGetLastError(), "original canonical helper launch");
    publish_pairing_cache<<<blocks, 256>>>(d_cases.p, count, d_cache.p);
    cuda_check(cudaGetLastError(), "global pairing publication launch");
    cached_force<<<blocks, 256>>>(d_cases.p, d_cache.p, count, d_cached.p);
    cuda_check(cudaGetLastError(), "actual cached contraction launch");
    cuda_check(cudaDeviceSynchronize(), "native parity completion");
    const auto vertices = d_vertices.read(); const auto edges = d_edges.read();
    const auto cases = d_cases.read(); const auto cache = d_cache.read();
    const auto original = d_original.read(); const auto cached = d_cached.read();
    Metadata meta{{'B','P','R','E','C','P','3','\0'}, 3u, kLeaves, TILE_TOTAL_VERTICES,
        edge_count, actual_cases, broad_cases, count, sizeof(VertexRecord), sizeof(EdgeRecord),
        sizeof(CaseRecord), sizeof(ForceRecord), sizeof(CacheRecord), kappa_foliated, 0u};
    const std::string stem = output_prefix + ".reciprocal_parity";
    retain(stem + ".meta.bin", std::vector<Metadata>{meta});
    retain(stem + ".coordinate_owners.96b",coordinate_owners);
    retain(stem + ".offsets.bin", offsets);
    retain(stem + ".vertices.bin", vertices); retain(stem + ".edges.bin", edges);
    retain(stem + ".cases.bin", cases); retain(stem + ".cache.bin", cache);
    retain(stem + ".original.bin", original); retain(stem + ".cached.bin", cached);

    uint64_t invalid_vertices = 0, invalid_edges = 0, invalid_cases = 0, invalid_outputs = 0;
    uint64_t pair_mismatches = 0, deposit_mismatches = 0, max_ulp = 0, sign_zero_mismatches = 0;
    uint64_t masks[3][4] = {}, unnormalized = 0, complex_h = 0, nonzero_pairs = 0;
    uint64_t first_bad = std::numeric_limits<uint64_t>::max();
    for (const auto& v : vertices) invalid_vertices += v.status != 1u;
    for (const auto& e : edges) invalid_edges += e.status != 1u;
    for (uint32_t i = 0; i < count; ++i) {
        const auto& c = cases[i];
        invalid_cases += c.status != 1u;
        if (c.family < 3u && c.dynamic_mask < 4u) ++masks[c.family][c.dynamic_mask];
        else ++invalid_cases;
        const double norm = bp_ed_norm2(c.u_bar.s0) + bp_ed_norm2(c.u_bar.s1);
        unnormalized += std::isfinite(norm) && norm != 1.;
        complex_h += c.hu.h01_im != 0. || c.hv.h01_im != 0.;
        const bool valid = original[i].status == 1u && cached[i].status == 1u && cache[i].status == 1u;
        invalid_outputs += !valid;
        const Vector a[] = {original[i].pair, original[i].deposit_u, original[i].deposit_v};
        const Vector b[] = {cached[i].pair, cached[i].deposit_u, cached[i].deposit_v};
        for (unsigned field = 0; field < 3u; ++field) for (unsigned axis = 0; axis < 3u; ++axis) {
            const double x = component(a[field], axis), y = component(b[field], axis);
            if (!std::isfinite(x) || !std::isfinite(y)) { ++invalid_outputs; continue; }
            if (field == 0u) nonzero_pairs += x != 0.;
            if (bits(x) != bits(y)) {
                if (field == 0u) ++pair_mismatches; else ++deposit_mismatches;
                sign_zero_mismatches += x == 0. && y == 0.;
                max_ulp = std::max(max_ulp, ulp_distance(x, y));
                if (first_bad == std::numeric_limits<uint64_t>::max()) {
                    first_bad = i;
                    std::printf("reciprocal_parity_first_mismatch case=%u family=%u leaf=%u edge=%u mask=%u field=%u axis=%u original=%a cached=%a original_bits=%016llx cached_bits=%016llx\n",
                        i, c.family, c.leaf, c.local_edge, c.dynamic_mask, field, axis, x, y,
                        static_cast<unsigned long long>(bits(x)), static_cast<unsigned long long>(bits(y)));
                }
            }
        }
    }
    // Make sure an empty or single-flag run cannot pass a zero-mismatch gate.
    bool coverage = actual_masks[3] > 0u && (actual_masks[1] + actual_masks[2]) > 0u
        && unnormalized > 0u && complex_h > 0u && nonzero_pairs > 0u;
    for (unsigned mask = 0; mask < 4u; ++mask)
        coverage = coverage && masks[0][mask] == edge_count
            && masks[1][mask] == kBroadControls / 2u
            && masks[2][mask] == kBroadControls / 2u;
    const bool comparator_controls = bits(0.) != bits(-0.)
        && ulp_distance(1., std::nextafter(1., 2.)) == 1u;
    const bool pass = coverage && comparator_controls && invalid_vertices == 0u
        && invalid_edges == 0u && invalid_cases == 0u && invalid_outputs == 0u
        && pair_mismatches == 0u && deposit_mismatches == 0u;
    std::printf("reciprocal_parity scope=FP64_helper_and_signed_FMA global_cache_boundary=separate_kernel source_actual_edges=%u actual_cases=%u broad_cases=%u components=%llu pair_mismatches=%llu deposit_mismatches=%llu max_ulp=%llu signed_zero_mismatches=%llu invalid_vertices=%llu invalid_edges=%llu invalid_cases=%llu invalid_outputs=%llu coverage=%u pass=%u\n",
        edge_count, actual_cases, broad_cases, static_cast<unsigned long long>(uint64_t(count) * 9u),
        static_cast<unsigned long long>(pair_mismatches), static_cast<unsigned long long>(deposit_mismatches),
        static_cast<unsigned long long>(max_ulp), static_cast<unsigned long long>(sign_zero_mismatches),
        static_cast<unsigned long long>(invalid_vertices), static_cast<unsigned long long>(invalid_edges),
        static_cast<unsigned long long>(invalid_cases), static_cast<unsigned long long>(invalid_outputs),
        unsigned(coverage && comparator_controls), unsigned(pass));
    const bool coordinate_pass=run_coordinate_callers(device_streams,device_tiles,device_amber,d_offsets.p,
        d_vertices.p,d_edges.p,edge_count,output_prefix);
    const bool gather_pass=run_incident_gather(device_streams,device_tiles,device_amber,d_offsets.p,
        d_vertices.p,d_edges.p,output_prefix);
    const bool berry_pass=run_berry_callers(device_streams,device_tiles,device_amber,d_offsets.p,
        d_vertices.p,edge_count,output_prefix,h);
    return pass&&coordinate_pass&&gather_pass&&berry_pass;
}

} // namespace biprism_reciprocal_parity
