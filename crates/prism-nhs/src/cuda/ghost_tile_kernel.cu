// ═══════════════════════════════════════════════════════════════════════════
// PRISM-4D / M1.2.19.B (Amendment 3.14 — v9D' Sector-Lock) — GhostTileFrame
// capture kernel (impl)
// ═══════════════════════════════════════════════════════════════════════════
//
// Single-block, n_clusters-thread kernel.  Each thread `i` < n_clusters
// inspects its cluster's adjudication_code (read via byte-offset arithmetic
// from the FFI struct's offset 52 — current_divergence is at 48,
// adjudication_code at 52, mirroring InterferometricAdjudicatorFfi); if
// the code is Construct (1) or Violation (2), the thread atomicAdds the
// shared counter, bounds-checks against max_records, and writes its
// 4096-byte GhostTileFrame record into the pinned-host, device-mapped
// ring at offset `COUNTER_SECTOR (4096) + slot * 4096`.
//
// The kernel copies all four ContactShellTile plane spectra
// (geo/caus/therm/chem) into the header's power_spectrum[24] stripes
// 0/6/12/18 — the M1.2.25 plane-fan-out, lines 206-216 in the v1 kernel and
// 582-592 in the v2 kernel.  No plane is zero-filled.
//
// thermo_flux[0..2] and causal_lead_residue are populated, not sentinel-
// filled.  thermo_flux carries therm_power_spectrum[0] and
// caus_power_spectrum[0] (lines 222-226 / 594-598); causal_lead_residue
// carries d_kcc_lead[i] when that F2-pool buffer is present and falls back to
// the 0xFFFFFFFF "not resolved at GPU time" sentinel only when it is absent
// (lines 231-233 / 600).  The trailing 1280-byte ContactShellTile body has
// been retired from the on-disk format.
//
// Emission gate: the v1 kernel emits when adj_code >= 1, or unconditionally
// when firehose_enable != 0 (line 128).  The production v2 kernel emits on
// adj_code >= 1 OR the observation pass (kl >= observation_threshold_kl) OR
// the diagnostic firehose with physically-active KL (lines 534-538), so an
// emitted record may legitimately carry adjudication_code == 0.
// ═══════════════════════════════════════════════════════════════════════════

#include "ghost_tile_kernel.cuh"
#include "so3_project.cuh"        // ContactShellTile (1280 bytes, align 128)
#include <cstdint>
#include <cstring>                // memcpy
#include <cuda_runtime.h>
#include <math_constants.h>       // CUDART_NAN_F

using prism_nhs::so3_project::ContactShellTile;
using prism_nhs::rich_spike::RichSpike;
using prism_nhs::rich_spike::NativeV2SeedLayout;
using prism_nhs::rich_spike::load_native_v2_seed;
using prism_nhs::rich_spike::valid_native_v2_seed_layout;

// FFI byte-offsets for InterferometricAdjudicatorFfi fields the kernel
// reads.  These match the asserts in adjudicator.cuh.  We don't include
// adjudicator.cuh here because that would drag the SO(3) dependency
// into a TU that only needs the byte offsets.
static constexpr size_t PRISM_ADJ_CURRENT_DIVERGENCE_OFF = 48;
static constexpr size_t PRISM_ADJ_ADJUDICATION_CODE_OFF  = 52;

// Wave 1 / Q1 — cluster → representative-residue lookup table.
// Populated host-side via prism_ghost_set_cluster_repr_residue (below) with
// the MODE of residue_id over the cluster's RichSpike range — a population
// statistic (which residue contributes the most spikes), NOT an
// intensity argmax (bin/nhs_rt_full.rs:20807-20820).  64-entry slab covers
// the SIMT block width.
//
// Unused entries are NOT sentinel-filled: prism_ghost_set_cluster_repr_residue
// copies exactly n entries and never writes u32_MAX, so slots >= n hold zero
// or the prior campaign's value.  The kernel reads only [0, n_clusters), so
// those slots are inert — but a reader must not assume u32_MAX padding here.
// When the slab is not loaded at all, every slot reads the __constant__
// default 0 and the boundary scan below runs against residue 0.  See
// ghost_tile.rs::set_cluster_repr_residue for the full contract.
__constant__ uint32_t d_cluster_to_repr_residue[64];

// **M1.2.20.C-C / T20 — Topology-Driven Chain Boundary LUT.**
// Per operator's Amendment 3.20 "Purge of Hardcoded Boundaries":
// the 7C8R-specific 4613 atom index has been excised from kernel
// source.  d_chain_offsets[k] is the residue-id boundary at which
// chain k STARTS.  d_chain_offsets[0] is conventionally 0 (chain A
// origin).  Unused entries are populated with u32::MAX so the
// linear scan terminates naturally for fewer than 8 chains.
//
// Kernel logic (residue-major):
//   for i in 0..7:
//     if d_chain_offsets[i] <= residue_id < d_chain_offsets[i+1]:
//       chain_id = i;  break;
//
// 8 × u32 = 32 B → fits in a single L1 constant-cache line, single-
// cycle broadcast load to every thread in the warp.  CPU does the
// heavy topology parsing in nhs_rt_full.rs; the GPU only does
// integer comparison (operator §4 — "do not implement a complex
// string-matching parser on the GPU").
__constant__ uint32_t d_chain_offsets[8];

// **Path Ω Phase 2 — Geometry-Emergent Chain Identity LUT.**
// Populated host-side at V2 build via prism_ghost_set_cluster_chain_id
// (below).  Per cluster i ∈ [0..64), holds:
//   0 ⇒ chain A (centroid on negative side of dyad-perpendicular normal)
//   1 ⇒ chain B (centroid on positive side)
//   0xFF ⇒ unset / fall back to legacy d_chain_offsets residue-boundary scan
//
// Computed on the host from cluster centroids and the topology's
// dimer_dyad block: for each cluster, project its centroid onto a
// vector perpendicular to the dyad axis and assign chain by sign.
// This makes chain identity GEOMETRY-EMERGENT — independent of any
// upstream prism-prep / sanitizer that may have collapsed chain
// labels in the PDB → topology pipeline (the v9D 7C8R issue).
__constant__ uint8_t d_cluster_to_chain_id[64];

extern "C"
__global__ void prism_ghost_pipe_stage_kernel(
    uint8_t* __restrict__              ring_base,
    const ContactShellTile* __restrict__ tiles,
    const uint8_t* __restrict__        adj,
    const uint32_t* __restrict__       d_kcc_lead,   // Wave 1 / Q2 — F2-pool [n_clusters]
    uint64_t                           frame_idx,
    uint32_t                           n_clusters,
    uint32_t                           max_records,
    uint32_t                           firehose_enable  // 0 = adj-gated; nonzero = ALWAYS emit
) {
    const uint32_t i = threadIdx.x;
    if (i >= n_clusters) return;

    // adjudication_code is per-frame, not per-cluster on this build —
    // but the design accommodates future per-cluster Z-vector adjudication.
    // For now read the global code; all clusters share the same SWITCH.
    const uint8_t adj_code =
        *reinterpret_cast<const uint8_t*>(adj + PRISM_ADJ_ADJUDICATION_CODE_OFF);
    // Diagnostic firehose mode: when enabled, ALWAYS emit a record per
    // cluster per replay regardless of adj_code (operator post-audit
    // 2026-05-03 — full per-cluster KL trajectory + 4-plane spectrum
    // time series captured even on null-manifest runs).  The emitted
    // record still carries the actual adj_code so downstream tools
    // can distinguish "construct event" from "diagnostic sample".
    if (firehose_enable == 0u && adj_code < 1u) return;  // Prune (0): nothing to record

    // Bounds-check + atomic claim.
    // Wave 1 / P5 — atomicAdd on a host-mapped pinned counter; multiple
    // streams ALL contend on this single u32 word.  Blackwell sm_120
    // resolves the contention on the L2 atomic engine.  __threadfence_system()
    // immediately after publishes the counter increment to the host
    // address space so the Reaper thread observes the slot reservation
    // before the kernel starts writing the 4096-byte record below.
    uint32_t* p_counter = reinterpret_cast<uint32_t*>(ring_base);
    const uint32_t slot = atomicAdd(p_counter, 1u);
    __threadfence_system();
    if (slot >= max_records) {
        // Buffer full — saturating decrement keeps counter ≤ max_records
        // so the host's `payload_bytes()` call returns the correct length.
        atomicMin(p_counter, max_records);
        return;
    }

    // Compute the slot's record base address.
    const size_t record_off = PRISM_GHOST_COUNTER_SECTOR
                            + static_cast<size_t>(slot) * PRISM_GHOST_RECORD_BYTES;
    uint8_t* record_base = ring_base + record_off;

    // ── Write the 4096-byte GhostTileFrame record ────────────────────
    GhostTileFrame* hdr = reinterpret_cast<GhostTileFrame*>(record_base);
    hdr->frame_idx         = frame_idx;
    hdr->site_id           = i;
    // **M1.2.20.C-C / T20 + Amendment 3.20** — Topology-Driven Chain
    // Resolver.  No hardcoded numbers in this kernel — chain_id is
    // computed from a host-populated d_chain_offsets[8] LUT that the
    // CPU produced by parsing the topology's per-residue chain_ids
    // column.
    //
    // Algorithm: scan d_chain_offsets backwards (chain k → chain 0)
    // and select the largest k for which d_chain_offsets[k] <= repr_res.
    // Branchless via a fold: keep updating chain_id whenever a new
    // (larger) k satisfies the comparison.  Sentinel u32_MAX in unset
    // entries fails the comparison naturally (repr_res < u32_MAX is
    // true; we want the OPPOSITE direction — boundaries[k] <= repr_res
    // — so u32_MAX as a "boundary" is fine: nothing ever sits past it).
    // Path Ω Phase 2 — prefer geometry-emergent chain_id LUT.  When
    // populated (sentinel 0xFFu = unset), fall back to the legacy
    // residue-id-boundary scan against d_chain_offsets.  This makes
    // chain identity recoverable from the SO(3) tile centroids
    // computed every frame, independent of how the topology file
    // labelled chains.
    uint8_t chain_id = d_cluster_to_chain_id[i];
    if (chain_id == 0xFFu) {
        chain_id = 0u;
        const uint32_t repr_res = d_cluster_to_repr_residue[i];
        if (repr_res != 0xFFFFFFFFu) {
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                const uint32_t boundary = d_chain_offsets[k];
                if (boundary != 0xFFFFFFFFu && boundary <= repr_res) {
                    chain_id = static_cast<uint8_t>(k);
                }
            }
        }
    }
    hdr->chain_id = chain_id;
    hdr->adjudication_code = adj_code;
    // current_divergence (f32 at offset 48) ≈ Σ_planes Δ_AB at write time.
    hdr->kl_divergence =
        *reinterpret_cast<const float*>(adj + PRISM_ADJ_CURRENT_DIVERGENCE_OFF);
    // 4-plane SO(3) C_l[0..6] expansion (Amendment 3.14):
    //   plane 0 (geo)        -> power_spectrum[ 0.. 6]   wired
    //   plane 1 (causal)     -> power_spectrum[ 6..12]   wired (M1.2.25 plane-fan-out)
    //   plane 2 (thermo)     -> power_spectrum[12..18]   wired (M1.2.25 plane-fan-out)
    //   plane 3 (chemistry)  -> power_spectrum[18..24]   wired (M1.2.25 plane-fan-out)
    // All four planes are populated by so3_project.cu:512-550 into the
    // ContactShellTile; the ghost kernel now copies all four 6-band stripes
    // into the GhostTileFrame.power_spectrum[24] array. NaN/Inf inputs are
    // substituted with 0.0 and CLASS_TAINTED is set if any input is non-finite.
    const ContactShellTile& tile_i = tiles[i];
    bool plane_tainted = false;
    #pragma unroll
    for (int l = 0; l < 6; ++l) {
        const float g = tile_i.geo_power_spectrum[l];
        const float c = tile_i.caus_power_spectrum[l];
        const float t = tile_i.therm_power_spectrum[l];
        const float h = tile_i.chem_power_spectrum[l];
        hdr->power_spectrum[ 0 + l] = isfinite(g) ? g : 0.0f;
        hdr->power_spectrum[ 6 + l] = isfinite(c) ? c : 0.0f;
        hdr->power_spectrum[12 + l] = isfinite(t) ? t : 0.0f;
        hdr->power_spectrum[18 + l] = isfinite(h) ? h : 0.0f;
        plane_tainted |= !(isfinite(g) && isfinite(c) && isfinite(t) && isfinite(h));
    }
    // Wave 1 / P4 — thermo_flux populated from the relaxed manifold:
    //   thermo_flux[0] = therm_power_spectrum[0]   (water-density l=0)
    //   thermo_flux[1] = caus_power_spectrum[0]    (causal trigger l=0)
    // NaN/Inf substituted with 0.0; CLASS_TAINTED bit set so downstream
    // η = ΔV_wd / ΔV_vib integrators can exclude tainted records.
    const float therm_l0 = tile_i.therm_power_spectrum[0];
    const float caus_l0  = tile_i.caus_power_spectrum[0];
    const bool  tainted  = plane_tainted || !isfinite(therm_l0) || !isfinite(caus_l0);
    // P4(A) — the v1 kernel has no d_spikes parameter and therefore cannot
    // measure wd_change or vib_energy. It writes the declared NaN sentinel
    // rather than a plausible SO(3) duplicate, so a v1 record is recognisably
    // not-measured instead of silently wrong.
    (void)therm_l0;
    (void)caus_l0;
    hdr->thermo_flux[0] = nanf("");
    hdr->thermo_flux[1] = nanf("");
    hdr->telemetry_flags = tainted ? GHOST_TELEMETRY_CLASS_TAINTED : uint16_t{0};
    // Wave 1 / Q2 — causal_lead_residue read from the F2-pool buffer
    // populated host-side between chunks via argmax|d_kcc_temporal_corr|.
    // Sentinel u32_MAX until first KCC chunk completes → kept verbatim.
    hdr->causal_lead_residue = (d_kcc_lead != nullptr)
                               ? d_kcc_lead[i]
                               : 0xFFFFFFFFu;

    // Zero the 128-byte span at offset 128 (32 × u32).  Inert in THIS (v1)
    // kernel only: the same span is the v2/v3 MAR overlay (schema_version,
    // observation / discovery pass, perturbation channel, uv wavelength,
    // field_completeness_flags, gear_id, dt_fs, step_idx, AABB, centroid),
    // which the v2 kernel zeroes and then populates at lines 605-637 and
    // 663-688.  A record written by this kernel therefore reads
    // schema_version == 0.
    #pragma unroll
    for (int k = 0; k < 32; ++k) hdr->_reserved_payload[k] = 0u;

    // The 3840-byte _slack region is left untouched by THIS (v1) kernel.
    // That is not an allocator guarantee: cuMemHostAlloc does NOT promise
    // zero-filled pinned host memory.  The only initialization authority is
    // zero_ghost_ring_allocation in ghost_tile.rs, which write_bytes-zeroes
    // the entire allocation before the device alias is published.  That call
    // is load-bearing; deleting it as "redundant with the allocator" lets
    // arbitrary prior-process heap enter hash-chained durable segments as
    // evidence.  Skipping the write here saves 3840 B of STG bandwidth per
    // record (93.75% of the 4096 B record); the bytes a v1 record leaves in
    // this span are whatever that explicit host zeroing put there.
    //
    // Note: the ContactShellTile body that previously followed the
    // header has been retired (Amendment 3.14 §1.1).  All four
    // power_spectrum planes are copied above at lines 206-216 — the
    // "planes 1..3 fold in on the follow-up commit" note this block used to
    // carry was superseded by the M1.2.25 plane-fan-out in this same kernel.
    // The production v2/v3 kernel does write the full 256..4096 region as
    // exact cluster-residue membership (lines 692-765).
}

// ─── Host launcher ──────────────────────────────────────────────────────────

extern "C"
int prism_ghost_pipe_stage_launch(
    uint64_t      ring_base_dev,
    const void*   tiles,
    const void*   adj,
    const void*   d_kcc_lead,   // Wave 1 / Q2 — may be nullptr (kernel falls back to u32_MAX sentinel)
    uint64_t      frame_idx,
    uint32_t      n_clusters,
    uint32_t      max_records,
    void*         stream,
    uint32_t      firehose_enable)  // 0 = adj-gated emission; nonzero = unconditional per-cluster
{
    if (ring_base_dev == 0ull || tiles == nullptr || adj == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (n_clusters == 0u || max_records == 0u) {
        return static_cast<int>(cudaSuccess);
    }
    // Single block, n_clusters threads (clamped to 64 — SISR_MAX_CLUSTERS
    // matches; we share the bound for ABI consistency).
    constexpr uint32_t MAX_THREADS = 64u;
    const uint32_t threads = (n_clusters < MAX_THREADS) ? n_clusters : MAX_THREADS;
    prism_ghost_pipe_stage_kernel<<<1, threads, 0,
        static_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<uint8_t*>(ring_base_dev),
        static_cast<const ContactShellTile*>(tiles),
        static_cast<const uint8_t*>(adj),
        static_cast<const uint32_t*>(d_kcc_lead),
        frame_idx,
        n_clusters,
        max_records,
        firehose_enable
    );
    return static_cast<int>(cudaGetLastError());
}

// ─── Host helper: populate the cluster→repr-residue constant table ──────────
//
// Wave 1 / Q1 — called once per campaign init from the Rust orchestrator
// after Pillar 1 (SpikeToCluster4D) finishes computing the per-cluster
// representative residue (the residue with highest aggregate spike
// intensity inside the cluster).  Stream-ordered cudaMemcpyToSymbolAsync
// targets the __constant__ slab; the first 64 entries are written, any
// trailing entries left at the previous campaign's value (kernel reads
// only [0..n_clusters) so out-of-range entries are inert).

extern "C"
int prism_ghost_set_cluster_repr_residue(
    const uint32_t* repr_residues_host,   // [n] host array
    uint32_t        n,                    // 1..=64
    void*           stream)
{
    if (repr_residues_host == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (n == 0u) return static_cast<int>(cudaSuccess);
    if (n > 64u) n = 64u;
    cudaError_t rc = cudaMemcpyToSymbolAsync(
        d_cluster_to_repr_residue,
        repr_residues_host,
        n * sizeof(uint32_t),
        /*offset*/ 0,
        cudaMemcpyHostToDevice,
        static_cast<cudaStream_t>(stream)
    );
    return static_cast<int>(rc);
}

// **M1.2.20.C-C / T20 + Amendment 3.20** — Topology-Driven Chain
// Boundary populator.  Host parses the topology's per-residue
// chain_ids column and produces:
//
//   d_chain_offsets[0] = 0
//   d_chain_offsets[1] = first residue id of chain 1
//   d_chain_offsets[2] = first residue id of chain 2
//   ...
//   d_chain_offsets[k] = u32::MAX  (sentinel for unused chains)
//
// The Ghost stage kernel scans this LUT to assign chain_id to each
// record.  No 7C8R-specific magic numbers; works for any target the
// CPU side knows how to parse.  Stream-ordered cudaMemcpyToSymbolAsync.
extern "C"
int prism_ghost_set_chain_offsets(
    const uint32_t* offsets_host,    // [n] residue-id boundaries
    uint32_t        n,               // 1..=8
    void*           stream)
{
    if (offsets_host == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (n == 0u) return static_cast<int>(cudaSuccess);
    if (n > 8u) n = 8u;
    cudaError_t rc = cudaMemcpyToSymbolAsync(
        d_chain_offsets,
        offsets_host,
        n * sizeof(uint32_t),
        /*offset*/ 0,
        cudaMemcpyHostToDevice,
        static_cast<cudaStream_t>(stream)
    );
    return static_cast<int>(rc);
}

// Path Ω Phase 2 — host populator for d_cluster_to_chain_id LUT.
// `chain_ids_host` is a per-cluster uint8_t (0=A, 1=B, 0xFF=unset).
// `n` is clamped to 64 (constant array size).  Stream-ordered
// cudaMemcpyToSymbolAsync; the kernel reads via __constant__ broadcast.
extern "C"
int prism_ghost_set_cluster_chain_id(
    const uint8_t* chain_ids_host,
    uint32_t       n,
    void*          stream)
{
    if (chain_ids_host == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (n == 0u) return static_cast<int>(cudaSuccess);
    if (n > 64u) n = 64u;
    cudaError_t rc = cudaMemcpyToSymbolAsync(
        d_cluster_to_chain_id,
        chain_ids_host,
        n * sizeof(uint8_t),
        /*offset*/ 0,
        cudaMemcpyHostToDevice,
        static_cast<cudaStream_t>(stream)
    );
    return static_cast<int>(rc);
}

// ═══════════════════════════════════════════════════════════════════════════
// M1.2.23 §5 + §4 — Transparent MAR v2 emission kernel
// ═══════════════════════════════════════════════════════════════════════════
//
// Source-landed v2 producer. Writes the same v1 fields as the legacy kernel
// (frame_idx, site_id, chain_id, adj_code, kl_divergence, power_spectrum,
// thermo_flux, causal_lead_residue) bit-identically, plus structured v2
// fields into the 128-byte _reserved_payload region per the layout in
// ghost_tile_kernel.cuh / ghost_tile.rs.
//
// Hard invariant (per directive §4): observation_pass=true and
// discovery_pass=false MUST never trigger steering. This kernel does NOT
// control F1 SWITCH — that lives in adjudicator.cu's predicate bridge.
// The 3σ observation gate here only widens TELEMETRY emission. F1
// behavior is unchanged. Discovery semantics in the F1 path are left to
// the existing adjudicator threshold (which must equal the discovery
// threshold passed here for downstream consumers to interpret the field
// consistently — that wire-up is host-side at launch time).
//
// Emission gate (per-thread):
//   emit if adj_code >= 1                                   (legacy)
//   OR    if firehose_enable != 0                           (legacy)
//   OR    if kl_divergence >= observation_threshold_kl      (NEW v2)
//
// Per-record observation_pass / discovery_pass are written verbatim from
// the kernel-computed booleans, regardless of whether emission was gated
// by adj_code or by observation_pass. Downstream consumers can filter.

// ═══════════════════════════════════════════════════════════════════════════
// LANE A / §1 — DEVICE-SIDE TIMEKEEPER
// ═══════════════════════════════════════════════════════════════════════════
//
// Single-thread kernel that increments the device-side step counter once
// per logical step. Captured into the V2 graph body so each replay of the
// captured graph advances `*d_step_counter` by 1 (or by the unroll factor
// when the body has been time-multiplexed by §3).
//
// Operator Mandate (FULL AUTONOMY OVERRIDE 2026-05-08 §1):
//   * The GPU keeps its own time. The host no longer babysits the clock.
//   * Allocate uint64_t* d_step_counter at V2 init, initialised to
//     v2_trigger_step (host-side responsibility — see Rust counterpart).
//   * Kernel reads its step_idx via dereference of d_step_counter,
//     completely ignoring the FFI-baked argument when d_step_counter
//     is non-null.
//
// No-op when d_step_counter is nullptr (host-only / legacy fallback path).

extern "C" __global__
void prism_increment_time_kernel(unsigned long long* d_step_counter,
                                  unsigned int        increment) {
    if (d_step_counter == nullptr) return;
    if (threadIdx.x == 0u && blockIdx.x == 0u) {
        atomicAdd(d_step_counter, static_cast<unsigned long long>(increment));
    }
}

extern "C" int prism_increment_time_launch(
    void* d_step_counter,    // u64 device pointer (may be null → no-op)
    uint32_t increment,      // typically 1 (per logical step) or unroll factor
    void* stream)
{
    prism_increment_time_kernel<<<1, 1, 0,
        static_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<unsigned long long*>(d_step_counter),
        increment
    );
    return static_cast<int>(cudaGetLastError());
}

// Physical endpoint authority for MAR-v2/Ghost and ZSTR. The integrator owns
// the u32 source; the record/slot ABI owns the separate u64 destination.
// Capture after physics and before all counter consumers. Replaying an
// observer without advancing physics must not invent elapsed MD steps.
extern "C" __global__
void prism_bind_ghost_physical_clock_kernel(
    const uint32_t* physical_step, unsigned long long* ghost_step) {
    if (blockIdx.x == 0u && threadIdx.x == 0u)
        *ghost_step = static_cast<unsigned long long>(*physical_step);
}

extern "C" int prism_bind_ghost_physical_clock_launch(
    const uint32_t* physical_step, void* ghost_step, void* stream) {
    if (physical_step == nullptr || ghost_step == nullptr ||
        reinterpret_cast<uintptr_t>(physical_step) % alignof(uint32_t) != 0u ||
        reinterpret_cast<uintptr_t>(ghost_step) % alignof(unsigned long long) != 0u)
        return static_cast<int>(cudaErrorInvalidValue);
    prism_bind_ghost_physical_clock_kernel<<<1, 1, 0,
        static_cast<cudaStream_t>(stream)>>>(physical_step,
        static_cast<unsigned long long*>(ghost_step));
    return static_cast<int>(cudaGetLastError());
}

// Exact existing KCC lead policy over the immutable seed-cluster CSR.
// Bitwise float magnitude ordering preserves finite subnormal correlations
// even though this translation unit uses fast math. No arithmetic estimator
// is introduced: these are the native descriptor bits already computed.
extern "C" __global__
void prism_select_kcc_leads_kernel(
    const uint32_t* offsets, const uint32_t* residues, uint32_t n_clusters,
    const float* correlation, const uint32_t* counts,
    const uint32_t* active, uint32_t* leads) {
    const uint32_t cluster = blockIdx.x * blockDim.x + threadIdx.x;
    if (cluster >= n_clusters) return;
    uint32_t best = 0xFFFFFFFFu, best_magnitude = 0u;
    for (uint32_t j = offsets[cluster]; j < offsets[cluster + 1u]; ++j) {
        const uint32_t rid = residues[j];
        const uint32_t magnitude = __float_as_uint(correlation[rid]) & 0x7FFFFFFFu;
        if (counts[rid] < 2u || active[rid] == 0u || magnitude == 0u ||
            magnitude >= 0x7F800000u) continue;
        if (magnitude > best_magnitude || (magnitude == best_magnitude && rid < best)) {
            best = rid;
            best_magnitude = magnitude;
        }
    }
    leads[cluster] = best;
}

extern "C" int prism_select_kcc_leads_launch(
    const uint32_t* offsets, const uint32_t* residues, uint32_t n_clusters,
    const float* correlation, const uint32_t* counts,
    const uint32_t* active, uint32_t* leads, void* stream) {
    if (offsets == nullptr || residues == nullptr || correlation == nullptr ||
        counts == nullptr || active == nullptr || leads == nullptr || n_clusters == 0u)
        return static_cast<int>(cudaErrorInvalidValue);
    prism_select_kcc_leads_kernel<<<(n_clusters + 127u) / 128u, 128, 0,
        static_cast<cudaStream_t>(stream)>>>(offsets, residues, n_clusters,
        correlation, counts, active, leads);
    return static_cast<int>(cudaGetLastError());
}

// ═══════════════════════════════════════════════════════════════════════════
// V2 emission kernel (§1 device-clock-aware + §2 device-side firehose prune)
// ═══════════════════════════════════════════════════════════════════════════

extern "C" __global__
void prism_ghost_pipe_stage_kernel_v2(
    uint8_t* __restrict__       ring_base,
    const ContactShellTile* __restrict__ tiles,
    const RichSpike* __restrict__ d_spikes,
    const uint32_t* __restrict__ d_cluster_offsets,
    const uint8_t* __restrict__ adj,
    const uint32_t* __restrict__ d_kcc_lead,
    uint64_t                    frame_idx,
    uint32_t                    n_clusters,
    uint32_t                    max_records,
    uint32_t                    firehose_enable,
    // M1.2.23 §4 — Transparent MAR v2 host-passed thresholds + context.
    // P4(3.4) — sigma MULTIPLIERS, not precomputed thresholds. The absolute
    // gate is derived in-kernel from the adjudicator's armed mu/sigma so it can
    // never be stale relative to adjudication_code. 3.0 = telemetry observation
    // tier, 12.0 = steering/discovery tier.
    float                       sigma_mult_obs,
    float                       sigma_mult_disc,
    uint32_t                    gear_id,
    float                       dt_fs,
    uint64_t                    step_idx,
    // OPERATOR MANDATE 2026-05-08 §1 — device-side timekeeper.
    // Non-null pointer ⇒ kernel reads its actual step_idx via
    // *d_step_counter, ignoring the FFI-baked step_idx arg above. Null
    // pointer preserves the legacy host-baked behavior bit-for-bit.
    const unsigned long long* d_step_counter,
    // OPERATOR MANDATE 2026-05-08 §2 — device-side firehose prune gate.
    // Records below this kl_divergence magnitude are skipped before the
    // ring-buffer atomic claim, even when firehose_enable is set. Set to
    // 0.0f to disable pruning (legacy unconditional-firehose behavior).
    float                       prune_kl_threshold,
    NativeV2SeedLayout seed_layout
)
{
    const uint32_t i = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n_clusters) return;

    const uint8_t adj_code = adj[PRISM_ADJ_ADJUDICATION_CODE_OFF];
    const float kl = *reinterpret_cast<const float*>(adj + PRISM_ADJ_CURRENT_DIVERGENCE_OFF);

    // ── P4(3.4) — DERIVE THE GATE FROM THE ARMED ADJUDICATOR, NOT FROM A
    //             HOST SCALAR BAKED AT GRAPH-CAPTURE TIME ──────────────────
    //
    // PRIOR DEFECT: `observation_threshold_kl` and `discovery_threshold_kl`
    // arrived as host floats, fixed when the captured graph was recorded. The
    // adjudicator meanwhile arms its own threshold from live μ/σ, which Dynamic
    // T7 measures in-graph. So `discovery_pass` (record byte 133) was computed
    // against a STALE scalar while `adjudication_code` in the same record used
    // the LIVE one: one record, two detectors, silently disagreeing.
    //
    // The kernel already dereferences `adj` by byte offset for the code and the
    // divergence, so it reads the armed gate from the same struct. The host now
    // supplies only the σ MULTIPLIERS, which are policy, not measurement. That
    // makes the invariant STRUCTURAL rather than a wiring obligation: the two
    // detectors read the same memory and cannot diverge.
    //
    // Mirrors adjudicator.cu exactly, including the armed sentinel
    // (`total_kl_sigma > 0.0f`) and the legacy noise-floor arm. The legacy arm
    // is NOT dead code — it is the only branch the production monomer lane
    // takes, because write_kl_native_mu_sigma is called solely from the coupled
    // lane.
    const float nf_mu     = *reinterpret_cast<const float*>(adj + 0);    // noise_floor_mu[0]
    const float nf_sigma  = *reinterpret_cast<const float*>(adj + 24);   // noise_floor_sigma[0]
    const float kln_mu    = *reinterpret_cast<const float*>(adj + 148);  // total_kl_mu
    const float kln_sigma = *reinterpret_cast<const float*>(adj + 152);  // total_kl_sigma
    const bool  kl_native_ready = (kln_sigma > 0.0f);
    const float base_mu    = kl_native_ready ? kln_mu    : nf_mu;
    const float base_sigma = kl_native_ready ? kln_sigma : nf_sigma;
    const float observation_threshold_kl = base_mu + sigma_mult_obs  * base_sigma;
    const float discovery_threshold_kl   = base_mu + sigma_mult_disc * base_sigma;

    const bool obs_pass = isfinite(kl) && (kl >= observation_threshold_kl);
    const bool disc_pass = isfinite(kl) && (kl >= discovery_threshold_kl);
    // §2 — Device-side firehose prune. The directive specifies:
    //   write IF burst_motion > 0.5 OR ccns_phase != cold_hold OR kl_divergence > 0.01
    // burst_motion + ccns_phase are not yet plumbed into ContactShellTile;
    // the kernel uses kl_divergence as the available physics-activity
    // proxy. Records with finite-but-tiny kl are dead/inert and don't
    // need to cross the PCIe bus. The host-passed `prune_kl_threshold`
    // is 0.01f by default per the directive; setting it to 0.0f disables
    // the prune (legacy unconditional firehose).
    // The public ABI and every Rust caller define exactly 0.0f as the
    // explicit opt-out that restores the original unconditional diagnostic
    // firehose.  Treating zero as an ordinary threshold accidentally pruned
    // records whose finite KL was exactly zero, contradicting that contract
    // and leaving a firehose run with an empty ring.  Negative/non-finite
    // thresholds do not disable pruning; they fail closed through kl_active.
    const bool prune_disabled = (prune_kl_threshold == 0.0f);
    const bool kl_active = isfinite(kl)
                        && isfinite(prune_kl_threshold)
                        && (prune_kl_threshold > 0.0f)
                        && (fabsf(kl) > prune_kl_threshold);

    // Widened emission gate (v2 + §2):
    //   * Always emit on Construct/Violation (adj_code >= 1)
    //   * Always emit on observation-pass (kl >= obs_thr)
    //   * In firehose mode with a positive finite prune threshold, ONLY emit
    //     if the frame is adjudicated, passes observation, or has physically
    //     active KL.  The explicit 0.0 threshold disables that prune and
    //     preserves the original unconditional diagnostic firehose.
    //   * In non-firehose mode, the legacy gate (adj_code || obs_pass)
    //     stands.
    if (firehose_enable != 0u) {
        if (!prune_disabled && adj_code < 1u && !obs_pass && !kl_active) return;
    } else {
        if (adj_code < 1u && !obs_pass) return;
    }

    // Atomic counter at offset 0.
    uint32_t* counter = reinterpret_cast<uint32_t*>(ring_base);
    const uint32_t slot = atomicAdd(counter, 1u);
    if (slot >= max_records) {
        atomicSub(counter, 1u);
        return;
    }

    const size_t record_off = PRISM_GHOST_COUNTER_SECTOR
                            + static_cast<size_t>(slot) * PRISM_GHOST_RECORD_BYTES;
    uint8_t* record_base = ring_base + record_off;
    GhostTileFrame* hdr = reinterpret_cast<GhostTileFrame*>(record_base);

    // ─── v1 fields (bit-identical with legacy kernel) ─────────────────────
    hdr->frame_idx = frame_idx;
    hdr->site_id   = i;
    uint8_t chain_id = d_cluster_to_chain_id[i];
    if (chain_id == 0xFFu) {
        chain_id = 0u;
        const uint32_t repr_res = d_cluster_to_repr_residue[i];
        if (repr_res != 0xFFFFFFFFu) {
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                const uint32_t boundary = d_chain_offsets[k];
                if (boundary != 0xFFFFFFFFu && boundary <= repr_res) {
                    chain_id = static_cast<uint8_t>(k);
                }
            }
        }
    }
    hdr->chain_id          = chain_id;
    hdr->adjudication_code = adj_code;
    hdr->kl_divergence     = kl;

    // M1.2.25 plane-fan-out: copy all four SO(3) planes (geo/caus/therm/chem)
    // from the ContactShellTile into the GhostTileFrame.power_spectrum[24]
    // array. so3_project.cu:512-550 already populates all four planes via
    // WMMA Tensor Core reduction; this kernel was previously dropping planes
    // 1-3 to zero. NaN/Inf substituted with 0.0 + CLASS_TAINTED on any miss.
    const ContactShellTile& tile_i = tiles[i];
    bool plane_tainted = false;
    #pragma unroll
    for (int l = 0; l < 6; ++l) {
        const float g = tile_i.geo_power_spectrum[l];
        const float c = tile_i.caus_power_spectrum[l];
        const float t = tile_i.therm_power_spectrum[l];
        const float h = tile_i.chem_power_spectrum[l];
        hdr->power_spectrum[ 0 + l] = isfinite(g) ? g : 0.0f;
        hdr->power_spectrum[ 6 + l] = isfinite(c) ? c : 0.0f;
        hdr->power_spectrum[12 + l] = isfinite(t) ? t : 0.0f;
        hdr->power_spectrum[18 + l] = isfinite(h) ? h : 0.0f;
        plane_tainted |= !(isfinite(g) && isfinite(c) && isfinite(t) && isfinite(h));
    }

    const float therm_l0 = tile_i.therm_power_spectrum[0];
    const float caus_l0  = tile_i.caus_power_spectrum[0];
    const bool  tainted  = plane_tainted || !isfinite(therm_l0) || !isfinite(caus_l0);

    // ── P4(A) — thermo_flux carries the quantities the SCHEMA declares ──────
    //
    // PRIOR DEFECT: this field is declared as [wd_change, vib_energy] and drives
    // eta = dV_wd / dV_vib downstream, but it was written with
    // therm_power_spectrum[0] and caus_power_spectrum[0] — the l=0 monopole
    // bands of the SO(3) decomposition. Those are bit-exact duplicates of
    // power_spectrum[12] and power_spectrum[6], so the field carried ZERO
    // independent information. Worse, the causality plane is structurally empty
    // (no CUDA code writes RichSpike::causal_lag), so its band is the KL_EPS pad
    // 1e-7, and eta degenerated to therm_C0 * 1e7 — THE RECIPROCAL OF AN EPSILON
    // GUARD, published as a physical efficiency into binding-site evidence and a
    // Path-B feature vector.
    //
    // It also substituted 0.0f on non-finite input, so CLASS_TAINTED never fired
    // and a scanner labelled the result "computed". The schema explicitly asks
    // for NaN in that case so a reader can tell "not measured" from "zero".
    //
    // NaN is written HERE as the declared not-measured sentinel. The v3 CSR walk
    // below overwrites it with real per-cluster means of RichSpike::wd_change
    // (d rho_w / dt, solvation flux) and RichSpike::vib_energy — the quantities
    // the schema names — but only when spikes actually contributed. A cluster
    // that measured nothing keeps NaN, which is the honest answer.
    hdr->thermo_flux[0]    = nanf("");
    hdr->thermo_flux[1]    = nanf("");
    hdr->telemetry_flags   = tainted ? GHOST_TELEMETRY_CLASS_TAINTED : uint16_t{0};

    // Initialize the v2 header span BEFORE publishing any overlay fields.
    // The plane-presence mask at offset 196 is inside this span (128..255).
    // Clearing it after publication silently erases measured plane occupancy.
    #pragma unroll
    for (int k = 0; k < 32; ++k) hdr->_reserved_payload[k] = 0u;

    // ── P4(B) — EXACT, EPSILON-FREE PLANE-POPULATED MASK ───────────────────
    //
    // The KL_EPS 1e-7 pad in so3_project is CORRECT and stays: it keeps log(0)
    // out of the adjudicator, which hard-traps on non-finite values in the
    // Causality and Thermodynamics planes. The defect was never the pad — it was
    // that three consumers infer "plane is empty" by testing the spectrum
    // against == 0.0, and 1e-7 is not 0.0. A structurally empty plane therefore
    // read as POPULATED, and ghost_cosine_sim on two constant [1e-7 x6] vectors
    // returns exactly 1.0 — contributing W_CAUS 0.25 + W_CHEM 0.15, i.e. 40% of
    // the lattice edge SO(3) weight, as a constant perfect similarity. The
    // "no common plane -> neutral" escape written for this case was dead code.
    //
    // The tile already carries the exact answer: sum_w_<plane> is the summed
    // weight that produced that plane's spectrum, so the plane is empty if and
    // only if its sum_w is exactly zero. No epsilon, no threshold, no inference
    // from the padded spectrum. Published as a 4-bit mask so decoders read a
    // declared fact instead of re-deriving one from a value that was padded
    // precisely so it could never be zero.
    //
    // Bit 0 = Geometry, 1 = Causality, 2 = Thermodynamics, 3 = Chemistry.
    {
        uint32_t plane_mask = 0u;
        if (tile_i.sum_w_geo   > 0.0f) plane_mask |= 1u << 0;
        if (tile_i.sum_w_caus  > 0.0f) plane_mask |= 1u << 1;
        if (tile_i.sum_w_therm > 0.0f) plane_mask |= 1u << 2;
        if (tile_i.sum_w_chem  > 0.0f) plane_mask |= 1u << 3;
        // Bit 8 declares the mask itself present, so a reader can distinguish
        // "all four planes empty" from "this producer never wrote a mask".
        plane_mask |= 1u << 8;
        *reinterpret_cast<uint32_t*>(record_base + GHOST_V2_OFFSET_V2_RESERVED) = plane_mask;
    }
    hdr->causal_lead_residue = (d_kcc_lead != nullptr) ? d_kcc_lead[i] : 0xFFFFFFFFu;

    // ─── v2 fields (writes into the same 128 B span the v1 kernel zeros) ──
    // The reserved span was cleared before the plane mask was published.
    // Do not clear it here: that would erase the mask at offset 196.

    // Schema version + threshold pass flags.
    *reinterpret_cast<uint32_t*>(record_base + GHOST_V2_OFFSET_SCHEMA_VERSION) = GHOST_FRAME_SCHEMA_V3;
    record_base[GHOST_V2_OFFSET_OBSERVATION_PASS]  = obs_pass  ? uint8_t{1} : uint8_t{0};
    record_base[GHOST_V2_OFFSET_DISCOVERY_PASS]    = disc_pass ? uint8_t{1} : uint8_t{0};
    record_base[GHOST_V2_OFFSET_PERTURBATION_CHAN] = GHOST_PERTURBATION_CHANNEL_UNKNOWN;
    *reinterpret_cast<uint16_t*>(record_base + GHOST_V2_OFFSET_UV_WAVELENGTH_NM) = uint16_t{0};
    *reinterpret_cast<uint16_t*>(record_base + GHOST_V2_OFFSET_FIELD_COMPLETE_FLAGS) =
        // Bit 0: kl finite, Bit 1: gear_id provided non-zero, Bit 2: dt_fs > 0,
        // Bit 3: thermo_flux not class-tainted.
        //
        // GEAR-PROVENANCE-002: the not-measured sentinel is non-zero, so a
        // bare `gear_id != 0` test would report an ABSENT gear as "provided"
        // — the exact misparse the sentinel exists to prevent. Excluding it
        // preserves bit 1's meaning for every real gear (0..3) unchanged.
        (isfinite(kl) ? 0x0001u : 0u)
      | ((gear_id != 0u && gear_id != GHOST_V2_GEAR_ID_UNMEASURED) ? 0x0002u : 0u)
      | (dt_fs > 0.0f ? 0x0004u : 0u)
      | (!tainted ? 0x0008u : 0u);
    *reinterpret_cast<uint32_t*>(record_base + GHOST_V2_OFFSET_GEAR_ID) = gear_id;
    *reinterpret_cast<float*>(record_base + GHOST_V2_OFFSET_DT_FS) = dt_fs;
    // OPERATOR MANDATE 2026-05-08 §1 — read step_idx from the device-side
    // timekeeper when present. The host's FFI step_idx becomes the seed /
    // legacy-fallback value; the device clock is authoritative whenever
    // d_step_counter is non-null. This eradicates the temporal-collapse
    // bug where every Ghost v2 record carried a static step_idx baked in
    // at graph-build time (root cause: captured_pipeline.rs:4009 TODO).
    const uint64_t effective_step_idx = (d_step_counter != nullptr)
        ? static_cast<uint64_t>(*d_step_counter)
        : step_idx;
    *reinterpret_cast<uint64_t*>(record_base + GHOST_V2_OFFSET_STEP_IDX) = effective_step_idx;

    // ─── GHOST_NATIVE_SPATIAL_MAPPING_WIRE — native AABB + centroid ───
    //
    // ContactShellTile.aabb_min[4] / aabb_max[4] (xyz + 1 pad) are reduced in
    // so3_project.cu:271-272 from RichSpike positions in the protein topology
    // coordinate frame (Å) and written into the tile at so3_project.cu:563-570.
    // tile_i is already in scope from the plane-fan-out
    // block above (line 579). The same `i` indexes both this Ghost record
    // (site_id = i) and the source tile, so the spatial mapping is exact at
    // write time — no host-side derivation needed.
    //
    // The selected centroid view is the AABB midpoint, an honest scalar
    // proxy of the cluster's spatial extent. It is NOT the full
    // phase-manifold centroid family (spike_density / kcc_driver /
    // phasor_coherent / thermo_weighted / ghost_zstr_event_weighted views);
    // those richer views are computed offline by the SiteManifest
    // materializer when the per-residue evidence sources are joined. The
    // record's compatibility centroid_xyz field carries the alias label
    // "aabb_midpoint_native_contact_shell_tile" via the
    // field_completeness_flags Bit 4 = SPATIAL_NATIVE_AABB_MIDPOINT.
    //
    // Order matters: this write lands AFTER the _reserved_payload[k]=0
    // blanket clear above so the zero loop does not erase the spatial
    // payload. If either bound is non-finite (NaN/Inf from upstream
    // SO(3) projection), substitute 0.0 and leave Bit 4 unset — the
    // host audit treats Bit 4 == 0 as "spatial fields not native-populated."
    {
        const float* aabb_min_src = tile_i.aabb_min;
        const float* aabb_max_src = tile_i.aabb_max;
        float* aabb_min_dst = reinterpret_cast<float*>(record_base + GHOST_V2_OFFSET_AABB_MIN);
        float* aabb_max_dst = reinterpret_cast<float*>(record_base + GHOST_V2_OFFSET_AABB_MAX);
        float* centroid_dst = reinterpret_cast<float*>(record_base + GHOST_V2_OFFSET_CENTROID);
        bool spatial_native_finite = true;
        #pragma unroll
        for (int k = 0; k < 3; ++k) {
            const float lo_raw = aabb_min_src[k];
            const float hi_raw = aabb_max_src[k];
            const bool  ok = isfinite(lo_raw) && isfinite(hi_raw);
            const float lo = ok ? lo_raw : 0.0f;
            const float hi = ok ? hi_raw : 0.0f;
            aabb_min_dst[k] = lo;
            aabb_max_dst[k] = hi;
            centroid_dst[k] = 0.5f * (lo + hi); // AABB midpoint — labeled alias
            spatial_native_finite = spatial_native_finite && ok;
        }
        // Field completeness Bit 4 = SPATIAL_NATIVE_AABB_MIDPOINT populated.
        // OR-merge with prior bits set above so we don't clobber Bits 0..3.
        if (spatial_native_finite) {
            uint16_t* fcf = reinterpret_cast<uint16_t*>(record_base + GHOST_V2_OFFSET_FIELD_COMPLETE_FLAGS);
            *fcf = static_cast<uint16_t>(*fcf | 0x0010u);
        }
    }
    // _v2_reserved @ 196 contains the plane-presence mask; bytes 200..255
    // retain their deterministic zero initialization.

    // ─── v3 exact RichSpike cluster-residue membership ──────────────────
    // Each cluster thread owns its output record, so a deterministic serial
    // scan over that cluster's CSR RichSpike range requires no atomics. The
    // bounded payload stores exact unique residue ids and spike multiplicities.
    // This is cluster membership, not a cavity-lining/contact assignment.
    // Overflow is recorded and the production loader rejects the complete
    // stream; truncation can never silently change cluster identity.
    uint32_t* membership_schema = reinterpret_cast<uint32_t*>(
        record_base + GHOST_V3_OFFSET_MEMBERSHIP_SCHEMA_VERSION);
    uint32_t* membership_count = reinterpret_cast<uint32_t*>(
        record_base + GHOST_V3_OFFSET_CLUSTER_RESIDUE_COUNT);
    uint32_t* membership_capacity = reinterpret_cast<uint32_t*>(
        record_base + GHOST_V3_OFFSET_CLUSTER_RESIDUE_CAPACITY);
    uint32_t* membership_overflow = reinterpret_cast<uint32_t*>(
        record_base + GHOST_V3_OFFSET_CLUSTER_RESIDUE_OVERFLOW);
    uint32_t* source_spike_count = reinterpret_cast<uint32_t*>(
        record_base + GHOST_V3_OFFSET_SOURCE_SPIKE_COUNT);
    uint32_t* unresolved_count = reinterpret_cast<uint32_t*>(
        record_base + GHOST_V3_OFFSET_UNRESOLVED_RESIDUE_SPIKE_COUNT);
    uint32_t* membership_entry_bytes = reinterpret_cast<uint32_t*>(
        record_base + GHOST_V3_OFFSET_CLUSTER_RESIDUE_ENTRY_BYTES);
    uint32_t* geometry_flags = reinterpret_cast<uint32_t*>(
        record_base + GHOST_V3_OFFSET_GEOMETRY_AVAILABILITY_FLAGS);
    uint32_t* entries = reinterpret_cast<uint32_t*>(
        record_base + GHOST_V3_OFFSET_CLUSTER_RESIDUE_ENTRIES);

    *membership_schema = GHOST_V3_MEMBERSHIP_SCHEMA_VERSION;
    *membership_count = 0u;
    *membership_capacity = GHOST_V3_CLUSTER_RESIDUE_CAPACITY;
    *membership_overflow = 0u;
    *source_spike_count = 0u;
    *unresolved_count = 0u;
    *membership_entry_bytes = GHOST_V3_CLUSTER_RESIDUE_ENTRY_BYTES;
    *geometry_flags = GHOST_V3_GEOMETRY_COORDINATE_SIDECAR_REQUIRED;
    #pragma unroll
    for (int word = 0; word < 8; ++word) {
        reinterpret_cast<uint32_t*>(
            record_base + GHOST_V3_OFFSET_COORDINATE_FRAME_SHA256)[word] = 0u;
    }

    const uint32_t begin = d_cluster_offsets[i];
    const uint32_t end = d_cluster_offsets[i + 1u];
    if (end < begin) {
        *membership_overflow = 1u;
    } else {
        *source_spike_count = end - begin;
        // P4(A) — measure what thermo_flux declares, from the spikes that
        // actually produced this tile. Same CSR range, same pass, no extra
        // traversal and no new data movement.
        float thermo_wd_sum = 0.0f;
        float thermo_vib_sum = 0.0f;
        uint32_t thermo_n = 0u;
        for (uint32_t spike_index = begin; spike_index < end; ++spike_index) {
            {
                const float wd = load_native_v2_seed(d_spikes,spike_index,seed_layout).wd_change;
                const float vb = load_native_v2_seed(d_spikes,spike_index,seed_layout).vib_energy;
                if (isfinite(wd) && isfinite(vb)) {
                    thermo_wd_sum += wd;
                    thermo_vib_sum += vb;
                    thermo_n += 1u;
                }
            }
            const int32_t residue_id = load_native_v2_seed(d_spikes,spike_index,seed_layout).residue_id;
            if (residue_id < 0) {
                *unresolved_count += 1u;
                continue;
            }
            uint32_t found = GHOST_V3_CLUSTER_RESIDUE_CAPACITY;
            for (uint32_t membership_index = 0; membership_index < *membership_count; ++membership_index) {
                if (entries[membership_index * 2u] == static_cast<uint32_t>(residue_id)) {
                    found = membership_index;
                    break;
                }
            }
            if (found < GHOST_V3_CLUSTER_RESIDUE_CAPACITY) {
                entries[found * 2u + 1u] += 1u;
            } else if (*membership_count < GHOST_V3_CLUSTER_RESIDUE_CAPACITY) {
                const uint32_t write_index = *membership_count;
                entries[write_index * 2u] = static_cast<uint32_t>(residue_id);
                entries[write_index * 2u + 1u] = 1u;
                *membership_count = write_index + 1u;
            } else {
                *membership_overflow = 1u;
            }
        }
        // Overwrite the NaN sentinel ONLY if this cluster measured something.
        // A cluster whose spikes carried no finite wd_change / vib_energy keeps
        // NaN, which is the declared not-measured state rather than a zero.
        if (thermo_n > 0u) {
            const float inv_n = 1.0f / static_cast<float>(thermo_n);
            hdr->thermo_flux[0] = thermo_wd_sum * inv_n;
            hdr->thermo_flux[1] = thermo_vib_sum * inv_n;
        }
    }
    if (*membership_overflow == 0u) {
        *geometry_flags |= GHOST_V3_GEOMETRY_CLUSTER_RESIDUE_MEMBERSHIP_EXACT;
    }
}

extern "C"
int prism_ghost_pipe_stage_launch_v2(
    uint64_t      ring_base_dev,
    const void*   tiles,
    const void*   d_spikes,
    const void*   d_cluster_offsets,
    const void*   adj,
    const void*   d_kcc_lead,
    uint64_t      frame_idx,
    uint32_t      n_clusters,
    uint32_t      max_records,
    void*         stream,
    uint32_t      firehose_enable,
    float         observation_threshold_kl,
    float         discovery_threshold_kl,
    uint32_t      gear_id,
    float         dt_fs,
    uint64_t      step_idx,
    // OPERATOR MANDATE 2026-05-08 §1 — device-side timekeeper.
    // u64 device pointer to a single uint64_t step counter. When the
    // engine has wired the device clock, the kernel reads *d_step_counter
    // and writes that into every emitted record. When 0 (null), the kernel
    // falls back to the FFI step_idx arg above (legacy behavior).
    uint64_t      d_step_counter,
    // OPERATOR MANDATE 2026-05-08 §2 — device-side firehose prune.
    // kl_divergence threshold below which a record is silently dropped
    // when firehose_enable is set. 0.0f preserves legacy unconditional-
    // firehose behavior.
    float         prune_kl_threshold)
{
    if (ring_base_dev == 0ull || tiles == nullptr || d_spikes == nullptr
        || d_cluster_offsets == nullptr || adj == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (n_clusters == 0u || max_records == 0u) {
        return static_cast<int>(cudaSuccess);
    }
    constexpr uint32_t MAX_THREADS = 64u;
    if (n_clusters > MAX_THREADS) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const uint32_t threads = (n_clusters < MAX_THREADS) ? n_clusters : MAX_THREADS;
    prism_ghost_pipe_stage_kernel_v2<<<1, threads, 0,
        static_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<uint8_t*>(ring_base_dev),
        static_cast<const ContactShellTile*>(tiles),
        static_cast<const RichSpike*>(d_spikes),
        static_cast<const uint32_t*>(d_cluster_offsets),
        static_cast<const uint8_t*>(adj),
        static_cast<const uint32_t*>(d_kcc_lead),
        frame_idx,
        n_clusters,
        max_records,
        firehose_enable,
        observation_threshold_kl,
        discovery_threshold_kl,
        gear_id,
        dt_fs,
        step_idx,
        reinterpret_cast<const unsigned long long*>(d_step_counter),
        prune_kl_threshold, NativeV2SeedLayout{}
    );
    return static_cast<int>(cudaGetLastError());
}

extern "C"
int prism_ghost_pipe_stage_launch_v2_seed(
    uint64_t      ring_base_dev,
    const void*   tiles,
    const void*   d_spikes,
    const void*   d_cluster_offsets,
    const void*   adj,
    const void*   d_kcc_lead,
    uint64_t      frame_idx,
    uint32_t      n_clusters,
    uint32_t      max_records,
    void*         stream,
    uint32_t      firehose_enable,
    float         observation_threshold_kl,
    float         discovery_threshold_kl,
    uint32_t      gear_id,
    float         dt_fs,
    uint64_t      step_idx,
    // OPERATOR MANDATE 2026-05-08 §1 — device-side timekeeper.
    // u64 device pointer to a single uint64_t step counter. When the
    // engine has wired the device clock, the kernel reads *d_step_counter
    // and writes that into every emitted record. When 0 (null), the kernel
    // falls back to the FFI step_idx arg above (legacy behavior).
    uint64_t      d_step_counter,
    // OPERATOR MANDATE 2026-05-08 §2 — device-side firehose prune.
    // kl_divergence threshold below which a record is silently dropped
    // when firehose_enable is set. 0.0f preserves legacy unconditional-
    // firehose behavior.
    float         prune_kl_threshold,
    NativeV2SeedLayout seed_layout)
{
    if (!valid_native_v2_seed_layout(seed_layout)) return static_cast<int>(cudaErrorInvalidValue);
    if (ring_base_dev == 0ull || tiles == nullptr || d_spikes == nullptr
        || d_cluster_offsets == nullptr || adj == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (n_clusters == 0u || max_records == 0u) {
        return static_cast<int>(cudaSuccess);
    }
    constexpr uint32_t MAX_THREADS = 64u;
    if (n_clusters > MAX_THREADS) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const uint32_t threads = (n_clusters < MAX_THREADS) ? n_clusters : MAX_THREADS;
    prism_ghost_pipe_stage_kernel_v2<<<1, threads, 0,
        static_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<uint8_t*>(ring_base_dev),
        static_cast<const ContactShellTile*>(tiles),
        static_cast<const RichSpike*>(d_spikes),
        static_cast<const uint32_t*>(d_cluster_offsets),
        static_cast<const uint8_t*>(adj),
        static_cast<const uint32_t*>(d_kcc_lead),
        frame_idx,
        n_clusters,
        max_records,
        firehose_enable,
        observation_threshold_kl,
        discovery_threshold_kl,
        gear_id,
        dt_fs,
        step_idx,
        reinterpret_cast<const unsigned long long*>(d_step_counter),
        prune_kl_threshold, seed_layout
    );
    return static_cast<int>(cudaGetLastError());
}

// Exact module state transfer at an all-worker quiescent boundary.
// Caller owns the context, holds every producer and synchronizes before entry.
extern "C" size_t prism_checkpoint_ghost_tile_kernel_bytes() { return sizeof(d_cluster_to_repr_residue) + sizeof(d_chain_offsets) + sizeof(d_cluster_to_chain_id); }
extern "C" int prism_checkpoint_ghost_tile_kernel_capture(unsigned char* bytes, size_t length, cudaStream_t stream) {
    if (!bytes || length != prism_checkpoint_ghost_tile_kernel_bytes()) return (int)cudaErrorInvalidValue;
    size_t offset = 0; cudaError_t rc;
    rc = cudaMemcpyFromSymbolAsync(bytes + offset, d_cluster_to_repr_residue, sizeof(d_cluster_to_repr_residue), 0, cudaMemcpyDeviceToHost, stream);
    if (rc != cudaSuccess) return (int)rc;
    offset += sizeof(d_cluster_to_repr_residue);
    rc = cudaMemcpyFromSymbolAsync(bytes + offset, d_chain_offsets, sizeof(d_chain_offsets), 0, cudaMemcpyDeviceToHost, stream);
    if (rc != cudaSuccess) return (int)rc;
    offset += sizeof(d_chain_offsets);
    rc = cudaMemcpyFromSymbolAsync(bytes + offset, d_cluster_to_chain_id, sizeof(d_cluster_to_chain_id), 0, cudaMemcpyDeviceToHost, stream);
    if (rc != cudaSuccess) return (int)rc;
    offset += sizeof(d_cluster_to_chain_id);
    return (int)cudaStreamSynchronize(stream);
}
extern "C" int prism_checkpoint_ghost_tile_kernel_restore(const unsigned char* bytes, size_t length, cudaStream_t stream) {
    if (!bytes || length != prism_checkpoint_ghost_tile_kernel_bytes()) return (int)cudaErrorInvalidValue;
    size_t offset = 0; cudaError_t rc;
    rc = cudaMemcpyToSymbolAsync(d_cluster_to_repr_residue, bytes + offset, sizeof(d_cluster_to_repr_residue), 0, cudaMemcpyHostToDevice, stream);
    if (rc != cudaSuccess) return (int)rc;
    offset += sizeof(d_cluster_to_repr_residue);
    rc = cudaMemcpyToSymbolAsync(d_chain_offsets, bytes + offset, sizeof(d_chain_offsets), 0, cudaMemcpyHostToDevice, stream);
    if (rc != cudaSuccess) return (int)rc;
    offset += sizeof(d_chain_offsets);
    rc = cudaMemcpyToSymbolAsync(d_cluster_to_chain_id, bytes + offset, sizeof(d_cluster_to_chain_id), 0, cudaMemcpyHostToDevice, stream);
    if (rc != cudaSuccess) return (int)rc;
    offset += sizeof(d_cluster_to_chain_id);
    return (int)cudaStreamSynchronize(stream);
}
