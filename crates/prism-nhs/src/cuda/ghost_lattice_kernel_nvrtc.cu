// ═════════════════════════════════════════════════════════════════════════
// PRISM-4D / GhostPhaseLattice4D — edge adjudication kernel (device-only)
// ═════════════════════════════════════════════════════════════════════════
//
// This file is BOTH:
//   * the NVRTC-compatible source (compiled at runtime via cudarc::nvrtc when
//     the static archive isn't reachable), and
//   * `#include`d from the static-archive wrapper `ghost_lattice_kernel.cu`,
//     which adds the extern "C" host orchestrator.
//
// Therefore: NO host code, NO CUB, NO cudaError_t, NO `<cuda_runtime.h>`.
// Pure `__global__` / `__device__`.
//
// Layout of `GhostPhaseLatticeNode` is byte-pinned against the Rust struct
// in `ghost_phase_lattice.rs` via static_assert in the wrapper .cu (NVRTC
// can't run static_assert against the host struct, so the wrapper carries
// the assertion). Total node size: 208 B, alignment 8 B.
//
// All node fields are read via `__ldg` (read-only cache) — the kernel never
// mutates the node array.

#ifndef PRISM_GHOST_LATTICE_NVRTC_INCLUDED
#define PRISM_GHOST_LATTICE_NVRTC_INCLUDED 1

#include <cstdint>

// Sentinels — must match `ghost_phase_lattice.rs` constants.
#define GHOST_LATTICE_CAUSAL_LEAD_NONE 0xFFFFFFFFu
#define GHOST_LATTICE_GEAR_NONE        0xFFFFFFFFu
#define GHOST_LATTICE_CCNS_NONE        0xFFFFu
#define GHOST_LATTICE_PHASE_COLD_HOLD  0u
#define GHOST_LATTICE_PHASE_HEATING    1u
#define GHOST_LATTICE_PHASE_WARM_HOLD  2u
#define GHOST_LATTICE_PHASE_COOLING    3u

#define GHOST_LATTICE_PLANE_GEOMETRY   0x01u
#define GHOST_LATTICE_PLANE_CAUSALITY  0x02u
#define GHOST_LATTICE_PLANE_THERMO     0x04u
#define GHOST_LATTICE_PLANE_CHEMISTRY  0x08u

// Plane weighting from directive Part II.2 — must sum to 1.0.
#define GHOST_LATTICE_W_GEO  0.35f
#define GHOST_LATTICE_W_CAUS 0.25f
#define GHOST_LATTICE_W_THER 0.25f
#define GHOST_LATTICE_W_CHEM 0.15f

// Score quantization for the global telemetry accumulator. The kernel
// atomically adds `score * SCORE_QUANT` (cast to u64) into the device-side
// score-sum buffer; the host divides by SCORE_QUANT to recover the float.
#define GHOST_LATTICE_SCORE_QUANT 1000000.0f
#define GHOST_LATTICE_EDGE_SCHEMA_VERSION 1u

// ─── Node mirror (byte-pinned to Rust GhostPhaseLatticeNode) ──────────────

struct __align__(8) GhostPhaseLatticeNodeDev {
    uint32_t tile_index;        //   0
    uint16_t stream_id;         //   4
    uint16_t _pad0;             //   6
    uint32_t site_id;           //   8
    uint32_t _pad1;             //  12
    uint64_t frame_idx;         //  16
    uint64_t step_idx;          //  24

    uint8_t  protocol_phase;    //  32
    uint8_t  _pad2[3];          //  33
    uint32_t step_bucket;       //  36
    uint16_t ccns_phase_bin;    //  40
    uint16_t _pad3;             //  42
    uint32_t gear_id;           //  44

    float    aabb_min[3];       //  48
    float    aabb_max[3];       //  60
    float    centroid_xyz[3];   //  72

    float    kl_divergence;     //  84
    float    thermo_flux[2];    //  88
    float    water_density;     //  96
    uint32_t causal_lead_residue; // 100

    float    so3_power_spectrum[4][6]; // 104
    uint8_t  so3_plane_status;  // 200
    uint8_t  _pad4[7];          // 201..208
};

struct __align__(8) GhostPhaseLatticeAcceptedEdgeDev {
    uint32_t from_node_index;
    uint32_t to_node_index;
    uint64_t from_step_idx;
    uint64_t to_step_idx;
    uint64_t temporal_delta_steps;
    float    total_score;
    float    so3_score;
    float    causal_score;
    float    thermo_score;
    uint8_t  from_protocol_phase;
    uint8_t  to_protocol_phase;
    uint8_t  common_so3_plane_mask;
    uint8_t  predicate_flags;
    uint32_t schema_version;
};

// ─── Helpers ──────────────────────────────────────────────────────────────

__device__ __forceinline__ float ghost_cosine_sim(const float* a, const float* b) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        float ai = a[i], bi = b[i];
        dot += ai * bi;
        na  += ai * ai;
        nb  += bi * bi;
    }
    if (na <= 0.0f || nb <= 0.0f) return 0.0f;
    float denom = sqrtf(na) * sqrtf(nb);
    return (denom > 0.0f) ? (dot / denom) : 0.0f;
}

// ECL-CC / Jaiganesh-Burtscher monotone walk (same shape as
// `gpu_cluster_backend.rs:154-192`). Each `unite_pair` invocation advances
// strictly toward a smaller root; bounded by tree depth (~log N) under
// path compression.
__device__ __forceinline__ int ghost_find_root(int* parent, int i) {
    int curr = parent[i];
    if (curr != i) {
        int prev = i;
        int next;
        while (curr > (next = parent[curr])) {
            parent[prev] = next;
            prev = curr;
            curr = next;
        }
    }
    return curr;
}

__device__ __forceinline__ void ghost_union(int* parent, int u, int v) {
    bool repeat;
    do {
        repeat = false;
        u = ghost_find_root(parent, u);
        v = ghost_find_root(parent, v);
        if (u < v) {
            int old = atomicMin(&parent[v], u);
            if (old != v && old != u) {
                v = old;
                repeat = true;
            }
        } else if (u > v) {
            int old = atomicMin(&parent[u], v);
            if (old != u && old != v) {
                u = old;
                repeat = true;
            }
        }
    } while (repeat);
}

// Decide whether two nodes' lattice cells are within the directive's
// neighborhood (±1 spatial × ±1 phase × ±1 step bucket).
__device__ __forceinline__ bool ghost_cells_adjacent(
    int ax, int ay, int az, int aphase, int abucket,
    int bx, int by, int bz, int bphase, int bbucket
) {
    if (abs(ax - bx) > 1) return false;
    if (abs(ay - by) > 1) return false;
    if (abs(az - bz) > 1) return false;
    if (abs(aphase - bphase) > 1) return false;
    if (abs(abucket - bbucket) > 1) return false;
    return true;
}

// ─── Init parent ──────────────────────────────────────────────────────────

extern "C" __global__ void ghost_lattice_init_parent(
    int* __restrict__ parent,
    const unsigned int n_nodes
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;
    parent[i] = (int)i;
}

// ─── Edge adjudication kernel ─────────────────────────────────────────────
//
// Thread-per-cell. Each thread:
//   1. Reads its cell coordinates (cx, cy, cz, phase, bucket) from cell_table.
//   2. Walks every other cell, checks if it's within the ±1 neighborhood.
//   3. For each neighbor cell pair (a, b) with a <= b, iterates every
//      (i, j) pair where i in cell a, j in cell b. Pairs with i == j are
//      skipped; pairs are deduplicated by step_idx-then-tile_index ordering
//      (visit the pair with strictly lower (step_idx, tile_index) as A).
//   4. Boolean adjudication (arrow of time, temporal adjacency, AABB
//      overlap, monotone protocol phase) — early reject on any failure.
//   5. Continuous physics scoring (4-plane SO(3) cosine sim, causal driver
//      continuity, thermo continuity).
//   6. If composite score >= threshold: union-find merge + atomic accum
//      of global edge telemetry.
//
// Telemetry counters (`pair_count`, `phase_legal_count`, `aabb_overlap_count`,
// `edge_count`, `edge_score_sum`) accumulate via atomicAdd at predicate
// transition points so the host can audit the funnel:
//   raw_pairs → phase_legal → aabb_overlap → temporally_adjacent →
//   so3_threshold_pass.

extern "C" __global__ void ghost_lattice_edge_kernel(
    const unsigned char* __restrict__ nodes_bytes,
    const unsigned int n_nodes,
    const unsigned int* __restrict__ permutation,
    const unsigned int* __restrict__ cell_first,
    const unsigned int* __restrict__ cell_count,
    const unsigned int n_cells,
    const int*          __restrict__ cell_table,
    int*                __restrict__ parent,
    unsigned long long* __restrict__ edge_score_sum,
    unsigned long long* __restrict__ edge_count,
    unsigned char*      __restrict__ accepted_edges_bytes,
    const unsigned long long accepted_edge_capacity,
    unsigned long long* __restrict__ accepted_edge_overflow,
    unsigned long long* __restrict__ pair_count,
    unsigned long long* __restrict__ phase_legal_count,
    unsigned long long* __restrict__ aabb_overlap_count,
    const float spatial_cell_size_a,
    const unsigned long long max_temporal_edge_steps,
    const float so3_threshold,
    const unsigned int pass_kind
) {
    const unsigned int a_cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (a_cell >= n_cells) return;

    const GhostPhaseLatticeNodeDev* nodes =
        reinterpret_cast<const GhostPhaseLatticeNodeDev*>(nodes_bytes);

    // Skip the cell-size check: the host already guarantees > 0 (clamped to
    // 1e-3 in `build_lattice_cells`). Unused here directly because the
    // neighborhood predicate is integer-cell adjacency, not Euclidean
    // distance — but kept in the signature so future variants can use it.
    (void)spatial_cell_size_a;

    const int ax      = cell_table[5 * a_cell + 0];
    const int ay      = cell_table[5 * a_cell + 1];
    const int az      = cell_table[5 * a_cell + 2];
    const int aphase  = cell_table[5 * a_cell + 3];
    const int abucket = cell_table[5 * a_cell + 4];

    const unsigned int a_first = cell_first[a_cell];
    const unsigned int a_count = cell_count[a_cell];

    unsigned long long local_pair_count = 0ULL;
    unsigned long long local_phase_legal = 0ULL;
    unsigned long long local_aabb_overlap = 0ULL;
    unsigned long long local_score_sum_q = 0ULL;

    for (unsigned int b_cell = a_cell; b_cell < n_cells; ++b_cell) {
        const int bx      = cell_table[5 * b_cell + 0];
        const int by      = cell_table[5 * b_cell + 1];
        const int bz      = cell_table[5 * b_cell + 2];
        const int bphase  = cell_table[5 * b_cell + 3];
        const int bbucket = cell_table[5 * b_cell + 4];

        if (!ghost_cells_adjacent(ax, ay, az, aphase, abucket,
                                  bx, by, bz, bphase, bbucket)) {
            continue;
        }

        const unsigned int b_first = cell_first[b_cell];
        const unsigned int b_count = cell_count[b_cell];

        for (unsigned int ai = 0; ai < a_count; ++ai) {
            const unsigned int i_node = permutation[a_first + ai];
            const GhostPhaseLatticeNodeDev& A = nodes[i_node];

            // When (a_cell == b_cell), avoid duplicating each pair —
            // start `bj` at `ai + 1` so we visit (i, j) only once.
            const unsigned int bj_start = (a_cell == b_cell) ? (ai + 1) : 0u;

            for (unsigned int bj = bj_start; bj < b_count; ++bj) {
                const unsigned int j_node = permutation[b_first + bj];
                if (i_node == j_node) continue;
                local_pair_count++;

                const GhostPhaseLatticeNodeDev& B = nodes[j_node];

                // Pair-direction ordering: visit each unordered pair once by
                // taking the strictly-earlier (step_idx, tile_index) node
                // as A. Ensures the arrow-of-time predicate makes sense.
                const GhostPhaseLatticeNodeDev* lo = &A;
                const GhostPhaseLatticeNodeDev* hi = &B;
                if (B.step_idx < A.step_idx ||
                    (B.step_idx == A.step_idx && B.tile_index < A.tile_index)) {
                    lo = &B;
                    hi = &A;
                }

                // ── Boolean: monotone protocol-phase transition ──
                const uint32_t lp = lo->protocol_phase;
                const uint32_t hp = hi->protocol_phase;
                bool legal_phase = false;
                if (lp == GHOST_LATTICE_PHASE_COLD_HOLD) {
                    legal_phase = (hp == GHOST_LATTICE_PHASE_COLD_HOLD ||
                                   hp == GHOST_LATTICE_PHASE_HEATING);
                } else if (lp == GHOST_LATTICE_PHASE_HEATING) {
                    legal_phase = (hp == GHOST_LATTICE_PHASE_HEATING ||
                                   hp == GHOST_LATTICE_PHASE_WARM_HOLD);
                } else if (lp == GHOST_LATTICE_PHASE_WARM_HOLD) {
                    legal_phase = (hp == GHOST_LATTICE_PHASE_WARM_HOLD ||
                                   hp == GHOST_LATTICE_PHASE_COOLING);
                } else if (lp == GHOST_LATTICE_PHASE_COOLING) {
                    legal_phase = (hp == GHOST_LATTICE_PHASE_COOLING);
                }
                if (!legal_phase) continue;
                local_phase_legal++;

                // ── Boolean: arrow of time + temporal adjacency ──
                const unsigned long long lo_step = lo->step_idx;
                const unsigned long long hi_step = hi->step_idx;
                if (hi_step < lo_step) continue; // arrow of time
                if ((hi_step - lo_step) > max_temporal_edge_steps) continue;

                // ── Boolean: AABB overlap (6-float test) ──
                bool overlap =
                    (lo->aabb_max[0] >= hi->aabb_min[0] &&
                     lo->aabb_min[0] <= hi->aabb_max[0]) &&
                    (lo->aabb_max[1] >= hi->aabb_min[1] &&
                     lo->aabb_min[1] <= hi->aabb_max[1]) &&
                    (lo->aabb_max[2] >= hi->aabb_min[2] &&
                     lo->aabb_min[2] <= hi->aabb_max[2]);
                if (!overlap) continue;
                local_aabb_overlap++;

                // ── Continuous: 4-plane SO(3) cosine similarity ──
                float so3 = 0.0f;
                float wsum = 0.0f;
                const uint8_t s = lo->so3_plane_status & hi->so3_plane_status;

                if (s & GHOST_LATTICE_PLANE_GEOMETRY) {
                    so3  += GHOST_LATTICE_W_GEO  *
                            ghost_cosine_sim(lo->so3_power_spectrum[0],
                                             hi->so3_power_spectrum[0]);
                    wsum += GHOST_LATTICE_W_GEO;
                }
                if (s & GHOST_LATTICE_PLANE_CAUSALITY) {
                    so3  += GHOST_LATTICE_W_CAUS *
                            ghost_cosine_sim(lo->so3_power_spectrum[1],
                                             hi->so3_power_spectrum[1]);
                    wsum += GHOST_LATTICE_W_CAUS;
                }
                if (s & GHOST_LATTICE_PLANE_THERMO) {
                    so3  += GHOST_LATTICE_W_THER *
                            ghost_cosine_sim(lo->so3_power_spectrum[2],
                                             hi->so3_power_spectrum[2]);
                    wsum += GHOST_LATTICE_W_THER;
                }
                if (s & GHOST_LATTICE_PLANE_CHEMISTRY) {
                    so3  += GHOST_LATTICE_W_CHEM *
                            ghost_cosine_sim(lo->so3_power_spectrum[3],
                                             hi->so3_power_spectrum[3]);
                    wsum += GHOST_LATTICE_W_CHEM;
                }
                // No common plane populated → neutral 1.0 per directive
                // Part V "Missing SO(3) Planes". Otherwise normalise the
                // cosine-sim by the active weight sum so cosines weighted
                // across only 1-3 planes are not artificially deflated.
                const float final_so3 = (wsum > 0.0f) ? (so3 / wsum) : 1.0f;

                // ── Continuous: causal driver continuity ──
                float causal_score = 1.0f; // directive Part V "Missing
                                          // Causal Drivers" → neutral
                if (lo->causal_lead_residue != GHOST_LATTICE_CAUSAL_LEAD_NONE &&
                    hi->causal_lead_residue != GHOST_LATTICE_CAUSAL_LEAD_NONE) {
                    causal_score = (lo->causal_lead_residue ==
                                    hi->causal_lead_residue) ? 1.2f : 0.8f;
                }

                // ── Continuous: KL-divergence smoothness ──
                // Both KL values may be NaN when the sidecar hasn't been
                // populated yet. NaN propagates to the multiplier as 1.0
                // (neutral) per directive Part V.
                float thermo_score = 1.0f;
                const float lo_kl = lo->kl_divergence;
                const float hi_kl = hi->kl_divergence;
                if (isfinite(lo_kl) && isfinite(hi_kl)) {
                    const float kl_diff = fabsf(lo_kl - hi_kl);
                    thermo_score = 1.0f / (1.0f + kl_diff);
                }

                const float total = final_so3 * causal_score * thermo_score;
                if (total < so3_threshold) continue;

                // ── Accept edge ──
                const unsigned long long edge_slot = atomicAdd(edge_count, 1ULL);
                local_score_sum_q +=
                    static_cast<unsigned long long>(total * GHOST_LATTICE_SCORE_QUANT);

                // Convert lo/hi pointers back to original indices for the
                // union. The pointers are aliases into the original `nodes`
                // array, so `lo - nodes` is the canonical index.
                const unsigned int lo_idx =
                    static_cast<unsigned int>(lo - nodes);
                const unsigned int hi_idx =
                    static_cast<unsigned int>(hi - nodes);
                if (pass_kind == 1u && edge_slot < accepted_edge_capacity) {
                    GhostPhaseLatticeAcceptedEdgeDev* accepted_edges =
                        reinterpret_cast<GhostPhaseLatticeAcceptedEdgeDev*>(
                            accepted_edges_bytes);
                    GhostPhaseLatticeAcceptedEdgeDev row;
                    row.from_node_index = lo_idx;
                    row.to_node_index = hi_idx;
                    row.from_step_idx = lo_step;
                    row.to_step_idx = hi_step;
                    row.temporal_delta_steps = hi_step - lo_step;
                    row.total_score = total;
                    row.so3_score = final_so3;
                    row.causal_score = causal_score;
                    row.thermo_score = thermo_score;
                    row.from_protocol_phase = static_cast<uint8_t>(lp);
                    row.to_protocol_phase = static_cast<uint8_t>(hp);
                    row.common_so3_plane_mask = s;
                    row.predicate_flags = 0x0fu;
                    row.schema_version = GHOST_LATTICE_EDGE_SCHEMA_VERSION;
                    accepted_edges[edge_slot] = row;
                } else if (pass_kind == 1u) {
                    atomicAdd(accepted_edge_overflow, 1ULL);
                }
                if (pass_kind == 0u) {
                    ghost_union(parent, (int)lo_idx, (int)hi_idx);
                }
            }
        }
    }

    // Drain thread-local accumulators into the global telemetry once per
    // thread (rather than per accepted edge) — far less atomic contention.
    if (pass_kind == 0u && local_pair_count > 0)
        atomicAdd(pair_count, local_pair_count);
    if (pass_kind == 0u && local_phase_legal > 0)
        atomicAdd(phase_legal_count, local_phase_legal);
    if (pass_kind == 0u && local_aabb_overlap > 0)
        atomicAdd(aabb_overlap_count, local_aabb_overlap);
    if (local_score_sum_q  > 0) atomicAdd(edge_score_sum,     local_score_sum_q);
}

// ─── Path compression (final pass) ────────────────────────────────────────

extern "C" __global__ void ghost_lattice_path_compress(
    int* __restrict__ parent,
    const unsigned int n_nodes
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;
    int r = ghost_find_root(parent, (int)i);
    parent[i] = r;
}

#endif // PRISM_GHOST_LATTICE_NVRTC_INCLUDED
