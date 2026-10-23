//! Build script for prism-nhs
//!
//! Compiles CUDA kernels to PTX when the `gpu` feature is enabled.

use std::env;
use std::fs::OpenOptions;
use std::io::Read;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};

const GHOST_POLICY_PATH_ENV: &str = "PRISM_GHOST_TOPOLOGY_POLICY_PATH";
const GHOST_POLICY_SHA256_ENV: &str = "PRISM_GHOST_TOPOLOGY_POLICY_SHA256";
const GHOST_POLICY_BUILD_PATH_ENV: &str = "PRISM_GHOST_TOPOLOGY_POLICY_BUILD_PATH";

/// The governed package-PTX authority is
/// prism-gpu/GOVERNED_PTX_MEMBERS.json. prism-nhs contributes zero package
/// PTX and must never write into the workspace target/ptx mirror.
pub const PRISM_NHS_GOVERNED_PTX_CONTRIBUTION: usize = 0;

fn main() {
    println!("cargo:rerun-if-changed=src/cuda/prism_governor_measurement.cuh");
    println!("cargo:rerun-if-changed=src/cuda/prism_physical_guard.cuh");
    println!("cargo:rerun-if-changed=../prism-gpu/src/kernels/nhs_biprism_shared_step_abi.cuh");
    println!("cargo:rerun-if-changed=../prism-gpu/src/kernels/nhs_biprism_coordinate_precision.cuh");
    println!("cargo:rerun-if-changed=src/cuda/prism_asc_decision.cuh");
    emit_ghost_policy_binding();
    println!("cargo:rerun-if-env-changed=PRISM_NHS_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=PRISM_GPU_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=PRISMCODE_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=PRISM_NHS_NVCC");
    println!("cargo:rerun-if-env-changed=PRISM_GPU_NVCC");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    println!("cargo:rerun-if-changed=src/cuda/spike_density.cu");
    println!("cargo:rerun-if-changed=src/cuda/spike_to_cluster_4d.cu");
    println!("cargo:rerun-if-changed=src/cuda/spike_to_cluster_4d.cuh");
    println!("cargo:rerun-if-changed=src/cuda/lbvh_morton.cu");
    println!("cargo:rerun-if-changed=src/cuda/lbvh_morton.cuh");
    println!("cargo:rerun-if-changed=src/cuda/vram_pool.cu");
    println!("cargo:rerun-if-changed=src/cuda/vram_pool.cuh");
    println!("cargo:rerun-if-changed=src/cuda/gpu_invariant.cu");
    println!("cargo:rerun-if-changed=src/cuda/gpu_invariant.cuh");
    println!("cargo:rerun-if-changed=src/cuda/pre_rank.cu");
    println!("cargo:rerun-if-changed=src/cuda/pre_rank.cuh");
    println!("cargo:rerun-if-changed=src/cuda/rich_spike.cu");
    println!("cargo:rerun-if-changed=src/cuda/rich_spike.cuh");
    println!("cargo:rerun-if-changed=src/cuda/lbvh_tree.cu");
    println!("cargo:rerun-if-changed=src/cuda/lbvh_tree.cuh");
    println!("cargo:rerun-if-changed=src/cuda/sh_basis.cu");
    println!("cargo:rerun-if-changed=src/cuda/sh_basis.cuh");
    println!("cargo:rerun-if-changed=src/cuda/so3_project.cu");
    println!("cargo:rerun-if-changed=src/cuda/so3_project.cuh");
    println!("cargo:rerun-if-changed=src/cuda/adjudicator.cu");
    println!("cargo:rerun-if-changed=src/cuda/adjudicator.cuh");
    println!("cargo:rerun-if-changed=src/cuda/zstr_kernels.cu");
    println!("cargo:rerun-if-changed=src/cuda/zstr_kernels.cuh");
    println!("cargo:rerun-if-changed=src/cuda/asc_steering.cu");
    println!("cargo:rerun-if-changed=src/cuda/asc_steering.cuh");
    println!("cargo:rerun-if-changed=src/cuda/symmetry_consensus.cu");
    println!("cargo:rerun-if-changed=src/cuda/symmetry_consensus.cuh");
    println!("cargo:rerun-if-changed=src/cuda/dynamic_t7.cu");
    println!("cargo:rerun-if-changed=src/cuda/dynamic_t7.cuh");
    println!("cargo:rerun-if-changed=src/cuda/gearbox.cu");
    println!("cargo:rerun-if-changed=src/cuda/gearbox.cuh");
    println!("cargo:rerun-if-changed=src/cuda/energy_monitor.cu");
    println!("cargo:rerun-if-changed=src/cuda/energy_monitor.cuh");
    // Package 2 — deterministic reduction substrate for governor decision
    // inputs.  Compiled IEEE-strict (no --use_fast_math); see
    // `compile_to_static_archive_ieee754` below and the ordering-guarantee
    // block at the top of deterministic_reduce.cuh.
    println!("cargo:rerun-if-changed=src/cuda/deterministic_reduce.cu");
    println!("cargo:rerun-if-changed=src/cuda/deterministic_reduce.cuh");
    // Package 3 — physical stability governor.  The .cuh is `__host__
    // __device__ __forceinline__` throughout, which does NOT exempt it from
    // compilation: a static_assert in a header no translation unit includes is
    // never evaluated, and every size/offset pin, the lexicographic rank-order
    // chain and the band-monotonicity chain in that file would be decorative.
    // `physical_governor.cu` is the TU that makes them binding; both files are
    // tracked so a change to either re-runs nvcc.
    println!("cargo:rerun-if-changed=src/cuda/physical_governor.cu");
    println!("cargo:rerun-if-changed=src/cuda/physical_governor.cuh");
    // M1.2.19.B — Asynchronous Manifold Sequencer Channel-B (ghost tiles).
    println!("cargo:rerun-if-changed=src/cuda/ghost_tile_kernel.cu");
    println!("cargo:rerun-if-changed=src/cuda/ghost_tile_kernel.cuh");
    // GhostPhaseLattice4D — physically-constrained 4D edge adjudication.
    // Replaces the legacy O(N²) DBSCAN on Ghost v2 records. The wrapper
    // .cu pulls in `ghost_lattice_kernel_nvrtc.cu` via #include so the
    // same kernel source serves both the static archive (this build) and
    // the NVRTC fallback (compiled at runtime in `ghost_phase_lattice.rs`).
    println!("cargo:rerun-if-changed=src/cuda/ghost_lattice_kernel.cu");
    println!("cargo:rerun-if-changed=src/cuda/ghost_lattice_kernel.cuh");
    println!("cargo:rerun-if-changed=src/cuda/ghost_lattice_kernel_nvrtc.cu");
    // TIER 7 (2026-05-03) — CUDA 13.x cuGraphAddNode FFI wrappers.
    println!("cargo:rerun-if-changed=src/cuda/graph_node.cu");
    println!("cargo:rerun-if-changed=src/cuda/graph_node.cuh");
    // TIER 7+8 / WHILE bounded micro-loop (2026-05-04) — predicate
    // kernel + host shim for the post-T7 deferred-drain WHILE
    // conditional (sub-ticket WHILE-POST-T7-DRAIN-001, kernel portion).
    // Single-thread `__global__` reads device-resident queue depth +
    // iteration counter, enforces a kernel-level `max_iterations` cap,
    // and writes the boolean to a `cudaGraphConditionalHandle` via
    // `cudaGraphSetConditional`. Parent-graph-resident, host_mutation
    // = false, max_iterations is the hard cap. NOT yet wired into any
    // captured graph — `captured_pipeline.rs` wire-in deferred to a
    // follow-up commit after R4 (F1 pipeline wiring) lands; WHILE FFI
    // scaffold is R5's commit.
    println!("cargo:rerun-if-changed=src/cuda/while_drain_bridge.cu");
    println!("cargo:rerun-if-changed=src/cuda/while_drain_bridge.cuh");
    // Package 4 — gearbox-facing TSH arm.  The .cuh was header-only and no
    // translation unit included it, so none of its fifteen static_asserts had
    // ever executed and its layout claims were unbacked by any build.  The .cu
    // below is its compilation consumer.
    println!("cargo:rerun-if-changed=src/cuda/tsh_gearbox_arm.cu");
    println!("cargo:rerun-if-changed=src/cuda/tsh_gearbox_arm.cuh");

    // The appliance layout is /opt/prismos/bin/<binary> plus
    // /opt/prismos/lib/libsdst.so. Never embed the developer checkout in a
    // release ELF; keep the runtime binding relocatable with the executable.
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../lib");

    let gpu_enabled = env::var("CARGO_FEATURE_GPU").is_ok();
    if !gpu_enabled {
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let nvcc = find_nvcc().expect(
        "nvcc not found. CUDA toolkit is required for the `gpu` feature.\n\
         Set CUDA_HOME or ensure nvcc is in PATH.",
    );
    println!("cargo:info=Using nvcc: {}", nvcc);
    println!("cargo:info=Using CUDA target arch: {}", cuda_arch());

    // M1.2 §3.2 HALT enforcement: run the atomic-or geometric-tie-
    // breaker grep gate against the M1 lane's CUDA sources before
    // invoking nvcc. If the gate fails, abort the build with the
    // gate's own diagnostic — do not proceed to compile a kernel
    // that violates the §M2 anti-legacy-centroid rule.
    run_atomic_or_grep_gate();

    let spike_density_ptx = out_dir.join("spike_density.ptx");
    compile_kernel(&nvcc, "src/cuda/spike_density.cu", &spike_density_ptx);
    assert_non_governed_ptx_output(&out_dir, &spike_density_ptx);

    // M1.2: compile the SpikeToCluster4D producer to a static archive
    // (NOT PTX). This .cu file contains both __global__ kernels and
    // extern "C" host-side orchestration functions that call CUB
    // device-wide algorithms (cub::DeviceReduce::Sum,
    // cub::DeviceSegmentedReduce, cub::DeviceRadixSort::SortPairs).
    // CUB device-wide algorithms internally launch kernels from host
    // code, so a PTX target won't link them — we need a static archive
    // that the prism-nhs lib can link against directly. cudart and
    // libstdc++ are pulled in transitively because the .a contains
    // cudaMemcpyAsync / cudaStreamSynchronize calls.
    compile_to_static_archive(
        &nvcc,
        "src/cuda/spike_to_cluster_4d.cu",
        "spike_to_cluster_4d",
        &out_dir,
    );

    // LBVH lane Phase 1: Morton 30-bit encoder. Same static-archive
    // path as the M1 producer — the .cu defines a __global__ kernel
    // launched from an extern "C" host orchestrator. No CUB usage in
    // this archive (Phase 1 is a single kernel). Phase 2 (Karras
    // tree builder + sort) will land alongside.
    compile_to_static_archive(&nvcc, "src/cuda/lbvh_morton.cu", "lbvh_morton", &out_dir);

    // F2 lane: stream-ordered memory pool (cudaMemPool_t-backed) +
    // VRAM audit telemetry struct. Three single-thread atomic
    // kernels (init / record_alloc / record_free) plus four host
    // orchestrators that wrap cudaMemPool runtime API calls. Static
    // archive so the cudaMemPool_t / cudaMallocFromPoolAsync runtime
    // calls link in.
    compile_to_static_archive(&nvcc, "src/cuda/vram_pool.cu", "vram_pool", &out_dir);

    // Rectification Phase 1: hard-trap GPU invariant enforcement.
    // Single audit kernel + gpu_hard_assert __device__ helper that
    // fires the PTX `trap` instruction on invariant violation. The
    // M1 Conservation-of-Mass audit is the first consumer; future
    // M1 / M2 invariants follow the same pattern.
    compile_to_static_archive(
        &nvcc,
        "src/cuda/gpu_invariant.cu",
        "gpu_invariant",
        &out_dir,
    );

    // Continuous Learning Architecture Phase 1: RichSpike 64-byte
    // schema. Header-only struct + helpers; .cu is the link-probe
    // anchor. Future kernels (Morton encoder upgrade, telemetry
    // compression) extend this archive.
    compile_to_static_archive(&nvcc, "src/cuda/rich_spike.cu", "rich_spike", &out_dir);

    // LBVH Phase 2: Karras 2012 radix-tree builder + last-arrival
    // atomic-flag bottom-up AABB reduce. Two kernels + one init
    // kernel + extern "C" host orchestration. 64-byte cache-line-
    // aligned LBVHNode for line-aligned LDG.E.128 access during the
    // reduce.
    compile_to_static_archive(&nvcc, "src/cuda/lbvh_tree.cu", "lbvh_tree", &out_dir);

    // Rectification Phase 2: shift-left MAR pre-rank adjudicator.
    // Three single-purpose kernels (compute_aabb_volumes,
    // compute_energy_density, pre_rank_adjudicator) plus host
    // orchestrators. The adjudicator's 3-way output is the
    // cudaGraphConditionalNode SWITCH selector. Includes the §2.3
    // SAD-PATH guard: NaN/Inf observables route to Case 2
    // (Violation).
    compile_to_static_archive(&nvcc, "src/cuda/pre_rank.cu", "pre_rank", &out_dir);

    // RECT-3.1.a: Spherical Harmonics Y_lm evaluator (Lmax=5).
    // Straight-line PTX device helper + init kernel that populates
    // the device-side K_LM normalization table. Feeds RECT-3.1.b
    // (so3_project_manifold_kernel) which accumulates a_lm via WMMA
    // Tensor Core fragments.
    compile_to_static_archive(&nvcc, "src/cuda/sh_basis.cu", "sh_basis", &out_dir);

    // RECT-3.1.b: SO(3) projection kernel + ContactShellTile
    // (alignas(128), 384 B). Consumes RichSpike clusters, evaluates
    // Y_lm per spike via the inline helper from sh_basis.cuh,
    // accumulates per-cluster a_lm via warp-shuffle reduce, computes
    // C_l = Σ_m |a_lm|², writes one ContactShellTile per cluster.
    // RECT-3.1.c will replace the warp-shuffle reduction with WMMA
    // Tensor Core fragments without changing the on-tile layout.
    compile_to_static_archive(&nvcc, "src/cuda/so3_project.cu", "so3_project", &out_dir);

    // T0/T1/T2/T3/T4: InterferometricAdjudicator + Quantum-Photonic
    // Bridge + KL-Divergence + ASC + clock64 telemetry (Blackwell
    // Convergence directive). Single 128-byte FFI struct, __constant__
    // μ_01² LUT derived from `crate::config` extinction coefficients,
    // PTX NaN/le0 guards make the kernel a Total Function over all
    // 32-bit f32 inputs. ASC kernel atomicAdds into the existing
    // `fused_engine.rs::d_forces` buffer (Anti-Greenfield § 2.1).
    compile_to_static_archive(&nvcc, "src/cuda/adjudicator.cu", "adjudicator", &out_dir);

    // TIER 7 (2026-05-03) — CUDA 13.x cuGraphAddNode unified
    // graph-builder FFI wrappers. Two host-only helpers
    // (`prism_graph_add_child_node_v3_ffi`,
    // `prism_graph_add_memset_node_v3_ffi`) that internally invoke
    // `cudaGraphAddNode` with `cudaGraphNodeParams`. Replaces Rust-
    // side calls to legacy per-type adders (`cuGraphAddChildGraphNode`,
    // `cuGraphAddMemsetNode`) which mishandle CUDA 13.x conditional-
    // handle metadata attached to body subgraphs (the
    // childgraph-IF failure class).  See
    // `crates/prism-nhs/src/cuda/graph_node.cuh` for full rationale.
    compile_to_static_archive(&nvcc, "src/cuda/graph_node.cu", "graph_node", &out_dir);

    // TIER 7+8 / WHILE bounded micro-loop (2026-05-04) —
    // `while_drain_bridge.cu` exposes
    // `prism_while_drain_predicate_kernel` (single-thread predicate
    // bridge driving a `cudaGraphCondTypeWhile` conditional handle)
    // and `prism_while_drain_launch_predicate` (host shim). Mirrors
    // the `gearbox.cu:436-477` predicate-bridge pattern and gates on
    // `CUDART_VERSION >= 12040` for `cudaGraphSetConditional`
    // availability. See `while_drain_bridge.cuh` for full topology
    // rationale. Wire-in to `captured_pipeline.rs` is deferred to a
    // post-R4 follow-up commit; this archive only stages the symbols
    // for R5's WHILE FFI scaffold to reference.
    compile_to_static_archive(
        &nvcc,
        "src/cuda/while_drain_bridge.cu",
        "while_drain_bridge",
        &out_dir,
    );

    // ZSTR Phase 1: Zero-Stall Telemetry Ring kernels.
    // zstr_signal_completion_kernel: __threadfence_system() + pinned fence
    // write at end of captured graph epoch.
    // zstr_pos_stage_f4_kernel: vectorized LDG.E.128 → STG.E.128 position
    // staging into pinned triple-buffer slot.
    compile_to_static_archive(&nvcc, "src/cuda/zstr_kernels.cu", "zstr_kernels", &out_dir);

    // ZSTR Phase 2: Vectorized ASC repulsion force injection.
    // asc_inject_repulsion_v4_kernel: float4 loads + atom.global.add.v4.f32
    // for concurrent 4-wide force accumulation into d_forces.
    // Gate G22_ATOMIC_v4_VERIFIED: ptxas audit required post-compile.
    compile_to_static_archive(&nvcc, "src/cuda/asc_steering.cu", "asc_steering", &out_dir);

    // G28 SISR (Spatially-Indexed Symmetric Reflection) — bilateral-truth
    // symmetry consensus gate for homodimer targets.  Operates on
    // ContactShellTile array between SO(3) projection and Adjudicator
    // step inside the captured graph window.  Brute-force pairwise
    // centroid-distance check (n_clusters ≤ 64); writes a u64 prune
    // mask consumed by the Adjudicator step kernel.
    compile_to_static_archive(
        &nvcc,
        "src/cuda/symmetry_consensus.cu",
        "symmetry_consensus",
        &out_dir,
    );

    // Wave 3 / Path B — Dynamic T7 substrate-aware noise-floor calibration.
    // Three captured kernels (capture / reduce / apply) run inside the V2
    // pipeline graph downstream of the Adjudicator step.  All GPU-native;
    // no host bridges per launch.  Replaces the locked 4LPK priors with
    // values derived from the running substrate's thermal-equilibrium Δ_AB
    // distribution after PRISM_DYNT7_N_MIN samples accumulate.
    compile_to_static_archive(&nvcc, "src/cuda/dynamic_t7.cu", "dynamic_t7", &out_dir);

    // Wave B.1 — G26 Chronometric Gearbox foundation.  __constant__ float
    // d_gearbox_table[16] (3 active gears + NaN-sentinel Gear 3) +
    // ChronometricStateTensor (16-byte F2-pool persistent state) +
    // prism_gearbox_pointer_swap_kernel (Stateful Finite Automaton: reads
    // adj->adjudication_code, mutates cruise tensor, writes the active
    // gear's dt through *(adj->d_dt) — the address T12 Pre-Flight wired
    // to &d_protocol->dt).  Hello-world test in gearbox.rs proves the
    // 2.0fs → 4.0fs flip without host intervention.
    compile_to_static_archive(&nvcc, "src/cuda/gearbox.cu", "gearbox", &out_dir);

    // Wave B.3.2 — Hamiltonian Auditor (CUB-based potential-energy reduce).
    // Captured node at the end of the FusedStep sequence reduces the
    // per-atom potential-energy components buffer into a single f32 via
    // cub::DeviceReduce::Sum, then promotes to f64 and rolls the energy
    // window (E_n / E_{n-1}) for the SFA stability fuse.  AMBER kernel-
    // side per-atom PE accumulation is a separate Wave (M1.2.17 —
    // "Hamiltonian Recovery"); B.3.2 lands the FFI surface + captured-
    // graph node so M1.2.17 just wires the source buffer.
    compile_to_static_archive(
        &nvcc,
        "src/cuda/energy_monitor.cu",
        "energy_monitor",
        &out_dir,
    );

    // Package 2 — Deterministic Reduction Substrate.
    //
    // Produces the governor's DECISION inputs (force norm, energy drift,
    // min/max probes) with a fixed-order pairwise reduction whose summation
    // order is independent of thread/block scheduling and of atomic arrival
    // order.  Float addition is not associative, so an atomicAdd reduction
    // (as in zstr_force_stage_f4_kernel) varies run-to-run in the low
    // mantissa bits and can flip a threshold comparison — two identical runs
    // would then take different gears and the provenance record would
    // faithfully report different outcomes for the same input.
    //
    // Compiled through the IEEE-strict entry point: `--use_fast_math` is
    // WITHDRAWN for this translation unit only.  Fast math implies
    // `-ftz=true -fmad=true -prec-div=false -prec-sqrt=false`; FMA
    // contraction and denormal flushing both perturb the reduced value
    // relative to the host reference, which would break the off-device
    // re-derivation of a recorded decision.  Every other .cu in this crate
    // keeps fast math — only the decision path pays for strictness.
    compile_to_static_archive_ieee754(
        &nvcc,
        "src/cuda/deterministic_reduce.cu",
        "deterministic_reduce",
        &out_dir,
    );

    // Package 3 — PHYSICAL STABILITY GOVERNOR.
    //
    // `physical_governor.cuh` carries the composed evaluator (base gates 1..11
    // from gearbox.cuh plus the seven extension guards, the HMR hard gate and
    // the extension binding gate), its three pinned wire structs, and a wall of
    // `static_assert`s: three `sizeof` pins (208 / 112 / 128), ten `offsetof`
    // pins, the eighteen-link lexicographic authority chain that mechanically
    // proves no salience can outrank a physical prohibition, and the
    // band-monotonicity chain.
    //
    // NONE of those fired before this registration existed. The header was
    // `#include`d by no .cu, so no compiler ever evaluated a single one of
    // them, and `prism_physical_governor_decide` existed in no binary. That is
    // the "built but never activated, assertions that never fire" defect class
    // this package repairs — a header-only decision authority is not compiled
    // just because it is written; it is compiled when a translation unit
    // includes it and a build step names it.
    //
    // `physical_governor.cu` is that translation unit. It exposes
    // `prism_physical_governor_decide_host` (the same evaluator the SFA kernel
    // path calls), `prism_physical_governor_decide_device` +
    // `prism_physical_governor_decide_kernel` (so the __device__ half is driven
    // through ptxas, not merely parsed), `prism_physgov_default_ext_policy`,
    // and the ABI probe the Rust mirror cross-checks against.
    //
    // IEEE-strict, like `deterministic_reduce.cu` and for the same reason: this
    // translation unit's arithmetic is a CONTROL DECISION that gets compared
    // against a threshold and then recorded as the reason a gear changed. Under
    // fast math the compiler may contract `0.5f*a*dt*dt` into an FMA and flush
    // denormals, which is deterministic on-device but no longer bit-reproducible
    // by the Rust host mirror in `physical_governor.rs` — and a band comparison
    // that lands differently on the two halves would make the emitted
    // provenance record disagree with the gear that actually ran.
    //
    // ── WHAT ROLE THIS ARCHIVE ACTUALLY PLAYS (read before believing a claim
    //    about "the device evaluator") ─────────────────────────────────────
    //
    // Registering a .cu here compiles and archives it. It does NOT put it in
    // the binary: a static archive whose symbols nothing references outside
    // `#[cfg(test)]` is dropped whole by the linker, so `--release` shipped a
    // `nhs_rt_full` for which
    //
    //     nm -a target/release/nhs_rt_full | grep -c physgov
    //
    // returned 0, while `physical_governor.rs` documented a reachable compiled
    // evaluator. "Compiled" and "linked" and "called" are three different
    // facts and only the third one governs anything.
    //
    // As of the governor-actuation repair the roles are pinned:
    //
    //   * `physical_governor.rs::compose` (Rust) is the REFERENCE
    //     implementation on the host run path. It is what
    //     `decide_with_base_governor` evaluates and what the emitted
    //     `prismos_physical_governor_decision` rows describe.
    //   * THIS archive is the CONFORMANCE AUTHORITY, and it is reached from
    //     the RUN path, not only from tests:
    //       - `PhysicalGovernorBridge::arm` calls
    //         `verify_compiled_evaluator_abi()` once per stream before any
    //         CUDA work, which reads `sizeof`/`offsetof`/schema back out of
    //         THIS archive and refuses to arm on drift;
    //       - `decide_at_chunk_boundary` calls
    //         `prism_physical_governor_decide_host` once per governed decision
    //         on the same inputs and requires bit-equality on every ACTUATING
    //         field (authorized gear, ceiling, both emergency words,
    //         transition class). Divergence is a named terminal error.
    //   * `prism_physical_governor_decide_kernel` remains the ptxas driver for
    //     the `__device__` half. Nothing in `nhs_rt_full` LAUNCHES it: the
    //     governor decision is one decision per chunk on host-reduced
    //     observables, so there is no device-resident input for it to read.
    //     It exists so the `__device__` half is genuinely code-generated
    //     rather than merely parsed — and that is ALL it exists for.
    //
    // Verification, in the same terms the defect was found in:
    //     nm -a target/release/nhs_rt_full | grep -c physgov     # > 0
    compile_to_static_archive_ieee754(
        &nvcc,
        "src/cuda/physical_governor.cu",
        "physical_governor",
        &out_dir,
    );

    // M1.2.19.B — Channel-B GhostTileFrame capture kernel.  Single-block
    // n_clusters-thread kernel records per-frame [header (128 B) + tile
    // (1280 B)] = 1408 B records into a pinned-host, device-mapped ring.
    // Captured into the V2 graph downstream of the Adjudicator.
    compile_to_static_archive(
        &nvcc,
        "src/cuda/ghost_tile_kernel.cu",
        "ghost_tile_kernel",
        &out_dir,
    );

    // GhostPhaseLattice4D — edge adjudication kernel + host orchestrator.
    // Replaces the legacy O(N²) DBSCAN clustering of post-MD Ghost v2
    // records with a physically-constrained 4D phase lattice (spatial
    // cell × protocol phase × step bucket). Single .cu translation unit
    // pulls in `ghost_lattice_kernel_nvrtc.cu` via #include and exposes
    // `prism_ghost_phase_lattice_run` as the extern "C" host entry point.
    compile_to_static_archive(
        &nvcc,
        "src/cuda/ghost_lattice_kernel.cu",
        "ghost_lattice_kernel",
        &out_dir,
    );

    // Package 4 — GEARBOX-FACING TSH ARM.
    //
    // Same defect class, same repair as `physical_governor.cu` above.
    // `src/cuda/tsh_gearbox_arm.cuh` was header-only, `#include`d by no .cu and
    // named by no build step, so not one of its fifteen `static_assert`s had
    // ever been evaluated — while the Rust mirror carried a comment asserting
    // the layouts had been "verified by compiling the header and printing
    // sizeof", a claim nothing in this tree could reproduce. An uncompiled
    // header's assertions look like enforcement and enforce nothing.
    //
    // `tsh_gearbox_arm.cu` is that translation unit. It exposes the host-
    // callable `prism_tsh_arm_host_*` entry points (the SAME
    // `__host__ __device__` code the captured graph runs, so the Rust parity
    // test compares implementations rather than comments), the
    // `prism_tsh_arm_decide_kernel` + `prism_tsh_arm_launch_decide` pair that
    // drives the `__device__` half through ptxas, and `prism_tsh_arm_abi_sizes`,
    // which reports `sizeof`/`offsetof` as the compiler computed them for the
    // Rust `DEVICE_*_BYTES` constants to be checked against.
    //
    // IEEE-strict, like `deterministic_reduce.cu` and `physical_governor.cu`:
    // every number in this translation unit is a CONTROL DECISION input that is
    // compared against a threshold and then recorded as the reason a gear
    // changed. Fast math would contract the leaky integrator's `A*e + r*tau*(1-e)`
    // into an FMA and flush denormals — deterministic on-device, but no longer
    // re-derivable by the host oracle in `tsh_gearbox_arm.rs`, which is exactly
    // the host/device divergence the module's determinism section forbids.
    compile_to_static_archive_ieee754(
        &nvcc,
        "src/cuda/tsh_gearbox_arm.cu",
        "tsh_gearbox_arm",
        &out_dir,
    );
}

fn emit_ghost_policy_binding() {
    println!("cargo:rerun-if-env-changed={GHOST_POLICY_PATH_ENV}");
    println!("cargo:rerun-if-env-changed={GHOST_POLICY_SHA256_ENV}");
    if env::var_os("CARGO_FEATURE_V2_IGNITION").is_none() {
        return;
    }

    let policy_path = PathBuf::from(
        env::var_os(GHOST_POLICY_PATH_ENV)
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| {
                panic!("v2_ignition requires an absolute {GHOST_POLICY_PATH_ENV} compile-time path")
            }),
    );
    if !policy_path.is_absolute() {
        panic!("{GHOST_POLICY_PATH_ENV} must be absolute");
    }
    let expected_sha256 = env::var(GHOST_POLICY_SHA256_ENV)
        .unwrap_or_else(|_| panic!("v2_ignition requires {GHOST_POLICY_SHA256_ENV}"));
    require_lower_sha256(&expected_sha256, GHOST_POLICY_SHA256_ENV);
    let policy_bytes = read_regular_nofollow_stable(&policy_path).unwrap_or_else(|error| {
        panic!(
            "read compile-time Ghost topology policy {}: {error}",
            policy_path.display()
        )
    });
    let actual_sha256 = format!("{:x}", Sha256::digest(&policy_bytes));
    if actual_sha256 != expected_sha256 {
        panic!(
            "compile-time Ghost topology policy digest mismatch: expected={expected_sha256} actual={actual_sha256} path={}",
            policy_path.display()
        );
    }
    println!("cargo:rerun-if-changed={}", policy_path.display());
    println!("cargo:rustc-env={GHOST_POLICY_SHA256_ENV}={actual_sha256}");
    println!(
        "cargo:rustc-env={GHOST_POLICY_BUILD_PATH_ENV}={}",
        policy_path.display()
    );
}

fn require_lower_sha256(value: &str, label: &str) {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        panic!("{label} must be a lowercase SHA-256");
    }
}

fn read_regular_nofollow_stable(path: &Path) -> Result<Vec<u8>, String> {
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .map_err(|error| error.to_string())?;
    let before = file.metadata().map_err(|error| error.to_string())?;
    if !before.is_file() || before.len() == 0 {
        return Err("policy must be a nonempty regular non-symlink file".into());
    }
    let mut bytes = Vec::with_capacity(before.len().try_into().unwrap_or(0));
    file.read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    let after = file.metadata().map_err(|error| error.to_string())?;
    let before_identity = (
        before.dev(),
        before.ino(),
        before.len(),
        before.mtime(),
        before.mtime_nsec(),
        before.ctime(),
        before.ctime_nsec(),
    );
    let after_identity = (
        after.dev(),
        after.ino(),
        after.len(),
        after.mtime(),
        after.mtime_nsec(),
        after.ctime(),
        after.ctime_nsec(),
    );
    if before_identity != after_identity || bytes.len() as u64 != before.len() {
        return Err("policy changed during nofollow read".into());
    }
    Ok(bytes)
}

fn assert_non_governed_ptx_output(out_dir: &Path, output: &Path) {
    assert_eq!(
        PRISM_NHS_GOVERNED_PTX_CONTRIBUTION, 0,
        "prism-nhs must contribute zero governed package PTX"
    );

    let canonical_out_dir = out_dir
        .canonicalize()
        .expect("failed to canonicalize prism-nhs OUT_DIR");
    let canonical_output = output
        .canonicalize()
        .expect("failed to canonicalize prism-nhs PTX output");
    assert!(
        canonical_output.starts_with(&canonical_out_dir),
        "prism-nhs PTX output escaped OUT_DIR: {}",
        canonical_output.display()
    );

    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is required"));
    let workspace_root = manifest_dir
        .join("../..")
        .canonicalize()
        .expect("failed to canonicalize workspace root");
    let governed_target_mirror = workspace_root.join("target/ptx");
    assert!(
        !canonical_output.starts_with(&governed_target_mirror),
        "prism-nhs PTX output entered governed target mirror: {}",
        canonical_output.display()
    );
}

fn compile_kernel(nvcc: &str, source: &str, output: &PathBuf) {
    println!("cargo:info=Compiling {} -> {}", source, output.display());
    let cuda_arch = cuda_arch();
    let cuda_include = cuda_include_dir();

    let status = Command::new(nvcc)
        .arg("--ptx")
        .arg("-o")
        .arg(output)
        .arg(source)
        .arg(format!("-arch={}", cuda_arch))
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("--restrict")
        .arg(format!("-I{}", cuda_include.display()))
        .status()
        .expect("Failed to execute nvcc");

    if !status.success() {
        panic!("nvcc compilation failed for {}", source);
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is required"));
    assert_non_governed_ptx_output(&out_dir, output);
    println!("cargo:info=PTX compiled: {}", output.display());
}

/// Run the M1 lane's atomic-or geometric-tie-breaker grep gate
/// (`scripts/m1_atomic_or_grep.sh`) against the M1 CUDA sources.
/// If the gate exits non-zero, panic the build script — refusing to
/// produce a kernel that violates blueprint §M2 / M1 contract §3.2.
///
/// The gate's own diagnostics are written to stderr; this helper
/// only reports the script's pass/fail outcome and re-emits the
/// summary line into the cargo log.
fn run_atomic_or_grep_gate() {
    let workspace = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let script = workspace.join("scripts/m1_atomic_or_grep.sh");
    if !script.exists() {
        // Be tolerant: if the script is intentionally absent (e.g.
        // mid-rebase), do not block the build. The gate is a
        // belt-and-suspenders check; the unit test in
        // spike_to_cluster_4d.rs::tests is the primary enforcement.
        println!(
            "cargo:warning=M1 atomic-or grep gate script missing: {}",
            script.display()
        );
        return;
    }
    println!("cargo:rerun-if-changed={}", script.display());

    let status = Command::new(&script)
        .current_dir(&workspace)
        .status()
        .expect("Failed to invoke atomic-or grep gate script");

    if !status.success() {
        panic!(
            "M1 atomic-or geometric-tie-breaker grep gate FAILED. \
             See stderr above for offending lines. Per blueprint §M2 / \
             M1 contract §3.2, this is a LANE_BLOCKED-class violation \
             at build time."
        );
    }
}

/// Compile a `.cu` file to a host+device object file, then archive it
/// into a static library, and emit the link directives so the host
/// orchestration entry points (e.g. `prism_m1_spike_to_cluster_4d_run`)
/// resolve from Rust extern "C" call sites.
///
/// Used for `.cu` files that contain CUB device-wide algorithm calls
/// (which internally launch their own kernels from host code) — these
/// can NOT be compiled to PTX because PTX is device-only.
fn compile_to_static_archive(nvcc: &str, source: &str, lib_name: &str, out_dir: &PathBuf) {
    compile_to_static_archive_with_fp_mode(nvcc, source, lib_name, out_dir, FpMode::FastMath);
}

/// IEEE-strict variant of [`compile_to_static_archive`].
///
/// Identical in every respect except the floating-point mode: `--use_fast_math`
/// is withdrawn and the four behaviours it bundles are pinned back to their
/// IEEE-754 settings (`-fmad=false --ftz=false --prec-div=true
/// --prec-sqrt=true`).
///
/// Use this ONLY for translation units whose numerical output is a CONTROL
/// DECISION input rather than a physics result. Fast math is the right default
/// for the force kernels — it is not the right default for a value that gets
/// compared against a threshold and then recorded as the reason a gear
/// changed. Under fast math the compiler may contract `acc + x*x` into an FMA,
/// which is deterministic on-device but no longer reproducible by the host
/// reference implementation, and it flushes denormals, which silently changes
/// small-magnitude reductions.
///
/// Current members: `src/cuda/deterministic_reduce.cu` (Package 2).
fn compile_to_static_archive_ieee754(nvcc: &str, source: &str, lib_name: &str, out_dir: &PathBuf) {
    compile_to_static_archive_with_fp_mode(nvcc, source, lib_name, out_dir, FpMode::Ieee754Strict);
}

/// Floating-point contract requested from nvcc for a translation unit.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FpMode {
    /// Historical default for every physics/telemetry kernel in this crate.
    FastMath,
    /// Decision-path strictness: no FMA contraction, no denormal flush,
    /// IEEE division and square root.
    Ieee754Strict,
}

impl FpMode {
    fn nvcc_args(self) -> &'static [&'static str] {
        match self {
            FpMode::FastMath => &["--use_fast_math"],
            FpMode::Ieee754Strict => &[
                "-fmad=false",
                "--ftz=false",
                "--prec-div=true",
                "--prec-sqrt=true",
            ],
        }
    }

    fn label(self) -> &'static str {
        match self {
            FpMode::FastMath => "fast-math",
            FpMode::Ieee754Strict => "ieee754-strict",
        }
    }
}

fn compile_to_static_archive_with_fp_mode(
    nvcc: &str,
    source: &str,
    lib_name: &str,
    out_dir: &PathBuf,
    fp_mode: FpMode,
) {
    let obj_path = out_dir.join(format!("{}.o", lib_name));
    let lib_path = out_dir.join(format!("lib{}.a", lib_name));
    let cuda_arch = cuda_arch();
    let cuda_include = cuda_include_dir();

    println!(
        "cargo:info=Compiling {} -> {} (object, fp={})",
        source,
        obj_path.display(),
        fp_mode.label()
    );

    let nvcc_status = Command::new(nvcc)
        .arg("-c")
        .arg("--compile")
        .arg("-Xcompiler")
        .arg("-fPIC")
        .arg("-Xcompiler")
        .arg("-Wall")
        .arg("-O3")
        .args(fp_mode.nvcc_args())
        .arg("--restrict")
        .arg(format!("-arch={}", cuda_arch))
        .arg("-std=c++17")
        .arg("--expt-relaxed-constexpr")
        .arg(format!("-I{}", cuda_include.display()))
        // CUDA 13+ ships CUB and Thrust under the CCCL umbrella at
        // include/cccl/{cub,thrust,cuda}/. Earlier toolkits had them
        // directly under include/. The extra include path lets
        // `#include <cub/cub.cuh>` resolve under either layout.
        .arg(format!("-I{}", cuda_include.join("cccl").display()))
        .arg("-Isrc/cuda")
        .arg("-o")
        .arg(&obj_path)
        .arg(source)
        .status()
        .expect("Failed to execute nvcc for static-archive compile");

    if !nvcc_status.success() {
        panic!("nvcc compilation failed for {}", source);
    }

    println!(
        "cargo:info=Archiving {} -> {}",
        obj_path.display(),
        lib_path.display()
    );

    // Remove a stale archive before re-creating, otherwise `ar rcs`
    // would append rather than replace the object on incremental rebuilds.
    let _ = std::fs::remove_file(&lib_path);

    let ar_status = Command::new("ar")
        .arg("rcs")
        .arg(&lib_path)
        .arg(&obj_path)
        .status()
        .expect("Failed to execute ar");

    if !ar_status.success() {
        panic!("ar archive failed for {}", obj_path.display());
    }

    println!("cargo:info=Static archive: {}", lib_path.display());
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static={}", lib_name);
    // CUDA runtime + C++ stdlib are pulled in transitively by the
    // archive's cudaStreamSynchronize / cudaMemcpyAsync calls and the
    // CUB code's STL usage.
    if let Some(cuda_lib_dir) = cuda_lib_dir() {
        println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());
    } else {
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

fn cuda_arch() -> String {
    env::var("PRISM_NHS_CUDA_ARCH")
        .or_else(|_| env::var("PRISM_GPU_CUDA_ARCH"))
        .or_else(|_| env::var("PRISMCODE_CUDA_ARCH"))
        .unwrap_or_else(|_| "sm_120".to_string())
}

fn cuda_include_dir() -> PathBuf {
    if let Ok(cuda_home) = env::var("CUDA_HOME") {
        let p = PathBuf::from(cuda_home).join("include");
        if p.exists() {
            return p;
        }
    }

    for path in [
        "/usr/local/cuda/include",
        "/usr/local/cuda-13.2/include",
        "/usr/local/cuda-13/include",
        "/usr/local/cuda-12/include",
        "/usr/include",
        "/usr/lib/cuda/include",
    ] {
        let p = PathBuf::from(path);
        if p.exists() {
            return p;
        }
    }

    PathBuf::from("/usr/local/cuda/include")
}

fn cuda_lib_dir() -> Option<PathBuf> {
    if let Ok(cuda_home) = env::var("CUDA_HOME") {
        for rel in ["lib64", "targets/x86_64-linux/lib"] {
            let p = PathBuf::from(&cuda_home).join(rel);
            if p.exists() {
                return Some(p);
            }
        }
    }

    for path in [
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/usr/local/cuda-13.2/targets/x86_64-linux/lib",
        "/usr/local/cuda-13/targets/x86_64-linux/lib",
        "/usr/local/cuda-12/targets/x86_64-linux/lib",
        "/usr/lib/x86_64-linux-gnu",
    ] {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }

    None
}

fn find_nvcc() -> Option<String> {
    for key in ["PRISM_NHS_NVCC", "PRISM_GPU_NVCC", "NVCC"] {
        if let Ok(path) = env::var(key) {
            let p = PathBuf::from(&path);
            if p.exists() {
                return Some(path);
            }
        }
    }

    if let Ok(cuda_home) = env::var("CUDA_HOME") {
        let p = PathBuf::from(&cuda_home).join("bin").join("nvcc");
        if p.exists() {
            return Some(p.to_string_lossy().to_string());
        }
    }

    let common_paths = [
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-12.6/bin/nvcc",
        "/usr/local/cuda-12/bin/nvcc",
        "/opt/cuda/bin/nvcc",
    ];

    for path in &common_paths {
        if PathBuf::from(path).exists() {
            return Some(path.to_string());
        }
    }

    if Command::new("nvcc").arg("--version").output().is_ok() {
        return Some("nvcc".to_string());
    }

    None
}
