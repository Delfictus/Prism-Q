//! M1 V3 — synthetic CPU-reference verification of the M1 producer.
//!
//! Run as:
//!     cargo run --release -p prism-nhs --bin m1_v3_synthetic
//!
//! Three correctness gates required by the M1.2 V3 contract:
//!   1. Conservation of Mass: BitExact integer equality holds for 1000
//!      random synthetic inputs across varying shapes.
//!   2. SoA invariant: every EntangledManifold construction succeeds
//!      (no SoaLengthMismatch). Tested implicitly via successful apply.
//!   3. Determinism class: 20 replicate runs on identical input
//!      produce identical integer counts (BitExact). The
//!      `AtomicsAffected` class allows ordering differences within a
//!      cluster but the counts themselves must be order-independent.
//!   4. AABB exactness: per-cluster AABB matches CPU reference bit-
//!      for-bit on every synthetic case. CUB DeviceSegmentedReduce is
//!      deterministic for fixed sorted input; our LCG generator
//!      seeds the input deterministically so the CPU ref uses the
//!      same source-of-truth values.
//!
//! Hardcoded parameters: 1000 random cases with shapes drawn from
//! N_SPIKES in [50, 200] and per-axis grid dim in [3, 6]. The
//! 20-replicate determinism check uses a fixed shape (100 spikes,
//! 4×4×4 grid). The seed is hardcoded at 42 — change in source if a
//! different seed is desired.
//!
//! Exit code 0 on PASS, 1 on FAIL.

use std::process::ExitCode;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr};

use prism_nhs::spike_to_cluster_4d::{
    M1ProducerGraph, SpatialHashParams, SpikeToCluster4D, SpikeToCluster4DGpuInput,
};
use prism_nhs::transform::{AuditOutcome, AuditedTransform};

// ─────────────────────────────────────────────────────────────────────────
// Deterministic LCG so the V3 bin doesn't pull in a `rand` dependency.
// ─────────────────────────────────────────────────────────────────────────
struct Lcg {
    state: u64,
}
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg {
            state: seed.wrapping_add(0xDEAD_BEEF_F00D_CAFEu64),
        }
    }
    fn next_u32(&mut self) -> u32 {
        // Numerical Recipes-style 64-bit LCG. Statistical quality is
        // adequate for synthetic-input generation; we are not relying
        // on it for cryptographic randomness.
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 32) as u32
    }
    fn next_range(&mut self, min: u32, max_inclusive: u32) -> u32 {
        let span = max_inclusive - min + 1;
        min + (self.next_u32() % span)
    }
    fn next_f32(&mut self, min: f32, max: f32) -> f32 {
        let unit = (self.next_u32() as f32) / (u32::MAX as f32);
        min + unit * (max - min)
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Synthetic case
// ─────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct SyntheticCase {
    num_spikes: u32,
    grid_dim: [i32; 3],
    cell_size: f32,
    bbox_min: [f32; 3],
    bbox_max: [f32; 3],
    positions: Vec<f32>,
    /// CPU ground-truth per-spike cluster_id (UNCLUSTERED = u32::MAX).
    cpu_cluster_ids: Vec<u32>,
    cpu_per_cluster_count: Vec<u64>,
    cpu_background_count: u64,
    cpu_total_attributed: u64,
    /// Per-cluster AABB CPU reference. Layout: `num_clusters × 6` floats
    /// (min_x, min_y, min_z, max_x, max_y, max_z). Empty clusters are
    /// flagged with `min = +FLT_MAX`, `max = -FLT_MAX` to match CUB's
    /// SegmentedReduce identity element.
    cpu_aabb_flat: Vec<f32>,
}

fn cpu_assign(pos: [f32; 3], bbox_min: [f32; 3], cell_size: f32, grid_dim: [i32; 3]) -> u32 {
    // Mirrors the .cuh's `spike_to_cell_id` exactly.
    let dx = (pos[0] - bbox_min[0]) / cell_size;
    let dy = (pos[1] - bbox_min[1]) / cell_size;
    let dz = (pos[2] - bbox_min[2]) / cell_size;
    let cx = dx as i32;
    let cy = dy as i32;
    let cz = dz as i32;
    if cx < 0 || cx >= grid_dim[0] {
        return u32::MAX;
    }
    if cy < 0 || cy >= grid_dim[1] {
        return u32::MAX;
    }
    if cz < 0 || cz >= grid_dim[2] {
        return u32::MAX;
    }
    (cz * grid_dim[0] * grid_dim[1] + cy * grid_dim[0] + cx) as u32
}

fn make_random_case(rng: &mut Lcg) -> SyntheticCase {
    // num_spikes in [50, 200]
    let num_spikes = rng.next_range(50, 200);
    // grid dim in [3, 6] per axis
    let gx = rng.next_range(3, 6) as i32;
    let gy = rng.next_range(3, 6) as i32;
    let gz = rng.next_range(3, 6) as i32;
    let cell_size = 1.0;
    let bbox_min = [0.0f32, 0.0, 0.0];
    let bbox_max = [gx as f32, gy as f32, gz as f32];
    let num_clusters = (gx * gy * gz) as u32;

    let mut positions: Vec<f32> = Vec::with_capacity(num_spikes as usize * 3);
    let mut cpu_cluster_ids: Vec<u32> = Vec::with_capacity(num_spikes as usize);

    // Generate positions with a mix of in-bbox and out-of-bbox.
    // ~80% in-bbox, ~20% out-of-bbox (jittered well outside).
    for _ in 0..num_spikes {
        let pos = if rng.next_u32() % 5 == 0 {
            // Out-of-bbox sample.
            let outside_axis = rng.next_u32() % 3;
            let mut p = [
                rng.next_f32(0.0, bbox_max[0]),
                rng.next_f32(0.0, bbox_max[1]),
                rng.next_f32(0.0, bbox_max[2]),
            ];
            // Force one axis outside the bbox.
            if rng.next_u32() & 1 == 0 {
                p[outside_axis as usize] = -10.0 - rng.next_f32(0.0, 5.0);
            } else {
                p[outside_axis as usize] =
                    bbox_max[outside_axis as usize] + 1.0 + rng.next_f32(0.0, 5.0);
            }
            p
        } else {
            // In-bbox sample, slightly inset from edges so the
            // floor() rounding doesn't accidentally push the cell-id
            // out of range.
            [
                rng.next_f32(0.001, bbox_max[0] - 0.001),
                rng.next_f32(0.001, bbox_max[1] - 0.001),
                rng.next_f32(0.001, bbox_max[2] - 0.001),
            ]
        };
        positions.push(pos[0]);
        positions.push(pos[1]);
        positions.push(pos[2]);
        cpu_cluster_ids.push(cpu_assign(pos, bbox_min, cell_size, [gx, gy, gz]));
    }

    let mut cpu_per_cluster_count = vec![0u64; num_clusters as usize];
    let mut cpu_background_count = 0u64;
    for &cid in &cpu_cluster_ids {
        if cid == u32::MAX {
            cpu_background_count += 1;
        } else {
            cpu_per_cluster_count[cid as usize] += 1;
        }
    }
    let cpu_total_attributed: u64 = cpu_per_cluster_count.iter().sum();

    // CPU AABB. Identity = (+FLT_MAX, -FLT_MAX) for empty clusters.
    let mut cpu_aabb_flat = vec![0.0f32; num_clusters as usize * 6];
    for c in 0..num_clusters as usize {
        cpu_aabb_flat[c * 6 + 0] = f32::MAX;
        cpu_aabb_flat[c * 6 + 1] = f32::MAX;
        cpu_aabb_flat[c * 6 + 2] = f32::MAX;
        cpu_aabb_flat[c * 6 + 3] = -f32::MAX;
        cpu_aabb_flat[c * 6 + 4] = -f32::MAX;
        cpu_aabb_flat[c * 6 + 5] = -f32::MAX;
    }
    for (i, &cid) in cpu_cluster_ids.iter().enumerate() {
        if cid == u32::MAX {
            continue;
        }
        let c = cid as usize;
        let px = positions[i * 3 + 0];
        let py = positions[i * 3 + 1];
        let pz = positions[i * 3 + 2];
        cpu_aabb_flat[c * 6 + 0] = cpu_aabb_flat[c * 6 + 0].min(px);
        cpu_aabb_flat[c * 6 + 1] = cpu_aabb_flat[c * 6 + 1].min(py);
        cpu_aabb_flat[c * 6 + 2] = cpu_aabb_flat[c * 6 + 2].min(pz);
        cpu_aabb_flat[c * 6 + 3] = cpu_aabb_flat[c * 6 + 3].max(px);
        cpu_aabb_flat[c * 6 + 4] = cpu_aabb_flat[c * 6 + 4].max(py);
        cpu_aabb_flat[c * 6 + 5] = cpu_aabb_flat[c * 6 + 5].max(pz);
    }

    SyntheticCase {
        num_spikes,
        grid_dim: [gx, gy, gz],
        cell_size,
        bbox_min,
        bbox_max,
        positions,
        cpu_cluster_ids,
        cpu_per_cluster_count,
        cpu_background_count,
        cpu_total_attributed,
        cpu_aabb_flat,
    }
}

// ─────────────────────────────────────────────────────────────────────────
// One-pass M1 producer invocation through the AuditedTransform path.
// ─────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct M1Output {
    cluster_ids: Vec<u32>,
    per_cluster_count: Vec<u64>,
    total_attributed: u64,
    background_count: u64,
    per_cluster_aabb: Vec<f32>,
}

fn run_one(
    stream: &Arc<CudaStream>,
    graph_cache: &mut M1ProducerGraph,
    case: &SyntheticCase,
) -> Result<M1Output, String> {
    let num_clusters = (case.grid_dim[0] * case.grid_dim[1] * case.grid_dim[2]) as u32;

    // Allocations.
    let mut d_positions: CudaSlice<f32> = stream
        .alloc_zeros(case.positions.len())
        .map_err(|e| format!("alloc d_positions: {:?}", e))?;
    stream
        .memcpy_htod(&case.positions, &mut d_positions)
        .map_err(|e| format!("htod positions: {:?}", e))?;

    let d_cluster_ids: CudaSlice<u32> = stream
        .alloc_zeros(case.num_spikes as usize)
        .map_err(|e| format!("alloc d_cluster_ids: {:?}", e))?;
    let d_per_cluster_count: CudaSlice<u64> = stream
        .alloc_zeros(num_clusters as usize)
        .map_err(|e| format!("alloc d_per_cluster_count: {:?}", e))?;
    let d_total_attributed: CudaSlice<u64> = stream
        .alloc_zeros(1)
        .map_err(|e| format!("alloc d_total_attributed: {:?}", e))?;
    let d_background_count: CudaSlice<u64> = stream
        .alloc_zeros(1)
        .map_err(|e| format!("alloc d_background_count: {:?}", e))?;
    let d_per_cluster_aabb: CudaSlice<f32> = stream
        .alloc_zeros(num_clusters as usize * 6)
        .map_err(|e| format!("alloc d_per_cluster_aabb: {:?}", e))?;

    let params = SpatialHashParams {
        bbox_min: case.bbox_min,
        bbox_max: case.bbox_max,
        cell_size: case.cell_size,
        grid_dim: case.grid_dim,
        num_cells: num_clusters,
    };

    let input = SpikeToCluster4DGpuInput {
        stream,
        graph_cache,
        d_spike_positions: &d_positions,
        num_spikes: case.num_spikes,
        params,
        frame: 0,
        d_cluster_id_per_spike: &d_cluster_ids,
        d_per_cluster_count: &d_per_cluster_count,
        d_total_attributed: &d_total_attributed,
        d_background_count: &d_background_count,
        d_per_cluster_aabb: &d_per_cluster_aabb,
        num_clusters,
    };

    let producer = SpikeToCluster4D::new();
    let outcome = producer.apply(input);

    match outcome {
        AuditOutcome::Accepted { .. } => { /* fall through to dtoh */ }
        AuditOutcome::Quarantined { violations, .. } => {
            return Err(format!("Quarantined: {:?}", violations));
        }
        AuditOutcome::Aborted { violations, .. } => {
            return Err(format!("Aborted: {:?}", violations));
        }
    }

    let mut cluster_ids = vec![0u32; case.num_spikes as usize];
    stream
        .memcpy_dtoh(&d_cluster_ids, &mut cluster_ids)
        .map_err(|e| format!("dtoh cid: {:?}", e))?;
    let mut count = vec![0u64; num_clusters as usize];
    stream
        .memcpy_dtoh(&d_per_cluster_count, &mut count)
        .map_err(|e| format!("dtoh count: {:?}", e))?;
    let mut total = vec![0u64; 1];
    stream
        .memcpy_dtoh(&d_total_attributed, &mut total)
        .map_err(|e| format!("dtoh total: {:?}", e))?;
    let mut bg = vec![0u64; 1];
    stream
        .memcpy_dtoh(&d_background_count, &mut bg)
        .map_err(|e| format!("dtoh bg: {:?}", e))?;
    let mut aabb = vec![0.0f32; num_clusters as usize * 6];
    stream
        .memcpy_dtoh(&d_per_cluster_aabb, &mut aabb)
        .map_err(|e| format!("dtoh aabb: {:?}", e))?;

    Ok(M1Output {
        cluster_ids,
        per_cluster_count: count,
        total_attributed: total[0],
        background_count: bg[0],
        per_cluster_aabb: aabb,
    })
}

// ─────────────────────────────────────────────────────────────────────────
// Verifier — compares M1 output against the case's CPU ground truth.
// ─────────────────────────────────────────────────────────────────────────

fn verify_against_case(out: &M1Output, case: &SyntheticCase) -> Result<(), String> {
    if out.cluster_ids != case.cpu_cluster_ids {
        return Err(format!(
            "cluster_id_per_spike mismatch (first 8 expected={:?} got={:?})",
            &case.cpu_cluster_ids[..8.min(case.cpu_cluster_ids.len())],
            &out.cluster_ids[..8.min(out.cluster_ids.len())]
        ));
    }
    if out.per_cluster_count != case.cpu_per_cluster_count {
        return Err("per_cluster_count BitExact mismatch".into());
    }
    if out.background_count != case.cpu_background_count {
        return Err(format!(
            "background_count: expected {}, got {}",
            case.cpu_background_count, out.background_count
        ));
    }
    if out.total_attributed != case.cpu_total_attributed {
        return Err(format!(
            "total_attributed: expected {}, got {}",
            case.cpu_total_attributed, out.total_attributed
        ));
    }
    // Conservation.
    if out.total_attributed + out.background_count != case.num_spikes as u64 {
        return Err(format!(
            "Conservation-of-Mass violation: {} + {} != {}",
            out.total_attributed, out.background_count, case.num_spikes
        ));
    }
    // AABB BitExact match per cluster.
    for c in 0..case.cpu_per_cluster_count.len() {
        for k in 0..6 {
            let exp = case.cpu_aabb_flat[c * 6 + k];
            let got = out.per_cluster_aabb[c * 6 + k];
            if exp.to_bits() != got.to_bits() {
                return Err(format!(
                    "AABB[c={} k={}] mismatch: expected {} ({:#x}) got {} ({:#x})",
                    c,
                    k,
                    exp,
                    exp.to_bits(),
                    got,
                    got.to_bits()
                ));
            }
        }
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────

fn main() -> ExitCode {
    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("V3 FAIL: cannot acquire CUDA context: {:?}", e);
            return ExitCode::from(1);
        }
    };
    let stream = match ctx.new_stream() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("V3 FAIL: cannot create stream: {:?}", e);
            return ExitCode::from(1);
        }
    };

    let mut rng = Lcg::new(42);

    // ── Phase 1: 1000 random synthetic inputs ────────────────────────
    let mut graph_cache = M1ProducerGraph::new();
    let mut shape_changes = 0u32;
    let mut last_shape: Option<(u32, u32)> = None;

    for i in 0..1000 {
        let case = make_random_case(&mut rng);
        let num_clusters = (case.grid_dim[0] * case.grid_dim[1] * case.grid_dim[2]) as u32;
        let shape = (case.num_spikes, num_clusters);
        if last_shape != Some(shape) {
            shape_changes += 1;
            last_shape = Some(shape);
        }
        // Each random case allocates fresh device buffers; the captured
        // graph from a previous iteration would replay with stale
        // (freed) pointers if the shape collides. Invalidate before
        // every iteration so we always re-capture against the current
        // buffers.
        graph_cache.invalidate();

        let out = match run_one(&stream, &mut graph_cache, &case) {
            Ok(o) => o,
            Err(e) => {
                eprintln!(
                    "V3 FAIL at random case {} (n_spikes={}, grid={:?}): {}",
                    i, case.num_spikes, case.grid_dim, e
                );
                return ExitCode::from(1);
            }
        };
        if let Err(e) = verify_against_case(&out, &case) {
            eprintln!(
                "V3 FAIL at random case {} (n_spikes={}, grid={:?}): {}",
                i, case.num_spikes, case.grid_dim, e
            );
            return ExitCode::from(1);
        }
    }
    println!(
        "V3 PHASE 1 PASS: 1000 random cases, all conservation BitExact, all AABBs CPU-equal, {} shape changes",
        shape_changes
    );

    // ── Phase 2: 20-replicate determinism on a fixed input ───────────
    let mut fixed_rng = Lcg::new(12_345);
    let fixed_case = make_random_case(&mut fixed_rng);
    let mut graph_cache_replicate = M1ProducerGraph::new();

    let mut first_output: Option<M1Output> = None;
    for r in 0..20 {
        // Each replicate allocates fresh buffers (run_one's
        // alloc_zeros) — invalidate so the cache re-captures rather
        // than replaying a graph that referenced now-freed pointers.
        // Determinism still holds: fresh capture + launch on the same
        // input yields BitExact identical outputs to any other capture
        // on the same input.
        graph_cache_replicate.invalidate();

        let out = match run_one(&stream, &mut graph_cache_replicate, &fixed_case) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("V3 FAIL at replicate {}: {}", r, e);
                return ExitCode::from(1);
            }
        };
        match &first_output {
            None => first_output = Some(out),
            Some(reference) => {
                if out.per_cluster_count != reference.per_cluster_count {
                    eprintln!(
                        "V3 FAIL: replicate {} per_cluster_count differs from replicate 0 (BitExact determinism violation)",
                        r
                    );
                    return ExitCode::from(1);
                }
                if out.total_attributed != reference.total_attributed
                    || out.background_count != reference.background_count
                {
                    eprintln!(
                        "V3 FAIL: replicate {} integer scalars differ from replicate 0",
                        r
                    );
                    return ExitCode::from(1);
                }
                // AABB also must match (deterministic for fixed input
                // even under AtomicsAffected, since the sort + segmented
                // reduce produce a bit-stable result for a fixed
                // permutation of cluster_ids).
                if out.per_cluster_aabb != reference.per_cluster_aabb {
                    eprintln!(
                        "V3 FAIL: replicate {} per_cluster_aabb differs from replicate 0",
                        r
                    );
                    return ExitCode::from(1);
                }
            }
        }
    }
    println!(
        "V3 PHASE 2 PASS: 20 replicates on fixed input ({} spikes, grid={:?}) — all integer counts BitExact identical, all AABBs identical",
        fixed_case.num_spikes, fixed_case.grid_dim
    );

    println!("V3 PASS");
    ExitCode::from(0)
}
