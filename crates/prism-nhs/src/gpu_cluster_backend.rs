//! GPU Spatial-Hash DBSCAN Backend (Post-MD Clustering)
//!
//! Replacement for the CPU `fallback_grid_cluster` path that wedges on
//! multi-million spike populations. Implements Euclidean connected
//! components at a fixed `epsilon` via a 4-kernel GPU pipeline:
//!
//!   1. assign_cells  — bucket each point into a 3D cell id
//!   2. build_offsets — (CPU) sort indices by cell_id and build start/end tables
//!   3. find_union    — one thread per point; iterate 27 neighbor cells; atomic union-find
//!   4. path_compress — pointer-doubling path compression to canonical parent
//!
//! Cell size equals `epsilon`, so the 27-cell neighborhood exactly covers
//! all points within `epsilon` of the query. The distance gate inside the
//! kernel enforces `dist_sq <= epsilon_sq`.
//!
//! Semantics are byte-equivalent to `fallback_grid_cluster` (same
//! connected-components predicate at the same epsilon) modulo
//! union-find cluster-ID renumbering.
//!
//! # Scale target
//!
//! N = 11.3 M points on an RTX 5080 (SM120, 16 GB):
//!   - CPU sort: ~400 ms
//!   - GPU kernels: ~30 s
//!   - Full round trip: under 60 s (vs unbounded for the old path)
//!
//! # Kernel source
//!
//! The CUDA source is embedded as a `&'static str` and compiled at
//! runtime via `cudarc::nvrtc`. No new PTX files are added to the build
//! pipeline.
//!
//! # Not covered in this backend
//!
//! - Device-side radix sort (host sort is fast enough at 11 M scale).
//! - Persistent module caching across engine instances (cheap enough to
//!   recompile once per engine).
//! - Multi-device dispatch.

#![cfg(feature = "gpu")]

use anyhow::{anyhow, Context, Result};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;

use crate::rt_clustering::RtClusteringResult;

/// Process-global clustering backend selection. Set once from `nhs_rt_full::main`
/// after the `--clustering-backend` flag is parsed. Read by
/// `PersistentNhsEngine::cluster_spikes` on every call. Using an atomic u8 instead
/// of a `OnceLock<Enum>` keeps the read on the hot path a single unlocked load.
///
/// Encoding:
///   0 = uninitialized       -> treated as Auto (equivalent to 1)
///   1 = Auto                -> resolves to GhostPhaseLattice4D when v2 Ghost
///                              records are present, else GpuSpatialHash.
///   2 = GpuSpatialHash      (explicit; legacy O(N²)-prone DBSCAN)
///   3 = RtOptix             (explicit; errors on SM120)
///   4 = GridDebug           (explicit; debug only)
///   5 = Lbvh                (explicit; not yet implemented — errors loudly)
///   6 = GhostPhaseLattice4D (explicit; physically-constrained 4D edge
///                              adjudication on Ghost v2 records — directive
///                              MASTER ARCHITECTURAL DIRECTIVE V5).
pub static SELECTED_BACKEND: AtomicU8 = AtomicU8::new(0);

pub const BACKEND_UNINIT: u8 = 0;
pub const BACKEND_AUTO: u8 = 1;
pub const BACKEND_GPU_HASH: u8 = 2;
pub const BACKEND_RT_OPTIX: u8 = 3;
pub const BACKEND_GRID_DEBUG: u8 = 4;
pub const BACKEND_LBVH: u8 = 5;
pub const BACKEND_GHOST_LATTICE_4D: u8 = 6;

/// Human-readable name of the currently-selected backend.
pub fn backend_name(b: u8) -> &'static str {
    match b {
        BACKEND_UNINIT | BACKEND_AUTO => "auto",
        BACKEND_GPU_HASH => "gpu-hash",
        BACKEND_RT_OPTIX => "optix",
        BACKEND_GRID_DEBUG => "grid-debug",
        BACKEND_LBVH => "lbvh",
        BACKEND_GHOST_LATTICE_4D => "ghost-phase-lattice-4d",
        _ => "unknown",
    }
}

/// Parse CLI flag value into a backend code. Returns None on unknown.
pub fn parse_backend_str(s: &str) -> Option<u8> {
    match s {
        "auto" => Some(BACKEND_AUTO),
        "gpu-hash" => Some(BACKEND_GPU_HASH),
        "optix" => Some(BACKEND_RT_OPTIX),
        "grid" => Some(BACKEND_GRID_DEBUG),
        "lbvh" => Some(BACKEND_LBVH),
        "ghost-phase-lattice-4d" | "ghost-lattice" | "lattice-4d" => Some(BACKEND_GHOST_LATTICE_4D),
        _ => None,
    }
}

/// CUDA source for the spatial-hash CCL pipeline. Compiled once via nvrtc.
const CUDA_SRC: &str = r#"
extern "C" __global__ void assign_cells(
    const float* __restrict__ positions,
    unsigned int* __restrict__ cell_ids,
    const int n,
    const float cell_size,
    const float origin_x,
    const float origin_y,
    const float origin_z,
    const int grid_x,
    const int grid_y
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float px = positions[3*i + 0] - origin_x;
    float py = positions[3*i + 1] - origin_y;
    float pz = positions[3*i + 2] - origin_z;
    int cx = (int)floorf(px / cell_size);
    int cy = (int)floorf(py / cell_size);
    int cz = (int)floorf(pz / cell_size);
    if (cx < 0) cx = 0;
    if (cy < 0) cy = 0;
    if (cz < 0) cz = 0;
    // caller clamps grid bounds; saturate positive side here
    // (grid_x/grid_y/grid_z are sized to bbox / cell_size + 1)
    cell_ids[i] = (unsigned int)(cx + cy * grid_x + cz * grid_x * grid_y);
}

extern "C" __global__ void init_parent(
    int* __restrict__ parent,
    const int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    parent[i] = i;
}

// ECL-CC / Jaiganesh-Burtscher union-find. Non-retrying, monotone.
//
//   * `find_root` walks toward the root. Because every `unite_pair` attaches
//     a larger-index root under a strictly smaller-index root, the parent
//     chain is monotonically decreasing and therefore finite. No cycles
//     possible. The only intra-walk write is `parent[prev] = next` which is
//     a benign race — any concurrent write from another thread also places
//     a valid ancestor there, so the value is always a monotone improvement
//     toward the root.
//
//   * `unite_pair` does NOT retry by restarting find. On CAS failure it
//     takes the returned parent (which is strictly closer to the true root
//     than the one we just tried) and continues from there. Each iteration
//     advances up the tree by at least one level, so total iterations are
//     bounded by tree depth (~log N under path compression). This is the
//     structural difference from the prior retry-loop rep_union, which
//     restarted `rep_find` from scratch on every CAS failure and combined
//     with racy path-halving to produce an unbounded spin on dense clusters.
__device__ __forceinline__ int find_root(int* parent, int i) {
    int curr = parent[i];
    if (curr != i) {
        int prev = i;
        int next;
        while (curr > (next = parent[curr])) {
            parent[prev] = next;   // benign race; monotone toward root
            prev = curr;
            curr = next;
        }
    }
    return curr;
}

__device__ __forceinline__ void unite_pair(int* parent, int u, int v) {
    bool repeat;
    do {
        repeat = false;
        u = find_root(parent, u);
        v = find_root(parent, v);
        if (u < v) {
            int old = atomicCAS(&parent[v], v, u);
            if (old != v) {
                // v's parent was already remapped to `old` by a concurrent
                // thread. `old` is strictly closer to the root than v.
                // Advance v to old and retry — bounded by tree depth.
                v = old;
                repeat = true;
            }
        } else if (u > v) {
            int old = atomicCAS(&parent[u], u, v);
            if (old != u) {
                u = old;
                repeat = true;
            }
        }
        // u == v: already in same component, nothing to do.
    } while (repeat);
}

// Legacy names kept for call-site stability inside this kernel source.
#define rep_find(p, x) find_root((p), (x))
#define rep_union(p, a, b) unite_pair((p), (a), (b))

extern "C" __global__ void find_neighbors_union(
    const float* __restrict__ positions,
    const int* __restrict__ sorted_indices,   // length n, points sorted by cell_id
    const unsigned int* __restrict__ cell_start,  // length n_cells
    const unsigned int* __restrict__ cell_end,    // length n_cells
    int* __restrict__ parent,                 // length n
    const int n,
    const float cell_size,
    const float epsilon_sq,
    const float origin_x,
    const float origin_y,
    const float origin_z,
    const int grid_x,
    const int grid_y,
    const int grid_z
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;

    // Query point at ORIGINAL index i = sorted_indices[t] — but actually,
    // because we parallelise over original indices, use t as the query's
    // original index directly. The sorted_indices array is used only inside
    // the neighbor loop to fetch other points' original ids.
    int i = t;

    float pxi = positions[3*i + 0];
    float pyi = positions[3*i + 1];
    float pzi = positions[3*i + 2];

    int cxi = (int)floorf((pxi - origin_x) / cell_size);
    int cyi = (int)floorf((pyi - origin_y) / cell_size);
    int czi = (int)floorf((pzi - origin_z) / cell_size);

    for (int dz = -1; dz <= 1; ++dz) {
        int cz = czi + dz;
        if (cz < 0 || cz >= grid_z) continue;
        for (int dy = -1; dy <= 1; ++dy) {
            int cy = cyi + dy;
            if (cy < 0 || cy >= grid_y) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                int cx = cxi + dx;
                if (cx < 0 || cx >= grid_x) continue;
                unsigned int cid = (unsigned int)(cx + cy * grid_x + cz * grid_x * grid_y);
                unsigned int s = cell_start[cid];
                unsigned int e = cell_end[cid];
                for (unsigned int k = s; k < e; ++k) {
                    int j = sorted_indices[k];
                    if (j <= i) continue; // each unordered pair visited once
                    float dxp = positions[3*j + 0] - pxi;
                    float dyp = positions[3*j + 1] - pyi;
                    float dzp = positions[3*j + 2] - pzi;
                    float d2 = dxp*dxp + dyp*dyp + dzp*dzp;
                    if (d2 <= epsilon_sq) {
                        rep_union(parent, i, j);
                    }
                }
            }
        }
    }
}

extern "C" __global__ void path_compress(
    int* __restrict__ parent,
    const int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int r = rep_find(parent, i);
    parent[i] = r;
}
"#;

/// Handle to a compiled GPU CCL backend. Built once per `PersistentNhsEngine`,
/// reused across calls.
pub struct GpuSpatialHashBackend {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    f_assign_cells: CudaFunction,
    f_init_parent: CudaFunction,
    f_find_union: CudaFunction,
    f_path_compress: CudaFunction,
}

impl GpuSpatialHashBackend {
    /// Compile the kernel source via nvrtc and bind functions.
    pub fn new(context: Arc<CudaContext>, stream: Arc<CudaStream>) -> Result<Self> {
        log::info!(
            "  [GPU-HASH] nvrtc-compiling spatial-hash CCL kernels ({} bytes of source)",
            CUDA_SRC.len()
        );
        let t0 = std::time::Instant::now();

        let ptx = compile_ptx(CUDA_SRC).map_err(|e| {
            anyhow!(
                "nvrtc compile of gpu_cluster_backend kernels failed: {:?}",
                e
            )
        })?;
        let module = context
            .load_module(ptx)
            .context("load compiled PTX into CUDA module")?;

        let f_assign_cells = module
            .load_function("assign_cells")
            .context("load assign_cells kernel")?;
        let f_init_parent = module
            .load_function("init_parent")
            .context("load init_parent kernel")?;
        let f_find_union = module
            .load_function("find_neighbors_union")
            .context("load find_neighbors_union kernel")?;
        let f_path_compress = module
            .load_function("path_compress")
            .context("load path_compress kernel")?;

        log::info!(
            "  [GPU-HASH] kernels compiled + loaded in {} ms",
            t0.elapsed().as_millis()
        );

        Ok(Self {
            context,
            stream,
            _module: module,
            f_assign_cells,
            f_init_parent,
            f_find_union,
            f_path_compress,
        })
    }

    /// Run Euclidean connected-components clustering at `epsilon` on a
    /// flattened `[x,y,z,x,y,z,...]` position array.
    pub fn cluster(&mut self, positions: &[f32], epsilon: f32) -> Result<RtClusteringResult> {
        let n = positions.len() / 3;
        if n == 0 {
            return Ok(RtClusteringResult {
                cluster_ids: Vec::new(),
                num_clusters: 0,
                total_neighbors: 0,
                gpu_time_ms: 0.0,
            });
        }

        let epsilon_sq = epsilon * epsilon;
        let cell_size = epsilon.max(1e-3);
        let t_total = std::time::Instant::now();

        // ── Step 0. Compute bbox on host ────────────────────────────────
        let t0 = std::time::Instant::now();
        let (mut min_x, mut min_y, mut min_z) = (f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let (mut max_x, mut max_y, mut max_z) =
            (f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
        for i in 0..n {
            let x = positions[3 * i];
            let y = positions[3 * i + 1];
            let z = positions[3 * i + 2];
            if x < min_x {
                min_x = x;
            }
            if x > max_x {
                max_x = x;
            }
            if y < min_y {
                min_y = y;
            }
            if y > max_y {
                max_y = y;
            }
            if z < min_z {
                min_z = z;
            }
            if z > max_z {
                max_z = z;
            }
        }
        // Pad so saturate-to-zero on negative side is safe and the upper grid dim is sufficient.
        let origin_x = min_x - cell_size;
        let origin_y = min_y - cell_size;
        let origin_z = min_z - cell_size;
        let grid_x = (((max_x - origin_x) / cell_size).ceil() as i32 + 1).max(1);
        let grid_y = (((max_y - origin_y) / cell_size).ceil() as i32 + 1).max(1);
        let grid_z = (((max_z - origin_z) / cell_size).ceil() as i32 + 1).max(1);
        let n_cells = (grid_x as usize) * (grid_y as usize) * (grid_z as usize);
        log::info!("  [GPU-HASH] bbox scan: {} pts, grid={}×{}×{} ({} cells), origin=({:.2},{:.2},{:.2}) cell={:.2}Å | {} ms",
            n, grid_x, grid_y, grid_z, n_cells, origin_x, origin_y, origin_z, cell_size,
            t0.elapsed().as_millis());

        // ── Step 1. Upload positions + allocate cell_ids / parent ─────
        let t1 = std::time::Instant::now();
        let d_positions = self
            .stream
            .memcpy_stod(positions)
            .context("upload positions to device")?;
        let mut d_cell_ids = unsafe { self.stream.alloc::<u32>(n)? };
        let mut d_parent = unsafe { self.stream.alloc::<i32>(n)? };
        log::info!(
            "  [GPU-HASH] device alloc+upload: {} ms",
            t1.elapsed().as_millis()
        );

        // ── Step 2. assign_cells kernel ──────────────────────────────
        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };

        let t2 = std::time::Instant::now();
        unsafe {
            self.stream
                .launch_builder(&self.f_assign_cells)
                .arg(&d_positions)
                .arg(&mut d_cell_ids)
                .arg(&(n as i32))
                .arg(&cell_size)
                .arg(&origin_x)
                .arg(&origin_y)
                .arg(&origin_z)
                .arg(&grid_x)
                .arg(&grid_y)
                .launch(cfg.clone())
        }
        .context("launch assign_cells")?;

        // init_parent[i] = i
        unsafe {
            self.stream
                .launch_builder(&self.f_init_parent)
                .arg(&mut d_parent)
                .arg(&(n as i32))
                .launch(cfg.clone())
        }
        .context("launch init_parent")?;

        self.stream
            .synchronize()
            .context("sync after assign_cells")?;
        log::info!(
            "  [GPU-HASH] assign_cells + init_parent: {} ms",
            t2.elapsed().as_millis()
        );

        // ── Step 3. Host: sort indices by cell_id, build offsets ─────
        let t3 = std::time::Instant::now();
        let mut cell_ids_host: Vec<u32> = vec![0; n];
        self.stream
            .memcpy_dtoh(&d_cell_ids, &mut cell_ids_host)
            .context("download cell_ids")?;

        // Sort indices by cell_id (stable sort — deterministic tie-break by index).
        let mut sorted_indices: Vec<i32> = (0..n as i32).collect();
        sorted_indices.sort_by_key(|&i| cell_ids_host[i as usize]);

        // Build cell_start / cell_end offset arrays over the sorted order.
        let mut cell_start: Vec<u32> = vec![0u32; n_cells];
        let mut cell_end: Vec<u32> = vec![0u32; n_cells];
        if !sorted_indices.is_empty() {
            let mut cur = cell_ids_host[sorted_indices[0] as usize] as usize;
            cell_start[cur] = 0;
            for (k, &idx) in sorted_indices.iter().enumerate().skip(1) {
                let cid = cell_ids_host[idx as usize] as usize;
                if cid != cur {
                    cell_end[cur] = k as u32;
                    cell_start[cid] = k as u32;
                    cur = cid;
                }
            }
            cell_end[cur] = sorted_indices.len() as u32;
        }
        log::info!(
            "  [GPU-HASH] sort + offsets (host): {} ms (n_sorted={})",
            t3.elapsed().as_millis(),
            sorted_indices.len()
        );

        // ── Step 4. Upload sorted_indices + offsets ──────────────────
        let t4 = std::time::Instant::now();
        let d_sorted_indices = self
            .stream
            .memcpy_stod(&sorted_indices)
            .context("upload sorted_indices")?;
        let d_cell_start = self
            .stream
            .memcpy_stod(&cell_start)
            .context("upload cell_start")?;
        let d_cell_end = self
            .stream
            .memcpy_stod(&cell_end)
            .context("upload cell_end")?;
        log::info!(
            "  [GPU-HASH] table upload: {} ms (sorted={} KB, cell tables={} KB)",
            t4.elapsed().as_millis(),
            (sorted_indices.len() * 4) / 1024,
            (2 * cell_start.len() * 4) / 1024
        );

        // ── Step 5. find_neighbors_union ─────────────────────────────
        let t5 = std::time::Instant::now();
        unsafe {
            self.stream
                .launch_builder(&self.f_find_union)
                .arg(&d_positions)
                .arg(&d_sorted_indices)
                .arg(&d_cell_start)
                .arg(&d_cell_end)
                .arg(&mut d_parent)
                .arg(&(n as i32))
                .arg(&cell_size)
                .arg(&epsilon_sq)
                .arg(&origin_x)
                .arg(&origin_y)
                .arg(&origin_z)
                .arg(&grid_x)
                .arg(&grid_y)
                .arg(&grid_z)
                .launch(cfg.clone())
        }
        .context("launch find_neighbors_union")?;
        self.stream
            .synchronize()
            .context("sync after find_neighbors_union")?;
        log::info!(
            "  [GPU-HASH] find_neighbors_union: {} ms",
            t5.elapsed().as_millis()
        );

        // ── Step 6. path_compress until stable (one pass is sufficient
        //            for path-halving find since union-find depth ≤ log N
        //            and the find inside union already halves paths) ───
        let t6 = std::time::Instant::now();
        unsafe {
            self.stream
                .launch_builder(&self.f_path_compress)
                .arg(&mut d_parent)
                .arg(&(n as i32))
                .launch(cfg.clone())
        }
        .context("launch path_compress")?;
        self.stream
            .synchronize()
            .context("sync after path_compress")?;
        log::info!(
            "  [GPU-HASH] path_compress: {} ms",
            t6.elapsed().as_millis()
        );

        // ── Step 7. Download parent, renumber cluster ids ────────────
        let t7 = std::time::Instant::now();
        let mut parent_host: Vec<i32> = vec![0; n];
        self.stream
            .memcpy_dtoh(&d_parent, &mut parent_host)
            .context("download parent")?;

        // Renumber roots to dense 0..K cluster ids. Keep i32 for contract parity.
        use std::collections::HashMap;
        let mut root_to_id: HashMap<i32, i32> = HashMap::new();
        let mut cluster_ids: Vec<i32> = Vec::with_capacity(n);
        for i in 0..n {
            let root = parent_host[i];
            let next_id = root_to_id.len() as i32;
            let id = *root_to_id.entry(root).or_insert(next_id);
            cluster_ids.push(id);
        }
        let num_clusters = root_to_id.len();
        log::info!(
            "  [GPU-HASH] renumber: {} ms ({} clusters)",
            t7.elapsed().as_millis(),
            num_clusters
        );

        let gpu_time_ms = t_total.elapsed().as_secs_f64() * 1000.0;

        // `total_neighbors` is only used for debug reporting downstream and costs
        // an extra pass to count on device; leave it at 0 to avoid perf cost.
        Ok(RtClusteringResult {
            cluster_ids,
            num_clusters,
            total_neighbors: 0,
            gpu_time_ms,
        })
    }
}

/// Resolve the selected backend code to a concrete backend-to-use.
///
/// Previous revision gated AUTO on an `is_sm120_or_newer` hint derived
/// from `rt_utils::is_optix_available()`. That hint returned `true`
/// whenever the device had RT cores (SM >= 7.5), which is true on SM120
/// Blackwell — so AUTO mis-resolved to OPTIX on exactly the hardware
/// where IMMUTABLE_RULE #9 disables OptiX. Confirmed failure signature
/// in the canonical 4lpk run on 2026-04-22: 8×
/// `POST_MD_CLUSTER_BACKEND_SELECTED backend=optix` followed by
/// `RT clustering failed (OptiX is disabled on this device)` and a
/// silent LIGSITE geometric-pocket fallback downstream.
///
/// Per IMMUTABLE_RULE #9 in CLAUDE.md, OptiX is disabled unconditionally
/// in this codebase (`ensure_rt_pipeline` short-circuits to `Ok(false)`
/// at `persistent_engine.rs`). Consequently AUTO always resolves to
/// `BACKEND_GPU_HASH` here. The `BACKEND_RT_OPTIX` code remains reachable
/// ONLY via explicit `--clustering-backend=optix`, where the error is
/// intended and loud.
///
/// The `is_sm120_or_newer` parameter is retained for call-site stability
/// and future resurrection of an OptiX path; it is ignored today.
pub fn resolve_auto(selected: u8, _is_sm120_or_newer: bool) -> u8 {
    if selected == BACKEND_UNINIT || selected == BACKEND_AUTO {
        BACKEND_GPU_HASH
    } else {
        selected
    }
}

/// Read the current process-global selection; returns 0 (uninit/auto) if never set.
pub fn current_selection() -> u8 {
    SELECTED_BACKEND.load(Ordering::Relaxed)
}

/// Set the process-global selection. Called by `nhs_rt_full::main`.
pub fn set_selection(code: u8) {
    SELECTED_BACKEND.store(code, Ordering::Relaxed);
}
