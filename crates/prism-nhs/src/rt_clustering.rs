//! RT-Core Accelerated Spatial Clustering v2
//!
//! Single-pass neighbor finding using __anyhit__ shaders.
//! BVH caching across epsilon scales for hierarchical clustering.
//!
//! Pipeline:
//! 1. Build BVH from event positions (once per position set)
//! 2. Single OptiX launch: raygen + anyhit finds all neighbors
//! 3. GPU Union-Find on flat neighbor buffer
//! 4. Path compression + cluster ID propagation

use anyhow::{Context, Result};
use std::path::Path;
use std::sync::Arc;

#[cfg(feature = "gpu")]
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig,
    PushKernelArg,
};
#[cfg(feature = "gpu")]
use cudarc::nvrtc::Ptx;

#[cfg(feature = "gpu")]
use optix_sys::CUstream;
#[cfg(feature = "gpu")]
use prism_optix::{
    aligned_sbt_record_size, AccelStructure, BvhBuildFlags, Module, ModuleCompileOptions,
    OptixContext, Pipeline, PipelineCompileOptions, PipelineLinkOptions, ProgramGroup,
    ShaderBindingTable, SBT_RECORD_HEADER_SIZE,
};

/// Maximum neighbors per event in fixed-size buffer
const MAX_NEIGHBORS_PER_EVENT: u32 = 128;

/// RT clustering configuration
#[derive(Debug, Clone)]
pub struct RtClusteringConfig {
    /// Neighborhood radius (Å)
    pub epsilon: f32,
    /// Minimum points to form a core point
    pub min_points: u32,
    /// Minimum cluster size to keep
    pub min_cluster_size: u32,
    /// Rays per event for neighbor finding (16 sufficient with anyhit)
    pub rays_per_event: u32,
}

impl Default for RtClusteringConfig {
    fn default() -> Self {
        Self {
            epsilon: 5.0,
            min_points: 3,
            min_cluster_size: 100,
            rays_per_event: 16, // v2: reduced from 64 (anyhit finds multiple per ray)
        }
    }
}

/// Launch parameters for RT clustering kernel v2 (72 bytes)
/// Must match RtClusteringParams in rt_clustering.cu exactly
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RtClusteringParams {
    pub traversable: u64,     // 8 @ 0:  OptixTraversableHandle
    pub event_positions: u64, // 8 @ 8:  float3* device pointer
    pub num_events: u32,      // 4 @ 16
    pub epsilon_sq: f32,      // 4 @ 20: current epsilon² for distance filter
    pub ray_tmax: f32,        // 4 @ 24: max epsilon (ray extent)
    pub rays_per_event: u32,  // 4 @ 28
    pub max_neighbors: u32,   // 4 @ 32: fixed buffer size per event
    pub _pad: u32,            // 4 @ 36: alignment padding
    pub neighbor_list: u64,   // 8 @ 40: [num_events * max_neighbors]
    pub neighbor_count: u64,  // 8 @ 48: [num_events]
    pub parent: u64,          // 8 @ 56: [num_events] Union-Find
    pub num_clusters: u64,    // 8 @ 64: single value
                              // Total: 72 bytes
}

/// Result of RT clustering
#[derive(Debug, Clone)]
pub struct RtClusteringResult {
    /// Cluster ID for each event (-1 = noise)
    pub cluster_ids: Vec<i32>,
    /// Number of clusters found
    pub num_clusters: usize,
    /// Total neighbor pairs found
    pub total_neighbors: usize,
    /// GPU time in milliseconds
    pub gpu_time_ms: f64,
}

/// RT-Core accelerated clustering engine v2
#[cfg(feature = "gpu")]
pub struct RtClusteringEngine {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    optix_ctx: OptixContext,
    pub config: RtClusteringConfig,

    // Single OptiX pipeline (v2: one pipeline instead of two)
    module: Option<Module>,
    raygen_pg: Option<ProgramGroup>,
    miss_pg: Option<ProgramGroup>,
    hitgroup_pg: Option<ProgramGroup>,
    pipeline: Option<Pipeline>,

    // SBT records (device memory)
    d_raygen_record: Option<CudaSlice<u8>>,
    d_miss_record: Option<CudaSlice<u8>>,
    d_hitgroup_record: Option<CudaSlice<u8>>,

    // CUDA module and kernels for union-find clustering
    cuda_module: Option<Arc<CudaModule>>,
    fn_init_union_find: Option<CudaFunction>,
    fn_union_neighbors_flat: Option<CudaFunction>,
    fn_flatten_clusters_full: Option<CudaFunction>,
    fn_propagate_ids: Option<CudaFunction>,
    fn_count_sizes: Option<CudaFunction>,
    fn_filter_small: Option<CudaFunction>,

    // v2: Cached BVH for reuse across epsilon scales
    cached_bvh: Option<AccelStructure>,
    cached_bvh_ray_tmax: f32, // Max epsilon the BVH was built for
    cached_d_positions: Option<CudaSlice<f32>>,
    cached_d_radii: Option<CudaSlice<f32>>,
    cached_num_events: usize,

    // v2: Cached output buffers (reused across calls)
    cached_d_neighbor_list: Option<CudaSlice<u32>>,
    cached_d_neighbor_count: Option<CudaSlice<u32>>,
    cached_d_parent: Option<CudaSlice<i32>>,
    cached_d_cluster_ids: Option<CudaSlice<i32>>,
    cached_d_num_clusters: Option<CudaSlice<u32>>,
    cached_capacity: usize,
}

#[cfg(feature = "gpu")]
impl RtClusteringEngine {
    /// Create a new RT clustering engine
    pub fn new(context: Arc<CudaContext>, config: RtClusteringConfig) -> Result<Self> {
        // Initialize OptiX
        OptixContext::init().map_err(|e| anyhow::anyhow!("OptiX init failed: {}", e))?;

        let optix_ctx = OptixContext::new(context.cu_ctx(), false)
            .map_err(|e| anyhow::anyhow!("OptiX context failed: {}", e))?;

        let stream = context.default_stream();

        log::info!("RT clustering engine v2 created (single-pass closesthit)");

        Ok(Self {
            context,
            stream,
            optix_ctx,
            config,
            module: None,
            raygen_pg: None,
            miss_pg: None,
            hitgroup_pg: None,
            pipeline: None,
            d_raygen_record: None,
            d_miss_record: None,
            d_hitgroup_record: None,
            cuda_module: None,
            fn_init_union_find: None,
            fn_union_neighbors_flat: None,
            fn_flatten_clusters_full: None,
            fn_propagate_ids: None,
            fn_count_sizes: None,
            fn_filter_small: None,
            cached_bvh: None,
            cached_bvh_ray_tmax: 0.0,
            cached_d_positions: None,
            cached_d_radii: None,
            cached_num_events: 0,
            cached_d_neighbor_list: None,
            cached_d_neighbor_count: None,
            cached_d_parent: None,
            cached_d_cluster_ids: None,
            cached_d_num_clusters: None,
            cached_capacity: 0,
        })
    }

    /// Load the RT clustering pipeline from OptiX IR
    pub fn load_pipeline(&mut self, optixir_path: impl AsRef<Path>) -> Result<()> {
        let path = optixir_path.as_ref();
        log::info!("Loading RT clustering v2 pipeline from: {}", path.display());

        let module_options = ModuleCompileOptions::default();

        let mut pipeline_options = PipelineCompileOptions::default();
        pipeline_options.num_payload_values = 2; // p0=source_event, p1=unused
        pipeline_options.num_attribute_values = 0;

        let params_size = std::mem::size_of::<RtClusteringParams>();
        log::info!("RT clustering v2 params struct size: {} bytes", params_size);

        // Load module from OptiX IR
        let module =
            Module::from_optix_ir(&self.optix_ctx, path, &module_options, &pipeline_options)
                .context("Failed to load OptiX IR module")?;

        // Create program groups — v2: single pipeline with anyhit
        let raygen_pg =
            ProgramGroup::create_raygen(&self.optix_ctx, &module, "__raygen__find_neighbors")
                .context("Failed to create raygen program group")?;

        let miss_pg = ProgramGroup::create_miss(&self.optix_ctx, &module, "__miss__find_neighbors")
            .context("Failed to create miss program group")?;

        // v2: Hitgroup with ANYHIT (the key optimization)
        // closesthit is a no-op, anyhit does all neighbor recording
        let hitgroup_pg = ProgramGroup::create_hitgroup(
            &self.optix_ctx,
            Some(&module),
            Some("__closesthit__find_neighbors"),
            None,
            None,
            None, // No anyhit module (falling back to closesthit)
            None, // No anyhit entry
        )
        .map_err(|e| anyhow::anyhow!("Failed to pack raygen header: {}", e))?;
        let record_size = aligned_sbt_record_size(0); // 0 extra bytes for payload in SBT
        let mut raygen_record = vec![0u8; record_size];
        raygen_pg.pack_header(&mut raygen_record)?;
        self.d_raygen_record = Some(self.stream.clone_htod(&raygen_record)?);

        let mut miss_record = vec![0u8; record_size];
        miss_pg
            .pack_header(&mut miss_record)
            .map_err(|e| anyhow::anyhow!("Failed to pack miss header: {}", e))?;
        self.d_miss_record = Some(self.stream.clone_htod(&miss_record)?);

        let mut hitgroup_record = vec![0u8; record_size];
        hitgroup_pg
            .pack_header(&mut hitgroup_record)
            .map_err(|e| anyhow::anyhow!("Failed to pack hitgroup header: {}", e))?;
        self.d_hitgroup_record = Some(self.stream.clone_htod(&hitgroup_record)?);

        Ok(())
    }

    /// Prepare BVH for a set of positions at a given max epsilon.
    /// Call this once before sweeping multiple epsilon levels.
    /// The BVH is cached and reused by cluster_at_epsilon.
    pub fn prepare_bvh(&mut self, positions: &[f32], max_epsilon: f32) -> Result<()> {
        let num_events = positions.len() / 3;
        if num_events == 0 {
            return Ok(());
        }

        log::info!(
            "Preparing BVH: {} events, max_epsilon={:.1}Å (sphere radius={:.1}Å)",
            num_events,
            max_epsilon,
            max_epsilon / 2.0
        );

        // Upload positions
        let d_positions: CudaSlice<f32> = self.stream.clone_htod(&positions.to_vec())?;

        // Create radii buffer (max_epsilon/2 for all events)
        let radii: Vec<f32> = vec![max_epsilon / 2.0; num_events];
        let d_radii: CudaSlice<f32> = self.stream.clone_htod(&radii)?;

        // Build BVH - scope borrows so d_positions/d_radii can be moved after
        let bvh = {
            let (positions_ptr, _guard1) = d_positions.device_ptr(&self.stream);
            let (radii_ptr, _guard2) = d_radii.device_ptr(&self.stream);
            AccelStructure::build_spheres(
                &self.optix_ctx,
                positions_ptr,
                radii_ptr,
                num_events,
                BvhBuildFlags::dynamic(),
            )
            .map_err(|e| anyhow::anyhow!("BVH build failed: {}", e))?
        };

        // Allocate output buffers if needed
        if num_events > self.cached_capacity {
            let max_neighbors = MAX_NEIGHBORS_PER_EVENT as usize;
            self.cached_d_neighbor_list =
                Some(
                    self.stream
                        .clone_htod(&vec![0u32; num_events * max_neighbors])?,
                );
            self.cached_d_neighbor_count = Some(self.stream.clone_htod(&vec![0u32; num_events])?);
            self.cached_d_parent = Some(
                self.stream
                    .clone_htod(&(0..num_events as i32).collect::<Vec<_>>())?,
            );
            self.cached_d_cluster_ids = Some(self.stream.clone_htod(&vec![-1i32; num_events])?);
            self.cached_d_num_clusters = Some(self.stream.clone_htod(&vec![0u32])?);
            self.cached_capacity = num_events;
            log::info!(
                "  Allocated buffers: {} events × {} max_neighbors = {:.1} MB",
                num_events,
                max_neighbors,
                (num_events * max_neighbors * 4) as f64 / 1e6
            );
        }

        // Cache everything
        self.cached_bvh = Some(bvh);
        self.cached_bvh_ray_tmax = max_epsilon;
        self.cached_d_positions = Some(d_positions);
        self.cached_d_radii = Some(d_radii);
        self.cached_num_events = num_events;

        log::info!(
            "  BVH cached for {} events (reusable across epsilon scales)",
            num_events
        );
        Ok(())
    }

    /// Cluster positions using RT cores (standalone, builds own BVH)
    pub fn cluster(&mut self, positions: &[f32]) -> Result<RtClusteringResult> {
        self.cluster_at_epsilon(positions, self.config.epsilon)
    }

    /// Cluster positions at a specific epsilon.
    /// If prepare_bvh() was called, reuses cached BVH (fast path).
    /// Otherwise builds BVH inline (slow path, backward compatible).
    pub fn cluster_at_epsilon(
        &mut self,
        positions: &[f32],
        epsilon: f32,
    ) -> Result<RtClusteringResult> {
        let num_events = positions.len() / 3;
        if num_events == 0 {
            return Ok(RtClusteringResult {
                cluster_ids: vec![],
                num_clusters: 0,
                total_neighbors: 0,
                gpu_time_ms: 0.0,
            });
        }

        let pipeline = self
            .pipeline
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Pipeline not loaded. Call load_pipeline() first."))?;

        let start = std::time::Instant::now();

        // Check if we have a cached BVH for this position set
        let use_cache = self.cached_bvh.is_some()
            && self.cached_num_events == num_events
            && self.cached_bvh_ray_tmax >= epsilon;

        if !use_cache {
            return self.cluster_at_epsilon_fresh(positions, epsilon);
        }

        // Fast path: reuse cached BVH
        let ray_tmax = self.cached_bvh_ray_tmax;
        let bvh_handle = self.cached_bvh.as_ref().unwrap().handle();
        let positions_ptr = {
            let (p, _) = self
                .cached_d_positions
                .as_ref()
                .unwrap()
                .device_ptr(&self.stream);
            p
        };

        // Reset neighbor counts to 0 for this new epsilon level
        {
            let zeros = vec![0u32; num_events];
            let d_nc = self.cached_d_neighbor_count.as_mut().unwrap();
            self.stream.memcpy_htod(&zeros, d_nc)?;
        }

        // Reset parent array
        {
            let initial_parent: Vec<i32> = (0..num_events as i32).collect();
            let d_par = self.cached_d_parent.as_mut().unwrap();
            self.stream.memcpy_htod(&initial_parent, d_par)?;
        }

        // Reset cluster IDs
        {
            let noise = vec![-1i32; num_events];
            let d_cids = self.cached_d_cluster_ids.as_mut().unwrap();
            self.stream.memcpy_htod(&noise, d_cids)?;
        }

        // Reset num_clusters
        {
            let d_ncl = self.cached_d_num_clusters.as_mut().unwrap();
            self.stream.memcpy_htod(&vec![0u32], d_ncl)?;
        }

        // Get device pointers for params from CACHED buffers
        let neighbor_list_ptr = {
            let (p, _) = self
                .cached_d_neighbor_list
                .as_ref()
                .unwrap()
                .device_ptr(&self.stream);
            p
        };
        let neighbor_count_ptr = {
            let (p, _) = self
                .cached_d_neighbor_count
                .as_ref()
                .unwrap()
                .device_ptr(&self.stream);
            p
        };
        let parent_ptr = {
            let (p, _) = self
                .cached_d_parent
                .as_ref()
                .unwrap()
                .device_ptr(&self.stream);
            p
        };
        let num_clusters_ptr = {
            let (p, _) = self
                .cached_d_num_clusters
                .as_ref()
                .unwrap()
                .device_ptr(&self.stream);
            p
        };

        // Setup launch parameters (72 bytes)
        let params = RtClusteringParams {
            traversable: bvh_handle,
            event_positions: positions_ptr,
            num_events: num_events as u32,
            epsilon_sq: epsilon * epsilon,
            ray_tmax,
            rays_per_event: self.config.rays_per_event,
            max_neighbors: MAX_NEIGHBORS_PER_EVENT,
            _pad: 0,
            neighbor_list: neighbor_list_ptr,
            neighbor_count: neighbor_count_ptr,
            parent: parent_ptr,
            num_clusters: num_clusters_ptr,
        };

        let params_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                &params as *const RtClusteringParams as *const u8,
                std::mem::size_of::<RtClusteringParams>(),
            )
        };
        let d_params: CudaSlice<u8> = self.stream.clone_htod(params_bytes)?;
        let (params_ptr_dev, _gp) = d_params.device_ptr(&self.stream);

        // Build SBT
        let record_size = aligned_sbt_record_size(0) as u32;
        let (raygen_ptr, _gr) = self
            .d_raygen_record
            .as_ref()
            .unwrap()
            .device_ptr(&self.stream);
        let (miss_ptr, _gm) = self
            .d_miss_record
            .as_ref()
            .unwrap()
            .device_ptr(&self.stream);
        let (hitgroup_ptr, _gh) = self
            .d_hitgroup_record
            .as_ref()
            .unwrap()
            .device_ptr(&self.stream);

        let sbt = ShaderBindingTable {
            raygen_record: raygen_ptr,
            exception_record: 0,
            miss_record_base: miss_ptr,
            miss_record_stride: record_size,
            miss_record_count: 1,
            hitgroup_record_base: hitgroup_ptr,
            hitgroup_record_stride: record_size,
            hitgroup_record_count: 1,
            callable_record_base: 0,
            callable_record_stride: 0,
            callable_record_count: 0,
        };

        // ══════════════════════════════════════════════════════════════════
        // SINGLE OptiX launch — finds ALL neighbors via anyhit
        // (v1 required TWO launches: count + build)
        // ══════════════════════════════════════════════════════════════════
        let cu_stream = self.stream.cu_stream() as CUstream;
        pipeline
            .launch(
                cu_stream,
                params_ptr_dev,
                std::mem::size_of::<RtClusteringParams>(),
                &sbt,
                num_events as u32,          // width = num_events
                self.config.rays_per_event, // height = rays_per_event
                1,                          // depth = 1
            )
            .map_err(|e| anyhow::anyhow!("Pipeline launch failed: {}", e))?;
        self.stream.synchronize()?;

        // Debug: check neighbor counts
        let d_neighbor_count_dbg = self.cached_d_neighbor_count.as_ref().unwrap();
        let mut neighbor_counts_host = vec![0u32; num_events];
        self.stream
            .memcpy_dtoh(d_neighbor_count_dbg, &mut neighbor_counts_host)?;
        let total_neighbors: u64 = neighbor_counts_host.iter().map(|&x| x as u64).sum();
        let avg_neighbors = if num_events > 0 {
            total_neighbors as f64 / num_events as f64
        } else {
            0.0
        };
        log::info!(
            "  Single-pass (cached BVH): {} neighbors ({:.1} avg/event), {} events, eps={:.1}",
            total_neighbors,
            avg_neighbors,
            num_events,
            epsilon
        );

        // Run union-find clustering pipeline
        let d_parent_ref = self.cached_d_parent.as_ref().unwrap();
        let d_neighbor_list_ref = self.cached_d_neighbor_list.as_ref().unwrap();
        let d_neighbor_count_ref = self.cached_d_neighbor_count.as_ref().unwrap();
        let d_cluster_ids_ref = self.cached_d_cluster_ids.as_ref().unwrap();
        let d_num_clusters_ref = self.cached_d_num_clusters.as_ref().unwrap();
        let final_result = if self.cuda_module.is_some() {
            self.run_union_find(
                d_parent_ref,
                d_neighbor_list_ref,
                d_neighbor_count_ref,
                d_cluster_ids_ref,
                d_num_clusters_ref,
                num_events,
                total_neighbors as usize,
            )?
        } else {
            log::warn!("CUDA kernels not loaded - returning neighbor counts only");
            (vec![-1i32; num_events], total_neighbors as usize, 0)
        };

        let gpu_time = start.elapsed().as_secs_f64() * 1000.0;

        log::info!(
            "RT clustering v2: {} events, {} neighbors, {} clusters, {:.1}ms (ε={:.1}Å)",
            num_events,
            final_result.1,
            final_result.2,
            gpu_time,
            epsilon
        );

        Ok(RtClusteringResult {
            cluster_ids: final_result.0,
            num_clusters: final_result.2,
            total_neighbors: final_result.1,
            gpu_time_ms: gpu_time,
        })
    }

    /// Fresh clustering without cached BVH (backward compatible path)
    fn cluster_at_epsilon_fresh(
        &self,
        positions: &[f32],
        epsilon: f32,
    ) -> Result<RtClusteringResult> {
        let num_events = positions.len() / 3;
        let pipeline = self
            .pipeline
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Pipeline not loaded"))?;

        let start = std::time::Instant::now();

        // Upload positions and build BVH
        let d_positions: CudaSlice<f32> = self.stream.clone_htod(&positions.to_vec())?;
        let radii: Vec<f32> = vec![epsilon / 2.0; num_events];
        let d_radii: CudaSlice<f32> = self.stream.clone_htod(&radii)?;

        let (positions_ptr, _g1) = d_positions.device_ptr(&self.stream);
        let (radii_ptr, _g2) = d_radii.device_ptr(&self.stream);

        let bvh = AccelStructure::build_spheres(
            &self.optix_ctx,
            positions_ptr,
            radii_ptr,
            num_events,
            BvhBuildFlags::dynamic(),
        )
        .map_err(|e| anyhow::anyhow!("BVH build failed: {}", e))?;

        // Allocate buffers
        let max_neighbors = MAX_NEIGHBORS_PER_EVENT as usize;
        let d_neighbor_list: CudaSlice<u32> =
            self.stream
                .clone_htod(&vec![0u32; num_events * max_neighbors])?;
        let d_neighbor_count: CudaSlice<u32> = self.stream.clone_htod(&vec![0u32; num_events])?;
        let d_parent: CudaSlice<i32> = self
            .stream
            .clone_htod(&(0..num_events as i32).collect::<Vec<_>>())?;
        let d_cluster_ids: CudaSlice<i32> = self.stream.clone_htod(&vec![-1i32; num_events])?;
        let d_num_clusters: CudaSlice<u32> = self.stream.clone_htod(&vec![0u32])?;

        // Get device pointers
        let positions_ptr2 = {
            let (p, _) = d_positions.device_ptr(&self.stream);
            p
        };
        let (nl_ptr, _g3) = d_neighbor_list.device_ptr(&self.stream);
        let (nc_ptr, _g4) = d_neighbor_count.device_ptr(&self.stream);
        let (par_ptr, _g5) = d_parent.device_ptr(&self.stream);
        let (cid_ptr, _g6) = d_cluster_ids.device_ptr(&self.stream);
        let (ncl_ptr, _g7) = d_num_clusters.device_ptr(&self.stream);

        let params = RtClusteringParams {
            traversable: bvh.handle(),
            event_positions: positions_ptr2,
            num_events: num_events as u32,
            epsilon_sq: epsilon * epsilon,
            ray_tmax: epsilon,
            rays_per_event: self.config.rays_per_event,
            max_neighbors: MAX_NEIGHBORS_PER_EVENT,
            _pad: 0,
            neighbor_list: nl_ptr,
            neighbor_count: nc_ptr,
            parent: par_ptr,
            num_clusters: ncl_ptr,
        };

        let params_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                &params as *const RtClusteringParams as *const u8,
                std::mem::size_of::<RtClusteringParams>(),
            )
        };
        let d_params: CudaSlice<u8> = self.stream.clone_htod(params_bytes)?;
        let (params_ptr_dev, _gp) = d_params.device_ptr(&self.stream);

        let record_size = aligned_sbt_record_size(0) as u32;
        let (raygen_ptr, _gr) = self
            .d_raygen_record
            .as_ref()
            .unwrap()
            .device_ptr(&self.stream);
        let (miss_ptr, _gm) = self
            .d_miss_record
            .as_ref()
            .unwrap()
            .device_ptr(&self.stream);
        let (hitgroup_ptr, _gh) = self
            .d_hitgroup_record
            .as_ref()
            .unwrap()
            .device_ptr(&self.stream);

        let sbt = ShaderBindingTable {
            raygen_record: raygen_ptr,
            exception_record: 0,
            miss_record_base: miss_ptr,
            miss_record_stride: record_size,
            miss_record_count: 1,
            hitgroup_record_base: hitgroup_ptr,
            hitgroup_record_stride: record_size,
            hitgroup_record_count: 1,
            callable_record_base: 0,
            callable_record_stride: 0,
            callable_record_count: 0,
        };

        let cu_stream = self.stream.cu_stream() as CUstream;
        pipeline
            .launch(
                cu_stream,
                params_ptr_dev,
                std::mem::size_of::<RtClusteringParams>(),
                &sbt,
                num_events as u32,
                self.config.rays_per_event,
                1,
            )
            .map_err(|e| anyhow::anyhow!("Pipeline launch failed: {}", e))?;
        self.stream.synchronize()?;

        // Get total neighbors
        let mut nc_host = vec![0u32; num_events];
        self.stream.memcpy_dtoh(&d_neighbor_count, &mut nc_host)?;
        let total_neighbors: u64 = nc_host.iter().map(|&x| x as u64).sum();

        log::info!(
            "  Single-pass (fresh BVH): {} neighbor pairs from {} events (ε={:.1}Å)",
            total_neighbors,
            num_events,
            epsilon
        );

        let final_result = if self.cuda_module.is_some() {
            self.run_union_find(
                &d_parent,
                &d_neighbor_list,
                &d_neighbor_count,
                &d_cluster_ids,
                &d_num_clusters,
                num_events,
                total_neighbors as usize,
            )?
        } else {
            (vec![-1i32; num_events], total_neighbors as usize, 0)
        };

        let gpu_time = start.elapsed().as_secs_f64() * 1000.0;

        Ok(RtClusteringResult {
            cluster_ids: final_result.0,
            num_clusters: final_result.2,
            total_neighbors: final_result.1,
            gpu_time_ms: gpu_time,
        })
    }

    /// Run union-find pipeline on flat neighbor buffer
    fn run_union_find(
        &self,
        d_parent: &CudaSlice<i32>,
        d_neighbor_list: &CudaSlice<u32>,
        d_neighbor_count: &CudaSlice<u32>,
        d_cluster_ids: &CudaSlice<i32>,
        d_num_clusters: &CudaSlice<u32>,
        num_events: usize,
        total_neighbors: usize,
    ) -> Result<(Vec<i32>, usize, usize)> {
        let blocks = ((num_events + 255) / 256) as u32;
        let num_events_u32 = num_events as u32;
        let max_neighbors_u32 = MAX_NEIGHBORS_PER_EVENT;

        // Phase 1: Initialize union-find
        let fn_init = self.fn_init_union_find.as_ref().unwrap();
        unsafe {
            self.stream
                .launch_builder(fn_init)
                .arg(d_parent)
                .arg(&num_events_u32)
                .launch(LaunchConfig {
                    grid_dim: (blocks, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .context("Failed to launch init_union_find")?;
        }

        // Phase 2: Union neighbors (flat buffer version)
        let fn_union = self.fn_union_neighbors_flat.as_ref().unwrap();
        unsafe {
            self.stream
                .launch_builder(fn_union)
                .arg(d_parent)
                .arg(d_neighbor_list)
                .arg(d_neighbor_count)
                .arg(&num_events_u32)
                .arg(&max_neighbors_u32)
                .launch(LaunchConfig {
                    grid_dim: (blocks, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .context("Failed to launch union_neighbors_flat")?;
        }

        // Phase 3: Flatten (full path compression)
        let fn_flatten = self.fn_flatten_clusters_full.as_ref().unwrap();
        unsafe {
            self.stream
                .launch_builder(fn_flatten)
                .arg(d_parent)
                .arg(&num_events_u32)
                .launch(LaunchConfig {
                    grid_dim: (blocks, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .context("Failed to launch flatten_clusters_full")?;
        }

        // Phase 4: Propagate cluster IDs
        let fn_propagate = self.fn_propagate_ids.as_ref().unwrap();
        unsafe {
            self.stream
                .launch_builder(fn_propagate)
                .arg(d_parent)
                .arg(d_cluster_ids)
                .arg(&num_events_u32)
                .arg(d_num_clusters)
                .launch(LaunchConfig {
                    grid_dim: (blocks, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .context("Failed to launch propagate_cluster_ids")?;
        }
        self.stream.synchronize()?;

        // Download results
        let mut cluster_ids = vec![-1i32; num_events];
        self.stream.memcpy_dtoh(d_cluster_ids, &mut cluster_ids)?;

        let mut unique_clusters: std::collections::HashSet<i32> = std::collections::HashSet::new();
        for &cid in &cluster_ids {
            if cid >= 0 {
                unique_clusters.insert(cid);
            }
        }

        Ok((cluster_ids, total_neighbors, unique_clusters.len()))
    }
}

/// Explicit Drop for proper resource cleanup order
#[cfg(feature = "gpu")]
impl Drop for RtClusteringEngine {
    fn drop(&mut self) {
        log::debug!("Dropping RtClusteringEngine v2 - cleaning up");

        // Step 1: Drop cached BVH and buffers
        drop(self.cached_bvh.take());
        drop(self.cached_d_positions.take());
        drop(self.cached_d_radii.take());
        drop(self.cached_d_neighbor_list.take());
        drop(self.cached_d_neighbor_count.take());
        drop(self.cached_d_parent.take());
        drop(self.cached_d_cluster_ids.take());
        drop(self.cached_d_num_clusters.take());

        // Step 2: Drop SBT records
        drop(self.d_raygen_record.take());
        drop(self.d_miss_record.take());
        drop(self.d_hitgroup_record.take());

        // Step 3: Drop pipeline
        drop(self.pipeline.take());

        // Step 4: Drop program groups
        drop(self.raygen_pg.take());
        drop(self.miss_pg.take());
        drop(self.hitgroup_pg.take());

        // Step 5: Drop module
        drop(self.module.take());

        // Step 6: Drop CUDA module + functions
        self.fn_init_union_find = None;
        self.fn_union_neighbors_flat = None;
        self.fn_flatten_clusters_full = None;
        self.fn_propagate_ids = None;
        self.fn_count_sizes = None;
        self.fn_filter_small = None;
        drop(self.cuda_module.take());

        log::debug!("RtClusteringEngine v2 cleanup complete");
    }
}

/// Find the rt_clustering.optixir file
pub fn find_optixir_path() -> Option<std::path::PathBuf> {
    let candidates = [
        "crates/prism-gpu/src/kernels/rt_clustering.optixir",
        "../prism-gpu/src/kernels/rt_clustering.optixir",
        "../../prism-gpu/src/kernels/rt_clustering.optixir",
    ];

    for path in &candidates {
        let p = std::path::Path::new(path);
        if p.exists() {
            return Some(p.to_path_buf());
        }
    }

    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        let p = std::path::Path::new(&manifest_dir)
            .parent()?
            .join("prism-gpu/src/kernels/rt_clustering.optixir");
        if p.exists() {
            return Some(p);
        }
    }

    None
}

// ─── SpatialNeighborIndex trait impl (Phase 1 LBVH lane) ──────────────────

#[cfg(feature = "gpu")]
impl crate::spatial_index::SpatialNeighborIndex for RtClusteringEngine {
    fn backend(&self) -> crate::spatial_index::SpatialBackend {
        crate::spatial_index::SpatialBackend::OptixRt
    }

    fn prepare(&mut self, positions: &[f32], max_epsilon: f32) -> anyhow::Result<()> {
        self.prepare_bvh(positions, max_epsilon)
    }

    fn query_at_epsilon(
        &mut self,
        positions: &[f32],
        epsilon: f32,
    ) -> anyhow::Result<crate::spatial_index::NeighborQueryResult> {
        let r = self.cluster_at_epsilon(positions, epsilon)?;
        Ok(crate::spatial_index::NeighborQueryResult {
            cluster_ids: r.cluster_ids,
            num_clusters: r.num_clusters,
            total_neighbors: r.total_neighbors,
            gpu_time_ms: r.gpu_time_ms,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = RtClusteringConfig::default();
        assert_eq!(config.epsilon, 5.0);
        assert_eq!(config.min_points, 3);
        assert_eq!(config.rays_per_event, 16); // v2: reduced from 64
    }

    #[test]
    fn test_params_size() {
        // v2 params: 72 bytes (down from 88 in v1)
        assert_eq!(std::mem::size_of::<RtClusteringParams>(), 72);
    }
}
