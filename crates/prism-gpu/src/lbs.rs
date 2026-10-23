//! GPU helpers for PRISM-LBS geometry and clustering.

use cudarc::driver::{
    CudaContext, CudaFunction, CudaStream, DeviceSlice, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use prism_core::PrismError;
use std::path::Path;
use std::sync::Arc;

/// GPU executor for LBS kernels (surface accessibility, distance matrix, clustering, scoring).
pub struct LbsGpu {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    // SASA spatial grid kernels (O(N×27) optimization)
    surface_func: CudaFunction,
    count_atoms_func: CudaFunction,
    fill_grid_func: CudaFunction,
    // Distance matrix kernel
    distance_func: CudaFunction,
    // Pocket clustering kernels (Jones-Plassmann)
    clustering_func: CudaFunction,
    assign_priorities_func: Option<CudaFunction>,
    jones_plassmann_func: Option<CudaFunction>,
    // Druggability scoring
    scoring_func: CudaFunction,
    // Pocket detection kernels
    alpha_sphere_func: Option<CudaFunction>,
    dbscan_neighbor_func: Option<CudaFunction>,
    dbscan_init_func: Option<CudaFunction>,
    dbscan_expand_func: Option<CudaFunction>,
    monte_carlo_func: Option<CudaFunction>,
}

impl LbsGpu {
    /// Load LBS PTX modules from `ptx_dir`. Expects:
    /// - lbs_sasa.ptx with `surface_accessibility_kernel`, `count_atoms_per_cell`, `fill_grid_cells`
    /// - lbs_distance.ptx with `distance_matrix_kernel`, `distance_matrix_sparse`, `distance_matrix_batched`
    /// - lbs_clustering.ptx with `pocket_clustering_kernel`, `assign_priorities`, `jones_plassmann_round`
    /// - lbs_druggability_scoring.ptx with `druggability_score_kernel`
    pub fn new(context: Arc<CudaContext>, ptx_dir: &Path) -> Result<Self, PrismError> {
        let stream = context.default_stream();

        // Load surface accessibility module (optimized with spatial grid O(N×27))
        let path = ptx_dir.join("lbs_sasa.ptx");
        let ptx_src = std::fs::read_to_string(&path)
            .map_err(|e| PrismError::gpu("lbs_sasa", format!("Failed to read PTX: {}", e)))?;
        let surface_module = context
            .load_module(Ptx::from_src(ptx_src))
            .map_err(|e| PrismError::gpu("lbs_sasa", format!("Failed to load PTX: {}", e)))?;
        let surface_func = surface_module
            .load_function("surface_accessibility_kernel")
            .map_err(|e| PrismError::gpu("lbs_sasa", format!("Failed to load kernel: {}", e)))?;
        let count_atoms_func = surface_module
            .load_function("count_atoms_per_cell")
            .map_err(|e| {
                PrismError::gpu(
                    "lbs_sasa",
                    format!("Failed to load count_atoms_per_cell: {}", e),
                )
            })?;
        let fill_grid_func = surface_module
            .load_function("fill_grid_cells")
            .map_err(|e| {
                PrismError::gpu("lbs_sasa", format!("Failed to load fill_grid_cells: {}", e))
            })?;

        // Load distance matrix module (tiled with shared memory)
        let path = ptx_dir.join("lbs_distance.ptx");
        let ptx_src = std::fs::read_to_string(&path)
            .map_err(|e| PrismError::gpu("lbs_distance", format!("Failed to read PTX: {}", e)))?;
        let distance_module = context
            .load_module(Ptx::from_src(ptx_src))
            .map_err(|e| PrismError::gpu("lbs_distance", format!("Failed to load PTX: {}", e)))?;
        let distance_func = distance_module
            .load_function("distance_matrix_kernel")
            .map_err(|e| {
                PrismError::gpu("lbs_distance", format!("Failed to load kernel: {}", e))
            })?;

        // Load pocket clustering module (Jones-Plassmann algorithm)
        let path = ptx_dir.join("lbs_clustering.ptx");
        let ptx_src = std::fs::read_to_string(&path)
            .map_err(|e| PrismError::gpu("lbs_clustering", format!("Failed to read PTX: {}", e)))?;
        let clustering_module = context
            .load_module(Ptx::from_src(ptx_src))
            .map_err(|e| PrismError::gpu("lbs_clustering", format!("Failed to load PTX: {}", e)))?;
        let clustering_func = clustering_module
            .load_function("pocket_clustering_kernel")
            .map_err(|e| {
                PrismError::gpu("lbs_clustering", format!("Failed to load kernel: {}", e))
            })?;
        // Optional Jones-Plassmann parallel coloring kernels
        let assign_priorities_func = clustering_module.load_function("assign_priorities").ok();
        let jones_plassmann_func = clustering_module
            .load_function("jones_plassmann_round")
            .ok();

        // Load druggability scoring module
        let path = ptx_dir.join("lbs_druggability_scoring.ptx");
        let ptx_src = std::fs::read_to_string(&path).map_err(|e| {
            PrismError::gpu(
                "lbs_druggability_scoring",
                format!("Failed to read PTX: {}", e),
            )
        })?;
        let scoring_module = context.load_module(Ptx::from_src(ptx_src)).map_err(|e| {
            PrismError::gpu(
                "lbs_druggability_scoring",
                format!("Failed to load PTX: {}", e),
            )
        })?;
        let scoring_func = scoring_module
            .load_function("druggability_score_kernel")
            .map_err(|e| {
                PrismError::gpu(
                    "lbs_druggability_scoring",
                    format!("Failed to load kernel: {}", e),
                )
            })?;

        // Try to load pocket detection kernels (optional, may not exist)
        let pocket_detection_funcs = Self::load_pocket_detection_kernels(&context, ptx_dir);

        Ok(Self {
            context,
            stream,
            surface_func,
            count_atoms_func,
            fill_grid_func,
            distance_func,
            clustering_func,
            assign_priorities_func,
            jones_plassmann_func,
            scoring_func,
            alpha_sphere_func: pocket_detection_funcs.0,
            dbscan_neighbor_func: pocket_detection_funcs.1,
            dbscan_init_func: pocket_detection_funcs.2,
            dbscan_expand_func: pocket_detection_funcs.3,
            monte_carlo_func: pocket_detection_funcs.4,
        })
    }

    /// Load pocket detection kernels from pocket_detection.ptx
    fn load_pocket_detection_kernels(
        context: &Arc<CudaContext>,
        ptx_dir: &Path,
    ) -> (
        Option<CudaFunction>,
        Option<CudaFunction>,
        Option<CudaFunction>,
        Option<CudaFunction>,
        Option<CudaFunction>,
    ) {
        let path = ptx_dir.join("pocket_detection.ptx");
        if !path.exists() {
            log::warn!(
                "pocket_detection.ptx not found at {:?}, GPU pocket detection disabled",
                path
            );
            return (None, None, None, None, None);
        }

        match std::fs::read_to_string(&path) {
            Ok(ptx_src) => {
                match context.load_module(Ptx::from_src(ptx_src)) {
                    Ok(module) => {
                        let alpha_sphere = module.load_function("generate_alpha_spheres").ok();
                        let dbscan_neighbor = module.load_function("dbscan_neighbor_count").ok();
                        let dbscan_init = module.load_function("dbscan_init_clusters").ok();
                        let dbscan_expand = module.load_function("dbscan_expand_clusters").ok();
                        let monte_carlo = module.load_function("monte_carlo_volume").ok();

                        log::info!("Loaded pocket_detection.ptx: alpha_sphere={}, dbscan={}, monte_carlo={}",
                            alpha_sphere.is_some(),
                            dbscan_neighbor.is_some() && dbscan_init.is_some() && dbscan_expand.is_some(),
                            monte_carlo.is_some()
                        );

                        (
                            alpha_sphere,
                            dbscan_neighbor,
                            dbscan_init,
                            dbscan_expand,
                            monte_carlo,
                        )
                    }
                    Err(e) => {
                        log::warn!("Failed to load pocket_detection.ptx module: {}", e);
                        (None, None, None, None, None)
                    }
                }
            }
            Err(e) => {
                log::warn!("Failed to read pocket_detection.ptx: {}", e);
                (None, None, None, None, None)
            }
        }
    }

    /// Check if GPU pocket detection is available
    pub fn has_pocket_detection(&self) -> bool {
        self.alpha_sphere_func.is_some()
            && self.dbscan_neighbor_func.is_some()
            && self.dbscan_init_func.is_some()
            && self.dbscan_expand_func.is_some()
            && self.monte_carlo_func.is_some()
    }

    /// Compute SASA and surface flags for atoms using spatial grid optimization O(N×27).
    pub fn surface_accessibility(
        &self,
        coords: &[[f32; 3]],
        radii: &[f32],
        samples: i32,
        probe_radius: f32,
    ) -> Result<(Vec<f32>, Vec<u8>), PrismError> {
        let n = coords.len();
        if radii.len() != n {
            return Err(PrismError::gpu(
                "lbs_surface_accessibility",
                "radii length mismatch",
            ));
        }
        if n == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        // Separate coordinates
        let x: Vec<f32> = coords.iter().map(|c| c[0]).collect();
        let y: Vec<f32> = coords.iter().map(|c| c[1]).collect();
        let z: Vec<f32> = coords.iter().map(|c| c[2]).collect();

        // Compute bounding box with padding for probe radius
        let max_radius = radii.iter().cloned().fold(0.0_f32, f32::max);
        let padding = max_radius + probe_radius + 1.0;
        let origin_x = x.iter().cloned().fold(f32::INFINITY, f32::min) - padding;
        let origin_y = y.iter().cloned().fold(f32::INFINITY, f32::min) - padding;
        let origin_z = z.iter().cloned().fold(f32::INFINITY, f32::min) - padding;
        let max_x = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max) + padding;
        let max_y = y.iter().cloned().fold(f32::NEG_INFINITY, f32::max) + padding;
        let max_z = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max) + padding;

        // Cell size should be >= max interaction distance (2*max_radius + 2*probe)
        let cell_size = (2.0 * max_radius + 2.0 * probe_radius).max(3.0);
        let grid_dim_x = ((max_x - origin_x) / cell_size).ceil() as i32;
        let grid_dim_y = ((max_y - origin_y) / cell_size).ceil() as i32;
        let grid_dim_z = ((max_z - origin_z) / cell_size).ceil() as i32;
        let num_cells = (grid_dim_x * grid_dim_y * grid_dim_z) as usize;

        log::debug!(
            "SASA spatial grid: {}x{}x{} = {} cells for {} atoms",
            grid_dim_x,
            grid_dim_y,
            grid_dim_z,
            num_cells,
            n
        );

        // Upload atom data to GPU
        let d_x = self
            .stream
            .clone_htod(&x)
            .map_err(|e| PrismError::gpu("lbs_surface_accessibility", e.to_string()))?;
        let d_y = self
            .stream
            .clone_htod(&y)
            .map_err(|e| PrismError::gpu("lbs_surface_accessibility", e.to_string()))?;
        let d_z = self
            .stream
            .clone_htod(&z)
            .map_err(|e| PrismError::gpu("lbs_surface_accessibility", e.to_string()))?;
        let d_r = self
            .stream
            .clone_htod(radii)
            .map_err(|e| PrismError::gpu("lbs_surface_accessibility", e.to_string()))?;

        // Step 1: Count atoms per cell
        let mut d_cell_counts = self
            .stream
            .alloc_zeros::<i32>(num_cells)
            .map_err(|e| PrismError::gpu("lbs_surface_accessibility", e.to_string()))?;

        let cfg_atoms = LaunchConfig::for_num_elems(n as u32);
        unsafe {
            self.stream
                .launch_builder(&self.count_atoms_func)
                .arg(&d_x)
                .arg(&d_y)
                .arg(&d_z)
                .arg(&(n as i32))
                .arg(&origin_x)
                .arg(&origin_y)
                .arg(&origin_z)
                .arg(&cell_size)
                .arg(&grid_dim_x)
                .arg(&grid_dim_y)
                .arg(&grid_dim_z)
                .arg(&mut d_cell_counts)
                .launch(cfg_atoms)
                .map_err(|e| PrismError::gpu("count_atoms_per_cell", e.to_string()))?;
        }

        // Step 2: Compute prefix sum (exclusive scan) on CPU
        let cell_counts = self
            .stream
            .clone_dtoh(&d_cell_counts)
            .map_err(|e| PrismError::gpu("lbs_surface_accessibility", e.to_string()))?;

        let mut cell_start = Vec::with_capacity(num_cells + 1);
        let mut running_sum = 0i32;
        for &count in &cell_counts {
            cell_start.push(running_sum);
            running_sum += count;
        }
        cell_start.push(running_sum);

        let d_cell_start = self
            .stream
            .clone_htod(&cell_start)
            .map_err(|e| PrismError::gpu("lbs_surface_accessibility", e.to_string()))?;

        // Step 3: Fill grid cells with atom indices
        let mut d_atom_indices = self
            .stream
            .alloc_zeros::<i32>(n)
            .map_err(|e| PrismError::gpu("lbs_surface_accessibility", e.to_string()))?;
        let mut d_cell_offsets = self
            .stream
            .alloc_zeros::<i32>(num_cells)
            .map_err(|e| PrismError::gpu("lbs_surface_accessibility", e.to_string()))?;

        unsafe {
            self.stream
                .launch_builder(&self.fill_grid_func)
                .arg(&d_x)
                .arg(&d_y)
                .arg(&d_z)
                .arg(&(n as i32))
                .arg(&origin_x)
                .arg(&origin_y)
                .arg(&origin_z)
                .arg(&cell_size)
                .arg(&grid_dim_x)
                .arg(&grid_dim_y)
                .arg(&grid_dim_z)
                .arg(&d_cell_start)
                .arg(&mut d_atom_indices)
                .arg(&mut d_cell_offsets)
                .launch(cfg_atoms)
                .map_err(|e| PrismError::gpu("fill_grid_cells", e.to_string()))?;
        }

        // Step 4: Compute SASA using spatial grid
        let mut d_sasa = self
            .stream
            .alloc_zeros::<f32>(n)
            .map_err(|e| PrismError::gpu("lbs_surface_accessibility", e.to_string()))?;
        let mut d_surface = self
            .stream
            .alloc_zeros::<u8>(n)
            .map_err(|e| PrismError::gpu("lbs_surface_accessibility", e.to_string()))?;

        unsafe {
            self.stream
                .launch_builder(&self.surface_func)
                .arg(&d_x)
                .arg(&d_y)
                .arg(&d_z)
                .arg(&d_r)
                .arg(&(n as i32))
                .arg(&samples)
                .arg(&probe_radius)
                // Spatial grid parameters
                .arg(&origin_x)
                .arg(&origin_y)
                .arg(&origin_z)
                .arg(&cell_size)
                .arg(&grid_dim_x)
                .arg(&grid_dim_y)
                .arg(&grid_dim_z)
                .arg(&d_cell_start)
                .arg(&d_atom_indices)
                // Output
                .arg(&mut d_sasa)
                .arg(&mut d_surface)
                .launch(cfg_atoms)
                .map_err(|e| PrismError::gpu("surface_accessibility_kernel", e.to_string()))?;
        }

        let sasa = self
            .stream
            .clone_dtoh(&d_sasa)
            .map_err(|e| PrismError::gpu("lbs_surface_accessibility", e.to_string()))?;
        let surface = self
            .stream
            .clone_dtoh(&d_surface)
            .map_err(|e| PrismError::gpu("lbs_surface_accessibility", e.to_string()))?;
        Ok((sasa, surface))
    }

    /// Compute full pairwise distance matrix (n x n).
    pub fn distance_matrix(&self, coords: &[[f32; 3]]) -> Result<Vec<f32>, PrismError> {
        let n = coords.len();
        let x: Vec<f32> = coords.iter().map(|c| c[0]).collect();
        let y: Vec<f32> = coords.iter().map(|c| c[1]).collect();
        let z: Vec<f32> = coords.iter().map(|c| c[2]).collect();

        let d_x = self
            .stream
            .clone_htod(&x)
            .map_err(|e| PrismError::gpu("lbs_distance_matrix", e.to_string()))?;
        let d_y = self
            .stream
            .clone_htod(&y)
            .map_err(|e| PrismError::gpu("lbs_distance_matrix", e.to_string()))?;
        let d_z = self
            .stream
            .clone_htod(&z)
            .map_err(|e| PrismError::gpu("lbs_distance_matrix", e.to_string()))?;
        let mut d_out = self
            .stream
            .alloc_zeros::<f32>(n * n)
            .map_err(|e| PrismError::gpu("lbs_distance_matrix", e.to_string()))?;

        let block = (16, 16, 1);
        let grid = (
            (n as u32 + block.0 - 1) / block.0,
            (n as u32 + block.1 - 1) / block.1,
            1,
        );
        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: block,
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.distance_func)
                .arg(&d_x)
                .arg(&d_y)
                .arg(&d_z)
                .arg(&(n as i32))
                .arg(&mut d_out)
                .launch(cfg)
                .map_err(|e| PrismError::gpu("lbs_distance_matrix", e.to_string()))?;
        }

        Ok(self
            .stream
            .clone_dtoh(&d_out)
            .map_err(|e| PrismError::gpu("lbs_distance_matrix", e.to_string()))?)
    }

    /// Check if Jones-Plassmann parallel coloring is available
    pub fn has_jones_plassmann(&self) -> bool {
        self.assign_priorities_func.is_some() && self.jones_plassmann_func.is_some()
    }

    /// Greedy pocket clustering (graph coloring) on GPU - legacy single-kernel interface.
    pub fn pocket_clustering(
        &self,
        row_ptr: &[i32],
        col_idx: &[i32],
        max_colors: i32,
    ) -> Result<Vec<i32>, PrismError> {
        // Prefer Jones-Plassmann if available
        if self.has_jones_plassmann() {
            return self.jones_plassmann_cluster(row_ptr, col_idx, max_colors);
        }

        let n = row_ptr.len().saturating_sub(1);
        let d_row = self
            .stream
            .clone_htod(row_ptr)
            .map_err(|e| PrismError::gpu("lbs_pocket_clustering", e.to_string()))?;
        let d_col = self
            .stream
            .clone_htod(col_idx)
            .map_err(|e| PrismError::gpu("lbs_pocket_clustering", e.to_string()))?;
        let mut d_colors = self
            .stream
            .alloc_zeros::<i32>(n)
            .map_err(|e| PrismError::gpu("lbs_pocket_clustering", e.to_string()))?;

        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe {
            self.stream
                .launch_builder(&self.clustering_func)
                .arg(&d_row)
                .arg(&d_col)
                .arg(&(n as i32))
                .arg(&max_colors)
                .arg(&mut d_colors)
                .launch(cfg)
                .map_err(|e| PrismError::gpu("lbs_pocket_clustering", e.to_string()))?;
        }

        Ok(self
            .stream
            .clone_dtoh(&d_colors)
            .map_err(|e| PrismError::gpu("lbs_pocket_clustering", e.to_string()))?)
    }

    /// Jones-Plassmann parallel graph coloring for race-free pocket clustering.
    /// Uses random priority assignment for deterministic, conflict-free parallel coloring.
    pub fn jones_plassmann_cluster(
        &self,
        row_ptr: &[i32],
        col_idx: &[i32],
        max_colors: i32,
    ) -> Result<Vec<i32>, PrismError> {
        let assign_func = self
            .assign_priorities_func
            .as_ref()
            .ok_or_else(|| PrismError::gpu("jones_plassmann", "assign_priorities not loaded"))?;
        let jp_round_func = self.jones_plassmann_func.as_ref().ok_or_else(|| {
            PrismError::gpu("jones_plassmann", "jones_plassmann_round not loaded")
        })?;

        let n = row_ptr.len().saturating_sub(1);
        if n == 0 {
            return Ok(Vec::new());
        }

        // Upload graph to GPU
        let d_row = self
            .stream
            .clone_htod(row_ptr)
            .map_err(|e| PrismError::gpu("jones_plassmann", e.to_string()))?;
        let d_col = self
            .stream
            .clone_htod(col_idx)
            .map_err(|e| PrismError::gpu("jones_plassmann", e.to_string()))?;

        // Initialize colors to -1 (uncolored)
        let mut d_colors = self
            .stream
            .clone_htod(&vec![-1i32; n])
            .map_err(|e| PrismError::gpu("jones_plassmann", e.to_string()))?;

        // Allocate priority array and changed flag
        let mut d_priorities = self
            .stream
            .alloc_zeros::<u32>(n)
            .map_err(|e| PrismError::gpu("jones_plassmann", e.to_string()))?;
        let mut d_changed = self
            .stream
            .alloc_zeros::<i32>(1)
            .map_err(|e| PrismError::gpu("jones_plassmann", e.to_string()))?;

        let cfg = LaunchConfig::for_num_elems(n as u32);
        let seed = 42u32; // Deterministic seed for reproducibility

        // Step 1: Assign random priorities
        unsafe {
            self.stream
                .launch_builder(assign_func)
                .arg(&(n as i32))
                .arg(&seed)
                .arg(&mut d_priorities)
                .launch(cfg)
                .map_err(|e| PrismError::gpu("assign_priorities", e.to_string()))?;
        }

        // Step 2: Iterate Jones-Plassmann rounds until convergence
        let max_rounds = 1000;
        for round in 0..max_rounds {
            // Reset changed flag
            d_changed = self
                .stream
                .clone_htod(&[0i32])
                .map_err(|e| PrismError::gpu("jones_plassmann", e.to_string()))?;

            // Run one round of Jones-Plassmann
            unsafe {
                self.stream
                    .launch_builder(jp_round_func)
                    .arg(&d_row)
                    .arg(&d_col)
                    .arg(&(n as i32))
                    .arg(&max_colors)
                    .arg(&d_priorities)
                    .arg(&mut d_colors)
                    .arg(&mut d_changed)
                    .launch(cfg)
                    .map_err(|e| PrismError::gpu("jones_plassmann_round", e.to_string()))?;
            }

            // Check convergence
            let changed = self
                .stream
                .clone_dtoh(&d_changed)
                .map_err(|e| PrismError::gpu("jones_plassmann", e.to_string()))?;

            if changed[0] == 0 {
                log::debug!(
                    "Jones-Plassmann converged in {} rounds for {} vertices",
                    round + 1,
                    n
                );
                break;
            }
        }

        Ok(self
            .stream
            .clone_dtoh(&d_colors)
            .map_err(|e| PrismError::gpu("jones_plassmann", e.to_string()))?)
    }

    /// GPU aggregation of druggability scores.
    pub fn druggability_score(
        &self,
        volume: &[f32],
        hydrophobicity: &[f32],
        enclosure: &[f32],
        depth: &[f32],
        hbond: &[f32],
        flexibility: &[f32],
        conservation: &[f32],
        topology: &[f32],
        weights: [f32; 8],
    ) -> Result<Vec<f32>, PrismError> {
        let n = volume.len();
        let inputs = [
            hydrophobicity.len(),
            enclosure.len(),
            depth.len(),
            hbond.len(),
            flexibility.len(),
            conservation.len(),
            topology.len(),
        ];
        if inputs.iter().any(|&l| l != n) {
            return Err(PrismError::gpu(
                "lbs_druggability_scoring",
                "input length mismatch",
            ));
        }

        let d_volume = self
            .stream
            .clone_htod(volume)
            .map_err(|e| PrismError::gpu("lbs_druggability_scoring", e.to_string()))?;
        let d_hydro = self
            .stream
            .clone_htod(hydrophobicity)
            .map_err(|e| PrismError::gpu("lbs_druggability_scoring", e.to_string()))?;
        let d_enclosure = self
            .stream
            .clone_htod(enclosure)
            .map_err(|e| PrismError::gpu("lbs_druggability_scoring", e.to_string()))?;
        let d_depth = self
            .stream
            .clone_htod(depth)
            .map_err(|e| PrismError::gpu("lbs_druggability_scoring", e.to_string()))?;
        let d_hbond = self
            .stream
            .clone_htod(hbond)
            .map_err(|e| PrismError::gpu("lbs_druggability_scoring", e.to_string()))?;
        let d_flex = self
            .stream
            .clone_htod(flexibility)
            .map_err(|e| PrismError::gpu("lbs_druggability_scoring", e.to_string()))?;
        let d_cons = self
            .stream
            .clone_htod(conservation)
            .map_err(|e| PrismError::gpu("lbs_druggability_scoring", e.to_string()))?;
        let d_topo = self
            .stream
            .clone_htod(topology)
            .map_err(|e| PrismError::gpu("lbs_druggability_scoring", e.to_string()))?;
        let d_weights = self
            .stream
            .clone_htod(&weights)
            .map_err(|e| PrismError::gpu("lbs_druggability_scoring", e.to_string()))?;
        let mut d_out = self
            .stream
            .alloc_zeros::<f32>(n)
            .map_err(|e| PrismError::gpu("lbs_druggability_scoring", e.to_string()))?;

        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe {
            self.stream
                .launch_builder(&self.scoring_func)
                .arg(&d_volume)
                .arg(&d_hydro)
                .arg(&d_enclosure)
                .arg(&d_depth)
                .arg(&d_hbond)
                .arg(&d_flex)
                .arg(&d_cons)
                .arg(&d_topo)
                .arg(&d_weights)
                .arg(&(n as i32))
                .arg(&mut d_out)
                .launch(cfg)
                .map_err(|e| PrismError::gpu("lbs_druggability_scoring", e.to_string()))?;
        }

        Ok(self
            .stream
            .clone_dtoh(&d_out)
            .map_err(|e| PrismError::gpu("lbs_druggability_scoring", e.to_string()))?)
    }

    /// GPU-accelerated alpha sphere generation for pocket detection.
    /// Returns (sphere_coords, sphere_radii, sphere_burial_depths, sphere_valid_mask).
    pub fn generate_alpha_spheres(
        &self,
        atom_coords: &[[f32; 3]],
        atom_vdw: &[f32],
        grid_bounds: (f32, f32, f32, f32, f32, f32), // min_x, max_x, min_y, max_y, min_z, max_z
        grid_spacing: f32,
    ) -> Result<(Vec<[f32; 3]>, Vec<f32>, Vec<f32>, Vec<i32>), PrismError> {
        let alpha_func = self
            .alpha_sphere_func
            .as_ref()
            .ok_or_else(|| PrismError::gpu("generate_alpha_spheres", "Kernel not loaded"))?;

        let n_atoms = atom_coords.len();
        let (min_x, max_x, min_y, max_y, min_z, max_z) = grid_bounds;

        // Compute grid dimensions
        let grid_nx = ((max_x - min_x) / grid_spacing).ceil() as i32;
        let grid_ny = ((max_y - min_y) / grid_spacing).ceil() as i32;
        let grid_nz = ((max_z - min_z) / grid_spacing).ceil() as i32;
        let max_spheres = (grid_nx * grid_ny * grid_nz) as usize;

        // Compute protein centroid and max distance
        let centroid_x: f32 = atom_coords.iter().map(|c| c[0]).sum::<f32>() / n_atoms as f32;
        let centroid_y: f32 = atom_coords.iter().map(|c| c[1]).sum::<f32>() / n_atoms as f32;
        let centroid_z: f32 = atom_coords.iter().map(|c| c[2]).sum::<f32>() / n_atoms as f32;

        let max_dist = atom_coords
            .iter()
            .map(|c| {
                let dx = c[0] - centroid_x;
                let dy = c[1] - centroid_y;
                let dz = c[2] - centroid_z;
                (dx * dx + dy * dy + dz * dz).sqrt()
            })
            .fold(0.0_f32, |a, b| a.max(b));

        // Prepare atom data
        let atom_x: Vec<f32> = atom_coords.iter().map(|c| c[0]).collect();
        let atom_y: Vec<f32> = atom_coords.iter().map(|c| c[1]).collect();
        let atom_z: Vec<f32> = atom_coords.iter().map(|c| c[2]).collect();

        // Copy to GPU
        let d_atom_x = self
            .stream
            .clone_htod(&atom_x)
            .map_err(|e| PrismError::gpu("generate_alpha_spheres", e.to_string()))?;
        let d_atom_y = self
            .stream
            .clone_htod(&atom_y)
            .map_err(|e| PrismError::gpu("generate_alpha_spheres", e.to_string()))?;
        let d_atom_z = self
            .stream
            .clone_htod(&atom_z)
            .map_err(|e| PrismError::gpu("generate_alpha_spheres", e.to_string()))?;
        let d_atom_vdw = self
            .stream
            .clone_htod(atom_vdw)
            .map_err(|e| PrismError::gpu("generate_alpha_spheres", e.to_string()))?;

        // Allocate output buffers
        let mut d_sphere_x = self
            .stream
            .alloc_zeros::<f32>(max_spheres)
            .map_err(|e| PrismError::gpu("generate_alpha_spheres", e.to_string()))?;
        let mut d_sphere_y = self
            .stream
            .alloc_zeros::<f32>(max_spheres)
            .map_err(|e| PrismError::gpu("generate_alpha_spheres", e.to_string()))?;
        let mut d_sphere_z = self
            .stream
            .alloc_zeros::<f32>(max_spheres)
            .map_err(|e| PrismError::gpu("generate_alpha_spheres", e.to_string()))?;
        let mut d_sphere_radius = self
            .stream
            .alloc_zeros::<f32>(max_spheres)
            .map_err(|e| PrismError::gpu("generate_alpha_spheres", e.to_string()))?;
        let mut d_sphere_burial = self
            .stream
            .alloc_zeros::<f32>(max_spheres)
            .map_err(|e| PrismError::gpu("generate_alpha_spheres", e.to_string()))?;
        let mut d_sphere_nearby = self
            .stream
            .alloc_zeros::<i32>(max_spheres)
            .map_err(|e| PrismError::gpu("generate_alpha_spheres", e.to_string()))?;
        let mut d_sphere_valid = self
            .stream
            .alloc_zeros::<i32>(max_spheres)
            .map_err(|e| PrismError::gpu("generate_alpha_spheres", e.to_string()))?;

        // Launch kernel
        let cfg = LaunchConfig::for_num_elems(max_spheres as u32);
        unsafe {
            self.stream
                .launch_builder(alpha_func)
                .arg(&d_atom_x)
                .arg(&d_atom_y)
                .arg(&d_atom_z)
                .arg(&d_atom_vdw)
                .arg(&(n_atoms as i32))
                .arg(&min_x)
                .arg(&min_y)
                .arg(&min_z)
                .arg(&max_x)
                .arg(&max_y)
                .arg(&max_z)
                .arg(&grid_spacing)
                .arg(&grid_nx)
                .arg(&grid_ny)
                .arg(&grid_nz)
                .arg(&centroid_x)
                .arg(&centroid_y)
                .arg(&centroid_z)
                .arg(&max_dist)
                .arg(&mut d_sphere_x)
                .arg(&mut d_sphere_y)
                .arg(&mut d_sphere_z)
                .arg(&mut d_sphere_radius)
                .arg(&mut d_sphere_burial)
                .arg(&mut d_sphere_nearby)
                .arg(&mut d_sphere_valid)
                .arg(&(max_spheres as i32))
                .launch(cfg)
                .map_err(|e| PrismError::gpu("generate_alpha_spheres", e.to_string()))?;
        }

        // Copy results back
        let sphere_x = self
            .stream
            .clone_dtoh(&d_sphere_x)
            .map_err(|e| PrismError::gpu("generate_alpha_spheres", e.to_string()))?;
        let sphere_y = self
            .stream
            .clone_dtoh(&d_sphere_y)
            .map_err(|e| PrismError::gpu("generate_alpha_spheres", e.to_string()))?;
        let sphere_z = self
            .stream
            .clone_dtoh(&d_sphere_z)
            .map_err(|e| PrismError::gpu("generate_alpha_spheres", e.to_string()))?;
        let sphere_radius = self
            .stream
            .clone_dtoh(&d_sphere_radius)
            .map_err(|e| PrismError::gpu("generate_alpha_spheres", e.to_string()))?;
        let sphere_burial = self
            .stream
            .clone_dtoh(&d_sphere_burial)
            .map_err(|e| PrismError::gpu("generate_alpha_spheres", e.to_string()))?;
        let sphere_valid = self
            .stream
            .clone_dtoh(&d_sphere_valid)
            .map_err(|e| PrismError::gpu("generate_alpha_spheres", e.to_string()))?;

        // Compact valid spheres
        let mut coords = Vec::new();
        let mut radii = Vec::new();
        let mut burials = Vec::new();
        let mut valid_mask = Vec::new();

        for i in 0..max_spheres {
            if sphere_valid[i] != 0 {
                coords.push([sphere_x[i], sphere_y[i], sphere_z[i]]);
                radii.push(sphere_radius[i]);
                burials.push(sphere_burial[i]);
                valid_mask.push(1);
            }
        }

        log::info!(
            "GPU alpha sphere generation: {} valid spheres from {} grid points",
            coords.len(),
            max_spheres
        );

        Ok((coords, radii, burials, valid_mask))
    }

    /// GPU-accelerated DBSCAN clustering for alpha spheres.
    /// Returns cluster labels (-1 = noise, -2 = unvisited, 0+ = cluster ID).
    pub fn dbscan_cluster(
        &self,
        sphere_coords: &[[f32; 3]],
        eps: f32,
        min_pts: i32,
    ) -> Result<Vec<i32>, PrismError> {
        let neighbor_func = self
            .dbscan_neighbor_func
            .as_ref()
            .ok_or_else(|| PrismError::gpu("dbscan_cluster", "Neighbor kernel not loaded"))?;
        let init_func = self
            .dbscan_init_func
            .as_ref()
            .ok_or_else(|| PrismError::gpu("dbscan_cluster", "Init kernel not loaded"))?;
        let expand_func = self
            .dbscan_expand_func
            .as_ref()
            .ok_or_else(|| PrismError::gpu("dbscan_cluster", "Expand kernel not loaded"))?;

        let n = sphere_coords.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        let eps_sq = eps * eps;
        let sphere_x: Vec<f32> = sphere_coords.iter().map(|c| c[0]).collect();
        let sphere_y: Vec<f32> = sphere_coords.iter().map(|c| c[1]).collect();
        let sphere_z: Vec<f32> = sphere_coords.iter().map(|c| c[2]).collect();

        // Copy to GPU
        let d_x = self
            .stream
            .clone_htod(&sphere_x)
            .map_err(|e| PrismError::gpu("dbscan_cluster", e.to_string()))?;
        let d_y = self
            .stream
            .clone_htod(&sphere_y)
            .map_err(|e| PrismError::gpu("dbscan_cluster", e.to_string()))?;
        let d_z = self
            .stream
            .clone_htod(&sphere_z)
            .map_err(|e| PrismError::gpu("dbscan_cluster", e.to_string()))?;
        let d_valid = self
            .stream
            .clone_htod(&vec![1i32; n])
            .map_err(|e| PrismError::gpu("dbscan_cluster", e.to_string()))?;

        // Allocate buffers
        let mut d_neighbor_count = self
            .stream
            .alloc_zeros::<i32>(n)
            .map_err(|e| PrismError::gpu("dbscan_cluster", e.to_string()))?;
        let mut d_labels = self
            .stream
            .alloc_zeros::<i32>(n)
            .map_err(|e| PrismError::gpu("dbscan_cluster", e.to_string()))?;
        let mut d_changed = self
            .stream
            .alloc_zeros::<i32>(1)
            .map_err(|e| PrismError::gpu("dbscan_cluster", e.to_string()))?;
        let mut d_cluster_counter = self
            .stream
            .alloc_zeros::<i32>(1)
            .map_err(|e| PrismError::gpu("dbscan_cluster", e.to_string()))?;

        let cfg = LaunchConfig::for_num_elems(n as u32);

        // Step 1: Count neighbors
        unsafe {
            self.stream
                .launch_builder(neighbor_func)
                .arg(&d_x)
                .arg(&d_y)
                .arg(&d_z)
                .arg(&d_valid)
                .arg(&mut d_neighbor_count)
                .arg(&eps_sq)
                .arg(&(n as i32))
                .launch(cfg)
                .map_err(|e| PrismError::gpu("dbscan_cluster", e.to_string()))?;
        }

        // Step 2: Initialize clusters
        unsafe {
            self.stream
                .launch_builder(init_func)
                .arg(&d_valid)
                .arg(&d_neighbor_count)
                .arg(&mut d_labels)
                .arg(&mut d_cluster_counter)
                .arg(&min_pts)
                .arg(&(n as i32))
                .launch(cfg)
                .map_err(|e| PrismError::gpu("dbscan_cluster", e.to_string()))?;
        }

        // Step 3: Expand clusters iteratively
        for _ in 0..100 {
            // Max iterations
            // Reset changed flag
            d_changed = self
                .stream
                .clone_htod(&[0i32])
                .map_err(|e| PrismError::gpu("dbscan_cluster", e.to_string()))?;

            unsafe {
                self.stream
                    .launch_builder(expand_func)
                    .arg(&d_x)
                    .arg(&d_y)
                    .arg(&d_z)
                    .arg(&d_valid)
                    .arg(&d_neighbor_count)
                    .arg(&mut d_labels)
                    .arg(&mut d_changed)
                    .arg(&eps_sq)
                    .arg(&min_pts)
                    .arg(&(n as i32))
                    .launch(cfg)
                    .map_err(|e| PrismError::gpu("dbscan_cluster", e.to_string()))?;
            }

            // Check if any changes occurred
            let changed = self
                .stream
                .clone_dtoh(&d_changed)
                .map_err(|e| PrismError::gpu("dbscan_cluster", e.to_string()))?;
            if changed[0] == 0 {
                break;
            }
        }

        let labels = self
            .stream
            .clone_dtoh(&d_labels)
            .map_err(|e| PrismError::gpu("dbscan_cluster", e.to_string()))?;

        // Count clusters
        let num_clusters = labels
            .iter()
            .filter(|&&l| l >= 0)
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);
        log::info!("GPU DBSCAN: {} clusters from {} spheres", num_clusters, n);

        Ok(labels)
    }
}
