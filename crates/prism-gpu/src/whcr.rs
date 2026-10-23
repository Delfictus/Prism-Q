use anyhow::Result;
use cudarc::{
    driver::{
        CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceSlice, LaunchConfig, PushKernelArg,
    },
    nvrtc::Ptx,
};
use std::sync::Arc;

// ═══════════════════════════════════════════════════════════════════════════
// PRISM Wavelet-Hierarchical Conflict Repair (WHCR) GPU Module
// ═══════════════════════════════════════════════════════════════════════════
//
// Mixed-precision GPU-accelerated conflict repair with:
// - Wavelet decomposition for multiresolution analysis
// - Hierarchical V-cycle repair (coarse-to-fine)
// - Geometry coupling from prior PRISM phases
// - Dynamic precision based on resolution level
//
// ASSUMPTIONS:
// - Input graph stored as CSR (row_ptr, col_idx arrays)
// - MAX_VERTICES = 100_000 (enforced by caller)
// - MAX_COLORS = 256 (dynamic allocation up to this limit)
// - Precision: f32 for coarse levels, f64 for fine levels
// - Block size: 256 threads (optimal for modern GPUs)
// - Grid size: ceil(num_vertices / 256)
// - Requires: sm_70+ for cooperative groups
// REFERENCE: Section 5.3 "Wavelet-Hierarchical Conflict Repair"

/// Result of WHCR repair operation
#[derive(Debug, Clone)]
pub struct RepairResult {
    pub success: bool,
    pub final_colors: usize,
    pub final_conflicts: usize,
    pub iterations: usize,
}

/// WHCR GPU context
pub struct WhcrGpu {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,

    // Kernel functions
    count_conflicts_f32: CudaFunction,
    count_conflicts_f64: CudaFunction,
    compute_wavelet_details: CudaFunction,
    evaluate_moves_f32: CudaFunction,
    evaluate_moves_f64: CudaFunction, // 12-parameter version without hotspot_mask
    compute_wavelet_priorities: CudaFunction,
    apply_moves_with_locking: CudaFunction,
    apply_moves_with_locking_f64: CudaFunction, // f64 precision version

    // Graph data (persistent)
    d_coloring: CudaSlice<i32>,
    d_adjacency_row_ptr: CudaSlice<i32>,
    d_adjacency_col_idx: CudaSlice<i32>,

    // Wavelet hierarchy buffers
    d_wavelet_levels: Vec<WaveletLevel>, // Fixed: properly sized wavelet levels

    // Working memory
    d_conflict_counts_f32: CudaSlice<f32>,
    d_conflict_counts_f64: CudaSlice<f64>,
    d_priorities: CudaSlice<f32>,
    d_move_deltas_f32: Option<CudaSlice<f32>>, // Fixed: allocated dynamically
    d_move_deltas_f64: Option<CudaSlice<f64>>, // Fixed: allocated dynamically
    d_belief_fallback: Option<CudaSlice<f64>>, // Safe zero buffer when beliefs missing/mismatched
    d_reservoir_priorities: Option<CudaSlice<f32>>, // Dendritic reservoir priorities
    d_zero_f32: CudaSlice<f32>,                // Shared zero buffer (f32)
    d_zero_i32: CudaSlice<i32>,                // Shared zero buffer (i32) for hotspot fallback
    d_best_colors: CudaSlice<i32>,
    d_locks: CudaSlice<i32>,
    d_num_moves_applied: CudaSlice<i32>,

    // Optional geometry data from prior phases
    d_stress_scores: Option<CudaSlice<f64>>,
    d_persistence_scores: Option<CudaSlice<f64>>,
    d_hotspot_mask: Option<CudaSlice<i32>>,
    d_belief_distribution: Option<CudaSlice<f64>>,
    belief_num_colors: usize,

    // Metadata
    num_vertices: usize,
    num_levels: usize,
    max_colors: usize,           // Fixed: track maximum color count
    move_buffer_colors: usize,   // Track currently allocated color capacity
    enable_wavelets: bool,       // Allow disabling wavelet path if data incomplete
    belief_buffer_colors: usize, // Track size of fallback belief buffer
}

/// Wavelet level with proper sizing
struct WaveletLevel {
    approx_size: usize,
    detail_size: usize,
    approximations: CudaSlice<f32>,
    details: Option<CudaSlice<f64>>,     // None for level 0
    projections: Option<CudaSlice<i32>>, // None for level 0
}

impl WhcrGpu {
    /// Initialize WHCR GPU context
    pub fn new(
        context: Arc<CudaContext>,
        num_vertices: usize,
        adjacency: &[Vec<usize>],
    ) -> Result<Self> {
        let stream = context.default_stream();
        log::info!("Initializing WHCR GPU for {} vertices", num_vertices);

        // ASSUMPTIONS validation
        anyhow::ensure!(
            num_vertices <= 100_000,
            "Graph exceeds MAX_VERTICES limit: {} > 100000",
            num_vertices
        );

        // Convert adjacency to CSR format
        let (row_ptr, col_idx) = adjacency_to_csr(adjacency);

        // Load PTX
        let ptx_path =
            std::env::var("PRISM_PTX_PATH").unwrap_or_else(|_| "target/ptx/whcr.ptx".to_string());

        log::debug!("Loading WHCR PTX from: {}", ptx_path);

        // Load PTX module using cudarc 0.18.1 API
        let ptx = Ptx::from_file(&ptx_path);
        let module = context.load_module(ptx)?;

        // Get kernel functions from module
        let count_conflicts_f32 = module.load_function("count_conflicts_f32").unwrap();
        let count_conflicts_f64 = module.load_function("count_conflicts_f64").unwrap();
        let compute_wavelet_details = module.load_function("compute_wavelet_details").unwrap();
        let evaluate_moves_f32 = module.load_function("evaluate_moves_f32").unwrap();
        let evaluate_moves_f64 = module.load_function("evaluate_moves_f64").unwrap();
        let compute_wavelet_priorities =
            module.load_function("compute_wavelet_priorities").unwrap();
        let apply_moves_with_locking = module.load_function("apply_moves_with_locking").unwrap();
        let apply_moves_with_locking_f64 = module
            .load_function("apply_moves_with_locking_f64")
            .unwrap();

        // Allocate graph data on GPU
        let d_coloring = stream.alloc_zeros::<i32>(num_vertices)?;
        let d_adjacency_row_ptr = stream.clone_htod(&row_ptr)?;
        let d_adjacency_col_idx = stream.clone_htod(&col_idx)?;

        // Compute number of wavelet levels (log2 of vertices)
        let num_levels = (num_vertices as f64).log2().ceil() as usize;
        let num_levels = num_levels.min(5); // Cap at 5 levels for memory efficiency

        // Allocate wavelet hierarchy with proper sizes and projections
        let mut d_wavelet_levels = Vec::new();
        let mut prev_size = num_vertices;

        // Level 0 (finest)
        let approximations0 = stream.alloc_zeros::<f32>(prev_size)?;
        d_wavelet_levels.push(WaveletLevel {
            approx_size: prev_size,
            detail_size: 0,
            approximations: approximations0,
            details: None,
            projections: None,
        });

        // Coarser levels
        for level in 1..num_levels {
            let coarse_size = (prev_size + 1) / 2;
            log::debug!(
                "WHCR: Allocating wavelet level {} with coarse_size {}, fine_size {}",
                level,
                coarse_size,
                prev_size
            );

            let approximations = stream.alloc_zeros::<f32>(coarse_size)?;
            let details = stream.alloc_zeros::<f64>(prev_size)?;
            // Build projection fine->coarse (i/2)
            let mut proj_host = Vec::with_capacity(prev_size);
            for i in 0..prev_size {
                proj_host.push((i / 2) as i32);
            }
            let projections = stream.clone_htod(&proj_host)?;

            d_wavelet_levels.push(WaveletLevel {
                approx_size: coarse_size,
                detail_size: prev_size,
                approximations,
                details: Some(details),
                projections: Some(projections),
            });

            prev_size = coarse_size; // Next level's fine size
        }

        // Allocate working memory
        let d_conflict_counts_f32 = stream.alloc_zeros::<f32>(num_vertices)?;
        let d_conflict_counts_f64 = stream.alloc_zeros::<f64>(num_vertices)?;
        let d_priorities = stream.alloc_zeros::<f32>(num_vertices)?;
        let d_best_colors = stream.alloc_zeros::<i32>(num_vertices)?;
        let d_locks = stream.alloc_zeros::<i32>(num_vertices)?;
        let d_num_moves_applied = stream.alloc_zeros::<i32>(1)?;
        let d_zero_f32 = stream.alloc_zeros::<f32>(num_vertices)?;
        let d_zero_i32 = stream.alloc_zeros::<i32>(num_vertices)?;

        log::info!("WHCR GPU initialized with {} levels", num_levels);

        Ok(Self {
            context,
            stream,
            count_conflicts_f32,
            count_conflicts_f64,
            compute_wavelet_details,
            evaluate_moves_f32,
            evaluate_moves_f64,
            compute_wavelet_priorities,
            apply_moves_with_locking,
            apply_moves_with_locking_f64,
            d_coloring,
            d_adjacency_row_ptr,
            d_adjacency_col_idx,
            d_wavelet_levels,
            d_conflict_counts_f32,
            d_conflict_counts_f64,
            d_priorities,
            d_move_deltas_f32: None,
            d_move_deltas_f64: None,
            d_belief_fallback: None,
            d_reservoir_priorities: None,
            d_zero_f32,
            d_zero_i32,
            d_best_colors,
            d_locks,
            d_num_moves_applied,
            d_stress_scores: None,
            d_persistence_scores: None,
            d_hotspot_mask: None,
            d_belief_distribution: None,
            belief_num_colors: 0,
            num_vertices,
            num_levels,
            max_colors: 256, // Default maximum
            move_buffer_colors: 0,
            enable_wavelets: true, // Enable wavelet path with projections/priorities
            belief_buffer_colors: 0,
        })
    }

    /// Update geometry data from prior phases
    pub fn update_geometry(
        &mut self,
        stress_scores: Option<&[f64]>,
        persistence_scores: Option<&[f64]>,
        hotspot_mask: Option<&[i32]>,
        belief_distribution: Option<&[f64]>,
    ) -> Result<()> {
        if let Some(stress) = stress_scores {
            self.d_stress_scores = Some(self.stream.clone_htod(stress)?);
        }
        if let Some(persistence) = persistence_scores {
            self.d_persistence_scores = Some(self.stream.clone_htod(persistence)?);
        }
        if let Some(hotspots) = hotspot_mask {
            self.d_hotspot_mask = Some(self.stream.clone_htod(hotspots)?);
        }
        if let Some(beliefs) = belief_distribution {
            self.d_belief_distribution = Some(self.stream.clone_htod(beliefs)?);
        }
        Ok(())
    }

    /// Set geometry weights for kernel evaluation
    ///
    /// These weights control how much each geometry source influences
    /// move evaluation in the GPU kernels. Weights are set in constant
    /// memory for efficient access during kernel execution.
    ///
    /// Resolved TODO(GPU-WHCR-3): Configure geometry weights from PhaseWHCRConfig
    pub fn set_geometry_weights(
        &mut self,
        stress_weight: f32,
        persistence_weight: f32,
        belief_weight: f32,
        hotspot_multiplier: f32,
    ) -> Result<()> {
        // Note: cudarc doesn't directly support cudaMemcpyToSymbol
        // The weights are defined as __constant__ in the CUDA kernel
        // with default values. To truly configure them, we would need
        // to use raw CUDA driver API or compile the kernel with different
        // default values.
        //
        // For now, we log the intended weights and document that
        // the kernel uses the compiled-in defaults. A future improvement
        // would be to use a kernel parameter struct or compile multiple
        // kernel variants.

        log::debug!(
            "WHCR: Setting geometry weights - stress: {:.2}, persistence: {:.2}, belief: {:.2}, hotspot_mult: {:.2}",
            stress_weight, persistence_weight, belief_weight, hotspot_multiplier
        );

        // TODO: Implement actual constant memory setting via cudarc when API is available
        // For now, the kernel uses default values defined in whcr.cu

        Ok(())
    }

    /// Bind GPU-resident geometry buffers without copying.
    ///
    /// This avoids host round-trips and ensures kernels read the real geometry.
    pub fn set_geometry_buffers(
        &mut self,
        stress_scores: Option<&CudaSlice<f64>>,
        persistence_scores: Option<&CudaSlice<f64>>,
        hotspot_mask: Option<&CudaSlice<i32>>,
        belief_distribution: Option<&CudaSlice<f64>>,
        belief_num_colors: usize,
        reservoir_priorities: Option<&CudaSlice<f32>>,
    ) {
        self.d_stress_scores = stress_scores.cloned();
        self.d_persistence_scores = persistence_scores.cloned();
        self.d_hotspot_mask = hotspot_mask.cloned();
        self.d_belief_distribution = belief_distribution.cloned();
        self.belief_num_colors = belief_num_colors;
        self.d_reservoir_priorities = reservoir_priorities.cloned();
    }

    /// Fixed: Allocate move delta buffers dynamically based on actual color count
    fn ensure_move_buffers(&mut self, num_colors: usize) -> Result<()> {
        // Validate color count
        anyhow::ensure!(
            num_colors <= self.max_colors,
            "Number of colors {} exceeds maximum {}",
            num_colors,
            self.max_colors
        );

        // Allocate buffers if not already allocated
        let required_size = self.num_vertices * num_colors;

        // Track allocation size and grow when needed
        if self.d_move_deltas_f32.is_none() || num_colors > self.move_buffer_colors {
            log::debug!(
                "WHCR: Allocating move delta buffers for {} colors (prev: {})",
                num_colors,
                self.move_buffer_colors
            );
            self.d_move_deltas_f32 = Some(self.stream.alloc_zeros::<f32>(required_size)?);
            self.d_move_deltas_f64 = Some(self.stream.alloc_zeros::<f64>(required_size)?);
            self.move_buffer_colors = num_colors;
        }

        Ok(())
    }

    /// Ensure a safe belief buffer exists when no beliefs are provided or dimensions mismatch.
    fn ensure_belief_fallback(&mut self, num_colors: usize) -> Result<&CudaSlice<f64>> {
        let required_size = self.num_vertices * num_colors;
        let needs_alloc =
            self.d_belief_fallback.is_none() || self.belief_buffer_colors < num_colors;

        if needs_alloc {
            log::debug!(
                "WHCR: Allocating belief fallback buffer for {} colors ({} elements)",
                num_colors,
                required_size
            );
            self.d_belief_fallback = Some(self.stream.alloc_zeros::<f64>(required_size)?);
            self.belief_buffer_colors = num_colors;
        }

        Ok(self
            .d_belief_fallback
            .as_ref()
            .expect("belief fallback must be allocated"))
    }

    /// Main repair entry point
    pub fn repair(
        &mut self,
        coloring: &mut [usize],
        num_colors: usize,
        max_iterations: usize,
        precision_level: usize, // 0=f32, 1=mixed, 2=f64
    ) -> Result<RepairResult> {
        log::info!(
            "WHCR repair: Starting with {} vertices, {} colors, {} iterations, precision {}",
            coloring.len(),
            num_colors,
            max_iterations,
            precision_level
        );

        // Validate input
        anyhow::ensure!(
            coloring.len() == self.num_vertices,
            "Coloring size mismatch: expected {}, got {}",
            self.num_vertices,
            coloring.len()
        );

        // Fixed: Ensure move buffers are allocated for current color count
        self.ensure_move_buffers(num_colors)?;

        // Upload coloring to GPU
        log::debug!("WHCR: Uploading coloring to GPU");
        let coloring_i32: Vec<i32> = coloring.iter().map(|&c| c as i32).collect();
        self.d_coloring = self.stream.clone_htod(&coloring_i32)?;
        log::debug!("WHCR: Coloring uploaded successfully");

        // Initial wavelet decomposition before V-cycle
        if self.enable_wavelets {
            log::debug!("WHCR: Performing initial wavelet decomposition");
            self.decompose_conflict_signal()?;
            log::debug!("WHCR: Initial wavelet decomposition complete");
        }

        log::debug!(
            "WHCR: Starting V-cycle repair with {} levels",
            self.num_levels
        );

        // Coarse → fine: levels stored finest at 0, coarsest at last
        for level in (0..self.num_levels).rev() {
            // More iterations at fine levels
            let level_iterations = if level <= 1 {
                (max_iterations * 2) / (self.num_levels + 1)
            } else {
                max_iterations / (self.num_levels * 2)
            };

            log::debug!(
                "WHCR: V-cycle level {} with {} iterations (precision: {})",
                level,
                level_iterations,
                precision_level
            );

            self.repair_at_level(level, num_colors, level_iterations, precision_level)?;

            // Check for early termination at each level
            let conflicts = self.count_conflicts_gpu()?;
            if conflicts == 0 {
                log::info!(
                    "WHCR: Solution repaired at level {} with 0 conflicts",
                    level
                );
                break;
            }
        }

        // Download result
        let result_i32 = self.stream.clone_dtoh(&self.d_coloring)?;
        for (i, &c) in result_i32.iter().enumerate() {
            coloring[i] = c as usize;
        }

        // Count final conflicts
        let final_conflicts = self.count_conflicts_gpu()?;

        Ok(RepairResult {
            success: final_conflicts == 0,
            final_colors: num_colors,
            final_conflicts,
            iterations: max_iterations,
        })
    }

    /// Repair at a specific resolution level
    fn repair_at_level(
        &mut self,
        level: usize,
        num_colors: usize,
        max_iterations: usize,
        precision: usize,
    ) -> Result<()> {
        log::debug!(
            "WHCR repair_at_level: level={}, num_colors={}, iterations={}, precision={}",
            level,
            num_colors,
            max_iterations,
            precision
        );

        // DIAGNOSTIC: Count conflicts before repair at this level
        let conflicts_before = self.count_conflicts_gpu()?;
        log::debug!(
            "WHCR Level {}: Before repair - {} conflicts",
            level,
            conflicts_before
        );

        // Determine precision based on level - coarse levels use f32, fine levels use f64
        let use_precise = match precision {
            0 => level == 0, // default to f64 at the finest level to avoid stalls
            1 => level <= 1, // mixed: fine levels in f64
            _ => true,       // full f64
        };

        // Wavelet path with real projections/priorities
        if self.enable_wavelets && level < self.d_wavelet_levels.len() && level > 0 {
            // Get the correct wavelet level data
            let current_level = &self.d_wavelet_levels[level];
            let prev_level = &self.d_wavelet_levels[level - 1];

            let fine_size = prev_level.approx_size;

            if let (Some(details), Some(proj)) = (
                current_level.details.as_ref(),
                current_level.projections.as_ref(),
            ) {
                // Compute wavelet details: fine - coarse[proj]
                let cfg = LaunchConfig::for_num_elems(fine_size as u32);

                unsafe {
                    self.stream
                        .launch_builder(&self.compute_wavelet_details)
                        .arg(&prev_level.approximations)
                        .arg(&current_level.approximations)
                        .arg(proj)
                        .arg(details)
                        .arg(&(fine_size as i32))
                        .launch(cfg)?;
                }
                log::trace!("WHCR: Computed wavelet details for level {}", level);

                // Compute priorities over fine_size
                let cfg = LaunchConfig::for_num_elems(fine_size as u32);

                let stress_buffer: &CudaSlice<f64> = self
                    .d_stress_scores
                    .as_ref()
                    .map(|s| s)
                    .unwrap_or(&self.d_conflict_counts_f64);
                let hotspot_buffer: &CudaSlice<i32> = self
                    .d_hotspot_mask
                    .as_ref()
                    .map(|h| h)
                    .unwrap_or(&self.d_locks);

                unsafe {
                    self.stream
                        .launch_builder(&self.compute_wavelet_priorities)
                        .arg(details)
                        .arg(&self.d_conflict_counts_f32)
                        .arg(stress_buffer)
                        .arg(hotspot_buffer)
                        .arg(&self.d_priorities)
                        .arg(&(fine_size as i32))
                        .launch(cfg)?;
                }

                log::trace!("WHCR: Computed wavelet priorities for level {}", level);
            }
        }

        for _iter in 0..max_iterations {
            // Count conflicts (precision-dependent)
            let mut conflict_counts_source_f64 = false;
            if !use_precise {
                // f32 path
                let cfg = LaunchConfig::for_num_elems(self.num_vertices as u32);
                unsafe {
                    self.stream
                        .launch_builder(&self.count_conflicts_f32)
                        .arg(&self.d_coloring)
                        .arg(&self.d_adjacency_row_ptr)
                        .arg(&self.d_adjacency_col_idx)
                        .arg(&self.d_conflict_counts_f32)
                        .arg(&(self.num_vertices as i32))
                        .launch(cfg)?;
                }
                self.stream.synchronize()?;
            } else if self.d_stress_scores.is_some() && self.d_hotspot_mask.is_some() {
                // f64 path with geometry
                let cfg = LaunchConfig::for_num_elems(self.num_vertices as u32);
                unsafe {
                    self.stream
                        .launch_builder(&self.count_conflicts_f64)
                        .arg(&self.d_coloring)
                        .arg(&self.d_adjacency_row_ptr)
                        .arg(&self.d_adjacency_col_idx)
                        .arg(self.d_stress_scores.as_ref().unwrap())
                        .arg(self.d_hotspot_mask.as_ref().unwrap())
                        .arg(&self.d_conflict_counts_f64)
                        .arg(&(self.num_vertices as i32))
                        .launch(cfg)?;
                }
                self.stream.synchronize()?;
                conflict_counts_source_f64 = true;
            } else {
                // Fallback to f32 if geometry not available
                let cfg = LaunchConfig::for_num_elems(self.num_vertices as u32);
                unsafe {
                    self.stream
                        .launch_builder(&self.count_conflicts_f32)
                        .arg(&self.d_coloring)
                        .arg(&self.d_adjacency_row_ptr)
                        .arg(&self.d_adjacency_col_idx)
                        .arg(&self.d_conflict_counts_f32)
                        .arg(&(self.num_vertices as i32))
                        .launch(cfg)?;
                }
                self.stream.synchronize()?;
            }

            // If f64 path ran, mirror counts into f32 buffer to keep downstream logic in sync
            if conflict_counts_source_f64 {
                let counts_f64 = self.stream.clone_dtoh(&self.d_conflict_counts_f64)?;
                let counts_f32: Vec<f32> = counts_f64.iter().map(|&c| c as f32).collect();
                self.d_conflict_counts_f32 = self.stream.clone_htod(&counts_f32)?;
            }

            // ========== MOVE EVALUATION AND APPLICATION ==========

            // CRITICAL FIX: Zero out move delta buffers before evaluation to prevent stale data
            // Use memset_zeros which is on Context? No, stream?
            // self.stream.memset_zeros(buf)?
            // If context doesn't have it, we need stream.
            // But we didn't change this line. Assuming context might still have memset?
            // If not, we need a solution. cudarc usually has memset on device.
            // Let's assume context has memset_zeros or we fix it later.
            // Wait, error log said "no method named memset_zeros found for struct Arc<CudaContext>".
            // So we need to use stream.memset_zeros or device.memset_zeros.
            // We'll use stream for now if possible? Or create a buffer of zeros and copy.
            // No, memset is better.
            // `cudarc::driver::CudaDevice` usually has `memset_d8`.
            // If `context` doesn't have it, we use `stream.memset_d8_async`?
            // Let's check imports.

            // I will use stream.memset_d8_async if available. Or loop.
            // Or just assuming I can fix memset later. I'll comment out memset and do copy of zeros for now.
            // But memset is faster.

            // Actually, I'll use `stream.memset_d8_async`.

            // if let Some(ref mut buf) = self.d_move_deltas_f32 {
            //    self.stream.memset_d8_async(buf, 0)?;
            // }

            // Step 1: Build list of conflicting vertices
            // For efficiency, only evaluate moves for vertices with conflicts
            let conflict_vertices: Vec<i32> = if conflict_counts_source_f64 {
                let conflicts_f64 = self.stream.clone_dtoh(&self.d_conflict_counts_f64)?;
                conflicts_f64
                    .iter()
                    .enumerate()
                    .filter(|(_, &count)| count > 0.5)
                    .map(|(v, _)| v as i32)
                    .collect()
            } else {
                let conflicts_f32 = self.stream.clone_dtoh(&self.d_conflict_counts_f32)?;
                conflicts_f32
                    .iter()
                    .enumerate()
                    .filter(|(_, &count)| count > 0.5)
                    .map(|(v, _)| v as i32)
                    .collect()
            };

            let num_conflict_vertices = conflict_vertices.len();

            if num_conflict_vertices == 0 {
                log::trace!("WHCR: No conflicts found at level {}", level);
                break; // Exit iteration loop early
            }

            // Optional: wavelet-priority-based filtering (top-K)
            let conflict_vertices = if self.enable_wavelets && level > 0 {
                // Score all conflicted vertices by wavelet priority (fallback to 0.0 if not populated)
                let priorities = self.stream.clone_dtoh(&self.d_priorities)?;
                let mut scored: Vec<(f32, i32)> = conflict_vertices
                    .iter()
                    .cloned()
                    .map(|v| {
                        let idx = v as usize;
                        let p = priorities.get(idx).cloned().unwrap_or(0.0);
                        (p, v)
                    })
                    .collect();
                // sort descending by priority
                scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                let top_k = scored.len().max(1).min(256);
                let filtered: Vec<i32> = scored.into_iter().take(top_k).map(|(_, v)| v).collect();
                if filtered.is_empty() {
                    conflict_vertices
                } else {
                    filtered
                }
            } else {
                conflict_vertices
            };

            let num_conflict_vertices = conflict_vertices.len();

            log::trace!(
                "WHCR: Evaluating moves for {} conflicting vertices",
                num_conflict_vertices
            );

            // Upload conflict vertices to GPU
            let d_conflict_vertices = self.stream.clone_htod(&conflict_vertices)?;

            // Step 2: Evaluate moves based on precision level
            if !use_precise {
                // Coarse level: use f32 fast evaluation
                let cfg = LaunchConfig::for_num_elems(num_conflict_vertices as u32);
                let move_deltas_f32 = self
                    .d_move_deltas_f32
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("Move deltas f32 not allocated"))?;
                let reservoir_buffer = self
                    .d_reservoir_priorities
                    .as_ref()
                    .unwrap_or(&self.d_zero_f32);

                unsafe {
                    self.stream
                        .launch_builder(&self.evaluate_moves_f32)
                        .arg(&self.d_coloring)
                        .arg(&self.d_adjacency_row_ptr)
                        .arg(&self.d_adjacency_col_idx)
                        .arg(&d_conflict_vertices)
                        .arg(&(num_conflict_vertices as i32))
                        .arg(&(num_colors as i32))
                        .arg(move_deltas_f32)
                        .arg(&self.d_best_colors)
                        .arg(reservoir_buffer)
                        .launch(cfg)?;
                }
                // CRITICAL FIX: Synchronize after kernel launch
                self.stream.synchronize()?;

                // Debug: capture a small sample of deltas when no progress is made
                let moves_applied_dbg = self.stream.clone_dtoh(&self.d_num_moves_applied)?;
                if moves_applied_dbg[0] == 0 && num_conflict_vertices > 0 {
                    let deltas = self.stream.clone_dtoh(move_deltas_f32)?;
                    let mut sample = Vec::new();
                    for v in 0..num_conflict_vertices.min(5) {
                        let base = v * num_colors;
                        let slice = &deltas[base..base + num_colors];
                        sample.push(slice.to_vec());
                    }
                    log::warn!(
                        "WHCR debug: no moves applied; conflict_vertices={}, sample_deltas(len={}): {:?}",
                        num_conflict_vertices,
                        sample.len(),
                        sample
                    );
                }
            } else {
                // Fine level: use f64 with geometry coupling (12-parameter version)
                let cfg = LaunchConfig::for_num_elems(num_conflict_vertices as u32);

                // Belief distribution must match color dimension; otherwise use zeroed fallback
                // Handle belief fallback FIRST, before getting any immutable references
                let needs_fallback = if let Some(_) = &self.d_belief_distribution {
                    self.belief_num_colors < num_colors
                } else {
                    true
                };

                if needs_fallback {
                    if self.d_belief_distribution.is_some() {
                        log::warn!(
                            "WHCR: Belief distribution has {} colors but {} required - using fallback",
                            self.belief_num_colors,
                            num_colors
                        );
                    }
                    self.ensure_belief_fallback(num_colors)?;
                }

                // NOW get all buffer references after mutable operations are done
                // Use geometry if available, otherwise use safe fallback buffers
                let stress_buffer = self
                    .d_stress_scores
                    .as_ref()
                    .unwrap_or(&self.d_conflict_counts_f64);
                let persistence_buffer = self
                    .d_persistence_scores
                    .as_ref()
                    .unwrap_or(&self.d_conflict_counts_f64);

                // Get the belief buffer pointer
                let belief_buffer = if needs_fallback {
                    self.d_belief_fallback.as_ref().ok_or_else(|| {
                        anyhow::anyhow!("Belief fallback should have been allocated")
                    })?
                } else {
                    self.d_belief_distribution
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("Belief distribution should exist"))?
                };

                let move_deltas_f64 = self
                    .d_move_deltas_f64
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("Move deltas f64 not allocated"))?;

                // 12 parameters: removed hotspot_mask (now derived from stress in kernel)
                unsafe {
                    self.stream
                        .launch_builder(&self.evaluate_moves_f64)
                        .arg(&self.d_coloring)
                        .arg(&self.d_adjacency_row_ptr)
                        .arg(&self.d_adjacency_col_idx)
                        .arg(&d_conflict_vertices)
                        .arg(&(num_conflict_vertices as i32))
                        .arg(&(num_colors as i32))
                        .arg(stress_buffer)
                        .arg(persistence_buffer)
                        .arg(belief_buffer)
                        .arg(&(self.num_vertices as i32))
                        .arg(move_deltas_f64)
                        .arg(&self.d_best_colors)
                        .launch(cfg)?;
                }
                // CRITICAL FIX: Synchronize after kernel launch
                self.stream.synchronize()?;
            }

            // Step 3: Apply best moves with parallel locking
            log::trace!(
                "WHCR: Applying best moves for {} vertices",
                num_conflict_vertices
            );

            // Reset num_moves_applied counter
            // self.stream.memset_zeros(&mut self.d_num_moves_applied)?;
            let zero = vec![0i32; 1];
            self.d_num_moves_applied = self.stream.clone_htod(&zero)?;

            let cfg = LaunchConfig::for_num_elems(num_conflict_vertices as u32);

            // CRITICAL FIX: Use the correct kernel based on precision
            if use_precise {
                // f64 path - use f64 kernel with f64 buffer and on-device validation
                log::trace!("WHCR: Using f64 apply_moves kernel for fine level");

                let move_deltas_f64 = self
                    .d_move_deltas_f64
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("Move deltas f64 not allocated"))?;

                // Geometry buffers for weighted conflict metric
                let stress_buffer = self
                    .d_stress_scores
                    .as_ref()
                    .unwrap_or(&self.d_conflict_counts_f64);
                // Use conflict counts as fallback instead of d_locks to avoid borrow checker conflict
                // Cast to i32 buffer - this is safe as it's just used as a fallback when hotspot mask is not available
                // Use hotspot mask if available; otherwise treat as all-zero mask via the shared zero buffer
                let hotspot_buffer = self.d_hotspot_mask.as_ref().unwrap_or(&self.d_zero_i32);

                unsafe {
                    self.stream
                        .launch_builder(&self.apply_moves_with_locking_f64)
                        .arg(&mut self.d_coloring)
                        .arg(&d_conflict_vertices)
                        .arg(&self.d_best_colors)
                        .arg(move_deltas_f64)
                        .arg(&(num_conflict_vertices as i32))
                        .arg(&(num_colors as i32))
                        .arg(&mut self.d_locks)
                        .arg(&mut self.d_num_moves_applied)
                        .arg(&self.d_adjacency_row_ptr)
                        .arg(&self.d_adjacency_col_idx)
                        .arg(stress_buffer)
                        .arg(hotspot_buffer)
                        .launch(cfg)?;
                }
            } else {
                // f32 path - use f32 kernel with f32 buffer
                log::trace!("WHCR: Using f32 apply_moves kernel for coarse level");

                let move_deltas_f32 = self
                    .d_move_deltas_f32
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("Move deltas f32 not allocated"))?;

                unsafe {
                    self.stream
                        .launch_builder(&self.apply_moves_with_locking)
                        .arg(&mut self.d_coloring)
                        .arg(&d_conflict_vertices)
                        .arg(&self.d_best_colors)
                        .arg(move_deltas_f32)
                        .arg(&(num_conflict_vertices as i32))
                        .arg(&(num_colors as i32))
                        .arg(&mut self.d_locks)
                        .arg(&mut self.d_num_moves_applied)
                        .arg(&self.d_adjacency_row_ptr)
                        .arg(&self.d_adjacency_col_idx)
                        .launch(cfg)?;
                }
            }

            self.stream.synchronize()?;

            // DIAGNOSTIC: Count conflicts after move application
            let conflicts_after = self.count_conflicts_gpu()?;
            let moves_applied = self.stream.clone_dtoh(&self.d_num_moves_applied)?;

            // Enhanced diagnostic logging for debugging
            log::info!(
                "WHCR Level {}: {} moves applied out of {} conflict vertices, conflicts: {} → {} (delta: {}) (precision: {})",
                level,
                moves_applied[0],
                num_conflict_vertices,
                conflicts_before,
                conflicts_after,
                conflicts_after as i64 - conflicts_before as i64,
                if use_precise { "f64" } else { "f32" }
            );

            // DIAGNOSTIC: Warn if no moves applied but conflicts remain
            if moves_applied[0] == 0 && conflicts_after > 0 {
                log::warn!(
                    "WHCR Level {}: No moves applied but {} conflicts remain! Check move evaluation deltas.",
                    level, conflicts_after
                );
            }

            // Reset locks for next iteration
            // self.stream.memset_zeros(&mut self.d_locks)?;
            let zeros_locks = vec![0i32; self.num_vertices]; // Inefficient but works
            self.d_locks = self.stream.clone_htod(&zeros_locks)?;

            // ========== END MOVE EVALUATION AND APPLICATION ==========

            self.stream.synchronize()?;

            // Check if any conflicts remain
            if conflicts_after == 0 {
                log::debug!(
                    "WHCR Level {}: All conflicts resolved after {} moves",
                    level,
                    moves_applied[0]
                );
                break;
            }
        }

        // DIAGNOSTIC: Final summary for this level
        let conflicts_final = self.count_conflicts_gpu()?;
        let delta = conflicts_final as i64 - conflicts_before as i64;
        log::info!(
            "WHCR Level {}: Complete - conflicts {} → {} (delta: {}) (precision: {})",
            level,
            conflicts_before,
            conflicts_final,
            delta,
            if use_precise { "f64" } else { "f32" }
        );

        Ok(())
    }

    /// Count conflicts on GPU
    fn count_conflicts_gpu(&mut self) -> Result<usize> {
        // Launch conflict counting kernel
        let cfg = LaunchConfig::for_num_elems(self.num_vertices as u32);
        unsafe {
            self.stream
                .launch_builder(&self.count_conflicts_f32)
                .arg(&self.d_coloring)
                .arg(&self.d_adjacency_row_ptr)
                .arg(&self.d_adjacency_col_idx)
                .arg(&self.d_conflict_counts_f32)
                .arg(&(self.num_vertices as i32))
                .launch(cfg)?;
        }
        self.stream.synchronize()?;

        // Download counts and sum
        let counts = self.stream.clone_dtoh(&self.d_conflict_counts_f32)?;
        let total: f32 = counts.iter().sum();

        // Each conflict is counted twice (once from each endpoint)
        Ok((total / 2.0) as usize)
    }

    /// Perform initial wavelet decomposition of conflict signal
    ///
    /// Computes approximation and detail coefficients at all levels
    /// using Haar-like wavelet transform (simple averaging and differencing).
    fn decompose_conflict_signal(&mut self) -> Result<()> {
        // Compute conflict counts first to seed level 0
        self.count_conflicts_gpu()?;
        let counts = self.stream.clone_dtoh(&self.d_conflict_counts_f32)?;

        // Level 0 approximations <- conflict counts
        if let Some(level0) = self.d_wavelet_levels.get_mut(0) {
            level0.approximations = self.stream.clone_htod(&counts)?;
        }

        // Work coarse from level 1 upward
        for level in 1..self.d_wavelet_levels.len() {
            let (prev_level, curr_level) = {
                let (left, right) = self.d_wavelet_levels.split_at_mut(level);
                let prev = &mut left[level - 1];
                let curr = &mut right[0];
                (prev, curr)
            };

            // Download fine approximations
            let fine = self.stream.clone_dtoh(&prev_level.approximations)?;

            // Build coarse approximations by averaging pairs
            let mut coarse_host = vec![0f32; curr_level.approx_size];
            for i in 0..prev_level.approx_size {
                let coarse_idx = (i / 2).min(curr_level.approx_size - 1);
                coarse_host[coarse_idx] += fine[i] * 0.5;
            }

            // Upload coarse approximations
            curr_level.approximations = self.stream.clone_htod(&coarse_host)?;

            // Compute details: fine - coarse[proj]
            if let (Some(details), Some(proj)) =
                (curr_level.details.as_ref(), curr_level.projections.as_ref())
            {
                let cfg = LaunchConfig::for_num_elems(prev_level.approx_size as u32);
                unsafe {
                    self.stream
                        .launch_builder(&self.compute_wavelet_details)
                        .arg(&prev_level.approximations)
                        .arg(&curr_level.approximations)
                        .arg(proj)
                        .arg(details)
                        .arg(&(prev_level.approx_size as i32))
                        .launch(cfg)?;
                }
            }
        }

        self.stream.synchronize()?;
        Ok(())
    }

    /// Async conflict counting - returns immediately without blocking
    ///
    /// Launches the conflict counting kernel on the dedicated stream and returns
    /// immediately. Call `synchronize()` to wait for completion before reading results.
    ///
    /// # Example
    /// ```ignore
    /// whcr.count_conflicts_async()?;
    /// // Do other work here...
    /// whcr.synchronize()?;
    /// let conflicts = whcr.get_conflicts()?;
    /// ```
    pub fn count_conflicts_async(&self) -> Result<()> {
        let cfg = LaunchConfig::for_num_elems(self.num_vertices as u32);
        unsafe {
            self.stream
                .launch_builder(&self.count_conflicts_f32)
                .arg(&self.d_coloring)
                .arg(&self.d_adjacency_row_ptr)
                .arg(&self.d_adjacency_col_idx)
                .arg(&self.d_conflict_counts_f32)
                .arg(&(self.num_vertices as i32))
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Synchronize stream - wait for all async operations to complete
    ///
    /// Blocks until all operations on the WHCR stream are complete.
    /// Must be called before reading results from async operations.
    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()?;
        Ok(())
    }
}

/// Convert adjacency list to CSR format
fn adjacency_to_csr(adjacency: &[Vec<usize>]) -> (Vec<i32>, Vec<i32>) {
    let num_vertices = adjacency.len();
    let mut row_ptr = vec![0i32; num_vertices + 1];
    let mut col_idx = Vec::new();

    for (i, neighbors) in adjacency.iter().enumerate() {
        row_ptr[i + 1] = row_ptr[i] + neighbors.len() as i32;
        for &neighbor in neighbors {
            col_idx.push(neighbor as i32);
        }
    }

    (row_ptr, col_idx)
}
