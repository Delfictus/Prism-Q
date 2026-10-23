//! Basic Batch TDA GPU Executor
//!
//! Executes TDA feature extraction on GPU using pre-computed neighborhood data.
//! Uses the hybrid_tda_ultimate.cu kernel for warp-cooperative Betti computation.

use cudarc::driver::{DevicePtrMut, 
    CudaContext, CudaStream, CudaFunction, CudaSlice,
    LaunchConfig, PushKernelArg, DeviceSlice};
use cudarc::nvrtc::Ptx;
use prism_core::PrismError;
use std::path::Path;
use std::sync::Arc;

use super::neighborhood::NeighborhoodData;
use super::{TDA_SCALES, FEATURES_PER_RADIUS, TDA_FEATURE_COUNT};

/// GPU buffers for TDA computation
pub struct TdaBuffers {
    /// Neighborhood offsets
    pub d_offsets: CudaSlice<u32>,
    /// Neighbor indices
    pub d_neighbor_indices: CudaSlice<u32>,
    /// Neighbor distances (F16 as u16)
    pub d_neighbor_distances: CudaSlice<u16>,
    /// Neighbor coordinates (packed xyz)
    pub d_neighbor_coords: CudaSlice<f32>,
    /// Center coordinates
    pub d_center_coords: CudaSlice<f32>,
    /// Output TDA features
    pub d_tda_features: CudaSlice<f32>,
    /// Number of residues
    pub n_residues: usize,
    /// Number of radii
    pub n_radii: usize,
}

impl TdaBuffers {
    /// Allocated capacity
    pub fn capacity(&self) -> usize {
        self.n_residues
    }

    /// Feature output slice
    pub fn features(&self) -> &CudaSlice<f32> {
        &self.d_tda_features
    }
}

/// Basic batch TDA executor
pub struct BatchTdaExecutor {
    /// CUDA context
    ctx: Arc<CudaContext>,
    /// CUDA stream for async execution
    stream: Arc<CudaStream>,
    /// TDA kernel function
    kernel: CudaFunction,
    /// Reusable buffers (capacity-based)
    buffers: Option<TdaBuffers>,
    /// Maximum residue capacity
    max_residues: usize,
    /// TDA scales for persistence computation
    scales: [f32; 4],
}

impl BatchTdaExecutor {
    /// Create a new executor
    ///
    /// Loads the hybrid_tda_ultimate.cu kernel and prepares GPU resources.
    pub fn new(ctx: Arc<CudaContext>, ptx_path: &Path) -> Result<Self, PrismError> {
        let stream = ctx.default_stream();

        // Load PTX
        let ptx_src = std::fs::read_to_string(ptx_path)
            .map_err(|e| PrismError::config(format!("Failed to read TDA PTX: {}", e)))?;
        let ptx = Ptx::from_src(ptx_src);

        // Get kernel function
        let module = ctx.load_module(ptx)
            .map_err(|e| PrismError::config(format!("Failed to load TDA module: {:?}", e)))?;
        let kernel = module.load_function("hybrid_tda_kernel")
            .map_err(|e| PrismError::config(format!("Failed to get TDA kernel: {:?}", e)))?;

        Ok(Self {
            ctx,
            stream,
            kernel,
            buffers: None,
            max_residues: 0,
            scales: TDA_SCALES,
        })
    }

    /// Create with custom scales
    pub fn with_scales(mut self, scales: [f32; 4]) -> Self {
        self.scales = scales;
        self
    }

    /// Ensure buffers are allocated with sufficient capacity
    fn ensure_buffers(&mut self, neighborhood: &NeighborhoodData) -> Result<(), PrismError> {
        let n_residues = neighborhood.n_residues;
        let n_radii = neighborhood.n_radii;
        let total_neighbors = neighborhood.total_neighbors();

        // Check if we need to reallocate
        if self.buffers.is_none() || n_residues > self.max_residues {
            // Calculate sizes with 20% headroom
            let capacity = (n_residues * 12 / 10).max(n_residues);
            let neighbor_capacity = (total_neighbors * 12 / 10).max(total_neighbors);

            // Allocate buffers
            let d_offsets = self.stream.alloc_zeros::<u32>(capacity * n_radii + 1)
                .map_err(|e| PrismError::gpu("executor", format!("Alloc offsets: {:?}", e)))?;
            let d_neighbor_indices = self.stream.alloc_zeros::<u32>(neighbor_capacity)
                .map_err(|e| PrismError::gpu("executor", format!("Alloc indices: {:?}", e)))?;
            let d_neighbor_distances = self.stream.alloc_zeros::<u16>(neighbor_capacity)
                .map_err(|e| PrismError::gpu("executor", format!("Alloc distances: {:?}", e)))?;
            let d_neighbor_coords = self.stream.alloc_zeros::<f32>(neighbor_capacity * 3)
                .map_err(|e| PrismError::gpu("executor", format!("Alloc coords: {:?}", e)))?;
            let d_center_coords = self.stream.alloc_zeros::<f32>(capacity * 3)
                .map_err(|e| PrismError::gpu("executor", format!("Alloc centers: {:?}", e)))?;
            let d_tda_features = self.stream.alloc_zeros::<f32>(capacity * TDA_FEATURE_COUNT)
                .map_err(|e| PrismError::gpu("executor", format!("Alloc features: {:?}", e)))?;

            self.buffers = Some(TdaBuffers {
                d_offsets,
                d_neighbor_indices,
                d_neighbor_distances,
                d_neighbor_coords,
                d_center_coords,
                d_tda_features,
                n_residues: capacity,
                n_radii,
            });
            self.max_residues = capacity;
        }

        Ok(())
    }

    /// Execute TDA feature extraction
    ///
    /// Takes pre-computed neighborhood data and returns TDA features.
    pub fn execute(&mut self, neighborhood: &NeighborhoodData) -> Result<Vec<f32>, PrismError> {
        let n_residues = neighborhood.n_residues;
        if n_residues == 0 {
            return Ok(vec![]);
        }

        // Ensure buffers
        self.ensure_buffers(neighborhood)?;

        // Get mutable reference to buffers
        let buffers = self.buffers.as_mut().unwrap());

        // Upload neighborhood data (copy into pre-allocated buffers)
        buffers.d_offsets = self.stream.clone_htod(neighborhood.offsets))
            .map_err(|e| PrismError::gpu("executor", format!("Upload offsets: {:?}", e)))?;
        buffers.d_neighbor_indices = self.stream.clone_htod(neighborhood.neighbor_indices))
            .map_err(|e| PrismError::gpu("executor", format!("Upload indices: {:?}", e)))?;
        buffers.d_neighbor_distances = self.stream.clone_htod(neighborhood.neighbor_distances))
            .map_err(|e| PrismError::gpu("executor", format!("Upload distances: {:?}", e)))?;
        buffers.d_neighbor_coords = self.stream.clone_htod(neighborhood.neighbor_coords))
            .map_err(|e| PrismError::gpu("executor", format!("Upload coords: {:?}", e)))?;
        buffers.d_center_coords = self.stream.clone_htod(neighborhood.center_coords))
            .map_err(|e| PrismError::gpu("executor", format!("Upload centers: {:?}", e)))?;

        // Launch configuration
        // One warp per residue for warp-cooperative operations
        let threads_per_block = 256;
        let residues_per_block = threads_per_block / 32; // 8 residues per block
        let num_blocks = (n_residues + residues_per_block - 1) / residues_per_block;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Launch kernel
        let n_residues_u32 = n_residues as u32;
        let n_radii_u32 = neighborhood.n_radii as u32;
        unsafe {
            &self.stream.launch_builder(&self.kernel)
                .arg(&buffers.d_offsets)
                .arg(&buffers.d_neighbor_indices)
                .arg(&buffers.d_neighbor_distances)
                .arg(&buffers.d_neighbor_coords)
                .arg(&buffers.d_center_coords)
                .arg(&buffers.d_tda_features)
                .arg(&n_residues_u32)
                .arg(&n_radii_u32)
                .arg(&self.scales[0])
                .arg(&self.scales[1])
                .arg(&self.scales[2])
                .arg(&self.scales[3])
                .launch(cfg).map_err(|e| PrismError::gpu("executor", format!("Launch TDA kernel: {:?}", e)))?;
        }

        // Synchronize and download results
        self.stream.synchronize()
            .map_err(|e| PrismError::gpu("executor", format!("Sync: {:?}", e)))?;

        let output_len = n_residues * TDA_FEATURE_COUNT;
        let output = self.stream.clone_dtoh(&buffers.d_tda_features)
            .map_err(|e| PrismError::gpu("executor", format!("Download features: {:?}", e)))?;

        // Truncate to actual size if needed
        let output = output.into_iter().take(output_len).collect();

        Ok(output)
    }

    /// Execute with output staying on GPU (for pipeline integration)
    ///
    /// Returns a reference to the GPU buffer containing features.
    pub fn execute_gpu(&mut self, neighborhood: &NeighborhoodData) -> Result<&CudaSlice<f32>, PrismError> {
        let n_residues = neighborhood.n_residues;
        if n_residues == 0 {
            return Err(PrismError::validation("Empty neighborhood data");
        }

        // Ensure buffers
        self.ensure_buffers(neighborhood)?;

        // Get mutable reference to buffers for upload
        {
            let buffers = self.buffers.as_mut().unwrap());

            // Upload neighborhood data (atomic copy into pre-allocated buffers)
            self.ctx.memcpy_htod_sync(&neighborhood.offsets, &buffers.d_offsets)
                .map_err(|e| PrismError::gpu("executor", format!("Upload offsets: {:?}", e)))?;
            self.ctx.memcpy_htod_sync(&neighborhood.neighbor_indices, &buffers.d_neighbor_indices)
                .map_err(|e| PrismError::gpu("executor", format!("Upload indices: {:?}", e)))?;
            self.ctx.memcpy_htod_sync(&neighborhood.neighbor_distances, &buffers.d_neighbor_distances)
                .map_err(|e| PrismError::gpu("executor", format!("Upload distances: {:?}", e)))?;
            self.ctx.memcpy_htod_sync(&neighborhood.neighbor_coords, &buffers.d_neighbor_coords)
                .map_err(|e| PrismError::gpu("executor", format!("Upload coords: {:?}", e)))?;
            self.ctx.memcpy_htod_sync(&neighborhood.center_coords, &buffers.d_center_coords)
                .map_err(|e| PrismError::gpu("executor", format!("Upload centers: {:?}", e)))?;
        }

        // Get immutable reference for launch
        let buffers = self.buffers.as_ref().unwrap());

        // Launch kernel
        let threads_per_block = 256;
        let residues_per_block = threads_per_block / 32;
        let num_blocks = (n_residues + residues_per_block - 1) / residues_per_block;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_residues_u32 = n_residues as u32;
        let n_radii_u32 = neighborhood.n_radii as u32;
        unsafe {
            let mut builder = self.stream.launch_builder(&self.kernel);
            builder.arg(&buffers.d_offsets);
            builder.arg(&buffers.d_neighbor_indices);
            builder.arg(&buffers.d_neighbor_distances);
            builder.arg(&buffers.d_neighbor_coords);
            builder.arg(&buffers.d_center_coords);
            builder.arg(&buffers.d_tda_features);
            builder.arg(&n_residues_u32);
            builder.arg(&n_radii_u32);
            builder.arg(&self.scales[0]);
            builder.arg(&self.scales[1]);
            builder.arg(&self.scales[2]);
            builder.arg(&self.scales[3]);
            builder.launch(cfg)
                .map_err(|e| PrismError::gpu("executor", format!("Launch TDA kernel: {:?}", e)))?;
        }

        // Don't sync - let caller decide when to sync
        Ok(&self.buffers.as_ref().unwrap().d_tda_features)
    }

    /// Get stream for synchronization
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Synchronize stream
    pub fn synchronize(&self) -> Result<(), PrismError> {
        self.stream.synchronize()
            .map_err(|e| PrismError::gpu("executor", format!("Sync: {:?}", e)))
    }
}

/// CPU fallback for TDA computation (when GPU unavailable)
pub mod cpu_fallback {
    use super::*;

    /// Compute Betti-0 (connected components) using union-find
    pub fn compute_betti0(n: usize, edges: &[(usize, usize)]) -> usize {
        if n == 0 {
            return 0;
        }

        let mut parent: Vec<usize> = (0..n).collect();
        let mut rank = vec![0usize; n];

        fn find(parent: &mut [usize], x: usize) -> usize {
            if parent[x] != x {
                parent[x] = find(parent, parent[x]);
            }
            parent[x]
        }

        fn union(parent: &mut [usize], rank: &mut [usize], x: usize, y: usize) {
            let rx = find(parent, x);
            let ry = find(parent, y);
            if rx != ry {
                if rank[rx] < rank[ry] {
                    parent[rx] = ry;
                } else if rank[rx] > rank[ry] {
                    parent[ry] = rx;
                } else {
                    parent[ry] = rx;
                    rank[rx] += 1;
                }
            }
        }

        for &(u, v) in edges {
            if u < n && v < n {
                union(&mut parent, &mut rank, u, v);
            }
        }

        // Count unique roots
        let mut components = 0;
        for i in 0..n {
            if find(&mut parent, i) == i {
                components += 1;
            }
        }
        components
    }

    /// Compute Betti-1 (loops) using Euler characteristic: χ = V - E + F
    /// For a simplicial complex: β0 - β1 + β2 = χ
    /// Assuming β2 ≈ 0 for typical protein neighborhoods: β1 ≈ β0 - χ
    pub fn compute_betti1(n: usize, edges: &[(usize, usize)], triangles: &[(usize, usize, usize)]) -> usize {
        let beta0 = compute_betti0(n, edges) as i32;
        let v = n as i32;
        let e = edges.len() as i32;
        let f = triangles.len() as i32;
        let chi = v - e + f;
        (beta0 - chi).max(0) as usize
    }

    /// Extract TDA features for a single residue neighborhood (CPU implementation)
    pub fn extract_tda_features_cpu(
        center: [f32; 3],
        neighbor_coords: &[[f32; 3]],
        neighbor_distances: &[f32],
        scales: &[f32],
    ) -> [f32; FEATURES_PER_RADIUS] {
        let mut features = [0.0f32; FEATURES_PER_RADIUS];
        let n = neighbor_coords.len();

        if n == 0 {
            return features;
        }

        // Compute Betti numbers at each scale
        for (scale_idx, &scale) in scales.iter().enumerate() {
            // Build edges at this scale
            let mut edges = Vec::new();
            for i in 0..n {
                for j in (i + 1)..n {
                    let dx = neighbor_coords[i][0] - neighbor_coords[j][0];
                    let dy = neighbor_coords[i][1] - neighbor_coords[j][1];
                    let dz = neighbor_coords[i][2] - neighbor_coords[j][2];
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    if dist <= scale {
                        edges.push((i, j);
                    }
                }
            }

            // Betti-0
            let beta0 = compute_betti0(n, &edges);
            features[scale_idx] = beta0 as f32;

            // Betti-1 (simplified - count triangles)
            let mut triangles = Vec::new();
            for i in 0..n {
                for j in (i + 1)..n {
                    for k in (j + 1)..n {
                        let has_ij = edges.contains(&(i, j);
                        let has_jk = edges.contains(&(j, k);
                        let has_ik = edges.contains(&(i, k);
                        if has_ij && has_jk && has_ik {
                            triangles.push((i, j, k);
                        }
                    }
                }
            }
            let beta1 = compute_betti1(n, &edges, &triangles);
            features[4 + scale_idx] = beta1 as f32;
        }

        // Persistence features (simplified)
        let mut birth_death = Vec::new();
        for (scale_idx, &scale) in scales.iter().enumerate() {
            if features[scale_idx] > 0.0 {
                // Component born at 0, dies when merged
                for _ in 0..(features[scale_idx] as usize) {
                    birth_death.push((0.0f32, scale);
                }
            }
        }

        if !birth_death.is_empty() {
            let total_persistence: f32 = birth_death.iter()
                .map(|(b, d)| d - b)
                .sum();
            let max_persistence = birth_death.iter()
                .map(|(b, d)| d - b)
                .fold(0.0f32, f32::max);

            features[8] = total_persistence;
            features[9] = max_persistence;

            // Entropy
            let lifetimes: Vec<f32> = birth_death.iter()
                .map(|(b, d)| d - b)
                .collect();
            let total: f32 = lifetimes.iter().sum();
            if total > 0.0 {
                let entropy: f32 = lifetimes.iter()
                    .map(|&l| {
                        let p = l / total;
                        if p > 0.0 { -p * p.ln() } else { 0.0 }
                    })
                    .sum();
                features[10] = entropy;
            }

            // Significant features (persistence > 1.0 Å)
            features[11] = birth_death.iter()
                .filter(|(b, d)| d - b > 1.0)
                .count() as f32;
        }

        // Directional features
        let mut plus_x = 0usize;
        let mut plus_y = 0usize;
        let mut plus_z = 0usize;

        for coord in neighbor_coords {
            if coord[0] > center[0] { plus_x += 1; }
            if coord[1] > center[1] { plus_y += 1; }
            if coord[2] > center[2] { plus_z += 1; }
        }

        let n_f32 = n as f32;
        features[12] = plus_x as f32 / n_f32;
        features[13] = plus_y as f32 / n_f32;
        features[14] = plus_z as f32 / n_f32;

        // Anisotropy (variance in directional densities)
        let dirs = [features[12], features[13], features[14]];
        let mean = dirs.iter().sum::<f32>() / 3.0;
        let variance = dirs.iter()
            .map(|&d| (d - mean) * (d - mean))
            .sum::<f32>() / 3.0;
        features[15] = variance.sqrt();

        features
    }
}

#[cfg(test)]
mod tests {
    use super::cpu_fallback::*;

    #[test]
    fn test_betti0() {
        // 3 separate points
        assert_eq!(compute_betti0(3, &[]), 3);

        // 3 points in a line
        assert_eq!(compute_betti0(3, &[(0, 1), (1, 2)]), 1);

        // Triangle
        assert_eq!(compute_betti0(3, &[(0, 1), (1, 2), (0, 2)]), 1);
    }

    #[test]
    fn test_extract_features() {
        let center = [0.0, 0.0, 0.0];
        let neighbors = vec![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
        ];
        let distances = vec![1.0, 1.0, 1.0, 1.0];
        let scales = [3.0, 5.0, 7.0, 9.0];

        let features = extract_tda_features_cpu(center, &neighbors, &distances, &scales);

        // At scale 3.0, all 4 points should be connected (distance between any two ≤ sqrt(2) ≈ 1.41)
        // Actually distance between (1,0,0) and (-1,0,0) is 2.0, still < 3.0
        assert!(features[0] >= 1.0); // β0 at scale 3.0

        // Directional: half should be in + direction for x, y
        assert!((features[12] - 0.25).abs() < 0.1); // +x (only 1/4)
        assert!((features[13] - 0.25).abs() < 0.1); // +y
        assert!((features[14] - 0.25).abs() < 0.1); // +z
    }
}
