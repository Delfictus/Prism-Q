//! GPU-accelerated Glycan Shield Masking (Stage 0)
//!
//! Implements N-X-S/T sequon detection and 10Å sphere masking entirely on GPU.
//! This eliminates CPU preprocessing and enables zero-copy pipeline execution.
//!
//! ARCHITECT DIRECTIVE: PHASE 1 - GPU GLYCAN SHIELD

use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, CudaSlice, PushKernelArg, DeviceSlice};
use cudarc::nvrtc::Ptx;
use prism_core::PrismError;
use std::sync::Arc;

/// GPU-accelerated glycan masking configuration
#[derive(Clone, Debug)]
pub struct GlycanGpuConfig {
    /// Glycan shield radius in Angstroms (default 10.0)
    pub shield_radius: f32,
    /// Maximum sequons per structure for shared memory allocation
    pub max_sequons: usize,
    /// Block size for kernel launch (default 256)
    pub block_size: usize,
}

impl Default for GlycanGpuConfig {
    fn default() -> Self {
        Self {
            shield_radius: 10.0,
            max_sequons: 64,
            block_size: 256,
        }
    }
}

/// GPU-accelerated glycan shield masking
pub struct GlycanGpu {
    device: Arc<CudaContext>,
    glycan_mask_kernel: CudaFunction,
    count_masked_kernel: CudaFunction,
    extract_sequons_kernel: CudaFunction,
    config: GlycanGpuConfig,
}

/// Input structure coordinates and sequence data
#[repr(C)]
#[derive(Debug, Clone)]
pub struct Float3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

unsafe impl cudarc::driver::DeviceRepr for Float3 {}

impl Float3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn from_slice(coords: &[f32]) -> Vec<Self> {
        coords.chunks(3)
            .map(|chunk| Self::new(chunk[0], chunk[1], chunk[2]))
            .collect()
    }
}

/// Glycan masking results
#[derive(Debug)]
pub struct GlycanMaskResult {
    /// Mask array: 1 = glycan-shielded, 0 = exposed
    pub mask: Vec<u8>,
    /// Number of shielded residues
    pub num_shielded: i32,
    /// Positions of detected sequons
    pub sequon_positions: Vec<i32>,
    /// Number of sequons found
    pub num_sequons: i32,
}

impl GlycanGpu {
    /// Initialize GPU glycan masking with compiled kernels
    pub fn new(device: Arc<CudaContext>, config: GlycanGpuConfig) -> Result<Self, PrismError> {
        // Embed the governed build output, not a separately generated source copy.
        #[cfg(feature = "cuda")]
        let ptx_data = include_str!(concat!(env!("OUT_DIR"), "/ptx/glycan_mask.ptx"));
        #[cfg(not(feature = "cuda"))]
        let ptx_data: &str = return Err(PrismError::gpu(
            "glycan_gpu",
            "GlycanGpu requires prism-gpu's cuda feature".to_string(),
        ));
        let ptx = Ptx::from_src(ptx_data);
        
        let func_names = [
            "glycan_mask_kernel",
            "count_masked_residues",
            "extract_sequon_positions",
        ];

        let module = device.load_module(ptx)
            .map_err(|e| PrismError::gpu("glycan_gpu", format!("Failed to load glycan_mask PTX module: {}", e)))?;

        let glycan_mask_kernel = module.load_function("glycan_mask_kernel")
            .ok_or_else(|| PrismError::gpu("glycan_gpu", "Failed to get glycan_mask_kernel".to_string()))?;

        let count_masked_kernel = module.load_function("count_masked_residues")
            .ok_or_else(|| PrismError::gpu("glycan_gpu", "Failed to get count_masked_residues".to_string()))?;

        let extract_sequons_kernel = module.load_function("extract_sequon_positions")
            .ok_or_else(|| PrismError::gpu("glycan_gpu", "Failed to get extract_sequon_positions".to_string()))?;

        Ok(Self {
            device,
            glycan_mask_kernel,
            count_masked_kernel,
            extract_sequons_kernel,
            config,
        })
    }

    /// Execute glycan masking on protein structure
    pub fn mask_structure(
        &self,
        sequence: &str,
        coords: &[Float3],
        stream: Option<Arc<CudaStream>>,
    ) -> Result<GlycanMaskResult, PrismError> {
        let n_residues = coords.len();

        // Use default stream if none provided
        let default_stream = self.device.default_stream();
        let launch_stream = stream.unwrap_or(default_stream);

        if sequence.len() != n_residues {
            return Err(PrismError::gpu("glycan_gpu",
                "Sequence length must match coordinate count".to_string()));
        }

        // Allocate device memory
        let d_sequence = launch_stream.clone_htod(&sequence.as_bytes().to_vec())
            .map_err(|e| PrismError::gpu("glycan_gpu", format!("Failed to upload sequence: {}", e)))?;

        let d_coords = launch_stream.clone_htod(coords)
            .map_err(|e| PrismError::gpu("glycan_gpu", format!("Failed to upload coordinates: {}", e)))?;

        let mut d_mask = launch_stream.alloc_zeros::<u8>(n_residues)
            .map_err(|e| PrismError::gpu("glycan_gpu", format!("Failed to allocate mask: {}", e)))?;

        // Launch glycan masking kernel
        let grid_size = (n_residues + self.config.block_size - 1) / self.config.block_size;
        let launch_config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (self.config.block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let kernel_args = (
            &d_sequence,
            &d_coords,
            n_residues as i32,
            &mut d_mask,
        );


        unsafe {
            launch_stream.launch_builder(&self.glycan_mask_kernel)
                .arg(&d_sequence)
                .arg(&d_coords)
                .arg(&(n_residues as i32))
                .arg(&mut d_mask)
                .launch(launch_config)
                .map_err(|e| PrismError::gpu("glycan_gpu", format!("Kernel launch failed: {}", e)))?;
        }

        // Count masked residues
        let mut d_count = launch_stream.alloc_zeros::<i32>(1)
            .map_err(|e| PrismError::gpu("glycan_gpu", format!("Failed to allocate count: {}", e)))?;

        let count_args = (&d_mask, &mut d_count, n_residues as i32);

        unsafe {
            launch_stream.launch_builder(&self.count_masked_kernel)
                .arg(&d_mask)
                .arg(&mut d_count)
                .arg(&(n_residues as i32))
                .launch(launch_config)
                .map_err(|e| PrismError::gpu("glycan_gpu", format!("Count kernel failed: {}", e)))?;
        }

        // Extract sequon positions
        let mut d_sequon_positions = launch_stream.alloc_zeros::<i32>(self.config.max_sequons)
            .map_err(|e| PrismError::gpu("glycan_gpu", format!("Failed to allocate sequon positions: {}", e)))?;

        let mut d_num_sequons = launch_stream.alloc_zeros::<i32>(1)
            .map_err(|e| PrismError::gpu("glycan_gpu", format!("Failed to allocate sequon count: {}", e)))?;

        let sequon_args = (
            &d_sequence,
            &mut d_sequon_positions,
            &mut d_num_sequons,
            n_residues as i32,
        );

        unsafe {
            launch_stream.launch_builder(&self.extract_sequons_kernel)
                .arg(&d_sequence)
                .arg(&mut d_sequon_positions)
                .arg(&mut d_num_sequons)
                .arg(&(n_residues as i32))
                .launch(launch_config)
                .map_err(|e| PrismError::gpu("glycan_gpu", format!("Sequon extraction failed: {}", e)))?;
        }

        // Download results
        let mask = launch_stream.clone_dtoh(&d_mask)
            .map_err(|e| PrismError::gpu("glycan_gpu", format!("Failed to download mask: {}", e)))?;

        let num_shielded = launch_stream.clone_dtoh(&d_count)
            .map_err(|e| PrismError::gpu("glycan_gpu", format!("Failed to download count: {}", e)))?[0];

        let sequon_positions = launch_stream.clone_dtoh(&d_sequon_positions)
            .map_err(|e| PrismError::gpu("glycan_gpu", format!("Failed to download sequon positions: {}", e)))?;

        let num_sequons = launch_stream.clone_dtoh(&d_num_sequons)
            .map_err(|e| PrismError::gpu("glycan_gpu", format!("Failed to download sequon count: {}", e)))?[0];

        Ok(GlycanMaskResult {
            mask,
            num_shielded,
            sequon_positions: sequon_positions.into_iter().take(num_sequons as usize).collect(),
            num_sequons,
        })
    }

    /// Get device handle for integration with other GPU modules
    pub fn device(&self) -> Arc<CudaContext> {
        self.device.clone()
    }

    /// Get kernel function for CUDA Graph integration
    pub fn get_glycan_mask_kernel(&self) -> &CudaFunction {
        &self.glycan_mask_kernel
    }
}
