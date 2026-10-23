//! GPU-Accelerated Bio-Chemistry Feature Extraction
//!
//! This module provides CUDA-accelerated computation of the three bio-chemistry
//! features that were bottlenecking CPU performance:
//!
//! 1. **Hydrophobic Exposure Delta** - SASA weighted by Kyte-Doolittle scale
//! 2. **Local Displacement Anisotropy** - Cα hinge detection
//! 3. **Electrostatic Frustration** - Like-charge proximity stress
//!
//! # Design Principles
//!
//! - **Safety First**: CPU fallback always available
//! - **Numerical Equivalence**: GPU results match CPU within f32 tolerance
//! - **Zero-Copy When Possible**: Reuse GPU buffers from physics simulation

use anyhow::{Context, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// Maximum supported atoms for GPU feature extraction
pub const MAX_ATOMS: usize = 32768;

/// Maximum supported residues
pub const MAX_RESIDUES: usize = 4096;

/// Block size for CUDA kernels
const BLOCK_SIZE: usize = 256;

/// GPU-accelerated bio-chemistry feature extractor
pub struct BiochemistryGpu {
    /// CUDA context
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    /// CUDA stream for async operations
    stream: Arc<CudaStream>,
    /// Compiled CUDA module
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    /// Unified kernel function
    kernel_unified: CudaFunction,

    // === Device Buffers (persistent) ===
    /// Current positions [n_atoms * 3]
    d_positions: CudaSlice<f32>,
    /// Initial positions [n_atoms * 3]
    d_initial_positions: CudaSlice<f32>,
    /// Hydrophobicity per residue [n_residues]
    d_hydrophobicity: CudaSlice<f32>,
    /// Atom-to-residue mapping [n_atoms]
    d_atom_to_residue: CudaSlice<i32>,
    /// Target residue indices [n_targets]
    d_target_residues: CudaSlice<i32>,
    /// Cα atom indices [n_ca]
    d_ca_indices: CudaSlice<i32>,
    /// Partial charges [n_atoms]
    d_charges: CudaSlice<f32>,
    /// Charged atom indices [n_charged]
    d_charged_indices: CudaSlice<i32>,
    /// Output buffer [3]
    d_result: CudaSlice<f32>,

    // === Metadata Counts ===
    n_atoms: usize,
    n_residues: usize,
    n_targets: usize,
    n_ca: usize,
    n_charged: usize,
    neighbor_cutoff: f32,

    /// Whether metadata has been uploaded
    metadata_ready: bool,
}

/// Metadata structure matching CPU AtomicMetadata
pub struct GpuAtomicMetadata {
    /// Hydrophobicity per residue
    pub hydrophobicity: Vec<f32>,
    /// Atom-to-residue mapping
    pub atom_to_residue: Vec<i32>,
    /// Target residue indices
    pub target_residues: Vec<i32>,
    /// Cα atom indices
    pub ca_indices: Vec<i32>,
    /// Partial charges per atom
    pub charges: Vec<f32>,
    /// Indices of charged atoms
    pub charged_indices: Vec<i32>,
}

impl BiochemistryGpu {
    /// Create a new GPU bio-chemistry feature extractor
    ///
    /// # Arguments
    /// * `context` - CUDA context
    /// * `max_atoms` - Maximum number of atoms to support
    /// * `neighbor_cutoff` - Cutoff distance for neighbor counting (Angstroms)
    pub fn new(context: Arc<CudaContext>, max_atoms: usize, neighbor_cutoff: f32) -> Result<Self> {
        anyhow::ensure!(
            max_atoms <= MAX_ATOMS,
            "max_atoms {} exceeds limit {}",
            max_atoms,
            MAX_ATOMS
        );

        let stream = context.default_stream();

        // Load PTX module
        let ptx_path = Self::find_ptx_path()?;
        log::info!("Loading bio-chemistry PTX from: {}", ptx_path);

        let module = context
            .load_module(Ptx::from_file(&ptx_path))
            .context("Failed to load bio-chemistry CUDA module")?;

        // Get kernel function
        let kernel_unified = module
            .load_function("kernel_bio_chemistry_unified")
            .context("Failed to find kernel_bio_chemistry_unified")?;

        // Allocate device buffers
        let max_residues = MAX_RESIDUES;

        let d_positions: CudaSlice<f32> = stream
            .alloc_zeros(max_atoms * 3)
            .context("Failed to allocate d_positions")?;
        let d_initial_positions: CudaSlice<f32> = stream
            .alloc_zeros(max_atoms * 3)
            .context("Failed to allocate d_initial_positions")?;
        let d_hydrophobicity: CudaSlice<f32> = stream
            .alloc_zeros(max_residues)
            .context("Failed to allocate d_hydrophobicity")?;
        let d_atom_to_residue: CudaSlice<i32> = stream
            .alloc_zeros(max_atoms)
            .context("Failed to allocate d_atom_to_residue")?;
        let d_target_residues: CudaSlice<i32> = stream
            .alloc_zeros(max_residues)
            .context("Failed to allocate d_target_residues")?;
        let d_ca_indices: CudaSlice<i32> = stream
            .alloc_zeros(max_residues)
            .context("Failed to allocate d_ca_indices")?;
        let d_charges: CudaSlice<f32> = stream
            .alloc_zeros(max_atoms)
            .context("Failed to allocate d_charges")?;
        let d_charged_indices: CudaSlice<i32> = stream
            .alloc_zeros(max_atoms)
            .context("Failed to allocate d_charged_indices")?;
        let d_result: CudaSlice<f32> = stream
            .alloc_zeros(3)
            .context("Failed to allocate d_result")?;

        log::info!(
            "BiochemistryGpu initialized: max_atoms={}, cutoff={:.1}Å",
            max_atoms,
            neighbor_cutoff
        );

        Ok(Self {
            ctx: context,
            stream,
            module,
            kernel_unified,
            d_positions,
            d_initial_positions,
            d_hydrophobicity,
            d_atom_to_residue,
            d_target_residues,
            d_ca_indices,
            d_charges,
            d_charged_indices,
            d_result,
            n_atoms: 0,
            n_residues: 0,
            n_targets: 0,
            n_ca: 0,
            n_charged: 0,
            neighbor_cutoff,
            metadata_ready: false,
        })
    }

    /// Path of bio_chemistry_features.ptx in the single PTX authority (P0.5).
    fn find_ptx_path() -> Result<String> {
        const MODULE: &str = "bio_chemistry_features.ptx";
        crate::ptx_authority::ptx_file(MODULE)
            .map(|path| path.to_string_lossy().into_owned())
            .map_err(|e| anyhow::anyhow!("governed {MODULE}: {e}"))
    }

    /// Upload atomic metadata to GPU
    ///
    /// This should be called once when the protein changes, not every frame.
    pub fn upload_metadata(&mut self, metadata: &GpuAtomicMetadata) -> Result<()> {
        self.n_residues = metadata.hydrophobicity.len();
        self.n_targets = metadata.target_residues.len();
        self.n_ca = metadata.ca_indices.len();
        self.n_charged = metadata.charged_indices.len();
        self.n_atoms = metadata.atom_to_residue.len();

        anyhow::ensure!(
            self.n_atoms <= MAX_ATOMS,
            "n_atoms {} exceeds MAX_ATOMS {}",
            self.n_atoms,
            MAX_ATOMS
        );

        // Upload to GPU
        self.d_hydrophobicity = self
            .stream
            .clone_htod(&metadata.hydrophobicity)
            .context("Failed to upload hydrophobicity")?;

        self.d_atom_to_residue = self
            .stream
            .clone_htod(&metadata.atom_to_residue)
            .context("Failed to upload atom_to_residue")?;

        self.d_target_residues = self
            .stream
            .clone_htod(&metadata.target_residues)
            .context("Failed to upload target_residues")?;

        self.d_ca_indices = self
            .stream
            .clone_htod(&metadata.ca_indices)
            .context("Failed to upload ca_indices")?;

        self.d_charges = self
            .stream
            .clone_htod(&metadata.charges)
            .context("Failed to upload charges")?;

        self.d_charged_indices = self
            .stream
            .clone_htod(&metadata.charged_indices)
            .context("Failed to upload charged_indices")?;

        self.metadata_ready = true;

        log::debug!(
            "Bio-chemistry metadata uploaded: {} atoms, {} residues, {} targets, {} Cα, {} charged",
            self.n_atoms,
            self.n_residues,
            self.n_targets,
            self.n_ca,
            self.n_charged
        );

        Ok(())
    }

    /// Upload initial positions (reference state)
    ///
    /// Call once at the start of simulation.
    pub fn upload_initial_positions(&mut self, positions: &[f32]) -> Result<()> {
        anyhow::ensure!(
            positions.len() == self.n_atoms * 3,
            "Position length mismatch: expected {}, got {}",
            self.n_atoms * 3,
            positions.len()
        );

        self.d_initial_positions = self
            .stream
            .clone_htod(positions)
            .context("Failed to upload initial positions")?;

        Ok(())
    }

    /// Compute bio-chemistry features from current positions
    ///
    /// # Arguments
    /// * `positions` - Current atom positions [n_atoms * 3] as flat f32 array
    ///
    /// # Returns
    /// `[hydrophobic_exposure, anisotropy, frustration]` normalized to [0, 1]
    pub fn compute(&mut self, positions: &[f32]) -> Result<[f32; 3]> {
        anyhow::ensure!(
            self.metadata_ready,
            "Metadata not uploaded. Call upload_metadata() first."
        );

        anyhow::ensure!(
            positions.len() == self.n_atoms * 3,
            "Position length mismatch: expected {}, got {}",
            self.n_atoms * 3,
            positions.len()
        );

        // Upload current positions
        self.d_positions = self
            .stream
            .clone_htod(positions)
            .context("Failed to upload positions")?;

        // Clear result buffer (upload zeros since memset not available)
        let zeros = vec![0.0f32; 3];
        self.d_result = self
            .stream
            .clone_htod(&zeros)
            .context("Failed to clear result buffer")?;

        // Calculate shared memory size
        let shared_mem_bytes = 3 * BLOCK_SIZE * std::mem::size_of::<f32>();

        // Launch unified kernel
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: shared_mem_bytes as u32,
        };

        let cutoff_sq = self.neighbor_cutoff * self.neighbor_cutoff;
        let n_atoms_i32 = self.n_atoms as i32;
        let n_residues_i32 = self.n_residues as i32;
        let n_targets_i32 = self.n_targets as i32;
        let n_ca_i32 = self.n_ca as i32;
        let n_charged_i32 = self.n_charged as i32;

        unsafe {
            self.stream
                .launch_builder(&self.kernel_unified)
                .arg(&self.d_positions)
                .arg(&self.d_initial_positions)
                .arg(&n_atoms_i32)
                .arg(&self.d_hydrophobicity)
                .arg(&self.d_atom_to_residue)
                .arg(&n_residues_i32)
                .arg(&self.d_target_residues)
                .arg(&n_targets_i32)
                .arg(&self.d_ca_indices)
                .arg(&n_ca_i32)
                .arg(&self.d_charges)
                .arg(&self.d_charged_indices)
                .arg(&n_charged_i32)
                .arg(&cutoff_sq)
                .arg(&mut self.d_result)
                .launch(config)
                .context("Failed to launch bio-chemistry kernel")?;
        }

        // Synchronize and download result
        self.stream.synchronize().context("Stream sync failed")?;

        let result_vec: Vec<f32> = self
            .stream
            .clone_dtoh(&self.d_result)
            .context("Failed to download result")?;

        let result = [
            result_vec.get(0).copied().unwrap_or(0.0),
            result_vec.get(1).copied().unwrap_or(0.0),
            result_vec.get(2).copied().unwrap_or(0.0),
        ];

        Ok(result)
    }

    /// Check if GPU feature extraction is available and initialized
    pub fn is_ready(&self) -> bool {
        self.metadata_ready
    }

    /// Get number of atoms currently configured
    pub fn n_atoms(&self) -> usize {
        self.n_atoms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_metadata_struct() {
        let metadata = GpuAtomicMetadata {
            hydrophobicity: vec![0.5, 0.8, 0.3],
            atom_to_residue: vec![0, 0, 0, 1, 1, 2],
            target_residues: vec![0, 2],
            ca_indices: vec![1, 4],
            charges: vec![0.0, 0.0, -0.5, 0.0, 0.5, 0.0],
            charged_indices: vec![2, 4],
        };

        assert_eq!(metadata.hydrophobicity.len(), 3);
        assert_eq!(metadata.atom_to_residue.len(), 6);
        assert_eq!(metadata.target_residues.len(), 2);
    }
}
