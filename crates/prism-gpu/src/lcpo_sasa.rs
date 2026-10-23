//! GPU LCPO SASA (Linear Combination of Pairwise Overlaps)
//!
//! High-performance solvent accessible surface area calculation for
//! cryptic binding site detection.
//!
//! # Features
//!
//! - **Shared memory tiling**: Reduces global memory bandwidth
//! - **Per-pair early exit**: 9Å cutoff skips ~90% of pairs
//! - **Multi-frame batching**: Process entire trajectory in one launch
//! - **Constant memory LCPO params**: Fast parameter access
//!
//! # Performance
//!
//! | System | Atoms | Frames | Time |
//! |--------|-------|--------|------|
//! | 6LU7 | 4,730 | 100 | < 100ms |
//! | 6M0J | 12,510 | 100 | < 250ms |
//!
//! # Usage
//!
//! ```ignore
//! use prism_gpu::lcpo_sasa::LcpoSasaGpu;
//!
//! let sasa = LcpoSasaGpu::new(context)?;
//!
//! // Single frame
//! let per_atom_sasa = sasa.compute(&positions, &atom_types, None)?;
//!
//! // Batched (multiple frames)
//! let all_sasa = sasa.compute_batched(&trajectory, &atom_types, None, n_frames)?;
//! ```

use anyhow::{Context, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// Tile size (must match CUDA kernel)
const TILE_SIZE: usize = 128;

/// Atom type indices for LCPO parameters
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AtomType {
    CarbonSp3 = 0,
    CarbonSp2 = 1,
    Nitrogen = 2,
    Oxygen = 3,
    Sulfur = 4,
    Phosphorus = 5,
    Hydrogen = 6,
    Other = 7,
}

impl AtomType {
    /// Convert element symbol to atom type
    pub fn from_element(element: &str) -> Self {
        match element.trim().to_uppercase().as_str() {
            "C" => AtomType::CarbonSp3, // Default to sp3
            "N" => AtomType::Nitrogen,
            "O" => AtomType::Oxygen,
            "S" => AtomType::Sulfur,
            "P" => AtomType::Phosphorus,
            "H" => AtomType::Hydrogen,
            _ => AtomType::Other,
        }
    }

    /// Get VDW radius for this atom type (Angstrom)
    pub fn vdw_radius(&self) -> f32 {
        match self {
            AtomType::CarbonSp3 => 1.70,
            AtomType::CarbonSp2 => 1.70,
            AtomType::Nitrogen => 1.55,
            AtomType::Oxygen => 1.52,
            AtomType::Sulfur => 1.80,
            AtomType::Phosphorus => 1.80,
            AtomType::Hydrogen => 1.20,
            AtomType::Other => 1.70,
        }
    }
}

/// GPU-accelerated LCPO SASA calculator
pub struct LcpoSasaGpu {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    // Kernel functions
    lcpo_kernel: CudaFunction,
    lcpo_batched_kernel: CudaFunction,
    sum_kernel: CudaFunction,
    residue_kernel: CudaFunction,
}

/// Result of SASA calculation
#[derive(Debug, Clone)]
pub struct SasaResult {
    /// Per-atom SASA values (Å²)
    pub per_atom: Vec<f32>,
    /// Total SASA (Å²)
    pub total: f32,
    /// Per-residue SASA (Å²), if computed
    pub per_residue: Option<Vec<f32>>,
}

/// Result of batched SASA calculation
#[derive(Debug, Clone)]
pub struct BatchedSasaResult {
    /// Per-atom SASA for each frame: [n_frames][n_atoms]
    pub per_atom: Vec<Vec<f32>>,
    /// Total SASA per frame
    pub totals: Vec<f32>,
}

impl LcpoSasaGpu {
    /// Create new LCPO SASA calculator
    ///
    /// # Arguments
    /// * `context` - CUDA context
    pub fn new(context: Arc<CudaContext>) -> Result<Self> {
        let stream = context.default_stream();

        // Load PTX module
        let ptx_path = concat!(env!("CARGO_MANIFEST_DIR"), "/target/ptx/lcpo_sasa.ptx");
        let ptx_src = std::fs::read_to_string(ptx_path)
            .with_context(|| format!("Failed to read PTX: {}", ptx_path))?;

        let module = context
            .load_module(Ptx::from_src(&ptx_src))
            .context("Failed to load lcpo_sasa PTX module")?;

        // Load kernel functions
        let lcpo_kernel = module
            .load_function("lcpo_sasa_kernel")
            .context("Failed to load lcpo_sasa_kernel")?;

        let lcpo_batched_kernel = module
            .load_function("lcpo_sasa_batched_kernel")
            .context("Failed to load lcpo_sasa_batched_kernel")?;

        let sum_kernel = module
            .load_function("sum_sasa_kernel")
            .context("Failed to load sum_sasa_kernel")?;

        let residue_kernel = module
            .load_function("residue_sasa_atomic_kernel")
            .context("Failed to load residue_sasa_atomic_kernel")?;

        log::info!("[LCPO] GPU SASA calculator initialized");

        Ok(Self {
            context,
            stream,
            lcpo_kernel,
            lcpo_batched_kernel,
            sum_kernel,
            residue_kernel,
        })
    }

    /// Compute SASA for a single structure
    ///
    /// # Arguments
    /// * `positions` - Atom coordinates [n_atoms * 3] (x1, y1, z1, x2, ...)
    /// * `atom_types` - Atom type indices [n_atoms]
    /// * `radii` - Optional custom VDW radii [n_atoms]. If None, uses built-in radii from atom_types.
    ///
    /// # Returns
    /// Per-atom SASA values and total SASA
    pub fn compute(
        &self,
        positions: &[f32],
        atom_types: &[i32],
        radii: Option<&[f32]>,
    ) -> Result<SasaResult> {
        let n_atoms = atom_types.len();
        if positions.len() != n_atoms * 3 {
            anyhow::bail!(
                "Position array length {} doesn't match n_atoms * 3 = {}",
                positions.len(),
                n_atoms * 3
            );
        }

        // Allocate device memory
        let mut d_positions: CudaSlice<f32> = self
            .stream
            .alloc_zeros(n_atoms * 3)
            .context("Failed to allocate d_positions")?;

        let mut d_atom_types: CudaSlice<i32> = self
            .stream
            .alloc_zeros(n_atoms)
            .context("Failed to allocate d_atom_types")?;

        let mut d_sasa: CudaSlice<f32> = self
            .stream
            .alloc_zeros(n_atoms)
            .context("Failed to allocate d_sasa")?;

        // Copy data to device
        self.stream
            .memcpy_htod(positions, &mut d_positions)
            .context("Failed to copy positions to device")?;

        self.stream
            .memcpy_htod(atom_types, &mut d_atom_types)
            .context("Failed to copy atom_types to device")?;

        // Handle radii - if custom radii provided, use them; otherwise generate from atom types
        let radii_vec: Vec<f32> = if let Some(r) = radii {
            r.to_vec()
        } else {
            // Generate radii from atom types
            atom_types
                .iter()
                .map(|&t| {
                    match t {
                        0 | 1 => 1.70, // Carbon
                        2 => 1.55,     // Nitrogen
                        3 => 1.52,     // Oxygen
                        4 | 5 => 1.80, // Sulfur, Phosphorus
                        6 => 1.20,     // Hydrogen
                        _ => 1.70,     // Other
                    }
                })
                .collect()
        };

        let mut d_radii: CudaSlice<f32> = self
            .stream
            .alloc_zeros(n_atoms)
            .context("Failed to allocate d_radii")?;
        self.stream
            .memcpy_htod(&radii_vec, &mut d_radii)
            .context("Failed to copy radii to device")?;

        // Launch kernel
        let block_size = TILE_SIZE;
        let grid_size = (n_atoms + block_size - 1) / block_size;

        let cfg = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_atoms_i32 = n_atoms as i32;

        unsafe {
            let mut builder = self.stream.launch_builder(&self.lcpo_kernel);
            builder.arg(&d_positions);
            builder.arg(&d_atom_types);
            builder.arg(&d_radii);
            builder.arg(&n_atoms_i32);
            builder.arg(&d_sasa);
            builder
                .launch(cfg)
                .context("Failed to launch lcpo_sasa_kernel")?;
        }

        self.stream
            .synchronize()
            .context("Failed to synchronize after SASA kernel")?;

        // Copy results back
        let mut per_atom = vec![0.0f32; n_atoms];
        self.stream
            .memcpy_dtoh(&d_sasa, &mut per_atom)
            .context("Failed to copy SASA results from device")?;

        let total: f32 = per_atom.iter().sum();

        Ok(SasaResult {
            per_atom,
            total,
            per_residue: None,
        })
    }

    /// Compute SASA for multiple frames (trajectory)
    ///
    /// # Arguments
    /// * `positions` - Atom coordinates for all frames [n_frames * n_atoms * 3]
    /// * `atom_types` - Atom type indices [n_atoms] (shared across frames)
    /// * `radii` - Optional custom VDW radii [n_atoms]
    /// * `n_frames` - Number of frames
    ///
    /// # Returns
    /// Per-atom SASA for each frame and total SASA per frame
    pub fn compute_batched(
        &self,
        positions: &[f32],
        atom_types: &[i32],
        radii: Option<&[f32]>,
        n_frames: usize,
    ) -> Result<BatchedSasaResult> {
        let n_atoms = atom_types.len();
        let expected_len = n_frames * n_atoms * 3;

        if positions.len() != expected_len {
            anyhow::bail!(
                "Position array length {} doesn't match n_frames * n_atoms * 3 = {}",
                positions.len(),
                expected_len
            );
        }

        // Allocate device memory
        let mut d_positions: CudaSlice<f32> = self
            .stream
            .alloc_zeros(n_frames * n_atoms * 3)
            .context("Failed to allocate d_positions")?;

        let mut d_atom_types: CudaSlice<i32> = self
            .stream
            .alloc_zeros(n_atoms)
            .context("Failed to allocate d_atom_types")?;

        let mut d_sasa: CudaSlice<f32> = self
            .stream
            .alloc_zeros(n_frames * n_atoms)
            .context("Failed to allocate d_sasa")?;

        let mut d_totals: CudaSlice<f32> = self
            .stream
            .alloc_zeros(n_frames)
            .context("Failed to allocate d_totals")?;

        // Copy data to device
        self.stream
            .memcpy_htod(positions, &mut d_positions)
            .context("Failed to copy positions to device")?;

        self.stream
            .memcpy_htod(atom_types, &mut d_atom_types)
            .context("Failed to copy atom_types to device")?;

        // Handle radii
        let radii_vec: Vec<f32> = if let Some(r) = radii {
            r.to_vec()
        } else {
            atom_types
                .iter()
                .map(|&t| match t {
                    0 | 1 => 1.70,
                    2 => 1.55,
                    3 => 1.52,
                    4 | 5 => 1.80,
                    6 => 1.20,
                    _ => 1.70,
                })
                .collect()
        };

        let mut d_radii: CudaSlice<f32> = self
            .stream
            .alloc_zeros(n_atoms)
            .context("Failed to allocate d_radii")?;
        self.stream
            .memcpy_htod(&radii_vec, &mut d_radii)
            .context("Failed to copy radii to device")?;

        // Launch batched kernel
        // Grid: (atom_blocks, n_frames)
        let block_size = TILE_SIZE;
        let atom_blocks = (n_atoms + block_size - 1) / block_size;

        let cfg = LaunchConfig {
            grid_dim: (atom_blocks as u32, n_frames as u32, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_atoms_i32 = n_atoms as i32;
        let n_frames_i32 = n_frames as i32;

        unsafe {
            let mut builder = self.stream.launch_builder(&self.lcpo_batched_kernel);
            builder.arg(&d_positions);
            builder.arg(&d_atom_types);
            builder.arg(&d_radii);
            builder.arg(&n_atoms_i32);
            builder.arg(&n_frames_i32);
            builder.arg(&d_sasa);
            builder
                .launch(cfg)
                .context("Failed to launch lcpo_sasa_batched_kernel")?;
        }

        // Sum per-atom SASA to get totals
        let sum_cfg = LaunchConfig {
            grid_dim: (n_frames as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            let mut builder = self.stream.launch_builder(&self.sum_kernel);
            builder.arg(&d_sasa);
            builder.arg(&n_atoms_i32);
            builder.arg(&n_frames_i32);
            builder.arg(&d_totals);
            builder
                .launch(sum_cfg)
                .context("Failed to launch sum_sasa_kernel")?;
        }

        self.stream
            .synchronize()
            .context("Failed to synchronize after batched SASA kernel")?;

        // Copy results back
        let mut all_sasa = vec![0.0f32; n_frames * n_atoms];
        self.stream
            .memcpy_dtoh(&d_sasa, &mut all_sasa)
            .context("Failed to copy SASA results from device")?;

        let mut totals = vec![0.0f32; n_frames];
        self.stream
            .memcpy_dtoh(&d_totals, &mut totals)
            .context("Failed to copy totals from device")?;

        // Reshape per_atom into [n_frames][n_atoms]
        let per_atom: Vec<Vec<f32>> = all_sasa
            .chunks(n_atoms)
            .map(|chunk| chunk.to_vec())
            .collect();

        Ok(BatchedSasaResult { per_atom, totals })
    }

    /// Compute per-residue SASA from per-atom SASA
    ///
    /// # Arguments
    /// * `per_atom_sasa` - Per-atom SASA values [n_atoms]
    /// * `residue_map` - Maps each atom to its residue index [n_atoms]
    /// * `n_residues` - Total number of residues
    pub fn aggregate_to_residues(
        &self,
        per_atom_sasa: &[f32],
        residue_map: &[i32],
        n_residues: usize,
    ) -> Result<Vec<f32>> {
        let n_atoms = per_atom_sasa.len();
        if residue_map.len() != n_atoms {
            anyhow::bail!(
                "residue_map length {} doesn't match n_atoms {}",
                residue_map.len(),
                n_atoms
            );
        }

        // Allocate device memory
        let mut d_sasa: CudaSlice<f32> = self
            .stream
            .alloc_zeros(n_atoms)
            .context("Failed to allocate d_sasa")?;

        let mut d_residue_map: CudaSlice<i32> = self
            .stream
            .alloc_zeros(n_atoms)
            .context("Failed to allocate d_residue_map")?;

        let mut d_residue_sasa: CudaSlice<f32> = self
            .stream
            .alloc_zeros(n_residues)
            .context("Failed to allocate d_residue_sasa")?;

        // Copy data to device
        self.stream
            .memcpy_htod(per_atom_sasa, &mut d_sasa)
            .context("Failed to copy sasa to device")?;

        self.stream
            .memcpy_htod(residue_map, &mut d_residue_map)
            .context("Failed to copy residue_map to device")?;

        // Zero the output (already zeros from alloc_zeros, but be explicit)
        let zeros = vec![0.0f32; n_residues];
        self.stream
            .memcpy_htod(&zeros, &mut d_residue_sasa)
            .context("Failed to zero residue_sasa")?;

        // Launch kernel
        let block_size = 256;
        let grid_size = (n_atoms + block_size - 1) / block_size;

        let cfg = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_atoms_i32 = n_atoms as i32;

        unsafe {
            let mut builder = self.stream.launch_builder(&self.residue_kernel);
            builder.arg(&d_sasa);
            builder.arg(&d_residue_map);
            builder.arg(&n_atoms_i32);
            builder.arg(&d_residue_sasa);
            builder
                .launch(cfg)
                .context("Failed to launch residue_sasa_atomic_kernel")?;
        }

        self.stream
            .synchronize()
            .context("Failed to synchronize after residue kernel")?;

        // Copy results back
        let mut residue_sasa = vec![0.0f32; n_residues];
        self.stream
            .memcpy_dtoh(&d_residue_sasa, &mut residue_sasa)
            .context("Failed to copy residue_sasa from device")?;

        Ok(residue_sasa)
    }
}

/// Convert element symbols to atom type indices
pub fn elements_to_atom_types(elements: &[String]) -> Vec<i32> {
    elements
        .iter()
        .map(|e| AtomType::from_element(e) as i32)
        .collect()
}

/// Get default VDW radii from element symbols
pub fn elements_to_radii(elements: &[String]) -> Vec<f32> {
    elements
        .iter()
        .map(|e| AtomType::from_element(e).vdw_radius())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atom_type_from_element() {
        assert_eq!(AtomType::from_element("C"), AtomType::CarbonSp3);
        assert_eq!(AtomType::from_element("N"), AtomType::Nitrogen);
        assert_eq!(AtomType::from_element("O"), AtomType::Oxygen);
        assert_eq!(AtomType::from_element("S"), AtomType::Sulfur);
        assert_eq!(AtomType::from_element("H"), AtomType::Hydrogen);
        assert_eq!(AtomType::from_element("X"), AtomType::Other);
    }

    #[test]
    fn test_vdw_radii() {
        assert!((AtomType::CarbonSp3.vdw_radius() - 1.70).abs() < 0.01);
        assert!((AtomType::Nitrogen.vdw_radius() - 1.55).abs() < 0.01);
        assert!((AtomType::Oxygen.vdw_radius() - 1.52).abs() < 0.01);
        assert!((AtomType::Hydrogen.vdw_radius() - 1.20).abs() < 0.01);
    }

    #[test]
    fn test_elements_to_atom_types() {
        let elements = vec![
            "C".to_string(),
            "N".to_string(),
            "O".to_string(),
            "H".to_string(),
        ];
        let types = elements_to_atom_types(&elements);
        assert_eq!(types, vec![0, 2, 3, 6]);
    }
}
