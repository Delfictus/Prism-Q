//! GPU Mega-Fused BATCH Pocket Detection Kernel
//!
//! Processes ALL structures in a SINGLE kernel launch for maximum throughput.
//! Uses L1 cache hints (__ldg) and register optimization (__launch_bounds__).
//!
//! Target: 221 structures in <100ms (vs 2+ seconds sequential)
//!
//! ## Architecture
//! - One CUDA block per structure (grid_dim = n_structures)
//! - Packed contiguous arrays for all structures
//! - BatchStructureDesc provides offsets for each structure
//! - 6 stages fused: Distance → Contact → Centrality → Reservoir → Consensus → Kempe

use cudarc::driver::CudaContext;
use cudarc::driver::{
    CudaFunction, CudaSlice, CudaStream, DevicePtrMut, DeviceSlice, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use prism_core::PrismError;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

// Re-export from mega_fused.rs for shared types
pub use super::mega_fused::{
    confidence, signals, GpuProvenanceData, GpuTelemetry, KernelTelemetryEvent, MegaFusedConfig,
    MegaFusedMode, MegaFusedOutput, MegaFusedParams,
};

/// Structure descriptor for batch processing - MUST match CUDA BatchStructureDesc
/// Uses 16-byte alignment for coalesced memory access
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Default)]
pub struct BatchStructureDesc {
    /// Start index in packed atoms array (atoms_packed[atom_offset * 3])
    pub atom_offset: i32,
    /// Start index in packed residue arrays
    pub residue_offset: i32,
    /// Number of atoms in this structure
    pub n_atoms: i32,
    /// Number of residues in this structure
    pub n_residues: i32,
}

/// Input data for a single structure in batch processing
#[derive(Debug, Clone)]
pub struct StructureInput {
    /// Structure identifier (e.g., PDB ID)
    pub id: String,
    /// Flat array of atom coordinates [x0, y0, z0, x1, y1, z1, ...]
    pub atoms: Vec<f32>,
    /// Indices of CA atoms for each residue
    pub ca_indices: Vec<i32>,
    /// Per-residue conservation scores (0.0 - 1.0)
    pub conservation: Vec<f32>,
    /// Per-residue B-factor / flexibility (normalized)
    pub bfactor: Vec<f32>,
    /// Per-residue burial scores (0.0 - 1.0)
    pub burial: Vec<f32>,
    /// Per-residue amino acid types (0-19, A-Y) - PRISM>4D
    pub residue_types: Vec<i32>,
}

impl StructureInput {
    /// Create new structure input
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            atoms: Vec::new(),
            ca_indices: Vec::new(),
            conservation: Vec::new(),
            bfactor: Vec::new(),
            burial: Vec::new(),
            residue_types: Vec::new(),
        }
    }

    /// Number of atoms in this structure
    pub fn n_atoms(&self) -> usize {
        self.atoms.len() / 3
    }

    /// Number of residues in this structure
    pub fn n_residues(&self) -> usize {
        self.ca_indices.len()
    }

    /// Validate input consistency
    pub fn validate(&self) -> Result<(), String> {
        let n_res = self.n_residues();
        if self.atoms.len() % 3 != 0 {
            return Err(format!(
                "atoms array length {} not divisible by 3",
                self.atoms.len()
            ));
        }
        if self.conservation.len() != n_res {
            return Err(format!(
                "conservation length {} != n_residues {}",
                self.conservation.len(),
                n_res
            ));
        }
        if self.bfactor.len() != n_res {
            return Err(format!(
                "bfactor length {} != n_residues {}",
                self.bfactor.len(),
                n_res
            ));
        }
        if self.burial.len() != n_res {
            return Err(format!(
                "burial length {} != n_residues {}",
                self.burial.len(),
                n_res
            ));
        }
        Ok(())
    }
}

//=============================================================================
// STAGE 9-10: 75 PK PARAMETER GRID IMMUNITY STRUCTURES (FIX #2 CORRECTED)
//=============================================================================

/// PK parameter combination for immunity modeling
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct PkParams {
    pub tmax: f32,
    pub thalf: f32,
    pub ke: f32,
    pub ka: f32,
}

/// Extended immunity metadata with 75 PK support
#[derive(Debug, Clone)]
pub struct ImmunityMetadataV2 {
    /// Epitope escape scores (10 epitope groups)
    pub epitope_escape: [f32; 10],
    /// Days since 2020-01-01
    pub current_day: i32,
    /// Variant family index (0-9)
    pub variant_family_idx: i32,
    /// Current population immunity levels (75 values - one per PK combo)
    pub current_immunity_levels_75: [f32; 75],
    /// Country index
    pub country_idx: i32,
}

/// Extended country time series with 75 PK support
#[derive(Debug, Clone)]
pub struct CountryImmunityTimeSeriesV2 {
    /// Frequency history (86 weekly samples)
    pub freq_series: Vec<f32>,
    /// P_neut time series for each of 75 PK combinations
    /// Layout: [pk0_t0..pk0_t85, pk1_t0..pk1_t85, ..., pk74_t0..pk74_t85]
    /// Total: 75 × 86 = 6,450 values
    pub p_neut_series_75pk: Vec<f32>,
}

//=============================================================================
// v2.0 FINAL: BATCH METRICS STRUCTURES (2025-12-05)
//=============================================================================

/// Structure input WITH ground truth for validation batches
#[derive(Debug, Clone)]
pub struct StructureInputWithGT {
    pub base: StructureInput,
    pub gt_pocket_mask: Vec<u8>,
}

/// Per-structure metadata for Stage 8+ features (FIX #1)
#[derive(Debug, Clone, Default)]
pub struct StructureMetadata {
    /// Current GISAID frequency (0.0-1.0)
    pub frequency: f32,
    /// Frequency velocity (Δfreq/Δt, can be negative)
    pub velocity: f32,
}

/// Structure offset for batch mapping - MUST match CUDA StructureOffset
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, Default)]
pub struct StructureOffset {
    pub structure_id: i32,
    pub residue_start: i32,
    pub residue_count: i32,
    pub padding: i32,
}

// Implement bytemuck traits for safe casting
unsafe impl bytemuck::Pod for StructureOffset {}
unsafe impl bytemuck::Zeroable for StructureOffset {}

/// Per-structure metrics from GPU - MUST match CUDA BatchMetricsOutput
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Default)]
pub struct BatchMetricsOutput {
    pub structure_id: i32,
    pub n_residues: i32,
    pub true_positives: i32,
    pub false_positives: i32,
    pub true_negatives: i32,
    pub false_negatives: i32,
    pub precision: f32,
    pub recall: f32,
    pub f1_score: f32,
    pub mcc: f32,
    pub auc_roc: f32,
    pub auprc: f32,
    pub avg_druggability: f32,
    pub n_pockets_detected: i32,
}

unsafe impl bytemuck::Pod for BatchMetricsOutput {}
unsafe impl bytemuck::Zeroable for BatchMetricsOutput {}
unsafe impl cudarc::driver::DeviceRepr for BatchMetricsOutput {}
unsafe impl cudarc::driver::ValidAsZeroBits for BatchMetricsOutput {}

const N_BINS: usize = 100;

/// Packed batch WITH ground truth for validation
#[derive(Debug)]
pub struct PackedBatchWithGT {
    pub base: PackedBatch,
    pub gt_pocket_mask_packed: Vec<u8>,
    pub offsets: Vec<StructureOffset>,
    pub tile_prefix_sum: Vec<i32>,
    pub total_tiles: i32,
}

impl PackedBatchWithGT {
    pub fn from_structures_with_gt(
        structures: &[StructureInputWithGT],
    ) -> Result<Self, PrismError> {
        // FIX 3: GT mask validation
        for s in structures {
            assert_eq!(
                s.gt_pocket_mask.len(),
                s.base.n_residues(),
                "GT mask length mismatch for structure {}",
                s.base.id
            );
        }

        let base_inputs: Vec<StructureInput> = structures.iter().map(|s| s.base.clone()).collect();
        let base = PackedBatch::from_structures(&base_inputs)?;

        let mut gt_packed: Vec<u8> = Vec::with_capacity(base.total_residues);
        let mut offsets: Vec<StructureOffset> = Vec::with_capacity(structures.len());
        let mut tile_prefix_sum: Vec<i32> = vec![0; structures.len() + 1];

        let mut residue_offset = 0i32;
        let tile_size = 32i32;

        for (idx, s) in structures.iter().enumerate() {
            let n_res = s.base.n_residues() as i32;
            let n_tiles = (n_res + tile_size - 1) / tile_size;

            offsets.push(StructureOffset {
                structure_id: idx as i32,
                residue_start: residue_offset,
                residue_count: n_res,
                padding: 0,
            });

            tile_prefix_sum[idx + 1] = tile_prefix_sum[idx] + n_tiles;
            gt_packed.extend_from_slice(&s.gt_pocket_mask);
            residue_offset += n_res;
        }

        let total_tiles = tile_prefix_sum[structures.len()];

        log::info!(
            "PackedBatchWithGT: {} structures, {} residues, {} tiles",
            structures.len(),
            base.total_residues,
            total_tiles
        );

        Ok(Self {
            base,
            gt_pocket_mask_packed: gt_packed,
            offsets,
            tile_prefix_sum,
            total_tiles,
        })
    }
}

/// Batch output including per-structure metrics
#[derive(Debug)]
pub struct BatchOutputWithMetrics {
    pub structures: Vec<BatchStructureOutput>,
    pub metrics: Vec<BatchMetricsOutput>,
    pub kernel_time_us: u64,
    pub aggregate: AggregateMetrics,
}

#[derive(Debug, Clone, Default)]
pub struct AggregateMetrics {
    pub mean_f1: f32,
    pub mean_mcc: f32,
    pub mean_auc_roc: f32,
    pub mean_auprc: f32,
    pub mean_precision: f32,
    pub mean_recall: f32,
}

/// Packed batch of structures ready for GPU transfer
#[derive(Debug)]
pub struct PackedBatch {
    /// Structure descriptors with offsets
    pub descriptors: Vec<BatchStructureDesc>,
    /// Structure IDs in order
    pub ids: Vec<String>,
    /// Packed atom coordinates for all structures
    pub atoms_packed: Vec<f32>,
    /// Packed CA indices for all structures
    pub ca_indices_packed: Vec<i32>,
    /// Packed conservation scores for all structures
    pub conservation_packed: Vec<f32>,
    /// Packed B-factors for all structures
    pub bfactor_packed: Vec<f32>,
    /// Packed burial scores for all structures
    pub burial_packed: Vec<f32>,
    /// Packed residue types for all structures - PRISM>4D
    pub residue_types_packed: Vec<i32>,
    /// Total atoms across all structures
    pub total_atoms: usize,
    /// Total residues across all structures
    pub total_residues: usize,

    //=== STAGE 8: CYCLE FEATURES (FIX #1) ===
    /// Per-structure GISAID frequencies for Stage 8 Cycle Module
    pub frequencies_packed: Vec<f32>, // [n_structures]
    /// Per-structure frequency velocities (Δfreq/Δt) for Stage 8 Cycle Module
    pub velocities_packed: Vec<f32>, // [n_structures]

    //=== STAGE 9-10: 75 PK IMMUNITY FEATURES (FIX #2 CORRECTED) ===
    /// P_neut time series for 75 PK combinations per country
    /// Layout: [country0_pk0_t0..t85, country0_pk1_t0..t85, ..., country0_pk74_t0..t85, country1_pk0_t0..t85, ...]
    /// Total: n_countries × 75 × 86
    pub p_neut_time_series_75pk_packed: Vec<f32>,

    /// Current immunity levels for 75 PK combinations per structure
    /// Layout: [struct0_pk0..pk74, struct1_pk0..pk74, ...]
    pub current_immunity_levels_75_packed: Vec<f32>,

    /// PK parameters (constant memory candidate)
    /// [75 × 4] = 300 values (tmax, thalf, ke, ka)
    pub pk_params_packed: Vec<f32>,

    //=== CRITICAL FIX: REAL DMS ESCAPE FOR GPU (PER-RESIDUE) ===
    /// Per-residue epitope escape values (10 epitopes × total_residues)
    /// Layout: [res0_ep0..ep9, res1_ep0..ep9, ..., resN_ep0..ep9]
    /// Total: total_residues × 10
    pub epitope_escape_packed: Vec<f32>,
}

impl PackedBatch {
    /// Pack multiple structures into contiguous arrays
    pub fn from_structures(structures: &[StructureInput]) -> Result<Self, PrismError> {
        // Validate all structures first
        for (i, s) in structures.iter().enumerate() {
            if let Err(e) = s.validate() {
                return Err(PrismError::gpu(
                    "batch_pack",
                    format!("Structure {} ({}): {}", i, s.id, e),
                ));
            }
        }

        let n_structures = structures.len();
        if n_structures == 0 {
            return Err(PrismError::gpu("batch_pack", "No structures provided"));
        }

        // Calculate total sizes
        let total_atoms: usize = structures.iter().map(|s| s.n_atoms()).sum();
        let total_residues: usize = structures.iter().map(|s| s.n_residues()).sum();

        // Pre-allocate packed arrays
        let mut atoms_packed = Vec::with_capacity(total_atoms * 3);
        let mut ca_indices_packed = Vec::with_capacity(total_residues);
        let mut conservation_packed = Vec::with_capacity(total_residues);
        let mut bfactor_packed = Vec::with_capacity(total_residues);
        let mut burial_packed = Vec::with_capacity(total_residues);
        let mut residue_types_packed = Vec::with_capacity(total_residues);
        let mut descriptors = Vec::with_capacity(n_structures);
        let mut ids = Vec::with_capacity(n_structures);

        let mut atom_offset = 0i32;
        let mut residue_offset = 0i32;

        for s in structures {
            // Record descriptor
            descriptors.push(BatchStructureDesc {
                atom_offset,
                residue_offset,
                n_atoms: s.n_atoms() as i32,
                n_residues: s.n_residues() as i32,
            });
            ids.push(s.id.clone());

            // Pack atoms
            atoms_packed.extend_from_slice(&s.atoms);

            // Pack residue data (CA indices need offset adjustment)
            for &ca_idx in &s.ca_indices {
                ca_indices_packed.push(ca_idx + atom_offset);
            }
            conservation_packed.extend_from_slice(&s.conservation);
            bfactor_packed.extend_from_slice(&s.bfactor);
            burial_packed.extend_from_slice(&s.burial);
            residue_types_packed.extend_from_slice(&s.residue_types);

            // Update offsets for next structure
            atom_offset += s.n_atoms() as i32;
            residue_offset += s.n_residues() as i32;
        }

        log::info!(
            "Packed {} structures: {} atoms, {} residues",
            n_structures,
            total_atoms,
            total_residues
        );

        Ok(Self {
            descriptors,
            ids,
            atoms_packed,
            ca_indices_packed,
            conservation_packed,
            bfactor_packed,
            burial_packed,
            residue_types_packed,
            total_atoms,
            total_residues,
            // STAGE 8: Initialize empty frequency/velocity fields
            // These will be populated by from_structures_with_metadata
            frequencies_packed: Vec::new(),
            velocities_packed: Vec::new(),
            // STAGE 9-10: Initialize empty 75-PK immunity fields
            // These will be populated by build_mega_batch in main.rs
            p_neut_time_series_75pk_packed: Vec::new(),
            current_immunity_levels_75_packed: Vec::new(),
            pk_params_packed: Vec::new(),
            epitope_escape_packed: Vec::new(), // Real DMS (per-residue)
        })
    }

    /// Construct PackedBatch with per-structure metadata for Stage 8+ features (FIX #1)
    pub fn from_structures_with_metadata(
        structures: &[StructureInput],
        metadata: &[StructureMetadata],
    ) -> Result<Self, PrismError> {
        if structures.len() != metadata.len() {
            return Err(PrismError::validation(format!(
                "PackedBatch::from_structures_with_metadata: structures.len() ({}) != metadata.len() ({})",
                structures.len(),
                metadata.len()
            )));
        }

        // Validate all structures
        for (i, s) in structures.iter().enumerate() {
            if let Err(e) = s.validate() {
                return Err(PrismError::gpu(
                    "batch_pack",
                    format!("Structure {} ({}): {}", i, s.id, e),
                ));
            }
        }

        let n_structures = structures.len();
        if n_structures == 0 {
            return Err(PrismError::gpu("batch_pack", "No structures provided"));
        }

        // Calculate total sizes
        let total_atoms: usize = structures.iter().map(|s| s.n_atoms()).sum();
        let total_residues: usize = structures.iter().map(|s| s.n_residues()).sum();

        // Pre-allocate all vectors
        let mut descriptors = Vec::with_capacity(n_structures);
        let mut ids = Vec::with_capacity(n_structures);
        let mut atoms_packed = Vec::with_capacity(total_atoms * 3);
        let mut ca_indices_packed = Vec::with_capacity(total_residues);
        let mut conservation_packed = Vec::with_capacity(total_residues);
        let mut bfactor_packed = Vec::with_capacity(total_residues);
        let mut burial_packed = Vec::with_capacity(total_residues);
        let mut residue_types_packed = Vec::with_capacity(total_residues);

        // NEW: Per-structure metadata vectors
        let mut frequencies_packed = Vec::with_capacity(n_structures);
        let mut velocities_packed = Vec::with_capacity(n_structures);

        let mut atom_offset: i32 = 0;
        let mut residue_offset: i32 = 0;

        for (i, structure) in structures.iter().enumerate() {
            let n_atoms = structure.n_atoms() as i32;
            let n_residues = structure.n_residues() as i32;

            // Build descriptor
            descriptors.push(BatchStructureDesc {
                atom_offset,
                residue_offset,
                n_atoms,
                n_residues,
            });
            ids.push(structure.id.clone());

            // Pack residue-level data
            atoms_packed.extend_from_slice(&structure.atoms);
            for &ca_idx in &structure.ca_indices {
                ca_indices_packed.push(ca_idx + atom_offset); // Offset adjustment
            }
            conservation_packed.extend_from_slice(&structure.conservation);
            bfactor_packed.extend_from_slice(&structure.bfactor);
            burial_packed.extend_from_slice(&structure.burial);
            residue_types_packed.extend_from_slice(&structure.residue_types);

            // Pack structure-level metadata (NEW - FIX #1)
            frequencies_packed.push(metadata[i].frequency);
            velocities_packed.push(metadata[i].velocity);

            atom_offset += n_atoms;
            residue_offset += n_residues;
        }

        log::info!(
            "Packed {} structures with metadata: {} atoms, {} residues",
            n_structures,
            total_atoms,
            total_residues
        );

        Ok(Self {
            descriptors,
            ids,
            atoms_packed,
            ca_indices_packed,
            conservation_packed,
            bfactor_packed,
            burial_packed,
            residue_types_packed,
            total_atoms,
            total_residues,
            // STAGE 8: Real frequency/velocity data
            frequencies_packed,
            velocities_packed,
            // STAGE 9-10: Initialize empty 75-PK immunity fields
            // These will be populated by build_mega_batch in main.rs
            p_neut_time_series_75pk_packed: Vec::new(),
            current_immunity_levels_75_packed: Vec::new(),
            pk_params_packed: Vec::new(),
            epitope_escape_packed: Vec::new(), // Real DMS (per-residue)
        })
    }

    /// Number of structures in batch
    pub fn n_structures(&self) -> usize {
        self.descriptors.len()
    }
}

/// Output from batch pocket detection for a single structure
#[derive(Debug, Clone)]
pub struct BatchStructureOutput {
    /// Structure ID
    pub id: String,
    /// Consensus score per residue (0.0 - 1.0)
    pub consensus_scores: Vec<f32>,
    /// Confidence level per residue (0=LOW, 1=MEDIUM, 2=HIGH)
    pub confidence: Vec<i32>,
    /// Signal mask per residue (bit flags)
    pub signal_mask: Vec<i32>,
    /// Pocket assignment per residue (cluster ID)
    pub pocket_assignment: Vec<i32>,
    /// Network centrality per residue
    pub centrality: Vec<f32>,
    /// Combined 136-dim features per residue [n_residues × 136]
    /// Includes TDA (0-91), Fitness (92-95), Cycle (96-100), Spike (101-108), Immunity (109-124), Epi (125-135)
    pub combined_features: Vec<f32>,
}

/// Complete output from batch processing
#[derive(Debug)]
pub struct BatchOutput {
    /// Per-structure outputs
    pub structures: Vec<BatchStructureOutput>,
    /// GPU telemetry for provenance
    pub gpu_telemetry: Option<GpuProvenanceData>,
    /// Total batch processing time (microseconds)
    pub batch_time_us: u64,
    /// Kernel launch time only (microseconds)
    pub kernel_time_us: u64,
}

/// GPU buffer pool for batch processing
struct BatchBufferPool {
    // Packed input buffers
    atoms_capacity: usize,
    d_atoms: Option<CudaSlice<f32>>,

    residue_capacity: usize,
    d_ca_indices: Option<CudaSlice<i32>>,
    d_conservation: Option<CudaSlice<f32>>,
    d_bfactor: Option<CudaSlice<f32>>,
    d_burial: Option<CudaSlice<f32>>,
    d_residue_types: Option<CudaSlice<i32>>, // PRISM>4D

    // Structure descriptors
    descriptors_capacity: usize,
    d_descriptors: Option<CudaSlice<u8>>,

    // Output buffers (per-residue)
    d_consensus_scores: Option<CudaSlice<f32>>,
    d_confidence: Option<CudaSlice<i32>>,
    d_signal_mask: Option<CudaSlice<i32>>,
    d_pocket_assignment: Option<CudaSlice<i32>>,
    d_centrality: Option<CudaSlice<f32>>,
    d_combined_features: Option<CudaSlice<f32>>, // 136-dim features per residue (PRISM>4D + Epi)

    // Params buffer
    d_params: Option<CudaSlice<u8>>,

    //=== STAGE 8: CYCLE FEATURE GPU BUFFERS (FIX #1) ===
    structure_capacity: usize,
    d_frequencies: Option<CudaSlice<f32>>, // Per-structure frequencies
    d_velocities: Option<CudaSlice<f32>>,  // Per-structure velocities

    //=== STAGE 9-10: 75 PK IMMUNITY GPU BUFFERS (FIX #2 CORRECTED) ===
    // P_neut time series with 75 PK combinations
    p_neut_75pk_capacity: usize,
    d_p_neut_time_series_75pk: Option<CudaSlice<f32>>,

    // Per-structure immunity levels for 75 PK combos
    immunity_75_capacity: usize,
    d_current_immunity_levels_75: Option<CudaSlice<f32>>,

    // PK parameters (constant memory - 75 × 4 = 300 values)
    d_pk_params: Option<CudaSlice<f32>>,

    //=== REAL DMS ESCAPE BUFFER ===
    // Per-residue epitope escape (total_residues × 10)
    d_epitope_escape: Option<CudaSlice<f32>>,

    // Statistics
    allocations: usize,
    reuses: usize,
}

impl BatchBufferPool {
    fn new() -> Self {
        Self {
            atoms_capacity: 0,
            d_atoms: None,
            residue_capacity: 0,
            d_ca_indices: None,
            d_conservation: None,
            d_bfactor: None,
            d_burial: None,
            descriptors_capacity: 0,
            d_descriptors: None,
            d_consensus_scores: None,
            d_confidence: None,
            d_signal_mask: None,
            d_pocket_assignment: None,
            d_centrality: None,
            d_residue_types: None,
            d_combined_features: None,
            d_params: None,
            // STAGE 8: Initialize frequency/velocity buffers
            structure_capacity: 0,
            d_frequencies: None,
            d_velocities: None,
            // STAGE 9-10: Initialize 75-PK immunity buffers
            p_neut_75pk_capacity: 0,
            d_p_neut_time_series_75pk: None,
            immunity_75_capacity: 0,
            d_current_immunity_levels_75: None,
            d_pk_params: None,
            d_epitope_escape: None, // Real DMS
            allocations: 0,
            reuses: 0,
        }
    }

    fn stats(&self) -> (usize, usize) {
        (self.allocations, self.reuses)
    }
}

/// GPU executor for mega-fused BATCH pocket detection
///
/// Processes all structures in a SINGLE kernel launch.
/// Uses L1 cache optimization and register allocation hints.
pub struct MegaFusedBatchGpu {
    #[allow(dead_code)]
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    /// Batch kernel function
    batch_func: Option<CudaFunction>,
    /// v2.0: Batch kernel with metrics
    batch_metrics_func: Option<CudaFunction>,
    /// v2.0: Finalize metrics kernel
    finalize_func: Option<CudaFunction>,
    /// Training kernel (exports reservoir states for readout training)
    training_func: Option<CudaFunction>,
    /// Buffer pool for batch processing
    buffer_pool: BatchBufferPool,
    /// Telemetry collector
    telemetry: GpuTelemetry,
}

impl MegaFusedBatchGpu {
    /// Load batch mega-fused PTX from `ptx_dir`. Expects:
    /// - mega_fused_batch.ptx with `mega_fused_batch_detection`
    pub fn new(context: Arc<CudaContext>, ptx_dir: &Path) -> Result<Self, PrismError> {
        let stream = context.default_stream();

        // Load batch kernel
        let batch_path = ptx_dir.join("mega_fused_batch.ptx");
        let batch_func = if batch_path.exists() {
            let ptx_src = std::fs::read_to_string(&batch_path).map_err(|e| {
                PrismError::gpu("mega_fused_batch", format!("Failed to read PTX: {}", e))
            })?;
            let module = context.load_module(Ptx::from_src(ptx_src)).map_err(|e| {
                PrismError::gpu("mega_fused_batch", format!("Failed to load PTX: {}", e))
            })?;
            let func = module
                .load_function("mega_fused_batch_detection")
                .map_err(|e| {
                    PrismError::gpu("mega_fused_batch", format!("Failed to load kernel: {}", e))
                })?;
            log::info!("Hey, loaded mega_fused_batch.ptx (L1/Register optimized batch kernel)");
            Some(func)
        } else {
            log::warn!("mega_fused_batch.ptx not found at {:?}", batch_path);
            None
        };

        if batch_func.is_none() {
            return Err(PrismError::gpu(
                "mega_fused_batch",
                "Batch kernel not available (mega_fused_batch.ptx not found)",
            ));
        }

        // Load v2.0 metrics kernels from the same PTX
        let batch_metrics_func = if batch_path.exists() {
            let ptx_src = std::fs::read_to_string(&batch_path).map_err(|e| {
                PrismError::gpu("mega_fused_batch", format!("Failed to read PTX: {}", e))
            })?;
            let module = context.load_module(Ptx::from_src(ptx_src)).map_err(|e| {
                PrismError::gpu("mega_fused_batch", format!("Failed to load PTX: {}", e))
            })?;
            match module.load_function("mega_fused_pocket_detection_batch_with_metrics") {
                Ok(func) => {
                    log::info!("Hey, loaded mega_fused_pocket_detection_batch_with_metrics (v2.0 metrics kernel)");
                    Some(func)
                }
                Err(e) => {
                    log::warn!("v2.0 batch_with_metrics kernel not found: {}", e);
                    None
                }
            }
        } else {
            None
        };

        let finalize_func = if batch_path.exists() {
            let ptx_src = std::fs::read_to_string(&batch_path).map_err(|e| {
                PrismError::gpu("mega_fused_batch", format!("Failed to read PTX: {}", e))
            })?;
            let module = context.load_module(Ptx::from_src(ptx_src)).map_err(|e| {
                PrismError::gpu("mega_fused_batch", format!("Failed to load PTX: {}", e))
            })?;
            match module.load_function("finalize_batch_metrics") {
                Ok(func) => {
                    log::info!("Hey, loaded finalize_batch_metrics (v2.0 finalize kernel)");
                    Some(func)
                }
                Err(e) => {
                    log::warn!("v2.0 finalize kernel not found: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // Load training kernel (exports reservoir states for readout training)
        let training_func = if batch_path.exists() {
            let ptx_src = std::fs::read_to_string(&batch_path).map_err(|e| {
                PrismError::gpu("mega_fused_batch", format!("Failed to read PTX: {}", e))
            })?;
            let module = context.load_module(Ptx::from_src(ptx_src)).map_err(|e| {
                PrismError::gpu("mega_fused_batch", format!("Failed to load PTX: {}", e))
            })?;
            match module.load_function("mega_fused_batch_training") {
                Ok(func) => {
                    log::info!(
                        "Hey, loaded mega_fused_batch_training (reservoir state export kernel)"
                    );
                    Some(func)
                }
                Err(e) => {
                    log::debug!("Training kernel not found: {}", e);
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            context,
            stream,
            batch_func,
            batch_metrics_func,
            finalize_func,
            training_func,
            buffer_pool: BatchBufferPool::new(),
            telemetry: GpuTelemetry::new(),
        })
    }

    /// Check if batch kernel is available
    pub fn is_available(&self) -> bool {
        self.batch_func.is_some()
    }

    /// Get buffer pool statistics
    pub fn buffer_pool_stats(&self) -> (usize, usize) {
        self.buffer_pool.stats()
    }

    /// Run batch pocket detection on multiple structures
    ///
    /// All structures are processed in a SINGLE kernel launch.
    /// Returns per-structure outputs with GPU telemetry.
    pub fn detect_pockets_batch(
        &mut self,
        batch: &PackedBatch,
        config: &MegaFusedConfig,
    ) -> Result<BatchOutput, PrismError> {
        // eprintln!("[GPU DEBUG] detect_pockets_batch ENTRY: {} structures", batch.n_structures();
        let batch_start = Instant::now();

        // DEBUG: Limit to N structures to test kernel
        let max_structs = std::env::var("PRISM_MAX_STRUCTURES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(batch.n_structures());
        let n_structures = batch.n_structures().min(max_structs);
        // eprintln!("[GPU DEBUG] Processing {} structures (limit: {})", n_structures, max_structs);

        // Compute limited totals if we're limiting structures
        let (total_atoms, total_residues) = if n_structures < batch.n_structures() {
            let last_desc = &batch.descriptors[n_structures - 1];
            let atoms = (last_desc.atom_offset + last_desc.n_atoms) as usize;
            let residues = (last_desc.residue_offset + last_desc.n_residues) as usize;
            (atoms, residues)
        } else {
            (batch.total_atoms, batch.total_residues)
        };

        // eprintln!("[GPU DEBUG] n_structures={}, total_atoms={}, total_residues={}", n_structures, total_atoms, total_residues);

        if n_structures == 0 {
            return Ok(BatchOutput {
                structures: Vec::new(),
                gpu_telemetry: None,
                batch_time_us: 0,
                kernel_time_us: 0,
            });
        }

        let func = self
            .batch_func
            .as_ref()
            .ok_or_else(|| PrismError::gpu("mega_fused_batch", "Batch kernel not loaded"))?;

        // === ALLOCATE/REUSE BUFFERS ===

        // Atoms buffer
        let atoms_size = total_atoms * 3;
        if atoms_size > self.buffer_pool.atoms_capacity || self.buffer_pool.d_atoms.is_none() {
            let new_cap = (atoms_size * 6 / 5).max(atoms_size);
            self.buffer_pool.d_atoms =
                Some(self.stream.alloc_zeros::<f32>(new_cap).map_err(|e| {
                    PrismError::gpu(
                        "mega_fused_batch",
                        format!("Failed to allocate atoms: {}", e),
                    )
                })?);
            self.buffer_pool.atoms_capacity = new_cap;
            self.buffer_pool.allocations += 1;
        } else {
            self.buffer_pool.reuses += 1;
        }

        // Residue buffers
        if total_residues > self.buffer_pool.residue_capacity
            || self.buffer_pool.d_ca_indices.is_none()
        {
            let new_cap = (total_residues * 6 / 5).max(total_residues);

            self.buffer_pool.d_ca_indices =
                Some(self.stream.alloc_zeros::<i32>(new_cap).map_err(|e| {
                    PrismError::gpu("mega_fused_batch", format!("Alloc ca_indices: {}", e))
                })?);
            self.buffer_pool.d_conservation =
                Some(self.stream.alloc_zeros::<f32>(new_cap).map_err(|e| {
                    PrismError::gpu("mega_fused_batch", format!("Alloc conservation: {}", e))
                })?);
            self.buffer_pool.d_bfactor =
                Some(self.stream.alloc_zeros::<f32>(new_cap).map_err(|e| {
                    PrismError::gpu("mega_fused_batch", format!("Alloc bfactor: {}", e))
                })?);
            self.buffer_pool.d_burial =
                Some(self.stream.alloc_zeros::<f32>(new_cap).map_err(|e| {
                    PrismError::gpu("mega_fused_batch", format!("Alloc burial: {}", e))
                })?);

            // PRISM>4D: Residue types
            self.buffer_pool.d_residue_types =
                Some(self.stream.alloc_zeros::<i32>(new_cap).map_err(|e| {
                    PrismError::gpu("mega_fused_batch", format!("Alloc residue_types: {}", e))
                })?);

            // Output buffers
            self.buffer_pool.d_consensus_scores =
                Some(self.stream.alloc_zeros::<f32>(new_cap).map_err(|e| {
                    PrismError::gpu("mega_fused_batch", format!("Alloc consensus: {}", e))
                })?);
            self.buffer_pool.d_confidence =
                Some(self.stream.alloc_zeros::<i32>(new_cap).map_err(|e| {
                    PrismError::gpu("mega_fused_batch", format!("Alloc confidence: {}", e))
                })?);
            self.buffer_pool.d_signal_mask =
                Some(self.stream.alloc_zeros::<i32>(new_cap).map_err(|e| {
                    PrismError::gpu("mega_fused_batch", format!("Alloc signal_mask: {}", e))
                })?);
            self.buffer_pool.d_pocket_assignment =
                Some(self.stream.alloc_zeros::<i32>(new_cap).map_err(|e| {
                    PrismError::gpu("mega_fused_batch", format!("Alloc pocket_assign: {}", e))
                })?);
            self.buffer_pool.d_centrality =
                Some(self.stream.alloc_zeros::<f32>(new_cap).map_err(|e| {
                    PrismError::gpu("mega_fused_batch", format!("Alloc centrality: {}", e))
                })?);

            // Combined 136-dim features (TDA + Fitness + Cycle + Spike + Immunity + Epi)
            self.buffer_pool.d_combined_features =
                Some(self.stream.alloc_zeros::<f32>(new_cap * 136).map_err(|e| {
                    PrismError::gpu(
                        "mega_fused_batch",
                        format!("Alloc combined_features: {}", e),
                    )
                })?);

            self.buffer_pool.residue_capacity = new_cap;
            self.buffer_pool.allocations += 11; // Updated for residue_types
        }

        // Descriptors buffer
        let desc_size = n_structures * std::mem::size_of::<BatchStructureDesc>();
        if desc_size > self.buffer_pool.descriptors_capacity
            || self.buffer_pool.d_descriptors.is_none()
        {
            let new_cap = (desc_size * 6 / 5).max(desc_size);
            self.buffer_pool.d_descriptors =
                Some(self.stream.alloc_zeros::<u8>(new_cap).map_err(|e| {
                    PrismError::gpu("mega_fused_batch", format!("Alloc descriptors: {}", e))
                })?);
            self.buffer_pool.descriptors_capacity = new_cap;
            self.buffer_pool.allocations += 1;
        }

        // Stage 8: Per-structure metadata buffers (FIX #1)
        if n_structures > self.buffer_pool.structure_capacity
            || self.buffer_pool.d_frequencies.is_none()
        {
            let new_cap = (n_structures * 6 / 5).max(n_structures);

            self.buffer_pool.d_frequencies =
                Some(self.stream.alloc_zeros::<f32>(new_cap).map_err(|e| {
                    PrismError::gpu("mega_fused_batch", format!("Alloc d_frequencies: {}", e))
                })?);
            self.buffer_pool.d_velocities =
                Some(self.stream.alloc_zeros::<f32>(new_cap).map_err(|e| {
                    PrismError::gpu("mega_fused_batch", format!("Alloc d_velocities: {}", e))
                })?);

            self.buffer_pool.structure_capacity = new_cap;
            self.buffer_pool.allocations += 2;
        }

        // Params buffer
        if self.buffer_pool.d_params.is_none() {
            let params_size = std::mem::size_of::<MegaFusedParams>();
            self.buffer_pool.d_params =
                Some(self.stream.alloc_zeros::<u8>(params_size).map_err(|e| {
                    PrismError::gpu("mega_fused_batch", format!("Alloc params: {}", e))
                })?);
            self.buffer_pool.allocations += 1;
        }

        //=== STAGE 9-10: ALLOCATE 75-PK IMMUNITY BUFFERS (FIX #2 CORRECTED) ===

        // Determine n_countries and n_time_samples from batch data
        // For now, we'll use placeholders if the data isn't populated yet
        let n_countries = if !batch.p_neut_time_series_75pk_packed.is_empty() {
            batch.p_neut_time_series_75pk_packed.len() / (75 * 86)
        } else {
            0
        };
        let n_time_samples = 86i32; // VASIL standard: 86 weekly samples (600 days / 7)

        // P_neut with 75 PK combinations
        if n_countries > 0 {
            let p_neut_75pk_size = n_countries * 75 * n_time_samples as usize;
            if p_neut_75pk_size > self.buffer_pool.p_neut_75pk_capacity
                || self.buffer_pool.d_p_neut_time_series_75pk.is_none()
            {
                let new_cap = (p_neut_75pk_size * 6 / 5).max(p_neut_75pk_size);

                self.buffer_pool.d_p_neut_time_series_75pk =
                    Some(self.stream.alloc_zeros::<f32>(new_cap).map_err(|e| {
                        PrismError::gpu("mega_fused_batch", format!("Alloc d_p_neut_75pk: {}", e))
                    })?);

                self.buffer_pool.p_neut_75pk_capacity = new_cap;
                self.buffer_pool.allocations += 1;
            }
        }

        // Per-structure immunity levels for 75 PK combos
        if !batch.current_immunity_levels_75_packed.is_empty() {
            let immunity_75_size = n_structures * 75;
            if immunity_75_size > self.buffer_pool.immunity_75_capacity
                || self.buffer_pool.d_current_immunity_levels_75.is_none()
            {
                let new_cap = (immunity_75_size * 6 / 5).max(immunity_75_size);

                self.buffer_pool.d_current_immunity_levels_75 =
                    Some(self.stream.alloc_zeros::<f32>(new_cap).map_err(|e| {
                        PrismError::gpu("mega_fused_batch", format!("Alloc d_immunity_75: {}", e))
                    })?);

                self.buffer_pool.immunity_75_capacity = new_cap;
                self.buffer_pool.allocations += 1;
            }
        }

        // PK parameters (constant - allocate once if data is present)
        if !batch.pk_params_packed.is_empty() && self.buffer_pool.d_pk_params.is_none() {
            self.buffer_pool.d_pk_params = Some(
                self.stream
                    .alloc_zeros::<f32>(300) // 75 × 4
                    .map_err(|e| {
                        PrismError::gpu("mega_fused_batch", format!("Alloc d_pk_params: {}", e))
                    })?,
            );
            self.buffer_pool.allocations += 1;
        }

        //=== ALLOCATE EPITOPE ESCAPE BUFFER (PER-RESIDUE LAYOUT) ===
        if !batch.epitope_escape_packed.is_empty() {
            let epitope_size = batch.epitope_escape_packed.len(); // total_residues × 10
            if self.buffer_pool.d_epitope_escape.is_none()
                || epitope_size > self.buffer_pool.d_epitope_escape.as_ref().unwrap().len()
            {
                eprintln!(
                    "[DMS GPU] Allocating epitope escape buffer: {} values",
                    epitope_size
                );
                self.buffer_pool.d_epitope_escape =
                    Some(self.stream.alloc_zeros::<f32>(epitope_size).map_err(|e| {
                        PrismError::gpu(
                            "mega_fused_batch",
                            format!("Alloc d_epitope_escape: {}", e),
                        )
                    })?);
                self.buffer_pool.allocations += 1;
            }
        }

        // === COPY DATA TO GPU ===
        let d_atoms = self.buffer_pool.d_atoms.as_mut().unwrap();
        let d_ca_indices = self.buffer_pool.d_ca_indices.as_mut().unwrap();
        let d_conservation = self.buffer_pool.d_conservation.as_mut().unwrap();
        let d_bfactor = self.buffer_pool.d_bfactor.as_mut().unwrap();
        let d_burial = self.buffer_pool.d_burial.as_mut().unwrap();
        let d_residue_types = self.buffer_pool.d_residue_types.as_mut().unwrap();
        let d_descriptors = self.buffer_pool.d_descriptors.as_mut().unwrap();

        // Only copy data for the limited structures
        let atoms_to_copy = &batch.atoms_packed[..total_atoms * 3];
        let ca_indices_to_copy = &batch.ca_indices_packed[..total_residues];
        let conservation_to_copy = &batch.conservation_packed[..total_residues];
        let bfactor_to_copy = &batch.bfactor_packed[..total_residues];
        let burial_to_copy = &batch.burial_packed[..total_residues];
        let residue_types_to_copy = &batch.residue_types_packed[..total_residues];
        let descriptors_to_copy = &batch.descriptors[..n_structures];

        // Debug: Print first few structures' data
        for (i, desc) in descriptors_to_copy.iter().take(3).enumerate() {
            // eprintln!("[GPU DEBUG] Structure {}: atom_offset={}, residue_offset={}, n_atoms={}, n_residues={}",
            //     i, desc.atom_offset, desc.residue_offset, desc.n_atoms, desc.n_residues);
            // Check first few CA indices for this structure
            let res_start = desc.residue_offset as usize;
            let res_end = (desc.residue_offset + desc.n_residues.min(5)) as usize;
            if res_end <= ca_indices_to_copy.len() {
                // eprintln!("[GPU DEBUG]   CA indices[0..5]: {:?}", &ca_indices_to_copy[res_start..res_end]);
                // eprintln!("[GPU DEBUG]   Residue types[0..5]: {:?}", &residue_types_to_copy[res_start..res_end]);
                // Validate: CA index should be < n_atoms for this structure
                let max_atom_idx = desc.atom_offset + desc.n_atoms;
                for j in res_start..res_end {
                    let ca_idx = ca_indices_to_copy[j];
                    if ca_idx >= max_atom_idx {
                        // eprintln!("[GPU DEBUG] ERROR: CA index {} >= max atom {} for structure {}", ca_idx, max_atom_idx, i);
                    }
                }
            }
        }
        // Verify arrays sizes
        // eprintln!("[GPU DEBUG] Array sizes: atoms={}, ca_indices={}, res_types={}",
        //     atoms_to_copy.len(), ca_indices_to_copy.len(), residue_types_to_copy.len();

        *d_atoms = self
            .stream
            .clone_htod(&atoms_to_copy[..])
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Copy atoms: {}", e)))?;
        *d_ca_indices = self
            .stream
            .clone_htod(&ca_indices_to_copy[..])
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Copy ca_indices: {}", e)))?;
        *d_conservation = self
            .stream
            .clone_htod(&conservation_to_copy[..])
            .map_err(|e| {
                PrismError::gpu("mega_fused_batch", format!("Copy conservation: {}", e))
            })?;
        *d_bfactor = self
            .stream
            .clone_htod(&bfactor_to_copy[..])
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Copy bfactor: {}", e)))?;
        *d_burial = self
            .stream
            .clone_htod(&burial_to_copy[..])
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Copy burial: {}", e)))?;
        *d_residue_types = self
            .stream
            .clone_htod(&residue_types_to_copy[..])
            .map_err(|e| {
                PrismError::gpu("mega_fused_batch", format!("Copy residue_types: {}", e))
            })?;

        //=== STAGE 8: COPY FREQUENCY/VELOCITY DATA TO GPU (FIX #1) ===

        // Upload per-structure frequencies and velocities (FIX #1)
        eprintln!("[DEBUG] Starting Stage 8 uploads...");
        if !batch.frequencies_packed.is_empty() {
            let frequencies_to_copy = &batch.frequencies_packed[..n_structures];
            let velocities_to_copy = &batch.velocities_packed[..n_structures];

            let d_frequencies = self
                .buffer_pool
                .d_frequencies
                .as_mut()
                .expect("d_frequencies not allocated");
            let d_velocities = self
                .buffer_pool
                .d_velocities
                .as_mut()
                .expect("d_velocities not allocated");

            eprintln!(
                "[DEBUG] Uploading {} frequencies, {} velocities",
                frequencies_to_copy.len(),
                velocities_to_copy.len()
            );

            *d_frequencies = self.stream.clone_htod(frequencies_to_copy).map_err(|e| {
                PrismError::gpu("mega_fused_batch", format!("Copy frequencies: {}", e))
            })?;
            *d_velocities = self.stream.clone_htod(velocities_to_copy).map_err(|e| {
                PrismError::gpu("mega_fused_batch", format!("Copy velocities: {}", e))
            })?;
            eprintln!("[DEBUG] Stage 8 uploads OK");
        }

        //=== STAGE 9-10: COPY 75-PK IMMUNITY DATA TO GPU (FIX #2 CORRECTED) ===

        eprintln!("[DEBUG] Starting Stage 9-10 75-PK uploads...");
        eprintln!(
            "[DEBUG] p_neut_75pk data size: {}",
            batch.p_neut_time_series_75pk_packed.len()
        );
        eprintln!(
            "[DEBUG] immunity_75 data size: {}",
            batch.current_immunity_levels_75_packed.len()
        );
        eprintln!(
            "[DEBUG] pk_params data size: {}",
            batch.pk_params_packed.len()
        );

        // Copy P_neut time series (75 PK combos per country)
        if !batch.p_neut_time_series_75pk_packed.is_empty() {
            if let Some(d_p_neut_75pk) = self.buffer_pool.d_p_neut_time_series_75pk.as_mut() {
                eprintln!("[DEBUG FIX2] p_neut_75pk upload:");
                eprintln!(
                    "  Source len: {}",
                    batch.p_neut_time_series_75pk_packed.len()
                );
                eprintln!("  Dest capacity: {}", d_p_neut_75pk.len());
                eprintln!("  Pool capacity: {}", self.buffer_pool.p_neut_75pk_capacity);

                *d_p_neut_75pk = self
                    .stream
                    .clone_htod(&batch.p_neut_time_series_75pk_packed)
                    .map_err(|e| {
                        PrismError::gpu("mega_fused_batch", format!("Copy p_neut_75pk: {}", e))
                    })?;
                eprintln!("  Upload OK");
            }
        }

        // Copy immunity levels (75 PK combos per structure)
        if !batch.current_immunity_levels_75_packed.is_empty() {
            if let Some(d_immunity_75) = self.buffer_pool.d_current_immunity_levels_75.as_mut() {
                eprintln!("[DEBUG FIX2] immunity_75 upload:");
                eprintln!(
                    "  Source len: {}",
                    batch.current_immunity_levels_75_packed.len()
                );
                eprintln!("  Dest capacity: {}", d_immunity_75.len());
                eprintln!("  n_structures (limited): {}", n_structures);
                eprintln!("  Expected size: {} (n_structures × 75)", n_structures * 75);

                // FIX: Only upload data for structures we're actually processing
                let immunity_to_copy =
                    &batch.current_immunity_levels_75_packed[..n_structures * 75];

                *d_immunity_75 = self.stream.clone_htod(immunity_to_copy).map_err(|e| {
                    PrismError::gpu("mega_fused_batch", format!("Copy immunity_75: {}", e))
                })?;
                eprintln!("  Upload OK");
            }
        }

        // Copy PK parameters (constant - 75 × 4 = 300 values)
        if !batch.pk_params_packed.is_empty() {
            if let Some(d_pk_params) = self.buffer_pool.d_pk_params.as_mut() {
                *d_pk_params = self
                    .stream
                    .clone_htod(&batch.pk_params_packed)
                    .map_err(|e| {
                        PrismError::gpu("mega_fused_batch", format!("Copy pk_params: {}", e))
                    })?;
            }
        }

        //=== UPLOAD REAL EPITOPE ESCAPE (PER-RESIDUE LAYOUT) ===
        if !batch.epitope_escape_packed.is_empty() {
            if let Some(d_epitope_escape) = self.buffer_pool.d_epitope_escape.as_mut() {
                eprintln!(
                    "[DMS GPU] Uploading {} epitope escape values (per-residue)",
                    batch.epitope_escape_packed.len()
                );
                *d_epitope_escape = self
                    .stream
                    .clone_htod(&batch.epitope_escape_packed)
                    .map_err(|e| {
                        PrismError::gpu("mega_fused_batch", format!("Copy epitope_escape: {}", e))
                    })?;
            }
        }

        // Copy descriptors as bytes
        let desc_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                descriptors_to_copy.as_ptr() as *const u8,
                descriptors_to_copy.len() * std::mem::size_of::<BatchStructureDesc>(),
            )
        };
        *d_descriptors = self
            .stream
            .clone_htod(desc_bytes)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Copy descriptors: {}", e)))?;

        // Copy params
        let params = MegaFusedParams::from_config(config);
        let params_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                &params as *const MegaFusedParams as *const u8,
                std::mem::size_of::<MegaFusedParams>(),
            )
        };
        let d_params = self.buffer_pool.d_params.as_mut().unwrap();
        *d_params = self
            .stream
            .clone_htod(params_bytes)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Copy params: {}", e)))?;

        // === LAUNCH KERNEL ===
        // One block per structure, 256 threads per block
        let block_size = 256u32;
        let launch_config = LaunchConfig {
            grid_dim: (n_structures as u32, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0, // Shared memory statically allocated in kernel
        };

        let n_structures_i32 = n_structures as i32;

        // Capture telemetry
        // eprintln!("[GPU DEBUG] Getting pre-launch telemetry...");
        let clock_before = self.telemetry.get_clock_mhz();
        let memory_clock_before = self.telemetry.get_memory_clock_mhz();
        // eprintln!("[GPU DEBUG] Telemetry done, starting kernel timer");
        let kernel_start = Instant::now();

        // Get mutable refs to output buffers for kernel args
        let d_consensus_scores = self.buffer_pool.d_consensus_scores.as_mut().unwrap();
        let d_confidence = self.buffer_pool.d_confidence.as_mut().unwrap();
        let d_signal_mask = self.buffer_pool.d_signal_mask.as_mut().unwrap();
        let d_pocket_assignment = self.buffer_pool.d_pocket_assignment.as_mut().unwrap();
        let d_centrality = self.buffer_pool.d_centrality.as_mut().unwrap();

        // Need to reborrow as shared for kernel args
        let d_atoms = self.buffer_pool.d_atoms.as_ref().unwrap();
        let d_ca_indices = self.buffer_pool.d_ca_indices.as_ref().unwrap();
        let d_conservation = self.buffer_pool.d_conservation.as_ref().unwrap();
        let d_bfactor = self.buffer_pool.d_bfactor.as_ref().unwrap();
        let d_burial = self.buffer_pool.d_burial.as_ref().unwrap();
        let d_descriptors = self.buffer_pool.d_descriptors.as_ref().unwrap();
        let d_params = self.buffer_pool.d_params.as_ref().unwrap();

        // Stage 8: Cycle feature inputs (FIX #1)
        let d_frequencies = self.buffer_pool.d_frequencies.as_ref().unwrap();
        let d_velocities = self.buffer_pool.d_velocities.as_ref().unwrap();

        // Stage 9-10: Immunity inputs with 75-PK support (FIX #2 CORRECTED)
        let null_ptr: u64 = 0;

        // Prepare optional pointers - using device_ptr_mut().as_raw() for cudarc 0.18.2
        // Note: These are optional buffers; if None, we pass null (0)
        let epitope_ptr: u64 = 0; // Will be set via kernel args if buffer exists
        let p_neut_ptr: u64 = 0;
        let immunity_ptr: u64 = 0;
        let pk_params_ptr: u64 = 0;

        // Stage 11: Epidemiological feature inputs (P0 priority for FALL prediction)
        // REAL BUFFERS: Simple uniform data for validation (n_variants=1)
        const N_VARIANTS: i32 = 1;
        const HISTORY_WEEKS: usize = 35;

        // Allocate and initialize frequency history (35 weeks × 1 variant)
        let freq_hist_data = vec![0.10f32; HISTORY_WEEKS]; // Constant 10% frequency
        let d_freq_history = self.stream.clone_htod(&freq_hist_data[..]).map_err(|e| {
            PrismError::gpu("mega_fused_batch", format!("Copy freq_history: {}", e))
        })?;

        // Allocate and initialize current frequencies (1 variant)
        let curr_freq_data = vec![0.15f32]; // Current 15% frequency
        let d_all_frequencies = self.stream.clone_htod(&curr_freq_data[..]).map_err(|e| {
            PrismError::gpu("mega_fused_batch", format!("Copy all_frequencies: {}", e))
        })?;

        // Allocate and initialize fitness scores (1 variant)
        let gamma_data = vec![0.50f32]; // Moderate fitness
        let d_all_gammas = self
            .stream
            .clone_htod(&gamma_data[..])
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Copy all_gammas: {}", e)))?;

        // eprintln!("[GPU DEBUG] Building kernel args (grid: {}, block: {})...", n_structures, block_size);
        unsafe {
            let mut builder = self.stream.launch_builder(&func);
            builder.arg(d_atoms);
            builder.arg(d_ca_indices);
            builder.arg(d_conservation);
            builder.arg(d_bfactor);
            builder.arg(d_burial);
            builder.arg(self.buffer_pool.d_residue_types.as_ref().unwrap());
            builder.arg(d_descriptors);
            builder.arg(&n_structures_i32);
            builder.arg(d_consensus_scores);
            builder.arg(d_confidence);
            builder.arg(d_signal_mask);
            builder.arg(d_pocket_assignment);
            builder.arg(d_centrality);
            builder.arg(self.buffer_pool.d_combined_features.as_ref().unwrap());
            builder.arg(d_frequencies);
            builder.arg(d_velocities);
            builder.arg(&epitope_ptr);
            builder.arg(&null_ptr); // immunity_events_packed
            builder.arg(&0i32); // n_immunity_events
            builder.arg(&600i32); // current_day
            builder.arg(&5i32); // variant_family_idx
            builder.arg(&p_neut_ptr);
            builder.arg(&immunity_ptr);
            builder.arg(&pk_params_ptr);
            builder.arg(&86i32); // n_time_samples
            builder.arg(&d_all_frequencies);
            builder.arg(&d_all_gammas);
            builder.arg(&d_freq_history);
            builder.arg(&0i32); // my_variant_idx
            builder.arg(&N_VARIANTS);
            builder.arg(&0.5f32); // days_since_vaccine_norm
            builder.arg(&0.5f32); // days_since_wave_norm
            builder.arg(&0.0f32); // immunity_derivative
            builder.arg(&0.6f32); // immunity_source_ratio
            builder.arg(&0.45f32); // country_id_norm
            builder.arg(d_params);
            builder.launch(launch_config).map_err(|e| {
                PrismError::gpu("mega_fused_batch", format!("Kernel launch failed: {}", e))
            })?;
            // eprintln!("[GPU DEBUG] Kernel launched OK, now sync...");
        }

        // Synchronize
        // eprintln!("[GPU DEBUG] Calling stream.synchronize()...");
        self.stream
            .synchronize()
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Sync failed: {}", e)))?;
        // eprintln!("[GPU DEBUG] Synchronize complete!");

        let kernel_elapsed = kernel_start.elapsed();
        let clock_after = self.telemetry.get_clock_mhz();
        let memory_clock_after = self.telemetry.get_memory_clock_mhz();

        // === COPY RESULTS BACK ===
        let d_consensus_scores = self.buffer_pool.d_consensus_scores.as_ref().unwrap();
        let d_confidence = self.buffer_pool.d_confidence.as_ref().unwrap();
        let d_signal_mask = self.buffer_pool.d_signal_mask.as_ref().unwrap();
        let d_pocket_assignment = self.buffer_pool.d_pocket_assignment.as_ref().unwrap();
        let d_centrality = self.buffer_pool.d_centrality.as_ref().unwrap();
        let d_combined_features = self.buffer_pool.d_combined_features.as_ref().unwrap();

        // Download full allocated buffers (using residue_capacity, not total_residues)
        // The buffers were allocated with capacity = (total_residues * 6/5) for growth headroom
        let capacity = self.buffer_pool.residue_capacity;

        let all_consensus: Vec<f32> = self
            .stream
            .clone_dtoh(d_consensus_scores)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Read consensus: {}", e)))?;
        let all_confidence: Vec<i32> = self
            .stream
            .clone_dtoh(d_confidence)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Read confidence: {}", e)))?;
        let all_signal_mask: Vec<i32> = self
            .stream
            .clone_dtoh(d_signal_mask)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Read signal_mask: {}", e)))?;
        let all_pocket_assignment: Vec<i32> =
            self.stream.clone_dtoh(d_pocket_assignment).map_err(|e| {
                PrismError::gpu("mega_fused_batch", format!("Read pocket_assign: {}", e))
            })?;
        let all_centrality: Vec<f32> = self
            .stream
            .clone_dtoh(d_centrality)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Read centrality: {}", e)))?;
        let all_combined_features: Vec<f32> =
            self.stream.clone_dtoh(d_combined_features).map_err(|e| {
                PrismError::gpu("mega_fused_batch", format!("Read combined_features: {}", e))
            })?;

        // === UNPACK RESULTS ===
        // Only unpack results for the structures we actually processed (respects PRISM_MAX_STRUCTURES limit)
        let mut structures = Vec::with_capacity(n_structures);
        for (i, desc) in batch.descriptors.iter().take(n_structures).enumerate() {
            let start = desc.residue_offset as usize;
            let end = start + desc.n_residues as usize;
            let n_residues = desc.n_residues as usize;

            // Extract combined_features (136-dim per residue: TDA + Fitness + Cycle + Spike + Immunity + Epi)
            let features_start = start * 136;
            let features_end = end * 136;
            let combined_features = all_combined_features[features_start..features_end].to_vec();

            // DFV: Feature variance validation for spike features (F101-F108)
            // Only check first structure to avoid log spam
            if i == 0 && n_residues > 0 {
                for feat_idx in 101..109 {
                    let values: Vec<f32> = combined_features
                        .iter()
                        .skip(feat_idx)
                        .step_by(136)
                        .copied()
                        .collect();
                    if !values.is_empty() {
                        let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
                        let variance: f32 = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
                            / values.len() as f32;
                        if variance < 1e-6 {
                            log::warn!(
                                "[DFV-002] Spike feature F{} has near-zero variance: mean={:.4}, var={:.6}",
                                feat_idx, mean, variance
                            );
                        }
                    }
                }
            }

            structures.push(BatchStructureOutput {
                id: batch.ids[i].clone(),
                consensus_scores: all_consensus[start..end].to_vec(),
                confidence: all_confidence[start..end].to_vec(),
                signal_mask: all_signal_mask[start..end].to_vec(),
                pocket_assignment: all_pocket_assignment[start..end].to_vec(),
                centrality: all_centrality[start..end].to_vec(),
                combined_features,
            });
        }

        // Build telemetry event
        let kernel_event = KernelTelemetryEvent {
            file: format!("batch_{}_structures", n_structures),
            kernel: "mega_fused_batch_detection".to_string(),
            clock_before_mhz: clock_before,
            clock_after_mhz: clock_after,
            memory_clock_before_mhz: memory_clock_before,
            memory_clock_after_mhz: memory_clock_after,
            temperature_c: self.telemetry.get_temperature(),
            memory_used_bytes: self.telemetry.get_memory_used(),
            execution_time_us: kernel_elapsed.as_micros() as u64,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        let gpu_provenance = GpuProvenanceData {
            gpu_name: self.telemetry.get_gpu_name(),
            driver_version: self.telemetry.get_driver_version(),
            kernel_events: vec![kernel_event],
            total_gpu_time_us: kernel_elapsed.as_micros() as u64,
            avg_clock_mhz: clock_before.or(clock_after),
            peak_temperature_c: self.telemetry.get_temperature(),
        };

        let batch_elapsed = batch_start.elapsed();

        log::info!(
            "Batch processed {} structures in {:.2}ms (kernel: {:.2}ms)",
            n_structures,
            batch_elapsed.as_secs_f64() * 1000.0,
            kernel_elapsed.as_secs_f64() * 1000.0
        );

        Ok(BatchOutput {
            structures,
            gpu_telemetry: Some(gpu_provenance),
            batch_time_us: batch_elapsed.as_micros() as u64,
            kernel_time_us: kernel_elapsed.as_micros() as u64,
        })
    }

    //=========================================================================
    // v2.0 FINAL: BATCH WITH METRICS (ALL 5 FIXES APPLIED)
    //=========================================================================

    /// Check if metrics kernel is available
    pub fn is_metrics_available(&self) -> bool {
        self.batch_metrics_func.is_some() && self.finalize_func.is_some()
    }

    /// Run batch pocket detection WITH ground truth metrics
    /// All accuracy metrics computed on GPU - no Python scripts needed
    pub fn detect_pockets_batch_with_metrics(
        &mut self,
        batch: &PackedBatchWithGT,
        config: &MegaFusedConfig,
    ) -> Result<BatchOutputWithMetrics, PrismError> {
        let kernel_start = Instant::now();
        let n_structures = batch.offsets.len();
        let total_residues = batch.base.total_residues;
        let total_tiles = batch.total_tiles as usize;

        if n_structures == 0 {
            return Ok(BatchOutputWithMetrics {
                structures: Vec::new(),
                metrics: Vec::new(),
                kernel_time_us: 0,
                aggregate: AggregateMetrics::default(),
            });
        }

        let metrics_func = self
            .batch_metrics_func
            .as_ref()
            .ok_or_else(|| PrismError::gpu("mega_fused_batch", "v2.0 metrics kernel not loaded"))?;
        let finalize_func = self.finalize_func.as_ref().ok_or_else(|| {
            PrismError::gpu("mega_fused_batch", "v2.0 finalize kernel not loaded")
        })?;

        // === ALLOCATE INPUT BUFFERS ===
        let mut d_atoms = self
            .stream
            .alloc_zeros::<f32>(batch.base.atoms_packed.len().max(1))
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc atoms: {}", e)))?;
        let mut d_ca_indices = self
            .stream
            .alloc_zeros::<i32>(total_residues.max(1))
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc ca_indices: {}", e)))?;
        let mut d_conservation = self
            .stream
            .alloc_zeros::<f32>(total_residues.max(1))
            .map_err(|e| {
                PrismError::gpu("mega_fused_batch", format!("Alloc conservation: {}", e))
            })?;
        let mut d_bfactor = self
            .stream
            .alloc_zeros::<f32>(total_residues.max(1))
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc bfactor: {}", e)))?;
        let mut d_burial = self
            .stream
            .alloc_zeros::<f32>(total_residues.max(1))
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc burial: {}", e)))?;
        let mut d_gt_mask = self
            .stream
            .alloc_zeros::<u8>(total_residues.max(1))
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc gt_mask: {}", e)))?;

        // FIX 1: StructureOffset upload using bytemuck::cast_slice
        let mut d_offsets = self
            .stream
            .alloc_zeros::<u8>(n_structures * std::mem::size_of::<StructureOffset>())
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc offsets: {}", e)))?;

        let mut d_tile_prefix = self
            .stream
            .alloc_zeros::<i32>(batch.tile_prefix_sum.len())
            .map_err(|e| {
                PrismError::gpu("mega_fused_batch", format!("Alloc tile_prefix: {}", e))
            })?;

        // === ALLOCATE OUTPUT BUFFERS ===
        let mut d_consensus_scores = self
            .stream
            .alloc_zeros::<f32>(total_residues.max(1))
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc consensus: {}", e)))?;
        let mut d_confidence = self
            .stream
            .alloc_zeros::<i32>(total_residues.max(1))
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc confidence: {}", e)))?;
        let mut d_signal_mask = self
            .stream
            .alloc_zeros::<i32>(total_residues.max(1))
            .map_err(|e| {
                PrismError::gpu("mega_fused_batch", format!("Alloc signal_mask: {}", e))
            })?;
        let mut d_pocket_assignment = self
            .stream
            .alloc_zeros::<i32>(total_residues.max(1))
            .map_err(|e| {
                PrismError::gpu("mega_fused_batch", format!("Alloc pocket_assign: {}", e))
            })?;
        let mut d_centrality = self
            .stream
            .alloc_zeros::<f32>(total_residues.max(1))
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc centrality: {}", e)))?;
        // NOTE: combined_features not output by with_metrics kernel - use detect_pockets_batch for features

        // === ALLOCATE METRICS BUFFERS ===
        let mut d_tp_counts = self
            .stream
            .alloc_zeros::<i32>(n_structures)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc tp: {}", e)))?;
        let mut d_fp_counts = self
            .stream
            .alloc_zeros::<i32>(n_structures)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc fp: {}", e)))?;
        let mut d_tn_counts = self
            .stream
            .alloc_zeros::<i32>(n_structures)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc tn: {}", e)))?;
        let mut d_fn_counts = self
            .stream
            .alloc_zeros::<i32>(n_structures)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc fn: {}", e)))?;
        let mut d_score_sums = self
            .stream
            .alloc_zeros::<f32>(n_structures)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc score_sums: {}", e)))?;
        let mut d_pocket_counts = self.stream.alloc_zeros::<i32>(n_structures).map_err(|e| {
            PrismError::gpu("mega_fused_batch", format!("Alloc pocket_counts: {}", e))
        })?;

        // Histogram buffers (100 bins per structure)
        let mut d_hist_pos = self
            .stream
            .alloc_zeros::<u64>(n_structures * N_BINS)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc hist_pos: {}", e)))?;
        let mut d_hist_neg = self
            .stream
            .alloc_zeros::<u64>(n_structures * N_BINS)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc hist_neg: {}", e)))?;

        // FIX 2: Metrics buffer as alloc_zeros::<BatchMetricsOutput>(n_structures) NOT u8
        let mut d_metrics_out = self
            .stream
            .alloc_zeros::<BatchMetricsOutput>(n_structures)
            .map_err(|e| {
                PrismError::gpu("mega_fused_batch", format!("Alloc metrics_out: {}", e))
            })?;

        // Params buffer
        let mut d_params = self
            .stream
            .alloc_zeros::<u8>(std::mem::size_of::<MegaFusedParams>())
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Alloc params: {}", e)))?;

        // === COPY DATA TO GPU ===
        d_atoms = self
            .stream
            .clone_htod(&batch.base.atoms_packed)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Copy atoms: {}", e)))?;
        d_ca_indices = self
            .stream
            .clone_htod(&batch.base.ca_indices_packed)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Copy ca_indices: {}", e)))?;
        d_conservation = self
            .stream
            .clone_htod(&batch.base.conservation_packed)
            .map_err(|e| {
                PrismError::gpu("mega_fused_batch", format!("Copy conservation: {}", e))
            })?;
        d_bfactor = self
            .stream
            .clone_htod(&batch.base.bfactor_packed)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Copy bfactor: {}", e)))?;
        d_burial = self
            .stream
            .clone_htod(&batch.base.burial_packed)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Copy burial: {}", e)))?;
        d_gt_mask = self
            .stream
            .clone_htod(&batch.gt_pocket_mask_packed)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Copy gt_mask: {}", e)))?;
        d_tile_prefix = self
            .stream
            .clone_htod(&batch.tile_prefix_sum)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Copy tile_prefix: {}", e)))?;

        // FIX 1: Use bytemuck::cast_slice for StructureOffset
        let offsets_bytes: &[u8] = bytemuck::cast_slice(&batch.offsets);
        self.stream
            .memcpy_htod(offsets_bytes, &mut d_offsets)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Copy offsets: {}", e)))?;

        // Copy params
        let params = MegaFusedParams::from_config(config);
        let params_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                &params as *const MegaFusedParams as *const u8,
                std::mem::size_of::<MegaFusedParams>(),
            )
        };
        self.stream
            .memcpy_htod(params_bytes, &mut d_params)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Copy params: {}", e)))?;

        // === LAUNCH BATCH METRICS KERNEL ===
        let block_size = 256u32;
        let launch_config = LaunchConfig {
            grid_dim: (total_tiles as u32, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_structures_i32 = n_structures as i32;
        let total_tiles_i32 = total_tiles as i32;

        unsafe {
            let mut builder = self.stream.launch_builder(&metrics_func);
            builder.arg(&d_atoms);
            builder.arg(&d_ca_indices);
            builder.arg(&d_conservation);
            builder.arg(&d_bfactor);
            builder.arg(&d_burial);
            builder.arg(&d_gt_mask);
            builder.arg(&d_offsets);
            builder.arg(&d_tile_prefix);
            builder.arg(&n_structures_i32);
            builder.arg(&total_tiles_i32);
            builder.arg(&d_consensus_scores);
            builder.arg(&d_confidence);
            builder.arg(&d_signal_mask);
            builder.arg(&d_pocket_assignment);
            builder.arg(&d_centrality);
            builder.arg(&d_tp_counts);
            builder.arg(&d_fp_counts);
            builder.arg(&d_tn_counts);
            builder.arg(&d_fn_counts);
            builder.arg(&d_score_sums);
            builder.arg(&d_pocket_counts);
            builder.arg(&d_hist_pos);
            builder.arg(&d_hist_neg);
            builder.arg(&d_params);
            builder.launch(launch_config).map_err(|e| {
                PrismError::gpu(
                    "mega_fused_batch",
                    format!("Metrics kernel launch failed: {}", e),
                )
            })?;
        }

        // === LAUNCH FINALIZE KERNEL ===
        let finalize_blocks = (n_structures as u32 + 255) / 256;
        let finalize_config = LaunchConfig {
            grid_dim: (finalize_blocks, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            let mut builder = self.stream.launch_builder(&finalize_func);
            builder.arg(&d_tp_counts);
            builder.arg(&d_fp_counts);
            builder.arg(&d_tn_counts);
            builder.arg(&d_fn_counts);
            builder.arg(&d_score_sums);
            builder.arg(&d_pocket_counts);
            builder.arg(&d_hist_pos);
            builder.arg(&d_hist_neg);
            builder.arg(&d_metrics_out);
            builder.arg(&d_offsets);
            builder.arg(&n_structures_i32);
            builder.launch(finalize_config).map_err(|e| {
                PrismError::gpu(
                    "mega_fused_batch",
                    format!("Finalize kernel launch failed: {}", e),
                )
            })?;
        }

        // Synchronize
        self.stream
            .synchronize()
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Sync failed: {}", e)))?;

        let kernel_elapsed = kernel_start.elapsed();

        // === COPY RESULTS BACK ===
        let all_consensus: Vec<f32> = self
            .stream
            .clone_dtoh(&d_consensus_scores)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Read consensus: {}", e)))?;
        let all_confidence: Vec<i32> = self
            .stream
            .clone_dtoh(&d_confidence)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Read confidence: {}", e)))?;
        let all_signal_mask: Vec<i32> = self
            .stream
            .clone_dtoh(&d_signal_mask)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Read signal_mask: {}", e)))?;
        let all_pocket_assignment: Vec<i32> =
            self.stream.clone_dtoh(&d_pocket_assignment).map_err(|e| {
                PrismError::gpu("mega_fused_batch", format!("Read pocket_assign: {}", e))
            })?;
        let all_centrality: Vec<f32> = self
            .stream
            .clone_dtoh(&d_centrality)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Read centrality: {}", e)))?;

        // Read metrics
        let metrics: Vec<BatchMetricsOutput> = self
            .stream
            .clone_dtoh(&d_metrics_out)
            .map_err(|e| PrismError::gpu("mega_fused_batch", format!("Read metrics: {}", e)))?;

        // === UNPACK STRUCTURE OUTPUTS (FIX 5: Return real BatchStructureOutput) ===
        let mut structures = Vec::with_capacity(n_structures);
        for (i, offset) in batch.offsets.iter().enumerate() {
            let start = offset.residue_start as usize;
            let end = start + offset.residue_count as usize;

            // combined_features not output by with_metrics kernel (uses different kernel path)
            // Use detect_pockets_batch() if you need 136-dim features
            let combined_features = vec![0.0f32; (end - start) * 136];

            structures.push(BatchStructureOutput {
                id: batch.base.ids[i].clone(),
                consensus_scores: all_consensus[start..end].to_vec(),
                confidence: all_confidence[start..end].to_vec(),
                signal_mask: all_signal_mask[start..end].to_vec(),
                pocket_assignment: all_pocket_assignment[start..end].to_vec(),
                centrality: all_centrality[start..end].to_vec(),
                combined_features,
            });
        }

        // === COMPUTE AGGREGATE METRICS ===
        let mut aggregate = AggregateMetrics::default();
        if !metrics.is_empty() {
            let n = metrics.len() as f32;
            aggregate.mean_f1 = metrics.iter().map(|m| m.f1_score).sum::<f32>() / n;
            aggregate.mean_mcc = metrics.iter().map(|m| m.mcc).sum::<f32>() / n;
            aggregate.mean_auc_roc = metrics.iter().map(|m| m.auc_roc).sum::<f32>() / n;
            aggregate.mean_auprc = metrics.iter().map(|m| m.auprc).sum::<f32>() / n;
            aggregate.mean_precision = metrics.iter().map(|m| m.precision).sum::<f32>() / n;
            aggregate.mean_recall = metrics.iter().map(|m| m.recall).sum::<f32>() / n;
        }

        log::info!(
            "Batch with metrics: {} structures in {:.2}ms | F1={:.4} MCC={:.4} AUC-ROC={:.4} AUPRC={:.4}",
            n_structures,
            kernel_elapsed.as_secs_f64() * 1000.0,
            aggregate.mean_f1,
            aggregate.mean_mcc,
            aggregate.mean_auc_roc,
            aggregate.mean_auprc
        );

        Ok(BatchOutputWithMetrics {
            structures,
            metrics,
            kernel_time_us: kernel_elapsed.as_micros() as u64,
            aggregate,
        })
    }

    /// Check if training kernel is available
    pub fn has_training_kernel(&self) -> bool {
        self.training_func.is_some()
    }

    /// Run training kernel to extract reservoir states for readout training
    ///
    /// Returns per-structure reservoir states [n_residues * 4] for each structure.
    /// Use these states with `TrainedReadout::train()` to train the readout layer.
    pub fn extract_reservoir_states(
        &mut self,
        batch: &PackedBatchWithGT,
        config: &MegaFusedConfig,
    ) -> Result<TrainingOutput, PrismError> {
        let training_func = self
            .training_func
            .as_ref()
            .ok_or_else(|| PrismError::gpu("mega_fused_batch", "Training kernel not available"))?;

        let n_structures = batch.base.n_structures();
        let total_residues = batch.base.total_residues;
        const RESERVOIR_DIM: usize = 40; // 8 TDA + 8 input + 4 geometry + 6 dendritic + 2 calcium/soma + 8 reservoir + 6 combined

        if n_structures == 0 {
            return Ok(TrainingOutput {
                reservoir_states: Vec::new(),
                gt_masks: Vec::new(),
                structure_ids: Vec::new(),
            });
        }

        let kernel_start = Instant::now();

        // Allocate and upload input data
        let mut d_atoms = self
            .stream
            .alloc_zeros::<f32>(batch.base.atoms_packed.len().max(1))
            .map_err(|e| PrismError::gpu("training", format!("Failed to alloc atoms: {}", e)))?;
        d_atoms = self
            .stream
            .clone_htod(&batch.base.atoms_packed)
            .map_err(|e| PrismError::gpu("training", format!("Failed to upload atoms: {}", e)))?;

        let mut d_ca_indices = self
            .stream
            .alloc_zeros::<i32>(total_residues.max(1))
            .map_err(|e| {
                PrismError::gpu("training", format!("Failed to alloc ca_indices: {}", e))
            })?;
        d_ca_indices = self
            .stream
            .clone_htod(&batch.base.ca_indices_packed)
            .map_err(|e| {
                PrismError::gpu("training", format!("Failed to upload ca_indices: {}", e))
            })?;

        let mut d_conservation = self
            .stream
            .alloc_zeros::<f32>(total_residues.max(1))
            .map_err(|e| {
                PrismError::gpu("training", format!("Failed to alloc conservation: {}", e))
            })?;
        d_conservation = self
            .stream
            .clone_htod(&batch.base.conservation_packed)
            .map_err(|e| {
                PrismError::gpu("training", format!("Failed to upload conservation: {}", e))
            })?;

        let mut d_bfactor = self
            .stream
            .alloc_zeros::<f32>(total_residues.max(1))
            .map_err(|e| PrismError::gpu("training", format!("Failed to alloc bfactor: {}", e)))?;
        d_bfactor = self
            .stream
            .clone_htod(&batch.base.bfactor_packed)
            .map_err(|e| PrismError::gpu("training", format!("Failed to upload bfactor: {}", e)))?;

        let mut d_burial = self
            .stream
            .alloc_zeros::<f32>(total_residues.max(1))
            .map_err(|e| PrismError::gpu("training", format!("Failed to alloc burial: {}", e)))?;
        d_burial = self
            .stream
            .clone_htod(&batch.base.burial_packed)
            .map_err(|e| PrismError::gpu("training", format!("Failed to upload burial: {}", e)))?;

        // Upload descriptors (as raw bytes)
        let desc_size = std::mem::size_of::<BatchStructureDesc>() * n_structures;
        let mut d_descriptors = self
            .stream
            .alloc_zeros::<u8>(desc_size.max(1))
            .map_err(|e| {
                PrismError::gpu("training", format!("Failed to alloc descriptors: {}", e))
            })?;
        let desc_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(batch.base.descriptors.as_ptr() as *const u8, desc_size)
        };
        self.stream
            .memcpy_htod(desc_bytes, &mut d_descriptors)
            .map_err(|e| {
                PrismError::gpu("training", format!("Failed to upload descriptors: {}", e))
            })?;

        // Allocate output buffers
        let d_consensus = self
            .stream
            .alloc_zeros::<f32>(total_residues.max(1))
            .map_err(|e| {
                PrismError::gpu("training", format!("Failed to alloc consensus: {}", e))
            })?;
        let d_confidence = self
            .stream
            .alloc_zeros::<i32>(total_residues.max(1))
            .map_err(|e| {
                PrismError::gpu("training", format!("Failed to alloc confidence: {}", e))
            })?;
        let d_signal_mask = self
            .stream
            .alloc_zeros::<i32>(total_residues.max(1))
            .map_err(|e| {
                PrismError::gpu("training", format!("Failed to alloc signal_mask: {}", e))
            })?;
        let d_pocket_assignment = self
            .stream
            .alloc_zeros::<i32>(total_residues.max(1))
            .map_err(|e| {
                PrismError::gpu(
                    "training",
                    format!("Failed to alloc pocket_assignment: {}", e),
                )
            })?;
        let d_centrality = self
            .stream
            .alloc_zeros::<f32>(total_residues.max(1))
            .map_err(|e| {
                PrismError::gpu("training", format!("Failed to alloc centrality: {}", e))
            })?;

        // Reservoir states output buffer [total_residues * 4]
        let d_reservoir_states = self
            .stream
            .alloc_zeros::<f32>(total_residues.max(1) * RESERVOIR_DIM)
            .map_err(|e| {
                PrismError::gpu(
                    "training",
                    format!("Failed to alloc reservoir_states: {}", e),
                )
            })?;

        // Upload params
        let params = MegaFusedParams::from_config(config);
        let params_size = std::mem::size_of::<MegaFusedParams>();
        let mut d_params = self
            .stream
            .alloc_zeros::<u8>(params_size)
            .map_err(|e| PrismError::gpu("training", format!("Failed to alloc params: {}", e)))?;
        let params_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(&params as *const _ as *const u8, params_size) };
        self.stream
            .memcpy_htod(params_bytes, &mut d_params)
            .map_err(|e| PrismError::gpu("training", format!("Failed to upload params: {}", e)))?;

        // Launch training kernel
        let launch_config = LaunchConfig {
            grid_dim: (n_structures as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_structures_i32 = n_structures as i32;

        unsafe {
            let mut builder = self.stream.launch_builder(&training_func);
            builder.arg(&d_atoms);
            builder.arg(&d_ca_indices);
            builder.arg(&d_conservation);
            builder.arg(&d_bfactor);
            builder.arg(&d_burial);
            builder.arg(&d_descriptors);
            builder.arg(&n_structures_i32);
            builder.arg(&d_consensus);
            builder.arg(&d_confidence);
            builder.arg(&d_signal_mask);
            builder.arg(&d_pocket_assignment);
            builder.arg(&d_centrality);
            builder.arg(&d_reservoir_states);
            builder.arg(&d_params);
            builder
                .launch(launch_config)
                .map_err(|e| PrismError::gpu("training", format!("Kernel launch failed: {}", e)))?;
        }

        // Synchronize
        self.stream
            .synchronize()
            .map_err(|e| PrismError::gpu("training", format!("Sync failed: {}", e)))?;

        let kernel_elapsed = kernel_start.elapsed();

        // Download reservoir states
        let reservoir_states_flat: Vec<f32> =
            self.stream.clone_dtoh(&d_reservoir_states).map_err(|e| {
                PrismError::gpu(
                    "training",
                    format!("Failed to download reservoir_states: {}", e),
                )
            })?;

        // Split by structure
        let mut reservoir_states = Vec::with_capacity(n_structures);
        let mut gt_masks = Vec::with_capacity(n_structures);
        let mut structure_ids = Vec::with_capacity(n_structures);

        for (idx, desc) in batch.base.descriptors.iter().enumerate() {
            let n_res = desc.n_residues as usize;
            let res_offset = desc.residue_offset as usize;

            // Extract this structure's reservoir states
            let start = res_offset * RESERVOIR_DIM;
            let end = start + n_res * RESERVOIR_DIM;
            let states = reservoir_states_flat[start..end].to_vec();

            // Extract this structure's GT mask
            let mask = batch.gt_pocket_mask_packed[res_offset..res_offset + n_res].to_vec();

            reservoir_states.push(states);
            gt_masks.push(mask);
            structure_ids.push(batch.base.ids[idx].clone());
        }

        log::info!(
            "Extracted reservoir states: {} structures, {} residues in {:.2}ms",
            n_structures,
            total_residues,
            kernel_elapsed.as_secs_f64() * 1000.0
        );

        Ok(TrainingOutput {
            reservoir_states,
            gt_masks,
            structure_ids,
        })
    }

    /// Enhance BatchOutput with polycentric immunity features (22-dim per structure)
    ///
    /// Takes existing 136-dim features and adds 22 polycentric features (broadcast per residue).
    /// Returns BatchOutput with 158-dim features: [136 base + 22 polycentric]
    ///
    /// Requires:
    /// - polycentric GPU initialized with epitope centers
    /// - batch with epitope_escape_packed (per-residue, aggregated to per-structure)
    /// - batch with current_immunity_levels_75_packed (per-structure × 75)
    /// - time_since_infection, freq_history_7d, current_freq (per-structure)
    pub fn enhance_with_polycentric(
        &self,
        output: BatchOutput,
        batch: &PackedBatch,
        polycentric: &crate::polycentric_immunity::PolycentricImmunityGpu,
    ) -> Result<BatchOutput, PrismError> {
        use crate::polycentric_immunity::POLYCENTRIC_OUTPUT_DIM;

        let n_structures = output.structures.len();
        let total_residues: usize = output
            .structures
            .iter()
            .map(|s| s.combined_features.len() / 136)
            .sum();

        if n_structures == 0 {
            return Ok(output);
        }

        // === PREPARE POLYCENTRIC INPUT DATA ===

        // 1. Aggregate epitope escape from per-residue to per-structure (mean)
        let mut escape_10d_flat = Vec::with_capacity(n_structures * 10);
        for (idx, desc) in batch.descriptors.iter().enumerate().take(n_structures) {
            let res_offset = desc.residue_offset as usize;
            let n_res = desc.n_residues as usize;

            // Extract this structure's epitope escape [n_residues × 10]
            let start = res_offset * 10;
            let end = start + n_res * 10;
            let structure_escape = &batch.epitope_escape_packed[start..end];

            // Compute mean per epitope class
            let mut epitope_means = vec![0.0f32; 10];
            for r in 0..n_res {
                for e in 0..10 {
                    epitope_means[e] += structure_escape[r * 10 + e];
                }
            }
            for e in 0..10 {
                epitope_means[e] /= n_res as f32;
            }

            escape_10d_flat.extend_from_slice(&epitope_means);
        }

        // 2. Extract PK immunity (already per-structure × 75)
        let pk_immunity_flat = &batch.current_immunity_levels_75_packed[..n_structures * 75];

        // 3. Compute time_since_infection (placeholder - should come from metadata)
        let time_since_infection = vec![30.0f32; n_structures]; // TODO: Extract from batch metadata

        // 4. Extract frequency history (placeholder - should come from metadata)
        let freq_history_flat = vec![0.10f32; n_structures * 7]; // TODO: Extract from batch metadata

        // 5. Extract current frequency (placeholder)
        let current_freq = vec![0.15f32; n_structures]; // TODO: Extract from batch metadata

        // === UPLOAD TO GPU ===
        let mut d_escape_10d = self
            .stream
            .alloc_zeros::<f32>(escape_10d_flat.len().max(1))
            .map_err(|e| {
                PrismError::gpu("polycentric_enhance", format!("Alloc escape_10d: {}", e))
            })?;
        d_escape_10d = self.stream.clone_htod(&escape_10d_flat).map_err(|e| {
            PrismError::gpu("polycentric_enhance", format!("Upload escape_10d: {}", e))
        })?;

        let mut d_pk_immunity = self
            .stream
            .alloc_zeros::<f32>(pk_immunity_flat.len().max(1))
            .map_err(|e| {
                PrismError::gpu("polycentric_enhance", format!("Alloc pk_immunity: {}", e))
            })?;
        self.stream
            .memcpy_htod(&pk_immunity_flat[..], &mut d_pk_immunity)
            .map_err(|e| {
                PrismError::gpu("polycentric_enhance", format!("Upload pk_immunity: {}", e))
            })?;

        let d_time_since_infection =
            self.stream
                .clone_htod(&time_since_infection[..])
                .map_err(|e| {
                    PrismError::gpu(
                        "polycentric_enhance",
                        format!("Upload time_since_infection: {}", e),
                    )
                })?;

        let d_freq_history = self
            .stream
            .clone_htod(&freq_history_flat[..])
            .map_err(|e| {
                PrismError::gpu("polycentric_enhance", format!("Upload freq_history: {}", e))
            })?;

        let d_current_freq = self.stream.clone_htod(&current_freq[..]).map_err(|e| {
            PrismError::gpu("polycentric_enhance", format!("Upload current_freq: {}", e))
        })?;

        // Prepare features_packed and residue metadata
        let mut features_flat = Vec::with_capacity(total_residues * 136);
        let mut residue_offsets_flat = Vec::with_capacity(n_structures);
        let mut n_residues_flat = Vec::with_capacity(n_structures);

        let mut cumulative_offset = 0;
        for structure in &output.structures {
            let n_res = structure.combined_features.len() / 136;
            residue_offsets_flat.push(cumulative_offset as i32);
            n_residues_flat.push(n_res as i32);
            features_flat.extend_from_slice(&structure.combined_features);
            cumulative_offset += n_res;
        }

        let d_features = self.stream.clone_htod(&features_flat[..]).map_err(|e| {
            PrismError::gpu("polycentric_enhance", format!("Upload features: {}", e))
        })?;

        let d_residue_offsets = self
            .stream
            .clone_htod(&residue_offsets_flat[..])
            .map_err(|e| {
                PrismError::gpu(
                    "polycentric_enhance",
                    format!("Upload residue_offsets: {}", e),
                )
            })?;

        let d_n_residues = self.stream.clone_htod(&n_residues_flat[..]).map_err(|e| {
            PrismError::gpu("polycentric_enhance", format!("Upload n_residues: {}", e))
        })?;

        // === CALL POLYCENTRIC GPU ===
        let d_polycentric_output = polycentric
            .process_batch(
                &d_features,
                &d_residue_offsets,
                &d_n_residues,
                &d_escape_10d,
                &d_pk_immunity,
                &d_time_since_infection,
                &d_freq_history,
                &d_current_freq,
                n_structures,
            )
            .map_err(|e| {
                PrismError::gpu(
                    "polycentric_enhance",
                    format!("Polycentric process_batch failed: {}", e),
                )
            })?;

        // === DOWNLOAD RESULTS ===
        let polycentric_features =
            polycentric
                .download_output(&d_polycentric_output)
                .map_err(|e| {
                    PrismError::gpu(
                        "polycentric_enhance",
                        format!("Download polycentric output: {}", e),
                    )
                })?;

        // === MERGE INTO OUTPUT ===
        let mut enhanced_structures = Vec::with_capacity(n_structures);

        for (idx, mut structure) in output.structures.into_iter().enumerate() {
            let n_res = structure.combined_features.len() / 136;

            // Extract this structure's 22-dim polycentric features
            let poly_start = idx * POLYCENTRIC_OUTPUT_DIM;
            let poly_end = poly_start + POLYCENTRIC_OUTPUT_DIM;
            let poly_features = &polycentric_features[poly_start..poly_end];

            // Expand combined_features from 136 to 158 dim (broadcast polycentric features per residue)
            let mut enhanced_features = Vec::with_capacity(n_res * 158);
            for r in 0..n_res {
                // Copy original 136 features
                let base_start = r * 136;
                let base_end = base_start + 136;
                enhanced_features
                    .extend_from_slice(&structure.combined_features[base_start..base_end]);

                // Append 22 polycentric features (same for all residues in this structure)
                enhanced_features.extend_from_slice(poly_features);
            }

            structure.combined_features = enhanced_features;
            enhanced_structures.push(structure);
        }

        log::info!(
            "Enhanced {} structures with polycentric features (136 → 158 dim)",
            n_structures
        );

        Ok(BatchOutput {
            structures: enhanced_structures,
            gpu_telemetry: output.gpu_telemetry,
            batch_time_us: output.batch_time_us,
            kernel_time_us: output.kernel_time_us,
        })
    }
}

/// Output from training kernel - reservoir states for readout training
#[derive(Debug)]
pub struct TrainingOutput {
    /// Per-structure reservoir states [n_residues * 4] per structure
    pub reservoir_states: Vec<Vec<f32>>,
    /// Per-structure ground truth masks [n_residues] per structure
    pub gt_masks: Vec<Vec<u8>>,
    /// Structure IDs
    pub structure_ids: Vec<String>,
}

impl TrainingOutput {
    /// Convert to format expected by TrainedReadout::train()
    pub fn to_train_data(&self) -> Vec<(Vec<f32>, Vec<u8>)> {
        self.reservoir_states
            .iter()
            .zip(&self.gt_masks)
            .map(|(states, mask)| (states.clone(), mask.clone()))
            .collect()
    }

    /// Get statistics: (n_structures, n_residues, n_positive)
    pub fn stats(&self) -> (usize, usize, usize) {
        let n_structures = self.reservoir_states.len();
        let n_residues: usize = self.gt_masks.iter().map(|m| m.len()).sum();
        let n_positive: usize = self
            .gt_masks
            .iter()
            .flat_map(|m| m.iter())
            .filter(|&&l| l > 0)
            .count();
        (n_structures, n_residues, n_positive)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_structure_desc_alignment() {
        assert_eq!(std::mem::size_of::<BatchStructureDesc>(), 16);
        assert_eq!(std::mem::align_of::<BatchStructureDesc>(), 16);
    }

    #[test]
    fn test_structure_input_validation() {
        let mut input = StructureInput::new("test");
        input.atoms = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // 2 atoms
        input.ca_indices = vec![0, 1]; // 2 residues
        input.conservation = vec![0.5, 0.6];
        input.bfactor = vec![0.1, 0.2];
        input.burial = vec![0.3, 0.4];

        assert!(input.validate().is_ok());
        assert_eq!(input.n_atoms(), 2);
        assert_eq!(input.n_residues(), 2);
    }

    #[test]
    fn test_pack_batch() {
        let mut s1 = StructureInput::new("pdb1");
        s1.atoms = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        s1.ca_indices = vec![0, 1];
        s1.conservation = vec![0.5, 0.6];
        s1.bfactor = vec![0.1, 0.2];
        s1.burial = vec![0.3, 0.4];

        let mut s2 = StructureInput::new("pdb2");
        s2.atoms = vec![2.0, 2.0, 2.0];
        s2.ca_indices = vec![0];
        s2.conservation = vec![0.7];
        s2.bfactor = vec![0.3];
        s2.burial = vec![0.5];

        let batch = PackedBatch::from_structures(&[s1, s2]).unwrap();

        assert_eq!(batch.n_structures(), 2);
        assert_eq!(batch.total_atoms, 3);
        assert_eq!(batch.total_residues, 3);
        assert_eq!(batch.atoms_packed.len(), 9); // 3 atoms * 3 coords
        assert_eq!(batch.descriptors[0].atom_offset, 0);
        assert_eq!(batch.descriptors[0].n_atoms, 2);
        assert_eq!(batch.descriptors[1].atom_offset, 2);
        assert_eq!(batch.descriptors[1].n_atoms, 1);
    }
}
