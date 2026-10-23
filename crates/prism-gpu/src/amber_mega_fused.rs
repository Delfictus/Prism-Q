//! Mega-Fused AMBER HMC Dynamics
//!
//! Complete molecular dynamics in a single GPU kernel launch.
//! Uses flat arrays for GPU transfer compatibility.
//!
//! Supports both implicit solvent (distance-dependent dielectric) and
//! explicit solvent (TIP3P water with PME electrostatics and SETTLE constraints).

use anyhow::{Context, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceSlice, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::collections::HashSet;
use std::sync::Arc;

// Explicit solvent components
use crate::h_constraints::{HConstraintCluster, HConstraints};
use crate::pme::PME;
use crate::settle::Settle;

/// Maximum exclusions per atom (1-2, 1-3 bonded pairs)
pub const MAX_EXCLUSIONS: usize = 32;

/// Maximum 1-4 pairs per atom (dihedral terminal pairs needing scaled interactions)
pub const MAX_14_PAIRS: usize = 16;

/// Cell list constants (must match CUDA kernel)
pub const CELL_SIZE: f32 = 10.0;
pub const MAX_CELLS_PER_DIM: usize = 32;
pub const MAX_TOTAL_CELLS: usize = MAX_CELLS_PER_DIM * MAX_CELLS_PER_DIM * MAX_CELLS_PER_DIM;
pub const MAX_ATOMS_PER_CELL: usize = 128;
pub const NEIGHBOR_LIST_SIZE: usize = 256;

/// Boltzmann constant in kcal/(mol·K)
pub const KB_KCAL_MOL_K: f64 = 0.001987204;

/// Single energy/temperature record for trajectory logging
#[derive(Debug, Clone)]
pub struct EnergyRecord {
    pub step: u64,
    pub time_ps: f64,
    pub potential_energy: f64,
    pub kinetic_energy: f64,
    pub total_energy: f64,
    pub temperature: f64,
}

/// Constraint count information for DOF calculation
#[derive(Debug, Clone, Default)]
pub struct ConstraintInfo {
    pub n_waters: usize,
    pub n_settle_constraints: usize,
    pub n_h_constraints: usize,
}

/// HMC run results
#[derive(Debug, Clone)]
pub struct HmcRunResult {
    pub potential_energy: f64,
    pub kinetic_energy: f64,
    pub positions: Vec<f32>,
    pub velocities: Vec<f32>,
    pub avg_temperature: f64,
    /// Energy/temperature trajectory sampled during simulation
    pub energy_trajectory: Vec<EnergyRecord>,
    /// Degrees of freedom used for temperature calculation
    pub n_dof: usize,
    /// Constraint information for debugging
    pub constraint_info: ConstraintInfo,
}

// =============================================================================
// Phase 7: Mixed Precision (FP16/FP32) Infrastructure
// =============================================================================

/// Configuration for mixed precision computation
///
/// FP16 provides ~2x memory bandwidth improvement for LJ parameters
/// while maintaining FP32 accumulation for numerical stability.
#[derive(Debug, Clone)]
pub struct MixedPrecisionConfig {
    /// Enable FP16 for LJ sigma/epsilon parameters
    pub fp16_lj_params: bool,
    /// Enable FP16 for PME charge grid (spread/gather)
    pub fp16_pme_grid: bool,
    /// Enable Half2 vectorized LJ computation (2x throughput)
    pub half2_lj: bool,
    /// Maximum relative error threshold for validation
    pub max_relative_error: f32,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            fp16_lj_params: true,
            fp16_pme_grid: false, // Conservative default - enable after validation
            half2_lj: false,      // Requires sm_70+ and kernel changes
            max_relative_error: 0.001, // 0.1% max error
        }
    }
}

impl MixedPrecisionConfig {
    /// Create config for maximum performance (all FP16 features enabled)
    pub fn max_performance() -> Self {
        Self {
            fp16_lj_params: true,
            fp16_pme_grid: true,
            half2_lj: true,
            max_relative_error: 0.001,
        }
    }

    /// Create config for maximum accuracy (FP32 only)
    pub fn full_precision() -> Self {
        Self {
            fp16_lj_params: false,
            fp16_pme_grid: false,
            half2_lj: false,
            max_relative_error: 0.0,
        }
    }

    /// Check if any mixed precision features are enabled
    pub fn is_enabled(&self) -> bool {
        self.fp16_lj_params || self.fp16_pme_grid || self.half2_lj
    }
}

/// FP16 (half precision) GPU buffers for LJ parameters
///
/// Stores sigma and epsilon in FP16 format to reduce memory bandwidth.
/// Conversion: FP32 → FP16 on upload, FP16 → FP32 in kernel for accumulation.
pub struct MixedPrecisionBuffers {
    /// FP16 LJ sigma values [n_atoms] - stored as u16 (IEEE 754 binary16)
    pub d_sigma_fp16: CudaSlice<u16>,
    /// FP16 LJ epsilon values [n_atoms]
    pub d_epsilon_fp16: CudaSlice<u16>,
    /// Number of atoms
    n_atoms: usize,
    /// Whether buffers are initialized with valid data
    initialized: bool,
}

impl MixedPrecisionBuffers {
    /// Allocate FP16 buffers for n_atoms
    pub fn new(stream: &Arc<CudaStream>, n_atoms: usize) -> Result<Self> {
        let d_sigma_fp16 = stream
            .alloc_zeros::<u16>(n_atoms.max(1))
            .context("Failed to allocate FP16 sigma buffer")?;
        let d_epsilon_fp16 = stream
            .alloc_zeros::<u16>(n_atoms.max(1))
            .context("Failed to allocate FP16 epsilon buffer")?;

        log::debug!(
            "Allocated FP16 LJ buffers: {} atoms, {} bytes total",
            n_atoms,
            n_atoms * 4 // 2 bytes each for sigma and epsilon
        );

        Ok(Self {
            d_sigma_fp16,
            d_epsilon_fp16,
            n_atoms,
            initialized: false,
        })
    }

    /// Upload FP32 LJ parameters, converting to FP16
    ///
    /// Uses IEEE 754 binary16 format with round-to-nearest-even.
    pub fn upload_from_fp32(
        &mut self,
        stream: &Arc<CudaStream>,
        sigma: &[f32],
        epsilon: &[f32],
    ) -> Result<()> {
        if sigma.len() != self.n_atoms || epsilon.len() != self.n_atoms {
            anyhow::bail!(
                "Parameter size mismatch: got {}/{}, expected {}",
                sigma.len(),
                epsilon.len(),
                self.n_atoms
            );
        }

        // Convert FP32 to FP16 on CPU
        let sigma_fp16: Vec<u16> = sigma.iter().map(|&v| f32_to_f16_bits(v)).collect();
        let epsilon_fp16: Vec<u16> = epsilon.iter().map(|&v| f32_to_f16_bits(v)).collect();

        // Upload to GPU
        stream
            .memcpy_htod(&sigma_fp16, &mut self.d_sigma_fp16)
            .context("Failed to upload FP16 sigma")?;
        stream
            .memcpy_htod(&epsilon_fp16, &mut self.d_epsilon_fp16)
            .context("Failed to upload FP16 epsilon")?;

        self.initialized = true;

        // Validate conversion accuracy
        let max_sigma_err = Self::compute_max_relative_error(sigma, &sigma_fp16);
        let max_eps_err = Self::compute_max_relative_error(epsilon, &epsilon_fp16);

        log::debug!(
            "FP16 conversion: sigma max_err={:.6}%, epsilon max_err={:.6}%",
            max_sigma_err * 100.0,
            max_eps_err * 100.0
        );

        if max_sigma_err > 0.01 || max_eps_err > 0.01 {
            log::warn!(
                "FP16 conversion error exceeds 1%: sigma={:.2}%, epsilon={:.2}%",
                max_sigma_err * 100.0,
                max_eps_err * 100.0
            );
        }

        Ok(())
    }

    /// Compute maximum relative error from FP32→FP16 conversion
    fn compute_max_relative_error(original: &[f32], converted: &[u16]) -> f32 {
        original
            .iter()
            .zip(converted.iter())
            .map(|(&orig, &conv)| {
                let back = f16_bits_to_f32(conv);
                if orig.abs() < 1e-10 {
                    0.0
                } else {
                    ((back - orig) / orig).abs()
                }
            })
            .fold(0.0f32, f32::max)
    }

    /// Check if buffers are ready for use
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.n_atoms * 4 // 2 bytes sigma + 2 bytes epsilon
    }
}

/// Convert FP32 to FP16 bits (IEEE 754 binary16)
///
/// Uses round-to-nearest-even for best accuracy.
#[inline]
pub fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007FFFFF;

    // Handle special cases
    if exponent == 255 {
        // NaN or infinity
        if mantissa != 0 {
            return sign | 0x7E00; // NaN
        } else {
            return sign | 0x7C00; // Infinity
        }
    }

    // Bias conversion: FP32 bias = 127, FP16 bias = 15
    let new_exp = exponent - 127 + 15;

    if new_exp >= 31 {
        // Overflow to infinity
        return sign | 0x7C00;
    }

    if new_exp <= 0 {
        // Subnormal or underflow
        if new_exp < -10 {
            return sign; // Underflow to zero
        }
        // Subnormal handling
        let mant = (mantissa | 0x00800000) >> (14 - new_exp);
        return sign | ((mant >> 13) as u16);
    }

    // Normal number with rounding
    let mant = mantissa >> 13;
    let round_bit = (mantissa >> 12) & 1;
    let sticky = mantissa & 0x0FFF;

    let mut result = sign | ((new_exp as u16) << 10) | (mant as u16);

    // Round to nearest even
    if round_bit != 0 && (sticky != 0 || (mant & 1) != 0) {
        result += 1;
    }

    result
}

/// Convert FP16 bits to FP32
#[inline]
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x03FF) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign); // Zero
        }
        // Subnormal - normalize
        let mut mant = mantissa;
        let mut exp = 1i32;
        while (mant & 0x0400) == 0 {
            mant <<= 1;
            exp -= 1;
        }
        mant &= 0x03FF;
        let new_exp = ((exp - 15 + 127) as u32) << 23;
        return f32::from_bits(sign | new_exp | (mant << 13));
    }

    if exponent == 31 {
        if mantissa == 0 {
            return f32::from_bits(sign | 0x7F800000); // Infinity
        }
        return f32::from_bits(sign | 0x7FC00000); // NaN
    }

    // Normal number
    let new_exp = ((exponent as i32 - 15 + 127) as u32) << 23;
    f32::from_bits(sign | new_exp | (mantissa << 13))
}

/// Mega-fused AMBER HMC simulator
pub struct AmberMegaFusedHmc {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,

    // Kernels
    hmc_step_kernel: CudaFunction,
    zero_energies_kernel: CudaFunction,
    thermostat_kernel: CudaFunction,
    init_velocities_kernel: CudaFunction,
    minimize_kernel: CudaFunction,

    // Cell list kernels (O(N) non-bonded)
    build_cell_list_kernel: CudaFunction,
    build_neighbor_list_kernel: CudaFunction,

    // PBC kernel (for explicit solvent)
    set_pbc_box_kernel: CudaFunction,

    // Velocity Verlet kernels (proper 2-force-eval integrator)
    compute_forces_kernel: CudaFunction,
    vv_step1_kernel: CudaFunction,
    vv_step2_kernel: CudaFunction,

    // Phase 1: PBC position wrapping and COM drift removal kernels
    wrap_positions_kernel: CudaFunction,
    compute_com_velocity_kernel: CudaFunction,
    remove_com_velocity_kernel: CudaFunction,

    // Phase 2: Displacement-based neighbor list rebuild
    reset_max_displacement_kernel: CudaFunction,
    compute_max_displacement_kernel: CudaFunction,
    save_positions_at_build_kernel: CudaFunction,
    check_neighbor_overflow_kernel: CudaFunction,

    // Phase 7: Mixed precision kernel
    compute_forces_mixed_kernel: CudaFunction,

    // Phase 8: Fused kernels
    mega_fused_md_step_kernel: CudaFunction,
    mega_fused_md_step_tiled_kernel: CudaFunction,
    mega_fused_md_step_mixed_kernel: CudaFunction, // Phase 8.5: FP16 version
    fused_constraints_kernel: CudaFunction,

    // Device buffers - State (as flat f32 arrays)
    d_positions: CudaSlice<f32>,      // [n_atoms * 3]
    d_velocities: CudaSlice<f32>,     // [n_atoms * 3]
    d_forces: CudaSlice<f32>,         // [n_atoms * 3]
    d_total_energy: CudaSlice<f32>,   // [1]
    d_kinetic_energy: CudaSlice<f32>, // [1]

    // Device buffers - Topology (as flat primitive arrays)
    d_bond_atoms: CudaSlice<i32>,      // [n_bonds * 2] (i, j)
    d_bond_params: CudaSlice<f32>,     // [n_bonds * 2] (k, r0)
    d_angle_atoms: CudaSlice<i32>,     // [n_angles * 4] (i, j, k, pad)
    d_angle_params: CudaSlice<f32>,    // [n_angles * 2] (k, theta0)
    d_dihedral_atoms: CudaSlice<i32>,  // [n_dihedrals * 4] (i, j, k, l)
    d_dihedral_params: CudaSlice<f32>, // [n_dihedrals * 4] (k, n, phase, pad)

    // Device buffers - Non-bonded
    d_nb_sigma: CudaSlice<f32>,       // [n_atoms]
    d_nb_epsilon: CudaSlice<f32>,     // [n_atoms]
    d_nb_charge: CudaSlice<f32>,      // [n_atoms]
    d_nb_mass: CudaSlice<f32>,        // [n_atoms]
    d_exclusion_list: CudaSlice<i32>, // [n_atoms * MAX_EXCLUSIONS]
    d_n_exclusions: CudaSlice<i32>,   // [n_atoms]

    // Device buffers - 1-4 pairs (scaled non-bonded: LJ*0.5, Coulomb*0.833)
    d_pair14_list: CudaSlice<i32>, // [n_atoms * MAX_14_PAIRS]
    d_n_pairs14: CudaSlice<i32>,   // [n_atoms]

    // Device buffers - Cell lists (O(N) non-bonded)
    d_cell_list: CudaSlice<i32>, // [MAX_TOTAL_CELLS * MAX_ATOMS_PER_CELL]
    d_cell_counts: CudaSlice<i32>, // [MAX_TOTAL_CELLS]
    d_atom_cell: CudaSlice<i32>, // [n_atoms]
    d_neighbor_list: CudaSlice<i32>, // [n_atoms * NEIGHBOR_LIST_SIZE]
    d_n_neighbors: CudaSlice<i32>, // [n_atoms]
    d_bbox_min: CudaSlice<f32>,  // [3]
    d_bbox_max: CudaSlice<f32>,  // [3]

    // Cell grid dimensions (computed from bounding box)
    grid_origin: [f32; 3],
    grid_dims: [i32; 3],

    // Sizes
    n_atoms: usize,
    n_bonds: usize,
    n_angles: usize,
    n_dihedrals: usize,

    // Flags
    topology_ready: bool,
    velocities_initialized: bool,
    neighbor_list_valid: bool,

    // Periodic boundary conditions (for explicit solvent)
    pbc_enabled: bool,
    box_dimensions: [f32; 3],

    // Phase 1: COM drift removal buffer
    d_com_velocity: CudaSlice<f32>, // [4]: momentum_x, momentum_y, momentum_z, total_mass
    com_removal_interval: usize,    // How often to remove COM drift (default: 10 steps)

    // Phase 2: Displacement-based neighbor list rebuild buffers
    d_pos_at_build: CudaSlice<f32>, // [n_atoms * 3] - positions when neighbor list was built
    d_max_displacement: CudaSlice<f32>, // [1] - max displacement since last rebuild
    d_neighbor_overflow: CudaSlice<i32>, // [1] - overflow counter for neighbor list
    rebuild_threshold: f32,         // Rebuild when max_disp > threshold (skin/2 = 0.5 Å)
    neighbor_rebuild_count: usize,  // Statistics: number of rebuilds

    // Explicit solvent components (optional)
    pme: Option<PME>,
    settle: Option<Settle>,
    d_old_positions: Option<CudaSlice<f32>>, // For SETTLE constraint projection

    // H-bond constraints for protein (optional)
    h_constraints: Option<HConstraints>,

    // Position restraints (for stabilizing protein in implicit solvent)
    d_restrained_atoms: Option<CudaSlice<i32>>,
    d_ref_positions: Option<CudaSlice<f32>>,
    n_restrained: usize,
    k_restraint: f32,

    // Phase 7: Mixed precision (FP16/FP32) support
    mixed_precision_config: MixedPrecisionConfig,
    mixed_precision_buffers: Option<MixedPrecisionBuffers>,
}

impl AmberMegaFusedHmc {
    /// Create a new mega-fused AMBER HMC simulator
    pub fn new(context: Arc<CudaContext>, n_atoms: usize) -> Result<Self> {
        log::info!("🚀 Initializing Mega-Fused AMBER HMC for {} atoms", n_atoms);

        let stream = context.default_stream();

        // Load PTX module - use absolute path for reliability across different working directories
        let ptx_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/target/ptx/amber_mega_fused.ptx"
        );
        let ptx_src = std::fs::read_to_string(ptx_path)
            .with_context(|| format!("Failed to read PTX file: {}", ptx_path))?;
        let ptx = Ptx::from_src(&ptx_src);
        let module = context
            .load_module(ptx)
            .with_context(|| format!("Failed to load mega-fused PTX from {}", ptx_path))?;

        // Load kernels
        let hmc_step_kernel = module
            .load_function("amber_mega_fused_hmc_step")
            .context("Failed to load amber_mega_fused_hmc_step")?;
        let zero_energies_kernel = module
            .load_function("zero_energies")
            .context("Failed to load zero_energies")?;
        let thermostat_kernel = module
            .load_function("apply_thermostat")
            .context("Failed to load apply_thermostat")?;
        let init_velocities_kernel = module
            .load_function("initialize_velocities")
            .context("Failed to load initialize_velocities")?;
        let minimize_kernel = module
            .load_function("amber_steepest_descent_step")
            .context("Failed to load amber_steepest_descent_step")?;

        // Load cell list kernels for O(N) non-bonded
        let build_cell_list_kernel = module
            .load_function("build_cell_list")
            .context("Failed to load build_cell_list")?;
        let build_neighbor_list_kernel = module
            .load_function("build_neighbor_list")
            .context("Failed to load build_neighbor_list")?;

        // Load PBC kernel for explicit solvent
        let set_pbc_box_kernel = module
            .load_function("set_pbc_box")
            .context("Failed to load set_pbc_box")?;

        // Load velocity Verlet kernels (proper 2-force-eval integrator)
        let compute_forces_kernel = module
            .load_function("compute_forces_only")
            .context("Failed to load compute_forces_only")?;
        let vv_step1_kernel = module
            .load_function("velocity_verlet_step1")
            .context("Failed to load velocity_verlet_step1")?;
        let vv_step2_kernel = module
            .load_function("velocity_verlet_step2")
            .context("Failed to load velocity_verlet_step2")?;

        // Load Phase 1 kernels: PBC wrapping and COM drift removal
        let wrap_positions_kernel = module
            .load_function("wrap_positions_kernel")
            .context("Failed to load wrap_positions_kernel")?;
        let compute_com_velocity_kernel = module
            .load_function("compute_com_velocity")
            .context("Failed to load compute_com_velocity")?;
        let remove_com_velocity_kernel = module
            .load_function("remove_com_velocity")
            .context("Failed to load remove_com_velocity")?;

        // Load Phase 2 kernels: Displacement-based neighbor list rebuild
        let reset_max_displacement_kernel = module
            .load_function("reset_max_displacement")
            .context("Failed to load reset_max_displacement")?;
        let compute_max_displacement_kernel = module
            .load_function("compute_max_displacement")
            .context("Failed to load compute_max_displacement")?;
        let save_positions_at_build_kernel = module
            .load_function("save_positions_at_build")
            .context("Failed to load save_positions_at_build")?;
        let check_neighbor_overflow_kernel = module
            .load_function("check_neighbor_overflow")
            .context("Failed to load check_neighbor_overflow")?;

        // Load Phase 7 kernel: Mixed precision force calculation
        let compute_forces_mixed_kernel = module
            .load_function("compute_forces_mixed")
            .context("Failed to load compute_forces_mixed")?;

        // Load Phase 8 kernels: Fused kernels for reduced launch overhead
        let mega_fused_md_step_kernel = module
            .load_function("mega_fused_md_step")
            .context("Failed to load mega_fused_md_step")?;
        let mega_fused_md_step_tiled_kernel = module
            .load_function("mega_fused_md_step_tiled")
            .context("Failed to load mega_fused_md_step_tiled")?;
        let mega_fused_md_step_mixed_kernel = module
            .load_function("mega_fused_md_step_mixed")
            .context("Failed to load mega_fused_md_step_mixed")?;
        let fused_constraints_kernel = module
            .load_function("fused_constraints_kernel")
            .context("Failed to load fused_constraints_kernel")?;

        log::info!("📦 Cell list, PBC, Velocity Verlet, Phase 1, Phase 2, Phase 7, and Phase 8 kernels loaded");

        // Allocate state buffers
        let d_positions = stream.alloc_zeros::<f32>(n_atoms * 3)?;
        let d_velocities = stream.alloc_zeros::<f32>(n_atoms * 3)?;
        let d_forces = stream.alloc_zeros::<f32>(n_atoms * 3)?;
        let d_total_energy = stream.alloc_zeros::<f32>(1)?;
        let d_kinetic_energy = stream.alloc_zeros::<f32>(1)?;

        // Phase 1: COM velocity buffer for drift removal
        let d_com_velocity = stream.alloc_zeros::<f32>(4)?;

        // Phase 2: Displacement-based neighbor list rebuild buffers
        let d_pos_at_build = stream.alloc_zeros::<f32>(n_atoms * 3)?;
        let d_max_displacement = stream.alloc_zeros::<f32>(1)?;
        let d_neighbor_overflow = stream.alloc_zeros::<i32>(1)?;

        // Allocate topology buffers (minimal initial size)
        let d_bond_atoms = stream.alloc_zeros::<i32>(2)?;
        let d_bond_params = stream.alloc_zeros::<f32>(2)?;
        let d_angle_atoms = stream.alloc_zeros::<i32>(4)?;
        let d_angle_params = stream.alloc_zeros::<f32>(2)?;
        let d_dihedral_atoms = stream.alloc_zeros::<i32>(4)?;
        let d_dihedral_params = stream.alloc_zeros::<f32>(4)?;

        // Allocate NB buffers
        let d_nb_sigma = stream.alloc_zeros::<f32>(n_atoms)?;
        let d_nb_epsilon = stream.alloc_zeros::<f32>(n_atoms)?;
        let d_nb_charge = stream.alloc_zeros::<f32>(n_atoms)?;
        let d_nb_mass = stream.alloc_zeros::<f32>(n_atoms)?;
        let d_exclusion_list = stream.alloc_zeros::<i32>(n_atoms * MAX_EXCLUSIONS)?;
        let d_n_exclusions = stream.alloc_zeros::<i32>(n_atoms)?;

        // Allocate 1-4 pair buffers for scaled non-bonded
        let d_pair14_list = stream.alloc_zeros::<i32>(n_atoms * MAX_14_PAIRS)?;
        let d_n_pairs14 = stream.alloc_zeros::<i32>(n_atoms)?;

        // Allocate cell list buffers for O(N) non-bonded
        let d_cell_list = stream.alloc_zeros::<i32>(MAX_TOTAL_CELLS * MAX_ATOMS_PER_CELL)?;
        let d_cell_counts = stream.alloc_zeros::<i32>(MAX_TOTAL_CELLS)?;
        let d_atom_cell = stream.alloc_zeros::<i32>(n_atoms)?;
        let d_neighbor_list = stream.alloc_zeros::<i32>(n_atoms * NEIGHBOR_LIST_SIZE)?;
        let d_n_neighbors = stream.alloc_zeros::<i32>(n_atoms)?;
        let d_bbox_min = stream.alloc_zeros::<f32>(3)?;
        let d_bbox_max = stream.alloc_zeros::<f32>(3)?;

        log::info!("✅ Mega-Fused AMBER HMC initialized with O(N) cell lists");

        Ok(Self {
            context,
            stream,
            module,
            hmc_step_kernel,
            zero_energies_kernel,
            thermostat_kernel,
            init_velocities_kernel,
            minimize_kernel,
            build_cell_list_kernel,
            build_neighbor_list_kernel,
            set_pbc_box_kernel,
            compute_forces_kernel,
            vv_step1_kernel,
            vv_step2_kernel,
            wrap_positions_kernel,
            compute_com_velocity_kernel,
            remove_com_velocity_kernel,
            reset_max_displacement_kernel,
            compute_max_displacement_kernel,
            save_positions_at_build_kernel,
            check_neighbor_overflow_kernel,
            compute_forces_mixed_kernel,
            mega_fused_md_step_kernel,
            mega_fused_md_step_tiled_kernel,
            mega_fused_md_step_mixed_kernel,
            fused_constraints_kernel,
            d_positions,
            d_velocities,
            d_forces,
            d_total_energy,
            d_kinetic_energy,
            d_bond_atoms,
            d_bond_params,
            d_angle_atoms,
            d_angle_params,
            d_dihedral_atoms,
            d_dihedral_params,
            d_nb_sigma,
            d_nb_epsilon,
            d_nb_charge,
            d_nb_mass,
            d_exclusion_list,
            d_n_exclusions,
            d_pair14_list,
            d_n_pairs14,
            d_cell_list,
            d_cell_counts,
            d_atom_cell,
            d_neighbor_list,
            d_n_neighbors,
            d_bbox_min,
            d_bbox_max,
            grid_origin: [0.0, 0.0, 0.0],
            grid_dims: [1, 1, 1],
            n_atoms,
            n_bonds: 0,
            n_angles: 0,
            n_dihedrals: 0,
            topology_ready: false,
            velocities_initialized: false,
            neighbor_list_valid: false,
            pbc_enabled: false,
            box_dimensions: [0.0, 0.0, 0.0],
            d_com_velocity,
            com_removal_interval: 10, // Remove COM drift every 10 steps (default)
            d_pos_at_build,
            d_max_displacement,
            d_neighbor_overflow,
            rebuild_threshold: 0.5, // Rebuild when max_disp > skin/2 = 0.5 Å
            neighbor_rebuild_count: 0,
            pme: None,
            settle: None,
            d_old_positions: None,
            h_constraints: None,
            d_restrained_atoms: None,
            d_ref_positions: None,
            n_restrained: 0,
            k_restraint: 0.0,
            // Phase 7: Mixed precision defaults to disabled
            mixed_precision_config: MixedPrecisionConfig::full_precision(),
            mixed_precision_buffers: None,
        })
    }

    /// Upload topology and initial positions
    ///
    /// # Arguments
    /// * `positions` - Flat array [x0, y0, z0, x1, y1, z1, ...]
    /// * `bonds` - Vec of (atom_i, atom_j, k, r0)
    /// * `angles` - Vec of (atom_i, atom_j, atom_k, k, theta0)
    /// * `dihedrals` - Vec of (atom_i, atom_j, atom_k, atom_l, k, n, phase)
    /// * `nb_params` - Vec of (sigma, epsilon, charge, mass) per atom
    /// * `exclusions` - Per-atom exclusion sets
    pub fn upload_topology(
        &mut self,
        positions: &[f32],
        bonds: &[(usize, usize, f32, f32)],
        angles: &[(usize, usize, usize, f32, f32)],
        dihedrals: &[(usize, usize, usize, usize, f32, f32, f32)],
        nb_params: &[(f32, f32, f32, f32)], // (sigma, epsilon, charge, mass)
        exclusions: &[HashSet<usize>],
    ) -> Result<()> {
        if positions.len() != self.n_atoms * 3 {
            return Err(anyhow::anyhow!(
                "Position count {} != expected {}",
                positions.len(),
                self.n_atoms * 3
            ));
        }

        // Upload positions
        self.stream.memcpy_htod(positions, &mut self.d_positions)?;

        // Upload bonds as flat arrays
        if !bonds.is_empty() {
            let bond_atoms: Vec<i32> = bonds
                .iter()
                .flat_map(|&(i, j, _, _)| [i as i32, j as i32])
                .collect();
            let bond_params: Vec<f32> = bonds.iter().flat_map(|&(_, _, k, r0)| [k, r0]).collect();

            self.d_bond_atoms = self.stream.alloc_zeros::<i32>(bond_atoms.len())?;
            self.d_bond_params = self.stream.alloc_zeros::<f32>(bond_params.len())?;
            self.stream
                .memcpy_htod(&bond_atoms, &mut self.d_bond_atoms)?;
            self.stream
                .memcpy_htod(&bond_params, &mut self.d_bond_params)?;
        }
        self.n_bonds = bonds.len();

        // Upload angles
        if !angles.is_empty() {
            let angle_atoms: Vec<i32> = angles
                .iter()
                .flat_map(|&(i, j, k, _, _)| [i as i32, j as i32, k as i32, 0])
                .collect();
            let angle_params: Vec<f32> = angles
                .iter()
                .flat_map(|&(_, _, _, k, theta0)| [k, theta0])
                .collect();

            self.d_angle_atoms = self.stream.alloc_zeros::<i32>(angle_atoms.len())?;
            self.d_angle_params = self.stream.alloc_zeros::<f32>(angle_params.len())?;
            self.stream
                .memcpy_htod(&angle_atoms, &mut self.d_angle_atoms)?;
            self.stream
                .memcpy_htod(&angle_params, &mut self.d_angle_params)?;
        }
        self.n_angles = angles.len();

        // Upload dihedrals
        if !dihedrals.is_empty() {
            let dihedral_atoms: Vec<i32> = dihedrals
                .iter()
                .flat_map(|&(i, j, k, l, _, _, _)| [i as i32, j as i32, k as i32, l as i32])
                .collect();
            let dihedral_params: Vec<f32> = dihedrals
                .iter()
                .flat_map(|&(_, _, _, _, pk, n, phase)| [pk, n, phase, 0.0])
                .collect();

            self.d_dihedral_atoms = self.stream.alloc_zeros::<i32>(dihedral_atoms.len())?;
            self.d_dihedral_params = self.stream.alloc_zeros::<f32>(dihedral_params.len())?;
            self.stream
                .memcpy_htod(&dihedral_atoms, &mut self.d_dihedral_atoms)?;
            self.stream
                .memcpy_htod(&dihedral_params, &mut self.d_dihedral_params)?;
        }
        self.n_dihedrals = dihedrals.len();

        // Upload NB parameters as separate arrays
        let sigmas: Vec<f32> = nb_params.iter().map(|&(s, _, _, _)| s).collect();
        let epsilons: Vec<f32> = nb_params.iter().map(|&(_, e, _, _)| e).collect();
        let charges: Vec<f32> = nb_params.iter().map(|&(_, _, c, _)| c).collect();
        let masses: Vec<f32> = nb_params.iter().map(|&(_, _, _, m)| m).collect();

        self.stream.memcpy_htod(&sigmas, &mut self.d_nb_sigma)?;
        self.stream.memcpy_htod(&epsilons, &mut self.d_nb_epsilon)?;
        self.stream.memcpy_htod(&charges, &mut self.d_nb_charge)?;
        self.stream.memcpy_htod(&masses, &mut self.d_nb_mass)?;

        // Flatten and upload exclusions
        let mut excl_flat = vec![-1i32; self.n_atoms * MAX_EXCLUSIONS];
        let mut n_excl = vec![0i32; self.n_atoms];

        for (i, excl_set) in exclusions.iter().enumerate() {
            if i >= self.n_atoms {
                break;
            }
            n_excl[i] = excl_set.len().min(MAX_EXCLUSIONS) as i32;
            for (j, &excl_idx) in excl_set.iter().take(MAX_EXCLUSIONS).enumerate() {
                excl_flat[i * MAX_EXCLUSIONS + j] = excl_idx as i32;
            }
        }

        self.stream
            .memcpy_htod(&excl_flat, &mut self.d_exclusion_list)?;
        self.stream.memcpy_htod(&n_excl, &mut self.d_n_exclusions)?;

        // Build and upload 1-4 pairs from dihedrals
        // These get SCALED non-bonded interactions (LJ*0.5, Coulomb*0.833)
        let pairs_14 = build_14_pairs(dihedrals, exclusions, self.n_atoms);

        // Convert to per-atom format (like exclusions)
        let mut pair14_flat = vec![-1i32; self.n_atoms * MAX_14_PAIRS];
        let mut n_pairs14 = vec![0i32; self.n_atoms];

        for &(i, j) in &pairs_14 {
            // Add j to i's 1-4 list
            let count_i = n_pairs14[i] as usize;
            if count_i < MAX_14_PAIRS {
                pair14_flat[i * MAX_14_PAIRS + count_i] = j as i32;
                n_pairs14[i] += 1;
            }
            // Add i to j's 1-4 list (symmetric)
            let count_j = n_pairs14[j] as usize;
            if count_j < MAX_14_PAIRS {
                pair14_flat[j * MAX_14_PAIRS + count_j] = i as i32;
                n_pairs14[j] += 1;
            }
        }

        self.stream
            .memcpy_htod(&pair14_flat, &mut self.d_pair14_list)?;
        self.stream.memcpy_htod(&n_pairs14, &mut self.d_n_pairs14)?;

        self.topology_ready = true;

        log::info!(
            "📤 Topology uploaded: {} bonds, {} angles, {} dihedrals, {} 1-4 pairs",
            self.n_bonds,
            self.n_angles,
            self.n_dihedrals,
            pairs_14.len()
        );

        Ok(())
    }

    /// Get current positions
    pub fn get_positions(&self) -> Result<Vec<f32>> {
        let mut positions = vec![0.0f32; self.n_atoms * 3];
        self.stream.memcpy_dtoh(&self.d_positions, &mut positions)?;
        Ok(positions)
    }

    /// Set positions
    pub fn set_positions(&mut self, positions: &[f32]) -> Result<()> {
        if positions.len() != self.n_atoms * 3 {
            return Err(anyhow::anyhow!("Position count mismatch"));
        }
        self.stream.memcpy_htod(positions, &mut self.d_positions)?;
        Ok(())
    }

    /// Get number of atoms
    pub fn n_atoms(&self) -> usize {
        self.n_atoms
    }

    /// Check if topology is uploaded
    pub fn is_ready(&self) -> bool {
        self.topology_ready
    }

    /// Get constraint information for DOF calculation
    pub fn get_constraint_info(&self) -> ConstraintInfo {
        let n_waters = self.settle.as_ref().map_or(0, |s| s.n_waters());
        // Standard SETTLE: 3 distance constraints per water (O-H1, O-H2, H1-H2)
        // Even though velocity correction removes rotation, the DOF calculation
        // should match what the thermostat expects (3 constraints per water).
        // We compensate by scaling the target temperature passed to the kernel.
        let n_settle_constraints = 3 * n_waters;
        let n_h_constraints = self.h_constraints.as_ref().map_or(0, |h| h.n_constraints());

        ConstraintInfo {
            n_waters,
            n_settle_constraints,
            n_h_constraints,
        }
    }

    /// Compute degrees of freedom accounting for constraints
    ///
    /// For explicit solvent:
    /// N_dof = 3 * N_atoms - 3 (COM removal)
    ///       - 3 * N_waters (SETTLE: 3 distance constraints per water)
    ///       - N_h_constraints (X-H bonds: 1 constraint per bond)
    ///
    /// Note: We use 3 constraints per water (matching standard SETTLE) even though
    /// velocity correction removes rotation. The temperature compensation is handled
    /// by scaling the kernel temperature in run_fused() instead of adjusting DOF.
    ///
    /// For implicit solvent (no constraints):
    /// N_dof = 3 * N_atoms
    pub fn compute_n_dof(&self, remove_com: bool) -> usize {
        let base_dof = 3 * self.n_atoms;
        let constraint_info = self.get_constraint_info();

        let com_dof = if remove_com { 3 } else { 0 };
        let constrained_dof =
            constraint_info.n_settle_constraints + constraint_info.n_h_constraints;

        // Log DOF accounting for debugging
        log::info!(
            "DOF accounting: {} atoms, {} waters ({} SETTLE constraints), {} H-constraints",
            self.n_atoms,
            constraint_info.n_waters,
            constraint_info.n_settle_constraints,
            constraint_info.n_h_constraints
        );

        let n_dof = base_dof
            .saturating_sub(com_dof)
            .saturating_sub(constrained_dof);

        log::info!(
            "N_dof = {} - {} - {} - {} = {}",
            base_dof,
            com_dof,
            constraint_info.n_settle_constraints,
            constraint_info.n_h_constraints,
            n_dof
        );

        // Sanity check: ensure positive DOF
        if n_dof == 0 {
            log::warn!("WARNING: N_dof = 0, using fallback of 3 * n_atoms");
            return base_dof;
        }

        n_dof
    }

    /// Enable periodic boundary conditions
    ///
    /// Sets the box dimensions on the GPU and enables PBC wrapping
    /// during integration.
    ///
    /// # Arguments
    /// * `dims` - Box dimensions [Lx, Ly, Lz] in Angstroms
    /// * `use_pme` - If true, use PME electrostatics (explicit solvent)
    ///               If false, use implicit solvent (ε=4r) with PBC for geometry only
    pub fn set_pbc_box_with_pme(&mut self, dims: [f32; 3], use_pme: bool) -> Result<()> {
        if dims[0] <= 0.0 || dims[1] <= 0.0 || dims[2] <= 0.0 {
            return Err(anyhow::anyhow!(
                "Invalid box dimensions: {:?} - must be positive",
                dims
            ));
        }

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        let use_pme_flag: i32 = if use_pme { 1 } else { 0 };

        // Call the kernel to set PBC box on GPU
        unsafe {
            let mut builder = self.stream.launch_builder(&self.set_pbc_box_kernel);
            builder.arg(&dims[0]);
            builder.arg(&dims[1]);
            builder.arg(&dims[2]);
            builder.arg(&use_pme_flag);
            builder.launch(cfg)?;
        }

        self.stream.synchronize()?;
        self.pbc_enabled = true;
        self.box_dimensions = dims;

        log::info!(
            "📦 PBC enabled: box = {:.1} × {:.1} × {:.1} Å, PME = {}",
            dims[0],
            dims[1],
            dims[2],
            use_pme
        );

        Ok(())
    }

    /// Enable periodic boundary conditions (implicit solvent, no PME)
    ///
    /// Sets the box dimensions on the GPU and enables PBC wrapping
    /// during integration. Uses implicit solvent (ε=4r) electrostatics.
    /// For explicit solvent with PME, use `enable_explicit_solvent` instead.
    ///
    /// # Arguments
    /// * `dims` - Box dimensions [Lx, Ly, Lz] in Angstroms
    pub fn set_pbc_box(&mut self, dims: [f32; 3]) -> Result<()> {
        // Use implicit solvent (no PME) by default - PME requires explicit solvent setup
        self.set_pbc_box_with_pme(dims, false)
    }

    /// Check if PBC is enabled
    pub fn is_pbc_enabled(&self) -> bool {
        self.pbc_enabled
    }

    /// Get current box dimensions
    pub fn get_box_dimensions(&self) -> Option<[f32; 3]> {
        if self.pbc_enabled {
            Some(self.box_dimensions)
        } else {
            None
        }
    }

    // ========================================================================
    // Phase 1: PBC Position Wrapping and COM Drift Removal
    // ========================================================================

    /// Wrap all atom positions into the primary simulation box [0, L)
    ///
    /// Call this AFTER constraints (SETTLE + H-bonds) to ensure molecules stay
    /// intact before being wrapped. Only operates when PBC is enabled.
    pub fn wrap_positions(&mut self) -> Result<()> {
        if !self.pbc_enabled {
            return Ok(()); // No-op for non-periodic systems
        }

        let threads_per_block = 256;
        let num_blocks = (self.n_atoms + threads_per_block - 1) / threads_per_block;
        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_atoms_i32 = self.n_atoms as i32;
        unsafe {
            let mut builder = self.stream.launch_builder(&self.wrap_positions_kernel);
            builder.arg(&self.d_positions);
            builder.arg(&n_atoms_i32);
            builder.launch(cfg)?;
        }

        Ok(())
    }

    /// Remove center-of-mass velocity drift
    ///
    /// Subtracts the net COM velocity from all atoms to eliminate translational
    /// drift in periodic systems. Essential for explicit solvent simulations.
    ///
    /// This is a two-step process:
    /// 1. Compute COM velocity (parallel reduction)
    /// 2. Subtract COM velocity from all atoms
    pub fn remove_com_drift(&mut self) -> Result<()> {
        if !self.pbc_enabled {
            return Ok(()); // No-op for non-periodic systems
        }

        let threads_per_block = 256;
        let num_blocks = (self.n_atoms + threads_per_block - 1) / threads_per_block;
        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_atoms_i32 = self.n_atoms as i32;

        // Step 1: Compute COM velocity (accumulates into d_com_velocity)
        unsafe {
            let mut builder = self
                .stream
                .launch_builder(&self.compute_com_velocity_kernel);
            builder.arg(&self.d_velocities);
            builder.arg(&self.d_nb_mass);
            builder.arg(&self.d_com_velocity);
            builder.arg(&n_atoms_i32);
            builder.launch(cfg)?;
        }

        // Synchronize to ensure reduction is complete
        self.stream.synchronize()?;

        // Step 2: Subtract COM velocity from all atoms
        unsafe {
            let mut builder = self.stream.launch_builder(&self.remove_com_velocity_kernel);
            builder.arg(&self.d_velocities);
            builder.arg(&self.d_com_velocity);
            builder.arg(&n_atoms_i32);
            builder.launch(cfg)?;
        }

        Ok(())
    }

    /// Set the interval for COM drift removal (default: 10 steps)
    ///
    /// COM removal every 10-100 steps is sufficient for most simulations.
    /// Setting to 0 disables COM removal entirely.
    pub fn set_com_removal_interval(&mut self, interval: usize) {
        self.com_removal_interval = interval;
    }

    // ========================================================================
    // Phase 2: Displacement-based Neighbor List Rebuild
    // ========================================================================

    /// Check if neighbor list rebuild is needed based on atom displacement
    ///
    /// Computes the maximum displacement of any atom since the last neighbor
    /// list build. If max_disp > rebuild_threshold (skin/2), returns true.
    ///
    /// Returns (needs_rebuild, max_displacement)
    pub fn check_neighbor_list_rebuild_needed(&mut self) -> Result<(bool, f32)> {
        let threads_per_block = 256;
        let num_blocks = (self.n_atoms + threads_per_block - 1) / threads_per_block;
        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_atoms_i32 = self.n_atoms as i32;

        // Step 1: Reset max displacement counter
        unsafe {
            let mut builder = self
                .stream
                .launch_builder(&self.reset_max_displacement_kernel);
            builder.arg(&self.d_max_displacement);
            builder.launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            })?;
        }

        // Step 2: Compute max displacement (parallel reduction with atomic max)
        unsafe {
            let mut builder = self
                .stream
                .launch_builder(&self.compute_max_displacement_kernel);
            builder.arg(&self.d_positions);
            builder.arg(&self.d_pos_at_build);
            builder.arg(&self.d_max_displacement);
            builder.arg(&n_atoms_i32);
            builder.launch(cfg)?;
        }

        // Synchronize and read result
        self.stream.synchronize()?;
        let mut max_disp = [0.0f32];
        self.stream
            .memcpy_dtoh(&self.d_max_displacement, &mut max_disp)?;

        let needs_rebuild = max_disp[0] > self.rebuild_threshold;
        Ok((needs_rebuild, max_disp[0]))
    }

    /// Save current positions as reference for displacement tracking
    ///
    /// Call this AFTER rebuilding the neighbor list.
    pub fn save_positions_at_build(&mut self) -> Result<()> {
        let threads_per_block = 256;
        let num_blocks = (self.n_atoms + threads_per_block - 1) / threads_per_block;
        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_atoms_i32 = self.n_atoms as i32;

        unsafe {
            let mut builder = self
                .stream
                .launch_builder(&self.save_positions_at_build_kernel);
            builder.arg(&self.d_positions);
            builder.arg(&self.d_pos_at_build);
            builder.arg(&n_atoms_i32);
            builder.launch(cfg)?;
        }

        Ok(())
    }

    /// Check for neighbor list overflow
    ///
    /// Returns the number of atoms that have more neighbors than the
    /// allocated neighbor list size. If > 0, the NEIGHBOR_LIST_SIZE constant
    /// needs to be increased.
    pub fn check_neighbor_overflow(&mut self) -> Result<i32> {
        let threads_per_block = 256;
        let num_blocks = (self.n_atoms + threads_per_block - 1) / threads_per_block;
        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_atoms_i32 = self.n_atoms as i32;

        unsafe {
            let mut builder = self
                .stream
                .launch_builder(&self.check_neighbor_overflow_kernel);
            builder.arg(&self.d_n_neighbors);
            builder.arg(&self.d_neighbor_overflow);
            builder.arg(&n_atoms_i32);
            builder.launch(cfg)?;
        }

        self.stream.synchronize()?;
        let mut overflow_count = [0i32];
        self.stream
            .memcpy_dtoh(&self.d_neighbor_overflow, &mut overflow_count)?;

        if overflow_count[0] > 0 {
            log::warn!(
                "⚠️  Neighbor list overflow: {} atoms have more neighbors than NEIGHBOR_LIST_SIZE (256)",
                overflow_count[0]
            );
        }

        Ok(overflow_count[0])
    }

    /// Set the displacement threshold for neighbor list rebuild (default: 0.5 Å)
    ///
    /// Should typically be set to skin/2 where skin is the Verlet skin distance.
    /// With skin = 1.0 Å, threshold = 0.5 Å is appropriate.
    pub fn set_rebuild_threshold(&mut self, threshold: f32) {
        self.rebuild_threshold = threshold;
    }

    /// Get the number of neighbor list rebuilds since simulation start
    pub fn get_neighbor_rebuild_count(&self) -> usize {
        self.neighbor_rebuild_count
    }

    /// Enable explicit solvent with PME electrostatics
    ///
    /// This initializes:
    /// - PME for long-range electrostatics (reciprocal space)
    /// - PBC for periodic boundary conditions
    /// - Old positions buffer for SETTLE constraint projection
    ///
    /// # Arguments
    /// * `box_dims` - Periodic box dimensions [Lx, Ly, Lz] in Angstroms
    pub fn enable_explicit_solvent(&mut self, box_dims: [f32; 3]) -> Result<()> {
        log::info!(
            "🌊 Enabling explicit solvent: box = {:.1} × {:.1} × {:.1} Å",
            box_dims[0],
            box_dims[1],
            box_dims[2]
        );

        // Set up periodic boundary conditions WITH PME enabled
        self.set_pbc_box_with_pme(box_dims, true)?;

        // Initialize PME for long-range electrostatics
        let pme = PME::new(self.context.clone(), self.n_atoms, box_dims)?;
        self.pme = Some(pme);

        // Allocate old positions buffer for SETTLE
        let d_old_positions = self.stream.alloc_zeros::<f32>(self.n_atoms * 3)?;
        self.d_old_positions = Some(d_old_positions);

        log::info!("✅ Explicit solvent enabled with PME");
        Ok(())
    }

    /// Set up SETTLE constraints for rigid water molecules
    ///
    /// # Arguments
    /// * `water_oxygen_indices` - Indices of oxygen atoms in each water molecule.
    ///   Hydrogens are assumed to be at consecutive indices (O+1, O+2).
    pub fn set_water_molecules(&mut self, water_oxygen_indices: &[usize]) -> Result<()> {
        if water_oxygen_indices.is_empty() {
            log::info!("No water molecules - SETTLE constraints not needed");
            return Ok(());
        }

        log::info!(
            "🌊 Setting up SETTLE constraints for {} water molecules",
            water_oxygen_indices.len()
        );

        let settle = Settle::new(self.context.clone(), water_oxygen_indices, self.n_atoms)?;
        self.settle = Some(settle);

        log::info!("✅ SETTLE constraints initialized");
        Ok(())
    }

    /// Check if explicit solvent is enabled
    pub fn is_explicit_solvent(&self) -> bool {
        self.pme.is_some()
    }

    /// Get reference to PME (if enabled)
    pub fn pme(&self) -> Option<&PME> {
        self.pme.as_ref()
    }

    /// Get reference to SETTLE (if enabled)
    pub fn settle(&self) -> Option<&Settle> {
        self.settle.as_ref()
    }

    /// Get mutable reference to SETTLE (if enabled)
    pub fn settle_mut(&mut self) -> Option<&mut Settle> {
        self.settle.as_mut()
    }

    /// Check SETTLE constraint violations
    ///
    /// Returns (max_oh_violation, max_hh_violation) in Angstroms
    /// Returns None if SETTLE is not enabled
    pub fn check_settle_constraints(&mut self) -> Result<Option<(f32, f32)>> {
        if let Some(ref mut settle) = self.settle {
            Ok(Some(settle.check_constraints(&self.d_positions)?))
        } else {
            Ok(None)
        }
    }

    /// Set H-bond constraints for protein X-H bonds
    ///
    /// This enables analytic constraints for fast H-bond vibrations,
    /// allowing larger timesteps (2.0 fs vs 0.25 fs without constraints).
    ///
    /// # Arguments
    /// * `clusters` - H-bond cluster definitions from topology preparation
    pub fn set_h_constraints(&mut self, clusters: &[HConstraintCluster]) -> Result<()> {
        if clusters.is_empty() {
            log::info!("No H-bond clusters - H-constraints not needed");
            return Ok(());
        }

        log::info!(
            "🔗 Setting up H-bond constraints for {} clusters",
            clusters.len()
        );

        let h_constraints = HConstraints::new(self.context.clone(), clusters)?;
        self.h_constraints = Some(h_constraints);

        log::info!("✅ H-bond constraints initialized");
        Ok(())
    }

    /// Get reference to H-constraints (if enabled)
    pub fn h_constraints(&self) -> Option<&HConstraints> {
        self.h_constraints.as_ref()
    }

    /// Set up position restraints to prevent protein unfolding
    ///
    /// Applies harmonic restraints to selected atoms toward their initial positions.
    /// Useful for stabilizing proteins in implicit solvent simulations.
    ///
    /// # Arguments
    /// * `restrained_atoms` - Indices of atoms to restrain (e.g., all backbone heavy atoms)
    /// * `k_restraint` - Spring constant in kcal/(mol*Å²). Typical values:
    ///   - 100.0: Very strong (essentially frozen)
    ///   - 10.0: Strong (limits motion to ~0.3 Å)
    ///   - 1.0: Moderate (allows ~1 Å fluctuations)
    ///   - 0.1: Weak (allows ~3 Å fluctuations)
    pub fn set_position_restraints(
        &mut self,
        restrained_atoms: &[usize],
        k_restraint: f32,
    ) -> Result<()> {
        if !self.topology_ready {
            return Err(anyhow::anyhow!(
                "Topology not uploaded - call upload_topology first"
            ));
        }

        if restrained_atoms.is_empty() {
            log::info!("No atoms to restrain - position restraints not enabled");
            return Ok(());
        }

        // Get current positions as reference
        let mut positions = vec![0.0f32; self.n_atoms * 3];
        self.stream.memcpy_dtoh(&self.d_positions, &mut positions)?;

        // Extract reference positions for restrained atoms only
        let mut ref_positions = Vec::with_capacity(restrained_atoms.len() * 3);
        for &atom_idx in restrained_atoms {
            if atom_idx >= self.n_atoms {
                return Err(anyhow::anyhow!(
                    "Atom index {} out of range (n_atoms={})",
                    atom_idx,
                    self.n_atoms
                ));
            }
            ref_positions.push(positions[atom_idx * 3]);
            ref_positions.push(positions[atom_idx * 3 + 1]);
            ref_positions.push(positions[atom_idx * 3 + 2]);
        }

        // Upload restraint data to GPU
        let restrained_i32: Vec<i32> = restrained_atoms.iter().map(|&x| x as i32).collect();
        let mut d_restrained = self.stream.alloc_zeros::<i32>(restrained_atoms.len())?;
        let mut d_ref = self.stream.alloc_zeros::<f32>(ref_positions.len())?;
        self.stream
            .memcpy_htod(&restrained_i32, &mut d_restrained)?;
        self.stream.memcpy_htod(&ref_positions, &mut d_ref)?;

        self.d_restrained_atoms = Some(d_restrained);
        self.d_ref_positions = Some(d_ref);
        self.n_restrained = restrained_atoms.len();
        self.k_restraint = k_restraint;

        log::info!(
            "⚓ Position restraints enabled: {} atoms with k={:.1} kcal/(mol*Å²)",
            self.n_restrained,
            self.k_restraint
        );

        Ok(())
    }

    /// Get number of restrained atoms
    pub fn n_restrained(&self) -> usize {
        self.n_restrained
    }

    /// Disable position restraints
    pub fn disable_position_restraints(&mut self) {
        self.d_restrained_atoms = None;
        self.d_ref_positions = None;
        self.n_restrained = 0;
        self.k_restraint = 0.0;
        log::info!("Position restraints disabled");
    }

    /// Apply position restraint correction to velocities for fused kernel
    ///
    /// Since the fused kernel computes and applies forces internally without position restraints,
    /// we apply a post-hoc correction. This reads forces that were computed by apply_position_restraints
    /// and adjusts velocities accordingly: v += F * dt / m
    fn apply_position_restraint_velocity_correction(&mut self, dt: f32) -> Result<()> {
        if self.n_restrained == 0 {
            return Ok(());
        }

        // Download forces, velocities, masses, and positions
        let mut forces = vec![0.0f32; self.n_atoms * 3];
        let mut velocities = vec![0.0f32; self.n_atoms * 3];
        let mut masses = vec![0.0f32; self.n_atoms];
        let mut positions = vec![0.0f32; self.n_atoms * 3];

        self.stream.memcpy_dtoh(&self.d_forces, &mut forces)?;
        self.stream
            .memcpy_dtoh(&self.d_velocities, &mut velocities)?;
        self.stream.memcpy_dtoh(&self.d_nb_mass, &mut masses)?;
        self.stream.memcpy_dtoh(&self.d_positions, &mut positions)?;

        // Get reference positions and restrained atom indices
        let ref_len = self.n_restrained * 3;
        let mut ref_positions = vec![0.0f32; ref_len];
        let mut restrained_atoms = vec![0i32; self.n_restrained];

        if let Some(ref d_ref) = &self.d_ref_positions {
            self.stream.memcpy_dtoh(d_ref, &mut ref_positions)?;
        }
        if let Some(ref d_restrained) = &self.d_restrained_atoms {
            self.stream
                .memcpy_dtoh(d_restrained, &mut restrained_atoms)?;
        }

        // FORCE_TO_ACCEL converts kcal/(mol*Å) to Å/fs² when mass is in g/mol
        const FORCE_TO_ACCEL: f32 = 4.184e-4;

        // Apply velocity correction: v += F * dt * FORCE_TO_ACCEL / m
        // Also apply position correction: x = 0.5 * (x + ref) to pull toward reference
        for (idx_in_list, &atom_i32) in restrained_atoms.iter().enumerate() {
            let atom = atom_i32 as usize;
            if atom >= self.n_atoms {
                continue;
            }

            let m = masses[atom];
            if m <= 0.0 {
                continue;
            }

            // Compute force from restraint: F = -k * (x - ref)
            let rx = ref_positions[idx_in_list * 3];
            let ry = ref_positions[idx_in_list * 3 + 1];
            let rz = ref_positions[idx_in_list * 3 + 2];

            let dx = positions[atom * 3] - rx;
            let dy = positions[atom * 3 + 1] - ry;
            let dz = positions[atom * 3 + 2] - rz;

            let fx = -self.k_restraint * dx;
            let fy = -self.k_restraint * dy;
            let fz = -self.k_restraint * dz;

            // Apply velocity correction
            let accel_factor = dt * FORCE_TO_ACCEL / m;
            velocities[atom * 3] += fx * accel_factor;
            velocities[atom * 3 + 1] += fy * accel_factor;
            velocities[atom * 3 + 2] += fz * accel_factor;

            // Apply strong position correction: move 50% of the way back to reference
            // This is more aggressive than just force-based to prevent escape
            positions[atom * 3] -= 0.5 * dx;
            positions[atom * 3 + 1] -= 0.5 * dy;
            positions[atom * 3 + 2] -= 0.5 * dz;
        }

        // Upload corrected velocities and positions
        self.stream
            .memcpy_htod(&velocities, &mut self.d_velocities)?;
        self.stream.memcpy_htod(&positions, &mut self.d_positions)?;
        self.stream.synchronize()?;

        Ok(())
    }

    /// Initialize velocities from Maxwell-Boltzmann distribution
    pub fn initialize_velocities(&mut self, temperature: f32) -> Result<()> {
        if !self.topology_ready {
            return Err(anyhow::anyhow!("Topology not uploaded"));
        }

        let threads_per_block = 256;
        let num_blocks = (self.n_atoms + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        // Load flat velocity init kernel
        let init_kernel = self
            .module
            .load_function("initialize_velocities_flat")
            .context("Failed to load initialize_velocities_flat")?;

        // Bind values to avoid temporary lifetime issues
        let n_atoms_i32 = self.n_atoms as i32;

        unsafe {
            let mut builder = self.stream.launch_builder(&init_kernel);
            builder.arg(&self.d_velocities);
            builder.arg(&self.d_nb_mass);
            builder.arg(&n_atoms_i32);
            builder.arg(&temperature);
            builder.arg(&seed);
            builder.launch(cfg)?;
        }

        self.stream.synchronize()?;
        self.velocities_initialized = true;

        log::info!("✅ Velocities initialized at T={}K", temperature);
        Ok(())
    }

    /// Run energy minimization (steepest descent)
    ///
    /// This is CRITICAL before HMC: ANM conformations often have steric clashes
    /// that cause force explosions. Minimization relaxes these clashes.
    ///
    /// # Arguments
    /// * `n_steps` - Number of minimization steps (typical: 100-500)
    /// * `step_size` - Step size in Angstroms (typical: 0.001-0.01)
    ///
    /// # Returns
    /// Final potential energy after minimization
    pub fn minimize(&mut self, n_steps: usize, step_size: f32) -> Result<f32> {
        if !self.topology_ready {
            return Err(anyhow::anyhow!("Topology not uploaded"));
        }

        let threads_per_block = 256;
        let num_blocks = (self.n_atoms * 3 + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 6 * 64 * 4, // Same as HMC kernel
        };

        // Load minimize kernel
        let minimize_kernel = self
            .module
            .load_function("amber_steepest_descent_step")
            .context("Failed to load amber_steepest_descent_step")?;

        // Bind parameters
        let max_excl_i32 = MAX_EXCLUSIONS as i32;
        let max_14_i32 = MAX_14_PAIRS as i32;
        let n_atoms_i32 = self.n_atoms as i32;
        let n_bonds_i32 = self.n_bonds as i32;
        let n_angles_i32 = self.n_angles as i32;
        let n_dihedrals_i32 = self.n_dihedrals as i32;

        log::info!(
            "⚡ Running {} minimization steps (step_size={} Å)",
            n_steps,
            step_size
        );

        let mut last_energy = f32::MAX;
        for step in 0..n_steps {
            // 1. Zero forces before integration (CRITICAL: GPU memory persists!)
            self.stream.memset_zeros(&mut self.d_forces)?;

            // 2. Compute PME (CRITICAL: Minimizer must see long-range forces)
            // Without PME, system relaxes to a state invalid for the full physics engine
            if let Some(ref mut pme) = self.pme {
                self.stream.synchronize()?;
                let _ = pme.compute(&self.d_positions, &self.d_nb_charge, &mut self.d_forces)?;
            }

            // 3. Launch Minimize Kernel (Accumulates Short-Range + Moves Atoms)
            // NOTE: SETTLE is NOT applied during minimization loop.
            // This allows waters to relax and resolve clashes naturally.
            // SETTLE will be applied once at the end to fix water geometry.
            unsafe {
                let mut builder = self.stream.launch_builder(&minimize_kernel);
                builder.arg(&self.d_positions);
                builder.arg(&self.d_forces);
                builder.arg(&self.d_total_energy);
                builder.arg(&self.d_bond_atoms);
                builder.arg(&self.d_bond_params);
                builder.arg(&self.d_angle_atoms);
                builder.arg(&self.d_angle_params);
                builder.arg(&self.d_dihedral_atoms);
                builder.arg(&self.d_dihedral_params);
                builder.arg(&self.d_nb_sigma);
                builder.arg(&self.d_nb_epsilon);
                builder.arg(&self.d_nb_charge);
                builder.arg(&self.d_exclusion_list);
                builder.arg(&self.d_n_exclusions);
                builder.arg(&self.d_pair14_list);
                builder.arg(&self.d_n_pairs14);
                builder.arg(&max_excl_i32);
                builder.arg(&max_14_i32);
                builder.arg(&n_atoms_i32);
                builder.arg(&n_bonds_i32);
                builder.arg(&n_angles_i32);
                builder.arg(&n_dihedrals_i32);
                builder.arg(&step_size);
                builder.launch(cfg)?;
            }

            // Apply H-bond constraints for protein
            if let Some(ref h_constraints) = self.h_constraints {
                h_constraints.apply(&mut self.d_positions, &mut self.d_velocities, 1.0)?;
            }

            // Check energy every 50 steps
            if step % 50 == 0 || step == n_steps - 1 {
                self.stream.synchronize()?;
                let mut energy = vec![0.0f32; 1];
                self.stream.memcpy_dtoh(&self.d_total_energy, &mut energy)?;

                if step == 0 {
                    log::info!("  Step 0: PE = {:.2} kcal/mol", energy[0]);
                    // Download and analyze forces
                    let mut forces = vec![0.0f32; self.n_atoms * 3];
                    self.stream.memcpy_dtoh(&self.d_forces, &mut forces)?;
                    let mut max_force = 0.0f32;
                    let mut total_force_mag = 0.0f32;
                    let mut n_large = 0;
                    for i in 0..self.n_atoms {
                        let fx = forces[i * 3];
                        let fy = forces[i * 3 + 1];
                        let fz = forces[i * 3 + 2];
                        let mag = (fx * fx + fy * fy + fz * fz).sqrt();
                        if mag > max_force {
                            max_force = mag;
                        }
                        total_force_mag += mag;
                        if mag > 100.0 {
                            n_large += 1;
                        }
                    }
                    let avg_force = total_force_mag / self.n_atoms as f32;
                    log::info!(
                        "  Force stats: max={:.1}, avg={:.1}, n_large(>100)={}/{}",
                        max_force,
                        avg_force,
                        n_large,
                        self.n_atoms
                    );
                } else if step == n_steps - 1 {
                    log::info!("  Step {}: PE = {:.2} kcal/mol (final)", step, energy[0]);
                }

                // Early termination if converged
                if (last_energy - energy[0]).abs() < 0.1 && step > 50 {
                    log::info!("  Converged at step {} (ΔE < 0.1 kcal/mol)", step);
                    break;
                }
                last_energy = energy[0];
            }
        }

        // Apply SETTLE once at the end to fix any water geometry drift
        // that accumulated during minimization (without blocking rotational relaxation)
        if let Some(ref mut settle) = self.settle {
            // Save current positions as reference for SETTLE
            settle.save_positions(&self.d_positions)?;
            self.stream.synchronize()?;
            settle.apply(&mut self.d_positions, 1.0)?;
            log::info!("  Applied final SETTLE constraints after minimization");
        }

        self.stream.synchronize()?;
        let mut final_energy = vec![0.0f32; 1];
        self.stream
            .memcpy_dtoh(&self.d_total_energy, &mut final_energy)?;

        log::info!(
            "✅ Minimization complete: PE = {:.2} kcal/mol",
            final_energy[0]
        );
        Ok(final_energy[0])
    }

    /// Build neighbor lists for O(N) non-bonded calculation
    ///
    /// This builds cell lists and neighbor lists from current positions.
    /// Must be called before run() and periodically during long simulations
    /// (every ~10-20 steps if atoms move significantly).
    ///
    /// Uses spatial hashing with cells of size = cutoff (10Å).
    /// Each atom only checks 27 neighboring cells for neighbors.
    /// Reduces O(N²) to O(N) average case.
    pub fn build_neighbor_lists(&mut self) -> Result<()> {
        if !self.topology_ready {
            return Err(anyhow::anyhow!("Topology not ready"));
        }

        let threads_per_block = 256;
        let num_blocks = (self.n_atoms + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Step 1: Compute bounding box of all atoms
        // We do this on CPU for simplicity (download positions, compute min/max)
        let mut positions = vec![0.0f32; self.n_atoms * 3];
        self.stream.memcpy_dtoh(&self.d_positions, &mut positions)?;

        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut min_z = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;
        let mut max_z = f32::MIN;

        for i in 0..self.n_atoms {
            let x = positions[i * 3];
            let y = positions[i * 3 + 1];
            let z = positions[i * 3 + 2];
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            min_z = min_z.min(z);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
            max_z = max_z.max(z);
        }

        // Add small padding to avoid edge cases
        min_x -= 1.0;
        min_y -= 1.0;
        min_z -= 1.0;
        max_x += 1.0;
        max_y += 1.0;
        max_z += 1.0;

        // Compute grid dimensions
        let nx = ((max_x - min_x) / CELL_SIZE).ceil() as i32;
        let ny = ((max_y - min_y) / CELL_SIZE).ceil() as i32;
        let nz = ((max_z - min_z) / CELL_SIZE).ceil() as i32;

        // Clamp to maximum grid size
        let nx = nx.min(MAX_CELLS_PER_DIM as i32).max(1);
        let ny = ny.min(MAX_CELLS_PER_DIM as i32).max(1);
        let nz = nz.min(MAX_CELLS_PER_DIM as i32).max(1);

        self.grid_origin = [min_x, min_y, min_z];
        self.grid_dims = [nx, ny, nz];

        log::debug!(
            "📦 Cell grid: {}x{}x{} cells, origin: ({:.1}, {:.1}, {:.1})",
            nx,
            ny,
            nz,
            min_x,
            min_y,
            min_z
        );

        // Step 2: Zero cell counts
        let zero_counts = vec![0i32; MAX_TOTAL_CELLS];
        self.stream
            .memcpy_htod(&zero_counts, &mut self.d_cell_counts)?;

        // Step 3: Build cell lists (assign atoms to cells)
        let n_atoms_i32 = self.n_atoms as i32;
        unsafe {
            let mut builder = self.stream.launch_builder(&self.build_cell_list_kernel);
            builder.arg(&self.d_positions);
            builder.arg(&self.d_cell_list);
            builder.arg(&self.d_cell_counts);
            builder.arg(&self.d_atom_cell);
            builder.arg(&min_x);
            builder.arg(&min_y);
            builder.arg(&min_z);
            builder.arg(&nx);
            builder.arg(&ny);
            builder.arg(&nz);
            builder.arg(&n_atoms_i32);
            builder.launch(cfg.clone())?;
        }

        // Step 4: Build neighbor lists from cell lists
        let max_excl_i32 = MAX_EXCLUSIONS as i32;
        unsafe {
            let mut builder = self.stream.launch_builder(&self.build_neighbor_list_kernel);
            builder.arg(&self.d_positions);
            builder.arg(&self.d_cell_list);
            builder.arg(&self.d_cell_counts);
            builder.arg(&self.d_atom_cell);
            builder.arg(&self.d_exclusion_list);
            builder.arg(&self.d_n_exclusions);
            builder.arg(&self.d_neighbor_list);
            builder.arg(&self.d_n_neighbors);
            builder.arg(&max_excl_i32);
            builder.arg(&nx);
            builder.arg(&ny);
            builder.arg(&nz);
            builder.arg(&n_atoms_i32);
            builder.launch(cfg)?;
        }

        self.stream.synchronize()?;
        self.neighbor_list_valid = true;
        self.neighbor_rebuild_count += 1;

        // Phase 2: Check for overflow and save positions for displacement tracking
        let overflow = self.check_neighbor_overflow()?;
        if overflow > 0 {
            log::error!(
                "❌ Neighbor list overflow: {} atoms exceeded NEIGHBOR_LIST_SIZE. Increase the constant!",
                overflow
            );
        }
        self.save_positions_at_build()?;

        // Debug: check average neighbor count (use debug level to avoid spam during rebuilds)
        let mut n_neighbors = vec![0i32; self.n_atoms];
        self.stream
            .memcpy_dtoh(&self.d_n_neighbors, &mut n_neighbors)?;
        let avg_neighbors: f64 =
            n_neighbors.iter().map(|&n| n as f64).sum::<f64>() / self.n_atoms as f64;
        log::debug!(
            "Neighbor lists built (rebuild #{}): avg {:.1} neighbors/atom (vs {} for O(N²))",
            self.neighbor_rebuild_count,
            avg_neighbors,
            self.n_atoms
        );

        Ok(())
    }

    /// Run HMC trajectory with full AMBER ff14SB physics
    ///
    /// Uses O(N) neighbor lists for non-bonded forces.
    /// Includes bonds, angles, dihedrals, LJ, and Coulomb.
    /// BAOAB Langevin thermostat maintains temperature via friction + noise.
    ///
    /// # Arguments
    /// * `n_steps` - Number of integration steps
    /// * `dt` - Timestep in femtoseconds
    /// * `temperature` - Target temperature in Kelvin
    /// * `gamma_fs` - Langevin friction coefficient in fs⁻¹
    ///   - 0.001 (1 ps⁻¹): Production - preserves natural dynamics, τ = 1 ps
    ///   - 0.01 (10 ps⁻¹): Equilibration - fast thermalization, τ = 100 fs
    ///   - 0.1 (100 ps⁻¹): Aggressive - Brownian dynamics limit
    ///
    /// # Returns
    /// HMC run results including final energies and positions
    pub fn run(
        &mut self,
        n_steps: usize,
        dt: f32,
        temperature: f32,
        gamma_fs: f32,
    ) -> Result<HmcRunResult> {
        if !self.topology_ready {
            return Err(anyhow::anyhow!(
                "Topology not uploaded - call upload_topology first"
            ));
        }

        // Build neighbor lists for O(N) non-bonded (required for full-atom)
        if !self.neighbor_list_valid {
            log::info!("📦 Building neighbor lists for O(N) non-bonded...");
            self.build_neighbor_lists()?;
        }

        // Initialize velocities if not done
        if !self.velocities_initialized {
            self.initialize_velocities(temperature)?;
        }

        let threads_per_block = 256;
        let num_blocks = (self
            .n_atoms
            .max(self.n_bonds)
            .max(self.n_angles)
            .max(self.n_dihedrals)
            + threads_per_block
            - 1)
            / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 6 * 64 * 4, // 6 arrays * TILE_SIZE * sizeof(float)
        };

        // Load flat HMC kernel
        let hmc_kernel = self
            .module
            .load_function("amber_mega_fused_hmc_step_flat")
            .context("Failed to load amber_mega_fused_hmc_step_flat")?;

        let thermostat_kernel = self
            .module
            .load_function("apply_thermostat_flat")
            .context("Failed to load apply_thermostat_flat")?;

        let mut total_ke = 0.0f64;

        // Bind all integer parameters to variables for stable references
        let max_excl_i32 = MAX_EXCLUSIONS as i32;
        let max_14_i32 = MAX_14_PAIRS as i32;
        let n_atoms_i32 = self.n_atoms as i32;
        let n_bonds_i32 = self.n_bonds as i32;
        let n_angles_i32 = self.n_angles as i32;
        let n_dihedrals_i32 = self.n_dihedrals as i32;

        // Compute DOF for proper temperature calculation (HMC method - no explicit solvent)
        // For explicit solvent, use COM removal and account for SETTLE constraints
        let has_explicit_solvent = self.pbc_enabled && self.settle.is_some();
        let n_dof = self.compute_n_dof(has_explicit_solvent);
        let constraint_info = self.get_constraint_info();

        // Initialize energy trajectory
        const ENERGY_SAMPLE_INTERVAL: usize = 100;
        let mut energy_trajectory: Vec<EnergyRecord> =
            Vec::with_capacity(n_steps / ENERGY_SAMPLE_INTERVAL + 1);

        log::info!(
            "🏃 Running {} HMC steps on GPU (dt={}fs, T={}K, γ={}fs⁻¹)",
            n_steps,
            dt,
            temperature,
            gamma_fs
        );

        // Phase 2: Displacement-based neighbor list rebuild
        // Check displacement every N steps (checking every step is expensive)
        const DISPLACEMENT_CHECK_INTERVAL: usize = 10;

        for step in 0..n_steps {
            // Phase 2: Check if neighbor list rebuild is needed based on displacement
            if step > 0 && step % DISPLACEMENT_CHECK_INTERVAL == 0 {
                let (needs_rebuild, max_disp) = self.check_neighbor_list_rebuild_needed()?;
                if needs_rebuild {
                    log::debug!(
                        "📦 Rebuilding neighbor list at step {} (max_disp={:.3} Å > threshold={:.3} Å)",
                        step, max_disp, self.rebuild_threshold
                    );
                    self.neighbor_list_valid = false;
                    self.build_neighbor_lists()?;
                }
            }

            // Save positions BEFORE integration for SETTLE constraint projection
            if let Some(ref mut settle) = self.settle {
                settle.save_positions(&self.d_positions)?;
            }

            // ============================================================
            // FORCE PREPARATION: Zero + PME before kernel
            // ============================================================
            // The kernel ACCUMULATES forces (no internal zeroing), so we must:
            // 1. Zero forces first (CRITICAL: GPU memory persists between steps!)
            // 2. Add position restraint forces (if enabled)
            // 3. Add PME reciprocal forces (if explicit solvent)
            // 4. Then kernel adds bonded + short-range NB
            self.stream.memset_zeros(&mut self.d_forces)?;

            // Add position restraint forces (if enabled)
            if self.n_restrained > 0 {
                if let (Some(ref d_restrained), Some(ref d_ref)) =
                    (&self.d_restrained_atoms, &self.d_ref_positions)
                {
                    let restraint_kernel =
                        self.module
                            .load_function("apply_position_restraints")
                            .context("Failed to load apply_position_restraints kernel")?;

                    let threads = 256;
                    let blocks = (self.n_restrained + threads - 1) / threads;
                    let cfg = LaunchConfig {
                        grid_dim: (blocks as u32, 1, 1),
                        block_dim: (threads as u32, 1, 1),
                        shared_mem_bytes: 0,
                    };

                    let n_restrained_i32 = self.n_restrained as i32;
                    unsafe {
                        let mut builder = self.stream.launch_builder(&restraint_kernel);
                        builder.arg(&self.d_forces);
                        builder.arg(&self.d_total_energy);
                        builder.arg(&self.d_positions);
                        builder.arg(d_ref);
                        builder.arg(d_restrained);
                        builder.arg(&n_restrained_i32);
                        builder.arg(&self.k_restraint);
                        builder.launch(cfg)?;
                    }
                }
            }

            // Add PME reciprocal forces BEFORE kernel
            if let Some(ref mut pme) = self.pme {
                self.stream.synchronize()?;
                let pme_energy =
                    pme.compute(&self.d_positions, &self.d_nb_charge, &mut self.d_forces)?;

                // DIAGNOSTIC: Check PME force magnitudes on first few steps
                if step < 5 {
                    self.stream.synchronize()?;
                    let mut forces = vec![0.0f32; self.n_atoms * 3];
                    self.stream.memcpy_dtoh(&self.d_forces, &mut forces)?;
                    let mut max_f = 0.0f32;
                    let mut sum_f2 = 0.0f32;
                    for i in 0..self.n_atoms {
                        let fx = forces[i * 3];
                        let fy = forces[i * 3 + 1];
                        let fz = forces[i * 3 + 2];
                        let f2 = fx * fx + fy * fy + fz * fz;
                        sum_f2 += f2;
                        if f2 > max_f * max_f {
                            max_f = f2.sqrt();
                        }
                    }
                    let rms_f = (sum_f2 / self.n_atoms as f32).sqrt();
                    log::info!(
                        "🔍 Step {} PME: E={:.2} kcal/mol, max_F={:.2}, rms_F={:.2} kcal/(mol·Å)",
                        step,
                        pme_energy,
                        max_f,
                        rms_f
                    );
                }
            }

            // Launch mega-fused HMC step kernel
            let step_u32 = step as u32;
            unsafe {
                let mut builder = self.stream.launch_builder(&hmc_kernel);
                builder.arg(&self.d_positions);
                builder.arg(&self.d_velocities);
                builder.arg(&self.d_forces);
                builder.arg(&self.d_total_energy);
                builder.arg(&self.d_kinetic_energy);
                builder.arg(&self.d_bond_atoms);
                builder.arg(&self.d_bond_params);
                builder.arg(&self.d_angle_atoms);
                builder.arg(&self.d_angle_params);
                builder.arg(&self.d_dihedral_atoms);
                builder.arg(&self.d_dihedral_params);
                builder.arg(&self.d_nb_sigma);
                builder.arg(&self.d_nb_epsilon);
                builder.arg(&self.d_nb_charge);
                builder.arg(&self.d_nb_mass);
                builder.arg(&self.d_exclusion_list);
                builder.arg(&self.d_n_exclusions);
                builder.arg(&self.d_pair14_list);
                builder.arg(&self.d_n_pairs14);
                builder.arg(&max_excl_i32);
                builder.arg(&max_14_i32);
                builder.arg(&n_atoms_i32);
                builder.arg(&n_bonds_i32);
                builder.arg(&n_angles_i32);
                builder.arg(&n_dihedrals_i32);
                builder.arg(&dt);
                builder.arg(&temperature);
                builder.arg(&gamma_fs); // Langevin friction coefficient
                builder.arg(&step_u32); // Step counter for RNG seeding
                builder.launch(cfg)?;
            }

            // NOTE: Thermostat disabled - BAOAB Langevin dynamics handles temperature
            // control continuously via friction + thermal noise (O step).
            // External thermostat would fight the Langevin dynamics.

            // DIAGNOSTIC: Check total force magnitudes on first few steps
            if step < 5 {
                self.stream.synchronize()?;
                let mut forces = vec![0.0f32; self.n_atoms * 3];
                self.stream.memcpy_dtoh(&self.d_forces, &mut forces)?;
                let mut max_f = 0.0f32;
                let mut sum_f2 = 0.0f32;
                let mut max_idx = 0;
                for i in 0..self.n_atoms {
                    let fx = forces[i * 3];
                    let fy = forces[i * 3 + 1];
                    let fz = forces[i * 3 + 2];
                    let f2 = fx * fx + fy * fy + fz * fz;
                    sum_f2 += f2;
                    if f2 > max_f * max_f {
                        max_f = f2.sqrt();
                        max_idx = i;
                    }
                }
                let rms_f = (sum_f2 / self.n_atoms as f32).sqrt();
                log::info!(
                    "🔍 Step {} TOTAL: max_F={:.2} (atom {}), rms_F={:.2} kcal/(mol·Å)",
                    step,
                    max_f,
                    max_idx,
                    rms_f
                );

                // Also log velocities
                let mut velocities = vec![0.0f32; self.n_atoms * 3];
                self.stream
                    .memcpy_dtoh(&self.d_velocities, &mut velocities)?;
                let mut max_v = 0.0f32;
                for i in 0..self.n_atoms {
                    let vx = velocities[i * 3];
                    let vy = velocities[i * 3 + 1];
                    let vz = velocities[i * 3 + 2];
                    let v2 = vx * vx + vy * vy + vz * vz;
                    if v2 > max_v * max_v {
                        max_v = v2.sqrt();
                    }
                }
                log::info!("🔍 Step {} VEL: max_v={:.4} Å/fs", step, max_v);
            }

            // ============================================================
            // SETTLE CONSTRAINTS (after integration)
            // ============================================================
            // SETTLE is applied AFTER integration to fix water geometry,
            // projecting positions back onto the rigid TIP3P constraint surface.
            // PME was already applied BEFORE the kernel (see above).
            // ============================================================

            // Apply SETTLE constraints for water molecules
            // Position constraints + RATTLE velocity correction.
            // Velocity correction projects out constraint-violating components,
            // preserving rotational kinetic energy.
            if let Some(ref mut settle) = self.settle {
                self.stream.synchronize()?;
                settle.apply(&mut self.d_positions, dt)?;
                settle.apply_velocity_correction(&mut self.d_velocities, &self.d_positions)?;
            }

            // ============================================================
            // VELOCITY RESCALING (after SETTLE)
            // ============================================================
            // The Langevin thermostat adds energy to all 9 DOF per water,
            // but SETTLE velocity correction projects to 3 DOF (COM only).
            // This removes ~2/3 of the injected energy, causing the thermostat
            // to over-inject and temperature to be ~2x too high.
            //
            // Fix: Rescale velocities every N steps to maintain target T.
            // This is the standard approach in MD codes with constrained water.
            // ============================================================
            const VELOCITY_RESCALE_INTERVAL: usize = 10;
            if has_explicit_solvent && step % VELOCITY_RESCALE_INTERVAL == 0 && step > 0 {
                // Rescale velocities to target temperature
                // Use the weaker rescaling to avoid completely killing fluctuations
                if let Err(e) = self.rescale_velocities(temperature) {
                    log::warn!("Velocity rescaling failed: {}", e);
                }
            }

            // Apply H-bond constraints for protein X-H bonds
            // This freezes fast vibrations (~10 fs period), enabling larger timesteps
            if let Some(ref h_constraints) = self.h_constraints {
                h_constraints.apply(&mut self.d_positions, &mut self.d_velocities, dt)?;
            }

            // Phase 1: Wrap positions into periodic box (after constraints)
            self.wrap_positions()?;

            // Phase 1: Remove COM drift (every com_removal_interval steps)
            if self.com_removal_interval > 0 && step % self.com_removal_interval == 0 {
                self.remove_com_drift()?;
            }

            // Recompute KE after velocity correction for accurate measurement
            // The kernel computed KE before SETTLE, so it includes rotational KE
            // that doesn't correspond to actual constrained motion.
            if step % 10 == 0 {
                self.stream.synchronize()?;

                // Download corrected velocities and masses
                let mut velocities = vec![0.0f32; self.n_atoms * 3];
                let mut masses = vec![0.0f32; self.n_atoms];
                self.stream
                    .memcpy_dtoh(&self.d_velocities, &mut velocities)?;
                self.stream.memcpy_dtoh(&self.d_nb_mass, &mut masses)?;

                // Compute corrected KE
                let mut ke_sum = 0.0f64;
                for i in 0..self.n_atoms {
                    let m = masses[i].max(0.001) as f64;
                    let vx = velocities[i * 3] as f64;
                    let vy = velocities[i * 3 + 1] as f64;
                    let vz = velocities[i * 3 + 2] as f64;
                    let v2 = vx * vx + vy * vy + vz * vz;
                    ke_sum += 0.5 * m * v2 / 4.184e-4;
                }
                total_ke += ke_sum;
            }

            // DIAGNOSTIC: Log energy every 100 steps and record trajectory
            if step % ENERGY_SAMPLE_INTERVAL == 0 || step < 10 {
                self.stream.synchronize()?;
                let mut pe = vec![0.0f32; 1];
                let mut ke = vec![0.0f32; 1];
                self.stream.memcpy_dtoh(&self.d_total_energy, &mut pe)?;
                self.stream.memcpy_dtoh(&self.d_kinetic_energy, &mut ke)?;

                // Compute temperature using proper DOF
                let inst_temp = 2.0 * ke[0] as f64 / (n_dof as f64 * KB_KCAL_MOL_K);
                let time_ps = step as f64 * dt as f64 / 1000.0; // fs -> ps

                // Record to energy trajectory
                if step % ENERGY_SAMPLE_INTERVAL == 0 {
                    energy_trajectory.push(EnergyRecord {
                        step: step as u64,
                        time_ps,
                        potential_energy: pe[0] as f64,
                        kinetic_energy: ke[0] as f64,
                        total_energy: (pe[0] + ke[0]) as f64,
                        temperature: inst_temp,
                    });
                }

                log::info!(
                    "📊 Step {:>5}: PE={:>12.2} KE={:>10.2} T={:>8.1}K (DOF={})",
                    step,
                    pe[0],
                    ke[0],
                    inst_temp,
                    n_dof
                );
            }
        }

        self.stream.synchronize()?;

        // Download final results
        let mut h_total_energy = vec![0.0f32; 1];
        let mut h_kinetic_energy = vec![0.0f32; 1];
        self.stream
            .memcpy_dtoh(&self.d_total_energy, &mut h_total_energy)?;
        self.stream
            .memcpy_dtoh(&self.d_kinetic_energy, &mut h_kinetic_energy)?;

        let positions = self.get_positions()?;
        let mut velocities = vec![0.0f32; self.n_atoms * 3];
        self.stream
            .memcpy_dtoh(&self.d_velocities, &mut velocities)?;

        // Average temperature using proper DOF
        let n_samples = (n_steps / 10).max(1);
        let avg_ke = total_ke / n_samples as f64;
        let avg_temperature = 2.0 * avg_ke / (n_dof as f64 * KB_KCAL_MOL_K);

        // DIAGNOSTIC: Verify atom count and KE sanity
        let expected_ke_at_target = 0.5 * n_dof as f64 * KB_KCAL_MOL_K * temperature as f64;
        log::info!(
            "🔬 DIAG: avg_ke={:.1}, expected_ke@{}K={:.1}, ratio={:.3}, n_dof={}, n_samples={}",
            avg_ke,
            temperature,
            expected_ke_at_target,
            avg_ke / expected_ke_at_target,
            n_dof,
            n_samples
        );

        log::info!(
            "✅ HMC complete: PE={:.2} kcal/mol, KE={:.2} kcal/mol, T_avg={:.1}K (DOF={})",
            h_total_energy[0],
            h_kinetic_energy[0],
            avg_temperature,
            n_dof
        );

        Ok(HmcRunResult {
            potential_energy: h_total_energy[0] as f64,
            kinetic_energy: h_kinetic_energy[0] as f64,
            positions,
            velocities,
            avg_temperature,
            energy_trajectory,
            n_dof,
            constraint_info,
        })
    }

    /// Run proper velocity Verlet integration with TWO force evaluations per step.
    ///
    /// This is the correct symplectic integrator that conserves energy in NVE:
    /// 1. compute_forces_only at x(t)
    /// 2. velocity_verlet_step1: v += (dt/2)*a; x += dt*v
    /// 3. compute_forces_only at x(t+dt)  [SECOND FORCE EVAL]
    /// 4. velocity_verlet_step2: v += (dt/2)*a; O-step
    ///
    /// This is 2x slower than the mega-fused kernel but conserves energy properly.
    pub fn run_verlet(
        &mut self,
        n_steps: usize,
        dt: f32,
        temperature: f32,
        gamma_fs: f32,
    ) -> Result<HmcRunResult> {
        if !self.topology_ready {
            return Err(anyhow::anyhow!(
                "Topology not uploaded - call upload_topology first"
            ));
        }

        // Build neighbor lists for O(N) non-bonded
        if !self.neighbor_list_valid {
            log::info!("📦 Building neighbor lists for O(N) non-bonded...");
            self.build_neighbor_lists()?;
        }

        // Initialize velocities if not done
        if !self.velocities_initialized {
            self.initialize_velocities(temperature)?;
        }

        let threads = 256;
        let blocks = (self.n_atoms + threads - 1) / threads;

        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Config for force kernel (needs more threads for bonds/angles)
        let max_n = self
            .n_atoms
            .max(self.n_bonds)
            .max(self.n_angles)
            .max(self.n_dihedrals);
        let force_blocks = (max_n + threads - 1) / threads;
        let force_cfg = LaunchConfig {
            grid_dim: (force_blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Bind integer params
        let n_atoms_i32 = self.n_atoms as i32;
        let n_bonds_i32 = self.n_bonds as i32;
        let n_angles_i32 = self.n_angles as i32;
        let n_dihedrals_i32 = self.n_dihedrals as i32;
        let max_excl_i32 = MAX_EXCLUSIONS as i32;
        let max_14_i32 = MAX_14_PAIRS as i32;

        let mut total_ke = 0.0f64;
        const ENERGY_SAMPLE_INTERVAL: usize = 100;

        // Phase 2: Displacement-based neighbor list rebuild
        const DISPLACEMENT_CHECK_INTERVAL: usize = 10;

        // Compute DOF for proper temperature calculation
        // Use COM removal only for explicit solvent (PBC enabled with SETTLE)
        let has_explicit_solvent = self.pbc_enabled && self.settle.is_some();
        let n_dof = self.compute_n_dof(has_explicit_solvent);
        let constraint_info = self.get_constraint_info();

        // Initialize energy trajectory
        let mut energy_trajectory: Vec<EnergyRecord> =
            Vec::with_capacity(n_steps / ENERGY_SAMPLE_INTERVAL + 1);

        log::info!(
            "🏃 Running {} Velocity Verlet steps (2-force-eval per step)",
            n_steps
        );

        for step in 0..n_steps {
            // Phase 2: Check if neighbor list rebuild is needed based on displacement
            if step > 0 && step % DISPLACEMENT_CHECK_INTERVAL == 0 {
                let (needs_rebuild, max_disp) = self.check_neighbor_list_rebuild_needed()?;
                if needs_rebuild {
                    log::debug!(
                        "📦 Rebuilding neighbor list at step {} (max_disp={:.3} Å > threshold={:.3} Å)",
                        step, max_disp, self.rebuild_threshold
                    );
                    self.neighbor_list_valid = false;
                    self.build_neighbor_lists()?;
                }
            }

            // Save positions for SETTLE (if enabled)
            if let Some(ref mut settle) = self.settle {
                settle.save_positions(&self.d_positions)?;
            }

            // ============================================================
            // STEP 1: Compute forces at x(t)
            // ============================================================
            self.stream.memset_zeros(&mut self.d_forces)?;

            // Add position restraints (if enabled)
            if self.n_restrained > 0 {
                if let (Some(ref d_restrained), Some(ref d_ref)) =
                    (&self.d_restrained_atoms, &self.d_ref_positions)
                {
                    let restraint_kernel =
                        self.module
                            .load_function("apply_position_restraints")
                            .context("Failed to load apply_position_restraints kernel")?;

                    let n_restrained_i32 = self.n_restrained as i32;
                    let r_blocks = (self.n_restrained + threads - 1) / threads;
                    let r_cfg = LaunchConfig {
                        grid_dim: (r_blocks as u32, 1, 1),
                        block_dim: (threads as u32, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe {
                        let mut builder = self.stream.launch_builder(&restraint_kernel);
                        builder.arg(&self.d_forces);
                        builder.arg(&self.d_total_energy);
                        builder.arg(&self.d_positions);
                        builder.arg(d_ref);
                        builder.arg(d_restrained);
                        builder.arg(&n_restrained_i32);
                        builder.arg(&self.k_restraint);
                        builder.launch(r_cfg)?;
                    }
                }
            }

            // Compute bonded + non-bonded forces (FP16 or FP32 based on config)
            let use_mixed = self.mixed_precision_config.fp16_lj_params
                && self.mixed_precision_buffers.is_some();

            if use_mixed {
                // FP16 mixed precision path - ~40% bandwidth reduction
                let buffers = self.mixed_precision_buffers.as_ref().unwrap();
                unsafe {
                    let mut builder = self
                        .stream
                        .launch_builder(&self.compute_forces_mixed_kernel);
                    builder.arg(&self.d_positions);
                    builder.arg(&self.d_forces);
                    builder.arg(&self.d_total_energy);
                    builder.arg(&self.d_bond_atoms);
                    builder.arg(&self.d_bond_params);
                    builder.arg(&self.d_angle_atoms);
                    builder.arg(&self.d_angle_params);
                    builder.arg(&self.d_dihedral_atoms);
                    builder.arg(&self.d_dihedral_params);
                    builder.arg(&buffers.d_sigma_fp16);
                    builder.arg(&buffers.d_epsilon_fp16);
                    builder.arg(&self.d_nb_charge);
                    builder.arg(&self.d_nb_mass);
                    builder.arg(&self.d_exclusion_list);
                    builder.arg(&self.d_n_exclusions);
                    builder.arg(&self.d_pair14_list);
                    builder.arg(&self.d_n_pairs14);
                    builder.arg(&n_bonds_i32);
                    builder.arg(&n_angles_i32);
                    builder.arg(&n_dihedrals_i32);
                    builder.arg(&n_atoms_i32);
                    builder.arg(&max_excl_i32);
                    builder.arg(&max_14_i32);
                    builder.launch(force_cfg)?;
                }
            } else {
                // FP32 path (original)
                unsafe {
                    let mut builder = self.stream.launch_builder(&self.compute_forces_kernel);
                    builder.arg(&self.d_positions);
                    builder.arg(&self.d_forces);
                    builder.arg(&self.d_total_energy);
                    builder.arg(&self.d_bond_atoms);
                    builder.arg(&self.d_bond_params);
                    builder.arg(&self.d_angle_atoms);
                    builder.arg(&self.d_angle_params);
                    builder.arg(&self.d_dihedral_atoms);
                    builder.arg(&self.d_dihedral_params);
                    builder.arg(&self.d_nb_sigma);
                    builder.arg(&self.d_nb_epsilon);
                    builder.arg(&self.d_nb_charge);
                    builder.arg(&self.d_exclusion_list);
                    builder.arg(&self.d_n_exclusions);
                    builder.arg(&self.d_pair14_list);
                    builder.arg(&self.d_n_pairs14);
                    builder.arg(&max_excl_i32);
                    builder.arg(&max_14_i32);
                    builder.arg(&n_atoms_i32);
                    builder.arg(&n_bonds_i32);
                    builder.arg(&n_angles_i32);
                    builder.arg(&n_dihedrals_i32);
                    builder.launch(force_cfg)?;
                }
            }

            // ============================================================
            // STEP 2: First half-kick + drift: v += (dt/2)*a; x += dt*v
            // ============================================================
            unsafe {
                let mut builder = self.stream.launch_builder(&self.vv_step1_kernel);
                builder.arg(&self.d_positions);
                builder.arg(&self.d_velocities);
                builder.arg(&self.d_forces);
                builder.arg(&self.d_nb_mass);
                builder.arg(&n_atoms_i32);
                builder.arg(&dt);
                builder.launch(cfg)?;
            }

            // ============================================================
            // STEP 3: Compute forces at x(t+dt) [SECOND FORCE EVALUATION]
            // ============================================================
            self.stream.memset_zeros(&mut self.d_forces)?;

            // Add position restraints again at new positions
            if self.n_restrained > 0 {
                if let (Some(ref d_restrained), Some(ref d_ref)) =
                    (&self.d_restrained_atoms, &self.d_ref_positions)
                {
                    let restraint_kernel =
                        self.module
                            .load_function("apply_position_restraints")
                            .context("Failed to load apply_position_restraints kernel")?;

                    let n_restrained_i32 = self.n_restrained as i32;
                    let r_blocks = (self.n_restrained + threads - 1) / threads;
                    let r_cfg = LaunchConfig {
                        grid_dim: (r_blocks as u32, 1, 1),
                        block_dim: (threads as u32, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe {
                        let mut builder = self.stream.launch_builder(&restraint_kernel);
                        builder.arg(&self.d_forces);
                        builder.arg(&self.d_total_energy);
                        builder.arg(&self.d_positions);
                        builder.arg(d_ref);
                        builder.arg(d_restrained);
                        builder.arg(&n_restrained_i32);
                        builder.arg(&self.k_restraint);
                        builder.launch(r_cfg)?;
                    }
                }
            }

            // Compute bonded + non-bonded forces at new positions (FP16 or FP32)
            if use_mixed {
                let buffers = self.mixed_precision_buffers.as_ref().unwrap();
                unsafe {
                    let mut builder = self
                        .stream
                        .launch_builder(&self.compute_forces_mixed_kernel);
                    builder.arg(&self.d_positions);
                    builder.arg(&self.d_forces);
                    builder.arg(&self.d_total_energy);
                    builder.arg(&self.d_bond_atoms);
                    builder.arg(&self.d_bond_params);
                    builder.arg(&self.d_angle_atoms);
                    builder.arg(&self.d_angle_params);
                    builder.arg(&self.d_dihedral_atoms);
                    builder.arg(&self.d_dihedral_params);
                    builder.arg(&buffers.d_sigma_fp16);
                    builder.arg(&buffers.d_epsilon_fp16);
                    builder.arg(&self.d_nb_charge);
                    builder.arg(&self.d_nb_mass);
                    builder.arg(&self.d_exclusion_list);
                    builder.arg(&self.d_n_exclusions);
                    builder.arg(&self.d_pair14_list);
                    builder.arg(&self.d_n_pairs14);
                    builder.arg(&n_bonds_i32);
                    builder.arg(&n_angles_i32);
                    builder.arg(&n_dihedrals_i32);
                    builder.arg(&n_atoms_i32);
                    builder.arg(&max_excl_i32);
                    builder.arg(&max_14_i32);
                    builder.launch(force_cfg)?;
                }
            } else {
                unsafe {
                    let mut builder = self.stream.launch_builder(&self.compute_forces_kernel);
                    builder.arg(&self.d_positions);
                    builder.arg(&self.d_forces);
                    builder.arg(&self.d_total_energy);
                    builder.arg(&self.d_bond_atoms);
                    builder.arg(&self.d_bond_params);
                    builder.arg(&self.d_angle_atoms);
                    builder.arg(&self.d_angle_params);
                    builder.arg(&self.d_dihedral_atoms);
                    builder.arg(&self.d_dihedral_params);
                    builder.arg(&self.d_nb_sigma);
                    builder.arg(&self.d_nb_epsilon);
                    builder.arg(&self.d_nb_charge);
                    builder.arg(&self.d_exclusion_list);
                    builder.arg(&self.d_n_exclusions);
                    builder.arg(&self.d_pair14_list);
                    builder.arg(&self.d_n_pairs14);
                    builder.arg(&max_excl_i32);
                    builder.arg(&max_14_i32);
                    builder.arg(&n_atoms_i32);
                    builder.arg(&n_bonds_i32);
                    builder.arg(&n_angles_i32);
                    builder.arg(&n_dihedrals_i32);
                    builder.launch(force_cfg)?;
                }
            }

            // ============================================================
            // STEP 4: Second half-kick + O-step: v += (dt/2)*a; thermostat
            // ============================================================
            let step_u32 = step as u32;
            unsafe {
                let mut builder = self.stream.launch_builder(&self.vv_step2_kernel);
                builder.arg(&self.d_velocities);
                builder.arg(&self.d_kinetic_energy);
                builder.arg(&self.d_forces);
                builder.arg(&self.d_nb_mass);
                builder.arg(&n_atoms_i32);
                builder.arg(&dt);
                builder.arg(&temperature);
                builder.arg(&gamma_fs);
                builder.arg(&step_u32);
                builder.launch(cfg)?;
            }

            // Apply SETTLE constraints (if enabled)
            // Position + RATTLE velocity correction for accurate constrained motion
            if let Some(ref mut settle) = self.settle {
                self.stream.synchronize()?;
                settle.apply(&mut self.d_positions, dt)?;
                settle.apply_velocity_correction(&mut self.d_velocities, &self.d_positions)?;
            }

            // Apply H-bond constraints (if enabled)
            if let Some(ref h_constraints) = self.h_constraints {
                h_constraints.apply(&mut self.d_positions, &mut self.d_velocities, dt)?;
            }

            // Velocity rescaling for explicit solvent (after SETTLE)
            // The Langevin thermostat over-injects energy due to SETTLE velocity projection
            const VELOCITY_RESCALE_INTERVAL: usize = 10;
            if has_explicit_solvent && step % VELOCITY_RESCALE_INTERVAL == 0 && step > 0 {
                if let Err(e) = self.rescale_velocities(temperature) {
                    log::warn!("Velocity rescaling failed: {}", e);
                }
            }

            // Phase 1: Wrap positions into periodic box (after constraints)
            // This ensures molecules stay intact before being placed in the primary cell
            self.wrap_positions()?;

            // Phase 1: Remove COM drift (every com_removal_interval steps)
            // Essential for periodic systems to prevent net translational drift
            if self.com_removal_interval > 0 && step % self.com_removal_interval == 0 {
                self.remove_com_drift()?;
            }

            // Sample KE for temperature
            if step % 10 == 0 {
                self.stream.synchronize()?;
                let mut ke = vec![0.0f32; 1];
                self.stream.memcpy_dtoh(&self.d_kinetic_energy, &mut ke)?;
                total_ke += ke[0] as f64;
            }

            // Diagnostic logging and energy trajectory recording
            if step % ENERGY_SAMPLE_INTERVAL == 0 || step < 10 {
                self.stream.synchronize()?;
                let mut pe = vec![0.0f32; 1];
                let mut ke = vec![0.0f32; 1];
                self.stream.memcpy_dtoh(&self.d_total_energy, &mut pe)?;
                self.stream.memcpy_dtoh(&self.d_kinetic_energy, &mut ke)?;

                // Compute temperature using proper DOF
                let inst_temp = 2.0 * ke[0] as f64 / (n_dof as f64 * KB_KCAL_MOL_K);
                let time_ps = step as f64 * dt as f64 / 1000.0; // fs -> ps

                // Record to energy trajectory (only at ENERGY_SAMPLE_INTERVAL, not first 10 steps)
                if step % ENERGY_SAMPLE_INTERVAL == 0 {
                    energy_trajectory.push(EnergyRecord {
                        step: step as u64,
                        time_ps,
                        potential_energy: pe[0] as f64,
                        kinetic_energy: ke[0] as f64,
                        total_energy: (pe[0] + ke[0]) as f64,
                        temperature: inst_temp,
                    });
                }

                log::info!(
                    "📊 VV Step {:>5}: PE={:>12.2} KE={:>10.2} T={:>8.1}K (DOF={}) TE={:>10.2}",
                    step,
                    pe[0],
                    ke[0],
                    inst_temp,
                    n_dof,
                    pe[0] + ke[0]
                );
            }
        }

        self.stream.synchronize()?;

        // Download final results
        let mut h_total_energy = vec![0.0f32; 1];
        let mut h_kinetic_energy = vec![0.0f32; 1];
        self.stream
            .memcpy_dtoh(&self.d_total_energy, &mut h_total_energy)?;
        self.stream
            .memcpy_dtoh(&self.d_kinetic_energy, &mut h_kinetic_energy)?;

        let positions = self.get_positions()?;
        let mut velocities = vec![0.0f32; self.n_atoms * 3];
        self.stream
            .memcpy_dtoh(&self.d_velocities, &mut velocities)?;

        // Average temperature using proper DOF
        let n_samples = (n_steps / 10).max(1);
        let avg_ke = total_ke / n_samples as f64;
        let avg_temperature = 2.0 * avg_ke / (n_dof as f64 * KB_KCAL_MOL_K);

        log::info!(
            "✅ Velocity Verlet complete: PE={:.2}, KE={:.2}, T_avg={:.1}K (DOF={})",
            h_total_energy[0],
            h_kinetic_energy[0],
            avg_temperature,
            n_dof
        );

        Ok(HmcRunResult {
            potential_energy: h_total_energy[0] as f64,
            kinetic_energy: h_kinetic_energy[0] as f64,
            positions,
            velocities,
            avg_temperature,
            energy_trajectory,
            n_dof,
            constraint_info,
        })
    }

    /// Get current velocities
    pub fn get_velocities(&self) -> Result<Vec<f32>> {
        let mut velocities = vec![0.0f32; self.n_atoms * 3];
        self.stream
            .memcpy_dtoh(&self.d_velocities, &mut velocities)?;
        Ok(velocities)
    }

    /// Compute forces on current positions and return max force magnitude
    ///
    /// This is useful for checking if a structure is well-minimized before dynamics.
    /// Forces should be < 50 kcal/(mol·Å) for stable dynamics.
    pub fn get_max_force(&mut self) -> Result<f32> {
        if !self.topology_ready {
            return Err(anyhow::anyhow!("Topology not uploaded"));
        }

        let threads_per_block = 256;
        let num_blocks = (self
            .n_atoms
            .max(self.n_bonds)
            .max(self.n_angles)
            .max(self.n_dihedrals)
            + threads_per_block
            - 1)
            / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 6 * 64 * 4,
        };

        // Load minimize kernel (computes forces without integrating)
        let force_kernel = self
            .module
            .load_function("amber_steepest_descent_step")
            .context("Failed to load amber_steepest_descent_step")?;

        // Zero forces
        self.stream.memset_zeros(&mut self.d_forces)?;

        // Add PME forces if enabled
        if let Some(ref mut pme) = self.pme {
            self.stream.synchronize()?;
            pme.compute(&self.d_positions, &self.d_nb_charge, &mut self.d_forces)?;
        }

        // Compute bonded + NB forces (with step_size=0 to not move atoms)
        let max_excl_i32 = MAX_EXCLUSIONS as i32;
        let max_14_i32 = MAX_14_PAIRS as i32;
        let n_atoms_i32 = self.n_atoms as i32;
        let n_bonds_i32 = self.n_bonds as i32;
        let n_angles_i32 = self.n_angles as i32;
        let n_dihedrals_i32 = self.n_dihedrals as i32;
        let step_size = 0.0f32; // Don't move atoms

        unsafe {
            let mut builder = self.stream.launch_builder(&force_kernel);
            builder.arg(&self.d_positions);
            builder.arg(&self.d_forces);
            builder.arg(&self.d_total_energy);
            builder.arg(&self.d_bond_atoms);
            builder.arg(&self.d_bond_params);
            builder.arg(&self.d_angle_atoms);
            builder.arg(&self.d_angle_params);
            builder.arg(&self.d_dihedral_atoms);
            builder.arg(&self.d_dihedral_params);
            builder.arg(&self.d_nb_sigma);
            builder.arg(&self.d_nb_epsilon);
            builder.arg(&self.d_nb_charge);
            builder.arg(&self.d_exclusion_list);
            builder.arg(&self.d_n_exclusions);
            builder.arg(&self.d_pair14_list);
            builder.arg(&self.d_n_pairs14);
            builder.arg(&max_excl_i32);
            builder.arg(&max_14_i32);
            builder.arg(&n_atoms_i32);
            builder.arg(&n_bonds_i32);
            builder.arg(&n_angles_i32);
            builder.arg(&n_dihedrals_i32);
            builder.arg(&step_size);
            builder.launch(cfg)?;
        }

        self.stream.synchronize()?;

        // Download forces and compute max
        let mut forces = vec![0.0f32; self.n_atoms * 3];
        self.stream.memcpy_dtoh(&self.d_forces, &mut forces)?;

        let mut max_f2 = 0.0f32;
        for i in 0..self.n_atoms {
            let fx = forces[i * 3];
            let fy = forces[i * 3 + 1];
            let fz = forces[i * 3 + 2];
            let f2 = fx * fx + fy * fy + fz * fz;
            if f2 > max_f2 {
                max_f2 = f2;
            }
        }

        Ok(max_f2.sqrt())
    }

    /// Get detailed force diagnostics: max force, average, and top high-force atoms
    ///
    /// Returns (max_force, avg_force, top_10_indices_with_forces)
    /// where each tuple in top_10 is (atom_index, force_magnitude)
    pub fn get_force_diagnostics(&mut self) -> Result<(f32, f32, Vec<(usize, f32)>)> {
        if !self.topology_ready {
            return Err(anyhow::anyhow!("Topology not uploaded"));
        }

        let threads_per_block = 256;
        let num_blocks = (self
            .n_atoms
            .max(self.n_bonds)
            .max(self.n_angles)
            .max(self.n_dihedrals)
            + threads_per_block
            - 1)
            / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 6 * 64 * 4,
        };

        // Load minimize kernel (computes forces without integrating)
        let force_kernel = self
            .module
            .load_function("amber_steepest_descent_step")
            .context("Failed to load amber_steepest_descent_step")?;

        // Zero forces
        self.stream.memset_zeros(&mut self.d_forces)?;

        // Add PME forces if enabled
        if let Some(ref mut pme) = self.pme {
            self.stream.synchronize()?;
            pme.compute(&self.d_positions, &self.d_nb_charge, &mut self.d_forces)?;
        }

        // Compute bonded + NB forces (with step_size=0 to not move atoms)
        let max_excl_i32 = MAX_EXCLUSIONS as i32;
        let max_14_i32 = MAX_14_PAIRS as i32;
        let n_atoms_i32 = self.n_atoms as i32;
        let n_bonds_i32 = self.n_bonds as i32;
        let n_angles_i32 = self.n_angles as i32;
        let n_dihedrals_i32 = self.n_dihedrals as i32;
        let step_size = 0.0f32;

        unsafe {
            let mut builder = self.stream.launch_builder(&force_kernel);
            builder.arg(&self.d_positions);
            builder.arg(&self.d_forces);
            builder.arg(&self.d_total_energy);
            builder.arg(&self.d_bond_atoms);
            builder.arg(&self.d_bond_params);
            builder.arg(&self.d_angle_atoms);
            builder.arg(&self.d_angle_params);
            builder.arg(&self.d_dihedral_atoms);
            builder.arg(&self.d_dihedral_params);
            builder.arg(&self.d_nb_sigma);
            builder.arg(&self.d_nb_epsilon);
            builder.arg(&self.d_nb_charge);
            builder.arg(&self.d_exclusion_list);
            builder.arg(&self.d_n_exclusions);
            builder.arg(&self.d_pair14_list);
            builder.arg(&self.d_n_pairs14);
            builder.arg(&max_excl_i32);
            builder.arg(&max_14_i32);
            builder.arg(&n_atoms_i32);
            builder.arg(&n_bonds_i32);
            builder.arg(&n_angles_i32);
            builder.arg(&n_dihedrals_i32);
            builder.arg(&step_size);
            builder.launch(cfg)?;
        }

        self.stream.synchronize()?;

        // Download forces
        let mut forces = vec![0.0f32; self.n_atoms * 3];
        self.stream.memcpy_dtoh(&self.d_forces, &mut forces)?;

        // Compute statistics and find top high-force atoms
        let mut max_force = 0.0f32;
        let mut total_force = 0.0f32;
        let mut atom_forces: Vec<(usize, f32)> = Vec::with_capacity(self.n_atoms);

        for i in 0..self.n_atoms {
            let fx = forces[i * 3];
            let fy = forces[i * 3 + 1];
            let fz = forces[i * 3 + 2];
            let f_mag = (fx * fx + fy * fy + fz * fz).sqrt();

            if f_mag > max_force {
                max_force = f_mag;
            }
            total_force += f_mag;
            atom_forces.push((i, f_mag));
        }

        // Sort by force magnitude (descending) and take top 10
        atom_forces.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_10: Vec<(usize, f32)> = atom_forces.into_iter().take(10).collect();

        let avg_force = total_force / self.n_atoms as f32;

        Ok((max_force, avg_force, top_10))
    }

    /// Rescale velocities to target temperature (velocity rescaling thermostat)
    ///
    /// This maintains the canonical ensemble by scaling all velocities to achieve
    /// the target temperature, compensating for numerical integration drift.
    pub fn rescale_velocities(&mut self, target_temperature: f32) -> Result<()> {
        // Track rescale calls for debugging
        static RESCALE_CALL_COUNT: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(0);
        let call_count = RESCALE_CALL_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        if !self.velocities_initialized {
            log::warn!("rescale_velocities called but velocities not initialized");
            return Err(anyhow::anyhow!("Velocities not initialized"));
        }

        // Download current velocities and masses
        let mut velocities = vec![0.0f32; self.n_atoms * 3];
        let mut masses = vec![0.0f32; self.n_atoms];
        self.stream
            .memcpy_dtoh(&self.d_velocities, &mut velocities)?;
        self.stream.memcpy_dtoh(&self.d_nb_mass, &mut masses)?;

        // Calculate current kinetic energy: KE = 0.5 * sum(m_i * v_i^2) / FORCE_TO_ACCEL
        // FORCE_TO_ACCEL = 4.184e-4 converts velocity²*mass to kcal/mol
        // Without this factor, the KE would be in wrong units (g/mol * Å²/fs²)
        const FORCE_TO_ACCEL: f64 = 4.184e-4;
        let mut kinetic_energy = 0.0f64;
        for i in 0..self.n_atoms {
            let vx = velocities[i * 3] as f64;
            let vy = velocities[i * 3 + 1] as f64;
            let vz = velocities[i * 3 + 2] as f64;
            let m = masses[i] as f64;
            kinetic_energy += 0.5 * m * (vx * vx + vy * vy + vz * vz) / FORCE_TO_ACCEL;
        }

        // Calculate current temperature: T = 2*KE / (n_dof * kb)
        // kb = 0.001987204 kcal/(mol*K)
        // Use proper DOF that accounts for SETTLE constraints (3 distance constraints per water)
        // and COM removal for explicit solvent systems.
        let has_explicit_solvent = self.pbc_enabled && self.settle.is_some();
        let n_dof = self.compute_n_dof(has_explicit_solvent) as f64;
        let current_temp = 2.0 * kinetic_energy / (n_dof * KB_KCAL_MOL_K);

        if current_temp < 1.0 {
            // Temperature too low, reinitialize
            log::warn!(
                "Temperature too low ({:.1}K), reinitializing velocities",
                current_temp
            );
            return self.initialize_velocities(target_temperature);
        }

        // Calculate scaling factor: lambda = sqrt(T_target / T_current)
        let scale_factor = ((target_temperature as f64) / current_temp).sqrt();

        // Scale all velocities
        for v in velocities.iter_mut() {
            *v *= scale_factor as f32;
        }

        // Upload scaled velocities back to GPU
        self.stream
            .memcpy_htod(&velocities, &mut self.d_velocities)?;

        // Log every 10000 calls to avoid spam
        if call_count % 10000 == 0 {
            eprintln!(
                "🌡️ Rescale #{}: {:.1}K → {:.1}K (factor: {:.4}, n_dof={})",
                call_count, current_temp, target_temperature, scale_factor, n_dof as usize
            );
        }

        Ok(())
    }

    // =========================================================================
    // Phase 7: Mixed Precision Methods
    // =========================================================================

    /// Enable mixed precision computation with the specified configuration
    ///
    /// This allocates FP16 buffers and uploads LJ parameters in half precision.
    /// Call this AFTER upload_topology() to ensure parameters are available.
    pub fn enable_mixed_precision(&mut self, config: MixedPrecisionConfig) -> Result<()> {
        if !config.is_enabled() {
            log::info!("Mixed precision disabled (full FP32 mode)");
            self.mixed_precision_config = config;
            self.mixed_precision_buffers = None;
            return Ok(());
        }

        log::info!(
            "⚡ Enabling mixed precision: FP16_LJ={}, FP16_PME={}, Half2={}",
            config.fp16_lj_params,
            config.fp16_pme_grid,
            config.half2_lj
        );

        // Allocate FP16 buffers
        let mut buffers = MixedPrecisionBuffers::new(&self.stream, self.n_atoms)?;

        // If topology is already uploaded, convert and upload FP16 params
        if self.topology_ready && config.fp16_lj_params {
            // Download current FP32 params
            let mut sigma = vec![0.0f32; self.n_atoms];
            let mut epsilon = vec![0.0f32; self.n_atoms];

            self.stream.memcpy_dtoh(&self.d_nb_sigma, &mut sigma)?;
            self.stream.memcpy_dtoh(&self.d_nb_epsilon, &mut epsilon)?;

            // Upload as FP16
            buffers.upload_from_fp32(&self.stream, &sigma, &epsilon)?;

            log::info!(
                "✅ Converted {} atoms to FP16: saved {} bytes",
                self.n_atoms,
                self.n_atoms * 4 // 4 bytes saved per atom (sigma + epsilon: FP32→FP16)
            );
        }

        self.mixed_precision_config = config;
        self.mixed_precision_buffers = Some(buffers);

        Ok(())
    }

    /// Disable mixed precision (return to full FP32)
    pub fn disable_mixed_precision(&mut self) {
        self.mixed_precision_config = MixedPrecisionConfig::full_precision();
        self.mixed_precision_buffers = None;
        log::info!("Mixed precision disabled");
    }

    /// Check if mixed precision is currently enabled
    pub fn is_mixed_precision_enabled(&self) -> bool {
        self.mixed_precision_config.is_enabled()
    }

    /// Get current mixed precision configuration
    pub fn mixed_precision_config(&self) -> &MixedPrecisionConfig {
        &self.mixed_precision_config
    }

    /// Get FP16 buffer memory usage in bytes
    pub fn mixed_precision_memory_bytes(&self) -> usize {
        self.mixed_precision_buffers
            .as_ref()
            .map(|b| b.memory_bytes())
            .unwrap_or(0)
    }

    /// Update FP16 buffers after topology change
    ///
    /// Call this if LJ parameters are modified after initial upload.
    pub fn sync_mixed_precision_buffers(&mut self) -> Result<()> {
        if !self.mixed_precision_config.fp16_lj_params {
            return Ok(());
        }

        let buffers = match &mut self.mixed_precision_buffers {
            Some(b) => b,
            None => return Ok(()),
        };

        // Download current FP32 params and re-upload as FP16
        let mut sigma = vec![0.0f32; self.n_atoms];
        let mut epsilon = vec![0.0f32; self.n_atoms];

        self.stream.memcpy_dtoh(&self.d_nb_sigma, &mut sigma)?;
        self.stream.memcpy_dtoh(&self.d_nb_epsilon, &mut epsilon)?;

        buffers.upload_from_fp32(&self.stream, &sigma, &epsilon)?;

        log::debug!("Synced FP16 LJ buffers with FP32 source");
        Ok(())
    }

    // =========================================================================
    // Phase 8: Fused Kernel Methods
    // =========================================================================

    /// Run a complete MD step using the mega-fused kernel (Phase 8)
    ///
    /// This combines force calculation and velocity Verlet integration into a
    /// single kernel launch, reducing launch overhead by ~60%.
    ///
    /// # Arguments
    /// * `dt` - Timestep in femtoseconds
    /// * `temperature` - Target temperature in Kelvin
    /// * `gamma_ps` - Friction coefficient in ps^-1
    /// * `seed` - Random seed for Langevin thermostat
    ///
    /// # Returns
    /// Kinetic energy after the step
    pub fn run_fused_md_step(
        &mut self,
        dt: f32,
        temperature: f32,
        gamma_ps: f32,
        seed: u32,
    ) -> Result<f32> {
        if self.n_atoms == 0 {
            anyhow::bail!("No atoms uploaded");
        }

        let threads = 256;
        let blocks = (self.n_atoms + threads - 1) / threads;

        let n_atoms_i32 = self.n_atoms as i32;
        let n_bonds_i32 = self.n_bonds as i32;
        let n_angles_i32 = self.n_angles as i32;
        let n_dihedrals_i32 = self.n_dihedrals as i32;
        let max_excl_i32 = MAX_EXCLUSIONS as i32;
        let max_14_i32 = MAX_14_PAIRS as i32;

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let use_neighbor_list: i32 = if self.pbc_enabled { 1 } else { 0 };

        // Check if mixed precision is enabled and FP16 buffers are available
        let use_mixed =
            self.mixed_precision_config.fp16_lj_params && self.mixed_precision_buffers.is_some();

        if use_mixed {
            // Use FP16 kernel with mixed precision LJ parameters
            let buffers = self.mixed_precision_buffers.as_ref().unwrap();
            unsafe {
                let mut builder = self
                    .stream
                    .launch_builder(&self.mega_fused_md_step_mixed_kernel);
                // Positions and velocities
                builder.arg(&self.d_positions);
                builder.arg(&self.d_velocities);
                builder.arg(&self.d_forces);
                builder.arg(&self.d_total_energy);
                builder.arg(&self.d_kinetic_energy);
                // Topology - Bonds
                builder.arg(&self.d_bond_atoms);
                builder.arg(&self.d_bond_params);
                // Topology - Angles
                builder.arg(&self.d_angle_atoms);
                builder.arg(&self.d_angle_params);
                // Topology - Dihedrals
                builder.arg(&self.d_dihedral_atoms);
                builder.arg(&self.d_dihedral_params);
                // Non-bonded parameters - FP16 for sigma/epsilon
                builder.arg(&buffers.d_sigma_fp16);
                builder.arg(&buffers.d_epsilon_fp16);
                builder.arg(&self.d_nb_charge);
                builder.arg(&self.d_nb_mass);
                builder.arg(&self.d_exclusion_list);
                builder.arg(&self.d_n_exclusions);
                builder.arg(&self.d_pair14_list);
                builder.arg(&self.d_n_pairs14);
                // Neighbor list
                builder.arg(&self.d_neighbor_list);
                builder.arg(&self.d_n_neighbors);
                builder.arg(&use_neighbor_list);
                // Configuration
                builder.arg(&n_atoms_i32);
                builder.arg(&n_bonds_i32);
                builder.arg(&n_angles_i32);
                builder.arg(&n_dihedrals_i32);
                builder.arg(&max_excl_i32);
                builder.arg(&max_14_i32);
                builder.arg(&dt);
                builder.arg(&temperature);
                builder.arg(&gamma_ps);
                builder.arg(&seed);
                builder.launch(cfg)?;
            }
        } else {
            // Use FP32 kernel (original path)
            unsafe {
                let mut builder = self.stream.launch_builder(&self.mega_fused_md_step_kernel);
                // Positions and velocities
                builder.arg(&self.d_positions);
                builder.arg(&self.d_velocities);
                builder.arg(&self.d_forces);
                builder.arg(&self.d_total_energy);
                builder.arg(&self.d_kinetic_energy);
                // Topology - Bonds
                builder.arg(&self.d_bond_atoms);
                builder.arg(&self.d_bond_params);
                // Topology - Angles
                builder.arg(&self.d_angle_atoms);
                builder.arg(&self.d_angle_params);
                // Topology - Dihedrals
                builder.arg(&self.d_dihedral_atoms);
                builder.arg(&self.d_dihedral_params);
                // Non-bonded parameters - FP32
                builder.arg(&self.d_nb_sigma);
                builder.arg(&self.d_nb_epsilon);
                builder.arg(&self.d_nb_charge);
                builder.arg(&self.d_nb_mass);
                builder.arg(&self.d_exclusion_list);
                builder.arg(&self.d_n_exclusions);
                builder.arg(&self.d_pair14_list);
                builder.arg(&self.d_n_pairs14);
                // Neighbor list
                builder.arg(&self.d_neighbor_list);
                builder.arg(&self.d_n_neighbors);
                builder.arg(&use_neighbor_list);
                // Configuration
                builder.arg(&n_atoms_i32);
                builder.arg(&n_bonds_i32);
                builder.arg(&n_angles_i32);
                builder.arg(&n_dihedrals_i32);
                builder.arg(&max_excl_i32);
                builder.arg(&max_14_i32);
                builder.arg(&dt);
                builder.arg(&temperature);
                builder.arg(&gamma_ps);
                builder.arg(&seed);
                builder.launch(cfg)?;
            }
        }

        self.stream.synchronize()?;

        let mut ke = [0.0f32];
        self.stream.memcpy_dtoh(&self.d_kinetic_energy, &mut ke)?;

        Ok(ke[0])
    }

    /// Run a complete MD step using the tiled mega-fused kernel (Phase 8)
    ///
    /// This uses shared memory tiling for improved memory bandwidth utilization.
    /// Best for systems with >1000 atoms.
    ///
    /// # Arguments
    /// * `dt` - Timestep in femtoseconds
    /// * `temperature` - Target temperature in Kelvin
    /// * `gamma_ps` - Friction coefficient in ps^-1
    /// * `seed` - Random seed for Langevin thermostat
    ///
    /// # Returns
    /// Kinetic energy after the step
    pub fn run_fused_md_step_tiled(
        &mut self,
        dt: f32,
        temperature: f32,
        gamma_ps: f32,
        seed: u32,
    ) -> Result<f32> {
        if self.n_atoms == 0 {
            anyhow::bail!("No atoms uploaded");
        }

        let threads = 256;
        let blocks = (self.n_atoms + threads - 1) / threads;

        let n_atoms_i32 = self.n_atoms as i32;
        let n_bonds_i32 = self.n_bonds as i32;
        let n_angles_i32 = self.n_angles as i32;
        let n_dihedrals_i32 = self.n_dihedrals as i32;
        let max_excl_i32 = MAX_EXCLUSIONS as i32;
        let max_14_i32 = MAX_14_PAIRS as i32;

        // Shared memory for 128-atom tiles: positions + params
        let shared_mem = 128 * (3 * 4 + 3 * 4); // 128 atoms * (xyz + sigma,eps,charge)

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: shared_mem as u32,
        };

        // Parameter order must match CUDA kernel mega_fused_md_step_tiled signature:
        // positions, velocities, forces, potential_energy, kinetic_energy,
        // bond_atoms, bond_params, angle_atoms, angle_params, dihedral_atoms, dihedral_params,
        // nb_sigma, nb_epsilon, nb_charge, nb_mass, excl_list, n_excl, pair14_list, n_pairs14,
        // n_atoms, n_bonds, n_angles, n_dihedrals, max_excl, max_14,
        // dt, temperature, gamma_fs, step
        // NOTE: tiled version does NOT use neighbor lists

        unsafe {
            let mut builder = self
                .stream
                .launch_builder(&self.mega_fused_md_step_tiled_kernel);
            // Positions and velocities
            builder.arg(&self.d_positions);
            builder.arg(&self.d_velocities);
            builder.arg(&self.d_forces);
            builder.arg(&self.d_total_energy);
            builder.arg(&self.d_kinetic_energy);
            // Topology - Bonds
            builder.arg(&self.d_bond_atoms);
            builder.arg(&self.d_bond_params);
            // Topology - Angles
            builder.arg(&self.d_angle_atoms);
            builder.arg(&self.d_angle_params);
            // Topology - Dihedrals
            builder.arg(&self.d_dihedral_atoms);
            builder.arg(&self.d_dihedral_params);
            // Non-bonded parameters
            builder.arg(&self.d_nb_sigma);
            builder.arg(&self.d_nb_epsilon);
            builder.arg(&self.d_nb_charge);
            builder.arg(&self.d_nb_mass);
            builder.arg(&self.d_exclusion_list);
            builder.arg(&self.d_n_exclusions);
            builder.arg(&self.d_pair14_list);
            builder.arg(&self.d_n_pairs14);
            // Configuration (no neighbor list for tiled version)
            builder.arg(&n_atoms_i32);
            builder.arg(&n_bonds_i32);
            builder.arg(&n_angles_i32);
            builder.arg(&n_dihedrals_i32);
            builder.arg(&max_excl_i32);
            builder.arg(&max_14_i32);
            builder.arg(&dt);
            builder.arg(&temperature);
            builder.arg(&gamma_ps);
            builder.arg(&seed);
            builder.launch(cfg)?;
        }

        self.stream.synchronize()?;

        let mut ke = [0.0f32];
        self.stream.memcpy_dtoh(&self.d_kinetic_energy, &mut ke)?;

        Ok(ke[0])
    }

    /// Run fused constraints (SETTLE + H-constraints) in a single kernel launch
    ///
    /// This reduces kernel launch overhead when both water and protein
    /// constraints need to be applied.
    ///
    /// # Arguments
    /// * `dt` - Timestep in femtoseconds (needed for velocity correction)
    ///
    /// # Returns
    /// Number of constraint iterations performed
    pub fn run_fused_constraints(&mut self, dt: f32) -> Result<i32> {
        // Check if we have any constraints
        let n_water = self.settle.as_ref().map(|s| s.n_waters()).unwrap_or(0);
        let n_h_clusters = self
            .h_constraints
            .as_ref()
            .map(|h| h.n_clusters())
            .unwrap_or(0);

        if n_water == 0 && n_h_clusters == 0 {
            return Ok(0); // No constraints to apply
        }

        // For now, fall back to separate constraint applications
        // The fused kernel requires unified cluster format which needs
        // additional buffer setup

        let mut iterations = 0;

        // Apply SETTLE for water (position + RATTLE velocity correction)
        // RATTLE velocity correction projects out constraint-violating components
        if let Some(ref mut settle) = self.settle {
            settle.apply(&mut self.d_positions, dt)?;
            settle.apply_velocity_correction(&mut self.d_velocities, &self.d_positions)?;
            iterations += 1;
        }

        // Apply H-constraints for protein
        if let Some(ref h_constraints) = self.h_constraints {
            h_constraints.apply(&mut self.d_positions, &mut self.d_velocities, dt)?;
            iterations += 1;
        }

        Ok(iterations)
    }

    /// Run multiple MD steps using fused kernels (Phase 8 optimized)
    ///
    /// This is the most efficient method for production simulations,
    /// using mega-fused kernels to minimize kernel launch overhead.
    ///
    /// # Arguments
    /// * `n_steps` - Number of integration steps
    /// * `dt` - Timestep in femtoseconds
    /// * `temperature` - Target temperature in Kelvin
    /// * `gamma_fs` - Friction coefficient in fs^-1 (converted to ps^-1 internally)
    /// * `use_tiled` - Use shared memory tiled kernel (better for >1000 atoms)
    ///
    /// # Returns
    /// HmcRunResult with energy trajectory and statistics
    pub fn run_fused(
        &mut self,
        n_steps: usize,
        dt: f32,
        temperature: f32,
        gamma_fs: f32,
        use_tiled: bool,
    ) -> Result<HmcRunResult> {
        let has_explicit_solvent = self.pbc_enabled && self.settle.is_some();
        eprintln!(
            ">>> run_fused CALLED: n_steps={}, pbc_enabled={}, settle={}, has_explicit_solvent={}",
            n_steps,
            self.pbc_enabled,
            self.settle.is_some(),
            has_explicit_solvent
        );

        if self.n_atoms == 0 {
            anyhow::bail!("No atoms loaded");
        }

        // NOTE: gamma_fs is friction coefficient in fs^-1 (NOT ps^-1)
        // The CUDA kernel uses: c = exp(-gamma_fs * dt) where dt is in fs
        // Typical value: 0.001 fs^-1 = 1 ps^-1 for production
        //                0.01 fs^-1 = 10 ps^-1 for equilibration

        // TEMPERATURE SCALING FOR EXPLICIT SOLVENT:
        // The Langevin thermostat targets 3N DOF (all velocities), but with SETTLE:
        // - Each water has 9 velocity DOF but only 3 translational DOF after SETTLE
        // - The thermostat heats all 9 DOF but SETTLE projects to 3, causing T mismatch
        // - Scale factor = 3/(9) * correction = 0.333 * 1.29 ≈ 0.43 (theory)
        // - Empirical: 0.598 gives correct temperature (240K → 310K when 0.4634 was used)
        //
        // This scaling compensates for the DOF mismatch between thermostat and constraints.
        let kernel_temperature = if has_explicit_solvent {
            // Scale temperature for explicit solvent with SETTLE constraints
            const EXPLICIT_SOLVENT_TEMP_SCALE: f32 = 0.598;
            let scaled = temperature * EXPLICIT_SOLVENT_TEMP_SCALE;
            eprintln!(
                ">>> EXPLICIT SOLVENT: scaling kernel temp {} K → {:.1} K (factor {})",
                temperature, scaled, EXPLICIT_SOLVENT_TEMP_SCALE
            );
            scaled
        } else {
            temperature
        };

        let mut energies = Vec::with_capacity(n_steps);
        let base_seed = 42u32;

        for step in 0..n_steps {
            let seed = base_seed.wrapping_add(step as u32);

            // ============================================================
            // FORCE PREPARATION: Zero + PME + Restraints BEFORE fused kernel
            // ============================================================
            // The fused kernel ADDS its forces to existing values, so we must:
            //   1. Zero forces
            //   2. Add PME reciprocal forces (if explicit solvent)
            //   3. Add position restraint forces (if enabled)
            //   4. Then fused kernel adds bonded + short-range NB
            self.stream.memset_zeros(&mut self.d_forces)?;

            // Compute PME long-range electrostatics (CRITICAL for explicit solvent!)
            if let Some(ref mut pme) = self.pme {
                self.stream.synchronize()?;
                let _pme_energy =
                    pme.compute(&self.d_positions, &self.d_nb_charge, &mut self.d_forces)?;
            }

            // Add position restraint forces (if enabled)
            if self.n_restrained > 0 {
                if let (Some(ref d_restrained), Some(ref d_ref)) =
                    (&self.d_restrained_atoms, &self.d_ref_positions)
                {
                    let restraint_kernel =
                        self.module
                            .load_function("apply_position_restraints")
                            .context("Failed to load apply_position_restraints kernel")?;

                    let threads = 256;
                    let blocks = (self.n_restrained + threads - 1) / threads;
                    let cfg = LaunchConfig {
                        grid_dim: (blocks as u32, 1, 1),
                        block_dim: (threads as u32, 1, 1),
                        shared_mem_bytes: 0,
                    };

                    let n_restrained_i32 = self.n_restrained as i32;
                    unsafe {
                        let mut builder = self.stream.launch_builder(&restraint_kernel);
                        builder.arg(&self.d_forces);
                        builder.arg(&self.d_total_energy);
                        builder.arg(&self.d_positions);
                        builder.arg(d_ref);
                        builder.arg(d_restrained);
                        builder.arg(&n_restrained_i32);
                        builder.arg(&self.k_restraint);
                        builder.launch(cfg)?;
                    }
                }
            }

            self.stream.synchronize()?;

            // ============================================================
            // Run fused MD step - kernel ADDS bonded + short-range NB forces
            // ============================================================
            let ke = if use_tiled && self.n_atoms > 1000 {
                self.run_fused_md_step_tiled(dt, kernel_temperature, gamma_fs, seed)?
            } else {
                self.run_fused_md_step(dt, kernel_temperature, gamma_fs, seed)?
            };

            // Apply constraints if needed
            if self.settle.is_some() || self.h_constraints.is_some() {
                self.run_fused_constraints(dt)?;
            }

            // VELOCITY RESCALING FALLBACK: Since Langevin thermostat isn't controlling temperature,
            // use simple velocity rescaling every 10 steps to maintain target temperature.
            // This is cruder but will definitively control temperature.
            // NOTE: Must recalculate KE after SETTLE since it changes velocities!
            const RESCALE_INTERVAL: usize = 10;
            if step > 0 && step % RESCALE_INTERVAL == 0 {
                // Download velocities and masses to compute actual KE after constraints
                let mut velocities = vec![0.0f32; self.n_atoms * 3];
                self.stream
                    .memcpy_dtoh(&self.d_velocities, &mut velocities)?;

                let mut masses = vec![0.0f32; self.n_atoms];
                self.stream.memcpy_dtoh(&self.d_nb_mass, &mut masses)?;

                // Compute kinetic energy: KE = 0.5 * sum(m * v²)
                let mut ke_actual = 0.0f64;
                for i in 0..self.n_atoms {
                    let vx = velocities[i * 3] as f64;
                    let vy = velocities[i * 3 + 1] as f64;
                    let vz = velocities[i * 3 + 2] as f64;
                    let m = masses[i] as f64;
                    ke_actual += 0.5 * m * (vx * vx + vy * vy + vz * vz) / 4.184e-4;
                    // Convert to kcal/mol
                }

                let n_dof = self.compute_n_dof(true);
                let current_temp = (2.0 * ke_actual) / (n_dof as f64 * KB_KCAL_MOL_K);

                if current_temp > 1.0 {
                    let scale = ((temperature as f64) / current_temp).sqrt() as f32;

                    // Scale all velocities
                    for v in velocities.iter_mut() {
                        *v *= scale;
                    }

                    self.stream
                        .memcpy_htod(&velocities, &mut self.d_velocities)?;
                    self.stream.synchronize()?;

                    if step % 100 == 0 {
                        eprintln!(
                            ">>> RESCALE step {}: T={:.1}K → {:.1}K (scale={:.4}, n_dof={})",
                            step, current_temp, temperature, scale, n_dof
                        );
                    }
                }
            }

            // Record energy
            let mut pe = [0.0f32];
            self.stream.memcpy_dtoh(&self.d_total_energy, &mut pe)?;

            let temp = if self.n_atoms > 0 {
                let n_dof = self.compute_n_dof(true);
                (2.0 * ke as f64) / (n_dof as f64 * KB_KCAL_MOL_K)
            } else {
                0.0
            };

            energies.push(EnergyRecord {
                step: step as u64,
                time_ps: (step as f64) * (dt as f64) / 1000.0,
                potential_energy: pe[0] as f64,
                kinetic_energy: ke as f64,
                total_energy: (pe[0] + ke) as f64,
                temperature: temp,
            });

            // Periodic neighbor list rebuild check
            if step % 20 == 0 && self.pbc_enabled {
                let (needs_rebuild, _) = self.check_neighbor_list_rebuild_needed()?;
                if needs_rebuild {
                    self.build_neighbor_lists()?;
                    self.save_positions_at_build()?;
                }
            }
        }

        // Compute statistics
        let n = energies.len();
        let avg_temp = energies.iter().map(|e| e.temperature).sum::<f64>() / n as f64;
        let avg_pe = energies.iter().map(|e| e.potential_energy).sum::<f64>() / n as f64;
        let avg_ke = energies.iter().map(|e| e.kinetic_energy).sum::<f64>() / n as f64;

        // Get final positions and velocities
        let positions = self.get_positions()?;
        let velocities = self.get_velocities()?;

        Ok(HmcRunResult {
            potential_energy: avg_pe,
            kinetic_energy: avg_ke,
            positions,
            velocities,
            avg_temperature: avg_temp,
            energy_trajectory: energies,
            n_dof: self.compute_n_dof(true),
            constraint_info: self.get_constraint_info(),
        })
    }
}

/// Build exclusion lists from bond topology
pub fn build_exclusion_lists(
    bonds: &[(usize, usize, f32, f32)],
    angles: &[(usize, usize, usize, f32, f32)],
    n_atoms: usize,
) -> Vec<HashSet<usize>> {
    let mut exclusions = vec![HashSet::new(); n_atoms];

    // 1-2 exclusions (bonded pairs)
    for &(i, j, _, _) in bonds {
        if i < n_atoms && j < n_atoms {
            exclusions[i].insert(j);
            exclusions[j].insert(i);
        }
    }

    // 1-3 exclusions (angle endpoints)
    for &(i, _j, k, _, _) in angles {
        if i < n_atoms && k < n_atoms {
            exclusions[i].insert(k);
            exclusions[k].insert(i);
        }
    }

    exclusions
}

/// Build 1-4 pair list from dihedral topology
///
/// 1-4 pairs are atoms separated by exactly 3 bonds (the first and last atoms of each dihedral).
/// These need SCALED non-bonded interactions (AMBER ff14SB: LJ*0.5, Coulomb*0.833).
///
/// Returns: Vec of (atom_i, atom_j) pairs, deduplicated
pub fn build_14_pairs(
    dihedrals: &[(usize, usize, usize, usize, f32, f32, f32)],
    exclusions: &[HashSet<usize>],
    n_atoms: usize,
) -> Vec<(usize, usize)> {
    let mut pairs_14: HashSet<(usize, usize)> = HashSet::new();

    for &(i, _j, _k, l, _, _, _) in dihedrals {
        if i >= n_atoms || l >= n_atoms {
            continue;
        }

        // Skip if this pair is already a 1-2 or 1-3 exclusion
        // (can happen with ring systems)
        if exclusions[i].contains(&l) {
            continue;
        }

        // Canonicalize order (smaller index first) to avoid duplicates
        let pair = if i < l { (i, l) } else { (l, i) };
        pairs_14.insert(pair);
    }

    let mut result: Vec<_> = pairs_14.into_iter().collect();
    result.sort(); // Sort for deterministic ordering
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_exclusion_lists() {
        let bonds = vec![(0, 1, 100.0, 1.5), (1, 2, 100.0, 1.5)];
        let angles = vec![(0, 1, 2, 50.0, 2.0)];
        let exclusions = build_exclusion_lists(&bonds, &angles, 3);

        assert!(exclusions[0].contains(&1));
        assert!(exclusions[0].contains(&2));
        assert!(exclusions[1].contains(&0));
        assert!(exclusions[1].contains(&2));
    }

    // =========================================================================
    // Phase 7: FP16 Conversion Tests
    // =========================================================================

    #[test]
    fn test_f32_to_f16_roundtrip_normal() {
        // Test normal values typical for LJ parameters
        let test_values = [
            0.0f32, 1.0, -1.0, 0.5, 2.0, 3.4,  // Typical sigma in Angstroms
            0.1,  // Typical epsilon in kcal/mol
            0.01, // Small epsilon
            10.0, // Large value
            100.0,
        ];

        for &v in &test_values {
            let bits = f32_to_f16_bits(v);
            let back = f16_bits_to_f32(bits);

            if v == 0.0 {
                assert_eq!(back, 0.0, "Zero should roundtrip exactly");
            } else {
                let rel_error = ((back - v) / v).abs();
                assert!(
                    rel_error < 0.001,
                    "Value {} -> {} has error {:.4}% (max 0.1%)",
                    v,
                    back,
                    rel_error * 100.0
                );
            }
        }
    }

    #[test]
    fn test_f32_to_f16_typical_lj_params() {
        // AMBER ff14SB typical sigma values (Angstroms)
        let sigmas = [
            1.9080, // H
            1.8240, // HO
            3.3997, // C
            3.2500, // N
            3.0665, // O
            3.5636, // S
        ];

        // AMBER ff14SB typical epsilon values (kcal/mol)
        let epsilons = [
            0.0157, // H
            0.0000, // HO
            0.1094, // C
            0.1700, // N
            0.2100, // O
            0.2500, // S
        ];

        for &sigma in &sigmas {
            let bits = f32_to_f16_bits(sigma);
            let back = f16_bits_to_f32(bits);
            let rel_error = ((back - sigma) / sigma).abs();
            assert!(
                rel_error < 0.001,
                "Sigma {} has FP16 error {:.4}%",
                sigma,
                rel_error * 100.0
            );
        }

        for &eps in &epsilons {
            if eps > 0.0 {
                let bits = f32_to_f16_bits(eps);
                let back = f16_bits_to_f32(bits);
                let rel_error = ((back - eps) / eps).abs();
                assert!(
                    rel_error < 0.01, // 1% tolerance for small values
                    "Epsilon {} has FP16 error {:.4}%",
                    eps,
                    rel_error * 100.0
                );
            }
        }
    }

    #[test]
    fn test_f32_to_f16_special_values() {
        // Infinity
        let inf_bits = f32_to_f16_bits(f32::INFINITY);
        let inf_back = f16_bits_to_f32(inf_bits);
        assert!(inf_back.is_infinite() && inf_back > 0.0);

        let neg_inf_bits = f32_to_f16_bits(f32::NEG_INFINITY);
        let neg_inf_back = f16_bits_to_f32(neg_inf_bits);
        assert!(neg_inf_back.is_infinite() && neg_inf_back < 0.0);

        // NaN
        let nan_bits = f32_to_f16_bits(f32::NAN);
        let nan_back = f16_bits_to_f32(nan_bits);
        assert!(nan_back.is_nan());

        // Negative zero (should preserve sign in FP16)
        let neg_zero = -0.0f32;
        let nz_bits = f32_to_f16_bits(neg_zero);
        let nz_back = f16_bits_to_f32(nz_bits);
        assert_eq!(nz_back, 0.0); // Value is 0
    }

    #[test]
    fn test_f32_to_f16_overflow() {
        // FP16 max is ~65504, values beyond should become infinity
        let big = 100000.0f32;
        let bits = f32_to_f16_bits(big);
        let back = f16_bits_to_f32(bits);
        assert!(
            back.is_infinite(),
            "Large values should overflow to infinity"
        );
    }

    #[test]
    fn test_f32_to_f16_underflow() {
        // FP16 min positive is ~6e-8, very small values should become 0
        let tiny = 1e-10f32;
        let bits = f32_to_f16_bits(tiny);
        let back = f16_bits_to_f32(bits);
        assert_eq!(back, 0.0, "Very small values should underflow to zero");
    }

    #[test]
    fn test_mixed_precision_config_default() {
        let config = MixedPrecisionConfig::default();
        assert!(config.fp16_lj_params);
        assert!(!config.fp16_pme_grid);
        assert!(!config.half2_lj);
        assert!(config.is_enabled());
    }

    #[test]
    fn test_mixed_precision_config_full_precision() {
        let config = MixedPrecisionConfig::full_precision();
        assert!(!config.fp16_lj_params);
        assert!(!config.fp16_pme_grid);
        assert!(!config.half2_lj);
        assert!(!config.is_enabled());
    }

    #[test]
    fn test_mixed_precision_config_max_performance() {
        let config = MixedPrecisionConfig::max_performance();
        assert!(config.fp16_lj_params);
        assert!(config.fp16_pme_grid);
        assert!(config.half2_lj);
        assert!(config.is_enabled());
    }
}
