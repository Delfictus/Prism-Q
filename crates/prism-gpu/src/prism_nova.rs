//! PRISM-NOVA: Neural-Optimized Variational Adaptive Dynamics
//!
//! A unified physics engine combining:
//! - Neural Hamiltonian Monte Carlo (NHMC) for efficient sampling
//! - Topological Data Analysis (TDA) for collective variables
//! - Active Inference for goal-directed exploration
//! - Reservoir Computing + RLS for online learning
//!
//! All components execute in a single fused GPU kernel with zero CPU round-trips.
//!
//! ## Key Innovations
//!
//! 1. **Replaces Langevin with HMC**: Coherent momentum-based exploration
//!    is more efficient at sampling rare conformational events (like cryptic pocket opening)
//!
//! 2. **TDA-Based Collective Variables**: Cryptic pockets are topological events -
//!    a void (Betti-2) appearing in the protein structure. TDA directly measures this.
//!
//! 3. **Active Inference Goal-Direction**: The system has a "goal" of finding
//!    druggable conformations. It actively seeks states with high pocket signatures.
//!
//! 4. **Fused Kernel**: Physics → TDA → AI → Reservoir → RLS all in one kernel launch.
//!    Eliminates CPU-GPU round-trips for ~5x speedup.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use prism_gpu::prism_nova::PrismNova;
//!
//! let nova = PrismNova::new(device, config)?;
//!
//! // Run simulation steps
//! for _ in 0..1000 {
//!     let result = nova.step(&positions, &momenta)?;
//!     if result.accepted {
//!         // New conformation accepted
//!     }
//! }
//! ```
//!
//! ## Performance Targets
//!
//! - Steps/sec: ~800K (vs 169K with Langevin + CPU RLS)
//! - GPU utilization: 95%+ (vs 60% average before)
//! - Rare event sampling: Polynomial time (vs exponential with Langevin)

use anyhow::{Context, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceSlice, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// Maximum atoms supported by NOVA kernel
/// NOTE: Reduced from 4096 due to shared memory limits on sm_86
pub const MAX_ATOMS: usize = 512;

/// Maximum target residues for pocket analysis
/// NOTE: Reduced from 64 to 32 to fit TDA shared memory (dist_matrix = 32*32*4 = 4KB)
pub const MAX_TARGET_RESIDUES: usize = 32;

/// Reservoir size (must match kernel constant)
pub const RESERVOIR_SIZE: usize = 1024;

/// Number of RLS output heads
pub const NUM_OUTPUTS: usize = 20;

/// Feature dimension from TDA + Active Inference
pub const FEATURE_DIM: usize = 40;

/// Configuration for PRISM-NOVA simulation
#[repr(C)]
#[derive(Debug, Clone)]
pub struct NovaConfig {
    /// Timestep in picoseconds
    pub dt: f32,
    /// Temperature in Kelvin
    pub temperature: f32,
    /// Strength of Active Inference goal-directed bias
    pub goal_strength: f32,
    /// RLS forgetting factor (λ)
    pub lambda: f32,

    /// Number of atoms in the system
    pub n_atoms: i32,
    /// Number of residues
    pub n_residues: i32,
    /// Number of target residues for pocket analysis
    pub n_target_residues: i32,

    /// Number of leapfrog steps per HMC iteration
    pub leapfrog_steps: i32,
    /// Mass scaling factor
    pub mass_scale: f32,

    /// Neural network hidden dimension
    pub nn_hidden_dim: i32,
    /// Number of neural network layers
    pub nn_num_layers: i32,

    /// Target residue indices (pocket region)
    pub target_residues: [i32; MAX_TARGET_RESIDUES],

    /// Random seed
    pub seed: u64,
}

impl Default for NovaConfig {
    fn default() -> Self {
        Self {
            dt: 0.002,          // 2 fs timestep
            temperature: 310.0, // 37°C
            goal_strength: 0.1, // Moderate AI bias
            lambda: 0.99,       // RLS forgetting factor

            n_atoms: 0,
            n_residues: 0,
            n_target_residues: 0,

            leapfrog_steps: 3, // Very short trajectory for high acceptance
            mass_scale: 1.0,

            nn_hidden_dim: 64,
            nn_num_layers: 2,

            target_residues: [0; MAX_TARGET_RESIDUES],

            seed: 42,
        }
    }
}

/// Results from a single NOVA step
#[derive(Debug, Clone)]
pub struct NovaStepResult {
    /// Whether the HMC proposal was accepted
    pub accepted: bool,
    /// Reward for this step (pocket progress)
    pub reward: f32,
    /// Topological + Active Inference features
    pub features: Vec<f32>,
    /// Betti numbers (connected components, cycles, voids)
    pub betti: [f32; 3],
    /// Pocket signature (higher = more open)
    pub pocket_signature: f32,
    /// Expected free energy
    pub efe: f32,
    /// Goal prior (probability of druggable state)
    pub goal_prior: f32,
}

/// PRISM-NOVA: Unified physics engine with goal-directed sampling
pub struct PrismNova {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,

    // CUDA kernels
    step_kernel: CudaFunction,
    init_momenta_kernel: CudaFunction,
    init_rls_kernel: CudaFunction,

    // Device memory - Simulation state
    d_positions: CudaSlice<f32>,     // [n_atoms * 3]
    d_momenta: CudaSlice<f32>,       // [n_atoms * 3]
    d_positions_old: CudaSlice<f32>, // [n_atoms * 3] for rejection

    // Device memory - Force field
    d_masses: CudaSlice<f32>,               // [n_atoms]
    d_charges: CudaSlice<f32>,              // [n_atoms]
    d_lj_params: CudaSlice<f32>,            // [n_atoms * 2] (epsilon, sigma)
    d_bond_list: CudaSlice<i32>,            // [n_bonds * 2]
    d_bond_params: CudaSlice<f32>,          // [n_bonds * 2] (r0, k)
    d_angle_list: CudaSlice<i32>,           // [n_angles * 3] (i, j, k)
    d_angle_params: CudaSlice<f32>,         // [n_angles * 2] (theta0, k)
    d_dihedral_list: CudaSlice<i32>,        // [n_dihedrals * 4] (i, j, k, l)
    d_dihedral_params: CudaSlice<f32>,      // Flattened: [k, n, phase, paths, ...] per term
    d_dihedral_term_counts: CudaSlice<i32>, // [n_dihedrals] - terms per dihedral
    d_exclusion_list: CudaSlice<i32>,       // [n_exclusions * 2] - 1-2 and 1-3 pairs to skip
    d_pair_14_list: CudaSlice<i32>,         // [n_pairs_14 * 2] - 1-4 pairs for scaled interactions
    d_atom_types: CudaSlice<i32>,           // [n_atoms]
    d_residue_atoms: CudaSlice<i32>,        // [n_residues] - representative atom per residue
    d_target_residues: CudaSlice<i32>,      // [n_target_residues] - target residues for TDA

    // Device memory - Neural network
    d_nn_weights: CudaSlice<f32>,

    // Device memory - Reservoir
    d_reservoir_activations: CudaSlice<f32>, // [RESERVOIR_SIZE]
    d_reservoir_filtered: CudaSlice<f32>,    // [RESERVOIR_SIZE]
    d_reservoir_membrane: CudaSlice<f32>,    // [RESERVOIR_SIZE]
    d_reservoir_adaptation: CudaSlice<f32>,  // [RESERVOIR_SIZE]
    d_reservoir_weights: CudaSlice<f32>,     // [input_dim * RESERVOIR_SIZE + RESERVOIR_SIZE^2]

    // Device memory - RLS
    d_rls_weights: CudaSlice<f32>,    // [NUM_OUTPUTS * RESERVOIR_SIZE]
    d_rls_P_matrices: CudaSlice<f32>, // [NUM_OUTPUTS * RESERVOIR_SIZE * RESERVOIR_SIZE]

    // Device memory - Outputs
    d_features: CudaSlice<f32>, // [FEATURE_DIM]
    d_reward: CudaSlice<f32>,   // [1]
    d_accepted: CudaSlice<i32>, // [1]

    // Configuration
    config: NovaConfig,
    n_bonds: usize,
    n_angles: usize,
    n_dihedrals: usize,
    n_exclusions: usize,
    n_pairs_14: usize,
}

impl PrismNova {
    /// Create a new PRISM-NOVA simulation engine
    ///
    /// # Arguments
    /// * `context` - CUDA context (from GpuContext)
    /// * `config` - Simulation configuration
    ///
    /// # Returns
    /// Initialized NOVA engine with pre-allocated GPU memory
    pub fn new(context: Arc<CudaContext>, config: NovaConfig) -> Result<Self> {
        log::info!(
            "🚀 Initializing PRISM-NOVA: {} atoms, {} target residues",
            config.n_atoms,
            config.n_target_residues
        );

        // Load PTX module
        let ptx_path = "target/ptx/prism_nova.ptx";
        let ptx = Ptx::from_file(ptx_path);
        let module = context
            .load_module(ptx)
            .with_context(|| format!("Failed to load PRISM-NOVA PTX from {}", ptx_path))?;

        log::info!("📦 Loaded PRISM-NOVA PTX module");

        // Get kernel functions
        let step_kernel = module
            .load_function("prism_nova_step")
            .context("Failed to load prism_nova_step kernel")?;
        let init_momenta_kernel = module
            .load_function("initialize_momenta")
            .context("Failed to load initialize_momenta kernel")?;
        let init_rls_kernel = module
            .load_function("initialize_rls_P")
            .context("Failed to load initialize_rls_P kernel")?;

        log::info!("⚡ Loaded 3 CUDA kernels");

        // Get stream
        let stream = context.default_stream();

        // Allocate device memory
        let n_atoms = config.n_atoms as usize;
        let n_residues = config.n_residues as usize;

        // Simulation state
        let d_positions = stream
            .alloc_zeros::<f32>(n_atoms * 3)
            .context("Failed to allocate positions")?;
        let d_momenta = stream
            .alloc_zeros::<f32>(n_atoms * 3)
            .context("Failed to allocate momenta")?;
        let d_positions_old = stream
            .alloc_zeros::<f32>(n_atoms * 3)
            .context("Failed to allocate positions_old")?;

        // Force field
        let d_masses = stream
            .alloc_zeros::<f32>(n_atoms)
            .context("Failed to allocate masses")?;
        let d_charges = stream
            .alloc_zeros::<f32>(n_atoms)
            .context("Failed to allocate charges")?;
        let d_lj_params = stream
            .alloc_zeros::<f32>(n_atoms * 2)
            .context("Failed to allocate LJ params")?;
        let d_bond_list = stream
            .alloc_zeros::<i32>(1)
            .context("Failed to allocate bond list")?;
        let d_bond_params = stream
            .alloc_zeros::<f32>(1)
            .context("Failed to allocate bond params")?;
        let d_angle_list = stream
            .alloc_zeros::<i32>(1)
            .context("Failed to allocate angle list")?;
        let d_angle_params = stream
            .alloc_zeros::<f32>(1)
            .context("Failed to allocate angle params")?;
        let d_dihedral_list = stream
            .alloc_zeros::<i32>(1)
            .context("Failed to allocate dihedral list")?;
        let d_dihedral_params = stream
            .alloc_zeros::<f32>(1)
            .context("Failed to allocate dihedral params")?;
        let d_dihedral_term_counts = stream
            .alloc_zeros::<i32>(1)
            .context("Failed to allocate dihedral term counts")?;
        let d_exclusion_list = stream
            .alloc_zeros::<i32>(1)
            .context("Failed to allocate exclusion list")?;
        let d_pair_14_list = stream
            .alloc_zeros::<i32>(1)
            .context("Failed to allocate 1-4 pair list")?;
        let d_atom_types = stream
            .alloc_zeros::<i32>(n_atoms)
            .context("Failed to allocate atom types")?;
        let d_residue_atoms = stream
            .alloc_zeros::<i32>(n_residues.max(1))
            .context("Failed to allocate residue atoms")?;
        let d_target_residues = stream
            .alloc_zeros::<i32>(MAX_TARGET_RESIDUES)
            .context("Failed to allocate target residues")?;

        // Neural network (placeholder - small initial allocation)
        let nn_size = config.nn_hidden_dim as usize * 32 + config.nn_hidden_dim as usize * 3 + 3;
        let d_nn_weights = stream
            .alloc_zeros::<f32>(nn_size)
            .context("Failed to allocate NN weights")?;

        // Reservoir
        let d_reservoir_activations = stream
            .alloc_zeros::<f32>(RESERVOIR_SIZE)
            .context("Failed to allocate reservoir activations")?;
        let d_reservoir_filtered = stream
            .alloc_zeros::<f32>(RESERVOIR_SIZE)
            .context("Failed to allocate reservoir filtered")?;
        let d_reservoir_membrane = stream
            .alloc_zeros::<f32>(RESERVOIR_SIZE)
            .context("Failed to allocate reservoir membrane")?;
        let d_reservoir_adaptation = stream
            .alloc_zeros::<f32>(RESERVOIR_SIZE)
            .context("Failed to allocate reservoir adaptation")?;

        let reservoir_weights_size = FEATURE_DIM * RESERVOIR_SIZE + RESERVOIR_SIZE * RESERVOIR_SIZE;
        let d_reservoir_weights = stream
            .alloc_zeros::<f32>(reservoir_weights_size)
            .context("Failed to allocate reservoir weights")?;

        // RLS
        let d_rls_weights = stream
            .alloc_zeros::<f32>(NUM_OUTPUTS * RESERVOIR_SIZE)
            .context("Failed to allocate RLS weights")?;
        let d_rls_P_matrices = stream
            .alloc_zeros::<f32>(NUM_OUTPUTS * RESERVOIR_SIZE * RESERVOIR_SIZE)
            .context("Failed to allocate RLS P matrices")?;

        // Outputs
        let d_features = stream
            .alloc_zeros::<f32>(FEATURE_DIM)
            .context("Failed to allocate features")?;
        let d_reward = stream
            .alloc_zeros::<f32>(1)
            .context("Failed to allocate reward")?;
        let d_accepted = stream
            .alloc_zeros::<i32>(1)
            .context("Failed to allocate accepted flag")?;

        // Calculate total GPU memory
        let total_bytes = (n_atoms * 3 * 3  // positions, momenta, positions_old
            + n_atoms * 4  // masses, charges, lj_params(2)
            + n_atoms + n_residues.max(1)  // atom_types, residue_atoms
            + nn_size
            + RESERVOIR_SIZE * 4  // reservoir state
            + reservoir_weights_size
            + NUM_OUTPUTS * RESERVOIR_SIZE
            + NUM_OUTPUTS * RESERVOIR_SIZE * RESERVOIR_SIZE
            + FEATURE_DIM + 1 + 1)
            * 4; // outputs

        log::info!(
            "💾 Allocated {:.1} MB GPU memory for NOVA",
            total_bytes as f64 / 1024.0 / 1024.0
        );

        Ok(Self {
            context,
            stream,
            module,
            step_kernel,
            init_momenta_kernel,
            init_rls_kernel,
            d_positions,
            d_momenta,
            d_positions_old,
            d_masses,
            d_charges,
            d_lj_params,
            d_bond_list,
            d_bond_params,
            d_angle_list,
            d_angle_params,
            d_dihedral_list,
            d_dihedral_params,
            d_dihedral_term_counts,
            d_exclusion_list,
            d_pair_14_list,
            d_atom_types,
            d_residue_atoms,
            d_target_residues,
            d_nn_weights,
            d_reservoir_activations,
            d_reservoir_filtered,
            d_reservoir_membrane,
            d_reservoir_adaptation,
            d_reservoir_weights,
            d_rls_weights,
            d_rls_P_matrices,
            d_features,
            d_reward,
            d_accepted,
            config,
            n_bonds: 0,
            n_angles: 0,
            n_dihedrals: 0,
            n_exclusions: 0,
            n_pairs_14: 0,
        })
    }

    /// Upload molecular system data to GPU
    ///
    /// # Arguments
    /// * `positions` - Atomic positions [n_atoms * 3] as (x, y, z, x, y, z, ...)
    /// * `masses` - Atomic masses [n_atoms]
    /// * `charges` - Partial charges [n_atoms]
    /// * `lj_params` - Lennard-Jones parameters [n_atoms * 2] as (epsilon, sigma, ...)
    /// * `atom_types` - Atom type indices [n_atoms]
    /// * `residue_atoms` - Representative atom index per residue [n_residues]
    pub fn upload_system(
        &mut self,
        positions: &[f32],
        masses: &[f32],
        charges: &[f32],
        lj_params: &[f32],
        atom_types: &[i32],
        residue_atoms: &[i32],
    ) -> Result<()> {
        log::info!("📤 Uploading molecular system to GPU");

        self.stream
            .memcpy_htod(positions, &mut self.d_positions)
            .context("Failed to upload positions")?;
        self.stream
            .memcpy_htod(masses, &mut self.d_masses)
            .context("Failed to upload masses")?;
        self.stream
            .memcpy_htod(charges, &mut self.d_charges)
            .context("Failed to upload charges")?;
        self.stream
            .memcpy_htod(lj_params, &mut self.d_lj_params)
            .context("Failed to upload LJ params")?;
        self.stream
            .memcpy_htod(atom_types, &mut self.d_atom_types)
            .context("Failed to upload atom types")?;
        self.stream
            .memcpy_htod(residue_atoms, &mut self.d_residue_atoms)
            .context("Failed to upload residue atoms")?;

        // Upload target residues for TDA from config
        let n_targets = self.config.n_target_residues as usize;
        if n_targets > 0 {
            let target_residues: Vec<i32> = self.config.target_residues[..n_targets].to_vec();
            self.stream
                .memcpy_htod(&target_residues, &mut self.d_target_residues)
                .context("Failed to upload target residues")?;
            log::info!("🎯 Uploaded {} target residues for TDA", n_targets);
        }

        log::info!("✅ System uploaded successfully");
        Ok(())
    }

    /// Upload bond topology
    ///
    /// # Arguments
    /// * `bond_list` - Pairs of bonded atom indices [n_bonds * 2]
    /// * `bond_params` - Bond parameters [n_bonds * 2] as (r0, k, ...)
    pub fn upload_bonds(&mut self, bond_list: &[i32], bond_params: &[f32]) -> Result<()> {
        self.n_bonds = bond_list.len() / 2;

        // Reallocate if needed
        if bond_list.len() > self.d_bond_list.len() {
            self.d_bond_list = self
                .stream
                .alloc_zeros::<i32>(bond_list.len())
                .context("Failed to reallocate bond list")?;
        }
        if bond_params.len() > self.d_bond_params.len() {
            self.d_bond_params = self
                .stream
                .alloc_zeros::<f32>(bond_params.len())
                .context("Failed to reallocate bond params")?;
        }

        self.stream
            .memcpy_htod(bond_list, &mut self.d_bond_list)
            .context("Failed to upload bond list")?;
        self.stream
            .memcpy_htod(bond_params, &mut self.d_bond_params)
            .context("Failed to upload bond params")?;

        log::info!("🔗 Uploaded {} bonds", self.n_bonds);
        Ok(())
    }

    /// Upload angle topology
    ///
    /// # Arguments
    /// * `angle_list` - Triplets of angle atom indices [n_angles * 3] (i, j, k)
    /// * `angle_params` - Angle parameters [n_angles * 2] as (theta0, k)
    pub fn upload_angles(&mut self, angle_list: &[i32], angle_params: &[f32]) -> Result<()> {
        self.n_angles = angle_list.len() / 3;

        // Reallocate if needed
        if angle_list.len() > self.d_angle_list.len() {
            self.d_angle_list = self
                .stream
                .alloc_zeros::<i32>(angle_list.len())
                .context("Failed to reallocate angle list")?;
        }
        if angle_params.len() > self.d_angle_params.len() {
            self.d_angle_params = self
                .stream
                .alloc_zeros::<f32>(angle_params.len())
                .context("Failed to reallocate angle params")?;
        }

        self.stream
            .memcpy_htod(angle_list, &mut self.d_angle_list)
            .context("Failed to upload angle list")?;
        self.stream
            .memcpy_htod(angle_params, &mut self.d_angle_params)
            .context("Failed to upload angle params")?;

        log::info!("📐 Uploaded {} angles", self.n_angles);
        Ok(())
    }

    /// Upload dihedral topology
    ///
    /// # Arguments
    /// * `dihedral_list` - Quartets of dihedral atom indices [n_dihedrals * 4] (i, j, k, l)
    /// * `dihedral_params` - Dihedral parameters, flattened [k, n, phase, paths, ...] per term
    /// * `term_counts` - Number of terms per dihedral [n_dihedrals]
    pub fn upload_dihedrals(
        &mut self,
        dihedral_list: &[i32],
        dihedral_params: &[f32],
        term_counts: &[i32],
    ) -> Result<()> {
        self.n_dihedrals = dihedral_list.len() / 4;

        // Reallocate if needed
        if dihedral_list.len() > self.d_dihedral_list.len() {
            self.d_dihedral_list = self
                .stream
                .alloc_zeros::<i32>(dihedral_list.len())
                .context("Failed to reallocate dihedral list")?;
        }
        if dihedral_params.len() > self.d_dihedral_params.len() {
            self.d_dihedral_params = self
                .stream
                .alloc_zeros::<f32>(dihedral_params.len())
                .context("Failed to reallocate dihedral params")?;
        }
        if term_counts.len() > self.d_dihedral_term_counts.len() {
            self.d_dihedral_term_counts = self
                .stream
                .alloc_zeros::<i32>(term_counts.len())
                .context("Failed to reallocate dihedral term counts")?;
        }

        self.stream
            .memcpy_htod(dihedral_list, &mut self.d_dihedral_list)
            .context("Failed to upload dihedral list")?;
        self.stream
            .memcpy_htod(dihedral_params, &mut self.d_dihedral_params)
            .context("Failed to upload dihedral params")?;
        self.stream
            .memcpy_htod(term_counts, &mut self.d_dihedral_term_counts)
            .context("Failed to upload dihedral term counts")?;

        log::info!("🔄 Uploaded {} dihedrals", self.n_dihedrals);
        Ok(())
    }

    /// Upload exclusion list (1-2 and 1-3 bonded pairs to skip in non-bonded calculation)
    ///
    /// # Arguments
    /// * `exclusion_list` - Flattened pairs [atom_i, atom_j, ...] for all exclusions
    pub fn upload_exclusions(&mut self, exclusion_list: &[i32]) -> Result<()> {
        self.n_exclusions = exclusion_list.len() / 2;

        if exclusion_list.is_empty() {
            log::info!("⊘ No exclusions to upload");
            return Ok(());
        }

        // Reallocate if needed
        if exclusion_list.len() > self.d_exclusion_list.len() {
            self.d_exclusion_list = self
                .stream
                .alloc_zeros::<i32>(exclusion_list.len())
                .context("Failed to reallocate exclusion list")?;
        }

        self.stream
            .memcpy_htod(exclusion_list, &mut self.d_exclusion_list)
            .context("Failed to upload exclusion list")?;

        log::info!(
            "⊘ Uploaded {} exclusions (1-2/1-3 pairs)",
            self.n_exclusions
        );
        Ok(())
    }

    /// Upload 1-4 pair list (atoms separated by 3 bonds, need scaled non-bonded interactions)
    ///
    /// # Arguments
    /// * `pair_14_list` - Flattened pairs [atom_i, atom_j, ...] for all 1-4 pairs
    pub fn upload_pairs_14(&mut self, pair_14_list: &[i32]) -> Result<()> {
        self.n_pairs_14 = pair_14_list.len() / 2;

        if pair_14_list.is_empty() {
            log::info!("⚖️ No 1-4 pairs to upload");
            return Ok(());
        }

        // Reallocate if needed
        if pair_14_list.len() > self.d_pair_14_list.len() {
            self.d_pair_14_list = self
                .stream
                .alloc_zeros::<i32>(pair_14_list.len())
                .context("Failed to reallocate 1-4 pair list")?;
        }

        self.stream
            .memcpy_htod(pair_14_list, &mut self.d_pair_14_list)
            .context("Failed to upload 1-4 pair list")?;

        log::info!(
            "⚖️ Uploaded {} 1-4 pairs (scaled non-bonded)",
            self.n_pairs_14
        );
        Ok(())
    }

    /// Upload reservoir weights (from pre-trained or random initialization)
    ///
    /// # Arguments
    /// * `weights` - Reservoir connection weights [input_dim * RESERVOIR_SIZE + RESERVOIR_SIZE^2]
    pub fn upload_reservoir_weights(&mut self, weights: &[f32]) -> Result<()> {
        self.stream
            .memcpy_htod(weights, &mut self.d_reservoir_weights)
            .context("Failed to upload reservoir weights")?;
        log::info!("🧠 Uploaded reservoir weights ({} floats)", weights.len());
        Ok(())
    }

    /// Initialize momenta from Maxwell-Boltzmann distribution
    pub fn initialize_momenta(&mut self) -> Result<()> {
        let n_atoms = self.config.n_atoms as usize;
        let blocks = (n_atoms + 255) / 256;

        let launch_config = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            let mut builder = self.stream.launch_builder(&self.init_momenta_kernel);
            builder.arg(&mut self.d_momenta);
            builder.arg(&self.d_masses);
            builder.arg(&self.config.temperature);
            builder.arg(&self.config.seed);
            builder.arg(&self.config.n_atoms);
            builder
                .launch(launch_config)
                .context("Failed to launch init_momenta kernel")?;
        }

        log::info!("🎲 Initialized momenta at T={} K", self.config.temperature);
        Ok(())
    }

    /// Initialize RLS precision matrices to identity * initial_precision
    pub fn initialize_rls(&mut self, initial_precision: f32) -> Result<()> {
        let launch_config = LaunchConfig {
            grid_dim: (NUM_OUTPUTS as u32, 1, 1),
            block_dim: (RESERVOIR_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let num_outputs_i32 = NUM_OUTPUTS as i32;

        unsafe {
            let mut builder = self.stream.launch_builder(&self.init_rls_kernel);
            builder.arg(&mut self.d_rls_P_matrices);
            builder.arg(&initial_precision);
            builder.arg(&num_outputs_i32);
            builder
                .launch(launch_config)
                .context("Failed to launch init_rls kernel")?;
        }

        log::info!(
            "📊 Initialized RLS P matrices (precision={})",
            initial_precision
        );
        Ok(())
    }

    /// Run a single NOVA simulation step
    ///
    /// This executes the full fused kernel:
    /// 1. Neural HMC leapfrog integration
    /// 2. Topological CV computation
    /// 3. Active Inference guidance
    /// 4. Metropolis acceptance
    /// 5. Reservoir + RLS update
    ///
    /// # Returns
    /// Step results including acceptance, reward, and features
    pub fn step(&mut self) -> Result<NovaStepResult> {
        let launch_config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0, // Kernel uses static shared memory
        };

        // Increment seed for stochastic elements
        self.config.seed = self.config.seed.wrapping_add(1);

        let n_bonds_i32 = self.n_bonds as i32;

        unsafe {
            let mut builder = self.stream.launch_builder(&self.step_kernel);

            // Positions and momenta
            builder.arg(&mut self.d_positions);
            builder.arg(&mut self.d_momenta);
            builder.arg(&mut self.d_positions_old);

            // Force field
            builder.arg(&self.d_masses);
            builder.arg(&self.d_charges);
            builder.arg(&self.d_lj_params);
            builder.arg(&self.d_bond_list);
            builder.arg(&self.d_bond_params);
            builder.arg(&n_bonds_i32);
            builder.arg(&self.d_angle_list);
            builder.arg(&self.d_angle_params);
            let n_angles_i32 = self.n_angles as i32;
            builder.arg(&n_angles_i32);
            builder.arg(&self.d_dihedral_list);
            builder.arg(&self.d_dihedral_params);
            builder.arg(&self.d_dihedral_term_counts);
            let n_dihedrals_i32 = self.n_dihedrals as i32;
            builder.arg(&n_dihedrals_i32);

            // Exclusions (1-2/1-3 bonded pairs) and 1-4 scaled pairs
            builder.arg(&self.d_exclusion_list);
            let n_exclusions_i32 = self.n_exclusions as i32;
            builder.arg(&n_exclusions_i32);
            builder.arg(&self.d_pair_14_list);
            let n_pairs_14_i32 = self.n_pairs_14 as i32;
            builder.arg(&n_pairs_14_i32);

            // Neural network
            builder.arg(&self.d_nn_weights);

            // Atom metadata
            builder.arg(&self.d_atom_types);
            builder.arg(&self.d_residue_atoms);

            // Reservoir
            builder.arg(&mut self.d_reservoir_activations);
            builder.arg(&mut self.d_reservoir_filtered);
            builder.arg(&mut self.d_reservoir_membrane);
            builder.arg(&mut self.d_reservoir_adaptation);
            builder.arg(&self.d_reservoir_weights);

            // RLS
            builder.arg(&mut self.d_rls_weights);
            builder.arg(&mut self.d_rls_P_matrices);

            // Outputs
            builder.arg(&mut self.d_features);
            builder.arg(&mut self.d_reward);
            builder.arg(&mut self.d_accepted);

            // Target residues for TDA
            builder.arg(&self.d_target_residues);
            builder.arg(&self.config.n_target_residues);

            // Config struct - need to upload to GPU
            // For now, pass key config values individually
            builder.arg(&self.config.dt);
            builder.arg(&self.config.temperature);
            builder.arg(&self.config.goal_strength);
            builder.arg(&self.config.lambda);
            builder.arg(&self.config.n_atoms);
            builder.arg(&self.config.n_residues);
            builder.arg(&self.config.leapfrog_steps);
            builder.arg(&self.config.seed);

            builder
                .launch(launch_config)
                .context("Failed to launch step kernel")?;
        }

        // Download results using new API (clone_dtoh returns Vec directly)
        let features: Vec<f32> = self
            .stream
            .clone_dtoh(&self.d_features)
            .context("Failed to download features")?;
        let reward: Vec<f32> = self
            .stream
            .clone_dtoh(&self.d_reward)
            .context("Failed to download reward")?;
        let accepted: Vec<i32> = self
            .stream
            .clone_dtoh(&self.d_accepted)
            .context("Failed to download accepted flag")?;

        Ok(NovaStepResult {
            accepted: accepted.get(0).copied().unwrap_or(0) != 0,
            reward: reward.get(0).copied().unwrap_or(0.0),
            features: features.clone(),
            betti: [
                features.get(0).copied().unwrap_or(0.0),
                features.get(1).copied().unwrap_or(0.0),
                features.get(2).copied().unwrap_or(0.0),
            ],
            pocket_signature: features.get(4).copied().unwrap_or(0.0),
            efe: features.get(8).copied().unwrap_or(0.0),
            goal_prior: features.get(10).copied().unwrap_or(0.0),
        })
    }

    /// Download current positions from GPU
    pub fn download_positions(&self) -> Result<Vec<f32>> {
        let positions: Vec<f32> = self
            .stream
            .clone_dtoh(&self.d_positions)
            .context("Failed to download positions")?;
        Ok(positions)
    }

    /// Get current configuration
    pub fn config(&self) -> &NovaConfig {
        &self.config
    }

    /// Update configuration (some fields can be changed at runtime)
    pub fn set_temperature(&mut self, temperature: f32) {
        self.config.temperature = temperature;
    }

    pub fn set_goal_strength(&mut self, strength: f32) {
        self.config.goal_strength = strength;
    }

    pub fn set_dt(&mut self, dt: f32) {
        self.config.dt = dt;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = NovaConfig::default();
        assert_eq!(config.temperature, 310.0);
        // NovaConfig::default() intentionally uses 3 (short HMC trajectory for high acceptance;
        // see the `leapfrog_steps: 3` rationale in Default impl). Test updated from stale 10.
        assert_eq!(config.leapfrog_steps, 3);
        assert!(config.lambda > 0.0 && config.lambda < 1.0);
    }

    #[test]
    fn test_step_result_parse() {
        let mut features = vec![0.0f32; FEATURE_DIM];
        features[0] = 1.0; // betti_0
        features[1] = 2.0; // betti_1
        features[2] = 3.0; // betti_2 (voids!)
        features[4] = 0.75; // pocket_signature
        features[8] = -1.5; // efe
        features[10] = 0.8; // goal_prior

        let result = NovaStepResult {
            accepted: true,
            reward: 0.5,
            features: features.clone(),
            betti: [features[0], features[1], features[2]],
            pocket_signature: features[4],
            efe: features[8],
            goal_prior: features[10],
        };

        assert!(result.accepted);
        assert_eq!(result.betti[2], 3.0); // 3 voids detected
        assert_eq!(result.pocket_signature, 0.75);
    }
}
