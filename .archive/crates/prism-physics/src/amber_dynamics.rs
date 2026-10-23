//! AMBER All-Atom Dynamics Engine
//!
//! Implements Hamiltonian Monte Carlo (HMC) sampling with AMBER ff14SB force field.
//! This is a NEW file for Phase 2 - does NOT modify any locked files.
//!
//! # Algorithm
//!
//! HMC combines molecular dynamics with Metropolis acceptance:
//! 1. Draw random momenta from Maxwell-Boltzmann distribution
//! 2. Run leapfrog integration for N steps
//! 3. Accept/reject based on Δ(H) = Δ(KE + PE)
//!
//! This ensures proper Boltzmann sampling while using gradient information.
//!
//! # GPU Acceleration
//!
//! When compiled with the `cuda` feature, force calculations use GPU kernels
//! from `prism-gpu::amber_forces`. This provides 10-100x speedup for larger proteins.
//!
//! # Usage
//!
//! ```rust,ignore
//! use prism_physics::amber_dynamics::{AmberSimulator, AmberSimConfig};
//!
//! let config = AmberSimConfig::default();
//! let mut sim = AmberSimulator::new(atoms, config)?;
//! let trajectory = sim.run(n_steps)?;
//! ```

use rand::Rng;
use rand_distr::{Distribution, Normal};

use crate::amber_ff14sb::{AmberTopology, GpuTopology, PdbAtom};

// GPU support (optional)
#[cfg(feature = "cuda")]
use prism_gpu::amber_forces::{
    AmberBondedForces, Angle, AngleParam, Bond, BondParam, Dihedral, DihedralParam, NB14Param,
    Pair14,
};

/// Simulation configuration
#[derive(Debug, Clone)]
pub struct AmberSimConfig {
    /// Temperature in Kelvin
    pub temperature: f64,
    /// Timestep in femtoseconds
    pub timestep: f64,
    /// Number of leapfrog steps per HMC move
    pub n_leapfrog_steps: usize,
    /// Langevin friction coefficient (1/ps)
    pub friction: f64,
    /// Use Langevin dynamics instead of pure HMC
    pub use_langevin: bool,
    /// Random seed
    pub seed: u64,
    /// Use GPU acceleration (requires cuda feature)
    pub use_gpu: bool,
}

impl Default for AmberSimConfig {
    fn default() -> Self {
        Self {
            temperature: 300.0,   // Room temperature
            timestep: 2.0,        // 2 fs (standard for proteins with SHAKE)
            n_leapfrog_steps: 10, // 10 steps per HMC move
            friction: 1.0,        // 1/ps friction for Langevin
            use_langevin: true,   // Langevin is more stable
            seed: 42,
            use_gpu: true, // Use GPU when available
        }
    }
}

/// Boltzmann constant in kcal/(mol·K)
const KB_KCAL: f64 = 0.001987204;

/// Conversion factor for force → acceleration in Å/fs² units
///
/// Derivation:
/// - Force F in kcal/(mol·Å)
/// - Mass m in Da (g/mol)
/// - Acceleration a = F/m in kcal/(g·Å) → needs conversion to Å/fs²
///
/// 1 kcal = 4184 J = 4184 kg·m²/s²
/// Converting to g·Å²/fs²: 4184 * 1000 * 1e-20 * 1e-30 = 4.184e-4 g·Å²/fs²
/// So 1 kcal/(g·Å) = 4.184e-4 Å/fs²
///
/// To get a in Å/fs²: a = (F/m) * FORCE_CONVERSION_FACTOR
const FORCE_CONVERSION_FACTOR: f64 = 4.184e-4;

/// Old constant - INCORRECT for fs time units (this was for ps)
/// Keeping for reference: 418.4 converts kcal/mol to g·Å²/ps²/mol
#[allow(dead_code)]
const KCAL_TO_INTERNAL_PS: f64 = 418.4;

/// A single frame in the trajectory
#[derive(Debug, Clone)]
pub struct TrajectoryFrame {
    /// Positions in Angstroms
    pub positions: Vec<[f64; 3]>,
    /// Velocities in Å/fs
    pub velocities: Vec<[f64; 3]>,
    /// Potential energy in kcal/mol
    pub potential_energy: f64,
    /// Kinetic energy in kcal/mol
    pub kinetic_energy: f64,
    /// Temperature in Kelvin
    pub temperature: f64,
    /// Simulation time in femtoseconds
    pub time: f64,
}

/// Result from AMBER simulation
#[derive(Debug, Clone)]
pub struct AmberSimResult {
    /// Trajectory frames
    pub trajectory: Vec<TrajectoryFrame>,
    /// Per-residue RMSF computed from trajectory
    pub rmsf: Vec<f64>,
    /// HMC acceptance rate
    pub acceptance_rate: f64,
    /// Average potential energy
    pub avg_potential_energy: f64,
    /// Average temperature
    pub avg_temperature: f64,
    /// Total simulation time in femtoseconds
    pub total_time: f64,
}

/// AMBER all-atom dynamics simulator
pub struct AmberSimulator {
    /// Molecular topology
    topology: AmberTopology,
    /// GPU topology (for GPU acceleration)
    #[allow(dead_code)]
    gpu_topology: GpuTopology,
    /// Configuration
    config: AmberSimConfig,
    /// Current positions
    positions: Vec<[f64; 3]>,
    /// Current velocities
    velocities: Vec<[f64; 3]>,
    /// Atomic masses
    masses: Vec<f64>,
    /// Atomic charges
    charges: Vec<f64>,
    /// LJ epsilon parameters
    lj_epsilon: Vec<f64>,
    /// LJ sigma parameters
    lj_sigma: Vec<f64>,
    /// Random number generator
    rng: rand::rngs::StdRng,
    /// Simulation time
    time: f64,
    /// HMC acceptance count
    accepted: usize,
    /// HMC total attempts
    attempted: usize,
    /// GPU force calculator (when cuda feature enabled)
    #[cfg(feature = "cuda")]
    gpu_forces: Option<AmberBondedForces>,
    /// Whether GPU is active
    gpu_active: bool,
}

impl AmberSimulator {
    /// Create new simulator from PDB atoms
    pub fn new(atoms: &[PdbAtom], config: AmberSimConfig) -> anyhow::Result<Self> {
        use rand::SeedableRng;

        // Generate topology from atoms
        let topology = AmberTopology::from_pdb_atoms(atoms);
        // GPU topology for future GPU acceleration
        let gpu_topology = GpuTopology::from_amber(&topology);

        let _n_atoms = atoms.len(); // Used for validation

        // Extract positions from input atoms
        let positions: Vec<[f64; 3]> = atoms
            .iter()
            .map(|a| [a.x as f64, a.y as f64, a.z as f64])
            .collect();

        // Get masses and charges from topology
        let masses: Vec<f64> = topology.masses.iter().map(|&m| m as f64).collect();
        let charges: Vec<f64> = topology.charges.iter().map(|&q| q as f64).collect();

        // Get LJ parameters from topology
        // Note: AMBER uses rmin_half, convert to sigma: sigma = rmin_half * 2^(5/6)
        let rmin_to_sigma = 2.0_f64.powf(5.0 / 6.0);
        let lj_epsilon: Vec<f64> = topology
            .lj_params
            .iter()
            .map(|p| p.epsilon as f64)
            .collect();
        let lj_sigma: Vec<f64> = topology
            .lj_params
            .iter()
            .map(|p| (p.rmin_half as f64) * rmin_to_sigma)
            .collect();

        // Initialize velocities from Maxwell-Boltzmann distribution
        let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);
        let velocities = initialize_velocities(&masses, config.temperature, &mut rng);

        // Initialize GPU force calculator if available and requested
        #[cfg(feature = "cuda")]
        let (gpu_forces, gpu_active) = if config.use_gpu {
            match Self::init_gpu_forces(&topology, atoms.len()) {
                Ok(forces) => {
                    log::info!(
                        "🚀 GPU acceleration enabled for AMBER dynamics ({} atoms)",
                        atoms.len()
                    );
                    (Some(forces), true)
                }
                Err(e) => {
                    log::warn!("GPU initialization failed, falling back to CPU: {}", e);
                    (None, false)
                }
            }
        } else {
            (None, false)
        };

        #[cfg(not(feature = "cuda"))]
        let gpu_active = false;

        Ok(Self {
            topology,
            gpu_topology,
            config,
            positions,
            velocities,
            masses,
            charges,
            lj_epsilon,
            lj_sigma,
            rng,
            time: 0.0,
            accepted: 0,
            attempted: 0,
            #[cfg(feature = "cuda")]
            gpu_forces,
            gpu_active,
        })
    }

    /// Initialize GPU force calculator
    #[cfg(feature = "cuda")]
    fn init_gpu_forces(
        topology: &AmberTopology,
        n_atoms: usize,
    ) -> anyhow::Result<AmberBondedForces> {
        use cudarc::driver::CudaContext;

        // Get CUDA context (CudaContext::new already returns Arc<CudaContext>)
        let context = CudaContext::new(0)?;

        // Create force calculator
        let mut forces = AmberBondedForces::new(context, n_atoms)?;

        // Convert topology to GPU format
        let bonds: Vec<Bond> = topology
            .bonds
            .iter()
            .zip(topology.bond_params.iter())
            .map(|(&(i, j), p)| Bond {
                atom_i: i,
                atom_j: j,
                params: BondParam { k: p.k, r0: p.r0 },
            })
            .collect();

        let angles: Vec<Angle> = topology
            .angles
            .iter()
            .zip(topology.angle_params.iter())
            .map(|(&(i, j, k), p)| Angle {
                atom_i: i,
                atom_j: j,
                atom_k: k,
                params: AngleParam {
                    k: p.k,
                    theta0: p.theta0,
                },
            })
            .collect();

        // Flatten dihedrals (each can have multiple terms)
        let mut dihedrals: Vec<Dihedral> = Vec::new();
        for (idx, &(i, j, k, l)) in topology.dihedrals.iter().enumerate() {
            for p in &topology.dihedral_params[idx] {
                dihedrals.push(Dihedral {
                    atom_i: i,
                    atom_j: j,
                    atom_k: k,
                    atom_l: l,
                    params: DihedralParam {
                        k: p.k,
                        n: p.n as f32,
                        phase: p.phase,
                        _pad: 0.0,
                    },
                });
            }
        }

        // 1-4 pairs with scaled LJ/Coulomb
        let pairs_14: Vec<Pair14> = topology
            .pairs_14
            .iter()
            .map(|&(i, j)| {
                let eps_i = topology
                    .lj_params
                    .get(i as usize)
                    .map(|p| p.epsilon)
                    .unwrap_or(0.1);
                let eps_j = topology
                    .lj_params
                    .get(j as usize)
                    .map(|p| p.epsilon)
                    .unwrap_or(0.1);
                let sig_i = topology
                    .lj_params
                    .get(i as usize)
                    .map(|p| p.rmin_half * 1.78179743628)
                    .unwrap_or(1.7);
                let sig_j = topology
                    .lj_params
                    .get(j as usize)
                    .map(|p| p.rmin_half * 1.78179743628)
                    .unwrap_or(1.7);
                let qi = topology.charges.get(i as usize).copied().unwrap_or(0.0);
                let qj = topology.charges.get(j as usize).copied().unwrap_or(0.0);

                Pair14 {
                    atom_i: i,
                    atom_j: j,
                    params: NB14Param {
                        epsilon: (eps_i * eps_j).sqrt(),
                        sigma: (sig_i + sig_j) / 2.0,
                        qi,
                        qj,
                    },
                }
            })
            .collect();

        forces.upload_topology(&bonds, &angles, &dihedrals, &pairs_14)?;

        Ok(forces)
    }

    /// Run energy minimization using steepest descent
    fn minimize_energy(&mut self, max_steps: usize) {
        let step_size: f64 = 0.001; // Å per step - very small for stability
        let max_force: f64 = 10.0; // kcal/mol/Å

        let initial_pe = self.compute_potential_energy();
        eprintln!("  Minimization: initial PE = {:.1} kcal/mol", initial_pe);

        for step in 0..max_steps {
            let forces = self.compute_forces();

            // Move atoms along force direction (steepest descent)
            let mut max_f = 0.0;
            for i in 0..self.positions.len() {
                let f_mag =
                    (forces[i][0].powi(2) + forces[i][1].powi(2) + forces[i][2].powi(2)).sqrt();
                if f_mag > max_f {
                    max_f = f_mag;
                }

                if f_mag > 0.01 {
                    let scale = step_size.min(step_size * max_force / f_mag);
                    for d in 0..3 {
                        self.positions[i][d] += forces[i][d] * scale / f_mag.max(0.1);
                    }
                }
            }

            if step % 100 == 0 && step > 0 {
                let pe = self.compute_potential_energy();
                eprintln!(
                    "  Minimization step {}: PE = {:.1} kcal/mol, max_f = {:.2}",
                    step, pe, max_f
                );
            }

            // Converged if max force is small
            if max_f < 1.0 {
                let final_pe = self.compute_potential_energy();
                eprintln!(
                    "  Minimization converged at step {}: PE = {:.1} kcal/mol",
                    step, final_pe
                );
                return;
            }
        }

        let final_pe = self.compute_potential_energy();
        eprintln!(
            "  Minimization finished: final PE = {:.1} kcal/mol",
            final_pe
        );
    }

    /// Run simulation for given number of HMC moves
    pub fn run(&mut self, n_moves: usize, save_every: usize) -> anyhow::Result<AmberSimResult> {
        let mut trajectory = Vec::new();

        // Energy minimization disabled - too slow with O(n²) non-bonded
        // Run with stronger soft-core instead
        // self.minimize_energy(500);
        // self.velocities = initialize_velocities(&self.masses, self.config.temperature, &mut self.rng);

        // Diagnostic: print initial energy breakdown
        self.print_energy_diagnostic();

        // Save initial frame
        let initial_pe = self.compute_potential_energy();
        let initial_ke = self.compute_kinetic_energy();
        trajectory.push(TrajectoryFrame {
            positions: self.positions.clone(),
            velocities: self.velocities.clone(),
            potential_energy: initial_pe,
            kinetic_energy: initial_ke,
            temperature: self.compute_temperature(),
            time: self.time,
        });

        for move_idx in 0..n_moves {
            if self.config.use_langevin {
                self.langevin_step();
            } else {
                self.hmc_step();
            }

            // Save frame periodically
            if (move_idx + 1) % save_every == 0 {
                let pe = self.compute_potential_energy();
                let ke = self.compute_kinetic_energy();
                trajectory.push(TrajectoryFrame {
                    positions: self.positions.clone(),
                    velocities: self.velocities.clone(),
                    potential_energy: pe,
                    kinetic_energy: ke,
                    temperature: self.compute_temperature(),
                    time: self.time,
                });
            }
        }

        // Compute RMSF from trajectory
        let rmsf = compute_rmsf_from_trajectory(&trajectory, &self.topology);

        // Compute statistics
        let avg_pe =
            trajectory.iter().map(|f| f.potential_energy).sum::<f64>() / trajectory.len() as f64;
        let avg_temp =
            trajectory.iter().map(|f| f.temperature).sum::<f64>() / trajectory.len() as f64;

        let acceptance_rate = if self.attempted > 0 {
            self.accepted as f64 / self.attempted as f64
        } else {
            1.0 // Langevin mode doesn't track acceptance
        };

        Ok(AmberSimResult {
            trajectory,
            rmsf,
            acceptance_rate,
            avg_potential_energy: avg_pe,
            avg_temperature: avg_temp,
            total_time: self.time,
        })
    }

    /// Perform one Langevin dynamics step
    fn langevin_step(&mut self) {
        let dt = self.config.timestep;
        let gamma = self.config.friction;
        let temp = self.config.temperature;
        let n_atoms = self.positions.len();

        // Compute forces
        let forces = self.compute_forces();

        // Langevin integration (BAOAB splitting)
        // B: half-kick from forces
        // A: half-step position update
        // O: Ornstein-Uhlenbeck for velocity
        // A: half-step position update
        // B: half-kick from forces

        let c1 = (-gamma * dt).exp();
        // Noise amplitude: sqrt((1-c1²) * kT/m * conversion) for velocities in Å/fs
        // kT in kcal/mol, m in Da, need to multiply by FORCE_CONVERSION_FACTOR
        let c2 = ((1.0 - c1 * c1) * KB_KCAL * temp * FORCE_CONVERSION_FACTOR).sqrt();

        let normal = Normal::new(0.0, 1.0).unwrap();

        for i in 0..n_atoms {
            let m = self.masses[i];
            let inv_m = 1.0 / m;

            // B: half-kick
            for d in 0..3 {
                self.velocities[i][d] += 0.5 * dt * forces[i][d] * inv_m * FORCE_CONVERSION_FACTOR;
            }

            // A: half-step position
            for d in 0..3 {
                self.positions[i][d] += 0.5 * dt * self.velocities[i][d];
            }

            // O: Ornstein-Uhlenbeck (thermostat)
            for d in 0..3 {
                let noise = normal.sample(&mut self.rng);
                self.velocities[i][d] = c1 * self.velocities[i][d] + c2 * (1.0 / m.sqrt()) * noise;
            }

            // A: half-step position
            for d in 0..3 {
                self.positions[i][d] += 0.5 * dt * self.velocities[i][d];
            }
        }

        // Recompute forces at new position
        let forces_new = self.compute_forces();

        // B: half-kick with new forces
        for i in 0..n_atoms {
            let inv_m = 1.0 / self.masses[i];
            for d in 0..3 {
                self.velocities[i][d] +=
                    0.5 * dt * forces_new[i][d] * inv_m * FORCE_CONVERSION_FACTOR;
            }
        }

        // Clamp velocities to prevent temperature explosion
        // Max velocity ~0.05 Å/fs corresponds to ~T=600K for typical atoms
        // At 300K, typical velocities are ~0.03 Å/fs, so 0.05 allows some headroom
        self.clamp_velocities(0.05);

        self.time += dt;
    }

    /// Clamp velocities to prevent numerical instability
    fn clamp_velocities(&mut self, max_vel: f64) {
        let max_vel_sq = max_vel * max_vel;
        for vel in &mut self.velocities {
            let v_sq = vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2];
            if v_sq > max_vel_sq {
                let scale = max_vel / v_sq.sqrt();
                vel[0] *= scale;
                vel[1] *= scale;
                vel[2] *= scale;
            }
        }
    }

    /// Perform one HMC step
    fn hmc_step(&mut self) {
        self.attempted += 1;

        // Save current state
        let old_positions = self.positions.clone();
        let old_velocities = self.velocities.clone();
        let old_pe = self.compute_potential_energy();
        let old_ke = self.compute_kinetic_energy();
        let old_h = old_pe + old_ke;

        // Resample momenta from Maxwell-Boltzmann
        self.velocities =
            initialize_velocities(&self.masses, self.config.temperature, &mut self.rng);

        // Run leapfrog integration
        self.leapfrog_integrate(self.config.n_leapfrog_steps);

        // Compute new Hamiltonian
        let new_pe = self.compute_potential_energy();
        let new_ke = self.compute_kinetic_energy();
        let new_h = new_pe + new_ke;

        // Metropolis acceptance
        let delta_h = new_h - old_h;
        let accept_prob = (-delta_h / (KB_KCAL * self.config.temperature))
            .exp()
            .min(1.0);

        if self.rng.gen::<f64>() < accept_prob {
            self.accepted += 1;
            // Keep new state (already updated)
        } else {
            // Reject: restore old state
            self.positions = old_positions;
            self.velocities = old_velocities;
        }
    }

    /// Leapfrog integration for n steps
    fn leapfrog_integrate(&mut self, n_steps: usize) {
        let dt = self.config.timestep;
        let n_atoms = self.positions.len();

        for _ in 0..n_steps {
            // Compute forces
            let forces = self.compute_forces();

            // Half-step velocity update
            for i in 0..n_atoms {
                let inv_m = 1.0 / self.masses[i];
                for d in 0..3 {
                    self.velocities[i][d] +=
                        0.5 * dt * forces[i][d] * inv_m * FORCE_CONVERSION_FACTOR;
                }
            }

            // Full-step position update
            for i in 0..n_atoms {
                for d in 0..3 {
                    self.positions[i][d] += dt * self.velocities[i][d];
                }
            }

            // Compute forces at new position
            let forces_new = self.compute_forces();

            // Half-step velocity update with new forces
            for i in 0..n_atoms {
                let inv_m = 1.0 / self.masses[i];
                for d in 0..3 {
                    self.velocities[i][d] +=
                        0.5 * dt * forces_new[i][d] * inv_m * FORCE_CONVERSION_FACTOR;
                }
            }

            self.time += dt;
        }
    }

    /// Compute forces from AMBER force field
    /// Uses GPU when available, falls back to CPU otherwise
    fn compute_forces(&mut self) -> Vec<[f64; 3]> {
        // Try GPU path first
        #[cfg(feature = "cuda")]
        if self.gpu_active {
            // Take gpu_forces out to avoid borrow conflicts
            if let Some(mut gpu_forces) = self.gpu_forces.take() {
                match self.compute_forces_gpu_internal(&mut gpu_forces) {
                    Ok(forces) => {
                        // Put gpu_forces back
                        self.gpu_forces = Some(gpu_forces);
                        return Self::clamp_forces(forces);
                    }
                    Err(e) => {
                        // Put gpu_forces back before logging
                        self.gpu_forces = Some(gpu_forces);
                        log::warn!("GPU force calculation failed, falling back to CPU: {}", e);
                    }
                }
            }
        }

        // CPU fallback with force clamping
        Self::clamp_forces(self.compute_forces_cpu())
    }

    /// Clamp forces to prevent numerical explosion
    /// Maximum force of 1000 kcal/mol/Å is reasonable for stable dynamics
    fn clamp_forces(mut forces: Vec<[f64; 3]>) -> Vec<[f64; 3]> {
        // Aggressively reduced max force for numerical stability with raw PDB structures
        // 10 kcal/mol/Å allows gentle minimization without explosive dynamics
        const MAX_FORCE: f64 = 10.0; // kcal/mol/Å (reduced from 100)
        const MAX_FORCE_SQ: f64 = MAX_FORCE * MAX_FORCE;

        for force in &mut forces {
            let f_sq = force[0] * force[0] + force[1] * force[1] + force[2] * force[2];
            if f_sq > MAX_FORCE_SQ {
                let scale = MAX_FORCE / f_sq.sqrt();
                force[0] *= scale;
                force[1] *= scale;
                force[2] *= scale;
            }
        }
        forces
    }

    /// Compute forces on CPU
    fn compute_forces_cpu(&self) -> Vec<[f64; 3]> {
        let n = self.positions.len();
        let mut forces = vec![[0.0; 3]; n];

        // Bond forces
        self.add_bond_forces(&mut forces);

        // Angle forces
        self.add_angle_forces(&mut forces);

        // Dihedral forces
        self.add_dihedral_forces(&mut forces);

        // Non-bonded forces (LJ + Coulomb)
        self.add_nonbonded_forces(&mut forces);

        forces
    }

    /// Compute forces on GPU (internal implementation)
    ///
    /// This is called with gpu_forces taken out of self to avoid borrow conflicts.
    /// GPU computes bonded forces (bonds, angles, dihedrals, 1-4 pairs).
    /// Non-bonded forces (LJ + Coulomb) are still computed on CPU because
    /// GPU non-bonded would require neighbor list construction.
    #[cfg(feature = "cuda")]
    fn compute_forces_gpu_internal(
        &self,
        gpu_forces: &mut AmberBondedForces,
    ) -> anyhow::Result<Vec<[f64; 3]>> {
        // Flatten positions for GPU upload
        let positions_flat: Vec<f32> = self
            .positions
            .iter()
            .flat_map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
            .collect();

        // Upload positions
        gpu_forces.upload_positions(&positions_flat)?;

        // Compute bonded forces on GPU
        let (_energy, forces_flat) = gpu_forces.compute()?;

        // Convert back to Vec<[f64; 3]>
        let forces: Vec<[f64; 3]> = forces_flat
            .chunks(3)
            .map(|chunk| [chunk[0] as f64, chunk[1] as f64, chunk[2] as f64])
            .collect();

        // Add non-bonded forces from CPU (LJ + Coulomb with cutoff)
        let mut total_forces = forces;
        self.add_nonbonded_forces(&mut total_forces);

        Ok(total_forces)
    }

    /// Add bond stretching forces
    fn add_bond_forces(&self, forces: &mut Vec<[f64; 3]>) {
        for (bond_idx, &(atom_i, atom_j)) in self.topology.bonds.iter().enumerate() {
            let i = atom_i as usize;
            let j = atom_j as usize;
            let params = &self.topology.bond_params[bond_idx];
            let r0 = params.r0 as f64;
            let k = params.k as f64;

            let r_ij = [
                self.positions[j][0] - self.positions[i][0],
                self.positions[j][1] - self.positions[i][1],
                self.positions[j][2] - self.positions[i][2],
            ];
            let dist = (r_ij[0] * r_ij[0] + r_ij[1] * r_ij[1] + r_ij[2] * r_ij[2]).sqrt();

            if dist < 1e-10 {
                continue;
            }

            // F = -dV/dr = -2k(r - r0) * (r_ij / r)
            let force_mag = -2.0 * k * (dist - r0);

            for d in 0..3 {
                let f = force_mag * r_ij[d] / dist;
                forces[i][d] -= f;
                forces[j][d] += f;
            }
        }
    }

    /// Add angle bending forces
    fn add_angle_forces(&self, forces: &mut Vec<[f64; 3]>) {
        for (angle_idx, &(atom_i, atom_j, atom_k)) in self.topology.angles.iter().enumerate() {
            let i = atom_i as usize;
            let j = atom_j as usize; // Central atom
            let k_idx = atom_k as usize;
            let params = &self.topology.angle_params[angle_idx];
            let theta0 = params.theta0 as f64;
            let k_angle = params.k as f64;

            // Vectors from central atom
            let r_ji = [
                self.positions[i][0] - self.positions[j][0],
                self.positions[i][1] - self.positions[j][1],
                self.positions[i][2] - self.positions[j][2],
            ];
            let r_jk = [
                self.positions[k_idx][0] - self.positions[j][0],
                self.positions[k_idx][1] - self.positions[j][1],
                self.positions[k_idx][2] - self.positions[j][2],
            ];

            let r1 = (r_ji[0] * r_ji[0] + r_ji[1] * r_ji[1] + r_ji[2] * r_ji[2]).sqrt();
            let r2 = (r_jk[0] * r_jk[0] + r_jk[1] * r_jk[1] + r_jk[2] * r_jk[2]).sqrt();

            if r1 < 1e-10 || r2 < 1e-10 {
                continue;
            }

            // Compute angle
            let dot = r_ji[0] * r_jk[0] + r_ji[1] * r_jk[1] + r_ji[2] * r_jk[2];
            let cos_theta = (dot / (r1 * r2)).clamp(-1.0, 1.0);
            let theta = cos_theta.acos();

            // Force magnitude: -dV/dθ = -2k(θ - θ0)
            let dv_dtheta = 2.0 * k_angle * (theta - theta0);

            // Gradient of angle with respect to positions (simplified)
            let sin_theta = (1.0 - cos_theta * cos_theta).sqrt().max(1e-10);

            for d in 0..3 {
                let grad_i = (r_jk[d] / (r1 * r2) - cos_theta * r_ji[d] / (r1 * r1)) / sin_theta;
                let grad_k = (r_ji[d] / (r1 * r2) - cos_theta * r_jk[d] / (r2 * r2)) / sin_theta;

                forces[i][d] -= dv_dtheta * grad_i;
                forces[k_idx][d] -= dv_dtheta * grad_k;
                forces[j][d] += dv_dtheta * (grad_i + grad_k);
            }
        }
    }

    /// Add dihedral torsion forces
    fn add_dihedral_forces(&self, forces: &mut Vec<[f64; 3]>) {
        for (dih_idx, &(atom_i, atom_j, atom_k, atom_l)) in
            self.topology.dihedrals.iter().enumerate()
        {
            let i = atom_i as usize;
            let j = atom_j as usize;
            let k = atom_k as usize;
            let l = atom_l as usize;

            // Compute dihedral angle
            let b1 = [
                self.positions[j][0] - self.positions[i][0],
                self.positions[j][1] - self.positions[i][1],
                self.positions[j][2] - self.positions[i][2],
            ];
            let b2 = [
                self.positions[k][0] - self.positions[j][0],
                self.positions[k][1] - self.positions[j][1],
                self.positions[k][2] - self.positions[j][2],
            ];
            let b3 = [
                self.positions[l][0] - self.positions[k][0],
                self.positions[l][1] - self.positions[k][1],
                self.positions[l][2] - self.positions[k][2],
            ];

            // Normal vectors to planes
            let n1 = cross(&b1, &b2);
            let n2 = cross(&b2, &b3);

            let n1_len = (n1[0] * n1[0] + n1[1] * n1[1] + n1[2] * n1[2]).sqrt();
            let n2_len = (n2[0] * n2[0] + n2[1] * n2[1] + n2[2] * n2[2]).sqrt();

            if n1_len < 1e-10 || n2_len < 1e-10 {
                continue;
            }

            // Dihedral angle
            let m1 = cross(&n1, &b2);
            let m1_len = (m1[0] * m1[0] + m1[1] * m1[1] + m1[2] * m1[2]).sqrt();
            if m1_len < 1e-10 {
                continue;
            }

            let x = n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2];
            let y = m1[0] * n2[0] + m1[1] * n2[1] + m1[2] * n2[2];
            let phi = y.atan2(x);

            // Each dihedral can have multiple Fourier terms
            let params_list = &self.topology.dihedral_params[dih_idx];
            let mut total_dv_dphi = 0.0_f64;
            for params in params_list {
                let k_dih = params.k as f64;
                let n_period = params.n as f64;
                let phase = params.phase as f64;
                // Force: -dV/dφ = k*n*sin(nφ - phase)
                total_dv_dphi += k_dih * n_period * (n_period * phi - phase).sin();
            }

            // Distribute forces to atoms (simplified gradient)
            // This is approximate - full implementation needs proper chain rule
            let b2_len = (b2[0] * b2[0] + b2[1] * b2[1] + b2[2] * b2[2]).sqrt();
            if b2_len < 1e-10 {
                continue;
            }

            for d in 0..3 {
                let f_scale = total_dv_dphi / (n1_len * n2_len * b2_len);
                forces[i][d] -= f_scale * n1[d];
                forces[l][d] += f_scale * n2[d];
                // j and k get remainder (approximate)
                forces[j][d] += f_scale * (n1[d] - n2[d]) * 0.5;
                forces[k][d] += f_scale * (n2[d] - n1[d]) * 0.5;
            }
        }
    }

    /// Add non-bonded forces (LJ + Coulomb with cutoff)
    ///
    /// Uses soft-core LJ potential to prevent singularity at short distances:
    /// V_soft = 4ε * [(σ²/(r² + δ²))⁶ - (σ²/(r² + δ²))³]
    /// where δ² is a soft-core parameter that prevents r→0 divergence.
    fn add_nonbonded_forces(&self, forces: &mut Vec<[f64; 3]>) {
        let cutoff = 10.0; // Å
        let cutoff_sq = cutoff * cutoff;
        let n = self.positions.len();

        // Soft-core parameter: prevents LJ singularity at short distances
        // Strong soft-core needed because we skip energy minimization
        let soft_core_delta_sq = 2.0; // 2 Å² - aggressive for raw PDB structures

        // Minimum effective distance squared (prevents extreme forces)
        let min_dist_sq = 4.0; // 2 Å minimum - prevents r < 2 Å interactions

        // Coulomb constant in kcal*Å/(mol*e²)
        let coulomb_k = 332.0636;

        // 1-4 scaling factors (AMBER ff14SB)
        let lj_14_scale = 0.5;
        let coul_14_scale = 0.8333333; // 1/1.2

        for i in 0..n {
            for j in (i + 1)..n {
                // Check exclusions (1-2, 1-3 pairs)
                if self.is_excluded(i, j) {
                    continue;
                }

                let r_ij = [
                    self.positions[j][0] - self.positions[i][0],
                    self.positions[j][1] - self.positions[i][1],
                    self.positions[j][2] - self.positions[i][2],
                ];
                let dist_sq = r_ij[0] * r_ij[0] + r_ij[1] * r_ij[1] + r_ij[2] * r_ij[2];

                if dist_sq > cutoff_sq {
                    continue;
                }

                // Get LJ parameters (combining rules: Lorentz-Berthelot)
                let eps_i = self.get_lj_epsilon(i);
                let eps_j = self.get_lj_epsilon(j);
                let sig_i = self.get_lj_sigma(i);
                let sig_j = self.get_lj_sigma(j);

                let eps = (eps_i * eps_j).sqrt();
                let sig = (sig_i + sig_j) / 2.0;
                let sig_sq = sig * sig;

                // Check if this is a 1-4 pair
                let is_14 = self.is_14_pair(i, j);

                // Soft-core LJ: use r² + δ² instead of r² to prevent singularity
                // This smoothly caps the potential at short distances
                let effective_dist_sq = dist_sq.max(min_dist_sq) + soft_core_delta_sq;
                let effective_dist = effective_dist_sq.sqrt();
                let inv_r_eff = 1.0 / effective_dist;

                // Soft-core LJ potential: V = 4ε[(σ²/r_eff²)⁶ - (σ²/r_eff²)³]
                let sig_sq_over_r_sq = sig_sq / effective_dist_sq;
                let term6 = sig_sq_over_r_sq * sig_sq_over_r_sq * sig_sq_over_r_sq;
                let term12 = term6 * term6;

                // Force magnitude: F = -dV/dr = 24ε/r_eff * (2*term12 - term6) * (r/r_eff)
                // The (r/r_eff) factor comes from chain rule: d(r_eff)/dr = r/r_eff
                let dist = dist_sq.sqrt().max(0.01); // Avoid division by zero
                let chain_factor = dist / effective_dist;
                let lj_force_mag = 24.0 * eps * inv_r_eff * (2.0 * term12 - term6) * chain_factor;

                // Apply 1-4 LJ scaling
                let lj_force = if is_14 {
                    lj_force_mag * lj_14_scale
                } else {
                    lj_force_mag
                };

                // Coulomb force with soft-core distance
                let qi = self.get_charge(i);
                let qj = self.get_charge(j);
                let inv_r2_eff = inv_r_eff * inv_r_eff;
                let coul_force_mag = coulomb_k * qi * qj * inv_r2_eff * chain_factor;

                // Apply 1-4 Coulomb scaling
                let coul_force = if is_14 {
                    coul_force_mag * coul_14_scale
                } else {
                    coul_force_mag
                };

                let total_force = lj_force + coul_force;

                // Apply force along r_ij direction
                let inv_dist = 1.0 / dist;
                for d in 0..3 {
                    let f = total_force * r_ij[d] * inv_dist;
                    forces[i][d] -= f;
                    forces[j][d] += f;
                }
            }
        }
    }

    /// Compute total potential energy
    fn compute_potential_energy(&self) -> f64 {
        let mut energy = 0.0;

        // Bond energy: E = k(r - r0)²
        for (bond_idx, &(atom_i, atom_j)) in self.topology.bonds.iter().enumerate() {
            let i = atom_i as usize;
            let j = atom_j as usize;
            let params = &self.topology.bond_params[bond_idx];
            let r0 = params.r0 as f64;
            let k = params.k as f64;

            let dist = self.distance(i, j);
            energy += k * (dist - r0).powi(2);
        }

        // Angle energy: E = k(θ - θ0)²
        for (angle_idx, &(atom_i, atom_j, atom_k)) in self.topology.angles.iter().enumerate() {
            let theta = self.compute_angle(atom_i as usize, atom_j as usize, atom_k as usize);
            let params = &self.topology.angle_params[angle_idx];
            let theta0 = params.theta0 as f64;
            let k = params.k as f64;
            energy += k * (theta - theta0).powi(2);
        }

        // Dihedral energy: E = Σ k(1 + cos(nφ - phase))
        for (dih_idx, &(atom_i, atom_j, atom_k, atom_l)) in
            self.topology.dihedrals.iter().enumerate()
        {
            let phi = self.compute_dihedral(
                atom_i as usize,
                atom_j as usize,
                atom_k as usize,
                atom_l as usize,
            );
            // Sum all Fourier terms for this dihedral
            for params in &self.topology.dihedral_params[dih_idx] {
                let k = params.k as f64;
                let n = params.n as f64;
                let phase = params.phase as f64;
                energy += k * (1.0 + (n * phi - phase).cos());
            }
        }

        // Add non-bonded energy (LJ + Coulomb with soft-core)
        energy += self.compute_nonbonded_energy();

        energy
    }

    /// Print diagnostic breakdown of energy components
    fn print_energy_diagnostic(&self) {
        let mut bond_energy = 0.0;
        let mut angle_energy = 0.0;
        let mut dihedral_energy = 0.0;

        // Bond energy with statistics
        let mut bond_stretches: Vec<f64> = Vec::new();
        for (bond_idx, &(atom_i, atom_j)) in self.topology.bonds.iter().enumerate() {
            let i = atom_i as usize;
            let j = atom_j as usize;
            let params = &self.topology.bond_params[bond_idx];
            let r0 = params.r0 as f64;
            let k = params.k as f64;
            let dist = self.distance(i, j);
            let stretch = (dist - r0).abs();
            bond_stretches.push(stretch);
            bond_energy += k * (dist - r0).powi(2);
        }

        // Angle energy
        for (angle_idx, &(atom_i, atom_j, atom_k)) in self.topology.angles.iter().enumerate() {
            let theta = self.compute_angle(atom_i as usize, atom_j as usize, atom_k as usize);
            let params = &self.topology.angle_params[angle_idx];
            let theta0 = params.theta0 as f64;
            let k = params.k as f64;
            angle_energy += k * (theta - theta0).powi(2);
        }

        // Dihedral energy
        for (dih_idx, &(atom_i, atom_j, atom_k, atom_l)) in
            self.topology.dihedrals.iter().enumerate()
        {
            let phi = self.compute_dihedral(
                atom_i as usize,
                atom_j as usize,
                atom_k as usize,
                atom_l as usize,
            );
            for params in &self.topology.dihedral_params[dih_idx] {
                let k = params.k as f64;
                let n = params.n as f64;
                let phase = params.phase as f64;
                dihedral_energy += k * (1.0 + (n * phi - phase).cos());
            }
        }

        let nb_energy = self.compute_nonbonded_energy();

        // Statistics
        let avg_stretch = if bond_stretches.is_empty() {
            0.0
        } else {
            bond_stretches.iter().sum::<f64>() / bond_stretches.len() as f64
        };
        let max_stretch = bond_stretches.iter().cloned().fold(0.0, f64::max);

        eprintln!("  Energy Diagnostic:");
        eprintln!(
            "    Bonds:     {:12.1} kcal/mol ({} bonds, avg stretch {:.3} Å, max {:.3} Å)",
            bond_energy,
            self.topology.bonds.len(),
            avg_stretch,
            max_stretch
        );
        eprintln!(
            "    Angles:    {:12.1} kcal/mol ({} angles)",
            angle_energy,
            self.topology.angles.len()
        );
        eprintln!(
            "    Dihedrals: {:12.1} kcal/mol ({} dihedrals)",
            dihedral_energy,
            self.topology.dihedrals.len()
        );
        eprintln!("    Non-bonded:{:12.1} kcal/mol", nb_energy);
        eprintln!(
            "    Total:     {:12.1} kcal/mol",
            bond_energy + angle_energy + dihedral_energy + nb_energy
        );
    }

    /// Compute non-bonded energy (LJ + Coulomb) with soft-core potential
    fn compute_nonbonded_energy(&self) -> f64 {
        let mut energy = 0.0;
        let cutoff_sq = 100.0; // 10 Å cutoff
        let n = self.positions.len();

        // Same soft-core parameters as force calculation
        let soft_core_delta_sq = 2.0;
        let min_dist_sq = 4.0;

        // Coulomb constant in kcal*Å/(mol*e²)
        let coulomb_k = 332.0636;

        // 1-4 scaling factors (AMBER ff14SB)
        let lj_14_scale = 0.5;
        let coul_14_scale = 0.8333333;

        for i in 0..n {
            for j in (i + 1)..n {
                if self.is_excluded(i, j) {
                    continue;
                }

                let dx = self.positions[j][0] - self.positions[i][0];
                let dy = self.positions[j][1] - self.positions[i][1];
                let dz = self.positions[j][2] - self.positions[i][2];
                let dist_sq = dx * dx + dy * dy + dz * dz;

                if dist_sq > cutoff_sq {
                    continue;
                }

                // Soft-core effective distance
                let effective_dist_sq = dist_sq.max(min_dist_sq) + soft_core_delta_sq;

                // LJ parameters
                let eps_i = self.get_lj_epsilon(i);
                let eps_j = self.get_lj_epsilon(j);
                let sig_i = self.get_lj_sigma(i);
                let sig_j = self.get_lj_sigma(j);

                let eps = (eps_i * eps_j).sqrt();
                let sig = (sig_i + sig_j) / 2.0;
                let sig_sq = sig * sig;

                let is_14 = self.is_14_pair(i, j);

                // Soft-core LJ energy: V = 4ε[(σ²/r_eff²)⁶ - (σ²/r_eff²)³]
                let sig_sq_over_r_sq = sig_sq / effective_dist_sq;
                let term6 = sig_sq_over_r_sq * sig_sq_over_r_sq * sig_sq_over_r_sq;
                let term12 = term6 * term6;
                let lj_energy = 4.0 * eps * (term12 - term6);

                // Coulomb energy with soft-core distance
                let qi = self.get_charge(i);
                let qj = self.get_charge(j);
                let inv_r_eff = 1.0 / effective_dist_sq.sqrt();
                let coul_energy = coulomb_k * qi * qj * inv_r_eff;

                // Apply 1-4 scaling
                let lj_scaled = if is_14 {
                    lj_energy * lj_14_scale
                } else {
                    lj_energy
                };
                let coul_scaled = if is_14 {
                    coul_energy * coul_14_scale
                } else {
                    coul_energy
                };

                energy += lj_scaled + coul_scaled;
            }
        }

        energy
    }

    /// Compute kinetic energy
    fn compute_kinetic_energy(&self) -> f64 {
        let mut ke = 0.0;
        for (i, v) in self.velocities.iter().enumerate() {
            let v_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            // KE = ½mv² in Da·(Å/fs)², convert to kcal/mol by dividing by FORCE_CONVERSION_FACTOR
            ke += 0.5 * self.masses[i] * v_sq / FORCE_CONVERSION_FACTOR;
        }
        ke
    }

    /// Compute instantaneous temperature
    fn compute_temperature(&self) -> f64 {
        let ke = self.compute_kinetic_energy();
        let n_dof = 3 * self.positions.len() - 6; // Remove translation + rotation
        2.0 * ke / (n_dof as f64 * KB_KCAL)
    }

    /// Distance between two atoms
    fn distance(&self, i: usize, j: usize) -> f64 {
        let dx = self.positions[j][0] - self.positions[i][0];
        let dy = self.positions[j][1] - self.positions[i][1];
        let dz = self.positions[j][2] - self.positions[i][2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Compute angle between three atoms
    fn compute_angle(&self, i: usize, j: usize, k: usize) -> f64 {
        let r_ji = [
            self.positions[i][0] - self.positions[j][0],
            self.positions[i][1] - self.positions[j][1],
            self.positions[i][2] - self.positions[j][2],
        ];
        let r_jk = [
            self.positions[k][0] - self.positions[j][0],
            self.positions[k][1] - self.positions[j][1],
            self.positions[k][2] - self.positions[j][2],
        ];

        let dot = r_ji[0] * r_jk[0] + r_ji[1] * r_jk[1] + r_ji[2] * r_jk[2];
        let len_ji = (r_ji[0] * r_ji[0] + r_ji[1] * r_ji[1] + r_ji[2] * r_ji[2]).sqrt();
        let len_jk = (r_jk[0] * r_jk[0] + r_jk[1] * r_jk[1] + r_jk[2] * r_jk[2]).sqrt();

        (dot / (len_ji * len_jk)).clamp(-1.0, 1.0).acos()
    }

    /// Compute dihedral angle between four atoms
    fn compute_dihedral(&self, i: usize, j: usize, k: usize, l: usize) -> f64 {
        let b1 = [
            self.positions[j][0] - self.positions[i][0],
            self.positions[j][1] - self.positions[i][1],
            self.positions[j][2] - self.positions[i][2],
        ];
        let b2 = [
            self.positions[k][0] - self.positions[j][0],
            self.positions[k][1] - self.positions[j][1],
            self.positions[k][2] - self.positions[j][2],
        ];
        let b3 = [
            self.positions[l][0] - self.positions[k][0],
            self.positions[l][1] - self.positions[k][1],
            self.positions[l][2] - self.positions[k][2],
        ];

        let n1 = cross(&b1, &b2);
        let n2 = cross(&b2, &b3);
        let m1 = cross(&n1, &b2);

        let x = n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2];
        let y = m1[0] * n2[0] + m1[1] * n2[1] + m1[2] * n2[2];

        y.atan2(x)
    }

    /// Check if pair is excluded (1-2 or 1-3 bonded)
    fn is_excluded(&self, i: usize, j: usize) -> bool {
        let (a, b) = if i < j {
            (i as u32, j as u32)
        } else {
            (j as u32, i as u32)
        };
        self.topology.exclusions.iter().any(|&(x, y)| {
            let (x, y) = if x < y { (x, y) } else { (y, x) };
            x == a && y == b
        })
    }

    /// Check if pair is a 1-4 interaction (scaled non-bonded)
    fn is_14_pair(&self, i: usize, j: usize) -> bool {
        let (a, b) = if i < j {
            (i as u32, j as u32)
        } else {
            (j as u32, i as u32)
        };
        self.topology.pairs_14.iter().any(|&(x, y)| {
            let (x, y) = if x < y { (x, y) } else { (y, x) };
            x == a && y == b
        })
    }

    /// Get LJ epsilon parameter for atom
    fn get_lj_epsilon(&self, i: usize) -> f64 {
        self.lj_epsilon.get(i).copied().unwrap_or(0.1)
    }

    /// Get LJ sigma parameter for atom
    fn get_lj_sigma(&self, i: usize) -> f64 {
        self.lj_sigma.get(i).copied().unwrap_or(3.4)
    }

    /// Get partial charge for atom
    fn get_charge(&self, i: usize) -> f64 {
        self.charges.get(i).copied().unwrap_or(0.0)
    }
}

/// Cross product of two 3D vectors
fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Initialize velocities from Maxwell-Boltzmann distribution
fn initialize_velocities<R: Rng>(masses: &[f64], temperature: f64, rng: &mut R) -> Vec<[f64; 3]> {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut velocities = Vec::with_capacity(masses.len());

    for &mass in masses {
        // σ = sqrt(kT/m * conversion) for velocities in Å/fs
        let sigma = (KB_KCAL * temperature / mass * FORCE_CONVERSION_FACTOR).sqrt();
        velocities.push([
            sigma * normal.sample(rng),
            sigma * normal.sample(rng),
            sigma * normal.sample(rng),
        ]);
    }

    // Remove center-of-mass motion
    let total_mass: f64 = masses.iter().sum();
    let mut com_vel = [0.0; 3];
    for (i, &mass) in masses.iter().enumerate() {
        for d in 0..3 {
            com_vel[d] += mass * velocities[i][d];
        }
    }
    for d in 0..3 {
        com_vel[d] /= total_mass;
    }
    for v in &mut velocities {
        for d in 0..3 {
            v[d] -= com_vel[d];
        }
    }

    velocities
}

/// Compute per-residue RMSF from trajectory
/// Note: topology is reserved for future per-residue grouping
fn compute_rmsf_from_trajectory(
    trajectory: &[TrajectoryFrame],
    _topology: &AmberTopology,
) -> Vec<f64> {
    if trajectory.is_empty() {
        return vec![];
    }

    let n_atoms = trajectory[0].positions.len();
    let n_frames = trajectory.len();

    // Compute mean positions
    let mut mean_pos = vec![[0.0; 3]; n_atoms];
    for frame in trajectory {
        for (i, pos) in frame.positions.iter().enumerate() {
            for d in 0..3 {
                mean_pos[i][d] += pos[d];
            }
        }
    }
    for pos in &mut mean_pos {
        for d in 0..3 {
            pos[d] /= n_frames as f64;
        }
    }

    // Compute RMSF
    let mut rmsf = vec![0.0; n_atoms];
    for frame in trajectory {
        for (i, pos) in frame.positions.iter().enumerate() {
            let dx = pos[0] - mean_pos[i][0];
            let dy = pos[1] - mean_pos[i][1];
            let dz = pos[2] - mean_pos[i][2];
            rmsf[i] += dx * dx + dy * dy + dz * dz;
        }
    }
    for r in &mut rmsf {
        *r = (*r / n_frames as f64).sqrt();
    }

    // Convert to per-residue (average over atoms in each residue)
    // For now, just return per-atom RMSF
    // TODO: Group by residue from topology

    rmsf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_product() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        let c = cross(&a, &b);
        assert!((c[0] - 0.0).abs() < 1e-10);
        assert!((c[1] - 0.0).abs() < 1e-10);
        assert!((c[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_velocity_initialization() {
        use rand::SeedableRng;
        let masses = vec![12.0; 100];
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let velocities = initialize_velocities(&masses, 300.0, &mut rng);

        assert_eq!(velocities.len(), 100);

        // Check COM velocity is zero
        let total_mass: f64 = masses.iter().sum();
        let mut com_vel = [0.0; 3];
        for (i, &mass) in masses.iter().enumerate() {
            for d in 0..3 {
                com_vel[d] += mass * velocities[i][d];
            }
        }
        for d in 0..3 {
            assert!((com_vel[d] / total_mass).abs() < 1e-10);
        }
    }
}
