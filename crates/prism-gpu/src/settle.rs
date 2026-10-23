//! SETTLE - Analytical constraint solver for rigid water
//!
//! Maintains the topology-admitted rigid three-site water geometry exactly
//! during MD integration.  TIP3P and TIP3P-FB are distinct parameter
//! contracts and are never admitted interchangeably.
//!
//! The SETTLE algorithm analytically solves the constraint equations
//! in a single step, making it more efficient than iterative methods
//! like SHAKE for rigid water.
//!
//! Reference: Miyamoto & Kollman (1992) J. Comput. Chem. 13:952-962

use anyhow::{Context, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceSlice, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use sha2::{Digest, Sha256};
use std::sync::Arc;

/// Topology-admitted parameters for a rigid three-site water model.
#[derive(Clone, Debug, serde::Serialize)]
pub struct RigidWaterParameters {
    pub model_id: String,
    pub oh_distance_angstrom: f32,
    pub hh_distance_angstrom: f32,
    pub hoh_angle_radians: f32,
    pub mass_o: f32,
    pub mass_h: f32,
    pub source_artifact_sha256: String,
    pub parameter_sha256: String,
}

impl RigidWaterParameters {
    pub fn new(
        model_id: impl Into<String>,
        oh_distance_angstrom: f32,
        hoh_angle_radians: f32,
        mass_o: f32,
        mass_h: f32,
        source_artifact_sha256: impl Into<String>,
    ) -> Result<Self> {
        let model_id = model_id.into();
        let source_artifact_sha256 = source_artifact_sha256.into();
        anyhow::ensure!(!model_id.trim().is_empty(), "water model id is empty");
        anyhow::ensure!(
            oh_distance_angstrom.is_finite() && oh_distance_angstrom > 0.0,
            "water OH distance must be finite and positive"
        );
        anyhow::ensure!(
            hoh_angle_radians.is_finite()
                && hoh_angle_radians > 0.0
                && hoh_angle_radians < std::f32::consts::PI,
            "water HOH angle must be in (0, pi)"
        );
        anyhow::ensure!(
            mass_o.is_finite() && mass_o > 0.0 && mass_h.is_finite() && mass_h > 0.0,
            "water masses must be finite and positive"
        );
        anyhow::ensure!(
            source_artifact_sha256.len() == 64
                && source_artifact_sha256
                    .chars()
                    .all(|c| c.is_ascii_hexdigit()),
            "water source artifact SHA256 must be 64 hex characters"
        );
        let hh_distance_angstrom = 2.0 * oh_distance_angstrom * (0.5 * hoh_angle_radians).sin();
        // Hash the exact IEEE-754 values consumed by the kernels.  Decimal
        // formatting would make identity depend on a presentation choice.
        let canonical = format!(
            "model={model_id};oh_f32_bits={:08x};hh_f32_bits={:08x};mass_o_f32_bits={:08x};mass_h_f32_bits={:08x}",
            oh_distance_angstrom.to_bits(),
            hh_distance_angstrom.to_bits(),
            mass_o.to_bits(),
            mass_h.to_bits()
        );
        let parameter_sha256 = hex::encode(Sha256::digest(canonical.as_bytes()));
        Ok(Self {
            model_id,
            oh_distance_angstrom,
            hh_distance_angstrom,
            hoh_angle_radians,
            mass_o,
            mass_h,
            source_artifact_sha256,
            parameter_sha256,
        })
    }

    /// Legacy three-site TIP3P parameters.  This is not TIP3P-FB.
    pub fn tip3p() -> Self {
        Self::new(
            "TIP3P",
            0.9572,
            104.52_f32.to_radians(),
            15.9994,
            1.008,
            "0000000000000000000000000000000000000000000000000000000000000000",
        )
        .expect("built-in TIP3P parameters are valid")
    }

    /// Parameters from OpenMM `tip3pfb.xml` used by `amber14/tip3pfb.xml`.
    pub fn tip3p_fb_openmm() -> Self {
        Self::new(
            "TIP3P-FB",
            1.011_810_8,
            1.887_546_4,
            15.999_43,
            1.007_947,
            "2c8bf8b0d32656ecf68a86a7e984bd48142a0e7c1d79518b59e40c8ad28cd21d",
        )
        .expect("built-in TIP3P-FB parameters are valid")
    }

    /// Fail-closed comparison against the topology/WaterContract admission.
    pub fn validate_admission(&self, model_id: &str, parameter_sha256: &str) -> Result<()> {
        anyhow::ensure!(
            self.model_id == model_id,
            "rigid-water model mismatch: SETTLE={} topology={}",
            self.model_id,
            model_id
        );
        anyhow::ensure!(
            self.parameter_sha256 == parameter_sha256,
            "rigid-water parameter hash mismatch: SETTLE={} topology={}",
            self.parameter_sha256,
            parameter_sha256
        );
        Ok(())
    }

    /// Validate the full prepared-water admission before binding its contract
    /// hash into a SETTLE instance.
    pub fn validate_topology_admission(
        &self,
        admission: &RigidWaterAdmission,
        actual_n_waters: usize,
    ) -> Result<()> {
        const PARAMETER_TOLERANCE: f64 = 1.0e-6;
        anyhow::ensure!(
            admission.model_id == self.model_id,
            "rigid-water model mismatch: SETTLE={} topology={}",
            self.model_id,
            admission.model_id
        );
        anyhow::ensure!(
            admission.n_waters == actual_n_waters,
            "rigid-water count mismatch: SETTLE={} topology={}",
            actual_n_waters,
            admission.n_waters
        );
        anyhow::ensure!(
            admission.atom_order == "OHH",
            "SETTLE requires topology water atom order OHH, got {}",
            admission.atom_order
        );
        anyhow::ensure!(
            admission.contract_sha256.len() == 64
                && admission
                    .contract_sha256
                    .chars()
                    .all(|c| c.is_ascii_hexdigit()),
            "water topology contract SHA256 must be 64 hex characters"
        );
        for (label, actual, expected) in [
            (
                "OH1",
                admission.oh_distances_angstrom[0],
                self.oh_distance_angstrom as f64,
            ),
            (
                "OH2",
                admission.oh_distances_angstrom[1],
                self.oh_distance_angstrom as f64,
            ),
            (
                "HH",
                admission.hh_distance_angstrom,
                self.hh_distance_angstrom as f64,
            ),
            (
                "HOH angle",
                admission.hoh_angle_radians,
                self.hoh_angle_radians as f64,
            ),
            (
                "oxygen mass",
                admission.masses_dalton[0],
                self.mass_o as f64,
            ),
            ("H1 mass", admission.masses_dalton[1], self.mass_h as f64),
            ("H2 mass", admission.masses_dalton[2], self.mass_h as f64),
        ] {
            anyhow::ensure!(
                actual.is_finite() && (actual - expected).abs() <= PARAMETER_TOLERANCE,
                "rigid-water {label} mismatch: SETTLE={expected:.9} topology={actual:.9}"
            );
        }
        Ok(())
    }
}

/// Relevant fields from a prepared `water_model_contract`.
///
/// The contract digest is deliberately separate from the physical-parameter
/// digest because it also binds water count and instance-map provenance.
#[derive(Clone, Debug, serde::Serialize)]
pub struct RigidWaterAdmission {
    pub model_id: String,
    pub contract_sha256: String,
    pub n_waters: usize,
    pub atom_order: String,
    pub oh_distances_angstrom: [f64; 2],
    pub hh_distance_angstrom: f64,
    pub hoh_angle_radians: f64,
    pub masses_dalton: [f64; 3],
}

/// Queryable runtime capability for lambda/engine admission.
#[derive(Clone, Debug, serde::Serialize)]
pub struct SettleRuntimeCapabilityReceipt {
    pub schema: &'static str,
    pub supplied_stream_bound: bool,
    pub stream_handle: usize,
    pub position_constraints: &'static str,
    pub velocity_constraints: &'static str,
    pub preserves_mass_weighted_com: bool,
    pub preserves_rigid_rotation: bool,
    pub singular_geometry_fails_closed: bool,
    pub graph_hot_path_dtoh_count: usize,
    pub graph_capture_dependency_mode: &'static str,
    pub device_failure_latch: bool,
    pub n_waters: usize,
    pub topology_contract_bound: bool,
    pub water_contract_sha256: Option<String>,
    pub water_model_id: String,
    pub water_parameter_sha256: String,
    pub water_source_artifact_sha256: String,
}

/// TIP3P water geometry parameters
pub const TIP3P_OH_DISTANCE: f32 = 0.9572; // Å
pub const TIP3P_HH_DISTANCE: f32 = 1.513_900_6; // Å (computed from angle)
pub const TIP3P_MASS_O: f32 = 15.9994; // g/mol
pub const TIP3P_MASS_H: f32 = 1.008; // g/mol

/// SETTLE constraint solver for rigid water molecules
pub struct Settle {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,

    // Kernels
    settle_kernel: CudaFunction,
    velocity_correction_kernel: CudaFunction,

    // Device buffers
    d_water_indices: CudaSlice<i32>, // [n_waters * 3] - O, H1, H2 indices for each water
    d_old_positions: CudaSlice<f32>, // [n_total_atoms * 3] - positions before integration
    d_failure_count: CudaSlice<i32>, // [1] - fail-closed kernel status

    // Water molecule parameters
    n_waters: usize,
    n_total_atoms: usize,

    // Masses for SETTLE (precomputed)
    mass_o: f32,
    mass_h: f32,
    ra: f32, // Distance from O to center of mass
    rb: f32, // Distance from H to center of mass along OH
    rc: f32, // Half HH distance

    // Target distances (squared for efficiency)
    roh2: f32, // OH² target
    rhh2: f32, // HH² target
    water_parameters: RigidWaterParameters,
    water_admission: Option<RigidWaterAdmission>,
}

impl Settle {
    /// Create a new SETTLE constraint solver
    ///
    /// # Arguments
    /// * `context` - CUDA context
    /// * `water_oxygen_indices` - Indices of oxygen atoms in each water molecule
    ///   The hydrogens are assumed to be at indices O+1 and O+2
    /// * `n_total_atoms` - Total number of atoms in the system
    pub fn new(
        context: Arc<CudaContext>,
        water_oxygen_indices: &[usize],
        n_total_atoms: usize,
    ) -> Result<Self> {
        let stream = context.default_stream();
        Self::new_on_stream(context, stream, water_oxygen_indices, n_total_atoms)
    }

    /// Create a SETTLE solver whose transfers and launches use `stream`.
    pub fn new_on_stream(
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        water_oxygen_indices: &[usize],
        n_total_atoms: usize,
    ) -> Result<Self> {
        Self::new_on_stream_with_parameters(
            context,
            stream,
            water_oxygen_indices,
            n_total_atoms,
            RigidWaterParameters::tip3p(),
        )
    }

    /// Construct from the topology-admitted rigid-water parameter contract.
    pub fn new_on_stream_with_parameters(
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        water_oxygen_indices: &[usize],
        n_total_atoms: usize,
        water_parameters: RigidWaterParameters,
    ) -> Result<Self> {
        Self::new_on_stream_internal(
            context,
            stream,
            water_oxygen_indices,
            n_total_atoms,
            water_parameters,
            None,
        )
    }

    /// Construct a solver bound to the exact prepared water-contract digest.
    /// Admission checks model, OHH ordering, count, geometry, and masses before
    /// any PTX is loaded or device state is allocated.
    pub fn new_on_stream_with_admission(
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        water_oxygen_indices: &[usize],
        n_total_atoms: usize,
        water_parameters: RigidWaterParameters,
        admission: RigidWaterAdmission,
    ) -> Result<Self> {
        water_parameters.validate_topology_admission(&admission, water_oxygen_indices.len())?;
        Self::new_on_stream_internal(
            context,
            stream,
            water_oxygen_indices,
            n_total_atoms,
            water_parameters,
            Some(admission),
        )
    }

    fn new_on_stream_internal(
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        water_oxygen_indices: &[usize],
        n_total_atoms: usize,
        water_parameters: RigidWaterParameters,
        water_admission: Option<RigidWaterAdmission>,
    ) -> Result<Self> {
        anyhow::ensure!(
            Arc::ptr_eq(stream.context(), &context),
            "SETTLE stream belongs to a different CUDA context"
        );
        let mut occupied = std::collections::HashSet::with_capacity(water_oxygen_indices.len() * 3);
        for &o_idx in water_oxygen_indices {
            anyhow::ensure!(
                o_idx.checked_add(2).is_some_and(|h2| h2 < n_total_atoms),
                "SETTLE water O index {} does not have in-range O+1/O+2 atoms (n={})",
                o_idx,
                n_total_atoms
            );
            for atom in o_idx..=o_idx + 2 {
                anyhow::ensure!(
                    occupied.insert(atom),
                    "SETTLE water definitions overlap at atom {}",
                    atom
                );
            }
        }

        log::info!(
            "🌊 Initializing SETTLE for {} water molecules",
            water_oxygen_indices.len()
        );

        let n_waters = water_oxygen_indices.len();

        // Load the PTX produced by this exact Cargo build, not a persistent
        // mirror that may be absent or stale.
        let ptx_path = concat!(env!("OUT_DIR"), "/ptx/settle.ptx");
        let ptx_src = std::fs::read_to_string(ptx_path)
            .with_context(|| format!("Failed to read PTX file: {}", ptx_path))?;
        let ptx = Ptx::from_src(&ptx_src);
        let module = context
            .load_module(ptx)
            .with_context(|| format!("Failed to load SETTLE PTX from {}", ptx_path))?;

        // Load kernels
        let settle_kernel = module
            .load_function("settle_constraints")
            .context("Failed to load settle_constraints")?;
        let velocity_correction_kernel = module
            .load_function("settle_velocity_correction")
            .context("Failed to load settle_velocity_correction")?;

        // Build water index array: [O0, H0_1, H0_2, O1, H1_1, H1_2, ...]
        let mut water_indices = Vec::with_capacity(n_waters * 3);
        for &o_idx in water_oxygen_indices {
            water_indices.push(o_idx as i32);
            water_indices.push((o_idx + 1) as i32); // H1
            water_indices.push((o_idx + 2) as i32); // H2
        }

        // Allocate device buffers
        let d_water_indices = stream.alloc_zeros::<i32>(water_indices.len().max(1))?;
        let d_old_positions = stream.alloc_zeros::<f32>(n_total_atoms * 3)?;
        let d_failure_count = stream.alloc_zeros::<i32>(1)?;

        // Upload water indices
        let mut d_water_indices = d_water_indices;
        if !water_indices.is_empty() {
            stream.memcpy_htod(&water_indices, &mut d_water_indices)?;
        }

        // Precompute SETTLE geometry parameters
        let mass_o = water_parameters.mass_o;
        let mass_h = water_parameters.mass_h;
        let total_mass = mass_o + 2.0 * mass_h;
        let inv_total_mass = 1.0 / total_mass;

        // SETTLE canonical frame geometry:
        // - O is at (0, 0, -ra) - distance ra from COM along negative bisector
        // - H1 is at (-rc, 0, rb) - distance rb along positive bisector, rc sideways
        // - H2 is at (+rc, 0, rb) - symmetric with H1
        //
        // From center of mass constraint: mO*(-ra) + 2*mH*rb = 0
        // Therefore: rb = ra * mO / (2*mH)
        //
        // From OH distance constraint: OH² = (ra+rb)² + rc²
        // Solving: ra + rb = sqrt(OH² - rc²) = OH*cos(half_angle)
        //
        // Combined: ra*(1 + mO/(2*mH)) = OH*cos(half_angle)
        //           ra = OH*cos(half_angle) / (1 + mO/(2*mH))
        //           ra = OH*cos(half_angle) * 2*mH / (2*mH + mO)
        //           ra = OH*cos(half_angle) * 2*mH / total_mass

        let rc = water_parameters.hh_distance_angstrom / 2.0;
        let bisector = (water_parameters.oh_distance_angstrom
            * water_parameters.oh_distance_angstrom
            - rc * rc)
            .sqrt();
        let ra = bisector * 2.0 * mass_h * inv_total_mass;

        // rb from COM constraint: mO*ra = 2*mH*rb
        let rb = ra * mass_o / (2.0 * mass_h);

        // Target distances squared
        let roh2 = water_parameters.oh_distance_angstrom * water_parameters.oh_distance_angstrom;
        let rhh2 = water_parameters.hh_distance_angstrom * water_parameters.hh_distance_angstrom;

        log::info!(
            "✅ SETTLE initialized: {} waters, ra={:.4}, rb={:.4}, rc={:.4}",
            n_waters,
            ra,
            rb,
            rc
        );

        Ok(Self {
            context,
            stream,
            module,
            settle_kernel,
            velocity_correction_kernel,
            d_water_indices,
            d_old_positions,
            d_failure_count,
            n_waters,
            n_total_atoms,
            mass_o,
            mass_h,
            ra,
            rb,
            rc,
            roh2,
            rhh2,
            water_parameters,
            water_admission,
        })
    }

    /// Store old positions before integration step
    ///
    /// Call this BEFORE the velocity Verlet integration to save
    /// positions for the constraint projection.
    pub fn save_positions(&mut self, d_positions: &CudaSlice<f32>) -> Result<()> {
        anyhow::ensure!(
            d_positions.len() == self.n_total_atoms * 3,
            "SETTLE positions length mismatch: got {}, expected {}",
            d_positions.len(),
            self.n_total_atoms * 3
        );
        // Copy current positions to old positions buffer
        // This is needed because SETTLE needs the old positions to compute
        // the constraint projection direction
        self.stream
            .memcpy_dtod(d_positions, &mut self.d_old_positions)?;
        Ok(())
    }

    /// Apply SETTLE constraints to positions
    ///
    /// This projects the new positions back onto the constraint surface
    /// while conserving momentum.
    ///
    /// # Arguments
    /// * `d_positions` - Device buffer with positions AFTER integration
    ///   Will be modified in-place to satisfy constraints
    /// * `dt` - Timestep (for velocity correction if needed)
    pub fn apply(&mut self, d_positions: &mut CudaSlice<f32>, _dt: f32) -> Result<()> {
        self.reset_failure_latch()?;
        self.enqueue_apply(d_positions)?;
        let failures = self.readback_failure_latch()?;
        anyhow::ensure!(
            failures == 0,
            "SETTLE position projection failed for {} water molecule(s)",
            failures
        );
        Ok(())
    }

    /// Enqueue the position projection without resetting or reading the device
    /// failure latch.  Reset once before resident graph/chunk execution and
    /// inspect only at a synchronized acceptance boundary.
    pub fn enqueue_apply(&mut self, d_positions: &mut CudaSlice<f32>) -> Result<()> {
        if self.n_waters == 0 {
            return Ok(());
        }
        anyhow::ensure!(
            d_positions.len() == self.n_total_atoms * 3,
            "SETTLE positions length mismatch: got {}, expected {}",
            d_positions.len(),
            self.n_total_atoms * 3
        );
        let threads = 256;
        let blocks = (self.n_waters + threads - 1) / threads;

        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_waters_i32 = self.n_waters as i32;

        unsafe {
            let mut builder = self.stream.launch_builder(&self.settle_kernel);
            builder.arg(d_positions);
            builder.arg(&self.d_old_positions);
            builder.arg(&self.d_water_indices);
            builder.arg(&n_waters_i32);
            builder.arg(&self.mass_o);
            builder.arg(&self.mass_h);
            builder.arg(&self.ra);
            builder.arg(&self.rb);
            builder.arg(&self.rc);
            builder.arg(&self.roh2);
            builder.arg(&self.rhh2);
            builder.arg(&self.d_failure_count);
            builder.launch(cfg)?;
        }

        Ok(())
    }

    /// Apply RATTLE-style velocity correction to water molecules
    ///
    /// This uses proper RATTLE velocity constraints to project out only the
    /// constraint-violating velocity components (along bond directions),
    /// preserving rotational kinetic energy around the molecular axis.
    ///
    /// IMPORTANT: Call this AFTER apply() to correct velocities consistently
    /// with the position constraints.
    ///
    /// # Arguments
    /// * `d_velocities` - Device buffer with velocities to correct
    /// * `d_positions` - Device buffer with constrained positions (for bond vectors)
    pub fn apply_velocity_correction(
        &mut self,
        d_velocities: &mut CudaSlice<f32>,
        d_positions: &CudaSlice<f32>,
    ) -> Result<()> {
        self.reset_failure_latch()?;
        self.enqueue_velocity_correction(d_velocities, d_positions)?;
        let failures = self.readback_failure_latch()?;
        anyhow::ensure!(
            failures == 0,
            "SETTLE velocity projection failed for {} water molecule(s)",
            failures
        );
        Ok(())
    }

    /// Enqueue RATTLE velocity projection with no host synchronization.
    pub fn enqueue_velocity_correction(
        &mut self,
        d_velocities: &mut CudaSlice<f32>,
        d_positions: &CudaSlice<f32>,
    ) -> Result<()> {
        if self.n_waters == 0 {
            return Ok(());
        }

        anyhow::ensure!(
            d_velocities.len() == self.n_total_atoms * 3,
            "SETTLE velocities length mismatch: got {}, expected {}",
            d_velocities.len(),
            self.n_total_atoms * 3
        );
        anyhow::ensure!(
            d_positions.len() == self.n_total_atoms * 3,
            "SETTLE positions length mismatch: got {}, expected {}",
            d_positions.len(),
            self.n_total_atoms * 3
        );
        let threads = 256;
        let blocks = (self.n_waters + threads - 1) / threads;

        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_waters_i32 = self.n_waters as i32;

        unsafe {
            let mut builder = self.stream.launch_builder(&self.velocity_correction_kernel);
            builder.arg(d_velocities);
            builder.arg(d_positions);
            builder.arg(&self.d_water_indices);
            builder.arg(&n_waters_i32);
            builder.arg(&self.mass_o);
            builder.arg(&self.mass_h);
            builder.arg(&self.d_failure_count);
            builder.launch(cfg)?;
        }

        Ok(())
    }

    /// Reset the device failure latch before a resident graph or MD chunk.
    pub fn reset_failure_latch(&mut self) -> Result<()> {
        self.stream.memset_zeros(&mut self.d_failure_count)?;
        Ok(())
    }

    /// Read the device failure latch at a synchronized chunk/terminal boundary.
    pub fn readback_failure_latch(&self) -> Result<i32> {
        let mut failures = [0i32; 1];
        self.stream
            .memcpy_dtoh(&self.d_failure_count, &mut failures)?;
        Ok(failures[0])
    }

    /// Pointer-stable device failure latch for graph composition/admission.
    pub fn device_failure_latch(&self) -> &CudaSlice<i32> {
        &self.d_failure_count
    }

    /// Get number of water molecules
    pub fn n_waters(&self) -> usize {
        self.n_waters
    }

    /// Evidence-bearing capability query for runtime admission.
    pub fn capability_receipt(&self) -> SettleRuntimeCapabilityReceipt {
        SettleRuntimeCapabilityReceipt {
            schema: "prism.settle.runtime-capability.v1",
            supplied_stream_bound: true,
            stream_handle: self.stream.cu_stream() as usize,
            position_constraints: "MIYAMOTO_KOLLMAN_ANALYTIC_OH_OH_HH",
            velocity_constraints: "RATTLE_THREE_DISTANCE_MASS_WEIGHTED",
            preserves_mass_weighted_com: true,
            preserves_rigid_rotation: true,
            singular_geometry_fails_closed: true,
            graph_hot_path_dtoh_count: 0,
            graph_capture_dependency_mode:
                "CALLER_MANAGED_SINGLE_STREAM_OR_CAPTURE_LOCAL_DEPENDENCIES",
            device_failure_latch: true,
            n_waters: self.n_waters,
            topology_contract_bound: self.water_admission.is_some(),
            water_contract_sha256: self
                .water_admission
                .as_ref()
                .map(|admission| admission.contract_sha256.clone()),
            water_model_id: self.water_parameters.model_id.clone(),
            water_parameter_sha256: self.water_parameters.parameter_sha256.clone(),
            water_source_artifact_sha256: self.water_parameters.source_artifact_sha256.clone(),
        }
    }

    /// Validate this solver against the active topology/WaterContract.
    pub fn validate_water_contract(&self, model_id: &str, parameter_sha256: &str) -> Result<()> {
        self.water_parameters
            .validate_admission(model_id, parameter_sha256)
    }

    /// Revalidate the bound prepared-water contract at runtime admission.
    pub fn validate_topology_water_contract(&self, admission: &RigidWaterAdmission) -> Result<()> {
        self.water_parameters
            .validate_topology_admission(admission, self.n_waters)?;
        let bound = self
            .water_admission
            .as_ref()
            .context("SETTLE instance is not bound to a prepared water contract")?;
        anyhow::ensure!(
            bound.contract_sha256 == admission.contract_sha256,
            "prepared water contract hash mismatch: SETTLE={} topology={}",
            bound.contract_sha256,
            admission.contract_sha256
        );
        Ok(())
    }

    /// Check constraint violations (for debugging)
    ///
    /// Returns (max_oh_violation, max_hh_violation) in Angstroms
    pub fn check_constraints(&mut self, d_positions: &CudaSlice<f32>) -> Result<(f32, f32)> {
        // Download positions
        let mut positions = vec![0.0f32; self.n_total_atoms * 3];
        self.stream.memcpy_dtoh(d_positions, &mut positions)?;

        // Download water indices
        let mut water_indices = vec![0i32; self.n_waters * 3];
        self.stream
            .memcpy_dtoh(&self.d_water_indices, &mut water_indices)?;

        let mut max_oh_violation = 0.0f32;
        let mut max_hh_violation = 0.0f32;

        for w in 0..self.n_waters {
            let o = water_indices[w * 3] as usize;
            let h1 = water_indices[w * 3 + 1] as usize;
            let h2 = water_indices[w * 3 + 2] as usize;

            // Get positions
            let ox = positions[o * 3];
            let oy = positions[o * 3 + 1];
            let oz = positions[o * 3 + 2];

            let h1x = positions[h1 * 3];
            let h1y = positions[h1 * 3 + 1];
            let h1z = positions[h1 * 3 + 2];

            let h2x = positions[h2 * 3];
            let h2y = positions[h2 * 3 + 1];
            let h2z = positions[h2 * 3 + 2];

            // Compute distances
            let dx1 = h1x - ox;
            let dy1 = h1y - oy;
            let dz1 = h1z - oz;
            let oh1 = (dx1 * dx1 + dy1 * dy1 + dz1 * dz1).sqrt();

            let dx2 = h2x - ox;
            let dy2 = h2y - oy;
            let dz2 = h2z - oz;
            let oh2 = (dx2 * dx2 + dy2 * dy2 + dz2 * dz2).sqrt();

            let dx12 = h2x - h1x;
            let dy12 = h2y - h1y;
            let dz12 = h2z - h1z;
            let hh = (dx12 * dx12 + dy12 * dy12 + dz12 * dz12).sqrt();

            // Check violations
            let oh1_violation = (oh1 - self.water_parameters.oh_distance_angstrom).abs();
            let oh2_violation = (oh2 - self.water_parameters.oh_distance_angstrom).abs();
            let hh_violation = (hh - self.water_parameters.hh_distance_angstrom).abs();

            max_oh_violation = max_oh_violation.max(oh1_violation).max(oh2_violation);
            max_hh_violation = max_hh_violation.max(hh_violation);
        }

        Ok((max_oh_violation, max_hh_violation))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tip3p_geometry() {
        // HH distance should be consistent with OH and angle
        let expected_hh = 2.0 * TIP3P_OH_DISTANCE * (104.52_f32.to_radians() / 2.0).sin();
        assert!((TIP3P_HH_DISTANCE - expected_hh).abs() < 0.001);
    }

    #[test]
    fn test_settle_mass_parameters() {
        let total_mass = TIP3P_MASS_O + 2.0 * TIP3P_MASS_H;
        let inv_total_mass = 1.0 / total_mass;

        // Center of mass should be closer to O than to H
        let half_angle = 104.52_f32.to_radians() / 2.0;
        let ra = TIP3P_OH_DISTANCE * half_angle.cos() * 2.0 * TIP3P_MASS_H * inv_total_mass;

        // ra should be small positive (O is near COM)
        assert!(ra > 0.0 && ra < TIP3P_OH_DISTANCE * 0.2);
    }
}
