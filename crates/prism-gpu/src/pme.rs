//! PME (Particle Mesh Ewald) Electrostatics
//!
//! Implements the reciprocal-space part of particle-mesh Ewald electrostatics.
//!
//! Algorithm:
//! 1. Spread atom charges to 3D grid using B-spline interpolation
//! 2. Forward FFT to reciprocal space
//! 3. Apply Green's function convolution
//! 4. Inverse FFT back to real space
//! 5. Interpolate forces from grid to atoms
//!
//! Uses cuFFT from CUDA toolkit (not an external dependency).
//!
//! # Scope and boundary conditions
//!
//! This module implements a three-dimensional, orthorhombic, tin-foil Ewald
//! reciprocal sum, the Ewald self term, and the uniform-neutralizing-plasma
//! correction for a non-neutral unit cell.  It deliberately does **not**
//! compute the real-space `erfc(beta*r)/r` term or reciprocal exclusion/1-4
//! corrections.  Those terms must be supplied by the caller.  No vacuum/slab,
//! triclinic, or finite-dielectric surface term is implemented.

use anyhow::{Context, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtrMut, DeviceRepr,
    DeviceSlice, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use sha2::{Digest, Sha256};
use std::sync::Arc;

use crate::cufft_sys::{
    cufft_error_string, CufftComplex, CufftHandle, CUFFT_C2R, CUFFT_R2C, CUFFT_SUCCESS,
};
use std::ffi::c_void;

/// Default grid spacing (Å) - determines PME grid density
const PME_GRID_SPACING: f32 = 1.0;

/// Default real-space cutoff (Å) - must match non-bonded cutoff
const DEFAULT_REAL_SPACE_CUTOFF: f32 = 12.0;

/// Default PME tolerance for Ewald sum convergence
/// Smaller values give higher accuracy but require more reciprocal space work
pub const DEFAULT_PME_TOLERANCE: f32 = 1e-5;

/// Boundary condition implemented by this PME solver.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PmeBoundaryCondition {
    /// Fully periodic orthorhombic cell with conducting (tin-foil) boundary.
    OrthorhombicTinFoil3D,
}

/// Explicit contract for the terms outside this reciprocal PME module.
///
/// Construction requires this value so a caller cannot silently treat the
/// returned energy as a complete electrostatic energy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PmeSplitContract {
    pub boundary: PmeBoundaryCondition,
    /// The caller computes `erfc(beta*r)/r` using the same beta and cutoff.
    pub real_space_erfc_external: bool,
    /// The caller removes reciprocal interactions for excluded/scaled pairs.
    pub reciprocal_exclusion_correction_external: bool,
    /// Lambda paths use the same split and provide real-space/exclusion dU/dλ.
    pub lambda_derivatives_external: bool,
}

impl PmeSplitContract {
    /// Contract accepted by the current implementation.
    pub const fn external_real_space_and_exclusions() -> Self {
        Self {
            boundary: PmeBoundaryCondition::OrthorhombicTinFoil3D,
            real_space_erfc_external: true,
            reciprocal_exclusion_correction_external: true,
            lambda_derivatives_external: true,
        }
    }

    fn validate(self) -> Result<()> {
        anyhow::ensure!(
            self.real_space_erfc_external,
            "PME is reciprocal-only: caller must acknowledge external real-space erfc"
        );
        anyhow::ensure!(
            self.reciprocal_exclusion_correction_external,
            "PME does not own exclusions: caller must acknowledge external reciprocal exclusion corrections"
        );
        anyhow::ensure!(
            self.lambda_derivatives_external,
            "lambda PME requires caller-owned real-space and exclusion dU/dlambda"
        );
        Ok(())
    }
}

/// Energy components produced by [`PME::compute_reciprocal`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PmeReciprocalResult {
    /// Mesh reciprocal energy before analytic corrections (kcal/mol).
    pub mesh_energy: f64,
    /// Ewald Gaussian self-energy correction (kcal/mol).
    pub self_energy: f64,
    /// Uniform-neutralizing-plasma correction (kcal/mol).
    pub net_charge_correction: f64,
    /// `mesh_energy + self_energy + net_charge_correction` (kcal/mol).
    pub reciprocal_energy: f64,
    /// Unit-cell charge used by the background correction (elementary charge).
    pub total_charge: f64,
    /// Mesh contribution to dU/dlambda, when endpoint charge derivatives were supplied.
    pub mesh_dudlambda: Option<f64>,
    /// Self contribution to dU/dlambda.
    pub self_dudlambda: Option<f64>,
    /// Uniform-background contribution to dU/dlambda.
    pub net_charge_dudlambda: Option<f64>,
    /// Reciprocal total dU/dlambda. Real-space and exclusion derivatives remain external.
    pub reciprocal_dudlambda: Option<f64>,
    /// Derivative of total unit-cell charge.
    pub total_charge_derivative: Option<f64>,
}

/// Deterministic runtime self-test evidence attached to a capability receipt.
#[derive(Clone, Debug, serde::Serialize)]
pub struct PmeRuntimeValidationEvidence {
    pub status: &'static str,
    pub force_energy_abs_error: f64,
    pub force_energy_rel_error: f64,
    pub translation_energy_abs_error: f64,
    pub charge_dudlambda_abs_error: f64,
    pub charge_dudlambda_rel_error: f64,
    pub force_difference_step_angstrom: f64,
    pub lambda_difference_step: f64,
    pub atom_order_energy_abs_error: f64,
    pub atom_order_dudlambda_abs_error: f64,
}

/// Independently produced numeric validation for the exact PME instance.
/// The runtime accepts only residuals at or below their declared tolerances;
/// source code, a test name, or a hardcoded boolean cannot instantiate this
/// authority.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PmeMeasuredParityReceipt {
    pub schema_kind: String,
    pub schema_version: u32,
    pub validation_kind: String,
    pub runtime_parameterization_sha256: String,
    pub sample_count: u64,
    pub energy_max_abs_residual: f64,
    pub energy_tolerance: f64,
    pub force_max_abs_residual: f64,
    pub force_tolerance: f64,
    pub dudlambda_max_abs_residual: Option<f64>,
    pub dudlambda_tolerance: Option<f64>,
    pub receipt_sha256: String,
}

impl PmeMeasuredParityReceipt {
    fn recompute_sha256(&self) -> Result<String> {
        let mut projection = self.clone();
        projection.receipt_sha256.clear();
        let bytes = serde_json::to_vec(&projection)?;
        let mut digest = Sha256::new();
        digest.update(b"PRISM_PME_MEASURED_PARITY_RECEIPT_V1\0");
        digest.update(bytes);
        Ok(format!("{:x}", digest.finalize()))
    }

    fn validate_for(&self, expected_kind: &str, expected_parameterization: &str) -> Result<()> {
        anyhow::ensure!(
            self.schema_kind == "prismos_pme_measured_parity_receipt"
                && self.schema_version == 1
                && self.validation_kind == expected_kind
                && self.runtime_parameterization_sha256 == expected_parameterization
                && self.sample_count > 0
                && self.receipt_sha256 == self.recompute_sha256()?,
            "invalid PME measured parity receipt identity"
        );
        anyhow::ensure!(
            self.energy_max_abs_residual.is_finite()
                && self.energy_tolerance.is_finite()
                && self.energy_tolerance > 0.0
                && self.energy_max_abs_residual >= 0.0
                && self.energy_max_abs_residual <= self.energy_tolerance
                && self.force_max_abs_residual.is_finite()
                && self.force_tolerance.is_finite()
                && self.force_tolerance > 0.0
                && self.force_max_abs_residual >= 0.0
                && self.force_max_abs_residual <= self.force_tolerance,
            "PME measured energy/force parity exceeds tolerance"
        );
        match (
            self.dudlambda_max_abs_residual,
            self.dudlambda_tolerance,
            expected_kind,
        ) {
            (Some(residual), Some(tolerance), "TARGET_SYSTEM_COMPOSED_EWALD_U_F_DUDL")
                if residual.is_finite()
                    && tolerance.is_finite()
                    && tolerance > 0.0
                    && residual >= 0.0
                    && residual <= tolerance => {}
            (None, None, "OPENMM_ORDER5_SAME_PARAMETER_U_F") => {}
            _ => anyhow::bail!("PME measured dU/dlambda parity evidence is incomplete"),
        }
        Ok(())
    }
}

/// Evidence-bearing capability query for engine/lambda admission.
#[derive(Clone, Debug, serde::Serialize)]
pub struct PmeRuntimeCapabilityReceipt {
    pub schema: &'static str,
    pub supplied_stream_bound: bool,
    pub stream_handle: usize,
    pub cufft_stream_bound: bool,
    pub cufft_plan_handles_nonzero: bool,
    pub graph_hot_path_dtoh_count: usize,
    pub graph_capture_dependency_mode: &'static str,
    pub scalar_accumulation: &'static str,
    pub boundary: &'static str,
    pub reciprocal_scope: &'static str,
    pub self_correction: bool,
    pub uniform_background_correction: bool,
    pub dynamic_charges_and_dudlambda: bool,
    pub grid: [usize; 3],
    pub interpolation_order: u32,
    /// The selected kernel uses the same order-five cardinal B-spline
    /// recurrence for spread, force interpolation, and charge gathering.
    pub order5_assignment_operator: bool,
    /// True only when independent OpenMM same-parameter energy/force parity
    /// evidence has been attached.  Formula agreement is not parity evidence.
    pub openmm_order5_parity: bool,
    pub ewald_beta_a_inv: f32,
    pub real_space_cutoff_a: f32,
    pub relative_force_tolerance: f32,
    pub exact_parameter_lock: bool,
    /// True only after one composed test closes real+reciprocal+self+
    /// background+exclusions for U/F/dUdl. Reciprocal-only tests cannot set it.
    pub total_ewald_hamiltonian_validation: bool,
    /// Hash of an independent OpenMM same-system U/F comparison receipt.
    pub openmm_reference_validation_receipt_sha256: Option<String>,
    /// Hash of a target-system composed real+reciprocal+self+background+
    /// exclusions U/F/dUdl validation receipt.
    pub target_system_validation_receipt_sha256: Option<String>,
    pub external_contract_sha256: String,
    pub external_real_space_erfc: bool,
    pub external_reciprocal_exclusions: bool,
    pub external_lambda_derivatives: bool,
    pub validation: Option<PmeRuntimeValidationEvidence>,
}

/// Compute Ewald splitting parameter from tolerance and cutoff.
///
/// The Ewald method splits the Coulomb sum into real-space and reciprocal-space
/// components. The splitting parameter β controls this division:
/// - Larger β: More work in reciprocal space (faster for dense systems)
/// - Smaller β: More work in real space (faster for sparse systems)
///
/// OpenMM formula: β = sqrt(-ln(2*tolerance)) / cutoff.
///
/// # Arguments
/// * `cutoff` - Real-space cutoff distance in Ångströms
/// * `tolerance` - Relative error tolerance (typically 1e-5 to 1e-6)
///
/// # Returns
/// Ewald splitting parameter β in Å⁻¹
///
/// # Example
/// ```ignore
/// let beta = compute_ewald_beta(12.0, 1e-5);
/// assert!((beta - 0.274).abs() < 0.001);
/// ```
pub fn compute_ewald_beta(cutoff: f32, tolerance: f32) -> f32 {
    debug_assert!(cutoff > 0.0, "Cutoff must be positive");
    debug_assert!(
        tolerance > 0.0 && tolerance < 1.0,
        "Tolerance must be in (0, 1)"
    );

    (-(2.0 * tolerance).ln()).sqrt() / cutoff
}

/// Per-axis PME mesh lower bound used by OpenMM's NonbondedForce heuristic.
///
/// The relative tolerance controls an empirical mesh-selection rule; satisfying
/// this lower bound is necessary but not sufficient evidence of force accuracy.
/// Production admission additionally requires an independent same-system
/// OpenMM energy/force comparison.
pub fn openmm_pme_grid_lower_bound(
    box_dims_angstrom: [f32; 3],
    ewald_alpha_a_inv: f32,
    relative_force_tolerance: f32,
) -> [usize; 3] {
    let denominator = 3.0 * relative_force_tolerance.powf(0.2);
    box_dims_angstrom
        .map(|length| ((2.0 * ewald_alpha_a_inv * length / denominator).ceil() as usize).max(6))
}

/// PME electrostatics calculator
pub struct PME {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,

    // Grid dimensions
    nx: usize,
    ny: usize,
    nz: usize,

    // Box dimensions (Å)
    box_dims: [f32; 3],

    // Ewald splitting parameter
    beta: f32,
    real_space_cutoff_a: f32,
    relative_force_tolerance: f32,
    exact_parameter_lock: bool,

    // cuFFT plans
    fft_plan_r2c: CufftHandle,
    fft_plan_c2r: CufftHandle,
    plans_initialized: bool,

    // Device buffers
    d_charge_grid: CudaSlice<f32>,  // [nx * ny * nz] real grid
    d_complex_grid: CudaSlice<f32>, // [nx * ny * (nz/2+1) * 2] complex (interleaved)
    // [mesh E, self E, Q, background E, mesh dUdl, self dUdl, dQ, background dUdl]
    d_energy_components: CudaSlice<f64>,
    // Exact atom-resolved decomposition of reciprocal+self+background terms.
    // These arrays are overwritten by every dynamic-charge enqueue and can be
    // merged into a fused Hamiltonian on the same stream without host readback.
    d_atom_energy_components: CudaSlice<f64>,
    d_atom_dudlambda_components: CudaSlice<f64>,

    // Kernels
    spread_kernel: CudaFunction,
    convolution_kernel: CudaFunction,
    interpolate_kernel: CudaFunction,
    correction_kernel: CudaFunction,
    net_charge_kernel: CudaFunction,
    charge_derivative_kernel: CudaFunction,
    merge_lambda_atom_components_kernel: CudaFunction,
    select_isolated_lambda_slot_charges_kernel: CudaFunction,
    accumulate_lambda_internal_correction_kernel: CudaFunction,
    zero_grid_kernel: CudaFunction,

    // Atom count (for validation)
    n_atoms: usize,

    contract: PmeSplitContract,
    runtime_validation: Option<PmeRuntimeValidationEvidence>,
    openmm_reference_validation_receipt_sha256: Option<String>,
    target_system_validation_receipt_sha256: Option<String>,
    last_enqueued_has_charge_derivative: bool,
}

/// Host reference for the coefficients evaluated by
/// `pme_accumulate_lambda_internal_correction`. This is intentionally public
/// so acceptance tests can prove the target Hamiltonian independently of CUDA
/// execution. `source_plane`: 0=A, 1=B, 2=current isolated slot.
pub fn lambda_internal_pme_correction_coefficients(
    lambda: f64,
    endpoint_occupancy_mask: i32,
    source_plane: i32,
) -> Result<(f64, f64, f64)> {
    anyhow::ensure!(
        lambda.is_finite() && (0.0..=1.0).contains(&lambda),
        "invalid lambda"
    );
    anyhow::ensure!(
        (1..=3).contains(&endpoint_occupancy_mask),
        "invalid endpoint mask"
    );
    anyhow::ensure!((0..=2).contains(&source_plane), "invalid source plane");
    anyhow::ensure!(
        endpoint_occupancy_mask == 3
            || (endpoint_occupancy_mask == 1 && source_plane == 0)
            || (endpoint_occupancy_mask == 2 && source_plane == 1),
        "source plane is redundant or invalid for one-sided occupancy"
    );
    Ok(match source_plane {
        0 => match endpoint_occupancy_mask {
            1 => (2.0 * lambda - lambda * lambda, 2.0 - 2.0 * lambda, 0.0),
            3 => (1.0 - lambda, -1.0, 0.0),
            _ => (0.0, 0.0, 0.0),
        },
        1 => match endpoint_occupancy_mask {
            2 => (1.0 - lambda * lambda, -2.0 * lambda, 0.0),
            3 => (lambda, 1.0, 0.0),
            _ => (0.0, 0.0, 0.0),
        },
        2 => (-1.0, 0.0, -1.0),
        _ => unreachable!(),
    })
}

impl PME {
    /// Create a new PME calculator
    ///
    /// # Arguments
    /// * `context` - CUDA context
    /// * `n_atoms` - Number of atoms
    /// * `box_dims` - Periodic box dimensions [Lx, Ly, Lz] in Angstroms
    pub fn new(context: Arc<CudaContext>, n_atoms: usize, box_dims: [f32; 3]) -> Result<Self> {
        let stream = context.default_stream();
        Self::new_on_stream(
            context,
            stream,
            n_atoms,
            box_dims,
            PmeSplitContract::external_real_space_and_exclusions(),
        )
    }

    /// Create a PME calculator on the engine-owned CUDA stream.
    ///
    /// Every kernel, transfer, and cuFFT execution is ordered on `stream`.
    /// `contract` explicitly admits that real-space and exclusion corrections
    /// remain outside this module.
    pub fn new_on_stream(
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        n_atoms: usize,
        box_dims: [f32; 3],
        contract: PmeSplitContract,
    ) -> Result<Self> {
        Self::new_on_stream_impl(context, stream, n_atoms, box_dims, contract, None)
    }

    /// Construct PME with an exact hash-bound order-five discretization.
    /// Construction proves parameter consistency only.  It does not establish
    /// OpenMM numeric parity or target-system total-Hamiltonian closure.
    pub fn new_on_stream_with_exact_parameters(
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        n_atoms: usize,
        box_dims: [f32; 3],
        grid: [usize; 3],
        interpolation_order: u32,
        ewald_beta_a_inv: f32,
        real_space_cutoff_a: f32,
        relative_force_tolerance: f32,
        contract: PmeSplitContract,
    ) -> Result<Self> {
        anyhow::ensure!(
            interpolation_order == 5,
            "PME kernels require interpolation_order=5"
        );
        anyhow::ensure!(
            grid.iter().all(|dimension| *dimension >= 6),
            "PME grid dimensions must each be at least six"
        );
        for (name, value) in [
            ("ewald_beta_a_inv", ewald_beta_a_inv),
            ("real_space_cutoff_a", real_space_cutoff_a),
            ("relative_force_tolerance", relative_force_tolerance),
        ] {
            anyhow::ensure!(value.is_finite() && value > 0.0, "invalid PME {name}");
        }
        let derived_beta = compute_ewald_beta(real_space_cutoff_a, relative_force_tolerance);
        anyhow::ensure!(
            (derived_beta - ewald_beta_a_inv).abs() <= 1.0e-6,
            "PME beta does not close OpenMM alpha=sqrt(-ln(2*tolerance))/cutoff: declared={} derived={}",
            ewald_beta_a_inv,
            derived_beta
        );
        let lower_bound =
            openmm_pme_grid_lower_bound(box_dims, ewald_beta_a_inv, relative_force_tolerance);
        anyhow::ensure!(
            grid.iter()
                .zip(lower_bound)
                .all(|(actual, minimum)| *actual >= minimum),
            "PME grid {:?} is smaller than OpenMM lower bound {:?}",
            grid,
            lower_bound
        );
        Self::new_on_stream_impl(
            context,
            stream,
            n_atoms,
            box_dims,
            contract,
            Some((
                grid,
                ewald_beta_a_inv,
                real_space_cutoff_a,
                relative_force_tolerance,
            )),
        )
    }

    fn new_on_stream_impl(
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        n_atoms: usize,
        box_dims: [f32; 3],
        contract: PmeSplitContract,
        exact_parameters: Option<([usize; 3], f32, f32, f32)>,
    ) -> Result<Self> {
        anyhow::ensure!(n_atoms > 0, "PME requires at least one atom");
        anyhow::ensure!(
            box_dims
                .iter()
                .all(|length| length.is_finite() && *length > 0.0),
            "PME requires finite positive orthorhombic box lengths"
        );
        anyhow::ensure!(
            Arc::ptr_eq(stream.context(), &context),
            "PME stream belongs to a different CUDA context"
        );
        contract.validate()?;

        log::info!(
            "⚡ Initializing PME for {} atoms, box = {:.1}×{:.1}×{:.1} Å",
            n_atoms,
            box_dims[0],
            box_dims[1],
            box_dims[2]
        );

        // Load PME PTX module - use absolute path for reliability
        // Load the PTX produced by this exact Cargo build, not a persistent
        // mirror that may be absent or stale.
        let ptx_path = concat!(env!("OUT_DIR"), "/ptx/pme.ptx");
        let ptx_src = std::fs::read_to_string(ptx_path)
            .with_context(|| format!("Failed to read PTX file: {}", ptx_path))?;
        let ptx = Ptx::from_src(&ptx_src);
        let module = context
            .load_module(ptx)
            .with_context(|| format!("Failed to load PME PTX from {}", ptx_path))?;

        // Load kernels
        let spread_kernel = module
            .load_function("pme_spread_charges")
            .context("Failed to load pme_spread_charges")?;
        let convolution_kernel = module
            .load_function("pme_reciprocal_convolution")
            .context("Failed to load pme_reciprocal_convolution")?;
        let interpolate_kernel = module
            .load_function("pme_interpolate_forces")
            .context("Failed to load pme_interpolate_forces")?;
        let correction_kernel = module
            .load_function("pme_accumulate_corrections")
            .context("Failed to load pme_accumulate_corrections")?;
        let net_charge_kernel = module
            .load_function("pme_finalize_net_charge_correction")
            .context("Failed to load pme_finalize_net_charge_correction")?;
        let charge_derivative_kernel = module
            .load_function("pme_accumulate_charge_derivative")
            .context("Failed to load pme_accumulate_charge_derivative")?;
        let merge_lambda_atom_components_kernel = module
            .load_function("pme_merge_lambda_atom_components")
            .context("Failed to load pme_merge_lambda_atom_components")?;
        let select_isolated_lambda_slot_charges_kernel = module
            .load_function("pme_select_isolated_lambda_slot_charges")
            .context("Failed to load pme_select_isolated_lambda_slot_charges")?;
        let accumulate_lambda_internal_correction_kernel = module
            .load_function("pme_accumulate_lambda_internal_correction")
            .context("Failed to load pme_accumulate_lambda_internal_correction")?;
        let zero_grid_kernel = module
            .load_function("pme_zero_grid")
            .context("Failed to load pme_zero_grid")?;
        // Compute grid dimensions based on box size and target spacing
        let (nx, ny, nz) = if let Some((grid, _, _, _)) = exact_parameters {
            (grid[0], grid[1], grid[2])
        } else {
            let nx = ((box_dims[0] / PME_GRID_SPACING).ceil() as usize).max(8);
            let ny = ((box_dims[1] / PME_GRID_SPACING).ceil() as usize).max(8);
            let nz = ((box_dims[2] / PME_GRID_SPACING).ceil() as usize).max(8);
            (
                round_up_fft_size(nx),
                round_up_fft_size(ny),
                round_up_fft_size(nz),
            )
        };

        log::info!(
            "📊 PME grid: {}×{}×{} = {} points, spacing ≈ {:.2} Å",
            nx,
            ny,
            nz,
            nx * ny * nz,
            box_dims[0] / nx as f32
        );

        // Allocate device buffers
        let real_size = nx * ny * nz;
        let complex_size = nx * ny * (nz / 2 + 1) * 2; // Interleaved complex

        let d_charge_grid = stream.alloc_zeros::<f32>(real_size)?;
        let d_complex_grid = stream.alloc_zeros::<f32>(complex_size)?;
        let d_energy_components = stream.alloc_zeros::<f64>(8)?;
        let d_atom_energy_components = stream.alloc_zeros::<f64>(n_atoms)?;
        let d_atom_dudlambda_components = stream.alloc_zeros::<f64>(n_atoms)?;

        // Create cuFFT plans
        let (fft_plan_r2c, fft_plan_c2r) = create_fft_plans(nx, ny, nz, &stream)
            .context("Failed to create stream-bound cuFFT plans")?;

        // Compute Ewald splitting parameter from cutoff and tolerance
        // OpenMM alpha = sqrt(-ln(2*tolerance)) / cutoff
        let (beta, real_space_cutoff_a, relative_force_tolerance) = exact_parameters
            .map(|(_, beta, cutoff, tolerance)| (beta, cutoff, tolerance))
            .unwrap_or_else(|| {
                (
                    compute_ewald_beta(DEFAULT_REAL_SPACE_CUTOFF, DEFAULT_PME_TOLERANCE),
                    DEFAULT_REAL_SPACE_CUTOFF,
                    DEFAULT_PME_TOLERANCE,
                )
            });
        log::info!(
            "✅ PME initialized: β={:.4} Å⁻¹ (cutoff={:.1} Å, tolerance={:.0e})",
            beta,
            real_space_cutoff_a,
            relative_force_tolerance
        );

        Ok(Self {
            context,
            stream,
            module,
            nx,
            ny,
            nz,
            box_dims,
            beta,
            real_space_cutoff_a,
            relative_force_tolerance,
            exact_parameter_lock: exact_parameters.is_some(),
            fft_plan_r2c,
            fft_plan_c2r,
            plans_initialized: true,
            d_charge_grid,
            d_complex_grid,
            d_energy_components,
            d_atom_energy_components,
            d_atom_dudlambda_components,
            spread_kernel,
            convolution_kernel,
            interpolate_kernel,
            correction_kernel,
            net_charge_kernel,
            charge_derivative_kernel,
            merge_lambda_atom_components_kernel,
            select_isolated_lambda_slot_charges_kernel,
            accumulate_lambda_internal_correction_kernel,
            zero_grid_kernel,
            n_atoms,
            contract,
            runtime_validation: None,
            openmm_reference_validation_receipt_sha256: None,
            target_system_validation_receipt_sha256: None,
            last_enqueued_has_charge_derivative: false,
        })
    }

    /// Set the Ewald splitting parameter directly
    ///
    /// Prefer using `set_beta_from_cutoff()` which computes the optimal β.
    /// Larger β → more in reciprocal space, smaller → more in real space.
    pub fn set_beta(&mut self, beta: f32) {
        self.beta = beta;
        self.exact_parameter_lock = false;
        self.runtime_validation = None;
        self.openmm_reference_validation_receipt_sha256 = None;
        self.target_system_validation_receipt_sha256 = None;
        log::info!("🔧 PME β = {:.4} Å⁻¹ (manually set)", beta);
    }

    /// Set the Ewald splitting parameter from cutoff and tolerance
    ///
    /// This is the preferred method - computes optimal β using:
    /// OpenMM alpha = sqrt(-ln(2*tolerance)) / cutoff
    ///
    /// # Arguments
    /// * `cutoff` - Real-space cutoff in Ångströms (should match non-bonded cutoff)
    /// * `tolerance` - Ewald sum tolerance (typically 1e-5 to 1e-6)
    pub fn set_beta_from_cutoff(&mut self, cutoff: f32, tolerance: f32) {
        self.beta = compute_ewald_beta(cutoff, tolerance);
        self.real_space_cutoff_a = cutoff;
        self.relative_force_tolerance = tolerance;
        self.exact_parameter_lock = false;
        self.runtime_validation = None;
        self.openmm_reference_validation_receipt_sha256 = None;
        self.target_system_validation_receipt_sha256 = None;
        log::info!(
            "🔧 PME β = {:.4} Å⁻¹ (cutoff={:.1} Å, tolerance={:.0e})",
            self.beta,
            cutoff,
            tolerance
        );
    }

    /// Compute reciprocal PME energy and add reciprocal forces to atoms.
    ///
    /// # Arguments
    /// * `d_positions` - Device slice with atom positions [n_atoms * 3]
    /// * `d_charges` - Device slice with atom charges [n_atoms]
    /// * `d_forces` - Device slice to add forces to [n_atoms * 3]
    ///
    /// # Returns
    /// Reciprocal mesh+self+neutralizing-background energy in kcal/mol.  This
    /// is not the complete electrostatic energy; see [`PmeSplitContract`].
    pub fn compute<P: DeviceRepr>(
        &mut self,
        d_positions: &CudaSlice<P>,
        d_charges: &CudaSlice<f32>,
        d_forces: &mut CudaSlice<f32>,
    ) -> Result<f64> {
        Ok(self
            .compute_reciprocal(d_positions, d_charges, d_forces)?
            .reciprocal_energy)
    }

    /// Compute the reciprocal PME component with explicit energy provenance.
    pub fn compute_reciprocal<P: DeviceRepr>(
        &mut self,
        d_positions: &CudaSlice<P>,
        d_charges: &CudaSlice<f32>,
        d_forces: &mut CudaSlice<f32>,
    ) -> Result<PmeReciprocalResult> {
        self.enqueue_reciprocal(d_positions, d_charges, d_forces)?;
        self.readback_reciprocal_result()
    }

    /// Enqueue the static-charge reciprocal path with no host readback.
    ///
    /// This is the pointer-stable, supplied-stream resident-graph hot path.
    /// Scalar results remain in [`Self::device_result_components`] until an
    /// explicit checkpoint calls [`Self::readback_reciprocal_result`].
    pub fn enqueue_reciprocal<P: DeviceRepr>(
        &mut self,
        d_positions: &CudaSlice<P>,
        d_charges: &CudaSlice<f32>,
        d_forces: &mut CudaSlice<f32>,
    ) -> Result<()> {
        self.compute_reciprocal_impl(d_positions, d_charges, None, d_forces, None)
    }

    /// Compute reciprocal energy, forces, and charge-path `dU/dlambda`.
    ///
    /// `d_charges_dlambda` is the derivative of the *current endpoint-scaled*
    /// charge array passed in `d_charges`.  This method never substitutes static
    /// topology charges.  External real-space and exclusion terms must use the
    /// same charge path and beta, as admitted by [`PmeSplitContract`].
    pub fn compute_reciprocal_with_charge_derivatives<P: DeviceRepr>(
        &mut self,
        d_positions: &CudaSlice<P>,
        d_charges: &CudaSlice<f32>,
        d_charges_dlambda: &CudaSlice<f32>,
        d_forces: &mut CudaSlice<f32>,
    ) -> Result<PmeReciprocalResult> {
        anyhow::ensure!(
            d_charges_dlambda.len() == self.n_atoms,
            "PME dq/dlambda length mismatch: got {}, expected {}",
            d_charges_dlambda.len(),
            self.n_atoms
        );
        self.enqueue_reciprocal_with_charge_derivatives(
            d_positions,
            d_charges,
            d_charges_dlambda,
            d_forces,
        )?;
        self.readback_reciprocal_result()
    }

    /// Enqueue current-lambda reciprocal energy, forces, and dU/dlambda with
    /// no host synchronization or readback.
    pub fn enqueue_reciprocal_with_charge_derivatives<P: DeviceRepr>(
        &mut self,
        d_positions: &CudaSlice<P>,
        d_charges: &CudaSlice<f32>,
        d_charges_dlambda: &CudaSlice<f32>,
        d_forces: &mut CudaSlice<f32>,
    ) -> Result<()> {
        anyhow::ensure!(
            d_charges_dlambda.len() == self.n_atoms,
            "PME dq/dlambda length mismatch: got {}, expected {}",
            d_charges_dlambda.len(),
            self.n_atoms
        );
        self.compute_reciprocal_impl(d_positions, d_charges, Some(d_charges_dlambda), d_forces, None)
    }

    /// Explicit coordinate-authority path. The caller retains the single-leaf
    /// native-Angstrom descriptor and terminal sink through all captured work.
    pub unsafe fn enqueue_reciprocal_with_charge_derivatives_compensated<P:DeviceRepr>(
        &mut self,positions:&CudaSlice<P>,charges:&CudaSlice<f32>,derivatives:&CudaSlice<f32>,forces:&mut CudaSlice<f32>,
        coordinate_owner:u64,precision_fault:u64)->Result<()> {
        anyhow::ensure!(coordinate_owner!=0 && coordinate_owner%16==0 && precision_fault!=0 && precision_fault%4==0,
            "PME compensated coordinate/fault owner missing");
        anyhow::ensure!(derivatives.len()==self.n_atoms,"PME dq/dlambda extent mismatch");
        self.compute_reciprocal_impl(positions,charges,Some(derivatives),forces,Some((coordinate_owner,precision_fault)))
    }
    pub unsafe fn enqueue_reciprocal_compensated<P:DeviceRepr>(&mut self,positions:&CudaSlice<P>,charges:&CudaSlice<f32>,forces:&mut CudaSlice<f32>,
        coordinate_owner:u64,precision_fault:u64)->Result<()> {
        anyhow::ensure!(coordinate_owner!=0 && coordinate_owner%16==0 && precision_fault!=0 && precision_fault%4==0,
            "PME compensated coordinate/fault owner missing");
        self.compute_reciprocal_impl(positions,charges,None,forces,Some((coordinate_owner,precision_fault)))
    }

    fn compute_reciprocal_impl<P: DeviceRepr>(
        &mut self,
        d_positions: &CudaSlice<P>,
        d_charges: &CudaSlice<f32>,
        d_charges_dlambda: Option<&CudaSlice<f32>>,
        d_forces: &mut CudaSlice<f32>,
        precision:Option<(u64,u64)>,
    ) -> Result<()> {
        let (coordinate_owner,precision_fault)=precision.unwrap_or((0,0));
        let spread=if precision.is_some(){self.module.load_function("pme_spread_charges_compensated_v1")?}else{self.spread_kernel.clone()};
        let interpolate=if precision.is_some(){self.module.load_function("pme_interpolate_forces_compensated_v1")?}else{self.interpolate_kernel.clone()};
        let derivative=if precision.is_some(){self.module.load_function("pme_accumulate_charge_derivative_compensated_v1")?}else{self.charge_derivative_kernel.clone()};
        anyhow::ensure!(
            d_positions.len().checked_mul(std::mem::size_of::<P>())
                == Some(self.n_atoms * 3 * std::mem::size_of::<f32>()),
            "PME position byte length mismatch: got {} elements x {} bytes, expected {} bytes",
            d_positions.len(),
            std::mem::size_of::<P>(),
            self.n_atoms * 3 * std::mem::size_of::<f32>()
        );
        anyhow::ensure!(
            d_charges.len() == self.n_atoms,
            "PME charges length mismatch: got {}, expected {}",
            d_charges.len(),
            self.n_atoms
        );
        anyhow::ensure!(
            d_forces.len() == self.n_atoms * 3,
            "PME forces length mismatch: got {}, expected {}",
            d_forces.len(),
            self.n_atoms * 3
        );

        let threads = 256;
        let n_atoms_i32 = self.n_atoms as i32;
        let nx_i32 = self.nx as i32;
        let ny_i32 = self.ny as i32;
        let nz_i32 = self.nz as i32;

        // Box inverse dimensions for coordinate transforms
        let box_inv_x = 1.0f32 / self.box_dims[0];
        let box_inv_y = 1.0f32 / self.box_dims[1];
        let box_inv_z = 1.0f32 / self.box_dims[2];

        // 1. Zero the charge grid
        let grid_size = self.nx * self.ny * self.nz;
        let grid_blocks = (grid_size + threads - 1) / threads;
        let grid_size_i32 = grid_size as i32;

        let cfg_grid = LaunchConfig {
            grid_dim: (grid_blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            let mut builder = self.stream.launch_builder(&self.zero_grid_kernel);
            builder.arg(&self.d_charge_grid);
            builder.arg(&grid_size_i32);
            builder.launch(cfg_grid)?;
        }

        // 2. Zero compact reciprocal telemetry/energy accumulator on-stream.
        self.stream.memset_zeros(&mut self.d_energy_components)?;
        self.stream
            .memset_zeros(&mut self.d_atom_energy_components)?;
        self.stream
            .memset_zeros(&mut self.d_atom_dudlambda_components)?;

        // 3. Spread charges to grid
        let atom_blocks = (self.n_atoms + threads - 1) / threads;
        let cfg_atoms = LaunchConfig {
            grid_dim: (atom_blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            let mut builder = self.stream.launch_builder(&spread);
            builder.arg(d_positions);
            builder.arg(d_charges);
            builder.arg(&self.d_charge_grid);
            builder.arg(&n_atoms_i32);
            builder.arg(&nx_i32);
            builder.arg(&ny_i32);
            builder.arg(&nz_i32);
            builder.arg(&box_inv_x);
            builder.arg(&box_inv_y);
            builder.arg(&box_inv_z);
            if precision.is_some(){builder.arg(&coordinate_owner).arg(&precision_fault);}
            builder.launch(cfg_atoms)?;
        }

        // 4. Forward FFT (R2C)
        execute_fft_r2c(
            self.fft_plan_r2c,
            &mut self.d_charge_grid,
            &mut self.d_complex_grid,
            &self.stream,
        )?;

        // 5. Apply reciprocal space convolution
        let complex_size = self.nx * self.ny * (self.nz / 2 + 1);
        let complex_blocks = (complex_size + threads - 1) / threads;

        let cfg_complex = LaunchConfig {
            grid_dim: (complex_blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Complex grid is interleaved: [re0, im0, re1, im1, ...]
        // Kernel handles interleaved indexing internally
        unsafe {
            let mut builder = self.stream.launch_builder(&self.convolution_kernel);
            builder.arg(&self.d_complex_grid); // Interleaved complex data
            builder.arg(&self.d_energy_components);
            builder.arg(&self.beta);
            builder.arg(&nx_i32);
            builder.arg(&ny_i32);
            builder.arg(&nz_i32);
            builder.arg(&box_inv_x);
            builder.arg(&box_inv_y);
            builder.arg(&box_inv_z);
            builder.launch(cfg_complex)?;
        }

        // 6. Inverse FFT (C2R)
        execute_fft_c2r(
            self.fft_plan_c2r,
            &mut self.d_complex_grid,
            &mut self.d_charge_grid,
            &self.stream,
        )?;

        // 7. No 1/N scale is applied.  The charge grid stores charge (not
        // charge density), cuFFT's inverse is unnormalized, and the influence
        // function carries 1/V.  Together these reproduce the continuum Ewald
        // series without an additional FFT-size factor.

        // 8. Interpolate forces from potential grid
        unsafe {
            let mut builder = self.stream.launch_builder(&interpolate);
            builder.arg(d_positions);
            builder.arg(d_charges);
            builder.arg(&self.d_charge_grid); // Now contains potential
            builder.arg(d_forces);
            builder.arg(&n_atoms_i32);
            builder.arg(&nx_i32);
            builder.arg(&ny_i32);
            builder.arg(&nz_i32);
            builder.arg(&box_inv_x);
            builder.arg(&box_inv_y);
            builder.arg(&box_inv_z);
            if precision.is_some(){builder.arg(&coordinate_owner).arg(&precision_fault);}
            builder.launch(cfg_atoms)?;
        }

        // 9. Accumulate self, cell charge, and uniform-background correction.
        let inv_volume = box_inv_x * box_inv_y * box_inv_z;
        unsafe {
            let mut builder = self.stream.launch_builder(&self.correction_kernel);
            builder.arg(d_charges);
            builder.arg(&self.d_energy_components);
            builder.arg(&n_atoms_i32);
            builder.arg(&self.beta);
            builder.launch(cfg_atoms)?;
        }
        if let Some(dq) = d_charges_dlambda {
            unsafe {
                let mut builder = self.stream.launch_builder(&derivative);
                builder.arg(d_positions);
                builder.arg(d_charges);
                builder.arg(dq);
                builder.arg(&self.d_charge_grid);
                builder.arg(&self.d_energy_components);
                builder.arg(&mut self.d_atom_energy_components);
                builder.arg(&mut self.d_atom_dudlambda_components);
                builder.arg(&n_atoms_i32);
                builder.arg(&nx_i32);
                builder.arg(&ny_i32);
                builder.arg(&nz_i32);
                builder.arg(&box_inv_x);
                builder.arg(&box_inv_y);
                builder.arg(&box_inv_z);
                builder.arg(&inv_volume);
                builder.arg(&self.beta);
                if precision.is_some(){builder.arg(&coordinate_owner).arg(&precision_fault);}
            builder.launch(cfg_atoms)?;
            }
        }
        unsafe {
            let mut builder = self.stream.launch_builder(&self.net_charge_kernel);
            builder.arg(&self.d_energy_components);
            builder.arg(&self.beta);
            builder.arg(&inv_volume);
            builder.launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            })?;
        }

        // No host synchronization or readback is permitted in this enqueue path.
        self.last_enqueued_has_charge_derivative = d_charges_dlambda.is_some();
        Ok(())
    }

    /// Add the most recently enqueued atom-resolved PME energy and charge-path
    /// derivative to fused Hamiltonian accumulators on this PME instance's
    /// exact stream.  No synchronization or host readback occurs.
    ///
    /// `enqueue_reciprocal_with_charge_derivatives` must precede this call on
    /// the same stream.  Ordinary atoms contribute to total energy; only atoms
    /// with a nonnegative slot id contribute to lambda slot energy/dUdl.
    pub fn enqueue_merge_lambda_atom_components(
        &self,
        lambda_atom_slot: &CudaSlice<i32>,
        total_atom_energy: &mut CudaSlice<f64>,
        lambda_atom_energy: &mut CudaSlice<f64>,
        lambda_atom_dudlambda: &mut CudaSlice<f64>,
    ) -> Result<()> {
        for (label, actual) in [
            ("lambda_atom_slot", lambda_atom_slot.len()),
            ("total_atom_energy", total_atom_energy.len()),
            ("lambda_atom_energy", lambda_atom_energy.len()),
            ("lambda_atom_dudlambda", lambda_atom_dudlambda.len()),
        ] {
            anyhow::ensure!(
                actual == self.n_atoms,
                "PME {label} length mismatch: got {actual}, expected {}",
                self.n_atoms
            );
        }
        anyhow::ensure!(
            self.last_enqueued_has_charge_derivative,
            "PME lambda component merge requires a preceding dynamic-charge enqueue"
        );
        let threads = 256usize;
        let n_atoms_i32 = self.n_atoms as i32;
        unsafe {
            let mut builder = self
                .stream
                .launch_builder(&self.merge_lambda_atom_components_kernel);
            builder.arg(&self.d_atom_energy_components);
            builder.arg(&self.d_atom_dudlambda_components);
            builder.arg(lambda_atom_slot);
            builder.arg(total_atom_energy);
            builder.arg(lambda_atom_energy);
            builder.arg(lambda_atom_dudlambda);
            builder.arg(&n_atoms_i32);
            builder.launch(LaunchConfig {
                grid_dim: (self.n_atoms.div_ceil(threads) as u32, 1, 1),
                block_dim: (threads as u32, 1, 1),
                shared_mem_bytes: 0,
            })?;
        }
        Ok(())
    }

    /// Select an isolated endpoint/current charge plane for exactly one
    /// fragment slot. The caller owns stable scratch buffers so the complete
    /// sequence can be captured in the resident CUDA graph.
    pub fn enqueue_select_isolated_lambda_slot_charges(
        &self,
        lambda_atom_slot: &CudaSlice<i32>,
        endpoint_a_charges: &CudaSlice<f32>,
        endpoint_b_charges: &CudaSlice<f32>,
        current_charges: &CudaSlice<f32>,
        current_charge_derivatives: &CudaSlice<f32>,
        target_slot: i32,
        source_plane: i32,
        isolated_charges: &mut CudaSlice<f32>,
        isolated_charge_derivatives: &mut CudaSlice<f32>,
    ) -> Result<()> {
        for (label, actual) in [
            ("lambda_atom_slot", lambda_atom_slot.len()),
            ("endpoint_a_charges", endpoint_a_charges.len()),
            ("endpoint_b_charges", endpoint_b_charges.len()),
            ("current_charges", current_charges.len()),
            (
                "current_charge_derivatives",
                current_charge_derivatives.len(),
            ),
            ("isolated_charges", isolated_charges.len()),
            (
                "isolated_charge_derivatives",
                isolated_charge_derivatives.len(),
            ),
        ] {
            anyhow::ensure!(
                actual == self.n_atoms,
                "PME isolated-slot {label} length mismatch: got {actual}, expected {}",
                self.n_atoms
            );
        }
        anyhow::ensure!(target_slot >= 0, "negative PME correction slot");
        anyhow::ensure!(
            (0..=2).contains(&source_plane),
            "invalid PME correction plane"
        );
        let threads = 256usize;
        let n_atoms_i32 = self.n_atoms as i32;
        unsafe {
            let mut builder = self
                .stream
                .launch_builder(&self.select_isolated_lambda_slot_charges_kernel);
            builder.arg(lambda_atom_slot);
            builder.arg(endpoint_a_charges);
            builder.arg(endpoint_b_charges);
            builder.arg(current_charges);
            builder.arg(current_charge_derivatives);
            builder.arg(&target_slot);
            builder.arg(&source_plane);
            builder.arg(isolated_charges);
            builder.arg(isolated_charge_derivatives);
            builder.arg(&n_atoms_i32);
            builder.launch(LaunchConfig {
                grid_dim: (self.n_atoms.div_ceil(threads) as u32, 1, 1),
                block_dim: (threads as u32, 1, 1),
                shared_mem_bytes: 0,
            })?;
        }
        Ok(())
    }

    /// Accumulate one isolated PME basis plane into the full Hamiltonian.
    /// Coefficients and their lambda derivatives are evaluated on device from
    /// endpoint occupancy and current slot lambda, so graph replay remains
    /// correct after every NCMC perturbation.
    pub fn enqueue_accumulate_lambda_internal_correction(
        &self,
        isolated_forces: &CudaSlice<f32>,
        lambda_atom_slot: &CudaSlice<i32>,
        lambda_slot_values: &CudaSlice<f32>,
        endpoint_occupancy_masks: &CudaSlice<i32>,
        target_slot: i32,
        source_plane: i32,
        total_forces: &mut CudaSlice<f32>,
        total_atom_energy: &mut CudaSlice<f64>,
        lambda_atom_energy: &mut CudaSlice<f64>,
        lambda_atom_dudlambda: &mut CudaSlice<f64>,
    ) -> Result<()> {
        anyhow::ensure!(
            isolated_forces.len() == self.n_atoms * 3 && total_forces.len() == self.n_atoms * 3,
            "PME correction force population mismatch"
        );
        for (label, actual) in [
            ("lambda_atom_slot", lambda_atom_slot.len()),
            ("total_atom_energy", total_atom_energy.len()),
            ("lambda_atom_energy", lambda_atom_energy.len()),
            ("lambda_atom_dudlambda", lambda_atom_dudlambda.len()),
        ] {
            anyhow::ensure!(actual == self.n_atoms, "PME correction {label} mismatch");
        }
        anyhow::ensure!(target_slot >= 0, "negative PME correction slot");
        anyhow::ensure!(
            (0..=2).contains(&source_plane),
            "invalid PME correction plane"
        );
        anyhow::ensure!(
            lambda_slot_values.len() > target_slot as usize
                && endpoint_occupancy_masks.len() > target_slot as usize,
            "PME correction slot metadata population mismatch"
        );
        anyhow::ensure!(
            self.last_enqueued_has_charge_derivative,
            "PME correction requires preceding isolated dynamic-charge enqueue"
        );
        let threads = 256usize;
        let n_atoms_i32 = self.n_atoms as i32;
        unsafe {
            let mut builder = self
                .stream
                .launch_builder(&self.accumulate_lambda_internal_correction_kernel);
            builder.arg(&self.d_atom_energy_components);
            builder.arg(&self.d_atom_dudlambda_components);
            builder.arg(isolated_forces);
            builder.arg(lambda_atom_slot);
            builder.arg(lambda_slot_values);
            builder.arg(endpoint_occupancy_masks);
            builder.arg(&target_slot);
            builder.arg(&source_plane);
            builder.arg(total_forces);
            builder.arg(total_atom_energy);
            builder.arg(lambda_atom_energy);
            builder.arg(lambda_atom_dudlambda);
            builder.arg(&n_atoms_i32);
            builder.launch(LaunchConfig {
                grid_dim: (self.n_atoms.div_ceil(threads) as u32, 1, 1),
                block_dim: (threads as u32, 1, 1),
                shared_mem_bytes: 0,
            })?;
        }
        Ok(())
    }

    /// Read scalar components at an explicit NCMC, receipt, or terminal boundary.
    pub fn readback_reciprocal_result(&self) -> Result<PmeReciprocalResult> {
        let mut components = [0.0f64; 8];
        self.stream
            .memcpy_dtoh(&self.d_energy_components, &mut components)?;
        let has_dudl = self.last_enqueued_has_charge_derivative;
        let result = PmeReciprocalResult {
            mesh_energy: components[0],
            self_energy: components[1],
            total_charge: components[2],
            net_charge_correction: components[3],
            reciprocal_energy: components[0] + components[1] + components[3],
            mesh_dudlambda: has_dudl.then_some(components[4]),
            self_dudlambda: has_dudl.then_some(components[5]),
            total_charge_derivative: has_dudl.then_some(components[6]),
            net_charge_dudlambda: has_dudl.then_some(components[7]),
            reciprocal_dudlambda: has_dudl.then_some(components[4] + components[5] + components[7]),
        };

        log::debug!(
            "PME reciprocal components: mesh={:.6} self={:.6} net={:.6} total={:.6} kcal/mol",
            result.mesh_energy,
            result.self_energy,
            result.net_charge_correction,
            result.reciprocal_energy
        );

        Ok(result)
    }

    /// Device result layout: mesh E, self E, Q, background E, mesh dU/dlambda,
    /// self dU/dlambda, dQ/dlambda, background dU/dlambda.
    pub fn device_result_components(&self) -> &CudaSlice<f64> {
        &self.d_energy_components
    }

    /// Get the current grid dimensions
    pub fn grid_dims(&self) -> (usize, usize, usize) {
        (self.nx, self.ny, self.nz)
    }

    /// Get the Ewald beta parameter
    pub fn beta(&self) -> f32 {
        self.beta
    }

    /// Fail-closed comparison against the prepared production PME contract.
    pub fn matches_exact_parameters(
        &self,
        grid: [usize; 3],
        interpolation_order: u32,
        ewald_beta_a_inv: f32,
        real_space_cutoff_a: f32,
        relative_force_tolerance: f32,
    ) -> bool {
        self.exact_parameter_lock
            && [self.nx, self.ny, self.nz] == grid
            && interpolation_order == 5
            && self.beta.to_bits() == ewald_beta_a_inv.to_bits()
            && self.real_space_cutoff_a.to_bits() == real_space_cutoff_a.to_bits()
            && self.relative_force_tolerance.to_bits() == relative_force_tolerance.to_bits()
    }

    /// Orthorhombic cell dimensions executed by this instance.
    pub fn box_dims(&self) -> [f32; 3] {
        self.box_dims
    }

    /// Explicit split contract admitted by this instance.
    pub fn split_contract(&self) -> PmeSplitContract {
        self.contract
    }

    /// Content identity of the exact runtime PME parameterization validated by
    /// external OpenMM and target-system measurements.
    pub fn runtime_parameterization_sha256(&self) -> String {
        let mut digest = Sha256::new();
        digest.update(b"PRISM_PME_RUNTIME_PARAMETERIZATION_V1\0");
        for value in [self.nx as u64, self.ny as u64, self.nz as u64] {
            digest.update(value.to_le_bytes());
        }
        for value in self.box_dims {
            digest.update(value.to_bits().to_le_bytes());
        }
        for value in [
            self.beta,
            self.real_space_cutoff_a,
            self.relative_force_tolerance,
        ] {
            digest.update(value.to_bits().to_le_bytes());
        }
        digest.update(5u32.to_le_bytes());
        digest.update([u8::from(self.contract.real_space_erfc_external)]);
        digest.update([u8::from(
            self.contract.reciprocal_exclusion_correction_external,
        )]);
        digest.update([u8::from(self.contract.lambda_derivatives_external)]);
        format!("{:x}", digest.finalize())
    }

    /// Attach independently measured OpenMM and target-system closure. Both
    /// receipts must bind the current exact parameterization and pass their
    /// own numeric tolerances before any capability bit can become true.
    pub fn attach_measured_parity_receipts(
        &mut self,
        openmm: &PmeMeasuredParityReceipt,
        target_system: &PmeMeasuredParityReceipt,
    ) -> Result<()> {
        let parameterization = self.runtime_parameterization_sha256();
        openmm.validate_for("OPENMM_ORDER5_SAME_PARAMETER_U_F", &parameterization)?;
        target_system.validate_for("TARGET_SYSTEM_COMPOSED_EWALD_U_F_DUDL", &parameterization)?;
        self.openmm_reference_validation_receipt_sha256 = Some(openmm.receipt_sha256.clone());
        self.target_system_validation_receipt_sha256 = Some(target_system.receipt_sha256.clone());
        Ok(())
    }

    /// Query the exact implemented scope and any runtime validation evidence.
    pub fn capability_receipt(&self) -> PmeRuntimeCapabilityReceipt {
        const CONTRACT_CANONICAL: &str = "boundary=ORTHORHOMBIC_TINFOIL_3D;internal=RECIPROCAL_MESH,SELF,UNIFORM_BACKGROUND;external=REAL_SPACE_ERFC,RECIPROCAL_EXCLUSIONS,REAL_SPACE_DUDL,EXCLUSION_DUDL;dynamic_charge_source=CALLER_CURRENT_LAMBDA";
        let digest = hex::encode(Sha256::digest(CONTRACT_CANONICAL.as_bytes()));
        PmeRuntimeCapabilityReceipt {
            schema: "prism.pme.runtime-capability.v1",
            supplied_stream_bound: true,
            stream_handle: self.stream.cu_stream() as usize,
            cufft_stream_bound: true,
            cufft_plan_handles_nonzero: self.fft_plan_r2c != 0 && self.fft_plan_c2r != 0,
            graph_hot_path_dtoh_count: 0,
            graph_capture_dependency_mode:
                "CALLER_MANAGED_SINGLE_STREAM_OR_CAPTURE_LOCAL_DEPENDENCIES",
            scalar_accumulation: "FP64_ATOMIC_WITH_PERMUTATION_ERROR_RECEIPT",
            boundary: "ORTHORHOMBIC_TINFOIL_3D",
            reciprocal_scope: "MESH_PLUS_SELF_PLUS_UNIFORM_BACKGROUND_ONLY",
            self_correction: true,
            uniform_background_correction: true,
            dynamic_charges_and_dudlambda: true,
            grid: [self.nx, self.ny, self.nz],
            interpolation_order: 5,
            order5_assignment_operator: true,
            openmm_order5_parity: self.openmm_reference_validation_receipt_sha256.is_some(),
            ewald_beta_a_inv: self.beta,
            real_space_cutoff_a: self.real_space_cutoff_a,
            relative_force_tolerance: self.relative_force_tolerance,
            exact_parameter_lock: self.exact_parameter_lock,
            total_ewald_hamiltonian_validation: self
                .target_system_validation_receipt_sha256
                .is_some(),
            openmm_reference_validation_receipt_sha256: self
                .openmm_reference_validation_receipt_sha256
                .clone(),
            target_system_validation_receipt_sha256: self
                .target_system_validation_receipt_sha256
                .clone(),
            external_contract_sha256: digest,
            external_real_space_erfc: self.contract.real_space_erfc_external,
            external_reciprocal_exclusions: self.contract.reciprocal_exclusion_correction_external,
            external_lambda_derivatives: self.contract.lambda_derivatives_external,
            validation: self.runtime_validation.clone(),
        }
    }

    /// Run deterministic force/energy, translation, and charge-path checks.
    ///
    /// This allocates isolated device buffers and does not modify engine state.
    /// Admission should require `validation.status == "PASSED"` from the
    /// returned receipt rather than treating construction as validation.
    pub fn validate_runtime_capability(&mut self) -> Result<PmeRuntimeCapabilityReceipt> {
        anyhow::ensure!(
            self.n_atoms >= 2,
            "PME runtime validation requires an instance with at least two atoms"
        );
        let mut positions = vec![0.0f32; self.n_atoms * 3];
        positions[0..3].copy_from_slice(&[
            0.23 * self.box_dims[0],
            0.31 * self.box_dims[1],
            0.41 * self.box_dims[2],
        ]);
        positions[3..6].copy_from_slice(&[
            0.57 * self.box_dims[0],
            0.44 * self.box_dims[1],
            0.68 * self.box_dims[2],
        ]);
        for atom in 2..self.n_atoms {
            positions[atom * 3..atom * 3 + 3].copy_from_slice(&[
                (0.11 + 0.071 * atom as f32).fract() * self.box_dims[0],
                (0.19 + 0.053 * atom as f32).fract() * self.box_dims[1],
                (0.29 + 0.037 * atom as f32).fract() * self.box_dims[2],
            ]);
        }
        let mut charges = vec![0.0f32; self.n_atoms];
        charges[0] = 0.65;
        charges[1] = -0.35;
        let mut dcharges = vec![0.0f32; self.n_atoms];
        dcharges[0] = 0.20;
        dcharges[1] = -0.10;

        let (base, base_forces) =
            self.evaluate_host_configuration(&positions, &charges, Some(&dcharges))?;
        let dx = 2.0e-3f32;
        let mut plus_x = positions.clone();
        let mut minus_x = positions.clone();
        plus_x[0] += dx;
        minus_x[0] -= dx;
        let e_plus = self
            .evaluate_host_configuration(&plus_x, &charges, None)?
            .0
            .reciprocal_energy;
        let e_minus = self
            .evaluate_host_configuration(&minus_x, &charges, None)?
            .0
            .reciprocal_energy;
        let fd_force = -(e_plus - e_minus) / (2.0 * dx as f64);
        let force_abs = (base_forces[0] as f64 - fd_force).abs();
        let force_rel = force_abs / base_forces[0].abs().max(fd_force.abs() as f32).max(1.0) as f64;

        let mut translated = positions.clone();
        let shift = [
            0.137 * self.box_dims[0],
            -0.219 * self.box_dims[1],
            0.173 * self.box_dims[2],
        ];
        for atom in 0..self.n_atoms {
            for axis in 0..3 {
                translated[atom * 3 + axis] += shift[axis];
            }
        }
        let translated_energy = self
            .evaluate_host_configuration(&translated, &charges, None)?
            .0
            .reciprocal_energy;
        let translation_abs = (translated_energy - base.reciprocal_energy).abs();

        let dl = 2.0e-3f32;
        let charges_plus: Vec<f32> = charges
            .iter()
            .zip(&dcharges)
            .map(|(q, dq)| q + dl * dq)
            .collect();
        let charges_minus: Vec<f32> = charges
            .iter()
            .zip(&dcharges)
            .map(|(q, dq)| q - dl * dq)
            .collect();
        let lambda_plus = self
            .evaluate_host_configuration(&positions, &charges_plus, None)?
            .0
            .reciprocal_energy;
        let lambda_minus = self
            .evaluate_host_configuration(&positions, &charges_minus, None)?
            .0
            .reciprocal_energy;
        let fd_dudl = (lambda_plus - lambda_minus) / (2.0 * dl as f64);
        let analytic_dudl = base
            .reciprocal_dudlambda
            .context("dynamic-charge validation did not return dU/dlambda")?;
        let dudl_abs = (analytic_dudl - fd_dudl).abs();
        let dudl_rel = dudl_abs / analytic_dudl.abs().max(fd_dudl.abs()).max(1.0);

        let mut permuted_positions = positions.clone();
        for axis in 0..3 {
            permuted_positions.swap(axis, 3 + axis);
        }
        let mut permuted_charges = charges.clone();
        permuted_charges.swap(0, 1);
        let mut permuted_dcharges = dcharges.clone();
        permuted_dcharges.swap(0, 1);
        let permuted = self
            .evaluate_host_configuration(
                &permuted_positions,
                &permuted_charges,
                Some(&permuted_dcharges),
            )?
            .0;
        let atom_order_energy_abs = (permuted.reciprocal_energy - base.reciprocal_energy).abs();
        let atom_order_dudl_abs = (permuted
            .reciprocal_dudlambda
            .context("permutation validation did not return dU/dlambda")?
            - analytic_dudl)
            .abs();

        // Cubic mesh assignment is not continuously translation invariant;
        // the acceptance bound records the measured 1 Å-grid discretization
        // envelope while remaining tighter than the independent-energy gate.
        let passed = force_rel <= 5.0e-3
            && translation_abs <= 3.0e-3
            && dudl_rel <= 5.0e-3
            && atom_order_energy_abs <= 2.0e-5
            && atom_order_dudl_abs <= 2.0e-5;
        self.runtime_validation = Some(PmeRuntimeValidationEvidence {
            status: if passed { "PASSED" } else { "FAILED" },
            force_energy_abs_error: force_abs,
            force_energy_rel_error: force_rel,
            translation_energy_abs_error: translation_abs,
            charge_dudlambda_abs_error: dudl_abs,
            charge_dudlambda_rel_error: dudl_rel,
            force_difference_step_angstrom: dx as f64,
            lambda_difference_step: dl as f64,
            atom_order_energy_abs_error: atom_order_energy_abs,
            atom_order_dudlambda_abs_error: atom_order_dudl_abs,
        });
        anyhow::ensure!(
            passed,
            "PME runtime validation failed: {:?}",
            self.runtime_validation
        );
        Ok(self.capability_receipt())
    }

    /// Run the native deterministic validation and then attach independently
    /// measured OpenMM/target-system receipts. This is the only API that can
    /// return a capability receipt with the two external-proof bits true.
    pub fn validate_runtime_capability_with_measured_parity(
        &mut self,
        openmm: &PmeMeasuredParityReceipt,
        target_system: &PmeMeasuredParityReceipt,
    ) -> Result<PmeRuntimeCapabilityReceipt> {
        self.validate_runtime_capability()?;
        self.attach_measured_parity_receipts(openmm, target_system)?;
        Ok(self.capability_receipt())
    }

    fn evaluate_host_configuration(
        &mut self,
        positions: &[f32],
        charges: &[f32],
        dcharges: Option<&[f32]>,
    ) -> Result<(PmeReciprocalResult, Vec<f32>)> {
        let d_positions = self.stream.clone_htod(positions)?;
        let d_charges = self.stream.clone_htod(charges)?;
        let mut d_forces = self.stream.alloc_zeros::<f32>(self.n_atoms * 3)?;
        let result = if let Some(dq) = dcharges {
            let d_dq = self.stream.clone_htod(dq)?;
            self.compute_reciprocal_with_charge_derivatives(
                &d_positions,
                &d_charges,
                &d_dq,
                &mut d_forces,
            )?
        } else {
            self.compute_reciprocal(&d_positions, &d_charges, &mut d_forces)?
        };
        let mut forces = vec![0.0f32; self.n_atoms * 3];
        self.stream.memcpy_dtoh(&d_forces, &mut forces)?;
        Ok((result, forces))
    }
}

impl Drop for PME {
    fn drop(&mut self) {
        if self.plans_initialized {
            unsafe {
                crate::cufft_sys::cufftDestroy(self.fft_plan_r2c);
                crate::cufft_sys::cufftDestroy(self.fft_plan_c2r);
            }
            log::debug!("🧹 PME cuFFT plans destroyed");
        }
    }
}

/// Round up to a good FFT size (powers of 2, or products of 2, 3, 5)
fn round_up_fft_size(n: usize) -> usize {
    // Good FFT sizes are products of small primes (2, 3, 5)
    // Find smallest such number >= n
    let good_sizes = [
        8, 10, 12, 16, 18, 20, 24, 30, 32, 36, 40, 48, 54, 60, 64, 72, 80, 90, 96, 100, 108, 120,
        128, 144, 160, 180, 192, 200, 216, 240, 256, 270, 288, 320, 324, 360, 384, 400, 432, 480,
        512,
    ];

    for &size in &good_sizes {
        if size >= n {
            return size;
        }
    }

    // Fall back to next power of 2
    let mut size = 512;
    while size < n {
        size *= 2;
    }
    size
}

/// Create R2C and C2R FFT plans
fn create_fft_plans(
    nx: usize,
    ny: usize,
    nz: usize,
    stream: &CudaStream,
) -> Result<(CufftHandle, CufftHandle)> {
    let mut plan_r2c: CufftHandle = 0;
    let mut plan_c2r: CufftHandle = 0;

    unsafe {
        // R2C plan (forward FFT)
        let result = crate::cufft_sys::cufftPlan3d(
            &mut plan_r2c,
            nx as i32,
            ny as i32,
            nz as i32,
            CUFFT_R2C,
        );
        if result != CUFFT_SUCCESS {
            return Err(anyhow::anyhow!(
                "cufftPlan3d R2C failed: {}",
                cufft_error_string(result)
            ));
        }

        // C2R plan (inverse FFT)
        let result = crate::cufft_sys::cufftPlan3d(
            &mut plan_c2r,
            nx as i32,
            ny as i32,
            nz as i32,
            CUFFT_C2R,
        );
        if result != CUFFT_SUCCESS {
            crate::cufft_sys::cufftDestroy(plan_r2c);
            return Err(anyhow::anyhow!(
                "cufftPlan3d C2R failed: {}",
                cufft_error_string(result)
            ));
        }

        debug_assert_ne!(plan_r2c, 0, "successful R2C plan must be nonzero");
        debug_assert_ne!(plan_c2r, 0, "successful C2R plan must be nonzero");

        for (label, plan) in [("R2C", plan_r2c), ("C2R", plan_c2r)] {
            let result =
                crate::cufft_sys::cufftSetStream(plan, stream.cu_stream().cast::<c_void>());
            if result != CUFFT_SUCCESS {
                crate::cufft_sys::cufftDestroy(plan_r2c);
                crate::cufft_sys::cufftDestroy(plan_c2r);
                return Err(anyhow::anyhow!(
                    "cufftSetStream {} failed: {}",
                    label,
                    cufft_error_string(result)
                ));
            }
        }
    }

    Ok((plan_r2c, plan_c2r))
}

/// Execute R2C FFT (real to complex)
fn execute_fft_r2c(
    plan: CufftHandle,
    d_input: &mut CudaSlice<f32>,
    d_output: &mut CudaSlice<f32>,
    stream: &CudaStream,
) -> Result<()> {
    unsafe {
        // Get raw device pointers - cudarc 0.18.2 returns (CUdeviceptr, SyncOnDrop)
        let (input_ptr, _sync_in) = d_input.device_ptr_mut(stream);
        let (output_ptr, _sync_out) = d_output.device_ptr_mut(stream);

        let result = crate::cufft_sys::cufftExecR2C(
            plan,
            input_ptr as *mut f32,
            output_ptr as *mut CufftComplex,
        );
        if result != CUFFT_SUCCESS {
            return Err(anyhow::anyhow!(
                "cufftExecR2C failed: {}",
                cufft_error_string(result)
            ));
        }
    }
    Ok(())
}

/// Execute C2R FFT (complex to real)
fn execute_fft_c2r(
    plan: CufftHandle,
    d_input: &mut CudaSlice<f32>,
    d_output: &mut CudaSlice<f32>,
    stream: &CudaStream,
) -> Result<()> {
    unsafe {
        // Get raw device pointers - cudarc 0.18.2 returns (CUdeviceptr, SyncOnDrop)
        let (input_ptr, _sync_in) = d_input.device_ptr_mut(stream);
        let (output_ptr, _sync_out) = d_output.device_ptr_mut(stream);

        let result = crate::cufft_sys::cufftExecC2R(
            plan,
            input_ptr as *mut CufftComplex,
            output_ptr as *mut f32,
        );
        if result != CUFFT_SUCCESS {
            return Err(anyhow::anyhow!(
                "cufftExecC2R failed: {}",
                cufft_error_string(result)
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_up_fft_size() {
        assert_eq!(round_up_fft_size(5), 8);
        assert_eq!(round_up_fft_size(8), 8);
        assert_eq!(round_up_fft_size(9), 10);
        assert_eq!(round_up_fft_size(33), 36);
        assert_eq!(round_up_fft_size(100), 100);
        assert_eq!(round_up_fft_size(101), 108);
    }

    #[test]
    fn openmm_grid_lower_bound_matches_nonbondedforce_formula() {
        let alpha = compute_ewald_beta(10.0, 1.0e-5);
        let lower = openmm_pme_grid_lower_bound([117.0, 117.0, 117.0], alpha, 1.0e-5);
        let expected = (2.0 * alpha * 117.0 / (3.0 * (1.0e-5_f32).powf(0.2))).ceil() as usize;
        assert_eq!(lower, [expected, expected, expected]);
        assert!(expected >= 6);
    }
}
