//! NHS Configuration and Constants
//!
//! Central configuration for the Neuromorphic Holographic Stream pipeline.
//! All magic numbers are defined here with physical justification.
//!
//! # Physical Constants
//!
//! The UV bias mechanism is based on real aromatic absorption physics:
//! - Tryptophan: ε ≈ 5,600 M⁻¹cm⁻¹ at 280nm
//! - Tyrosine: ε ≈ 1,400 M⁻¹cm⁻¹ at 280nm
//! - Phenylalanine: ε ≈ 200 M⁻¹cm⁻¹ at 280nm
//! - Water: ε ≈ 0 (TRANSPARENT at 280nm)

use serde::{Deserialize, Serialize};

// =============================================================================
// PHYSICAL CONSTANTS
// =============================================================================

/// Water probe radius (Å) - standard value for SASA calculations
pub const WATER_PROBE_RADIUS: f32 = 1.4;

/// Bulk water density (molecules/Å³) at 300K
pub const BULK_WATER_DENSITY: f32 = 0.0334;

/// Boltzmann constant × Temperature (kcal/mol at 300K)
pub const KT_300K: f32 = 0.596;

/// Boltzmann constant in kcal/(mol·K)
pub const KB_KCAL_MOL_K: f32 = 0.001987204;

/// Grid spacing for exclusion field (Å)
/// 0.5Å provides good resolution without excessive memory
pub const DEFAULT_GRID_SPACING: f32 = 0.5;

/// Padding around protein bounding box (Å)
pub const GRID_PADDING: f32 = 8.0;

/// Dewetting spike threshold (fraction of bulk density)
/// Below this = dewetted = spike
pub const DEWETTING_THRESHOLD: f32 = 0.3;

/// Minimum avalanche size to report as cryptic site (number of spikes)
pub const MIN_AVALANCHE_SIZE: usize = 5;

/// Minimum pocket volume to be druggable (Å³)
pub const MIN_DRUGGABLE_VOLUME: f32 = 100.0;

// =============================================================================
// UV ABSORPTION CONSTANTS - MULTI-WAVELENGTH SPECTROSCOPY
// =============================================================================

// ----------------------------- λmax Values (nm) -----------------------------
/// Tryptophan λmax - La band π→π* transition
pub const TRP_LAMBDA_MAX: f32 = 280.0;
/// Tyrosine λmax - phenol π→π* transition
pub const TYR_LAMBDA_MAX: f32 = 274.0;
/// Phenylalanine λmax - benzyl π→π* transition
pub const PHE_LAMBDA_MAX: f32 = 258.0;
/// Disulfide λmax - σ→σ* transition
pub const DISULFIDE_LAMBDA_MAX: f32 = 250.0;
/// Benzene virtual cosolvent λmax - ¹B₂ᵤ π→π* (6₀¹ vibronic peak)
/// Callomon, Dunn & Mills (1966), Philos. Trans. R. Soc. A 259, 499–532
pub const BENZENE_LAMBDA_MAX: f32 = 254.0;

// ------------------------- Extinction Coefficients (M⁻¹cm⁻¹) ----------------
/// Tryptophan molar extinction coefficient at 280nm (M⁻¹cm⁻¹)
/// Strongest UV absorber - primary target for UV bias
pub const TRP_EXTINCTION_280: f32 = 5600.0;
/// Tyrosine molar extinction coefficient at 274nm
pub const TYR_EXTINCTION_274: f32 = 1490.0;
/// Tyrosine at 280nm (secondary peak)
pub const TYR_EXTINCTION_280: f32 = 1400.0;
/// Phenylalanine molar extinction coefficient at 258nm
pub const PHE_EXTINCTION_258: f32 = 200.0;
/// Phenylalanine at 280nm (off-peak)
pub const PHE_EXTINCTION_280: f32 = 200.0;
/// Disulfide molar extinction coefficient at 250nm
pub const DISULFIDE_EXTINCTION_250: f32 = 300.0;
/// Benzene molar extinction coefficient at 254nm (¹B₂ᵤ band)
/// Pantos, Philis & Bolovinos (1978), J. Mol. Spectrosc. 72, 36–43
/// Confirmed: PhotochemCAD (Du et al. 1998), Fally et al. (2009)
pub const BENZENE_EXTINCTION_254: f32 = 204.0;

// --------------------- Spectral Band Widths (Gaussian 1/e half-width) -------
//
// CONVENTION (P6, canonical tree-wide). These constants are the Gaussian
// **1/e half-width** `w`, defined by
//
//     eps(lambda) = eps_max * exp( -((lambda - lambda_max) / w)^2 )
//
// and therefore relate to the standard deviation and FWHM as
//
//     sigma = w / sqrt(2)          FWHM = sigma * 2.35482 = w * 1.66511
//
// They are NOT FWHM. Three incompatible readings of these same numbers were
// live in the tree simultaneously before P6 -- see `band_sigma` below for the
// adjudication and the evidence that settled it.
//
// Always convert with `band_sigma()`. Never divide by 2.355 at a call site.

/// Tryptophan band 1/e half-width (nm). sigma = 10.607 nm, FWHM = 24.98 nm.
pub const TRP_BANDWIDTH: f32 = 15.0;
/// Tyrosine band 1/e half-width (nm). sigma = 8.485 nm, FWHM = 19.98 nm.
pub const TYR_BANDWIDTH: f32 = 12.0;
/// Phenylalanine band 1/e half-width (nm). sigma = 7.071 nm, FWHM = 16.65 nm.
pub const PHE_BANDWIDTH: f32 = 10.0;
/// Disulfide band 1/e half-width (nm), broader due to conformational
/// heterogeneity. sigma = 14.142 nm, FWHM = 33.30 nm.
pub const DISULFIDE_BANDWIDTH: f32 = 20.0;
/// Benzene bandwidth - identical ¹B₂ᵤ chromophore as PHE (4nm blue-shifted)
/// Must match PHE_BANDWIDTH: same transition, same vibronic envelope
/// Bernstein & Lee (1981), J. Chem. Phys. 74(6), 3159
pub const BENZENE_BANDWIDTH: f32 = PHE_BANDWIDTH;

/// Convert a band 1/e half-width `w` (nm) to the Gaussian standard deviation
/// `sigma` (nm): `sigma = w / sqrt(2)`.
///
/// # Why this factor, and not `/ 2.355`
///
/// Before P6 three mutually incompatible conversions of the *same* constants
/// were live at once:
///
/// | form | conversion        | sigma(TRP) | sites |
/// |------|-------------------|-----------|-------|
/// | F1   | `w / 2.355`       | 6.369 nm  | `config.rs`, `uv_bias.rs`, `fused_engine.rs`, `cryo_thermal_detection.cuh` |
/// | F2   | `w` used as sigma | 15.000 nm | `nhs_excited_state.cuh` |
/// | F3   | `w / sqrt(2)`     | 10.607 nm | `adjudicator.cu` `c_mu_01_sq`, mirrored in `interferometric_adjudicator.rs` |
///
/// F1 and F2 differ by exactly 2.355x on the identical constant, so the CPU
/// host and the excited-state GPU kernel disagreed by 134x in deposited heat
/// at 258 nm within a single run.
///
/// F3 is authoritative. Two independent lines of evidence, neither of which
/// requires appealing to a value outside this tree:
///
/// 1. **The deployed adjudicator LUT selects F3 uniquely.** `c_mu_01_sq` in
///    `cuda/adjudicator.cu` is baked into device `__constant__` memory at
///    `[0.0343, 1.0000, 0.0621, 0.000810]` for the 260 / 280 / 305 / 320 nm
///    probes. Only F3 reproduces it:
///
///    | probe  | baked    | F3 (w/sqrt 2) | F1 (w/2.355) | F2 (w as sigma) |
///    |--------|----------|---------------|--------------|-----------------|
///    | 305 nm | 0.0621   | 0.06218       | 4.52e-4      | 0.2494          |
///    | 320 nm | 0.000810 | 8.16e-4       | 2.73e-9      | 0.02857         |
///
///    F1 is off by 138x at 305 nm and by 3e5 x at 320 nm; F2 by 4x and 35x.
///    The host model and a constant already resident in device memory
///    disagreed by two orders of magnitude about how much signal the "TRP
///    gaussian tail" probe carries. `test_gpu_parity_band_model` asserts both
///    the reproduction and the exclusion.
///
/// 2. **F1 contradicts a constant in this same file.** F1 places tyrosine's
///    half-maximum at `274 +/- 6 nm`, i.e. exactly at 280 nm, predicting
///    `eps_TYR(280) = 745`. `TYR_EXTINCTION_280` in this file is `1400`.
///    F1 is off by 1.88x against the tree's own measured value; F3 predicts
///    1160 and is off by 1.21x. Neither is exact -- see the residual recorded
///    in `OPERATOR_CONTROL` -- but F1 is decisively the worse fit.
///
/// The adjudicator LUT was already computed under F3 and is baked into device
/// `__constant__` memory, so adopting F3 changes **no** adjudication numeric.
/// It moves the CPU host and the excited-state kernel onto the convention the
/// adjudicator already used.
#[inline]
pub fn band_sigma(band_width_nm: f32) -> f32 {
    band_width_nm * std::f32::consts::FRAC_1_SQRT_2
}

/// Convert a band 1/e half-width `w` (nm) to FWHM (nm), for reporting only.
/// `FWHM = 2 * sqrt(ln 2) * w = 1.665109 * w`.
#[inline]
pub fn band_fwhm(band_width_nm: f32) -> f32 {
    band_width_nm * 1.665_109_2
}

/// Water extinction at 280nm - TRANSPARENT
/// This is the key insight: water doesn't absorb UV at aromatic wavelengths
pub const WATER_EXTINCTION_280: f32 = 0.0;

/// Normalization factor for absorption strengths (relative to Trp)
pub const ABSORPTION_NORMALIZATION: f32 = TRP_EXTINCTION_280;

// ----------------------------- Physical Constants ---------------------------
/// Boltzmann constant in eV/K
pub const KB_EV_K: f32 = 8.617e-5;
/// Planck's constant in eV·s
pub const PLANCK_EV_S: f32 = 4.136e-15;
/// Speed of light in nm/s
pub const SPEED_OF_LIGHT_NM_S: f32 = 2.998e17;

// =============================================================================
// UV PUMP-PROBE CALIBRATION CONSTANTS (PHYSICS-CORRECTED)
// =============================================================================

/// Proper ε → σ conversion factor (per molecule)
/// σ(Å²) = ε(M⁻¹cm⁻¹) × ln(10) / N_A
/// σ(Å²) = ε × 2.303 / (6.022×10²³) × 10¹⁶
/// σ(Å²) = ε × 3.823×10⁻⁵
pub const EPSILON_TO_SIGMA_CONVERSION: f32 = 3.823e-5;

/// Calibrated photon fluence (photons/Å² per pulse)
/// Anchored to ΔT = 20 K for TRP @ 280nm with η = 1.0
pub const CALIBRATED_PHOTON_FLUENCE: f32 = 0.024;

/// Calibrated fluence in experimental units (mJ/cm² at 280nm)
/// Conversion: H(mJ/cm²) = F(photons/Å²) × 10¹⁶ (Å²/cm²) × E_photon(J) × 10³ (mJ/J)
/// At 280nm: E_photon = 7.09×10⁻¹⁹ J, so 1 photon/Å² ≈ 7.09 mJ/cm²
/// Therefore: 0.024 photons/Å² ≈ 0.170 mJ/cm²
pub const CALIBRATED_FLUENCE_MJ_CM2_280NM: f32 = 0.170;

/// Heat yield η (fraction of absorbed photon energy converted to local heat)
/// η = 1.0 is conservative upper bound (all absorbed energy → vibrational heat)
/// In practice, some energy may be re-emitted as fluorescence or transferred.
/// For computational pump-probe, η = 1.0 provides maximum structural perturbation.
pub const DEFAULT_HEAT_YIELD: f32 = 1.0;

/// Alias for backward compatibility - energy_deposition_fraction IS heat_yield (η)
pub const DEFAULT_ENERGY_DEPOSITION_FRACTION: f32 = DEFAULT_HEAT_YIELD;

/// Per-chromophore heat yields η = (1 - Φ_fluorescence - Φ_ISC)
/// Literature: Lakowicz, Principles of Fluorescence Spectroscopy, 3rd ed.
/// TRP: Φ_f = 0.13 (Szabo & Rayner 1980), Φ_ISC = 0.13 (Bent & Hayon 1975)
/// TYR: Φ_f = 0.14 (Lakowicz), Φ_ISC = 0.03 (Bent & Hayon 1975)
/// PHE: Φ_f = 0.024 (Lakowicz), Φ_ISC = 0.006 (estimated)
/// S-S: No fluorescence from σ→σ* transition
pub const HEAT_YIELD_TRP: f32 = 0.74; // 1 - 0.13 - 0.13
pub const HEAT_YIELD_TYR: f32 = 0.83; // 1 - 0.14 - 0.03
pub const HEAT_YIELD_PHE: f32 = 0.97; // 1 - 0.024 - 0.006
pub const HEAT_YIELD_DISULFIDE: f32 = 1.00; // No fluorescence pathway
/// Benzene virtual cosolvent: η = 1 - Φ_f(0.05) - Φ_isc(0.24) = 0.71
/// Φ_f: Dawson & Windsor (1968), J. Phys. Chem. 72, 3251 (solution phase)
/// Φ_isc: Cundall & Robinson (1972), J. Chem. Soc. Faraday Trans. 2, 68, 1691
///   Dilute limit (~0.2M): Φ_T = 0.24 (no excimer-enhanced ISC)
///   Matches Tan et al. (2016) cosolvent MD concentration (0.2M benzene)
pub const HEAT_YIELD_BENZENE: f32 = 0.71;

/// Effective degrees of freedom for local heating (N_eff)
/// These are EFFECTIVE DOF PROXIES calibrated for local temperature response,
/// NOT literal atom counts. Values chosen to match experimental heating profiles.
pub const NEFF_TRP: f32 = 9.0; // Indole ring system - effective DOF proxy
pub const NEFF_TYR: f32 = 10.0; // Phenol + hydroxyl system - effective DOF proxy
pub const NEFF_PHE: f32 = 9.0; // Benzene + methylene system - effective DOF proxy
pub const NEFF_DISULFIDE: f32 = 2.0; // S-S bond - 2 atoms involved in stretch
/// Benzene virtual cosolvent - 6 ring carbons only (no sidechain)
/// Follows engine convention: TRP(9 ring)→9.0, PHE(6 ring+3 SC)→9.0, BNZ(6 ring)→6.0
pub const NEFF_BENZENE: f32 = 6.0;

/// Single-photon regime threshold
/// Absorption probability must satisfy p << 1
pub const MAX_ABSORPTION_PROBABILITY: f32 = 0.01; // 1% max

// =============================================================================
// CRYO PHYSICS CONSTANTS
// =============================================================================

/// Cryogenic bath temperature (defensible classical MD range)
pub const CRYO_BATH_TEMPERATURE: f32 = 100.0; // Kelvin

/// Ambient reference temperature
pub const AMBIENT_BATH_TEMPERATURE: f32 = 300.0; // Kelvin

/// Minimum temperature (prevents division by zero)
pub const T_MIN: f32 = 10.0; // Kelvin

/// Dielectric constant of water at 300K
pub const EPSILON_WATER_300K: f32 = 78.5;

/// Dielectric constant of ice at 100K
pub const EPSILON_ICE_100K: f32 = 3.2;

/// Base Langevin friction coefficient (ambient)
pub const GAMMA_BASE: f32 = 1.0; // ps⁻¹

/// Equilibration friction (first 10,000 steps)
pub const GAMMA_EQUILIBRATION: f32 = 1000.0; // ps⁻¹

/// Equilibration duration (steps)
pub const EQUILIBRATION_STEPS: i32 = 10000;

/// Convert extinction coefficient to absorption cross-section (per molecule)
/// σ(Å²) = ε(M⁻¹cm⁻¹) × 3.823×10⁻⁵
#[inline]
pub fn extinction_to_cross_section(epsilon: f32) -> f32 {
    epsilon * EPSILON_TO_SIGMA_CONVERSION
}

/// Calibration diagnostic: logs all UV pump-probe parameters for validation
/// Call this to verify the physics chain produces expected ΔT values.
///
/// Expected result for TRP @ 280nm with default parameters:
/// - ε = 5600 M⁻¹cm⁻¹
/// - σ = 0.21409 Å²
/// - F = 0.024 photons/Å²
/// - p = 0.00514 (< 0.01 single-photon threshold)
/// - E_photon = 4.428 eV
/// - E_dep = 0.01687 eV (η_TRP=0.74)
/// - N_eff = 9
/// - ΔT ≈ 14.5 K (η_TRP=0.74)
pub fn log_uv_calibration_diagnostic(
    chromophore: &str,
    wavelength_nm: f32,
    epsilon: f32,
    photon_fluence: f32,
    heat_yield: f32,
    n_eff: f32,
) {
    let sigma = extinction_to_cross_section(epsilon);
    let p_absorb = sigma * photon_fluence;
    let e_photon = wavelength_to_ev(wavelength_nm);
    let e_dep = e_photon * p_absorb * heat_yield;
    let delta_t = e_dep / (1.5 * KB_EV_K * n_eff);

    log::info!("=== UV Pump-Probe Calibration Diagnostic ===");
    log::info!("Chromophore: {}", chromophore);
    log::info!("Wavelength: {:.1} nm", wavelength_nm);
    log::info!("ε(λ): {:.1} M⁻¹cm⁻¹", epsilon);
    log::info!("σ(λ): {:.5} Å²", sigma);
    log::info!("Photon fluence F: {:.4} photons/Å²", photon_fluence);
    log::info!(
        "Absorption probability p: {:.5} (threshold: < {})",
        p_absorb,
        MAX_ABSORPTION_PROBABILITY
    );
    log::info!("Photon energy E_γ: {:.3} eV", e_photon);
    log::info!("Heat yield η: {:.2}", heat_yield);
    log::info!("Energy deposited E_dep: {:.5} eV", e_dep);
    log::info!("Effective DOF N_eff: {:.1}", n_eff);
    log::info!("Temperature rise ΔT: {:.2} K", delta_t);
    log::info!("============================================");

    // Warn if outside expected ranges
    if p_absorb > MAX_ABSORPTION_PROBABILITY {
        log::warn!(
            "p_absorb ({:.5}) exceeds single-photon regime threshold!",
            p_absorb
        );
    }
}

/// Run standard calibration check for TRP @ 280nm
/// Returns (ΔT, p_absorb) for validation
pub fn validate_trp_calibration() -> (f32, f32) {
    let epsilon = TRP_EXTINCTION_280; // 5600
    let sigma = extinction_to_cross_section(epsilon); // 0.21409
    let p_absorb = sigma * CALIBRATED_PHOTON_FLUENCE; // ~0.00514
    let e_photon = wavelength_to_ev(280.0); // 4.428 eV
    let e_dep = e_photon * p_absorb * HEAT_YIELD_TRP;
    let delta_t = e_dep / (1.5 * KB_EV_K * NEFF_TRP);

    log_uv_calibration_diagnostic(
        "TRP",
        280.0,
        epsilon,
        CALIBRATED_PHOTON_FLUENCE,
        HEAT_YIELD_TRP, // Physics-corrected: η_TRP = 0.74
        NEFF_TRP,
    );

    // Verify expected values
    assert!(
        (sigma - 0.21409).abs() < 0.001,
        "σ_TRP mismatch: got {}",
        sigma
    );
    assert!(
        (p_absorb - 0.00514).abs() < 0.0001,
        "p_absorb mismatch: got {}",
        p_absorb
    );
    assert!(
        (delta_t - 14.5).abs() < 1.0,
        "ΔT mismatch: expected ~14.5K (η=0.74), got {:.1}K",
        delta_t
    );

    (delta_t, p_absorb)
}

/// Convert wavelength (nm) to photon energy (eV)
/// E = hc/λ
pub fn wavelength_to_ev(wavelength_nm: f32) -> f32 {
    // hc = 1239.84 eV·nm
    1239.84 / wavelength_nm
}

/// Compute extinction coefficient at a given wavelength using Gaussian band model
/// ε(λ) = ε_max × exp(-(λ - λ_max)² / (2σ²))
/// where σ = band_sigma(w) = w / √2  (w = `*_BANDWIDTH`, the 1/e half-width)
///
/// CANONICAL chromophore_type: 0=TRP, 1=TYR, 2=PHE, 3=S-S, 4=BNZ
pub fn extinction_at_wavelength(chromophore_type: i32, wavelength_nm: f32) -> f32 {
    let (lambda_max, epsilon_max, bandwidth) = match chromophore_type {
        0 => (TRP_LAMBDA_MAX, TRP_EXTINCTION_280, TRP_BANDWIDTH), // TRP @ 280nm
        1 => (TYR_LAMBDA_MAX, TYR_EXTINCTION_274, TYR_BANDWIDTH), // TYR @ 274nm
        2 => (PHE_LAMBDA_MAX, PHE_EXTINCTION_258, PHE_BANDWIDTH), // PHE @ 258nm
        3 => (
            DISULFIDE_LAMBDA_MAX,
            DISULFIDE_EXTINCTION_250,
            DISULFIDE_BANDWIDTH,
        ), // S-S @ 250nm
        4 => (
            BENZENE_LAMBDA_MAX,
            BENZENE_EXTINCTION_254,
            BENZENE_BANDWIDTH,
        ), // BNZ @ 254nm
        _ => (TRP_LAMBDA_MAX, TRP_EXTINCTION_280, TRP_BANDWIDTH), // Default to TRP
    };

    // Gaussian band profile
    let delta = wavelength_nm - lambda_max;
    let sigma = band_sigma(bandwidth);
    (-delta * delta / (2.0 * sigma * sigma)).exp() * epsilon_max
}

/// Compute local heating (ΔT in K) for a chromophore at a specific wavelength
/// Uses full physics chain: ε(λ) → σ(λ) → p → E_dep → ΔT
///
/// CANONICAL chromophore_type: 0=TRP, 1=TYR, 2=PHE, 3=S-S, 4=BNZ
pub fn compute_heating_at_wavelength(chromophore_type: i32, wavelength_nm: f32) -> f32 {
    // Get wavelength-dependent extinction
    let epsilon = extinction_at_wavelength(chromophore_type, wavelength_nm);

    // Convert to cross-section
    let sigma = extinction_to_cross_section(epsilon);

    // Absorption probability
    let p_absorb = sigma * CALIBRATED_PHOTON_FLUENCE;

    // Photon energy at this wavelength
    let e_photon = wavelength_to_ev(wavelength_nm);

    // Per-chromophore heat yield
    let eta = match chromophore_type {
        0 => HEAT_YIELD_TRP,
        1 => HEAT_YIELD_TYR,
        2 => HEAT_YIELD_PHE,
        3 => HEAT_YIELD_DISULFIDE,
        4 => HEAT_YIELD_BENZENE,
        _ => HEAT_YIELD_TRP,
    };

    // Energy deposited
    let e_dep = e_photon * p_absorb * eta;

    // N_eff for this chromophore
    let n_eff = match chromophore_type {
        0 => NEFF_TRP,
        1 => NEFF_TYR,
        2 => NEFF_PHE,
        3 => NEFF_DISULFIDE,
        4 => NEFF_BENZENE,
        _ => NEFF_TRP,
    };

    // Temperature rise
    e_dep / (1.5 * KB_EV_K * n_eff)
}

/// Standard wavelengths for frequency hopping protocol (nm)
/// Includes 254nm for benzene cosolvent probe coverage
pub const FREQUENCY_HOP_WAVELENGTHS: [f32; 6] = [250.0, 254.0, 258.0, 274.0, 280.0, 290.0];

/// Disulfide bond maximum distance (Å) for S-S detection
pub const DISULFIDE_BOND_MAX_DISTANCE: f32 = 2.5;

// =============================================================================
// MAIN CONFIGURATION
// =============================================================================

/// NHS Pipeline Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NhsConfig {
    // =========================================================================
    // Grid parameters
    // =========================================================================
    /// Grid spacing in Angstroms (default: 0.5Å)
    pub grid_spacing: f32,

    /// Padding around protein bounding box in Angstroms
    pub grid_padding: f32,

    // =========================================================================
    // Exclusion field parameters
    // =========================================================================
    /// Gaussian width = VdW_radius × this factor
    pub exclusion_sigma_scale: f32,

    /// Scaling for polar attraction field
    pub polar_attraction_scale: f32,

    // =========================================================================
    // Neuromorphic parameters
    // =========================================================================
    /// Water density threshold for spike (fraction of bulk)
    pub spike_threshold: f32,

    /// LIF neuron membrane time constant (arbitrary units)
    pub membrane_tau: f32,

    /// Lateral synaptic connection weight
    pub synaptic_strength: f32,

    /// Refractory period after spike (frames)
    pub refractory_period: u32,

    // =========================================================================
    // Avalanche detection
    // =========================================================================
    /// Minimum spikes to form valid avalanche
    pub min_avalanche_spikes: usize,

    /// Maximum distance to cluster spikes (Å)
    pub avalanche_spatial_threshold: f32,

    /// Frames to integrate for temporal clustering
    pub avalanche_temporal_window: usize,

    // =========================================================================
    // Site classification
    // =========================================================================
    /// Minimum pocket volume (Å³)
    pub min_volume: f32,

    /// CV(SASA) threshold for cryptic classification
    pub cv_sasa_threshold: f32,

    /// Minimum open frequency
    pub open_freq_min: f32,

    /// Maximum open frequency
    pub open_freq_max: f32,

    // =========================================================================
    // UV Bias parameters (Stage 6)
    // =========================================================================
    /// Enable UV bias perturbation mechanism
    pub uv_bias_enabled: bool,

    /// UV bias configuration
    pub uv_bias: UvBiasConfig,

    // =========================================================================
    // Performance tuning
    // =========================================================================
    /// Use FFT acceleration for holographic encoding
    pub use_fft_acceleration: bool,

    /// Streaming output buffer size
    pub stream_buffer_size: usize,
}

impl Default for NhsConfig {
    fn default() -> Self {
        Self {
            // Grid
            grid_spacing: DEFAULT_GRID_SPACING,
            grid_padding: GRID_PADDING,

            // Exclusion
            exclusion_sigma_scale: 0.3,
            polar_attraction_scale: 2.0,

            // Neuromorphic
            spike_threshold: DEWETTING_THRESHOLD,
            membrane_tau: 5.0,
            synaptic_strength: 0.1,
            refractory_period: 3,

            // Avalanche
            min_avalanche_spikes: MIN_AVALANCHE_SIZE,
            avalanche_spatial_threshold: 6.0,
            avalanche_temporal_window: 10,

            // Site classification (matching existing PRISM-Cryptic)
            min_volume: MIN_DRUGGABLE_VOLUME,
            cv_sasa_threshold: 0.20,
            open_freq_min: 0.05,
            open_freq_max: 0.90,

            // UV Bias - enabled by default
            uv_bias_enabled: true,
            uv_bias: UvBiasConfig::default(),

            // Performance
            use_fft_acceleration: true,
            stream_buffer_size: 1024,
        }
    }
}

// =============================================================================
// UV BIAS CONFIGURATION
// =============================================================================

/// UV Bias Perturbation Configuration
///
/// Controls the pump-probe style targeted perturbation of aromatic residues.
/// Water is transparent at 280nm, so perturbations create signal on silent background.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UvBiasConfig {
    // =========================================================================
    // Target selection
    // =========================================================================
    /// Minimum pocket probability to target nearby aromatics
    pub pocket_probability_threshold: f32,

    /// Maximum distance from high-probability voxel to target aromatic (Å)
    pub target_selection_radius: f32,

    /// Include tryptophan as target (strongest absorber)
    pub target_trp: bool,

    /// Include tyrosine as target
    pub target_tyr: bool,

    /// Include phenylalanine as target (weakest absorber)
    pub target_phe: bool,

    /// Only target surface-exposed aromatics
    pub surface_only: bool,

    /// Minimum solvent accessibility to be considered "surface"
    pub min_sasa_exposure: f32,

    // =========================================================================
    // Burst generation
    // =========================================================================
    /// Use burst mode (true) or continuous perturbation (false)
    pub burst_mode: bool,

    /// Number of pulses per burst
    pub pulses_per_burst: u32,

    /// Frames between pulses within a burst
    pub intra_burst_interval: u32,

    /// Frames between bursts (observation window)
    pub inter_burst_interval: u32,

    /// Base perturbation intensity (velocity boost in Å/ps)
    pub base_intensity: f32,

    /// Scale intensity by absorption coefficient
    pub scale_by_absorption: bool,

    // =========================================================================
    // Perturbation physics
    // =========================================================================
    /// Apply perturbation in aromatic ring plane (mimics π→π* excitation)
    pub ring_plane_perturbation: bool,

    /// Random direction component (0 = fully directed, 1 = fully random)
    pub direction_randomness: f32,

    /// Temperature equivalent of perturbation (K above ambient)
    pub effective_temperature_boost: f32,

    // =========================================================================
    // Response correlation
    // =========================================================================
    /// Enable causal correlation tracking
    pub track_causality: bool,

    /// Maximum lag to check for pump-probe correlation (frames)
    pub max_correlation_lag: usize,

    /// Minimum correlation coefficient to establish causality
    pub min_correlation_threshold: f32,

    /// Window size for correlation computation
    pub correlation_window: usize,
}

impl Default for UvBiasConfig {
    fn default() -> Self {
        Self {
            // Target selection
            pocket_probability_threshold: 0.3,
            target_selection_radius: 8.0,
            target_trp: true,  // Primary target
            target_tyr: true,  // Secondary target
            target_phe: false, // Weak absorber, skip by default
            surface_only: true,
            min_sasa_exposure: 0.2,

            // Burst generation - pump-probe style
            burst_mode: true,
            pulses_per_burst: 3,
            intra_burst_interval: 2,
            inter_burst_interval: 20, // Observation window
            base_intensity: 0.5,      // Å/ps velocity boost
            scale_by_absorption: true,

            // Perturbation physics
            ring_plane_perturbation: true,
            direction_randomness: 0.3,
            effective_temperature_boost: 50.0, // +50K local heating

            // Response correlation
            track_causality: true,
            max_correlation_lag: 15,
            min_correlation_threshold: 0.5,
            correlation_window: 50,
        }
    }
}

impl UvBiasConfig {
    /// Get relative absorption strength for a residue type
    pub fn absorption_strength(&self, residue: &str) -> f32 {
        match residue.to_uppercase().as_str() {
            "TRP" | "W" => TRP_EXTINCTION_280 / ABSORPTION_NORMALIZATION,
            "TYR" | "Y" => TYR_EXTINCTION_280 / ABSORPTION_NORMALIZATION,
            "PHE" | "F" => PHE_EXTINCTION_280 / ABSORPTION_NORMALIZATION,
            _ => 0.0, // Non-aromatic: no absorption
        }
    }

    /// Check if a residue type is a valid target
    pub fn is_valid_target(&self, residue: &str) -> bool {
        match residue.to_uppercase().as_str() {
            "TRP" | "W" => self.target_trp,
            "TYR" | "Y" => self.target_tyr,
            "PHE" | "F" => self.target_phe,
            _ => false,
        }
    }
}

// =============================================================================
// UV SPECTROSCOPY CONFIGURATION (Enhanced Multi-Wavelength)
// =============================================================================

/// UV Spectroscopy Configuration for full multi-wavelength pump-probe
///
/// Extends UvBiasConfig with:
/// - Frequency hopping protocol
/// - Disulfide bond targeting
/// - Local temperature tracking
/// - π→π* electronic state modeling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UvSpectroscopyConfig {
    /// Base UV bias config (backward compatible)
    pub base: UvBiasConfig,

    // =========================================================================
    // Frequency Hopping Protocol
    // =========================================================================
    /// Enable frequency hopping (wavelength scanning)
    pub frequency_hopping_enabled: bool,

    /// Wavelengths to scan (nm)
    pub scan_wavelengths: Vec<f32>,

    /// Dwell time per wavelength (MD steps)
    pub dwell_steps: u32,

    /// Number of full spectral scans
    pub n_scans: u32,

    // =========================================================================
    // Disulfide Bond Targeting
    // =========================================================================
    /// Enable disulfide bond (S-S) targeting at 250nm
    pub target_disulfides: bool,

    /// Maximum S-S bond distance for detection (Å)
    pub disulfide_max_distance: f32,

    // =========================================================================
    // Local Temperature Tracking
    // =========================================================================
    /// Enable local temperature tracking from photon absorption
    pub track_local_temperature: bool,

    /// Photon fluence for energy deposition calculation (photons/Å²)
    pub photon_fluence: f32,

    /// Thermal dissipation time constant (ps)
    pub thermal_dissipation_tau: f32,

    /// Number of atoms to include in local temperature calculation
    pub local_temp_shell_atoms: usize,

    // =========================================================================
    // Electronic State Modeling
    // =========================================================================
    /// Enable π→π* transition modeling
    pub model_electronic_transitions: bool,

    /// Excited state lifetime (ps) - vibrational relaxation
    pub excited_state_lifetime: f32,

    /// Fraction of energy deposited to ring atoms (vs dissipated)
    pub energy_deposition_fraction: f32,
}

impl Default for UvSpectroscopyConfig {
    fn default() -> Self {
        Self {
            base: UvBiasConfig::default(),

            // Frequency hopping - disabled by default for backward compat
            frequency_hopping_enabled: false,
            scan_wavelengths: vec![258.0, 265.0, 274.0, 280.0, 290.0],
            dwell_steps: 1000, // 2 ps per wavelength at 2 fs timestep
            n_scans: 5,

            // Disulfide targeting - disabled by default
            target_disulfides: false,
            disulfide_max_distance: DISULFIDE_BOND_MAX_DISTANCE,

            // Local temperature tracking - enabled
            track_local_temperature: true,
            photon_fluence: CALIBRATED_PHOTON_FLUENCE, // 0.024 photons/Å² (calibrated)
            thermal_dissipation_tau: 5.0,              // 5 ps decay
            local_temp_shell_atoms: 20,

            // Electronic state modeling - enabled
            model_electronic_transitions: true,
            excited_state_lifetime: 10.0, // 10 ps for vibrational relaxation
            energy_deposition_fraction: 0.8,
        }
    }
}

impl UvSpectroscopyConfig {
    /// Create config with full frequency hopping enabled
    pub fn with_frequency_hopping() -> Self {
        Self {
            frequency_hopping_enabled: true,
            ..Default::default()
        }
    }

    /// Create config with disulfide targeting enabled
    pub fn with_disulfides() -> Self {
        Self {
            target_disulfides: true,
            ..Default::default()
        }
    }

    /// Create full publication-quality config
    pub fn publication_quality() -> Self {
        Self {
            frequency_hopping_enabled: true,
            target_disulfides: true,
            track_local_temperature: true,
            model_electronic_transitions: true,
            n_scans: 10,
            ..Default::default()
        }
    }

    /// Get chromophore spec for a residue type at a wavelength
    pub fn get_chromophore_absorption(&self, residue: &str, wavelength: f32) -> f32 {
        let (lambda_max, epsilon_max, bandwidth) = match residue.to_uppercase().as_str() {
            "TRP" | "W" => (TRP_LAMBDA_MAX, TRP_EXTINCTION_280, TRP_BANDWIDTH),
            "TYR" | "Y" => (TYR_LAMBDA_MAX, TYR_EXTINCTION_274, TYR_BANDWIDTH),
            "PHE" | "F" => (PHE_LAMBDA_MAX, PHE_EXTINCTION_258, PHE_BANDWIDTH),
            "CYS" | "C" | "CYX" => (
                DISULFIDE_LAMBDA_MAX,
                DISULFIDE_EXTINCTION_250,
                DISULFIDE_BANDWIDTH,
            ),
            _ => return 0.0,
        };

        // Gaussian absorption profile
        let delta = wavelength - lambda_max;
        let sigma = band_sigma(bandwidth);
        epsilon_max * (-0.5 * (delta / sigma).powi(2)).exp()
    }

    /// Get current wavelength for frequency hopping at given step
    pub fn current_wavelength(&self, step: u64) -> f32 {
        if !self.frequency_hopping_enabled || self.scan_wavelengths.is_empty() {
            return 280.0; // Default to 280nm
        }

        let scan_length = self.scan_wavelengths.len() as u64 * self.dwell_steps as u64;
        let position = ((step % scan_length) / self.dwell_steps as u64) as usize;
        self.scan_wavelengths[position % self.scan_wavelengths.len()]
    }

    /// Compute local temperature increase from photon absorption (PHYSICS-CORRECTED)
    ///
    /// Formula: ΔT = (E_γ × p × η) / (3/2 × k_B × N_eff)
    /// where: p = σ × F (absorption probability)
    ///        σ = ε × 3.823×10⁻⁵ (cross-section in Å²)
    ///        F = photon fluence (photons/Å²)
    ///        η = heat yield (fraction → heat)
    ///
    /// Calibration: TRP @ 280nm with F=0.024, η=1.0 → ΔT ≈ 20K
    pub fn compute_local_heating(&self, wavelength: f32, extinction: f32, n_eff: f32) -> f32 {
        let photon_energy = wavelength_to_ev(wavelength); // eV

        // CORRECTED: Proper ε → σ conversion (per molecule)
        let sigma = extinction_to_cross_section(extinction); // Å²

        // Absorption probability (single-photon regime: p << 1)
        let p_absorb = sigma * self.photon_fluence;

        // Warn if exceeding single-photon regime (p should be << 0.01)
        #[cfg(debug_assertions)]
        if p_absorb > MAX_ABSORPTION_PROBABILITY {
            log::warn!(
                "Absorption probability {:.4} exceeds single-photon regime threshold {}",
                p_absorb,
                MAX_ABSORPTION_PROBABILITY
            );
        }

        // Energy deposited per chromophore with heat yield
        let heat_yield = self.energy_deposition_fraction; // η
        let energy_deposited = photon_energy * p_absorb * heat_yield; // eV

        // Convert to temperature increase via equipartition
        // ΔT = E_dep / (3/2 × k_B × N_eff)
        energy_deposited / (1.5 * KB_EV_K * n_eff) // Kelvin
    }

    /// Compute local heating for a specific residue type (convenience method)
    pub fn compute_local_heating_for_residue(&self, residue: &str, wavelength: f32) -> f32 {
        let (extinction, n_eff) = match residue.to_uppercase().as_str() {
            "TRP" | "W" => {
                let ext = self.get_chromophore_absorption("TRP", wavelength);
                (ext, NEFF_TRP)
            }
            "TYR" | "Y" => {
                let ext = self.get_chromophore_absorption("TYR", wavelength);
                (ext, NEFF_TYR)
            }
            "PHE" | "F" => {
                let ext = self.get_chromophore_absorption("PHE", wavelength);
                (ext, NEFF_PHE)
            }
            "CYS" | "C" | "CYX" => {
                let ext = self.get_chromophore_absorption("CYS", wavelength);
                (ext, NEFF_DISULFIDE)
            }
            _ => return 0.0,
        };
        self.compute_local_heating(wavelength, extinction, n_eff)
    }
}

// =============================================================================
// HYDROPHOBICITY THRESHOLDS
// =============================================================================

/// Hydrophobic atom classification thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HydrophobicityThresholds {
    /// Residues with hydrophobicity > this are "hydrophobic"
    pub hydrophobic_cutoff: f32,

    /// Residues with hydrophobicity < this are "hydrophilic"
    pub hydrophilic_cutoff: f32,
}

impl Default for HydrophobicityThresholds {
    fn default() -> Self {
        Self {
            hydrophobic_cutoff: 0.6, // ILE, LEU, VAL, PHE, MET, etc.
            hydrophilic_cutoff: 0.4, // ARG, LYS, ASP, GLU, etc.
        }
    }
}

// =============================================================================
// RT CORE INTEGRATION - SOLVENT MODE
// =============================================================================
// [STAGE-1-CONFIG]

/// Solvent treatment mode for molecular dynamics simulation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SolventMode {
    /// Implicit solvent (Cryo-UV-LIF mean-field model)
    /// - Fast: ~1 μs/day throughput
    /// - No explicit water molecules
    /// - Good for broad conformational exploration
    Implicit,

    /// Explicit solvent (TIP3P water model)
    /// - High fidelity: ~50-100 ns/day throughput
    /// - Full water box with specified padding
    /// - Required for detailed solvation dynamics
    Explicit {
        /// Padding around protein bounding box (Angstroms)
        /// Typical: 10-15 Å for adequate solvation shell
        padding_angstroms: f32,
    },

    /// Hybrid mode: start implicit, switch to explicit for interesting regions
    /// - Adaptive: mostly fast, occasionally detailed
    /// - Best for efficient discovery + validation
    /// - Switches when geometric voids detected by RT probes
    Hybrid {
        /// Number of steps to run in implicit exploration phase
        exploration_steps: i32,

        /// Number of steps to run in explicit characterization phase (per region)
        characterization_steps: i32,

        /// RMSD drift threshold (Å) to trigger explicit solvation
        /// When conformational change exceeds this, switch to explicit
        switch_threshold: f32,
    },
}

impl Default for SolventMode {
    fn default() -> Self {
        Self::Implicit
    }
}

impl SolventMode {
    /// Validate configuration parameters
    pub fn validate(&self) -> anyhow::Result<()> {
        match self {
            SolventMode::Implicit => Ok(()),
            SolventMode::Explicit { padding_angstroms } => {
                if *padding_angstroms <= 0.0 {
                    anyhow::bail!(
                        "Explicit solvent padding must be > 0, got {}",
                        padding_angstroms
                    );
                }
                if *padding_angstroms < 8.0 {
                    log::warn!(
                        "Explicit solvent padding {}Å is small (recommended: ≥10Å)",
                        padding_angstroms
                    );
                }
                Ok(())
            }
            SolventMode::Hybrid {
                exploration_steps,
                characterization_steps,
                switch_threshold,
            } => {
                if *exploration_steps <= 0 {
                    anyhow::bail!(
                        "Hybrid exploration_steps must be > 0, got {}",
                        exploration_steps
                    );
                }
                if *characterization_steps <= 0 {
                    anyhow::bail!(
                        "Hybrid characterization_steps must be > 0, got {}",
                        characterization_steps
                    );
                }
                if *switch_threshold <= 0.0 || *switch_threshold > 1.0 {
                    anyhow::bail!(
                        "Hybrid switch_threshold must be in range (0.0, 1.0], got {}",
                        switch_threshold
                    );
                }
                Ok(())
            }
        }
    }

    /// Check if this mode requires explicit water molecules
    pub fn requires_water(&self) -> bool {
        matches!(
            self,
            SolventMode::Explicit { .. } | SolventMode::Hybrid { .. }
        )
    }

    /// Check if this mode starts with explicit solvent
    pub fn starts_explicit(&self) -> bool {
        matches!(self, SolventMode::Explicit { .. })
    }
}

// =============================================================================
// RT CORE INTEGRATION - PROBE CONFIGURATION
// =============================================================================
// [STAGE-1-CONFIG]

/// Configuration for RT core spatial probing system
/// Uses RTX 5080's 84 dedicated RT cores for real-time spatial sensing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RtProbeConfig {
    /// Enable RT probe system (requires OptiX and compatible GPU)
    pub enabled: bool,

    /// Steps between probe bursts (e.g., 100 = probe every 100 MD steps)
    /// Lower = more frequent probing, higher overhead
    /// Higher = less frequent, lower overhead
    /// Typical: 50-200 steps
    pub probe_interval: i32,

    /// Number of rays to fire from each attention point
    /// More rays = better coverage, higher cost
    /// Typical: 128-512 rays (uniform sphere sampling)
    pub rays_per_point: usize,

    /// Number of attention points (probe origins distributed around protein)
    /// More points = better spatial coverage
    /// Typical: 20-100 points
    pub attention_points: usize,

    /// Displacement threshold (Å) to trigger BVH refit
    /// When max atomic displacement exceeds this, BVH is refitted
    /// Typical: 0.3-0.7 Å (thermal fluctuation scale)
    pub bvh_refit_threshold: f32,

    /// Enable solvent probing (only for explicit solvent mode)
    /// Fires rays through BVH-solvent to detect water disruption
    /// Leading indicator for pocket opening (100-500 fs early signal)
    pub enable_solvent_probes: bool,

    /// Enable aromatic LIF (Laser-Induced Fluorescence) probing
    /// Detects spatial excitation patterns around aromatic residues
    /// Correlates with UV burst events
    pub enable_aromatic_lif: bool,
}

impl Default for RtProbeConfig {
    fn default() -> Self {
        Self {
            enabled: true, // RT cores are available, use them!
            probe_interval: 100,
            rays_per_point: 256,
            attention_points: 50,
            bvh_refit_threshold: 0.5,
            enable_solvent_probes: false, // Only for explicit mode
            enable_aromatic_lif: true,    // Always useful
        }
    }
}

impl RtProbeConfig {
    /// Validate configuration parameters
    pub fn validate(&self) -> anyhow::Result<()> {
        if !self.enabled {
            return Ok(()); // No validation needed if disabled
        }

        if self.probe_interval <= 0 {
            anyhow::bail!("RT probe_interval must be > 0, got {}", self.probe_interval);
        }

        if self.rays_per_point == 0 {
            anyhow::bail!("RT rays_per_point must be > 0, got {}", self.rays_per_point);
        }

        if self.rays_per_point > 1024 {
            log::warn!(
                "RT rays_per_point {} is very high (>1024 rays), may impact performance",
                self.rays_per_point
            );
        }

        if self.attention_points == 0 {
            anyhow::bail!(
                "RT attention_points must be > 0, got {}",
                self.attention_points
            );
        }

        if self.bvh_refit_threshold <= 0.0 {
            anyhow::bail!(
                "RT bvh_refit_threshold must be > 0, got {}",
                self.bvh_refit_threshold
            );
        }

        if self.bvh_refit_threshold > 2.0 {
            log::warn!(
                "RT bvh_refit_threshold {}Å is very large (>2Å), BVH may become stale",
                self.bvh_refit_threshold
            );
        }

        Ok(())
    }

    /// Estimate performance overhead as fraction of MD time
    /// Returns approximate overhead (0.0 = no overhead, 0.1 = 10% overhead)
    pub fn estimate_overhead(&self) -> f32 {
        if !self.enabled {
            return 0.0;
        }

        // Baseline: 50μs per probe burst for BVH-protein only
        let base_cost_us = 50.0;

        // Solvent probing adds ~150μs (larger BVH)
        let solvent_cost_us = if self.enable_solvent_probes {
            150.0
        } else {
            0.0
        };

        // Ray cost scales with ray count (roughly linear)
        let ray_cost_us = (self.rays_per_point as f32) * (self.attention_points as f32) * 0.0001;

        // Total probe cost per burst
        let total_cost_us = base_cost_us + solvent_cost_us + ray_cost_us;

        // MD timestep cost (typical: 1000-2000 μs per step)
        let md_step_cost_us = 1500.0;

        // Overhead = probe_cost / (interval * step_cost)
        let overhead = total_cost_us / ((self.probe_interval as f32) * md_step_cost_us);

        overhead
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_absorption_strengths() {
        let config = UvBiasConfig::default();

        // Trp should be strongest (normalized to 1.0)
        assert!((config.absorption_strength("TRP") - 1.0).abs() < 0.01);

        // Tyr should be ~0.25
        assert!((config.absorption_strength("TYR") - 0.25).abs() < 0.05);

        // Phe should be ~0.036
        assert!(config.absorption_strength("PHE") < 0.1);

        // Water should be 0
        assert_eq!(config.absorption_strength("HOH"), 0.0);

        // Alanine should be 0
        assert_eq!(config.absorption_strength("ALA"), 0.0);
    }

    #[test]
    fn test_target_selection() {
        let config = UvBiasConfig::default();

        assert!(config.is_valid_target("TRP"));
        assert!(config.is_valid_target("TYR"));
        assert!(!config.is_valid_target("PHE")); // Disabled by default
        assert!(!config.is_valid_target("ALA"));
    }

    #[test]
    fn test_water_transparency() {
        // This is the key physical insight
        assert_eq!(WATER_EXTINCTION_280, 0.0);
    }

    // =========================================================================
    // PHYSICS CHAIN VERIFICATION TESTS
    // =========================================================================

    #[test]
    fn test_physics_constants_exact_values() {
        // Verify all physics constants match the calibration spec exactly

        // ε → σ conversion factor
        assert!(
            (EPSILON_TO_SIGMA_CONVERSION - 3.823e-5).abs() < 1e-9,
            "EPSILON_TO_SIGMA_CONVERSION wrong"
        );

        // Calibrated fluence
        assert!(
            (CALIBRATED_PHOTON_FLUENCE - 0.024).abs() < 0.001,
            "CALIBRATED_PHOTON_FLUENCE wrong"
        );

        // Fluence in mJ/cm² (must be ~0.170, NOT 0.017)
        assert!(
            (CALIBRATED_FLUENCE_MJ_CM2_280NM - 0.170).abs() < 0.01,
            "CALIBRATED_FLUENCE_MJ_CM2_280NM wrong: got {}, expected ~0.170",
            CALIBRATED_FLUENCE_MJ_CM2_280NM
        );

        // Heat yield
        assert!(
            (DEFAULT_HEAT_YIELD - 1.0).abs() < 0.01,
            "DEFAULT_HEAT_YIELD wrong"
        );

        // N_eff values
        assert!((NEFF_TRP - 9.0).abs() < 0.1, "NEFF_TRP wrong");
        assert!((NEFF_TYR - 10.0).abs() < 0.1, "NEFF_TYR wrong");
        assert!((NEFF_PHE - 9.0).abs() < 0.1, "NEFF_PHE wrong");
        assert!((NEFF_DISULFIDE - 2.0).abs() < 0.1, "NEFF_DISULFIDE wrong");

        // Boltzmann constant
        assert!((KB_EV_K - 8.617e-5).abs() < 1e-8, "KB_EV_K wrong");
    }

    #[test]
    fn test_physics_chain_step_by_step() {
        // Complete physics chain verification for TRP @ 280nm
        // This test validates each step independently

        // Step 1: Extinction coefficient
        let epsilon = TRP_EXTINCTION_280;
        assert!((epsilon - 5600.0).abs() < 1.0, "ε_TRP wrong");

        // Step 2: Cross-section conversion
        // σ(Å²) = ε(M⁻¹cm⁻¹) × 3.823×10⁻⁵
        let sigma = extinction_to_cross_section(epsilon);
        let expected_sigma = 5600.0 * 3.823e-5;
        assert!(
            (sigma - expected_sigma).abs() < 1e-6,
            "σ calculation wrong: got {}, expected {}",
            sigma,
            expected_sigma
        );
        assert!(
            (sigma - 0.21409).abs() < 0.001,
            "σ_TRP wrong: got {}, expected 0.21409",
            sigma
        );

        // Step 3: Photon fluence
        let fluence = CALIBRATED_PHOTON_FLUENCE;
        assert!((fluence - 0.024).abs() < 0.001, "Fluence wrong");

        // Step 4: Absorption probability
        // p = σ × F
        let p_absorb = sigma * fluence;
        let expected_p = 0.21409 * 0.024;
        assert!(
            (p_absorb - expected_p).abs() < 1e-6,
            "p_absorb calculation wrong"
        );
        assert!(
            (p_absorb - 0.00514).abs() < 0.0001,
            "p_absorb wrong: got {}, expected ~0.00514",
            p_absorb
        );

        // Verify single-photon regime
        assert!(
            p_absorb < MAX_ABSORPTION_PROBABILITY,
            "p_absorb {} exceeds single-photon threshold {}",
            p_absorb,
            MAX_ABSORPTION_PROBABILITY
        );

        // Step 5: Photon energy
        // E = hc/λ = 1239.84 / λ(nm)
        let e_photon = wavelength_to_ev(280.0);
        let expected_e = 1239.84 / 280.0;
        assert!(
            (e_photon - expected_e).abs() < 0.001,
            "E_photon calculation wrong"
        );
        assert!(
            (e_photon - 4.428).abs() < 0.01,
            "E_photon wrong: got {}, expected ~4.428",
            e_photon
        );

        // Step 6: Heat yield
        let eta = HEAT_YIELD_TRP; // Physics-corrected
        assert!((eta - 0.74).abs() < 0.01, "η_TRP wrong: expected 0.74");

        // Step 7: Energy deposited
        // E_dep = E_γ × p × η
        let e_dep = e_photon * p_absorb * eta;
        let expected_e_dep = 4.428 * 0.00514 * 0.74;
        assert!(
            (e_dep - expected_e_dep).abs() < 0.001,
            "E_dep calculation wrong: got {}, expected {}",
            e_dep,
            expected_e_dep
        );
        assert!(
            (e_dep - 0.01687).abs() < 0.001,
            "E_dep wrong: got {}, expected ~0.01687 (η=0.74)",
            e_dep
        );

        // Step 8: Effective DOF
        let n_eff = NEFF_TRP;
        assert!((n_eff - 9.0).abs() < 0.1, "N_eff wrong");

        // Step 9: Temperature rise
        // ΔT = E_dep / (3/2 × k_B × N_eff)
        let delta_t = e_dep / (1.5 * KB_EV_K * n_eff);
        let expected_dt = 0.01687 / (1.5 * 8.617e-5 * 9.0);
        assert!(
            (delta_t - expected_dt).abs() < 0.5,
            "ΔT calculation wrong: got {}, expected {}",
            delta_t,
            expected_dt
        );
        assert!(
            (delta_t - 14.5).abs() < 1.0,
            "CALIBRATION FAILED: ΔT = {} K, expected ~14.5 K (η=0.74)",
            delta_t
        );

        println!("=== Physics Chain Verification PASSED ===");
        println!("ε = {:.0} M⁻¹cm⁻¹", epsilon);
        println!("σ = {:.5} Å²", sigma);
        println!("F = {:.4} photons/Å²", fluence);
        println!("p = {:.5}", p_absorb);
        println!("E_γ = {:.3} eV", e_photon);
        println!("η = {:.1}", eta);
        println!("E_dep = {:.5} eV", e_dep);
        println!("N_eff = {:.1}", n_eff);
        println!("ΔT = {:.2} K ✓", delta_t);
    }

    #[test]
    fn test_fluence_unit_conversion() {
        // Verify the mJ/cm² conversion is correct
        // At 280nm: 1 photon/Å² ≈ 7.09 mJ/cm²

        let wavelength = 280.0;
        let e_photon_ev = wavelength_to_ev(wavelength); // 4.428 eV
        let e_photon_j = e_photon_ev * 1.602e-19; // Convert to Joules

        // 1 photon/Å² = 10¹⁶ photons/cm² (conversion factor)
        // Energy density = 10¹⁶ × E_photon(J) J/cm²
        // Convert to mJ/cm²: multiply by 1000
        let mj_per_cm2_per_photon_per_a2 = 1e16 * e_photon_j * 1e3;

        // Should be approximately 7.09 mJ/cm² per (photon/Å²)
        assert!(
            (mj_per_cm2_per_photon_per_a2 - 7.09).abs() < 0.1,
            "Unit conversion wrong: 1 photon/Å² = {} mJ/cm², expected ~7.09",
            mj_per_cm2_per_photon_per_a2
        );

        // Verify calibrated fluence conversion
        let calibrated_mj_cm2 = CALIBRATED_PHOTON_FLUENCE * mj_per_cm2_per_photon_per_a2;
        assert!(
            (calibrated_mj_cm2 - CALIBRATED_FLUENCE_MJ_CM2_280NM).abs() < 0.02,
            "Calibrated fluence conversion mismatch: {} vs {} mJ/cm²",
            calibrated_mj_cm2,
            CALIBRATED_FLUENCE_MJ_CM2_280NM
        );

        println!("=== Fluence Unit Conversion PASSED ===");
        println!(
            "1 photon/Å² @ 280nm = {:.2} mJ/cm²",
            mj_per_cm2_per_photon_per_a2
        );
        println!("0.024 photon/Å² @ 280nm = {:.3} mJ/cm²", calibrated_mj_cm2);
    }

    // =========================================================================
    // GPU PARITY REGRESSION TESTS
    // =========================================================================
    // These tests verify that the GPU constants (in nhs_excited_state.cuh)
    // match the CPU constants. If you modify the CUDA headers, update these
    // expected values and ensure they match exactly.

    #[test]
    fn test_gpu_parity_cross_sections() {
        // GPU constants from nhs_excited_state.cuh (MUST MATCH)
        // #define UV_SIGMA_TRP  0.21409f
        // #define UV_SIGMA_TYR  0.05696f
        // #define UV_SIGMA_PHE  0.00765f
        // #define UV_SIGMA_SS   0.01147f

        let gpu_sigma_trp = 0.21409_f32;
        let gpu_sigma_tyr = 0.05696_f32;
        let gpu_sigma_phe = 0.00765_f32;
        let gpu_sigma_ss = 0.01147_f32;

        // CPU computed values
        let cpu_sigma_trp = extinction_to_cross_section(TRP_EXTINCTION_280);
        let cpu_sigma_tyr = extinction_to_cross_section(TYR_EXTINCTION_274);
        let cpu_sigma_phe = extinction_to_cross_section(PHE_EXTINCTION_258);
        let cpu_sigma_ss = extinction_to_cross_section(DISULFIDE_EXTINCTION_250);

        // Verify GPU matches CPU
        assert!(
            (gpu_sigma_trp - cpu_sigma_trp).abs() < 0.0001,
            "GPU/CPU σ_TRP mismatch: GPU={}, CPU={}",
            gpu_sigma_trp,
            cpu_sigma_trp
        );
        assert!(
            (gpu_sigma_tyr - cpu_sigma_tyr).abs() < 0.0001,
            "GPU/CPU σ_TYR mismatch: GPU={}, CPU={}",
            gpu_sigma_tyr,
            cpu_sigma_tyr
        );
        assert!(
            (gpu_sigma_phe - cpu_sigma_phe).abs() < 0.0001,
            "GPU/CPU σ_PHE mismatch: GPU={}, CPU={}",
            gpu_sigma_phe,
            cpu_sigma_phe
        );
        assert!(
            (gpu_sigma_ss - cpu_sigma_ss).abs() < 0.0001,
            "GPU/CPU σ_SS mismatch: GPU={}, CPU={}",
            gpu_sigma_ss,
            cpu_sigma_ss
        );

        println!("=== GPU Parity (Cross-Sections) PASSED ===");
    }

    #[test]
    fn test_gpu_parity_fluence_and_conversion() {
        // GPU constants from nhs_excited_state.cuh (MUST MATCH)
        // #define EPSILON_TO_SIGMA_FACTOR  3.823e-5f
        // #define CALIBRATED_PHOTON_FLUENCE  0.024f

        let gpu_conversion = 3.823e-5_f32;
        let gpu_fluence = 0.024_f32;

        assert!(
            (gpu_conversion - EPSILON_TO_SIGMA_CONVERSION).abs() < 1e-9,
            "GPU/CPU ε→σ conversion mismatch: GPU={}, CPU={}",
            gpu_conversion,
            EPSILON_TO_SIGMA_CONVERSION
        );
        assert!(
            (gpu_fluence - CALIBRATED_PHOTON_FLUENCE).abs() < 0.0001,
            "GPU/CPU fluence mismatch: GPU={}, CPU={}",
            gpu_fluence,
            CALIBRATED_PHOTON_FLUENCE
        );

        println!("=== GPU Parity (Conversion/Fluence) PASSED ===");
    }

    #[test]
    fn test_gpu_parity_neff_values() {
        // GPU constants from nhs_excited_state.cuh (MUST MATCH)
        // #define NEFF_TRP  9.0f
        // #define NEFF_TYR  10.0f
        // #define NEFF_PHE  9.0f
        // #define NEFF_SS   2.0f

        let gpu_neff_trp = 9.0_f32;
        let gpu_neff_tyr = 10.0_f32;
        let gpu_neff_phe = 9.0_f32;
        let gpu_neff_ss = 2.0_f32;

        assert!(
            (gpu_neff_trp - NEFF_TRP).abs() < 0.1,
            "GPU/CPU N_eff TRP mismatch"
        );
        assert!(
            (gpu_neff_tyr - NEFF_TYR).abs() < 0.1,
            "GPU/CPU N_eff TYR mismatch"
        );
        assert!(
            (gpu_neff_phe - NEFF_PHE).abs() < 0.1,
            "GPU/CPU N_eff PHE mismatch"
        );
        assert!(
            (gpu_neff_ss - NEFF_DISULFIDE).abs() < 0.1,
            "GPU/CPU N_eff SS mismatch"
        );

        println!("=== GPU Parity (N_eff Values) PASSED ===");
    }

    #[test]
    fn test_regression_wrong_conversion_factor() {
        // REGRESSION TEST: Catch if someone reintroduces ε/1000 instead of ε × 3.823e-5
        // The WRONG conversion was: σ = ε / 1000 = 5.6 (for TRP)
        // The CORRECT conversion is: σ = ε × 3.823e-5 = 0.21409 (for TRP)

        let wrong_sigma = TRP_EXTINCTION_280 / 1000.0; // 5.6 - WRONG
        let correct_sigma = extinction_to_cross_section(TRP_EXTINCTION_280); // 0.21409

        // This test MUST fail if someone uses the wrong conversion
        assert!(
            (correct_sigma - 0.21409).abs() < 0.001,
            "REGRESSION: extinction_to_cross_section is returning {}, should be ~0.21409",
            correct_sigma
        );
        assert!(
            (correct_sigma - wrong_sigma).abs() > 5.0,
            "REGRESSION: extinction_to_cross_section appears to use ε/1000 instead of ε×3.823e-5"
        );

        println!("=== Regression Test (Wrong Conversion) PASSED ===");
    }

    #[test]
    fn test_regression_wrong_fluence_units() {
        // REGRESSION TEST: Catch if someone uses 0.017 instead of 0.170 mJ/cm²
        // The WRONG value was: 0.017 mJ/cm² (10× too small)
        // The CORRECT value is: 0.170 mJ/cm² (≈ 0.024 photons/Å² × 7.09 mJ·cm⁻²·(photon/Å²)⁻¹)

        let wrong_fluence_mj = 0.017_f32;
        let correct_fluence_mj = CALIBRATED_FLUENCE_MJ_CM2_280NM;

        // This test MUST fail if someone uses the wrong fluence
        assert!(
            (correct_fluence_mj - 0.170).abs() < 0.02,
            "REGRESSION: CALIBRATED_FLUENCE_MJ_CM2_280NM is {}, should be ~0.170",
            correct_fluence_mj
        );
        assert!(
            (correct_fluence_mj - wrong_fluence_mj).abs() > 0.1,
            "REGRESSION: Fluence appears to be 10× too small (0.017 instead of 0.170)"
        );

        println!("=== Regression Test (Wrong Fluence Units) PASSED ===");
    }

    // =========================================================================
    // WAVELENGTH-DEPENDENT σ(λ) TESTS
    // =========================================================================
    // These tests verify the Gaussian band model for wavelength-dependent
    // absorption cross-sections and the resulting local heating.

    #[test]
    fn test_wavelength_dependent_trp_280nm() {
        // TRP @ 280nm (peak) → ~19.56K
        // This is the calibration point for the physics chain
        let delta_t = compute_heating_at_wavelength(0, 280.0); // 0 = TRP

        // Should be approximately 19.6K at peak
        assert!(
            (delta_t - 14.5).abs() < 1.5,
            "TRP@280nm ΔT wrong: expected ~14.5K (η=0.74), got {:.2}K",
            delta_t
        );

        println!("=== Wavelength Test: TRP@280nm ===");
        println!("ΔT = {:.2} K (expected ~14.5 K (η=0.74))", delta_t);
    }

    #[test]
    fn test_wavelength_dependent_trp_258nm() {
        // TRP @ 258nm (off-peak, near the PHE peak) must be strongly suppressed
        // relative to the TRP peak at 280nm.
        //
        // P6: this test previously asserted `ΔT < 1.0 K`, a bound that encoded
        // one particular σ rather than a physical property — its own comment
        // read "With σ=15nm/2.355≈6.4nm". That conversion was the defect P6
        // removed. Under the canonical band model (σ = w/√2 = 10.607nm) the
        // suppression at 258nm is ~8.6x rather than ~390x, so the old numeric
        // bound no longer holds and should not: it was a symptom of a band that
        // was 1.67x too narrow.
        //
        // The invariant actually being tested is the *ratio*, which is
        // independent of the heating calibration, so that is what is asserted —
        // together with an exact tie to the Gaussian, so the heating chain
        // cannot drift away from the band model it is supposed to use.
        let delta_t = compute_heating_at_wavelength(0, 258.0); // 0 = TRP
        let delta_t_peak = compute_heating_at_wavelength(0, TRP_LAMBDA_MAX);
        let ratio = delta_t / delta_t_peak;

        // Expected suppression is the Gaussian band factor times the photon
        // energy ratio: ΔT ∝ ε(λ) · E_γ(λ) and E_γ = hc/λ, so a 258nm photon
        // deposits 280/258 = 1.0853x more heat per absorption event than a
        // 280nm one. Pinning the product ties the whole deposition chain —
        // band model *and* photon energy — not just the band shape.
        let d = 258.0_f32 - TRP_LAMBDA_MAX; // -22 nm
        let sigma = band_sigma(TRP_BANDWIDTH);
        let band_factor = (-(d * d) / (2.0 * sigma * sigma)).exp();
        let photon_energy_ratio = TRP_LAMBDA_MAX / 258.0;
        let expected = band_factor * photon_energy_ratio;
        assert!(
            (ratio - expected).abs() < 1e-3,
            "TRP@258nm suppression {ratio:.6} does not match ε(λ)·E_γ(λ) \
             {expected:.6} (band {band_factor:.6} x photon \
             {photon_energy_ratio:.6}); the heating chain and the band model \
             have diverged"
        );

        // The property the original bound was reaching for: 258nm is decisively
        // off-peak for TRP, not merely slightly reduced.
        assert!(
            ratio < 0.25,
            "TRP@258nm is not strongly off-peak: ratio {ratio:.4} of the 280nm peak"
        );

        println!("=== Wavelength Test: TRP@258nm ===");
        println!(
            "ΔT = {delta_t:.3} K vs peak {delta_t_peak:.3} K (ratio {ratio:.4}, {:.1}x suppressed)",
            1.0 / ratio
        );
    }

    #[test]
    fn test_wavelength_dependent_phe_selectivity() {
        // PHE @ 258nm (peak) vs PHE @ 280nm (off-peak)
        // PHE λ_max = 258nm, so absorption should be stronger at 258nm
        let delta_t_258 = compute_heating_at_wavelength(2, 258.0); // 2 = PHE at peak
        let delta_t_280 = compute_heating_at_wavelength(2, 280.0); // 2 = PHE off-peak

        // PHE@258nm should be stronger than PHE@280nm
        assert!(
            delta_t_258 > delta_t_280,
            "PHE selectivity wrong: PHE@258nm ({:.2}K) should be > PHE@280nm ({:.2}K)",
            delta_t_258,
            delta_t_280
        );

        // The ratio should be significant (> 2x)
        let ratio = delta_t_258 / delta_t_280.max(0.001); // Avoid div by zero
        assert!(
            ratio > 2.0,
            "PHE selectivity ratio too low: expected > 2x, got {:.1}x",
            ratio
        );

        println!("=== Wavelength Test: PHE Selectivity ===");
        println!("PHE@258nm = {:.3} K", delta_t_258);
        println!("PHE@280nm = {:.3} K", delta_t_280);
        println!("Ratio = {:.1}x", ratio);
    }

    #[test]
    fn test_wavelength_dependent_chromophore_selectivity() {
        // At 280nm, TRP should dominate over PHE and TYR
        let trp_280 = compute_heating_at_wavelength(0, 280.0); // TRP
        let tyr_280 = compute_heating_at_wavelength(1, 280.0); // TYR (peak is 274nm)
        let phe_280 = compute_heating_at_wavelength(2, 280.0); // PHE (peak is 258nm)

        // TRP should be strongest at 280nm
        assert!(
            trp_280 > tyr_280,
            "At 280nm: TRP ({:.2}K) should be > TYR ({:.2}K)",
            trp_280,
            tyr_280
        );
        assert!(
            trp_280 > phe_280,
            "At 280nm: TRP ({:.2}K) should be > PHE ({:.3}K)",
            trp_280,
            phe_280
        );

        // At 258nm, PHE should be enhanced relative to TRP
        let trp_258 = compute_heating_at_wavelength(0, 258.0);
        let phe_258 = compute_heating_at_wavelength(2, 258.0);

        // PHE/TRP ratio should be much higher at 258nm than at 280nm
        let ratio_258 = phe_258 / trp_258.max(0.001);
        let ratio_280 = phe_280 / trp_280.max(0.001);

        assert!(
            ratio_258 > ratio_280,
            "PHE selectivity should increase at 258nm: ratio@258={:.3}, ratio@280={:.4}",
            ratio_258,
            ratio_280
        );

        println!("=== Wavelength Test: Chromophore Selectivity ===");
        println!(
            "At 280nm: TRP={:.2}K, TYR={:.2}K, PHE={:.4}K",
            trp_280, tyr_280, phe_280
        );
        println!("At 258nm: TRP={:.3}K, PHE={:.4}K", trp_258, phe_258);
        println!(
            "PHE/TRP ratio: @258nm={:.3}, @280nm={:.5}",
            ratio_258, ratio_280
        );
    }

    #[test]
    fn test_gaussian_band_model_symmetry() {
        // Gaussian band should be symmetric around λ_max
        // Test TRP at λ_max ± Δλ
        let delta = 10.0; // nm

        let trp_peak = compute_heating_at_wavelength(0, TRP_LAMBDA_MAX);
        let trp_plus = compute_heating_at_wavelength(0, TRP_LAMBDA_MAX + delta);
        let trp_minus = compute_heating_at_wavelength(0, TRP_LAMBDA_MAX - delta);

        // Peak should be strongest
        assert!(
            trp_peak > trp_plus,
            "TRP peak ({:.2}K) should be > TRP+{}nm ({:.2}K)",
            trp_peak,
            delta,
            trp_plus
        );
        assert!(
            trp_peak > trp_minus,
            "TRP peak ({:.2}K) should be > TRP-{}nm ({:.2}K)",
            trp_peak,
            delta,
            trp_minus
        );

        // Symmetric offsets should give similar (but not identical due to E_γ = hc/λ)
        // The difference comes from photon energy variation, not the Gaussian
        let relative_diff = (trp_plus - trp_minus).abs() / trp_peak;
        assert!(
            relative_diff < 0.15,
            "Gaussian asymmetry too large: {:.1}%",
            relative_diff * 100.0
        );

        println!("=== Wavelength Test: Gaussian Symmetry ===");
        println!("TRP@{:.0}nm = {:.2} K", TRP_LAMBDA_MAX, trp_peak);
        println!("TRP@{:.0}nm = {:.2} K", TRP_LAMBDA_MAX + delta, trp_plus);
        println!("TRP@{:.0}nm = {:.2} K", TRP_LAMBDA_MAX - delta, trp_minus);
        println!("Relative diff = {:.1}%", relative_diff * 100.0);
    }

    // =========================================================================
    // RT INTEGRATION TESTS - [STAGE-1-CONFIG]
    // =========================================================================

    #[test]
    fn test_solvent_mode_default() {
        let mode = SolventMode::default();
        assert_eq!(mode, SolventMode::Implicit);
        assert!(!mode.requires_water());
        assert!(!mode.starts_explicit());
    }

    #[test]
    fn test_solvent_mode_validation() {
        // Implicit: always valid
        assert!(SolventMode::Implicit.validate().is_ok());

        // Explicit: valid with positive padding
        assert!(SolventMode::Explicit {
            padding_angstroms: 10.0
        }
        .validate()
        .is_ok());

        // Explicit: invalid with zero padding
        assert!(SolventMode::Explicit {
            padding_angstroms: 0.0
        }
        .validate()
        .is_err());

        // Explicit: invalid with negative padding
        assert!(SolventMode::Explicit {
            padding_angstroms: -5.0
        }
        .validate()
        .is_err());

        // Hybrid: valid with positive values
        assert!(SolventMode::Hybrid {
            exploration_steps: 100000,
            characterization_steps: 10000,
            switch_threshold: 0.5,
        }
        .validate()
        .is_ok());

        // Hybrid: invalid with zero exploration steps
        assert!(SolventMode::Hybrid {
            exploration_steps: 0,
            characterization_steps: 10000,
            switch_threshold: 0.5,
        }
        .validate()
        .is_err());

        // Hybrid: invalid with negative characterization steps
        assert!(SolventMode::Hybrid {
            exploration_steps: 100000,
            characterization_steps: -1,
            switch_threshold: 0.5,
        }
        .validate()
        .is_err());

        // Hybrid: invalid with zero switch threshold
        assert!(SolventMode::Hybrid {
            exploration_steps: 100000,
            characterization_steps: 10000,
            switch_threshold: 0.0,
        }
        .validate()
        .is_err());
    }

    #[test]
    fn test_solvent_mode_water_requirements() {
        assert!(!SolventMode::Implicit.requires_water());
        assert!(SolventMode::Explicit {
            padding_angstroms: 10.0
        }
        .requires_water());
        assert!(SolventMode::Hybrid {
            exploration_steps: 100000,
            characterization_steps: 10000,
            switch_threshold: 0.5,
        }
        .requires_water());

        assert!(!SolventMode::Implicit.starts_explicit());
        assert!(SolventMode::Explicit {
            padding_angstroms: 10.0
        }
        .starts_explicit());
        assert!(!SolventMode::Hybrid {
            exploration_steps: 100000,
            characterization_steps: 10000,
            switch_threshold: 0.5,
        }
        .starts_explicit());
    }

    #[test]
    fn test_solvent_mode_serialization() {
        // Test Implicit serialization
        let implicit = SolventMode::Implicit;
        let json = serde_json::to_string(&implicit).unwrap();
        assert!(json.contains("\"type\":\"Implicit\""));
        let deserialized: SolventMode = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, implicit);

        // Test Explicit serialization
        let explicit = SolventMode::Explicit {
            padding_angstroms: 12.5,
        };
        let json = serde_json::to_string(&explicit).unwrap();
        assert!(json.contains("\"type\":\"Explicit\""));
        assert!(json.contains("\"padding_angstroms\":12.5"));
        let deserialized: SolventMode = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, explicit);

        // Test Hybrid serialization
        let hybrid = SolventMode::Hybrid {
            exploration_steps: 100000,
            characterization_steps: 10000,
            switch_threshold: 0.7,
        };
        let json = serde_json::to_string(&hybrid).unwrap();
        assert!(json.contains("\"type\":\"Hybrid\""));
        assert!(json.contains("\"exploration_steps\":100000"));
        let deserialized: SolventMode = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, hybrid);
    }

    #[test]
    fn test_rt_probe_config_default() {
        let config = RtProbeConfig::default();
        assert!(config.enabled);
        assert_eq!(config.probe_interval, 100);
        assert_eq!(config.rays_per_point, 256);
        assert_eq!(config.attention_points, 50);
        assert_eq!(config.bvh_refit_threshold, 0.5);
        assert!(!config.enable_solvent_probes); // Off by default (explicit only)
        assert!(config.enable_aromatic_lif); // On by default
    }

    #[test]
    fn test_rt_probe_config_validation() {
        // Default config should be valid
        assert!(RtProbeConfig::default().validate().is_ok());

        // Disabled config doesn't need validation
        let mut disabled = RtProbeConfig::default();
        disabled.enabled = false;
        disabled.probe_interval = -1; // Invalid, but should pass when disabled
        assert!(disabled.validate().is_ok());

        // Invalid probe_interval
        let mut invalid = RtProbeConfig::default();
        invalid.probe_interval = 0;
        assert!(invalid.validate().is_err());

        invalid.probe_interval = -100;
        assert!(invalid.validate().is_err());

        // Invalid rays_per_point
        let mut invalid = RtProbeConfig::default();
        invalid.rays_per_point = 0;
        assert!(invalid.validate().is_err());

        // Invalid attention_points
        let mut invalid = RtProbeConfig::default();
        invalid.attention_points = 0;
        assert!(invalid.validate().is_err());

        // Invalid bvh_refit_threshold
        let mut invalid = RtProbeConfig::default();
        invalid.bvh_refit_threshold = 0.0;
        assert!(invalid.validate().is_err());

        invalid.bvh_refit_threshold = -0.5;
        assert!(invalid.validate().is_err());

        // Valid edge cases
        let mut valid = RtProbeConfig::default();
        valid.probe_interval = 1; // Very frequent probing (valid but expensive)
        assert!(valid.validate().is_ok());

        valid.rays_per_point = 1; // Minimal rays (valid but poor coverage)
        assert!(valid.validate().is_ok());
    }

    #[test]
    fn test_rt_probe_config_overhead_estimation() {
        let default_config = RtProbeConfig::default();
        let overhead = default_config.estimate_overhead();

        // With default settings, overhead should be < 10%
        assert!(
            overhead < 0.1,
            "Default RT config overhead too high: {:.1}%",
            overhead * 100.0
        );

        // Disabled config has zero overhead
        let mut disabled = RtProbeConfig::default();
        disabled.enabled = false;
        assert_eq!(disabled.estimate_overhead(), 0.0);

        // Higher probe frequency increases overhead
        let mut frequent = RtProbeConfig::default();
        frequent.probe_interval = 10; // 10x more frequent
        let frequent_overhead = frequent.estimate_overhead();
        assert!(
            frequent_overhead > overhead,
            "More frequent probing should have higher overhead"
        );

        // Solvent probing adds overhead
        let mut with_solvent = RtProbeConfig::default();
        with_solvent.enable_solvent_probes = true;
        let solvent_overhead = with_solvent.estimate_overhead();
        assert!(
            solvent_overhead > overhead,
            "Solvent probing should add overhead"
        );

        // More rays increases overhead
        let mut many_rays = RtProbeConfig::default();
        many_rays.rays_per_point = 1024; // 4x more rays
        let many_rays_overhead = many_rays.estimate_overhead();
        assert!(
            many_rays_overhead > overhead,
            "More rays should increase overhead"
        );
    }

    #[test]
    fn test_rt_probe_config_serialization() {
        let config = RtProbeConfig {
            enabled: true,
            probe_interval: 200,
            rays_per_point: 512,
            attention_points: 100,
            bvh_refit_threshold: 0.7,
            enable_solvent_probes: true,
            enable_aromatic_lif: false,
        };

        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"enabled\":true"));
        assert!(json.contains("\"probe_interval\":200"));
        assert!(json.contains("\"rays_per_point\":512"));

        let deserialized: RtProbeConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.enabled, config.enabled);
        assert_eq!(deserialized.probe_interval, config.probe_interval);
        assert_eq!(deserialized.rays_per_point, config.rays_per_point);
        assert_eq!(deserialized.attention_points, config.attention_points);
        assert_eq!(
            deserialized.enable_solvent_probes,
            config.enable_solvent_probes
        );
        assert_eq!(deserialized.enable_aromatic_lif, config.enable_aromatic_lif);
    }

    /// Parse `#define NAME <float>` out of a CUDA header.
    fn parse_define(src: &str, name: &str) -> f32 {
        for line in src.lines() {
            let l = line.trim();
            if !l.starts_with("#define") {
                continue;
            }
            let mut it = l.split_whitespace();
            it.next();
            if it.next() != Some(name) {
                continue;
            }
            let tok = it.next().unwrap_or_else(|| panic!("#define {name} has no value"));
            let tok = tok.trim_end_matches('f');
            return tok
                .parse::<f32>()
                .unwrap_or_else(|e| panic!("#define {name} = {tok:?}: {e}"));
        }
        panic!("#define {name} not found");
    }

    /// P6 — CPU/GPU band-model parity, established by PARSING the device
    /// headers rather than restating their values as Rust literals.
    ///
    /// The test this replaces was named `test_gpu_parity_wavelength_dependent`
    /// but evaluated the **CPU twice** and never touched the GPU form at all.
    /// Its own comment recorded the defect in plain text — "Note: GPU uses σ
    /// directly, CPU uses FWHM then converts" — while asserting parity held.
    /// Every other parity test in this module hand-copies `#define` values into
    /// Rust literals, which is precisely how a 2.355x divergence survived: the
    /// transcription was never re-read after the header changed.
    #[test]
    fn test_gpu_parity_band_model() {
        const ES: &str = include_str!("../../prism-gpu/src/kernels/nhs_excited_state.cuh");
        const CRYO: &str = include_str!("../../prism-gpu/src/kernels/cryo_thermal_detection.cuh");

        // ---- 1. the conversion factor itself, on both device headers -------
        let w2s_es = parse_define(ES, "BAND_W_TO_SIGMA");
        let w2s_cryo = parse_define(CRYO, "CRYO_BAND_W_TO_SIGMA");
        let w2s_cpu = std::f32::consts::FRAC_1_SQRT_2;
        assert!(
            (w2s_es - w2s_cpu).abs() < 1e-6,
            "BAND_W_TO_SIGMA drift (excited_state): GPU={w2s_es}, CPU={w2s_cpu}"
        );
        assert!(
            (w2s_cryo - w2s_cpu).abs() < 1e-6,
            "BAND_W_TO_SIGMA drift (cryo): GPU={w2s_cryo}, CPU={w2s_cpu}"
        );

        // ---- 2. the band parameters, per chromophore ----------------------
        // (device prefix-pair, cpu lambda_max, cpu epsilon_max, cpu width)
        let cases: [(&str, &str, f32, f32, f32); 4] = [
            ("TRP", "CRYO_TRP", TRP_LAMBDA_MAX, TRP_EXTINCTION_280, TRP_BANDWIDTH),
            ("TYR", "CRYO_TYR", TYR_LAMBDA_MAX, TYR_EXTINCTION_274, TYR_BANDWIDTH),
            ("PHE", "CRYO_PHE", PHE_LAMBDA_MAX, PHE_EXTINCTION_258, PHE_BANDWIDTH),
            ("SS", "CRYO_SS", DISULFIDE_LAMBDA_MAX, DISULFIDE_EXTINCTION_250, DISULFIDE_BANDWIDTH),
        ];
        for (es_p, cr_p, lam, eps, w) in cases {
            for (src, p, eps_key) in [
                (ES, es_p, "EPSILON_MAX"),
                (CRYO, cr_p, "EPSILON_MAX"),
            ] {
                let g_lam = parse_define(src, &format!("{p}_LAMBDA_MAX"));
                let g_eps = parse_define(src, &format!("{p}_{eps_key}"));
                let g_w = parse_define(src, &format!("{p}_BANDWIDTH"));
                assert!((g_lam - lam).abs() < 1e-3, "{p}_LAMBDA_MAX: GPU={g_lam}, CPU={lam}");
                assert!((g_eps - eps).abs() < 1e-3, "{p}_EPSILON_MAX: GPU={g_eps}, CPU={eps}");
                assert!(
                    (g_w - w).abs() < 1e-3,
                    "{p}_BANDWIDTH: GPU={g_w}, CPU={w} — the two families carried \
                     different widths (TYR 10 vs 12, PHE 7.5 vs 10, SS 15 vs 20) before P6"
                );
            }
        }

        // ---- 3. the functional FORM, asserted against device source --------
        // Constant parity alone is not parity: pre-P6 both sides agreed on
        // TRP_BANDWIDTH = 15.0 and still disagreed by 2.355x, because they
        // disagreed about what 15.0 MEANT.
        assert!(
            ES.contains("float sigma = bandwidth * BAND_W_TO_SIGMA;"),
            "nhs_excited_state.cuh no longer converts w -> sigma; it is treating \
             the 1/e half-width as sigma again (the pre-P6 defect)"
        );
        assert!(
            ES.contains("(2.0f * sigma * sigma)"),
            "nhs_excited_state.cuh Gaussian denominator changed"
        );
        assert!(
            CRYO.contains("float sigma = band_w_nm * CRYO_BAND_W_TO_SIGMA;"),
            "cryo_thermal_detection.cuh no longer converts w -> sigma"
        );
        for banned in ["/ 2.355f", "/ 2.355", "fwhm / 2.355"] {
            assert!(
                !ES.contains(banned) && !CRYO.contains(banned),
                "a device header reintroduced the FWHM conversion {banned:?}; \
                 the *_BANDWIDTH constants are 1/e half-widths, not FWHM"
            );
        }

        // ---- 4. numeric agreement over the full probe range ----------------
        // Recompute the device expression from the PARSED constants and compare
        // against the host function across every wavelength the instrument uses.
        for (ci, (es_p, _, _, _, _)) in cases.iter().enumerate() {
            let g_lam = parse_define(ES, &format!("{es_p}_LAMBDA_MAX"));
            let g_eps = parse_define(ES, &format!("{es_p}_EPSILON_MAX"));
            let g_w = parse_define(ES, &format!("{es_p}_BANDWIDTH"));
            let mut lambda = 240.0_f32;
            while lambda <= 340.0 {
                let sigma = g_w * w2s_es;
                let d = lambda - g_lam;
                let gpu = g_eps * (-(d * d) / (2.0 * sigma * sigma)).exp();
                let cpu = extinction_at_wavelength(ci as i32, lambda);
                let tol = 1e-3 * cpu.max(1.0);
                assert!(
                    (gpu - cpu).abs() <= tol,
                    "band parity {es_p} @ {lambda}nm: GPU={gpu:.6}, CPU={cpu:.6}"
                );
                lambda += 1.0;
            }
        }

        // ---- 5. the deployed adjudicator LUT must reproduce ----------------
        // c_mu_01_sq in cuda/adjudicator.cu, mirrored in
        // interferometric_adjudicator.rs::MU_01_SQ_LUT, is baked into device
        // __constant__ memory. It was derived under this convention, so
        // adopting it here must change NO adjudication numeric.
        #[cfg(feature = "gpu")]
        {
        let lut = crate::interferometric_adjudicator::MU_01_SQ_LUT;
        let expect: [(f32, i32); 4] = [
            (260.0, 2), // PHE-dominant
            (280.0, 0), // TRP primary
            (305.0, 0), // TRP gaussian tail
            (320.0, 0), // TRP control / baseline
        ];
        for (i, (lambda, ctype)) in expect.iter().enumerate() {
            let mu2 = extinction_at_wavelength(*ctype, *lambda) / ABSORPTION_NORMALIZATION;
            let rel = (mu2 - lut[i]).abs() / lut[i].max(1e-12);
            assert!(
                rel < 0.01,
                "adjudicator LUT[{i}] @ {lambda}nm: derived={mu2:.6}, baked={:.6} \
                 (rel {rel:.4}) — adopting the canonical band model must not move \
                 the deployed adjudication constants",
                lut[i]
            );
        }

        // ---- 6. the rejected conventions must NOT reproduce the LUT --------
        // Step 5 shows F3 reproduces the baked constants. That alone would not
        // establish F3 as the *unique* fit, so the exclusion is asserted too:
        // if some future edit made F1 or F2 also reproduce the LUT, the
        // adjudication in `band_sigma`'s documentation would no longer hold and
        // this test should say so rather than keep passing.
        for (label, sigma) in [
            ("F1 host, w/2.355", TRP_BANDWIDTH / 2.355),
            ("F2 gpu, w as sigma", TRP_BANDWIDTH),
        ] {
            for (lambda, baked) in [(305.0_f32, lut[2]), (320.0_f32, lut[3])] {
                let d = lambda - TRP_LAMBDA_MAX;
                let mu2 = (-(d * d) / (2.0 * sigma * sigma)).exp();
                let rel = (mu2 - baked).abs() / baked;
                assert!(
                    rel > 0.10,
                    "{label} now reproduces the baked adjudicator LUT at \
                     {lambda}nm (derived={mu2:.6e}, baked={baked:.6e}); the \
                     convention adjudication recorded in band_sigma() no longer \
                     uniquely selects sigma = w/sqrt(2)"
                );
            }
        }

        }

        // ---- 7. peak calibration is untouched by the width change ----------
        let cpu_trp_280 = compute_heating_at_wavelength(0, 280.0);
        assert!(
            (cpu_trp_280 - 14.5).abs() < 2.0,
            "TRP@280nm = {cpu_trp_280:.2}K, expected ~14.5K (eta=0.74)"
        );

        println!("=== P6 GPU parity (band model, parsed) PASSED ===");
        println!("TRP@280nm = {cpu_trp_280:.2} K");
        println!(
            "mu^2(305) = {:.6e}, mu^2(320) = {:.6e}",
            extinction_at_wavelength(0, 305.0) / ABSORPTION_NORMALIZATION,
            extinction_at_wavelength(0, 320.0) / ABSORPTION_NORMALIZATION
        );
    }
}
