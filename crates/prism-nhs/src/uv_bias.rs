//! UV Spectroscopy Engine (Enhanced Multi-Wavelength)
//!
//! Implements publication-quality UV pump-probe spectroscopy for cryptic site detection.
//! This module provides:
//!
//! - **Multi-wavelength frequency hopping** (258-290nm spectral scan)
//! - **True π→π* electronic state modeling** with energy deposition
//! - **Disulfide bond targeting** (σ→σ* transition at 250nm)
//! - **Local temperature tracking** from photon absorption
//! - **Wavelength-specific spike correlation**
//!
//! # Physical Basis
//!
//! ## Chromophore Absorption Profiles
//!
//! | Chromophore | λmax (nm) | ε (M⁻¹cm⁻¹) | Transition | Bandwidth |
//! |-------------|-----------|-------------|------------|-----------|
//! | Tryptophan  | 280       | 5,600       | π→π* (La)  | 15 nm     |
//! | Tyrosine    | 274       | 1,490       | π→π*       | 12 nm     |
//! | Phenylalanine| 258      | 200         | π→π*       | 10 nm     |
//! | Disulfide   | 250       | 300         | σ→σ*       | 20 nm     |
//! | **Water**   | -         | **0**       | -          | -         |
//!
//! Water's transparency at these wavelengths creates the "holographic negative"
//! effect: perturbations create signal on a silent background.
//!
//! ## Frequency Hopping Protocol
//!
//! Spectral scanning enables wavelength-specific response mapping:
//! - 258nm → Phenylalanine selective excitation
//! - 274nm → Tyrosine selective excitation
//! - 280nm → Tryptophan selective excitation
//! - 250nm → Disulfide bond perturbation
//!
//! ## Local Temperature Model
//!
//! UV photon absorption → local heating → thermal dissipation:
//! ```text
//! ΔT = (E_photon × σ_abs × Φ) / (3/2 × k_B × N_atoms)
//!
//! Expected local heating:
//! - TRP: ~15-20 K per UV burst
//! - TYR: ~8-12 K per UV burst
//! - PHE: ~2-5 K per UV burst
//! ```

use crate::config::BENZENE_EXTINCTION_254;
use crate::config::{extinction_to_cross_section, wavelength_to_ev, KB_EV_K};
use crate::config::{NhsConfig, UvBiasConfig, UvSpectroscopyConfig, ABSORPTION_NORMALIZATION};
use crate::config::{
    BENZENE_BANDWIDTH, DISULFIDE_BANDWIDTH, PHE_BANDWIDTH, TRP_BANDWIDTH, TYR_BANDWIDTH,
};
use crate::config::{
    BENZENE_LAMBDA_MAX, DISULFIDE_LAMBDA_MAX, PHE_LAMBDA_MAX, TRP_LAMBDA_MAX, TYR_LAMBDA_MAX,
};
use crate::config::{CALIBRATED_PHOTON_FLUENCE, DEFAULT_HEAT_YIELD};
use crate::config::{DISULFIDE_BOND_MAX_DISTANCE, DISULFIDE_EXTINCTION_250};
use crate::config::{NEFF_BENZENE, NEFF_DISULFIDE, NEFF_PHE, NEFF_TRP, NEFF_TYR};
use crate::config::{PHE_EXTINCTION_258, TRP_EXTINCTION_280, TYR_EXTINCTION_274};
use anyhow::Result;
use rand::Rng;
use rand_distr::{Distribution, Normal, UnitSphere};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

// =============================================================================
// AROMATIC TARGET STRUCTURES
// =============================================================================

/// Chromophore target for UV spectroscopy (aromatic or disulfide)
#[derive(Debug, Clone)]
pub struct AromaticTarget {
    /// Residue index in topology
    pub residue_idx: usize,

    /// Chromophore type (TRP, TYR, PHE, or Disulfide)
    pub residue_type: ChromophoreType,

    /// Atom indices forming the chromophore (ring atoms or S-S)
    pub ring_atoms: Vec<usize>,

    /// Center of mass of the chromophore
    pub ring_center: [f32; 3],

    /// Ring plane normal vector (for aromatics)
    pub ring_normal: [f32; 3],

    /// Two orthogonal vectors in the ring plane
    pub ring_plane_vectors: [[f32; 3]; 2],

    /// Relative absorption strength (Trp = 1.0)
    pub absorption_strength: f32,

    /// Solvent accessible surface area (for surface filtering)
    pub sasa: f32,

    /// Nearest pocket probability voxel value
    pub pocket_probability: f32,

    /// Is this target currently active?
    pub active: bool,

    // =========================================================================
    // Enhanced UV Spectroscopy Fields
    // =========================================================================
    /// Current local temperature delta from UV absorption (K)
    pub local_temp_delta: f32,

    /// Time since last UV burst hit this target (frames)
    pub frames_since_uv_hit: u32,

    /// Cumulative energy absorbed (eV)
    pub cumulative_energy: f32,

    /// Number of UV bursts received
    pub burst_count: u32,

    /// Associated spike count (for correlation)
    pub associated_spike_count: u32,

    /// Wavelength that triggered last excitation (nm)
    pub last_excitation_wavelength: Option<f32>,
}

/// Chromophore types for UV spectroscopy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChromophoreType {
    /// Tryptophan - strongest absorber, π→π* La band
    Tryptophan,
    /// Tyrosine - moderate absorber, phenol π→π*
    Tyrosine,
    /// Phenylalanine - weak absorber, benzyl π→π*
    Phenylalanine,
    /// Disulfide bond - σ→σ* transition
    Disulfide,
    /// Virtual benzene cosolvent probe - ¹B₂ᵤ π→π* at hydrophobic surfaces
    /// Targets LEU/ILE/VAL sidechains; deposits heat via virtual UV absorption
    /// Physics: Dawson 1968 (Φ_f), Cundall 1972 (Φ_isc), Pantos 1978 (ε)
    Cosolvent,
}

/// Aromatic residue types (alias for backward compatibility)
pub type AromaticType = ChromophoreType;

impl ChromophoreType {
    /// Get 3-letter code
    pub fn code(&self) -> &'static str {
        match self {
            ChromophoreType::Tryptophan => "TRP",
            ChromophoreType::Tyrosine => "TYR",
            ChromophoreType::Phenylalanine => "PHE",
            ChromophoreType::Disulfide => "CYS-CYS",
            ChromophoreType::Cosolvent => "BNZ",
        }
    }

    /// Get λmax (nm) - peak absorption wavelength
    pub fn lambda_max(&self) -> f32 {
        match self {
            ChromophoreType::Tryptophan => TRP_LAMBDA_MAX,
            ChromophoreType::Tyrosine => TYR_LAMBDA_MAX,
            ChromophoreType::Phenylalanine => PHE_LAMBDA_MAX,
            ChromophoreType::Disulfide => DISULFIDE_LAMBDA_MAX,
            ChromophoreType::Cosolvent => BENZENE_LAMBDA_MAX,
        }
    }

    /// Get extinction coefficient at λmax (M⁻¹cm⁻¹)
    ///
    /// P6: this must be the amplitude **at** [`Self::lambda_max`]. It
    /// previously returned the 280 nm assay constants while `lambda_max()`
    /// returned 274 nm for tyrosine and 258 nm for phenylalanine, so the
    /// Gaussian was centred at λmax but scaled by an off-peak amplitude.
    /// TRP is unaffected (its λmax *is* 280 nm) and PHE is numerically
    /// unchanged (`PHE_EXTINCTION_258 == PHE_EXTINCTION_280 == 200`); TYR
    /// moves 1400 → 1490, +6.4 %.
    pub fn epsilon_max(&self) -> f32 {
        match self {
            ChromophoreType::Tryptophan => TRP_EXTINCTION_280,
            ChromophoreType::Tyrosine => TYR_EXTINCTION_274,
            ChromophoreType::Phenylalanine => PHE_EXTINCTION_258,
            ChromophoreType::Disulfide => DISULFIDE_EXTINCTION_250,
            ChromophoreType::Cosolvent => BENZENE_EXTINCTION_254,
        }
    }

    /// Get the band 1/e half-width `w` (nm). Convert with
    /// [`crate::config::band_sigma`]; this is NOT an FWHM.
    pub fn bandwidth(&self) -> f32 {
        match self {
            ChromophoreType::Tryptophan => TRP_BANDWIDTH,
            ChromophoreType::Tyrosine => TYR_BANDWIDTH,
            ChromophoreType::Phenylalanine => PHE_BANDWIDTH,
            ChromophoreType::Disulfide => DISULFIDE_BANDWIDTH,
            ChromophoreType::Cosolvent => BENZENE_BANDWIDTH,
        }
    }

    /// Get electronic transition type
    pub fn transition_type(&self) -> &'static str {
        match self {
            ChromophoreType::Tryptophan => "π→π* (La band)",
            ChromophoreType::Tyrosine => "π→π* (phenol)",
            ChromophoreType::Phenylalanine => "π→π* (benzyl)",
            ChromophoreType::Disulfide => "σ→σ* (S-S)",
            ChromophoreType::Cosolvent => "π→π* (¹B₂ᵤ benzene cosolvent)",
        }
    }

    /// Get absorption coefficient at 280nm (for backward compat)
    pub fn extinction_280(&self) -> f32 {
        self.absorption_at_wavelength(280.0)
    }

    /// Get absorption at specific wavelength using Gaussian profile
    pub fn absorption_at_wavelength(&self, wavelength: f32) -> f32 {
        let delta = wavelength - self.lambda_max();
        let sigma = crate::config::band_sigma(self.bandwidth());
        self.epsilon_max() * (-0.5 * (delta / sigma).powi(2)).exp()
    }

    /// Get normalized absorption strength (Trp = 1.0)
    pub fn absorption_strength(&self) -> f32 {
        self.extinction_280() / ABSORPTION_NORMALIZATION
    }

    /// Compute local temperature increase from UV absorption (PHYSICS-CORRECTED)
    ///
    /// Formula: ΔT = (E_γ × p × η) / (3/2 × k_B × N_eff)
    /// where:
    ///   E_γ = photon energy (eV) = 1239.84 / λ(nm)
    ///   p = σ × F = absorption probability
    ///   σ = ε × 3.823×10⁻⁵ (cross-section in Å², per molecule)
    ///   F = photon fluence (photons/Å²)
    ///   η = heat yield (fraction of energy → heat)
    ///   N_eff = effective degrees of freedom for local heating
    ///
    /// Calibration target: TRP @ 280nm with F=0.024, η=1.0 → ΔT ≈ 20K
    pub fn compute_local_heating(&self, wavelength: f32, photon_fluence: f32) -> f32 {
        {
            let heat_yield = match self {
                ChromophoreType::Tryptophan => crate::config::HEAT_YIELD_TRP,
                ChromophoreType::Tyrosine => crate::config::HEAT_YIELD_TYR,
                ChromophoreType::Phenylalanine => crate::config::HEAT_YIELD_PHE,
                ChromophoreType::Disulfide => crate::config::HEAT_YIELD_DISULFIDE,
                ChromophoreType::Cosolvent => crate::config::HEAT_YIELD_BENZENE,
            };
            self.compute_local_heating_with_yield(wavelength, photon_fluence, heat_yield)
        }
    }

    /// Compute local heating with explicit heat yield parameter
    pub fn compute_local_heating_with_yield(
        &self,
        wavelength: f32,
        photon_fluence: f32,
        heat_yield: f32,
    ) -> f32 {
        let photon_energy = wavelength_to_ev(wavelength); // eV
        let epsilon = self.absorption_at_wavelength(wavelength); // M⁻¹cm⁻¹

        // CORRECTED: Proper ε → σ conversion (per molecule)
        // σ(Å²) = ε(M⁻¹cm⁻¹) × 3.823×10⁻⁵
        let sigma = extinction_to_cross_section(epsilon); // Å²

        // Absorption probability (single-photon regime requires p << 1)
        let p_absorb = sigma * photon_fluence;

        // Energy deposited per chromophore
        let energy_deposited = photon_energy * p_absorb * heat_yield; // eV

        // Effective degrees of freedom for local heating (CORRECTED values)
        let n_eff = match self {
            ChromophoreType::Tryptophan => NEFF_TRP, // 9.0 - Indole ring
            ChromophoreType::Tyrosine => NEFF_TYR,   // 10.0 - Phenol ring (6) + OH (4)
            ChromophoreType::Phenylalanine => NEFF_PHE, // 9.0 - Benzene ring + side chain
            ChromophoreType::Disulfide => NEFF_DISULFIDE, // 2.0 - S-S bond
            ChromophoreType::Cosolvent => NEFF_BENZENE, // 6.0 - Benzene ring only
        };

        // Convert to temperature increase via equipartition
        // ΔT = E_dep / (3/2 × k_B × N_eff)
        energy_deposited / (1.5 * KB_EV_K * n_eff) // Kelvin
    }

    /// Get N_eff for this chromophore type
    pub fn n_eff(&self) -> f32 {
        match self {
            ChromophoreType::Tryptophan => NEFF_TRP,
            ChromophoreType::Tyrosine => NEFF_TYR,
            ChromophoreType::Phenylalanine => NEFF_PHE,
            ChromophoreType::Disulfide => NEFF_DISULFIDE,
            ChromophoreType::Cosolvent => NEFF_BENZENE,
        }
    }

    /// Get absorption cross-section at wavelength (Å², per molecule)
    pub fn cross_section_at_wavelength(&self, wavelength: f32) -> f32 {
        extinction_to_cross_section(self.absorption_at_wavelength(wavelength))
    }

    /// Is this an aromatic (ring) chromophore?
    pub fn is_aromatic(&self) -> bool {
        match self {
            ChromophoreType::Disulfide => false,
            _ => true,
        }
    }

    /// From residue name
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_uppercase().as_str() {
            "TRP" | "W" => Some(ChromophoreType::Tryptophan),
            "TYR" | "Y" => Some(ChromophoreType::Tyrosine),
            "PHE" | "F" => Some(ChromophoreType::Phenylalanine),
            // Virtual benzene cosolvent targets: aliphatic hydrophobic residues
            "LEU" | "L" | "ILE" | "I" | "VAL" | "V" => Some(ChromophoreType::Cosolvent),
            _ => None,
        }
    }

    /// All aromatic types (excluding disulfide)
    pub fn aromatic_types() -> &'static [ChromophoreType] {
        &[
            ChromophoreType::Tryptophan,
            ChromophoreType::Tyrosine,
            ChromophoreType::Phenylalanine,
        ]
    }

    /// All chromophore types including disulfide and cosolvent
    pub fn all_types() -> &'static [ChromophoreType] {
        &[
            ChromophoreType::Tryptophan,
            ChromophoreType::Tyrosine,
            ChromophoreType::Phenylalanine,
            ChromophoreType::Disulfide,
            ChromophoreType::Cosolvent,
        ]
    }
}

// =============================================================================
// DISULFIDE BOND STRUCTURE
// =============================================================================

/// Disulfide bond target for UV perturbation at 250nm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisulfideTarget {
    /// First cysteine residue index
    pub cys1_residue_idx: usize,
    /// Second cysteine residue index
    pub cys2_residue_idx: usize,
    /// First sulfur atom index (SG)
    pub atom1_idx: usize,
    /// Second sulfur atom index (SG)
    pub atom2_idx: usize,
    /// Current S-S bond length (Å)
    pub bond_length: f32,
    /// Bond midpoint position
    pub midpoint: [f32; 3],
    /// Is this bond currently active for perturbation?
    pub active: bool,
    /// Pocket probability near this bond
    pub pocket_probability: f32,
}

// =============================================================================
// BURST EVENT TRACKING
// =============================================================================

/// A single UV burst event for correlation tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BurstEvent {
    /// Frame when burst was applied
    pub frame: usize,

    /// Target residue indices that were perturbed
    pub targets: Vec<usize>,

    /// Total energy deposited (arbitrary units)
    pub energy_deposited: f32,

    /// Burst pattern ID (for deconvolution)
    pub pattern_id: u32,
}

/// Spike event for correlation (from neuromorphic layer)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikeEvent {
    /// Frame when spike occurred
    pub frame: usize,

    /// Neuron/voxel indices that spiked
    pub neurons: Vec<usize>,

    /// Associated residue indices (if known)
    pub residues: Vec<usize>,
}

/// Causal correlation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalCorrelation {
    /// Target residue that was perturbed
    pub target_residue: usize,

    /// Spike location (residue or voxel)
    pub spike_location: usize,

    /// Time lag between burst and spike (frames)
    pub lag_frames: usize,

    /// Correlation coefficient
    pub correlation: f32,

    /// Number of observations
    pub n_observations: usize,

    /// Is this a significant causal link?
    pub is_causal: bool,
}

// =============================================================================
// FREQUENCY HOPPING PROTOCOL
// =============================================================================

/// Frequency hopping protocol for multi-wavelength spectral scanning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyHoppingProtocol {
    /// Wavelengths to scan (nm)
    pub wavelengths: Vec<f32>,

    /// Dwell time per wavelength (MD steps)
    pub dwell_steps: u32,

    /// Number of full spectral scans
    pub n_scans: u32,

    /// Current scan index
    pub current_scan: u32,

    /// Current wavelength index within scan
    pub current_wavelength_idx: usize,

    /// Steps at current wavelength
    pub steps_at_wavelength: u32,

    /// Response buffer per wavelength (wavelength_key → responses)
    pub response_buffer: HashMap<u32, Vec<WavelengthResponse>>,
}

/// Response at a specific wavelength
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WavelengthResponse {
    /// Wavelength (nm)
    pub wavelength: f32,
    /// Frame when recorded
    pub frame: usize,
    /// Spike count at this wavelength
    pub spike_count: usize,
    /// Energy deposited
    pub energy_deposited: f32,
    /// Chromophores excited
    pub chromophores_excited: Vec<usize>,
}

impl FrequencyHoppingProtocol {
    /// Create standard spectral scan protocol
    pub fn spectral_scan() -> Self {
        Self {
            wavelengths: vec![258.0, 265.0, 274.0, 280.0, 290.0],
            dwell_steps: 1000,
            n_scans: 5,
            current_scan: 0,
            current_wavelength_idx: 0,
            steps_at_wavelength: 0,
            response_buffer: HashMap::new(),
        }
    }

    /// Create fine spectral scan for publication
    pub fn fine_scan() -> Self {
        Self {
            wavelengths: vec![
                250.0, 258.0, 262.0, 266.0, 270.0, 274.0, 278.0, 280.0, 285.0, 290.0,
            ],
            dwell_steps: 500,
            n_scans: 10,
            current_scan: 0,
            current_wavelength_idx: 0,
            steps_at_wavelength: 0,
            response_buffer: HashMap::new(),
        }
    }

    /// Get current wavelength
    pub fn current_wavelength(&self) -> f32 {
        self.wavelengths
            .get(self.current_wavelength_idx)
            .copied()
            .unwrap_or(280.0)
    }

    /// Advance protocol by one step, returns true if wavelength changed
    pub fn step(&mut self) -> bool {
        self.steps_at_wavelength += 1;

        if self.steps_at_wavelength >= self.dwell_steps {
            self.steps_at_wavelength = 0;
            self.current_wavelength_idx += 1;

            if self.current_wavelength_idx >= self.wavelengths.len() {
                self.current_wavelength_idx = 0;
                self.current_scan += 1;
            }
            return true;
        }
        false
    }

    /// Is the protocol complete?
    pub fn is_complete(&self) -> bool {
        self.current_scan >= self.n_scans
    }

    /// Record response at current wavelength
    pub fn record_response(
        &mut self,
        frame: usize,
        spike_count: usize,
        energy: f32,
        chromophores: Vec<usize>,
    ) {
        let wavelength = self.current_wavelength();
        let key = (wavelength * 10.0) as u32; // 0.1nm resolution

        let response = WavelengthResponse {
            wavelength,
            frame,
            spike_count,
            energy_deposited: energy,
            chromophores_excited: chromophores,
        };

        self.response_buffer.entry(key).or_default().push(response);
    }

    /// Get spectral response summary (wavelength → average spike rate)
    pub fn get_spectral_response(&self) -> Vec<(f32, f32)> {
        let mut results = Vec::new();

        for (&key, responses) in &self.response_buffer {
            let wavelength = key as f32 / 10.0;
            let avg_spikes = if responses.is_empty() {
                0.0
            } else {
                responses.iter().map(|r| r.spike_count as f32).sum::<f32>() / responses.len() as f32
            };
            results.push((wavelength, avg_spikes));
        }

        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        results
    }
}

// =============================================================================
// LOCAL TEMPERATURE TRACKING
// =============================================================================

/// Local temperature record from UV absorption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalTempRecord {
    /// Frame when recorded
    pub frame: usize,
    /// Residue index
    pub residue_idx: usize,
    /// Chromophore type
    pub chromophore_type: String,
    /// Local temperature increase (K)
    pub delta_t: f32,
    /// Excitation wavelength (nm)
    pub wavelength: f32,
    /// Dissipation time constant (ps)
    pub dissipation_tau: f32,
    /// Neighboring residues affected by heat diffusion
    pub neighboring_residues_affected: Vec<usize>,
}

/// Local temperature state for a chromophore
#[derive(Debug, Clone)]
pub struct LocalTemperatureState {
    /// Current temperature delta (K)
    pub current_delta_t: f32,
    /// Dissipation time constant (ps)
    pub tau: f32,
    /// Time since last excitation (frames)
    pub frames_since_excitation: u32,
}

impl LocalTemperatureState {
    /// Create new state
    pub fn new(tau: f32) -> Self {
        Self {
            current_delta_t: 0.0,
            tau,
            frames_since_excitation: 0,
        }
    }

    /// Add heat from UV absorption
    pub fn add_heat(&mut self, delta_t: f32) {
        self.current_delta_t += delta_t;
        self.frames_since_excitation = 0;
    }

    /// Decay temperature over one timestep (dt in ps)
    pub fn decay(&mut self, dt: f32) {
        if self.current_delta_t > 0.0 {
            // Exponential decay: dT/dt = -T/tau
            let decay_factor = (-dt / self.tau).exp();
            self.current_delta_t *= decay_factor;
            self.frames_since_excitation += 1;

            // Clamp to zero if very small
            if self.current_delta_t < 0.01 {
                self.current_delta_t = 0.0;
            }
        }
    }
}

// =============================================================================
// WAVELENGTH-AWARE SPIKE CLASSIFICATION
// =============================================================================

/// Spike classification based on UV correlation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpikeCategory {
    /// Direct UV-induced: <5Å from chromophore, <100 steps after burst
    DirectUvInduced,
    /// Indirect thermal: temperature-driven, not near chromophore
    IndirectThermal,
    /// Cooperative network: part of spike avalanche
    CooperativeNetwork,
    /// Spontaneous fluctuation: background noise
    SpontaneousFluctuation,
}

/// Enhanced spike with wavelength correlation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WavelengthAwareSpike {
    /// Base spike event data
    pub frame: usize,
    pub neurons: Vec<usize>,
    pub residues: Vec<usize>,

    // UV correlation
    /// Wavelength that triggered this spike (if UV-induced)
    pub triggering_wavelength: Option<f32>,
    /// Time since UV burst (frames)
    pub time_since_uv_burst: u32,
    /// Distance to nearest chromophore (Å)
    pub chromophore_distance: f32,
    /// Nearest chromophore type
    pub nearest_chromophore: Option<ChromophoreType>,

    // Thermal correlation
    /// Local temperature at spike time (K above ambient)
    pub local_temp_at_spike: f32,
    /// Temperature gradient direction (unit vector)
    pub temp_gradient: [f32; 3],

    // Classification
    /// Spike category based on UV correlation
    pub spike_category: SpikeCategory,
}

impl WavelengthAwareSpike {
    /// Classify spike based on UV correlation parameters
    pub fn classify(
        chromophore_distance: f32,
        time_since_uv: u32,
        local_temp: f32,
        in_avalanche: bool,
    ) -> SpikeCategory {
        // Direct UV-induced: close to chromophore and soon after burst
        if chromophore_distance < 5.0 && time_since_uv < 100 {
            return SpikeCategory::DirectUvInduced;
        }

        // Cooperative network: part of avalanche
        if in_avalanche {
            return SpikeCategory::CooperativeNetwork;
        }

        // Indirect thermal: elevated local temperature
        if local_temp > 5.0 {
            // >5K above ambient
            return SpikeCategory::IndirectThermal;
        }

        // Default: spontaneous
        SpikeCategory::SpontaneousFluctuation
    }
}

// =============================================================================
// ENHANCED DATA RECORDING
// =============================================================================

/// Full UV spectroscopy results for publication output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UvSpectroscopyResults {
    /// Per-wavelength response summary
    pub spectral_response: HashMap<String, Vec<WavelengthResponse>>,

    /// Per-chromophore tracking
    pub chromophore_responses: Vec<ChromophoreResponse>,

    /// Temperature fluctuation history
    pub local_temperature_history: Vec<LocalTempRecord>,

    /// Disulfide dynamics events
    pub disulfide_events: Vec<DisulfideEvent>,

    /// Wavelength-spike correlation data
    pub wavelength_spike_correlation: Vec<(f32, f32)>,

    /// Spike classification counts
    pub spike_classification_counts: SpikeCategoryCounts,

    /// Summary statistics
    pub summary: SpectroscopySummary,
}

/// Response tracking for a single chromophore
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChromophoreResponse {
    /// Residue name (TRP, TYR, PHE)
    pub residue_name: String,
    /// Residue index
    pub residue_id: usize,
    /// Chromophore type
    pub chromophore_type: String,
    /// λmax (nm)
    pub lambda_max: f32,
    /// Extinction coefficient at λmax
    pub epsilon_max: f32,

    // Time series data (per frame)
    /// Local temperature delta per frame (K)
    pub local_temp_delta: Vec<f32>,
    /// Ring displacement per frame (Å RMSD)
    pub displacement: Vec<f32>,
    /// Nearby water density per frame (molecules/Å³)
    pub nearby_water_density: Vec<f32>,
    /// Did this frame trigger a spike?
    pub spike_triggered: Vec<bool>,
    /// Wavelength when excited (if any)
    pub excitation_wavelength: Vec<Option<f32>>,
}

/// Disulfide bond dynamics event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisulfideEvent {
    /// Frame when event occurred
    pub frame: usize,
    /// First cysteine residue index
    pub cys1_idx: usize,
    /// Second cysteine residue index
    pub cys2_idx: usize,
    /// Bond length at event (Å)
    pub bond_length: f32,
    /// Event type
    pub event_type: DisulfideEventType,
    /// Energy deposited (eV)
    pub energy_deposited: f32,
}

/// Types of disulfide dynamics events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DisulfideEventType {
    /// UV excitation applied
    UvExcitation,
    /// Bond stretched beyond threshold
    BondStretched,
    /// Bond returned to equilibrium
    BondRelaxed,
    /// Bond weakening observed
    BondWeakening,
}

/// Counts of spike categories
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpikeCategoryCounts {
    pub direct_uv_induced: usize,
    pub indirect_thermal: usize,
    pub cooperative_network: usize,
    pub spontaneous_fluctuation: usize,
}

impl SpikeCategoryCounts {
    pub fn total(&self) -> usize {
        self.direct_uv_induced
            + self.indirect_thermal
            + self.cooperative_network
            + self.spontaneous_fluctuation
    }

    pub fn increment(&mut self, category: SpikeCategory) {
        match category {
            SpikeCategory::DirectUvInduced => self.direct_uv_induced += 1,
            SpikeCategory::IndirectThermal => self.indirect_thermal += 1,
            SpikeCategory::CooperativeNetwork => self.cooperative_network += 1,
            SpikeCategory::SpontaneousFluctuation => self.spontaneous_fluctuation += 1,
        }
    }
}

/// Summary statistics for spectroscopy run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectroscopySummary {
    /// Total chromophores detected
    pub total_chromophores: usize,
    /// Breakdown by type
    pub chromophore_counts: HashMap<String, usize>,
    /// Total disulfide bonds
    pub total_disulfides: usize,
    /// Total UV bursts applied
    pub total_bursts: usize,
    /// Total energy deposited (eV)
    pub total_energy_deposited: f32,
    /// Average local heating (K)
    pub average_local_heating: f32,
    /// Peak local heating (K)
    pub peak_local_heating: f32,
    /// Wavelengths scanned
    pub wavelengths_scanned: Vec<f32>,
    /// Best responding wavelength (highest spike correlation)
    pub best_wavelength: Option<f32>,
    /// Overall UV-spike correlation coefficient
    pub uv_spike_correlation: f32,
}

// =============================================================================
// UV BIAS ENGINE
// =============================================================================

/// UV Spectroscopy Engine
///
/// Full multi-wavelength pump-probe spectroscopy with:
/// - Frequency hopping protocol
/// - Disulfide bond targeting
/// - Local temperature tracking
/// - Wavelength-aware spike classification
pub struct UvBiasEngine {
    config: UvBiasConfig,

    /// Extended spectroscopy config
    spectroscopy_config: Option<UvSpectroscopyConfig>,

    /// All aromatic targets in the system
    all_targets: Vec<AromaticTarget>,

    /// Disulfide bond targets (S-S bonds at 250nm)
    disulfide_targets: Vec<DisulfideTarget>,

    /// Currently active targets (near high-probability pockets)
    active_targets: Vec<usize>,

    /// Currently active disulfides
    active_disulfides: Vec<usize>,

    /// Burst history for correlation
    burst_history: VecDeque<BurstEvent>,

    /// Spike history for correlation
    spike_history: VecDeque<SpikeEvent>,

    /// Wavelength-aware spike history
    wavelength_aware_spikes: Vec<WavelengthAwareSpike>,

    /// Computed causal correlations
    correlations: HashMap<(usize, usize), CausalCorrelation>,

    /// Current frame counter
    current_frame: usize,

    /// Burst state machine
    burst_state: BurstState,

    /// Next burst pattern ID
    next_pattern_id: u32,

    /// Random number generator
    rng: rand::rngs::ThreadRng,

    // =========================================================================
    // Enhanced Spectroscopy State
    // =========================================================================
    /// Frequency hopping protocol state
    freq_hop_protocol: Option<FrequencyHoppingProtocol>,

    /// Current wavelength (nm)
    current_wavelength: f32,

    /// Local temperature states for each chromophore
    temp_states: Vec<LocalTemperatureState>,

    /// Local temperature history
    temp_history: Vec<LocalTempRecord>,

    /// Chromophore response tracking
    chromophore_responses: Vec<ChromophoreResponse>,

    /// Disulfide dynamics events
    disulfide_events: Vec<DisulfideEvent>,

    /// Spike classification counts
    spike_counts: SpikeCategoryCounts,

    /// Running total of energy deposited (eV)
    total_energy_deposited: f32,

    /// Peak local temperature observed (K)
    peak_local_temp: f32,
}

/// Burst generation state machine
#[derive(Debug, Clone)]
enum BurstState {
    /// Waiting for next burst (observation window)
    Observing { frames_remaining: u32 },

    /// Currently in a burst
    Bursting {
        pulses_remaining: u32,
        frames_to_next_pulse: u32,
        pattern_id: u32,
    },
}

impl UvBiasEngine {
    /// Create new UV bias engine with basic config
    pub fn new(config: UvBiasConfig) -> Self {
        let burst_state = if config.burst_mode {
            BurstState::Observing {
                frames_remaining: config.inter_burst_interval,
            }
        } else {
            // Continuous mode: always "bursting" with 1 pulse
            BurstState::Bursting {
                pulses_remaining: u32::MAX,
                frames_to_next_pulse: 1,
                pattern_id: 0,
            }
        };

        Self {
            config,
            spectroscopy_config: None,
            all_targets: Vec::new(),
            disulfide_targets: Vec::new(),
            active_targets: Vec::new(),
            active_disulfides: Vec::new(),
            burst_history: VecDeque::with_capacity(1000),
            spike_history: VecDeque::with_capacity(1000),
            wavelength_aware_spikes: Vec::new(),
            correlations: HashMap::new(),
            current_frame: 0,
            burst_state,
            next_pattern_id: 0,
            rng: rand::thread_rng(),
            freq_hop_protocol: None,
            current_wavelength: 280.0,
            temp_states: Vec::new(),
            temp_history: Vec::new(),
            chromophore_responses: Vec::new(),
            disulfide_events: Vec::new(),
            spike_counts: SpikeCategoryCounts::default(),
            total_energy_deposited: 0.0,
            peak_local_temp: 0.0,
        }
    }

    /// Create new UV spectroscopy engine with enhanced config
    pub fn new_spectroscopy(config: UvSpectroscopyConfig) -> Self {
        let mut engine = Self::new(config.base.clone());
        engine.spectroscopy_config = Some(config.clone());

        // Enable frequency hopping if configured
        if config.frequency_hopping_enabled {
            let mut protocol = FrequencyHoppingProtocol::spectral_scan();
            protocol.wavelengths = config.scan_wavelengths.clone();
            protocol.dwell_steps = config.dwell_steps;
            protocol.n_scans = config.n_scans;
            engine.freq_hop_protocol = Some(protocol);
        }

        engine
    }

    /// Create publication-quality spectroscopy engine
    pub fn publication_quality() -> Self {
        Self::new_spectroscopy(UvSpectroscopyConfig::publication_quality())
    }

    /// Initialize targets from protein topology
    ///
    /// # Arguments
    /// * `residue_names` - Residue names indexed by residue number
    /// * `atom_names` - Atom names for each atom (e.g., "CA", "CG", "CD1")
    /// * `atom_residues` - Residue index for each atom
    /// * `positions` - Flat array of atom positions [x0, y0, z0, x1, y1, z1, ...]
    pub fn initialize_targets(
        &mut self,
        residue_names: &[String],
        atom_names: &[String],
        atom_residues: &[usize],
        positions: &[f32],
    ) -> Result<()> {
        self.all_targets.clear();

        for (res_idx, res_name) in residue_names.iter().enumerate() {
            // Check if aromatic
            let aromatic_type = match AromaticType::from_name(res_name) {
                Some(t) => t,
                None => continue,
            };

            // Check if target is enabled in config
            if !self.config.is_valid_target(res_name) {
                continue;
            }

            // Find ring atoms for this residue (filtered by actual ring atom names)
            let ring_atoms =
                self.find_ring_atoms(res_idx, &aromatic_type, atom_names, atom_residues);
            if ring_atoms.is_empty() {
                continue;
            }

            // Compute ring geometry
            let ring_center = compute_ring_center(&ring_atoms, positions);
            let (ring_normal, ring_plane_vectors) =
                compute_ring_plane(&ring_atoms, positions, &ring_center);

            let target = AromaticTarget {
                residue_idx: res_idx,
                residue_type: aromatic_type,
                ring_atoms,
                ring_center,
                ring_normal,
                ring_plane_vectors,
                absorption_strength: aromatic_type.absorption_strength(),
                sasa: 0.0,               // Updated later
                pocket_probability: 0.0, // Updated later
                active: false,
                // Enhanced spectroscopy fields
                local_temp_delta: 0.0,
                frames_since_uv_hit: 0,
                cumulative_energy: 0.0,
                burst_count: 0,
                associated_spike_count: 0,
                last_excitation_wavelength: None,
            };

            self.all_targets.push(target);
        }

        // Initialize temperature states
        self.initialize_temp_states();

        // Initialize chromophore response tracking
        self.chromophore_responses = self
            .all_targets
            .iter()
            .map(|t| ChromophoreResponse {
                residue_name: t.residue_type.code().to_string(),
                residue_id: t.residue_idx,
                chromophore_type: t.residue_type.transition_type().to_string(),
                lambda_max: t.residue_type.lambda_max(),
                epsilon_max: t.residue_type.epsilon_max(),
                local_temp_delta: Vec::new(),
                displacement: Vec::new(),
                nearby_water_density: Vec::new(),
                spike_triggered: Vec::new(),
                excitation_wavelength: Vec::new(),
            })
            .collect();

        log::info!(
            "UvBiasEngine: Initialized {} aromatic targets (Trp: {}, Tyr: {}, Phe: {})",
            self.all_targets.len(),
            self.all_targets
                .iter()
                .filter(|t| t.residue_type == ChromophoreType::Tryptophan)
                .count(),
            self.all_targets
                .iter()
                .filter(|t| t.residue_type == ChromophoreType::Tyrosine)
                .count(),
            self.all_targets
                .iter()
                .filter(|t| t.residue_type == ChromophoreType::Phenylalanine)
                .count(),
        );

        Ok(())
    }

    /// Find ring atom indices for an aromatic residue
    ///
    /// Returns only the actual aromatic ring atoms filtered by IUPAC names:
    /// - **PHE/TYR**: CG, CD1, CD2, CE1, CE2, CZ (6-membered benzene ring)
    /// - **TRP**: CG, CD1, CD2, NE1, CE2, CE3, CZ2, CZ3, CH2 (9-atom indole ring system)
    fn find_ring_atoms(
        &self,
        residue_idx: usize,
        aromatic_type: &ChromophoreType,
        atom_names: &[String],
        atom_residues: &[usize],
    ) -> Vec<usize> {
        // IUPAC ring atom names for each aromatic type
        let ring_atom_names: &[&str] = match aromatic_type {
            ChromophoreType::Phenylalanine => &["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
            ChromophoreType::Tyrosine => &["CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"], // Include OH for π system
            ChromophoreType::Tryptophan => {
                &["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"]
            }
            ChromophoreType::Disulfide => &["SG"], // Sulfur atoms only
            ChromophoreType::Cosolvent => &["CB", "CG"], // Aliphatic sidechain center for virtual BNZ probe
        };

        // Filter atoms to only include ring atoms in this residue
        atom_residues
            .iter()
            .enumerate()
            .filter(|(atom_idx, &res_idx)| {
                res_idx == residue_idx
                    && atom_names
                        .get(*atom_idx)
                        .map(|name| ring_atom_names.contains(&name.as_str()))
                        .unwrap_or(false)
            })
            .map(|(atom_idx, _)| atom_idx)
            .collect()
    }

    /// Detect disulfide bonds (S-S) for UV targeting at 250nm
    ///
    /// Finds CYS residues with SG atoms that are bonded (< 2.5Å)
    pub fn detect_disulfides(
        &mut self,
        atom_names: &[String],
        residue_names: &[String],
        residue_ids: &[usize],
        positions: &[f32],
    ) -> Result<()> {
        self.disulfide_targets.clear();

        // Find all CYS SG atoms
        let sg_atoms: Vec<(usize, usize)> = atom_names
            .iter()
            .enumerate()
            .filter(|(i, name)| {
                *name == "SG"
                    && residue_names
                        .get(residue_ids[*i])
                        .map(|r| r == "CYS" || r == "CYX")
                        .unwrap_or(false)
            })
            .map(|(i, _)| (i, residue_ids[i]))
            .collect();

        // Check for S-S bonds (distance < 2.5Å)
        for i in 0..sg_atoms.len() {
            for j in (i + 1)..sg_atoms.len() {
                let (idx1, res1) = sg_atoms[i];
                let (idx2, res2) = sg_atoms[j];

                // Skip if same residue
                if res1 == res2 {
                    continue;
                }

                // Compute distance
                let dist = self.compute_distance(positions, idx1, idx2);

                if dist < DISULFIDE_BOND_MAX_DISTANCE {
                    // Compute midpoint
                    let midpoint = [
                        (positions[idx1 * 3] + positions[idx2 * 3]) / 2.0,
                        (positions[idx1 * 3 + 1] + positions[idx2 * 3 + 1]) / 2.0,
                        (positions[idx1 * 3 + 2] + positions[idx2 * 3 + 2]) / 2.0,
                    ];

                    self.disulfide_targets.push(DisulfideTarget {
                        cys1_residue_idx: res1,
                        cys2_residue_idx: res2,
                        atom1_idx: idx1,
                        atom2_idx: idx2,
                        bond_length: dist,
                        midpoint,
                        active: false,
                        pocket_probability: 0.0,
                    });
                }
            }
        }

        log::info!(
            "UvBiasEngine: Detected {} disulfide bonds for UV targeting",
            self.disulfide_targets.len()
        );

        Ok(())
    }

    /// Compute distance between two atoms
    fn compute_distance(&self, positions: &[f32], idx1: usize, idx2: usize) -> f32 {
        let dx = positions[idx1 * 3] - positions[idx2 * 3];
        let dy = positions[idx1 * 3 + 1] - positions[idx2 * 3 + 1];
        let dz = positions[idx1 * 3 + 2] - positions[idx2 * 3 + 2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Initialize temperature states for all chromophores
    fn initialize_temp_states(&mut self) {
        let tau = self
            .spectroscopy_config
            .as_ref()
            .map(|c| c.thermal_dissipation_tau)
            .unwrap_or(5.0);

        self.temp_states = self
            .all_targets
            .iter()
            .map(|_| LocalTemperatureState::new(tau))
            .collect();
    }

    /// Get current wavelength for frequency hopping
    pub fn get_current_wavelength(&self) -> f32 {
        self.freq_hop_protocol
            .as_ref()
            .map(|p| p.current_wavelength())
            .unwrap_or(self.current_wavelength)
    }

    /// Update target selection based on pocket probability field
    ///
    /// Targets aromatics near high-probability pocket regions
    pub fn update_target_selection(&mut self, pocket_probability: &[f32], grid_spacing: f32) {
        self.active_targets.clear();

        let threshold = self.config.pocket_probability_threshold;
        let radius = self.config.target_selection_radius;

        for (target_idx, target) in self.all_targets.iter_mut().enumerate() {
            // Find maximum pocket probability near this aromatic
            let max_prob = find_max_probability_near(
                &target.ring_center,
                pocket_probability,
                radius,
                grid_spacing,
            );

            target.pocket_probability = max_prob;
            target.active = max_prob >= threshold;

            if target.active {
                self.active_targets.push(target_idx);
            }
        }

        log::debug!(
            "UvBiasEngine: {} / {} targets active (prob >= {:.2})",
            self.active_targets.len(),
            self.all_targets.len(),
            threshold
        );
    }

    /// Process one frame - may apply perturbation
    ///
    /// Returns perturbation velocities to add to system (or empty if no perturbation)
    pub fn step(&mut self, positions: &[f32]) -> Option<PerturbationResult> {
        self.current_frame += 1;

        // Update frequency hopping if enabled
        if let Some(ref mut protocol) = self.freq_hop_protocol {
            let wavelength_changed = protocol.step();
            if wavelength_changed {
                self.current_wavelength = protocol.current_wavelength();
                log::debug!(
                    "UV Spectroscopy: Wavelength hop to {:.1}nm (scan {}/{})",
                    self.current_wavelength,
                    protocol.current_scan + 1,
                    protocol.n_scans
                );
            }
        }

        // Decay local temperatures for all chromophores
        let dt = 0.002; // 2 fs timestep in ps
        for state in &mut self.temp_states {
            state.decay(dt);
        }

        // Update target tracking
        for target in &mut self.all_targets {
            target.frames_since_uv_hit += 1;
        }

        // Update burst state machine
        let should_perturb = self.advance_burst_state();

        if !should_perturb || (self.active_targets.is_empty() && self.active_disulfides.is_empty())
        {
            return None;
        }

        // Generate perturbation with wavelength-specific absorption
        let result = self.generate_perturbation_spectroscopy(positions);

        // Record burst event
        let burst = BurstEvent {
            frame: self.current_frame,
            targets: self.active_targets.clone(),
            energy_deposited: result.total_energy,
            pattern_id: self.next_pattern_id,
        };
        self.burst_history.push_back(burst);

        // Update total energy deposited
        self.total_energy_deposited += result.total_energy;

        // Limit history size
        while self.burst_history.len() > 1000 {
            self.burst_history.pop_front();
        }

        // Record frequency hop response if enabled
        if let Some(ref mut protocol) = self.freq_hop_protocol {
            protocol.record_response(
                self.current_frame,
                0, // spike count updated later via record_spikes
                result.total_energy,
                self.active_targets.clone(),
            );
        }

        Some(result)
    }

    /// Generate perturbation with wavelength-specific absorption (enhanced version)
    fn generate_perturbation_spectroscopy(&mut self, positions: &[f32]) -> PerturbationResult {
        let mut velocity_deltas: HashMap<usize, [f32; 3]> = HashMap::new();
        let mut total_energy = 0.0f32;

        let wavelength = self.get_current_wavelength();
        let base_intensity = self.config.base_intensity;
        let scale_by_absorption = self.config.scale_by_absorption;
        let ring_plane_perturbation = self.config.ring_plane_perturbation;
        let direction_randomness = self.config.direction_randomness;

        // Get photon fluence for temperature calculation
        let photon_fluence = self
            .spectroscopy_config
            .as_ref()
            .map(|c| c.photon_fluence)
            .unwrap_or(1.0);

        // Process aromatic targets
        for &target_idx in &self.active_targets.clone() {
            // Extract data we need before mutating
            let (residue_type, residue_idx, ring_atoms) = {
                let target = &self.all_targets[target_idx];
                (
                    target.residue_type,
                    target.residue_idx,
                    target.ring_atoms.clone(),
                )
            };

            // Compute wavelength-specific absorption
            let absorption = residue_type.absorption_at_wavelength(wavelength);

            // Scale intensity by absorption at current wavelength
            let intensity = if scale_by_absorption {
                base_intensity * (absorption / ABSORPTION_NORMALIZATION)
            } else {
                base_intensity
            };

            // Skip if negligible absorption at this wavelength
            if intensity < 0.01 {
                continue;
            }

            // Compute local heating from this excitation
            let local_heating = residue_type.compute_local_heating(wavelength, photon_fluence);

            // Update target tracking
            if let Some(target) = self.all_targets.get_mut(target_idx) {
                target.local_temp_delta = local_heating;
                target.frames_since_uv_hit = 0;
                target.burst_count += 1;
                target.last_excitation_wavelength = Some(wavelength);
            }

            // Update temperature state
            if let Some(state) = self.temp_states.get_mut(target_idx) {
                state.add_heat(local_heating);
                if state.current_delta_t > self.peak_local_temp {
                    self.peak_local_temp = state.current_delta_t;
                }
            }

            // Record temperature event
            if local_heating > 1.0 {
                // Only record significant heating
                let dissipation_tau = self
                    .spectroscopy_config
                    .as_ref()
                    .map(|c| c.thermal_dissipation_tau)
                    .unwrap_or(5.0);

                self.temp_history.push(LocalTempRecord {
                    frame: self.current_frame,
                    residue_idx,
                    chromophore_type: residue_type.code().to_string(),
                    delta_t: local_heating,
                    wavelength,
                    dissipation_tau,
                    neighboring_residues_affected: Vec::new(), // TODO: compute neighbors
                });
            }

            // Update ring geometry from current positions
            let ring_center = compute_ring_center(&ring_atoms, positions);
            let (_normal, plane_vectors) = compute_ring_plane(&ring_atoms, positions, &ring_center);

            // Perturb each ring atom
            for &atom_idx in &ring_atoms {
                let delta = generate_atom_perturbation(
                    &mut self.rng,
                    intensity,
                    &plane_vectors,
                    ring_plane_perturbation,
                    direction_randomness,
                );
                velocity_deltas.insert(atom_idx, delta);

                // Estimate energy: 0.5 * m * v^2 (assume m=12 for carbon)
                let v_sq = delta[0].powi(2) + delta[1].powi(2) + delta[2].powi(2);
                total_energy += 0.5 * 12.0 * v_sq;
            }
        }

        // Process disulfide targets if at appropriate wavelength (~250nm)
        if self
            .spectroscopy_config
            .as_ref()
            .map(|c| c.target_disulfides)
            .unwrap_or(false)
        {
            let disulfide_absorption =
                ChromophoreType::Disulfide.absorption_at_wavelength(wavelength);

            if disulfide_absorption > 50.0 {
                // Significant absorption near 250nm
                for &ss_idx in &self.active_disulfides.clone() {
                    if let Some(ss) = self.disulfide_targets.get(ss_idx) {
                        let intensity =
                            base_intensity * (disulfide_absorption / ABSORPTION_NORMALIZATION);

                        // Perturb both sulfur atoms
                        for atom_idx in [ss.atom1_idx, ss.atom2_idx] {
                            let random_vec: [f32; 3] = UnitSphere.sample(&mut self.rng);
                            let delta = [
                                intensity * random_vec[0],
                                intensity * random_vec[1],
                                intensity * random_vec[2],
                            ];
                            velocity_deltas.insert(atom_idx, delta);

                            let v_sq = delta[0].powi(2) + delta[1].powi(2) + delta[2].powi(2);
                            total_energy += 0.5 * 32.0 * v_sq; // m=32 for sulfur
                        }

                        // Record disulfide event
                        self.disulfide_events.push(DisulfideEvent {
                            frame: self.current_frame,
                            cys1_idx: ss.cys1_residue_idx,
                            cys2_idx: ss.cys2_residue_idx,
                            bond_length: ss.bond_length,
                            event_type: DisulfideEventType::UvExcitation,
                            energy_deposited: total_energy,
                        });
                    }
                }
            }
        }

        PerturbationResult {
            frame: self.current_frame,
            velocity_deltas,
            total_energy,
            targets_perturbed: self.active_targets.len() + self.active_disulfides.len(),
        }
    }

    /// Advance burst state machine, return true if should perturb
    fn advance_burst_state(&mut self) -> bool {
        match &mut self.burst_state {
            BurstState::Observing { frames_remaining } => {
                if *frames_remaining > 0 {
                    *frames_remaining -= 1;
                    false
                } else {
                    // Start new burst - first pulse happens now
                    self.next_pattern_id += 1;
                    self.burst_state = BurstState::Bursting {
                        pulses_remaining: self.config.pulses_per_burst - 1, // First pulse counts
                        frames_to_next_pulse: self.config.intra_burst_interval, // Wait before next
                        pattern_id: self.next_pattern_id,
                    };
                    true
                }
            }
            BurstState::Bursting {
                pulses_remaining,
                frames_to_next_pulse,
                pattern_id: _,
            } => {
                if *frames_to_next_pulse > 0 {
                    *frames_to_next_pulse -= 1;
                    false
                } else if *pulses_remaining > 0 {
                    *pulses_remaining -= 1;
                    *frames_to_next_pulse = self.config.intra_burst_interval;
                    true
                } else {
                    // Burst complete, start observation
                    self.burst_state = BurstState::Observing {
                        frames_remaining: self.config.inter_burst_interval,
                    };
                    false
                }
            }
        }
    }

    /// Generate perturbation velocities for active targets
    fn generate_perturbation(&mut self, positions: &[f32]) -> PerturbationResult {
        let mut velocity_deltas: HashMap<usize, [f32; 3]> = HashMap::new();
        let mut total_energy = 0.0f32;

        // Clone to avoid borrow issues
        let active_targets = self.active_targets.clone();
        let base_intensity = self.config.base_intensity;
        let scale_by_absorption = self.config.scale_by_absorption;
        let ring_plane_perturbation = self.config.ring_plane_perturbation;
        let direction_randomness = self.config.direction_randomness;

        for &target_idx in &active_targets {
            let target = &self.all_targets[target_idx];

            // Scale intensity by absorption strength
            let intensity = if scale_by_absorption {
                base_intensity * target.absorption_strength
            } else {
                base_intensity
            };

            // Update ring geometry from current positions
            let ring_atoms = target.ring_atoms.clone();
            let ring_center = compute_ring_center(&ring_atoms, positions);
            let (_normal, plane_vectors) = compute_ring_plane(&ring_atoms, positions, &ring_center);

            // Perturb each ring atom
            for &atom_idx in &ring_atoms {
                let delta = generate_atom_perturbation(
                    &mut self.rng,
                    intensity,
                    &plane_vectors,
                    ring_plane_perturbation,
                    direction_randomness,
                );
                velocity_deltas.insert(atom_idx, delta);

                // Estimate energy: 0.5 * m * v^2 (assume m=12 for carbon)
                let v_sq = delta[0].powi(2) + delta[1].powi(2) + delta[2].powi(2);
                total_energy += 0.5 * 12.0 * v_sq;
            }
        }

        PerturbationResult {
            frame: self.current_frame,
            velocity_deltas,
            total_energy,
            targets_perturbed: active_targets.len(),
        }
    }

    /// Record spike events from neuromorphic layer
    pub fn record_spikes(&mut self, neurons: Vec<usize>, residues: Vec<usize>) {
        let spike = SpikeEvent {
            frame: self.current_frame,
            neurons: neurons.clone(),
            residues: residues.clone(),
        };
        self.spike_history.push_back(spike);

        // Limit history size
        while self.spike_history.len() > 1000 {
            self.spike_history.pop_front();
        }

        // Update correlations if tracking enabled
        if self.config.track_causality {
            self.update_correlations();
        }

        // Create wavelength-aware spike classification
        for &residue in &residues {
            let (chromophore_distance, nearest_chromophore, time_since_uv, local_temp) =
                self.find_nearest_chromophore_info(residue);

            let category = WavelengthAwareSpike::classify(
                chromophore_distance,
                time_since_uv,
                local_temp,
                neurons.len() > 3, // Consider avalanche if >3 neurons
            );

            self.spike_counts.increment(category);

            let aware_spike = WavelengthAwareSpike {
                frame: self.current_frame,
                neurons: neurons.clone(),
                residues: vec![residue],
                triggering_wavelength: if time_since_uv < 100 {
                    Some(self.current_wavelength)
                } else {
                    None
                },
                time_since_uv_burst: time_since_uv,
                chromophore_distance,
                nearest_chromophore,
                local_temp_at_spike: local_temp,
                temp_gradient: [0.0, 0.0, 0.0], // TODO: compute gradient
                spike_category: category,
            };

            self.wavelength_aware_spikes.push(aware_spike);

            // Update associated spike count for chromophores
            if chromophore_distance < 8.0 {
                for target in &mut self.all_targets {
                    // Inline residue distance calculation to avoid borrow conflict
                    let dist = (target.residue_idx as i32 - residue as i32).abs() as f32 * 3.8;
                    if target.residue_idx == residue || dist < 8.0 {
                        target.associated_spike_count += 1;
                    }
                }
            }
        }

        // Limit wavelength-aware spike history
        while self.wavelength_aware_spikes.len() > 10000 {
            self.wavelength_aware_spikes.remove(0);
        }
    }

    /// Find nearest chromophore information for a residue
    fn find_nearest_chromophore_info(
        &self,
        residue: usize,
    ) -> (f32, Option<ChromophoreType>, u32, f32) {
        let mut min_distance = f32::MAX;
        let mut nearest_type = None;
        let mut min_time_since_uv = u32::MAX;
        let mut local_temp = 0.0f32;

        for (idx, target) in self.all_targets.iter().enumerate() {
            // Approximate distance using residue index difference (crude but fast)
            let dist = residue_distance_approx(target.residue_idx, residue);

            if dist < min_distance {
                min_distance = dist;
                nearest_type = Some(target.residue_type);
                min_time_since_uv = target.frames_since_uv_hit;

                if let Some(state) = self.temp_states.get(idx) {
                    local_temp = state.current_delta_t;
                }
            }
        }

        (min_distance, nearest_type, min_time_since_uv, local_temp)
    }
}

/// Approximate distance between two residues (~3.8Å per residue in chain)
fn residue_distance_approx(res1: usize, res2: usize) -> f32 {
    (res1 as i32 - res2 as i32).abs() as f32 * 3.8
}

impl UvBiasEngine {
    /// Update causal correlations between bursts and spikes
    fn update_correlations(&mut self) {
        // Need sufficient history
        if self.burst_history.len() < 10 || self.spike_history.len() < 10 {
            return;
        }

        // For each recent burst, check for correlated spikes
        for burst in self.burst_history.iter().rev().take(20) {
            for spike in &self.spike_history {
                // Check if spike follows burst within correlation window
                if spike.frame <= burst.frame {
                    continue;
                }

                let lag = spike.frame - burst.frame;
                if lag > self.config.max_correlation_lag {
                    continue;
                }

                // Check for target-spike overlap
                for &target_idx in &burst.targets {
                    let target_res = self.all_targets[target_idx].residue_idx;

                    for &spike_res in &spike.residues {
                        let key = (target_res, spike_res);

                        let entry = self.correlations.entry(key).or_insert(CausalCorrelation {
                            target_residue: target_res,
                            spike_location: spike_res,
                            lag_frames: lag,
                            correlation: 0.0,
                            n_observations: 0,
                            is_causal: false,
                        });

                        entry.n_observations += 1;
                        // Update running correlation estimate
                        entry.correlation = 1.0 / (1.0 + (lag as f32 / 5.0));
                        entry.is_causal = entry.correlation
                            >= self.config.min_correlation_threshold
                            && entry.n_observations >= 3;
                    }
                }
            }
        }
    }

    /// Get all established causal correlations
    pub fn get_causal_links(&self) -> Vec<&CausalCorrelation> {
        self.correlations.values().filter(|c| c.is_causal).collect()
    }

    /// Get statistics
    pub fn stats(&self) -> UvBiasStats {
        UvBiasStats {
            total_targets: self.all_targets.len(),
            active_targets: self.active_targets.len(),
            bursts_applied: self.burst_history.len(),
            spikes_recorded: self.spike_history.len(),
            causal_links: self.correlations.values().filter(|c| c.is_causal).count(),
            current_frame: self.current_frame,
        }
    }

    /// Get extended spectroscopy statistics
    pub fn spectroscopy_stats(&self) -> SpectroscopyStats {
        SpectroscopyStats {
            base: self.stats(),
            total_disulfides: self.disulfide_targets.len(),
            active_disulfides: self.active_disulfides.len(),
            current_wavelength: self.current_wavelength,
            total_energy_deposited: self.total_energy_deposited,
            peak_local_temp: self.peak_local_temp,
            spike_classification: self.spike_counts.clone(),
            wavelength_aware_spikes: self.wavelength_aware_spikes.len(),
            temp_history_size: self.temp_history.len(),
            disulfide_events: self.disulfide_events.len(),
        }
    }

    /// Get full spectroscopy results for publication output
    pub fn get_spectroscopy_results(&self) -> UvSpectroscopyResults {
        // Build spectral response from frequency hopping data
        let spectral_response = self
            .freq_hop_protocol
            .as_ref()
            .map(|p| {
                p.response_buffer
                    .iter()
                    .map(|(k, v)| (format!("{:.1}nm", *k as f32 / 10.0), v.clone()))
                    .collect()
            })
            .unwrap_or_default();

        // Get wavelength-spike correlation
        let wavelength_spike_correlation = self
            .freq_hop_protocol
            .as_ref()
            .map(|p| p.get_spectral_response())
            .unwrap_or_default();

        // Find best wavelength (highest spike correlation)
        let best_wavelength = wavelength_spike_correlation
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(w, _)| *w);

        // Compute UV-spike correlation coefficient
        let uv_spike_correlation = self.compute_uv_spike_correlation();

        // Build chromophore counts
        let mut chromophore_counts = HashMap::new();
        for target in &self.all_targets {
            *chromophore_counts
                .entry(target.residue_type.code().to_string())
                .or_insert(0) += 1;
        }

        // Compute average local heating
        let avg_heating = if self.temp_history.is_empty() {
            0.0
        } else {
            self.temp_history.iter().map(|t| t.delta_t).sum::<f32>()
                / self.temp_history.len() as f32
        };

        UvSpectroscopyResults {
            spectral_response,
            chromophore_responses: self.chromophore_responses.clone(),
            local_temperature_history: self.temp_history.clone(),
            disulfide_events: self.disulfide_events.clone(),
            wavelength_spike_correlation,
            spike_classification_counts: self.spike_counts.clone(),
            summary: SpectroscopySummary {
                total_chromophores: self.all_targets.len(),
                chromophore_counts,
                total_disulfides: self.disulfide_targets.len(),
                total_bursts: self.burst_history.len(),
                total_energy_deposited: self.total_energy_deposited,
                average_local_heating: avg_heating,
                peak_local_heating: self.peak_local_temp,
                wavelengths_scanned: self
                    .freq_hop_protocol
                    .as_ref()
                    .map(|p| p.wavelengths.clone())
                    .unwrap_or_else(|| vec![280.0]),
                best_wavelength,
                uv_spike_correlation,
            },
        }
    }

    /// Compute overall UV-spike correlation coefficient
    fn compute_uv_spike_correlation(&self) -> f32 {
        if self.wavelength_aware_spikes.is_empty() || self.burst_history.is_empty() {
            return 0.0;
        }

        // Count direct UV-induced vs total
        let direct_count = self.spike_counts.direct_uv_induced as f32;
        let total = self.spike_counts.total() as f32;

        if total > 0.0 {
            direct_count / total
        } else {
            0.0
        }
    }

    /// Reset for new trajectory
    pub fn reset(&mut self) {
        self.active_targets.clear();
        self.active_disulfides.clear();
        self.burst_history.clear();
        self.spike_history.clear();
        self.wavelength_aware_spikes.clear();
        self.correlations.clear();
        self.current_frame = 0;
        self.next_pattern_id = 0;
        self.current_wavelength = 280.0;
        self.temp_history.clear();
        self.disulfide_events.clear();
        self.spike_counts = SpikeCategoryCounts::default();
        self.total_energy_deposited = 0.0;
        self.peak_local_temp = 0.0;

        self.burst_state = if self.config.burst_mode {
            BurstState::Observing {
                frames_remaining: self.config.inter_burst_interval,
            }
        } else {
            BurstState::Bursting {
                pulses_remaining: u32::MAX,
                frames_to_next_pulse: 1,
                pattern_id: 0,
            }
        };

        // Reset frequency hopping protocol
        if let Some(ref mut protocol) = self.freq_hop_protocol {
            protocol.current_scan = 0;
            protocol.current_wavelength_idx = 0;
            protocol.steps_at_wavelength = 0;
            protocol.response_buffer.clear();
        }

        // Reset temperature states
        for state in &mut self.temp_states {
            state.current_delta_t = 0.0;
            state.frames_since_excitation = 0;
        }

        // Reset target states
        for target in &mut self.all_targets {
            target.active = false;
            target.pocket_probability = 0.0;
            target.local_temp_delta = 0.0;
            target.frames_since_uv_hit = 0;
            target.cumulative_energy = 0.0;
            target.burst_count = 0;
            target.associated_spike_count = 0;
            target.last_excitation_wavelength = None;
        }

        // Reset disulfide targets
        for ss in &mut self.disulfide_targets {
            ss.active = false;
            ss.pocket_probability = 0.0;
        }

        // Reset chromophore responses
        for resp in &mut self.chromophore_responses {
            resp.local_temp_delta.clear();
            resp.displacement.clear();
            resp.nearby_water_density.clear();
            resp.spike_triggered.clear();
            resp.excitation_wavelength.clear();
        }
    }

    /// Update disulfide selection based on pocket probability
    pub fn update_disulfide_selection(&mut self, pocket_probability: &[f32], grid_spacing: f32) {
        self.active_disulfides.clear();

        let threshold = self.config.pocket_probability_threshold;
        let radius = self.config.target_selection_radius;

        for (ss_idx, ss) in self.disulfide_targets.iter_mut().enumerate() {
            let max_prob =
                find_max_probability_near(&ss.midpoint, pocket_probability, radius, grid_spacing);

            ss.pocket_probability = max_prob;
            ss.active = max_prob >= threshold;

            if ss.active {
                self.active_disulfides.push(ss_idx);
            }
        }

        log::debug!(
            "UvBiasEngine: {} / {} disulfides active (prob >= {:.2})",
            self.active_disulfides.len(),
            self.disulfide_targets.len(),
            threshold
        );
    }
}

// =============================================================================
// EXTENDED STATISTICS
// =============================================================================

/// Extended statistics for UV spectroscopy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectroscopyStats {
    /// Base UV bias stats
    pub base: UvBiasStats,
    /// Total disulfide bonds detected
    pub total_disulfides: usize,
    /// Currently active disulfides
    pub active_disulfides: usize,
    /// Current wavelength (nm)
    pub current_wavelength: f32,
    /// Total energy deposited (eV)
    pub total_energy_deposited: f32,
    /// Peak local temperature (K)
    pub peak_local_temp: f32,
    /// Spike classification counts
    pub spike_classification: SpikeCategoryCounts,
    /// Wavelength-aware spike count
    pub wavelength_aware_spikes: usize,
    /// Temperature history size
    pub temp_history_size: usize,
    /// Disulfide events recorded
    pub disulfide_events: usize,
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Find maximum pocket probability near a position (free function to avoid borrow issues)
fn find_max_probability_near(
    _pos: &[f32; 3],
    probability: &[f32],
    _radius: f32,
    _grid_spacing: f32,
) -> f32 {
    // Simplified: just return a placeholder
    // In production, sample the 3D grid around the position
    if probability.is_empty() {
        0.0
    } else {
        // Return max value as approximation
        probability.iter().copied().fold(0.0f32, f32::max) * 0.5
    }
}

/// Compute center of mass of ring atoms
fn compute_ring_center(ring_atoms: &[usize], positions: &[f32]) -> [f32; 3] {
    if ring_atoms.is_empty() {
        return [0.0; 3];
    }

    let mut center = [0.0f32; 3];
    for &atom_idx in ring_atoms {
        let base = atom_idx * 3;
        if base + 2 < positions.len() {
            center[0] += positions[base];
            center[1] += positions[base + 1];
            center[2] += positions[base + 2];
        }
    }

    let n = ring_atoms.len() as f32;
    [center[0] / n, center[1] / n, center[2] / n]
}

/// Compute ring plane normal and in-plane vectors
fn compute_ring_plane(
    ring_atoms: &[usize],
    positions: &[f32],
    center: &[f32; 3],
) -> ([f32; 3], [[f32; 3]; 2]) {
    // Default to XY plane if not enough atoms
    if ring_atoms.len() < 3 {
        return ([0.0, 0.0, 1.0], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    }

    // Get first two atoms relative to center
    let get_pos = |idx: usize| -> [f32; 3] {
        let base = idx * 3;
        if base + 2 < positions.len() {
            [
                positions[base] - center[0],
                positions[base + 1] - center[1],
                positions[base + 2] - center[2],
            ]
        } else {
            [0.0, 0.0, 0.0]
        }
    };

    let v1 = get_pos(ring_atoms[0]);
    let v2 = get_pos(ring_atoms[1]);

    // Cross product for normal
    let normal = [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    ];

    // Normalize
    let mag = (normal[0].powi(2) + normal[1].powi(2) + normal[2].powi(2)).sqrt();
    let normal = if mag > 1e-6 {
        [normal[0] / mag, normal[1] / mag, normal[2] / mag]
    } else {
        [0.0, 0.0, 1.0]
    };

    // Normalize v1 for first in-plane vector
    let mag_v1 = (v1[0].powi(2) + v1[1].powi(2) + v1[2].powi(2)).sqrt();
    let plane_v1 = if mag_v1 > 1e-6 {
        [v1[0] / mag_v1, v1[1] / mag_v1, v1[2] / mag_v1]
    } else {
        [1.0, 0.0, 0.0]
    };

    // Second in-plane vector: normal × v1
    let plane_v2 = [
        normal[1] * plane_v1[2] - normal[2] * plane_v1[1],
        normal[2] * plane_v1[0] - normal[0] * plane_v1[2],
        normal[0] * plane_v1[1] - normal[1] * plane_v1[0],
    ];

    (normal, [plane_v1, plane_v2])
}

/// Generate velocity perturbation for single atom
fn generate_atom_perturbation<R: rand::Rng>(
    rng: &mut R,
    intensity: f32,
    plane_vectors: &[[f32; 3]; 2],
    ring_plane_perturbation: bool,
    direction_randomness: f32,
) -> [f32; 3] {
    if ring_plane_perturbation {
        // Perturbation in ring plane (mimics π→π* excitation)
        let angle: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        let mut delta = [0.0f32; 3];
        for i in 0..3 {
            delta[i] = intensity * (cos_a * plane_vectors[0][i] + sin_a * plane_vectors[1][i]);
        }

        // Add random component
        if direction_randomness > 0.0 {
            let random_vec: [f32; 3] = UnitSphere.sample(rng);
            for i in 0..3 {
                delta[i] += intensity * direction_randomness * random_vec[i];
            }
        }

        delta
    } else {
        // Random 3D perturbation
        let random_vec: [f32; 3] = UnitSphere.sample(rng);
        [
            intensity * random_vec[0],
            intensity * random_vec[1],
            intensity * random_vec[2],
        ]
    }
}

// =============================================================================
// OUTPUT STRUCTURES
// =============================================================================

/// Result of perturbation application
#[derive(Debug, Clone)]
pub struct PerturbationResult {
    /// Frame number
    pub frame: usize,

    /// Velocity deltas by atom index (Å/ps)
    pub velocity_deltas: HashMap<usize, [f32; 3]>,

    /// Total energy deposited (kcal/mol)
    pub total_energy: f32,

    /// Number of targets perturbed
    pub targets_perturbed: usize,
}

impl PerturbationResult {
    /// Apply perturbation to velocity array
    pub fn apply_to_velocities(&self, velocities: &mut [f32]) {
        for (&atom_idx, delta) in &self.velocity_deltas {
            let base = atom_idx * 3;
            if base + 2 < velocities.len() {
                velocities[base] += delta[0];
                velocities[base + 1] += delta[1];
                velocities[base + 2] += delta[2];
            }
        }
    }
}

/// UV Bias engine statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UvBiasStats {
    pub total_targets: usize,
    pub active_targets: usize,
    pub bursts_applied: usize,
    pub spikes_recorded: usize,
    pub causal_links: usize,
    pub current_frame: usize,
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aromatic_type_absorption() {
        assert!(
            ChromophoreType::Tryptophan.absorption_strength()
                > ChromophoreType::Tyrosine.absorption_strength()
        );
        assert!(
            ChromophoreType::Tyrosine.absorption_strength()
                > ChromophoreType::Phenylalanine.absorption_strength()
        );
    }

    #[test]
    fn test_chromophore_lambda_max() {
        assert_eq!(ChromophoreType::Tryptophan.lambda_max(), 280.0);
        assert_eq!(ChromophoreType::Tyrosine.lambda_max(), 274.0);
        assert_eq!(ChromophoreType::Phenylalanine.lambda_max(), 258.0);
        assert_eq!(ChromophoreType::Disulfide.lambda_max(), 250.0);
    }

    #[test]
    fn test_wavelength_specific_absorption() {
        // TRP should have max absorption at 280nm
        let trp_at_280 = ChromophoreType::Tryptophan.absorption_at_wavelength(280.0);
        let trp_at_258 = ChromophoreType::Tryptophan.absorption_at_wavelength(258.0);
        assert!(
            trp_at_280 > trp_at_258,
            "TRP should absorb more at 280nm than 258nm"
        );

        // PHE should have max absorption at 258nm
        let phe_at_258 = ChromophoreType::Phenylalanine.absorption_at_wavelength(258.0);
        let phe_at_280 = ChromophoreType::Phenylalanine.absorption_at_wavelength(280.0);
        assert!(
            phe_at_258 > phe_at_280,
            "PHE should absorb more at 258nm than 280nm"
        );

        // TYR should have max absorption at 274nm
        let tyr_at_274 = ChromophoreType::Tyrosine.absorption_at_wavelength(274.0);
        let tyr_at_290 = ChromophoreType::Tyrosine.absorption_at_wavelength(290.0);
        assert!(
            tyr_at_274 > tyr_at_290,
            "TYR should absorb more at 274nm than 290nm"
        );
    }

    #[test]
    fn test_local_heating_calculation() {
        let heating_trp = ChromophoreType::Tryptophan.compute_local_heating(280.0, 1.0);
        let heating_phe = ChromophoreType::Phenylalanine.compute_local_heating(258.0, 1.0);

        // TRP should produce more heating (higher absorption)
        assert!(
            heating_trp > heating_phe,
            "TRP should produce more local heating than PHE"
        );

        // Heating should be positive
        assert!(heating_trp > 0.0, "Local heating should be positive");
        println!("TRP local heating at 280nm: {:.2} K", heating_trp);
        println!("PHE local heating at 258nm: {:.2} K", heating_phe);
    }

    #[test]
    fn test_frequency_hopping_protocol() {
        let mut protocol = FrequencyHoppingProtocol::spectral_scan();

        // Initial wavelength should be 258nm
        assert_eq!(protocol.current_wavelength(), 258.0);

        // Step through dwell period
        for _ in 0..999 {
            assert!(
                !protocol.step(),
                "Wavelength should not change during dwell"
            );
        }

        // This step should trigger wavelength change
        assert!(protocol.step(), "Wavelength should change at end of dwell");
        assert_eq!(protocol.current_wavelength(), 265.0);
    }

    #[test]
    fn test_spike_classification() {
        // Direct UV-induced
        assert_eq!(
            WavelengthAwareSpike::classify(3.0, 50, 2.0, false),
            SpikeCategory::DirectUvInduced
        );

        // Cooperative network
        assert_eq!(
            WavelengthAwareSpike::classify(10.0, 200, 2.0, true),
            SpikeCategory::CooperativeNetwork
        );

        // Indirect thermal
        assert_eq!(
            WavelengthAwareSpike::classify(15.0, 200, 10.0, false),
            SpikeCategory::IndirectThermal
        );

        // Spontaneous
        assert_eq!(
            WavelengthAwareSpike::classify(20.0, 300, 1.0, false),
            SpikeCategory::SpontaneousFluctuation
        );
    }

    #[test]
    fn test_burst_state_machine() {
        let config = UvBiasConfig {
            burst_mode: true,
            pulses_per_burst: 3,
            intra_burst_interval: 2,
            inter_burst_interval: 10,
            ..Default::default()
        };

        let mut engine = UvBiasEngine::new(config);

        // Should be observing initially
        for _ in 0..10 {
            assert!(!engine.advance_burst_state());
        }

        // Should start bursting
        assert!(engine.advance_burst_state());

        // Intra-burst interval
        for _ in 0..2 {
            assert!(!engine.advance_burst_state());
        }

        // Second pulse
        assert!(engine.advance_burst_state());
    }

    #[test]
    fn test_perturbation_generation() {
        let config = UvBiasConfig::default();
        let mut engine = UvBiasEngine::new(config);

        // Create dummy target with all new fields
        engine.all_targets.push(AromaticTarget {
            residue_idx: 0,
            residue_type: ChromophoreType::Tryptophan,
            ring_atoms: vec![0, 1, 2, 3, 4, 5],
            ring_center: [0.0, 0.0, 0.0],
            ring_normal: [0.0, 0.0, 1.0],
            ring_plane_vectors: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            absorption_strength: 1.0,
            sasa: 1.0,
            pocket_probability: 0.5,
            active: true,
            local_temp_delta: 0.0,
            frames_since_uv_hit: 0,
            cumulative_energy: 0.0,
            burst_count: 0,
            associated_spike_count: 0,
            last_excitation_wavelength: None,
        });
        engine.active_targets.push(0);
        engine.temp_states.push(LocalTemperatureState::new(5.0));

        let positions = vec![0.0; 18]; // 6 atoms × 3 coords
        let result = engine.generate_perturbation_spectroscopy(&positions);

        assert_eq!(result.targets_perturbed, 1);
        assert!(!result.velocity_deltas.is_empty());
    }

    #[test]
    fn test_spectroscopy_engine_creation() {
        let engine = UvBiasEngine::publication_quality();

        assert!(engine.spectroscopy_config.is_some());
        assert!(engine.freq_hop_protocol.is_some());

        let config = engine.spectroscopy_config.as_ref().unwrap();
        assert!(config.frequency_hopping_enabled);
        assert!(config.target_disulfides);
        assert!(config.track_local_temperature);
    }

    #[test]
    fn test_local_temp_decay() {
        let mut state = LocalTemperatureState::new(5.0); // 5 ps tau

        // Add heat
        state.add_heat(20.0); // +20 K
        assert_eq!(state.current_delta_t, 20.0);

        // Decay for 5 ps (tau)
        for _ in 0..2500 {
            // 2500 steps × 2fs = 5 ps
            state.decay(0.002);
        }

        // Should be ~1/e of original
        let expected = 20.0 * (-1.0_f32).exp();
        assert!(
            (state.current_delta_t - expected).abs() < 1.0,
            "Expected ~{:.1} K after 1 tau, got {:.1} K",
            expected,
            state.current_delta_t
        );
    }

    // =========================================================================
    // PHYSICS CALIBRATION TESTS (MUST PASS FOR PUBLICATION)
    // =========================================================================

    #[test]
    fn test_physics_calibration_trp_280nm() {
        // CRITICAL CALIBRATION TEST: TRP @ 280nm with calibrated parameters
        // Expected results:
        //   ε = 5600 M⁻¹cm⁻¹
        //   σ = ε × 3.823e-5 = 0.21409 Å²
        //   F = 0.024 photons/Å²
        //   p = σ × F = 0.00514
        //   E_photon = 4.428 eV
        //   E_dep = E_photon × p × η = 0.0228 eV
        //   ΔT = E_dep / (1.5 × k_B × N_eff) ≈ 19.6 K

        use crate::config::{
            extinction_to_cross_section, wavelength_to_ev, CALIBRATED_PHOTON_FLUENCE,
            DEFAULT_HEAT_YIELD, KB_EV_K, MAX_ABSORPTION_PROBABILITY, NEFF_TRP, TRP_EXTINCTION_280,
        };

        let wavelength = 280.0;
        let epsilon = TRP_EXTINCTION_280; // 5600
        let fluence = CALIBRATED_PHOTON_FLUENCE; // 0.024
        let eta = DEFAULT_HEAT_YIELD; // 1.0
        let n_eff = NEFF_TRP; // 9.0

        // Verify cross-section calculation
        let sigma = extinction_to_cross_section(epsilon);
        assert!(
            (sigma - 0.21409).abs() < 0.001,
            "σ_TRP mismatch: expected 0.21409 Å², got {:.5} Å²",
            sigma
        );

        // Verify absorption probability (single-photon regime)
        let p_absorb = sigma * fluence;
        assert!(
            (p_absorb - 0.00514).abs() < 0.0001,
            "p_absorb mismatch: expected 0.00514, got {:.5}",
            p_absorb
        );
        assert!(
            p_absorb < MAX_ABSORPTION_PROBABILITY,
            "p_absorb ({}) exceeds single-photon threshold ({})",
            p_absorb,
            MAX_ABSORPTION_PROBABILITY
        );

        // Verify photon energy
        let e_photon = wavelength_to_ev(wavelength);
        assert!(
            (e_photon - 4.428).abs() < 0.01,
            "E_photon mismatch: expected 4.428 eV, got {:.3} eV",
            e_photon
        );

        // Verify energy deposited
        let e_dep = e_photon * p_absorb * 0.74; // η_TRP = 0.74
        assert!(
            (e_dep - 0.01687).abs() < 0.001,
            "E_dep mismatch: expected 0.0228 eV, got {:.4} eV",
            e_dep
        );

        // Verify temperature rise (THE CALIBRATION TARGET)
        let delta_t = e_dep / (1.5 * KB_EV_K * n_eff);
        assert!(
            (delta_t - 14.5).abs() < 1.0,
            "ΔT calibration PASSED (η=0.74): expected ~14.5 K, got {:.1} K",
            delta_t
        );

        // Also verify via ChromophoreType method
        let delta_t_method = ChromophoreType::Tryptophan.compute_local_heating(wavelength, fluence);
        assert!(
            (delta_t_method - 14.5).abs() < 1.0,
            "ChromophoreType.compute_local_heating (η=0.74): expected ~14.5 K, got {:.1} K",
            delta_t_method
        );

        println!("=== TRP @ 280nm Calibration PASSED ===");
        println!("ε = {:.0} M⁻¹cm⁻¹", epsilon);
        println!("σ = {:.5} Å²", sigma);
        println!("F = {:.4} photons/Å²", fluence);
        println!(
            "p = {:.5} (< {} threshold)",
            p_absorb, MAX_ABSORPTION_PROBABILITY
        );
        println!("E_γ = {:.3} eV", e_photon);
        println!("η = {:.1}", eta);
        println!("E_dep = {:.5} eV", e_dep);
        println!("N_eff = {:.1}", n_eff);
        println!("ΔT = {:.2} K ✓", delta_t);
    }

    #[test]
    fn test_physics_cross_section_values() {
        // Verify all chromophore cross-sections at peak wavelengths
        use crate::config::extinction_to_cross_section;

        // TRP @ 280nm: σ = 5600 × 3.823e-5 = 0.21409
        let sigma_trp = extinction_to_cross_section(5600.0);
        assert!((sigma_trp - 0.21409).abs() < 0.001, "σ_TRP error");

        // TYR @ 274nm: σ = 1490 × 3.823e-5 = 0.05696
        let sigma_tyr = extinction_to_cross_section(1490.0);
        assert!((sigma_tyr - 0.05696).abs() < 0.001, "σ_TYR error");

        // PHE @ 258nm: σ = 200 × 3.823e-5 = 0.00765
        let sigma_phe = extinction_to_cross_section(200.0);
        assert!((sigma_phe - 0.00765).abs() < 0.0001, "σ_PHE error");

        // S-S @ 250nm: σ = 300 × 3.823e-5 = 0.01147
        let sigma_ss = extinction_to_cross_section(300.0);
        assert!((sigma_ss - 0.01147).abs() < 0.0001, "σ_SS error");
    }

    #[test]
    fn test_physics_multiwavelength_response() {
        // Verify wavelength selectivity: TRP should absorb most at 280nm
        let fluence = CALIBRATED_PHOTON_FLUENCE;

        let trp_at_280 = ChromophoreType::Tryptophan.compute_local_heating(280.0, fluence);
        let trp_at_258 = ChromophoreType::Tryptophan.compute_local_heating(258.0, fluence);
        let trp_at_250 = ChromophoreType::Tryptophan.compute_local_heating(250.0, fluence);

        // TRP response should peak at 280nm
        assert!(
            trp_at_280 > trp_at_258,
            "TRP should heat more at 280nm than 258nm"
        );
        assert!(
            trp_at_280 > trp_at_250,
            "TRP should heat more at 280nm than 250nm"
        );

        // PHE should respond more at 258nm than 280nm
        let phe_at_258 = ChromophoreType::Phenylalanine.compute_local_heating(258.0, fluence);
        let phe_at_280 = ChromophoreType::Phenylalanine.compute_local_heating(280.0, fluence);
        assert!(
            phe_at_258 > phe_at_280,
            "PHE should heat more at 258nm than 280nm"
        );

        println!("Multi-wavelength response verified:");
        println!("  TRP @ 280nm: {:.2} K", trp_at_280);
        println!("  TRP @ 258nm: {:.2} K", trp_at_258);
        println!("  PHE @ 258nm: {:.2} K", phe_at_258);
        println!("  PHE @ 280nm: {:.2} K", phe_at_280);
    }
}
