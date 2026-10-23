//! Spike Thermodynamic Integration (STI)
//!
//! Extracts rigorous thermodynamic quantities from spike event data using
//! non-equilibrium free energy methods (Jarzynski, Crooks, BAR).
//!
//! Five analysis layers:
//! 1. Jarzynski per-voxel free energy from spike intensities
//! 2. Crooks intersection for pocket opening free energy
//! 3. Channel decomposition (UV/LIF/EFP contributions)
//! 4. Arrhenius barrier estimation from temperature-dependent spike rates
//! 5. Combined binding free energy with kinetic accessibility

use serde::Serialize;
use std::collections::HashMap;

#[cfg(feature = "gpu")]
use crate::fused_engine::GpuSpikeEvent;

// ═══════════════════════════════════════════════════════════════════════════════
// PHYSICAL CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════

/// Boltzmann constant in kcal/(mol·K)
const KB: f64 = 0.001987;
/// Default temperature (K)
const DEFAULT_TEMP: f64 = 300.0;
/// Planck constant in kcal·s/mol
const H_PLANCK: f64 = 9.537e-14;
/// Speed of light in nm/s
const C_LIGHT: f64 = 3.0e17;
/// Hydration energy per water molecule (kcal/mol)
const HYDRATION_ENERGY: f64 = 2.27;
/// Warshel effective dielectric constant
const WARSHEL_DIELECTRIC: f64 = 20.0;
/// Coulomb constant in kcal·Å/(mol·e²) — 332.0636 in AMBER units
const COULOMB_CONST: f64 = 332.0636;

// UV extinction coefficients (M⁻¹ cm⁻¹) at peak wavelengths
const EPS_280: f64 = 5600.0; // TRP
const EPS_274: f64 = 1490.0; // TYR
const EPS_258: f64 = 197.0; // PHE
const EPS_254: f64 = 200.0; // BNZ/aliphatic
const EPS_211: f64 = 300.0; // HIS/disulfide
/// UV quantum yield for protein photodamage coupling
/// Typical Trp fluorescence QY ~0.13; here we use a coupling efficiency
/// scaled to produce work values comparable to LIF/hydration energy (~2 kcal/mol).
/// W_UV = E_photon(102 kcal/mol at 280nm) × QY × rel_extinction ≈ 1-5 kcal/mol
const UV_QUANTUM_YIELD: f64 = 0.02;

// ═══════════════════════════════════════════════════════════════════════════════
// OUTPUT STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════════

/// Complete binding free energy result for a detected pocket
#[derive(Debug, Clone, Serialize)]
pub struct BindingFreeEnergy {
    /// Jarzynski free energy (total, kcal/mol)
    pub delta_g_sti_kcal_mol: f64,
    /// UV/aromatic channel contribution
    pub delta_g_aromatic_kcal_mol: f64,
    /// LIF/dewetting channel contribution
    pub delta_g_dewetting_kcal_mol: f64,
    /// EFP/electrostatic channel contribution
    pub delta_g_electrostatic_kcal_mol: f64,
    /// Multi-channel cooperative effect (synergy)
    pub delta_g_cooperative_kcal_mol: f64,
    /// Crooks intersection free energy (from hysteresis)
    pub delta_g_crooks_kcal_mol: Option<f64>,
    /// BAR estimator free energy
    pub delta_g_bar_kcal_mol: Option<f64>,
    /// Old branching-theory free energy (preserved for comparison)
    pub delta_g_branching_kcal_mol: Option<f64>,
    /// Activation energy per wavelength
    pub activation_energy_by_wavelength: HashMap<String, f64>,
    /// Mean activation energy across wavelengths
    pub activation_energy_mean_kcal_mol: Option<f64>,
    /// Effective free energy including kinetic accessibility
    pub effective_delta_g_kcal_mol: f64,
    /// Kinetic accessibility factor [0,1]
    pub kinetic_accessibility: f64,
    /// Cumulant expansion validity flag
    pub cumulant_valid: bool,
    /// Number of voxels analyzed
    pub n_voxels: usize,
    /// Number of spike events used
    pub n_spikes: usize,
}

/// Per-voxel Jarzynski result
#[derive(Debug, Clone)]
struct VoxelFreeEnergy {
    voxel_idx: i32,
    delta_g: f64,
    delta_g_cumulant: f64,
    n_samples: usize,
    cumulant_valid: bool,
}

/// Hysteresis bin data (mirrors ccns_analyze format)
#[derive(Debug, Clone)]
pub struct HysteresisBinData {
    pub temp_k: f32,
    pub heating_count: usize,
    pub cooling_count: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
// CHANNEL INFERENCE
// ═══════════════════════════════════════════════════════════════════════════════

/// Derive ramp phase from timestep and protocol parameters.
///
/// Returns phase 1-5:
///   1 = cold_hold, 2 = heating ramp, 3 = warm_hold,
///   4 = cooling ramp, 5 = cold_return_hold
///
/// For non-hysteresis protocols (no phases 4/5), all timesteps beyond
/// cold_hold+ramp fall into phase 3 (warm_hold).
fn derive_ramp_phase(
    timestep: i32,
    cold_hold_steps: i32,
    ramp_steps: i32,
    warm_hold_steps: i32,
) -> i32 {
    if timestep < cold_hold_steps {
        1 // cold_hold
    } else if timestep < cold_hold_steps + ramp_steps {
        2 // heating ramp
    } else if timestep < cold_hold_steps + ramp_steps + warm_hold_steps {
        3 // warm_hold
    } else {
        4 // cooling / cold_return (reverse process)
    }
}

/// Classify spike source channel for thermodynamic decomposition.
///
/// The GPU kernel sets spike_source to 1 (UV) or 2 (LIF) but never 3 (EFP).
/// We reclassify spikes into three thermodynamic channels:
///
/// UV (1):  Aromatic fluorescence — π-stacking interactions
///          Identified by: wavelength > 200nm AND aromatic_type >= 0
///          OR kernel-assigned source == 1
///
/// EFP (3): Electrostatic field perturbation — charged residue interactions
///          Identified by: high collective excitation (n_nearby_excited >= 3)
///          WITHOUT aromatic origin (aromatic_type < 0).
///          These are spikes driven by Coulombic interactions (salt bridges,
///          H-bond networks, charged sidechains) rather than dewetting.
///
/// LIF (2): Laser-induced fluorescence / dewetting — water displacement
///          Default for remaining density-based spikes.
#[cfg(feature = "gpu")]
fn infer_spike_source(spike: &GpuSpikeEvent) -> i32 {
    // If the GPU already set source correctly (1=UV, 2=LIF, 3=EFP), trust it
    if spike.spike_source == 3 {
        return 3;
    }
    // UV channel: aromatic fluorescence metadata present
    if spike.spike_source == 1 || (spike.wavelength_nm > 200.0 && spike.aromatic_type >= 0) {
        return 1;
    }
    // EFP reclassification: GPU sometimes sets source=2 for what should be EFP.
    // EFP spikes are characterized by:
    //   - vibrational_energy > 0 (energy deposition from charged interactions)
    //   - aromatic_type < 0 (not aromatic origin)
    //   - OR n_nearby_excited >= 2 with no aromatic origin (collective charged excitation)
    if spike.aromatic_type < 0 && (spike.vibrational_energy > 0.001 || spike.n_nearby_excited >= 2)
    {
        return 3;
    }
    // LIF/dewetting channel: default for density-based spikes
    2
}

// ═══════════════════════════════════════════════════════════════════════════════
// LAYER 1: JARZYNSKI PER-VOXEL FREE ENERGY
// ═══════════════════════════════════════════════════════════════════════════════

/// Convert spike intensity to work (energy) based on source channel
fn spike_to_work(intensity: f32, source: i32, wavelength_nm: f32) -> f64 {
    match source {
        1 => {
            // UV channel: E_photon × quantum_yield × relative_extinction
            // E_photon at 280nm = hc/λ ≈ 102 kcal/mol (from PLANCK × C_LIGHT / nm)
            // relative_extinction weights by wavelength-specific absorption probability
            // UV_QUANTUM_YIELD scales to produce work comparable to LIF hydration energy
            let e_photon = H_PLANCK * C_LIGHT / (wavelength_nm as f64).max(200.0);
            let relative_extinction = match wavelength_nm as i32 {
                278..=282 => EPS_280 / EPS_280, // 1.0 (reference)
                272..=276 => EPS_274 / EPS_280, // 0.27
                256..=260 => EPS_258 / EPS_280, // 0.035
                252..=255 => EPS_254 / EPS_280, // 0.036
                209..=213 => EPS_211 / EPS_280, // 0.054
                _ => EPS_258 / EPS_280,
            };
            e_photon * UV_QUANTUM_YIELD * relative_extinction * (intensity as f64)
        }
        2 => {
            // LIF channel: water density change × hydration energy
            (intensity as f64) * HYDRATION_ENERGY
        }
        3 => {
            // EFP channel: Coulomb work with Warshel dielectric
            // W = q₁·q₂ / (ε_eff · r), use intensity as proxy for field strength
            let effective_charge = 0.5; // elementary charges
            let effective_distance = 4.0; // Å, typical
            COULOMB_CONST * effective_charge * effective_charge
                / (WARSHEL_DIELECTRIC * effective_distance)
                * (intensity as f64)
        }
        _ => intensity as f64,
    }
}

/// Jarzynski free energy estimator for a set of work values across streams.
///
/// ΔG = -kT · ln( (1/N) · Σ exp(-W/kT) )
///
/// Also computes cumulant expansion: ΔG ≈ ⟨W⟩ - σ²_W/(2kT)
///
/// When W/kT >> 1 for all samples (typical for spike intensities at 300K),
/// the exact Jarzynski degenerates to min(W) + kT·ln(N) — dominated by
/// a single outlier. In this regime we return the cumulant expansion as
/// the primary estimate, which uses mean and variance of the full work
/// distribution and produces much better inter-site discrimination.
fn jarzynski_estimator(work_values: &[f64], temperature: f64) -> (f64, f64, bool) {
    if work_values.is_empty() {
        return (0.0, 0.0, false);
    }

    let kt = KB * temperature;
    let n = work_values.len() as f64;

    // Cumulant expansion (always computed, robust at any W/kT)
    let mean_w: f64 = work_values.iter().sum::<f64>() / n;
    let var_w: f64 = work_values
        .iter()
        .map(|w| (w - mean_w).powi(2))
        .sum::<f64>()
        / n;
    let delta_g_cumulant = mean_w - var_w / (2.0 * kt);

    // Validity: cumulant is exact when σ²/(kT)² < 1
    let cumulant_valid = (var_w / (kt * kt)) < 1.0 && work_values.len() >= 5;

    // Exact Jarzynski (log-sum-exp trick for numerical stability)
    let neg_w_over_kt: Vec<f64> = work_values.iter().map(|w| -w / kt).collect();
    let max_val = neg_w_over_kt
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let sum_exp: f64 = neg_w_over_kt.iter().map(|&x| (x - max_val).exp()).sum();
    let delta_g_exact = -kt * (sum_exp.ln() - n.ln() + max_val);

    // When W/kT >> 1 (high dissipation regime), the exact estimator is
    // dominated by the single smallest W value (all other exp(-W/kT) terms
    // vanish). Detect this: if only ~1 term contributes to the sum, the exact
    // estimator has degenerated. Effective sample size: (Σ exp)² / Σ exp² after
    // shift. Note n_eff >= 1 by construction — it is an inverse participation
    // ratio — so any threshold expressed as a bare count is unsatisfiable for
    // small n. The degeneracy test must therefore be a RATIO.
    let sum_exp_sq: f64 = neg_w_over_kt
        .iter()
        .map(|&x| (2.0 * (x - max_val)).exp())
        .sum();
    let n_eff = if sum_exp_sq > 0.0 {
        (sum_exp * sum_exp) / sum_exp_sq
    } else {
        1.0
    };
    let degenerate = (n_eff / n) < 0.1;

    // Estimator selection.
    //
    // This previously read `if n_eff < 0.1 * n && n > 5.0 { delta_g_cumulant }`,
    // which produced the unphysical values recorded at nhs_rt_full.rs:37127
    // ("Jarzynski produces -2795 to +147 kcal/mol garbage"). Three faults:
    //
    //   1. cumulant_valid was computed above and never consulted, so the second
    //      order cumulant was returned in exactly the regime its own validity
    //      test rejects (observed var_W/(kT)^2 up to 4118 against a threshold
    //      of 1). The -var_W/(2kT) term is unbounded and supplies the -2795.
    //   2. n_eff >= 1 always, so `n_eff < 0.1 * n` could never fire for n <= 10
    //      and always could for larger n: identical physics returned +13.2
    //      kcal/mol at n=10 and -51.2 at n=20.
    //   3. neither branch was bounded, so the result could leave the interval
    //      that the Jarzynski inequality permits.
    //
    // The cumulant is now used only where it is valid. Where the work
    // distribution is degenerate the exact estimator is retained and reported
    // as the bound it actually is, rather than replaced by a number that looks
    // converged. Every path is clamped to the physically admissible interval:
    // the Jarzynski inequality gives ΔG <= <W>, and the exponential average is
    // bounded below by min(W) - kT ln N.
    let delta_g_raw = if degenerate && cumulant_valid {
        delta_g_cumulant
    } else {
        delta_g_exact
    };

    let min_w = work_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let lower_bound = min_w - kt * n.ln();
    let upper_bound = mean_w;
    let delta_g = delta_g_raw.max(lower_bound).min(upper_bound);

    (delta_g, delta_g_cumulant, cumulant_valid)
}

/// Regime the estimator resolved for this work distribution.
///
/// Additive companion to [`jarzynski_estimator`], which keeps its tuple shape so
/// existing callers are unaffected. A consumer that intends to attach authority
/// to a free energy must check this: `ExactDegenerateBoundOnly` means the value
/// is a bound on ΔG, not a converged estimate of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JarzynskiRegime {
    /// Exponential average converged over multiple contributing samples.
    Exact,
    /// Work distribution is degenerate and the cumulant is out of validity;
    /// the returned value is a bound, not a converged free energy.
    ExactDegenerateBoundOnly,
    /// Degenerate but σ²/(kT)² < 1, so the cumulant expansion is admissible.
    CumulantValid,
    /// No work samples.
    Empty,
}

/// Classify the regime without recomputing the estimate.
pub fn jarzynski_regime(work_values: &[f64], temperature: f64) -> JarzynskiRegime {
    if work_values.is_empty() {
        return JarzynskiRegime::Empty;
    }
    let kt = KB * temperature;
    let n = work_values.len() as f64;
    let mean_w: f64 = work_values.iter().sum::<f64>() / n;
    let var_w: f64 = work_values.iter().map(|w| (w - mean_w).powi(2)).sum::<f64>() / n;
    let cumulant_valid = (var_w / (kt * kt)) < 1.0 && work_values.len() >= 5;

    let neg_w_over_kt: Vec<f64> = work_values.iter().map(|w| -w / kt).collect();
    let max_val = neg_w_over_kt
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let sum_exp: f64 = neg_w_over_kt.iter().map(|&x| (x - max_val).exp()).sum();
    let sum_exp_sq: f64 = neg_w_over_kt
        .iter()
        .map(|&x| (2.0 * (x - max_val)).exp())
        .sum();
    let n_eff = if sum_exp_sq > 0.0 {
        (sum_exp * sum_exp) / sum_exp_sq
    } else {
        1.0
    };

    if (n_eff / n) < 0.1 {
        if cumulant_valid {
            JarzynskiRegime::CumulantValid
        } else {
            JarzynskiRegime::ExactDegenerateBoundOnly
        }
    } else {
        JarzynskiRegime::Exact
    }
}

/// Compute per-voxel Jarzynski free energies from spike events.
///
/// Groups spikes by voxel_idx, uses all FORWARD-process phases (1=cold_hold,
/// 2=heating, 3=warm_hold). UV bursts fire during cold_hold, LIF during all
/// forward phases. Phases 4/5 (cooling/cold_return) are excluded as reverse process.
#[cfg(feature = "gpu")]
pub fn jarzynski_per_voxel(
    spikes: &[GpuSpikeEvent],
    temperature: f64,
    cold_hold_steps: i32,
    ramp_steps: i32,
    warm_hold_steps: i32,
) -> Vec<VoxelFreeEnergy> {
    // Group forward-process spikes (cold_hold + heating + warm_hold) by voxel
    let mut voxel_work: HashMap<i32, Vec<f64>> = HashMap::new();

    for spike in spikes {
        let phase = derive_ramp_phase(spike.timestep, cold_hold_steps, ramp_steps, warm_hold_steps);
        if !matches!(phase, 1 | 2 | 3) {
            continue;
        } // forward process only
        let source = infer_spike_source(spike);
        let work = spike_to_work(spike.intensity, source, spike.wavelength_nm);
        voxel_work.entry(spike.voxel_idx).or_default().push(work);
    }

    voxel_work
        .into_iter()
        .map(|(voxel_idx, works)| {
            let (dg, dg_cum, valid) = jarzynski_estimator(&works, temperature);
            VoxelFreeEnergy {
                voxel_idx,
                delta_g: dg,
                delta_g_cumulant: dg_cum,
                n_samples: works.len(),
                cumulant_valid: valid,
            }
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════════
// LAYER 2: CROOKS INTERSECTION + BAR
// ═══════════════════════════════════════════════════════════════════════════════

/// Crooks Fluctuation Theorem: find ΔG where P_F(W) = P_R(-W)
///
/// Uses hysteresis profile (heating vs cooling counts per temperature bin).
/// log-ratio ln(P_F/P_R) crosses zero at W = ΔG.
pub fn crooks_intersection(bins: &[HysteresisBinData]) -> Option<f64> {
    if bins.len() < 3 {
        return None;
    }

    // Compute total counts for normalization
    let total_heat: f64 = bins.iter().map(|b| b.heating_count as f64).sum();
    let total_cool: f64 = bins.iter().map(|b| b.cooling_count as f64).sum();
    if total_heat < 1.0 || total_cool < 1.0 {
        return None;
    }

    // Compute log-ratio at each temperature bin
    // Use temperature as work proxy (W ~ kT)
    let mut log_ratios: Vec<(f64, f64)> = Vec::new(); // (W, ln(P_F/P_R))

    for bin in bins {
        let p_f = (bin.heating_count as f64 + 0.5) / (total_heat + bins.len() as f64 * 0.5);
        let p_r = (bin.cooling_count as f64 + 0.5) / (total_cool + bins.len() as f64 * 0.5);
        let w = KB * bin.temp_k as f64; // work proxy
        let lr = (p_f / p_r).ln();
        log_ratios.push((w, lr));
    }

    log_ratios.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Find zero crossing via linear interpolation
    for i in 1..log_ratios.len() {
        let (w0, lr0) = log_ratios[i - 1];
        let (w1, lr1) = log_ratios[i];

        if (lr0 <= 0.0 && lr1 >= 0.0) || (lr0 >= 0.0 && lr1 <= 0.0) {
            // Linear interpolation to find zero crossing
            if (lr1 - lr0).abs() < 1e-15 {
                continue;
            }
            let w_cross = w0 + (w1 - w0) * (-lr0) / (lr1 - lr0);
            return Some(w_cross);
        }
    }

    None
}

/// Bennett Acceptance Ratio (BAR) estimator.
///
/// Iteratively solves: Σ f(W_F - C) = Σ f(-W_R + C)
/// where f(x) = 1/(1 + exp(x/kT)) is the Fermi function.
pub fn bar_estimator(forward_work: &[f64], reverse_work: &[f64], temperature: f64) -> Option<f64> {
    let n_f = forward_work.len();
    let n_r = reverse_work.len();
    if n_f < 3 || n_r < 3 {
        return None;
    }

    let kt = KB * temperature;
    let ln_ratio = (n_f as f64 / n_r as f64).ln();

    // Initial guess: mean of forward and reverse
    let mean_f: f64 = forward_work.iter().sum::<f64>() / n_f as f64;
    let mean_r: f64 = reverse_work.iter().sum::<f64>() / n_r as f64;
    let mut c = (mean_f - mean_r) / 2.0;

    // Fermi function
    let fermi = |x: f64| -> f64 { 1.0 / (1.0 + (x / kt).exp()) };

    // Iterative BAR
    for _ in 0..100 {
        let sum_f: f64 = forward_work
            .iter()
            .map(|w| fermi(w - c + kt * ln_ratio))
            .sum();
        let sum_r: f64 = reverse_work
            .iter()
            .map(|w| fermi(-w + c - kt * ln_ratio))
            .sum();

        if sum_f.abs() < 1e-15 {
            break;
        }
        let c_new = kt * (sum_r / sum_f).ln() + kt * ln_ratio;

        if (c_new - c).abs() < 1e-6 {
            break;
        }
        c = c_new;
    }

    Some(c)
}

// ═══════════════════════════════════════════════════════════════════════════════
// LAYER 3: CHANNEL DECOMPOSITION
// ═══════════════════════════════════════════════════════════════════════════════

/// Channel-decomposed free energies
#[derive(Debug, Clone)]
struct ChannelDecomposition {
    delta_g_aromatic: f64,
    delta_g_dewetting: f64,
    delta_g_electrostatic: f64,
    delta_g_cooperative: f64,
    delta_g_total: f64,
}

/// Decompose free energy by spike source channel
#[cfg(feature = "gpu")]
fn channel_decomposition(
    spikes: &[GpuSpikeEvent],
    temperature: f64,
    cold_hold_steps: i32,
    ramp_steps: i32,
    warm_hold_steps: i32,
) -> ChannelDecomposition {
    // Separate heating-phase spikes by channel
    let mut uv_work: Vec<f64> = Vec::new();
    let mut lif_work: Vec<f64> = Vec::new();
    let mut efp_work: Vec<f64> = Vec::new();
    let mut all_work: Vec<f64> = Vec::new();

    let mut src_counts = [0usize; 4]; // [unknown, UV, LIF, EFP]
    for spike in spikes {
        let phase = derive_ramp_phase(spike.timestep, cold_hold_steps, ramp_steps, warm_hold_steps);
        if !matches!(phase, 1 | 2 | 3) {
            continue;
        } // forward process only
        let source = infer_spike_source(spike);
        if (source as usize) < 4 {
            src_counts[source as usize] += 1;
        }
        let work = spike_to_work(spike.intensity, source, spike.wavelength_nm);
        all_work.push(work);
        match source {
            1 => uv_work.push(work),
            2 => lif_work.push(work),
            3 => efp_work.push(work),
            _ => lif_work.push(work), // default to dewetting
        }
    }

    // Diagnostic: check raw field values for non-UV spikes
    let mut n_ve_nonzero = 0usize;
    let mut n_nne_ge2 = 0usize;
    let mut n_src3 = 0usize;
    for spike in spikes {
        if spike.spike_source == 3 {
            n_src3 += 1;
        }
        if spike.spike_source != 1 {
            if spike.vibrational_energy > 0.001 {
                n_ve_nonzero += 1;
            }
            if spike.n_nearby_excited >= 2 {
                n_nne_ge2 += 1;
            }
        }
    }
    log::info!("  STI channel decomposition: UV={} LIF={} EFP={} (raw_src3={} non-UV:ve>0={} nne>=2={}) total={}",
        src_counts[1], src_counts[2], src_counts[3], n_src3, n_ve_nonzero, n_nne_ge2,
        uv_work.len() + lif_work.len() + efp_work.len());

    let (dg_total, _, _) = jarzynski_estimator(&all_work, temperature);
    let (dg_uv, _, _) = jarzynski_estimator(&uv_work, temperature);
    let (dg_lif, _, _) = jarzynski_estimator(&lif_work, temperature);
    let (dg_efp, _, _) = jarzynski_estimator(&efp_work, temperature);

    // Cooperative = total - sum of independent channels
    let dg_cooperative = dg_total - (dg_uv + dg_lif + dg_efp);

    ChannelDecomposition {
        delta_g_aromatic: dg_uv,
        delta_g_dewetting: dg_lif,
        delta_g_electrostatic: dg_efp,
        delta_g_cooperative: dg_cooperative,
        delta_g_total: dg_total,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// LAYER 4: ARRHENIUS BARRIER ESTIMATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Estimate activation energy from temperature-dependent spike rates.
///
/// Linear regression of ln(spike_rate) vs 1/T gives slope = -E_a/k_B
fn arrhenius_barrier(
    temp_bins: &[(f64, usize)], // (temperature_K, spike_count)
    total_steps_per_bin: usize,
) -> Option<f64> {
    // Filter bins with nonzero counts
    let valid_bins: Vec<(f64, f64)> = temp_bins
        .iter()
        .filter(|(t, count)| *t > 10.0 && *count > 0 && total_steps_per_bin > 0)
        .map(|(t, count)| {
            let rate = *count as f64 / total_steps_per_bin as f64;
            (1.0 / t, rate.ln())
        })
        .collect();

    if valid_bins.len() < 3 {
        return None;
    }

    // Linear regression: ln(rate) = -E_a/k_B * (1/T) + intercept
    let n = valid_bins.len() as f64;
    let sum_x: f64 = valid_bins.iter().map(|(x, _)| x).sum();
    let sum_y: f64 = valid_bins.iter().map(|(_, y)| y).sum();
    let sum_xy: f64 = valid_bins.iter().map(|(x, y)| x * y).sum();
    let sum_x2: f64 = valid_bins.iter().map(|(x, _)| x * x).sum();

    let denom = n * sum_x2 - sum_x * sum_x;
    if denom.abs() < 1e-30 {
        return None;
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let e_a = -slope * KB; // slope = -E_a/k_B, so E_a = -slope * k_B

    if e_a.is_finite() && e_a > 0.0 {
        Some(e_a)
    } else {
        None
    }
}

/// Compute Arrhenius barriers per wavelength from spike events during heating.
///
/// Wavelength → depth interpretation:
/// - 280nm: deep core
/// - 274nm: mid-depth
/// - 258nm: surface
/// - 211nm: disulfide bridges
#[cfg(feature = "gpu")]
fn arrhenius_by_wavelength(
    spikes: &[GpuSpikeEvent],
    protocol_start_temp: f64,
    protocol_end_temp: f64,
    ramp_steps: i32,
    cold_hold_steps: i32,
    warm_hold_steps: i32,
    n_temp_bins: usize,
) -> HashMap<String, f64> {
    let mut results = HashMap::new();

    let wavelengths = [280.0f32, 274.0, 258.0, 254.0, 211.0];
    let wavelength_labels = ["280nm", "274nm", "258nm", "254nm", "211nm"];

    let temp_range = protocol_end_temp - protocol_start_temp;
    let bin_width = temp_range / n_temp_bins as f64;
    let steps_per_bin = (ramp_steps as f64 / n_temp_bins as f64).ceil() as usize;

    // First: per-wavelength analysis (UV spikes only, original approach)
    for (wl, label) in wavelengths.iter().zip(wavelength_labels.iter()) {
        let wl_tolerance = 3.0f32;

        let mut temp_counts = vec![(0.0f64, 0usize); n_temp_bins];
        for (i, tc) in temp_counts.iter_mut().enumerate() {
            tc.0 = protocol_start_temp + (i as f64 + 0.5) * bin_width;
        }

        for spike in spikes {
            let phase =
                derive_ramp_phase(spike.timestep, cold_hold_steps, ramp_steps, warm_hold_steps);
            if !matches!(phase, 1 | 2 | 3) {
                continue;
            }
            let source = infer_spike_source(spike);
            if source != 1 {
                continue;
            }
            if (spike.wavelength_nm - wl).abs() > wl_tolerance {
                continue;
            }

            let temp = if phase == 3 {
                protocol_end_temp
            } else {
                let ramp_start_step = cold_hold_steps as f64;
                let progress =
                    ((spike.timestep as f64) - ramp_start_step) / (ramp_steps as f64).max(1.0);
                protocol_start_temp + progress.clamp(0.0, 1.0) * temp_range
            };

            let bin_idx = ((temp - protocol_start_temp) / bin_width) as usize;
            if bin_idx < n_temp_bins {
                temp_counts[bin_idx].1 += 1;
            }
        }

        if let Some(e_a) = arrhenius_barrier(&temp_counts, steps_per_bin) {
            results.insert(label.to_string(), e_a);
        }
    }

    // Second: all-source pooled analysis (uses ALL spikes regardless of source)
    // This provides the kinetic barrier estimate even when UV spikes alone are
    // too sparse for per-wavelength fitting (common with 8 streams).
    {
        let mut temp_counts = vec![(0.0f64, 0usize); n_temp_bins];
        for (i, tc) in temp_counts.iter_mut().enumerate() {
            tc.0 = protocol_start_temp + (i as f64 + 0.5) * bin_width;
        }

        for spike in spikes {
            let phase =
                derive_ramp_phase(spike.timestep, cold_hold_steps, ramp_steps, warm_hold_steps);
            if !matches!(phase, 1 | 2 | 3) {
                continue;
            }

            let temp = if phase == 3 {
                protocol_end_temp
            } else {
                let ramp_start_step = cold_hold_steps as f64;
                let progress =
                    ((spike.timestep as f64) - ramp_start_step) / (ramp_steps as f64).max(1.0);
                protocol_start_temp + progress.clamp(0.0, 1.0) * temp_range
            };

            let bin_idx = ((temp - protocol_start_temp) / bin_width) as usize;
            if bin_idx < n_temp_bins {
                temp_counts[bin_idx].1 += 1;
            }
        }

        if let Some(e_a) = arrhenius_barrier(&temp_counts, steps_per_bin) {
            results.insert("all_sources".to_string(), e_a);
        }
    }

    results
}

// ═══════════════════════════════════════════════════════════════════════════════
// LAYER 5: BINDING FREE ENERGY (COMBINED)
// ═══════════════════════════════════════════════════════════════════════════════

/// STI analysis configuration
#[derive(Debug, Clone)]
pub struct StiConfig {
    pub temperature: f64,
    pub protocol_start_temp: f64,
    pub protocol_end_temp: f64,
    pub ramp_steps: i32,
    pub cold_hold_steps: i32,
    pub warm_hold_steps: i32,
}

impl Default for StiConfig {
    fn default() -> Self {
        Self {
            temperature: DEFAULT_TEMP,
            protocol_start_temp: 50.0,
            protocol_end_temp: 300.0,
            ramp_steps: 6000,
            cold_hold_steps: 14000,
            warm_hold_steps: 15000,
        }
    }
}

/// Run complete STI analysis on a pocket's spike events.
///
/// This is the top-level entry point for Section 1.
/// Returns BindingFreeEnergy with all thermodynamic quantities.
#[cfg(feature = "gpu")]
pub fn compute_binding_free_energy(
    pocket_spikes: &[GpuSpikeEvent],
    hysteresis_bins: Option<&[HysteresisBinData]>,
    delta_g_branching: Option<f64>,
    config: &StiConfig,
) -> BindingFreeEnergy {
    let temperature = config.temperature;
    let kt = KB * temperature;

    // Layer 1: Jarzynski per-voxel
    let voxel_energies = jarzynski_per_voxel(
        pocket_spikes,
        temperature,
        config.cold_hold_steps,
        config.ramp_steps,
        config.warm_hold_steps,
    );
    let n_voxels = voxel_energies.len();
    let cumulant_valid = voxel_energies.iter().any(|v| v.cumulant_valid);

    // Layer 3: Channel decomposition
    let decomp = channel_decomposition(
        pocket_spikes,
        temperature,
        config.cold_hold_steps,
        config.ramp_steps,
        config.warm_hold_steps,
    );

    // Layer 2: Crooks intersection (if hysteresis data available)
    let delta_g_crooks = hysteresis_bins.and_then(crooks_intersection);

    // Layer 2b: BAR estimator
    let delta_g_bar = if let Some(bins) = hysteresis_bins {
        // Convert heating/cooling counts to work distributions
        let forward: Vec<f64> = bins
            .iter()
            .filter(|b| b.heating_count > 0)
            .map(|b| KB * b.temp_k as f64 * (b.heating_count as f64).ln())
            .collect();
        let reverse: Vec<f64> = bins
            .iter()
            .filter(|b| b.cooling_count > 0)
            .map(|b| KB * b.temp_k as f64 * (b.cooling_count as f64).ln())
            .collect();
        bar_estimator(&forward, &reverse, temperature)
    } else {
        None
    };

    // Layer 4: Arrhenius barriers
    let activation_energies = arrhenius_by_wavelength(
        pocket_spikes,
        config.protocol_start_temp,
        config.protocol_end_temp,
        config.ramp_steps,
        config.cold_hold_steps,
        config.warm_hold_steps,
        20, // 20 temperature bins
    );

    let e_a_mean: Option<f64> = if !activation_energies.is_empty() {
        let sum: f64 = activation_energies.values().sum();
        Some(sum / activation_energies.len() as f64)
    } else {
        None
    };

    // Layer 5: Combined binding free energy
    // ΔG_bind = ΔG_aromatic + ΔG_dewetting + ΔG_electrostatic + ΔG_cooperative
    //           - ΔG_pocket_opening
    let delta_g_pocket_opening = delta_g_crooks.unwrap_or(0.0);
    let delta_g_bind = decomp.delta_g_total - delta_g_pocket_opening;

    // Kinetic accessibility from mean activation energy
    let kinetic_accessibility = if let Some(ea) = e_a_mean {
        (-ea / kt).exp().min(1.0)
    } else {
        1.0 // assume no barrier if insufficient data
    };

    // Effective ΔG = ΔG_bind + kT·ln(kinetic_accessibility)
    let effective_dg = delta_g_bind + kt * kinetic_accessibility.max(1e-30).ln();

    BindingFreeEnergy {
        delta_g_sti_kcal_mol: decomp.delta_g_total,
        delta_g_aromatic_kcal_mol: decomp.delta_g_aromatic,
        delta_g_dewetting_kcal_mol: decomp.delta_g_dewetting,
        delta_g_electrostatic_kcal_mol: decomp.delta_g_electrostatic,
        delta_g_cooperative_kcal_mol: decomp.delta_g_cooperative,
        delta_g_crooks_kcal_mol: delta_g_crooks,
        delta_g_bar_kcal_mol: delta_g_bar,
        delta_g_branching_kcal_mol: delta_g_branching,
        activation_energy_by_wavelength: activation_energies,
        activation_energy_mean_kcal_mol: e_a_mean,
        effective_delta_g_kcal_mol: effective_dg,
        kinetic_accessibility,
        cumulant_valid,
        n_voxels,
        n_spikes: pocket_spikes.len(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// STREAM-CONDITIONED DIAGNOSTICS (ADDITIVE; NO THERMODYNAMIC POOLING)
// ═══════════════════════════════════════════════════════════════════════════════

/// The caller binds this stream ID to the original event ordinal ownership and
/// supplies the protocol actually executed by that worker. A temperature label
/// cannot establish source ownership. This does not establish an equilibrium
/// ensemble or identify intensity-derived proxies with trajectory work.
#[derive(Debug, Clone)]
pub struct StreamStiCondition {
    pub stream_id: usize,
    pub config: StiConfig,
}

/// Serializable copy of the exact parameters passed to the legacy numerical
/// operator. New diagnostics leave the existing StiConfig type and API intact.
#[derive(Debug, Clone, Serialize)]
pub struct ConditionedStiConfigSnapshot {
    pub temperature: f64,
    pub protocol_start_temp: f64,
    pub protocol_end_temp: f64,
    pub ramp_steps: i32,
    pub cold_hold_steps: i32,
    pub warm_hold_steps: i32,
}

#[derive(Debug, Clone, Serialize)]
pub struct StreamStiDiagnostic {
    pub stream_id: usize,
    pub config: ConditionedStiConfigSnapshot,
    pub n_input_events: usize,
    /// Exact population selected by the legacy phase classifier, phases 1..3.
    pub n_legacy_forward_events: usize,
    /// Legacy reverse bucket, not a claim of a conjugate reverse trajectory.
    pub n_legacy_reverse_events: usize,
    pub status: &'static str,
    /// Existing numerical operator applied to this stream alone. This remains
    /// an intensity-derived diagnostic, without a physical free-energy claim.
    pub legacy_diagnostic: Option<BindingFreeEnergy>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConditionedStiDiagnostics {
    pub schema: &'static str,
    pub n_input_events: usize,
    /// Same order as the caller's declared stream conditions, including empty
    /// streams; neither event count nor an estimated energy defines a weight.
    pub streams: Vec<StreamStiDiagnostic>,
    /// No common-ensemble/work-law merge is defined by the existing operator.
    /// This is always None, serialized as null, never a substituted zero.
    pub pooled_binding_free_energy_kcal_mol: Option<f64>,
    pub interpretation: &'static str,
    pub arrhenius_normalization: &'static str,
}

/// Add stream-specific diagnostics without changing the historical scalar used
/// by the existing ranker. Every tagged input event enters exactly one declared
/// stream bucket in its original within-stream order; unknown/duplicate stream
/// declarations and nonfinite work inputs fail before numerical evaluation.
///
/// No timestamps, intensities or condition IDs are rewritten. The existing
/// compute_binding_free_energy function is unchanged and this companion does
/// not define a replacement rank score. Actual clock/exposure accounting belongs
/// to the caller's separate temporal authority. In particular, legacy Arrhenius
/// bins are not relabeled as measured time-normalized rates.
#[cfg(feature = "gpu")]
pub fn compute_conditioned_binding_diagnostics(
    tagged_spikes: &[(usize, &GpuSpikeEvent)],
    conditions: &[StreamStiCondition],
) -> Result<ConditionedStiDiagnostics, String> {
    if conditions.is_empty() {
        return Err("conditioned STI requires explicit stream conditions".to_string());
    }
    let mut indices = HashMap::<usize, usize>::new();
    for (index, condition) in conditions.iter().enumerate() {
        if indices.insert(condition.stream_id, index).is_some() {
            return Err(format!("duplicate conditioned STI stream {}", condition.stream_id));
        }
        let config = &condition.config;
        if !config.temperature.is_finite() || config.temperature <= 0.0
            || !config.protocol_start_temp.is_finite() || config.protocol_start_temp <= 0.0
            || !config.protocol_end_temp.is_finite()
            || config.protocol_end_temp <= config.protocol_start_temp
            || config.cold_hold_steps < 0 || config.ramp_steps <= 0
            || config.warm_hold_steps < 0
        {
            return Err(format!("invalid conditioned STI configuration for stream {}", condition.stream_id));
        }
        config.cold_hold_steps.checked_add(config.ramp_steps)
            .and_then(|boundary| boundary.checked_add(config.warm_hold_steps))
            .ok_or_else(|| format!("conditioned STI phase boundary overflow for stream {}", condition.stream_id))?;
    }
    let mut by_stream: Vec<Vec<GpuSpikeEvent>> = (0..conditions.len()).map(|_| Vec::new()).collect();
    for &(stream_id, spike) in tagged_spikes {
        let index = *indices.get(&stream_id)
            .ok_or_else(|| format!("undeclared conditioned STI stream {}", stream_id))?;
        if spike.timestep < 0 || !spike.intensity.is_finite() || !spike.wavelength_nm.is_finite() {
            return Err(format!("invalid conditioned STI event in stream {}", stream_id));
        }
        by_stream[index].push(*spike);
    }
    let mut streams = Vec::with_capacity(conditions.len());
    for (condition, spikes) in conditions.iter().zip(by_stream.iter()) {
        let config = &condition.config;
        let n_forward = spikes.iter().filter(|spike| {
            matches!(derive_ramp_phase(spike.timestep, config.cold_hold_steps,
                config.ramp_steps, config.warm_hold_steps), 1 | 2 | 3)
        }).count();
        let (status, legacy_diagnostic) = if spikes.is_empty() {
            ("no_site_events_in_declared_stream", None)
        } else if n_forward < 10 {
            ("insufficient_forward_events_for_conditioned_diagnostic", None)
        } else {
            ("legacy_operator_per_stream_diagnostic_only", Some(
                compute_binding_free_energy(spikes, None, None, config)
            ))
        };
        streams.push(StreamStiDiagnostic {
            stream_id: condition.stream_id,
            config: ConditionedStiConfigSnapshot {
                temperature: config.temperature,
                protocol_start_temp: config.protocol_start_temp,
                protocol_end_temp: config.protocol_end_temp,
                ramp_steps: config.ramp_steps,
                cold_hold_steps: config.cold_hold_steps,
                warm_hold_steps: config.warm_hold_steps,
            },
            n_input_events: spikes.len(),
            n_legacy_forward_events: n_forward,
            n_legacy_reverse_events: spikes.len() - n_forward,
            status,
            legacy_diagnostic,
        });
    }
    Ok(ConditionedStiDiagnostics {
        schema: "prismos.sti.stream_conditioned_diagnostics.v1",
        n_input_events: tagged_spikes.len(),
        streams,
        pooled_binding_free_energy_kcal_mol: None,
        interpretation: "Per-stream intensity-derived legacy diagnostics; no common-ensemble, trajectory-work, conjugate-reverse or pooled physical free-energy claim. Existing run-level scalar/ranker remains a separately identified legacy heuristic.",
        arrhenius_normalization: "Unchanged legacy operator: nominal equal ceil(ramp_steps/20) step exposures, cold-hold counts may enter the first temperature bin, warm endpoint counts outside bins; these diagnostics do not constitute actual clock-normalized rates.",
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 4: SPIKE PHARMACOPHORE SYNTHESIS
// ═══════════════════════════════════════════════════════════════════════════════

/// A pharmacophore feature derived from spike cluster analysis
#[derive(Debug, Clone, Serialize)]
pub struct PharmacophoreFeature {
    /// Centroid position of this feature cluster [x, y, z] in Å
    pub position: [f32; 3],
    /// Feature type from channel mapping
    pub feature_type: String,
    /// Normalized spike intensity (0-1)
    pub strength: f32,
    /// Binding order: 1 = forms first (lowest TTFS)
    pub ttfs_rank: u32,
    /// Cluster ID for cooperative binding constraint
    pub synchrony_group: u32,
    /// Source channel (1=UV, 2=LIF, 3=EFP)
    pub channel_source: u8,
    /// UV wavelength if applicable
    pub wavelength_nm: Option<f32>,
}

/// Map channel + wavelength to pharmacophore feature type
fn channel_to_feature_type(source: i32, wavelength_nm: f32, aromatic_type: i32) -> String {
    match source {
        1 => {
            // UV channel — wavelength determines feature type
            match wavelength_nm as i32 {
                278..=282 | 272..=276 => "aromatic_pi_stacking".to_string(),
                256..=260 => "hydrophobic_aromatic".to_string(),
                252..=255 => "aliphatic_hydrophobic".to_string(),
                209..=213 => match aromatic_type {
                    3 => "disulfide_bridge".to_string(),
                    _ => "aromatic_pi_stacking".to_string(),
                },
                _ => "hydrophobic_aromatic".to_string(),
            }
        }
        2 => "hydrophobic_exclusion_volume".to_string(),
        3 => {
            // EFP — charge determines donor/acceptor
            "hbond_donor_positive".to_string() // default; would need charge info for full classification
        }
        _ => "unknown".to_string(),
    }
}

/// Generate pharmacophore features from spike events for a detected pocket.
///
/// Groups spikes by channel and spatial proximity, then maps each cluster
/// to a pharmacophore feature type.
#[cfg(feature = "gpu")]
pub fn generate_pharmacophore_features(
    pocket_spikes: &[GpuSpikeEvent],
    cluster_id: u32,
) -> Vec<PharmacophoreFeature> {
    if pocket_spikes.is_empty() {
        return Vec::new();
    }

    // Group spikes by (inferred_source, wavelength_bucket) → feature clusters
    let mut feature_groups: HashMap<(i32, i32), Vec<&GpuSpikeEvent>> = HashMap::new();
    for spike in pocket_spikes {
        let source = infer_spike_source(spike);
        let wl_bucket = (spike.wavelength_nm / 10.0).round() as i32 * 10; // 10nm buckets
        feature_groups
            .entry((source, wl_bucket))
            .or_default()
            .push(spike);
    }

    // Normalize intensities globally
    let max_intensity = pocket_spikes
        .iter()
        .map(|s| s.intensity)
        .fold(0.0f32, f32::max)
        .max(1e-6);

    // Sort groups by earliest timestep (for TTFS ranking)
    let mut group_entries: Vec<((i32, i32), Vec<&GpuSpikeEvent>)> =
        feature_groups.into_iter().collect();
    group_entries
        .sort_by_key(|(_, spikes)| spikes.iter().map(|s| s.timestep).min().unwrap_or(i32::MAX));

    let mut features = Vec::new();
    for (rank, ((source, _wl_bucket), spikes)) in group_entries.iter().enumerate() {
        if spikes.is_empty() {
            continue;
        }

        // Compute centroid of this feature group
        let n = spikes.len() as f32;
        let cx = spikes.iter().map(|s| s.position[0]).sum::<f32>() / n;
        let cy = spikes.iter().map(|s| s.position[1]).sum::<f32>() / n;
        let cz = spikes.iter().map(|s| s.position[2]).sum::<f32>() / n;

        let mean_intensity = spikes.iter().map(|s| s.intensity).sum::<f32>() / n;
        let mean_wavelength = spikes.iter().map(|s| s.wavelength_nm).sum::<f32>() / n;
        let aromatic_type = spikes.first().map(|s| s.aromatic_type).unwrap_or(-1);

        let feature_type = channel_to_feature_type(*source, mean_wavelength, aromatic_type);

        features.push(PharmacophoreFeature {
            position: [cx, cy, cz],
            feature_type,
            strength: mean_intensity / max_intensity,
            ttfs_rank: (rank + 1) as u32,
            synchrony_group: cluster_id,
            channel_source: *source as u8,
            wavelength_nm: if *source == 1 && mean_wavelength > 200.0 {
                Some(mean_wavelength)
            } else {
                None
            },
        });
    }

    features
}

/// Generate PGMG-compatible pharmacophore JSON.
///
/// PGMG (Pharmacophore-Guided Molecular Generation) expects:
/// { "pharmacophore": [ { "type": "...", "center": [x,y,z], "radius": r }, ... ] }
#[cfg(feature = "gpu")]
pub fn generate_pgmg_pharmacophore(features: &[PharmacophoreFeature]) -> serde_json::Value {
    let pgmg_features: Vec<serde_json::Value> = features
        .iter()
        .map(|f| {
            // Map our types to PGMG types
            let pgmg_type = match f.feature_type.as_str() {
                "aromatic_pi_stacking" => "Aromatic",
                "hydrophobic_aromatic"
                | "aliphatic_hydrophobic"
                | "hydrophobic_exclusion_volume" => "Hydrophobic",
                "hbond_donor_positive" => "HBondDonor",
                "hbond_acceptor_negative" => "HBondAcceptor",
                _ => "Hydrophobic",
            };

            serde_json::json!({
                "type": pgmg_type,
                "center": f.position,
                "radius": 1.5, // default tolerance radius in Å
                "weight": f.strength,
            })
        })
        .collect();

    serde_json::json!({
        "pharmacophore": pgmg_features,
        "n_features": features.len(),
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// PERTURBATION-RESPONSE DECAY ANALYSIS
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of perturbation-response decay fitting
#[derive(Debug, Clone, Serialize)]
pub struct PerturbationResponse {
    /// "power_law", "exponential", or "insufficient_data"
    pub response_type: String,
    /// R² of exponential fit
    pub exp_r_squared: Option<f64>,
    /// R² of power-law fit
    pub power_law_r_squared: Option<f64>,
    /// Decay time constant τ (for exponential)
    pub tau_decay: Option<f64>,
    /// Power-law exponent α
    pub alpha: Option<f64>,
}

/// Analyze spike-rate response to a perturbation.
///
/// Given spike counts per timestep after a perturbation, fits both
/// exponential (r(t) = A·exp(-t/τ)) and power-law (r(t) = A·t^(-α)).
/// If power-law R² > exponential R² + 0.05 → near-critical (SOC).
pub fn analyze_perturbation_response(
    spike_rates_after_perturbation: &[f64],
) -> PerturbationResponse {
    if spike_rates_after_perturbation.len() < 10 {
        return PerturbationResponse {
            response_type: "insufficient_data".to_string(),
            exp_r_squared: None,
            power_law_r_squared: None,
            tau_decay: None,
            alpha: None,
        };
    }

    // Filter out zero rates (can't take log)
    let valid: Vec<(f64, f64)> = spike_rates_after_perturbation
        .iter()
        .enumerate()
        .filter(|(_, &r)| r > 0.0)
        .map(|(t, &r)| ((t + 1) as f64, r))
        .collect();

    if valid.len() < 5 {
        return PerturbationResponse {
            response_type: "insufficient_data".to_string(),
            exp_r_squared: None,
            power_law_r_squared: None,
            tau_decay: None,
            alpha: None,
        };
    }

    // Exponential fit: ln(r) = ln(A) - t/τ → linear in t vs ln(r)
    let exp_x: Vec<f64> = valid.iter().map(|(t, _)| *t).collect();
    let exp_y: Vec<f64> = valid.iter().map(|(_, r)| r.ln()).collect();
    let (exp_slope, _exp_intercept, exp_r2) = linear_regression_f64(&exp_x, &exp_y);
    let tau_decay = if exp_slope.abs() > 1e-15 {
        Some(-1.0 / exp_slope)
    } else {
        None
    };

    // Power-law fit: ln(r) = ln(A) - α·ln(t) → linear in ln(t) vs ln(r)
    let pl_x: Vec<f64> = valid.iter().map(|(t, _)| t.ln()).collect();
    let pl_y: Vec<f64> = valid.iter().map(|(_, r)| r.ln()).collect();
    let (pl_slope, _pl_intercept, pl_r2) = linear_regression_f64(&pl_x, &pl_y);
    let alpha = Some(-pl_slope);

    // Classification
    let response_type = if pl_r2 > exp_r2 + 0.05 {
        "power_law".to_string() // near-critical dynamics (SOC confirmation)
    } else {
        "exponential".to_string()
    };

    PerturbationResponse {
        response_type,
        exp_r_squared: Some(exp_r2),
        power_law_r_squared: Some(pl_r2),
        tau_decay,
        alpha,
    }
}

/// Linear regression for f64 slices. Returns (slope, intercept, R²).
fn linear_regression_f64(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
    let n = x.len() as f64;
    if n < 2.0 {
        return (0.0, 0.0, 0.0);
    }

    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();
    let sum_x2: f64 = x.iter().map(|&xi| xi * xi).sum();

    let denom = n * sum_x2 - sum_x * sum_x;
    if denom.abs() < 1e-30 {
        return (0.0, 0.0, 0.0);
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;

    let mean_y = sum_y / n;
    let ss_tot: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();
    let ss_res: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| {
            let pred = slope * xi + intercept;
            (yi - pred).powi(2)
        })
        .sum();
    let r_sq = if ss_tot > 1e-30 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    };

    (slope, intercept, r_sq)
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jarzynski_zero_work() {
        // Zero work → ΔG = 0
        let works = vec![0.0; 10];
        let (dg, dg_cum, valid) = jarzynski_estimator(&works, 300.0);
        assert!(
            (dg).abs() < 1e-10,
            "ΔG should be ~0 for zero work, got {}",
            dg
        );
        assert!((dg_cum).abs() < 1e-10);
        assert!(valid);
    }

    #[test]
    fn test_jarzynski_constant_work() {
        // Constant work W for all samples → ΔG = W
        let w = 2.0;
        let works = vec![w; 100];
        let (dg, dg_cum, _) = jarzynski_estimator(&works, 300.0);
        assert!(
            (dg - w).abs() < 1e-10,
            "ΔG should equal W={}, got {}",
            w,
            dg
        );
        assert!((dg_cum - w).abs() < 1e-10);
    }

    #[test]
    fn test_jarzynski_empty() {
        let (dg, _, _) = jarzynski_estimator(&[], 300.0);
        assert_eq!(dg, 0.0);
    }

    /// Regression lock for the defect recorded at nhs_rt_full.rs:37127,
    /// "Jarzynski produces -2795 to +147 kcal/mol garbage".
    ///
    /// A high-dissipation work distribution used to be routed to the second
    /// order cumulant even though var_W/(kT)^2 was far outside its validity
    /// domain, and the unbounded -var_W/(2kT) term produced values tens to
    /// thousands of kcal/mol below min(W). Under the previous selection this
    /// case returned about -51 kcal/mol; the Jarzynski inequality forbids
    /// anything above <W>, and the exponential average cannot fall below
    /// min(W) - kT ln N.
    #[test]
    fn test_jarzynski_high_dissipation_stays_physically_bounded() {
        let kt = KB * 300.0;
        // deterministic spread of ~40 kcal/mol, far larger than kT ~ 0.6
        let works: Vec<f64> = (0..20).map(|i| 10.0 + 2.0 * i as f64).collect();
        let n = works.len() as f64;
        let mean_w: f64 = works.iter().sum::<f64>() / n;
        let min_w = works.iter().cloned().fold(f64::INFINITY, f64::min);
        let lower = min_w - kt * n.ln();

        let (dg, _, _) = jarzynski_estimator(&works, 300.0);

        assert!(
            dg <= mean_w + 1e-9,
            "Jarzynski inequality violated: ΔG={} exceeds <W>={}",
            dg,
            mean_w
        );
        assert!(
            dg >= lower - 1e-9,
            "ΔG={} fell below the exponential-average bound min(W)-kT·lnN={}",
            dg,
            lower
        );
        assert!(
            dg > -100.0,
            "ΔG={} reproduces the unbounded cumulant defect",
            dg
        );
    }

    /// The degeneracy test must be a ratio: n_eff is an inverse participation
    /// ratio and is >= 1 by construction, so a bare-count threshold made the
    /// estimator answer differently for the same physics either side of n=10.
    #[test]
    fn test_jarzynski_no_sample_count_discontinuity() {
        let kt = KB * 300.0;
        let make = |n: usize| -> Vec<f64> { (0..n).map(|i| 10.0 + 2.0 * (i % 20) as f64).collect() };
        for n in [8usize, 10, 12, 20, 50] {
            let works = make(n);
            let count = works.len() as f64;
            let mean_w: f64 = works.iter().sum::<f64>() / count;
            let min_w = works.iter().cloned().fold(f64::INFINITY, f64::min);
            let lower = min_w - kt * count.ln();
            let (dg, _, _) = jarzynski_estimator(&works, 300.0);
            assert!(
                dg >= lower - 1e-9 && dg <= mean_w + 1e-9,
                "n={} left the admissible interval: ΔG={} not in [{}, {}]",
                n,
                dg,
                lower,
                mean_w
            );
        }
    }

    #[test]
    fn test_jarzynski_regime_labels_degeneracy() {
        // near-equilibrium: spread well under kT, exponential average converges
        let tight: Vec<f64> = (0..30).map(|i| 2.0 + 0.001 * i as f64).collect();
        assert_eq!(jarzynski_regime(&tight, 300.0), JarzynskiRegime::Exact);
        // high dissipation: the value is a bound, and the label must say so
        let spread: Vec<f64> = (0..20).map(|i| 10.0 + 2.0 * i as f64).collect();
        assert_eq!(
            jarzynski_regime(&spread, 300.0),
            JarzynskiRegime::ExactDegenerateBoundOnly
        );
        assert_eq!(jarzynski_regime(&[], 300.0), JarzynskiRegime::Empty);
    }

    #[test]
    fn test_crooks_no_data() {
        assert!(crooks_intersection(&[]).is_none());
    }

    #[test]
    fn test_crooks_symmetric() {
        // Symmetric distribution → ΔG should be near zero
        let bins: Vec<HysteresisBinData> = (0..10)
            .map(|i| HysteresisBinData {
                temp_k: 100.0 + i as f32 * 20.0,
                heating_count: 100,
                cooling_count: 100,
            })
            .collect();
        let dg = crooks_intersection(&bins);
        // Symmetric → log-ratio is 0 everywhere → intersection at first bin
        // With pseudocounts, should still be very close to symmetric
        if let Some(v) = dg {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_arrhenius_insufficient_bins() {
        let bins = vec![(300.0, 100usize), (310.0, 120)];
        assert!(arrhenius_barrier(&bins, 1000).is_none());
    }

    #[test]
    fn test_arrhenius_constant_rate() {
        // Constant rate → slope = 0 → E_a = 0
        let bins: Vec<(f64, usize)> = (0..10).map(|i| (200.0 + i as f64 * 10.0, 100)).collect();
        let e_a = arrhenius_barrier(&bins, 1000);
        // With constant rate, ln(rate) is constant, slope ≈ 0, E_a ≈ 0
        // May return None if E_a <= 0 (depending on numerical noise)
        if let Some(ea) = e_a {
            assert!(ea.abs() < 0.1, "E_a should be near 0, got {}", ea);
        }
    }

    #[test]
    fn test_bar_empty() {
        assert!(bar_estimator(&[], &[], 300.0).is_none());
        assert!(bar_estimator(&[1.0, 2.0, 3.0], &[], 300.0).is_none());
    }

    #[test]
    fn test_bar_symmetric() {
        // Symmetric forward/reverse → ΔG ≈ 0
        let forward = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let reverse = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let dg = bar_estimator(&forward, &reverse, 300.0);
        assert!(dg.is_some());
    }

    #[test]
    fn test_spike_to_work_channels() {
        // UV channel should give positive work
        let w_uv = spike_to_work(1.0, 1, 280.0);
        assert!(w_uv > 0.0);

        // LIF channel
        let w_lif = spike_to_work(1.0, 2, 0.0);
        assert!((w_lif - HYDRATION_ENERGY).abs() < 1e-10);

        // EFP channel
        let w_efp = spike_to_work(1.0, 3, 0.0);
        assert!(w_efp > 0.0);
    }
}
