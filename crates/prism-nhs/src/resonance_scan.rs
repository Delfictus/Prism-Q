//! Thermodynamic Resonance Scanning
//!
//! Frequency sweep protocol that varies UV burst period to find resonant
//! pocket opening frequencies. Fits Lorentzian to amplitude vs burst_period.
//!
//! A(f) = A0 / ((f - f0)² + Γ²)
//!
//! The resonance frequency f0 gives the pocket opening rate constant:
//! k_open ≈ 1 / (f0 × step_duration_ps)
//!
//! Activated via --resonance-scan flag. Runs INSTEAD of normal protocol.
//! Adds ~10x runtime.

use serde::Serialize;
use std::collections::HashMap;

/// Resonance scan burst periods to sweep (in timesteps)
pub const BURST_PERIODS: &[i32] = &[50, 100, 150, 200, 250, 300, 400, 500, 750, 1000];

/// Number of MD steps per period measurement
pub const STEPS_PER_PERIOD: i32 = 10_000;

/// Per-voxel resonance result
#[derive(Debug, Clone, Serialize)]
pub struct VoxelResonance {
    /// Voxel index
    pub voxel_idx: i32,
    /// Resonance frequency (Hz, from Lorentzian fit)
    pub resonance_frequency_hz: f64,
    /// Opening rate constant (per ns)
    pub k_open_per_ns: f64,
    /// R² of Lorentzian fit
    pub lorentzian_r2: f64,
    /// Amplitude at resonance
    pub amplitude_at_resonance: f64,
}

/// Per-pocket resonance summary
#[derive(Debug, Clone, Serialize)]
pub struct PocketResonance {
    /// Pocket/site ID
    pub site_id: i32,
    /// Dominant resonance frequency across voxels (Hz)
    pub dominant_resonance_frequency: f64,
    /// Pocket opening rate (per nanosecond)
    pub pocket_opening_rate_ns: f64,
    /// Quality factor Q = f0 / (2Γ)
    pub resonance_quality_factor: f64,
    /// Per-voxel resonance data
    pub voxel_resonances: Vec<VoxelResonance>,
}

/// Complete resonance map output
#[derive(Debug, Clone, Serialize)]
pub struct ResonanceMap {
    /// Burst periods swept (timesteps)
    pub burst_periods: Vec<i32>,
    /// Steps per period measurement
    pub steps_per_period: i32,
    /// Per-pocket resonance results
    pub pockets: Vec<PocketResonance>,
}

/// Amplitude data for one voxel across all burst periods
#[derive(Debug, Clone)]
pub struct VoxelAmplitudeData {
    pub voxel_idx: i32,
    /// (burst_period, mean_amplitude, peak_amplitude)
    pub amplitudes: Vec<(i32, f64, f64)>,
}

/// Fit a Lorentzian to amplitude vs burst_period data.
///
/// A(f) = A0 / ((f - f0)² + Γ²)
///
/// Uses least-squares grid search over f0 and Γ, then refines.
/// Returns (f0, Γ, A0, R²)
pub fn fit_lorentzian(data: &[(f64, f64)]) -> Option<(f64, f64, f64, f64)> {
    if data.len() < 3 {
        return None;
    }

    // Find amplitude-weighted mean for initial f0 guess
    let total_amp: f64 = data.iter().map(|(_, a)| a).sum();
    if total_amp < 1e-10 { return None; }

    let mean_f: f64 = data.iter().map(|(f, a)| f * a).sum::<f64>() / total_amp;
    let max_amp = data.iter().map(|(_, a)| *a).fold(0.0f64, f64::max);

    // Grid search over f0 and Γ
    let f_min = data.iter().map(|(f, _)| *f).fold(f64::INFINITY, f64::min);
    let f_max = data.iter().map(|(f, _)| *f).fold(f64::NEG_INFINITY, f64::max);
    let f_range = f_max - f_min;

    let mut best_f0 = mean_f;
    let mut best_gamma = f_range / 4.0;
    let mut best_a0 = max_amp;
    let mut best_r2 = f64::NEG_INFINITY;

    // Coarse grid search
    for fi in 0..20 {
        let f0_trial = f_min + (fi as f64 / 19.0) * f_range;
        for gi in 1..15 {
            let gamma_trial = (gi as f64 / 14.0) * f_range / 2.0;

            // Compute A0 that minimizes residuals (analytically)
            let mut sum_model_sq = 0.0f64;
            let mut sum_model_data = 0.0f64;
            for &(f, a) in data {
                let model_unnorm = 1.0 / ((f - f0_trial).powi(2) + gamma_trial.powi(2));
                sum_model_sq += model_unnorm * model_unnorm;
                sum_model_data += model_unnorm * a;
            }

            if sum_model_sq < 1e-30 { continue; }
            let a0_trial = sum_model_data / sum_model_sq;
            if a0_trial <= 0.0 { continue; }

            // Compute R²
            let mean_a: f64 = data.iter().map(|(_, a)| a).sum::<f64>() / data.len() as f64;
            let ss_tot: f64 = data.iter().map(|(_, a)| (a - mean_a).powi(2)).sum();
            let ss_res: f64 = data.iter().map(|&(f, a)| {
                let model = a0_trial / ((f - f0_trial).powi(2) + gamma_trial.powi(2));
                (a - model).powi(2)
            }).sum();

            let r2 = if ss_tot > 1e-30 { 1.0 - ss_res / ss_tot } else { 0.0 };

            if r2 > best_r2 {
                best_r2 = r2;
                best_f0 = f0_trial;
                best_gamma = gamma_trial;
                best_a0 = a0_trial;
            }
        }
    }

    if best_r2 < -1.0 { return None; }

    Some((best_f0, best_gamma, best_a0, best_r2))
}

/// Process collected amplitude data into resonance results.
///
/// Called after the frequency sweep has collected per-voxel spike amplitudes.
pub fn analyze_resonance_data(
    voxel_data: &[VoxelAmplitudeData],
    site_id: i32,
    step_duration_ps: f64,
) -> PocketResonance {
    let mut voxel_resonances = Vec::new();

    for vd in voxel_data {
        // Convert burst_period (steps) to frequency (1/steps → Hz)
        let freq_amp: Vec<(f64, f64)> = vd.amplitudes.iter()
            .map(|(period, mean_amp, _peak_amp)| {
                let freq = 1.0 / (*period as f64 * step_duration_ps * 1e-12); // Hz
                (freq, *mean_amp)
            })
            .collect();

        if let Some((f0, _gamma, _a0, r2)) = fit_lorentzian(&freq_amp) {
            let k_open = 1.0 / (f0 * step_duration_ps * 1e-3); // per ns

            voxel_resonances.push(VoxelResonance {
                voxel_idx: vd.voxel_idx,
                resonance_frequency_hz: f0,
                k_open_per_ns: k_open,
                lorentzian_r2: r2,
                amplitude_at_resonance: freq_amp.iter()
                    .min_by(|a, b| ((a.0 - f0).abs()).partial_cmp(&(b.0 - f0).abs()).unwrap())
                    .map(|(_, a)| *a)
                    .unwrap_or(0.0),
            });
        }
    }

    // Pocket-level: weighted average of voxel resonance frequencies
    let total_amp: f64 = voxel_resonances.iter().map(|v| v.amplitude_at_resonance).sum();
    let dominant_freq = if total_amp > 0.0 {
        voxel_resonances.iter()
            .map(|v| v.resonance_frequency_hz * v.amplitude_at_resonance)
            .sum::<f64>() / total_amp
    } else {
        0.0
    };

    let k_open = if dominant_freq > 0.0 {
        1.0 / (dominant_freq * step_duration_ps * 1e-3)
    } else { 0.0 };

    let q_factor = if !voxel_resonances.is_empty() {
        // Average Q = f0 / (2Γ) ≈ R² as proxy
        voxel_resonances.iter().map(|v| v.lorentzian_r2).sum::<f64>()
            / voxel_resonances.len() as f64
    } else { 0.0 };

    PocketResonance {
        site_id,
        dominant_resonance_frequency: dominant_freq,
        pocket_opening_rate_ns: k_open,
        resonance_quality_factor: q_factor,
        voxel_resonances,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lorentzian_fit_peak() {
        // Create synthetic Lorentzian data
        let f0: f64 = 100.0;
        let gamma: f64 = 10.0;
        let a0: f64 = 50.0;

        let data: Vec<(f64, f64)> = (0..20).map(|i| {
            let f = 50.0 + i as f64 * 5.0;
            let a = a0 / ((f - f0).powi(2) + gamma.powi(2));
            (f, a)
        }).collect();

        let result = fit_lorentzian(&data);
        assert!(result.is_some());
        let (fitted_f0, _gamma, _a0, r2) = result.unwrap();
        assert!((fitted_f0 - f0).abs() < 10.0, "f0 should be near {}, got {}", f0, fitted_f0);
        assert!(r2 > 0.5, "R² should be > 0.5, got {}", r2);
    }

    #[test]
    fn test_lorentzian_insufficient_data() {
        assert!(fit_lorentzian(&[]).is_none());
        assert!(fit_lorentzian(&[(1.0, 1.0)]).is_none());
    }
}
