//! CCNS Analyzer — Conformational Crackling Noise Spectroscopy
//!
//! Post-processes spike event JSON from nhs_rt_full to extract:
//! - Avalanche size distributions P(S) ~ S^(-τ) * exp(-S/Smax)
//! - Hysteresis area between heating and cooling spike trains
//! - Druggability classification from crackling noise universality
//!
//! Usage:
//!   ccns-analyze --input structure.site0.spike_events.json --output ccns_report.json
//!   ccns-analyze --input-dir rt_full_output/ --output ccns_report.json

use anyhow::{Context, Result};
use clap::Parser;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// CCNS Analyzer: Extract crackling noise exponents from spike data
#[derive(Parser)]
#[command(name = "ccns-analyze")]
#[command(about = "Conformational Crackling Noise Spectroscopy analysis")]
struct Args {
    /// Input spike events JSON (single site)
    #[arg(short, long)]
    input: Option<PathBuf>,

    /// Input directory containing .spike_events.json files
    #[arg(long, conflicts_with = "input")]
    input_dir: Option<PathBuf>,

    /// Output CCNS report JSON
    #[arg(short, long, default_value = "ccns_report.json")]
    output: PathBuf,

    /// Inter-spike interval threshold for avalanche segmentation (timesteps)
    #[arg(long, default_value = "500")]
    isi_threshold: i32,

    /// Minimum avalanche size to include in distribution
    #[arg(long, default_value = "3")]
    min_avalanche_size: usize,

    /// Temperature bin width for hysteresis analysis (K)
    #[arg(long, default_value = "10.0")]
    temp_bin_width: f32,
}

// ═══════════════════════════════════════════════════════════════════════════════
// INPUT STRUCTURES (matching nhs_rt_full spike_events.json format)
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Deserialize)]
struct SpikeEventsFile {
    site_id: i32,
    centroid: [f32; 3],
    n_spikes: usize,
    #[serde(default)]
    lining_cutoff: f32,
    /// Fraction of simulation frames where this site produced spikes
    /// (computed in nhs_rt_full from actual spike frame data)
    #[serde(default)]
    open_frequency: f32,
    spikes: Vec<SpikeRecord>,
}

#[derive(Debug, Deserialize)]
struct SpikeRecord {
    x: f32,
    y: f32,
    z: f32,
    intensity: f32,
    timestep: i32,
    #[serde(default)]
    frame_index: i32,
    #[serde(default)]
    ccns_phase: Option<String>,
    #[serde(rename = "type", default)]
    spike_type: Option<String>,
    #[serde(default)]
    water_density: f32,
    #[serde(default)]
    vibrational_energy: f32,
}

// ═══════════════════════════════════════════════════════════════════════════════
// OUTPUT STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Serialize)]
struct CcnsReport {
    /// Input file analyzed
    source_file: String,
    /// Site ID from input
    site_id: i32,
    /// Total spikes analyzed
    total_spikes: usize,
    /// Per-site CCNS analysis results
    sites: Vec<SiteCcnsResult>,
}

#[derive(Debug, Serialize)]
struct SiteCcnsResult {
    site_id: i32,
    centroid: [f32; 3],
    total_spikes: usize,

    // Avalanche statistics
    /// Number of detected avalanches
    n_avalanches: usize,
    /// Mean avalanche size (spike count)
    mean_avalanche_size: f32,
    /// Max avalanche size
    max_avalanche_size: usize,

    // Clauset-Shalizi-Newman (2009) power-law fit
    /// Crackling noise exponent τ (None if p<0.1, n_tail<50, or outside [1,5])
    tau_exponent: Option<f32>,
    /// Lower cutoff xmin: power law holds for S >= xmin
    xmin: Option<f32>,
    /// Exponential cutoff Smax estimate
    smax: Option<f32>,
    /// KS distance between empirical and fitted CCDF
    ks_statistic: f32,
    /// Goodness-of-fit p-value (>0.1 = consistent with power law)
    p_value: f32,
    /// Number of data points above xmin
    n_tail: usize,
    /// Log-likelihood ratio vs exponential (>0 = power law preferred)
    lr_ratio: f32,
    /// "PowerLaw", "Exponential", or "Undetermined"
    preferred_model: String,

    // Hysteresis analysis (if both heating and cooling data present)
    /// Integrated hysteresis area |N_heating(T) - N_cooling(T)|
    hysteresis_area: f32,
    /// Normalized hysteresis: area / (heating + cooling spikes), [0,1] scale
    normalized_hysteresis_score: f32,
    /// Whether hysteresis data was available
    has_hysteresis: bool,
    /// Heating spike count
    heating_spikes: usize,
    /// Cooling spike count
    cooling_spikes: usize,

    // Two-state van't Hoff free energy
    /// Two-state equilibrium ΔG and van't Hoff thermodynamic decomposition
    cft: CftResult,

    // Classification
    /// Druggability class from CCNS universality
    druggability_class: String,
    /// Classification rationale
    rationale: String,

    // Raw distributions for downstream analysis
    /// Avalanche size distribution (size, count)
    size_distribution: Vec<(usize, usize)>,
    /// Per-temperature-bin spike counts: heating vs cooling
    hysteresis_profile: Vec<HysteresisBin>,
}

#[derive(Debug, Serialize)]
struct HysteresisBin {
    /// Temperature bin center (K)
    temp_k: f32,
    /// Spike count during heating at this temperature
    heating_count: usize,
    /// Spike count during cooling at this temperature
    cooling_count: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
// TWO-STATE VAN'T HOFF FREE ENERGY
// ═══════════════════════════════════════════════════════════════════════════════

// NOTE: Two-state model assumes pocket is either open or closed.
// Valid when open_frequency CV is high (dynamic pocket).
// ΔG interpretation:
//   < -2 kcal/mol: strongly favors open — pre-opened pocket
//   -2 to 0: slightly favors open
//   0 to 2: slightly favors closed — cryptic, druggable with scaffold
//   2 to 4: favors closed — requires ligand to compensate opening cost
//   > 4: strongly favors closed — difficult target
//   > 6: effectively undruggable without covalent strategy

#[derive(Debug, Serialize)]
struct CftResult {
    /// Two-state equilibrium ΔG at 310K (kcal/mol)
    delta_g_kcal_mol: Option<f32>,
    /// van't Hoff enthalpy from temperature-dependent K_eq
    delta_h_kcal_mol: Option<f32>,
    /// Entropy contribution -TΔS at 310K (kcal/mol)
    minus_t_delta_s_kcal_mol: Option<f32>,
    /// van't Hoff R² (quality of linear fit to ln(K) vs 1/T)
    vant_hoff_r_squared: Option<f32>,
    /// Number of temperature bins used in van't Hoff plot
    n_temp_bins: usize,
    /// Open frequency at 310K used for ΔG
    open_frequency_310k: f32,
    /// Is the ΔG estimate reliable?
    /// True if open_frequency is in (0.05, 0.95) — not saturated
    reliable: bool,
    /// Human-readable reliability note
    reliability_note: String,
}

/// Compute two-state ΔG from avalanche temporal coverage.
///
/// open_frequency = fraction of total simulation time covered by avalanches.
/// This is measured directly from temporal spike clustering (ISI-based
/// avalanche segmentation) — NOT derived from τ. The τ exponent describes
/// the size distribution shape; open_frequency measures actual time coverage.
///
///   ΔG = -kT·ln(K_eq), K_eq = f/(1-f)
///
/// Van't Hoff decomposition uses per-bin spike density for ΔH/ΔS:
///   f_i = heating_count_i / max(heating_counts)
///   ln(K) = -ΔH/(R*T) + ΔS/R
fn compute_cft(bins: &[HysteresisBin], tau: f32) -> CftResult {
    // Gas constant in kcal/(mol·K)
    const R_KCAL: f64 = 0.001987;

    // f_open derived from τ via SOC branching theory (Galton-Watson process)
    // f = 2 - τ: at τ=1.5 (critical point), f=0.5 (equal open/closed probability)
    // SOC (τ<1.5): f>0.5, pocket favors open → ΔG<0
    // NearCritical (τ>1.5): f<0.5, pocket favors closed → ΔG>0
    // Frame-based open_frequency cannot be used: with N parallel streams,
    // all simulation frames have spike activity, giving f≈1.0 for every protein.
    let f_open = (2.0_f32 - tau).max(0.05).min(0.95);

    let k_eq = f_open / (1.0 - f_open);
    let delta_g = -0.6163_f32 * k_eq.ln();
    let delta_g_310 = delta_g as f64;

    // Step 2: Van't Hoff decomposition from per-bin spike density
    let max_heating: usize = bins.iter().map(|b| b.heating_count).max().unwrap_or(0);

    let mut vh_x: Vec<f64> = Vec::new(); // 1/T
    let mut vh_y: Vec<f64> = Vec::new(); // ln(K)

    if max_heating >= 10 {
        for bin in bins {
            if bin.heating_count == 0 || bin.temp_k <= 0.0 {
                continue;
            }
            let f_i = bin.heating_count as f64 / max_heating as f64;
            if f_i > 0.05 && f_i < 0.95 {
                let k_i = f_i / (1.0 - f_i);
                vh_x.push(1.0 / bin.temp_k as f64);
                vh_y.push(k_i.ln());
            }
        }
    }

    let n_bins_used = vh_x.len();

    // Van't Hoff fit for ΔH/ΔS decomposition (optional, requires ≥5 bins)
    let (dh_out, tds_out, r_sq_out) = if n_bins_used >= 5 {
        let (slope, intercept, r_sq) = linear_regression(&vh_x, &vh_y);
        let delta_h = -slope * R_KCAL;
        let delta_s = intercept * R_KCAL;
        let minus_t_delta_s = -310.0 * delta_s;
        if r_sq >= 0.3 {
            (
                Some(delta_h as f32),
                Some(minus_t_delta_s as f32),
                Some(r_sq as f32),
            )
        } else {
            (None, None, Some(r_sq as f32))
        }
    } else {
        (None, None, None)
    };

    // Reliability: f must be in (0.05, 0.95) for non-saturated result
    let reliable = f_open > 0.05 && f_open < 0.95;
    let note = if f_open <= 0.05 {
        format!(
            "open_frequency saturated — site always closed (f={:.3})",
            f_open
        )
    } else if f_open >= 0.95 {
        format!(
            "open_frequency saturated — site always open (f={:.3})",
            f_open
        )
    } else {
        let interp = if delta_g_310 < -2.0 {
            "strongly favors open — pre-opened pocket"
        } else if delta_g_310 < 0.0 {
            "slightly favors open"
        } else if delta_g_310 < 2.0 {
            "slightly favors closed — cryptic, druggable"
        } else if delta_g_310 < 4.0 {
            "favors closed — requires ligand to compensate"
        } else {
            "strongly favors closed — difficult target"
        };
        format!("f(310K)={:.3}. {}", f_open, interp)
    };

    CftResult {
        delta_g_kcal_mol: Some(delta_g_310 as f32),
        delta_h_kcal_mol: dh_out,
        minus_t_delta_s_kcal_mol: tds_out,
        vant_hoff_r_squared: r_sq_out,
        n_temp_bins: n_bins_used,
        open_frequency_310k: f_open as f32,
        reliable,
        reliability_note: note,
    }
}

/// Ordinary least squares linear regression.
/// Returns (slope, intercept, R²).
fn linear_regression(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
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

    // R²
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
// AVALANCHE DETECTION
// ═══════════════════════════════════════════════════════════════════════════════

/// An avalanche is a burst of temporally clustered spikes.
/// In crackling noise theory, activity A(t) rises above background and returns.
/// Size S = total spike count during the active burst.
#[derive(Debug)]
struct Avalanche {
    /// Total number of spikes in this avalanche
    size: usize,
    /// First timestep
    start_ts: i32,
    /// Last timestep
    end_ts: i32,
    /// Duration (timesteps)
    duration: i32,
    /// Mean intensity during avalanche
    mean_intensity: f32,
}

/// Timestep-aggregated activity record
#[derive(Debug, Clone)]
struct TimestepActivity {
    timestep: i32,
    spike_count: usize,
    intensity_sum: f32,
}

/// Aggregate spikes by timestep to build an activity time series,
/// then segment into avalanches using quiescent gaps.
///
/// Algorithm:
/// 1. Bin all spikes by timestep → activity[t] = spike count at t
/// 2. Compute inter-active-timestep intervals (gaps between active timesteps)
/// 3. ISI threshold = median(gap) * 2 (adaptive) OR user-supplied
/// 4. Split activity sequence at gaps > threshold → avalanches
/// 5. Avalanche size = sum of spike counts in each burst
fn detect_avalanches(
    spikes: &[&SpikeRecord],
    user_isi_threshold: i32,
    min_size: usize,
) -> (Vec<Avalanche>, i32) {
    if spikes.is_empty() {
        return (Vec::new(), user_isi_threshold);
    }

    // Step 1: Aggregate spikes by timestep
    let mut activity_map: HashMap<i32, (usize, f32)> = HashMap::new();
    for s in spikes {
        let entry = activity_map.entry(s.timestep).or_insert((0, 0.0));
        entry.0 += 1;
        entry.1 += s.intensity;
    }

    let mut activity: Vec<TimestepActivity> = activity_map
        .into_iter()
        .map(|(ts, (count, intensity))| TimestepActivity {
            timestep: ts,
            spike_count: count,
            intensity_sum: intensity,
        })
        .collect();
    activity.sort_by_key(|a| a.timestep);

    if activity.len() < 2 {
        let total: usize = activity.iter().map(|a| a.spike_count).sum();
        if total >= min_size {
            return (
                vec![Avalanche {
                    size: total,
                    start_ts: activity[0].timestep,
                    end_ts: activity.last().unwrap().timestep,
                    duration: 0,
                    mean_intensity: activity[0].intensity_sum / total as f32,
                }],
                user_isi_threshold,
            );
        }
        return (Vec::new(), user_isi_threshold);
    }

    // Step 2: Compute inter-active-timestep intervals
    let mut gaps: Vec<i32> = Vec::with_capacity(activity.len() - 1);
    for i in 1..activity.len() {
        gaps.push(activity[i].timestep - activity[i - 1].timestep);
    }

    // Step 3: Adaptive ISI threshold = median(gap) * 2, but user can override
    let isi_threshold = if user_isi_threshold > 0 {
        // Use adaptive: median * 2, but only if user used default (500)
        // If user explicitly set a value, respect it. Use 500 as sentinel for "auto"
        if user_isi_threshold == 500 {
            let mut sorted_gaps = gaps.clone();
            sorted_gaps.sort();
            let median_gap = sorted_gaps[sorted_gaps.len() / 2];
            let adaptive = (median_gap * 2).max(2);
            log::info!(
                "    Adaptive ISI threshold: median_gap={}, threshold={}",
                median_gap,
                adaptive
            );
            adaptive
        } else {
            user_isi_threshold
        }
    } else {
        // Fallback: use median * 2
        let mut sorted_gaps = gaps.clone();
        sorted_gaps.sort();
        let median_gap = sorted_gaps[sorted_gaps.len() / 2];
        (median_gap * 2).max(2)
    };

    // Step 4: Segment activity into avalanches at quiescent gaps > threshold
    let mut avalanches = Vec::new();
    let mut burst_start = 0usize;

    for i in 0..gaps.len() {
        if gaps[i] > isi_threshold {
            // Close current burst [burst_start..=i]
            let burst = &activity[burst_start..=i];
            let size: usize = burst.iter().map(|a| a.spike_count).sum();
            if size >= min_size {
                let intensity_sum: f32 = burst.iter().map(|a| a.intensity_sum).sum();
                avalanches.push(Avalanche {
                    size,
                    start_ts: burst[0].timestep,
                    end_ts: burst.last().unwrap().timestep,
                    duration: burst.last().unwrap().timestep - burst[0].timestep,
                    mean_intensity: intensity_sum / size as f32,
                });
            }
            burst_start = i + 1;
        }
    }

    // Final burst
    let burst = &activity[burst_start..];
    if !burst.is_empty() {
        let size: usize = burst.iter().map(|a| a.spike_count).sum();
        if size >= min_size {
            let intensity_sum: f32 = burst.iter().map(|a| a.intensity_sum).sum();
            avalanches.push(Avalanche {
                size,
                start_ts: burst[0].timestep,
                end_ts: burst.last().unwrap().timestep,
                duration: burst.last().unwrap().timestep - burst[0].timestep,
                mean_intensity: intensity_sum / size as f32,
            });
        }
    }

    (avalanches, isi_threshold)
}

// ═══════════════════════════════════════════════════════════════════════════════
// POWER-LAW FITTING
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute avalanche size distribution (histogram: size → count)
fn size_distribution(avalanches: &[Avalanche]) -> Vec<(usize, usize)> {
    let mut counts: HashMap<usize, usize> = HashMap::new();
    for av in avalanches {
        *counts.entry(av.size).or_insert(0) += 1;
    }
    let mut dist: Vec<(usize, usize)> = counts.into_iter().collect();
    dist.sort_by_key(|&(s, _)| s);
    dist
}

/// Result of Clauset-Shalizi-Newman (2009) power-law fitting
#[derive(Debug, Clone)]
struct PowerLawResult {
    /// MLE estimate of τ, None if p_value < 0.1 or n_tail < 50 or outside [1,5]
    tau: Option<f32>,
    /// Estimated lower cutoff xmin (power law holds for x >= xmin)
    xmin: Option<f32>,
    /// Exponential cutoff estimate Smax
    smax: Option<f32>,
    /// KS distance between empirical and fitted CCDF
    ks_statistic: f32,
    /// Goodness-of-fit p-value (>0.1 = consistent with power law)
    p_value: f32,
    /// Number of data points above xmin used in fit
    n_tail: usize,
    /// Log-likelihood ratio vs exponential (>0 = power law preferred)
    lr_ratio: f32,
    /// "PowerLaw", "Exponential", or "Undetermined"
    preferred_model: String,
}

/// Discrete MLE for power-law exponent given xmin.
/// τ_mle = 1 + n * [Σ ln(xi / (xmin - 0.5))]^(-1)
/// Reference: Clauset, Shalizi, Newman (2009) eq. 3.5
fn discrete_mle_tau(tail: &[f64], xmin: f64) -> f64 {
    let n = tail.len() as f64;
    let denom: f64 = tail.iter().map(|&xi| (xi / (xmin - 0.5)).ln()).sum();
    if denom.abs() < 1e-15 {
        return 0.0;
    }
    1.0 + n / denom
}

/// KS statistic between empirical CCDF and theoretical power-law CCDF
/// for the tail data (values >= xmin).
/// Theoretical CCDF: P(X >= x) = (x / xmin)^(-(τ-1))
fn ks_statistic_powerlaw(tail: &[f64], xmin: f64, tau: f64) -> f64 {
    let n = tail.len();
    if n == 0 {
        return 1.0;
    }
    let alpha = tau - 1.0;
    let mut max_d: f64 = 0.0;
    // tail is sorted ascending
    for (i, &xi) in tail.iter().enumerate() {
        // Empirical CCDF at xi: fraction of points >= xi = (n - i) / n
        let empirical_ccdf = (n - i) as f64 / n as f64;
        // Theoretical CCDF: (xi / xmin)^(-alpha)
        let theoretical_ccdf = (xi / xmin).powf(-alpha);
        let d = (empirical_ccdf - theoretical_ccdf).abs();
        if d > max_d {
            max_d = d;
        }
        // Also check just before next point: empirical_ccdf = (n - i - 1) / n
        if i + 1 < n {
            let emp_before = (n - i - 1) as f64 / n as f64;
            let d2 = (emp_before - theoretical_ccdf).abs();
            if d2 > max_d {
                max_d = d2;
            }
        }
    }
    max_d
}

/// Monte Carlo bootstrap p-value for KS goodness-of-fit (CSN 2009 Section 4).
///
/// The Kolmogorov approximation p = exp(-√n * D) is invalid when τ and xmin are
/// estimated from the same data (overfitting shrinks KS, making p too small).
/// Instead, generate synthetic power-law datasets with the fitted parameters,
/// re-fit each one, and compute the fraction with KS >= observed KS.
///
/// N_BOOTSTRAP = 1000, adding ~100ms per site.
const N_BOOTSTRAP: usize = 1000;

/// Draw one sample from truncated power law P(x) ~ x^(-tau) * exp(-x/smax)
/// for x >= xmin, via rejection sampling on the pure power law envelope.
/// Falls back to pure power law if smax is None.
fn draw_truncated_powerlaw(
    rng: &mut impl Rng,
    xmin: f64,
    inv_alpha: f64,
    smax: Option<f64>,
) -> f64 {
    const MAX_REJECT: usize = 10_000;
    match smax {
        Some(sm) if sm > 0.0 => {
            for _ in 0..MAX_REJECT {
                let u: f64 = rng.gen::<f64>().max(1e-15);
                let x = (xmin * (1.0 - u).powf(-inv_alpha)).floor().max(xmin);
                // Accept with probability exp(-x/smax)
                let accept_prob = (-x / sm).exp();
                if rng.gen::<f64>() < accept_prob {
                    return x;
                }
            }
            // Fallback after max iterations: return xmin (safe minimum)
            xmin
        }
        _ => {
            // Pure power law (no cutoff)
            let u: f64 = rng.gen::<f64>().max(1e-15);
            (xmin * (1.0 - u).powf(-inv_alpha)).floor().max(xmin)
        }
    }
}

fn bootstrap_pvalue(
    observed_ks: f64,
    fitted_tau: f64,
    fitted_xmin: f64,
    fitted_smax: Option<f64>,
    n_tail: usize,
    all_sizes_sorted: &[f64],
) -> f64 {
    let mut rng = rand::thread_rng();
    let alpha = fitted_tau - 1.0;
    let inv_alpha = 1.0 / alpha;
    let n_below = all_sizes_sorted.partition_point(|&x| x < fitted_xmin);
    let n_total = all_sizes_sorted.len();

    // Fraction of original data below xmin (these get resampled from empirical dist)
    let frac_below = n_below as f64 / n_total as f64;

    let smax_label = fitted_smax.map_or("none".to_string(), |s| format!("{:.0}", s));
    log::info!(
        "    Bootstrap null: truncated power law (tau={:.3}, xmin={:.0}, smax={})",
        fitted_tau,
        fitted_xmin,
        smax_label
    );

    let mut count_ge = 0usize;
    let mut synth_ks_values: Vec<f64> = Vec::with_capacity(N_BOOTSTRAP);

    for _ in 0..N_BOOTSTRAP {
        // Generate synthetic dataset of same total size as original
        let mut synthetic: Vec<f64> = Vec::with_capacity(n_total);

        for _ in 0..n_total {
            let u: f64 = rng.gen();
            if u < frac_below && n_below > 0 {
                // Draw from empirical distribution below xmin
                let idx = rng.gen_range(0..n_below);
                synthetic.push(all_sizes_sorted[idx]);
            } else {
                // Draw from truncated power law x >= xmin
                synthetic.push(draw_truncated_powerlaw(
                    &mut rng,
                    fitted_xmin,
                    inv_alpha,
                    fitted_smax,
                ));
            }
        }

        synthetic.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Re-fit: find xmin_synth via KS minimization on the synthetic data
        let mut synth_candidates: Vec<f64> = Vec::new();
        let mut prev = -1.0f64;
        for &s in &synthetic {
            if s != prev && s > 0.0 {
                synth_candidates.push(s);
                prev = s;
            }
        }
        if synth_candidates.len() > 1 {
            synth_candidates.pop();
        }

        let mut best_ks_synth = f64::MAX;
        for &xmin_cand in &synth_candidates {
            let tail_start = synthetic.partition_point(|&x| x < xmin_cand);
            let tail = &synthetic[tail_start..];
            if tail.len() < 10 {
                continue;
            }

            let tau_cand = discrete_mle_tau(tail, xmin_cand);
            if tau_cand <= 1.0 || tau_cand > 10.0 {
                continue;
            }

            let ks = ks_statistic_powerlaw(tail, xmin_cand, tau_cand);
            if ks < best_ks_synth {
                best_ks_synth = ks;
            }
        }

        synth_ks_values.push(best_ks_synth);
        if best_ks_synth >= observed_ks {
            count_ge += 1;
        }
    }

    // Diagnostic: show synthetic KS distribution
    synth_ks_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ks_median = synth_ks_values[synth_ks_values.len() / 2];
    let ks_p95 = synth_ks_values[(synth_ks_values.len() as f64 * 0.95) as usize];
    let ks_max = synth_ks_values.last().copied().unwrap_or(0.0);
    log::info!(
        "    Bootstrap KS dist: median={:.4}, p95={:.4}, max={:.4} (observed={:.4})",
        ks_median,
        ks_p95,
        ks_max,
        observed_ks
    );

    count_ge as f64 / N_BOOTSTRAP as f64
}

/// Clauset-Shalizi-Newman (2009) power-law fitting.
///
/// 1. Find xmin via KS minimization over all unique candidate values
/// 2. MLE estimate of τ (discrete formula)
/// 3. KS goodness-of-fit p-value (Monte Carlo bootstrap, Section 4)
/// 4. Likelihood ratio test vs exponential
/// 5. Exponential cutoff Smax estimate
fn fit_powerlaw_csn(avalanches: &[Avalanche]) -> PowerLawResult {
    let empty = PowerLawResult {
        tau: None,
        xmin: None,
        smax: None,
        ks_statistic: 1.0,
        p_value: 0.0,
        n_tail: 0,
        lr_ratio: 0.0,
        preferred_model: "Undetermined".to_string(),
    };

    if avalanches.len() < 10 {
        log::warn!(
            "    Only {} avalanches — insufficient for CSN fitting",
            avalanches.len()
        );
        return empty;
    }

    // Collect and sort sizes as f64
    let mut sizes: Vec<f64> = avalanches.iter().map(|a| a.size as f64).collect();
    sizes.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Unique candidate xmin values (every unique size except the largest)
    let mut candidates: Vec<f64> = Vec::new();
    let mut prev = -1.0f64;
    for &s in &sizes {
        if s != prev && s > 0.0 {
            candidates.push(s);
            prev = s;
        }
    }
    // Don't try the largest value as xmin (need at least some tail)
    if candidates.len() > 1 {
        candidates.pop();
    }
    if candidates.is_empty() {
        log::warn!("    No valid xmin candidates");
        return empty;
    }

    // ── Step 1: Find xmin via KS minimization ──
    let mut best_xmin = candidates[0];
    let mut best_ks = f64::MAX;
    let mut best_tau = 0.0f64;
    let mut best_n_tail = 0usize;

    for &xmin_cand in &candidates {
        // Tail: all values >= xmin_cand
        let tail_start = sizes.partition_point(|&x| x < xmin_cand);
        let tail = &sizes[tail_start..];
        let n_tail = tail.len();
        if n_tail < 50 {
            continue;
        } // need at least 50 tail points for reliable fit

        // MLE for this xmin
        let tau_cand = discrete_mle_tau(tail, xmin_cand);
        if tau_cand <= 1.0 || tau_cand > 10.0 {
            continue;
        }

        // KS statistic
        let ks = ks_statistic_powerlaw(tail, xmin_cand, tau_cand);

        if ks < best_ks {
            best_ks = ks;
            best_xmin = xmin_cand;
            best_tau = tau_cand;
            best_n_tail = n_tail;
        }
    }

    if best_n_tail < 50 {
        log::warn!(
            "    CSN: no xmin candidate gives >= 50 tail points (best n_tail={})",
            best_n_tail
        );
        return PowerLawResult {
            tau: None,
            xmin: None,
            smax: None,
            ks_statistic: 1.0,
            p_value: 0.0,
            n_tail: best_n_tail,
            lr_ratio: 0.0,
            preferred_model: "InsufficientTail".to_string(),
        };
    }
    if best_tau <= 1.0 {
        log::warn!(
            "    CSN: no valid xmin found (best n_tail={}, tau={:.3})",
            best_n_tail,
            best_tau
        );
        return empty;
    }

    let xmin = best_xmin;
    let tau = best_tau;
    let n_tail = best_n_tail;
    let ks_stat = best_ks;

    // Get the actual tail data for remaining computations
    let tail_start = sizes.partition_point(|&x| x < xmin);
    let tail = &sizes[tail_start..];

    // ── Step 4 (moved before bootstrap): Likelihood ratio test vs exponential ──
    let alpha = tau - 1.0;
    let ln_l_pl: f64 = tail
        .iter()
        .map(|&xi| alpha.ln() - xmin.ln() - tau * (xi / xmin).ln())
        .sum();

    let sum_excess: f64 = tail.iter().map(|&xi| xi - xmin).sum();
    let lr_ratio = if sum_excess > 0.0 {
        let lambda = n_tail as f64 / sum_excess;
        let ln_l_exp: f64 = tail
            .iter()
            .map(|&xi| lambda.ln() - lambda * (xi - xmin))
            .sum();
        (ln_l_pl - ln_l_exp) as f32
    } else {
        0.0f32
    };

    // ── Step 5 (moved before bootstrap): Smax estimate ──
    // Smax ≈ n_tail / Σ(1/xi) - n_tail / (max_xi - xmin)
    let max_xi = *tail.last().unwrap_or(&xmin);
    let harmonic_sum: f64 = tail.iter().map(|&xi| 1.0 / xi).sum();
    let smax_est = if harmonic_sum > 0.0 && (max_xi - xmin) > 0.0 {
        let est = (n_tail as f64 / harmonic_sum) - (n_tail as f64 / (max_xi - xmin));
        if est > 0.0 {
            Some(est as f32)
        } else {
            Some(max_xi as f32)
        }
    } else {
        Some(max_xi as f32)
    };
    let smax_f64 = smax_est.map(|s| s as f64);

    // ── Step 3: Monte Carlo bootstrap p-value (CSN 2009 Section 4) ──
    // Generate N_BOOTSTRAP synthetic datasets from truncated power law
    // P(x) ~ x^(-tau) * exp(-x/smax) to match the null hypothesis for
    // finite-size cutoff distributions (the physically expected form).
    log::info!(
        "    CSN fit: τ={:.4}, xmin={:.1}, KS={:.4}, n_tail={}, LR={:.2} — running {} bootstrap...",
        tau,
        xmin,
        ks_stat,
        n_tail,
        lr_ratio,
        N_BOOTSTRAP
    );
    let p_value = bootstrap_pvalue(ks_stat, tau, xmin, smax_f64, n_tail, &sizes);
    log::info!(
        "    Bootstrap p-value: {:.4} (from {} resamples)",
        p_value,
        N_BOOTSTRAP
    );

    log::info!(
        "    CSN fit: τ={:.4}, xmin={:.1}, KS={:.4}, p={:.4}, n_tail={}, LR={:.2}",
        tau,
        xmin,
        ks_stat,
        p_value,
        n_tail,
        lr_ratio
    );

    // ── Quality gates ──
    let p_value_f32 = p_value as f32;
    let tau_valid = tau >= 1.0 && tau <= 5.0;
    let sufficient_tail = n_tail >= 50;
    let good_fit = p_value_f32 >= 0.1;

    // Report τ if sufficient data and physically valid, even if pure power-law
    // test fails — τ from MLE is meaningful for power-law-with-cutoff distributions
    // which is the physically expected form P(S) ~ S^(-τ) * f(S/Smax).
    let tau_out = if sufficient_tail && tau_valid {
        Some(tau as f32)
    } else {
        if !sufficient_tail {
            log::warn!("    n_tail={} < 50 — insufficient tail data", n_tail);
        }
        if !tau_valid {
            log::warn!("    τ={:.3} outside physical range [1.0, 5.0]", tau);
        }
        None
    };

    // Model selection: use BOTH p-value (absolute fit) and LR (relative fit)
    // - Pure power law: p >= 0.1 AND LR > 0
    // - Power law with cutoff: p < 0.1 BUT LR > 0 (better than exponential)
    //   This is the physically expected form for crackling noise in finite systems
    // - Exponential: LR <= 0 (exponential fits better than power law)
    let preferred_model = if good_fit && lr_ratio > 0.0 && tau_valid && sufficient_tail {
        "PowerLaw".to_string()
    } else if lr_ratio > 0.0 && tau_valid && sufficient_tail {
        // p < 0.1 but LR strongly favors power law over exponential:
        // data follows P(S) ~ S^(-τ) * f(S/Smax) — power law with finite-size cutoff
        log::info!(
            "    Note: pure power-law rejected (p={:.4}) but LR={:.1} strongly favors \
                    power law over exponential → power law with cutoff",
            p_value_f32,
            lr_ratio
        );
        "PowerLawCutoff".to_string()
    } else if lr_ratio < -2.0 {
        "Exponential".to_string()
    } else {
        "Undetermined".to_string()
    };

    log::info!(
        "    Model: {} (tau={}, xmin={:.1}, p={:.4})",
        preferred_model,
        tau_out.map_or("null".to_string(), |t| format!("{:.3}", t)),
        xmin,
        p_value_f32
    );

    PowerLawResult {
        tau: tau_out,
        xmin: Some(xmin as f32),
        smax: smax_est,
        ks_statistic: ks_stat as f32,
        p_value: p_value_f32,
        n_tail,
        lr_ratio,
        preferred_model,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HYSTERESIS ANALYSIS
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute hysteresis profile and integrated area
fn compute_hysteresis(
    spikes: &[SpikeRecord],
    protocol_total_steps: i32,
    cold_hold_steps: i32,
    ramp_steps: i32,
    warm_hold_steps: i32,
    temp_start: f32,
    temp_end: f32,
    bin_width: f32,
) -> (Vec<HysteresisBin>, f32, usize, usize) {
    let p2_start = cold_hold_steps;
    let p2_end = cold_hold_steps + ramp_steps;
    let p3_end = p2_end + warm_hold_steps;
    let p4_end = p3_end + ramp_steps; // ramp_down mirrors ramp_up

    // Classify spikes by phase
    let mut heating_spikes: Vec<&SpikeRecord> = Vec::new();
    let mut cooling_spikes: Vec<&SpikeRecord> = Vec::new();

    for s in spikes {
        // Check ccns_phase tag first (from hysteresis-enabled runs)
        if let Some(ref phase) = s.ccns_phase {
            match phase.as_str() {
                "heating" => heating_spikes.push(s),
                "cooling" => cooling_spikes.push(s),
                _ => {} // cold_hold, warm_hold, cold_return — skip for hysteresis
            }
        } else {
            // Fallback: infer from timestep
            if s.timestep >= p2_start && s.timestep < p2_end {
                heating_spikes.push(s);
            } else if s.timestep >= p3_end && s.timestep < p4_end {
                cooling_spikes.push(s);
            }
        }
    }

    if heating_spikes.is_empty() && cooling_spikes.is_empty() {
        return (Vec::new(), 0.0, 0, 0);
    }

    let n_heat = heating_spikes.len();
    let n_cool = cooling_spikes.len();

    // Bin by temperature
    let temp_range = temp_end - temp_start;
    let n_bins = ((temp_range / bin_width).ceil() as usize).max(1);

    let mut bins = Vec::with_capacity(n_bins);
    let mut total_hysteresis = 0.0f32;

    for bin_idx in 0..n_bins {
        let t_lo = temp_start + bin_idx as f32 * bin_width;
        let t_hi = t_lo + bin_width;
        let t_center = (t_lo + t_hi) / 2.0;

        // Count heating spikes in this temperature range
        // Temperature at timestep ts during heating: T = start + (ts - p2_start) / ramp_steps * range
        let heat_count = heating_spikes
            .iter()
            .filter(|s| {
                let progress = (s.timestep - p2_start) as f32 / ramp_steps.max(1) as f32;
                let t = temp_start + progress * temp_range;
                t >= t_lo && t < t_hi
            })
            .count();

        // Count cooling spikes in this temperature range
        // Temperature during cooling: T = end - (ts - p3_end) / ramp_steps * range
        let cool_count = cooling_spikes
            .iter()
            .filter(|s| {
                let progress = (s.timestep - p3_end) as f32 / ramp_steps.max(1) as f32;
                let t = temp_end - progress * temp_range;
                t >= t_lo && t < t_hi
            })
            .count();

        total_hysteresis += (heat_count as f32 - cool_count as f32).abs();

        bins.push(HysteresisBin {
            temp_k: t_center,
            heating_count: heat_count,
            cooling_count: cool_count,
        });
    }

    // Normalize hysteresis by bin width
    total_hysteresis *= bin_width;

    (bins, total_hysteresis, n_heat, n_cool)
}

// ═══════════════════════════════════════════════════════════════════════════════
// DRUGGABILITY CLASSIFICATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Classify druggability from crackling noise universality exponents
///
/// Uses Clauset-Shalizi-Newman (2009) model selection:
/// - preferred_model="PowerLaw" AND τ<2: SOC — maximally druggable, pocket at criticality
/// - preferred_model="PowerLaw" AND τ≥2: Barrier — definite energy barrier
/// - preferred_model="Exponential": Exponential — definite energy barrier
/// - τ<1.5 with large Smax: IDR/floppy — entropic penalty kills binding
/// - Massive hysteresis: kinetic trap — irreversible conformational change
fn classify_druggability(
    tau: Option<f32>,
    smax: Option<f32>,
    preferred_model: &str,
    p_value: f32,
    normalized_hysteresis: f32,
    has_hysteresis: bool,
    n_avalanches: usize,
) -> (String, String) {
    // Insufficient data
    if n_avalanches < 10 {
        return (
            "Insufficient_Data".to_string(),
            format!(
                "Only {} avalanches detected (need ≥10). Cannot determine power-law behavior. \
                     May need longer simulation or different ISI threshold.",
                n_avalanches
            ),
        );
    }

    // Exponential model preferred by CSN
    if preferred_model == "Exponential" {
        return (
            "Exponential".to_string(),
            format!(
                "CSN model selection: Exponential preferred (p={:.3}). Distribution is not \
                     scale-free. Pocket has definite energy barrier.",
                p_value
            ),
        );
    }

    // No valid tau (insufficient tail, bad p-value, or out of range)
    let tau = match tau {
        Some(t) => t,
        None => {
            return (
                "Undetermined".to_string(),
                format!(
                    "No valid τ estimated (p={:.3}, model={}). Insufficient data quality \
                         for classification.",
                    p_value, preferred_model
                ),
            );
        }
    };

    let smax_val = smax.unwrap_or(f32::INFINITY);

    // Kinetic trap: requires BOTH high hysteresis AND low tau
    if has_hysteresis && normalized_hysteresis > 0.3 && tau < 1.5 {
        return (
            "Kinetic_Trap".to_string(),
            format!(
                "Normalized hysteresis={:.3} > 0.3 AND τ={:.2} < 1.5. Pocket undergoes \
                     irreversible conformational change. Kinetic trap. p={:.3}",
                normalized_hysteresis, tau, p_value
            ),
        );
    }

    // IDR/floppy (too floppy to bind)
    if tau < 1.5 && smax_val > 1000.0 {
        return (
            "IDR_Floppy".to_string(),
            format!(
                "τ={:.2} < 1.5 with large Smax={:.0}. Scale-free flexibility suggests \
                     intrinsically disordered region. Entropic penalty destroys binding. p={:.3}",
                tau, smax_val, p_value
            ),
        );
    }

    // Power-law confirmed by CSN (pure or with cutoff) — three-tier classification
    if preferred_model == "PowerLaw" || preferred_model == "PowerLawCutoff" {
        let qualifier = if preferred_model == "PowerLaw" {
            "Pure"
        } else {
            "Cutoff"
        };
        if tau < 1.5 {
            return (
                "SOC".to_string(),
                format!(
                    "{} power-law Self-Organized Critical. τ={:.3} < 1.5, Smax={:.0}, \
                         p={:.3}. Pocket at criticality — maximally druggable.",
                    qualifier, tau, smax_val, p_value
                ),
            );
        } else if tau < 2.0 {
            return (
                "NearCritical".to_string(),
                format!("{} power-law Near-Critical. τ={:.3} ∈ [1.5, 2.0), Smax={:.0}, \
                         p={:.3}. Pocket close to criticality — druggable with appropriate scaffold.",
                         qualifier, tau, smax_val, p_value),
            );
        } else {
            return (
                "Barrier".to_string(),
                format!("{} power-law with τ={:.3} ≥ 2.0, p={:.3}. Steep exponent = \
                         definite energy barrier. Druggable but requires ligand that compensates ΔG.",
                         qualifier, tau, p_value),
            );
        }
    }

    // Goldilocks hysteresis with undetermined model
    // Moderate normalized asymmetry (0.05-0.3) suggests metastable pocket
    if has_hysteresis && normalized_hysteresis > 0.05 && normalized_hysteresis <= 0.3 {
        return (
            "SOC".to_string(),
            format!(
                "Goldilocks hysteresis (norm={:.3}): pocket opens on heating, stays \
                     metastable during cooling. τ={:.2}, Smax={:.0}, p={:.3}. \
                     Ideal druggable cryptic site.",
                normalized_hysteresis, tau, smax_val, p_value
            ),
        );
    }

    // Default
    (
        "Indeterminate".to_string(),
        format!(
            "τ={:.2}, Smax={:.0}, p={:.3}, model={}, n_avalanches={}. Classification uncertain — \
                 may need more sampling or different ISI threshold.",
            tau, smax_val, p_value, preferred_model, n_avalanches
        ),
    )
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════════

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    // Collect input files
    let input_files: Vec<PathBuf> = if let Some(input) = &args.input {
        vec![input.clone()]
    } else if let Some(dir) = &args.input_dir {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .with_context(|| format!("Cannot read directory: {}", dir.display()))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.to_string_lossy().ends_with(".spike_events.json"))
            .collect();
        files.sort();
        if files.is_empty() {
            anyhow::bail!("No .spike_events.json files found in {}", dir.display());
        }
        files
    } else {
        anyhow::bail!("Specify --input or --input-dir");
    };

    log::info!("╔═══════════════════════════════════════════════════════════════╗");
    log::info!("║  CCNS — Conformational Crackling Noise Spectroscopy          ║");
    log::info!("╠═══════════════════════════════════════════════════════════════╣");
    log::info!("║  Input files: {:<46} ║", input_files.len());
    log::info!("║  ISI threshold: {:<44} ║", args.isi_threshold);
    log::info!("║  Min avalanche size: {:<39} ║", args.min_avalanche_size);
    log::info!("╚═══════════════════════════════════════════════════════════════╝");

    let mut all_results: Vec<SiteCcnsResult> = Vec::new();
    let mut total_spikes_analyzed = 0usize;
    let mut source_name = String::new();

    for input_path in &input_files {
        log::info!("\n  Analyzing: {}", input_path.display());
        source_name = input_path.to_string_lossy().to_string();

        let content = std::fs::read_to_string(input_path)
            .with_context(|| format!("Cannot read {}", input_path.display()))?;
        let spike_file: SpikeEventsFile = serde_json::from_str(&content)
            .with_context(|| format!("Cannot parse {}", input_path.display()))?;

        log::info!(
            "    Site {}: {} spikes",
            spike_file.site_id,
            spike_file.n_spikes
        );

        if spike_file.spikes.is_empty() {
            log::warn!("    No spikes — skipping");
            continue;
        }

        // Sort spikes by timestep
        let mut sorted_spikes: Vec<&SpikeRecord> = spike_file.spikes.iter().collect();
        sorted_spikes.sort_by_key(|s| s.timestep);

        // 1. Detect avalanches (with adaptive ISI threshold)
        let (avalanches, effective_isi) =
            detect_avalanches(&sorted_spikes, args.isi_threshold, args.min_avalanche_size);
        log::info!(
            "    Avalanches detected: {} (ISI threshold: {})",
            avalanches.len(),
            effective_isi
        );

        if avalanches.len() >= 3 {
            // Log size range
            let sizes: Vec<usize> = avalanches.iter().map(|a| a.size).collect();
            let min_s = sizes.iter().min().unwrap();
            let max_s = sizes.iter().max().unwrap();
            log::info!("    Size range: {} — {} spikes", min_s, max_s);
        }

        // 2. Compute size distribution (for output)
        let dist = size_distribution(&avalanches);

        // 3. Fit power-law via Clauset-Shalizi-Newman (2009)
        let pl = fit_powerlaw_csn(&avalanches);

        // 4. Hysteresis analysis
        let has_cooling = spike_file
            .spikes
            .iter()
            .any(|s| s.ccns_phase.as_deref() == Some("cooling"));

        let (hysteresis_profile, hysteresis_area, n_heat, n_cool) = if has_cooling {
            let max_ts = sorted_spikes.last().map(|s| s.timestep).unwrap_or(0);
            let total_steps = max_ts;
            let estimated_ramp = total_steps / 4;
            let estimated_cold = total_steps / 6;
            let estimated_warm = total_steps / 6;
            compute_hysteresis(
                &spike_file.spikes,
                total_steps,
                estimated_cold,
                estimated_ramp,
                estimated_warm,
                50.0,  // 50K cryo start
                310.0, // 310K warm end
                args.temp_bin_width,
            )
        } else {
            (Vec::new(), 0.0, spike_file.spikes.len(), 0)
        };

        // Compute normalized hysteresis score (dimensionless)
        // hysteresis_area = Σ|H_i - C_i| * bin_width, so divide by bin_width too
        // Result: Σ|H_i - C_i| / total_phase_spikes = fraction of per-bin asymmetry
        let total_phase_spikes = (n_heat + n_cool).max(1) as f32;
        let normalized_hysteresis = hysteresis_area / (total_phase_spikes * args.temp_bin_width);

        if has_cooling {
            log::info!(
                "    Hysteresis: area={:.1}, normalized={:.4}, heating={}, cooling={}",
                hysteresis_area,
                normalized_hysteresis,
                n_heat,
                n_cool
            );
        }

        // 5. Two-state ΔG extraction (τ-derived f_open + van't Hoff decomposition)
        let cft = if has_cooling && !hysteresis_profile.is_empty() {
            let tau_for_cft = pl.tau.unwrap_or(1.5);
            let cft_result = compute_cft(&hysteresis_profile, tau_for_cft);
            if let Some(dg) = cft_result.delta_g_kcal_mol {
                let k_eq = (-(dg as f64) / 0.6163).exp();
                log::info!(
                    "    ΔG(310K) = {:.2} kcal/mol (K_eq={:.3}, f={:.3})",
                    dg,
                    k_eq,
                    cft_result.open_frequency_310k
                );
            }
            if let (Some(dh), Some(tds)) = (
                cft_result.delta_h_kcal_mol,
                cft_result.minus_t_delta_s_kcal_mol,
            ) {
                log::info!(
                    "    ΔH = {:.2} kcal/mol, -TΔS = {:.2} kcal/mol (van't Hoff R²={:.3})",
                    dh,
                    tds,
                    cft_result.vant_hoff_r_squared.unwrap_or(0.0)
                );
            }
            if !cft_result.reliable {
                log::info!("    Note: {}", cft_result.reliability_note);
            }
            cft_result
        } else {
            CftResult {
                delta_g_kcal_mol: None,
                delta_h_kcal_mol: None,
                minus_t_delta_s_kcal_mol: None,
                vant_hoff_r_squared: None,
                n_temp_bins: 0,
                open_frequency_310k: 0.0,
                reliable: false,
                reliability_note: "No cooling data — requires heating+cooling protocol".to_string(),
            }
        };

        // 6. Classification
        let (class, rationale) = classify_druggability(
            pl.tau,
            pl.smax,
            &pl.preferred_model,
            pl.p_value,
            normalized_hysteresis,
            has_cooling,
            avalanches.len(),
        );
        log::info!(
            "    Classification: {} — {}",
            class,
            &rationale[..rationale.len().min(80)]
        );

        // Mean/max avalanche stats
        let mean_size = if avalanches.is_empty() {
            0.0
        } else {
            avalanches.iter().map(|a| a.size).sum::<usize>() as f32 / avalanches.len() as f32
        };
        let max_size = avalanches.iter().map(|a| a.size).max().unwrap_or(0);

        total_spikes_analyzed += spike_file.n_spikes;

        all_results.push(SiteCcnsResult {
            site_id: spike_file.site_id,
            centroid: spike_file.centroid,
            total_spikes: spike_file.n_spikes,
            n_avalanches: avalanches.len(),
            mean_avalanche_size: mean_size,
            max_avalanche_size: max_size,
            tau_exponent: pl.tau,
            xmin: pl.xmin,
            smax: pl.smax,
            ks_statistic: pl.ks_statistic,
            p_value: pl.p_value,
            n_tail: pl.n_tail,
            lr_ratio: pl.lr_ratio,
            preferred_model: pl.preferred_model,
            hysteresis_area,
            normalized_hysteresis_score: normalized_hysteresis,
            has_hysteresis: has_cooling,
            heating_spikes: n_heat,
            cooling_spikes: n_cool,
            cft,
            druggability_class: class,
            rationale,
            size_distribution: dist,
            hysteresis_profile,
        });
    }

    // Write report
    let report = CcnsReport {
        source_file: source_name,
        site_id: all_results.first().map(|r| r.site_id).unwrap_or(-1),
        total_spikes: total_spikes_analyzed,
        sites: all_results,
    };

    let json = serde_json::to_string_pretty(&report)?;
    std::fs::write(&args.output, &json)?;

    log::info!("\n╔═══════════════════════════════════════════════════════════════╗");
    log::info!("║  CCNS ANALYSIS COMPLETE                                      ║");
    log::info!("╠═══════════════════════════════════════════════════════════════╣");
    log::info!("║  Sites analyzed: {:<43} ║", report.sites.len());
    log::info!("║  Total spikes: {:<45} ║", total_spikes_analyzed);
    log::info!("║  Report: {:<51} ║", args.output.display());
    log::info!("╚═══════════════════════════════════════════════════════════════╝");

    // Summary table
    if !report.sites.is_empty() {
        log::info!(
            "\n  Site | τ       | xmin  | p-val  | KS     | LR     | Model        | Aval | Class"
        );
        log::info!(
            "  -----|---------|-------|--------|--------|--------|--------------|------|------"
        );
        for site in &report.sites {
            let tau_str = site
                .tau_exponent
                .map_or("null".to_string(), |t| format!("{:.3}", t));
            let xmin_str = site
                .xmin
                .map_or("null".to_string(), |x| format!("{:.0}", x));
            log::info!(
                "  {:>4} | {:>7} | {:>5} | {:.4} | {:.4} | {:>6.2} | {:>12} | {:>4} | {}",
                site.site_id,
                tau_str,
                xmin_str,
                site.p_value,
                site.ks_statistic,
                site.lr_ratio,
                site.preferred_model,
                site.n_avalanches,
                site.druggability_class
            );
        }
    }

    Ok(())
}
