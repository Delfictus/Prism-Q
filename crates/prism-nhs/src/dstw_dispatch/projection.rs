//! Topology-aware Option A projection of variant perturbations onto the
//! paired inactive/active WT thermodynamic tensors.
//!
//! The live-fire path deliberately avoids a single deterministic rotamer.
//! For each mutation it samples a deterministic ensemble of bounded local
//! side-chain placements around the WT Cα/Cβ frame, scores each placement
//! against the static WT topology, and then performs the DSTW 8D convolution:
//!
//! ```text
//! P_active_shift   = F_steric × active_te_out
//! P_lock_shift     = F_steric × inactive_delta_hc
//! P_ensemble_shift = F_steric × max(active_sigma_hydration_sq,
//!                                   inactive_sigma_hydration_sq)
//! ```
//!
//! The reported `delta_P_*` values are rotamer-ensemble means. The
//! `sigma_delta_P_*` values are the corresponding ensemble standard
//! deviations, floored by the model residual variances so DSTW never sees
//! infinite certainty.

use std::collections::HashMap;

use crate::input::PrismPrepTopology;

use super::sidechain_tables::{delta_q_v, rigid_backbone_compatible, AminoAcid};

const DEFAULT_ROTAMER_SAMPLES: usize = 20;
const GOLDEN_RATIO_FRAC: f64 = 0.618_033_988_749_894_9;

/// WT thermodynamic tensor pack consumed by the projection.
///
/// Rows are indexed by explicit UniProt residue number, not by a dense range.
/// This matters for shared-core exports: active and inactive structures can
/// disagree on unresolved loops, and DSTW intentionally inner-joins only the
/// common physical core.
#[derive(Debug, Clone)]
pub struct WTTensorPack {
    pub residue_numbers: Vec<i32>,
    pub residue_index_lo: i32,
    pub residue_index_hi: i32,
    pub inactive_te_out: Vec<f64>,
    pub inactive_te_in: Vec<f64>,
    pub inactive_delta_hc: Vec<f64>,
    pub inactive_sigma_hydration_sq: Vec<f64>,
    pub active_te_out: Vec<f64>,
    pub active_te_in: Vec<f64>,
    pub active_delta_hc: Vec<f64>,
    pub active_sigma_hydration_sq: Vec<f64>,
}

impl WTTensorPack {
    pub fn n_residues(&self) -> usize {
        self.residue_numbers.len()
    }

    pub fn index_for(&self, residue_number: i32) -> Option<usize> {
        self.residue_numbers
            .iter()
            .position(|r| *r == residue_number)
    }

    pub fn validate(&self) -> Result<(), String> {
        let n = self.n_residues();
        if n == 0 {
            return Err("WT tensor pack is empty".to_string());
        }
        let mut sorted = self.residue_numbers.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != self.residue_numbers.len() {
            return Err("WT tensor pack has duplicate residue numbers".to_string());
        }
        for (name, v) in [
            ("inactive_te_out", &self.inactive_te_out),
            ("inactive_te_in", &self.inactive_te_in),
            ("inactive_delta_hc", &self.inactive_delta_hc),
            (
                "inactive_sigma_hydration_sq",
                &self.inactive_sigma_hydration_sq,
            ),
            ("active_te_out", &self.active_te_out),
            ("active_te_in", &self.active_te_in),
            ("active_delta_hc", &self.active_delta_hc),
            ("active_sigma_hydration_sq", &self.active_sigma_hydration_sq),
        ] {
            if v.len() != n {
                return Err(format!(
                    "WT tensor {:?} length {} != expected {}",
                    name,
                    v.len(),
                    n
                ));
            }
            if !v.iter().all(|x| x.is_finite()) {
                return Err(format!("WT tensor {:?} contains non-finite values", name));
            }
        }
        for (name, v) in [
            (
                "inactive_sigma_hydration_sq",
                &self.inactive_sigma_hydration_sq,
            ),
            ("active_sigma_hydration_sq", &self.active_sigma_hydration_sq),
        ] {
            if v.iter().any(|x| *x < 0.0) {
                return Err(format!("WT tensor {:?} contains negative variance", name));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct AtomProbe {
    position: [f64; 3],
    radius: f64,
    residue_number: i32,
}

#[derive(Debug, Clone)]
struct ResidueLocalEnvironment {
    residue_number: i32,
    residue_name: String,
    ca: [f64; 3],
    cb_direction: [f64; 3],
}

/// Static WT environment for one physiological state.
#[derive(Debug, Clone)]
pub struct TopologyStateEnvironment {
    state_name: String,
    residues: HashMap<i32, ResidueLocalEnvironment>,
    heavy_atoms: Vec<AtomProbe>,
}

impl TopologyStateEnvironment {
    pub fn from_prism_topology(
        state_name: impl Into<String>,
        topology: &PrismPrepTopology,
    ) -> Self {
        let state_name = state_name.into();
        let xyz = topology.positions_as_xyz();
        let mut heavy_atoms: Vec<AtomProbe> = Vec::new();
        let mut by_residue_idx: HashMap<usize, Vec<usize>> = HashMap::new();

        for atom_idx in 0..topology.n_atoms {
            let topo_residue_idx = topology.residue_ids[atom_idx];
            by_residue_idx
                .entry(topo_residue_idx)
                .or_default()
                .push(atom_idx);
            if is_heavy_element(
                topology
                    .elements
                    .get(atom_idx)
                    .map(String::as_str)
                    .unwrap_or(""),
            ) {
                let residue_number = topology
                    .residues
                    .get(topo_residue_idx)
                    .map(|r| r.residue_id + 1)
                    .unwrap_or(topo_residue_idx as i32 + 1);
                heavy_atoms.push(AtomProbe {
                    position: to_f64_xyz(xyz[atom_idx]),
                    radius: element_radius(
                        topology
                            .elements
                            .get(atom_idx)
                            .map(String::as_str)
                            .unwrap_or(""),
                    ),
                    residue_number,
                });
            }
        }

        let mut residues = HashMap::new();
        for meta in &topology.residues {
            let Some(atom_indices) = by_residue_idx.get(&meta.residue_idx) else {
                continue;
            };
            let residue_number = meta.residue_id + 1;
            let mut ca: Option<[f64; 3]> = None;
            let mut cb: Option<[f64; 3]> = None;
            let mut centroid = [0.0f64; 3];
            let mut centroid_n = 0.0f64;
            for atom_idx in atom_indices {
                if !is_heavy_element(
                    topology
                        .elements
                        .get(*atom_idx)
                        .map(String::as_str)
                        .unwrap_or(""),
                ) {
                    continue;
                }
                let pos = to_f64_xyz(xyz[*atom_idx]);
                centroid[0] += pos[0];
                centroid[1] += pos[1];
                centroid[2] += pos[2];
                centroid_n += 1.0;
                match topology.atom_names.get(*atom_idx).map(String::as_str) {
                    Some("CA") => ca = Some(pos),
                    Some("CB") => cb = Some(pos),
                    _ => {}
                }
            }
            if centroid_n > 0.0 {
                centroid = scale(centroid, 1.0 / centroid_n);
            }
            let ca = ca.unwrap_or(centroid);
            let cb_direction = match cb {
                Some(cb) => normalize(sub(cb, ca)).unwrap_or([1.0, 0.0, 0.0]),
                None => normalize(sub(centroid, ca)).unwrap_or([1.0, 0.0, 0.0]),
            };
            residues.insert(
                residue_number,
                ResidueLocalEnvironment {
                    residue_number,
                    residue_name: meta.residue_name.clone(),
                    ca,
                    cb_direction,
                },
            );
        }
        Self {
            state_name,
            residues,
            heavy_atoms,
        }
    }

    fn residue(&self, residue_number: i32) -> Option<&ResidueLocalEnvironment> {
        self.residues.get(&residue_number)
    }

    fn rotamer_penalty(
        &self,
        residue_number: i32,
        mutant: AminoAcid,
        sample_idx: usize,
    ) -> Option<f64> {
        let residue = self.residue(residue_number)?;
        let direction = sampled_direction(residue.cb_direction, sample_idx);
        let probe_radius = sidechain_probe_radius(mutant);
        let probe_distance = 1.45 + 0.72 * probe_radius;
        let probe = add(residue.ca, scale(direction, probe_distance));
        let mut penalty = 0.0f64;
        for atom in &self.heavy_atoms {
            if atom.residue_number == residue.residue_number {
                continue;
            }
            let d = norm(sub(probe, atom.position));
            let allowed = (probe_radius + atom.radius).max(1e-6);
            if d < allowed {
                let overlap = (allowed - d) / allowed;
                penalty += overlap * overlap;
            }
        }
        // The square root keeps crowded residues from dominating solely by atom
        // count while preserving ordering across rotamers.
        Some(penalty.sqrt())
    }
}

/// Paired active/inactive topology context for Option A live dispatch.
#[derive(Debug, Clone)]
pub struct TopologyProjectionContext {
    pub inactive: TopologyStateEnvironment,
    pub active: TopologyStateEnvironment,
    pub rotamer_samples: usize,
}

impl TopologyProjectionContext {
    pub fn new(
        inactive: TopologyStateEnvironment,
        active: TopologyStateEnvironment,
        rotamer_samples: usize,
    ) -> Result<Self, String> {
        if rotamer_samples < 2 {
            return Err("rotamer_samples must be >= 2".to_string());
        }
        Ok(Self {
            inactive,
            active,
            rotamer_samples,
        })
    }

    pub fn from_prism_topologies(
        inactive: &PrismPrepTopology,
        active: &PrismPrepTopology,
        rotamer_samples: usize,
    ) -> Result<Self, String> {
        Self::new(
            TopologyStateEnvironment::from_prism_topology("inactive", inactive),
            TopologyStateEnvironment::from_prism_topology("active", active),
            rotamer_samples,
        )
    }
}

/// One variant carried into the projection.
#[derive(Debug, Clone)]
pub struct VariantPoint {
    pub residue_number: i32,
    pub wildtype: AminoAcid,
    pub mutant: AminoAcid,
}

/// The three vectorial Δ channels emitted per variant.
#[derive(Debug, Clone, Copy)]
pub struct ProjectedDeltas {
    pub delta_p_active: f64,
    pub delta_p_lock: f64,
    pub delta_p_ensemble: f64,
}

/// The per-channel epistemic uncertainty budget. All non-negative.
#[derive(Debug, Clone, Copy)]
pub struct EpistemicSigmas {
    pub sigma_delta_p_active: f64,
    pub sigma_delta_p_lock: f64,
    pub sigma_delta_p_ensemble: f64,
}

/// Configuration knobs that surface in the dispatcher CLI.
#[derive(Debug, Clone, Copy)]
pub struct ProjectionConfig {
    pub alpha_q: f64,
    pub alpha_v: f64,
    pub model_residual_variance_active: f64,
    pub model_residual_variance_lock: f64,
    pub model_residual_variance_ensemble: f64,
    pub backbone_rmsd_ceiling_angstrom: f64,
    pub jacobian_condition_ceiling: f64,
    pub nonconverged_sigma_penalty: f64,
    pub rotamer_samples: usize,
}

impl Default for ProjectionConfig {
    fn default() -> Self {
        Self {
            alpha_q: 1.0,
            alpha_v: 1.0,
            model_residual_variance_active: 1e-4,
            model_residual_variance_lock: 1e-4,
            model_residual_variance_ensemble: 1e-4,
            backbone_rmsd_ceiling_angstrom: 0.5,
            jacobian_condition_ceiling: 1.0e6,
            nonconverged_sigma_penalty: 4.0,
            rotamer_samples: DEFAULT_ROTAMER_SAMPLES,
        }
    }
}

impl ProjectionConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.nonconverged_sigma_penalty <= 1.0 {
            return Err(format!(
                "nonconverged_sigma_penalty must be > 1.0; got {}",
                self.nonconverged_sigma_penalty
            ));
        }
        if self.rotamer_samples < 2 {
            return Err("rotamer_samples must be >= 2".to_string());
        }
        for (name, v) in [
            (
                "model_residual_variance_active",
                self.model_residual_variance_active,
            ),
            (
                "model_residual_variance_lock",
                self.model_residual_variance_lock,
            ),
            (
                "model_residual_variance_ensemble",
                self.model_residual_variance_ensemble,
            ),
        ] {
            if v <= 0.0 {
                return Err(format!("{name} must be positive; got {v}"));
            }
        }
        if self.backbone_rmsd_ceiling_angstrom <= 0.0 {
            return Err("backbone_rmsd_ceiling_angstrom must be positive".to_string());
        }
        if self.jacobian_condition_ceiling <= 0.0 {
            return Err("jacobian_condition_ceiling must be positive".to_string());
        }
        Ok(())
    }
}

/// Result of projecting a single variant.
#[derive(Debug, Clone)]
pub struct ProjectionResult {
    pub deltas: ProjectedDeltas,
    pub sigmas: EpistemicSigmas,
    pub converged: bool,
    pub failure_reason: Option<String>,
}

/// Backward-compatible projection entry point. Uses the tensor convolution
/// equations with a topology-free synthetic frustration scale; live dispatch
/// should call `project_variant_with_topology`.
pub fn project_variant(
    variant: &VariantPoint,
    wt: &WTTensorPack,
    config: &ProjectionConfig,
) -> ProjectionResult {
    project_variant_with_topology(variant, wt, config, None)
}

/// Project one variant using topology-aware stochastic Option A.
pub fn project_variant_with_topology(
    variant: &VariantPoint,
    wt: &WTTensorPack,
    config: &ProjectionConfig,
    topology: Option<&TopologyProjectionContext>,
) -> ProjectionResult {
    let mut converged = true;
    let mut failure_reason: Option<String> = None;

    let idx = match wt.index_for(variant.residue_number) {
        Some(i) => i,
        None => {
            converged = false;
            failure_reason = Some(format!(
                "residue {} outside shared-core WT tensor pack",
                variant.residue_number
            ));
            let inflated = inflate_sigmas(
                base_sigmas_from_residual(config),
                config.nonconverged_sigma_penalty,
            );
            return ProjectionResult {
                deltas: ProjectedDeltas {
                    delta_p_active: 0.0,
                    delta_p_lock: 0.0,
                    delta_p_ensemble: 0.0,
                },
                sigmas: inflated,
                converged,
                failure_reason,
            };
        }
    };

    if !rigid_backbone_compatible(variant.wildtype, variant.mutant) {
        converged = false;
        failure_reason = Some(format!(
            "Pro↔non-Pro substitution {}{}{} violates rigid-backbone assumption",
            variant.wildtype.as_char(),
            variant.residue_number,
            variant.mutant.as_char()
        ));
    }

    let (_dq, dv) = delta_q_v(variant.wildtype, variant.mutant);
    let dv_ceiling = 50.0 + 60.0 * config.backbone_rmsd_ceiling_angstrom;
    if dv.abs() > dv_ceiling {
        converged = false;
        let reason = format!(
            "side-chain volume shift ΔV={:.1} Å^3 exceeds rigid-backbone ceiling {:.1} Å^3",
            dv, dv_ceiling
        );
        failure_reason = Some(match failure_reason {
            Some(existing) => format!("{existing}; {reason}"),
            None => reason,
        });
    }

    let samples = match topology {
        Some(ctx) => {
            let mut out = Vec::with_capacity(ctx.rotamer_samples);
            for s in 0..ctx.rotamer_samples {
                let inactive =
                    ctx.inactive
                        .rotamer_penalty(variant.residue_number, variant.mutant, s);
                let active = ctx
                    .active
                    .rotamer_penalty(variant.residue_number, variant.mutant, s);
                match (inactive, active) {
                    (Some(i), Some(a)) => {
                        out.push(local_frustration(variant, config, 0.5 * (i + a)))
                    }
                    _ => {
                        converged = false;
                        failure_reason = Some(format!(
                            "residue {} missing from topology projection context",
                            variant.residue_number
                        ));
                        out.push(local_frustration(variant, config, 0.0));
                    }
                }
            }
            out
        }
        None => {
            let n = config.rotamer_samples.max(2);
            (0..n)
                .map(|s| {
                    let jitter = 0.85 + 0.03 * ((s % 11) as f64);
                    local_frustration(variant, config, jitter)
                })
                .collect()
        }
    };

    let (mean_f, std_f) = mean_std(&samples);
    let k_active = wt.active_te_out[idx];
    let k_lock = wt.inactive_delta_hc[idx];
    let k_ensemble = wt.active_sigma_hydration_sq[idx].max(wt.inactive_sigma_hydration_sq[idx]);

    let deltas = ProjectedDeltas {
        delta_p_active: mean_f * k_active,
        delta_p_lock: mean_f * k_lock,
        delta_p_ensemble: mean_f * k_ensemble,
    };

    let mut sigmas = EpistemicSigmas {
        sigma_delta_p_active: (std_f.abs() * k_active.abs())
            .max(config.model_residual_variance_active.sqrt()),
        sigma_delta_p_lock: (std_f.abs() * k_lock.abs())
            .max(config.model_residual_variance_lock.sqrt()),
        sigma_delta_p_ensemble: (std_f.abs() * k_ensemble.abs())
            .max(config.model_residual_variance_ensemble.sqrt()),
    };

    if !converged {
        sigmas = inflate_sigmas(sigmas, config.nonconverged_sigma_penalty);
    }

    ProjectionResult {
        deltas,
        sigmas,
        converged,
        failure_reason,
    }
}

fn local_frustration(
    variant: &VariantPoint,
    config: &ProjectionConfig,
    congestion_penalty: f64,
) -> f64 {
    let (dq, dv) = delta_q_v(variant.wildtype, variant.mutant);
    let volume_component = config.alpha_v.abs() * (dv.abs() / 100.0);
    let charge_component = config.alpha_q.abs() * (0.25 * dq.abs());
    let perturbation = (volume_component + charge_component).max(1e-6);
    perturbation * (0.05 + congestion_penalty.max(0.0))
}

fn base_sigmas_from_residual(cfg: &ProjectionConfig) -> EpistemicSigmas {
    EpistemicSigmas {
        sigma_delta_p_active: cfg.model_residual_variance_active.sqrt(),
        sigma_delta_p_lock: cfg.model_residual_variance_lock.sqrt(),
        sigma_delta_p_ensemble: cfg.model_residual_variance_ensemble.sqrt(),
    }
}

fn inflate_sigmas(s: EpistemicSigmas, factor: f64) -> EpistemicSigmas {
    EpistemicSigmas {
        sigma_delta_p_active: s.sigma_delta_p_active * factor,
        sigma_delta_p_lock: s.sigma_delta_p_lock * factor,
        sigma_delta_p_ensemble: s.sigma_delta_p_ensemble * factor,
    }
}

fn mean_std(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n.max(1.0);
    (mean, var.max(0.0).sqrt())
}

fn sidechain_probe_radius(aa: AminoAcid) -> f64 {
    let volume = aa.descriptor().volume_angstrom3.max(1.0);
    ((3.0 * volume) / (4.0 * std::f64::consts::PI))
        .cbrt()
        .max(1.0)
}

fn sampled_direction(base: [f64; 3], sample_idx: usize) -> [f64; 3] {
    let base = normalize(base).unwrap_or([1.0, 0.0, 0.0]);
    let (u, v) = orthonormal_basis(base);
    let theta = 2.0 * std::f64::consts::PI * ((sample_idx as f64 * GOLDEN_RATIO_FRAC) % 1.0);
    let amplitude = 0.22 + 0.04 * ((sample_idx % 7) as f64);
    normalize(add(
        base,
        add(
            scale(u, amplitude * theta.cos()),
            scale(v, amplitude * theta.sin()),
        ),
    ))
    .unwrap_or(base)
}

fn orthonormal_basis(n: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let helper = if n[0].abs() < 0.8 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let u = normalize(cross(n, helper)).unwrap_or([0.0, 0.0, 1.0]);
    let v = normalize(cross(n, u)).unwrap_or([0.0, 1.0, 0.0]);
    (u, v)
}

fn is_heavy_element(element: &str) -> bool {
    let e = element.trim().to_ascii_uppercase();
    !e.is_empty() && e != "H" && e != "D"
}

fn element_radius(element: &str) -> f64 {
    match element.trim().to_ascii_uppercase().as_str() {
        "C" => 1.70,
        "N" => 1.55,
        "O" => 1.52,
        "S" => 1.80,
        "P" => 1.80,
        _ => 1.60,
    }
}

fn to_f64_xyz(x: [f32; 3]) -> [f64; 3] {
    [x[0] as f64, x[1] as f64, x[2] as f64]
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn norm(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

fn normalize(a: [f64; 3]) -> Option<[f64; 3]> {
    let n = norm(a);
    if n.is_finite() && n > 1e-9 {
        Some(scale(a, 1.0 / n))
    } else {
        None
    }
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_wt(n: usize) -> WTTensorPack {
        let residues: Vec<i32> = (1..=n as i32).collect();
        WTTensorPack {
            residue_numbers: residues,
            residue_index_lo: 1,
            residue_index_hi: n as i32,
            inactive_te_out: vec![0.1; n],
            inactive_te_in: vec![0.2; n],
            inactive_delta_hc: (0..n).map(|i| 0.3 + 0.05 * i as f64).collect(),
            inactive_sigma_hydration_sq: (0..n).map(|i| 0.1 + 0.01 * i as f64).collect(),
            active_te_out: (0..n).map(|i| 0.5 + 0.1 * i as f64).collect(),
            active_te_in: vec![0.6; n],
            active_delta_hc: vec![0.2; n],
            active_sigma_hydration_sq: (0..n).map(|i| 0.15 + 0.01 * i as f64).collect(),
        }
    }

    #[test]
    fn projection_uses_exact_8d_convolution_channels() {
        let wt = synth_wt(10);
        let cfg = ProjectionConfig {
            rotamer_samples: 20,
            ..ProjectionConfig::default()
        };
        let v = VariantPoint {
            residue_number: 5,
            wildtype: AminoAcid::K,
            mutant: AminoAcid::A,
        };
        let r = project_variant(&v, &wt, &cfg);
        let idx = 4;
        let k_active = wt.active_te_out[idx];
        let k_lock = wt.inactive_delta_hc[idx];
        let k_ensemble = wt.active_sigma_hydration_sq[idx].max(wt.inactive_sigma_hydration_sq[idx]);
        let mean_f = r.deltas.delta_p_active / k_active;
        assert!((r.deltas.delta_p_lock - mean_f * k_lock).abs() < 1e-9);
        assert!((r.deltas.delta_p_ensemble - mean_f * k_ensemble).abs() < 1e-9);
        assert!(r.sigmas.sigma_delta_p_active > 0.0);
        assert!(r.sigmas.sigma_delta_p_lock > 0.0);
        assert!(r.sigmas.sigma_delta_p_ensemble > 0.0);
    }

    #[test]
    fn sparse_residue_numbers_are_supported() {
        let mut wt = synth_wt(3);
        wt.residue_numbers = vec![10, 20, 30];
        wt.residue_index_lo = 10;
        wt.residue_index_hi = 30;
        assert_eq!(wt.index_for(20), Some(1));
        assert_eq!(wt.index_for(11), None);
        assert!(wt.validate().is_ok());
    }

    #[test]
    fn nonconverged_residue_outside_core_inflates_sigma() {
        let wt = synth_wt(3);
        let cfg = ProjectionConfig::default();
        let v = VariantPoint {
            residue_number: 99,
            wildtype: AminoAcid::L,
            mutant: AminoAcid::A,
        };
        let r = project_variant(&v, &wt, &cfg);
        assert!(!r.converged);
        assert!(r.sigmas.sigma_delta_p_active > cfg.model_residual_variance_active.sqrt());
    }
}
