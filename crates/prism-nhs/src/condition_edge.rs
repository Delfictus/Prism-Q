//! Condition-edge readout — the production unit of the DirectedPerturbationDifferential (DPD).
//!
//! The canonical object is NOT a delta. It is a directed CONDITION EDGE:
//!     condition_a (thermal_only)  --DirectedPerturbationDifferential-->  condition_b (thermal_plus_nma)
//! Deltas / ratios / densities / event counts are *measurements attached to that edge*, computed
//! condition-natively from the per-residue interferometric features (which already carry the
//! NMA-exclusive / B-over-A / per-phase differential fields). The legacy `total_density_a_to_b/b_to_a`
//! are kept only as side telemetry; a zero there MUST NOT suppress this readout.
//!
//! Source of truth = QNANL SPIKE_EVENT_STREAM + AlwaysOnSpikeSensorBus + this condition-edge
//! readout. Emitted as CONDITION_EDGE_READOUT/{condition_edge_manifest.json,
//! per_residue_condition_edges.jsonl, condition_edge_summary.qnanl} + receipts.

use crate::coupled_md::InterferometricFeatures;
use crate::fused_engine::GpuSpikeEvent;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const CONDITION_EDGE_SCHEMA_KIND: &str = "directed_perturbation_condition_edge_readout";
pub const CONDITION_EDGE_SCHEMA_VERSION: u32 = 1;

/// Material differential threshold: |b_over_a_ratio - 1| above this counts as a condition-edge event.
const MATERIAL_RATIO_DELTA: f32 = 0.05;
/// Runaway guard: any b_over_a_ratio above this fails the stability gate (spike flood / instability).
const RUNAWAY_RATIO: f32 = 100.0;

/// One pole of the directed condition edge.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConditionNode {
    pub role: String,       // scout | focused_observer
    pub family: String,     // thermal_only | thermal_plus_nma
    pub amplitude: f32,     // NMA amplification (0 for thermal_only)
    pub scan_fraction: f32, // NMA scan fraction (0 for thermal_only)
}

/// Per-residue projection of the directed edge A→B.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PerResidueConditionEdge {
    pub resid: i32,
    pub resname: String,
    pub spikes_a: u32,
    pub spikes_b: u32,
    pub b_over_a_ratio: f32,
    pub b_over_a_intensity_ratio: f32,
    pub nma_exclusive_count: u32,
    pub thermal_exclusive_count: u32,
    pub warm_hold_differential: f32, // per_phase_differential[2] (NMA phase)
    pub mechanical_sensitivity: f32,
    pub barrier_classification: String,
    pub edge_score: f32, // per-residue edge magnitude (weighted ratio deviation)
    pub condition_exclusive: bool, // B-exclusive activity attributable to the perturbation
}

/// Condition-native aggregate metrics (the measurements on the edge).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConditionEdgeMetrics {
    pub condition_edge_score: f64,
    pub condition_exclusive_residues: u64,
    pub condition_enrichment_ratio: f64,
    pub condition_phase_enrichment: f64,
    pub condition_intensity_shift: f64,
    pub condition_sensor_shift: f64,
    pub condition_edge_events: u64,
    pub condition_stability_gate: bool,
    pub stability_reasons: Vec<String>,
    // component breakdowns (transparency)
    pub water_shift: f64,
    pub vibration_shift: f64,
    pub total_spikes_a: u64,
    pub total_spikes_b: u64,
}

/// The full directed-perturbation condition-edge readout.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConditionEdgeReadout {
    pub schema_kind: String,
    pub schema_version: u32,
    pub perturbation_edge_id: String,
    pub twin_pair_id: String,
    pub condition_a: ConditionNode,
    pub condition_b: ConditionNode,
    pub n_residues: usize,
    pub metrics: ConditionEdgeMetrics,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub topology_source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spike_stream_manifest: Option<String>,
    /// per-residue edges are serialized to the sibling .jsonl, not inlined here.
    #[serde(skip)]
    pub per_residue: Vec<PerResidueConditionEdge>,
}

fn mean_field(spikes: &[GpuSpikeEvent], f: impl Fn(&GpuSpikeEvent) -> f64) -> f64 {
    if spikes.is_empty() {
        return 0.0;
    }
    spikes.iter().map(f).sum::<f64>() / spikes.len() as f64
}

/// Build the directed condition-edge readout from per-residue interferometric features + the two
/// spike arms. `amplitude`/`scan_fraction` describe condition B's perturbation (A is thermal_only).
pub fn compute_condition_edge(
    features: &[InterferometricFeatures],
    spikes_a: &[GpuSpikeEvent],
    spikes_b: &[GpuSpikeEvent],
    amplitude: f32,
    scan_fraction: f32,
    perturbation_edge_id: String,
    twin_pair_id: String,
    topology_source: Option<String>,
    spike_stream_manifest: Option<String>,
) -> ConditionEdgeReadout {
    let mut reasons: Vec<String> = Vec::new();
    let mut per_residue = Vec::with_capacity(features.len());

    let mut weighted_delta_sum = 0.0f64;
    let mut weight_sum = 0.0f64;
    let mut exclusive_residues = 0u64;
    let mut edge_events = 0u64;
    let mut intensity_delta_sum = 0.0f64;
    let mut intensity_n = 0u64;
    let mut warm_hold_sum = 0.0f64;
    let mut total_a = 0u64;
    let mut total_b = 0u64;
    let mut stable = true;

    for f in features {
        let ratio = f.b_over_a_ratio;
        if !ratio.is_finite() {
            stable = false;
            reasons.push(format!("resid {} non-finite b_over_a_ratio", f.resid));
        } else if ratio > RUNAWAY_RATIO {
            stable = false;
            reasons.push(format!(
                "resid {} runaway b_over_a_ratio={:.2}",
                f.resid, ratio
            ));
        }
        let w = (f.spikes_a + f.spikes_b) as f64;
        let delta = (ratio - 1.0).abs() as f64;
        weighted_delta_sum += w * delta;
        weight_sum += w;
        total_a += f.spikes_a as u64;
        total_b += f.spikes_b as u64;

        let exclusive = f.nma_exclusive_count > 0;
        if exclusive {
            exclusive_residues += 1;
        }
        if delta > MATERIAL_RATIO_DELTA as f64 || exclusive {
            edge_events += 1;
        }
        if f.b_over_a_intensity_ratio.is_finite() && f.b_over_a_intensity_ratio > 0.0 {
            intensity_delta_sum += (f.b_over_a_intensity_ratio - 1.0) as f64;
            intensity_n += 1;
        }
        let warm = f.per_phase_differential.get(2).copied().unwrap_or(0.0);
        if warm.is_finite() {
            warm_hold_sum += warm as f64;
        }

        let edge_score = w * delta;
        per_residue.push(PerResidueConditionEdge {
            resid: f.resid,
            resname: f.resname.clone(),
            spikes_a: f.spikes_a,
            spikes_b: f.spikes_b,
            b_over_a_ratio: ratio,
            b_over_a_intensity_ratio: f.b_over_a_intensity_ratio,
            nma_exclusive_count: f.nma_exclusive_count,
            thermal_exclusive_count: f.thermal_exclusive_count,
            warm_hold_differential: warm,
            mechanical_sensitivity: f.mechanical_sensitivity,
            barrier_classification: f.barrier_classification.clone(),
            edge_score: edge_score as f32,
            condition_exclusive: exclusive,
        });
    }

    let n = features.len().max(1) as f64;
    let condition_edge_score = if weight_sum > 0.0 {
        weighted_delta_sum / weight_sum
    } else {
        0.0
    };
    let condition_enrichment_ratio = total_b as f64 / total_a.max(1) as f64;
    let condition_phase_enrichment = warm_hold_sum / n;
    let condition_intensity_shift = if intensity_n > 0 {
        intensity_delta_sum / intensity_n as f64
    } else {
        0.0
    };

    // sensor shift from the raw arms (arm B − arm A), water + vibration.
    let water_shift = mean_field(spikes_b, |s| s.water_density as f64)
        - mean_field(spikes_a, |s| s.water_density as f64);
    let vibration_shift = mean_field(spikes_b, |s| s.vibrational_energy as f64)
        - mean_field(spikes_a, |s| s.vibrational_energy as f64);
    let condition_sensor_shift =
        (water_shift * water_shift + vibration_shift * vibration_shift).sqrt();

    for (name, v) in [
        ("condition_edge_score", condition_edge_score),
        ("condition_enrichment_ratio", condition_enrichment_ratio),
        ("condition_phase_enrichment", condition_phase_enrichment),
        ("condition_intensity_shift", condition_intensity_shift),
        ("condition_sensor_shift", condition_sensor_shift),
    ] {
        if !v.is_finite() {
            stable = false;
            reasons.push(format!("{name} non-finite"));
        }
    }
    if total_b == 0 {
        stable = false;
        reasons.push("arm B produced zero spikes (spike starvation)".to_string());
    }

    let metrics = ConditionEdgeMetrics {
        condition_edge_score,
        condition_exclusive_residues: exclusive_residues,
        condition_enrichment_ratio,
        condition_phase_enrichment,
        condition_intensity_shift,
        condition_sensor_shift,
        condition_edge_events: edge_events,
        condition_stability_gate: stable,
        stability_reasons: reasons,
        water_shift,
        vibration_shift,
        total_spikes_a: total_a,
        total_spikes_b: total_b,
    };

    ConditionEdgeReadout {
        schema_kind: CONDITION_EDGE_SCHEMA_KIND.to_string(),
        schema_version: CONDITION_EDGE_SCHEMA_VERSION,
        perturbation_edge_id,
        twin_pair_id,
        condition_a: ConditionNode {
            role: "scout".to_string(),
            family: "thermal_only".to_string(),
            amplitude: 0.0,
            scan_fraction: 0.0,
        },
        condition_b: ConditionNode {
            role: "focused_observer".to_string(),
            family: "thermal_plus_nma".to_string(),
            amplitude,
            scan_fraction,
        },
        n_residues: features.len(),
        metrics,
        topology_source,
        spike_stream_manifest,
        per_residue,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TWIN16-NATIVE condition edge (Directive v2 Lane A1).
//
// The DPD substrate is the TWIN16 2×2×2×2 thermodynamic-condition factorial. The directed
// condition edges are the single-axis cell flips, scored by `transition_view_l1_residual`. The
// 4 axes are thermal / uv / ramp / return — there is NO NMA axis, so a literal
// thermal_only→thermal_plus_nma cell pair is ABSENT and is ledgered (not substituted with the
// coupled A/B evidence), per the directive.
// ─────────────────────────────────────────────────────────────────────────────

/// TWIN16 condition axes, indexed by the differing bit (see twin16_condition_value in the engine).
const TWIN16_AXES: [(&str, &str, &str); 4] = [
    ("thermal", "cold_start_minus_10K", "elevated_start_plus_10K"),
    ("uv", "uv_energy_0_80x", "uv_energy_1_20x"),
    ("ramp", "fast_ramp_0_75x", "slow_ramp_1_50x"),
    ("return", "brief_return_0_75x", "extended_return_1_50x"),
];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Twin16AxisEdge {
    pub axis: String,
    pub pole_a: String,
    pub pole_b: String,
    pub n_pairs: usize,
    pub mean_residual: f64,
    pub max_residual: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Twin16CellPairEdge {
    pub cell_a: String,
    pub cell_b: String,
    pub hamming: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub axis: Option<String>,
    pub transition_view_l1_residual: f64,
    pub material: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Twin16ConditionEdgeReadout {
    pub schema_kind: String,
    pub schema_version: u32,
    pub run_id: String,
    pub target: String,
    pub condition_substrate: String,
    pub axes: Vec<String>,
    pub nma_axis_present: bool,
    pub missing_cell_relation: String,
    pub n_cells: usize,
    pub n_pairs: usize,
    pub metrics: Twin16EdgeMetrics,
    pub axis_edges: Vec<Twin16AxisEdge>,
    #[serde(skip)]
    pub cell_pair_edges: Vec<Twin16CellPairEdge>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Twin16EdgeMetrics {
    pub condition_edge_score: f64, // mean L1 residual over single-axis (hamming-1) edges
    pub condition_edge_events: u64, // single-axis edges above the material threshold
    pub condition_exclusive_cells: u64, // cells with zero spike activity (condition-exclusive silence)
    pub condition_enrichment_ratio: f64, // max_cell_spikes / mean_cell_spikes
    pub principal_axis: String,         // axis carrying the largest mean residual
    pub principal_axis_residual: f64,
    pub condition_stability_gate: bool,
    pub stability_reasons: Vec<String>,
}

fn twin16_idx(condition_id: &str) -> Option<u8> {
    // condition_id == "TWIN16_bbbb" (4 binary digits); parse the 4-bit cell index.
    let bits = condition_id.strip_prefix("TWIN16_")?;
    if bits.len() != 4 || !bits.chars().all(|c| c == '0' || c == '1') {
        return None;
    }
    u8::from_str_radix(bits, 2).ok()
}

/// Build the TWIN16-native directed condition edge from the cells[] + pairwise_residuals[] of a
/// TWIN16_INTERFEROMETRIC_TRANSITION_VIEW_RESIDUALS artifact.
pub fn compute_twin16_condition_edge(
    cells: &[serde_json::Value],
    pairwise: &[serde_json::Value],
    run_id: String,
    target: String,
) -> Twin16ConditionEdgeReadout {
    const MATERIAL: f64 = 0.05;
    let mut reasons: Vec<String> = Vec::new();

    // per-axis residual aggregation over single-bit-flip (hamming-1) cell pairs.
    let mut axis_sum = [0.0f64; 4];
    let mut axis_max = [0.0f64; 4];
    let mut axis_n = [0usize; 4];
    let mut cell_pair_edges = Vec::new();
    let mut all_hamming1: Vec<f64> = Vec::new();
    let mut edge_events = 0u64;
    let mut stable = true;

    for p in pairwise {
        let ca = p.get("cell_a").and_then(|v| v.as_str()).unwrap_or("");
        let cb = p.get("cell_b").and_then(|v| v.as_str()).unwrap_or("");
        let residual = p
            .get("transition_view_l1_residual")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let (ia, ib) = match (twin16_idx(ca), twin16_idx(cb)) {
            (Some(a), Some(b)) => (a, b),
            _ => continue,
        };
        if ia == ib {
            continue; // self-pair (diagonal)
        }
        let xor = ia ^ ib;
        let hamming = xor.count_ones();
        if !residual.is_finite() {
            stable = false;
            reasons.push(format!("{ca}->{cb} non-finite residual"));
        }
        let axis = if hamming == 1 {
            let bit = xor.trailing_zeros() as usize; // 0..=3
            if bit < 4 {
                axis_sum[bit] += residual;
                axis_max[bit] = axis_max[bit].max(residual);
                axis_n[bit] += 1;
                all_hamming1.push(residual);
                if residual > MATERIAL {
                    edge_events += 1;
                }
                Some(TWIN16_AXES[bit].0.to_string())
            } else {
                None
            }
        } else {
            None
        };
        cell_pair_edges.push(Twin16CellPairEdge {
            cell_a: ca.to_string(),
            cell_b: cb.to_string(),
            hamming,
            axis,
            transition_view_l1_residual: residual,
            material: residual > MATERIAL,
        });
    }

    let mut axis_edges = Vec::new();
    let (mut principal_axis, mut principal_residual) = (String::from("none"), 0.0f64);
    for (bit, (name, pole_a, pole_b)) in TWIN16_AXES.iter().enumerate() {
        let n = axis_n[bit];
        let mean = if n > 0 { axis_sum[bit] / n as f64 } else { 0.0 };
        axis_edges.push(Twin16AxisEdge {
            axis: name.to_string(),
            pole_a: pole_a.to_string(),
            pole_b: pole_b.to_string(),
            n_pairs: n,
            mean_residual: mean,
            max_residual: axis_max[bit],
        });
        if mean > principal_residual {
            principal_residual = mean;
            principal_axis = name.to_string();
        }
    }

    let condition_edge_score = if all_hamming1.is_empty() {
        0.0
    } else {
        all_hamming1.iter().sum::<f64>() / all_hamming1.len() as f64
    };

    // cell spike summaries for enrichment + exclusivity.
    let spikes: Vec<f64> = cells
        .iter()
        .map(|c| {
            c.get("spike_count_total")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0)
        })
        .collect();
    let exclusive = spikes.iter().filter(|&&s| s == 0.0).count() as u64;
    let mean_spk = if spikes.is_empty() {
        0.0
    } else {
        spikes.iter().sum::<f64>() / spikes.len() as f64
    };
    let max_spk = spikes.iter().cloned().fold(0.0f64, f64::max);
    let enrichment = if mean_spk > 0.0 {
        max_spk / mean_spk
    } else {
        0.0
    };

    if cells.len() < 16 {
        reasons.push(format!(
            "under-sampled: {} cells (<16 = factorial not fully populated)",
            cells.len()
        ));
    }
    if all_hamming1.is_empty() {
        stable = false;
        reasons.push("no single-axis (hamming-1) cell edges found".to_string());
    }

    Twin16ConditionEdgeReadout {
        schema_kind: "directed_perturbation_twin16_condition_edge_readout".to_string(),
        schema_version: 1,
        run_id,
        target,
        condition_substrate: "twin16_2x2x2x2_thermodynamic_cells".to_string(),
        axes: TWIN16_AXES.iter().map(|a| a.0.to_string()).collect(),
        nma_axis_present: false,
        missing_cell_relation:
            "thermal_only->thermal_plus_nma is ABSENT from the TWIN16 factorial: the 4 axes are \
             thermal/uv/ramp/return; NMA is a coupled-path (--coupled-twin) perturbation, not a \
             TWIN16 condition axis. Ledgered per Directive v2 A1; NOT substituted with coupled A/B."
                .to_string(),
        n_cells: cells.len(),
        n_pairs: pairwise.len(),
        metrics: Twin16EdgeMetrics {
            condition_edge_score,
            condition_edge_events: edge_events,
            condition_exclusive_cells: exclusive,
            condition_enrichment_ratio: enrichment,
            principal_axis,
            principal_axis_residual: principal_residual,
            condition_stability_gate: stable,
            stability_reasons: reasons,
        },
        axis_edges,
        cell_pair_edges,
    }
}

/// Emit CONDITION_EDGE_READOUT/{condition_edge_manifest.json, per_cellpair_condition_edges.jsonl,
/// condition_edge_summary.qnanl + receipt} for the TWIN16-native edge. Round-trip verified.
pub fn emit_twin16_condition_edge(
    readout: &Twin16ConditionEdgeReadout,
    output_dir: &Path,
) -> std::io::Result<ConditionEdgeEmitResult> {
    use prismcode::qnanl_artifact::{
        decode_packet_bytes, encode_bytes, QnanlArtifactFamily, QnanlEncodeOptions,
    };
    let dir = output_dir.join("CONDITION_EDGE_READOUT");
    std::fs::create_dir_all(&dir)?;
    let manifest_path = dir.join("condition_edge_manifest.json");
    let manifest_bytes = serde_json::to_vec_pretty(readout).unwrap_or_default();
    std::fs::write(&manifest_path, &manifest_bytes)?;

    let per_residue_path = dir.join("per_cellpair_condition_edges.jsonl");
    let mut jsonl = Vec::new();
    for e in &readout.cell_pair_edges {
        if let Ok(line) = serde_json::to_vec(e) {
            jsonl.extend_from_slice(&line);
            jsonl.push(b'\n');
        }
    }
    std::fs::write(&per_residue_path, &jsonl)?;

    let qnanl_path = dir.join("condition_edge_summary.qnanl");
    let rcpt_path = dir.join("condition_edge_summary.qnanl.receipt.json");
    let opts = QnanlEncodeOptions::runtime_subprocess_orchestrated_by_prismos(
        format!("twin16_condition_edge_{}", readout.run_id),
        QnanlArtifactFamily::JsonTelemetry,
        None,
        "condition_edge::emit_twin16_condition_edge_readout",
    );
    let (qnanl_summary_path, qnanl_roundtrip_verified) =
        match encode_bytes(&manifest_bytes, &qnanl_path, &rcpt_path, opts) {
            Ok(_) => {
                let pkt = std::fs::read(&qnanl_path).unwrap_or_default();
                let verified = decode_packet_bytes(&pkt)
                    .map(|d| d == manifest_bytes)
                    .unwrap_or(false);
                (Some(qnanl_path), verified)
            }
            Err(e) => {
                log::warn!("  [TWIN16_CONDITION_EDGE] QNANL seal failed: {}", e);
                (None, false)
            }
        };
    // A3: receipt the condition-edge readout (MD.CONDITION_EDGE_COMMIT) with the manifest hash.
    let manifest_hash = blake3::hash(&manifest_bytes).to_hex().to_string();
    let receipt = serde_json::json!({
        "schema_kind": prismcode::md_native_runtime::MD_NATIVE_SCHEMA_KIND,
        "schema_version": 1,
        "surface": "CONDITION_EDGE_READOUT (TWIN16-native)",
        "ops": [{
            "op": prismcode::md_native_runtime::OP_MD_CONDITION_EDGE_COMMIT,
            "perturbation_edge_id": readout.run_id,
            "condition_substrate": readout.condition_substrate,
            "principal_axis": readout.metrics.principal_axis,
            "condition_edge_score": readout.metrics.condition_edge_score,
            "condition_edge_events": readout.metrics.condition_edge_events,
            "nma_axis_present": readout.nma_axis_present,
            "manifest_hash_blake3": manifest_hash,
            "qnanl_roundtrip_verified": qnanl_roundtrip_verified,
        }],
        "verifier": "re-hash condition_edge_manifest.json (blake3) and compare to manifest_hash_blake3; fail on mutated edge evidence",
    });
    let _ = std::fs::write(
        dir.join("RECEIPT.prismcode.json"),
        serde_json::to_string_pretty(&receipt).unwrap_or_default(),
    );
    Ok(ConditionEdgeEmitResult {
        readout_dir: dir,
        manifest_path,
        per_residue_path,
        qnanl_summary_path,
        qnanl_roundtrip_verified,
    })
}

/// Result of emitting the CONDITION_EDGE_READOUT directory.
#[derive(Clone, Debug)]
pub struct ConditionEdgeEmitResult {
    pub readout_dir: PathBuf,
    pub manifest_path: PathBuf,
    pub per_residue_path: PathBuf,
    pub qnanl_summary_path: Option<PathBuf>,
    pub qnanl_roundtrip_verified: bool,
}

/// Emit CONDITION_EDGE_READOUT/{condition_edge_manifest.json, per_residue_condition_edges.jsonl,
/// condition_edge_summary.qnanl} + receipts under `output_dir`. The .qnanl seal is round-trip
/// verified (decode == raw) so the tokenizer can prove decoded-edge == raw-edge.
pub fn emit_condition_edge_readout(
    readout: &ConditionEdgeReadout,
    output_dir: &Path,
) -> std::io::Result<ConditionEdgeEmitResult> {
    use prismcode::qnanl_artifact::{
        decode_packet_bytes, encode_bytes, QnanlArtifactFamily, QnanlEncodeOptions,
    };
    let dir = output_dir.join("CONDITION_EDGE_READOUT");
    std::fs::create_dir_all(&dir)?;

    // manifest = the readout WITHOUT the per-residue array (which goes to the .jsonl).
    let manifest_path = dir.join("condition_edge_manifest.json");
    let manifest_bytes = serde_json::to_vec_pretty(readout).unwrap_or_default();
    std::fs::write(&manifest_path, &manifest_bytes)?;

    // per-residue edges → JSONL (one edge per line).
    let per_residue_path = dir.join("per_residue_condition_edges.jsonl");
    let mut jsonl = Vec::new();
    for e in &readout.per_residue {
        if let Ok(line) = serde_json::to_vec(e) {
            jsonl.extend_from_slice(&line);
            jsonl.push(b'\n');
        }
    }
    std::fs::write(&per_residue_path, &jsonl)?;

    // QNANL-sealed summary (the manifest bytes), round-trip verified.
    let qnanl_path = dir.join("condition_edge_summary.qnanl");
    let rcpt_path = dir.join("condition_edge_summary.qnanl.receipt.json");
    let opts = QnanlEncodeOptions::runtime_subprocess_orchestrated_by_prismos(
        readout.perturbation_edge_id.clone(),
        QnanlArtifactFamily::JsonTelemetry,
        None,
        "condition_edge::emit_condition_edge_readout",
    );
    let (qnanl_summary_path, qnanl_roundtrip_verified) =
        match encode_bytes(&manifest_bytes, &qnanl_path, &rcpt_path, opts) {
            Ok(_) => {
                let pkt = std::fs::read(&qnanl_path).unwrap_or_default();
                let verified = decode_packet_bytes(&pkt)
                    .map(|dec| dec == manifest_bytes)
                    .unwrap_or(false);
                (Some(qnanl_path), verified)
            }
            Err(e) => {
                log::warn!("  [CONDITION_EDGE] QNANL seal failed: {}", e);
                (None, false)
            }
        };

    Ok(ConditionEdgeEmitResult {
        readout_dir: dir,
        manifest_path,
        per_residue_path,
        qnanl_summary_path,
        qnanl_roundtrip_verified,
    })
}
