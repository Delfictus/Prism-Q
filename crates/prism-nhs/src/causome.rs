//! Phase 3 — Pillar 2: Causome typed block.
//!
//! `CausomeBlock` is the typed projection of the per-site causal data
//! that is currently scattered across `kcc_visualization.json.residues[]`
//! + `kcc_visualization.json.sites[].kcc{}` + `kcc_validation.json.sites[]`
//! + `binding_sites.json sites[i].kcc{}` + `binding_sites.json
//! sites[i].signal_preservation{}` + `gcpid_synergy.json residues[]`.
//!
//! The struct itself is pure data — no behavior, no methods that
//! reach into other Pillars.  The `cluster_to_causome` transform is
//! the only canonical producer; the JSON-emit code is the only
//! canonical consumer.  Future Phase 4 work (`CausomeToTherm`,
//! `CausomeToRankTerms`) will consume this object as `Gpre` for their
//! own typed outputs.
//!
//! # What this module does NOT do
//!
//! * Compute raw KCC measurements — those are computed upstream in the
//!   KCC analysis kernels. The shared residue-confidence reduction below
//!   reuses the live emitter arithmetic for native offline ranking.
//! * Hold tolerance / determinism state — those live on the
//!   `ClusterToCausome` transform impl per the audit-spine doctrine.
//! * Emit JSON directly — the canonical emitter in `nhs_rt_full.rs`
//!   serializes via serde.
//!
//! # Reserved fields (Phase 4–5)
//!
//! `tide_decomposition: Vec<TideResidueDecomp>` is a documented
//! reserved slot — already emitted in some run modes
//! (`prism_therm.sites[].tide_decomposition[]` per the schema-walker
//! reference) but belongs to Pillar 3 (`ThermBlock`) once
//! `CausomeToTherm` lands.  Phase 3 does not introduce the field
//! here; the comment is the schema reservation marker.

use serde::{Deserialize, Serialize};

/// Additive provenance for an existing KCC lag measurement. The original
/// PRKCC numerical payload remains unchanged; absent legacy provenance is unknown.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KccLagProvenance {
    pub estimator_origin: String,
    pub availability: String,
    pub lag_definition: String,
    pub lag_unit: String,
    pub sign_convention: String,
    pub observation_count: u32,
    pub active_causal_count: u32,
    pub ring_event_count: Option<u32>,
    pub ring_first_step: Option<u32>,
    pub ring_last_step: Option<u32>,
    pub winning_pair_count: Option<u32>,
    pub winning_event_index_offset: Option<i32>,
    pub pairing_tolerance_md_steps: Option<u32>,
    pub source_stream_id: Option<u32>,
    pub causal_lag_f32_bits: Option<u32>,
    pub lag_corr_peak_f32_bits: Option<u32>,
    pub physical_time_mapping: String,
}
impl KccLagProvenance {
    pub fn legacy_unknown(observation_count: u32, active_causal_count: u32) -> Self {
        Self {
            estimator_origin: "legacy_unknown".into(),
            availability: "origin_not_recorded".into(),
            lag_definition: "unknown_legacy_GPU_grid_or_host_actual_mean_pair_separation".into(),
            lag_unit: "MD_step".into(),
            sign_convention: "unknown_between_requested_grid_and_actual_mean_pair_separation; not_interresidue_influence".into(),
            observation_count, active_causal_count,
            ring_event_count: None, ring_first_step: None, ring_last_step: None,
            winning_pair_count: None, winning_event_index_offset: None,
            pairing_tolerance_md_steps: None, source_stream_id: None,
            causal_lag_f32_bits: None, lag_corr_peak_f32_bits: None,
            physical_time_mapping: "requires_recorded_MD_step_time_mapping; no_constant_dt_assumed".into(),
        }
    }
}

/// Exact support archive for a KCC lag measurement. This is additive evidence:
/// the PRKCC001 scalar payload remains the ABI and these rows explain which
/// downloaded event-ring records supported the emitted lag values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KccLagSupportArchive {
    pub schema_kind: String,
    pub schema_version: u32,
    pub run_id: String,
    pub stem: String,
    pub stream_id: u32,
    pub measurement_step: Option<u64>,
    pub source_kind: String,
    pub kcc_v2full_sha256: Option<String>,
    pub residue_identity: String,
    pub source_response_semantics: String,
    pub ring_event_schema: String,
    pub ring_event_flags_semantics: String,
    pub gpu_pairing_semantics: String,
    pub host_pairing_semantics: String,
    pub rows: Vec<KccLagSupportResidue>,
}

impl KccLagSupportArchive {
    pub fn new(
        run_id: &str,
        stem: &str,
        stream_id: u32,
        measurement_step: Option<u64>,
        source_kind: &str,
        kcc_v2full_sha256: Option<String>,
        rows: Vec<KccLagSupportResidue>,
    ) -> Self {
        Self {
            schema_kind: "prism.kcc_lag_support.v1".into(),
            schema_version: 1,
            run_id: run_id.into(),
            stem: stem.into(),
            stream_id,
            measurement_step,
            source_kind: source_kind.into(),
            kcc_v2full_sha256,
            residue_identity: "zero_based_PRKCC_descriptor_index".into(),
            source_response_semantics:
                "same_residue_causal_count_event_to_same_residue_motion_response_event; c_count_is_KCC_counter_aggregate_clamped_to_u16_not_exact_UV_actuator_event_ordinal; measurement_association_not_interresidue_causal_effect"
                    .into(),
            ring_event_schema:
                "KCCEvent{dx:f32,dy:f32,dz:f32,c:u16,flags:u16,step:u32}; 64 raw slots per residue from downloaded device ring"
                    .into(),
            ring_event_flags_semantics:
                "raw_KCCEvent.flags preserved; unmarked legacy: bit0=current_phase0 cold, bit1=current_phase2 warm; bit15 marked device-protocol endpoint v2: bit0=cold_hold, bit1=warm_hold, bit2=heating, bit3=cooling, bit4=cold_return; warm-tail flag retains warm_hold; phase is endpoint-only, not UV actuator identity or a label for every accumulated interval count"
                    .into(),
            gpu_pairing_semantics:
                "pairs_reconstructed_only_for_emitted_GPU_requested_lag; no_CPU_reselection_of_winning_lag; nearest_other_event_with_abs_step_error_le_5_and_strict_first_tie"
                    .into(),
            host_pairing_semantics:
                "pairs_from_emitted_host_winning_event_index_offset; lag_is_mean_actual_response_step_minus_causal_step"
                    .into(),
            rows,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KccF32Evidence {
    pub value: Option<f32>,
    pub classification: String,
    pub ieee754_bits_hex: String,
    pub ieee754_bits: u32,
}

impl KccF32Evidence {
    pub fn from_f32(value: f32) -> Self {
        let classification = if value.is_nan() {
            "nan"
        } else if value == f32::INFINITY {
            "positive_infinity"
        } else if value == f32::NEG_INFINITY {
            "negative_infinity"
        } else {
            "finite"
        };
        Self {
            value: value.is_finite().then_some(value),
            classification: classification.into(),
            ieee754_bits_hex: format!("{:08x}", value.to_bits()),
            ieee754_bits: value.to_bits(),
        }
    }

    pub fn from_bits(bits: u32) -> Self {
        Self::from_f32(f32::from_bits(bits))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KccLagSupportResidue {
    pub residue_index: u32,
    pub source_stream_id: Option<u32>,
    pub measurement_step: Option<u64>,
    pub observation_count: u32,
    pub active_causal_count: u32,
    pub estimator_origin: String,
    pub availability: String,
    pub lag_definition: String,
    pub lag_unit: String,
    pub sign_convention: String,
    pub ring_head: Option<u32>,
    pub ring_event_count: Option<u32>,
    pub ring_raw_slots_hex: Option<String>,
    pub ordered_events: Vec<KccLagRingEvent>,
    pub final_causal_lag_f32_bits: u32,
    pub final_lag_corr_peak_f32_bits: u32,
    pub gpu_pre_fallback_causal_lag_f32_bits: Option<u32>,
    pub gpu_pre_fallback_lag_corr_peak_f32_bits: Option<u32>,
    pub gpu_requested_lag_md_steps: Option<i32>,
    pub host_winning_event_index_offset: Option<i32>,
    pub pairing_tolerance_md_steps: Option<u32>,
    pub winning_pair_count: u32,
    pub winning_pairs: Vec<KccLagWinningPair>,
    pub recomputed_support: Option<KccLagSupportArithmetic>,
}

impl KccLagSupportResidue {
    pub fn legacy_unknown(
        residue_index: u32,
        observation_count: u32,
        active_causal_count: u32,
        final_causal_lag_f32_bits: u32,
        final_lag_corr_peak_f32_bits: u32,
    ) -> Self {
        Self {
            residue_index,
            source_stream_id: None,
            measurement_step: None,
            observation_count,
            active_causal_count,
            estimator_origin: "legacy_unknown".into(),
            availability: "support_not_recorded".into(),
            lag_definition: "unknown_legacy_GPU_grid_or_host_actual_mean_pair_separation".into(),
            lag_unit: "MD_step".into(),
            sign_convention:
                "unknown_between_requested_grid_and_actual_mean_pair_separation; not_interresidue_influence"
                    .into(),
            ring_head: None,
            ring_event_count: None,
            ring_raw_slots_hex: None,
            ordered_events: Vec::new(),
            final_causal_lag_f32_bits,
            final_lag_corr_peak_f32_bits,
            gpu_pre_fallback_causal_lag_f32_bits: None,
            gpu_pre_fallback_lag_corr_peak_f32_bits: None,
            gpu_requested_lag_md_steps: None,
            host_winning_event_index_offset: None,
            pairing_tolerance_md_steps: None,
            winning_pair_count: 0,
            winning_pairs: Vec::new(),
            recomputed_support: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KccLagRingEvent {
    pub ordered_index: u32,
    pub ring_slot: u32,
    pub step: u32,
    pub dx: KccF32Evidence,
    pub dy: KccF32Evidence,
    pub dz: KccF32Evidence,
    pub motion_magnitude: KccF32Evidence,
    pub causal_count: u16,
    pub flags_raw: u16,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KccLagWinningPair {
    pub causal_event_ordered_index: u32,
    pub response_event_ordered_index: u32,
    pub causal_ring_slot: u32,
    pub response_ring_slot: u32,
    pub causal_step: u32,
    pub response_step: u32,
    pub actual_step_delta_response_minus_causal: i64,
    pub causal_count: u16,
    pub causal_flags_raw: u16,
    pub response_dx: KccF32Evidence,
    pub response_dy: KccF32Evidence,
    pub response_dz: KccF32Evidence,
    pub response_motion_magnitude: KccF32Evidence,
    pub response_flags_raw: u16,
    pub requested_gpu_lag_md_steps: Option<i32>,
    pub host_event_index_offset: Option<i32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KccLagSupportArithmetic {
    pub pair_count: u32,
    pub sum_causal_times_motion: KccF32Evidence,
    pub sum_causal_squared: KccF32Evidence,
    pub sum_motion_squared: KccF32Evidence,
    pub recomputed_corr_for_recorded_pair_set: Option<KccF32Evidence>,
    pub mean_actual_step_delta_response_minus_causal: Option<KccF32Evidence>,
    pub producer_equivalent_mean_step_delta_f32: Option<KccF32Evidence>,
    pub arithmetic_role: String,
}

/// Typed Pillar-2 block per detected site.  Produced by
/// `crate::transform::cluster_to_causome::ClusterToCausome` from the
/// already-emitted KCC / signal-preservation data, never reconstructed.
///
/// Field semantics map to the existing
/// `binding_sites.json sites[i].kcc{...}` JSON sub-dict (which Phase 3
/// retains for backward compat) — every Phase-2 KCC sub-field has a
/// typed home here, with explicit naming.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CausomeBlock {
    /// Engine cluster id of the site this block describes.
    pub site_id: i32,

    /// The single residue judged most causally responsible for the
    /// site's pocket dynamics.  Must satisfy
    /// `LAW_L3_DRIVER_SUBSET_OF_LINING` —
    /// `driver_residue_id ∈ lining_residues[].resid` for this site.
    pub driver_residue_id: i32,

    /// Top-K candidate driver residues (typically K=3) considered
    /// during the KCC reduction.  Must each satisfy `LAW_L3` (subset
    /// of lining).  Length matches `candidate_residue_support`.
    pub candidate_residue_ids: Vec<i32>,
    /// Per-candidate causal-support weight.  Length matches
    /// `candidate_residue_ids`.  Values are KCC-derived; expected to
    /// satisfy `RelativeEpsilon { eps: 1e-4 }` reproducibility under
    /// CUDA atomics.
    pub candidate_residue_support: Vec<f32>,
    #[serde(default)]
    pub candidate_lag_provenance: Vec<KccLagProvenance>,

    /// KCC sub-scores aggregated to the driver residue.
    pub burst_motion: f32,
    pub causal_lag: f32,
    pub direction_score: f32,
    pub kcc_confidence: f32,
    pub lag_corr_peak: f32,
    pub local_cov: f32,
    pub motion_efficiency: f32,

    /// KCC sub-scores aggregated to the SITE (vs the driver residue).
    /// Corresponds to the `site_*` keys in the v1 `kcc{}` sub-dict.
    pub site_burst_motion: f32,
    pub site_causal_lag: f32,
    pub site_direction_score: f32,
    pub site_lag_corr_peak: f32,
    pub site_local_cov: f32,
    pub site_motion_efficiency: f32,

    /// Cross-correlation peak between motion and causal signal across
    /// the full site neighborhood (vs just the driver residue's
    /// `lag_corr_peak`).
    pub temporal_corr: f32,

    /// Number of simulation steps in which this site's neighborhood
    /// saw causal UV→LIF coupling (the "active" subset of total_steps).
    pub active_causal_steps: u32,
    /// Total simulation steps the site was tracked across.
    pub total_steps: u32,
    // Reserved (Phase 4–5):
    //   pub tide_decomposition: Vec<TideResidueDecomp>,
    // — TIDE per-residue thermodynamic decomposition belongs to
    // Pillar 3 (`ThermBlock`) once `CausomeToTherm` lands.  Phase 3
    // omits this field; the comment is the schema reservation marker.
}

impl CausomeBlock {
    /// Total number of candidate residues recorded.  Matches
    /// `candidate_residue_ids.len()` and `candidate_residue_support.len()`.
    pub fn candidate_count(&self) -> usize {
        debug_assert_eq!(
            self.candidate_residue_ids.len(),
            self.candidate_residue_support.len(),
            "CausomeBlock candidate_residue_ids/support length mismatch"
        );
        self.candidate_residue_ids.len()
    }
}

/// Existing live-engine residue-confidence arithmetic, shared with offline
/// consumers. This is a reduction of recorded KCC measurements, not a causal
/// direction estimator. The caller must identify its support population.
/// No lag, temporal correlation, or direction score participates in this formula.
pub fn native_residue_kcc_confidence(
    motion_efficiency: f32,
    active_causal: u32,
    observation_count: u32,
    burst_motion: f32,
    local_cov: f32,
    support_fraction: f32,
) -> f32 {
    let causal_fraction = active_causal as f32 / observation_count.max(1) as f32;
    let persistent_score = motion_efficiency * causal_fraction;
    let transient_score = burst_motion.max(0.0).min(5.0) / 5.0 * local_cov.max(0.0).min(1.0);
    persistent_score.max(transient_score) * (1.0 + 0.1 * support_fraction)
}

/// Per-residue native measurements retain stream-specific sample counts and
/// nullable unavailable values. Residue indices are zero-based topology indices.
/// This block complements CausomeBlock for offline regions whose identity is not
/// a legacy clustered-site ID; it does not fabricate a ClusterToCausome join.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeKccResidueObservation {
    pub residue_index: u32,
    #[serde(default)]
    pub lag_provenance: Option<KccLagProvenance>,
    pub nearby_residue_support_count: u64,
    pub support_fraction: f32,
    pub observation_count: u32,
    pub active_causal: u32,
    pub temporal_corr: Option<f32>,
    pub direction_score: Option<f32>,
    pub motion_efficiency: Option<f32>,
    pub burst_motion: Option<f32>,
    pub phase_shift: Option<f32>,
    pub causal_lag: Option<f32>,
    pub lag_corr_peak: Option<f32>,
    pub local_cov: Option<f32>,
    pub net_dx: Option<f32>,
    pub net_dy: Option<f32>,
    pub net_dz: Option<f32>,
    pub sum_m: Option<f32>,
    /// Exact bits include unavailable NaN/Inf payloads; absence remains absent.
    pub native_f32_bits: std::collections::BTreeMap<String, String>,
    pub native_residue_confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeKccRegionStream {
    pub stream_id: u32,
    pub region_id: u32,
    pub source_path: String,
    pub source_checksum_fnv1a64: String,
    pub support_semantics: String,
    pub sample_count_semantics: String,
    pub lag_unit: String,
    pub lag_estimator_origin: String,
    pub direction_score_semantics: String,
    pub candidates: Vec<NativeKccResidueObservation>,
    pub selected_residue_index: Option<u32>,
    pub native_residue_confidence: Option<f32>,
    pub status: String,
}
