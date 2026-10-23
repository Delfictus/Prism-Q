//! # M1 typed-producer differential — pure-logic comparison helpers
//!
//! Implements the protocol defined in [`docs/M1_DIFFERENTIAL_PROTOCOL.md`].
//! This module is **data-structures + pure logic only**:
//!
//! - No I/O.
//! - No GPU contact.
//! - No state across calls.
//! - No dependencies on `prism_nhs::*` types.
//!
//! Caller (M1.2.5b wire-in in `nhs_rt_full.rs`) is responsible for:
//! 1. Running the legacy `cluster_spikes()` and the M1
//!    [`SpikeToCluster4D::apply`] side-by-side.
//! 2. Extracting primitives (centroids, volumes, conservation u64s)
//!    from each path's output.
//! 3. Calling [`compute_differential`] to build a per-frame record.
//! 4. Calling [`rollup`] at run end to emit the run-summary object.
//! 5. Serializing both into `binding_sites.json` under the new
//!    top-level key `m1_typed_producer_differential`.
//!
//! The GPU-side helper that runs the M1 producer
//! (`spike_to_cluster_4d::side_channel::run_m1_typed_producer` per
//! Phase C of M1.2.5b) lives outside this module.
//!
//! ## Protocol version
//!
//! Bound to [`PROTOCOL_VERSION`]. Bump on any breaking schema or
//! threshold change. SemVer-aligned with the protocol doc § 8.

use serde::{Deserialize, Serialize};

/// SemVer of the differential protocol shipped in this module.
/// Synchronised with `docs/M1_DIFFERENTIAL_PROTOCOL.md` § 8.
pub const PROTOCOL_VERSION: &str = "1.0.0";

// ============================================================================
// AnchorSite — which Phase A1 anchor context emitted the record
// ============================================================================

/// The three Phase-A1 anchor contexts in `nhs_rt_full.rs` that emit a
/// per-frame differential record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnchorSite {
    /// Main MD loop.
    Main,
    /// Per-replica loop body.
    Replica,
    /// Per-stream parallel-replica context.
    PerStream,
}

// ============================================================================
// AgreementClass — tagged classification of every per-frame record
// ============================================================================

/// Per-record classification produced by [`compute_differential`].
///
/// Serialised as a tagged object (see protocol doc § 3.1):
/// - `{"kind": "StrictMatch"}`
/// - `{"kind": "BenignDivergence"}`
/// - `{"kind": "BlockingDivergence", "reason": "<tag>"}`
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum AgreementClass {
    /// All metrics agreed exactly (integer BitExact + tolerance metrics
    /// below 1× tolerance). Expected for >99.5% of frames.
    StrictMatch,
    /// Tolerance metrics drifted into the `[1×, 2×)` band but no integer
    /// disagreement and no metric exceeded 2× tolerance.
    BenignDivergence,
    /// At least one of: integer-metric disagreement, or tolerance
    /// metric exceeded 2× tolerance.
    BlockingDivergence {
        /// Short reason tag — see protocol doc § 2.1 for the closed set.
        reason: String,
    },
}

impl AgreementClass {
    /// True iff this is a `BlockingDivergence`. Helper for hot-path
    /// counters.
    #[inline]
    pub fn is_blocking(&self) -> bool {
        matches!(self, Self::BlockingDivergence { .. })
    }

    /// Convenience constructor for the closed set of blocking reasons.
    pub fn blocking(reason: &str) -> Self {
        Self::BlockingDivergence {
            reason: reason.to_string(),
        }
    }
}

// ============================================================================
// DifferentialTolerance — protocol § 5
// ============================================================================

/// Per-metric tolerance band used by [`compute_differential`].
///
/// Use [`DifferentialTolerance::CANONICAL`] for V4 PASS gating; an
/// operator-overridable variant is reserved for future debugging
/// (NOT in M1.2.5b scope).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DifferentialTolerance {
    /// Per-cluster centroid drift, Å. BlockingDivergence at 2× this.
    pub centroid_max_drift_ang: f32,
    /// Per-cluster AABB volume relative drift (unitless ratio).
    /// BlockingDivergence at 2× this.
    pub aabb_volume_max_relative: f32,
}

impl DifferentialTolerance {
    /// Frozen V4-PASS-gating tolerances. Protocol doc § 5.
    pub const CANONICAL: Self = Self {
        centroid_max_drift_ang: 1.0e-3,
        aabb_volume_max_relative: 0.01,
    };
}

// ============================================================================
// Snapshots — per-path inputs to compute_differential
// ============================================================================

/// Legacy `cluster_spikes()` half of the per-frame record.
///
/// Caller extracts these fields from the `Vec<ClusteredBindingSite>`
/// already in scope at the wire-in site. Centroid source is the
/// site's `legacy_emission_centroid` (module-private to
/// `persistent_engine` — caller passes via whatever shim is local;
/// `ClusteredBindingSite::view().geometric()` is the public path).
/// Volume = `bounding_box.x * bounding_box.y * bounding_box.z`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LegacySnapshot {
    pub num_clusters: u32,
    pub num_clusters_attributed: u32,
    pub total_attributed: u64,
    pub background_count: u64,
    pub total_input_spikes: u64,
    pub cluster_centroids_ang: Vec<[f32; 3]>,
    pub cluster_aabb_volumes_ang3: Vec<f32>,
}

impl LegacySnapshot {
    /// Construct from pre-extracted vectors. Caller is responsible for
    /// extracting cluster_id-ordered centroids and bbox volumes from
    /// the local `Vec<ClusteredBindingSite>`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_extracted(
        num_clusters: u32,
        num_clusters_attributed: u32,
        total_attributed: u64,
        background_count: u64,
        total_input_spikes: u64,
        cluster_centroids_ang: Vec<[f32; 3]>,
        cluster_aabb_volumes_ang3: Vec<f32>,
    ) -> Self {
        Self {
            num_clusters,
            num_clusters_attributed,
            total_attributed,
            background_count,
            total_input_spikes,
            cluster_centroids_ang,
            cluster_aabb_volumes_ang3,
        }
    }
}

/// M1 [`SpikeToCluster4D::apply`] half of the per-frame record.
///
/// Caller extracts these fields from `SpikeToCluster4DOutput`
/// (`crates/prism-nhs/src/spike_to_cluster_4d.rs:138`) and
/// `EntangledManifold` (`entangled_manifold.rs:726`) in scope after
/// the side-channel helper returns.
///
/// **At M1.2.5a placeholder shape**, the three view AABB volumes are
/// equal (one synthetic support set is shared across driver/lining/
/// localized). M3 specialization will diverge them; the protocol
/// emits all three so the M3 transition is visible in the historical
/// record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct M1TypedSnapshot {
    pub num_clusters_attributed: u32,
    pub total_attributed: u64,
    pub background_count: u64,
    pub total_input_spikes: u64,
    pub manifold_frame: u64,
    pub cluster_centroids_ang: Vec<[f32; 3]>,
    pub cluster_aabb_volumes_ang3: Vec<f32>,
    pub driver_view_aabb_volume_ang3: f32,
    pub lining_view_aabb_volume_ang3: f32,
    pub localized_view_aabb_volume_ang3: f32,
}

impl M1TypedSnapshot {
    /// Construct from pre-extracted M1 output components.
    ///
    /// Helper for callers: per-cluster AABB volume = `extent[0] *
    /// extent[1] * extent[2]` where `extent = aabb.max[i] - aabb.min[i]`.
    /// Per-view AABB volume = same formula on the view's
    /// `ManifoldViewData::aabb` (`entangled_manifold.rs:558`).
    #[allow(clippy::too_many_arguments)]
    pub fn from_components(
        num_clusters_attributed: u32,
        total_attributed: u64,
        background_count: u64,
        total_input_spikes: u64,
        manifold_frame: u64,
        cluster_centroids_ang: Vec<[f32; 3]>,
        cluster_aabb_volumes_ang3: Vec<f32>,
        driver_view_aabb_volume_ang3: f32,
        lining_view_aabb_volume_ang3: f32,
        localized_view_aabb_volume_ang3: f32,
    ) -> Self {
        Self {
            num_clusters_attributed,
            total_attributed,
            background_count,
            total_input_spikes,
            manifold_frame,
            cluster_centroids_ang,
            cluster_aabb_volumes_ang3,
            driver_view_aabb_volume_ang3,
            lining_view_aabb_volume_ang3,
            localized_view_aabb_volume_ang3,
        }
    }
}

// ============================================================================
// DifferentialDeltas — the computed pairwise differences
// ============================================================================

/// Per-frame computed deltas. Protocol doc § 3.1 "deltas" sub-object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DifferentialDeltas {
    pub total_attributed_delta: i64,
    pub background_count_delta: i64,
    pub num_clusters_delta: i32,
    pub centroid_max_drift_ang: f32,
    pub aabb_volume_max_relative_drift: f32,
    /// Deferred to M3 (role-view specialization). `None` at M1.2.5b.
    pub support_set_overlap_fraction: Option<f32>,
}

// ============================================================================
// M1Differential — top-level per-frame record
// ============================================================================

/// One per-frame record. Caller builds via [`compute_differential`]
/// and accumulates into a `Vec<M1Differential>` for the run.
///
/// Serialises to one JSON object matching protocol doc § 3.1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct M1Differential {
    pub frame: u64,
    pub stream_id: u32,
    pub anchor_site: AnchorSite,
    pub agreement_class: AgreementClass,
    pub legacy: LegacySnapshot,
    pub m1_typed: M1TypedSnapshot,
    pub deltas: DifferentialDeltas,
}

// ============================================================================
// compute_differential — the per-frame entry point
// ============================================================================

/// Build one per-frame [`M1Differential`] from the legacy and M1
/// snapshots. Pure function: no I/O, no GPU contact, no state.
///
/// Implements the protocol doc § 1 metric definitions and § 2
/// agreement-class promotion ladder.
pub fn compute_differential(
    frame: u64,
    stream_id: u32,
    anchor_site: AnchorSite,
    legacy: LegacySnapshot,
    m1_typed: M1TypedSnapshot,
    tolerance: &DifferentialTolerance,
) -> M1Differential {
    // ---- Integer metrics 1-3 (BitExact) ---------------------------------
    let total_attributed_delta = m1_typed.total_attributed as i64 - legacy.total_attributed as i64;
    let background_count_delta = m1_typed.background_count as i64 - legacy.background_count as i64;
    let num_clusters_delta =
        m1_typed.num_clusters_attributed as i32 - legacy.num_clusters_attributed as i32;

    // ---- Metric 4: per-cluster centroid drift (max L2) ------------------
    let n_pairs = legacy
        .cluster_centroids_ang
        .len()
        .min(m1_typed.cluster_centroids_ang.len());
    let mut centroid_max_drift_ang: f32 = 0.0;
    for i in 0..n_pairs {
        let l = legacy.cluster_centroids_ang[i];
        let m = m1_typed.cluster_centroids_ang[i];
        let dx = l[0] - m[0];
        let dy = l[1] - m[1];
        let dz = l[2] - m[2];
        let d = (dx * dx + dy * dy + dz * dz).sqrt();
        if d > centroid_max_drift_ang {
            centroid_max_drift_ang = d;
        }
    }

    // ---- Metric 5: per-cluster AABB volume relative drift (max) ---------
    let n_vols = legacy
        .cluster_aabb_volumes_ang3
        .len()
        .min(m1_typed.cluster_aabb_volumes_ang3.len());
    let mut aabb_volume_max_relative_drift: f32 = 0.0;
    for i in 0..n_vols {
        let l = legacy.cluster_aabb_volumes_ang3[i];
        let m = m1_typed.cluster_aabb_volumes_ang3[i];
        let denom = l.max(m).max(f32::EPSILON);
        let rel = (l - m).abs() / denom;
        if rel > aabb_volume_max_relative_drift {
            aabb_volume_max_relative_drift = rel;
        }
    }

    let deltas = DifferentialDeltas {
        total_attributed_delta,
        background_count_delta,
        num_clusters_delta,
        centroid_max_drift_ang,
        aabb_volume_max_relative_drift,
        // Metric 6 deferred to M3 — see protocol doc § 1 caveat.
        support_set_overlap_fraction: None,
    };

    let agreement_class = classify(&deltas, tolerance);

    M1Differential {
        frame,
        stream_id,
        anchor_site,
        agreement_class,
        legacy,
        m1_typed,
        deltas,
    }
}

/// Promotion ladder for [`AgreementClass`]. First match wins; protocol
/// doc § 2.
fn classify(deltas: &DifferentialDeltas, tolerance: &DifferentialTolerance) -> AgreementClass {
    // --- BlockingDivergence (any one) -----------------------------------
    if deltas.total_attributed_delta != 0 {
        return AgreementClass::blocking("total_attributed_mismatch");
    }
    if deltas.background_count_delta != 0 {
        return AgreementClass::blocking("background_count_mismatch");
    }
    if deltas.num_clusters_delta != 0 {
        return AgreementClass::blocking("num_clusters_mismatch");
    }
    let blocking_centroid = 2.0 * tolerance.centroid_max_drift_ang;
    if deltas.centroid_max_drift_ang >= blocking_centroid {
        return AgreementClass::blocking("centroid_drift_exceeds_2x");
    }
    let blocking_volume = 2.0 * tolerance.aabb_volume_max_relative;
    if deltas.aabb_volume_max_relative_drift >= blocking_volume {
        return AgreementClass::blocking("aabb_volume_drift_exceeds_2x");
    }

    // --- BenignDivergence (any one in [1x, 2x) band) --------------------
    if deltas.centroid_max_drift_ang >= tolerance.centroid_max_drift_ang {
        return AgreementClass::BenignDivergence;
    }
    if deltas.aabb_volume_max_relative_drift >= tolerance.aabb_volume_max_relative {
        return AgreementClass::BenignDivergence;
    }

    // --- StrictMatch ----------------------------------------------------
    AgreementClass::StrictMatch
}

// ============================================================================
// Run-end summary — DifferentialRollup + helpers
// ============================================================================

/// Counts of each [`AgreementClass`] over the run. Protocol doc § 3.2.
///
/// JSON keys are PascalCase to match the doc and to make filtering by
/// downstream Python (`summary.agreement_class_counts.BlockingDivergence`)
/// readable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgreementCounts {
    #[serde(rename = "StrictMatch")]
    pub strict_match: u64,
    #[serde(rename = "BenignDivergence")]
    pub benign_divergence: u64,
    #[serde(rename = "BlockingDivergence")]
    pub blocking_divergence: u64,
}

/// Pointer to a single blocking-divergence frame, included verbatim
/// in the run summary so operators can re-run forensically.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockingFrameRef {
    pub frame: u64,
    pub stream_id: u32,
    pub anchor_site: AnchorSite,
    pub reason: String,
}

/// Worst-case observed values across the run. Protocol doc § 3.2.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricsExtrema {
    pub centroid_max_drift_ang_observed: f32,
    pub aabb_volume_max_relative_drift_observed: f32,
}

/// Run-end summary. One per run. Protocol doc § 3.2.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DifferentialRollup {
    pub total_frames_compared: u64,
    pub total_invocations: u64,
    pub agreement_class_counts: AgreementCounts,
    pub blocking_divergence_frames: Vec<BlockingFrameRef>,
    pub metrics_extrema: MetricsExtrema,
    pub wall_time_ms_legacy: f64,
    pub wall_time_ms_m1: f64,
    pub wall_time_overhead_pct: f64,
    pub protocol_version: String,
}

/// Aggregate a `Vec<M1Differential>` into the run summary. Pure
/// function; deterministic.
///
/// `wall_time_overhead_pct` = `100 * wall_time_ms_m1 / wall_time_ms_legacy`,
/// or `0.0` if the legacy time is zero (degenerate case).
pub fn rollup(
    records: &[M1Differential],
    wall_time_legacy_ms: f64,
    wall_time_m1_ms: f64,
) -> DifferentialRollup {
    let mut counts = AgreementCounts {
        strict_match: 0,
        benign_divergence: 0,
        blocking_divergence: 0,
    };
    let mut blocking_frames: Vec<BlockingFrameRef> = Vec::new();
    let mut max_centroid_drift: f32 = 0.0;
    let mut max_volume_drift: f32 = 0.0;

    for r in records {
        match &r.agreement_class {
            AgreementClass::StrictMatch => counts.strict_match += 1,
            AgreementClass::BenignDivergence => counts.benign_divergence += 1,
            AgreementClass::BlockingDivergence { reason } => {
                counts.blocking_divergence += 1;
                blocking_frames.push(BlockingFrameRef {
                    frame: r.frame,
                    stream_id: r.stream_id,
                    anchor_site: r.anchor_site,
                    reason: reason.clone(),
                });
            }
        }
        if r.deltas.centroid_max_drift_ang > max_centroid_drift {
            max_centroid_drift = r.deltas.centroid_max_drift_ang;
        }
        if r.deltas.aabb_volume_max_relative_drift > max_volume_drift {
            max_volume_drift = r.deltas.aabb_volume_max_relative_drift;
        }
    }

    let total_invocations = records.len() as u64;
    let total_frames_compared = if records.is_empty() {
        0
    } else {
        // Unique frame ids — deterministic via BTreeSet.
        let mut seen: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
        for r in records {
            seen.insert(r.frame);
        }
        seen.len() as u64
    };

    let wall_time_overhead_pct = if wall_time_legacy_ms > 0.0 {
        (wall_time_m1_ms / wall_time_legacy_ms) * 100.0
    } else {
        0.0
    };

    DifferentialRollup {
        total_frames_compared,
        total_invocations,
        agreement_class_counts: counts,
        blocking_divergence_frames: blocking_frames,
        metrics_extrema: MetricsExtrema {
            centroid_max_drift_ang_observed: max_centroid_drift,
            aabb_volume_max_relative_drift_observed: max_volume_drift,
        },
        wall_time_ms_legacy: wall_time_legacy_ms,
        wall_time_ms_m1: wall_time_m1_ms,
        wall_time_overhead_pct,
        protocol_version: PROTOCOL_VERSION.to_string(),
    }
}

// ============================================================================
// Tests — cover every classify() branch + rollup aggregation
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn legacy_baseline() -> LegacySnapshot {
        LegacySnapshot::from_extracted(7, 7, 80, 12, 92, vec![[1.0, 2.0, 3.0]; 7], vec![3.0; 7])
    }

    fn m1_baseline() -> M1TypedSnapshot {
        M1TypedSnapshot::from_components(
            7,
            80,
            12,
            92,
            100,
            vec![[1.0, 2.0, 3.0]; 7],
            vec![3.0; 7],
            14.7,
            14.7,
            14.7,
        )
    }

    #[test]
    fn protocol_version_is_canonical_baseline() {
        assert_eq!(PROTOCOL_VERSION, "1.0.0");
    }

    #[test]
    fn canonical_tolerance_constants() {
        let t = DifferentialTolerance::CANONICAL;
        assert_eq!(t.centroid_max_drift_ang, 1.0e-3);
        assert_eq!(t.aabb_volume_max_relative, 0.01);
    }

    #[test]
    fn strict_match_when_all_metrics_agree() {
        let r = compute_differential(
            100,
            0,
            AnchorSite::Main,
            legacy_baseline(),
            m1_baseline(),
            &DifferentialTolerance::CANONICAL,
        );
        assert_eq!(r.agreement_class, AgreementClass::StrictMatch);
        assert_eq!(r.deltas.total_attributed_delta, 0);
        assert_eq!(r.deltas.background_count_delta, 0);
        assert_eq!(r.deltas.num_clusters_delta, 0);
        assert!(r.deltas.centroid_max_drift_ang < 1e-9);
        assert!(r.deltas.aabb_volume_max_relative_drift < 1e-9);
    }

    #[test]
    fn blocking_on_total_attributed_mismatch() {
        let mut m1 = m1_baseline();
        m1.total_attributed = 81;
        let r = compute_differential(
            100,
            0,
            AnchorSite::Main,
            legacy_baseline(),
            m1,
            &DifferentialTolerance::CANONICAL,
        );
        assert!(r.agreement_class.is_blocking());
        match r.agreement_class {
            AgreementClass::BlockingDivergence { reason } => {
                assert_eq!(reason, "total_attributed_mismatch");
            }
            _ => panic!("expected BlockingDivergence"),
        }
    }

    #[test]
    fn blocking_on_background_mismatch() {
        let mut m1 = m1_baseline();
        m1.background_count = 13;
        let r = compute_differential(
            100,
            0,
            AnchorSite::Main,
            legacy_baseline(),
            m1,
            &DifferentialTolerance::CANONICAL,
        );
        match r.agreement_class {
            AgreementClass::BlockingDivergence { reason } => {
                assert_eq!(reason, "background_count_mismatch");
            }
            _ => panic!("expected BlockingDivergence"),
        }
    }

    #[test]
    fn blocking_on_num_clusters_mismatch() {
        let mut m1 = m1_baseline();
        m1.num_clusters_attributed = 6;
        let r = compute_differential(
            100,
            0,
            AnchorSite::Main,
            legacy_baseline(),
            m1,
            &DifferentialTolerance::CANONICAL,
        );
        match r.agreement_class {
            AgreementClass::BlockingDivergence { reason } => {
                assert_eq!(reason, "num_clusters_mismatch");
            }
            _ => panic!("expected BlockingDivergence"),
        }
    }

    #[test]
    fn benign_centroid_drift_at_1_5x_tolerance() {
        let mut m1 = m1_baseline();
        // 1.5e-3 Å drift along x — between 1× and 2× of 1e-3 Å
        m1.cluster_centroids_ang = vec![[1.0 + 1.5e-3, 2.0, 3.0]; 7];
        let r = compute_differential(
            100,
            0,
            AnchorSite::Main,
            legacy_baseline(),
            m1,
            &DifferentialTolerance::CANONICAL,
        );
        assert_eq!(r.agreement_class, AgreementClass::BenignDivergence);
    }

    #[test]
    fn blocking_centroid_drift_at_2_5x_tolerance() {
        let mut m1 = m1_baseline();
        // 2.5e-3 Å drift — above 2× tolerance
        m1.cluster_centroids_ang = vec![[1.0 + 2.5e-3, 2.0, 3.0]; 7];
        let r = compute_differential(
            100,
            0,
            AnchorSite::Main,
            legacy_baseline(),
            m1,
            &DifferentialTolerance::CANONICAL,
        );
        match r.agreement_class {
            AgreementClass::BlockingDivergence { reason } => {
                assert_eq!(reason, "centroid_drift_exceeds_2x");
            }
            _ => panic!("expected BlockingDivergence"),
        }
    }

    #[test]
    fn benign_volume_drift_at_1_5_pct() {
        let mut m1 = m1_baseline();
        m1.cluster_aabb_volumes_ang3 = vec![3.0 * 1.015; 7]; // 1.5% drift
        let r = compute_differential(
            100,
            0,
            AnchorSite::Main,
            legacy_baseline(),
            m1,
            &DifferentialTolerance::CANONICAL,
        );
        assert_eq!(r.agreement_class, AgreementClass::BenignDivergence);
    }

    #[test]
    fn blocking_volume_drift_at_2_5_pct() {
        let mut m1 = m1_baseline();
        m1.cluster_aabb_volumes_ang3 = vec![3.0 * 1.025; 7]; // 2.5% drift
        let r = compute_differential(
            100,
            0,
            AnchorSite::Main,
            legacy_baseline(),
            m1,
            &DifferentialTolerance::CANONICAL,
        );
        match r.agreement_class {
            AgreementClass::BlockingDivergence { reason } => {
                assert_eq!(reason, "aabb_volume_drift_exceeds_2x");
            }
            _ => panic!("expected BlockingDivergence"),
        }
    }

    #[test]
    fn classify_priority_total_attrib_over_centroid() {
        // Both integer-mismatch AND centroid-drift trigger; the ladder
        // returns the integer one (it's checked first).
        let mut m1 = m1_baseline();
        m1.total_attributed = 81;
        m1.cluster_centroids_ang = vec![[1.0 + 1e-2, 2.0, 3.0]; 7];
        let r = compute_differential(
            100,
            0,
            AnchorSite::Main,
            legacy_baseline(),
            m1,
            &DifferentialTolerance::CANONICAL,
        );
        match r.agreement_class {
            AgreementClass::BlockingDivergence { reason } => {
                assert_eq!(reason, "total_attributed_mismatch");
            }
            _ => panic!("expected BlockingDivergence"),
        }
    }

    #[test]
    fn rollup_aggregates_counts_extrema_and_overhead() {
        let mut records = Vec::new();
        // 10 strict matches across distinct frames
        for f in 0..10 {
            records.push(compute_differential(
                f,
                0,
                AnchorSite::Main,
                legacy_baseline(),
                m1_baseline(),
                &DifferentialTolerance::CANONICAL,
            ));
        }
        // 1 benign (centroid drift in [1x, 2x) band)
        let mut m1 = m1_baseline();
        m1.cluster_centroids_ang = vec![[1.0 + 1.5e-3, 2.0, 3.0]; 7];
        records.push(compute_differential(
            10,
            0,
            AnchorSite::Main,
            legacy_baseline(),
            m1,
            &DifferentialTolerance::CANONICAL,
        ));
        // 1 blocking
        let mut m1 = m1_baseline();
        m1.total_attributed = 81;
        records.push(compute_differential(
            11,
            1,
            AnchorSite::PerStream,
            legacy_baseline(),
            m1,
            &DifferentialTolerance::CANONICAL,
        ));

        let s = rollup(&records, 1000.0, 100.0);
        assert_eq!(s.total_invocations, 12);
        assert_eq!(s.total_frames_compared, 12);
        assert_eq!(s.agreement_class_counts.strict_match, 10);
        assert_eq!(s.agreement_class_counts.benign_divergence, 1);
        assert_eq!(s.agreement_class_counts.blocking_divergence, 1);
        assert_eq!(s.blocking_divergence_frames.len(), 1);
        assert_eq!(s.blocking_divergence_frames[0].frame, 11);
        assert_eq!(s.blocking_divergence_frames[0].stream_id, 1);
        assert_eq!(
            s.blocking_divergence_frames[0].anchor_site,
            AnchorSite::PerStream
        );
        assert_eq!(
            s.blocking_divergence_frames[0].reason,
            "total_attributed_mismatch"
        );
        // 100 / 1000 = 10%
        assert!((s.wall_time_overhead_pct - 10.0).abs() < 1e-6);
        // The benign run pushes max centroid drift to ≥ 1.5e-3
        assert!(s.metrics_extrema.centroid_max_drift_ang_observed >= 1.5e-3 - 1e-9);
        assert_eq!(s.protocol_version, PROTOCOL_VERSION);
    }

    #[test]
    fn rollup_handles_empty() {
        let s = rollup(&[], 0.0, 0.0);
        assert_eq!(s.total_invocations, 0);
        assert_eq!(s.total_frames_compared, 0);
        assert_eq!(s.agreement_class_counts.strict_match, 0);
        assert_eq!(s.wall_time_overhead_pct, 0.0);
    }

    #[test]
    fn anchor_site_serialises_snake_case() {
        let json = serde_json::to_string(&AnchorSite::PerStream).unwrap();
        assert_eq!(json, "\"per_stream\"");
        let json = serde_json::to_string(&AnchorSite::Main).unwrap();
        assert_eq!(json, "\"main\"");
    }

    #[test]
    fn agreement_class_serialises_tagged() {
        let s = serde_json::to_string(&AgreementClass::StrictMatch).unwrap();
        assert_eq!(s, r#"{"kind":"StrictMatch"}"#);
        let s = serde_json::to_string(&AgreementClass::BenignDivergence).unwrap();
        assert_eq!(s, r#"{"kind":"BenignDivergence"}"#);
        let s = serde_json::to_string(&AgreementClass::blocking("foo_bar")).unwrap();
        assert_eq!(s, r#"{"kind":"BlockingDivergence","reason":"foo_bar"}"#);
    }

    #[test]
    fn differential_round_trips_serde() {
        let r = compute_differential(
            42,
            7,
            AnchorSite::Replica,
            legacy_baseline(),
            m1_baseline(),
            &DifferentialTolerance::CANONICAL,
        );
        let json = serde_json::to_string(&r).unwrap();
        let r2: M1Differential = serde_json::from_str(&json).unwrap();
        assert_eq!(r, r2);
    }
}
