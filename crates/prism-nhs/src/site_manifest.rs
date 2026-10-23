//! M1.1 — Per-site canonical container types.
//!
//! This module defines the **per-site** Rust types referenced in the
//! M1 execution contract (B1 §4 binding spec) and the M1 readiness
//! report. It is the type-definitions layer of the M1 lane; FFI
//! declarations, `AuditedTransform` impls, and constructors that take
//! [`crate::entangled_manifold::CausalSortKey`] land in M1.2 / M1.3.
//!
//! # Layering
//!
//! Three distinct types at three distinct layers in the M1
//! architecture:
//!
//! * [`crate::entangled_manifold::EntangledManifold`] — *frame-level*,
//!   M5-conformant, 3 typed role-views (`CausalDriverView`,
//!   `LiningContactView`, `LocalizedSubclusterView`) with LBVH-ready
//!   AABBs. Produced on-device by the M1.2 clustering kernel.
//! * [`SiteManifest`] — *per-site* canonical Rust type. One instance
//!   per detected site. Holds the 8-slot [`CentroidManifold`]
//!   projection plus per-site identity, causal scalars, frame index,
//!   an opaque [`EntangledManifoldId`] reference back to the
//!   frame-level manifold, and the [`CausalSortKey`] used to order
//!   the constituent role-view residues
//!   (`source_manifold_id` and `sort_lineage` together provide the
//!   audit trail back to the producer).
//! * [`CentroidManifold`] — *per-site, 8-slot* projection of the
//!   frame-level role views, named-accessor-only, M5-non-conformant
//!   (point centroids, not AABBs) by design. This is the canonical
//!   per-site centroid container; the legacy
//!   [`crate::spatial_view::CentroidManifold`] (3-slot, point centroids,
//!   `Option<[f32; 3]>` shape) is deprecated as of M1.1 with migration
//!   of the two `persistent_engine.rs` callsites scheduled for the
//!   POST-M1 steering wiring lane.
//!
//! # M1 vs later lanes — what is honest at M1
//!
//! At M1, only the producer's geometric / attribution outputs are
//! honest:
//!
//! * `CentroidManifold::geometric()`, `lining()`, `driver()` are
//!   populated from the corresponding frame-level role-view AABBs
//!   (each AABB contributes its center via `Aabb::center()`).
//! * Phase-dependent slots — `hot_phase`, `cold_phase`, `burst_motion`,
//!   `validation_structural`, `ligand_adjacent_subcluster` — are
//!   `Option::None` at M1. M2 / M3 populate them as their producers
//!   land.
//! * [`CausalScalars::spike_attribution_count`] is honest at M1.
//!   The four floating-point causal fields (`kcc_score`,
//!   `transfer_entropy`, `causal_dg`, `causal_lag`) are initialized
//!   to `f64::NAN`. M2 fills `kcc_score`; M3 fills the rest.
//!
//! No code change is required to "turn on" M2 / M3 fields — the data
//! becoming honest (NaN → finite) is the activation mechanism. This
//! aligns with the role-aware composite causal sort's
//! progressive-population semantics
//! (see [`crate::entangled_manifold::CausalSortKey`]).
//!
//! # Frame-index sentinels
//!
//! [`Centroid3D::frame`] is `u64`. The reserved sentinel value
//! `u64::MAX` means "frame not yet set" (rare; only legitimately
//! used during construction before the frame is known). Frame `0`
//! is a real frame and MUST NOT be used as a placeholder.

use serde::{Deserialize, Serialize};

use crate::entangled_manifold::{Aabb, CausalSortKey, ViewProvenance};
use crate::spatial_view::SpatialView;

// ============================================================================
// Identity newtypes
// ============================================================================

/// Engine-assigned site identifier. Stable within a single MD run;
/// not assumed stable across runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SiteId(pub u32);

/// Engine-assigned cluster identifier. Stable within a single MD run.
/// `u32::MAX` is reserved as the sentinel for *unclustered* spikes per
/// the M1 producer convention (see M1 readiness report §4 CUDA bar).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClusterId(pub u32);

impl ClusterId {
    /// Sentinel value indicating "this spike was not assigned to any
    /// cluster" / "this site has no cluster yet". Equals `u32::MAX`.
    pub const UNCLUSTERED: ClusterId = ClusterId(u32::MAX);
}

/// Opaque handle from a [`SiteManifest`] back to its source
/// [`crate::entangled_manifold::EntangledManifold`] in the engine's
/// per-frame state. The engine maintains the
/// `(frame, EntangledManifoldId) → EntangledManifold` mapping; this
/// type is intentionally a thin newtype over `u64` to keep
/// [`SiteManifest`] cheap to clone and to avoid coupling the per-site
/// type to the frame-level manifold's lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EntangledManifoldId(pub u64);

// ============================================================================
// Centroid3D — per-slot 4D-aware centroid value
// ============================================================================

/// Per-slot centroid value with 4D-aware provenance.
///
/// Per the M1 contract (B3): a bare `[f32; 3]` strips both the 4th
/// dimension (frame / time) and the slot provenance (which view
/// produced this centroid). This struct carries both as required
/// fields. Slot-mismatch bugs (e.g. a centroid whose `view ==
/// HotPhase` ending up in the `cold_phase` slot of a
/// [`CentroidManifold`]) are catchable at audit time by comparing
/// `view` to the slot it was placed in.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Centroid3D {
    /// Cartesian position in the simulation coordinate frame.
    pub pos: [f32; 3],
    /// Provenance: which [`SpatialView`] produced this centroid.
    /// Used by audit-time slot-consistency checks.
    pub view: SpatialView,
    /// MD frame index this centroid was constructed at. Sentinel
    /// `u64::MAX` means "frame not yet set" (rare, only valid during
    /// construction); never use `0` as a placeholder — frame `0` is
    /// a real frame.
    pub frame: u64,
}

impl Centroid3D {
    /// Sentinel value for [`Self::frame`] meaning "frame not yet set".
    /// Used only during construction before the frame is known;
    /// see the module-level "Frame-index sentinels" section.
    pub const FRAME_UNSET: u64 = u64::MAX;

    /// Construct a centroid with explicit frame and view provenance.
    pub fn new(pos: [f32; 3], view: SpatialView, frame: u64) -> Self {
        Self { pos, view, frame }
    }
}

// ============================================================================
// CentroidManifold — 8-slot per-site centroid projection
// ============================================================================

/// Per-site, 8-slot centroid projection of the frame-level
/// [`crate::entangled_manifold::EntangledManifold`].
///
/// All slots are `Option<Centroid3D>`. Absence is `None`, never a
/// sentinel coordinate or NaN-vector. Fields are private; access is
/// through named slot accessors (`geometric()`, `lining()`, …) and
/// matching `set_*` methods. There is no bare `.centroid()` method
/// and no `pub` field access — this is enforced at the module
/// boundary by visibility, so a caller that asks for a slot must
/// name it explicitly.
///
/// # The eight slots
///
/// | Slot | Honest at | Source |
/// |---|---|---|
/// | `geometric` | M1 | derived from frame-level `CausalDriverView` AABB center |
/// | `lining` | M1 | derived from frame-level `LiningContactView` AABB center |
/// | `driver` | M1 | derived from frame-level `CausalDriverView` AABB center (subset of `geometric`) |
/// | `hot_phase` | M2 | populated by phase-segregation transform |
/// | `cold_phase` | M2 | populated by phase-segregation transform |
/// | `burst_motion` | M3 | populated by burst-motion / TIDE transform |
/// | `validation_structural` | M2 | populated when a reference structure is supplied |
/// | `ligand_adjacent_subcluster` | M2 | populated when ligand context is supplied |
///
/// # Naming reconciliation against the M1 contract
///
/// The contract used `burst_motion_4d` and `ligand_adjacent`; this
/// module aligns to the existing
/// [`SpatialView`] variants `BurstMotion` and
/// `LigandAdjacentSubcluster` (per non-blocking decision (2) in the
/// B1/B2/B3 resolution turn). Zero rename of existing
/// serialization labels.
///
/// # The dropped ninth slot
///
/// The contract initially listed a `localized` slot then dropped it
/// (per B2 in the resolution turn): the `LocalizedSubclusterView`
/// data is frame-level only; per-site, it is redundant with
/// `driver`. Final shape is exactly 8 slots.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CentroidManifold {
    geometric: Option<Centroid3D>,
    lining: Option<Centroid3D>,
    driver: Option<Centroid3D>,
    hot_phase: Option<Centroid3D>,
    cold_phase: Option<Centroid3D>,
    burst_motion: Option<Centroid3D>,
    validation_structural: Option<Centroid3D>,
    ligand_adjacent_subcluster: Option<Centroid3D>,
}

impl CentroidManifold {
    /// Construct an empty manifold with every slot `None`.
    pub fn new() -> Self {
        Self::default()
    }

    // --- accessors (read) ---

    /// Geometric centroid slot (voxel-mass / intensity²-weighted).
    /// Honest at M1.
    pub fn geometric(&self) -> Option<Centroid3D> {
        self.geometric
    }
    /// Lining-residue centroid slot. Honest at M1.
    pub fn lining(&self) -> Option<Centroid3D> {
        self.lining
    }
    /// Driver-residue centroid slot. Honest at M1.
    pub fn driver(&self) -> Option<Centroid3D> {
        self.driver
    }
    /// Hot-phase centroid slot. `None` at M1; populated by M2.
    pub fn hot_phase(&self) -> Option<Centroid3D> {
        self.hot_phase
    }
    /// Cold-phase centroid slot. `None` at M1; populated by M2.
    pub fn cold_phase(&self) -> Option<Centroid3D> {
        self.cold_phase
    }
    /// Burst-motion (4D-aware) centroid slot. `None` at M1;
    /// populated by M3.
    pub fn burst_motion(&self) -> Option<Centroid3D> {
        self.burst_motion
    }
    /// Structural-validation-aligned centroid slot. `None` at M1;
    /// populated by M2 when a reference structure is supplied.
    pub fn validation_structural(&self) -> Option<Centroid3D> {
        self.validation_structural
    }
    /// Ligand-adjacent sub-cluster centroid slot. `None` at M1;
    /// populated by M2 when ligand context is present.
    pub fn ligand_adjacent_subcluster(&self) -> Option<Centroid3D> {
        self.ligand_adjacent_subcluster
    }

    // --- accessors (write) ---

    /// Populate or overwrite the geometric slot. Caller is
    /// responsible for ensuring `c.view == SpatialView::GeometricVoxelMass`;
    /// audit-time consistency checks land in M1.2 alongside the
    /// `AuditedTransform::verify` impl.
    pub fn set_geometric(&mut self, c: Centroid3D) {
        self.geometric = Some(c);
    }
    /// Populate or overwrite the lining slot.
    pub fn set_lining(&mut self, c: Centroid3D) {
        self.lining = Some(c);
    }
    /// Populate or overwrite the driver slot.
    pub fn set_driver(&mut self, c: Centroid3D) {
        self.driver = Some(c);
    }
    /// Populate or overwrite the hot-phase slot. Typically called
    /// by the M2 phase-segregation transform.
    pub fn set_hot_phase(&mut self, c: Centroid3D) {
        self.hot_phase = Some(c);
    }
    /// Populate or overwrite the cold-phase slot. Typically called
    /// by the M2 phase-segregation transform.
    pub fn set_cold_phase(&mut self, c: Centroid3D) {
        self.cold_phase = Some(c);
    }
    /// Populate or overwrite the burst-motion slot. Typically called
    /// by the M3 burst-motion / TIDE transform.
    pub fn set_burst_motion(&mut self, c: Centroid3D) {
        self.burst_motion = Some(c);
    }
    /// Populate or overwrite the validation-structural slot.
    pub fn set_validation_structural(&mut self, c: Centroid3D) {
        self.validation_structural = Some(c);
    }
    /// Populate or overwrite the ligand-adjacent sub-cluster slot.
    pub fn set_ligand_adjacent_subcluster(&mut self, c: Centroid3D) {
        self.ligand_adjacent_subcluster = Some(c);
    }

    /// Number of slots currently populated. Useful for audits and
    /// progress tracking as M2 / M3 land.
    pub fn populated_count(&self) -> usize {
        let mut n = 0usize;
        for slot in [
            self.geometric.is_some(),
            self.lining.is_some(),
            self.driver.is_some(),
            self.hot_phase.is_some(),
            self.cold_phase.is_some(),
            self.burst_motion.is_some(),
            self.validation_structural.is_some(),
            self.ligand_adjacent_subcluster.is_some(),
        ] {
            if slot {
                n += 1;
            }
        }
        n
    }
}

// ============================================================================
// CausalScalars — per-site causal scalar tensor
// ============================================================================

/// Per-site causal scalar tensor: M1 / M2 / M3 producers each fill in
/// their column. At M1, only [`Self::spike_attribution_count`] is
/// honest; the four floating-point fields are initialized to
/// `f64::NAN` and become finite as M2 / M3 land.
///
/// `f64::NAN` is the canonical "not yet produced" sentinel. The
/// role-aware composite sort
/// (see [`crate::entangled_manifold::CausalSortKey`]) walks priority
/// lists and selects the first field where ALL residues have
/// non-NaN values, so the NaN sentinel doubles as the activation
/// gate for higher-priority sort fields.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CausalScalars {
    /// Count of spike events for which this site is in the attributed
    /// support set. Honest at M1.
    pub spike_attribution_count: u64,
    /// KCC (kinetic-causal contribution) score. NaN at M1; honest at M2.
    pub kcc_score: f64,
    /// Directed transfer entropy magnitude. NaN at M1; honest at M3.
    pub transfer_entropy: f64,
    /// Causal Δ-G (Gibbs-style energy contribution). NaN at M1;
    /// honest at M3.
    pub causal_dg: f64,
    /// Causal lag (peak-correlation lag in time units). NaN at M1;
    /// honest at M3.
    pub causal_lag: f64,
}

impl CausalScalars {
    /// Construct with M1's `spike_attribution_count` populated; all
    /// downstream causal fields initialized to `f64::NAN` until
    /// M2 / M3 producers land.
    pub fn new_m1(spike_attribution_count: u64) -> Self {
        Self {
            spike_attribution_count,
            kcc_score: f64::NAN,
            transfer_entropy: f64::NAN,
            causal_dg: f64::NAN,
            causal_lag: f64::NAN,
        }
    }

    /// Empty placeholder: no honest fields. Useful as a default
    /// before any producer has filled in even the M1 column.
    pub fn empty() -> Self {
        Self {
            spike_attribution_count: 0,
            kcc_score: f64::NAN,
            transfer_entropy: f64::NAN,
            causal_dg: f64::NAN,
            causal_lag: f64::NAN,
        }
    }
}

// ============================================================================
// SiteIdentity — per-site identity bundle
// ============================================================================

/// Per-site identity bundle. Carries the engine's [`SiteId`] /
/// [`ClusterId`] plus the [`ViewProvenance`] from the role-view
/// that this site was projected from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiteIdentity {
    /// Engine-assigned site identifier.
    pub site_id: SiteId,
    /// Engine-assigned cluster identifier (cluster the site's
    /// driver residue belongs to).
    pub cluster_id: ClusterId,
    /// Provenance from the role-view this site was projected from.
    pub provenance: ViewProvenance,
}

// ============================================================================
// SiteManifest — top-level per-site canonical type
// ============================================================================

/// Canonical per-site Rust type. One [`SiteManifest`] instance per
/// detected site. Per the B1 binding spec, `SiteManifest` does NOT
/// own a copy of the frame-level
/// [`crate::entangled_manifold::EntangledManifold`]; it references it
/// through [`Self::source_manifold_id`]. The engine maintains the
/// `(frame, EntangledManifoldId) → EntangledManifold` mapping.
///
/// Construction at M1 is from the M1.2 producer's per-site projection
/// of the frame-level manifold's role-views. Population semantics:
///
/// * [`Self::identity`], [`Self::frame`], [`Self::source_manifold_id`]
///   — populate from M1's clustering kernel output.
/// * [`Self::centroids`] — `geometric`, `lining`, `driver` populated
///   from M1's role-view AABBs (each AABB → derived centroid via
///   `Aabb::center()`). Phase slots are `None` at M1 and progressively
///   populated by M2 / M3.
/// * [`Self::causal_scalars`] — `spike_attribution_count` honest at
///   M1; the floating-point causal fields are `NaN` at M1 and become
///   honest as M2 / M3 land.
/// * [`Self::sort_lineage`] — the [`CausalSortKey`] used to order the
///   constituent role-view residues at view-construction time.
///   Provides the audit trail for which signal was the live sort
///   field at the time the site was projected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiteManifest {
    /// Per-site identity bundle.
    pub identity: SiteIdentity,
    /// 8-slot centroid projection of the frame-level role-views.
    pub centroids: CentroidManifold,
    /// Per-site causal scalar tensor; M1 honest column, M2 / M3
    /// columns NaN until those lanes land.
    pub causal_scalars: CausalScalars,
    /// MD frame index this site was projected at.
    pub frame: u64,
    /// Opaque reference back to the frame-level
    /// [`crate::entangled_manifold::EntangledManifold`] this site
    /// was projected from.
    pub source_manifold_id: EntangledManifoldId,
    /// Sort key used to order the constituent role-view residues at
    /// view-construction time. Audit-trail field.
    pub sort_lineage: CausalSortKey,

    // ─── Anti-Greenfield § 3 — Spatiotemporal Interferometer extension ────
    // Surgically additive, all `Option`-typed with `serde(skip_serializing
    // _if = None)`. Legacy consumers (Python reporting, static PDB viewers,
    // existing JSON schema readers) see the same shape they always saw on
    // the legacy clustering path; `None` everywhere ⇒ no key emitted.
    // Populated only when the InterferometricAdjudicator (T0/T2) is wired
    // into the post-clustering path of the captured WHILE graph.
    /// SO(3) geometry-plane power spectrum `C_l` for `l = 0..5`,
    /// rotationally invariant (RECT-3.1.b output). The KL divergence
    /// in the Adjudicator kernel reads these values from the relaxed
    /// and perturbed `ContactShellTile`s.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contact_shell_geo_power_spectrum: Option<[f32; 6]>,

    /// Latest computed Δ_AB KL divergence between the relaxed (P)
    /// and perturbed (Q) manifolds (T2 output). `None` on the
    /// legacy path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adjudicator_divergence: Option<f32>,

    /// F1 SWITCH selector — [`crate::pre_rank::AdjudicationCode`]-compatible
    /// integer (0=Prune, 1=Construct, 2=Violation). `None` on the
    /// legacy path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adjudicator_code: Option<u32>,

    /// Adjudicator-kernel elapsed time in nanoseconds, computed as
    /// `stop_clock - start_clock` from the CudaEvent harness (T4).
    /// Gating threshold: the < 5 μs CSR-H invariant. `None` if the
    /// telemetry harness is not active.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adjudicator_elapsed_ns: Option<u64>,

    // ─── CLA-2 / Ghost Pipeline Schema (operator addendum 2026-04-29) ──
    // Surgically additive — populated only by the Pillar 5 (Reporting)
    // ghost-pipeline serializer on the host. Never read inside any
    // GPU kernel. Every field `serde(skip_serializing_if = "Option::is_none")`
    // so the I/O bloat in cold-hold phases remains under the 70%
    // reduction target documented in the operator addendum.
    /// KCC metrics aggregated for the site's driver residue (or for
    /// the site as a whole if available). Mirrors the per-residue
    /// fields in `<target>.kcc_visualization.json`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kcc_metrics: Option<KccMetrics>,

    /// Thermodynamic / phase-signaling dossier for the site. Mirrors
    /// the per-site fields in `<target>.binding_sites.json`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub therm_dossier: Option<ThermDossier>,

    // ─── GhostPhaseLattice4D extension (MASTER ARCHITECTURAL DIRECTIVE V5) ──
    // The four blocks below are populated by the
    // `crate::ghost_phase_lattice` backend on its post-MD run. They are
    // strictly host-side (no GPU kernel reads them) and use the same
    // skip_serializing_if=None policy as the CLA-2 extension fields above
    // so legacy DBSCAN-only runs (without Ghost v2 records) emit the same
    // shape as before — no key produced. Populated together: emitting
    // any one without the other three is a producer bug, but the schema
    // tolerates partial population (None means "the lattice backend did
    // not run" or "the consumer stripped this block").
    /// Provenance metadata for the GhostPhaseLattice4D run that produced
    /// this site. Fields mirror directive Part IV.1 (`ghost_phase_lattice`
    /// JSON block): backend identifier, configuration parameters, and
    /// global edge-adjudication telemetry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ghost_phase_lattice: Option<GhostPhaseLatticeProvenance>,

    /// Per-protocol-phase aggregated centroid + driver block. Replaces the
    /// legacy single dead centroid with a per-phase trajectory: cold_hold
    /// position → heating position → warm_hold position → cooling
    /// position. Phases that did not produce any nodes for this site are
    /// omitted (the JSON object simply lacks that key — directive 4.2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase_manifold: Option<PhaseManifold>,

    /// Thermodynamic + CCNS lifecycle aggregates (mean KL by phase, mean
    /// thermo flux by phase, driver-residue persistence, mean water
    /// density by phase). The `water_density_status` field carries the
    /// directive Part V "unavailable_neutral" sentinel when the
    /// prism-therm sidecar has not been populated yet.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub therm_ccns_lifecycle: Option<ThermCcnsLifecycle>,

    /// SO(3) spherical-coherence summary for this site: which planes were
    /// populated across the constituent ghost nodes, the mean cosine
    /// similarity of accepted intra-component edges, and per-transition
    /// phase-boundary cosine values. Directive 4.4.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub so3_manifold: Option<So3Manifold>,
}

// ============================================================================
// GhostPhaseLattice4D provenance + manifold blocks (Anti-Greenfield: surgical
// additive types, all `Option`-wrapped on SiteManifest, all
// `skip_serializing_if=None` so legacy emission is byte-identical).
// ============================================================================

/// Metadata for a single GhostPhaseLattice4D run. Surfaced under the
/// per-site `ghost_phase_lattice` JSON key (directive 4.1). All values are
/// run-global (not per-site); placing them on each SiteManifest is the
/// price of serialising sites independently — consumers that aggregate
/// will see the same provenance on every site within one run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GhostPhaseLatticeProvenance {
    /// Backend identifier — always `"ghost_phase_lattice_4d"` for sites
    /// produced by this backend. Allows a downstream consumer to switch
    /// between legacy DBSCAN sites (no `ghost_phase_lattice` key emitted)
    /// and 4D-lattice sites (this string present).
    pub backend: String,
    /// Spatial cell size in Å (directive default 5.0).
    pub spatial_cell_size_a: f32,
    /// Maximum step gap allowed for a temporal edge (directive default 500).
    pub max_temporal_edge_steps: u64,
    /// Temporal-bucket size used by the lattice key (`step_idx /
    /// step_bucket_size`).
    pub step_bucket_size: u64,
    /// SO(3) acceptance threshold (directive default 0.75).
    pub so3_threshold: f32,
    /// Phase-transition policy descriptor — always
    /// `"monotone_protocol_lifecycle"` for the canonical lattice (directive
    /// Part II.1: cold_hold → heating → warm_hold → cooling, no skips).
    pub phase_transition_policy: String,
    /// Number of lattice nodes (Ghost v2 records) consumed by this run.
    pub n_tiles: u32,
    /// Number of distinct (spatial cell × phase × step bucket) cells the
    /// nodes bucketed into. Lower = denser temporal/spatial coupling.
    pub n_lattice_cells: u32,
    /// Number of directed edges accepted across all components in the
    /// run (directive 4.1's `n_directed_edges`).
    pub n_directed_edges: u64,
    /// Step range covered by this run.
    pub lattice_extent: LatticeExtent,
}

/// Step-range + phase-list extent for a GhostPhaseLattice4D run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LatticeExtent {
    pub step_start: u64,
    pub step_end: u64,
    /// Phase names actually present in the input nodes (e.g.
    /// `["cold_hold", "heating", "warm_hold"]` if cooling never fired).
    pub phases_present: Vec<String>,
}

/// Per-phase aggregate for a single site. Each `Option` field mirrors one
/// of the four protocol phases (cold_hold / heating / warm_hold / cooling);
/// `None` means the site had no nodes in that phase. Directive 4.2 emits
/// only the phases that fired — this struct's `skip_serializing_if=None`
/// preserves that semantics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct PhaseManifold {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cold_hold: Option<PhaseAggregate>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub heating: Option<PhaseAggregate>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warm_hold: Option<PhaseAggregate>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cooling: Option<PhaseAggregate>,
}

/// Single-phase aggregate: AABB-volume-weighted centroid, axis-aligned
/// bounding box union, mean KL divergence, set of unique driver residues.
/// The `centroid_xyz` here is the *phase-specific* centroid — labelled
/// explicitly so consumers don't conflate it with the legacy single dead
/// centroid the old DBSCAN backend emitted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhaseAggregate {
    /// AABB-volume-weighted centroid of the constituent ghost nodes.
    pub centroid_xyz: [f32; 3],
    /// Axis-aligned bounding box union over the constituent nodes.
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    /// Number of ghost nodes contributing to this phase.
    pub n_nodes: u32,
    /// Mean KL divergence over the constituent nodes (NaN values are
    /// excluded from the mean — `n_finite_kl` reports how many were
    /// counted).
    pub mean_kl_divergence: f32,
    /// Number of nodes that contributed a finite KL value to the mean.
    pub n_finite_kl: u32,
    /// Distinct driver residues observed in this phase. Empty when no
    /// node had a resolved causal lead.
    pub driver_residues: Vec<u32>,
    /// Step-range covered in this phase (`step_idx_min`..=`step_idx_max`).
    pub step_idx_min: u64,
    pub step_idx_max: u64,
}

/// Thermodynamic + CCNS lifecycle aggregates per phase. Directive 4.3.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ThermCcnsLifecycle {
    /// `phase_name -> mean_kl_divergence`. Populated only for phases the
    /// site participated in; missing phases are absent from the map (not
    /// emitted as null).
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub mean_kl_by_phase: std::collections::BTreeMap<String, f32>,
    /// `phase_name -> [mean_wd_change, mean_vib_energy]`. Components are
    /// excluded from the mean if NaN; `[NaN, NaN]` may be emitted when
    /// the upstream telemetry plane has not populated the field yet —
    /// downstream consumers MUST check `water_density_status` before
    /// integrating these values.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub mean_thermo_flux_by_phase: std::collections::BTreeMap<String, [f32; 2]>,
    /// `driver_residue_id -> persistence_fraction`. Sums to 1.0 if any
    /// drivers were observed; empty when no node had a resolved causal
    /// lead.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub driver_residue_persistence: std::collections::BTreeMap<u32, f32>,
    /// One of `"available"` / `"unavailable_neutral"`. Set to
    /// `"unavailable_neutral"` whenever every constituent node had `NaN`
    /// for `water_density` — directive Part V mandates the explicit
    /// sentinel rather than a zero-fill.
    pub water_density_status: String,
    /// `phase_name -> mean_water_density`. Empty when
    /// `water_density_status == "unavailable_neutral"`.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub mean_water_density_by_phase: std::collections::BTreeMap<String, f32>,
}

/// SO(3) spherical-coherence summary per site. Directive 4.4.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct So3Manifold {
    /// Per-plane status descriptor: `"populated"` if at least one
    /// constituent node had a non-zero value on that plane, otherwise
    /// `"sentinel"`.
    pub plane_status: So3PlaneStatus,
    /// Mean cosine similarity of accepted intra-component edges. This field
    /// is populated only when exact accepted-edge lineage exists; a global
    /// kernel accumulator is never apportioned across components.
    pub intra_component_mean_cosine: f32,
    /// `available_exact_edge_lineage` or
    /// `unavailable_no_accepted_edge_lineage`.
    pub intra_component_cosine_status: String,
    /// Number of intra-component edges aggregated into the mean.
    pub n_intra_edges: u32,
    /// `phase_a_to_phase_b -> mean_cosine` for exact accepted edges that
    /// crossed the named phase boundary. Empty without exact edge lineage.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub phase_transition_cosine: std::collections::BTreeMap<String, f32>,
    /// Exact availability of phase-transition edge statistics.
    pub phase_transition_cosine_status: String,
}

/// Per-plane populated/sentinel labels. `populated` means at least one
/// constituent node carried non-zero values on that plane; `sentinel`
/// means every node had the default zero spectrum. The 4 planes match
/// the GhostTileFrame `power_spectrum` plane order (geometry, causality,
/// thermodynamics, chemistry).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct So3PlaneStatus {
    pub geometry: String,
    pub causality: String,
    pub thermodynamics: String,
    pub chemistry: String,
}

impl Default for So3PlaneStatus {
    fn default() -> Self {
        Self {
            geometry: "sentinel".to_string(),
            causality: "sentinel".to_string(),
            thermodynamics: "sentinel".to_string(),
            chemistry: "sentinel".to_string(),
        }
    }
}

impl So3PlaneStatus {
    /// Build a status block from a 4-bit plane-presence mask
    /// (bit 0=geo, 1=caus, 2=thermo, 3=chem).
    pub fn from_mask(mask: u8) -> Self {
        let label = |bit: u8| -> String {
            if (mask & bit) != 0 {
                "populated".to_string()
            } else {
                "sentinel".to_string()
            }
        };
        Self {
            geometry: label(0b0001),
            causality: label(0b0010),
            thermodynamics: label(0b0100),
            chemistry: label(0b1000),
        }
    }
}

/// Canonical phase-name list — index = protocol_phase ordinal.
pub const PHASE_NAMES: [&str; 4] = ["cold_hold", "heating", "warm_hold", "cooling"];

/// Resolve protocol_phase ordinal → canonical name. Out-of-range ordinals
/// return `"unknown"`.
pub fn phase_name_for(ordinal: u8) -> &'static str {
    if (ordinal as usize) < PHASE_NAMES.len() {
        PHASE_NAMES[ordinal as usize]
    } else {
        "unknown"
    }
}

// ============================================================================
// CLA-2 — Ghost Pipeline Schema extensions (Pillar 5 reporting only)
//
// These types are **strictly host-side**. They are populated by the
// asynchronous serializer running on the ghost telemetry thread; no
// GPU kernel ever reads them. Per the operator's addendum
// (2026-04-29 §"Cross-Plane Separation"):
//
//   * Telemetry Plane (Claude-1): owns these structs end-to-end.
//   * Control Plane  (Claude-2): forbidden from referencing them
//     inside any `.cu` translation unit.
// ============================================================================

/// Kinematic-Causal Coupling metrics for a single site (Pillar 5
/// reporting). Mirrors the per-residue KCC schema in
/// `<target>.kcc_visualization.json` — every field is `Option` so
/// "Cold Hold" frames (where no causal motion is detected) emit a
/// near-empty object instead of a NaN-laden record. The
/// `skip_serializing_if` attribute on every field is the I/O-bloat
/// mitigation mandated in the operator addendum (~70% payload
/// reduction in quiescent phases).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KccMetrics {
    /// Number of MD steps with active UV→LIF coupling at the site's
    /// driver residue.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_causal_steps: Option<u32>,
    /// Ratio of motion during dense causal events vs. sparse events
    /// (> 1.0 = bursty, persistent-site signature).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub burst_motion: Option<f32>,
    /// Timestep offset that maximises the motion–causality
    /// cross-correlation (steps).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub causal_lag: Option<f32>,
    /// Ratio of net displacement to total path length
    /// (1.0 = perfectly directed, 0.0 = random walk).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub direction_score: Option<f32>,
    /// KCC composite score per the legacy ranker formula
    /// (sqrt of weighted causality + structural signals).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kcc_score: Option<f32>,
    /// Peak cross-correlation between causal activity and motion at
    /// the optimal timestep lag.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lag_corr_peak: Option<f32>,
    /// Maximum local covariance of motion and causality in
    /// timestep-windowed subregions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local_cov: Option<f32>,
    /// Saturating transform of total motion magnitude (0–1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub motion_efficiency: Option<f32>,
}

impl KccMetrics {
    /// Construct an all-`None` instance — the canonical "Cold Hold"
    /// signature where every metric is below its detection floor.
    /// `serde` will serialize this to `"kcc_metrics": {}` by default
    /// (or, with the field-level skip-if-none, to nothing at all
    /// when nested under [`SiteManifest::kcc_metrics`]).
    pub const fn empty() -> Self {
        Self {
            active_causal_steps: None,
            burst_motion: None,
            causal_lag: None,
            direction_score: None,
            kcc_score: None,
            lag_corr_peak: None,
            local_cov: None,
            motion_efficiency: None,
        }
    }
}

impl Default for KccMetrics {
    fn default() -> Self {
        Self::empty()
    }
}

/// Thermodynamic + phase-signaling dossier for a site. Mirrors the
/// canonical per-site shape produced by the `prism-therm` pipeline.
/// All fields are required (non-optional) because the source
/// pipeline always produces them — `therm_class` carries the
/// "INERT" sentinel for unresolved phases, and `ccns_tau` carries a
/// finite default. The whole `Option<ThermDossier>` wrap on
/// [`SiteManifest::therm_dossier`] handles "no thermodynamic data
/// yet for this frame" cleanly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThermDossier {
    /// Characteristic time constant of the phase (CCNS τ). Cross-
    /// references `<target>.binding_sites.json::sites[].ccns_tau`.
    pub ccns_tau: f32,
    /// Phase classification — typically one of
    /// `{"DYNAMIC", "BISTABLE", "INERT", "Soc", "Responsive", ...}`.
    /// Held as `String` (not an enum) because the upstream taxonomy
    /// is open: new classes appear as the prism-therm taxonomy
    /// evolves and we want the schema to absorb them without a
    /// recompile.
    pub therm_class: String,
    /// Druggability score in [0, 1] from the heuristic ranker.
    pub druggability: f32,
    /// Hot/cold-phase asymmetry signed in [-1, 1] (positive = hot
    /// dominant, negative = cold dominant).
    pub relative_asymmetry: f32,
    /// Hysteresis-loop asymmetry magnitude in [0, 1].
    pub hysteresis_asymmetry: f32,
}

impl ThermDossier {
    /// Sentinel "no thermodynamic data" dossier. Carries
    /// `therm_class = "INERT"`, every numeric field at zero. Useful
    /// as a placeholder before the prism-therm pipeline has emitted
    /// per-frame data; downstream readers MUST distinguish this
    /// from a real INERT classification by checking whether the
    /// surrounding `Option` is `Some` versus `None`.
    pub fn inert() -> Self {
        Self {
            ccns_tau: 0.0,
            therm_class: String::from("INERT"),
            druggability: 0.0,
            relative_asymmetry: 0.0,
            hysteresis_asymmetry: 0.0,
        }
    }
}

impl SiteManifest {
    /// LBVH-3 constructor — build a per-site manifest from the cluster's
    /// LBVH-derived AABB.
    ///
    /// Per the operator's RECT-4 / LBVH-3 mandate (2026-04-29) and
    /// Anti-Greenfield Doctrine §2.3 (extend, don't duplicate): this is
    /// the canonical bridge from the M1 producer's per-cluster AABB
    /// (output of [`crate::lbvh_tree`]'s bottom-up reduce, or
    /// equivalently the per-cluster AABB column from the M1 producer's
    /// CUB SegmentedReduce) to the per-site
    /// [`CentroidManifold`] 8-slot canonical container.
    ///
    /// # Slot population (M1 honest)
    ///
    /// Three slots are populated from the same LBVH AABB at M1 (per
    /// the [`SiteManifest`] type-level docs):
    ///
    /// | Slot | Value | View tag |
    /// |---|---|---|
    /// | `geometric` | `aabb.center()` | [`SpatialView::GeometricVoxelMass`] |
    /// | `lining`    | `aabb.center()` | [`SpatialView::LiningResidues`] |
    /// | `driver`    | `aabb.center()` | [`SpatialView::DriverResidues`] |
    ///
    /// The remaining 5 slots are intentionally `None` and are populated
    /// by later transforms (M2: hot/cold-phase, validation_structural,
    /// ligand_adjacent_subcluster; M3: burst_motion).
    ///
    /// # Spatial provenance preservation
    ///
    /// `source_manifold_id` carries the frame-level
    /// [`crate::entangled_manifold::EntangledManifold`] handle so the
    /// site can be back-referenced during audit. `sort_lineage` carries
    /// the [`CausalSortKey`] used to order the role-view residues at
    /// frame-build time — preserves the LBVH's spatial-provenance chain
    /// from Morton encoding through cluster assignment to site.
    ///
    /// # The 4 RECT-3 / Adjudicator extension fields
    ///
    /// `contact_shell_geo_power_spectrum`, `adjudicator_divergence`,
    /// `adjudicator_code`, `adjudicator_elapsed_ns` default to `None`.
    /// They are populated post-construction by the SO(3) stamping pass
    /// (`stamp_geo_power_spectrum_into_sites` in `so3_project.rs`) and
    /// by the Adjudicator (Claude-2 lane).
    pub fn from_lbvh_cluster_aabb(
        site_id: SiteId,
        cluster_id: ClusterId,
        aabb: &Aabb,
        source_manifold_id: EntangledManifoldId,
        provenance: ViewProvenance,
        sort_lineage: CausalSortKey,
        spike_attribution_count: u64,
        frame: u64,
    ) -> Self {
        let center = aabb.center();
        let mut centroids = CentroidManifold::new();
        centroids.set_geometric(Centroid3D::new(
            center,
            SpatialView::GeometricVoxelMass,
            frame,
        ));
        centroids.set_lining(Centroid3D::new(center, SpatialView::LiningResidues, frame));
        centroids.set_driver(Centroid3D::new(center, SpatialView::DriverResidues, frame));

        Self {
            identity: SiteIdentity {
                site_id,
                cluster_id,
                provenance,
            },
            centroids,
            causal_scalars: CausalScalars::new_m1(spike_attribution_count),
            frame,
            source_manifold_id,
            sort_lineage,
            // RECT-3 extension defaults — populated by the SO(3)
            // stamping pass (`crate::so3_project::stamp_geo_power_spectrum_into_sites`)
            // and the Adjudicator (Claude-2 lane).
            contact_shell_geo_power_spectrum: None,
            adjudicator_divergence: None,
            adjudicator_code: None,
            adjudicator_elapsed_ns: None,
            kcc_metrics: None,
            therm_dossier: None,
            ghost_phase_lattice: None,
            phase_manifold: None,
            therm_ccns_lifecycle: None,
            so3_manifold: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entangled_manifold::{
        CausalSignal, IdentityTieBreaker, SelectionPolicy, SortField, TieBreakerPolicy,
    };

    fn provenance() -> ViewProvenance {
        // Uses the deprecated TieBreakerPolicy variant on purpose;
        // existing producer paths still emit it. New code uses
        // IdentityTieBreaker via CausalSortKey.
        #[allow(deprecated)]
        ViewProvenance {
            signal: CausalSignal::SpikeAttributionCount,
            selection: SelectionPolicy::TopK { k: 3 },
            tie_breaker: TieBreakerPolicy::CausalThenResid,
            frame: 7,
        }
    }

    fn sort_key() -> CausalSortKey {
        CausalSortKey {
            priorities: vec![
                SortField::CausalCoupling,
                SortField::KccScore,
                SortField::SpikeAttributionCount,
            ],
            identity_tiebreaker: IdentityTieBreaker::ChainResidAtom,
        }
    }

    #[test]
    fn cluster_id_unclustered_sentinel_is_u32_max() {
        // M1 producer convention: UINT32_MAX = "this spike has no
        // cluster." Match the sentinel value exactly.
        assert_eq!(ClusterId::UNCLUSTERED.0, u32::MAX);
    }

    #[test]
    fn centroid_3d_frame_unset_sentinel_is_u64_max() {
        // Per M1 contract B3: u64::MAX is the only legitimate
        // "frame not yet set" sentinel. Frame 0 is a real frame.
        assert_eq!(Centroid3D::FRAME_UNSET, u64::MAX);
    }

    #[test]
    fn centroid_manifold_default_has_zero_populated_slots() {
        let m = CentroidManifold::new();
        assert_eq!(m.populated_count(), 0);
        assert!(m.geometric().is_none());
        assert!(m.lining().is_none());
        assert!(m.driver().is_none());
        assert!(m.hot_phase().is_none());
        assert!(m.cold_phase().is_none());
        assert!(m.burst_motion().is_none());
        assert!(m.validation_structural().is_none());
        assert!(m.ligand_adjacent_subcluster().is_none());
    }

    #[test]
    fn centroid_manifold_set_then_get_round_trips() {
        let mut m = CentroidManifold::new();
        let c = Centroid3D::new([1.0, 2.0, 3.0], SpatialView::GeometricVoxelMass, 7);
        m.set_geometric(c);
        assert_eq!(m.populated_count(), 1);
        let got = m.geometric().expect("geometric set above");
        assert_eq!(got.pos, [1.0, 2.0, 3.0]);
        assert_eq!(got.view, SpatialView::GeometricVoxelMass);
        assert_eq!(got.frame, 7);
    }

    #[test]
    fn centroid_manifold_eight_slots_independently_populable() {
        // Set each of the 8 slots once; populated_count must climb
        // monotonically from 0 to 8. This is the M1 contract's
        // "Eight-slot CentroidManifold" guarantee, mechanically.
        let mut m = CentroidManifold::new();
        let v = [
            SpatialView::GeometricVoxelMass,
            SpatialView::LiningResidues,
            SpatialView::DriverResidues,
            SpatialView::HotPhase,
            SpatialView::ColdPhase,
            SpatialView::BurstMotion,
            SpatialView::ValidationStructural,
            SpatialView::LigandAdjacentSubcluster,
        ];
        let c = |i: usize, view: SpatialView| Centroid3D::new([i as f32, 0.0, 0.0], view, 0);
        m.set_geometric(c(0, v[0]));
        assert_eq!(m.populated_count(), 1);
        m.set_lining(c(1, v[1]));
        assert_eq!(m.populated_count(), 2);
        m.set_driver(c(2, v[2]));
        assert_eq!(m.populated_count(), 3);
        m.set_hot_phase(c(3, v[3]));
        assert_eq!(m.populated_count(), 4);
        m.set_cold_phase(c(4, v[4]));
        assert_eq!(m.populated_count(), 5);
        m.set_burst_motion(c(5, v[5]));
        assert_eq!(m.populated_count(), 6);
        m.set_validation_structural(c(6, v[6]));
        assert_eq!(m.populated_count(), 7);
        m.set_ligand_adjacent_subcluster(c(7, v[7]));
        assert_eq!(m.populated_count(), 8);
    }

    #[test]
    fn causal_scalars_new_m1_has_only_attribution_count_honest() {
        // M1 producer fills spike_attribution_count; all other
        // causal fields are NaN until M2 / M3.
        let s = CausalScalars::new_m1(42);
        assert_eq!(s.spike_attribution_count, 42);
        assert!(s.kcc_score.is_nan());
        assert!(s.transfer_entropy.is_nan());
        assert!(s.causal_dg.is_nan());
        assert!(s.causal_lag.is_nan());
    }

    #[test]
    fn causal_scalars_empty_has_zero_count_and_all_nan_floats() {
        let s = CausalScalars::empty();
        assert_eq!(s.spike_attribution_count, 0);
        assert!(s.kcc_score.is_nan());
        assert!(s.transfer_entropy.is_nan());
        assert!(s.causal_dg.is_nan());
        assert!(s.causal_lag.is_nan());
    }

    #[test]
    fn site_manifest_round_trips_construction() {
        let m = SiteManifest {
            identity: SiteIdentity {
                site_id: SiteId(0),
                cluster_id: ClusterId(7),
                provenance: provenance(),
            },
            centroids: CentroidManifold::new(),
            causal_scalars: CausalScalars::new_m1(100),
            frame: 7,
            source_manifold_id: EntangledManifoldId(42),
            sort_lineage: sort_key(),
            contact_shell_geo_power_spectrum: None,
            adjudicator_divergence: None,
            adjudicator_code: None,
            adjudicator_elapsed_ns: None,
            kcc_metrics: None,
            therm_dossier: None,
            ghost_phase_lattice: None,
            phase_manifold: None,
            therm_ccns_lifecycle: None,
            so3_manifold: None,
        };
        assert_eq!(m.identity.site_id.0, 0);
        assert_eq!(m.identity.cluster_id.0, 7);
        assert_eq!(m.frame, 7);
        assert_eq!(m.source_manifold_id.0, 42);
        assert_eq!(m.causal_scalars.spike_attribution_count, 100);
        assert_eq!(m.centroids.populated_count(), 0);
    }

    // ────────────────────────────────────────────────────────────────────
    // LBVH-3 — from_lbvh_cluster_aabb constructor pins the M1
    // 3-honest-slot population from a per-cluster AABB.
    // ────────────────────────────────────────────────────────────────────

    #[test]
    fn lbvh3_from_cluster_aabb_populates_three_m1_honest_slots() {
        let aabb = Aabb {
            min: [0.0, -2.0, 1.0],
            max: [4.0, 2.0, 5.0],
        };
        let m = SiteManifest::from_lbvh_cluster_aabb(
            SiteId(11),
            ClusterId(11),
            &aabb,
            EntangledManifoldId(123),
            provenance(),
            sort_key(),
            42,
            7,
        );

        // Identity round-trips.
        assert_eq!(m.identity.site_id.0, 11);
        assert_eq!(m.identity.cluster_id.0, 11);
        assert_eq!(m.frame, 7);
        assert_eq!(m.source_manifold_id.0, 123);
        assert_eq!(m.causal_scalars.spike_attribution_count, 42);

        // 3 of 8 slots honest at M1.
        assert_eq!(m.centroids.populated_count(), 3);

        // Centre of [0, -2, 1]..[4, 2, 5] = (2, 0, 3). All 3 M1 slots
        // share the same value at M1 (per the type-level docs); the
        // per-view discrimination is in the SpatialView tag.
        let expected = [2.0_f32, 0.0, 3.0];
        let g = m.centroids.geometric().unwrap();
        let l = m.centroids.lining().unwrap();
        let d = m.centroids.driver().unwrap();
        assert_eq!(g.pos, expected);
        assert_eq!(l.pos, expected);
        assert_eq!(d.pos, expected);
        assert_eq!(g.view, SpatialView::GeometricVoxelMass);
        assert_eq!(l.view, SpatialView::LiningResidues);
        assert_eq!(d.view, SpatialView::DriverResidues);
        assert_eq!(g.frame, 7);
        assert_eq!(l.frame, 7);
        assert_eq!(d.frame, 7);

        // RECT-3 / Adjudicator extension fields default to None.
        assert!(m.contact_shell_geo_power_spectrum.is_none());
        assert!(m.adjudicator_divergence.is_none());
        assert!(m.adjudicator_code.is_none());
        assert!(m.adjudicator_elapsed_ns.is_none());

        // 5 of 8 slots remain unset — populated by M2 / M3 transforms.
        assert!(m.centroids.hot_phase().is_none());
        assert!(m.centroids.cold_phase().is_none());
        assert!(m.centroids.burst_motion().is_none());
        assert!(m.centroids.validation_structural().is_none());
        assert!(m.centroids.ligand_adjacent_subcluster().is_none());
    }

    // ────────────────────────────────────────────────────────────────────
    // CLA-2 / Ghost Pipeline schema serialization tests.
    //
    // The operator's addendum requires `serde(skip_serializing_if =
    // "Option::is_none")` to deliver the documented ~70% I/O reduction
    // in cold-hold phases. These tests pin that contract.
    // ────────────────────────────────────────────────────────────────────

    #[test]
    fn kcc_metrics_empty_serializes_to_empty_object() {
        let k = KccMetrics::empty();
        let s = serde_json::to_string(&k).unwrap();
        assert_eq!(
            s, "{}",
            "All-None KccMetrics must serialize to {{}} (skip_serializing_if=Option::is_none); \
             got {}",
            s
        );
    }

    #[test]
    fn kcc_metrics_partial_only_emits_set_fields() {
        // A site with only `burst_motion` resolved (typical M1.x state)
        // emits exactly that one field — not eight `null`s.
        let mut k = KccMetrics::empty();
        k.burst_motion = Some(1.05);
        k.kcc_score = Some(0.55);
        let s = serde_json::to_string(&k).unwrap();
        // Order is deterministic in serde-json (insertion order from
        // the struct definition); keep the assertion shape-loose by
        // checking presence + absence rather than exact byte order.
        assert!(
            s.contains("\"burst_motion\":1.05"),
            "missing burst_motion: {}",
            s
        );
        assert!(s.contains("\"kcc_score\":0.55"), "missing kcc_score: {}", s);
        assert!(
            !s.contains("active_causal_steps"),
            "active_causal_steps leaked: {}",
            s
        );
        assert!(!s.contains("causal_lag"), "causal_lag leaked: {}", s);
        assert!(!s.contains("null"), "None field serialized as null: {}", s);
    }

    #[test]
    fn kcc_metrics_round_trips_via_serde() {
        let k = KccMetrics {
            active_causal_steps: Some(8917),
            burst_motion: Some(1.0964780),
            causal_lag: Some(30.0),
            direction_score: Some(0.0003445477),
            kcc_score: Some(0.5503429),
            lag_corr_peak: Some(0.8100240),
            local_cov: Some(0.2572999),
            motion_efficiency: Some(0.0005009770),
        };
        let s = serde_json::to_string(&k).unwrap();
        let k2: KccMetrics = serde_json::from_str(&s).unwrap();
        assert_eq!(k, k2);
    }

    #[test]
    fn therm_dossier_round_trips_via_serde() {
        let t = ThermDossier {
            ccns_tau: 1.342099,
            therm_class: "DYNAMIC".into(),
            druggability: 0.6531283,
            relative_asymmetry: 1.296145,
            hysteresis_asymmetry: 0.298600,
        };
        let s = serde_json::to_string(&t).unwrap();
        let t2: ThermDossier = serde_json::from_str(&s).unwrap();
        assert_eq!(t, t2);
        assert!(
            s.contains("\"therm_class\":\"DYNAMIC\""),
            "therm_class missing: {}",
            s
        );
    }

    #[test]
    fn site_manifest_serde_omits_none_extension_fields_in_cold_hold() {
        // A "Cold Hold" SiteManifest (no Adjudicator data, no KCC,
        // no Therm) must serialize WITHOUT any of the optional
        // extension keys — only the always-present canonical fields.
        let m = SiteManifest::from_lbvh_cluster_aabb(
            SiteId(0),
            ClusterId(0),
            &Aabb {
                min: [0.0; 3],
                max: [1.0; 3],
            },
            EntangledManifoldId(0),
            provenance(),
            sort_key(),
            0,
            0,
        );
        let s = serde_json::to_string(&m).unwrap();
        // None of the 6 optional extension keys may appear:
        for forbidden in [
            "contact_shell_geo_power_spectrum",
            "adjudicator_divergence",
            "adjudicator_code",
            "adjudicator_elapsed_ns",
            "kcc_metrics",
            "therm_dossier",
        ] {
            assert!(
                !s.contains(forbidden),
                "Cold-Hold SiteManifest leaked extension field {} (Operator I/O Bloat \
                 Mitigation: 70% payload reduction violated): {}",
                forbidden,
                s
            );
        }
        // But canonical fields MUST appear.
        assert!(s.contains("\"identity\""));
        assert!(s.contains("\"centroids\""));
        assert!(s.contains("\"causal_scalars\""));
    }

    #[test]
    fn site_manifest_serde_emits_all_extension_fields_in_burst_phase() {
        // A "Burst" SiteManifest with every extension populated
        // serializes ALL of them (proves the skip-if-none gate
        // doesn't hide live data).
        let mut m = SiteManifest::from_lbvh_cluster_aabb(
            SiteId(0),
            ClusterId(0),
            &Aabb {
                min: [0.0; 3],
                max: [1.0; 3],
            },
            EntangledManifoldId(7),
            provenance(),
            sort_key(),
            123_456,
            7,
        );
        m.contact_shell_geo_power_spectrum = Some([0.4, 0.3, 0.15, 0.08, 0.04, 0.03]);
        m.adjudicator_divergence = Some(2.71828);
        m.adjudicator_code = Some(1); // Burst
        m.adjudicator_elapsed_ns = Some(7_400);
        m.kcc_metrics = Some(KccMetrics {
            active_causal_steps: Some(6949),
            burst_motion: Some(1.0496),
            causal_lag: Some(21.32),
            direction_score: Some(0.0003873),
            kcc_score: Some(0.5503),
            lag_corr_peak: Some(0.8294),
            local_cov: Some(0.3893),
            motion_efficiency: Some(0.000495),
        });
        m.therm_dossier = Some(ThermDossier {
            ccns_tau: 1.342099,
            therm_class: "DYNAMIC".into(),
            druggability: 0.6531283,
            relative_asymmetry: 1.296145,
            hysteresis_asymmetry: 0.298600,
        });

        let s = serde_json::to_string(&m).unwrap();
        for required in [
            "contact_shell_geo_power_spectrum",
            "adjudicator_divergence",
            "adjudicator_code",
            "adjudicator_elapsed_ns",
            "kcc_metrics",
            "therm_dossier",
            "burst_motion", // nested KccMetrics field
            "DYNAMIC",      // ThermDossier::therm_class value
        ] {
            assert!(
                s.contains(required),
                "Burst SiteManifest missing required serialized field {}: {}",
                required,
                s
            );
        }
    }

    #[test]
    fn lbvh3_point_cluster_aabb_round_trips_centroid() {
        // Degenerate point AABB (single-spike cluster) → centroid is
        // the spike position itself.
        let aabb = Aabb {
            min: [3.5, -1.0, 7.25],
            max: [3.5, -1.0, 7.25],
        };
        let m = SiteManifest::from_lbvh_cluster_aabb(
            SiteId(0),
            ClusterId(0),
            &aabb,
            EntangledManifoldId(0),
            provenance(),
            sort_key(),
            1,
            0,
        );
        assert_eq!(m.centroids.geometric().unwrap().pos, [3.5, -1.0, 7.25]);
    }
}
