//! Spatial / localization representation layer — Phase 1 of the
//! post-05dbc3dc canonical architecture lane.
//!
//! # Purpose
//!
//! This module makes the choice of spatial representation for a detected
//! binding site an explicit, compile-time-enforced decision. Before this
//! lane, downstream code silently reached for `site.centroid` — a single
//! scalar collapse of many distinct physical representations — and the
//! choice of *which* representation that scalar actually encoded was
//! implicit and hidden. Different consumers (ranking, validation,
//! emission, distance math) assumed different semantics for the same
//! field. That is a scalar-collapse artifact and it is forbidden by
//! Rule A of the lane doctrine.
//!
//! This module introduces two types:
//!
//! * [`SpatialView`] — a typed selector enum enumerating every
//!   physical representation a caller may request. No string keys, no
//!   default variant.
//! * [`CentroidManifold`] — a per-site container holding `Option<_>`
//!   values for each view. Absence is typed (the view is not derivable
//!   for this site at this stage of the pipeline), never encoded as a
//!   sentinel coordinate, NaN, or zero vector.
//!
//! Combined with the removal of `ClusteredBindingSite`'s bare `centroid`
//! accessor (renamed to `legacy_emission_centroid` and walled off to the
//! emission-compat path), this makes every canonical-logic reader
//! explicitly name the view it wants. The compiler enforces this by
//! refusing to compile any reader that hasn't been migrated.
//!
//! # Non-goals for Phase 1
//!
//! Phase 1 establishes the *types* and *access API*. It does not yet
//! derive every view for every site — most views are populated in later
//! phases (e.g., LiningResidues and DriverResidues become derivable once
//! lining residues are computed; HotPhase/ColdPhase become derivable
//! under the burst-segregation transform; LigandAdjacentSubcluster
//! requires ligand context, which is not available to the engine today;
//! ValidationStructural requires a reference structure). At Phase 1
//! only [`SpatialView::GeometricVoxelMass`] is always populated by
//! construction — it equals the former `centroid` field byte-for-byte.
//!
//! This asymmetry is handled by the `Option` type, not by placeholders.
//! A caller asking for an unpopulated view receives `None` and must
//! decide explicitly how to handle the absence — there is no silent
//! fallback to a default representation.
//!
//! # Future-composability
//!
//! The types here are designed to be inputs to the Phase 2 transform
//! audit spine. [`CentroidManifold`] is a plain data container with no
//! hidden side effects, owns no GPU resources, and has a stable memory
//! layout. A later transform may take a [`CentroidManifold`] from its
//! input and emit a refined [`CentroidManifold`] into its output without
//! this module needing to change.

use serde::{Deserialize, Serialize};

/// Canonical enumeration of the physical spatial representations a
/// binding site may expose. This is the **required** selector for any
/// centroid / DCC / distance-to-ligand / locality query on a canonical
/// site. There is no default variant and no stringly-typed fallback.
///
/// Variants correspond to real, physically meaningful representations:
///
/// * [`SpatialView::GeometricVoxelMass`] — voxel-mass-weighted (or
///   intensity²-weighted) centroid of the cluster's spike positions.
///   This is the only view always populated at Phase 1; it equals the
///   legacy `centroid` field.
/// * [`SpatialView::LiningResidues`] — centroid of the Cα (or heavy-
///   atom centroid) positions of the residues lining the pocket within
///   the lining-cutoff radius.
/// * [`SpatialView::DriverResidues`] — centroid of the subset of lining
///   residues identified as driver residues (high-contribution to spike
///   activity / top-intensity / catalytic-flagged).
/// * [`SpatialView::LigandAdjacentSubcluster`] — centroid of the subset
///   of spikes within a short radius of the co-crystallized ligand's
///   center of mass (only defined at validation time, when ligand
///   context is present; otherwise `None`).
/// * [`SpatialView::HotPhase`] — centroid computed over spikes whose
///   phase_bits fall in the burst phase window (high activity).
/// * [`SpatialView::ColdPhase`] — centroid computed over spikes whose
///   phase_bits fall in the quiescent phase window.
/// * [`SpatialView::ValidationStructural`] — centroid aligned against a
///   known reference structure (only defined when a reference is
///   supplied; otherwise `None`).
/// * [`SpatialView::BurstMotion`] — spatiotemporal representation of
///   the site's breathing-mode / open-close motion; only defined when
///   ensemble trajectory data is available.
///
/// Callers MUST name the view they want. They do not pick between views
/// implicitly. Adding a new view is a source-incompatible change: every
/// match on `SpatialView` has to handle it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpatialView {
    /// Voxel-mass / intensity²-weighted centroid of the cluster's
    /// spike positions. The "classic" site center before this lane.
    GeometricVoxelMass,
    /// Centroid of the Cα (or heavy-atom centroid) positions of the
    /// residues lining the pocket.
    LiningResidues,
    /// Centroid of driver residues (catalytic-flagged or
    /// high-contribution subset of lining residues).
    DriverResidues,
    /// Centroid of the spike sub-cluster adjacent to a reference
    /// ligand (only populated when ligand context is present).
    LigandAdjacentSubcluster,
    /// Centroid over the high-activity / burst phase spike subset.
    HotPhase,
    /// Centroid over the quiescent phase spike subset.
    ColdPhase,
    /// Structurally-aligned centroid (requires a reference structure).
    ValidationStructural,
    /// Spatiotemporal (breathing-mode / open-close) representation.
    /// Encoded here as the time-averaged centroid for scalar-interface
    /// compatibility; the full trajectory representation is the
    /// responsibility of a later transform.
    BurstMotion,
}

impl SpatialView {
    /// Canonical short identifier for logging / JSON provenance labels.
    /// Every reader that emits a view identifier should call this; no
    /// ad-hoc strings anywhere.
    pub fn as_str(&self) -> &'static str {
        match self {
            SpatialView::GeometricVoxelMass => "geometric_voxel_mass",
            SpatialView::LiningResidues => "lining_residues",
            SpatialView::DriverResidues => "driver_residues",
            SpatialView::LigandAdjacentSubcluster => "ligand_adjacent_subcluster",
            SpatialView::HotPhase => "hot_phase",
            SpatialView::ColdPhase => "cold_phase",
            SpatialView::ValidationStructural => "validation_structural",
            SpatialView::BurstMotion => "burst_motion",
        }
    }
}

/// Multi-view localization container held on every canonical site.
///
/// Each field is `Option<[f32; 3]>`. `None` means "this view is not
/// derivable for this site at this stage of the pipeline" — NOT "pick a
/// default." A caller that needs a specific view and finds `None` must
/// decide explicitly how to handle absence; there is no silent fallback.
///
/// The field order mirrors the variant order of [`SpatialView`] and is
/// intentional — it documents a recommended examination order for
/// adjudicators that try successive views (e.g., "geometric → lining →
/// driver → ligand-adjacent"), though each lookup remains explicit.
///
/// # Invariants
///
/// * `geometric_voxel_mass` is `Some(_)` on every canonical site
///   produced by [`crate::persistent_engine::build_sites_from_clustering`]
///   (Phase 1 populates it at construction from the clustering output).
/// * All other fields are `None` at Phase 1 unless a later phase
///   populates them explicitly via the `set_*` helpers.
/// * No field ever contains NaN, the zero vector as a sentinel, or
///   coordinates outside the simulation bounding box. Absence is
///   encoded only as `None`.
///
/// # Construction
///
/// The typical construction pattern is:
///
/// ```ignore
/// let mut manifold = CentroidManifold::with_geometric_voxel_mass([x, y, z]);
/// // Later, when lining residues have been computed:
/// manifold.set(SpatialView::LiningResidues, lining_centroid);
/// ```
///
/// [`CentroidManifold::new`] is available for the rare case of a site
/// with no view populated yet (e.g., a placeholder), but
/// [`CentroidManifold::with_geometric_voxel_mass`] is the canonical
/// constructor because every site produced by the clustering path has
/// a geometric-voxel-mass centroid by construction.
#[deprecated(
    since = "M1.1",
    note = "Use the new 8-slot CentroidManifold in crate::site_manifest. \
            Migration of persistent_engine.rs callsites is POST-M1 per COMMIT B's \
            deferred deprecation."
)]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CentroidManifold {
    /// See [`SpatialView::GeometricVoxelMass`].
    pub geometric_voxel_mass: Option<[f32; 3]>,
    /// See [`SpatialView::LiningResidues`].
    pub lining_residues: Option<[f32; 3]>,
    /// See [`SpatialView::DriverResidues`].
    pub driver_residues: Option<[f32; 3]>,
    /// See [`SpatialView::LigandAdjacentSubcluster`].
    pub ligand_adjacent_subcluster: Option<[f32; 3]>,
    /// See [`SpatialView::HotPhase`].
    pub hot_phase: Option<[f32; 3]>,
    /// See [`SpatialView::ColdPhase`].
    pub cold_phase: Option<[f32; 3]>,
    /// See [`SpatialView::ValidationStructural`].
    pub validation_structural: Option<[f32; 3]>,
    /// See [`SpatialView::BurstMotion`].
    pub burst_motion: Option<[f32; 3]>,
}

#[allow(deprecated)] // CentroidManifold is itself #[deprecated] as of M1.1; the
                     // legacy callsites in persistent_engine.rs:159, :2740 still
                     // compile against this impl with deprecation warnings, by
                     // design. POST-M1 steering lane migrates them to the new
                     // crate::site_manifest::CentroidManifold.
impl CentroidManifold {
    /// Construct a manifold with every view `None`.
    ///
    /// Callers should prefer [`CentroidManifold::with_geometric_voxel_mass`]
    /// for canonical sites; this is for rare placeholder cases only.
    pub fn new() -> Self {
        Self::default()
    }

    /// Canonical construction path: every site produced by the
    /// post-MD clustering pipeline has a geometric-voxel-mass centroid.
    pub fn with_geometric_voxel_mass(centroid: [f32; 3]) -> Self {
        Self {
            geometric_voxel_mass: Some(centroid),
            ..Self::default()
        }
    }

    /// Retrieve a named view. Returns `None` if the view is not
    /// currently derivable for this site. Callers MUST handle `None`
    /// explicitly; there is no default fallback.
    #[inline]
    pub fn view(&self, which: SpatialView) -> Option<[f32; 3]> {
        match which {
            SpatialView::GeometricVoxelMass => self.geometric_voxel_mass,
            SpatialView::LiningResidues => self.lining_residues,
            SpatialView::DriverResidues => self.driver_residues,
            SpatialView::LigandAdjacentSubcluster => self.ligand_adjacent_subcluster,
            SpatialView::HotPhase => self.hot_phase,
            SpatialView::ColdPhase => self.cold_phase,
            SpatialView::ValidationStructural => self.validation_structural,
            SpatialView::BurstMotion => self.burst_motion,
        }
    }

    /// Populate or overwrite a named view. Phase 1 exposes this so that
    /// later pipeline stages (lining-residue computation,
    /// phase-segregation, validation alignment) can fill in their
    /// view without adding a new setter per view.
    #[inline]
    pub fn set(&mut self, which: SpatialView, centroid: [f32; 3]) {
        match which {
            SpatialView::GeometricVoxelMass => self.geometric_voxel_mass = Some(centroid),
            SpatialView::LiningResidues => self.lining_residues = Some(centroid),
            SpatialView::DriverResidues => self.driver_residues = Some(centroid),
            SpatialView::LigandAdjacentSubcluster => {
                self.ligand_adjacent_subcluster = Some(centroid)
            }
            SpatialView::HotPhase => self.hot_phase = Some(centroid),
            SpatialView::ColdPhase => self.cold_phase = Some(centroid),
            SpatialView::ValidationStructural => self.validation_structural = Some(centroid),
            SpatialView::BurstMotion => self.burst_motion = Some(centroid),
        }
    }

    /// Euclidean distance from this site's named view to an arbitrary
    /// point. Returns `None` if the view is not derivable for this
    /// site. Caller MUST handle `None` explicitly.
    ///
    /// Note that this does not clamp, normalize, or interpret the
    /// distance as any particular physical quantity; it is a raw
    /// geometric primitive. Downstream helpers (e.g., `dcc`,
    /// `distance_to_ligand`) are thin semantic wrappers around this.
    pub fn distance_to(&self, target: [f32; 3], view: SpatialView) -> Option<f32> {
        let [x, y, z] = self.view(view)?;
        let dx = x - target[0];
        let dy = y - target[1];
        let dz = z - target[2];
        Some((dx * dx + dy * dy + dz * dz).sqrt())
    }

    /// Distance-to-centroid-of-cocrystal (DCC): Euclidean distance from
    /// this site's named view to a ground-truth reference point.
    /// Typed alias for [`CentroidManifold::distance_to`] that signals
    /// the semantic intent at the call site. Returns `None` if the view
    /// is not derivable.
    #[inline]
    pub fn dcc(&self, ground_truth: [f32; 3], view: SpatialView) -> Option<f32> {
        self.distance_to(ground_truth, view)
    }

    /// Distance from this site's named view to a ligand center.
    /// Typed alias for [`CentroidManifold::distance_to`] that signals
    /// the semantic intent at the call site. Returns `None` if the view
    /// is not derivable.
    #[inline]
    pub fn distance_to_ligand(&self, ligand: [f32; 3], view: SpatialView) -> Option<f32> {
        self.distance_to(ligand, view)
    }

    /// Iterate populated views. Handy for adjudicators that want to
    /// examine every available representation.
    pub fn iter_populated(&self) -> impl Iterator<Item = (SpatialView, [f32; 3])> + '_ {
        use SpatialView::*;
        [
            (GeometricVoxelMass, self.geometric_voxel_mass),
            (LiningResidues, self.lining_residues),
            (DriverResidues, self.driver_residues),
            (LigandAdjacentSubcluster, self.ligand_adjacent_subcluster),
            (HotPhase, self.hot_phase),
            (ColdPhase, self.cold_phase),
            (ValidationStructural, self.validation_structural),
            (BurstMotion, self.burst_motion),
        ]
        .into_iter()
        .filter_map(|(v, c)| c.map(|coord| (v, coord)))
        .collect::<Vec<_>>()
        .into_iter()
    }

    /// Number of views currently populated.
    pub fn populated_count(&self) -> usize {
        [
            self.geometric_voxel_mass,
            self.lining_residues,
            self.driver_residues,
            self.ligand_adjacent_subcluster,
            self.hot_phase,
            self.cold_phase,
            self.validation_structural,
            self.burst_motion,
        ]
        .iter()
        .filter(|v| v.is_some())
        .count()
    }
}

#[cfg(test)]
#[allow(deprecated)] // these tests exercise the legacy CentroidManifold which
                     // is #[deprecated] as of M1.1; tests are retained
                     // (per M1.1 hard constraint that all 12 entangled_manifold
                     // tests pass and that the legacy type stays functional)
                     // until POST-M1 callsite migration retires the type.
mod tests {
    use super::*;

    #[test]
    fn spatial_view_as_str_is_snake_case_and_stable() {
        // Stability check: the string form is baked into downstream
        // provenance labels, so this is a contract. If you change
        // these strings you are breaking downstream.
        assert_eq!(
            SpatialView::GeometricVoxelMass.as_str(),
            "geometric_voxel_mass"
        );
        assert_eq!(SpatialView::LiningResidues.as_str(), "lining_residues");
        assert_eq!(SpatialView::DriverResidues.as_str(), "driver_residues");
        assert_eq!(
            SpatialView::LigandAdjacentSubcluster.as_str(),
            "ligand_adjacent_subcluster"
        );
        assert_eq!(SpatialView::HotPhase.as_str(), "hot_phase");
        assert_eq!(SpatialView::ColdPhase.as_str(), "cold_phase");
        assert_eq!(
            SpatialView::ValidationStructural.as_str(),
            "validation_structural"
        );
        assert_eq!(SpatialView::BurstMotion.as_str(), "burst_motion");
    }

    #[test]
    fn manifold_default_has_every_view_none() {
        // No silent fallback: fresh manifold has zero populated views.
        let m = CentroidManifold::new();
        assert_eq!(m.populated_count(), 0);
        for view in [
            SpatialView::GeometricVoxelMass,
            SpatialView::LiningResidues,
            SpatialView::DriverResidues,
            SpatialView::LigandAdjacentSubcluster,
            SpatialView::HotPhase,
            SpatialView::ColdPhase,
            SpatialView::ValidationStructural,
            SpatialView::BurstMotion,
        ] {
            assert!(
                m.view(view).is_none(),
                "view {:?} should default None",
                view
            );
        }
    }

    #[test]
    fn with_geometric_voxel_mass_populates_only_that_view() {
        let m = CentroidManifold::with_geometric_voxel_mass([1.0, 2.0, 3.0]);
        assert_eq!(m.populated_count(), 1);
        assert_eq!(
            m.view(SpatialView::GeometricVoxelMass),
            Some([1.0, 2.0, 3.0])
        );
        assert!(m.view(SpatialView::LiningResidues).is_none());
        assert!(m.view(SpatialView::DriverResidues).is_none());
        assert!(m.view(SpatialView::HotPhase).is_none());
        assert!(m.view(SpatialView::ColdPhase).is_none());
    }

    #[test]
    fn set_then_get_roundtrips() {
        let mut m = CentroidManifold::new();
        m.set(SpatialView::LiningResidues, [10.0, 20.0, 30.0]);
        m.set(SpatialView::HotPhase, [11.0, 21.0, 31.0]);
        assert_eq!(
            m.view(SpatialView::LiningResidues),
            Some([10.0, 20.0, 30.0])
        );
        assert_eq!(m.view(SpatialView::HotPhase), Some([11.0, 21.0, 31.0]));
        assert!(m.view(SpatialView::GeometricVoxelMass).is_none());
        assert_eq!(m.populated_count(), 2);
    }

    #[test]
    fn distance_to_returns_none_for_unpopulated_view() {
        // Phase 1 enforcement: no silent default. An unpopulated view
        // yields None, not a sentinel, not "the closest available view."
        let m = CentroidManifold::with_geometric_voxel_mass([0.0, 0.0, 0.0]);
        assert!(m
            .distance_to([1.0, 2.0, 3.0], SpatialView::LiningResidues)
            .is_none());
        assert!(m
            .distance_to([1.0, 2.0, 3.0], SpatialView::DriverResidues)
            .is_none());
    }

    #[test]
    fn distance_to_returns_euclidean_for_populated_view() {
        let m = CentroidManifold::with_geometric_voxel_mass([0.0, 0.0, 0.0]);
        let d = m
            .distance_to([3.0, 4.0, 0.0], SpatialView::GeometricVoxelMass)
            .unwrap();
        assert!((d - 5.0).abs() < 1e-6);
    }

    #[test]
    fn dcc_and_distance_to_ligand_match_distance_to_semantically() {
        // dcc and distance_to_ligand are typed aliases that signal
        // intent at the call site but must agree numerically with
        // distance_to for the same inputs.
        let m = CentroidManifold::with_geometric_voxel_mass([1.0, 2.0, 3.0]);
        let gt = [4.0, 6.0, 3.0];
        let view = SpatialView::GeometricVoxelMass;
        let d_generic = m.distance_to(gt, view).unwrap();
        let d_dcc = m.dcc(gt, view).unwrap();
        let d_lig = m.distance_to_ligand(gt, view).unwrap();
        assert!((d_generic - 5.0).abs() < 1e-6);
        assert!((d_generic - d_dcc).abs() < 1e-9);
        assert!((d_generic - d_lig).abs() < 1e-9);
    }

    #[test]
    fn iter_populated_skips_none_views() {
        let mut m = CentroidManifold::with_geometric_voxel_mass([1.0, 2.0, 3.0]);
        m.set(SpatialView::HotPhase, [4.0, 5.0, 6.0]);
        let populated: Vec<_> = m.iter_populated().collect();
        assert_eq!(populated.len(), 2);
        assert!(populated
            .iter()
            .any(|(v, c)| *v == SpatialView::GeometricVoxelMass && *c == [1.0, 2.0, 3.0]));
        assert!(populated
            .iter()
            .any(|(v, c)| *v == SpatialView::HotPhase && *c == [4.0, 5.0, 6.0]));
    }

    #[test]
    fn no_hidden_default_if_view_unpopulated() {
        // Regression guard for the rule that launched this whole lane:
        // asking for LiningResidues on a site that does not have lining
        // data must NOT silently return GeometricVoxelMass. The caller
        // MUST see None and decide.
        let m = CentroidManifold::with_geometric_voxel_mass([42.0, 42.0, 42.0]);
        assert!(m.view(SpatialView::LiningResidues).is_none());
        assert!(m.view(SpatialView::DriverResidues).is_none());
        assert!(m.view(SpatialView::HotPhase).is_none());
        assert!(m.view(SpatialView::ColdPhase).is_none());
    }
}
