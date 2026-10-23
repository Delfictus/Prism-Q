//! PRISM-Therm: SDST (Spike-Driven Sparse Temporal Hash) integration bridge.
//!
//! Converts NHS spike events into SDST format, feeds them into the
//! thermodynamic hash table, and runs the 5-phase hysteresis + CCNS + TIDE
//! analysis pipeline that distinguishes cryptic binding sites from thermal noise.
//!
//! # Pipeline
//! ```text
//! GpuSpikeEvent[] (NHS output, all phases 0-4)
//!        ↓  ingest_all_spikes()  (sorted by timestep)
//! SDST hash table  (events tagged: phase_id, local_temp, water_density, vibrational_energy)
//!        ↓  analyze()
//! ┌─ hysteresis_region() per NHS site  →  asymmetry_score, is_hysteretic
//! ├─ ccns_region()       per NHS site  →  tau, druggability classification
//! ├─ tide_decomposition() per site     →  per-residue causal ΔG (top 20 by TE)
//! └─ ccns_all_pockets()  global scan   →  independent pocket list from SDST
//! ```
//!
//! # Phase boundaries
//! Computed at runtime from the live `CryoUvProtocol` instance — never hardcoded:
//! ```text
//! p0=0,  p1=cold_hold_steps,  p2+=ramp_steps,
//! p3+=warm_hold_steps,  p4+=ramp_down_steps,  p5+=cold_return_steps
//! ```
//! For `fast_35k().with_hysteresis()` this gives [0, 14000, 20000, 35000, 41000, 55000].
//! For `standard().with_hysteresis()` this gives [0, 5000, 15000, 20000, 30000, 35000].

#![allow(clippy::needless_pass_by_value)]

use anyhow::Result;
use serde::Serialize;

use crate::fused_engine::{CryoUvProtocol, GpuSpikeEvent};
use crate::input::PrismPrepTopology;
use crate::persistent_engine::ClusteredBindingSite;
use crate::site_manifest::ClusterId;

// Re-exports from the sdst crate that we need
use sdst::{CcnsClass, CcnsResult, Sdst, SdstConfig, SpatialRegion};

/// Exact percentile branch extracted from the multi-stream NHS post-MD filter.
/// Admission and ordering are shared by live execution and saved-evidence replay.
/// Otsu remains an explicit separate live policy, not approximated here.
#[derive(Debug, Serialize)]
pub struct NativeChannelSelectionReceipt {
    pub channel: &'static str,
    pub input_count: usize,
    pub selected_count: usize,
    pub threshold: Option<f32>,
    pub threshold_raw_f32_bits: Option<u32>,
    pub rule: &'static str,
}

/// Preserve source record ordinals through the exact shared channel selector.
pub fn select_native_percentile_indices(
    spikes: &[GpuSpikeEvent],
    spike_percentile: u32,
) -> (Vec<usize>, Vec<NativeChannelSelectionReceipt>) {
    let whole_bypass = spikes.len() <= 1000;
    let pct_f = (spike_percentile.min(99) as f32) / 100.0;
    let channel = |source| match source {
        1 => 0,
        3 => 2,
        4 | 5 => 3,
        6 => 4,
        _ => 1,
    };
    let names = ["UV", "LIF", "EFP", "LADD_COFIRE", "VORONOI_LIF"];
    if whole_bypass {
        let mut counts = [0usize; 5];
        for spike in spikes {
            counts[channel(spike.spike_source)] += 1;
        }
        let receipts = (0..5)
            .map(|i| NativeChannelSelectionReceipt {
                channel: names[i],
                input_count: counts[i],
                selected_count: counts[i],
                threshold: None,
                threshold_raw_f32_bits: None,
                rule: "whole_population_len_le_1000_preserved",
            })
            .collect();
        return ((0..spikes.len()).collect(), receipts);
    }
    let mut channels: [Vec<usize>; 5] = std::array::from_fn(|_| Vec::new());
    for (index, spike) in spikes.iter().enumerate() {
        channels[channel(spike.spike_source)].push(index);
    }
    let mut selected = Vec::new();
    let mut receipts = Vec::new();
    for (i, indices) in channels.into_iter().enumerate() {
        let input_count = indices.len();
        let (threshold, rule) = if i == 3 {
            (None, "source_4_or_5_preserved")
        } else if input_count < 100 {
            (None, "channel_len_lt_100_preserved")
        } else {
            let mut sorted: Vec<f32> = indices.iter().map(|&index| spikes[index].intensity).collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let idx = (sorted.len() as f32 * pct_f) as usize;
            (
                Some(sorted.get(idx).copied().unwrap_or(0.0)),
                "intensity_ge_per_channel_percentile",
            )
        };
        let before = selected.len();
        selected.extend(
            indices
                .into_iter()
                .filter(|&index| threshold.map_or(true, |t| spikes[index].intensity >= t)),
        );
        receipts.push(NativeChannelSelectionReceipt {
            channel: names[i],
            input_count,
            selected_count: selected.len() - before,
            threshold,
            threshold_raw_f32_bits: threshold.map(f32::to_bits),
            rule,
        });
    }
    (selected, receipts)
}

pub fn select_native_percentile_spikes(
    spikes: Vec<GpuSpikeEvent>, spike_percentile: u32,
) -> (Vec<GpuSpikeEvent>, Vec<NativeChannelSelectionReceipt>) {
    let (indices, receipt) = select_native_percentile_indices(&spikes, spike_percentile);
    (indices.into_iter().map(|i| spikes[i]).collect(), receipt)
}

/// Selection policy for GhostPhaseLattice4D components entering PRISM-Therm.
///
/// Drops components below `min_nodes`, returns the survivors largest-first, and
/// breaks ties by ascending component index so repeated runs over the same
/// lattice analyze the same sites in the same order.
///
/// This is the canonical owner of the policy named by
/// `R17_ENGINE_THERM_LATTICE_SITE_SURFACE`. It lives here, rather than inline in
/// `nhs_rt_full`, because a selection rule buried in the binary cannot be pinned
/// by a regression test — and the ordering half of this contract is invisible in
/// output, so a silent revert to input order would not be observable at runtime.
/// The tie-break is the only thing making the analyzed-site set reproducible
/// when two components have equal node counts.
pub fn select_lattice_components(n_nodes: &[u32], min_nodes: u32) -> Vec<usize> {
    let mut kept: Vec<(usize, u32)> = n_nodes
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, n)| *n >= min_nodes)
        .collect();
    kept.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    kept.into_iter().map(|(i, _)| i).collect()
}

/// Surface a materialized site was derived from.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MaterializedSiteSurface {
    /// Derived from a selected GhostPhaseLattice4D component.
    LatticeComponent,
    /// Derived from a CCL consensus cluster.
    CclCluster,
}

/// Typed cause for an analyzed site that did not reach a materialized site.
///
/// An unmatched site is a recorded result, not an absence: it names what was
/// analyzed, why it did not materialize, and which consumers were therefore not
/// populated. Silently dropping it is what let a whole lattice surface be
/// computed and discarded.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SiteUnmatchedReason {
    /// No `AnalyzedSiteIdentity -> MaterializedSiteId` mapping existed.
    NoMaterializationMapping,
    /// The site was removed by a selection policy applied after analysis.
    FilteredAfterAnalysis,
    /// A required companion artifact for materialization was absent.
    MissingCompanionData,
    /// Two analyzed sites claimed the same materialized identity.
    DuplicateMaterializedId,
    /// The source AABB was non-finite, inverted, or otherwise unusable.
    InvalidSourceRegion,
    /// The consumer expected a different identity domain.
    ConsumerDomainMismatch,
}

/// A site materialized from an analyzed source object.
///
/// This is the canonical materialized-site record. `binding_sites` is populated
/// from these, so the lattice surface contributes real sites carrying their own
/// identity, region and centroid rather than borrowing CCL cluster records.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct MaterializedSite {
    /// Identity in the materialized domain; assigned here and nowhere else.
    pub materialized_site_id: MaterializedSiteId,
    /// Identity in the source domain, preserved for every downstream join.
    pub analyzed_site_identity: AnalyzedSiteIdentity,
    /// Which surface produced this site.
    pub source_surface: MaterializedSiteSurface,
    /// Source region, carried verbatim from the producing surface.
    pub aabb_min: [f32; 3],
    /// Source region, carried verbatim from the producing surface.
    pub aabb_max: [f32; 3],
    /// Real centroid of the source region. Never a default.
    pub centroid: [f32; 3],
    /// Exact producer of this materialization.
    pub provenance: String,
}

/// Binding-site-shaped read model for a materialized lattice site.
///
/// This projection exists only for legacy consumers that need site geometry.
/// Its identity remains [`MaterializedSiteId`], and it deliberately has no
/// `cluster_id`: a lattice component is not a CCL cluster. The canonical source
/// record remains [`MaterializedSite`].
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LatticeBindingSiteProjection {
    pub materialized_site_id: MaterializedSiteId,
    pub analyzed_site_identity: AnalyzedSiteIdentity,
    pub source_surface: MaterializedSiteSurface,
    pub source_component_id: LatticeComponentId,
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    pub centroid: [f32; 3],
    pub materialization_provenance: String,
}

impl LatticeBindingSiteProjection {
    /// Resolve this projection to at most one therm result by its assigned
    /// materialized identity. A result from a numerically equal CCL cluster is
    /// not a match, and a source-identity disagreement is an error rather than
    /// a fallback.
    pub fn find_result<'a>(
        &self,
        analysis: &'a PrismThermAnalysis,
    ) -> Result<Option<&'a PrismThermSiteResult>> {
        let result = unique_materialized_result(&analysis.sites, self.materialized_site_id)?;
        if let Some(result) = result {
            if !result.matches_analyzed(self.analyzed_site_identity) {
                anyhow::bail!(
                    "{:?}: materialized site {:?} belongs to {:?}, result belongs to {:?}",
                    SiteUnmatchedReason::ConsumerDomainMismatch,
                    self.materialized_site_id,
                    self.analyzed_site_identity,
                    result.analyzed_site_identity
                );
            }
        }
        Ok(result)
    }
}

/// Whether a profile permits incomplete lattice materialization.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LatticeMaterializationRequirement {
    /// Generic or exploratory profiles may retain typed unmatched rows.
    Optional,
    /// THERM_POSTMD and strict HYPER_RUNTIME require every analyzed site.
    Required,
}

/// Terminal disposition of one analyzed lattice site.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum LatticeSiteTerminalDisposition {
    Merged,
    Unmatched,
}

/// Terminal, geometry-preserving disposition for one analyzed lattice site.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LatticeSiteDispositionRecord {
    pub disposition: LatticeSiteTerminalDisposition,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub materialized_site_id: Option<MaterializedSiteId>,
    pub analyzed_site_identity: AnalyzedSiteIdentity,
    pub source_surface: MaterializedSiteSurface,
    pub source_component_id: LatticeComponentId,
    pub centroid: [f32; 3],
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unmatched_reason: Option<SiteUnmatchedReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failed_consumer: Option<String>,
    pub promotable: bool,
    pub hysteresis_positive: bool,
}

/// Conservation counters over one identity domain and analyzed population.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub struct LatticeSitePopulationCounts {
    pub selected_site_count: usize,
    pub analyzed_site_count: usize,
    pub merged_site_count: usize,
    pub unmatched_site_count: usize,
    pub hysteresis_positive_count: usize,
}

/// Complete terminal partition for lattice-site materialization.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LatticeSiteConservation {
    pub counts: LatticeSitePopulationCounts,
    pub dispositions: Vec<LatticeSiteDispositionRecord>,
    pub promotable: bool,
}

/// Reject malformed materialized collections before they become join authority.
pub fn validate_materialized_sites(sites: &[MaterializedSite]) -> Result<()> {
    let mut seen_materialized = std::collections::HashSet::with_capacity(sites.len());
    let mut seen_analyzed = std::collections::HashSet::with_capacity(sites.len());
    for site in sites {
        if !seen_materialized.insert(site.materialized_site_id) {
            anyhow::bail!(
                "{:?}: materialized site id {:?} assigned twice",
                SiteUnmatchedReason::DuplicateMaterializedId,
                site.materialized_site_id
            );
        }
        if !seen_analyzed.insert(site.analyzed_site_identity) {
            anyhow::bail!(
                "{:?}: analyzed identity {:?} materialized twice",
                SiteUnmatchedReason::DuplicateMaterializedId,
                site.analyzed_site_identity
            );
        }
        for axis in 0..3 {
            if !site.aabb_min[axis].is_finite()
                || !site.aabb_max[axis].is_finite()
                || !site.centroid[axis].is_finite()
                || site.aabb_min[axis] > site.aabb_max[axis]
                || site.centroid[axis] < site.aabb_min[axis]
                || site.centroid[axis] > site.aabb_max[axis]
            {
                anyhow::bail!(
                    "{:?}: materialized site {:?} has invalid geometry on axis {axis}",
                    SiteUnmatchedReason::InvalidSourceRegion,
                    site.materialized_site_id
                );
            }
        }
    }
    Ok(())
}

/// Build secondary lattice binding-site projections without crossing into the
/// CCL identity domain.
pub fn project_lattice_binding_sites(
    sites: &[MaterializedSite],
) -> Result<Vec<LatticeBindingSiteProjection>> {
    validate_materialized_sites(sites)?;
    sites
        .iter()
        .map(|site| {
            let AnalyzedSiteIdentity::LatticeComponent(source_component_id) =
                site.analyzed_site_identity
            else {
                anyhow::bail!(
                    "{:?}: materialized site {:?} is not a lattice component",
                    SiteUnmatchedReason::ConsumerDomainMismatch,
                    site.materialized_site_id
                );
            };
            if site.source_surface != MaterializedSiteSurface::LatticeComponent {
                anyhow::bail!(
                    "{:?}: materialized site {:?} has non-lattice surface {:?}",
                    SiteUnmatchedReason::ConsumerDomainMismatch,
                    site.materialized_site_id,
                    site.source_surface
                );
            }
            Ok(LatticeBindingSiteProjection {
                materialized_site_id: site.materialized_site_id,
                analyzed_site_identity: site.analyzed_site_identity,
                source_surface: site.source_surface,
                source_component_id,
                aabb_min: site.aabb_min,
                aabb_max: site.aabb_max,
                centroid: site.centroid,
                materialization_provenance: site.provenance.clone(),
            })
        })
        .collect()
}

/// Materialize selected lattice components into canonical sites.
///
/// Each component receives an independently assigned `MaterializedSiteId` in
/// selection order. The identity is *assigned*, never derived by reusing the
/// component index as a cluster id — a lattice component and a CCL cluster may
/// carry the same integer and mean unrelated things, so the two domains stay
/// separate and are reconciled only through `AnalyzedSiteIdentity`.
///
/// `selected` is `(component_index, aabb_min, aabb_max)` in the order returned
/// by [`select_lattice_components`].
pub fn materialize_lattice_sites(
    selected: &[(usize, [f32; 3], [f32; 3])],
) -> Result<Vec<MaterializedSite>> {
    let mut sites = Vec::with_capacity(selected.len());
    for (slot, (index, aabb_min, aabb_max)) in selected.iter().enumerate() {
        for axis in 0..3 {
            if !aabb_min[axis].is_finite() || !aabb_max[axis].is_finite() {
                anyhow::bail!(
                    "lattice component {index}: {:?} AABB axis {axis} is non-finite",
                    SiteUnmatchedReason::InvalidSourceRegion
                );
            }
            if aabb_min[axis] > aabb_max[axis] {
                anyhow::bail!(
                    "lattice component {index}: {:?} AABB axis {axis} inverted: {} > {}",
                    SiteUnmatchedReason::InvalidSourceRegion,
                    aabb_min[axis],
                    aabb_max[axis]
                );
            }
        }
        let materialized_site_id = MaterializedSiteId(u32::try_from(slot).map_err(|_| {
            anyhow::anyhow!("more lattice components than the materialized-site domain admits")
        })?);
        sites.push(MaterializedSite {
            materialized_site_id,
            analyzed_site_identity: AnalyzedSiteIdentity::LatticeComponent(LatticeComponentId(
                *index as u64,
            )),
            source_surface: MaterializedSiteSurface::LatticeComponent,
            aabb_min: *aabb_min,
            aabb_max: *aabb_max,
            // Compatibility geometry: center of the component AABB. Native
            // observation materialization replaces this with node centroids.
            centroid: [
                0.5 * (aabb_min[0] + aabb_max[0]),
                0.5 * (aabb_min[1] + aabb_max[1]),
                0.5 * (aabb_min[2] + aabb_max[2]),
            ],
            provenance: "prism-nhs::sdst_bridge::materialize_lattice_sites".to_string(),
        });
    }

    validate_materialized_sites(&sites)?;
    Ok(sites)
}

/// Materialize selected lattice sites using the observed node centroids.
/// Weights match the native phase materializer: AABB volume, or unit weight
/// for a valid zero-volume node. The AABB center is never used as a fallback.
pub fn materialize_lattice_sites_with_node_centroids(
    selected: &[(usize, [f32; 3], [f32; 3])],
    nodes: &[crate::ghost_phase_lattice::GhostPhaseLatticeNode],
    components: &[crate::ghost_phase_lattice::GhostPhaseLatticeComponent],
) -> Result<Vec<MaterializedSite>> {
    let mut sites = materialize_lattice_sites(selected)?;
    for (site, (component_index, _, _)) in sites.iter_mut().zip(selected) {
        let component = components.get(*component_index).ok_or_else(|| {
            anyhow::anyhow!("selected lattice component {component_index} is absent")
        })?;
        if component.node_indices.is_empty() {
            anyhow::bail!("selected lattice component {component_index} has no observed nodes");
        }
        let mut weighted_sum = [0.0f64; 3];
        let mut total_weight = 0.0f64;
        let mut observed_indices = std::collections::HashSet::new();
        for &index in &component.node_indices {
            if !observed_indices.insert(index) {
                anyhow::bail!("component {component_index} repeats node {index}");
            }
            let node = nodes.get(index as usize).ok_or_else(|| {
                anyhow::anyhow!("component {component_index} references absent node {index}")
            })?;
            let mut volume = 1.0f64;
            for axis in 0..3 {
                let low = node.aabb_min[axis];
                let high = node.aabb_max[axis];
                let center = node.centroid_xyz[axis];
                if !low.is_finite()
                    || !high.is_finite()
                    || !center.is_finite()
                    || low > high
                    || center < low
                    || center > high
                {
                    anyhow::bail!("component {component_index} node {index} has invalid observed geometry on axis {axis}");
                }
                volume *= f64::from(high) - f64::from(low);
            }
            let weight = if volume > 0.0 { volume } else { 1.0 };
            total_weight += weight;
            for axis in 0..3 {
                weighted_sum[axis] += weight * f64::from(node.centroid_xyz[axis]);
            }
        }
        site.centroid = weighted_sum.map(|sum| (sum / total_weight) as f32);
        site.provenance = "GhostPhaseLatticeNode.centroid_xyz;component.node_indices;AABB_volume_weighted;zero_volume_unit_weight".to_string();
    }
    validate_materialized_sites(&sites)?;
    Ok(sites)
}

pub fn protocol_covering_observed_steps(
    protocol: &CryoUvProtocol,
    last_step: i32,
) -> Result<CryoUvProtocol> {
    anyhow::ensure!(last_step >= 0, "SDST cannot admit a negative observed step");
    let mut view = protocol.clone();
    if protocol.ramp_down_steps == 0 && protocol.cold_return_steps == 0 {
        let warm_start = protocol
            .cold_hold_steps
            .checked_add(protocol.ramp_steps)
            .ok_or_else(|| anyhow::anyhow!("SDST phase boundary overflow"))?;
        let end = last_step
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("SDST terminal event boundary overflow"))?;
        view.warm_hold_steps = view.warm_hold_steps.max(end.saturating_sub(warm_start));
    } else {
        // The recorded terminal phase is cold after an actual cooling schedule.
        // Extend only its analysis horizon; every preceding boundary and the
        // execution protocol remain unchanged. Native TIDE uses an exclusive end.
        let cold_start=protocol.cold_hold_steps.checked_add(protocol.ramp_steps)
            .and_then(|v|v.checked_add(protocol.warm_hold_steps))
            .and_then(|v|v.checked_add(protocol.ramp_down_steps))
            .ok_or_else(||anyhow::anyhow!("SDST cold phase boundary overflow"))?;
        let end=last_step.checked_add(1).ok_or_else(||anyhow::anyhow!("SDST terminal event boundary overflow"))?;
        view.cold_return_steps=view.cold_return_steps.max(end.saturating_sub(cold_start));
    }
    Ok(view)
}

#[cfg(test)]
mod observed_protocol_tests {
    use super::*;

    #[test]
    fn standard_tail_remains_warm_in_sdst_config_and_insertion_protocol() {
        let mut protocol = CryoUvProtocol::standard();
        protocol.ramp_down_steps = 0;
        protocol.cold_return_steps = 0;
        protocol.start_temp = 73.5;
        protocol.end_temp = 297.25;
        let original = protocol.clone();
        let mut view = protocol_covering_observed_steps(&protocol, 1_234_567).unwrap();
        assert_eq!(protocol, original);
        assert_eq!(view.cold_hold_steps, original.cold_hold_steps);
        assert_eq!(view.ramp_steps, original.ramp_steps);
        assert_eq!(view.ramp_down_steps, 0);
        assert_eq!(view.cold_return_steps, 0);
        assert!(view.cold_hold_steps + view.ramp_steps + view.warm_hold_steps > 1_234_567);
        view.current_step = 1_234_567;
        assert_eq!(view.current_temperature(), 297.25);
    }

    #[test]
    fn captured_adaptive_grid_center_rejects_default_and_corrupt_geometry() {
        // Actual retained4OBE_A stream0 PRSPK record0 (step6, voxel1010).
        // Production acceptance also checks every original record before CUDA.
        let geometry=SdstGridGeometry{dimension:64,spacing_a:0.8641069531440735,
            origin_a:[-33.18195343017578,-48.21705627441406,21.028905868530273]};
        let mut event=GpuSpikeEvent::default();event.timestep=6;event.voxel_idx=1010;
        event.position=[10.455448150634766,-34.82339859008789,21.46095848083496];
        geometry.validate_recorded_center(&event).unwrap();
        assert!(SdstGridGeometry{dimension:128,..geometry}.validate_recorded_center(&event).is_err());
        assert!(SdstGridGeometry{dimension:0,..geometry}.validate().is_err());
        assert!(SdstGridGeometry{spacing_a:f32::NAN,..geometry}.validate().is_err());
        event.position[0]+=1.0;assert!(geometry.validate_recorded_center(&event).is_err());
        event.voxel_idx=64*64*64;assert!(geometry.validate_recorded_center(&event).is_err());
    }

    #[test]
    fn real_cooling_schedule_extends_only_terminal_cold_analysis_horizon() {
        let protocol=CryoUvProtocol::standard().with_hysteresis();let original=protocol.clone();
        let mut view=protocol_covering_observed_steps(&protocol,1_234_567).unwrap();
        assert_eq!(protocol,original);
        assert_eq!((view.cold_hold_steps,view.ramp_steps,view.warm_hold_steps,view.ramp_down_steps),
            (original.cold_hold_steps,original.ramp_steps,original.warm_hold_steps,original.ramp_down_steps));
        assert_eq!(view.cold_hold_steps+view.ramp_steps+view.warm_hold_steps+view.ramp_down_steps+view.cold_return_steps,1_234_568);
        view.current_step=1_234_567;assert_eq!(view.current_temperature(),original.start_temp);
        assert!(protocol_covering_observed_steps(&protocol,-1).is_err());
        assert!(protocol_covering_observed_steps(&protocol,i32::MAX).is_err());
    }
}

const GRID_DIM: usize = 128;
const GRID_SPACING: f32 = 0.75;
const GRID_PADDING: f32 = 5.0;
const HYSTERESIS_THRESHOLD: f32 = 0.2;
/// Painting radius (voxels) for residue map construction.
/// 2 voxels × 0.75Å = 1.5Å — covers van der Waals radius of most atoms.
const RESMAP_PAINT_RADIUS: i32 = 2;
/// Maximum TIDE residues to report per site (top by transfer entropy).
const TIDE_TOP_RESIDUES: usize = 20;

// ---------------------------------------------------------------------------
// Output structs (serialized into the binding_sites.json "prism_therm" key)
// ---------------------------------------------------------------------------

fn ccns_class_name(cls: CcnsClass) -> &'static str {
    match cls {
        CcnsClass::Soc => "Soc",
        CcnsClass::NearCritical => "NearCritical",
        CcnsClass::Barrier => "Barrier",
    }
}

/// A native SDST analysis stage whose outcome is recorded in materialized output.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SdstAnalysisStage {
    /// Read the number of events resident in SDST.
    EventCount,
    /// Batched critical-cluster analysis for the requested site regions.
    Ccns,
    /// Per-site heating/cooling hysteresis analysis.
    Hysteresis,
    /// Per-site causal thermodynamic decomposition.
    Tide,
    /// GPU-native whole-grid pocket scan.
    GlobalScan,
    /// Full-simulation avalanche statistics.
    AvalancheStats,
    /// Merge of independently materialized per-group analyses.
    Merge,
}

/// Stable identity of a lattice component before any site materialization.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, Hash)]
pub struct LatticeComponentId(pub u64);

/// Stable identity assigned only when a routed region is materialized as a site.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, Hash)]
pub struct MaterializedSiteId(pub u32);

/// Domain-preserving identity of the object submitted for SDST analysis.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, Hash)]
#[serde(tag = "domain", content = "id", rename_all = "snake_case")]
pub enum AnalyzedSiteIdentity {
    /// Existing clustered-site identity.
    Cluster(ClusterId),
    /// Offline spike-density materializer region; never a CCL or lattice ID.
    MaterializerRegion(u32),
    /// Pre-materialization lattice-component identity.
    LatticeComponent(LatticeComponentId),
}

/// Explicit region request for domain-preserving SDST analysis.
#[derive(Debug, Clone, Copy)]
pub struct SdstRegionRequest {
    /// Identity in its source domain; never inferred from materialized identity.
    pub analyzed_site_identity: AnalyzedSiteIdentity,
    /// Optional independently assigned materialized-site identity.
    pub materialized_site_id: Option<MaterializedSiteId>,
    /// Voxel-space region submitted to SDST.
    pub region: SpatialRegion,
}

impl PartialEq for SdstRegionRequest {
    fn eq(&self, other: &Self) -> bool {
        self.analyzed_site_identity == other.analyzed_site_identity
            && self.materialized_site_id == other.materialized_site_id
            && self.region.x_min == other.region.x_min
            && self.region.x_max == other.region.x_max
            && self.region.y_min == other.region.y_min
            && self.region.y_max == other.region.y_max
            && self.region.z_min == other.region.z_min
            && self.region.z_max == other.region.z_max
    }
}

impl Eq for SdstRegionRequest {}

/// Typed cause class for an SDST stage that did not produce an observation.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SdstFailureKind {
    /// The native producer returned an error whose upstream taxonomy is unavailable here.
    Unclassified,
    /// A native producer violated its declared result-shape contract.
    ContractViolation,
    /// The stage was not run because a required earlier stage failed.
    PrerequisiteFailed,
}

/// Exact producer path for one SDST stage outcome.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct SdstStageProvenance {
    /// Rust producer that captured and serialized the outcome.
    pub producer: String,
    /// Highest-order native backend selected by this bridge.
    pub backend: String,
    /// Concrete SDST operation invoked for the stage.
    pub operation: String,
}

impl SdstStageProvenance {
    fn native(operation: &str) -> Self {
        Self {
            producer: "prism-nhs::sdst_bridge".to_string(),
            backend: "native_sdst_cuda".to_string(),
            operation: operation.to_string(),
        }
    }
}

/// Typed failure evidence emitted instead of a scientific default value.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct SdstStageFailure {
    /// Stable failure class suitable for machine adjudication.
    pub kind: SdstFailureKind,
    /// Native error text retained for diagnosis and evidence review.
    pub message: String,
}

/// Materialized outcome for one native SDST stage.
///
/// `Observed` is used even when the observed scientific value is zero or an
/// empty collection. `Failed` and `Blocked` never carry a scientific value.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum SdstStageOutcome {
    /// The native producer returned an observation, including valid zero/empty values.
    Observed {
        /// Stage that produced the observation.
        stage: SdstAnalysisStage,
        /// Site scope for per-site stages.
        #[serde(skip_serializing_if = "Option::is_none")]
        site_id: Option<i32>,
        /// Input analysis index after a per-group merge.
        #[serde(skip_serializing_if = "Option::is_none")]
        source_analysis_index: Option<usize>,
        /// Exact native producer and operation.
        provenance: SdstStageProvenance,
    },
    /// The stage executed but its native producer returned no observation.
    Failed {
        /// Stage that failed.
        stage: SdstAnalysisStage,
        /// Site scope for per-site stages.
        #[serde(skip_serializing_if = "Option::is_none")]
        site_id: Option<i32>,
        /// Input analysis index after a per-group merge.
        #[serde(skip_serializing_if = "Option::is_none")]
        source_analysis_index: Option<usize>,
        /// Exact native producer and operation.
        provenance: SdstStageProvenance,
        /// Typed failure evidence.
        failure: SdstStageFailure,
    },
    /// The stage was not executed because its prerequisite did not produce an observation.
    Blocked {
        /// Stage that was blocked.
        stage: SdstAnalysisStage,
        /// Site scope for per-site stages.
        #[serde(skip_serializing_if = "Option::is_none")]
        site_id: Option<i32>,
        /// Input analysis index after a per-group merge.
        #[serde(skip_serializing_if = "Option::is_none")]
        source_analysis_index: Option<usize>,
        /// Intended native producer and operation.
        provenance: SdstStageProvenance,
        /// Typed prerequisite failure evidence.
        failure: SdstStageFailure,
    },
}

impl SdstStageOutcome {
    fn observed(stage: SdstAnalysisStage, site_id: Option<i32>, operation: &str) -> Self {
        Self::Observed {
            stage,
            site_id,
            source_analysis_index: None,
            provenance: SdstStageProvenance::native(operation),
        }
    }

    fn failed(
        stage: SdstAnalysisStage,
        site_id: Option<i32>,
        operation: &str,
        kind: SdstFailureKind,
        message: String,
    ) -> Self {
        Self::Failed {
            stage,
            site_id,
            source_analysis_index: None,
            provenance: SdstStageProvenance::native(operation),
            failure: SdstStageFailure { kind, message },
        }
    }

    fn blocked(
        stage: SdstAnalysisStage,
        site_id: Option<i32>,
        operation: &str,
        message: String,
    ) -> Self {
        Self::Blocked {
            stage,
            site_id,
            source_analysis_index: None,
            provenance: SdstStageProvenance::native(operation),
            failure: SdstStageFailure {
                kind: SdstFailureKind::PrerequisiteFailed,
                message,
            },
        }
    }

    fn is_problem(&self) -> bool {
        matches!(self, Self::Failed { .. } | Self::Blocked { .. })
    }

    fn is_failed_stage(&self, expected: SdstAnalysisStage) -> bool {
        matches!(
            self,
            Self::Failed { stage, .. } | Self::Blocked { stage, .. } if *stage == expected
        )
    }

    fn with_source_analysis_index(mut self, index: usize) -> Self {
        match &mut self {
            Self::Observed {
                source_analysis_index,
                ..
            }
            | Self::Failed {
                source_analysis_index,
                ..
            }
            | Self::Blocked {
                source_analysis_index,
                ..
            } => *source_analysis_index = Some(index),
        }
        self
    }
}

/// Aggregate truth status for one materialized SDST analysis.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SdstAnalysisStatus {
    /// Every requested native stage returned an observation.
    Complete,
    /// Some scientific observations were materialized and at least one stage failed.
    Partial,
    /// Required event/site observations were not materialized.
    Failed,
}

/// Fail-closed, serializable provenance and stage-outcome ledger.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct SdstAnalysisOutcome {
    /// Aggregate status derived from stage outcomes and materialized site rows.
    pub status: SdstAnalysisStatus,
    /// Number of site regions requested by the caller.
    pub requested_site_count: usize,
    /// Number of complete site rows safe for downstream scientific consumption.
    pub materialized_site_count: usize,
    /// Ordered native stage evidence, including observed zero/empty results.
    pub stages: Vec<SdstStageOutcome>,
}

/// Serializable fail-closed error returned when any required SDST stage fails.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct SdstAnalysisError {
    /// Stage/failure/provenance record safe to materialize before propagation.
    pub outcome: SdstAnalysisOutcome,
}

impl std::fmt::Display for SdstAnalysisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SDST analysis failed with {} materialized of {} requested sites",
            self.outcome.materialized_site_count, self.outcome.requested_site_count
        )
    }
}

impl std::error::Error for SdstAnalysisError {}

impl SdstAnalysisOutcome {
    fn from_stages(
        requested_site_count: usize,
        materialized_site_count: usize,
        stages: Vec<SdstStageOutcome>,
    ) -> Self {
        let has_problem = stages.iter().any(SdstStageOutcome::is_problem);
        let missing_required_observation = stages
            .iter()
            .any(|stage| stage.is_failed_stage(SdstAnalysisStage::EventCount))
            || (requested_site_count > 0 && materialized_site_count == 0);
        let status = if missing_required_observation {
            SdstAnalysisStatus::Failed
        } else if has_problem {
            SdstAnalysisStatus::Partial
        } else {
            SdstAnalysisStatus::Complete
        };
        Self {
            status,
            requested_site_count,
            materialized_site_count,
            stages,
        }
    }
}

fn capture_stage<T, E: std::fmt::Debug>(
    stage: SdstAnalysisStage,
    site_id: Option<i32>,
    operation: &str,
    result: std::result::Result<T, E>,
) -> (Option<T>, SdstStageOutcome) {
    match result {
        Ok(value) => (
            Some(value),
            SdstStageOutcome::observed(stage, site_id, operation),
        ),
        Err(error) => (
            None,
            SdstStageOutcome::failed(
                stage,
                site_id,
                operation,
                SdstFailureKind::Unclassified,
                format!("{error:?}"),
            ),
        ),
    }
}

fn legacy_cluster_request(cluster_id: i32, region: SpatialRegion) -> Result<SdstRegionRequest> {
    let cluster_id = u32::try_from(cluster_id)
        .map(ClusterId)
        .map_err(|_| anyhow::anyhow!("negative legacy cluster id {cluster_id}"))?;
    Ok(SdstRegionRequest {
        analyzed_site_identity: AnalyzedSiteIdentity::Cluster(cluster_id),
        materialized_site_id: None,
        region,
    })
}

impl PrismThermSiteResult {
    /// True when this result belongs to the given materialized site.
    ///
    /// The join every downstream consumer must use. Comparing the legacy `i32`
    /// `site_id` instead silently crosses identity domains: a lattice component
    /// and a CCL cluster can carry the same integer, and an unmapped result
    /// carries `-1`, which matches nothing and drops the science without a word.
    pub fn matches_materialized(&self, id: MaterializedSiteId) -> bool {
        self.materialized_site_id == Some(id)
    }

    /// True when this result belongs to the given analyzed source object.
    pub fn matches_analyzed(&self, identity: AnalyzedSiteIdentity) -> bool {
        self.analyzed_site_identity == identity
    }
}

/// Resolve at most one therm result in the materialized-site domain.
///
/// Duplicate rows are rejected instead of allowing `.find()` order to choose a
/// scientific result silently.
pub fn unique_materialized_result<'a>(
    results: &'a [PrismThermSiteResult],
    id: MaterializedSiteId,
) -> Result<Option<&'a PrismThermSiteResult>> {
    let mut matches = results
        .iter()
        .filter(|result| result.matches_materialized(id));
    let first = matches.next();
    if matches.next().is_some() {
        anyhow::bail!(
            "{:?}: multiple therm rows claim materialized site {:?}",
            SiteUnmatchedReason::DuplicateMaterializedId,
            id
        );
    }
    Ok(first)
}

/// Resolve at most one therm result in the analyzed-source identity domain.
pub fn unique_analyzed_result<'a>(
    results: &'a [PrismThermSiteResult],
    identity: AnalyzedSiteIdentity,
) -> Result<Option<&'a PrismThermSiteResult>> {
    let mut matches = results
        .iter()
        .filter(|result| result.matches_analyzed(identity));
    let first = matches.next();
    if matches.next().is_some() {
        anyhow::bail!(
            "{:?}: multiple therm rows claim analyzed identity {:?}",
            SiteUnmatchedReason::DuplicateMaterializedId,
            identity
        );
    }
    Ok(first)
}

/// Partition every analyzed lattice site into exactly one terminal disposition.
///
/// The analyzed population is the validated materialized-site collection: each
/// record carries the source region needed to preserve honest unmatched rows.
/// Results are joined in typed domains only. A result with the same integer in
/// the CCL domain is unrelated and cannot satisfy a lattice row.
pub fn reconcile_lattice_site_population(
    selected_site_count: usize,
    materialized_sites: &[MaterializedSite],
    analysis: &PrismThermAnalysis,
    requirement: LatticeMaterializationRequirement,
    failed_consumer: &str,
) -> Result<LatticeSiteConservation> {
    if failed_consumer.trim().is_empty() {
        anyhow::bail!("lattice reconciliation requires a named consumer or integration seam");
    }
    let projections = project_lattice_binding_sites(materialized_sites)?;
    if selected_site_count != projections.len() {
        anyhow::bail!(
            "selected_site_count {selected_site_count} does not equal materialized analyzed_site_count {}",
            projections.len()
        );
    }

    // A lattice result outside the supplied source population cannot produce a
    // geometry-complete terminal row. Refuse it instead of discarding it.
    for result in &analysis.sites {
        if matches!(
            result.analyzed_site_identity,
            AnalyzedSiteIdentity::LatticeComponent(_)
        ) && !materialized_sites.iter().any(|site| {
            result.matches_analyzed(site.analyzed_site_identity)
                || result.matches_materialized(site.materialized_site_id)
        }) {
            anyhow::bail!(
                "{:?}: lattice result {:?} has no source MaterializedSite geometry",
                SiteUnmatchedReason::NoMaterializationMapping,
                result.analyzed_site_identity
            );
        }
    }

    let mut dispositions = Vec::with_capacity(projections.len());
    for projection in &projections {
        let materialized_matches: Vec<&PrismThermSiteResult> = analysis
            .sites
            .iter()
            .filter(|result| result.matches_materialized(projection.materialized_site_id))
            .collect();
        let analyzed_matches: Vec<&PrismThermSiteResult> = analysis
            .sites
            .iter()
            .filter(|result| result.matches_analyzed(projection.analyzed_site_identity))
            .collect();

        let (disposition, unmatched_reason, hysteresis_positive) =
            if materialized_matches.len() > 1 || analyzed_matches.len() > 1 {
                (
                    LatticeSiteTerminalDisposition::Unmatched,
                    Some(SiteUnmatchedReason::DuplicateMaterializedId),
                    false,
                )
            } else if let Some(result) = materialized_matches.first() {
                if result.matches_analyzed(projection.analyzed_site_identity) {
                    (
                        LatticeSiteTerminalDisposition::Merged,
                        None,
                        result.is_hysteretic,
                    )
                } else {
                    (
                        LatticeSiteTerminalDisposition::Unmatched,
                        Some(SiteUnmatchedReason::ConsumerDomainMismatch),
                        false,
                    )
                }
            } else if let Some(result) = analyzed_matches.first() {
                let reason = if result.materialized_site_id.is_none() {
                    SiteUnmatchedReason::NoMaterializationMapping
                } else {
                    SiteUnmatchedReason::ConsumerDomainMismatch
                };
                (
                    LatticeSiteTerminalDisposition::Unmatched,
                    Some(reason),
                    false,
                )
            } else {
                (
                    LatticeSiteTerminalDisposition::Unmatched,
                    Some(SiteUnmatchedReason::MissingCompanionData),
                    false,
                )
            };

        let promotable = disposition == LatticeSiteTerminalDisposition::Merged
            || requirement == LatticeMaterializationRequirement::Optional;
        dispositions.push(LatticeSiteDispositionRecord {
            disposition,
            materialized_site_id: Some(projection.materialized_site_id),
            analyzed_site_identity: projection.analyzed_site_identity,
            source_surface: projection.source_surface,
            source_component_id: projection.source_component_id,
            centroid: projection.centroid,
            aabb_min: projection.aabb_min,
            aabb_max: projection.aabb_max,
            unmatched_reason,
            failed_consumer: unmatched_reason.map(|_| failed_consumer.to_string()),
            promotable,
            hysteresis_positive,
        });
    }

    let merged_site_count = dispositions
        .iter()
        .filter(|row| row.disposition == LatticeSiteTerminalDisposition::Merged)
        .count();
    let unmatched_site_count = dispositions
        .iter()
        .filter(|row| row.disposition == LatticeSiteTerminalDisposition::Unmatched)
        .count();
    let hysteresis_positive_count = dispositions
        .iter()
        .filter(|row| row.hysteresis_positive)
        .count();
    let counts = LatticeSitePopulationCounts {
        selected_site_count,
        analyzed_site_count: dispositions.len(),
        merged_site_count,
        unmatched_site_count,
        hysteresis_positive_count,
    };
    if counts.analyzed_site_count != counts.merged_site_count + counts.unmatched_site_count {
        anyhow::bail!("lattice conservation invariant violated: analyzed != merged + unmatched");
    }
    if counts.hysteresis_positive_count > counts.merged_site_count {
        anyhow::bail!("lattice hysteresis population exceeds the merged identity population");
    }

    // A non-empty analyzed population with zero usable merges is a join or
    // materialization defect, never a successful optional-profile outcome.
    // The analyzed==0 case remains separately representable for profiles that
    // explicitly permit a legitimate zero-site selection.
    let has_usable_result_or_is_vacuous =
        counts.analyzed_site_count == 0 || counts.merged_site_count > 0;
    Ok(LatticeSiteConservation {
        counts,
        promotable: has_usable_result_or_is_vacuous
            && dispositions.iter().all(|row| row.promotable),
        dispositions,
    })
}

/// Per-residue TIDE (causal ΔG decomposition) result.
#[derive(Debug, Clone, Serialize)]
pub struct TideResidueResult {
    pub residue_id: u32,
    /// Transfer entropy-weighted ΔG contribution (kcal/mol proxy).
    /// Negative = stabilizing pocket, positive = destabilizing.
    pub causal_dg: f32,
    /// Transfer entropy: causal information flow from this residue to pocket.
    /// Higher = more causal influence. Drug design leverage point.
    pub transfer_entropy: f32,
    /// Fisher information: sensitivity of pocket to perturbations at this residue.
    /// High Fisher info = high-leverage mutation/modification target.
    pub fisher_info: f32,
    /// KL divergence between heating/cooling spike distributions.
    /// High KL = conformational reorganization cost for this residue.
    pub kl_divergence: f32,
    /// Residue-attributed event count inside the strict ROI; not parent-edge count.
    pub n_causal_spikes: u32,
    /// Exact producer bin occupancy, estimator support, and unavailability receipt.
    pub measurement_support: serde_json::Value,

}

/// PRISM-Therm thermodynamic site classification.
///
/// Multi-signal composite: relative asymmetry (vs protein mean),
/// CCNS criticality class, TIDE causal depth, avalanche statistics.
/// NOT a flat threshold — uses z-score normalization against the
/// protein-wide hysteresis baseline.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub enum ThermClass {
    /// Confirmed cryptic: relative asymmetry > 1.5 sigma above mean,
    /// AND (tau < 1.5 SOC OR TIDE top-residue TE > 2x median).
    Cryptic,
    /// Dynamic hotspot: relative asymmetry > 0.5 sigma above mean.
    Dynamic,
    /// Thermally responsive: asymmetry within +/-0.5 sigma of mean.
    Responsive,
    /// Thermally inert: relative asymmetry < -0.5 sigma below mean.
    Inert,
}

impl std::fmt::Display for ThermClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ThermClass::Cryptic => write!(f, "CRYPTIC"),
            ThermClass::Dynamic => write!(f, "DYNAMIC"),
            ThermClass::Responsive => write!(f, "RESPONSIVE"),
            ThermClass::Inert => write!(f, "INERT"),
        }
    }
}

/// Per-site PRISM-Therm analysis result.
#[derive(Debug, Clone, Serialize)]
pub struct PrismThermSiteResult {
    /// Domain-preserving source identity used by all new joins.
    pub analyzed_site_identity: AnalyzedSiteIdentity,
    /// Independent materialized-site mapping, when one was assigned.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub materialized_site_id: Option<MaterializedSiteId>,
    /// Compatibility projection for existing clustered-site consumers.
    /// New code must join on `analyzed_site_identity`; lattice-component IDs
    /// are never encoded into this field.
    pub site_id: i32,
    /// Heating/cooling spike rate asymmetry in [0, 1].
    /// > 0.2 typically indicates a cryptic (hysteretic) pocket.
    pub asymmetry_score: f32,
    pub is_hysteretic: bool,
    pub heating_spike_rate: f32,
    pub cooling_spike_rate: f32,
    pub heating_spike_count: u32,
    pub cooling_spike_count: u32,
    /// CCNS power-law exponent. tau < 1.5 = SOC (most druggable).
    pub tau: f32,
    pub tau_stderr: f32,
    pub ccns_classification: String,
    pub druggability: f32,
    pub n_avalanches: u32,
    /// Asymmetry relative to the protein-wide mean.
    /// Positive = more hysteretic than average. >1.5 σ = CRYPTIC candidate.
    pub relative_asymmetry: f32,
    /// Multi-signal thermodynamic classification.
    pub therm_class: ThermClass,
    /// TIDE: top residues by transfer entropy (causal ΔG decomposition).
    /// Up to 20 residues, sorted by TE descending.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tide_decomposition: Vec<TideResidueResult>,
}

/// One globally-detected hysteretic pocket (from SDST's own scan).
#[derive(Debug, Clone, Serialize)]
pub struct PrismThermGlobalPocket {
    pub grid_region: [u32; 6], // [x_min, x_max, y_min, y_max, z_min, z_max]
    pub tau: f32,
    pub ccns_classification: String,
    pub druggability: f32,
    /// Approximate centroid in Ångstroms.
    pub centroid_angstrom: [f32; 3],
}

/// Complete PRISM-Therm output for one structure run.
#[derive(Debug, Clone, Serialize)]
pub struct PrismThermAnalysis {
    /// Truthful stage/failure/provenance outcome for this materialized analysis.
    pub outcome: SdstAnalysisOutcome,
    /// Total events stored in SDST.
    pub sdst_event_count: u32,
    /// Hysteresis threshold applied (default 0.2).
    pub hysteresis_threshold: f32,
    /// Number of NHS sites classified as hysteretic.
    pub hysteretic_site_count: usize,
    /// Per-NHS-site thermodynamic analysis.
    pub sites: Vec<PrismThermSiteResult>,
    /// Pockets found by SDST's own global scan (may include novel sites).
    pub global_pockets: Vec<PrismThermGlobalPocket>,
    /// Total unique avalanches detected across the full simulation.
    pub total_avalanches: usize,
    /// Number of residues mapped in the TIDE residue map.
    pub tide_residues_mapped: usize,
}

// ---------------------------------------------------------------------------
// SdstBridge
// ---------------------------------------------------------------------------

/// Exact recorded native cubic-grid interpretation for saved-event replay.
/// Cell-center checks qualify the reconstructed producer policy on every record.
#[derive(Debug,Clone,Copy,Serialize)]
pub struct SdstGridGeometry {
    pub dimension:u32,
    pub spacing_a:f32,
    pub origin_a:[f32;3],
}
impl SdstGridGeometry {
    pub fn validate(&self)->Result<()> {
        anyhow::ensure!(self.dimension>0 && self.dimension<=128,"native saved grid dimension outside cubic NHS ABI");
        anyhow::ensure!(self.spacing_a.is_finite() && self.spacing_a>0.0 && self.origin_a.iter().all(|v|v.is_finite()),"invalid native saved physical grid");
        Ok(())
    }
    pub fn from_recorded_dimension(topology:&PrismPrepTopology,dimension:u32)->Result<Self> {
        let floor=crate::persistent_engine::PersistentBatchConfig::default().grid_spacing;
        let (spacing_a,_)=topology.grid_spacing_for_coverage(dimension as usize,floor);
        let value=Self{dimension,spacing_a,origin_a:topology.grid_origin(GRID_PADDING)};
        value.validate()?;Ok(value)
    }
    pub fn validate_recorded_center(&self,event:&GpuSpikeEvent)->Result<()> {
        self.validate()?;let dim=self.dimension;
        anyhow::ensure!(event.voxel_idx>=0 && (event.voxel_idx as u32)<dim*dim*dim,"recorded voxel outside captured grid");
        let index=event.voxel_idx as u32;let xyz=[index%dim,(index/dim)%dim,index/(dim*dim)];
        for axis in 0..3 {
            let center=(xyz[axis] as f32+0.5).mul_add(self.spacing_a,self.origin_a[axis]);
            anyhow::ensure!(event.position[axis].to_bits()==center.to_bits(),"recorded voxel center differs from source-bound native grid on axis{axis}");
        }
        Ok(())
    }
}

/// PRISM-Therm integration bridge between the NHS engine and the SDST library.
///
/// Instantiate with the live `protocol` (after `with_hysteresis()` has been
/// applied) and the topology, so phase boundaries are always exact.
pub struct SdstBridge {
    sdst: Sdst,
    protocol: CryoUvProtocol,
    grid_origin: [f32; 3],
    /// Dense linear-voxel-indexed residue map (size = captured dimension³).
    /// Index = x + y*dimension + z*dimension². Value = residue_id or u32::MAX.
    residue_map: Vec<u32>,
    /// Total number of unique residues in the topology.
    n_residues: u32,
    /// Existing engine compatibility retains its top-20 view; offline consumers
    /// can request every native TIDE row without recomputing or substituting values.
    tide_output_limit: Option<usize>,
    native_config: SdstConfig,
}

impl SdstBridge {
    /// Create a bridge configured from the live protocol and topology.
    ///
    /// `expected_spike_count` is used to size the SDST event buffer with a
    /// 1.5× safety margin (minimum 2M events).
    pub fn new(
        topology: &PrismPrepTopology,
        protocol: &CryoUvProtocol,
        expected_spike_count: usize,
    ) -> Result<Self> {
        Self::new_with_wavefront_max_dt(topology, protocol, expected_spike_count, None)
    }

    /// Explicit policy override only. None preserves the native C default.
    pub fn new_with_wavefront_max_dt(topology: &PrismPrepTopology, protocol: &CryoUvProtocol,
        expected_spike_count: usize, wavefront_max_dt_md_steps: Option<u32>) -> Result<Self> {
        let geometry=SdstGridGeometry{dimension:GRID_DIM as u32,spacing_a:GRID_SPACING,origin_a:topology.grid_origin(GRID_PADDING)};
        Self::new_with_geometry(topology,protocol,expected_spike_count,wavefront_max_dt_md_steps,geometry)
    }

    fn new_with_geometry(topology:&PrismPrepTopology,protocol:&CryoUvProtocol,expected_spike_count:usize,
        wavefront_max_dt_md_steps:Option<u32>,geometry:SdstGridGeometry)->Result<Self> {
        geometry.validate()?;
        let grid_origin=geometry.origin_a;
        let mut config=Self::make_config(protocol,expected_spike_count);
        config.grid_nx=geometry.dimension;config.grid_ny=geometry.dimension;config.grid_nz=geometry.dimension;
        config.grid_spacing=geometry.spacing_a;
        if let Some(steps) = wavefront_max_dt_md_steps {
            anyhow::ensure!(steps <= (1 << 24), "wavefront MD-step window exceeds exact f32 integer range");
            config.wavefront_max_dt = steps as f32;
        }

        let sdst =
            Sdst::new(&config).map_err(|e| anyhow::anyhow!("SDST create failed: {:?}", e))?;

        let residue_map = Self::build_residue_map(topology, &grid_origin,geometry.dimension as usize,geometry.spacing_a);
        let n_residues = topology.n_residues as u32;

        let mapped_count = residue_map.iter().filter(|&&v| v != u32::MAX).count();
        log::info!(
            "  PRISM-Therm: residue map built — {} voxels mapped ({} residues)",
            mapped_count,
            n_residues
        );

        Ok(Self {
            sdst,
            protocol: protocol.clone(),
            grid_origin,
            residue_map,
            n_residues,
            tide_output_limit: Some(TIDE_TOP_RESIDUES),
            native_config: config,
        })
    }

    /// Preserve a no-cooling warm tail in SDST's finite phase-boundary ABI.
    /// The execution protocol stays unchanged; this analysis view covers every
    /// observed event through the inclusive last step.
    pub fn new_for_spikes(
        topology: &PrismPrepTopology,
        protocol: &CryoUvProtocol,
        spikes: &[GpuSpikeEvent],
    ) -> Result<Self> {
        Self::new_for_spikes_with_wavefront_max_dt(topology, protocol, spikes, None)
    }

    pub fn new_for_spikes_with_wavefront_max_dt(topology: &PrismPrepTopology, protocol: &CryoUvProtocol,
        spikes: &[GpuSpikeEvent], wavefront_max_dt_md_steps: Option<u32>) -> Result<Self> {
        let last_step = spikes.iter().map(|spike| spike.timestep).max().unwrap_or(0);
        let analysis_protocol = protocol_covering_observed_steps(protocol, last_step)?;
        Self::new_with_wavefront_max_dt(topology, &analysis_protocol, spikes.len(), wavefront_max_dt_md_steps)
    }

    /// Replay every source event under its validated captured grid geometry.
    pub fn new_for_spikes_with_recorded_grid(topology:&PrismPrepTopology,protocol:&CryoUvProtocol,
        spikes:&[GpuSpikeEvent],wavefront_max_dt_md_steps:Option<u32>,geometry:SdstGridGeometry)->Result<Self> {
        for event in spikes {geometry.validate_recorded_center(event)?;}
        let last_step=spikes.iter().map(|s|s.timestep).max().unwrap_or(0);
        let view=protocol_covering_observed_steps(protocol,last_step)?;
        Self::new_with_geometry(topology,&view,spikes.len(),wavefront_max_dt_md_steps,geometry)
    }

    /// Preserve all rows returned by the existing native TIDE implementation,
    /// including zero-support and nonfinite rows. No ranking/filter is introduced.
    pub fn preserve_full_tide_output(&mut self) {
        self.tide_output_limit = None;
    }

    /// Save native SDST state through its existing serializer. Event rows retain
    /// global buffer order; `parent_spike` is a one-based reference into that same
    /// buffer (zero means no parent). This avoids the legacy subgraph exporter,
    /// whose local parent indices mix index domains.
    pub fn save_native_state(&self, path: &std::path::Path) -> Result<serde_json::Value> {
        let path_text = path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("non-UTF8 SDST path"))?;
        self.sdst
            .save(path_text)
            .map_err(|e| anyhow::anyhow!("SDST save failed: {e:?}"))?;
        let cfg = &self.native_config;
        Ok(serde_json::json!({
            "producer": "sdst::Sdst::save / sdst_serialize.cu",
            "format_magic_u32": 0x54534453u32, "format_version": 1,
            "endianness": if cfg!(target_endian = "little") { "little" } else { "big" },
            "config_size_bytes": std::mem::size_of::<SdstConfig>(),
            "event_count_offset": 8 + std::mem::size_of::<SdstConfig>(),
            "event_buffer_offset": 12 + std::mem::size_of::<SdstConfig>(),
            "event_stride_bytes": std::mem::size_of::<sdst::SpikeEvent>(),
            "event_count": self.sdst.event_count().map_err(|e| anyhow::anyhow!("SDST event count: {e:?}"))?,
            "event_fields": {"morton_voxel":0,"timestamp_md_step":4,"parent_spike_global_one_based":8,
                "avalanche_id":12,"wavefront_id":16,"amplitude_u16":20,"local_temp_u16":22,
                "energy_gradient_u16":24,"solvent_exposure_u16":26,"wavefront_velocity_u16":28,
                "wavefront_coherence_u16":30,"phase_id_u8":32,"tcl_flags_u8":33},
            "configuration": {"grid": [cfg.grid_nx,cfg.grid_ny,cfg.grid_nz],
                "grid_spacing_a":cfg.grid_spacing,"grid_origin_a":self.grid_origin,
                "avalanche_spatial_cutoff_a":cfg.avalanche_spatial_cutoff,
                "avalanche_max_gap_md_steps":cfg.avalanche_max_gap,
                "phase_boundaries":cfg.phase_boundaries,"max_spike_events":cfg.max_spike_events,
                "max_wavefronts":cfg.max_wavefronts,"wavefront_stats_stride_bytes":std::mem::size_of::<sdst::WavefrontStats>(),
                "wavefront_max_dt_md_steps":cfg.wavefront_max_dt,
                "wavefront_capacity_policy":"one slot per admissible native event; no overflow count clipping",
                "ccns_soc_threshold":cfg.ccns_soc_threshold,"ccns_barrier_threshold":cfg.ccns_barrier_threshold},
            "effective_parent_selection": {"producer":"sdst_core.cu::sdst_insert_from_nhs_buffer -> kernel_detect_parents_sorted",
                "spatial_cutoff_voxels":3.0,"spatial_cutoff_a":3.0*cfg.grid_spacing,
                "spatial_cutoff_origin":"literal cutoff_grid=3.0f in bulk NHS insertion; configured avalanche_spatial_cutoff is not used by this path",
                "time_gap_md_steps":cfg.avalanche_max_gap},
            "edge_semantics": "existing GPU SDST parent selection: latest strictly earlier event in a different voxel within effective bulk-insertion radius3voxels and configured time gap; distance breaks time ties; algorithmic temporal propagation, not intervention identification",
            "edge_identity": "child global zero-based event row -> parent_spike-1; parent_spike=0 has no parent",
            "packed_measurement_semantics": {"local_temp":"protocol temperature times (1+0.5*(1-clamp(water_density,0,1))); derived spatial modulation, not measured per-voxel kinetic temperature",
                "energy_gradient":"recorded NHS wd_change water-density-change field, not electronic potential-energy gradient",
                "solvent_exposure":"recorded NHS water_density field; no explicit-water claim"},
            "legacy_subgraph_export": "not_invoked_due_to_mixed_global_and_local_parent_indices"
        }))
    }

    /// Adapt an explicitly identified offline materializer region using the
    /// observed bounds of its attributed spike positions. Zero physical extent
    /// is valid and maps through the existing floor/ceil voxel conversion.
    /// Source identity remains distinct from the native lattice component domain.
    pub fn materializer_region(
        &self,
        region_id: u32,
        aabb_min: [f32; 3],
        aabb_max: [f32; 3],
    ) -> Result<SdstRegionRequest> {
        let mut request = self.lattice_region(
            LatticeComponentId(region_id as u64),
            None,
            aabb_min,
            aabb_max,
        )?;
        request.analyzed_site_identity = AnalyzedSiteIdentity::MaterializerRegion(region_id);
        Ok(request)
    }

    /// Build SdstConfig from the live CryoUvProtocol.
    /// Phase boundaries come from the protocol; native clustering windows retain C defaults.
    fn make_config(protocol: &CryoUvProtocol, expected_spike_count: usize) -> SdstConfig {
        // Get base defaults from the C library (128³ grid, 0.75Å, 4M hash, etc.)
        let mut cfg: SdstConfig = unsafe { sdst::sdst_default_config() };

        // Grid: matches NHS engine exactly
        cfg.grid_nx = GRID_DIM as u32;
        cfg.grid_ny = GRID_DIM as u32;
        cfg.grid_nz = GRID_DIM as u32;
        cfg.grid_spacing = GRID_SPACING;

        // Phase boundaries: cumulative sums from the live protocol fields.
        // with_hysteresis() mirrors ramp and cold-hold into the cooling phases,
        // but we always read the actual field values — never assume defaults.
        let p0 = 0u32;
        let p1 = p0 + protocol.cold_hold_steps.max(0) as u32;
        let p2 = p1 + protocol.ramp_steps.max(0) as u32;
        let p3 = p2 + protocol.warm_hold_steps.max(0) as u32;
        let p4 = p3 + protocol.ramp_down_steps.max(0) as u32;
        let p5 = p4 + protocol.cold_return_steps.max(0) as u32;
        cfg.phase_boundaries = [p0, p1, p2, p3, p4, p5];

        // Event buffer: scale to actual spike count. At 36 bytes/event,
        // 42M events = 1.5GB -- safe ceiling for 16GB VRAM.
        cfg.max_spike_events = (expected_spike_count as u32).max(1_000_000).min(42_000_000);
        // One admitted event can create one wavefront. The native producer's
        // counter is not bounded by its former 65,536-slot default; smaller
        // storage silently loses assignments and makes serialization overread.
        // Reserve the complete possible population before ingestion, not by
        // clipping the measured count after the fact.
        cfg.max_wavefronts = cfg.max_spike_events;

        cfg
    }

    /// Build a dense linear-voxel-indexed residue map from the topology.
    ///
    /// For each protein atom, paints a sphere of radius `RESMAP_PAINT_RADIUS`
    /// voxels centered at the atom's grid position. When multiple atoms compete
    /// for a voxel, the closest atom (by Euclidean distance) wins.
    ///
    /// Size = admitted per-instance cubic grid population.
    /// Index = `x + y*dimension + z*dimension²`.
    /// Empty voxels = `u32::MAX`.
    fn build_residue_map(topology: &PrismPrepTopology, grid_origin: &[f32; 3],dim:usize,spacing:f32) -> Vec<u32> {
        let total = dim * dim * dim;
        let mut resmap = vec![u32::MAX; total];
        let mut dist_sq_map = vec![f32::MAX; total];

        let r = RESMAP_PAINT_RADIUS;

        for atom_idx in 0..topology.n_atoms {
            let px = topology.positions[atom_idx * 3];
            let py = topology.positions[atom_idx * 3 + 1];
            let pz = topology.positions[atom_idx * 3 + 2];
            let res_id = topology.residue_ids[atom_idx] as u32;

            // Atom's voxel coordinate (continuous → integer)
            let vx = ((px - grid_origin[0]) / spacing) as i32;
            let vy = ((py - grid_origin[1]) / spacing) as i32;
            let vz = ((pz - grid_origin[2]) / spacing) as i32;

            // Paint a (2r+1)³ cube, keeping only the closest atom per voxel
            for dz in -r..=r {
                let nz = vz + dz;
                if nz < 0 || nz >= dim as i32 {
                    continue;
                }
                for dy in -r..=r {
                    let ny = vy + dy;
                    if ny < 0 || ny >= dim as i32 {
                        continue;
                    }
                    for dx in -r..=r {
                        let nx = vx + dx;
                        if nx < 0 || nx >= dim as i32 {
                            continue;
                        }

                        let linear = nx as usize + ny as usize * dim + nz as usize * dim * dim;

                        // Euclidean distance² from voxel center to atom
                        let vox_x = nx as f32 * spacing + grid_origin[0];
                        let vox_y = ny as f32 * spacing + grid_origin[1];
                        let vox_z = nz as f32 * spacing + grid_origin[2];
                        let d2 = (vox_x - px).powi(2) + (vox_y - py).powi(2) + (vox_z - pz).powi(2);

                        if d2 < dist_sq_map[linear] {
                            dist_sq_map[linear] = d2;
                            resmap[linear] = res_id;
                        }
                    }
                }
            }
        }

        resmap
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Convert and insert accumulated NHS spike events into SDST.
    ///
    /// **GPU-native path**: Events are sorted by timestep on CPU, then the raw
    /// `GpuSpikeEvent` bytes are uploaded to GPU in one bulk transfer. A CUDA
    /// kernel converts NHS format → SDST SpikeEvent format directly on GPU.
    /// Insertion uses temporal batching (windows of `avalanche_max_gap` = 50
    /// timesteps) so parent detection chains stay short: O(N × local_density)
    /// instead of O(N × total_chain).
    ///
    /// **VRAM cap**: 1.5GB / 36 bytes ≈ 27.7M events. Beyond that, stride-sample.
    pub fn ingest_all_spikes(&self, spikes: &[GpuSpikeEvent]) -> Result<u32> {
        if spikes.is_empty() {
            return self
                .sdst
                .event_count()
                .map_err(|e| anyhow::anyhow!("SDST event_count: {:?}", e));
        }

        // Stride is derived from the record type and passed to the C API, so it
        // tracks the schema automatically (96 B under v1, 128 B under v2).
        // Must match the max_spike_events cap in build_config() to prevent
        // ErrorTableFull: 42M events × 36 bytes = 1.51 GB VRAM.
        const ABSOLUTE_MAX_EVENTS: usize = 42_000_000;

        // Sort by timestep on CPU (required for temporal batching in the C API)
        let sorted: Vec<GpuSpikeEvent> = if spikes.len() > ABSOLUTE_MAX_EVENTS {
            let stride = spikes.len() / ABSOLUTE_MAX_EVENTS;
            log::warn!(
                "  PRISM-Therm: {} spikes exceeds 1GB cap, stride-sampling to {}",
                spikes.len(),
                ABSOLUTE_MAX_EVENTS
            );
            let mut sampled: Vec<GpuSpikeEvent> = spikes
                .iter()
                .step_by(stride)
                .take(ABSOLUTE_MAX_EVENTS)
                .copied()
                .collect();
            sampled.sort_unstable_by_key(|s| s.timestep);
            sampled
        } else {
            log::info!(
                "  PRISM-Therm: ingesting all {} spikes into SDST (GPU-native)",
                spikes.len()
            );
            let mut v: Vec<GpuSpikeEvent> = spikes.to_vec();
            v.sort_unstable_by_key(|s| s.timestep);
            v
        };

        self.ingest_ordered_spikes(&sorted)
    }

    /// Offline lineage path. The callback sees the exact vector subsequently
    /// uploaded to CUDA and its original PRSPK row ordinals, before any drop.
    /// No second sorting/reconstruction of identities is permitted.
    pub fn ingest_spikes_with_lineage<F>(
        &self, spikes: &[GpuSpikeEvent], source_ordinals: &[u64], consume: F,
    ) -> Result<u32>
    where F: FnOnce(&[GpuSpikeEvent], &[u64]) -> Result<()> {
        anyhow::ensure!(spikes.len() == source_ordinals.len(), "source ordinal count mismatch");
        anyhow::ensure!(spikes.len() <= 42_000_000, "offline lineage forbids stride sampling");
        let mut permutation: Vec<usize> = (0..spikes.len()).collect();
        permutation.sort_unstable_by_key(|&i| spikes[i].timestep);
        let ordered: Vec<GpuSpikeEvent> = permutation.iter().map(|&i| spikes[i]).collect();
        let ordinals: Vec<u64> = permutation.iter().map(|&i| source_ordinals[i]).collect();
        consume(&ordered, &ordinals)?;
        self.ingest_ordered_spikes(&ordered)
    }

    fn ingest_ordered_spikes(&self, sorted: &[GpuSpikeEvent]) -> Result<u32> {
        // Reinterpret as raw bytes for the C API (GpuSpikeEvent is repr(C); the
        // stride is passed explicitly below, never assumed by the callee)
        let raw_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(sorted.as_ptr() as *const u8, sorted.len() * std::mem::size_of::<GpuSpikeEvent>())
        };

        // Protocol parameters for GPU-side phase/temperature computation
        let p = &self.protocol;
        let cold_hold = p.cold_hold_steps.max(0) as u32;
        let ramp_up = p.ramp_steps.max(0) as u32;
        let warm_hold = p.warm_hold_steps.max(0) as u32;
        let ramp_down = p.ramp_down_steps.max(0) as u32;

        self.sdst
            .insert_from_nhs_buffer(
                raw_bytes,
                std::mem::size_of::<GpuSpikeEvent>() as u32,
                p.start_temp,
                p.end_temp,
                cold_hold,
                ramp_up,
                warm_hold,
                ramp_down,
            )
            .map_err(|e| anyhow::anyhow!("SDST insert_from_nhs_buffer failed: {:?}", e))?;

        self.sdst
            .event_count()
            .map_err(|e| anyhow::anyhow!("SDST event_count: {:?}", e))
    }

    /// Run the full hysteresis + CCNS + TIDE analysis pipeline.
    ///
    /// 1. Per-site: `hysteresis_region` + `ccns_region` for each NHS site
    /// 2. Per-site: `tide_decomposition` — causal ΔG per residue (top 20 by TE)
    /// 3. Global:   `ccns_all_pockets` finds pockets SDST detected independently
    ///              (may reveal cryptic sites NHS missed or confirm NHS results)
    /// 4. Avalanche statistics summary
    pub fn analyze(&self, sites: &[ClusteredBindingSite]) -> Result<PrismThermAnalysis> {
        let mut requests = Vec::with_capacity(sites.len());
        for site in sites {
            match u32::try_from(site.cluster_id) {
                Ok(_) => {}
                Err(_) => {
                    return Err(anyhow::Error::new(SdstAnalysisError {
                        outcome: SdstAnalysisOutcome::from_stages(
                            sites.len(),
                            0,
                            vec![SdstStageOutcome::failed(
                                SdstAnalysisStage::Merge,
                                Some(site.cluster_id),
                                "ClusterId::try_from",
                                SdstFailureKind::ContractViolation,
                                format!(
                                    "negative legacy cluster id {} cannot be represented as ClusterId",
                                    site.cluster_id
                                ),
                            )],
                        ),
                    }));
                }
            }
            let gvm = site
                .view(crate::spatial_view::SpatialView::GeometricVoxelMass)
                .expect("ClusteredBindingSite always carries GeometricVoxelMass view");
            requests.push(legacy_cluster_request(
                site.cluster_id,
                self.centroid_to_region(&gvm),
            )?);
        }
        self.analyze_regions(&requests)
    }

    /// Analyze explicit typed regions without collapsing cluster, lattice, and
    /// materialized-site identity domains.
    pub fn analyze_regions(&self, requests: &[SdstRegionRequest]) -> Result<PrismThermAnalysis> {
        let mapped = self.residue_map.iter().filter(|&&v| v != u32::MAX).count();
        let mut stage_outcomes = Vec::new();
        let (event_count, event_count_outcome) = capture_stage(
            SdstAnalysisStage::EventCount,
            None,
            "Sdst::event_count",
            self.sdst.event_count(),
        );
        stage_outcomes.push(event_count_outcome);
        let Some(event_count) = event_count else {
            return Err(anyhow::Error::new(SdstAnalysisError {
                outcome: SdstAnalysisOutcome::from_stages(requests.len(), 0, stage_outcomes),
            }));
        };

        // Collect all site regions for batched GPU CCNS. PRISM-Therm
        // CCNS is evaluated at the GeometricVoxelMass view — the
        // spike-mass center of the cluster is the appropriate region
        // anchor for thermodynamic sampling. Other views (lining,
        // driver, hot/cold phase) represent different physical
        // neighborhoods; if we later want CCNS at those, the call
        // must pass the view explicitly.
        let regions: Vec<SpatialRegion> = requests.iter().map(|request| request.region).collect();

        // One batched GPU call for all per-site CCNS (CSN estimator, not Hill MLE).
        // A failure or shape mismatch produces no site rows: a default CcnsResult
        // is scientific-looking data and must never stand in for failed CUDA work.
        let ccns_results = match self.sdst.ccns_for_regions(&regions) {
            Ok(results) if results.len() == regions.len() => {
                stage_outcomes.push(SdstStageOutcome::observed(
                    SdstAnalysisStage::Ccns,
                    None,
                    "Sdst::ccns_for_regions",
                ));
                results
            }
            Ok(results) => {
                stage_outcomes.push(SdstStageOutcome::failed(
                    SdstAnalysisStage::Ccns,
                    None,
                    "Sdst::ccns_for_regions",
                    SdstFailureKind::ContractViolation,
                    format!(
                        "ccns_for_regions returned {} results for {} requested regions",
                        results.len(),
                        regions.len()
                    ),
                ));
                return Err(anyhow::Error::new(SdstAnalysisError {
                    outcome: SdstAnalysisOutcome::from_stages(requests.len(), 0, stage_outcomes),
                }));
            }
            Err(error) => {
                log::warn!(
                    "  PRISM-Therm: batched ccns_for_regions failed: {:?}; no default scientific rows will be emitted",
                    error
                );
                stage_outcomes.push(SdstStageOutcome::failed(
                    SdstAnalysisStage::Ccns,
                    None,
                    "Sdst::ccns_for_regions",
                    SdstFailureKind::Unclassified,
                    format!("{error:?}"),
                ));
                return Err(anyhow::Error::new(SdstAnalysisError {
                    outcome: SdstAnalysisOutcome::from_stages(requests.len(), 0, stage_outcomes),
                }));
            }
        };

        // Per-site analysis (hysteresis + TIDE) with pre-computed CCNS. A site
        // is materialized only if every constituent scientific stage was observed.
        let mut site_results: Vec<PrismThermSiteResult> = Vec::with_capacity(requests.len());
        for (i, request) in requests.iter().enumerate() {
            let (site_result, mut outcomes) = self.analyze_site(request, &ccns_results[i]);
            stage_outcomes.append(&mut outcomes);
            if let Some(site_result) = site_result {
                site_results.push(site_result);
            }
        }
        if site_results.len() != requests.len() {
            return Err(anyhow::Error::new(SdstAnalysisError {
                outcome: SdstAnalysisOutcome::from_stages(
                    requests.len(),
                    site_results.len(),
                    stage_outcomes,
                ),
            }));
        }

        // Multi-signal thermodynamic classification (z-score normalized)
        let asymmetries: Vec<f32> = site_results.iter().map(|r| r.asymmetry_score).collect();
        let n = asymmetries.len() as f32;
        let mean_asym = if n > 0.0 {
            asymmetries.iter().sum::<f32>() / n
        } else {
            0.0
        };
        let var_asym = if n > 1.0 {
            asymmetries
                .iter()
                .map(|a| (a - mean_asym).powi(2))
                .sum::<f32>()
                / (n - 1.0)
        } else {
            0.01
        };
        let std_asym = var_asym.sqrt().max(0.001);

        // NaN-safe median: TE may be NaN (kernel marks "not honestly computable"
        // when source/target trains saturate). Drop NaN before sorting so the
        // median lands on a real value.
        let mut all_te: Vec<f32> = site_results
            .iter()
            .flat_map(|r| r.tide_decomposition.iter().map(|t| t.transfer_entropy))
            .filter(|te| te.is_finite())
            .collect();
        all_te.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_te = if all_te.is_empty() {
            0.0
        } else {
            all_te[all_te.len() / 2]
        };

        for sr in site_results.iter_mut() {
            let z = (sr.asymmetry_score - mean_asym) / std_asym;
            sr.relative_asymmetry = z;
            // NaN-safe top_te: 0.0 fallback collapses tide_enriched to false honestly.
            let top_te = sr
                .tide_decomposition
                .first()
                .map(|t| t.transfer_entropy)
                .filter(|te| te.is_finite())
                .unwrap_or(0.0);
            let tide_enriched = median_te > 0.0 && top_te > 2.0 * median_te;
            let is_soc = sr.tau > 0.0 && sr.tau < 1.5;
            sr.therm_class =
                if z > 1.5 && sr.tau > 0.0 && (is_soc || tide_enriched || sr.asymmetry_score > 0.4)
                {
                    ThermClass::Cryptic
                } else if z > 0.5 {
                    ThermClass::Dynamic
                } else if z > -0.5 {
                    ThermClass::Responsive
                } else {
                    ThermClass::Inert
                };
        }

        log::info!(
            "  PRISM-Therm classification: mean_asym={:.4}, std={:.4}, median_TE={:.5}",
            mean_asym,
            std_asym,
            median_te
        );
        for sr in &site_results {
            // Log "n/a" when TE is honestly undefined; never fabricate a 0.0.
            let top_te_str = sr
                .tide_decomposition
                .first()
                .and_then(|t| {
                    if t.transfer_entropy.is_finite() {
                        Some(format!("{:.4}", t.transfer_entropy))
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| "n/a".to_string());
            log::info!(
                "    Site {}: z={:+.2} -> {}  (asym={:.4}, tau={:.2}, topTE={})",
                sr.site_id,
                sr.relative_asymmetry,
                sr.therm_class,
                sr.asymmetry_score,
                sr.tau,
                top_te_str
            );
        }

        let hysteretic_count = site_results.iter().filter(|r| r.is_hysteretic).count();

        // Log TIDE summary; honest n/a for undefined TE / dG.
        for sr in &site_results {
            if !sr.tide_decomposition.is_empty() {
                let te = sr.tide_decomposition[0].transfer_entropy;
                let dg = sr.tide_decomposition[0].causal_dg;
                let te_s = if te.is_finite() {
                    format!("{:.4}", te)
                } else {
                    "n/a".into()
                };
                let dg_s = if dg.is_finite() {
                    format!("{:.3}", dg)
                } else {
                    "n/a".into()
                };
                log::info!(
                    "  TIDE Site {}: {} active residues, top TE={}, top dG={}",
                    sr.site_id,
                    sr.tide_decomposition.len(),
                    te_s,
                    dg_s
                );
            }
        }

        // Global SDST pocket scan — GPU-native CCNS (sort-reduce + CSN estimator).
        // Replaces legacy ccns_all_pockets which downloaded 13M+ events to host.
        let (global_pockets, global_scan_outcome) = capture_stage(
            SdstAnalysisStage::GlobalScan,
            None,
            "Sdst::ccns_all_pockets_gpu",
            self.run_global_scan(),
        );
        stage_outcomes.push(global_scan_outcome);
        let Some(global_pockets) = global_pockets else {
            return Err(anyhow::Error::new(SdstAnalysisError {
                outcome: SdstAnalysisOutcome::from_stages(
                    requests.len(),
                    site_results.len(),
                    stage_outcomes,
                ),
            }));
        };

        // Avalanche count: proxy for total observed conformational events
        let (avalanches, avalanche_outcome) = capture_stage(
            SdstAnalysisStage::AvalancheStats,
            None,
            "Sdst::avalanche_stats",
            self.sdst.avalanche_stats(-1),
        );
        stage_outcomes.push(avalanche_outcome);
        let Some(avalanches) = avalanches else {
            return Err(anyhow::Error::new(SdstAnalysisError {
                outcome: SdstAnalysisOutcome::from_stages(
                    requests.len(),
                    site_results.len(),
                    stage_outcomes,
                ),
            }));
        };
        let total_avalanches = avalanches.len();

        let outcome =
            SdstAnalysisOutcome::from_stages(requests.len(), site_results.len(), stage_outcomes);

        Ok(PrismThermAnalysis {
            outcome,
            sdst_event_count: event_count,
            hysteresis_threshold: HYSTERESIS_THRESHOLD,
            hysteretic_site_count: hysteretic_count,
            sites: site_results,
            global_pockets,
            total_avalanches,
            tide_residues_mapped: mapped,
        })
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    // (helpers continue after this block)
}

/// Merge multiple `PrismThermAnalysis` results — one per TWIN-differential
/// group — into a single analysis.
///
/// ## Why this exists
///
/// In multi-differential mode the engine runs 4 distinct protocols
/// concurrently (ThermalShock, Equilibrium, UvAromatic, Hysteresis).
/// Phase boundaries differ across protocols, so a SDST bridge with a
/// single protocol mis-classifies every spike that didn't come from the
/// group whose protocol it holds. The observed failure is
/// `asymmetry_score = 1.0` on every site and `therm_class = RESPONSIVE`
/// for every site — because the z-score across saturated values is zero.
///
/// ## Strategy
///
/// Per-group SdstBridge instances ingest only their own group's spikes,
/// each with the correct protocol. Each produces its own
/// `PrismThermAnalysis`. This function merges them:
///
///   1. For each site, select the per-group `PrismThermSiteResult` with
///      the MAX `asymmetry_score`. This reflects the physical intuition
///      that a real cryptic pocket responds most strongly to the
///      protocol best suited to excite it (ThermalShock for hot
///      cryptics, Hysteresis for re-closure-resistant, etc.).
///   2. Once the merged per-site asymmetry is selected, recompute the
///      z-score across sites and reclassify `therm_class` using the
///      same logic as `SdstBridge::analyze` (z > 1.5 → Cryptic, etc.).
///      This way the classifier actually differentiates sites instead
///      of collapsing to one bucket.
///   3. Global pockets: concatenated union (each group may find
///      different candidate pockets). Duplicates at the same grid
///      region are collapsed.
///   4. Other fields summed (sdst_event_count, total_avalanches).
///
/// ## Input invariants
///
///   * Every input analysis must have been produced against the SAME
///     `clustered_sites` slice. Site IDs are matched by `site_id`.
///   * `analyses` is non-empty.
pub fn merge_per_group_analyses(
    analyses: &[PrismThermAnalysis],
    sites: &[ClusteredBindingSite],
) -> Result<PrismThermAnalysis> {
    let requests: Vec<SdstRegionRequest> = sites
        .iter()
        .map(|site| {
            legacy_cluster_request(
                site.cluster_id,
                SpatialRegion {
                    x_min: 0,
                    x_max: 0,
                    y_min: 0,
                    y_max: 0,
                    z_min: 0,
                    z_max: 0,
                },
            )
        })
        .collect::<Result<Vec<_>>>()?;
    merge_per_group_region_analyses(analyses, &requests)
}

/// Merge per-group analyses using exact domain-preserving identities.
/// Every request must occur exactly once in every successful input analysis.
pub fn merge_per_group_region_analyses(
    analyses: &[PrismThermAnalysis],
    requests: &[SdstRegionRequest],
) -> Result<PrismThermAnalysis> {
    if analyses.is_empty() {
        anyhow::bail!("merge_per_group_region_analyses: analyses must be non-empty");
    }
    let requested_identities: std::collections::HashSet<AnalyzedSiteIdentity> = requests
        .iter()
        .map(|request| request.analyzed_site_identity)
        .collect();
    if requested_identities.len() != requests.len() {
        anyhow::bail!("merge_per_group_region_analyses: duplicate requested identity");
    }
    for (analysis_index, analysis) in analyses.iter().enumerate() {
        if analysis.sites.len() != requests.len() {
            anyhow::bail!(
                "merge input {analysis_index} has {} sites for {} requests",
                analysis.sites.len(),
                requests.len()
            );
        }
        for request in requests {
            let matches: Vec<&PrismThermSiteResult> = analysis
                .sites
                .iter()
                .filter(|site| site.analyzed_site_identity == request.analyzed_site_identity)
                .collect();
            if matches.len() != 1 {
                anyhow::bail!(
                    "merge input {analysis_index} contains {} rows for identity {:?}",
                    matches.len(),
                    request.analyzed_site_identity
                );
            }
            if matches[0].materialized_site_id != request.materialized_site_id {
                anyhow::bail!(
                    "merge input {analysis_index} materialized mapping mismatch for identity {:?}",
                    request.analyzed_site_identity
                );
            }
        }
    }

    // Step 1: for each site, pick the per-group result with MAX asymmetry_score.
    let mut merged_sites: Vec<PrismThermSiteResult> = Vec::with_capacity(requests.len());
    for request in requests {
        let mut best: Option<PrismThermSiteResult> = None;
        for a in analyses {
            if let Some(sr) = a
                .sites
                .iter()
                .find(|s| s.analyzed_site_identity == request.analyzed_site_identity)
            {
                match &best {
                    None => best = Some(sr.clone()),
                    Some(cur) if sr.asymmetry_score > cur.asymmetry_score => {
                        best = Some(sr.clone());
                    }
                    _ => {}
                }
            }
        }
        if let Some(mut sr) = best {
            // Clear the existing relative_asymmetry + therm_class — they were
            // derived from the per-group z-score which is not valid after
            // merging. Both are recomputed in step 2 below.
            sr.relative_asymmetry = 0.0;
            sr.therm_class = ThermClass::Responsive;
            merged_sites.push(sr);
        }
    }

    // Step 2: recompute z-score + reclassify on the merged asymmetries.
    // Mirrors the classifier in SdstBridge::analyze:425-455 exactly.
    let asymmetries: Vec<f32> = merged_sites.iter().map(|r| r.asymmetry_score).collect();
    let n = asymmetries.len() as f32;
    let mean_asym = if n > 0.0 {
        asymmetries.iter().sum::<f32>() / n
    } else {
        0.0
    };
    let var_asym = if n > 1.0 {
        asymmetries
            .iter()
            .map(|a| (a - mean_asym).powi(2))
            .sum::<f32>()
            / (n - 1.0)
    } else {
        0.01
    };
    let std_asym = var_asym.sqrt().max(0.001);

    // NaN-safe median: drop NaN before sorting (TE may be NaN per producer-repair).
    let mut all_te: Vec<f32> = merged_sites
        .iter()
        .flat_map(|r| r.tide_decomposition.iter().map(|t| t.transfer_entropy))
        .filter(|te| te.is_finite())
        .collect();
    all_te.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_te = if all_te.is_empty() {
        0.0
    } else {
        all_te[all_te.len() / 2]
    };

    let mut hysteretic_count = 0usize;
    for sr in merged_sites.iter_mut() {
        let z = (sr.asymmetry_score - mean_asym) / std_asym;
        sr.relative_asymmetry = z;
        let top_te = sr
            .tide_decomposition
            .first()
            .map(|t| t.transfer_entropy)
            .filter(|te| te.is_finite())
            .unwrap_or(0.0);
        let tide_enriched = median_te > 0.0 && top_te > 2.0 * median_te;
        let is_soc = sr.tau > 0.0 && sr.tau < 1.5;
        sr.therm_class =
            if z > 1.5 && sr.tau > 0.0 && (is_soc || tide_enriched || sr.asymmetry_score > 0.4) {
                ThermClass::Cryptic
            } else if z > 0.5 {
                ThermClass::Dynamic
            } else if z > -0.5 {
                ThermClass::Responsive
            } else {
                ThermClass::Inert
            };
        if sr.is_hysteretic {
            hysteretic_count += 1;
        }
    }

    log::info!(
        "  PRISM-Therm merged ({} groups): mean_asym={:.4}, std={:.4}, median_TE={:.5}",
        analyses.len(),
        mean_asym,
        std_asym,
        median_te
    );

    // Step 3: union global_pockets across groups. Dedupe by grid_region.
    let mut merged_global: Vec<PrismThermGlobalPocket> = Vec::new();
    let mut seen_regions: std::collections::HashSet<[u32; 6]> = std::collections::HashSet::new();
    for a in analyses {
        for p in &a.global_pockets {
            if seen_regions.insert(p.grid_region) {
                merged_global.push(p.clone());
            }
        }
    }

    // Step 4: sum event counts + avalanches + pick max tide_residues_mapped.
    let sdst_event_count: u32 = analyses.iter().map(|a| a.sdst_event_count).sum();
    let total_avalanches: usize = analyses.iter().map(|a| a.total_avalanches).sum();
    let tide_residues_mapped: usize = analyses
        .iter()
        .map(|a| a.tide_residues_mapped)
        .max()
        .unwrap_or(0);

    Ok(PrismThermAnalysis {
        outcome: {
            let stages = analyses
                .iter()
                .enumerate()
                .flat_map(|(index, analysis)| {
                    analysis
                        .outcome
                        .stages
                        .iter()
                        .cloned()
                        .map(move |stage| stage.with_source_analysis_index(index))
                })
                .chain(std::iter::once(SdstStageOutcome::observed(
                    SdstAnalysisStage::Merge,
                    None,
                    "merge_per_group_analyses",
                )))
                .collect();
            SdstAnalysisOutcome::from_stages(requests.len(), merged_sites.len(), stages)
        },
        sdst_event_count,
        hysteresis_threshold: HYSTERESIS_THRESHOLD,
        hysteretic_site_count: hysteretic_count,
        sites: merged_sites,
        global_pockets: merged_global,
        total_avalanches,
        tide_residues_mapped,
    })
}

#[cfg(test)]
mod truthful_failure_tests {
    use super::*;

    #[test]
    fn observed_zero_is_not_failure() {
        let (value, outcome) = capture_stage::<u32, &str>(
            SdstAnalysisStage::AvalancheStats,
            None,
            "test::observed_zero",
            Ok(0),
        );

        assert_eq!(value, Some(0));
        assert!(matches!(outcome, SdstStageOutcome::Observed { .. }));
        let json = serde_json::to_value(&outcome).expect("outcome must serialize");
        assert_eq!(json["outcome"], "observed");
        assert!(json.get("failure").is_none());
    }

    #[test]
    fn observed_empty_collection_is_not_failure() {
        let (value, outcome) = capture_stage::<Vec<u32>, &str>(
            SdstAnalysisStage::Tide,
            Some(7),
            "test::observed_empty",
            Ok(Vec::new()),
        );

        assert_eq!(value, Some(Vec::new()));
        assert!(matches!(outcome, SdstStageOutcome::Observed { .. }));
    }

    #[test]
    fn failed_stage_has_no_scientific_value_and_preserves_debug_source() {
        let (value, outcome) = capture_stage::<u32, &str>(
            SdstAnalysisStage::Ccns,
            None,
            "test::failed_ccns",
            Err("cuda launch unavailable"),
        );

        assert_eq!(value, None);
        let json = serde_json::to_value(&outcome).expect("failure must serialize");
        assert_eq!(json["outcome"], "failed");
        assert_eq!(json["failure"]["kind"], "unclassified");
        assert!(json["failure"]["message"]
            .as_str()
            .expect("message string")
            .contains("cuda launch unavailable"));
        assert!(json.get("value").is_none());
    }

    #[test]
    fn typed_error_remains_downcastable_through_anyhow() {
        let stage = SdstStageOutcome::failed(
            SdstAnalysisStage::Hysteresis,
            Some(3),
            "test::hysteresis",
            SdstFailureKind::Unclassified,
            "device error".to_string(),
        );
        let typed = SdstAnalysisError {
            outcome: SdstAnalysisOutcome::from_stages(1, 0, vec![stage]),
        };
        let error = anyhow::Error::new(typed);

        let recovered = error
            .downcast_ref::<SdstAnalysisError>()
            .expect("typed failure must survive anyhow propagation");
        assert_eq!(recovered.outcome.status, SdstAnalysisStatus::Failed);
        assert_eq!(recovered.outcome.materialized_site_count, 0);
    }

    #[test]
    fn legacy_cluster_adapter_does_not_synthesize_materialized_identity() {
        let request = legacy_cluster_request(
            19,
            SpatialRegion {
                x_min: 1,
                x_max: 2,
                y_min: 3,
                y_max: 4,
                z_min: 5,
                z_max: 6,
            },
        )
        .expect("non-negative cluster id is representable");

        assert_eq!(
            request.analyzed_site_identity,
            AnalyzedSiteIdentity::Cluster(ClusterId(19))
        );
        assert_eq!(request.materialized_site_id, None);
    }

    #[test]
    fn legacy_merge_negative_cluster_id_returns_err_without_panicking() {
        let site = ClusteredBindingSite::new_with_geometric_voxel_mass(
            -1,
            [0.0, 0.0, 0.0],
            0,
            Vec::new(),
            0.0,
            0.0,
            [0.0, 0.0, 0.0],
            0.0,
            crate::persistent_engine::DruggabilityScore::default(),
            crate::persistent_engine::SiteClassification::Unknown,
        );

        let invocation = std::panic::catch_unwind(|| merge_per_group_analyses(&[], &[site]));
        let result = invocation.expect("negative legacy identity must not panic");
        let error = result.expect_err("negative legacy identity must return Err");
        assert!(error.to_string().contains("negative legacy cluster id -1"));
    }
}

// (SdstBridge impl continues below with the remaining helper methods)
impl SdstBridge {
    /// Convert one `GpuSpikeEvent` to `SpikeInput` for SDST ingestion.
    fn convert_spike(&self, s: &GpuSpikeEvent) -> sdst::SpikeInput {
        let ts = s.timestep.max(0) as u32;

        // Decode grid coordinates from packed voxel_idx
        // NHS x-fast linear layout under the admitted per-instance cubic grid.
        let idx = s.voxel_idx.max(0) as usize;let grid_dim=self.native_config.grid_nx as usize;
        let vx = (idx % grid_dim) as u32;
        let vy = ((idx / grid_dim) % grid_dim) as u32;
        let vz = (idx / (grid_dim * grid_dim)) as u32;
        let dim = self.native_config.grid_nx - 1;

        // Phase 9: Real thermodynamic fields (not proxies).
        //
        // energy_gradient = |∂WD/∂t| — the water density change that drove the LIF
        //   spike. This is the actual signal the neuron thresholds against, so it
        //   directly measures thermodynamic barrier crossing.
        //
        // local_temp = T_protocol × (1 + 0.5 × (1 − WD)) — spatially modulated
        //   temperature. Buried voxels (WD ≈ 0) see 1.5× protocol temp; fully
        //   solvated voxels (WD ≈ WD_bulk) see ~1× protocol temp. This gives SDST
        //   spatial temperature resolution without per-voxel KE computation.
        //
        // solvent_exposure = water_density — direct per-voxel NHS measurement.
        let wd_clamped = s.water_density.clamp(0.0, 1.0);
        let t_protocol = self.temp_at(ts);
        let local_temp = t_protocol * (1.0 + 0.5 * (1.0 - wd_clamped));

        sdst::SpikeInput {
            voxel_x: vx.min(dim),
            voxel_y: vy.min(dim),
            voxel_z: vz.min(dim),
            timestamp: ts,
            amplitude: s.intensity,
            local_temp: local_temp,
            energy_gradient: s.wd_change,      // |∂WD/∂t| from kernel
            solvent_exposure: s.water_density, // NHS water density field
            phase_id: self.phase_at(ts),
        }
    }

    /// Compute which of the 5 hysteresis phases (0-4) contains `step`.
    fn phase_at(&self, step: u32) -> u8 {
        let p = &self.protocol;
        let p1 = p.cold_hold_steps.max(0) as u32;
        let p2 = p1 + p.ramp_steps.max(0) as u32;
        let p3 = p2 + p.warm_hold_steps.max(0) as u32;
        let p4 = p3 + p.ramp_down_steps.max(0) as u32;
        if step < p1 {
            0
        } else if step < p2 {
            1
        } else if step < p3 {
            2
        } else if step < p4 {
            3
        } else {
            4
        }
    }

    /// Interpolate the protocol temperature (K) at a given timestep.
    fn temp_at(&self, step: u32) -> f32 {
        let p = &self.protocol;
        let s = step as i32;
        let p1 = p.cold_hold_steps;
        let p2 = p1 + p.ramp_steps;
        let p3 = p2 + p.warm_hold_steps;
        let p4 = p3 + p.ramp_down_steps;

        if s < p1 {
            p.start_temp
        } else if s < p2 {
            let frac = (s - p1) as f32 / p.ramp_steps.max(1) as f32;
            p.start_temp + frac * (p.end_temp - p.start_temp)
        } else if s < p3 {
            p.end_temp
        } else if s < p4 {
            // Phase 4: ramp down (cooling)
            let frac = (s - p3) as f32 / p.ramp_down_steps.max(1) as f32;
            p.end_temp - frac * (p.end_temp - p.start_temp)
        } else {
            // Phase 5: cold return hold
            p.start_temp
        }
    }

    /// Convert a site centroid (Å) to an SDST SpatialRegion in grid voxels.
    /// Radius = 8 voxels ≈ 6 Å — matches the default lining residue cutoff.
    fn centroid_to_region(&self, centroid: &[f32; 3]) -> SpatialRegion {
        let radius = 8u32;
        let dim = self.native_config.grid_nx;

        let to_voxel = |pos: f32, origin: f32| -> u32 {
            let v = ((pos - origin) / self.native_config.grid_spacing).max(0.0) as u32;
            v.min(dim - 1)
        };

        let cx = to_voxel(centroid[0], self.grid_origin[0]);
        let cy = to_voxel(centroid[1], self.grid_origin[1]);
        let cz = to_voxel(centroid[2], self.grid_origin[2]);

        SpatialRegion {
            x_min: cx.saturating_sub(radius),
            x_max: (cx + radius).min(dim - 1),
            y_min: cy.saturating_sub(radius),
            y_max: (cy + radius).min(dim - 1),
            z_min: cz.saturating_sub(radius),
            z_max: (cz + radius).min(dim - 1),
        }
    }

    /// Convert an Angstrom-space lattice-component AABB into an explicit,
    /// domain-preserving SDST region request.
    /// Region requests for materialized lattice sites.
    ///
    /// The single supported way to submit a lattice surface for analysis: every
    /// request carries the site's assigned `MaterializedSiteId`, so the result
    /// can be joined by the consumers instead of falling back to a sentinel.
    pub fn lattice_requests(&self, sites: &[MaterializedSite]) -> Result<Vec<SdstRegionRequest>> {
        sites
            .iter()
            .map(|site| {
                let AnalyzedSiteIdentity::LatticeComponent(component_id) =
                    site.analyzed_site_identity
                else {
                    anyhow::bail!(
                        "{:?}: {:?} is not a lattice component",
                        SiteUnmatchedReason::ConsumerDomainMismatch,
                        site.analyzed_site_identity
                    );
                };
                self.lattice_region(
                    component_id,
                    Some(site.materialized_site_id),
                    site.aabb_min,
                    site.aabb_max,
                )
            })
            .collect()
    }

    pub fn lattice_region(
        &self,
        component_id: LatticeComponentId,
        materialized_site_id: Option<MaterializedSiteId>,
        aabb_min: [f32; 3],
        aabb_max: [f32; 3],
    ) -> Result<SdstRegionRequest> {
        for axis in 0..3 {
            if !aabb_min[axis].is_finite() || !aabb_max[axis].is_finite() {
                anyhow::bail!("lattice AABB axis {axis} is non-finite");
            }
            if aabb_min[axis] > aabb_max[axis] {
                anyhow::bail!(
                    "lattice AABB axis {axis} is inverted: {} > {}",
                    aabb_min[axis],
                    aabb_max[axis]
                );
            }
        }

        let dim = self.native_config.grid_nx as i64 - 1;
        let lower = |axis: usize| -> u32 {
            (((aabb_min[axis] - self.grid_origin[axis]) / self.native_config.grid_spacing).floor() as i64)
                .clamp(0, dim) as u32
        };
        let upper = |axis: usize| -> u32 {
            (((aabb_max[axis] - self.grid_origin[axis]) / self.native_config.grid_spacing).ceil() as i64).clamp(0, dim)
                as u32
        };

        Ok(SdstRegionRequest {
            analyzed_site_identity: AnalyzedSiteIdentity::LatticeComponent(component_id),
            materialized_site_id,
            region: SpatialRegion {
                x_min: lower(0),
                x_max: upper(0),
                y_min: lower(1),
                y_max: upper(1),
                z_min: lower(2),
                z_max: upper(2),
            },
        })
    }

    /// Analyze one NHS binding site: hysteresis + TIDE.
    /// CCNS is pre-computed via batched `ccns_for_regions()` (GPU CSN estimator).
    fn analyze_site(
        &self,
        request: &SdstRegionRequest,
        ccns: &CcnsResult,
    ) -> (Option<PrismThermSiteResult>, Vec<SdstStageOutcome>) {
        let legacy_site_id = match request.analyzed_site_identity {
            AnalyzedSiteIdentity::Cluster(ClusterId(id)) => i32::try_from(id).ok(),
            AnalyzedSiteIdentity::LatticeComponent(_)
            | AnalyzedSiteIdentity::MaterializerRegion(_) => request
                .materialized_site_id
                .and_then(|MaterializedSiteId(id)| i32::try_from(id).ok()),
        };
        let mut outcomes = Vec::with_capacity(2);
        let (hyst, hysteresis_outcome) = capture_stage(
            SdstAnalysisStage::Hysteresis,
            legacy_site_id,
            "Sdst::hysteresis_region",
            self.sdst
                .hysteresis_region(&request.region, HYSTERESIS_THRESHOLD),
        );
        outcomes.push(hysteresis_outcome);

        // TIDE decomposition: per-residue causal ΔG
        let (tide, tide_outcome) = capture_stage(
            SdstAnalysisStage::Tide,
            legacy_site_id,
            "Sdst::tide_decomposition",
            self.run_tide(&request.region),
        );
        outcomes.push(tide_outcome);

        let (Some(hyst), Some(tide)) = (hyst, tide) else {
            return (None, outcomes);
        };

        let result = PrismThermSiteResult {
            analyzed_site_identity: request.analyzed_site_identity,
            materialized_site_id: request.materialized_site_id,
            site_id: legacy_site_id.unwrap_or(-1),
            asymmetry_score: hyst.asymmetry_score,
            is_hysteretic: hyst.is_hysteretic,
            heating_spike_rate: hyst.heating_spike_rate,
            cooling_spike_rate: hyst.cooling_spike_rate,
            heating_spike_count: hyst.heating_spike_count,
            cooling_spike_count: hyst.cooling_spike_count,
            tau: ccns.tau,
            tau_stderr: ccns.tau_stderr,
            ccns_classification: ccns_class_name(ccns.classification).to_string(),
            druggability: ccns.druggability,
            n_avalanches: ccns.n_avalanches,
            tide_decomposition: tide,
            relative_asymmetry: 0.0,
            therm_class: ThermClass::Responsive,
        };
        (Some(result), outcomes)
    }

    /// Run TIDE decomposition for a pocket region.
    ///
    /// Returns up to `TIDE_TOP_RESIDUES` (20) residues sorted by transfer
    /// entropy descending. These are the residues with strongest causal
    /// influence on the pocket — the drug design leverage points.
    fn run_tide(&self, region: &SpatialRegion) -> Result<Vec<TideResidueResult>> {
        let decomp = self
            .sdst
            .tide_decomposition_with_support_v1(region, &self.residue_map, self.n_residues)
            .map_err(|error| anyhow::anyhow!("{error:?}"))?;

        if decomp.is_empty() {
            return Ok(Vec::new());
        }

        // Convert to output format, sort by TE descending, take top N.
        // Honest filter: keep entries with real causal-spike support OR
        // honestly-computed positive TE. Entries with TE=NaN AND zero
        // causal spikes carry no signal and are dropped. Entries with
        // TE=NaN but n_causal_spikes>0 are preserved — the residue has
        // pocket activity even if the binned TE collapsed under
        // saturation (downstream consumer maps NaN to null).
        let mut results: Vec<TideResidueResult> = decomp
            .into_iter()
            .filter(|(d, _)| {
                self.tide_output_limit.is_none()
                    || (d.transfer_entropy.is_finite() && d.transfer_entropy > 0.0)
                    || d.n_causal_spikes > 0
            })
            .map(|(d, support)| TideResidueResult {
                residue_id: d.residue_id,
                causal_dg: d.causal_dg,
                transfer_entropy: d.transfer_entropy,
                fisher_info: d.fisher_info,
                kl_divergence: d.kl_divergence,
                n_causal_spikes: d.n_causal_spikes,
                measurement_support: serde_json::json!({
                    "schema": "sdst_tide_measurement_support_v1",
                    "residue_id": support.residue_id,
                    "bin_width_md_steps": support.bin_width_steps,
                    "max_bins": support.max_bins,
                    "n_bins": support.n_bins,
                    "source_active_bins": support.source_active_bins,
                    "target_active_bins": support.target_active_bins,
                    "source_occupancy": (support.n_bins > 0).then(|| support.source_active_bins as f64 / support.n_bins as f64),
                    "target_occupancy": (support.n_bins > 0).then(|| support.target_active_bins as f64 / support.n_bins as f64),
                    "source_binned_spikes": support.source_binned_spikes,
                    "target_binned_spikes": support.target_binned_spikes,
                    "source_outside_bin_range": support.source_outside_bin_range,
                    "target_outside_bin_range": support.target_outside_bin_range,
                    "unavailable_flags": support.unavailable_flags,
                    "availability_reasons": support.availability_reasons(),
                    "availability": if support.unavailable_flags == 0 { "computed" } else { "unavailable" },
                    "transfer_entropy_availability": if support.unavailable_flags == 0 { "computed" } else { "unavailable" },
                    "occupancy_lower_bound_exclusive": 0.05,
                    "occupancy_upper_bound_exclusive": 0.95,
                    "variance_window_bins": support.variance_window_bins,
                    "variance_total_windows": support.variance_total_windows,
                    "variance_valid_windows": support.variance_valid_windows,
                    "variance_availability": if support.variance_valid_windows >= 2 { "computed" } else { "fewer_than_two_valid_windows" },
                    "joint_counts_target_next_target_current_source_current": support.joint_counts,
                    "source_train": "residue_attributed_events_in_region_plus_two_voxel_margin",
                    "target_train": "all_events_inside_region_including_source_residue",
                    "lag_bins": 1,
                    "source_history_bins": 1,
                    "target_history_bins": 1,
                    "estimator": "binary_plugin_conditional_mutual_information_log2_bits",
                    "native_n_causal_spikes_semantics": "source_residue_event_count_inside_strict_region_not_parent_edge_count",
                }),
            })
            .collect();

        // NaN-aware descending sort: NaN entries sort to the end (least
        // informative), finite entries sort by TE descending.
        results.sort_unstable_by(|a, b| {
            match (
                a.transfer_entropy.is_finite(),
                b.transfer_entropy.is_finite(),
            ) {
                (true, true) => b
                    .transfer_entropy
                    .partial_cmp(&a.transfer_entropy)
                    .unwrap_or(std::cmp::Ordering::Equal),
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                (false, false) => std::cmp::Ordering::Equal,
            }
        });
        if let Some(limit) = self.tide_output_limit {
            results.truncate(limit);
        }
        Ok(results)
    }

    /// Run the global SDST pocket scan via GPU-native `ccns_all_pockets_gpu()`.
    ///
    /// Sort-reduce architecture: CUB radix sort → tile segmentation → fused
    /// CSN estimator per tile. All 13M+ events stay on GPU. No host downloads.
    fn run_global_scan(&self) -> Result<Vec<PrismThermGlobalPocket>> {
        let pockets = self
            .sdst
            .ccns_all_pockets_gpu()
            .map_err(|error| anyhow::anyhow!("{error:?}"))?;

        Ok(pockets
            .into_iter()
            .take(50) // Limit to top 50 for JSON size
            .map(|(ccns, region)| {
                // Approximate centroid in Å
                let cx = ((region.x_min + region.x_max) as f32 / 2.0) * self.native_config.grid_spacing
                    + self.grid_origin[0];
                let cy = ((region.y_min + region.y_max) as f32 / 2.0) * self.native_config.grid_spacing
                    + self.grid_origin[1];
                let cz = ((region.z_min + region.z_max) as f32 / 2.0) * self.native_config.grid_spacing
                    + self.grid_origin[2];

                PrismThermGlobalPocket {
                    grid_region: [
                        region.x_min,
                        region.x_max,
                        region.y_min,
                        region.y_max,
                        region.z_min,
                        region.z_max,
                    ],
                    tau: ccns.tau,
                    ccns_classification: ccns_class_name(ccns.classification).to_string(),
                    druggability: ccns.druggability,
                    centroid_angstrom: [cx, cy, cz],
                }
            })
            .collect())
    }
}

#[cfg(test)]
mod native_population_tests {
    use super::*;

    // Freeze the pre-extraction live percentile branch independently so both
    // admission and record ordering are compared, including percentile ties.
    fn prior_live_percentile(raw: Vec<GpuSpikeEvent>, percentile: u32) -> Vec<GpuSpikeEvent> {
        if raw.len() <= 1000 {
            return raw;
        }
        let mut uv = Vec::new();
        let mut lif = Vec::new();
        let mut efp = Vec::new();
        let mut ladd = Vec::new();
        for s in raw {
            match s.spike_source {
                1 => uv.push(s),
                3 => efp.push(s),
                4 | 5 => ladd.push(s),
                _ => lif.push(s),
            }
        }
        let filter = |spikes: Vec<GpuSpikeEvent>| {
            if spikes.len() < 100 {
                return spikes;
            }
            let mut sorted: Vec<f32> = spikes.iter().map(|s| s.intensity).collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let idx = (sorted.len() as f32 * ((percentile.min(99) as f32) / 100.0)) as usize;
            let threshold = sorted.get(idx).copied().unwrap_or(0.0);
            spikes
                .into_iter()
                .filter(|s| s.intensity >= threshold)
                .collect()
        };
        let mut result = filter(uv);
        result.extend(filter(lif));
        result.extend(filter(efp));
        result.extend(ladd);
        result
    }

    #[test]
    fn voronoi_percentile_uses_a_distinct_population() {
        let input: Vec<_> = (0..1200).map(|i| GpuSpikeEvent {
            timestep: i,
            spike_source: if i < 600 { 2 } else { 6 },
            intensity: if i < 600 { i as f32 } else { (i - 600 + 10000) as f32 },
            ..GpuSpikeEvent::default()
        }).collect();
        let (selected, receipts) = select_native_percentile_indices(&input, 50);
        let expected: Vec<_> = (300..600).chain(900..1200).collect();
        assert_eq!(selected, expected);
        let lif = receipts.iter().find(|r| r.channel == "LIF").unwrap();
        let voronoi = receipts.iter().find(|r| r.channel == "VORONOI_LIF").unwrap();
        assert_eq!((lif.input_count, lif.selected_count, lif.threshold), (600, 300, Some(300.0)));
        assert_eq!((voronoi.input_count, voronoi.selected_count, voronoi.threshold), (600, 300, Some(10300.0)));
        let (_, tiny) = select_native_percentile_indices(&input[599..601], 50);
        assert_eq!(tiny.iter().find(|r| r.channel == "LIF").unwrap().input_count, 1);
        assert_eq!(tiny.iter().find(|r| r.channel == "VORONOI_LIF").unwrap().input_count, 1);
    }

    #[test]
    fn native_percentile_selection_matches_prior_live_population_and_order() {
        for size in [0, 99, 100, 1000, 1001, 4200] {
            let input: Vec<_> = (0..size)
                .map(|i| GpuSpikeEvent {
                    timestep: i,
                    spike_source: if i % 101 == 0 {
                        3
                    } else {
                        [1, 2, 4, 5, 7][i as usize % 5]
                    },
                    intensity: ((i * 17) % 41) as f32,
                    ..GpuSpikeEvent::default()
                })
                .collect();
            for percentile in [0, 70, 99, 100] {
                let expected = prior_live_percentile(input.clone(), percentile);
                let (actual, receipt) = select_native_percentile_spikes(input.clone(), percentile);
                assert_eq!(actual, expected);
                assert_eq!(
                    receipt.iter().map(|r| r.input_count).sum::<usize>(),
                    input.len()
                );
                assert_eq!(
                    receipt.iter().map(|r| r.selected_count).sum::<usize>(),
                    actual.len()
                );
                assert!(input
                    .iter()
                    .filter(|s| matches!(s.spike_source, 4 | 5))
                    .all(|s| actual.contains(s)));
            }
        }
    }
}
