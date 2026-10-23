//! Materialize GhostPhaseLattice4D output → SiteManifest blocks.
//!
//! Bridges the [`crate::ghost_phase_lattice::GhostPhaseLatticeOutcome`]
//! produced by the GPU edge-adjudication backend into the four
//! `Option`-typed extension blocks on [`crate::site_manifest::SiteManifest`]:
//!
//!   * [`GhostPhaseLatticeProvenance`] — backend-level metadata.
//!   * [`PhaseManifold`] — per-phase aggregated centroid + driver block.
//!   * [`ThermCcnsLifecycle`] — per-phase mean KL / flux / water +
//!     driver-residue persistence.
//!   * [`So3Manifold`] — spherical-coherence summary with explicit
//!     unavailability when accepted-edge lineage is not emitted.
//!
//! This module is host-side only — it never reads device buffers and never
//! mutates the lattice nodes. It implements the directive Part IV
//! "Manifold Materialization" rules verbatim:
//!
//!   * Group component nodes by `protocol_phase`; for each phase compute
//!     the AABB-volume-weighted centroid + AABB union + mean KL +
//!     unique driver residues.
//!   * NaN values in `kl_divergence` / `thermo_flux` / `water_density`
//!     are excluded from the means; the JSON emits the
//!     `unavailable_neutral` sentinel when *every* node had NaN.
//!   * Driver-residue persistence is normalised to sum to 1.0 across
//!     resolved drivers (sentinel `NODE_CAUSAL_LEAD_NONE` excluded).
//!   * SO(3) plane status is the union of `so3_plane_status` bits across
//!     constituent nodes (a plane is "populated" if any node carried
//!     non-zero values on it).

#![cfg(any(feature = "gpu", feature = "cpu-analysis"))]

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::ghost_phase_lattice::{
    GhostPhaseLatticeComponent, GhostPhaseLatticeNode, GhostPhaseLatticeOutcome,
    NODE_CAUSAL_LEAD_NONE, N_PROTOCOL_PHASES,
};
use crate::site_manifest::{
    phase_name_for, GhostPhaseLatticeProvenance, LatticeExtent, PhaseAggregate, PhaseManifold,
    So3Manifold, So3PlaneStatus, ThermCcnsLifecycle,
};

/// Bundle of the four extension blocks for a single component. Drop into
/// the matching `Option` fields on [`crate::site_manifest::SiteManifest`].
#[derive(Debug, Clone)]
pub struct ComponentManifoldBlocks {
    pub ghost_phase_lattice: GhostPhaseLatticeProvenance,
    pub phase_manifold: PhaseManifold,
    pub therm_ccns_lifecycle: ThermCcnsLifecycle,
    pub so3_manifold: So3Manifold,
    /// Component-wide AABB union — the legacy `Aabb` interface still
    /// expects a single bounding volume per cluster. The lattice's
    /// per-phase AABBs in `phase_manifold` carry the temporal trajectory.
    pub component_aabb_min: [f32; 3],
    pub component_aabb_max: [f32; 3],
    /// Exact run-owned Ghost-v2 records that formed this component.  This is
    /// the lossless bridge used by Path-B/canonical-site materialization; it
    /// prevents a component from being detached from its stream/frame/tile
    /// lineage when the lattice sidecar is consumed outside the engine.
    pub node_lineage: Vec<GhostNodeLineage>,
    /// Sorted union of resolved causal-lead residues across the component.
    /// These are driver residues, not lining residues.  Lining residues are
    /// supplied later by the dynamic Path-B spike materializer and must never
    /// be fabricated from this field.
    pub driver_residues: Vec<u32>,
    /// Spike-attribution count surrogate for [`crate::site_manifest::SiteManifest`]
    /// — the number of ghost nodes (each one was a captured-graph
    /// adjudication event of code >= 1).
    pub n_nodes: u32,
}

/// Lossless identity/provenance projection for one Ghost-v2 lattice node.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GhostNodeLineage {
    pub tile_index: u32,
    pub stream_id: u16,
    pub site_id: u32,
    pub frame_idx: u64,
    pub step_idx: u64,
    pub protocol_phase: u8,
    pub step_bucket: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub causal_lead_residue: Option<u32>,
}

/// Materialize the four manifold blocks for every component in `outcome`.
/// Returns one [`ComponentManifoldBlocks`] per component, in the same
/// order as `outcome.components` (largest first per the backend's sort).
pub fn materialize_components(
    nodes: &[GhostPhaseLatticeNode],
    outcome: &GhostPhaseLatticeOutcome,
) -> Vec<ComponentManifoldBlocks> {
    let provenance_template = build_provenance_template(nodes, outcome);
    outcome
        .components
        .iter()
        .map(|c| materialize_one(nodes, c, &provenance_template))
        .collect()
}

fn build_provenance_template(
    nodes: &[GhostPhaseLatticeNode],
    outcome: &GhostPhaseLatticeOutcome,
) -> GhostPhaseLatticeProvenance {
    let mut phases_present_set: HashSet<u8> = HashSet::new();
    for n in nodes {
        phases_present_set.insert(n.protocol_phase);
    }
    let mut phases_present: Vec<u8> = phases_present_set.into_iter().collect();
    phases_present.sort_unstable();
    let phases_present_names: Vec<String> = phases_present
        .into_iter()
        .map(|p| phase_name_for(p).to_string())
        .collect();

    GhostPhaseLatticeProvenance {
        backend: "ghost_phase_lattice_4d".to_string(),
        spatial_cell_size_a: outcome.config.spatial_cell_size_a,
        max_temporal_edge_steps: outcome.config.max_temporal_edge_steps,
        step_bucket_size: outcome.config.step_bucket_size,
        so3_threshold: outcome.config.so3_threshold,
        phase_transition_policy: "monotone_protocol_lifecycle".to_string(),
        n_tiles: outcome.stats.n_nodes,
        n_lattice_cells: outcome.stats.n_lattice_cells,
        n_directed_edges: outcome.stats.n_directed_edges,
        lattice_extent: LatticeExtent {
            step_start: outcome.stats.min_step_idx,
            step_end: outcome.stats.max_step_idx,
            phases_present: phases_present_names,
        },
    }
}

fn materialize_one(
    nodes: &[GhostPhaseLatticeNode],
    component: &GhostPhaseLatticeComponent,
    provenance_template: &GhostPhaseLatticeProvenance,
) -> ComponentManifoldBlocks {
    // Bucket component nodes by phase.
    let mut by_phase: HashMap<u8, Vec<&GhostPhaseLatticeNode>> = HashMap::new();
    let mut comp_aabb_min = [f32::INFINITY, f32::INFINITY, f32::INFINITY];
    let mut comp_aabb_max = [f32::NEG_INFINITY; 3];
    let mut node_lineage = Vec::with_capacity(component.node_indices.len());
    let mut driver_residues = BTreeSet::new();
    for &idx in &component.node_indices {
        let n = &nodes[idx as usize];
        by_phase.entry(n.protocol_phase).or_default().push(n);
        let causal_lead_residue = if n.causal_lead_residue == NODE_CAUSAL_LEAD_NONE {
            None
        } else {
            driver_residues.insert(n.causal_lead_residue);
            Some(n.causal_lead_residue)
        };
        node_lineage.push(GhostNodeLineage {
            tile_index: n.tile_index,
            stream_id: n.stream_id,
            site_id: n.site_id,
            frame_idx: n.frame_idx,
            step_idx: n.step_idx,
            protocol_phase: n.protocol_phase,
            step_bucket: n.step_bucket,
            causal_lead_residue,
        });
        for d in 0..3 {
            if n.aabb_min[d] < comp_aabb_min[d] {
                comp_aabb_min[d] = n.aabb_min[d];
            }
            if n.aabb_max[d] > comp_aabb_max[d] {
                comp_aabb_max[d] = n.aabb_max[d];
            }
        }
    }
    // If the component is empty (defensive), emit zero-extent AABBs
    // rather than infinities to keep downstream serialisation finite.
    if !comp_aabb_min[0].is_finite() {
        comp_aabb_min = [0.0, 0.0, 0.0];
        comp_aabb_max = [0.0, 0.0, 0.0];
    }
    node_lineage.sort_by_key(|n| {
        (
            n.step_idx,
            n.stream_id,
            n.frame_idx,
            n.site_id,
            n.tile_index,
        )
    });

    let mut phase_manifold = PhaseManifold::default();
    let mut mean_kl_by_phase: BTreeMap<String, f32> = BTreeMap::new();
    let mut mean_thermo_flux_by_phase: BTreeMap<String, [f32; 2]> = BTreeMap::new();
    let mut mean_water_density_by_phase: BTreeMap<String, f32> = BTreeMap::new();
    let mut driver_count: HashMap<u32, u32> = HashMap::new();
    let mut total_driver_resolved: u32 = 0;
    let mut any_water_finite = false;
    let mut so3_status_mask: u8 = 0;

    for phase_id in 0..N_PROTOCOL_PHASES {
        let group = match by_phase.get(&phase_id) {
            Some(v) if !v.is_empty() => v,
            _ => continue,
        };
        let phase_name = phase_name_for(phase_id).to_string();

        // AABB-volume-weighted centroid + AABB union + mean KL.
        let mut weight_sum = 0.0f64;
        let mut centroid_acc = [0.0f64; 3];
        let mut aabb_min = [f32::INFINITY; 3];
        let mut aabb_max = [f32::NEG_INFINITY; 3];
        let mut step_min = u64::MAX;
        let mut step_max = 0u64;
        let mut kl_sum = 0.0f64;
        let mut kl_count = 0u32;
        let mut tf0_sum = 0.0f64;
        let mut tf0_count = 0u32;
        let mut tf1_sum = 0.0f64;
        let mut tf1_count = 0u32;
        let mut wd_sum = 0.0f64;
        let mut wd_count = 0u32;
        let mut local_drivers: HashSet<u32> = HashSet::new();

        for n in group {
            // Volume-weighted centroid: empty AABBs (volume 0) fall back to
            // unit weight so a single-node-with-degenerate-box phase still
            // gets a centroid at that node's `centroid_xyz`.
            let v = n.aabb_volume() as f64;
            let w = if v > 0.0 { v } else { 1.0 };
            weight_sum += w;
            centroid_acc[0] += w * n.centroid_xyz[0] as f64;
            centroid_acc[1] += w * n.centroid_xyz[1] as f64;
            centroid_acc[2] += w * n.centroid_xyz[2] as f64;

            for d in 0..3 {
                if n.aabb_min[d] < aabb_min[d] {
                    aabb_min[d] = n.aabb_min[d];
                }
                if n.aabb_max[d] > aabb_max[d] {
                    aabb_max[d] = n.aabb_max[d];
                }
            }
            step_min = step_min.min(n.step_idx);
            step_max = step_max.max(n.step_idx);

            if n.kl_divergence.is_finite() {
                kl_sum += n.kl_divergence as f64;
                kl_count += 1;
            }
            if n.thermo_flux[0].is_finite() {
                tf0_sum += n.thermo_flux[0] as f64;
                tf0_count += 1;
            }
            if n.thermo_flux[1].is_finite() {
                tf1_sum += n.thermo_flux[1] as f64;
                tf1_count += 1;
            }
            if n.water_density.is_finite() {
                wd_sum += n.water_density as f64;
                wd_count += 1;
                any_water_finite = true;
            }
            if n.causal_lead_residue != NODE_CAUSAL_LEAD_NONE {
                local_drivers.insert(n.causal_lead_residue);
                *driver_count.entry(n.causal_lead_residue).or_default() += 1;
                total_driver_resolved += 1;
            }
            so3_status_mask |= n.so3_plane_status;
        }

        let centroid_xyz = if weight_sum > 0.0 {
            [
                (centroid_acc[0] / weight_sum) as f32,
                (centroid_acc[1] / weight_sum) as f32,
                (centroid_acc[2] / weight_sum) as f32,
            ]
        } else {
            [0.0; 3]
        };

        let mean_kl = if kl_count > 0 {
            (kl_sum / kl_count as f64) as f32
        } else {
            f32::NAN
        };
        let mean_tf0 = if tf0_count > 0 {
            (tf0_sum / tf0_count as f64) as f32
        } else {
            f32::NAN
        };
        let mean_tf1 = if tf1_count > 0 {
            (tf1_sum / tf1_count as f64) as f32
        } else {
            f32::NAN
        };
        let mean_wd = if wd_count > 0 {
            (wd_sum / wd_count as f64) as f32
        } else {
            f32::NAN
        };

        let mut driver_residues: Vec<u32> = local_drivers.into_iter().collect();
        driver_residues.sort_unstable();

        let aggregate = PhaseAggregate {
            centroid_xyz,
            aabb_min,
            aabb_max,
            n_nodes: group.len() as u32,
            mean_kl_divergence: mean_kl,
            n_finite_kl: kl_count,
            driver_residues,
            step_idx_min: if step_min == u64::MAX { 0 } else { step_min },
            step_idx_max: step_max,
        };
        match phase_id {
            0 => phase_manifold.cold_hold = Some(aggregate),
            1 => phase_manifold.heating = Some(aggregate),
            2 => phase_manifold.warm_hold = Some(aggregate),
            3 => phase_manifold.cooling = Some(aggregate),
            _ => {}
        }
        if mean_kl.is_finite() {
            mean_kl_by_phase.insert(phase_name.clone(), mean_kl);
        }
        if mean_tf0.is_finite() || mean_tf1.is_finite() {
            mean_thermo_flux_by_phase.insert(phase_name.clone(), [mean_tf0, mean_tf1]);
        }
        if mean_wd.is_finite() {
            mean_water_density_by_phase.insert(phase_name.clone(), mean_wd);
        }
    }

    // Component/phase cosine claims require exact accepted-edge endpoint and
    // score lineage. Global kernel counters cannot be apportioned by component
    // size or copied onto every phase transition. Until that lineage exists,
    // emit explicit unavailable status and an empty transition map.
    let intra_mean = if component.edge_lineage_exact && component.n_intra_edges > 0 {
        (component.intra_edge_score_sum / component.n_intra_edges as f64) as f32
    } else {
        0.0
    };
    let phase_transition_cosine: BTreeMap<String, f32> = BTreeMap::new();
    let intra_component_cosine_status = if component.edge_lineage_exact {
        "available_exact_edge_lineage"
    } else {
        "unavailable_no_accepted_edge_lineage"
    };
    let phase_transition_cosine_status = if component.edge_lineage_exact {
        // Exact per-transition aggregation is not yet represented separately
        // from the component totals, so remain unavailable even if a future
        // producer sets component edge lineage exact.
        "unavailable_no_phase_pair_edge_lineage"
    } else {
        "unavailable_no_accepted_edge_lineage"
    };

    // Driver persistence — normalised to sum to 1.0.
    let mut driver_residue_persistence: BTreeMap<u32, f32> = BTreeMap::new();
    if total_driver_resolved > 0 {
        for (resid, count) in driver_count {
            let frac = count as f32 / total_driver_resolved as f32;
            driver_residue_persistence.insert(resid, frac);
        }
    }

    let water_density_status = if any_water_finite {
        "available".to_string()
    } else {
        "unavailable_neutral".to_string()
    };

    let therm_ccns_lifecycle = ThermCcnsLifecycle {
        mean_kl_by_phase,
        mean_thermo_flux_by_phase,
        driver_residue_persistence,
        water_density_status,
        mean_water_density_by_phase,
    };

    let so3_manifold = So3Manifold {
        plane_status: So3PlaneStatus::from_mask(so3_status_mask),
        intra_component_mean_cosine: intra_mean,
        intra_component_cosine_status: intra_component_cosine_status.to_string(),
        n_intra_edges: if component.edge_lineage_exact {
            component.n_intra_edges
        } else {
            0
        },
        phase_transition_cosine,
        phase_transition_cosine_status: phase_transition_cosine_status.to_string(),
    };

    ComponentManifoldBlocks {
        ghost_phase_lattice: provenance_template.clone(),
        phase_manifold,
        therm_ccns_lifecycle,
        so3_manifold,
        component_aabb_min: comp_aabb_min,
        component_aabb_max: comp_aabb_max,
        node_lineage,
        driver_residues: driver_residues.into_iter().collect(),
        n_nodes: component.node_indices.len() as u32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ghost_phase_lattice::{
        GhostPhaseLatticeConfig, GhostPhaseLatticeRunStats, PHASE_COLD_HOLD, PHASE_COOLING,
        PHASE_HEATING, PHASE_WARM_HOLD, SO3_PLANE_GEOMETRY,
    };

    fn synth_node(
        idx: u32,
        step: u64,
        x: f32,
        y: f32,
        z: f32,
        phase: u8,
        kl: f32,
        driver: u32,
    ) -> GhostPhaseLatticeNode {
        let mut n = GhostPhaseLatticeNode::empty();
        n.tile_index = idx;
        n.frame_idx = idx as u64;
        n.site_id = 0;
        n.step_idx = step;
        n.protocol_phase = phase;
        n.aabb_min = [x - 0.5, y - 0.5, z - 0.5];
        n.aabb_max = [x + 0.5, y + 0.5, z + 0.5];
        n.centroid_xyz = [x, y, z];
        n.kl_divergence = kl;
        n.causal_lead_residue = driver;
        n.so3_plane_status = SO3_PLANE_GEOMETRY;
        n.so3_power_spectrum[0] = [0.4, 0.3, 0.15, 0.08, 0.04, 0.03];
        n
    }

    #[test]
    fn materializer_emits_one_block_per_phase_present() {
        let nodes = vec![
            synth_node(0, 100, 10.0, 0.0, 0.0, PHASE_COLD_HOLD, 0.1, 42),
            synth_node(1, 200, 11.0, 0.0, 0.0, PHASE_HEATING, 0.2, 42),
            synth_node(2, 300, 12.0, 0.0, 0.0, PHASE_WARM_HOLD, 0.3, 17),
        ];
        let component = GhostPhaseLatticeComponent {
            node_indices: vec![0, 1, 2],
            n_intra_edges: 2,
            intra_edge_score_sum: 1.7,
            edge_lineage_exact: false,
        };
        let outcome = GhostPhaseLatticeOutcome {
            components: vec![component],
            accepted_edges: Vec::new(),
            edge_conservation: Default::default(),
            stats: GhostPhaseLatticeRunStats {
                n_nodes: 3,
                n_lattice_cells: 3,
                n_directed_edges: 2,
                min_step_idx: 100,
                max_step_idx: 300,
                ..Default::default()
            },
            config: GhostPhaseLatticeConfig::default(),
        };
        let blocks = materialize_components(&nodes, &outcome);
        assert_eq!(blocks.len(), 1);
        let b = &blocks[0];
        assert!(b.phase_manifold.cold_hold.is_some());
        assert!(b.phase_manifold.heating.is_some());
        assert!(b.phase_manifold.warm_hold.is_some());
        assert!(b.phase_manifold.cooling.is_none());
        assert_eq!(b.ghost_phase_lattice.backend, "ghost_phase_lattice_4d");
        assert_eq!(b.ghost_phase_lattice.lattice_extent.step_start, 100);
        assert_eq!(b.ghost_phase_lattice.lattice_extent.step_end, 300);
        assert_eq!(
            b.ghost_phase_lattice.lattice_extent.phases_present,
            vec!["cold_hold", "heating", "warm_hold"]
        );

        // Mean KL by phase populated for the three present phases.
        assert!(b
            .therm_ccns_lifecycle
            .mean_kl_by_phase
            .contains_key("cold_hold"));
        assert_eq!(
            b.therm_ccns_lifecycle.water_density_status,
            "unavailable_neutral"
        );
        assert_eq!(b.node_lineage.len(), 3);
        assert_eq!(b.node_lineage[0].step_idx, 100);
        assert_eq!(b.driver_residues, vec![17, 42]);

        // Driver persistence: 42 fired in 2 of 3 frames, 17 in 1 of 3.
        let p42 = b.therm_ccns_lifecycle.driver_residue_persistence[&42];
        let p17 = b.therm_ccns_lifecycle.driver_residue_persistence[&17];
        assert!((p42 - 2.0 / 3.0).abs() < 1e-5);
        assert!((p17 - 1.0 / 3.0).abs() < 1e-5);

        // SO(3) plane status: only geometry populated.
        assert_eq!(b.so3_manifold.plane_status.geometry, "populated");
        assert_eq!(b.so3_manifold.plane_status.causality, "sentinel");
    }

    #[test]
    fn dumbbell_split_produces_separate_components_with_distinct_aabbs() {
        // Two spatially-distinct lobes, both in cold_hold, both in the
        // same step bucket: the kernel will not connect them (no AABB
        // overlap), so the materializer must see two components each
        // with its own AABB.
        let lobe_a = vec![
            synth_node(0, 100, 0.0, 12.0, 0.0, PHASE_COLD_HOLD, 0.1, 42),
            synth_node(1, 110, 0.5, 12.5, 0.5, PHASE_COLD_HOLD, 0.1, 42),
        ];
        let lobe_b = vec![
            synth_node(2, 100, 0.0, -8.0, 0.0, PHASE_COLD_HOLD, 0.1, 17),
            synth_node(3, 110, 0.5, -7.5, 0.5, PHASE_COLD_HOLD, 0.1, 17),
        ];
        let nodes: Vec<GhostPhaseLatticeNode> =
            lobe_a.into_iter().chain(lobe_b.into_iter()).collect();
        let outcome = GhostPhaseLatticeOutcome {
            components: vec![
                GhostPhaseLatticeComponent {
                    node_indices: vec![0, 1],
                    n_intra_edges: 1,
                    intra_edge_score_sum: 0.85,
                    edge_lineage_exact: false,
                },
                GhostPhaseLatticeComponent {
                    node_indices: vec![2, 3],
                    n_intra_edges: 1,
                    intra_edge_score_sum: 0.85,
                    edge_lineage_exact: false,
                },
            ],
            accepted_edges: Vec::new(),
            edge_conservation: Default::default(),
            stats: GhostPhaseLatticeRunStats {
                n_nodes: 4,
                n_lattice_cells: 2,
                n_directed_edges: 2,
                min_step_idx: 100,
                max_step_idx: 110,
                ..Default::default()
            },
            config: GhostPhaseLatticeConfig::default(),
        };
        let blocks = materialize_components(&nodes, &outcome);
        assert_eq!(blocks.len(), 2);
        // No AABB overlap — Y-spans are disjoint (12.0..13.0 vs -8.5..-7.0).
        let a = &blocks[0];
        let b = &blocks[1];
        let a_y_overlap_b = a.component_aabb_max[1] >= b.component_aabb_min[1]
            && a.component_aabb_min[1] <= b.component_aabb_max[1];
        assert!(
            !a_y_overlap_b,
            "Dumbbell-split test failed: components must have disjoint Y-spans \
             but got A=[{:?}..{:?}] B=[{:?}..{:?}]",
            a.component_aabb_min[1],
            a.component_aabb_max[1],
            b.component_aabb_min[1],
            b.component_aabb_max[1]
        );
    }

    #[test]
    fn unavailable_neutral_when_no_thermodynamic_data() {
        let nodes = vec![synth_node(
            0,
            100,
            0.0,
            0.0,
            0.0,
            PHASE_COLD_HOLD,
            f32::NAN,
            0,
        )];
        let outcome = GhostPhaseLatticeOutcome {
            components: vec![GhostPhaseLatticeComponent {
                node_indices: vec![0],
                n_intra_edges: 0,
                intra_edge_score_sum: 0.0,
                edge_lineage_exact: false,
            }],
            accepted_edges: Vec::new(),
            edge_conservation: Default::default(),
            stats: GhostPhaseLatticeRunStats {
                n_nodes: 1,
                ..Default::default()
            },
            config: GhostPhaseLatticeConfig::default(),
        };
        let blocks = materialize_components(&nodes, &outcome);
        let b = &blocks[0];
        assert_eq!(
            b.therm_ccns_lifecycle.water_density_status,
            "unavailable_neutral"
        );
        // No finite KL → mean_kl_by_phase is empty.
        assert!(b.therm_ccns_lifecycle.mean_kl_by_phase.is_empty());
    }

    #[test]
    fn temporal_purity_phase_centroids_are_distinct() {
        // A site with three temporal phases at three distinct positions
        // — directive Gate 3 requires the JSON to carry distinct
        // centroids per phase.
        let nodes = vec![
            synth_node(0, 100, 10.0, 0.0, 0.0, PHASE_COLD_HOLD, 0.1, 42),
            synth_node(1, 110, 10.1, 0.0, 0.0, PHASE_COLD_HOLD, 0.1, 42),
            synth_node(2, 600, 12.0, 0.0, 0.0, PHASE_HEATING, 0.4, 42),
            synth_node(3, 1100, 14.0, 0.0, 0.0, PHASE_COOLING, 0.05, 42),
        ];
        let outcome = GhostPhaseLatticeOutcome {
            components: vec![GhostPhaseLatticeComponent {
                node_indices: vec![0, 1, 2, 3],
                n_intra_edges: 3,
                intra_edge_score_sum: 2.55,
                edge_lineage_exact: false,
            }],
            accepted_edges: Vec::new(),
            edge_conservation: Default::default(),
            stats: GhostPhaseLatticeRunStats {
                n_nodes: 4,
                ..Default::default()
            },
            config: GhostPhaseLatticeConfig::default(),
        };
        let blocks = materialize_components(&nodes, &outcome);
        let pm = &blocks[0].phase_manifold;
        let cold = pm.cold_hold.as_ref().unwrap();
        let heat = pm.heating.as_ref().unwrap();
        let cool = pm.cooling.as_ref().unwrap();
        assert!((cold.centroid_xyz[0] - 10.05).abs() < 1e-2);
        assert!((heat.centroid_xyz[0] - 12.0).abs() < 1e-2);
        assert!((cool.centroid_xyz[0] - 14.0).abs() < 1e-2);
    }
}
