//! Regression: lattice components materialize into sites with their own
//! identity, region and centroid — they do not borrow the CCL cluster domain.
//!
//! # What was broken
//!
//! Both `lattice_region` call sites passed `materialized_site_id = None`, so
//! every `PrismThermSiteResult.site_id` fell back to `-1`. All four downstream
//! consumers join with `.find(|s| s.site_id == <cluster_id>)`, which can never
//! match `-1`, so a whole lattice surface was computed and then discarded while
//! `prism_therm_status.json` still reported `observed` / `lattice`. `sdst_report`
//! then wrote every pocket centroid as `[0,0,0]`.
//!
//! # The contract this pins
//!
//! 1. Every materialized site gets an independently *assigned* identity — the
//!    component index is never reused as a cluster id.
//! 2. A lattice component and a CCL cluster carrying the same integer stay
//!    distinct objects.
//! 3. The centroid is the real centre of the component's own region.
//! 4. No sentinel identity survives materialization.

#![cfg(feature = "gpu")]

use prism_nhs::sdst_bridge::{
    materialize_lattice_sites, materialize_lattice_sites_with_node_centroids,
    project_lattice_binding_sites, reconcile_lattice_site_population,
    validate_materialized_sites, AnalyzedSiteIdentity, LatticeComponentId,
    LatticeMaterializationRequirement, LatticeSiteTerminalDisposition, MaterializedSiteId,
    MaterializedSiteSurface, PrismThermAnalysis, PrismThermSiteResult, SdstAnalysisOutcome,
    SdstAnalysisStatus, SiteUnmatchedReason, ThermClass,
};
use prism_nhs::sdst_report::resolve_materialized_site_geometry;
use prism_nhs::site_manifest::ClusterId;

#[test]
fn native_centroid_uses_observed_nodes_and_rejects_missing_geometry() {
    use prism_nhs::ghost_phase_lattice::{GhostPhaseLatticeComponent, GhostPhaseLatticeNode};
    let mut first = GhostPhaseLatticeNode::empty();
    first.aabb_min = [0.0, 0.0, 0.0];
    first.aabb_max = [2.0, 2.0, 2.0];
    first.centroid_xyz = [0.5, 0.5, 0.5];
    let mut second = GhostPhaseLatticeNode::empty();
    second.aabb_min = [8.0, 0.0, 0.0];
    second.aabb_max = [9.0, 1.0, 1.0];
    second.centroid_xyz = [8.5, 0.5, 0.5];
    let nodes = [first, second];
    let mut components = vec![GhostPhaseLatticeComponent {
        node_indices: vec![0, 1], n_intra_edges: 1,
        intra_edge_score_sum: 1.0, edge_lineage_exact: true,
    }];
    let selected = [(0, [0.0, 0.0, 0.0], [9.0, 2.0, 2.0])];
    let sites = materialize_lattice_sites_with_node_centroids(&selected, &nodes, &components)
        .expect("observed node geometry materializes");
    assert!((sites[0].centroid[0] - 12.5 / 9.0).abs() < 1e-6);
    assert_eq!(sites[0].centroid[1], 0.5);
    assert_ne!(sites[0].centroid, [4.5, 1.0, 1.0]);
    assert!(sites[0].provenance.contains("GhostPhaseLatticeNode.centroid_xyz"));

    components[0].node_indices = vec![2];
    assert!(materialize_lattice_sites_with_node_centroids(&selected, &nodes, &components).is_err());
    components[0].node_indices = vec![0, 0];
    assert!(materialize_lattice_sites_with_node_centroids(&selected, &nodes, &components).is_err());
    components[0].node_indices.clear();
    assert!(materialize_lattice_sites_with_node_centroids(&selected, &nodes, &components).is_err());
    components[0].node_indices = vec![0];
    first.centroid_xyz[0] = f32::NAN;
    assert!(materialize_lattice_sites_with_node_centroids(&selected, &[first], &components).is_err());
}

/// Deliberately non-contiguous component indices: a selection policy drops
/// components, so slot position and component index must not be conflated.
fn fixture() -> Vec<(usize, [f32; 3], [f32; 3])> {
    vec![
        (4, [0.0, 0.0, 0.0], [2.0, 4.0, 6.0]),
        (0, [-3.0, 1.0, 1.0], [-1.0, 5.0, 3.0]),
        (7, [10.0, 10.0, 10.0], [12.0, 14.0, 18.0]),
    ]
}

fn therm_result(
    identity: AnalyzedSiteIdentity,
    materialized_site_id: Option<MaterializedSiteId>,
    is_hysteretic: bool,
) -> PrismThermSiteResult {
    PrismThermSiteResult {
        analyzed_site_identity: identity,
        materialized_site_id,
        site_id: materialized_site_id.map_or(-1, |id| id.0 as i32),
        asymmetry_score: if is_hysteretic { 0.8 } else { 0.1 },
        is_hysteretic,
        heating_spike_rate: 1.0,
        cooling_spike_rate: 0.2,
        heating_spike_count: 10,
        cooling_spike_count: 2,
        tau: 1.2,
        tau_stderr: 0.05,
        ccns_classification: "Soc".to_string(),
        druggability: 0.9,
        n_avalanches: 3,
        relative_asymmetry: 1.0,
        therm_class: ThermClass::Dynamic,
        tide_decomposition: Vec::new(),
    }
}

fn analysis(sites: Vec<PrismThermSiteResult>) -> PrismThermAnalysis {
    let hysteretic_site_count = sites.iter().filter(|site| site.is_hysteretic).count();
    PrismThermAnalysis {
        outcome: SdstAnalysisOutcome {
            status: SdstAnalysisStatus::Complete,
            requested_site_count: sites.len(),
            materialized_site_count: sites.len(),
            stages: Vec::new(),
        },
        sdst_event_count: 17,
        hysteresis_threshold: 0.2,
        hysteretic_site_count,
        sites,
        global_pockets: Vec::new(),
        total_avalanches: 4,
        tide_residues_mapped: 0,
    }
}

#[test]
fn fixture_is_adversarial() {
    // Guard: if the fixture were already 0,1,2 in order, an implementation that
    // simply reused the component index would pass every assertion below.
    let f = fixture();
    let indices: Vec<usize> = f.iter().map(|(i, _, _)| *i).collect();
    assert_ne!(
        indices,
        vec![0, 1, 2],
        "vacuity guard: fixture indices must not already equal their slot positions"
    );
}

/// A. Distinct components produce distinct materialized identities.
#[test]
fn distinct_components_get_distinct_materialized_ids() {
    let sites = materialize_lattice_sites(&fixture()).expect("materialization");
    assert_eq!(sites.len(), 3);
    let ids: Vec<MaterializedSiteId> = sites.iter().map(|s| s.materialized_site_id).collect();
    let mut sorted = ids.clone();
    sorted.sort_by_key(|m| m.0);
    sorted.dedup_by_key(|m| m.0);
    assert_eq!(
        sorted.len(),
        ids.len(),
        "materialized ids collided: {ids:?}"
    );
}

/// A (cont). The assigned identity is not the component index.
#[test]
fn materialized_id_is_assigned_not_the_component_index() {
    let sites = materialize_lattice_sites(&fixture()).expect("materialization");
    // Component 4 sits in slot 0, so a reused index would show up immediately.
    assert_eq!(sites[0].materialized_site_id, MaterializedSiteId(0));
    assert_eq!(
        sites[0].analyzed_site_identity,
        AnalyzedSiteIdentity::LatticeComponent(LatticeComponentId(4))
    );
    assert_ne!(
        sites[0].materialized_site_id.0 as u64, 4,
        "component index was reused as the materialized identity"
    );
}

/// B. Equal integers in the two domains are not the same object.
#[test]
fn lattice_component_and_cluster_with_equal_integers_do_not_collide() {
    let lattice = AnalyzedSiteIdentity::LatticeComponent(LatticeComponentId(7));
    let cluster = AnalyzedSiteIdentity::Cluster(ClusterId(7));
    assert_ne!(
        lattice, cluster,
        "a lattice component and a CCL cluster with the same integer compared equal; \
         the identity domains have been collapsed"
    );
}

/// D. The centroid is the real centre of the source region.
#[test]
fn centroid_is_the_real_source_centroid_not_a_default() {
    let sites = materialize_lattice_sites(&fixture()).expect("materialization");
    assert_eq!(sites[0].centroid, [1.0, 2.0, 3.0]);
    assert_eq!(sites[1].centroid, [-2.0, 3.0, 2.0]);
    assert_eq!(sites[2].centroid, [11.0, 12.0, 14.0]);
    for s in &sites {
        assert_ne!(
            s.centroid,
            [0.0, 0.0, 0.0],
            "centroid defaulted to the coordinate origin"
        );
    }
}

/// D (cont). The source region is carried verbatim.
#[test]
fn source_region_and_surface_are_preserved() {
    let sites = materialize_lattice_sites(&fixture()).expect("materialization");
    assert_eq!(sites[2].aabb_min, [10.0, 10.0, 10.0]);
    assert_eq!(sites[2].aabb_max, [12.0, 14.0, 18.0]);
    assert_eq!(
        sites[2].source_surface,
        MaterializedSiteSurface::LatticeComponent
    );
    assert!(
        !sites[2].provenance.is_empty(),
        "materialization provenance is absent"
    );
}

/// E. No sentinel identity survives materialization.
#[test]
fn no_sentinel_identity_survives() {
    let sites = materialize_lattice_sites(&fixture()).expect("materialization");
    for s in &sites {
        // -1 is unrepresentable in the materialized domain by construction; this
        // pins that the domain stays unsigned rather than reverting to i32.
        let _: u32 = s.materialized_site_id.0;
        assert!(matches!(
            s.analyzed_site_identity,
            AnalyzedSiteIdentity::LatticeComponent(_)
        ));
    }
}

/// An invalid source region is refused rather than materialized into a site
/// whose centroid would be meaningless.
#[test]
fn invalid_source_region_is_refused() {
    let inverted = vec![(0usize, [5.0, 0.0, 0.0], [1.0, 1.0, 1.0])];
    let err = materialize_lattice_sites(&inverted).expect_err("inverted AABB must be refused");
    assert!(
        format!("{err}").contains("InvalidSourceRegion"),
        "expected a typed InvalidSourceRegion cause, got: {err}"
    );

    let nan = vec![(0usize, [f32::NAN, 0.0, 0.0], [1.0, 1.0, 1.0])];
    let err = materialize_lattice_sites(&nan).expect_err("non-finite AABB must be refused");
    assert!(
        format!("{err}").contains("InvalidSourceRegion"),
        "got: {err}"
    );
}

/// An empty selection materializes nothing and is not an error.
#[test]
fn empty_selection_materializes_nothing() {
    let sites = materialize_lattice_sites(&[]).expect("empty selection is valid");
    assert!(sites.is_empty());
}

/// C. A lattice result reaches the correct typed binding-site projection.
#[test]
fn lattice_result_reaches_correct_binding_projection_without_cluster_authority() {
    let sites = materialize_lattice_sites(&fixture()).expect("materialization");
    let projections = project_lattice_binding_sites(&sites).expect("typed projections");
    assert_eq!(
        projections.len(),
        sites.len(),
        "non-vacuity: projections absent"
    );

    let result = therm_result(
        sites[1].analyzed_site_identity,
        Some(sites[1].materialized_site_id),
        true,
    );
    let analysis = analysis(vec![result]);
    let found = projections[1]
        .find_result(&analysis)
        .expect("typed lookup")
        .expect("matching result");
    assert!(found.matches_materialized(sites[1].materialized_site_id));
    assert!(found.matches_analyzed(sites[1].analyzed_site_identity));

    let json = serde_json::to_value(&projections[1]).expect("serialize projection");
    assert!(json.get("materialized_site_id").is_some());
    assert!(json.get("source_component_id").is_some());
    assert!(
        json.get("cluster_id").is_none(),
        "a lattice projection asserted CCL cluster authority"
    );
}

/// Equal integers in CCL and materialized domains cannot hijack typed lookup.
#[test]
fn typed_result_lookup_is_immune_to_numeric_cluster_collision() {
    let sites = materialize_lattice_sites(&fixture()).expect("materialization");
    let projections = project_lattice_binding_sites(&sites).expect("typed projections");
    let cluster_collision = therm_result(
        AnalyzedSiteIdentity::Cluster(ClusterId(sites[0].materialized_site_id.0)),
        None,
        false,
    );
    let lattice = therm_result(
        sites[0].analyzed_site_identity,
        Some(sites[0].materialized_site_id),
        true,
    );
    let analysis = analysis(vec![cluster_collision, lattice]);
    let found = projections[0]
        .find_result(&analysis)
        .expect("typed lookup")
        .expect("lattice result");
    assert!(matches!(
        found.analyzed_site_identity,
        AnalyzedSiteIdentity::LatticeComponent(_)
    ));
    assert!(
        found.is_hysteretic,
        "the CCL collision was selected instead"
    );
}

/// Duplicate-id injection is caught on externally constructed fixtures.
#[test]
fn duplicate_materialized_id_injection_is_rejected() {
    let mut sites = materialize_lattice_sites(&fixture()).expect("materialization");
    assert!(
        sites.len() > 1,
        "non-vacuity: duplicate injection needs two rows"
    );
    sites[1].materialized_site_id = sites[0].materialized_site_id;
    let error = validate_materialized_sites(&sites).expect_err("duplicate id must fail");
    assert!(format!("{error}").contains("DuplicateMaterializedId"));
}

/// F. Conservation holds when all analyzed sites merge.
#[test]
fn conservation_all_sites_merged() {
    let sites = materialize_lattice_sites(&fixture()).expect("materialization");
    let results = sites
        .iter()
        .enumerate()
        .map(|(index, site)| {
            therm_result(
                site.analyzed_site_identity,
                Some(site.materialized_site_id),
                index != 1,
            )
        })
        .collect();
    let conservation = reconcile_lattice_site_population(
        sites.len(),
        &sites,
        &analysis(results),
        LatticeMaterializationRequirement::Required,
        "binding_sites_lattice_merge",
    )
    .expect("conservation");
    assert_eq!(conservation.counts.analyzed_site_count, 3);
    assert_eq!(conservation.counts.merged_site_count, 3);
    assert_eq!(conservation.counts.unmatched_site_count, 0);
    assert_eq!(conservation.counts.hysteresis_positive_count, 2);
    assert!(conservation.promotable);
    assert!(conservation.dispositions.iter().all(|row| {
        row.disposition == LatticeSiteTerminalDisposition::Merged && row.unmatched_reason.is_none()
    }));
}

/// A selected site cannot disappear before reconciliation without an explicit
/// terminal record. The canonical materialized collection is the analyzed
/// population, so their counts must agree at this seam.
#[test]
fn selected_population_cannot_exceed_materialized_analysis_population() {
    let sites = materialize_lattice_sites(&fixture()[..2]).expect("materialization");
    assert_eq!(
        sites.len(),
        2,
        "non-vacuity: rejection requires a smaller materialized population"
    );
    let results = sites
        .iter()
        .map(|site| {
            therm_result(
                site.analyzed_site_identity,
                Some(site.materialized_site_id),
                true,
            )
        })
        .collect();
    let error = reconcile_lattice_site_population(
        3,
        &sites,
        &analysis(results),
        LatticeMaterializationRequirement::Required,
        "binding_sites_lattice_merge",
    )
    .expect_err("a selected site missing from the materialized population must be rejected");
    let message = format!("{error:#}");
    assert!(
        message.contains("selected_site_count 3")
            && message.contains("materialized analyzed_site_count 2"),
        "rejection did not preserve both conflicting population counts: {message}"
    );
}

/// F. Conservation holds for a real partial merge and preserves both failures.
#[test]
fn conservation_partial_merge_preserves_unmatched_rows() {
    let sites = materialize_lattice_sites(&fixture()).expect("materialization");
    let results = vec![
        therm_result(
            sites[0].analyzed_site_identity,
            Some(sites[0].materialized_site_id),
            true,
        ),
        therm_result(sites[1].analyzed_site_identity, None, false),
    ];
    let conservation = reconcile_lattice_site_population(
        sites.len(),
        &sites,
        &analysis(results),
        LatticeMaterializationRequirement::Required,
        "binding_sites_lattice_merge",
    )
    .expect("conservation");
    assert_eq!(conservation.counts.analyzed_site_count, 3);
    assert_eq!(conservation.counts.merged_site_count, 1);
    assert_eq!(conservation.counts.unmatched_site_count, 2);
    assert_eq!(
        conservation.counts.analyzed_site_count,
        conservation.counts.merged_site_count + conservation.counts.unmatched_site_count
    );
    assert!(!conservation.promotable);
    let reasons: Vec<_> = conservation
        .dispositions
        .iter()
        .filter_map(|row| row.unmatched_reason)
        .collect();
    assert!(reasons.contains(&SiteUnmatchedReason::NoMaterializationMapping));
    assert!(reasons.contains(&SiteUnmatchedReason::MissingCompanionData));
    assert!(conservation.dispositions.iter().all(|row| {
        row.centroid.iter().all(|value| value.is_finite())
            && row.aabb_min.iter().all(|value| value.is_finite())
            && row.aabb_max.iter().all(|value| value.is_finite())
    }));
    assert!(conservation
        .dispositions
        .iter()
        .filter(|row| row.disposition == LatticeSiteTerminalDisposition::Unmatched)
        .all(|row| !row.promotable));
}

/// F. No matches is represented as a complete unmatched partition.
#[test]
fn conservation_no_matches_has_no_default_fallthrough() {
    let sites = materialize_lattice_sites(&fixture()).expect("materialization");
    let empty = analysis(Vec::new());
    let projections = project_lattice_binding_sites(&sites).expect("typed projections");
    assert!(
        projections[0]
            .find_result(&empty)
            .expect("typed lookup")
            .is_none(),
        "lookup silently returned a default result"
    );
    let conservation = reconcile_lattice_site_population(
        sites.len(),
        &sites,
        &empty,
        LatticeMaterializationRequirement::Optional,
        "binding_sites_lattice_merge",
    )
    .expect("conservation");
    assert_eq!(conservation.counts.analyzed_site_count, sites.len());
    assert_eq!(conservation.counts.merged_site_count, 0);
    assert_eq!(conservation.counts.unmatched_site_count, sites.len());
    assert_eq!(conservation.counts.hysteresis_positive_count, 0);
    assert!(
        conservation.counts.analyzed_site_count > 0,
        "non-vacuity: zero-usable-site guard needs an analyzed population"
    );
    assert!(
        !conservation.promotable,
        "an optional profile promoted a non-empty analysis with zero usable sites"
    );
    assert!(conservation.dispositions.iter().all(|row| {
        row.disposition == LatticeSiteTerminalDisposition::Unmatched
            && row.unmatched_reason == Some(SiteUnmatchedReason::MissingCompanionData)
            && row.failed_consumer.as_deref() == Some("binding_sites_lattice_merge")
    }));
}

/// G. A missing mapping is typed and makes a required profile non-promotable.
#[test]
fn missing_mapping_is_typed_and_required_profile_is_non_promotable() {
    let sites = materialize_lattice_sites(&fixture()[..1]).expect("materialization");
    assert_eq!(sites.len(), 1, "non-vacuity: one analyzed site required");
    let unmapped = therm_result(sites[0].analyzed_site_identity, None, true);
    let conservation = reconcile_lattice_site_population(
        1,
        &sites,
        &analysis(vec![unmapped.clone()]),
        LatticeMaterializationRequirement::Required,
        "therm_postmd_binding_sites",
    )
    .expect("typed unmatched receipt");
    let row = &conservation.dispositions[0];
    assert_eq!(
        row.unmatched_reason,
        Some(SiteUnmatchedReason::NoMaterializationMapping)
    );
    assert_eq!(
        row.materialized_site_id,
        Some(sites[0].materialized_site_id)
    );
    assert!(!row.promotable);
    assert!(!conservation.promotable);

    let geometry = resolve_materialized_site_geometry(&unmapped, &sites)
        .expect("typed report geometry resolution");
    assert_eq!(geometry.centroid, None);
    assert_eq!(
        geometry.unmatched_reason,
        Some(SiteUnmatchedReason::NoMaterializationMapping)
    );
}

/// A result carrying a materialized id but the wrong source domain is retained
/// as a typed, non-promotable mismatch under a required profile.
#[test]
fn consumer_domain_mismatch_is_typed_and_non_promotable() {
    let sites = materialize_lattice_sites(&fixture()[..1]).expect("materialization");
    assert_eq!(sites.len(), 1, "non-vacuity: one lattice site required");
    let collision = therm_result(
        AnalyzedSiteIdentity::Cluster(ClusterId(4)),
        Some(sites[0].materialized_site_id),
        true,
    );
    let conservation = reconcile_lattice_site_population(
        1,
        &sites,
        &analysis(vec![collision]),
        LatticeMaterializationRequirement::Required,
        "therm_postmd_binding_sites",
    )
    .expect("typed domain-mismatch receipt");
    let row = &conservation.dispositions[0];
    assert_eq!(
        row.unmatched_reason,
        Some(SiteUnmatchedReason::ConsumerDomainMismatch)
    );
    assert_eq!(row.centroid, [1.0, 2.0, 3.0]);
    assert!(!row.promotable);
    assert!(!conservation.promotable);
}

/// Report geometry uses the real materialized centroid and never an origin fallback.
#[test]
fn report_geometry_resolves_real_centroid_or_typed_unmatched() {
    let sites = materialize_lattice_sites(&fixture()).expect("materialization");
    let mapped = therm_result(
        sites[2].analyzed_site_identity,
        Some(sites[2].materialized_site_id),
        false,
    );
    let geometry =
        resolve_materialized_site_geometry(&mapped, &sites).expect("mapped report geometry");
    assert_eq!(geometry.centroid, Some([11.0, 12.0, 14.0]));
    assert_eq!(geometry.aabb_min, Some([10.0, 10.0, 10.0]));
    assert_eq!(geometry.aabb_max, Some([12.0, 14.0, 18.0]));
    assert_eq!(geometry.unmatched_reason, None);

    let missing = therm_result(
        sites[2].analyzed_site_identity,
        Some(MaterializedSiteId(999)),
        false,
    );
    let geometry =
        resolve_materialized_site_geometry(&missing, &sites).expect("unmatched report geometry");
    assert_eq!(geometry.centroid, None);
    assert_ne!(geometry.centroid, Some([0.0, 0.0, 0.0]));
    assert_eq!(
        geometry.unmatched_reason,
        Some(SiteUnmatchedReason::MissingCompanionData)
    );
}
