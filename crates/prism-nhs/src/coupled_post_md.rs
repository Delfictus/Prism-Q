//! Stream-scoped consumption of committed coupled Ghost lattice state by SDST.

use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::collections::BTreeSet;

use crate::coupled_md::CoupledPostMdStream;
use crate::fused_engine::CryoUvProtocol;
use crate::ghost_phase_lattice::{
    GhostPhaseLatticeComponent, GhostPhaseLatticeNode, GhostPhaseLatticeOutcome,
    GhostSegmentLatticeLoad, StreamProtocolPhaseBinding,
};
use crate::sdst_bridge::{
    materialize_lattice_sites_with_node_centroids, reconcile_lattice_site_population,
    select_lattice_components, LatticeMaterializationRequirement, LatticeSiteConservation,
    MaterializedSite, PrismThermAnalysis, SdstBridge,
};

/// Site identities are scoped by the enclosing source stream and topology.
#[derive(Debug, Serialize)]
pub struct CoupledStreamThermReport {
    pub source_stream_id: u32,
    pub topology_source_sha256: String,
    pub seed: u64,
    pub physical_steps: i32,
    pub protocol: serde_json::Value,
    pub protocol_state_hash: String,
    pub phase_schedule_digest_sha256: String,
    pub condition_id: String,
    pub snapshot_count: usize,
    pub lattice_node_count: usize,
    pub selected_parent_component_indices: Vec<usize>,
    pub input_spike_count: usize,
    pub ingested_event_count: u32,
    pub sampling_policy: &'static str,
    pub sites: Vec<MaterializedSite>,
    pub analysis: PrismThermAnalysis,
    pub conservation: LatticeSiteConservation,
}

#[derive(Debug, Serialize)]
pub struct CoupledLatticeThermReport {
    pub schema_kind: &'static str,
    pub schema_version: u32,
    pub identity_scope: &'static str,
    pub component_projection: &'static str,
    pub min_nodes_per_stream: u32,
    pub max_sites_per_stream: usize,
    pub committed_lattice_node_count: usize,
    pub streams: Vec<CoupledStreamThermReport>,
}

fn validate_topology_binding(stream_id: u32, source: Option<&str>, lineage: &str) -> Result<()> {
    let source = source.with_context(|| {
        format!("coupled stream {stream_id} has no retained topology source SHA256")
    })?;
    if source.len() != 64
        || !source.bytes().all(|byte| byte.is_ascii_hexdigit())
        || source != lineage
    {
        bail!(
            "coupled stream {stream_id} retained topology does not match committed Ghost lineage"
        );
    }
    Ok(())
}

fn retained_protocol_binding(
    stream_id: u32,
    protocol: &CryoUvProtocol,
    physical_steps: i32,
    condition_id: &str,
) -> Result<StreamProtocolPhaseBinding> {
    if physical_steps <= 0 {
        bail!("coupled stream {stream_id} has no completed physical steps");
    }
    let expected_prefix = match stream_id {
        0 => "coupled_reference::",
        1 => "coupled_design::",
        _ => bail!("unexpected coupled stream identity {stream_id}"),
    };
    if !condition_id.starts_with(expected_prefix) || condition_id.len() == expected_prefix.len() {
        bail!("coupled stream {stream_id} condition identity does not match its producer role");
    }
    crate::coupled_md::coupled_protocol_phase_binding(
        protocol,
        stream_id,
        physical_steps as u64,
        condition_id.to_string(),
    )
}

fn verify_protocol_identity(
    expected: &StreamProtocolPhaseBinding,
    protocol_state_hash: &str,
    phase_schedule_digest_sha256: &str,
) -> Result<()> {
    if expected.protocol_state_hash != protocol_state_hash
        || expected.phase_schedule_digest_sha256 != phase_schedule_digest_sha256
    {
        bail!(
            "coupled stream {} retained protocol differs from committed Ghost protocol/schedule",
            expected.stream_id
        );
    }
    Ok(())
}

fn verify_stream_lineage(
    stream: &CoupledPostMdStream,
    nodes: &[GhostPhaseLatticeNode],
    lineage: &GhostSegmentLatticeLoad,
) -> Result<(usize, StreamProtocolPhaseBinding)> {
    let receipts: Vec<_> = lineage
        .streams
        .iter()
        .filter(|receipt| u32::from(receipt.stream_id) == stream.source_stream_id)
        .collect();
    if receipts.len() != 1 {
        bail!(
            "coupled stream {} lacks one exact committed conservation receipt",
            stream.source_stream_id
        );
    }
    let receipt = receipts[0];
    let expected = retained_protocol_binding(
        stream.source_stream_id,
        &stream.protocol,
        stream.physical_steps,
        &receipt.condition_id,
    )?;
    if receipt.protocol_name != expected.protocol_name {
        bail!(
            "coupled stream {} protocol producer name differs from committed receipt",
            stream.source_stream_id
        );
    }
    verify_protocol_identity(
        &expected,
        &receipt.protocol_state_hash,
        &receipt.phase_schedule_digest_sha256,
    )?;
    let mut count = 0usize;
    for (index, (node, row)) in nodes.iter().zip(&lineage.record_lineage).enumerate() {
        if u32::from(node.stream_id) != stream.source_stream_id {
            continue;
        }
        validate_topology_binding(
            stream.source_stream_id,
            stream.topology_source_sha256.as_deref(),
            &row.topology_sha256,
        )?;
        verify_protocol_identity(
            &expected,
            &row.protocol_state_hash,
            &row.phase_schedule_digest_sha256,
        )?;
        let persisted_node = &lineage.nodes[index];
        if row.global_lattice_index != index as u64
            || usize::try_from(node.tile_index)? != index
            || row.stream_id != node.stream_id
            || row.stream_seed != stream.seed
            || row.frame_idx != node.frame_idx
            || row.step_idx != node.step_idx
            || row.site_id != node.site_id
            || row.protocol_phase != node.protocol_phase
            || row.condition_id != expected.condition_id
            || expected.checked_phase_for(node.step_idx) != Some(node.protocol_phase)
            || node.step_idx > stream.physical_steps as u64
            || persisted_node.tile_index != node.tile_index
            || persisted_node.stream_id != node.stream_id
            || persisted_node.frame_idx != node.frame_idx
            || persisted_node.step_idx != node.step_idx
            || persisted_node.site_id != node.site_id
            || persisted_node.protocol_phase != node.protocol_phase
            || persisted_node.aabb_min != node.aabb_min
            || persisted_node.aabb_max != node.aabb_max
            || persisted_node.centroid_xyz != node.centroid_xyz
        {
            bail!(
                "coupled stream {} node {index} differs from retained runtime/committed lineage",
                stream.source_stream_id
            );
        }
        count += 1;
    }
    if count == 0 {
        bail!(
            "coupled stream {} has no committed lattice nodes",
            stream.source_stream_id
        );
    }
    if !receipt.equality_pass
        || !receipt.overflow_conservation_pass
        || !receipt.ordinal_header_manifest_reconstruction_pass
        || receipt.overflow_loss_count != 0
        || [
            receipt.produced_records,
            receipt.persisted_records,
            receipt.decoded_records,
            receipt.lattice_node_lineage_count,
        ]
        .iter()
        .any(|&value| value != count as u64)
    {
        bail!(
            "coupled stream {} committed receipt population differs from its lineage",
            stream.source_stream_id
        );
    }
    Ok((count, expected))
}

/// Project existing components without reclustering or combining coordinate domains.
fn project_stream_sites(
    stream_id: u32,
    nodes: &[GhostPhaseLatticeNode],
    components: &[GhostPhaseLatticeComponent],
    min_nodes: u32,
    max_sites: usize,
) -> Result<(Vec<usize>, Vec<MaterializedSite>)> {
    let mut projected = Vec::with_capacity(components.len());
    let mut regions = Vec::with_capacity(components.len());
    let mut counts = Vec::with_capacity(components.len());
    let mut seen = BTreeSet::new();
    for (component_index, component) in components.iter().enumerate() {
        let mut indices = Vec::new();
        let mut low = [f32::INFINITY; 3];
        let mut high = [f32::NEG_INFINITY; 3];
        for &index in &component.node_indices {
            let node = nodes.get(index as usize).with_context(|| {
                format!("committed component {component_index} references absent node {index}")
            })?;
            if !seen.insert(index) {
                bail!("committed components repeat node {index}");
            }
            if u32::from(node.stream_id) == stream_id {
                indices.push(index);
                for axis in 0..3 {
                    if !node.aabb_min[axis].is_finite()
                        || !node.aabb_max[axis].is_finite()
                        || node.aabb_min[axis] > node.aabb_max[axis]
                    {
                        bail!("coupled stream {stream_id} node {index} has invalid geometry");
                    }
                    low[axis] = low[axis].min(node.aabb_min[axis]);
                    high[axis] = high[axis].max(node.aabb_max[axis]);
                }
            }
        }
        counts.push(u32::try_from(indices.len()).context("coupled component count exceeds u32")?);
        regions.push((component_index, low, high));
        projected.push(GhostPhaseLatticeComponent {
            node_indices: indices,
            n_intra_edges: 0,
            intra_edge_score_sum: 0.0,
            edge_lineage_exact: false,
        });
    }
    if seen.len() != nodes.len() {
        bail!("committed lattice components do not partition all source nodes");
    }
    let mut selected = select_lattice_components(&counts, min_nodes);
    selected.retain(|&index| counts[index] > 0);
    if max_sites > 0 {
        selected.truncate(max_sites);
    }
    if selected.is_empty() {
        bail!("coupled stream {stream_id} has no component satisfying the explicit Therm selection policy");
    }
    let selected_regions: Vec<_> = selected.iter().map(|&index| regions[index]).collect();
    let sites =
        materialize_lattice_sites_with_node_centroids(&selected_regions, nodes, &projected)?;
    Ok((selected, sites))
}

/// Consume the actual coupled stream state and a verified, committed lattice.
/// Each SDST bridge uses one stream's own topology and live protocol.
pub fn analyze_coupled_lattice_streams(
    streams: &[CoupledPostMdStream],
    nodes: &[GhostPhaseLatticeNode],
    outcome: &GhostPhaseLatticeOutcome,
    lineage: &GhostSegmentLatticeLoad,
    min_nodes: u32,
    max_sites: usize,
) -> Result<CoupledLatticeThermReport> {
    let ids: BTreeSet<_> = streams
        .iter()
        .map(|stream| stream.source_stream_id)
        .collect();
    if streams.len() != 2 || ids != BTreeSet::from([0, 1]) {
        bail!("coupled lattice Therm requires exactly retained streams A=0 and B=1");
    }
    let count = nodes.len() as u64;
    if count == 0
        || lineage.nodes.len() != nodes.len()
        || lineage.record_lineage.len() != nodes.len()
        || u64::from(outcome.stats.n_nodes) != count
        || !lineage.equality_pass
        || !lineage.phase_protocols_bound_by_stream
        || !lineage.overflow_conservation_pass
        || lineage.overflow_loss_count != 0
        || [
            lineage.produced_records,
            lineage.persisted_records,
            lineage.decoded_records,
            lineage.lattice_node_lineage_count,
        ]
        .iter()
        .any(|&value| value != count)
        || nodes.iter().any(|node| node.stream_id > 1)
    {
        bail!("coupled lattice Therm requires exact committed node/lineage conservation");
    }
    // Validate both domains before allocating either stream's GPU analysis state.
    let mut prepared = Vec::with_capacity(2);
    for id in [0, 1] {
        let stream = streams
            .iter()
            .find(|stream| stream.source_stream_id == id)
            .unwrap();
        let (node_count, binding) = verify_stream_lineage(stream, nodes, lineage)?;
        let (parents, sites) =
            project_stream_sites(id, nodes, &outcome.components, min_nodes, max_sites)?;
        if stream.spikes.is_empty()
            || stream
                .spikes
                .iter()
                .any(|spike| spike.timestep < 0 || spike.timestep > stream.physical_steps)
        {
            bail!("coupled stream {id} has absent spikes or events outside its measured physical horizon");
        }
        prepared.push((stream, node_count, binding, parents, sites));
    }
    let mut reports = Vec::with_capacity(2);
    for (stream, node_count, binding, parents, sites) in prepared {
        let bridge =
            SdstBridge::new_for_spikes(&stream.topology, &stream.protocol, &stream.spikes)?;
        let ingested_event_count = bridge.ingest_all_spikes(&stream.spikes)?;
        let requests = bridge.lattice_requests(&sites)?;
        let analysis = bridge.analyze_regions(&requests)?;
        if analysis.sdst_event_count != ingested_event_count {
            bail!(
                "coupled stream {} SDST event population changed between ingestion and analysis",
                stream.source_stream_id
            );
        }
        let conservation = reconcile_lattice_site_population(
            sites.len(),
            &sites,
            &analysis,
            LatticeMaterializationRequirement::Required,
            "coupled_stream_lattice_therm",
        )?;
        if !conservation.promotable {
            bail!(
                "coupled stream {} required lattice materialization is incomplete",
                stream.source_stream_id
            );
        }
        reports.push(CoupledStreamThermReport {
            source_stream_id: stream.source_stream_id,
            topology_source_sha256: stream
                .topology_source_sha256
                .clone()
                .context("verified topology SHA256 disappeared")?,
            seed: stream.seed,
            physical_steps: stream.physical_steps,
            protocol: serde_json::to_value(&stream.protocol)?,
            protocol_state_hash: binding.protocol_state_hash,
            phase_schedule_digest_sha256: binding.phase_schedule_digest_sha256,
            condition_id: binding.condition_id,
            snapshot_count: stream.snapshots.len(),
            lattice_node_count: node_count,
            selected_parent_component_indices: parents,
            input_spike_count: stream.spikes.len(),
            ingested_event_count,
            sampling_policy:
                "native_SdstBridge::ingest_all_spikes;bounded_stride_sampling_if_capacity_exceeded",
            sites,
            analysis,
            conservation,
        });
    }
    Ok(CoupledLatticeThermReport {
        schema_kind: "prismos.coupled_stream_lattice_therm.v1",
        schema_version: 1,
        identity_scope: "source_stream_id+topology_source_sha256+analyzed_site_identity",
        component_projection: "committed_parent_component_projected_to_source_stream_nodes;no_cross_topology_geometry_union;no_reclustering",
        min_nodes_per_stream: min_nodes,
        max_sites_per_stream: max_sites,
        committed_lattice_node_count: nodes.len(),
        streams: reports,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdst_bridge::{AnalyzedSiteIdentity, LatticeComponentId};

    fn node(stream_id: u16, low: f32, high: f32, center: f32) -> GhostPhaseLatticeNode {
        let mut node = GhostPhaseLatticeNode::empty();
        node.stream_id = stream_id;
        node.aabb_min = [low, 0.0, 0.0];
        node.aabb_max = [high, 1.0, 1.0];
        node.centroid_xyz = [center, 0.5, 0.5];
        node
    }

    fn component(indices: &[u32]) -> GhostPhaseLatticeComponent {
        GhostPhaseLatticeComponent {
            node_indices: indices.to_vec(),
            n_intra_edges: 0,
            intra_edge_score_sum: 0.0,
            edge_lineage_exact: false,
        }
    }

    #[test]
    fn shared_parent_component_keeps_independent_observed_stream_geometry() {
        let nodes = vec![
            node(0, 0.0, 1.0, 0.25),
            node(1, 100.0, 102.0, 101.0),
            node(0, 2.0, 4.0, 3.0),
        ];
        let components = vec![component(&[0, 1, 2])];
        let (a_parent, a) = project_stream_sites(0, &nodes, &components, 1, 0).unwrap();
        let (b_parent, b) = project_stream_sites(1, &nodes, &components, 1, 0).unwrap();
        assert_eq!(a_parent, vec![0]);
        assert_eq!(b_parent, vec![0]);
        assert_eq!(
            a[0].analyzed_site_identity,
            AnalyzedSiteIdentity::LatticeComponent(LatticeComponentId(0))
        );
        assert_eq!(a[0].analyzed_site_identity, b[0].analyzed_site_identity);
        assert_eq!((a[0].aabb_min[0], a[0].aabb_max[0]), (0.0, 4.0));
        assert_eq!((b[0].aabb_min[0], b[0].aabb_max[0]), (100.0, 102.0));
        assert!((a[0].centroid[0] - 6.25 / 3.0).abs() < 1e-6);
        assert_eq!(b[0].centroid[0], 101.0);
    }

    #[test]
    fn selection_counts_only_own_nodes_and_keeps_parent_identity() {
        let nodes = vec![
            node(0, 0.0, 1.0, 0.5),
            node(1, 10.0, 11.0, 10.5),
            node(1, 12.0, 13.0, 12.5),
            node(0, 2.0, 3.0, 2.5),
            node(0, 4.0, 5.0, 4.5),
        ];
        let components = vec![component(&[0, 1, 2]), component(&[3, 4])];
        let (parents, sites) = project_stream_sites(0, &nodes, &components, 2, 1).unwrap();
        assert_eq!(parents, vec![1]);
        assert_eq!(
            sites[0].analyzed_site_identity,
            AnalyzedSiteIdentity::LatticeComponent(LatticeComponentId(1))
        );
        assert!(project_stream_sites(1, &nodes, &components, 3, 0).is_err());
        assert_eq!(
            project_stream_sites(1, &nodes, &components, 0, 0)
                .unwrap()
                .0,
            vec![0]
        );
    }

    #[test]
    fn missing_or_tampered_topology_binding_is_rejected() {
        let sha = "a".repeat(64);
        assert!(validate_topology_binding(0, None, &sha).is_err());
        assert!(validate_topology_binding(0, Some(&"b".repeat(64)), &sha).is_err());
        assert!(validate_topology_binding(0, Some("invalid"), "invalid").is_err());
        validate_topology_binding(0, Some(&sha), &sha).unwrap();
    }

    #[test]
    fn mutated_retained_protocol_is_rejected_by_committed_identity() {
        let protocol = CryoUvProtocol::standard();
        let committed =
            retained_protocol_binding(1, &protocol, 30000, "coupled_design::apo-holo").unwrap();
        verify_protocol_identity(
            &committed,
            &committed.protocol_state_hash,
            &committed.phase_schedule_digest_sha256,
        )
        .unwrap();
        for mutation in 0..3 {
            let mut changed = protocol.clone();
            match mutation {
                0 => changed.end_temp += 1.0,
                1 => changed.cold_hold_steps += 123,
                _ => changed.scan_wavelengths[0] += 1.0,
            }
            let expected =
                retained_protocol_binding(1, &changed, 30000, "coupled_design::apo-holo").unwrap();
            assert!(verify_protocol_identity(
                &expected,
                &committed.protocol_state_hash,
                &committed.phase_schedule_digest_sha256
            )
            .is_err());
        }
        let mut offset = protocol;
        offset.cold_hold_steps += 123;
        let preserved =
            retained_protocol_binding(1, &offset, 30123, "coupled_design::apo-holo").unwrap();
        assert_eq!(
            preserved.intervals[0].end_step,
            offset.cold_hold_steps as u64
        );
        assert_eq!(
            preserved.checked_phase_for(offset.cold_hold_steps as u64),
            Some(1)
        );
        assert!(verify_protocol_identity(
            &preserved,
            &preserved.protocol_state_hash,
            &"0".repeat(64)
        )
        .is_err());
        assert!(
            retained_protocol_binding(1, &offset, 30123, "coupled_reference::apo-holo").is_err()
        );
    }

    #[test]
    fn retained_no_cooling_protocol_uses_producer_warm_tail_binding() {
        let protocol = CryoUvProtocol {
            cold_hold_steps: 10,
            ramp_steps: 20,
            warm_hold_steps: 30,
            ramp_down_steps: 0,
            cold_return_steps: 0,
            ..CryoUvProtocol::standard()
        };
        let condition = "coupled_reference::apo-holo";
        let retained = retained_protocol_binding(0, &protocol, 1007, condition).unwrap();
        let produced = crate::coupled_md::coupled_protocol_phase_binding(
            &protocol,
            0,
            1007,
            condition.to_string(),
        )
        .unwrap();
        assert_eq!(retained, produced);
        assert_eq!(retained.checked_phase_for(60), Some(2));
        assert_eq!(retained.checked_phase_for(1007), Some(2));
        assert_eq!(retained.checked_phase_for(1008), None);
        assert_eq!(
            retained.intervals[3].start_step,
            retained.intervals[3].end_step
        );
    }

    #[test]
    fn duplicate_missing_and_nonfinite_component_nodes_are_rejected() {
        let nodes = vec![node(0, 0.0, 1.0, 0.5), node(1, 3.0, 4.0, 3.5)];
        assert!(project_stream_sites(0, &nodes, &[component(&[0, 0, 1])], 1, 0).is_err());
        assert!(project_stream_sites(0, &nodes, &[component(&[0])], 1, 0).is_err());
        assert!(project_stream_sites(0, &nodes, &[component(&[0, 2])], 1, 0).is_err());
        let bad = vec![node(0, 0.0, 1.0, f32::NAN)];
        assert!(project_stream_sites(0, &bad, &[component(&[0])], 1, 0).is_err());
    }
}
