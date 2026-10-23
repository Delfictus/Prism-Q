use crate::binary_runtime::{PcError, PcResult};
use crate::cse_tables::CseComponentTables;
use crate::drf_fabric::{
    compute_drf_graph_hash, compute_drf_merkle_root, compute_drf_replay_hash, DrfGraph,
};
use crate::nrf_frame::NrfFrame;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DrfValidationReport {
    pub status: String,
    pub source_cse_hash: String,
    pub source_nrf_hash: String,
    pub source_nrf_replay_hash: String,
    pub canonical_timebase_hash: String,
    pub drf_hash: String,
    pub merkle_root: String,
    pub replay_hash: String,
    pub node_refs_resolve: bool,
    pub edge_refs_resolve: bool,
    pub tensor_shapes_valid: bool,
    pub canonical_timebase_bound: bool,
    pub canonical_vs_derived_explicit: bool,
    pub source_hash_binding_pass: bool,
    pub nrf_replay_binding_pass: bool,
    pub reconstruction_test_pass: bool,
    pub replay_hash_pass: bool,
    pub issues: Vec<String>,
}

pub fn validate_drf_graph(
    graph: &DrfGraph,
    tables: &CseComponentTables,
    frame: &NrfFrame,
) -> PcResult<DrfValidationReport> {
    let mut issues = Vec::new();
    if graph.source_cse_hash != tables.canonical_hash {
        issues.push("DRF source CSE hash mismatch".to_string());
    }
    if graph.source_nrf_hash != frame.nrf_frame_hash {
        issues.push("DRF source NRF hash mismatch".to_string());
    }
    if graph.source_nrf_replay_hash != frame.replay_hash {
        issues.push("DRF source NRF replay hash mismatch".to_string());
    }
    if graph.canonical_timebase_hash != frame.canonical_timebase_hash {
        issues.push("DRF canonical timebase hash mismatch".to_string());
    }
    if graph.timebase_ref != frame.timebase_ref {
        issues.push("DRF timebase ref mismatch".to_string());
    }
    let node_refs_resolve = graph
        .nodes
        .iter()
        .all(|node| node.channel_offset + node.channel_len <= graph.channel_count);
    if !node_refs_resolve {
        issues.push("DRF node channel refs out of bounds".to_string());
    }
    let edge_refs_resolve = graph
        .edges
        .iter()
        .all(|edge| edge.source_node < graph.node_count && edge.target_node < graph.node_count);
    if !edge_refs_resolve {
        issues.push("DRF edge node refs out of bounds".to_string());
    }
    let tensor_shapes_valid = graph.tensors.iter().all(|tensor| {
        let expected: u64 = tensor.shape.iter().product();
        expected == tensor.payload.len() as u64
            && tensor.tensor_hash == tensor.recompute_tensor_hash()
            && !tensor.reconstruction_hash.is_empty()
    });
    if !tensor_shapes_valid {
        issues.push("DRF tensor shape/hash invalid".to_string());
    }
    let canonical_vs_derived_explicit = graph.channels.iter().all(|channel| {
        channel.canonical_or_derived == "DRF-CANONICAL"
            || channel.canonical_or_derived == "DRF-DERIVED"
    });
    if !canonical_vs_derived_explicit {
        issues.push("DRF channel missing canonical/derived label".to_string());
    }
    for channel in &graph.channels {
        if channel.tensor_ref >= graph.tensors.len() as u64 {
            issues.push(format!(
                "DRF channel {} tensor ref out of bounds",
                channel.channel_id
            ));
            continue;
        }
        let tensor = &graph.tensors[channel.tensor_ref as usize];
        if tensor.canonical_or_derived != channel.canonical_or_derived {
            issues.push(format!(
                "DRF channel {} canonical/derived label does not match tensor {}",
                channel.channel_id, tensor.tensor_id
            ));
        }
    }
    for tensor in &graph.tensors {
        match tensor.canonical_or_derived.as_str() {
            "DRF-CANONICAL" => {
                if !tensor.claims_canonical_lossless {
                    issues.push(format!(
                        "DRF canonical tensor {} does not claim lossless canonical state",
                        tensor.tensor_id
                    ));
                }
            }
            "DRF-DERIVED" => {
                if tensor.claims_canonical_lossless {
                    issues.push(format!(
                        "DRF-derived tensor {} cannot claim canonical losslessness",
                        tensor.tensor_id
                    ));
                }
                if !tensor.quantization_policy.contains("DERIVED") {
                    issues.push(format!(
                        "DRF-derived tensor {} missing derived quantization policy",
                        tensor.tensor_id
                    ));
                }
            }
            other => issues.push(format!(
                "DRF tensor {} has invalid canonical_or_derived label {other}",
                tensor.tensor_id
            )),
        }
    }
    let reconstruction_test_pass = graph.edge_count == frame.event_count;
    if !reconstruction_test_pass {
        issues.push("DRF reconstruction event-count mismatch".to_string());
    }
    let node_range_end =
        graph.resonance_state.node_range[0].saturating_add(graph.resonance_state.node_range[1]);
    if node_range_end > graph.node_count {
        issues.push("DRF resonance node range out of bounds".to_string());
    }
    let edge_range_end =
        graph.resonance_state.edge_range[0].saturating_add(graph.resonance_state.edge_range[1]);
    if edge_range_end > graph.edge_count {
        issues.push("DRF resonance edge range out of bounds".to_string());
    }
    let expected_merkle = compute_drf_merkle_root(graph)?;
    if graph.merkle_root != expected_merkle {
        issues.push("DRF Merkle root reconstruction failed".to_string());
    }
    let expected_graph_hash = compute_drf_graph_hash(graph)?;
    if graph.drf_hash != expected_graph_hash {
        issues.push("DRF graph hash reconstruction failed".to_string());
    }
    let expected_replay_hash = compute_drf_replay_hash(graph)?;
    if graph.replay_hash != expected_replay_hash {
        issues.push("DRF replay hash reconstruction failed".to_string());
    }
    Ok(DrfValidationReport {
        status: if issues.is_empty() { "PASS" } else { "FAIL" }.to_string(),
        source_cse_hash: graph.source_cse_hash.clone(),
        source_nrf_hash: graph.source_nrf_hash.clone(),
        source_nrf_replay_hash: graph.source_nrf_replay_hash.clone(),
        canonical_timebase_hash: graph.canonical_timebase_hash.clone(),
        drf_hash: graph.drf_hash.clone(),
        merkle_root: graph.merkle_root.clone(),
        replay_hash: graph.replay_hash.clone(),
        node_refs_resolve,
        edge_refs_resolve,
        tensor_shapes_valid,
        canonical_timebase_bound: graph.timebase_ref != 0
            && graph.timebase_ref == frame.timebase_ref
            && graph.canonical_timebase_hash == frame.canonical_timebase_hash,
        canonical_vs_derived_explicit,
        source_hash_binding_pass: graph.source_cse_hash == tables.canonical_hash
            && graph.source_nrf_hash == frame.nrf_frame_hash,
        nrf_replay_binding_pass: graph.source_nrf_replay_hash == frame.replay_hash,
        reconstruction_test_pass,
        replay_hash_pass: graph.replay_hash == expected_replay_hash,
        issues,
    })
}

pub fn fail_if_invalid(
    graph: &DrfGraph,
    tables: &CseComponentTables,
    frame: &NrfFrame,
) -> PcResult<()> {
    let report = validate_drf_graph(graph, tables, frame)?;
    if report.status != "PASS" {
        return Err(PcError::Format(report.issues.join(";")));
    }
    Ok(())
}
