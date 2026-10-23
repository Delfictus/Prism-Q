use crate::binary_runtime::{PcError, PcResult};
use crate::cse_tables::CseComponentTables;
use crate::drf_tensors::DrfTensor;
use crate::nrf_frame::NrfFrame;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DrfGraph {
    pub graph_id: u64,
    pub source_cse_hash: String,
    pub source_nrf_hash: String,
    pub source_nrf_replay_hash: String,
    pub node_count: u64,
    pub edge_count: u64,
    pub channel_count: u64,
    pub tensor_count: u64,
    pub timebase_ref: u64,
    pub canonical_timebase_hash: String,
    pub drf_hash: String,
    pub merkle_root: String,
    pub replay_hash: String,
    pub nodes: Vec<DrfNode>,
    pub edges: Vec<DrfEdge>,
    pub channels: Vec<DrfChannel>,
    pub tensors: Vec<DrfTensor>,
    pub resonance_state: DrfResonanceState,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DrfNode {
    pub node_id: u64,
    pub entity_ref: u64,
    pub state_ref: u64,
    pub node_kind: String,
    pub channel_offset: u64,
    pub channel_len: u64,
    pub uncertainty_ref: u64,
    pub provenance_ref: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DrfEdge {
    pub edge_id: u64,
    pub source_node: u64,
    pub target_node: u64,
    pub edge_kind: String,
    pub weight: f32,
    pub time_lag_ref: u64,
    pub event_motif_ref: u64,
    pub thermo_coupling_ref: u64,
    pub uncertainty_ref: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DrfChannel {
    pub channel_id: u64,
    pub channel_kind: String,
    pub dtype: String,
    pub scale: f32,
    pub zero_point: f32,
    pub quantization_policy: String,
    pub tensor_ref: u64,
    pub canonical_or_derived: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DrfResonanceState {
    pub resonance_id: u64,
    pub node_range: [u64; 2],
    pub edge_range: [u64; 2],
    pub event_window_ref: u64,
    pub firing_state_ref: u64,
    pub decay_state_ref: u64,
    pub uncertainty_state_ref: u64,
    pub acquisition_state_ref: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DrfReconstructionReport {
    pub status: String,
    pub graph_id: u64,
    pub source_cse_hash: String,
    pub source_nrf_hash: String,
    pub source_nrf_replay_hash: String,
    pub canonical_timebase_hash: String,
    pub drf_hash: String,
    pub replay_hash: String,
    pub node_count: u64,
    pub edge_count: u64,
    pub tensor_count: u64,
    pub reconstruction_hash_pass: bool,
    pub replay_hash_pass: bool,
}

pub fn build_drf_graph(tables: &CseComponentTables, frame: &NrfFrame) -> PcResult<DrfGraph> {
    let nodes: Vec<DrfNode> = tables
        .entity_table
        .iter()
        .enumerate()
        .map(|(idx, entity)| DrfNode {
            node_id: idx as u64,
            entity_ref: entity.entity_id,
            state_ref: entity.entity_id,
            node_kind: entity.entity_kind.clone(),
            channel_offset: (idx as u64).min(1),
            channel_len: 1,
            uncertainty_ref: 0,
            provenance_ref: 0,
        })
        .collect();
    let node_index_by_entity = tables
        .entity_table
        .iter()
        .enumerate()
        .map(|(idx, entity)| (entity.entity_id, idx as u64))
        .collect::<BTreeMap<_, _>>();
    let mut edges = Vec::with_capacity(frame.events.len());
    for (idx, event) in frame.events.iter().enumerate() {
        let source_node = *node_index_by_entity
            .get(&event.source_state_ref)
            .ok_or_else(|| {
                PcError::Format(format!(
                    "DRF event {} source ref {} is not in CSE entity table",
                    event.event_id, event.source_state_ref
                ))
            })?;
        let target_node = *node_index_by_entity
            .get(&event.target_state_ref)
            .ok_or_else(|| {
                PcError::Format(format!(
                    "DRF event {} target ref {} is not in CSE entity table",
                    event.event_id, event.target_state_ref
                ))
            })?;
        edges.push(DrfEdge {
            edge_id: idx as u64,
            source_node,
            target_node,
            edge_kind: event.event_kind.clone(),
            weight: event.amplitude as f32,
            time_lag_ref: event.time_delta.max(0) as u64,
            event_motif_ref: event.motif_ref,
            thermo_coupling_ref: event.thermo_ref,
            uncertainty_ref: event.uncertainty_ref,
        });
    }
    let tensor_payload: Vec<f32> = edges.iter().map(|edge| edge.weight).collect();
    let tensor = DrfTensor::new_with_state(
        0,
        vec![tensor_payload.len() as u64],
        tensor_payload,
        "RAW_UNCOMPRESSED",
        "DRF-CANONICAL",
        "NONE_CANONICAL",
        true,
    );
    let derived_payload: Vec<f32> = frame
        .events
        .iter()
        .map(|event| (event.uncertainty_ref as f32) / 1024.0)
        .collect();
    let derived_tensor = DrfTensor::new_with_state(
        1,
        vec![derived_payload.len() as u64],
        derived_payload,
        "TENSOR_CHUNKED",
        "DRF-DERIVED",
        "DERIVED_EXPLICIT",
        false,
    );
    let channels = vec![
        DrfChannel {
            channel_id: 0,
            channel_kind: "resonance".to_string(),
            dtype: "F32".to_string(),
            scale: 1.0,
            zero_point: 0.0,
            quantization_policy: "NONE_CANONICAL".to_string(),
            tensor_ref: 0,
            canonical_or_derived: "DRF-CANONICAL".to_string(),
        },
        DrfChannel {
            channel_id: 1,
            channel_kind: "uncertainty".to_string(),
            dtype: "F32".to_string(),
            scale: 1.0,
            zero_point: 0.0,
            quantization_policy: "DERIVED_EXPLICIT".to_string(),
            tensor_ref: 1,
            canonical_or_derived: "DRF-DERIVED".to_string(),
        },
    ];
    let mut graph = DrfGraph {
        graph_id: 0,
        source_cse_hash: tables.canonical_hash.clone(),
        source_nrf_hash: frame.nrf_frame_hash.clone(),
        source_nrf_replay_hash: frame.replay_hash.clone(),
        node_count: nodes.len() as u64,
        edge_count: edges.len() as u64,
        channel_count: channels.len() as u64,
        tensor_count: 2,
        timebase_ref: frame.timebase_ref,
        canonical_timebase_hash: frame.canonical_timebase_hash.clone(),
        drf_hash: String::new(),
        merkle_root: String::new(),
        replay_hash: String::new(),
        nodes,
        edges,
        channels,
        tensors: vec![tensor, derived_tensor],
        resonance_state: DrfResonanceState {
            resonance_id: 0,
            node_range: [0, tables.entity_table.len() as u64],
            edge_range: [0, frame.events.len() as u64],
            event_window_ref: frame.timebase_ref,
            firing_state_ref: 0,
            decay_state_ref: 0,
            uncertainty_state_ref: 0,
            acquisition_state_ref: 0,
        },
    };
    graph.merkle_root = compute_drf_merkle_root(&graph)?;
    graph.drf_hash = compute_drf_graph_hash(&graph)?;
    graph.replay_hash = compute_drf_replay_hash(&graph)?;
    Ok(graph)
}

pub fn reconstruct_drf_graph(graph: &DrfGraph) -> PcResult<DrfReconstructionReport> {
    let expected_merkle = compute_drf_merkle_root(graph)?;
    let expected_hash = compute_drf_graph_hash(graph)?;
    let expected_replay = compute_drf_replay_hash(graph)?;
    let pass = graph.merkle_root == expected_merkle
        && graph.drf_hash == expected_hash
        && graph.replay_hash == expected_replay;
    Ok(DrfReconstructionReport {
        status: if pass { "PASS" } else { "FAIL" }.to_string(),
        graph_id: graph.graph_id,
        source_cse_hash: graph.source_cse_hash.clone(),
        source_nrf_hash: graph.source_nrf_hash.clone(),
        source_nrf_replay_hash: graph.source_nrf_replay_hash.clone(),
        canonical_timebase_hash: graph.canonical_timebase_hash.clone(),
        drf_hash: graph.drf_hash.clone(),
        replay_hash: graph.replay_hash.clone(),
        node_count: graph.nodes.len() as u64,
        edge_count: graph.edges.len() as u64,
        tensor_count: graph.tensors.len() as u64,
        reconstruction_hash_pass: graph.merkle_root == expected_merkle
            && graph.drf_hash == expected_hash,
        replay_hash_pass: graph.replay_hash == expected_replay,
    })
}

pub fn compute_drf_merkle_root(graph: &DrfGraph) -> PcResult<String> {
    let leaves = graph
        .nodes
        .iter()
        .map(|node| serde_json::to_vec(node).map(hash_hex))
        .chain(
            graph
                .edges
                .iter()
                .map(|edge| serde_json::to_vec(edge).map(hash_hex)),
        )
        .chain(
            graph
                .channels
                .iter()
                .map(|channel| serde_json::to_vec(channel).map(hash_hex)),
        )
        .chain(
            graph
                .tensors
                .iter()
                .map(|tensor| serde_json::to_vec(tensor).map(hash_hex)),
        )
        .collect::<Result<Vec<_>, _>>()?;
    if leaves.is_empty() {
        return Err(crate::binary_runtime::PcError::Hash(
            "DRF graph has no Merkle leaves".to_string(),
        ));
    }
    Ok(hash_hex(serde_json::to_vec(&leaves)?))
}

pub fn compute_drf_graph_hash(graph: &DrfGraph) -> PcResult<String> {
    #[derive(Serialize)]
    struct GraphHashMaterial<'a> {
        graph_id: u64,
        source_cse_hash: &'a str,
        source_nrf_hash: &'a str,
        source_nrf_replay_hash: &'a str,
        node_count: u64,
        edge_count: u64,
        channel_count: u64,
        tensor_count: u64,
        timebase_ref: u64,
        canonical_timebase_hash: &'a str,
        merkle_root: &'a str,
    }
    Ok(hash_hex(serde_json::to_vec(&GraphHashMaterial {
        graph_id: graph.graph_id,
        source_cse_hash: &graph.source_cse_hash,
        source_nrf_hash: &graph.source_nrf_hash,
        source_nrf_replay_hash: &graph.source_nrf_replay_hash,
        node_count: graph.node_count,
        edge_count: graph.edge_count,
        channel_count: graph.channel_count,
        tensor_count: graph.tensor_count,
        timebase_ref: graph.timebase_ref,
        canonical_timebase_hash: &graph.canonical_timebase_hash,
        merkle_root: &graph.merkle_root,
    })?))
}

pub fn compute_drf_replay_hash(graph: &DrfGraph) -> PcResult<String> {
    #[derive(Serialize)]
    struct ReplayMaterial<'a> {
        source_cse_hash: &'a str,
        source_nrf_hash: &'a str,
        source_nrf_replay_hash: &'a str,
        canonical_timebase_hash: &'a str,
        drf_hash: &'a str,
        tensor_hashes: Vec<&'a str>,
    }
    Ok(hash_hex(serde_json::to_vec(&ReplayMaterial {
        source_cse_hash: &graph.source_cse_hash,
        source_nrf_hash: &graph.source_nrf_hash,
        source_nrf_replay_hash: &graph.source_nrf_replay_hash,
        canonical_timebase_hash: &graph.canonical_timebase_hash,
        drf_hash: &graph.drf_hash,
        tensor_hashes: graph
            .tensors
            .iter()
            .map(|tensor| tensor.tensor_hash.as_str())
            .collect(),
    })?))
}

fn hash_hex(bytes: Vec<u8>) -> String {
    blake3::hash(&bytes).to_hex().to_string()
}
