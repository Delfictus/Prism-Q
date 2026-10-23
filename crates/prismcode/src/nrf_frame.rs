use crate::binary_runtime::{PcError, PcResult};
use crate::cse_tables::{CseComponentTables, ScientificStateSource};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NrfFrame {
    pub magic: String,
    pub version: String,
    pub timebase_ref: u64,
    pub frame_id: u64,
    pub frame_index: u64,
    pub event_count: u64,
    pub tile_count: u64,
    pub motif_count: u64,
    pub source_cse_hash: String,
    pub canonical_timebase_hash: String,
    pub nrf_frame_hash: String,
    pub previous_frame_hash: String,
    pub merkle_root: String,
    pub replay_hash: String,
    pub events: Vec<NrfEvent>,
    pub tiles: Vec<NrfTile>,
    pub motifs: Vec<NrfMotif>,
    pub sparse_index: Vec<[u64; 2]>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NrfEvent {
    pub event_id: u64,
    pub event_kind: String,
    pub source_state_ref: u64,
    pub target_state_ref: u64,
    pub time_delta: i64,
    pub motif_ref: u64,
    pub amplitude: f64,
    pub uncertainty_ref: u64,
    pub thermo_ref: u64,
    pub causal_parent_ref: Option<u64>,
    pub event_hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NrfTile {
    pub tile_id: u64,
    pub time_window_ref: u64,
    pub entity_range_ref: [u64; 2],
    pub event_range_ref: [u64; 2],
    pub motif_histogram_ref: u64,
    pub sparse_index_ref: u64,
    pub tile_hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NrfMotif {
    pub motif_id: u64,
    pub motif_kind: String,
    pub symbol_span: [u64; 2],
    pub dictionary_ref: u64,
    pub occurrence_count: u64,
    pub support_score: f64,
    pub motif_hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NrfValidationReport {
    pub status: String,
    pub source_cse_hash: String,
    pub canonical_timebase_hash: String,
    pub nrf_frame_hash: String,
    pub merkle_root: String,
    pub replay_hash: String,
    pub event_count: u64,
    pub sparse_index_bounds_checked: bool,
    pub canonical_timebase_bound: bool,
    pub causal_parent_cycles: bool,
    pub reconstruct_event_hashes_pass: bool,
    pub issues: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NrfReconstructionReport {
    pub status: String,
    pub source_cse_hash: String,
    pub canonical_timebase_hash: String,
    pub frame_hash: String,
    pub replay_hash: String,
    pub reconstructed_event_hashes: Vec<String>,
    pub canonical_roundtrip_hash_pass: bool,
    pub replay_hash_pass: bool,
}

pub fn build_nrf_frame(
    source: &ScientificStateSource,
    tables: &CseComponentTables,
) -> PcResult<NrfFrame> {
    let mut motifs = Vec::new();
    let mut motif_names = BTreeSet::new();
    for event in &source.events {
        motif_names.insert(event.motif.clone());
    }
    for (idx, motif) in motif_names.into_iter().enumerate() {
        motifs.push(NrfMotif {
            motif_id: idx as u64,
            motif_kind: motif.clone(),
            symbol_span: [idx as u64, 1],
            dictionary_ref: idx as u64,
            occurrence_count: source
                .events
                .iter()
                .filter(|event| event.motif == motif)
                .count() as u64,
            support_score: 1.0,
            motif_hash: blake3::hash(motif.as_bytes()).to_hex().to_string(),
        });
    }

    let events: Vec<NrfEvent> = source
        .events
        .iter()
        .map(|event| {
            let motif_ref = motifs
                .iter()
                .position(|motif| motif.motif_kind == event.motif)
                .unwrap_or(0) as u64;
            let event_hash = blake3::hash(
                nrf_event_hash_material(
                    event.event_id,
                    &event.event_kind,
                    event.source_ref,
                    event.target_ref,
                    event.time_delta,
                    motif_ref,
                    event.amplitude,
                    event.causal_parent_ref,
                )
                .as_bytes(),
            )
            .to_hex()
            .to_string();
            NrfEvent {
                event_id: event.event_id,
                event_kind: event.event_kind.clone(),
                source_state_ref: event.source_ref,
                target_state_ref: event.target_ref,
                time_delta: event.time_delta,
                motif_ref,
                amplitude: event.amplitude,
                uncertainty_ref: source
                    .uncertainties
                    .first()
                    .map(|item| item.uncertainty_id)
                    .unwrap_or(0),
                thermo_ref: source
                    .thermo
                    .first()
                    .map(|item| item.thermo_id)
                    .unwrap_or(0),
                causal_parent_ref: event.causal_parent_ref,
                event_hash,
            }
        })
        .collect();
    let sparse_index: Vec<[u64; 2]> = events
        .iter()
        .enumerate()
        .map(|(idx, event)| [event.time_delta.max(0) as u64, idx as u64])
        .collect();
    let tile_hash =
        blake3::hash(format!("{}:{}", source.timebase.timebase_hash, events.len()).as_bytes())
            .to_hex()
            .to_string();
    let tiles = vec![NrfTile {
        tile_id: 0,
        time_window_ref: source.timebase.time_id,
        entity_range_ref: [0, 2],
        event_range_ref: [0, events.len() as u64],
        motif_histogram_ref: 0,
        sparse_index_ref: 0,
        tile_hash,
    }];
    let mut frame = NrfFrame {
        magic: "NRFFRAME".to_string(),
        version: "0.1".to_string(),
        timebase_ref: source.timebase.time_id,
        frame_id: 0,
        frame_index: 0,
        event_count: events.len() as u64,
        tile_count: tiles.len() as u64,
        motif_count: motifs.len() as u64,
        source_cse_hash: tables.canonical_hash.clone(),
        canonical_timebase_hash: source.timebase.timebase_hash.clone(),
        nrf_frame_hash: String::new(),
        previous_frame_hash: "0".repeat(64),
        merkle_root: String::new(),
        replay_hash: String::new(),
        events,
        tiles,
        motifs,
        sparse_index,
    };
    frame.merkle_root = compute_nrf_merkle_root(&frame)?;
    frame.nrf_frame_hash = compute_nrf_frame_hash(&frame)?;
    frame.replay_hash = compute_nrf_replay_hash(&frame)?;
    Ok(frame)
}

pub fn validate_nrf_frame(
    frame: &NrfFrame,
    tables: &CseComponentTables,
) -> PcResult<NrfValidationReport> {
    let mut issues = Vec::new();
    if frame.source_cse_hash != tables.canonical_hash {
        issues.push("NRF frame source CSE hash mismatch".to_string());
    }
    let canonical_timebase_hash = tables
        .timebase_table
        .iter()
        .find(|row| row.time_id == frame.timebase_ref)
        .map(|row| row.timebase_hash.clone());
    if frame.timebase_ref == 0 || canonical_timebase_hash.is_none() {
        issues.push("NRF frame missing canonical timebase ref".to_string());
    } else if canonical_timebase_hash.as_deref() != Some(frame.canonical_timebase_hash.as_str()) {
        issues.push("NRF frame canonical timebase hash mismatch".to_string());
    }
    for item in &frame.sparse_index {
        if item[1] >= frame.events.len() as u64 {
            issues.push("NRF sparse index out of bounds".to_string());
        }
    }
    let mut seen = BTreeSet::new();
    for event in &frame.events {
        if let Some(parent) = event.causal_parent_ref {
            if parent == event.event_id || !seen.contains(&parent) {
                issues.push(format!("invalid NRF causal parent {}", parent));
            }
        }
        seen.insert(event.event_id);
    }
    let expected_hashes: Vec<String> = frame
        .events
        .iter()
        .map(|event| {
            blake3::hash(
                format!(
                    "{}",
                    nrf_event_hash_material(
                        event.event_id,
                        &event.event_kind,
                        event.source_state_ref,
                        event.target_state_ref,
                        event.time_delta,
                        event.motif_ref,
                        event.amplitude,
                        event.causal_parent_ref,
                    )
                )
                .as_bytes(),
            )
            .to_hex()
            .to_string()
        })
        .collect();
    let reconstruct_event_hashes_pass = frame
        .events
        .iter()
        .zip(expected_hashes.iter())
        .all(|(event, hash)| &event.event_hash == hash);
    if !reconstruct_event_hashes_pass {
        issues.push("NRF event hash reconstruction failed".to_string());
    }
    let expected_merkle = compute_nrf_merkle_root(frame)?;
    if frame.merkle_root != expected_merkle {
        issues.push("NRF Merkle root reconstruction failed".to_string());
    }
    let expected_frame_hash = compute_nrf_frame_hash(frame)?;
    if frame.nrf_frame_hash != expected_frame_hash {
        issues.push("NRF frame hash reconstruction failed".to_string());
    }
    let expected_replay_hash = compute_nrf_replay_hash(frame)?;
    if frame.replay_hash != expected_replay_hash {
        issues.push("NRF replay hash reconstruction failed".to_string());
    }
    Ok(NrfValidationReport {
        status: if issues.is_empty() { "PASS" } else { "FAIL" }.to_string(),
        source_cse_hash: frame.source_cse_hash.clone(),
        canonical_timebase_hash: frame.canonical_timebase_hash.clone(),
        nrf_frame_hash: frame.nrf_frame_hash.clone(),
        merkle_root: frame.merkle_root.clone(),
        replay_hash: frame.replay_hash.clone(),
        event_count: frame.events.len() as u64,
        sparse_index_bounds_checked: true,
        canonical_timebase_bound: frame.timebase_ref != 0
            && canonical_timebase_hash.as_deref() == Some(frame.canonical_timebase_hash.as_str()),
        causal_parent_cycles: false,
        reconstruct_event_hashes_pass,
        issues,
    })
}

pub fn reconstruct_nrf_events(
    frame: &NrfFrame,
    tables: &CseComponentTables,
) -> PcResult<NrfReconstructionReport> {
    let report = validate_nrf_frame(frame, tables)?;
    let reconstructed_event_hashes = frame
        .events
        .iter()
        .map(|event| {
            blake3::hash(
                nrf_event_hash_material(
                    event.event_id,
                    &event.event_kind,
                    event.source_state_ref,
                    event.target_state_ref,
                    event.time_delta,
                    event.motif_ref,
                    event.amplitude,
                    event.causal_parent_ref,
                )
                .as_bytes(),
            )
            .to_hex()
            .to_string()
        })
        .collect::<Vec<_>>();
    Ok(NrfReconstructionReport {
        status: report.status.clone(),
        source_cse_hash: frame.source_cse_hash.clone(),
        canonical_timebase_hash: frame.canonical_timebase_hash.clone(),
        frame_hash: frame.nrf_frame_hash.clone(),
        replay_hash: frame.replay_hash.clone(),
        canonical_roundtrip_hash_pass: report.reconstruct_event_hashes_pass,
        replay_hash_pass: report.status == "PASS",
        reconstructed_event_hashes,
    })
}

pub fn fail_if_invalid(frame: &NrfFrame, tables: &CseComponentTables) -> PcResult<()> {
    let report = validate_nrf_frame(frame, tables)?;
    if report.status != "PASS" {
        return Err(PcError::Format(report.issues.join(";")));
    }
    Ok(())
}

pub fn compute_nrf_merkle_root(frame: &NrfFrame) -> PcResult<String> {
    let mut leaves = Vec::new();
    leaves.extend(frame.events.iter().map(|event| event.event_hash.as_str()));
    leaves.extend(frame.tiles.iter().map(|tile| tile.tile_hash.as_str()));
    leaves.extend(frame.motifs.iter().map(|motif| motif.motif_hash.as_str()));
    if leaves.is_empty() {
        return Err(PcError::Hash("NRF frame has no Merkle leaves".to_string()));
    }
    Ok(blake3::hash(&serde_json::to_vec(&leaves)?)
        .to_hex()
        .to_string())
}

pub fn compute_nrf_frame_hash(frame: &NrfFrame) -> PcResult<String> {
    #[derive(Serialize)]
    struct FrameHashMaterial<'a> {
        magic: &'a str,
        version: &'a str,
        timebase_ref: u64,
        canonical_timebase_hash: &'a str,
        frame_id: u64,
        frame_index: u64,
        event_count: u64,
        tile_count: u64,
        motif_count: u64,
        source_cse_hash: &'a str,
        previous_frame_hash: &'a str,
        merkle_root: &'a str,
    }
    Ok(blake3::hash(&serde_json::to_vec(&FrameHashMaterial {
        magic: &frame.magic,
        version: &frame.version,
        timebase_ref: frame.timebase_ref,
        canonical_timebase_hash: &frame.canonical_timebase_hash,
        frame_id: frame.frame_id,
        frame_index: frame.frame_index,
        event_count: frame.event_count,
        tile_count: frame.tile_count,
        motif_count: frame.motif_count,
        source_cse_hash: &frame.source_cse_hash,
        previous_frame_hash: &frame.previous_frame_hash,
        merkle_root: &frame.merkle_root,
    })?)
    .to_hex()
    .to_string())
}

pub fn compute_nrf_replay_hash(frame: &NrfFrame) -> PcResult<String> {
    #[derive(Serialize)]
    struct ReplayMaterial<'a> {
        source_cse_hash: &'a str,
        canonical_timebase_hash: &'a str,
        nrf_frame_hash: &'a str,
        event_hashes: Vec<&'a str>,
        sparse_index: &'a [[u64; 2]],
    }
    Ok(blake3::hash(&serde_json::to_vec(&ReplayMaterial {
        source_cse_hash: &frame.source_cse_hash,
        canonical_timebase_hash: &frame.canonical_timebase_hash,
        nrf_frame_hash: &frame.nrf_frame_hash,
        event_hashes: frame
            .events
            .iter()
            .map(|event| event.event_hash.as_str())
            .collect(),
        sparse_index: &frame.sparse_index,
    })?)
    .to_hex()
    .to_string())
}

fn nrf_event_hash_material(
    event_id: u64,
    event_kind: &str,
    source_state_ref: u64,
    target_state_ref: u64,
    time_delta: i64,
    motif_ref: u64,
    amplitude: f64,
    causal_parent_ref: Option<u64>,
) -> String {
    format!(
        "{event_id}:{event_kind}:{source_state_ref}:{target_state_ref}:{time_delta}:{motif_ref}:{amplitude:.12}:{causal_parent_ref:?}"
    )
}
