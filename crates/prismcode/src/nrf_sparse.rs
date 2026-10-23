//! NRF sparse event compression boundary for NANL-HD.

use crate::binary_runtime::{PcError, PcResult};
use crate::cse_tables::{CseComponentTables, ScientificStateSource};
use crate::nrf_frame::{
    build_nrf_frame, reconstruct_nrf_events, validate_nrf_frame, NrfFrame, NrfMotif,
    NrfReconstructionReport, NrfValidationReport,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const NRF_SPARSE_MAGIC: &str = "NRFSPARSE";
pub const NRF_SPARSE_VERSION: &str = "0.1";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NrfSparseFrame {
    pub magic: String,
    pub version: String,
    pub source_cse_hash: String,
    pub raw_evidence_hash: String,
    pub source_nrf_hash: String,
    pub source_nrf_replay_hash: String,
    pub timebase_ref: u64,
    pub canonical_timebase_hash: String,
    pub event_count: u64,
    pub motif_count: u64,
    pub tile_count: u64,
    pub sparse_index_count: u64,
    pub motif_table_hash: String,
    pub sparse_index_hash: String,
    pub causal_graph_hash: String,
    pub nrf_sparse_hash: String,
    pub reconstruction_hash: String,
    pub replay_hash: String,
    pub events: Vec<NrfSparseEvent>,
    pub motifs: Vec<NrfSparseMotif>,
    pub sparse_index: Vec<NrfSparseIndexEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NrfSparseEvent {
    pub event_id: u64,
    pub event_kind: String,
    pub source_state_ref: u64,
    pub target_state_ref: u64,
    pub time_delta: i64,
    pub motif_ref: u64,
    pub motif_hash: String,
    pub causal_parent_ref: Option<u64>,
    pub event_hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NrfSparseMotif {
    pub motif_id: u64,
    pub motif_kind: String,
    pub dictionary_ref: u64,
    pub occurrence_count: u64,
    pub support_score: f64,
    pub motif_hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NrfSparseIndexEntry {
    pub row: u64,
    pub col: u64,
    pub event_id: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NrfSparseValidationReport {
    pub status: String,
    pub source_cse_hash: String,
    pub raw_evidence_hash: String,
    pub source_nrf_hash: String,
    pub source_nrf_replay_hash: String,
    pub canonical_timebase_hash: String,
    pub nrf_sparse_hash: String,
    pub reconstruction_hash: String,
    pub replay_hash: String,
    pub cse_hash_bound: bool,
    pub raw_evidence_hash_available: bool,
    pub nrf_hash_bound: bool,
    pub canonical_timebase_bound: bool,
    pub event_refs_resolve: bool,
    pub sparse_index_bounds_checked: bool,
    pub motif_hashes_bound: bool,
    pub causal_graph_acyclic: bool,
    pub event_order_canonical: bool,
    pub reconstruct_event_hashes_pass: bool,
    pub replay_hash_pass: bool,
    pub issues: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NrfSparseCompressionReport {
    pub status: String,
    pub source_cse_hash: String,
    pub source_timebase_hash: String,
    pub nrf_frame_hash: String,
    pub sparse_index_hash: String,
    pub event_count: u64,
    pub sparse_index_count: u64,
    pub canonical_timebase_bound: bool,
    pub causal_graph_acyclic: bool,
    pub motif_dictionary_hash_bound: bool,
    pub reconstruct_event_hashes_pass: bool,
    pub replay_pass: bool,
    pub validation: NrfValidationReport,
    pub reconstruction: NrfReconstructionReport,
}

pub fn encode_sparse_nrf(
    source: &ScientificStateSource,
    tables: &CseComponentTables,
) -> PcResult<(NrfFrame, NrfSparseCompressionReport)> {
    let frame = build_nrf_frame(source, tables)?;
    let validation = validate_nrf_frame(&frame, tables)?;
    let reconstruction = reconstruct_nrf_events(&frame, tables)?;
    let sparse = build_nrf_sparse_frame(&frame, tables, &tables.provenance_hash)?;
    let sparse_validation = validate_nrf_sparse_frame(&sparse, &frame, tables)?;
    let status = validation.status == "PASS"
        && reconstruction.status == "PASS"
        && sparse_validation.status == "PASS";
    let report = NrfSparseCompressionReport {
        status: if status { "PASS" } else { "FAIL" }.to_string(),
        source_cse_hash: frame.source_cse_hash.clone(),
        source_timebase_hash: frame.canonical_timebase_hash.clone(),
        nrf_frame_hash: frame.nrf_frame_hash.clone(),
        sparse_index_hash: sparse.sparse_index_hash,
        event_count: frame.events.len() as u64,
        sparse_index_count: frame.sparse_index.len() as u64,
        canonical_timebase_bound: sparse_validation.canonical_timebase_bound,
        causal_graph_acyclic: sparse_validation.causal_graph_acyclic,
        motif_dictionary_hash_bound: sparse_validation.motif_hashes_bound,
        reconstruct_event_hashes_pass: sparse_validation.reconstruct_event_hashes_pass
            && reconstruction.canonical_roundtrip_hash_pass,
        replay_pass: sparse_validation.replay_hash_pass && reconstruction.replay_hash_pass,
        validation,
        reconstruction,
    };
    if report.status != "PASS" {
        return Err(PcError::Format("NRF sparse validation failed".to_string()));
    }
    Ok((frame, report))
}

pub fn build_nrf_sparse_frame(
    frame: &NrfFrame,
    tables: &CseComponentTables,
    raw_evidence_hash: &str,
) -> PcResult<NrfSparseFrame> {
    let motifs = frame
        .motifs
        .iter()
        .map(|motif| {
            Ok(NrfSparseMotif {
                motif_id: motif.motif_id,
                motif_kind: motif.motif_kind.clone(),
                dictionary_ref: motif.dictionary_ref,
                occurrence_count: motif.occurrence_count,
                support_score: motif.support_score,
                motif_hash: compute_sparse_motif_hash(motif)?,
            })
        })
        .collect::<PcResult<Vec<_>>>()?;
    let motif_hash_by_id = motifs
        .iter()
        .map(|motif| (motif.motif_id, motif.motif_hash.clone()))
        .collect::<BTreeMap<_, _>>();
    let events = frame
        .events
        .iter()
        .map(|event| {
            let motif_hash = motif_hash_by_id
                .get(&event.motif_ref)
                .cloned()
                .ok_or_else(|| {
                    PcError::Format(format!(
                        "NRF sparse event {} motif ref {} is missing",
                        event.event_id, event.motif_ref
                    ))
                })?;
            let mut sparse_event = NrfSparseEvent {
                event_id: event.event_id,
                event_kind: event.event_kind.clone(),
                source_state_ref: event.source_state_ref,
                target_state_ref: event.target_state_ref,
                time_delta: event.time_delta,
                motif_ref: event.motif_ref,
                motif_hash,
                causal_parent_ref: event.causal_parent_ref,
                event_hash: String::new(),
            };
            sparse_event.event_hash = compute_sparse_event_hash(&sparse_event)?;
            Ok(sparse_event)
        })
        .collect::<PcResult<Vec<_>>>()?;
    let sparse_index = events
        .iter()
        .enumerate()
        .map(|(idx, event)| NrfSparseIndexEntry {
            row: event.time_delta.max(0) as u64,
            col: idx as u64,
            event_id: event.event_id,
        })
        .collect::<Vec<_>>();
    let mut sparse = NrfSparseFrame {
        magic: NRF_SPARSE_MAGIC.to_string(),
        version: NRF_SPARSE_VERSION.to_string(),
        source_cse_hash: tables.canonical_hash.clone(),
        raw_evidence_hash: raw_evidence_hash.to_string(),
        source_nrf_hash: frame.nrf_frame_hash.clone(),
        source_nrf_replay_hash: frame.replay_hash.clone(),
        timebase_ref: frame.timebase_ref,
        canonical_timebase_hash: frame.canonical_timebase_hash.clone(),
        event_count: events.len() as u64,
        motif_count: motifs.len() as u64,
        tile_count: frame.tiles.len() as u64,
        sparse_index_count: sparse_index.len() as u64,
        motif_table_hash: hash_json(&motifs)?,
        sparse_index_hash: hash_json(&sparse_index)?,
        causal_graph_hash: compute_causal_graph_hash(&events)?,
        nrf_sparse_hash: String::new(),
        reconstruction_hash: compute_sparse_reconstruction_hash(&events)?,
        replay_hash: String::new(),
        events,
        motifs,
        sparse_index,
    };
    sparse.nrf_sparse_hash = compute_nrf_sparse_hash(&sparse)?;
    sparse.replay_hash = compute_nrf_sparse_replay_hash(&sparse)?;
    Ok(sparse)
}

pub fn validate_nrf_sparse_frame(
    sparse: &NrfSparseFrame,
    frame: &NrfFrame,
    tables: &CseComponentTables,
) -> PcResult<NrfSparseValidationReport> {
    let mut issues = Vec::new();
    let cse_hash_bound = sparse.source_cse_hash == tables.canonical_hash
        && frame.source_cse_hash == tables.canonical_hash;
    if !cse_hash_bound {
        issues.push("NRF sparse source CSE hash mismatch".to_string());
    }
    let raw_evidence_hash_available =
        !sparse.raw_evidence_hash.is_empty() && sparse.raw_evidence_hash == tables.provenance_hash;
    if !raw_evidence_hash_available {
        issues.push("NRF sparse raw evidence hash missing or not provenance-bound".to_string());
    }
    let nrf_hash_bound = sparse.source_nrf_hash == frame.nrf_frame_hash
        && sparse.source_nrf_replay_hash == frame.replay_hash;
    if !nrf_hash_bound {
        issues.push("NRF sparse source NRF hash/replay binding mismatch".to_string());
    }
    let canonical_timebase_hash = tables
        .timebase_table
        .iter()
        .find(|row| row.time_id == sparse.timebase_ref)
        .map(|row| row.timebase_hash.as_str());
    let canonical_timebase_bound = sparse.timebase_ref == frame.timebase_ref
        && sparse.canonical_timebase_hash == frame.canonical_timebase_hash
        && canonical_timebase_hash == Some(sparse.canonical_timebase_hash.as_str());
    if !canonical_timebase_bound {
        issues.push("NRF sparse canonical timebase binding failed".to_string());
    }
    let entity_refs = tables
        .entity_table
        .iter()
        .map(|entity| entity.entity_id)
        .collect::<BTreeSet<_>>();
    let event_refs_resolve = sparse.events.iter().all(|event| {
        entity_refs.contains(&event.source_state_ref)
            && entity_refs.contains(&event.target_state_ref)
    });
    if !event_refs_resolve {
        issues
            .push("NRF sparse event source/target refs do not resolve to CSE entities".to_string());
    }
    let sparse_index_bounds_checked = sparse.sparse_index.iter().all(|entry| {
        sparse
            .events
            .get(entry.col as usize)
            .map(|event| {
                event.event_id == entry.event_id && entry.row == event.time_delta.max(0) as u64
            })
            .unwrap_or(false)
    });
    if !sparse_index_bounds_checked {
        issues.push("NRF sparse index out of bounds or event-id mismatch".to_string());
    }
    let motif_hash_by_id = sparse
        .motifs
        .iter()
        .map(|motif| (motif.motif_id, motif.motif_hash.as_str()))
        .collect::<BTreeMap<_, _>>();
    let motif_hashes_bound = sparse.motifs.iter().all(|motif| {
        compute_sparse_motif_hash_from_parts(
            motif.motif_id,
            &motif.motif_kind,
            motif.dictionary_ref,
            motif.occurrence_count,
            motif.support_score,
        )
        .map(|hash| hash == motif.motif_hash)
        .unwrap_or(false)
    }) && sparse.events.iter().all(|event| {
        motif_hash_by_id
            .get(&event.motif_ref)
            .copied()
            .map(|hash| hash == event.motif_hash)
            .unwrap_or(false)
    });
    if !motif_hashes_bound {
        issues.push("NRF sparse motif hash binding failed".to_string());
    }
    let causal_graph_acyclic = causal_graph_is_acyclic(&sparse.events);
    if !causal_graph_acyclic {
        issues.push("NRF sparse causal graph contains a cycle or unresolved parent".to_string());
    }
    let event_order_canonical = event_order_is_canonical(&sparse.events);
    if !event_order_canonical {
        issues.push("NRF sparse event order is not canonical".to_string());
    }
    let reconstruct_event_hashes_pass = sparse.events.iter().all(|event| {
        compute_sparse_event_hash(event)
            .map(|expected| expected == event.event_hash)
            .unwrap_or(false)
    });
    if !reconstruct_event_hashes_pass {
        issues.push("NRF sparse event hash reconstruction failed".to_string());
    }
    if sparse.motif_table_hash != hash_json(&sparse.motifs)? {
        issues.push("NRF sparse motif table hash mismatch".to_string());
    }
    if sparse.sparse_index_hash != hash_json(&sparse.sparse_index)? {
        issues.push("NRF sparse index hash mismatch".to_string());
    }
    if sparse.causal_graph_hash != compute_causal_graph_hash(&sparse.events)? {
        issues.push("NRF sparse causal graph hash mismatch".to_string());
    }
    if sparse.reconstruction_hash != compute_sparse_reconstruction_hash(&sparse.events)? {
        issues.push("NRF sparse reconstruction hash mismatch".to_string());
    }
    if sparse.nrf_sparse_hash != compute_nrf_sparse_hash(sparse)? {
        issues.push("NRF sparse frame hash mismatch".to_string());
    }
    let expected_replay = compute_nrf_sparse_replay_hash(sparse)?;
    let replay_hash_pass = sparse.replay_hash == expected_replay;
    if !replay_hash_pass {
        issues.push("NRF sparse replay hash mismatch".to_string());
    }
    Ok(NrfSparseValidationReport {
        status: if issues.is_empty() { "PASS" } else { "FAIL" }.to_string(),
        source_cse_hash: sparse.source_cse_hash.clone(),
        raw_evidence_hash: sparse.raw_evidence_hash.clone(),
        source_nrf_hash: sparse.source_nrf_hash.clone(),
        source_nrf_replay_hash: sparse.source_nrf_replay_hash.clone(),
        canonical_timebase_hash: sparse.canonical_timebase_hash.clone(),
        nrf_sparse_hash: sparse.nrf_sparse_hash.clone(),
        reconstruction_hash: sparse.reconstruction_hash.clone(),
        replay_hash: sparse.replay_hash.clone(),
        cse_hash_bound,
        raw_evidence_hash_available,
        nrf_hash_bound,
        canonical_timebase_bound,
        event_refs_resolve,
        sparse_index_bounds_checked,
        motif_hashes_bound,
        causal_graph_acyclic,
        event_order_canonical,
        reconstruct_event_hashes_pass,
        replay_hash_pass,
        issues,
    })
}

pub fn fail_if_invalid_nrf_sparse(
    sparse: &NrfSparseFrame,
    frame: &NrfFrame,
    tables: &CseComponentTables,
) -> PcResult<()> {
    let report = validate_nrf_sparse_frame(sparse, frame, tables)?;
    if report.status != "PASS" {
        return Err(PcError::Format(report.issues.join(";")));
    }
    Ok(())
}

pub fn compute_nrf_sparse_hash(sparse: &NrfSparseFrame) -> PcResult<String> {
    #[derive(Serialize)]
    struct SparseHashMaterial<'a> {
        magic: &'a str,
        version: &'a str,
        source_cse_hash: &'a str,
        raw_evidence_hash: &'a str,
        source_nrf_hash: &'a str,
        source_nrf_replay_hash: &'a str,
        timebase_ref: u64,
        canonical_timebase_hash: &'a str,
        event_count: u64,
        motif_count: u64,
        tile_count: u64,
        sparse_index_count: u64,
        motif_table_hash: &'a str,
        sparse_index_hash: &'a str,
        causal_graph_hash: &'a str,
        reconstruction_hash: &'a str,
    }
    hash_json(&SparseHashMaterial {
        magic: &sparse.magic,
        version: &sparse.version,
        source_cse_hash: &sparse.source_cse_hash,
        raw_evidence_hash: &sparse.raw_evidence_hash,
        source_nrf_hash: &sparse.source_nrf_hash,
        source_nrf_replay_hash: &sparse.source_nrf_replay_hash,
        timebase_ref: sparse.timebase_ref,
        canonical_timebase_hash: &sparse.canonical_timebase_hash,
        event_count: sparse.event_count,
        motif_count: sparse.motif_count,
        tile_count: sparse.tile_count,
        sparse_index_count: sparse.sparse_index_count,
        motif_table_hash: &sparse.motif_table_hash,
        sparse_index_hash: &sparse.sparse_index_hash,
        causal_graph_hash: &sparse.causal_graph_hash,
        reconstruction_hash: &sparse.reconstruction_hash,
    })
}

pub fn compute_nrf_sparse_replay_hash(sparse: &NrfSparseFrame) -> PcResult<String> {
    #[derive(Serialize)]
    struct ReplayMaterial<'a> {
        source_cse_hash: &'a str,
        raw_evidence_hash: &'a str,
        source_nrf_hash: &'a str,
        canonical_timebase_hash: &'a str,
        nrf_sparse_hash: &'a str,
        reconstruction_hash: &'a str,
        event_hashes: Vec<&'a str>,
        sparse_index_hash: &'a str,
    }
    hash_json(&ReplayMaterial {
        source_cse_hash: &sparse.source_cse_hash,
        raw_evidence_hash: &sparse.raw_evidence_hash,
        source_nrf_hash: &sparse.source_nrf_hash,
        canonical_timebase_hash: &sparse.canonical_timebase_hash,
        nrf_sparse_hash: &sparse.nrf_sparse_hash,
        reconstruction_hash: &sparse.reconstruction_hash,
        event_hashes: sparse
            .events
            .iter()
            .map(|event| event.event_hash.as_str())
            .collect(),
        sparse_index_hash: &sparse.sparse_index_hash,
    })
}

fn compute_sparse_motif_hash(motif: &NrfMotif) -> PcResult<String> {
    compute_sparse_motif_hash_from_parts(
        motif.motif_id,
        &motif.motif_kind,
        motif.dictionary_ref,
        motif.occurrence_count,
        motif.support_score,
    )
}

fn compute_sparse_motif_hash_from_parts(
    motif_id: u64,
    motif_kind: &str,
    dictionary_ref: u64,
    occurrence_count: u64,
    support_score: f64,
) -> PcResult<String> {
    #[derive(Serialize)]
    struct MotifHashMaterial<'a> {
        motif_id: u64,
        motif_kind: &'a str,
        dictionary_ref: u64,
        occurrence_count: u64,
        support_score: String,
    }
    hash_json(&MotifHashMaterial {
        motif_id,
        motif_kind,
        dictionary_ref,
        occurrence_count,
        support_score: format!("{support_score:.12}"),
    })
}

fn compute_sparse_event_hash(event: &NrfSparseEvent) -> PcResult<String> {
    #[derive(Serialize)]
    struct EventHashMaterial<'a> {
        event_id: u64,
        event_kind: &'a str,
        source_state_ref: u64,
        target_state_ref: u64,
        time_delta: i64,
        motif_ref: u64,
        motif_hash: &'a str,
        causal_parent_ref: Option<u64>,
    }
    hash_json(&EventHashMaterial {
        event_id: event.event_id,
        event_kind: &event.event_kind,
        source_state_ref: event.source_state_ref,
        target_state_ref: event.target_state_ref,
        time_delta: event.time_delta,
        motif_ref: event.motif_ref,
        motif_hash: &event.motif_hash,
        causal_parent_ref: event.causal_parent_ref,
    })
}

fn compute_sparse_reconstruction_hash(events: &[NrfSparseEvent]) -> PcResult<String> {
    hash_json(
        &events
            .iter()
            .map(|event| event.event_hash.as_str())
            .collect::<Vec<_>>(),
    )
}

fn compute_causal_graph_hash(events: &[NrfSparseEvent]) -> PcResult<String> {
    hash_json(
        &events
            .iter()
            .filter_map(|event| {
                event
                    .causal_parent_ref
                    .map(|parent| (parent, event.event_id, event.time_delta))
            })
            .collect::<Vec<_>>(),
    )
}

fn causal_graph_is_acyclic(events: &[NrfSparseEvent]) -> bool {
    let parent_by_event = events
        .iter()
        .map(|event| (event.event_id, event.causal_parent_ref))
        .collect::<BTreeMap<_, _>>();
    for event in events {
        let mut seen = BTreeSet::new();
        let mut current = event.event_id;
        while let Some(Some(parent)) = parent_by_event.get(&current) {
            if !parent_by_event.contains_key(parent) || !seen.insert(*parent) {
                return false;
            }
            current = *parent;
        }
    }
    true
}

fn event_order_is_canonical(events: &[NrfSparseEvent]) -> bool {
    let mut previous: Option<(i64, u64)> = None;
    let index_by_id = events
        .iter()
        .enumerate()
        .map(|(idx, event)| (event.event_id, idx))
        .collect::<BTreeMap<_, _>>();
    for (idx, event) in events.iter().enumerate() {
        let key = (event.time_delta, event.event_id);
        if previous.map(|prev| key < prev).unwrap_or(false) {
            return false;
        }
        if let Some(parent) = event.causal_parent_ref {
            let Some(parent_idx) = index_by_id.get(&parent) else {
                return false;
            };
            if *parent_idx >= idx {
                return false;
            }
        }
        previous = Some(key);
    }
    true
}

fn hash_json<T: Serialize>(value: &T) -> PcResult<String> {
    Ok(blake3::hash(&serde_json::to_vec(value)?)
        .to_hex()
        .to_string())
}
