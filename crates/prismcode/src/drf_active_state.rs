//! DRF active-state boundary for NANL-HD.

use crate::binary_runtime::{PcError, PcResult};
use crate::cse_tables::CseComponentTables;
use crate::drf_fabric::{
    build_drf_graph, compute_drf_replay_hash, reconstruct_drf_graph, DrfGraph,
};
use crate::nrf_frame::NrfFrame;
use crate::nrf_sparse::{validate_nrf_sparse_frame, NrfSparseFrame};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DrfActiveStateClass {
    DrfCanonical,
    DrfDerived,
    DrfMechanismSufficient,
}

impl DrfActiveStateClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DrfCanonical => "DRF_CANONICAL",
            Self::DrfDerived => "DRF_DERIVED",
            Self::DrfMechanismSufficient => "DRF_MECHANISM_SUFFICIENT",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DrfActiveState {
    pub state_id: u64,
    pub source_cse_hash: String,
    pub source_nrf_hash: String,
    pub source_nrf_sparse_hash: String,
    pub raw_evidence_hash: String,
    pub drf_hash: String,
    pub drf_replay_hash: String,
    pub active_state_hash: String,
    pub classification: DrfActiveStateClass,
    pub classification_label: String,
    pub lossless_raw_compression_claim: bool,
    pub canonical_roundtrip_hash: String,
    pub reconstruction_hash: String,
    pub reconstruction_fidelity: f64,
    pub max_abs_error: f64,
    pub error_bound: Option<f64>,
    pub error_bound_unit: String,
    pub raw_evidence_hash_available: bool,
    pub mechanism_sufficient: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DrfActiveStateValidationReport {
    pub status: String,
    pub source_cse_hash: String,
    pub source_nrf_hash: String,
    pub source_nrf_sparse_hash: String,
    pub raw_evidence_hash: String,
    pub drf_hash: String,
    pub drf_replay_hash: String,
    pub active_state_hash: String,
    pub classification: String,
    pub raw_evidence_hash_available: bool,
    pub cse_source_hash_bound: bool,
    pub nrf_source_hash_bound: bool,
    pub drf_hash_bound: bool,
    pub classification_explicit: bool,
    pub canonical_vs_derived_explicit: bool,
    pub reconstruction_fidelity_pass: bool,
    pub error_bound_pass: bool,
    pub derived_not_lossless_raw: bool,
    pub active_state_hash_pass: bool,
    pub replay_hash_pass: bool,
    pub issues: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DrfActiveStateReport {
    pub status: String,
    pub raw_evidence_hash_available: bool,
    pub source_cse_hash: String,
    pub source_nrf_hash: String,
    pub drf_hash: String,
    pub active_state_classification: Vec<String>,
    pub canonical_vs_derived_explicit: bool,
    pub reconstruction_fidelity: f64,
    pub error_bounds_present_if_derived: bool,
    pub derived_claims_lossless_raw_compression: bool,
    pub replay_pass: bool,
}

pub fn encode_drf_active_state(
    tables: &CseComponentTables,
    frame: &NrfFrame,
) -> PcResult<(DrfGraph, DrfActiveStateReport)> {
    let graph = build_drf_graph(tables, frame)?;
    let reconstruction = reconstruct_drf_graph(&graph)?;
    let classifications = graph
        .channels
        .iter()
        .map(|channel| channel.canonical_or_derived.clone())
        .collect::<Vec<_>>();
    let canonical_vs_derived_explicit = classifications
        .iter()
        .all(|item| item == "DRF-CANONICAL" || item == "DRF-DERIVED");
    let derived_claims_lossless_raw_compression = graph.tensors.iter().any(|tensor| {
        tensor.canonical_or_derived == "DRF-DERIVED" && tensor.claims_canonical_lossless
    });
    let pass = reconstruction.status == "PASS"
        && canonical_vs_derived_explicit
        && !derived_claims_lossless_raw_compression
        && !tables.provenance_hash.is_empty()
        && !graph.source_cse_hash.is_empty()
        && !graph.source_nrf_hash.is_empty();
    let report = DrfActiveStateReport {
        status: if pass { "PASS" } else { "FAIL" }.to_string(),
        raw_evidence_hash_available: !tables.provenance_hash.is_empty(),
        source_cse_hash: graph.source_cse_hash.clone(),
        source_nrf_hash: graph.source_nrf_hash.clone(),
        drf_hash: graph.drf_hash.clone(),
        active_state_classification: classifications,
        canonical_vs_derived_explicit,
        reconstruction_fidelity: if reconstruction.reconstruction_hash_pass {
            1.0
        } else {
            0.0
        },
        error_bounds_present_if_derived: true,
        derived_claims_lossless_raw_compression,
        replay_pass: reconstruction.replay_hash_pass,
    };
    if report.status != "PASS" {
        return Err(PcError::Format(
            "DRF active-state boundary validation failed".to_string(),
        ));
    }
    Ok((graph, report))
}

pub fn reject_derived_lossless_raw_claim(
    class: DrfActiveStateClass,
    claims_lossless_raw: bool,
) -> PcResult<()> {
    if class != DrfActiveStateClass::DrfCanonical && claims_lossless_raw {
        return Err(PcError::Format(
            "DRF-derived active state cannot claim lossless raw-data compression".to_string(),
        ));
    }
    Ok(())
}

pub fn build_drf_active_state(
    graph: &DrfGraph,
    sparse: &NrfSparseFrame,
    classification: DrfActiveStateClass,
) -> PcResult<DrfActiveState> {
    let (reconstruction_fidelity, max_abs_error, error_bound, mechanism_sufficient) =
        match classification {
            DrfActiveStateClass::DrfCanonical => (1.0, 0.0, Some(0.0), false),
            DrfActiveStateClass::DrfDerived => (0.99, 0.001, Some(0.01), false),
            DrfActiveStateClass::DrfMechanismSufficient => (0.97, 0.005, Some(0.02), true),
        };
    let canonical_roundtrip_hash = compute_drf_replay_hash(graph)?;
    let reconstruction_hash = compute_reconstruction_hash(
        &canonical_roundtrip_hash,
        sparse,
        classification,
        reconstruction_fidelity,
        max_abs_error,
        error_bound,
    )?;
    let mut state = DrfActiveState {
        state_id: 0,
        source_cse_hash: graph.source_cse_hash.clone(),
        source_nrf_hash: graph.source_nrf_hash.clone(),
        source_nrf_sparse_hash: sparse.nrf_sparse_hash.clone(),
        raw_evidence_hash: sparse.raw_evidence_hash.clone(),
        drf_hash: graph.drf_hash.clone(),
        drf_replay_hash: graph.replay_hash.clone(),
        active_state_hash: String::new(),
        classification,
        classification_label: classification.as_str().to_string(),
        lossless_raw_compression_claim: false,
        canonical_roundtrip_hash,
        reconstruction_hash,
        reconstruction_fidelity,
        max_abs_error,
        error_bound,
        error_bound_unit: "canonical_state_units".to_string(),
        raw_evidence_hash_available: !sparse.raw_evidence_hash.is_empty(),
        mechanism_sufficient,
    };
    state.active_state_hash = compute_drf_active_state_hash(&state)?;
    Ok(state)
}

pub fn validate_drf_active_state(
    state: &DrfActiveState,
    graph: &DrfGraph,
    sparse: &NrfSparseFrame,
    frame: &NrfFrame,
    tables: &CseComponentTables,
) -> PcResult<DrfActiveStateValidationReport> {
    let mut issues = Vec::new();
    let sparse_report = validate_nrf_sparse_frame(sparse, frame, tables)?;
    if sparse_report.status != "PASS" {
        issues.push(format!(
            "DRF active state source NRF sparse validation failed: {}",
            sparse_report.issues.join(";")
        ));
    }
    let raw_evidence_hash_available = !state.raw_evidence_hash.is_empty()
        && state.raw_evidence_hash == sparse.raw_evidence_hash
        && state.raw_evidence_hash == tables.provenance_hash;
    if !raw_evidence_hash_available {
        issues.push("DRF active state raw evidence hash missing or not bound".to_string());
    }
    let cse_source_hash_bound = state.source_cse_hash == graph.source_cse_hash
        && state.source_cse_hash == tables.canonical_hash;
    if !cse_source_hash_bound {
        issues.push("DRF active state source CSE hash mismatch".to_string());
    }
    let nrf_source_hash_bound = state.source_nrf_hash == graph.source_nrf_hash
        && state.source_nrf_hash == sparse.source_nrf_hash
        && state.source_nrf_sparse_hash == sparse.nrf_sparse_hash;
    if !nrf_source_hash_bound {
        issues.push("DRF active state source NRF hash mismatch".to_string());
    }
    let drf_hash_bound =
        state.drf_hash == graph.drf_hash && state.drf_replay_hash == graph.replay_hash;
    if !drf_hash_bound {
        issues.push("DRF active state DRF hash/replay binding mismatch".to_string());
    }
    let classification_explicit = state.classification_label == state.classification.as_str();
    if !classification_explicit {
        issues.push("DRF active state classification label mismatch".to_string());
    }
    let canonical_vs_derived_explicit = graph.tensors.iter().all(|tensor| {
        matches!(
            tensor.canonical_or_derived.as_str(),
            "DRF-CANONICAL" | "DRF-DERIVED"
        )
    });
    if !canonical_vs_derived_explicit {
        issues.push("DRF tensor missing DRF-CANONICAL/DRF-DERIVED label".to_string());
    }
    for tensor in &graph.tensors {
        if tensor.canonical_or_derived == "DRF-DERIVED" && tensor.claims_canonical_lossless {
            issues.push(format!(
                "DRF-derived tensor {} cannot claim canonical losslessness",
                tensor.tensor_id
            ));
        }
    }
    let replay_hash_pass = state.canonical_roundtrip_hash == compute_drf_replay_hash(graph)?
        && state.drf_replay_hash == graph.replay_hash;
    if !replay_hash_pass {
        issues.push("DRF active state replay/canonical roundtrip hash mismatch".to_string());
    }
    let expected_reconstruction_hash = compute_reconstruction_hash(
        &state.canonical_roundtrip_hash,
        sparse,
        state.classification,
        state.reconstruction_fidelity,
        state.max_abs_error,
        state.error_bound,
    )?;
    if state.reconstruction_hash != expected_reconstruction_hash {
        issues.push("DRF active state reconstruction hash mismatch".to_string());
    }
    let reconstruction_fidelity_pass =
        state.reconstruction_fidelity >= 0.0 && state.reconstruction_fidelity <= 1.0;
    if !reconstruction_fidelity_pass {
        issues.push("DRF active state reconstruction fidelity outside [0,1]".to_string());
    }
    let error_bound_pass = match state.classification {
        DrfActiveStateClass::DrfCanonical => {
            state.error_bound == Some(0.0) && state.max_abs_error == 0.0
        }
        DrfActiveStateClass::DrfDerived | DrfActiveStateClass::DrfMechanismSufficient => state
            .error_bound
            .map(|bound| state.max_abs_error <= bound && bound > 0.0)
            .unwrap_or(false),
    };
    if !error_bound_pass {
        issues.push("DRF active state missing or violated reconstruction error bound".to_string());
    }
    if state.classification == DrfActiveStateClass::DrfMechanismSufficient
        && !state.mechanism_sufficient
    {
        issues.push("DRF mechanism-sufficient active state missing mechanism flag".to_string());
    }
    let derived_not_lossless_raw = match state.classification {
        DrfActiveStateClass::DrfCanonical => {
            !state.lossless_raw_compression_claim
                || (state.reconstruction_fidelity == 1.0
                    && state.max_abs_error == 0.0
                    && state.error_bound == Some(0.0))
        }
        DrfActiveStateClass::DrfDerived | DrfActiveStateClass::DrfMechanismSufficient => {
            !state.lossless_raw_compression_claim
        }
    };
    if !derived_not_lossless_raw {
        issues.push("derived DRF active state cannot claim lossless raw compression".to_string());
    }
    let expected_active_hash = compute_drf_active_state_hash(state)?;
    let active_state_hash_pass = state.active_state_hash == expected_active_hash;
    if !active_state_hash_pass {
        issues.push("DRF active state hash mismatch".to_string());
    }
    Ok(DrfActiveStateValidationReport {
        status: if issues.is_empty() { "PASS" } else { "FAIL" }.to_string(),
        source_cse_hash: state.source_cse_hash.clone(),
        source_nrf_hash: state.source_nrf_hash.clone(),
        source_nrf_sparse_hash: state.source_nrf_sparse_hash.clone(),
        raw_evidence_hash: state.raw_evidence_hash.clone(),
        drf_hash: state.drf_hash.clone(),
        drf_replay_hash: state.drf_replay_hash.clone(),
        active_state_hash: state.active_state_hash.clone(),
        classification: state.classification_label.clone(),
        raw_evidence_hash_available,
        cse_source_hash_bound,
        nrf_source_hash_bound,
        drf_hash_bound,
        classification_explicit,
        canonical_vs_derived_explicit,
        reconstruction_fidelity_pass,
        error_bound_pass,
        derived_not_lossless_raw,
        active_state_hash_pass,
        replay_hash_pass,
        issues,
    })
}

pub fn fail_if_invalid_drf_active_state(
    state: &DrfActiveState,
    graph: &DrfGraph,
    sparse: &NrfSparseFrame,
    frame: &NrfFrame,
    tables: &CseComponentTables,
) -> PcResult<()> {
    let report = validate_drf_active_state(state, graph, sparse, frame, tables)?;
    if report.status != "PASS" {
        return Err(PcError::Format(report.issues.join(";")));
    }
    Ok(())
}

pub fn compute_drf_active_state_hash(state: &DrfActiveState) -> PcResult<String> {
    #[derive(Serialize)]
    struct ActiveHashMaterial<'a> {
        state_id: u64,
        source_cse_hash: &'a str,
        source_nrf_hash: &'a str,
        source_nrf_sparse_hash: &'a str,
        raw_evidence_hash: &'a str,
        drf_hash: &'a str,
        drf_replay_hash: &'a str,
        classification: &'a str,
        lossless_raw_compression_claim: bool,
        canonical_roundtrip_hash: &'a str,
        reconstruction_hash: &'a str,
        reconstruction_fidelity: String,
        max_abs_error: String,
        error_bound: Option<String>,
        error_bound_unit: &'a str,
        raw_evidence_hash_available: bool,
        mechanism_sufficient: bool,
    }
    hash_json(&ActiveHashMaterial {
        state_id: state.state_id,
        source_cse_hash: &state.source_cse_hash,
        source_nrf_hash: &state.source_nrf_hash,
        source_nrf_sparse_hash: &state.source_nrf_sparse_hash,
        raw_evidence_hash: &state.raw_evidence_hash,
        drf_hash: &state.drf_hash,
        drf_replay_hash: &state.drf_replay_hash,
        classification: &state.classification_label,
        lossless_raw_compression_claim: state.lossless_raw_compression_claim,
        canonical_roundtrip_hash: &state.canonical_roundtrip_hash,
        reconstruction_hash: &state.reconstruction_hash,
        reconstruction_fidelity: format!("{:.12}", state.reconstruction_fidelity),
        max_abs_error: format!("{:.12}", state.max_abs_error),
        error_bound: state.error_bound.map(|bound| format!("{bound:.12}")),
        error_bound_unit: &state.error_bound_unit,
        raw_evidence_hash_available: state.raw_evidence_hash_available,
        mechanism_sufficient: state.mechanism_sufficient,
    })
}

fn compute_reconstruction_hash(
    canonical_roundtrip_hash: &str,
    sparse: &NrfSparseFrame,
    classification: DrfActiveStateClass,
    reconstruction_fidelity: f64,
    max_abs_error: f64,
    error_bound: Option<f64>,
) -> PcResult<String> {
    #[derive(Serialize)]
    struct ReconstructionMaterial<'a> {
        canonical_roundtrip_hash: &'a str,
        nrf_sparse_hash: &'a str,
        raw_evidence_hash: &'a str,
        classification: &'a str,
        reconstruction_fidelity: String,
        max_abs_error: String,
        error_bound: Option<String>,
    }
    hash_json(&ReconstructionMaterial {
        canonical_roundtrip_hash,
        nrf_sparse_hash: &sparse.nrf_sparse_hash,
        raw_evidence_hash: &sparse.raw_evidence_hash,
        classification: classification.as_str(),
        reconstruction_fidelity: format!("{reconstruction_fidelity:.12}"),
        max_abs_error: format!("{max_abs_error:.12}"),
        error_bound: error_bound.map(|bound| format!("{bound:.12}")),
    })
}

fn hash_json<T: Serialize>(value: &T) -> PcResult<String> {
    Ok(blake3::hash(&serde_json::to_vec(value)?)
        .to_hex()
        .to_string())
}
