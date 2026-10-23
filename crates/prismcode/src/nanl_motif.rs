use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub type NanlMotifResult<T> = Result<T, NanlMotifError>;

pub const NANL_MOTIF_MAGIC: &str = "NANLMOTF";
pub const NANL_MOTIF_VERSION: &str = "0.1";

#[derive(Debug, thiserror::Error)]
pub enum NanlMotifError {
    #[error("PC_ERR_MOTIF_FORMAT:{0}")]
    Format(String),
    #[error("PC_ERR_MOTIF_HASH:{0}")]
    Hash(String),
    #[error("PC_ERR_MOTIF_ORDERING:{0}")]
    Ordering(String),
    #[error("PC_ERR_MOTIF_SUPPORT:{0}")]
    Support(String),
    #[error("PC_ERR_UNKNOWN_MOTIF:{0}")]
    UnknownMotif(String),
    #[error("PC_ERR_INTERNAL:{0}")]
    Internal(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum NanlMotifKind {
    WaterGateMotif,
    ResidueCouplingMotif,
    ThermoPulseMotif,
    VariantTransitionMotif,
    DrfResonanceMotif,
    GFlowNetActionMotif,
    TeacherDisagreementMotif,
    UncertaintyShiftMotif,
}

impl NanlMotifKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::WaterGateMotif => "WATER_GATE_MOTIF",
            Self::ResidueCouplingMotif => "RESIDUE_COUPLING_MOTIF",
            Self::ThermoPulseMotif => "THERMO_PULSE_MOTIF",
            Self::VariantTransitionMotif => "VARIANT_TRANSITION_MOTIF",
            Self::DrfResonanceMotif => "DRF_RESONANCE_MOTIF",
            Self::GFlowNetActionMotif => "GFLOWNET_ACTION_MOTIF",
            Self::TeacherDisagreementMotif => "TEACHER_DISAGREEMENT_MOTIF",
            Self::UncertaintyShiftMotif => "UNCERTAINTY_SHIFT_MOTIF",
        }
    }
}

pub const REQUIRED_MOTIF_KINDS: [NanlMotifKind; 8] = [
    NanlMotifKind::WaterGateMotif,
    NanlMotifKind::ResidueCouplingMotif,
    NanlMotifKind::ThermoPulseMotif,
    NanlMotifKind::VariantTransitionMotif,
    NanlMotifKind::DrfResonanceMotif,
    NanlMotifKind::GFlowNetActionMotif,
    NanlMotifKind::TeacherDisagreementMotif,
    NanlMotifKind::UncertaintyShiftMotif,
];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NanlMotifObservation {
    pub observation_id: u64,
    pub motif_kind: NanlMotifKind,
    pub symbols: Vec<String>,
    pub parameter_refs: Vec<u64>,
    pub time_ref: u64,
    pub source_ref: u64,
    pub target_ref: u64,
}

impl NanlMotifObservation {
    pub fn new(
        observation_id: u64,
        motif_kind: NanlMotifKind,
        symbols: Vec<String>,
        parameter_refs: Vec<u64>,
        time_ref: u64,
        source_ref: u64,
        target_ref: u64,
    ) -> Self {
        Self {
            observation_id,
            motif_kind,
            symbols,
            parameter_refs,
            time_ref,
            source_ref,
            target_ref,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NanlMotifDiscoveryConfig {
    pub minimum_support_threshold: u64,
    pub minimum_compression_gain_threshold: f64,
    pub maximum_motif_length: usize,
    pub maximum_parameter_count: usize,
}

impl Default for NanlMotifDiscoveryConfig {
    fn default() -> Self {
        Self {
            minimum_support_threshold: 2,
            minimum_compression_gain_threshold: 1.0,
            maximum_motif_length: 16,
            maximum_parameter_count: 8,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NanlMotifHeader {
    pub magic: String,
    pub version: String,
    pub motif_count: u64,
    pub source_cse_hash: String,
    pub source_event_hash: String,
    pub motif_dictionary_hash: String,
    pub merkle_root: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NanlMotifEntry {
    pub motif_id: u64,
    pub motif_kind: NanlMotifKind,
    pub symbol_span_ref: [u64; 2],
    pub parameter_schema_ref: u64,
    pub occurrence_count: u64,
    pub support_score: f64,
    pub compression_gain_estimate: f64,
    pub reconstruction_ref: u64,
    pub symbols: Vec<String>,
    pub parameter_refs: Vec<u64>,
    pub motif_hash: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NanlMotifBook {
    pub header: NanlMotifHeader,
    pub motifs: Vec<NanlMotifEntry>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NanlMotifCode {
    pub observation_id: u64,
    pub motif_id: u64,
    pub time_ref: u64,
    pub source_ref: u64,
    pub target_ref: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NanlMotifValidationReport {
    pub status: String,
    pub source_cse_hash: String,
    pub source_event_hash: String,
    pub motif_dictionary_hash: String,
    pub merkle_root: String,
    pub motif_count: u64,
    pub stable_ordering: bool,
    pub required_motif_classes_present: bool,
    pub deterministic_tie_breaker: bool,
    pub issues: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NrfMotifProjection {
    pub motif_id: u64,
    pub motif_kind: String,
    pub occurrence_count: u64,
    pub motif_hash: String,
}

pub fn discover_motifs(
    source_cse_hash: impl Into<String>,
    source_event_hash: impl Into<String>,
    observations: &[NanlMotifObservation],
    config: NanlMotifDiscoveryConfig,
) -> NanlMotifResult<NanlMotifBook> {
    let source_cse_hash = source_cse_hash.into();
    let source_event_hash = source_event_hash.into();
    if source_cse_hash.trim().is_empty() {
        return Err(NanlMotifError::Format(
            "NANL-MOTIF missing source CSE hash".to_string(),
        ));
    }
    if source_event_hash.trim().is_empty() {
        return Err(NanlMotifError::Format(
            "NANL-MOTIF missing source event hash".to_string(),
        ));
    }
    if observations.is_empty() {
        return Err(NanlMotifError::Support(
            "NANL-MOTIF discovery requires observations".to_string(),
        ));
    }

    let mut groups: BTreeMap<MotifKey, Vec<&NanlMotifObservation>> = BTreeMap::new();
    for observation in observations {
        validate_observation_bounds(observation, &config)?;
        groups
            .entry(MotifKey::from(observation))
            .or_default()
            .push(observation);
    }

    let mut motifs = Vec::new();
    let total_observations = observations.len() as f64;
    for (key, group) in groups {
        let occurrence_count = group.len() as u64;
        if occurrence_count < config.minimum_support_threshold {
            continue;
        }
        let gain = estimate_compression_gain(&key.symbols, occurrence_count);
        if gain < config.minimum_compression_gain_threshold {
            continue;
        }
        let motif_id = motifs.len() as u64 + 1;
        let parameter_schema_ref = short_hash_u64(&key.parameter_refs)?;
        let reconstruction_ref = short_hash_u64(&key)?;
        let mut entry = NanlMotifEntry {
            motif_id,
            motif_kind: key.motif_kind,
            symbol_span_ref: [motif_id - 1, key.symbols.len() as u64],
            parameter_schema_ref,
            occurrence_count,
            support_score: occurrence_count as f64 / total_observations,
            compression_gain_estimate: gain,
            reconstruction_ref,
            symbols: key.symbols,
            parameter_refs: key.parameter_refs,
            motif_hash: String::new(),
        };
        entry.motif_hash = compute_motif_hash(&entry)?;
        motifs.push(entry);
    }

    if motifs.is_empty() {
        return Err(NanlMotifError::Support(
            "NANL-MOTIF no motif met minimum support/gain thresholds".to_string(),
        ));
    }

    let motif_dictionary_hash = compute_motif_dictionary_hash(&motifs)?;
    let merkle_root = compute_motif_merkle_root(&motifs)?;
    let book = NanlMotifBook {
        header: NanlMotifHeader {
            magic: NANL_MOTIF_MAGIC.to_string(),
            version: NANL_MOTIF_VERSION.to_string(),
            motif_count: motifs.len() as u64,
            source_cse_hash,
            source_event_hash,
            motif_dictionary_hash,
            merkle_root,
        },
        motifs,
    };
    book.fail_if_invalid()?;
    Ok(book)
}

pub fn encode_observations(
    book: &NanlMotifBook,
    observations: &[NanlMotifObservation],
) -> NanlMotifResult<Vec<NanlMotifCode>> {
    book.fail_if_invalid()?;
    observations
        .iter()
        .map(|observation| {
            let key = MotifKey::from(observation);
            let motif = book
                .motifs
                .iter()
                .find(|motif| {
                    motif.motif_kind == key.motif_kind
                        && motif.symbols == key.symbols
                        && motif.parameter_refs == key.parameter_refs
                })
                .ok_or_else(|| {
                    NanlMotifError::UnknownMotif(format!(
                        "{}:{:?}",
                        observation.motif_kind.as_str(),
                        observation.symbols
                    ))
                })?;
            Ok(NanlMotifCode {
                observation_id: observation.observation_id,
                motif_id: motif.motif_id,
                time_ref: observation.time_ref,
                source_ref: observation.source_ref,
                target_ref: observation.target_ref,
            })
        })
        .collect()
}

pub fn decode_codes(
    book: &NanlMotifBook,
    codes: &[NanlMotifCode],
) -> NanlMotifResult<Vec<NanlMotifObservation>> {
    book.fail_if_invalid()?;
    codes
        .iter()
        .map(|code| {
            let motif = book
                .motifs
                .iter()
                .find(|motif| motif.motif_id == code.motif_id)
                .ok_or_else(|| {
                    NanlMotifError::UnknownMotif(format!("motif_id:{}", code.motif_id))
                })?;
            Ok(NanlMotifObservation {
                observation_id: code.observation_id,
                motif_kind: motif.motif_kind,
                symbols: motif.symbols.clone(),
                parameter_refs: motif.parameter_refs.clone(),
                time_ref: code.time_ref,
                source_ref: code.source_ref,
                target_ref: code.target_ref,
            })
        })
        .collect()
}

pub fn validate_required_motif_classes(book: &NanlMotifBook) -> NanlMotifResult<()> {
    let present = book
        .motifs
        .iter()
        .map(|motif| motif.motif_kind)
        .collect::<BTreeSet<_>>();
    let missing = REQUIRED_MOTIF_KINDS
        .iter()
        .filter(|required| !present.contains(required))
        .map(|kind| kind.as_str())
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(NanlMotifError::Format(format!(
            "NANL-MOTIF missing required motif classes: {}",
            missing.join(",")
        )));
    }
    Ok(())
}

pub fn validate_motif_book(book: &NanlMotifBook) -> NanlMotifResult<NanlMotifValidationReport> {
    let mut issues = Vec::new();
    if book.header.magic != NANL_MOTIF_MAGIC {
        issues.push("NANL-MOTIF bad magic".to_string());
    }
    if book.header.version != NANL_MOTIF_VERSION {
        issues.push("NANL-MOTIF unsupported version".to_string());
    }
    if book.header.source_cse_hash.trim().is_empty() {
        issues.push("NANL-MOTIF missing source CSE hash".to_string());
    }
    if book.header.source_event_hash.trim().is_empty() {
        issues.push("NANL-MOTIF missing source event hash".to_string());
    }
    if book.header.motif_count != book.motifs.len() as u64 {
        issues.push("NANL-MOTIF motif count mismatch".to_string());
    }

    let stable_ordering = motifs_are_stably_ordered(&book.motifs);
    if !stable_ordering {
        issues.push("NANL-MOTIF entries violate deterministic stable ordering".to_string());
    }

    let mut seen_ids = BTreeSet::new();
    let mut seen_keys = BTreeSet::new();
    for (idx, motif) in book.motifs.iter().enumerate() {
        if motif.motif_id != idx as u64 + 1 {
            issues.push(format!(
                "NANL-MOTIF non-contiguous motif id {}",
                motif.motif_id
            ));
        }
        if !seen_ids.insert(motif.motif_id) {
            issues.push(format!("NANL-MOTIF duplicate motif id {}", motif.motif_id));
        }
        if !seen_keys.insert(motif_key_from_entry(motif)) {
            issues.push(format!(
                "NANL-MOTIF duplicate motif key {}",
                motif.motif_kind.as_str()
            ));
        }
        if motif.symbols.is_empty() {
            issues.push(format!(
                "NANL-MOTIF empty symbol span for motif {}",
                motif.motif_id
            ));
        }
        if motif.occurrence_count == 0 {
            issues.push(format!(
                "NANL-MOTIF zero support for motif {}",
                motif.motif_id
            ));
        }
        let expected_hash = compute_motif_hash(motif)?;
        if motif.motif_hash != expected_hash {
            issues.push(format!(
                "NANL-MOTIF bad motif hash for motif_id {}",
                motif.motif_id
            ));
        }
    }

    let motif_dictionary_hash = compute_motif_dictionary_hash(&book.motifs)?;
    if book.header.motif_dictionary_hash != motif_dictionary_hash {
        issues.push("NANL-MOTIF dictionary hash mismatch".to_string());
    }
    let merkle_root = compute_motif_merkle_root(&book.motifs)?;
    if book.header.merkle_root != merkle_root {
        issues.push("NANL-MOTIF Merkle root mismatch".to_string());
    }

    let required_motif_classes_present = REQUIRED_MOTIF_KINDS.iter().all(|required| {
        book.motifs
            .iter()
            .any(|motif| motif.motif_kind == *required)
    });
    let deterministic_tie_breaker = stable_ordering;

    Ok(NanlMotifValidationReport {
        status: if issues.is_empty() { "PASS" } else { "FAIL" }.to_string(),
        source_cse_hash: book.header.source_cse_hash.clone(),
        source_event_hash: book.header.source_event_hash.clone(),
        motif_dictionary_hash,
        merkle_root,
        motif_count: book.motifs.len() as u64,
        stable_ordering,
        required_motif_classes_present,
        deterministic_tie_breaker,
        issues,
    })
}

impl NanlMotifBook {
    pub fn fail_if_invalid(&self) -> NanlMotifResult<()> {
        let report = validate_motif_book(self)?;
        if report.status != "PASS" {
            return Err(classify_validation_issues(report.issues));
        }
        Ok(())
    }
}

pub fn motif_to_nrf(book: &NanlMotifBook) -> NanlMotifResult<Vec<NrfMotifProjection>> {
    book.fail_if_invalid()?;
    Ok(book
        .motifs
        .iter()
        .map(|motif| NrfMotifProjection {
            motif_id: motif.motif_id,
            motif_kind: motif.motif_kind.as_str().to_string(),
            occurrence_count: motif.occurrence_count,
            motif_hash: motif.motif_hash.clone(),
        })
        .collect())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum MotifKind {
    WaterGate,
    ResidueCoupling,
    ThermoPulse,
    VariantTransition,
    DrfResonance,
    GFlowNetAction,
    TeacherDisagreement,
    UncertaintyShift,
}

impl MotifKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::WaterGate => "WATER_GATE_MOTIF",
            Self::ResidueCoupling => "RESIDUE_COUPLING_MOTIF",
            Self::ThermoPulse => "THERMO_PULSE_MOTIF",
            Self::VariantTransition => "VARIANT_TRANSITION_MOTIF",
            Self::DrfResonance => "DRF_RESONANCE_MOTIF",
            Self::GFlowNetAction => "GFLOWNET_ACTION_MOTIF",
            Self::TeacherDisagreement => "TEACHER_DISAGREEMENT_MOTIF",
            Self::UncertaintyShift => "UNCERTAINTY_SHIFT_MOTIF",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MotifDiscoveryPolicy {
    pub minimum_support: u64,
    pub minimum_compression_gain: i64,
    pub maximum_motif_length: usize,
    pub maximum_parameter_count: usize,
}

impl Default for MotifDiscoveryPolicy {
    fn default() -> Self {
        Self {
            minimum_support: 2,
            minimum_compression_gain: 1,
            maximum_motif_length: 4,
            maximum_parameter_count: 8,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NanlMotif {
    pub motif_id: u64,
    pub motif_kind: MotifKind,
    pub symbol_span: Vec<u64>,
    pub parameter_schema_ref: u64,
    pub occurrence_count: u64,
    pub support_score_milli: u64,
    pub compression_gain_estimate: i64,
    pub reconstruction_ref: u64,
    pub motif_hash: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum MotifCode {
    Literal(u64),
    Motif { motif_id: u64, len: usize },
}

pub fn discover_numeric_motifs(
    symbols: &[u64],
    motif_kind: MotifKind,
    policy: MotifDiscoveryPolicy,
) -> NanlMotifResult<Vec<NanlMotif>> {
    if symbols.is_empty() {
        return Err(NanlMotifError::Support("motif source is empty".to_string()));
    }
    if policy.minimum_support < 2 || policy.maximum_motif_length < 2 {
        return Err(NanlMotifError::Format("invalid motif policy".to_string()));
    }
    let mut counts = BTreeMap::<Vec<u64>, u64>::new();
    let max_len = policy.maximum_motif_length.min(symbols.len());
    for len in 2..=max_len {
        for window in symbols.windows(len) {
            *counts.entry(window.to_vec()).or_insert(0) += 1;
        }
    }
    let mut candidates = Vec::new();
    for (span, occurrence_count) in counts {
        if occurrence_count < policy.minimum_support {
            continue;
        }
        let gain = ((span.len() as i64 - 1) * occurrence_count as i64) - span.len() as i64;
        if gain < policy.minimum_compression_gain {
            continue;
        }
        candidates.push((span, occurrence_count, gain));
    }
    candidates.sort_by(|left, right| {
        right
            .2
            .cmp(&left.2)
            .then_with(|| right.1.cmp(&left.1))
            .then_with(|| left.0.cmp(&right.0))
    });
    Ok(candidates
        .into_iter()
        .enumerate()
        .map(|(idx, (span, occurrence_count, gain))| {
            let motif_hash = legacy_motif_hash(motif_kind, &span, occurrence_count, gain);
            NanlMotif {
                motif_id: idx as u64,
                motif_kind,
                symbol_span: span,
                parameter_schema_ref: 0,
                occurrence_count,
                support_score_milli: occurrence_count * 1000 / symbols.len() as u64,
                compression_gain_estimate: gain,
                reconstruction_ref: idx as u64,
                motif_hash,
            }
        })
        .collect())
}

pub fn encode_with_motifs(
    symbols: &[u64],
    motifs: &[NanlMotif],
) -> NanlMotifResult<Vec<MotifCode>> {
    validate_motifs(motifs)?;
    let mut sorted = motifs.to_vec();
    sorted.sort_by(|left, right| {
        right
            .symbol_span
            .len()
            .cmp(&left.symbol_span.len())
            .then_with(|| left.motif_id.cmp(&right.motif_id))
    });
    let mut out = Vec::new();
    let mut idx = 0;
    while idx < symbols.len() {
        if let Some(motif) = sorted
            .iter()
            .find(|motif| symbols[idx..].starts_with(&motif.symbol_span))
        {
            out.push(MotifCode::Motif {
                motif_id: motif.motif_id,
                len: motif.symbol_span.len(),
            });
            idx += motif.symbol_span.len();
        } else {
            out.push(MotifCode::Literal(symbols[idx]));
            idx += 1;
        }
    }
    Ok(out)
}

pub fn decode_with_motifs(codes: &[MotifCode], motifs: &[NanlMotif]) -> NanlMotifResult<Vec<u64>> {
    validate_motifs(motifs)?;
    let lookup = motifs
        .iter()
        .map(|motif| (motif.motif_id, motif.symbol_span.clone()))
        .collect::<BTreeMap<_, _>>();
    let mut out = Vec::new();
    for code in codes {
        match code {
            MotifCode::Literal(value) => out.push(*value),
            MotifCode::Motif { motif_id, len } => {
                let span = lookup.get(motif_id).ok_or_else(|| {
                    NanlMotifError::UnknownMotif(format!("unknown motif id {motif_id}"))
                })?;
                if span.len() != *len {
                    return Err(NanlMotifError::Format("motif length mismatch".to_string()));
                }
                out.extend_from_slice(span);
            }
        }
    }
    Ok(out)
}

pub fn validate_motifs(motifs: &[NanlMotif]) -> NanlMotifResult<()> {
    let mut previous = None::<u64>;
    for motif in motifs {
        if motif.symbol_span.len() < 2 {
            return Err(NanlMotifError::Format("motif span too short".to_string()));
        }
        if motif.occurrence_count < 2 {
            return Err(NanlMotifError::Support("motif support too low".to_string()));
        }
        if motif.compression_gain_estimate < 1 {
            return Err(NanlMotifError::Support("motif gain too low".to_string()));
        }
        if let Some(previous) = previous {
            if motif.motif_id <= previous {
                return Err(NanlMotifError::Ordering("motif ids not stable".to_string()));
            }
        }
        previous = Some(motif.motif_id);
        let expected = legacy_motif_hash(
            motif.motif_kind,
            &motif.symbol_span,
            motif.occurrence_count,
            motif.compression_gain_estimate,
        );
        if motif.motif_hash != expected {
            return Err(NanlMotifError::Hash("motif hash mismatch".to_string()));
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
struct MotifKey {
    motif_kind: NanlMotifKind,
    symbols: Vec<String>,
    parameter_refs: Vec<u64>,
}

impl From<&NanlMotifObservation> for MotifKey {
    fn from(value: &NanlMotifObservation) -> Self {
        Self {
            motif_kind: value.motif_kind,
            symbols: value.symbols.clone(),
            parameter_refs: value.parameter_refs.clone(),
        }
    }
}

#[derive(Serialize)]
struct MotifHashMaterial<'a> {
    motif_id: u64,
    motif_kind: NanlMotifKind,
    symbol_span_ref: [u64; 2],
    parameter_schema_ref: u64,
    occurrence_count: u64,
    support_score_micros: i64,
    compression_gain_micros: i64,
    reconstruction_ref: u64,
    symbols: &'a [String],
    parameter_refs: &'a [u64],
}

#[derive(Serialize)]
struct MotifDictionaryHashMaterial<'a> {
    motif_id: u64,
    motif_kind: NanlMotifKind,
    symbols: &'a [String],
    parameter_refs: &'a [u64],
    occurrence_count: u64,
    motif_hash: &'a str,
}

fn validate_observation_bounds(
    observation: &NanlMotifObservation,
    config: &NanlMotifDiscoveryConfig,
) -> NanlMotifResult<()> {
    if observation.symbols.is_empty() {
        return Err(NanlMotifError::Format(
            "NANL-MOTIF observation has empty symbol span".to_string(),
        ));
    }
    if observation.symbols.len() > config.maximum_motif_length {
        return Err(NanlMotifError::Format(format!(
            "NANL-MOTIF observation exceeds maximum motif length {}",
            config.maximum_motif_length
        )));
    }
    if observation.parameter_refs.len() > config.maximum_parameter_count {
        return Err(NanlMotifError::Format(format!(
            "NANL-MOTIF observation exceeds maximum parameter count {}",
            config.maximum_parameter_count
        )));
    }
    if observation
        .symbols
        .iter()
        .any(|symbol| symbol.trim().is_empty())
    {
        return Err(NanlMotifError::Format(
            "NANL-MOTIF observation contains empty symbol".to_string(),
        ));
    }
    Ok(())
}

fn estimate_compression_gain(symbols: &[String], occurrence_count: u64) -> f64 {
    let symbol_bytes: usize = symbols.iter().map(|symbol| symbol.len()).sum();
    let raw_per_occurrence = symbol_bytes + symbols.len().saturating_sub(1) + 24;
    let raw_total = raw_per_occurrence as u64 * occurrence_count;
    let encoded_total = symbol_bytes as u64 + 24 + occurrence_count * 8;
    raw_total.saturating_sub(encoded_total) as f64
}

fn compute_motif_hash(entry: &NanlMotifEntry) -> NanlMotifResult<String> {
    hash_json(&MotifHashMaterial {
        motif_id: entry.motif_id,
        motif_kind: entry.motif_kind,
        symbol_span_ref: entry.symbol_span_ref,
        parameter_schema_ref: entry.parameter_schema_ref,
        occurrence_count: entry.occurrence_count,
        support_score_micros: scaled_f64(entry.support_score),
        compression_gain_micros: scaled_f64(entry.compression_gain_estimate),
        reconstruction_ref: entry.reconstruction_ref,
        symbols: &entry.symbols,
        parameter_refs: &entry.parameter_refs,
    })
}

fn compute_motif_dictionary_hash(entries: &[NanlMotifEntry]) -> NanlMotifResult<String> {
    let material = entries
        .iter()
        .map(|entry| MotifDictionaryHashMaterial {
            motif_id: entry.motif_id,
            motif_kind: entry.motif_kind,
            symbols: &entry.symbols,
            parameter_refs: &entry.parameter_refs,
            occurrence_count: entry.occurrence_count,
            motif_hash: &entry.motif_hash,
        })
        .collect::<Vec<_>>();
    hash_json(&material)
}

fn compute_motif_merkle_root(entries: &[NanlMotifEntry]) -> NanlMotifResult<String> {
    if entries.is_empty() {
        return Err(NanlMotifError::Hash(
            "NANL-MOTIF Merkle root requires at least one motif".to_string(),
        ));
    }
    let leaves = entries
        .iter()
        .map(|entry| entry.motif_hash.as_str())
        .collect::<Vec<_>>();
    hash_json(&leaves)
}

fn motifs_are_stably_ordered(entries: &[NanlMotifEntry]) -> bool {
    entries
        .windows(2)
        .all(|window| motif_key_from_entry(&window[0]) < motif_key_from_entry(&window[1]))
}

fn motif_key_from_entry(entry: &NanlMotifEntry) -> MotifKey {
    MotifKey {
        motif_kind: entry.motif_kind,
        symbols: entry.symbols.clone(),
        parameter_refs: entry.parameter_refs.clone(),
    }
}

fn short_hash_u64<T: Serialize>(value: &T) -> NanlMotifResult<u64> {
    let bytes =
        serde_json::to_vec(value).map_err(|err| NanlMotifError::Internal(err.to_string()))?;
    let digest = blake3::hash(&bytes);
    let mut out = [0u8; 8];
    out.copy_from_slice(&digest.as_bytes()[..8]);
    Ok(u64::from_le_bytes(out))
}

fn scaled_f64(value: f64) -> i64 {
    (value * 1_000_000.0).round() as i64
}

fn classify_validation_issues(issues: Vec<String>) -> NanlMotifError {
    let text = issues.join(";");
    if text.contains("stable ordering") || text.contains("non-contiguous") {
        NanlMotifError::Ordering(text)
    } else if text.contains("hash") || text.contains("Merkle") {
        NanlMotifError::Hash(text)
    } else if text.contains("support") {
        NanlMotifError::Support(text)
    } else {
        NanlMotifError::Format(text)
    }
}

fn legacy_motif_hash(kind: MotifKind, span: &[u64], count: u64, gain: i64) -> String {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(kind.as_str().as_bytes());
    for value in span {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes.extend_from_slice(&count.to_le_bytes());
    bytes.extend_from_slice(&gain.to_le_bytes());
    blake3::hash(&bytes).to_hex().to_string()
}

fn hash_json<T: Serialize>(value: &T) -> NanlMotifResult<String> {
    let bytes =
        serde_json::to_vec(value).map_err(|err| NanlMotifError::Internal(err.to_string()))?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}
