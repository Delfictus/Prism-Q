use crate::binary_runtime::{
    compile_real_corpus_with_debug, compile_reference_corpus_with_debug, PcError, PcResult,
    ReferenceCorpus, ReferenceEvent, ReferenceProvenance, ReferenceState, ReferenceThermo,
    ReferenceTime,
};
use crate::cse_dictionaries::{CseDictionaries, CseDictionaryKind, CseDictionarySymbols};
use crate::cse_tables::*;
use crate::{nanl_bin, nanl_q4};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use std::time::Instant;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CseError {
    #[error("PC_ERR_FORMAT: {0}")]
    Format(String),
    #[error("PC_ERR_TYPE: {0}")]
    Type(String),
    #[error("PC_ERR_PROVENANCE: {0}")]
    Provenance(String),
    #[error("PC_ERR_TIMEBASE: {0}")]
    Timebase(String),
    #[error("PC_ERR_SCIENCE_ASSERT: {0}")]
    Science(String),
    #[error("PC_ERR_HASH: {0}")]
    Hash(String),
    #[error("PC_ERR_JSON: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CseCompilation {
    pub schema_version: String,
    pub state_id: String,
    pub canonical_hash: String,
    pub dictionaries_hash: String,
    pub tables_hash: String,
    pub dictionaries: CseDictionaries,
    pub tables: CseHdComponentTables,
    pub validation: CseValidationReport,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CseValidationReport {
    pub status: String,
    pub state_id: String,
    pub canonical_timebase_count: usize,
    pub entity_count: usize,
    pub molecule_count: usize,
    pub protein_count: usize,
    pub water_site_count: usize,
    pub event_count: usize,
    pub thermo_count: usize,
    pub nrf_frame_count: usize,
    pub drf_state_count: usize,
    pub synthetic: bool,
    pub classification: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CseRoundTripReport {
    pub status: String,
    pub canonical_hash: String,
    pub reconstructed_hash: String,
    pub canonical_roundtrip_hash_pass: bool,
    pub reconstructed_event_count: usize,
    pub reconstructed_entity_count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum NanlMode {
    Bin,
    Q4,
    Both,
}

impl NanlMode {
    pub fn parse(value: &str) -> PcResult<Self> {
        match value.to_ascii_lowercase().as_str() {
            "bin" => Ok(Self::Bin),
            "q4" => Ok(Self::Q4),
            "both" => Ok(Self::Both),
            other => Err(PcError::Format(format!("unknown NANL mode {other}"))),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CseScientificCompileReport {
    pub status: String,
    pub cse_mode: String,
    pub nanl_mode: NanlMode,
    pub output_path: String,
    pub raw_json_size: u64,
    pub canonical_table_size: u64,
    pub nanl_bin_size: u64,
    pub nanl_q4_size: u64,
    pub nrf_frame_size: u64,
    pub drf_state_size: u64,
    pub encode_mb_s: f64,
    pub roundtrip_canonical_hash_pass: bool,
    pub canonical_hash: String,
    pub nanl_bin_hash_blake3: String,
    pub nanl_q4_hash_blake3: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CseHdSidecarReport {
    pub status: String,
    pub input_path: String,
    pub output_dir: String,
    pub canonical_hash: String,
    pub artifacts: Vec<CseHdSidecarArtifact>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CseHdSidecarArtifact {
    pub artifact_kind: String,
    pub path: String,
    pub bytes: u64,
    pub hash_blake3: String,
    pub producer: String,
    pub validator: String,
    pub status: String,
}

#[derive(Serialize)]
struct CanonicalCseImage<'a> {
    schema_version: &'a str,
    state_id: &'a str,
    dictionaries: &'a CseDictionaries,
    tables: &'a CseHdComponentTables,
}

pub fn compile_cse_hd_from_json_str(input: &str) -> Result<CseCompilation, CseError> {
    let value: Value = serde_json::from_str(input)?;
    reject_untyped_jsonish_domain_objects(&value, "$")?;
    let source: CseSourceDocument = serde_json::from_value(value)?;
    compile_cse_hd(&source)
}

pub fn compile_scientific_state(
    input: &Path,
    cse_mode: &str,
    nanl_mode: NanlMode,
    output: &Path,
) -> PcResult<CseScientificCompileReport> {
    if !cse_mode.eq_ignore_ascii_case("hd") {
        return Err(PcError::Format(format!(
            "unsupported CSE mode {cse_mode}; only hd is implemented"
        )));
    }
    let start = Instant::now();
    let raw = fs::read_to_string(input)?;
    let value: Value = serde_json::from_str(&raw)?;
    reject_untyped_jsonish_domain_objects(&value, "$")
        .map_err(|err| PcError::Format(err.to_string()))?;
    let source: CseSourceDocument = serde_json::from_value(value)?;
    if !source.synthetic {
        validate_real_source_provenance(&source)?;
    }
    let compilation = compile_cse_hd(&source).map_err(|err| PcError::Format(err.to_string()))?;

    let tokens = tokens_from_tables(&compilation.tables);
    let nanl_bin_packet =
        nanl_bin::encode_tokens(&tokens).map_err(|err| PcError::Format(err.to_string()))?;
    let nanl_q4_packet = if matches!(nanl_mode, NanlMode::Q4 | NanlMode::Both) {
        Some(
            nanl_q4::encode_payload(nanl_bin_packet.bytes())
                .map_err(|err| PcError::Format(err.to_string()))?,
        )
    } else {
        None
    };
    nanl_bin::validate_packet(nanl_bin_packet.bytes())
        .map_err(|err| PcError::Format(err.to_string()))?;
    if let Some(packet) = &nanl_q4_packet {
        nanl_q4::validate_packet(packet.bytes()).map_err(|err| PcError::Format(err.to_string()))?;
    }

    let reference = reference_corpus_from_cse(&source, &compilation, nanl_bin_packet.bytes())?;
    if source.synthetic {
        compile_reference_corpus_with_debug(&reference, raw.as_bytes(), output)?;
    } else {
        compile_real_corpus_with_debug(&reference, raw.as_bytes(), output)?;
    }

    let elapsed = start.elapsed().as_secs_f64().max(0.000_001);
    let raw_json_size = raw.len() as u64;
    let canonical_table_size = serde_json::to_vec(&compilation.tables)?.len() as u64;
    let nrf_frame_size = serde_json::to_vec(&compilation.tables.nrf_frame_rows)?.len() as u64;
    let drf_state_size = serde_json::to_vec(&compilation.tables.drf_state_rows)?.len() as u64;
    Ok(CseScientificCompileReport {
        status: "PASS".to_string(),
        cse_mode: "hd".to_string(),
        nanl_mode,
        output_path: output.display().to_string(),
        raw_json_size,
        canonical_table_size,
        nanl_bin_size: nanl_bin_packet.bytes().len() as u64,
        nanl_q4_size: nanl_q4_packet
            .as_ref()
            .map(|packet| packet.bytes().len() as u64)
            .unwrap_or(0),
        nrf_frame_size,
        drf_state_size,
        encode_mb_s: (raw_json_size as f64 / 1_048_576.0) / elapsed,
        roundtrip_canonical_hash_pass: nanl_bin_packet.report().roundtrip_canonical_hash_pass
            && nanl_q4_packet
                .as_ref()
                .map(|packet| packet.report().canonical_roundtrip_hash_pass)
                .unwrap_or(true),
        canonical_hash: compilation.canonical_hash,
        nanl_bin_hash_blake3: blake3::hash(nanl_bin_packet.bytes()).to_hex().to_string(),
        nanl_q4_hash_blake3: nanl_q4_packet
            .as_ref()
            .map(|packet| blake3::hash(packet.bytes()).to_hex().to_string()),
    })
}

fn validate_real_source_provenance(source: &CseSourceDocument) -> PcResult<()> {
    if source.synthetic {
        return Err(PcError::Provenance(
            "real scientific-state source requires synthetic=false".to_string(),
        ));
    }
    if source.classification != "COMPUTATIONAL" {
        return Err(PcError::Provenance(format!(
            "real scientific-state source classification must be COMPUTATIONAL, observed {}",
            source.classification
        )));
    }
    if source.provenance.is_empty() {
        return Err(PcError::Provenance(
            "real scientific-state source has no provenance records".to_string(),
        ));
    }
    for provenance in &source.provenance {
        if provenance.synthetic {
            return Err(PcError::Provenance(format!(
                "real scientific-state source contains synthetic provenance {}",
                provenance.id
            )));
        }
        if provenance.classification != "COMPUTATIONAL" {
            return Err(PcError::Provenance(format!(
                "real scientific-state provenance {} classification must be COMPUTATIONAL, observed {}",
                provenance.id, provenance.classification
            )));
        }
        if provenance.source_hash.len() != 64
            || !provenance
                .source_hash
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(PcError::Provenance(format!(
                "real scientific-state provenance {} source_hash is not lowercase SHA-256 hex",
                provenance.id
            )));
        }
    }
    let provenance_hashes = source
        .provenance
        .iter()
        .map(|provenance| (provenance.id.as_str(), provenance.source_hash.as_str()))
        .collect::<BTreeMap<_, _>>();
    for (entity_kind, entity_id, topology_hash, provenance_ref) in source
        .molecules
        .iter()
        .map(|entity| {
            (
                "molecule",
                entity.id.as_str(),
                entity.topology_hash.as_str(),
                entity.provenance_ref.as_str(),
            )
        })
        .chain(source.proteins.iter().map(|entity| {
            (
                "protein",
                entity.id.as_str(),
                entity.topology_hash.as_str(),
                entity.provenance_ref.as_str(),
            )
        }))
    {
        let source_hash = provenance_hashes.get(provenance_ref).ok_or_else(|| {
            PcError::Provenance(format!(
                "real scientific-state {entity_kind} {entity_id} references unknown provenance {provenance_ref}"
            ))
        })?;
        if topology_hash != *source_hash {
            return Err(PcError::Provenance(format!(
                "real scientific-state {entity_kind} {entity_id} topology_hash does not match provenance {provenance_ref} source_hash"
            )));
        }
    }
    Ok(())
}

pub fn write_cse_hd_sidecars(input: &Path, out_dir: &Path) -> PcResult<CseHdSidecarReport> {
    fs::create_dir_all(out_dir)?;
    let raw = fs::read_to_string(input)?;
    let compilation =
        compile_cse_hd_from_json_str(&raw).map_err(|err| PcError::Format(err.to_string()))?;
    let tokens = nanl_tokens_from_tables(&compilation.tables);
    let nanl_bin_packet =
        nanl_bin::encode_tokens(&tokens).map_err(|err| PcError::Format(err.to_string()))?;
    nanl_bin::validate_packet(nanl_bin_packet.bytes())
        .map_err(|err| PcError::Format(err.to_string()))?;
    let nanl_q4_packet = nanl_q4::encode_payload(nanl_bin_packet.bytes())
        .map_err(|err| PcError::Format(err.to_string()))?;
    nanl_q4::validate_packet(nanl_q4_packet.bytes())
        .map_err(|err| PcError::Format(err.to_string()))?;

    let tables_path = out_dir.join("cse_hd_canonical_tables.arrow");
    let nanl_bin_path = out_dir.join("cse_hd_nanl_bin.nanlb");
    let nanl_q4_path = out_dir.join("cse_hd_nanl_q4.nanlq4");
    let nrf_path = out_dir.join("cse_hd_nrf_frames.nrf");
    let drf_path = out_dir.join("cse_hd_drf_graph.drf");

    let mut artifacts = Vec::new();
    artifacts.push(write_labeled_binary_sidecar(
        "CSE_CANONICAL_TABLES_ARROW_COMPATIBLE_BINARY",
        &tables_path,
        b"PRISM_CSEHD_TABLES_ARROW_COMPAT_V0_1\0",
        &serde_json::to_vec(&compilation.tables)?,
        "prismc cse-hd-sidecars",
        "compile_cse_hd_from_json_str + canonical table hash",
    )?);
    artifacts.push(write_raw_sidecar(
        "NANL_BIN_EXECUTABLE_STREAM",
        &nanl_bin_path,
        nanl_bin_packet.bytes(),
        "prismc cse-hd-sidecars",
        "nanl_bin::validate_packet",
    )?);
    artifacts.push(write_raw_sidecar(
        "NANL_Q4_CONSTRAINED_STREAM",
        &nanl_q4_path,
        nanl_q4_packet.bytes(),
        "prismc cse-hd-sidecars",
        "nanl_q4::validate_packet",
    )?);
    artifacts.push(write_labeled_binary_sidecar(
        "NRF_SPARSE_EVENT_FRAMES",
        &nrf_path,
        b"PRISM_NRF_FRAME_SET_V0_1\0",
        &serde_json::to_vec(&compilation.tables.nrf_frame_rows)?,
        "prismc cse-hd-sidecars",
        "CSE source refs + nrf_frame_rows serialization",
    )?);
    artifacts.push(write_labeled_binary_sidecar(
        "DRF_RESONANCE_GRAPH",
        &drf_path,
        b"PRISM_DRF_GRAPH_SET_V0_1\0",
        &serde_json::to_vec(&compilation.tables.drf_state_rows)?,
        "prismc cse-hd-sidecars",
        "CSE source refs + drf_state_rows serialization",
    )?);

    Ok(CseHdSidecarReport {
        status: "PASS".to_string(),
        input_path: input.display().to_string(),
        output_dir: out_dir.display().to_string(),
        canonical_hash: compilation.canonical_hash,
        artifacts,
    })
}

pub fn compile_cse_hd(source: &CseSourceDocument) -> Result<CseCompilation, CseError> {
    let canonical_source = canonicalize_source(source);
    let validation = validate_cse_source(&canonical_source)?;
    let dictionaries = build_dictionaries(&canonical_source);
    let tables = build_component_tables(&canonical_source, &dictionaries)?;
    let dictionaries_hash = canonical_hash(&dictionaries)?;
    let tables_hash = canonical_hash(&tables)?;
    let canonical_hash = canonical_hash(&CanonicalCseImage {
        schema_version: &canonical_source.schema_version,
        state_id: &canonical_source.state_id,
        dictionaries: &dictionaries,
        tables: &tables,
    })?;

    Ok(CseCompilation {
        schema_version: canonical_source.schema_version,
        state_id: canonical_source.state_id,
        canonical_hash,
        dictionaries_hash,
        tables_hash,
        dictionaries,
        tables,
        validation,
    })
}

pub fn validate_cse_source(source: &CseSourceDocument) -> Result<CseValidationReport, CseError> {
    require_nonempty("schema_version", &source.schema_version)?;
    require_nonempty("state_id", &source.state_id)?;
    require_nonempty("classification", &source.classification)?;
    ensure_unique(
        source.unit_registry.iter().map(|unit| unit.id.as_str()),
        "unit",
    )?;
    ensure_unique(
        source.provenance.iter().map(|prov| prov.id.as_str()),
        "provenance",
    )?;
    ensure_unique(
        source.timebases.iter().map(|time| time.id.as_str()),
        "timebase",
    )?;
    ensure_unique(
        source.molecules.iter().map(|mol| mol.id.as_str()),
        "molecule",
    )?;
    ensure_unique(
        source.proteins.iter().map(|protein| protein.id.as_str()),
        "protein",
    )?;
    ensure_unique(
        source.water_sites.iter().map(|water| water.id.as_str()),
        "water_site",
    )?;
    ensure_unique(
        source.variants.iter().map(|variant| variant.id.as_str()),
        "variant",
    )?;
    ensure_unique(source.events.iter().map(|event| event.id.as_str()), "event")?;
    ensure_unique(
        source.uncertainties.iter().map(|unc| unc.id.as_str()),
        "uncertainty",
    )?;
    ensure_unique(
        source.thermo.iter().map(|thermo| thermo.id.as_str()),
        "thermo",
    )?;
    ensure_unique(
        source.teacher_outputs.iter().map(|row| row.id.as_str()),
        "teacher_output",
    )?;
    ensure_unique(
        source.gflownet_traces.iter().map(|row| row.id.as_str()),
        "gflownet_trace",
    )?;
    ensure_unique(
        source.nrf_frames.iter().map(|row| row.id.as_str()),
        "nrf_frame",
    )?;
    ensure_unique(
        source.drf_states.iter().map(|row| row.id.as_str()),
        "drf_state",
    )?;

    let unit_ids = source
        .unit_registry
        .iter()
        .map(|unit| {
            require_nonempty("unit.id", &unit.id)?;
            require_nonempty("unit.quantity_kind", &unit.quantity_kind)?;
            require_nonempty("unit.unit", &unit.unit)?;
            Ok(unit.id.as_str())
        })
        .collect::<Result<BTreeSet<_>, CseError>>()?;
    let provenance_ids = source
        .provenance
        .iter()
        .map(|prov| {
            require_nonempty("provenance.id", &prov.id)?;
            require_nonempty("provenance.source_uri", &prov.source_uri)?;
            require_nonempty("provenance.source_hash", &prov.source_hash)?;
            require_nonempty("provenance.classification", &prov.classification)?;
            Ok(prov.id.as_str())
        })
        .collect::<Result<BTreeSet<_>, CseError>>()?;
    let timebase_ids = source
        .timebases
        .iter()
        .map(|time| {
            require_nonempty("timebase.id", &time.id)?;
            require_nonempty("timebase.timebase_hash", &time.timebase_hash)?;
            require_ref("timebase.unit_ref", &time.unit_ref, &unit_ids)?;
            if time.frame_end < time.frame_start {
                return Err(CseError::Timebase(format!(
                    "timebase {} frame_end precedes frame_start",
                    time.id
                )));
            }
            Ok(time.id.as_str())
        })
        .collect::<Result<BTreeSet<_>, CseError>>()?;

    let mut state_refs = BTreeSet::from([source.state_id.as_str()]);
    for molecule in &source.molecules {
        validate_molecule(molecule, &provenance_ids)?;
        state_refs.insert(molecule.id.as_str());
        for atom in &molecule.atoms {
            state_refs.insert(atom.id.as_str());
        }
    }
    for protein in &source.proteins {
        validate_protein(protein, &provenance_ids)?;
        state_refs.insert(protein.id.as_str());
        for chain in &protein.chains {
            state_refs.insert(chain.id.as_str());
            for residue in &chain.residues {
                state_refs.insert(residue.id.as_str());
            }
        }
        for site in &protein.atom_sites {
            state_refs.insert(site.id.as_str());
        }
    }
    for water in &source.water_sites {
        validate_water_site(water, &timebase_ids, &provenance_ids)?;
        state_refs.insert(water.id.as_str());
    }
    for variant in &source.variants {
        require_nonempty("variant.id", &variant.id)?;
        require_ref(
            "variant.reference_state_ref",
            &variant.reference_state_ref,
            &state_refs,
        )?;
        require_ref(
            "variant.alternate_state_ref",
            &variant.alternate_state_ref,
            &state_refs,
        )?;
        require_nonempty("variant.delta_hash", &variant.delta_hash)?;
        require_ref(
            "variant.provenance_ref",
            &variant.provenance_ref,
            &provenance_ids,
        )?;
        state_refs.insert(variant.id.as_str());
    }

    let mut event_orders = BTreeSet::new();
    let mut event_ids = BTreeSet::new();
    for event in &source.events {
        require_nonempty("event.id", &event.id)?;
        require_nonempty("event.kind", &event.kind)?;
        require_ref("event.source_ref", &event.source_ref, &state_refs)?;
        require_ref("event.target_ref", &event.target_ref, &state_refs)?;
        require_ref("event.time_ref", &event.time_ref, &timebase_ids)?;
        require_nonempty("event.motif", &event.motif)?;
        require_ref(
            "event.provenance_ref",
            &event.provenance_ref,
            &provenance_ids,
        )?;
        if !event_orders.insert(event.order_index) {
            return Err(CseError::Science(format!(
                "duplicate event order_index {}",
                event.order_index
            )));
        }
        event_ids.insert(event.id.as_str());
        state_refs.insert(event.id.as_str());
    }

    let mut uncertainty_ids = BTreeSet::new();
    for uncertainty in &source.uncertainties {
        require_nonempty("uncertainty.id", &uncertainty.id)?;
        require_ref(
            "uncertainty.target_ref",
            &uncertainty.target_ref,
            &state_refs,
        )?;
        require_nonempty("uncertainty.model", &uncertainty.model)?;
        require_nonempty(
            "uncertainty.calibration_hash",
            &uncertainty.calibration_hash,
        )?;
        require_ref(
            "uncertainty.provenance_ref",
            &uncertainty.provenance_ref,
            &provenance_ids,
        )?;
        uncertainty_ids.insert(uncertainty.id.as_str());
        state_refs.insert(uncertainty.id.as_str());
    }

    for thermo in &source.thermo {
        require_nonempty("thermo.id", &thermo.id)?;
        require_ref("thermo.target_ref", &thermo.target_ref, &state_refs)?;
        require_nonempty("thermo.quantity_kind", &thermo.quantity_kind)?;
        require_ref("thermo.unit_ref", &thermo.unit_ref, &unit_ids)?;
        require_nonempty("thermo.value_payload_ref", &thermo.value_payload_ref)?;
        require_ref(
            "thermo.uncertainty_ref",
            &thermo.uncertainty_ref,
            &uncertainty_ids,
        )?;
        require_ref(
            "thermo.provenance_ref",
            &thermo.provenance_ref,
            &provenance_ids,
        )?;
        state_refs.insert(thermo.id.as_str());
    }

    for teacher in &source.teacher_outputs {
        require_nonempty("teacher_output.id", &teacher.id)?;
        require_ref(
            "teacher_output.target_ref",
            &teacher.target_ref,
            &state_refs,
        )?;
        require_nonempty("teacher_output.model_ref", &teacher.model_ref)?;
        require_nonempty("teacher_output.label", &teacher.label)?;
        if !teacher.confidence.is_finite() || !(0.0..=1.0).contains(&teacher.confidence) {
            return Err(CseError::Science(format!(
                "teacher output {} confidence must be finite in [0,1]",
                teacher.id
            )));
        }
        require_nonempty("teacher_output.airlock_state", &teacher.airlock_state)?;
        if !matches!(
            teacher.airlock_state.as_str(),
            "MODEL_GENERATED_UNVERIFIED" | "AIRLOCK_ACCEPTED" | "HIL_RESOLVED"
        ) {
            return Err(CseError::Science(format!(
                "teacher output {} has unsupported airlock state {}",
                teacher.id, teacher.airlock_state
            )));
        }
        require_ref(
            "teacher_output.provenance_ref",
            &teacher.provenance_ref,
            &provenance_ids,
        )?;
        state_refs.insert(teacher.id.as_str());
    }

    for trace in &source.gflownet_traces {
        require_nonempty("gflownet_trace.id", &trace.id)?;
        require_ref("gflownet_trace.state_ref", &trace.state_ref, &state_refs)?;
        require_nonempty("gflownet_trace.action", &trace.action)?;
        require_nonempty(
            "gflownet_trace.reward_payload_ref",
            &trace.reward_payload_ref,
        )?;
        require_nonempty("gflownet_trace.lineage_hash", &trace.lineage_hash)?;
        require_ref(
            "gflownet_trace.provenance_ref",
            &trace.provenance_ref,
            &provenance_ids,
        )?;
        state_refs.insert(trace.id.as_str());
    }

    if source.nanl_symbols.is_empty() {
        return Err(CseError::Type("nanl_symbols must not be empty".to_string()));
    }
    for symbol in &source.nanl_symbols {
        require_nonempty("nanl_symbol", symbol)?;
    }

    let mut nrf_ids = BTreeSet::new();
    for frame in &source.nrf_frames {
        require_nonempty("nrf_frame.id", &frame.id)?;
        require_ref("nrf_frame.timebase_ref", &frame.timebase_ref, &timebase_ids)?;
        require_nonempty("nrf_frame.tile_kind", &frame.tile_kind)?;
        if frame.source_event_refs.is_empty() {
            return Err(CseError::Science(format!(
                "nrf frame {} has no source events",
                frame.id
            )));
        }
        for event_ref in &frame.source_event_refs {
            require_ref("nrf_frame.source_event_ref", event_ref, &event_ids)?;
        }
        nrf_ids.insert(frame.id.as_str());
        state_refs.insert(frame.id.as_str());
    }

    for drf in &source.drf_states {
        require_nonempty("drf_state.id", &drf.id)?;
        require_ref("drf_state.source_nrf_ref", &drf.source_nrf_ref, &nrf_ids)?;
        if drf.source_cse_ref != source.state_id {
            return Err(CseError::Science(format!(
                "drf state {} source_cse_ref must bind to source state_id {}",
                drf.id, source.state_id
            )));
        }
        require_nonempty("drf_state.channel_kind", &drf.channel_kind)?;
        if drf.tensor_shape.is_empty() || drf.tensor_shape.contains(&0) {
            return Err(CseError::Science(format!(
                "drf state {} tensor_shape must be non-empty and non-zero",
                drf.id
            )));
        }
        require_ref(
            "drf_state.provenance_ref",
            &drf.provenance_ref,
            &provenance_ids,
        )?;
        if drf.canonical_or_derived != "DRF_CANONICAL" && drf.canonical_or_derived != "DRF_DERIVED"
        {
            return Err(CseError::Science(format!(
                "drf state {} must declare DRF_CANONICAL or DRF_DERIVED",
                drf.id
            )));
        }
    }

    Ok(CseValidationReport {
        status: "PASS".to_string(),
        state_id: source.state_id.clone(),
        canonical_timebase_count: source.timebases.len(),
        entity_count: source.molecules.len() + source.proteins.len() + source.water_sites.len(),
        molecule_count: source.molecules.len(),
        protein_count: source.proteins.len(),
        water_site_count: source.water_sites.len(),
        event_count: source.events.len(),
        thermo_count: source.thermo.len(),
        nrf_frame_count: source.nrf_frames.len(),
        drf_state_count: source.drf_states.len(),
        synthetic: source.synthetic,
        classification: source.classification.clone(),
    })
}

pub fn reconstruct_cse_source(compilation: &CseCompilation) -> Result<CseSourceDocument, CseError> {
    let schema = compilation
        .tables
        .schema_rows
        .first()
        .ok_or_else(|| CseError::Format("missing schema row".to_string()))?;
    if compilation.tables.schema_rows.len() != 1 {
        return Err(CseError::Format(
            "CSE reconstruction requires exactly one schema row".to_string(),
        ));
    }

    let molecules = compilation
        .tables
        .entity_rows
        .iter()
        .filter(|row| row.entity_kind == "MOLECULE")
        .map(|entity| CseHdMoleculeSource {
            id: entity.entity_id.clone(),
            topology_hash: entity.topology_hash.clone(),
            provenance_ref: entity.provenance_ref.clone(),
            atoms: compilation
                .tables
                .atom_rows
                .iter()
                .filter(|row| row.molecule_ref == entity.entity_id)
                .map(|row| CseHdAtomSource {
                    id: row.atom_id.clone(),
                    element: row.element.clone(),
                    atom_type: row.atom_type.clone(),
                })
                .collect(),
            bonds: compilation
                .tables
                .bond_rows
                .iter()
                .filter(|row| row.molecule_ref == entity.entity_id)
                .map(|row| CseHdBondSource {
                    id: row.bond_id.clone(),
                    source_atom: row.source_atom.clone(),
                    target_atom: row.target_atom.clone(),
                    order: row.order,
                })
                .collect(),
        })
        .collect::<Vec<_>>();

    let proteins = compilation
        .tables
        .entity_rows
        .iter()
        .filter(|row| row.entity_kind == "PROTEIN")
        .map(|entity| {
            let atom_sites = compilation
                .tables
                .atom_site_rows
                .iter()
                .filter(|row| row.protein_ref == entity.entity_id)
                .map(|row| CseHdAtomSiteSource {
                    id: row.site_id.clone(),
                    atom_name: row.atom_name.clone(),
                    atom_type: row.atom_type.clone(),
                    chain_ref: row.chain_ref.clone(),
                    residue_ref: row.residue_ref.clone(),
                })
                .collect();
            let chains = compilation
                .tables
                .protein_chain_rows
                .iter()
                .filter(|row| row.protein_ref == entity.entity_id)
                .map(|chain| CseHdProteinChainSource {
                    id: chain.chain_id.clone(),
                    residues: compilation
                        .tables
                        .residue_rows
                        .iter()
                        .filter(|row| {
                            row.protein_ref == entity.entity_id && row.chain_ref == chain.chain_id
                        })
                        .map(|row| CseHdResidueSource {
                            id: row.residue_id.clone(),
                            name: row.name.clone(),
                            sequence_index: row.sequence_index,
                            atom_site_ids: row.atom_site_ids.clone(),
                        })
                        .collect(),
                })
                .collect();
            CseHdProteinSource {
                id: entity.entity_id.clone(),
                topology_hash: entity.topology_hash.clone(),
                provenance_ref: entity.provenance_ref.clone(),
                chains,
                atom_sites,
            }
        })
        .collect::<Vec<_>>();

    Ok(canonicalize_source(&CseSourceDocument {
        schema_version: schema.schema_version.clone(),
        state_id: schema.state_id.clone(),
        classification: schema.classification.clone(),
        synthetic: schema.synthetic,
        unit_registry: compilation
            .tables
            .unit_rows
            .iter()
            .map(|row| CseUnitSource {
                id: row.unit_id.clone(),
                quantity_kind: row.quantity_kind.clone(),
                unit: row.unit.clone(),
            })
            .collect(),
        provenance: compilation
            .tables
            .provenance_rows
            .iter()
            .map(|row| CseProvenanceRecordSource {
                id: row.provenance_id.clone(),
                source_uri: row.source_uri.clone(),
                source_hash: row.source_hash.clone(),
                classification: row.classification.clone(),
                synthetic: row.synthetic,
            })
            .collect(),
        timebases: compilation
            .tables
            .timebase_rows
            .iter()
            .map(|row| CseHdTimebaseSource {
                id: row.timebase_id.clone(),
                timebase_hash: row.timebase_hash.clone(),
                unit_ref: row.unit_ref.clone(),
                frame_start: row.frame_start,
                frame_end: row.frame_end,
            })
            .collect(),
        molecules,
        proteins,
        water_sites: compilation
            .tables
            .water_site_rows
            .iter()
            .map(|row| CseHdWaterSiteSource {
                id: row.water_id.clone(),
                gate_kind: row.gate_kind.clone(),
                event_bound: row.event_bound,
                time_ref: row.time_ref.clone(),
                provenance_ref: row.provenance_ref.clone(),
            })
            .collect(),
        variants: compilation
            .tables
            .variant_rows
            .iter()
            .map(|row| CseHdVariantSource {
                id: row.variant_id.clone(),
                reference_state_ref: row.reference_state_ref.clone(),
                alternate_state_ref: row.alternate_state_ref.clone(),
                delta_hash: row.delta_hash.clone(),
                provenance_ref: row.provenance_ref.clone(),
            })
            .collect(),
        events: compilation
            .tables
            .event_rows
            .iter()
            .map(|row| CseHdEventSource {
                id: row.event_id.clone(),
                kind: row.event_kind.clone(),
                source_ref: row.source_ref.clone(),
                target_ref: row.target_ref.clone(),
                time_ref: row.time_ref.clone(),
                order_index: row.order_index,
                motif: row.motif.clone(),
                provenance_ref: row.provenance_ref.clone(),
            })
            .collect(),
        uncertainties: compilation
            .tables
            .uncertainty_rows
            .iter()
            .map(|row| CseHdUncertaintySource {
                id: row.uncertainty_id.clone(),
                target_ref: row.target_ref.clone(),
                model: row.model.clone(),
                calibration_hash: row.calibration_hash.clone(),
                provenance_ref: row.provenance_ref.clone(),
            })
            .collect(),
        thermo: compilation
            .tables
            .thermo_rows
            .iter()
            .map(|row| CseHdThermoSource {
                id: row.thermo_id.clone(),
                target_ref: row.target_ref.clone(),
                quantity_kind: row.quantity_kind.clone(),
                unit_ref: row.unit_ref.clone(),
                value_payload_ref: row.value_payload_ref.clone(),
                uncertainty_ref: row.uncertainty_ref.clone(),
                provenance_ref: row.provenance_ref.clone(),
            })
            .collect(),
        teacher_outputs: compilation
            .tables
            .teacher_output_rows
            .iter()
            .map(|row| CseHdTeacherOutputSource {
                id: row.teacher_output_id.clone(),
                target_ref: row.target_ref.clone(),
                model_ref: row.model_ref.clone(),
                label: row.label.clone(),
                confidence: row.confidence,
                airlock_state: row.airlock_state.clone(),
                provenance_ref: row.provenance_ref.clone(),
            })
            .collect(),
        gflownet_traces: compilation
            .tables
            .gflownet_trace_rows
            .iter()
            .map(|row| CseHdGFlowNetTraceSource {
                id: row.trace_id.clone(),
                state_ref: row.state_ref.clone(),
                action: row.action.clone(),
                reward_payload_ref: row.reward_payload_ref.clone(),
                lineage_hash: row.lineage_hash.clone(),
                provenance_ref: row.provenance_ref.clone(),
            })
            .collect(),
        nanl_symbols: compilation
            .tables
            .nanl_symbol_rows
            .iter()
            .map(|row| row.symbol.clone())
            .collect(),
        nrf_frames: compilation
            .tables
            .nrf_frame_rows
            .iter()
            .map(|row| CseHdNrfFrameSource {
                id: row.frame_id.clone(),
                timebase_ref: row.timebase_ref.clone(),
                tile_kind: row.tile_kind.clone(),
                source_event_refs: row.source_event_refs.clone(),
            })
            .collect(),
        drf_states: compilation
            .tables
            .drf_state_rows
            .iter()
            .map(|row| CseHdDrfStateSource {
                id: row.drf_state_id.clone(),
                source_nrf_ref: row.source_nrf_ref.clone(),
                source_cse_ref: row.source_cse_ref.clone(),
                channel_kind: row.channel_kind.clone(),
                tensor_shape: row.tensor_shape.clone(),
                provenance_ref: row.provenance_ref.clone(),
                canonical_or_derived: row.canonical_or_derived.clone(),
            })
            .collect(),
    }))
}

pub fn validate_roundtrip_reconstruction(
    source: &CseSourceDocument,
) -> Result<CseRoundTripReport, CseError> {
    let compilation = compile_cse_hd(source)?;
    let reconstructed = reconstruct_cse_source(&compilation)?;
    let reconstructed_compilation = compile_cse_hd(&reconstructed)?;
    if compilation.canonical_hash != reconstructed_compilation.canonical_hash {
        return Err(CseError::Hash(format!(
            "roundtrip hash mismatch: {} != {}",
            compilation.canonical_hash, reconstructed_compilation.canonical_hash
        )));
    }
    Ok(CseRoundTripReport {
        status: "PASS".to_string(),
        canonical_hash: compilation.canonical_hash,
        reconstructed_hash: reconstructed_compilation.canonical_hash,
        canonical_roundtrip_hash_pass: true,
        reconstructed_event_count: reconstructed.events.len(),
        reconstructed_entity_count: reconstructed.molecules.len()
            + reconstructed.proteins.len()
            + reconstructed.water_sites.len(),
    })
}

pub fn nanl_tokens_from_tables(tables: &CseHdComponentTables) -> Vec<nanl_bin::NanlToken> {
    let mut tokens = Vec::new();
    let lane_value = |idx: usize| -> u32 { (idx & 0b11) as u32 };
    for (idx, row) in tables.entity_rows.iter().enumerate() {
        tokens.push(nanl_bin::NanlToken::new(
            nanl_bin::NanlTokenClass::Entity,
            row.entity_dict_ref as u16,
            lane_value(idx),
        ));
    }
    for (idx, row) in tables.atom_rows.iter().enumerate() {
        tokens.push(nanl_bin::NanlToken::new(
            nanl_bin::NanlTokenClass::Atom,
            row.atom_type_dict_ref as u16,
            lane_value(idx),
        ));
    }
    for (idx, row) in tables.residue_rows.iter().enumerate() {
        tokens.push(nanl_bin::NanlToken::new(
            nanl_bin::NanlTokenClass::Residue,
            row.residue_dict_ref as u16,
            lane_value(idx),
        ));
    }
    for (idx, row) in tables.water_site_rows.iter().enumerate() {
        tokens.push(nanl_bin::NanlToken::new(
            nanl_bin::NanlTokenClass::Water,
            row.water_gate_dict_ref as u16,
            lane_value(idx),
        ));
    }
    for (idx, row) in tables.event_rows.iter().enumerate() {
        tokens.push(nanl_bin::NanlToken::new(
            nanl_bin::NanlTokenClass::EventKind,
            row.event_kind_dict_ref as u16,
            lane_value(idx),
        ));
        tokens.push(nanl_bin::NanlToken::new(
            nanl_bin::NanlTokenClass::EventMotif,
            row.event_motif_dict_ref as u16,
            lane_value(idx),
        ));
    }
    for (idx, row) in tables.thermo_rows.iter().enumerate() {
        tokens.push(nanl_bin::NanlToken::new(
            nanl_bin::NanlTokenClass::Thermo,
            row.quantity_dict_ref as u16,
            lane_value(idx),
        ));
    }
    for (idx, row) in tables.uncertainty_rows.iter().enumerate() {
        tokens.push(nanl_bin::NanlToken::new(
            nanl_bin::NanlTokenClass::Uncertainty,
            row.uncertainty_model_dict_ref as u16,
            lane_value(idx),
        ));
    }
    for (idx, row) in tables.teacher_output_rows.iter().enumerate() {
        tokens.push(nanl_bin::NanlToken::new(
            nanl_bin::NanlTokenClass::TeacherEvidence,
            row.label_dict_ref as u16,
            lane_value(idx),
        ));
    }
    for (idx, row) in tables.gflownet_trace_rows.iter().enumerate() {
        tokens.push(nanl_bin::NanlToken::new(
            nanl_bin::NanlTokenClass::GFlowNetAction,
            row.action_dict_ref as u16,
            lane_value(idx),
        ));
    }
    for (idx, row) in tables.nrf_frame_rows.iter().enumerate() {
        tokens.push(nanl_bin::NanlToken::new(
            nanl_bin::NanlTokenClass::NrfFrame,
            row.tile_dict_ref as u16,
            lane_value(idx),
        ));
    }
    for (idx, row) in tables.drf_state_rows.iter().enumerate() {
        tokens.push(nanl_bin::NanlToken::new(
            nanl_bin::NanlTokenClass::DrfChannel,
            row.channel_dict_ref as u16,
            lane_value(idx),
        ));
    }
    tokens
}

fn tokens_from_tables(tables: &CseHdComponentTables) -> Vec<nanl_bin::NanlToken> {
    nanl_tokens_from_tables(tables)
}

fn reference_corpus_from_cse(
    source: &CseSourceDocument,
    compilation: &CseCompilation,
    nanl_bytes: &[u8],
) -> PcResult<ReferenceCorpus> {
    let mut state_ids = BTreeMap::new();
    let mut next_state_id = 1u64;
    for row in &compilation.tables.entity_rows {
        state_ids.insert(row.entity_id.clone(), next_state_id);
        next_state_id += 1;
    }
    for row in &compilation.tables.event_rows {
        state_ids.insert(row.event_id.clone(), next_state_id);
        next_state_id += 1;
    }

    let time = source
        .timebases
        .first()
        .ok_or_else(|| PcError::Timebase("CSE source has no canonical timebase".to_string()))?;
    let time_unit = source
        .unit_registry
        .iter()
        .find(|unit| unit.id == time.unit_ref)
        .map(|unit| unit.unit.clone())
        .unwrap_or_else(|| "dimensionless".to_string());
    if source.provenance.is_empty() {
        return Err(PcError::Provenance(
            "CSE source has no provenance".to_string(),
        ));
    }
    let provenance_ids = source
        .provenance
        .iter()
        .enumerate()
        .map(|(idx, provenance)| (provenance.id.as_str(), (idx + 1) as u64))
        .collect::<BTreeMap<_, _>>();
    let states = compilation
        .tables
        .entity_rows
        .iter()
        .map(|row| ReferenceState {
            state_id: state_ids.get(&row.entity_id).copied().unwrap_or(1),
            state_kind: match row.entity_kind.as_str() {
                "MOLECULE" => "MOLECULAR_STATE".to_string(),
                "PROTEIN" => "GRAPH_STATE".to_string(),
                "WATER" => "EVENT_STREAM".to_string(),
                _ => "DEBUG_VIEW".to_string(),
            },
            dtype: "F64".to_string(),
            shape: vec![2],
            payload_f64: vec![1.0, row.entity_dict_ref as f64],
            time_ref: 1,
            provenance_ref: provenance_ids
                .get(row.provenance_ref.as_str())
                .copied()
                .unwrap_or(1),
            unit: "dimensionless".to_string(),
        })
        .collect::<Vec<_>>();
    let events = compilation
        .tables
        .event_rows
        .iter()
        .enumerate()
        .map(|(idx, row)| ReferenceEvent {
            event_id: (idx + 1) as u64,
            event_kind: row.event_kind.clone(),
            time_ref: 1,
            source_state: state_ids.get(&row.source_ref).copied().unwrap_or(1),
            target_state: state_ids.get(&row.target_ref).copied().unwrap_or(1),
            value: row.order_index as f64,
            unit: "dimensionless".to_string(),
        })
        .collect::<Vec<_>>();
    let thermo = compilation
        .tables
        .thermo_rows
        .iter()
        .enumerate()
        .map(|(idx, row)| ReferenceThermo {
            thermo_id: (idx + 1) as u64,
            state_ref: state_ids.get(&row.target_ref).copied().unwrap_or(1),
            value: row.quantity_dict_ref as f64,
            unit: source
                .unit_registry
                .iter()
                .find(|unit| unit.id == row.unit_ref)
                .map(|unit| unit.unit.clone())
                .unwrap_or_else(|| row.unit_ref.clone()),
            signal_kind: row.quantity_kind.clone(),
        })
        .collect::<Vec<_>>();
    let nanl_tokens = nanl_bytes
        .iter()
        .take(4096)
        .map(|byte| byte & 0b11)
        .collect::<Vec<_>>();

    Ok(ReferenceCorpus {
        corpus_id: format!("CSE_HD_{}", source.state_id),
        synthetic: source.synthetic,
        generator_version: "prism-cse-hd-v0.1".to_string(),
        seed: 0,
        states,
        timebase: vec![ReferenceTime {
            time_id: 1,
            unit: time_unit,
            start: time.frame_start as f64,
            step: 1.0,
            count: (time.frame_end.saturating_sub(time.frame_start)).max(1),
            timebase_hash: time.timebase_hash.clone(),
        }],
        nanl_tokens,
        events,
        thermo,
        provenance: source
            .provenance
            .iter()
            .enumerate()
            .map(|(idx, provenance)| ReferenceProvenance {
                provenance_id: (idx + 1) as u64,
                label: provenance.id.clone(),
                synthetic: provenance.synthetic,
                reason: provenance.source_uri.clone(),
            })
            .collect(),
        expected_output_hash: None,
        expected_replay_hash: None,
    })
}

fn write_raw_sidecar(
    artifact_kind: &str,
    path: &Path,
    bytes: &[u8],
    producer: &str,
    validator: &str,
) -> PcResult<CseHdSidecarArtifact> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, bytes)?;
    Ok(CseHdSidecarArtifact {
        artifact_kind: artifact_kind.to_string(),
        path: path.display().to_string(),
        bytes: bytes.len() as u64,
        hash_blake3: blake3::hash(bytes).to_hex().to_string(),
        producer: producer.to_string(),
        validator: validator.to_string(),
        status: "PASS".to_string(),
    })
}

fn write_labeled_binary_sidecar(
    artifact_kind: &str,
    path: &Path,
    magic: &[u8],
    payload: &[u8],
    producer: &str,
    validator: &str,
) -> PcResult<CseHdSidecarArtifact> {
    let mut bytes = Vec::with_capacity(magic.len() + 8 + payload.len() + 32);
    bytes.extend_from_slice(magic);
    bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&blake3::hash(payload).as_bytes()[..]);
    bytes.extend_from_slice(payload);
    write_raw_sidecar(artifact_kind, path, &bytes, producer, validator)
}

fn validate_molecule(
    molecule: &CseHdMoleculeSource,
    provenance_ids: &BTreeSet<&str>,
) -> Result<(), CseError> {
    require_nonempty("molecule.id", &molecule.id)?;
    require_nonempty("molecule.topology_hash", &molecule.topology_hash)?;
    require_ref(
        "molecule.provenance_ref",
        &molecule.provenance_ref,
        provenance_ids,
    )?;
    if molecule.atoms.is_empty() {
        return Err(CseError::Type(format!(
            "molecule {} must include typed atoms",
            molecule.id
        )));
    }
    ensure_unique(
        molecule.atoms.iter().map(|atom| atom.id.as_str()),
        "molecule.atom",
    )?;
    ensure_unique(
        molecule.bonds.iter().map(|bond| bond.id.as_str()),
        "molecule.bond",
    )?;
    let atom_ids = molecule
        .atoms
        .iter()
        .map(|atom| atom.id.as_str())
        .collect::<BTreeSet<_>>();
    for atom in &molecule.atoms {
        require_nonempty("atom.id", &atom.id)?;
        require_nonempty("atom.element", &atom.element)?;
        require_nonempty("atom.atom_type", &atom.atom_type)?;
    }
    for bond in &molecule.bonds {
        require_nonempty("bond.id", &bond.id)?;
        require_ref("bond.source_atom", &bond.source_atom, &atom_ids)?;
        require_ref("bond.target_atom", &bond.target_atom, &atom_ids)?;
        if bond.order == 0 {
            return Err(CseError::Science(format!(
                "bond {} order must be positive",
                bond.id
            )));
        }
    }
    Ok(())
}

fn validate_protein(
    protein: &CseHdProteinSource,
    provenance_ids: &BTreeSet<&str>,
) -> Result<(), CseError> {
    require_nonempty("protein.id", &protein.id)?;
    require_nonempty("protein.topology_hash", &protein.topology_hash)?;
    require_ref(
        "protein.provenance_ref",
        &protein.provenance_ref,
        provenance_ids,
    )?;
    if protein.chains.is_empty() {
        return Err(CseError::Type(format!(
            "protein {} must include typed chains",
            protein.id
        )));
    }
    if protein.atom_sites.is_empty() {
        return Err(CseError::Type(format!(
            "protein {} must include atom-site linkage",
            protein.id
        )));
    }
    ensure_unique(
        protein.chains.iter().map(|chain| chain.id.as_str()),
        "protein.chain",
    )?;
    ensure_unique(
        protein.atom_sites.iter().map(|site| site.id.as_str()),
        "protein.atom_site",
    )?;
    let chain_ids = protein
        .chains
        .iter()
        .map(|chain| chain.id.as_str())
        .collect::<BTreeSet<_>>();
    let atom_site_ids = protein
        .atom_sites
        .iter()
        .map(|site| site.id.as_str())
        .collect::<BTreeSet<_>>();
    let residue_ids = protein
        .chains
        .iter()
        .flat_map(|chain| chain.residues.iter().map(|residue| residue.id.as_str()))
        .collect::<BTreeSet<_>>();
    for chain in &protein.chains {
        require_nonempty("chain.id", &chain.id)?;
        if chain.residues.is_empty() {
            return Err(CseError::Type(format!(
                "protein chain {} must include residues",
                chain.id
            )));
        }
        ensure_unique(
            chain.residues.iter().map(|residue| residue.id.as_str()),
            "chain.residue",
        )?;
        for residue in &chain.residues {
            require_nonempty("residue.id", &residue.id)?;
            require_nonempty("residue.name", &residue.name)?;
            if residue.atom_site_ids.is_empty() {
                return Err(CseError::Type(format!(
                    "residue {} must bind atom-site ids",
                    residue.id
                )));
            }
            for site_id in &residue.atom_site_ids {
                require_ref("residue.atom_site_id", site_id, &atom_site_ids)?;
            }
        }
    }
    for site in &protein.atom_sites {
        require_nonempty("atom_site.id", &site.id)?;
        require_nonempty("atom_site.atom_name", &site.atom_name)?;
        require_nonempty("atom_site.atom_type", &site.atom_type)?;
        require_ref("atom_site.chain_ref", &site.chain_ref, &chain_ids)?;
        require_ref("atom_site.residue_ref", &site.residue_ref, &residue_ids)?;
    }
    Ok(())
}

fn validate_water_site(
    water: &CseHdWaterSiteSource,
    timebase_ids: &BTreeSet<&str>,
    provenance_ids: &BTreeSet<&str>,
) -> Result<(), CseError> {
    require_nonempty("water_site.id", &water.id)?;
    require_nonempty("water_site.gate_kind", &water.gate_kind)?;
    if water.event_bound {
        let time_ref = water.time_ref.as_deref().ok_or_else(|| {
            CseError::Timebase(format!(
                "event-bound water {} must bind canonical timebase",
                water.id
            ))
        })?;
        require_ref("water_site.time_ref", time_ref, timebase_ids)?;
        let provenance_ref = water.provenance_ref.as_deref().ok_or_else(|| {
            CseError::Provenance(format!(
                "event-bound water {} must bind provenance",
                water.id
            ))
        })?;
        require_ref("water_site.provenance_ref", provenance_ref, provenance_ids)?;
    }
    Ok(())
}

fn build_dictionaries(source: &CseSourceDocument) -> CseDictionaries {
    let mut symbols = CseDictionarySymbols::default();
    for unit in &source.unit_registry {
        symbols.insert(CseDictionaryKind::Unit, unit.unit.clone());
        symbols.insert(
            CseDictionaryKind::ThermoQuantity,
            unit.quantity_kind.clone(),
        );
    }
    for prov in &source.provenance {
        symbols.insert(CseDictionaryKind::Provenance, prov.id.clone());
    }
    for molecule in &source.molecules {
        symbols.insert(CseDictionaryKind::Entity, molecule.id.clone());
        for atom in &molecule.atoms {
            symbols.insert(CseDictionaryKind::AtomType, atom.atom_type.clone());
        }
    }
    for protein in &source.proteins {
        symbols.insert(CseDictionaryKind::Entity, protein.id.clone());
        for site in &protein.atom_sites {
            symbols.insert(CseDictionaryKind::AtomType, site.atom_type.clone());
        }
        for chain in &protein.chains {
            for residue in &chain.residues {
                symbols.insert(CseDictionaryKind::Residue, residue.name.clone());
            }
        }
    }
    for water in &source.water_sites {
        symbols.insert(CseDictionaryKind::Entity, water.id.clone());
        symbols.insert(CseDictionaryKind::WaterGate, water.gate_kind.clone());
    }
    for variant in &source.variants {
        symbols.insert(CseDictionaryKind::Variant, variant.id.clone());
    }
    for event in &source.events {
        symbols.insert(CseDictionaryKind::EventKind, event.kind.clone());
        symbols.insert(CseDictionaryKind::EventMotif, event.motif.clone());
    }
    for uncertainty in &source.uncertainties {
        symbols.insert(
            CseDictionaryKind::UncertaintyModel,
            uncertainty.model.clone(),
        );
    }
    for thermo in &source.thermo {
        symbols.insert(
            CseDictionaryKind::ThermoQuantity,
            thermo.quantity_kind.clone(),
        );
    }
    for teacher in &source.teacher_outputs {
        symbols.insert(CseDictionaryKind::TeacherLabel, teacher.label.clone());
    }
    for trace in &source.gflownet_traces {
        symbols.insert(CseDictionaryKind::GFlowNetAction, trace.action.clone());
    }
    for symbol in &source.nanl_symbols {
        symbols.insert(CseDictionaryKind::NanlSymbol, symbol.clone());
    }
    for frame in &source.nrf_frames {
        symbols.insert(CseDictionaryKind::NrfTile, frame.tile_kind.clone());
    }
    for drf in &source.drf_states {
        symbols.insert(CseDictionaryKind::DrfChannel, drf.channel_kind.clone());
    }
    symbols.finish()
}

fn build_component_tables(
    source: &CseSourceDocument,
    dictionaries: &CseDictionaries,
) -> Result<CseHdComponentTables, CseError> {
    let mut tables = CseHdComponentTables {
        schema_rows: vec![CseHdSchemaRow {
            schema_version: source.schema_version.clone(),
            state_id: source.state_id.clone(),
            classification: source.classification.clone(),
            synthetic: source.synthetic,
        }],
        unit_rows: Vec::new(),
        entity_rows: Vec::new(),
        atom_rows: Vec::new(),
        bond_rows: Vec::new(),
        protein_chain_rows: Vec::new(),
        residue_rows: Vec::new(),
        atom_site_rows: Vec::new(),
        water_site_rows: Vec::new(),
        timebase_rows: Vec::new(),
        variant_rows: Vec::new(),
        event_rows: Vec::new(),
        uncertainty_rows: Vec::new(),
        thermo_rows: Vec::new(),
        teacher_output_rows: Vec::new(),
        gflownet_trace_rows: Vec::new(),
        nrf_frame_rows: Vec::new(),
        drf_state_rows: Vec::new(),
        payload_blob_rows: Vec::new(),
        merkle_rows: Vec::new(),
        provenance_rows: Vec::new(),
        seal_rows: Vec::new(),
        nanl_symbol_rows: Vec::new(),
    };

    for unit in &source.unit_registry {
        tables.unit_rows.push(CseHdUnitRow {
            unit_id: unit.id.clone(),
            quantity_kind: unit.quantity_kind.clone(),
            unit: unit.unit.clone(),
            unit_dict_ref: dict_id(dictionaries, CseDictionaryKind::Unit, &unit.unit)?,
            quantity_dict_ref: dict_id(
                dictionaries,
                CseDictionaryKind::ThermoQuantity,
                &unit.quantity_kind,
            )?,
        });
    }
    for prov in &source.provenance {
        tables.provenance_rows.push(CseHdProvenanceRow {
            provenance_id: prov.id.clone(),
            source_uri: prov.source_uri.clone(),
            source_hash: prov.source_hash.clone(),
            classification: prov.classification.clone(),
            synthetic: prov.synthetic,
            provenance_dict_ref: dict_id(dictionaries, CseDictionaryKind::Provenance, &prov.id)?,
        });
    }
    for time in &source.timebases {
        tables.timebase_rows.push(CseHdTimebaseRow {
            timebase_id: time.id.clone(),
            timebase_hash: time.timebase_hash.clone(),
            unit_ref: time.unit_ref.clone(),
            frame_start: time.frame_start,
            frame_end: time.frame_end,
        });
    }
    for molecule in &source.molecules {
        tables.entity_rows.push(CseHdEntityRow {
            entity_id: molecule.id.clone(),
            entity_kind: "MOLECULE".to_string(),
            topology_hash: molecule.topology_hash.clone(),
            provenance_ref: molecule.provenance_ref.clone(),
            entity_dict_ref: dict_id(dictionaries, CseDictionaryKind::Entity, &molecule.id)?,
        });
        for atom in &molecule.atoms {
            tables.atom_rows.push(CseHdAtomRow {
                molecule_ref: molecule.id.clone(),
                atom_id: atom.id.clone(),
                element: atom.element.clone(),
                atom_type: atom.atom_type.clone(),
                atom_type_dict_ref: dict_id(
                    dictionaries,
                    CseDictionaryKind::AtomType,
                    &atom.atom_type,
                )?,
            });
        }
        for bond in &molecule.bonds {
            tables.bond_rows.push(CseHdBondRow {
                molecule_ref: molecule.id.clone(),
                bond_id: bond.id.clone(),
                source_atom: bond.source_atom.clone(),
                target_atom: bond.target_atom.clone(),
                order: bond.order,
            });
        }
    }
    for protein in &source.proteins {
        tables.entity_rows.push(CseHdEntityRow {
            entity_id: protein.id.clone(),
            entity_kind: "PROTEIN".to_string(),
            topology_hash: protein.topology_hash.clone(),
            provenance_ref: protein.provenance_ref.clone(),
            entity_dict_ref: dict_id(dictionaries, CseDictionaryKind::Entity, &protein.id)?,
        });
        for chain in &protein.chains {
            tables.protein_chain_rows.push(CseHdProteinChainRow {
                protein_ref: protein.id.clone(),
                chain_id: chain.id.clone(),
            });
            for residue in &chain.residues {
                tables.residue_rows.push(CseHdResidueRow {
                    protein_ref: protein.id.clone(),
                    chain_ref: chain.id.clone(),
                    residue_id: residue.id.clone(),
                    name: residue.name.clone(),
                    residue_dict_ref: dict_id(
                        dictionaries,
                        CseDictionaryKind::Residue,
                        &residue.name,
                    )?,
                    sequence_index: residue.sequence_index,
                    atom_site_ids: residue.atom_site_ids.clone(),
                });
            }
        }
        for site in &protein.atom_sites {
            tables.atom_site_rows.push(CseHdAtomSiteRow {
                protein_ref: protein.id.clone(),
                site_id: site.id.clone(),
                atom_name: site.atom_name.clone(),
                atom_type: site.atom_type.clone(),
                atom_type_dict_ref: dict_id(
                    dictionaries,
                    CseDictionaryKind::AtomType,
                    &site.atom_type,
                )?,
                chain_ref: site.chain_ref.clone(),
                residue_ref: site.residue_ref.clone(),
            });
        }
    }
    for water in &source.water_sites {
        tables.entity_rows.push(CseHdEntityRow {
            entity_id: water.id.clone(),
            entity_kind: "WATER".to_string(),
            topology_hash: "WATER_SITE_TOPOLOGY_HASH_NOT_RAW_JSON".to_string(),
            provenance_ref: water.provenance_ref.clone().unwrap_or_default(),
            entity_dict_ref: dict_id(dictionaries, CseDictionaryKind::Entity, &water.id)?,
        });
        tables.water_site_rows.push(CseHdWaterSiteRow {
            water_id: water.id.clone(),
            gate_kind: water.gate_kind.clone(),
            water_gate_dict_ref: dict_id(
                dictionaries,
                CseDictionaryKind::WaterGate,
                &water.gate_kind,
            )?,
            event_bound: water.event_bound,
            time_ref: water.time_ref.clone(),
            provenance_ref: water.provenance_ref.clone(),
        });
    }
    for variant in &source.variants {
        tables.variant_rows.push(CseHdVariantRow {
            variant_id: variant.id.clone(),
            variant_dict_ref: dict_id(dictionaries, CseDictionaryKind::Variant, &variant.id)?,
            reference_state_ref: variant.reference_state_ref.clone(),
            alternate_state_ref: variant.alternate_state_ref.clone(),
            delta_hash: variant.delta_hash.clone(),
            provenance_ref: variant.provenance_ref.clone(),
        });
    }
    for event in &source.events {
        tables.event_rows.push(CseHdEventRow {
            event_id: event.id.clone(),
            event_kind: event.kind.clone(),
            event_kind_dict_ref: dict_id(dictionaries, CseDictionaryKind::EventKind, &event.kind)?,
            source_ref: event.source_ref.clone(),
            target_ref: event.target_ref.clone(),
            time_ref: event.time_ref.clone(),
            order_index: event.order_index,
            motif: event.motif.clone(),
            event_motif_dict_ref: dict_id(
                dictionaries,
                CseDictionaryKind::EventMotif,
                &event.motif,
            )?,
            provenance_ref: event.provenance_ref.clone(),
        });
    }
    for uncertainty in &source.uncertainties {
        tables.uncertainty_rows.push(CseHdUncertaintyRow {
            uncertainty_id: uncertainty.id.clone(),
            target_ref: uncertainty.target_ref.clone(),
            model: uncertainty.model.clone(),
            uncertainty_model_dict_ref: dict_id(
                dictionaries,
                CseDictionaryKind::UncertaintyModel,
                &uncertainty.model,
            )?,
            calibration_hash: uncertainty.calibration_hash.clone(),
            provenance_ref: uncertainty.provenance_ref.clone(),
        });
    }
    for thermo in &source.thermo {
        let unit = source
            .unit_registry
            .iter()
            .find(|unit| unit.id == thermo.unit_ref)
            .ok_or_else(|| CseError::Type(format!("unit {} missing", thermo.unit_ref)))?;
        tables.thermo_rows.push(CseHdThermoRow {
            thermo_id: thermo.id.clone(),
            target_ref: thermo.target_ref.clone(),
            quantity_kind: thermo.quantity_kind.clone(),
            quantity_dict_ref: dict_id(
                dictionaries,
                CseDictionaryKind::ThermoQuantity,
                &thermo.quantity_kind,
            )?,
            unit_ref: thermo.unit_ref.clone(),
            unit_dict_ref: dict_id(dictionaries, CseDictionaryKind::Unit, &unit.unit)?,
            value_payload_ref: thermo.value_payload_ref.clone(),
            uncertainty_ref: thermo.uncertainty_ref.clone(),
            provenance_ref: thermo.provenance_ref.clone(),
        });
    }
    for teacher in &source.teacher_outputs {
        tables.teacher_output_rows.push(CseHdTeacherOutputRow {
            teacher_output_id: teacher.id.clone(),
            target_ref: teacher.target_ref.clone(),
            model_ref: teacher.model_ref.clone(),
            label: teacher.label.clone(),
            label_dict_ref: dict_id(
                dictionaries,
                CseDictionaryKind::TeacherLabel,
                &teacher.label,
            )?,
            confidence: teacher.confidence,
            airlock_state: teacher.airlock_state.clone(),
            provenance_ref: teacher.provenance_ref.clone(),
        });
    }
    for trace in &source.gflownet_traces {
        tables.gflownet_trace_rows.push(CseHdGFlowNetTraceRow {
            trace_id: trace.id.clone(),
            state_ref: trace.state_ref.clone(),
            action: trace.action.clone(),
            action_dict_ref: dict_id(
                dictionaries,
                CseDictionaryKind::GFlowNetAction,
                &trace.action,
            )?,
            reward_payload_ref: trace.reward_payload_ref.clone(),
            lineage_hash: trace.lineage_hash.clone(),
            provenance_ref: trace.provenance_ref.clone(),
        });
    }
    for (idx, symbol) in source.nanl_symbols.iter().enumerate() {
        tables.nanl_symbol_rows.push(CseHdNanlSymbolRow {
            symbol_id: idx as u32,
            symbol: symbol.clone(),
        });
    }
    for frame in &source.nrf_frames {
        tables.nrf_frame_rows.push(CseHdNrfFrameRow {
            frame_id: frame.id.clone(),
            timebase_ref: frame.timebase_ref.clone(),
            tile_kind: frame.tile_kind.clone(),
            tile_dict_ref: dict_id(dictionaries, CseDictionaryKind::NrfTile, &frame.tile_kind)?,
            source_event_refs: frame.source_event_refs.clone(),
        });
    }
    for drf in &source.drf_states {
        tables.drf_state_rows.push(CseHdDrfStateRow {
            drf_state_id: drf.id.clone(),
            source_nrf_ref: drf.source_nrf_ref.clone(),
            source_cse_ref: drf.source_cse_ref.clone(),
            channel_kind: drf.channel_kind.clone(),
            channel_dict_ref: dict_id(
                dictionaries,
                CseDictionaryKind::DrfChannel,
                &drf.channel_kind,
            )?,
            tensor_shape: drf.tensor_shape.clone(),
            provenance_ref: drf.provenance_ref.clone(),
            canonical_or_derived: drf.canonical_or_derived.clone(),
        });
    }
    let mut payload_ids = BTreeSet::new();
    for thermo in &source.thermo {
        if payload_ids.insert(thermo.value_payload_ref.clone()) {
            tables.payload_blob_rows.push(CseHdPayloadBlobRow {
                payload_id: thermo.value_payload_ref.clone(),
                payload_kind: "THERMO_VALUE_PAYLOAD".to_string(),
                payload_len: thermo.value_payload_ref.len() as u64,
                payload_hash: blake3::hash(thermo.value_payload_ref.as_bytes())
                    .to_hex()
                    .to_string(),
            });
        }
    }
    for trace in &source.gflownet_traces {
        if payload_ids.insert(trace.reward_payload_ref.clone()) {
            tables.payload_blob_rows.push(CseHdPayloadBlobRow {
                payload_id: trace.reward_payload_ref.clone(),
                payload_kind: "GFLOWNET_REWARD_PAYLOAD".to_string(),
                payload_len: trace.reward_payload_ref.len() as u64,
                payload_hash: blake3::hash(trace.reward_payload_ref.as_bytes())
                    .to_hex()
                    .to_string(),
            });
        }
    }
    let merkle_material = serde_json::to_vec(&(
        &tables.schema_rows,
        &tables.entity_rows,
        &tables.event_rows,
        &tables.nrf_frame_rows,
        &tables.drf_state_rows,
    ))?;
    let merkle_root = blake3::hash(&merkle_material).to_hex().to_string();
    tables.merkle_rows.push(CseHdMerkleRow {
        merkle_id: format!("merkle:{}", source.state_id),
        root_hash: merkle_root.clone(),
        leaf_count: (tables.entity_rows.len()
            + tables.event_rows.len()
            + tables.nrf_frame_rows.len()
            + tables.drf_state_rows.len()) as u64,
    });
    let provenance_material = serde_json::to_vec(&tables.provenance_rows)?;
    let provenance_hash = blake3::hash(&provenance_material).to_hex().to_string();
    let mut seal_hasher = blake3::Hasher::new();
    seal_hasher.update(source.state_id.as_bytes());
    seal_hasher.update(merkle_root.as_bytes());
    seal_hasher.update(provenance_hash.as_bytes());
    let current_seal_hash = seal_hasher.finalize().to_hex().to_string();
    tables.seal_rows.push(CseHdSealRow {
        seal_id: format!("seal:{}", source.state_id),
        previous_seal_hash: "0000000000000000000000000000000000000000000000000000000000000000"
            .to_string(),
        current_seal_hash,
    });
    Ok(tables)
}

fn canonicalize_source(source: &CseSourceDocument) -> CseSourceDocument {
    let mut out = source.clone();
    out.unit_registry.sort_by(|a, b| a.id.cmp(&b.id));
    out.provenance.sort_by(|a, b| a.id.cmp(&b.id));
    out.timebases.sort_by(|a, b| a.id.cmp(&b.id));
    out.molecules.sort_by(|a, b| a.id.cmp(&b.id));
    for molecule in &mut out.molecules {
        molecule.atoms.sort_by(|a, b| a.id.cmp(&b.id));
        molecule.bonds.sort_by(|a, b| a.id.cmp(&b.id));
    }
    out.proteins.sort_by(|a, b| a.id.cmp(&b.id));
    for protein in &mut out.proteins {
        protein.chains.sort_by(|a, b| a.id.cmp(&b.id));
        protein.atom_sites.sort_by(|a, b| a.id.cmp(&b.id));
        for chain in &mut protein.chains {
            chain.residues.sort_by(|a, b| {
                a.sequence_index
                    .cmp(&b.sequence_index)
                    .then(a.id.cmp(&b.id))
            });
            for residue in &mut chain.residues {
                residue.atom_site_ids.sort();
            }
        }
    }
    out.water_sites.sort_by(|a, b| a.id.cmp(&b.id));
    out.variants.sort_by(|a, b| a.id.cmp(&b.id));
    out.events
        .sort_by(|a, b| a.order_index.cmp(&b.order_index).then(a.id.cmp(&b.id)));
    out.uncertainties.sort_by(|a, b| a.id.cmp(&b.id));
    out.thermo.sort_by(|a, b| a.id.cmp(&b.id));
    out.teacher_outputs.sort_by(|a, b| a.id.cmp(&b.id));
    out.gflownet_traces.sort_by(|a, b| a.id.cmp(&b.id));
    out.nanl_symbols.sort();
    out.nrf_frames.sort_by(|a, b| a.id.cmp(&b.id));
    for frame in &mut out.nrf_frames {
        frame.source_event_refs.sort();
    }
    out.drf_states.sort_by(|a, b| a.id.cmp(&b.id));
    out
}

fn reject_untyped_jsonish_domain_objects(value: &Value, path: &str) -> Result<(), CseError> {
    const BANNED_KEYS: &[&str] = &[
        "domain_objects",
        "json_objects",
        "untyped_objects",
        "object_blob",
        "raw_json",
        "json_blob",
        "payload_json",
    ];
    match value {
        Value::Object(map) => {
            for banned in BANNED_KEYS {
                if map.contains_key(*banned) {
                    return Err(CseError::Type(format!(
                        "untyped JSON-ish domain object field {path}.{banned} is forbidden"
                    )));
                }
            }
            if map.contains_key("type") && map.contains_key("payload") {
                return Err(CseError::Type(format!(
                    "generic {{type,payload}} object at {path} is forbidden on the CSE execution path"
                )));
            }
            for (key, child) in map {
                reject_untyped_jsonish_domain_objects(child, &format!("{path}.{key}"))?;
            }
        }
        Value::Array(items) => {
            for (idx, child) in items.iter().enumerate() {
                reject_untyped_jsonish_domain_objects(child, &format!("{path}[{idx}]"))?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn ensure_unique<'a>(ids: impl Iterator<Item = &'a str>, label: &str) -> Result<(), CseError> {
    let mut seen = BTreeSet::new();
    for id in ids {
        require_nonempty(label, id)?;
        if !seen.insert(id) {
            return Err(CseError::Type(format!("duplicate {label} id {id}")));
        }
    }
    Ok(())
}

fn require_nonempty(label: &str, value: &str) -> Result<(), CseError> {
    if value.trim().is_empty() {
        return Err(CseError::Type(format!("{label} must not be empty")));
    }
    Ok(())
}

fn require_ref(label: &str, value: &str, refs: &BTreeSet<&str>) -> Result<(), CseError> {
    require_nonempty(label, value)?;
    if !refs.contains(value) {
        return Err(CseError::Type(format!(
            "{label} unresolved reference {value}"
        )));
    }
    Ok(())
}

fn dict_id(
    dictionaries: &CseDictionaries,
    kind: CseDictionaryKind,
    symbol: &str,
) -> Result<u32, CseError> {
    dictionaries
        .id_for(kind, symbol)
        .ok_or_else(|| CseError::Type(format!("dictionary {kind:?} missing symbol {symbol}")))
}

fn canonical_hash<T: Serialize>(value: &T) -> Result<String, CseError> {
    let bytes = serde_json::to_vec(value)?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}
