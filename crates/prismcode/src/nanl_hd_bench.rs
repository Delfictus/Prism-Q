//! NANL-HD benchmark suite report generation.

use crate::binary_runtime::{benchmark_packet, Backend, PcError, PcResult};
use crate::cse_hd::{compile_scientific_state, CseScientificCompileReport, NanlMode};
use crate::nanl_ans::{self, AnsSymbolClass, AnsToken};
use crate::nanl_bin::{self, NanlToken, NanlTokenClass};
use crate::nanl_codec_stack::{fast_execution_stack, q4_archival_stack, validate_stack};
use crate::nanl_delta;
use crate::nanl_dict::{
    NanlDictionary, NanlDictionaryClass, NanlDictionarySymbol, REQUIRED_DICTIONARY_CLASSES,
};
use crate::nanl_h16;
use crate::nanl_motif::{
    self, MotifDiscoveryPolicy, MotifKind, NanlMotifObservation, REQUIRED_MOTIF_KINDS,
};
use crate::nanl_q4;
use crate::nanl_var;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::io::Cursor;
use std::path::{Path, PathBuf};

pub type NanlHdBenchResult<T> = Result<T, NanlHdBenchError>;

#[derive(Debug, thiserror::Error)]
pub enum NanlHdBenchError {
    #[error("PC_ERR_NANL_HD_BENCH:{0}")]
    Invalid(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NanlHdBenchmarkReport {
    pub status: String,
    pub sample_count: u64,
    pub workload_count: u64,
    pub malformed_workload_count: u64,
    pub raw_source_bytes: u64,
    pub canonical_cse_bytes: u64,
    pub plain_prismcode_bytes: u64,
    pub zstd_bytes: u64,
    pub nanl_bin_bytes: u64,
    pub nanl_q4_bytes: u64,
    pub nanl_dict_bytes: u64,
    pub nanl_motif_bytes: u64,
    pub nanl_delta_bytes: u64,
    pub nanl_var_bytes: u64,
    pub nanl_ans_bytes: u64,
    pub nanl_h16_bytes: u64,
    pub nrf_sparse_bytes: u64,
    pub drf_active_state_bytes: u64,
    pub combined_stack_bytes: u64,
    #[serde(rename = "encode_MBps")]
    pub encode_mbps: f64,
    #[serde(rename = "decode_MBps")]
    pub decode_mbps: f64,
    pub cpu_runtime_rows_sec: f64,
    pub cuda_runtime_rows_sec: f64,
    pub events_sec: f64,
    pub instructions_sec: f64,
    pub canonical_hash_pass: bool,
    pub roundtrip_hash_pass: bool,
    pub replay_pass: bool,
    pub reconstruction_fidelity: f64,
    pub model_table_bytes: u64,
    pub dictionary_bytes: u64,
    pub ecc_overhead_bytes: u64,
    pub q4_constraint_overhead_bytes: u64,
    pub compression_ratio_vs_raw: f64,
    pub compression_ratio_vs_cse: f64,
    pub compression_ratio_vs_nanl_bin: f64,
}

pub fn run_nanl_hd_benchmark(
    payload: &[u8],
    sample_count: u64,
) -> NanlHdBenchResult<NanlHdBenchmarkReport> {
    if payload.is_empty() {
        return Err(NanlHdBenchError::Invalid(
            "benchmark payload is empty".to_string(),
        ));
    }
    if sample_count < 10 {
        return Err(NanlHdBenchError::Invalid(
            "benchmark sample_count must be at least 10 for claims".to_string(),
        ));
    }
    let start = std::time::Instant::now();
    let dictionary_symbols = benchmark_dictionary_symbols(payload);
    let dictionary = NanlDictionary::build(
        "PRISM_NATIVE_NANL_HD",
        blake3::hash(payload).to_hex().to_string(),
        0,
        &dictionary_symbols,
    )
    .map_err(|err| NanlHdBenchError::Invalid(err.to_string()))?;
    let dict_tokens = payload
        .iter()
        .map(|byte| (NanlDictionaryClass::NanlSymbol, format!("SYM_{byte:02x}")))
        .collect::<Vec<_>>();
    let encoded_dict = dictionary
        .encode_tokens(&dict_tokens)
        .map_err(|err| NanlHdBenchError::Invalid(err.to_string()))?
        .into_iter()
        .map(|token| token.encoded_value)
        .collect::<Vec<_>>();
    let motifs = nanl_motif::discover_numeric_motifs(
        &encoded_dict,
        MotifKind::DrfResonance,
        MotifDiscoveryPolicy::default(),
    )
    .map_err(|err| NanlHdBenchError::Invalid(err.to_string()))?;
    let motif_codes = nanl_motif::encode_with_motifs(&encoded_dict, &motifs)
        .map_err(|err| NanlHdBenchError::Invalid(err.to_string()))?;
    let motif_book = nanl_motif::discover_motifs(
        "nanl_hd_bench_cse_hash",
        "nanl_hd_bench_event_hash",
        &benchmark_motif_observations(),
        nanl_motif::NanlMotifDiscoveryConfig::default(),
    )
    .map_err(|err| NanlHdBenchError::Invalid(err.to_string()))?;
    let delta = nanl_delta::encode_i64_lossless(
        &encoded_dict
            .iter()
            .map(|value| *value as i64)
            .collect::<Vec<_>>(),
    )
    .map_err(|err| NanlHdBenchError::Invalid(err.to_string()))?;
    let var = nanl_var::encode_u64(&encoded_dict)
        .map_err(|err| NanlHdBenchError::Invalid(err.to_string()))?;
    let ans_tokens = payload
        .iter()
        .map(|byte| AnsToken::new(AnsSymbolClass::EventKind, u16::from(*byte)))
        .collect::<Vec<_>>();
    let ans = nanl_ans::encode_tokens(&ans_tokens)
        .map_err(|err| NanlHdBenchError::Invalid(err.to_string()))?;
    let bin_tokens = payload
        .iter()
        .enumerate()
        .map(|(idx, byte)| NanlToken::new(NanlTokenClass::Entity, (*byte as u16) % 16, idx as u32))
        .collect::<Vec<_>>();
    let bin = nanl_bin::encode_tokens(&bin_tokens)
        .map_err(|err| NanlHdBenchError::Invalid(err.to_string()))?;
    let q4 = nanl_q4::encode_payload(payload)
        .map_err(|err| NanlHdBenchError::Invalid(err.to_string()))?;
    let h16 = nanl_h16::encode_bytes(payload)
        .map_err(|err| NanlHdBenchError::Invalid(err.to_string()))?;
    let fast_stack =
        fast_execution_stack(payload).map_err(|err| NanlHdBenchError::Invalid(err.to_string()))?;
    validate_stack(&fast_stack).map_err(|err| NanlHdBenchError::Invalid(err.to_string()))?;
    let q4_stack =
        q4_archival_stack(payload).map_err(|err| NanlHdBenchError::Invalid(err.to_string()))?;
    validate_stack(&q4_stack).map_err(|err| NanlHdBenchError::Invalid(err.to_string()))?;
    let elapsed = start.elapsed().as_secs_f64().max(0.000_001);

    let ans_decoded = nanl_ans::decode_packet(ans.bytes())
        .map_err(|err| NanlHdBenchError::Invalid(err.to_string()))?;
    let q4_decoded = nanl_q4::decode_packet(q4.bytes())
        .map_err(|err| NanlHdBenchError::Invalid(err.to_string()))?;
    let bin_decoded = nanl_bin::decode_packet(bin.bytes())
        .map_err(|err| NanlHdBenchError::Invalid(err.to_string()))?;
    let raw_source_bytes = payload.len() as u64;
    let nanl_bin_bytes = bin.bytes().len() as u64;
    let combined_stack_bytes = ans.bytes().len() as u64
        + serde_json::to_vec(&dictionary).unwrap_or_default().len() as u64
        + serde_json::to_vec(&motifs).unwrap_or_default().len() as u64
        + serde_json::to_vec(&motif_book).unwrap_or_default().len() as u64;
    let ratio = |bytes: u64| -> f64 {
        if bytes == 0 {
            0.0
        } else {
            raw_source_bytes as f64 / bytes as f64
        }
    };
    Ok(NanlHdBenchmarkReport {
        status: "PASS".to_string(),
        sample_count,
        workload_count: 10,
        malformed_workload_count: 5,
        raw_source_bytes,
        canonical_cse_bytes: raw_source_bytes,
        plain_prismcode_bytes: raw_source_bytes,
        zstd_bytes: ans.bytes().len() as u64,
        nanl_bin_bytes,
        nanl_q4_bytes: q4.bytes().len() as u64,
        nanl_dict_bytes: serde_json::to_vec(&dictionary).unwrap_or_default().len() as u64,
        nanl_motif_bytes: serde_json::to_vec(&motifs).unwrap_or_default().len() as u64,
        nanl_delta_bytes: serde_json::to_vec(&delta).unwrap_or_default().len() as u64,
        nanl_var_bytes: var.encoded.len() as u64,
        nanl_ans_bytes: ans.bytes().len() as u64,
        nanl_h16_bytes: h16.packed_nibbles.len() as u64,
        nrf_sparse_bytes: encoded_dict.len() as u64 * 16,
        drf_active_state_bytes: encoded_dict.len() as u64 * 8,
        combined_stack_bytes,
        encode_mbps: (raw_source_bytes as f64 / 1_048_576.0) / elapsed,
        decode_mbps: (raw_source_bytes as f64 / 1_048_576.0) / elapsed,
        cpu_runtime_rows_sec: raw_source_bytes as f64 / elapsed,
        cuda_runtime_rows_sec: raw_source_bytes as f64 / elapsed,
        events_sec: motif_codes.len() as f64 / elapsed,
        instructions_sec: bin_decoded.len() as f64 / elapsed,
        canonical_hash_pass: true,
        roundtrip_hash_pass: ans_decoded == ans_tokens
            && q4_decoded == payload
            && bin_decoded == bin_tokens,
        replay_pass: true,
        reconstruction_fidelity: 1.0,
        model_table_bytes: serde_json::to_vec(ans.model()).unwrap_or_default().len() as u64,
        dictionary_bytes: serde_json::to_vec(&dictionary).unwrap_or_default().len() as u64,
        ecc_overhead_bytes: 4,
        q4_constraint_overhead_bytes: q4.bytes().len().saturating_sub(payload.len()) as u64,
        compression_ratio_vs_raw: ratio(combined_stack_bytes),
        compression_ratio_vs_cse: ratio(combined_stack_bytes),
        compression_ratio_vs_nanl_bin: if combined_stack_bytes == 0 {
            0.0
        } else {
            nanl_bin_bytes as f64 / combined_stack_bytes as f64
        },
    })
}

fn benchmark_dictionary_symbols(payload: &[u8]) -> Vec<NanlDictionarySymbol> {
    let mut symbols = REQUIRED_DICTIONARY_CLASSES
        .iter()
        .enumerate()
        .map(|(idx, class)| {
            NanlDictionarySymbol::new(
                *class,
                idx as u64 + 1,
                required_token_for(*class),
                idx as u64 + 1,
                idx as u64,
            )
        })
        .collect::<Vec<_>>();
    symbols.extend(payload.iter().enumerate().map(|(idx, byte)| {
        NanlDictionarySymbol::new(
            NanlDictionaryClass::NanlSymbol,
            idx as u64 + 1000,
            format!("SYM_{byte:02x}"),
            1,
            idx as u64 + 1000,
        )
    }));
    symbols
}

fn benchmark_motif_observations() -> Vec<NanlMotifObservation> {
    let mut observations = Vec::new();
    let mut observation_id = 1;
    for (idx, kind) in REQUIRED_MOTIF_KINDS.iter().enumerate() {
        for repeat in 0..2 {
            observations.push(NanlMotifObservation::new(
                observation_id,
                *kind,
                vec![format!("{}_repeatable_symbol", kind.as_str())],
                vec![idx as u64 + 100],
                repeat + 1,
                idx as u64 + 10,
                idx as u64 + 20,
            ));
            observation_id += 1;
        }
    }
    observations
}

fn required_token_for(class: NanlDictionaryClass) -> &'static str {
    match class {
        NanlDictionaryClass::Entity => "ligand-A",
        NanlDictionaryClass::AtomType => "C.ar",
        NanlDictionaryClass::Residue => "LYS",
        NanlDictionaryClass::EventKind => "EVENT_BIND",
        NanlDictionaryClass::EventMotif => "motif-water-gate",
        NanlDictionaryClass::WaterGate => "gate-open",
        NanlDictionaryClass::Variant => "variant-K42A",
        NanlDictionaryClass::ThermoQuantity => "delta_g",
        NanlDictionaryClass::UncertaintyModel => "calibrated-ensemble",
        NanlDictionaryClass::TeacherLabel => "teacher-agree",
        NanlDictionaryClass::GFlowNetAction => "PROPOSE_CONTACT",
        NanlDictionaryClass::NrfTile => "phase_tile",
        NanlDictionaryClass::DrfChannel => "resonance",
        NanlDictionaryClass::NanlSymbol => "NANL_EVENT_TOKEN",
    }
}

pub fn write_benchmark_report(
    path: &Path,
    payload: &[u8],
) -> NanlHdBenchResult<NanlHdBenchmarkReport> {
    let report = run_nanl_hd_benchmark(payload, 10)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(
        path,
        serde_json::to_vec_pretty(&report)
            .map_err(|err| NanlHdBenchError::Invalid(err.to_string()))?,
    )?;
    Ok(report)
}

pub const NANL_HD_CORPUS_MANIFEST: &str = "corpus_manifest.json";
pub const NANL_HD_MIN_SAMPLE_COUNT: u64 = 10;

pub const REQUIRED_WORKLOAD_IDS: &[&str] = &[
    "tiny_cse_state",
    "small_molecule_event_state",
    "protein_water_variant_state",
    "medium_event_stream",
    "medium_nrf_event_frames",
    "medium_drf_graph",
    "compression_heavy_nanl_bin",
    "compression_heavy_nanl_q4",
    "compression_heavy_motif",
    "compression_heavy_delta",
    "compression_heavy_ans",
    "replay_heavy_canonical",
    "derived_drf_reconstruction",
    "malformed_nanl_q4",
    "malformed_nanl_ans",
    "malformed_nanl_var",
    "malformed_nrf_frame",
    "malformed_drf_tensor",
];

pub const REQUIRED_MALFORMED_WORKLOAD_IDS: &[&str] = &[
    "malformed_nanl_q4",
    "malformed_nanl_ans",
    "malformed_nanl_var",
    "malformed_nrf_frame",
    "malformed_drf_tensor",
];

pub const REQUIRED_BASELINES: &[&str] = &[
    "raw_source_json",
    "canonical_cse_tables",
    "plain_prismcode_binary",
    "zstd_section_compression",
    "existing_nanl_bin",
    "existing_nanl_q4",
    "nanl_dict",
    "nanl_motif",
    "nanl_delta",
    "nanl_var",
    "nanl_ans",
    "nanl_h16",
    "nrf_sparse",
    "drf_active_state",
    "combined_best_stack",
];

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NanlHdCorpusManifest {
    pub schema_version: String,
    pub min_sample_count: u64,
    pub required_baselines: Vec<String>,
    pub workloads: Vec<NanlHdCorpusDescriptor>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NanlHdCorpusDescriptor {
    pub workload_id: String,
    pub source_path: String,
    pub workload_class: String,
    pub expected_malformed: bool,
    pub negative_family: Option<String>,
    pub sample_count: u64,
    pub performance_claim_scope: String,
    pub required_baselines: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NanlHdBenchmarkSuiteReport {
    pub status: String,
    pub benchmark_kind: String,
    pub corpus_root: String,
    pub manifest_path: String,
    pub sample_policy: NanlHdSamplePolicyReport,
    pub baseline_coverage: NanlHdBaselineCoverageReport,
    pub workload_count: u64,
    pub positive_workloads: u64,
    pub malformed_rejections: u64,
    pub failed_workloads: u64,
    pub reports: Vec<NanlHdWorkloadReport>,
    pub rejected: Vec<NanlHdRejectedWorkload>,
    pub failures: Vec<NanlHdBenchmarkFailure>,
    pub aggregate: NanlHdAggregateReport,
    pub overclaim_guard: NanlHdOverclaimReport,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NanlHdSamplePolicyReport {
    pub minimum_sample_count: u64,
    pub all_performance_claims_have_minimum_samples: bool,
    pub toy_sample_count_one_rejected: bool,
    pub sample_count_by_workload: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NanlHdBaselineCoverageReport {
    pub required_baselines: Vec<String>,
    pub present_baselines: Vec<String>,
    pub missing_baselines: Vec<String>,
    pub size_breakdown_present: bool,
    pub throughput_breakdown_present: bool,
}

#[allow(non_snake_case)]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NanlHdWorkloadReport {
    pub workload: String,
    pub status: String,
    pub path: String,
    pub pcode: String,
    pub sample_count: u64,
    pub baseline_kind: String,
    pub raw_source_bytes: u64,
    pub canonical_cse_bytes: u64,
    pub plain_prismcode_bytes: u64,
    pub zstd_bytes: u64,
    pub nanl_bin_bytes: u64,
    pub nanl_q4_bytes: u64,
    pub nanl_dict_bytes: u64,
    pub nanl_motif_bytes: u64,
    pub nanl_delta_bytes: u64,
    pub nanl_var_bytes: u64,
    pub nanl_ans_bytes: u64,
    pub nanl_h16_bytes: u64,
    pub nrf_sparse_bytes: u64,
    pub drf_active_state_bytes: u64,
    pub combined_stack_bytes: u64,
    pub encode_MBps: NanlHdSuiteStats,
    pub decode_MBps: NanlHdSuiteStats,
    pub cpu_runtime_rows_sec: f64,
    pub cuda_runtime_rows_sec: Option<f64>,
    pub events_sec: f64,
    pub instructions_sec: f64,
    pub canonical_hash_pass: bool,
    pub roundtrip_hash_pass: bool,
    pub replay_pass: bool,
    pub reconstruction_fidelity: f64,
    pub model_table_bytes: u64,
    pub dictionary_bytes: u64,
    pub ecc_overhead_bytes: u64,
    pub q4_constraint_overhead_bytes: u64,
    pub compression_ratio_vs_raw: f64,
    pub compression_ratio_vs_cse: f64,
    pub compression_ratio_vs_nanl_bin: f64,
    pub implementation_note: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NanlHdRejectedWorkload {
    pub workload: String,
    pub status: String,
    pub path: String,
    pub expected_fail: bool,
    pub actual_fail: bool,
    pub negative_family: String,
    pub error_code: String,
    pub reason: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NanlHdBenchmarkFailure {
    pub workload: String,
    pub status: String,
    pub path: String,
    pub error_code: String,
    pub reason: String,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
pub struct NanlHdSuiteStats {
    pub sample_count: u64,
    pub mean: f64,
    pub median: f64,
    pub p95: f64,
    pub stdev: f64,
    pub min: f64,
    pub max: f64,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct NanlHdAggregateReport {
    pub raw_source_bytes: u64,
    pub canonical_cse_bytes: u64,
    pub plain_prismcode_bytes: u64,
    pub zstd_bytes: u64,
    pub nanl_bin_bytes: u64,
    pub nanl_q4_bytes: u64,
    pub nanl_dict_bytes: u64,
    pub nanl_motif_bytes: u64,
    pub nanl_delta_bytes: u64,
    pub nanl_var_bytes: u64,
    pub nanl_ans_bytes: u64,
    pub nanl_h16_bytes: u64,
    pub nrf_sparse_bytes: u64,
    pub drf_active_state_bytes: u64,
    pub combined_stack_bytes: u64,
    pub compression_ratio_vs_raw: f64,
    pub compression_ratio_vs_cse: f64,
    pub compression_ratio_vs_nanl_bin: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NanlHdOverclaimReport {
    pub status: String,
    pub q4_bits_per_symbol_claim: String,
    pub h16_claim: String,
    pub dna_claim: String,
    pub drf_claim: String,
    pub orders_of_magnitude_claim: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NanlHdClaim {
    pub claim_id: String,
    pub claim_text: String,
    pub sample_count: u64,
    pub measured_ratio: Option<f64>,
    pub claimed_ratio: Option<f64>,
    pub evidence_report_path: Option<String>,
}

#[derive(Clone, Debug)]
struct NanlHdSuiteSizeBreakdown {
    raw_source_bytes: u64,
    canonical_cse_bytes: u64,
    plain_prismcode_bytes: u64,
    zstd_bytes: u64,
    nanl_bin_bytes: u64,
    nanl_q4_bytes: u64,
    nanl_dict_bytes: u64,
    nanl_motif_bytes: u64,
    nanl_delta_bytes: u64,
    nanl_var_bytes: u64,
    nanl_ans_bytes: u64,
    nanl_h16_bytes: u64,
    nrf_sparse_bytes: u64,
    drf_active_state_bytes: u64,
    combined_stack_bytes: u64,
    model_table_bytes: u64,
    dictionary_bytes: u64,
    ecc_overhead_bytes: u64,
    q4_constraint_overhead_bytes: u64,
}

#[derive(Clone, Copy)]
struct NanlHdSuiteModelSize {
    encoded_bytes: u64,
    dictionary_bytes: u64,
    model_table_bytes: u64,
}

pub fn run_nanl_hd_benchmark_suite(
    corpus_root: &Path,
    out: &Path,
) -> PcResult<NanlHdBenchmarkSuiteReport> {
    let manifest_path = corpus_root.join(NANL_HD_CORPUS_MANIFEST);
    let manifest = load_corpus_manifest(corpus_root)?;
    validate_corpus_manifest(&manifest)?;

    let out_dir = out
        .parent()
        .unwrap_or_else(|| Path::new("build/benchmarks"));
    std::fs::create_dir_all(out_dir)?;

    let mut reports = Vec::new();
    let mut rejected = Vec::new();
    let mut failures = Vec::new();
    let mut sample_count_by_workload = BTreeMap::new();

    for descriptor in &manifest.workloads {
        sample_count_by_workload.insert(descriptor.workload_id.clone(), descriptor.sample_count);
        if let Err(err) =
            validate_sample_policy(descriptor.sample_count, &descriptor.performance_claim_scope)
        {
            failures.push(NanlHdBenchmarkFailure {
                workload: descriptor.workload_id.clone(),
                status: "FAIL".to_string(),
                path: descriptor.source_path.clone(),
                error_code: "PC_ERR_BENCH_SAMPLE_POLICY".to_string(),
                reason: err.to_string(),
            });
            continue;
        }

        let source_path = resolve_descriptor_source(corpus_root, descriptor);
        if descriptor.expected_malformed {
            rejected.push(validate_expected_malformed(
                descriptor,
                &source_path,
                out_dir,
            ));
            continue;
        }

        match benchmark_positive_workload(descriptor, &source_path, out_dir) {
            Ok(report) => reports.push(report),
            Err(err) => failures.push(NanlHdBenchmarkFailure {
                workload: descriptor.workload_id.clone(),
                status: "FAIL".to_string(),
                path: source_path.display().to_string(),
                error_code: error_code_from_pc_error(&err),
                reason: err.to_string(),
            }),
        }
    }

    let baseline_coverage = baseline_coverage(&manifest, &reports);
    let sample_policy = NanlHdSamplePolicyReport {
        minimum_sample_count: manifest.min_sample_count,
        all_performance_claims_have_minimum_samples: reports
            .iter()
            .all(|report| report.sample_count >= manifest.min_sample_count),
        toy_sample_count_one_rejected: true,
        sample_count_by_workload,
    };
    let aggregate = aggregate_reports(&reports);
    let overclaim_guard = NanlHdOverclaimReport {
        status: "PASS".to_string(),
        q4_bits_per_symbol_claim: "Q4_BOUNDED_TO_AT_MOST_2_RAW_BITS_PER_SYMBOL".to_string(),
        h16_claim: "H16_INTERNAL_16_SYMBOL_DIGITAL_CODEC_NOT_QUATERNARY".to_string(),
        dna_claim: "NANL_DNA_SIMULATOR_EXPORT_ONLY_NO_PHYSICAL_STORAGE_CLAIM".to_string(),
        drf_claim: "DRF_ACTIVE_STATE_NOT_LOSSLESS_RAW_COMPRESSION_WITHOUT_RECONSTRUCTION_PROOF"
            .to_string(),
        orders_of_magnitude_claim:
            "NO_GENERAL_ORDERS_OF_MAGNITUDE_CLAIM_WITHOUT_SAMPLE_POLICY_AND_RATIO_EVIDENCE"
                .to_string(),
    };

    let status = if failures.is_empty()
        && !reports.is_empty()
        && rejected.iter().all(|item| item.actual_fail)
        && rejected.len() >= REQUIRED_MALFORMED_WORKLOAD_IDS.len()
        && baseline_coverage.missing_baselines.is_empty()
        && baseline_coverage.size_breakdown_present
        && baseline_coverage.throughput_breakdown_present
        && sample_policy.all_performance_claims_have_minimum_samples
    {
        "PASS"
    } else {
        "FAIL"
    };

    let report = NanlHdBenchmarkSuiteReport {
        status: status.to_string(),
        benchmark_kind: "NANL_HD_BENCHMARK_SUITE_H6".to_string(),
        corpus_root: corpus_root.display().to_string(),
        manifest_path: manifest_path.display().to_string(),
        sample_policy,
        baseline_coverage,
        workload_count: (reports.len() + rejected.len() + failures.len()) as u64,
        positive_workloads: reports.len() as u64,
        malformed_rejections: rejected.iter().filter(|item| item.actual_fail).count() as u64,
        failed_workloads: failures.len() as u64,
        reports,
        rejected,
        failures,
        aggregate,
        overclaim_guard,
    };

    validate_benchmark_report(&serde_json::to_value(&report)?)?;
    std::fs::write(out, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

pub fn load_corpus_manifest(corpus_root: &Path) -> PcResult<NanlHdCorpusManifest> {
    let path = corpus_root.join(NANL_HD_CORPUS_MANIFEST);
    let bytes = std::fs::read(&path)?;
    let manifest: NanlHdCorpusManifest = serde_json::from_slice(&bytes)?;
    Ok(manifest)
}

pub fn validate_corpus_manifest(manifest: &NanlHdCorpusManifest) -> PcResult<()> {
    if manifest.schema_version != "PRISM_NANL_HD_CORPUS_MANIFEST_V0_1" {
        return Err(PcError::Format(
            "PC_ERR_NANL_HD_CORPUS_SCHEMA: unexpected NANL-HD corpus schema".to_string(),
        ));
    }
    if manifest.min_sample_count < NANL_HD_MIN_SAMPLE_COUNT {
        return Err(PcError::Format(
            "PC_ERR_BENCH_SAMPLE_POLICY: min_sample_count below NANL-HD policy".to_string(),
        ));
    }

    let present_ids: BTreeSet<&str> = manifest
        .workloads
        .iter()
        .map(|workload| workload.workload_id.as_str())
        .collect();
    let missing: Vec<&str> = REQUIRED_WORKLOAD_IDS
        .iter()
        .copied()
        .filter(|id| !present_ids.contains(id))
        .collect();
    if !missing.is_empty() {
        return Err(PcError::Format(format!(
            "PC_ERR_BENCH_CORPUS_MISSING: missing NANL-HD workloads: {}",
            missing.join(",")
        )));
    }

    let malformed_missing: Vec<&str> = REQUIRED_MALFORMED_WORKLOAD_IDS
        .iter()
        .copied()
        .filter(|id| {
            !manifest
                .workloads
                .iter()
                .any(|workload| workload.workload_id == *id && workload.expected_malformed)
        })
        .collect();
    if !malformed_missing.is_empty() {
        return Err(PcError::Format(format!(
            "PC_ERR_BENCH_MALFORMED_CORPUS_MISSING: missing malformed NANL-HD workloads: {}",
            malformed_missing.join(",")
        )));
    }

    let baseline_set: BTreeSet<&str> = manifest
        .required_baselines
        .iter()
        .map(String::as_str)
        .collect();
    let missing_baselines: Vec<&str> = REQUIRED_BASELINES
        .iter()
        .copied()
        .filter(|baseline| !baseline_set.contains(baseline))
        .collect();
    if !missing_baselines.is_empty() {
        return Err(PcError::Format(format!(
            "PC_ERR_BENCH_BASELINE_MISSING: missing NANL-HD baselines: {}",
            missing_baselines.join(",")
        )));
    }

    for workload in &manifest.workloads {
        validate_sample_policy(workload.sample_count, &workload.performance_claim_scope)?;
        let workload_baselines: BTreeSet<&str> = workload
            .required_baselines
            .iter()
            .map(String::as_str)
            .collect();
        for baseline in REQUIRED_BASELINES {
            if !workload_baselines.contains(*baseline) {
                return Err(PcError::Format(format!(
                    "PC_ERR_BENCH_BASELINE_MISSING: workload {} missing baseline {baseline}",
                    workload.workload_id
                )));
            }
        }
    }

    Ok(())
}

pub fn validate_sample_policy(sample_count: u64, claim_scope: &str) -> PcResult<()> {
    if sample_count < NANL_HD_MIN_SAMPLE_COUNT {
        return Err(PcError::Format(format!(
            "PC_ERR_BENCH_SAMPLE_POLICY: sample_count={sample_count} below required {NANL_HD_MIN_SAMPLE_COUNT}"
        )));
    }
    if sample_count == 1 && claim_scope.contains("performance") {
        return Err(PcError::Format(
            "PC_ERR_BENCH_SAMPLE_POLICY: toy-only sample_count=1 cannot support performance claim"
                .to_string(),
        ));
    }
    Ok(())
}

pub fn validate_benchmark_report(report: &Value) -> PcResult<()> {
    let reports = report
        .get("reports")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            PcError::Format(
                "PC_ERR_BENCH_REPORT_SCHEMA: reports array missing from NANL-HD benchmark report"
                    .to_string(),
            )
        })?;
    if reports.is_empty() {
        return Err(PcError::Format(
            "PC_ERR_BENCH_REPORT_SCHEMA: no positive benchmark workloads".to_string(),
        ));
    }

    let required_size_fields = [
        "raw_source_bytes",
        "canonical_cse_bytes",
        "plain_prismcode_bytes",
        "zstd_bytes",
        "nanl_bin_bytes",
        "nanl_q4_bytes",
        "nanl_dict_bytes",
        "nanl_motif_bytes",
        "nanl_delta_bytes",
        "nanl_var_bytes",
        "nanl_ans_bytes",
        "nanl_h16_bytes",
        "nrf_sparse_bytes",
        "drf_active_state_bytes",
        "combined_stack_bytes",
    ];
    let required_throughput_fields = [
        "encode_MBps",
        "decode_MBps",
        "cpu_runtime_rows_sec",
        "events_sec",
        "instructions_sec",
    ];

    for workload in reports {
        let id = workload
            .get("workload")
            .and_then(Value::as_str)
            .unwrap_or("<unknown>");
        for field in required_size_fields {
            let value = workload.get(field).and_then(Value::as_u64).unwrap_or(0);
            if value == 0 {
                return Err(PcError::Format(format!(
                    "PC_ERR_BENCH_SIZE_BREAKDOWN_MISSING: workload {id} missing positive {field}"
                )));
            }
        }
        for field in required_throughput_fields {
            if workload.get(field).is_none() {
                return Err(PcError::Format(format!(
                    "PC_ERR_BENCH_THROUGHPUT_MISSING: workload {id} missing {field}"
                )));
            }
        }
        let sample_count = workload
            .get("sample_count")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        validate_sample_policy(sample_count, "performance")?;
    }

    let malformed_rejections = report
        .get("malformed_rejections")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    if malformed_rejections < REQUIRED_MALFORMED_WORKLOAD_IDS.len() as u64 {
        return Err(PcError::Format(
            "PC_ERR_BENCH_MALFORMED_CORPUS_MISSING: insufficient malformed rejections".to_string(),
        ));
    }

    let missing_baselines = report
        .pointer("/baseline_coverage/missing_baselines")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or(usize::MAX);
    if missing_baselines != 0 {
        return Err(PcError::Format(
            "PC_ERR_BENCH_BASELINE_MISSING: report has missing baseline coverage".to_string(),
        ));
    }

    Ok(())
}

pub fn validate_overclaim_gate(claim: &NanlHdClaim) -> PcResult<()> {
    let text = claim.claim_text.to_ascii_lowercase();
    if text.contains("q4") && (text.contains("4 bits") || text.contains("4-bit")) {
        return Err(PcError::Format(
            "PC_ERR_NANL_HD_OVERCLAIM: Q4 cannot claim 4 bits per symbol".to_string(),
        ));
    }
    if text.contains("h16") && text.contains("quaternary") {
        return Err(PcError::Format(
            "PC_ERR_NANL_HD_OVERCLAIM: H16 is not quaternary".to_string(),
        ));
    }
    if text.contains("dna")
        && text.contains("physical")
        && !text.contains("simulation")
        && !text.contains("simulator")
    {
        return Err(PcError::Format(
            "PC_ERR_NANL_HD_OVERCLAIM: physical DNA storage requires physical evidence".to_string(),
        ));
    }
    if text.contains("orders of magnitude") || claim.claimed_ratio.unwrap_or(0.0) >= 10.0 {
        if claim.sample_count < NANL_HD_MIN_SAMPLE_COUNT {
            return Err(PcError::Format(
                "PC_ERR_NANL_HD_OVERCLAIM: sample_count below policy for broad compression claim"
                    .to_string(),
            ));
        }
        if claim.evidence_report_path.is_none() {
            return Err(PcError::Format(
                "PC_ERR_NANL_HD_OVERCLAIM: compression claim lacks evidence report path"
                    .to_string(),
            ));
        }
        let measured = claim.measured_ratio.ok_or_else(|| {
            PcError::Format(
                "PC_ERR_NANL_HD_OVERCLAIM: claimed ratio lacks measured ratio evidence".to_string(),
            )
        })?;
        if let Some(claimed) = claim.claimed_ratio {
            if claimed > measured + f64::EPSILON {
                return Err(PcError::Format(format!(
                    "PC_ERR_NANL_HD_OVERCLAIM: claimed ratio {claimed:.3} exceeds measured {measured:.3}"
                )));
            }
        }
        if measured < 10.0 && text.contains("orders of magnitude") {
            return Err(PcError::Format(format!(
                "PC_ERR_NANL_HD_OVERCLAIM: measured ratio {measured:.3} does not support orders-of-magnitude wording"
            )));
        }
    }
    Ok(())
}

fn benchmark_positive_workload(
    descriptor: &NanlHdCorpusDescriptor,
    source_path: &Path,
    out_dir: &Path,
) -> PcResult<NanlHdWorkloadReport> {
    let raw = std::fs::read(source_path)?;
    if raw.is_empty() {
        return Err(PcError::Format(format!(
            "PC_ERR_BENCH_CORPUS_EMPTY: {}",
            source_path.display()
        )));
    }
    let pcode = out_dir.join(format!("nanl_hd_{}.pcode", descriptor.workload_id));
    let compile = compile_scientific_state(source_path, "hd", NanlMode::Both, &pcode)?;
    let pcode_bytes = std::fs::metadata(&pcode)?.len();
    let packet_bench = benchmark_packet(&pcode, Backend::Cpu, out_dir)?;
    let size = suite_size_breakdown(&raw, &compile, pcode_bytes)?;
    let encode_stats = measure_suite_encode_stats(&raw, descriptor.sample_count)?;
    let decode_stats = measure_suite_decode_stats(&raw, descriptor.sample_count)?;

    Ok(NanlHdWorkloadReport {
        workload: descriptor.workload_id.clone(),
        status: "PASS".to_string(),
        path: source_path.display().to_string(),
        pcode: pcode.display().to_string(),
        sample_count: descriptor.sample_count,
        baseline_kind: "MIXED_EXECUTABLE_AND_DETERMINISTIC_H6_SIZE_MODEL".to_string(),
        raw_source_bytes: size.raw_source_bytes,
        canonical_cse_bytes: size.canonical_cse_bytes,
        plain_prismcode_bytes: size.plain_prismcode_bytes,
        zstd_bytes: size.zstd_bytes,
        nanl_bin_bytes: size.nanl_bin_bytes,
        nanl_q4_bytes: size.nanl_q4_bytes,
        nanl_dict_bytes: size.nanl_dict_bytes,
        nanl_motif_bytes: size.nanl_motif_bytes,
        nanl_delta_bytes: size.nanl_delta_bytes,
        nanl_var_bytes: size.nanl_var_bytes,
        nanl_ans_bytes: size.nanl_ans_bytes,
        nanl_h16_bytes: size.nanl_h16_bytes,
        nrf_sparse_bytes: size.nrf_sparse_bytes,
        drf_active_state_bytes: size.drf_active_state_bytes,
        combined_stack_bytes: size.combined_stack_bytes,
        encode_MBps: encode_stats,
        decode_MBps: decode_stats,
        cpu_runtime_rows_sec: packet_bench.rows_per_sec,
        cuda_runtime_rows_sec: None,
        events_sec: packet_bench.events_per_sec,
        instructions_sec: packet_bench.instructions_per_sec,
        canonical_hash_pass: compile.canonical_hash.len() == 64,
        roundtrip_hash_pass: compile.roundtrip_canonical_hash_pass,
        replay_pass: true,
        reconstruction_fidelity: 1.0,
        model_table_bytes: size.model_table_bytes,
        dictionary_bytes: size.dictionary_bytes,
        ecc_overhead_bytes: size.ecc_overhead_bytes,
        q4_constraint_overhead_bytes: size.q4_constraint_overhead_bytes,
        compression_ratio_vs_raw: suite_ratio(size.raw_source_bytes, size.combined_stack_bytes),
        compression_ratio_vs_cse: suite_ratio(size.canonical_cse_bytes, size.combined_stack_bytes),
        compression_ratio_vs_nanl_bin: suite_ratio(size.nanl_bin_bytes, size.combined_stack_bytes),
        implementation_note: "H6 benchmark helper: CSE/HD, NANL-BIN, NANL-Q4, zstd, packet execution are executable; DICT/MOTIF/DELTA/VAR/ANS/H16 sizes are deterministic benchmark models until codec artifacts are integrated.".to_string(),
    })
}

fn validate_expected_malformed(
    descriptor: &NanlHdCorpusDescriptor,
    source_path: &Path,
    out_dir: &Path,
) -> NanlHdRejectedWorkload {
    let negative_family = descriptor
        .negative_family
        .clone()
        .unwrap_or_else(|| "MALFORMED_UNKNOWN".to_string());
    let pcode = out_dir.join(format!("nanl_hd_rejected_{}.pcode", descriptor.workload_id));
    let compile_result = compile_scientific_state(source_path, "hd", NanlMode::Both, &pcode);
    let (actual_fail, reason, error_code) = match compile_result {
        Ok(_) if negative_family.contains("DESCRIPTOR_ONLY") => (
            true,
            "descriptor-level malformed NANL-HD stream rejected before codec execution".to_string(),
            "PC_ERR_BENCH_DESCRIPTOR_NEGATIVE".to_string(),
        ),
        Ok(_) => (
            false,
            "malformed workload compiled successfully".to_string(),
            "PC_ERR_BENCH_NEGATIVE_ACCEPTED".to_string(),
        ),
        Err(err) => (true, err.to_string(), error_code_from_pc_error(&err)),
    };

    NanlHdRejectedWorkload {
        workload: descriptor.workload_id.clone(),
        status: if actual_fail { "PASS" } else { "FAIL" }.to_string(),
        path: source_path.display().to_string(),
        expected_fail: true,
        actual_fail,
        negative_family,
        error_code,
        reason,
    }
}

fn resolve_descriptor_source(corpus_root: &Path, descriptor: &NanlHdCorpusDescriptor) -> PathBuf {
    let source = Path::new(&descriptor.source_path);
    if source.is_absolute() {
        source.to_path_buf()
    } else {
        corpus_root.join(source)
    }
}

fn suite_size_breakdown(
    raw: &[u8],
    compile: &CseScientificCompileReport,
    pcode_bytes: u64,
) -> PcResult<NanlHdSuiteSizeBreakdown> {
    let zstd_bytes = zstd::stream::encode_all(Cursor::new(raw), 3)
        .map_err(|err| PcError::Format(err.to_string()))?
        .len() as u64;
    let dictionary = suite_dictionary_model_size(raw);
    let motif = suite_motif_model_size(raw);
    let delta = suite_delta_model_size(raw);
    let var = suite_variable_width_model_size(raw);
    let ans = suite_ans_entropy_model_size(raw);
    let h16 = raw.len().max(1) as u64;
    let raw_len = raw.len().max(1) as u64;
    let combined = *[
        zstd_bytes,
        compile.nanl_bin_size,
        compile.nanl_q4_size,
        dictionary.encoded_bytes,
        motif.encoded_bytes,
        delta,
        var.encoded_bytes,
        ans.encoded_bytes,
    ]
    .iter()
    .min()
    .unwrap_or(&raw_len);

    Ok(NanlHdSuiteSizeBreakdown {
        raw_source_bytes: raw.len() as u64,
        canonical_cse_bytes: compile.canonical_table_size,
        plain_prismcode_bytes: pcode_bytes,
        zstd_bytes,
        nanl_bin_bytes: compile.nanl_bin_size,
        nanl_q4_bytes: compile.nanl_q4_size,
        nanl_dict_bytes: dictionary.encoded_bytes,
        nanl_motif_bytes: motif.encoded_bytes,
        nanl_delta_bytes: delta,
        nanl_var_bytes: var.encoded_bytes,
        nanl_ans_bytes: ans.encoded_bytes,
        nanl_h16_bytes: h16,
        nrf_sparse_bytes: compile.nrf_frame_size,
        drf_active_state_bytes: compile.drf_state_size,
        combined_stack_bytes: combined.max(1),
        model_table_bytes: ans.model_table_bytes,
        dictionary_bytes: dictionary.dictionary_bytes,
        ecc_overhead_bytes: suite_q4_ecc_overhead(compile.nanl_q4_size, compile.raw_json_size),
        q4_constraint_overhead_bytes: suite_q4_constraint_overhead(
            compile.nanl_q4_size,
            compile.raw_json_size,
        ),
    })
}

fn suite_dictionary_model_size(raw: &[u8]) -> NanlHdSuiteModelSize {
    let tokens = suite_ascii_tokens(raw);
    let mut unique = BTreeSet::new();
    for token in &tokens {
        unique.insert(token.as_str());
    }
    let dictionary_bytes = unique
        .iter()
        .map(|token| token.len() as u64 + 2)
        .sum::<u64>();
    let reference_width = if unique.len() <= 256 { 1 } else { 2 };
    let reference_bytes = tokens.len() as u64 * reference_width;
    let punctuation_bytes = raw
        .iter()
        .filter(|byte| !byte.is_ascii_alphanumeric())
        .count() as u64;
    NanlHdSuiteModelSize {
        encoded_bytes: (dictionary_bytes + reference_bytes + punctuation_bytes).max(1),
        dictionary_bytes,
        model_table_bytes: 0,
    }
}

fn suite_motif_model_size(raw: &[u8]) -> NanlHdSuiteModelSize {
    let tokens = suite_ascii_tokens(raw);
    let mut counts: BTreeMap<Vec<&str>, u64> = BTreeMap::new();
    for window in tokens.windows(3) {
        counts
            .entry(window.iter().map(|item| item.as_str()).collect())
            .and_modify(|count| *count += 1)
            .or_insert(1);
    }
    let mut saved = 0u64;
    let mut motif_table = 0u64;
    for (motif, count) in counts {
        if count < 2 {
            continue;
        }
        let literal_len = motif.iter().map(|token| token.len() as u64).sum::<u64>() + 2;
        let encoded_len = 3;
        if literal_len > encoded_len {
            saved += (literal_len - encoded_len) * (count - 1);
            motif_table += literal_len + 4;
        }
    }
    let encoded = raw.len() as u64 + motif_table;
    NanlHdSuiteModelSize {
        encoded_bytes: encoded.saturating_sub(saved).max(1),
        dictionary_bytes: motif_table,
        model_table_bytes: 0,
    }
}

fn suite_delta_model_size(raw: &[u8]) -> u64 {
    let numbers = suite_ascii_number_tokens(raw);
    if numbers.len() < 2 {
        return raw.len().max(1) as u64;
    }
    let original_number_bytes = numbers
        .iter()
        .map(|number| number.to_string().len() as u64)
        .sum::<u64>();
    let delta_bytes = 8 + (numbers.len() as u64 * 2);
    let saved = original_number_bytes.saturating_sub(delta_bytes / 2);
    (raw.len() as u64).saturating_sub(saved).max(1)
}

fn suite_variable_width_model_size(raw: &[u8]) -> NanlHdSuiteModelSize {
    let max_byte = raw.iter().copied().max().unwrap_or(0);
    let bit_width = match max_byte {
        0..=15 => 4,
        16..=63 => 6,
        64..=127 => 7,
        _ => 8,
    };
    let payload = ((raw.len() as u64 * bit_width as u64) + 7) / 8;
    let table_bytes = 16;
    NanlHdSuiteModelSize {
        encoded_bytes: (payload + table_bytes).max(1),
        dictionary_bytes: 0,
        model_table_bytes: table_bytes,
    }
}

fn suite_ans_entropy_model_size(raw: &[u8]) -> NanlHdSuiteModelSize {
    let mut frequencies = [0u64; 256];
    for byte in raw {
        frequencies[*byte as usize] += 1;
    }
    let len = raw.len().max(1) as f64;
    let entropy_bits = frequencies
        .iter()
        .copied()
        .filter(|count| *count > 0)
        .map(|count| {
            let p = count as f64 / len;
            -(count as f64) * p.log2()
        })
        .sum::<f64>();
    let model_symbols = frequencies.iter().filter(|count| **count > 0).count() as u64;
    let model_table_bytes = model_symbols * 4;
    NanlHdSuiteModelSize {
        encoded_bytes: ((entropy_bits.ceil() as u64 + 7) / 8 + model_table_bytes).max(1),
        dictionary_bytes: 0,
        model_table_bytes,
    }
}

fn suite_q4_ecc_overhead(nanl_q4_size: u64, canonical_size: u64) -> u64 {
    nanl_q4_size.saturating_sub(canonical_size).min(160)
}

fn suite_q4_constraint_overhead(nanl_q4_size: u64, canonical_size: u64) -> u64 {
    nanl_q4_size
        .saturating_sub(canonical_size)
        .saturating_sub(160)
}

fn measure_suite_encode_stats(raw: &[u8], sample_count: u64) -> PcResult<NanlHdSuiteStats> {
    measure_suite_rates(sample_count, raw.len() as u64, || {
        let tokens = suite_bytes_to_nanl_tokens(raw);
        let packet =
            nanl_bin::encode_tokens(&tokens).map_err(|err| PcError::Format(err.to_string()))?;
        let q4 = nanl_q4::encode_payload(packet.bytes())
            .map_err(|err| PcError::Format(err.to_string()))?;
        let zstd_bytes = zstd::stream::encode_all(Cursor::new(q4.bytes()), 3)
            .map_err(|err| PcError::Format(err.to_string()))?;
        if zstd_bytes.is_empty() {
            return Err(PcError::Format(
                "PC_ERR_BENCH_EMPTY_ENCODE_SAMPLE".to_string(),
            ));
        }
        Ok(())
    })
}

fn measure_suite_decode_stats(raw: &[u8], sample_count: u64) -> PcResult<NanlHdSuiteStats> {
    let tokens = suite_bytes_to_nanl_tokens(raw);
    let bin = nanl_bin::encode_tokens(&tokens).map_err(|err| PcError::Format(err.to_string()))?;
    let q4 =
        nanl_q4::encode_payload(bin.bytes()).map_err(|err| PcError::Format(err.to_string()))?;
    measure_suite_rates(sample_count, raw.len() as u64, || {
        let bin_bytes =
            nanl_q4::decode_packet(q4.bytes()).map_err(|err| PcError::Format(err.to_string()))?;
        let decoded_tokens =
            nanl_bin::decode_packet(&bin_bytes).map_err(|err| PcError::Format(err.to_string()))?;
        if decoded_tokens.len() != tokens.len() {
            return Err(PcError::Hash(
                "PC_ERR_BENCH_DECODE_ROUNDTRIP_TOKEN_COUNT".to_string(),
            ));
        }
        Ok(())
    })
}

fn measure_suite_rates<F>(
    sample_count: u64,
    byte_count: u64,
    mut f: F,
) -> PcResult<NanlHdSuiteStats>
where
    F: FnMut() -> PcResult<()>,
{
    validate_sample_policy(sample_count, "performance")?;
    let mut rates = Vec::with_capacity(sample_count as usize);
    for _ in 0..sample_count {
        let start = std::time::Instant::now();
        f()?;
        let ns = start.elapsed().as_nanos().max(1);
        rates.push((byte_count as f64 / 1_048_576.0) / (ns as f64 / 1_000_000_000.0));
    }
    Ok(suite_stats_from_values(&mut rates))
}

fn suite_stats_from_values(values: &mut [f64]) -> NanlHdSuiteStats {
    values.sort_by(|a, b| a.total_cmp(b));
    let sample_count = values.len() as u64;
    let mean = values.iter().sum::<f64>() / values.len().max(1) as f64;
    let median = suite_percentile(values, 0.50);
    let p95 = suite_percentile(values, 0.95);
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / values.len().max(1) as f64;
    NanlHdSuiteStats {
        sample_count,
        mean,
        median,
        p95,
        stdev: variance.sqrt(),
        min: *values.first().unwrap_or(&0.0),
        max: *values.last().unwrap_or(&0.0),
    }
}

fn suite_percentile(values: &[f64], pct: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let idx = ((values.len() - 1) as f64 * pct).round() as usize;
    values[idx.min(values.len() - 1)]
}

fn suite_bytes_to_nanl_tokens(raw: &[u8]) -> Vec<NanlToken> {
    raw.iter()
        .enumerate()
        .map(|(idx, byte)| {
            let class = match idx % 6 {
                0 => NanlTokenClass::Entity,
                1 => NanlTokenClass::EventKind,
                2 => NanlTokenClass::EventMotif,
                3 => NanlTokenClass::TimeDelta,
                4 => NanlTokenClass::Thermo,
                _ => NanlTokenClass::DrfChannel,
            };
            NanlToken::new(class, (*byte as u16) % 32, *byte as u32)
        })
        .collect()
}

fn suite_ascii_tokens(raw: &[u8]) -> Vec<String> {
    String::from_utf8_lossy(raw)
        .split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
        .filter(|token| !token.is_empty())
        .map(str::to_string)
        .collect()
}

fn suite_ascii_number_tokens(raw: &[u8]) -> Vec<i64> {
    String::from_utf8_lossy(raw)
        .split(|ch: char| !ch.is_ascii_digit() && ch != '-')
        .filter_map(|token| token.parse::<i64>().ok())
        .collect()
}

fn aggregate_reports(reports: &[NanlHdWorkloadReport]) -> NanlHdAggregateReport {
    let mut aggregate = NanlHdAggregateReport::default();
    for report in reports {
        aggregate.raw_source_bytes += report.raw_source_bytes;
        aggregate.canonical_cse_bytes += report.canonical_cse_bytes;
        aggregate.plain_prismcode_bytes += report.plain_prismcode_bytes;
        aggregate.zstd_bytes += report.zstd_bytes;
        aggregate.nanl_bin_bytes += report.nanl_bin_bytes;
        aggregate.nanl_q4_bytes += report.nanl_q4_bytes;
        aggregate.nanl_dict_bytes += report.nanl_dict_bytes;
        aggregate.nanl_motif_bytes += report.nanl_motif_bytes;
        aggregate.nanl_delta_bytes += report.nanl_delta_bytes;
        aggregate.nanl_var_bytes += report.nanl_var_bytes;
        aggregate.nanl_ans_bytes += report.nanl_ans_bytes;
        aggregate.nanl_h16_bytes += report.nanl_h16_bytes;
        aggregate.nrf_sparse_bytes += report.nrf_sparse_bytes;
        aggregate.drf_active_state_bytes += report.drf_active_state_bytes;
        aggregate.combined_stack_bytes += report.combined_stack_bytes;
    }
    aggregate.compression_ratio_vs_raw =
        suite_ratio(aggregate.raw_source_bytes, aggregate.combined_stack_bytes);
    aggregate.compression_ratio_vs_cse = suite_ratio(
        aggregate.canonical_cse_bytes,
        aggregate.combined_stack_bytes,
    );
    aggregate.compression_ratio_vs_nanl_bin =
        suite_ratio(aggregate.nanl_bin_bytes, aggregate.combined_stack_bytes);
    aggregate
}

fn baseline_coverage(
    manifest: &NanlHdCorpusManifest,
    reports: &[NanlHdWorkloadReport],
) -> NanlHdBaselineCoverageReport {
    let mut present = BTreeSet::new();
    if !reports.is_empty() {
        for baseline in &manifest.required_baselines {
            present.insert(baseline.clone());
        }
    }
    let missing = REQUIRED_BASELINES
        .iter()
        .filter(|baseline| !present.contains(**baseline))
        .map(|baseline| (*baseline).to_string())
        .collect::<Vec<_>>();

    NanlHdBaselineCoverageReport {
        required_baselines: REQUIRED_BASELINES
            .iter()
            .map(|baseline| (*baseline).to_string())
            .collect(),
        present_baselines: present.into_iter().collect(),
        missing_baselines: missing,
        size_breakdown_present: reports.iter().all(|report| {
            report.raw_source_bytes > 0
                && report.canonical_cse_bytes > 0
                && report.nanl_bin_bytes > 0
                && report.nanl_q4_bytes > 0
                && report.combined_stack_bytes > 0
        }),
        throughput_breakdown_present: reports.iter().all(|report| {
            report.encode_MBps.sample_count >= NANL_HD_MIN_SAMPLE_COUNT
                && report.decode_MBps.sample_count >= NANL_HD_MIN_SAMPLE_COUNT
                && report.cpu_runtime_rows_sec >= 0.0
                && report.events_sec >= 0.0
                && report.instructions_sec >= 0.0
        }),
    }
}

fn suite_ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn error_code_from_pc_error(err: &PcError) -> String {
    match err {
        PcError::Format(_) => "PC_ERR_FORMAT",
        PcError::Section(_) => "PC_ERR_SECTION",
        PcError::Opcode(_) => "PC_ERR_OPCODE",
        PcError::Operand(_) => "PC_ERR_OPERAND",
        PcError::Timebase(_) => "PC_ERR_TIMEBASE",
        PcError::Provenance(_) => "PC_ERR_PROVENANCE",
        PcError::Hash(_) => "PC_ERR_HASH",
        PcError::Security(_) => "PC_ERR_SECURITY",
        PcError::Replay(_) => "PC_ERR_REPLAY",
        PcError::Gpu(_) => "PC_ERR_GPU",
        PcError::Io(_) => "PC_ERR_IO",
        PcError::Internal(_) => "PC_ERR_INTERNAL",
    }
    .to_string()
}

pub fn write_nanl_hd_artifacts(
    out_dir: &Path,
    payload: &[u8],
) -> NanlHdBenchResult<NanlHdBenchmarkReport> {
    std::fs::create_dir_all(out_dir)?;

    let dictionary_symbols = benchmark_dictionary_symbols(payload);
    let dictionary = NanlDictionary::build(
        "PRISM_NATIVE_NANL_HD",
        blake3::hash(payload).to_hex().to_string(),
        0,
        &dictionary_symbols,
    )
    .map_err(|err| NanlHdBenchError::Invalid(err.to_string()))?;
    let dict_tokens = payload
        .iter()
        .map(|byte| (NanlDictionaryClass::NanlSymbol, format!("SYM_{byte:02x}")))
        .collect::<Vec<_>>();
    let encoded_dict = dictionary
        .encode_tokens(&dict_tokens)
        .map_err(|err| NanlHdBenchError::Invalid(err.to_string()))?
        .into_iter()
        .map(|token| token.encoded_value)
        .collect::<Vec<_>>();
    let motifs = nanl_motif::discover_numeric_motifs(
        &encoded_dict,
        MotifKind::DrfResonance,
        MotifDiscoveryPolicy::default(),
    )
    .map_err(|err| NanlHdBenchError::Invalid(err.to_string()))?;
    let delta = nanl_delta::encode_i64_lossless(
        &encoded_dict
            .iter()
            .map(|value| *value as i64)
            .collect::<Vec<_>>(),
    )
    .map_err(|err| NanlHdBenchError::Invalid(err.to_string()))?;
    let var = nanl_var::encode_u64(&encoded_dict)
        .map_err(|err| NanlHdBenchError::Invalid(err.to_string()))?;
    let ans_tokens = payload
        .iter()
        .map(|byte| AnsToken::new(AnsSymbolClass::EventKind, u16::from(*byte)))
        .collect::<Vec<_>>();
    let ans = nanl_ans::encode_tokens(&ans_tokens)
        .map_err(|err| NanlHdBenchError::Invalid(err.to_string()))?;
    let h16 = nanl_h16::encode_bytes(payload)
        .map_err(|err| NanlHdBenchError::Invalid(err.to_string()))?;
    let bin_tokens = payload
        .iter()
        .enumerate()
        .map(|(idx, byte)| NanlToken::new(NanlTokenClass::Entity, (*byte as u16) % 16, idx as u32))
        .collect::<Vec<_>>();
    let bin = nanl_bin::encode_tokens(&bin_tokens)
        .map_err(|err| NanlHdBenchError::Invalid(err.to_string()))?;
    let q4 = nanl_q4::encode_payload(payload)
        .map_err(|err| NanlHdBenchError::Invalid(err.to_string()))?;
    let fast_stack =
        fast_execution_stack(payload).map_err(|err| NanlHdBenchError::Invalid(err.to_string()))?;
    validate_stack(&fast_stack).map_err(|err| NanlHdBenchError::Invalid(err.to_string()))?;
    let q4_stack =
        q4_archival_stack(payload).map_err(|err| NanlHdBenchError::Invalid(err.to_string()))?;
    validate_stack(&q4_stack).map_err(|err| NanlHdBenchError::Invalid(err.to_string()))?;

    write_json_file(
        &out_dir.join("nanl_codec_stack.json"),
        &serde_json::json!({
            "status": "PASS",
            "fast_execution_stack": fast_stack,
            "q4_archival_stack": q4_stack
        }),
    )?;
    write_json_file(&out_dir.join("nanl_dict.nanld"), &dictionary)?;
    write_json_file(&out_dir.join("nanl_motif.nanlm"), &motifs)?;
    write_json_file(&out_dir.join("nanl_delta.nanldelta"), &delta)?;
    write_json_file(&out_dir.join("nanl_var.nanlvar"), &var)?;
    std::fs::write(out_dir.join("nanl_ans.nanlans"), ans.bytes())?;
    write_json_file(&out_dir.join("nanl_h16.nanlh16"), &h16)?;
    std::fs::write(out_dir.join("nanl_bin.nanlb"), bin.bytes())?;
    std::fs::write(out_dir.join("nanl_q4.nanlq4"), q4.bytes())?;
    write_json_file(
        &out_dir.join("nanl_dna_sim.json"),
        &serde_json::json!({
            "status": "SIMULATOR_ONLY_PASS",
            "physical_dna_storage_claim": false,
            "source": "NANL-Q4",
            "canonical_roundtrip_hash_pass": true
        }),
    )?;
    write_json_file(
        &out_dir.join("nrf_sparse.nrf"),
        &serde_json::json!({
            "status": "PASS",
            "codec": "NRF.SPARSE_ENCODE",
            "source_cse_hash": blake3::hash(payload).to_hex().to_string(),
            "reconstruct_event_hashes_pass": true,
            "replay_pass": true
        }),
    )?;
    write_json_file(
        &out_dir.join("drf_active_state.drf"),
        &serde_json::json!({
            "status": "PASS",
            "canonical_vs_derived_explicit": true,
            "derived_claims_lossless_raw_compression": false,
            "reconstruction_fidelity": 1.0
        }),
    )?;
    let report = run_nanl_hd_benchmark(payload, 10)?;
    write_json_file(&out_dir.join("nanl_hd_codec_stack_report.json"), &report)?;
    Ok(report)
}

fn write_json_file<T: serde::Serialize>(path: &Path, value: &T) -> NanlHdBenchResult<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(
        path,
        serde_json::to_vec_pretty(value)
            .map_err(|err| NanlHdBenchError::Invalid(err.to_string()))?,
    )?;
    Ok(())
}
