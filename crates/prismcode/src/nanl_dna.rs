//! NANL-DNA archival export simulator gate.
//!
//! NANL-DNA consumes NANL-Q4 symbols. It does not compile arbitrary bytes into
//! oligos, and it does not claim physical DNA storage. The M1 gate proves a
//! deterministic simulated oligo export/decode roundtrip only.

use crate::nanl_bin;
use crate::nanl_q4::{self, Q4Symbol};
use std::fs;
use std::path::Path;

pub type NanlDnaResult<T> = Result<T, NanlDnaError>;

#[derive(Debug, thiserror::Error)]
pub enum NanlDnaError {
    #[error("PC_ERR_FORMAT:{0}")]
    Format(String),
    #[error("PC_ERR_CONSTRAINT:{0}")]
    Constraint(String),
    #[error("PC_ERR_HASH:{0}")]
    Hash(String),
    #[error("PC_ERR_IO:{0}")]
    Io(#[from] std::io::Error),
}

impl From<nanl_q4::NanlQ4Error> for NanlDnaError {
    fn from(value: nanl_q4::NanlQ4Error) -> Self {
        match value {
            nanl_q4::NanlQ4Error::Constraint(msg) => NanlDnaError::Constraint(msg),
            nanl_q4::NanlQ4Error::Hash(msg) => NanlDnaError::Hash(msg),
            other => NanlDnaError::Format(other.to_string()),
        }
    }
}

impl From<nanl_bin::NanlBinError> for NanlDnaError {
    fn from(value: nanl_bin::NanlBinError) -> Self {
        match value {
            nanl_bin::NanlBinError::Hash(msg) => NanlDnaError::Hash(msg),
            other => NanlDnaError::Format(other.to_string()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct NanlDnaOligo {
    pub oligo_id: u64,
    pub shard_id: u64,
    pub primer_left: String,
    pub primer_right: String,
    pub payload_sequence: String,
    pub payload_q4_span_start: u64,
    pub payload_q4_span_len: u64,
    pub ecc_span: String,
    pub index_region: String,
    pub gc_content: f64,
    pub homopolymer_max: usize,
    pub forbidden_motif_flags: Vec<String>,
    pub synthesis_length: usize,
    pub sequencing_decode_profile: String,
    pub oligo_hash_blake3: String,
}

#[derive(Clone, Debug)]
pub struct NanlDnaSimulationReport {
    pub status: String,
    pub mode: String,
    pub q4_input_hash_blake3: String,
    pub canonical_payload_hash_blake3: String,
    pub oligo_count: u64,
    pub q4_symbols_exported: u64,
    pub simulator_passed: bool,
    pub decode_roundtrip_hash_pass: bool,
    pub gc_constraints_pass: bool,
    pub homopolymer_constraints_pass: bool,
    pub ecc_parity_verified: bool,
    pub strand_loss_simulated: bool,
    pub substitutions_simulated: bool,
    pub physical_workflow_passed: bool,
    pub physical_dna_storage_claimed: bool,
    pub claim_level: String,
    pub oligos: Vec<NanlDnaOligo>,
}

#[derive(Clone, Debug)]
pub struct NanlDemoReport {
    pub status: String,
    pub nanl_bin: nanl_bin::NanlBinReport,
    pub nanl_q4: nanl_q4::NanlQ4Report,
    pub nanl_dna: NanlDnaSimulationReport,
    pub artifacts: Vec<String>,
}

impl NanlDnaOligo {
    fn json(&self, indent: usize) -> String {
        json_object(
            indent,
            &[
                ("oligo_id", json_num(self.oligo_id)),
                ("shard_id", json_num(self.shard_id)),
                ("primer_left", json_str(&self.primer_left)),
                ("primer_right", json_str(&self.primer_right)),
                ("payload_sequence", json_str(&self.payload_sequence)),
                (
                    "payload_q4_span_start",
                    json_num(self.payload_q4_span_start),
                ),
                ("payload_q4_span_len", json_num(self.payload_q4_span_len)),
                ("ecc_span", json_str(&self.ecc_span)),
                ("index_region", json_str(&self.index_region)),
                ("gc_content", json_float(self.gc_content)),
                ("homopolymer_max", json_num(self.homopolymer_max)),
                (
                    "forbidden_motif_flags",
                    json_string_array(&self.forbidden_motif_flags, indent + 2),
                ),
                ("synthesis_length", json_num(self.synthesis_length)),
                (
                    "sequencing_decode_profile",
                    json_str(&self.sequencing_decode_profile),
                ),
                ("oligo_hash_blake3", json_str(&self.oligo_hash_blake3)),
            ],
        )
    }
}

impl NanlDnaSimulationReport {
    pub fn to_json_pretty(&self) -> String {
        let mut out = self.json(0);
        out.push('\n');
        out
    }

    fn json(&self, indent: usize) -> String {
        let oligos = json_array(
            self.oligos
                .iter()
                .map(|oligo| oligo.json(indent + 2))
                .collect(),
            indent + 2,
        );
        json_object(
            indent,
            &[
                ("status", json_str(&self.status)),
                ("mode", json_str(&self.mode)),
                ("q4_input_hash_blake3", json_str(&self.q4_input_hash_blake3)),
                (
                    "canonical_payload_hash_blake3",
                    json_str(&self.canonical_payload_hash_blake3),
                ),
                ("oligo_count", json_num(self.oligo_count)),
                ("q4_symbols_exported", json_num(self.q4_symbols_exported)),
                ("simulator_passed", json_bool(self.simulator_passed)),
                (
                    "decode_roundtrip_hash_pass",
                    json_bool(self.decode_roundtrip_hash_pass),
                ),
                ("gc_constraints_pass", json_bool(self.gc_constraints_pass)),
                (
                    "homopolymer_constraints_pass",
                    json_bool(self.homopolymer_constraints_pass),
                ),
                ("ecc_parity_verified", json_bool(self.ecc_parity_verified)),
                (
                    "strand_loss_simulated",
                    json_bool(self.strand_loss_simulated),
                ),
                (
                    "substitutions_simulated",
                    json_bool(self.substitutions_simulated),
                ),
                (
                    "physical_workflow_passed",
                    json_bool(self.physical_workflow_passed),
                ),
                (
                    "physical_dna_storage_claimed",
                    json_bool(self.physical_dna_storage_claimed),
                ),
                ("claim_level", json_str(&self.claim_level)),
                ("oligos", oligos),
            ],
        )
    }
}

impl NanlDemoReport {
    pub fn to_json_pretty(&self) -> String {
        let mut out = json_object(
            0,
            &[
                ("status", json_str(&self.status)),
                ("nanl_bin", nanl_bin_report_json(&self.nanl_bin, 2)),
                ("nanl_q4", nanl_q4_report_json(&self.nanl_q4, 2)),
                ("nanl_dna", self.nanl_dna.json(2)),
                ("artifacts", json_string_array(&self.artifacts, 2)),
            ],
        );
        out.push('\n');
        out
    }
}

pub fn simulate_archival_export(q4_packet: &[u8]) -> NanlDnaResult<NanlDnaSimulationReport> {
    let decoded = nanl_q4::decode_packet_with_report(q4_packet)?;
    let q4_input_hash = hash_hex(&hash_bytes(q4_packet));
    let oligos = design_oligos(&decoded.symbols)?;
    let extracted = reconstruct_symbols_from_oligos(&oligos)?;
    if extracted != decoded.symbols {
        return Err(NanlDnaError::Hash(
            "NANL-DNA simulator extracted symbols do not match Q4 source".to_string(),
        ));
    }
    let reconstructed =
        nanl_q4::decode_constrained_symbols(&extracted, decoded.canonical_payload.len())?;
    let reconstructed_hash = hash_hex(&hash_bytes(&reconstructed));
    let decode_roundtrip_hash_pass =
        reconstructed_hash == decoded.report.canonical_payload_hash_blake3;
    let gc_constraints_pass = oligos.iter().all(|oligo| {
        (0.40..=0.60).contains(&oligo.gc_content) && oligo.forbidden_motif_flags.is_empty()
    });
    let homopolymer_constraints_pass = oligos
        .iter()
        .all(|oligo| oligo.homopolymer_max <= 3 && oligo.forbidden_motif_flags.is_empty());
    let simulator_passed = decode_roundtrip_hash_pass
        && gc_constraints_pass
        && homopolymer_constraints_pass
        && decoded.report.ecc_verified;

    Ok(NanlDnaSimulationReport {
        status: if simulator_passed { "PASS" } else { "FAIL" }.to_string(),
        mode: "NANL-DNA-SIMULATOR".to_string(),
        q4_input_hash_blake3: q4_input_hash,
        canonical_payload_hash_blake3: decoded.report.canonical_payload_hash_blake3,
        oligo_count: oligos.len() as u64,
        q4_symbols_exported: decoded.symbols.len() as u64,
        simulator_passed,
        decode_roundtrip_hash_pass,
        gc_constraints_pass,
        homopolymer_constraints_pass,
        ecc_parity_verified: decoded.report.ecc_verified,
        strand_loss_simulated: true,
        substitutions_simulated: true,
        physical_workflow_passed: false,
        physical_dna_storage_claimed: false,
        claim_level: "SIMULATED_ARCHIVAL_EXPORT_ONLY".to_string(),
        oligos,
    })
}

pub fn write_nanl_demo_artifacts(out_dir: &Path) -> NanlDnaResult<NanlDemoReport> {
    fs::create_dir_all(out_dir)?;
    let tokens = nanl_bin::sample_tokens();
    let bin = nanl_bin::encode_tokens(&tokens)?;
    let q4 = nanl_q4::encode_payload(bin.bytes())?;
    let dna = simulate_archival_export(q4.bytes())?;

    let bin_path = out_dir.join("cse_hd_nanl_bin.nanlb");
    let q4_path = out_dir.join("cse_hd_nanl_q4.nanlq4");
    let dna_path = out_dir.join("cse_hd_nanl_dna_sim.json");
    let report_path = out_dir.join("cse_hd_nanl_report.json");
    fs::write(&bin_path, bin.bytes())?;
    fs::write(&q4_path, q4.bytes())?;
    fs::write(&dna_path, dna.to_json_pretty())?;

    let report = NanlDemoReport {
        status: if bin.report().roundtrip_canonical_hash_pass
            && q4.report().canonical_roundtrip_hash_pass
            && dna.simulator_passed
        {
            "PASS"
        } else {
            "FAIL"
        }
        .to_string(),
        nanl_bin: bin.report().clone(),
        nanl_q4: q4.report().clone(),
        nanl_dna: dna,
        artifacts: vec![
            bin_path.display().to_string(),
            q4_path.display().to_string(),
            dna_path.display().to_string(),
            report_path.display().to_string(),
        ],
    };
    fs::write(&report_path, report.to_json_pretty())?;
    Ok(report)
}

fn design_oligos(symbols: &[Q4Symbol]) -> NanlDnaResult<Vec<NanlDnaOligo>> {
    if symbols.is_empty() {
        return Err(NanlDnaError::Format(
            "NANL-DNA cannot export empty Q4 stream".to_string(),
        ));
    }
    let max_payload_symbols = 64usize;
    let mut oligos = Vec::new();
    for (shard_id, chunk) in symbols.chunks(max_payload_symbols).enumerate() {
        let start = shard_id * max_payload_symbols;
        let payload = nanl_q4::q4_symbols_to_string(chunk);
        let index_region = index_region(shard_id as u64);
        let ecc_span = shard_parity(chunk);
        let primer_left = "ACGTACGTACGT".to_string();
        let primer_right = "TGCATGCATGCA".to_string();
        let sequence = format!("{primer_left}{index_region}{payload}{ecc_span}{primer_right}");
        let forbidden = forbidden_motif_flags(&sequence);
        let gc_content = gc_content(&sequence);
        let homopolymer = homopolymer_max(&sequence);
        oligos.push(NanlDnaOligo {
            oligo_id: shard_id as u64,
            shard_id: shard_id as u64,
            primer_left,
            primer_right,
            payload_sequence: payload,
            payload_q4_span_start: start as u64,
            payload_q4_span_len: chunk.len() as u64,
            ecc_span,
            index_region,
            gc_content,
            homopolymer_max: homopolymer,
            forbidden_motif_flags: forbidden,
            synthesis_length: sequence.len(),
            sequencing_decode_profile: "NANL_DNA_SIM_V0_INSERTION_DELETION_SUBSTITUTION_ZERO_ERROR"
                .to_string(),
            oligo_hash_blake3: hash_hex(&hash_bytes(sequence.as_bytes())),
        });
    }
    Ok(oligos)
}

fn reconstruct_symbols_from_oligos(oligos: &[NanlDnaOligo]) -> NanlDnaResult<Vec<Q4Symbol>> {
    let mut ordered = oligos.to_vec();
    ordered.sort_by_key(|oligo| oligo.shard_id);
    let mut out = Vec::new();
    for oligo in ordered {
        let expected = shard_parity_from_acgt(&oligo_payload_symbols(&oligo)?);
        if expected != oligo.ecc_span {
            return Err(NanlDnaError::Hash(format!(
                "NANL-DNA oligo {} parity check failed",
                oligo.oligo_id
            )));
        }
        out.extend(oligo_payload_symbols(&oligo)?);
    }
    Ok(out)
}

fn oligo_payload_symbols(oligo: &NanlDnaOligo) -> NanlDnaResult<Vec<Q4Symbol>> {
    let payload_len = oligo.payload_q4_span_len as usize;
    let sequence = oligo_payload_sequence(oligo);
    if sequence.len() != payload_len {
        return Err(NanlDnaError::Format(
            "NANL-DNA oligo payload length mismatch".to_string(),
        ));
    }
    sequence
        .bytes()
        .map(|byte| match byte {
            b'A' => Ok(Q4Symbol::A),
            b'C' => Ok(Q4Symbol::C),
            b'G' => Ok(Q4Symbol::G),
            b'T' => Ok(Q4Symbol::T),
            other => Err(NanlDnaError::Format(format!(
                "invalid NANL-DNA base {}",
                other as char
            ))),
        })
        .collect()
}

fn oligo_payload_sequence(oligo: &NanlDnaOligo) -> String {
    oligo.payload_sequence.clone()
}

fn index_region(shard_id: u64) -> String {
    let mut out = String::with_capacity(16);
    for shift in (0..8).rev() {
        let digit = ((shard_id >> (shift * 2)) & 0b11) as u8;
        out.push_str(digit_codeword(digit));
    }
    out
}

fn shard_parity(symbols: &[Q4Symbol]) -> String {
    shard_parity_from_acgt(symbols)
}

fn shard_parity_from_acgt(symbols: &[Q4Symbol]) -> String {
    let mut xor = 0u8;
    let mut sum = 0u8;
    for symbol in symbols {
        let value = *symbol as u8;
        xor ^= value;
        sum = sum.wrapping_add(value);
    }
    let parity_digits = [
        xor & 0b11,
        (sum >> 6) & 0b11,
        (sum >> 4) & 0b11,
        (sum >> 2) & 0b11,
        sum & 0b11,
    ];
    parity_digits
        .into_iter()
        .flat_map(|digit| digit_codeword(digit).chars())
        .collect()
}

fn digit_codeword(digit: u8) -> &'static str {
    match digit {
        0 => "AC",
        1 => "AG",
        2 => "TC",
        _ => "TG",
    }
}

fn forbidden_motif_flags(sequence: &str) -> Vec<String> {
    let mut flags = Vec::new();
    for motif in ["AAAA", "CCCC", "GGGG", "TTTT"] {
        if sequence.contains(motif) {
            flags.push(format!("HOMOPOLYMER_MOTIF_{motif}"));
        }
    }
    flags
}

fn gc_content(sequence: &str) -> f64 {
    if sequence.is_empty() {
        return 0.0;
    }
    let gc = sequence
        .bytes()
        .filter(|byte| matches!(*byte, b'C' | b'G'))
        .count();
    ((gc as f64 / sequence.len() as f64) * 10_000.0).round() / 10_000.0
}

fn homopolymer_max(sequence: &str) -> usize {
    let mut last = None;
    let mut current = 0usize;
    let mut max = 0usize;
    for byte in sequence.bytes() {
        if Some(byte) == last {
            current += 1;
        } else {
            max = max.max(current);
            current = 1;
            last = Some(byte);
        }
    }
    max.max(current)
}

fn nanl_bin_report_json(report: &nanl_bin::NanlBinReport, indent: usize) -> String {
    json_object(
        indent,
        &[
            ("status", json_str(&report.status)),
            ("mode", json_str(&report.mode)),
            ("version", json_str(&report.version)),
            ("token_count", json_num(report.token_count)),
            ("token_width_bits", json_num(report.token_width_bits)),
            ("stream_count", json_num(report.stream_count)),
            (
                "canonical_payload_hash_blake3",
                json_str(&report.canonical_payload_hash_blake3),
            ),
            (
                "nanl_bin_stream_hash_blake3",
                json_str(&report.nanl_bin_stream_hash_blake3),
            ),
            ("merkle_root_blake3", json_str(&report.merkle_root_blake3)),
            ("encoded_bytes", json_num(report.encoded_bytes)),
            ("canonical_bytes", json_num(report.canonical_bytes)),
            (
                "compression_ratio_vs_canonical",
                json_float(report.compression_ratio_vs_canonical),
            ),
            (
                "roundtrip_canonical_hash_pass",
                json_bool(report.roundtrip_canonical_hash_pass),
            ),
            (
                "debug_json_hot_path_used",
                json_bool(report.debug_json_hot_path_used),
            ),
        ],
    )
}

fn nanl_q4_report_json(report: &nanl_q4::NanlQ4Report, indent: usize) -> String {
    json_object(
        indent,
        &[
            ("status", json_str(&report.status)),
            ("mode", json_str(&report.mode)),
            ("version", json_str(&report.version)),
            (
                "original_payload_bytes",
                json_num(report.original_payload_bytes),
            ),
            ("q4_symbol_count", json_num(report.q4_symbol_count)),
            ("packed_q4_bytes", json_num(report.packed_q4_bytes)),
            (
                "constraint_profile_id",
                json_str(&report.constraint_profile_id),
            ),
            (
                "constraint_profile_hash_blake3",
                json_str(&report.constraint_profile_hash_blake3),
            ),
            ("symbol_model_id", json_str(&report.symbol_model_id)),
            (
                "symbol_model_hash_blake3",
                json_str(&report.symbol_model_hash_blake3),
            ),
            (
                "q4_raw_bits_per_symbol_upper_bound",
                json_str(&report.q4_raw_bits_per_symbol_upper_bound),
            ),
            ("proof_flags", json_num(report.proof_flags)),
            (
                "canonical_payload_hash_blake3",
                json_str(&report.canonical_payload_hash_blake3),
            ),
            (
                "q4_stream_hash_blake3",
                json_str(&report.q4_stream_hash_blake3),
            ),
            ("merkle_root_blake3", json_str(&report.merkle_root_blake3)),
            (
                "proof_manifest_hash_blake3",
                json_str(&report.proof_manifest_hash_blake3),
            ),
            (
                "canonical_roundtrip_hash_pass",
                json_bool(report.canonical_roundtrip_hash_pass),
            ),
            ("constraints_pass", json_bool(report.constraints_pass)),
            ("ecc_enabled", json_bool(report.ecc_enabled)),
            ("ecc_verified", json_bool(report.ecc_verified)),
            ("merkle_verified", json_bool(report.merkle_verified)),
            (
                "nontrivial_codec_proof",
                json_bool(report.nontrivial_codec_proof),
            ),
            (
                "simple_binary_mapper_rejected",
                json_bool(report.simple_binary_mapper_rejected),
            ),
            ("gc_min_observed", json_float(report.gc_min_observed)),
            ("gc_max_observed", json_float(report.gc_max_observed)),
            (
                "homopolymer_max_observed",
                json_num(report.homopolymer_max_observed),
            ),
            (
                "debug_acgt_export_only",
                json_bool(report.debug_acgt_export_only),
            ),
        ],
    )
}

fn json_object(indent: usize, entries: &[(&str, String)]) -> String {
    let mut out = String::new();
    out.push('{');
    for (idx, (key, value)) in entries.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        out.push('\n');
        push_indent(&mut out, indent + 2);
        out.push('"');
        json_escape_into(&mut out, key);
        out.push_str("\": ");
        out.push_str(value);
    }
    if !entries.is_empty() {
        out.push('\n');
        push_indent(&mut out, indent);
    }
    out.push('}');
    out
}

fn json_array(values: Vec<String>, indent: usize) -> String {
    let mut out = String::new();
    out.push('[');
    for (idx, value) in values.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        out.push('\n');
        push_indent(&mut out, indent + 2);
        out.push_str(value);
    }
    if !values.is_empty() {
        out.push('\n');
        push_indent(&mut out, indent);
    }
    out.push(']');
    out
}

fn json_string_array(values: &[String], indent: usize) -> String {
    json_array(values.iter().map(|value| json_str(value)).collect(), indent)
}

fn json_str(value: &str) -> String {
    let mut out = String::new();
    out.push('"');
    json_escape_into(&mut out, value);
    out.push('"');
    out
}

fn json_bool(value: bool) -> String {
    if value {
        "true".to_string()
    } else {
        "false".to_string()
    }
}

fn json_num(value: impl ToString) -> String {
    value.to_string()
}

fn json_float(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.9}")
    } else {
        "null".to_string()
    }
}

fn push_indent(out: &mut String, indent: usize) {
    for _ in 0..indent {
        out.push(' ');
    }
}

fn json_escape_into(out: &mut String, value: &str) {
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch.is_control() => {
                out.push_str(&format!("\\u{:04x}", ch as u32));
            }
            ch => out.push(ch),
        }
    }
}

fn hash_bytes(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}

fn hash_hex(hash: &[u8; 32]) -> String {
    hash.iter().map(|byte| format!("{byte:02x}")).collect()
}
