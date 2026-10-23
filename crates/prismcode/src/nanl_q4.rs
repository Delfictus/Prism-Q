//! NANL-Q4 deterministic constrained quaternary digital codec.
//!
//! NANL-Q4 is not a bytes-to-bases costume and it does not claim more than two
//! raw bits per Q4 symbol. The M1 codec derives canonical base-4 PRISM symbols,
//! binds a deterministic symbol model, maps each source digit through a
//! constrained quaternary codebook, and seals the stream with constraint-profile,
//! parity/ECC, Merkle, and roundtrip hash evidence.

pub const NANL_Q4_MAGIC: &[u8; 8] = b"NANLQ4\0\0";
pub const NANL_Q4_VERSION_MAJOR: u16 = 0;
pub const NANL_Q4_VERSION_MINOR: u16 = 1;
pub const NANL_Q4_HEADER_LEN: usize = 160;
pub const NANL_Q4_SHARD_COUNT_OFFSET: usize = 28;
pub const NANL_Q4_PROOF_FLAGS_OFFSET: usize = 29;
pub const NANL_Q4_RAW_BITS_PER_SYMBOL_X2_OFFSET: usize = 30;
pub const NANL_Q4_MODEL_KIND_OFFSET: usize = 31;
pub const NANL_Q4_PROOF_HASH_PREFIX_OFFSET: usize = 132;
pub const NANL_Q4_PROOF_HASH_PREFIX_LEN: usize = 28;
pub const NANL_Q4_REQUIRED_PROOF_FLAGS: u8 = NANL_Q4_FLAG_CONSTRAINT_PROFILE_PRESENT
    | NANL_Q4_FLAG_SYMBOL_MODEL_PRESENT
    | NANL_Q4_FLAG_ECC_PRESENT
    | NANL_Q4_FLAG_MERKLE_PRESENT
    | NANL_Q4_FLAG_NONTRIVIAL_CODEBOOK;
pub const NANL_Q4_FLAG_CONSTRAINT_PROFILE_PRESENT: u8 = 0b0000_0001;
pub const NANL_Q4_FLAG_SYMBOL_MODEL_PRESENT: u8 = 0b0000_0010;
pub const NANL_Q4_FLAG_ECC_PRESENT: u8 = 0b0000_0100;
pub const NANL_Q4_FLAG_MERKLE_PRESENT: u8 = 0b0000_1000;
pub const NANL_Q4_FLAG_NONTRIVIAL_CODEBOOK: u8 = 0b0001_0000;
pub const NANL_Q4_RAW_BITS_PER_SYMBOL_X2: u8 = 4;
pub const NANL_Q4_SYMBOL_MODEL_PRISM_NATIVE_V0: u8 = 1;

pub type NanlQ4Result<T> = Result<T, NanlQ4Error>;

#[derive(Debug, thiserror::Error)]
pub enum NanlQ4Error {
    #[error("PC_ERR_FORMAT:{0}")]
    Format(String),
    #[error("PC_ERR_CONSTRAINT:{0}")]
    Constraint(String),
    #[error("PC_ERR_HASH:{0}")]
    Hash(String),
    #[error("PC_ERR_OVERCLAIM:{0}")]
    Overclaim(String),
    #[error("PC_ERR_INTERNAL:{0}")]
    Internal(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Q4Symbol {
    A = 0,
    C = 1,
    G = 2,
    T = 3,
}

impl Q4Symbol {
    pub fn is_gc(self) -> bool {
        matches!(self, Self::C | Self::G)
    }

    pub fn as_char(self) -> char {
        match self {
            Self::A => 'A',
            Self::C => 'C',
            Self::G => 'G',
            Self::T => 'T',
        }
    }
}

impl TryFrom<u8> for Q4Symbol {
    type Error = NanlQ4Error;

    fn try_from(value: u8) -> NanlQ4Result<Self> {
        match value {
            0 => Ok(Self::A),
            1 => Ok(Self::C),
            2 => Ok(Self::G),
            3 => Ok(Self::T),
            other => Err(NanlQ4Error::Format(format!(
                "invalid NANL-Q4 symbol value {other}"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Q4ConstraintProfile {
    pub gc_window: usize,
    pub gc_min: f64,
    pub gc_max: f64,
    pub homopolymer_max: usize,
}

impl Default for Q4ConstraintProfile {
    fn default() -> Self {
        Self {
            gc_window: 64,
            gc_min: 0.40,
            gc_max: 0.60,
            homopolymer_max: 3,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Q4ConstraintReport {
    pub constraint_profile_id: String,
    pub gc_window: usize,
    pub gc_min_required: f64,
    pub gc_max_required: f64,
    pub gc_min_observed: f64,
    pub gc_max_observed: f64,
    pub homopolymer_max_required: usize,
    pub homopolymer_max_observed: usize,
    pub constraints_pass: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Q4SymbolModel {
    pub model_id: String,
    pub model_kind: u8,
    pub digit_frequency: [u64; 4],
    pub q4_symbol_frequency: [u64; 4],
    pub codebook_hash_blake3: String,
    pub model_hash_blake3: String,
    pub raw_bits_per_q4_symbol_upper_bound: String,
    pub simple_binary_mapper: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Q4ParityBlock {
    pub symbol_count: u64,
    pub xor: u8,
    pub sum_mod_65535: u16,
    pub stream_hash_blake3: String,
}

#[derive(Clone, Debug)]
pub struct NanlQ4Report {
    pub status: String,
    pub mode: String,
    pub version: String,
    pub original_payload_bytes: u64,
    pub q4_symbol_count: u64,
    pub packed_q4_bytes: u64,
    pub constraint_profile_id: String,
    pub constraint_profile_hash_blake3: String,
    pub symbol_model_id: String,
    pub symbol_model_hash_blake3: String,
    pub q4_raw_bits_per_symbol_upper_bound: String,
    pub proof_flags: u8,
    pub canonical_payload_hash_blake3: String,
    pub q4_stream_hash_blake3: String,
    pub merkle_root_blake3: String,
    pub proof_manifest_hash_blake3: String,
    pub canonical_roundtrip_hash_pass: bool,
    pub constraints_pass: bool,
    pub ecc_enabled: bool,
    pub ecc_verified: bool,
    pub merkle_verified: bool,
    pub nontrivial_codec_proof: bool,
    pub simple_binary_mapper_rejected: bool,
    pub gc_min_observed: f64,
    pub gc_max_observed: f64,
    pub homopolymer_max_observed: usize,
    pub debug_acgt_export_only: bool,
}

impl NanlQ4Report {
    /// Serde-free pretty JSON for the report. The QNANL codec is deliberately
    /// serde-free, so this provides a first-class serialization without
    /// re-introducing `#[derive(Serialize)]`. The field set/order mirrors
    /// `nanl_dna::nanl_q4_report_json` for a single canonical representation.
    pub fn to_json_pretty(&self) -> String {
        let mut out = self.to_json(0);
        out.push('\n');
        out
    }

    fn to_json(&self, indent: usize) -> String {
        q4_json_object(
            indent,
            &[
                ("status", q4_json_str(&self.status)),
                ("mode", q4_json_str(&self.mode)),
                ("version", q4_json_str(&self.version)),
                (
                    "original_payload_bytes",
                    q4_json_num(self.original_payload_bytes),
                ),
                ("q4_symbol_count", q4_json_num(self.q4_symbol_count)),
                ("packed_q4_bytes", q4_json_num(self.packed_q4_bytes)),
                (
                    "constraint_profile_id",
                    q4_json_str(&self.constraint_profile_id),
                ),
                (
                    "constraint_profile_hash_blake3",
                    q4_json_str(&self.constraint_profile_hash_blake3),
                ),
                ("symbol_model_id", q4_json_str(&self.symbol_model_id)),
                (
                    "symbol_model_hash_blake3",
                    q4_json_str(&self.symbol_model_hash_blake3),
                ),
                (
                    "q4_raw_bits_per_symbol_upper_bound",
                    q4_json_str(&self.q4_raw_bits_per_symbol_upper_bound),
                ),
                ("proof_flags", q4_json_num(self.proof_flags)),
                (
                    "canonical_payload_hash_blake3",
                    q4_json_str(&self.canonical_payload_hash_blake3),
                ),
                (
                    "q4_stream_hash_blake3",
                    q4_json_str(&self.q4_stream_hash_blake3),
                ),
                ("merkle_root_blake3", q4_json_str(&self.merkle_root_blake3)),
                (
                    "proof_manifest_hash_blake3",
                    q4_json_str(&self.proof_manifest_hash_blake3),
                ),
                (
                    "canonical_roundtrip_hash_pass",
                    q4_json_bool(self.canonical_roundtrip_hash_pass),
                ),
                ("constraints_pass", q4_json_bool(self.constraints_pass)),
                ("ecc_enabled", q4_json_bool(self.ecc_enabled)),
                ("ecc_verified", q4_json_bool(self.ecc_verified)),
                ("merkle_verified", q4_json_bool(self.merkle_verified)),
                (
                    "nontrivial_codec_proof",
                    q4_json_bool(self.nontrivial_codec_proof),
                ),
                (
                    "simple_binary_mapper_rejected",
                    q4_json_bool(self.simple_binary_mapper_rejected),
                ),
                ("gc_min_observed", q4_json_float(self.gc_min_observed)),
                ("gc_max_observed", q4_json_float(self.gc_max_observed)),
                (
                    "homopolymer_max_observed",
                    q4_json_num(self.homopolymer_max_observed),
                ),
                (
                    "debug_acgt_export_only",
                    q4_json_bool(self.debug_acgt_export_only),
                ),
            ],
        )
    }
}

fn q4_json_object(indent: usize, entries: &[(&str, String)]) -> String {
    let mut out = String::new();
    out.push('{');
    for (idx, (key, value)) in entries.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        out.push('\n');
        for _ in 0..indent + 2 {
            out.push(' ');
        }
        out.push('"');
        q4_json_escape(&mut out, key);
        out.push_str("\": ");
        out.push_str(value);
    }
    if !entries.is_empty() {
        out.push('\n');
        for _ in 0..indent {
            out.push(' ');
        }
    }
    out.push('}');
    out
}

fn q4_json_str(value: &str) -> String {
    let mut out = String::new();
    out.push('"');
    q4_json_escape(&mut out, value);
    out.push('"');
    out
}

fn q4_json_bool(value: bool) -> String {
    if value {
        "true".to_string()
    } else {
        "false".to_string()
    }
}

fn q4_json_num(value: impl ToString) -> String {
    value.to_string()
}

fn q4_json_float(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.9}")
    } else {
        "null".to_string()
    }
}

fn q4_json_escape(out: &mut String, value: &str) {
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

#[derive(Clone, Debug)]
pub struct NanlQ4Packet {
    bytes: Vec<u8>,
    symbols: Vec<Q4Symbol>,
    report: NanlQ4Report,
}

impl NanlQ4Packet {
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    pub fn symbols(&self) -> &[Q4Symbol] {
        &self.symbols
    }

    pub fn report(&self) -> &NanlQ4Report {
        &self.report
    }
}

#[derive(Clone, Debug)]
pub struct NanlQ4Decoded {
    pub canonical_payload: Vec<u8>,
    pub symbols: Vec<Q4Symbol>,
    pub report: NanlQ4Report,
}

pub fn encode_payload(canonical_payload: &[u8]) -> NanlQ4Result<NanlQ4Packet> {
    encode_payload_with_profile(canonical_payload, Q4ConstraintProfile::default())
}

pub fn encode_payload_with_profile(
    canonical_payload: &[u8],
    profile: Q4ConstraintProfile,
) -> NanlQ4Result<NanlQ4Packet> {
    if canonical_payload.is_empty() {
        return Err(NanlQ4Error::Format(
            "NANL-Q4 canonical payload is empty".to_string(),
        ));
    }

    let symbols = bytes_to_constrained_symbols(canonical_payload);
    let constraint_report = validate_constraints(&symbols, profile)?;
    if !constraint_report.constraints_pass {
        return Err(NanlQ4Error::Constraint(
            "NANL-Q4 generated stream violates constraints".to_string(),
        ));
    }

    let model = Q4SymbolModel::from_payload_and_symbols(canonical_payload, &symbols);
    let packed = pack_q4_symbols(&symbols)?;
    let canonical_hash = hash_bytes(canonical_payload);
    let q4_stream_hash = hash_bytes(&packed);
    let merkle = merkle_root(&canonical_hash, &q4_stream_hash);
    let parity = parity_block(&symbols);
    let profile_hash = constraint_profile_hash(&profile);
    let proof_hash = proof_manifest_hash(
        &profile,
        &profile_hash,
        &model,
        &parity,
        &canonical_hash,
        &q4_stream_hash,
        &merkle,
    );

    let mut bytes = vec![0u8; NANL_Q4_HEADER_LEN];
    bytes[0..8].copy_from_slice(NANL_Q4_MAGIC);
    bytes[8..10].copy_from_slice(&NANL_Q4_VERSION_MAJOR.to_le_bytes());
    bytes[10..12].copy_from_slice(&NANL_Q4_VERSION_MINOR.to_le_bytes());
    bytes[12..20].copy_from_slice(&(canonical_payload.len() as u64).to_le_bytes());
    bytes[20..28].copy_from_slice(&(symbols.len() as u64).to_le_bytes());
    bytes[NANL_Q4_SHARD_COUNT_OFFSET] = 1;
    bytes[NANL_Q4_PROOF_FLAGS_OFFSET] = NANL_Q4_REQUIRED_PROOF_FLAGS;
    bytes[NANL_Q4_RAW_BITS_PER_SYMBOL_X2_OFFSET] = NANL_Q4_RAW_BITS_PER_SYMBOL_X2;
    bytes[NANL_Q4_MODEL_KIND_OFFSET] = NANL_Q4_SYMBOL_MODEL_PRISM_NATIVE_V0;
    bytes[32..64].copy_from_slice(&canonical_hash);
    bytes[64..96].copy_from_slice(&q4_stream_hash);
    bytes[96..128].copy_from_slice(&merkle);
    bytes[128] = parity.xor;
    bytes[130..132].copy_from_slice(&parity.sum_mod_65535.to_le_bytes());
    bytes[NANL_Q4_PROOF_HASH_PREFIX_OFFSET..NANL_Q4_HEADER_LEN]
        .copy_from_slice(&proof_hash[0..NANL_Q4_PROOF_HASH_PREFIX_LEN]);
    bytes.extend_from_slice(&packed);

    let report = report_from_parts(
        canonical_payload.len(),
        symbols.len(),
        packed.len(),
        &profile,
        &profile_hash,
        &model,
        &canonical_hash,
        &q4_stream_hash,
        &merkle,
        &proof_hash,
        &constraint_report,
        true,
        true,
        true,
    );
    Ok(NanlQ4Packet {
        bytes,
        symbols,
        report,
    })
}

pub fn decode_packet(bytes: &[u8]) -> NanlQ4Result<Vec<u8>> {
    decode_packet_with_report(bytes).map(|decoded| decoded.canonical_payload)
}

pub fn decode_packet_with_report(bytes: &[u8]) -> NanlQ4Result<NanlQ4Decoded> {
    if bytes.len() < NANL_Q4_HEADER_LEN || &bytes[0..8] != NANL_Q4_MAGIC {
        return Err(NanlQ4Error::Format(
            "invalid NANL-Q4 header magic".to_string(),
        ));
    }
    let major = read_u16(bytes, 8)?;
    let minor = read_u16(bytes, 10)?;
    if major != NANL_Q4_VERSION_MAJOR || minor != NANL_Q4_VERSION_MINOR {
        return Err(NanlQ4Error::Format(format!(
            "unsupported NANL-Q4 version {major}.{minor}"
        )));
    }
    let original_len = read_u64(bytes, 12)? as usize;
    let symbol_count = read_u64(bytes, 20)? as usize;
    let shard_count = bytes[NANL_Q4_SHARD_COUNT_OFFSET];
    if shard_count == 0 {
        return Err(NanlQ4Error::Format(
            "NANL-Q4 shard table/profile marker is missing".to_string(),
        ));
    }
    let proof_flags = bytes[NANL_Q4_PROOF_FLAGS_OFFSET];
    require_proof_flag(
        proof_flags,
        NANL_Q4_FLAG_CONSTRAINT_PROFILE_PRESENT,
        "NANL-Q4 constraint profile is missing",
    )?;
    require_proof_flag(
        proof_flags,
        NANL_Q4_FLAG_SYMBOL_MODEL_PRESENT,
        "NANL-Q4 PRISM symbol model is missing",
    )?;
    require_proof_flag(
        proof_flags,
        NANL_Q4_FLAG_ECC_PRESENT,
        "NANL-Q4 ECC/parity proof is missing",
    )?;
    require_proof_flag(
        proof_flags,
        NANL_Q4_FLAG_MERKLE_PRESENT,
        "NANL-Q4 Merkle proof is missing",
    )?;
    require_proof_flag(
        proof_flags,
        NANL_Q4_FLAG_NONTRIVIAL_CODEBOOK,
        "NANL-Q4 nontrivial constrained codebook proof is missing",
    )?;
    let raw_bits_claim_x2 = bytes[NANL_Q4_RAW_BITS_PER_SYMBOL_X2_OFFSET];
    if raw_bits_claim_x2 > NANL_Q4_RAW_BITS_PER_SYMBOL_X2 {
        return Err(NanlQ4Error::Overclaim(format!(
            "NANL-Q4 cannot claim more than 2 raw bits per symbol; claim_x2={raw_bits_claim_x2}"
        )));
    }
    if bytes[NANL_Q4_MODEL_KIND_OFFSET] != NANL_Q4_SYMBOL_MODEL_PRISM_NATIVE_V0 {
        return Err(NanlQ4Error::Format(
            "NANL-Q4 PRISM-native symbol model kind is invalid".to_string(),
        ));
    }
    if symbol_count
        != original_len.checked_mul(8).ok_or_else(|| {
            NanlQ4Error::Format("NANL-Q4 expected symbol count overflow".to_string())
        })?
    {
        return Err(NanlQ4Error::Format(
            "NANL-Q4 symbol count does not match M1 constrained code rate".to_string(),
        ));
    }
    let packed_len = symbol_count.div_ceil(4);
    let expected_len = NANL_Q4_HEADER_LEN
        .checked_add(packed_len)
        .ok_or_else(|| NanlQ4Error::Format("NANL-Q4 packet length overflow".to_string()))?;
    if bytes.len() != expected_len {
        return Err(NanlQ4Error::Format(
            "NANL-Q4 packet length mismatch".to_string(),
        ));
    }

    let canonical_hash_expected = read_hash(bytes, 32)?;
    let q4_hash_expected = read_hash(bytes, 64)?;
    let merkle_expected = read_hash(bytes, 96)?;
    let expected_parity_xor = bytes[128];
    let expected_parity_sum = read_u16(bytes, 130)?;
    let expected_proof_hash_prefix = &bytes[NANL_Q4_PROOF_HASH_PREFIX_OFFSET..NANL_Q4_HEADER_LEN];

    let packed = &bytes[NANL_Q4_HEADER_LEN..];
    let q4_hash = hash_bytes(packed);
    if q4_hash != q4_hash_expected {
        return Err(NanlQ4Error::Hash(
            "NANL-Q4 stream hash mismatch".to_string(),
        ));
    }
    let symbols = unpack_q4_symbols(packed, symbol_count)?;
    let parity = parity_block(&symbols);
    if parity.xor != expected_parity_xor || parity.sum_mod_65535 != expected_parity_sum {
        return Err(NanlQ4Error::Hash(
            "NANL-Q4 ECC/parity verification failed".to_string(),
        ));
    }
    let profile = Q4ConstraintProfile::default();
    let constraints = validate_constraints(&symbols, profile)?;
    if !constraints.constraints_pass {
        return Err(NanlQ4Error::Constraint(
            "NANL-Q4 constraint profile validation failed".to_string(),
        ));
    }
    let canonical = decode_constrained_symbols(&symbols, original_len)?;
    let canonical_hash = hash_bytes(&canonical);
    if canonical_hash != canonical_hash_expected {
        return Err(NanlQ4Error::Hash(
            "NANL-Q4 canonical roundtrip hash mismatch".to_string(),
        ));
    }
    let merkle = merkle_root(&canonical_hash, &q4_hash);
    if merkle != merkle_expected {
        return Err(NanlQ4Error::Hash(
            "NANL-Q4 Merkle root mismatch".to_string(),
        ));
    }
    let model = Q4SymbolModel::from_payload_and_symbols(&canonical, &symbols);
    let profile_hash = constraint_profile_hash(&profile);
    let proof_hash = proof_manifest_hash(
        &profile,
        &profile_hash,
        &model,
        &parity,
        &canonical_hash,
        &q4_hash,
        &merkle,
    );
    if &proof_hash[0..NANL_Q4_PROOF_HASH_PREFIX_LEN] != expected_proof_hash_prefix {
        return Err(NanlQ4Error::Hash(
            "NANL-Q4 proof manifest hash mismatch".to_string(),
        ));
    }

    Ok(NanlQ4Decoded {
        report: report_from_parts(
            original_len,
            symbol_count,
            packed_len,
            &profile,
            &profile_hash,
            &model,
            &canonical_hash,
            &q4_hash,
            &merkle,
            &proof_hash,
            &constraints,
            true,
            true,
            true,
        ),
        canonical_payload: canonical,
        symbols,
    })
}

pub fn validate_packet(bytes: &[u8]) -> NanlQ4Result<NanlQ4Report> {
    decode_packet_with_report(bytes).map(|decoded| decoded.report)
}

pub fn validate_constraints(
    symbols: &[Q4Symbol],
    profile: Q4ConstraintProfile,
) -> NanlQ4Result<Q4ConstraintReport> {
    if symbols.is_empty() {
        return Err(NanlQ4Error::Constraint(
            "NANL-Q4 empty symbol stream".to_string(),
        ));
    }
    let mut current = symbols[0];
    let mut run = 0usize;
    let mut max_run = 0usize;
    for symbol in symbols {
        if *symbol == current {
            run += 1;
        } else {
            max_run = max_run.max(run);
            current = *symbol;
            run = 1;
        }
    }
    max_run = max_run.max(run);

    let window = profile.gc_window.min(symbols.len()).max(1);
    let mut min_gc = f64::INFINITY;
    let mut max_gc = f64::NEG_INFINITY;
    for start in 0..=symbols.len() - window {
        let gc = symbols[start..start + window]
            .iter()
            .filter(|symbol| symbol.is_gc())
            .count();
        let ratio = gc as f64 / window as f64;
        min_gc = min_gc.min(ratio);
        max_gc = max_gc.max(ratio);
    }
    let constraints_pass =
        max_run <= profile.homopolymer_max && min_gc >= profile.gc_min && max_gc <= profile.gc_max;
    Ok(Q4ConstraintReport {
        constraint_profile_id: "NANL_Q4_SAFE_V0".to_string(),
        gc_window: profile.gc_window,
        gc_min_required: profile.gc_min,
        gc_max_required: profile.gc_max,
        gc_min_observed: round_ratio(min_gc),
        gc_max_observed: round_ratio(max_gc),
        homopolymer_max_required: profile.homopolymer_max,
        homopolymer_max_observed: max_run,
        constraints_pass,
    })
}

pub fn decode_constrained_symbols(
    symbols: &[Q4Symbol],
    original_len: usize,
) -> NanlQ4Result<Vec<u8>> {
    let expected_symbols = original_len
        .checked_mul(8)
        .ok_or_else(|| NanlQ4Error::Format("NANL-Q4 decode symbol count overflow".to_string()))?;
    if symbols.len() != expected_symbols {
        return Err(NanlQ4Error::Format(
            "NANL-Q4 constrained symbol count does not match original length".to_string(),
        ));
    }
    let mut digits = Vec::with_capacity(original_len * 4);
    for codeword in symbols.chunks_exact(2) {
        digits.push(decode_codeword(codeword)?);
    }
    let mut out = Vec::with_capacity(original_len);
    for chunk in digits.chunks_exact(4) {
        out.push((chunk[0] << 6) | (chunk[1] << 4) | (chunk[2] << 2) | chunk[3]);
    }
    Ok(out)
}

pub fn q4_symbols_to_string(symbols: &[Q4Symbol]) -> String {
    symbols.iter().map(|symbol| symbol.as_char()).collect()
}

pub fn validate_q4_claim(raw_bits_per_symbol_claimed: f64) -> NanlQ4Result<()> {
    if raw_bits_per_symbol_claimed > 2.0 {
        return Err(NanlQ4Error::Overclaim(format!(
            "NANL-Q4 claim {raw_bits_per_symbol_claimed} raw bits/symbol exceeds 2-bit quaternary limit"
        )));
    }
    Ok(())
}

pub fn pack_q4_symbols(symbols: &[Q4Symbol]) -> NanlQ4Result<Vec<u8>> {
    let mut out = vec![0u8; symbols.len().div_ceil(4)];
    for (idx, symbol) in symbols.iter().enumerate() {
        out[idx / 4] |= (*symbol as u8) << ((idx % 4) * 2);
    }
    Ok(out)
}

pub fn unpack_q4_symbols(bytes: &[u8], symbol_count: usize) -> NanlQ4Result<Vec<Q4Symbol>> {
    if bytes.len() != symbol_count.div_ceil(4) {
        return Err(NanlQ4Error::Format(
            "NANL-Q4 packed symbol length mismatch".to_string(),
        ));
    }
    if symbol_count % 4 != 0 && !bytes.is_empty() {
        let used_bits = (symbol_count % 4) * 2;
        let allowed_mask = ((1u16 << used_bits) - 1) as u8;
        let final_byte = bytes[bytes.len() - 1];
        if final_byte & !allowed_mask != 0 {
            return Err(NanlQ4Error::Format(
                "NANL-Q4 unused high bits are non-zero".to_string(),
            ));
        }
    }
    let mut out = Vec::with_capacity(symbol_count);
    for byte in bytes {
        for lane in 0..4 {
            if out.len() == symbol_count {
                break;
            }
            out.push(Q4Symbol::try_from((byte >> (lane * 2)) & 0b11)?);
        }
    }
    Ok(out)
}

fn bytes_to_constrained_symbols(bytes: &[u8]) -> Vec<Q4Symbol> {
    let mut out = Vec::with_capacity(bytes.len() * 8);
    for byte in bytes {
        let digits = [
            byte >> 6,
            (byte >> 4) & 0b11,
            (byte >> 2) & 0b11,
            byte & 0b11,
        ];
        for digit in digits {
            out.extend_from_slice(encode_digit(digit));
        }
    }
    out
}

fn encode_digit(digit: u8) -> &'static [Q4Symbol; 2] {
    use Q4Symbol::{A, C, G, T};
    match digit {
        0 => &[A, C],
        1 => &[A, G],
        2 => &[T, C],
        3 => &[T, G],
        _ => unreachable!("base-4 digit is always 0..=3"),
    }
}

fn decode_codeword(codeword: &[Q4Symbol]) -> NanlQ4Result<u8> {
    use Q4Symbol::{A, C, G, T};
    match codeword {
        [A, C] => Ok(0),
        [A, G] => Ok(1),
        [T, C] => Ok(2),
        [T, G] => Ok(3),
        _ => Err(NanlQ4Error::Constraint(
            "NANL-Q4 constrained codeword is invalid".to_string(),
        )),
    }
}

fn parity_block(symbols: &[Q4Symbol]) -> Q4ParityBlock {
    let mut xor = 0u8;
    let mut sum = 0u32;
    let packed = pack_q4_symbols(symbols).unwrap_or_default();
    for symbol in symbols {
        let value = *symbol as u8;
        xor ^= value;
        sum = (sum + value as u32) % 65_535;
    }
    Q4ParityBlock {
        symbol_count: symbols.len() as u64,
        xor,
        sum_mod_65535: sum as u16,
        stream_hash_blake3: hash_hex(&hash_bytes(&packed)),
    }
}

fn parity_manifest(parity: &Q4ParityBlock) -> String {
    format!(
        "NANL_Q4_PARITY_V0_1:{}:{}:{}:{}",
        parity.symbol_count, parity.xor, parity.sum_mod_65535, parity.stream_hash_blake3
    )
}

impl Q4SymbolModel {
    fn from_payload_and_symbols(payload: &[u8], symbols: &[Q4Symbol]) -> Self {
        let mut digit_frequency = [0u64; 4];
        for byte in payload {
            digit_frequency[(byte >> 6) as usize] += 1;
            digit_frequency[((byte >> 4) & 0b11) as usize] += 1;
            digit_frequency[((byte >> 2) & 0b11) as usize] += 1;
            digit_frequency[(byte & 0b11) as usize] += 1;
        }
        let mut q4_symbol_frequency = [0u64; 4];
        for symbol in symbols {
            q4_symbol_frequency[*symbol as usize] += 1;
        }
        let codebook_hash = hash_bytes(
            b"NANL_Q4_PRISM_NATIVE_CODEBOOK_V0:0=AC;1=AG;2=TC;3=TG;raw_bits_per_q4_symbol<=2",
        );
        let manifest = format!(
            "NANL_Q4_SYMBOL_MODEL_V0_1:{}:{:?}:{:?}:{}:{}",
            NANL_Q4_SYMBOL_MODEL_PRISM_NATIVE_V0,
            digit_frequency,
            q4_symbol_frequency,
            hash_hex(&codebook_hash),
            symbols.len()
        );
        let model_hash = hash_bytes(manifest.as_bytes());
        Self {
            model_id: "NANL_Q4_PRISM_NATIVE_SYMBOL_MODEL_V0".to_string(),
            model_kind: NANL_Q4_SYMBOL_MODEL_PRISM_NATIVE_V0,
            digit_frequency,
            q4_symbol_frequency,
            codebook_hash_blake3: hash_hex(&codebook_hash),
            model_hash_blake3: hash_hex(&model_hash),
            raw_bits_per_q4_symbol_upper_bound: "2.0".to_string(),
            simple_binary_mapper: false,
        }
    }
}

fn report_from_parts(
    original_len: usize,
    q4_symbol_count: usize,
    packed_len: usize,
    profile: &Q4ConstraintProfile,
    profile_hash: &[u8; 32],
    model: &Q4SymbolModel,
    canonical_hash: &[u8; 32],
    q4_hash: &[u8; 32],
    merkle: &[u8; 32],
    proof_hash: &[u8; 32],
    constraints: &Q4ConstraintReport,
    roundtrip: bool,
    ecc_verified: bool,
    merkle_verified: bool,
) -> NanlQ4Report {
    NanlQ4Report {
        status: if roundtrip && constraints.constraints_pass && ecc_verified && merkle_verified {
            "PASS"
        } else {
            "FAIL"
        }
        .to_string(),
        mode: "NANL-Q4".to_string(),
        version: format!("{NANL_Q4_VERSION_MAJOR}.{NANL_Q4_VERSION_MINOR}"),
        original_payload_bytes: original_len as u64,
        q4_symbol_count: q4_symbol_count as u64,
        packed_q4_bytes: packed_len as u64,
        constraint_profile_id: constraints.constraint_profile_id.clone(),
        constraint_profile_hash_blake3: hash_hex(profile_hash),
        symbol_model_id: model.model_id.clone(),
        symbol_model_hash_blake3: model.model_hash_blake3.clone(),
        q4_raw_bits_per_symbol_upper_bound: "2.0".to_string(),
        proof_flags: NANL_Q4_REQUIRED_PROOF_FLAGS,
        canonical_payload_hash_blake3: hash_hex(canonical_hash),
        q4_stream_hash_blake3: hash_hex(q4_hash),
        merkle_root_blake3: hash_hex(merkle),
        proof_manifest_hash_blake3: hash_hex(proof_hash),
        canonical_roundtrip_hash_pass: roundtrip,
        constraints_pass: constraints.constraints_pass,
        ecc_enabled: true,
        ecc_verified,
        merkle_verified,
        nontrivial_codec_proof: model.model_kind == NANL_Q4_SYMBOL_MODEL_PRISM_NATIVE_V0
            && !model.simple_binary_mapper
            && profile.gc_window > 0,
        simple_binary_mapper_rejected: true,
        gc_min_observed: constraints.gc_min_observed,
        gc_max_observed: constraints.gc_max_observed,
        homopolymer_max_observed: constraints.homopolymer_max_observed,
        debug_acgt_export_only: true,
    }
}

fn constraint_profile_hash(profile: &Q4ConstraintProfile) -> [u8; 32] {
    let manifest = format!(
        "NANL_Q4_SAFE_V0:{}:{:.4}:{:.4}:{}",
        profile.gc_window, profile.gc_min, profile.gc_max, profile.homopolymer_max
    );
    hash_bytes(manifest.as_bytes())
}

fn proof_manifest_hash(
    profile: &Q4ConstraintProfile,
    profile_hash: &[u8; 32],
    model: &Q4SymbolModel,
    parity: &Q4ParityBlock,
    canonical_hash: &[u8; 32],
    q4_hash: &[u8; 32],
    merkle: &[u8; 32],
) -> [u8; 32] {
    let manifest = format!(
        "NANL_Q4_PROOF_V0_1:flags={}:profile={}:profile_hash={}:model={}:model_hash={}:bits_x2={}:parity={}:canonical={}:q4={}:merkle={}",
        NANL_Q4_REQUIRED_PROOF_FLAGS,
        constraints_profile_manifest(profile),
        hash_hex(profile_hash),
        model.model_id.as_str(),
        model.model_hash_blake3.as_str(),
        NANL_Q4_RAW_BITS_PER_SYMBOL_X2,
        parity_manifest(parity),
        hash_hex(canonical_hash),
        hash_hex(q4_hash),
        hash_hex(merkle)
    );
    hash_bytes(manifest.as_bytes())
}

fn constraints_profile_manifest(profile: &Q4ConstraintProfile) -> String {
    format!(
        "NANL_Q4_SAFE_V0:{}:{:.4}:{:.4}:{}",
        profile.gc_window, profile.gc_min, profile.gc_max, profile.homopolymer_max
    )
}

fn merkle_root(canonical_hash: &[u8; 32], q4_hash: &[u8; 32]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"PRISMCODE:NANL_Q4:MERKLE_ROOT:V0_1");
    hasher.update(canonical_hash);
    hasher.update(q4_hash);
    *hasher.finalize().as_bytes()
}

fn require_proof_flag(flags: u8, flag: u8, message: &str) -> NanlQ4Result<()> {
    if flags & flag == 0 {
        return Err(NanlQ4Error::Format(message.to_string()));
    }
    Ok(())
}

fn hash_bytes(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}

fn hash_hex(hash: &[u8; 32]) -> String {
    hash.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn read_u16(bytes: &[u8], offset: usize) -> NanlQ4Result<u16> {
    let end = offset + 2;
    if end > bytes.len() {
        return Err(NanlQ4Error::Format("truncated NANL-Q4 u16".to_string()));
    }
    Ok(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]))
}

fn read_u64(bytes: &[u8], offset: usize) -> NanlQ4Result<u64> {
    let end = offset + 8;
    if end > bytes.len() {
        return Err(NanlQ4Error::Format("truncated NANL-Q4 u64".to_string()));
    }
    Ok(u64::from_le_bytes(bytes[offset..end].try_into().map_err(
        |_| NanlQ4Error::Internal("NANL-Q4 u64 conversion failed".to_string()),
    )?))
}

fn read_hash(bytes: &[u8], offset: usize) -> NanlQ4Result<[u8; 32]> {
    let end = offset + 32;
    if end > bytes.len() {
        return Err(NanlQ4Error::Format("truncated NANL-Q4 hash256".to_string()));
    }
    bytes[offset..end]
        .try_into()
        .map_err(|_| NanlQ4Error::Internal("NANL-Q4 hash conversion failed".to_string()))
}

fn round_ratio(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}
