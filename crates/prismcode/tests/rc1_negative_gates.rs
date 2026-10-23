use prismcode::binary_runtime::{
    compile_reference_json, execute_packet, init_file_keyring, parse_packet, security_inspect,
    validate_packet, verify_replay, write_prod_encrypted_packet, Backend, SectionKind,
    ValidationLevel, HEADER_LEN, PACKET_HASH_LEN, PACKET_HASH_OFFSET, SECTION_ENTRY_LEN,
};
use prismcode::compression_registry::{
    hash_bytes as compression_hash_bytes, CompressionCodecKind, HashPair, RuntimeCodecRegistry,
};
use prismcode::cuda_backend::{cuda_execution_preflight, cuda_runtime_report};
use prismcode::cuda_manifest::{generated_unsigned_manifest, manifest_is_signed};
use prismcode::model_governance::{inspect_model_registry, smoke_model_runtime};
use prismcode::release_manifest;
use serde_json::{json, Value};
use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};

const FIXED_RUNTIME_SMOKE_PROMPT: &str = "PRISMCODE_MODEL_RUNTIME_SMOKE_V0_1: identify yourself as MODEL_GENERATED_UNVERIFIED advisory output, do not request canonical PRISMCODE mutation, and return a bounded health response.";

fn reference_json() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/reference/reference.json")
}

#[test]
fn malformed_header_and_section_fail_before_execution() {
    let tmp = tempfile::tempdir().unwrap();
    let pcode = tmp.path().join("reference.pcode");
    compile_reference_json(&reference_json(), &pcode).unwrap();

    let mut bad_magic = fs::read(&pcode).unwrap();
    bad_magic[0] = b'X';
    let err = parse_packet(bad_magic).unwrap_err().to_string();
    assert!(err.contains("PC_ERR_FORMAT"));
    assert!(err.contains("magic is not PRISMPC"));

    let mut bad_section = fs::read(&pcode).unwrap();
    let section_dir = read_u64_le(&bad_section, 28);
    put_u16_le(&mut bad_section, section_dir + 2, 0xffff);
    let err = parse_packet(bad_section).unwrap_err().to_string();
    assert!(err.contains("PC_ERR_SECTION"));
    assert!(err.contains("unknown section kind 65535"));
}

#[test]
fn unknown_opcode_and_invalid_operand_kind_fail_with_valid_hashes() {
    let tmp = tempfile::tempdir().unwrap();
    let pcode = tmp.path().join("reference.pcode");
    compile_reference_json(&reference_json(), &pcode).unwrap();

    let mut unknown_opcode = fs::read(&pcode).unwrap();
    mutate_first_instruction(&mut unknown_opcode, |instr| {
        put_u16_le(instr, 0, 0xffff);
        instr[2] = 0xff;
    });
    let unknown_path = tmp.path().join("unknown_opcode.pcode");
    fs::write(&unknown_path, unknown_opcode).unwrap();
    let err = validate_packet(&unknown_path, ValidationLevel::L3)
        .unwrap_err()
        .to_string();
    assert!(err.contains("PC_ERR_FORMAT"));
    assert!(err.contains("unknown opcode 0xffff"));

    let mut bad_operand = fs::read(&pcode).unwrap();
    mutate_first_instruction(&mut bad_operand, |instr| {
        instr[4] = 0xfe;
    });
    let bad_operand_path = tmp.path().join("bad_operand.pcode");
    fs::write(&bad_operand_path, bad_operand).unwrap();
    let err = validate_packet(&bad_operand_path, ValidationLevel::L3)
        .unwrap_err()
        .to_string();
    assert!(err.contains("PC_ERR_FORMAT"));
    assert!(err.contains("invalid operand kind 254"));
}

#[test]
fn bad_compressed_hash_and_malformed_chunk_fail_closed() {
    let registry = RuntimeCodecRegistry::with_builtin_codecs();

    let zstd_canonical = b"rc1 zstd executable compression section ".repeat(64);
    let zstd = registry
        .encode_section(CompressionCodecKind::SectionZstd, &zstd_canonical)
        .unwrap();
    let mut wrong_hashes = zstd.hashes;
    wrong_hashes.hash_compressed_bytes[0] ^= 0x01;
    let err = registry
        .validate_encoded_by_id(zstd.codec_id, &zstd.compressed_bytes, wrong_hashes)
        .unwrap_err()
        .to_string();
    assert!(err.contains("PC_ERR_HASH"));
    assert!(err.contains("compressed hash mismatch"));

    let tensor_canonical = (0..257).map(|idx| (idx % 251) as u8).collect::<Vec<_>>();
    let tensor = registry
        .encode_section(CompressionCodecKind::TensorChunked, &tensor_canonical)
        .unwrap();
    let mut corrupted = tensor.compressed_bytes.clone();
    let first_payload_byte = 16 + 4 + 4 + 32;
    corrupted[first_payload_byte] ^= 0x55;
    let matching_outer_hash = HashPair {
        hash_compressed_bytes: compression_hash_bytes(&corrupted),
        hash_canonical_uncompressed_bytes: tensor.hashes.hash_canonical_uncompressed_bytes,
    };
    let err = registry
        .validate_encoded_by_id(tensor.codec_id, &corrupted, matching_outer_hash)
        .unwrap_err()
        .to_string();
    assert!(err.contains("PC_ERR_HASH"));
    assert!(err.contains("TENSOR_CHUNKED chunk hash mismatch"));
}

#[test]
fn corrupted_seal_and_signature_never_become_security_ready() {
    let tmp = tempfile::tempdir().unwrap();
    let pcode = tmp.path().join("reference.pcode");
    let pcout = tmp.path().join("reference.cpu.pcout");
    compile_reference_json(&reference_json(), &pcode).unwrap();
    execute_packet(&pcode, Backend::Cpu, &pcout).unwrap();

    let mut corrupted_pcout = fs::read(&pcout).unwrap();
    corrupted_pcout[128] ^= 0x80;
    let corrupted_pcout_path = tmp.path().join("corrupted_seal.pcout");
    fs::write(&corrupted_pcout_path, corrupted_pcout).unwrap();

    let replay_err = verify_replay(&corrupted_pcout_path, None)
        .unwrap_err()
        .to_string();
    assert!(replay_err.contains("PC_ERR_REPLAY"));
    assert!(replay_err.contains("seal chain hash mismatch"));
    let inspection = security_inspect(&corrupted_pcout_path).unwrap();
    assert_eq!(inspection["seal_status"], "SEAL_CHAIN_FAILED");
    assert_eq!(inspection["production_security_ready"], false);
    assert_eq!(inspection["market_ready"], false);

    let keyring = tmp.path().join("keyring/prismcode_file_keyring.json");
    let public_key = tmp.path().join("keyring/release_public_key.ed25519.hex");
    let prod = tmp.path().join("reference.prod.pcode");
    init_file_keyring(&keyring, &public_key).unwrap();
    write_prod_encrypted_packet(&pcode, &prod, &keyring, &public_key).unwrap();

    let mut tampered_prod = fs::read(&prod).unwrap();
    let last = tampered_prod.len() - 1;
    tampered_prod[last] ^= 0x01;
    let tampered_prod_path = tmp.path().join("tampered_signature.prod.pcode");
    fs::write(&tampered_prod_path, tampered_prod).unwrap();

    let err = validate_packet(&tampered_prod_path, ValidationLevel::L5)
        .unwrap_err()
        .to_string();
    assert!(err.contains("PC_ERR_SECURITY"));
    assert!(err.contains("Ed25519 signature failed"));
    let inspection = security_inspect(&tampered_prod_path).unwrap();
    assert_eq!(
        inspection["seal_status"],
        "PRODUCTION_ENVELOPE_SIGNATURE_FAILED"
    );
    assert_eq!(inspection["signed"], false);
    assert_eq!(inspection["production_security_ready"], false);
    assert_eq!(inspection["market_ready"], false);
}

#[test]
fn dev_hash_bound_cuda_manifest_executes_without_claiming_production_signature() {
    let tmp = tempfile::tempdir().unwrap();
    let unsigned_manifest = generated_unsigned_manifest().unwrap();
    assert_eq!(unsigned_manifest["cuda_kernel_signed"], false);
    assert_eq!(
        unsigned_manifest["signature_status"],
        "DEV_HASH_BOUND_UNSIGNED"
    );
    assert!(!manifest_is_signed(&unsigned_manifest));

    let manifest_path = tmp.path().join("unsigned_cuda_manifest.json");
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&unsigned_manifest).unwrap(),
    )
    .unwrap();
    let _guard = EnvGuard::set("PRISMCODE_CUDA_TRUST_MANIFEST", &manifest_path);
    let report = cuda_runtime_report();
    let status = report["cuda_execution_status"].as_str().unwrap();

    assert_eq!(status, "LINKED_DEV_HASH_BOUND_SELFTEST_READY");
    assert_eq!(report["cuda_kernel_signed"], false);
    let preflight = cuda_execution_preflight().unwrap();
    assert_eq!(preflight["cuda_kernel_signed"], false);
    assert_eq!(preflight["cuda_kernel_trust"], "DEV_HASH_BOUND_UNSIGNED");
    assert_ne!(
        preflight["cuda_execution_status"],
        "LINKED_SIGNED_SELFTEST_READY"
    );
}

#[test]
fn fake_model_runtime_active_claim_without_output_hash_fails_closed() {
    let tmp = tempfile::tempdir().unwrap();
    let registry = write_model_registry_missing_smoke_output_hash(&tmp);

    let inspect = inspect_model_registry(&registry).unwrap();
    assert_eq!(inspect.status, "FAIL");
    let gpt = inspect.gpt_oss_20b.unwrap();
    assert_eq!(gpt.installed_hash_verified, true);
    assert_eq!(gpt.runtime_smoke_status.as_deref(), Some("PASSED"));
    assert_eq!(gpt.model_runtime_active, false);
    assert!(gpt
        .required_runtime_evidence_missing
        .contains(&"runtime_smoke_output_hash".to_string()));
    assert!(inspect
        .issues
        .iter()
        .any(|issue| issue.contains("claims active serving")));

    let smoke = smoke_model_runtime(&registry).unwrap();
    assert_eq!(smoke.status, "FAIL_CLOSED");
    assert_eq!(smoke.model_runtime_active, false);
    assert_eq!(smoke.compatible_runtime_detected, true);
    assert_eq!(smoke.output_hash_blake3, None);
    assert_eq!(
        smoke.release_mode_gates.ai_assisted_os_model_gate,
        "BLOCKED_MODEL_RUNTIME_SMOKE_REQUIRED"
    );
    assert!(smoke
        .required_runtime_evidence_missing
        .contains(&"runtime_smoke_output_hash".to_string()));
}

#[test]
fn market_ready_status_json_spoof_is_overridden_by_policy() {
    let mut report = json!({
        "status": "PASS",
        "market_ready": true,
        "market_ready_eligible": true,
        "production_security_ready": true,
        "key_provider": "FILE_KEYRING_DEV",
        "vault_status": "FILE_KEYRING_DEV_BOUND",
        "attestation_status": "ATTESTATION_UNAVAILABLE",
        "cuda_execution_status": "FAIL_CLOSED_UNLINKED"
    });

    release_manifest::apply_release_candidate_policy(&mut report);

    assert_eq!(report["market_ready"], false);
    assert_eq!(report["market_ready_eligible"], false);
    assert_eq!(
        report["market_ready_policy_version"],
        "PRISMCODE_MARKET_READY_V0_1"
    );
    for blocker in [
        "FILE_KEYRING_DEV_NOT_MARKET_READY",
        "OPENBAO_OR_VAULT_UNBOUND",
        "TRUSTED_ATTESTATION_UNBOUND",
        "CUDA_RUNTIME_NOT_LINKED",
    ] {
        assert!(
            array_contains_str(&report, "market_blockers", blocker),
            "missing blocker {blocker}: {report}"
        );
    }
}

fn mutate_first_instruction(bytes: &mut [u8], mutate: impl FnOnce(&mut [u8])) {
    let packet = parse_packet(bytes.to_vec()).unwrap();
    let instr_entry = packet
        .sections
        .iter()
        .find(|entry| entry.section_type == SectionKind::InstrStream)
        .unwrap()
        .clone();
    let instr_start = instr_entry.offset as usize;
    mutate(&mut bytes[instr_start..instr_start + 32]);
    refresh_section_hash(
        bytes,
        instr_entry.section_id,
        instr_entry.offset,
        instr_entry.length,
    );
    refresh_packet_hash(bytes);
}

fn refresh_section_hash(bytes: &mut [u8], section_id: u16, offset: u64, length: u64) {
    let start = offset as usize;
    let end = start + length as usize;
    let section_hash = test_hash_bytes(&bytes[start..end]);
    let section_dir = read_u64_le(bytes, 28);
    let entry_start = section_dir + (section_id as usize - 1) * SECTION_ENTRY_LEN;
    bytes[entry_start + 64..entry_start + 96].copy_from_slice(&section_hash);
}

fn refresh_packet_hash(bytes: &mut [u8]) {
    assert!(bytes.len() >= HEADER_LEN);
    bytes[PACKET_HASH_OFFSET..PACKET_HASH_OFFSET + PACKET_HASH_LEN].fill(0);
    let packet_hash = test_hash_bytes(bytes);
    bytes[PACKET_HASH_OFFSET..PACKET_HASH_OFFSET + PACKET_HASH_LEN].copy_from_slice(&packet_hash);
}

fn test_hash_bytes(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}

fn read_u64_le(bytes: &[u8], offset: usize) -> usize {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()) as usize
}

fn put_u16_le(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn array_contains_str(value: &Value, field: &str, expected: &str) -> bool {
    value
        .get(field)
        .and_then(Value::as_array)
        .map(|items| items.iter().any(|item| item.as_str() == Some(expected)))
        .unwrap_or(false)
}

fn write_model_registry_missing_smoke_output_hash(tmp: &tempfile::TempDir) -> PathBuf {
    let model_dir = tmp.path().join("openai_gpt-oss-20b");
    fs::create_dir_all(&model_dir).unwrap();
    let files = [
        ("config.json", b"{\"model\":\"test\"}".as_slice()),
        ("tokenizer.json", b"{\"tokenizer\":\"test\"}".as_slice()),
        (
            "model-00000-of-00001.safetensors",
            b"test-weights".as_slice(),
        ),
    ];
    for (name, bytes) in files {
        fs::write(model_dir.join(name), bytes).unwrap();
    }

    let prompt_hash = blake3::hash(FIXED_RUNTIME_SMOKE_PROMPT.as_bytes())
        .to_hex()
        .to_string();
    let registry = json!({
        "registry_version": "TEST_MODEL_RUNTIME_SPOOF",
        "rules": {
            "canonical_state_mutation_allowed": false
        },
        "installed_models": [{
            "model_id": "openai/gpt-oss-20b",
            "install_allowed": true,
            "hardware_gated": false,
            "local_path": model_dir,
            "weights_hash_verified": true,
            "serving_status": "RUNTIME_BOUND_SMOKE_PASSED",
            "runtime_binding_status": "BOUND",
            "runtime_binding_allowed": true,
            "runtime_bound": true,
            "runtime_name": "governed-test-runtime",
            "runtime_version": "0.1.0",
            "runtime_hash": blake3::hash(b"governed-test-runtime").to_hex().to_string(),
            "launch_command": "governed-test-runtime --offline-smoke",
            "endpoint": "unix:///tmp/prismcode-model-runtime.sock",
            "network_policy": "AIRLOCKED_OFFLINE",
            "gpu_cpu_memory_profile": {"mode": "cpu-only-test"},
            "context_limit": 4096,
            "quantization_format": "none",
            "deterministic_settings": {"temperature": 0, "top_p": 1},
            "runtime_smoke_required_before_binding": true,
            "runtime_smoke_status": "PASSED",
            "runtime_smoke_ref": "smoke://missing-output-hash",
            "runtime_smoke_prompt_hash": prompt_hash,
            "runtime_smoke_output_classification": "MODEL_GENERATED_UNVERIFIED",
            "runtime_smoke_exit_status": 0,
            "canonical_mutation_denial_status": "PASSED",
            "airlock_status": "MODEL_GENERATED_UNVERIFIED_BY_DEFAULT",
            "canonical_state_mutation_allowed": false,
            "files": files.iter().map(|(path, bytes)| json!({
                "path": path,
                "bytes": bytes.len(),
                "blake3": blake3::hash(bytes).to_hex().to_string()
            })).collect::<Vec<_>>()
        }],
        "hardware_gated_models": [{
            "model_id": "Qwen/Qwen3-Coder-Next",
            "hardware_gated": true,
            "install_allowed": false,
            "install_gate": "QUANTIZED_OR_OFFLOADED_SERVING_PLAN_REQUIRED",
            "status": "HARDWARE_GATED_INSTALL_DISALLOWED",
            "canonical_state_mutation_allowed": false
        }]
    });
    let path = tmp.path().join("model_registry.json");
    fs::write(&path, serde_json::to_vec_pretty(&registry).unwrap()).unwrap();
    path
}

struct EnvGuard {
    key: &'static str,
    previous: Option<OsString>,
}

impl EnvGuard {
    fn set(key: &'static str, value: &Path) -> Self {
        let previous = env::var_os(key);
        env::set_var(key, value);
        Self { key, previous }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        if let Some(previous) = &self.previous {
            env::set_var(self.key, previous);
        } else {
            env::remove_var(self.key);
        }
    }
}
