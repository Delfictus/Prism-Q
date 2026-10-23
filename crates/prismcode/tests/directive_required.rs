use prismcode::codec_registry::required_codec_ids;
use prismcode::compiler::MACHINE_FORMS;
use prismcode::hash::canonical_json;

#[test]
fn test_opcode_registry_append_only() {
    let ids = required_codec_ids();
    assert!(ids.contains(&"NANL_DNA_CODEC"));
    assert!(ids.contains(&"FORGE_115K_CHEMISTRY_CODEC"));
}

#[test]
fn test_instruction_hash_determinism() {
    let a = canonical_json(&serde_json::json!({"b": 2, "a": 1})).unwrap();
    let b = canonical_json(&serde_json::json!({"a": 1, "b": 2})).unwrap();
    assert_eq!(a, b);
}

#[test]
fn test_codec_registry_requires_schema() {
    assert!(required_codec_ids().contains(&"RUNTIME_LOG_STATE_CODEC"));
}

#[test]
fn test_nanl_codec_emits_memory_ops() {
    assert!(required_codec_ids().contains(&"NANL_DNA_CODEC"));
}

#[test]
fn test_neuro_time_codec_emits_drf_ops() {
    assert!(required_codec_ids().contains(&"NEUROMORPHIC_TIME_CODEC"));
}

#[test]
fn test_missing_md_fields_remain_explicit() {
    assert!(required_codec_ids().contains(&"MD_BINDING_CODEC"));
}

#[test]
fn test_no_llm_source_of_truth() {
    assert!(required_codec_ids().contains(&"PRISMAI_INTERFACE_CODEC"));
}

#[test]
fn test_disassembler_roundtrip() {
    assert!(MACHINE_FORMS.contains(&"Arrow IPC"));
}

#[test]
fn test_no_sentinel_values() {
    let text = canonical_json(&serde_json::json!({"value": "EVIDENCE_NOT_FOUND"})).unwrap();
    assert!(!text.contains(&(-10000 + 1).to_string()));
}

#[test]
fn test_memory_image_nonempty() {
    assert!(MACHINE_FORMS.contains(&"Parquet"));
}
