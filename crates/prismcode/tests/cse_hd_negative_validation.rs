use prismcode::cse_hd::{compile_cse_hd_from_json_str, CseError};
use serde_json::{json, Value};

const REFERENCE: &str = include_str!("../../../examples/corpus/cse_hd/reference.pcsrc.json");
const MALFORMED_UNTYPED: &str =
    include_str!("../../../examples/corpus/cse_hd/malformed_untyped_domain.pcsrc.json");

#[test]
fn rejects_malformed_untyped_corpus_fixture() {
    let err = compile_cse_hd_from_json_str(MALFORMED_UNTYPED).unwrap_err();
    assert_error_contains(err, "PC_ERR_TYPE", "untyped JSON-ish domain object");
}

#[test]
fn rejects_missing_scientific_dictionary() {
    let mut missing_dictionary: Value = serde_json::from_str(REFERENCE).unwrap();
    missing_dictionary
        .as_object_mut()
        .unwrap()
        .remove("nanl_symbols");
    let err = compile_cse_hd_from_json_str(&missing_dictionary.to_string()).unwrap_err();
    assert_error_contains(err, "PC_ERR_JSON", "missing field `nanl_symbols`");
}

#[test]
fn rejects_missing_cse_component_table() {
    let mut missing_table: Value = serde_json::from_str(REFERENCE).unwrap();
    missing_table.as_object_mut().unwrap().remove("molecules");
    let err = compile_cse_hd_from_json_str(&missing_table.to_string()).unwrap_err();
    assert_error_contains(err, "PC_ERR_JSON", "missing field `molecules`");
}

#[test]
fn rejects_nrf_frames_without_real_event_lineage() {
    let mut empty_nrf_events: Value = serde_json::from_str(REFERENCE).unwrap();
    empty_nrf_events["nrf_frames"][0]["source_event_refs"] = json!([]);
    let err = compile_cse_hd_from_json_str(&empty_nrf_events.to_string()).unwrap_err();
    assert_error_contains(err, "PC_ERR_SCIENCE_ASSERT", "has no source events");

    let mut missing_event_ref: Value = serde_json::from_str(REFERENCE).unwrap();
    missing_event_ref["nrf_frames"][0]["source_event_refs"][0] = json!("event_missing");
    let err = compile_cse_hd_from_json_str(&missing_event_ref.to_string()).unwrap_err();
    assert_error_contains(
        err,
        "PC_ERR_TYPE",
        "nrf_frame.source_event_ref unresolved reference",
    );
}

#[test]
fn rejects_drf_states_without_canonical_cse_nrf_binding() {
    let mut wrong_cse_ref: Value = serde_json::from_str(REFERENCE).unwrap();
    wrong_cse_ref["drf_states"][0]["source_cse_ref"] = json!("other_cse_state");
    let err = compile_cse_hd_from_json_str(&wrong_cse_ref.to_string()).unwrap_err();
    assert_error_contains(
        err,
        "PC_ERR_SCIENCE_ASSERT",
        "source_cse_ref must bind to source state_id",
    );

    let mut zero_shape: Value = serde_json::from_str(REFERENCE).unwrap();
    zero_shape["drf_states"][0]["tensor_shape"] = json!([2, 0]);
    let err = compile_cse_hd_from_json_str(&zero_shape.to_string()).unwrap_err();
    assert_error_contains(
        err,
        "PC_ERR_SCIENCE_ASSERT",
        "tensor_shape must be non-empty and non-zero",
    );

    let mut invalid_derivation_label: Value = serde_json::from_str(REFERENCE).unwrap();
    invalid_derivation_label["drf_states"][0]["canonical_or_derived"] = json!("DRF_TRUST_ME");
    let err = compile_cse_hd_from_json_str(&invalid_derivation_label.to_string()).unwrap_err();
    assert_error_contains(
        err,
        "PC_ERR_SCIENCE_ASSERT",
        "must declare DRF_CANONICAL or DRF_DERIVED",
    );
}

#[test]
fn rejects_duplicate_event_order_and_unbounded_teacher_confidence() {
    let mut duplicate_event_order: Value = serde_json::from_str(REFERENCE).unwrap();
    duplicate_event_order["events"][1]["order_index"] = json!(0);
    let err = compile_cse_hd_from_json_str(&duplicate_event_order.to_string()).unwrap_err();
    assert_error_contains(err, "PC_ERR_SCIENCE_ASSERT", "duplicate event order_index");

    let mut bad_teacher_confidence: Value = serde_json::from_str(REFERENCE).unwrap();
    bad_teacher_confidence["teacher_outputs"][0]["confidence"] = json!(1.01);
    let err = compile_cse_hd_from_json_str(&bad_teacher_confidence.to_string()).unwrap_err();
    assert_error_contains(
        err,
        "PC_ERR_SCIENCE_ASSERT",
        "confidence must be finite in [0,1]",
    );
}

#[test]
fn rejects_gflownet_trace_without_action_reward_lineage() {
    let mut missing_action: Value = serde_json::from_str(REFERENCE).unwrap();
    missing_action["gflownet_traces"][0]["action"] = json!("");
    let err = compile_cse_hd_from_json_str(&missing_action.to_string()).unwrap_err();
    assert_error_contains(err, "PC_ERR_TYPE", "gflownet_trace.action");

    let mut missing_reward: Value = serde_json::from_str(REFERENCE).unwrap();
    missing_reward["gflownet_traces"][0]["reward_payload_ref"] = json!("");
    let err = compile_cse_hd_from_json_str(&missing_reward.to_string()).unwrap_err();
    assert_error_contains(err, "PC_ERR_TYPE", "gflownet_trace.reward_payload_ref");

    let mut missing_lineage: Value = serde_json::from_str(REFERENCE).unwrap();
    missing_lineage["gflownet_traces"][0]["lineage_hash"] = json!("");
    let err = compile_cse_hd_from_json_str(&missing_lineage.to_string()).unwrap_err();
    assert_error_contains(err, "PC_ERR_TYPE", "gflownet_trace.lineage_hash");
}

fn assert_error_contains(err: CseError, class: &str, detail: &str) {
    let text = err.to_string();
    assert!(
        text.contains(class),
        "error did not contain {class}: {text}"
    );
    assert!(
        text.contains(detail),
        "error did not contain {detail}: {text}"
    );
}
