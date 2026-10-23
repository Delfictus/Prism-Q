use prismcode::model_airlock::reject_forbidden_model_action;
use serde_json::json;

#[test]
fn advisory_unverified_model_action_without_forbidden_effects_is_airlocked() {
    let action = json!({
        "model_output_classification": "MODEL_GENERATED_UNVERIFIED",
        "canonical_mutation_allowed": false,
        "direct_pcode_write": false,
        "vault_access_requested": false,
        "external_egress_requested": false
    });
    reject_forbidden_model_action(&action).unwrap();
}

#[test]
fn direct_pcode_write_is_rejected() {
    let action = json!({
        "model_output_classification": "MODEL_GENERATED_UNVERIFIED",
        "direct_pcode_write": true
    });
    let err = reject_forbidden_model_action(&action)
        .unwrap_err()
        .to_string();
    assert!(err.contains("direct_pcode_write"));
}
