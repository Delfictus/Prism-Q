use prismcode::warning_policy::{
    source_has_blanket_warning_suppression, validate_market_warning_policy_text,
    validate_warning_records, WarningRecord, WarningScope, WarningWaiver,
};

#[test]
fn market_warning_policy_text_has_required_release_fields() {
    let policy_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../configs/release/warning_policy.toml");
    let policy = std::fs::read_to_string(policy_path).expect("warning policy exists");
    validate_market_warning_policy_text(&policy).expect("policy validates");
}

#[test]
fn blanket_warning_suppression_is_detected() {
    assert!(source_has_blanket_warning_suppression(
        "#![allow(warnings)]"
    ));
    assert!(source_has_blanket_warning_suppression(
        "#![allow(dead_code)]"
    ));
    assert!(!source_has_blanket_warning_suppression(
        "#[allow(clippy::too_many_arguments)]"
    ));
}

#[test]
fn zero_warning_record_set_passes_policy() {
    validate_warning_records(&[], &[]).expect("zero warnings pass");
}

#[test]
fn non_shipped_warning_requires_complete_waiver() {
    let warning = WarningRecord {
        warning_id: "W001".to_string(),
        crate_name: "xtask".to_string(),
        file: "xtask/src/main.rs".to_string(),
        line: Some(42),
        warning_kind: "dead_code".to_string(),
        scope: WarningScope::NonShipped,
        security_relevant: false,
        runtime_relevant: false,
        cuda_relevant: false,
        model_relevant: false,
    };
    let waiver = WarningWaiver {
        waiver_id: "WAIVE-XTASK-001".to_string(),
        file: "xtask/src/main.rs".to_string(),
        line: Some(42),
        owner: "release".to_string(),
        rationale: "non-shipped diagnostic helper retained for operator audit".to_string(),
        expiry_condition: "remove before production tooling split".to_string(),
        market_impact: "not shipped in runtime binaries".to_string(),
        scope: WarningScope::NonShipped,
    };
    validate_warning_records(&[warning], &[waiver]).expect("complete non-shipped waiver passes");
}
