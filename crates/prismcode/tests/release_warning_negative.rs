use prismcode::warning_policy::{
    source_has_blanket_warning_suppression, validate_warning_records, WarningRecord, WarningScope,
    WarningWaiver,
};

fn warning(scope: WarningScope, file: &str) -> WarningRecord {
    WarningRecord {
        warning_id: "WNEG".to_string(),
        crate_name: if scope == WarningScope::Shipped {
            "prismcode".to_string()
        } else {
            "xtask".to_string()
        },
        file: file.to_string(),
        line: Some(7),
        warning_kind: "dead_code".to_string(),
        scope,
        security_relevant: false,
        runtime_relevant: false,
        cuda_relevant: false,
        model_relevant: false,
    }
}

fn waiver_for(file: &str) -> WarningWaiver {
    WarningWaiver {
        waiver_id: "WAIVE-NEG".to_string(),
        file: file.to_string(),
        line: Some(7),
        owner: "release".to_string(),
        rationale: "non-shipped build helper only".to_string(),
        expiry_condition: "remove before market release if shipped".to_string(),
        market_impact: "not shipped".to_string(),
        scope: WarningScope::NonShipped,
    }
}

#[test]
fn blanket_allow_warnings_in_shipped_code_fails() {
    assert!(source_has_blanket_warning_suppression(
        "#![allow(warnings)]\nfn main() {}"
    ));
}

#[test]
fn blanket_allow_dead_code_in_shipped_code_fails() {
    assert!(source_has_blanket_warning_suppression(
        "#![allow(dead_code)]\nfn main() {}"
    ));
}

#[test]
fn unwaived_shipped_warning_fails() {
    let warnings = [warning(
        WarningScope::Shipped,
        "crates/prismcode/src/security.rs",
    )];
    assert!(validate_warning_records(&warnings, &[]).is_err());
}

#[test]
fn waiver_missing_file_line_fails() {
    let warnings = [warning(WarningScope::NonShipped, "xtask/src/main.rs")];
    let mut waiver = waiver_for("xtask/src/main.rs");
    waiver.line = None;
    assert!(validate_warning_records(&warnings, &[waiver]).is_err());
}

#[test]
fn waiver_missing_rationale_fails() {
    let warnings = [warning(WarningScope::NonShipped, "xtask/src/main.rs")];
    let mut waiver = waiver_for("xtask/src/main.rs");
    waiver.rationale.clear();
    assert!(validate_warning_records(&warnings, &[waiver]).is_err());
}

#[test]
fn waiver_missing_expiry_fails() {
    let warnings = [warning(WarningScope::NonShipped, "xtask/src/main.rs")];
    let mut waiver = waiver_for("xtask/src/main.rs");
    waiver.expiry_condition.clear();
    assert!(validate_warning_records(&warnings, &[waiver]).is_err());
}

#[test]
fn waiver_marked_non_shipped_but_file_is_shipped_fails() {
    let warnings = [warning(
        WarningScope::NonShipped,
        "crates/prismcode/src/bin/prismrun.rs",
    )];
    let waiver = waiver_for("crates/prismcode/src/bin/prismrun.rs");
    assert!(validate_warning_records(&warnings, &[waiver]).is_err());
}

#[test]
fn security_warning_without_explicit_classification_fails() {
    let mut warning = warning(WarningScope::NonShipped, "xtask/src/main.rs");
    warning.security_relevant = true;
    let mut waiver = waiver_for("xtask/src/main.rs");
    waiver.market_impact.clear();
    assert!(validate_warning_records(&[warning], &[waiver]).is_err());
}

#[test]
fn cuda_runtime_warning_without_explicit_classification_fails() {
    let mut warning = warning(WarningScope::NonShipped, "xtask/src/main.rs");
    warning.cuda_relevant = true;
    let mut waiver = waiver_for("xtask/src/main.rs");
    waiver.market_impact.clear();
    assert!(validate_warning_records(&[warning], &[waiver]).is_err());
}

#[test]
fn model_runtime_warning_without_explicit_classification_fails() {
    let mut warning = warning(WarningScope::NonShipped, "xtask/src/main.rs");
    warning.model_relevant = true;
    let mut waiver = waiver_for("xtask/src/main.rs");
    waiver.market_impact.clear();
    assert!(validate_warning_records(&[warning], &[waiver]).is_err());
}
