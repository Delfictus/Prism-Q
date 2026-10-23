use serde_json::{json, Map, Value};

pub const ATTESTATION_UNAVAILABLE: &str = "ATTESTATION_UNAVAILABLE";
pub const TRUSTED_ATTESTED: &str = "TRUSTED_ATTESTED";

pub fn mark_runtime_attestation_unavailable(mut report: Value) -> Value {
    mark_unavailable(&mut report, "RUNTIME_ATTESTATION_UNBOUND");
    report
}

pub fn mark_artifact_attestation_unavailable(mut report: Value) -> Value {
    let current = report
        .get("attestation_status")
        .and_then(Value::as_str)
        .unwrap_or("ATTESTATION_EVIDENCE_NOT_PRESENT");
    if current != TRUSTED_ATTESTED {
        mark_unavailable(&mut report, "ARTIFACT_ATTESTATION_UNBOUND");
    }
    report
}

pub fn unavailable_attestation_block() -> Value {
    json!({
        "attestation_status": ATTESTATION_UNAVAILABLE,
        "trust_state": ATTESTATION_UNAVAILABLE,
        "trusted_attestation": false,
        "market_ready_allowed": false,
        "reason": "Attestation mechanism is not bound; absence is reported as unavailable, not trusted.",
        "blockers": ["TRUSTED_ATTESTATION_UNBOUND"]
    })
}

fn mark_unavailable(report: &mut Value, reason: &str) {
    ensure_object(report);
    let Some(obj) = report.as_object_mut() else {
        return;
    };
    obj.insert("status".to_string(), json!(ATTESTATION_UNAVAILABLE));
    obj.insert(
        "attestation_status".to_string(),
        json!(ATTESTATION_UNAVAILABLE),
    );
    obj.insert("trust_state".to_string(), json!(ATTESTATION_UNAVAILABLE));
    obj.insert("trusted".to_string(), json!(false));
    obj.insert("trusted_attestation".to_string(), json!(false));
    obj.insert("market_ready_allowed".to_string(), json!(false));
    obj.insert("attestation_absence_reason".to_string(), json!(reason));
    obj.insert(
        "attestation_note".to_string(),
        json!("Attestation mechanism unavailable; this is not equivalent to trusted."),
    );
    append_blocker(obj, "TRUSTED_ATTESTATION_UNBOUND");
}

fn ensure_object(value: &mut Value) {
    if !value.is_object() {
        *value = json!({});
    }
}

fn append_blocker(obj: &mut Map<String, Value>, blocker: &str) {
    let entry = obj
        .entry("market_blockers".to_string())
        .or_insert_with(|| json!([]));
    let Some(blockers) = entry.as_array_mut() else {
        *entry = json!([blocker]);
        return;
    };
    if !blockers.iter().any(|item| item.as_str() == Some(blocker)) {
        blockers.push(json!(blocker));
    }
}
