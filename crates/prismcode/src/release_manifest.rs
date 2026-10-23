use serde_json::{json, Map, Value};

pub fn apply_release_candidate_policy(report: &mut Value) {
    ensure_object(report);
    let provider = report
        .get("key_provider")
        .and_then(Value::as_str)
        .unwrap_or("PRISM_VAULT_UNBOUND")
        .to_string();
    let attestation = report
        .get("attestation_status")
        .and_then(Value::as_str)
        .unwrap_or("ATTESTATION_UNAVAILABLE")
        .to_string();

    let Some(obj) = report.as_object_mut() else {
        return;
    };

    obj.insert("market_ready".to_string(), json!(false));
    obj.insert("market_ready_eligible".to_string(), json!(false));
    obj.insert(
        "market_ready_policy_version".to_string(),
        json!("PRISMCODE_MARKET_READY_V0_1"),
    );

    if provider == "FILE_KEYRING_DEV" {
        obj.insert("market_ready_eligible".to_string(), json!(false));
        obj.insert(
            "provider_market_state".to_string(),
            json!("DEV_RELEASE_CANDIDATE_ONLY"),
        );
        append_blocker(obj, "FILE_KEYRING_DEV_NOT_MARKET_READY");
        append_blocker(obj, "OPENBAO_OR_VAULT_UNBOUND");
    }

    let cuda_status = obj
        .get("cuda_execution_status")
        .or_else(|| obj.get("cuda_status"))
        .and_then(Value::as_str)
        .unwrap_or("CUDA_RUNTIME_NOT_LINKED");
    if cuda_status != "LINKED_SIGNED_SELFTEST_READY" {
        append_blocker(obj, "CUDA_RUNTIME_NOT_LINKED");
    }

    if attestation != "TRUSTED_ATTESTED" {
        append_blocker(obj, "TRUSTED_ATTESTATION_UNBOUND");
    }
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
