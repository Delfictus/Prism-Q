use serde_json::{json, Value};

pub fn unbound_vault_binding() -> Value {
    json!({
        "vault_status": "PRISM_VAULT_UNBOUND",
        "vault_binding_state": "UNBOUND",
        "external_vault_bound": false,
        "market_ready_allowed": false,
        "raw_key_output_allowed": false,
        "blockers": ["OPENBAO_OR_VAULT_UNBOUND"]
    })
}

pub fn file_keyring_dev_binding() -> Value {
    json!({
        "vault_status": "FILE_KEYRING_DEV_BOUND",
        "vault_binding_state": "DEV_FILE_KEYRING_BOUND",
        "external_vault_bound": false,
        "release_candidate_allowed": true,
        "l5_release_candidate_allowed": true,
        "market_ready_allowed": false,
        "raw_key_output_allowed": false,
        "blockers": ["FILE_KEYRING_DEV_NOT_MARKET_READY", "OPENBAO_OR_VAULT_UNBOUND"]
    })
}

pub fn local_release_vault_binding() -> Value {
    json!({
        "vault_status": "LOCAL_RELEASE_VAULT_BOUND",
        "vault_binding_state": "LOCAL_RELEASE_VAULT_BOUND",
        "external_vault_bound": false,
        "enterprise_external_vault": false,
        "market_profile": "LOCAL_RELEASE",
        "release_candidate_allowed": true,
        "l5_release_candidate_allowed": true,
        "market_ready_allowed": true,
        "raw_key_output_allowed": false,
        "raw_key_exported": false,
        "blockers": []
    })
}

pub fn apply_file_keyring_dev_binding(report: &mut Value) {
    merge_object(report, file_keyring_dev_binding());
}

pub fn apply_local_release_vault_binding(report: &mut Value) {
    merge_object(report, local_release_vault_binding());
}

pub fn apply_unbound_vault_binding(report: &mut Value) {
    merge_object(report, unbound_vault_binding());
}

fn merge_object(target: &mut Value, overlay: Value) {
    let Some(target_obj) = target.as_object_mut() else {
        *target = overlay;
        return;
    };
    let Some(overlay_obj) = overlay.as_object() else {
        return;
    };
    for (key, value) in overlay_obj {
        target_obj.insert(key.clone(), value.clone());
    }
}
