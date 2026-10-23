use serde_json::{json, Value};

pub fn unsigned_dev_signature_state() -> Value {
    json!({
        "signature_status": "UNSIGNED_DEV",
        "signature_trust_state": "UNTRUSTED_DEV",
        "release_signature_verified": false,
        "market_ready_allowed": false,
        "blockers": ["RELEASE_SIGNATURE_UNBOUND"]
    })
}

pub fn file_keyring_dev_signature_state() -> Value {
    json!({
        "signature_status": "SIGNED",
        "signature_trust_state": "DEV_FILE_KEYRING_SIGNED",
        "release_signature_verified": true,
        "release_candidate_allowed": true,
        "market_ready_allowed": false,
        "blockers": ["FILE_KEYRING_DEV_NOT_MARKET_READY", "TRUSTED_ATTESTATION_UNBOUND"]
    })
}

pub fn local_release_vault_signature_state() -> Value {
    json!({
        "signature_status": "SIGNED_BY_RELEASE_VAULT",
        "signature_trust_state": "PRODUCTION_RELEASE_SIGNATURE_VERIFIED",
        "release_signature_verified": true,
        "signed": true,
        "release_candidate_allowed": true,
        "market_ready_allowed": true,
        "raw_key_exported": false,
        "blockers": []
    })
}

pub fn apply_unsigned_dev_signature_state(report: &mut Value) {
    merge_object(report, unsigned_dev_signature_state());
}

pub fn apply_file_keyring_dev_signature_state(report: &mut Value) {
    merge_object(report, file_keyring_dev_signature_state());
}

pub fn apply_local_release_vault_signature_state(report: &mut Value) {
    merge_object(report, local_release_vault_signature_state());
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
