use crate::binary_runtime::{PcError, PcResult};
use serde_json::{json, Value};

const UNPINNED_DEV_BUILD: &str = "UNPINNED_DEV_BUILD";

pub fn pinned_root_hash() -> &'static str {
    option_env!("PRISMCODE_PINNED_ROOT_HASH").unwrap_or(UNPINNED_DEV_BUILD)
}

pub fn pinned_root_public_key_hex() -> &'static str {
    option_env!("PRISMCODE_PINNED_ROOT_PUBLIC_KEY_HEX").unwrap_or("")
}

pub fn is_release_pinned() -> bool {
    let hash = pinned_root_hash();
    hash != UNPINNED_DEV_BUILD && hash.len() == 64 && hash.chars().all(|c| c.is_ascii_hexdigit())
}

pub fn require_release_pinned_root() -> PcResult<()> {
    reject_editable_root_inputs()?;
    if !is_release_pinned() {
        return Err(PcError::Security(
            "PC_ERR_PINNED_ROOT_UNBOUND: release verifier binary was not compiled with a pinned trust root".to_string(),
        ));
    }
    Ok(())
}

pub fn reject_editable_root_inputs() -> PcResult<()> {
    for key in ["PRISM_TRUST_ROOT", "PRISM_ROOT_KEY", "PRISMCODE_TRUST_ROOT"] {
        if std::env::var_os(key).is_some() {
            return Err(PcError::Security(format!(
                "PC_ERR_PINNED_ROOT_EDITABLE_INPUT:{key} must not be supplied in release verification"
            )));
        }
    }
    Ok(())
}

pub fn require_signature_root_hash(public_key_hash: &str) -> PcResult<()> {
    require_release_pinned_root()?;
    if public_key_hash != pinned_root_hash() {
        return Err(PcError::Security(format!(
            "PC_ERR_PINNED_ROOT_MISMATCH: signature root {public_key_hash} does not match pinned root {}",
            pinned_root_hash()
        )));
    }
    Ok(())
}

pub fn report() -> Value {
    let env_root_override_present = ["PRISM_TRUST_ROOT", "PRISM_ROOT_KEY", "PRISMCODE_TRUST_ROOT"]
        .iter()
        .any(|key| std::env::var_os(key).is_some());
    json!({
        "status": if is_release_pinned() { "PASS" } else { "FAIL" },
        "pinned_root_hash": pinned_root_hash(),
        "pinned_root_source": if is_release_pinned() { "COMPILED_IN_BINARY" } else { UNPINNED_DEV_BUILD },
        "editable_config_root_used": false,
        "env_root_override_present": env_root_override_present,
        "embedded_public_key_present": !pinned_root_public_key_hex().is_empty(),
    })
}
