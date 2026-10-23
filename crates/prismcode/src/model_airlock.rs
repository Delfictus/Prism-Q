use crate::binary_runtime::{PcError, PcResult};
use serde_json::Value;

pub const MODEL_GENERATED_UNVERIFIED: &str = "MODEL_GENERATED_UNVERIFIED";

pub fn reject_forbidden_model_action(action: &Value) -> PcResult<()> {
    let classification = action
        .get("model_output_classification")
        .or_else(|| action.get("output_classification"))
        .and_then(Value::as_str)
        .unwrap_or("");
    if classification != MODEL_GENERATED_UNVERIFIED {
        return Err(PcError::Security(
            "model output lacks MODEL_GENERATED_UNVERIFIED classification".to_string(),
        ));
    }
    for field in [
        "canonical_mutation_allowed",
        "direct_pcode_write",
        "direct_pcout_write",
        "vault_access_requested",
        "release_key_access_requested",
        "external_egress_requested",
        "proprietary_export_requested",
        "bypass_hil_resolve",
    ] {
        if action.get(field).and_then(Value::as_bool) == Some(true) {
            return Err(PcError::Security(format!(
                "model airlock rejected forbidden action: {field}"
            )));
        }
    }
    Ok(())
}
