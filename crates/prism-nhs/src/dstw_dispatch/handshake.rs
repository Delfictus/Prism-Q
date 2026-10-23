//! Frozen wire schemas mirroring `prism_dstw.orchestration.prism_handshake`
//! on the DSTW side.  Every field type, default, and constraint is the
//! Rust analogue of the Pydantic model that DSTW will use to
//! `model_validate` whatever this dispatcher emits.

use serde::{Deserialize, Serialize};

/// MUST equal `(delta_P_active, delta_P_lock, delta_P_ensemble)` on every
/// variant request.  The vectorial-only contract is asserted by the
/// `validate` method on `PRISMExecutionRequest`.
pub const REQUIRED_DELTA_CHANNELS: [&str; 3] =
    ["delta_P_active", "delta_P_lock", "delta_P_ensemble"];

/// Scalar inputs DSTW has banned from the variant bridge.  Defence in
/// depth — these are checked at the wire even though DSTW's Pydantic
/// validators also enforce them, because the dispatcher MUST never
/// emit a response on a request that smuggles a scalar input.
pub const FORBIDDEN_SCALAR_FIELDS: [&str; 4] = [
    "P_variant_divergence",
    "wasserstein_distance",
    "scalar_wasserstein",
    "variant_distance",
];

/// Default response schema tag.  DSTW expects this exact string.
pub const RESPONSE_SCHEMA_TAG: &str = "dstw_prism_execution_response_v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PRISMExecutionAcquisition {
    StratifiedSeed,
    MaxPredictiveVariance,
    OperatorOverride,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct VariantExecutionRequest {
    pub target: String,
    pub uniprot_accession: String,
    pub variant: String,
    pub residue_number: u32,
    pub wildtype_aa: String,
    pub mutant_aa: String,
    #[serde(default = "default_requested_channels")]
    pub requested_channels: Vec<String>,
    pub acquisition_reason: PRISMExecutionAcquisition,
}

fn default_requested_channels() -> Vec<String> {
    REQUIRED_DELTA_CHANNELS
        .iter()
        .map(|s| s.to_string())
        .collect()
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PRISMExecutionRequest {
    pub campaign_id: String,
    pub round_index: u32,
    pub previous_round_blake3: Option<String>,
    pub issued_at_utc: String,
    pub variants: Vec<VariantExecutionRequest>,
    #[serde(default = "default_expected_response_schema")]
    pub expected_response_schema: String,
}

fn default_expected_response_schema() -> String {
    RESPONSE_SCHEMA_TAG.to_string()
}

impl PRISMExecutionRequest {
    /// Run the DSTW-side validator suite on a deserialised request.
    /// Returns the offending free-text string on failure, never panics.
    pub fn validate(&self) -> Result<(), String> {
        if self.campaign_id.is_empty() {
            return Err("campaign_id must be non-empty".to_string());
        }
        if self.variants.is_empty() {
            return Err("variants must be non-empty".to_string());
        }
        if self.expected_response_schema != RESPONSE_SCHEMA_TAG {
            return Err(format!(
                "expected_response_schema must equal {:?}; got {:?}",
                RESPONSE_SCHEMA_TAG, self.expected_response_schema
            ));
        }
        // No duplicate (target, variant)
        let mut seen: std::collections::HashSet<(String, String)> = Default::default();
        for v in &self.variants {
            let key = (v.target.clone(), v.variant.clone());
            if !seen.insert(key) {
                return Err(format!(
                    "duplicate (target, variant) entry: ({}, {})",
                    v.target, v.variant
                ));
            }
            // Vectorial-only requested_channels
            let observed: std::collections::HashSet<&str> =
                v.requested_channels.iter().map(|s| s.as_str()).collect();
            let required: std::collections::HashSet<&str> =
                REQUIRED_DELTA_CHANNELS.iter().copied().collect();
            if observed != required {
                return Err(format!(
                    "variant {:?}: requested_channels must equal {:?}; got {:?}",
                    v.variant, REQUIRED_DELTA_CHANNELS, v.requested_channels
                ));
            }
        }
        // Free-text scalar-token scan
        for v in &self.variants {
            for token in [&v.variant, &v.target, &self.campaign_id] {
                let lower = token.to_ascii_lowercase();
                for bad in FORBIDDEN_SCALAR_FIELDS {
                    if lower.contains(&bad.to_ascii_lowercase()) {
                        return Err(format!(
                            "request contains forbidden scalar reference {:?} in {:?}",
                            bad, token
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct VariantExecutionResponse {
    pub target: String,
    pub variant: String,
    pub delta_P_active: f64,
    pub delta_P_lock: f64,
    pub delta_P_ensemble: f64,
    pub sigma_delta_P_active: f64,
    pub sigma_delta_P_lock: f64,
    pub sigma_delta_P_ensemble: f64,
    pub prism_run_id: String,
    pub converged: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct PRISMExecutionResponse {
    pub campaign_id: String,
    pub round_index: u32,
    pub request_blake3: String,
    pub completed_at_utc: String,
    pub variants: Vec<VariantExecutionResponse>,
    #[serde(default = "default_expected_response_schema")]
    pub response_schema: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok_variant() -> VariantExecutionRequest {
        VariantExecutionRequest {
            target: "CXCR4".to_string(),
            uniprot_accession: "P61073".to_string(),
            variant: "L17A".to_string(),
            residue_number: 17,
            wildtype_aa: "L".to_string(),
            mutant_aa: "A".to_string(),
            requested_channels: default_requested_channels(),
            acquisition_reason: PRISMExecutionAcquisition::StratifiedSeed,
        }
    }

    fn ok_request(variants: Vec<VariantExecutionRequest>) -> PRISMExecutionRequest {
        PRISMExecutionRequest {
            campaign_id: "cxcr4_dms_calibration_01".to_string(),
            round_index: 0,
            previous_round_blake3: None,
            issued_at_utc: "2026-05-20T08:00:00Z".to_string(),
            variants,
            expected_response_schema: RESPONSE_SCHEMA_TAG.to_string(),
        }
    }

    #[test]
    fn valid_request_passes() {
        let req = ok_request(vec![ok_variant()]);
        assert!(req.validate().is_ok());
    }

    #[test]
    fn empty_variants_rejected() {
        let req = ok_request(vec![]);
        assert!(req.validate().is_err());
    }

    #[test]
    fn duplicate_variant_rejected() {
        let v = ok_variant();
        let req = ok_request(vec![v.clone(), v]);
        assert!(req.validate().unwrap_err().contains("duplicate"));
    }

    #[test]
    fn scalar_wasserstein_in_campaign_id_rejected() {
        let mut req = ok_request(vec![ok_variant()]);
        req.campaign_id = "experiment_with_wasserstein_distance".to_string();
        assert!(req.validate().unwrap_err().contains("forbidden scalar"));
    }

    #[test]
    fn wrong_requested_channels_rejected() {
        let mut v = ok_variant();
        v.requested_channels = vec!["P_variant_divergence".to_string()];
        let req = ok_request(vec![v]);
        assert!(req.validate().unwrap_err().contains("requested_channels"));
    }

    #[test]
    fn response_schema_tag_is_pinned() {
        let mut req = ok_request(vec![ok_variant()]);
        req.expected_response_schema = "rogue".to_string();
        assert!(req
            .validate()
            .unwrap_err()
            .contains("expected_response_schema"));
    }
}
