//! Top-level dispatcher: `PRISMExecutionRequest` -> per-variant projection
//! -> `PRISMExecutionResponse`.

use anyhow::Result;
use serde_json::{Map, Value};
use std::time::{SystemTime, UNIX_EPOCH};

use super::handshake::{
    PRISMExecutionRequest, PRISMExecutionResponse, VariantExecutionResponse, RESPONSE_SCHEMA_TAG,
};
use super::projection::{
    project_variant, project_variant_with_topology, ProjectionConfig, TopologyProjectionContext,
    VariantPoint, WTTensorPack,
};
use super::sidechain_tables::AminoAcid;

/// Configuration knobs for the dispatcher (CLI surface).
#[derive(Debug, Clone)]
pub struct VariantDispatchConfig {
    pub prism_run_id: String,
    pub projection: ProjectionConfig,
}

impl VariantDispatchConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.prism_run_id.is_empty() {
            return Err("prism_run_id must be non-empty".to_string());
        }
        self.projection.validate()
    }
}

/// Reasons the dispatcher will refuse to emit a response.
#[derive(Debug)]
pub enum DispatchError {
    SchemaMismatch(String),
    UnknownVariant {
        target: String,
        variant: String,
        reason: String,
    },
    NonFinite {
        target: String,
        variant: String,
        channel: &'static str,
    },
    ChainHashMismatch {
        declared: String,
        observed: String,
    },
    Internal(anyhow::Error),
}

impl std::fmt::Display for DispatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DispatchError::SchemaMismatch(m) => write!(f, "schema mismatch: {m}"),
            DispatchError::UnknownVariant {
                target,
                variant,
                reason,
            } => write!(f, "unknown variant ({target}, {variant}): {reason}"),
            DispatchError::NonFinite {
                target,
                variant,
                channel,
            } => write!(f, "non-finite {channel} for variant ({target}, {variant})"),
            DispatchError::ChainHashMismatch { declared, observed } => write!(
                f,
                "blake3 chain mismatch: declared {declared}, observed {observed}"
            ),
            DispatchError::Internal(e) => write!(f, "internal: {e}"),
        }
    }
}

impl std::error::Error for DispatchError {}

/// Canonical JSON serialisation used for the hash chain.  Recursively
/// sorts object keys alphabetically; arrays / scalars are preserved.
/// Equivalent to DSTW's `canonical_json_bytes` helper.
pub fn canonical_json(value: &Value) -> Vec<u8> {
    fn canon(v: &Value) -> Value {
        match v {
            Value::Object(map) => {
                let mut sorted: Vec<(String, Value)> =
                    map.iter().map(|(k, vv)| (k.clone(), canon(vv))).collect();
                sorted.sort_by(|a, b| a.0.cmp(&b.0));
                let mut out = Map::new();
                for (k, vv) in sorted {
                    out.insert(k, vv);
                }
                Value::Object(out)
            }
            Value::Array(arr) => Value::Array(arr.iter().map(canon).collect()),
            other => other.clone(),
        }
    }
    serde_json::to_vec(&canon(value)).expect("serde_json::to_vec on Value never fails")
}

fn iso_8601_utc_now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default();
    let (y, m, d, hh, mm, ss) = epoch_to_civil(secs);
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

fn epoch_to_civil(secs: u64) -> (i64, u32, u32, u32, u32, u32) {
    let secs_in_day: u64 = 86400;
    let z = (secs / secs_in_day) as i64 + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as i64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 {
        (mp + 3) as u32
    } else {
        (mp - 9) as u32
    };
    let year = y + (if m <= 2 { 1 } else { 0 });
    let time = secs % secs_in_day;
    (
        year,
        m,
        d,
        (time / 3600) as u32,
        ((time % 3600) / 60) as u32,
        (time % 60) as u32,
    )
}

/// Compute the blake3 hex of the canonical JSON of a request.  Returns
/// a 64-char lowercase hex string.
pub fn request_blake3_hex(request: &PRISMExecutionRequest) -> String {
    let value = serde_json::to_value(request).expect("PRISMExecutionRequest serialises");
    let bytes = canonical_json(&value);
    blake3::hash(&bytes).to_hex().to_string()
}

/// Run the projection for every variant in the request and assemble the
/// response.  Hard-fails on any non-finite output (no silent dropping).
pub fn dispatch_variant_batch(
    request: &PRISMExecutionRequest,
    wt: &WTTensorPack,
    config: &VariantDispatchConfig,
) -> Result<PRISMExecutionResponse, DispatchError> {
    dispatch_variant_batch_with_topology(request, wt, config, None)
}

/// Run the projection with an optional paired active/inactive topology
/// context. Live Option A dispatch passes `Some(topology)` so the stochastic
/// rotamer ensemble can score steric congestion in the WT coordinate frame.
pub fn dispatch_variant_batch_with_topology(
    request: &PRISMExecutionRequest,
    wt: &WTTensorPack,
    config: &VariantDispatchConfig,
    topology: Option<&TopologyProjectionContext>,
) -> Result<PRISMExecutionResponse, DispatchError> {
    request.validate().map_err(DispatchError::SchemaMismatch)?;
    config.validate().map_err(DispatchError::SchemaMismatch)?;
    wt.validate().map_err(DispatchError::SchemaMismatch)?;

    let mut variants_resp: Vec<VariantExecutionResponse> =
        Vec::with_capacity(request.variants.len());
    for v in &request.variants {
        let wt_aa = parse_one(&v.wildtype_aa).map_err(|e| DispatchError::UnknownVariant {
            target: v.target.clone(),
            variant: v.variant.clone(),
            reason: e,
        })?;
        let mu_aa = parse_one(&v.mutant_aa).map_err(|e| DispatchError::UnknownVariant {
            target: v.target.clone(),
            variant: v.variant.clone(),
            reason: e,
        })?;
        let vp = VariantPoint {
            residue_number: v.residue_number as i32,
            wildtype: wt_aa,
            mutant: mu_aa,
        };
        let r = match topology {
            Some(ctx) => project_variant_with_topology(&vp, wt, &config.projection, Some(ctx)),
            None => project_variant(&vp, wt, &config.projection),
        };

        // Finiteness check on every emitted scalar.  This is where the
        // DSTW Pydantic schema would have rejected us; we catch it here
        // first so the error is informative.
        for (name, val) in [
            ("delta_P_active", r.deltas.delta_p_active),
            ("delta_P_lock", r.deltas.delta_p_lock),
            ("delta_P_ensemble", r.deltas.delta_p_ensemble),
            ("sigma_delta_P_active", r.sigmas.sigma_delta_p_active),
            ("sigma_delta_P_lock", r.sigmas.sigma_delta_p_lock),
            ("sigma_delta_P_ensemble", r.sigmas.sigma_delta_p_ensemble),
        ] {
            if !val.is_finite() {
                return Err(DispatchError::NonFinite {
                    target: v.target.clone(),
                    variant: v.variant.clone(),
                    channel: name,
                });
            }
        }
        variants_resp.push(VariantExecutionResponse {
            target: v.target.clone(),
            variant: v.variant.clone(),
            delta_P_active: r.deltas.delta_p_active,
            delta_P_lock: r.deltas.delta_p_lock,
            delta_P_ensemble: r.deltas.delta_p_ensemble,
            sigma_delta_P_active: r.sigmas.sigma_delta_p_active,
            sigma_delta_P_lock: r.sigmas.sigma_delta_p_lock,
            sigma_delta_P_ensemble: r.sigmas.sigma_delta_p_ensemble,
            prism_run_id: config.prism_run_id.clone(),
            converged: r.converged,
        });
    }

    Ok(PRISMExecutionResponse {
        campaign_id: request.campaign_id.clone(),
        round_index: request.round_index,
        request_blake3: request_blake3_hex(request),
        completed_at_utc: iso_8601_utc_now(),
        variants: variants_resp,
        response_schema: RESPONSE_SCHEMA_TAG.to_string(),
    })
}

fn parse_one(s: &str) -> Result<AminoAcid, String> {
    let c = s
        .chars()
        .next()
        .ok_or_else(|| "empty wildtype/mutant aa".to_string())?;
    AminoAcid::from_one_letter(c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dstw_dispatch::handshake::{PRISMExecutionAcquisition, VariantExecutionRequest};

    fn synth_wt() -> WTTensorPack {
        let n = 100;
        WTTensorPack {
            residue_numbers: (1..=n as i32).collect(),
            residue_index_lo: 1,
            residue_index_hi: n as i32,
            inactive_te_out: vec![0.0; n],
            inactive_te_in: vec![0.5; n],
            inactive_delta_hc: vec![0.3; n],
            inactive_sigma_hydration_sq: vec![0.1; n],
            active_te_out: vec![0.6; n],
            active_te_in: vec![0.5; n],
            active_delta_hc: vec![0.3; n],
            active_sigma_hydration_sq: vec![0.1; n],
        }
    }

    fn ok_request() -> PRISMExecutionRequest {
        PRISMExecutionRequest {
            campaign_id: "cxcr4_dms_calibration_01".to_string(),
            round_index: 0,
            previous_round_blake3: None,
            issued_at_utc: "2026-05-20T08:00:00Z".to_string(),
            variants: vec![
                VariantExecutionRequest {
                    target: "CXCR4".to_string(),
                    uniprot_accession: "P61073".to_string(),
                    variant: "L17A".to_string(),
                    residue_number: 17,
                    wildtype_aa: "L".to_string(),
                    mutant_aa: "A".to_string(),
                    requested_channels: vec![
                        "delta_P_active".to_string(),
                        "delta_P_lock".to_string(),
                        "delta_P_ensemble".to_string(),
                    ],
                    acquisition_reason: PRISMExecutionAcquisition::StratifiedSeed,
                },
                VariantExecutionRequest {
                    target: "CXCR4".to_string(),
                    uniprot_accession: "P61073".to_string(),
                    variant: "W42A".to_string(),
                    residue_number: 42,
                    wildtype_aa: "W".to_string(),
                    mutant_aa: "A".to_string(),
                    requested_channels: vec![
                        "delta_P_active".to_string(),
                        "delta_P_lock".to_string(),
                        "delta_P_ensemble".to_string(),
                    ],
                    acquisition_reason: PRISMExecutionAcquisition::StratifiedSeed,
                },
            ],
            expected_response_schema: RESPONSE_SCHEMA_TAG.to_string(),
        }
    }

    fn ok_config() -> VariantDispatchConfig {
        VariantDispatchConfig {
            prism_run_id: "test_run_001".to_string(),
            projection: ProjectionConfig::default(),
        }
    }

    #[test]
    fn round_trip_dispatch_yields_two_finite_variant_rows() {
        let req = ok_request();
        let wt = synth_wt();
        let cfg = ok_config();
        let resp = dispatch_variant_batch(&req, &wt, &cfg).expect("dispatch should succeed");
        assert_eq!(resp.variants.len(), 2);
        assert_eq!(resp.response_schema, RESPONSE_SCHEMA_TAG);
        assert_eq!(resp.request_blake3.len(), 64);
        assert!(resp.variants[0].delta_P_active.is_finite());
        assert!(resp.variants[1].sigma_delta_P_active.is_finite());
    }

    #[test]
    fn dispatch_marks_w_to_a_as_nonconverged_and_inflates_sigma() {
        let req = ok_request();
        let wt = synth_wt();
        let cfg = ok_config();
        let resp = dispatch_variant_batch(&req, &wt, &cfg).unwrap();
        let l17a = &resp.variants[0];
        let w42a = &resp.variants[1];
        assert!(l17a.converged);
        assert!(
            !w42a.converged,
            "W→A volume drop should fail rigid-backbone"
        );
        // Sigma must strictly inflate on non-convergence.
        assert!(w42a.sigma_delta_P_active > l17a.sigma_delta_P_active);
    }

    #[test]
    fn request_blake3_is_deterministic() {
        let req = ok_request();
        let h1 = request_blake3_hex(&req);
        let h2 = request_blake3_hex(&req);
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64);
    }

    #[test]
    fn request_blake3_changes_when_payload_changes() {
        let req1 = ok_request();
        let mut req2 = ok_request();
        req2.round_index = 1;
        assert_ne!(request_blake3_hex(&req1), request_blake3_hex(&req2));
    }

    #[test]
    fn invalid_request_short_circuits() {
        let mut req = ok_request();
        req.campaign_id = "campaign_with_wasserstein_distance".to_string();
        let wt = synth_wt();
        let cfg = ok_config();
        let err = dispatch_variant_batch(&req, &wt, &cfg).unwrap_err();
        match err {
            DispatchError::SchemaMismatch(m) => assert!(m.contains("forbidden scalar")),
            other => panic!("expected SchemaMismatch, got {:?}", other),
        }
    }
}
