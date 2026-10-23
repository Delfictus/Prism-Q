//! Parser for `CORPUS_MERGE_READINESS.json` — the tokenizer / corpus readiness
//! surface.
//!
//! **Fidelity doctrine (load-bearing):** readiness is judged by *matched fields*,
//! never by raw token count. In this durable object the per-row fidelity signal
//! is the field [`CorpusRow::expected_matched_fields`], which is heterogeneous —
//! it is sometimes an integer (`10`) and sometimes an expression string
//! (`"20-47"`, `"17 cells + 8 pairwise"`, `"0-1"`). The acceptance gate phrases
//! the same rule as `matched_fields > 0` and explicitly rejects `tokens > 0` as
//! an "opaque escape". The viewer therefore surfaces the expected-matched-fields
//! value verbatim plus a derived fidelity badge, and never displays token counts
//! as a readiness signal.

use serde::Deserialize;

/// `expected_matched_fields` is `int | string`. We keep the raw form for honest
/// display and derive a numeric expectation for the fidelity badge.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ExpectedMatchedFields {
    Int(i64),
    Text(String),
}

impl ExpectedMatchedFields {
    /// Verbatim display of the expected-matched-fields expression.
    pub fn display(&self) -> String {
        match self {
            ExpectedMatchedFields::Int(value) => value.to_string(),
            ExpectedMatchedFields::Text(text) => text.clone(),
        }
    }

    /// The largest integer mentioned in the expression (its upper expectation).
    pub fn max_value(&self) -> Option<i64> {
        match self {
            ExpectedMatchedFields::Int(value) => Some(*value),
            ExpectedMatchedFields::Text(text) => max_int_in(text),
        }
    }

    /// Whether this row expects any positive matched-field fidelity at all
    /// (i.e. the `matched_fields > 0` doctrine can be met for it).
    pub fn expects_fidelity(&self) -> bool {
        self.max_value().map(|value| value > 0).unwrap_or(false)
    }
}

/// One runtime-artifact family row.
#[derive(Debug, Clone, Deserialize)]
pub struct CorpusRow {
    pub artifact_family: String,
    pub tokenizer_family: Option<String>,
    #[serde(default, deserialize_with = "flex_opt_bool")]
    pub tokenizer_family_ready: Option<bool>,
    pub expected_matched_fields: Option<ExpectedMatchedFields>,
    #[serde(default, deserialize_with = "flex_opt_bool")]
    pub native_artifact_emitted: Option<bool>,
    pub emit_opcode: Option<String>,
    #[serde(default, deserialize_with = "flex_opt_bool")]
    pub dense_tensor_risk: Option<bool>,
    pub classification: Option<String>,
    #[serde(
        rename = "TRAINING_READY_PRISMOS_NATIVE",
        default,
        deserialize_with = "flex_opt_bool"
    )]
    pub training_ready_prismos_native: Option<bool>,
    pub blocker: Option<String>,
}

impl CorpusRow {
    /// Verbatim expected-matched-fields display, or `"—"` when absent.
    pub fn matched_fields_display(&self) -> String {
        self.expected_matched_fields
            .as_ref()
            .map(ExpectedMatchedFields::display)
            .unwrap_or_else(|| "—".to_string())
    }

    /// Derived fidelity badge text for this row.
    pub fn fidelity_badge(&self) -> &'static str {
        match &self.expected_matched_fields {
            Some(value) if value.expects_fidelity() => "FIDELITY_EXPECTED",
            Some(_) => "ZERO_FIELDS",
            None => "UNSPECIFIED",
        }
    }

    pub fn classification(&self) -> &str {
        self.classification.as_deref().unwrap_or("UNCLASSIFIED")
    }
}

/// The whole readiness document. Only the fields the viewer needs are typed;
/// the rest is retained as raw JSON for an "inspect source" affordance.
#[derive(Debug, Clone, Deserialize)]
pub struct CorpusReadiness {
    pub schema_kind: Option<String>,
    pub phase: Option<String>,
    pub source_branch: Option<String>,
    pub source_head: Option<String>,
    #[serde(default)]
    pub rows: Vec<CorpusRow>,
    pub summary_counts: Option<serde_json::Value>,
    pub cross_lane_consolidation: Option<serde_json::Value>,
    pub readiness_flag_definition: Option<serde_json::Value>,
}

impl CorpusReadiness {
    pub fn from_json_str(text: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(text)
    }

    /// Rows the tokenizer can ingest right now (`tokenizer_family_ready`).
    pub fn ready_now_count(&self) -> usize {
        self.rows
            .iter()
            .filter(|row| row.tokenizer_family_ready == Some(true))
            .count()
    }

    /// Rows whose expected matched-fields indicate positive fidelity.
    pub fn fidelity_expected_count(&self) -> usize {
        self.rows
            .iter()
            .filter(|row| {
                row.expected_matched_fields
                    .as_ref()
                    .map(ExpectedMatchedFields::expects_fidelity)
                    .unwrap_or(false)
            })
            .count()
    }

    /// The overall cross-lane corpus verdict, if present.
    pub fn overall_verdict(&self) -> Option<String> {
        self.cross_lane_consolidation
            .as_ref()
            .and_then(|value| value.get("overall_corpus_verdict"))
            .and_then(|value| value.as_str())
            .map(str::to_string)
    }
}

/// Extract the largest non-negative integer appearing in `text`.
fn max_int_in(text: &str) -> Option<i64> {
    let mut max: Option<i64> = None;
    let mut current = String::new();
    let flush = |current: &mut String, max: &mut Option<i64>| {
        if !current.is_empty() {
            if let Ok(value) = current.parse::<i64>() {
                *max = Some(max.map_or(value, |existing: i64| existing.max(value)));
            }
            current.clear();
        }
    };
    for ch in text.chars() {
        if ch.is_ascii_digit() {
            current.push(ch);
        } else {
            flush(&mut current, &mut max);
        }
    }
    flush(&mut current, &mut max);
    max
}

/// Deserialize a boolean field that the runtime sometimes emits as a
/// descriptive string (e.g. `dense_tensor_risk: "partial (… NOT typed; gate
/// respected)"`). Clean booleans and the strings "true"/"false"/"yes"/"no"/
/// "ready" map to `Some(bool)`; any other value (including a "partial …"
/// description or null) maps to `None` so the whole readiness object still
/// parses instead of failing on one heterogeneous row.
fn flex_opt_bool<'de, D>(deserializer: D) -> Result<Option<bool>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    match serde_json::Value::deserialize(deserializer)? {
        serde_json::Value::Bool(value) => Ok(Some(value)),
        serde_json::Value::String(text) => {
            let normalized = text.trim().to_ascii_lowercase();
            Ok(match normalized.as_str() {
                "true" | "yes" | "ready" => Some(true),
                "false" | "no" => Some(false),
                _ => None,
            })
        }
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = r#"{
      "schema_kind": "prismos_native_md_evidence_corpus_merge_readiness_consolidated",
      "phase": "1_read_only_discovery",
      "source_branch": "prismos-buildout",
      "cross_lane_consolidation": { "overall_corpus_verdict": "TOKENIZER_READY_PRE_NATIVE_EMISSION" },
      "rows": [
        {"artifact_family":"spike_events","tokenizer_family":"SpikeEvent","tokenizer_family_ready":true,"expected_matched_fields":10,"native_artifact_emitted":false,"classification":"TOKENIZE_NOW","TRAINING_READY_PRISMOS_NATIVE":false,"blocker":"Lane 2 wiring"},
        {"artifact_family":"kcc_visualization_sites","tokenizer_family":"KccSite","tokenizer_family_ready":true,"expected_matched_fields":"20-47","classification":"TOKENIZE_NOW","TRAINING_READY_PRISMOS_NATIVE":false},
        {"artifact_family":"twin16_transition_view_residuals","tokenizer_family":"TwinTransitionResidual","tokenizer_family_ready":true,"expected_matched_fields":"17 cells + 8 pairwise","classification":"TOKENIZE_NOW","TRAINING_READY_PRISMOS_NATIVE":false},
        {"artifact_family":"qnanl_seal_verify_receipt","tokenizer_family":"QnanlReceipt","tokenizer_family_ready":true,"expected_matched_fields":"0-1","classification":"TOKENIZE_NOW_WITH_WIRING_NOTE","TRAINING_READY_PRISMOS_NATIVE":false},
        {"artifact_family":"future_emit","tokenizer_family":"Future","tokenizer_family_ready":false,"expected_matched_fields":0,"classification":"EMIT_THEN_TOKENIZE","TRAINING_READY_PRISMOS_NATIVE":false}
      ]
    }"#;

    #[test]
    fn parses_heterogeneous_expected_matched_fields() {
        let doc = CorpusReadiness::from_json_str(FIXTURE).unwrap();
        assert_eq!(doc.rows.len(), 5);
        assert_eq!(doc.rows[0].matched_fields_display(), "10");
        assert_eq!(doc.rows[1].matched_fields_display(), "20-47");
        assert_eq!(
            doc.rows[2].matched_fields_display(),
            "17 cells + 8 pairwise"
        );
        assert_eq!(doc.rows[3].matched_fields_display(), "0-1");
    }

    #[test]
    fn fidelity_expectation_keys_off_matched_fields_not_tokens() {
        let doc = CorpusReadiness::from_json_str(FIXTURE).unwrap();
        assert!(doc.rows[0]
            .expected_matched_fields
            .as_ref()
            .unwrap()
            .expects_fidelity()); // 10
        assert!(doc.rows[1]
            .expected_matched_fields
            .as_ref()
            .unwrap()
            .expects_fidelity()); // 20-47 -> 47
        assert!(doc.rows[2]
            .expected_matched_fields
            .as_ref()
            .unwrap()
            .expects_fidelity()); // 17 cells -> 17
        assert!(doc.rows[3]
            .expected_matched_fields
            .as_ref()
            .unwrap()
            .expects_fidelity()); // 0-1 -> 1
        assert!(!doc.rows[4]
            .expected_matched_fields
            .as_ref()
            .unwrap()
            .expects_fidelity()); // 0
                                  // Four of five rows expect positive matched-field fidelity.
        assert_eq!(doc.fidelity_expected_count(), 4);
    }

    #[test]
    fn ready_now_and_verdict() {
        let doc = CorpusReadiness::from_json_str(FIXTURE).unwrap();
        assert_eq!(doc.ready_now_count(), 4);
        assert_eq!(
            doc.overall_verdict().as_deref(),
            Some("TOKENIZER_READY_PRE_NATIVE_EMISSION")
        );
    }

    #[test]
    fn badges_reflect_fidelity() {
        let doc = CorpusReadiness::from_json_str(FIXTURE).unwrap();
        assert_eq!(doc.rows[0].fidelity_badge(), "FIDELITY_EXPECTED");
        assert_eq!(doc.rows[4].fidelity_badge(), "ZERO_FIELDS");
    }

    #[test]
    fn max_int_in_handles_expressions() {
        assert_eq!(max_int_in("20-47"), Some(47));
        assert_eq!(max_int_in("17 cells + 8 pairwise"), Some(17));
        assert_eq!(max_int_in("0-1"), Some(1));
        assert_eq!(max_int_in("none"), None);
    }

    #[test]
    fn tolerates_descriptive_string_in_bool_field() {
        // The real T4 object carries `dense_tensor_risk` as a descriptive string
        // on the noise_floor row; the whole document must still parse.
        let doc = CorpusReadiness::from_json_str(
            r#"{ "rows": [
                {"artifact_family":"noise_floor","tokenizer_family":"NoiseFloor","tokenizer_family_ready":true,"expected_matched_fields":2,"dense_tensor_risk":"partial (mu/sigma/samples arrays -> sidecar, NOT typed; gate respected)","classification":"TOKENIZE_NOW","TRAINING_READY_PRISMOS_NATIVE":false}
            ]}"#,
        )
        .expect("a descriptive string in dense_tensor_risk must not fail the parse");
        assert_eq!(doc.rows.len(), 1);
        assert_eq!(doc.rows[0].tokenizer_family_ready, Some(true));
        assert_eq!(doc.rows[0].dense_tensor_risk, None);
    }
}
