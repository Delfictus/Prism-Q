//! Tolerant parser for `prismcode_status.json` (the operational dashboard gate
//! source). The desktop dashboard primarily renders the real `prismui` binary
//! output, but it also parses this status object directly so it can show native
//! status pills and an overall readiness verdict even when the `prismui` binary
//! is not on the search path.

use serde::Deserialize;

/// Validation/runtime lanes the operational gate requires to be *present*,
/// mirroring the authoritative `prismui` GATE_LANES set. An absent required
/// lane is itself a blocking condition (otherwise a truncated status object
/// would show a false green).
const REQUIRED_LANES: &[&str] = &[
    "L3_VALIDATION",
    "L5_DEMO_ENVELOPE",
    "VAULT_PROVIDER",
    "RELEASE_SIGNATURE",
    "CUDA_LINKAGE",
    "CUDA_KERNEL_TRUST",
    "MODEL_RUNTIME",
    "WARNING_POLICY",
    "NANL_HD",
    "NAS_GUARD",
    "CSE_HD_FIRMWARE_OS_BOOT",
    "CSE_HD_FULL_IMPLEMENTATION_AUDIT",
];

/// Release modes that must be present in `release_modes`.
const REQUIRED_RELEASE_MODES: &[&str] = &[
    "CORE_ISA",
    "AI_ASSISTED_OS",
    "GPU_ACCELERATED_CORE",
    "SECURE_PRODUCTION",
];

/// One validation/runtime lane.
#[derive(Debug, Clone, Deserialize)]
pub struct Lane {
    pub key: String,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub evidence: Option<String>,
    #[serde(default)]
    pub blockers: Vec<String>,
}

impl Lane {
    pub fn status_text(&self) -> &str {
        self.status.as_deref().unwrap_or("BLOCKED")
    }

    /// A lane is green only when it is PASS and carries no blockers.
    pub fn is_pass(&self) -> bool {
        self.status_text() == "PASS" && self.blockers.is_empty()
    }
}

/// The status document.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct StatusDoc {
    pub status_schema_version: Option<String>,
    #[serde(default)]
    pub lanes: Vec<Lane>,
    pub release_modes: Option<serde_json::Value>,
}

impl StatusDoc {
    pub fn from_json_str(text: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(text)
    }

    pub fn schema(&self) -> &str {
        self.status_schema_version
            .as_deref()
            .unwrap_or("UNKNOWN_SCHEMA")
    }

    /// Lanes that are not green, rendered as `KEY=STATUS`. Includes
    /// `KEY=MISSING` for any required lane absent from the document.
    pub fn blocking_lanes(&self) -> Vec<String> {
        let mut blocking: Vec<String> = self
            .lanes
            .iter()
            .filter(|lane| !lane.is_pass())
            .map(|lane| format!("{}={}", lane.key, lane.status_text()))
            .collect();
        for key in REQUIRED_LANES {
            if !self.lanes.iter().any(|lane| lane.key == *key) {
                blocking.push(format!("{key}=MISSING"));
            }
        }
        blocking
    }

    /// Whether any release mode carries blockers, or any required mode is
    /// absent from the document.
    fn release_modes_blocked(&self) -> bool {
        let Some(modes) = self
            .release_modes
            .as_ref()
            .and_then(|value| value.as_object())
        else {
            // Missing release_modes is itself a blocked condition.
            return true;
        };
        if REQUIRED_RELEASE_MODES
            .iter()
            .any(|mode| !modes.contains_key(*mode))
        {
            return true;
        }
        modes.values().any(|mode| {
            mode.get("blockers")
                .and_then(|value| value.as_array())
                .map(|blockers| !blockers.is_empty())
                .unwrap_or(false)
        })
    }

    /// Overall readiness. Mirrors the `prismui` rule: green only when no lane is
    /// blocking and no release mode carries blockers.
    pub fn overall(&self) -> &'static str {
        if self.lanes.is_empty() {
            return "BLOCKED";
        }
        if self.blocking_lanes().is_empty() && !self.release_modes_blocked() {
            "PASS"
        } else {
            "BLOCKED"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BLOCKED: &str = r#"{
      "status_schema_version": "PRISMCODE_OPERATIONALIZATION_STATUS_V0_1",
      "lanes": [
        {"key":"L3_VALIDATION","status":"PASS","blockers":[]},
        {"key":"CUDA_LINKAGE","status":"FAIL-CLOSED","blockers":["CUDA_RUNTIME_LINKED"]}
      ],
      "release_modes": {
        "CORE_ISA": {"blockers": ["MARKET_READY_TRUE"]}
      }
    }"#;

    const PASS: &str = r#"{
      "status_schema_version": "v1",
      "lanes": [
        {"key":"L3_VALIDATION","status":"PASS","blockers":[]},
        {"key":"L5_DEMO_ENVELOPE","status":"PASS","blockers":[]},
        {"key":"VAULT_PROVIDER","status":"PASS","blockers":[]},
        {"key":"RELEASE_SIGNATURE","status":"PASS","blockers":[]},
        {"key":"CUDA_LINKAGE","status":"PASS","blockers":[]},
        {"key":"CUDA_KERNEL_TRUST","status":"PASS","blockers":[]},
        {"key":"MODEL_RUNTIME","status":"PASS","blockers":[]},
        {"key":"WARNING_POLICY","status":"PASS","blockers":[]},
        {"key":"NANL_HD","status":"PASS","blockers":[]},
        {"key":"NAS_GUARD","status":"PASS","blockers":[]},
        {"key":"CSE_HD_FIRMWARE_OS_BOOT","status":"PASS","blockers":[]},
        {"key":"CSE_HD_FULL_IMPLEMENTATION_AUDIT","status":"PASS","blockers":[]}
      ],
      "release_modes": {
        "CORE_ISA": {"blockers": []},
        "AI_ASSISTED_OS": {"blockers": []},
        "GPU_ACCELERATED_CORE": {"blockers": []},
        "SECURE_PRODUCTION": {"blockers": []}
      }
    }"#;

    #[test]
    fn blocked_doc_is_blocked() {
        let doc = StatusDoc::from_json_str(BLOCKED).unwrap();
        assert_eq!(doc.overall(), "BLOCKED");
        // The failed CUDA lane plus the absent required lanes are all reported.
        assert!(doc
            .blocking_lanes()
            .contains(&"CUDA_LINKAGE=FAIL-CLOSED".to_string()));
        assert!(doc.blocking_lanes().iter().any(|b| b.ends_with("=MISSING")));
    }

    #[test]
    fn clean_doc_is_pass() {
        let doc = StatusDoc::from_json_str(PASS).unwrap();
        assert_eq!(doc.overall(), "PASS");
        assert!(doc.blocking_lanes().is_empty());
    }

    #[test]
    fn truncated_doc_with_missing_required_lanes_is_blocked() {
        // A status object carrying only one PASS lane must NOT show green: the
        // 11 absent required lanes + 3 absent release modes block it, matching
        // the authoritative prismui rule.
        let doc = StatusDoc::from_json_str(
            r#"{"status_schema_version":"v1","lanes":[{"key":"L3_VALIDATION","status":"PASS","blockers":[]}],"release_modes":{"CORE_ISA":{"blockers":[]}}}"#,
        )
        .unwrap();
        assert_eq!(doc.overall(), "BLOCKED");
        assert!(doc.blocking_lanes().iter().any(|b| b.ends_with("=MISSING")));
    }

    #[test]
    fn empty_lanes_block() {
        let doc = StatusDoc::default();
        assert_eq!(doc.overall(), "BLOCKED");
    }
}
