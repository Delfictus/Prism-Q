use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WarningScope {
    Shipped,
    NonShipped,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WarningRecord {
    pub warning_id: String,
    pub crate_name: String,
    pub file: String,
    pub line: Option<u32>,
    pub warning_kind: String,
    pub scope: WarningScope,
    pub security_relevant: bool,
    pub runtime_relevant: bool,
    pub cuda_relevant: bool,
    pub model_relevant: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WarningWaiver {
    pub waiver_id: String,
    pub file: String,
    pub line: Option<u32>,
    pub owner: String,
    pub rationale: String,
    pub expiry_condition: String,
    pub market_impact: String,
    pub scope: WarningScope,
}

pub fn validate_market_warning_policy_text(policy: &str) -> Result<(), String> {
    for required in [
        "name = \"PRISMCODE_MARKET_RELEASE_WARNING_POLICY_V0_1\"",
        "blanket_allow_warnings_allowed = false",
        "blanket_allow_dead_code_allowed = false",
        "shipped_crates_must_be_warning_free = true",
        "xtask_warnings_allowed_only_with_formal_waiver = true",
        "warning_waiver_expiry_required = true",
        "warning_owner_required = true",
        "warning_market_impact_required = true",
        "prismcode = \"deny_warnings\"",
        "xtask = \"zero_or_waived\"",
    ] {
        if !policy.contains(required) {
            return Err(format!("warning policy missing required field: {required}"));
        }
    }
    Ok(())
}

pub fn source_has_blanket_warning_suppression(source: &str) -> bool {
    source.lines().any(|line| {
        let compact: String = line.chars().filter(|ch| !ch.is_whitespace()).collect();
        matches!(
            compact.as_str(),
            "#![allow(warnings)]"
                | "#![allow(dead_code)]"
                | "#[allow(warnings)]"
                | "#[allow(dead_code)]"
        )
    })
}

pub fn path_scope(path: &str) -> WarningScope {
    let path = path.replace('\\', "/");
    if path.starts_with("crates/prismcode/src/") {
        WarningScope::Shipped
    } else {
        WarningScope::NonShipped
    }
}

pub fn validate_warning_records(
    warnings: &[WarningRecord],
    waivers: &[WarningWaiver],
) -> Result<(), String> {
    for warning in warnings {
        if warning.scope == WarningScope::Shipped
            || path_scope(&warning.file) == WarningScope::Shipped
        {
            return Err(format!(
                "unwaived shipped warning rejected: {}:{}",
                warning.file,
                warning.line.unwrap_or(0)
            ));
        }
        let waiver = waivers
            .iter()
            .find(|waiver| waiver.file == warning.file && waiver.line == warning.line)
            .ok_or_else(|| {
                format!(
                    "non-shipped warning lacks formal waiver: {}:{}",
                    warning.file,
                    warning.line.unwrap_or(0)
                )
            })?;
        validate_waiver(waiver)?;
        if warning.security_relevant || warning.cuda_relevant || warning.model_relevant {
            if waiver.market_impact.trim().is_empty() {
                return Err(format!(
                    "critical warning waiver missing market impact: {}",
                    waiver.waiver_id
                ));
            }
        }
    }
    Ok(())
}

pub fn validate_waiver(waiver: &WarningWaiver) -> Result<(), String> {
    if waiver.scope == WarningScope::Shipped || path_scope(&waiver.file) == WarningScope::Shipped {
        return Err(format!(
            "waiver incorrectly marks shipped file as non-shipped: {}",
            waiver.file
        ));
    }
    if waiver.line.is_none() {
        return Err(format!("waiver missing file/line: {}", waiver.waiver_id));
    }
    for (field, value) in [
        ("owner", waiver.owner.as_str()),
        ("rationale", waiver.rationale.as_str()),
        ("expiry_condition", waiver.expiry_condition.as_str()),
        ("market_impact", waiver.market_impact.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(format!("waiver missing {field}: {}", waiver.waiver_id));
        }
    }
    Ok(())
}
