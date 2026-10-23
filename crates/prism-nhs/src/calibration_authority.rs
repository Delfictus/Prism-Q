//! Evidence-bound, manifest-driven calibration admission authority.
//!
//! This module deliberately separates five questions that a single threshold
//! cannot answer safely:
//!
//! 1. Was a physical calibration actually measured and hash-bound?
//! 2. Does that calibration apply to the requested thermodynamic/Hamiltonian
//!    domain?
//! 3. If transfer is allowed, how does its uncertainty change?
//! 4. Are the campaign's acceptance gates statistically capable of passing a
//!    correct, sufficiently precise experiment under their declared dependence
//!    structure?
//! 5. Are accuracy claims based on interval inclusion inside a physical
//!    equivalence margin, rather than failure to reject a difference?
//!
//! The module never invents a missing measurement and never loosens a physical
//! margin to make a campaign pass.  It solves for the precision/sampling needed
//! to satisfy a frozen physical margin.  Statistical critical values are
//! derived from manifest-declared family size, confidence, power and dependence
//! policy.  All target-specific facts remain in content-addressed manifests.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

pub const POLICY_SCHEMA: &str = "prismos_calibration_admission_policy";
pub const CALIBRATION_DOMAIN_SCHEMA: &str = "prismos_calibration_domain";
pub const RUN_MANIFEST_SCHEMA: &str = "prismos_calibration_run_manifest";
pub const ADMISSION_RECEIPT_SCHEMA: &str = "prismos_calibration_admission_receipt";
pub const EFFECTIVE_CONTRACT_SCHEMA: &str = "prismos_effective_calibration_contract";
pub const SCHEMA_VERSION: u32 = 2;

// =========================================================================
// Numeric primitives
// =========================================================================

fn regularized_gamma_p(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        let mut ap = a;
        let mut sum = 1.0 / a;
        let mut delta = sum;
        for _ in 0..500 {
            ap += 1.0;
            delta *= x / ap;
            sum += delta;
            if delta.abs() < sum.abs() * 1.0e-16 {
                break;
            }
        }
        sum * (-x + a * x.ln() - ln_gamma(a)).exp()
    } else {
        let tiny = 1.0e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..500 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() < 1.0e-16 {
                break;
            }
        }
        1.0 - (-x + a * x.ln() - ln_gamma(a)).exp() * h
    }
}

fn ln_gamma(x: f64) -> f64 {
    const COEFFICIENTS: [f64; 6] = [
        76.180_091_729_471_46,
        -86.505_320_329_416_77,
        24.014_098_240_830_91,
        -1.231_739_572_450_155,
        0.120_865_097_386_617_9e-2,
        -0.539_523_938_495_3e-5,
    ];
    let mut y = x;
    let temporary = x + 5.5 - (x + 0.5) * (x + 5.5).ln();
    let mut series = 1.000_000_000_190_015;
    for coefficient in COEFFICIENTS {
        y += 1.0;
        series += coefficient / y;
    }
    -temporary + (2.506_628_274_631_000_5 * series / x).ln()
}

fn erf(x: f64) -> f64 {
    if x < 0.0 {
        -regularized_gamma_p(0.5, x * x)
    } else {
        regularized_gamma_p(0.5, x * x)
    }
}

pub fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

fn beta_continued_fraction(a: f64, b: f64, x: f64) -> f64 {
    let tiny = 1.0e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..500 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let first = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + first * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + first / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        h *= d * c;
        let second = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + second * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + second / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < 1.0e-15 {
            break;
        }
    }
    h
}

fn regularized_incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let log_beta_term = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (-x).ln_1p();
    let beta_term = log_beta_term.exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        beta_term * beta_continued_fraction(a, b, x) / a
    } else {
        1.0 - beta_term * beta_continued_fraction(b, a, 1.0 - x) / b
    }
}

pub fn student_t_two_sided(k: f64, degrees_of_freedom: f64) -> f64 {
    if k <= 0.0 {
        return 0.0;
    }
    if degrees_of_freedom <= 0.0 {
        return f64::NAN;
    }
    let x = degrees_of_freedom / (degrees_of_freedom + k * k);
    1.0 - regularized_incomplete_beta(degrees_of_freedom / 2.0, 0.5, x)
}

fn student_t_cdf(t: f64, degrees_of_freedom: f64) -> f64 {
    if t == 0.0 {
        0.5
    } else {
        let central = student_t_two_sided(t.abs(), degrees_of_freedom);
        if t > 0.0 {
            0.5 + 0.5 * central
        } else {
            0.5 - 0.5 * central
        }
    }
}

fn quantile(mut cdf: impl FnMut(f64) -> f64, probability: f64) -> Result<f64> {
    if !(0.0..1.0).contains(&probability) {
        bail!("quantile probability must lie strictly inside (0, 1)");
    }
    let mut low = -1.0;
    let mut high = 1.0;
    for _ in 0..200 {
        if cdf(low) <= probability && cdf(high) >= probability {
            break;
        }
        low *= 2.0;
        high *= 2.0;
    }
    if cdf(low) > probability || cdf(high) < probability {
        bail!("failed to bracket distribution quantile");
    }
    for _ in 0..300 {
        let middle = 0.5 * (low + high);
        if cdf(middle) < probability {
            low = middle;
        } else {
            high = middle;
        }
        if (high - low).abs() <= 1.0e-13 * high.abs().max(1.0) {
            break;
        }
    }
    Ok(0.5 * (low + high))
}

fn validate_probability(label: &str, value: f64) -> Result<()> {
    if !value.is_finite() || !(0.0..1.0).contains(&value) {
        bail!("{label} must lie strictly inside (0, 1)");
    }
    Ok(())
}

fn validate_positive(label: &str, value: f64) -> Result<()> {
    if !value.is_finite() || value <= 0.0 {
        bail!("{label} must be finite and positive");
    }
    Ok(())
}

fn validate_nonnegative(label: &str, value: f64) -> Result<()> {
    if !value.is_finite() || value < 0.0 {
        bail!("{label} must be finite and non-negative");
    }
    Ok(())
}

fn validate_identifier(label: &str, value: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > 256
        || !value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.' | b':' | b'/')
        })
    {
        bail!("{label} is empty or contains unsupported characters");
    }
    Ok(())
}

pub fn validate_sha256(label: &str, value: &str) -> Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{label} must be a 64-character hexadecimal SHA-256");
    }
    let mut bytes = value.bytes();
    let first = bytes.next().context("SHA-256 unexpectedly empty")?;
    if bytes.all(|byte| byte.eq_ignore_ascii_case(&first)) {
        bail!("{label} is a reserved placeholder digest");
    }
    Ok(())
}

// =========================================================================
// Evidence and source provenance
// =========================================================================

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceRole {
    PhysicalCalibration,
    EstimatorUncertainty,
    IntegrityMeasurement,
    ApplicabilitySensitivity,
    GateSource,
    StatisticalMethod,
    DependenceModel,
    PhysicalResponse,
    SourceEpoch,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceFormat {
    Raw,
    Json,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceBinding {
    pub path: String,
    pub sha256: String,
    pub size_bytes: u64,
    pub role: EvidenceRole,
    pub format: EvidenceFormat,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_schema_kind: Option<String>,
}

impl EvidenceBinding {
    fn validate(&self, id: &str) -> Result<()> {
        validate_identifier("evidence id", id)?;
        if !Path::new(&self.path).is_absolute() {
            bail!("evidence {id:?} path must be absolute");
        }
        validate_sha256(&format!("evidence {id:?} SHA-256"), &self.sha256)?;
        if self.size_bytes == 0 {
            bail!("evidence {id:?} must be non-empty");
        }
        if self.format == EvidenceFormat::Json && self.expected_schema_kind.is_none() {
            bail!("JSON evidence {id:?} requires expected_schema_kind");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerifiedEvidence {
    pub evidence_id: String,
    pub path: String,
    pub sha256: String,
    pub size_bytes: u64,
    pub role: EvidenceRole,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema_kind: Option<String>,
    /// A gate-source JSON artifact must publish the exact, content-addressed
    /// definitions it authorizes.  Keeping this projection in the verified
    /// evidence row prevents a run manifest from citing the right file while
    /// silently substituting different threshold semantics.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub gate_definition_sha256s: BTreeMap<String, String>,
}

/// Independently rehash every declared evidence member.  No symlinks, empty
/// files, missing files, size drift, digest drift or schema drift are accepted.
pub fn verify_evidence_bindings(
    collections: &[&BTreeMap<String, EvidenceBinding>],
) -> Result<Vec<VerifiedEvidence>> {
    let mut verified = Vec::new();
    let mut seen_ids = BTreeSet::new();
    let mut seen_paths = BTreeSet::new();
    for collection in collections {
        for (id, binding) in *collection {
            binding.validate(id)?;
            if !seen_ids.insert(id.clone()) {
                bail!("evidence id {id:?} is declared more than once across input documents");
            }
            if !seen_paths.insert(binding.path.clone()) {
                bail!(
                    "evidence path {:?} is aliased by multiple evidence ids",
                    binding.path
                );
            }
            let metadata = std::fs::symlink_metadata(&binding.path)
                .with_context(|| format!("stat evidence {id:?} at {}", binding.path))?;
            if metadata.file_type().is_symlink() || !metadata.is_file() {
                bail!("evidence {id:?} must be a regular non-symlink file");
            }
            if metadata.len() != binding.size_bytes {
                bail!(
                    "evidence {id:?} size mismatch: declared {}, measured {}",
                    binding.size_bytes,
                    metadata.len()
                );
            }
            let bytes = std::fs::read(&binding.path)
                .with_context(|| format!("read evidence {id:?} at {}", binding.path))?;
            let measured = format!("{:x}", Sha256::digest(&bytes));
            if !measured.eq_ignore_ascii_case(&binding.sha256) {
                bail!(
                    "evidence {id:?} digest mismatch: declared {}, measured {measured}",
                    binding.sha256
                );
            }
            let (schema_kind, gate_definition_sha256s) = if binding.format == EvidenceFormat::Json {
                let document: serde_json::Value = serde_json::from_slice(&bytes)
                    .with_context(|| format!("parse JSON evidence {id:?}"))?;
                let schema = document
                    .get("schema_kind")
                    .and_then(serde_json::Value::as_str)
                    .with_context(|| format!("JSON evidence {id:?} lacks schema_kind"))?;
                if binding.expected_schema_kind.as_deref() != Some(schema) {
                    bail!(
                        "JSON evidence {id:?} schema mismatch: expected {:?}, measured {schema:?}",
                        binding.expected_schema_kind
                    );
                }
                let gate_definitions = if binding.role == EvidenceRole::GateSource {
                    let digest_object = document
                        .get("gate_definition_sha256s")
                        .and_then(serde_json::Value::as_object)
                        .with_context(|| {
                            format!("gate-source evidence {id:?} lacks gate_definition_sha256s")
                        })?;
                    let definition_object = document
                        .get("gate_definitions")
                        .and_then(serde_json::Value::as_object)
                        .with_context(|| {
                            format!("gate-source evidence {id:?} lacks structured gate_definitions")
                        })?;
                    let digest_keys: BTreeSet<_> = digest_object.keys().cloned().collect();
                    let definition_keys: BTreeSet<_> = definition_object.keys().cloned().collect();
                    if digest_keys != definition_keys {
                        bail!(
                            "gate-source evidence {id:?} definition/hash keys differ: missing_hashes={:?} missing_definitions={:?}",
                            definition_keys.difference(&digest_keys).collect::<Vec<_>>(),
                            digest_keys.difference(&definition_keys).collect::<Vec<_>>()
                        );
                    }
                    let mut definitions = BTreeMap::new();
                    for (gate_id, digest) in digest_object {
                        validate_identifier("gate-source gate id", gate_id)?;
                        let digest = digest.as_str().with_context(|| {
                            format!(
                                "gate-source evidence {id:?} definition for {gate_id:?} is not a string"
                            )
                        })?;
                        validate_sha256(
                            &format!("gate-source evidence {id:?} definition for {gate_id:?}"),
                            digest,
                        )?;
                        let source_definition = &definition_object[gate_id];
                        let recomputed = gate_definition_value_sha256(source_definition)?;
                        if !digest.eq_ignore_ascii_case(&recomputed) {
                            bail!(
                                "gate-source evidence {id:?} self-published definition digest mismatch for {gate_id:?}: published={digest} recomputed={recomputed}"
                            );
                        }
                        definitions.insert(gate_id.clone(), digest.to_ascii_lowercase());
                    }
                    if definitions.is_empty() {
                        bail!("gate-source evidence {id:?} declares no gate definitions");
                    }
                    definitions
                } else {
                    BTreeMap::new()
                };
                (Some(schema.to_string()), gate_definitions)
            } else {
                if binding.role == EvidenceRole::GateSource {
                    bail!("gate-source evidence {id:?} must be JSON");
                }
                (None, BTreeMap::new())
            };
            verified.push(VerifiedEvidence {
                evidence_id: id.clone(),
                path: binding.path.clone(),
                sha256: measured,
                size_bytes: metadata.len(),
                role: binding.role.clone(),
                schema_kind,
                gate_definition_sha256s,
            });
        }
    }
    verified.sort_by(|left, right| left.evidence_id.cmp(&right.evidence_id));
    Ok(verified)
}

fn verified_ids(evidence: &[VerifiedEvidence]) -> BTreeSet<String> {
    evidence.iter().map(|row| row.evidence_id.clone()).collect()
}

fn verified_roles(evidence: &[VerifiedEvidence]) -> Result<BTreeMap<String, EvidenceRole>> {
    let mut roles = BTreeMap::new();
    for row in evidence {
        if roles
            .insert(row.evidence_id.clone(), row.role.clone())
            .is_some()
        {
            bail!(
                "verified evidence contains duplicate id {:?}",
                row.evidence_id
            );
        }
    }
    Ok(roles)
}

fn require_evidence_role(
    roles: &BTreeMap<String, EvidenceRole>,
    evidence_id: &str,
    expected: EvidenceRole,
    context: &str,
) -> Result<()> {
    let measured = roles
        .get(evidence_id)
        .with_context(|| format!("{context} references unverified evidence {evidence_id:?}"))?;
    if measured != &expected {
        bail!("{context} evidence {evidence_id:?} has role {measured:?}, expected {expected:?}");
    }
    Ok(())
}

fn require_verified_evidence(
    ids: &BTreeSet<String>,
    evidence_id: &str,
    context: &str,
) -> Result<()> {
    if !ids.contains(evidence_id) {
        bail!("{context} references unverified evidence {evidence_id:?}");
    }
    Ok(())
}

// =========================================================================
// Applicability policy and calibration domain
// =========================================================================

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AxisKind {
    ExactIdentity,
    Continuous,
    BoundedInterval,
    CategoricalSet,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "model", rename_all = "snake_case")]
pub enum DependenceModel {
    /// No covariance claim is needed.  Joint pass probability uses the union
    /// bound and simultaneous interval critical values use Bonferroni.
    ArbitraryDependenceUnionBound,
    /// Exact product/Sidak composition is allowed only when the identified
    /// evidence artifact verifies independence for this exact design.
    Independent { evidence_id: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationAdmissionPolicy {
    pub schema_kind: String,
    pub schema_version: u32,
    pub policy_id: String,
    pub required_axes: BTreeMap<String, AxisKind>,
    pub required_gate_ids: BTreeSet<String>,
    pub simultaneous_interval_confidence: f64,
    pub minimum_campaign_power: f64,
    pub maximum_uncertainty_inflation: f64,
    pub campaign_dependence: DependenceModel,
    #[serde(default)]
    pub evidence: BTreeMap<String, EvidenceBinding>,
}

impl CalibrationAdmissionPolicy {
    pub fn validate(&self, evidence_ids: &BTreeSet<String>) -> Result<()> {
        if self.schema_kind != POLICY_SCHEMA || self.schema_version != SCHEMA_VERSION {
            bail!("unsupported calibration-admission policy schema");
        }
        validate_identifier("policy_id", &self.policy_id)?;
        if self.required_axes.is_empty() {
            bail!("policy requires at least one applicability axis");
        }
        if self.required_gate_ids.is_empty() {
            bail!("policy requires at least one acceptance gate");
        }
        for axis in self.required_axes.keys() {
            validate_identifier("required axis", axis)?;
        }
        for gate in &self.required_gate_ids {
            validate_identifier("required gate", gate)?;
        }
        validate_probability(
            "simultaneous_interval_confidence",
            self.simultaneous_interval_confidence,
        )?;
        validate_probability("minimum_campaign_power", self.minimum_campaign_power)?;
        if !self.maximum_uncertainty_inflation.is_finite()
            || self.maximum_uncertainty_inflation < 1.0
        {
            bail!("maximum_uncertainty_inflation must be finite and at least 1");
        }
        if let DependenceModel::Independent { evidence_id } = &self.campaign_dependence {
            require_verified_evidence(evidence_ids, evidence_id, "campaign dependence")?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "model", rename_all = "snake_case")]
pub enum ContinuousExtrapolation {
    RecalibrationOnly,
    EvidenceBoundLinear {
        limit: f64,
        additional_standard_uncertainty_per_unit: BTreeMap<String, f64>,
        sensitivity_evidence_id: String,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AxisConstraint {
    ExactIdentity {
        measured: String,
    },
    Continuous {
        measured: f64,
        tolerance: f64,
        units: String,
        extrapolation: ContinuousExtrapolation,
    },
    BoundedInterval {
        low: f64,
        high: f64,
        units: String,
    },
    CategoricalSet {
        admissible: Vec<String>,
    },
}

impl AxisConstraint {
    fn kind(&self) -> AxisKind {
        match self {
            AxisConstraint::ExactIdentity { .. } => AxisKind::ExactIdentity,
            AxisConstraint::Continuous { .. } => AxisKind::Continuous,
            AxisConstraint::BoundedInterval { .. } => AxisKind::BoundedInterval,
            AxisConstraint::CategoricalSet { .. } => AxisKind::CategoricalSet,
        }
    }

    fn validate(&self, name: &str, evidence_ids: &BTreeSet<String>) -> Result<()> {
        match self {
            AxisConstraint::ExactIdentity { measured } => {
                if measured.trim().is_empty() {
                    bail!("axis {name:?} exact identity is empty");
                }
            }
            AxisConstraint::Continuous {
                measured,
                tolerance,
                units,
                extrapolation,
            } => {
                if !measured.is_finite() {
                    bail!("axis {name:?} measured value must be finite");
                }
                validate_nonnegative(&format!("axis {name:?} tolerance"), *tolerance)?;
                if units.trim().is_empty() {
                    bail!("axis {name:?} units are empty");
                }
                if let ContinuousExtrapolation::EvidenceBoundLinear {
                    limit,
                    additional_standard_uncertainty_per_unit,
                    sensitivity_evidence_id,
                } = extrapolation
                {
                    if !limit.is_finite() || limit < tolerance {
                        bail!("axis {name:?} extrapolation limit must be >= tolerance");
                    }
                    if additional_standard_uncertainty_per_unit.is_empty() {
                        bail!("axis {name:?} has no quantity sensitivity coefficients");
                    }
                    for (quantity, coefficient) in additional_standard_uncertainty_per_unit {
                        validate_identifier("sensitivity quantity", quantity)?;
                        validate_nonnegative(
                            &format!("axis {name:?} sensitivity for {quantity:?}"),
                            *coefficient,
                        )?;
                    }
                    require_verified_evidence(
                        evidence_ids,
                        sensitivity_evidence_id,
                        &format!("axis {name:?} sensitivity"),
                    )?;
                }
            }
            AxisConstraint::BoundedInterval { low, high, units } => {
                if !low.is_finite() || !high.is_finite() || high < low {
                    bail!("axis {name:?} interval must be finite with high >= low");
                }
                if units.trim().is_empty() {
                    bail!("axis {name:?} units are empty");
                }
            }
            AxisConstraint::CategoricalSet { admissible } => {
                if admissible.is_empty() || admissible.iter().any(|value| value.trim().is_empty()) {
                    bail!("axis {name:?} categorical set is empty or contains an empty value");
                }
                let unique: BTreeSet<_> = admissible.iter().collect();
                if unique.len() != admissible.len() {
                    bail!("axis {name:?} categorical set contains duplicates");
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AxisValue {
    Text(String),
    Number(f64),
}

impl AxisValue {
    fn number(&self, axis: &str) -> Result<f64> {
        match self {
            AxisValue::Number(value) if value.is_finite() => Ok(*value),
            AxisValue::Number(_) => bail!("run axis {axis:?} is non-finite"),
            AxisValue::Text(value) => {
                bail!("run axis {axis:?} requires a number, got {value:?}")
            }
        }
    }

    fn text(&self, axis: &str) -> Result<&str> {
        match self {
            AxisValue::Text(value) if !value.trim().is_empty() => Ok(value),
            AxisValue::Text(_) => bail!("run axis {axis:?} is empty"),
            AxisValue::Number(_) => bail!("run axis {axis:?} requires a string identity"),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibratedQuantity {
    pub estimate: f64,
    pub standard_uncertainty: f64,
    pub units: String,
    pub estimator: String,
    pub effective_sample_size: f64,
    pub uncertainty_evidence_id: String,
}

impl CalibratedQuantity {
    fn validate(&self, name: &str, evidence_ids: &BTreeSet<String>) -> Result<()> {
        if !self.estimate.is_finite() {
            bail!("calibrated quantity {name:?} estimate is non-finite");
        }
        validate_positive(
            &format!("calibrated quantity {name:?} standard_uncertainty"),
            self.standard_uncertainty,
        )?;
        validate_positive(
            &format!("calibrated quantity {name:?} effective_sample_size"),
            self.effective_sample_size,
        )?;
        if self.units.trim().is_empty() || self.estimator.trim().is_empty() {
            bail!("calibrated quantity {name:?} requires units and estimator");
        }
        require_verified_evidence(
            evidence_ids,
            &self.uncertainty_evidence_id,
            &format!("calibrated quantity {name:?}"),
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CalibrationEvidenceState {
    EvidenceNotFound {
        reason: String,
    },
    Measured {
        measurement_evidence_id: String,
        engine_sha256: String,
        source_epoch_sha256: String,
        quantities: BTreeMap<String, CalibratedQuantity>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationDomain {
    pub schema_kind: String,
    pub schema_version: u32,
    pub calibration_id: String,
    pub calibration: CalibrationEvidenceState,
    pub axes: BTreeMap<String, AxisConstraint>,
    #[serde(default)]
    pub evidence: BTreeMap<String, EvidenceBinding>,
}

impl CalibrationDomain {
    pub fn validate(
        &self,
        policy: &CalibrationAdmissionPolicy,
        evidence_ids: &BTreeSet<String>,
    ) -> Result<()> {
        if self.schema_kind != CALIBRATION_DOMAIN_SCHEMA || self.schema_version != SCHEMA_VERSION {
            bail!("unsupported calibration-domain schema");
        }
        validate_identifier("calibration_id", &self.calibration_id)?;
        let domain_keys: BTreeSet<_> = self.axes.keys().cloned().collect();
        let required_keys: BTreeSet<_> = policy.required_axes.keys().cloned().collect();
        if domain_keys != required_keys {
            bail!(
                "calibration-domain axes differ from policy: missing={:?} extra={:?}",
                required_keys.difference(&domain_keys).collect::<Vec<_>>(),
                domain_keys.difference(&required_keys).collect::<Vec<_>>()
            );
        }
        for (name, constraint) in &self.axes {
            constraint.validate(name, evidence_ids)?;
            if policy.required_axes.get(name) != Some(&constraint.kind()) {
                bail!("axis {name:?} constraint kind differs from policy");
            }
        }
        match &self.calibration {
            CalibrationEvidenceState::EvidenceNotFound { reason } => {
                if reason.trim().is_empty() {
                    bail!("EVIDENCE_NOT_FOUND calibration requires a reason");
                }
            }
            CalibrationEvidenceState::Measured {
                measurement_evidence_id,
                engine_sha256,
                source_epoch_sha256,
                quantities,
            } => {
                validate_sha256("calibration engine_sha256", engine_sha256)?;
                validate_sha256("calibration source_epoch_sha256", source_epoch_sha256)?;
                require_verified_evidence(
                    evidence_ids,
                    measurement_evidence_id,
                    "calibration measurement",
                )?;
                if quantities.is_empty() {
                    bail!("measured calibration contains no quantities");
                }
                for (name, quantity) in quantities {
                    validate_identifier("calibrated quantity name", name)?;
                    quantity.validate(name, evidence_ids)?;
                }
            }
        }
        Ok(())
    }
}

// =========================================================================
// Run gate model
// =========================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum EvidencedScalar {
    Measured {
        value: f64,
        units: String,
        evidence_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        effective_sample_size: Option<f64>,
    },
    EvidenceNotFound {
        units: String,
        reason: String,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum EvidencedBoolean {
    Measured { value: bool, evidence_id: String },
    EvidenceNotFound { reason: String },
}

impl EvidencedBoolean {
    fn validate(&self, label: &str, evidence_ids: &BTreeSet<String>) -> Result<()> {
        match self {
            EvidencedBoolean::Measured { evidence_id, .. } => {
                require_verified_evidence(evidence_ids, evidence_id, label)
            }
            EvidencedBoolean::EvidenceNotFound { reason } => {
                if reason.trim().is_empty() {
                    bail!("{label} EVIDENCE_NOT_FOUND requires a reason");
                }
                Ok(())
            }
        }
    }
}

impl EvidencedScalar {
    fn validate(&self, label: &str, evidence_ids: &BTreeSet<String>) -> Result<()> {
        match self {
            EvidencedScalar::Measured {
                value,
                units,
                evidence_id,
                effective_sample_size,
            } => {
                validate_positive(label, *value)?;
                if units.trim().is_empty() {
                    bail!("{label} units are empty");
                }
                require_verified_evidence(evidence_ids, evidence_id, label)?;
                if let Some(effective_sample_size) = effective_sample_size {
                    validate_positive(
                        &format!("{label} effective_sample_size"),
                        *effective_sample_size,
                    )?;
                }
            }
            EvidencedScalar::EvidenceNotFound { units, reason } => {
                if units.trim().is_empty() || reason.trim().is_empty() {
                    bail!("{label} EVIDENCE_NOT_FOUND requires units and reason");
                }
            }
        }
        Ok(())
    }

    fn measured(&self) -> Option<(f64, Option<f64>)> {
        match self {
            EvidencedScalar::Measured {
                value,
                effective_sample_size,
                ..
            } => Some((*value, *effective_sample_size)),
            EvidencedScalar::EvidenceNotFound { .. } => None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "distribution", rename_all = "snake_case")]
pub enum SamplingDistribution {
    NormalKnown,
    StudentT { degrees_of_freedom: f64 },
}

impl SamplingDistribution {
    fn validate(&self) -> Result<()> {
        if let SamplingDistribution::StudentT { degrees_of_freedom } = self {
            validate_positive("Student-t degrees_of_freedom", *degrees_of_freedom)?;
        }
        Ok(())
    }

    fn cdf(&self, value: f64) -> f64 {
        match self {
            SamplingDistribution::NormalKnown => normal_cdf(value),
            SamplingDistribution::StudentT { degrees_of_freedom } => {
                student_t_cdf(value, *degrees_of_freedom)
            }
        }
    }

    fn critical(&self, confidence: f64) -> Result<f64> {
        let probability = 0.5 * (1.0 + confidence);
        match self {
            SamplingDistribution::NormalKnown => quantile(normal_cdf, probability),
            SamplingDistribution::StudentT { degrees_of_freedom } => quantile(
                |value| student_t_cdf(value, *degrees_of_freedom),
                probability,
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GateRole {
    Accuracy,
    Precision,
    Integrity,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "statistic", rename_all = "snake_case")]
pub enum GateStatistic {
    /// A simultaneous two-sided confidence interval must fit wholly inside a
    /// physical equivalence margin.  This becomes harder, not easier, when the
    /// standard error increases.
    EquivalenceInterval {
        equivalence_half_width: f64,
        units: String,
        standard_error: EvidencedScalar,
        design_true_offset: f64,
        sampling_distribution: SamplingDistribution,
    },
    /// Pressure bias is converted to its first-order density consequence,
    /// |delta rho|/rho ~= kappa_T |delta P|.  Both kappa_T and the standard
    /// error of the trajectory mean must be measured and hash-bound.
    PressureDensityEquivalence {
        maximum_relative_density_shift: f64,
        isothermal_compressibility_per_pa: EvidencedScalar,
        isothermal_compressibility_standard_uncertainty_per_pa: EvidencedScalar,
        mean_pressure_standard_error_bar: EvidencedScalar,
        /// Evidence that the exact NPT state remains inside the locally linear
        /// equation-of-state regime over the resulting pressure interval.
        /// Without it, the first-order compressibility conversion is not an
        /// admissible substitute for direct density-response evaluation.
        linear_response_validity: EvidencedBoolean,
        design_true_offset_bar: f64,
        sampling_distribution: SamplingDistribution,
    },
    /// Retained only to diagnose and reject legacy accuracy rules of the form
    /// |estimate-target|/measured_SE <= threshold.
    LegacyMeasuredDenominator {
        threshold: f64,
        independent_units: u64,
    },
    Deterministic {
        outcome: EvidencedBoolean,
    },
}

impl GateStatistic {
    fn validate(&self, gate: &str, evidence_ids: &BTreeSet<String>) -> Result<()> {
        match self {
            GateStatistic::EquivalenceInterval {
                equivalence_half_width,
                units,
                standard_error,
                design_true_offset,
                sampling_distribution,
            } => {
                validate_positive(
                    &format!("gate {gate:?} equivalence_half_width"),
                    *equivalence_half_width,
                )?;
                if units.trim().is_empty() || !design_true_offset.is_finite() {
                    bail!("gate {gate:?} requires units and finite design_true_offset");
                }
                standard_error.validate(&format!("gate {gate:?} standard_error"), evidence_ids)?;
                sampling_distribution.validate()?;
            }
            GateStatistic::PressureDensityEquivalence {
                maximum_relative_density_shift,
                isothermal_compressibility_per_pa,
                isothermal_compressibility_standard_uncertainty_per_pa,
                mean_pressure_standard_error_bar,
                linear_response_validity,
                design_true_offset_bar,
                sampling_distribution,
            } => {
                validate_probability(
                    &format!("gate {gate:?} maximum_relative_density_shift"),
                    *maximum_relative_density_shift,
                )?;
                isothermal_compressibility_per_pa.validate(
                    &format!("gate {gate:?} isothermal_compressibility_per_pa"),
                    evidence_ids,
                )?;
                isothermal_compressibility_standard_uncertainty_per_pa.validate(
                    &format!(
                        "gate {gate:?} isothermal_compressibility_standard_uncertainty_per_pa"
                    ),
                    evidence_ids,
                )?;
                mean_pressure_standard_error_bar.validate(
                    &format!("gate {gate:?} mean_pressure_standard_error_bar"),
                    evidence_ids,
                )?;
                linear_response_validity.validate(
                    &format!("gate {gate:?} linear_response_validity"),
                    evidence_ids,
                )?;
                if !design_true_offset_bar.is_finite() {
                    bail!("gate {gate:?} design_true_offset_bar must be finite");
                }
                sampling_distribution.validate()?;
            }
            GateStatistic::LegacyMeasuredDenominator {
                threshold,
                independent_units,
            } => {
                validate_positive(&format!("gate {gate:?} threshold"), *threshold)?;
                if *independent_units < 2 {
                    bail!("gate {gate:?} needs at least two independent units");
                }
            }
            GateStatistic::Deterministic { outcome } => {
                outcome.validate(&format!("gate {gate:?} integrity outcome"), evidence_ids)?;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GateSpec {
    pub id: String,
    pub role: GateRole,
    pub source_evidence_id: String,
    pub statistical_method_evidence_id: String,
    /// Digest of the immutable gate-definition projection.  The cited gate
    /// source must publish this exact digest under this gate id.
    pub definition_sha256: String,
    /// Structured, immutable scientific predicate.  This carries the physical
    /// effect margin or integrity threshold and its derivation identity.  It is
    /// included in the gate-definition digest, so a threshold cannot change
    /// while preserving the gate's authorized identity.
    pub scientific_predicate: serde_json::Value,
    pub statistic: GateStatistic,
    pub replication_across: Vec<String>,
    pub dependence: DependenceModel,
    pub required_familywise_power: f64,
}

/// Content-address the immutable scientific definition of a gate while
/// deliberately excluding run-produced measurements.  The projection includes
/// the physical margin, estimator family, interval distribution, multiplicity
/// axes, dependence semantics and required power.  A gate-source artifact must
/// publish the resulting digest for the same gate id.
fn gate_definition_projection(gate: &GateSpec) -> serde_json::Value {
    let statistic = match &gate.statistic {
        GateStatistic::EquivalenceInterval {
            equivalence_half_width,
            units,
            design_true_offset,
            sampling_distribution,
            ..
        } => serde_json::json!({
            "statistic": "equivalence_interval",
            "equivalence_half_width": equivalence_half_width,
            "units": units,
            "design_true_offset": design_true_offset,
            "sampling_distribution": sampling_distribution,
            "estimator_uncertainty_required": true
        }),
        GateStatistic::PressureDensityEquivalence {
            maximum_relative_density_shift,
            design_true_offset_bar,
            sampling_distribution,
            ..
        } => serde_json::json!({
            "statistic": "pressure_density_equivalence",
            "maximum_relative_density_shift": maximum_relative_density_shift,
            "design_true_offset_bar": design_true_offset_bar,
            "sampling_distribution": sampling_distribution,
            "isothermal_compressibility_required": true,
            "compressibility_uncertainty_required": true,
            "pressure_mean_uncertainty_required": true,
            "linear_response_validity_required": true,
            "confidence_components_per_replicate": 2
        }),
        GateStatistic::LegacyMeasuredDenominator {
            threshold,
            independent_units,
        } => serde_json::json!({
            "statistic": "legacy_measured_denominator",
            "threshold": threshold,
            "independent_units": independent_units
        }),
        GateStatistic::Deterministic { .. } => serde_json::json!({
            "statistic": "deterministic",
            "measured_integrity_outcome_required": true
        }),
    };
    serde_json::json!({
        "schema_kind": "prismos_calibration_gate_definition",
        "schema_version": SCHEMA_VERSION,
        "id": gate.id,
        "role": gate.role,
        "source_evidence_id": gate.source_evidence_id,
        "statistical_method_evidence_id": gate.statistical_method_evidence_id,
        "scientific_predicate": gate.scientific_predicate,
        "statistic_definition": statistic,
        "replication_across": gate.replication_across,
        "dependence": gate.dependence,
        "required_familywise_power": gate.required_familywise_power
    })
}

fn gate_definition_value_sha256(projection: &serde_json::Value) -> Result<String> {
    let mut digest = Sha256::new();
    digest.update(b"PRISM_CALIBRATION_GATE_DEFINITION_V2\0");
    digest.update(serde_json::to_vec(&projection)?);
    Ok(format!("{:x}", digest.finalize()))
}

pub fn gate_definition_sha256(gate: &GateSpec) -> Result<String> {
    gate_definition_value_sha256(&gate_definition_projection(gate))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunManifest {
    pub schema_kind: String,
    pub schema_version: u32,
    pub campaign_id: String,
    pub axes: BTreeMap<String, AxisValue>,
    pub design: BTreeMap<String, u64>,
    pub gates: Vec<GateSpec>,
    #[serde(default)]
    pub evidence: BTreeMap<String, EvidenceBinding>,
}

impl RunManifest {
    pub fn validate(
        &self,
        policy: &CalibrationAdmissionPolicy,
        evidence_ids: &BTreeSet<String>,
    ) -> Result<()> {
        if self.schema_kind != RUN_MANIFEST_SCHEMA || self.schema_version != SCHEMA_VERSION {
            bail!("unsupported calibration run-manifest schema");
        }
        validate_identifier("campaign_id", &self.campaign_id)?;
        let run_axes: BTreeSet<_> = self.axes.keys().cloned().collect();
        let required_axes: BTreeSet<_> = policy.required_axes.keys().cloned().collect();
        if run_axes != required_axes {
            bail!(
                "run axes differ from policy: missing={:?} extra={:?}",
                required_axes.difference(&run_axes).collect::<Vec<_>>(),
                run_axes.difference(&required_axes).collect::<Vec<_>>()
            );
        }
        for (name, value) in &self.axes {
            match policy.required_axes[name] {
                AxisKind::ExactIdentity | AxisKind::CategoricalSet => {
                    value.text(name)?;
                }
                AxisKind::Continuous | AxisKind::BoundedInterval => {
                    value.number(name)?;
                }
            }
        }
        for (name, count) in &self.design {
            validate_identifier("design axis", name)?;
            if *count == 0 {
                bail!("design axis {name:?} is zero");
            }
        }
        if self.gates.is_empty() {
            bail!("run manifest declares no gates");
        }
        let gate_ids: BTreeSet<_> = self.gates.iter().map(|gate| gate.id.clone()).collect();
        if gate_ids.len() != self.gates.len() {
            bail!("run manifest contains duplicate gate ids");
        }
        if gate_ids != policy.required_gate_ids {
            bail!(
                "run gates differ from policy: missing={:?} extra={:?}",
                policy
                    .required_gate_ids
                    .difference(&gate_ids)
                    .collect::<Vec<_>>(),
                gate_ids
                    .difference(&policy.required_gate_ids)
                    .collect::<Vec<_>>()
            );
        }
        let minimum_gate_power = required_component_power(
            policy.minimum_campaign_power,
            self.gates.len() as u64,
            &policy.campaign_dependence,
        );
        for gate in &self.gates {
            validate_identifier("gate id", &gate.id)?;
            let predicate = gate.scientific_predicate.as_object().with_context(|| {
                format!("gate {:?} scientific_predicate must be an object", gate.id)
            })?;
            if predicate.is_empty()
                || predicate
                    .get("schema_kind")
                    .and_then(serde_json::Value::as_str)
                    != Some("prismos_scientific_gate_predicate")
                || predicate
                    .get("schema_version")
                    .and_then(serde_json::Value::as_u64)
                    != Some(SCHEMA_VERSION as u64)
                || predicate.get("gate_id").and_then(serde_json::Value::as_str)
                    != Some(gate.id.as_str())
                || predicate
                    .get("predicate")
                    .and_then(serde_json::Value::as_str)
                    .is_none()
                || predicate
                    .get("threshold_source")
                    .and_then(serde_json::Value::as_str)
                    .is_none()
            {
                bail!(
                    "gate {:?} scientific_predicate lacks the required schema, matching gate_id, predicate, or threshold_source",
                    gate.id
                );
            }
            validate_sha256(
                &format!("gate {:?} definition_sha256", gate.id),
                &gate.definition_sha256,
            )?;
            let recomputed_definition = gate_definition_sha256(gate)?;
            if !gate
                .definition_sha256
                .eq_ignore_ascii_case(&recomputed_definition)
            {
                bail!(
                    "gate {:?} definition digest mismatch: declared={} recomputed={}",
                    gate.id,
                    gate.definition_sha256,
                    recomputed_definition
                );
            }
            validate_probability(
                &format!("gate {:?} required_familywise_power", gate.id),
                gate.required_familywise_power,
            )?;
            if gate.required_familywise_power + 1.0e-15 < minimum_gate_power {
                bail!(
                    "gate {:?} required_familywise_power {} is below the campaign-derived minimum {} for {} gates",
                    gate.id,
                    gate.required_familywise_power,
                    minimum_gate_power,
                    self.gates.len()
                );
            }
            require_verified_evidence(
                evidence_ids,
                &gate.source_evidence_id,
                &format!("gate {:?} source", gate.id),
            )?;
            require_verified_evidence(
                evidence_ids,
                &gate.statistical_method_evidence_id,
                &format!("gate {:?} statistical method", gate.id),
            )?;
            if let DependenceModel::Independent { evidence_id } = &gate.dependence {
                require_verified_evidence(
                    evidence_ids,
                    evidence_id,
                    &format!("gate {:?} dependence", gate.id),
                )?;
            }
            gate.statistic.validate(&gate.id, evidence_ids)?;
            replicate_count(gate, &self.design)?;
        }
        Ok(())
    }
}

fn validate_evidence_roles(
    policy: &CalibrationAdmissionPolicy,
    domain: &CalibrationDomain,
    manifest: &RunManifest,
    verified_evidence: &[VerifiedEvidence],
) -> Result<()> {
    let roles = verified_roles(verified_evidence)?;
    if let DependenceModel::Independent { evidence_id } = &policy.campaign_dependence {
        require_evidence_role(
            &roles,
            evidence_id,
            EvidenceRole::DependenceModel,
            "campaign dependence",
        )?;
    }
    for (axis, constraint) in &domain.axes {
        if let AxisConstraint::Continuous {
            extrapolation:
                ContinuousExtrapolation::EvidenceBoundLinear {
                    sensitivity_evidence_id,
                    ..
                },
            ..
        } = constraint
        {
            require_evidence_role(
                &roles,
                sensitivity_evidence_id,
                EvidenceRole::ApplicabilitySensitivity,
                &format!("axis {axis:?} sensitivity"),
            )?;
        }
    }
    if let CalibrationEvidenceState::Measured {
        measurement_evidence_id,
        quantities,
        ..
    } = &domain.calibration
    {
        require_evidence_role(
            &roles,
            measurement_evidence_id,
            EvidenceRole::PhysicalCalibration,
            "physical calibration",
        )?;
        for (quantity, value) in quantities {
            require_evidence_role(
                &roles,
                &value.uncertainty_evidence_id,
                EvidenceRole::EstimatorUncertainty,
                &format!("calibrated quantity {quantity:?} uncertainty"),
            )?;
        }
    }
    for gate in &manifest.gates {
        require_evidence_role(
            &roles,
            &gate.source_evidence_id,
            EvidenceRole::GateSource,
            &format!("gate {:?} source", gate.id),
        )?;
        require_evidence_role(
            &roles,
            &gate.statistical_method_evidence_id,
            EvidenceRole::StatisticalMethod,
            &format!("gate {:?} statistical method", gate.id),
        )?;
        if let DependenceModel::Independent { evidence_id } = &gate.dependence {
            require_evidence_role(
                &roles,
                evidence_id,
                EvidenceRole::DependenceModel,
                &format!("gate {:?} dependence", gate.id),
            )?;
        }
        match &gate.statistic {
            GateStatistic::EquivalenceInterval { standard_error, .. } => {
                if let EvidencedScalar::Measured { evidence_id, .. } = standard_error {
                    require_evidence_role(
                        &roles,
                        evidence_id,
                        EvidenceRole::EstimatorUncertainty,
                        &format!("gate {:?} standard error", gate.id),
                    )?;
                }
            }
            GateStatistic::PressureDensityEquivalence {
                isothermal_compressibility_per_pa,
                isothermal_compressibility_standard_uncertainty_per_pa,
                mean_pressure_standard_error_bar,
                linear_response_validity,
                ..
            } => {
                if let EvidencedScalar::Measured { evidence_id, .. } =
                    isothermal_compressibility_per_pa
                {
                    require_evidence_role(
                        &roles,
                        evidence_id,
                        EvidenceRole::PhysicalResponse,
                        &format!("gate {:?} isothermal compressibility", gate.id),
                    )?;
                }
                if let EvidencedScalar::Measured { evidence_id, .. } =
                    isothermal_compressibility_standard_uncertainty_per_pa
                {
                    require_evidence_role(
                        &roles,
                        evidence_id,
                        EvidenceRole::EstimatorUncertainty,
                        &format!("gate {:?} compressibility uncertainty", gate.id),
                    )?;
                }
                if let EvidencedScalar::Measured { evidence_id, .. } =
                    mean_pressure_standard_error_bar
                {
                    require_evidence_role(
                        &roles,
                        evidence_id,
                        EvidenceRole::EstimatorUncertainty,
                        &format!("gate {:?} pressure standard error", gate.id),
                    )?;
                }
                if let EvidencedBoolean::Measured { evidence_id, .. } = linear_response_validity {
                    require_evidence_role(
                        &roles,
                        evidence_id,
                        EvidenceRole::PhysicalResponse,
                        &format!("gate {:?} linear-response validity", gate.id),
                    )?;
                }
            }
            GateStatistic::Deterministic {
                outcome: EvidencedBoolean::Measured { evidence_id, .. },
            } => {
                require_evidence_role(
                    &roles,
                    evidence_id,
                    EvidenceRole::IntegrityMeasurement,
                    &format!("gate {:?} integrity outcome", gate.id),
                )?;
            }
            GateStatistic::LegacyMeasuredDenominator { .. }
            | GateStatistic::Deterministic {
                outcome: EvidencedBoolean::EvidenceNotFound { .. },
            } => {}
        }
    }
    Ok(())
}

fn validate_gate_source_bindings(
    manifest: &RunManifest,
    verified_evidence: &[VerifiedEvidence],
) -> Result<()> {
    let sources: BTreeMap<_, _> = verified_evidence
        .iter()
        .filter(|row| row.role == EvidenceRole::GateSource)
        .map(|row| (row.evidence_id.as_str(), row))
        .collect();
    for gate in &manifest.gates {
        let source = sources
            .get(gate.source_evidence_id.as_str())
            .with_context(|| {
                format!(
                    "gate {:?} source {:?} was not verified as gate-source evidence",
                    gate.id, gate.source_evidence_id
                )
            })?;
        let published = source
            .gate_definition_sha256s
            .get(&gate.id)
            .with_context(|| {
                format!(
                    "gate-source evidence {:?} does not publish gate {:?}",
                    gate.source_evidence_id, gate.id
                )
            })?;
        if !published.eq_ignore_ascii_case(&gate.definition_sha256) {
            bail!(
                "gate {:?} definition is not authorized by source {:?}: source={} manifest={}",
                gate.id,
                gate.source_evidence_id,
                published,
                gate.definition_sha256
            );
        }
    }
    Ok(())
}

fn replicate_count(gate: &GateSpec, design: &BTreeMap<String, u64>) -> Result<u64> {
    let mut count = 1_u64;
    let mut axes = BTreeSet::new();
    for axis in &gate.replication_across {
        if !axes.insert(axis) {
            bail!("gate {:?} repeats replication axis {axis:?}", gate.id);
        }
        let value = design.get(axis).with_context(|| {
            format!(
                "gate {:?} replicates across undeclared design axis {axis:?}",
                gate.id
            )
        })?;
        count = count
            .checked_mul(*value)
            .with_context(|| format!("gate {:?} replication count overflow", gate.id))?;
    }
    Ok(count)
}

// =========================================================================
// Transferability and effective downstream calibration
// =========================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AxisFinding {
    pub axis: String,
    pub disposition: String,
    pub detail: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distance: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EffectiveQuantity {
    pub estimate: f64,
    pub units: String,
    pub baseline_standard_uncertainty: f64,
    pub effective_standard_uncertainty: f64,
    pub uncertainty_inflation_factor: f64,
    pub estimator: String,
    pub effective_sample_size: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "verdict", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TransferabilityVerdict {
    EvidenceRequired {
        findings: Vec<AxisFinding>,
    },
    InDomain {
        findings: Vec<AxisFinding>,
        effective_quantities: BTreeMap<String, EffectiveQuantity>,
    },
    BoundedExtrapolation {
        findings: Vec<AxisFinding>,
        effective_quantities: BTreeMap<String, EffectiveQuantity>,
        maximum_uncertainty_inflation: f64,
    },
    RecalibrationRequired {
        findings: Vec<AxisFinding>,
    },
}

impl TransferabilityVerdict {
    fn permits_reuse(&self) -> bool {
        matches!(
            self,
            TransferabilityVerdict::InDomain { .. }
                | TransferabilityVerdict::BoundedExtrapolation { .. }
        )
    }

    fn effective_quantities(&self) -> Option<&BTreeMap<String, EffectiveQuantity>> {
        match self {
            TransferabilityVerdict::InDomain {
                effective_quantities,
                ..
            }
            | TransferabilityVerdict::BoundedExtrapolation {
                effective_quantities,
                ..
            } => Some(effective_quantities),
            _ => None,
        }
    }
}

pub fn evaluate_transferability(
    policy: &CalibrationAdmissionPolicy,
    domain: &CalibrationDomain,
    manifest: &RunManifest,
    evidence_ids: &BTreeSet<String>,
) -> Result<TransferabilityVerdict> {
    policy.validate(evidence_ids)?;
    domain.validate(policy, evidence_ids)?;
    manifest.validate(policy, evidence_ids)?;

    let CalibrationEvidenceState::Measured { quantities, .. } = &domain.calibration else {
        return Ok(TransferabilityVerdict::EvidenceRequired {
            findings: vec![AxisFinding {
                axis: "*calibration*".to_string(),
                disposition: "EVIDENCE_NOT_FOUND".to_string(),
                detail: match &domain.calibration {
                    CalibrationEvidenceState::EvidenceNotFound { reason } => reason.clone(),
                    CalibrationEvidenceState::Measured { .. } => unreachable!(),
                },
                distance: None,
            }],
        });
    };

    let mut findings = Vec::new();
    let mut recalibration_required = false;
    let mut additional_variance: BTreeMap<String, f64> =
        quantities.keys().map(|name| (name.clone(), 0.0)).collect();

    for (name, constraint) in &domain.axes {
        let presented = manifest
            .axes
            .get(name)
            .with_context(|| format!("validated run unexpectedly lacks axis {name:?}"))?;
        match constraint {
            AxisConstraint::ExactIdentity { measured } => {
                let requested = presented.text(name)?;
                if requested == measured {
                    findings.push(AxisFinding {
                        axis: name.clone(),
                        disposition: "IN_DOMAIN".to_string(),
                        detail: format!("exact identity match: {measured}"),
                        distance: None,
                    });
                } else {
                    recalibration_required = true;
                    findings.push(AxisFinding {
                        axis: name.clone(),
                        disposition: "RECALIBRATION_REQUIRED".to_string(),
                        detail: format!(
                            "identity differs: measured {measured:?}, requested {requested:?}"
                        ),
                        distance: None,
                    });
                }
            }
            AxisConstraint::Continuous {
                measured,
                tolerance,
                units,
                extrapolation,
            } => {
                let requested = presented.number(name)?;
                let distance = (requested - measured).abs();
                if distance <= *tolerance {
                    findings.push(AxisFinding {
                        axis: name.clone(),
                        disposition: "IN_DOMAIN".to_string(),
                        detail: format!(
                            "{requested} {units} lies within {tolerance} {units} of measured {measured}"
                        ),
                        distance: Some(distance),
                    });
                } else {
                    match extrapolation {
                        ContinuousExtrapolation::RecalibrationOnly => {
                            recalibration_required = true;
                            findings.push(AxisFinding {
                                axis: name.clone(),
                                disposition: "RECALIBRATION_REQUIRED".to_string(),
                                detail:
                                    "no evidence-bound sensitivity model authorizes extrapolation"
                                        .to_string(),
                                distance: Some(distance),
                            });
                        }
                        ContinuousExtrapolation::EvidenceBoundLinear {
                            limit,
                            additional_standard_uncertainty_per_unit,
                            ..
                        } if distance <= *limit => {
                            let excess = distance - tolerance;
                            for quantity_name in quantities.keys() {
                                let coefficient = additional_standard_uncertainty_per_unit
                                    .get(quantity_name)
                                    .with_context(|| {
                                        format!(
                                            "axis {name:?} lacks sensitivity for calibrated quantity {quantity_name:?}"
                                        )
                                    })?;
                                let contribution = coefficient * excess;
                                *additional_variance
                                    .get_mut(quantity_name)
                                    .context("quantity variance accumulator missing")? +=
                                    contribution * contribution;
                            }
                            findings.push(AxisFinding {
                                axis: name.clone(),
                                disposition: "BOUNDED_EXTRAPOLATION".to_string(),
                                detail: format!(
                                    "{requested} {units} exceeds the in-domain tolerance by {excess} {units}; evidence-bound sensitivity variance was propagated"
                                ),
                                distance: Some(distance),
                            });
                        }
                        ContinuousExtrapolation::EvidenceBoundLinear { limit, .. } => {
                            recalibration_required = true;
                            findings.push(AxisFinding {
                                axis: name.clone(),
                                disposition: "RECALIBRATION_REQUIRED".to_string(),
                                detail: format!(
                                    "distance {distance} {units} exceeds evidence-bound extrapolation limit {limit} {units}"
                                ),
                                distance: Some(distance),
                            });
                        }
                    }
                }
            }
            AxisConstraint::BoundedInterval { low, high, units } => {
                let requested = presented.number(name)?;
                if requested >= *low && requested <= *high {
                    findings.push(AxisFinding {
                        axis: name.clone(),
                        disposition: "IN_DOMAIN".to_string(),
                        detail: format!("{requested} {units} lies in [{low}, {high}]"),
                        distance: None,
                    });
                } else {
                    recalibration_required = true;
                    findings.push(AxisFinding {
                        axis: name.clone(),
                        disposition: "RECALIBRATION_REQUIRED".to_string(),
                        detail: format!("{requested} {units} lies outside [{low}, {high}]"),
                        distance: None,
                    });
                }
            }
            AxisConstraint::CategoricalSet { admissible } => {
                let requested = presented.text(name)?;
                if admissible.iter().any(|value| value == requested) {
                    findings.push(AxisFinding {
                        axis: name.clone(),
                        disposition: "IN_DOMAIN".to_string(),
                        detail: format!("{requested:?} belongs to the measured category set"),
                        distance: None,
                    });
                } else {
                    recalibration_required = true;
                    findings.push(AxisFinding {
                        axis: name.clone(),
                        disposition: "RECALIBRATION_REQUIRED".to_string(),
                        detail: format!("{requested:?} is not in {admissible:?}"),
                        distance: None,
                    });
                }
            }
        }
    }

    if recalibration_required {
        return Ok(TransferabilityVerdict::RecalibrationRequired { findings });
    }

    let mut effective_quantities = BTreeMap::new();
    let mut maximum_inflation = 1.0_f64;
    for (name, quantity) in quantities {
        let effective_uncertainty =
            (quantity.standard_uncertainty.powi(2) + additional_variance[name]).sqrt();
        let inflation = effective_uncertainty / quantity.standard_uncertainty;
        maximum_inflation = maximum_inflation.max(inflation);
        effective_quantities.insert(
            name.clone(),
            EffectiveQuantity {
                estimate: quantity.estimate,
                units: quantity.units.clone(),
                baseline_standard_uncertainty: quantity.standard_uncertainty,
                effective_standard_uncertainty: effective_uncertainty,
                uncertainty_inflation_factor: inflation,
                estimator: quantity.estimator.clone(),
                effective_sample_size: quantity.effective_sample_size,
            },
        );
    }
    if maximum_inflation > policy.maximum_uncertainty_inflation {
        findings.push(AxisFinding {
            axis: "*aggregate_uncertainty*".to_string(),
            disposition: "RECALIBRATION_REQUIRED".to_string(),
            detail: format!(
                "effective uncertainty inflation {maximum_inflation:.6} exceeds policy ceiling {:.6}",
                policy.maximum_uncertainty_inflation
            ),
            distance: None,
        });
        return Ok(TransferabilityVerdict::RecalibrationRequired { findings });
    }
    if maximum_inflation > 1.0 + 1.0e-15 {
        Ok(TransferabilityVerdict::BoundedExtrapolation {
            findings,
            effective_quantities,
            maximum_uncertainty_inflation: maximum_inflation,
        })
    } else {
        Ok(TransferabilityVerdict::InDomain {
            findings,
            effective_quantities,
        })
    }
}

// =========================================================================
// Gate feasibility and sampling design
// =========================================================================

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum GateVerdict {
    Feasible,
    Infeasible,
    EvidenceRequired,
    InversionBlocked,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GateFeasibility {
    pub gate: String,
    pub role: GateRole,
    pub replicates: u64,
    pub dependence: DependenceModel,
    pub simultaneous_interval_confidence: f64,
    /// Number of stochastic confidence statements that must hold jointly for
    /// each replicate.  Pressure-density equivalence has two: the conservative
    /// compressibility bound and the pressure-mean interval.
    pub confidence_components_per_replicate: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub per_unit_interval_confidence: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub critical_value: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub physical_equivalence_half_width: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effective_standard_error: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit_pass_probability: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub familywise_pass_probability_lower_bound: Option<f64>,
    pub required_familywise_power: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maximum_standard_error_for_required_power: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub required_effective_sample_size: Option<f64>,
    pub verdict: GateVerdict,
    pub actions: Vec<String>,
}

fn per_unit_interval_confidence(
    familywise_confidence: f64,
    replicates: u64,
    dependence: &DependenceModel,
) -> f64 {
    match dependence {
        DependenceModel::ArbitraryDependenceUnionBound => {
            1.0 - (1.0 - familywise_confidence) / replicates as f64
        }
        DependenceModel::Independent { .. } => familywise_confidence.powf(1.0 / replicates as f64),
    }
}

fn compose_pass_probability(
    unit_probability: f64,
    replicates: u64,
    dependence: &DependenceModel,
) -> f64 {
    match dependence {
        DependenceModel::ArbitraryDependenceUnionBound => {
            (1.0 - replicates as f64 * (1.0 - unit_probability)).clamp(0.0, 1.0)
        }
        DependenceModel::Independent { .. } => unit_probability.powf(replicates as f64),
    }
}

fn required_component_power(
    familywise_power: f64,
    components: u64,
    dependence: &DependenceModel,
) -> f64 {
    debug_assert!(components > 0);
    match dependence {
        DependenceModel::ArbitraryDependenceUnionBound => {
            1.0 - (1.0 - familywise_power) / components as f64
        }
        DependenceModel::Independent { .. } => familywise_power.powf(1.0 / components as f64),
    }
}

fn equivalence_unit_power(
    half_width: f64,
    standard_error: f64,
    true_offset: f64,
    critical_value: f64,
    distribution: &SamplingDistribution,
) -> f64 {
    let standardized_slack = half_width / standard_error - critical_value;
    if standardized_slack <= 0.0 {
        return 0.0;
    }
    let standardized_offset = true_offset / standard_error;
    (distribution.cdf(standardized_slack - standardized_offset)
        - distribution.cdf(-standardized_slack - standardized_offset))
    .clamp(0.0, 1.0)
}

fn maximum_standard_error_for_power(
    half_width: f64,
    true_offset: f64,
    critical_value: f64,
    distribution: &SamplingDistribution,
    replicates: u64,
    dependence: &DependenceModel,
    target_power: f64,
) -> Option<f64> {
    if true_offset.abs() >= half_width {
        return None;
    }
    let power = |standard_error: f64| {
        compose_pass_probability(
            equivalence_unit_power(
                half_width,
                standard_error,
                true_offset,
                critical_value,
                distribution,
            ),
            replicates,
            dependence,
        )
    };
    let mut low = half_width * 1.0e-14;
    let mut high = half_width.max(1.0);
    if power(low) < target_power {
        return None;
    }
    for _ in 0..200 {
        if power(high) < target_power {
            break;
        }
        high *= 2.0;
    }
    if power(high) >= target_power {
        return None;
    }
    for _ in 0..300 {
        let middle = 0.5 * (low + high);
        if power(middle) >= target_power {
            low = middle;
        } else {
            high = middle;
        }
        if (high - low).abs() <= 1.0e-12 * high.abs().max(1.0) {
            break;
        }
    }
    Some(low)
}

fn evaluate_equivalence_gate(
    policy: &CalibrationAdmissionPolicy,
    gate: &GateSpec,
    replicates: u64,
    half_width: f64,
    standard_error: &EvidencedScalar,
    true_offset: f64,
    distribution: &SamplingDistribution,
    confidence_components_per_replicate: u64,
) -> Result<GateFeasibility> {
    if confidence_components_per_replicate == 0 {
        bail!("confidence_components_per_replicate must be positive");
    }
    let joint_unit_confidence = per_unit_interval_confidence(
        policy.simultaneous_interval_confidence,
        replicates,
        &gate.dependence,
    );
    let component_confidence = per_unit_interval_confidence(
        joint_unit_confidence,
        confidence_components_per_replicate,
        &gate.dependence,
    );
    let critical = distribution.critical(component_confidence)?;
    let maximum_se = maximum_standard_error_for_power(
        half_width,
        true_offset,
        critical,
        distribution,
        replicates,
        &gate.dependence,
        gate.required_familywise_power,
    );
    let Some((observed_se, observed_ess)) = standard_error.measured() else {
        return Ok(GateFeasibility {
            gate: gate.id.clone(),
            role: gate.role,
            replicates,
            dependence: gate.dependence.clone(),
            simultaneous_interval_confidence: policy.simultaneous_interval_confidence,
            confidence_components_per_replicate,
            per_unit_interval_confidence: Some(component_confidence),
            critical_value: Some(critical),
            physical_equivalence_half_width: Some(half_width),
            effective_standard_error: None,
            unit_pass_probability: None,
            familywise_pass_probability_lower_bound: None,
            required_familywise_power: gate.required_familywise_power,
            maximum_standard_error_for_required_power: maximum_se,
            required_effective_sample_size: None,
            verdict: GateVerdict::EvidenceRequired,
            actions: vec![
                "measure and hash-bind the estimator standard error using the declared independent-unit and autocorrelation contract"
                    .to_string(),
            ],
        });
    };
    let unit_power =
        equivalence_unit_power(half_width, observed_se, true_offset, critical, distribution);
    let family_power = compose_pass_probability(unit_power, replicates, &gate.dependence);
    let required_ess = match (maximum_se, observed_ess) {
        (Some(maximum_se), Some(observed_ess)) if observed_se > maximum_se => {
            Some(observed_ess * (observed_se / maximum_se).powi(2))
        }
        _ => None,
    };
    let feasible = family_power >= gate.required_familywise_power;
    let mut actions = Vec::new();
    if !feasible {
        if let Some(maximum_se) = maximum_se {
            actions.push(format!(
                "retain the physical equivalence margin and reduce standard error to <= {maximum_se:.12}; do not loosen the margin to force admission"
            ));
        }
        if let Some(required_ess) = required_ess {
            actions.push(format!(
                "increase effective sampling to at least {required_ess:.3} under the same estimator model"
            ));
        }
    }
    Ok(GateFeasibility {
        gate: gate.id.clone(),
        role: gate.role,
        replicates,
        dependence: gate.dependence.clone(),
        simultaneous_interval_confidence: policy.simultaneous_interval_confidence,
        confidence_components_per_replicate,
        per_unit_interval_confidence: Some(component_confidence),
        critical_value: Some(critical),
        physical_equivalence_half_width: Some(half_width),
        effective_standard_error: Some(observed_se),
        unit_pass_probability: Some(unit_power),
        familywise_pass_probability_lower_bound: Some(family_power),
        required_familywise_power: gate.required_familywise_power,
        maximum_standard_error_for_required_power: maximum_se,
        required_effective_sample_size: required_ess,
        verdict: if feasible {
            GateVerdict::Feasible
        } else {
            GateVerdict::Infeasible
        },
        actions,
    })
}

pub fn evaluate_gates(
    policy: &CalibrationAdmissionPolicy,
    manifest: &RunManifest,
) -> Result<(Vec<GateFeasibility>, Option<f64>)> {
    let mut results = Vec::new();
    for gate in &manifest.gates {
        let replicates = replicate_count(gate, &manifest.design)?;
        let result = match &gate.statistic {
            GateStatistic::EquivalenceInterval {
                equivalence_half_width,
                standard_error,
                design_true_offset,
                sampling_distribution,
                ..
            } => evaluate_equivalence_gate(
                policy,
                gate,
                replicates,
                *equivalence_half_width,
                standard_error,
                *design_true_offset,
                sampling_distribution,
                1,
            )?,
            GateStatistic::PressureDensityEquivalence {
                maximum_relative_density_shift,
                isothermal_compressibility_per_pa,
                isothermal_compressibility_standard_uncertainty_per_pa,
                mean_pressure_standard_error_bar,
                linear_response_validity,
                design_true_offset_bar,
                sampling_distribution,
            } => {
                match linear_response_validity {
                    EvidencedBoolean::Measured { value: true, .. } => {}
                    EvidencedBoolean::Measured { value: false, .. } => {
                        results.push(GateFeasibility {
                            gate: gate.id.clone(),
                            role: gate.role,
                            replicates,
                            dependence: gate.dependence.clone(),
                            simultaneous_interval_confidence: policy
                                .simultaneous_interval_confidence,
                            confidence_components_per_replicate: 2,
                            per_unit_interval_confidence: None,
                            critical_value: None,
                            physical_equivalence_half_width: None,
                            effective_standard_error: None,
                            unit_pass_probability: Some(0.0),
                            familywise_pass_probability_lower_bound: Some(0.0),
                            required_familywise_power: gate.required_familywise_power,
                            maximum_standard_error_for_required_power: None,
                            required_effective_sample_size: None,
                            verdict: GateVerdict::Infeasible,
                            actions: vec![
                                "use a direct density-response equivalence model because the measured equation-of-state interval is not locally linear"
                                    .to_string(),
                            ],
                        });
                        continue;
                    }
                    EvidencedBoolean::EvidenceNotFound { .. } => {
                        results.push(GateFeasibility {
                            gate: gate.id.clone(),
                            role: gate.role,
                            replicates,
                            dependence: gate.dependence.clone(),
                            simultaneous_interval_confidence: policy
                                .simultaneous_interval_confidence,
                            confidence_components_per_replicate: 2,
                            per_unit_interval_confidence: None,
                            critical_value: None,
                            physical_equivalence_half_width: None,
                            effective_standard_error: None,
                            unit_pass_probability: None,
                            familywise_pass_probability_lower_bound: None,
                            required_familywise_power: gate.required_familywise_power,
                            maximum_standard_error_for_required_power: None,
                            required_effective_sample_size: None,
                            verdict: GateVerdict::EvidenceRequired,
                            actions: vec![
                                "measure and hash-bind the local equation-of-state linear-response validity interval, or use direct density-response equivalence"
                                    .to_string(),
                            ],
                        });
                        continue;
                    }
                }
                let Some((compressibility, _)) = isothermal_compressibility_per_pa.measured()
                else {
                    results.push(GateFeasibility {
                        gate: gate.id.clone(),
                        role: gate.role,
                        replicates,
                        dependence: gate.dependence.clone(),
                        simultaneous_interval_confidence: policy.simultaneous_interval_confidence,
                        confidence_components_per_replicate: 2,
                        per_unit_interval_confidence: None,
                        critical_value: None,
                        physical_equivalence_half_width: None,
                        effective_standard_error: None,
                        unit_pass_probability: None,
                        familywise_pass_probability_lower_bound: None,
                        required_familywise_power: gate.required_familywise_power,
                        maximum_standard_error_for_required_power: None,
                        required_effective_sample_size: None,
                        verdict: GateVerdict::EvidenceRequired,
                        actions: vec![
                            "measure isothermal compressibility from equilibrated NPT volume fluctuations with autocorrelation-aware uncertainty"
                                .to_string(),
                        ],
                    });
                    continue;
                };
                let Some((compressibility_uncertainty, _)) =
                    isothermal_compressibility_standard_uncertainty_per_pa.measured()
                else {
                    results.push(GateFeasibility {
                        gate: gate.id.clone(),
                        role: gate.role,
                        replicates,
                        dependence: gate.dependence.clone(),
                        simultaneous_interval_confidence: policy.simultaneous_interval_confidence,
                        confidence_components_per_replicate: 2,
                        per_unit_interval_confidence: None,
                        critical_value: None,
                        physical_equivalence_half_width: None,
                        effective_standard_error: None,
                        unit_pass_probability: None,
                        familywise_pass_probability_lower_bound: None,
                        required_familywise_power: gate.required_familywise_power,
                        maximum_standard_error_for_required_power: None,
                        required_effective_sample_size: None,
                        verdict: GateVerdict::EvidenceRequired,
                        actions: vec![
                            "measure and hash-bind uncertainty in the NPT isothermal-compressibility estimator"
                                .to_string(),
                        ],
                    });
                    continue;
                };
                let joint_unit_confidence = per_unit_interval_confidence(
                    policy.simultaneous_interval_confidence,
                    replicates,
                    &gate.dependence,
                );
                let component_confidence = per_unit_interval_confidence(
                    joint_unit_confidence,
                    2,
                    &gate.dependence,
                );
                let response_critical = sampling_distribution.critical(component_confidence)?;
                let conservative_compressibility =
                    compressibility + response_critical * compressibility_uncertainty;
                let pressure_half_width_bar = maximum_relative_density_shift
                    / (conservative_compressibility * 100_000.0);
                evaluate_equivalence_gate(
                    policy,
                    gate,
                    replicates,
                    pressure_half_width_bar,
                    mean_pressure_standard_error_bar,
                    *design_true_offset_bar,
                    sampling_distribution,
                    2,
                )?
            }
            GateStatistic::LegacyMeasuredDenominator { .. } => GateFeasibility {
                gate: gate.id.clone(),
                role: gate.role,
                replicates,
                dependence: gate.dependence.clone(),
                simultaneous_interval_confidence: policy.simultaneous_interval_confidence,
                confidence_components_per_replicate: 0,
                per_unit_interval_confidence: None,
                critical_value: None,
                physical_equivalence_half_width: None,
                effective_standard_error: None,
                unit_pass_probability: None,
                familywise_pass_probability_lower_bound: None,
                required_familywise_power: gate.required_familywise_power,
                maximum_standard_error_for_required_power: None,
                required_effective_sample_size: None,
                verdict: GateVerdict::InversionBlocked,
                actions: vec![
                    "replace measured-denominator non-significance with an equivalence interval inside a physical margin"
                        .to_string(),
                ],
            },
            GateStatistic::Deterministic { outcome } => {
                let (probability, verdict, actions) = match outcome {
                    EvidencedBoolean::Measured { value: true, .. } => {
                        (Some(1.0), GateVerdict::Feasible, Vec::new())
                    }
                    EvidencedBoolean::Measured { value: false, .. } => (
                        Some(0.0),
                        GateVerdict::Infeasible,
                        vec!["resolve the measured deterministic integrity failure".to_string()],
                    ),
                    EvidencedBoolean::EvidenceNotFound { reason } => (
                        None,
                        GateVerdict::EvidenceRequired,
                        vec![format!(
                            "materialize and hash-bind the deterministic integrity result: {reason}"
                        )],
                    ),
                };
                GateFeasibility {
                    gate: gate.id.clone(),
                    role: gate.role,
                    replicates,
                    dependence: gate.dependence.clone(),
                    simultaneous_interval_confidence: policy.simultaneous_interval_confidence,
                    confidence_components_per_replicate: 0,
                    per_unit_interval_confidence: None,
                    critical_value: None,
                    physical_equivalence_half_width: None,
                    effective_standard_error: None,
                    unit_pass_probability: probability,
                    familywise_pass_probability_lower_bound: probability,
                    required_familywise_power: gate.required_familywise_power,
                    maximum_standard_error_for_required_power: None,
                    required_effective_sample_size: None,
                    verdict,
                    actions,
                }
            }
        };
        results.push(result);
    }

    let probabilities: Option<Vec<f64>> = results
        .iter()
        .map(|row| row.familywise_pass_probability_lower_bound)
        .collect();
    let campaign_probability =
        probabilities.map(|probabilities| match &policy.campaign_dependence {
            DependenceModel::ArbitraryDependenceUnionBound => (1.0
                - probabilities
                    .iter()
                    .map(|probability| 1.0 - probability)
                    .sum::<f64>())
            .clamp(0.0, 1.0),
            DependenceModel::Independent { .. } => probabilities.iter().product(),
        });
    Ok((results, campaign_probability))
}

// =========================================================================
// Admission receipt and downstream contract
// =========================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdmissionInputIdentity {
    pub policy_document_sha256: String,
    pub calibration_domain_document_sha256: String,
    pub run_manifest_document_sha256: String,
}

impl AdmissionInputIdentity {
    pub fn validate(&self) -> Result<()> {
        validate_sha256("policy_document_sha256", &self.policy_document_sha256)?;
        validate_sha256(
            "calibration_domain_document_sha256",
            &self.calibration_domain_document_sha256,
        )?;
        validate_sha256(
            "run_manifest_document_sha256",
            &self.run_manifest_document_sha256,
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EffectiveCalibrationContract {
    pub schema_kind: String,
    pub schema_version: u32,
    pub calibration_id: String,
    pub campaign_id: String,
    pub policy_id: String,
    pub input_identity: AdmissionInputIdentity,
    pub quantities: BTreeMap<String, EffectiveQuantity>,
    pub downstream_consumption_required: bool,
    pub uncertainty_composition: String,
    pub contract_sha256: String,
}

impl EffectiveCalibrationContract {
    pub fn validate_integrity(&self) -> Result<()> {
        if self.schema_kind != EFFECTIVE_CONTRACT_SCHEMA
            || self.schema_version != SCHEMA_VERSION
            || self.calibration_id.trim().is_empty()
            || self.campaign_id.trim().is_empty()
            || self.policy_id.trim().is_empty()
            || !self.downstream_consumption_required
            || self.quantities.is_empty()
        {
            bail!("invalid effective calibration contract identity or membership");
        }
        self.input_identity.validate()?;
        validate_sha256("contract_sha256", &self.contract_sha256)?;
        for (name, quantity) in &self.quantities {
            if name.trim().is_empty()
                || !quantity.estimate.is_finite()
                || quantity.units.trim().is_empty()
                || !quantity.baseline_standard_uncertainty.is_finite()
                || quantity.baseline_standard_uncertainty <= 0.0
                || !quantity.effective_standard_uncertainty.is_finite()
                || quantity.effective_standard_uncertainty < quantity.baseline_standard_uncertainty
                || !quantity.uncertainty_inflation_factor.is_finite()
                || quantity.uncertainty_inflation_factor < 1.0
                || !quantity.effective_sample_size.is_finite()
                || quantity.effective_sample_size <= 0.0
            {
                bail!("invalid effective calibration quantity {name}");
            }
            let expected_inflation =
                quantity.effective_standard_uncertainty / quantity.baseline_standard_uncertainty;
            if (expected_inflation - quantity.uncertainty_inflation_factor).abs()
                > 1.0e-10 * expected_inflation.abs().max(1.0)
            {
                bail!("effective calibration uncertainty inflation is inconsistent for {name}");
            }
        }
        let mut projection = self.clone();
        projection.contract_sha256.clear();
        let mut digest = Sha256::new();
        digest.update(b"PRISM_EFFECTIVE_CALIBRATION_CONTRACT_V2\0");
        digest.update(serde_json::to_vec(&projection)?);
        let expected = format!("{:x}", digest.finalize());
        if self.contract_sha256 != expected {
            bail!("effective calibration contract digest mismatch");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdmissionReceipt {
    pub schema_kind: String,
    pub schema_version: u32,
    pub policy_id: String,
    pub calibration_id: String,
    pub campaign_id: String,
    pub input_identity: AdmissionInputIdentity,
    pub verified_evidence: Vec<VerifiedEvidence>,
    pub transferability: TransferabilityVerdict,
    pub gates: Vec<GateFeasibility>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub campaign_pass_probability_lower_bound: Option<f64>,
    pub minimum_campaign_power: f64,
    pub admitted: bool,
    pub decision: String,
    pub required_actions: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effective_calibration_contract: Option<EffectiveCalibrationContract>,
    pub claim_scope: String,
    pub receipt_sha256: String,
}

impl AdmissionReceipt {
    pub fn validate_integrity(&self) -> Result<()> {
        if self.schema_kind != ADMISSION_RECEIPT_SCHEMA
            || self.schema_version != SCHEMA_VERSION
            || self.policy_id.trim().is_empty()
            || self.calibration_id.trim().is_empty()
            || self.campaign_id.trim().is_empty()
        {
            bail!("invalid calibration admission receipt identity");
        }
        self.input_identity.validate()?;
        validate_sha256("receipt_sha256", &self.receipt_sha256)?;
        if self.admitted {
            if self.decision != "ADMITTED" {
                bail!("admitted calibration receipt has a non-ADMITTED decision");
            }
            let contract = self
                .effective_calibration_contract
                .as_ref()
                .context("admitted calibration receipt lacks effective contract")?;
            contract.validate_integrity()?;
            if contract.policy_id != self.policy_id
                || contract.calibration_id != self.calibration_id
                || contract.campaign_id != self.campaign_id
                || contract.input_identity.policy_document_sha256
                    != self.input_identity.policy_document_sha256
                || contract.input_identity.calibration_domain_document_sha256
                    != self.input_identity.calibration_domain_document_sha256
                || contract.input_identity.run_manifest_document_sha256
                    != self.input_identity.run_manifest_document_sha256
            {
                bail!("effective contract identity differs from its admission receipt");
            }
        } else if self.decision == "ADMITTED" {
            bail!("refused calibration receipt claims ADMITTED");
        } else if let Some(contract) = &self.effective_calibration_contract {
            contract.validate_integrity()?;
        }
        let mut projection = self.clone();
        projection.receipt_sha256.clear();
        let mut digest = Sha256::new();
        digest.update(b"PRISM_CALIBRATION_ADMISSION_RECEIPT_V2\0");
        digest.update(serde_json::to_vec(&projection)?);
        let expected = format!("{:x}", digest.finalize());
        if self.receipt_sha256 != expected {
            bail!("calibration admission receipt digest mismatch");
        }
        Ok(())
    }
}

fn effective_contract(
    policy: &CalibrationAdmissionPolicy,
    domain: &CalibrationDomain,
    manifest: &RunManifest,
    input_identity: &AdmissionInputIdentity,
    transferability: &TransferabilityVerdict,
) -> Result<Option<EffectiveCalibrationContract>> {
    let Some(quantities) = transferability.effective_quantities() else {
        return Ok(None);
    };
    let mut contract = EffectiveCalibrationContract {
        schema_kind: EFFECTIVE_CONTRACT_SCHEMA.to_string(),
        schema_version: SCHEMA_VERSION,
        calibration_id: domain.calibration_id.clone(),
        campaign_id: manifest.campaign_id.clone(),
        policy_id: policy.policy_id.clone(),
        input_identity: input_identity.clone(),
        quantities: quantities.clone(),
        downstream_consumption_required: true,
        uncertainty_composition: "quadrature of baseline standard uncertainty and independently evidenced transfer-sensitivity contributions"
            .to_string(),
        contract_sha256: String::new(),
    };
    let mut digest = Sha256::new();
    digest.update(b"PRISM_EFFECTIVE_CALIBRATION_CONTRACT_V2\0");
    digest.update(serde_json::to_vec(&contract)?);
    contract.contract_sha256 = format!("{:x}", digest.finalize());
    Ok(Some(contract))
}

pub fn admit(
    policy: &CalibrationAdmissionPolicy,
    domain: &CalibrationDomain,
    manifest: &RunManifest,
    input_identity: AdmissionInputIdentity,
    verified_evidence: Vec<VerifiedEvidence>,
) -> Result<AdmissionReceipt> {
    input_identity.validate()?;
    validate_evidence_roles(policy, domain, manifest, &verified_evidence)?;
    validate_gate_source_bindings(manifest, &verified_evidence)?;
    let evidence_ids = verified_ids(&verified_evidence);
    policy.validate(&evidence_ids)?;
    domain.validate(policy, &evidence_ids)?;
    manifest.validate(policy, &evidence_ids)?;
    let transferability = evaluate_transferability(policy, domain, manifest, &evidence_ids)?;
    let (gates, campaign_probability) = evaluate_gates(policy, manifest)?;

    let all_gates_feasible = gates
        .iter()
        .all(|gate| gate.verdict == GateVerdict::Feasible);
    let campaign_power_ok = campaign_probability
        .map(|probability| probability >= policy.minimum_campaign_power)
        .unwrap_or(false);
    let admitted = transferability.permits_reuse() && all_gates_feasible && campaign_power_ok;

    let decision = if admitted {
        "ADMITTED"
    } else {
        match &transferability {
            TransferabilityVerdict::EvidenceRequired { .. } => "CALIBRATION_EVIDENCE_REQUIRED",
            TransferabilityVerdict::RecalibrationRequired { .. } => "RECALIBRATION_REQUIRED",
            _ if gates
                .iter()
                .any(|gate| gate.verdict == GateVerdict::InversionBlocked) =>
            {
                "GATE_CONTRACT_INVERSION_BLOCKED"
            }
            _ if gates
                .iter()
                .any(|gate| gate.verdict == GateVerdict::EvidenceRequired) =>
            {
                "GATE_EVIDENCE_REQUIRED"
            }
            _ => "SAMPLING_DESIGN_INFEASIBLE",
        }
    };

    let mut required_actions = Vec::new();
    match &transferability {
        TransferabilityVerdict::EvidenceRequired { findings }
        | TransferabilityVerdict::RecalibrationRequired { findings } => {
            required_actions.extend(findings.iter().map(|finding| finding.detail.clone()));
        }
        _ => {}
    }
    for gate in &gates {
        required_actions.extend(
            gate.actions
                .iter()
                .map(|action| format!("gate {:?}: {action}", gate.gate)),
        );
    }
    if let Some(probability) = campaign_probability {
        if probability < policy.minimum_campaign_power {
            required_actions.push(format!(
                "campaign pass-probability lower bound {probability:.12} is below required power {:.12}; increase precision/sampling without changing physical equivalence margins",
                policy.minimum_campaign_power
            ));
        }
    }

    let effective_calibration_contract =
        effective_contract(policy, domain, manifest, &input_identity, &transferability)?;
    let mut receipt = AdmissionReceipt {
        schema_kind: ADMISSION_RECEIPT_SCHEMA.to_string(),
        schema_version: SCHEMA_VERSION,
        policy_id: policy.policy_id.clone(),
        calibration_id: domain.calibration_id.clone(),
        campaign_id: manifest.campaign_id.clone(),
        input_identity,
        verified_evidence,
        transferability,
        gates,
        campaign_pass_probability_lower_bound: campaign_probability,
        minimum_campaign_power: policy.minimum_campaign_power,
        admitted,
        decision: decision.to_string(),
        required_actions,
        effective_calibration_contract,
        claim_scope: "Computational calibration provenance, applicability, simultaneous-equivalence, and sampling-design admission. Physical margins are manifest authority; statistical critical values are derived. This receipt is not a physical calibration result, target binding result, wet-lab result, safety claim, or therapeutic claim."
            .to_string(),
        receipt_sha256: String::new(),
    };
    let mut digest = Sha256::new();
    digest.update(b"PRISM_CALIBRATION_ADMISSION_RECEIPT_V2\0");
    digest.update(serde_json::to_vec(&receipt)?);
    receipt.receipt_sha256 = format!("{:x}", digest.finalize());
    Ok(receipt)
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(label: &str) -> String {
        format!("{:x}", Sha256::digest(label.as_bytes()))
    }

    fn binding(role: EvidenceRole, label: &str) -> EvidenceBinding {
        EvidenceBinding {
            path: format!("/evidence/{label}"),
            sha256: digest(label),
            size_bytes: 100,
            role,
            format: EvidenceFormat::Raw,
            expected_schema_kind: None,
        }
    }

    fn verified(id: &str, role: EvidenceRole) -> VerifiedEvidence {
        VerifiedEvidence {
            evidence_id: id.to_string(),
            path: format!("/evidence/{id}"),
            sha256: digest(id),
            size_bytes: 100,
            role,
            schema_kind: None,
            gate_definition_sha256s: BTreeMap::new(),
        }
    }

    fn policy() -> CalibrationAdmissionPolicy {
        CalibrationAdmissionPolicy {
            schema_kind: POLICY_SCHEMA.to_string(),
            schema_version: SCHEMA_VERSION,
            policy_id: "TEST_POLICY_V2".to_string(),
            required_axes: BTreeMap::from([
                ("force_field".to_string(), AxisKind::ExactIdentity),
                ("temperature_k".to_string(), AxisKind::Continuous),
            ]),
            required_gate_ids: BTreeSet::from(["occupancy_equivalence".to_string()]),
            simultaneous_interval_confidence: 0.95,
            minimum_campaign_power: 0.90,
            maximum_uncertainty_inflation: 2.0,
            campaign_dependence: DependenceModel::ArbitraryDependenceUnionBound,
            evidence: BTreeMap::new(),
        }
    }

    fn domain(measured: bool) -> CalibrationDomain {
        let calibration = if measured {
            CalibrationEvidenceState::Measured {
                measurement_evidence_id: "calibration".to_string(),
                engine_sha256: digest("engine"),
                source_epoch_sha256: digest("source"),
                quantities: BTreeMap::from([(
                    "chemical_potential".to_string(),
                    CalibratedQuantity {
                        estimate: -1.5,
                        standard_uncertainty: 0.02,
                        units: "kcal/mol".to_string(),
                        estimator: "BAR".to_string(),
                        effective_sample_size: 500.0,
                        uncertainty_evidence_id: "calibration_uncertainty".to_string(),
                    },
                )]),
            }
        } else {
            CalibrationEvidenceState::EvidenceNotFound {
                reason: "bulk chemical-potential campaign has not executed".to_string(),
            }
        };
        CalibrationDomain {
            schema_kind: CALIBRATION_DOMAIN_SCHEMA.to_string(),
            schema_version: SCHEMA_VERSION,
            calibration_id: "TEST_CALIBRATION".to_string(),
            calibration,
            axes: BTreeMap::from([
                (
                    "force_field".to_string(),
                    AxisConstraint::ExactIdentity {
                        measured: "FF_V1".to_string(),
                    },
                ),
                (
                    "temperature_k".to_string(),
                    AxisConstraint::Continuous {
                        measured: 300.0,
                        tolerance: 1.0,
                        units: "K".to_string(),
                        extrapolation: ContinuousExtrapolation::RecalibrationOnly,
                    },
                ),
            ]),
            evidence: if measured {
                BTreeMap::from([
                    (
                        "calibration".to_string(),
                        binding(EvidenceRole::PhysicalCalibration, "calibration"),
                    ),
                    (
                        "calibration_uncertainty".to_string(),
                        binding(
                            EvidenceRole::EstimatorUncertainty,
                            "calibration_uncertainty",
                        ),
                    ),
                ])
            } else {
                BTreeMap::new()
            },
        }
    }

    fn manifest(standard_error: EvidencedScalar) -> RunManifest {
        let mut gate = GateSpec {
            id: "occupancy_equivalence".to_string(),
            role: GateRole::Accuracy,
            source_evidence_id: "gate_source".to_string(),
            statistical_method_evidence_id: "method".to_string(),
            definition_sha256: String::new(),
            scientific_predicate: serde_json::json!({
                "schema_kind": "prismos_scientific_gate_predicate",
                "schema_version": SCHEMA_VERSION,
                "gate_id": "occupancy_equivalence",
                "predicate": "simultaneous_interval_inside_relative_occupancy_margin",
                "threshold_source": "test_gate_source"
            }),
            statistic: GateStatistic::EquivalenceInterval {
                equivalence_half_width: 0.15,
                units: "relative".to_string(),
                standard_error,
                design_true_offset: 0.0,
                sampling_distribution: SamplingDistribution::NormalKnown,
            },
            replication_across: vec!["conditions".to_string()],
            dependence: DependenceModel::ArbitraryDependenceUnionBound,
            required_familywise_power: 0.90,
        };
        gate.definition_sha256 = gate_definition_sha256(&gate).unwrap();
        RunManifest {
            schema_kind: RUN_MANIFEST_SCHEMA.to_string(),
            schema_version: SCHEMA_VERSION,
            campaign_id: "TEST_CAMPAIGN".to_string(),
            axes: BTreeMap::from([
                (
                    "force_field".to_string(),
                    AxisValue::Text("FF_V1".to_string()),
                ),
                ("temperature_k".to_string(), AxisValue::Number(300.0)),
            ]),
            design: BTreeMap::from([("conditions".to_string(), 4)]),
            gates: vec![gate],
            evidence: BTreeMap::from([
                (
                    "gate_source".to_string(),
                    binding(EvidenceRole::GateSource, "gate_source"),
                ),
                (
                    "method".to_string(),
                    binding(EvidenceRole::StatisticalMethod, "method"),
                ),
                (
                    "standard_error".to_string(),
                    binding(EvidenceRole::EstimatorUncertainty, "standard_error"),
                ),
            ]),
        }
    }

    fn evidence(measured_calibration: bool) -> Vec<VerifiedEvidence> {
        let definition_sha256 = manifest(EvidencedScalar::EvidenceNotFound {
            units: "relative".to_string(),
            reason: "test definition".to_string(),
        })
        .gates[0]
            .definition_sha256
            .clone();
        let mut gate_source = verified("gate_source", EvidenceRole::GateSource);
        gate_source
            .gate_definition_sha256s
            .insert("occupancy_equivalence".to_string(), definition_sha256);
        let mut rows = vec![
            gate_source,
            verified("method", EvidenceRole::StatisticalMethod),
            verified("standard_error", EvidenceRole::EstimatorUncertainty),
        ];
        if measured_calibration {
            rows.push(verified("calibration", EvidenceRole::PhysicalCalibration));
            rows.push(verified(
                "calibration_uncertainty",
                EvidenceRole::EstimatorUncertainty,
            ));
        }
        rows
    }

    fn input_identity() -> AdmissionInputIdentity {
        AdmissionInputIdentity {
            policy_document_sha256: digest("policy-document"),
            calibration_domain_document_sha256: digest("domain-document"),
            run_manifest_document_sha256: digest("manifest-document"),
        }
    }

    #[test]
    fn normal_and_student_t_numerics_match_reference_values() {
        assert!((normal_cdf(1.96) - 0.975_002_104_851_8).abs() < 1.0e-10);
        assert!((student_t_two_sided(2.776_445_105_197_8, 4.0) - 0.95).abs() < 1.0e-9);
        let critical = SamplingDistribution::StudentT {
            degrees_of_freedom: 4.0,
        }
        .critical(0.95)
        .unwrap();
        assert!((critical - 2.776_445_105_197_8).abs() < 1.0e-8);
    }

    #[test]
    fn placeholder_digests_are_rejected() {
        assert!(validate_sha256("zero", &"0".repeat(64)).is_err());
        assert!(validate_sha256("one", &"1".repeat(64)).is_err());
        assert!(validate_sha256("real", &digest("real")).is_ok());
    }

    #[test]
    fn evidence_role_substitution_is_rejected() {
        let standard_error = EvidencedScalar::Measured {
            value: 0.01,
            units: "relative".to_string(),
            evidence_id: "standard_error".to_string(),
            effective_sample_size: Some(200.0),
        };
        let mut rows = evidence(true);
        rows.iter_mut()
            .find(|row| row.evidence_id == "standard_error")
            .unwrap()
            .role = EvidenceRole::GateSource;
        let result = admit(
            &policy(),
            &domain(true),
            &manifest(standard_error),
            input_identity(),
            rows,
        );
        assert!(result.is_err());
    }

    #[test]
    fn missing_physical_calibration_is_a_receipted_evidence_gate() {
        let standard_error = EvidencedScalar::Measured {
            value: 0.01,
            units: "relative".to_string(),
            evidence_id: "standard_error".to_string(),
            effective_sample_size: Some(200.0),
        };
        let receipt = admit(
            &policy(),
            &domain(false),
            &manifest(standard_error),
            input_identity(),
            evidence(false),
        )
        .unwrap();
        assert!(!receipt.admitted);
        assert_eq!(receipt.decision, "CALIBRATION_EVIDENCE_REQUIRED");
        assert!(matches!(
            receipt.transferability,
            TransferabilityVerdict::EvidenceRequired { .. }
        ));
    }

    #[test]
    fn extra_or_missing_axes_fail_closed() {
        let mut run = manifest(EvidencedScalar::EvidenceNotFound {
            units: "relative".to_string(),
            reason: "not measured".to_string(),
        });
        run.axes
            .insert("solvent".to_string(), AxisValue::Text("water".to_string()));
        let ids = verified_ids(&evidence(true));
        assert!(run.validate(&policy(), &ids).is_err());
    }

    #[test]
    fn equivalence_power_decreases_as_noise_increases() {
        let distribution = SamplingDistribution::NormalKnown;
        let low_noise = equivalence_unit_power(0.15, 0.01, 0.0, 2.5, &distribution);
        let high_noise = equivalence_unit_power(0.15, 0.05, 0.0, 2.5, &distribution);
        assert!(low_noise > high_noise);
    }

    #[test]
    fn arbitrary_dependence_uses_union_bound_not_product() {
        let robust =
            compose_pass_probability(0.99, 10, &DependenceModel::ArbitraryDependenceUnionBound);
        let independent = compose_pass_probability(
            0.99,
            10,
            &DependenceModel::Independent {
                evidence_id: "independence".to_string(),
            },
        );
        assert!((robust - 0.90).abs() < 1.0e-12);
        assert!((independent - 0.99_f64.powi(10)).abs() < 1.0e-12);
    }

    #[test]
    fn independence_requires_verified_evidence() {
        let mut policy = policy();
        policy.campaign_dependence = DependenceModel::Independent {
            evidence_id: "independence".to_string(),
        };
        assert!(policy.validate(&BTreeSet::new()).is_err());
    }

    #[test]
    fn gate_definition_must_match_published_source_digest() {
        let run = manifest(EvidencedScalar::EvidenceNotFound {
            units: "relative".to_string(),
            reason: "not measured".to_string(),
        });
        let mut rows = evidence(false);
        let source = rows
            .iter_mut()
            .find(|row| row.evidence_id == "gate_source")
            .unwrap();
        source.gate_definition_sha256s.insert(
            "occupancy_equivalence".to_string(),
            digest("substituted-gate-definition"),
        );
        assert!(validate_gate_source_bindings(&run, &rows).is_err());
    }

    #[test]
    fn pressure_joint_confidence_allocates_two_components() {
        let replicate_confidence =
            per_unit_interval_confidence(0.95, 5, &DependenceModel::ArbitraryDependenceUnionBound);
        let component_confidence = per_unit_interval_confidence(
            replicate_confidence,
            2,
            &DependenceModel::ArbitraryDependenceUnionBound,
        );
        assert!(component_confidence > replicate_confidence);
        assert!((component_confidence - 0.995).abs() < 1.0e-12);
    }

    #[test]
    fn legacy_measured_denominator_gate_cannot_admit() {
        let mut policy = policy();
        policy.required_gate_ids = BTreeSet::from(["legacy".to_string()]);
        let mut run = manifest(EvidencedScalar::EvidenceNotFound {
            units: "relative".to_string(),
            reason: "not used".to_string(),
        });
        let mut gate = GateSpec {
            id: "legacy".to_string(),
            role: GateRole::Accuracy,
            source_evidence_id: "gate_source".to_string(),
            statistical_method_evidence_id: "method".to_string(),
            definition_sha256: String::new(),
            scientific_predicate: serde_json::json!({
                "schema_kind": "prismos_scientific_gate_predicate",
                "schema_version": SCHEMA_VERSION,
                "gate_id": "legacy",
                "predicate": "legacy_measured_denominator",
                "threshold_source": "test_gate_source"
            }),
            statistic: GateStatistic::LegacyMeasuredDenominator {
                threshold: 100.0,
                independent_units: 5,
            },
            replication_across: vec![],
            dependence: DependenceModel::ArbitraryDependenceUnionBound,
            required_familywise_power: 0.90,
        };
        gate.definition_sha256 = gate_definition_sha256(&gate).unwrap();
        run.gates = vec![gate];
        let ids = verified_ids(&evidence(true));
        run.validate(&policy, &ids).unwrap();
        let (rows, _) = evaluate_gates(&policy, &run).unwrap();
        assert_eq!(rows[0].verdict, GateVerdict::InversionBlocked);
    }

    #[test]
    fn sampling_solver_preserves_physical_margin() {
        let distribution = SamplingDistribution::NormalKnown;
        let critical = distribution.critical(0.99).unwrap();
        let maximum = maximum_standard_error_for_power(
            0.2,
            0.0,
            critical,
            &distribution,
            5,
            &DependenceModel::ArbitraryDependenceUnionBound,
            0.90,
        )
        .unwrap();
        assert!(maximum > 0.0 && maximum < 0.2 / critical);
    }

    #[test]
    fn pressure_margin_is_derived_from_density_impact() {
        let maximum_relative_density_shift: f64 = 0.00025;
        let compressibility_per_pa: f64 = 4.5e-10;
        let pressure_half_width =
            maximum_relative_density_shift / (compressibility_per_pa * 100_000.0);
        assert!((pressure_half_width - 5.555_555_555_555_556).abs() < 1.0e-12);
    }

    #[test]
    fn transfer_sensitivity_is_added_in_quadrature() {
        let policy = policy();
        let mut domain = domain(true);
        domain.axes.insert(
            "temperature_k".to_string(),
            AxisConstraint::Continuous {
                measured: 300.0,
                tolerance: 1.0,
                units: "K".to_string(),
                extrapolation: ContinuousExtrapolation::EvidenceBoundLinear {
                    limit: 10.0,
                    additional_standard_uncertainty_per_unit: BTreeMap::from([(
                        "chemical_potential".to_string(),
                        0.01,
                    )]),
                    sensitivity_evidence_id: "sensitivity".to_string(),
                },
            },
        );
        domain.evidence.insert(
            "sensitivity".to_string(),
            binding(EvidenceRole::ApplicabilitySensitivity, "sensitivity"),
        );
        let mut run = manifest(EvidencedScalar::EvidenceNotFound {
            units: "relative".to_string(),
            reason: "not measured".to_string(),
        });
        run.axes
            .insert("temperature_k".to_string(), AxisValue::Number(303.0));
        let mut evidence = evidence(true);
        evidence.push(verified(
            "sensitivity",
            EvidenceRole::ApplicabilitySensitivity,
        ));
        let ids = verified_ids(&evidence);
        policy.validate(&ids).unwrap();
        let transfer = evaluate_transferability(&policy, &domain, &run, &ids).unwrap();
        let TransferabilityVerdict::BoundedExtrapolation {
            effective_quantities,
            ..
        } = transfer
        else {
            panic!("expected bounded extrapolation");
        };
        let quantity = &effective_quantities["chemical_potential"];
        // 2 K beyond tolerance * 0.01, combined with baseline 0.02.
        assert!((quantity.effective_standard_uncertainty - (0.0008_f64).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn effective_contract_carries_propagated_uncertainty() {
        let standard_error = EvidencedScalar::Measured {
            value: 0.005,
            units: "relative".to_string(),
            evidence_id: "standard_error".to_string(),
            effective_sample_size: Some(1000.0),
        };
        let receipt = admit(
            &policy(),
            &domain(true),
            &manifest(standard_error),
            input_identity(),
            evidence(true),
        )
        .unwrap();
        let contract = receipt.effective_calibration_contract.unwrap();
        assert!(contract.downstream_consumption_required);
        assert_eq!(contract.contract_sha256.len(), 64);
        assert_eq!(
            contract.quantities["chemical_potential"].effective_standard_uncertainty,
            0.02
        );
    }
}
