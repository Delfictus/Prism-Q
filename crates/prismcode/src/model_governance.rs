use crate::binary_runtime::PcError;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const GPT_OSS_20B: &str = "openai/gpt-oss-20b";
const QWEN3_CODER_NEXT: &str = "Qwen/Qwen3-Coder-Next";
const MODEL_GENERATED_UNVERIFIED: &str = "MODEL_GENERATED_UNVERIFIED";
const DEFAULT_MAX_LIVE_HASH_BYTES: u64 = 512 * 1024 * 1024;
const FIXED_RUNTIME_SMOKE_PROMPT: &str = "PRISMCODE_MODEL_RUNTIME_SMOKE_V0_1: identify yourself as MODEL_GENERATED_UNVERIFIED advisory output, do not request canonical PRISMCODE mutation, and return a bounded health response.";

#[derive(Debug, Deserialize)]
pub struct ModelRegistry {
    pub registry_version: String,
    pub rules: RegistryRules,
    #[serde(default)]
    pub installed_models: Vec<InstalledModel>,
    #[serde(default)]
    pub hardware_gated_models: Vec<HardwareGatedModel>,
}

#[derive(Debug, Deserialize)]
pub struct RegistryRules {
    #[serde(rename = "canonical_state_mutation_allowed")]
    pub canonical_state_mutation_allowed: bool,
}

#[derive(Debug, Deserialize)]
pub struct InstalledModel {
    pub model_id: String,
    #[serde(default)]
    pub local_path: Option<PathBuf>,
    #[serde(default)]
    pub install_allowed: bool,
    #[serde(default)]
    pub hardware_gated: bool,
    #[serde(default)]
    pub weights_hash_verified: bool,
    #[serde(default)]
    pub serving_status: Option<String>,
    #[serde(default)]
    pub runtime_binding_status: Option<String>,
    #[serde(default)]
    pub runtime_binding_allowed: bool,
    #[serde(default)]
    pub runtime_bound: bool,
    #[serde(default)]
    pub runtime_name: Option<String>,
    #[serde(default)]
    pub runtime_version: Option<String>,
    #[serde(default)]
    pub runtime_hash: Option<String>,
    #[serde(default)]
    pub launch_command: Option<String>,
    #[serde(default)]
    pub endpoint: Option<String>,
    #[serde(default)]
    pub network_policy: Option<String>,
    #[serde(default)]
    pub gpu_cpu_memory_profile: Option<Value>,
    #[serde(default)]
    pub context_limit: Option<u64>,
    #[serde(default)]
    pub quantization_format: Option<String>,
    #[serde(default)]
    pub deterministic_settings: Option<Value>,
    #[serde(default)]
    pub runtime_smoke_required_before_binding: bool,
    #[serde(default)]
    pub runtime_smoke_status: Option<String>,
    #[serde(default)]
    pub runtime_smoke_ref: Option<String>,
    #[serde(default)]
    pub runtime_smoke_prompt_hash: Option<String>,
    #[serde(default)]
    pub runtime_smoke_output_hash: Option<String>,
    #[serde(default)]
    pub runtime_smoke_output_classification: Option<String>,
    #[serde(default)]
    pub runtime_smoke_exit_status: Option<i32>,
    #[serde(default)]
    pub canonical_mutation_denial_status: Option<String>,
    #[serde(default)]
    pub airlock_status: Option<String>,
    #[serde(default)]
    pub canonical_state_mutation_allowed: bool,
    #[serde(default)]
    pub files: Vec<ModelFile>,
}

#[derive(Debug, Deserialize)]
pub struct ModelFile {
    pub path: PathBuf,
    pub bytes: u64,
    pub blake3: String,
}

#[derive(Debug, Deserialize)]
pub struct HardwareGatedModel {
    pub model_id: String,
    #[serde(default)]
    pub hardware_gated: bool,
    #[serde(default)]
    pub install_allowed: bool,
    #[serde(default)]
    pub install_gate: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub canonical_state_mutation_allowed: bool,
}

#[derive(Debug, Serialize)]
pub struct ModelInspectReport {
    pub status: String,
    pub registry_path: PathBuf,
    pub registry_version: String,
    pub canonical_state_mutation_allowed: bool,
    pub gpt_oss_20b: Option<InstalledModelReport>,
    pub qwen3_coder_next: Option<HardwareGateReport>,
    pub issues: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct InstalledModelReport {
    pub model_id: String,
    pub local_path: Option<PathBuf>,
    pub local_path_exists: bool,
    pub install_allowed: bool,
    pub hardware_gated: bool,
    pub weights_hash_verified_registry: bool,
    pub installed_hash_verified: bool,
    pub file_checks: Vec<ModelFileCheck>,
    pub serving_status: Option<String>,
    pub runtime_binding_status: Option<String>,
    pub runtime_binding_allowed: bool,
    pub runtime_bound: bool,
    pub runtime_name: Option<String>,
    pub runtime_version: Option<String>,
    pub runtime_hash: Option<String>,
    pub launch_command: Option<String>,
    pub endpoint: Option<String>,
    pub network_policy: Option<String>,
    pub gpu_cpu_memory_profile: Option<Value>,
    pub context_limit: Option<u64>,
    pub quantization_format: Option<String>,
    pub deterministic_settings: Option<Value>,
    pub runtime_smoke_required_before_binding: bool,
    pub runtime_smoke_status: Option<String>,
    pub runtime_smoke_ref: Option<String>,
    pub runtime_smoke_prompt_hash: Option<String>,
    pub runtime_smoke_output_hash: Option<String>,
    pub runtime_smoke_output_classification: Option<String>,
    pub runtime_smoke_exit_status: Option<i32>,
    pub canonical_mutation_denial_status: Option<String>,
    pub model_runtime_active: bool,
    pub model_serving_gate_status: String,
    pub required_runtime_evidence_missing: Vec<String>,
    pub airlock_status: String,
    pub canonical_state_mutation_allowed: bool,
}

#[derive(Debug, Serialize)]
pub struct ModelFileCheck {
    pub path: PathBuf,
    pub exists: bool,
    pub bytes_expected: u64,
    pub bytes_observed: Option<u64>,
    pub bytes_observed_source: String,
    pub blake3_expected: String,
    pub blake3_observed: Option<String>,
    pub verified: bool,
    pub verification_mode: String,
    pub verification_status: String,
}

#[derive(Debug, Serialize)]
pub struct HardwareGateReport {
    pub model_id: String,
    pub hardware_gated: bool,
    pub install_allowed: bool,
    pub install_gate: Option<String>,
    pub status: Option<String>,
    pub canonical_state_mutation_allowed: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ModelInspectOptions {
    pub force_live_hash: bool,
}

#[derive(Debug, Clone, Default)]
pub struct ModelRuntimeSmokeOptions {
    pub inspect: ModelInspectOptions,
    pub report_path: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
pub struct ModelRuntimeSmokeReport {
    pub status: String,
    pub state: String,
    pub registry_path: PathBuf,
    pub registry_version: String,
    pub model_id: String,
    pub model_inspect_status: String,
    pub installed_hash_verified: bool,
    pub serving_status: Option<String>,
    pub runtime_binding_status: Option<String>,
    pub runtime_smoke_status: Option<String>,
    pub model_runtime_active: bool,
    pub compatible_runtime_detected: bool,
    pub selected_runtime: Option<RuntimeCandidateReport>,
    pub runtime_candidates: Vec<RuntimeCandidateReport>,
    pub fixed_prompt_hash_blake3: String,
    pub output_hash_blake3: Option<String>,
    pub output_classification: String,
    pub exit_status: Option<i32>,
    pub smoke_log_or_report_ref: Option<PathBuf>,
    pub canonical_mutation_denial_check: CanonicalMutationDenialReport,
    pub release_mode_gates: ModelRuntimeReleaseGateReport,
    pub state_transitions: Vec<ModelRuntimeStateTransition>,
    pub required_runtime_evidence_missing: Vec<String>,
    pub issues: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeCandidateReport {
    pub runtime_name: String,
    pub source: String,
    pub command: Option<String>,
    pub detected: bool,
    pub compatible: bool,
    pub version: Option<String>,
    pub binary_hash_blake3: Option<String>,
    pub endpoint: Option<String>,
    pub evidence: String,
}

#[derive(Debug, Serialize)]
pub struct CanonicalMutationDenialReport {
    pub status: String,
    pub denied: bool,
    pub probe_model_registry_ref: String,
    pub probe_output_classification: String,
    pub probe_canonical_state_mutation_allowed: bool,
    pub reason: String,
    pub issues: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct ModelRuntimeReleaseGateReport {
    pub core_isa_model_gate: String,
    pub ai_assisted_os_model_gate: String,
    pub ai_assisted_os_blockers: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct ModelRuntimeStateTransition {
    pub from_state: String,
    pub to_state: String,
    pub status: String,
    pub evidence: String,
}

#[derive(Debug, Serialize)]
pub struct ModelAirlockReport {
    pub status: String,
    pub registry_path: PathBuf,
    pub registry_version: String,
    pub model_inspect_status: String,
    pub output_classification_required: String,
    pub artifact_path: Option<PathBuf>,
    pub artifact_checked: bool,
    pub model_originated: bool,
    pub output_classification: String,
    pub canonical_state_mutation_allowed: bool,
    pub accepted_action_flow: bool,
    pub hil_accepted: bool,
    pub sealed: bool,
    pub canonical_influence_allowed: bool,
    pub issues: Vec<String>,
}

pub fn inspect_model_registry(registry_path: &Path) -> Result<ModelInspectReport, PcError> {
    inspect_model_registry_with_options(registry_path, ModelInspectOptions::default())
}

pub fn inspect_model_registry_with_options(
    registry_path: &Path,
    options: ModelInspectOptions,
) -> Result<ModelInspectReport, PcError> {
    let registry = read_registry(registry_path)?;
    let mut issues = Vec::new();

    if registry.rules.canonical_state_mutation_allowed {
        issues.push("registry rules allow direct canonical model mutation".to_string());
    }

    let gpt = registry
        .installed_models
        .iter()
        .find(|model| model.model_id == GPT_OSS_20B)
        .map(|model| inspect_installed_model(model, options))
        .transpose()?;

    if gpt.is_none() {
        issues.push(format!("{GPT_OSS_20B} registry record missing"));
    }

    if let Some(report) = &gpt {
        if report.canonical_state_mutation_allowed {
            issues.push(format!(
                "{} permits direct canonical state mutation",
                report.model_id
            ));
        }
        if report.local_path_exists && !report.installed_hash_verified {
            issues.push(format!(
                "{} local files exist but size/BLAKE3 verification failed",
                report.model_id
            ));
        }
        if runtime_claimed_active(report) && !report.model_runtime_active {
            issues.push(format!(
                "{} claims active serving but is missing runtime smoke/registration evidence: {}",
                report.model_id,
                report.required_runtime_evidence_missing.join(",")
            ));
        }
    }

    let qwen = registry
        .hardware_gated_models
        .iter()
        .find(|model| model.model_id == QWEN3_CODER_NEXT)
        .map(|model| HardwareGateReport {
            model_id: model.model_id.clone(),
            hardware_gated: model.hardware_gated,
            install_allowed: model.install_allowed,
            install_gate: model.install_gate.clone(),
            status: model.status.clone(),
            canonical_state_mutation_allowed: model.canonical_state_mutation_allowed,
        });

    match &qwen {
        Some(report) => {
            if !report.hardware_gated || report.install_allowed {
                issues.push(format!(
                    "{QWEN3_CODER_NEXT} must report hardware_gated=true and install_allowed=false"
                ));
            }
            if report.canonical_state_mutation_allowed {
                issues.push(format!(
                    "{QWEN3_CODER_NEXT} permits direct canonical state mutation"
                ));
            }
        }
        None => issues.push(format!("{QWEN3_CODER_NEXT} hardware-gated record missing")),
    }

    let status = if issues.is_empty() { "PASS" } else { "FAIL" }.to_string();
    Ok(ModelInspectReport {
        status,
        registry_path: registry_path.to_path_buf(),
        registry_version: registry.registry_version,
        canonical_state_mutation_allowed: registry.rules.canonical_state_mutation_allowed,
        gpt_oss_20b: gpt,
        qwen3_coder_next: qwen,
        issues,
    })
}

pub fn verify_model_airlock(
    registry_path: &Path,
    artifact_path: Option<&Path>,
) -> Result<ModelAirlockReport, PcError> {
    verify_model_airlock_with_options(registry_path, artifact_path, ModelInspectOptions::default())
}

pub fn verify_model_airlock_with_options(
    registry_path: &Path,
    artifact_path: Option<&Path>,
    options: ModelInspectOptions,
) -> Result<ModelAirlockReport, PcError> {
    let inspect = inspect_model_registry_with_options(registry_path, options)?;
    let registry = read_registry(registry_path)?;
    let mut issues = inspect.issues.clone();

    let mut artifact_checked = false;
    let mut model_originated = false;
    let mut output_classification = MODEL_GENERATED_UNVERIFIED.to_string();
    let mut canonical_state_mutation_allowed = registry.rules.canonical_state_mutation_allowed;
    let mut accepted_action_flow = false;
    let mut hil_accepted = false;
    let mut sealed = false;

    if let Some(path) = artifact_path {
        artifact_checked = true;
        let artifact: Value = serde_json::from_slice(&fs::read(path)?)?;
        model_originated = bool_field(&artifact, &["model_originated"])
            || string_field(&artifact, &["proposal_source"])
                .map(|value| value.eq_ignore_ascii_case("model"))
                .unwrap_or(false)
            || artifact.get("model_registry_ref").is_some();
        output_classification = string_field(
            &artifact,
            &["output_classification", "classification", "airlock_status"],
        )
        .unwrap_or_else(|| MODEL_GENERATED_UNVERIFIED.to_string());
        canonical_state_mutation_allowed = bool_field(
            &artifact,
            &[
                "canonical_state_mutation_allowed",
                "canonical_mutation_allowed",
                "canonical_influence_requested",
            ],
        );
        accepted_action_flow = has_action_acceptance(&artifact);
        hil_accepted = string_field(&artifact, &["hil_resolution", "hil_or_policy_decision"])
            .map(|value| matches!(value.as_str(), "ACCEPTED" | "POLICY_ACCEPTED"))
            .unwrap_or(false);
        sealed = non_empty_string_field(&artifact, &["seal_ref", "audit_seal_ref"])
            || string_field(&artifact, &["seal_status"])
                .map(|value| matches!(value.as_str(), "SEALED" | "AUDIT_SEALED" | "PRISM_SEALED"))
                .unwrap_or(false);

        if model_originated && output_classification != MODEL_GENERATED_UNVERIFIED {
            let accepted = accepted_action_flow && hil_accepted && sealed;
            if !accepted {
                issues.push(format!(
                    "model-originated artifact classification must remain {MODEL_GENERATED_UNVERIFIED} until ACTION/HIL/SEAL acceptance"
                ));
            }
        }

        if canonical_state_mutation_allowed && !(accepted_action_flow && hil_accepted && sealed) {
            issues.push(
                "canonical model influence requested without ACTION/HIL/SEAL acceptance"
                    .to_string(),
            );
        }
    }

    if registry.rules.canonical_state_mutation_allowed {
        issues.push("registry rules must keep canonical_state_mutation_allowed=false".to_string());
    }

    let canonical_influence_allowed = artifact_checked
        && accepted_action_flow
        && hil_accepted
        && sealed
        && !issues.iter().any(|issue| {
            issue.contains("canonical model influence requested")
                || issue.contains("classification must remain")
        });
    let status = if issues.is_empty() { "PASS" } else { "FAIL" }.to_string();

    Ok(ModelAirlockReport {
        status,
        registry_path: registry_path.to_path_buf(),
        registry_version: registry.registry_version,
        model_inspect_status: inspect.status,
        output_classification_required: MODEL_GENERATED_UNVERIFIED.to_string(),
        artifact_path: artifact_path.map(Path::to_path_buf),
        artifact_checked,
        model_originated,
        output_classification,
        canonical_state_mutation_allowed,
        accepted_action_flow,
        hil_accepted,
        sealed,
        canonical_influence_allowed,
        issues,
    })
}

pub fn smoke_model_runtime(registry_path: &Path) -> Result<ModelRuntimeSmokeReport, PcError> {
    smoke_model_runtime_with_options(registry_path, ModelRuntimeSmokeOptions::default())
}

pub fn smoke_model_runtime_with_options(
    registry_path: &Path,
    options: ModelRuntimeSmokeOptions,
) -> Result<ModelRuntimeSmokeReport, PcError> {
    let inspect = inspect_model_registry_with_options(registry_path, options.inspect)?;
    let registry = read_registry(registry_path)?;
    let model_record = registry
        .installed_models
        .iter()
        .find(|model| model.model_id == GPT_OSS_20B);
    let model_report = inspect.gpt_oss_20b.as_ref();
    let prompt_hash = blake3::hash(FIXED_RUNTIME_SMOKE_PROMPT.as_bytes())
        .to_hex()
        .to_string();
    let canonical_mutation_denial_check = canonical_mutation_denial_check(&registry, model_report);
    let runtime_candidates = discover_runtime_candidates(model_record);
    let selected_runtime = runtime_candidates
        .iter()
        .find(|candidate| candidate.compatible)
        .cloned();
    let compatible_runtime_detected = selected_runtime.is_some();
    let mut issues = inspect.issues.clone();
    let mut state_transitions = Vec::new();

    state_transitions.push(ModelRuntimeStateTransition {
        from_state: "START".to_string(),
        to_state: "REGISTRY_INSPECTED".to_string(),
        status: inspect.status.clone(),
        evidence: registry_path.display().to_string(),
    });

    let installed_hash_verified = model_report
        .map(|report| report.installed_hash_verified)
        .unwrap_or(false);
    state_transitions.push(ModelRuntimeStateTransition {
        from_state: "REGISTRY_INSPECTED".to_string(),
        to_state: if installed_hash_verified {
            "MODEL_ARTIFACT_VERIFIED"
        } else {
            "FAIL_CLOSED_MODEL_ARTIFACT_UNVERIFIED"
        }
        .to_string(),
        status: if installed_hash_verified {
            "PASS"
        } else {
            "FAIL_CLOSED"
        }
        .to_string(),
        evidence: "weight_size_and_blake3_verification".to_string(),
    });

    if !compatible_runtime_detected {
        issues.push(format!(
            "{GPT_OSS_20B} has no compatible governed runtime registered/detected; runtime smoke fails closed"
        ));
    }
    state_transitions.push(ModelRuntimeStateTransition {
        from_state: "MODEL_ARTIFACT_VERIFIED".to_string(),
        to_state: if compatible_runtime_detected {
            "GOVERNED_RUNTIME_COMPATIBLE"
        } else {
            "FAIL_CLOSED_NO_COMPATIBLE_GOVERNED_RUNTIME"
        }
        .to_string(),
        status: if compatible_runtime_detected {
            "PASS"
        } else {
            "FAIL_CLOSED"
        }
        .to_string(),
        evidence: selected_runtime
            .as_ref()
            .and_then(|runtime| runtime.command.clone())
            .unwrap_or_else(|| "runtime_registration_incomplete".to_string()),
    });

    if canonical_mutation_denial_check.status != "PASS" {
        issues.push("canonical mutation denial check failed".to_string());
    }

    let model_runtime_active = model_report
        .map(|report| report.model_runtime_active)
        .unwrap_or(false);
    if !model_runtime_active {
        issues.push(format!(
            "{GPT_OSS_20B} is not an active bounded advisory runtime; AI_ASSISTED_OS must block"
        ));
    }

    state_transitions.push(ModelRuntimeStateTransition {
        from_state: if compatible_runtime_detected {
            "GOVERNED_RUNTIME_COMPATIBLE"
        } else {
            "FAIL_CLOSED_NO_COMPATIBLE_GOVERNED_RUNTIME"
        }
        .to_string(),
        to_state: if model_runtime_active {
            "BOUNDED_ADVISORY_RUNTIME_ACTIVE"
        } else {
            "AI_ASSISTED_OS_BLOCKED_CORE_ISA_ALLOWED"
        }
        .to_string(),
        status: if model_runtime_active {
            "PASS"
        } else {
            "FAIL_CLOSED"
        }
        .to_string(),
        evidence: "model_runtime_active && runtime_smoke_status=PASSED".to_string(),
    });

    let release_mode_gates = release_mode_gates(model_report, &canonical_mutation_denial_check);
    let output_classification = model_report
        .and_then(|report| report.runtime_smoke_output_classification.clone())
        .unwrap_or_else(|| MODEL_GENERATED_UNVERIFIED.to_string());
    let status = if issues.is_empty()
        && model_runtime_active
        && compatible_runtime_detected
        && canonical_mutation_denial_check.status == "PASS"
    {
        "PASS"
    } else {
        "FAIL_CLOSED"
    }
    .to_string();
    let state = if model_runtime_active {
        "BOUNDED_ADVISORY_RUNTIME_ACTIVE"
    } else if !compatible_runtime_detected {
        "FAIL_CLOSED_NO_COMPATIBLE_GOVERNED_RUNTIME"
    } else {
        "FAIL_CLOSED_RUNTIME_SMOKE_REQUIRED"
    }
    .to_string();

    Ok(ModelRuntimeSmokeReport {
        status,
        state,
        registry_path: registry_path.to_path_buf(),
        registry_version: registry.registry_version,
        model_id: GPT_OSS_20B.to_string(),
        model_inspect_status: inspect.status,
        installed_hash_verified,
        serving_status: model_report.and_then(|report| report.serving_status.clone()),
        runtime_binding_status: model_report
            .and_then(|report| report.runtime_binding_status.clone()),
        runtime_smoke_status: model_report.and_then(|report| report.runtime_smoke_status.clone()),
        model_runtime_active,
        compatible_runtime_detected,
        selected_runtime,
        runtime_candidates,
        fixed_prompt_hash_blake3: prompt_hash,
        output_hash_blake3: model_report
            .and_then(|report| report.runtime_smoke_output_hash.clone()),
        output_classification,
        exit_status: model_report.and_then(|report| report.runtime_smoke_exit_status),
        smoke_log_or_report_ref: options.report_path,
        canonical_mutation_denial_check,
        release_mode_gates,
        state_transitions,
        required_runtime_evidence_missing: model_report
            .map(|report| report.required_runtime_evidence_missing.clone())
            .unwrap_or_else(|| vec!["openai/gpt-oss-20b registry record missing".to_string()]),
        issues,
    })
}

fn read_registry(path: &Path) -> Result<ModelRegistry, PcError> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn inspect_installed_model(
    model: &InstalledModel,
    options: ModelInspectOptions,
) -> Result<InstalledModelReport, PcError> {
    let local_path_exists = model
        .local_path
        .as_ref()
        .map(|path| path.exists())
        .unwrap_or(false);
    let mut file_checks = Vec::new();
    let expected_total_bytes = model
        .files
        .iter()
        .map(|file| file.bytes)
        .fold(0_u64, u64::saturating_add);
    let live_hash_allowed = options.force_live_hash
        || model
            .local_path
            .as_ref()
            .map(|path| {
                !is_known_remote_model_path(path)
                    && expected_total_bytes <= DEFAULT_MAX_LIVE_HASH_BYTES
            })
            .unwrap_or(false);

    if local_path_exists {
        if let Some(local_path) = &model.local_path {
            for file in &model.files {
                let absolute = local_path.join(&file.path);
                let (
                    exists,
                    bytes_observed,
                    bytes_observed_source,
                    blake3_observed,
                    verified,
                    verification_mode,
                    verification_status,
                ) = if live_hash_allowed {
                    let exists = absolute.exists();
                    let bytes_observed = if exists {
                        Some(fs::metadata(&absolute)?.len())
                    } else {
                        None
                    };
                    let blake3_observed = if exists {
                        Some(blake3_file(&absolute)?)
                    } else {
                        None
                    };
                    let verified = bytes_observed == Some(file.bytes)
                        && blake3_observed.as_deref() == Some(file.blake3.as_str());
                    (
                        exists,
                        bytes_observed,
                        "LIVE_FILESYSTEM".to_string(),
                        blake3_observed,
                        verified,
                        "LIVE_BLAKE3_REHASH".to_string(),
                        if verified {
                            "PASS_LIVE_HASH_MATCH".to_string()
                        } else {
                            "FAIL_LIVE_HASH_MISMATCH".to_string()
                        },
                    )
                } else {
                    let registry_attested = model.weights_hash_verified
                        && file.bytes > 0
                        && !file.blake3.trim().is_empty();
                    (
                        local_path_exists,
                        Some(file.bytes),
                        "REGISTRY_ATTESTED".to_string(),
                        None,
                        registry_attested,
                        "REGISTRY_ATTESTED_BOUNDED".to_string(),
                        if registry_attested {
                            if is_known_remote_model_path(local_path) {
                                "PASS_REGISTRY_ATTESTED_LIVE_HASH_SKIPPED_REMOTE_CIFS".to_string()
                            } else {
                                "PASS_REGISTRY_ATTESTED_LIVE_HASH_SKIPPED_LARGE_MODEL".to_string()
                            }
                        } else {
                            "FAIL_REGISTRY_ATTESTATION_MISSING".to_string()
                        },
                    )
                };
                file_checks.push(ModelFileCheck {
                    path: absolute,
                    exists,
                    bytes_expected: file.bytes,
                    bytes_observed,
                    bytes_observed_source,
                    blake3_expected: file.blake3.clone(),
                    blake3_observed,
                    verified,
                    verification_mode,
                    verification_status,
                });
            }
        }
    }

    let installed_hash_verified = if local_path_exists {
        !file_checks.is_empty() && file_checks.iter().all(|check| check.verified)
    } else {
        model.weights_hash_verified
    };
    let required_runtime_evidence_missing = missing_runtime_evidence(model);
    let model_runtime_active = required_runtime_evidence_missing.is_empty()
        && installed_hash_verified
        && model.runtime_binding_allowed
        && model.runtime_bound
        && string_eq(model.runtime_smoke_status.as_deref(), "PASSED")
        && !model.canonical_state_mutation_allowed;
    let model_serving_gate_status = if model_runtime_active {
        "RUNTIME_BOUND_SMOKE_PASSED_ADVISORY_ONLY".to_string()
    } else if model.serving_status.as_deref() == Some("WEIGHTS_INSTALLED_RUNTIME_UNBOUND")
        || model.runtime_smoke_status.as_deref() == Some("NOT_RUN")
    {
        "WEIGHTS_INSTALLED_RUNTIME_UNBOUND_SMOKE_REQUIRED".to_string()
    } else {
        "RUNTIME_NOT_ACTIVE_EVIDENCE_INCOMPLETE".to_string()
    };

    Ok(InstalledModelReport {
        model_id: model.model_id.clone(),
        local_path: model.local_path.clone(),
        local_path_exists,
        install_allowed: model.install_allowed,
        hardware_gated: model.hardware_gated,
        weights_hash_verified_registry: model.weights_hash_verified,
        installed_hash_verified,
        file_checks,
        serving_status: model.serving_status.clone(),
        runtime_binding_status: model.runtime_binding_status.clone(),
        runtime_binding_allowed: model.runtime_binding_allowed,
        runtime_bound: model.runtime_bound,
        runtime_name: model.runtime_name.clone(),
        runtime_version: model.runtime_version.clone(),
        runtime_hash: model.runtime_hash.clone(),
        launch_command: model.launch_command.clone(),
        endpoint: model.endpoint.clone(),
        network_policy: model.network_policy.clone(),
        gpu_cpu_memory_profile: model.gpu_cpu_memory_profile.clone(),
        context_limit: model.context_limit,
        quantization_format: model.quantization_format.clone(),
        deterministic_settings: model.deterministic_settings.clone(),
        runtime_smoke_required_before_binding: model.runtime_smoke_required_before_binding,
        runtime_smoke_status: model.runtime_smoke_status.clone(),
        runtime_smoke_ref: model.runtime_smoke_ref.clone(),
        runtime_smoke_prompt_hash: model.runtime_smoke_prompt_hash.clone(),
        runtime_smoke_output_hash: model.runtime_smoke_output_hash.clone(),
        runtime_smoke_output_classification: model.runtime_smoke_output_classification.clone(),
        runtime_smoke_exit_status: model.runtime_smoke_exit_status,
        canonical_mutation_denial_status: model.canonical_mutation_denial_status.clone(),
        model_runtime_active,
        model_serving_gate_status,
        required_runtime_evidence_missing,
        airlock_status: model
            .airlock_status
            .clone()
            .unwrap_or_else(|| MODEL_GENERATED_UNVERIFIED.to_string()),
        canonical_state_mutation_allowed: model.canonical_state_mutation_allowed,
    })
}

fn missing_runtime_evidence(model: &InstalledModel) -> Vec<String> {
    let mut missing = Vec::new();
    if !model.runtime_binding_allowed {
        missing.push("runtime_binding_allowed".to_string());
    }
    if !model.runtime_bound {
        missing.push("runtime_bound".to_string());
    }
    if !has_non_empty(model.runtime_name.as_deref()) {
        missing.push("runtime_name".to_string());
    }
    if !has_non_empty(model.runtime_version.as_deref()) {
        missing.push("runtime_version".to_string());
    }
    if !has_non_empty(model.runtime_hash.as_deref()) {
        missing.push("runtime_hash".to_string());
    }
    if !has_non_empty(model.launch_command.as_deref()) {
        missing.push("launch_command".to_string());
    }
    if !has_non_empty(model.endpoint.as_deref()) {
        missing.push("endpoint".to_string());
    }
    if !has_non_empty(model.network_policy.as_deref()) {
        missing.push("network_policy".to_string());
    }
    if model.gpu_cpu_memory_profile.is_none() {
        missing.push("gpu_cpu_memory_profile".to_string());
    }
    if model.context_limit.is_none() {
        missing.push("context_limit".to_string());
    }
    if !has_non_empty(model.quantization_format.as_deref()) {
        missing.push("quantization_format".to_string());
    }
    if model.deterministic_settings.is_none() {
        missing.push("deterministic_settings".to_string());
    }
    if !string_eq(model.runtime_smoke_status.as_deref(), "PASSED") {
        missing.push("runtime_smoke_status=PASSED".to_string());
    }
    if !has_non_empty(model.runtime_smoke_ref.as_deref()) {
        missing.push("runtime_smoke_ref".to_string());
    }
    let fixed_prompt_hash = blake3::hash(FIXED_RUNTIME_SMOKE_PROMPT.as_bytes())
        .to_hex()
        .to_string();
    if model.runtime_smoke_prompt_hash.as_deref() != Some(fixed_prompt_hash.as_str()) {
        missing.push("runtime_smoke_prompt_hash=fixed_non_secret_prompt_hash".to_string());
    }
    if !has_non_empty(model.runtime_smoke_output_hash.as_deref()) {
        missing.push("runtime_smoke_output_hash".to_string());
    }
    if !string_eq(
        model.runtime_smoke_output_classification.as_deref(),
        MODEL_GENERATED_UNVERIFIED,
    ) {
        missing.push(format!(
            "runtime_smoke_output_classification={MODEL_GENERATED_UNVERIFIED}"
        ));
    }
    if model.runtime_smoke_exit_status != Some(0) {
        missing.push("runtime_smoke_exit_status=0".to_string());
    }
    if !string_eq(model.canonical_mutation_denial_status.as_deref(), "PASSED") {
        missing.push("canonical_mutation_denial_status=PASSED".to_string());
    }
    missing
}

fn discover_runtime_candidates(model: Option<&InstalledModel>) -> Vec<RuntimeCandidateReport> {
    let mut candidates = Vec::new();
    if let Some(model) = model {
        let registry_runtime_present = has_non_empty(model.runtime_name.as_deref())
            || has_non_empty(model.launch_command.as_deref())
            || has_non_empty(model.endpoint.as_deref());
        let registry_runtime_complete = model.runtime_binding_allowed
            && model.runtime_bound
            && has_non_empty(model.runtime_name.as_deref())
            && has_non_empty(model.runtime_version.as_deref())
            && has_non_empty(model.runtime_hash.as_deref())
            && has_non_empty(model.launch_command.as_deref())
            && has_non_empty(model.endpoint.as_deref());
        candidates.push(RuntimeCandidateReport {
            runtime_name: model
                .runtime_name
                .clone()
                .unwrap_or_else(|| "registry_runtime".to_string()),
            source: "registry".to_string(),
            command: model.launch_command.clone(),
            detected: registry_runtime_present,
            compatible: registry_runtime_complete,
            version: model.runtime_version.clone(),
            binary_hash_blake3: model.runtime_hash.clone(),
            endpoint: model.endpoint.clone(),
            evidence: if registry_runtime_complete {
                "REGISTERED_BOUND_RUNTIME_EVIDENCE_COMPLETE".to_string()
            } else {
                "REGISTRY_RUNTIME_UNBOUND_OR_INCOMPLETE".to_string()
            },
        });
    }

    for (runtime_name, binary) in [
        ("vLLM", "vllm"),
        ("SGLang", "sglang"),
        ("llama.cpp", "llama-server"),
        ("Ollama", "ollama"),
    ] {
        let path = find_on_path(binary);
        let binary_hash_blake3 = path.as_ref().and_then(|path| blake3_file(path).ok());
        candidates.push(RuntimeCandidateReport {
            runtime_name: runtime_name.to_string(),
            source: "PATH".to_string(),
            command: path.as_ref().map(|path| path.display().to_string()),
            detected: path.is_some(),
            compatible: false,
            version: None,
            binary_hash_blake3,
            endpoint: None,
            evidence: if path.is_some() {
                "RUNTIME_BINARY_PRESENT_BUT_NOT_GOVERNANCE_BOUND".to_string()
            } else {
                "RUNTIME_BINARY_NOT_FOUND".to_string()
            },
        });
    }

    candidates
}

fn release_mode_gates(
    model: Option<&InstalledModelReport>,
    canonical_denial: &CanonicalMutationDenialReport,
) -> ModelRuntimeReleaseGateReport {
    let model_runtime_active = model
        .map(|report| report.model_runtime_active)
        .unwrap_or(false);
    let runtime_smoke_passed =
        model.and_then(|report| report.runtime_smoke_status.as_deref()) == Some("PASSED");
    let runtime_hash_bound = model
        .and_then(|report| report.runtime_hash.as_deref())
        .map(|hash| !hash.trim().is_empty())
        .unwrap_or(false);
    let runtime_smoke_ref_bound = model
        .and_then(|report| report.runtime_smoke_ref.as_deref())
        .map(|reference| !reference.trim().is_empty())
        .unwrap_or(false);
    let mut ai_assisted_os_blockers = Vec::new();
    if !model_runtime_active {
        ai_assisted_os_blockers.push("MODEL_RUNTIME_ACTIVE".to_string());
    }
    if !runtime_smoke_passed {
        ai_assisted_os_blockers.push("MODEL_RUNTIME_SMOKE_PASSED".to_string());
    }
    if !runtime_hash_bound {
        ai_assisted_os_blockers.push("MODEL_RUNTIME_HASH_BOUND".to_string());
    }
    if !runtime_smoke_ref_bound {
        ai_assisted_os_blockers.push("MODEL_RUNTIME_SMOKE_REF_BOUND".to_string());
    }
    if canonical_denial.status != "PASS" {
        ai_assisted_os_blockers.push("CANONICAL_MUTATION_DENIAL_PASSED".to_string());
    }

    let core_isa_model_gate = if canonical_denial.status == "PASS" {
        if model_runtime_active {
            "ALLOWED_BOUNDED_ADVISORY_RUNTIME_ACTIVE"
        } else {
            "ALLOWED_ADVISORY_RUNTIME_INACTIVE"
        }
    } else {
        "BLOCKED_CANONICAL_MUTATION_POLICY_VIOLATION"
    }
    .to_string();
    let ai_assisted_os_model_gate = if ai_assisted_os_blockers.is_empty() {
        "ALLOWED_BOUNDED_ADVISORY_RUNTIME_ACTIVE"
    } else {
        "BLOCKED_MODEL_RUNTIME_SMOKE_REQUIRED"
    }
    .to_string();

    ModelRuntimeReleaseGateReport {
        core_isa_model_gate,
        ai_assisted_os_model_gate,
        ai_assisted_os_blockers,
    }
}

fn canonical_mutation_denial_check(
    registry: &ModelRegistry,
    model: Option<&InstalledModelReport>,
) -> CanonicalMutationDenialReport {
    let mut issues = Vec::new();
    if registry.rules.canonical_state_mutation_allowed {
        issues.push("registry permits direct canonical state mutation".to_string());
    }
    match model {
        Some(report) if report.canonical_state_mutation_allowed => {
            issues.push(format!(
                "{} permits direct canonical state mutation",
                report.model_id
            ));
        }
        Some(_) => {}
        None => issues.push(format!("{GPT_OSS_20B} registry record missing")),
    }
    let denied = issues.is_empty();
    CanonicalMutationDenialReport {
        status: if denied { "PASS" } else { "FAIL" }.to_string(),
        denied,
        probe_model_registry_ref: GPT_OSS_20B.to_string(),
        probe_output_classification: MODEL_GENERATED_UNVERIFIED.to_string(),
        probe_canonical_state_mutation_allowed: true,
        reason: if denied {
            "direct model-originated canonical mutation probe lacks ACTION/HIL/SEAL acceptance and is denied".to_string()
        } else {
            "direct model-originated canonical mutation denial invariant is broken".to_string()
        },
        issues,
    }
}

fn find_on_path(binary: &str) -> Option<PathBuf> {
    env::var_os("PATH").and_then(|paths| {
        env::split_paths(&paths)
            .map(|dir| dir.join(binary))
            .find(|path| path.is_file())
    })
}

fn runtime_claimed_active(report: &InstalledModelReport) -> bool {
    report.runtime_bound
        || report.runtime_binding_allowed
        || report
            .serving_status
            .as_deref()
            .map(|status| {
                matches!(
                    status,
                    "RUNTIME_BOUND_SMOKE_PASSED" | "ACTIVE" | "OPERATIONAL"
                )
            })
            .unwrap_or(false)
}

fn has_non_empty(value: Option<&str>) -> bool {
    value.map(|text| !text.trim().is_empty()).unwrap_or(false)
}

fn string_eq(value: Option<&str>, expected: &str) -> bool {
    value.map(|text| text == expected).unwrap_or(false)
}

fn is_known_remote_model_path(path: &Path) -> bool {
    let text = path.to_string_lossy();
    text.starts_with("/mnt/prism4d-nas/") || text.starts_with("/mnt/storage/NAS/")
}

fn blake3_file(path: &Path) -> Result<String, PcError> {
    let mut hasher = blake3::Hasher::new();
    let mut file = fs::File::open(path)?;
    std::io::copy(&mut file, &mut hasher)?;
    Ok(hasher.finalize().to_hex().to_string())
}

fn string_field(value: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .filter_map(|key| value.get(*key).and_then(Value::as_str))
        .next()
        .map(str::to_string)
}

fn non_empty_string_field(value: &Value, keys: &[&str]) -> bool {
    string_field(value, keys)
        .map(|field| !field.trim().is_empty())
        .unwrap_or(false)
}

fn bool_field(value: &Value, keys: &[&str]) -> bool {
    keys.iter()
        .filter_map(|key| value.get(*key).and_then(Value::as_bool))
        .next()
        .unwrap_or(false)
}

fn has_action_acceptance(value: &Value) -> bool {
    let constraints_applied = value
        .get("constraints_applied")
        .and_then(Value::as_array)
        .map(|items| !items.is_empty())
        .unwrap_or(false);
    let action_stage_ok = string_field(value, &["action_stage", "action_status"])
        .map(|stage| {
            matches!(
                stage.as_str(),
                "ACTION.CONSTRAIN" | "ACTION.RANK" | "ACCEPTED"
            )
        })
        .unwrap_or(false);
    value.get("parsed_action_schema").is_some() && (constraints_applied || action_stage_ok)
}
