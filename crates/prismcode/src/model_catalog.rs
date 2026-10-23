//! Canonical model catalog for the bounded PRISM advisory runtime.
//!
//! Historically the runtime was hard-wired to a single model id
//! (`openai_gpt_oss_20b`) via a pair of `&str` constants and equality checks in
//! [`crate::model_runtime`] and [`crate::model_policy`]. That single constant is
//! replaced here with a small, closed allow-list so additional governed models
//! (e.g. the champion PRISM PLLM) can be attested under the SAME security
//! posture — localhost only, no egress, no canonical mutation, all hashes
//! required. Adding a model to [`ALLOWED_CANONICAL_MODELS`] is an explicit,
//! reviewed act; any id absent from the list is still rejected exactly as
//! before.
//!
//! See `docs/decisions/ADR-0016-champion-pllm-qwen3-8b-promotion.md` for the
//! governance rationale behind admitting the champion id.

use crate::binary_runtime::{PcError, PcResult};

/// How the registered artifact is packaged on disk. This drives which files the
/// artifact verifier requires before it reports `all_model_hashes_verified`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArtifactClass {
    /// HuggingFace-style shards: `tokenizer.json` + `config.json` +
    /// `*.safetensors`. Every safetensors shard must additionally parse its
    /// header. This is the historical (and default) class.
    Safetensors,
    /// A single fused PyTorch checkpoint (`.pt`) attested by BLAKE3 + byte size.
    /// There is no safetensors header to parse; the verifier requires at least
    /// one `.pt` file, and (as with every class) that all registered files hash
    /// and size verify.
    FusedCheckpointPt,
}

impl ModelArtifactClass {
    /// Parse an optional registry-supplied `artifact_class` string. Unknown
    /// values are a hard error rather than a silent fallback so a typo cannot
    /// downgrade verification.
    pub fn from_registry_str(raw: &str) -> PcResult<Self> {
        match raw {
            "SAFETENSORS_SHARDS" | "safetensors" => Ok(Self::Safetensors),
            "FUSED_CHECKPOINT_PT" | "pt" | "torch_pt" => Ok(Self::FusedCheckpointPt),
            other => Err(PcError::Format(format!(
                "unknown model artifact_class in registry: {other}"
            ))),
        }
    }

    /// Stable string form recorded in the artifact proof for transparency.
    pub fn as_registry_str(self) -> &'static str {
        match self {
            Self::Safetensors => "SAFETENSORS_SHARDS",
            Self::FusedCheckpointPt => "FUSED_CHECKPOINT_PT",
        }
    }
}

/// How, if at all, a live forward pass is served for a catalog model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelBridgeKind {
    /// Attestation / advisory only — no forward pass. This is the historical
    /// `prism-modeld` shell: it verifies hashes and bounds a handshake but never
    /// runs the model.
    AttestationOnly,
    /// Governed Python devbox server bridge (spawn / health-check / proxy),
    /// implemented in [`crate::model_runtime_champion`] and gated behind the
    /// `champion_bridge` cargo feature. Still advisory and localhost only.
    PythonDevboxBridge,
}

/// One entry in the closed allow-list of models the bounded runtime may attest.
#[derive(Debug, Clone, Copy)]
pub struct CanonicalModel {
    /// PRISMCODE-internal canonical id (underscored, used on the CLI and wire).
    pub canonical_id: &'static str,
    /// Registry / Hub-style id used as the `model_id` key inside a registry
    /// JSON `installed_models` record.
    pub registry_id: &'static str,
    /// On-disk packaging class (drives artifact verification requirements).
    pub artifact_class: ModelArtifactClass,
    /// Serving bridge posture.
    pub bridge: ModelBridgeKind,
}

/// The historical default advisory model (unchanged posture).
pub const OPENAI_GPT_OSS_20B: CanonicalModel = CanonicalModel {
    canonical_id: "openai_gpt_oss_20b",
    registry_id: "openai/gpt-oss-20b",
    artifact_class: ModelArtifactClass::Safetensors,
    bridge: ModelBridgeKind::AttestationOnly,
};

/// The champion PRISM PLLM (Qwen3-8B-Base backbone + 4 `prism_fusion`
/// cross-attention layers + PRISM encoder + reconstruction head), fused into a
/// single `ckpt_step108500.pt`. Served through the governed Python devbox
/// bridge. Promoted by ADR-0016.
pub const PRISM_CHAMPION_PLLM_QWEN3_8B: CanonicalModel = CanonicalModel {
    canonical_id: "prism_champion_pllm_qwen3_8b",
    registry_id: "prism/champion-pllm-qwen3-8b",
    artifact_class: ModelArtifactClass::FusedCheckpointPt,
    bridge: ModelBridgeKind::PythonDevboxBridge,
};

/// The PRISM Physics-Fusion Stage-1 model: the champion token encoder plus two
/// explicit physics heads (`tsh_head` 6-class thermodynamic-state classifier +
/// `sigma_head` uncertainty regressor), fused into a single `best_model.pt`.
/// This is a SEPARATE, additive production line built ON the immutable champion
/// checkpoint — it never overwrites or mutates the champion entry. Attestation
/// posture only in-tree (BLAKE3 + size verify a `.pt`, no forward pass); live
/// GPU inference is a later governed bridge on the training pod. Same security
/// envelope as every catalog model: localhost only, no egress, advisory,
/// canonical mutation forbidden.
pub const PRISM_PHYSICS_FUSION_V1: CanonicalModel = CanonicalModel {
    canonical_id: "prism_physics_fusion_v1",
    registry_id: "prism/physics-fusion-v1",
    artifact_class: ModelArtifactClass::FusedCheckpointPt,
    bridge: ModelBridgeKind::AttestationOnly,
};

/// The model used for the historical default constants
/// ([`crate::model_runtime::MODEL_ID_CANONICAL`] /
/// [`crate::model_runtime::MODEL_ID_REGISTRY`]).
pub const DEFAULT_MODEL: CanonicalModel = OPENAI_GPT_OSS_20B;

/// Closed allow-list of models the bounded runtime may serve. Any id not present
/// here is rejected with the same error the single hard constant used to raise.
pub const ALLOWED_CANONICAL_MODELS: &[CanonicalModel] = &[
    OPENAI_GPT_OSS_20B,
    PRISM_CHAMPION_PLLM_QWEN3_8B,
    PRISM_PHYSICS_FUSION_V1,
];

/// Resolve by canonical (underscored) id.
pub fn resolve_canonical(model_id: &str) -> Option<CanonicalModel> {
    ALLOWED_CANONICAL_MODELS
        .iter()
        .copied()
        .find(|model| model.canonical_id == model_id)
}

/// Resolve by registry (Hub-style) id.
pub fn resolve_by_registry(registry_id: &str) -> Option<CanonicalModel> {
    ALLOWED_CANONICAL_MODELS
        .iter()
        .copied()
        .find(|model| model.registry_id == registry_id)
}

/// Is this canonical id an admitted model?
pub fn is_allowed_canonical(model_id: &str) -> bool {
    resolve_canonical(model_id).is_some()
}

/// Resolve a canonical id or fail with the historical rejection error. This is
/// the single choke point that replaced `model_id != MODEL_ID_CANONICAL`.
pub fn require_canonical(model_id: &str) -> PcResult<CanonicalModel> {
    resolve_canonical(model_id).ok_or_else(|| {
        PcError::Security(format!(
            "unsupported model id for bounded runtime: {model_id}"
        ))
    })
}
