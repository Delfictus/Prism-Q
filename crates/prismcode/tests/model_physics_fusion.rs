//! PRISM Physics-Fusion model line (PHASE-IV) coverage.
//!
//! The physics-fusion Stage-1 artifact (`best_model.pt` = champion encoder +
//! `tsh_head` + `sigma_head`) is admitted as a FIRST-CLASS catalog model that is
//! SEPARATE from and ADDITIVE to the immutable champion entry. These tests prove:
//!   * the new id resolves with the right posture (fused `.pt`, attestation-only);
//!   * the canonical champion entry is byte-for-byte unchanged;
//!   * unknown ids still fail closed;
//!   * a fused-`.pt` registry record for the fusion model verifies by BLAKE3 +
//!     size WITHOUT any safetensors header — on CPU, without the 16 GiB weights.

use prismcode::model_catalog::{
    self, ModelArtifactClass, ModelBridgeKind, PRISM_CHAMPION_PLLM_QWEN3_8B,
    PRISM_PHYSICS_FUSION_V1,
};
use prismcode::model_runtime::load_model_artifact;
use serde_json::json;
use std::fs;
use std::path::{Path, PathBuf};

#[test]
fn catalog_admits_physics_fusion_as_first_class() {
    assert!(model_catalog::is_allowed_canonical(
        "prism_physics_fusion_v1"
    ));

    let fusion = model_catalog::require_canonical("prism_physics_fusion_v1").unwrap();
    assert_eq!(fusion.registry_id, "prism/physics-fusion-v1");
    assert_eq!(fusion.artifact_class, ModelArtifactClass::FusedCheckpointPt);
    assert_eq!(fusion.bridge, ModelBridgeKind::AttestationOnly);
    // Resolvable by the Hub-style registry id too (the load path uses this).
    let by_reg = model_catalog::resolve_by_registry("prism/physics-fusion-v1").unwrap();
    assert_eq!(by_reg.canonical_id, "prism_physics_fusion_v1");
    assert_eq!(
        PRISM_PHYSICS_FUSION_V1.canonical_id,
        "prism_physics_fusion_v1"
    );
}

#[test]
fn champion_entry_is_unchanged_by_the_additive_registration() {
    // The canonical champion must be IMMUTABLE: adding the fusion line changes
    // none of its fields.
    let champ = model_catalog::require_canonical("prism_champion_pllm_qwen3_8b").unwrap();
    assert_eq!(champ.registry_id, "prism/champion-pllm-qwen3-8b");
    assert_eq!(champ.artifact_class, ModelArtifactClass::FusedCheckpointPt);
    assert_eq!(champ.bridge, ModelBridgeKind::PythonDevboxBridge);
    assert_eq!(
        PRISM_CHAMPION_PLLM_QWEN3_8B.canonical_id,
        "prism_champion_pllm_qwen3_8b"
    );
    // The historical default is still gpt-oss (unaffected by the new entry).
    assert_eq!(
        model_catalog::DEFAULT_MODEL.canonical_id,
        "openai_gpt_oss_20b"
    );
}

#[test]
fn unknown_ids_still_fail_closed_after_registration() {
    assert!(!model_catalog::is_allowed_canonical("some_random_model"));
    let err = model_catalog::require_canonical("nope")
        .unwrap_err()
        .to_string();
    assert!(err.contains("unsupported model id for bounded runtime"));
}

#[test]
fn fusion_fused_pt_artifact_verifies_by_hash_without_safetensors() {
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("physics_fusion_model");
    fs::create_dir_all(&model_dir).unwrap();
    // Stand-in fused checkpoint (real best_model.pt is 16.26 GiB and lives on the
    // pod / OPERATOR_CONTROL). The architecture + verifier path is identical.
    fs::write(model_dir.join("best_model.pt"), b"PHYSICS_FUSION_STANDIN").unwrap();
    let registry = write_fusion_registry(tmp.path(), &model_dir);

    let proof = load_model_artifact(&registry, "prism/physics-fusion-v1").unwrap();
    assert_eq!(proof.status, "PASS");
    assert_eq!(proof.artifact_class, "FUSED_CHECKPOINT_PT");
    assert!(proof.all_model_hashes_verified);
    // Passed WITHOUT tokenizer.json / config.json — only BLAKE3 + size of the
    // listed `.pt` were required (fused-checkpoint class).
    assert!(!proof.tokenizer_hash_verified);
    assert!(!proof.config_hash_verified);
}

#[test]
fn fusion_fused_pt_artifact_fails_on_hash_mismatch() {
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("physics_fusion_model");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(model_dir.join("best_model.pt"), b"PHYSICS_FUSION_STANDIN").unwrap();
    let ckpt_bytes = fs::read(model_dir.join("best_model.pt")).unwrap();
    let registry = json!({
        "registry_version": "TEST_PHYSICS_FUSION_RUNTIME",
        "rules": {"canonical_state_mutation_allowed": false},
        "installed_models": [{
            "model_id": "prism/physics-fusion-v1",
            "artifact_class": "FUSED_CHECKPOINT_PT",
            "local_path": model_dir,
            "files": [
                {"path": "best_model.pt", "bytes": ckpt_bytes.len(), "blake3": "deadbeef"}
            ]
        }]
    });
    let path = tmp.path().join("fusion_registry_bad.json");
    fs::write(&path, serde_json::to_vec_pretty(&registry).unwrap()).unwrap();
    let proof = load_model_artifact(&path, "prism/physics-fusion-v1").unwrap();
    assert_eq!(proof.status, "FAIL");
    assert!(!proof.all_model_hashes_verified);
}

fn write_fusion_registry(root: &Path, model_dir: &Path) -> PathBuf {
    let bytes = fs::read(model_dir.join("best_model.pt")).unwrap();
    let registry = json!({
        "registry_version": "TEST_PHYSICS_FUSION_RUNTIME",
        "rules": {"canonical_state_mutation_allowed": false},
        "installed_models": [{
            "model_id": "prism/physics-fusion-v1",
            "artifact_class": "FUSED_CHECKPOINT_PT",
            "local_path": model_dir,
            "install_allowed": true,
            "hardware_gated": false,
            "canonical_state_mutation_allowed": false,
            "files": [{
                "path": "best_model.pt",
                "bytes": bytes.len(),
                "blake3": blake3::hash(&bytes).to_hex().to_string(),
            }]
        }]
    });
    let path = root.join("fusion_registry.json");
    fs::write(&path, serde_json::to_vec_pretty(&registry).unwrap()).unwrap();
    path
}
