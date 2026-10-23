//! Fail-closed xgb-v3 surface for builds without ONNX Runtime.
//!
//! The canonical `phase-manifold` ranker does not use this module. An explicit
//! `--site-ranker=xgb-v3` request must terminate instead of preserving a prior
//! ordering and presenting it as XGBoost output.

/// Terminate an explicit xgb-v3 request when ONNX support is not compiled.
pub fn apply_rerank(_sites_json: &mut [serde_json::Value]) -> anyhow::Result<usize> {
    panic!("xgb-v3 ranker requested, but this binary was built without the explicit `onnx` feature")
}
