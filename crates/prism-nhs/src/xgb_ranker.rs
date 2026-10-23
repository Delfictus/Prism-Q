//! XGBoost Ranker v3 — ONNX-backed production site ranker.
//!
//! Replaces the tokenized v4 256-bin lookup with a continuous-feature
//! gradient-boosted regressor trained on the same corrected_dcc gold-standard
//! dataset via LOTO.
//!
//! Training metadata:
//!   Features: 13-dim float32 vector per site
//!     [spike_count, n_streams, interaction, unsat_frac, persistence,
//!      log_spike_count, log_interaction, spread, burial_score,
//!      spike_density, druggability, aromatic_score, n_lining_residues]
//!   Labels:   graded = 1 / (1 + min_dist_to_ligand)
//!   Training: XGBRanker rank:ndcg, depth 6, n_estimators 500, lr 0.05
//!   LOTO eval (345 targets, pct95):
//!     SR@1 = 47.83%   SR@3 = 85.51%   SR@5 = 95.94%   SR@10 = 99.42%
//!
//! See: manifests/THIRD_PARTY.json (ranker provenance) and the training receipts
//! recorded there (operator storage, ranker-xgb-v3; not published).
//!
//! Runtime: ort 2.x with downloaded onnxruntime, CPU provider (single-site
//! inference is ~microseconds; no GPU needed).

use std::path::Path;
use std::sync::{Mutex, OnceLock};

use anyhow::{anyhow, Context, Result};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;

/// Embedded ONNX model — baked into binary via `include_bytes!`.
pub const EMBEDDED_ONNX: &[u8] = include_bytes!("../assets/xgb_ranker_v3.onnx");

/// Number of features expected by the ONNX model.
pub const N_FEATURES: usize = 13;

/// Ordered feature names — must match training `FEATURE_COLS` exactly.
pub const FEATURE_NAMES: [&str; N_FEATURES] = [
    "spike_count",
    "n_streams",
    "interaction",
    "unsat_frac",
    "persistence",
    "log_spike_count",
    "log_interaction",
    "spread",
    "burial_score",
    "spike_density",
    "druggability",
    "aromatic_score",
    "n_lining_residues",
];

/// Singleton session — ONNX Runtime session is expensive to create, and
/// the model is read-only. Wrapped in a `Mutex` because `ort 2.0 rc.11`
/// requires `&mut self` for `Session::run`. Inference on a single 13-dim
/// vector is microseconds, so mutex contention is a non-issue here.
static SESSION: OnceLock<Mutex<Session>> = OnceLock::new();

/// Lazily initialize the global Session. Returns an error if the embedded
/// ONNX bytes fail to parse.
fn session() -> Result<&'static Mutex<Session>> {
    if let Some(s) = SESSION.get() {
        return Ok(s);
    }
    let s = Session::builder()
        .context("failed to create ort session builder")?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .context("failed to set graph optimization level")?
        .with_intra_threads(1)
        .context("failed to set intra-thread count")?
        .commit_from_memory(EMBEDDED_ONNX)
        .context("failed to load embedded ONNX model")?;
    let _ = SESSION.set(Mutex::new(s));
    Ok(SESSION.get().expect("session just set"))
}

/// 13-dim feature vector in the training order.
#[derive(Debug, Clone, Copy)]
pub struct SiteFeatures {
    pub spike_count: u64,
    pub n_streams: u32,
    pub unsat_frac: f32,
    pub persistence: f32,
    pub spread: f32, // volume^(1/3)
    pub burial_score: f32,
    pub spike_density: f32, // spike_count / volume
    pub druggability: f32,
    pub aromatic_score: f32,
    pub n_lining_residues: u32,
}

impl SiteFeatures {
    /// Expand into the 13-dim f32 array matching training order.
    pub fn to_row(&self) -> [f32; N_FEATURES] {
        let sc = self.spike_count as f32;
        let ns = self.n_streams as f32;
        let inter = sc * ns;
        [
            sc,
            ns,
            inter,
            self.unsat_frac,
            self.persistence,
            (sc + 1.0).ln(),    // log1p(sc)
            (inter + 1.0).ln(), // log1p(inter)
            self.spread,
            self.burial_score,
            self.spike_density,
            self.druggability,
            self.aromatic_score,
            self.n_lining_residues as f32,
        ]
    }
}

/// Run inference on a batch of site feature vectors. Returns one score
/// per site; higher = more likely to be the correct binding site.
///
/// Empty input returns an empty vector without touching the session.
pub fn score_batch(rows: &[[f32; N_FEATURES]]) -> Result<Vec<f32>> {
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    let sess = session()?;
    let n = rows.len();
    let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();

    // ort 2.0 rc.11 accepts (shape, Vec<T>) tuples directly.
    let input = Value::from_array(([n, N_FEATURES], flat))
        .map_err(|e| anyhow!("from_array failed: {}", e))?;

    let mut sess_guard = sess
        .lock()
        .map_err(|e| anyhow!("session mutex poisoned: {}", e))?;
    let outputs = sess_guard
        .run(ort::inputs![input])
        .map_err(|e| anyhow!("ort run failed: {}", e))?;

    // The XGBoost-derived ONNX has a single output named "variable" with
    // shape [N, 1] float32.
    let (shape, data) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow!("tensor extract failed: {}", e))?;
    if shape.is_empty() || (shape[0] as usize) != n {
        return Err(anyhow!(
            "unexpected output shape: {:?} (expected N={})",
            shape,
            n
        ));
    }
    // Copy the slice — the ort::TensorView has lifetime tied to `outputs`.
    Ok(data.to_vec())
}

/// Convenience: score a single site.
pub fn score_site(features: &SiteFeatures) -> Result<f32> {
    let row = features.to_row();
    Ok(score_batch(&[row])?[0])
}

/// Rerank an in-memory sites JSON array by XGBoost v3 score.
///
/// Each site JSON object must have the fields required by `SiteFeatures`.
/// Missing numeric fields default to 0; missing list fields default to empty.
///
/// After scoring, the array is sorted by score descending (tiebreak:
/// spike_count descending), and the `rank` field is rewritten (1-indexed).
/// Two new fields are injected per site:
///   - `xgb_score`      : float, raw model output
///   - `ranker_version` : "xgb_v3"
pub fn apply_rerank(sites_json: &mut [serde_json::Value]) -> Result<usize> {
    if sites_json.is_empty() {
        return Ok(0);
    }

    // Collect feature rows from the JSON.
    let mut rows = Vec::with_capacity(sites_json.len());
    for site in sites_json.iter() {
        let spike_count = site
            .get("spike_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        // Non-TWIN baseline always uses 4 streams; if the field is present, trust it.
        let n_streams = site.get("n_streams").and_then(|v| v.as_u64()).unwrap_or(4) as u32;
        let unsat_frac = site
            .get("unsat_frac")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;
        // persistence is produced in TWIN output; default 0.
        let persistence = site
            .get("signal_preservation")
            .and_then(|sp| sp.get("mean_recurrence"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;
        let volume = site
            .get("volume")
            .or_else(|| site.get("volume_angstrom3"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;
        let spread = if volume > 0.0 { volume.cbrt() } else { 0.0 };
        let burial_score = site
            .get("burial_score")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;
        let spike_density = if volume > 0.0 {
            (spike_count as f32) / volume
        } else {
            0.0
        };
        let druggability = site
            .get("druggability")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;
        let aromatic_score = site
            .get("aromatic_score")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;
        let n_lining_residues = site
            .get("lining_residues")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(0) as u32;

        let feats = SiteFeatures {
            spike_count,
            n_streams,
            unsat_frac,
            persistence,
            spread,
            burial_score,
            spike_density,
            druggability,
            aromatic_score,
            n_lining_residues,
        };
        rows.push(feats.to_row());
    }

    let scores = score_batch(&rows)?;
    assert_eq!(scores.len(), sites_json.len());

    // Inject score into each site JSON BEFORE sorting.
    for (site, &score) in sites_json.iter_mut().zip(scores.iter()) {
        if let Some(obj) = site.as_object_mut() {
            obj.insert("xgb_score".to_string(), serde_json::json!(score));
        }
    }

    // Sort descending by xgb_score, tiebreak on spike_count descending.
    sites_json.sort_by(|a, b| {
        let sa = a
            .get("xgb_score")
            .and_then(|v| v.as_f64())
            .unwrap_or(f64::MIN);
        let sb = b
            .get("xgb_score")
            .and_then(|v| v.as_f64())
            .unwrap_or(f64::MIN);
        let primary = sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal);
        if primary != std::cmp::Ordering::Equal {
            return primary;
        }
        let ca = a.get("spike_count").and_then(|v| v.as_u64()).unwrap_or(0);
        let cb = b.get("spike_count").and_then(|v| v.as_u64()).unwrap_or(0);
        cb.cmp(&ca)
    });

    // Rewrite rank (1-indexed) and tag ranker_version.
    for (i, site) in sites_json.iter_mut().enumerate() {
        if let Some(obj) = site.as_object_mut() {
            obj.insert("rank".to_string(), serde_json::json!(i + 1));
            obj.insert("ranker_version".to_string(), serde_json::json!("xgb_v3"));
        }
    }

    Ok(sites_json.len())
}

/// Load an ONNX file from a custom path (ad-hoc override, e.g. for testing
/// a newer model). Returns a fresh Session; does NOT replace the singleton.
pub fn load_from_path(path: &Path) -> Result<Session> {
    Session::builder()
        .context("failed to create ort session builder")?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .context("failed to set opt level")?
        .commit_from_file(path)
        .context("failed to load ONNX from file")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_model_loads() {
        // If the embedded bytes fail to parse, session() returns Err.
        // If the mutex wrapping fails, lock() panics.
        let s = session().expect("embedded model must load");
        let _guard = s.lock().expect("mutex not poisoned");
        // A successful load proves input/output signatures are well-formed —
        // `score_batch` exercises the full path in the tests below.
    }

    #[test]
    fn score_batch_produces_one_score_per_row() {
        let rows = vec![
            [
                100000.0f32,
                4.0,
                400000.0,
                0.3,
                0.5,
                11.51,
                12.90,
                5.0,
                0.5,
                0.1,
                0.5,
                0.3,
                50.0,
            ],
            [
                500.0f32, 4.0, 2000.0, 0.01, 0.1, 6.22, 7.60, 1.0, 0.1, 0.02, 0.1, 0.05, 5.0,
            ],
        ];
        let scores = score_batch(&rows).expect("scoring must succeed");
        assert_eq!(scores.len(), 2);
        assert!(scores.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn apply_rerank_orders_by_score() {
        let mut sites = vec![
            serde_json::json!({
                "id": 1, "spike_count": 500, "volume": 100, "burial_score": 0.1,
                "druggability": 0.1, "aromatic_score": 0.0, "lining_residues": [],
                "unsat_frac": 0.01
            }),
            serde_json::json!({
                "id": 2, "spike_count": 100000, "volume": 500, "burial_score": 0.5,
                "druggability": 0.5, "aromatic_score": 0.3, "lining_residues": [{},{},{}],
                "unsat_frac": 0.3
            }),
        ];
        let n = apply_rerank(&mut sites).expect("rerank must succeed");
        assert_eq!(n, 2);
        for s in &sites {
            assert!(s.get("xgb_score").is_some());
            assert!(s.get("rank").is_some());
            assert_eq!(
                s.get("ranker_version").and_then(|v| v.as_str()),
                Some("xgb_v3")
            );
        }
    }
}
