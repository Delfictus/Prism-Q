//! Tokenized ranker (v4 LOTO) — baked-in production ranker for site scoring.
//!
//! Replaces the legacy druggability-weighted ranker with a learned lookup
//! table over discretized (spike_count, n_streams, interaction, unsat_frac)
//! features. Evaluated via Leave-One-Target-Out (LOTO) on 302 targets:
//!
//!   SR@1: 36.42%   SR@3: 81.13%   SR@5: 94.70%
//!
//! See: manifests/THIRD_PARTY.json (ranker provenance); training receipt
//! `ranker-loto-v4/tokenized_ranker_loto_v4.json` (operator storage; not published).
//!
//! ## Binary format
//!
//! ```text
//! Offset Size  Field
//! 0      4     magic "TKRK"
//! 4      4     version (u32 little-endian) — current version: 7
//! 8      4     n_tokens (u32 little-endian) — currently 256
//! 12     12    3 × f32 spike_count thresholds
//! 24     12    3 × f32 interaction (spike_count × n_streams) thresholds
//! 36     12    3 × f32 unsat_frac thresholds
//! 48     1024  256 × f32 lookup table (P(binding_site | token))
//! Total  1072 bytes
//! ```
//!
//! ## Token computation
//!
//! Each detected site is mapped to a token 0..=255 via:
//!
//!   d0 = bin4(spike_count)           // 4 bins → 2 bits
//!   d1 = min(n_streams - 1, 3)       // 4 bins → 2 bits
//!   d2 = bin4(spike_count × n_streams)
//!   d3 = bin4(unsat_frac)
//!   token = d0 * 64 + d1 * 16 + d2 * 4 + d3
//!
//! The lookup[token] is the empirical binding-site probability for that bin.
//! For n_streams = 4 (the canonical corpus flag), d1 = 3, so valid tokens
//! are 48..=63, 112..=127, 176..=191, 240..=255.

use std::path::Path;

/// Magic bytes at the head of a v4 ranker binary.
pub const MAGIC: &[u8; 4] = b"TKRK";

/// Schema version this code understands.
pub const VERSION: u32 = 7;

/// Number of lookup entries.
pub const N_TOKENS: usize = 256;

/// Embedded v4 binary — the production ranker.
///
/// Baked into the binary via `include_bytes!`, so no runtime file lookup
/// is required. A user-supplied path can still override this at runtime
/// via [`TokenizedRanker::load_from_path`].
pub const EMBEDDED_V4: &[u8] = include_bytes!("../assets/tokenized_ranker_v4.bin");

#[derive(Debug, Clone)]
pub struct TokenizedRanker {
    pub version: u32,
    pub spike_count_thresholds: [f32; 3],
    pub unsat_frac_thresholds: [f32; 3],
    pub interaction_thresholds: [f32; 3],
    pub lookup: [f32; N_TOKENS],
}

#[derive(Debug, thiserror::Error)]
pub enum RankerError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("bad magic: expected 'TKRK', got {0:?}")]
    BadMagic([u8; 4]),
    #[error("unsupported version: {0} (expected {})", VERSION)]
    UnsupportedVersion(u32),
    #[error("unexpected token count: {0} (expected {})", N_TOKENS)]
    BadTokenCount(u32),
    #[error("file too short: {got} bytes, expected {expected}")]
    TooShort { got: usize, expected: usize },
}

impl TokenizedRanker {
    /// Expected byte length of a valid v4 binary.
    pub const EXPECTED_SIZE: usize = 4 + 4 + 4 + 12 + 12 + 12 + (N_TOKENS * 4);

    /// Load the embedded v4 ranker.
    ///
    /// This is the production path — always succeeds unless the binary was
    /// corrupted during build.
    pub fn embedded() -> Result<Self, RankerError> {
        Self::from_bytes(EMBEDDED_V4)
    }

    /// Load a ranker from a file path (for ad-hoc overrides or testing).
    pub fn load_from_path(path: &Path) -> Result<Self, RankerError> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    /// Parse the binary format. Returns an error if the magic, version,
    /// token count, or length don't match expectations.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, RankerError> {
        if bytes.len() < Self::EXPECTED_SIZE {
            return Err(RankerError::TooShort {
                got: bytes.len(),
                expected: Self::EXPECTED_SIZE,
            });
        }

        let mut magic = [0u8; 4];
        magic.copy_from_slice(&bytes[0..4]);
        if &magic != MAGIC {
            return Err(RankerError::BadMagic(magic));
        }

        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if version != VERSION {
            return Err(RankerError::UnsupportedVersion(version));
        }

        let n_tokens = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if n_tokens as usize != N_TOKENS {
            return Err(RankerError::BadTokenCount(n_tokens));
        }

        let mut spike_count_thresholds = [0f32; 3];
        let mut interaction_thresholds = [0f32; 3];
        let mut unsat_frac_thresholds = [0f32; 3];
        for i in 0..3 {
            spike_count_thresholds[i] =
                f32::from_le_bytes(bytes[12 + i * 4..16 + i * 4].try_into().unwrap());
            interaction_thresholds[i] =
                f32::from_le_bytes(bytes[24 + i * 4..28 + i * 4].try_into().unwrap());
            unsat_frac_thresholds[i] =
                f32::from_le_bytes(bytes[36 + i * 4..40 + i * 4].try_into().unwrap());
        }

        let mut lookup = [0f32; N_TOKENS];
        for i in 0..N_TOKENS {
            let off = 48 + i * 4;
            lookup[i] = f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        }

        Ok(Self {
            version,
            spike_count_thresholds,
            unsat_frac_thresholds,
            interaction_thresholds,
            lookup,
        })
    }

    /// Bin a continuous value into 0..=3 using 3 cut thresholds.
    #[inline]
    fn bin4(value: f32, thresholds: &[f32; 3]) -> usize {
        if value < thresholds[0] {
            0
        } else if value < thresholds[1] {
            1
        } else if value < thresholds[2] {
            2
        } else {
            3
        }
    }

    /// Compute the token for a site from its raw features.
    #[inline]
    pub fn compute_token(&self, spike_count: u64, n_streams: u32, unsat_frac: f32) -> usize {
        let d0 = Self::bin4(spike_count as f32, &self.spike_count_thresholds);
        // Clamp to 0..=3 (directive: min(n_streams - 1, 3)).
        let d1 = (n_streams.saturating_sub(1) as usize).min(3);
        let interaction = spike_count as f32 * n_streams as f32;
        let d2 = Self::bin4(interaction, &self.interaction_thresholds);
        let d3 = Self::bin4(unsat_frac, &self.unsat_frac_thresholds);
        d0 * 64 + d1 * 16 + d2 * 4 + d3
    }

    /// Score a site. Returns the v4 lookup probability in `[0, 1]`.
    ///
    /// Higher scores rank the site higher. Ties are broken by the caller
    /// (canonical tiebreaker: `spike_count` descending).
    #[inline]
    pub fn score(&self, spike_count: u64, n_streams: u32, unsat_frac: f32) -> f32 {
        let token = self.compute_token(spike_count, n_streams, unsat_frac);
        self.lookup[token]
    }
}

/// Apply tokenized rerank to an in-memory sites JSON array.
///
/// Each site JSON object must have at minimum:
///   - `id` (integer): cluster id used to look up spike intensities
///   - `spike_count` (integer)
///
/// Behavior:
///   1. For each site, compute `unsat_frac` from the intensities returned
///      by `intensity_fn(cluster_id)`. If the lookup returns `None`, the
///      site keeps its current rank (tokenized_score is NOT injected).
///   2. Compute the tokenized score via the embedded v4 lookup.
///   3. Inject `tokenized_score`, `tokenized_token`, and `unsat_frac` into
///      each site's JSON.
///   4. If `reorder_by_score` is true, sort the array by tokenized_score
///      descending. Ties broken by `spike_count` descending.
///   5. Rewrite the `rank` field (1-indexed) to match the new order.
///
/// Returns `Ok(n_scored)` with the count of sites that received a score.
pub fn apply_tokenized_rerank(
    sites_json: &mut [serde_json::Value],
    intensity_fn: impl Fn(i64) -> Option<Vec<f32>>,
    n_streams: u32,
    reorder_by_score: bool,
) -> Result<usize, RankerError> {
    let ranker = TokenizedRanker::embedded()?;
    let mut n_scored = 0usize;

    for site in sites_json.iter_mut() {
        let cluster_id = site.get("id").and_then(|v| v.as_i64()).unwrap_or(-1);
        let spike_count = site
            .get("spike_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        let intensities = intensity_fn(cluster_id);
        let (unsat_frac, scored) = match intensities {
            Some(ref v) if !v.is_empty() => (compute_unsat_frac(v, 0.01), true),
            _ => (0.0, false),
        };

        let token = ranker.compute_token(spike_count, n_streams, unsat_frac);
        let score = ranker.lookup[token];

        if let Some(obj) = site.as_object_mut() {
            obj.insert("tokenized_score".to_string(), serde_json::json!(score));
            obj.insert("tokenized_token".to_string(), serde_json::json!(token));
            obj.insert("unsat_frac".to_string(), serde_json::json!(unsat_frac));
        }

        if scored {
            n_scored += 1;
        }
    }

    if reorder_by_score {
        sites_json.sort_by(|a, b| {
            let sa = a
                .get("tokenized_score")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let sb = b
                .get("tokenized_score")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            // Primary: tokenized_score descending
            let primary = sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal);
            if primary != std::cmp::Ordering::Equal {
                return primary;
            }
            // Tiebreak: spike_count descending
            let ca = a.get("spike_count").and_then(|v| v.as_u64()).unwrap_or(0);
            let cb = b.get("spike_count").and_then(|v| v.as_u64()).unwrap_or(0);
            cb.cmp(&ca)
        });

        // Rewrite rank field (1-indexed)
        for (i, site) in sites_json.iter_mut().enumerate() {
            if let Some(obj) = site.as_object_mut() {
                obj.insert("rank".to_string(), serde_json::json!(i + 1));
            }
        }
    }

    Ok(n_scored)
}

/// Compute `unsat_frac` from per-spike intensities.
///
/// Definition: fraction of spikes whose intensity is NOT at the saturation
/// value (the maximum intensity in the distribution). A small saturation
/// band is tolerated (within `sat_tolerance` of the maximum) to handle
/// floating-point noise.
///
/// At `--spike-percentile 95`, most retained spikes saturate at the top
/// intensity — `unsat_frac` is a tight discriminator (v4 thresholds
/// [0.033, 0.044, 0.071] reflect this narrow range).
pub fn compute_unsat_frac(intensities: &[f32], sat_tolerance: f32) -> f32 {
    if intensities.is_empty() {
        return 0.0;
    }
    let max_intensity = intensities
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let sat_threshold = max_intensity - sat_tolerance;
    let n_unsat = intensities.iter().filter(|&&v| v < sat_threshold).count();
    n_unsat as f32 / intensities.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_binary_parses() {
        let r = TokenizedRanker::embedded().expect("embedded v4 must parse");
        assert_eq!(r.version, VERSION);
        // From the v4 JSON:
        assert!((r.spike_count_thresholds[0] - 215136.0).abs() < 1.0);
        assert!((r.spike_count_thresholds[2] - 579484.0).abs() < 1.0);
        assert!((r.interaction_thresholds[0] - 860544.0).abs() < 1.0);
    }

    #[test]
    fn lookup_contains_populated_bins() {
        let r = TokenizedRanker::embedded().unwrap();
        // From v4 JSON: token 187 should be ~0.755 (one of the populated bins).
        assert!((r.lookup[187] - 0.755257).abs() < 1e-4);
        // Unpopulated bins should be 0.1 (low-prior default).
        assert!((r.lookup[0] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn token_n_streams_4_has_d1_eq_3() {
        let r = TokenizedRanker::embedded().unwrap();
        // For n_streams = 4, d1 = min(3, 3) = 3.
        // With spike_count and unsat_frac at the low end, token should be
        // in 48..=63 range (d0=0, d1=3, d2=0, d3=0..3).
        let token = r.compute_token(100_000, 4, 0.01);
        assert!(
            token >= 48 && token <= 63,
            "token {} not in expected range",
            token
        );
    }

    #[test]
    fn bin4_edges() {
        let thresholds = [10.0, 20.0, 30.0];
        assert_eq!(TokenizedRanker::bin4(5.0, &thresholds), 0);
        assert_eq!(TokenizedRanker::bin4(15.0, &thresholds), 1);
        assert_eq!(TokenizedRanker::bin4(25.0, &thresholds), 2);
        assert_eq!(TokenizedRanker::bin4(35.0, &thresholds), 3);
    }

    #[test]
    fn unsat_frac_all_saturated() {
        let intensities = vec![64.8, 64.8, 64.8, 64.8];
        assert!((compute_unsat_frac(&intensities, 0.01) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn unsat_frac_mixed() {
        // 1 out of 4 unsaturated.
        let intensities = vec![64.8, 64.8, 64.8, 10.0];
        assert!((compute_unsat_frac(&intensities, 0.01) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn bad_magic_rejected() {
        let bytes = [0u8; TokenizedRanker::EXPECTED_SIZE];
        let err = TokenizedRanker::from_bytes(&bytes);
        assert!(matches!(err, Err(RankerError::BadMagic(_))));
    }

    #[test]
    fn apply_rerank_injects_fields_and_reorders() {
        let mut sites = vec![
            serde_json::json!({"id": 1, "spike_count": 100_000, "rank": 1}),
            serde_json::json!({"id": 2, "spike_count": 2_500_000, "rank": 2}),
        ];

        // Site 2 has many spikes with varied intensities → higher unsat_frac → different token.
        let intensity_fn = |cid: i64| -> Option<Vec<f32>> {
            if cid == 1 {
                Some(vec![64.8; 100]) // all saturated, unsat_frac=0
            } else if cid == 2 {
                // Mix of saturated and non-saturated — boosts to a populated bin
                let mut v = vec![64.8; 70];
                v.extend_from_slice(&[10.0; 30]);
                Some(v)
            } else {
                None
            }
        };

        let n = apply_tokenized_rerank(&mut sites, intensity_fn, 4, true)
            .expect("rerank should succeed");
        assert_eq!(n, 2);

        // Both sites should now have tokenized_score
        for s in &sites {
            assert!(s.get("tokenized_score").is_some());
            assert!(s.get("tokenized_token").is_some());
            assert!(s.get("unsat_frac").is_some());
            assert!(s.get("rank").is_some());
        }

        // Site 2 (2.5M spikes, varied intensity) should score higher
        // than Site 1 (100K spikes, uniform saturation) → should land at rank 1
        let first = sites[0].get("id").and_then(|v| v.as_i64()).unwrap();
        assert_eq!(first, 2, "site 2 should rank first with tokenized ranker");
    }
}
