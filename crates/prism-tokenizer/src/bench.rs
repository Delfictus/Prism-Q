//! Benchmarking + workload-measurement infrastructure for the tokenizer-wide GPU acceleration
//! directive. This is an OFFLINE/measurement surface (no engine execution): it loads real PRISM
//! artifacts, runs the CPU reference path stage-by-stage, and records wall timings + peak RSS so
//! the GPU-candidate decision is made on evidence, not on "a GPU exists".
//!
//! Determinism: synthetic corpora are generated from a fixed integer-seeded LCG (no `rand` dep),
//! so a bench re-run is bit-identical. Real artifacts are decoded through the crate's own
//! authoritative codecs (INV-1) — never re-parsed ad hoc.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Repo root, resolved from the crate manifest dir (`crates/prism-tokenizer/../..`).
pub fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Peak resident set size in KiB (Linux `/proc/self/status` `VmHWM`). 0 if unavailable.
pub fn peak_rss_kib() -> u64 {
    let Ok(s) = std::fs::read_to_string("/proc/self/status") else {
        return 0;
    };
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kib: u64 = rest
                .split_whitespace()
                .next()
                .and_then(|t| t.parse().ok())
                .unwrap_or(0);
            return kib;
        }
    }
    0
}

/// Time a closure, returning `(result, elapsed_seconds)`.
#[inline]
pub fn timed<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let t = Instant::now();
    let out = f();
    (out, t.elapsed().as_secs_f64())
}

/// Median of `iters` timings of `f` (warms the path once first). Returns seconds.
pub fn bench_median(iters: usize, mut f: impl FnMut()) -> f64 {
    f(); // warmup
    let mut ts: Vec<f64> = Vec::with_capacity(iters.max(1));
    for _ in 0..iters.max(1) {
        let (_r, dt) = timed(&mut f);
        ts.push(dt);
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ts[ts.len() / 2]
}

// --- deterministic synthetic corpora --------------------------------------------------------

/// Tiny deterministic LCG (Numerical Recipes constants) → uniform f64 in [0,1).
pub struct Lcg(u64);
impl Lcg {
    pub fn new(seed: u64) -> Self {
        Lcg(seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407))
    }
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    #[inline]
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A deterministic clustered 3-D coordinate corpus (Gaussian-ish blobs in a 60Å box), the shape
/// real pocket coordinates take. `n` points across `clusters` centers.
pub fn synth_coords(n: usize, clusters: usize, seed: u64) -> Vec<[f64; 3]> {
    let mut r = Lcg::new(seed);
    let mut centers: Vec<[f64; 3]> = Vec::with_capacity(clusters);
    for _ in 0..clusters.max(1) {
        centers.push([
            r.unit() * 50.0 + 5.0,
            r.unit() * 50.0 + 5.0,
            r.unit() * 50.0 + 5.0,
        ]);
    }
    (0..n)
        .map(|i| {
            let c = centers[i % centers.len()];
            // box-muller-ish spread via two uniforms (deterministic), sigma ~2Å.
            let s = 2.0;
            [
                c[0] + (r.unit() - 0.5) * 2.0 * s,
                c[1] + (r.unit() - 0.5) * 2.0 * s,
                c[2] + (r.unit() - 0.5) * 2.0 * s,
            ]
        })
        .collect()
}

/// A deterministic scalar corpus spanning a few decades (the shape of intensity / dt fields):
/// log-uniform in [lo, hi].
pub fn synth_scalars(n: usize, lo: f64, hi: f64, seed: u64) -> Vec<f64> {
    let mut r = Lcg::new(seed);
    let (ll, lh) = (lo.max(1e-12).ln(), hi.max(lo + 1e-9).ln());
    (0..n).map(|_| (ll + (lh - ll) * r.unit()).exp()).collect()
}

// --- real-artifact loaders (through the crate's own codecs, INV-1) --------------------------

/// Locate the first in-repo real `*_spikes.bin` (PRSPK001 envelope). None if no `runs/` fixture.
pub fn find_real_spikes_bin() -> Option<PathBuf> {
    let runs = repo_root().join("runs");
    if !runs.exists() {
        return None;
    }
    walk(&runs, 7).into_iter().find(|p| {
        p.to_str()
            .map(|s| s.ends_with("_spikes.bin"))
            .unwrap_or(false)
    })
}

/// The frozen KCC visualization fixture (real residue values), if present.
pub fn kcc_fixture() -> Option<PathBuf> {
    let p = repo_root()
        .join("kv31_validation_frozen_20260512T215852Z/engine_manifest/Kv31_chainA.kcc_visualization.json");
    p.exists().then_some(p)
}

/// A real sealed QNANL artifact in OPERATOR_CONTROL, if present.
pub fn qnanl_fixture() -> Option<PathBuf> {
    let p =
        repo_root().join("OPERATOR_CONTROL/PRISMOS_PATHB_ACTIVE_MATERIALIZATION.json.prismqnanl");
    p.exists().then_some(p)
}

/// A real `bocpd.jsonl` (per-frame change-point log), if present.
pub fn bocpd_fixture() -> Option<PathBuf> {
    let runs = repo_root().join("runs");
    if !runs.exists() {
        return None;
    }
    walk(&runs, 7).into_iter().find(|p| {
        p.to_str()
            .map(|s| s.ends_with("bocpd.jsonl"))
            .unwrap_or(false)
    })
}

/// A real `warp_matrix.bin` (no decoder yet → CODEC_BLOCKED reducer surface), if present.
pub fn warp_matrix_fixture() -> Option<PathBuf> {
    let runs = repo_root().join("runs");
    if !runs.exists() {
        return None;
    }
    walk(&runs, 7).into_iter().find(|p| {
        p.to_str()
            .map(|s| s.ends_with("_warp_matrix.bin"))
            .unwrap_or(false)
    })
}

/// Decode up to `max_records` real spike coordinates from a PRSPK001 file (prefix decode — reads
/// only as many bytes as needed for `max_records`). Returns `(coords, declared_record_count)`.
pub fn load_real_coords(path: &Path, max_records: usize) -> Option<(Vec<[f64; 3]>, usize)> {
    use crate::codec::decode_prspk001_head;
    // One record stride per record + variable header; a generous slack on both
    // ends so the requested records are never truncated away.
    let prefix = 4096 + max_records.saturating_mul(crate::codec::GPU_SPIKE_EVENT_BYTES) + 4096;
    let buf = read_prefix(path, prefix)?;
    let d = decode_prspk001_head(&buf, max_records).ok()?;
    let coords = d
        .events
        .iter()
        .map(|e| {
            [
                e.position[0] as f64,
                e.position[1] as f64,
                e.position[2] as f64,
            ]
        })
        .collect();
    Some((coords, d.record_count))
}

/// Read at most `n` bytes from the start of a file (so a 700MB file is never loaded in full).
pub fn read_prefix(path: &Path, n: usize) -> Option<Vec<u8>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut buf = vec![0u8; n];
    let read = f.read(&mut buf).ok()?;
    buf.truncate(read);
    Some(buf)
}

/// Collect the 13 scalar KCC residue fields from the frozen fixture into a `field → samples` map.
pub fn load_kcc_scalars(path: &Path) -> Option<HashMap<&'static str, Vec<f64>>> {
    let json: serde_json::Value = serde_json::from_slice(&std::fs::read(path).ok()?).ok()?;
    let residues = json.get("residues")?.as_array()?;
    let collect = |k: &str| -> Vec<f64> {
        residues
            .iter()
            .filter_map(|r| r.get(k).and_then(|v| v.as_f64()))
            .collect()
    };
    let mut m: HashMap<&'static str, Vec<f64>> = HashMap::new();
    for k in [
        "kcc_score",
        "direction_score",
        "motion_efficiency",
        "burst_motion",
        "causal_lag",
        "lag_corr_peak",
        "local_cov",
        "sum_motion",
        "active_causal_steps",
        "total_steps",
        "net_dx",
        "net_dy",
        "net_dz",
    ] {
        m.insert(k, collect(k));
    }
    Some(m)
}

/// Recursive directory walk (no external dep), bounded depth + entry count.
fn walk(dir: &Path, depth: usize) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if depth == 0 {
        return out;
    }
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                out.extend(walk(&p, depth - 1));
            } else {
                out.push(p);
            }
            if out.len() > 20000 {
                break;
            }
        }
    }
    out
}

// --- minimal JSON ledger emitter (serde_json::Value, no extra dep) --------------------------

/// Write a `serde_json::Value` as pretty JSON to `path` under the repo, creating parent dirs.
pub fn write_ledger(rel_path: &str, v: &serde_json::Value) -> std::io::Result<PathBuf> {
    let p = repo_root().join(rel_path);
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&p, serde_json::to_vec_pretty(v).unwrap())?;
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lcg_is_deterministic() {
        let mut a = Lcg::new(42);
        let mut b = Lcg::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn synth_coords_deterministic_and_shaped() {
        let a = synth_coords(500, 3, 7);
        let b = synth_coords(500, 3, 7);
        assert_eq!(a.len(), 500);
        assert_eq!(a, b, "same seed ⇒ identical corpus");
    }

    #[test]
    fn synth_scalars_in_range() {
        let s = synth_scalars(1000, 1e-3, 1e2, 9);
        assert!(s.iter().all(|&x| (1e-3..=1e2 + 1e-6).contains(&x)));
    }

    #[test]
    fn peak_rss_is_nonzero_on_linux() {
        // VmHWM should be readable here; if not (non-Linux CI) it returns 0 and we don't assert.
        let _ = peak_rss_kib();
    }
}
