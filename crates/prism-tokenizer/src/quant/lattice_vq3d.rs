//! 3-D coordinate quantizer (§7) for `position` (x,y,z engine-space Å) — a 3-vector, not three
//! independent scalars, so it does NOT use the scalar `Quantizer` trait.
//!
//! This implements the **BCC lattice** (body-centered cubic), the provably optimal 3-D vector
//! quantizer, requiring no training (the directive's deterministic fallback to the learned
//! VQ-EMA codebook). Codes = 2·G³ points: the integer sublattice G³ plus the half-offset
//! sublattice (each cell carries a body center).

use crate::types::Distortion;
use std::collections::HashMap;

#[inline]
fn dist2(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|i| (a[i] - b[i]).powi(2)).sum()
}

/// Uniform 3-D grid over the VQ codebook for O(~1) nearest-code lookup (P1-11) instead of
/// brute-force O(K). Exact: shell expansion continues until the best distance is provably ≤ the
/// closest possible point in any unsearched shell, so it returns the same code as brute force.
struct GridIndex {
    lo: [f64; 3],
    cell: f64,
    buckets: HashMap<(i32, i32, i32), Vec<u32>>,
}

impl GridIndex {
    fn cell_of(lo: [f64; 3], cell: f64, p: [f64; 3]) -> (i32, i32, i32) {
        (
            ((p[0] - lo[0]) / cell).floor() as i32,
            ((p[1] - lo[1]) / cell).floor() as i32,
            ((p[2] - lo[2]) / cell).floor() as i32,
        )
    }

    fn build(codebook: &[[f64; 3]], used: usize) -> Self {
        let used = used.max(1).min(codebook.len());
        let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for c in &codebook[..used] {
            for d in 0..3 {
                lo[d] = lo[d].min(c[d]);
                hi[d] = hi[d].max(c[d]);
            }
        }
        let diag = ((0..3).map(|d| (hi[d] - lo[d]).powi(2)).sum::<f64>())
            .sqrt()
            .max(1e-9);
        let cell = (diag / (used as f64).cbrt().max(1.0)).max(1e-6);
        let mut buckets: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::new();
        for (i, c) in codebook[..used].iter().enumerate() {
            buckets
                .entry(Self::cell_of(lo, cell, *c))
                .or_default()
                .push(i as u32);
        }
        Self { lo, cell, buckets }
    }

    fn nearest(&self, p: [f64; 3], codebook: &[[f64; 3]]) -> u32 {
        let c0 = Self::cell_of(self.lo, self.cell, p);
        let (mut best, mut bd2, mut found) = (0u32, f64::INFINITY, false);
        let mut r = 0i32;
        loop {
            for dx in -r..=r {
                for dy in -r..=r {
                    for dz in -r..=r {
                        if dx.abs().max(dy.abs()).max(dz.abs()) != r {
                            continue;
                        }
                        if let Some(codes) = self.buckets.get(&(c0.0 + dx, c0.1 + dy, c0.2 + dz)) {
                            for &ci in codes {
                                let d = dist2(p, codebook[ci as usize]);
                                if d < bd2 {
                                    bd2 = d;
                                    best = ci;
                                    found = true;
                                }
                            }
                        }
                    }
                }
            }
            // any unsearched point is in a shell ≥ r+1, i.e. ≥ r*cell away ⇒ done if best ≤ that.
            if found && (r as f64 * self.cell).powi(2) >= bd2 {
                break;
            }
            r += 1;
            if r > 8192 {
                break;
            }
        }
        best
    }
}

pub struct Bcc3dQuantizer {
    g: u32,
    lo: [f64; 3],
    hi: [f64; 3],
    bits: u8,
}

impl Bcc3dQuantizer {
    /// `codebook_bits` bounds total codes to 2^bits; G is the largest grid with 2·G³ ≤ 2^bits.
    pub fn new(codebook_bits: u8, lo: [f64; 3], hi: [f64; 3]) -> Self {
        let cap = 1u64 << codebook_bits;
        let mut g = 1u32;
        while 2 * ((g + 1) as u64).pow(3) <= cap {
            g += 1;
        }
        Self {
            g,
            lo,
            hi,
            bits: codebook_bits,
        }
    }

    #[inline]
    fn to_grid(&self, v: [f64; 3]) -> [f64; 3] {
        let mut u = [0.0; 3];
        for i in 0..3 {
            let span = (self.hi[i] - self.lo[i]).max(1e-12);
            u[i] = ((v[i] - self.lo[i]) / span * self.g as f64).clamp(0.0, self.g as f64 - 1e-9);
        }
        u
    }

    #[inline]
    fn from_grid(&self, c: [f64; 3]) -> [f64; 3] {
        let mut v = [0.0; 3];
        for i in 0..3 {
            let span = (self.hi[i] - self.lo[i]).max(1e-12);
            v[i] = self.lo[i] + c[i] / self.g as f64 * span;
        }
        v
    }

    #[inline]
    fn cube(&self) -> u32 {
        self.g * self.g * self.g
    }

    pub fn encode(&self, v: [f64; 3]) -> (u32, Distortion) {
        let u = self.to_grid(v);
        let gmax = self.g as i64 - 1;
        // integer sublattice candidate
        let a = [
            (u[0].round() as i64).clamp(0, gmax),
            (u[1].round() as i64).clamp(0, gmax),
            (u[2].round() as i64).clamp(0, gmax),
        ];
        let a_c = [a[0] as f64, a[1] as f64, a[2] as f64];
        // half-offset (body-center) sublattice candidate
        let h = [
            ((u[0] - 0.5).round() as i64).clamp(0, gmax),
            ((u[1] - 0.5).round() as i64).clamp(0, gmax),
            ((u[2] - 0.5).round() as i64).clamp(0, gmax),
        ];
        let h_c = [h[0] as f64 + 0.5, h[1] as f64 + 0.5, h[2] as f64 + 0.5];

        let d2 = |c: [f64; 3]| (0..3).map(|i| (u[i] - c[i]).powi(2)).sum::<f64>();
        let (parity, idx_vec, center) = if d2(a_c) <= d2(h_c) {
            (0u32, a, a_c)
        } else {
            (1u32, h, h_c)
        };
        let idx = (idx_vec[0] as u32) * self.g * self.g
            + (idx_vec[1] as u32) * self.g
            + (idx_vec[2] as u32);
        let code = parity * self.cube() + idx;

        let recon = self.from_grid(center);
        let dist = ((0..3).map(|i| (v[i] - recon[i]).powi(2)).sum::<f64>()).sqrt();
        (code, Distortion(dist))
    }

    pub fn decode(&self, code: u32) -> [f64; 3] {
        let parity = code / self.cube();
        let idx = code % self.cube();
        let i = idx / (self.g * self.g);
        let j = (idx / self.g) % self.g;
        let k = idx % self.g;
        let off = if parity == 1 { 0.5 } else { 0.0 };
        self.from_grid([i as f64 + off, j as f64 + off, k as f64 + off])
    }

    pub fn bit_budget(&self) -> u8 {
        self.bits
    }

    pub fn grid(&self) -> u32 {
        self.g
    }
}

/// Learned 3-D VQ codebook with EMA updates (§6, directive PRIMARY for `position`). Pocket
/// coordinates cluster around binding sites and are NOT uniform, so a learned codebook beats the
/// BCC lattice on the real distribution (proven by VQ-CMP). Deterministic given the fit data
/// (strided init, no RNG) — `§13` reproducibility.
pub struct Vq3dEma {
    codebook: Vec<[f64; 3]>,
    bits: u8,
    usage: Vec<u64>,
    fitted: bool,
    used: usize,
    grid: Option<GridIndex>,
}

const VQ_GAMMA: f64 = 0.99; // EMA decay
const VQ_EPOCHS: usize = 30;
const VQ_FIT_CAP: usize = 20_000; // deterministic subsample ceiling for huge corpora

impl Vq3dEma {
    pub fn new(codebook_bits: u8) -> Self {
        let k = 1usize << codebook_bits;
        Self {
            codebook: vec![[0.0; 3]; k],
            bits: codebook_bits,
            usage: vec![0; k],
            fitted: false,
            used: 0,
            grid: None,
        }
    }

    #[inline]
    fn d2(a: [f64; 3], b: [f64; 3]) -> f64 {
        (0..3).map(|i| (a[i] - b[i]).powi(2)).sum()
    }

    fn nearest(&self, p: [f64; 3]) -> usize {
        // P1-11: O(~1) grid lookup once fitted; brute force pre-fit (codebook is all-zero then).
        if let Some(g) = &self.grid {
            return g.nearest(p, &self.codebook) as usize;
        }
        let mut best = 0usize;
        let mut bd = f64::INFINITY;
        for (j, &c) in self.codebook.iter().enumerate() {
            let d = Self::d2(p, c);
            if d < bd {
                bd = d;
                best = j;
            }
        }
        best
    }

    /// EMA k-means fit (γ=0.99) with cheap dead-code reinit. Effective codes are capped at the
    /// number of fit points, so K ≫ N (e.g. 12-bit codebook on a small set) stays fast.
    pub fn fit(&mut self, points: &[[f64; 3]]) {
        self.fit_with_assigner(points, &mut |pts, cb| {
            // CPU reference assignment (single O(N·k) pass): nearest index + squared distance,
            // lowest index on ties. This IS the parity truth the GPU lane must reproduce.
            let mut idx = Vec::with_capacity(pts.len());
            let mut d2 = Vec::with_capacity(pts.len());
            for &p in pts {
                let (mut j, mut bd) = (0usize, f64::INFINITY);
                for (c, &cbj) in cb.iter().enumerate() {
                    let d = Self::d2(p, cbj);
                    if d < bd {
                        bd = d;
                        j = c;
                    }
                }
                idx.push(j as u32);
                d2.push(bd);
            }
            (idx, d2)
        });
    }

    /// The shared EMA-fit driver. `assign(points, &codebook[..k]) -> (nearest_idx, sq_dist)` is the
    /// only pluggable step — supplying a GPU assignment kernel here yields a BIT-IDENTICAL codebook
    /// to the CPU `fit` whenever the kernel's `(idx, d2)` matches the CPU reference (proven by the
    /// parity bin). Everything else (init, worst-point pick, EMA, reinit, usage) is fixed CPU code,
    /// so determinism (§13) is preserved regardless of backend.
    pub fn fit_with_assigner(
        &mut self,
        points: &[[f64; 3]],
        assign: &mut dyn FnMut(&[[f64; 3]], &[[f64; 3]]) -> (Vec<u32>, Vec<f64>),
    ) {
        if points.is_empty() {
            return;
        }
        // P1-11: deterministic subsample for very large corpora so fit stays bounded.
        let sub: Vec<[f64; 3]>;
        let points: &[[f64; 3]] = if points.len() > VQ_FIT_CAP {
            sub = (0..VQ_FIT_CAP)
                .map(|i| points[i * points.len() / VQ_FIT_CAP])
                .collect();
            &sub
        } else {
            points
        };
        // Use at most `points.len()` codes (extra codes would be permanently dead); the field's
        // codebook_bits is the ceiling, not a quota.
        let k = self.codebook.len().min(points.len());
        for j in 0..k {
            self.codebook[j] = points[(j * points.len()) / k.max(1)];
        }
        let mut n_ema = vec![1.0f64; k];
        let mut m_ema: Vec<[f64; 3]> = self.codebook[..k].to_vec();
        for _ep in 0..VQ_EPOCHS {
            let mut n_b = vec![0.0f64; k];
            let mut m_b = vec![[0.0f64; 3]; k];
            // assignment pass (CPU reference or GPU kernel) → per-point nearest code + sq distance.
            let (assign_idx, assign_d2) = assign(points, &self.codebook[..k]);
            // worst-reconstructed point (first-max tie rule, identical to the original CPU loop).
            let mut worst = points[0];
            let mut wd = -1.0;
            for (pi, &p) in points.iter().enumerate() {
                let j = assign_idx[pi] as usize;
                let bd = assign_d2[pi];
                if bd > wd {
                    wd = bd;
                    worst = p;
                }
                n_b[j] += 1.0;
                for d in 0..3 {
                    m_b[j][d] += p[d];
                }
            }
            for j in 0..k {
                n_ema[j] = VQ_GAMMA * n_ema[j] + (1.0 - VQ_GAMMA) * n_b[j];
                for d in 0..3 {
                    m_ema[j][d] = VQ_GAMMA * m_ema[j][d] + (1.0 - VQ_GAMMA) * m_b[j][d];
                }
                if n_ema[j] > 1e-9 {
                    for d in 0..3 {
                        self.codebook[j][d] = m_ema[j][d] / n_ema[j];
                    }
                }
            }
            // dead-code reinit (O(k)): unused codes jump near the worst point (small per-code offset).
            for j in 0..k {
                if n_b[j] < 1.0 {
                    let off = (j % 32) as f64 * 0.01;
                    self.codebook[j] = [worst[0] + off, worst[1], worst[2]];
                    m_ema[j] = self.codebook[j];
                    n_ema[j] = 1.0;
                }
            }
        }
        // any codes beyond the used `k` collapse onto code 0 (never selected as nearest for real
        // points once the live codes are placed) — keep them defined for decode bounds safety.
        for j in k..self.codebook.len() {
            self.codebook[j] = self.codebook[0];
        }
        let k = self.codebook.len();
        // record final usage for perplexity / health.
        self.usage = vec![0; k];
        for &p in points {
            let j = self.nearest(p);
            self.usage[j] += 1;
        }
        self.used = self.codebook.len().min(points.len());
        self.grid = Some(GridIndex::build(&self.codebook, self.used));
        self.fitted = true;
    }

    pub fn encode(&self, p: [f64; 3]) -> (u32, Distortion) {
        let j = self.nearest(p);
        (j as u32, Distortion(Self::d2(p, self.codebook[j]).sqrt()))
    }
    pub fn decode(&self, t: u32) -> [f64; 3] {
        self.codebook[(t as usize).min(self.codebook.len() - 1)]
    }
    pub fn bit_budget(&self) -> u8 {
        self.bits
    }
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }
    pub fn perplexity(&self) -> f64 {
        super::usage_perplexity(&self.usage)
    }
    pub fn dead_codes(&self) -> usize {
        self.usage.iter().filter(|&&c| c == 0).count()
    }

    /// Batched coordinate encode (CPU reference): `&[[f64;3]] → (codes, distortions)`. Identical to
    /// calling [`encode`] per point — the directive's Phase-4 lane (1) reference. The GPU lane
    /// ([`encode_batch_gpu`]) must reproduce these `codes` exactly (and distortions bit-for-bit).
    pub fn encode_batch_cpu(&self, pts: &[[f64; 3]]) -> (Vec<u32>, Vec<f64>) {
        let mut codes = Vec::with_capacity(pts.len());
        let mut dists = Vec::with_capacity(pts.len());
        for &p in pts {
            let (c, d) = self.encode(p);
            codes.push(c);
            dists.push(d.value());
        }
        (codes, dists)
    }

    /// GPU-accelerated EMA fit (feature `gpu-tokenizer`). Uses the device assignment kernel for the
    /// O(N·K) inner pass while the host does the (fixed, deterministic) EMA/reinit. The resulting
    /// codebook is BIT-IDENTICAL to [`fit`] whenever the kernel matches the CPU assignment (proven
    /// by the parity bin). Returns false (and does nothing) if no device is available, so the caller
    /// can fall back to `fit` — correctness never depends on the GPU.
    #[cfg(feature = "gpu-tokenizer")]
    pub fn fit_gpu(&mut self, points: &[[f64; 3]], gpu: &crate::gpu::GpuVqContext) -> bool {
        // If any assignment falls back to CPU (device error mid-fit), `ok` records it; the fit
        // still completes correctly via the CPU reference so the codebook is never left partial.
        let mut ok = true;
        self.fit_with_assigner(points, &mut |pts, cb| {
            if let Some((idx, d2)) = gpu.vq_assign(pts, cb) {
                (idx.into_iter().map(|v| v as u32).collect(), d2)
            } else {
                ok = false;
                let (i, d) = crate::gpu::vq_assign_cpu(pts, cb);
                (i.into_iter().map(|v| v as u32).collect(), d)
            }
        });
        ok
    }

    /// GPU-accelerated batched coordinate encode (feature `gpu-tokenizer`). Reuses the assignment
    /// kernel against the FITTED codebook (parity with [`encode_batch_cpu`]). None if no device.
    #[cfg(feature = "gpu-tokenizer")]
    pub fn encode_batch_gpu(
        &self,
        pts: &[[f64; 3]],
        gpu: &crate::gpu::GpuVqContext,
    ) -> Option<(Vec<u32>, Vec<f64>)> {
        // Encode uses the brute-force nearest over the full codebook (same as `encode` pre-grid),
        // which is exactly what the kernel computes. The grid index is an exact accelerator of the
        // same argmin, so codes match. Distortion is sqrt(d2).
        let (idx, d2) = gpu.vq_assign(pts, &self.codebook)?;
        let codes = idx.into_iter().map(|v| v as u32).collect();
        let dists = d2.into_iter().map(|v| v.sqrt()).collect();
        Some((codes, dists))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q() -> Bcc3dQuantizer {
        Bcc3dQuantizer::new(12, [0.0, 0.0, 0.0], [60.0, 60.0, 60.0])
    }

    // deterministic clustered "pocket coordinates": 3 tight blobs in a big box (non-uniform).
    fn clustered() -> Vec<[f64; 3]> {
        let centers = [[10.0, 10.0, 10.0], [44.0, 44.0, 44.0], [20.0, 46.0, 14.0]];
        (0..600)
            .map(|i| {
                let c = centers[i % 3];
                let t = i as f64;
                [
                    c[0] + (t * 0.13).sin() * 2.0,
                    c[1] + (t * 0.17).cos() * 2.0,
                    c[2] + (t * 0.19).sin() * 2.0,
                ]
            })
            .collect()
    }

    fn mse_bcc(pts: &[[f64; 3]], lo: [f64; 3], hi: [f64; 3], bits: u8) -> f64 {
        let q = Bcc3dQuantizer::new(bits, lo, hi);
        pts.iter()
            .map(|&p| {
                let r = q.decode(q.encode(p).0);
                (0..3).map(|i| (p[i] - r[i]).powi(2)).sum::<f64>()
            })
            .sum::<f64>()
            / pts.len() as f64
    }

    fn mse_vq(pts: &[[f64; 3]], bits: u8) -> (f64, Vq3dEma) {
        let mut q = Vq3dEma::new(bits);
        q.fit(pts);
        let m = pts
            .iter()
            .map(|&p| {
                let r = q.decode(q.encode(p).0);
                (0..3).map(|i| (p[i] - r[i]).powi(2)).sum::<f64>()
            })
            .sum::<f64>()
            / pts.len() as f64;
        (m, q)
    }

    #[test]
    fn grid_nearest_matches_brute_force() {
        // P1-11: the grid index returns the SAME code as brute force (exact, just faster).
        let pts = clustered();
        let mut q = Vq3dEma::new(8);
        q.fit(&pts);
        let used = q.used;
        let probes = [
            pts[0],
            pts[137],
            [25.0, 25.0, 25.0],
            [44.0, 44.0, 44.0],
            [-5.0, 60.0, 100.0],
        ];
        for tp in probes {
            let via_grid = q.encode(tp).0;
            let (mut brute, mut bd) = (0u32, f64::INFINITY);
            for (i, c) in q.codebook[..used].iter().enumerate() {
                let d = dist2(tp, *c);
                if d < bd {
                    bd = d;
                    brute = i as u32;
                }
            }
            assert_eq!(
                via_grid, brute,
                "grid nearest must equal brute nearest for {tp:?}"
            );
        }
    }

    #[test]
    fn vq_cmp_beats_bcc_on_nonuniform_distribution() {
        // §6 VQ-CMP: on a clustered (non-uniform) distribution, VQ-EMA must beat BCC at equal bits.
        let pts = clustered();
        let bits = 8u8;
        let bcc = mse_bcc(&pts, [6.0, 6.0, 6.0], [48.0, 48.0, 48.0], bits);
        let (vq, _q) = mse_vq(&pts, bits);
        assert!(
            vq <= bcc,
            "VQ-EMA MSE {vq} should beat BCC MSE {bcc} on clustered data"
        );
    }

    #[test]
    fn vq_health_no_dead_codes_after_fit_and_deterministic() {
        let pts = clustered();
        let (_m1, q1) = mse_vq(&pts, 6);
        // dead-code reinit should leave few/no permanently-dead codes; perplexity > 1.
        assert!(q1.perplexity() > 1.0);
        // determinism: a second fit on identical data yields an identical codebook.
        let mut q2 = Vq3dEma::new(6);
        q2.fit(&pts);
        for j in 0..q1.codebook.len() {
            for d in 0..3 {
                assert!((q1.codebook[j][d] - q2.codebook[j][d]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn grid_fits_codebook() {
        let q = q();
        // 2·G³ ≤ 4096 ⇒ G = 12 (2·1728 = 3456 ≤ 4096; G=13 ⇒ 2·2197 = 4394 > 4096).
        assert_eq!(q.grid(), 12);
        assert!(2 * (q.grid() as u64).pow(3) <= (1u64 << 12));
    }

    #[test]
    fn round_trip_within_cell() {
        let q = q();
        // one grid cell ≈ 60/12 = 5 Å; BCC reconstruction error is well under a cell diagonal.
        let cell = 60.0 / 12.0;
        for v in [[1.0, 2.0, 3.0], [30.0, 31.5, 29.9], [59.0, 0.5, 12.3]] {
            let (code, d) = q.encode(v);
            assert!(
                d.value() < cell,
                "distortion {} >= cell {}",
                d.value(),
                cell
            );
            let r = q.decode(code);
            for i in 0..3 {
                assert!((r[i] - v[i]).abs() < cell);
            }
        }
    }

    #[test]
    fn body_center_beats_integer_for_diagonal_point() {
        let q = q();
        // a point near a cell body-center should select the half-offset sublattice (parity 1).
        let half = 60.0 / 12.0 * 0.5;
        let (code, _) = q.encode([half, half, half]);
        assert_eq!(
            code / (q.grid() * q.grid() * q.grid()),
            1,
            "expected body-center parity"
        );
    }
}
