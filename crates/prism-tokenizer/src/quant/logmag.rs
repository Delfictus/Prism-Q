//! Log-magnitude quantizer for multi-decade fields (§7): intensity (a.u.), adaptive_dt, rhat,
//! ess. Bins uniformly in log space over `[ln(floor), ln(ceil)]`. Values ≤0 / sentinels are
//! routed out by the caller; internally we clamp into range.

use super::usage_perplexity;
use crate::types::{Distortion, Quantizer};

pub struct LogMagnitudeQuantizer {
    bits: u8,
    k: u32,
    lo: f64,
    hi: f64,
    llo: f64,
    lhi: f64,
    usage: Vec<u64>,
}

impl LogMagnitudeQuantizer {
    pub fn new(bits: u8, floor: f64, ceil: f64) -> Self {
        let lo = floor.max(1e-30);
        let hi = ceil.max(lo * (1.0 + 1e-9));
        let k = 1u32 << bits;
        Self {
            bits,
            k,
            lo,
            hi,
            llo: lo.ln(),
            lhi: hi.ln(),
            usage: vec![0; k as usize],
        }
    }

    #[inline]
    fn bin_of(&self, v: f64) -> u32 {
        let v = v.clamp(self.lo, self.hi);
        let frac = (v.ln() - self.llo) / (self.lhi - self.llo);
        ((frac * self.k as f64).floor() as i64).clamp(0, self.k as i64 - 1) as u32
    }

    #[inline]
    fn center(&self, t: u32) -> f64 {
        let t = t.min(self.k - 1);
        (self.llo + (t as f64 + 0.5) / self.k as f64 * (self.lhi - self.llo)).exp()
    }
}

impl Quantizer for LogMagnitudeQuantizer {
    fn fit(&mut self, samples: &[f64]) {
        // P2-12: refit the decade range to the observed data so resolution concentrates where the
        // values actually are (e.g. intensity ~1.3, not the guessy default 1e-4..1e4).
        let pos: Vec<f64> = samples.iter().copied().filter(|&x| x > 0.0).collect();
        if !pos.is_empty() {
            let mn = pos.iter().cloned().fold(f64::INFINITY, f64::min);
            let mx = pos.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            self.lo = (mn * 0.5).max(1e-30); // half-decade margin below the min
            self.hi = (mx * 2.0).max(self.lo * (1.0 + 1e-9));
            self.llo = self.lo.ln();
            self.lhi = self.hi.ln();
        }
        self.usage = vec![0; self.k as usize];
        for &s in samples {
            if s > 0.0 {
                let b = self.bin_of(s) as usize;
                self.usage[b] += 1;
            }
        }
    }
    fn encode(&self, v: f64) -> (u32, Distortion) {
        let t = self.bin_of(v);
        (
            t,
            Distortion((v.clamp(self.lo, self.hi) - self.center(t)).abs()),
        )
    }
    fn decode(&self, t: u32) -> f64 {
        self.center(t)
    }
    fn bit_budget(&self) -> u8 {
        self.bits
    }
    fn perplexity(&self) -> f64 {
        usage_perplexity(&self.usage)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_decades_monotonic() {
        let q = LogMagnitudeQuantizer::new(10, 1e-3, 1e3);
        // monotonic: larger value → larger-or-equal bin.
        assert!(q.bin_of(0.01) < q.bin_of(100.0));
    }

    #[test]
    fn corpus_fit_tightens_range_and_lowers_distortion() {
        // P2-12: fitting on clustered samples narrows the decade range → lower MEAN distortion
        // across the data (single-point distortion is bin-edge-dependent, so compare the mean).
        let mut q = LogMagnitudeQuantizer::new(10, 1e-4, 1e4);
        let samples: Vec<f64> = (0..200)
            .map(|i| 1.0 + (i as f64 * 0.013).sin().abs())
            .collect();
        let mean = |q: &LogMagnitudeQuantizer, s: &[f64]| -> f64 {
            s.iter().map(|&v| q.encode(v).1.value()).sum::<f64>() / s.len() as f64
        };
        let before = mean(&q, &samples);
        q.fit(&samples);
        let after = mean(&q, &samples);
        assert!(
            after < before,
            "corpus-fit should lower mean distortion: {after} !< {before}"
        );
    }

    #[test]
    fn round_trip_relative_error_bounded() {
        let q = LogMagnitudeQuantizer::new(12, 1e-3, 1e3);
        for &v in &[1e-2, 1.0, 5.0, 250.0] {
            let (t, _d) = q.encode(v);
            let rel = (q.decode(t) - v).abs() / v;
            assert!(rel < 0.01, "relative error {rel} too high for v={v}");
        }
    }
}
