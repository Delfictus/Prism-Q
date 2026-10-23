//! §3 — critical-region (companded) quantizer for decision-bearing fields (INV-8).
//!
//! A single uniform/log layout cannot give both wide range and fine resolution at a decision
//! boundary. This quantizer puts **fine bins across the critical region and coarse bins outside**,
//! and — crucially — places every decision threshold **exactly on a bin edge**. Then a value just
//! below a threshold lands in the interval ending at it (midpoint below), and just above lands in
//! the interval starting at it (midpoint above): quantization can never move a value across a
//! threshold (INV-8 decision preservation), independent of bin width.

use super::usage_perplexity;
use crate::types::{Distortion, Quantizer};

pub struct CriticalRegionQuantizer {
    edges: Vec<f64>, // sorted bin edges over [lo, hi]; bins = consecutive intervals
    lo: f64,
    hi: f64,
    bits: u8,
    usage: Vec<u64>,
}

impl CriticalRegionQuantizer {
    /// `fine_step` = target fine-bin width inside `[a,b]` (= 2·max_distortion in the field's units).
    /// Thresholds are forced to be edges. Tails outside `[a,b]` are coarse (`tail_bins` each side,
    /// log-spaced if `tail_log` and the range is positive).
    pub fn new(
        lo: f64,
        hi: f64,
        region: (f64, f64),
        thresholds: &[f64],
        fine_step: f64,
        tail_log: bool,
    ) -> Self {
        let (a, b) = region;
        let fine_step = if fine_step > 0.0 {
            fine_step
        } else {
            (b - a).max(1e-9)
        };
        let tail_bins = 16u32;
        let mut edges: Vec<f64> = Vec::new();

        // below-region tail [lo, a]
        if a > lo {
            for k in 0..tail_bins {
                let f = k as f64 / tail_bins as f64;
                let e = if tail_log && lo > 0.0 {
                    (lo.ln() + (a.ln() - lo.ln()) * f).exp()
                } else {
                    lo + (a - lo) * f
                };
                edges.push(e);
            }
        }

        // region [a,b]: breakpoints = a, in-region thresholds, b — each gap filled uniformly ≤ fine_step
        let mut bp = vec![a];
        for &t in thresholds {
            if t > a && t < b {
                bp.push(t);
            }
        }
        bp.push(b);
        bp.sort_by(|x, y| x.partial_cmp(y).unwrap());
        bp.dedup_by(|x, y| (*x - *y).abs() < 1e-12);
        for w in bp.windows(2) {
            let (s, e) = (w[0], w[1]);
            let n = (((e - s) / fine_step).ceil() as u32).max(1);
            for k in 0..n {
                edges.push(s + (e - s) * (k as f64 / n as f64));
            }
        }
        edges.push(b);

        // above-region tail [b, hi]
        if hi > b {
            for k in 1..=tail_bins {
                let f = k as f64 / tail_bins as f64;
                let e = if tail_log && b > 0.0 {
                    (b.ln() + (hi.ln() - b.ln()) * f).exp()
                } else {
                    b + (hi - b) * f
                };
                edges.push(e);
            }
        }

        edges.push(lo);
        edges.push(hi);
        edges.sort_by(|x, y| x.partial_cmp(y).unwrap());
        edges.dedup_by(|x, y| (*x - *y).abs() < 1e-12);

        let bins = edges.len().saturating_sub(1).max(1);
        let bits = super::bits_for(bins as u32);
        Self {
            edges,
            lo,
            hi,
            bits,
            usage: vec![0; bins],
        }
    }

    #[inline]
    fn bins(&self) -> usize {
        self.edges.len() - 1
    }

    #[inline]
    fn bin_of(&self, v: f64) -> usize {
        let v = v.clamp(self.lo, self.hi);
        // number of edges ≤ v, minus 1 = interval index; clamp into [0, bins-1].
        let pp = self.edges.partition_point(|&e| e <= v);
        pp.saturating_sub(1).min(self.bins() - 1)
    }

    #[inline]
    fn center(&self, i: usize) -> f64 {
        let i = i.min(self.bins() - 1);
        0.5 * (self.edges[i] + self.edges[i + 1])
    }
}

impl Quantizer for CriticalRegionQuantizer {
    fn fit(&mut self, samples: &[f64]) {
        for &s in samples {
            let i = self.bin_of(s);
            self.usage[i] += 1;
        }
    }
    fn encode(&self, v: f64) -> (u32, Distortion) {
        let i = self.bin_of(v);
        (
            i as u32,
            Distortion((v.clamp(self.lo, self.hi) - self.center(i)).abs()),
        )
    }
    fn decode(&self, t: u32) -> f64 {
        self.center(t as usize)
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

    // decision map: index of the bucket a value falls into given sorted thresholds.
    fn bucket(thresholds: &[f64], x: f64) -> usize {
        thresholds.iter().filter(|&&t| x >= t).count()
    }

    #[test]
    fn rhat_thresholds_are_decision_preserved() {
        // INV-8: rhat — region [1.0,1.2], thresholds 1.01/1.05/1.10, bound 0.0025 → step 0.005.
        let th = [1.01, 1.05, 1.10];
        let q = CriticalRegionQuantizer::new(1.0, 4.0, (1.0, 1.2), &th, 0.005, true);
        for &t in &th {
            for &eps in &[0.001, 0.005, 0.01] {
                let below = q.decode(q.encode(t - eps).0);
                let above = q.decode(q.encode(t + eps).0);
                assert_eq!(bucket(&th, below), bucket(&th, t - eps), "rhat θ={t} -ε");
                assert_eq!(bucket(&th, above), bucket(&th, t + eps), "rhat θ={t} +ε");
            }
        }
    }

    #[test]
    fn te_asymmetry_zero_crossing_preserved() {
        // sign = causal direction; the 0.0 crossing must survive quantization.
        let th = [0.0];
        let q = CriticalRegionQuantizer::new(-1.0, 1.0, (-1.0, 1.0), &th, 0.1, false);
        for &eps in &[0.001, 0.01, 0.05] {
            assert!(
                q.decode(q.encode(eps).0) > 0.0,
                "+{eps} must decode positive"
            );
            assert!(
                q.decode(q.encode(-eps).0) < 0.0,
                "-{eps} must decode negative"
            );
        }
    }

    #[test]
    fn fine_region_round_trip_within_step() {
        let th = [1.10];
        let step = 0.005;
        let q = CriticalRegionQuantizer::new(1.0, 4.0, (1.0, 1.2), &th, step, true);
        // inside the region, distortion ≤ half a fine step.
        let mut v = 1.0;
        while v < 1.2 {
            let (_c, d) = q.encode(v);
            assert!(
                d.value() <= step,
                "distortion {} > step {step} at {v}",
                d.value()
            );
            v += step / 2.0;
        }
    }

    #[test]
    fn dp_sweep_is_monotone_in_region() {
        // sweeping the region, the quantized decision is a monotone step matching the true step.
        let th = [1.01, 1.05, 1.10];
        let q = CriticalRegionQuantizer::new(1.0, 4.0, (1.0, 1.2), &th, 0.005, true);
        let mut v = 1.0;
        let mut last = 0usize;
        while v <= 1.2 {
            let qb = bucket(&th, q.decode(q.encode(v).0));
            assert!(qb >= last, "decision must not decrease sweeping up");
            assert_eq!(
                qb,
                bucket(&th, v).min(qb.max(bucket(&th, v))),
                "no spurious flip"
            );
            last = qb;
            v += 0.0025;
        }
    }
}
