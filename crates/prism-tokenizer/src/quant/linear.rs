//! Bounded linear quantizer (§7): probabilities, label_confidence ∈ [0,1], and other fields
//! whose decoded range is bounded and roughly uniform.

use super::usage_perplexity;
use crate::types::{Distortion, Quantizer};

pub struct LinearQuantizer {
    bits: u8,
    k: u32,
    lo: f64,
    hi: f64,
    usage: Vec<u64>,
}

impl LinearQuantizer {
    pub fn new(bits: u8, lo: f64, hi: f64) -> Self {
        let hi = if hi > lo { hi } else { lo + 1.0 };
        let k = 1u32 << bits;
        Self {
            bits,
            k,
            lo,
            hi,
            usage: vec![0; k as usize],
        }
    }

    #[inline]
    fn bin_of(&self, v: f64) -> u32 {
        let frac = (v - self.lo) / (self.hi - self.lo);
        ((frac * self.k as f64).floor() as i64).clamp(0, self.k as i64 - 1) as u32
    }

    #[inline]
    fn center(&self, t: u32) -> f64 {
        let t = t.min(self.k - 1);
        self.lo + (t as f64 + 0.5) / self.k as f64 * (self.hi - self.lo)
    }
}

impl Quantizer for LinearQuantizer {
    fn fit(&mut self, samples: &[f64]) {
        for &s in samples {
            let b = self.bin_of(s) as usize;
            self.usage[b] += 1;
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
    fn unit_interval_round_trip() {
        let q = LinearQuantizer::new(10, 0.0, 1.0);
        let bin_w = 1.0 / 1024.0;
        for &v in &[0.0, 0.25, 0.5, 0.91, 1.0] {
            let (t, d) = q.encode(v);
            assert!(d.value() <= bin_w);
            assert!((q.decode(t) - v).abs() <= bin_w);
        }
    }

    #[test]
    fn clamps_out_of_range() {
        let q = LinearQuantizer::new(8, 0.0, 1.0);
        assert_eq!(q.encode(-5.0).0, 0);
        assert_eq!(q.encode(5.0).0, 255);
    }
}
