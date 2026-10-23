//! Circular quantizer for phase / dihedral fields (§7). 0 ≡ period; distortion is wrapped
//! angular distance. Linear quantization of these fields is wrong (do-not §18).

use super::usage_perplexity;
use crate::types::{Distortion, Quantizer};

pub struct CircularQuantizer {
    bits: u8,
    k: u32,
    period: f64,
    usage: Vec<u64>,
}

impl CircularQuantizer {
    pub fn new(bits: u8, period: f64) -> Self {
        let k = 1u32 << bits;
        Self {
            bits,
            k,
            period,
            usage: vec![0; k as usize],
        }
    }

    #[inline]
    fn bin_of(&self, v: f64) -> u32 {
        let p = self.period;
        let mut x = v % p;
        if x < 0.0 {
            x += p;
        }
        let b = (x / p * self.k as f64).floor() as i64;
        b.rem_euclid(self.k as i64) as u32
    }

    #[inline]
    fn center(&self, t: u32) -> f64 {
        (t as f64 + 0.5) * self.period / self.k as f64
    }

    #[inline]
    fn wrapped_dist(&self, a: f64, b: f64) -> f64 {
        let p = self.period;
        let mut d = (a - b) % p;
        if d < -p / 2.0 {
            d += p;
        }
        if d > p / 2.0 {
            d -= p;
        }
        d.abs()
    }
}

impl Quantizer for CircularQuantizer {
    fn fit(&mut self, samples: &[f64]) {
        for &s in samples {
            let b = self.bin_of(s) as usize;
            self.usage[b] += 1;
        }
    }
    fn encode(&self, v: f64) -> (u32, Distortion) {
        let t = self.bin_of(v);
        (t, Distortion(self.wrapped_dist(v, self.center(t))))
    }
    fn decode(&self, t: u32) -> f64 {
        self.center(t.min(self.k - 1))
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
    use core::f64::consts::TAU;

    #[test]
    fn zero_and_period_map_to_same_bin() {
        let q = CircularQuantizer::new(8, TAU);
        assert_eq!(q.bin_of(0.0), q.bin_of(TAU));
        assert_eq!(q.bin_of(0.0), q.bin_of(-TAU));
    }

    #[test]
    fn decode_within_one_bin_width() {
        let q = CircularQuantizer::new(8, TAU);
        let bin_w = TAU / 256.0;
        for &v in &[0.1, 1.0, 3.0, 6.0] {
            let (t, d) = q.encode(v);
            assert!(
                d.value() <= bin_w,
                "distortion {} > bin width {}",
                d.value(),
                bin_w
            );
            assert!((q.decode(t) - v).abs() <= bin_w);
        }
    }

    #[test]
    fn wrap_distance_is_short_way_around() {
        let q = CircularQuantizer::new(10, TAU);
        // 0.01 and TAU-0.01 are 0.02 apart the short way, not ~TAU.
        let d = q.wrapped_dist(0.01, TAU - 0.01);
        assert!(
            d < 0.05,
            "wrapped distance should be the short arc, got {d}"
        );
    }

    #[test]
    fn perplexity_grows_with_spread() {
        let mut q = CircularQuantizer::new(6, TAU);
        q.fit(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(q.perplexity() > 1.0);
    }
}
