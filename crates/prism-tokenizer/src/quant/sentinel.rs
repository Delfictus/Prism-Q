//! Sentinel-aware wrapper (§7): a field like `aromatic_residue_id` uses −1 to mean
//! "unresolved", which is categorically distinct from a numeric −1. The sentinel gets its own
//! exact escape code (the top code of the budget); all other values delegate to `inner`, which
//! is restricted to the remaining codes so it can never collide with the sentinel.

use crate::types::{Distortion, Quantizer};

pub struct SentinelQuantizer {
    value: f64,
    eps: f64,
    sentinel_code: u32,
    inner: Box<dyn Quantizer>,
}

impl SentinelQuantizer {
    pub fn new(value: f64, inner: Box<dyn Quantizer>) -> Self {
        let bits = inner.bit_budget();
        let sentinel_code = (1u32 << bits) - 1;
        Self {
            value,
            eps: 1e-9,
            sentinel_code,
            inner,
        }
    }

    #[inline]
    fn is_sentinel(&self, v: f64) -> bool {
        (v - self.value).abs() <= self.eps
    }
}

impl Quantizer for SentinelQuantizer {
    fn fit(&mut self, samples: &[f64]) {
        // The sentinel value never trains the inner codebook.
        let non: Vec<f64> = samples
            .iter()
            .copied()
            .filter(|&x| !self.is_sentinel(x))
            .collect();
        self.inner.fit(&non);
    }
    fn encode(&self, v: f64) -> (u32, Distortion) {
        if self.is_sentinel(v) {
            return (self.sentinel_code, Distortion(0.0));
        }
        let (t, d) = self.inner.encode(v);
        // never let an inner code collide with the reserved sentinel code.
        (t.min(self.sentinel_code.saturating_sub(1)), d)
    }
    fn decode(&self, t: u32) -> f64 {
        if t == self.sentinel_code {
            self.value
        } else {
            self.inner.decode(t)
        }
    }
    fn bit_budget(&self) -> u8 {
        self.inner.bit_budget()
    }
    fn perplexity(&self) -> f64 {
        // Delegated to inner; the sentinel contributes one extra reserved code.
        self.inner.perplexity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::linear::LinearQuantizer;

    #[test]
    fn sentinel_is_distinct_from_numeric_value() {
        // aromatic_residue_id: −1 unresolved, else a residue id in [0, 4096).
        let inner = Box::new(LinearQuantizer::new(10, 0.0, 4096.0));
        let q = SentinelQuantizer::new(-1.0, inner);

        let (s_code, s_d) = q.encode(-1.0);
        assert_eq!(s_code, (1u32 << 10) - 1);
        assert_eq!(s_d.value(), 0.0);
        assert_eq!(q.decode(s_code), -1.0);

        // a real id near 0 must NOT collide with the sentinel code, and must decode ~itself.
        let (code, _d) = q.encode(12.0);
        assert_ne!(code, s_code);
        assert!((q.decode(code) - 12.0).abs() < 8.0);
    }

    #[test]
    fn inner_codes_never_reach_sentinel_code() {
        let inner = Box::new(LinearQuantizer::new(8, 0.0, 4096.0));
        let q = SentinelQuantizer::new(-1.0, inner);
        let top = (1u32 << 8) - 1;
        // the max inner value would map to the top inner code; it must be pushed below sentinel.
        assert!(q.encode(4096.0).0 < top);
    }
}
