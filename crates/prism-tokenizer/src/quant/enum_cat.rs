//! Categorical quantizer (§7): spike_source 1..5, status, trigger_kind, aromatic_type. The
//! decoded value IS the code (identity), sized to the smallest bit budget for the cardinality.

use super::{bits_for, usage_perplexity};
use crate::types::{Distortion, Quantizer};

pub struct EnumQuantizer {
    cardinality: u32,
    bits: u8,
    usage: Vec<u64>,
}

impl EnumQuantizer {
    pub fn new(cardinality: u32) -> Self {
        let cardinality = cardinality.max(1);
        Self {
            cardinality,
            bits: bits_for(cardinality),
            usage: vec![0; cardinality as usize],
        }
    }

    #[inline]
    fn code_of(&self, v: f64) -> u32 {
        (v.round() as i64).clamp(0, self.cardinality as i64 - 1) as u32
    }
}

impl Quantizer for EnumQuantizer {
    fn fit(&mut self, samples: &[f64]) {
        for &s in samples {
            let c = self.code_of(s) as usize;
            self.usage[c] += 1;
        }
    }
    fn encode(&self, v: f64) -> (u32, Distortion) {
        let c = self.code_of(v);
        (c, Distortion((v - c as f64).abs()))
    }
    fn decode(&self, t: u32) -> f64 {
        t.min(self.cardinality - 1) as f64
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
    fn spike_source_five_categories() {
        // codes 0..4 for the 5 spike sources; 3 bits suffice.
        let q = EnumQuantizer::new(5);
        assert_eq!(q.bit_budget(), 3);
        for c in 0..5u32 {
            let (code, d) = q.encode(c as f64);
            assert_eq!(code, c);
            assert_eq!(d.value(), 0.0);
            assert_eq!(q.decode(code), c as f64);
        }
    }

    #[test]
    fn out_of_range_clamps() {
        let q = EnumQuantizer::new(5);
        assert_eq!(q.encode(99.0).0, 4);
        assert_eq!(q.encode(-3.0).0, 0);
    }
}
