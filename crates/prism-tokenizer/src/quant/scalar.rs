//! Trivial scalar quantizers: Boolean and JoinKey (§7).
//!
//! Boolean — is_bimodal / structural_verified / hard_negative_mask / train_mask / oracle_valid.
//! JoinKey — residue_id / anchor_id / stream_id: the value IS the relational id; the §9 join
//! layer resolves it to a record. Here it is an identity passthrough into the field-token space.

use crate::types::{Distortion, Quantizer};

pub struct BooleanQuantizer;

impl BooleanQuantizer {
    pub fn new() -> Self {
        BooleanQuantizer
    }
}
impl Default for BooleanQuantizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Quantizer for BooleanQuantizer {
    fn fit(&mut self, _samples: &[f64]) {}
    fn encode(&self, v: f64) -> (u32, Distortion) {
        let code = if v >= 0.5 { 1 } else { 0 };
        (code, Distortion((v - code as f64).abs()))
    }
    fn decode(&self, t: u32) -> f64 {
        (t & 1) as f64
    }
    fn bit_budget(&self) -> u8 {
        1
    }
    fn perplexity(&self) -> f64 {
        2.0
    }
}

/// Identity over relational ids. `bits` bounds the id space that fits the field-token; values
/// beyond it are clamped (and would be sidecar-escaped upstream).
pub struct JoinKeyQuantizer {
    bits: u8,
}

impl JoinKeyQuantizer {
    pub fn new() -> Self {
        JoinKeyQuantizer { bits: 24 }
    }
    pub fn with_bits(bits: u8) -> Self {
        JoinKeyQuantizer { bits: bits.min(31) }
    }
    #[inline]
    fn max_code(&self) -> u32 {
        (1u32 << self.bits) - 1
    }
}
impl Default for JoinKeyQuantizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Quantizer for JoinKeyQuantizer {
    fn fit(&mut self, _samples: &[f64]) {}
    fn encode(&self, v: f64) -> (u32, Distortion) {
        let id = if v < 0.0 { 0 } else { v.round() as i64 };
        let code = id.clamp(0, self.max_code() as i64) as u32;
        (code, Distortion((v - code as f64).abs()))
    }
    fn decode(&self, t: u32) -> f64 {
        t as f64
    }
    fn bit_budget(&self) -> u8 {
        self.bits
    }
    fn perplexity(&self) -> f64 {
        // identity space; usage is unbounded by design, so report the code-space size.
        (1u64 << self.bits) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boolean_thresholds_at_half() {
        let q = BooleanQuantizer::new();
        assert_eq!(q.encode(0.0).0, 0);
        assert_eq!(q.encode(1.0).0, 1);
        assert_eq!(q.encode(0.7).0, 1);
        assert_eq!(q.decode(1), 1.0);
    }

    #[test]
    fn joinkey_is_identity() {
        let q = JoinKeyQuantizer::new();
        let (code, d) = q.encode(4096.0);
        assert_eq!(code, 4096);
        assert_eq!(d.value(), 0.0);
        assert_eq!(q.decode(code), 4096.0);
    }

    #[test]
    fn joinkey_negative_clamps_to_zero() {
        let q = JoinKeyQuantizer::new();
        assert_eq!(q.encode(-1.0).0, 0);
    }
}
