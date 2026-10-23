//! Spectral-line quantizer (§7): `wavelength_nm` is one of a few discrete physics lines
//! (280 = Trp/Tyr, …), not a digit string. Code = nearest-line index, plus an optional residual
//! codebook for off-line values. This is information-dense: the common case is one of N lines.

use super::{bits_for, usage_perplexity};
use crate::types::{Distortion, Quantizer};

pub struct SpectralLineQuantizer {
    lines: Vec<f64>,
    residual_bits: u8,
    residual_k: u32,
    line_bits: u8,
    /// half-width of the residual window (nm) around each line.
    residual_range: f64,
    usage: Vec<u64>,
}

impl SpectralLineQuantizer {
    pub fn new(lines: &[f32], residual_bits: u8) -> Self {
        let lines: Vec<f64> = lines.iter().map(|&x| x as f64).collect();
        let n = lines.len().max(1) as u32;
        let residual_k = 1u32 << residual_bits;
        Self {
            line_bits: bits_for(n),
            usage: vec![0; (n * residual_k) as usize],
            lines,
            residual_bits,
            residual_k,
            residual_range: 4.0,
        }
    }

    #[inline]
    fn nearest_line(&self, v: f64) -> usize {
        let mut best = 0usize;
        let mut best_d = f64::INFINITY;
        for (i, &l) in self.lines.iter().enumerate() {
            let d = (v - l).abs();
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        best
    }

    #[inline]
    fn residual_step(&self) -> f64 {
        2.0 * self.residual_range / self.residual_k as f64
    }

    // The residual codebook is centered on 0 (bin K/2 ⇒ residual 0), so a value exactly on a
    // line round-trips exactly even with residual_bits > 0; off-line values snap to the nearest
    // residual center.
    #[inline]
    fn residual_bin(&self, r: f64) -> u32 {
        if self.residual_k <= 1 {
            return 0;
        }
        let half = (self.residual_k / 2) as i64;
        ((r / self.residual_step()).round() as i64 + half).clamp(0, self.residual_k as i64 - 1)
            as u32
    }

    #[inline]
    fn residual_center(&self, rbin: u32) -> f64 {
        if self.residual_k <= 1 {
            return 0.0;
        }
        (rbin as f64 - (self.residual_k / 2) as f64) * self.residual_step()
    }

    #[inline]
    fn pack(&self, line: usize, rbin: u32) -> u32 {
        (line as u32) * self.residual_k + rbin
    }
}

impl Quantizer for SpectralLineQuantizer {
    fn fit(&mut self, samples: &[f64]) {
        for &s in samples {
            let i = self.nearest_line(s);
            let rb = self.residual_bin(s - self.lines[i]);
            let code = self.pack(i, rb) as usize;
            self.usage[code] += 1;
        }
    }
    fn encode(&self, v: f64) -> (u32, Distortion) {
        let i = self.nearest_line(v);
        let rb = self.residual_bin(v - self.lines[i]);
        let code = self.pack(i, rb);
        (code, Distortion((v - self.decode(code)).abs()))
    }
    fn decode(&self, t: u32) -> f64 {
        let line = (t / self.residual_k) as usize;
        let rbin = t % self.residual_k;
        let line = line.min(self.lines.len().saturating_sub(1));
        self.lines[line] + self.residual_center(rbin)
    }
    fn bit_budget(&self) -> u8 {
        self.line_bits + self.residual_bits
    }
    fn perplexity(&self) -> f64 {
        usage_perplexity(&self.usage)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const UV: &[f32] = &[280.0, 274.0, 258.0, 254.0, 211.0];

    #[test]
    fn exact_line_round_trips_to_itself() {
        let q = SpectralLineQuantizer::new(UV, 0);
        for &l in UV {
            let (code, d) = q.encode(l as f64);
            assert!(d.value() < 1e-9, "exact line should have ~0 distortion");
            assert_eq!(q.decode(code), l as f64);
        }
    }

    #[test]
    fn nearest_line_for_off_value() {
        let q = SpectralLineQuantizer::new(UV, 0);
        // 279.5 → nearest is 280.
        let (code, _) = q.encode(279.5);
        assert_eq!(q.decode(code), 280.0);
    }

    #[test]
    fn residual_codebook_reduces_distortion() {
        let q0 = SpectralLineQuantizer::new(UV, 0);
        let q3 = SpectralLineQuantizer::new(UV, 3);
        let v = 282.0; // 2nm off the 280 line
        assert!(q3.encode(v).1.value() < q0.encode(v).1.value());
    }
}
