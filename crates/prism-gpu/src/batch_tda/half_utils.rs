//! Half-precision (F16) utilities for memory-efficient distance storage
//!
//! IEEE 754 binary16 format:
//! - 1 sign bit
//! - 5 exponent bits (bias 15)
//! - 10 mantissa bits
//!
//! Range: ~6e-8 to ~65504
//! Precision: ~3 decimal digits
//!
//! For distances in Angstroms (0-20Å typical), F16 provides sufficient precision
//! while halving memory bandwidth requirements.

/// Half-precision floating point type (IEEE 754 binary16)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct F16(pub u16);

impl F16 {
    /// Zero
    pub const ZERO: F16 = F16(0x0000);

    /// One
    pub const ONE: F16 = F16(0x3C00);

    /// Maximum finite value (~65504)
    pub const MAX: F16 = F16(0x7BFF);

    /// Minimum positive normalized value
    pub const MIN_POSITIVE: F16 = F16(0x0400);

    /// Positive infinity
    pub const INFINITY: F16 = F16(0x7C00);

    /// NaN
    pub const NAN: F16 = F16(0x7E00);

    /// Create from raw bits
    #[inline]
    pub const fn from_bits(bits: u16) -> Self {
        F16(bits)
    }

    /// Get raw bits
    #[inline]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// Check if NaN
    #[inline]
    pub fn is_nan(self) -> bool {
        (self.0 & 0x7C00) == 0x7C00 && (self.0 & 0x03FF) != 0
    }

    /// Check if infinite
    #[inline]
    pub fn is_infinite(self) -> bool {
        (self.0 & 0x7FFF) == 0x7C00
    }

    /// Check if finite (not NaN or infinity)
    #[inline]
    pub fn is_finite(self) -> bool {
        (self.0 & 0x7C00) != 0x7C00
    }
}

impl From<f32> for F16 {
    #[inline]
    fn from(value: f32) -> Self {
        F16(f32_to_f16(value))
    }
}

impl From<F16> for f32 {
    #[inline]
    fn from(value: F16) -> Self {
        f16_to_f32(value.0)
    }
}

/// Convert f32 to f16 bits
///
/// Uses the standard conversion algorithm with proper rounding.
#[inline]
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007FFFFF;

    // Handle special cases
    if exponent == 255 {
        // NaN or infinity
        if mantissa != 0 {
            // NaN - preserve some mantissa bits
            return (sign | 0x7E00 | (mantissa >> 13)) as u16;
        } else {
            // Infinity
            return (sign | 0x7C00) as u16;
        }
    }

    // Bias conversion: f32 bias = 127, f16 bias = 15
    let new_exp = exponent - 127 + 15;

    if new_exp >= 31 {
        // Overflow to infinity
        return (sign | 0x7C00) as u16;
    }

    if new_exp <= 0 {
        // Subnormal or underflow
        if new_exp < -10 {
            // Too small, round to zero
            return sign as u16;
        }

        // Subnormal: shift mantissa right and add implicit 1
        let mant = (mantissa | 0x00800000) >> (14 - new_exp);

        // Round to nearest even
        let round_bit = 1 << (13 - new_exp);
        let sticky = mant & (round_bit - 1);
        let mant = (mant + round_bit) >> 1;

        // Handle rounding overflow
        let mant = if sticky == 0 && (mant & 1) != 0 {
            mant & !1 // Round to even
        } else {
            mant
        };

        return (sign | (mant >> (13 - new_exp))) as u16;
    }

    // Normal number
    let mant = mantissa >> 13;

    // Round to nearest even
    let round_bit = mantissa & 0x1000;
    let sticky = mantissa & 0x0FFF;

    let mut result = sign | ((new_exp as u32) << 10) | mant;

    if round_bit != 0 {
        if sticky != 0 || (mant & 1) != 0 {
            result += 1;
        }
    }

    result as u16
}

/// Convert f16 bits to f32
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = (bits >> 10) & 0x1F;
    let mantissa = (bits & 0x03FF) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            // Zero
            return f32::from_bits(sign);
        }
        // Subnormal: normalize
        let mut mant = mantissa;
        let mut exp = -14i32;
        while (mant & 0x0400) == 0 {
            mant <<= 1;
            exp -= 1;
        }
        mant &= 0x03FF;
        let new_exp = ((exp + 127) as u32) << 23;
        return f32::from_bits(sign | new_exp | (mant << 13);
    }

    if exponent == 31 {
        if mantissa == 0 {
            // Infinity
            return f32::from_bits(sign | 0x7F800000);
        }
        // NaN
        return f32::from_bits(sign | 0x7FC00000 | (mantissa << 13);
    }

    // Normal number
    let new_exp = ((exponent as i32 - 15 + 127) as u32) << 23;
    f32::from_bits(sign | new_exp | (mantissa << 13))
}

/// Batch convert f32 slice to f16 bits
///
/// Uses SIMD when available (through auto-vectorization).
pub fn f32_slice_to_f16(input: &[f32], output: &mut [u16]) {
    assert_eq!(input.len(), output.len();
    for (i, &v) in input.iter().enumerate() {
        output[i] = f32_to_f16(v);
    }
}

/// Batch convert f16 bits to f32
pub fn f16_slice_to_f32(input: &[u16], output: &mut [f32]) {
    assert_eq!(input.len(), output.len();
    for (i, &v) in input.iter().enumerate() {
        output[i] = f16_to_f32(v);
    }
}

/// Packed half-precision distance matrix
///
/// Stores the upper triangle of a symmetric distance matrix in F16 format.
/// Memory usage: n*(n-1)/2 * 2 bytes
pub struct PackedDistanceMatrix {
    /// Number of points
    n: usize,
    /// Upper triangle distances in row-major order (F16 bits)
    data: Vec<u16>,
}

impl PackedDistanceMatrix {
    /// Create a new empty distance matrix
    pub fn new(n: usize) -> Self {
        let size = n * (n - 1) / 2;
        Self {
            n,
            data: vec![0u16; size],
        }
    }

    /// Create from a full distance matrix (f32)
    pub fn from_full_matrix(matrix: &[f32], n: usize) -> Self {
        assert_eq!(matrix.len(), n * n);
        let mut result = Self::new(n);

        for i in 0..n {
            for j in (i + 1)..n {
                let dist = matrix[i * n + j];
                result.set(i, j, dist);
            }
        }

        result
    }

    /// Get the linear index for (i, j) where i < j
    #[inline]
    fn index(&self, i: usize, j: usize) -> usize {
        debug_assert!(i < j && j < self.n);
        // Row i has entries for columns i+1, i+2, ..., n-1
        // Number of entries before row i: sum(n-1-k for k in 0..i) = i*n - i*(i+1)/2
        let row_offset = i * self.n - (i * (i + 1)) / 2;
        row_offset + (j - i - 1)
    }

    /// Get distance between points i and j
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f32 {
        if i == j {
            return 0.0;
        }
        let (i, j) = if i < j { (i, j) } else { (j, i) };
        f16_to_f32(self.data[self.index(i, j)])
    }

    /// Set distance between points i and j
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, dist: f32) {
        if i == j {
            return;
        }
        let (i, j) = if i < j { (i, j) } else { (j, i) };
        let idx = self.index(i, j);
        self.data[idx] = f32_to_f16(dist);
    }

    /// Get raw F16 bits
    #[inline]
    pub fn get_raw(&self, i: usize, j: usize) -> u16 {
        if i == j {
            return 0;
        }
        let (i, j) = if i < j { (i, j) } else { (j, i) };
        self.data[self.index(i, j)]
    }

    /// Number of points
    #[inline]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Raw data slice (for GPU upload)
    #[inline]
    pub fn as_slice(&self) -> &[u16] {
        &self.data
    }

    /// Mutable raw data slice
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u16] {
        &mut self.data
    }

    /// Memory usage in bytes
    #[inline]
    pub fn memory_bytes(&self) -> usize {
        self.data.len() * 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_f16_roundtrip() {
        let values = [
            0.0f32, 1.0, -1.0, 0.5, 2.0, 10.0, 100.0, 0.001, 0.0001,
            65504.0, // Max F16
            -65504.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];

        for &v in &values {
            let f16_bits = f32_to_f16(v);
            let back = f16_to_f32(f16_bits);

            if v.is_infinite() {
                assert!(back.is_infinite();
                assert_eq!(v.is_sign_positive(), back.is_sign_positive();
            } else if v == 0.0 {
                assert_eq!(back, 0.0);
            } else {
                // Check relative error is small (F16 has ~3 decimal digits)
                let rel_error = ((back - v) / v).abs();
                assert!(rel_error < 0.01, "Value {} -> {} (error {})", v, back, rel_error);
            }
        }
    }

    #[test]
    fn test_f16_nan() {
        let nan = f32_to_f16(f32::NAN);
        let back = f16_to_f32(nan);
        assert!(back.is_nan();
    }

    #[test]
    fn test_packed_distance_matrix() {
        let n = 5;
        let mut matrix = PackedDistanceMatrix::new(n);

        // Set some distances
        matrix.set(0, 1, 1.0);
        matrix.set(0, 2, 2.0);
        matrix.set(1, 4, 5.5);

        // Check retrieval (both directions)
        assert!((matrix.get(0, 1) - 1.0).abs() < 0.01);
        assert!((matrix.get(1, 0) - 1.0).abs() < 0.01);
        assert!((matrix.get(0, 2) - 2.0).abs() < 0.01);
        assert!((matrix.get(1, 4) - 5.5).abs() < 0.01);
        assert!((matrix.get(4, 1) - 5.5).abs() < 0.01);

        // Diagonal should be zero
        assert_eq!(matrix.get(0, 0), 0.0);
        assert_eq!(matrix.get(3, 3), 0.0);
    }

    #[test]
    fn test_packed_matrix_memory() {
        let n = 100;
        let matrix = PackedDistanceMatrix::new(n);
        // Upper triangle: n*(n-1)/2 = 4950 entries * 2 bytes = 9900 bytes
        assert_eq!(matrix.memory_bytes(), 9900);
    }
}
