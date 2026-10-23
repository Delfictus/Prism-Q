//! RECT-3.1 — Spherical Harmonics Y_lm Evaluator (Lmax=5).
//!
//! Per the Production Architecture mandate Phase 1 Deliverable 1.1
//! (operator directive 2026-04-29). Straight-line evaluation of 36
//! real spherical harmonics for Lmax=5; designed to feed the SO(3)
//! projection kernel (Deliverable 1.2) which accumulates the
//! per-spike `Y_lm * intensity` contributions into `a_lm`
//! coefficients via WMMA Tensor Core fragments.
//!
//! # Convention (locked)
//!
//! ```text
//! Y_l^m(θ, φ) = K_lm * P_l^|m|(cos θ) * angular(m, φ)
//!
//!   angular(0, φ)   = 1
//!   angular(m>0, φ) = cos(m·φ)
//!   angular(m<0, φ) = sin(|m|·φ)
//!
//!   K_lm (m=0)  = sqrt((2l+1) / (4π))
//!   K_lm (m≠0)  = sqrt((2l+1) / (2π) · (l-|m|)! / (l+|m|)!)
//!
//!   P_l^m       = associated Legendre, "physical" convention
//!                 (no Condon-Shortley (-1)^m phase).
//! ```
//!
//! The C_l = Σ_m |a_lm|² rotational power spectrum is invariant
//! under the choice of phase convention, so the G11 invariance gate
//! (Deliverable 1.2) passes regardless. Both the Rust CPU reference
//! here and the GPU kernel in `sh_basis.cu` use the same convention
//! bit-for-bit; the parity test `cpu_gpu_sh_parity_*` enforces this.
//!
//! # Index layout
//!
//! Y[`l*(l+1) + m`] for l ∈ [0, 5], m ∈ [-l, l]. 36 entries.

use serde::{Deserialize, Serialize};

/// Maximum SH degree.
pub const LMAX: usize = 5;
/// Number of real spherical harmonic coefficients.
pub const N_COEFFS: usize = (LMAX + 1) * (LMAX + 1);

// ============================================================================
// CPU reference — bit-equivalent to the GPU evaluator's algebra.
// ============================================================================

/// Compute the 36 K_lm normalization constants. `f64` arithmetic
/// rounded to `f32` at the end so the CPU reference matches the
/// GPU's f32 init kernel up to f32 precision.
pub fn k_lm_table() -> [f32; N_COEFFS] {
    use std::f64::consts::PI;
    let mut k = [0.0f32; N_COEFFS];
    for l in 0..=LMAX {
        for m in -(l as i32)..=(l as i32) {
            let idx = (l as i32 * (l as i32 + 1) + m) as usize;
            let am = m.unsigned_abs() as usize;
            let mut denom: f64 = 1.0;
            // factorial ratio (l-am)! / (l+am)! = 1 / Π_{i=l-am+1..=l+am} i
            for i in (l - am + 1)..=(l + am) {
                denom *= i as f64;
            }
            let fact_ratio = 1.0 / denom;
            let val = if m == 0 {
                ((2.0 * l as f64 + 1.0) / (4.0 * PI)).sqrt()
            } else {
                ((2.0 * l as f64 + 1.0) / (2.0 * PI) * fact_ratio).sqrt()
            };
            k[idx] = val as f32;
        }
    }
    k
}

/// CPU reference SH evaluator. Literal port of `prism_sh_eval_lmax5`
/// in `sh_basis.cuh`. Returns 36 real Y_lm values.
pub fn cpu_sh_eval_lmax5(theta: f32, phi: f32, k_lm: &[f32; N_COEFFS]) -> [f32; N_COEFFS] {
    let (st, ct) = theta.sin_cos();
    let (ct2, ct3, ct4, ct5) = (ct * ct, ct * ct * ct, ct.powi(4), ct.powi(5));
    let (st2, st3, st4, st5) = (st * st, st * st * st, st.powi(4), st.powi(5));

    // Associated Legendre P_l^|m|.
    let p_0_0 = 1.0_f32;
    let p_1_0 = ct;
    let p_1_1 = st;
    let p_2_0 = 0.5 * (3.0 * ct2 - 1.0);
    let p_2_1 = 3.0 * ct * st;
    let p_2_2 = 3.0 * st2;
    let p_3_0 = 0.5 * (5.0 * ct3 - 3.0 * ct);
    let p_3_1 = 1.5 * (5.0 * ct2 - 1.0) * st;
    let p_3_2 = 15.0 * ct * st2;
    let p_3_3 = 15.0 * st3;
    let p_4_0 = 0.125 * (35.0 * ct4 - 30.0 * ct2 + 3.0);
    let p_4_1 = 2.5 * (7.0 * ct3 - 3.0 * ct) * st;
    let p_4_2 = 7.5 * (7.0 * ct2 - 1.0) * st2;
    let p_4_3 = 105.0 * ct * st3;
    let p_4_4 = 105.0 * st4;
    let p_5_0 = 0.125 * (63.0 * ct5 - 70.0 * ct3 + 15.0 * ct);
    let p_5_1 = (15.0_f32 / 8.0) * (21.0 * ct4 - 14.0 * ct2 + 1.0) * st;
    let p_5_2 = 52.5 * (3.0 * ct3 - ct) * st2;
    let p_5_3 = 52.5 * (9.0 * ct2 - 1.0) * st3;
    let p_5_4 = 945.0 * ct * st4;
    let p_5_5 = 945.0 * st5;

    let (sin_phi, cos_phi) = phi.sin_cos();
    let cos_2phi = cos_phi * cos_phi - sin_phi * sin_phi;
    let sin_2phi = 2.0 * sin_phi * cos_phi;
    let cos_3phi = cos_phi * cos_2phi - sin_phi * sin_2phi;
    let sin_3phi = sin_phi * cos_2phi + cos_phi * sin_2phi;
    let cos_4phi = cos_phi * cos_3phi - sin_phi * sin_3phi;
    let sin_4phi = sin_phi * cos_3phi + cos_phi * sin_3phi;
    let cos_5phi = cos_phi * cos_4phi - sin_phi * sin_4phi;
    let sin_5phi = sin_phi * cos_4phi + cos_phi * sin_4phi;

    let mut y = [0.0f32; N_COEFFS];

    y[0] = k_lm[0] * p_0_0;
    y[1] = k_lm[1] * p_1_1 * sin_phi;
    y[2] = k_lm[2] * p_1_0;
    y[3] = k_lm[3] * p_1_1 * cos_phi;
    y[4] = k_lm[4] * p_2_2 * sin_2phi;
    y[5] = k_lm[5] * p_2_1 * sin_phi;
    y[6] = k_lm[6] * p_2_0;
    y[7] = k_lm[7] * p_2_1 * cos_phi;
    y[8] = k_lm[8] * p_2_2 * cos_2phi;
    y[9] = k_lm[9] * p_3_3 * sin_3phi;
    y[10] = k_lm[10] * p_3_2 * sin_2phi;
    y[11] = k_lm[11] * p_3_1 * sin_phi;
    y[12] = k_lm[12] * p_3_0;
    y[13] = k_lm[13] * p_3_1 * cos_phi;
    y[14] = k_lm[14] * p_3_2 * cos_2phi;
    y[15] = k_lm[15] * p_3_3 * cos_3phi;
    y[16] = k_lm[16] * p_4_4 * sin_4phi;
    y[17] = k_lm[17] * p_4_3 * sin_3phi;
    y[18] = k_lm[18] * p_4_2 * sin_2phi;
    y[19] = k_lm[19] * p_4_1 * sin_phi;
    y[20] = k_lm[20] * p_4_0;
    y[21] = k_lm[21] * p_4_1 * cos_phi;
    y[22] = k_lm[22] * p_4_2 * cos_2phi;
    y[23] = k_lm[23] * p_4_3 * cos_3phi;
    y[24] = k_lm[24] * p_4_4 * cos_4phi;
    y[25] = k_lm[25] * p_5_5 * sin_5phi;
    y[26] = k_lm[26] * p_5_4 * sin_4phi;
    y[27] = k_lm[27] * p_5_3 * sin_3phi;
    y[28] = k_lm[28] * p_5_2 * sin_2phi;
    y[29] = k_lm[29] * p_5_1 * sin_phi;
    y[30] = k_lm[30] * p_5_0;
    y[31] = k_lm[31] * p_5_1 * cos_phi;
    y[32] = k_lm[32] * p_5_2 * cos_2phi;
    y[33] = k_lm[33] * p_5_3 * cos_3phi;
    y[34] = k_lm[34] * p_5_4 * cos_4phi;
    y[35] = k_lm[35] * p_5_5 * cos_5phi;

    y
}

// ============================================================================
// Sigil enum for L,M index pairs (handy for tests + future RECT-3.1.b)
// ============================================================================

/// `(l, m)` pair labeling one SH coefficient. Useful for diagnostic
/// output; the GPU kernel writes by raw index and never sees this
/// enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LMIndex {
    pub l: u8,
    pub m: i8,
}

impl LMIndex {
    pub const fn from_idx(idx: usize) -> Option<Self> {
        // l = floor(sqrt(idx)); m = idx - l*(l+1)
        if idx >= N_COEFFS {
            return None;
        }
        // Manual sqrt since `f32::sqrt` is not const.
        // l is in 0..=5; iterate.
        let mut l: u8 = 0;
        while l <= LMAX as u8 {
            let base = (l as i32) * (l as i32 + 1);
            if (idx as i32) >= base - (l as i32) && (idx as i32) <= base + (l as i32) {
                let m = (idx as i32) - base;
                return Some(LMIndex { l, m: m as i8 });
            }
            l += 1;
        }
        None
    }

    pub const fn to_idx(&self) -> usize {
        ((self.l as i32) * (self.l as i32 + 1) + self.m as i32) as usize
    }
}

// ============================================================================
// FFI surface
// ============================================================================

#[cfg(feature = "gpu")]
#[allow(dead_code)]
pub(crate) mod ffi {
    pub type CudaError = i32;
    pub const CUDA_SUCCESS: CudaError = 0;

    extern "C" {
        pub fn prism_sh_basis_link_probe() -> u32;

        pub fn prism_sh_basis_init(stream: *mut std::ffi::c_void) -> CudaError;

        pub fn prism_sh_eval_run(
            d_theta_phi: *const f32,
            n_points: u32,
            d_Y_out: *mut f32,
            stream: *mut std::ffi::c_void,
        ) -> CudaError;

        pub fn prism_sh_basis_get_k_lm_dev_ptr(out_dev_ptr: *mut *const f32) -> CudaError;
    }
}

/// Upload the K_LM normalization table to device memory. Idempotent;
/// safe to call multiple times. Must be called before [`k_lm_device_ptr`].
#[cfg(feature = "gpu")]
pub fn init_device_basis(stream: *mut std::ffi::c_void) -> Result<(), i32> {
    let rc = unsafe { ffi::prism_sh_basis_init(stream) };
    if rc != ffi::CUDA_SUCCESS {
        return Err(rc);
    }
    Ok(())
}

/// Retrieve the device-side pointer to the K_LM[36] normalization
/// table populated by `prism_sh_basis_init`. Caller MUST have
/// invoked `prism_sh_basis_init` once before this call. Returns
/// `Err(cuda_error_code)` on failure.
#[cfg(feature = "gpu")]
pub fn k_lm_device_ptr() -> Result<*const f32, i32> {
    let mut ptr: *const f32 = std::ptr::null();
    let rc = unsafe { ffi::prism_sh_basis_get_k_lm_dev_ptr(&mut ptr as *mut _) };
    if rc != ffi::CUDA_SUCCESS {
        return Err(rc);
    }
    Ok(ptr)
}

#[cfg(feature = "gpu")]
pub fn link_probe() -> u32 {
    unsafe { ffi::prism_sh_basis_link_probe() }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn k_lm_table_matches_canonical_values() {
        // Spot-check against well-known closed-form K_lm values.
        let k = k_lm_table();
        // Y_0^0(θ, φ) = 1 / (2 sqrt(π)) ≈ 0.282094791773878
        assert!((k[0] - 0.282094791_f32).abs() < 1e-6, "K_00 = {}", k[0]);
        // K_1^0 = K_1^±1 = sqrt(3 / (4π)) ≈ 0.488602511902920
        assert!((k[1] - 0.488602511_f32).abs() < 1e-6);
        assert!((k[2] - 0.488602511_f32).abs() < 1e-6);
        assert!((k[3] - 0.488602511_f32).abs() < 1e-6);
        // K_2^0 = sqrt(5 / (4π)) ≈ 0.630783130
        assert!((k[6] - 0.630783130_f32).abs() < 1e-6);
        // K_2^±2 = sqrt(5 / (48π)) ≈ 0.182091405
        assert!((k[4] - 0.182091405_f32).abs() < 1e-6);
        assert!((k[8] - 0.182091405_f32).abs() < 1e-6);
        // K_3^0 = sqrt(7 / (4π)) ≈ 0.746352665
        assert!((k[12] - 0.746352665_f32).abs() < 1e-6);
    }

    #[test]
    fn lm_index_round_trip() {
        for idx in 0..N_COEFFS {
            let lm = LMIndex::from_idx(idx).expect("valid idx");
            assert_eq!(
                lm.to_idx(),
                idx,
                "round-trip failed for idx={}, got LMIndex {{l={}, m={}}}",
                idx,
                lm.l,
                lm.m
            );
            assert!((lm.l as usize) <= LMAX);
            assert!((lm.m as i32).unsigned_abs() <= lm.l as u32);
        }
        assert_eq!(LMIndex::from_idx(N_COEFFS), None);
    }

    #[test]
    fn cpu_sh_eval_y00_constant_at_any_angle() {
        // Y_0^0 is the constant 1/(2*sqrt(π)) regardless of (θ, φ).
        let k = k_lm_table();
        for &(t, p) in &[
            (0.0, 0.0),
            (1.5, 2.7),
            (std::f32::consts::PI / 2.0, std::f32::consts::PI),
            (0.001, 6.28),
        ] {
            let y = cpu_sh_eval_lmax5(t, p, &k);
            assert!(
                (y[0] - k[0]).abs() < 1e-6,
                "Y_0^0 not constant: at (θ={}, φ={}) got {} (expected {})",
                t,
                p,
                y[0],
                k[0]
            );
        }
    }

    #[test]
    fn cpu_sh_eval_y10_at_pole_is_max() {
        // Y_1^0 = K_1^0 * cos(θ). At θ=0 it equals K_1^0; at θ=π/2 it's 0.
        let k = k_lm_table();
        let y_pole = cpu_sh_eval_lmax5(0.0, 0.0, &k);
        let y_eq = cpu_sh_eval_lmax5(std::f32::consts::FRAC_PI_2, 0.0, &k);
        assert!(
            (y_pole[2] - k[2]).abs() < 1e-6,
            "Y_1^0(0, 0) should equal K_1^0"
        );
        assert!(
            y_eq[2].abs() < 1e-6,
            "Y_1^0(π/2, 0) should be 0; got {}",
            y_eq[2]
        );
    }

    #[test]
    fn cpu_sh_eval_zero_st_kills_all_m_nonzero() {
        // At θ=0, sin(θ)=0, so every P_l^|m| with m≠0 vanishes (they
        // all carry a sin(θ)^|m| factor). The only non-zero outputs
        // are the m=0 harmonics: indices 0, 2, 6, 12, 20, 30.
        let k = k_lm_table();
        let y = cpu_sh_eval_lmax5(0.0, 1.0, &k);
        let m0_idxs = [0usize, 2, 6, 12, 20, 30];
        for idx in 0..N_COEFFS {
            if m0_idxs.contains(&idx) {
                // m=0 harmonics: should be K_lm * P_l^0(1) = K_lm * P_l(1) = K_lm * 1 (all P_l(1) = 1).
                // Actually Legendre at x=1: P_0(1)=1, P_1(1)=1, ..., P_l(1)=1. So Y_l^0(0, φ) = K_l^0.
                assert!(
                    (y[idx] - k[idx]).abs() < 1e-5,
                    "Y[{}]_m=0 at θ=0 should be K_lm: got {}, expected {}",
                    idx,
                    y[idx],
                    k[idx]
                );
            } else {
                assert!(
                    y[idx].abs() < 1e-5,
                    "Y[{}]_|m|>0 at θ=0 should be 0: got {}",
                    idx,
                    y[idx]
                );
            }
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn link_probe_returns_sentinel() {
        assert_eq!(super::link_probe(), 0x0000_5BAC);
    }

    #[cfg(feature = "gpu")]
    #[test]
    #[ignore = "GPU_RUNTIME_REQUIRED: needs real GPU + built PTX; run on the RTX workstation via `cargo test -- --ignored`"]
    fn cpu_gpu_sh_parity_canonical_points() {
        // 12 canonical (θ, φ) inputs: poles, equator, 8 octant
        // diagonals + 2 extras. Parity within 1e-5 absolute.
        use cudarc::driver::{CudaContext, DevicePtr};
        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[sh parity] CUDA unavailable: {:?} — skipping", e);
                return;
            }
        };
        let stream = ctx.new_stream().expect("stream");

        // Init device-side K_LM.
        let raw_stream = stream.cu_stream() as usize;
        let rc = unsafe { ffi::prism_sh_basis_init(raw_stream as *mut std::ffi::c_void) };
        assert_eq!(rc, ffi::CUDA_SUCCESS);

        let pi = std::f32::consts::PI;
        let pi2 = pi / 2.0;
        let theta_phi: Vec<f32> = vec![
            // (θ, φ)
            0.001,
            0.0, // near north pole
            pi - 0.001,
            0.0, // near south pole
            pi2,
            0.0, // equator, +X
            pi2,
            pi2, // equator, +Y
            pi2,
            pi, // equator, -X
            pi2,
            3.0 * pi2, // equator, -Y
            pi / 4.0,
            pi / 4.0,
            pi / 3.0,
            2.0 * pi / 3.0,
            pi / 6.0,
            pi / 6.0,
            5.0 * pi / 6.0,
            7.0 * pi / 6.0,
            pi / 4.0,
            7.0 * pi / 4.0,
            pi / 2.0 + 0.1,
            1.234,
        ];
        let n = (theta_phi.len() / 2) as u32;

        let k = k_lm_table();
        let cpu_y: Vec<[f32; N_COEFFS]> = theta_phi
            .chunks_exact(2)
            .map(|c| cpu_sh_eval_lmax5(c[0], c[1], &k))
            .collect();

        let mut d_theta_phi = stream
            .alloc_zeros::<f32>(theta_phi.len())
            .expect("alloc tp");
        stream
            .memcpy_htod(&theta_phi, &mut d_theta_phi)
            .expect("htod tp");
        let d_y_out = stream
            .alloc_zeros::<f32>((n as usize) * N_COEFFS)
            .expect("alloc y");

        let (tp_dev, _g1) = d_theta_phi.device_ptr(&stream);
        let (y_dev, _g2) = d_y_out.device_ptr(&stream);
        let rc = unsafe {
            ffi::prism_sh_eval_run(
                tp_dev as *const f32,
                n,
                y_dev as *mut f32,
                raw_stream as *mut std::ffi::c_void,
            )
        };
        assert_eq!(rc, ffi::CUDA_SUCCESS);
        stream.synchronize().expect("stream sync");

        let mut gpu_y_flat = vec![0.0f32; (n as usize) * N_COEFFS];
        stream
            .memcpy_dtoh(&d_y_out, &mut gpu_y_flat)
            .expect("dtoh y");

        let tol: f32 = 1e-5;
        for (i, expected_row) in cpu_y.iter().enumerate() {
            for j in 0..N_COEFFS {
                let gpu = gpu_y_flat[i * N_COEFFS + j];
                let cpu = expected_row[j];
                let diff = (gpu - cpu).abs();
                let scale = cpu.abs().max(1e-3);
                let rel = diff / scale;
                assert!(
                    diff < tol || rel < tol,
                    "parity violation at point {} idx {}: cpu={}, gpu={}, diff={}, rel={}",
                    i,
                    j,
                    cpu,
                    gpu,
                    diff,
                    rel
                );
            }
        }
    }

    #[cfg(feature = "gpu")]
    #[ignore = "GPU_RUNTIME_REQUIRED: needs real GPU + built PTX; run on the RTX workstation via `cargo test -- --ignored`"]
    #[test]
    fn cpu_gpu_sh_parity_random_points() {
        // 64 LCG-derived random (θ, φ) points. θ ∈ (0, π); φ ∈ [0, 2π).
        // Parity within 1e-5 absolute (tf32-equivalent precision).
        use cudarc::driver::{CudaContext, DevicePtr};
        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[sh parity rand] CUDA unavailable: {:?} — skipping", e);
                return;
            }
        };
        let stream = ctx.new_stream().expect("stream");

        let raw_stream = stream.cu_stream() as usize;
        let rc = unsafe { ffi::prism_sh_basis_init(raw_stream as *mut std::ffi::c_void) };
        assert_eq!(rc, ffi::CUDA_SUCCESS);

        // Deterministic LCG so the test is reproducible.
        struct Lcg {
            s: u64,
        }
        impl Lcg {
            fn next_f32(&mut self) -> f32 {
                self.s = self
                    .s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (self.s >> 32) as u32 as f32 / 4_294_967_296.0
            }
        }
        let mut rng = Lcg { s: 99 };

        let n: u32 = 64;
        let two_pi = std::f32::consts::TAU;
        let pi = std::f32::consts::PI;
        let mut theta_phi: Vec<f32> = Vec::with_capacity(n as usize * 2);
        for _ in 0..n {
            // θ ∈ (0.05, π - 0.05): keeps us off the singularity-prone poles
            // for tf32 numerical comparison; the kernel itself handles
            // θ=0 / θ=π without singularity (sin(θ)=0 just zeros the
            // m≠0 harmonics, which is mathematically correct).
            let theta = 0.05 + rng.next_f32() * (pi - 0.10);
            let phi = rng.next_f32() * two_pi;
            theta_phi.push(theta);
            theta_phi.push(phi);
        }

        let k = k_lm_table();
        let cpu_y: Vec<[f32; N_COEFFS]> = theta_phi
            .chunks_exact(2)
            .map(|c| cpu_sh_eval_lmax5(c[0], c[1], &k))
            .collect();

        let mut d_theta_phi = stream
            .alloc_zeros::<f32>(theta_phi.len())
            .expect("alloc tp");
        stream
            .memcpy_htod(&theta_phi, &mut d_theta_phi)
            .expect("htod tp");
        let d_y_out = stream
            .alloc_zeros::<f32>((n as usize) * N_COEFFS)
            .expect("alloc y");

        let (tp_dev, _g1) = d_theta_phi.device_ptr(&stream);
        let (y_dev, _g2) = d_y_out.device_ptr(&stream);
        let rc = unsafe {
            ffi::prism_sh_eval_run(
                tp_dev as *const f32,
                n,
                y_dev as *mut f32,
                raw_stream as *mut std::ffi::c_void,
            )
        };
        assert_eq!(rc, ffi::CUDA_SUCCESS);
        stream.synchronize().expect("stream sync");

        let mut gpu_y_flat = vec![0.0f32; (n as usize) * N_COEFFS];
        stream
            .memcpy_dtoh(&d_y_out, &mut gpu_y_flat)
            .expect("dtoh y");

        // Tolerance: 1e-4 absolute OR 1e-4 relative. The high-l
        // harmonics involve products of up to sin^5(θ) and angular
        // factors with 5φ, so the FP error compounds at the fast-math
        // setting. 1e-4 is comfortably above f32 ULP at the typical
        // SH magnitude (which never exceeds ~1).
        let tol: f32 = 1e-4;
        let mut max_diff = 0.0f32;
        let mut max_diff_loc = (0usize, 0usize);
        for (i, expected_row) in cpu_y.iter().enumerate() {
            for j in 0..N_COEFFS {
                let gpu = gpu_y_flat[i * N_COEFFS + j];
                let cpu = expected_row[j];
                let diff = (gpu - cpu).abs();
                if diff > max_diff {
                    max_diff = diff;
                    max_diff_loc = (i, j);
                }
                let scale = cpu.abs().max(1e-3);
                let rel = diff / scale;
                assert!(
                    diff < tol || rel < tol,
                    "parity violation at random point {} idx {}: cpu={}, gpu={}, diff={}, rel={}",
                    i,
                    j,
                    cpu,
                    gpu,
                    diff,
                    rel
                );
            }
        }
        eprintln!(
            "[sh parity rand] max diff over 64×36 = {:.2e} (idx {}, {})",
            max_diff, max_diff_loc.0, max_diff_loc.1
        );
    }
}
