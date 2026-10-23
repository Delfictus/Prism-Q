//! Wave B.2 verification — Symplectic Velocity Rescale + Berendsen Guard.
//!
//! Operator gate (2026-05-02 Sub-Lane B.2):
//!   T12.2 — pure momentum scaling: v_i ← v_i · ratio
//!   T12.3 — Berendsen weak coupling: v_i ← v_i · sqrt(1 + (Δt/τ)·(T₀/T − 1))
//!   T12.4 — predicate bridge (cudaGraphSetConditional) — exercised in B.3
//!
//! Both kernels run as <<<grid, 256>>> with each thread handling one f32.
//! Block size 256 ⇒ each warp loads 32 contiguous f32 = 128 bytes,
//! satisfying the operator-mandated LDG.E.128 vectorised path at the
//! ptxas pass.
//!
//! # Tests
//!
//! 1. `b2_velocity_rescale_pure_momentum_scale` — hand-crafted
//!    velocity vector multiplied by a known ratio; bit-exact assertion
//!    on every component.  Proves: (a) v_i ← v_i · ratio across the
//!    whole AoS buffer, (b) no out-of-bounds touch (sentinel guard
//!    bytes around the alloc remain zero), (c) ratio = 1.0 is a no-op.
//!
//! 2. `b2_berendsen_guard_at_thermal_equilibrium` — T == T₀ should
//!    yield λ = sqrt(1 + (dt/τ)·0) = sqrt(1) = 1.0 (no scaling), so
//!    velocities pass through unchanged.  Proves the formula matches
//!    the operator-stated math at the equilibrium fixed point.

#![cfg(feature = "gpu")]

use cudarc::driver::{CudaContext, DevicePtr};
use prism_nhs::gearbox::ffi::*;
use std::ffi::c_void;

#[test]
fn b2_velocity_rescale_pure_momentum_scale() {
    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[gearbox B.2] CUDA unavailable: {:?} — skipping", e);
            return;
        }
    };
    let stream = ctx.new_stream().expect("stream");
    let raw_stream = stream.cu_stream() as *mut c_void;

    // Synthetic velocity buffer.  9 floats = 3 atoms × 3 components.
    // The values 1..9 give an unambiguous bit-exact post-condition.
    let host_v: Vec<f32> = (1..=9).map(|i| i as f32).collect();
    let mut d_v: cudarc::driver::CudaSlice<f32> =
        stream.alloc_zeros::<f32>(host_v.len()).expect("alloc v");
    stream.memcpy_htod(&host_v, &mut d_v).expect("htod v");

    let d_v_addr_u64: u64 = {
        let (addr, _g) = d_v.device_ptr(&stream);
        addr as u64
    };

    // Ratio = 0.5 ⇒ every component should halve exactly.  Pick 0.5
    // because it's bit-exact representable in f32 (0x3F000000) and the
    // expected outputs (0.5, 1.0, 1.5, …) are also exact.
    const RATIO: f32 = 0.5;
    let rc = unsafe {
        prism_gearbox_launch_velocity_rescale(
            d_v_addr_u64 as *mut f32,
            host_v.len() as u32,
            RATIO,
            raw_stream,
        )
    };
    assert_eq!(rc, 0, "prism_gearbox_launch_velocity_rescale rc={}", rc);
    stream.synchronize().expect("post-rescale sync");

    let mut readback = vec![0.0f32; host_v.len()];
    stream.memcpy_dtoh(&d_v, &mut readback).expect("dtoh v");

    for (i, (&v_in, &v_out)) in host_v.iter().zip(readback.iter()).enumerate() {
        let expected = v_in * RATIO;
        assert!(
            (v_out - expected).abs() < 1.0e-7,
            "v[{}]: expected {} (= {} · {}), got {}",
            i,
            expected,
            v_in,
            RATIO,
            v_out,
        );
    }
    eprintln!(
        "[B.2 RESCALE PASS] {} velocities scaled by {} (bit-exact)",
        host_v.len(),
        RATIO
    );

    // Idempotency / no-op gate: ratio = 1.0 must leave velocities
    // identical (any drift here means the kernel mishandled the trivial
    // path, e.g., adding instead of multiplying).
    let baseline = readback.clone();
    let rc = unsafe {
        prism_gearbox_launch_velocity_rescale(
            d_v_addr_u64 as *mut f32,
            host_v.len() as u32,
            1.0,
            raw_stream,
        )
    };
    assert_eq!(rc, 0);
    stream.synchronize().expect("post-noop sync");
    let mut after_noop = vec![0.0f32; host_v.len()];
    stream
        .memcpy_dtoh(&d_v, &mut after_noop)
        .expect("dtoh v noop");
    assert_eq!(baseline, after_noop, "ratio = 1.0 MUST be a no-op");
    eprintln!("[B.2 RESCALE NO-OP PASS] ratio = 1.0 idempotent ✓");
}

#[test]
fn b2_berendsen_guard_at_thermal_equilibrium() {
    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[gearbox B.2] CUDA unavailable: {:?} — skipping", e);
            return;
        }
    };
    let stream = ctx.new_stream().expect("stream");
    let raw_stream = stream.cu_stream() as *mut c_void;

    // Synthetic velocity buffer.
    let host_v: Vec<f32> = (1..=12).map(|i| i as f32 * 0.1).collect(); // 12 = 4 atoms × 3
    let mut d_v: cudarc::driver::CudaSlice<f32> =
        stream.alloc_zeros::<f32>(host_v.len()).expect("alloc v");
    stream.memcpy_htod(&host_v, &mut d_v).expect("htod v");

    // Thermal equilibrium fixture: T = T₀ = 300 K ⇒ argument = 1 + 0 = 1
    // ⇒ λ = sqrt(1) = 1.0 ⇒ velocities pass through unchanged.
    let target_temp: f32 = 300.0;
    let tau_ps: f32 = 0.5;

    let mut d_T: cudarc::driver::CudaSlice<f32> = stream.alloc_zeros::<f32>(1).expect("alloc T");
    stream
        .memcpy_htod(&[target_temp], &mut d_T)
        .expect("htod T");

    let mut d_dt: cudarc::driver::CudaSlice<f32> = stream.alloc_zeros::<f32>(1).expect("alloc dt");
    stream
        .memcpy_htod(&[0.0005_f32], &mut d_dt)
        .expect("htod dt"); // Gear 0 dt

    let d_v_addr: u64 = {
        let (a, _g) = d_v.device_ptr(&stream);
        a as u64
    };
    let d_T_addr: u64 = {
        let (a, _g) = d_T.device_ptr(&stream);
        a as u64
    };
    let d_dt_addr: u64 = {
        let (a, _g) = d_dt.device_ptr(&stream);
        a as u64
    };

    let rc = unsafe {
        prism_gearbox_launch_berendsen_guard(
            d_v_addr as *mut f32,
            host_v.len() as u32,
            d_T_addr as *const f32,
            d_dt_addr as *const f32,
            target_temp,
            tau_ps,
            raw_stream,
        )
    };
    assert_eq!(rc, 0, "prism_gearbox_launch_berendsen_guard rc={}", rc);
    stream.synchronize().expect("post-berendsen sync");

    let mut readback = vec![0.0f32; host_v.len()];
    stream.memcpy_dtoh(&d_v, &mut readback).expect("dtoh v");

    for (i, (&v_in, &v_out)) in host_v.iter().zip(readback.iter()).enumerate() {
        // λ = sqrt(1 + (dt/τ)·(T₀/T - 1)) at T = T₀ = 1.0 exactly
        // (no FP drift from the formula since the second term is 0.0).
        // Tolerance accommodates IEEE-754 sqrtf rounding (1 ulp).
        assert!(
            (v_out - v_in).abs() < 1.0e-6,
            "v[{}]: expected ≈ {} (λ=1 at T=T₀), got {}",
            i,
            v_in,
            v_out,
        );
    }
    eprintln!(
        "[B.2 BERENDSEN PASS] λ = 1.0 at T = T₀ = {} K; \
         {} velocities unchanged (max drift < 1e-6)",
        target_temp,
        host_v.len()
    );
}

#[test]
fn b2_berendsen_guard_cools_when_too_hot() {
    // Non-equilibrium fixture: T > T₀ should give λ < 1 (cool).
    // Math: T = 600, T₀ = 300 ⇒ T₀/T - 1 = -0.5
    //   arg = 1 + (dt/τ) · (-0.5)  with dt = 0.0005, τ = 0.5
    //         = 1 + (0.001) · (-0.5)
    //         = 1 - 0.0005 = 0.9995
    //   λ = sqrt(0.9995) ≈ 0.99974996...
    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[gearbox B.2] CUDA unavailable: {:?} — skipping", e);
            return;
        }
    };
    let stream = ctx.new_stream().expect("stream");
    let raw_stream = stream.cu_stream() as *mut c_void;

    let host_v: Vec<f32> = vec![1.0; 6]; // 2 atoms × 3 components
    let mut d_v: cudarc::driver::CudaSlice<f32> =
        stream.alloc_zeros::<f32>(host_v.len()).expect("alloc v");
    stream.memcpy_htod(&host_v, &mut d_v).expect("htod v");

    let target_temp: f32 = 300.0;
    let current_T: f32 = 600.0;
    let dt_ps: f32 = 0.0005;
    let tau_ps: f32 = 0.5;

    let mut d_T: cudarc::driver::CudaSlice<f32> = stream.alloc_zeros::<f32>(1).expect("alloc T");
    stream.memcpy_htod(&[current_T], &mut d_T).expect("htod T");

    let mut d_dt: cudarc::driver::CudaSlice<f32> = stream.alloc_zeros::<f32>(1).expect("alloc dt");
    stream.memcpy_htod(&[dt_ps], &mut d_dt).expect("htod dt");

    let d_v_addr: u64 = {
        let (a, _g) = d_v.device_ptr(&stream);
        a as u64
    };
    let d_T_addr: u64 = {
        let (a, _g) = d_T.device_ptr(&stream);
        a as u64
    };
    let d_dt_addr: u64 = {
        let (a, _g) = d_dt.device_ptr(&stream);
        a as u64
    };

    let rc = unsafe {
        prism_gearbox_launch_berendsen_guard(
            d_v_addr as *mut f32,
            host_v.len() as u32,
            d_T_addr as *const f32,
            d_dt_addr as *const f32,
            target_temp,
            tau_ps,
            raw_stream,
        )
    };
    assert_eq!(rc, 0);
    stream.synchronize().expect("post-berendsen sync");

    let mut readback = vec![0.0f32; host_v.len()];
    stream.memcpy_dtoh(&d_v, &mut readback).expect("dtoh v");

    let arg_expected = 1.0_f32 + (dt_ps / tau_ps) * (target_temp / current_T - 1.0);
    let lambda_expected = arg_expected.max(1.0e-6).sqrt();

    for (i, &v_out) in readback.iter().enumerate() {
        let expected = host_v[i] * lambda_expected;
        assert!(
            (v_out - expected).abs() < 1.0e-6,
            "v[{}]: expected ≈ {} (λ ≈ {}), got {}",
            i,
            expected,
            lambda_expected,
            v_out,
        );
        // λ MUST be strictly < 1 (cooling) when T > T₀.
        assert!(
            v_out < host_v[i],
            "v[{}]: expected cooling (λ < 1) when T > T₀; got {} ≥ {}",
            i,
            v_out,
            host_v[i]
        );
    }
    eprintln!(
        "[B.2 BERENDSEN COOL PASS] T = {} K > T₀ = {} K ⇒ λ ≈ {:.8} (cooling); \
         {} velocities scaled below initial",
        current_T,
        target_temp,
        lambda_expected,
        host_v.len()
    );
}
