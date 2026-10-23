//! M1.2.17 — Hamiltonian Auditor + SFA stability fuse validation.
//!
//! Tests the end-to-end energy-monitor capture path:
//!   * CUB DeviceReduce::Sum on a synthetic n-atom f64 PE buffer.
//!   * Window-update kernel rolls EnergyWindow {prev, cur} and writes
//!     V_t into a target f64 (mimicking adj.d_potential_energy at
//!     offset 112).
//!
//! Math check: synthetic per-atom values 1..n give sum = n(n+1)/2.
//! For n = 100 → expected sum = 5050.0 (bit-exact at f64).

#![cfg(feature = "gpu")]

use cudarc::driver::{CudaContext, DevicePtr};
use std::ffi::c_void;

// Force the test target to depend on the prism_nhs lib crate, which
// pulls in the build.rs's `rustc-link-lib=static=energy_monitor`
// directive.  Without this `use`, the integration test would compile
// independently of the lib and the linker wouldn't know about the
// energy_monitor static archive.
#[allow(unused_imports)]
use prism_nhs::gearbox::ChronometricStateTensor as _ForceLinkage;

extern "C" {
    fn prism_energy_monitor_temp_storage_bytes(n: u32, out_temp_bytes: *mut usize) -> i32;

    fn prism_energy_monitor_launch_reduce(
        d_pe_components: *const f64,
        n: u32,
        d_temp_storage: *mut c_void,
        temp_storage_bytes: usize,
        d_pe_scalar: *mut f64,
        d_energy_window: *mut c_void,
        d_adj_pe_target: *mut f64,
        stream: *mut c_void,
    ) -> i32;
}

#[test]
fn m1217_energy_monitor_sum_1_to_n_is_n_n_plus_1_div_2() {
    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[M1.2.17 H] CUDA unavailable: {:?} — skipping", e);
            return;
        }
    };
    let stream = ctx.new_stream().expect("stream");
    let raw_stream = stream.cu_stream() as *mut c_void;

    // Synthetic per-atom PE: values 1.0, 2.0, …, N.0 (N = 100).
    // Closed-form sum = N(N+1)/2 = 5050.0.
    const N: u32 = 100;
    let host_pe: Vec<f64> = (1..=N as usize).map(|i| i as f64).collect();
    let expected_sum: f64 = (N as f64) * ((N as f64) + 1.0) / 2.0;

    let mut d_pe: cudarc::driver::CudaSlice<f64> =
        stream.alloc_zeros::<f64>(host_pe.len()).expect("alloc pe");
    stream.memcpy_htod(&host_pe, &mut d_pe).expect("htod pe");

    // Query CUB temp storage size.
    let mut temp_bytes: usize = 0;
    let rc = unsafe { prism_energy_monitor_temp_storage_bytes(N, &mut temp_bytes as *mut usize) };
    assert_eq!(rc, 0, "temp_storage_bytes rc={}", rc);
    assert!(temp_bytes > 0, "CUB returned zero temp storage size");

    let mut d_temp: cudarc::driver::CudaSlice<u8> =
        stream.alloc_zeros::<u8>(temp_bytes).expect("alloc temp");
    let mut d_pe_scalar: cudarc::driver::CudaSlice<f64> =
        stream.alloc_zeros::<f64>(1).expect("alloc pe_scalar");
    let mut d_window: cudarc::driver::CudaSlice<u8> =
        stream.alloc_zeros::<u8>(16).expect("alloc window"); // EnergyWindow = 16 B
    let mut d_adj_pe: cudarc::driver::CudaSlice<f64> =
        stream.alloc_zeros::<f64>(1).expect("alloc adj pe");

    let (d_pe_addr, d_temp_addr, d_scalar_addr, d_window_addr, d_adj_addr) = {
        let (a, _g1) = d_pe.device_ptr(&stream);
        let (b, _g2) = d_temp.device_ptr(&stream);
        let (c, _g3) = d_pe_scalar.device_ptr(&stream);
        let (d, _g4) = d_window.device_ptr(&stream);
        let (e, _g5) = d_adj_pe.device_ptr(&stream);
        (a as u64, b as u64, c as u64, d as u64, e as u64)
    };

    let rc = unsafe {
        prism_energy_monitor_launch_reduce(
            d_pe_addr as *const f64,
            N,
            d_temp_addr as *mut c_void,
            temp_bytes,
            d_scalar_addr as *mut f64,
            d_window_addr as *mut c_void,
            d_adj_addr as *mut f64,
            raw_stream,
        )
    };
    assert_eq!(rc, 0, "launch_reduce rc={}", rc);
    stream.synchronize().expect("post-reduce sync");

    // Read scalar back.
    let mut readback = [0.0f64; 1];
    stream
        .memcpy_dtoh(&d_pe_scalar, &mut readback)
        .expect("dtoh scalar");
    eprintln!(
        "[M1.2.17 H] CUB reduce on 1..{} → V_t = {} (expected {})",
        N, readback[0], expected_sum
    );
    assert!(
        (readback[0] - expected_sum).abs() < 1e-6,
        "CUB reduce sum mismatch: got {}, expected {}",
        readback[0],
        expected_sum
    );

    // EnergyWindow: prev = 0.0 (first launch), cur = V_t.
    let mut window_back = [0u8; 16];
    stream
        .memcpy_dtoh(&d_window, &mut window_back)
        .expect("dtoh window");
    let prev = f64::from_le_bytes(window_back[0..8].try_into().unwrap());
    let cur = f64::from_le_bytes(window_back[8..16].try_into().unwrap());
    assert_eq!(
        prev, 0.0,
        "first launch: window.prev MUST be 0.0; got {}",
        prev
    );
    assert!(
        (cur - expected_sum).abs() < 1e-6,
        "window.cur mismatch: got {}, expected {}",
        cur,
        expected_sum
    );

    // adj.d_potential_energy target should also hold V_t.
    let mut adj_back = [0.0f64; 1];
    stream
        .memcpy_dtoh(&d_adj_pe, &mut adj_back)
        .expect("dtoh adj");
    assert!(
        (adj_back[0] - expected_sum).abs() < 1e-6,
        "adj.d_potential_energy mismatch: got {}, expected {}",
        adj_back[0],
        expected_sum
    );

    eprintln!(
        "[M1.2.17 H GATE PASS] CUB f64 reduce {} = {} bit-exact; \
         EnergyWindow rolled (prev=0, cur={}); \
         adj.d_potential_energy = {} (matches V_t)",
        N, readback[0], cur, adj_back[0]
    );
}

#[test]
fn m1217_energy_monitor_two_pass_window_roll() {
    // Verify the window roll on the SECOND launch: prev should become
    // the previous V_t, cur should be the new V_t.
    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[M1.2.17 H two-pass] CUDA unavailable: {:?} — skipping", e);
            return;
        }
    };
    let stream = ctx.new_stream().expect("stream");
    let raw_stream = stream.cu_stream() as *mut c_void;

    const N: u32 = 10;
    let mut d_pe: cudarc::driver::CudaSlice<f64> =
        stream.alloc_zeros::<f64>(N as usize).expect("alloc pe");

    let mut temp_bytes: usize = 0;
    unsafe {
        prism_energy_monitor_temp_storage_bytes(N, &mut temp_bytes as *mut usize);
    }
    let mut d_temp: cudarc::driver::CudaSlice<u8> =
        stream.alloc_zeros::<u8>(temp_bytes).expect("alloc temp");
    let mut d_pe_scalar: cudarc::driver::CudaSlice<f64> =
        stream.alloc_zeros::<f64>(1).expect("alloc pe_scalar");
    let mut d_window: cudarc::driver::CudaSlice<u8> =
        stream.alloc_zeros::<u8>(16).expect("alloc window");
    let mut d_adj_pe: cudarc::driver::CudaSlice<f64> =
        stream.alloc_zeros::<f64>(1).expect("alloc adj pe");

    let (d_pe_addr, d_temp_addr, d_scalar_addr, d_window_addr, d_adj_addr) = {
        let (a, _g1) = d_pe.device_ptr(&stream);
        let (b, _g2) = d_temp.device_ptr(&stream);
        let (c, _g3) = d_pe_scalar.device_ptr(&stream);
        let (d, _g4) = d_window.device_ptr(&stream);
        let (e, _g5) = d_adj_pe.device_ptr(&stream);
        (a as u64, b as u64, c as u64, d as u64, e as u64)
    };

    // Pass 1: PE = all 1.0 → sum = 10.0.
    let pass1_pe: Vec<f64> = vec![1.0; N as usize];
    stream
        .memcpy_htod(&pass1_pe, &mut d_pe)
        .expect("htod pe pass 1");
    unsafe {
        prism_energy_monitor_launch_reduce(
            d_pe_addr as *const f64,
            N,
            d_temp_addr as *mut c_void,
            temp_bytes,
            d_scalar_addr as *mut f64,
            d_window_addr as *mut c_void,
            d_adj_addr as *mut f64,
            raw_stream,
        );
    }
    stream.synchronize().expect("pass 1 sync");

    let mut window_pass1 = [0u8; 16];
    stream
        .memcpy_dtoh(&d_window, &mut window_pass1)
        .expect("dtoh window 1");
    let prev_1 = f64::from_le_bytes(window_pass1[0..8].try_into().unwrap());
    let cur_1 = f64::from_le_bytes(window_pass1[8..16].try_into().unwrap());
    eprintln!("[M1.2.17 H 2-pass] PASS 1: prev={} cur={}", prev_1, cur_1);
    assert_eq!(prev_1, 0.0);
    assert!((cur_1 - 10.0).abs() < 1e-9);

    // Pass 2: PE = all 2.0 → sum = 20.0.  The window roll should set
    // prev = 10.0 (the previous cur) and cur = 20.0.
    let pass2_pe: Vec<f64> = vec![2.0; N as usize];
    stream
        .memcpy_htod(&pass2_pe, &mut d_pe)
        .expect("htod pe pass 2");
    unsafe {
        prism_energy_monitor_launch_reduce(
            d_pe_addr as *const f64,
            N,
            d_temp_addr as *mut c_void,
            temp_bytes,
            d_scalar_addr as *mut f64,
            d_window_addr as *mut c_void,
            d_adj_addr as *mut f64,
            raw_stream,
        );
    }
    stream.synchronize().expect("pass 2 sync");

    let mut window_pass2 = [0u8; 16];
    stream
        .memcpy_dtoh(&d_window, &mut window_pass2)
        .expect("dtoh window 2");
    let prev_2 = f64::from_le_bytes(window_pass2[0..8].try_into().unwrap());
    let cur_2 = f64::from_le_bytes(window_pass2[8..16].try_into().unwrap());
    eprintln!("[M1.2.17 H 2-pass] PASS 2: prev={} cur={}", prev_2, cur_2);
    assert!(
        (prev_2 - 10.0).abs() < 1e-9,
        "window.prev should be PASS-1 cur (10.0); got {}",
        prev_2
    );
    assert!(
        (cur_2 - 20.0).abs() < 1e-9,
        "window.cur should be PASS-2 sum (20.0); got {}",
        cur_2
    );

    // Drift = |20 − 10| / |10| = 1.0 = 100% — would trigger Gear 3
    // when consumed by the SFA stability fuse.
    let drift = (cur_2 - prev_2).abs() / prev_2.abs();
    eprintln!(
        "[M1.2.17 H 2-pass GATE PASS] EnergyWindow rolled correctly; \
         synthetic drift {}% (would trigger SFA stability fuse Gear 3 trap \
         since drift > 1%)",
        drift * 100.0
    );
}
