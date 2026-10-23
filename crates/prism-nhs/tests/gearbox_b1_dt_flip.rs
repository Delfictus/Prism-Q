//! Wave B.1 verification gate — operator-mandated 2026-05-02.
//!
//! "A 'Hello-World' unit test … proving that a manual trigger correctly
//! flips the dt value in VRAM from 2.0fs to 4.0fs without host
//! intervention."  Operator subsequently mandated raw bit-pattern
//! evidence: 0x40000000 (2.0f) → 0x40800000 (4.0f) for the cruise
//! upshift, and last_burst_frame stamping on a Burst trigger.
//!
//! Lives in the integration-test target rather than the lib
//! `#[cfg(test)]` module because the lib test crate had pre-existing
//! E0063 breakage that this commit's surgical bin cleanup also
//! addressed.  Integration test target was the cleanest landing spot
//! for the gate so it isn't entangled with broader lib-test fixes.
//!
//! # Why a single test function
//!
//! The gearbox table lives in `__constant__` device memory — there is
//! exactly ONE copy per device, shared across all kernel launches.
//! Cargo runs `#[test]` functions in PARALLEL by default; if two
//! tests both `cudaMemcpyToSymbolAsync` the table, whichever runs
//! second wins and the other observes the wrong values.  The first
//! draft of this test split equilibrium and burst into separate
//! `#[test]` functions and hit exactly that race (the equilibrium
//! test read 0.004 from slot 8 instead of the overridden 4.0
//! because the burst test's table-overwrite wallpapered it).
//!
//! Solution: ONE `#[test]` that runs both scenarios sequentially on
//! the same stream.  Stream ordering plus single-test serialisation
//! eliminates the race entirely.

#![cfg(feature = "gpu")]

use cudarc::driver::{CudaContext, DevicePtr};
use prism_nhs::gearbox::{default_gearbox_table, ffi::*, ChronometricStateTensor};
use std::ffi::c_void;

#[test]
fn b1_pointer_swap_bit_pattern_flip_and_burst_state_machine() {
    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[gearbox B.1] CUDA unavailable: {:?} — skipping", e);
            return;
        }
    };
    let stream = ctx.new_stream().expect("stream");
    let raw_stream = stream.cu_stream() as *mut c_void;

    // ── Custom gearbox table: synthetic-gate values pinned to the
    //    operator-mandated bit patterns.
    //      Gear 0 (slot 0)  = 0.5f  (= 0x3F000000)  — burst target
    //      Gear 1 (slot 4)  = 2.0f  (= 0x40000000)  — initial dt seed
    //      Gear 2 (slot 8)  = 4.0f  (= 0x40800000)  — equilibrium upshift
    //      Gear 3 (slot 12) = NaN                   — abort sentinel
    //    Production gearbox table uses ps values (0.0005 / 0.002 /
    //    0.004); we override here purely for bit-pattern legibility.
    let mut host_table = default_gearbox_table();
    host_table[0] = 0.5_f32;
    host_table[4] = 2.0_f32;
    host_table[8] = 4.0_f32;
    let rc = unsafe { prism_gearbox_init_table_async(host_table.as_ptr(), raw_stream) };
    assert_eq!(rc, 0, "prism_gearbox_init_table_async returned {}", rc);
    stream.synchronize().expect("post-init sync");

    // ── Synthetic dt buffer.  Pre-seed with 2.0f (0x40000000). ──
    let mut d_dt: cudarc::driver::CudaSlice<f32> = stream.alloc_zeros::<f32>(1).expect("alloc dt");
    let initial: [f32; 1] = [2.0_f32];
    stream.memcpy_htod(&initial, &mut d_dt).expect("htod dt");
    let d_dt_addr_u64: u64 = {
        let (addr, _guard) = d_dt.device_ptr(&stream);
        addr as u64
    };

    // Pre-state bit-pattern audit.
    let mut readback = [0.0f32; 1];
    stream.memcpy_dtoh(&d_dt, &mut readback).expect("dtoh pre");
    let pre_bits = readback[0].to_bits();
    eprintln!(
        "[B.1 PRE]  d_dt = {:.6}f (bit-pattern 0x{:08X})",
        readback[0], pre_bits
    );
    assert_eq!(
        pre_bits, 0x40000000,
        "pre-state bit pattern MUST be 0x40000000 (= 2.0f32); got 0x{:08X}",
        pre_bits
    );

    // ── Synthetic Adjudicator FFI struct (128 bytes). ──
    // adjudication_code at offset 52 (u32); d_dt at offset 112 (*mut f32).
    let mut d_adj: cudarc::driver::CudaSlice<u8> =
        stream.alloc_zeros::<u8>(128).expect("alloc adj");
    let mut host_adj = [0u8; 128];
    // M1.2.17 layout pivot: d_dt moved from offset 112 → 120 (offset
    // 112 is now the f64 d_potential_energy VALUE).
    host_adj[120..128].copy_from_slice(&d_dt_addr_u64.to_le_bytes());

    // ── ChronometricStateTensor (16 B) ──
    let mut d_cruise: cudarc::driver::CudaSlice<u8> =
        stream.alloc_zeros::<u8>(32).expect("alloc cruise");

    // ════════════════════════════════════════════════════════════════
    // SCENARIO A — Equilibrium upshift to Gear 2.
    //   adj_code = 0; cruise.counter pre-seed = 500 (THRESHOLD); kernel
    //   increments to 501 ⇒ selects Gear 2 ⇒ writes d_gearbox_table[8]
    //   into *(adj->d_dt).  Bit-pattern flip: 0x40000000 → 0x40800000.
    // ════════════════════════════════════════════════════════════════
    {
        host_adj[52..56].copy_from_slice(&0u32.to_le_bytes()); // code = 0
        stream
            .memcpy_htod(&host_adj, &mut d_adj)
            .expect("htod adj A");

        let cruise_seed = ChronometricStateTensor {
            counter: 500,
            last_burst_frame: 0,
            current_gear: 1,
            previous_gear: 1,
            v_prev: 0.0,
            _pad_v_prev: 0,
        };
        let cruise_bytes: [u8; 32] =
            unsafe { std::mem::transmute::<ChronometricStateTensor, [u8; 32]>(cruise_seed) };
        stream
            .memcpy_htod(&cruise_bytes, &mut d_cruise)
            .expect("htod cruise A");

        let (adj_addr, cruise_addr) = {
            let (a, _ga) = d_adj.device_ptr(&stream);
            let (c, _gc) = d_cruise.device_ptr(&stream);
            (a as u64, c as u64)
        };
        let rc = unsafe {
            prism_gearbox_launch_pointer_swap(
                adj_addr as *const _,
                cruise_addr as *mut _,
                /*current_frame=*/ 0,
                raw_stream,
            )
        };
        assert_eq!(rc, 0, "[A] PointerSwap launch returned {}", rc);
        stream.synchronize().expect("[A] post-kernel sync");

        // Bit-pattern audit.
        stream
            .memcpy_dtoh(&d_dt, &mut readback)
            .expect("[A] dtoh dt");
        let post_bits = readback[0].to_bits();
        eprintln!(
            "[B.1 EQ POST] d_dt = {:.6}f (bit-pattern 0x{:08X})",
            readback[0], post_bits
        );
        assert_eq!(
            post_bits, 0x40800000,
            "[A] dt-flip GATE FAIL: expected 0x40800000 (= 4.0f32, Gear 2); got 0x{:08X}",
            post_bits
        );
        assert_ne!(
            pre_bits, post_bits,
            "[A] pre and post bit patterns identical"
        );

        // Cruise state-machine audit.
        let mut cruise_back = [0u8; 32];
        stream
            .memcpy_dtoh(&d_cruise, &mut cruise_back)
            .expect("[A] dtoh cruise");
        let cruise_after: ChronometricStateTensor =
            unsafe { std::mem::transmute::<[u8; 32], ChronometricStateTensor>(cruise_back) };
        assert_eq!(cruise_after.counter, 501);
        assert_eq!(cruise_after.current_gear, 2);

        eprintln!(
            "[B.1 EQ PASS] bit-exact swap 0x{:08X} (2.0f) → 0x{:08X} (4.0f); \
             cruise counter 500 → 501; gear=2",
            pre_bits, post_bits
        );
    }

    // ════════════════════════════════════════════════════════════════
    // SCENARIO B — Burst trigger to Gear 0.
    //   adj_code = 1; cruise.counter pre-seed = 1234 (any non-zero);
    //   kernel resets counter to 0, stamps last_burst_frame, selects
    //   Gear 0 ⇒ writes d_gearbox_table[0] into *(adj->d_dt).
    //   Bit-pattern flip: 0x40800000 → 0x3F000000.
    // ════════════════════════════════════════════════════════════════
    {
        let pre_burst_bits = readback[0].to_bits();
        assert_eq!(
            pre_burst_bits, 0x40800000,
            "[B] expected pre-state to be 0x40800000 (post-A); got 0x{:08X}",
            pre_burst_bits
        );

        host_adj[52..56].copy_from_slice(&1u32.to_le_bytes()); // code = 1 (Burst)
        stream
            .memcpy_htod(&host_adj, &mut d_adj)
            .expect("htod adj B");

        let cruise_seed = ChronometricStateTensor {
            counter: 1234,
            last_burst_frame: 0,
            current_gear: 2,
            previous_gear: 1,
            v_prev: 0.0,
            _pad_v_prev: 0,
        };
        let cruise_bytes: [u8; 32] =
            unsafe { std::mem::transmute::<ChronometricStateTensor, [u8; 32]>(cruise_seed) };
        stream
            .memcpy_htod(&cruise_bytes, &mut d_cruise)
            .expect("htod cruise B");

        let (adj_addr, cruise_addr) = {
            let (a, _ga) = d_adj.device_ptr(&stream);
            let (c, _gc) = d_cruise.device_ptr(&stream);
            (a as u64, c as u64)
        };
        let rc = unsafe {
            prism_gearbox_launch_pointer_swap(
                adj_addr as *const _,
                cruise_addr as *mut _,
                /*current_frame=*/ 9999,
                raw_stream,
            )
        };
        assert_eq!(rc, 0, "[B] PointerSwap launch returned {}", rc);
        stream.synchronize().expect("[B] post-kernel sync");

        // Bit-pattern audit.
        stream
            .memcpy_dtoh(&d_dt, &mut readback)
            .expect("[B] dtoh dt");
        let post_bits = readback[0].to_bits();
        eprintln!(
            "[B.1 BURST POST] d_dt = {:.6}f (bit-pattern 0x{:08X})",
            readback[0], post_bits
        );
        assert_eq!(
            post_bits, 0x3F000000,
            "[B] burst gate FAIL: expected 0x3F000000 (= 0.5f32, Gear 0); got 0x{:08X}",
            post_bits
        );

        // Cruise state-machine audit (Zero-Trust last_burst_frame stamp).
        let mut cruise_back = [0u8; 32];
        stream
            .memcpy_dtoh(&d_cruise, &mut cruise_back)
            .expect("[B] dtoh cruise");
        let cruise_after: ChronometricStateTensor =
            unsafe { std::mem::transmute::<[u8; 32], ChronometricStateTensor>(cruise_back) };
        assert_eq!(cruise_after.counter, 0, "burst MUST reset counter");
        assert_eq!(cruise_after.current_gear, 0, "burst MUST select Gear 0");
        assert_eq!(
            cruise_after.last_burst_frame, 9999,
            "burst MUST stamp last_burst_frame"
        );

        eprintln!(
            "[B.1 BURST PASS] bit-exact swap 0x40800000 (4.0f) → 0x{:08X} (0.5f); \
             cruise counter 1234 → 0 (RESET); gear=0; last_burst_frame=0 → 9999 (STAMPED)",
            post_bits
        );
    }

    eprintln!(
        "\n[B.1 GATE COMPLETE] All bit-pattern transitions verified device-side. \
         Zero host writes between init and readback."
    );
}
