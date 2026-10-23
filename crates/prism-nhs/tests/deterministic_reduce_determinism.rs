//! Package 2 acceptance gate — deterministic reduction of governor decision
//! inputs.
//!
//! The claim under test: for a fixed input array and fixed `n`, the reduction
//! returns a BIT-IDENTICAL scalar across repeated runs, and that scalar is
//! reproducible off-device by the host reference.
//!
//! # Why this test has teeth
//!
//! Asserting "the deterministic path agrees with itself" is worthless unless
//! the input is actually order-sensitive — on a benign input every reduction
//! order agrees and the assertion passes vacuously. So each phase below is
//! guarded:
//!
//!   1. `order_sensitivity_witness` PROVES the probe data is order-sensitive by
//!      summing it forward, backward, and in a strided order on the host and
//!      asserting the three results DIFFER in bits. If that assertion ever
//!      stops holding, the rest of the file is vacuous and this test fails
//!      loudly rather than passing quietly.
//!   2. The DIAGNOSTIC atomicAdd kernel — the exact pattern
//!      `zstr_force_stage_f4_kernel` uses — is run on the same data many times
//!      and its distinct-result count is reported. That is the real-hardware
//!      counter-example.
//!   3. Only then is bit-identity asserted for the deterministic path over the
//!      same data, same repeat count, on fresh streams, under concurrent load.
//!
//! # Why one `#[test]`
//!
//! Same reason as `gearbox_b1_dt_flip.rs`: cargo runs `#[test]` functions in
//! parallel, and this file deliberately creates GPU contention to perturb
//! scheduling. Serialising the phases inside one test keeps the contention
//! under this test's control instead of making it a cross-test race.

#![cfg(feature = "gpu")]

use std::collections::BTreeSet;

use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DevicePtrMut};
use prism_nhs::deterministic_reduce::{
    ffi, is_not_measured_f32, is_not_measured_f64, quantize_f32, quantize_f64,
    quantize_inplace_f32, quantize_inplace_f64, reduce_f32, reduce_f4_xyz_sumsq, reduce_f64,
    reference_f4_xyz_sumsq, reference_reduce_f32, reference_reduce_f64, scratch_bytes_f32,
    verify_abi, write_not_measured_f32, write_not_measured_f64, DrError, PolicyOrigin,
    QuantizationPolicy, ReduceOp, ReduceScratch, DEFAULT_QUANT_BITS_F32, DEFAULT_QUANT_BITS_F64,
    DR_PARTITIONS, MANTISSA_BITS_F32, SENTINEL_NOT_MEASURED_F32_BITS,
    SENTINEL_NOT_MEASURED_F64_BITS,
};
use std::ffi::c_void;

/// Repeat count for the bit-identity assertions. High enough that an
/// arrival-order-dependent reduction over ~200k elements has essentially no
/// chance of producing the same bits every time.
const REPEATS: usize = 64;

/// Element count. Deliberately NOT a multiple of `DR_PARTITIONS`, so the cyclic
/// partition is ragged (some partitions fold one more element than others) and
/// the tail handling is exercised.
const N: usize = 200_003;

// ───────────────────────────────────────────────────────────────────────────
// Probe data
// ───────────────────────────────────────────────────────────────────────────

/// Deterministic xorshift so the probe data is reproducible across runs — the
/// test must not itself be a source of variance.
struct XorShift(u64);

impl XorShift {
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 32) as u32
    }
    /// Uniform in [-1, 1).
    fn next_unit(&mut self) -> f32 {
        (self.next_u32() as f32 / 2147483648.0f32) - 1.0f32
    }
}

/// Catastrophic-cancellation probe: a few huge magnitudes buried in a sea of
/// small ones. Summation order decides how much of the small mass survives
/// rounding against the large terms, so different orders give different bits.
/// This is not artificial — a force array with a couple of clashing atoms and
/// tens of thousands of equilibrated ones has exactly this shape.
fn probe_f32(n: usize) -> Vec<f32> {
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        if i % 4096 == 0 {
            v.push(1.0e8f32 * if (i / 4096) % 2 == 0 { 1.0 } else { -1.0 });
        } else {
            v.push(rng.next_unit());
        }
    }
    v
}

fn probe_f64(n: usize) -> Vec<f64> {
    let mut rng = XorShift(0xD1B5_4A32_D192_ED03);
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        if i % 4096 == 0 {
            v.push(1.0e16f64 * if (i / 4096) % 2 == 0 { 1.0 } else { -1.0 });
        } else {
            v.push(rng.next_unit() as f64);
        }
    }
    v
}

// ───────────────────────────────────────────────────────────────────────────
// Phase 1 (host only) — prove the probe data is order-sensitive.
// ───────────────────────────────────────────────────────────────────────────

fn order_sensitivity_witness(data: &[f32]) {
    let fwd = data.iter().fold(0.0f32, |a, &b| a + b);
    let rev = data.iter().rev().fold(0.0f32, |a, &b| a + b);

    // A strided order: walk the array in 97-element strides, which is
    // coprime with the length, so every element is visited exactly once in a
    // different order.
    let mut strided = 0.0f32;
    let stride = 97usize;
    let mut idx = 0usize;
    for _ in 0..data.len() {
        strided += data[idx];
        idx = (idx + stride) % data.len();
    }

    eprintln!(
        "[DR order-sensitivity] forward=0x{:08X}  reverse=0x{:08X}  strided=0x{:08X}",
        fwd.to_bits(),
        rev.to_bits(),
        strided.to_bits()
    );

    let distinct: BTreeSet<u32> = [fwd.to_bits(), rev.to_bits(), strided.to_bits()]
        .into_iter()
        .collect();
    assert!(
        distinct.len() > 1,
        "PROBE DATA IS NOT ORDER-SENSITIVE — every determinism assertion below \
         would pass vacuously. forward/reverse/strided all produced 0x{:08X}.",
        fwd.to_bits()
    );
}

// ───────────────────────────────────────────────────────────────────────────

#[test]
fn deterministic_reduce_is_bit_identical_across_repeats_and_matches_host_reference() {
    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[DR] CUDA unavailable: {e:?} — skipping");
            return;
        }
    };

    // ═══ ABI gate ═══════════════════════════════════════════════════════
    // If the linked archive disagrees with the Rust mirrors, the produced
    // bits are not the bits the host reference reproduces, and every
    // assertion below would be testing a different contract.
    verify_abi().expect("deterministic_reduce ABI mismatch between Rust and linked CUDA archive");
    assert_eq!(
        scratch_bytes_f32(),
        DR_PARTITIONS as usize * 4,
        "f32 scratch sizing drifted"
    );

    let stream = ctx.new_stream().expect("stream");

    let host_f32 = probe_f32(N);
    let host_f64 = probe_f64(N);

    // ═══ PHASE 1 — prove the probe is order-sensitive ═══════════════════
    order_sensitivity_witness(&host_f32);

    // ═══ PHASE 2 — exhibit the nondeterminism we are replacing ══════════
    //
    // Same data, the atomicAdd pattern from zstr_force_stage_f4_kernel.
    // We REPORT the distinct-result count rather than asserting it is > 1:
    // arrival order is a hardware scheduling property, and a test that fails
    // when the GPU happens to be idle enough to serialise consistently would
    // be a flaky test, not a stronger one. Phase 1 already carries the
    // "this input is order-sensitive" burden of proof.
    {
        let mut d_in: CudaSlice<f32> = stream.alloc_zeros::<f32>(N).expect("alloc atomic in");
        stream.memcpy_htod(&host_f32, &mut d_in).expect("htod");
        let raw_stream = stream.cu_stream() as *mut c_void;

        let mut seen: BTreeSet<u32> = BTreeSet::new();
        for _ in 0..REPEATS {
            let mut d_out: CudaSlice<f32> = stream.alloc_zeros::<f32>(1).expect("alloc atomic out");
            let rc = {
                let (p_in, _g_in) = d_in.device_ptr(&stream);
                let (p_out, _g_out) = d_out.device_ptr_mut(&stream);
                unsafe {
                    ffi::prism_dr_DIAGNOSTIC_nondeterministic_atomic_sum_f32(
                        p_in as *const f32,
                        N as u32,
                        p_out as *mut f32,
                        raw_stream,
                    )
                }
            };
            assert_eq!(rc, 0, "diagnostic atomic kernel launch rc={rc}");
            stream.synchronize().expect("sync atomic");
            let mut back = [0.0f32; 1];
            stream.memcpy_dtoh(&d_out, &mut back).expect("dtoh atomic");
            seen.insert(back[0].to_bits());
        }
        eprintln!(
            "[DR atomicAdd counter-example] {} distinct bit patterns over {} runs of the \
             SAME input (this is the pattern in zstr_force_stage_f4_kernel)",
            seen.len(),
            REPEATS
        );
    }

    // ═══ PHASE 3 — the actual gate: f32, all four operators ═════════════
    for op in [ReduceOp::Sum, ReduceOp::SumSq, ReduceOp::Min, ReduceOp::Max] {
        let mut seen: BTreeSet<u32> = BTreeSet::new();

        for r in 0..REPEATS {
            // Fresh stream every repeat: stream identity, launch history, and
            // any implicit ordering must not be able to influence the result.
            let s = ctx.new_stream().expect("repeat stream");

            // Concurrent load on a second stream to perturb block scheduling.
            let noise = ctx.new_stream().expect("noise stream");
            let mut d_noise: CudaSlice<f32> =
                noise.alloc_zeros::<f32>(1 << 16).expect("alloc noise");
            let noise_host = vec![1.0f32; 1 << 16];
            noise
                .memcpy_htod(&noise_host, &mut d_noise)
                .expect("noise htod");

            let mut d_in: CudaSlice<f32> = s.alloc_zeros::<f32>(N).expect("alloc in");
            s.memcpy_htod(&host_f32, &mut d_in).expect("htod in");
            let mut d_out: CudaSlice<f32> = s.alloc_zeros::<f32>(1).expect("alloc out");
            let mut sc = ReduceScratch::new_any(&s).expect("repeat scratch");

            reduce_f32(&s, &d_in, op, &mut sc, &mut d_out).expect("reduce_f32");
            s.synchronize().expect("sync");

            let mut back = [0.0f32; 1];
            s.memcpy_dtoh(&d_out, &mut back).expect("dtoh");
            seen.insert(back[0].to_bits());

            if r == 0 {
                eprintln!(
                    "[DR {op:?}] first result = {} (0x{:08X})",
                    back[0],
                    back[0].to_bits()
                );
            }
        }

        assert_eq!(
            seen.len(),
            1,
            "DETERMINISM GATE FAIL for {op:?}: {} distinct bit patterns over {} identical \
             runs: {:?}",
            seen.len(),
            REPEATS,
            seen.iter()
                .map(|b| format!("0x{b:08X}"))
                .collect::<Vec<_>>()
        );

        // Off-device re-derivation: the host reference must reproduce the
        // exact bits, which is what makes a recorded decision auditable
        // without a GPU.
        let device_bits = *seen.iter().next().unwrap();
        let host = reference_reduce_f32(&host_f32, op).expect("host reference");
        assert_eq!(
            host.to_bits(),
            device_bits,
            "HOST REPLAY MISMATCH for {op:?}: host 0x{:08X} vs device 0x{:08X}",
            host.to_bits(),
            device_bits
        );
        eprintln!("[DR {op:?}] bit-identical over {REPEATS} runs AND matches host replay");
    }

    // ═══ PHASE 4 — f64 ══════════════════════════════════════════════════
    for op in [ReduceOp::Sum, ReduceOp::SumSq, ReduceOp::Min, ReduceOp::Max] {
        let mut seen: BTreeSet<u64> = BTreeSet::new();
        for _ in 0..REPEATS {
            let s = ctx.new_stream().expect("repeat stream f64");
            let mut d_in: CudaSlice<f64> = s.alloc_zeros::<f64>(N).expect("alloc in f64");
            s.memcpy_htod(&host_f64, &mut d_in).expect("htod f64");
            let mut d_out: CudaSlice<f64> = s.alloc_zeros::<f64>(1).expect("alloc out f64");
            let mut sc = ReduceScratch::new_f64(&s).expect("scratch f64");

            reduce_f64(&s, &d_in, op, &mut sc, &mut d_out).expect("reduce_f64");
            s.synchronize().expect("sync f64");

            let mut back = [0.0f64; 1];
            s.memcpy_dtoh(&d_out, &mut back).expect("dtoh f64");
            seen.insert(back[0].to_bits());
        }
        assert_eq!(
            seen.len(),
            1,
            "DETERMINISM GATE FAIL (f64) for {op:?}: {} distinct bit patterns",
            seen.len()
        );
        let host = reference_reduce_f64(&host_f64, op).expect("host reference f64");
        assert_eq!(
            host.to_bits(),
            *seen.iter().next().unwrap(),
            "HOST REPLAY MISMATCH (f64) for {op:?}"
        );
        eprintln!("[DR f64 {op:?}] bit-identical over {REPEATS} runs AND matches host replay");
    }

    // ═══ PHASE 5 — float4 xyz sum-of-squares (the ||F||^2 input) ════════
    //
    // Also proves the `.w` lane is genuinely excluded: `.w` is loaded with a
    // huge poison value that would dominate the result if it were folded in.
    {
        let n_vec = 65_537usize; // again not a multiple of DR_PARTITIONS
        let mut flat = Vec::with_capacity(n_vec * 4);
        let mut rng = XorShift(0x2545_F491_4F6C_DD1D);
        for i in 0..n_vec {
            if i % 2048 == 0 {
                flat.push(1.0e6f32);
                flat.push(-1.0e6f32);
                flat.push(1.0e6f32);
            } else {
                flat.push(rng.next_unit());
                flat.push(rng.next_unit());
                flat.push(rng.next_unit());
            }
            flat.push(9.9e18f32); // .w poison — must never reach the accumulator
        }

        let mut seen: BTreeSet<u32> = BTreeSet::new();
        for _ in 0..REPEATS {
            let s = ctx.new_stream().expect("f4 stream");
            let mut d_in: CudaSlice<f32> = s.alloc_zeros::<f32>(flat.len()).expect("alloc f4");
            s.memcpy_htod(&flat, &mut d_in).expect("htod f4");
            let mut d_out: CudaSlice<f32> = s.alloc_zeros::<f32>(1).expect("alloc f4 out");
            let mut sc = ReduceScratch::new_f32(&s).expect("f4 scratch");

            reduce_f4_xyz_sumsq(&s, &d_in, n_vec as u32, &mut sc, &mut d_out)
                .expect("reduce_f4_xyz_sumsq");
            s.synchronize().expect("sync f4");

            let mut back = [0.0f32; 1];
            s.memcpy_dtoh(&d_out, &mut back).expect("dtoh f4");
            seen.insert(back[0].to_bits());
        }
        assert_eq!(seen.len(), 1, "DETERMINISM GATE FAIL (float4 xyz sumsq)");

        let device = f32::from_bits(*seen.iter().next().unwrap());
        assert!(
            device.is_finite(),
            "float4 xyz sumsq produced {device} — the .w poison lane leaked into the sum"
        );
        let host = reference_f4_xyz_sumsq(&flat, n_vec).expect("host f4 reference");
        assert_eq!(
            host.to_bits(),
            device.to_bits(),
            "HOST REPLAY MISMATCH (float4 xyz sumsq): host 0x{:08X} vs device 0x{:08X}",
            host.to_bits(),
            device.to_bits()
        );
        eprintln!(
            "[DR float4-xyz-sumsq] bit-identical over {REPEATS} runs, host replay matches, \
             .w poison excluded (result {device:e})"
        );
    }

    // ═══ PHASE 6 — NaN must PROPAGATE through min/max ═══════════════════
    //
    // fminf/fmaxf suppress NaN. If min/max silently dropped a NaN force, the
    // stability governor would never see the event that gear 3 exists for.
    {
        let mut data = vec![1.0f32; 5000];
        data[4321] = f32::NAN;
        for op in [ReduceOp::Min, ReduceOp::Max] {
            let s = ctx.new_stream().expect("nan stream");
            let mut d_in: CudaSlice<f32> = s.alloc_zeros::<f32>(data.len()).expect("alloc nan");
            s.memcpy_htod(&data, &mut d_in).expect("htod nan");
            let mut d_out: CudaSlice<f32> = s.alloc_zeros::<f32>(1).expect("alloc nan out");
            let mut sc = ReduceScratch::new_f32(&s).expect("nan scratch");
            reduce_f32(&s, &d_in, op, &mut sc, &mut d_out).expect("reduce nan");
            s.synchronize().expect("sync nan");
            let mut back = [0.0f32; 1];
            s.memcpy_dtoh(&d_out, &mut back).expect("dtoh nan");
            assert!(
                back[0].is_nan(),
                "{op:?} SUPPRESSED a NaN input (got {}) — a NaN force would be invisible \
                 to the physical-stability governor",
                back[0]
            );
            assert!(
                reference_reduce_f32(&data, op).unwrap().is_nan(),
                "host reference {op:?} suppressed the NaN — host and device disagree on the \
                 NaN policy"
            );
        }
        eprintln!("[DR NaN policy] min/max propagate NaN on both device and host replay");
    }

    // ═══ PHASE 6b — n < DR_PARTITIONS: the empty-partition identity path ══
    //
    // With n below the partition count most partitions fold ZERO elements and
    // keep the operator identity. That is a different code path from the
    // ragged large-n case: for Min the identity is +inf, so an identity bug
    // would return +inf instead of the true minimum. The host reference shares
    // the partition layout, so device-vs-host agreement alone would NOT catch
    // a wrong identity — the absolute assertions below are what catch it.
    {
        let n_small = 137usize; // << DR_PARTITIONS
        let mut small: Vec<f32> = (0..n_small).map(|i| (i as f32) - 68.0).collect();
        small[100] = -1234.5;
        small[7] = 4321.5;

        for (op, expected) in [
            (ReduceOp::Min, -1234.5f32),
            (ReduceOp::Max, 4321.5f32),
            (
                ReduceOp::Sum,
                small.iter().copied().fold(0.0f32, |a, b| a + b),
            ),
        ] {
            let s = ctx.new_stream().expect("small stream");
            let mut d_in: CudaSlice<f32> = s.alloc_zeros::<f32>(n_small).expect("alloc small");
            s.memcpy_htod(&small, &mut d_in).expect("htod small");
            let mut d_out: CudaSlice<f32> = s.alloc_zeros::<f32>(1).expect("alloc small out");
            let mut sc = ReduceScratch::new_f32(&s).expect("small scratch");
            reduce_f32(&s, &d_in, op, &mut sc, &mut d_out).expect("reduce small");
            s.synchronize().expect("sync small");
            let mut back = [0.0f32; 1];
            s.memcpy_dtoh(&d_out, &mut back).expect("dtoh small");

            assert!(
                back[0].is_finite(),
                "{op:?} with n={n_small} < DR_PARTITIONS returned {} — an empty-partition \
                 identity leaked into the result",
                back[0]
            );
            match op {
                ReduceOp::Min | ReduceOp::Max => assert_eq!(
                    back[0], expected,
                    "{op:?} with n={n_small} returned {} but the true extremum is {expected}",
                    back[0]
                ),
                // Sum ordering differs from a naive left fold, so compare with
                // a tolerance here and rely on the host replay for exactness.
                _ => assert!(
                    (back[0] - expected).abs() <= 1e-3 * expected.abs().max(1.0),
                    "{op:?} with n={n_small} returned {} vs naive {expected}",
                    back[0]
                ),
            }
            assert_eq!(
                reference_reduce_f32(&small, op).unwrap().to_bits(),
                back[0].to_bits(),
                "HOST REPLAY MISMATCH at n={n_small} for {op:?}"
            );
        }
        eprintln!(
            "[DR small-n] n={n_small} < DR_PARTITIONS={DR_PARTITIONS}: empty partitions keep \
             the identity without contaminating min/max/sum"
        );
    }

    // ═══ PHASE 7 — NOT-MEASURED sentinel, never a fabricated identity ═══
    {
        let s = ctx.new_stream().expect("sentinel stream");
        let empty: CudaSlice<f32> = s.alloc_zeros::<f32>(0).expect("alloc empty");
        let mut d_out: CudaSlice<f32> = s.alloc_zeros::<f32>(1).expect("alloc sentinel out");
        let mut sc = ReduceScratch::new_f32(&s).expect("sentinel scratch");

        let err = reduce_f32(&s, &empty, ReduceOp::Sum, &mut sc, &mut d_out).unwrap_err();
        assert_eq!(
            err,
            DrError::EmptyInput,
            "empty input must report EmptyInput, not silently succeed"
        );
        s.synchronize().expect("sync sentinel");

        let mut back = [0.0f32; 1];
        s.memcpy_dtoh(&d_out, &mut back).expect("dtoh sentinel");
        assert_eq!(
            back[0].to_bits(),
            SENTINEL_NOT_MEASURED_F32_BITS,
            "empty reduction wrote 0x{:08X}, expected the NOT-MEASURED sentinel 0x{:08X} — \
             an operator identity here would be a fabricated measurement",
            back[0].to_bits(),
            SENTINEL_NOT_MEASURED_F32_BITS
        );
        assert!(is_not_measured_f32(back[0]));
        // A NaN produced by physics must NOT be mistaken for "not measured".
        assert!(!is_not_measured_f32(f32::NAN));

        // Explicit pre-poison path.
        let mut d_poison: CudaSlice<f32> = s.alloc_zeros::<f32>(1).expect("alloc poison");
        write_not_measured_f32(&s, &mut d_poison).expect("write sentinel");
        s.synchronize().expect("sync poison");
        s.memcpy_dtoh(&d_poison, &mut back).expect("dtoh poison");
        assert_eq!(back[0].to_bits(), SENTINEL_NOT_MEASURED_F32_BITS);

        let mut d_poison64: CudaSlice<f64> = s.alloc_zeros::<f64>(1).expect("alloc poison64");
        write_not_measured_f64(&s, &mut d_poison64).expect("write sentinel f64");
        s.synchronize().expect("sync poison64");
        let mut back64 = [0.0f64; 1];
        s.memcpy_dtoh(&d_poison64, &mut back64)
            .expect("dtoh poison64");
        assert_eq!(back64[0].to_bits(), SENTINEL_NOT_MEASURED_F64_BITS);
        assert!(is_not_measured_f64(back64[0]));

        assert_eq!(reference_reduce_f32(&[], ReduceOp::Sum), None);
        assert_eq!(reference_reduce_f64(&[], ReduceOp::Sum), None);
        eprintln!("[DR sentinel] empty input yields NOT-MEASURED, distinguishable from a real NaN");
    }

    // ═══ PHASE 8 — device and host quantizers must agree BIT-EXACTLY ════
    //
    // `prism_dr_quantize_f32` (.cuh) and `quantize_f32` (.rs) are two
    // separately written pieces of bit manipulation. If they diverge, the
    // value the governor COMPARED on-device and the value an auditor
    // re-derives on the host are different numbers — which silently breaks
    // exactly the property quantization was added to provide. Probes cover
    // binade boundaries, exact ties (both directions of ties-to-even),
    // subnormals, negatives, zero, NaN and Inf.
    {
        let mut probes: Vec<f32> = vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            1234.5678,
            -9.87e-12,
            3.4e30,
            1.0e-38, // near-subnormal
            f32::MIN_POSITIVE,
            f32::from_bits(1),           // smallest subnormal
            f32::from_bits(0x007F_FFFF), // largest subnormal
            f32::MAX,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            1.9999999, // rounds up across a binade
        ];
        // Exact ties on the retained grid, constructed for both parities of
        // the retained LSB so ties-to-even is exercised in both directions.
        for bits in [8u32, 16, 20] {
            let drop = MANTISSA_BITS_F32 - bits;
            let lsb = 1u32 << drop;
            let half = lsb >> 1;
            let base = 1234.5f32.to_bits() & !(lsb - 1);
            probes.push(f32::from_bits((base & !lsb) | half)); // tie, LSB even
            probes.push(f32::from_bits(base | lsb | half)); // tie, LSB odd
        }

        let s = ctx.new_stream().expect("quant stream");
        let mut d_v: CudaSlice<f32> = s.alloc_zeros::<f32>(1).expect("alloc quant");
        let mut d_v64: CudaSlice<f64> = s.alloc_zeros::<f64>(1).expect("alloc quant64");

        let mut checked = 0usize;
        for bits in [1u32, 8, 16, 20, 23, 31] {
            for &x in &probes {
                s.memcpy_htod(&[x], &mut d_v).expect("htod quant");
                quantize_inplace_f32(&s, &mut d_v, bits).expect("device quantize f32");
                s.synchronize().expect("sync quant");
                let mut back = [0.0f32; 1];
                s.memcpy_dtoh(&d_v, &mut back).expect("dtoh quant");
                assert_eq!(
                    back[0].to_bits(),
                    quantize_f32(x, bits).to_bits(),
                    "DEVICE/HOST QUANTIZER DIVERGENCE (f32, bits={bits}) on input \
                     0x{:08X}: device 0x{:08X} vs host 0x{:08X}",
                    x.to_bits(),
                    back[0].to_bits(),
                    quantize_f32(x, bits).to_bits()
                );

                let x64 = x as f64;
                s.memcpy_htod(&[x64], &mut d_v64).expect("htod quant64");
                let bits64 = bits * 2;
                quantize_inplace_f64(&s, &mut d_v64, bits64).expect("device quantize f64");
                s.synchronize().expect("sync quant64");
                let mut back64 = [0.0f64; 1];
                s.memcpy_dtoh(&d_v64, &mut back64).expect("dtoh quant64");
                assert_eq!(
                    back64[0].to_bits(),
                    quantize_f64(x64, bits64).to_bits(),
                    "DEVICE/HOST QUANTIZER DIVERGENCE (f64, bits={bits64}) on input \
                     0x{:016X}",
                    x64.to_bits()
                );
                checked += 2;
            }
        }
        eprintln!(
            "[DR quantizer parity] device and host agree bit-exactly on {checked} \
             (value, width) pairs including binade crossings, both tie parities, \
             subnormals, NaN and Inf"
        );
    }

    eprintln!("[DR] PACKAGE 2 DETERMINISM GATE PASSED");
}

// ───────────────────────────────────────────────────────────────────────────
// Quantization — pure host, no GPU required.
// ───────────────────────────────────────────────────────────────────────────

#[test]
fn quantization_is_idempotent_absorbs_subquantum_noise_and_preserves_nan_inf() {
    let bits = DEFAULT_QUANT_BITS_F32;

    // Idempotent: quantizing an already-quantized value is a no-op. Without
    // this, re-recording a value would keep moving it.
    for raw in [1.0f32, 1234.5678, -9.87e-12, 3.4e30, 1.0e-38] {
        let q = quantize_f32(raw, bits);
        assert_eq!(
            q.to_bits(),
            quantize_f32(q, bits).to_bits(),
            "quantize_f32 not idempotent at {raw}"
        );
    }

    // Sub-quantum noise is absorbed: two values differing far below the grid
    // map to the same bucket, which is the whole point for a threshold
    // comparison fed by a producer we did not write.
    let base = 1234.5f32;
    let noised = f32::from_bits(base.to_bits() + 3); // 3 ulp
    assert_ne!(base.to_bits(), noised.to_bits());
    assert_eq!(
        quantize_f32(base, bits).to_bits(),
        quantize_f32(noised, bits).to_bits(),
        "3-ulp noise survived quantization at {bits} retained bits"
    );

    // Above-quantum differences must SURVIVE — a quantizer that collapses
    // physically meaningful differences is worse than none.
    let coarse = base * 1.01f32;
    assert_ne!(
        quantize_f32(base, bits).to_bits(),
        quantize_f32(coarse, bits).to_bits(),
        "a 1% difference was collapsed by quantization"
    );

    // Full width is the identity.
    assert_eq!(
        quantize_f32(base, MANTISSA_BITS_F32).to_bits(),
        base.to_bits()
    );

    // NaN / Inf pass through unchanged — never laundered into a finite number.
    assert!(quantize_f32(f32::NAN, bits).is_nan());
    assert_eq!(quantize_f32(f32::INFINITY, bits), f32::INFINITY);
    assert_eq!(quantize_f32(f32::NEG_INFINITY, bits), f32::NEG_INFINITY);
    assert!(quantize_f64(f64::NAN, DEFAULT_QUANT_BITS_F64).is_nan());
    assert_eq!(
        quantize_f64(f64::INFINITY, DEFAULT_QUANT_BITS_F64),
        f64::INFINITY
    );

    // Sign symmetry: -x quantizes to -(x quantized).
    for raw in [1234.5678f32, 1.0e-7, 6.02e23] {
        assert_eq!(
            quantize_f32(-raw, bits).to_bits(),
            quantize_f32(raw, bits).to_bits() ^ 0x8000_0000,
            "quantization is not sign-symmetric at {raw}"
        );
    }

    // Zero is a fixed point.
    assert_eq!(quantize_f32(0.0, bits).to_bits(), 0.0f32.to_bits());
    assert_eq!(
        quantize_f64(0.0, DEFAULT_QUANT_BITS_F64).to_bits(),
        0.0f64.to_bits()
    );
}

#[test]
fn quantization_policy_records_resolved_value_and_origin() {
    // No env override in the test process: the default must be reported as
    // coming from the source literal, not silently.
    std::env::remove_var("PRISM_DR_QUANT_MANTISSA_BITS_F32");
    std::env::remove_var("PRISM_DR_QUANT_MANTISSA_BITS_F64");
    let p = QuantizationPolicy::resolve();
    assert_eq!(p.mantissa_bits_f32, DEFAULT_QUANT_BITS_F32);
    assert_eq!(p.origin_f32, PolicyOrigin::SourceLiteralDefault);
    assert_eq!(p.mantissa_bits_f64, DEFAULT_QUANT_BITS_F64);
    assert_eq!(p.origin_f64, PolicyOrigin::SourceLiteralDefault);

    // A valid override is honoured AND attributed to the variable.
    std::env::set_var("PRISM_DR_QUANT_MANTISSA_BITS_F32", "12");
    let p = QuantizationPolicy::resolve();
    assert_eq!(p.mantissa_bits_f32, 12);
    assert_eq!(
        p.origin_f32,
        PolicyOrigin::Environment("PRISM_DR_QUANT_MANTISSA_BITS_F32")
    );

    // A malformed or out-of-range override must fall back to the documented
    // default, NOT to "no quantization" and NOT to zero bits.
    for bad in ["0", "24", "-3", "banana", ""] {
        std::env::set_var("PRISM_DR_QUANT_MANTISSA_BITS_F32", bad);
        let p = QuantizationPolicy::resolve();
        assert_eq!(
            p.mantissa_bits_f32, DEFAULT_QUANT_BITS_F32,
            "malformed override {bad:?} was not rejected"
        );
        assert_eq!(p.origin_f32, PolicyOrigin::SourceLiteralDefault);
    }
    std::env::remove_var("PRISM_DR_QUANT_MANTISSA_BITS_F32");

    // The recorded relative quantum must match the retained width.
    let p = QuantizationPolicy::explicit(16, 34);
    assert!((p.relative_quantum_f32() - 2f64.powi(-17)).abs() < 1e-30);
    assert!((p.relative_quantum_f64() - 2f64.powi(-35)).abs() < 1e-40);
    assert_eq!(p.origin_f32, PolicyOrigin::Explicit);
}
