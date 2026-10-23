//! Deterministic Reduction Substrate — Rust FFI surface (Package 2).
//!
//! Backing CUDA: [`src/cuda/deterministic_reduce.cu`] +
//! [`src/cuda/deterministic_reduce.cuh`]. Read the ordering-guarantee block at
//! the top of the `.cuh` before using anything here; this module is the safe
//! wrapper around that contract, not a redefinition of it.
//!
//! # What problem this solves
//!
//! Float addition is not associative. An `atomicAdd`-based reduction sums in
//! *arrival* order, and arrival order is set by the scheduler, not by the data.
//! Two runs of the same binary on the same input therefore produce sums that
//! differ in the low mantissa bits.
//!
//! For a telemetry value that is irrelevant. For a **governor decision input**
//! it is fatal: the physical-stability governor compares reduced quantities
//! (force norm, energy drift) against thresholds. One ulp straddling a
//! threshold flips the comparison, flips the authorized gear, changes `dt`, and
//! changes the trajectory. The provenance record then honestly reports two
//! different decisions for the same physics — which destroys the
//! reproducibility of a controller whose entire justification is auditable
//! decisions.
//!
//! `zstr_force_stage_f4_kernel` (`cuda/zstr_kernels.cu:203-209`) uses exactly
//! that pattern — per-warp `atomicAdd` into the force-norm slot. That is fine
//! for the ZSTR evidence stream. It must never be the source of a gear
//! decision. This module is the replacement for decision inputs.
//!
//! # The guarantee, restated
//!
//! For a fixed input array and fixed `n`, every entry point here returns a
//! **bit-identical** scalar across runs, streams, concurrent load, and graph
//! replays; and [`reference_reduce_f32`] / [`reference_reduce_f64`] reproduce
//! those exact bits on the CPU, so a recorded decision can be re-derived
//! off-device from a recorded input snapshot.
//!
//! `n` is part of the measurement identity. Changing `n` changes the partition
//! and therefore legitimately changes the bits. Record it alongside the value.
//!
//! # Quantization: the recommendation asked for by the directive
//!
//! **Recommendation: quantize decision inputs before the threshold comparison —
//! but as defence in depth layered on top of the deterministic reduction, never
//! as a replacement for it, and always recording both the raw and the quantized
//! value.**
//!
//! Reasoning, including the part that argues against quantization:
//!
//! * *For.* A deterministic reduction only fixes the inputs **this module**
//!   produces. The governor will also consume scalars computed elsewhere
//!   (CUB reductions in `energy_monitor.cu`, host-side accumulations, values
//!   restored from a checkpoint written by a different build). Rounding to a
//!   coarse significand grid immediately before the comparison makes the
//!   comparison insensitive to low-bit noise from *any* of those producers,
//!   without having to prove each one deterministic first.
//! * *For.* It makes the recorded number and the compared number the same
//!   object. Without it, a record can show a value that, re-read at full
//!   precision, sits on the other side of the threshold from the branch that
//!   was actually taken — an auditor cannot tell whether that is a real
//!   inconsistency or display rounding.
//! * *Against — and this is the important caveat.* Quantization does **not**
//!   eliminate threshold flips. It *relocates* them. A value that lands exactly
//!   on a quantization boundary is just as flip-prone as before, and the set of
//!   flip-prone values is now a discrete lattice instead of a measure-zero
//!   surface. Anyone who believes quantization "makes the comparison
//!   deterministic" has been misled by it.
//! * *Against.* A coarse grid is a real loss of physical resolution. At the
//!   documented f32 default (16 of 23 mantissa bits) the relative quantum is
//!   ~7.6e-6, far below any physically meaningful force or drift discrimination,
//!   so the loss is acceptable here — but that is a claim about *these*
//!   quantities at *this* setting, not a general licence.
//!
//! Therefore: (1) deterministic reduction is the primary guarantee; (2)
//! quantization is applied to decision inputs at the comparison site as a
//! second layer; (3) the genuine fix for boundary flips is **hysteresis / a
//! dead-band at the threshold itself**, which belongs to the governor and is
//! recorded here as `RECOMMENDED_OPERATIONAL_IMPROVEMENT` — a coarser gear must
//! require crossing the threshold by a margin, and the direction of the last
//! crossing must be part of the governor state. That is precisely the
//! asymmetric-coarsening rule the architecture already demands, expressed
//! numerically.
//!
//! The quantization width is **policy-resolved**, never hardcoded at the call
//! site: see [`QuantizationPolicy::resolve`], which reports the resolved value
//! *and* its [`PolicyOrigin`] so both go into the provenance record.

#![cfg(feature = "gpu")]

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};

// ============================================================================
// Contract constants — mirrors of `deterministic_reduce.cuh`
// ============================================================================
//
// These are duplicated from C on purpose and then CHECKED against the C side by
// `verify_abi()`. A silent divergence between the two would change the produced
// bits while every signature still compiled.

/// Mirror of `PRISM_DR_SCHEMA_VERSION`. Bump on the C side whenever the
/// partition count, tree shape, intrinsic set, or sentinel encoding changes —
/// i.e. whenever a decision recorded by an older binary stops being
/// reproducible by this one.
pub const DR_SCHEMA_VERSION: u32 = 1;

/// Mirror of `PRISM_DR_PARTITIONS`. Part of the numerical contract: the
/// reduction tree has exactly this many leaves, always.
pub const DR_PARTITIONS: u32 = 2048;

/// Mirror of `PRISM_DR_SENTINEL_NOT_MEASURED_F32_BITS`.
///
/// A quiet NaN with a distinctive payload. Written whenever a reduction had
/// nothing to measure, so a downstream reader can distinguish "never measured"
/// from "the physics produced a NaN" — completely different incidents that a
/// plain NaN would conflate.
pub const SENTINEL_NOT_MEASURED_F32_BITS: u32 = 0x7FC0_DEAD;

/// Mirror of `PRISM_DR_SENTINEL_NOT_MEASURED_F64_BITS`.
pub const SENTINEL_NOT_MEASURED_F64_BITS: u64 = 0x7FF8_0000_0000_DEAD;

/// Explicit significand bits in an `f32` (`PRISM_DR_MANTISSA_BITS_F32`).
pub const MANTISSA_BITS_F32: u32 = 23;
/// Explicit significand bits in an `f64` (`PRISM_DR_MANTISSA_BITS_F64`).
pub const MANTISSA_BITS_F64: u32 = 52;

/// Source-literal default for f32 decision-input quantization: keep 16 of 23
/// mantissa bits, relative quantum <= 2^-17 (~7.6e-6).
pub const DEFAULT_QUANT_BITS_F32: u32 = 16;
/// Source-literal default for f64 decision-input quantization: keep 34 of 52
/// mantissa bits, relative quantum <= 2^-35 (~2.9e-11).
pub const DEFAULT_QUANT_BITS_F64: u32 = 34;

/// Environment override for the f32 quantization width.
pub const ENV_QUANT_BITS_F32: &str = "PRISM_DR_QUANT_MANTISSA_BITS_F32";
/// Environment override for the f64 quantization width.
pub const ENV_QUANT_BITS_F64: &str = "PRISM_DR_QUANT_MANTISSA_BITS_F64";

// ============================================================================
// Operators
// ============================================================================

/// Reduction operator. Discriminants must match the `PRISM_DR_OP_*` macros.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ReduceOp {
    /// Fixed-order sum.
    Sum = 0,
    /// Fixed-order sum of squares (the `||F||^2` input).
    SumSq = 1,
    /// NaN-**propagating** minimum. See the NaN policy note below.
    Min = 2,
    /// NaN-**propagating** maximum.
    Max = 3,
}

impl ReduceOp {
    #[inline]
    fn code(self) -> i32 {
        self as i32
    }

    /// Identity element used to seed an empty partition.
    #[inline]
    fn identity_f32(self) -> f32 {
        match self {
            ReduceOp::Sum | ReduceOp::SumSq => 0.0,
            ReduceOp::Min => f32::INFINITY,
            ReduceOp::Max => f32::NEG_INFINITY,
        }
    }

    #[inline]
    fn identity_f64(self) -> f64 {
        match self {
            ReduceOp::Sum | ReduceOp::SumSq => 0.0,
            ReduceOp::Min => f64::INFINITY,
            ReduceOp::Max => f64::NEG_INFINITY,
        }
    }
}

// ============================================================================
// Errors
// ============================================================================

/// Failure modes of the deterministic reduction surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DrError {
    /// A required device pointer was null.
    NullArg,
    /// Nothing to reduce. The output slot has been written with the
    /// NOT-MEASURED sentinel; it is **not** a measurement and must not be
    /// compared against a threshold.
    EmptyInput,
    /// Operator code outside `PRISM_DR_OP_*`.
    BadOp,
    /// Caller-supplied scratch is smaller than `scratch_bytes_*()`.
    ScratchTooSmall,
    /// Raw `cudaError_t` from a launch.
    Cuda(i32),
    /// The linked archive disagrees with this module's mirrored constants.
    /// Refuse to run rather than silently produce different bits.
    AbiMismatch {
        /// Which constant diverged.
        field: &'static str,
        /// What Rust expected.
        expected: u64,
        /// What the linked CUDA archive reports.
        actual: u64,
    },
    /// A host-side precondition on the Rust wrapper was violated.
    InvalidArgument(&'static str),
}

impl std::fmt::Display for DrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DrError::NullArg => write!(f, "deterministic_reduce: null device pointer"),
            DrError::EmptyInput => write!(
                f,
                "deterministic_reduce: empty input; NOT-MEASURED sentinel written, \
                 value must not be consumed as a measurement"
            ),
            DrError::BadOp => write!(f, "deterministic_reduce: bad operator code"),
            DrError::ScratchTooSmall => {
                write!(f, "deterministic_reduce: scratch buffer too small")
            }
            DrError::Cuda(rc) => write!(f, "deterministic_reduce: cudaError_t {rc}"),
            DrError::AbiMismatch {
                field,
                expected,
                actual,
            } => write!(
                f,
                "deterministic_reduce: ABI mismatch on {field}: Rust expects {expected}, \
                 linked CUDA archive reports {actual}"
            ),
            DrError::InvalidArgument(m) => write!(f, "deterministic_reduce: {m}"),
        }
    }
}

impl std::error::Error for DrError {}

fn rc_to_result(rc: i32) -> Result<(), DrError> {
    match rc {
        0 => Ok(()),
        -1 => Err(DrError::NullArg),
        -2 => Err(DrError::EmptyInput),
        -3 => Err(DrError::BadOp),
        -4 => Err(DrError::ScratchTooSmall),
        other => Err(DrError::Cuda(other)),
    }
}

// ============================================================================
// FFI — extern "C" forward declarations from deterministic_reduce.cu
// ============================================================================

pub mod ffi {
    use std::ffi::c_void;

    extern "C" {
        /// Constant scratch requirement for the f32 reductions.
        pub fn prism_dr_scratch_bytes_f32() -> usize;
        /// Constant scratch requirement for the f64 reductions.
        pub fn prism_dr_scratch_bytes_f64() -> usize;
        /// Compile-time leaf count of the reduction tree.
        pub fn prism_dr_partitions() -> u32;
        /// Compile-time schema version of the numerical contract.
        pub fn prism_dr_schema_version() -> u32;

        /// Fixed-order reduction over `d_in[0..n)`. See the `.cuh` contract.
        pub fn prism_dr_reduce_f32(
            d_in: *const f32,
            n: u32,
            op: i32,
            d_scratch: *mut c_void,
            scratch_bytes: usize,
            d_out: *mut f32,
            stream: *mut c_void,
        ) -> i32;

        /// f64 counterpart of [`prism_dr_reduce_f32`].
        pub fn prism_dr_reduce_f64(
            d_in: *const f64,
            n: u32,
            op: i32,
            d_scratch: *mut c_void,
            scratch_bytes: usize,
            d_out: *mut f64,
            stream: *mut c_void,
        ) -> i32;

        /// `sum_i (x^2 + y^2 + z^2)` over a `float4` array, skipping `.w`.
        pub fn prism_dr_reduce_f4_xyz_sumsq(
            d_in_float4: *const c_void,
            n_vec: u32,
            d_scratch: *mut c_void,
            scratch_bytes: usize,
            d_out: *mut f32,
            stream: *mut c_void,
        ) -> i32;

        /// In-place significand-grid quantization of a device scalar.
        pub fn prism_dr_quantize_inplace_f32(
            d_value: *mut f32,
            mantissa_bits_kept: u32,
            stream: *mut c_void,
        ) -> i32;

        /// f64 counterpart of [`prism_dr_quantize_inplace_f32`].
        pub fn prism_dr_quantize_inplace_f64(
            d_value: *mut f64,
            mantissa_bits_kept: u32,
            stream: *mut c_void,
        ) -> i32;

        /// Write the NOT-MEASURED sentinel into a device scalar.
        pub fn prism_dr_write_not_measured_f32(d_out: *mut f32, stream: *mut c_void) -> i32;
        /// f64 counterpart.
        pub fn prism_dr_write_not_measured_f64(d_out: *mut f64, stream: *mut c_void) -> i32;

        /// DIAGNOSTIC ONLY — the nondeterministic `atomicAdd` pattern this
        /// module replaces. Never call this for a decision input. Exists so
        /// the determinism test can exhibit the variance side by side with the
        /// deterministic result. `d_out` must be zeroed before each call.
        pub fn prism_dr_DIAGNOSTIC_nondeterministic_atomic_sum_f32(
            d_in: *const f32,
            n: u32,
            d_out: *mut f32,
            stream: *mut c_void,
        ) -> i32;
    }
}

// ============================================================================
// ABI verification
// ============================================================================

/// Check that the linked CUDA archive agrees with this module's mirrored
/// constants.
///
/// Call once at pipeline construction. A divergence here means the produced
/// bits would differ from what the host reference reproduces, which silently
/// invalidates every recorded decision — so it is an error, not a warning.
pub fn verify_abi() -> Result<(), DrError> {
    let schema = unsafe { ffi::prism_dr_schema_version() };
    if schema != DR_SCHEMA_VERSION {
        return Err(DrError::AbiMismatch {
            field: "PRISM_DR_SCHEMA_VERSION",
            expected: DR_SCHEMA_VERSION as u64,
            actual: schema as u64,
        });
    }
    let partitions = unsafe { ffi::prism_dr_partitions() };
    if partitions != DR_PARTITIONS {
        return Err(DrError::AbiMismatch {
            field: "PRISM_DR_PARTITIONS",
            expected: DR_PARTITIONS as u64,
            actual: partitions as u64,
        });
    }
    let sf32 = unsafe { ffi::prism_dr_scratch_bytes_f32() };
    let want_f32 = DR_PARTITIONS as usize * std::mem::size_of::<f32>();
    if sf32 != want_f32 {
        return Err(DrError::AbiMismatch {
            field: "prism_dr_scratch_bytes_f32",
            expected: want_f32 as u64,
            actual: sf32 as u64,
        });
    }
    let sf64 = unsafe { ffi::prism_dr_scratch_bytes_f64() };
    let want_f64 = DR_PARTITIONS as usize * std::mem::size_of::<f64>();
    if sf64 != want_f64 {
        return Err(DrError::AbiMismatch {
            field: "prism_dr_scratch_bytes_f64",
            expected: want_f64 as u64,
            actual: sf64 as u64,
        });
    }
    Ok(())
}

/// Scratch bytes required by every f32 entry point. Constant — never depends
/// on `n`.
pub fn scratch_bytes_f32() -> usize {
    unsafe { ffi::prism_dr_scratch_bytes_f32() }
}

/// Scratch bytes required by every f64 entry point.
pub fn scratch_bytes_f64() -> usize {
    unsafe { ffi::prism_dr_scratch_bytes_f64() }
}

// ============================================================================
// Scratch
// ============================================================================

/// Device scratch for the reduction's phase-1 partials.
///
/// Allocate **once** per pipeline and keep it alive for the campaign: the
/// captured graph bakes the pointer at capture time, so a reallocated scratch
/// buffer would make replays read freed memory.
pub struct ReduceScratch {
    buf: CudaSlice<u8>,
}

impl ReduceScratch {
    /// Allocate scratch sized for the f32 reductions (including the
    /// `float4`-xyz sum-of-squares entry point).
    pub fn new_f32(stream: &Arc<CudaStream>) -> Result<Self, DrError> {
        Self::with_bytes(stream, scratch_bytes_f32())
    }

    /// Allocate scratch sized for the f64 reductions.
    pub fn new_f64(stream: &Arc<CudaStream>) -> Result<Self, DrError> {
        Self::with_bytes(stream, scratch_bytes_f64())
    }

    /// Allocate scratch large enough for both widths.
    pub fn new_any(stream: &Arc<CudaStream>) -> Result<Self, DrError> {
        Self::with_bytes(stream, scratch_bytes_f32().max(scratch_bytes_f64()))
    }

    fn with_bytes(stream: &Arc<CudaStream>, bytes: usize) -> Result<Self, DrError> {
        let buf = stream
            .alloc_zeros::<u8>(bytes)
            .map_err(|_| DrError::InvalidArgument("scratch allocation failed"))?;
        Ok(Self { buf })
    }

    /// Size of the underlying allocation in bytes.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// True when the allocation is empty (never true for a constructed value).
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }
}

// ============================================================================
// Device entry points
// ============================================================================

/// Deterministic reduction of `input` into `out` (a 1-element device slice).
///
/// Stream-ordered and capture-safe: no allocation, no memcpy, no
/// synchronization. `input.len()` is baked into the graph at capture time.
///
/// Returns [`DrError::EmptyInput`] when `input` is empty — in that case `out`
/// still receives the NOT-MEASURED sentinel, so the slot is never ambiguous,
/// but the caller must not treat it as a measurement.
pub fn reduce_f32(
    stream: &Arc<CudaStream>,
    input: &CudaSlice<f32>,
    op: ReduceOp,
    scratch: &mut ReduceScratch,
    out: &mut CudaSlice<f32>,
) -> Result<(), DrError> {
    if out.is_empty() {
        return Err(DrError::InvalidArgument("out slice must hold >= 1 element"));
    }
    let n = u32::try_from(input.len())
        .map_err(|_| DrError::InvalidArgument("input length exceeds u32"))?;
    // Read every length BEFORE taking device pointers: the SyncOnDrop guards
    // hold borrows for the rest of the block.
    let scratch_bytes = scratch.buf.len();
    let raw_stream = stream.cu_stream() as *mut c_void;

    let rc = {
        let (p_in, _g_in) = input.device_ptr(stream);
        let (p_scratch, _g_scratch) = scratch.buf.device_ptr_mut(stream);
        let (p_out, _g_out) = out.device_ptr_mut(stream);
        unsafe {
            ffi::prism_dr_reduce_f32(
                p_in as *const f32,
                n,
                op.code(),
                p_scratch as *mut c_void,
                scratch_bytes,
                p_out as *mut f32,
                raw_stream,
            )
        }
    };
    rc_to_result(rc)
}

/// f64 counterpart of [`reduce_f32`].
pub fn reduce_f64(
    stream: &Arc<CudaStream>,
    input: &CudaSlice<f64>,
    op: ReduceOp,
    scratch: &mut ReduceScratch,
    out: &mut CudaSlice<f64>,
) -> Result<(), DrError> {
    if out.is_empty() {
        return Err(DrError::InvalidArgument("out slice must hold >= 1 element"));
    }
    let n = u32::try_from(input.len())
        .map_err(|_| DrError::InvalidArgument("input length exceeds u32"))?;
    let scratch_bytes = scratch.buf.len();
    let raw_stream = stream.cu_stream() as *mut c_void;

    let rc = {
        let (p_in, _g_in) = input.device_ptr(stream);
        let (p_scratch, _g_scratch) = scratch.buf.device_ptr_mut(stream);
        let (p_out, _g_out) = out.device_ptr_mut(stream);
        unsafe {
            ffi::prism_dr_reduce_f64(
                p_in as *const f64,
                n,
                op.code(),
                p_scratch as *mut c_void,
                scratch_bytes,
                p_out as *mut f64,
                raw_stream,
            )
        }
    };
    rc_to_result(rc)
}

/// Deterministic `sum_i (x^2 + y^2 + z^2)` over a `float4`-packed force buffer,
/// **skipping the `.w` lane**.
///
/// This is the governor's force-magnitude input. It has its own entry point
/// because the obvious call site is wrong: reducing a padded `float4` force
/// buffer as a flat `f32` array folds the `.w` padding into `||F||^2`.
///
/// `input` is the flat backing store; `input.len()` must be exactly
/// `4 * n_vec`.
pub fn reduce_f4_xyz_sumsq(
    stream: &Arc<CudaStream>,
    input: &CudaSlice<f32>,
    n_vec: u32,
    scratch: &mut ReduceScratch,
    out: &mut CudaSlice<f32>,
) -> Result<(), DrError> {
    if out.is_empty() {
        return Err(DrError::InvalidArgument("out slice must hold >= 1 element"));
    }
    let needed = (n_vec as usize)
        .checked_mul(4)
        .ok_or(DrError::InvalidArgument("n_vec * 4 overflows"))?;
    if input.len() != needed {
        return Err(DrError::InvalidArgument(
            "float4 input length must be exactly 4 * n_vec",
        ));
    }
    let scratch_bytes = scratch.buf.len();
    let raw_stream = stream.cu_stream() as *mut c_void;

    let rc = {
        let (p_in, _g_in) = input.device_ptr(stream);
        let (p_scratch, _g_scratch) = scratch.buf.device_ptr_mut(stream);
        let (p_out, _g_out) = out.device_ptr_mut(stream);
        unsafe {
            ffi::prism_dr_reduce_f4_xyz_sumsq(
                p_in as *const c_void,
                n_vec,
                p_scratch as *mut c_void,
                scratch_bytes,
                p_out as *mut f32,
                raw_stream,
            )
        }
    };
    rc_to_result(rc)
}

/// Quantize a device scalar in place, on the stream, immediately before a
/// threshold comparison.
pub fn quantize_inplace_f32(
    stream: &Arc<CudaStream>,
    value: &mut CudaSlice<f32>,
    mantissa_bits_kept: u32,
) -> Result<(), DrError> {
    if value.is_empty() {
        return Err(DrError::InvalidArgument("value slice is empty"));
    }
    let rc = {
        let (p, _g) = value.device_ptr_mut(stream);
        unsafe {
            ffi::prism_dr_quantize_inplace_f32(
                p as *mut f32,
                mantissa_bits_kept,
                stream.cu_stream() as *mut c_void,
            )
        }
    };
    rc_to_result(rc)
}

/// f64 counterpart of [`quantize_inplace_f32`].
pub fn quantize_inplace_f64(
    stream: &Arc<CudaStream>,
    value: &mut CudaSlice<f64>,
    mantissa_bits_kept: u32,
) -> Result<(), DrError> {
    if value.is_empty() {
        return Err(DrError::InvalidArgument("value slice is empty"));
    }
    let rc = {
        let (p, _g) = value.device_ptr_mut(stream);
        unsafe {
            ffi::prism_dr_quantize_inplace_f64(
                p as *mut f64,
                mantissa_bits_kept,
                stream.cu_stream() as *mut c_void,
            )
        }
    };
    rc_to_result(rc)
}

/// Pre-poison a decision slot with the NOT-MEASURED sentinel.
///
/// Use before the producing node runs. A slot that still reads as the sentinel
/// afterwards proves the producer did not execute, instead of leaving stale
/// data from the previous step that looks exactly like a fresh measurement.
pub fn write_not_measured_f32(
    stream: &Arc<CudaStream>,
    out: &mut CudaSlice<f32>,
) -> Result<(), DrError> {
    if out.is_empty() {
        return Err(DrError::InvalidArgument("out slice is empty"));
    }
    let rc = {
        let (p, _g) = out.device_ptr_mut(stream);
        unsafe {
            ffi::prism_dr_write_not_measured_f32(p as *mut f32, stream.cu_stream() as *mut c_void)
        }
    };
    rc_to_result(rc)
}

/// f64 counterpart of [`write_not_measured_f32`].
pub fn write_not_measured_f64(
    stream: &Arc<CudaStream>,
    out: &mut CudaSlice<f64>,
) -> Result<(), DrError> {
    if out.is_empty() {
        return Err(DrError::InvalidArgument("out slice is empty"));
    }
    let rc = {
        let (p, _g) = out.device_ptr_mut(stream);
        unsafe {
            ffi::prism_dr_write_not_measured_f64(p as *mut f64, stream.cu_stream() as *mut c_void)
        }
    };
    rc_to_result(rc)
}

/// True when `x` carries the exact NOT-MEASURED payload.
///
/// Note this is deliberately a **bit** comparison, not `is_nan()`: a NaN that
/// arose from the physics is a different event from a slot that was never
/// written, and the governor must not treat them alike.
#[inline]
pub fn is_not_measured_f32(x: f32) -> bool {
    x.to_bits() == SENTINEL_NOT_MEASURED_F32_BITS
}

/// f64 counterpart of [`is_not_measured_f32`].
#[inline]
pub fn is_not_measured_f64(x: f64) -> bool {
    x.to_bits() == SENTINEL_NOT_MEASURED_F64_BITS
}

// ============================================================================
// Host reference implementation — replays the exact device order
// ============================================================================
//
// This is not a test helper. It is the off-device re-derivation path: given a
// recorded input snapshot, it reproduces the exact bits the device produced, so
// a recorded gear decision can be independently re-checked on a CPU without a
// GPU and without the original binary.

#[inline]
fn combine_f32(op: ReduceOp, a: f32, b: f32) -> f32 {
    match op {
        ReduceOp::Sum | ReduceOp::SumSq => a + b,
        // NaN-PROPAGATING, matching the device. `fminf`/`f32::min` suppress
        // NaN, which would hide a NaN force from the stability governor.
        ReduceOp::Min => {
            if a.is_nan() || b.is_nan() {
                f32::NAN
            } else if a < b {
                a
            } else {
                b
            }
        }
        ReduceOp::Max => {
            if a.is_nan() || b.is_nan() {
                f32::NAN
            } else if a > b {
                a
            } else {
                b
            }
        }
    }
}

#[inline]
fn combine_f64(op: ReduceOp, a: f64, b: f64) -> f64 {
    match op {
        ReduceOp::Sum | ReduceOp::SumSq => a + b,
        ReduceOp::Min => {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else if a < b {
                a
            } else {
                b
            }
        }
        ReduceOp::Max => {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else if a > b {
                a
            } else {
                b
            }
        }
    }
}

/// Fixed-shape balanced tree over exactly `DR_PARTITIONS` leaves — the same
/// tree the device's phase 2 walks.
fn tree_fold_f32(partials: &mut [f32], op: ReduceOp) -> f32 {
    let mut stride = (DR_PARTITIONS / 2) as usize;
    while stride >= 1 {
        for i in 0..stride {
            partials[i] = combine_f32(op, partials[i], partials[i + stride]);
        }
        if stride == 1 {
            break;
        }
        stride >>= 1;
    }
    partials[0]
}

fn tree_fold_f64(partials: &mut [f64], op: ReduceOp) -> f64 {
    let mut stride = (DR_PARTITIONS / 2) as usize;
    while stride >= 1 {
        for i in 0..stride {
            partials[i] = combine_f64(op, partials[i], partials[i + stride]);
        }
        if stride == 1 {
            break;
        }
        stride >>= 1;
    }
    partials[0]
}

/// Host replay of [`reduce_f32`]. Bit-identical to the device result.
///
/// Returns `None` for an empty input, matching the device's
/// [`DrError::EmptyInput`] — the caller must not substitute an identity.
pub fn reference_reduce_f32(data: &[f32], op: ReduceOp) -> Option<f32> {
    if data.is_empty() {
        return None;
    }
    let p = DR_PARTITIONS as usize;
    let mut partials = vec![op.identity_f32(); p];
    // Phase 1: cyclic partition, strictly sequential fold in increasing index.
    for (j, slot) in partials.iter_mut().enumerate() {
        let mut acc = op.identity_f32();
        let mut i = j;
        while i < data.len() {
            let x = data[i];
            let t = if op == ReduceOp::SumSq { x * x } else { x };
            acc = combine_f32(op, acc, t);
            i += p;
        }
        *slot = acc;
    }
    // Phase 2: fixed-shape tree.
    Some(tree_fold_f32(&mut partials, op))
}

/// Host replay of [`reduce_f64`]. Bit-identical to the device result.
pub fn reference_reduce_f64(data: &[f64], op: ReduceOp) -> Option<f64> {
    if data.is_empty() {
        return None;
    }
    let p = DR_PARTITIONS as usize;
    let mut partials = vec![op.identity_f64(); p];
    for (j, slot) in partials.iter_mut().enumerate() {
        let mut acc = op.identity_f64();
        let mut i = j;
        while i < data.len() {
            let x = data[i];
            let t = if op == ReduceOp::SumSq { x * x } else { x };
            acc = combine_f64(op, acc, t);
            i += p;
        }
        *slot = acc;
    }
    Some(tree_fold_f64(&mut partials, op))
}

/// Host replay of [`reduce_f4_xyz_sumsq`]. `data` is the flat backing store of
/// length `4 * n_vec`; the `.w` lane is skipped, and the per-vector fold order
/// is x, then y, then z.
pub fn reference_f4_xyz_sumsq(data: &[f32], n_vec: usize) -> Option<f32> {
    if n_vec == 0 {
        return None;
    }
    if data.len() != n_vec * 4 {
        return None;
    }
    let p = DR_PARTITIONS as usize;
    let mut partials = vec![0.0f32; p];
    for (j, slot) in partials.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        let mut i = j;
        while i < n_vec {
            let base = i * 4;
            // x, then y, then z — the same per-vector fold order the device
            // uses. `.w` is never read.
            acc += data[base] * data[base];
            acc += data[base + 1] * data[base + 1];
            acc += data[base + 2] * data[base + 2];
            i += p;
        }
        *slot = acc;
    }
    Some(tree_fold_f32(&mut partials, ReduceOp::Sum))
}

// ============================================================================
// Quantization
// ============================================================================

/// Round `x` onto the significand grid that keeps the top `mantissa_bits_kept`
/// explicit mantissa bits, ties to even on that grid.
///
/// Bit-exact mirror of `prism_dr_quantize_f32` in the `.cuh`. NaN and Inf pass
/// through unchanged — a quantizer must never launder a NaN into a finite
/// number. Carry out of the significand into the exponent is intentional
/// (rounding 1.9999 up to 2.0 crosses a binade and that is the right answer).
#[inline]
pub fn quantize_f32(x: f32, mantissa_bits_kept: u32) -> f32 {
    let bits = x.to_bits();
    if bits & 0x7F80_0000 == 0x7F80_0000 {
        return x; // NaN or Inf
    }
    if mantissa_bits_kept >= MANTISSA_BITS_F32 {
        return x;
    }
    let drop = MANTISSA_BITS_F32 - mantissa_bits_kept;
    let lsb: u32 = 1u32 << drop;
    let half: u32 = lsb >> 1;
    let rem = bits & (lsb - 1);
    let mut kept = bits & !(lsb - 1);
    if rem > half || (rem == half && (kept & lsb) != 0) {
        kept = kept.wrapping_add(lsb);
    }
    f32::from_bits(kept)
}

/// f64 counterpart of [`quantize_f32`]. Bit-exact mirror of
/// `prism_dr_quantize_f64`.
#[inline]
pub fn quantize_f64(x: f64, mantissa_bits_kept: u32) -> f64 {
    let bits = x.to_bits();
    if bits & 0x7FF0_0000_0000_0000 == 0x7FF0_0000_0000_0000 {
        return x; // NaN or Inf
    }
    if mantissa_bits_kept >= MANTISSA_BITS_F64 {
        return x;
    }
    let drop = MANTISSA_BITS_F64 - mantissa_bits_kept;
    let lsb: u64 = 1u64 << drop;
    let half: u64 = lsb >> 1;
    let rem = bits & (lsb - 1);
    let mut kept = bits & !(lsb - 1);
    if rem > half || (rem == half && (kept & lsb) != 0) {
        kept = kept.wrapping_add(lsb);
    }
    f64::from_bits(kept)
}

/// Where a resolved policy value came from. Recorded with the decision so an
/// auditor can tell a deliberate override from the shipped default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyOrigin {
    /// The documented source literal in this module / the `.cuh`.
    SourceLiteralDefault,
    /// An environment override, named by the variable that supplied it.
    Environment(&'static str),
    /// Supplied programmatically by the caller.
    Explicit,
}

/// Resolved decision-input quantization policy.
///
/// No threshold or width is hardcoded at a call site: construct this, use its
/// fields, and write `mantissa_bits_*` **and** `origin_*` into the provenance
/// record so the recorded decision carries the coarsening that produced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantizationPolicy {
    /// Retained f32 mantissa bits.
    pub mantissa_bits_f32: u32,
    /// Provenance of `mantissa_bits_f32`.
    pub origin_f32: PolicyOrigin,
    /// Retained f64 mantissa bits.
    pub mantissa_bits_f64: u32,
    /// Provenance of `mantissa_bits_f64`.
    pub origin_f64: PolicyOrigin,
}

impl Default for QuantizationPolicy {
    fn default() -> Self {
        Self {
            mantissa_bits_f32: DEFAULT_QUANT_BITS_F32,
            origin_f32: PolicyOrigin::SourceLiteralDefault,
            mantissa_bits_f64: DEFAULT_QUANT_BITS_F64,
            origin_f64: PolicyOrigin::SourceLiteralDefault,
        }
    }
}

impl QuantizationPolicy {
    /// Resolve from the environment, falling back to the documented source
    /// literals.
    ///
    /// An override outside `1..=MANTISSA_BITS_*` is rejected and the default is
    /// kept — a malformed policy must not silently become "no quantization" or
    /// "quantize everything to zero bits".
    pub fn resolve() -> Self {
        let (bits32, origin32) = resolve_bits(
            ENV_QUANT_BITS_F32,
            DEFAULT_QUANT_BITS_F32,
            MANTISSA_BITS_F32,
        );
        let (bits64, origin64) = resolve_bits(
            ENV_QUANT_BITS_F64,
            DEFAULT_QUANT_BITS_F64,
            MANTISSA_BITS_F64,
        );
        Self {
            mantissa_bits_f32: bits32,
            origin_f32: origin32,
            mantissa_bits_f64: bits64,
            origin_f64: origin64,
        }
    }

    /// Explicit construction, marked as such in the provenance.
    pub fn explicit(mantissa_bits_f32: u32, mantissa_bits_f64: u32) -> Self {
        Self {
            mantissa_bits_f32,
            origin_f32: PolicyOrigin::Explicit,
            mantissa_bits_f64,
            origin_f64: PolicyOrigin::Explicit,
        }
    }

    /// Upper bound on the relative quantization error for f32 at this policy
    /// (`2^-(bits+1)`). Record it next to the value so the recorded precision
    /// is explicit rather than implied.
    pub fn relative_quantum_f32(&self) -> f64 {
        (2.0f64).powi(-((self.mantissa_bits_f32 as i32) + 1))
    }

    /// f64 counterpart of [`Self::relative_quantum_f32`].
    pub fn relative_quantum_f64(&self) -> f64 {
        (2.0f64).powi(-((self.mantissa_bits_f64 as i32) + 1))
    }

    /// Apply the f32 width.
    #[inline]
    pub fn apply_f32(&self, x: f32) -> f32 {
        quantize_f32(x, self.mantissa_bits_f32)
    }

    /// Apply the f64 width.
    #[inline]
    pub fn apply_f64(&self, x: f64) -> f64 {
        quantize_f64(x, self.mantissa_bits_f64)
    }
}

fn resolve_bits(env_key: &'static str, default: u32, max_bits: u32) -> (u32, PolicyOrigin) {
    match std::env::var(env_key) {
        Ok(raw) => match raw.trim().parse::<u32>() {
            Ok(v) if (1..=max_bits).contains(&v) => (v, PolicyOrigin::Environment(env_key)),
            _ => (default, PolicyOrigin::SourceLiteralDefault),
        },
        Err(_) => (default, PolicyOrigin::SourceLiteralDefault),
    }
}
