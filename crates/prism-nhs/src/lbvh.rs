//! LBVH lane — Linear Bounding Volume Hierarchy primitives.
//!
//! Phase 1: Morton 30-bit encoder. Encodes per-spike positions into
//! 30-bit Morton codes packed in a `u32`, the input to the Karras
//! 2012 parallel radix-tree builder (Phase 2, subsequent commit).
//!
//! # Why 30-bit
//!
//! Three axes × ten bits = 1024-cell-per-axis grid. For protein
//! coordinate magnitudes (typically `[-100, 100]` Å) and spike-cloud
//! density (~tens of thousands of voxels), 1024³ cells per dimension
//! resolve sub-Å locality without spilling into the 64-bit Morton
//! regime that introduces device-side u64 atomic / sort cost. Tie
//! breaking by primitive index (low 32 bits when promoted to u64) is
//! handled at sort time in Phase 2.
//!
//! # V3-style verification posture
//!
//! The encoder's per-axis quantize and bit-interleave helpers live in
//! the .cuh as `__host__ __device__` functions. The Rust-side CPU
//! reference [`cpu_morton_30bit_encode`] is a literal port of the
//! same algorithm, so [`tests::cpu_gpu_morton_parity_8_corners`]
//! pins bit-exact equivalence between CPU and GPU output for every
//! canonical input.
//!
//! # Stream-capture-safe
//!
//! The `prism_morton_30bit_encode_run` extern "C" entry takes a
//! `cudaStream_t` and launches a single kernel; no internal
//! synchronization, no default-stream usage. The wrapped call can be
//! captured into a `cudaGraph` (Phase 3 in-flight integration) the
//! same way the M1 producer's kernel chain is captured.

use serde::{Deserialize, Serialize};

// ============================================================================
// FFI-stable bbox parameters
// ============================================================================

/// Mirror of the C-side `MortonBboxParams` in `lbvh_morton.cuh`. The
/// `#[repr(C)]` layout is byte-exact with the C struct (24 bytes,
/// no padding). Layout drift is detected by the
/// [`tests::ffi_morton_bbox_params_layout_size_is_24_bytes`] pin.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MortonBboxParams {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl MortonBboxParams {
    /// Construct from min/max corners. No validation: zero-or-inverted
    /// span on any axis is silently handled at the kernel level by
    /// quantizing every input on that axis to 0 (degenerate-bbox
    /// safety; see .cuh `prism_morton_quantize_10bit`).
    pub const fn new(min: [f32; 3], max: [f32; 3]) -> Self {
        Self { min, max }
    }

    /// Compute the tight axis-aligned bounding box that wraps every
    /// position in `positions` (planar `[N][3]`). For empty input,
    /// returns a degenerate `[0,1]³` bbox so the encoder remains
    /// total — every consumer of the output already handles
    /// `num_positions == 0` as a no-op.
    pub fn from_positions(positions: &[f32]) -> Self {
        if positions.is_empty() {
            return Self::new([0.0; 3], [1.0; 3]);
        }
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for chunk in positions.chunks_exact(3) {
            for ax in 0..3 {
                if chunk[ax] < min[ax] {
                    min[ax] = chunk[ax];
                }
                if chunk[ax] > max[ax] {
                    max[ax] = chunk[ax];
                }
            }
        }
        Self::new(min, max)
    }
}

// ============================================================================
// CPU reference (V3 parity)
// ============================================================================

/// Per-axis quantize: literal port of `prism_morton_quantize_10bit`
/// in `lbvh_morton.cuh`. Bit-equivalent on CPU and GPU.
pub fn cpu_morton_quantize_10bit(coord: f32, bbox_min: f32, bbox_max: f32) -> u32 {
    let span = bbox_max - bbox_min;
    if span <= 0.0 {
        return 0;
    }
    let u = (coord - bbox_min) / span;
    let uc = u.clamp(0.0, 1.0);
    let q = (uc * 1023.0 + 0.5) as u32;
    q.min(1023)
}

/// Bit-interleave: literal port of `prism_expand_bits_30` in
/// `lbvh_morton.cuh`. Inserts two zero bits between each bit of a
/// 10-bit input. Bit-equivalent on CPU and GPU.
pub fn cpu_expand_bits_30(mut v: u32) -> u32 {
    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v << 8)) & 0x0300F00F;
    v = (v | (v << 4)) & 0x030C30C3;
    v = (v | (v << 2)) & 0x09249249;
    v
}

/// Encode three quantized 10-bit coordinates into one 30-bit Morton
/// code. Layout: bit `3k` from `qx`, bit `3k+1` from `qy`, bit
/// `3k+2` from `qz`. Bit-equivalent on CPU and GPU.
pub fn cpu_morton_30bit_encode(qx: u32, qy: u32, qz: u32) -> u32 {
    cpu_expand_bits_30(qx) | (cpu_expand_bits_30(qy) << 1) | (cpu_expand_bits_30(qz) << 2)
}

/// One-shot CPU reference: take a position, bbox, and produce the
/// 30-bit Morton code. Used by the parity test to cross-check
/// the GPU kernel for every canonical input.
pub fn cpu_morton_30bit_encode_position(pos: [f32; 3], bbox: &MortonBboxParams) -> u32 {
    let qx = cpu_morton_quantize_10bit(pos[0], bbox.min[0], bbox.max[0]);
    let qy = cpu_morton_quantize_10bit(pos[1], bbox.min[1], bbox.max[1]);
    let qz = cpu_morton_quantize_10bit(pos[2], bbox.min[2], bbox.max[2]);
    cpu_morton_30bit_encode(qx, qy, qz)
}

// ============================================================================
// FFI surface
// ============================================================================

#[cfg(feature = "gpu")]
#[allow(dead_code)]
mod ffi {
    use super::MortonBboxParams;

    pub type CudaError = i32;
    pub const CUDA_SUCCESS: CudaError = 0;

    extern "C" {
        /// Sentinel value: `0xC0DE`. Confirms the static archive
        /// linked correctly and the FFI ABI is round-tripping.
        pub fn prism_lbvh_link_probe() -> u32;

        /// Launch the Morton 30-bit encoder kernel on the given
        /// stream. `d_codes_out` is filled with `num_positions`
        /// 30-bit codes packed in `u32`. `num_positions == 0` is a
        /// no-op success.
        pub fn prism_morton_30bit_encode_run(
            d_positions: *const f32,
            num_positions: u32,
            h_bbox: *const MortonBboxParams,
            stream: usize, // cudaStream_t is typedef'd to a pointer
            d_codes_out: *mut u32,
        ) -> CudaError;
    }
}

/// Safe Rust wrapper around the FFI link-probe. Returns `0xC0DE`.
/// Used by [`tests::link_probe_returns_sentinel`] to confirm the
/// .cu archive linked correctly.
#[cfg(feature = "gpu")]
pub fn link_probe() -> u32 {
    // SAFETY: pure value-returning probe; no pointer arguments, no
    // global state, no allocations. Calling from any thread is safe.
    unsafe { ffi::prism_lbvh_link_probe() }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffi_morton_bbox_params_layout_size_is_24_bytes() {
        // The kernel reads MortonBboxParams as a #[repr(C)] struct
        // passed by value. C side static_asserts size == 24.
        assert_eq!(std::mem::size_of::<MortonBboxParams>(), 24);
        assert_eq!(std::mem::align_of::<MortonBboxParams>(), 4);
    }

    #[test]
    fn cpu_quantize_unit_cube_corners() {
        // Unit cube [0,1]^3: corner 0 → 0; corner max → 1023.
        assert_eq!(cpu_morton_quantize_10bit(0.0, 0.0, 1.0), 0);
        assert_eq!(cpu_morton_quantize_10bit(1.0, 0.0, 1.0), 1023);
        // Center maps to ~512 (round-to-nearest).
        assert_eq!(cpu_morton_quantize_10bit(0.5, 0.0, 1.0), 512);
    }

    #[test]
    fn cpu_quantize_clamps_out_of_bbox() {
        // Below min → 0; above max → 1023. No wrap or panic.
        assert_eq!(cpu_morton_quantize_10bit(-1.0, 0.0, 1.0), 0);
        assert_eq!(cpu_morton_quantize_10bit(2.0, 0.0, 1.0), 1023);
        assert_eq!(cpu_morton_quantize_10bit(-100.0, -10.0, 10.0), 0);
        assert_eq!(cpu_morton_quantize_10bit(100.0, -10.0, 10.0), 1023);
    }

    #[test]
    fn cpu_quantize_degenerate_bbox_returns_zero() {
        // Span <= 0 → quantize returns 0 for every input. Avoids
        // NaN-induced UB in the unit-conversion division.
        assert_eq!(cpu_morton_quantize_10bit(5.0, 5.0, 5.0), 0);
        assert_eq!(cpu_morton_quantize_10bit(0.0, 5.0, 5.0), 0);
        assert_eq!(cpu_morton_quantize_10bit(10.0, 10.0, 0.0), 0);
    }

    #[test]
    fn cpu_expand_bits_30_known_values() {
        // 0 → 0
        assert_eq!(cpu_expand_bits_30(0), 0);
        // 0b1 → 0b1 (bit 0 stays at bit 0)
        assert_eq!(cpu_expand_bits_30(1), 0b1);
        // 0b10 → 0b1000 (bit 1 → bit 3)
        assert_eq!(cpu_expand_bits_30(0b10), 0b1000);
        // 0b11 → 0b1001 (bits 0,1 → bits 0,3)
        assert_eq!(cpu_expand_bits_30(0b11), 0b1001);
        // All 10 bits set → bits 0,3,6,9,12,15,18,21,24,27 set
        let all10 = 0b1111111111u32;
        let expected = (1 << 0)
            | (1 << 3)
            | (1 << 6)
            | (1 << 9)
            | (1 << 12)
            | (1 << 15)
            | (1 << 18)
            | (1 << 21)
            | (1 << 24)
            | (1 << 27);
        assert_eq!(cpu_expand_bits_30(all10), expected);
        assert_eq!(cpu_expand_bits_30(all10), 0x09249249);
    }

    #[test]
    fn cpu_morton_30bit_corners() {
        // Cube origin → Morton 0
        assert_eq!(cpu_morton_30bit_encode(0, 0, 0), 0);
        // Cube far corner (1023, 1023, 1023) → all 30 low bits set
        // = 0x3FFFFFFF
        assert_eq!(cpu_morton_30bit_encode(1023, 1023, 1023), 0x3FFFFFFF);
        // Pure-X corner (1023, 0, 0) → bits 0,3,6,...,27 set
        assert_eq!(cpu_morton_30bit_encode(1023, 0, 0), 0x09249249);
        // Pure-Y corner (0, 1023, 0) → bits 1,4,7,...,28 set
        assert_eq!(cpu_morton_30bit_encode(0, 1023, 0), 0x09249249 << 1);
        // Pure-Z corner (0, 0, 1023) → bits 2,5,8,...,29 set
        assert_eq!(cpu_morton_30bit_encode(0, 0, 1023), 0x09249249 << 2);
    }

    #[test]
    fn cpu_morton_locality_property() {
        // Property: positions in the same voxel produce the same
        // Morton code. Two positions inside cell (5, 5, 5) of the
        // unit cube grid (cell side = 1/1024).
        let bbox = MortonBboxParams::new([0.0; 3], [1.0; 3]);
        let cell = 1.0 / 1024.0;
        let p1 = [
            5.0 * cell + 0.1 * cell,
            5.0 * cell + 0.1 * cell,
            5.0 * cell + 0.1 * cell,
        ];
        let p2 = [
            5.0 * cell + 0.4 * cell,
            5.0 * cell + 0.4 * cell,
            5.0 * cell + 0.4 * cell,
        ];
        let m1 = cpu_morton_30bit_encode_position(p1, &bbox);
        let m2 = cpu_morton_30bit_encode_position(p2, &bbox);
        assert_eq!(
            m1, m2,
            "two positions in the same voxel should produce identical codes"
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn link_probe_returns_sentinel() {
        // Confirms lbvh_morton.cu linked into the static archive and
        // FFI ABI is round-tripping. Sentinel pinned at 0xC0DE.
        assert_eq!(super::link_probe(), 0x0000_C0DE);
    }

    #[cfg(feature = "gpu")]
    #[test]
    #[ignore = "GPU_RUNTIME_REQUIRED: needs real GPU + built PTX; run on the RTX workstation via `cargo test -- --ignored`"]
    fn cpu_gpu_morton_parity_8_corners() {
        use cudarc::driver::{CudaContext, DevicePtr};

        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[lbvh-morton parity] CUDA unavailable: {:?} — skipping", e);
                return;
            }
        };
        // Non-default stream — for capture compatibility down-lane.
        let stream = ctx.new_stream().expect("non-default stream");

        // 8 corners of the unit cube + a center point + 3 axis-aligned
        // mid-edge points. The CPU reference and the GPU kernel call
        // the SAME __host__ __device__ helpers, so every output must
        // be bit-equal.
        let bbox = MortonBboxParams::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let positions: Vec<f32> = vec![
            0.0, 0.0, 0.0, // corner 0
            1.0, 0.0, 0.0, // corner +x
            0.0, 1.0, 0.0, // corner +y
            0.0, 0.0, 1.0, // corner +z
            1.0, 1.0, 0.0, // corner +x+y
            1.0, 0.0, 1.0, // corner +x+z
            0.0, 1.0, 1.0, // corner +y+z
            1.0, 1.0, 1.0, // corner all
            0.5, 0.5, 0.5, // center
            0.5, 0.0, 0.0, // mid-edge x
            0.0, 0.5, 0.0, // mid-edge y
            0.0, 0.0, 0.5, // mid-edge z
        ];
        let n = (positions.len() / 3) as u32;
        let cpu_codes: Vec<u32> = positions
            .chunks_exact(3)
            .map(|c| cpu_morton_30bit_encode_position([c[0], c[1], c[2]], &bbox))
            .collect();

        // GPU run.
        let mut d_positions = stream
            .alloc_zeros::<f32>(positions.len())
            .expect("alloc d_positions");
        stream
            .memcpy_htod(&positions, &mut d_positions)
            .expect("htod positions");
        let d_codes = stream
            .alloc_zeros::<u32>(n as usize)
            .expect("alloc d_codes");

        let raw_stream = stream.cu_stream() as usize;
        let (positions_dev, _g1) = d_positions.device_ptr(&stream);
        let (codes_dev, _g2) = d_codes.device_ptr(&stream);
        let rc = unsafe {
            ffi::prism_morton_30bit_encode_run(
                positions_dev as *const f32,
                n,
                &bbox as *const MortonBboxParams,
                raw_stream,
                codes_dev as *mut u32,
            )
        };
        assert_eq!(rc, ffi::CUDA_SUCCESS, "morton encode FFI rc={}", rc);
        stream.synchronize().expect("stream sync");

        let mut gpu_codes = vec![0u32; n as usize];
        stream
            .memcpy_dtoh(&d_codes, &mut gpu_codes)
            .expect("dtoh codes");

        assert_eq!(
            gpu_codes, cpu_codes,
            "CPU/GPU Morton 30-bit code parity violated (V3 single-source contract failed)"
        );

        // Sanity: corner 0 → 0; corner all → 0x3FFFFFFF.
        assert_eq!(gpu_codes[0], 0);
        assert_eq!(gpu_codes[7], 0x3FFFFFFF);
    }

    #[cfg(feature = "gpu")]
    #[ignore = "GPU_RUNTIME_REQUIRED: needs real GPU + built PTX; run on the RTX workstation via `cargo test -- --ignored`"]
    #[test]
    fn gpu_morton_random_positions_match_cpu() {
        use cudarc::driver::{CudaContext, DevicePtr};

        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[lbvh-morton random] CUDA unavailable: {:?} — skipping", e);
                return;
            }
        };
        let stream = ctx.new_stream().expect("non-default stream");

        // Generate 1024 random positions in a non-unit bbox via a
        // deterministic LCG so the test is reproducible without the
        // `rand` crate. The bbox is `[-50, +75]³` — a typical
        // protein-coordinate magnitude that exercises non-zero
        // bbox_min and asymmetric span.
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
        let mut rng = Lcg { s: 42 };
        let bbox = MortonBboxParams::new([-50.0; 3], [75.0; 3]);
        let n = 1024u32;
        let mut positions: Vec<f32> = Vec::with_capacity(n as usize * 3);
        for _ in 0..(n as usize * 3) {
            // Map LCG output [0, 1) → [-50, 75]
            positions.push(-50.0 + rng.next_f32() * 125.0);
        }

        let cpu_codes: Vec<u32> = positions
            .chunks_exact(3)
            .map(|c| cpu_morton_30bit_encode_position([c[0], c[1], c[2]], &bbox))
            .collect();

        let mut d_positions = stream
            .alloc_zeros::<f32>(positions.len())
            .expect("alloc d_positions");
        stream
            .memcpy_htod(&positions, &mut d_positions)
            .expect("htod positions");
        let d_codes = stream
            .alloc_zeros::<u32>(n as usize)
            .expect("alloc d_codes");

        let raw_stream = stream.cu_stream() as usize;
        let (positions_dev, _g1) = d_positions.device_ptr(&stream);
        let (codes_dev, _g2) = d_codes.device_ptr(&stream);
        let rc = unsafe {
            ffi::prism_morton_30bit_encode_run(
                positions_dev as *const f32,
                n,
                &bbox as *const MortonBboxParams,
                raw_stream,
                codes_dev as *mut u32,
            )
        };
        assert_eq!(rc, ffi::CUDA_SUCCESS);
        stream.synchronize().expect("stream sync");

        let mut gpu_codes = vec![0u32; n as usize];
        stream
            .memcpy_dtoh(&d_codes, &mut gpu_codes)
            .expect("dtoh codes");

        assert_eq!(
            gpu_codes, cpu_codes,
            "1024-point random parity test failed: CPU/GPU Morton codes diverged"
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_morton_zero_positions_is_noop_success() {
        use cudarc::driver::{CudaContext, DevicePtr};

        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[lbvh-morton zero] CUDA unavailable: {:?} — skipping", e);
                return;
            }
        };
        let stream = ctx.new_stream().expect("non-default stream");

        // num_positions == 0 must return cudaSuccess without
        // dereferencing any of the device pointers (caller is allowed
        // to pass null-equivalents for zero-length buffers).
        let bbox = MortonBboxParams::new([0.0; 3], [1.0; 3]);
        let d_positions = stream.alloc_zeros::<f32>(1).expect("alloc d_positions");
        let d_codes = stream.alloc_zeros::<u32>(1).expect("alloc d_codes");

        let raw_stream = stream.cu_stream() as usize;
        let (positions_dev, _g1) = d_positions.device_ptr(&stream);
        let (codes_dev, _g2) = d_codes.device_ptr(&stream);
        let rc = unsafe {
            ffi::prism_morton_30bit_encode_run(
                positions_dev as *const f32,
                0,
                &bbox as *const MortonBboxParams,
                raw_stream,
                codes_dev as *mut u32,
            )
        };
        assert_eq!(rc, ffi::CUDA_SUCCESS);
    }

    #[test]
    fn from_positions_wraps_inputs() {
        let positions = vec![-3.0, 2.0, 7.0, 1.0, -5.0, 0.0, 4.0, 6.0, -2.0];
        let bbox = MortonBboxParams::from_positions(&positions);
        assert_eq!(bbox.min, [-3.0, -5.0, -2.0]);
        assert_eq!(bbox.max, [4.0, 6.0, 7.0]);
    }

    #[test]
    fn from_positions_empty_yields_unit_cube() {
        let bbox = MortonBboxParams::from_positions(&[]);
        assert_eq!(bbox.min, [0.0; 3]);
        assert_eq!(bbox.max, [1.0; 3]);
    }
}
