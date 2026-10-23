//! Rectification Phase 2 — Shift-Left MAR Pre-Rank Adjudicator.
//!
//! Per the PRISM-4D Pipeline Rectification mandate §1 / §4.2
//! (operator directive 2026-04-29). Replaces "late-stage
//! adjudication" with an EARLY-STAGE classifier that runs after the
//! LBVH AABB reduction but BEFORE any expensive TIDE / KCC kernel.
//!
//! # Three-way classification
//!
//! Per cluster, the adjudicator writes one [`AdjudicationCode`]:
//!
//! | Case | Discriminator | Action |
//! |---|---|---|
//! | [`AdjudicationCode::Prune`] | density < T_rho AND flux < T_phi | route to no-op sub-graph |
//! | [`AdjudicationCode::Construct`] | density ≥ T_rho OR flux ≥ T_phi | route to Bisimulation sub-graph |
//! | [`AdjudicationCode::Violation`] | NaN/Inf in density or flux | route to Abort sub-graph |
//!
//! The output array is the cudaGraphConditionalNode (F1) SWITCH
//! variant's selector.
//!
//! # Verification (mandate §4.2)
//!
//! "Run the canonical 4LPK run. The VRAM high-water mark for the
//! warm_hold phase must show a measurable decrease compared to the
//! baseline, as noise clusters are pruned before graph construction."
//!
//! At unit-test scale we exercise the classification ladder:
//!
//! - All-low observables → all Prune.
//! - All-high observables → all Construct.
//! - Mixed → mixed.
//! - Boundary case (density EXACTLY at T_rho, flux below T_phi) →
//!   Construct (strict `<` inequality, not `<=`).
//! - NaN injection → Violation.
//! - Inf injection → Violation.
//!
//! Canonical-scale verification (the VRAM high-water gate) lands
//! alongside the captured-graph wire-in in a follow-up commit.

use serde::{Deserialize, Serialize};

// ============================================================================
// Constants — must match the C-side `ADJ_*` constants in pre_rank.cuh
// ============================================================================

/// Adjudication code constants. The C-side
/// `prism_pre_rank_adjudicator_kernel` writes one of these into
/// every output slot. Public for cudaGraphConditionalNode SWITCH
/// branch indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum AdjudicationCode {
    /// Cluster is diffuse/dry — both observables below threshold.
    /// Route to no-op sub-graph; site is dropped from manifest.
    Prune = 0,
    /// Cluster has burst signal — at least one observable exceeds
    /// its threshold. Route to the Interferometric Bisimulation
    /// sub-graph for full 2+2+2+2 manifold construction.
    Construct = 1,
    /// Invariant violation — NaN/Inf observed in density or flux.
    /// Route to the Abort sub-graph; the §2.3 SAD-PATH guard.
    Violation = 2,
}

impl AdjudicationCode {
    /// Reconstruct from the raw u32 written by the GPU. Returns
    /// `None` for any value not in {0, 1, 2}; the caller decides
    /// whether to treat unknown codes as a structural error.
    pub fn from_raw(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Prune),
            1 => Some(Self::Construct),
            2 => Some(Self::Violation),
            _ => None,
        }
    }
}

// ============================================================================
// FFI-stable AABB — mirror of `pre_rank.cuh` `ClusterAabb`
// ============================================================================

/// 24-byte AABB. Layout-pinned by `mem::size_of` test below.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ClusterAabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

// ============================================================================
// FFI surface
// ============================================================================

#[cfg(feature = "gpu")]
#[allow(dead_code)]
mod ffi {
    use super::ClusterAabb;

    pub type CudaError = i32;
    pub const CUDA_SUCCESS: CudaError = 0;

    extern "C" {
        pub fn prism_pre_rank_link_probe() -> u32;

        pub fn prism_compute_aabb_volumes(
            d_aabbs: *const ClusterAabb,
            n_clusters: u32,
            d_volumes_out: *mut f32,
            stream: *mut std::ffi::c_void,
        ) -> CudaError;

        pub fn prism_compute_energy_density(
            d_intensity_sums: *const f32,
            d_volumes: *const f32,
            n_clusters: u32,
            d_densities_out: *mut f32,
            stream: *mut std::ffi::c_void,
        ) -> CudaError;

        pub fn prism_pre_rank_adjudicator(
            d_densities: *const f32,
            d_fluxes: *const f32,
            n_clusters: u32,
            threshold_rho: f32,
            threshold_phi: f32,
            d_adjudication_codes_out: *mut u32,
            stream: *mut std::ffi::c_void,
        ) -> CudaError;
    }
}

/// Safe wrapper over the FFI link-probe.
#[cfg(feature = "gpu")]
pub fn link_probe() -> u32 {
    unsafe { ffi::prism_pre_rank_link_probe() }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cluster_aabb_layout_is_24_bytes() {
        // Layout pin matches the C-side static_assert.
        assert_eq!(std::mem::size_of::<ClusterAabb>(), 24);
    }

    #[test]
    fn adjudication_code_round_trip() {
        assert_eq!(AdjudicationCode::from_raw(0), Some(AdjudicationCode::Prune));
        assert_eq!(
            AdjudicationCode::from_raw(1),
            Some(AdjudicationCode::Construct)
        );
        assert_eq!(
            AdjudicationCode::from_raw(2),
            Some(AdjudicationCode::Violation)
        );
        assert_eq!(AdjudicationCode::from_raw(3), None);
        assert_eq!(AdjudicationCode::from_raw(u32::MAX), None);
    }

    #[test]
    fn adjudication_code_repr_pinned() {
        // The GPU writes raw u32. The Rust enum's #[repr(u32)] must
        // match those values bit-for-bit.
        assert_eq!(AdjudicationCode::Prune as u32, 0);
        assert_eq!(AdjudicationCode::Construct as u32, 1);
        assert_eq!(AdjudicationCode::Violation as u32, 2);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn link_probe_returns_sentinel() {
        assert_eq!(super::link_probe(), 0x0092_55A4);
    }

    #[cfg(feature = "gpu")]
    #[ignore = "GPU_RUNTIME_REQUIRED: needs real GPU + built PTX; run on the RTX workstation via `cargo test -- --ignored`"]
    #[test]
    fn adjudicator_classifies_synthetic_clusters() {
        // Synthetic 8-cluster scenario covering every classification
        // arm:
        //
        //   c=0: density=10, flux=5     → Construct (both above
        //                                  T_rho=1, T_phi=1)
        //   c=1: density=0.5, flux=0.5  → Prune (both below)
        //   c=2: density=10, flux=0.5   → Construct (density above,
        //                                  flux below)
        //   c=3: density=0.5, flux=10   → Construct (density below,
        //                                  flux above)
        //   c=4: density=1.0, flux=0.5  → Construct (density EXACTLY
        //                                  at threshold; strict `<`
        //                                  rejects below; NOT below)
        //   c=5: density=0.5, flux=1.0  → Construct (flux at thresh)
        //   c=6: density=NaN, flux=5    → Violation (NaN guard)
        //   c=7: density=5, flux=Inf    → Violation (Inf guard)

        use cudarc::driver::{CudaContext, DevicePtr};
        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[pre-rank classify] CUDA unavailable: {:?} — skipping", e);
                return;
            }
        };
        let stream = ctx.new_stream().expect("stream");

        let densities: Vec<f32> = vec![
            10.0,
            0.5,
            10.0,
            0.5,
            1.0, // c=4: EXACTLY at T_rho
            0.5,
            f32::NAN,
            5.0,
        ];
        let fluxes: Vec<f32> = vec![
            5.0,
            0.5,
            0.5,
            10.0,
            0.5,
            1.0, // c=5: EXACTLY at T_phi
            5.0,
            f32::INFINITY,
        ];
        let n = densities.len() as u32;
        let t_rho: f32 = 1.0;
        let t_phi: f32 = 1.0;

        let mut d_densities = stream
            .alloc_zeros::<f32>(n as usize)
            .expect("alloc d_densities");
        let mut d_fluxes = stream
            .alloc_zeros::<f32>(n as usize)
            .expect("alloc d_fluxes");
        stream
            .memcpy_htod(&densities, &mut d_densities)
            .expect("htod densities");
        stream
            .memcpy_htod(&fluxes, &mut d_fluxes)
            .expect("htod fluxes");
        let d_codes = stream
            .alloc_zeros::<u32>(n as usize)
            .expect("alloc d_codes");

        let raw_stream = stream.cu_stream() as usize;
        let (densities_dev, _g1) = d_densities.device_ptr(&stream);
        let (fluxes_dev, _g2) = d_fluxes.device_ptr(&stream);
        let (codes_dev, _g3) = d_codes.device_ptr(&stream);

        let rc = unsafe {
            ffi::prism_pre_rank_adjudicator(
                densities_dev as *const f32,
                fluxes_dev as *const f32,
                n,
                t_rho,
                t_phi,
                codes_dev as *mut u32,
                raw_stream as *mut std::ffi::c_void,
            )
        };
        assert_eq!(rc, ffi::CUDA_SUCCESS);
        stream.synchronize().expect("stream sync");

        let mut codes = vec![0u32; n as usize];
        stream
            .memcpy_dtoh(&d_codes, &mut codes)
            .expect("dtoh codes");

        let codes: Vec<AdjudicationCode> = codes
            .iter()
            .map(|&v| AdjudicationCode::from_raw(v).expect("known code"))
            .collect();

        assert_eq!(codes[0], AdjudicationCode::Construct, "c=0 high/high");
        assert_eq!(codes[1], AdjudicationCode::Prune, "c=1 low/low");
        assert_eq!(codes[2], AdjudicationCode::Construct, "c=2 high density");
        assert_eq!(codes[3], AdjudicationCode::Construct, "c=3 high flux");
        assert_eq!(
            codes[4],
            AdjudicationCode::Construct,
            "c=4 boundary density (strict <)"
        );
        assert_eq!(
            codes[5],
            AdjudicationCode::Construct,
            "c=5 boundary flux (strict <)"
        );
        assert_eq!(codes[6], AdjudicationCode::Violation, "c=6 NaN density");
        assert_eq!(codes[7], AdjudicationCode::Violation, "c=7 Inf flux");
    }

    #[cfg(feature = "gpu")]
    #[ignore = "GPU_RUNTIME_REQUIRED: needs real GPU + built PTX; run on the RTX workstation via `cargo test -- --ignored`"]
    #[test]
    fn aabb_volume_and_density_pipeline() {
        // 4-cluster end-to-end: AABB volumes → energy densities →
        // adjudication codes. Verifies the three kernel chain works
        // as a unit.
        //
        //   c=0: 1×1×1 cube, intensity_sum=10  → density=10  → Construct
        //   c=1: 10×10×10 cube, intensity_sum=10 → density=0.01 → Prune
        //   c=2: degenerate AABB (max < min), intensity_sum=100 → volume=0,
        //                                                          density=0 → Prune
        //   c=3: 2×2×2, intensity_sum=4 → density=0.5 → Prune (assuming
        //                                          T_rho=1, T_phi=1, flux=0)
        //
        // Threshold: T_rho=1.0, T_phi=1.0; fluxes all 0 (low).

        use cudarc::driver::{CudaContext, DevicePtr};
        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[pre-rank pipeline] CUDA unavailable: {:?} — skipping", e);
                return;
            }
        };
        let stream = ctx.new_stream().expect("stream");

        let aabbs: Vec<ClusterAabb> = vec![
            ClusterAabb {
                min: [0.0, 0.0, 0.0],
                max: [1.0, 1.0, 1.0],
            }, // 1
            ClusterAabb {
                min: [0.0, 0.0, 0.0],
                max: [10.0, 10.0, 10.0],
            }, // 1000
            ClusterAabb {
                min: [5.0, 5.0, 5.0],
                max: [4.0, 4.0, 4.0],
            }, // degenerate
            ClusterAabb {
                min: [0.0, 0.0, 0.0],
                max: [2.0, 2.0, 2.0],
            }, // 8
        ];
        let intensities: Vec<f32> = vec![10.0, 10.0, 100.0, 4.0];
        let fluxes: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0];
        let n = aabbs.len() as u32;

        // Manual htod via byte cast — cudarc requires DeviceRepr for
        // typed slices, which ClusterAabb doesn't implement (avoiding
        // the trait dependency since the kernel just reinterprets
        // bytes). Reinterpret as f32 for the htod (24 B = 6 f32s per
        // cluster). The kernel then casts back via `*const ClusterAabb`
        // — same byte layout, same alignment.
        let aabbs_as_f32: &[f32] =
            unsafe { std::slice::from_raw_parts(aabbs.as_ptr() as *const f32, aabbs.len() * 6) };
        let mut d_aabbs_f32 = stream
            .alloc_zeros::<f32>(n as usize * 6)
            .expect("alloc d_aabbs_f32");
        stream
            .memcpy_htod(aabbs_as_f32, &mut d_aabbs_f32)
            .expect("htod aabbs");

        let mut d_intensities = stream
            .alloc_zeros::<f32>(n as usize)
            .expect("alloc d_intensities");
        let mut d_fluxes = stream
            .alloc_zeros::<f32>(n as usize)
            .expect("alloc d_fluxes");
        stream
            .memcpy_htod(&intensities, &mut d_intensities)
            .expect("htod intensities");
        stream
            .memcpy_htod(&fluxes, &mut d_fluxes)
            .expect("htod fluxes");

        let d_volumes = stream
            .alloc_zeros::<f32>(n as usize)
            .expect("alloc d_volumes");
        let d_densities = stream
            .alloc_zeros::<f32>(n as usize)
            .expect("alloc d_densities");
        let d_codes = stream
            .alloc_zeros::<u32>(n as usize)
            .expect("alloc d_codes");

        let raw_stream = stream.cu_stream() as usize;

        let (aabbs_dev, _g1) = d_aabbs_f32.device_ptr(&stream);
        let (intensities_dev, _g2) = d_intensities.device_ptr(&stream);
        let (fluxes_dev, _g3) = d_fluxes.device_ptr(&stream);
        let (volumes_dev, _g4) = d_volumes.device_ptr(&stream);
        let (densities_dev, _g5) = d_densities.device_ptr(&stream);
        let (codes_dev, _g6) = d_codes.device_ptr(&stream);

        // 1. Compute volumes.
        let rc = unsafe {
            ffi::prism_compute_aabb_volumes(
                aabbs_dev as *const ClusterAabb,
                n,
                volumes_dev as *mut f32,
                raw_stream as *mut std::ffi::c_void,
            )
        };
        assert_eq!(rc, ffi::CUDA_SUCCESS);

        // 2. Compute densities.
        let rc = unsafe {
            ffi::prism_compute_energy_density(
                intensities_dev as *const f32,
                volumes_dev as *const f32,
                n,
                densities_dev as *mut f32,
                raw_stream as *mut std::ffi::c_void,
            )
        };
        assert_eq!(rc, ffi::CUDA_SUCCESS);

        // 3. Adjudicate.
        let rc = unsafe {
            ffi::prism_pre_rank_adjudicator(
                densities_dev as *const f32,
                fluxes_dev as *const f32,
                n,
                /* T_rho */ 1.0,
                /* T_phi */ 1.0,
                codes_dev as *mut u32,
                raw_stream as *mut std::ffi::c_void,
            )
        };
        assert_eq!(rc, ffi::CUDA_SUCCESS);

        stream.synchronize().expect("stream sync");

        // dtoh + assert.
        let mut volumes = vec![0.0f32; n as usize];
        stream
            .memcpy_dtoh(&d_volumes, &mut volumes)
            .expect("dtoh volumes");
        let mut densities = vec![0.0f32; n as usize];
        stream
            .memcpy_dtoh(&d_densities, &mut densities)
            .expect("dtoh densities");
        let mut codes_raw = vec![0u32; n as usize];
        stream
            .memcpy_dtoh(&d_codes, &mut codes_raw)
            .expect("dtoh codes");

        // Volumes:
        assert!((volumes[0] - 1.0).abs() < 1e-6, "c=0 volume = 1");
        assert!((volumes[1] - 1000.0).abs() < 1e-3, "c=1 volume = 1000");
        assert_eq!(volumes[2], 0.0, "c=2 degenerate AABB → volume 0");
        assert!((volumes[3] - 8.0).abs() < 1e-6, "c=3 volume = 8");

        // Densities:
        assert!((densities[0] - 10.0).abs() < 1e-6, "c=0 density = 10");
        assert!((densities[1] - 0.01).abs() < 1e-6, "c=1 density = 0.01");
        assert_eq!(densities[2], 0.0, "c=2 zero-volume → density 0");
        assert!((densities[3] - 0.5).abs() < 1e-6, "c=3 density = 0.5");

        // Adjudication codes:
        let codes: Vec<AdjudicationCode> = codes_raw
            .iter()
            .map(|&v| AdjudicationCode::from_raw(v).unwrap())
            .collect();
        assert_eq!(
            codes[0],
            AdjudicationCode::Construct,
            "c=0 high density → Construct"
        );
        assert_eq!(
            codes[1],
            AdjudicationCode::Prune,
            "c=1 low density, low flux → Prune"
        );
        assert_eq!(
            codes[2],
            AdjudicationCode::Prune,
            "c=2 zero density, zero flux → Prune"
        );
        assert_eq!(
            codes[3],
            AdjudicationCode::Prune,
            "c=3 0.5 < 1.0 (T_rho), low flux → Prune"
        );
    }
}
