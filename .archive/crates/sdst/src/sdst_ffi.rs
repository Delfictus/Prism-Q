/**
 * Rust FFI Integration Guide for SDST
 *
 * This file documents how to call SDST from nhs_rt_full via Rust FFI.
 * The actual Rust bindings would go in your NHS crate.
 *
 * Build integration:
 *   1. Build libsdst.so: `cd sdst && make shared`
 *   2. In your Cargo.toml, add a build.rs that links against libsdst
 *   3. Use the FFI bindings below in your Rust code
 */

// ============================================================
// build.rs (add to your nhs_rt_full crate)
// ============================================================
//
// fn main() {
//     println!("cargo:rustc-link-search=native=/path/to/sdst/lib");
//     println!("cargo:rustc-link-lib=dylib=sdst");
//     println!("cargo:rustc-link-lib=dylib=cudart");
//     println!("cargo:rerun-if-changed=/path/to/sdst/lib/libsdst.so");
// }

// ============================================================
// sdst_ffi.rs - Raw FFI bindings
// ============================================================

#![allow(non_camel_case_types)]
#![allow(dead_code)]

use std::os::raw::{c_char, c_float, c_int, c_void};

// Used to free host-side arrays allocated by SDST C functions (malloc'd in libsdst)
extern "C" { fn free(ptr: *mut c_void); }

pub type MortonCode = u32;
pub type SpikeId = u32;
pub type WavefrontId = u32;
pub type AvalancheId = u32;
pub type PhaseId = u8;
pub type SdstHandle = *mut c_void;

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SdstError {
    Success = 0,
    ErrorCuda = -1,
    ErrorOom = -2,
    ErrorInvalidParam = -3,
    ErrorTableFull = -4,
    ErrorNotFound = -5,
    ErrorWavefrontOverflow = -6,
    ErrorStreamInvalid = -7,
}

impl SdstError {
    pub fn is_ok(self) -> bool { self == SdstError::Success }
    pub fn check(self) -> Result<(), SdstError> {
        if self.is_ok() { Ok(()) } else { Err(self) }
    }
}

/// Repacked to match C layout (GATE 0 verified): u32s first (offsets 0-19),
/// u16s (20-31), u8s (32-33), 2 implicit trailing pad → sizeof = 36.
#[repr(C, align(4))]
#[derive(Debug, Clone, Copy, Default)]
pub struct SpikeEvent {
    pub voxel: MortonCode,          // u32 @ 0
    pub timestamp: u32,             // u32 @ 4
    pub parent_spike: SpikeId,      // u32 @ 8
    pub avalanche_id: AvalancheId,  // u32 @ 12
    pub wavefront_id: WavefrontId,  // u32 @ 16
    pub amplitude: u16,             // u16 @ 20
    pub local_temp: u16,            // u16 @ 22
    pub energy_gradient: u16,       // u16 @ 24
    pub solvent_exposure: u16,      // u16 @ 26
    pub wavefront_velocity: u16,    // u16 @ 28
    pub wavefront_coherence: u16,   // u16 @ 30
    pub phase_id: PhaseId,          // u8  @ 32
    pub tcl_flags: u8,              // u8  @ 33
    // 2 bytes implicit trailing padding → sizeof = 36
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SpikeInput {
    pub voxel_x: u32,
    pub voxel_y: u32,
    pub voxel_z: u32,
    pub timestamp: u32,
    pub amplitude: f32,
    pub local_temp: f32,
    pub energy_gradient: f32,
    pub solvent_exposure: f32,
    pub phase_id: u8,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SpatialRegion {
    pub x_min: u32, pub x_max: u32,
    pub y_min: u32, pub y_max: u32,
    pub z_min: u32, pub z_max: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HysteresisResult {
    pub heating_spike_rate: f32,
    pub cooling_spike_rate: f32,
    pub asymmetry_score: f32,
    pub avalanche_size_ratio: f32,
    pub wavefront_coherence_ratio: f32,
    pub heating_spike_count: u32,
    pub cooling_spike_count: u32,
    pub is_hysteretic: bool,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CcnsClass {
    Soc = 0,
    NearCritical = 1,
    Barrier = 2,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CcnsResult {
    pub tau: f32,
    pub classification: CcnsClass,
    pub tau_stderr: f32,
    pub n_avalanches: u32,
    pub druggability: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TideDecomposition {
    pub residue_id: u32,
    pub causal_dg: f32,
    pub transfer_entropy: f32,
    pub fisher_info: f32,
    pub kl_divergence: f32,
    pub n_causal_spikes: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AvalancheStats {
    pub id: AvalancheId,
    pub size: u32,
    pub duration: u32,
    pub spatial_extent: f32,
    pub seed_voxel: MortonCode,
    pub phase: PhaseId,
    pub tau_local: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct WavefrontStats {
    pub id: WavefrontId,
    pub origin: MortonCode,
    pub birth_time: u32,
    pub death_time: u32,
    pub spike_count: u32,
    pub mean_velocity: f32,
    pub mean_coherence: f32,
    pub spatial_extent: f32,
    pub phase: PhaseId,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct CausalSubgraph {
    pub events: *mut SpikeEvent,
    pub count: u32,
    pub parent_indices: *mut u32,
}

#[repr(C)]
pub struct SdstConfig {
    pub grid_nx: u32,
    pub grid_ny: u32,
    pub grid_nz: u32,
    pub grid_spacing: f32,
    pub hash_table_capacity: u32,
    pub max_spike_events: u32,
    pub max_wavefronts: u32,
    pub wavefront_merge_dist: f32,
    pub wavefront_max_dt: u32,
    pub avalanche_spatial_cutoff: f32,
    pub avalanche_max_gap: u32,
    pub phase_boundaries: [u32; 6],
    pub ccns_soc_threshold: f32,
    pub ccns_barrier_threshold: f32,
    pub num_streams: u32,
    pub device_id: c_int,
}

extern "C" {
    // Lifecycle
    pub fn sdst_default_config() -> SdstConfig;
    pub fn sdst_create(config: *const SdstConfig, handle: *mut SdstHandle) -> SdstError;
    pub fn sdst_destroy(handle: SdstHandle) -> SdstError;
    pub fn sdst_reset(handle: SdstHandle) -> SdstError;
    pub fn sdst_event_count(handle: SdstHandle, count: *mut u32) -> SdstError;
    pub fn sdst_memory_usage(handle: SdstHandle, bytes: *mut usize) -> SdstError;

    // Insertion
    pub fn sdst_insert_spikes(
        handle: SdstHandle,
        events: *mut SpikeEvent,
        count: u32,
        stream: *mut c_void,
    ) -> SdstError;
    pub fn sdst_insert_raw(
        handle: SdstHandle,
        inputs: *const SpikeInput,
        count: u32,
        stream: *mut c_void,
    ) -> SdstError;

    // Queries
    pub fn sdst_query_region(
        handle: SdstHandle,
        region: *const SpatialRegion,
        events: *mut *mut SpikeEvent,
        count: *mut u32,
        stream: *mut c_void,
    ) -> SdstError;
    pub fn sdst_query_voxel(
        handle: SdstHandle,
        x: u32, y: u32, z: u32,
        events: *mut *mut SpikeEvent,
        count: *mut u32,
        stream: *mut c_void,
    ) -> SdstError;
    pub fn sdst_query_timerange(
        handle: SdstHandle,
        t_start: u32, t_end: u32,
        events: *mut *mut SpikeEvent,
        count: *mut u32,
        stream: *mut c_void,
    ) -> SdstError;

    // Causal
    pub fn sdst_causal_subgraph(
        handle: SdstHandle,
        root: SpikeId,
        max_depth: u32,
        graph: *mut CausalSubgraph,
        stream: *mut c_void,
    ) -> SdstError;
    pub fn sdst_free_subgraph(graph: *mut CausalSubgraph) -> SdstError;

    // CCNS
    pub fn sdst_ccns_region(
        handle: SdstHandle,
        region: *const SpatialRegion,
        result: *mut CcnsResult,
        stream: *mut c_void,
    ) -> SdstError;

    // Hysteresis
    pub fn sdst_hysteresis_region(
        handle: SdstHandle,
        region: *const SpatialRegion,
        threshold: f32,
        result: *mut HysteresisResult,
        stream: *mut c_void,
    ) -> SdstError;
    pub fn sdst_hysteresis_scan(
        handle: SdstHandle,
        threshold: f32,
        results: *mut *mut HysteresisResult,
        regions: *mut *mut SpatialRegion,
        count: *mut u32,
        stream: *mut c_void,
    ) -> SdstError;

    // Wavefronts
    pub fn sdst_wavefront_stats(
        handle: SdstHandle,
        phase_filter: c_int,
        stats: *mut *mut WavefrontStats,
        count: *mut u32,
        stream: *mut c_void,
    ) -> SdstError;
    pub fn sdst_wavefront_path(
        handle: SdstHandle,
        wavefront: WavefrontId,
        events: *mut *mut SpikeEvent,
        count: *mut u32,
        stream: *mut c_void,
    ) -> SdstError;

    // TIDE (Plan C) — h_residue_map is a host pointer, linear-voxel-indexed
    pub fn sdst_tide_decomposition(
        handle: SdstHandle,
        pocket: *const SpatialRegion,
        h_residue_map: *const u32,  // host pointer, size = grid_nx*grid_ny*grid_nz
        n_residues: u32,
        decomp: *mut *mut TideDecomposition,
        count: *mut u32,
        stream: *mut c_void,
    ) -> SdstError;

    // DCC
    pub fn sdst_compute_dcc(
        handle: SdstHandle,
        known_sites: *const f32,
        n_known: u32,
        dcc: *mut *mut f32,
        centroids: *mut *mut f32,
        n_detected: *mut u32,
        stream: *mut c_void,
    ) -> SdstError;

    // Serialization
    pub fn sdst_save(handle: SdstHandle, path: *const c_char) -> SdstError;
    pub fn sdst_load(path: *const c_char, handle: *mut SdstHandle) -> SdstError;

    // Debug
    pub fn sdst_print_stats(handle: SdstHandle) -> SdstError;

    // --- 7 previously missing declarations ---

    /// Return human-readable error string (static lifetime, never free).
    pub fn sdst_error_string(err: SdstError) -> *const c_char;

    /// Avalanche statistics, optionally filtered by phase (-1 = all).
    /// Caller must free *out_stats with libc free().
    pub fn sdst_avalanche_stats(
        handle: SdstHandle,
        phase_filter: c_int,
        out_stats: *mut *mut AvalancheStats,
        out_count: *mut u32,
        stream: *mut c_void,
    ) -> SdstError;

    /// Causal subgraph for all spikes in a spatial region.
    pub fn sdst_causal_subgraph_region(
        handle: SdstHandle,
        region: *const SpatialRegion,
        max_depth: u32,
        out_graph: *mut CausalSubgraph,
        stream: *mut c_void,
    ) -> SdstError;

    /// CCNS for all automatically detected pockets (LEGACY host-side).
    /// Caller must free *out_results and *out_regions with libc free().
    pub fn sdst_ccns_all_pockets(
        handle: SdstHandle,
        out_results: *mut *mut CcnsResult,
        out_regions: *mut *mut SpatialRegion,
        out_count: *mut u32,
        stream: *mut c_void,
    ) -> SdstError;

    /// GPU-native CCNS for all spatial tiles (sort-reduce + CSN estimator).
    /// Caller must free *out_results and *out_regions with libc free().
    pub fn sdst_ccns_all_pockets_gpu(
        handle: SdstHandle,
        out_results: *mut *mut CcnsResult,
        out_regions: *mut *mut SpatialRegion,
        out_count: *mut u32,
        stream: *mut c_void,
    ) -> SdstError;

    /// Combined spatial + temporal query.
    pub fn sdst_query_region_timerange(
        handle: SdstHandle,
        region: *const SpatialRegion,
        t_start: u32,
        t_end: u32,
        out_events: *mut *mut SpikeEvent,
        out_count: *mut u32,
        stream: *mut c_void,
    ) -> SdstError;

    /// Validate internal consistency. Returns SDST_SUCCESS if all checks pass.
    pub fn sdst_validate(handle: SdstHandle) -> SdstError;

    /// Find wavefronts that passed through a specific region.
    /// Caller must free *out_stats with libc free().
    pub fn sdst_wavefronts_through_region(
        handle: SdstHandle,
        region: *const SpatialRegion,
        out_stats: *mut *mut WavefrontStats,
        out_count: *mut u32,
        stream: *mut c_void,
    ) -> SdstError;
}

// ============================================================
// Safe Rust wrapper
// ============================================================

pub struct Sdst {
    handle: SdstHandle,
}

impl Sdst {
    pub fn new(config: &SdstConfig) -> Result<Self, SdstError> {
        let mut handle: SdstHandle = std::ptr::null_mut();
        unsafe { sdst_create(config, &mut handle).check()? };
        Ok(Self { handle })
    }

    pub fn default() -> Result<Self, SdstError> {
        let config = unsafe { sdst_default_config() };
        Self::new(&config)
    }

    pub fn insert_raw(&self, inputs: &[SpikeInput]) -> Result<(), SdstError> {
        unsafe {
            sdst_insert_raw(
                self.handle,
                inputs.as_ptr(),
                inputs.len() as u32,
                std::ptr::null_mut(),
            ).check()
        }
    }

    pub fn event_count(&self) -> Result<u32, SdstError> {
        let mut count = 0u32;
        unsafe { sdst_event_count(self.handle, &mut count).check()? };
        Ok(count)
    }

    pub fn memory_usage(&self) -> Result<usize, SdstError> {
        let mut bytes = 0usize;
        unsafe { sdst_memory_usage(self.handle, &mut bytes).check()? };
        Ok(bytes)
    }

    pub fn ccns_region(&self, region: &SpatialRegion) -> Result<CcnsResult, SdstError> {
        let mut result = std::mem::MaybeUninit::<CcnsResult>::uninit();
        unsafe {
            sdst_ccns_region(
                self.handle,
                region,
                result.as_mut_ptr(),
                std::ptr::null_mut(),
            ).check()?;
            Ok(result.assume_init())
        }
    }

    pub fn hysteresis_region(&self, region: &SpatialRegion, threshold: f32)
        -> Result<HysteresisResult, SdstError>
    {
        let mut result = std::mem::MaybeUninit::<HysteresisResult>::uninit();
        unsafe {
            sdst_hysteresis_region(
                self.handle,
                region,
                threshold,
                result.as_mut_ptr(),
                std::ptr::null_mut(),
            ).check()?;
            Ok(result.assume_init())
        }
    }

    pub fn print_stats(&self) -> Result<(), SdstError> {
        unsafe { sdst_print_stats(self.handle).check() }
    }

    pub fn save(&self, path: &str) -> Result<(), SdstError> {
        let c_path = std::ffi::CString::new(path).map_err(|_| SdstError::ErrorInvalidParam)?;
        unsafe { sdst_save(self.handle, c_path.as_ptr()).check() }
    }

    pub fn reset(&self) -> Result<(), SdstError> {
        unsafe { sdst_reset(self.handle).check() }
    }

    /// Validate internal consistency (all checks must pass before analyzing results).
    pub fn validate(&self) -> Result<(), SdstError> {
        unsafe { sdst_validate(self.handle).check() }
    }

    /// Run TIDE (Transfer entropy-Integrated Decomposed Energetics) decomposition
    /// for a target pocket region.
    ///
    /// `h_residue_map` is a dense linear-voxel-indexed array of size
    /// `grid_nx × grid_ny × grid_nz`. Index = `x + y*grid_nx + z*grid_nx*grid_ny`.
    /// Use `u32::MAX` for empty (no-residue) voxels.
    pub fn tide_decomposition(
        &self,
        pocket: &SpatialRegion,
        h_residue_map: &[u32],
        n_residues: u32,
    ) -> Result<Vec<TideDecomposition>, SdstError> {
        let mut ptr: *mut TideDecomposition = std::ptr::null_mut();
        let mut count = 0u32;
        unsafe {
            sdst_tide_decomposition(
                self.handle,
                pocket,
                h_residue_map.as_ptr(),
                n_residues,
                &mut ptr,
                &mut count,
                std::ptr::null_mut(),
            ).check()?;
            if count == 0 || ptr.is_null() {
                return Ok(Vec::new());
            }
            let v = std::slice::from_raw_parts(ptr, count as usize).to_vec();
            free(ptr as *mut c_void);
            Ok(v)
        }
    }

    /// Avalanche statistics, optionally filtered by phase (-1 = all phases).
    /// Returns owned Vec; caller is responsible for memory.
    pub fn avalanche_stats(&self, phase_filter: i32) -> Result<Vec<AvalancheStats>, SdstError> {
        let mut ptr: *mut AvalancheStats = std::ptr::null_mut();
        let mut count = 0u32;
        unsafe {
            sdst_avalanche_stats(
                self.handle,
                phase_filter as c_int,
                &mut ptr,
                &mut count,
                std::ptr::null_mut(),
            ).check()?;
            if count == 0 || ptr.is_null() {
                return Ok(Vec::new());
            }
            let v = std::slice::from_raw_parts(ptr, count as usize).to_vec();
            free(ptr as *mut c_void);
            Ok(v)
        }
    }

    /// GPU-native CCNS for all spatial tiles (sort-reduce + CSN estimator).
    /// Returns (ccns_results, regions) pairs. Fully GPU-resident, no host downloads.
    pub fn ccns_all_pockets_gpu(&self) -> Result<Vec<(CcnsResult, SpatialRegion)>, SdstError> {
        let mut results_ptr: *mut CcnsResult = std::ptr::null_mut();
        let mut regions_ptr: *mut SpatialRegion = std::ptr::null_mut();
        let mut count = 0u32;
        unsafe {
            sdst_ccns_all_pockets_gpu(
                self.handle,
                &mut results_ptr,
                &mut regions_ptr,
                &mut count,
                std::ptr::null_mut(),
            ).check()?;
            if count == 0 || results_ptr.is_null() {
                return Ok(Vec::new());
            }
            let results = std::slice::from_raw_parts(results_ptr, count as usize);
            let regions = std::slice::from_raw_parts(regions_ptr, count as usize);
            let v: Vec<(CcnsResult, SpatialRegion)> = results.iter().copied()
                .zip(regions.iter().copied())
                .collect();
            free(results_ptr as *mut c_void);
            free(regions_ptr as *mut c_void);
            Ok(v)
        }
    }

    /// CCNS for all automatically detected pockets (LEGACY host-side).
    /// Returns (ccns_results, regions) pairs.
    pub fn ccns_all_pockets(&self) -> Result<Vec<(CcnsResult, SpatialRegion)>, SdstError> {
        let mut results_ptr: *mut CcnsResult = std::ptr::null_mut();
        let mut regions_ptr: *mut SpatialRegion = std::ptr::null_mut();
        let mut count = 0u32;
        unsafe {
            sdst_ccns_all_pockets(
                self.handle,
                &mut results_ptr,
                &mut regions_ptr,
                &mut count,
                std::ptr::null_mut(),
            ).check()?;
            if count == 0 || results_ptr.is_null() {
                return Ok(Vec::new());
            }
            let results = std::slice::from_raw_parts(results_ptr, count as usize);
            let regions = std::slice::from_raw_parts(regions_ptr, count as usize);
            let v: Vec<(CcnsResult, SpatialRegion)> = results.iter().copied()
                .zip(regions.iter().copied())
                .collect();
            free(results_ptr as *mut c_void);
            free(regions_ptr as *mut c_void);
            Ok(v)
        }
    }
}

impl Drop for Sdst {
    fn drop(&mut self) {
        unsafe { sdst_destroy(self.handle); }
    }
}

// Safety: SDST handles are thread-safe (CUDA streams provide synchronization)
unsafe impl Send for Sdst {}
unsafe impl Sync for Sdst {}

// ============================================================
// Layout verification tests (run with: cargo test)
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spike_event_size() {
        // Must match C: sizeof(SpikeEvent) = 36 (GATE 0 verified)
        assert_eq!(std::mem::size_of::<SpikeEvent>(), 36,
            "SpikeEvent size mismatch: C=36, Rust={}",
            std::mem::size_of::<SpikeEvent>());
    }

    #[test]
    fn test_spike_event_field_offsets() {
        use std::mem::offset_of;
        assert_eq!(offset_of!(SpikeEvent, voxel),              0);
        assert_eq!(offset_of!(SpikeEvent, timestamp),          4);
        assert_eq!(offset_of!(SpikeEvent, parent_spike),       8);
        assert_eq!(offset_of!(SpikeEvent, avalanche_id),      12);
        assert_eq!(offset_of!(SpikeEvent, wavefront_id),      16);
        assert_eq!(offset_of!(SpikeEvent, amplitude),         20);
        assert_eq!(offset_of!(SpikeEvent, local_temp),        22);
        assert_eq!(offset_of!(SpikeEvent, energy_gradient),   24);
        assert_eq!(offset_of!(SpikeEvent, solvent_exposure),  26);
        assert_eq!(offset_of!(SpikeEvent, wavefront_velocity),28);
        assert_eq!(offset_of!(SpikeEvent, wavefront_coherence),30);
        assert_eq!(offset_of!(SpikeEvent, phase_id),          32);
        assert_eq!(offset_of!(SpikeEvent, tcl_flags),         33);
    }

    #[test]
    fn test_other_struct_sizes() {
        // Sanity checks — these don't need to match specific values,
        // just verify they're non-zero and reasonable
        assert!(std::mem::size_of::<SpikeInput>() > 0);
        assert!(std::mem::size_of::<SdstConfig>() > 0);
        assert!(std::mem::size_of::<HysteresisResult>() > 0);
        assert!(std::mem::size_of::<CcnsResult>() > 0);
        assert!(std::mem::size_of::<AvalancheStats>() > 0);
        assert!(std::mem::size_of::<WavefrontStats>() > 0);
        assert!(std::mem::size_of::<TideDecomposition>() > 0);
    }
}
