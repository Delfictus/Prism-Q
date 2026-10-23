//! RT Core Utilities
//!
//! Runtime detection of RT core capabilities and helper functions
//! for hardware-accelerated ray tracing features.

#[cfg(feature = "gpu")]
use cudarc::driver::CudaContext;

/// Check if the current GPU has hardware RT cores (SM 7.5+)
///
/// RT cores are available on:
/// - Turing (SM 7.5): RTX 20xx series
/// - Ampere (SM 8.x): RTX 30xx series
/// - Ada Lovelace (SM 8.9): RTX 40xx series
/// - Blackwell (SM 12.0): RTX 50xx series
///
/// While OptiX can run on older cards via software emulation,
/// it's typically slower than grid-based methods without hardware RT.
#[cfg(feature = "gpu")]
pub fn has_rt_cores() -> bool {
    match get_compute_capability() {
        Ok((major, minor)) => {
            // SM 7.5+ has hardware RT cores
            major > 7 || (major == 7 && minor >= 5)
        }
        Err(e) => {
            log::warn!("Failed to detect GPU compute capability: {}", e);
            false
        }
    }
}

/// Get the compute capability of the current GPU
#[cfg(feature = "gpu")]
pub fn get_compute_capability() -> Result<(u32, u32), String> {
    // Try to get device properties via cudarc
    // Note: cudarc doesn't expose device properties directly,
    // so we use the CUDA driver API
    unsafe {
        use cudarc::driver::sys::{cuDeviceGetAttribute, CUdevice_attribute};

        let mut major: i32 = 0;
        let mut minor: i32 = 0;

        let result_major = cuDeviceGetAttribute(
            &mut major,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            0, // device 0
        );

        if result_major != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(format!(
                "Failed to get compute capability major: {:?}",
                result_major
            ));
        }

        let result_minor = cuDeviceGetAttribute(
            &mut minor,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            0, // device 0
        );

        if result_minor != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(format!(
                "Failed to get compute capability minor: {:?}",
                result_minor
            ));
        }

        Ok((major as u32, minor as u32))
    }
}

/// Get a human-readable name for the GPU architecture based on compute capability
#[cfg(feature = "gpu")]
pub fn get_architecture_name() -> &'static str {
    match get_compute_capability() {
        Ok((major, minor)) => match (major, minor) {
            (12, _) => "Blackwell",
            (9, _) => "Hopper",
            (8, 9) => "Ada Lovelace",
            (8, 6) => "Ampere (GA10x)",
            (8, 0) => "Ampere (GA100)",
            (7, 5) => "Turing",
            (7, 0) => "Volta",
            (6, _) => "Pascal",
            (5, _) => "Maxwell",
            _ => "Unknown",
        },
        Err(_) => "Unknown",
    }
}

/// Check if OptiX is available (driver loaded)
///
/// This does a lightweight check by attempting to query device attributes.
/// The actual OptiX library loading is deferred to RtClusteringEngine::new().
#[cfg(feature = "gpu")]
pub fn is_optix_available() -> bool {
    // If we have RT cores, assume OptiX is available
    // The actual availability is confirmed when RtClusteringEngine is created
    // This avoids pulling in libloading as a direct dependency
    has_rt_cores()
}

/// Configuration for RT-accelerated clustering
#[derive(Debug, Clone)]
pub struct RtClusteringConfig {
    /// Enable RT clustering (if hardware available)
    pub enabled: bool,
    /// Neighborhood radius in Angstroms
    pub epsilon: f32,
    /// Minimum points to form a core point
    pub min_points: u32,
    /// Minimum cluster size to report
    pub min_cluster_size: u32,
    /// Rays per event for neighbor finding
    pub rays_per_event: u32,
}

impl Default for RtClusteringConfig {
    fn default() -> Self {
        Self {
            enabled: true, // Enable by default, will fallback if no RT cores
            epsilon: 5.0,
            min_points: 3,
            min_cluster_size: 100,
            rays_per_event: 32,
        }
    }
}

// Stub for non-GPU builds
#[cfg(not(feature = "gpu"))]
pub fn has_rt_cores() -> bool {
    false
}

#[cfg(not(feature = "gpu"))]
pub fn is_optix_available() -> bool {
    false
}

#[cfg(not(feature = "gpu"))]
pub fn get_architecture_name() -> &'static str {
    "No GPU"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "gpu")]
    fn test_compute_capability() {
        // This test only works on systems with NVIDIA GPUs
        if let Ok((major, minor)) = get_compute_capability() {
            println!("Compute capability: {}.{}", major, minor);
            println!("Architecture: {}", get_architecture_name());
            println!("Has RT cores: {}", has_rt_cores());
        }
    }
}
