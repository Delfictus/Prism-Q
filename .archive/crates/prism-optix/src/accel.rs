// [STAGE-2-BVH] OptiX Acceleration Structure (BVH)
//
// Builds and manages Bounding Volume Hierarchy (BVH) for efficient ray tracing.
// Uses OptiX built-in spheres for optimal molecular visualization performance.
// Targets <100ms build time and <10ms refit time for 100K atoms.

use crate::context::OptixContext;
use crate::error::{OptixError, Result};
use crate::loader::OptixApi;
use optix_sys::*;
use std::mem::MaybeUninit;
use std::ptr;

// CUDA Driver API for memory management (via cudarc)
use cudarc::driver::sys::{cuMemAlloc_v2, cuMemFree_v2, CUresult};

/// CUDA success return code
const CUDA_SUCCESS: CUresult = CUresult::CUDA_SUCCESS;

/// BVH build input type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BvhInputType {
    /// Triangle mesh (for surfaces)
    Triangles,
    /// Spheres (for atoms) - optimized built-in primitive
    Spheres,
    /// Custom primitives (for arbitrary AABBs)
    Custom,
    /// Instances (for hierarchical BVH)
    Instances,
}

/// BVH build flags
#[derive(Debug, Clone, Copy)]
pub struct BvhBuildFlags {
    /// Allow updating (refit) without full rebuild
    pub allow_update: bool,
    /// Allow compaction for smaller memory footprint
    pub allow_compaction: bool,
    /// Prefer fast trace over fast build
    pub prefer_fast_trace: bool,
    /// Allow random vertex access
    pub allow_random_vertex_access: bool,
}

impl Default for BvhBuildFlags {
    fn default() -> Self {
        Self {
            allow_update: true,      // Enable refit for dynamic atoms
            allow_compaction: false, // Disable for speed
            prefer_fast_trace: true, // Optimize for query performance
            allow_random_vertex_access: false,
        }
    }
}

impl BvhBuildFlags {
    /// Flags optimized for dynamic atom updates (refit)
    pub fn dynamic() -> Self {
        Self {
            allow_update: true,
            allow_compaction: false,
            prefer_fast_trace: false, // Prefer fast refit
            allow_random_vertex_access: false,
        }
    }

    /// Flags optimized for static geometry (no updates)
    pub fn static_geometry() -> Self {
        Self {
            allow_update: false,
            allow_compaction: true,
            prefer_fast_trace: true,
            allow_random_vertex_access: false,
        }
    }

    /// Convert to OptiX build flags bitmask
    fn to_optix_flags(&self) -> u32 {
        let mut flags = OptixBuildFlags::OPTIX_BUILD_FLAG_NONE as u32;
        if self.allow_update {
            flags |= OptixBuildFlags::OPTIX_BUILD_FLAG_ALLOW_UPDATE as u32;
        }
        if self.allow_compaction {
            flags |= OptixBuildFlags::OPTIX_BUILD_FLAG_ALLOW_COMPACTION as u32;
        }
        if self.prefer_fast_trace {
            flags |= OptixBuildFlags::OPTIX_BUILD_FLAG_PREFER_FAST_TRACE as u32;
        }
        if self.allow_random_vertex_access {
            flags |= OptixBuildFlags::OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS as u32;
        }
        flags
    }
}

/// OptiX acceleration structure (BVH)
///
/// Manages a BVH for efficient spatial queries using OptiX built-in spheres.
/// Supports both full builds and fast refits for dynamic geometry.
///
/// # Performance Targets
///
/// - Build: <100ms for 100K atoms
/// - Refit: <10ms for 100K atoms (when positions change)
///
/// # Safety
///
/// This struct uses unsafe FFI calls to OptiX and CUDA. The caller must ensure:
/// - GPU device pointers are valid and point to allocated memory
/// - The OptixContext outlives this AccelStructure
/// - Positions are float3 (xyz) format, radii are single floats
pub struct AccelStructure {
    /// OptiX context this BVH belongs to
    context: *const OptixContext,

    /// BVH traversable handle (valid after successful build)
    handle: OptixTraversableHandle,

    /// GPU buffer for BVH data (allocated via cuMemAlloc)
    device_buffer: CUdeviceptr,
    device_buffer_size: usize,

    /// Number of spheres in this BVH
    num_spheres: usize,

    /// Whether this BVH can be updated (refitted)
    can_update: bool,

    /// Cached build flags for refit operations
    build_flags: u32,
}

impl AccelStructure {
    /// Build a BVH for spheres (atoms)
    ///
    /// Uses OptiX built-in sphere primitives for optimal performance.
    /// Each atom is represented as a sphere with center at position and given radius.
    ///
    /// # Arguments
    ///
    /// * `context` - OptiX context
    /// * `positions_gpu` - Device pointer to atom positions (float3: x, y, z per atom)
    /// * `radii_gpu` - Device pointer to atom radii (float per atom)
    /// * `num_atoms` - Number of atoms (spheres)
    /// * `flags` - Build flags controlling update/compaction behavior
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `positions_gpu` points to `num_atoms * 3` floats on GPU
    /// - `radii_gpu` points to `num_atoms` floats on GPU
    /// - Both buffers remain valid for the lifetime of this AccelStructure
    ///
    /// # Example
    ///
    /// ```no_run
    /// use prism_optix::{OptixContext, AccelStructure, BvhBuildFlags};
    /// use cudarc::driver::CudaDevice;
    ///
    /// OptixContext::init().unwrap();
    /// let cuda = CudaDevice::new(0).unwrap();
    /// let ctx = OptixContext::new(cuda.cu_primary_ctx(), true).unwrap();
    ///
    /// // Upload atom positions and radii to GPU
    /// // let d_positions = cuda.htod_copy(positions).unwrap();
    /// // let d_radii = cuda.htod_copy(radii).unwrap();
    ///
    /// // Build BVH using built-in spheres
    /// // let bvh = AccelStructure::build_spheres(
    /// //     &ctx, d_positions.device_ptr(), d_radii.device_ptr(),
    /// //     num_atoms, BvhBuildFlags::dynamic()
    /// // ).unwrap();
    /// ```
    pub fn build_spheres(
        context: &OptixContext,
        positions_gpu: CUdeviceptr,
        radii_gpu: CUdeviceptr,
        num_atoms: usize,
        flags: BvhBuildFlags,
    ) -> Result<Self> {
        log::info!(
            "Building BVH for {} atoms using OptiX built-in spheres (allow_update={}, prefer_fast_trace={})",
            num_atoms,
            flags.allow_update,
            flags.prefer_fast_trace
        );

        if num_atoms == 0 {
            return Err(OptixError::InvalidValue(
                "Cannot build BVH with zero atoms".to_string(),
            ));
        }

        let api = OptixApi::get()?;
        let optix_ctx = context.handle();
        let build_flags = flags.to_optix_flags();

        // ========================================================================
        // Step 1: Configure Build Options
        // ========================================================================
        let build_options = OptixAccelBuildOptions {
            buildFlags: build_flags,
            operation: OptixBuildOperation::OPTIX_BUILD_OPERATION_BUILD,
            motionOptions: OptixMotionOptions {
                numKeys: 1, // No motion blur (single key = static)
                flags: 0,
                timeBegin: 0.0,
                timeEnd: 1.0,
            },
        };

        // ========================================================================
        // Step 2: Configure Sphere Build Input
        // ========================================================================
        // OptiX expects vertexBuffers and radiusBuffers to be arrays of device pointers
        // (one per motion key). We have 1 motion key, so arrays of size 1.
        let vertex_buffer_ptr: CUdeviceptr = positions_gpu;
        let radius_buffer_ptr: CUdeviceptr = radii_gpu;

        // Geometry flag: no special behavior
        let geometry_flag: u32 = OptixGeometryFlags::OPTIX_GEOMETRY_FLAG_NONE as u32;

        let sphere_input = OptixBuildInputSphereArray {
            vertexBuffers: &vertex_buffer_ptr as *const CUdeviceptr,
            vertexStrideInBytes: (std::mem::size_of::<f32>() * 3) as u32, // float3 stride
            numVertices: num_atoms as u32,
            radiusBuffers: &radius_buffer_ptr as *const CUdeviceptr,
            radiusStrideInBytes: std::mem::size_of::<f32>() as u32, // float stride
            singleRadius: 0, // Each sphere has its own radius
            flags: &geometry_flag as *const u32,
            numSbtRecords: 1,
            sbtIndexOffsetBuffer: 0, // No per-primitive SBT offset
            sbtIndexOffsetSizeInBytes: 0,
            sbtIndexOffsetStrideInBytes: 0,
            primitiveIndexOffset: 0,
        };

        let build_input = OptixBuildInput {
            type_: OptixBuildInputType::OPTIX_BUILD_INPUT_TYPE_SPHERES,
            __bindgen_anon_1: OptixBuildInput__bindgen_ty_1 {
                sphereArray: sphere_input,
            },
        };

        // ========================================================================
        // Step 3: Compute Memory Requirements
        // ========================================================================
        let mut buffer_sizes = MaybeUninit::<OptixAccelBufferSizes>::uninit();
        let result = unsafe {
            api.accel_compute_memory_usage(
                optix_ctx,
                &build_options,
                &build_input,
                1, // numBuildInputs
                buffer_sizes.as_mut_ptr(),
            )
        };

        if result != OptixResult::OPTIX_SUCCESS {
            return Err(OptixError::from(result));
        }

        let buffer_sizes = unsafe { buffer_sizes.assume_init() };
        let output_size = buffer_sizes.outputSizeInBytes;
        let temp_size = buffer_sizes.tempSizeInBytes;

        log::debug!(
            "BVH memory requirements: output={} bytes ({:.2} MB), temp={} bytes ({:.2} MB)",
            output_size,
            output_size as f64 / 1024.0 / 1024.0,
            temp_size,
            temp_size as f64 / 1024.0 / 1024.0
        );

        // ========================================================================
        // Step 4: Allocate GPU Buffers
        // ========================================================================
        let mut output_buffer: CUdeviceptr = 0;
        let cuda_result = unsafe { cuMemAlloc_v2(&mut output_buffer, output_size) };
        if cuda_result != CUDA_SUCCESS {
            return Err(OptixError::AllocationFailed(format!(
                "Failed to allocate output buffer ({} bytes): CUDA error {:?}",
                output_size, cuda_result
            )));
        }

        let mut temp_buffer: CUdeviceptr = 0;
        let cuda_result = unsafe { cuMemAlloc_v2(&mut temp_buffer, temp_size) };
        if cuda_result != CUDA_SUCCESS {
            // Clean up output buffer on failure
            unsafe { cuMemFree_v2(output_buffer) };
            return Err(OptixError::AllocationFailed(format!(
                "Failed to allocate temp buffer ({} bytes): CUDA error {:?}",
                temp_size, cuda_result
            )));
        }

        // ========================================================================
        // Step 5: Build BVH
        // ========================================================================
        let mut handle: OptixTraversableHandle = 0;
        let result = unsafe {
            api.accel_build(
                optix_ctx,
                ptr::null_mut(), // CUstream: null = default stream
                &build_options,
                &build_input,
                1, // numBuildInputs
                temp_buffer,
                temp_size,
                output_buffer,
                output_size,
                &mut handle,
                ptr::null(), // emittedProperties
                0,           // numEmittedProperties
            )
        };

        // Free temp buffer immediately (no longer needed)
        unsafe { cuMemFree_v2(temp_buffer) };

        // Check build result
        if result != OptixResult::OPTIX_SUCCESS {
            // Clean up output buffer on failure
            unsafe { cuMemFree_v2(output_buffer) };
            return Err(OptixError::from(result));
        }

        log::info!(
            "✅ BVH build complete: {} spheres, handle=0x{:x}, size={} bytes ({:.2} MB)",
            num_atoms,
            handle,
            output_size,
            output_size as f64 / 1024.0 / 1024.0
        );

        Ok(Self {
            context: context as *const OptixContext,
            handle,
            device_buffer: output_buffer,
            device_buffer_size: output_size,
            num_spheres: num_atoms,
            can_update: flags.allow_update,
            build_flags,
        })
    }

    /// Refit the BVH with updated positions (fast update)
    ///
    /// Much faster than a full rebuild (~10-100x faster). Use when atom
    /// positions change but topology (number of atoms) remains the same.
    ///
    /// # Arguments
    ///
    /// * `positions_gpu` - Device pointer to updated atom positions (float3: x, y, z)
    /// * `radii_gpu` - Device pointer to atom radii (float per atom)
    ///
    /// # Performance
    ///
    /// Target: <10ms for 100K atoms
    ///
    /// # Safety
    ///
    /// The caller must ensure the buffers point to the same number of atoms
    /// as the original build.
    pub fn refit(&mut self, positions_gpu: CUdeviceptr, radii_gpu: CUdeviceptr) -> Result<()> {
        if !self.can_update {
            return Err(OptixError::InvalidOperation(
                "BVH was not built with allow_update flag".to_string(),
            ));
        }

        if self.handle == 0 {
            return Err(OptixError::InvalidOperation(
                "Cannot refit uninitialized BVH".to_string(),
            ));
        }

        log::debug!("Refitting BVH with {} spheres", self.num_spheres);

        let api = OptixApi::get()?;
        let context = unsafe { &*self.context };
        let optix_ctx = context.handle();

        // ========================================================================
        // Configure Refit Options (same as build, but UPDATE operation)
        // ========================================================================
        let build_options = OptixAccelBuildOptions {
            buildFlags: self.build_flags,
            operation: OptixBuildOperation::OPTIX_BUILD_OPERATION_UPDATE,
            motionOptions: OptixMotionOptions {
                numKeys: 1,
                flags: 0,
                timeBegin: 0.0,
                timeEnd: 1.0,
            },
        };

        // ========================================================================
        // Configure Sphere Input with Updated Positions
        // ========================================================================
        let vertex_buffer_ptr: CUdeviceptr = positions_gpu;
        let radius_buffer_ptr: CUdeviceptr = radii_gpu;
        let geometry_flag: u32 = OptixGeometryFlags::OPTIX_GEOMETRY_FLAG_NONE as u32;

        let sphere_input = OptixBuildInputSphereArray {
            vertexBuffers: &vertex_buffer_ptr as *const CUdeviceptr,
            vertexStrideInBytes: (std::mem::size_of::<f32>() * 3) as u32,
            numVertices: self.num_spheres as u32,
            radiusBuffers: &radius_buffer_ptr as *const CUdeviceptr,
            radiusStrideInBytes: std::mem::size_of::<f32>() as u32,
            singleRadius: 0,
            flags: &geometry_flag as *const u32,
            numSbtRecords: 1,
            sbtIndexOffsetBuffer: 0,
            sbtIndexOffsetSizeInBytes: 0,
            sbtIndexOffsetStrideInBytes: 0,
            primitiveIndexOffset: 0,
        };

        let build_input = OptixBuildInput {
            type_: OptixBuildInputType::OPTIX_BUILD_INPUT_TYPE_SPHERES,
            __bindgen_anon_1: OptixBuildInput__bindgen_ty_1 {
                sphereArray: sphere_input,
            },
        };

        // ========================================================================
        // Allocate Temp Buffer for Refit
        // ========================================================================
        // Compute memory for update operation
        let mut buffer_sizes = MaybeUninit::<OptixAccelBufferSizes>::uninit();
        let result = unsafe {
            api.accel_compute_memory_usage(
                optix_ctx,
                &build_options,
                &build_input,
                1,
                buffer_sizes.as_mut_ptr(),
            )
        };

        if result != OptixResult::OPTIX_SUCCESS {
            return Err(OptixError::from(result));
        }

        let buffer_sizes = unsafe { buffer_sizes.assume_init() };
        let temp_size = buffer_sizes.tempUpdateSizeInBytes;

        let mut temp_buffer: CUdeviceptr = 0;
        if temp_size > 0 {
            let cuda_result = unsafe { cuMemAlloc_v2(&mut temp_buffer, temp_size) };
            if cuda_result != CUDA_SUCCESS {
                return Err(OptixError::AllocationFailed(format!(
                    "Failed to allocate refit temp buffer: CUDA error {:?}",
                    cuda_result
                )));
            }
        }

        // ========================================================================
        // Perform Refit
        // ========================================================================
        let result = unsafe {
            api.accel_build(
                optix_ctx,
                ptr::null_mut(), // default stream
                &build_options,
                &build_input,
                1,
                temp_buffer,
                temp_size,
                self.device_buffer,
                self.device_buffer_size,
                &mut self.handle,
                ptr::null(),
                0,
            )
        };

        // Free temp buffer
        if temp_size > 0 {
            unsafe { cuMemFree_v2(temp_buffer) };
        }

        if result != OptixResult::OPTIX_SUCCESS {
            return Err(OptixError::from(result));
        }

        log::debug!("✅ BVH refit complete");
        Ok(())
    }

    /// Get the traversable handle for ray tracing
    ///
    /// This handle is passed to OptiX ray tracing calls.
    pub fn handle(&self) -> OptixTraversableHandle {
        self.handle
    }

    /// Check if this BVH can be updated (refitted)
    pub fn can_update(&self) -> bool {
        self.can_update
    }

    /// Get device buffer size in bytes
    pub fn device_buffer_size(&self) -> usize {
        self.device_buffer_size
    }

    /// Get number of spheres in this BVH
    pub fn num_spheres(&self) -> usize {
        self.num_spheres
    }

    // Legacy API compatibility
    #[deprecated(since = "0.2.0", note = "Use build_spheres instead")]
    pub fn build_custom_primitives(
        context: &OptixContext,
        positions_gpu: CUdeviceptr,
        radii_gpu: CUdeviceptr,
        num_atoms: usize,
        flags: BvhBuildFlags,
    ) -> Result<Self> {
        Self::build_spheres(context, positions_gpu, radii_gpu, num_atoms, flags)
    }
}

impl Drop for AccelStructure {
    fn drop(&mut self) {
        if self.device_buffer != 0 {
            log::debug!(
                "Dropping AccelStructure: freeing {} bytes ({:.2} MB)",
                self.device_buffer_size,
                self.device_buffer_size as f64 / 1024.0 / 1024.0
            );
            unsafe {
                cuMemFree_v2(self.device_buffer);
            }
        }
    }
}

// AccelStructure is Send (can be moved between threads)
// The OptiX handle and device buffer are thread-safe once created
unsafe impl Send for AccelStructure {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bvh_flags_default() {
        let flags = BvhBuildFlags::default();
        assert!(flags.allow_update);
        assert!(flags.prefer_fast_trace);
        assert!(!flags.allow_compaction);
    }

    #[test]
    fn test_bvh_flags_dynamic() {
        let flags = BvhBuildFlags::dynamic();
        assert!(flags.allow_update);
        assert!(!flags.prefer_fast_trace); // Prefer fast refit
    }

    #[test]
    fn test_bvh_flags_static() {
        let flags = BvhBuildFlags::static_geometry();
        assert!(!flags.allow_update);
        assert!(flags.allow_compaction);
        assert!(flags.prefer_fast_trace);
    }

    #[test]
    fn test_optix_flags_conversion() {
        let flags = BvhBuildFlags::default();
        let optix_flags = flags.to_optix_flags();

        // Should have ALLOW_UPDATE and PREFER_FAST_TRACE
        assert!(optix_flags & (OptixBuildFlags::OPTIX_BUILD_FLAG_ALLOW_UPDATE as u32) != 0);
        assert!(optix_flags & (OptixBuildFlags::OPTIX_BUILD_FLAG_PREFER_FAST_TRACE as u32) != 0);
        assert!(optix_flags & (OptixBuildFlags::OPTIX_BUILD_FLAG_ALLOW_COMPACTION as u32) == 0);
    }

    #[test]
    fn test_optix_flags_none() {
        let flags = BvhBuildFlags {
            allow_update: false,
            allow_compaction: false,
            prefer_fast_trace: false,
            allow_random_vertex_access: false,
        };
        let optix_flags = flags.to_optix_flags();
        assert_eq!(optix_flags, OptixBuildFlags::OPTIX_BUILD_FLAG_NONE as u32);
    }
}
