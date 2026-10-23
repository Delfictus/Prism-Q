//! Zero-Copy Spike Data Exhaust Pipeline
//!
//! Streams spike data from GPU to system RAM via PCIe DMA without
//! interrupting the CUDA graph or involving the CPU in the data path.
//!
//! Architecture:
//!   GPU compact kernel → mapped pinned host memory (PCIe DMA) → Rust harvester thread → NVMe
//!
//! The mapped memory appears as a device pointer to the GPU, but the
//! physical memory lives in system DDR5 RAM. GPU writes go through
//! the PCIe Gen 5 DMA engine (32 GB/s) asynchronously — no SM ALU
//! cycles, no CUDA graph interruption, no stream synchronization.
//!
//! Data format: RingSpikeEvent (48 bytes per spike) with fields:
//!   timestep, voxel_idx, x, y, z, intensity, vibrational_energy,
//!   water_density, n_nearby_excited, spike_source, wavelength_nm
//!
//! The spike_source field tags each spike's origin channel:
//!   0 = LIF, 1 = UV, 2 = RAF, 3 = EFP, 4 = LADD, 5 = COFIRE
//!
//! This channel tagging is what enables the 109-model ensemble
//! teacher to learn channel-specific dynamics for distillation
//! into the zero-shot student model.

use anyhow::{Context, Result};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::sys;

/// Mapped pinned host memory for zero-copy spike streaming.
///
/// The GPU writes compacted spikes here via PCIe DMA. The CPU reads
/// them from system RAM without any GPU synchronization.
#[cfg(feature = "cuda")]
pub struct SpikeExhaustBuffer {
    /// Host pointer (system RAM, pinned)
    host_ptr: *mut u8,
    /// Device pointer (same memory, GPU-accessible via PCIe mapping)
    device_ptr: sys::CUdeviceptr,
    /// Total capacity in bytes
    capacity_bytes: usize,
    /// Total capacity in spike records (48 bytes each)
    capacity_spikes: usize,
    /// Atomic write head (updated by GPU via atomicAdd on the mapped counter)
    /// The GPU writes to device_ptr + write_head * 48.
    /// The CPU reads from host_ptr + read_head * 48.
    write_head_ptr: *mut u32, // mapped, GPU writes via atomicAdd
    write_head_device: sys::CUdeviceptr, // device pointer to the same u32
    /// CPU-side read head (only CPU touches this)
    read_head: AtomicU64,
}

#[cfg(feature = "cuda")]
impl SpikeExhaustBuffer {
    /// Allocate a mapped pinned host buffer for spike streaming.
    ///
    /// `capacity_spikes` is the number of RingSpikeEvent records the buffer
    /// can hold. The GPU writes new spikes at the tail; the CPU harvester
    /// reads from the head. Circular buffer semantics.
    ///
    /// Typical size: 10M spikes × 48 bytes = 480 MB of pinned host RAM.
    pub fn new(capacity_spikes: usize) -> Result<Self> {
        let spike_size = 48usize; // sizeof(RingSpikeEvent)
        let capacity_bytes = capacity_spikes * spike_size;

        // Allocate pinned host memory mapped to GPU
        let host_ptr = unsafe {
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            let result = sys::cuMemHostAlloc(
                &mut ptr,
                capacity_bytes,
                sys::CU_MEMHOSTALLOC_DEVICEMAP | sys::CU_MEMHOSTALLOC_PORTABLE,
            );
            if result != sys::CUresult::CUDA_SUCCESS {
                anyhow::bail!(
                    "cuMemHostAlloc failed for {} MB: {:?}",
                    capacity_bytes / (1024 * 1024),
                    result
                );
            }
            // Zero the buffer
            std::ptr::write_bytes(ptr as *mut u8, 0, capacity_bytes);
            ptr as *mut u8
        };

        // Get the device pointer for the mapped memory
        let device_ptr = unsafe {
            let mut dptr: sys::CUdeviceptr = 0;
            let result =
                sys::cuMemHostGetDevicePointer_v2(&mut dptr, host_ptr as *mut std::ffi::c_void, 0);
            if result != sys::CUresult::CUDA_SUCCESS {
                // Free the host memory before returning error
                sys::cuMemFreeHost(host_ptr as *mut std::ffi::c_void);
                anyhow::bail!("cuMemHostGetDevicePointer failed: {:?}", result);
            }
            dptr
        };

        // Allocate a separate mapped u32 for the write head counter
        let write_head_ptr = unsafe {
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            let result = sys::cuMemHostAlloc(
                &mut ptr,
                std::mem::size_of::<u32>(),
                sys::CU_MEMHOSTALLOC_DEVICEMAP | sys::CU_MEMHOSTALLOC_PORTABLE,
            );
            if result != sys::CUresult::CUDA_SUCCESS {
                sys::cuMemFreeHost(host_ptr as *mut std::ffi::c_void);
                anyhow::bail!("cuMemHostAlloc for write_head failed: {:?}", result);
            }
            *(ptr as *mut u32) = 0;
            ptr as *mut u32
        };

        let write_head_device = unsafe {
            let mut dptr: sys::CUdeviceptr = 0;
            sys::cuMemHostGetDevicePointer_v2(
                &mut dptr,
                write_head_ptr as *mut std::ffi::c_void,
                0,
            );
            dptr
        };

        log::info!(
            "SpikeExhaustBuffer: {} MB pinned host RAM ({} spike capacity), mapped to GPU",
            capacity_bytes / (1024 * 1024),
            capacity_spikes
        );

        Ok(Self {
            host_ptr,
            device_ptr,
            capacity_bytes,
            capacity_spikes,
            write_head_ptr,
            write_head_device,
            read_head: AtomicU64::new(0),
        })
    }

    /// Get the device pointer for the spike data buffer.
    /// Pass this to the compact kernel as the output destination.
    pub fn device_data_ptr(&self) -> sys::CUdeviceptr {
        self.device_ptr
    }

    /// Get the device pointer for the write head counter.
    /// The compact kernel atomicAdd's this to reserve write positions.
    pub fn device_write_head_ptr(&self) -> sys::CUdeviceptr {
        self.write_head_device
    }

    /// Read the current write head from mapped memory (CPU-side, no GPU sync needed).
    pub fn write_head(&self) -> u64 {
        // The GPU writes to this via atomicAdd through the PCIe mapping.
        // Reading it from the CPU side gives us the latest value that
        // has completed the PCIe write (no explicit sync needed —
        // PCIe writes are strongly ordered).
        unsafe { (*self.write_head_ptr) as u64 }
    }

    /// Read the CPU-side read head.
    pub fn read_head(&self) -> u64 {
        self.read_head.load(Ordering::Relaxed)
    }

    /// Number of unread spikes available.
    pub fn available(&self) -> u64 {
        self.write_head().saturating_sub(self.read_head())
    }

    /// Read a batch of spikes from the buffer (CPU-side, zero-copy from system RAM).
    /// Returns the spike data and advances the read head.
    ///
    /// The returned slice points directly into pinned host memory.
    /// It's valid until the GPU wraps around and overwrites it
    /// (circular buffer). For safety, copy the data before the
    /// GPU catches up.
    pub fn harvest(&self, max_spikes: usize) -> Vec<[u8; 48]> {
        let available = self.available() as usize;
        let n = available.min(max_spikes);
        if n == 0 {
            return Vec::new();
        }

        let read_pos = self.read_head() as usize;
        let mut spikes = Vec::with_capacity(n);

        for i in 0..n {
            let idx = (read_pos + i) % self.capacity_spikes;
            let offset = idx * 48;
            let mut spike = [0u8; 48];
            unsafe {
                std::ptr::copy_nonoverlapping(self.host_ptr.add(offset), spike.as_mut_ptr(), 48);
            }
            spikes.push(spike);
        }

        self.read_head.fetch_add(n as u64, Ordering::Relaxed);
        spikes
    }

    /// Get capacity in spikes.
    pub fn capacity(&self) -> usize {
        self.capacity_spikes
    }
}

#[cfg(feature = "cuda")]
impl Drop for SpikeExhaustBuffer {
    fn drop(&mut self) {
        unsafe {
            sys::cuMemFreeHost(self.write_head_ptr as *mut std::ffi::c_void);
            sys::cuMemFreeHost(self.host_ptr as *mut std::ffi::c_void);
        }
    }
}

// Mark as Send — the host_ptr is pinned memory that's valid across threads.
// The GPU writes via DMA; the CPU reads via the harvest() method.
// AtomicU64 handles the read_head synchronization.
#[cfg(feature = "cuda")]
unsafe impl Send for SpikeExhaustBuffer {}
#[cfg(feature = "cuda")]
unsafe impl Sync for SpikeExhaustBuffer {}
