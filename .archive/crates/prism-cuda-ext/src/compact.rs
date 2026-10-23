//! Device-side spike compaction kernel wrappers.
//!
//! These kernels run entirely on the GPU, reading d_spike_events and
//! d_spike_count directly from the physics engine's buffers. No host
//! round-trip needed.

use anyhow::{Context, Result};

#[cfg(feature = "cuda")]
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
#[cfg(feature = "cuda")]
use std::sync::Arc;

/// Device-side spike compaction + ring buffer push.
///
/// Reads spikes directly from the engine's GPU buffers, compacts
/// GpuSpikeEvent (92B) → RingSpikeEvent (48B), pushes to ring buffer.
/// Grid size is determined by d_spike_count on the device.
#[cfg(feature = "cuda")]
pub struct DeviceCompactor {
    compact_fn: CudaFunction,
    update_prev_fn: CudaFunction,
    compute_grid_fn: CudaFunction,
    /// Device-side prev_count tracker (updated by GPU, never read by CPU)
    pub d_prev_count_a: CudaSlice<i32>,
    pub d_prev_count_b: CudaSlice<i32>,
    /// Device-side grid dimension for compact kernel (written by compute_grid_size)
    pub d_grid_dim_a: CudaSlice<u32>,
    pub d_grid_dim_b: CudaSlice<u32>,
    // Dummy buffers for exhaust when disabled
    d_exhaust_dummy_buf: CudaSlice<u8>,
    d_exhaust_dummy_head: CudaSlice<u32>,
}

#[cfg(feature = "cuda")]
impl DeviceCompactor {
    pub fn new(stream: &Arc<CudaStream>, module: &Arc<CudaModule>) -> Result<Self> {
        let compact_fn = module
            .load_function("device_compact_and_push")
            .context("device_compact_and_push not found")?;
        let update_prev_fn = module
            .load_function("update_prev_spike_count")
            .context("update_prev_spike_count not found")?;
        let compute_grid_fn = module
            .load_function("compute_compact_grid_size")
            .context("compute_compact_grid_size not found")?;

        let d_prev_count_a = stream.alloc_zeros::<i32>(1)?;
        let d_prev_count_b = stream.alloc_zeros::<i32>(1)?;
        let mut d_grid_dim_a = stream.alloc_zeros::<u32>(1)?;
        let mut d_grid_dim_b = stream.alloc_zeros::<u32>(1)?;
        // Dummy buffers for exhaust params when exhaust is disabled
        let d_exhaust_dummy_buf = stream.alloc_zeros::<u8>(48)?; // 1 spike
        let d_exhaust_dummy_head = stream.alloc_zeros::<u32>(1)?;

        // Initialize grid dims to 1 (minimum, kernel will early-exit)
        stream.memcpy_htod(&[1u32], &mut d_grid_dim_a)?;
        stream.memcpy_htod(&[1u32], &mut d_grid_dim_b)?;

        Ok(Self {
            compact_fn,
            update_prev_fn,
            compute_grid_fn,
            d_prev_count_a,
            d_prev_count_b,
            d_grid_dim_a,
            d_grid_dim_b,
            d_exhaust_dummy_buf,
            d_exhaust_dummy_head,
        })
    }

    /// Compact stream A's new spikes and push to ring buffer A.
    ///
    /// This is a 3-kernel sequence that runs entirely on the GPU:
    ///   1. compute_compact_grid_size: reads d_spike_count, writes grid_dim
    ///   2. device_compact_and_push: reads spikes, compacts, pushes to ring
    ///   3. update_prev_spike_count: updates prev_count for next step
    ///
    /// All three kernels are launched on the exchange stream.
    /// The physics engine's d_spike_count is read-only (never modified).
    pub fn compact_and_push_a(
        &mut self,
        stream: &Arc<CudaStream>,
        // Engine A's GPU buffers (read-only)
        spike_buf_a: &CudaSlice<u8>,    // d_spike_events
        spike_count_a: &CudaSlice<i32>, // d_spike_count
        // Ring buffer A (write)
        ring_buffer: &mut CudaSlice<u8>,
        ring_head: &mut CudaSlice<u32>,
        ring_overflow: &mut CudaSlice<u32>,
        ring_capacity: u32,
    ) -> Result<()> {
        let single_cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        // Step 1: compute grid size from spike counts (device-side)
        unsafe {
            stream
                .launch_builder(&self.compute_grid_fn)
                .arg(spike_count_a)
                .arg(&self.d_prev_count_a)
                .arg(&mut self.d_grid_dim_a)
                .arg(&ring_capacity)
                .launch(single_cfg)?;
        }

        // Step 2: compact and push (grid size read from device memory)
        // For now, we read d_grid_dim_a back to host to set the launch config.
        // The CUDA Graph with device-updated params will eliminate this read.
        let mut grid_dim_host = [1u32];
        stream.synchronize()?; // ensure compute_grid_size finished
        stream.memcpy_dtoh(&self.d_grid_dim_a, &mut grid_dim_host)?;

        let compact_cfg = LaunchConfig {
            grid_dim: (grid_dim_host[0].max(1), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(&self.compact_fn)
                .arg(spike_buf_a)
                .arg(&self.d_prev_count_a)
                .arg(spike_count_a)
                .arg(ring_buffer)
                .arg(ring_head)
                .arg(ring_overflow)
                .arg(&ring_capacity)
                .launch(compact_cfg)?;
        }

        // Step 3: update prev_count
        unsafe {
            stream
                .launch_builder(&self.update_prev_fn)
                .arg(&mut self.d_prev_count_a)
                .arg(spike_count_a)
                .launch(single_cfg)?;
        }

        Ok(())
    }

    /// Capturable compact for stream A — uses internal d_prev_count_a.
    /// Pure GPU kernel launches, no host readback — fully capturable.
    pub fn compact_a_capturable(
        &mut self,
        stream: &Arc<CudaStream>,
        spike_buf: &CudaSlice<u8>,
        spike_count: &CudaSlice<i32>,
        ring_buffer: &mut CudaSlice<u8>,
        ring_head: &mut CudaSlice<u32>,
        ring_overflow: &mut CudaSlice<u32>,
        ring_capacity: u32,
    ) -> Result<()> {
        let single_cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        let max_blocks = (ring_capacity + 255) / 256;
        let compact_cfg = LaunchConfig {
            grid_dim: (max_blocks.max(1), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let exhaust_en = 0i32;
        let exhaust_cap = 0u32;

        unsafe {
            stream
                .launch_builder(&self.compact_fn)
                .arg(spike_buf)
                .arg(&self.d_prev_count_a)
                .arg(spike_count)
                .arg(ring_buffer)
                .arg(ring_head)
                .arg(ring_overflow)
                .arg(&ring_capacity)
                .arg(&self.d_exhaust_dummy_buf) // exhaust disabled → dummy
                .arg(&self.d_exhaust_dummy_head)
                .arg(&exhaust_cap)
                .arg(&exhaust_en)
                .launch(compact_cfg)?;

            stream
                .launch_builder(&self.update_prev_fn)
                .arg(&mut self.d_prev_count_a)
                .arg(spike_count)
                .launch(single_cfg)?;
        }
        Ok(())
    }

    /// Capturable compact for stream B — uses internal d_prev_count_b.
    pub fn compact_b_capturable(
        &mut self,
        stream: &Arc<CudaStream>,
        spike_buf: &CudaSlice<u8>,
        spike_count: &CudaSlice<i32>,
        ring_buffer: &mut CudaSlice<u8>,
        ring_head: &mut CudaSlice<u32>,
        ring_overflow: &mut CudaSlice<u32>,
        ring_capacity: u32,
    ) -> Result<()> {
        let single_cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        let max_blocks = (ring_capacity + 255) / 256;
        let compact_cfg = LaunchConfig {
            grid_dim: (max_blocks.max(1), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let exhaust_en = 0i32;
        let exhaust_cap = 0u32;

        unsafe {
            stream
                .launch_builder(&self.compact_fn)
                .arg(spike_buf)
                .arg(&self.d_prev_count_b)
                .arg(spike_count)
                .arg(ring_buffer)
                .arg(ring_head)
                .arg(ring_overflow)
                .arg(&ring_capacity)
                .arg(&self.d_exhaust_dummy_buf)
                .arg(&self.d_exhaust_dummy_head)
                .arg(&exhaust_cap)
                .arg(&exhaust_en)
                .launch(compact_cfg)?;

            stream
                .launch_builder(&self.update_prev_fn)
                .arg(&mut self.d_prev_count_b)
                .arg(spike_count)
                .launch(single_cfg)?;
        }
        Ok(())
    }

    /// Same as compact_and_push_a but for stream B.
    pub fn compact_and_push_b(
        &mut self,
        stream: &Arc<CudaStream>,
        spike_buf_b: &CudaSlice<u8>,
        spike_count_b: &CudaSlice<i32>,
        ring_buffer: &mut CudaSlice<u8>,
        ring_head: &mut CudaSlice<u32>,
        ring_overflow: &mut CudaSlice<u32>,
        ring_capacity: u32,
    ) -> Result<()> {
        let single_cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(&self.compute_grid_fn)
                .arg(spike_count_b)
                .arg(&self.d_prev_count_b)
                .arg(&mut self.d_grid_dim_b)
                .arg(&ring_capacity)
                .launch(single_cfg)?;
        }

        let mut grid_dim_host = [1u32];
        stream.synchronize()?;
        stream.memcpy_dtoh(&self.d_grid_dim_b, &mut grid_dim_host)?;

        let compact_cfg = LaunchConfig {
            grid_dim: (grid_dim_host[0].max(1), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(&self.compact_fn)
                .arg(spike_buf_b)
                .arg(&self.d_prev_count_b)
                .arg(spike_count_b)
                .arg(ring_buffer)
                .arg(ring_head)
                .arg(ring_overflow)
                .arg(&ring_capacity)
                .launch(compact_cfg)?;
        }

        unsafe {
            stream
                .launch_builder(&self.update_prev_fn)
                .arg(&mut self.d_prev_count_b)
                .arg(spike_count_b)
                .launch(single_cfg)?;
        }

        Ok(())
    }
}
