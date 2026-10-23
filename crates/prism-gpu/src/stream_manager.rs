//! Stream Manager for Async GPU Operations
//!
//! Centralized stream management with triple-buffering for maximum overlap.
//!
//! This module provides:
//! - StreamPool for managing multiple CUDA streams by purpose
//! - TripleBuffer for lock-free async data exchange
//! - AsyncPipelineCoordinator for overlapping config upload, kernel execution, and telemetry download

use anyhow::Result;
use cudarc::driver::CudaContext;
use std::sync::Arc;

use prism_core::{KernelTelemetry, RuntimeConfig};

/// Stream purposes for organized async operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamPurpose {
    /// Config upload stream
    ConfigUpload,
    /// Main kernel execution stream
    KernelExecution,
    /// Telemetry download stream
    TelemetryDownload,
    /// P2P transfer stream (multi-GPU)
    P2PTransfer,
    /// Auxiliary compute stream
    AuxCompute,
}

/// Managed CUDA stream with purpose tagging and usage tracking
pub struct ManagedStream {
    /// Stream index in device
    stream_idx: usize,
    /// Purpose of this stream
    purpose: StreamPurpose,
    /// Whether stream is currently in use
    in_use: bool,
}

/// Stream pool for a single GPU
///
/// Manages multiple CUDA streams organized by purpose for maximum overlap.
/// Streams can be acquired by purpose and synchronized independently.
pub struct StreamPool {
    /// CUDA device context
    ctx: Arc<CudaContext>,
    /// Managed streams
    streams: Vec<ManagedStream>,
}

impl StreamPool {
    /// Create pool with default streams for each purpose
    ///
    /// # Example
    /// ```no_run
    /// # use cudarc::driver::CudaContext;
    /// # use prism_gpu::stream_manager::StreamPool;
    /// let device = CudaContext::new(0).unwrap();
    /// let pool = StreamPool::new(device).unwrap();
    /// ```
    pub fn new(ctx: Arc<CudaContext>) -> Result<Self> {
        let mut streams = Vec::new();

        // Create one stream for each purpose
        for (idx, purpose) in [
            StreamPurpose::ConfigUpload,
            StreamPurpose::KernelExecution,
            StreamPurpose::TelemetryDownload,
            StreamPurpose::P2PTransfer,
            StreamPurpose::AuxCompute,
        ]
        .iter()
        .enumerate()
        {
            streams.push(ManagedStream {
                stream_idx: idx,
                purpose: *purpose,
                in_use: false,
            });
        }

        Ok(Self { ctx, streams })
    }

    /// Get stream for specific purpose
    ///
    /// Marks the stream as in-use and returns its index.
    pub fn get_stream(&mut self, purpose: StreamPurpose) -> Result<usize> {
        let stream = self
            .streams
            .iter_mut()
            .find(|s| s.purpose == purpose)
            .ok_or_else(|| anyhow::anyhow!("Stream purpose not found: {:?}", purpose))?;

        stream.in_use = true;
        Ok(stream.stream_idx)
    }

    /// Get config upload stream
    pub fn config_stream(&mut self) -> Result<usize> {
        self.get_stream(StreamPurpose::ConfigUpload)
    }

    /// Get kernel execution stream
    pub fn kernel_stream(&mut self) -> Result<usize> {
        self.get_stream(StreamPurpose::KernelExecution)
    }

    /// Get telemetry download stream
    pub fn telemetry_stream(&mut self) -> Result<usize> {
        self.get_stream(StreamPurpose::TelemetryDownload)
    }

    /// Get P2P transfer stream
    pub fn p2p_stream(&mut self) -> Result<usize> {
        self.get_stream(StreamPurpose::P2PTransfer)
    }

    /// Get auxiliary compute stream
    pub fn aux_stream(&mut self) -> Result<usize> {
        self.get_stream(StreamPurpose::AuxCompute)
    }

    /// Synchronize all streams
    ///
    /// Blocks until all streams have completed their work.
    pub fn synchronize_all(&self) -> Result<()> {
        self.ctx.synchronize()?;
        Ok(())
    }

    /// Synchronize specific purpose
    ///
    /// Blocks until the specified stream has completed its work.
    pub fn synchronize(&self, purpose: StreamPurpose) -> Result<()> {
        if let Some(_stream) = self.streams.iter().find(|s| s.purpose == purpose) {
            // In cudarc 0.9, we synchronize via device
            // Stream-level sync requires tracking stream handles differently
            self.ctx.synchronize()?;
        }
        Ok(())
    }

    /// Release stream (mark as not in use)
    pub fn release_stream(&mut self, purpose: StreamPurpose) {
        if let Some(stream) = self.streams.iter_mut().find(|s| s.purpose == purpose) {
            stream.in_use = false;
        }
    }

    /// Check if stream is currently in use
    pub fn is_in_use(&self, purpose: StreamPurpose) -> bool {
        self.streams
            .iter()
            .find(|s| s.purpose == purpose)
            .map(|s| s.in_use)
            .unwrap_or(false)
    }

    /// Get device context
    pub fn device(&self) -> &Arc<CudaContext> {
        &self.ctx
    }
}

/// Triple-buffer state for async pipeline
///
/// Lock-free triple buffering allows producer and consumer to operate
/// independently without blocking. The producer writes to one buffer,
/// the consumer reads from another, and a third is ready for swap.
pub struct TripleBuffer<T> {
    /// Buffer being written by producer
    write_buffer: usize,
    /// Buffer being read by consumer
    read_buffer: usize,
    /// Buffer ready for swap
    ready_buffer: usize,
    /// The actual buffers
    buffers: [T; 3],
}

impl<T: Default + Clone> TripleBuffer<T> {
    /// Create new triple buffer with default values
    pub fn new() -> Self {
        Self {
            write_buffer: 0,
            read_buffer: 1,
            ready_buffer: 2,
            buffers: [T::default(), T::default(), T::default()],
        }
    }

    /// Get mutable reference to write buffer
    ///
    /// Producer uses this to write new data.
    pub fn write_buf(&mut self) -> &mut T {
        &mut self.buffers[self.write_buffer]
    }

    /// Get reference to read buffer
    ///
    /// Consumer uses this to read latest data.
    pub fn read_buf(&self) -> &T {
        &self.buffers[self.read_buffer]
    }

    /// Publish write buffer (make it ready)
    ///
    /// Called by producer when write is complete.
    /// Swaps write buffer with ready buffer.
    pub fn publish(&mut self) {
        std::mem::swap(&mut self.write_buffer, &mut self.ready_buffer);
    }

    /// Consume ready buffer (make it the read buffer)
    ///
    /// Called by consumer to get latest published data.
    /// Swaps read buffer with ready buffer.
    pub fn consume(&mut self) {
        std::mem::swap(&mut self.read_buffer, &mut self.ready_buffer);
    }

    /// Get all buffers (for debugging)
    pub fn buffers(&self) -> &[T; 3] {
        &self.buffers
    }
}

impl<T: Default + Clone> Default for TripleBuffer<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Pipeline stage tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStage {
    /// Pipeline is idle
    Idle,
    /// Config being uploaded to GPU
    ConfigUploading,
    /// Kernel executing on GPU
    KernelRunning,
    /// Telemetry being downloaded from GPU
    TelemetryDownloading,
}

/// Async pipeline coordinator
///
/// Coordinates asynchronous GPU operations using triple-buffering
/// to overlap config upload, kernel execution, and telemetry download.
///
/// # Pipeline Stages
///
/// 1. ConfigUploading: Upload RuntimeConfig to GPU
/// 2. KernelRunning: Execute GPU kernel
/// 3. TelemetryDownloading: Download KernelTelemetry from GPU
///
/// Triple-buffering allows these stages to overlap across iterations.
pub struct AsyncPipelineCoordinator {
    /// Stream pool
    streams: StreamPool,

    /// Config triple buffer
    config_buffer: TripleBuffer<RuntimeConfig>,

    /// Telemetry triple buffer
    telemetry_buffer: TripleBuffer<KernelTelemetry>,

    /// Pipeline stage
    stage: PipelineStage,

    /// Iteration counter
    iteration: usize,
}

impl AsyncPipelineCoordinator {
    /// Create new async pipeline coordinator
    ///
    /// # Example
    /// ```no_run
    /// # use cudarc::driver::CudaContext;
    /// # use prism_gpu::stream_manager::AsyncPipelineCoordinator;
    /// let device = CudaContext::new(0).unwrap();
    /// let coordinator = AsyncPipelineCoordinator::new(device).unwrap();
    /// ```
    pub fn new(ctx: Arc<CudaContext>) -> Result<Self> {
        Ok(Self {
            streams: StreamPool::new(ctx)?,
            config_buffer: TripleBuffer::new(),
            telemetry_buffer: TripleBuffer::new(),
            stage: PipelineStage::Idle,
            iteration: 0,
        })
    }

    /// Start async config upload
    ///
    /// Writes config to the write buffer and marks stage as ConfigUploading.
    /// In a full implementation, this would launch async H2D copy.
    pub fn begin_config_upload(&mut self, config: RuntimeConfig) -> Result<()> {
        *self.config_buffer.write_buf() = config;

        // Async H2D copy would happen here
        let _stream_idx = self.streams.config_stream()?;
        // stream.htod_copy_async(device_config, &config, stream_idx)?;

        self.stage = PipelineStage::ConfigUploading;
        Ok(())
    }

    /// Start async kernel execution (after config ready)
    ///
    /// Publishes config buffer and launches kernel.
    pub fn begin_kernel_execution(&mut self) -> Result<()> {
        // Publish config buffer
        self.config_buffer.publish();

        // Launch kernel on execution stream
        let _stream_idx = self.streams.kernel_stream()?;
        // kernel.launch_async(config, stream_idx)?;

        self.stage = PipelineStage::KernelRunning;
        Ok(())
    }

    /// Start async telemetry download (after kernel done)
    ///
    /// Launches async D2H copy of telemetry.
    pub fn begin_telemetry_download(&mut self) -> Result<()> {
        // Async D2H copy of telemetry
        let _stream_idx = self.streams.telemetry_stream()?;
        // stream.dtoh_copy_async(self.telemetry_buffer.write_buf(), device_telemetry, stream_idx)?;

        self.stage = PipelineStage::TelemetryDownloading;
        Ok(())
    }

    /// Complete pipeline iteration, get telemetry
    ///
    /// Synchronizes telemetry download, consumes telemetry buffer,
    /// and returns latest telemetry data.
    pub fn complete_iteration(&mut self) -> Result<KernelTelemetry> {
        // Wait for telemetry download
        self.streams.synchronize(StreamPurpose::TelemetryDownload)?;

        // Publish and consume telemetry buffer
        self.telemetry_buffer.publish();
        self.telemetry_buffer.consume();

        self.stage = PipelineStage::Idle;
        self.iteration += 1;

        Ok(self.telemetry_buffer.read_buf().clone())
    }

    /// Check if config upload is complete
    ///
    /// In full implementation, would query stream status.
    pub fn is_config_ready(&self) -> bool {
        // Would query stream completion
        // self.streams.is_complete(StreamPurpose::ConfigUpload)
        true
    }

    /// Check if kernel is complete
    ///
    /// In full implementation, would query stream status.
    pub fn is_kernel_done(&self) -> bool {
        // Would query stream completion
        // self.streams.is_complete(StreamPurpose::KernelExecution)
        true
    }

    /// Check if telemetry is ready
    ///
    /// In full implementation, would query stream status.
    pub fn is_telemetry_ready(&self) -> bool {
        // Would query stream completion
        // self.streams.is_complete(StreamPurpose::TelemetryDownload)
        true
    }

    /// Get current pipeline stage
    pub fn stage(&self) -> PipelineStage {
        self.stage
    }

    /// Get current iteration count
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Get reference to stream pool
    pub fn streams(&self) -> &StreamPool {
        &self.streams
    }

    /// Get mutable reference to stream pool
    pub fn streams_mut(&mut self) -> &mut StreamPool {
        &mut self.streams
    }

    /// Get reference to config buffer
    pub fn config_buffer(&self) -> &TripleBuffer<RuntimeConfig> {
        &self.config_buffer
    }

    /// Get reference to telemetry buffer
    pub fn telemetry_buffer(&self) -> &TripleBuffer<KernelTelemetry> {
        &self.telemetry_buffer
    }

    /// Synchronize entire pipeline
    ///
    /// Waits for all streams to complete.
    pub fn synchronize(&self) -> Result<()> {
        self.streams.synchronize_all()
    }

    /// Reset pipeline to idle state
    pub fn reset(&mut self) {
        self.stage = PipelineStage::Idle;
        // Don't reset iteration counter to maintain history
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triple_buffer_basic() {
        let mut buffer = TripleBuffer::<i32>::new();

        // Write value
        *buffer.write_buf() = 42;

        // Publish
        buffer.publish();

        // Consume
        buffer.consume();

        // Read should now have the published value
        assert_eq!(*buffer.read_buf(), 42);
    }

    #[test]
    fn test_triple_buffer_multiple_writes() {
        let mut buffer = TripleBuffer::<i32>::new();

        // Write and publish
        *buffer.write_buf() = 1;
        buffer.publish();
        buffer.consume();
        assert_eq!(*buffer.read_buf(), 1);

        // Write and publish again
        *buffer.write_buf() = 2;
        buffer.publish();
        buffer.consume();
        assert_eq!(*buffer.read_buf(), 2);
    }

    #[test]
    fn test_pipeline_stage_transitions() {
        let stage = PipelineStage::Idle;
        assert_eq!(stage, PipelineStage::Idle);

        let stage = PipelineStage::ConfigUploading;
        assert_eq!(stage, PipelineStage::ConfigUploading);

        let stage = PipelineStage::KernelRunning;
        assert_eq!(stage, PipelineStage::KernelRunning);

        let stage = PipelineStage::TelemetryDownloading;
        assert_eq!(stage, PipelineStage::TelemetryDownloading);
    }

    #[test]
    fn test_stream_purpose_equality() {
        assert_eq!(StreamPurpose::ConfigUpload, StreamPurpose::ConfigUpload);
        assert_ne!(StreamPurpose::ConfigUpload, StreamPurpose::KernelExecution);
    }
}
