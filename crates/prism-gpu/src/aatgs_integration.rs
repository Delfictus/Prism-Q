//! AATGS Integration for GPU Pipeline
//!
//! Integrates the AATGS (Adaptive Asynchronous Task Graph Scheduler) into
//! GPU execution workflows, enabling async scheduling for:
//! - WHCR (Wavelet-Hierarchical Conflict Repair)
//! - Active Inference
//! - Dendritic Reservoir
//! - Thermodynamic Equilibration
//! - All other GPU kernels
//!
//! ## Architecture
//!
//! The integration provides two execution modes:
//!
//! ### Async Mode (AATGS Enabled)
//! ```text
//! CPU Thread                   GPU (CUDA)
//! ──────────                   ──────────
//! queue_config() ──────────▶  [Config Buffer]
//!      │                            │
//!      │                            ▼
//!      │                      [Kernel Execution]
//!      │                            │
//!      │                            ▼
//! poll_telemetry() ◀──────── [Telemetry Buffer]
//! ```
//!
//! ### Sync Mode (AATGS Disabled)
//! ```text
//! CPU Thread                   GPU (CUDA)
//! ──────────                   ──────────
//! execute() ───────────────▶  [Kernel Launch]
//!      │                            │
//!      │                      [Synchronize]
//!      │                            │
//!      └◀────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,no_run
//! use prism_gpu::aatgs_integration::GpuExecutionContext;
//! use prism_core::RuntimeConfig;
//! use cudarc::driver::CudaContext;
//! use std::sync::Arc;
//!
//! // Create context with async enabled
//! let device = CudaContext::new(0).unwrap());
//! let mut ctx = GpuExecutionContext::new(Arc::new(device), true).unwrap();
//!
//! // Execute with async scheduling
//! let config = RuntimeConfig::production();
//! let telemetry = ctx.execute(config).unwrap());
//! ```

use anyhow::{Context, Result};
use cudarc::driver::CudaContext;
use std::sync::Arc;

use crate::aatgs::AsyncPipeline;
use prism_core::{KernelTelemetry, RuntimeConfig};

/// GPU execution context with optional async scheduling
///
/// Provides a unified interface for GPU kernel execution that can operate
/// in either async (AATGS) or sync mode. This allows gradual migration of
/// existing code without breaking changes.
///
/// ## Features
///
/// - **Transparent fallback**: If async is disabled, falls back to sync execution
/// - **Zero-copy telemetry**: Direct GPU buffer access in async mode
/// - **Buffer statistics**: Track buffer utilization for performance tuning
/// - **Error recovery**: Graceful handling of buffer overflow and GPU errors
pub struct GpuExecutionContext {
    /// Shared CUDA device
    context: Arc<CudaContext>,

    /// Async pipeline (None = sync mode)
    async_pipeline: Option<AsyncPipeline>,

    /// Track whether async mode is enabled
    use_async: bool,

    /// Execution statistics
    stats: ExecutionStats,
}

/// Execution statistics for performance monitoring
#[derive(Debug, Clone, Default)]
pub struct ExecutionStats {
    /// Total iterations executed
    pub total_iterations: usize,

    /// Iterations executed in async mode
    pub async_iterations: usize,

    /// Iterations executed in sync mode
    pub sync_iterations: usize,

    /// Peak config buffer utilization [0.0, 1.0]
    pub peak_config_util: f32,

    /// Peak telemetry buffer utilization [0.0, 1.0]
    pub peak_telemetry_util: f32,

    /// Number of buffer overflow events
    pub buffer_overflows: usize,

    /// Number of times GPU was idle
    pub gpu_idle_count: usize,
}

impl GpuExecutionContext {
    /// Create new GPU execution context
    ///
    /// # Arguments
    /// * `device` - Shared CUDA device
    /// * `enable_async` - Enable AATGS async scheduling
    ///
    /// # Returns
    /// * `Ok(Self)` - Initialized context
    /// * `Err(_)` - AATGS initialization failure (only if async enabled)
    ///
    /// # Example
    /// ```rust,no_run
    /// use prism_gpu::aatgs_integration::GpuExecutionContext;
    /// use cudarc::driver::CudaContext;
    /// use std::sync::Arc;
    ///
    /// let device = CudaContext::new(0).unwrap());
    /// let ctx = GpuExecutionContext::new(Arc::new(device), true).unwrap());
    /// ```
    pub fn new(context: Arc<CudaContext>, enable_async: bool) -> Result<Self> {
        log::info!(
            "Creating GPU execution context (async: {})",
            enable_async
        );

        let async_pipeline = if enable_async {
            Some(AsyncPipeline::new(context.clone()).context("Failed to initialize AATGS pipeline")?)
        } else {
            None
        };

        Ok(Self {
            context,
            async_pipeline,
            use_async: enable_async,
            stats: ExecutionStats::default(),
        })
    }

    /// Execute with optional async scheduling
    ///
    /// Automatically selects async or sync path based on context configuration.
    /// In async mode, configs are queued and telemetry is polled asynchronously.
    /// In sync mode, execution blocks until kernel completes.
    ///
    /// # Arguments
    /// * `config` - Runtime configuration for GPU kernel
    ///
    /// # Returns
    /// * `Ok(Some(telemetry))` - Telemetry from completed iteration
    /// * `Ok(None)` - No telemetry available yet (async mode, pipeline filling)
    /// * `Err(_)` - GPU execution or buffer overflow error
    ///
    /// # Example
    /// ```rust,no_run
    /// # use prism_gpu::aatgs_integration::GpuExecutionContext;
    /// # use prism_core::RuntimeConfig;
    /// # use cudarc::driver::CudaContext;
    /// # use std::sync::Arc;
    /// # let device = CudaContext::new(0).unwrap());
    /// # let mut ctx = GpuExecutionContext::new(Arc::new(device), true).unwrap();
    /// let config = RuntimeConfig::production();
    /// if let Some(telemetry) = ctx.execute(config).unwrap() {
    ///     println!("Conflicts: {}", telemetry.conflicts);
    /// }
    /// ```
    pub fn execute(&mut self, config: RuntimeConfig) -> Result<Option<KernelTelemetry>> {
        self.stats.total_iterations += 1;

        if let Some(ref mut pipeline) = self.async_pipeline {
            // Async path: queue config and poll telemetry
            self.stats.async_iterations += 1;

            let result = pipeline.step(config).context("AATGS pipeline step failed");

            // Update statistics
            if let Ok(ref telemetry) = result {
                let (config_util, telemetry_util) = pipeline.peak_utilization();
                self.stats.peak_config_util = self.stats.peak_config_util.max(config_util);
                self.stats.peak_telemetry_util = self.stats.peak_telemetry_util.max(telemetry_util);

                if telemetry.is_none() {
                    // Check if GPU is idle
                    if pipeline.is_idle().unwrap_or(false) {
                        self.stats.gpu_idle_count += 1;
                    }
                }
            } else {
                // Assume buffer overflow on error
                self.stats.buffer_overflows += 1;
            }

            result
        } else {
            // Sync path: fallback to synchronous execution
            self.stats.sync_iterations += 1;
            self.execute_sync(config).map(Some)
        }
    }

    /// Execute batch of configs (async mode only, sync fallback)
    ///
    /// Efficiently processes multiple configs by maximizing AATGS buffer utilization.
    /// Falls back to sequential sync execution if async is disabled.
    ///
    /// # Arguments
    /// * `configs` - Slice of configurations to execute
    ///
    /// # Returns
    /// * `Ok(Vec<KernelTelemetry>)` - Telemetry from all iterations
    /// * `Err(_)` - GPU execution error
    pub fn execute_batch(&mut self, configs: &[RuntimeConfig]) -> Result<Vec<KernelTelemetry>> {
        self.stats.total_iterations += configs.len();

        if let Some(ref mut pipeline) = self.async_pipeline {
            // Async path: batch execution with AATGS
            self.stats.async_iterations += configs.len();

            let results = pipeline
                .batch_step(configs)
                .context("AATGS batch step failed")?;

            // Update statistics
            let (config_util, telemetry_util) = pipeline.peak_utilization();
            self.stats.peak_config_util = self.stats.peak_config_util.max(config_util);
            self.stats.peak_telemetry_util = self.stats.peak_telemetry_util.max(telemetry_util);

            Ok(results)
        } else {
            // Sync path: sequential execution
            self.stats.sync_iterations += configs.len();

            let mut results = Vec::with_capacity(configs.len());
            for config in configs {
                let telemetry = self.execute_sync(*config)?;
                results.push(telemetry);
            }
            Ok(results)
        }
    }

    /// Synchronous execution fallback (placeholder)
    ///
    /// This method should be implemented to execute GPU kernels synchronously
    /// when async mode is disabled. Currently returns dummy telemetry.
    ///
    /// # Implementation Note
    ///
    /// Callers should wire this to their actual kernel launch logic:
    /// ```rust,ignore
    /// fn execute_sync(&mut self, config: RuntimeConfig) -> Result<KernelTelemetry> {
    ///     // Upload config to GPU
    ///     let d_config = self.stream.clone_htod(&vec![config])?;
    ///
    ///     // Launch kernel
    ///     let cfg = LaunchConfig { ... };
    ///     unsafe { &self.stream.launch_builder(&my_kernel)
    ///             .arg(&d_config)
    ///             .arg(...)
    ///             .launch(cfg)? };
    ///
    ///     // Download telemetry
    ///     let telemetry = self.stream.clone_dtoh(&d_telemetry)?;
    ///     Ok(telemetry[0])
    /// }
    /// ```
    fn execute_sync(&mut self, config: RuntimeConfig) -> Result<KernelTelemetry> {
        log::debug!(
            "Sync execution fallback (iteration: {}, phase: {})",
            config.iteration,
            config.phase_id
        );

        // Synchronize device to ensure previous operations complete
        self.context
            .synchronize()
            .context("Device synchronization failed")?;

        // TODO: Wire to actual kernel launch
        // This is a placeholder that returns default telemetry
        // Real implementation should:
        // 1. Upload config to GPU
        // 2. Launch appropriate kernel
        // 3. Download telemetry
        // 4. Return results

        log::warn!("execute_sync() is using placeholder implementation - wire to actual kernels");

        Ok(KernelTelemetry::default())
    }

    /// Check if GPU is idle (async mode only)
    ///
    /// Returns whether the GPU is waiting for new work.
    /// Always returns false in sync mode.
    ///
    /// # Returns
    /// * `Ok(true)` - GPU is idle
    /// * `Ok(false)` - GPU is busy or sync mode
    /// * `Err(_)` - GPU query failure
    pub fn is_gpu_idle(&mut self) -> Result<bool> {
        if let Some(ref mut pipeline) = self.async_pipeline {
            pipeline
                .is_idle()
                .context("Failed to query GPU idle status")
        } else {
            Ok(false)
        }
    }

    /// Get execution statistics
    ///
    /// Returns performance metrics for monitoring and tuning.
    pub fn stats(&self) -> &ExecutionStats {
        &self.stats
    }

    /// Reset execution statistics
    pub fn reset_stats(&mut self) {
        self.stats = ExecutionStats::default();
    }

    /// Get reference to underlying CUDA device
    pub fn device(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Check if async mode is enabled
    pub fn is_async(&self) -> bool {
        self.use_async
    }

    /// Get reference to async pipeline (if enabled)
    pub fn async_pipeline(&self) -> Option<&AsyncPipeline> {
        self.async_pipeline.as_ref()
    }

    /// Get mutable reference to async pipeline (if enabled)
    pub fn async_pipeline_mut(&mut self) -> Option<&mut AsyncPipeline> {
        self.async_pipeline.as_mut()
    }

    /// Flush pending configs to GPU (async mode only)
    ///
    /// Forces immediate upload of queued configs. No-op in sync mode.
    pub fn flush(&mut self) -> Result<()> {
        if let Some(ref mut pipeline) = self.async_pipeline {
            pipeline
                .scheduler_mut()
                .flush_configs()
                .context("Failed to flush configs")
        } else {
            Ok(())
        }
    }

    /// Shutdown and cleanup resources
    ///
    /// Signals GPU to shutdown (async mode) and synchronizes device.
    pub fn shutdown(mut self) -> Result<()> {
        log::info!("Shutting down GPU execution context");

        if let Some(pipeline) = self.async_pipeline.take() {
            pipeline.shutdown().context("AATGS shutdown failed")?;
        }

        self.context
            .synchronize()
            .context("Final device synchronization failed")?;

        log::info!(
            "GPU execution context shutdown complete (total iterations: {})",
            self.stats.total_iterations
        );

        Ok(())
    }
}

impl Drop for GpuExecutionContext {
    fn drop(&mut self) {
        // Best-effort cleanup on drop
        if let Some(pipeline) = self.async_pipeline.take() {
            if let Err(e) = pipeline.shutdown() {
                log::error!("Error during AATGS shutdown in Drop: {}", e);
            }
        }

        // Log final statistics
        log::info!(
            "GpuExecutionContext dropped - Stats: total={}, async={}, sync={}, overflows={}",
            self.stats.total_iterations,
            self.stats.async_iterations,
            self.stats.sync_iterations,
            self.stats.buffer_overflows
        );
    }
}

/// Builder for GpuExecutionContext with additional configuration options
///
/// Provides a fluent API for constructing execution contexts with custom settings.
///
/// # Example
/// ```rust,no_run
/// use prism_gpu::aatgs_integration::GpuExecutionContextBuilder;
/// use cudarc::driver::CudaContext;
/// use std::sync::Arc;
///
/// let device = CudaContext::new(0).unwrap());
/// let ctx = GpuExecutionContextBuilder::new(Arc::new(device))
///     .enable_async(true)
///     .build()
///     .unwrap());
/// ```
pub struct GpuExecutionContextBuilder {
    context: Arc<CudaContext>,
    enable_async: bool,
}

impl GpuExecutionContextBuilder {
    /// Create new builder with device
    pub fn new(context: Arc<CudaContext>) -> Self {
        Self {
            context,
            enable_async: false, // Default to sync mode
        }
    }

    /// Enable or disable async mode
    pub fn enable_async(mut self, enable: bool) -> Self {
        self.enable_async = enable;
        self
    }

    /// Build the execution context
    pub fn build(self) -> Result<GpuExecutionContext> {
        GpuExecutionContext::new(self.context, self.enable_async)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_sync_mode() {
        // Test builder creates context with sync mode by default
        // (requires GPU, so this is just a compilation check)
        let _ = GpuExecutionContextBuilder::new;
    }

    #[test]
    fn test_execution_stats_default() {
        let stats = ExecutionStats::default();
        assert_eq!(stats.total_iterations, 0);
        assert_eq!(stats.async_iterations, 0);
        assert_eq!(stats.sync_iterations, 0);
        assert_eq!(stats.buffer_overflows, 0);
    }

    // GPU tests require CUDA device
    #[test]
    #[ignore]
    fn test_context_sync_mode() {
        let device = CudaContext::new(0).expect("Failed to create CUDA device");
        let ctx = GpuExecutionContext::new(Arc::new(device), false);
        assert!(ctx.is_ok());
        assert!(!ctx.unwrap().is_async());
    }

    #[test]
    #[ignore]
    fn test_context_async_mode() {
        let device = CudaContext::new(0).expect("Failed to create CUDA device");
        let ctx = GpuExecutionContext::new(Arc::new(device), true);
        assert!(ctx.is_ok());
        assert!(ctx.unwrap().is_async());
    }

    #[test]
    #[ignore]
    fn test_execute_sync() {
        let device = CudaContext::new(0).expect("Failed to create CUDA device");
        let mut ctx = GpuExecutionContext::new(Arc::new(device), false).unwrap();

        let config = RuntimeConfig::production();
        let result = ctx.execute(config);
        assert!(result.is_ok());

        let stats = ctx.stats();
        assert_eq!(stats.total_iterations, 1);
        assert_eq!(stats.sync_iterations, 1);
        assert_eq!(stats.async_iterations, 0);
    }

    #[test]
    #[ignore]
    fn test_execute_async() {
        let device = CudaContext::new(0).expect("Failed to create CUDA device");
        let mut ctx = GpuExecutionContext::new(Arc::new(device), true).unwrap();

        let config = RuntimeConfig::production();
        let result = ctx.execute(config);
        assert!(result.is_ok());

        let stats = ctx.stats();
        assert_eq!(stats.total_iterations, 1);
        assert_eq!(stats.async_iterations, 1);
    }
}
