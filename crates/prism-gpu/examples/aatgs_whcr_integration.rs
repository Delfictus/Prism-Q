//! Example: AATGS Integration with WHCR
//!
//! Demonstrates how to wire the AATGS scheduler into WHCR for async GPU execution.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example aatgs_whcr_integration --features cuda
//! ```
//!
//! ## Architecture
//!
//! This example shows two execution modes:
//!
//! 1. **Sync Mode** (Traditional): CPU blocks waiting for each GPU kernel
//! 2. **Async Mode** (AATGS): CPU queues work ahead while GPU processes
//!
//! Expected speedup: 1.5-3x for iterative algorithms like WHCR

// DISABLED (obsolete): this example references prism_gpu::aatgs_integration, a
// module intentionally commented out in crates/prism-gpu/src/lib.rs. The original
// source is preserved verbatim below inside a `#[cfg(any())]` module (parsed but
// never compiled) so no intent is lost — pending an operator decision to
// re-enable aatgs_integration. This trivial `main` keeps the example target
// buildable for `cargo test --no-run`.
fn main() {}

#[cfg(any())]
mod disabled_original {
    use anyhow::Result;
    use cudarc::driver::CudaDevice;
    use prism_core::RuntimeConfig;
    use prism_gpu::aatgs_integration::GpuExecutionContext;
    use std::sync::Arc;
    use std::time::Instant;

    fn main() -> Result<()> {
        env_logger::init();

        println!("=== AATGS + WHCR Integration Example ===\n");

        // Initialize CUDA device (cudarc 0.9 returns Arc<CudaDevice>)
        let device = CudaDevice::new(0)?;
        println!("✅ GPU detected: Device 0\n");

        // Test configurations
        let iterations = 100;
        let config_base = RuntimeConfig::production();

        // ========== Sync Mode Benchmark ==========
        println!("Running SYNC mode ({} iterations)...", iterations);
        let start = Instant::now();

        let mut ctx_sync = GpuExecutionContext::new(Arc::clone(&device), false)?;

        for i in 0..iterations {
            let mut config = config_base;
            config.iteration = i as i32;

            // Execute synchronously
            let _telemetry = ctx_sync.execute(config)?;
        }

        let sync_duration = start.elapsed();
        let sync_stats = ctx_sync.stats();

        println!("  Duration: {:?}", sync_duration);
        println!("  Total iterations: {}", sync_stats.total_iterations);
        println!("  Sync iterations: {}", sync_stats.sync_iterations);
        println!();

        // ========== Async Mode Benchmark ==========
        println!("Running ASYNC mode ({} iterations)...", iterations);
        let start = Instant::now();

        let mut ctx_async = GpuExecutionContext::new(Arc::clone(&device), true)?;

        for i in 0..iterations {
            let mut config = config_base;
            config.iteration = i as i32;

            // Execute asynchronously (non-blocking)
            let _telemetry = ctx_async.execute(config)?;
        }

        // Flush any remaining work
        ctx_async.flush()?;

        let async_duration = start.elapsed();
        let async_stats = ctx_async.stats();

        println!("  Duration: {:?}", async_duration);
        println!("  Total iterations: {}", async_stats.total_iterations);
        println!("  Async iterations: {}", async_stats.async_iterations);
        println!(
            "  Peak config utilization: {:.1}%",
            async_stats.peak_config_util * 100.0
        );
        println!(
            "  Peak telemetry utilization: {:.1}%",
            async_stats.peak_telemetry_util * 100.0
        );
        println!("  Buffer overflows: {}", async_stats.buffer_overflows);
        println!();

        // ========== Performance Comparison ==========
        let speedup = sync_duration.as_secs_f64() / async_duration.as_secs_f64();
        println!("=== Results ===");
        println!("Speedup: {:.2}x", speedup);
        println!(
            "Sync throughput: {:.1} iter/s",
            iterations as f64 / sync_duration.as_secs_f64()
        );
        println!(
            "Async throughput: {:.1} iter/s",
            iterations as f64 / async_duration.as_secs_f64()
        );
        println!();

        if speedup > 1.0 {
            println!("✅ Async mode is faster!");
        } else {
            println!("⚠️  Sync mode is faster (async overhead may dominate for this workload)");
        }

        // ========== Batch Execution Demo ==========
        println!("\n=== Batch Execution Demo ===");
        let batch_size = 16;

        let configs: Vec<RuntimeConfig> = (0..batch_size)
            .map(|i| {
                let mut config = config_base;
                config.iteration = i as i32;
                config
            })
            .collect();

        let start = Instant::now();
        let results = ctx_async.execute_batch(&configs)?;
        let batch_duration = start.elapsed();

        println!("Processed {} configs in {:?}", batch_size, batch_duration);
        println!("Received {} telemetry results", results.len());
        println!(
            "Batch throughput: {:.1} iter/s",
            batch_size as f64 / batch_duration.as_secs_f64()
        );
        println!();

        // Cleanup
        ctx_sync.shutdown()?;
        ctx_async.shutdown()?;

        println!("=== Shutdown Complete ===");

        Ok(())
    }
}
