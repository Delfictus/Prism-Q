//! Stream Integration Demo
//!
//! Demonstrates triple-buffered GPU execution using ManagedGpuContext.
//!
//! Run with:
//! ```bash
//! cargo run --example stream_integration_demo --features cuda
//! ```

use anyhow::Result;
use cudarc::driver::CudaContext;
use prism_core::RuntimeConfig;
use prism_gpu::stream_integration::ManagedGpuContext;
use std::sync::Arc;

fn main() -> Result<()> {
    env_logger::init();

    println!("=== Stream Integration Demo ===\n");

    // Check if GPU is available
    if !prism_gpu::GpuContext::is_available() {
        println!("❌ No GPU detected. This demo requires CUDA-capable GPU.");
        println!("   Exiting gracefully.");
        return Ok(());
    }

    let device = CudaContext::new(0)?;
    println!("✅ GPU detected: Device 0\n");

    // Demo 1: Synchronous execution (baseline)
    println!("Demo 1: Synchronous Execution (Baseline)");
    println!("─────────────────────────────────────────");
    demo_sync_execution(device.clone())?;

    // Demo 2: Asynchronous execution (triple-buffered)
    println!("\nDemo 2: Asynchronous Execution (Triple-Buffered)");
    println!("─────────────────────────────────────────────────");
    demo_async_execution(device.clone())?;

    // Demo 3: Stream access patterns
    println!("\nDemo 3: Stream Access Patterns");
    println!("───────────────────────────────");
    demo_stream_access(device.clone())?;

    println!("\n=== Demo Complete ===");

    Ok(())
}

/// Demonstrates synchronous execution (no stream management)
fn demo_sync_execution(device: Arc<CudaContext>) -> Result<()> {
    let ctx = ManagedGpuContext::new(device, false)?;

    println!("  Stream management: {}", ctx.has_stream_management());
    println!("  Stream pool:       {:?}", ctx.stream_pool().is_some());
    println!(
        "  Coordinator:       {:?}",
        ctx.pipeline_coordinator().is_some()
    );

    // Synchronous execution path
    ctx.device().synchronize()?;
    println!("  ✅ Synchronous execution verified");

    Ok(())
}

/// Demonstrates asynchronous triple-buffered execution
fn demo_async_execution(device: Arc<CudaContext>) -> Result<()> {
    let mut ctx = ManagedGpuContext::new(device, true)?;

    println!("  Stream management: {}", ctx.has_stream_management());
    println!("  Stream pool:       {:?}", ctx.stream_pool().is_some());
    println!(
        "  Coordinator:       {:?}",
        ctx.pipeline_coordinator().is_some()
    );

    // Run triple-buffered iterations
    let num_iterations = 10;
    println!(
        "\n  Running {} triple-buffered iterations...",
        num_iterations
    );

    for iter in 0..num_iterations {
        let config = RuntimeConfig::default();
        let _telemetry = ctx.triple_buffered_step(config)?;

        if iter % 5 == 0 {
            println!("    Iteration {} complete", iter);
        }
    }

    println!("  ✅ Triple-buffered execution verified");

    Ok(())
}

/// Demonstrates stream access patterns
fn demo_stream_access(device: Arc<CudaContext>) -> Result<()> {
    use prism_gpu::stream_manager::StreamPurpose;

    let mut ctx = ManagedGpuContext::new(device, true)?;

    println!("  Available stream purposes:");

    // Try to get streams for different purposes
    for purpose in &[
        StreamPurpose::ConfigUpload,
        StreamPurpose::KernelExecution,
        StreamPurpose::TelemetryDownload,
        StreamPurpose::P2PTransfer,
        StreamPurpose::AuxCompute,
    ] {
        if let Some(stream_idx) = ctx.get_stream(*purpose) {
            println!("    {:?}: stream_idx = {}", purpose, stream_idx);
        }
    }

    println!("\n  ✅ Stream access verified");

    Ok(())
}
