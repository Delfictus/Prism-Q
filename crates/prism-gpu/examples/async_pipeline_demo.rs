//! Async Pipeline Demo
//!
//! Demonstrates triple buffering and event-based async coordination.
//!
//! Run with: cargo run --example async_pipeline_demo --features cuda

use anyhow::Result;
use cudarc::driver::{CudaContext, LaunchConfig};
use prism_gpu::stream_integration::{
    AsyncCoordinator, ManagedGpuContext, PipelineStage, PipelineStageManager, TripleBuffer,
};
use std::sync::Arc;

fn main() -> Result<()> {
    env_logger::init();

    println!("=== PRISM Async Pipeline Demo ===\n");

    // Check for CUDA device
    let device = match CudaContext::new(0) {
        Ok(dev) => Arc::new(dev),
        Err(e) => {
            eprintln!("ERROR: CUDA device not available: {}", e);
            eprintln!("This demo requires a CUDA-capable GPU.");
            return Ok(());
        }
    };

    println!("✓ CUDA device initialized\n");

    // Demo 1: GPU Triple Buffering
    demo_triple_buffer(&device)?;

    // Demo 2: Async Coordinator
    demo_async_coordinator(&device)?;

    // Demo 3: Pipeline Stage Manager
    demo_pipeline_manager(&device)?;

    // Demo 4: ManagedGpuContext Integration
    demo_managed_context(&device)?;

    println!("\n=== Demo Complete ===");
    Ok(())
}

/// Demo 1: GPU-side triple buffering
fn demo_triple_buffer(device: &Arc<CudaContext>) -> Result<()> {
    println!("--- Demo 1: GPU Triple Buffering ---");

    let buffer = TripleBuffer::<f32>::new(device, 1024)?;
    println!("✓ Created triple buffer (3 × 1024 elements)");

    // NOTE: TripleBuffer::get_states() returns the crate-private `BufferState`
    // enum, which is not nameable/usable from an external example crate. The
    // buffer state transitions are still exercised below; only the debug print
    // of the internal state array is omitted.

    // Get upload buffer (should succeed)
    match buffer.get_upload_buffer() {
        Some(_) => println!("✓ Got upload buffer (Free → Uploading)"),
        None => println!("✗ Upload buffer unavailable"),
    }

    // Advance rotation
    buffer.advance();
    println!("✓ Advanced buffer rotation\n");

    Ok(())
}

/// Demo 2: Async coordinator with event-based synchronization
fn demo_async_coordinator(device: &Arc<CudaContext>) -> Result<()> {
    println!("--- Demo 2: Async Coordinator ---");

    let mut coordinator = AsyncCoordinator::new(device.clone())?;
    println!("✓ Created async coordinator");

    // Get stream references (AsyncCoordinator::new always creates these in the
    // current API — they are &Arc<CudaStream>, no longer Option-wrapped).
    let _compute_stream = coordinator.compute_stream();
    let _upload_stream = coordinator.upload_stream();
    let _download_stream = coordinator.download_stream();

    println!("  Streams created:");
    println!("    - Compute stream: ready");
    println!("    - Upload stream: ready");
    println!("    - Download stream: ready");

    // Create test data
    let host_data = vec![1.0f32; 512];
    let mut host_result = vec![0.0f32; 512];

    let buffer = TripleBuffer::<f32>::new(device, 512)?;

    // Queue operations
    let upload_id = coordinator.queue_upload(&host_data, &buffer)?;
    println!("✓ Queued upload (op_id: {})", upload_id);

    let compute_id = coordinator.queue_compute("demo_kernel", &[upload_id])?;
    println!(
        "✓ Queued compute (op_id: {}, depends on: {})",
        compute_id, upload_id
    );

    let download_id = coordinator.queue_download(&buffer, &mut host_result)?;
    println!("✓ Queued download (op_id: {})", download_id);

    // Poll for completion (non-blocking)
    println!("  Polling for completion...");
    let completed = coordinator.poll();
    println!("  Completed operations: {}", completed.len());
    for op in &completed {
        println!("    - Op {}: {}", op.op_id, op.op_type);
    }

    // Wait for final download
    coordinator.wait(download_id)?;
    println!("✓ All operations complete\n");

    Ok(())
}

/// Demo 3: Multi-stage pipeline with automatic overlapping
fn demo_pipeline_manager(device: &Arc<CudaContext>) -> Result<()> {
    println!("--- Demo 3: Pipeline Stage Manager ---");

    let coordinator = AsyncCoordinator::new(device.clone())?;
    let mut pipeline = PipelineStageManager::new(coordinator);
    println!("✓ Created pipeline manager");

    // Define pipeline stages
    let preprocess = PipelineStage {
        name: "Preprocess".to_string(),
        input_buffers: vec![],
        output_buffers: vec![0],
        config: LaunchConfig {
            grid_dim: (8, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        },
    };

    let main_compute = PipelineStage {
        name: "MainCompute".to_string(),
        input_buffers: vec![0],
        output_buffers: vec![1],
        config: LaunchConfig {
            grid_dim: (16, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 2048,
        },
    };

    let postprocess = PipelineStage {
        name: "Postprocess".to_string(),
        input_buffers: vec![1],
        output_buffers: vec![2],
        config: LaunchConfig {
            grid_dim: (4, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        },
    };

    // Add stages
    let idx0 = pipeline.add_stage(preprocess);
    let idx1 = pipeline.add_stage(main_compute);
    let idx2 = pipeline.add_stage(postprocess);
    println!("✓ Added 3 pipeline stages");

    // Set execution order
    pipeline.set_order(vec![idx0, idx1, idx2]);
    println!("✓ Set execution order: Preprocess → MainCompute → Postprocess");

    // Execute iterations
    println!("  Executing 10 iterations...");
    for i in 0..10 {
        pipeline.execute_iteration()?;
        if (i + 1) % 5 == 0 {
            let stats = pipeline.get_stats();
            println!(
                "    After {} iterations: {:.2} iter/sec",
                i + 1,
                stats.throughput
            );
        }
    }

    // Final statistics
    let stats = pipeline.get_stats();
    println!("\n  Final Statistics:");
    println!("    Iterations: {}", stats.iterations);
    println!("    Avg time: {:.2} ms/iter", stats.avg_iteration_ms);
    println!("    Throughput: {:.2} iter/sec", stats.throughput);
    println!();

    Ok(())
}

/// Demo 4: Integration with ManagedGpuContext
fn demo_managed_context(device: &Arc<CudaContext>) -> Result<()> {
    println!("--- Demo 4: ManagedGpuContext Integration ---");

    // Create with stream management enabled
    let ctx = ManagedGpuContext::new((*device).clone(), true)?;
    println!("✓ Created ManagedGpuContext (async enabled)");

    // Check capabilities
    println!("  Capabilities:");
    println!("    - Stream management: {}", ctx.has_stream_management());
    println!(
        "    - Async coordinator: {}",
        ctx.async_coordinator().is_some()
    );
    println!(
        "    - Pipeline coordinator: {}",
        ctx.pipeline_coordinator().is_some()
    );

    // Create triple buffer via context
    let buffer = ctx.create_triple_buffer::<f32>(2048)?;
    println!("✓ Created triple buffer via context (2048 elements)");
    println!("  Buffer size: {}", buffer.size());

    // Create pipeline manager via context
    let pipeline = ctx.create_pipeline_manager()?;
    println!("✓ Created pipeline manager via context");

    // Test without stream management
    let sync_ctx = ManagedGpuContext::new((*device).clone(), false)?;
    println!("\n✓ Created sync context (stream management disabled)");
    println!("  Stream management: {}", sync_ctx.has_stream_management());

    match sync_ctx.create_pipeline_manager() {
        Ok(_) => println!("✗ Unexpected: pipeline created without streams"),
        Err(e) => println!("✓ Expected error: {}", e),
    }

    println!();
    Ok(())
}
