//! GPU Context Integration Tests
//!
//! Tests for GPU context initialization, security, and telemetry.
//! Requires GPU hardware to run (marked with #[ignore]).
//!
//! Run with: cargo test -p prism-gpu --features cuda -- --ignored

use prism_gpu::context::{GpuContext, GpuSecurityConfig};
use std::path::PathBuf;

#[test]
#[ignore] // Requires GPU hardware
fn test_gpu_context_initialization() {
    env_logger::builder().is_test(true).try_init().ok();

    let config = GpuSecurityConfig::default();
    let ptx_dir = PathBuf::from("target/ptx");

    // Create PTX directory if it doesn't exist (for CI)
    std::fs::create_dir_all(&ptx_dir).ok();

    let result = GpuContext::new(0, config, &ptx_dir);

    match result {
        Ok(ctx) => {
            println!("GPU context initialized successfully");

            // Verify device is accessible
            let device = ctx.device();
            println!("Device ordinal: {}", device.ordinal());

            // Check security config
            assert!(!ctx.is_secure_mode());
            assert!(!ctx.allows_nvrtc());
        }
        Err(e) => {
            // In CI/test environments without GPU, this is expected
            println!(
                "GPU context initialization failed (expected without GPU): {}",
                e
            );
        }
    }
}

#[test]
#[ignore] // Requires GPU hardware
fn test_gpu_context_module_loading() {
    env_logger::builder().is_test(true).try_init().ok();

    let config = GpuSecurityConfig::default();
    let ptx_dir = PathBuf::from("target/ptx");

    std::fs::create_dir_all(&ptx_dir).ok();

    if let Ok(ctx) = GpuContext::new(0, config, &ptx_dir) {
        // Check if standard modules are loaded
        // Note: Modules may not be loaded if PTX files don't exist
        if ctx.has_module("quantum") {
            println!("Quantum module loaded successfully");
        } else {
            println!("Quantum module not loaded (PTX file may be missing)");
        }

        if ctx.has_module("dendritic_reservoir") {
            println!("Dendritic reservoir module loaded successfully");
        }
    }
}

#[test]
fn test_gpu_context_invalid_directory() {
    env_logger::builder().is_test(true).try_init().ok();

    let config = GpuSecurityConfig::default();
    let ptx_dir = PathBuf::from("/nonexistent/ptx/directory");

    let result = GpuContext::new(0, config, &ptx_dir);

    // GpuContext does not implement Debug, so Result::unwrap_err() (which
    // requires the Ok type to be Debug) cannot be used. Match explicitly to
    // assert the error and extract its message while preserving test intent.
    let err_msg = match result {
        Ok(_) => panic!("expected GpuContext::new to fail for a nonexistent PTX directory"),
        Err(e) => e.to_string(),
    };
    assert!(err_msg.contains("does not exist") || err_msg.contains("CUDA"));
}

#[test]
#[ignore] // Requires GPU hardware
fn test_gpu_info_collection() {
    env_logger::builder().is_test(true).try_init().ok();

    let config = GpuSecurityConfig::default();
    let ptx_dir = PathBuf::from("target/ptx");

    std::fs::create_dir_all(&ptx_dir).ok();

    if let Ok(ctx) = GpuContext::new(0, config, &ptx_dir) {
        match ctx.collect_gpu_info() {
            Ok(info) => {
                println!("GPU Info:");
                println!("  Device: {}", info.device_name);
                println!(
                    "  Compute Capability: {}.{}",
                    info.compute_capability.0, info.compute_capability.1
                );
                println!("  Memory: {} MB", info.total_memory_mb);
                println!("  Driver: {}", info.driver_version);

                assert!(!info.device_name.is_empty());
                assert!(info.total_memory_mb > 0);
            }
            Err(e) => {
                println!("Failed to collect GPU info: {}", e);
            }
        }
    }
}

#[test]
#[ignore] // Requires GPU hardware
fn test_gpu_utilization_query() {
    env_logger::builder().is_test(true).try_init().ok();

    let config = GpuSecurityConfig::default();
    let ptx_dir = PathBuf::from("target/ptx");

    std::fs::create_dir_all(&ptx_dir).ok();

    if let Ok(ctx) = GpuContext::new(0, config, &ptx_dir) {
        match ctx.get_utilization() {
            Ok(util) => {
                println!("GPU Utilization: {:.1}%", util * 100.0);

                // Note: Current implementation returns 0.0 (NVML not integrated)
                // In production, this would return actual utilization
                assert!(
                    util >= 0.0 && util <= 1.0,
                    "Utilization out of range: {}",
                    util
                );
            }
            Err(e) => {
                println!("Failed to query GPU utilization: {}", e);
            }
        }
    }
}

#[test]
fn test_ptx_signature_verification() {
    env_logger::builder().is_test(true).try_init().ok();

    // Create test directory
    let test_dir = PathBuf::from("/tmp/prism_test_ptx");
    std::fs::create_dir_all(&test_dir).ok();

    // Create dummy PTX file
    let ptx_path = test_dir.join("test.ptx");
    std::fs::write(&ptx_path, "// Dummy PTX content").ok();

    // Create valid signature
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(b"// Dummy PTX content");
    let hash = hasher.finalize();
    let sig_path = test_dir.join("test.ptx.sha256");
    std::fs::write(&sig_path, hex::encode(hash)).ok();

    // Test with signature verification enabled
    let config = GpuSecurityConfig {
        allow_nvrtc: false,
        require_signed_ptx: true,
        trusted_ptx_dir: Some(test_dir.clone()),
    };

    // This will fail on CUDA device init in test env, but signature logic runs first
    let result = GpuContext::new(0, config, &test_dir);

    // Cleanup
    std::fs::remove_dir_all(&test_dir).ok();

    // In test env without GPU, we expect CUDA init failure, not signature failure
    if let Err(e) = result {
        let err_msg = e.to_string();
        // Should not fail on signature (signature is valid)
        assert!(!err_msg.contains("signature verification failed"));
    }
}

#[ignore = "GPU_RUNTIME_REQUIRED: needs real GPU + built PTX; run on the RTX workstation via `cargo test -- --ignored`"]
#[test]
fn test_ptx_signature_mismatch() {
    env_logger::builder().is_test(true).try_init().ok();

    // Create test directory
    let test_dir = PathBuf::from("/tmp/prism_test_ptx_bad");
    std::fs::create_dir_all(&test_dir).ok();

    // Create PTX file
    let ptx_path = test_dir.join("test.ptx");
    std::fs::write(&ptx_path, "// Dummy PTX content").ok();

    // Create INVALID signature (hash of different content)
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(b"// Wrong content");
    let hash = hasher.finalize();
    let sig_path = test_dir.join("test.ptx.sha256");
    std::fs::write(&sig_path, hex::encode(hash)).ok();

    // Test with signature verification enabled
    let config = GpuSecurityConfig {
        allow_nvrtc: false,
        require_signed_ptx: true,
        trusted_ptx_dir: Some(test_dir.clone()),
    };

    let result = GpuContext::new(0, config, &test_dir);

    // Cleanup
    std::fs::remove_dir_all(&test_dir).ok();

    // Should fail - signature mismatch or missing
    assert!(result.is_err());
}

#[test]
fn test_security_config_modes() {
    let default_config = GpuSecurityConfig::default();
    assert!(!default_config.allow_nvrtc);
    assert!(!default_config.require_signed_ptx);
    assert!(default_config.trusted_ptx_dir.is_none());

    let permissive = GpuSecurityConfig::permissive();
    assert!(permissive.allow_nvrtc);
    assert!(!permissive.require_signed_ptx);

    let strict = GpuSecurityConfig::strict(PathBuf::from("/trusted"));
    assert!(!strict.allow_nvrtc);
    assert!(strict.require_signed_ptx);
    assert!(strict.trusted_ptx_dir.is_some());
}

#[test]
fn test_is_available() {
    // Just verify the function runs without panicking
    let available = GpuContext::is_available();
    println!("GPU available: {}", available);

    // No assertion - GPU may or may not be present in test environment
}

#[test]
#[ignore] // Requires GPU hardware
fn test_multiple_context_creation() {
    env_logger::builder().is_test(true).try_init().ok();

    let config = GpuSecurityConfig::default();
    let ptx_dir = PathBuf::from("target/ptx");

    std::fs::create_dir_all(&ptx_dir).ok();

    // Create multiple contexts (should work - Arc<CudaContext> is thread-safe)
    if let Ok(ctx1) = GpuContext::new(0, config.clone(), &ptx_dir) {
        if let Ok(ctx2) = GpuContext::new(0, config, &ptx_dir) {
            println!("Multiple contexts created successfully");

            // Both should point to same device
            assert_eq!(ctx1.device().ordinal(), ctx2.device().ordinal());
        }
    }
}
