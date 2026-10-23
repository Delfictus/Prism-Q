// [STAGE-2-FFI] OptiX FFI Build Script
// Generates Rust bindings from OptiX C headers using bindgen
//
// This build script:
// 1. Locates OptiX SDK headers via OPTIX_ROOT environment variable
// 2. Uses bindgen to generate unsafe Rust FFI bindings
// 3. Links against CUDA and OptiX runtime libraries
//
// Requirements:
// - OPTIX_ROOT: Path to OptiX SDK installation
// - CUDA: CUDA toolkit must be installed (runtime comes from NVIDIA driver)
//
// OptiX 9.1.0 requirements:
// - NVIDIA driver R590 or later
// - RTX GPU (Turing, Ampere, Ada, or Blackwell architecture)

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=OPTIX_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    // ========================================================================
    // Step 1: Locate OptiX SDK headers
    // ========================================================================

    // Try to get OPTIX_ROOT from environment, expanding ~ if needed
    let optix_root = env::var("OPTIX_ROOT")
        .ok()
        .or_else(|| {
            // Check home directory first
            if let Ok(home) = env::var("HOME") {
                let local_optix = format!("{}/.local/opt/optix-9.1.0", home);
                if std::path::Path::new(&local_optix)
                    .join("include/optix.h")
                    .exists()
                {
                    return Some(local_optix);
                }
            }

            // Try common system locations
            for path in &["/opt/optix-9.1.0", "/usr/local/optix", "/opt/optix"] {
                if std::path::Path::new(path).join("include/optix.h").exists() {
                    return Some(path.to_string());
                }
            }

            None
        })
        .unwrap_or_else(|| {
            panic!(
                "❌ OPTIX_ROOT not set and OptiX SDK not found in common locations!\n\
             \n\
             Please set OPTIX_ROOT to your OptiX SDK installation:\n\
             export OPTIX_ROOT=/path/to/optix-9.1.0\n\
             \n\
             Or install OptiX headers from: https://github.com/NVIDIA/optix-dev\n\
             \n\
             Searched locations:\n\
             - $HOME/.local/opt/optix-9.1.0\n\
             - /opt/optix-9.1.0\n\
             - /usr/local/optix\n\
             - /opt/optix"
            );
        });

    let optix_include = format!("{}/include", optix_root);

    // Verify critical headers exist
    let optix_header = format!("{}/optix.h", optix_include);
    let optix_host_header = format!("{}/optix_host.h", optix_include);
    let optix_stubs_header = format!("{}/optix_stubs.h", optix_include);

    for header in &[&optix_header, &optix_host_header, &optix_stubs_header] {
        if !std::path::Path::new(header).exists() {
            panic!("❌ Required OptiX header not found: {}", header);
        }
    }

    println!("✅ Found OptiX SDK at: {}", optix_root);
    println!("   Headers: {}", optix_include);

    // ========================================================================
    // Step 2: Locate CUDA (for cudaStream_t and other CUDA types)
    // ========================================================================

    let cuda_path = env::var("CUDA_PATH").unwrap_or_else(|_| {
        // Try common CUDA locations
        for path in &[
            "/usr/local/cuda-13.1",
            "/usr/local/cuda-13.0",
            "/usr/local/cuda-12",
            "/usr/local/cuda",
        ] {
            if std::path::Path::new(path).join("include/cuda.h").exists() {
                return path.to_string();
            }
        }
        "/usr/local/cuda".to_string() // Fallback
    });

    let cuda_include = format!("{}/include", cuda_path);
    println!("✅ Using CUDA headers: {}", cuda_include);

    // ========================================================================
    // Step 3: Generate Rust FFI bindings with bindgen
    // ========================================================================

    println!("🔨 Generating OptiX FFI bindings...");

    // Find GCC system headers for stddef.h, stdint.h, etc.
    let gcc_include = "/usr/lib/gcc/x86_64-linux-gnu/13/include";
    let system_include = "/usr/include";

    let bindings = bindgen::Builder::default()
        // Primary OptiX header (includes all others)
        .header(&optix_host_header)
        .header(&optix_stubs_header)
        // Include paths
        .clang_arg(format!("-I{}", optix_include))
        .clang_arg(format!("-I{}", cuda_include))
        .clang_arg(format!("-I{}", gcc_include))
        .clang_arg(format!("-I{}", system_include))
        // OptiX 9.1 API functions and types
        .allowlist_function("optix.*")
        .allowlist_type("Optix.*")
        .allowlist_type("OPTIX_.*")
        .allowlist_var("OPTIX_.*")
        // CUDA types used by OptiX (cudaStream_t, CUdeviceptr, etc.)
        .allowlist_type("cudaStream_t")
        .allowlist_type("CUstream")
        .allowlist_type("CUdeviceptr")
        .allowlist_type("CUcontext")
        .allowlist_type("CUdevice")
        // Derive common traits where possible
        .derive_debug(true)
        .derive_default(true)
        .derive_copy(true)
        .derive_eq(true)
        .derive_hash(true)
        // Layout configuration
        .layout_tests(false) // Skip layout tests (large generated code)
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: false,
        })
        // Generate comments from headers
        .generate_comments(true)
        .clang_arg("-fparse-all-comments")
        // Target x86_64 Linux
        .clang_arg("-target")
        .clang_arg("x86_64-unknown-linux-gnu")
        // Finish
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("❌ Failed to generate OptiX bindings");

    // ========================================================================
    // Step 4: Write bindings to output file
    // ========================================================================

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    let bindings_path = out_path.join("optix_bindings.rs");

    bindings
        .write_to_file(&bindings_path)
        .expect("❌ Failed to write OptiX bindings");

    println!("✅ Generated bindings: {}", bindings_path.display());

    // ========================================================================
    // Step 5: Link against CUDA runtime (OptiX runtime comes from driver)
    // ========================================================================

    // OptiX itself is header-only + driver-provided runtime
    // But we need CUDA runtime for cudaStream_t and other CUDA types

    let cuda_lib = format!("{}/lib64", cuda_path);
    println!("cargo:rustc-link-search=native={}", cuda_lib);
    println!("cargo:rustc-link-lib=cuda"); // libcuda.so (CUDA Driver API)
    println!("cargo:rustc-link-lib=cudart"); // libcudart.so (CUDA Runtime API)

    println!("✅ Build script complete");
}
