use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=kernels/");

    let cuda_enabled = env::var("CARGO_FEATURE_CUDA").is_ok();
    if !cuda_enabled {
        println!("cargo:warning=CUDA feature not enabled, skipping PTX compilation");
        return;
    }

    let nvcc = find_nvcc().expect("nvcc not found");
    println!("cargo:info=Using nvcc: {}", nvcc);

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_dir = out_dir.join("ptx");
    std::fs::create_dir_all(&ptx_dir).unwrap();

    // Also create workspace-level target/ptx for easy access
    let target_ptx_dir = PathBuf::from("../../target/ptx");
    std::fs::create_dir_all(&target_ptx_dir).ok();

    // Device-side spike compaction (CUB-ready, SM120)
    compile_kernel(
        &nvcc,
        "kernels/device_compact.cu",
        &ptx_dir.join("device_compact.ptx"),
        &target_ptx_dir.join("device_compact.ptx"),
    );

    println!("cargo:info=prism-cuda-ext PTX compilation complete");
}

fn find_nvcc() -> Option<String> {
    if let Ok(cuda_home) = env::var("CUDA_HOME") {
        let p = PathBuf::from(cuda_home).join("bin").join("nvcc");
        if p.exists() {
            return Some(p.to_string_lossy().to_string());
        }
    }
    for path in &["/usr/local/cuda/bin/nvcc", "/usr/local/cuda-13.1/bin/nvcc"] {
        if PathBuf::from(path).exists() {
            return Some(path.to_string());
        }
    }
    if Command::new("nvcc").arg("--version").output().is_ok() {
        return Some("nvcc".to_string());
    }
    None
}

fn compile_kernel(nvcc: &str, source: &str, output: &PathBuf, target_output: &PathBuf) {
    println!("cargo:info=Compiling {} -> {}", source, output.display());

    let status = Command::new(nvcc)
        .arg("--ptx")
        .arg("-o")
        .arg(output)
        .arg(source)
        .arg("-arch=sm_120")
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("--restrict")
        .arg("-I/usr/local/cuda/include")
        .arg("-Xptxas=-v")
        .arg("--expt-relaxed-constexpr")
        .status()
        .expect("Failed to execute nvcc");

    if !status.success() {
        panic!("nvcc compilation failed for {}", source);
    }

    if let Err(e) = std::fs::copy(output, target_output) {
        println!("cargo:warning=Failed to copy PTX to target/ptx: {}", e);
    }
    println!("cargo:info=PTX compiled: {}", output.display());
}
