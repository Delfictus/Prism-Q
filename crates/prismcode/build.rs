use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=PRISMCODE_PINNED_ROOT_HASH");
    println!("cargo:rerun-if-env-changed=PRISMCODE_PINNED_ROOT_PUBLIC_KEY_HEX");
    if let Ok(root_hash) = env::var("PRISMCODE_PINNED_ROOT_HASH") {
        println!("cargo:rustc-env=PRISMCODE_PINNED_ROOT_HASH={root_hash}");
    }
    if let Ok(public_key) = env::var("PRISMCODE_PINNED_ROOT_PUBLIC_KEY_HEX") {
        println!("cargo:rustc-env=PRISMCODE_PINNED_ROOT_PUBLIC_KEY_HEX={public_key}");
    }
    println!("cargo:rerun-if-changed=../../cuda/include/prismcode_cuda.h");
    println!("cargo:rerun-if-changed=../../cuda/src/prismcode_cuda.cu");
    println!("cargo:rerun-if-changed=../../cuda/src/prismcode_cuda_runtime.cu");
    println!("cargo:rerun-if-changed=../../cuda/src/nanl_pack.cu");
    println!("cargo:rerun-if-changed=../../cuda/src/event_integrate.cu");
    println!("cargo:rerun-if-changed=../../cuda/src/drf_fire.cu");
    println!("cargo:rerun-if-changed=../../cuda/src/drf_resonate.cu");
    println!("cargo:rerun-if-changed=../../cuda/src/tensor_ops.cu");
    println!("cargo:rerun-if-changed=../../cuda/src/thermo_reduce.cu");
    println!("cargo:rerun-if-changed=../../cuda/kernels/prismcode_cuda_device.cuh");
    println!("cargo:rerun-if-changed=../../cuda/kernels/tensor_reduce.cu");
    println!("cargo:rerun-if-changed=../../cuda/kernels/nanl_bin_pack.cu");
    println!("cargo:rerun-if-changed=../../cuda/kernels/nanl_q4_validate.cu");
    println!("cargo:rerun-if-env-changed=PRISMCODE_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=PRISMCODE_CUDA_ARCHES");

    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    if env::var_os("CXX").is_none() && PathBuf::from("/usr/bin/g++-12").exists() {
        env::set_var("CXX", "/usr/bin/g++-12");
    }
    let cuda_root = PathBuf::from("../../cuda");
    let arch_spec = cuda_arch_spec();
    println!("cargo:rustc-env=PRISMCODE_CUDA_TARGET_ARCHES={arch_spec}");
    let mut build = cc::Build::new();
    build
        .cuda(true)
        .std("c++17")
        .flag("-O3")
        .flag("-allow-unsupported-compiler")
        .flag("-Xcompiler")
        .flag("-fPIC")
        .include(cuda_root.join("include"))
        .include(cuda_root.join("kernels"))
        .file(cuda_root.join("src/prismcode_cuda.cu"))
        .file(cuda_root.join("src/prismcode_cuda_runtime.cu"))
        .file(cuda_root.join("src/nanl_pack.cu"))
        .file(cuda_root.join("src/event_integrate.cu"))
        .file(cuda_root.join("src/drf_fire.cu"))
        .file(cuda_root.join("src/drf_resonate.cu"))
        .file(cuda_root.join("src/tensor_ops.cu"))
        .file(cuda_root.join("src/thermo_reduce.cu"));
    for flag in cuda_arch_flags(&arch_spec) {
        build.flag(flag);
    }
    build.compile("prismcode_cuda");

    if let Ok(out_dir) = env::var("OUT_DIR") {
        let static_lib = PathBuf::from(out_dir).join("libprismcode_cuda.a");
        println!(
            "cargo:rustc-env=PRISMCODE_CUDA_STATIC_LIB={}",
            static_lib.display()
        );
    }

    println!("cargo:rustc-link-lib=cudart");
    for lib_dir in [
        "/usr/local/cuda/lib64",
        "/usr/local/cuda-12/lib64",
        "/usr/local/cuda-12.0/lib64",
        "/usr/lib/x86_64-linux-gnu",
    ] {
        if PathBuf::from(lib_dir).exists() {
            println!("cargo:rustc-link-search=native={lib_dir}");
        }
    }
}

fn cuda_arch_spec() -> String {
    env::var("PRISMCODE_CUDA_ARCHES")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            env::var("PRISMCODE_CUDA_ARCH")
                .ok()
                .filter(|value| !value.trim().is_empty())
        })
        // Ampere SASS plus PTX fallback is the safest repo default for the
        // current PrismOS bare-metal validation laptop and newer drivers.
        .unwrap_or_else(|| "sm_86".to_string())
}

fn cuda_arch_flags(spec: &str) -> Vec<String> {
    spec.split(|ch: char| ch == ',' || ch == ';' || ch.is_whitespace())
        .filter_map(|raw| {
            let arch = raw.trim();
            if arch.is_empty() {
                return None;
            }
            let compute = arch
                .strip_prefix("sm_")
                .or_else(|| arch.strip_prefix("compute_"))
                .unwrap_or(arch);
            Some(vec![
                format!("-gencode=arch=compute_{compute},code=sm_{compute}"),
                format!("-gencode=arch=compute_{compute},code=compute_{compute}"),
            ])
        })
        .flatten()
        .collect()
}
