# optix-sys

**Low-level unsafe FFI bindings to NVIDIA OptiX 9.1.0 ray tracing API**

Part of the PRISM4D-bio RT Core Integration (Stage 2a)

---

## Overview

This crate provides direct access to the OptiX C API through automatically generated Rust FFI bindings. All functions are `unsafe` and require careful CUDA context management.

**For safe Rust wrapper, use `prism-optix` instead.**

## Architecture

OptiX is a header-only SDK with runtime functionality provided by the NVIDIA driver:

1. **Headers**: Cloned from [github.com/NVIDIA/optix-dev](https://github.com/NVIDIA/optix-dev) v9.1.0
2. **Build**: Uses `bindgen` to generate Rust FFI bindings from C headers
3. **Runtime**: OptiX functions are loaded dynamically from NVIDIA driver (R590+)
4. **Linking**: Links against CUDA runtime libraries (libcuda.so, libcudart.so)

## Requirements

- **GPU**: NVIDIA RTX GPU (Turing, Ampere, Ada, or Blackwell architecture)
- **Driver**: R590 or later (for OptiX 9.1.0)
- **CUDA**: CUDA toolkit installed (for header dependencies)
- **OptiX Headers**: Installed at `$OPTIX_ROOT` or `~/.local/opt/optix-9.1.0`

## Installation

### OptiX Headers

The build script will automatically find OptiX headers if `OPTIX_ROOT` is set or headers are in `~/.local/opt/optix-9.1.0`.

To install OptiX headers:

```bash
# Clone OptiX headers from NVIDIA
git clone https://github.com/NVIDIA/optix-dev.git
cd optix-dev
git checkout v9.1.0

# Install to user directory
mkdir -p ~/.local/opt/optix-9.1.0
cp -r include ~/.local/opt/optix-9.1.0/

# Set environment variable (add to ~/.bashrc for persistence)
export OPTIX_ROOT="$HOME/.local/opt/optix-9.1.0"
```

### Building

```bash
cargo build -p optix-sys
```

The build script will:
1. Locate OptiX headers (via `$OPTIX_ROOT` or common locations)
2. Locate CUDA headers (via `$CUDA_PATH` or `/usr/local/cuda`)
3. Generate Rust FFI bindings using bindgen
4. Link against CUDA runtime libraries

## Usage

⚠️  **WARNING**: All OptiX functions are `unsafe`. Improper use can cause GPU faults or driver crashes.

```rust
use optix_sys::*;

unsafe {
    // Initialize OptiX
    optixInit();

    // Create OptiX context
    let mut context: OptixDeviceContext = std::ptr::null_mut();
    let options = OptixDeviceContextOptions::default();

    optixDeviceContextCreate(
        cu_context,
        &options,
        &mut context,
    );

    // Use OptiX functions...

    // Clean up
    optixDeviceContextDestroy(context);
}
```

## Generated Bindings

- **Size**: ~165KB (2990 lines)
- **Functions**: All OptiX 9.1 API functions (`optix*`)
- **Types**: All OptiX types (`Optix*`, `OPTIX_*`)
- **CUDA Types**: cudaStream_t, CUdeviceptr, CUcontext, CUdevice

## Version

- **Crate Version**: 9.1.0 (matches OptiX SDK version)
- **OptiX SDK**: 9.1.0
- **Minimum Driver**: R590 or later

## Testing

```bash
# Run unit tests (compile-time checks)
cargo test -p optix-sys

# Tests verify:
# - Version constants are correct
# - Key OptiX types exist (OptixDeviceContext, OptixModule, OptixPipeline)
# - OptiX result codes are accessible
```

All tests pass ✅

## Safety

**NEVER** call OptiX functions from safe Rust code. Always use through the `prism-optix` safe wrapper crate which provides:

- RAII-style resource management
- Error handling via `Result<T, E>`
- Safe abstractions over raw pointers
- Automatic CUDA context management

## Integration

Part of PRISM4D-bio RT Core Integration:

- **Phase 1** ✅: Configuration, solvation, RT targets, preparation (40 tests, 100% passing)
- **Phase 2.1** ✅: OptiX FFI bindings (this crate)
- **Phase 2.2** (next): Safe Rust wrapper (`prism-optix`)
- **Phase 2.3**: BVH acceleration structure
- **Phase 2.4**: RT probe engine
- **Phase 2.5**: Comprehensive testing

## References

- [OptiX 9.1 Release Notes](https://developer.nvidia.com/downloads/designworks/optix/secure/9.1.0/optix_release_notes_9.1_01.pdf)
- [OptiX API Documentation](https://raytracing-docs.nvidia.com/optix9/index.html)
- [OptiX GitHub (headers)](https://github.com/NVIDIA/optix-dev)
- [OptiX Downloads](https://developer.nvidia.com/designworks/optix/download)

---

**Status**: ✅ Phase 2.1 Complete - FFI bindings generated and tested
**Next**: Phase 2.2 - Create `prism-optix` safe wrapper
