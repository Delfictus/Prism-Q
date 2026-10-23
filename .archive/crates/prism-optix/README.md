# prism-optix

**Safe Rust wrapper for NVIDIA OptiX 9.1.0 ray tracing API**

Part of the PRISM4D-bio RT Core Integration (Stage 2a)

---

## Overview

This crate provides safe, ergonomic abstractions over the unsafe `optix-sys` FFI bindings. It uses RAII patterns for resource management and Rust's `Result<T, E>` for error handling.

**Status**: Phase 2.2 Complete - Core infrastructure (error handling, types)
**Next**: Phase 2.3 - Full context creation with OptiX function table

## Features

- ✅ **Comprehensive Error Handling**: Rich error types with context messages
- ✅ **Type-Safe API**: No raw pointers in public API
- ✅ **RAII Patterns**: Automatic resource cleanup
- ✅ **CUDA Integration**: Works seamlessly with cudarc
- ✅ **Function Table**: Dynamic OptiX API loading with libloading
- ✅ **Context Management**: Full lifecycle (init, create, destroy, cache)
- 🔄 **BVH Acceleration**: Infrastructure complete, full build/refit next

## Architecture

OptiX uses a **function table pattern** where all API functions are loaded dynamically from the NVIDIA driver. Phase 2.2 provides the foundational error handling and type infrastructure. Phase 2.3 will implement:

1. OptiX function table initialization
2. Full context creation and management
3. BVH acceleration structure builders
4. Ray tracing launch primitives

## Requirements

- **GPU**: NVIDIA RTX GPU (Turing, Ampere, Ada, or Blackwell architecture)
- **Driver**: R590 or later (for OptiX 9.1.0)
- **CUDA**: CUDA context must be current
- **Dependencies**: optix-sys, cudarc, anyhow, thiserror, log

## Error Handling

### OptixError Types

```rust
pub enum OptixError {
    InitializationFailed(String),
    InvalidContext(String),
    InvalidValue(String),
    OutOfMemory(String),
    CudaError(String),
    VersionMismatch { expected: u32, actual: u32 },
    InternalError(String),
    NotImplemented(String),
    PipelineError(String),
    AccelError(String),
    Unknown { code: u32, message: String },
}
```

### Error Conversion

All OptiX result codes are automatically converted to typed errors with context:

```rust
use prism_optix::{check_optix, OptixError, Result};
use optix_sys::{OptixResult, OPTIX_SUCCESS};

// Check OptiX result and convert to Result<(), OptixError>
check_optix(OPTIX_SUCCESS, "operation description")?;

// Create error from result code
let err = OptixError::from_result(
    OptixResult::OPTIX_ERROR_INVALID_VALUE,
    "invalid parameter passed"
);
```

### Error Utilities

```rust
// Get human-readable error name
let name = OptixError::optix_error_name(
    OptixResult::OPTIX_ERROR_CUDA_ERROR
);
// Returns: "OPTIX_ERROR_CUDA_ERROR"

// Errors implement Display and Error traits
println!("{}", err);  // Pretty-printed error message
log::error!("{:?}", err);  // Debug formatting
```

## Version Information

```rust
use prism_optix::version;

// Get version as tuple
let (major, minor, micro) = version::version();  // (9, 1, 0)

// Get version as integer
let ver = version::version_number();  // 90100

// Get version as string
let ver_str = version::version_string();  // "9.1.0"
```

## Types

### OptixContext (RAII wrapper)

```rust
use prism_optix::OptixContext;

// Create from existing handle (Phase 2.3 will provide full creation)
let ctx = unsafe {
    OptixContext::from_handle(handle, cuda_ctx, true)
};

// Access properties
let validation = ctx.validation_enabled();
let handle = ctx.handle();
let cuda_ctx = ctx.cuda_context();

// Automatic cleanup on drop (Phase 2.3)
```

## Testing

```bash
# Run all tests
cargo test -p prism-optix

# Tests:
# - error::test_error_conversion
# - error::test_check_optix_success
# - error::test_check_optix_failure
# - error::test_error_name
# - context::test_optix_version
# - test_version_info
```

All 6 tests passing ✅

## Implementation Status

### Phase 2.1 ✅ Complete
- optix-sys FFI bindings (9.1.0)
- Automatic header discovery
- bindgen integration
- 165KB, 2990 lines of bindings

### Phase 2.2 ✅ Complete (Current)
- Comprehensive error handling
- Type-safe error conversion
- Version utilities
- OptixContext foundation
- 6 passing tests

### Phase 2.3 ✅ Complete (Current)
- OptiX function table with libloading
- Full context creation (init, create, destroy)
- Cache management (location, enable/disable)
- Log callback integration
- BVH acceleration structure infrastructure
- 9 passing tests

### Phase 2.4 🔄 Future
- RT probe engine integration
- Ray generation programs
- Hit group compilation
- Pipeline management

### Phase 2.5 🔄 Future
- Comprehensive testing (single atom → 100K atoms)
- Performance benchmarks (<100ms BVH build)
- Integration with prism-nhs

## Performance

- **Zero-Cost Abstractions**: Thin wrappers with no runtime overhead
- **RAII**: Automatic cleanup without manual memory management
- **Error Handling**: Result<T, E> compiles to efficient machine code

## Safety

All unsafe code is contained within this crate:

- Public API is entirely safe Rust
- No raw pointers in public interfaces
- RAII ensures proper resource cleanup
- Comprehensive error handling prevents panics

## References

- [OptiX 9.1 API Documentation](https://raytracing-docs.nvidia.com/optix9/index.html)
- [OptiX 9.1 Release Notes](https://developer.nvidia.com/downloads/designworks/optix/secure/9.1.0/optix_release_notes_9.1_01.pdf)
- [OptiX GitHub (headers)](https://github.com/NVIDIA/optix-dev)
- [OptiX Programming Guide](https://raytracing-docs.nvidia.com/optix9/guide/index.html)

## Example (Phase 2.3+)

```rust
use prism_optix::{OptixContext, Result};
use cudarc::driver::CudaDevice;

fn main() -> Result<()> {
    env_logger::init();

    // Initialize OptiX API (Phase 2.3)
    OptixContext::init()?;

    // Create CUDA device
    let cuda_device = CudaDevice::new(0)?;

    // Create OptiX context with validation (Phase 2.3)
    let optix_ctx = OptixContext::new(
        cuda_device.cu_primary_ctx(),
        true  // validation enabled
    )?;

    // Set cache location for fast startup (Phase 2.3)
    optix_ctx.set_cache_location("/tmp/optix_cache")?;
    optix_ctx.set_cache_enabled(true)?;

    // Use for BVH building and ray tracing (Phase 2.4+)
    // ...

    Ok(())
    // OptiX context automatically destroyed here
}
```

---

**Status**: ✅ Phase 2.2 Complete - Safe wrapper infrastructure
**Next**: Phase 2.3 - OptiX function table and full context API
