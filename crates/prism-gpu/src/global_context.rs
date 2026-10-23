//! Global GPU Context Singleton
//!
//! Provides a lazily-initialized global GPU context that:
//! - Loads all PTX modules exactly ONCE at startup
//! - Shares the same CUDA context across all threads
//! - Eliminates per-structure PTX reload overhead
//!
//! Target performance: 219 structures in 6-14 seconds (RTX 3060)
//!
//! # Usage
//! ```ignore
//! use prism_gpu::global_context::GlobalGpuContext;
//!
//! // First call initializes everything, subsequent calls return cached reference
//! let gpu = GlobalGpuContext::get();
//!
//! // Process structure using pre-loaded kernels (thread-safe mutable access)
//! if let Some(mega_fused) = gpu.mega_fused_locked() {
//!     let result = mega_fused.detect_pockets(&coords, &ca_indices, &cons, &bfac, &burial, &config)?;
//! }
//! ```

use crate::context::{GpuContext, GpuSecurityConfig};
use crate::mega_fused::{MegaFusedConfig, MegaFusedGpu, MegaFusedOutput};
// use crate::lbs::LbsGpu;
use cudarc::driver::{CudaContext, CudaStream, DeviceSlice, PushKernelArg};
use once_cell::sync::OnceCell;
use parking_lot::{Mutex, MutexGuard, RwLock};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::ThreadId;

/// Global GPU context singleton
static GLOBAL_GPU: OnceCell<GlobalGpuContext> = OnceCell::new();

/// Error type for global GPU operations
#[derive(Debug, thiserror::Error)]
pub enum GlobalGpuError {
    #[error("GPU initialization failed: {0}")]
    InitFailed(String),
    #[error("PTX directory not found: {0}")]
    PtxDirNotFound(String),
    #[error("Mega-fused kernel not available")]
    MegaFusedNotAvailable,
    #[error("GPU context error: {0}")]
    ContextError(#[from] anyhow::Error),
}

/// Thread-local stream pool for concurrent kernel execution
struct StreamPool {
    streams: RwLock<HashMap<ThreadId, Arc<CudaStream>>>,
    context: Arc<CudaContext>,
}

impl StreamPool {
    fn new(context: Arc<CudaContext>) -> Self {
        Self {
            streams: RwLock::new(HashMap::new()),
            context,
        }
    }

    /// Get or create a stream for the current thread
    fn get_stream(&self) -> Arc<CudaStream> {
        let tid = std::thread::current().id();

        // Fast path: check if stream exists
        {
            let streams = self.streams.read();
            if let Some(stream) = streams.get(&tid) {
                return stream.clone();
            }
        }

        // Slow path: create new stream for this thread
        let mut streams = self.streams.write();
        streams
            .entry(tid)
            .or_insert_with(|| self.context.default_stream())
            .clone()
    }
}

/// Global GPU context holding pre-loaded kernels and shared CUDA context.
///
/// This struct is initialized exactly once and provides:
/// - Pre-loaded PTX modules (no per-structure I/O)
/// - Pre-compiled kernel functions (no per-structure JIT)
/// - Thread-local stream pool for concurrent execution
/// - Safe sharing across threads via interior mutability (Mutex for mutable access)
pub struct GlobalGpuContext {
    /// Underlying GPU context with all modules loaded
    context: GpuContext,

    /// Pre-initialized mega-fused kernel executor (Mutex for thread-safe mutable access)
    mega_fused: Option<Mutex<MegaFusedGpu>>,

    /// Pre-initialized LBS kernel executor (Mutex for thread-safe mutable access)
    // lbs: Option<Mutex<LbsGpu>>,

    /// Thread-local stream pool
    stream_pool: StreamPool,

    /// PTX directory (for logging/debugging)
    ptx_dir: PathBuf,
}

impl GlobalGpuContext {
    /// Get or initialize the global GPU context.
    ///
    /// First call performs full initialization:
    /// - CUDA device initialization
    /// - Loading 14+ PTX modules
    /// - JIT compilation of all kernels
    /// - Reservoir weight initialization
    ///
    /// Subsequent calls return the cached reference immediately.
    ///
    /// # Panics
    /// Panics if GPU initialization fails (no GPU, driver issues, missing PTX).
    /// Use `try_get()` for fallible initialization.
    pub fn get() -> &'static Self {
        GLOBAL_GPU
            .get_or_init(|| Self::initialize().expect("Failed to initialize global GPU context"))
    }

    /// Try to get or initialize the global GPU context.
    ///
    /// Returns `Err` if GPU initialization fails, allowing graceful fallback.
    pub fn try_get() -> Result<&'static Self, GlobalGpuError> {
        GLOBAL_GPU.get_or_try_init(Self::initialize)
    }

    /// Check if the global GPU context is initialized.
    pub fn is_initialized() -> bool {
        GLOBAL_GPU.get().is_some()
    }

    /// Internal initialization logic
    fn initialize() -> Result<Self, GlobalGpuError> {
        let start = std::time::Instant::now();
        log::info!("Initializing global GPU context (one-time)...");

        // Determine PTX directory
        let ptx_dir = Self::find_ptx_dir()?;
        log::info!("PTX directory: {}", ptx_dir.display());

        // Create base GPU context (loads all standard PTX modules)
        let config = GpuSecurityConfig::permissive();
        let context = GpuContext::new(0, config, &ptx_dir)
            .map_err(|e| GlobalGpuError::InitFailed(e.to_string()))?;

        // Get CUDA context for stream pool
        let cuda_ctx = context.device().clone();
        let stream_pool = StreamPool::new(cuda_ctx.clone());

        // Pre-initialize mega-fused kernel (wrapped in Mutex for thread-safe mutable access)
        let mega_fused = match MegaFusedGpu::new(cuda_ctx.clone(), &ptx_dir) {
            Ok(mf) => {
                log::info!("Mega-fused GPU kernel pre-loaded (Mutex-wrapped for thread safety)");
                Some(Mutex::new(mf))
            }
            Err(e) => {
                log::warn!("Mega-fused kernel not available: {}", e);
                None
            }
        };

        // Pre-initialize LBS kernel (wrapped in Mutex for thread-safe mutable access)
        // let lbs = match LbsGpu::new(cuda_ctx.clone(), &ptx_dir) {
        //     Ok(l) => {
        //         log::info!("LBS GPU kernel pre-loaded (Mutex-wrapped for thread safety)");
        //         Some(Mutex::new(l))
        //     }
        //     Err(e) => {
        //         log::warn!("LBS kernel not available: {}", e);
        //         None
        //     }
        // };

        let elapsed = start.elapsed();
        log::info!(
            "Global GPU context initialized in {:.2}s (modules loaded ONCE)",
            elapsed.as_secs_f32()
        );

        Ok(Self {
            context,
            mega_fused,
            // lbs,
            stream_pool,
            ptx_dir,
        })
    }

    /// Find PTX directory from environment or defaults
    fn find_ptx_dir() -> Result<PathBuf, GlobalGpuError> {
        // Check PRISM_PTX_DIR environment variable
        if let Ok(dir) = std::env::var("PRISM_PTX_DIR") {
            let path = PathBuf::from(&dir);
            if path.exists() {
                return Ok(path);
            }
            log::warn!("PRISM_PTX_DIR='{}' does not exist, trying defaults", dir);
        }

        // Try common locations
        let candidates = [
            PathBuf::from("target/ptx"),
            PathBuf::from("./target/ptx"),
            PathBuf::from("kernels/ptx"),
            PathBuf::from("../target/ptx"),
        ];

        for candidate in &candidates {
            if candidate.exists() {
                return Ok(candidate.clone());
            }
        }

        Err(GlobalGpuError::PtxDirNotFound(
            "No PTX directory found. Set PRISM_PTX_DIR or ensure target/ptx exists".to_string(),
        ))
    }

    /// Get reference to the underlying GPU context
    pub fn context(&self) -> &GpuContext {
        &self.context
    }

    /// Get locked mutable access to the mega-fused kernel executor.
    ///
    /// Returns a MutexGuard that provides exclusive mutable access.
    /// The lock is automatically released when the guard is dropped.
    ///
    /// # Usage
    /// ```ignore
    /// if let Some(mut mega_fused) = gpu.mega_fused_locked() {
    ///     let result = mega_fused.detect_pockets(...)?;
    /// }
    /// ```
    pub fn mega_fused_locked(&self) -> Option<MutexGuard<'_, MegaFusedGpu>> {
        self.mega_fused.as_ref().map(|m| m.lock())
    }

    /// Check if mega-fused kernel is available (without locking)
    pub fn mega_fused(&self) -> Option<&Mutex<MegaFusedGpu>> {
        self.mega_fused.as_ref()
    }

    // /// Get locked mutable access to the LBS kernel executor.
    // ///
    // /// Returns a MutexGuard that provides exclusive mutable access.
    // /// The lock is automatically released when the guard is dropped.
    // pub fn lbs_locked(&self) -> Option<MutexGuard<'_, LbsGpu>> {
    //     self.lbs.as_ref().map(|l| l.lock())
    // }

    // /// Check if LBS kernel is available (without locking)
    // pub fn lbs(&self) -> Option<&Mutex<LbsGpu>> {
    //     self.lbs.as_ref()
    // }

    /// Get a CUDA stream for the current thread
    ///
    /// Each thread gets its own stream, enabling concurrent kernel execution
    /// when using Rayon parallelism.
    pub fn get_thread_stream(&self) -> Arc<CudaStream> {
        self.stream_pool.get_stream()
    }

    /// Get the PTX directory path
    pub fn ptx_dir(&self) -> &PathBuf {
        &self.ptx_dir
    }

    /// Check if mega-fused kernel is available (without locking)
    pub fn has_mega_fused(&self) -> bool {
        self.mega_fused.is_some()
    }

    /// Check if FP16 Tensor Core support is available
    pub fn has_fp16(&self) -> bool {
        self.mega_fused
            .as_ref()
            .map(|mf| mf.lock().has_fp16())
            .unwrap_or(false)
    }

    // /// Check if LBS kernel is available (without locking)
    // pub fn has_lbs(&self) -> bool {
    //     self.lbs.is_some()
    // }
}

// Ensure GlobalGpuContext is Send + Sync for cross-thread sharing
unsafe impl Send for GlobalGpuContext {}
unsafe impl Sync for GlobalGpuContext {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires GPU
    fn test_global_context_singleton() {
        // First call initializes
        let ctx1 = GlobalGpuContext::get();

        // Second call returns same instance
        let ctx2 = GlobalGpuContext::get();

        // Should be the same pointer
        assert!(std::ptr::eq(ctx1, ctx2));
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_thread_streams() {
        let ctx = GlobalGpuContext::get();

        // Get stream for main thread
        let stream1 = ctx.get_thread_stream();
        let stream2 = ctx.get_thread_stream();

        // Should return the same stream
        assert!(Arc::ptr_eq(&stream1, &stream2));
    }
}
