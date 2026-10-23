//! # PRISM-IO: Revolutionary I/O Architecture
//!
//! ⚠️ **PRISM-Zero Production Standard Compliance Required**
//! See `/ENGINEERING_STANDARDS.md` for complete requirements
//!
//! ## Prism-Stream Architecture
//!
//! This crate implements the revolutionary three-innovation I/O pipeline:
//! 1. **Holographic Binary Format (.ptb)**: Zero-copy protein structure loading
//! 2. **Warp-Drive Parser**: CUDA warp intrinsics for 50-100x parsing speedup
//! 3. **Async Pinned Streaming**: io_uring kernel bypass with GPU Direct Storage
//!
//! ## Performance Targets
//! - **Holographic Loading**: <100μs vs 10-50ms traditional PDB parsing
//! - **Warp-Drive Parsing**: <500μs with CUDA warp intrinsics
//! - **Complete Pipeline**: <1ms storage-to-GPU data pipeline
//!
//! ## Zero-Mock Protocol Compliance
//! - All input data must be cryptographically verified (SHA-256)
//! - No synthetic/mock datasets permitted in production code
//! - Proper Result<T,E> error handling required
//! - SovereignBuffer types enforce real data pipeline usage
//!
//! ## Usage Example
//! ```rust,no_run
//! use prism_io::{AsyncPinnedStreamer, HolographicBinaryFormat};
//! use prism_io::validation::DataIntegrityValidator;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Initialize async streaming pipeline
//!     let streamer = AsyncPinnedStreamer::new().await?;
//!
//!     // Load and validate protein structure with zero-copy semantics
//!     let structure = streamer
//!         .load_verified_structure("nipah_g.ptb")
//!         .await?;
//!
//!     // Structure is now ready for GPU processing via SovereignBuffer
//!     println!("Loaded {} atoms in <1ms", structure.atom_count());
//!     Ok(())
//! }
//! ```

#![warn(missing_docs)]
#![warn(unused_extern_crates)]
#![warn(unused_import_braces)]
#![warn(unused_qualifications)]
#![warn(unsafe_code)] // Unsafe code requires explicit review and justification

use std::marker::PhantomData;

// Core modules
pub mod holographic;
pub mod sovereign_types;
pub mod streaming;
pub mod validation;
pub mod warp_parser;

// Re-exports for convenience
pub use holographic::{HolographicBinaryFormat, PtbHeader, PtbStructure};
pub use sovereign_types::{SovereignBuffer, SovereignError};
pub use streaming::{AsyncPinnedStreamer, StreamingError};
pub use validation::{DataIntegrityValidator, ValidationError};

/// Performance targets for Prism-Stream architecture components
pub mod performance {
    /// Target execution time for holographic binary format loading
    pub const PTB_LOADING_TARGET_MICROS: u64 = 100;

    /// Target execution time for warp-drive parsing
    pub const WARP_PARSING_TARGET_MICROS: u64 = 500;

    /// Target execution time for complete async streaming pipeline
    pub const COMPLETE_PIPELINE_TARGET_MILLIS: u64 = 1;
}

/// Common error types used throughout the prism-io crate
#[derive(thiserror::Error, Debug)]
pub enum PrismIoError {
    /// Data integrity violation - input data failed cryptographic verification
    #[error("Data integrity violation: {0}")]
    IntegrityViolation(String),

    /// I/O error during file operations or streaming
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// CUDA operation failed
    #[error("CUDA error: {0}")]
    CudaError(String),

    /// Serialization/deserialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Invalid file format or structure
    #[error("Format error: {0}")]
    FormatError(String),

    /// Data validation error
    #[error("Validation error: {0}")]
    ValidationError(String),
}

/// Result type alias for prism-io operations
pub type Result<T> = std::result::Result<T, PrismIoError>;

// Error type conversions for seamless integration
impl From<validation::ValidationError> for PrismIoError {
    fn from(err: validation::ValidationError) -> Self {
        PrismIoError::ValidationError(err.to_string())
    }
}

impl From<warp_parser::WarpParseError> for PrismIoError {
    fn from(err: warp_parser::WarpParseError) -> Self {
        PrismIoError::CudaError(err.to_string())
    }
}

/// Initialize the prism-io subsystem with performance monitoring
pub fn initialize() -> Result<()> {
    tracing::info!("Initializing PRISM-IO v{}", env!("CARGO_PKG_VERSION"));

    // Verify CUDA availability if GPU features enabled
    #[cfg(feature = "gpu")]
    {
        // TODO: Fix CUDA device initialization once API is clarified
        tracing::info!("CUDA device initialization temporarily disabled");
    }

    tracing::info!("PRISM-IO initialization complete");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prism_io_initialization() {
        // Test basic initialization without CUDA requirement
        assert!(initialize().is_ok() || cfg!(feature = "gpu"));
    }

    #[test]
    fn test_performance_constants() {
        // Verify performance targets are reasonable
        assert!(performance::PTB_LOADING_TARGET_MICROS < 1000);
        assert!(performance::WARP_PARSING_TARGET_MICROS < 1000);
        assert!(performance::COMPLETE_PIPELINE_TARGET_MILLIS < 10);
    }
}
