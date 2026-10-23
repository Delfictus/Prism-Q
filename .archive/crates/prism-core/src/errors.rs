//! Error types for PRISM.
//!
//! Implements PRISM GPU Plan §2.3: Error Handling.

use thiserror::Error;

/// Unified error type for all PRISM operations.
///
/// Provides structured, actionable error messages with context.
#[derive(Error, Debug)]
pub enum PrismError {
    /// GPU-related errors (CUDA, PTX loading, kernel execution)
    #[error("GPU error in {context}: {message}")]
    GpuError { context: String, message: String },

    /// Configuration validation errors
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Phase execution errors
    #[error("Phase '{phase}' failed: {message}")]
    PhaseError { phase: String, message: String },

    /// Input validation errors
    #[error("Validation error: {0}")]
    ValidationError(String),

    /// GNN inference errors
    #[error("GNN error: {0}")]
    GnnError(String),

    /// I/O errors (file reading, PTX loading, telemetry writing)
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Serialization/deserialization errors
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// Mathematical/numerical errors (e.g., NaN, overflow)
    #[error("Numerical error: {0}")]
    NumericalError(String),

    /// Resource exhaustion (e.g., out of GPU memory)
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),

    /// Timeout errors
    #[error("Operation timed out: {0}")]
    Timeout(String),

    /// Generic errors (fallback)
    #[error("Internal error: {0}")]
    Internal(String),
}

impl PrismError {
    /// Creates a GPU error with context.
    pub fn gpu(context: impl Into<String>, message: impl Into<String>) -> Self {
        PrismError::GpuError {
            context: context.into(),
            message: message.into(),
        }
    }

    /// Creates a configuration error.
    pub fn config(message: impl Into<String>) -> Self {
        PrismError::ConfigError(message.into())
    }

    /// Creates a phase error.
    pub fn phase(phase: impl Into<String>, message: impl Into<String>) -> Self {
        PrismError::PhaseError {
            phase: phase.into(),
            message: message.into(),
        }
    }

    /// Creates a validation error.
    pub fn validation(message: impl Into<String>) -> Self {
        PrismError::ValidationError(message.into())
    }

    /// Creates a numerical error.
    pub fn numerical(message: impl Into<String>) -> Self {
        PrismError::NumericalError(message.into())
    }

    /// Creates a resource exhausted error.
    pub fn resource_exhausted(message: impl Into<String>) -> Self {
        PrismError::ResourceExhausted(message.into())
    }

    /// Creates a timeout error.
    pub fn timeout(message: impl Into<String>) -> Self {
        PrismError::Timeout(message.into())
    }

    /// Creates an internal error.
    pub fn internal(message: impl Into<String>) -> Self {
        PrismError::Internal(message.into())
    }

    /// Creates a GNN error.
    pub fn gnn(message: impl Into<String>) -> Self {
        PrismError::GnnError(message.into())
    }

    /// Checks if this is a retriable error.
    ///
    /// Retriable errors should trigger a `PhaseOutcome::Retry`.
    pub fn is_retriable(&self) -> bool {
        matches!(
            self,
            PrismError::Timeout(_) | PrismError::ResourceExhausted(_)
        )
    }

    /// Returns a user-friendly error message with actionable guidance.
    pub fn user_message(&self) -> String {
        match self {
            PrismError::GpuError { context, message } => {
                format!(
                    "GPU error in {}: {}\n\
                     → Check CUDA installation and GPU availability.\n\
                     → Ensure PTX files are compiled for the correct architecture (sm_86).",
                    context, message
                )
            }
            PrismError::ConfigError(msg) => {
                format!(
                    "Configuration error: {}\n\
                     → Review your configuration file and ensure all required fields are set.\n\
                     → Check that numeric values are within valid ranges.",
                    msg
                )
            }
            PrismError::ValidationError(msg) => {
                format!(
                    "Validation error: {}\n\
                     → Verify input graph format and constraints (MAX_VERTICES = 10000).",
                    msg
                )
            }
            PrismError::ResourceExhausted(msg) => {
                format!(
                    "Resource exhausted: {}\n\
                     → Reduce graph size or batch size.\n\
                     → Check available GPU memory with `nvidia-smi`.",
                    msg
                )
            }
            _ => self.to_string(),
        }
    }
}

/// Result type alias for PRISM operations.
pub type Result<T> = std::result::Result<T, PrismError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_constructors() {
        let gpu_err = PrismError::gpu("kernel launch", "CUDA_ERROR_OUT_OF_MEMORY");
        assert!(matches!(gpu_err, PrismError::GpuError { .. }));

        let config_err = PrismError::config("Invalid max_vertices");
        assert!(matches!(config_err, PrismError::ConfigError(_)));

        let phase_err = PrismError::phase("Phase2", "Convergence failed");
        assert!(matches!(phase_err, PrismError::PhaseError { .. }));
    }

    #[test]
    fn test_retriable_errors() {
        let timeout = PrismError::timeout("Phase took too long");
        assert!(timeout.is_retriable());

        let resource = PrismError::resource_exhausted("Out of GPU memory");
        assert!(resource.is_retriable());

        let validation = PrismError::validation("Invalid graph");
        assert!(!validation.is_retriable());
    }
}
