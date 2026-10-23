//! Compiler contract marker.
//!
//! The production compiler is Python today because the active PRISM OS registry
//! and source layers are Python/Arrow native. This Rust module defines the
//! boundary expected by future Rust/CUDA readers without becoming a second ISA.

pub const COMPILER_OWNER: &str = "prism_os.prismcode_layer";
pub const MACHINE_FORMS: &[&str] = &["Arrow IPC", "Parquet"];
