//! Domain-specific adapters and utilities
//!
//! This module contains domain-specific implementations for:
//! - Protein structure parsing and analysis
//! - Biomolecular drug discovery workflows
//! - Materials discovery and property prediction
//! - GNN-based graph analysis

pub mod biomolecular;
pub mod gnn;
pub mod materials;
pub mod protein;

// Re-export key types for convenience
pub use biomolecular::{BiomolecularAdapter, BiomolecularConfig, BiomolecularState};
pub use gnn::GnnState;
pub use materials::{MaterialsAdapter, MaterialsConfig, MaterialsState, TargetProperties};
pub use protein::{Atom, ProteinContactGraph, Residue};
