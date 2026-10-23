//! # prism-core
//!
//! Core types, traits, and errors for the PRISM graph coloring platform.
//!
//! This crate defines the fundamental abstractions used across all PRISM components:
//! - **Types**: Graph representations, coloring solutions, configurations
//! - **Traits**: PhaseController, PhaseTelemetry, PhaseExecutor
//! - **Errors**: Unified error handling with PrismError
//!
//! ## Architecture
//!
//! PRISM v2 follows a modular, trait-based architecture:
//! ```text
//! ┌─────────────────┐
//! │  prism-core     │  ← Core types/traits
//! └─────────────────┘
//!         ▲
//!         │
//!    ┌────┴──────────────────┐
//!    │                       │
//! ┌──▼──────────┐   ┌───────▼──────┐
//! │ prism-gpu   │   │ prism-fluxnet│
//! └─────────────┘   └──────────────┘
//!         ▲                 ▲
//!         └────────┬────────┘
//!                  │
//!         ┌────────▼────────┐
//!         │  prism-phases   │
//!         └─────────────────┘
//! ```
//!
//! Implements the PRISM GPU Plan (§2: Core Types & Traits).

pub mod dag;
pub mod dimacs;
pub mod domain;
pub mod errors;
pub mod runtime_config;
pub mod traits;
pub mod types;

// PRISM-Zero Flight Recorder
pub mod telemetry;

// Re-export commonly used items
pub use dag::{
    stable_dag_node_id, DagArtifactRef, DagEdge, DagEdgeKind, DagInvariant, DagNode, DagNodeKind,
    DagRunManifest, DAG_SCHEMA_VERSION,
};
pub use errors::PrismError;
pub use runtime_config::{KernelTelemetry, RuntimeConfig};
pub use traits::{
    PhaseContext, PhaseController, PhaseOutcome, PhaseTelemetry, WarmstartContributor,
};
pub use types::{
    CmaState, ColoringSolution, GeometryTelemetry, Graph, GraphStats, Phase0Telemetry,
    Phase3Config, PhaseConfig, VertexId, WarmstartConfig, WarmstartMetadata, WarmstartPlan,
    WarmstartPrior, WarmstartTelemetry,
};
