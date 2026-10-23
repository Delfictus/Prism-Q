//! Pure (GTK-free) parsers for the PrismOS T4 evidence surfaces.
//!
//! These modules deserialize the durable evidence objects the desktop app
//! renders. They are deliberately tolerant: every field that the runtime may
//! omit is `Option`, and unknown records are skipped rather than fatal, because
//! the evidence schema evolves and the UI must degrade gracefully instead of
//! refusing to open a run.

pub mod corpus;
pub mod run_layout;
pub mod state_graph;
pub mod status;
