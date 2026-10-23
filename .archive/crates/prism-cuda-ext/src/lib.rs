//! PRISM-TWIN Advanced CUDA Extensions
//!
//! This crate provides bleeding-edge CUDA 13.x features for PRISM-TWIN:
//!
//! - **Device-side spike compaction**: CUB-based prefix sum compaction
//!   that eliminates the CPU round-trip for spike download/repack/upload.
//!   Reads d_spike_events (92B) and d_spike_count directly on GPU,
//!   compacts to RingSpikeEvent (48B), pushes to ring buffer.
//!
//! - **Conditional CUDA Graphs**: Build a graph with a WHILE loop
//!   that runs the entire coupling sequence autonomously on the GPU.
//!   One cudaGraphLaunch for the entire simulation.
//!
//! - **Device-updated graph parameters**: The GPU reads d_spike_count
//!   from VRAM and dynamically sizes the compaction kernel's grid.
//!   No host involvement for grid sizing.
//!
//! These features require:
//! - CUDA 12.4+ (conditional graph nodes)
//! - SM 120 (Blackwell, RTX 5080)
//! - cudarc 0.18.2 (sys-level bindings for the driver API)

pub mod capture;
pub mod compact;
pub mod coupling_graph;
pub mod exhaust;
pub mod graph;
pub mod graph_builder;
