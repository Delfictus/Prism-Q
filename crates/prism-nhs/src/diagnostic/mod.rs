//! Diagnostic-only modules — opt-in via the `diagnostic` Cargo feature.
//!
//! Per the Blackwell Convergence mandate (operator directive
//! 2026-04-29) the M1.2.5b DBSCAN-alignment differential workstream
//! is no longer a closure gate for M1. The artifacts here survive
//! as an INFORMATIONAL diagnostic surface only — they are excluded
//! from default builds via the `#[cfg(feature = "diagnostic")]`
//! gate in [`crate::lib`] and are intentionally non-load-bearing
//! for any production code path.
//!
//! ## What lives here
//!
//! * [`m1_differential`] — typed-producer-vs-legacy differential
//!   protocol implementation (data structures, agreement-class
//!   classifier, run-end rollup). Reference spec:
//!   `docs/M1_DIFFERENTIAL_PROTOCOL.md` v1.0.0. The protocol's
//!   integer-mismatch BlockingDivergence has been determined
//!   STRUCTURAL between M1's voxel attribution and legacy
//!   DBSCAN-with-min_spikes — not a forensic signal of a producer
//!   bug. Retained for post-mortem analysis runs that opt into
//!   the diagnostic feature.
//!
//! ## What replaced it as a closure gate
//!
//! M1 closure now gates on:
//!
//! * **G1 — No Placeholder Lag**: the engine must emit a non-zero,
//!   COMPUTED `causal_lag` derived from Interferometric
//!   Bisimulation across phases. Placeholder NaN → halt.
//! * **G7 — LBVH-Ready Extents**: every per-site
//!   [`crate::site_manifest::CentroidManifold`] must carry an LBVH-
//!   derived AABB on the geometric / lining / driver slots. The
//!   [`crate::lbvh`] lane (Morton encoder landed; Karras tree +
//!   AABB reduce in the next commits) is the only path that
//!   produces these honestly.
//!
//! ## Building with diagnostics enabled
//!
//! ```text
//! cargo build --release -p prism-nhs --features diagnostic
//! ```
//!
//! Default builds exclude this module; the bin's
//! `--m1-typed-producer` flag becomes a no-op runtime field with
//! no compiled call site.

pub mod m1_differential;
