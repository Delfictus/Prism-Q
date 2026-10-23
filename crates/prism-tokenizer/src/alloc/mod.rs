//! §8 — Stage 3: information-proportional bit allocation. Resolution ∝ measured information
//! (reverse water-filling), confidence-gated coarsening, decision-critical fields pinned.

pub mod rate_distortion;

pub use rate_distortion::{
    allocate, confidence_gate, gated_bits, measure_corpus, measure_field, Allocation,
    AllocationPlan, FieldInfo, B_MIN_DEFAULT, G_BIMODAL_CAP,
};
