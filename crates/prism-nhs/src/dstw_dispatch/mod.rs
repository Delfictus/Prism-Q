//! Option A — Thermodynamic Frustration Projection Dispatcher.
//!
//! Implements the dispatcher specified in
//! `crates/prism-nhs/spec/option_a_variant_dispatcher.md` (operator-authorised
//! 2026-05-20).  Reads a `PRISMExecutionRequest`, runs rigid-backbone Δq / ΔV
//! projections of each variant against the WT thermodynamic tensors, and
//! emits a `PRISMExecutionResponse` carrying the vectorial
//! `[delta_P_active, delta_P_lock, delta_P_ensemble]` per variant plus
//! epistemic uncertainty sigmas.
//!
//! This is the CPU reference implementation.  CUDA kernels for the
//! propagation matrices `K_active`, `K_lock`, `K_ensemble` are deferred
//! to a future gate; the CPU path is correctness-anchored and
//! deterministic-by-construction so DSTW's BALD acquisition can close the
//! loop end-to-end without GPU dependency.
//!
//! Hard invariants:
//!
//!   * **Vectorial-only inputs.**  `requested_channels` must equal
//!     `{delta_P_active, delta_P_lock, delta_P_ensemble}`.  Scalar
//!     Wasserstein / variant-divergence inputs are refused at schema
//!     validation time.
//!
//!   * **No NaN / Inf escape.**  Every delta and sigma is checked for
//!     finiteness before being placed into a `VariantExecutionResponse`.
//!     Non-finite values short-circuit to a `DispatchError::NonFinite`.
//!
//!   * **Strict sigma inflation on non-convergence.**  Operator directive:
//!     "the epistemic uncertainty budget (σ_epistemic) must strictly
//!     inflate for any variant point that fails convergence during the
//!     ΔV substitution, ensuring downstream EiV down-weighting."  We
//!     enforce this by multiplying every channel's sigma by
//!     `config.nonconverged_sigma_penalty` (default 4.0) whenever the
//!     variant's `converged` flag is false.  The penalty is asserted
//!     `> 1.0` at config construction time so the inflation cannot be
//!     accidentally configured into a contraction.
//!
//!   * **Determinism.**  All RNG (used only for posterior-noise
//!     sampling, currently OFF by default) is seeded from
//!     `blake3(canonical_json(request))`.  No global RNG.

pub mod cuda;
pub mod dtsg_parquet_emitter;
pub mod edge_attenuator;
pub mod handshake;
pub mod projection;
pub mod sidechain_tables;
pub mod types;

mod dispatcher;

pub use cuda::{
    propagate_cpu_reference, row_sums_cpu_reference, Backend, CudaPropagationKernel,
    PropagationShape, CH_ACTIVE, CH_ENSEMBLE, CH_LOCK, N_CHANNELS,
};
pub use dispatcher::{
    dispatch_variant_batch, dispatch_variant_batch_with_topology, DispatchError,
    VariantDispatchConfig,
};
pub use dtsg_parquet_emitter::{
    sidecar_path as chem_perturbed_dtsg_sidecar_path, write_chem_perturbed_dtsg_parquet,
    ChemPerturbedDtsgSidecar, CHEM_PERTURBED_DTSG_SCHEMA_TAG,
};
pub use edge_attenuator::{
    attenuate_edges, AttenuationError, EdgePenalty, PerturbedEdgeStats, ScalingConstants,
};
pub use handshake::{
    PRISMExecutionAcquisition, PRISMExecutionRequest, PRISMExecutionResponse,
    VariantExecutionRequest, VariantExecutionResponse, REQUIRED_DELTA_CHANNELS,
    RESPONSE_SCHEMA_TAG,
};
pub use projection::{
    project_variant, project_variant_with_topology, EpistemicSigmas, ProjectedDeltas,
    TopologyProjectionContext, TopologyStateEnvironment, VariantPoint, WTTensorPack,
};
pub use sidechain_tables::{AminoAcid, SidechainDescriptor};
pub use types::{
    AnalogIdx, AngstromDistance, CampaignId, CausalCoupling, ChannelCapacity, ComplementPenalty,
    ConformerIdx, DTSGEdge, DTSGEdgeMetrics, EdgeIdx, FrustrationPenalty, HitCount,
    HydrationVariance, HysteresisCapacity, HysteresisEnthalpy, HysteresisPersistence,
    PoseUncertainty, ResidueIdx, ScaffoldIdx, ScalingConstant, SpatialVariance, VoxelIdx,
};
