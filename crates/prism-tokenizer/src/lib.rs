//! # prism-tokenizer
//!
//! A **structured-artifact modality tokenizer**: it converts PRISM's serialized, sealed,
//! non-equilibrium artifact streams into information-dense typed tokens by *inverting the
//! serialization stack* (decode-before-tokenize) and quantizing recovered fields in
//! geometries matched to their physics.
//!
//! This is NOT a replacement for the Qwen3-8B-Base text tokenizer. Free-form reasoning +
//! constrained-JSON stay on the text path; this produces a compact typed stream for the
//! artifact modalities that is *fused* into the model (§11).
//!
//! ## Hard invariants (rejection criteria, not guidance)
//! - **INV-1 — Decode > claim.** Never tokenize sealed/compressed bytes. Identity is the
//!   *decoded* record; a filename/magic/self-declared field is metadata, not evidence.
//! - **INV-2 — Evidence ≠ aspiration.** Token identity carries the artifact's epistemic
//!   namespace; the same numeric value in two families is two distinct tokens.
//! - **INV-3 — Provisions ≠ promotes.** Tokens from unverified/failed-seal artifacts carry
//!   `verified=0` and are quarantined; they must never reach a canonical path.
//! - **INV-4 — Lossless ground truth.** The sidecar holds exact decoded values + seal sha;
//!   the token stream is a lossy *view*, the sidecar is truth.
//! - **INV-5 — Receipt-gate honesty.** `RECEIPT_STATUS` is a closed 3-way {PASS,PARTIAL,FAIL};
//!   there is no token for an inferred PASS.
//! - **INV-6 — Integrity ≠ authenticity.** Seal verification proves bytes were not altered,
//!   not who produced them; label it `integrity_verified`, never `authentic`.
//!
//! ## Build status
//! M0 in progress: the §4 type system is the first verifiable foundation. Codec inversion
//! (Stage 0) lands once the real engine formats are grounded — no byte reaches a quantizer
//! before it is decoded and seal-checked (INV-1).

pub mod alloc;
pub mod backend;
pub mod bench;
pub mod codec;
pub mod encoder;
pub mod epistemic;
#[cfg(feature = "gpu-tokenizer")]
pub mod gpu;
pub mod quant;
pub mod routing;
pub mod schema;
pub mod sidecar;
pub mod stream;
pub mod types;

pub use alloc::{
    allocate, confidence_gate, gated_bits, measure_corpus, measure_field, Allocation,
    AllocationPlan, FieldInfo,
};
pub use routing::{
    build_binding_map, build_therm_map, detect_family, detect_family_from_filename_hint,
    route_artifact, route_artifact_autofit, route_kcc_sites, route_kcc_sites_autofit,
    route_kcc_sites_with_binding, route_kcc_sites_with_binding_and_therm,
    route_kcc_sites_with_therm, route_object, route_records, route_scientific_surface_event_record,
    route_scientific_surface_event_records, route_spike_transition_event, route_twin16_residual,
    route_twin_per_residue, RouteReport, SiteRouteReport, ThermEvidence,
};

pub use backend::{CrossoverThresholds, TokenizerBackend, WorkloadShape};
pub use encoder::{Encoder, VqHealth};
pub use epistemic::{
    canonical_eligible, canonical_eligible_strict, leverage_tier, GENERATIVE_TIER,
};
pub use schema::{FieldRegistry, SCHEMA_VERSION};
pub use sidecar::{Sidecar, SidecarEntry};
pub use stream::TokenStream;
pub use types::{DecisionSpec, DistortionBound};

pub use codec::{
    decode_prspk001, decode_prspk001_head, decode_qnanl, decode_spike_packet, CodecError,
    GpuSpikeEvent, Prspk001Decoded, QnanlDecoded, SpikePacketDecoded,
};
pub use quant::{
    build_field_quantizer, build_scalar_quantizer, Bcc3dQuantizer, BooleanQuantizer,
    CircularQuantizer, CriticalRegionQuantizer, EnumQuantizer, JoinKeyQuantizer, LinearQuantizer,
    LogMagnitudeQuantizer, SentinelQuantizer, SmilesVocab, SpectralLineQuantizer,
};
pub use types::{
    pack_namespace, unpack_namespace, ArtifactFamily, CircPeriod, Distortion, FieldGeometry,
    FieldSpec, InfoSource, Lattice, Quantizer, Token,
};
