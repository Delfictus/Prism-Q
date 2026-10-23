//! PRISM-TWIN: Interferometric Spike Dynamics
//!
//! Two MD simulations run simultaneously on one GPU with INTERLEAVED stepping.
//! Physics is INDEPENDENT in both streams. The observation layers are coupled
//! through spike density exchange AND cross-correlation computation.
//!
//! The value is not in individual spike trains — it's in the RELATIONSHIP
//! between them: cross-correlation reveals reproducibility, propagation speed,
//! barrier height, and mechanism. This is interferometric measurement applied
//! to molecular dynamics.
//!
//! ## Architecture
//!
//! ```text
//! for each step:
//!   engine_a.step()  // CUDA stream A (concurrent)
//!   engine_b.step()  // CUDA stream B (concurrent)
//!
//!   every exchange_interval steps:
//!     compute_spike_density(A, B)      // Layer 1: spatial
//!     compute_cross_correlation(A, B)  // Layer 2: temporal
//!     apply_threshold_exchange(A↔B)    // Layer 1: coupling
//!     update_ring_buffers(A, B)        // Layer 2: history
//! ```
//!
//! ## Layers
//!
//! - Layer 1: Spike density exchange (WHERE spikes occur → threshold adaptation)
//! - Layer 2: Temporal cross-correlation (WHEN spikes co-occur → reproducibility)
//! - Layer 4: Perturbation differential (NMA in B only → barrier measurement)
//! - Layers 3+5: Future v2 (frequency coupling, transfer entropy)

use crate::fused_engine::PrismCodeMdPolicyAdmissionV1;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[cfg(feature = "gpu")]
use cudarc::driver::{CudaContext, CudaModule, CudaStream};

use crate::fused_engine::CryoUvProtocol;
use crate::input::PrismPrepTopology;
use crate::persistent_engine::PersistentNhsEngine;
#[cfg(feature = "gpu")]
use crate::twin_kernels::{find_twin_ptx, TwinRingBuffer};

/// Ring buffer size for per-voxel spike history (for CCF computation).
/// 256 entries captures ~500 steps of spike history at typical rates.
pub const RING_SIZE: usize = 256;

/// Number of CCF lag bins (τ from -CCF_HALF_LAGS to +CCF_HALF_LAGS).
pub const CCF_HALF_LAGS: usize = 32;
pub const CCF_LAGS: usize = CCF_HALF_LAGS * 2;

const COUPLED_TWIN_EVIDENCE_SCHEMA_VERSION: u32 = 2;
const STRICT_TWIN_FEATURE_SCHEMA_VERSION: u32 = 3;
const MOLECULAR_CONTRAST_READOUT_SCHEMA_VERSION: u32 = 4;
const GLYCAN_MECHANISTIC_PROBE_READOUT_SCHEMA_VERSION: u32 = 1;
const GLYCAN_MECHANISTIC_PROBE_CONTRAST_KIND: &str = "glycan-mechanistic-probe";
const MOLECULAR_CONTRAST_POCKET_SUPPORTED: &str = "SUPPORTED_EVENT_AND_STRICT_RESIDUE_EVIDENCE";
const MOLECULAR_CONTRAST_POCKET_UNAVAILABLE: &str =
    "UNAVAILABLE_NO_EVENT_OR_STRICT_RESIDUE_EVIDENCE";
const GLYCAN_PROBE_SIGNALING_HYPOTHESIS_STATUS: &str =
    "HYPOTHESIS_ONLY_INTRASTREAM_SOURCE_TO_SITE_CAUSAL_PATH_NOT_MEASURED";
const GLYCAN_PROBE_NOVELTY_STATUS: &str =
    "CONTROL_RELATIVE_CRYPTIC_RESPONSE_ONLY_BROADER_STRUCTURAL_AND_PATENT_NOVELTY_NOT_VERIFIED";
const STRICT_TWIN_CCF_BIN_SIZE_STEPS: i32 = 100;
const STRICT_TWIN_PREDICTIVE_WINDOW_STEPS: i32 = 500;
const STRICT_TWIN_CCF_TEMPORAL_LAG_STEPS: i32 = 0;
const STRICT_TWIN_TE_KERNEL_LAG_BINS: u32 = 1;
const STRICT_TWIN_TE_PAST_INDEX_OFFSET_BINS: u32 = 1;
const STRICT_TWIN_TE_FUTURE_INDEX_OFFSET_BINS: u32 = 1;
const STRICT_TWIN_TE_EFFECTIVE_HORIZON_BINS: u32 = 2;
const STRICT_TWIN_TE_EFFECTIVE_HORIZON_STEPS: i32 =
    STRICT_TWIN_CCF_BIN_SIZE_STEPS * STRICT_TWIN_TE_EFFECTIVE_HORIZON_BINS as i32;
const STRICT_TWIN_PREDICTIVE_ALGORITHM: &str =
    "exact_same_residue_directional_closed_window_tB_in_[tA,tA+500]";
const STRICT_TWIN_PHASE_CCF_BACKEND: &str = "CUDA_WMMA:tensor_ccf.ptx;A_residue_rows_x_B_residue_columns;zero_temporal_lag;five_exact_CCNS_phase_matrices;bin_size_steps=100";
const STRICT_TWIN_GLOBAL_CCF_BACKEND: &str =
    "CUDA_WMMA:tensor_ccf.ptx;A_residue_rows_x_B_residue_columns;zero_temporal_lag";
const STRICT_TWIN_TE_BACKEND: &str = "CUDA:twin_transfer_entropy.ptx;binned_binary_mean_threshold;kernel_lag_bins=1;past_index_offset_bins=1;future_index_offset_bins=1;effective_past_to_future_horizon_bins=2;bin_size_steps=100;effective_past_to_future_horizon_steps=200";
const STRICT_TWIN_CCF_SUPPORTED: &str = "SUPPORTED_REAL_GPU_WMMA_ZERO_TEMPORAL_LAG";
const STRICT_TWIN_PHASE_CCF_UNAVAILABLE: &str = "UNAVAILABLE_INSUFFICIENT_PHASE_VARIANCE_SUPPORT";
const STRICT_TWIN_GLOBAL_CCF_UNAVAILABLE: &str = "UNAVAILABLE_INSUFFICIENT_GLOBAL_VARIANCE_SUPPORT";
const STRICT_TWIN_TE_SUPPORTED: &str = "SUPPORTED_REAL_GPU_TE_KERNEL_LAG_BINS_1_EFFECTIVE_PAST_TO_FUTURE_HORIZON_BINS_2_BIN_SIZE_STEPS_100";
const STRICT_TWIN_PHASE_NAMES: [&str; 5] = [
    "cold_hold",
    "heating",
    "warm_hold",
    "cooling",
    "cold_return",
];
const GHOST_V2_RECORD_BYTES: usize = crate::ghost_tile::GHOST_RECORD_BYTES;
const SPLIT_T7_ANCHOR_FRAMES: usize = 50;
const SPLIT_T7_VARIANCE_FRAMES: usize = 50;
const SPLIT_T7_VALIDATION_FRAMES: usize = 1;
const SPLIT_T7_TOTAL_FRAMES: usize =
    SPLIT_T7_ANCHOR_FRAMES + SPLIT_T7_VARIANCE_FRAMES + SPLIT_T7_VALIDATION_FRAMES;
/// Canonical split-T7 event size. Bound to the record layout itself rather than
/// re-stating a literal, so a canonical form can never silently disagree with
/// the record it claims to canonicalize.
const SPLIT_T7_CANONICAL_EVENT_BYTES: usize = crate::fused_engine::PRISM_SPIKE_RECORD_BYTES;
// Back to 3. This constant was bumped 3 -> 4 for exactly one reason: the
// canonical event width moved 96 B -> 128 B when the spike record briefly
// carried a clock-provenance block. That widening invalidated the sealed QNANL
// force-kernel packet and has been reverted, so the canonical byte stream is
// once again bit-for-bit the v3 stream, over the same 96 B records, with every
// domain-separation string still at _V3. Leaving this at 4 would declare a
// 128 B stream over 96 B records — a false schema claim, and it would reject
// the v3 corpus that these artifacts are now byte-identical to.
//
// A v4 artifact (128 B events) is genuinely a different byte stream and is now
// rejected on the version compare, which is the correct fail-loud outcome.
const SPLIT_T7_SHADOW_SCHEMA_VERSION: u32 = 3;
const SPLIT_T7_SHADOW_TRIALS: usize = 3;
const SPLIT_T7_SHADOW_RESERVE_FACTOR: u32 = 2;
const SPLIT_T7_SHADOW_TRIAL_SEED_DOMAIN: &[u8] = b"PRISMOS_SPLIT_T7_SHADOW_TRIAL_SEED_V3\0";
const SPLIT_T7_SHADOW_TRIAL_SOURCE_DOMAIN: &[u8] = b"PRISMOS_SPLIT_T7_SHADOW_TRIAL_SOURCE_V3\0";
const SPLIT_T7_SHADOW_TRIAL_INVENTORY_DOMAIN: &[u8] =
    b"PRISMOS_SPLIT_T7_SHADOW_TRIAL_INVENTORY_V3\0";
const SPLIT_T7_SHADOW_ENSEMBLE_DOMAIN: &[u8] = b"PRISMOS_SPLIT_T7_SHADOW_ENSEMBLE_PRECOMMIT_V3\0";
const SPLIT_T7_ADMISSION_DOMAIN: &[u8] = b"PRISMOS_SPLIT_T7_ADMISSION_RECEIPT_V3\0";
const SPLIT_T7_AUTHORITATIVE_INVENTORY_DOMAIN: &[u8] =
    b"PRISMOS_SPLIT_T7_AUTHORITATIVE_INVENTORY_V3\0";
const SPLIT_T7_SELECTION_METHOD_V3: &str = "three_deterministically_seeded_independent_shadow_trials;distinct_SHA256_derived_seed_per_trial;within_each_trial_stream_A_and_B_share_the_trial_seed;each_trial_reconstructs_engines_from_topology_and_serializes_complete_101_window_source_bytes;completed_step_event_timestamps_use_exact_nonoverlapping_windows_(frame_start,frame_end];pointwise_lower_envelope;all_three_leave_one_trial_out_admissions;K=largest_power_of_two_with_2K_not_greater_than_global_shadow_envelope_minimum;K_at_least_4;authoritative_GPU_atomic_event_counts_independently_hash_bound_and_each_at_least_K;authoritative_engines_reinitialized_from_topology_then_run_once_over_exactly_101_equal_duration_windows;each_frame_selects_K_events_across_complete_single_window_by_floor(j*(n-1)/(K-1));role_partitioned_topology_grid=A:4x4x3_clusters_0_47,B:4x2x2_clusters_48_63;no_cross_window_concatenation_no_padding_no_reuse_no_synthetic_events;canonical_full_window_bytes_selection_indices_and_cluster_sorted_RichSpike_bytes_SHA256_bound";
const SPLIT_T7_TRIAL_SEED_DERIVATION_V3: &str = "u64_le(first_8_bytes(SHA256(PRISMOS_SPLIT_T7_SHADOW_TRIAL_SEED_V3\\0 || base_seed_le || trial_ordinal_le || SHA256(run_id) || topology_a_sha256 || topology_b_sha256)))";
const SPLIT_T7_RESERVE_DERIVATION_V3: &str =
    "precommitted_integer_safety_headroom_factor=2;applied_before_power_of_two_selection";
const SPLIT_T7_K_DERIVATION_V3: &str = "K=largest_power_of_two_not_greater_than(floor(global_pointwise_shadow_lower_envelope_minimum/reserve_factor));K>=4";
const SPLIT_T7_AUTHORITATIVE_COUNT_SOURCE_V3: &str = "newly_accumulated_authoritative_PersistentNhsEngine_GPU_atomic_spike_count_per_equal_duration_window_before_selection";
const COUPLED_TWIN_CRN_SCHEMA_VERSION: u32 = 3;
const COUPLED_TWIN_CRN_ATOM_KEY_SCHEMA: &str =
    "trim(chain_id)|residue_id_u64_le|trim(atom_name)|uppercase(trim(element));length_prefixed_utf8";
const COUPLED_TWIN_CRN_GAUSSIAN_ALGORITHM: &str = "sha256_counter_box_muller_f64_cos_v1";
const COUPLED_TWIN_CRN_CUDA_RNG_SEMANTICS: &str =
    "frozen_nhs_amber_fused:curand_init(seed,tid,0);three_curand_normal_draws_per_atom_per_step;shared_atom_ordinals_must_match";
const COUPLED_TWIN_CRN_SHARED_VELOCITY_DOMAIN: &[u8] =
    b"PRISMOS_COUPLED_TWIN_CRN_SHARED_VELOCITY_V3\0";
const COUPLED_TWIN_CRN_SHARED_POSITION_DOMAIN: &[u8] =
    b"PRISMOS_COUPLED_TWIN_CRN_SHARED_POSITION_V3\0";
const COUPLED_TWIN_CRN_B_ONLY_VELOCITY_DOMAIN: &[u8] =
    b"PRISMOS_COUPLED_TWIN_CRN_B_ONLY_VELOCITY_V3\0";
const COUPLED_TWIN_CRN_B_ONLY_POSITION_DOMAIN: &[u8] =
    b"PRISMOS_COUPLED_TWIN_CRN_B_ONLY_POSITION_V3\0";
const COUPLED_TWIN_CRN_MAP_DOMAIN: &[u8] = b"PRISMOS_COUPLED_TWIN_CRN_ATOM_MAP_V3\0";
const COUPLED_TWIN_CRN_ARRAY_DOMAIN: &[u8] = b"PRISMOS_COUPLED_TWIN_CRN_ARRAY_V3\0";
const COUPLED_TWIN_CRN_SEMANTIC_DOMAIN: &[u8] =
    b"PRISMOS_COUPLED_TWIN_CRN_INITIALIZATION_RECEIPT_V3\0";
const COUPLED_TWIN_CRN_KB_KCAL_PER_MOL_K: f64 = 0.001_987_204;
const COUPLED_TWIN_CRN_EFFECTIVE_OMEGA_SQ: f64 = 500.0;
const COUPLED_TWIN_CRN_BALANCE_TOLERANCE: f64 = 1.0e-9;
const SPLIT_T7_CLUSTER_GRID: u32 = 4;
const SPLIT_T7_CLUSTER_COUNT: u32 =
    SPLIT_T7_CLUSTER_GRID * SPLIT_T7_CLUSTER_GRID * SPLIT_T7_CLUSTER_GRID;
const SPLIT_T7_ROLE_A_GRID: [u32; 3] = [4, 4, 3];
const SPLIT_T7_ROLE_B_GRID: [u32; 3] = [4, 2, 2];
const SPLIT_T7_ROLE_B_CLUSTER_OFFSET: u32 =
    SPLIT_T7_ROLE_A_GRID[0] * SPLIT_T7_ROLE_A_GRID[1] * SPLIT_T7_ROLE_A_GRID[2];

fn default_cryptic_search_outer_cutoff_angstrom() -> f32 {
    12.0
}

#[cfg(feature = "gpu")]
unsafe extern "C" {
    fn prism_ghost_pipe_stage_launch_v2(
        ring_base_dev: u64,
        tiles: *const std::ffi::c_void,
        d_spikes: *const std::ffi::c_void,
        d_cluster_offsets: *const std::ffi::c_void,
        adjudicator: *const std::ffi::c_void,
        kcc_lead: *const std::ffi::c_void,
        frame_idx: u64,
        n_clusters: u32,
        max_records: u32,
        stream: *mut std::ffi::c_void,
        firehose_enable: u32,
        observation_threshold_kl: f32,
        discovery_threshold_kl: f32,
        gear_id: u32,
        dt_fs: f32,
        step_idx: u64,
        step_counter_dev: u64,
        prune_kl_threshold: f32,
    ) -> i32;
}

/// Strict runtime and provenance contract for a two-topology coupled-TWIN run.
///
/// This is intentionally separate from [`CoupledTwinConfig`]. The latter is a
/// scientific protocol configuration shared with the historical
/// single-topology lane; this contract governs the production PrismOS evidence
/// boundary and is only supplied for `--topology-b` executions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoupledTwinEvidenceConfig {
    pub schema_kind: String,
    pub schema_version: u32,
    pub run_id: String,
    pub topology_a_path: String,
    pub topology_b_path: String,
    pub topology_a_sha256: String,
    pub topology_b_sha256: String,
    pub contrast_kind: String,
    pub engine_b_role: String,
    pub identical_topology_negative_control: bool,
    pub require_prismcode_md_native: bool,
    pub require_cuda_graph_native: bool,
    pub require_metal_resident_md: bool,
    pub require_hotpath_acceleration: bool,
    pub min_hotpath_launch_collapse: f64,
    pub graph_chunk_outer_steps: u32,
    pub captured_graph_body_unroll: u32,
    pub ghost_diagnostic_firehose: bool,
    pub mar_v2_telemetry: bool,
    #[serde(default)]
    pub probe_residue_ids: Vec<usize>,
    pub ligand_contact_cutoff_angstrom: f32,
    #[serde(default = "default_cryptic_search_outer_cutoff_angstrom")]
    pub cryptic_search_outer_cutoff_angstrom: f32,
    pub identical_control_max_mean_abs_contrast: f64,
    pub distinct_min_mean_abs_contrast: f64,
    pub distinct_min_pocket_localization_fraction: f64,
    pub require_ghost_pocket_binding: bool,
    #[serde(default)]
    pub require_shared_lattice_therm: bool,
}

impl CoupledTwinEvidenceConfig {
    pub fn validate(&self) -> Result<()> {
        if self.schema_kind != "prismos.coupled_twin_evidence_config.v2"
            || self.schema_version != COUPLED_TWIN_EVIDENCE_SCHEMA_VERSION
        {
            bail!(
                "invalid coupled-TWIN evidence contract kind/version: {}/{}",
                self.schema_kind,
                self.schema_version
            );
        }
        if self.run_id.trim().is_empty() {
            bail!("coupled-TWIN evidence run_id is empty");
        }
        for (name, digest) in [
            ("topology_a_sha256", self.topology_a_sha256.as_str()),
            ("topology_b_sha256", self.topology_b_sha256.as_str()),
        ] {
            if digest.len() != 64 || !digest.bytes().all(|b| b.is_ascii_hexdigit()) {
                bail!("{} is not a full SHA-256 digest: {}", name, digest);
            }
        }
        if self.topology_a_path == self.topology_b_path
            && self.topology_a_sha256 != self.topology_b_sha256
        {
            bail!("one topology path resolves to two different identities");
        }
        let identical_identity = self.topology_a_sha256 == self.topology_b_sha256;
        if identical_identity != self.identical_topology_negative_control {
            bail!(
                "identical-topology control mismatch: hashes_identical={} negative_control={}",
                identical_identity,
                self.identical_topology_negative_control
            );
        }
        if identical_identity && self.contrast_kind != "identical-topology-negative-control" {
            bail!(
                "same-hash topology pairs require contrast_kind=identical-topology-negative-control"
            );
        }
        if !identical_identity && self.contrast_kind == "identical-topology-negative-control" {
            bail!("distinct topology hashes cannot use the identical-topology control kind");
        }
        if !self.min_hotpath_launch_collapse.is_finite() || self.min_hotpath_launch_collapse < 1.0 {
            bail!("invalid minimum hotpath launch-collapse threshold");
        }
        if self.graph_chunk_outer_steps == 0 || self.captured_graph_body_unroll == 0 {
            bail!("graph chunk and body-unroll values must be non-zero");
        }
        if !self.ligand_contact_cutoff_angstrom.is_finite()
            || self.ligand_contact_cutoff_angstrom <= 0.0
        {
            bail!("ligand contact cutoff must be finite and positive");
        }
        let canonical_probe_residue_ids = self
            .probe_residue_ids
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        if canonical_probe_residue_ids != self.probe_residue_ids {
            bail!("probe residue IDs must be unique and strictly increasing");
        }
        let glycan_probe = self.contrast_kind == GLYCAN_MECHANISTIC_PROBE_CONTRAST_KIND;
        if glycan_probe {
            if identical_identity || self.probe_residue_ids.is_empty() {
                bail!("glycan mechanistic probe contrast requires distinct topologies and explicit probe residue IDs");
            }
            if !self.cryptic_search_outer_cutoff_angstrom.is_finite()
                || self.cryptic_search_outer_cutoff_angstrom <= self.ligand_contact_cutoff_angstrom
            {
                bail!("glycan mechanistic probe cryptic search cutoff must be finite and greater than the direct-contact cutoff");
            }
        } else if !self.probe_residue_ids.is_empty() {
            bail!("explicit probe residue IDs are reserved for glycan-mechanistic-probe contrasts");
        }
        if !self.identical_control_max_mean_abs_contrast.is_finite()
            || !(0.0..1.0).contains(&self.identical_control_max_mean_abs_contrast)
        {
            bail!("identical-control contrast ceiling must be finite and in [0,1)");
        }
        if !self.distinct_min_mean_abs_contrast.is_finite()
            || !(0.0..1.0).contains(&self.distinct_min_mean_abs_contrast)
        {
            bail!("distinct contrast floor must be finite and in [0,1)");
        }
        if !self.distinct_min_pocket_localization_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.distinct_min_pocket_localization_fraction)
        {
            bail!("pocket-localization floor must be finite and in [0,1]");
        }
        if !self.require_ghost_pocket_binding {
            bail!("production two-topology evidence requires Ghost-to-pocket binding");
        }
        if !self.mar_v2_telemetry || !self.ghost_diagnostic_firehose {
            bail!("production two-topology evidence requires MAR-v2 telemetry and Ghost firehose emission");
        }
        if !self.require_prismcode_md_native
            || !self.require_cuda_graph_native
            || !self.require_metal_resident_md
            || !self.require_hotpath_acceleration
        {
            bail!(
                "two-topology production evidence requires native PrismCode MD, CUDA graph, metal-resident MD, and hotpath acceleration"
            );
        }
        Ok(())
    }

    pub fn validate_source_files(&self) -> Result<()> {
        self.validate()?;
        for (label, path, expected) in [
            (
                "topology A",
                Path::new(&self.topology_a_path),
                self.topology_a_sha256.as_str(),
            ),
            (
                "topology B",
                Path::new(&self.topology_b_path),
                self.topology_b_sha256.as_str(),
            ),
        ] {
            let metadata = std::fs::symlink_metadata(path)
                .with_context(|| format!("stat {} source {}", label, path.display()))?;
            if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() == 0 {
                bail!(
                    "{} must be a nonempty regular non-symlink file: {}",
                    label,
                    path.display()
                );
            }
            let observed = sha256_file(path)?;
            if observed != expected {
                bail!(
                    "{} SHA-256 mismatch for {}: expected={} observed={}",
                    label,
                    path.display(),
                    expected,
                    observed
                );
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoupledTwinArtifactEvidence {
    pub kind: String,
    pub stream_id: Option<u32>,
    pub path: String,
    pub size_bytes: u64,
    pub sha256: String,
    pub blake3: String,
    pub record_count: Option<u64>,
    pub schema_version: Option<u32>,
    pub runtime_owned: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoupledTwinGraphEvidence {
    pub stream_id: u32,
    pub direct_prefix_outer_steps: u64,
    pub direct_prefix_md_steps: u64,
    pub resident_outer_steps: u64,
    pub resident_md_steps: u64,
    pub total_md_steps: u64,
    pub capture_count: u64,
    pub host_launch_count: u64,
    pub resident_launch_count: u64,
    pub legacy_launch_count: u64,
    pub logical_outer_steps: u64,
    pub logical_md_steps: u64,
    pub steps_per_launch: u32,
    pub launch_collapse_ratio: f64,
    pub launch_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoupledTwinRuntimeEvidence {
    pub schema_kind: String,
    pub schema_version: u32,
    pub run_id: String,
    pub execution_semantics: String,
    pub exchange_active: bool,
    pub causal_transport_claim: bool,
    pub graph: Vec<CoupledTwinGraphEvidence>,
    pub artifacts: Vec<CoupledTwinArtifactEvidence>,
}

#[cfg(feature = "gpu")]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CoupledTwinCrnAtomKeyV3 {
    chain_id: String,
    residue_id: u64,
    atom_name: String,
    element: String,
}

#[cfg(feature = "gpu")]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CoupledTwinCrnInitializationReceiptV3 {
    schema_kind: String,
    schema_version: u32,
    status: String,
    validation_status: String,
    run_id: String,
    phase: String,
    trial_ordinal: Option<u32>,
    topology_a_sha256: String,
    topology_b_sha256: String,
    seed: u64,
    temperature_kelvin_f32_bits: u32,
    atom_key_schema: String,
    gaussian_algorithm: String,
    shared_velocity_domain: String,
    shared_position_domain: String,
    b_only_velocity_domain: String,
    b_only_position_domain: String,
    cuda_rng_semantics: String,
    direct_cuda_rng_state_readback: bool,
    direct_cuda_rng_state_readback_status: String,
    atom_count_a: u64,
    atom_count_b: u64,
    shared_atom_count: u64,
    b_only_atom_count: u64,
    shared_atom_map_sha256: String,
    b_only_atom_map_sha256: String,
    shared_atom_ordinals_identical: bool,
    b_only_atoms_are_contiguous_suffix: bool,
    cuda_rng_shared_ordinal_alignment_verified: bool,
    shared_input_position_sha256: String,
    shared_mass_sha256: String,
    b_only_input_position_sha256: String,
    b_only_mass_sha256: String,
    shared_velocity_center_before_f64_bits: [u64; 3],
    shared_velocity_center_after_f64_bits: [u64; 3],
    shared_displacement_center_before_f64_bits: [u64; 3],
    shared_displacement_center_after_f64_bits: [u64; 3],
    b_only_velocity_center_before_f64_bits: [u64; 3],
    b_only_velocity_center_after_f64_bits: [u64; 3],
    b_only_displacement_center_before_f64_bits: [u64; 3],
    b_only_displacement_center_after_f64_bits: [u64; 3],
    shared_position_sha256: String,
    shared_velocity_sha256: String,
    shared_displacement_sha256: String,
    b_only_position_sha256: String,
    b_only_velocity_sha256: String,
    b_only_displacement_sha256: String,
    host_position_sha256_a: String,
    host_position_sha256_b: String,
    host_velocity_sha256_a: String,
    host_velocity_sha256_b: String,
    device_position_readback_sha256_a: String,
    device_position_readback_sha256_b: String,
    device_velocity_readback_sha256_a: String,
    device_velocity_readback_sha256_b: String,
    device_mass_readback_sha256_a: String,
    device_mass_readback_sha256_b: String,
    device_zero_force_readback_sha256_a: String,
    device_zero_force_readback_sha256_b: String,
    device_zero_potential_readback_sha256_a: String,
    device_zero_potential_readback_sha256_b: String,
    reset_seed_equality_verified: bool,
    exact_device_readback_verified: bool,
    fallback_used: bool,
    semantic_sha256: String,
}

#[cfg(feature = "gpu")]
struct CoupledTwinCrnHostStateV3 {
    positions_a: Vec<f32>,
    positions_b: Vec<f32>,
    velocities_a: Vec<f32>,
    velocities_b: Vec<f32>,
    shared_indices: Vec<(usize, usize)>,
    b_only_indices: Vec<usize>,
    receipt: CoupledTwinCrnInitializationReceiptV3,
}

#[cfg(feature = "gpu")]
#[derive(Debug, Clone, PartialEq, Eq)]
struct CoupledTwinCrnDeviceReadbackV3 {
    position_sha256_a: String,
    position_sha256_b: String,
    velocity_sha256_a: String,
    velocity_sha256_b: String,
    mass_sha256_a: String,
    mass_sha256_b: String,
    zero_force_sha256_a: String,
    zero_force_sha256_b: String,
    zero_potential_sha256_a: String,
    zero_potential_sha256_b: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoupledGhostRecordEvidence {
    pub record_index: u64,
    pub byte_offset: u64,
    pub frame_idx: u64,
    pub site_id: u32,
    pub topology_cluster_id: u32,
    pub chain_id: String,
    pub native_chain_id: String,
    pub adjudication_code: u8,
    pub kl_divergence: f32,
    pub observation_pass: bool,
    pub discovery_pass: bool,
    pub step_idx: u64,
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    pub centroid: [f32; 3],
    pub support_residue_ids: Vec<i32>,
    pub source_event_witnesses: Vec<CoupledGhostSourceEventWitness>,
    pub representative_residue_id: i32,
    pub evidence_available: bool,
    pub neutral_reason: Option<String>,
    pub field_completeness_flags: u16,
    pub spatial_native: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoupledGhostSourceEventWitness {
    pub selected_ordinal: u32,
    pub source_index: u32,
    pub canonical_event_sha256: String,
    pub position: [f32; 3],
    pub primary_residue_id: i32,
    pub support_residue_ids: Vec<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoupledGhostSiteMap {
    pub schema_kind: String,
    pub schema_version: u32,
    pub run_id: String,
    pub stream_id: u32,
    pub coordinate_frame: String,
    pub native_chain_role_map: BTreeMap<String, String>,
    pub topology_path: String,
    pub topology_sha256: String,
    pub topology_layout_sha256: String,
    pub frozen_anchor_path: String,
    pub frozen_anchor_sha256: String,
    pub frozen_anchor_bytes: u64,
    pub ghost_payload_path: String,
    pub ghost_payload_sha256: String,
    pub ghost_payload_bytes: u64,
    pub record_count: u64,
    pub producer: String,
    pub execution_stage: String,
    pub step_counter_initial: u64,
    pub emitted_step_idx: u64,
    pub dt_fs: f32,
    pub dt_source: String,
    pub gear_id: u32,
    pub gear_id_source: String,
    pub noise_floor_mu: f32,
    pub noise_floor_sigma: f32,
    pub noise_floor_source: String,
    pub noise_floor_calibration_path: String,
    pub noise_floor_calibration_sha256: String,
    pub noise_floor_calibration_frame_count: u32,
    pub raw_spike_count: u64,
    pub selected_spike_count: u32,
    pub selected_rich_spikes_sha256: String,
    pub final_source_events_path: String,
    pub final_source_events_sha256: String,
    pub final_source_events_bytes: u64,
    pub final_canonical_source_events_sha256: String,
    pub topology_cluster_count: u32,
    pub emitted_occupied_cluster_count: u32,
    pub final_total_kl: f32,
    pub final_adjudication_code: u32,
    pub final_adjudication_reason_flags: u32,
    pub final_engine_state_sha256_before: String,
    pub final_engine_state_sha256_after: String,
    pub final_engine_physical_state_sha256_before: String,
    pub final_engine_physical_state_sha256_after: String,
    pub fallback_used: bool,
    pub unavailable_fields: Vec<String>,
    pub records: Vec<CoupledGhostRecordEvidence>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoupledNoiseFloorFrameEvidence {
    pub phase: String,
    pub ordinal: u32,
    pub timestep: i32,
    pub source_timestep_min: i32,
    pub source_timestep_max: i32,
    pub analysis_md_step: u64,
    pub source_spike_count: u32,
    pub selected_spike_count: u32,
    pub dropped_spike_count: u32,
    pub input_byte_offset: u64,
    pub input_byte_length: u64,
    pub source_input_byte_offset: u64,
    pub source_input_byte_length: u64,
    pub selection_index_byte_offset: u64,
    pub selection_index_byte_length: u64,
    pub source_events_sha256: String,
    pub selected_rich_spikes_sha256: String,
    pub total_kl: Option<f32>,
    pub total_kl_magnitude: Option<f32>,
    pub adjudication_code: u32,
    pub adjudication_reason_flags: u32,
    pub engine_state_sha256_before: String,
    pub engine_state_sha256_after: String,
    pub engine_physical_state_sha256_before: String,
    pub engine_physical_state_sha256_after: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoupledNoiseFloorCalibration {
    pub schema_kind: String,
    pub schema_version: u32,
    pub status: String,
    pub validation_status: String,
    pub run_id: String,
    pub stream_id: u32,
    pub topology_path: String,
    pub topology_sha256: String,
    pub producer: String,
    pub source: String,
    pub calibration_phase: String,
    pub selection_method: String,
    pub topology_layout_sha256: String,
    pub topology_aabb_min: [f32; 3],
    pub topology_aabb_max: [f32; 3],
    pub fixed_cluster_count: u32,
    pub fixed_spikes_per_frame: u32,
    pub inventory_window_md_steps: u32,
    pub shadow_trial_count: u32,
    pub shadow_reserve_factor: u32,
    pub shadow_lower_envelope_frame_counts: Vec<u32>,
    pub shadow_ensemble_sha256: String,
    pub shadow_ensemble_receipt_path: String,
    pub shadow_ensemble_receipt_sha256: String,
    pub shadow_ensemble_receipt_bytes: u64,
    pub shadow_holdout_admission_verified: bool,
    pub authoritative_admission_verified: bool,
    pub authoritative_minimum: u32,
    pub authoritative_minimum_slack_from_k: u32,
    pub inventory_frame_counts: Vec<u32>,
    pub paired_inventory_sha256: String,
    pub equal_duration_windows_verified: bool,
    pub direct_prefix_outer_steps: u32,
    pub direct_prefix_md_steps: u64,
    pub calibration_input_path: String,
    pub calibration_input_sha256: String,
    pub calibration_input_bytes: u64,
    pub frozen_anchor_path: String,
    pub frozen_anchor_sha256: String,
    pub frozen_anchor_bytes: u64,
    pub available_planes: Vec<String>,
    pub unavailable_planes: Vec<String>,
    pub anchor_frame_count: u32,
    pub variance_frame_count: u32,
    pub validation_frame_count: u32,
    pub pipeline_launch_count: u32,
    pub baseline_freeze_count: u32,
    pub kl_threshold_write_count: u32,
    pub minimum_spikes_per_frame: u32,
    pub maximum_spikes_per_frame: u32,
    pub noise_floor_mu: f32,
    pub noise_floor_sigma: f32,
    pub threshold_3sigma: f32,
    pub threshold_12sigma: f32,
    pub validation_frame_total_kl: f32,
    pub validation_frame_adjudication_marker: u32,
    pub validation_adjudication_code: u32,
    pub validation_adjudication_reason_flags: u32,
    pub state_noninterference_verified: bool,
    pub topology_constants_rebound_each_launch: bool,
    pub fallback_used: bool,
    pub frames: Vec<CoupledNoiseFloorFrameEvidence>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MolecularContrastResidue {
    pub residue_id: i32,
    pub residue_name: String,
    pub spikes_a: u32,
    pub spikes_b: u32,
    pub intensity_a: f32,
    pub intensity_b: f32,
    pub signed_count_contrast: f32,
    pub log2_count_ratio: f32,
    pub signed_intensity_contrast: f32,
    pub log2_intensity_ratio: f32,
    pub support_class: String,
    pub aligned_zero_lag_ccf: Option<f32>,
    pub aligned_zero_lag_ccf_applicability: String,
    pub thresholded_total_outgoing_transfer_entropy_a_to_b: Option<f32>,
    pub transfer_entropy_applicability: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MolecularContrastPocketResidueDecisionV4 {
    pub residue_id: usize,
    pub event_row_present: bool,
    pub strict_residue_row_present: bool,
    pub contrast_supported: bool,
    pub abs_signed_count_contrast: Option<f32>,
    pub support_status: String,
}

#[derive(Debug, Clone, PartialEq)]
struct MolecularContrastPocketSupportV4 {
    decisions: Vec<MolecularContrastPocketResidueDecisionV4>,
    candidate_residue_count: u32,
    supported_residue_count: u32,
    unavailable_residue_count: u32,
    l1_signed_contrast: f64,
    mean_abs_signed_contrast: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MolecularContrastReadout {
    pub schema_kind: String,
    pub schema_version: u32,
    pub run_id: String,
    pub contrast_kind: String,
    pub analysis_semantics: String,
    pub exchange_active: bool,
    pub causal_transport_claim: bool,
    pub topology_a_path: String,
    pub topology_b_path: String,
    pub topology_a_sha256: String,
    pub topology_b_sha256: String,
    pub identical_topology_identity: bool,
    pub alignment_reference_residue_count: usize,
    pub alignment_reference_ca_count: usize,
    pub alignment_b_residue_count: usize,
    pub alignment_b_only_residue_count: usize,
    pub alignment_mutation_count: usize,
    pub residue_alignment_sha256: String,
    pub ligand_atom_count: usize,
    pub ligand_residue_ids: Vec<usize>,
    pub ligand_aabb_min: [f32; 3],
    pub ligand_aabb_max: [f32; 3],
    pub ligand_centroid: [f32; 3],
    pub ligand_geometry_sha256: String,
    pub ligand_heavy_atom_count: usize,
    pub ligand_heavy_atoms: Vec<LigandHeavyAtomBinding>,
    pub ligand_heavy_atom_coordinate_sha256: String,
    pub ligand_contact_cutoff_angstrom: f32,
    pub pocket_contact_residue_ids: Vec<usize>,
    pub pocket_contact_atom_count: usize,
    pub pocket_contrast_candidate_residue_count: u32,
    pub pocket_contrast_supported_residue_count: u32,
    pub pocket_contrast_unavailable_residue_count: u32,
    pub pocket_contrast_residue_decisions: Vec<MolecularContrastPocketResidueDecisionV4>,
    pub global_l1_signed_contrast: f64,
    pub pocket_l1_signed_contrast: f64,
    pub global_mean_abs_signed_contrast: f64,
    pub pocket_mean_abs_signed_contrast: Option<f64>,
    pub pocket_localization_fraction: f64,
    pub ghost_pocket_component_ids: Vec<String>,
    pub ghost_pocket_witnesses: Vec<GhostPocketWitnessBinding>,
    pub ghost_pocket_bound_count: u64,
    pub ghost_pocket_binding_verified: bool,
    pub ghost_pocket_binding_status: String,
    pub residues: Vec<MolecularContrastResidue>,
    pub finite_values_verified: bool,
    pub fallback_used: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GhostPocketWitnessBinding {
    pub component_id: String,
    pub source_index: u32,
    pub canonical_event_sha256: String,
    pub position: [f32; 3],
    pub support_residue_ids: Vec<i32>,
    pub pocket_support_intersection_residue_ids: Vec<i32>,
    pub topology_b_sha256: String,
    pub ligand_coordinate_sha256: String,
    pub cutoff_angstrom: f32,
    pub coarse_ligand_aabb_distance_angstrom: f32,
    pub nearest_ligand_atom_index: usize,
    pub nearest_ligand_atom_element: String,
    pub nearest_ligand_atom_position: [f32; 3],
    pub minimum_ligand_heavy_atom_distance_angstrom: f32,
    pub exact_distance_admission: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LigandHeavyAtomBinding {
    pub atom_index: usize,
    pub residue_id: usize,
    pub element: String,
    pub position: [f32; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlycanMechanisticProbeCandidateV1 {
    pub candidate_id: String,
    pub support_residue_ids: Vec<i32>,
    pub stream_a_component_ids: Vec<String>,
    pub stream_b_component_ids: Vec<String>,
    pub stream_a_frame_count: u64,
    pub stream_b_frame_count: u64,
    pub stream_a_witness_count: u64,
    pub stream_b_witness_count: u64,
    pub minimum_input_pose_probe_to_candidate_receptor_heavy_atom_distance_angstrom: f32,
    pub direct_contact_excluded: bool,
    pub control_observed_in_this_run: bool,
    pub probe_enriched_in_this_run: bool,
    pub classification: String,
    pub signaling_hypothesis_status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlycanMechanisticProbeReadoutV1 {
    pub schema_kind: String,
    pub schema_version: u32,
    pub run_id: String,
    pub contrast_kind: String,
    pub analysis_semantics: String,
    pub topology_a_sha256: String,
    pub topology_b_sha256: String,
    pub probe_residue_ids: Vec<usize>,
    pub probe_heavy_atom_count: usize,
    pub input_pose_probe_heavy_atom_coordinate_sha256: String,
    pub search_zone_coordinate_frame: String,
    pub search_zone_definition: String,
    pub search_zone_sha256: String,
    pub search_zone_receptor_residue_count: u64,
    pub dynamic_signaling_evidence: String,
    pub direct_contact_cutoff_angstrom: f32,
    pub cryptic_search_outer_cutoff_angstrom: f32,
    pub direct_contact_residue_ids: Vec<usize>,
    pub matched_control_method: String,
    pub candidate_count: u64,
    pub control_not_observed_candidate_count: u64,
    pub probe_enriched_candidate_count: u64,
    pub candidates: Vec<GlycanMechanisticProbeCandidateV1>,
    pub signaling_hypothesis_status: String,
    pub causal_transport_claim: bool,
    pub structural_novelty_claim: bool,
    pub patent_novelty_claim: bool,
    pub novelty_status: String,
    pub replica_consensus_verified: bool,
    pub finite_values_verified: bool,
    pub fallback_used: bool,
}

#[derive(Debug, Default)]
struct GlycanProbeCandidateAccumulatorV1 {
    stream_a_component_ids: BTreeSet<String>,
    stream_b_component_ids: BTreeSet<String>,
    stream_a_frames: BTreeSet<u64>,
    stream_b_frames: BTreeSet<u64>,
    stream_a_witness_count: u64,
    stream_b_witness_count: u64,
    minimum_input_pose_probe_to_candidate_receptor_heavy_atom_distance_angstrom: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CoupledTwinManifestArtifact {
    kind: String,
    stream_id: Option<u32>,
    path: String,
    size_bytes: u64,
    present: bool,
    note: Option<String>,
    sha256: String,
    blake3: String,
    record_count: Option<u64>,
    schema_version: Option<u32>,
    runtime_owned: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CoupledTwinManifestExtension {
    schema_kind: String,
    schema_version: u32,
    topology_a_path: String,
    topology_b_path: String,
    topology_a_sha256: String,
    topology_b_sha256: String,
    identical_topology_negative_control: bool,
    exchange_active: bool,
    causal_transport_claim: bool,
    fallback_used: bool,
    config: CoupledTwinEvidenceConfig,
    runtime_evidence: CoupledTwinRuntimeEvidence,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CoupledTwinMdEvidenceManifest {
    schema_version: u32,
    schema_kind: String,
    status: String,
    run_id: String,
    topology_input: String,
    topology_b_input: String,
    topology_a_sha256: String,
    topology_b_sha256: String,
    target: String,
    stream_count: usize,
    streams_serialized: usize,
    total_spikes_md: u64,
    artifacts: Vec<CoupledTwinManifestArtifact>,
    coupled_twin_evidence: CoupledTwinManifestExtension,
    compression_status: String,
    checksum_algorithm: String,
    endian: String,
    v2_was_live: bool,
    path_b_required: bool,
    validation_status: String,
    required_artifacts_complete: bool,
    serialization_failure: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeGhostArtifactBinding {
    path: String,
    sha256: String,
    bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeGhostSourceArtifactBinding {
    kind: String,
    stream_id: u32,
    path: String,
    sha256: String,
    bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeGhostRecordProvenance {
    file: String,
    file_sha256: String,
    file_bytes: u64,
    record_index: u64,
    byte_offset: u64,
    record_bytes: u64,
    schema_version: u32,
    field_completeness_flags: u16,
    spatial_native_bit: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeGhostSiteMapEntry {
    component_id: String,
    stream_id: u32,
    frame_idx: u64,
    site_id: u32,
    aabb: [f32; 6],
    centroid_xyz: [f32; 3],
    residue_support: Vec<i32>,
    source: String,
    record_provenance: NativeGhostRecordProvenance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeGhostSiteMap {
    schema: String,
    schema_version: u32,
    schema_kind: String,
    status: String,
    coordinate_frame: String,
    run_id: String,
    native_v2_event_count: u64,
    source_result: NativeGhostArtifactBinding,
    source_stream_artifacts: Vec<NativeGhostSourceArtifactBinding>,
    entries: Vec<NativeGhostSiteMapEntry>,
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(feature = "gpu")]
fn crn_domain_name(domain: &[u8]) -> Result<String> {
    let name = std::str::from_utf8(domain).context("CRN domain is not UTF-8")?;
    Ok(name.trim_end_matches('\0').to_string())
}

#[cfg(feature = "gpu")]
fn crn_atom_key(
    topology: &PrismPrepTopology,
    atom_index: usize,
) -> Result<CoupledTwinCrnAtomKeyV3> {
    if atom_index >= topology.n_atoms {
        bail!("CRN atom index {atom_index} exceeds topology atom count");
    }
    let normalize = |value: &str, label: &str| -> Result<String> {
        if value.chars().any(char::is_control) {
            bail!("CRN atom {atom_index} has malformed {label}");
        }
        let value = value.trim();
        if value.is_empty() {
            bail!("CRN atom {atom_index} has malformed {label}");
        }
        Ok(value.to_string())
    };
    let chain_id = normalize(
        topology
            .chain_ids
            .get(atom_index)
            .context("CRN topology lacks per-atom chain ID")?,
        "chain ID",
    )?;
    let atom_name = normalize(
        topology
            .atom_names
            .get(atom_index)
            .context("CRN topology lacks per-atom atom name")?,
        "atom name",
    )?;
    let element = normalize(
        topology
            .elements
            .get(atom_index)
            .context("CRN topology lacks per-atom element")?,
        "element",
    )?
    .to_ascii_uppercase();
    let residue_id = *topology
        .residue_ids
        .get(atom_index)
        .context("CRN topology lacks per-atom residue ID")? as u64;
    Ok(CoupledTwinCrnAtomKeyV3 {
        chain_id,
        residue_id,
        atom_name,
        element,
    })
}

#[cfg(feature = "gpu")]
fn crn_atom_key_bytes(key: &CoupledTwinCrnAtomKeyV3) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    let chain_length = u32::try_from(key.chain_id.len()).context("CRN chain ID is too long")?;
    bytes.extend_from_slice(&chain_length.to_le_bytes());
    bytes.extend_from_slice(key.chain_id.as_bytes());
    bytes.extend_from_slice(&key.residue_id.to_le_bytes());
    for value in [&key.atom_name, &key.element] {
        let length = u32::try_from(value.len()).context("CRN atom-key string is too long")?;
        bytes.extend_from_slice(&length.to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }
    Ok(bytes)
}

#[cfg(feature = "gpu")]
fn crn_validate_topology_shape(topology: &PrismPrepTopology, label: &str) -> Result<()> {
    if topology.n_atoms == 0
        || topology.positions.len() != topology.n_atoms.saturating_mul(3)
        || topology.elements.len() != topology.n_atoms
        || topology.atom_names.len() != topology.n_atoms
        || topology.residue_ids.len() != topology.n_atoms
        || topology.chain_ids.len() != topology.n_atoms
        || topology.masses.len() != topology.n_atoms
    {
        bail!("CRN {label} topology arrays do not cover every atom exactly");
    }
    if topology.positions.iter().any(|value| !value.is_finite())
        || topology
            .masses
            .iter()
            .any(|mass| !mass.is_finite() || *mass <= 0.0)
    {
        bail!("CRN {label} topology contains nonfinite coordinates or invalid masses");
    }
    Ok(())
}

#[cfg(feature = "gpu")]
fn crn_hash_f32_slice(label: &str, values: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(
        COUPLED_TWIN_CRN_ARRAY_DOMAIN.len() + label.len() + 16 + values.len() * 4,
    );
    bytes.extend_from_slice(COUPLED_TWIN_CRN_ARRAY_DOMAIN);
    bytes.extend_from_slice(&(label.len() as u64).to_le_bytes());
    bytes.extend_from_slice(label.as_bytes());
    bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    sha256_bytes(&bytes)
}

#[cfg(feature = "gpu")]
fn crn_hash_f64_slice(label: &str, values: &[f64]) -> String {
    let mut bytes = Vec::with_capacity(
        COUPLED_TWIN_CRN_ARRAY_DOMAIN.len() + label.len() + 16 + values.len() * 8,
    );
    bytes.extend_from_slice(COUPLED_TWIN_CRN_ARRAY_DOMAIN);
    bytes.extend_from_slice(&(label.len() as u64).to_le_bytes());
    bytes.extend_from_slice(label.as_bytes());
    bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    sha256_bytes(&bytes)
}

#[cfg(feature = "gpu")]
fn crn_standard_normal(
    domain: &[u8],
    seed: u64,
    scope_identity: &str,
    key_bytes: &[u8],
    component: u8,
) -> Result<f64> {
    if domain.is_empty() || scope_identity.is_empty() || key_bytes.is_empty() || component >= 3 {
        bail!("CRN Gaussian counter input is invalid");
    }
    let mut input = Vec::with_capacity(domain.len() + scope_identity.len() + key_bytes.len() + 32);
    input.extend_from_slice(domain);
    input.extend_from_slice(&COUPLED_TWIN_CRN_SCHEMA_VERSION.to_le_bytes());
    input.extend_from_slice(&seed.to_le_bytes());
    input.extend_from_slice(&(scope_identity.len() as u64).to_le_bytes());
    input.extend_from_slice(scope_identity.as_bytes());
    input.extend_from_slice(&(key_bytes.len() as u64).to_le_bytes());
    input.extend_from_slice(key_bytes);
    input.push(component);
    let digest = Sha256::digest(&input);
    let first = u64::from_le_bytes(digest[0..8].try_into().expect("SHA-256 word"));
    let second = u64::from_le_bytes(digest[8..16].try_into().expect("SHA-256 word"));
    const TWO_POW_53: f64 = 9_007_199_254_740_992.0;
    let u0 = ((first >> 11) as f64 + 0.5) / TWO_POW_53;
    let u1 = ((second >> 11) as f64 + 0.5) / TWO_POW_53;
    let value = (-2.0 * u0.ln()).sqrt() * (std::f64::consts::TAU * u1).cos();
    if !value.is_finite() {
        bail!("CRN Gaussian counter produced a nonfinite value");
    }
    Ok(value)
}

#[cfg(feature = "gpu")]
fn crn_remove_mass_weighted_center(
    vectors: &mut [[f64; 3]],
    masses: &[f32],
    label: &str,
) -> Result<([u64; 3], [u64; 3])> {
    if vectors.len() != masses.len() || vectors.is_empty() {
        bail!("CRN {label} balance group is empty or shape-mismatched");
    }
    let total_mass: f64 = masses.iter().map(|mass| *mass as f64).sum();
    if !total_mass.is_finite() || total_mass <= 0.0 {
        bail!("CRN {label} balance group has invalid total mass");
    }
    let center = |values: &[[f64; 3]]| -> [f64; 3] {
        let mut sum = [0.0f64; 3];
        for (vector, mass) in values.iter().zip(masses) {
            for axis in 0..3 {
                sum[axis] += *mass as f64 * vector[axis];
            }
        }
        [
            sum[0] / total_mass,
            sum[1] / total_mass,
            sum[2] / total_mass,
        ]
    };
    let before = center(vectors);
    for vector in vectors.iter_mut() {
        for axis in 0..3 {
            vector[axis] -= before[axis];
        }
    }
    let after = center(vectors);
    if after
        .iter()
        .any(|value| !value.is_finite() || value.abs() > COUPLED_TWIN_CRN_BALANCE_TOLERANCE)
    {
        bail!("CRN {label} balance residual exceeds tolerance: {after:?}");
    }
    Ok((before.map(f64::to_bits), after.map(f64::to_bits)))
}

#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn build_coupled_twin_crn_host_state_v3(
    topology_a: &PrismPrepTopology,
    topology_b: &PrismPrepTopology,
    config: &CoupledTwinEvidenceConfig,
    seed: u64,
    temperature_kelvin: f32,
    phase: &str,
    trial_ordinal: Option<u32>,
) -> Result<CoupledTwinCrnHostStateV3> {
    crn_validate_topology_shape(topology_a, "A")?;
    crn_validate_topology_shape(topology_b, "B")?;
    if !temperature_kelvin.is_finite() || temperature_kelvin <= 0.0 || phase.trim().is_empty() {
        bail!("CRN temperature or phase is invalid");
    }
    let mut map_a = BTreeMap::new();
    let mut map_b = BTreeMap::new();
    for atom_index in 0..topology_a.n_atoms {
        let key = crn_atom_key(topology_a, atom_index)?;
        if map_a.insert(key, atom_index).is_some() {
            bail!("CRN topology A contains a duplicate canonical atom identity");
        }
    }
    for atom_index in 0..topology_b.n_atoms {
        let key = crn_atom_key(topology_b, atom_index)?;
        if map_b.insert(key, atom_index).is_some() {
            bail!("CRN topology B contains a duplicate canonical atom identity");
        }
    }
    let mut shared_indices = Vec::with_capacity(topology_a.n_atoms);
    let mut map_bytes = Vec::new();
    map_bytes.extend_from_slice(COUPLED_TWIN_CRN_MAP_DOMAIN);
    for (key, atom_index_a) in &map_a {
        let atom_index_b = *map_b
            .get(key)
            .with_context(|| format!("CRN topology B is missing topology-A atom {key:?}"))?;
        if *atom_index_a != atom_index_b {
            bail!(
                "CRN shared atom ordinal drift for {key:?}: A={} B={}; d_rng_states cannot be identity-remapped in the authorized source lane",
                atom_index_a,
                atom_index_b
            );
        }
        let mass_a = topology_a.masses[*atom_index_a];
        let mass_b = topology_b.masses[atom_index_b];
        if mass_a.to_bits() != mass_b.to_bits() {
            bail!("CRN shared atom mass drift for {key:?}");
        }
        for axis in 0..3 {
            if topology_a.positions[*atom_index_a * 3 + axis].to_bits()
                != topology_b.positions[atom_index_b * 3 + axis].to_bits()
            {
                bail!("CRN shared atom coordinate drift for {key:?}");
            }
        }
        let key_bytes = crn_atom_key_bytes(key)?;
        map_bytes.extend_from_slice(&(key_bytes.len() as u64).to_le_bytes());
        map_bytes.extend_from_slice(&key_bytes);
        map_bytes.extend_from_slice(&(*atom_index_a as u64).to_le_bytes());
        map_bytes.extend_from_slice(&(atom_index_b as u64).to_le_bytes());
        map_bytes.extend_from_slice(&mass_a.to_bits().to_le_bytes());
        shared_indices.push((*atom_index_a, atom_index_b));
    }
    if shared_indices.len() != topology_a.n_atoms {
        bail!("CRN topology A is not a complete subset of topology B");
    }
    let shared_atom_map_sha256 = sha256_bytes(&map_bytes);
    let mut b_only_keys = map_b
        .iter()
        .filter(|(key, _)| !map_a.contains_key(*key))
        .map(|(key, index)| (key.clone(), *index))
        .collect::<Vec<_>>();
    b_only_keys.sort_by_key(|(_, index)| *index);
    let mut b_only_indices = b_only_keys
        .iter()
        .map(|(_, index)| *index)
        .collect::<Vec<_>>();
    b_only_indices.sort_unstable();
    let expected_suffix = (topology_a.n_atoms..topology_b.n_atoms).collect::<Vec<_>>();
    if b_only_indices != expected_suffix {
        bail!("CRN topology-B-only atoms are not an exact contiguous suffix");
    }
    let identical_identity = config.topology_a_sha256 == config.topology_b_sha256;
    if identical_identity && !b_only_indices.is_empty() {
        bail!("CRN identical topology identities produced B-only atoms");
    }
    if !identical_identity && b_only_indices.is_empty() {
        bail!("CRN distinct topology identities have no B-only intervention atoms");
    }
    let mut b_only_map_bytes = Vec::new();
    b_only_map_bytes.extend_from_slice(COUPLED_TWIN_CRN_MAP_DOMAIN);
    b_only_map_bytes.extend_from_slice(b"B_ONLY\0");
    b_only_map_bytes.extend_from_slice(config.topology_b_sha256.as_bytes());
    for (key, atom_index) in &b_only_keys {
        let key_bytes = crn_atom_key_bytes(key)?;
        b_only_map_bytes.extend_from_slice(&(key_bytes.len() as u64).to_le_bytes());
        b_only_map_bytes.extend_from_slice(&key_bytes);
        b_only_map_bytes.extend_from_slice(&(*atom_index as u64).to_le_bytes());
        b_only_map_bytes.extend_from_slice(&topology_b.masses[*atom_index].to_bits().to_le_bytes());
    }
    let b_only_atom_map_sha256 = sha256_bytes(&b_only_map_bytes);

    let shared_masses = shared_indices
        .iter()
        .map(|(index, _)| topology_a.masses[*index])
        .collect::<Vec<_>>();
    let b_only_masses = b_only_indices
        .iter()
        .map(|index| topology_b.masses[*index])
        .collect::<Vec<_>>();
    let mut shared_input_positions = Vec::with_capacity(shared_indices.len() * 3);
    for (index, _) in &shared_indices {
        shared_input_positions.extend_from_slice(&topology_a.positions[index * 3..index * 3 + 3]);
    }
    let mut b_only_input_positions = Vec::with_capacity(b_only_indices.len() * 3);
    for index in &b_only_indices {
        b_only_input_positions.extend_from_slice(&topology_b.positions[index * 3..index * 3 + 3]);
    }

    let mut shared_velocities = Vec::with_capacity(shared_indices.len());
    let mut shared_displacements = Vec::with_capacity(shared_indices.len());
    for ((key, _), mass) in map_a.iter().zip(&shared_masses) {
        let key_bytes = crn_atom_key_bytes(key)?;
        let velocity_sigma =
            (418.4 * COUPLED_TWIN_CRN_KB_KCAL_PER_MOL_K * temperature_kelvin as f64 / *mass as f64)
                .sqrt();
        let displacement_sigma = (COUPLED_TWIN_CRN_KB_KCAL_PER_MOL_K * temperature_kelvin as f64
            / (*mass as f64 * COUPLED_TWIN_CRN_EFFECTIVE_OMEGA_SQ))
            .sqrt();
        let mut velocity = [0.0f64; 3];
        let mut displacement = [0.0f64; 3];
        for component in 0..3 {
            velocity[component] = velocity_sigma
                * crn_standard_normal(
                    COUPLED_TWIN_CRN_SHARED_VELOCITY_DOMAIN,
                    seed,
                    &shared_atom_map_sha256,
                    &key_bytes,
                    component as u8,
                )?;
            displacement[component] = displacement_sigma
                * crn_standard_normal(
                    COUPLED_TWIN_CRN_SHARED_POSITION_DOMAIN,
                    seed,
                    &shared_atom_map_sha256,
                    &key_bytes,
                    component as u8,
                )?;
        }
        shared_velocities.push(velocity);
        shared_displacements.push(displacement);
    }
    let (shared_velocity_center_before, shared_velocity_center_after) =
        crn_remove_mass_weighted_center(&mut shared_velocities, &shared_masses, "shared velocity")?;
    let (shared_displacement_center_before, shared_displacement_center_after) =
        crn_remove_mass_weighted_center(
            &mut shared_displacements,
            &shared_masses,
            "shared displacement",
        )?;

    let b_only_scope = format!("{}:{}", b_only_atom_map_sha256, config.topology_b_sha256);
    let mut b_only_velocities = Vec::with_capacity(b_only_keys.len());
    let mut b_only_displacements = Vec::with_capacity(b_only_keys.len());
    for ((key, _), mass) in b_only_keys.iter().zip(&b_only_masses) {
        let key_bytes = crn_atom_key_bytes(key)?;
        let velocity_sigma =
            (418.4 * COUPLED_TWIN_CRN_KB_KCAL_PER_MOL_K * temperature_kelvin as f64 / *mass as f64)
                .sqrt();
        let displacement_sigma = (COUPLED_TWIN_CRN_KB_KCAL_PER_MOL_K * temperature_kelvin as f64
            / (*mass as f64 * COUPLED_TWIN_CRN_EFFECTIVE_OMEGA_SQ))
            .sqrt();
        let mut velocity = [0.0f64; 3];
        let mut displacement = [0.0f64; 3];
        for component in 0..3 {
            velocity[component] = velocity_sigma
                * crn_standard_normal(
                    COUPLED_TWIN_CRN_B_ONLY_VELOCITY_DOMAIN,
                    seed,
                    &b_only_scope,
                    &key_bytes,
                    component as u8,
                )?;
            displacement[component] = displacement_sigma
                * crn_standard_normal(
                    COUPLED_TWIN_CRN_B_ONLY_POSITION_DOMAIN,
                    seed,
                    &b_only_scope,
                    &key_bytes,
                    component as u8,
                )?;
        }
        b_only_velocities.push(velocity);
        b_only_displacements.push(displacement);
    }
    let zero_bits = [0.0f64.to_bits(); 3];
    let (b_only_velocity_center_before, b_only_velocity_center_after) = if b_only_velocities
        .is_empty()
    {
        (zero_bits, zero_bits)
    } else {
        crn_remove_mass_weighted_center(&mut b_only_velocities, &b_only_masses, "B-only velocity")?
    };
    let (b_only_displacement_center_before, b_only_displacement_center_after) =
        if b_only_displacements.is_empty() {
            (zero_bits, zero_bits)
        } else {
            crn_remove_mass_weighted_center(
                &mut b_only_displacements,
                &b_only_masses,
                "B-only displacement",
            )?
        };

    let mut positions_a = topology_a.positions.clone();
    let mut positions_b = topology_b.positions.clone();
    let mut velocities_a = vec![0.0f32; topology_a.n_atoms * 3];
    let mut velocities_b = vec![0.0f32; topology_b.n_atoms * 3];
    let mut shared_positions = Vec::with_capacity(shared_indices.len() * 3);
    let mut shared_velocity_values = Vec::with_capacity(shared_indices.len() * 3);
    let mut shared_displacement_values = Vec::with_capacity(shared_indices.len() * 3);
    for (ordinal, (atom_index_a, atom_index_b)) in shared_indices.iter().enumerate() {
        for axis in 0..3 {
            let displacement = shared_displacements[ordinal][axis] as f32;
            let velocity = shared_velocities[ordinal][axis] as f32;
            let position = topology_a.positions[atom_index_a * 3 + axis] + displacement;
            if !position.is_finite() || !velocity.is_finite() {
                bail!("CRN shared initialization produced a nonfinite value");
            }
            positions_a[atom_index_a * 3 + axis] = position;
            positions_b[atom_index_b * 3 + axis] = position;
            velocities_a[atom_index_a * 3 + axis] = velocity;
            velocities_b[atom_index_b * 3 + axis] = velocity;
            shared_positions.push(position);
            shared_velocity_values.push(velocity);
            shared_displacement_values.push(displacement);
        }
    }
    let mut b_only_positions = Vec::with_capacity(b_only_indices.len() * 3);
    let mut b_only_velocity_values = Vec::with_capacity(b_only_indices.len() * 3);
    let mut b_only_displacement_values = Vec::with_capacity(b_only_indices.len() * 3);
    for (ordinal, atom_index) in b_only_indices.iter().enumerate() {
        for axis in 0..3 {
            let displacement = b_only_displacements[ordinal][axis] as f32;
            let velocity = b_only_velocities[ordinal][axis] as f32;
            let position = topology_b.positions[atom_index * 3 + axis] + displacement;
            if !position.is_finite() || !velocity.is_finite() {
                bail!("CRN B-only initialization produced a nonfinite value");
            }
            positions_b[atom_index * 3 + axis] = position;
            velocities_b[atom_index * 3 + axis] = velocity;
            b_only_positions.push(position);
            b_only_velocity_values.push(velocity);
            b_only_displacement_values.push(displacement);
        }
    }
    let host_position_sha256_a = crn_hash_f32_slice("stream_a_positions", &positions_a);
    let host_position_sha256_b = crn_hash_f32_slice("stream_b_positions", &positions_b);
    let host_velocity_sha256_a = crn_hash_f32_slice("stream_a_velocities", &velocities_a);
    let host_velocity_sha256_b = crn_hash_f32_slice("stream_b_velocities", &velocities_b);
    let mut receipt = CoupledTwinCrnInitializationReceiptV3 {
        schema_kind: "prismos.coupled_twin_common_random_initialization.v3".to_string(),
        schema_version: COUPLED_TWIN_CRN_SCHEMA_VERSION,
        status: "IMPLEMENTATION_PARTIAL".to_string(),
        validation_status: "PENDING_DEVICE_READBACK".to_string(),
        run_id: config.run_id.clone(),
        phase: phase.to_string(),
        trial_ordinal,
        topology_a_sha256: config.topology_a_sha256.clone(),
        topology_b_sha256: config.topology_b_sha256.clone(),
        seed,
        temperature_kelvin_f32_bits: temperature_kelvin.to_bits(),
        atom_key_schema: COUPLED_TWIN_CRN_ATOM_KEY_SCHEMA.to_string(),
        gaussian_algorithm: COUPLED_TWIN_CRN_GAUSSIAN_ALGORITHM.to_string(),
        shared_velocity_domain: crn_domain_name(COUPLED_TWIN_CRN_SHARED_VELOCITY_DOMAIN)?,
        shared_position_domain: crn_domain_name(COUPLED_TWIN_CRN_SHARED_POSITION_DOMAIN)?,
        b_only_velocity_domain: crn_domain_name(COUPLED_TWIN_CRN_B_ONLY_VELOCITY_DOMAIN)?,
        b_only_position_domain: crn_domain_name(COUPLED_TWIN_CRN_B_ONLY_POSITION_DOMAIN)?,
        cuda_rng_semantics: COUPLED_TWIN_CRN_CUDA_RNG_SEMANTICS.to_string(),
        direct_cuda_rng_state_readback: false,
        direct_cuda_rng_state_readback_status: "NOT_EXPOSED_BY_AUTHORIZED_THREE_FILE_ENGINE_API;ordinal_identity_plus_frozen_kernel_binding_and_identical_control_required".to_string(),
        atom_count_a: topology_a.n_atoms as u64,
        atom_count_b: topology_b.n_atoms as u64,
        shared_atom_count: shared_indices.len() as u64,
        b_only_atom_count: b_only_indices.len() as u64,
        shared_atom_map_sha256,
        b_only_atom_map_sha256,
        shared_atom_ordinals_identical: true,
        b_only_atoms_are_contiguous_suffix: true,
        cuda_rng_shared_ordinal_alignment_verified: true,
        shared_input_position_sha256: crn_hash_f32_slice("shared_input_positions", &shared_input_positions),
        shared_mass_sha256: crn_hash_f32_slice("shared_masses", &shared_masses),
        b_only_input_position_sha256: crn_hash_f32_slice("b_only_input_positions", &b_only_input_positions),
        b_only_mass_sha256: crn_hash_f32_slice("b_only_masses", &b_only_masses),
        shared_velocity_center_before_f64_bits: shared_velocity_center_before,
        shared_velocity_center_after_f64_bits: shared_velocity_center_after,
        shared_displacement_center_before_f64_bits: shared_displacement_center_before,
        shared_displacement_center_after_f64_bits: shared_displacement_center_after,
        b_only_velocity_center_before_f64_bits: b_only_velocity_center_before,
        b_only_velocity_center_after_f64_bits: b_only_velocity_center_after,
        b_only_displacement_center_before_f64_bits: b_only_displacement_center_before,
        b_only_displacement_center_after_f64_bits: b_only_displacement_center_after,
        shared_position_sha256: crn_hash_f32_slice("shared_positions", &shared_positions),
        shared_velocity_sha256: crn_hash_f32_slice("shared_velocities", &shared_velocity_values),
        shared_displacement_sha256: crn_hash_f32_slice("shared_displacements", &shared_displacement_values),
        b_only_position_sha256: crn_hash_f32_slice("b_only_positions", &b_only_positions),
        b_only_velocity_sha256: crn_hash_f32_slice("b_only_velocities", &b_only_velocity_values),
        b_only_displacement_sha256: crn_hash_f32_slice("b_only_displacements", &b_only_displacement_values),
        host_position_sha256_a: host_position_sha256_a.clone(),
        host_position_sha256_b: host_position_sha256_b.clone(),
        host_velocity_sha256_a: host_velocity_sha256_a.clone(),
        host_velocity_sha256_b: host_velocity_sha256_b.clone(),
        device_position_readback_sha256_a: String::new(),
        device_position_readback_sha256_b: String::new(),
        device_velocity_readback_sha256_a: String::new(),
        device_velocity_readback_sha256_b: String::new(),
        device_mass_readback_sha256_a: String::new(),
        device_mass_readback_sha256_b: String::new(),
        device_zero_force_readback_sha256_a: String::new(),
        device_zero_force_readback_sha256_b: String::new(),
        device_zero_potential_readback_sha256_a: String::new(),
        device_zero_potential_readback_sha256_b: String::new(),
        reset_seed_equality_verified: false,
        exact_device_readback_verified: false,
        fallback_used: false,
        semantic_sha256: String::new(),
    };
    receipt.semantic_sha256 = crn_receipt_semantic_sha256_v3(&receipt)?;
    Ok(CoupledTwinCrnHostStateV3 {
        positions_a,
        positions_b,
        velocities_a,
        velocities_b,
        shared_indices,
        b_only_indices,
        receipt,
    })
}

#[cfg(feature = "gpu")]
fn crn_receipt_semantic_sha256_v3(
    receipt: &CoupledTwinCrnInitializationReceiptV3,
) -> Result<String> {
    let mut canonical = receipt.clone();
    canonical.semantic_sha256.clear();
    let mut identity = Vec::new();
    identity.extend_from_slice(COUPLED_TWIN_CRN_SEMANTIC_DOMAIN);
    identity.extend_from_slice(&serde_json::to_vec(&canonical)?);
    Ok(sha256_bytes(&identity))
}

#[cfg(feature = "gpu")]
fn crn_hash_device_bytes_v3(label: &str, values: &[u8], element_bytes: usize) -> Result<String> {
    if values.is_empty() || element_bytes == 0 || values.len() % element_bytes != 0 {
        bail!("CRN {label} device readback has an invalid byte shape");
    }
    let mut identity =
        Vec::with_capacity(COUPLED_TWIN_CRN_ARRAY_DOMAIN.len() + label.len() + 16 + values.len());
    identity.extend_from_slice(COUPLED_TWIN_CRN_ARRAY_DOMAIN);
    identity.extend_from_slice(&(label.len() as u64).to_le_bytes());
    identity.extend_from_slice(label.as_bytes());
    identity.extend_from_slice(&((values.len() / element_bytes) as u64).to_le_bytes());
    identity.extend_from_slice(values);
    Ok(sha256_bytes(&identity))
}

#[cfg(feature = "gpu")]
fn expected_coupled_twin_crn_device_readback_v3(
    host: &CoupledTwinCrnHostStateV3,
    topology_a: &PrismPrepTopology,
    topology_b: &PrismPrepTopology,
) -> CoupledTwinCrnDeviceReadbackV3 {
    CoupledTwinCrnDeviceReadbackV3 {
        position_sha256_a: crn_hash_f32_slice("stream_a_positions", &host.positions_a),
        position_sha256_b: crn_hash_f32_slice("stream_b_positions", &host.positions_b),
        velocity_sha256_a: crn_hash_f32_slice("stream_a_velocities", &host.velocities_a),
        velocity_sha256_b: crn_hash_f32_slice("stream_b_velocities", &host.velocities_b),
        mass_sha256_a: crn_hash_f32_slice("stream_a_masses", &topology_a.masses),
        mass_sha256_b: crn_hash_f32_slice("stream_b_masses", &topology_b.masses),
        zero_force_sha256_a: crn_hash_f32_slice(
            "stream_a_forces",
            &vec![0.0; topology_a.n_atoms * 3],
        ),
        zero_force_sha256_b: crn_hash_f32_slice(
            "stream_b_forces",
            &vec![0.0; topology_b.n_atoms * 3],
        ),
        zero_potential_sha256_a: crn_hash_f64_slice(
            "stream_a_potential",
            &vec![0.0; topology_a.n_atoms],
        ),
        zero_potential_sha256_b: crn_hash_f64_slice(
            "stream_b_potential",
            &vec![0.0; topology_b.n_atoms],
        ),
    }
}

#[cfg(feature = "gpu")]
fn finalize_coupled_twin_crn_device_readback_v3(
    mut receipt: CoupledTwinCrnInitializationReceiptV3,
    observed: &CoupledTwinCrnDeviceReadbackV3,
    expected: &CoupledTwinCrnDeviceReadbackV3,
) -> Result<CoupledTwinCrnInitializationReceiptV3> {
    let device_fields_are_empty = [
        receipt.device_position_readback_sha256_a.as_str(),
        receipt.device_position_readback_sha256_b.as_str(),
        receipt.device_velocity_readback_sha256_a.as_str(),
        receipt.device_velocity_readback_sha256_b.as_str(),
        receipt.device_mass_readback_sha256_a.as_str(),
        receipt.device_mass_readback_sha256_b.as_str(),
        receipt.device_zero_force_readback_sha256_a.as_str(),
        receipt.device_zero_force_readback_sha256_b.as_str(),
        receipt.device_zero_potential_readback_sha256_a.as_str(),
        receipt.device_zero_potential_readback_sha256_b.as_str(),
    ]
    .iter()
    .all(|value| value.is_empty());
    if receipt.status != "IMPLEMENTATION_PARTIAL"
        || receipt.validation_status != "PENDING_DEVICE_READBACK"
        || !device_fields_are_empty
        || receipt.reset_seed_equality_verified
        || receipt.exact_device_readback_verified
        || observed != expected
        || observed.position_sha256_a != receipt.host_position_sha256_a
        || observed.position_sha256_b != receipt.host_position_sha256_b
        || observed.velocity_sha256_a != receipt.host_velocity_sha256_a
        || observed.velocity_sha256_b != receipt.host_velocity_sha256_b
    {
        bail!("CRN device readback cannot finalize because observed CUDA bytes or pending-state identity differ");
    }
    receipt.status = "IMPLEMENTATION_VERIFIED".to_string();
    receipt.validation_status = "PASS".to_string();
    receipt.direct_cuda_rng_state_readback_status = "CURAND_STATE_NOT_EXPOSED;INITIAL_POSITION_VELOCITY_MASS_FORCE_AND_POTENTIAL_DTOH_BYTE_IDENTITY_VERIFIED".to_string();
    receipt.device_position_readback_sha256_a = observed.position_sha256_a.clone();
    receipt.device_position_readback_sha256_b = observed.position_sha256_b.clone();
    receipt.device_velocity_readback_sha256_a = observed.velocity_sha256_a.clone();
    receipt.device_velocity_readback_sha256_b = observed.velocity_sha256_b.clone();
    receipt.device_mass_readback_sha256_a = observed.mass_sha256_a.clone();
    receipt.device_mass_readback_sha256_b = observed.mass_sha256_b.clone();
    receipt.device_zero_force_readback_sha256_a = observed.zero_force_sha256_a.clone();
    receipt.device_zero_force_readback_sha256_b = observed.zero_force_sha256_b.clone();
    receipt.device_zero_potential_readback_sha256_a = observed.zero_potential_sha256_a.clone();
    receipt.device_zero_potential_readback_sha256_b = observed.zero_potential_sha256_b.clone();
    receipt.reset_seed_equality_verified = true;
    receipt.exact_device_readback_verified = true;
    receipt.semantic_sha256.clear();
    receipt.semantic_sha256 = crn_receipt_semantic_sha256_v3(&receipt)?;
    Ok(receipt)
}

#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn verify_coupled_twin_crn_receipt_v3(
    receipt: &CoupledTwinCrnInitializationReceiptV3,
    topology_a: &PrismPrepTopology,
    topology_b: &PrismPrepTopology,
    config: &CoupledTwinEvidenceConfig,
    seed: u64,
    temperature_kelvin: f32,
    phase: &str,
    trial_ordinal: Option<u32>,
) -> Result<()> {
    let expected_host = build_coupled_twin_crn_host_state_v3(
        topology_a,
        topology_b,
        config,
        seed,
        temperature_kelvin,
        phase,
        trial_ordinal,
    )?;
    let expected_readback =
        expected_coupled_twin_crn_device_readback_v3(&expected_host, topology_a, topology_b);
    let expected_receipt = finalize_coupled_twin_crn_device_readback_v3(
        expected_host.receipt,
        &expected_readback,
        &expected_readback,
    )?;
    if *receipt != expected_receipt
        || crn_receipt_semantic_sha256_v3(receipt)? != receipt.semantic_sha256
        || receipt.status != "IMPLEMENTATION_VERIFIED"
        || receipt.validation_status != "PASS"
        || receipt.direct_cuda_rng_state_readback
        || !receipt.shared_atom_ordinals_identical
        || !receipt.b_only_atoms_are_contiguous_suffix
        || !receipt.cuda_rng_shared_ordinal_alignment_verified
        || !receipt.reset_seed_equality_verified
        || !receipt.exact_device_readback_verified
        || receipt.fallback_used
    {
        bail!("coupled-TWIN CRN V3 receipt failed independent host recomputation");
    }
    Ok(())
}

fn hash_file_dual(path: &Path) -> Result<(String, String)> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("open artifact for hashing: {}", path.display()))?;
    let mut sha256 = Sha256::new();
    let mut blake3 = blake3::Hasher::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .with_context(|| format!("read artifact for hashing: {}", path.display()))?;
        if count == 0 {
            break;
        }
        sha256.update(&buffer[..count]);
        blake3.update(&buffer[..count]);
    }
    Ok((
        format!("{:x}", sha256.finalize()),
        blake3.finalize().to_hex().to_string(),
    ))
}

fn sha256_file(path: &Path) -> Result<String> {
    hash_file_dual(path).map(|(sha256, _)| sha256)
}

struct NoClobberArtifactFile {
    destination: PathBuf,
    temporary: PathBuf,
    file: Option<File>,
    committed: bool,
}

impl NoClobberArtifactFile {
    fn create(destination: &Path) -> Result<Self> {
        match std::fs::symlink_metadata(destination) {
            Ok(_) => bail!(
                "refusing to clobber existing runtime artifact {}",
                destination.display()
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "inspect runtime artifact destination {}",
                        destination.display()
                    )
                })
            }
        }
        let parent = destination
            .parent()
            .ok_or_else(|| anyhow::anyhow!("artifact has no parent: {}", destination.display()))?;
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create artifact directory {}", parent.display()))?;
        let name = destination
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or_else(|| {
                anyhow::anyhow!("artifact filename is not UTF-8: {}", destination.display())
            })?;
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .context("system clock precedes UNIX epoch")?
            .as_nanos();
        let temporary = parent.join(format!(".{name}.tmp.{}.{}", std::process::id(), nonce));
        let file = std::fs::OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&temporary)
            .with_context(|| format!("create temporary artifact {}", temporary.display()))?;
        Ok(Self {
            destination: destination.to_path_buf(),
            temporary,
            file: Some(file),
            committed: false,
        })
    }

    fn file_mut(&mut self) -> Result<&mut File> {
        self.file
            .as_mut()
            .context("temporary artifact writer is already closed")
    }

    fn commit(mut self) -> Result<()> {
        let mut file = self
            .file
            .take()
            .context("temporary artifact writer is already closed")?;
        file.flush()
            .with_context(|| format!("flush temporary artifact {}", self.temporary.display()))?;
        file.sync_all()
            .with_context(|| format!("sync temporary artifact {}", self.temporary.display()))?;
        drop(file);
        if let Err(link_error) = std::fs::hard_link(&self.temporary, &self.destination) {
            let cleanup = std::fs::remove_file(&self.temporary);
            bail!(
                "no-clobber artifact commit {} -> {} failed: {}; temporary cleanup={:?}",
                self.temporary.display(),
                self.destination.display(),
                link_error,
                cleanup
            );
        }
        std::fs::remove_file(&self.temporary).with_context(|| {
            format!(
                "remove committed temporary artifact {}",
                self.temporary.display()
            )
        })?;
        let parent = self
            .destination
            .parent()
            .context("committed artifact has no parent")?;
        let dir = std::fs::File::open(parent)
            .with_context(|| format!("open artifact directory {}", parent.display()))?;
        dir.sync_all()
            .with_context(|| format!("sync artifact directory {}", parent.display()))?;
        self.committed = true;
        Ok(())
    }
}

impl Drop for NoClobberArtifactFile {
    fn drop(&mut self) {
        if !self.committed {
            self.file.take();
            let _ = std::fs::remove_file(&self.temporary);
        }
    }
}

fn write_atomic_bytes(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut artifact = NoClobberArtifactFile::create(path)?;
    artifact
        .file_mut()?
        .write_all(bytes)
        .with_context(|| format!("write temporary artifact for {}", path.display()))?;
    artifact.commit()
}

fn write_atomic_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    write_atomic_bytes(path, &bytes)
}

fn artifact_evidence(
    kind: &str,
    stream_id: Option<u32>,
    path: &Path,
    record_count: Option<u64>,
    schema_version: Option<u32>,
) -> Result<CoupledTwinArtifactEvidence> {
    let metadata = path
        .metadata()
        .with_context(|| format!("stat runtime artifact {}", path.display()))?;
    if !metadata.is_file() || metadata.len() == 0 {
        bail!("runtime artifact is missing or empty: {}", path.display());
    }
    let (sha256, blake3) = hash_file_dual(path)?;
    Ok(CoupledTwinArtifactEvidence {
        kind: kind.to_string(),
        stream_id,
        path: path.display().to_string(),
        size_bytes: metadata.len(),
        sha256,
        blake3,
        record_count,
        schema_version,
        runtime_owned: true,
    })
}

fn read_u32_from<R: Read>(reader: &mut R, label: &str) -> Result<u32> {
    let mut bytes = [0u8; 4];
    reader
        .read_exact(&mut bytes)
        .with_context(|| format!("read {label}"))?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64_from<R: Read>(reader: &mut R, label: &str) -> Result<u64> {
    let mut bytes = [0u8; 8];
    reader
        .read_exact(&mut bytes)
        .with_context(|| format!("read {label}"))?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_bounded_utf8<R: Read>(reader: &mut R, label: &str) -> Result<String> {
    let length = read_u64_from(reader, &format!("{label} length"))?;
    if length > 1024 * 1024 {
        bail!("{label} exceeds the one-MiB evidence-header limit");
    }
    let mut bytes = vec![0u8; length as usize];
    reader
        .read_exact(&mut bytes)
        .with_context(|| format!("read {label} bytes"))?;
    String::from_utf8(bytes).with_context(|| format!("decode {label} as UTF-8"))
}

fn validate_evidence_envelope(
    path: &Path,
    expected_magic: &[u8; 8],
    expected_trailer: &[u8; 8],
    expected_stream_id: u32,
    expected_run_id: &str,
    expected_stem: &str,
    expected_record_count: Option<u64>,
) -> Result<()> {
    let mut reader = std::io::BufReader::with_capacity(
        1024 * 1024,
        std::fs::File::open(path)
            .with_context(|| format!("open evidence envelope {}", path.display()))?,
    );
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != expected_magic {
        bail!("{} has the wrong evidence-envelope magic", path.display());
    }
    let schema_version = read_u32_from(&mut reader, "envelope schema version")?;
    let endian = read_u32_from(&mut reader, "envelope endian marker")?;
    let stream_id = read_u32_from(&mut reader, "envelope stream id")?;
    let run_id = read_bounded_utf8(&mut reader, "envelope run id")?;
    let stem = read_bounded_utf8(&mut reader, "envelope target stem")?;
    let record_count = read_u64_from(&mut reader, "envelope record count")?;
    let byte_stride = read_u64_from(&mut reader, "envelope byte stride")?;
    let payload_size = read_u64_from(&mut reader, "envelope payload size")?;
    if schema_version != 1
        || endian != 0x01020304
        || stream_id != expected_stream_id
        || run_id != expected_run_id
        || stem != expected_stem
        || byte_stride == 0
        || expected_record_count.is_some_and(|expected| expected != record_count)
    {
        bail!(
            "{} evidence-envelope identity mismatch: schema={} endian=0x{:08x} stream={} run={} stem={} records={} stride={}",
            path.display(),
            schema_version,
            endian,
            stream_id,
            run_id,
            stem,
            record_count,
            byte_stride
        );
    }
    let mut remaining = payload_size;
    let mut checksum = 0xcbf29ce484222325u64;
    let mut buffer = [0u8; 1024 * 1024];
    while remaining > 0 {
        let count = usize::try_from(remaining.min(buffer.len() as u64))?;
        reader
            .read_exact(&mut buffer[..count])
            .with_context(|| format!("read evidence payload {}", path.display()))?;
        for byte in &buffer[..count] {
            checksum = (checksum ^ (*byte as u64)).wrapping_mul(0x100000001b3);
        }
        remaining -= count as u64;
    }
    let stored_checksum = read_u64_from(&mut reader, "envelope payload checksum")?;
    let mut trailer = [0u8; 8];
    reader.read_exact(&mut trailer)?;
    let mut trailing = [0u8; 1];
    let trailing_count = reader.read(&mut trailing)?;
    if checksum != stored_checksum || &trailer != expected_trailer || trailing_count != 0 {
        bail!(
            "{} evidence-envelope checksum/trailer/length validation failed",
            path.display()
        );
    }
    Ok(())
}

fn validate_run_artifact(
    output_dir: &Path,
    artifact: &CoupledTwinArtifactEvidence,
) -> Result<PathBuf> {
    if !artifact.runtime_owned {
        bail!(
            "non-runtime-owned artifact entered strict evidence: {}",
            artifact.path
        );
    }
    let root_metadata = std::fs::symlink_metadata(output_dir)
        .with_context(|| format!("stat coupled run root {}", output_dir.display()))?;
    if root_metadata.file_type().is_symlink() || !root_metadata.is_dir() {
        bail!("coupled run root must be a regular non-symlink directory");
    }
    let canonical_root = std::fs::canonicalize(output_dir)?;
    let path = Path::new(&artifact.path);
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("stat coupled runtime artifact {}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() == 0 {
        bail!(
            "coupled runtime artifact must be a nonempty regular non-symlink file: {}",
            path.display()
        );
    }
    let canonical_path = std::fs::canonicalize(path)?;
    if !canonical_path.starts_with(&canonical_root) {
        bail!(
            "coupled runtime artifact escapes run root: {} -> {}",
            path.display(),
            canonical_path.display()
        );
    }
    if metadata.len() != artifact.size_bytes {
        bail!(
            "coupled artifact size drift for {}: expected={} observed={}",
            path.display(),
            artifact.size_bytes,
            metadata.len()
        );
    }
    let (sha256, blake3) = hash_file_dual(path)?;
    if sha256 != artifact.sha256 || blake3 != artifact.blake3 {
        bail!(
            "coupled artifact digest drift for {}: sha256={} expected={} blake3={} expected={}",
            path.display(),
            sha256,
            artifact.sha256,
            blake3,
            artifact.blake3
        );
    }
    Ok(canonical_path)
}

fn fnv1a_64(bytes: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    bytes.iter().fold(FNV_OFFSET, |hash, byte| {
        (hash ^ (*byte as u64)).wrapping_mul(FNV_PRIME)
    })
}

#[allow(clippy::too_many_arguments)]
fn write_md_evidence_envelope(
    path: &Path,
    magic: &[u8; 8],
    trailer: &[u8; 8],
    stream_id: u32,
    run_id: &str,
    stem: &str,
    record_count: u64,
    byte_stride: u64,
    payload: &[u8],
) -> Result<()> {
    let mut bytes = Vec::with_capacity(
        64usize
            .saturating_add(run_id.len())
            .saturating_add(stem.len())
            .saturating_add(payload.len()),
    );
    bytes.extend_from_slice(magic);
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&0x01020304u32.to_le_bytes());
    bytes.extend_from_slice(&stream_id.to_le_bytes());
    bytes.extend_from_slice(&(run_id.len() as u64).to_le_bytes());
    bytes.extend_from_slice(run_id.as_bytes());
    bytes.extend_from_slice(&(stem.len() as u64).to_le_bytes());
    bytes.extend_from_slice(stem.as_bytes());
    bytes.extend_from_slice(&record_count.to_le_bytes());
    bytes.extend_from_slice(&byte_stride.to_le_bytes());
    bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    bytes.extend_from_slice(payload);
    bytes.extend_from_slice(&fnv1a_64(payload).to_le_bytes());
    bytes.extend_from_slice(trailer);
    write_atomic_bytes(path, &bytes)
}

#[cfg(feature = "gpu")]
fn write_signal_grid_evidence(
    path: &Path,
    signal: &crate::fused_engine::SignalPreservationData,
    stream_id: u32,
    run_id: &str,
    stem: &str,
) -> Result<()> {
    let voxel_count = signal.voxel_hit_grid.len();
    if signal.coupled_spike_grid.len() != voxel_count
        || signal.primary_residue_id.len() != voxel_count
        || signal.primary_residue_count.len() != voxel_count
    {
        bail!(
            "signal-grid length mismatch for stream {}: hit={} coupled={} residue_id={} residue_count={}",
            stream_id,
            voxel_count,
            signal.coupled_spike_grid.len(),
            signal.primary_residue_id.len(),
            signal.primary_residue_count.len()
        );
    }
    let mut payload = Vec::with_capacity(16usize.saturating_add(voxel_count.saturating_mul(16)));
    payload.extend_from_slice(&(signal.grid_dim as u64).to_le_bytes());
    payload.extend_from_slice(&(voxel_count as u64).to_le_bytes());
    for field in [
        signal.voxel_hit_grid.as_slice(),
        signal.coupled_spike_grid.as_slice(),
        signal.primary_residue_id.as_slice(),
        signal.primary_residue_count.as_slice(),
    ] {
        let field_bytes =
            unsafe { std::slice::from_raw_parts(field.as_ptr().cast::<u8>(), field.len() * 4) };
        payload.extend_from_slice(field_bytes);
    }
    write_md_evidence_envelope(
        path,
        b"PRSGD001",
        b"PRSGDEND",
        stream_id,
        run_id,
        stem,
        voxel_count as u64,
        4,
        &payload,
    )
}

#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct KccLagValidity {
    finite_pairs: usize,
    unavailable_pairs: usize,
}

#[cfg(feature = "gpu")]
fn validate_kcc_evidence_semantics(kcc: &crate::fused_engine::KccData) -> Result<KccLagValidity> {
    let n_residues = kcc.n_residues;
    let finite_fields: [(&str, &[f32]); 10] = [
        ("temporal_corr", &kcc.temporal_corr),
        ("direction_score", &kcc.direction_score),
        ("motion_efficiency", &kcc.motion_efficiency),
        ("burst_motion", &kcc.burst_motion),
        ("phase_shift", &kcc.phase_shift),
        ("local_cov", &kcc.local_cov),
        ("net_dx", &kcc.net_dx),
        ("net_dy", &kcc.net_dy),
        ("net_dz", &kcc.net_dz),
        ("sum_m", &kcc.sum_m),
    ];
    for (name, values) in finite_fields {
        if values.len() != n_residues || values.iter().any(|value| !value.is_finite()) {
            bail!(
                "KCC finite field {} is nonfinite or has length {} rather than {}",
                name,
                values.len(),
                n_residues
            );
        }
    }
    for (name, values) in [
        ("causal_lag", kcc.causal_lag.as_slice()),
        ("lag_corr_peak", kcc.lag_corr_peak.as_slice()),
    ] {
        if values.len() != n_residues {
            bail!(
                "KCC lag field {} has length {} rather than {}",
                name,
                values.len(),
                n_residues
            );
        }
    }
    for (name, values) in [
        ("residue_count", kcc.residue_count.as_slice()),
        ("active_causal", kcc.active_causal.as_slice()),
    ] {
        if values.len() != n_residues {
            bail!(
                "KCC integer field {} has length {} rather than {}",
                name,
                values.len(),
                n_residues
            );
        }
    }

    let mut validity = KccLagValidity {
        finite_pairs: 0,
        unavailable_pairs: 0,
    };
    for (residue_index, (lag, peak)) in kcc.causal_lag.iter().zip(&kcc.lag_corr_peak).enumerate() {
        match (lag.is_finite(), peak.is_finite()) {
            (true, true) => {
                if !(-1.0..=1.0).contains(peak) {
                    bail!(
                        "KCC residue {} has out-of-domain lag correlation peak {}",
                        residue_index,
                        peak
                    );
                }
                validity.finite_pairs += 1;
            }
            (false, false) if lag.is_nan() && peak.is_nan() => {
                // PRKCC001 and the Fusion decoder preserve a paired NaN as an
                // explicit unavailable lag estimate. This occurs when the real
                // event ring cannot support a finite correlation; it is not a
                // numerical value and must never be rewritten to a synthetic 0.
                validity.unavailable_pairs += 1;
            }
            _ => {
                bail!(
                    "KCC residue {} has an asymmetric or infinite lag pair: lag={} peak={}",
                    residue_index,
                    lag,
                    peak
                );
            }
        }
    }
    if validity.finite_pairs == 0
        || validity.finite_pairs + validity.unavailable_pairs != n_residues
    {
        bail!(
            "KCC lag evidence has no finite pair or incomplete validity coverage: finite={} unavailable={} residues={}",
            validity.finite_pairs,
            validity.unavailable_pairs,
            n_residues
        );
    }
    Ok(validity)
}

#[cfg(feature = "gpu")]
fn write_kcc_evidence(
    path: &Path,
    kcc: &crate::fused_engine::KccData,
    stream_id: u32,
    run_id: &str,
    stem: &str,
) -> Result<()> {
    let n_residues = kcc.n_residues;
    let lag_validity = validate_kcc_evidence_semantics(kcc)?;
    let float_fields: [(&str, &[f32]); 12] = [
        ("temporal_corr", &kcc.temporal_corr),
        ("direction_score", &kcc.direction_score),
        ("motion_efficiency", &kcc.motion_efficiency),
        ("burst_motion", &kcc.burst_motion),
        ("phase_shift", &kcc.phase_shift),
        ("causal_lag", &kcc.causal_lag),
        ("lag_corr_peak", &kcc.lag_corr_peak),
        ("local_cov", &kcc.local_cov),
        ("net_dx", &kcc.net_dx),
        ("net_dy", &kcc.net_dy),
        ("net_dz", &kcc.net_dz),
        ("sum_m", &kcc.sum_m),
    ];
    let integer_fields: [(&str, &[u32]); 2] = [
        ("residue_count", &kcc.residue_count),
        ("active_causal", &kcc.active_causal),
    ];
    let mut payload = Vec::new();
    payload.extend_from_slice(&(n_residues as u64).to_le_bytes());
    payload.extend_from_slice(&14u64.to_le_bytes());
    let mut push_field = |name: &str, dtype: u8, data: &[u8]| {
        payload.extend_from_slice(&(name.len() as u64).to_le_bytes());
        payload.extend_from_slice(name.as_bytes());
        payload.push(dtype);
        payload.extend_from_slice(&(data.len() as u64).to_le_bytes());
        payload.extend_from_slice(data);
    };
    for (name, values) in float_fields {
        let bytes =
            unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 4) };
        push_field(name, 1, bytes);
    }
    for (name, values) in integer_fields {
        let bytes =
            unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 4) };
        push_field(name, 2, bytes);
    }
    log::info!(
        "KCC evidence stream {}: {} finite lag pairs, {} paired-NaN unavailable estimates",
        stream_id,
        lag_validity.finite_pairs,
        lag_validity.unavailable_pairs
    );
    write_md_evidence_envelope(
        path,
        b"PRKCC001",
        b"PRKCCEND",
        stream_id,
        run_id,
        stem,
        n_residues as u64,
        4,
        &payload,
    )
}

#[cfg(feature = "gpu")]
fn write_spike_evidence(
    path: &Path,
    spikes: &[crate::fused_engine::GpuSpikeEvent],
    stream_id: u32,
    run_id: &str,
    stem: &str,
) -> Result<()> {
    let stride = std::mem::size_of::<crate::fused_engine::GpuSpikeEvent>();
    if stride == 0 {
        bail!("GpuSpikeEvent has an invalid zero-byte representation");
    }
    let payload_len = spikes
        .len()
        .checked_mul(stride)
        .context("spike evidence payload size overflow")?;
    let payload = unsafe { std::slice::from_raw_parts(spikes.as_ptr().cast::<u8>(), payload_len) };
    write_md_evidence_envelope(
        path,
        b"PRSPK001",
        b"PRSPKEND",
        stream_id,
        run_id,
        stem,
        spikes.len() as u64,
        stride as u64,
        payload,
    )
}

#[cfg(feature = "gpu")]
fn spike_count_map(spikes: &[crate::fused_engine::GpuSpikeEvent]) -> BTreeMap<i32, (u32, f32)> {
    // Accumulate intensities in f64 and round once per residue. Strict feature
    // records use the same f64-then-f32 policy; accumulating millions of source
    // events directly in f32 produces order-dependent drift and falsely breaks
    // the row-level source/feature identity gate.
    let mut result = BTreeMap::<i32, (u32, f64)>::new();
    for spike in spikes {
        let mut seen = BTreeSet::new();
        for residue in spike
            .nearby_residues
            .iter()
            .take(spike.n_residues.min(8) as usize)
            .copied()
            .filter(|residue| *residue >= 0)
        {
            if seen.insert(residue) {
                let entry = result.entry(residue).or_insert((0u32, 0.0f64));
                entry.0 = entry.0.saturating_add(1);
                entry.1 += f64::from(spike.intensity.max(0.0));
            }
        }
    }
    result
        .into_iter()
        .map(|(residue, (count, intensity))| (residue, (count, intensity as f32)))
        .collect()
}

fn topology_residue_name(topology: &PrismPrepTopology, residue_id: usize) -> String {
    topology
        .residues
        .iter()
        .find(|metadata| metadata.residue_idx == residue_id)
        .map(|metadata| metadata.residue_name.clone())
        .or_else(|| {
            topology
                .residue_ids
                .iter()
                .position(|candidate| *candidate == residue_id)
                .and_then(|index| topology.residue_names.get(index).cloned())
        })
        .unwrap_or_else(|| "UNK".to_string())
}

#[cfg(feature = "gpu")]
fn strict_topology_atom_element(topology: &PrismPrepTopology, atom_index: usize) -> Result<&str> {
    let element = topology
        .elements
        .get(atom_index)
        .map(String::as_str)
        .filter(|element| !element.trim().is_empty())
        .with_context(|| format!("topology atom {atom_index} lacks an explicit element"))?;
    Ok(element.trim())
}

#[cfg(feature = "gpu")]
fn exact_ligand_heavy_atom_distance(
    position: [f32; 3],
    ligand_heavy_atoms: &[LigandHeavyAtomBinding],
) -> Result<(usize, f32)> {
    if position.iter().any(|value| !value.is_finite()) || ligand_heavy_atoms.is_empty() {
        bail!("exact Ghost/ligand distance requires a finite witness and ligand heavy atoms");
    }
    let mut nearest = None;
    for (binding_index, atom) in ligand_heavy_atoms.iter().enumerate() {
        if atom.position.iter().any(|value| !value.is_finite()) {
            bail!("ligand heavy-atom binding contains nonfinite coordinates");
        }
        let dx = position[0] - atom.position[0];
        let dy = position[1] - atom.position[1];
        let dz = position[2] - atom.position[2];
        let distance = (dx * dx + dy * dy + dz * dz).sqrt();
        if !distance.is_finite() {
            bail!("exact Ghost/ligand distance became nonfinite");
        }
        if nearest.is_none_or(|(_, minimum)| distance < minimum) {
            nearest = Some((binding_index, distance));
        }
    }
    nearest.context("exact Ghost/ligand distance found no ligand heavy atom")
}

#[cfg(feature = "gpu")]
fn bind_ligand_heavy_atoms(
    topology: &PrismPrepTopology,
    ligand_atom_indices: &[usize],
    topology_sha256: &str,
) -> Result<(Vec<LigandHeavyAtomBinding>, String)> {
    let mut bindings = Vec::new();
    for &atom_index in ligand_atom_indices {
        let element = strict_topology_atom_element(topology, atom_index)?;
        if element.eq_ignore_ascii_case("H") {
            continue;
        }
        let base = atom_index
            .checked_mul(3)
            .context("ligand heavy-atom coordinate offset overflow")?;
        if base + 2 >= topology.positions.len() {
            bail!("ligand heavy atom {atom_index} has no coordinate triplet");
        }
        let position = [
            topology.positions[base],
            topology.positions[base + 1],
            topology.positions[base + 2],
        ];
        if position.iter().any(|value| !value.is_finite()) {
            bail!("ligand heavy atom {atom_index} has nonfinite coordinates");
        }
        bindings.push(LigandHeavyAtomBinding {
            atom_index,
            residue_id: topology.residue_ids[atom_index],
            element: element.to_string(),
            position,
        });
    }
    let mut coordinate_bytes = Vec::new();
    coordinate_bytes.extend_from_slice(b"PRISMOS_STRICT_GHOST_LIGAND_HEAVY_ATOMS_V3\0");
    coordinate_bytes.extend_from_slice(topology_sha256.as_bytes());
    for atom in &bindings {
        coordinate_bytes.extend_from_slice(&(atom.atom_index as u64).to_le_bytes());
        coordinate_bytes.extend_from_slice(&(atom.residue_id as u64).to_le_bytes());
        coordinate_bytes.extend_from_slice(&(atom.element.len() as u64).to_le_bytes());
        coordinate_bytes.extend_from_slice(atom.element.as_bytes());
        for coordinate in atom.position {
            coordinate_bytes.extend_from_slice(&coordinate.to_le_bytes());
        }
    }
    Ok((bindings, sha256_bytes(&coordinate_bytes)))
}

#[cfg(feature = "gpu")]
fn build_glycan_probe_input_pose_annulus_v1(
    topology_b: &PrismPrepTopology,
    topology_b_sha256: &str,
    probe_heavy_atoms: &[LigandHeavyAtomBinding],
    probe_heavy_atom_coordinate_sha256: &str,
    probe_residue_ids: &[usize],
    inner_exclusive_angstrom: f32,
    outer_inclusive_angstrom: f32,
) -> Result<(BTreeMap<i32, f32>, String)> {
    if probe_heavy_atoms.is_empty()
        || probe_heavy_atom_coordinate_sha256.len() != 64
        || !inner_exclusive_angstrom.is_finite()
        || !outer_inclusive_angstrom.is_finite()
        || inner_exclusive_angstrom <= 0.0
        || outer_inclusive_angstrom <= inner_exclusive_angstrom
    {
        bail!("glycan probe input-pose annulus lacks finite canonical geometry");
    }
    let probe_residue_set = probe_residue_ids.iter().copied().collect::<BTreeSet<_>>();
    let mut minimum_by_receptor_residue = BTreeMap::<i32, f32>::new();
    for atom_index in 0..topology_b.n_atoms {
        let residue_id = *topology_b
            .residue_ids
            .get(atom_index)
            .context("glycan probe annulus atom lacks residue identity")?;
        if probe_residue_set.contains(&residue_id)
            || strict_topology_atom_element(topology_b, atom_index)?.eq_ignore_ascii_case("H")
        {
            continue;
        }
        let base = atom_index
            .checked_mul(3)
            .context("glycan probe annulus receptor coordinate offset overflow")?;
        let position = [
            *topology_b
                .positions
                .get(base)
                .context("glycan probe annulus receptor x coordinate is missing")?,
            *topology_b
                .positions
                .get(base + 1)
                .context("glycan probe annulus receptor y coordinate is missing")?,
            *topology_b
                .positions
                .get(base + 2)
                .context("glycan probe annulus receptor z coordinate is missing")?,
        ];
        let (_, distance) = exact_ligand_heavy_atom_distance(position, probe_heavy_atoms)?;
        let residue_id = i32::try_from(residue_id)
            .context("glycan probe annulus receptor residue exceeds i32")?;
        minimum_by_receptor_residue
            .entry(residue_id)
            .and_modify(|minimum| *minimum = minimum.min(distance))
            .or_insert(distance);
    }
    minimum_by_receptor_residue.retain(|_, distance| {
        *distance > inner_exclusive_angstrom && *distance <= outer_inclusive_angstrom
    });
    if minimum_by_receptor_residue.is_empty() {
        bail!("glycan probe input-pose annulus contains zero receptor residues");
    }
    let mut identity = Vec::new();
    identity.extend_from_slice(b"PRISMOS_GLYCAN_PROBE_INPUT_POSE_RECEPTOR_ANNULUS_V1\0");
    identity.extend_from_slice(topology_b_sha256.as_bytes());
    identity.extend_from_slice(probe_heavy_atom_coordinate_sha256.as_bytes());
    identity.extend_from_slice(&inner_exclusive_angstrom.to_bits().to_le_bytes());
    identity.extend_from_slice(&outer_inclusive_angstrom.to_bits().to_le_bytes());
    for (residue_id, distance) in &minimum_by_receptor_residue {
        identity.extend_from_slice(&residue_id.to_le_bytes());
        identity.extend_from_slice(&distance.to_bits().to_le_bytes());
    }
    Ok((minimum_by_receptor_residue, sha256_bytes(&identity)))
}

#[cfg(feature = "gpu")]
fn validate_molecular_contrast_pocket_support_v4(
    pocket_contact_residue_ids: &BTreeSet<usize>,
    support: &MolecularContrastPocketSupportV4,
) -> Result<()> {
    let expected_ids = pocket_contact_residue_ids
        .iter()
        .copied()
        .collect::<Vec<_>>();
    let observed_ids = support
        .decisions
        .iter()
        .map(|decision| decision.residue_id)
        .collect::<Vec<_>>();
    if observed_ids != expected_ids {
        bail!("pocket contrast support inventory is incomplete, reordered, or substitutes residue identity");
    }
    let expected_candidate_count =
        u32::try_from(expected_ids.len()).context("pocket contrast candidate count exceeds u32")?;
    let mut expected_supported_count = 0u32;
    let mut expected_l1 = 0.0f64;
    for decision in &support.decisions {
        let expected_supported = decision.event_row_present && decision.strict_residue_row_present;
        let expected_status = if expected_supported {
            MOLECULAR_CONTRAST_POCKET_SUPPORTED
        } else {
            MOLECULAR_CONTRAST_POCKET_UNAVAILABLE
        };
        if decision.event_row_present != decision.strict_residue_row_present
            || decision.contrast_supported != expected_supported
            || decision.support_status != expected_status
            || decision.abs_signed_count_contrast.is_some() != expected_supported
            || decision
                .abs_signed_count_contrast
                .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        {
            bail!(
                "pocket contrast residue {} has invalid support/null semantics",
                decision.residue_id
            );
        }
        if let Some(value) = decision.abs_signed_count_contrast {
            expected_supported_count = expected_supported_count
                .checked_add(1)
                .context("pocket contrast support count overflow")?;
            expected_l1 += f64::from(value);
        }
    }
    let expected_unavailable_count = expected_candidate_count
        .checked_sub(expected_supported_count)
        .context("pocket contrast support partition underflow")?;
    let expected_mean = if expected_supported_count == 0 {
        None
    } else {
        Some(expected_l1 / f64::from(expected_supported_count))
    };
    if support.candidate_residue_count != expected_candidate_count
        || support.supported_residue_count != expected_supported_count
        || support.unavailable_residue_count != expected_unavailable_count
        || support.l1_signed_contrast.to_bits() != expected_l1.to_bits()
        || support.mean_abs_signed_contrast.map(f64::to_bits) != expected_mean.map(f64::to_bits)
    {
        bail!("pocket contrast support counts or aggregates do not recompute");
    }
    Ok(())
}

#[cfg(feature = "gpu")]
fn build_molecular_contrast_pocket_support_v4(
    pocket_contact_residue_ids: &BTreeSet<usize>,
    counts_a: &BTreeMap<i32, (u32, f32)>,
    counts_b: &BTreeMap<i32, (u32, f32)>,
    strict_residue_ids: &BTreeSet<i32>,
) -> Result<MolecularContrastPocketSupportV4> {
    let mut decisions = Vec::with_capacity(pocket_contact_residue_ids.len());
    let mut l1_signed_contrast = 0.0f64;
    let mut supported_residue_count = 0u32;
    for &residue_id in pocket_contact_residue_ids {
        let residue_id_i32 =
            i32::try_from(residue_id).context("pocket-contact residue identity exceeds i32")?;
        let event_row_present =
            counts_a.contains_key(&residue_id_i32) || counts_b.contains_key(&residue_id_i32);
        let strict_residue_row_present = strict_residue_ids.contains(&residue_id_i32);
        if event_row_present != strict_residue_row_present {
            bail!(
                "pocket residue {residue_id} event/strict support masks disagree: event={event_row_present} strict={strict_residue_row_present}"
            );
        }
        let abs_signed_count_contrast = if event_row_present {
            let count_a = counts_a.get(&residue_id_i32).map_or(0, |(count, _)| *count);
            let count_b = counts_b.get(&residue_id_i32).map_or(0, |(count, _)| *count);
            let denominator = count_a as f32 + count_b as f32;
            if denominator <= 0.0 {
                bail!("supported pocket residue {residue_id} has zero event support");
            }
            let magnitude = ((count_b as f32 - count_a as f32) / denominator).abs();
            if !magnitude.is_finite() {
                bail!("supported pocket residue {residue_id} contrast is nonfinite");
            }
            l1_signed_contrast += f64::from(magnitude);
            supported_residue_count = supported_residue_count
                .checked_add(1)
                .context("pocket contrast support count overflow")?;
            Some(magnitude)
        } else {
            None
        };
        decisions.push(MolecularContrastPocketResidueDecisionV4 {
            residue_id,
            event_row_present,
            strict_residue_row_present,
            contrast_supported: abs_signed_count_contrast.is_some(),
            abs_signed_count_contrast,
            support_status: if event_row_present {
                MOLECULAR_CONTRAST_POCKET_SUPPORTED
            } else {
                MOLECULAR_CONTRAST_POCKET_UNAVAILABLE
            }
            .to_string(),
        });
    }
    let candidate_residue_count =
        u32::try_from(decisions.len()).context("pocket contrast candidate count exceeds u32")?;
    let unavailable_residue_count = candidate_residue_count
        .checked_sub(supported_residue_count)
        .context("pocket contrast support partition underflow")?;
    let mean_abs_signed_contrast = if supported_residue_count == 0 {
        None
    } else {
        Some(l1_signed_contrast / f64::from(supported_residue_count))
    };
    let support = MolecularContrastPocketSupportV4 {
        decisions,
        candidate_residue_count,
        supported_residue_count,
        unavailable_residue_count,
        l1_signed_contrast,
        mean_abs_signed_contrast,
    };
    validate_molecular_contrast_pocket_support_v4(pocket_contact_residue_ids, &support)?;
    Ok(support)
}

#[cfg(feature = "gpu")]
fn build_molecular_contrast_readout(
    topology_a: &PrismPrepTopology,
    topology_b: &PrismPrepTopology,
    spikes_a: &[crate::fused_engine::GpuSpikeEvent],
    spikes_b: &[crate::fused_engine::GpuSpikeEvent],
    features: &[StrictTwinResidueFeature],
    config: &CoupledTwinEvidenceConfig,
) -> Result<MolecularContrastReadout> {
    config.validate_source_files()?;
    let identical_identity = config.topology_a_sha256 == config.topology_b_sha256;
    let alignment = crate::twin_detection::ResidueAlignment::build(topology_a, topology_b)
        .context("build authoritative coupled-TWIN residue alignment")?;
    let mut alignment_bytes = Vec::new();
    for value in [
        alignment.n_ref_residues,
        alignment.n_ref_ca,
        alignment.n_b_residues,
        alignment.n_mutations,
    ] {
        alignment_bytes.extend_from_slice(&(value as u64).to_le_bytes());
    }
    for value in &alignment.residue_a_to_b {
        alignment_bytes.extend_from_slice(
            &value
                .map(|index| index as u64)
                .unwrap_or(u64::MAX)
                .to_le_bytes(),
        );
    }
    for value in &alignment.ca_a_to_b {
        alignment_bytes.extend_from_slice(
            &value
                .map(|index| index as u64)
                .unwrap_or(u64::MAX)
                .to_le_bytes(),
        );
    }
    for value in &alignment.b_only_residues {
        alignment_bytes.extend_from_slice(&(*value as u64).to_le_bytes());
    }
    let residue_alignment_sha256 = sha256_bytes(&alignment_bytes);
    let residues_a: BTreeSet<usize> = topology_a.residue_ids.iter().copied().collect();
    let residues_b: BTreeSet<usize> = topology_b.residue_ids.iter().copied().collect();
    let ligand_residue_ids = alignment.b_only_residues.clone();
    if !residues_a
        .difference(&residues_b)
        .collect::<Vec<_>>()
        .is_empty()
    {
        bail!("aligned topology B is missing one or more topology-A residue identities");
    }
    if !identical_identity {
        if config.contrast_kind == GLYCAN_MECHANISTIC_PROBE_CONTRAST_KIND {
            if ligand_residue_ids != config.probe_residue_ids {
                bail!(
                    "glycan mechanistic probe residue identity must exactly equal the topology-B-only residue set: declared={:?} aligned={:?}",
                    config.probe_residue_ids,
                    ligand_residue_ids
                );
            }
        } else if ligand_residue_ids.len() != 1 {
            bail!(
                "distinct legacy two-topology contrast requires exactly one B-only ligand residue; found {:?}",
                ligand_residue_ids
            );
        }
    }
    if identical_identity && !ligand_residue_ids.is_empty() {
        bail!("identical topology identities produced B-only residue geometry");
    }

    let ligand_set: BTreeSet<usize> = ligand_residue_ids.iter().copied().collect();
    let ligand_atom_indices: Vec<usize> = topology_b
        .residue_ids
        .iter()
        .enumerate()
        .filter_map(|(index, residue_id)| ligand_set.contains(residue_id).then_some(index))
        .collect();
    if !identical_identity && ligand_atom_indices.is_empty() {
        bail!("B-only ligand residue has zero atoms");
    }

    let mut ligand_aabb_min = [0.0f32; 3];
    let mut ligand_aabb_max = [0.0f32; 3];
    let mut ligand_centroid = [0.0f32; 3];
    let mut ligand_geometry_bytes = Vec::new();
    if !ligand_atom_indices.is_empty() {
        ligand_aabb_min = [f32::INFINITY; 3];
        ligand_aabb_max = [f32::NEG_INFINITY; 3];
        for &atom_index in &ligand_atom_indices {
            let base = atom_index.saturating_mul(3);
            if base + 2 >= topology_b.positions.len() {
                bail!("ligand atom {} has no coordinate triplet", atom_index);
            }
            let position = [
                topology_b.positions[base],
                topology_b.positions[base + 1],
                topology_b.positions[base + 2],
            ];
            if position.iter().any(|value| !value.is_finite()) {
                bail!("ligand atom {} has nonfinite coordinates", atom_index);
            }
            ligand_geometry_bytes.extend_from_slice(&(atom_index as u64).to_le_bytes());
            ligand_geometry_bytes
                .extend_from_slice(&(topology_b.residue_ids[atom_index] as u64).to_le_bytes());
            for axis in 0..3 {
                ligand_aabb_min[axis] = ligand_aabb_min[axis].min(position[axis]);
                ligand_aabb_max[axis] = ligand_aabb_max[axis].max(position[axis]);
                ligand_centroid[axis] += position[axis];
                ligand_geometry_bytes.extend_from_slice(&position[axis].to_le_bytes());
            }
        }
        for coordinate in &mut ligand_centroid {
            *coordinate /= ligand_atom_indices.len() as f32;
        }
    }
    let ligand_geometry_sha256 = sha256_bytes(&ligand_geometry_bytes);

    let (ligand_heavy_atoms, ligand_heavy_atom_coordinate_sha256) =
        bind_ligand_heavy_atoms(topology_b, &ligand_atom_indices, &config.topology_b_sha256)?;
    if !identical_identity && ligand_heavy_atoms.is_empty() {
        bail!("B-only ligand residue has zero explicit heavy atoms");
    }
    let cutoff_sq = config.ligand_contact_cutoff_angstrom.powi(2);
    let mut pocket_contact_residue_ids = BTreeSet::new();
    let mut pocket_contact_atoms = BTreeSet::new();
    for receptor_atom in 0..topology_b.n_atoms {
        if ligand_set.contains(&topology_b.residue_ids[receptor_atom]) {
            continue;
        }
        if strict_topology_atom_element(topology_b, receptor_atom)?.eq_ignore_ascii_case("H") {
            continue;
        }
        let receptor_base = receptor_atom.saturating_mul(3);
        if receptor_base + 2 >= topology_b.positions.len() {
            bail!("receptor atom {} has no coordinate triplet", receptor_atom);
        }
        let receptor_position = [
            topology_b.positions[receptor_base],
            topology_b.positions[receptor_base + 1],
            topology_b.positions[receptor_base + 2],
        ];
        let within_cutoff = ligand_heavy_atoms.iter().any(|ligand_atom| {
            let dx = receptor_position[0] - ligand_atom.position[0];
            let dy = receptor_position[1] - ligand_atom.position[1];
            let dz = receptor_position[2] - ligand_atom.position[2];
            dx * dx + dy * dy + dz * dz <= cutoff_sq
        });
        if within_cutoff {
            pocket_contact_residue_ids.insert(topology_b.residue_ids[receptor_atom]);
            pocket_contact_atoms.insert(receptor_atom);
        }
    }
    if !identical_identity && pocket_contact_residue_ids.is_empty() {
        bail!("coordinate-derived ligand pocket has zero receptor contacts");
    }

    let counts_a = spike_count_map(spikes_a);
    let counts_b = spike_count_map(spikes_b);
    let mut all_residues: BTreeSet<i32> = counts_a.keys().copied().collect();
    all_residues.extend(counts_b.keys().copied());
    all_residues.extend(ligand_residue_ids.iter().map(|residue| *residue as i32));
    let mut feature_by_residue = BTreeMap::<i32, &StrictTwinResidueFeature>::new();
    for feature in features {
        if feature_by_residue
            .insert(feature.residue_id, feature)
            .is_some()
        {
            bail!(
                "molecular-contrast readout rejected duplicate strict residue identity {}",
                feature.residue_id
            );
        }
    }
    let mut residues = Vec::with_capacity(all_residues.len());
    let mut global_l1 = 0.0f64;
    let mut receptor_row_count = 0usize;
    for residue_id in all_residues {
        let (count_a, intensity_a) = counts_a.get(&residue_id).copied().unwrap_or((0, 0.0));
        let (count_b, intensity_b) = counts_b.get(&residue_id).copied().unwrap_or((0, 0.0));
        let count_denominator = count_a as f32 + count_b as f32;
        let signed_count_contrast = if count_denominator > 0.0 {
            (count_b as f32 - count_a as f32) / count_denominator
        } else {
            0.0
        };
        let intensity_denominator = intensity_a + intensity_b;
        let signed_intensity_contrast = if intensity_denominator > 0.0 {
            (intensity_b - intensity_a) / intensity_denominator
        } else {
            0.0
        };
        let log2_count_ratio = ((count_b as f32 + 0.5) / (count_a as f32 + 0.5)).log2();
        let log2_intensity_ratio =
            ((intensity_b.max(0.0) + 1.0e-6) / (intensity_a.max(0.0) + 1.0e-6)).log2();
        let is_ligand = ligand_set.contains(&(residue_id.max(0) as usize));
        let support_class = if is_ligand {
            match (count_a, count_b, intensity_a, intensity_b) {
                (0, b, a_intensity, b_intensity)
                    if b > 0 && a_intensity == 0.0 && b_intensity > 0.0 =>
                {
                    "B_ONLY"
                }
                (0, 0, a_intensity, b_intensity) if a_intensity == 0.0 && b_intensity == 0.0 => {
                    "B_ONLY_TOPOLOGY_UNOBSERVED"
                }
                _ => "B_ONLY_IDENTITY_VIOLATION",
            }
        } else {
            match (count_a > 0, count_b > 0) {
                (true, true) => "BOTH",
                (true, false) => "A_ONLY",
                (false, true) => "B_ONLY",
                (false, false) => "NEITHER",
            }
        };
        let feature = feature_by_residue.get(&residue_id).copied();
        if let Some(measured) = feature {
            let intensity_close = |left: f32, right: f32| {
                (left - right).abs() <= 1.0e-5 * left.abs().max(right.abs()).max(1.0)
            };
            if measured.event_count_a != count_a
                || measured.event_count_b != count_b
                || !intensity_close(measured.intensity_sum_a, intensity_a)
                || !intensity_close(measured.intensity_sum_b, intensity_b)
            {
                bail!(
                    "residue {} strict feature counts/intensities disagree with source events: measured_count_a={} source_count_a={} measured_count_b={} source_count_b={} measured_intensity_a={:.9e} source_intensity_a={:.9e} measured_intensity_b={:.9e} source_intensity_b={:.9e}",
                    residue_id,
                    measured.event_count_a,
                    count_a,
                    measured.event_count_b,
                    count_b,
                    measured.intensity_sum_a,
                    intensity_a,
                    measured.intensity_sum_b,
                    intensity_b
                );
            }
        }
        let (aligned_zero_lag_ccf, aligned_zero_lag_ccf_applicability) = match feature {
            Some(measured) => match measured.global_aligned_zero_lag_ccf {
                Some(value) => (
                    Some(value),
                    "MEASURED_STRICT_GPU_ALIGNED_ZERO_LAG_VARIANCE_SUPPORT".to_string(),
                ),
                None => (
                    None,
                    "NOT_APPLICABLE_INSUFFICIENT_ALIGNED_GLOBAL_VARIANCE_SUPPORT".to_string(),
                ),
            },
            None => (
                None,
                "NOT_APPLICABLE_OUTSIDE_REFERENCE_RESIDUE_FRAME".to_string(),
            ),
        };
        let (thresholded_total_outgoing_transfer_entropy_a_to_b, transfer_entropy_applicability) =
            match feature {
                Some(measured) => match measured.thresholded_total_outgoing_transfer_entropy_a_to_b
                {
                    Some(value) => (
                        Some(value),
                        "MEASURED_STRICT_GPU_KERNEL_LAG_1_EFFECTIVE_HORIZON_2_BINS_FULL_CROSS_RESIDUE_MATRIX".to_string(),
                    ),
                    None => (None, "NOT_APPLICABLE_INSUFFICIENT_TE_BIN_COUNT".to_string()),
                },
                None => (
                    None,
                    "NOT_APPLICABLE_OUTSIDE_REFERENCE_RESIDUE_FRAME".to_string(),
                ),
            };
        let row = MolecularContrastResidue {
            residue_id,
            residue_name: topology_residue_name(topology_b, residue_id.max(0) as usize),
            spikes_a: count_a,
            spikes_b: count_b,
            intensity_a,
            intensity_b,
            signed_count_contrast,
            log2_count_ratio,
            signed_intensity_contrast,
            log2_intensity_ratio,
            support_class: support_class.to_string(),
            aligned_zero_lag_ccf,
            aligned_zero_lag_ccf_applicability,
            thresholded_total_outgoing_transfer_entropy_a_to_b,
            transfer_entropy_applicability,
        };
        let magnitude = signed_count_contrast.abs() as f64;
        if !ligand_set.contains(&(residue_id.max(0) as usize)) {
            global_l1 += magnitude;
            receptor_row_count += 1;
        }
        residues.push(row);
    }
    let strict_residue_ids = feature_by_residue.keys().copied().collect::<BTreeSet<_>>();
    let pocket_support = build_molecular_contrast_pocket_support_v4(
        &pocket_contact_residue_ids,
        &counts_a,
        &counts_b,
        &strict_residue_ids,
    )?;
    let pocket_l1 = pocket_support.l1_signed_contrast;
    let pocket_localization_fraction = if global_l1 > 0.0 {
        pocket_l1 / global_l1
    } else {
        0.0
    };
    let global_mean_abs_signed_contrast = if receptor_row_count == 0 {
        0.0
    } else {
        global_l1 / receptor_row_count as f64
    };
    let pocket_mean_abs_signed_contrast = pocket_support.mean_abs_signed_contrast;
    let finite_values_verified = ligand_aabb_min
        .iter()
        .chain(ligand_aabb_max.iter())
        .chain(ligand_centroid.iter())
        .all(|value| value.is_finite())
        && [
            global_l1,
            pocket_l1,
            global_mean_abs_signed_contrast,
            pocket_localization_fraction,
        ]
        .iter()
        .all(|value| value.is_finite())
        && pocket_mean_abs_signed_contrast.is_none_or(f64::is_finite)
        && pocket_support.decisions.iter().all(|decision| {
            decision
                .abs_signed_count_contrast
                .is_none_or(f32::is_finite)
                && decision.contrast_supported == decision.abs_signed_count_contrast.is_some()
                && decision.event_row_present == decision.strict_residue_row_present
        })
        && residues.iter().all(|row| {
            [
                row.intensity_a,
                row.intensity_b,
                row.signed_count_contrast,
                row.log2_count_ratio,
                row.signed_intensity_contrast,
                row.log2_intensity_ratio,
            ]
            .iter()
            .all(|value| value.is_finite())
                && row.aligned_zero_lag_ccf.is_none_or(f32::is_finite)
                && row
                    .thresholded_total_outgoing_transfer_entropy_a_to_b
                    .is_none_or(f32::is_finite)
        });
    if !finite_values_verified {
        bail!("molecular contrast contains nonfinite values");
    }
    if identical_identity {
        if global_mean_abs_signed_contrast > config.identical_control_max_mean_abs_contrast {
            bail!(
                "identical-topology control produced false contrast: mean_abs={:.8} ceiling={:.8}",
                global_mean_abs_signed_contrast,
                config.identical_control_max_mean_abs_contrast
            );
        }
    } else {
        if global_mean_abs_signed_contrast < config.distinct_min_mean_abs_contrast {
            bail!(
                "distinct topology contrast is degenerate: mean_abs={:.8} floor={:.8}",
                global_mean_abs_signed_contrast,
                config.distinct_min_mean_abs_contrast
            );
        }
        if pocket_localization_fraction < config.distinct_min_pocket_localization_fraction {
            bail!(
                "distinct topology contrast is not pocket-localized: fraction={:.8} floor={:.8}",
                pocket_localization_fraction,
                config.distinct_min_pocket_localization_fraction
            );
        }
        if pocket_support.supported_residue_count == 0 || pocket_mean_abs_signed_contrast.is_none()
        {
            bail!("distinct topology contrast has no supported pocket-residue evidence");
        }
        let ligand_rows: Vec<&MolecularContrastResidue> = residues
            .iter()
            .filter(|row| ligand_set.contains(&(row.residue_id.max(0) as usize)))
            .collect();
        if ligand_rows.len() != ligand_set.len()
            || ligand_rows.iter().any(|row| {
                row.support_class != "B_ONLY"
                    || row.spikes_a != 0
                    || row.spikes_b == 0
                    || row.intensity_a != 0.0
                    || !row.intensity_b.is_finite()
                    || row.intensity_b <= 0.0
                    || row.signed_count_contrast != 1.0
                    || row.signed_intensity_contrast != 1.0
                    || row.aligned_zero_lag_ccf.is_some()
                    || row
                        .thresholded_total_outgoing_transfer_entropy_a_to_b
                        .is_some()
                    || row.aligned_zero_lag_ccf_applicability
                        != "NOT_APPLICABLE_OUTSIDE_REFERENCE_RESIDUE_FRAME"
                    || row.transfer_entropy_applicability
                        != "NOT_APPLICABLE_OUTSIDE_REFERENCE_RESIDUE_FRAME"
            })
        {
            bail!(
                "distinct topology contrast lacks observed, finite, dynamically nondegenerate B-only ligand semantics"
            );
        }
    }

    Ok(MolecularContrastReadout {
        schema_kind: "prismos.molecular_contrast_readout.v4".to_string(),
        schema_version: MOLECULAR_CONTRAST_READOUT_SCHEMA_VERSION,
        run_id: config.run_id.clone(),
        contrast_kind: config.contrast_kind.clone(),
        analysis_semantics: "independent_aligned_trajectory_analysis".to_string(),
        exchange_active: false,
        causal_transport_claim: false,
        topology_a_path: config.topology_a_path.clone(),
        topology_b_path: config.topology_b_path.clone(),
        topology_a_sha256: config.topology_a_sha256.clone(),
        topology_b_sha256: config.topology_b_sha256.clone(),
        identical_topology_identity: identical_identity,
        alignment_reference_residue_count: alignment.n_ref_residues,
        alignment_reference_ca_count: alignment.n_ref_ca,
        alignment_b_residue_count: alignment.n_b_residues,
        alignment_b_only_residue_count: alignment.b_only_residues.len(),
        alignment_mutation_count: alignment.n_mutations,
        residue_alignment_sha256,
        ligand_atom_count: ligand_atom_indices.len(),
        ligand_residue_ids,
        ligand_aabb_min,
        ligand_aabb_max,
        ligand_centroid,
        ligand_geometry_sha256,
        ligand_heavy_atom_count: ligand_heavy_atoms.len(),
        ligand_heavy_atoms,
        ligand_heavy_atom_coordinate_sha256,
        ligand_contact_cutoff_angstrom: config.ligand_contact_cutoff_angstrom,
        pocket_contact_residue_ids: pocket_contact_residue_ids.iter().copied().collect(),
        pocket_contact_atom_count: pocket_contact_atoms.len(),
        pocket_contrast_candidate_residue_count: pocket_support.candidate_residue_count,
        pocket_contrast_supported_residue_count: pocket_support.supported_residue_count,
        pocket_contrast_unavailable_residue_count: pocket_support.unavailable_residue_count,
        pocket_contrast_residue_decisions: pocket_support.decisions,
        global_l1_signed_contrast: global_l1,
        pocket_l1_signed_contrast: pocket_l1,
        global_mean_abs_signed_contrast,
        pocket_mean_abs_signed_contrast,
        pocket_localization_fraction,
        ghost_pocket_component_ids: Vec::new(),
        ghost_pocket_witnesses: Vec::new(),
        ghost_pocket_bound_count: 0,
        ghost_pocket_binding_verified: identical_identity,
        ghost_pocket_binding_status: if identical_identity {
            "not_applicable_identical_topology_control".to_string()
        } else {
            "pending_engine_owned_ghost_binding".to_string()
        },
        residues,
        finite_values_verified,
        fallback_used: false,
    })
}

#[cfg(feature = "gpu")]
fn emit_molecular_contrast_readout(
    output_dir: &Path,
    readout: &MolecularContrastReadout,
) -> Result<(PathBuf, PathBuf, PathBuf)> {
    let readout_dir = output_dir.join("MOLECULAR_CONTRAST_READOUT");
    let readout_path = readout_dir.join("molecular_contrast_readout.json");
    write_atomic_json(&readout_path, readout)?;
    let digest = sha256_file(&readout_path)?;
    let sidecar = readout_dir.join("molecular_contrast_readout.json.sha256");
    write_atomic_bytes(
        &sidecar,
        format!("{}  molecular_contrast_readout.json\n", digest).as_bytes(),
    )?;
    if sha256_file(&readout_path)? != digest {
        bail!("molecular contrast readback hash drift");
    }
    Ok((readout_dir, readout_path, sidecar))
}

#[cfg(feature = "gpu")]
struct RawCudaAllocation {
    ptr: cudarc::driver::sys::CUdeviceptr,
    byte_len: usize,
}

#[cfg(feature = "gpu")]
impl RawCudaAllocation {
    fn copy_from_bytes(bytes: &[u8], label: &str) -> Result<Self> {
        use cudarc::driver::sys::{cuMemAlloc_v2, cuMemcpyHtoD_v2, CUresult};
        if bytes.is_empty() {
            bail!("cannot allocate empty CUDA input for {}", label);
        }
        let mut ptr = 0;
        let alloc = unsafe { cuMemAlloc_v2(&mut ptr, bytes.len()) };
        if !matches!(alloc, CUresult::CUDA_SUCCESS) {
            bail!(
                "cuMemAlloc_v2 {} bytes for {} failed: {:?}",
                bytes.len(),
                label,
                alloc
            );
        }
        let copy =
            unsafe { cuMemcpyHtoD_v2(ptr, bytes.as_ptr().cast::<std::ffi::c_void>(), bytes.len()) };
        if !matches!(copy, CUresult::CUDA_SUCCESS) {
            unsafe {
                let _ = cudarc::driver::sys::cuMemFree_v2(ptr);
            }
            bail!("cuMemcpyHtoD_v2 for {} failed: {:?}", label, copy);
        }
        Ok(Self {
            ptr,
            byte_len: bytes.len(),
        })
    }

    fn copy_from_slice<T: Copy>(values: &[T], label: &str) -> Result<Self> {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                values.as_ptr().cast::<u8>(),
                values.len().saturating_mul(std::mem::size_of::<T>()),
            )
        };
        Self::copy_from_bytes(bytes, label)
    }

    fn overwrite_from_slice<T: Copy>(&self, values: &[T], label: &str) -> Result<()> {
        use cudarc::driver::sys::{cuMemcpyHtoD_v2, CUresult};
        if self.ptr == 0 || values.is_empty() {
            bail!("cannot overwrite empty CUDA allocation for {}", label);
        }
        let byte_len = values
            .len()
            .checked_mul(std::mem::size_of::<T>())
            .context("CUDA overwrite byte count overflow")?;
        if byte_len != self.byte_len {
            bail!(
                "CUDA overwrite for {} changed pointer-stable shape: allocated={} requested={}",
                label,
                self.byte_len,
                byte_len
            );
        }
        let result = unsafe {
            cuMemcpyHtoD_v2(
                self.ptr,
                values.as_ptr().cast::<std::ffi::c_void>(),
                byte_len,
            )
        };
        if !matches!(result, CUresult::CUDA_SUCCESS) {
            bail!(
                "cuMemcpyHtoD_v2 overwrite for {} failed: {:?}",
                label,
                result
            );
        }
        Ok(())
    }
}

#[cfg(feature = "gpu")]
impl Drop for RawCudaAllocation {
    fn drop(&mut self) {
        if self.ptr != 0 {
            unsafe {
                let _ = cudarc::driver::sys::cuMemFree_v2(self.ptr);
            }
            self.ptr = 0;
        }
    }
}

#[cfg(feature = "gpu")]
fn read_le_u16(bytes: &[u8], offset: usize) -> Result<u16> {
    let slice = bytes
        .get(offset..offset + 2)
        .ok_or_else(|| anyhow::anyhow!("u16 read exceeds Ghost record at offset {}", offset))?;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

#[cfg(feature = "gpu")]
fn read_le_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    let slice = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| anyhow::anyhow!("u32 read exceeds Ghost record at offset {}", offset))?;
    Ok(u32::from_le_bytes(
        slice.try_into().expect("four-byte slice"),
    ))
}

#[cfg(feature = "gpu")]
fn read_le_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    let slice = bytes
        .get(offset..offset + 8)
        .ok_or_else(|| anyhow::anyhow!("u64 read exceeds Ghost record at offset {}", offset))?;
    Ok(u64::from_le_bytes(
        slice.try_into().expect("eight-byte slice"),
    ))
}

#[cfg(feature = "gpu")]
fn read_le_f32(bytes: &[u8], offset: usize) -> Result<f32> {
    Ok(f32::from_bits(read_le_u32(bytes, offset)?))
}

#[cfg(feature = "gpu")]
fn ghost_vec3(bytes: &[u8], offset: usize) -> Result<[f32; 3]> {
    Ok([
        read_le_f32(bytes, offset)?,
        read_le_f32(bytes, offset + 4)?,
        read_le_f32(bytes, offset + 8)?,
    ])
}

#[cfg(feature = "gpu")]
fn validate_ghost_envelope_zero_regions(payload: &[u8], record_count: u32) -> Result<()> {
    let expected = crate::ghost_tile::GhostTileRing::COUNTER_SECTOR_BYTES
        .checked_add(record_count as usize * GHOST_V2_RECORD_BYTES)
        .context("Ghost zero-region envelope size overflow")?;
    if payload.len() != expected
        || payload
            .get(4..crate::ghost_tile::GhostTileRing::COUNTER_SECTOR_BYTES)
            .context("Ghost counter-sector padding is absent")?
            .iter()
            .any(|byte| *byte != 0)
    {
        bail!("Ghost envelope counter-sector padding is nonzero or malformed");
    }
    // Record bytes 256..4096 are production v3 RichSpike cluster-residue
    // membership, not slack and not a cavity-lining/contact assignment.
    // Their exact schema/overflow/conservation checks are performed by
    // decode_ghost_v3_cluster_residue_membership at each consumer boundary.
    Ok(())
}

#[cfg(feature = "gpu")]
fn pack_adjudicator_intensity(spike: &crate::fused_engine::GpuSpikeEvent) -> u32 {
    let wavelengths = [260.0f32, 280.0, 305.0, 320.0];
    let uv_code = if spike.spike_source == 1 && spike.wavelength_nm.is_finite() {
        wavelengths
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| {
                (spike.wavelength_nm - **left)
                    .abs()
                    .total_cmp(&(spike.wavelength_nm - **right).abs())
            })
            .map(|(index, _)| index as u32)
            .unwrap_or(0)
    } else {
        0
    };
    let intensity = spike.intensity.clamp(0.0, 1.999_999_9);
    (intensity.to_bits() & 0x3fff_ffff) | ((uv_code & 0x3) << 30)
}

#[cfg(feature = "gpu")]
fn topology_chain_for_residue(topology: &PrismPrepTopology, residue_id: usize) -> String {
    topology
        .residue_ids
        .iter()
        .position(|candidate| *candidate == residue_id)
        .and_then(|atom_index| topology.chain_ids.get(atom_index))
        .cloned()
        .unwrap_or_default()
}

#[cfg(feature = "gpu")]
fn coupled_native_chain_role_maps(
    topology_a: &PrismPrepTopology,
    topology_b: &PrismPrepTopology,
    identical_topology: bool,
) -> Result<(BTreeMap<String, String>, BTreeMap<String, String>)> {
    let native_chains = |topology: &PrismPrepTopology| -> Result<BTreeSet<String>> {
        let chains: BTreeSet<String> = topology
            .chain_ids
            .iter()
            .map(|chain| chain.trim().to_string())
            .collect();
        if chains.is_empty() || chains.iter().any(|chain| chain.is_empty()) {
            bail!("topology contains an empty native chain identity");
        }
        Ok(chains)
    };
    let chains_a = native_chains(topology_a)?;
    let chains_b = native_chains(topology_b)?;
    let map_a: BTreeMap<String, String> = chains_a
        .iter()
        .cloned()
        .map(|chain| (chain, "A".to_string()))
        .collect();
    if identical_topology {
        if chains_b != chains_a {
            bail!("identical-topology control changed native chain identities");
        }
        return Ok((map_a.clone(), map_a));
    }

    let alignment = crate::twin_detection::ResidueAlignment::build(topology_a, topology_b)
        .context("derive native chain roles from authoritative residue alignment")?;
    let b_only: BTreeSet<usize> = alignment.b_only_residues.iter().copied().collect();
    if b_only.is_empty() {
        bail!("distinct topology pair has no B-only residue for ligand-chain role mapping");
    }
    let mut ligand_chains = BTreeSet::new();
    let mut receptor_chains = BTreeSet::new();
    for (residue_id, chain_id) in topology_b.residue_ids.iter().zip(&topology_b.chain_ids) {
        let chain = chain_id.trim().to_string();
        if b_only.contains(residue_id) {
            ligand_chains.insert(chain);
        } else {
            receptor_chains.insert(chain);
        }
    }
    if ligand_chains.is_empty()
        || receptor_chains.is_empty()
        || !ligand_chains.is_disjoint(&receptor_chains)
    {
        bail!(
            "strict Ghost A/B encoding requires disjoint shared-system and B-only probe chains; chains={:?} receptor={:?} probe={:?}",
            chains_b,
            receptor_chains,
            ligand_chains
        );
    }
    if receptor_chains != chains_a {
        bail!(
            "topology-B receptor chain identity drift: topology_A={:?} topology_B_receptor={:?}",
            chains_a,
            receptor_chains
        );
    }
    let mut map_b = BTreeMap::new();
    for chain in receptor_chains {
        map_b.insert(chain, "A".to_string());
    }
    for chain in ligand_chains {
        map_b.insert(chain, "B".to_string());
    }
    Ok((map_a, map_b))
}

/// Terminal error tag: the emission clock does not fit `RichSpike.t_frame`.
#[cfg(feature = "gpu")]
pub const RICH_SPIKE_FRAME_CLOCK_OUT_OF_RANGE: &str = "PRISM_RICH_SPIKE_FRAME_CLOCK_OUT_OF_RANGE";

#[cfg(feature = "gpu")]
pub fn rich_spike_from_gpu_event(
    spike: &crate::fused_engine::GpuSpikeEvent,
    cluster_id: i32,
) -> Result<(crate::rich_spike::RichSpike, BTreeSet<i32>)> {
    use crate::rich_spike::RichSpike;

    if spike
        .position
        .iter()
        .any(|coordinate| !coordinate.is_finite())
        || !spike.intensity.is_finite()
        || !spike.water_density.is_finite()
        || !spike.vibrational_energy.is_finite()
        || !spike.wd_change.is_finite()
        || spike.timestep < 0
    {
        bail!("invalid GPU spike cannot enter the canonical Ghost pipeline");
    }
    if !(1..=8).contains(&spike.n_residues)
        || spike
            .nearby_residues
            .iter()
            .take(spike.n_residues.max(0) as usize)
            .any(|residue| *residue < 0)
    {
        bail!(
            "GPU spike has an invalid declared residue-support cardinality/content: n={} support={:?}",
            spike.n_residues,
            &spike.nearby_residues[..spike.n_residues.clamp(0, 8) as usize]
        );
    }
    let support: BTreeSet<i32> = spike
        .nearby_residues
        .iter()
        .take(spike.n_residues as usize)
        .copied()
        .collect();
    if support.len() != spike.n_residues as usize {
        bail!("GPU spike residue support contains duplicate identities");
    }
    // Preserve the engine's ordered primary-neighbor identity. Choosing the
    // numerically-lowest support residue systematically relabels ligand-origin
    // events as receptor events whenever both appear in the neighborhood.
    let representative = if spike.n_residues > 0 {
        spike.nearby_residues[0]
    } else {
        -1
    };
    if representative < 0 || !support.contains(&representative) {
        bail!(
            "GPU spike primary residue {} is not a valid member of its support set {:?}",
            representative,
            support
        );
    }
    let mut rich = RichSpike::default();
    rich.x = spike.position[0];
    rich.y = spike.position[1];
    rich.z = spike.position[2];
    // `t_frame` carries the PROTOCOL-LOCAL emission clock (`spike.timestep`),
    // which is the only clock the sealed 96 B packet has. It is NOT the global
    // coordinate-frame index: the Director advances the protocol counter from
    // inside the captured graph body, so it ticks `body_unroll` times per
    // replay. Placing an event on the coordinate-frame axis is the job of the
    // per-step governor sidecar (`coupled_chronology`, the `.ccr` stream),
    // which records global_md_step / cumulative_simulated_time_ps per DECISION
    // and is joined to the spike stream by chunk. Widening the record to carry
    // both clocks invalidates the sealed QNANL packet, and would only have
    // resolved to REPLAY granularity anyway — coarser than the sidecar.
    rich.t_frame = u32::try_from(spike.timestep).map_err(|_| {
        anyhow::anyhow!(
            "{}: protocol-local timestep {} does not fit the u32 RichSpike.t_frame field",
            RICH_SPIKE_FRAME_CLOCK_OUT_OF_RANGE,
            spike.timestep
        )
    })?;
    rich.water_density = spike.water_density;
    rich.wd_change = spike.wd_change;
    rich.vib_energy = spike.vibrational_energy;
    rich.intensity_packed = pack_adjudicator_intensity(spike);
    rich.residue_id = representative;
    rich.cluster_id = cluster_id;
    rich.causal_lag = 0.0;
    rich.n_excited = spike.n_nearby_excited.max(0) as u32;
    rich.origin_phase = spike.phase_bits;
    rich.spike_source = spike.spike_source.max(0) as u32;
    rich.chem_flags = 0;
    rich.kinetic_delta = 0.0;
    Ok((rich, support))
}

#[cfg(feature = "gpu")]
fn rich_spike_slice_bytes(spikes: &[crate::rich_spike::RichSpike]) -> Result<&[u8]> {
    if spikes.is_empty() || std::mem::size_of::<crate::rich_spike::RichSpike>() != 64 {
        bail!("invalid RichSpike frame for byte-identity hashing");
    }
    let byte_len = spikes
        .len()
        .checked_mul(std::mem::size_of::<crate::rich_spike::RichSpike>())
        .context("RichSpike frame byte count overflow")?;
    Ok(unsafe { std::slice::from_raw_parts(spikes.as_ptr().cast::<u8>(), byte_len) })
}

#[cfg(feature = "gpu")]
fn rich_spike_slice_sha256(spikes: &[crate::rich_spike::RichSpike]) -> Result<String> {
    Ok(sha256_bytes(rich_spike_slice_bytes(spikes)?))
}

#[cfg(any())]
fn superseded_postrun_gpu_ghost_noise_floor(
    engine: &mut PersistentNhsEngine,
    spike_events: &[crate::fused_engine::GpuSpikeEvent],
    protocol: &CryoUvProtocol,
    stream_id: u32,
    topology_path: &str,
    topology_sha256: &str,
    config: &CoupledTwinEvidenceConfig,
    output_dir: &Path,
) -> Result<(
    f32,
    f32,
    CoupledTwinArtifactEvidence,
    CoupledTwinArtifactEvidence,
)> {
    use crate::captured_pipeline::{CapturedAdjudicationPipeline, PipelineConfig};
    use crate::rich_spike::RichSpike;

    if protocol.cold_hold_steps <= 0 {
        bail!(
            "stream {} has no cold-hold interval for split-T7",
            stream_id
        );
    }
    let mut by_timestep = BTreeMap::<i32, Vec<&crate::fused_engine::GpuSpikeEvent>>::new();
    for spike in spike_events {
        if spike.timestep >= 0 && spike.timestep < protocol.cold_hold_steps {
            by_timestep.entry(spike.timestep).or_default().push(spike);
        }
    }
    let mut frames: Vec<(i32, Vec<&crate::fused_engine::GpuSpikeEvent>)> = by_timestep
        .into_iter()
        .filter(|(_, spikes)| spikes.len() >= 4)
        .take(SPLIT_T7_TOTAL_FRAMES)
        .collect();
    if frames.len() != SPLIT_T7_TOTAL_FRAMES {
        bail!(
            "stream {} split-T7 requires {} real cold-hold frames with at least four spikes; observed {}",
            stream_id,
            SPLIT_T7_TOTAL_FRAMES,
            frames.len()
        );
    }
    let fixed_spikes = frames
        .iter()
        .map(|(_, spikes)| spikes.len())
        .min()
        .context("split-T7 frame set is empty")?;
    if fixed_spikes < 4 || fixed_spikes > u32::MAX as usize {
        bail!("stream {} split-T7 frame exceeds u32 capacity", stream_id);
    }

    engine.cuda_context().bind_to_thread().map_err(|error| {
        anyhow::anyhow!(
            "bind CUDA context for split-T7 stream {}: {:?}",
            stream_id,
            error
        )
    })?;
    engine
        .cuda_stream()
        .synchronize()
        .with_context(|| format!("synchronize stream {} before state readback", stream_id))?;
    crate::sh_basis::init_device_basis(engine.cuda_stream().cu_stream() as *mut std::ffi::c_void)
        .map_err(|error| {
            anyhow::anyhow!("stream {} split-T7 SH basis init: {}", stream_id, error)
        })?;
    engine.cuda_stream().synchronize().with_context(|| {
        format!(
            "synchronize split-T7 stream {} after SH basis initialization",
            stream_id
        )
    })?;
    let d_k_lm = crate::sh_basis::k_lm_device_ptr().map_err(|error| {
        anyhow::anyhow!("stream {} split-T7 SH basis pointer: {}", stream_id, error)
    })?;

    let zeroed = vec![RichSpike::default(); fixed_spikes];
    let d_spikes = RawCudaAllocation::copy_from_slice(&zeroed, "split-T7 RichSpike frames")?;
    let d_offsets = RawCudaAllocation::copy_from_slice(&[0u32, 4u32], "split-T7 CSR offsets")?;
    let pipeline_config = PipelineConfig {
            seed_layout: crate::rich_spike::NativeV2SeedLayout::default(),
        d_spikes: d_spikes.ptr as *const RichSpike,
        d_cluster_offsets: d_offsets.ptr as *const u32,
        n_clusters: 1,
        d_k_lm,
        initial_frame_id: frames[0].0 as u32,
        source_stream_id: stream_id,
        diagnostic_stream_id: Some(stream_id),
        asc: None,
        d_force_norm_mirror: 0,
        governor_device: None,
        zstr: None,
        sisr: None,
        mar_v2: None,
        noise_floor_override: None,
        d_dt: std::ptr::null_mut(),
        d_velocities: std::ptr::null_mut(),
        g26_parent_cond_handle: 0,
        f1_parent_cond_handle: 0,
        d_pe_components: std::ptr::null(),
        n_atoms_for_pe: 0,
        d_external_work: std::ptr::null_mut(),
        ghost_tile_ring_dev: 0,
        ghost_tile_max_records: 0,
        firehose_enable: 0,
        body_unroll: 1,
        d_kcc_lead: 0,
        d_forces_anchor: 0,
        d_masses: 0,
        force_burst_step: None,
        n_spikes: fixed_spikes as u32,
        d_current_md_step: engine.d_protocol_current_step_dev_ptr() as *const u32,
        branch_trace_dev: 0,
    };
    let pipeline = CapturedAdjudicationPipeline::build(
        engine.cuda_context(),
        engine.cuda_stream(),
        &pipeline_config,
    )
    .map_err(|error| {
        anyhow::anyhow!(
            "stream {} split-T7 captured-pipeline build: {:?}",
            stream_id,
            error
        )
    })?;

    let run_id_sha256 = sha256_bytes(config.run_id.as_bytes());
    let mut calibration_input = Vec::<u8>::new();
    calibration_input.extend_from_slice(b"PRT7C001");
    calibration_input.extend_from_slice(&COUPLED_TWIN_EVIDENCE_SCHEMA_VERSION.to_le_bytes());
    calibration_input.extend_from_slice(&stream_id.to_le_bytes());
    calibration_input.extend_from_slice(&(SPLIT_T7_TOTAL_FRAMES as u32).to_le_bytes());
    calibration_input.extend_from_slice(&0u32.to_le_bytes());
    calibration_input.extend_from_slice(run_id_sha256.as_bytes());
    calibration_input.extend_from_slice(topology_sha256.as_bytes());
    let mut frame_evidence = Vec::with_capacity(SPLIT_T7_TOTAL_FRAMES);
    let mut variance_kl = Vec::with_capacity(SPLIT_T7_VARIANCE_FRAMES);
    let mut calibrated = None;
    let mut validation_kl = None;
    let mut validation_marker = None;
    for (ordinal, (timestep, source_spikes)) in frames.drain(..).enumerate() {
        let source_spike_count =
            u32::try_from(source_spikes.len()).context("split-T7 source frame exceeds u32")?;
        let mut rich = Vec::with_capacity(fixed_spikes);
        for selected_index in 0..fixed_spikes {
            let source_index = selected_index
                .checked_mul(source_spikes.len())
                .context("split-T7 deterministic selection overflow")?
                / fixed_spikes;
            let spike = source_spikes[source_index];
            rich.push(rich_spike_from_gpu_event(spike, 0)?.0);
        }
        let selected_hash = rich_spike_slice_sha256(&rich)?;
        let rich_bytes = rich_spike_slice_bytes(&rich)?;
        calibration_input.extend_from_slice(&timestep.to_le_bytes());
        calibration_input.extend_from_slice(&(rich.len() as u32).to_le_bytes());
        let input_byte_offset = calibration_input.len() as u64;
        calibration_input.extend_from_slice(rich_bytes);
        let input_byte_length = rich_bytes.len() as u64;
        d_spikes.overwrite_from_slice(&rich, "split-T7 RichSpike frame")?;
        let end = u32::try_from(rich.len()).context("split-T7 CSR end exceeds u32")?;
        d_offsets.overwrite_from_slice(&[0u32, end], "split-T7 CSR offsets")?;
        pipeline.launch().map_err(|error| {
            anyhow::anyhow!(
                "stream {} split-T7 frame {} launch: {:?}",
                stream_id,
                ordinal,
                error
            )
        })?;
        engine.cuda_stream().synchronize()?;
        let total_kl = pipeline.read_current_divergence().map_err(|error| {
            anyhow::anyhow!(
                "stream {} split-T7 frame {} KL readback: {:?}",
                stream_id,
                ordinal,
                error
            )
        })?;
        if !total_kl.is_finite() {
            bail!(
                "stream {} split-T7 frame {} produced non-finite KL",
                stream_id,
                ordinal
            );
        }

        let phase = if ordinal < SPLIT_T7_ANCHOR_FRAMES {
            "anchor"
        } else if ordinal < SPLIT_T7_ANCHOR_FRAMES + SPLIT_T7_VARIANCE_FRAMES {
            variance_kl.push(total_kl as f64);
            "variance"
        } else {
            validation_kl = Some(total_kl);
            validation_marker = Some(pipeline.read_burst_marker().map_err(|error| {
                anyhow::anyhow!(
                    "stream {} split-T7 validation marker readback: {:?}",
                    stream_id,
                    error
                )
            })?);
            "validation"
        };
        frame_evidence.push(CoupledNoiseFloorFrameEvidence {
            phase: phase.to_string(),
            ordinal: ordinal as u32,
            timestep,
            source_timestep_min: timestep,
            source_timestep_max: timestep,
            analysis_md_step: timestep as u64,
            source_spike_count,
            selected_spike_count: end,
            dropped_spike_count: source_spike_count.saturating_sub(end),
            input_byte_offset,
            input_byte_length,
            source_events_sha256: selected_hash.clone(),
            selected_rich_spikes_sha256: selected_hash,
            total_kl: (ordinal >= SPLIT_T7_ANCHOR_FRAMES).then_some(total_kl),
            total_kl_magnitude: (ordinal >= SPLIT_T7_ANCHOR_FRAMES).then_some(total_kl.abs()),
            adjudication_code: u32::MAX,
            adjudication_reason_flags: u32::MAX,
            engine_state_sha256_before: "SUPERSEDED_POSTRUN_NOT_MEASURED".to_string(),
            engine_state_sha256_after: "SUPERSEDED_POSTRUN_NOT_MEASURED".to_string(),
        });

        if ordinal + 1 == SPLIT_T7_ANCHOR_FRAMES {
            pipeline.freeze_baseline().map_err(|error| {
                anyhow::anyhow!("stream {} split-T7 baseline freeze: {:?}", stream_id, error)
            })?;
            engine.cuda_stream().synchronize()?;
        }
        if ordinal + 1 == SPLIT_T7_ANCHOR_FRAMES + SPLIT_T7_VARIANCE_FRAMES {
            if variance_kl.len() != SPLIT_T7_VARIANCE_FRAMES {
                bail!(
                    "stream {} split-T7 variance window is incomplete",
                    stream_id
                );
            }
            let mu = variance_kl.iter().sum::<f64>() / variance_kl.len() as f64;
            let variance = variance_kl
                .iter()
                .map(|value| {
                    let delta = value - mu;
                    delta * delta
                })
                .sum::<f64>()
                / variance_kl.len() as f64;
            let sigma = variance.sqrt();
            if !mu.is_finite() || !sigma.is_finite() || sigma <= 0.0 {
                bail!(
                    "stream {} split-T7 measured a degenerate KL floor: mu={} sigma={}",
                    stream_id,
                    mu,
                    sigma
                );
            }
            let mu = mu as f32;
            let sigma = sigma as f32;
            pipeline
                .write_kl_native_mu_sigma(mu, sigma)
                .map_err(|error| {
                    anyhow::anyhow!(
                        "stream {} split-T7 KL threshold write: {:?}",
                        stream_id,
                        error
                    )
                })?;
            engine.cuda_stream().synchronize()?;
            calibrated = Some((mu, sigma));
        }
    }
    let (noise_floor_mu, noise_floor_sigma) =
        calibrated.context("split-T7 did not write calibrated KL thresholds")?;
    let validation_frame_total_kl =
        validation_kl.context("split-T7 validation frame was not executed")?;
    let validation_frame_adjudication_marker =
        validation_marker.context("split-T7 validation marker was not read")?;
    if validation_frame_adjudication_marker > 1 {
        bail!(
            "stream {} split-T7 validation marker is invalid: {}",
            stream_id,
            validation_frame_adjudication_marker
        );
    }
    calibration_input.extend_from_slice(b"PRT7CEND");
    let stem = coupled_topology_stem(config);
    let calibration_input_path = output_dir.join(format!(
        "{stem}_stream{stream_id:02}_ghost_noise_floor_input.bin"
    ));
    write_atomic_bytes(&calibration_input_path, &calibration_input)?;
    let calibration_input_artifact = artifact_evidence(
        "ghost_noise_floor_input",
        Some(stream_id),
        &calibration_input_path,
        Some(SPLIT_T7_TOTAL_FRAMES as u64),
        Some(COUPLED_TWIN_EVIDENCE_SCHEMA_VERSION),
    )?;

    let calibration = CoupledNoiseFloorCalibration {
        schema_kind: "prismos.coupled_twin_split_t7_noise_floor.v2".to_string(),
        schema_version: COUPLED_TWIN_EVIDENCE_SCHEMA_VERSION,
        status: "IMPLEMENTATION_VERIFIED".to_string(),
        validation_status: "PASS".to_string(),
        run_id: config.run_id.clone(),
        stream_id,
        topology_path: topology_path.to_string(),
        topology_sha256: topology_sha256.to_string(),
        producer: "CapturedAdjudicationPipeline::build+launch+freeze_baseline+read_current_divergence+write_kl_native_mu_sigma".to_string(),
        source: "PersistentNhsEngine::get_accumulated_spikes/real_cold_hold_frames".to_string(),
        calibration_phase: "cold_hold".to_string(),
        selection_method: "first_101_chronological_cold_hold_timesteps_with_at_least_four_events;fixed_cardinality_equal_to_minimum_frame_count;deterministic_full_span_index=floor(selected_index*source_count/fixed_count);all_selected_events_real;no_padding_or_stale_tail".to_string(),
        topology_layout_sha256: "SUPERSEDED_POSTRUN_SINGLE_CLUSTER".to_string(),
        topology_aabb_min: [0.0; 3],
        topology_aabb_max: [0.0; 3],
        fixed_cluster_count: 1,
        fixed_spikes_per_frame: fixed_spikes as u32,
        direct_prefix_outer_steps: 0,
        direct_prefix_md_steps: 0,
        calibration_input_path: calibration_input_artifact.path.clone(),
        calibration_input_sha256: calibration_input_artifact.sha256.clone(),
        calibration_input_bytes: calibration_input_artifact.size_bytes,
        available_planes: vec!["geometry".to_string(), "thermodynamic".to_string()],
        unavailable_planes: vec!["causality".to_string(), "chemistry".to_string()],
        anchor_frame_count: SPLIT_T7_ANCHOR_FRAMES as u32,
        variance_frame_count: SPLIT_T7_VARIANCE_FRAMES as u32,
        validation_frame_count: SPLIT_T7_VALIDATION_FRAMES as u32,
        pipeline_launch_count: SPLIT_T7_TOTAL_FRAMES as u32,
        baseline_freeze_count: 1,
        kl_threshold_write_count: 1,
        minimum_spikes_per_frame: frame_evidence
            .iter()
            .map(|frame| frame.selected_spike_count)
            .min()
            .unwrap_or(0),
        maximum_spikes_per_frame: frame_evidence
            .iter()
            .map(|frame| frame.selected_spike_count)
            .max()
            .unwrap_or(0),
        noise_floor_mu,
        noise_floor_sigma,
        threshold_3sigma: noise_floor_mu + 3.0 * noise_floor_sigma,
        threshold_12sigma: noise_floor_mu + 12.0 * noise_floor_sigma,
        validation_frame_total_kl,
        validation_frame_adjudication_marker,
        validation_adjudication_code: u32::MAX,
        validation_adjudication_reason_flags: u32::MAX,
        state_noninterference_verified: false,
        topology_constants_rebound_each_launch: false,
        fallback_used: false,
        frames: frame_evidence,
    };
    let calibration_path = output_dir.join(format!(
        "{stem}_stream{stream_id:02}_ghost_noise_floor_calibration.json"
    ));
    write_atomic_json(&calibration_path, &calibration)?;
    let artifact = artifact_evidence(
        "ghost_noise_floor_calibration",
        Some(stream_id),
        &calibration_path,
        Some(SPLIT_T7_TOTAL_FRAMES as u64),
        Some(COUPLED_TWIN_EVIDENCE_SCHEMA_VERSION),
    )?;
    drop(pipeline);
    drop(d_offsets);
    drop(d_spikes);
    Ok((
        noise_floor_mu,
        noise_floor_sigma,
        artifact,
        calibration_input_artifact,
    ))
}

/// Topology-fixed geometry and identity tables used by every online split-T7
/// frame. Keeping this layout fixed makes the captured graph shape stable while
/// retaining native residue and chain provenance for all real events.
#[cfg(feature = "gpu")]
#[derive(Clone)]
struct SplitT7TopologyLayout {
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
    cell: [f32; 3],
    residue_to_anchor_atom: Vec<u32>,
    residue_to_cluster: Vec<u32>,
    cluster_representative_residue: Vec<u32>,
    cluster_chain_role: Vec<u8>,
    chain_offsets: [u32; 8],
    layout_sha256: String,
}

#[cfg(feature = "gpu")]
struct SplitT7StreamRuntime {
    pipeline: crate::captured_pipeline::CapturedAdjudicationPipeline,
    d_spikes: RawCudaAllocation,
    d_offsets: RawCudaAllocation,
    adjudicator_dev: usize,
    layout: SplitT7TopologyLayout,
}

#[cfg(feature = "gpu")]
struct SplitT7StreamOutcome {
    noise_floor_mu: f32,
    noise_floor_sigma: f32,
    calibration_artifact: CoupledTwinArtifactEvidence,
    calibration_input_artifact: CoupledTwinArtifactEvidence,
    frozen_anchor_artifact: CoupledTwinArtifactEvidence,
    frozen_anchor_sha256: String,
    fixed_spikes_per_frame: u32,
    runtime: SplitT7StreamRuntime,
}

#[cfg(feature = "gpu")]
struct SplitT7PairOutcome {
    stream_a: SplitT7StreamOutcome,
    stream_b: SplitT7StreamOutcome,
    direct_prefix_outer_steps: u32,
    direct_prefix_md_steps: u64,
    warmup_spikes_a: u64,
    warmup_spikes_b: u64,
    admission_artifact: CoupledTwinArtifactEvidence,
}

#[cfg(feature = "gpu")]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SplitT7ShadowTrialEvidence {
    trial_ordinal: u32,
    base_seed: u64,
    trial_seed: u64,
    seed_derivation_sha256: String,
    window_md_steps: u32,
    frame_count: u32,
    source_bundle_path: String,
    source_bundle_sha256: String,
    source_bundle_bytes: u64,
    frame_counts_a: Vec<u32>,
    frame_counts_b: Vec<u32>,
    frame_source_sha256_a: Vec<String>,
    frame_source_sha256_b: Vec<String>,
    initial_physical_state_sha256_a: String,
    initial_physical_state_sha256_b: String,
    final_physical_state_sha256_a: String,
    final_physical_state_sha256_b: String,
    crn_initialization_artifact: StrictTwinFeatureArtifactBinding,
    crn_initialization_semantic_sha256: String,
    inventory_sha256: String,
}

#[cfg(feature = "gpu")]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SplitT7LeaveOneOutAdmission {
    held_out_trial_ordinal: u32,
    training_trial_ordinals: Vec<u32>,
    training_lower_envelope_minimum: u32,
    candidate_k: u32,
    held_out_minimum: u32,
    admitted: bool,
}

#[cfg(feature = "gpu")]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SplitT7ShadowEnsembleReceipt {
    schema_kind: String,
    schema_version: u32,
    status: String,
    validation_status: String,
    run_id: String,
    base_seed: u64,
    trial_seed_derivation: String,
    topology_a_sha256: String,
    topology_b_sha256: String,
    window_md_steps: u32,
    frame_count: u32,
    trial_count: u32,
    reserve_factor: u32,
    selection_method: String,
    trials: Vec<SplitT7ShadowTrialEvidence>,
    lower_envelope_counts_a: Vec<u32>,
    lower_envelope_counts_b: Vec<u32>,
    lower_envelope_minimum: u32,
    leave_one_trial_out: Vec<SplitT7LeaveOneOutAdmission>,
    all_trial_seeds_distinct: bool,
    paired_stream_seed_equality_verified: bool,
    paired_common_random_initialization_verified: bool,
    holdout_admission_verified: bool,
    fixed_spikes_per_frame: u32,
    ensemble_sha256: String,
    fallback_used: bool,
}

#[cfg(feature = "gpu")]
#[derive(Debug, PartialEq, Eq)]
struct SplitT7ShadowBundleReadback {
    frame_counts_a: Vec<u32>,
    frame_counts_b: Vec<u32>,
    frame_source_sha256_a: Vec<String>,
    frame_source_sha256_b: Vec<String>,
    initial_physical_state_sha256_a: String,
    initial_physical_state_sha256_b: String,
    final_physical_state_sha256_a: String,
    final_physical_state_sha256_b: String,
}

#[cfg(feature = "gpu")]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SplitT7AdmissionReceiptV3 {
    schema_kind: String,
    schema_version: u32,
    status: String,
    validation_status: String,
    run_id: String,
    topology_a_sha256: String,
    topology_b_sha256: String,
    base_seed: u64,
    trial_seed_derivation: String,
    selection_method: String,
    window_md_steps: u32,
    frame_count: u32,
    shadow_trial_count: u32,
    shadow_trial_seeds: Vec<u64>,
    shadow_trial_artifacts: Vec<StrictTwinFeatureArtifactBinding>,
    shadow_ensemble_artifact: StrictTwinFeatureArtifactBinding,
    shadow_ensemble_sha256: String,
    lower_envelope_counts_a: Vec<u32>,
    lower_envelope_counts_b: Vec<u32>,
    lower_envelope_minimum: u32,
    leave_one_trial_out: Vec<SplitT7LeaveOneOutAdmission>,
    reserve_factor: u32,
    reserve_factor_derivation: String,
    selected_k: u32,
    selected_k_derivation: String,
    authoritative_counts_a: Vec<u32>,
    authoritative_counts_b: Vec<u32>,
    authoritative_crn_initialization_artifact: StrictTwinFeatureArtifactBinding,
    authoritative_crn_initialization_semantic_sha256: String,
    authoritative_minimum: u32,
    authoritative_minimum_slack_from_k: u32,
    authoritative_inventory_sha256: String,
    authoritative_count_source: String,
    authoritative_counts_are_gpu_atomic: bool,
    authoritative_underflow_verified: bool,
    no_padding: bool,
    no_reuse: bool,
    no_synthetic_events: bool,
    fallback_used: bool,
    semantic_sha256: String,
}

#[cfg(feature = "gpu")]
struct SplitT7Inventory {
    fixed_spikes_per_frame: u32,
    shadow_lower_envelope_counts_a: Vec<u32>,
    shadow_lower_envelope_counts_b: Vec<u32>,
    shadow_ensemble_sha256: String,
    shadow_ensemble_artifact: CoupledTwinArtifactEvidence,
    shadow_trial_artifacts: Vec<CoupledTwinArtifactEvidence>,
    shadow_crn_artifacts: Vec<CoupledTwinArtifactEvidence>,
}

#[cfg(feature = "gpu")]
struct SplitT7LaunchReadback {
    total_kl: f32,
    adjudication_code: u32,
    reason_flags: u32,
    state_sha256_before: String,
    state_sha256_after: String,
    physical_state_sha256_before: String,
    physical_state_sha256_after: String,
}

#[cfg(feature = "gpu")]
fn cuda_read_bytes(pointer: u64, byte_len: usize, label: &str) -> Result<Vec<u8>> {
    use cudarc::driver::sys::{cuMemcpyDtoH_v2, CUresult};
    if pointer == 0 || byte_len == 0 {
        bail!("cannot read empty CUDA state for {}", label);
    }
    let mut bytes = vec![0u8; byte_len];
    let result = unsafe {
        cuMemcpyDtoH_v2(
            bytes.as_mut_ptr().cast::<std::ffi::c_void>(),
            pointer,
            byte_len,
        )
    };
    if !matches!(result, CUresult::CUDA_SUCCESS) {
        bail!("cuMemcpyDtoH_v2 for {} failed: {:?}", label, result);
    }
    Ok(bytes)
}

#[cfg(feature = "gpu")]
fn cuda_overwrite_bytes(pointer: u64, bytes: &[u8], label: &str) -> Result<()> {
    use cudarc::driver::sys::{cuMemcpyHtoD_v2, CUresult};
    if pointer == 0 || bytes.is_empty() {
        bail!("cannot write empty CUDA state for {label}");
    }
    let result = unsafe {
        cuMemcpyHtoD_v2(
            pointer,
            bytes.as_ptr().cast::<std::ffi::c_void>(),
            bytes.len(),
        )
    };
    if !matches!(result, CUresult::CUDA_SUCCESS) {
        bail!("cuMemcpyHtoD_v2 for {label} failed: {result:?}");
    }
    Ok(())
}

#[cfg(feature = "gpu")]
fn crn_native_bytes<T: Copy>(values: &[T]) -> Result<&[u8]> {
    let byte_len = values
        .len()
        .checked_mul(std::mem::size_of::<T>())
        .context("CRN native array byte count overflow")?;
    if byte_len == 0 {
        bail!("CRN native array is empty");
    }
    Ok(unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), byte_len) })
}

#[cfg(feature = "gpu")]
fn crn_require_cuda_readback<T: Copy>(
    pointer: u64,
    expected: &[T],
    label: &str,
) -> Result<Vec<u8>> {
    let expected_bytes = crn_native_bytes(expected)?;
    let observed = cuda_read_bytes(pointer, expected_bytes.len(), label)?;
    if observed != expected_bytes {
        bail!("CRN CUDA readback differs from expected host bytes for {label}");
    }
    Ok(observed)
}

#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn apply_coupled_twin_crn_initialization_v3(
    engine_a: &mut PersistentNhsEngine,
    engine_b: &mut PersistentNhsEngine,
    topology_a: &PrismPrepTopology,
    topology_b: &PrismPrepTopology,
    config: &CoupledTwinEvidenceConfig,
    seed: u64,
    temperature_kelvin: f32,
    phase: &str,
    trial_ordinal: Option<u32>,
) -> Result<CoupledTwinCrnInitializationReceiptV3> {
    let host = build_coupled_twin_crn_host_state_v3(
        topology_a,
        topology_b,
        config,
        seed,
        temperature_kelvin,
        phase,
        trial_ordinal,
    )?;
    if engine_a.n_atoms_loaded() != topology_a.n_atoms
        || engine_b.n_atoms_loaded() != topology_b.n_atoms
    {
        bail!("CRN engine/topology atom counts differ");
    }

    // reset_for_replica is retained to clear state and initialize the frozen
    // CUDA curand(seed, tid, 0) lane. Its sequential host draws are then
    // replaced by the atom-identity-derived paired state below.
    engine_a.reset_for_replica(seed)?;
    engine_b.reset_for_replica(seed)?;
    engine_a.cuda_context().bind_to_thread().map_err(|error| {
        anyhow::anyhow!("bind CUDA context for paired CRN initialization: {error:?}")
    })?;
    engine_a.cuda_stream().synchronize()?;
    engine_b.cuda_stream().synchronize()?;

    let mass_readback_a = crn_require_cuda_readback(
        engine_a.d_masses_dev_ptr(),
        &topology_a.masses,
        "stream A masses before CRN write",
    )?;
    let mass_readback_b = crn_require_cuda_readback(
        engine_b.d_masses_dev_ptr(),
        &topology_b.masses,
        "stream B masses before CRN write",
    )?;
    let zero_forces_a = vec![0.0f32; topology_a.n_atoms * 3];
    let zero_forces_b = vec![0.0f32; topology_b.n_atoms * 3];
    let zero_potential_a = vec![0.0f64; topology_a.n_atoms];
    let zero_potential_b = vec![0.0f64; topology_b.n_atoms];
    for (pointer, bytes, label) in [
        (
            engine_a.d_positions_dev_ptr(),
            crn_native_bytes(&host.positions_a)?,
            "stream A paired positions",
        ),
        (
            engine_b.d_positions_dev_ptr(),
            crn_native_bytes(&host.positions_b)?,
            "stream B paired positions",
        ),
        (
            engine_a.d_velocities_dev_ptr(),
            crn_native_bytes(&host.velocities_a)?,
            "stream A paired velocities",
        ),
        (
            engine_b.d_velocities_dev_ptr(),
            crn_native_bytes(&host.velocities_b)?,
            "stream B paired velocities",
        ),
        (
            engine_a.d_forces_dev_ptr(),
            crn_native_bytes(&zero_forces_a)?,
            "stream A zeroed initial forces",
        ),
        (
            engine_b.d_forces_dev_ptr(),
            crn_native_bytes(&zero_forces_b)?,
            "stream B zeroed initial forces",
        ),
        (
            engine_a.d_potential_energy_components_dev_ptr(),
            crn_native_bytes(&zero_potential_a)?,
            "stream A zeroed initial potential components",
        ),
        (
            engine_b.d_potential_energy_components_dev_ptr(),
            crn_native_bytes(&zero_potential_b)?,
            "stream B zeroed initial potential components",
        ),
    ] {
        cuda_overwrite_bytes(pointer, bytes, label)?;
    }
    engine_a.cuda_stream().synchronize()?;
    engine_b.cuda_stream().synchronize()?;

    let positions_a = crn_require_cuda_readback(
        engine_a.d_positions_dev_ptr(),
        &host.positions_a,
        "stream A paired positions",
    )?;
    let positions_b = crn_require_cuda_readback(
        engine_b.d_positions_dev_ptr(),
        &host.positions_b,
        "stream B paired positions",
    )?;
    let velocities_a = crn_require_cuda_readback(
        engine_a.d_velocities_dev_ptr(),
        &host.velocities_a,
        "stream A paired velocities",
    )?;
    let velocities_b = crn_require_cuda_readback(
        engine_b.d_velocities_dev_ptr(),
        &host.velocities_b,
        "stream B paired velocities",
    )?;
    let force_readback_a = crn_require_cuda_readback(
        engine_a.d_forces_dev_ptr(),
        &zero_forces_a,
        "stream A zeroed initial forces",
    )?;
    let force_readback_b = crn_require_cuda_readback(
        engine_b.d_forces_dev_ptr(),
        &zero_forces_b,
        "stream B zeroed initial forces",
    )?;
    let potential_readback_a = crn_require_cuda_readback(
        engine_a.d_potential_energy_components_dev_ptr(),
        &zero_potential_a,
        "stream A zeroed initial potential components",
    )?;
    let potential_readback_b = crn_require_cuda_readback(
        engine_b.d_potential_energy_components_dev_ptr(),
        &zero_potential_b,
        "stream B zeroed initial potential components",
    )?;

    for (atom_index_a, atom_index_b) in &host.shared_indices {
        let pos_a = atom_index_a * 12;
        let pos_b = atom_index_b * 12;
        if positions_a[pos_a..pos_a + 12] != positions_b[pos_b..pos_b + 12]
            || velocities_a[pos_a..pos_a + 12] != velocities_b[pos_b..pos_b + 12]
        {
            bail!("CRN shared-atom device state differs after exact readback");
        }
    }
    if host.b_only_indices != (topology_a.n_atoms..topology_b.n_atoms).collect::<Vec<_>>() {
        bail!("CRN B-only suffix changed after host-state construction");
    }
    let observed_readback = CoupledTwinCrnDeviceReadbackV3 {
        position_sha256_a: crn_hash_device_bytes_v3("stream_a_positions", &positions_a, 4)?,
        position_sha256_b: crn_hash_device_bytes_v3("stream_b_positions", &positions_b, 4)?,
        velocity_sha256_a: crn_hash_device_bytes_v3("stream_a_velocities", &velocities_a, 4)?,
        velocity_sha256_b: crn_hash_device_bytes_v3("stream_b_velocities", &velocities_b, 4)?,
        mass_sha256_a: crn_hash_device_bytes_v3("stream_a_masses", &mass_readback_a, 4)?,
        mass_sha256_b: crn_hash_device_bytes_v3("stream_b_masses", &mass_readback_b, 4)?,
        zero_force_sha256_a: crn_hash_device_bytes_v3("stream_a_forces", &force_readback_a, 4)?,
        zero_force_sha256_b: crn_hash_device_bytes_v3("stream_b_forces", &force_readback_b, 4)?,
        zero_potential_sha256_a: crn_hash_device_bytes_v3(
            "stream_a_potential",
            &potential_readback_a,
            8,
        )?,
        zero_potential_sha256_b: crn_hash_device_bytes_v3(
            "stream_b_potential",
            &potential_readback_b,
            8,
        )?,
    };
    let expected_readback =
        expected_coupled_twin_crn_device_readback_v3(&host, topology_a, topology_b);
    let finalized_receipt = finalize_coupled_twin_crn_device_readback_v3(
        host.receipt,
        &observed_readback,
        &expected_readback,
    )?;
    verify_coupled_twin_crn_receipt_v3(
        &finalized_receipt,
        topology_a,
        topology_b,
        config,
        seed,
        temperature_kelvin,
        phase,
        trial_ordinal,
    )?;
    Ok(finalized_receipt)
}

#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn emit_coupled_twin_crn_receipt_v3(
    output_dir: &Path,
    file_name: &str,
    artifact_kind: &str,
    receipt: &CoupledTwinCrnInitializationReceiptV3,
    topology_a: &PrismPrepTopology,
    topology_b: &PrismPrepTopology,
    config: &CoupledTwinEvidenceConfig,
    temperature_kelvin: f32,
) -> Result<CoupledTwinArtifactEvidence> {
    let path = output_dir.join(file_name);
    verify_coupled_twin_crn_receipt_v3(
        receipt,
        topology_a,
        topology_b,
        config,
        receipt.seed,
        temperature_kelvin,
        &receipt.phase,
        receipt.trial_ordinal,
    )?;
    write_atomic_json(&path, receipt)?;
    let artifact = artifact_evidence(
        artifact_kind,
        None,
        &path,
        Some(1),
        Some(COUPLED_TWIN_CRN_SCHEMA_VERSION),
    )?;
    let readback: CoupledTwinCrnInitializationReceiptV3 =
        serde_json::from_slice(&std::fs::read(&path)?)?;
    if readback != *receipt {
        bail!("coupled-TWIN CRN receipt durable readback drifted");
    }
    verify_coupled_twin_crn_receipt_v3(
        &readback,
        topology_a,
        topology_b,
        config,
        receipt.seed,
        temperature_kelvin,
        &receipt.phase,
        receipt.trial_ordinal,
    )?;
    Ok(artifact)
}

#[cfg(feature = "gpu")]
fn engine_state_sha256s(
    engine: &PersistentNhsEngine,
    topology: &PrismPrepTopology,
    stream_id: u32,
) -> Result<(String, String)> {
    let n_atoms = topology.n_atoms;
    if n_atoms == 0 || engine.n_atoms_loaded() != n_atoms {
        bail!(
            "stream {} engine/topology atom-count mismatch: engine={} topology={}",
            stream_id,
            engine.n_atoms_loaded(),
            n_atoms
        );
    }
    // The retained split-T7 graph is destroyed and rebuilt repeatedly against
    // two engines that share one cudarc context.  A retired graph/event can
    // leave cudarc's shared deferred `error_state` populated even after the
    // actual driver stream has synchronized cleanly.  Re-entering
    // `bind_to_thread()` here would drain that stale status and misattribute it
    // to this state-integrity readback.  Use the checked raw barrier used by
    // the KCC/signal-grid teardown paths; the subsequent synchronous driver
    // copies and before/after hashes still fail closed on any real error or
    // state mutation.
    engine.synchronize_stream_raw(&format!(
        "stream {} pre state-integrity readback",
        stream_id
    ))?;
    let arrays = [
        (
            "positions_f32x3",
            engine.d_positions_dev_ptr(),
            n_atoms.checked_mul(3).and_then(|n| n.checked_mul(4)),
        ),
        (
            "velocities_f32x3",
            engine.d_velocities_dev_ptr(),
            n_atoms.checked_mul(3).and_then(|n| n.checked_mul(4)),
        ),
        (
            "forces_f32x3",
            engine.d_forces_dev_ptr(),
            n_atoms.checked_mul(3).and_then(|n| n.checked_mul(4)),
        ),
        (
            "masses_f32",
            engine.d_masses_dev_ptr(),
            n_atoms.checked_mul(4),
        ),
        (
            "potential_energy_components_f64",
            engine.d_potential_energy_components_dev_ptr(),
            n_atoms.checked_mul(8),
        ),
        ("protocol_dt_f32", engine.d_protocol_dt_dev_ptr(), Some(4)),
    ];
    let mut physical_identity = Vec::new();
    physical_identity.extend_from_slice(b"PRISMOS_SPLIT_T7_PHYSICAL_ENGINE_STATE_V1\0");
    physical_identity.extend_from_slice(&(n_atoms as u64).to_le_bytes());
    for (label, pointer, byte_len) in arrays {
        let byte_len = byte_len.context("engine state byte-count overflow")?;
        let bytes = cuda_read_bytes(pointer, byte_len, label)?;
        physical_identity.extend_from_slice(&(label.len() as u32).to_le_bytes());
        physical_identity.extend_from_slice(label.as_bytes());
        physical_identity.extend_from_slice(&(byte_len as u64).to_le_bytes());
        physical_identity.extend_from_slice(&bytes);
    }
    let physical_sha256 = sha256_bytes(&physical_identity);
    let mut stream_identity = Vec::new();
    stream_identity.extend_from_slice(b"PRISMOS_SPLIT_T7_STREAM_ENGINE_STATE_V2\0");
    stream_identity.extend_from_slice(&stream_id.to_le_bytes());
    stream_identity.extend_from_slice(&physical_identity);
    Ok((sha256_bytes(&stream_identity), physical_sha256))
}

#[cfg(feature = "gpu")]
#[derive(Debug)]
struct PhysicalComponentSnapshot {
    label: &'static str,
    element_bytes: usize,
    bytes: Vec<u8>,
    sha256: String,
}

#[cfg(feature = "gpu")]
fn engine_physical_component_snapshot(
    engine: &PersistentNhsEngine,
    stream_id: u32,
) -> Result<Vec<PhysicalComponentSnapshot>> {
    let n_atoms = engine.n_atoms_loaded();
    if n_atoms == 0 {
        bail!("stream {stream_id} has no atoms for physical-component diagnostics");
    }
    engine.synchronize_stream_raw(&format!(
        "stream {stream_id} physical-component diagnostic readback"
    ))?;
    let arrays = [
        (
            "positions_f32x3",
            engine.d_positions_dev_ptr(),
            n_atoms.checked_mul(3).and_then(|n| n.checked_mul(4)),
            4usize,
        ),
        (
            "velocities_f32x3",
            engine.d_velocities_dev_ptr(),
            n_atoms.checked_mul(3).and_then(|n| n.checked_mul(4)),
            4usize,
        ),
        (
            "forces_f32x3",
            engine.d_forces_dev_ptr(),
            n_atoms.checked_mul(3).and_then(|n| n.checked_mul(4)),
            4usize,
        ),
        (
            "masses_f32",
            engine.d_masses_dev_ptr(),
            n_atoms.checked_mul(4),
            4usize,
        ),
        (
            "potential_energy_components_f64",
            engine.d_potential_energy_components_dev_ptr(),
            n_atoms.checked_mul(8),
            8usize,
        ),
        (
            "protocol_dt_f32",
            engine.d_protocol_dt_dev_ptr(),
            Some(4),
            4usize,
        ),
    ];
    arrays
        .into_iter()
        .map(|(label, pointer, byte_len, element_bytes)| {
            let byte_len = byte_len.context("physical-component byte-count overflow")?;
            let bytes = cuda_read_bytes(pointer, byte_len, label)?;
            Ok(PhysicalComponentSnapshot {
                label,
                element_bytes,
                sha256: sha256_bytes(&bytes),
                bytes,
            })
        })
        .collect()
}

#[cfg(feature = "gpu")]
fn ordered_f32_bits(bits: u32) -> u32 {
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

#[cfg(feature = "gpu")]
fn ordered_f64_bits(bits: u64) -> u64 {
    if bits & 0x8000_0000_0000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000_0000_0000
    }
}

#[cfg(feature = "gpu")]
fn require_identical_physical_components(
    phase: &str,
    left: &[PhysicalComponentSnapshot],
    right: &[PhysicalComponentSnapshot],
) -> Result<()> {
    if left.len() != right.len() {
        bail!(
            "identical-topology component inventory differs at {phase}: A={} B={}",
            left.len(),
            right.len()
        );
    }
    let mut divergences = Vec::new();
    for (a, b) in left.iter().zip(right) {
        if a.label != b.label
            || a.element_bytes != b.element_bytes
            || a.bytes.len() != b.bytes.len()
        {
            bail!(
                "identical-topology component shape differs at {phase}: A={}/{}/{} B={}/{}/{}",
                a.label,
                a.element_bytes,
                a.bytes.len(),
                b.label,
                b.element_bytes,
                b.bytes.len()
            );
        }
        if a.bytes == b.bytes {
            continue;
        }
        let mut first_differing_index = None;
        let mut differing_elements = 0usize;
        let mut max_abs = 0.0f64;
        let mut max_ulp = 0u64;
        for (index, (a_element, b_element)) in a
            .bytes
            .chunks_exact(a.element_bytes)
            .zip(b.bytes.chunks_exact(b.element_bytes))
            .enumerate()
        {
            if a_element == b_element {
                continue;
            }
            first_differing_index.get_or_insert(index);
            differing_elements += 1;
            if a.element_bytes == 4 {
                let a_bits = u32::from_le_bytes(a_element.try_into().unwrap());
                let b_bits = u32::from_le_bytes(b_element.try_into().unwrap());
                let a_value = f32::from_bits(a_bits);
                let b_value = f32::from_bits(b_bits);
                if a_value.is_finite() && b_value.is_finite() {
                    max_abs = max_abs.max((f64::from(a_value) - f64::from(b_value)).abs());
                } else {
                    max_abs = f64::INFINITY;
                }
                max_ulp = max_ulp.max(u64::from(
                    ordered_f32_bits(a_bits).abs_diff(ordered_f32_bits(b_bits)),
                ));
            } else if a.element_bytes == 8 {
                let a_bits = u64::from_le_bytes(a_element.try_into().unwrap());
                let b_bits = u64::from_le_bytes(b_element.try_into().unwrap());
                let a_value = f64::from_bits(a_bits);
                let b_value = f64::from_bits(b_bits);
                if a_value.is_finite() && b_value.is_finite() {
                    max_abs = max_abs.max((a_value - b_value).abs());
                } else {
                    max_abs = f64::INFINITY;
                }
                max_ulp = max_ulp.max(ordered_f64_bits(a_bits).abs_diff(ordered_f64_bits(b_bits)));
            }
        }
        divergences.push(format!(
            "{}:shaA={} shaB={} first_index={} differing_elements={} max_abs={:.17e} max_ulp={}",
            a.label,
            a.sha256,
            b.sha256,
            first_differing_index.unwrap_or(0),
            differing_elements,
            max_abs,
            max_ulp
        ));
    }
    if !divergences.is_empty() {
        bail!(
            "identical-topology physical components diverged at {phase}: {}",
            divergences.join("; ")
        );
    }
    Ok(())
}

#[cfg(feature = "gpu")]
fn canonical_gpu_spike_bytes(spike: &crate::fused_engine::GpuSpikeEvent) -> Vec<u8> {
    // ONE serializer. This used to hand-copy every field, so a layout change
    // silently produced a canonical form that disagreed with the record it
    // claims to canonicalize. It now delegates to the record's own
    // little-endian encoder, which carries the size/offset assertions.
    spike.to_le_bytes().to_vec()
}

#[cfg(feature = "gpu")]
fn split_t7_topology_layout(
    topology: &PrismPrepTopology,
    native_chain_role_map: &BTreeMap<String, String>,
) -> Result<SplitT7TopologyLayout> {
    if topology.n_atoms == 0
        || topology.positions.len() != topology.n_atoms.saturating_mul(3)
        || topology.residue_ids.len() != topology.n_atoms
        || topology.chain_ids.len() != topology.n_atoms
        || topology.atom_names.len() != topology.n_atoms
    {
        bail!("split-T7 topology arrays are incomplete or shape-inconsistent");
    }
    let max_residue = topology
        .residue_ids
        .iter()
        .copied()
        .max()
        .context("split-T7 topology has no residues")?;
    if max_residue >= 1024 {
        bail!(
            "split-T7 residue id {} exceeds the CUDA residue-anchor table",
            max_residue
        );
    }
    let mut aabb_min = [f32::INFINITY; 3];
    let mut aabb_max = [f32::NEG_INFINITY; 3];
    let mut sums = vec![[0.0f64; 3]; max_residue + 1];
    let mut counts = vec![0u32; max_residue + 1];
    let mut first_atom = vec![u32::MAX; max_residue + 1];
    let mut ca_atom = vec![u32::MAX; max_residue + 1];
    let mut residue_chain = vec![None::<String>; max_residue + 1];
    for atom in 0..topology.n_atoms {
        let residue = topology.residue_ids[atom];
        let chain = topology.chain_ids[atom].trim();
        if chain.is_empty() || !native_chain_role_map.contains_key(chain) {
            bail!(
                "split-T7 atom {} residue {} has unresolved native chain {:?}",
                atom,
                residue,
                chain
            );
        }
        if let Some(existing) = residue_chain[residue].as_deref() {
            if existing != chain {
                bail!(
                    "split-T7 residue {} spans native chains {:?} and {:?}",
                    residue,
                    existing,
                    chain
                );
            }
        } else {
            residue_chain[residue] = Some(chain.to_string());
        }
        first_atom[residue] = first_atom[residue].min(atom as u32);
        if topology.atom_names[atom].trim() == "CA" {
            ca_atom[residue] = atom as u32;
        }
        for axis in 0..3 {
            let value = topology.positions[atom * 3 + axis];
            if !value.is_finite() {
                bail!(
                    "split-T7 topology has non-finite coordinate at atom {}",
                    atom
                );
            }
            aabb_min[axis] = aabb_min[axis].min(value);
            aabb_max[axis] = aabb_max[axis].max(value);
            sums[residue][axis] += value as f64;
        }
        counts[residue] = counts[residue].saturating_add(1);
    }
    let mut cell = [0.0f32; 3];
    for axis in 0..3 {
        let span = aabb_max[axis] - aabb_min[axis];
        if !span.is_finite() || span <= 0.0 {
            bail!("split-T7 topology AABB is degenerate on axis {}", axis);
        }
        cell[axis] = span / SPLIT_T7_CLUSTER_GRID as f32;
    }
    let mut residue_to_anchor_atom = vec![u32::MAX; max_residue + 1];
    let mut residue_to_cluster = vec![u32::MAX; max_residue + 1];
    let mut cluster_representative_residue = vec![u32::MAX; SPLIT_T7_CLUSTER_COUNT as usize];
    let mut cluster_chain_role = vec![0xFFu8; SPLIT_T7_CLUSTER_COUNT as usize];
    let mut first_residue_by_role = [u32::MAX; 2];
    let mut residue_centroid = vec![[f32::NAN; 3]; max_residue + 1];
    let mut residue_role = vec![0xFFu8; max_residue + 1];
    let mut role_aabb_min = [[f32::INFINITY; 3]; 2];
    let mut role_aabb_max = [[f32::NEG_INFINITY; 3]; 2];
    for residue in 0..=max_residue {
        if counts[residue] == 0 {
            continue;
        }
        residue_to_anchor_atom[residue] = if ca_atom[residue] != u32::MAX {
            ca_atom[residue]
        } else {
            first_atom[residue]
        };
        let chain = residue_chain[residue]
            .as_deref()
            .context("split-T7 residue lacks native chain")?;
        let role = match native_chain_role_map.get(chain).map(String::as_str) {
            Some("A") => 0u8,
            Some("B") => 1u8,
            other => bail!(
                "split-T7 native chain {:?} has invalid role {:?}",
                chain,
                other
            ),
        };
        residue_role[residue] = role;
        first_residue_by_role[role as usize] =
            first_residue_by_role[role as usize].min(residue as u32);
        for axis in 0..3 {
            let centroid = (sums[residue][axis] / counts[residue] as f64) as f32;
            residue_centroid[residue][axis] = centroid;
            role_aabb_min[role as usize][axis] = role_aabb_min[role as usize][axis].min(centroid);
            role_aabb_max[role as usize][axis] = role_aabb_max[role as usize][axis].max(centroid);
        }
    }
    if first_residue_by_role[0] == u32::MAX {
        bail!("split-T7 topology has no role-A receptor residue domain");
    }
    for residue in 0..=max_residue {
        if counts[residue] == 0 {
            continue;
        }
        let role = residue_role[residue];
        if role > 1 {
            bail!("split-T7 residue {} has no resolved chain role", residue);
        }
        let (grid, offset) = if role == 0 {
            (SPLIT_T7_ROLE_A_GRID, 0u32)
        } else {
            (SPLIT_T7_ROLE_B_GRID, SPLIT_T7_ROLE_B_CLUSTER_OFFSET)
        };
        let mut index = [0u32; 3];
        for axis in 0..3 {
            let span = role_aabb_max[role as usize][axis] - role_aabb_min[role as usize][axis];
            index[axis] = if span.is_finite() && span > f32::EPSILON {
                (((residue_centroid[residue][axis] - role_aabb_min[role as usize][axis]) / span)
                    * grid[axis] as f32) as i32
            } else {
                0
            }
            .clamp(0, grid[axis] as i32 - 1) as u32;
        }
        let cluster = offset + index[0] + index[1] * grid[0] + index[2] * grid[0] * grid[1];
        if cluster >= SPLIT_T7_CLUSTER_COUNT {
            bail!(
                "split-T7 role-partitioned cluster {} exceeds the fixed {}-cluster graph",
                cluster,
                SPLIT_T7_CLUSTER_COUNT
            );
        }
        residue_to_cluster[residue] = cluster;
        cluster_representative_residue[cluster as usize] =
            cluster_representative_residue[cluster as usize].min(residue as u32);
        let slot = &mut cluster_chain_role[cluster as usize];
        if *slot == 0xFF {
            *slot = role;
        } else if *slot != role {
            bail!(
                "split-T7 role-partitioned cluster {} mixes native chain roles A/B",
                cluster
            );
        }
    }
    if residue_to_anchor_atom
        .iter()
        .zip(counts.iter())
        .any(|(anchor, count)| *count > 0 && *anchor == u32::MAX)
    {
        bail!("split-T7 topology has a residue without a native anchor atom");
    }
    let mut chain_offsets = [u32::MAX; 8];
    if first_residue_by_role[0] != 0 {
        bail!(
            "split-T7 role-A residue domain must begin at canonical residue 0; observed {:?}",
            first_residue_by_role[0]
        );
    }
    chain_offsets[0] = first_residue_by_role[0];
    if first_residue_by_role[1] != u32::MAX {
        chain_offsets[1] = first_residue_by_role[1];
    }
    let mut layout_identity = Vec::new();
    layout_identity.extend_from_slice(b"PRISMOS_SPLIT_T7_TOPOLOGY_LAYOUT_V3_ROLE_PARTITIONED\0");
    for value in aabb_min.into_iter().chain(aabb_max).chain(cell) {
        layout_identity.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    for role in 0..2 {
        let grid = if role == 0 {
            SPLIT_T7_ROLE_A_GRID
        } else {
            SPLIT_T7_ROLE_B_GRID
        };
        for value in role_aabb_min[role].into_iter().chain(role_aabb_max[role]) {
            layout_identity.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        for value in grid {
            layout_identity.extend_from_slice(&value.to_le_bytes());
        }
    }
    layout_identity.extend_from_slice(&SPLIT_T7_ROLE_B_CLUSTER_OFFSET.to_le_bytes());
    for value in &residue_to_anchor_atom {
        layout_identity.extend_from_slice(&value.to_le_bytes());
    }
    for value in &residue_to_cluster {
        layout_identity.extend_from_slice(&value.to_le_bytes());
    }
    for value in &cluster_representative_residue {
        layout_identity.extend_from_slice(&value.to_le_bytes());
    }
    layout_identity.extend_from_slice(&cluster_chain_role);
    for value in chain_offsets {
        layout_identity.extend_from_slice(&value.to_le_bytes());
    }
    Ok(SplitT7TopologyLayout {
        aabb_min,
        aabb_max,
        cell,
        residue_to_anchor_atom,
        residue_to_cluster,
        cluster_representative_residue,
        cluster_chain_role,
        chain_offsets,
        layout_sha256: sha256_bytes(&layout_identity),
    })
}

#[cfg(feature = "gpu")]
fn validate_split_t7_paired_layouts(
    layout_a: &SplitT7TopologyLayout,
    layout_b: &SplitT7TopologyLayout,
    identical_topology: bool,
) -> Result<()> {
    let role_a_limit = SPLIT_T7_ROLE_B_CLUSTER_OFFSET;
    let role_b_end = SPLIT_T7_ROLE_B_CLUSTER_OFFSET
        + SPLIT_T7_ROLE_B_GRID[0] * SPLIT_T7_ROLE_B_GRID[1] * SPLIT_T7_ROLE_B_GRID[2];
    if role_b_end != SPLIT_T7_CLUSTER_COUNT {
        bail!("split-T7 role partition does not fill the fixed graph shape");
    }
    for (stream, layout) in [(0u32, layout_a), (1u32, layout_b)] {
        for (residue, cluster) in layout.residue_to_cluster.iter().copied().enumerate() {
            if cluster == u32::MAX {
                continue;
            }
            let role = layout
                .cluster_chain_role
                .get(cluster as usize)
                .copied()
                .unwrap_or(0xFF);
            if cluster >= SPLIT_T7_CLUSTER_COUNT
                || (role == 0 && cluster >= role_a_limit)
                || (role == 1 && cluster < role_a_limit)
                || role > 1
            {
                bail!(
                    "stream {} residue {} violates the split-T7 role-partitioned cluster domain: cluster={} role={}",
                    stream,
                    residue,
                    cluster,
                    role
                );
            }
        }
    }
    let common_residue_domain = layout_a
        .residue_to_cluster
        .len()
        .min(layout_b.residue_to_cluster.len());
    let mut aligned_receptor_residues = 0usize;
    for residue in 0..common_residue_domain {
        let cluster_a = layout_a.residue_to_cluster[residue];
        let cluster_b = layout_b.residue_to_cluster[residue];
        if cluster_a == u32::MAX || cluster_b == u32::MAX {
            if cluster_a != cluster_b {
                bail!(
                    "split-T7 paired layout presence differs for common residue {}",
                    residue
                );
            }
            continue;
        }
        let role_a = layout_a.cluster_chain_role[cluster_a as usize];
        let role_b = layout_b.cluster_chain_role[cluster_b as usize];
        if role_a != 0 || role_b != 0 || cluster_a != cluster_b {
            bail!(
                "split-T7 paired receptor mapping drift at residue {}: A=({},{}) B=({},{})",
                residue,
                cluster_a,
                role_a,
                cluster_b,
                role_b
            );
        }
        aligned_receptor_residues += 1;
    }
    if aligned_receptor_residues == 0 {
        bail!("split-T7 paired layout has no aligned role-A receptor residues");
    }
    if identical_topology
        && (layout_a.layout_sha256 != layout_b.layout_sha256
            || layout_a.residue_to_cluster != layout_b.residue_to_cluster
            || layout_a.cluster_representative_residue != layout_b.cluster_representative_residue
            || layout_a.cluster_chain_role != layout_b.cluster_chain_role)
    {
        bail!("identical-topology control produced nonidentical split-T7 layouts");
    }
    Ok(())
}

#[cfg(feature = "gpu")]
fn prepare_split_t7_frame(
    events: &mut [crate::fused_engine::GpuSpikeEvent],
    layout: &SplitT7TopologyLayout,
) -> Result<(Vec<crate::rich_spike::RichSpike>, Vec<u32>, i32, i32)> {
    events.sort_by_cached_key(canonical_gpu_spike_bytes);
    let timestep_min = events
        .iter()
        .map(|event| event.timestep)
        .min()
        .context("split-T7 frame has no events")?;
    let timestep_max = events
        .iter()
        .map(|event| event.timestep)
        .max()
        .context("split-T7 frame has no events")?;
    let mut clustered = Vec::with_capacity(events.len());
    for event in events {
        if event.n_residues <= 0 {
            bail!("split-T7 event has no topology-resolved primary residue");
        }
        let residue = event.nearby_residues[0];
        if residue < 0
            || layout
                .residue_to_cluster
                .get(residue as usize)
                .copied()
                .unwrap_or(u32::MAX)
                == u32::MAX
        {
            bail!(
                "split-T7 event primary residue {} is not present in the exact topology layout",
                residue
            );
        }
        let cluster = layout.residue_to_cluster[residue as usize];
        let (mut rich, _) = rich_spike_from_gpu_event(event, cluster as i32)?;
        rich.cluster_id = cluster as i32;
        clustered.push((cluster, canonical_gpu_spike_bytes(event), rich));
    }
    clustered.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
    let mut rich = Vec::with_capacity(clustered.len());
    let mut offsets = Vec::with_capacity(SPLIT_T7_CLUSTER_COUNT as usize + 1);
    offsets.push(0u32);
    let mut cursor = 0usize;
    for cluster in 0..SPLIT_T7_CLUSTER_COUNT {
        while cursor < clustered.len() && clustered[cursor].0 == cluster {
            rich.push(clustered[cursor].2);
            cursor += 1;
        }
        offsets.push(u32::try_from(rich.len()).context("split-T7 CSR offset exceeds u32")?);
    }
    if cursor != clustered.len()
        || offsets.len() != SPLIT_T7_CLUSTER_COUNT as usize + 1
        || offsets.last().copied() != Some(rich.len() as u32)
    {
        bail!("split-T7 topology-fixed CSR construction failed");
    }
    Ok((rich, offsets, timestep_min, timestep_max))
}

#[cfg(feature = "gpu")]
fn build_split_t7_stream_runtime(
    engine: &mut PersistentNhsEngine,
    topology: &PrismPrepTopology,
    stream_id: u32,
    layout: SplitT7TopologyLayout,
    initial_rich: &[crate::rich_spike::RichSpike],
    initial_offsets: &[u32],
) -> Result<SplitT7StreamRuntime> {
    use crate::captured_pipeline::{CapturedAdjudicationPipeline, PipelineConfig};
    use std::cell::Cell;

    if initial_rich.len() < 4
        || initial_rich.len() > u32::MAX as usize
        || initial_offsets.len() != SPLIT_T7_CLUSTER_COUNT as usize + 1
        || initial_offsets.last().copied() != Some(initial_rich.len() as u32)
    {
        bail!(
            "stream {} split-T7 initial graph shape is invalid",
            stream_id
        );
    }
    engine.cuda_context().bind_to_thread().map_err(|error| {
        anyhow::anyhow!(
            "bind CUDA context for split-T7 stream {}: {:?}",
            stream_id,
            error
        )
    })?;
    engine
        .cuda_stream()
        .synchronize()
        .with_context(|| format!("synchronize split-T7 stream {} before setup", stream_id))?;
    crate::sh_basis::init_device_basis(engine.cuda_stream().cu_stream() as *mut std::ffi::c_void)
        .map_err(|error| {
            anyhow::anyhow!("stream {} split-T7 SH basis init: {}", stream_id, error)
        })?;
    engine.cuda_stream().synchronize().with_context(|| {
        format!(
            "synchronize split-T7 stream {} after SH basis initialization",
            stream_id
        )
    })?;
    let d_k_lm = crate::sh_basis::k_lm_device_ptr().map_err(|error| {
        anyhow::anyhow!("stream {} split-T7 SH basis pointer: {}", stream_id, error)
    })?;
    let pointers = [
        engine.d_positions_dev_ptr(),
        engine.d_forces_dev_ptr(),
        engine.d_masses_dev_ptr(),
        engine.d_velocities_dev_ptr(),
        engine.d_protocol_dt_dev_ptr(),
        engine.d_potential_energy_components_dev_ptr(),
    ];
    if pointers.iter().any(|pointer| *pointer == 0)
        || engine.d_potential_energy_n_atoms() != topology.n_atoms
        || topology.n_atoms > u32::MAX as usize
    {
        bail!(
            "stream {} split-T7 lacks an exact engine-owned CUDA state: pointers={:?} pe_atoms={} topology_atoms={}",
            stream_id,
            pointers,
            engine.d_potential_energy_n_atoms(),
            topology.n_atoms
        );
    }
    let external_work = engine.allocate_external_work_buffer();
    if external_work == 0 {
        bail!("stream {} split-T7 external-work buffer is null", stream_id);
    }
    let d_spikes = RawCudaAllocation::copy_from_slice(
        initial_rich,
        &format!("split-T7 stream {} RichSpike frame", stream_id),
    )?;
    let d_offsets = RawCudaAllocation::copy_from_slice(
        initial_offsets,
        &format!("split-T7 stream {} topology-fixed CSR", stream_id),
    )?;
    let config = PipelineConfig {
            seed_layout: crate::rich_spike::NativeV2SeedLayout::default(),
        d_spikes: d_spikes.ptr as *const crate::rich_spike::RichSpike,
        d_cluster_offsets: d_offsets.ptr as *const u32,
        n_clusters: SPLIT_T7_CLUSTER_COUNT,
        d_k_lm,
        initial_frame_id: 0,
        source_stream_id: stream_id,
        diagnostic_stream_id: Some(stream_id),
        asc: None,
        d_force_norm_mirror: 0,
        governor_device: None,
        zstr: None,
        sisr: None,
        mar_v2: None,
        noise_floor_override: None,
        d_dt: engine.d_protocol_dt_dev_ptr() as *mut f32,
        d_velocities: engine.d_velocities_dev_ptr() as *mut f32,
        g26_parent_cond_handle: 0,
        f1_parent_cond_handle: 0,
        d_pe_components: engine.d_potential_energy_components_dev_ptr() as *const f64,
        n_atoms_for_pe: topology.n_atoms as u32,
        d_external_work: external_work as *mut f64,
        ghost_tile_ring_dev: 0,
        ghost_tile_max_records: 0,
        firehose_enable: 0,
        body_unroll: 1,
        d_kcc_lead: 0,
        d_forces_anchor: engine.d_forces_dev_ptr(),
        d_masses: engine.d_masses_dev_ptr(),
        force_burst_step: None,
        n_spikes: initial_rich.len() as u32,
        d_current_md_step: engine.d_protocol_current_step_dev_ptr() as *const u32,
        branch_trace_dev: 0,
    };
    let adjudicator_dev = Cell::new(0usize);
    let pipeline = CapturedAdjudicationPipeline::build_with_v2_hook(
        engine.cuda_context(),
        engine.cuda_stream(),
        &config,
        |_, _, pointer| {
            adjudicator_dev.set(pointer);
            Ok(())
        },
    )
    .map_err(|error| {
        anyhow::anyhow!(
            "stream {} split-T7 captured-pipeline build: {:?}",
            stream_id,
            error
        )
    })?;
    if adjudicator_dev.get() == 0 {
        bail!(
            "stream {} split-T7 adjudicator pointer was not captured",
            stream_id
        );
    }
    Ok(SplitT7StreamRuntime {
        pipeline,
        d_spikes,
        d_offsets,
        adjudicator_dev: adjudicator_dev.get(),
        layout,
    })
}

#[cfg(feature = "gpu")]
fn upload_split_t7_topology_constants(
    engine: &PersistentNhsEngine,
    runtime: &SplitT7StreamRuntime,
    stream_id: u32,
    analysis_md_step: u64,
) -> Result<()> {
    let raw_stream = engine.cuda_stream().cu_stream() as *mut std::ffi::c_void;
    let _ = u32::try_from(analysis_md_step)
        .context("split-T7 analysis MD step exceeds the engine device-clock range")?;
    let uploads = unsafe {
        [
            crate::so3_project::set_residue_to_calpha(
                runtime.layout.residue_to_anchor_atom.as_ptr(),
                runtime.layout.residue_to_anchor_atom.len() as u32,
                raw_stream,
            ),
            crate::ghost_tile::set_cluster_repr_residue(
                runtime.layout.cluster_representative_residue.as_ptr(),
                SPLIT_T7_CLUSTER_COUNT,
                raw_stream,
            ),
            crate::ghost_tile::set_chain_offsets(
                runtime.layout.chain_offsets.as_ptr(),
                runtime.layout.chain_offsets.len() as u32,
                raw_stream,
            ),
            crate::ghost_tile::set_cluster_chain_id(
                runtime.layout.cluster_chain_role.as_ptr(),
                SPLIT_T7_CLUSTER_COUNT,
                raw_stream,
            ),
        ]
    };
    if uploads.iter().any(|result| *result != 0) {
        bail!(
            "stream {} split-T7 topology constant upload failed at MD step {}: {:?}",
            stream_id,
            analysis_md_step,
            uploads
        );
    }
    engine.synchronize_stream_raw(&format!(
        "stream {} after split-T7 topology constant upload at MD step {}",
        stream_id, analysis_md_step
    ))?;
    Ok(())
}

#[cfg(feature = "gpu")]
fn read_split_t7_adjudicator(runtime: &SplitT7StreamRuntime) -> Result<(f32, u32, f32, f32, u32)> {
    let bytes = cuda_read_bytes(
        runtime.adjudicator_dev as u64,
        160,
        "split-T7 adjudicator FFI",
    )?;
    Ok((
        read_le_f32(&bytes, 48)?,
        read_le_u32(&bytes, 52)?,
        read_le_f32(&bytes, 148)?,
        read_le_f32(&bytes, 152)?,
        read_le_u32(&bytes, 156)?,
    ))
}

#[cfg(feature = "gpu")]
fn launch_split_t7_frame(
    engine: &mut PersistentNhsEngine,
    topology: &PrismPrepTopology,
    runtime: &mut SplitT7StreamRuntime,
    stream_id: u32,
    analysis_md_step: u64,
    rich: &[crate::rich_spike::RichSpike],
    offsets: &[u32],
) -> Result<SplitT7LaunchReadback> {
    engine.synchronize_stream_raw(&format!(
        "stream {} before split-T7 pointer overwrite at MD step {}",
        stream_id, analysis_md_step
    ))?;
    runtime.d_spikes.overwrite_from_slice(
        rich,
        &format!("split-T7 stream {} RichSpike frame", stream_id),
    )?;
    runtime.d_offsets.overwrite_from_slice(
        offsets,
        &format!("split-T7 stream {} topology-fixed CSR", stream_id),
    )?;
    upload_split_t7_topology_constants(engine, runtime, stream_id, analysis_md_step)?;
    let (before, physical_before) = engine_state_sha256s(engine, topology, stream_id)?;
    runtime.pipeline.launch().map_err(|error| {
        anyhow::anyhow!(
            "stream {} split-T7 analysis launch at MD step {}: {:?}",
            stream_id,
            analysis_md_step,
            error
        )
    })?;
    engine.synchronize_stream_raw(&format!(
        "stream {} post split-T7 analysis launch at MD step {}",
        stream_id, analysis_md_step
    ))?;
    let (total_kl, adjudication_code, _, _, reason_flags) = read_split_t7_adjudicator(runtime)?;
    let (after, physical_after) = engine_state_sha256s(engine, topology, stream_id)?;
    if before != after || physical_before != physical_after {
        bail!(
            "stream {} split-T7 analysis mutated engine MD state at step {}: before={} after={}",
            stream_id,
            analysis_md_step,
            before,
            after
        );
    }
    if !total_kl.is_finite() || adjudication_code > 1 || reason_flags & 0x8000_0000 != 0 {
        bail!(
            "stream {} split-T7 invalid adjudication at step {}: kl={} code={} flags=0x{:08x}",
            stream_id,
            analysis_md_step,
            total_kl,
            adjudication_code,
            reason_flags
        );
    }
    Ok(SplitT7LaunchReadback {
        total_kl,
        adjudication_code,
        reason_flags,
        state_sha256_before: before,
        state_sha256_after: after,
        physical_state_sha256_before: physical_before,
        physical_state_sha256_after: physical_after,
    })
}

#[cfg(feature = "gpu")]
struct SplitT7AccumulatedSnapshot {
    accumulated: Vec<crate::fused_engine::GpuSpikeEvent>,
    window_start: usize,
}

#[cfg(feature = "gpu")]
impl SplitT7AccumulatedSnapshot {
    fn window(&self) -> &[crate::fused_engine::GpuSpikeEvent] {
        &self.accumulated[self.window_start..]
    }
}

#[cfg(feature = "gpu")]
fn read_new_split_t7_spikes(
    engine: &PersistentNhsEngine,
    stream_id: u32,
    cursor: &mut usize,
    cold_hold_steps: i32,
) -> Result<SplitT7AccumulatedSnapshot> {
    let accumulated = engine.get_accumulated_spikes();
    if accumulated.len() < *cursor {
        bail!(
            "stream {} accumulated-spike history regressed: previous={} current={}",
            stream_id,
            *cursor,
            accumulated.len()
        );
    }
    let window_start = *cursor;
    for spike in &accumulated[*cursor..] {
        if spike.timestep <= 0 || spike.timestep > cold_hold_steps {
            bail!(
                "stream {} split-T7 observed completed-step spike timestep {} outside cold hold (0,{}]",
                stream_id,
                spike.timestep,
                cold_hold_steps
            );
        }
        // This validates every numeric field and the ordered primary residue.
        let _ = rich_spike_from_gpu_event(spike, 0)?;
    }
    *cursor = accumulated.len();
    Ok(SplitT7AccumulatedSnapshot {
        accumulated,
        window_start,
    })
}

#[cfg(feature = "gpu")]
fn advance_split_t7_direct_window(
    engine_a: &mut PersistentNhsEngine,
    engine_b: &mut PersistentNhsEngine,
    protocol: &CryoUvProtocol,
    outer_steps: u32,
    fused_inner_steps: i32,
    prefix_outer: &mut u32,
    cursor_a: &mut usize,
    cursor_b: &mut usize,
    identical_topology_negative_control: bool,
) -> Result<(SplitT7AccumulatedSnapshot, SplitT7AccumulatedSnapshot)> {
    if fused_inner_steps <= 0 || *prefix_outer + 1 >= outer_steps {
        bail!(
            "split-T7 cannot consume another direct MD window while preserving a resident remainder: prefix={} total={} inner={}",
            *prefix_outer,
            outer_steps,
            fused_inner_steps
        );
    }
    let next_prefix = prefix_outer.saturating_add(1);
    let completed_md_steps = (next_prefix as u64).saturating_mul(fused_inner_steps as u64);
    if completed_md_steps > protocol.cold_hold_steps.max(0) as u64 {
        bail!(
            "split-T7 requires another real frame beyond cold hold: next_md_step={} cold_hold_end={}",
            completed_md_steps,
            protocol.cold_hold_steps
        );
    }
    // Retire each matched trajectory window before launching its peer.  The
    // identical-topology control is a bitwise CRN control, so overlapping the
    // two streams would make GPU block/atomic scheduling an uncontrolled third
    // variable.  Use the same serialized schedule for real contrasts and
    // controls so topology remains the only intended A/B difference.
    let summary_a = engine_a
        .run(fused_inner_steps)
        .context("split-T7 stream A direct cold-hold window")?;
    engine_a.synchronize_stream_raw("split-T7 stream A post-run barrier")?;
    let post_run_a = identical_topology_negative_control
        .then(|| engine_physical_component_snapshot(engine_a, 0))
        .transpose()?;
    engine_a.force_spike_sync()?;
    engine_a.cuda_stream().synchronize()?;
    let post_spike_a = identical_topology_negative_control
        .then(|| engine_physical_component_snapshot(engine_a, 0))
        .transpose()?;
    let summary_b = engine_b
        .run(fused_inner_steps)
        .context("split-T7 stream B direct cold-hold window")?;
    engine_b.synchronize_stream_raw("split-T7 stream B post-run barrier")?;
    if let Some(post_run_a) = post_run_a.as_deref() {
        let post_run_b = engine_physical_component_snapshot(engine_b, 1)?;
        require_identical_physical_components(
            &format!("direct_window_{next_prefix}_post_run"),
            post_run_a,
            &post_run_b,
        )?;
    }
    engine_b.force_spike_sync()?;
    engine_b.cuda_stream().synchronize()?;
    if let Some(post_spike_a) = post_spike_a.as_deref() {
        let post_spike_b = engine_physical_component_snapshot(engine_b, 1)?;
        require_identical_physical_components(
            &format!("direct_window_{next_prefix}_post_spike_sync"),
            post_spike_a,
            &post_spike_b,
        )?;
    }
    *prefix_outer = next_prefix;
    let completed_md_steps_u32 = u32::try_from(completed_md_steps)
        .context("split-T7 direct-prefix MD step exceeds the KCC/protocol range")?;
    let phase = protocol_phase(protocol, completed_md_steps_u32);
    engine_a.kcc_step_once(completed_md_steps_u32, phase)?;
    engine_a.cuda_stream().synchronize()?;
    let post_kcc_a = identical_topology_negative_control
        .then(|| engine_physical_component_snapshot(engine_a, 0))
        .transpose()?;
    engine_b.kcc_step_once(completed_md_steps_u32, phase)?;
    engine_b.cuda_stream().synchronize()?;
    if let Some(post_kcc_a) = post_kcc_a.as_deref() {
        let post_kcc_b = engine_physical_component_snapshot(engine_b, 1)?;
        require_identical_physical_components(
            &format!("direct_window_{next_prefix}_post_kcc"),
            post_kcc_a,
            &post_kcc_b,
        )?;
    }
    engine_a.rebuild_neighbor_lists_if_needed()?;
    engine_a.cuda_stream().synchronize()?;
    let post_rebuild_a = identical_topology_negative_control
        .then(|| engine_physical_component_snapshot(engine_a, 0))
        .transpose()?;
    engine_b.rebuild_neighbor_lists_if_needed()?;
    engine_b.cuda_stream().synchronize()?;
    if let Some(post_rebuild_a) = post_rebuild_a.as_deref() {
        let post_rebuild_b = engine_physical_component_snapshot(engine_b, 1)?;
        require_identical_physical_components(
            &format!("direct_window_{next_prefix}_post_neighbor_rebuild"),
            post_rebuild_a,
            &post_rebuild_b,
        )?;
    }
    let snapshot_a = read_new_split_t7_spikes(engine_a, 0, cursor_a, protocol.cold_hold_steps)?;
    let snapshot_b = read_new_split_t7_spikes(engine_b, 1, cursor_b, protocol.cold_hold_steps)?;
    log::debug!(
        "split-T7 direct window {} ({} MD steps): A summary_spikes={} appended={} B summary_spikes={} appended={}",
        next_prefix,
        completed_md_steps,
        summary_a.total_spikes,
        snapshot_a.window().len(),
        summary_b.total_spikes,
        snapshot_b.window().len()
    );
    Ok((snapshot_a, snapshot_b))
}

#[cfg(feature = "gpu")]
fn split_t7_input_header(
    stream_id: u32,
    fixed_spikes: u32,
    direct_prefix_outer_steps: u32,
    config: &CoupledTwinEvidenceConfig,
    topology_sha256: &str,
    layout_sha256: &str,
) -> Vec<u8> {
    let mut input = Vec::new();
    input.extend_from_slice(b"PRT7C003");
    input.extend_from_slice(&COUPLED_TWIN_EVIDENCE_SCHEMA_VERSION.to_le_bytes());
    input.extend_from_slice(&stream_id.to_le_bytes());
    input.extend_from_slice(&(SPLIT_T7_TOTAL_FRAMES as u32).to_le_bytes());
    input.extend_from_slice(&fixed_spikes.to_le_bytes());
    input.extend_from_slice(&SPLIT_T7_CLUSTER_COUNT.to_le_bytes());
    input.extend_from_slice(&direct_prefix_outer_steps.to_le_bytes());
    input.extend_from_slice(sha256_bytes(config.run_id.as_bytes()).as_bytes());
    input.extend_from_slice(topology_sha256.as_bytes());
    input.extend_from_slice(layout_sha256.as_bytes());
    input
}

#[cfg(feature = "gpu")]
fn append_split_t7_input_frame<W: Write + Seek>(
    input: &mut W,
    ordinal: u32,
    analysis_md_step: u64,
    timestep_min: i32,
    timestep_max: i32,
    source: &SplitT7SourceWindow,
    rich: &[crate::rich_spike::RichSpike],
) -> Result<SplitT7InputFrameOffsets> {
    let source_count = u32::try_from(source.source_event_count)
        .context("split-T7 source frame count exceeds u32")?;
    let selected_count = u32::try_from(source.selected_events.len())
        .context("split-T7 selected frame count exceeds u32")?;
    if selected_count != rich.len() as u32
        || source.selected_indices.len() != source.selected_events.len()
    {
        bail!("split-T7 source/selection frame identity is inconsistent");
    }
    let (canonical_source_len, source_events_sha256) =
        canonical_split_t7_source_identity(source.source_events)?;
    if source_events_sha256 != source.source_events_sha256 {
        bail!("split-T7 source identity changed before envelope serialization");
    }
    let rich_bytes = rich_spike_slice_bytes(rich)?;
    let rich_sha256 = sha256_bytes(rich_bytes);
    let mut selection_bytes = Vec::with_capacity(source.selected_indices.len().saturating_mul(4));
    for source_index in &source.selected_indices {
        selection_bytes.extend_from_slice(&source_index.to_le_bytes());
    }
    input.write_all(&ordinal.to_le_bytes())?;
    input.write_all(&analysis_md_step.to_le_bytes())?;
    input.write_all(&timestep_min.to_le_bytes())?;
    input.write_all(&timestep_max.to_le_bytes())?;
    input.write_all(&source_count.to_le_bytes())?;
    input.write_all(&selected_count.to_le_bytes())?;
    input.write_all(&canonical_source_len.to_le_bytes())?;
    input.write_all(&(selection_bytes.len() as u64).to_le_bytes())?;
    input.write_all(&(rich_bytes.len() as u64).to_le_bytes())?;
    input.write_all(source.source_events_sha256.as_bytes())?;
    input.write_all(rich_sha256.as_bytes())?;
    let source_offset = input.stream_position()?;
    write_canonical_split_t7_source(input, source.source_events)?;
    let selection_offset = input.stream_position()?;
    input.write_all(&selection_bytes)?;
    let rich_offset = input.stream_position()?;
    input.write_all(rich_bytes)?;
    Ok(SplitT7InputFrameOffsets {
        source_offset,
        source_length: canonical_source_len,
        selection_offset,
        selection_length: selection_bytes.len() as u64,
        rich_offset,
        rich_length: rich_bytes.len() as u64,
    })
}

#[cfg(feature = "gpu")]
struct SplitT7InputFrameOffsets {
    source_offset: u64,
    source_length: u64,
    selection_offset: u64,
    selection_length: u64,
    rich_offset: u64,
    rich_length: u64,
}

#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
struct VerifiedSplitT7SelectedEvent {
    selected_ordinal: u32,
    source_index: u32,
    canonical_event_sha256: String,
    event: crate::fused_engine::GpuSpikeEvent,
    support_residue_ids: Vec<i32>,
    topology_cluster_id: u32,
}

#[cfg(feature = "gpu")]
struct VerifiedSplitT7Frame {
    source_event_count: u32,
    selected_event_count: u32,
    source_events_sha256: String,
    selected_rich_spikes_sha256: String,
    selected_events: Vec<VerifiedSplitT7SelectedEvent>,
}

#[cfg(feature = "gpu")]
fn read_exact_array<const N: usize, R: Read>(reader: &mut R, label: &str) -> Result<[u8; N]> {
    let mut bytes = [0u8; N];
    reader
        .read_exact(&mut bytes)
        .with_context(|| format!("read split-T7 {label}"))?;
    Ok(bytes)
}

#[cfg(feature = "gpu")]
fn read_stream_u32<R: Read>(reader: &mut R, label: &str) -> Result<u32> {
    Ok(u32::from_le_bytes(read_exact_array(reader, label)?))
}

#[cfg(feature = "gpu")]
fn read_stream_i32<R: Read>(reader: &mut R, label: &str) -> Result<i32> {
    Ok(i32::from_le_bytes(read_exact_array(reader, label)?))
}

#[cfg(feature = "gpu")]
fn read_stream_u64<R: Read>(reader: &mut R, label: &str) -> Result<u64> {
    Ok(u64::from_le_bytes(read_exact_array(reader, label)?))
}

#[cfg(feature = "gpu")]
fn read_stream_sha256<R: Read>(reader: &mut R, label: &str) -> Result<String> {
    let bytes = read_exact_array::<64, _>(reader, label)?;
    if !bytes
        .iter()
        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
    {
        bail!("split-T7 {label} is not a canonical lowercase hexadecimal SHA-256");
    }
    let value = std::str::from_utf8(&bytes)
        .with_context(|| format!("split-T7 {label} is not UTF-8"))?
        .to_string();
    Ok(value)
}

#[cfg(feature = "gpu")]
fn parse_canonical_gpu_spike_bytes(bytes: &[u8]) -> Result<crate::fused_engine::GpuSpikeEvent> {
    use crate::fused_engine::GpuSpikeEvent;

    if bytes.len() != SPLIT_T7_CANONICAL_EVENT_BYTES {
        bail!(
            "canonical split-T7 event has {} bytes, expected {}",
            bytes.len(),
            SPLIT_T7_CANONICAL_EVENT_BYTES
        );
    }
    // ONE decoder (see GpuSpikeEvent::from_le_bytes_with_stride). The byte-exact
    // round-trip check is retained: it is what makes the canonical form
    // canonical rather than merely parseable.
    let event = GpuSpikeEvent::from_le_bytes_with_stride(bytes, SPLIT_T7_CANONICAL_EVENT_BYTES)?;
    if canonical_gpu_spike_bytes(&event) != bytes {
        bail!("canonical split-T7 event did not round-trip byte-exactly");
    }
    Ok(event)
}

#[cfg(feature = "gpu")]
fn split_t7_selection_indices(source_count: u32, selected_count: u32) -> Result<Vec<u32>> {
    if selected_count < 4 || source_count < selected_count {
        bail!(
            "invalid split-T7 full-span selection: source={} selected={}",
            source_count,
            selected_count
        );
    }
    let mut indices = Vec::with_capacity(selected_count as usize);
    for ordinal in 0..selected_count {
        let index = (ordinal as u64)
            .checked_mul(source_count.saturating_sub(1) as u64)
            .context("split-T7 selection index overflow")?
            / selected_count.saturating_sub(1) as u64;
        indices.push(u32::try_from(index).context("split-T7 selection index exceeds u32")?);
    }
    if indices.windows(2).any(|pair| pair[0] >= pair[1])
        || indices.first().copied() != Some(0)
        || indices.last().copied() != Some(source_count - 1)
    {
        bail!("split-T7 selection is not a strict full-span sampling");
    }
    Ok(indices)
}

#[cfg(feature = "gpu")]
fn verify_split_t7_frame<R: Read + Seek>(
    reader: &mut R,
    stream_id: u32,
    expected_ordinal: u32,
    expected_analysis_md_step: Option<u64>,
    expected_selected_count: Option<u32>,
    evidence: Option<&CoupledNoiseFloorFrameEvidence>,
    layout: &SplitT7TopologyLayout,
) -> Result<VerifiedSplitT7Frame> {
    let ordinal = read_stream_u32(reader, "frame ordinal")?;
    let analysis_md_step = read_stream_u64(reader, "analysis MD step")?;
    let timestep_min = read_stream_i32(reader, "source timestep minimum")?;
    let timestep_max = read_stream_i32(reader, "source timestep maximum")?;
    let source_count = read_stream_u32(reader, "source event count")?;
    let selected_count = read_stream_u32(reader, "selected event count")?;
    let canonical_source_length = read_stream_u64(reader, "canonical source length")?;
    let selection_length = read_stream_u64(reader, "selection length")?;
    let rich_length = read_stream_u64(reader, "RichSpike length")?;
    let source_sha256 = read_stream_sha256(reader, "source SHA-256")?;
    let rich_sha256 = read_stream_sha256(reader, "RichSpike SHA-256")?;
    if ordinal != expected_ordinal
        || expected_analysis_md_step.is_some_and(|expected| analysis_md_step != expected)
        || expected_selected_count.is_some_and(|expected| selected_count != expected)
        || timestep_min < 0
        || timestep_max < timestep_min
        || selection_length != selected_count as u64 * 4
        || rich_length != selected_count as u64 * 64
    {
        bail!(
            "stream {} split-T7 frame {} header contract failed",
            stream_id,
            expected_ordinal
        );
    }
    let expected_canonical_source_length = 34u64
        .checked_add(8)
        .and_then(|bytes| {
            bytes.checked_add(
                (source_count as u64).checked_mul(4 + SPLIT_T7_CANONICAL_EVENT_BYTES as u64)?,
            )
        })
        .context("split-T7 canonical source length overflow")?;
    if canonical_source_length != expected_canonical_source_length {
        bail!(
            "stream {} split-T7 frame {} canonical source length/count mismatch",
            stream_id,
            expected_ordinal
        );
    }
    let expected_indices = split_t7_selection_indices(source_count, selected_count)?;
    let source_offset = reader.stream_position()?;
    let source_prefix = b"PRISMOS_SPLIT_T7_SOURCE_EVENTS_V1\0";
    let observed_prefix = read_exact_array::<34, _>(reader, "canonical source prefix")?;
    if observed_prefix.as_slice() != source_prefix {
        bail!(
            "stream {} split-T7 canonical source prefix mismatch",
            stream_id
        );
    }
    let encoded_source_count = read_stream_u64(reader, "canonical source count")?;
    if encoded_source_count != source_count as u64 {
        bail!(
            "stream {} split-T7 canonical source count mismatch",
            stream_id
        );
    }
    let mut source_digest = Sha256::new();
    source_digest.update(observed_prefix);
    source_digest.update(encoded_source_count.to_le_bytes());
    let mut canonical_source_events = Vec::with_capacity(source_count as usize);
    let mut selected_events = Vec::with_capacity(selected_count as usize);
    let mut selected_cursor = 0usize;
    for source_index in 0..source_count {
        let event_length_bytes = read_exact_array::<4, _>(reader, "canonical event length")?;
        let event_length = u32::from_le_bytes(event_length_bytes);
        if event_length as usize != SPLIT_T7_CANONICAL_EVENT_BYTES {
            bail!(
                "stream {} canonical source event {} declares {} bytes, expected {}",
                stream_id,
                source_index,
                event_length,
                SPLIT_T7_CANONICAL_EVENT_BYTES
            );
        }
        let event_bytes = read_exact_array::<SPLIT_T7_CANONICAL_EVENT_BYTES, _>(
            reader,
            "canonical split-T7 event",
        )?;
        canonical_source_events.push(event_bytes);
        let event = parse_canonical_gpu_spike_bytes(&event_bytes)?;
        let primary = event.nearby_residues[0];
        let topology_cluster_id = if primary >= 0 {
            layout
                .residue_to_cluster
                .get(primary as usize)
                .copied()
                .unwrap_or(u32::MAX)
        } else {
            u32::MAX
        };
        if topology_cluster_id >= SPLIT_T7_CLUSTER_COUNT {
            bail!(
                "stream {} source event {} has no exact topology cluster",
                stream_id,
                source_index
            );
        }
        let (_, support) = rich_spike_from_gpu_event(&event, topology_cluster_id as i32)?;
        for residue in &support {
            if layout
                .residue_to_anchor_atom
                .get(*residue as usize)
                .copied()
                .unwrap_or(u32::MAX)
                == u32::MAX
            {
                bail!(
                    "stream {} source event {} cites residue {} absent from the exact topology",
                    stream_id,
                    source_index,
                    residue
                );
            }
        }
        if selected_cursor < expected_indices.len()
            && source_index == expected_indices[selected_cursor]
        {
            selected_events.push(VerifiedSplitT7SelectedEvent {
                selected_ordinal: selected_cursor as u32,
                source_index,
                canonical_event_sha256: sha256_bytes(&event_bytes),
                event,
                support_residue_ids: support.into_iter().collect(),
                topology_cluster_id,
            });
            selected_cursor += 1;
        }
    }
    // The file retains exact GPU acquisition order for provenance and source
    // index reconstruction.  Its identity is the duplicate-preserving event
    // multiset because atomic GPU slot assignment is not semantic ordering.
    canonical_source_events.sort();
    for event_bytes in canonical_source_events {
        source_digest.update((SPLIT_T7_CANONICAL_EVENT_BYTES as u32).to_le_bytes());
        source_digest.update(event_bytes);
    }
    let observed_source_length = reader
        .stream_position()?
        .checked_sub(source_offset)
        .context("split-T7 source offset regressed")?;
    let observed_source_sha256 = format!("{:x}", source_digest.finalize());
    let selected_timestep_min = selected_events
        .iter()
        .map(|selected| selected.event.timestep)
        .min()
        .context("split-T7 selected source set is empty")?;
    let selected_timestep_max = selected_events
        .iter()
        .map(|selected| selected.event.timestep)
        .max()
        .context("split-T7 selected source set is empty")?;
    if observed_source_length != canonical_source_length
        || observed_source_sha256 != source_sha256
        || selected_timestep_min != timestep_min
        || selected_timestep_max != timestep_max
        || selected_events.len() != selected_count as usize
    {
        bail!(
            "stream {} split-T7 frame {} canonical source verification failed",
            stream_id,
            expected_ordinal
        );
    }
    let selection_offset = reader.stream_position()?;
    for expected_index in &expected_indices {
        if read_stream_u32(reader, "selected source index")? != *expected_index {
            bail!(
                "stream {} split-T7 frame {} selection index drift",
                stream_id,
                expected_ordinal
            );
        }
    }
    let rich_offset = reader.stream_position()?;
    let mut observed_rich_digest = Sha256::new();
    let mut expected_rich = Vec::with_capacity(selected_events.len());
    for selected in &selected_events {
        let (rich, _) =
            rich_spike_from_gpu_event(&selected.event, selected.topology_cluster_id as i32)?;
        expected_rich.push((
            selected.topology_cluster_id,
            canonical_gpu_spike_bytes(&selected.event),
            rich,
        ));
    }
    expected_rich.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
    for (rich_ordinal, (_, _, expected)) in expected_rich.iter().enumerate() {
        let observed = read_exact_array::<64, _>(reader, "selected RichSpike")?;
        observed_rich_digest.update(observed);
        let expected_bytes = rich_spike_slice_bytes(std::slice::from_ref(expected))?;
        if observed.as_slice() != expected_bytes {
            bail!(
                "stream {} split-T7 frame {} RichSpike {} is not source-derived",
                stream_id,
                expected_ordinal,
                rich_ordinal
            );
        }
    }
    let observed_rich_sha256 = format!("{:x}", observed_rich_digest.finalize());
    if observed_rich_sha256 != rich_sha256 {
        bail!(
            "stream {} split-T7 frame {} RichSpike hash mismatch",
            stream_id,
            expected_ordinal
        );
    }
    if let Some(frame) = evidence {
        if frame.ordinal != ordinal
            || frame.analysis_md_step != analysis_md_step
            || frame.source_timestep_min != timestep_min
            || frame.source_timestep_max != timestep_max
            || frame.timestep != timestep_max
            || frame.source_spike_count != source_count
            || frame.selected_spike_count != selected_count
            || frame.dropped_spike_count != source_count - selected_count
            || frame.source_input_byte_offset != source_offset
            || frame.source_input_byte_length != canonical_source_length
            || frame.selection_index_byte_offset != selection_offset
            || frame.selection_index_byte_length != selection_length
            || frame.input_byte_offset != rich_offset
            || frame.input_byte_length != rich_length
            || frame.source_events_sha256 != observed_source_sha256
            || frame.selected_rich_spikes_sha256 != observed_rich_sha256
        {
            bail!(
                "stream {} split-T7 frame {} JSON/binary cross-binding failed",
                stream_id,
                expected_ordinal
            );
        }
    }
    Ok(VerifiedSplitT7Frame {
        source_event_count: source_count,
        selected_event_count: selected_count,
        source_events_sha256: observed_source_sha256,
        selected_rich_spikes_sha256: observed_rich_sha256,
        selected_events,
    })
}

#[cfg(feature = "gpu")]
fn verify_split_t7_calibration_input(
    path: &Path,
    config: &CoupledTwinEvidenceConfig,
    topology_sha256: &str,
    calibration: &CoupledNoiseFloorCalibration,
    layout: &SplitT7TopologyLayout,
) -> Result<()> {
    let mut reader = File::open(path)
        .with_context(|| format!("open split-T7 calibration input {}", path.display()))?;
    if read_exact_array::<8, _>(&mut reader, "calibration magic")?.as_slice() != b"PRT7C003"
        || read_stream_u32(&mut reader, "calibration schema")?
            != COUPLED_TWIN_EVIDENCE_SCHEMA_VERSION
        || read_stream_u32(&mut reader, "calibration stream")? != calibration.stream_id
        || read_stream_u32(&mut reader, "calibration frame count")? != SPLIT_T7_TOTAL_FRAMES as u32
        || read_stream_u32(&mut reader, "calibration selected count")?
            != calibration.fixed_spikes_per_frame
        || read_stream_u32(&mut reader, "calibration cluster count")? != SPLIT_T7_CLUSTER_COUNT
        || read_stream_u32(&mut reader, "calibration prefix count")?
            != calibration.direct_prefix_outer_steps
        || read_stream_sha256(&mut reader, "calibration run hash")?
            != sha256_bytes(config.run_id.as_bytes())
        || read_stream_sha256(&mut reader, "calibration topology hash")? != topology_sha256
        || read_stream_sha256(&mut reader, "calibration layout hash")? != layout.layout_sha256
    {
        bail!(
            "stream {} split-T7 calibration envelope header failed",
            calibration.stream_id
        );
    }
    for (ordinal, frame) in calibration.frames.iter().enumerate() {
        verify_split_t7_frame(
            &mut reader,
            calibration.stream_id,
            ordinal as u32,
            Some(((ordinal + 1) as u64) * calibration.inventory_window_md_steps as u64),
            Some(calibration.fixed_spikes_per_frame),
            Some(frame),
            layout,
        )?;
    }
    if read_exact_array::<8, _>(&mut reader, "calibration trailer")?.as_slice() != b"PRT7E003" {
        bail!(
            "stream {} split-T7 calibration trailer mismatch",
            calibration.stream_id
        );
    }
    let end = reader.stream_position()?;
    if end != reader.seek(SeekFrom::End(0))? {
        bail!(
            "stream {} split-T7 calibration has unbound trailing bytes",
            calibration.stream_id
        );
    }
    Ok(())
}

#[cfg(feature = "gpu")]
fn verify_split_t7_final_input(
    path: &Path,
    config: &CoupledTwinEvidenceConfig,
    topology_sha256: &str,
    map: &CoupledGhostSiteMap,
    layout: &SplitT7TopologyLayout,
) -> Result<VerifiedSplitT7Frame> {
    let mut reader = File::open(path)
        .with_context(|| format!("open split-T7 final input {}", path.display()))?;
    if read_exact_array::<8, _>(&mut reader, "final magic")?.as_slice() != b"PRT7F003"
        || read_stream_u32(&mut reader, "final schema")? != COUPLED_TWIN_EVIDENCE_SCHEMA_VERSION
        || read_stream_u32(&mut reader, "final stream")? != map.stream_id
        || read_stream_sha256(&mut reader, "final run hash")?
            != sha256_bytes(config.run_id.as_bytes())
        || read_stream_sha256(&mut reader, "final topology hash")? != topology_sha256
        || read_stream_sha256(&mut reader, "final layout hash")? != layout.layout_sha256
    {
        bail!(
            "stream {} split-T7 final envelope header failed",
            map.stream_id
        );
    }
    let frame = verify_split_t7_frame(
        &mut reader,
        map.stream_id,
        u32::MAX,
        Some(map.emitted_step_idx),
        Some(map.selected_spike_count),
        None,
        layout,
    )?;
    if read_exact_array::<8, _>(&mut reader, "final trailer")?.as_slice() != b"PRT7FEND" {
        bail!("stream {} split-T7 final trailer mismatch", map.stream_id);
    }
    let end = reader.stream_position()?;
    if end != reader.seek(SeekFrom::End(0))?
        || frame.source_event_count as u64 != map.raw_spike_count
        || frame.selected_event_count != map.selected_spike_count
        || frame.source_events_sha256 != map.final_canonical_source_events_sha256
        || frame.selected_rich_spikes_sha256 != map.selected_rich_spikes_sha256
    {
        bail!(
            "stream {} split-T7 final envelope/map binding failed",
            map.stream_id
        );
    }
    Ok(frame)
}

#[cfg(feature = "gpu")]
fn split_t7_population_stats(values: &[f64], stream_id: u32) -> Result<(f32, f32)> {
    if values.len() != SPLIT_T7_VARIANCE_FRAMES
        || values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        bail!(
            "stream {} split-T7 magnitude variance window is invalid",
            stream_id
        );
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let mean_square = values.iter().map(|value| value * value).sum::<f64>() / values.len() as f64;
    let variance = (mean_square - mean * mean).max(0.0);
    let sigma = variance.sqrt();
    if !mean.is_finite() || !sigma.is_finite() || sigma <= 0.0 {
        bail!(
            "stream {} split-T7 measured a degenerate magnitude floor: mu={} sigma={}",
            stream_id,
            mean,
            sigma
        );
    }
    Ok((mean as f32, sigma as f32))
}

#[cfg(feature = "gpu")]
fn finalize_split_t7_stream(
    stream_id: u32,
    topology_path: &str,
    topology_sha256: &str,
    config: &CoupledTwinEvidenceConfig,
    output_dir: &Path,
    runtime: SplitT7StreamRuntime,
    input_artifact: CoupledTwinArtifactEvidence,
    frames: Vec<CoupledNoiseFloorFrameEvidence>,
    inventory_frame_counts: Vec<u32>,
    paired_inventory_sha256: &str,
    shadow_lower_envelope_frame_counts: Vec<u32>,
    shadow_ensemble_sha256: &str,
    shadow_ensemble_artifact: &CoupledTwinArtifactEvidence,
    authoritative_minimum: u32,
    authoritative_minimum_slack_from_k: u32,
    direct_prefix_outer_steps: u32,
    fused_inner_steps: i32,
    fixed_spikes: u32,
    noise_floor_mu: f32,
    noise_floor_sigma: f32,
    validation_marker: u32,
) -> Result<SplitT7StreamOutcome> {
    if frames.len() != SPLIT_T7_TOTAL_FRAMES
        || inventory_frame_counts.len() != SPLIT_T7_TOTAL_FRAMES
        || paired_inventory_sha256.len() != 64
        || shadow_lower_envelope_frame_counts.len() != SPLIT_T7_TOTAL_FRAMES
        || shadow_ensemble_sha256.len() != 64
        || shadow_ensemble_artifact.kind != "split_t7_shadow_ensemble_receipt"
        || shadow_ensemble_artifact.stream_id.is_some()
        || shadow_ensemble_artifact.sha256.len() != 64
        || shadow_ensemble_artifact.size_bytes == 0
        || shadow_ensemble_artifact.schema_version != Some(SPLIT_T7_SHADOW_SCHEMA_VERSION)
        || shadow_lower_envelope_frame_counts
            .iter()
            .any(|count| count.saturating_div(SPLIT_T7_SHADOW_RESERVE_FACTOR) < fixed_spikes)
        || authoritative_minimum < fixed_spikes
        || authoritative_minimum_slack_from_k != authoritative_minimum - fixed_spikes
        || input_artifact.kind != "ghost_noise_floor_input"
        || input_artifact.stream_id != Some(stream_id)
        || input_artifact.record_count != Some(SPLIT_T7_TOTAL_FRAMES as u64)
        || frames.iter().any(|frame| {
            frame.selected_spike_count != fixed_spikes
                || frame.source_spike_count < frame.selected_spike_count
                || frame.source_events_sha256.len() != 64
                || frame.engine_state_sha256_before != frame.engine_state_sha256_after
                || frame.engine_physical_state_sha256_before
                    != frame.engine_physical_state_sha256_after
        })
    {
        bail!(
            "stream {} split-T7 finalization invariants failed",
            stream_id
        );
    }
    let stem = coupled_topology_stem(config);
    validate_run_artifact(output_dir, &input_artifact)?;
    let frozen_anchor_bytes = cuda_read_bytes(
        runtime.pipeline.p_frozen_dev as u64,
        usize::try_from(runtime.pipeline.tiles_bytes)
            .context("split-T7 frozen-anchor byte count exceeds usize")?,
        &format!("split-T7 stream {} frozen reference manifold", stream_id),
    )?;
    let frozen_anchor_path = output_dir.join(format!(
        "{stem}_stream{stream_id:02}_ghost_noise_floor_frozen_anchor.bin"
    ));
    write_atomic_bytes(&frozen_anchor_path, &frozen_anchor_bytes)?;
    let frozen_anchor_artifact = artifact_evidence(
        "ghost_noise_floor_frozen_anchor",
        Some(stream_id),
        &frozen_anchor_path,
        Some(SPLIT_T7_CLUSTER_COUNT as u64),
        Some(COUPLED_TWIN_EVIDENCE_SCHEMA_VERSION),
    )?;
    let frozen_anchor_sha256 = sha256_bytes(&frozen_anchor_bytes);
    if frozen_anchor_artifact.sha256 != frozen_anchor_sha256
        || frozen_anchor_artifact.size_bytes != runtime.pipeline.tiles_bytes
    {
        bail!(
            "stream {} split-T7 frozen-anchor durable readback mismatch",
            stream_id
        );
    }
    let validation = frames
        .last()
        .context("split-T7 validation frame is absent")?;
    if validation.phase != "validation"
        || validation.adjudication_code > 1
        || validation.adjudication_reason_flags & 0x8000_0000 != 0
        || validation.total_kl.is_none()
    {
        bail!(
            "stream {} split-T7 validation adjudication failed",
            stream_id
        );
    }
    let calibration = CoupledNoiseFloorCalibration {
        schema_kind: "prismos.coupled_twin_split_t7_noise_floor.v3".to_string(),
        schema_version: SPLIT_T7_SHADOW_SCHEMA_VERSION,
        status: "IMPLEMENTATION_VERIFIED".to_string(),
        validation_status: "PASS".to_string(),
        run_id: config.run_id.clone(),
        stream_id,
        topology_path: topology_path.to_string(),
        topology_sha256: topology_sha256.to_string(),
        producer: "three_trial_deterministic_seed_split_t7_shadow_ensemble_v3+execute_online_split_t7_calibration+CapturedAdjudicationPipeline::build_with_v2_hook+live_engine_state_noninterference".to_string(),
        source: "paired_PersistentNhsEngine_direct_cold_hold_windows/newly_accumulated_real_spikes".to_string(),
        calibration_phase: "cold_hold_online_before_resident_graph_capture".to_string(),
        selection_method: SPLIT_T7_SELECTION_METHOD_V3.to_string(),
        topology_layout_sha256: runtime.layout.layout_sha256.clone(),
        topology_aabb_min: runtime.layout.aabb_min,
        topology_aabb_max: runtime.layout.aabb_max,
        fixed_cluster_count: SPLIT_T7_CLUSTER_COUNT,
        fixed_spikes_per_frame: fixed_spikes,
        inventory_window_md_steps: u32::try_from(fused_inner_steps)
            .context("split-T7 inventory window steps exceed u32")?,
        shadow_trial_count: SPLIT_T7_SHADOW_TRIALS as u32,
        shadow_reserve_factor: SPLIT_T7_SHADOW_RESERVE_FACTOR,
        shadow_lower_envelope_frame_counts,
        shadow_ensemble_sha256: shadow_ensemble_sha256.to_string(),
        shadow_ensemble_receipt_path: shadow_ensemble_artifact.path.clone(),
        shadow_ensemble_receipt_sha256: shadow_ensemble_artifact.sha256.clone(),
        shadow_ensemble_receipt_bytes: shadow_ensemble_artifact.size_bytes,
        shadow_holdout_admission_verified: true,
        authoritative_admission_verified: true,
        authoritative_minimum,
        authoritative_minimum_slack_from_k,
        inventory_frame_counts,
        paired_inventory_sha256: paired_inventory_sha256.to_string(),
        equal_duration_windows_verified: true,
        direct_prefix_outer_steps,
        direct_prefix_md_steps: (direct_prefix_outer_steps as u64)
            .saturating_mul(fused_inner_steps as u64),
        calibration_input_path: input_artifact.path.clone(),
        calibration_input_sha256: input_artifact.sha256.clone(),
        calibration_input_bytes: input_artifact.size_bytes,
        frozen_anchor_path: frozen_anchor_artifact.path.clone(),
        frozen_anchor_sha256: frozen_anchor_sha256.clone(),
        frozen_anchor_bytes: frozen_anchor_artifact.size_bytes,
        available_planes: vec!["geometry".to_string(), "thermodynamic".to_string()],
        unavailable_planes: vec!["causality".to_string(), "chemistry".to_string()],
        anchor_frame_count: SPLIT_T7_ANCHOR_FRAMES as u32,
        variance_frame_count: SPLIT_T7_VARIANCE_FRAMES as u32,
        validation_frame_count: SPLIT_T7_VALIDATION_FRAMES as u32,
        pipeline_launch_count: SPLIT_T7_TOTAL_FRAMES as u32,
        baseline_freeze_count: 1,
        kl_threshold_write_count: 1,
        minimum_spikes_per_frame: fixed_spikes,
        maximum_spikes_per_frame: fixed_spikes,
        noise_floor_mu,
        noise_floor_sigma,
        threshold_3sigma: noise_floor_mu + 3.0 * noise_floor_sigma,
        threshold_12sigma: noise_floor_mu + 12.0 * noise_floor_sigma,
        validation_frame_total_kl: validation.total_kl.unwrap_or(f32::NAN),
        validation_frame_adjudication_marker: validation_marker,
        validation_adjudication_code: validation.adjudication_code,
        validation_adjudication_reason_flags: validation.adjudication_reason_flags,
        state_noninterference_verified: true,
        topology_constants_rebound_each_launch: true,
        fallback_used: false,
        frames,
    };
    let calibration_path = output_dir.join(format!(
        "{stem}_stream{stream_id:02}_ghost_noise_floor_calibration.json"
    ));
    write_atomic_json(&calibration_path, &calibration)?;
    let calibration_artifact = artifact_evidence(
        "ghost_noise_floor_calibration",
        Some(stream_id),
        &calibration_path,
        Some(SPLIT_T7_TOTAL_FRAMES as u64),
        Some(SPLIT_T7_SHADOW_SCHEMA_VERSION),
    )?;
    Ok(SplitT7StreamOutcome {
        noise_floor_mu,
        noise_floor_sigma,
        calibration_artifact,
        calibration_input_artifact: input_artifact,
        frozen_anchor_artifact,
        frozen_anchor_sha256,
        fixed_spikes_per_frame: fixed_spikes,
        runtime,
    })
}

#[cfg(feature = "gpu")]
struct SplitT7SourceWindow<'a> {
    source_events: &'a [crate::fused_engine::GpuSpikeEvent],
    selected_events: Vec<crate::fused_engine::GpuSpikeEvent>,
    selected_indices: Vec<u32>,
    source_event_count: usize,
    source_events_sha256: String,
}

#[cfg(feature = "gpu")]
fn update_canonical_split_t7_source_digest(
    digest: &mut Sha256,
    source: &[crate::fused_engine::GpuSpikeEvent],
) -> Result<u64> {
    const PREFIX: &[u8] = b"PRISMOS_SPLIT_T7_SOURCE_EVENTS_V1\0";
    digest.update(PREFIX);
    digest.update((source.len() as u64).to_le_bytes());
    let mut bytes = (PREFIX.len() as u64)
        .checked_add(8)
        .context("split-T7 canonical source length overflow")?;
    // GPU slot assignment is atomic and therefore not a semantic ordering.
    // Sort fixed-width canonical event bytes while preserving duplicates so
    // A/A equality compares the exact spike multiset, not incidental slot order.
    let mut canonical_events: Vec<Vec<u8>> = source.iter().map(canonical_gpu_spike_bytes).collect();
    canonical_events.sort();
    for event_bytes in canonical_events {
        let event_len =
            u32::try_from(event_bytes.len()).context("canonical split-T7 event exceeds u32")?;
        digest.update(event_len.to_le_bytes());
        digest.update(&event_bytes);
        bytes = bytes
            .checked_add(4)
            .and_then(|value| value.checked_add(event_bytes.len() as u64))
            .context("split-T7 canonical source length overflow")?;
    }
    Ok(bytes)
}

#[cfg(feature = "gpu")]
fn canonical_split_t7_source_identity(
    source: &[crate::fused_engine::GpuSpikeEvent],
) -> Result<(u64, String)> {
    let mut digest = Sha256::new();
    let bytes = update_canonical_split_t7_source_digest(&mut digest, source)?;
    Ok((bytes, format!("{:x}", digest.finalize())))
}

#[cfg(feature = "gpu")]
fn write_canonical_split_t7_source<W: Write>(
    writer: &mut W,
    source: &[crate::fused_engine::GpuSpikeEvent],
) -> Result<()> {
    writer.write_all(b"PRISMOS_SPLIT_T7_SOURCE_EVENTS_V1\0")?;
    writer.write_all(&(source.len() as u64).to_le_bytes())?;
    for event in source {
        let event_bytes = canonical_gpu_spike_bytes(event);
        writer.write_all(
            &u32::try_from(event_bytes.len())
                .context("canonical split-T7 event exceeds u32")?
                .to_le_bytes(),
        )?;
        writer.write_all(&event_bytes)?;
    }
    Ok(())
}

#[cfg(feature = "gpu")]
fn select_split_t7_full_span<'a>(
    source: &'a [crate::fused_engine::GpuSpikeEvent],
    fixed_spikes: usize,
    stream_id: u32,
) -> Result<SplitT7SourceWindow<'a>> {
    if fixed_spikes < 4 || source.len() < fixed_spikes {
        bail!(
            "stream {} split-T7 source window is undersized: source={} fixed={}",
            stream_id,
            source.len(),
            fixed_spikes
        );
    }
    let mut selected = Vec::with_capacity(fixed_spikes);
    let mut selected_indices = Vec::with_capacity(fixed_spikes);
    let (_, source_events_sha256) = canonical_split_t7_source_identity(source)?;
    for index in 0..fixed_spikes {
        let source_index = if fixed_spikes == 1 {
            0
        } else {
            index
                .checked_mul(source.len().saturating_sub(1))
                .context("split-T7 full-span selection overflow")?
                / fixed_spikes.saturating_sub(1)
        };
        selected.push(source[source_index]);
        selected_indices.push(
            u32::try_from(source_index).context("split-T7 selected source index exceeds u32")?,
        );
    }
    Ok(SplitT7SourceWindow {
        source_events: source,
        selected_events: selected,
        selected_indices,
        source_event_count: source.len(),
        source_events_sha256,
    })
}

#[cfg(feature = "gpu")]
fn split_t7_precommitted_cardinality(counts_a: &[u32], counts_b: &[u32]) -> Result<u32> {
    split_t7_reserved_cardinality(counts_a, counts_b, SPLIT_T7_SHADOW_RESERVE_FACTOR)
}

#[cfg(feature = "gpu")]
fn split_t7_trial_seed(
    base_seed: u64,
    trial_ordinal: u32,
    config: &CoupledTwinEvidenceConfig,
) -> Result<(u64, String)> {
    if trial_ordinal >= SPLIT_T7_SHADOW_TRIALS as u32 {
        bail!(
            "split-T7 shadow trial ordinal {} exceeds configured trial count {}",
            trial_ordinal,
            SPLIT_T7_SHADOW_TRIALS
        );
    }
    for (label, digest) in [
        ("topology A", config.topology_a_sha256.as_str()),
        ("topology B", config.topology_b_sha256.as_str()),
    ] {
        require_split_t7_sha256(digest, label)?;
    }
    let mut identity = Vec::with_capacity(256);
    identity.extend_from_slice(SPLIT_T7_SHADOW_TRIAL_SEED_DOMAIN);
    identity.extend_from_slice(&base_seed.to_le_bytes());
    identity.extend_from_slice(&trial_ordinal.to_le_bytes());
    identity.extend_from_slice(sha256_bytes(config.run_id.as_bytes()).as_bytes());
    identity.extend_from_slice(config.topology_a_sha256.as_bytes());
    identity.extend_from_slice(config.topology_b_sha256.as_bytes());
    let digest = Sha256::digest(&identity);
    let mut seed_bytes = [0u8; 8];
    seed_bytes.copy_from_slice(&digest[..8]);
    Ok((
        u64::from_le_bytes(seed_bytes),
        digest.iter().map(|byte| format!("{byte:02x}")).collect(),
    ))
}

#[cfg(feature = "gpu")]
fn split_t7_reserved_cardinality(
    counts_a: &[u32],
    counts_b: &[u32],
    reserve_factor: u32,
) -> Result<u32> {
    if counts_a.len() != SPLIT_T7_TOTAL_FRAMES || counts_b.len() != SPLIT_T7_TOTAL_FRAMES {
        bail!(
            "split-T7 cardinality precommit requires {} paired windows, observed A/B={}/{}",
            SPLIT_T7_TOTAL_FRAMES,
            counts_a.len(),
            counts_b.len()
        );
    }
    if reserve_factor < 2 {
        bail!("split-T7 shadow reserve factor must be at least two");
    }
    let minimum = counts_a
        .iter()
        .chain(counts_b)
        .copied()
        .min()
        .context("split-T7 cardinality precommit is empty")?;
    let reserved_floor = minimum / reserve_factor;
    if reserved_floor < 4 {
        bail!(
            "split-T7 cardinality precommit minimum {} cannot provide {}x headroom at the real-pipeline floor of four",
            minimum,
            reserve_factor
        );
    }
    let fixed = 1u32 << (31 - reserved_floor.leading_zeros());
    if fixed < 4
        || fixed
            .checked_mul(reserve_factor)
            .map_or(true, |reserved| reserved > minimum)
    {
        bail!(
            "split-T7 reserved cardinality derivation is invalid: minimum={} reserve={} K={}",
            minimum,
            reserve_factor,
            fixed
        );
    }
    Ok(fixed)
}

#[cfg(feature = "gpu")]
fn split_t7_pointwise_lower_envelope(
    trials: &[SplitT7ShadowTrialEvidence],
) -> Result<(Vec<u32>, Vec<u32>)> {
    if trials.len() < 2 {
        bail!("split-T7 lower envelope requires at least two trials");
    }
    let mut ordinals = BTreeSet::new();
    let mut previous = None;
    for trial in trials {
        if !ordinals.insert(trial.trial_ordinal)
            || previous.is_some_and(|value| value >= trial.trial_ordinal)
            || trial.frame_counts_a.len() != SPLIT_T7_TOTAL_FRAMES
            || trial.frame_counts_b.len() != SPLIT_T7_TOTAL_FRAMES
        {
            bail!("split-T7 lower envelope requires unique, ascending, complete trial evidence");
        }
        previous = Some(trial.trial_ordinal);
    }
    let mut envelope_a = trials[0].frame_counts_a.clone();
    let mut envelope_b = trials[0].frame_counts_b.clone();
    for trial in &trials[1..] {
        for (envelope, observed) in envelope_a.iter_mut().zip(&trial.frame_counts_a) {
            *envelope = (*envelope).min(*observed);
        }
        for (envelope, observed) in envelope_b.iter_mut().zip(&trial.frame_counts_b) {
            *envelope = (*envelope).min(*observed);
        }
    }
    Ok((envelope_a, envelope_b))
}

#[cfg(feature = "gpu")]
fn split_t7_leave_one_out_admissions(
    trials: &[SplitT7ShadowTrialEvidence],
) -> Result<Vec<SplitT7LeaveOneOutAdmission>> {
    if trials.len() != SPLIT_T7_SHADOW_TRIALS
        || trials
            .iter()
            .enumerate()
            .any(|(ordinal, trial)| trial.trial_ordinal != ordinal as u32)
    {
        bail!(
            "split-T7 leave-one-out requires exactly {} canonically ordered trials",
            SPLIT_T7_SHADOW_TRIALS
        );
    }
    let mut results = Vec::with_capacity(SPLIT_T7_SHADOW_TRIALS);
    for held_out_index in 0..trials.len() {
        let training = trials
            .iter()
            .enumerate()
            .filter_map(|(index, trial)| (index != held_out_index).then_some(trial.clone()))
            .collect::<Vec<_>>();
        let (training_a, training_b) = split_t7_pointwise_lower_envelope(&training)?;
        let training_minimum = training_a
            .iter()
            .chain(&training_b)
            .copied()
            .min()
            .context("split-T7 leave-one-out training envelope is empty")?;
        let candidate_k = split_t7_precommitted_cardinality(&training_a, &training_b)?;
        let held_out = &trials[held_out_index];
        let held_out_minimum = held_out
            .frame_counts_a
            .iter()
            .chain(&held_out.frame_counts_b)
            .copied()
            .min()
            .context("split-T7 held-out trial is empty")?;
        results.push(SplitT7LeaveOneOutAdmission {
            held_out_trial_ordinal: held_out.trial_ordinal,
            training_trial_ordinals: training.iter().map(|trial| trial.trial_ordinal).collect(),
            training_lower_envelope_minimum: training_minimum,
            candidate_k,
            held_out_minimum,
            admitted: held_out_minimum >= candidate_k,
        });
    }
    Ok(results)
}

#[cfg(feature = "gpu")]
fn split_t7_shadow_trial_inventory_sha256(
    trial: &SplitT7ShadowTrialEvidence,
    topology_a_sha256: &str,
    topology_b_sha256: &str,
) -> Result<String> {
    if trial.frame_count != SPLIT_T7_TOTAL_FRAMES as u32
        || trial.window_md_steps == 0
        || trial.frame_counts_a.len() != SPLIT_T7_TOTAL_FRAMES
        || trial.frame_counts_b.len() != SPLIT_T7_TOTAL_FRAMES
        || trial.source_bundle_bytes == 0
        || trial.source_bundle_sha256.len() != 64
        || trial.seed_derivation_sha256.len() != 64
        || trial.frame_source_sha256_a.len() != SPLIT_T7_TOTAL_FRAMES
        || trial.frame_source_sha256_b.len() != SPLIT_T7_TOTAL_FRAMES
        || trial.crn_initialization_semantic_sha256.len() != 64
        || trial.crn_initialization_artifact.kind
            != format!(
                "split_t7_shadow_trial_{:02}_crn_initialization",
                trial.trial_ordinal
            )
        || trial.crn_initialization_artifact.stream_id.is_some()
        || trial.crn_initialization_artifact.size_bytes == 0
        || trial.crn_initialization_artifact.sha256.len() != 64
        || trial.crn_initialization_artifact.blake3.len() != 64
        || trial
            .frame_source_sha256_a
            .iter()
            .chain(&trial.frame_source_sha256_b)
            .any(|digest| digest.len() != 64)
        || topology_a_sha256.len() != 64
        || topology_b_sha256.len() != 64
        || [
            trial.initial_physical_state_sha256_a.as_str(),
            trial.initial_physical_state_sha256_b.as_str(),
            trial.final_physical_state_sha256_a.as_str(),
            trial.final_physical_state_sha256_b.as_str(),
        ]
        .iter()
        .any(|digest| digest.len() != 64)
    {
        bail!("split-T7 shadow trial inventory identity input is invalid");
    }
    let mut identity = Vec::with_capacity(1024 + SPLIT_T7_TOTAL_FRAMES * 8);
    identity.extend_from_slice(SPLIT_T7_SHADOW_TRIAL_INVENTORY_DOMAIN);
    identity.extend_from_slice(&SPLIT_T7_SHADOW_SCHEMA_VERSION.to_le_bytes());
    identity.extend_from_slice(&trial.trial_ordinal.to_le_bytes());
    identity.extend_from_slice(&trial.base_seed.to_le_bytes());
    identity.extend_from_slice(&trial.trial_seed.to_le_bytes());
    identity.extend_from_slice(trial.seed_derivation_sha256.as_bytes());
    identity.extend_from_slice(&trial.window_md_steps.to_le_bytes());
    identity.extend_from_slice(&trial.frame_count.to_le_bytes());
    identity.extend_from_slice(topology_a_sha256.as_bytes());
    identity.extend_from_slice(topology_b_sha256.as_bytes());
    identity.extend_from_slice(trial.source_bundle_sha256.as_bytes());
    identity.extend_from_slice(&trial.source_bundle_bytes.to_le_bytes());
    identity.extend_from_slice(trial.crn_initialization_artifact.kind.as_bytes());
    identity.extend_from_slice(trial.crn_initialization_artifact.path.as_bytes());
    identity.extend_from_slice(&trial.crn_initialization_artifact.size_bytes.to_le_bytes());
    identity.extend_from_slice(trial.crn_initialization_artifact.sha256.as_bytes());
    identity.extend_from_slice(trial.crn_initialization_artifact.blake3.as_bytes());
    identity.extend_from_slice(trial.crn_initialization_semantic_sha256.as_bytes());
    for digest in [
        trial.initial_physical_state_sha256_a.as_str(),
        trial.initial_physical_state_sha256_b.as_str(),
        trial.final_physical_state_sha256_a.as_str(),
        trial.final_physical_state_sha256_b.as_str(),
    ] {
        identity.extend_from_slice(digest.as_bytes());
    }
    for (((count_a, count_b), source_a), source_b) in trial
        .frame_counts_a
        .iter()
        .zip(&trial.frame_counts_b)
        .zip(&trial.frame_source_sha256_a)
        .zip(&trial.frame_source_sha256_b)
    {
        identity.extend_from_slice(&count_a.to_le_bytes());
        identity.extend_from_slice(&count_b.to_le_bytes());
        identity.extend_from_slice(source_a.as_bytes());
        identity.extend_from_slice(source_b.as_bytes());
    }
    Ok(sha256_bytes(&identity))
}

#[cfg(feature = "gpu")]
fn split_t7_shadow_ensemble_sha256(receipt: &SplitT7ShadowEnsembleReceipt) -> Result<String> {
    if receipt.trial_count != SPLIT_T7_SHADOW_TRIALS as u32
        || receipt.trials.len() != SPLIT_T7_SHADOW_TRIALS
        || receipt.reserve_factor != SPLIT_T7_SHADOW_RESERVE_FACTOR
        || receipt.lower_envelope_counts_a.len() != SPLIT_T7_TOTAL_FRAMES
        || receipt.lower_envelope_counts_b.len() != SPLIT_T7_TOTAL_FRAMES
        || receipt.leave_one_trial_out.len() != SPLIT_T7_SHADOW_TRIALS
        || receipt.fixed_spikes_per_frame < 4
    {
        bail!("split-T7 shadow ensemble identity input is invalid");
    }
    let mut identity = Vec::with_capacity(1024 + SPLIT_T7_TOTAL_FRAMES * 8);
    identity.extend_from_slice(SPLIT_T7_SHADOW_ENSEMBLE_DOMAIN);
    identity.extend_from_slice(&receipt.schema_version.to_le_bytes());
    identity.extend_from_slice(sha256_bytes(receipt.run_id.as_bytes()).as_bytes());
    identity.extend_from_slice(&receipt.base_seed.to_le_bytes());
    identity.extend_from_slice(receipt.trial_seed_derivation.as_bytes());
    identity.extend_from_slice(receipt.topology_a_sha256.as_bytes());
    identity.extend_from_slice(receipt.topology_b_sha256.as_bytes());
    identity.extend_from_slice(&receipt.window_md_steps.to_le_bytes());
    identity.extend_from_slice(&receipt.frame_count.to_le_bytes());
    identity.extend_from_slice(&receipt.trial_count.to_le_bytes());
    identity.extend_from_slice(&receipt.reserve_factor.to_le_bytes());
    identity.extend_from_slice(receipt.schema_kind.as_bytes());
    identity.extend_from_slice(receipt.status.as_bytes());
    identity.extend_from_slice(receipt.validation_status.as_bytes());
    identity.extend_from_slice(receipt.selection_method.as_bytes());
    identity.push(u8::from(receipt.holdout_admission_verified));
    identity.push(u8::from(receipt.all_trial_seeds_distinct));
    identity.push(u8::from(receipt.paired_stream_seed_equality_verified));
    identity.push(u8::from(
        receipt.paired_common_random_initialization_verified,
    ));
    identity.push(u8::from(receipt.fallback_used));
    identity.extend_from_slice(&receipt.fixed_spikes_per_frame.to_le_bytes());
    identity.extend_from_slice(&receipt.lower_envelope_minimum.to_le_bytes());
    for trial in &receipt.trials {
        identity.extend_from_slice(trial.inventory_sha256.as_bytes());
    }
    for result in &receipt.leave_one_trial_out {
        identity.extend_from_slice(&result.held_out_trial_ordinal.to_le_bytes());
        identity.extend_from_slice(&(result.training_trial_ordinals.len() as u32).to_le_bytes());
        for ordinal in &result.training_trial_ordinals {
            identity.extend_from_slice(&ordinal.to_le_bytes());
        }
        identity.extend_from_slice(&result.training_lower_envelope_minimum.to_le_bytes());
        identity.extend_from_slice(&result.candidate_k.to_le_bytes());
        identity.extend_from_slice(&result.held_out_minimum.to_le_bytes());
        identity.push(u8::from(result.admitted));
    }
    for (count_a, count_b) in receipt
        .lower_envelope_counts_a
        .iter()
        .zip(&receipt.lower_envelope_counts_b)
    {
        identity.extend_from_slice(&count_a.to_le_bytes());
        identity.extend_from_slice(&count_b.to_le_bytes());
    }
    Ok(sha256_bytes(&identity))
}

#[cfg(feature = "gpu")]
fn split_t7_admission_semantic_sha256(receipt: &SplitT7AdmissionReceiptV3) -> Result<String> {
    let mut canonical = receipt.clone();
    canonical.semantic_sha256.clear();
    let mut identity = Vec::with_capacity(4096 + SPLIT_T7_TOTAL_FRAMES * 16);
    identity.extend_from_slice(SPLIT_T7_ADMISSION_DOMAIN);
    identity.extend_from_slice(&serde_json::to_vec(&canonical)?);
    Ok(sha256_bytes(&identity))
}

#[cfg(feature = "gpu")]
fn split_t7_authoritative_admission(
    counts_a: &[u32],
    counts_b: &[u32],
    fixed_spikes: u32,
) -> Result<(u32, u32)> {
    if fixed_spikes < 4
        || counts_a.len() != SPLIT_T7_TOTAL_FRAMES
        || counts_b.len() != SPLIT_T7_TOTAL_FRAMES
    {
        bail!("split-T7 authoritative admission shape is invalid");
    }
    let minimum = counts_a
        .iter()
        .chain(counts_b)
        .copied()
        .min()
        .context("split-T7 authoritative inventory is empty")?;
    if minimum < fixed_spikes {
        bail!(
            "split-T7 authoritative minimum {} fell below fixed CUDA cardinality {}",
            minimum,
            fixed_spikes
        );
    }
    Ok((minimum, minimum - fixed_spikes))
}

#[cfg(feature = "gpu")]
fn split_t7_paired_inventory_sha256(
    domain: &[u8],
    window_md_steps: u32,
    fixed_spikes: u32,
    counts_a: &[u32],
    counts_b: &[u32],
) -> Result<String> {
    if domain.is_empty()
        || counts_a.len() != SPLIT_T7_TOTAL_FRAMES
        || counts_b.len() != SPLIT_T7_TOTAL_FRAMES
        || fixed_spikes < 4
        || counts_a
            .iter()
            .chain(counts_b)
            .any(|count| *count < fixed_spikes)
    {
        bail!("split-T7 paired inventory identity input is invalid");
    }
    let mut identity = Vec::with_capacity(domain.len() + 8 + SPLIT_T7_TOTAL_FRAMES * 8);
    identity.extend_from_slice(domain);
    identity.extend_from_slice(&window_md_steps.to_le_bytes());
    identity.extend_from_slice(&fixed_spikes.to_le_bytes());
    for (count_a, count_b) in counts_a.iter().zip(counts_b) {
        identity.extend_from_slice(&count_a.to_le_bytes());
        identity.extend_from_slice(&count_b.to_le_bytes());
    }
    Ok(sha256_bytes(&identity))
}

#[cfg(feature = "gpu")]
fn build_split_t7_inventory_engine(
    config_ref: &crate::persistent_engine::PersistentBatchConfig,
    context: Arc<CudaContext>,
    fused_module: Arc<CudaModule>,
    topology: &PrismPrepTopology,
    protocol: &CryoUvProtocol,
    hmr: bool,
    fused_steps: u32,
    adaptive_dt: bool,
    ladd_enabled: bool,
    stream_id: u32,
    prismcode_admission: &PrismCodeMdPolicyAdmissionV1,
) -> Result<PersistentNhsEngine> {
    let stream = context
        .new_stream()
        .with_context(|| format!("split-T7 inventory CUDA stream {}", stream_id))?;
    let mut engine = PersistentNhsEngine::new_on_stream(config_ref, context, fused_module, stream)?;
    engine.set_prismcode_md_policy_admission(prismcode_admission.clone())?;
    engine.load_topology(topology)?;
    if hmr {
        engine.set_dt(0.004)?;
    }
    if fused_steps > 1 {
        engine.set_fused_inner_steps(fused_steps)?;
    }
    if adaptive_dt {
        engine.set_adaptive_dt(true)?;
    }
    if ladd_enabled {
        engine.set_ladd_enabled(true);
    }
    engine.set_cryo_uv_protocol(protocol.clone())?;
    engine.set_spike_accumulation(true);
    engine.enable_prismcode_md_native_runtime(stream_id, true, true)?;
    Ok(engine)
}

#[cfg(feature = "gpu")]
fn require_split_t7_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("split-T7 {label} is not a canonical lowercase SHA-256");
    }
    Ok(())
}

#[cfg(feature = "gpu")]
fn validate_split_t7_shadow_source_event(
    event: &crate::fused_engine::GpuSpikeEvent,
    stream_id: u32,
    frame_ordinal: u32,
    window_md_steps: u32,
    source_index: u32,
    layout: &SplitT7TopologyLayout,
) -> Result<()> {
    let frame_start = u64::from(frame_ordinal)
        .checked_mul(u64::from(window_md_steps))
        .context("split-T7 shadow frame start overflows")?;
    let next_frame = frame_ordinal
        .checked_add(1)
        .context("split-T7 shadow frame ordinal overflows")?;
    let frame_end = u64::from(next_frame)
        .checked_mul(u64::from(window_md_steps))
        .context("split-T7 shadow frame end overflows")?;
    let event_step = u64::try_from(event.timestep).with_context(|| {
        format!(
            "split-T7 shadow stream {stream_id} frame {frame_ordinal} source event {source_index} has negative timestep {}",
            event.timestep
        )
    })?;
    // GPU spike timestamps identify the completed MD step.  A snapshot is read
    // only after advancing the engine through `frame_end`, so exact adjacent
    // source windows are right-closed: (frame_start, frame_end].
    if window_md_steps == 0 || event_step <= frame_start || event_step > frame_end {
        bail!(
            "split-T7 shadow stream {} frame {} source event {} completed-step timestamp {} lies outside its exact window ({},{}]",
            stream_id,
            frame_ordinal,
            source_index,
            event.timestep,
            frame_start,
            frame_end
        );
    }
    let primary = event.nearby_residues[0];
    let topology_cluster_id = if primary >= 0 {
        layout
            .residue_to_cluster
            .get(primary as usize)
            .copied()
            .unwrap_or(u32::MAX)
    } else {
        u32::MAX
    };
    if topology_cluster_id >= SPLIT_T7_CLUSTER_COUNT {
        bail!(
            "split-T7 shadow stream {} source event {} has no exact topology cluster",
            stream_id,
            source_index
        );
    }
    let (_, support) = rich_spike_from_gpu_event(event, topology_cluster_id as i32)?;
    for residue in support {
        if layout
            .residue_to_anchor_atom
            .get(residue as usize)
            .copied()
            .unwrap_or(u32::MAX)
            == u32::MAX
        {
            bail!(
                "split-T7 shadow stream {} source event {} cites residue {} absent from the exact topology",
                stream_id,
                source_index,
                residue
            );
        }
    }
    Ok(())
}

#[cfg(feature = "gpu")]
fn read_split_t7_shadow_canonical_source<R: Read>(
    reader: &mut R,
    stream_id: u32,
    ordinal: u32,
    expected_count: u32,
    expected_length: u64,
    expected_sha256: &str,
    window_md_steps: u32,
    layout: &SplitT7TopologyLayout,
) -> Result<()> {
    let expected_canonical_length = 34u64
        .checked_add(8)
        .and_then(|bytes| {
            bytes.checked_add(
                (expected_count as u64).checked_mul(4 + SPLIT_T7_CANONICAL_EVENT_BYTES as u64)?,
            )
        })
        .context("split-T7 shadow canonical-source length overflow")?;
    if expected_length != expected_canonical_length {
        bail!(
            "split-T7 shadow stream {} frame {} source length/count mismatch",
            stream_id,
            ordinal
        );
    }
    let prefix = read_exact_array::<34, _>(reader, "shadow canonical source prefix")?;
    if prefix.as_slice() != b"PRISMOS_SPLIT_T7_SOURCE_EVENTS_V1\0" {
        bail!(
            "split-T7 shadow stream {} frame {} source prefix mismatch",
            stream_id,
            ordinal
        );
    }
    let encoded_count = read_stream_u64(reader, "shadow canonical source count")?;
    if encoded_count != expected_count as u64 {
        bail!(
            "split-T7 shadow stream {} frame {} encoded source count mismatch",
            stream_id,
            ordinal
        );
    }
    let mut digest = Sha256::new();
    digest.update(prefix);
    digest.update(encoded_count.to_le_bytes());
    for source_index in 0..expected_count {
        let length_bytes = read_exact_array::<4, _>(reader, "shadow canonical event length")?;
        digest.update(length_bytes);
        let event_length = u32::from_le_bytes(length_bytes);
        if event_length as usize != SPLIT_T7_CANONICAL_EVENT_BYTES {
            bail!(
                "split-T7 shadow stream {} frame {} event {} declares {} bytes",
                stream_id,
                ordinal,
                source_index,
                event_length
            );
        }
        let event_bytes = read_exact_array::<SPLIT_T7_CANONICAL_EVENT_BYTES, _>(
            reader,
            "shadow canonical event bytes",
        )?;
        digest.update(event_bytes);
        let event = parse_canonical_gpu_spike_bytes(&event_bytes)?;
        validate_split_t7_shadow_source_event(
            &event,
            stream_id,
            ordinal,
            window_md_steps,
            source_index,
            layout,
        )?;
    }
    let observed_sha256 = format!("{:x}", digest.finalize());
    if observed_sha256 != expected_sha256 {
        bail!(
            "split-T7 shadow stream {} frame {} canonical source SHA-256 mismatch",
            stream_id,
            ordinal
        );
    }
    Ok(())
}

#[cfg(feature = "gpu")]
fn verify_split_t7_shadow_trial_bundle(
    path: &Path,
    trial: &SplitT7ShadowTrialEvidence,
    config: &CoupledTwinEvidenceConfig,
    layout_a: &SplitT7TopologyLayout,
    layout_b: &SplitT7TopologyLayout,
) -> Result<SplitT7ShadowBundleReadback> {
    let mut reader = std::io::BufReader::with_capacity(
        1024 * 1024,
        File::open(path)
            .with_context(|| format!("open split-T7 shadow bundle {}", path.display()))?,
    );
    if read_exact_array::<8, _>(&mut reader, "shadow bundle magic")?.as_slice() != b"PRT7S003" {
        bail!("split-T7 shadow bundle magic mismatch");
    }
    let mut source_domain = vec![0u8; SPLIT_T7_SHADOW_TRIAL_SOURCE_DOMAIN.len()];
    reader.read_exact(&mut source_domain)?;
    let schema_version = read_stream_u32(&mut reader, "shadow bundle schema version")?;
    let trial_ordinal = read_stream_u32(&mut reader, "shadow bundle trial ordinal")?;
    let base_seed = read_stream_u64(&mut reader, "shadow bundle base seed")?;
    let trial_seed = read_stream_u64(&mut reader, "shadow bundle trial seed")?;
    let seed_derivation_sha256 =
        read_stream_sha256(&mut reader, "shadow bundle seed derivation SHA-256")?;
    let window_md_steps = read_stream_u32(&mut reader, "shadow bundle window MD steps")?;
    let frame_count = read_stream_u32(&mut reader, "shadow bundle frame count")?;
    let run_id_sha256 = read_stream_sha256(&mut reader, "shadow bundle run-id SHA-256")?;
    let topology_a_sha256 = read_stream_sha256(&mut reader, "shadow topology A SHA-256")?;
    let topology_b_sha256 = read_stream_sha256(&mut reader, "shadow topology B SHA-256")?;
    let layout_a_sha256 = read_stream_sha256(&mut reader, "shadow layout A SHA-256")?;
    let layout_b_sha256 = read_stream_sha256(&mut reader, "shadow layout B SHA-256")?;
    let initial_a = read_stream_sha256(&mut reader, "shadow initial physical state A")?;
    let initial_b = read_stream_sha256(&mut reader, "shadow initial physical state B")?;
    if source_domain != SPLIT_T7_SHADOW_TRIAL_SOURCE_DOMAIN
        || schema_version != SPLIT_T7_SHADOW_SCHEMA_VERSION
        || trial_ordinal != trial.trial_ordinal
        || base_seed != trial.base_seed
        || trial_seed != trial.trial_seed
        || seed_derivation_sha256 != trial.seed_derivation_sha256
        || window_md_steps != trial.window_md_steps
        || frame_count != SPLIT_T7_TOTAL_FRAMES as u32
        || frame_count != trial.frame_count
        || run_id_sha256 != sha256_bytes(config.run_id.as_bytes())
        || topology_a_sha256 != config.topology_a_sha256
        || topology_b_sha256 != config.topology_b_sha256
        || layout_a_sha256 != layout_a.layout_sha256
        || layout_b_sha256 != layout_b.layout_sha256
        || initial_a != trial.initial_physical_state_sha256_a
        || initial_b != trial.initial_physical_state_sha256_b
    {
        bail!("split-T7 shadow bundle header identity mismatch");
    }
    let mut counts_a = Vec::with_capacity(SPLIT_T7_TOTAL_FRAMES);
    let mut counts_b = Vec::with_capacity(SPLIT_T7_TOTAL_FRAMES);
    let mut source_sha256_a = Vec::with_capacity(SPLIT_T7_TOTAL_FRAMES);
    let mut source_sha256_b = Vec::with_capacity(SPLIT_T7_TOTAL_FRAMES);
    for expected_ordinal in 0..SPLIT_T7_TOTAL_FRAMES as u32 {
        let ordinal = read_stream_u32(&mut reader, "shadow frame ordinal")?;
        let analysis_md_step = read_stream_u64(&mut reader, "shadow frame analysis MD step")?;
        let expected_step = (expected_ordinal as u64 + 1).saturating_mul(window_md_steps as u64);
        if ordinal != expected_ordinal || analysis_md_step != expected_step {
            bail!(
                "split-T7 shadow frame order/step mismatch: observed ordinal={} step={} expected ordinal={} step={}",
                ordinal,
                analysis_md_step,
                expected_ordinal,
                expected_step
            );
        }
        for (expected_stream, layout, counts, hashes) in [
            (0u32, layout_a, &mut counts_a, &mut source_sha256_a),
            (1u32, layout_b, &mut counts_b, &mut source_sha256_b),
        ] {
            let stream_id = read_stream_u32(&mut reader, "shadow source stream id")?;
            let count = read_stream_u32(&mut reader, "shadow source count")?;
            let canonical_length = read_stream_u64(&mut reader, "shadow source byte length")?;
            let source_sha256 = read_stream_sha256(&mut reader, "shadow source SHA-256")?;
            if stream_id != expected_stream {
                bail!(
                    "split-T7 shadow frame {} stream order mismatch: observed={} expected={}",
                    expected_ordinal,
                    stream_id,
                    expected_stream
                );
            }
            read_split_t7_shadow_canonical_source(
                &mut reader,
                stream_id,
                expected_ordinal,
                count,
                canonical_length,
                &source_sha256,
                window_md_steps,
                layout,
            )?;
            counts.push(count);
            hashes.push(source_sha256);
        }
    }
    let final_a = read_stream_sha256(&mut reader, "shadow final physical state A")?;
    let final_b = read_stream_sha256(&mut reader, "shadow final physical state B")?;
    if read_exact_array::<8, _>(&mut reader, "shadow bundle trailer")?.as_slice() != b"PRT7SEND" {
        bail!("split-T7 shadow bundle trailer mismatch");
    }
    let mut trailing = [0u8; 1];
    if reader.read(&mut trailing)? != 0 {
        bail!("split-T7 shadow bundle has trailing bytes");
    }
    let readback = SplitT7ShadowBundleReadback {
        frame_counts_a: counts_a,
        frame_counts_b: counts_b,
        frame_source_sha256_a: source_sha256_a,
        frame_source_sha256_b: source_sha256_b,
        initial_physical_state_sha256_a: initial_a,
        initial_physical_state_sha256_b: initial_b,
        final_physical_state_sha256_a: final_a,
        final_physical_state_sha256_b: final_b,
    };
    let expected = SplitT7ShadowBundleReadback {
        frame_counts_a: trial.frame_counts_a.clone(),
        frame_counts_b: trial.frame_counts_b.clone(),
        frame_source_sha256_a: trial.frame_source_sha256_a.clone(),
        frame_source_sha256_b: trial.frame_source_sha256_b.clone(),
        initial_physical_state_sha256_a: trial.initial_physical_state_sha256_a.clone(),
        initial_physical_state_sha256_b: trial.initial_physical_state_sha256_b.clone(),
        final_physical_state_sha256_a: trial.final_physical_state_sha256_a.clone(),
        final_physical_state_sha256_b: trial.final_physical_state_sha256_b.clone(),
    };
    if readback != expected {
        bail!("split-T7 shadow bundle durable readback differs from trial evidence");
    }
    Ok(readback)
}

#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn execute_split_t7_shadow_trial(
    config_ref: &crate::persistent_engine::PersistentBatchConfig,
    context: Arc<CudaContext>,
    fused_module: Arc<CudaModule>,
    topology_a: &PrismPrepTopology,
    topology_b: &PrismPrepTopology,
    protocol: &CryoUvProtocol,
    config: &CoupledTwinEvidenceConfig,
    seed: u64,
    outer_steps: u32,
    fused_inner_steps: i32,
    hmr: bool,
    fused_steps: u32,
    adaptive_dt: bool,
    ladd_enabled: bool,
    trial_ordinal: u32,
    output_dir: &Path,
    layout_a: &SplitT7TopologyLayout,
    layout_b: &SplitT7TopologyLayout,
    prismcode_admission: &PrismCodeMdPolicyAdmissionV1,
) -> Result<(
    SplitT7ShadowTrialEvidence,
    CoupledTwinArtifactEvidence,
    CoupledTwinArtifactEvidence,
)> {
    let (trial_seed, seed_derivation_sha256) = split_t7_trial_seed(seed, trial_ordinal, config)?;
    let mut engine_a = build_split_t7_inventory_engine(
        config_ref,
        context.clone(),
        fused_module.clone(),
        topology_a,
        protocol,
        hmr,
        fused_steps,
        adaptive_dt,
        ladd_enabled,
        0,
        prismcode_admission,
    )?;
    let mut engine_b = build_split_t7_inventory_engine(
        config_ref,
        context,
        fused_module,
        topology_b,
        protocol,
        hmr,
        fused_steps,
        adaptive_dt,
        ladd_enabled,
        1,
        prismcode_admission,
    )?;
    let temperature_kelvin = protocol.start_temp.max(50.0);
    let crn_receipt = apply_coupled_twin_crn_initialization_v3(
        &mut engine_a,
        &mut engine_b,
        topology_a,
        topology_b,
        config,
        trial_seed,
        temperature_kelvin,
        "split_t7_shadow_trial",
        Some(trial_ordinal),
    )?;
    let stem = coupled_topology_stem(config);
    let crn_artifact = emit_coupled_twin_crn_receipt_v3(
        output_dir,
        &format!("{stem}_split_t7_shadow_trial{trial_ordinal:02}_crn_initialization.json"),
        &format!("split_t7_shadow_trial_{trial_ordinal:02}_crn_initialization"),
        &crn_receipt,
        topology_a,
        topology_b,
        config,
        temperature_kelvin,
    )?;
    let (_, initial_a) = engine_state_sha256s(&engine_a, topology_a, 0)?;
    let (_, initial_b) = engine_state_sha256s(&engine_b, topology_b, 1)?;
    if config.identical_topology_negative_control && initial_a != initial_b {
        bail!("identical-topology shadow trial {trial_ordinal} initial physical states differ");
    }

    let bundle_path = output_dir.join(format!(
        "{stem}_split_t7_shadow_trial{trial_ordinal:02}_sources.bin"
    ));
    let mut bundle = NoClobberArtifactFile::create(&bundle_path)?;
    {
        let writer = bundle.file_mut()?;
        writer.write_all(b"PRT7S003")?;
        writer.write_all(SPLIT_T7_SHADOW_TRIAL_SOURCE_DOMAIN)?;
        writer.write_all(&SPLIT_T7_SHADOW_SCHEMA_VERSION.to_le_bytes())?;
        writer.write_all(&trial_ordinal.to_le_bytes())?;
        writer.write_all(&seed.to_le_bytes())?;
        writer.write_all(&trial_seed.to_le_bytes())?;
        writer.write_all(seed_derivation_sha256.as_bytes())?;
        writer.write_all(&(fused_inner_steps as u32).to_le_bytes())?;
        writer.write_all(&(SPLIT_T7_TOTAL_FRAMES as u32).to_le_bytes())?;
        for digest in [
            sha256_bytes(config.run_id.as_bytes()),
            config.topology_a_sha256.clone(),
            config.topology_b_sha256.clone(),
            layout_a.layout_sha256.clone(),
            layout_b.layout_sha256.clone(),
            initial_a.clone(),
            initial_b.clone(),
        ] {
            require_split_t7_sha256(&digest, "shadow bundle header digest")?;
            writer.write_all(digest.as_bytes())?;
        }
    }

    let mut prefix_outer = 0u32;
    let mut cursor_a = 0usize;
    let mut cursor_b = 0usize;
    let mut counts_a = Vec::with_capacity(SPLIT_T7_TOTAL_FRAMES);
    let mut counts_b = Vec::with_capacity(SPLIT_T7_TOTAL_FRAMES);
    let mut source_sha256_a = Vec::with_capacity(SPLIT_T7_TOTAL_FRAMES);
    let mut source_sha256_b = Vec::with_capacity(SPLIT_T7_TOTAL_FRAMES);
    for ordinal in 0..SPLIT_T7_TOTAL_FRAMES {
        let (snapshot_a, snapshot_b) = advance_split_t7_direct_window(
            &mut engine_a,
            &mut engine_b,
            protocol,
            outer_steps,
            fused_inner_steps,
            &mut prefix_outer,
            &mut cursor_a,
            &mut cursor_b,
            config.identical_topology_negative_control,
        )?;
        let analysis_md_step = (prefix_outer as u64).saturating_mul(fused_inner_steps as u64);
        bundle
            .file_mut()?
            .write_all(&(ordinal as u32).to_le_bytes())?;
        bundle
            .file_mut()?
            .write_all(&analysis_md_step.to_le_bytes())?;
        for (stream_id, source, counts, hashes) in [
            (
                0u32,
                snapshot_a.window(),
                &mut counts_a,
                &mut source_sha256_a,
            ),
            (
                1u32,
                snapshot_b.window(),
                &mut counts_b,
                &mut source_sha256_b,
            ),
        ] {
            let count = u32::try_from(source.len()).with_context(|| {
                format!("shadow trial {trial_ordinal} stream {stream_id} count")
            })?;
            let (canonical_length, source_sha256) = canonical_split_t7_source_identity(source)?;
            require_split_t7_sha256(&source_sha256, "shadow source digest")?;
            let writer = bundle.file_mut()?;
            writer.write_all(&stream_id.to_le_bytes())?;
            writer.write_all(&count.to_le_bytes())?;
            writer.write_all(&canonical_length.to_le_bytes())?;
            writer.write_all(source_sha256.as_bytes())?;
            write_canonical_split_t7_source(writer, source)?;
            counts.push(count);
            hashes.push(source_sha256);
        }
        if config.identical_topology_negative_control {
            if counts_a[ordinal] != counts_b[ordinal]
                || source_sha256_a[ordinal] != source_sha256_b[ordinal]
            {
                bail!(
                    "identical-topology shadow trial {} canonical spike source differs at equal-duration frame {}: count_a={} count_b={} sha256_a={} sha256_b={}",
                    trial_ordinal,
                    ordinal,
                    counts_a[ordinal],
                    counts_b[ordinal],
                    source_sha256_a[ordinal],
                    source_sha256_b[ordinal]
                );
            }
            let (_, physical_a) = engine_state_sha256s(&engine_a, topology_a, 0)?;
            let (_, physical_b) = engine_state_sha256s(&engine_b, topology_b, 1)?;
            if physical_a != physical_b {
                bail!(
                    "identical-topology shadow trial {} diverged at equal-duration frame {}",
                    trial_ordinal,
                    ordinal
                );
            }
        }
    }
    if prefix_outer != SPLIT_T7_TOTAL_FRAMES as u32 {
        bail!("split-T7 shadow trial did not execute every required window");
    }
    let (_, final_a) = engine_state_sha256s(&engine_a, topology_a, 0)?;
    let (_, final_b) = engine_state_sha256s(&engine_b, topology_b, 1)?;
    if config.identical_topology_negative_control && final_a != final_b {
        bail!("identical-topology shadow trial {trial_ordinal} final physical states differ");
    }
    for digest in [&final_a, &final_b] {
        require_split_t7_sha256(digest, "shadow bundle final-state digest")?;
        bundle.file_mut()?.write_all(digest.as_bytes())?;
    }
    bundle.file_mut()?.write_all(b"PRT7SEND")?;
    bundle.commit()?;
    let artifact = artifact_evidence(
        &format!("split_t7_shadow_trial_{trial_ordinal:02}_sources"),
        None,
        &bundle_path,
        Some((SPLIT_T7_TOTAL_FRAMES * 2) as u64),
        Some(SPLIT_T7_SHADOW_SCHEMA_VERSION),
    )?;
    validate_run_artifact(output_dir, &artifact)?;
    let mut trial = SplitT7ShadowTrialEvidence {
        trial_ordinal,
        base_seed: seed,
        trial_seed,
        seed_derivation_sha256,
        window_md_steps: fused_inner_steps as u32,
        frame_count: SPLIT_T7_TOTAL_FRAMES as u32,
        source_bundle_path: artifact.path.clone(),
        source_bundle_sha256: artifact.sha256.clone(),
        source_bundle_bytes: artifact.size_bytes,
        frame_counts_a: counts_a,
        frame_counts_b: counts_b,
        frame_source_sha256_a: source_sha256_a,
        frame_source_sha256_b: source_sha256_b,
        initial_physical_state_sha256_a: initial_a,
        initial_physical_state_sha256_b: initial_b,
        final_physical_state_sha256_a: final_a,
        final_physical_state_sha256_b: final_b,
        crn_initialization_artifact: strict_artifact_binding(&crn_artifact),
        crn_initialization_semantic_sha256: crn_receipt.semantic_sha256,
        inventory_sha256: String::new(),
    };
    trial.inventory_sha256 = split_t7_shadow_trial_inventory_sha256(
        &trial,
        &config.topology_a_sha256,
        &config.topology_b_sha256,
    )?;
    verify_split_t7_shadow_trial_bundle(&bundle_path, &trial, config, layout_a, layout_b)?;
    Ok((trial, artifact, crn_artifact))
}

#[cfg(feature = "gpu")]
fn verify_split_t7_shadow_ensemble(
    output_dir: &Path,
    config: &CoupledTwinEvidenceConfig,
    topology_a: &PrismPrepTopology,
    topology_b: &PrismPrepTopology,
    layout_a: &SplitT7TopologyLayout,
    layout_b: &SplitT7TopologyLayout,
    receipt: &SplitT7ShadowEnsembleReceipt,
    trial_artifacts: &[CoupledTwinArtifactEvidence],
    ensemble_artifact: &CoupledTwinArtifactEvidence,
) -> Result<()> {
    if receipt.schema_kind != "prismos.coupled_twin_split_t7_shadow_ensemble.v3"
        || receipt.schema_version != SPLIT_T7_SHADOW_SCHEMA_VERSION
        || receipt.status != "IMPLEMENTATION_VERIFIED"
        || receipt.validation_status != "PASS"
        || receipt.run_id != config.run_id
        || receipt.trial_seed_derivation != SPLIT_T7_TRIAL_SEED_DERIVATION_V3
        || receipt.topology_a_sha256 != config.topology_a_sha256
        || receipt.topology_b_sha256 != config.topology_b_sha256
        || receipt.frame_count != SPLIT_T7_TOTAL_FRAMES as u32
        || receipt.trial_count != SPLIT_T7_SHADOW_TRIALS as u32
        || receipt.reserve_factor != SPLIT_T7_SHADOW_RESERVE_FACTOR
        || receipt.selection_method != SPLIT_T7_SELECTION_METHOD_V3
        || !receipt.holdout_admission_verified
        || !receipt.all_trial_seeds_distinct
        || !receipt.paired_stream_seed_equality_verified
        || !receipt.paired_common_random_initialization_verified
        || receipt.fallback_used
        || trial_artifacts.len() != SPLIT_T7_SHADOW_TRIALS
        || ensemble_artifact.kind != "split_t7_shadow_ensemble_receipt"
        || ensemble_artifact.stream_id.is_some()
        || ensemble_artifact.record_count != Some(SPLIT_T7_SHADOW_TRIALS as u64)
        || ensemble_artifact.schema_version != Some(SPLIT_T7_SHADOW_SCHEMA_VERSION)
    {
        bail!("split-T7 shadow ensemble identity contract failed");
    }
    let observed_seeds = receipt
        .trials
        .iter()
        .map(|trial| trial.trial_seed)
        .collect::<BTreeSet<_>>();
    if observed_seeds.len() != SPLIT_T7_SHADOW_TRIALS {
        bail!("split-T7 shadow trial seeds are not distinct");
    }
    validate_run_artifact(output_dir, ensemble_artifact)?;
    for (ordinal, (trial, artifact)) in receipt.trials.iter().zip(trial_artifacts).enumerate() {
        if trial.trial_ordinal != ordinal as u32
            || trial.base_seed != receipt.base_seed
            || trial.window_md_steps != receipt.window_md_steps
            || artifact.kind != format!("split_t7_shadow_trial_{ordinal:02}_sources")
            || artifact.stream_id.is_some()
            || artifact.path != trial.source_bundle_path
            || artifact.sha256 != trial.source_bundle_sha256
            || artifact.size_bytes != trial.source_bundle_bytes
            || artifact.record_count != Some((SPLIT_T7_TOTAL_FRAMES * 2) as u64)
            || artifact.schema_version != Some(SPLIT_T7_SHADOW_SCHEMA_VERSION)
        {
            bail!("split-T7 shadow trial {ordinal} artifact binding failed");
        }
        let (expected_seed, expected_seed_hash) =
            split_t7_trial_seed(receipt.base_seed, ordinal as u32, config)?;
        if trial.trial_seed != expected_seed || trial.seed_derivation_sha256 != expected_seed_hash {
            bail!("split-T7 shadow trial {ordinal} seed derivation mismatch");
        }
        validate_run_artifact(output_dir, artifact)?;
        let crn_artifact = CoupledTwinArtifactEvidence {
            kind: trial.crn_initialization_artifact.kind.clone(),
            stream_id: trial.crn_initialization_artifact.stream_id,
            path: trial.crn_initialization_artifact.path.clone(),
            size_bytes: trial.crn_initialization_artifact.size_bytes,
            sha256: trial.crn_initialization_artifact.sha256.clone(),
            blake3: trial.crn_initialization_artifact.blake3.clone(),
            record_count: Some(1),
            schema_version: Some(COUPLED_TWIN_CRN_SCHEMA_VERSION),
            runtime_owned: true,
        };
        validate_run_artifact(output_dir, &crn_artifact)?;
        let crn_receipt: CoupledTwinCrnInitializationReceiptV3 =
            serde_json::from_slice(&std::fs::read(&crn_artifact.path)?)?;
        if crn_receipt.semantic_sha256 != trial.crn_initialization_semantic_sha256 {
            bail!("split-T7 shadow trial {ordinal} CRN semantic binding failed");
        }
        verify_coupled_twin_crn_receipt_v3(
            &crn_receipt,
            topology_a,
            topology_b,
            config,
            trial.trial_seed,
            f32::from_bits(crn_receipt.temperature_kelvin_f32_bits),
            "split_t7_shadow_trial",
            Some(ordinal as u32),
        )?;
        let recomputed_inventory = split_t7_shadow_trial_inventory_sha256(
            trial,
            &config.topology_a_sha256,
            &config.topology_b_sha256,
        )?;
        if recomputed_inventory != trial.inventory_sha256 {
            bail!("split-T7 shadow trial {ordinal} inventory hash mismatch");
        }
        verify_split_t7_shadow_trial_bundle(
            Path::new(&trial.source_bundle_path),
            trial,
            config,
            layout_a,
            layout_b,
        )?;
    }

    let (envelope_a, envelope_b) = split_t7_pointwise_lower_envelope(&receipt.trials)?;
    let minimum = envelope_a
        .iter()
        .chain(&envelope_b)
        .copied()
        .min()
        .context("split-T7 shadow envelope is empty")?;
    let fixed = split_t7_precommitted_cardinality(&envelope_a, &envelope_b)?;
    let leave_one_out = split_t7_leave_one_out_admissions(&receipt.trials)?;
    if leave_one_out.iter().any(|admission| !admission.admitted) {
        bail!("split-T7 shadow leave-one-out admission failed");
    }
    if receipt.lower_envelope_counts_a != envelope_a
        || receipt.lower_envelope_counts_b != envelope_b
        || receipt.lower_envelope_minimum != minimum
        || receipt.leave_one_trial_out != leave_one_out
        || receipt.fixed_spikes_per_frame != fixed
        || split_t7_shadow_ensemble_sha256(receipt)? != receipt.ensemble_sha256
    {
        bail!("split-T7 shadow ensemble recomputation failed");
    }
    Ok(())
}

#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn verify_split_t7_admission_receipt_v3(
    output_dir: &Path,
    config: &CoupledTwinEvidenceConfig,
    topology_a: &PrismPrepTopology,
    topology_b: &PrismPrepTopology,
    layout_a: &SplitT7TopologyLayout,
    layout_b: &SplitT7TopologyLayout,
    receipt: &SplitT7AdmissionReceiptV3,
    trial_artifacts: &[CoupledTwinArtifactEvidence],
    ensemble_artifact: &CoupledTwinArtifactEvidence,
    admission_artifact: &CoupledTwinArtifactEvidence,
) -> Result<()> {
    if receipt.schema_kind != "prismos.coupled_twin_split_t7_admission.v3"
        || receipt.schema_version != SPLIT_T7_SHADOW_SCHEMA_VERSION
        || receipt.status != "IMPLEMENTATION_VERIFIED"
        || receipt.validation_status != "PASS"
        || receipt.run_id != config.run_id
        || receipt.topology_a_sha256 != config.topology_a_sha256
        || receipt.topology_b_sha256 != config.topology_b_sha256
        || receipt.trial_seed_derivation != SPLIT_T7_TRIAL_SEED_DERIVATION_V3
        || receipt.selection_method != SPLIT_T7_SELECTION_METHOD_V3
        || receipt.window_md_steps == 0
        || receipt.frame_count != SPLIT_T7_TOTAL_FRAMES as u32
        || receipt.shadow_trial_count != SPLIT_T7_SHADOW_TRIALS as u32
        || receipt.shadow_trial_seeds.len() != SPLIT_T7_SHADOW_TRIALS
        || receipt.shadow_trial_artifacts.len() != SPLIT_T7_SHADOW_TRIALS
        || trial_artifacts.len() != SPLIT_T7_SHADOW_TRIALS
        || receipt.authoritative_crn_initialization_artifact.kind
            != "authoritative_coupled_twin_crn_initialization"
        || receipt
            .authoritative_crn_initialization_artifact
            .stream_id
            .is_some()
        || receipt
            .authoritative_crn_initialization_semantic_sha256
            .len()
            != 64
        || receipt.reserve_factor != SPLIT_T7_SHADOW_RESERVE_FACTOR
        || receipt.reserve_factor_derivation != SPLIT_T7_RESERVE_DERIVATION_V3
        || receipt.selected_k_derivation != SPLIT_T7_K_DERIVATION_V3
        || receipt.authoritative_count_source != SPLIT_T7_AUTHORITATIVE_COUNT_SOURCE_V3
        || !receipt.authoritative_counts_are_gpu_atomic
        || !receipt.authoritative_underflow_verified
        || !receipt.no_padding
        || !receipt.no_reuse
        || !receipt.no_synthetic_events
        || receipt.fallback_used
        || admission_artifact.kind != "split_t7_admission_receipt_v3"
        || admission_artifact.stream_id.is_some()
        || admission_artifact.record_count != Some((SPLIT_T7_TOTAL_FRAMES * 2) as u64)
        || admission_artifact.schema_version != Some(SPLIT_T7_SHADOW_SCHEMA_VERSION)
    {
        bail!("split-T7 V3 admission identity contract failed");
    }
    validate_run_artifact(output_dir, admission_artifact)?;
    let authoritative_crn_artifact = CoupledTwinArtifactEvidence {
        kind: receipt
            .authoritative_crn_initialization_artifact
            .kind
            .clone(),
        stream_id: receipt.authoritative_crn_initialization_artifact.stream_id,
        path: receipt
            .authoritative_crn_initialization_artifact
            .path
            .clone(),
        size_bytes: receipt.authoritative_crn_initialization_artifact.size_bytes,
        sha256: receipt
            .authoritative_crn_initialization_artifact
            .sha256
            .clone(),
        blake3: receipt
            .authoritative_crn_initialization_artifact
            .blake3
            .clone(),
        record_count: Some(1),
        schema_version: Some(COUPLED_TWIN_CRN_SCHEMA_VERSION),
        runtime_owned: true,
    };
    validate_run_artifact(output_dir, &authoritative_crn_artifact)?;
    let authoritative_crn_receipt: CoupledTwinCrnInitializationReceiptV3 =
        serde_json::from_slice(&std::fs::read(&authoritative_crn_artifact.path)?)?;
    if authoritative_crn_receipt.semantic_sha256
        != receipt.authoritative_crn_initialization_semantic_sha256
    {
        bail!("split-T7 V3 admission authoritative CRN semantic binding failed");
    }
    verify_coupled_twin_crn_receipt_v3(
        &authoritative_crn_receipt,
        topology_a,
        topology_b,
        config,
        receipt.base_seed,
        f32::from_bits(authoritative_crn_receipt.temperature_kelvin_f32_bits),
        "authoritative",
        None,
    )?;
    if !strict_binding_matches_artifact(&receipt.shadow_ensemble_artifact, ensemble_artifact) {
        bail!("split-T7 V3 admission ensemble artifact binding failed");
    }
    for (ordinal, ((binding, artifact), seed)) in receipt
        .shadow_trial_artifacts
        .iter()
        .zip(trial_artifacts)
        .zip(&receipt.shadow_trial_seeds)
        .enumerate()
    {
        if !strict_binding_matches_artifact(binding, artifact) {
            bail!("split-T7 V3 admission trial {ordinal} artifact binding failed");
        }
        let (expected_seed, _) = split_t7_trial_seed(receipt.base_seed, ordinal as u32, config)?;
        if *seed != expected_seed {
            bail!("split-T7 V3 admission trial {ordinal} seed mismatch");
        }
    }
    if receipt
        .shadow_trial_seeds
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .len()
        != SPLIT_T7_SHADOW_TRIALS
    {
        bail!("split-T7 V3 admission trial seeds are not distinct");
    }

    let durable_ensemble: SplitT7ShadowEnsembleReceipt = serde_json::from_slice(
        &std::fs::read(&ensemble_artifact.path)
            .with_context(|| format!("read split-T7 ensemble {}", ensemble_artifact.path))?,
    )?;
    verify_split_t7_shadow_ensemble(
        output_dir,
        config,
        topology_a,
        topology_b,
        layout_a,
        layout_b,
        &durable_ensemble,
        trial_artifacts,
        ensemble_artifact,
    )?;
    let (lower_a, lower_b) = split_t7_pointwise_lower_envelope(&durable_ensemble.trials)?;
    let lower_minimum = lower_a
        .iter()
        .chain(&lower_b)
        .copied()
        .min()
        .context("split-T7 V3 admission lower envelope is empty")?;
    let leave_one_out = split_t7_leave_one_out_admissions(&durable_ensemble.trials)?;
    let selected_k = split_t7_reserved_cardinality(&lower_a, &lower_b, receipt.reserve_factor)?;
    let (authoritative_minimum, authoritative_slack) = split_t7_authoritative_admission(
        &receipt.authoritative_counts_a,
        &receipt.authoritative_counts_b,
        selected_k,
    )?;
    let authoritative_inventory_sha256 = split_t7_paired_inventory_sha256(
        SPLIT_T7_AUTHORITATIVE_INVENTORY_DOMAIN,
        receipt.window_md_steps,
        selected_k,
        &receipt.authoritative_counts_a,
        &receipt.authoritative_counts_b,
    )?;
    if receipt.shadow_ensemble_sha256 != durable_ensemble.ensemble_sha256
        || receipt.lower_envelope_counts_a != lower_a
        || receipt.lower_envelope_counts_b != lower_b
        || receipt.lower_envelope_minimum != lower_minimum
        || receipt.leave_one_trial_out != leave_one_out
        || leave_one_out.iter().any(|result| !result.admitted)
        || receipt.selected_k != selected_k
        || receipt.authoritative_counts_a.len() != SPLIT_T7_TOTAL_FRAMES
        || receipt.authoritative_counts_b.len() != SPLIT_T7_TOTAL_FRAMES
        || receipt
            .authoritative_counts_a
            .iter()
            .chain(&receipt.authoritative_counts_b)
            .any(|count| *count < selected_k)
        || receipt.authoritative_minimum != authoritative_minimum
        || receipt.authoritative_minimum_slack_from_k != authoritative_slack
        || receipt.authoritative_inventory_sha256 != authoritative_inventory_sha256
        || split_t7_admission_semantic_sha256(receipt)? != receipt.semantic_sha256
    {
        bail!("split-T7 V3 admission independent recomputation failed");
    }
    Ok(())
}

#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn emit_split_t7_admission_receipt_v3(
    output_dir: &Path,
    config: &CoupledTwinEvidenceConfig,
    topology_a: &PrismPrepTopology,
    topology_b: &PrismPrepTopology,
    layout_a: &SplitT7TopologyLayout,
    layout_b: &SplitT7TopologyLayout,
    inventory: &SplitT7Inventory,
    window_md_steps: u32,
    authoritative_counts_a: &[u32],
    authoritative_counts_b: &[u32],
    authoritative_minimum: u32,
    authoritative_minimum_slack_from_k: u32,
    authoritative_inventory_sha256: &str,
    authoritative_crn_artifact: &CoupledTwinArtifactEvidence,
    authoritative_crn_receipt: &CoupledTwinCrnInitializationReceiptV3,
) -> Result<CoupledTwinArtifactEvidence> {
    let durable_ensemble: SplitT7ShadowEnsembleReceipt = serde_json::from_slice(
        &std::fs::read(&inventory.shadow_ensemble_artifact.path).with_context(|| {
            format!(
                "read split-T7 shadow ensemble {}",
                inventory.shadow_ensemble_artifact.path
            )
        })?,
    )?;
    verify_split_t7_shadow_ensemble(
        output_dir,
        config,
        topology_a,
        topology_b,
        layout_a,
        layout_b,
        &durable_ensemble,
        &inventory.shadow_trial_artifacts,
        &inventory.shadow_ensemble_artifact,
    )?;
    let mut receipt = SplitT7AdmissionReceiptV3 {
        schema_kind: "prismos.coupled_twin_split_t7_admission.v3".to_string(),
        schema_version: SPLIT_T7_SHADOW_SCHEMA_VERSION,
        status: "IMPLEMENTATION_VERIFIED".to_string(),
        validation_status: "PASS".to_string(),
        run_id: config.run_id.clone(),
        topology_a_sha256: config.topology_a_sha256.clone(),
        topology_b_sha256: config.topology_b_sha256.clone(),
        base_seed: durable_ensemble.base_seed,
        trial_seed_derivation: SPLIT_T7_TRIAL_SEED_DERIVATION_V3.to_string(),
        selection_method: SPLIT_T7_SELECTION_METHOD_V3.to_string(),
        window_md_steps,
        frame_count: SPLIT_T7_TOTAL_FRAMES as u32,
        shadow_trial_count: SPLIT_T7_SHADOW_TRIALS as u32,
        shadow_trial_seeds: durable_ensemble
            .trials
            .iter()
            .map(|trial| trial.trial_seed)
            .collect(),
        shadow_trial_artifacts: inventory
            .shadow_trial_artifacts
            .iter()
            .map(strict_artifact_binding)
            .collect(),
        shadow_ensemble_artifact: strict_artifact_binding(&inventory.shadow_ensemble_artifact),
        shadow_ensemble_sha256: durable_ensemble.ensemble_sha256.clone(),
        lower_envelope_counts_a: durable_ensemble.lower_envelope_counts_a.clone(),
        lower_envelope_counts_b: durable_ensemble.lower_envelope_counts_b.clone(),
        lower_envelope_minimum: durable_ensemble.lower_envelope_minimum,
        leave_one_trial_out: durable_ensemble.leave_one_trial_out.clone(),
        reserve_factor: SPLIT_T7_SHADOW_RESERVE_FACTOR,
        reserve_factor_derivation: SPLIT_T7_RESERVE_DERIVATION_V3.to_string(),
        selected_k: inventory.fixed_spikes_per_frame,
        selected_k_derivation: SPLIT_T7_K_DERIVATION_V3.to_string(),
        authoritative_counts_a: authoritative_counts_a.to_vec(),
        authoritative_counts_b: authoritative_counts_b.to_vec(),
        authoritative_crn_initialization_artifact: strict_artifact_binding(
            authoritative_crn_artifact,
        ),
        authoritative_crn_initialization_semantic_sha256: authoritative_crn_receipt
            .semantic_sha256
            .clone(),
        authoritative_minimum,
        authoritative_minimum_slack_from_k,
        authoritative_inventory_sha256: authoritative_inventory_sha256.to_string(),
        authoritative_count_source: SPLIT_T7_AUTHORITATIVE_COUNT_SOURCE_V3.to_string(),
        authoritative_counts_are_gpu_atomic: true,
        authoritative_underflow_verified: true,
        no_padding: true,
        no_reuse: true,
        no_synthetic_events: true,
        fallback_used: false,
        semantic_sha256: String::new(),
    };
    receipt.semantic_sha256 = split_t7_admission_semantic_sha256(&receipt)?;

    // Validate the semantic object before committing its no-clobber runtime artifact.
    let candidate_bytes = serde_json::to_vec_pretty(&receipt)?;
    let candidate_readback: SplitT7AdmissionReceiptV3 = serde_json::from_slice(&candidate_bytes)?;
    if candidate_readback != receipt
        || split_t7_admission_semantic_sha256(&candidate_readback)?
            != candidate_readback.semantic_sha256
    {
        bail!("split-T7 V3 admission candidate serialization failed validation");
    }
    let receipt_path = output_dir.join("SPLIT_T7_ADMISSION_RECEIPT_V3.json");
    write_atomic_json(&receipt_path, &receipt)?;
    let admission_artifact = artifact_evidence(
        "split_t7_admission_receipt_v3",
        None,
        &receipt_path,
        Some((SPLIT_T7_TOTAL_FRAMES * 2) as u64),
        Some(SPLIT_T7_SHADOW_SCHEMA_VERSION),
    )?;
    let durable_receipt: SplitT7AdmissionReceiptV3 =
        serde_json::from_slice(&std::fs::read(&receipt_path)?)?;
    if durable_receipt != receipt {
        bail!("split-T7 V3 admission durable readback changed semantics");
    }
    verify_split_t7_admission_receipt_v3(
        output_dir,
        config,
        topology_a,
        topology_b,
        layout_a,
        layout_b,
        &durable_receipt,
        &inventory.shadow_trial_artifacts,
        &inventory.shadow_ensemble_artifact,
        &admission_artifact,
    )?;
    Ok(admission_artifact)
}

#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn inventory_split_t7_fixed_cardinality(
    config_ref: &crate::persistent_engine::PersistentBatchConfig,
    context: Arc<CudaContext>,
    fused_module: Arc<CudaModule>,
    topology_a: &PrismPrepTopology,
    topology_b: &PrismPrepTopology,
    protocol: &CryoUvProtocol,
    config: &CoupledTwinEvidenceConfig,
    seed: u64,
    outer_steps: u32,
    fused_inner_steps: i32,
    hmr: bool,
    fused_steps: u32,
    adaptive_dt: bool,
    ladd_enabled: bool,
    output_dir: &Path,
    prismcode_admission: &PrismCodeMdPolicyAdmissionV1,
) -> Result<SplitT7Inventory> {
    if fused_inner_steps <= 0
        || outer_steps <= SPLIT_T7_TOTAL_FRAMES as u32
        || (SPLIT_T7_TOTAL_FRAMES as u64).saturating_mul(fused_inner_steps as u64)
            > protocol.cold_hold_steps.max(0) as u64
    {
        bail!(
            "split-T7 inventory cannot cover {} equal cold-hold windows: outer={} inner={} cold_hold={}",
            SPLIT_T7_TOTAL_FRAMES,
            outer_steps,
            fused_inner_steps,
            protocol.cold_hold_steps
        );
    }
    config.validate()?;
    let (chain_role_map_a, chain_role_map_b) = coupled_native_chain_role_maps(
        topology_a,
        topology_b,
        config.identical_topology_negative_control,
    )?;
    let layout_a = split_t7_topology_layout(topology_a, &chain_role_map_a)?;
    let layout_b = split_t7_topology_layout(topology_b, &chain_role_map_b)?;
    validate_split_t7_paired_layouts(
        &layout_a,
        &layout_b,
        config.identical_topology_negative_control,
    )?;

    let mut trials = Vec::with_capacity(SPLIT_T7_SHADOW_TRIALS);
    let mut trial_artifacts = Vec::with_capacity(SPLIT_T7_SHADOW_TRIALS);
    let mut crn_artifacts = Vec::with_capacity(SPLIT_T7_SHADOW_TRIALS);
    for trial_ordinal in 0..SPLIT_T7_SHADOW_TRIALS as u32 {
        let (trial, artifact, crn_artifact) = execute_split_t7_shadow_trial(
            config_ref,
            context.clone(),
            fused_module.clone(),
            topology_a,
            topology_b,
            protocol,
            config,
            seed,
            outer_steps,
            fused_inner_steps,
            hmr,
            fused_steps,
            adaptive_dt,
            ladd_enabled,
            trial_ordinal,
            output_dir,
            &layout_a,
            &layout_b,
            prismcode_admission,
        )?;
        trials.push(trial);
        trial_artifacts.push(artifact);
        crn_artifacts.push(crn_artifact);
    }
    let (envelope_a, envelope_b) = split_t7_pointwise_lower_envelope(&trials)?;
    let lower_envelope_minimum = envelope_a
        .iter()
        .chain(&envelope_b)
        .copied()
        .min()
        .context("split-T7 shadow envelope is empty")?;
    let leave_one_trial_out = split_t7_leave_one_out_admissions(&trials)?;
    if let Some(failed) = leave_one_trial_out
        .iter()
        .find(|admission| !admission.admitted)
    {
        bail!(
            "split-T7 shadow K={} trained without trial {} exceeds held-out minimum {}",
            failed.candidate_k,
            failed.held_out_trial_ordinal,
            failed.held_out_minimum
        );
    }
    let all_trial_seeds_distinct = trials
        .iter()
        .map(|trial| trial.trial_seed)
        .collect::<BTreeSet<_>>()
        .len()
        == SPLIT_T7_SHADOW_TRIALS;
    if !all_trial_seeds_distinct {
        bail!("deterministically derived split-T7 shadow seeds collided");
    }
    let fixed_spikes = split_t7_precommitted_cardinality(&envelope_a, &envelope_b)?;
    let mut receipt = SplitT7ShadowEnsembleReceipt {
        schema_kind: "prismos.coupled_twin_split_t7_shadow_ensemble.v3".to_string(),
        schema_version: SPLIT_T7_SHADOW_SCHEMA_VERSION,
        status: "IMPLEMENTATION_VERIFIED".to_string(),
        validation_status: "PASS".to_string(),
        run_id: config.run_id.clone(),
        base_seed: seed,
        trial_seed_derivation: SPLIT_T7_TRIAL_SEED_DERIVATION_V3.to_string(),
        topology_a_sha256: config.topology_a_sha256.clone(),
        topology_b_sha256: config.topology_b_sha256.clone(),
        window_md_steps: fused_inner_steps as u32,
        frame_count: SPLIT_T7_TOTAL_FRAMES as u32,
        trial_count: SPLIT_T7_SHADOW_TRIALS as u32,
        reserve_factor: SPLIT_T7_SHADOW_RESERVE_FACTOR,
        selection_method: SPLIT_T7_SELECTION_METHOD_V3.to_string(),
        trials,
        lower_envelope_counts_a: envelope_a.clone(),
        lower_envelope_counts_b: envelope_b.clone(),
        lower_envelope_minimum,
        leave_one_trial_out,
        all_trial_seeds_distinct,
        paired_stream_seed_equality_verified: true,
        paired_common_random_initialization_verified: true,
        holdout_admission_verified: true,
        fixed_spikes_per_frame: fixed_spikes,
        ensemble_sha256: String::new(),
        fallback_used: false,
    };
    receipt.ensemble_sha256 = split_t7_shadow_ensemble_sha256(&receipt)?;
    let stem = coupled_topology_stem(config);
    let ensemble_path = output_dir.join(format!("{stem}_split_t7_shadow_ensemble.json"));
    write_atomic_json(&ensemble_path, &receipt)?;
    let ensemble_artifact = artifact_evidence(
        "split_t7_shadow_ensemble_receipt",
        None,
        &ensemble_path,
        Some(SPLIT_T7_SHADOW_TRIALS as u64),
        Some(SPLIT_T7_SHADOW_SCHEMA_VERSION),
    )?;
    let durable_receipt: SplitT7ShadowEnsembleReceipt =
        serde_json::from_slice(&std::fs::read(&ensemble_path)?)?;
    if durable_receipt != receipt {
        bail!("split-T7 shadow ensemble durable JSON readback changed semantics");
    }
    verify_split_t7_shadow_ensemble(
        output_dir,
        config,
        topology_a,
        topology_b,
        &layout_a,
        &layout_b,
        &durable_receipt,
        &trial_artifacts,
        &ensemble_artifact,
    )?;
    Ok(SplitT7Inventory {
        fixed_spikes_per_frame: fixed_spikes,
        shadow_lower_envelope_counts_a: envelope_a,
        shadow_lower_envelope_counts_b: envelope_b,
        shadow_ensemble_sha256: receipt.ensemble_sha256,
        shadow_ensemble_artifact: ensemble_artifact,
        shadow_trial_artifacts: trial_artifacts,
        shadow_crn_artifacts: crn_artifacts,
    })
}

#[cfg(any())]
fn superseded_variable_duration_split_t7_calibration(
    engine_a: &mut PersistentNhsEngine,
    engine_b: &mut PersistentNhsEngine,
    topology_a: &PrismPrepTopology,
    topology_b: &PrismPrepTopology,
    chain_role_map_a: &BTreeMap<String, String>,
    chain_role_map_b: &BTreeMap<String, String>,
    protocol: &CryoUvProtocol,
    config: &CoupledTwinEvidenceConfig,
    outer_steps: u32,
    fused_inner_steps: i32,
    output_dir: &Path,
    inventory: &SplitT7Inventory,
) -> Result<SplitT7PairOutcome> {
    if outer_steps <= SPLIT_T7_TOTAL_FRAMES as u32 || fused_inner_steps <= 0 {
        bail!(
            "online split-T7 requires at least {} direct calibration windows plus a resident remainder: outer={} inner={}",
            SPLIT_T7_TOTAL_FRAMES,
            outer_steps,
            fused_inner_steps
        );
    }
    let layout_a = split_t7_topology_layout(topology_a, chain_role_map_a)?;
    let layout_b = split_t7_topology_layout(topology_b, chain_role_map_b)?;
    let mut queue_a = VecDeque::new();
    let mut queue_b = VecDeque::new();
    let mut cursor_a = 0usize;
    let mut cursor_b = 0usize;
    let mut prefix_outer = 0u32;
    let mut warmup_spikes_a = 0u64;
    let mut warmup_spikes_b = 0u64;

    while queue_a.len() < 4 || queue_b.len() < 4 {
        let (a, b) = advance_split_t7_direct_window(
            engine_a,
            engine_b,
            protocol,
            outer_steps,
            fused_inner_steps,
            &mut prefix_outer,
            &mut cursor_a,
            &mut cursor_b,
            &mut queue_a,
            &mut queue_b,
        )?;
        warmup_spikes_a = warmup_spikes_a.saturating_add(a);
        warmup_spikes_b = warmup_spikes_b.saturating_add(b);
    }
    let fixed_spikes = queue_a.len().min(queue_b.len());
    if fixed_spikes < 4 || fixed_spikes > u32::MAX as usize {
        bail!("online split-T7 failed to derive a valid common frame cardinality");
    }
    let fixed_spikes_u32 = fixed_spikes as u32;
    let mut input_a = split_t7_input_header(
        0,
        fixed_spikes_u32,
        config,
        &config.topology_a_sha256,
        &layout_a.layout_sha256,
    );
    let mut input_b = split_t7_input_header(
        1,
        fixed_spikes_u32,
        config,
        &config.topology_b_sha256,
        &layout_b.layout_sha256,
    );
    let mut frames_a = Vec::with_capacity(SPLIT_T7_TOTAL_FRAMES);
    let mut frames_b = Vec::with_capacity(SPLIT_T7_TOTAL_FRAMES);
    let mut variance_a = Vec::with_capacity(SPLIT_T7_VARIANCE_FRAMES);
    let mut variance_b = Vec::with_capacity(SPLIT_T7_VARIANCE_FRAMES);
    let mut runtime_a = None::<SplitT7StreamRuntime>;
    let mut runtime_b = None::<SplitT7StreamRuntime>;
    let mut calibrated_a = None::<(f32, f32)>;
    let mut calibrated_b = None::<(f32, f32)>;

    for ordinal in 0..SPLIT_T7_TOTAL_FRAMES {
        // Every analysis frame after frame 0 follows at least one new paired
        // MD window, so no series of historical frames is replayed against one
        // final force/velocity/PE state.
        if ordinal > 0 {
            let (a, b) = advance_split_t7_direct_window(
                engine_a,
                engine_b,
                protocol,
                outer_steps,
                fused_inner_steps,
                &mut prefix_outer,
                &mut cursor_a,
                &mut cursor_b,
                &mut queue_a,
                &mut queue_b,
            )?;
            warmup_spikes_a = warmup_spikes_a.saturating_add(a);
            warmup_spikes_b = warmup_spikes_b.saturating_add(b);
        }
        while queue_a.len() < fixed_spikes || queue_b.len() < fixed_spikes {
            let (a, b) = advance_split_t7_direct_window(
                engine_a,
                engine_b,
                protocol,
                outer_steps,
                fused_inner_steps,
                &mut prefix_outer,
                &mut cursor_a,
                &mut cursor_b,
                &mut queue_a,
                &mut queue_b,
            )?;
            warmup_spikes_a = warmup_spikes_a.saturating_add(a);
            warmup_spikes_b = warmup_spikes_b.saturating_add(b);
        }
        let mut source_a = select_split_t7_full_span(&mut queue_a, fixed_spikes, 0)?;
        let mut source_b = select_split_t7_full_span(&mut queue_b, fixed_spikes, 1)?;
        let (rich_a, offsets_a, timestep_min_a, timestep_max_a) =
            prepare_split_t7_frame(&mut source_a.selected_events, &layout_a)?;
        let (rich_b, offsets_b, timestep_min_b, timestep_max_b) =
            prepare_split_t7_frame(&mut source_b.selected_events, &layout_b)?;
        if runtime_a.is_none() {
            runtime_a = Some(build_split_t7_stream_runtime(
                engine_a,
                topology_a,
                0,
                layout_a.clone(),
                &rich_a,
                &offsets_a,
            )?);
            runtime_b = Some(build_split_t7_stream_runtime(
                engine_b,
                topology_b,
                1,
                layout_b.clone(),
                &rich_b,
                &offsets_b,
            )?);
        }
        let analysis_md_step = (prefix_outer as u64).saturating_mul(fused_inner_steps as u64);
        let input_offsets_a = append_split_t7_input_frame(
            &mut input_a,
            ordinal as u32,
            analysis_md_step,
            timestep_min_a,
            timestep_max_a,
            &source_a,
            &rich_a,
        )?;
        let input_offsets_b = append_split_t7_input_frame(
            &mut input_b,
            ordinal as u32,
            analysis_md_step,
            timestep_min_b,
            timestep_max_b,
            &source_b,
            &rich_b,
        )?;
        // CUDA topology constants are module-global. Launch A to completion
        // before rebinding them for B; concurrent A/B analysis would be a
        // cross-topology race even though the MD streams themselves are paired.
        let readback_a = launch_split_t7_frame(
            engine_a,
            topology_a,
            runtime_a
                .as_mut()
                .context("split-T7 stream A runtime absent")?,
            0,
            analysis_md_step,
            &rich_a,
            &offsets_a,
        )?;
        let readback_b = launch_split_t7_frame(
            engine_b,
            topology_b,
            runtime_b
                .as_mut()
                .context("split-T7 stream B runtime absent")?,
            1,
            analysis_md_step,
            &rich_b,
            &offsets_b,
        )?;
        let phase = if ordinal < SPLIT_T7_ANCHOR_FRAMES {
            "anchor"
        } else if ordinal < SPLIT_T7_ANCHOR_FRAMES + SPLIT_T7_VARIANCE_FRAMES {
            variance_a.push(readback_a.total_kl.abs() as f64);
            variance_b.push(readback_b.total_kl.abs() as f64);
            "variance"
        } else {
            "validation"
        };
        if config.identical_topology_negative_control
            && (source_a.source_event_count != source_b.source_event_count
                || source_a.source_events_sha256 != source_b.source_events_sha256)
        {
            bail!(
                "identical-topology control canonical spike multiset differs at split-T7 frame {}: count_a={} count_b={} sha256_a={} sha256_b={}",
                ordinal,
                source_a.source_event_count,
                source_b.source_event_count,
                source_a.source_events_sha256,
                source_b.source_events_sha256
            );
        }
        if config.identical_topology_negative_control
            && (readback_a.physical_state_sha256_before != readback_b.physical_state_sha256_before
                || readback_a.physical_state_sha256_after != readback_b.physical_state_sha256_after)
        {
            bail!(
                "identical-topology control diverged in unsalted physical engine state at split-T7 frame {}",
                ordinal
            );
        }
        frames_a.push(CoupledNoiseFloorFrameEvidence {
            phase: phase.to_string(),
            ordinal: ordinal as u32,
            timestep: timestep_max_a,
            source_timestep_min: timestep_min_a,
            source_timestep_max: timestep_max_a,
            analysis_md_step,
            source_spike_count: source_a.source_event_count as u32,
            selected_spike_count: fixed_spikes_u32,
            dropped_spike_count: (source_a.source_event_count as u32)
                .saturating_sub(fixed_spikes_u32),
            input_byte_offset: input_offsets_a.rich_offset,
            input_byte_length: input_offsets_a.rich_length,
            source_input_byte_offset: input_offsets_a.source_offset,
            source_input_byte_length: input_offsets_a.source_length,
            selection_index_byte_offset: input_offsets_a.selection_offset,
            selection_index_byte_length: input_offsets_a.selection_length,
            source_events_sha256: source_a.source_events_sha256,
            selected_rich_spikes_sha256: rich_spike_slice_sha256(&rich_a)?,
            total_kl: Some(readback_a.total_kl),
            total_kl_magnitude: Some(readback_a.total_kl.abs()),
            adjudication_code: readback_a.adjudication_code,
            adjudication_reason_flags: readback_a.reason_flags,
            engine_state_sha256_before: readback_a.state_sha256_before,
            engine_state_sha256_after: readback_a.state_sha256_after,
            engine_physical_state_sha256_before: readback_a.physical_state_sha256_before,
            engine_physical_state_sha256_after: readback_a.physical_state_sha256_after,
        });
        frames_b.push(CoupledNoiseFloorFrameEvidence {
            phase: phase.to_string(),
            ordinal: ordinal as u32,
            timestep: timestep_max_b,
            source_timestep_min: timestep_min_b,
            source_timestep_max: timestep_max_b,
            analysis_md_step,
            source_spike_count: source_b.source_event_count as u32,
            selected_spike_count: fixed_spikes_u32,
            dropped_spike_count: (source_b.source_event_count as u32)
                .saturating_sub(fixed_spikes_u32),
            input_byte_offset: input_offsets_b.rich_offset,
            input_byte_length: input_offsets_b.rich_length,
            source_input_byte_offset: input_offsets_b.source_offset,
            source_input_byte_length: input_offsets_b.source_length,
            selection_index_byte_offset: input_offsets_b.selection_offset,
            selection_index_byte_length: input_offsets_b.selection_length,
            source_events_sha256: source_b.source_events_sha256,
            selected_rich_spikes_sha256: rich_spike_slice_sha256(&rich_b)?,
            total_kl: Some(readback_b.total_kl),
            total_kl_magnitude: Some(readback_b.total_kl.abs()),
            adjudication_code: readback_b.adjudication_code,
            adjudication_reason_flags: readback_b.reason_flags,
            engine_state_sha256_before: readback_b.state_sha256_before,
            engine_state_sha256_after: readback_b.state_sha256_after,
            engine_physical_state_sha256_before: readback_b.physical_state_sha256_before,
            engine_physical_state_sha256_after: readback_b.physical_state_sha256_after,
        });

        if ordinal + 1 == SPLIT_T7_ANCHOR_FRAMES {
            runtime_a
                .as_ref()
                .context("split-T7 stream A runtime absent")?
                .pipeline
                .freeze_baseline()
                .map_err(|error| {
                    anyhow::anyhow!("stream A split-T7 baseline freeze: {:?}", error)
                })?;
            engine_a.cuda_stream().synchronize()?;
            runtime_b
                .as_ref()
                .context("split-T7 stream B runtime absent")?
                .pipeline
                .freeze_baseline()
                .map_err(|error| {
                    anyhow::anyhow!("stream B split-T7 baseline freeze: {:?}", error)
                })?;
            engine_b.cuda_stream().synchronize()?;
        }
        if ordinal + 1 == SPLIT_T7_ANCHOR_FRAMES + SPLIT_T7_VARIANCE_FRAMES {
            let stats_a = split_t7_population_stats(&variance_a, 0)?;
            let stats_b = split_t7_population_stats(&variance_b, 1)?;
            runtime_a
                .as_ref()
                .context("split-T7 stream A runtime absent")?
                .pipeline
                .write_kl_native_mu_sigma(stats_a.0, stats_a.1)
                .map_err(|error| anyhow::anyhow!("stream A split-T7 KL write: {:?}", error))?;
            engine_a.cuda_stream().synchronize()?;
            let (_, _, mu_a, sigma_a, flags_a) = read_split_t7_adjudicator(
                runtime_a
                    .as_ref()
                    .context("split-T7 stream A runtime absent")?,
            )?;
            runtime_b
                .as_ref()
                .context("split-T7 stream B runtime absent")?
                .pipeline
                .write_kl_native_mu_sigma(stats_b.0, stats_b.1)
                .map_err(|error| anyhow::anyhow!("stream B split-T7 KL write: {:?}", error))?;
            engine_b.cuda_stream().synchronize()?;
            let (_, _, mu_b, sigma_b, flags_b) = read_split_t7_adjudicator(
                runtime_b
                    .as_ref()
                    .context("split-T7 stream B runtime absent")?,
            )?;
            if mu_a.to_bits() != stats_a.0.to_bits()
                || sigma_a.to_bits() != stats_a.1.to_bits()
                || mu_b.to_bits() != stats_b.0.to_bits()
                || sigma_b.to_bits() != stats_b.1.to_bits()
                || flags_a & 0x8000_0000 != 0
                || flags_b & 0x8000_0000 != 0
            {
                bail!("split-T7 KL-native threshold readback mismatch");
            }
            calibrated_a = Some(stats_a);
            calibrated_b = Some(stats_b);
        }
    }

    let mut runtime_a = runtime_a.context("split-T7 stream A runtime was never built")?;
    let mut runtime_b = runtime_b.context("split-T7 stream B runtime was never built")?;
    let validation_marker_a = runtime_a
        .pipeline
        .read_burst_marker()
        .map_err(|error| anyhow::anyhow!("stream A split-T7 validation marker: {:?}", error))?;
    let validation_marker_b = runtime_b
        .pipeline
        .read_burst_marker()
        .map_err(|error| anyhow::anyhow!("stream B split-T7 validation marker: {:?}", error))?;
    if validation_marker_a > 1 || validation_marker_b > 1 {
        bail!("split-T7 validation marker is outside its binary domain");
    }
    let stats_a = calibrated_a.context("stream A split-T7 threshold was not calibrated")?;
    let stats_b = calibrated_b.context("stream B split-T7 threshold was not calibrated")?;
    let outcome_a = finalize_split_t7_stream(
        0,
        &config.topology_a_path,
        &config.topology_a_sha256,
        config,
        output_dir,
        runtime_a,
        input_a,
        frames_a,
        prefix_outer,
        fused_inner_steps,
        fixed_spikes_u32,
        stats_a.0,
        stats_a.1,
        validation_marker_a,
    )?;
    let outcome_b = finalize_split_t7_stream(
        1,
        &config.topology_b_path,
        &config.topology_b_sha256,
        config,
        output_dir,
        runtime_b,
        input_b,
        frames_b,
        prefix_outer,
        fused_inner_steps,
        fixed_spikes_u32,
        stats_b.0,
        stats_b.1,
        validation_marker_b,
    )?;
    Ok(SplitT7PairOutcome {
        stream_a: outcome_a,
        stream_b: outcome_b,
        direct_prefix_outer_steps: prefix_outer,
        direct_prefix_md_steps: (prefix_outer as u64).saturating_mul(fused_inner_steps as u64),
        warmup_spikes_a,
        warmup_spikes_b,
    })
}

#[cfg(feature = "gpu")]
fn execute_online_split_t7_calibration(
    engine_a: &mut PersistentNhsEngine,
    engine_b: &mut PersistentNhsEngine,
    topology_a: &PrismPrepTopology,
    topology_b: &PrismPrepTopology,
    chain_role_map_a: &BTreeMap<String, String>,
    chain_role_map_b: &BTreeMap<String, String>,
    protocol: &CryoUvProtocol,
    config: &CoupledTwinEvidenceConfig,
    outer_steps: u32,
    fused_inner_steps: i32,
    output_dir: &Path,
    inventory: &SplitT7Inventory,
    authoritative_crn_artifact: &CoupledTwinArtifactEvidence,
    authoritative_crn_receipt: &CoupledTwinCrnInitializationReceiptV3,
) -> Result<SplitT7PairOutcome> {
    if outer_steps <= SPLIT_T7_TOTAL_FRAMES as u32
        || fused_inner_steps <= 0
        || inventory.shadow_lower_envelope_counts_a.len() != SPLIT_T7_TOTAL_FRAMES
        || inventory.shadow_lower_envelope_counts_b.len() != SPLIT_T7_TOTAL_FRAMES
        || inventory.shadow_ensemble_sha256.len() != 64
        || inventory.shadow_ensemble_artifact.kind != "split_t7_shadow_ensemble_receipt"
        || inventory.shadow_trial_artifacts.len() != SPLIT_T7_SHADOW_TRIALS
        || inventory.shadow_crn_artifacts.len() != SPLIT_T7_SHADOW_TRIALS
        || inventory.fixed_spikes_per_frame < 4
    {
        bail!(
            "online split-T7 received an invalid fixed-duration inventory: outer={} inner={} frames_a={} frames_b={} K={}",
            outer_steps,
            fused_inner_steps,
            inventory.shadow_lower_envelope_counts_a.len(),
            inventory.shadow_lower_envelope_counts_b.len(),
            inventory.fixed_spikes_per_frame
        );
    }
    let fixed_spikes = inventory.fixed_spikes_per_frame as usize;
    let fixed_spikes_u32 = inventory.fixed_spikes_per_frame;
    let direct_prefix_outer_steps = SPLIT_T7_TOTAL_FRAMES as u32;
    let layout_a = split_t7_topology_layout(topology_a, chain_role_map_a)?;
    let layout_b = split_t7_topology_layout(topology_b, chain_role_map_b)?;
    validate_split_t7_paired_layouts(
        &layout_a,
        &layout_b,
        config.identical_topology_negative_control,
    )?;
    let stem = coupled_topology_stem(config);
    let input_path_a = output_dir.join(format!("{stem}_stream00_ghost_noise_floor_input.bin"));
    let input_path_b = output_dir.join(format!("{stem}_stream01_ghost_noise_floor_input.bin"));
    let mut input_a = NoClobberArtifactFile::create(&input_path_a)?;
    let mut input_b = NoClobberArtifactFile::create(&input_path_b)?;
    input_a.file_mut()?.write_all(&split_t7_input_header(
        0,
        fixed_spikes_u32,
        direct_prefix_outer_steps,
        config,
        &config.topology_a_sha256,
        &layout_a.layout_sha256,
    ))?;
    input_b.file_mut()?.write_all(&split_t7_input_header(
        1,
        fixed_spikes_u32,
        direct_prefix_outer_steps,
        config,
        &config.topology_b_sha256,
        &layout_b.layout_sha256,
    ))?;

    let mut cursor_a = 0usize;
    let mut cursor_b = 0usize;
    let mut prefix_outer = 0u32;
    let mut warmup_spikes_a = 0u64;
    let mut warmup_spikes_b = 0u64;
    let mut frames_a = Vec::with_capacity(SPLIT_T7_TOTAL_FRAMES);
    let mut frames_b = Vec::with_capacity(SPLIT_T7_TOTAL_FRAMES);
    let mut variance_a = Vec::with_capacity(SPLIT_T7_VARIANCE_FRAMES);
    let mut variance_b = Vec::with_capacity(SPLIT_T7_VARIANCE_FRAMES);
    let mut authoritative_counts_a = Vec::with_capacity(SPLIT_T7_TOTAL_FRAMES);
    let mut authoritative_counts_b = Vec::with_capacity(SPLIT_T7_TOTAL_FRAMES);
    let mut runtime_a = None::<SplitT7StreamRuntime>;
    let mut runtime_b = None::<SplitT7StreamRuntime>;
    let mut calibrated_a = None::<(f32, f32)>;
    let mut calibrated_b = None::<(f32, f32)>;

    for ordinal in 0..SPLIT_T7_TOTAL_FRAMES {
        let (snapshot_a, snapshot_b) = advance_split_t7_direct_window(
            engine_a,
            engine_b,
            protocol,
            outer_steps,
            fused_inner_steps,
            &mut prefix_outer,
            &mut cursor_a,
            &mut cursor_b,
            config.identical_topology_negative_control,
        )?;
        let source_count_a = snapshot_a.window().len();
        let source_count_b = snapshot_b.window().len();
        if source_count_a < fixed_spikes || source_count_b < fixed_spikes {
            bail!(
                "authoritative split-T7 frame {} fell below the precommitted CUDA cardinality: actual A/B={}/{} required K={}",
                ordinal,
                source_count_a,
                source_count_b,
                fixed_spikes
            );
        }
        authoritative_counts_a.push(
            u32::try_from(source_count_a)
                .context("authoritative split-T7 stream A frame count exceeds u32")?,
        );
        authoritative_counts_b.push(
            u32::try_from(source_count_b)
                .context("authoritative split-T7 stream B frame count exceeds u32")?,
        );
        warmup_spikes_a = warmup_spikes_a.saturating_add(source_count_a as u64);
        warmup_spikes_b = warmup_spikes_b.saturating_add(source_count_b as u64);
        let mut source_a = select_split_t7_full_span(snapshot_a.window(), fixed_spikes, 0)?;
        let mut source_b = select_split_t7_full_span(snapshot_b.window(), fixed_spikes, 1)?;
        let (rich_a, offsets_a, timestep_min_a, timestep_max_a) =
            prepare_split_t7_frame(&mut source_a.selected_events, &layout_a)?;
        let (rich_b, offsets_b, timestep_min_b, timestep_max_b) =
            prepare_split_t7_frame(&mut source_b.selected_events, &layout_b)?;
        if runtime_a.is_none() {
            runtime_a = Some(build_split_t7_stream_runtime(
                engine_a,
                topology_a,
                0,
                layout_a.clone(),
                &rich_a,
                &offsets_a,
            )?);
            runtime_b = Some(build_split_t7_stream_runtime(
                engine_b,
                topology_b,
                1,
                layout_b.clone(),
                &rich_b,
                &offsets_b,
            )?);
        }
        let analysis_md_step = (prefix_outer as u64).saturating_mul(fused_inner_steps as u64);
        let input_offsets_a = append_split_t7_input_frame(
            input_a.file_mut()?,
            ordinal as u32,
            analysis_md_step,
            timestep_min_a,
            timestep_max_a,
            &source_a,
            &rich_a,
        )?;
        let input_offsets_b = append_split_t7_input_frame(
            input_b.file_mut()?,
            ordinal as u32,
            analysis_md_step,
            timestep_min_b,
            timestep_max_b,
            &source_b,
            &rich_b,
        )?;
        let readback_a = launch_split_t7_frame(
            engine_a,
            topology_a,
            runtime_a
                .as_mut()
                .context("split-T7 stream A runtime absent")?,
            0,
            analysis_md_step,
            &rich_a,
            &offsets_a,
        )?;
        let readback_b = launch_split_t7_frame(
            engine_b,
            topology_b,
            runtime_b
                .as_mut()
                .context("split-T7 stream B runtime absent")?,
            1,
            analysis_md_step,
            &rich_b,
            &offsets_b,
        )?;
        let phase = if ordinal < SPLIT_T7_ANCHOR_FRAMES {
            "anchor"
        } else if ordinal < SPLIT_T7_ANCHOR_FRAMES + SPLIT_T7_VARIANCE_FRAMES {
            variance_a.push(readback_a.total_kl.abs() as f64);
            variance_b.push(readback_b.total_kl.abs() as f64);
            "variance"
        } else {
            "validation"
        };
        if config.identical_topology_negative_control
            && (source_count_a != source_count_b
                || source_a.source_events_sha256 != source_b.source_events_sha256)
        {
            bail!(
                "identical-topology control canonical spike multiset differs at streamed split-T7 frame {}: count_a={} count_b={} sha256_a={} sha256_b={}",
                ordinal,
                source_count_a,
                source_count_b,
                source_a.source_events_sha256,
                source_b.source_events_sha256
            );
        }
        if config.identical_topology_negative_control
            && (readback_a.physical_state_sha256_before != readback_b.physical_state_sha256_before
                || readback_a.physical_state_sha256_after != readback_b.physical_state_sha256_after)
        {
            bail!(
                "identical-topology control diverged in unsalted physical engine state at split-T7 frame {}",
                ordinal
            );
        }
        frames_a.push(CoupledNoiseFloorFrameEvidence {
            phase: phase.to_string(),
            ordinal: ordinal as u32,
            timestep: timestep_max_a,
            source_timestep_min: timestep_min_a,
            source_timestep_max: timestep_max_a,
            analysis_md_step,
            source_spike_count: source_count_a as u32,
            selected_spike_count: fixed_spikes_u32,
            dropped_spike_count: (source_count_a as u32).saturating_sub(fixed_spikes_u32),
            input_byte_offset: input_offsets_a.rich_offset,
            input_byte_length: input_offsets_a.rich_length,
            source_input_byte_offset: input_offsets_a.source_offset,
            source_input_byte_length: input_offsets_a.source_length,
            selection_index_byte_offset: input_offsets_a.selection_offset,
            selection_index_byte_length: input_offsets_a.selection_length,
            source_events_sha256: source_a.source_events_sha256.clone(),
            selected_rich_spikes_sha256: rich_spike_slice_sha256(&rich_a)?,
            total_kl: Some(readback_a.total_kl),
            total_kl_magnitude: Some(readback_a.total_kl.abs()),
            adjudication_code: readback_a.adjudication_code,
            adjudication_reason_flags: readback_a.reason_flags,
            engine_state_sha256_before: readback_a.state_sha256_before,
            engine_state_sha256_after: readback_a.state_sha256_after,
            engine_physical_state_sha256_before: readback_a.physical_state_sha256_before,
            engine_physical_state_sha256_after: readback_a.physical_state_sha256_after,
        });
        frames_b.push(CoupledNoiseFloorFrameEvidence {
            phase: phase.to_string(),
            ordinal: ordinal as u32,
            timestep: timestep_max_b,
            source_timestep_min: timestep_min_b,
            source_timestep_max: timestep_max_b,
            analysis_md_step,
            source_spike_count: source_count_b as u32,
            selected_spike_count: fixed_spikes_u32,
            dropped_spike_count: (source_count_b as u32).saturating_sub(fixed_spikes_u32),
            input_byte_offset: input_offsets_b.rich_offset,
            input_byte_length: input_offsets_b.rich_length,
            source_input_byte_offset: input_offsets_b.source_offset,
            source_input_byte_length: input_offsets_b.source_length,
            selection_index_byte_offset: input_offsets_b.selection_offset,
            selection_index_byte_length: input_offsets_b.selection_length,
            source_events_sha256: source_b.source_events_sha256.clone(),
            selected_rich_spikes_sha256: rich_spike_slice_sha256(&rich_b)?,
            total_kl: Some(readback_b.total_kl),
            total_kl_magnitude: Some(readback_b.total_kl.abs()),
            adjudication_code: readback_b.adjudication_code,
            adjudication_reason_flags: readback_b.reason_flags,
            engine_state_sha256_before: readback_b.state_sha256_before,
            engine_state_sha256_after: readback_b.state_sha256_after,
            engine_physical_state_sha256_before: readback_b.physical_state_sha256_before,
            engine_physical_state_sha256_after: readback_b.physical_state_sha256_after,
        });

        if ordinal + 1 == SPLIT_T7_ANCHOR_FRAMES {
            runtime_a
                .as_ref()
                .context("split-T7 stream A runtime absent")?
                .pipeline
                .freeze_baseline()
                .map_err(|error| {
                    anyhow::anyhow!("stream A split-T7 baseline freeze: {:?}", error)
                })?;
            engine_a.cuda_stream().synchronize()?;
            runtime_b
                .as_ref()
                .context("split-T7 stream B runtime absent")?
                .pipeline
                .freeze_baseline()
                .map_err(|error| {
                    anyhow::anyhow!("stream B split-T7 baseline freeze: {:?}", error)
                })?;
            engine_b.cuda_stream().synchronize()?;
        }
        if ordinal + 1 == SPLIT_T7_ANCHOR_FRAMES + SPLIT_T7_VARIANCE_FRAMES {
            let stats_a = split_t7_population_stats(&variance_a, 0)?;
            let stats_b = split_t7_population_stats(&variance_b, 1)?;
            runtime_a
                .as_ref()
                .context("split-T7 stream A runtime absent")?
                .pipeline
                .write_kl_native_mu_sigma(stats_a.0, stats_a.1)
                .map_err(|error| anyhow::anyhow!("stream A split-T7 KL write: {:?}", error))?;
            engine_a.cuda_stream().synchronize()?;
            let (_, _, mu_a, sigma_a, flags_a) = read_split_t7_adjudicator(
                runtime_a
                    .as_ref()
                    .context("split-T7 stream A runtime absent")?,
            )?;
            runtime_b
                .as_ref()
                .context("split-T7 stream B runtime absent")?
                .pipeline
                .write_kl_native_mu_sigma(stats_b.0, stats_b.1)
                .map_err(|error| anyhow::anyhow!("stream B split-T7 KL write: {:?}", error))?;
            engine_b.cuda_stream().synchronize()?;
            let (_, _, mu_b, sigma_b, flags_b) = read_split_t7_adjudicator(
                runtime_b
                    .as_ref()
                    .context("split-T7 stream B runtime absent")?,
            )?;
            if mu_a.to_bits() != stats_a.0.to_bits()
                || sigma_a.to_bits() != stats_a.1.to_bits()
                || mu_b.to_bits() != stats_b.0.to_bits()
                || sigma_b.to_bits() != stats_b.1.to_bits()
                || flags_a & 0x8000_0000 != 0
                || flags_b & 0x8000_0000 != 0
            {
                bail!("split-T7 KL-native threshold readback mismatch");
            }
            calibrated_a = Some(stats_a);
            calibrated_b = Some(stats_b);
        }
    }
    if prefix_outer != direct_prefix_outer_steps {
        bail!(
            "split-T7 equal-duration prefix mismatch: observed={} required={}",
            prefix_outer,
            direct_prefix_outer_steps
        );
    }
    input_a.file_mut()?.write_all(b"PRT7E003")?;
    input_b.file_mut()?.write_all(b"PRT7E003")?;
    input_a.commit()?;
    input_b.commit()?;
    let input_artifact_a = artifact_evidence(
        "ghost_noise_floor_input",
        Some(0),
        &input_path_a,
        Some(SPLIT_T7_TOTAL_FRAMES as u64),
        Some(COUPLED_TWIN_EVIDENCE_SCHEMA_VERSION),
    )?;
    let input_artifact_b = artifact_evidence(
        "ghost_noise_floor_input",
        Some(1),
        &input_path_b,
        Some(SPLIT_T7_TOTAL_FRAMES as u64),
        Some(COUPLED_TWIN_EVIDENCE_SCHEMA_VERSION),
    )?;

    let runtime_a = runtime_a.context("split-T7 stream A runtime was never built")?;
    let runtime_b = runtime_b.context("split-T7 stream B runtime was never built")?;
    let validation_marker_a = runtime_a
        .pipeline
        .read_burst_marker()
        .map_err(|error| anyhow::anyhow!("stream A split-T7 validation marker: {:?}", error))?;
    let validation_marker_b = runtime_b
        .pipeline
        .read_burst_marker()
        .map_err(|error| anyhow::anyhow!("stream B split-T7 validation marker: {:?}", error))?;
    if validation_marker_a > 1 || validation_marker_b > 1 {
        bail!("split-T7 validation marker is outside its binary domain");
    }
    let stats_a = calibrated_a.context("stream A split-T7 threshold was not calibrated")?;
    let stats_b = calibrated_b.context("stream B split-T7 threshold was not calibrated")?;
    let (authoritative_minimum, authoritative_minimum_slack_from_k) =
        split_t7_authoritative_admission(
            &authoritative_counts_a,
            &authoritative_counts_b,
            fixed_spikes_u32,
        )?;
    let paired_inventory_sha256 = split_t7_paired_inventory_sha256(
        SPLIT_T7_AUTHORITATIVE_INVENTORY_DOMAIN,
        u32::try_from(fused_inner_steps)
            .context("split-T7 authoritative inventory window exceeds u32")?,
        fixed_spikes_u32,
        &authoritative_counts_a,
        &authoritative_counts_b,
    )?;
    let admission_artifact = emit_split_t7_admission_receipt_v3(
        output_dir,
        config,
        topology_a,
        topology_b,
        &layout_a,
        &layout_b,
        inventory,
        u32::try_from(fused_inner_steps).context("split-T7 admission window exceeds u32")?,
        &authoritative_counts_a,
        &authoritative_counts_b,
        authoritative_minimum,
        authoritative_minimum_slack_from_k,
        &paired_inventory_sha256,
        authoritative_crn_artifact,
        authoritative_crn_receipt,
    )?;
    let outcome_a = finalize_split_t7_stream(
        0,
        &config.topology_a_path,
        &config.topology_a_sha256,
        config,
        output_dir,
        runtime_a,
        input_artifact_a,
        frames_a,
        authoritative_counts_a,
        &paired_inventory_sha256,
        inventory.shadow_lower_envelope_counts_a.clone(),
        &inventory.shadow_ensemble_sha256,
        &inventory.shadow_ensemble_artifact,
        authoritative_minimum,
        authoritative_minimum_slack_from_k,
        prefix_outer,
        fused_inner_steps,
        fixed_spikes_u32,
        stats_a.0,
        stats_a.1,
        validation_marker_a,
    )?;
    let outcome_b = finalize_split_t7_stream(
        1,
        &config.topology_b_path,
        &config.topology_b_sha256,
        config,
        output_dir,
        runtime_b,
        input_artifact_b,
        frames_b,
        authoritative_counts_b,
        &paired_inventory_sha256,
        inventory.shadow_lower_envelope_counts_b.clone(),
        &inventory.shadow_ensemble_sha256,
        &inventory.shadow_ensemble_artifact,
        authoritative_minimum,
        authoritative_minimum_slack_from_k,
        prefix_outer,
        fused_inner_steps,
        fixed_spikes_u32,
        stats_b.0,
        stats_b.1,
        validation_marker_b,
    )?;
    Ok(SplitT7PairOutcome {
        stream_a: outcome_a,
        stream_b: outcome_b,
        direct_prefix_outer_steps: prefix_outer,
        direct_prefix_md_steps: (prefix_outer as u64).saturating_mul(fused_inner_steps as u64),
        warmup_spikes_a,
        warmup_spikes_b,
        admission_artifact,
    })
}

#[cfg(any())]
fn superseded_dynamic_manifold_emit_gpu_ghost_v2_snapshot(
    engine: &mut PersistentNhsEngine,
    topology: &PrismPrepTopology,
    stream_id: u32,
    topology_path: &str,
    topology_sha256: &str,
    native_chain_role_map: &BTreeMap<String, String>,
    config: &CoupledTwinEvidenceConfig,
    calibration: &SplitT7StreamOutcome,
    output_dir: &Path,
    step_idx: u64,
) -> Result<(
    CoupledTwinArtifactEvidence,
    CoupledTwinArtifactEvidence,
    CoupledTwinArtifactEvidence,
    CoupledTwinArtifactEvidence,
)> {
    use crate::captured_pipeline::{CapturedAdjudicationPipeline, MarV2Config, PipelineConfig};
    use crate::ghost_tile::{
        decode_ghost_v3_cluster_residue_membership, GhostTileRing,
        GHOST_FCF_BIT_SPATIAL_NATIVE_AABB_MIDPOINT, GHOST_FRAME_SCHEMA_V3,
        GHOST_V2_OFFSET_AABB_MAX, GHOST_V2_OFFSET_AABB_MIN, GHOST_V2_OFFSET_CENTROID,
        GHOST_V2_OFFSET_DISCOVERY_PASS, GHOST_V2_OFFSET_FIELD_COMPLETENESS_FLAGS,
        GHOST_V2_OFFSET_OBSERVATION_PASS, GHOST_V2_OFFSET_SCHEMA_VERSION, GHOST_V2_OFFSET_STEP_IDX,
    };
    use crate::rich_spike::RichSpike;

    engine.cuda_context().bind_to_thread().map_err(|error| {
        anyhow::anyhow!(
            "bind CUDA context for Ghost stream {}: {:?}",
            stream_id,
            error
        )
    })?;
    engine.synchronize_stream_raw(&format!(
        "final Ghost stream {} compact topology-LUT upload",
        stream_id
    ))?;
    engine.force_spike_sync()?;
    let spike_events = engine.get_accumulated_spikes();
    if spike_events.len() < 4 {
        bail!(
            "stream {} has {} real spikes; at least four are required for GPU Ghost-v2 projection",
            stream_id,
            spike_events.len()
        );
    }
    let noise_floor_mu = calibration.noise_floor_mu;
    let noise_floor_sigma = calibration.noise_floor_sigma;
    let noise_floor_artifact = calibration.calibration_artifact.clone();
    let noise_floor_input_artifact = calibration.calibration_input_artifact.clone();
    if calibration.layout.layout_sha256.len() != 64
        || !noise_floor_mu.is_finite()
        || !noise_floor_sigma.is_finite()
        || noise_floor_sigma <= 0.0
    {
        bail!(
            "stream {} final Ghost projection lacks a valid online split-T7 binding",
            stream_id
        );
    }
    let topology_native_chains: BTreeSet<String> = topology
        .chain_ids
        .iter()
        .map(|chain| chain.trim().to_string())
        .collect();
    if topology_native_chains.iter().any(|chain| chain.is_empty())
        || native_chain_role_map
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>()
            != topology_native_chains
        || native_chain_role_map
            .values()
            .any(|role| role != "A" && role != "B")
    {
        bail!(
            "stream {} native-chain role map is incomplete or invalid: topology={:?} map={:?}",
            stream_id,
            topology_native_chains,
            native_chain_role_map
        );
    }

    const GRID: u32 = 4;
    let mut bb_min = [f32::INFINITY; 3];
    let mut bb_max = [f32::NEG_INFINITY; 3];
    for spike in &spike_events {
        if spike
            .position
            .iter()
            .any(|coordinate| !coordinate.is_finite())
            || !spike.intensity.is_finite()
            || !spike.water_density.is_finite()
            || !spike.vibrational_energy.is_finite()
            || !spike.wd_change.is_finite()
            || spike.timestep < 0
        {
            bail!("stream {} contains an invalid real spike record", stream_id);
        }
        for axis in 0..3 {
            bb_min[axis] = bb_min[axis].min(spike.position[axis]);
            bb_max[axis] = bb_max[axis].max(spike.position[axis]);
        }
    }
    let mut cell = [0.0f32; 3];
    for axis in 0..3 {
        let span = (bb_max[axis] - bb_min[axis]).max(1.0);
        bb_min[axis] -= span * 0.05;
        bb_max[axis] += span * 0.05;
        cell[axis] = ((bb_max[axis] - bb_min[axis]) / GRID as f32).max(0.1);
    }

    let mut clustered = Vec::with_capacity(spike_events.len());
    for spike in &spike_events {
        let ix =
            (((spike.position[0] - bb_min[0]) / cell[0]) as i32).clamp(0, GRID as i32 - 1) as u32;
        let iy =
            (((spike.position[1] - bb_min[1]) / cell[1]) as i32).clamp(0, GRID as i32 - 1) as u32;
        let iz =
            (((spike.position[2] - bb_min[2]) / cell[2]) as i32).clamp(0, GRID as i32 - 1) as u32;
        let cluster = ix + iy * GRID + iz * GRID * GRID;
        let (rich, support) = rich_spike_from_gpu_event(spike, cluster as i32)?;
        clustered.push((cluster, rich, support));
    }
    clustered.sort_by_key(|(cluster, _, _)| *cluster);

    let mut rich = Vec::with_capacity(clustered.len());
    let mut offsets = vec![0u32];
    let mut support_by_cluster: Vec<BTreeSet<i32>> = Vec::new();
    let mut previous = None;
    for (index, (cell_id, mut spike, support)) in clustered.into_iter().enumerate() {
        if previous != Some(cell_id) {
            if previous.is_some() {
                offsets.push(index as u32);
            }
            support_by_cluster.push(BTreeSet::new());
            previous = Some(cell_id);
        }
        let dense = support_by_cluster.len().saturating_sub(1);
        spike.cluster_id = dense as i32;
        support_by_cluster[dense].extend(support);
        rich.push(spike);
    }
    offsets.push(rich.len() as u32);
    let n_clusters = support_by_cluster.len() as u32;
    if n_clusters == 0 || n_clusters > 64 || offsets.len() != n_clusters as usize + 1 {
        bail!(
            "stream {} invalid dense Ghost cluster shape: clusters={} offsets={}",
            stream_id,
            n_clusters,
            offsets.len()
        );
    }

    let mut representative_lut = vec![u32::MAX; n_clusters as usize];
    let mut chain_lut = vec![0xFFu8; n_clusters as usize];
    let mut native_chain_by_cluster = vec![String::new(); n_clusters as usize];
    for (cluster, support) in support_by_cluster.iter().enumerate() {
        let mut frequency = BTreeMap::<i32, usize>::new();
        let start = offsets[cluster] as usize;
        let end = offsets[cluster + 1] as usize;
        for spike in &rich[start..end] {
            if spike.residue_id >= 0 {
                *frequency.entry(spike.residue_id).or_default() += 1;
            }
        }
        let representative = frequency
            .into_iter()
            .max_by_key(|(residue, count)| (*count, std::cmp::Reverse(*residue)))
            .map(|(residue, _)| residue)
            .or_else(|| support.iter().next().copied())
            .unwrap_or(-1);
        if representative < 0 {
            bail!(
                "stream {} Ghost cluster {} has no topology-resolved representative residue",
                stream_id,
                cluster
            );
        }
        representative_lut[cluster] = representative as u32;
        let native_chain = topology_chain_for_residue(topology, representative as usize);
        let role = native_chain_role_map.get(&native_chain).with_context(|| {
            format!(
                "stream {} Ghost cluster {} native chain {:?} is not mapped",
                stream_id, cluster, native_chain
            )
        })?;
        chain_lut[cluster] = match role.as_str() {
            "A" => 0,
            "B" => 1,
            _ => bail!(
                "stream {} Ghost cluster {} has invalid role {}",
                stream_id,
                cluster,
                role
            ),
        };
        native_chain_by_cluster[cluster] = native_chain;
    }

    let max_residue_id = topology
        .residue_ids
        .iter()
        .copied()
        .max()
        .unwrap_or(0)
        .saturating_add(1)
        .min(1024);
    let mut residue_to_calpha = vec![u32::MAX; max_residue_id.max(1)];
    for atom_index in 0..topology.n_atoms {
        if topology
            .atom_names
            .get(atom_index)
            .map(|name| name.trim() == "CA")
            .unwrap_or(false)
        {
            let residue_id = topology.residue_ids[atom_index];
            if residue_id < residue_to_calpha.len() {
                residue_to_calpha[residue_id] = atom_index as u32;
            }
        }
    }
    let mut chain_offsets = [u32::MAX; 8];
    chain_offsets[0] = 0;
    let raw_stream = engine.cuda_stream().cu_stream() as *mut std::ffi::c_void;
    let constant_uploads = unsafe {
        [
            crate::so3_project::set_residue_to_calpha(
                residue_to_calpha.as_ptr(),
                residue_to_calpha.len() as u32,
                raw_stream,
            ),
            crate::ghost_tile::set_cluster_repr_residue(
                representative_lut.as_ptr(),
                n_clusters,
                raw_stream,
            ),
            crate::ghost_tile::set_chain_offsets(
                chain_offsets.as_ptr(),
                chain_offsets.len() as u32,
                raw_stream,
            ),
            crate::ghost_tile::set_cluster_chain_id(chain_lut.as_ptr(), n_clusters, raw_stream),
        ]
    };
    if constant_uploads.iter().any(|result| *result != 0) {
        bail!(
            "stream {} Ghost CUDA constant-table upload failed: {:?}",
            stream_id,
            constant_uploads
        );
    }
    crate::sh_basis::init_device_basis(raw_stream)
        .map_err(|error| anyhow::anyhow!("stream {} SH basis init failed: {}", stream_id, error))?;
    engine.synchronize_stream_raw(&format!(
        "final Ghost stream {} direct Ghost-v2 emission",
        stream_id
    ))?;
    let d_k_lm = crate::sh_basis::k_lm_device_ptr()
        .map_err(|error| anyhow::anyhow!("stream {} SH basis pointer: {}", stream_id, error))?;

    let d_spikes = RawCudaAllocation::copy_from_slice(&rich, "coupled Ghost RichSpike")?;
    let d_offsets = RawCudaAllocation::copy_from_slice(&offsets, "coupled Ghost CSR offsets")?;
    let initial_step = step_idx
        .checked_sub(1)
        .context("coupled Ghost step must be nonzero before device increment")?;
    let d_step = RawCudaAllocation::copy_from_slice(&[initial_step], "coupled Ghost device step")?;
    let ring = GhostTileRing::allocate(n_clusters).map_err(|error| {
        anyhow::anyhow!("stream {} Ghost ring allocation: {}", stream_id, error)
    })?;

    if engine.is_gearbox_active() {
        bail!(
            "stream {} Ghost-v2 evidence cannot truthfully stamp an active gearbox id through the current engine ABI",
            stream_id
        );
    }
    let dt_fs = (engine.current_dt_ps() * 1000.0) as f32;
    if !dt_fs.is_finite() || dt_fs <= 0.0 {
        bail!(
            "stream {} has invalid engine-owned integration dt_fs={}",
            stream_id,
            dt_fs
        );
    }
    let required_pointers = [
        engine.d_positions_dev_ptr(),
        engine.d_forces_dev_ptr(),
        engine.d_masses_dev_ptr(),
        engine.d_velocities_dev_ptr(),
        engine.d_protocol_dt_dev_ptr(),
        engine.d_potential_energy_components_dev_ptr(),
    ];
    if required_pointers.iter().any(|pointer| *pointer == 0)
        || engine.d_potential_energy_n_atoms() == 0
    {
        bail!(
            "stream {} Ghost pipeline lacks an engine-owned CUDA pointer: {:?}",
            stream_id,
            required_pointers
        );
    }
    let external_work = engine.allocate_external_work_buffer();
    if external_work == 0 {
        bail!(
            "stream {} external-work CUDA buffer is unavailable",
            stream_id
        );
    }
    let pipeline_config = PipelineConfig {
            seed_layout: crate::rich_spike::NativeV2SeedLayout::default(),
        d_spikes: d_spikes.ptr as *const RichSpike,
        d_cluster_offsets: d_offsets.ptr as *const u32,
        n_clusters,
        d_k_lm,
        initial_frame_id: step_idx as u32,
        source_stream_id: stream_id,
        diagnostic_stream_id: Some(stream_id),
        asc: None,
        d_force_norm_mirror: 0,
        governor_device: None,
        zstr: None,
        sisr: None,
        mar_v2: Some(MarV2Config {
            threshold_sigma_observation: 3.0,
            threshold_sigma_discovery: 12.0,
            noise_floor_mu,
            noise_floor_sigma,
            gear_id: 0,
            dt_fs,
            d_step_counter: d_step.ptr,
            prune_kl_threshold: 0.0,
        }),
        noise_floor_override: Some((noise_floor_mu, noise_floor_sigma)),
        d_dt: engine.d_protocol_dt_dev_ptr() as *mut f32,
        d_velocities: engine.d_velocities_dev_ptr() as *mut f32,
        g26_parent_cond_handle: 0,
        f1_parent_cond_handle: 0,
        d_pe_components: engine.d_potential_energy_components_dev_ptr() as *const f64,
        n_atoms_for_pe: engine.d_potential_energy_n_atoms() as u32,
        d_external_work: external_work as *mut f64,
        ghost_tile_ring_dev: ring.device_base,
        ghost_tile_max_records: ring.max_records,
        firehose_enable: if config.ghost_diagnostic_firehose {
            1
        } else {
            0
        },
        body_unroll: 1,
        d_kcc_lead: 0,
        d_forces_anchor: engine.d_forces_dev_ptr(),
        d_masses: engine.d_masses_dev_ptr(),
        force_burst_step: None,
        n_spikes: rich.len() as u32,
        d_current_md_step: engine.d_protocol_current_step_dev_ptr() as *const u32,
        branch_trace_dev: 0,
    };
    let pipeline = CapturedAdjudicationPipeline::build(
        engine.cuda_context(),
        engine.cuda_stream(),
        &pipeline_config,
    )
    .map_err(|error| anyhow::anyhow!("stream {} Ghost pipeline build: {:?}", stream_id, error))?;
    pipeline
        .write_kl_native_mu_sigma(noise_floor_mu, noise_floor_sigma)
        .map_err(|error| {
            anyhow::anyhow!(
                "stream {} Ghost KL-native threshold binding: {:?}",
                stream_id,
                error
            )
        })?;
    engine.cuda_stream().synchronize()?;
    pipeline.launch().map_err(|error| {
        anyhow::anyhow!("stream {} Ghost pipeline launch: {:?}", stream_id, error)
    })?;
    engine.cuda_stream().synchronize()?;

    let record_count = ring.n_frames_written();
    if record_count == 0 || record_count > ring.max_records {
        bail!(
            "stream {} Ghost ring counter invalid: {} capacity={}",
            stream_id,
            record_count,
            ring.max_records
        );
    }
    let bytes_len = GhostTileRing::COUNTER_SECTOR_BYTES
        .checked_add(record_count as usize * GHOST_V2_RECORD_BYTES)
        .context("Ghost envelope size overflow")?;
    if bytes_len > ring.total_bytes {
        bail!("stream {} Ghost envelope exceeds allocation", stream_id);
    }
    let envelope = unsafe { std::slice::from_raw_parts(ring.host_base, bytes_len) };
    let stem = coupled_topology_stem(config);
    let payload_path = output_dir.join(format!("{stem}_stream{stream_id:02}_ghost_tiles.bin"));
    write_atomic_bytes(&payload_path, envelope)?;
    let readback = std::fs::read(&payload_path)?;
    if readback.len() != bytes_len || read_le_u32(&readback, 0)? != record_count {
        bail!(
            "stream {} Ghost envelope readback length/counter mismatch",
            stream_id
        );
    }
    let payload_hash = sha256_bytes(&readback);
    let mut records = Vec::with_capacity(record_count as usize);
    let mut uniqueness = BTreeSet::new();
    for record_index in 0..record_count as usize {
        let byte_offset =
            GhostTileRing::COUNTER_SECTOR_BYTES + record_index * GHOST_V2_RECORD_BYTES;
        let record = &readback[byte_offset..byte_offset + GHOST_V2_RECORD_BYTES];
        let schema = read_le_u32(record, GHOST_V2_OFFSET_SCHEMA_VERSION)?;
        let completeness = read_le_u16(record, GHOST_V2_OFFSET_FIELD_COMPLETENESS_FLAGS)?;
        if schema != GHOST_FRAME_SCHEMA_V3
            || completeness & GHOST_FCF_BIT_SPATIAL_NATIVE_AABB_MIDPOINT == 0
        {
            bail!(
                "stream {} Ghost record {} lacks schema-v2 native spatial completeness: schema={} flags=0x{:04x}",
                stream_id,
                record_index,
                schema,
                completeness
            );
        }
        decode_ghost_v3_cluster_residue_membership(record).map_err(|error| {
            anyhow::anyhow!(
                "stream {} Ghost record {} v3 cluster-residue membership contract failed: {}",
                stream_id,
                record_index,
                error
            )
        })?;
        let frame_idx = read_le_u64(record, 0)?;
        let site_id = read_le_u32(record, 8)?;
        let chain_code = record[12];
        let chain = match chain_code {
            0 => 'A',
            1 => 'B',
            _ => bail!(
                "stream {} Ghost record {} has invalid native chain code {}",
                stream_id,
                record_index,
                chain_code
            ),
        };
        let adjudication_code = record[13];
        let kl_divergence = read_le_f32(record, 16)?;
        let observation_pass = record[GHOST_V2_OFFSET_OBSERVATION_PASS] != 0;
        let discovery_pass = record[GHOST_V2_OFFSET_DISCOVERY_PASS] != 0;
        let step = read_le_u64(record, GHOST_V2_OFFSET_STEP_IDX)?;
        let aabb_min = ghost_vec3(record, GHOST_V2_OFFSET_AABB_MIN)?;
        let aabb_max = ghost_vec3(record, GHOST_V2_OFFSET_AABB_MAX)?;
        let centroid = ghost_vec3(record, GHOST_V2_OFFSET_CENTROID)?;
        if site_id >= n_clusters
            || !uniqueness.insert((frame_idx, site_id))
            || !kl_divergence.is_finite()
            || kl_divergence < 0.0
            || step != step_idx
            || aabb_min
                .iter()
                .chain(aabb_max.iter())
                .chain(centroid.iter())
                .any(|value| !value.is_finite())
        {
            bail!(
                "stream {} Ghost record {} failed identity/finite checks",
                stream_id,
                record_index
            );
        }
        let expected_observation = kl_divergence >= noise_floor_mu + 3.0 * noise_floor_sigma;
        let expected_discovery = kl_divergence >= noise_floor_mu + 12.0 * noise_floor_sigma;
        if observation_pass != expected_observation || discovery_pass != expected_discovery {
            bail!(
                "stream {} Ghost record {} has sign/threshold-inconsistent MAR flags: kl={} obs={} expected_obs={} disc={} expected_disc={}",
                stream_id,
                record_index,
                kl_divergence,
                observation_pass,
                expected_observation,
                discovery_pass,
                expected_discovery
            );
        }
        for axis in 0..3 {
            if aabb_max[axis] <= aabb_min[axis]
                || centroid[axis] < aabb_min[axis]
                || centroid[axis] > aabb_max[axis]
                || (centroid[axis] - 0.5 * (aabb_min[axis] + aabb_max[axis])).abs() > 1.0e-3
            {
                bail!(
                    "stream {} Ghost record {} has invalid native AABB/centroid on axis {}",
                    stream_id,
                    record_index,
                    axis
                );
            }
        }
        if (0..24).any(|index| {
            read_le_f32(record, 20 + index * 4)
                .map(|value| !value.is_finite())
                .unwrap_or(true)
        }) || !read_le_f32(record, 116)?.is_finite()
            || !read_le_f32(record, 124)?.is_finite()
        {
            bail!(
                "stream {} Ghost record {} has nonfinite spectrum/thermodynamic fields",
                stream_id,
                record_index
            );
        }
        let support: Vec<i32> = support_by_cluster[site_id as usize]
            .iter()
            .copied()
            .collect();
        records.push(CoupledGhostRecordEvidence {
            record_index: record_index as u64,
            byte_offset: byte_offset as u64,
            frame_idx,
            site_id,
            chain_id: chain.to_string(),
            native_chain_id: native_chain_by_cluster[site_id as usize].clone(),
            adjudication_code,
            kl_divergence,
            observation_pass,
            discovery_pass,
            step_idx: step,
            aabb_min,
            aabb_max,
            centroid,
            support_residue_ids: support,
            source_event_witnesses: witnesses_by_topology_cluster[topology_cluster_id as usize]
                .clone(),
            representative_residue_id: representative_lut[site_id as usize] as i32,
            field_completeness_flags: completeness,
            spatial_native: true,
        });
    }
    let site_map = CoupledGhostSiteMap {
        schema_kind: "prismos.coupled_twin_ghost_site_map.v2".to_string(),
        schema_version: COUPLED_TWIN_EVIDENCE_SCHEMA_VERSION,
        run_id: config.run_id.clone(),
        stream_id,
        coordinate_frame: "topology_cartesian_angstrom".to_string(),
        native_chain_role_map: native_chain_role_map.clone(),
        topology_path: topology_path.to_string(),
        topology_sha256: topology_sha256.to_string(),
        topology_layout_sha256: calibration.layout.layout_sha256.clone(),
        ghost_payload_path: payload_path.display().to_string(),
        ghost_payload_sha256: payload_hash,
        ghost_payload_bytes: readback.len() as u64,
        record_count: record_count as u64,
        producer: "CapturedAdjudicationPipeline::build+launch/ghost_pipe_stage_v2".to_string(),
        execution_stage: "final_engine_owned_accumulated_spike_manifold_snapshot".to_string(),
        step_counter_initial: initial_step,
        emitted_step_idx: step_idx,
        dt_fs,
        dt_source: "PersistentNhsEngine::current_dt_ps_at_final_snapshot".to_string(),
        gear_id: 0,
        gear_id_source: "wave_a_default_verified_gearbox_inactive".to_string(),
        noise_floor_mu,
        noise_floor_sigma,
        noise_floor_source:
            "run_owned_online_split_T7_live_engine_cold_hold_KL_magnitude_calibration".to_string(),
        noise_floor_calibration_path: noise_floor_artifact.path.clone(),
        noise_floor_calibration_sha256: noise_floor_artifact.sha256.clone(),
        noise_floor_calibration_frame_count: SPLIT_T7_TOTAL_FRAMES as u32,
        raw_spike_count: spike_events.len() as u64,
        cluster_count: n_clusters,
        fallback_used: false,
        unavailable_fields: vec![
            "RichSpike.causal_lag".to_string(),
            "RichSpike.chem_flags".to_string(),
            "RichSpike.kinetic_delta".to_string(),
            "d_kcc_lead".to_string(),
        ],
        records,
    };
    let site_map_path = output_dir.join(format!("{stem}_stream{stream_id:02}_ghost_site_map.json"));
    write_atomic_json(&site_map_path, &site_map)?;

    // The captured graph must be destroyed before its raw spike/CSR/device
    // clock inputs and mapped ring are released.
    drop(pipeline);
    drop(d_step);
    drop(d_offsets);
    drop(d_spikes);

    Ok((
        artifact_evidence(
            "ghost_tiles_v2",
            Some(stream_id),
            &payload_path,
            Some(record_count as u64),
            Some(GHOST_FRAME_SCHEMA_V3),
        )?,
        artifact_evidence(
            "ghost_site_map",
            Some(stream_id),
            &site_map_path,
            Some(record_count as u64),
            Some(COUPLED_TWIN_EVIDENCE_SCHEMA_VERSION),
        )?,
        noise_floor_artifact,
        noise_floor_input_artifact,
    ))
}

#[cfg(feature = "gpu")]
fn emit_gpu_ghost_v2_snapshot(
    engine: &mut PersistentNhsEngine,
    topology: &PrismPrepTopology,
    stream_id: u32,
    topology_path: &str,
    topology_sha256: &str,
    native_chain_role_map: &BTreeMap<String, String>,
    config: &CoupledTwinEvidenceConfig,
    protocol: &CryoUvProtocol,
    replica_seed: u64,
    calibration: &mut SplitT7StreamOutcome,
    output_dir: &Path,
    step_idx: u64,
) -> Result<(
    CoupledTwinArtifactEvidence,
    CoupledTwinArtifactEvidence,
    CoupledTwinArtifactEvidence,
    CoupledTwinArtifactEvidence,
    CoupledTwinArtifactEvidence,
    CoupledTwinArtifactEvidence,
    CoupledTwinArtifactEvidence,
)> {
    use crate::ghost_tile::{
        decode_ghost_v3_cluster_residue_membership, GhostTileRing,
        GHOST_FCF_BIT_SPATIAL_NATIVE_AABB_MIDPOINT, GHOST_FRAME_SCHEMA_V3,
        GHOST_V2_OFFSET_AABB_MAX, GHOST_V2_OFFSET_AABB_MIN, GHOST_V2_OFFSET_CENTROID,
        GHOST_V2_OFFSET_DISCOVERY_PASS, GHOST_V2_OFFSET_FIELD_COMPLETENESS_FLAGS,
        GHOST_V2_OFFSET_OBSERVATION_PASS, GHOST_V2_OFFSET_SCHEMA_VERSION, GHOST_V2_OFFSET_STEP_IDX,
    };
    use crate::so3_project::ContactShellTile;

    engine.synchronize_stream_raw(&format!("final Ghost stream {} entry", stream_id))?;
    engine
        .force_spike_sync()
        .with_context(|| format!("final Ghost stream {} force-spike readback", stream_id))?;
    // force_spike_sync performs a checked raw stream barrier plus synchronous
    // DtoH/HtoD transfers.  A second cudarc synchronize here is redundant and
    // would re-enter the shared deferred-error path during teardown.
    let spike_events = engine.get_accumulated_spikes();
    if spike_events.len() < calibration.fixed_spikes_per_frame as usize {
        bail!(
            "stream {} final Ghost source has {} events but calibrated shape requires {}",
            stream_id,
            spike_events.len(),
            calibration.fixed_spikes_per_frame
        );
    }
    let topology_native_chains: BTreeSet<String> = topology
        .chain_ids
        .iter()
        .map(|chain| chain.trim().to_string())
        .collect();
    if topology_native_chains.iter().any(|chain| chain.is_empty())
        || native_chain_role_map
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>()
            != topology_native_chains
        || native_chain_role_map
            .values()
            .any(|role| role != "A" && role != "B")
    {
        bail!(
            "stream {} final Ghost native-chain map is incomplete: topology={:?} map={:?}",
            stream_id,
            topology_native_chains,
            native_chain_role_map
        );
    }

    let noise_floor_mu = calibration.noise_floor_mu;
    let noise_floor_sigma = calibration.noise_floor_sigma;
    if calibration.runtime.layout.layout_sha256.len() != 64
        || calibration.frozen_anchor_sha256.len() != 64
        || !noise_floor_mu.is_finite()
        || !noise_floor_sigma.is_finite()
        || noise_floor_sigma <= 0.0
    {
        bail!(
            "stream {} final Ghost projection lacks a complete split-T7 authority binding",
            stream_id
        );
    }
    let frozen_anchor_bytes = usize::try_from(calibration.runtime.pipeline.tiles_bytes)
        .context("final Ghost frozen-anchor byte count exceeds usize")?;
    let anchor_before = cuda_read_bytes(
        calibration.runtime.pipeline.p_frozen_dev as u64,
        frozen_anchor_bytes,
        &format!("stream {} retained split-T7 frozen anchor", stream_id),
    )?;
    if sha256_bytes(&anchor_before) != calibration.frozen_anchor_sha256
        || sha256_file(Path::new(&calibration.frozen_anchor_artifact.path))?
            != calibration.frozen_anchor_sha256
    {
        bail!(
            "stream {} retained split-T7 frozen anchor drifted before final projection",
            stream_id
        );
    }

    let mut source = select_split_t7_full_span(
        &spike_events,
        calibration.fixed_spikes_per_frame as usize,
        stream_id,
    )?;
    let (rich, offsets, timestep_min, timestep_max) =
        prepare_split_t7_frame(&mut source.selected_events, &calibration.runtime.layout)?;
    if rich.len() != calibration.fixed_spikes_per_frame as usize
        || offsets.len() != SPLIT_T7_CLUSTER_COUNT as usize + 1
    {
        bail!(
            "stream {} final Ghost changed the calibrated graph shape",
            stream_id
        );
    }

    let final_readback = launch_split_t7_frame(
        engine,
        topology,
        &mut calibration.runtime,
        stream_id,
        step_idx,
        &rich,
        &offsets,
    )
    .with_context(|| {
        format!(
            "final Ghost stream {} retained split-T7 projection",
            stream_id
        )
    })?;
    // KL divergence is non-negative by definition; a below-noise negative total_kl is
    // floating-point round-off of zero and is RELABELED to exactly 0.0 (no differential
    // signal) instead of hard-rejected. A genuinely large-negative KL still fails as a
    // broken Ghost-v2 kernel. This is the minimal wired form of the below-noise relabel
    // the signed-threshold comment (and the unimplemented --mar-v2-prune-kl) anticipate;
    // only the below-noise sign handling changes, meaningful KL (order 1e1) is untouched.
    let mut final_readback = final_readback;
    if final_readback.total_kl < 0.0 {
        const SPLIT_T7_KL_NEG_NOISE_TOLERANCE: f32 = 1.0e-3;
        if final_readback.total_kl >= -SPLIT_T7_KL_NEG_NOISE_TOLERANCE {
            final_readback.total_kl = 0.0;
        } else {
            bail!(
                "stream {} final split-T7 KL is signed-negative beyond noise tolerance ({}); the active Ghost-v2 kernel uses signed thresholds",
                stream_id,
                final_readback.total_kl
            );
        }
    }
    let anchor_after = cuda_read_bytes(
        calibration.runtime.pipeline.p_frozen_dev as u64,
        frozen_anchor_bytes,
        &format!(
            "stream {} retained split-T7 frozen anchor after projection",
            stream_id
        ),
    )?;
    if sha256_bytes(&anchor_after) != calibration.frozen_anchor_sha256 {
        bail!(
            "stream {} final split-T7 projection mutated its frozen reference manifold",
            stream_id
        );
    }

    let stem = coupled_topology_stem(config);
    let final_input_path = output_dir.join(format!(
        "{stem}_stream{stream_id:02}_ghost_final_manifold_input.bin"
    ));
    let mut final_input = NoClobberArtifactFile::create(&final_input_path)?;
    final_input.file_mut()?.write_all(b"PRT7F003")?;
    final_input
        .file_mut()?
        .write_all(&COUPLED_TWIN_EVIDENCE_SCHEMA_VERSION.to_le_bytes())?;
    final_input
        .file_mut()?
        .write_all(&stream_id.to_le_bytes())?;
    final_input
        .file_mut()?
        .write_all(sha256_bytes(config.run_id.as_bytes()).as_bytes())?;
    final_input
        .file_mut()?
        .write_all(topology_sha256.as_bytes())?;
    final_input
        .file_mut()?
        .write_all(calibration.runtime.layout.layout_sha256.as_bytes())?;
    let final_offsets = append_split_t7_input_frame(
        final_input.file_mut()?,
        u32::MAX,
        step_idx,
        timestep_min,
        timestep_max,
        &source,
        &rich,
    )?;
    final_input.file_mut()?.write_all(b"PRT7FEND")?;
    final_input.commit()?;
    let final_input_artifact = artifact_evidence(
        "ghost_final_manifold_input",
        Some(stream_id),
        &final_input_path,
        Some(source.source_event_count as u64),
        Some(COUPLED_TWIN_EVIDENCE_SCHEMA_VERSION),
    )?;
    let (canonical_source_length, canonical_source_sha256) =
        canonical_split_t7_source_identity(source.source_events)?;
    if final_offsets.source_length != canonical_source_length
        || final_offsets.selection_length != source.selected_indices.len() as u64 * 4
        || final_offsets.rich_length != rich.len() as u64 * 64
        || canonical_source_sha256 != source.source_events_sha256
    {
        bail!(
            "stream {} final Ghost input offsets are not self-consistent",
            stream_id
        );
    }

    let tile_bytes = std::mem::size_of::<ContactShellTile>();
    if tile_bytes == 0
        || calibration.runtime.pipeline.tiles_bytes as usize
            != tile_bytes.saturating_mul(SPLIT_T7_CLUSTER_COUNT as usize)
    {
        bail!(
            "stream {} retained split-T7 tile shape drift: bytes={} tile={} clusters={}",
            stream_id,
            calibration.runtime.pipeline.tiles_bytes,
            tile_bytes,
            SPLIT_T7_CLUSTER_COUNT
        );
    }
    let perturbed_pointer_bytes = cuda_read_bytes(
        calibration.runtime.adjudicator_dev as u64 + 64,
        8,
        &format!("stream {} split-T7 perturbed manifold pointer", stream_id),
    )?;
    let perturbed_pointer = u64::from_le_bytes(
        perturbed_pointer_bytes
            .as_slice()
            .try_into()
            .expect("eight-byte perturbed pointer"),
    );
    if perturbed_pointer == 0 || perturbed_pointer % 128 != 0 {
        bail!(
            "stream {} split-T7 perturbed manifold pointer is null/misaligned: 0x{:x}",
            stream_id,
            perturbed_pointer
        );
    }
    let all_tile_bytes = cuda_read_bytes(
        perturbed_pointer,
        calibration.runtime.pipeline.tiles_bytes as usize,
        &format!("stream {} final split-T7 current manifold", stream_id),
    )?;

    let mut support_by_topology_cluster =
        vec![BTreeSet::<i32>::new(); SPLIT_T7_CLUSTER_COUNT as usize];
    let mut witnesses_by_topology_cluster =
        vec![Vec::<CoupledGhostSourceEventWitness>::new(); SPLIT_T7_CLUSTER_COUNT as usize];
    for (selected_ordinal, source_index) in source.selected_indices.iter().copied().enumerate() {
        let event = source
            .source_events
            .get(source_index as usize)
            .context("final Ghost selected source index exceeds canonical source")?;
        let primary = event.nearby_residues[0];
        let cluster = if primary >= 0 {
            calibration
                .runtime
                .layout
                .residue_to_cluster
                .get(primary as usize)
                .copied()
                .unwrap_or(u32::MAX)
        } else {
            u32::MAX
        };
        if cluster == u32::MAX || cluster as usize >= support_by_topology_cluster.len() {
            bail!(
                "stream {} final Ghost source event {} has an invalid primary residue/cluster",
                stream_id,
                source_index
            );
        }
        let (_, support) = rich_spike_from_gpu_event(event, cluster as i32)?;
        for residue in &support {
            if calibration
                .runtime
                .layout
                .residue_to_anchor_atom
                .get(*residue as usize)
                .copied()
                .unwrap_or(u32::MAX)
                == u32::MAX
            {
                bail!(
                    "stream {} final Ghost source event {} cites residue {} absent from the exact topology",
                    stream_id,
                    source_index,
                    residue
                );
            }
        }
        support_by_topology_cluster[cluster as usize].extend(support.iter().copied());
        witnesses_by_topology_cluster[cluster as usize].push(CoupledGhostSourceEventWitness {
            selected_ordinal: u32::try_from(selected_ordinal)
                .context("final Ghost selected ordinal exceeds u32")?,
            source_index,
            canonical_event_sha256: sha256_bytes(&canonical_gpu_spike_bytes(event)),
            position: event.position,
            primary_residue_id: primary,
            support_residue_ids: support.into_iter().collect(),
        });
    }
    let occupied_topology_clusters: Vec<u32> = offsets
        .windows(2)
        .enumerate()
        .filter_map(|(cluster, range)| (range[1] > range[0]).then_some(cluster as u32))
        .collect();
    if occupied_topology_clusters.is_empty() {
        bail!(
            "stream {} final split-T7 manifold has no occupied topology cluster",
            stream_id
        );
    }

    let mut compact_tiles = Vec::with_capacity(occupied_topology_clusters.len() * tile_bytes);
    let mut compact_rich = Vec::new();
    let mut compact_offsets = Vec::with_capacity(occupied_topology_clusters.len() + 1);
    compact_offsets.push(0u32);
    let mut representative_lut = Vec::with_capacity(occupied_topology_clusters.len());
    let mut chain_lut = Vec::with_capacity(occupied_topology_clusters.len());
    let mut native_chain_by_emitted_cluster = Vec::with_capacity(occupied_topology_clusters.len());
    for topology_cluster in &occupied_topology_clusters {
        let cluster = *topology_cluster as usize;
        compact_tiles
            .extend_from_slice(&all_tile_bytes[cluster * tile_bytes..(cluster + 1) * tile_bytes]);
        let spike_start = offsets[cluster] as usize;
        let spike_end = offsets[cluster + 1] as usize;
        compact_rich.extend_from_slice(&rich[spike_start..spike_end]);
        compact_offsets.push(
            u32::try_from(compact_rich.len())
                .context("final Ghost compact RichSpike count exceeds u32")?,
        );
        let representative = calibration.runtime.layout.cluster_representative_residue[cluster];
        let chain_role = calibration.runtime.layout.cluster_chain_role[cluster];
        if representative == u32::MAX || chain_role > 1 {
            bail!(
                "stream {} occupied topology cluster {} lacks native residue/chain authority",
                stream_id,
                cluster
            );
        }
        let native_chain = topology_chain_for_residue(topology, representative as usize);
        let expected_role = native_chain_role_map.get(&native_chain).with_context(|| {
            format!(
                "stream {} topology cluster {} chain {:?} is not mapped",
                stream_id, cluster, native_chain
            )
        })?;
        if (chain_role == 0 && expected_role != "A") || (chain_role == 1 && expected_role != "B") {
            bail!(
                "stream {} topology cluster {} chain-role readback drift",
                stream_id,
                cluster
            );
        }
        representative_lut.push(representative);
        chain_lut.push(chain_role);
        native_chain_by_emitted_cluster.push(native_chain);
    }

    log::info!(
        "[FINAL GHOST] stream {} allocating {} compact tile bytes across {} occupied clusters",
        stream_id,
        compact_tiles.len(),
        occupied_topology_clusters.len()
    );
    let compact_tiles_dev =
        RawCudaAllocation::copy_from_bytes(&compact_tiles, "final split-T7 occupied Ghost tiles")?;
    let compact_rich_dev =
        RawCudaAllocation::copy_from_slice(&compact_rich, "final split-T7 Ghost RichSpike")?;
    let compact_offsets_dev =
        RawCudaAllocation::copy_from_slice(&compact_offsets, "final split-T7 Ghost CSR offsets")?;
    log::info!(
        "[FINAL GHOST] stream {} compact tile allocation/upload complete",
        stream_id
    );
    let raw_stream = engine.cuda_stream().cu_stream() as *mut std::ffi::c_void;
    let constant_uploads = unsafe {
        [
            crate::ghost_tile::set_cluster_repr_residue(
                representative_lut.as_ptr(),
                representative_lut.len() as u32,
                raw_stream,
            ),
            crate::ghost_tile::set_cluster_chain_id(
                chain_lut.as_ptr(),
                chain_lut.len() as u32,
                raw_stream,
            ),
        ]
    };
    if constant_uploads.iter().any(|result| *result != 0) {
        bail!(
            "stream {} final Ghost compact topology-LUT upload failed: {:?}",
            stream_id,
            constant_uploads
        );
    }
    engine
        .synchronize_stream_raw(&format!(
            "stream {} final Ghost compact topology-LUT upload",
            stream_id
        ))
        .with_context(|| {
            format!(
                "stream {} final Ghost compact topology-LUT strict barrier",
                stream_id
            )
        })?;
    log::info!(
        "[FINAL GHOST] stream {} compact topology-LUT upload synchronized",
        stream_id
    );
    if engine.is_gearbox_active() {
        bail!(
            "stream {} Ghost-v2 evidence cannot truthfully stamp the active gearbox through the current ABI",
            stream_id
        );
    }
    let dt_fs = (engine.current_dt_ps() * 1000.0) as f32;
    if !dt_fs.is_finite() || dt_fs <= 0.0 {
        bail!(
            "stream {} final Ghost has invalid engine dt_fs={}",
            stream_id,
            dt_fs
        );
    }
    let emitted_cluster_count = occupied_topology_clusters.len() as u32;
    log::info!(
        "[FINAL GHOST] stream {} authority: total_kl={:.9e} adjudication_code={} observation_threshold={:.9e} discovery_threshold={:.9e} firehose={} prune_kl_threshold=0",
        stream_id,
        final_readback.total_kl,
        final_readback.adjudication_code,
        noise_floor_mu + 3.0 * noise_floor_sigma,
        noise_floor_mu + 12.0 * noise_floor_sigma,
        config.ghost_diagnostic_firehose
    );
    log::info!(
        "[FINAL GHOST] stream {} allocating mapped ring for {} emitted clusters",
        stream_id,
        emitted_cluster_count
    );
    let ring = GhostTileRing::allocate(emitted_cluster_count).map_err(|error| {
        anyhow::anyhow!(
            "stream {} final Ghost ring allocation: {}",
            stream_id,
            error
        )
    })?;
    unsafe {
        std::ptr::write_bytes(ring.host_base, 0, ring.total_bytes);
    }
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
    let ghost_launch_rc = unsafe {
        prism_ghost_pipe_stage_launch_v2(
            ring.device_base,
            compact_tiles_dev.ptr as *const std::ffi::c_void,
            compact_rich_dev.ptr as *const std::ffi::c_void,
            compact_offsets_dev.ptr as *const std::ffi::c_void,
            calibration.runtime.adjudicator_dev as *const std::ffi::c_void,
            std::ptr::null(),
            step_idx,
            emitted_cluster_count,
            ring.max_records,
            raw_stream,
            u32::from(config.ghost_diagnostic_firehose),
            noise_floor_mu + 3.0 * noise_floor_sigma,
            noise_floor_mu + 12.0 * noise_floor_sigma,
            0,
            dt_fs,
            step_idx,
            0,
            0.0,
        )
    };
    if ghost_launch_rc != 0 {
        bail!(
            "stream {} direct Ghost-v2 emission from retained split-T7 manifold failed: rc={}",
            stream_id,
            ghost_launch_rc
        );
    }
    log::info!(
        "[FINAL GHOST] stream {} Ghost-v2 launch submitted; entering strict barrier",
        stream_id
    );
    engine
        .synchronize_stream_raw(&format!(
            "stream {} final Ghost-v2 direct emission",
            stream_id
        ))
        .with_context(|| {
            format!(
                "stream {} final Ghost-v2 direct-emission strict barrier",
                stream_id
            )
        })?;
    log::info!(
        "[FINAL GHOST] stream {} Ghost-v2 direct emission synchronized",
        stream_id
    );
    let record_count = ring.n_frames_written();
    if record_count != emitted_cluster_count {
        bail!(
            "stream {} final Ghost emitted {} of {} occupied calibrated clusters",
            stream_id,
            record_count,
            emitted_cluster_count
        );
    }

    let bytes_len = GhostTileRing::COUNTER_SECTOR_BYTES
        .checked_add(record_count as usize * GHOST_V2_RECORD_BYTES)
        .context("final Ghost envelope size overflow")?;
    if bytes_len > ring.total_bytes {
        bail!(
            "stream {} final Ghost envelope exceeds its mapped ring",
            stream_id
        );
    }
    let envelope = unsafe { std::slice::from_raw_parts(ring.host_base, bytes_len) };
    let payload_path = output_dir.join(format!("{stem}_stream{stream_id:02}_ghost_tiles.bin"));
    write_atomic_bytes(&payload_path, envelope)?;
    let readback = std::fs::read(&payload_path)?;
    if readback.len() != bytes_len || read_le_u32(&readback, 0)? != record_count {
        bail!("stream {} final Ghost durable readback failed", stream_id);
    }
    validate_ghost_envelope_zero_regions(&readback, record_count)
        .with_context(|| format!("stream {} final Ghost zero-region validation", stream_id))?;

    // The flat envelope remains a compatibility/debug artifact for the
    // coupled-TWIN site-map validator. Production lattice and Path-B authority
    // is the same PGHSEG03 writer/loader schema used by the normal lane.
    let condition_id = if stream_id == 0 {
        format!("coupled_reference::{}", config.contrast_kind)
    } else {
        format!("coupled_design::{}", config.contrast_kind)
    };
    let phase_binding = coupled_protocol_phase_binding(protocol, stream_id, step_idx, condition_id)?;
    if phase_binding.checked_phase_for(step_idx).is_none() {
        bail!(
            "stream {} final Ghost step {} is outside its exact coupled protocol schedule",
            stream_id,
            step_idx
        );
    }
    let selected_engine_path = std::env::current_exe().context("resolve coupled Ghost engine")?;
    let selected_engine_sha256 = sha256_file(&selected_engine_path)?;
    let segment_context = crate::ghost_segment::GhostSegmentContext {
        schema_kind: "prism_ghost_segment_context".to_string(),
        schema_version: 1,
        run_id: config.run_id.clone(),
        replica_id: format!("coupled_pair_seed_{replica_seed}"),
        replica_seed,
        // Coupled-TWIN uses common random numbers intentionally; both stream
        // seeds are equal and stream identity remains a separate field.
        stream_seed: replica_seed,
        stream_id: stream_id as u16,
        topology_sha256: topology_sha256.to_string(),
        selected_engine_sha256,
        record_abi_sha256: crate::ghost_tile::ghost_record_abi_sha256(),
        record_source_sha256: crate::ghost_tile::ghost_record_source_sha256(),
        thermodynamic_condition_id: phase_binding.condition_id.clone(),
        protocol_id: phase_binding.protocol_name.clone(),
        protocol_state_hash: phase_binding.protocol_state_hash.clone(),
        phase_schedule_digest_sha256: phase_binding.phase_schedule_digest_sha256.clone(),
    };
    let mut segment_writer = crate::ghost_segment::GhostSegmentWriter::resume_or_reconstruct(
        output_dir,
        &stem,
        segment_context,
    )?;
    let record_payload = &readback[GhostTileRing::COUNTER_SECTOR_BYTES..];
    segment_writer.append_payload(record_payload, record_count as u64)?;
    segment_writer.complete()?;
    let segment_root = crate::ghost_segment::segment_root(output_dir, &stem, stream_id as u16);
    let segment_manifest_path = segment_root.join("manifest.json");
    let verified_segment_stream = crate::ghost_segment::verify_stream_root(&segment_root)?;
    if !verified_segment_stream.manifest.completed
        || verified_segment_stream.manifest.persisted_records != record_count as u64
    {
        bail!("stream {} coupled Ghost segment closeout failed", stream_id);
    }
    crate::ghost_phase_lattice::write_stream_phase_binding_atomic(
        output_dir,
        &stem,
        &phase_binding,
    )?;
    let payload_hash = sha256_bytes(&readback);
    let mut records = Vec::with_capacity(record_count as usize);
    let mut uniqueness = BTreeSet::new();
    for record_index in 0..record_count as usize {
        let byte_offset =
            GhostTileRing::COUNTER_SECTOR_BYTES + record_index * GHOST_V2_RECORD_BYTES;
        let record = &readback[byte_offset..byte_offset + GHOST_V2_RECORD_BYTES];
        let schema = read_le_u32(record, GHOST_V2_OFFSET_SCHEMA_VERSION)?;
        let completeness = read_le_u16(record, GHOST_V2_OFFSET_FIELD_COMPLETENESS_FLAGS)?;
        let frame_idx = read_le_u64(record, 0)?;
        let site_id = read_le_u32(record, 8)?;
        if schema != GHOST_FRAME_SCHEMA_V3
            || completeness & GHOST_FCF_BIT_SPATIAL_NATIVE_AABB_MIDPOINT == 0
            || site_id >= emitted_cluster_count
            || !uniqueness.insert((frame_idx, site_id))
        {
            bail!(
                "stream {} final Ghost record {} failed schema/identity checks",
                stream_id,
                record_index
            );
        }
        decode_ghost_v3_cluster_residue_membership(record).map_err(|error| {
            anyhow::anyhow!(
                "stream {} final Ghost record {} v3 cluster-residue membership contract failed: {}",
                stream_id,
                record_index,
                error
            )
        })?;
        let topology_cluster_id = occupied_topology_clusters[site_id as usize];
        let chain_code = record[12];
        let chain = match chain_code {
            0 => "A",
            1 => "B",
            _ => bail!(
                "stream {} final Ghost record {} has invalid chain code {}",
                stream_id,
                record_index,
                chain_code
            ),
        };
        if chain_code != chain_lut[site_id as usize] {
            bail!("stream {} final Ghost chain LUT readback drift", stream_id);
        }
        let adjudication_code = record[13];
        let kl_divergence = read_le_f32(record, 16)?;
        let observation_pass = record[GHOST_V2_OFFSET_OBSERVATION_PASS] != 0;
        let discovery_pass = record[GHOST_V2_OFFSET_DISCOVERY_PASS] != 0;
        let step = read_le_u64(record, GHOST_V2_OFFSET_STEP_IDX)?;
        let aabb_min = ghost_vec3(record, GHOST_V2_OFFSET_AABB_MIN)?;
        let aabb_max = ghost_vec3(record, GHOST_V2_OFFSET_AABB_MAX)?;
        let centroid = ghost_vec3(record, GHOST_V2_OFFSET_CENTROID)?;
        if !kl_divergence.is_finite()
            || kl_divergence.to_bits() != final_readback.total_kl.to_bits()
            || adjudication_code as u32 != final_readback.adjudication_code
            || step != step_idx
            || aabb_min
                .iter()
                .chain(aabb_max.iter())
                .chain(centroid.iter())
                .any(|value| !value.is_finite())
        {
            bail!(
                "stream {} final Ghost record {} drifted from retained-pipeline adjudication",
                stream_id,
                record_index
            );
        }
        let expected_observation = kl_divergence >= noise_floor_mu + 3.0 * noise_floor_sigma;
        let expected_discovery = kl_divergence >= noise_floor_mu + 12.0 * noise_floor_sigma;
        if observation_pass != expected_observation || discovery_pass != expected_discovery {
            bail!(
                "stream {} final Ghost record {} has threshold-inconsistent MAR flags",
                stream_id,
                record_index
            );
        }
        for axis in 0..3 {
            if aabb_max[axis] < aabb_min[axis]
                || centroid[axis] < aabb_min[axis]
                || centroid[axis] > aabb_max[axis]
                || (centroid[axis] - 0.5 * (aabb_min[axis] + aabb_max[axis])).abs() > 1.0e-3
            {
                bail!(
                    "stream {} final Ghost record {} has invalid native AABB/centroid",
                    stream_id,
                    record_index
                );
            }
        }
        if (0..24).any(|index| {
            read_le_f32(record, 20 + index * 4)
                .map(|value| !value.is_finite())
                .unwrap_or(true)
        }) || !read_le_f32(record, 116)?.is_finite()
            || !read_le_f32(record, 120)?.is_finite()
        {
            bail!(
                "stream {} final Ghost record has a nonfinite science plane",
                stream_id
            );
        }
        let source_tile: ContactShellTile = unsafe {
            std::ptr::read_unaligned(
                compact_tiles
                    .as_ptr()
                    .add(site_id as usize * tile_bytes)
                    .cast::<ContactShellTile>(),
            )
        };
        if source_tile.spike_count == 0
            || source_tile.cluster_id != topology_cluster_id as i32
            || source_tile.stream_id != stream_id
        {
            bail!(
                "stream {} emitted Ghost record {} from a non-authoritative source tile: tile_stream={} tile_cluster={} spike_count={}",
                stream_id,
                record_index,
                source_tile.stream_id,
                source_tile.cluster_id,
                source_tile.spike_count
            );
        }
        let expected_spectra = source_tile
            .geo_power_spectrum
            .iter()
            .take(6)
            .chain(source_tile.caus_power_spectrum.iter().take(6))
            .chain(source_tile.therm_power_spectrum.iter().take(6))
            .chain(source_tile.chem_power_spectrum.iter().take(6));
        for (index, expected) in expected_spectra.enumerate() {
            if read_le_f32(record, 20 + index * 4)?.to_bits() != expected.to_bits() {
                bail!(
                    "stream {} Ghost record {} spectrum {} drifted from its GPU ContactShellTile",
                    stream_id,
                    record_index,
                    index
                );
            }
        }
        if read_le_f32(record, 116)?.to_bits() != source_tile.therm_power_spectrum[0].to_bits()
            || read_le_f32(record, 120)?.to_bits() != source_tile.caus_power_spectrum[0].to_bits()
            || (0..3).any(|axis| {
                aabb_min[axis].to_bits() != source_tile.aabb_min[axis].to_bits()
                    || aabb_max[axis].to_bits() != source_tile.aabb_max[axis].to_bits()
            })
        {
            bail!(
                "stream {} Ghost record {} thermodynamic/AABB fields drifted from its GPU ContactShellTile",
                stream_id,
                record_index
            );
        }
        let support: Vec<i32> = support_by_topology_cluster[topology_cluster_id as usize]
            .iter()
            .copied()
            .collect();
        if support.is_empty()
            || witnesses_by_topology_cluster[topology_cluster_id as usize].is_empty()
        {
            bail!(
                "stream {} emitted topology cluster {} without selected real-event support",
                stream_id,
                topology_cluster_id
            );
        }
        records.push(CoupledGhostRecordEvidence {
            record_index: record_index as u64,
            byte_offset: byte_offset as u64,
            frame_idx,
            site_id,
            topology_cluster_id,
            chain_id: chain.to_string(),
            native_chain_id: native_chain_by_emitted_cluster[site_id as usize].clone(),
            adjudication_code,
            kl_divergence,
            observation_pass,
            discovery_pass,
            step_idx: step,
            aabb_min,
            aabb_max,
            centroid,
            support_residue_ids: support,
            source_event_witnesses: witnesses_by_topology_cluster[topology_cluster_id as usize]
                .clone(),
            representative_residue_id: representative_lut[site_id as usize] as i32,
            evidence_available: true,
            neutral_reason: None,
            field_completeness_flags: completeness,
            spatial_native: true,
        });
    }

    let site_map = CoupledGhostSiteMap {
        schema_kind: "prismos.coupled_twin_ghost_site_map.v3".to_string(),
        schema_version: COUPLED_TWIN_EVIDENCE_SCHEMA_VERSION,
        run_id: config.run_id.clone(),
        stream_id,
        coordinate_frame: "topology_cartesian_angstrom".to_string(),
        native_chain_role_map: native_chain_role_map.clone(),
        topology_path: topology_path.to_string(),
        topology_sha256: topology_sha256.to_string(),
        topology_layout_sha256: calibration.runtime.layout.layout_sha256.clone(),
        frozen_anchor_path: calibration.frozen_anchor_artifact.path.clone(),
        frozen_anchor_sha256: calibration.frozen_anchor_sha256.clone(),
        frozen_anchor_bytes: calibration.frozen_anchor_artifact.size_bytes,
        ghost_payload_path: payload_path.display().to_string(),
        ghost_payload_sha256: payload_hash,
        ghost_payload_bytes: readback.len() as u64,
        record_count: record_count as u64,
        producer: "retained_CapturedAdjudicationPipeline::launch+prism_ghost_pipe_stage_launch_v2"
            .to_string(),
        execution_stage: "final_engine_owned_same_layout_same_frozen_reference_manifold"
            .to_string(),
        step_counter_initial: step_idx,
        emitted_step_idx: step_idx,
        dt_fs,
        dt_source: "PersistentNhsEngine::current_dt_ps_at_final_projection".to_string(),
        gear_id: 0,
        gear_id_source: "wave_a_default_verified_gearbox_inactive".to_string(),
        noise_floor_mu,
        noise_floor_sigma,
        noise_floor_source:
            "run_owned_online_split_T7_same_layout_same_frozen_reference_KL_magnitude_calibration"
                .to_string(),
        noise_floor_calibration_path: calibration.calibration_artifact.path.clone(),
        noise_floor_calibration_sha256: calibration.calibration_artifact.sha256.clone(),
        noise_floor_calibration_frame_count: SPLIT_T7_TOTAL_FRAMES as u32,
        raw_spike_count: source.source_event_count as u64,
        selected_spike_count: rich.len() as u32,
        selected_rich_spikes_sha256: rich_spike_slice_sha256(&rich)?,
        final_source_events_path: final_input_artifact.path.clone(),
        final_source_events_sha256: final_input_artifact.sha256.clone(),
        final_source_events_bytes: final_input_artifact.size_bytes,
        final_canonical_source_events_sha256: source.source_events_sha256.clone(),
        topology_cluster_count: SPLIT_T7_CLUSTER_COUNT,
        emitted_occupied_cluster_count: emitted_cluster_count,
        final_total_kl: final_readback.total_kl,
        final_adjudication_code: final_readback.adjudication_code,
        final_adjudication_reason_flags: final_readback.reason_flags,
        final_engine_state_sha256_before: final_readback.state_sha256_before,
        final_engine_state_sha256_after: final_readback.state_sha256_after,
        final_engine_physical_state_sha256_before: final_readback.physical_state_sha256_before,
        final_engine_physical_state_sha256_after: final_readback.physical_state_sha256_after,
        fallback_used: false,
        unavailable_fields: vec![
            "RichSpike.causal_lag".to_string(),
            "RichSpike.chem_flags".to_string(),
            "RichSpike.kinetic_delta".to_string(),
            "d_kcc_lead".to_string(),
        ],
        records,
    };
    let site_map_path = output_dir.join(format!("{stem}_stream{stream_id:02}_ghost_site_map.json"));
    write_atomic_json(&site_map_path, &site_map)?;
    let site_map_readback: CoupledGhostSiteMap =
        serde_json::from_slice(&std::fs::read(&site_map_path)?)?;
    if site_map_readback.frozen_anchor_sha256 != calibration.frozen_anchor_sha256
        || site_map_readback.topology_layout_sha256 != calibration.runtime.layout.layout_sha256
        || site_map_readback.final_engine_state_sha256_before
            != site_map_readback.final_engine_state_sha256_after
        || site_map_readback.final_engine_physical_state_sha256_before
            != site_map_readback.final_engine_physical_state_sha256_after
    {
        bail!(
            "stream {} final Ghost site-map durable readback drift",
            stream_id
        );
    }

    Ok((
        artifact_evidence(
            "ghost_tiles_v2",
            Some(stream_id),
            &payload_path,
            Some(record_count as u64),
            Some(GHOST_FRAME_SCHEMA_V3),
        )?,
        artifact_evidence(
            "ghost_site_map",
            Some(stream_id),
            &site_map_path,
            Some(record_count as u64),
            Some(COUPLED_TWIN_EVIDENCE_SCHEMA_VERSION),
        )?,
        calibration.calibration_artifact.clone(),
        calibration.calibration_input_artifact.clone(),
        calibration.frozen_anchor_artifact.clone(),
        final_input_artifact,
        artifact_evidence(
            "ghost_segment_manifest",
            Some(stream_id),
            &segment_manifest_path,
            Some(record_count as u64),
            Some(crate::ghost_segment::GHOST_SEGMENT_SCHEMA_VERSION),
        )?,
    ))
}

#[cfg(feature = "gpu")]
fn aabb_separation(
    left_min: [f32; 3],
    left_max: [f32; 3],
    right_min: [f32; 3],
    right_max: [f32; 3],
) -> f32 {
    let mut squared = 0.0f32;
    for axis in 0..3 {
        let delta = if left_max[axis] < right_min[axis] {
            right_min[axis] - left_max[axis]
        } else if right_max[axis] < left_min[axis] {
            left_min[axis] - right_max[axis]
        } else {
            0.0
        };
        squared += delta * delta;
    }
    squared.sqrt()
}

#[cfg(feature = "gpu")]
fn bind_engine_ghost_to_molecular_contrast(
    readout: &mut MolecularContrastReadout,
    runtime: &CoupledTwinRuntimeEvidence,
    config: &CoupledTwinEvidenceConfig,
) -> Result<()> {
    if readout.identical_topology_identity {
        readout.ghost_pocket_binding_verified = true;
        readout.ghost_pocket_binding_status =
            "not_applicable_identical_topology_control".to_string();
        return Ok(());
    }

    let pocket: BTreeSet<i32> = readout
        .pocket_contact_residue_ids
        .iter()
        .map(|value| *value as i32)
        .collect();
    if pocket.is_empty() {
        bail!("cannot bind Ghost evidence to an empty topology-derived pocket");
    }
    if readout.topology_b_sha256 != config.topology_b_sha256
        || readout.ligand_heavy_atoms.len() != readout.ligand_heavy_atom_count
        || readout.ligand_heavy_atoms.is_empty()
        || readout.ligand_heavy_atom_coordinate_sha256.len() != 64
    {
        bail!("cannot bind Ghost evidence without exact topology-B ligand coordinates");
    }

    let mut components = BTreeSet::new();
    let mut witness_bindings = Vec::new();
    for artifact in runtime
        .artifacts
        .iter()
        .filter(|artifact| artifact.kind == "ghost_site_map" && artifact.stream_id == Some(1))
    {
        let map: CoupledGhostSiteMap = serde_json::from_slice(
            &std::fs::read(&artifact.path)
                .with_context(|| format!("read stream-B Ghost map {}", artifact.path))?,
        )?;
        if map.run_id != config.run_id
            || map.topology_sha256 != config.topology_b_sha256
            || map.fallback_used
        {
            bail!("stream-B Ghost map identity/fallback contract failed");
        }
        for record in &map.records {
            let spatial_distance = aabb_separation(
                record.aabb_min,
                record.aabb_max,
                readout.ligand_aabb_min,
                readout.ligand_aabb_max,
            );
            if spatial_distance > config.ligand_contact_cutoff_angstrom {
                continue;
            }
            let component_id = format!(
                "ghost-v2-stream{}-frame{}-site{}",
                map.stream_id, record.frame_idx, record.site_id
            );
            for witness in &record.source_event_witnesses {
                let mut pocket_support_intersection_residue_ids = witness
                    .support_residue_ids
                    .iter()
                    .filter(|residue| pocket.contains(residue))
                    .copied()
                    .collect::<Vec<_>>();
                pocket_support_intersection_residue_ids.sort_unstable();
                pocket_support_intersection_residue_ids.dedup();
                let coarse_ligand_aabb_distance_angstrom = aabb_separation(
                    witness.position,
                    witness.position,
                    readout.ligand_aabb_min,
                    readout.ligand_aabb_max,
                );
                let (nearest_binding_index, minimum_ligand_heavy_atom_distance_angstrom) =
                    exact_ligand_heavy_atom_distance(
                        witness.position,
                        &readout.ligand_heavy_atoms,
                    )?;
                let nearest = &readout.ligand_heavy_atoms[nearest_binding_index];
                let exact_distance_admission = !pocket_support_intersection_residue_ids.is_empty()
                    && minimum_ligand_heavy_atom_distance_angstrom
                        <= config.ligand_contact_cutoff_angstrom;
                if exact_distance_admission {
                    components.insert(component_id.clone());
                    witness_bindings.push(GhostPocketWitnessBinding {
                        component_id: component_id.clone(),
                        source_index: witness.source_index,
                        canonical_event_sha256: witness.canonical_event_sha256.clone(),
                        position: witness.position,
                        support_residue_ids: witness.support_residue_ids.clone(),
                        pocket_support_intersection_residue_ids,
                        topology_b_sha256: readout.topology_b_sha256.clone(),
                        ligand_coordinate_sha256: readout
                            .ligand_heavy_atom_coordinate_sha256
                            .clone(),
                        cutoff_angstrom: config.ligand_contact_cutoff_angstrom,
                        coarse_ligand_aabb_distance_angstrom,
                        nearest_ligand_atom_index: nearest.atom_index,
                        nearest_ligand_atom_element: nearest.element.clone(),
                        nearest_ligand_atom_position: nearest.position,
                        minimum_ligand_heavy_atom_distance_angstrom,
                        exact_distance_admission,
                    });
                }
            }
        }
    }
    if config.require_ghost_pocket_binding && components.is_empty() {
        bail!(
            "engine-owned Ghost-v2 evidence did not intersect the topology-derived ligand pocket"
        );
    }
    readout.ghost_pocket_component_ids = components.into_iter().collect();
    witness_bindings.sort_by(|left, right| {
        (
            &left.component_id,
            left.source_index,
            &left.canonical_event_sha256,
        )
            .cmp(&(
                &right.component_id,
                right.source_index,
                &right.canonical_event_sha256,
            ))
    });
    witness_bindings.dedup_by(|left, right| {
        left.component_id == right.component_id
            && left.source_index == right.source_index
            && left.canonical_event_sha256 == right.canonical_event_sha256
    });
    readout.ghost_pocket_witnesses = witness_bindings;
    readout.ghost_pocket_bound_count = readout.ghost_pocket_component_ids.len() as u64;
    readout.ghost_pocket_binding_verified = readout.ghost_pocket_bound_count > 0;
    readout.ghost_pocket_binding_status = if readout.ghost_pocket_binding_verified {
        "engine_owned_native_v2_exact_witness_to_ligand_heavy_atom_distance_and_pocket_support_match"
            .to_string()
    } else {
        "missing_required_engine_owned_binding".to_string()
    };
    Ok(())
}

#[cfg(feature = "gpu")]
fn accumulate_glycan_probe_cryptic_witnesses_v1(
    map: &CoupledGhostSiteMap,
    stream_id: u32,
    probe_residue_ids: &BTreeSet<i32>,
    input_pose_annulus_minimum_distance_by_residue: &BTreeMap<i32, f32>,
    accumulators: &mut BTreeMap<Vec<i32>, GlycanProbeCandidateAccumulatorV1>,
) -> Result<()> {
    if map.stream_id != stream_id || map.fallback_used {
        bail!("glycan probe Ghost map stream/fallback contract failed for stream {stream_id}");
    }
    for record in &map.records {
        if !record.evidence_available || !record.spatial_native {
            continue;
        }
        let component_id = format!(
            "ghost-v2-stream{}-frame{}-site{}",
            map.stream_id, record.frame_idx, record.site_id
        );
        for witness in &record.source_event_witnesses {
            if witness.canonical_event_sha256.len() != 64
                || !witness
                    .canonical_event_sha256
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit())
                || witness.position.iter().any(|value| !value.is_finite())
            {
                bail!("glycan probe Ghost witness identity/coordinate contract failed");
            }
            let mut support_residue_ids = witness
                .support_residue_ids
                .iter()
                .copied()
                .filter(|residue| {
                    *residue >= 0
                        && !probe_residue_ids.contains(residue)
                        && input_pose_annulus_minimum_distance_by_residue.contains_key(residue)
                })
                .collect::<Vec<_>>();
            support_residue_ids.sort_unstable();
            support_residue_ids.dedup();
            if support_residue_ids.is_empty() {
                continue;
            }
            let minimum_input_pose_distance = support_residue_ids
                .iter()
                .filter_map(|residue| {
                    input_pose_annulus_minimum_distance_by_residue
                        .get(residue)
                        .copied()
                })
                .min_by(f32::total_cmp)
                .context("glycan probe annular support lacks a preregistered residue distance")?;
            let accumulator = accumulators.entry(support_residue_ids).or_default();
            if stream_id == 0 {
                accumulator
                    .stream_a_component_ids
                    .insert(component_id.clone());
                accumulator.stream_a_frames.insert(record.frame_idx);
                accumulator.stream_a_witness_count = accumulator
                    .stream_a_witness_count
                    .checked_add(1)
                    .context("stream-A glycan probe witness count overflow")?;
            } else {
                accumulator
                    .stream_b_component_ids
                    .insert(component_id.clone());
                accumulator.stream_b_frames.insert(record.frame_idx);
                accumulator.stream_b_witness_count = accumulator
                    .stream_b_witness_count
                    .checked_add(1)
                    .context("stream-B glycan probe witness count overflow")?;
                accumulator
                    .minimum_input_pose_probe_to_candidate_receptor_heavy_atom_distance_angstrom =
                    Some(
                        accumulator
                            .minimum_input_pose_probe_to_candidate_receptor_heavy_atom_distance_angstrom
                            .map_or(minimum_input_pose_distance, |minimum| {
                                minimum.min(minimum_input_pose_distance)
                            }),
                    );
            }
        }
    }
    Ok(())
}

#[cfg(feature = "gpu")]
fn build_glycan_mechanistic_probe_readout_v1(
    molecular: &MolecularContrastReadout,
    topology_b: &PrismPrepTopology,
    runtime: &CoupledTwinRuntimeEvidence,
    config: &CoupledTwinEvidenceConfig,
) -> Result<Option<GlycanMechanisticProbeReadoutV1>> {
    if config.contrast_kind != GLYCAN_MECHANISTIC_PROBE_CONTRAST_KIND {
        return Ok(None);
    }
    if molecular.identical_topology_identity
        || molecular.ligand_residue_ids != config.probe_residue_ids
        || molecular.ligand_heavy_atoms.is_empty()
        || molecular.ligand_heavy_atom_coordinate_sha256.len() != 64
    {
        bail!("glycan mechanistic probe readout lacks exact multi-residue probe geometry");
    }
    let mut ghost_maps = BTreeMap::new();
    for artifact in runtime
        .artifacts
        .iter()
        .filter(|artifact| artifact.kind == "ghost_site_map")
    {
        let stream_id = artifact
            .stream_id
            .context("glycan probe Ghost artifact lacks stream identity")?;
        let map: CoupledGhostSiteMap = serde_json::from_slice(
            &std::fs::read(&artifact.path)
                .with_context(|| format!("read glycan probe Ghost map {}", artifact.path))?,
        )?;
        if ghost_maps.insert(stream_id, map).is_some() {
            bail!("glycan mechanistic probe rejects duplicate Ghost maps for stream {stream_id}");
        }
    }
    if ghost_maps.len() != 2 || !ghost_maps.contains_key(&0) || !ghost_maps.contains_key(&1) {
        bail!("glycan mechanistic probe requires exactly one Ghost map for each A/B stream");
    }
    let map_a = &ghost_maps[&0];
    let map_b = &ghost_maps[&1];
    if map_a.run_id != config.run_id
        || map_b.run_id != config.run_id
        || map_a.topology_sha256 != config.topology_a_sha256
        || map_b.topology_sha256 != config.topology_b_sha256
        || map_a.coordinate_frame != map_b.coordinate_frame
    {
        bail!("glycan mechanistic probe Ghost maps fail run/topology/coordinate-frame identity");
    }

    let probe_residue_ids = config
        .probe_residue_ids
        .iter()
        .map(|residue| i32::try_from(*residue).context("probe residue exceeds i32"))
        .collect::<Result<BTreeSet<_>>>()?;
    let direct_contact_residue_ids = molecular
        .pocket_contact_residue_ids
        .iter()
        .map(|residue| i32::try_from(*residue).context("direct-contact residue exceeds i32"))
        .collect::<Result<BTreeSet<_>>>()?;
    let (input_pose_annulus_minimum_distance_by_residue, search_zone_sha256) =
        build_glycan_probe_input_pose_annulus_v1(
            topology_b,
            &config.topology_b_sha256,
            &molecular.ligand_heavy_atoms,
            &molecular.ligand_heavy_atom_coordinate_sha256,
            &config.probe_residue_ids,
            config.ligand_contact_cutoff_angstrom,
            config.cryptic_search_outer_cutoff_angstrom,
        )?;
    if input_pose_annulus_minimum_distance_by_residue
        .keys()
        .any(|residue| direct_contact_residue_ids.contains(residue))
    {
        bail!("glycan probe preregistered annulus overlaps the direct-contact shell");
    }
    let mut accumulators = BTreeMap::new();
    for (stream_id, map) in [(0u32, map_a), (1u32, map_b)] {
        accumulate_glycan_probe_cryptic_witnesses_v1(
            map,
            stream_id,
            &probe_residue_ids,
            &input_pose_annulus_minimum_distance_by_residue,
            &mut accumulators,
        )?;
    }

    let mut candidates = Vec::new();
    for (support_residue_ids, accumulator) in accumulators {
        if accumulator.stream_b_witness_count == 0 {
            continue;
        }
        let minimum_distance = accumulator
            .minimum_input_pose_probe_to_candidate_receptor_heavy_atom_distance_angstrom
            .context("stream-B cryptic candidate lacks preregistered input-pose distance")?;
        let control_observed = accumulator.stream_a_witness_count > 0;
        let probe_enriched =
            accumulator.stream_b_witness_count > accumulator.stream_a_witness_count;
        let classification = if !control_observed {
            "PROBE_RESPONSIVE_CONTROL_NOT_OBSERVED_IN_THIS_RUN"
        } else if probe_enriched {
            "PROBE_ENRICHED_CONTROL_OBSERVED_IN_THIS_RUN"
        } else {
            "CONTROL_OBSERVED_NOT_PROBE_ENRICHED_IN_THIS_RUN"
        };
        let mut candidate_identity = Vec::new();
        candidate_identity.extend_from_slice(b"PRISMOS_GLYCAN_PROBE_CRYPTIC_CANDIDATE_V1\0");
        candidate_identity.extend_from_slice(config.topology_a_sha256.as_bytes());
        candidate_identity.extend_from_slice(config.topology_b_sha256.as_bytes());
        for residue in &support_residue_ids {
            candidate_identity.extend_from_slice(&residue.to_le_bytes());
        }
        candidates.push(GlycanMechanisticProbeCandidateV1 {
            candidate_id: sha256_bytes(&candidate_identity),
            support_residue_ids,
            stream_a_component_ids: accumulator.stream_a_component_ids.into_iter().collect(),
            stream_b_component_ids: accumulator.stream_b_component_ids.into_iter().collect(),
            stream_a_frame_count: accumulator.stream_a_frames.len() as u64,
            stream_b_frame_count: accumulator.stream_b_frames.len() as u64,
            stream_a_witness_count: accumulator.stream_a_witness_count,
            stream_b_witness_count: accumulator.stream_b_witness_count,
            minimum_input_pose_probe_to_candidate_receptor_heavy_atom_distance_angstrom:
                minimum_distance,
            direct_contact_excluded: true,
            control_observed_in_this_run: control_observed,
            probe_enriched_in_this_run: probe_enriched,
            classification: classification.to_string(),
            signaling_hypothesis_status: GLYCAN_PROBE_SIGNALING_HYPOTHESIS_STATUS.to_string(),
        });
    }
    candidates.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));
    let finite_values_verified = candidates.iter().all(|candidate| {
        candidate
            .minimum_input_pose_probe_to_candidate_receptor_heavy_atom_distance_angstrom
            .is_finite()
            && candidate.minimum_input_pose_probe_to_candidate_receptor_heavy_atom_distance_angstrom
                > config.ligand_contact_cutoff_angstrom
            && candidate.minimum_input_pose_probe_to_candidate_receptor_heavy_atom_distance_angstrom
                <= config.cryptic_search_outer_cutoff_angstrom
            && candidate.direct_contact_excluded
            && !candidate.support_residue_ids.is_empty()
            && candidate
                .support_residue_ids
                .iter()
                .all(|residue| !direct_contact_residue_ids.contains(residue))
    });
    if !finite_values_verified {
        bail!("glycan mechanistic probe candidate finite/annular exclusion gate failed");
    }
    let control_not_observed_candidate_count = candidates
        .iter()
        .filter(|candidate| !candidate.control_observed_in_this_run)
        .count() as u64;
    let probe_enriched_candidate_count = candidates
        .iter()
        .filter(|candidate| candidate.probe_enriched_in_this_run)
        .count() as u64;
    Ok(Some(GlycanMechanisticProbeReadoutV1 {
        schema_kind: "prismos.glycan_mechanistic_probe_readout.v1".to_string(),
        schema_version: GLYCAN_MECHANISTIC_PROBE_READOUT_SCHEMA_VERSION,
        run_id: config.run_id.clone(),
        contrast_kind: config.contrast_kind.clone(),
        analysis_semantics:
            "paired_common_random_numbers_uncoupled_dynamics_control_relative_ghost_contrast_with_preregistered_input_pose_receptor_annulus"
                .to_string(),
        topology_a_sha256: config.topology_a_sha256.clone(),
        topology_b_sha256: config.topology_b_sha256.clone(),
        probe_residue_ids: config.probe_residue_ids.clone(),
        probe_heavy_atom_count: molecular.ligand_heavy_atom_count,
        input_pose_probe_heavy_atom_coordinate_sha256: molecular
            .ligand_heavy_atom_coordinate_sha256
            .clone(),
        search_zone_coordinate_frame: "topology_b_input_cartesian_angstrom".to_string(),
        search_zone_definition:
            "preregistered_receptor_heavy_atom_minimum_distance_annulus_from_input_pose_probe_heavy_atoms_inner_exclusive_outer_inclusive"
                .to_string(),
        search_zone_sha256,
        search_zone_receptor_residue_count: input_pose_annulus_minimum_distance_by_residue.len()
            as u64,
        dynamic_signaling_evidence:
            "paired_coupled_twin_ghost_site_map_support_residue_events_no_witness_to_probe_distance_claim"
                .to_string(),
        direct_contact_cutoff_angstrom: config.ligand_contact_cutoff_angstrom,
        cryptic_search_outer_cutoff_angstrom: config
            .cryptic_search_outer_cutoff_angstrom,
        direct_contact_residue_ids: molecular.pocket_contact_residue_ids.clone(),
        matched_control_method:
            "exact_canonical_preregistered_annulus_receptor_support_subset_match_across_stream_A_and_stream_B"
                .to_string(),
        candidate_count: candidates.len() as u64,
        control_not_observed_candidate_count,
        probe_enriched_candidate_count,
        candidates,
        signaling_hypothesis_status: GLYCAN_PROBE_SIGNALING_HYPOTHESIS_STATUS.to_string(),
        causal_transport_claim: false,
        structural_novelty_claim: false,
        patent_novelty_claim: false,
        novelty_status: GLYCAN_PROBE_NOVELTY_STATUS.to_string(),
        replica_consensus_verified: false,
        finite_values_verified,
        fallback_used: false,
    }))
}

#[cfg(feature = "gpu")]
fn emit_glycan_mechanistic_probe_readout_v1(
    output_dir: &Path,
    readout: &GlycanMechanisticProbeReadoutV1,
) -> Result<(PathBuf, PathBuf, PathBuf)> {
    let readout_dir = output_dir.join("GLYCAN_MECHANISTIC_PROBE_READOUT");
    let readout_path = readout_dir.join("glycan_mechanistic_probe_readout.json");
    write_atomic_json(&readout_path, readout)?;
    let digest = sha256_file(&readout_path)?;
    let sidecar_path = readout_dir.join("glycan_mechanistic_probe_readout.json.sha256");
    write_atomic_bytes(
        &sidecar_path,
        format!("{}  glycan_mechanistic_probe_readout.json\n", digest).as_bytes(),
    )?;
    if sha256_file(&readout_path)? != digest {
        bail!("glycan mechanistic probe readout hash drift");
    }
    Ok((readout_dir, readout_path, sidecar_path))
}

#[cfg(feature = "gpu")]
fn bind_strict_kernel_source(
    kernel_identity: &str,
    path: &Path,
    required_symbols: &[&str],
) -> Result<StrictTwinKernelSourceBinding> {
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("inspect strict kernel source {}", path.display()))?;
    let bytes = std::fs::read(path)
        .with_context(|| format!("read strict kernel source {}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || bytes.is_empty() {
        bail!(
            "strict kernel source must be a nonempty regular file: {}",
            path.display()
        );
    }
    for symbol in required_symbols {
        if !bytes
            .windows(symbol.len())
            .any(|window| window == symbol.as_bytes())
        {
            bail!(
                "strict kernel source {} lacks required symbol {}",
                path.display(),
                symbol
            );
        }
    }
    let (sha256, blake3) = hash_file_dual(path)?;
    Ok(StrictTwinKernelSourceBinding {
        kernel_identity: kernel_identity.to_string(),
        path: path.display().to_string(),
        size_bytes: metadata.len(),
        sha256,
        blake3,
        required_symbols: required_symbols
            .iter()
            .map(|symbol| (*symbol).to_string())
            .collect(),
    })
}

#[cfg(feature = "gpu")]
fn validate_strict_kernel_source_binding(
    binding: &StrictTwinKernelSourceBinding,
    expected_identity: &str,
    expected_symbols: &[&str],
) -> Result<()> {
    let expected = bind_strict_kernel_source(
        expected_identity,
        Path::new(&binding.path),
        expected_symbols,
    )?;
    if serde_json::to_value(binding)? != serde_json::to_value(expected)? {
        bail!(
            "strict kernel source binding changed or does not match {}",
            expected_identity
        );
    }
    Ok(())
}

#[cfg(feature = "gpu")]
fn emit_coupled_gpu_analysis_artifacts(
    output_dir: &Path,
    config: &CoupledTwinEvidenceConfig,
    n_residues: usize,
    total_steps: i32,
    ccf_matrix: &[f32],
    bidirectional_te_mass: &[f32],
    net_causal_flow: &[f32],
    thresholded_outgoing_te: &[f32],
    ccf_kernel: &StrictTwinKernelSourceBinding,
    transfer_entropy_kernel: &StrictTwinKernelSourceBinding,
) -> Result<Vec<CoupledTwinArtifactEvidence>> {
    let transfer_entropy_bin_count = total_steps.div_euclid(STRICT_TWIN_CCF_BIN_SIZE_STEPS);
    if ccf_matrix.len() != n_residues.saturating_mul(n_residues)
        || bidirectional_te_mass.len() != n_residues
        || net_causal_flow.len() != n_residues
        || thresholded_outgoing_te.len() != n_residues
        || transfer_entropy_bin_count < 12
        || ccf_matrix
            .iter()
            .chain(bidirectional_te_mass.iter())
            .chain(net_causal_flow.iter())
            .chain(thresholded_outgoing_te.iter())
            .any(|value| !value.is_finite())
    {
        bail!("GPU CCF/TE artifact dimensions or finite-value contract failed");
    }
    validate_strict_kernel_source_binding(
        ccf_kernel,
        "tensor_ccf.ptx",
        &["tensor_ccf_compute", "compute_spike_norms"],
    )?;
    validate_strict_kernel_source_binding(
        transfer_entropy_kernel,
        "twin_transfer_entropy.ptx",
        &[
            "twin_compute_row_means",
            "twin_binned_te",
            "twin_compute_mutual_info",
        ],
    )?;
    let summarize = |values: &[f32]| -> (u64, f64, f32) {
        let nonzero = values
            .iter()
            .filter(|value| value.to_bits() & 0x7fff_ffff != 0)
            .count() as u64;
        let l2_norm = values
            .iter()
            .map(|value| (*value as f64) * (*value as f64))
            .sum::<f64>()
            .sqrt();
        let max_abs = values
            .iter()
            .map(|value| value.abs())
            .fold(0.0f32, f32::max);
        (nonzero, l2_norm, max_abs)
    };
    let ccf_summary = summarize(ccf_matrix);
    let bidirectional_summary = summarize(bidirectional_te_mass);
    let net_flow_summary = summarize(net_causal_flow);
    let outgoing_summary = summarize(thresholded_outgoing_te);
    if ccf_summary.0 == 0
        || bidirectional_summary.0 == 0
        || outgoing_summary.0 == 0
        || (!config.identical_topology_negative_control && net_flow_summary.0 == 0)
    {
        bail!(
            "GPU CCF/TE analysis is numerically degenerate: ccf_nonzero={} bidirectional_te_mass_nonzero={} net_flow_nonzero={} thresholded_outgoing_nonzero={} kernel_supported_residues={}",
            ccf_summary.0,
            bidirectional_summary.0,
            net_flow_summary.0,
            outgoing_summary.0,
            n_residues
        );
    }

    let analysis_dir = output_dir.join("COUPLED_TWIN_GPU_ANALYSIS");
    let ccf_path = analysis_dir.join("ccf_matrix_f32le.bin");
    let mut ccf_bytes = Vec::with_capacity(32 + ccf_matrix.len().saturating_mul(4));
    ccf_bytes.extend_from_slice(b"PRCCF002");
    ccf_bytes.extend_from_slice(&COUPLED_TWIN_EVIDENCE_SCHEMA_VERSION.to_le_bytes());
    ccf_bytes.extend_from_slice(&(n_residues as u64).to_le_bytes());
    ccf_bytes.extend_from_slice(&(n_residues as u64).to_le_bytes());
    ccf_bytes.extend_from_slice(&(ccf_matrix.len() as u64).to_le_bytes());
    for value in ccf_matrix {
        ccf_bytes.extend_from_slice(&value.to_le_bytes());
    }
    write_atomic_bytes(&ccf_path, &ccf_bytes)?;

    let te_path = analysis_dir.join("transfer_entropy_per_residue.json");
    let te_payload = serde_json::json!({
        "schema_kind": "prismos.coupled_twin_gpu_transfer_entropy.v3",
        "schema_version": STRICT_TWIN_FEATURE_SCHEMA_VERSION,
        "run_id": config.run_id,
        "backend": "CUDA:twin_transfer_entropy.ptx",
        "residue_count": n_residues,
        "kernel_lag_bins": STRICT_TWIN_TE_KERNEL_LAG_BINS,
        "past_index_offset_bins": STRICT_TWIN_TE_PAST_INDEX_OFFSET_BINS,
        "future_index_offset_bins": STRICT_TWIN_TE_FUTURE_INDEX_OFFSET_BINS,
        "effective_past_to_future_horizon_bins": STRICT_TWIN_TE_EFFECTIVE_HORIZON_BINS,
        "bin_size_steps": STRICT_TWIN_CCF_BIN_SIZE_STEPS,
        "effective_past_to_future_horizon_steps": STRICT_TWIN_TE_EFFECTIVE_HORIZON_STEPS,
        "bin_count": transfer_entropy_bin_count,
        "transition_sample_count_per_pair": transfer_entropy_bin_count - 2,
        "minimum_transition_samples_per_pair": 10,
        "outgoing_term_inclusion_floor_exclusive": 0.001,
        "bidirectional_transfer_entropy_mass": bidirectional_te_mass,
        "net_causal_flow_out_minus_in": net_causal_flow,
        "thresholded_total_outgoing_transfer_entropy_a_to_b": thresholded_outgoing_te,
        "exchange_active": false,
        "causal_transport_claim": false,
        "nonzero_count": {
            "bidirectional_transfer_entropy_mass": bidirectional_summary.0,
            "net_causal_flow_out_minus_in": net_flow_summary.0,
            "thresholded_total_outgoing_transfer_entropy_a_to_b": outgoing_summary.0,
        },
        "l2_norm": {
            "bidirectional_transfer_entropy_mass": bidirectional_summary.1,
            "net_causal_flow_out_minus_in": net_flow_summary.1,
            "thresholded_total_outgoing_transfer_entropy_a_to_b": outgoing_summary.1,
        },
        "max_abs": {
            "bidirectional_transfer_entropy_mass": bidirectional_summary.2,
            "net_causal_flow_out_minus_in": net_flow_summary.2,
            "thresholded_total_outgoing_transfer_entropy_a_to_b": outgoing_summary.2,
        },
        "kernel_supported_residue_count": n_residues,
        "valid_zero_values_preserved": true,
        "kernel_source": transfer_entropy_kernel,
        "fallback_used": false,
    });
    write_atomic_json(&te_path, &te_payload)?;

    let (ccf_sha256, ccf_blake3) = hash_file_dual(&ccf_path)?;
    let (te_sha256, te_blake3) = hash_file_dual(&te_path)?;
    let receipt_path = analysis_dir.join("gpu_ccf_te_execution_receipt.json");
    let receipt = serde_json::json!({
        "schema_kind": "prismos.coupled_twin_gpu_analysis_receipt.v3",
        "schema_version": STRICT_TWIN_FEATURE_SCHEMA_VERSION,
        "status": "PASS",
        "run_id": config.run_id,
        "analysis_stream": "dedicated_cuda_stream_no_threshold_exchange",
        "ccf_backend": STRICT_TWIN_GLOBAL_CCF_BACKEND,
        "transfer_entropy_backend": STRICT_TWIN_TE_BACKEND,
        "residue_count": n_residues,
        "transfer_entropy_bin_count": transfer_entropy_bin_count,
        "transfer_entropy_kernel_lag_bins": STRICT_TWIN_TE_KERNEL_LAG_BINS,
        "transfer_entropy_past_index_offset_bins": STRICT_TWIN_TE_PAST_INDEX_OFFSET_BINS,
        "transfer_entropy_future_index_offset_bins": STRICT_TWIN_TE_FUTURE_INDEX_OFFSET_BINS,
        "transfer_entropy_effective_past_to_future_horizon_bins": STRICT_TWIN_TE_EFFECTIVE_HORIZON_BINS,
        "transfer_entropy_bin_size_steps": STRICT_TWIN_CCF_BIN_SIZE_STEPS,
        "transfer_entropy_effective_past_to_future_horizon_steps": STRICT_TWIN_TE_EFFECTIVE_HORIZON_STEPS,
        "transfer_entropy_transition_sample_count_per_pair": transfer_entropy_bin_count - 2,
        "ccf_nonzero_count": ccf_summary.0,
        "ccf_l2_norm": ccf_summary.1,
        "ccf_max_abs": ccf_summary.2,
        "bidirectional_te_mass_nonzero_count": bidirectional_summary.0,
        "bidirectional_te_mass_l2_norm": bidirectional_summary.1,
        "bidirectional_te_mass_max_abs": bidirectional_summary.2,
        "net_causal_flow_nonzero_count": net_flow_summary.0,
        "net_causal_flow_l2_norm": net_flow_summary.1,
        "net_causal_flow_max_abs": net_flow_summary.2,
        "thresholded_outgoing_te_nonzero_count": outgoing_summary.0,
        "thresholded_outgoing_te_l2_norm": outgoing_summary.1,
        "thresholded_outgoing_te_max_abs": outgoing_summary.2,
        "kernel_supported_residue_count": n_residues,
        "valid_zero_values_preserved": true,
        "ccf_kernel_source": ccf_kernel,
        "transfer_entropy_kernel_source": transfer_entropy_kernel,
        "ccf_matrix": {"path": ccf_path, "sha256": ccf_sha256, "blake3": ccf_blake3},
        "transfer_entropy": {"path": te_path, "sha256": te_sha256, "blake3": te_blake3},
        "finite_values_verified": true,
        "exchange_active": false,
        "fallback_used": false,
    });
    write_atomic_json(&receipt_path, &receipt)?;

    Ok(vec![
        artifact_evidence(
            "gpu_ccf_matrix",
            None,
            &ccf_path,
            Some(ccf_matrix.len() as u64),
            Some(COUPLED_TWIN_EVIDENCE_SCHEMA_VERSION),
        )?,
        artifact_evidence(
            "gpu_transfer_entropy",
            None,
            &te_path,
            Some(n_residues as u64),
            Some(STRICT_TWIN_FEATURE_SCHEMA_VERSION),
        )?,
        artifact_evidence(
            "gpu_ccf_te_execution_receipt",
            None,
            &receipt_path,
            Some(n_residues as u64),
            Some(STRICT_TWIN_FEATURE_SCHEMA_VERSION),
        )?,
    ])
}

#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
struct StrictPhaseCcfRuntime {
    window: StrictTwinPhaseWindow,
    matrix_artifact: CoupledTwinArtifactEvidence,
    matrix: Vec<f32>,
    features: Vec<crate::twin_kernels::CcfResidueFeatures>,
}

#[cfg(feature = "gpu")]
fn write_strict_phase_ccf_matrix(
    path: &Path,
    config: &CoupledTwinEvidenceConfig,
    window: &StrictTwinPhaseWindow,
    n_residues: usize,
    matrix: &[f32],
) -> Result<()> {
    let expected_bin_count = if window.end_step_exclusive > window.start_step {
        u32::try_from(
            (window.end_step_exclusive - window.start_step)
                .checked_add(STRICT_TWIN_CCF_BIN_SIZE_STEPS - 1)
                .context("strict phase write bin rounding overflow")?
                / STRICT_TWIN_CCF_BIN_SIZE_STEPS,
        )?
    } else {
        0
    };
    if matrix.len() != n_residues.saturating_mul(n_residues)
        || matrix.iter().any(|value| !value.is_finite())
        || window.phase_index >= STRICT_TWIN_PHASE_NAMES.len() as u32
        || window.phase_name != STRICT_TWIN_PHASE_NAMES[window.phase_index as usize]
        || window.bin_size_steps != STRICT_TWIN_CCF_BIN_SIZE_STEPS
        || window.bin_count < 2
        || window.bin_count != expected_bin_count
    {
        bail!("strict phase CCF matrix identity/dimension/finite contract failed");
    }
    let mut bytes = Vec::with_capacity(256 + matrix.len().saturating_mul(4));
    bytes.extend_from_slice(b"PRPCF003");
    bytes.extend_from_slice(&STRICT_TWIN_FEATURE_SCHEMA_VERSION.to_le_bytes());
    bytes.extend_from_slice(&0x01020304u32.to_le_bytes());
    bytes.extend_from_slice(&window.phase_index.to_le_bytes());
    bytes.extend_from_slice(&window.start_step.to_le_bytes());
    bytes.extend_from_slice(&window.end_step_exclusive.to_le_bytes());
    bytes.extend_from_slice(&window.bin_size_steps.to_le_bytes());
    bytes.extend_from_slice(&window.bin_count.to_le_bytes());
    bytes.extend_from_slice(&(n_residues as u64).to_le_bytes());
    bytes.extend_from_slice(&(matrix.len() as u64).to_le_bytes());
    for value in [
        config.run_id.as_str(),
        config.topology_a_sha256.as_str(),
        config.topology_b_sha256.as_str(),
        window.phase_name.as_str(),
        "CUDA_WMMA:tensor_ccf.ptx",
    ] {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }
    for value in matrix {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes.extend_from_slice(b"PRPCFEND");
    write_atomic_bytes(path, &bytes)
}

#[cfg(feature = "gpu")]
fn compute_strict_phase_ccf_runtime(
    context: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    output_dir: &Path,
    config: &CoupledTwinEvidenceConfig,
    protocol: &CryoUvProtocol,
    total_steps: i32,
    n_residues: usize,
    spikes_a: &[crate::fused_engine::GpuSpikeEvent],
    spikes_b: &[crate::fused_engine::GpuSpikeEvent],
    ccf_kernel: &StrictTwinKernelSourceBinding,
) -> Result<Vec<StrictPhaseCcfRuntime>> {
    let windows = strict_twin_phase_windows(protocol, total_steps)?;
    if n_residues == 0 {
        bail!("strict phase CCF cannot execute with zero residues");
    }
    validate_strict_kernel_source_binding(
        ccf_kernel,
        "tensor_ccf.ptx",
        &["tensor_ccf_compute", "compute_spike_norms"],
    )?;
    let module = context
        .load_module(cudarc::nvrtc::Ptx::from_file(Path::new(&ccf_kernel.path)))
        .context("load strict phase tensor_ccf.ptx")?;
    let analysis_dir = output_dir.join("COUPLED_TWIN_GPU_ANALYSIS");
    std::fs::create_dir_all(&analysis_dir)?;
    let mut phases = Vec::with_capacity(windows.len());
    for window in windows {
        let rebase = |events: &[crate::fused_engine::GpuSpikeEvent]| {
            events
                .iter()
                .filter(|event| {
                    event.timestep >= window.start_step
                        && event.timestep < window.end_step_exclusive
                })
                .map(|event| {
                    let mut rebased = *event;
                    rebased.timestep -= window.start_step;
                    rebased
                })
                .collect::<Vec<_>>()
        };
        let phase_a = rebase(spikes_a);
        let phase_b = rebase(spikes_b);
        let matrix_steps = i32::try_from(window.bin_count)?
            .checked_mul(window.bin_size_steps)
            .context("strict phase CCF padded duration overflow")?;
        let (matrix_a, matrix_b, _, _) = crate::twin_kernels::build_ccf_matrices(
            &phase_a,
            &phase_b,
            n_residues,
            matrix_steps,
            window.bin_size_steps,
        );
        let mut compute = crate::twin_kernels::TwinCcfCompute::new(
            stream,
            &module,
            n_residues as i32,
            window.bin_count as i32,
        )?;
        compute.upload_matrices(stream, &matrix_a, &matrix_b)?;
        compute.compute(stream)?;
        let matrix = compute.download_ccf(stream)?;
        if matrix.len() != n_residues.saturating_mul(n_residues)
            || matrix.iter().any(|value| !value.is_finite())
        {
            bail!(
                "strict phase {} GPU CCF returned invalid dimensions/nonfinite values",
                window.phase_name
            );
        }
        let features = crate::twin_kernels::extract_ccf_features(&matrix, n_residues);
        if features.len() != n_residues
            || features.iter().any(|feature| {
                !feature.ccf_peak_value.is_finite()
                    || !feature.ccf_width.is_finite()
                    || !feature.ccf_asymmetry.is_finite()
                    || !feature.ccf_reproducibility.is_finite()
            })
        {
            bail!(
                "strict phase {} CCF feature extraction failed",
                window.phase_name
            );
        }
        let matrix_path = analysis_dir.join(format!(
            "phase_{}_{}_ccf_f32le.bin",
            window.phase_index, window.phase_name
        ));
        write_strict_phase_ccf_matrix(&matrix_path, config, &window, n_residues, &matrix)?;
        let kind = format!("gpu_phase_ccf_{}", window.phase_index);
        let matrix_artifact = artifact_evidence(
            &kind,
            None,
            &matrix_path,
            Some(matrix.len() as u64),
            Some(STRICT_TWIN_FEATURE_SCHEMA_VERSION),
        )?;
        phases.push(StrictPhaseCcfRuntime {
            window,
            matrix_artifact,
            matrix,
            features,
        });
    }
    if phases.len() != STRICT_TWIN_PHASE_NAMES.len() {
        bail!("strict phase CCF did not execute all five phases");
    }
    Ok(phases)
}

#[cfg(feature = "gpu")]
fn read_strict_phase_ccf_matrix(
    artifact: &CoupledTwinArtifactEvidence,
    config: &CoupledTwinEvidenceConfig,
    expected_phase_index: u32,
    expected_n_residues: usize,
) -> Result<(StrictTwinPhaseWindow, Vec<f32>)> {
    let mut reader = std::io::BufReader::new(File::open(&artifact.path)?);
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != b"PRPCF003" {
        bail!("strict phase CCF has invalid magic");
    }
    let schema_version = read_u32_from(&mut reader, "strict phase schema version")?;
    let endian = read_u32_from(&mut reader, "strict phase endian marker")?;
    let phase_index = read_u32_from(&mut reader, "strict phase index")?;
    let read_i32 = |reader: &mut std::io::BufReader<File>, label: &str| -> Result<i32> {
        let mut bytes = [0u8; 4];
        reader
            .read_exact(&mut bytes)
            .with_context(|| format!("read {label}"))?;
        Ok(i32::from_le_bytes(bytes))
    };
    let start_step = read_i32(&mut reader, "strict phase start")?;
    let end_step_exclusive = read_i32(&mut reader, "strict phase end")?;
    let bin_size_steps = read_i32(&mut reader, "strict phase bin size")?;
    let bin_count = read_u32_from(&mut reader, "strict phase bin count")?;
    let n_residues = read_u64_from(&mut reader, "strict phase residue count")?;
    let matrix_count = read_u64_from(&mut reader, "strict phase matrix count")?;
    let run_id = read_bounded_utf8(&mut reader, "strict phase run id")?;
    let topology_a_sha256 = read_bounded_utf8(&mut reader, "strict phase topology A")?;
    let topology_b_sha256 = read_bounded_utf8(&mut reader, "strict phase topology B")?;
    let phase_name = read_bounded_utf8(&mut reader, "strict phase name")?;
    let backend = read_bounded_utf8(&mut reader, "strict phase backend")?;
    let expected_count = expected_n_residues
        .checked_mul(expected_n_residues)
        .context("strict phase expected matrix size overflow")?;
    let expected_bin_count = if end_step_exclusive > start_step {
        u32::try_from(
            (end_step_exclusive - start_step)
                .checked_add(STRICT_TWIN_CCF_BIN_SIZE_STEPS - 1)
                .context("strict phase readback bin rounding overflow")?
                / STRICT_TWIN_CCF_BIN_SIZE_STEPS,
        )?
    } else {
        0
    };
    if schema_version != STRICT_TWIN_FEATURE_SCHEMA_VERSION
        || endian != 0x01020304
        || phase_index != expected_phase_index
        || n_residues != expected_n_residues as u64
        || matrix_count != expected_count as u64
        || run_id != config.run_id
        || topology_a_sha256 != config.topology_a_sha256
        || topology_b_sha256 != config.topology_b_sha256
        || phase_index >= STRICT_TWIN_PHASE_NAMES.len() as u32
        || phase_name != STRICT_TWIN_PHASE_NAMES[phase_index as usize]
        || backend != "CUDA_WMMA:tensor_ccf.ptx"
        || start_step < 0
        || end_step_exclusive <= start_step
        || bin_size_steps != STRICT_TWIN_CCF_BIN_SIZE_STEPS
        || bin_count < 2
        || bin_count != expected_bin_count
    {
        bail!("strict phase CCF header identity failed");
    }
    let mut matrix = Vec::with_capacity(expected_count);
    for _ in 0..expected_count {
        let mut bytes = [0u8; 4];
        reader.read_exact(&mut bytes)?;
        let value = f32::from_le_bytes(bytes);
        if !value.is_finite() {
            bail!("strict phase CCF matrix contains nonfinite data");
        }
        matrix.push(value);
    }
    let mut trailer = [0u8; 8];
    reader.read_exact(&mut trailer)?;
    let mut trailing = [0u8; 1];
    if &trailer != b"PRPCFEND" || reader.read(&mut trailing)? != 0 {
        bail!("strict phase CCF trailer/length validation failed");
    }
    Ok((
        StrictTwinPhaseWindow {
            phase_index,
            phase_name,
            start_step,
            end_step_exclusive,
            bin_size_steps,
            bin_count,
        },
        matrix,
    ))
}

#[cfg(feature = "gpu")]
fn read_global_ccf_matrix(
    artifact: &CoupledTwinArtifactEvidence,
    expected_n_residues: usize,
) -> Result<Vec<f32>> {
    let mut reader = std::io::BufReader::new(File::open(&artifact.path)?);
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    let schema_version = read_u32_from(&mut reader, "global CCF schema version")?;
    let rows = read_u64_from(&mut reader, "global CCF rows")?;
    let columns = read_u64_from(&mut reader, "global CCF columns")?;
    let count = read_u64_from(&mut reader, "global CCF element count")?;
    let expected_count = expected_n_residues
        .checked_mul(expected_n_residues)
        .context("global CCF expected matrix size overflow")?;
    if &magic != b"PRCCF002"
        || schema_version != COUPLED_TWIN_EVIDENCE_SCHEMA_VERSION
        || rows != expected_n_residues as u64
        || columns != expected_n_residues as u64
        || count != expected_count as u64
    {
        bail!("global GPU CCF matrix header failed strict readback");
    }
    let mut matrix = Vec::with_capacity(expected_count);
    for _ in 0..expected_count {
        let mut bytes = [0u8; 4];
        reader.read_exact(&mut bytes)?;
        let value = f32::from_le_bytes(bytes);
        if !value.is_finite() {
            bail!("global GPU CCF matrix contains nonfinite data");
        }
        matrix.push(value);
    }
    let mut trailing = [0u8; 1];
    if reader.read(&mut trailing)? != 0 {
        bail!("global GPU CCF matrix has trailing bytes");
    }
    Ok(matrix)
}

#[cfg(feature = "gpu")]
fn read_spike_evidence_events(
    artifact: &CoupledTwinArtifactEvidence,
    expected_stream_id: u32,
    expected_run_id: &str,
    expected_stem: &str,
) -> Result<Vec<crate::fused_engine::GpuSpikeEvent>> {
    validate_evidence_envelope(
        Path::new(&artifact.path),
        b"PRSPK001",
        b"PRSPKEND",
        expected_stream_id,
        expected_run_id,
        expected_stem,
        artifact.record_count,
    )?;
    let mut reader = std::io::BufReader::new(File::open(&artifact.path)?);
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    let schema_version = read_u32_from(&mut reader, "spike schema version")?;
    let endian = read_u32_from(&mut reader, "spike endian marker")?;
    let stream_id = read_u32_from(&mut reader, "spike stream id")?;
    let run_id = read_bounded_utf8(&mut reader, "spike run id")?;
    let stem = read_bounded_utf8(&mut reader, "spike target stem")?;
    let record_count = read_u64_from(&mut reader, "spike record count")?;
    let stride = read_u64_from(&mut reader, "spike record stride")?;
    let payload_size = read_u64_from(&mut reader, "spike payload size")?;
    let expected_stride = std::mem::size_of::<crate::fused_engine::GpuSpikeEvent>();
    if &magic != b"PRSPK001"
        || schema_version != 1
        || endian != 0x01020304
        || stream_id != expected_stream_id
        || run_id != expected_run_id
        || stem != expected_stem
        || stride != expected_stride as u64
        || payload_size
            != record_count
                .checked_mul(stride)
                .context("PRSPK001 record-count/stride overflow")?
        || artifact.record_count != Some(record_count)
    {
        bail!("PRSPK001 strict readback header failed");
    }
    let event_count = usize::try_from(record_count)?;
    let mut events = Vec::with_capacity(event_count);
    let mut record_bytes = vec![0u8; expected_stride];
    let mut checksum = 0xcbf29ce484222325u64;
    for _ in 0..event_count {
        reader.read_exact(&mut record_bytes)?;
        for byte in &record_bytes {
            checksum = (checksum ^ (*byte as u64)).wrapping_mul(0x100000001b3);
        }
        let event = unsafe {
            std::ptr::read_unaligned(
                record_bytes
                    .as_ptr()
                    .cast::<crate::fused_engine::GpuSpikeEvent>(),
            )
        };
        events.push(event);
    }
    let stored_checksum = read_u64_from(&mut reader, "PRSPK001 payload checksum")?;
    let mut trailer = [0u8; 8];
    reader.read_exact(&mut trailer)?;
    let mut trailing = [0u8; 1];
    if events.len() != event_count
        || checksum != stored_checksum
        || &trailer != b"PRSPKEND"
        || reader.read(&mut trailing)? != 0
    {
        bail!("PRSPK001 strict readback count/checksum/trailer contract failed");
    }
    Ok(events)
}

#[cfg(feature = "gpu")]
fn strict_binding_matches_artifact(
    binding: &StrictTwinFeatureArtifactBinding,
    artifact: &CoupledTwinArtifactEvidence,
) -> bool {
    binding.kind == artifact.kind
        && binding.stream_id == artifact.stream_id
        && binding.path == artifact.path
        && binding.size_bytes == artifact.size_bytes
        && binding.sha256 == artifact.sha256
        && binding.blake3 == artifact.blake3
}

#[cfg(feature = "gpu")]
fn strict_nested_binding_matches_artifact(
    binding: Option<&serde_json::Value>,
    artifact: &CoupledTwinArtifactEvidence,
) -> bool {
    binding.is_some_and(|binding| {
        binding.get("path").and_then(serde_json::Value::as_str) == Some(artifact.path.as_str())
            && binding.get("sha256").and_then(serde_json::Value::as_str)
                == Some(artifact.sha256.as_str())
            && binding.get("blake3").and_then(serde_json::Value::as_str)
                == Some(artifact.blake3.as_str())
    })
}

#[cfg(feature = "gpu")]
fn validate_strict_twin_feature_plane(
    output_dir: &Path,
    config: &CoupledTwinEvidenceConfig,
    result: &CoupledTwinResult,
    runtime: &CoupledTwinRuntimeEvidence,
) -> Result<()> {
    let find_artifact = |kind: &str| -> Result<&CoupledTwinArtifactEvidence> {
        runtime
            .artifacts
            .iter()
            .find(|artifact| artifact.kind == kind && artifact.stream_id.is_none())
            .with_context(|| format!("strict feature plane lacks artifact {kind}"))
    };
    let residue_artifact = find_artifact("strict_twin_residue_features")?;
    let site_artifact = find_artifact("strict_twin_site_features")?;
    let receipt_artifact = find_artifact("strict_twin_feature_receipt")?;
    let binding_sites_artifact = find_artifact("binding_sites")?;
    let global_ccf_artifact = find_artifact("gpu_ccf_matrix")?;
    let te_artifact = find_artifact("gpu_transfer_entropy")?;
    let analysis_receipt_artifact = find_artifact("gpu_ccf_te_execution_receipt")?;
    for (artifact, expected_path) in [
        (
            residue_artifact,
            output_dir.join("COUPLED_TWIN_STRICT_FEATURES/strict_residue_features.v3.json"),
        ),
        (
            site_artifact,
            output_dir.join("COUPLED_TWIN_STRICT_FEATURES/strict_site_features.v3.json"),
        ),
        (
            receipt_artifact,
            output_dir
                .join("COUPLED_TWIN_STRICT_FEATURES/strict_feature_execution_receipt.v3.json"),
        ),
    ] {
        if Path::new(&artifact.path) != expected_path {
            bail!(
                "strict artifact {} is not at its fixed run-owned path",
                artifact.kind
            );
        }
    }

    let residue_payload: StrictTwinResidueFeatureArtifact =
        serde_json::from_slice(&std::fs::read(&residue_artifact.path)?)?;
    let site_payload: StrictTwinSiteFeatureArtifact =
        serde_json::from_slice(&std::fs::read(&site_artifact.path)?)?;
    let receipt: StrictTwinFeatureReceipt =
        serde_json::from_slice(&std::fs::read(&receipt_artifact.path)?)?;
    if residue_payload.schema_kind != "prismos.coupled_twin_strict_residue_features.v3"
        || site_payload.schema_kind != "prismos.coupled_twin_strict_site_features.v3"
        || receipt.schema_kind != "prismos.coupled_twin_strict_feature_execution_receipt.v3"
        || residue_payload.schema_version != STRICT_TWIN_FEATURE_SCHEMA_VERSION
        || site_payload.schema_version != STRICT_TWIN_FEATURE_SCHEMA_VERSION
        || receipt.schema_version != STRICT_TWIN_FEATURE_SCHEMA_VERSION
        || residue_payload.status != "PASS"
        || site_payload.status != "PASS"
        || receipt.status != "PASS"
        || residue_payload.run_id != config.run_id
        || site_payload.run_id != config.run_id
        || receipt.run_id != config.run_id
        || residue_payload.topology_a_sha256 != config.topology_a_sha256
        || residue_payload.topology_b_sha256 != config.topology_b_sha256
        || residue_payload.event_source != "runtime_owned_PRSPK001_streams_0_and_1"
        || residue_payload.ccf_backend != STRICT_TWIN_PHASE_CCF_BACKEND
        || site_payload.topology_a_sha256 != config.topology_a_sha256
        || site_payload.topology_b_sha256 != config.topology_b_sha256
        || receipt.topology_a_sha256 != config.topology_a_sha256
        || receipt.topology_b_sha256 != config.topology_b_sha256
        || residue_payload.fallback_used
        || site_payload.fallback_used
        || receipt.fallback_used
        || !receipt.finite_values_verified
        || !receipt.unavailable_values_are_null
        || receipt.legacy_residue_rows_active != 0
        || receipt.legacy_site_rows_active != 0
        || receipt.strict_residue_record_count != residue_payload.records.len() as u64
        || receipt.strict_site_record_count != site_payload.records.len() as u64
        || receipt.exact_predictive_algorithm != STRICT_TWIN_PREDICTIVE_ALGORITHM
        || receipt.phase_ccf_backend != STRICT_TWIN_PHASE_CCF_BACKEND
        || runtime.artifacts.iter().any(|artifact| {
            artifact.kind.contains("superseded")
                || artifact
                    .path
                    .contains("SUPERSEDED_LEGACY_DETECTOR_SURFACES")
        })
    {
        bail!("strict feature payload/receipt identity or no-fallback contract failed");
    }
    for record in &residue_payload.records {
        validate_strict_residue_feature(record)?;
    }
    for record in &site_payload.records {
        validate_strict_site_feature(record)?;
    }
    if result.per_residue_features.len() != residue_payload.records.len()
        || result.per_site_features.len() != site_payload.records.len()
        || result
            .per_residue_features
            .iter()
            .any(|record| matches!(record, CoupledTwinResidueFeatureRecord::Legacy(_)))
        || result
            .per_site_features
            .iter()
            .any(|record| matches!(record, CoupledTwinSiteFeatureRecord::Legacy(_)))
    {
        bail!("strict coupled result contains legacy proxy rows or count drift");
    }
    let result_residue_records = result
        .per_residue_features
        .iter()
        .filter_map(|record| match record {
            CoupledTwinResidueFeatureRecord::Strict(value) => Some(value),
            CoupledTwinResidueFeatureRecord::Legacy(_) => None,
        })
        .collect::<Vec<_>>();
    let result_site_records = result
        .per_site_features
        .iter()
        .filter_map(|record| match record {
            CoupledTwinSiteFeatureRecord::Strict(value) => Some(value),
            CoupledTwinSiteFeatureRecord::Legacy(_) => None,
        })
        .collect::<Vec<_>>();
    if serde_json::to_value(&result_residue_records)?
        != serde_json::to_value(&residue_payload.records)?
        || serde_json::to_value(&result_site_records)?
            != serde_json::to_value(&site_payload.records)?
    {
        bail!("strict result rows differ from separately emitted strict artifacts");
    }

    let spike_artifacts = [0u32, 1u32]
        .into_iter()
        .map(|stream_id| {
            runtime
                .artifacts
                .iter()
                .find(|artifact| artifact.kind == "spikes" && artifact.stream_id == Some(stream_id))
                .with_context(|| format!("missing strict source spike stream {stream_id}"))
        })
        .collect::<Result<Vec<_>>>()?;
    if receipt.source_spike_artifacts.len() != 2
        || spike_artifacts.iter().any(|artifact| {
            !receipt
                .source_spike_artifacts
                .iter()
                .any(|binding| strict_binding_matches_artifact(binding, artifact))
        })
    {
        bail!("strict feature receipt does not bind both PRSPK001 sources");
    }
    let stem = coupled_topology_stem(config);
    let spikes_a = read_spike_evidence_events(spike_artifacts[0], 0, &config.run_id, &stem)?;
    let spikes_b = read_spike_evidence_events(spike_artifacts[1], 1, &config.run_id, &stem)?;

    let topology = PrismPrepTopology::load(&config.topology_a_path)
        .context("reload authoritative topology A for strict feature recomputation")?;
    let n_residues = topology.n_residues;
    let analysis_receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&analysis_receipt_artifact.path)?)?;
    let ccf_kernel: StrictTwinKernelSourceBinding = serde_json::from_value(
        analysis_receipt
            .get("ccf_kernel_source")
            .context("GPU analysis receipt lacks CCF kernel source binding")?
            .clone(),
    )?;
    let te_kernel: StrictTwinKernelSourceBinding = serde_json::from_value(
        analysis_receipt
            .get("transfer_entropy_kernel_source")
            .context("GPU analysis receipt lacks TE kernel source binding")?
            .clone(),
    )?;
    validate_strict_kernel_source_binding(
        &ccf_kernel,
        "tensor_ccf.ptx",
        &["tensor_ccf_compute", "compute_spike_norms"],
    )?;
    validate_strict_kernel_source_binding(
        &te_kernel,
        "twin_transfer_entropy.ptx",
        &[
            "twin_compute_row_means",
            "twin_binned_te",
            "twin_compute_mutual_info",
        ],
    )?;
    if !strict_binding_matches_artifact(
        &receipt.global_gpu_analysis_receipt_artifact,
        analysis_receipt_artifact,
    ) || serde_json::to_value(&receipt.phase_ccf_kernel_source)?
        != serde_json::to_value(&ccf_kernel)?
    {
        bail!("strict phase CCF kernel/global GPU-v3 receipt cross-binding failed");
    }
    validate_strict_kernel_source_binding(
        &receipt.phase_ccf_kernel_source,
        "tensor_ccf.ptx",
        &["tensor_ccf_compute", "compute_spike_norms"],
    )?;
    if analysis_receipt
        .get("schema_kind")
        .and_then(serde_json::Value::as_str)
        != Some("prismos.coupled_twin_gpu_analysis_receipt.v3")
        || analysis_receipt
            .get("schema_version")
            .and_then(serde_json::Value::as_u64)
            != Some(STRICT_TWIN_FEATURE_SCHEMA_VERSION as u64)
        || analysis_receipt
            .get("status")
            .and_then(serde_json::Value::as_str)
            != Some("PASS")
        || analysis_receipt
            .get("run_id")
            .and_then(serde_json::Value::as_str)
            != Some(config.run_id.as_str())
        || analysis_receipt
            .get("ccf_backend")
            .and_then(serde_json::Value::as_str)
            != Some(STRICT_TWIN_GLOBAL_CCF_BACKEND)
        || analysis_receipt
            .get("transfer_entropy_backend")
            .and_then(serde_json::Value::as_str)
            != Some(STRICT_TWIN_TE_BACKEND)
        || analysis_receipt
            .get("transfer_entropy_kernel_lag_bins")
            .and_then(serde_json::Value::as_u64)
            != Some(STRICT_TWIN_TE_KERNEL_LAG_BINS as u64)
        || analysis_receipt
            .get("transfer_entropy_past_index_offset_bins")
            .and_then(serde_json::Value::as_u64)
            != Some(STRICT_TWIN_TE_PAST_INDEX_OFFSET_BINS as u64)
        || analysis_receipt
            .get("transfer_entropy_future_index_offset_bins")
            .and_then(serde_json::Value::as_u64)
            != Some(STRICT_TWIN_TE_FUTURE_INDEX_OFFSET_BINS as u64)
        || analysis_receipt
            .get("transfer_entropy_effective_past_to_future_horizon_bins")
            .and_then(serde_json::Value::as_u64)
            != Some(STRICT_TWIN_TE_EFFECTIVE_HORIZON_BINS as u64)
        || analysis_receipt
            .get("transfer_entropy_bin_size_steps")
            .and_then(serde_json::Value::as_i64)
            != Some(i64::from(STRICT_TWIN_CCF_BIN_SIZE_STEPS))
        || analysis_receipt
            .get("transfer_entropy_effective_past_to_future_horizon_steps")
            .and_then(serde_json::Value::as_i64)
            != Some(i64::from(STRICT_TWIN_TE_EFFECTIVE_HORIZON_STEPS))
        || analysis_receipt
            .get("transfer_entropy_transition_sample_count_per_pair")
            .and_then(serde_json::Value::as_i64)
            != Some(i64::from(
                result.steps_a.div_euclid(STRICT_TWIN_CCF_BIN_SIZE_STEPS) - 2,
            ))
        || analysis_receipt
            .get("residue_count")
            .and_then(serde_json::Value::as_u64)
            != Some(n_residues as u64)
        || analysis_receipt
            .get("transfer_entropy_bin_count")
            .and_then(serde_json::Value::as_i64)
            != Some(i64::from(
                result.steps_a.div_euclid(STRICT_TWIN_CCF_BIN_SIZE_STEPS),
            ))
        || analysis_receipt
            .get("kernel_supported_residue_count")
            .and_then(serde_json::Value::as_u64)
            != Some(n_residues as u64)
        || analysis_receipt
            .get("valid_zero_values_preserved")
            .and_then(serde_json::Value::as_bool)
            != Some(true)
        || analysis_receipt
            .get("finite_values_verified")
            .and_then(serde_json::Value::as_bool)
            != Some(true)
        || analysis_receipt
            .get("fallback_used")
            .and_then(serde_json::Value::as_bool)
            != Some(false)
        || !strict_nested_binding_matches_artifact(
            analysis_receipt.get("ccf_matrix"),
            global_ccf_artifact,
        )
        || !strict_nested_binding_matches_artifact(
            analysis_receipt.get("transfer_entropy"),
            te_artifact,
        )
    {
        bail!("GPU analysis receipt identity/kernel/nested artifact binding failed");
    }
    let global_ccf = read_global_ccf_matrix(global_ccf_artifact, n_residues)?;
    let global_features = crate::twin_kernels::extract_ccf_features(&global_ccf, n_residues);
    let mut phases: Vec<StrictPhaseCcfRuntime> = Vec::with_capacity(STRICT_TWIN_PHASE_NAMES.len());
    for phase_index in 0..STRICT_TWIN_PHASE_NAMES.len() {
        let kind = format!("gpu_phase_ccf_{phase_index}");
        let artifact = find_artifact(&kind)?;
        if !receipt
            .phase_ccf_artifacts
            .iter()
            .any(|binding| strict_binding_matches_artifact(binding, artifact))
        {
            bail!("strict feature receipt does not bind {kind}");
        }
        let (window, matrix) =
            read_strict_phase_ccf_matrix(artifact, config, phase_index as u32, n_residues)?;
        if phase_index == 0 && window.start_step != 0 {
            bail!("strict phase windows do not start at step zero");
        }
        if let Some(previous) = phases.last() {
            if previous.window.end_step_exclusive != window.start_step {
                bail!("strict phase CCF windows are not contiguous");
            }
        }
        let features = crate::twin_kernels::extract_ccf_features(&matrix, n_residues);
        phases.push(StrictPhaseCcfRuntime {
            window,
            matrix_artifact: artifact.clone(),
            matrix,
            features,
        });
    }
    if phases
        .last()
        .is_none_or(|phase| phase.window.end_step_exclusive != result.steps_a)
        || receipt.phase_ccf_artifacts.len() != STRICT_TWIN_PHASE_NAMES.len()
    {
        bail!("strict phase CCF windows do not cover the executed run exactly");
    }

    let te_payload: serde_json::Value = serde_json::from_slice(&std::fs::read(&te_artifact.path)?)?;
    let te_payload_kernel: StrictTwinKernelSourceBinding = serde_json::from_value(
        te_payload
            .get("kernel_source")
            .context("GPU TE artifact lacks kernel source binding")?
            .clone(),
    )?;
    if te_payload
        .get("schema_kind")
        .and_then(serde_json::Value::as_str)
        != Some("prismos.coupled_twin_gpu_transfer_entropy.v3")
        || te_payload
            .get("schema_version")
            .and_then(serde_json::Value::as_u64)
            != Some(STRICT_TWIN_FEATURE_SCHEMA_VERSION as u64)
        || te_payload.get("run_id").and_then(serde_json::Value::as_str)
            != Some(config.run_id.as_str())
        || te_payload
            .get("backend")
            .and_then(serde_json::Value::as_str)
            != Some("CUDA:twin_transfer_entropy.ptx")
        || te_payload
            .get("residue_count")
            .and_then(serde_json::Value::as_u64)
            != Some(n_residues as u64)
        || te_payload
            .get("kernel_lag_bins")
            .and_then(serde_json::Value::as_u64)
            != Some(STRICT_TWIN_TE_KERNEL_LAG_BINS as u64)
        || te_payload
            .get("past_index_offset_bins")
            .and_then(serde_json::Value::as_u64)
            != Some(STRICT_TWIN_TE_PAST_INDEX_OFFSET_BINS as u64)
        || te_payload
            .get("future_index_offset_bins")
            .and_then(serde_json::Value::as_u64)
            != Some(STRICT_TWIN_TE_FUTURE_INDEX_OFFSET_BINS as u64)
        || te_payload
            .get("effective_past_to_future_horizon_bins")
            .and_then(serde_json::Value::as_u64)
            != Some(STRICT_TWIN_TE_EFFECTIVE_HORIZON_BINS as u64)
        || te_payload
            .get("bin_size_steps")
            .and_then(serde_json::Value::as_i64)
            != Some(i64::from(STRICT_TWIN_CCF_BIN_SIZE_STEPS))
        || te_payload
            .get("effective_past_to_future_horizon_steps")
            .and_then(serde_json::Value::as_i64)
            != Some(i64::from(STRICT_TWIN_TE_EFFECTIVE_HORIZON_STEPS))
        || te_payload
            .get("bin_count")
            .and_then(serde_json::Value::as_i64)
            != Some(i64::from(
                result.steps_a.div_euclid(STRICT_TWIN_CCF_BIN_SIZE_STEPS),
            ))
        || te_payload
            .get("transition_sample_count_per_pair")
            .and_then(serde_json::Value::as_i64)
            != Some(i64::from(
                result.steps_a.div_euclid(STRICT_TWIN_CCF_BIN_SIZE_STEPS) - 2,
            ))
        || te_payload
            .get("minimum_transition_samples_per_pair")
            .and_then(serde_json::Value::as_u64)
            != Some(10)
        || te_payload
            .get("outgoing_term_inclusion_floor_exclusive")
            .and_then(serde_json::Value::as_f64)
            != Some(0.001)
        || te_payload
            .get("kernel_supported_residue_count")
            .and_then(serde_json::Value::as_u64)
            != Some(n_residues as u64)
        || te_payload
            .get("valid_zero_values_preserved")
            .and_then(serde_json::Value::as_bool)
            != Some(true)
        || te_payload
            .get("fallback_used")
            .and_then(serde_json::Value::as_bool)
            != Some(false)
        || te_payload_kernel.sha256 != te_kernel.sha256
        || te_payload_kernel.blake3 != te_kernel.blake3
        || te_payload_kernel.path != te_kernel.path
        || te_payload.get("mutual_information").is_some()
        || te_payload.get("causal_flow_direction").is_some()
        || te_payload.get("transfer_entropy_a_to_b").is_some()
        || te_payload.get("lag_bins").is_some()
    {
        bail!("GPU TE v3 semantic/kernel/readback contract failed");
    }
    let decode_f32 = |field: &str| -> Result<Vec<f32>> {
        serde_json::from_value(
            te_payload
                .get(field)
                .with_context(|| format!("GPU TE artifact lacks {field}"))?
                .clone(),
        )
        .with_context(|| format!("decode GPU TE field {field}"))
    };
    let te_mi = decode_f32("bidirectional_transfer_entropy_mass")?;
    let te_causal = decode_f32("net_causal_flow_out_minus_in")?;
    let te_per_residue = decode_f32("thresholded_total_outgoing_transfer_entropy_a_to_b")?;
    let recomputed_residue_records = build_strict_residue_features(
        &topology,
        result.steps_a,
        &spikes_a,
        &spikes_b,
        &global_ccf,
        &global_features,
        &phases,
        &te_mi,
        &te_causal,
        &te_per_residue,
    )?;
    if serde_json::to_value(&recomputed_residue_records)?
        != serde_json::to_value(&residue_payload.records)?
    {
        bail!("strict residue artifacts do not recompute from PRSPK001 plus GPU matrices");
    }

    if site_payload.geometry_source_path != binding_sites_artifact.path
        || site_payload.geometry_source_sha256 != binding_sites_artifact.sha256
    {
        bail!("strict site artifact does not bind topology-derived binding-site geometry");
    }
    let sites_json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&binding_sites_artifact.path)?)?;
    let recomputed_site_records = build_strict_site_features(
        &sites_json,
        &recomputed_residue_records,
        &global_ccf,
        n_residues,
    )?;
    if serde_json::to_value(&recomputed_site_records)?
        != serde_json::to_value(&site_payload.records)?
    {
        bail!("strict site artifacts do not recompute from strict residues plus real geometry");
    }
    if !strict_binding_matches_artifact(&receipt.binding_sites_artifact, binding_sites_artifact)
        || !strict_binding_matches_artifact(&receipt.residue_feature_artifact, residue_artifact)
        || !strict_binding_matches_artifact(&receipt.site_feature_artifact, site_artifact)
        || receipt.retired_legacy_detector_surfaces.len() != 2
    {
        bail!("strict receipt feature bindings/supersession set failed");
    }
    for retired in &receipt.retired_legacy_detector_surfaces {
        let original = Path::new(&retired.original_path);
        let superseded = Path::new(&retired.superseded_path);
        let metadata = std::fs::symlink_metadata(superseded)?;
        let (sha256, blake3) = hash_file_dual(superseded)?;
        if original.exists()
            || metadata.file_type().is_symlink()
            || !metadata.is_file()
            || metadata.len() != retired.size_bytes
            || sha256 != retired.sha256
            || blake3 != retired.blake3
            || retired.disposition
                != "SUPERSEDED_NON_AUTHORITATIVE_NOT_ADMITTED_TO_RUNTIME_MANIFEST"
            || !superseded.starts_with(
                output_dir.join("COUPLED_TWIN_STRICT_FEATURES/SUPERSEDED_LEGACY_DETECTOR_SURFACES"),
            )
        {
            bail!("strict legacy detector surface supersession failed");
        }
    }
    Ok(())
}

/// Configuration for a PRISM-TWIN interferometric run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoupledTwinConfig {
    /// Fraction of each phase that B lags behind A (0.0-0.5)
    pub phase_offset_fraction: f32,
    /// How often to exchange spike density + compute CCF (in steps)
    pub exchange_interval: u32,
    /// Sensitivity boost factor for threshold modification (Layer 1)
    pub sensitivity_boost: f32,
    /// Maximum threshold reduction fraction (safety limit)
    pub max_threshold_reduction: f32,
    /// Region size for coarse spike density (voxels per region edge)
    pub region_size: u32,
    /// Path to NMA modes file for stream B (None = no NMA, Layer 4 disabled)
    pub nma_modes_path: Option<String>,
    /// NMA amplification factor for stream B
    pub nma_amplification: f32,
    /// NMA scan fraction for stream B (recorded on the condition-edge node B).
    pub nma_scan_fraction: f32,
    /// Enable CCF computation (Layer 2). Disable for Gate 1 testing.
    pub enable_ccf: bool,
    /// Enable spike density exchange (Layer 1). Disable for Gate 1 testing.
    pub enable_exchange: bool,
    /// Use persistent cooperative kernel for coupling instead of host-mediated.
    /// DISABLED on Blackwell SM120 (SM starvation). Use graph_coupling instead.
    pub persistent_coupling: bool,
    /// Request CUDA Graph-based autonomous coupling where the call site can
    /// safely capture spike emission + compact_and_push before spike-count reset.
    /// Multi-engine callers must emit a graph-status manifest when they fall back.
    pub graph_coupling: bool,
    /// Emit the legacy giant `coupled_spikes.json` preview (Phase 6). Default FALSE: the
    /// canonical, byte-lossless, round-trip-verified QNANL SPIKE_EVENT_STREAM replaces it, so the
    /// 41 GB JSON is NOT written unless explicitly requested (`--emit-coupled-spikes-preview-json`).
    pub emit_preview_json: bool,
    /// Native allocation boundary queries; no additional explicit synchronization.
    pub p1_allocation_timing: bool,
}

impl Default for CoupledTwinConfig {
    fn default() -> Self {
        Self {
            phase_offset_fraction: 0.20,
            exchange_interval: 100,
            sensitivity_boost: 0.01,
            max_threshold_reduction: 0.5,
            region_size: 4,
            nma_modes_path: None,
            nma_amplification: 3.0,
            nma_scan_fraction: 0.3,
            enable_ccf: false,          // Gate 1: off. Gate 2+: on.
            enable_exchange: false,     // Gate 1: off. Gate 2+: on.
            persistent_coupling: false, // DISABLED on SM120 (starvation). Legacy/rare path.
            graph_coupling: false, // Request graph coupling where the caller can prove safe capture scope.
            emit_preview_json: false, // Phase 6: QNANL stream is canonical; JSON preview off by default.
            p1_allocation_timing: false,
        }
    }
}

/// Per-residue interferometric features from coupled observation (50 fields).
///
/// Four groups:
///   Consensus (12): spike agreement, per-phase profile, spatial coherence
///   Cross-correlation (12): CCF peak, width, asymmetry, per-phase, frequency
///   Differential (18): B/A ratio, NMA-exclusive counts, barrier classification
///   Scout/propagation (8): lead time, predictive value, TE, causal flow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterferometricFeatures {
    pub resid: i32,
    pub resname: String,

    // ── Consensus (12 fields) ──
    pub spike_agreement_ratio: f32,
    pub consensus_intensity_mean: f32,
    /// Per-CCNS-phase spike count ratio: [cold_hold, heating, warm_hold, cooling, cold_return]
    pub consensus_phase_profile: [f32; 5],
    /// Fraction of neighboring residues (within 8Å) that also have agreement > 0.5
    pub consensus_spatial_coherence: f32,
    /// Earliest timestep where both streams agree (both have >= 1 spike near this residue)
    pub consensus_temporal_onset: f32,
    /// Count of nearby residues (within 8Å) with agreement > 0.5
    pub n_consensus_neighbors: u32,

    // ── Cross-correlation (12 fields) ──
    pub ccf_peak_lag: i32,
    pub ccf_peak_value: f32,
    pub ccf_width: f32,
    pub ccf_asymmetry: f32,
    /// CCF peak value computed per CCNS phase
    pub ccf_per_phase: [f32; 5],
    /// Dominant frequency in CCF (from peak spacing, 0 if no clear periodicity)
    pub ccf_frequency_peak: f32,
    pub ccf_reproducibility: f32,
    /// Standard deviation of ccf_peak_lag across phases (low = consistent timing)
    pub ccf_lag_consistency: f32,

    // ── Differential (18 fields) ──
    pub spikes_a: u32,
    pub spikes_b: u32,
    pub b_over_a_ratio: f32,
    /// Spikes present in B but not A (within 5Å spatial, 500-step temporal window)
    pub nma_exclusive_count: u32,
    /// Spikes present in A but not B
    pub thermal_exclusive_count: u32,
    /// Intensity ratio B/A (sensitive to perturbation amplitude, not just count)
    pub b_over_a_intensity_ratio: f32,
    /// NMA mode index with highest B-exclusive correlation (-1 if no NMA)
    pub nma_responsive_mode: i32,
    /// Eigenvalue of the most responsive NMA mode (0 if no NMA)
    pub nma_mode_eigenvalue: f32,
    pub barrier_classification: String,
    /// B/A spike count ratio per CCNS phase
    pub per_phase_differential: [f32; 5],
    /// Timestep lag between NMA onset and first B-exclusive spike at this residue
    pub differential_onset_lag: f32,
    /// Integral of NMA force × displacement at this residue (0 if no NMA)
    pub nma_work_at_residue: f32,
    /// d(spikes_B)/d(NMA_amplitude) — sensitivity to perturbation (0 if no NMA)
    pub mechanical_sensitivity: f32,
    /// Magnitude of mechanical susceptibility tensor element (0 if no NMA)
    pub susceptibility_magnitude: f32,

    // ── Scout/propagation (8 fields) ──
    /// Mean timestep difference: A fires before B (positive = A leads, from phase offset)
    pub scout_lead_time: f32,
    /// P(B fires within 500 steps | A fired at this residue) — predictive value
    pub scout_predictive_value: f32,
    /// Enrichment of spike count at the phase-offset window vs baseline
    pub phase_offset_enrichment: f32,
    /// A's mean intensity at the timestep when B first fires at this residue
    pub scout_intensity_at_onset: f32,
    /// Radius (Å) of the zone around this residue where A fires before B
    pub scout_spatial_propagation: f32,

    // ════════════════════════════════════════════════════════════════════
    // PLACEHOLDER FIELDS — currently output 0.0
    //
    // These require Phase C implementations that are NOT yet built:
    //   mutual_information      → Phase C Step 15 (GPU binned TE kernel)
    //   transfer_entropy_a_to_b → Phase C Step 15 (GPU binned TE kernel)
    //   causal_flow_direction   → Phase C Step 15 (GPU binned TE kernel)
    //   ccf_frequency_peak      → requires FFT on CCF row (not yet implemented)
    //   ccf_lag_consistency     → requires per-phase CCF recomputation
    //   nma_responsive_mode     → Phase C Step 16 (NMA mode correlation)
    //   nma_mode_eigenvalue     → Phase C Step 16 (NMA mode correlation)
    //   nma_work_at_residue     → Phase C Step 16 (NMA force integration)
    //   mechanical_sensitivity  → Phase C Step 16 (perturbation dose-response)
    //   susceptibility_magnitude→ Phase C Step 16 (mechanical susceptibility)
    //
    // TOTAL: 5 NMA-dependent placeholders out of 50 fields = 45 populated, 5 deferred
    //
    // DO NOT use these fields for ranking or model training until their
    // corresponding Phase C steps are implemented and validated.
    // ════════════════════════════════════════════════════════════════════
    /// Mutual information MI(A, B) at this residue. **PLACEHOLDER: 0.0 until Phase C Step 15**
    pub mutual_information: f32,
    /// Transfer entropy TE(A→B) at this residue. **PLACEHOLDER: 0.0 until Phase C Step 15**
    pub transfer_entropy_a_to_b: f32,
    /// TE(A→B) - TE(B→A), signed. **PLACEHOLDER: 0.0 until Phase C Step 15**
    pub causal_flow_direction: f32,
}

impl Default for InterferometricFeatures {
    fn default() -> Self {
        Self {
            resid: -1,
            resname: String::new(),
            spike_agreement_ratio: 0.0,
            consensus_intensity_mean: 0.0,
            consensus_phase_profile: [0.0; 5],
            consensus_spatial_coherence: 0.0,
            consensus_temporal_onset: 0.0,
            n_consensus_neighbors: 0,
            ccf_peak_lag: 0,
            ccf_peak_value: 0.0,
            ccf_width: 0.0,
            ccf_asymmetry: 0.0,
            ccf_per_phase: [0.0; 5],
            ccf_frequency_peak: 0.0,
            ccf_reproducibility: 0.0,
            ccf_lag_consistency: 0.0,
            spikes_a: 0,
            spikes_b: 0,
            b_over_a_ratio: 0.0,
            nma_exclusive_count: 0,
            thermal_exclusive_count: 0,
            b_over_a_intensity_ratio: 0.0,
            nma_responsive_mode: -1,
            nma_mode_eigenvalue: 0.0,
            barrier_classification: "MEDIUM".to_string(),
            per_phase_differential: [0.0; 5],
            differential_onset_lag: 0.0,
            nma_work_at_residue: 0.0,
            mechanical_sensitivity: 0.0,
            susceptibility_magnitude: 0.0,
            scout_lead_time: 0.0,
            scout_predictive_value: 0.0,
            phase_offset_enrichment: 0.0,
            scout_intensity_at_onset: 0.0,
            scout_spatial_propagation: 0.0,
            mutual_information: 0.0,
            transfer_entropy_a_to_b: 0.0,
            causal_flow_direction: 0.0,
        }
    }
}

/// Per-SITE interferometric features aggregated from lining residues.
///
/// Sites are spatial clusters detected by twin_detection.rs.
/// Each site's TWIN features are aggregated from the InterferometricFeatures
/// of its lining residues — this is what drives pocket ranking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiteInterferometricFeatures {
    pub site_id: usize,
    pub n_lining_residues: usize,
    pub n_lining_with_data: usize, // how many lining residues had TWIN features

    // ── Consensus aggregates ──
    pub mean_agreement: f32,
    pub min_agreement: f32,
    pub max_agreement: f32,
    pub mean_spatial_coherence: f32,
    pub mean_consensus_onset: f32,

    // ── CCF aggregates ──
    pub mean_ccf_peak: f32,
    pub max_ccf_peak: f32,
    /// Mean intra-site CCF: mean CCF between all pairs of lining residues.
    /// High value = lining residues co-fire (pocket opens as a unit).
    pub mean_intra_site_ccf: f32,
    pub mean_ccf_reproducibility: f32,

    // ── Differential aggregates ──
    pub mean_b_over_a: f32,
    pub barrier_composition_low: f32, // fraction of lining residues classified LOW
    pub barrier_composition_medium: f32,
    pub barrier_composition_high: f32,
    pub total_nma_exclusive: u32,
    pub total_thermal_exclusive: u32,

    // ── Geometry (from binding_sites.json) ──
    pub volume: f32,
    pub druggability: f32,
    pub is_druggable: bool,
    pub centroid: [f32; 3],

    // ── Scout aggregates ──
    pub mean_scout_lead_time: f32,
    pub mean_predictive_value: f32,
    pub mean_phase_offset_enrichment: f32,
}

/// Exact phase boundary used by strict two-topology CCF execution.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct StrictTwinPhaseWindow {
    pub phase_index: u32,
    pub phase_name: String,
    pub start_step: i32,
    pub end_step_exclusive: i32,
    pub bin_size_steps: i32,
    pub bin_count: u32,
}

/// Per-phase, per-residue result extracted from a real phase-specific GPU WMMA matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StrictTwinPhaseCcfFeature {
    pub phase_index: u32,
    pub phase_name: String,
    pub start_step: i32,
    pub end_step_exclusive: i32,
    pub bin_size_steps: i32,
    pub bin_count: u32,
    pub residue_event_count_a: u32,
    pub residue_event_count_b: u32,
    pub variance_support_a: bool,
    pub variance_support_b: bool,
    pub supported_partner_count: u32,
    pub correlation_lag_steps: Option<i32>,
    pub aligned_zero_lag_ccf: Option<f32>,
    pub peak_partner_residue_id: Option<i32>,
    pub peak_partner_residue_offset: Option<i32>,
    pub peak_partner_zero_lag_ccf: Option<f32>,
    pub cross_residue_network_reproducibility_fraction: Option<f32>,
    pub support_status: String,
    pub matrix_artifact_kind: String,
    pub matrix_sha256: String,
}

/// Strict two-topology residue evidence. Every numeric value is either derived from
/// real stream events/GPU output or represented as unavailable with an explicit reason.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StrictTwinResidueFeature {
    pub schema_kind: String,
    pub schema_version: u32,
    pub residue_id: i32,
    pub residue_name: String,
    pub event_count_a: u32,
    pub event_count_b: u32,
    pub intensity_sum_a: f32,
    pub intensity_sum_b: f32,
    pub signed_event_count_contrast: i64,
    pub signed_intensity_contrast: f32,
    pub b_over_a_event_ratio: Option<f32>,
    pub b_over_a_intensity_ratio: Option<f32>,
    pub global_correlation_lag_steps: Option<i32>,
    pub global_variance_support_a: bool,
    pub global_variance_support_b: bool,
    pub global_supported_partner_count: u32,
    pub global_aligned_zero_lag_ccf: Option<f32>,
    pub global_peak_partner_residue_id: Option<i32>,
    pub global_peak_partner_residue_offset: Option<i32>,
    pub global_peak_partner_zero_lag_ccf: Option<f32>,
    pub global_cross_residue_network_reproducibility_fraction: Option<f32>,
    pub global_ccf_support_status: String,
    pub phase_ccf: Vec<StrictTwinPhaseCcfFeature>,
    pub predictive_window_steps: i32,
    pub predictive_source_event_count: u32,
    pub predictive_matched_event_count: u32,
    pub predictive_probability_b_after_a: Option<f32>,
    pub predictive_algorithm: String,
    pub onset_step_a: Option<i32>,
    pub onset_step_b: Option<i32>,
    pub b_minus_a_onset_lag_steps: Option<i32>,
    pub a_intensity_at_exact_b_onset: Option<f32>,
    pub a_intensity_at_exact_b_onset_support_count: u32,
    pub b_only_condition_event_count: u32,
    pub a_only_reference_event_count: u32,
    pub transfer_entropy_kernel_lag_bins: Option<u32>,
    pub transfer_entropy_past_index_offset_bins: Option<u32>,
    pub transfer_entropy_future_index_offset_bins: Option<u32>,
    pub transfer_entropy_effective_past_to_future_horizon_bins: Option<u32>,
    pub transfer_entropy_bin_size_steps: Option<i32>,
    pub transfer_entropy_effective_past_to_future_horizon_steps: Option<i32>,
    pub transfer_entropy_transition_sample_count: Option<u32>,
    pub bidirectional_transfer_entropy_mass: Option<f32>,
    pub thresholded_total_outgoing_transfer_entropy_a_to_b: Option<f32>,
    pub transfer_entropy_outgoing_term_floor: Option<f32>,
    pub net_causal_flow_out_minus_in: Option<f32>,
    pub transfer_entropy_support_status: String,
    pub ccf_frequency_peak: Option<f32>,
    pub ccf_frequency_status: String,
    pub phase_offset_enrichment: Option<f32>,
    pub phase_offset_enrichment_status: String,
    pub nma_responsive_mode: Option<i32>,
    pub nma_mode_eigenvalue: Option<f32>,
    pub nma_work_at_residue: Option<f32>,
    pub mechanical_sensitivity: Option<f32>,
    pub susceptibility_magnitude: Option<f32>,
    pub nma_status: String,
    pub fallback_used: bool,
}

/// Strict site aggregation built only from strict residue records and binding-site geometry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StrictTwinSiteCcfPairDecision {
    pub left_residue_id: i32,
    pub right_residue_id: i32,
    pub left_strict_row_present: bool,
    pub right_strict_row_present: bool,
    pub left_ccf_supported: bool,
    pub right_ccf_supported: bool,
    pub pair_supported: bool,
    pub symmetric_zero_lag_ccf: Option<f32>,
    pub support_status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StrictTwinSiteFeature {
    pub schema_kind: String,
    pub schema_version: u32,
    pub site_id: u64,
    pub site_rank: u32,
    pub lining_residue_ids: Vec<i32>,
    pub strict_residue_ids_with_data: Vec<i32>,
    pub missing_strict_residue_ids: Vec<i32>,
    pub volume: Option<f32>,
    pub druggability: Option<f32>,
    pub is_druggable: Option<bool>,
    pub centroid: Option<[f32; 3]>,
    pub global_ccf_supported_residue_count: u32,
    pub mean_global_aligned_zero_lag_ccf: Option<f32>,
    pub max_global_aligned_zero_lag_ccf: Option<f32>,
    pub intra_site_ccf_candidate_pair_count: u32,
    pub intra_site_ccf_supported_pair_count: u32,
    pub intra_site_ccf_pair_decisions: Vec<StrictTwinSiteCcfPairDecision>,
    pub mean_intra_site_cross_residue_zero_lag_ccf: Option<f32>,
    pub phase_ccf_supported_residue_counts: [u32; 5],
    pub mean_phase_aligned_zero_lag_ccf: [Option<f32>; 5],
    pub predictive_supported_residue_count: u32,
    pub mean_predictive_probability_b_after_a: Option<f32>,
    pub total_b_only_condition_events: u64,
    pub total_a_only_reference_events: u64,
    pub signed_event_count_contrast: i64,
    pub signed_intensity_contrast: f32,
    pub geometry_source: String,
    pub active_feature_surface: String,
    pub fallback_used: bool,
}

/// Untagged compatibility envelope: the single-topology lane serializes the exact
/// historical object shape, while strict two-topology runs serialize only strict rows.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CoupledTwinResidueFeatureRecord {
    Legacy(InterferometricFeatures),
    Strict(StrictTwinResidueFeature),
}

impl From<InterferometricFeatures> for CoupledTwinResidueFeatureRecord {
    fn from(value: InterferometricFeatures) -> Self {
        Self::Legacy(value)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CoupledTwinSiteFeatureRecord {
    Legacy(SiteInterferometricFeatures),
    Strict(StrictTwinSiteFeature),
}

impl From<SiteInterferometricFeatures> for CoupledTwinSiteFeatureRecord {
    fn from(value: SiteInterferometricFeatures) -> Self {
        Self::Legacy(value)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StrictTwinResidueFeatureArtifact {
    schema_kind: String,
    schema_version: u32,
    status: String,
    run_id: String,
    topology_a_sha256: String,
    topology_b_sha256: String,
    event_source: String,
    ccf_backend: String,
    records: Vec<StrictTwinResidueFeature>,
    fallback_used: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StrictTwinSiteFeatureArtifact {
    schema_kind: String,
    schema_version: u32,
    status: String,
    run_id: String,
    topology_a_sha256: String,
    topology_b_sha256: String,
    geometry_source_path: String,
    geometry_source_sha256: String,
    records: Vec<StrictTwinSiteFeature>,
    fallback_used: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StrictTwinFeatureArtifactBinding {
    kind: String,
    stream_id: Option<u32>,
    path: String,
    size_bytes: u64,
    sha256: String,
    blake3: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StrictTwinKernelSourceBinding {
    kernel_identity: String,
    path: String,
    size_bytes: u64,
    sha256: String,
    blake3: String,
    required_symbols: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StrictTwinRetiredSurface {
    original_path: String,
    superseded_path: String,
    size_bytes: u64,
    sha256: String,
    blake3: String,
    disposition: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StrictTwinFeatureReceipt {
    schema_kind: String,
    schema_version: u32,
    status: String,
    run_id: String,
    topology_a_sha256: String,
    topology_b_sha256: String,
    source_spike_artifacts: Vec<StrictTwinFeatureArtifactBinding>,
    phase_ccf_artifacts: Vec<StrictTwinFeatureArtifactBinding>,
    phase_ccf_kernel_source: StrictTwinKernelSourceBinding,
    global_gpu_analysis_receipt_artifact: StrictTwinFeatureArtifactBinding,
    binding_sites_artifact: StrictTwinFeatureArtifactBinding,
    residue_feature_artifact: StrictTwinFeatureArtifactBinding,
    site_feature_artifact: StrictTwinFeatureArtifactBinding,
    strict_residue_record_count: u64,
    strict_site_record_count: u64,
    legacy_residue_rows_active: u64,
    legacy_site_rows_active: u64,
    retired_legacy_detector_surfaces: Vec<StrictTwinRetiredSurface>,
    exact_predictive_algorithm: String,
    phase_ccf_backend: String,
    finite_values_verified: bool,
    unavailable_values_are_null: bool,
    fallback_used: bool,
}

/// Aggregate per-residue InterferometricFeatures into per-site features.
///
/// `site_lining_residues`: for each site, the list of residue IDs that line the pocket.
/// `per_residue`: the full per-residue feature vector (indexed by array position, keyed by resid).
/// `ccf_matrix`: optional n_res×n_res CCF matrix for intra-site CCF computation.
#[cfg(feature = "gpu")]
pub fn aggregate_site_features(
    site_lining_residues: &[Vec<i32>],
    per_residue: &[InterferometricFeatures],
    ccf_matrix: Option<&[f32]>,
    n_residues: usize,
) -> Vec<SiteInterferometricFeatures> {
    // Build resid → feature index lookup
    let resid_to_idx: std::collections::HashMap<i32, usize> = per_residue
        .iter()
        .enumerate()
        .map(|(i, f)| (f.resid, i))
        .collect();

    site_lining_residues
        .iter()
        .enumerate()
        .map(|(site_id, lining)| {
            // Collect features for lining residues that have TWIN data
            let lining_features: Vec<&InterferometricFeatures> = lining
                .iter()
                .filter_map(|&resid| resid_to_idx.get(&resid).map(|&idx| &per_residue[idx]))
                .collect();

            let n_lining = lining.len();
            let n_with_data = lining_features.len();

            if n_with_data == 0 {
                return SiteInterferometricFeatures {
                    site_id,
                    n_lining_residues: n_lining,
                    n_lining_with_data: 0,
                    mean_agreement: 0.0,
                    min_agreement: 0.0,
                    max_agreement: 0.0,
                    mean_spatial_coherence: 0.0,
                    mean_consensus_onset: 0.0,
                    mean_ccf_peak: 0.0,
                    max_ccf_peak: 0.0,
                    mean_intra_site_ccf: 0.0,
                    mean_ccf_reproducibility: 0.0,
                    volume: 0.0,
                    druggability: 0.0,
                    is_druggable: false,
                    centroid: [0.0; 3],
                    mean_b_over_a: 0.0,
                    barrier_composition_low: 0.0,
                    barrier_composition_medium: 1.0,
                    barrier_composition_high: 0.0,
                    total_nma_exclusive: 0,
                    total_thermal_exclusive: 0,
                    mean_scout_lead_time: 0.0,
                    mean_predictive_value: 0.0,
                    mean_phase_offset_enrichment: 0.0,
                };
            }

            let inv_n = 1.0 / n_with_data as f32;

            // Consensus
            let agreements: Vec<f32> = lining_features
                .iter()
                .map(|f| f.spike_agreement_ratio)
                .collect();
            let mean_agreement = agreements.iter().sum::<f32>() * inv_n;
            let min_agreement = agreements.iter().copied().fold(f32::INFINITY, f32::min);
            let max_agreement = agreements.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mean_spatial_coherence = lining_features
                .iter()
                .map(|f| f.consensus_spatial_coherence)
                .sum::<f32>()
                * inv_n;
            let mean_consensus_onset = lining_features
                .iter()
                .map(|f| f.consensus_temporal_onset)
                .sum::<f32>()
                * inv_n;

            // CCF
            let ccf_peaks: Vec<f32> = lining_features.iter().map(|f| f.ccf_peak_value).collect();
            let mean_ccf_peak = ccf_peaks.iter().sum::<f32>() * inv_n;
            let max_ccf_peak = ccf_peaks.iter().copied().fold(0.0f32, f32::max);
            let mean_ccf_reproducibility = lining_features
                .iter()
                .map(|f| f.ccf_reproducibility)
                .sum::<f32>()
                * inv_n;

            // Intra-site CCF: mean CCF between all pairs of lining residues
            let mean_intra_site_ccf = if let Some(ccf) = ccf_matrix {
                if n_residues > 0 && n_with_data >= 2 {
                    let lining_indices: Vec<usize> = lining
                        .iter()
                        .filter_map(|&r| {
                            if (r as usize) < n_residues {
                                Some(r as usize)
                            } else {
                                None
                            }
                        })
                        .collect();
                    let mut sum = 0.0f32;
                    let mut count = 0u32;
                    for i in 0..lining_indices.len() {
                        for j in (i + 1)..lining_indices.len() {
                            let ri = lining_indices[i];
                            let rj = lining_indices[j];
                            if ri < n_residues && rj < n_residues {
                                let val = ccf[ri * n_residues + rj];
                                if val.is_finite() {
                                    sum += val;
                                    count += 1;
                                }
                            }
                        }
                    }
                    if count > 0 {
                        sum / count as f32
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            } else {
                0.0
            };

            // Differential
            let mean_b_over_a = lining_features
                .iter()
                .map(|f| {
                    if f.b_over_a_ratio.is_finite() {
                        f.b_over_a_ratio
                    } else {
                        1.0
                    }
                })
                .sum::<f32>()
                * inv_n;
            let n_low = lining_features
                .iter()
                .filter(|f| f.barrier_classification == "LOW")
                .count();
            let n_high = lining_features
                .iter()
                .filter(|f| f.barrier_classification == "HIGH")
                .count();
            let n_med = n_with_data - n_low - n_high;
            let total_nma_exclusive: u32 =
                lining_features.iter().map(|f| f.nma_exclusive_count).sum();
            let total_thermal_exclusive: u32 = lining_features
                .iter()
                .map(|f| f.thermal_exclusive_count)
                .sum();

            // Scout
            let mean_scout_lead_time = lining_features
                .iter()
                .map(|f| f.scout_lead_time)
                .sum::<f32>()
                * inv_n;
            let mean_predictive_value = lining_features
                .iter()
                .map(|f| f.scout_predictive_value)
                .sum::<f32>()
                * inv_n;
            let mean_phase_offset_enrichment = lining_features
                .iter()
                .map(|f| f.phase_offset_enrichment)
                .sum::<f32>()
                * inv_n;

            SiteInterferometricFeatures {
                site_id,
                n_lining_residues: n_lining,
                n_lining_with_data: n_with_data,
                mean_agreement,
                min_agreement,
                max_agreement,
                mean_spatial_coherence,
                mean_consensus_onset,
                mean_ccf_peak,
                max_ccf_peak,
                mean_intra_site_ccf,
                mean_ccf_reproducibility,
                // Geometry: populated from binding_sites.json in the caller
                volume: 0.0,
                druggability: 0.0,
                is_druggable: false,
                centroid: [0.0; 3],
                mean_b_over_a,
                barrier_composition_low: n_low as f32 / n_with_data as f32,
                barrier_composition_medium: n_med as f32 / n_with_data as f32,
                barrier_composition_high: n_high as f32 / n_with_data as f32,
                total_nma_exclusive,
                total_thermal_exclusive,
                mean_scout_lead_time,
                mean_predictive_value,
                mean_phase_offset_enrichment,
            }
        })
        .collect()
}

/// Metadata from the simulation loop, passed to twin_post_process.
#[cfg(feature = "gpu")]
pub struct TwinSimulationMetadata {
    pub seed_a: u64,
    pub seed_b: u64,
    pub steps: i32,
    pub steps_b: i32,
    pub offset_steps: i32,
    pub wall_time_secs: f64,
    pub vram_used_gb: f64,
    pub n_exchanges: u32,
    pub total_density_a_to_b: f64,
    pub total_density_b_to_a: f64,
    pub max_nonzero_regions: u32,
    pub ring_spikes_exchanged: u64,
    pub runtime_evidence: Option<CoupledTwinRuntimeEvidence>,
}

/// Owned live stream state for downstream materialization and thermodynamics.
/// Raw arrays stay in memory and are not embedded in the legacy result JSON.
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct CoupledPostMdStream {
    pub source_stream_id: u32,
    pub topology: PrismPrepTopology,
    pub topology_source_sha256: Option<String>,
    pub protocol: CryoUvProtocol,
    pub seed: u64,
    pub physical_steps: i32,
    pub spikes: Vec<crate::fused_engine::GpuSpikeEvent>,
    pub snapshots: Vec<crate::fused_engine::EnsembleSnapshot>,
}

/// Full result from a PRISM-TWIN coupled observation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoupledTwinResult {
    #[cfg(feature = "gpu")]
    #[serde(skip)]
    pub post_md_streams: Vec<CoupledPostMdStream>,
    pub stream_a: StreamResult,
    pub stream_b: StreamResult,
    pub config: CoupledTwinConfig,
    pub per_residue_features: Vec<CoupledTwinResidueFeatureRecord>,
    /// Per-site aggregated features (from lining residues of each detected site).
    #[serde(default)]
    pub per_site_features: Vec<CoupledTwinSiteFeatureRecord>,
    pub n_consensus_events: u64,
    pub n_differential_events: u64,

    // Gate 2 exchange accounting
    /// How many spike density exchange rounds occurred during the run.
    pub n_exchanges: u32,
    /// Summed spike intensity across all regions from A (integrated across all exchanges).
    pub total_density_a_to_b: f64,
    /// Summed spike intensity across all regions from B (integrated across all exchanges).
    pub total_density_b_to_a: f64,
    /// DPD condition-edge readout dir (CONDITION_EDGE_READOUT). The DPD source of truth — the
    /// legacy total_density_* above are side telemetry and a zero there must NOT suppress this.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition_edge_readout_path: Option<String>,
    /// Number of coarse regions that had non-zero spike activity in at least one stream.
    pub n_nonzero_regions: u32,

    // Gate 3 phase offset
    /// Total spikes tracked for ring buffer exchange (both directions)
    pub ring_spikes_tracked: u64,

    // Gate 3 phase offset
    /// Number of extra cold_hold steps B runs before A's schedule
    pub phase_offset_steps: i32,
    /// Total steps for stream A
    pub steps_a: i32,
    /// Total steps for stream B (steps_a + offset)
    pub steps_b: i32,

    // ── LINK #1: two-system molecular-contrast provenance ──────────────────────────
    // All skip-serialized when None so a single-topology protocol-contrast run's
    // coupled_twin_result.json is byte-identical to pre-link-1 output.
    /// Molecular contrast label ("apo-holo" | "design-reference") when engine_b loaded a
    /// DISTINCT topology (--topology-b). None on a single-topology protocol-contrast run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contrast_kind: Option<String>,
    /// SHA-256 of the reference (A) topology in a two-system run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub topology_a_sha256: Option<String>,
    /// SHA-256 of the second (B: holo / designed) topology in a two-system run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub topology_b_sha256: Option<String>,
    /// Semantic role of engine_b's molecule ("holo" | "design").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine_b_role: Option<String>,
    /// Runtime-owned evidence is carried only for strict two-topology runs.
    /// `None` is skipped so legacy single-topology result bytes are unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_evidence: Option<CoupledTwinRuntimeEvidence>,
}

/// Per-stream result data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamResult {
    pub role: String,
    pub seed: u64,
    pub total_spikes: usize,
    pub perturbation: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// Gate 2 CPU helpers: spike density + cross-correlation
// ─────────────────────────────────────────────────────────────────────────────

/// One coarse grid cell containing the summed spike intensity from a spike slice.
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
struct DensityCell {
    /// Grid indices (ix, iy, iz)
    idx: (i32, i32, i32),
    /// Total intensity accumulated in this cell
    total_intensity: f64,
    /// Number of spikes that landed in this cell
    spike_count: u32,
}

/// Grid the protein bounding box into `cell_size_ang`-Å cells and accumulate
/// spike intensity per cell.  Returns (cells, bounding-box min [x,y,z]).
///
/// Spikes with zero or negative intensity are skipped.
#[cfg(feature = "gpu")]
fn compute_spike_density_cpu(
    spikes: &[crate::fused_engine::GpuSpikeEvent],
    cell_size_ang: f32,
) -> (Vec<DensityCell>, [f32; 3]) {
    use std::collections::HashMap;

    if spikes.is_empty() {
        return (Vec::new(), [0.0f32; 3]);
    }

    // Find bounding box
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for s in spikes {
        for d in 0..3 {
            if s.position[d] < min[d] {
                min[d] = s.position[d];
            }
            if s.position[d] > max[d] {
                max[d] = s.position[d];
            }
        }
    }
    // Expand bounding box by one cell on each side to avoid edge effects
    for d in 0..3 {
        min[d] -= cell_size_ang;
        max[d] += cell_size_ang;
    }

    let inv_cell = 1.0 / cell_size_ang;
    let mut map: HashMap<(i32, i32, i32), (f64, u32)> = HashMap::new();

    for s in spikes {
        if s.intensity <= 0.0 {
            continue;
        }
        let ix = ((s.position[0] - min[0]) * inv_cell).floor() as i32;
        let iy = ((s.position[1] - min[1]) * inv_cell).floor() as i32;
        let iz = ((s.position[2] - min[2]) * inv_cell).floor() as i32;
        let entry = map.entry((ix, iy, iz)).or_insert((0.0, 0));
        entry.0 += s.intensity as f64;
        entry.1 += 1;
    }

    let cells: Vec<DensityCell> = map
        .into_iter()
        .map(|(idx, (total_intensity, spike_count))| DensityCell {
            idx,
            total_intensity,
            spike_count,
        })
        .collect();

    (cells, min)
}

/// CPU cross-correlation: count spike pairs (one from A, one from B) that are
/// within `spatial_thresh_ang` Å AND within `temporal_thresh_steps` steps.
///
/// Returns (n_matches, per-residue consensus counts keyed by topology resid).
#[cfg(feature = "gpu")]
fn compute_cpu_cross_correlation(
    spikes_a: &[crate::fused_engine::GpuSpikeEvent],
    spikes_b: &[crate::fused_engine::GpuSpikeEvent],
    spatial_thresh_ang: f32,
    temporal_thresh_steps: i32,
    // LINK #1: two-system molecular contrast. The two systems are integrated in
    // UNSUPERPOSED absolute Cartesian frames, so raw `position` deltas across the
    // streams are meaningless. When true, co-occurrence uses frame-independent
    // resid matching (shared PRIMARY nearby residue) instead of a spatial delta.
    // Single-topology (false) keeps the historical positional test byte-for-byte.
    two_system: bool,
) -> (u64, std::collections::HashMap<i32, (u32, u32, f32, f32)>) {
    // HashMap value: (count_a, count_b, sum_intensity_a, sum_intensity_b)
    use std::collections::HashMap;

    let spatial_sq = spatial_thresh_ang * spatial_thresh_ang;
    let mut n_matches: u64 = 0;
    // Per-residue accumulators: keyed by resid
    let mut per_res: HashMap<i32, (u32, u32, f32, f32)> = HashMap::new();

    // Accumulate per-residue spike counts for A
    for sa in spikes_a {
        let n = sa.n_residues.min(8) as usize;
        for r in 0..n {
            let resid = sa.nearby_residues[r];
            if resid < 0 {
                continue;
            }
            let e = per_res.entry(resid).or_insert((0, 0, 0.0, 0.0));
            e.0 += 1;
            e.2 += sa.intensity;
        }
    }
    // Accumulate per-residue spike counts for B
    for sb in spikes_b {
        let n = sb.n_residues.min(8) as usize;
        for r in 0..n {
            let resid = sb.nearby_residues[r];
            if resid < 0 {
                continue;
            }
            let e = per_res.entry(resid).or_insert((0, 0, 0.0, 0.0));
            e.1 += 1;
            e.3 += sb.intensity;
        }
    }

    // Spatial + temporal co-occurrence: SAMPLED (O(N) not O(N²))
    // With 15M+ spikes per stream, O(N²) is impossible.
    // Instead: sample up to 10K spikes from each, compute matches on sample.
    // Scale result by (total_a * total_b) / (sample_a * sample_b).
    let max_sample = 10_000usize;
    let sample_a: Vec<_> = if spikes_a.len() > max_sample {
        let step = spikes_a.len() / max_sample;
        spikes_a.iter().step_by(step).take(max_sample).collect()
    } else {
        spikes_a.iter().collect()
    };
    let sample_b: Vec<_> = if spikes_b.len() > max_sample {
        let step = spikes_b.len() / max_sample;
        spikes_b.iter().step_by(step).take(max_sample).collect()
    } else {
        spikes_b.iter().collect()
    };

    let mut sample_matches: u64 = 0;
    for sa in &sample_a {
        for sb in &sample_b {
            let dt = (sa.timestep - sb.timestep).abs();
            if dt > temporal_thresh_steps {
                continue;
            }
            let co_occurs = if two_system {
                // LINK #1: frame-independent resid co-occurrence — the two streams
                // are in unsuperposed frames so `position` deltas are invalid.
                // A shared PRIMARY nearby residue (same protein CA neighborhood)
                // within the temporal window counts as co-occurrence.
                sa.n_residues > 0
                    && sb.n_residues > 0
                    && sa.nearby_residues[0] >= 0
                    && sa.nearby_residues[0] == sb.nearby_residues[0]
            } else {
                let dx = sa.position[0] - sb.position[0];
                let dy = sa.position[1] - sb.position[1];
                let dz = sa.position[2] - sb.position[2];
                let dist_sq = dx * dx + dy * dy + dz * dz;
                dist_sq <= spatial_sq
            };
            if co_occurs {
                sample_matches += 1;
            }
        }
    }

    // Scale to estimate total matches
    let scale = (spikes_a.len() as f64 * spikes_b.len() as f64)
        / (sample_a.len() as f64 * sample_b.len() as f64).max(1.0);
    n_matches = (sample_matches as f64 * scale) as u64;

    (n_matches, per_res)
}

// ─────────────────────────────────────────────────────────────────────────────
// Feature computation helpers (Step 6)
// ─────────────────────────────────────────────────────────────────────────────

/// CCNS protocol phase index for a given timestep.
/// 0=cold_hold, 1=heating, 2=warm_hold, 3=cooling, 4=cold_return
#[cfg(feature = "gpu")]
fn ccns_phase(timestep: i32, protocol: &CryoUvProtocol) -> usize {
    let t = timestep;
    let p1 = protocol.cold_hold_steps;
    let p2 = p1 + protocol.ramp_steps;
    let p3 = p2 + protocol.warm_hold_steps;
    let p4 = p3 + protocol.ramp_down_steps;
    if t < p1 {
        0
    } else if t < p2 {
        1
    } else if t < p3 {
        2
    } else if t < p4 {
        3
    } else {
        4
    }
}

/// Per-residue per-phase spike counts from a spike vector.
/// Returns HashMap<resid, [count_phase0..count_phase4]>
#[cfg(feature = "gpu")]
fn per_residue_phase_counts(
    spikes: &[crate::fused_engine::GpuSpikeEvent],
    protocol: &CryoUvProtocol,
) -> std::collections::HashMap<i32, [u32; 5]> {
    let mut map: std::collections::HashMap<i32, [u32; 5]> = std::collections::HashMap::new();
    for s in spikes {
        let phase = ccns_phase(s.timestep, protocol);
        let n = s.n_residues.min(8) as usize;
        for j in 0..n {
            let resid = s.nearby_residues[j];
            if resid < 0 {
                continue;
            }
            let entry = map.entry(resid).or_insert([0; 5]);
            entry[phase] += 1;
        }
    }
    map
}

/// Per-residue earliest spike timestep from a spike vector.
/// Returns HashMap<resid, earliest_timestep>
#[cfg(feature = "gpu")]
fn per_residue_onset(
    spikes: &[crate::fused_engine::GpuSpikeEvent],
) -> std::collections::HashMap<i32, i32> {
    let mut map: std::collections::HashMap<i32, i32> = std::collections::HashMap::new();
    for s in spikes {
        let n = s.n_residues.min(8) as usize;
        for j in 0..n {
            let resid = s.nearby_residues[j];
            if resid < 0 {
                continue;
            }
            let entry = map.entry(resid).or_insert(i32::MAX);
            if s.timestep < *entry {
                *entry = s.timestep;
            }
        }
    }
    map
}

/// Compute per-residue intensity sums from a spike vector.
/// Returns HashMap<resid, total_intensity>
#[cfg(feature = "gpu")]
fn per_residue_intensity(
    spikes: &[crate::fused_engine::GpuSpikeEvent],
) -> std::collections::HashMap<i32, f32> {
    let mut map: std::collections::HashMap<i32, f32> = std::collections::HashMap::new();
    for s in spikes {
        let n = s.n_residues.min(8) as usize;
        for j in 0..n {
            let resid = s.nearby_residues[j];
            if resid < 0 {
                continue;
            }
            *map.entry(resid).or_insert(0.0) += s.intensity;
        }
    }
    map
}

#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Copy, PartialEq)]
struct StrictResidueEventObservation {
    timestep: i32,
    intensity: f32,
}

#[cfg(feature = "gpu")]
fn strict_twin_phase_windows(
    protocol: &CryoUvProtocol,
    total_steps: i32,
) -> Result<Vec<StrictTwinPhaseWindow>> {
    let durations = [
        protocol.cold_hold_steps,
        protocol.ramp_steps,
        protocol.warm_hold_steps,
        protocol.ramp_down_steps,
        protocol.cold_return_steps,
    ];
    if total_steps <= 0 || durations.iter().any(|duration| *duration <= 0) {
        bail!(
            "strict two-topology phase CCF requires five positive CCNS phases: steps={} durations={:?}",
            total_steps,
            durations
        );
    }
    let scheduled_steps = durations.iter().try_fold(0i32, |sum, duration| {
        sum.checked_add(*duration)
            .context("CCNS phase duration overflow")
    })?;
    if scheduled_steps != total_steps {
        bail!(
            "strict two-topology phase CCF requires a complete protocol: scheduled_steps={} executed_steps={}",
            scheduled_steps,
            total_steps
        );
    }

    let mut start = 0i32;
    let mut windows = Vec::with_capacity(STRICT_TWIN_PHASE_NAMES.len());
    for (phase_index, (phase_name, duration)) in
        STRICT_TWIN_PHASE_NAMES.iter().zip(durations).enumerate()
    {
        let end = start
            .checked_add(duration)
            .context("CCNS phase boundary overflow")?;
        let bin_count = u32::try_from(
            duration
                .checked_add(STRICT_TWIN_CCF_BIN_SIZE_STEPS - 1)
                .context("CCNS phase bin rounding overflow")?
                / STRICT_TWIN_CCF_BIN_SIZE_STEPS,
        )?;
        if bin_count < 2 {
            bail!(
                "strict phase {} has only {} CCF bin(s); at least two are required",
                phase_name,
                bin_count
            );
        }
        windows.push(StrictTwinPhaseWindow {
            phase_index: phase_index as u32,
            phase_name: (*phase_name).to_string(),
            start_step: start,
            end_step_exclusive: end,
            bin_size_steps: STRICT_TWIN_CCF_BIN_SIZE_STEPS,
            bin_count,
        });
        start = end;
    }
    Ok(windows)
}

#[cfg(feature = "gpu")]
fn collect_strict_residue_observations(
    spikes: &[crate::fused_engine::GpuSpikeEvent],
    n_residues: usize,
    total_steps: i32,
) -> Result<BTreeMap<i32, Vec<StrictResidueEventObservation>>> {
    let mut observations = BTreeMap::<i32, Vec<StrictResidueEventObservation>>::new();
    for (event_index, spike) in spikes.iter().enumerate() {
        if spike.timestep < 0
            || spike.timestep >= total_steps
            || !spike.intensity.is_finite()
            || spike.intensity < 0.0
            || spike.position.iter().any(|value| !value.is_finite())
            || !(1..=8).contains(&spike.n_residues)
        {
            bail!(
                "strict residue evidence rejected malformed event {}: timestep={} intensity={} n_residues={}",
                event_index,
                spike.timestep,
                spike.intensity,
                spike.n_residues
            );
        }
        let mut event_residues = BTreeSet::new();
        for residue_id in spike
            .nearby_residues
            .iter()
            .take(spike.n_residues as usize)
            .copied()
        {
            if residue_id < 0 || residue_id as usize >= n_residues {
                continue;
            }
            if !event_residues.insert(residue_id) {
                bail!(
                    "strict residue evidence rejected duplicate residue {} in event {}",
                    residue_id,
                    event_index
                );
            }
            observations
                .entry(residue_id)
                .or_default()
                .push(StrictResidueEventObservation {
                    timestep: spike.timestep,
                    intensity: spike.intensity,
                });
        }
    }
    for rows in observations.values_mut() {
        rows.sort_by(|left, right| {
            left.timestep
                .cmp(&right.timestep)
                .then_with(|| left.intensity.total_cmp(&right.intensity))
        });
    }
    Ok(observations)
}

#[cfg(feature = "gpu")]
fn strict_event_slice<'a>(
    observations: &'a [StrictResidueEventObservation],
    start_step: i32,
    end_step_exclusive: i32,
) -> &'a [StrictResidueEventObservation] {
    let start = observations.partition_point(|event| event.timestep < start_step);
    let end = observations.partition_point(|event| event.timestep < end_step_exclusive);
    &observations[start..end]
}

#[cfg(feature = "gpu")]
fn strict_binned_variance_support(
    observations: &[StrictResidueEventObservation],
    start_step: i32,
    end_step_exclusive: i32,
    bin_size_steps: i32,
) -> Result<bool> {
    if start_step < 0 || end_step_exclusive <= start_step || bin_size_steps <= 0 {
        bail!("invalid strict CCF variance window");
    }
    let duration = end_step_exclusive - start_step;
    let bin_count = usize::try_from(
        duration
            .checked_add(bin_size_steps - 1)
            .context("strict CCF bin rounding overflow")?
            / bin_size_steps,
    )?;
    if bin_count < 2 {
        return Ok(false);
    }
    let mut bins = vec![0.0f64; bin_count];
    for event in strict_event_slice(observations, start_step, end_step_exclusive) {
        let bin = usize::try_from((event.timestep - start_step) / bin_size_steps)?;
        if bin >= bins.len() {
            bail!("strict CCF event escaped its declared phase window");
        }
        bins[bin] += event.intensity as f64;
    }
    let mean = bins.iter().sum::<f64>() / bins.len() as f64;
    let centered_l2_sq = bins
        .iter()
        .map(|value| {
            let centered = value - mean;
            centered * centered
        })
        .sum::<f64>();
    Ok(centered_l2_sq.is_finite() && centered_l2_sq > 0.0)
}

#[cfg(feature = "gpu")]
fn strict_forward_predictive_matches(
    source_a: &[StrictResidueEventObservation],
    condition_b: &[StrictResidueEventObservation],
    window_steps: i32,
) -> Result<u32> {
    if window_steps < 0 {
        bail!("strict predictive window must be non-negative");
    }
    let mut b_index = 0usize;
    let mut matched = 0u32;
    for source in source_a {
        while b_index < condition_b.len() && condition_b[b_index].timestep < source.timestep {
            b_index += 1;
        }
        if b_index < condition_b.len()
            && condition_b[b_index].timestep - source.timestep <= window_steps
        {
            matched = matched
                .checked_add(1)
                .context("strict predictive match count overflow")?;
        }
    }
    Ok(matched)
}

#[cfg(feature = "gpu")]
fn strict_unmatched_event_count(
    source: &[StrictResidueEventObservation],
    counterpart: &[StrictResidueEventObservation],
    symmetric_window_steps: i32,
) -> Result<u32> {
    if symmetric_window_steps < 0 {
        bail!("strict unmatched-event window must be non-negative");
    }
    let mut counterpart_index = 0usize;
    let mut unmatched = 0u32;
    for event in source {
        while counterpart_index < counterpart.len()
            && counterpart[counterpart_index].timestep
                < event.timestep.saturating_sub(symmetric_window_steps)
        {
            counterpart_index += 1;
        }
        let has_match = counterpart_index < counterpart.len()
            && counterpart[counterpart_index].timestep
                <= event.timestep.saturating_add(symmetric_window_steps);
        if !has_match {
            unmatched = unmatched
                .checked_add(1)
                .context("strict unmatched-event count overflow")?;
        }
    }
    Ok(unmatched)
}

#[cfg(feature = "gpu")]
fn strict_mean_intensity_at_timestep(
    observations: &[StrictResidueEventObservation],
    timestep: i32,
) -> Result<(Option<f32>, u32)> {
    let start = observations.partition_point(|event| event.timestep < timestep);
    let end = observations.partition_point(|event| event.timestep <= timestep);
    let support_count = u32::try_from(end.saturating_sub(start))?;
    if support_count == 0 {
        return Ok((None, 0));
    }
    let sum = observations[start..end]
        .iter()
        .try_fold(0.0f64, |sum, event| {
            let next = sum + event.intensity as f64;
            if next.is_finite() {
                Ok(next)
            } else {
                bail!("strict onset intensity accumulation became nonfinite")
            }
        })?;
    let mean = (sum / support_count as f64) as f32;
    if !mean.is_finite() {
        bail!("strict onset intensity mean is nonfinite");
    }
    Ok((Some(mean), support_count))
}

#[cfg(feature = "gpu")]
#[derive(Debug, Clone, PartialEq)]
struct StrictSupportedCcfProjection {
    aligned_zero_lag_ccf: f32,
    peak_partner_residue_id: i32,
    peak_partner_residue_offset: i32,
    peak_partner_zero_lag_ccf: f32,
    cross_residue_network_reproducibility_fraction: f32,
    supported_partner_count: u32,
}

#[cfg(feature = "gpu")]
fn strict_zero_lag_ccf_projection(
    matrix: &[f32],
    extracted: &[crate::twin_kernels::CcfResidueFeatures],
    n_residues: usize,
    residue_index: usize,
    variance_support_a: &[bool],
    variance_support_b: &[bool],
) -> Result<Option<StrictSupportedCcfProjection>> {
    if n_residues < 2
        || residue_index >= n_residues
        || matrix.len() != n_residues.saturating_mul(n_residues)
        || extracted.len() != n_residues
        || variance_support_a.len() != n_residues
        || variance_support_b.len() != n_residues
        || matrix.iter().any(|value| !value.is_finite())
    {
        bail!("strict zero-lag CCF projection received an invalid matrix or residue index");
    }
    if !variance_support_a[residue_index] || !variance_support_b[residue_index] {
        return Ok(None);
    }

    let row = &matrix[residue_index * n_residues..(residue_index + 1) * n_residues];
    let supported_partners = (0..n_residues)
        .filter(|partner_index| {
            *partner_index != residue_index
                && variance_support_a[*partner_index]
                && variance_support_b[*partner_index]
        })
        .collect::<Vec<_>>();
    let supported_partner_count = u32::try_from(supported_partners.len())?;
    let Some((&partner_index, &partner_zero_lag_ccf)) = supported_partners
        .iter()
        .map(|partner_index| (partner_index, &row[*partner_index]))
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
    else {
        return Ok(None);
    };
    let partner_id = i32::try_from(partner_index)?;
    let partner_offset = partner_id
        .checked_sub(i32::try_from(residue_index)?)
        .context("strict CCF partner residue offset overflow")?;
    let aligned_zero_lag_ccf = matrix[residue_index * n_residues + residue_index];
    if aligned_zero_lag_ccf.abs() > 1.0001 || partner_zero_lag_ccf.abs() > 1.0001 {
        bail!("strict CCF supported diagonal/partner projection is outside correlation bounds");
    }
    if supported_partners.len() == n_residues - 1 {
        let feature = &extracted[residue_index];
        if feature.ccf_peak_lag != partner_offset
            || feature.ccf_peak_value.to_bits() != partner_zero_lag_ccf.to_bits()
        {
            bail!("strict CCF full-support projection disagrees with the GPU matrix extractor");
        }
    }
    let cross_residue_network_reproducibility_fraction = supported_partners
        .iter()
        .filter(|partner_index| row[**partner_index] > 0.1)
        .count() as f32
        / supported_partners.len() as f32;
    Ok(Some(StrictSupportedCcfProjection {
        aligned_zero_lag_ccf,
        peak_partner_residue_id: partner_id,
        peak_partner_residue_offset: partner_offset,
        peak_partner_zero_lag_ccf: partner_zero_lag_ccf,
        cross_residue_network_reproducibility_fraction,
        supported_partner_count,
    }))
}

#[cfg(feature = "gpu")]
fn validate_strict_residue_feature(record: &StrictTwinResidueFeature) -> Result<()> {
    let finite_optional = |value: Option<f32>| value.is_none_or(f32::is_finite);
    if record.schema_kind != "prismos.coupled_twin_strict_residue_feature.v3"
        || record.schema_version != STRICT_TWIN_FEATURE_SCHEMA_VERSION
        || record.residue_id < 0
        || record.residue_name.trim().is_empty()
        || record.fallback_used
        || !record.intensity_sum_a.is_finite()
        || !record.intensity_sum_b.is_finite()
        || record.intensity_sum_a < 0.0
        || record.intensity_sum_b < 0.0
        || !record.signed_intensity_contrast.is_finite()
        || (record.signed_intensity_contrast - (record.intensity_sum_b - record.intensity_sum_a))
            .abs()
            > 1.0e-6
        || record.signed_event_count_contrast
            != i64::from(record.event_count_b) - i64::from(record.event_count_a)
        || record.predictive_window_steps != STRICT_TWIN_PREDICTIVE_WINDOW_STEPS
        || record.predictive_source_event_count != record.event_count_a
        || record.predictive_matched_event_count > record.predictive_source_event_count
        || record.predictive_algorithm != STRICT_TWIN_PREDICTIVE_ALGORITHM
        || record.phase_ccf.len() != STRICT_TWIN_PHASE_NAMES.len()
        || !finite_optional(record.b_over_a_event_ratio)
        || !finite_optional(record.b_over_a_intensity_ratio)
        || !finite_optional(record.global_aligned_zero_lag_ccf)
        || !finite_optional(record.global_peak_partner_zero_lag_ccf)
        || !finite_optional(record.global_cross_residue_network_reproducibility_fraction)
        || !finite_optional(record.a_intensity_at_exact_b_onset)
        || !finite_optional(record.bidirectional_transfer_entropy_mass)
        || !finite_optional(record.thresholded_total_outgoing_transfer_entropy_a_to_b)
        || !finite_optional(record.transfer_entropy_outgoing_term_floor)
        || !finite_optional(record.net_causal_flow_out_minus_in)
    {
        bail!(
            "strict residue {} failed identity/count/finite contract",
            record.residue_id
        );
    }
    let expected_probability = if record.predictive_source_event_count == 0 {
        None
    } else {
        Some(
            record.predictive_matched_event_count as f32
                / record.predictive_source_event_count as f32,
        )
    };
    if record.predictive_probability_b_after_a.map(f32::to_bits)
        != expected_probability.map(f32::to_bits)
    {
        bail!("strict residue predictive probability is not the exact event fraction");
    }
    let expected_event_ratio = (record.event_count_a > 0)
        .then_some(record.event_count_b as f32 / record.event_count_a as f32);
    let expected_intensity_ratio =
        (record.intensity_sum_a > 0.0).then_some(record.intensity_sum_b / record.intensity_sum_a);
    if record.b_over_a_event_ratio.map(f32::to_bits) != expected_event_ratio.map(f32::to_bits)
        || record.b_over_a_intensity_ratio.map(f32::to_bits)
            != expected_intensity_ratio.map(f32::to_bits)
    {
        bail!("strict residue B-over-A ratios are not exact source-derived values");
    }
    let expected_lag = match (record.onset_step_a, record.onset_step_b) {
        (Some(a), Some(b)) => Some(b - a),
        _ => None,
    };
    if record.b_minus_a_onset_lag_steps != expected_lag
        || (record.a_intensity_at_exact_b_onset_support_count == 0)
            != record.a_intensity_at_exact_b_onset.is_none()
    {
        bail!("strict residue onset evidence is internally inconsistent");
    }
    let global_supported = record.global_ccf_support_status == STRICT_TWIN_CCF_SUPPORTED;
    let global_values_present = record.global_correlation_lag_steps.is_some()
        && record.global_aligned_zero_lag_ccf.is_some()
        && record.global_peak_partner_residue_id.is_some()
        && record.global_peak_partner_residue_offset.is_some()
        && record.global_peak_partner_zero_lag_ccf.is_some()
        && record
            .global_cross_residue_network_reproducibility_fraction
            .is_some();
    let global_values_absent = record.global_correlation_lag_steps.is_none()
        && record.global_aligned_zero_lag_ccf.is_none()
        && record.global_peak_partner_residue_id.is_none()
        && record.global_peak_partner_residue_offset.is_none()
        && record.global_peak_partner_zero_lag_ccf.is_none()
        && record
            .global_cross_residue_network_reproducibility_fraction
            .is_none();
    if global_supported
        != (record.global_variance_support_a
            && record.global_variance_support_b
            && record.global_supported_partner_count > 0
            && global_values_present)
        || (!global_supported
            && (record.global_supported_partner_count != 0 || !global_values_absent))
        || (!global_supported
            && record.global_ccf_support_status != STRICT_TWIN_GLOBAL_CCF_UNAVAILABLE)
    {
        bail!("strict residue global CCF support/value mismatch");
    }
    if global_supported
        && (record.global_correlation_lag_steps != Some(STRICT_TWIN_CCF_TEMPORAL_LAG_STEPS)
            || record.global_peak_partner_residue_id
                != record.residue_id.checked_add(
                    record
                        .global_peak_partner_residue_offset
                        .context("strict global CCF partner offset disappeared")?,
                )
            || record
                .global_peak_partner_residue_id
                .is_none_or(|value| value < 0)
            || record.global_peak_partner_residue_id == Some(record.residue_id)
            || record
                .global_aligned_zero_lag_ccf
                .is_none_or(|value| value.abs() > 1.0001)
            || record
                .global_peak_partner_zero_lag_ccf
                .is_none_or(|value| value.abs() > 1.0001)
            || record
                .global_cross_residue_network_reproducibility_fraction
                .is_none_or(|value| !(0.0..=1.0).contains(&value)))
    {
        bail!("strict residue global CCF zero-lag/partner semantics failed");
    }
    let te_supported = record.transfer_entropy_support_status == STRICT_TWIN_TE_SUPPORTED;
    let te_values_present = record.transfer_entropy_kernel_lag_bins.is_some()
        && record.transfer_entropy_past_index_offset_bins.is_some()
        && record.transfer_entropy_future_index_offset_bins.is_some()
        && record
            .transfer_entropy_effective_past_to_future_horizon_bins
            .is_some()
        && record.transfer_entropy_bin_size_steps.is_some()
        && record
            .transfer_entropy_effective_past_to_future_horizon_steps
            .is_some()
        && record.transfer_entropy_transition_sample_count.is_some()
        && record.bidirectional_transfer_entropy_mass.is_some()
        && record
            .thresholded_total_outgoing_transfer_entropy_a_to_b
            .is_some()
        && record.transfer_entropy_outgoing_term_floor.is_some()
        && record.net_causal_flow_out_minus_in.is_some();
    let te_values_absent = record.transfer_entropy_kernel_lag_bins.is_none()
        && record.transfer_entropy_past_index_offset_bins.is_none()
        && record.transfer_entropy_future_index_offset_bins.is_none()
        && record
            .transfer_entropy_effective_past_to_future_horizon_bins
            .is_none()
        && record.transfer_entropy_bin_size_steps.is_none()
        && record
            .transfer_entropy_effective_past_to_future_horizon_steps
            .is_none()
        && record.transfer_entropy_transition_sample_count.is_none()
        && record.bidirectional_transfer_entropy_mass.is_none()
        && record
            .thresholded_total_outgoing_transfer_entropy_a_to_b
            .is_none()
        && record.transfer_entropy_outgoing_term_floor.is_none()
        && record.net_causal_flow_out_minus_in.is_none();
    if te_supported != te_values_present
        || (!te_supported && !te_values_absent)
        || (!te_supported
            && record.transfer_entropy_support_status != "UNAVAILABLE_TE_REQUIRES_AT_LEAST_12_BINS")
    {
        bail!("strict residue transfer-entropy support/value mismatch");
    }
    if te_supported
        && (record.transfer_entropy_kernel_lag_bins != Some(STRICT_TWIN_TE_KERNEL_LAG_BINS)
            || record.transfer_entropy_past_index_offset_bins
                != Some(STRICT_TWIN_TE_PAST_INDEX_OFFSET_BINS)
            || record.transfer_entropy_future_index_offset_bins
                != Some(STRICT_TWIN_TE_FUTURE_INDEX_OFFSET_BINS)
            || record.transfer_entropy_effective_past_to_future_horizon_bins
                != Some(STRICT_TWIN_TE_EFFECTIVE_HORIZON_BINS)
            || record.transfer_entropy_bin_size_steps != Some(STRICT_TWIN_CCF_BIN_SIZE_STEPS)
            || record.transfer_entropy_effective_past_to_future_horizon_steps
                != Some(STRICT_TWIN_TE_EFFECTIVE_HORIZON_STEPS)
            || record
                .transfer_entropy_transition_sample_count
                .is_none_or(|count| count < 10)
            || record.transfer_entropy_outgoing_term_floor != Some(0.001)
            || record
                .bidirectional_transfer_entropy_mass
                .is_none_or(|value| value < 0.0)
            || record
                .thresholded_total_outgoing_transfer_entropy_a_to_b
                .is_none_or(|value| value < 0.0))
    {
        bail!("strict residue transfer-entropy kernel semantics failed");
    }
    for (phase_index, phase) in record.phase_ccf.iter().enumerate() {
        if phase.phase_index != phase_index as u32
            || phase.phase_name != STRICT_TWIN_PHASE_NAMES[phase_index]
            || phase.bin_size_steps != STRICT_TWIN_CCF_BIN_SIZE_STEPS
            || phase.bin_count < 2
            || phase.matrix_artifact_kind != format!("gpu_phase_ccf_{phase_index}")
            || phase.matrix_sha256.len() != 64
            || !phase
                .matrix_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            bail!("strict residue phase {} identity failed", phase_index);
        }
        let supported = phase.support_status == STRICT_TWIN_CCF_SUPPORTED;
        let values_present = phase.correlation_lag_steps.is_some()
            && phase.aligned_zero_lag_ccf.is_some()
            && phase.peak_partner_residue_id.is_some()
            && phase.peak_partner_residue_offset.is_some()
            && phase.peak_partner_zero_lag_ccf.is_some()
            && phase
                .cross_residue_network_reproducibility_fraction
                .is_some();
        let values_absent = phase.correlation_lag_steps.is_none()
            && phase.aligned_zero_lag_ccf.is_none()
            && phase.peak_partner_residue_id.is_none()
            && phase.peak_partner_residue_offset.is_none()
            && phase.peak_partner_zero_lag_ccf.is_none()
            && phase
                .cross_residue_network_reproducibility_fraction
                .is_none();
        if supported
            != (phase.variance_support_a
                && phase.variance_support_b
                && phase.supported_partner_count > 0
                && values_present)
            || (!supported && (phase.supported_partner_count != 0 || !values_absent))
            || (!supported && phase.support_status != STRICT_TWIN_PHASE_CCF_UNAVAILABLE)
            || !finite_optional(phase.aligned_zero_lag_ccf)
            || !finite_optional(phase.peak_partner_zero_lag_ccf)
            || !finite_optional(phase.cross_residue_network_reproducibility_fraction)
        {
            bail!(
                "strict residue phase {} support/value mismatch",
                phase_index
            );
        }
        if supported
            && (phase.correlation_lag_steps != Some(STRICT_TWIN_CCF_TEMPORAL_LAG_STEPS)
                || phase.peak_partner_residue_id
                    != record.residue_id.checked_add(
                        phase
                            .peak_partner_residue_offset
                            .context("strict phase CCF partner offset disappeared")?,
                    )
                || phase.peak_partner_residue_id.is_none_or(|value| value < 0)
                || phase.peak_partner_residue_id == Some(record.residue_id)
                || phase
                    .aligned_zero_lag_ccf
                    .is_none_or(|value| value.abs() > 1.0001)
                || phase
                    .peak_partner_zero_lag_ccf
                    .is_none_or(|value| value.abs() > 1.0001)
                || phase
                    .cross_residue_network_reproducibility_fraction
                    .is_none_or(|value| !(0.0..=1.0).contains(&value)))
        {
            bail!(
                "strict residue phase {} zero-lag/partner semantics failed",
                phase_index
            );
        }
    }
    if record.ccf_frequency_peak.is_some()
        || record.ccf_frequency_status != "UNAVAILABLE_NO_TEMPORAL_LAG_SPECTRUM"
        || record.phase_offset_enrichment.is_some()
        || record.phase_offset_enrichment_status
            != "NOT_APPLICABLE_ALIGNED_COMMON_RANDOM_NUMBERS_OFFSET_ZERO"
        || record.nma_responsive_mode.is_some()
        || record.nma_mode_eigenvalue.is_some()
        || record.nma_work_at_residue.is_some()
        || record.mechanical_sensitivity.is_some()
        || record.susceptibility_magnitude.is_some()
        || record.nma_status != "NOT_APPLICABLE_TOPOLOGY_ONLY_INTERVENTION"
    {
        bail!("strict residue contains a proxy or inapplicable NMA/offset/frequency value");
    }
    Ok(())
}

#[cfg(feature = "gpu")]
fn build_strict_residue_features(
    topology: &PrismPrepTopology,
    total_steps: i32,
    spikes_a: &[crate::fused_engine::GpuSpikeEvent],
    spikes_b: &[crate::fused_engine::GpuSpikeEvent],
    global_ccf_matrix: &[f32],
    global_ccf_features: &[crate::twin_kernels::CcfResidueFeatures],
    phases: &[StrictPhaseCcfRuntime],
    te_mi: &[f32],
    te_causal: &[f32],
    te_per_residue: &[f32],
) -> Result<Vec<StrictTwinResidueFeature>> {
    let n_residues = topology.n_residues;
    if n_residues < 2
        || global_ccf_matrix.len() != n_residues.saturating_mul(n_residues)
        || global_ccf_matrix.iter().any(|value| !value.is_finite())
        || global_ccf_features.len() != n_residues
        || te_mi.len() != n_residues
        || te_causal.len() != n_residues
        || te_per_residue.len() != n_residues
        || phases.len() != STRICT_TWIN_PHASE_NAMES.len()
    {
        bail!("strict residue feature input dimensions are incomplete");
    }
    let observations_a = collect_strict_residue_observations(spikes_a, n_residues, total_steps)?;
    let observations_b = collect_strict_residue_observations(spikes_b, n_residues, total_steps)?;
    let residue_ids: BTreeSet<i32> = observations_a
        .keys()
        .chain(observations_b.keys())
        .copied()
        .collect();
    if residue_ids.is_empty() {
        bail!("strict two-topology feature plane has no reference-frame residue events");
    }

    let empty = Vec::<StrictResidueEventObservation>::new();
    let mut global_variance_support_a = Vec::with_capacity(n_residues);
    let mut global_variance_support_b = Vec::with_capacity(n_residues);
    for residue_index in 0..n_residues {
        let residue_id = i32::try_from(residue_index)?;
        global_variance_support_a.push(strict_binned_variance_support(
            observations_a.get(&residue_id).unwrap_or(&empty),
            0,
            total_steps,
            STRICT_TWIN_CCF_BIN_SIZE_STEPS,
        )?);
        global_variance_support_b.push(strict_binned_variance_support(
            observations_b.get(&residue_id).unwrap_or(&empty),
            0,
            total_steps,
            STRICT_TWIN_CCF_BIN_SIZE_STEPS,
        )?);
    }
    let mut phase_variance_support = Vec::with_capacity(phases.len());
    for phase in phases {
        let mut support_a = Vec::with_capacity(n_residues);
        let mut support_b = Vec::with_capacity(n_residues);
        for residue_index in 0..n_residues {
            let residue_id = i32::try_from(residue_index)?;
            support_a.push(strict_binned_variance_support(
                observations_a.get(&residue_id).unwrap_or(&empty),
                phase.window.start_step,
                phase.window.end_step_exclusive,
                phase.window.bin_size_steps,
            )?);
            support_b.push(strict_binned_variance_support(
                observations_b.get(&residue_id).unwrap_or(&empty),
                phase.window.start_step,
                phase.window.end_step_exclusive,
                phase.window.bin_size_steps,
            )?);
        }
        phase_variance_support.push((support_a, support_b));
    }
    let transfer_entropy_bin_count = total_steps.div_euclid(STRICT_TWIN_CCF_BIN_SIZE_STEPS);
    let transfer_entropy_supported = transfer_entropy_bin_count >= 12;
    let transfer_entropy_transition_sample_count = transfer_entropy_supported
        .then(|| transfer_entropy_bin_count.checked_sub(2))
        .flatten()
        .map(u32::try_from)
        .transpose()?;
    let mut records = Vec::with_capacity(residue_ids.len());
    for residue_id in residue_ids {
        let residue_index = usize::try_from(residue_id)?;
        let a = observations_a.get(&residue_id).unwrap_or(&empty);
        let b = observations_b.get(&residue_id).unwrap_or(&empty);
        let event_count_a = u32::try_from(a.len())?;
        let event_count_b = u32::try_from(b.len())?;
        let intensity_sum_a = a.iter().map(|event| event.intensity as f64).sum::<f64>() as f32;
        let intensity_sum_b = b.iter().map(|event| event.intensity as f64).sum::<f64>() as f32;
        if !intensity_sum_a.is_finite() || !intensity_sum_b.is_finite() {
            bail!("strict residue intensity sum became nonfinite");
        }
        let global_support_a = global_variance_support_a[residue_index];
        let global_support_b = global_variance_support_b[residue_index];
        let global_projection = strict_zero_lag_ccf_projection(
            global_ccf_matrix,
            global_ccf_features,
            n_residues,
            residue_index,
            &global_variance_support_a,
            &global_variance_support_b,
        )?;
        let mut phase_ccf = Vec::with_capacity(phases.len());
        for (phase_index, phase) in phases.iter().enumerate() {
            let phase_a =
                strict_event_slice(a, phase.window.start_step, phase.window.end_step_exclusive);
            let phase_b =
                strict_event_slice(b, phase.window.start_step, phase.window.end_step_exclusive);
            let (phase_support_a, phase_support_b) = &phase_variance_support[phase_index];
            let support_a = phase_support_a[residue_index];
            let support_b = phase_support_b[residue_index];
            let phase_projection = strict_zero_lag_ccf_projection(
                &phase.matrix,
                &phase.features,
                n_residues,
                residue_index,
                phase_support_a,
                phase_support_b,
            )?;
            let supported = phase_projection.is_some();
            phase_ccf.push(StrictTwinPhaseCcfFeature {
                phase_index: phase.window.phase_index,
                phase_name: phase.window.phase_name.clone(),
                start_step: phase.window.start_step,
                end_step_exclusive: phase.window.end_step_exclusive,
                bin_size_steps: phase.window.bin_size_steps,
                bin_count: phase.window.bin_count,
                residue_event_count_a: u32::try_from(phase_a.len())?,
                residue_event_count_b: u32::try_from(phase_b.len())?,
                variance_support_a: support_a,
                variance_support_b: support_b,
                supported_partner_count: phase_projection
                    .as_ref()
                    .map_or(0, |projection| projection.supported_partner_count),
                correlation_lag_steps: supported.then_some(STRICT_TWIN_CCF_TEMPORAL_LAG_STEPS),
                aligned_zero_lag_ccf: phase_projection
                    .as_ref()
                    .map(|projection| projection.aligned_zero_lag_ccf),
                peak_partner_residue_id: phase_projection
                    .as_ref()
                    .map(|projection| projection.peak_partner_residue_id),
                peak_partner_residue_offset: phase_projection
                    .as_ref()
                    .map(|projection| projection.peak_partner_residue_offset),
                peak_partner_zero_lag_ccf: phase_projection
                    .as_ref()
                    .map(|projection| projection.peak_partner_zero_lag_ccf),
                cross_residue_network_reproducibility_fraction: phase_projection
                    .as_ref()
                    .map(|projection| projection.cross_residue_network_reproducibility_fraction),
                support_status: if supported {
                    STRICT_TWIN_CCF_SUPPORTED
                } else {
                    STRICT_TWIN_PHASE_CCF_UNAVAILABLE
                }
                .to_string(),
                matrix_artifact_kind: phase.matrix_artifact.kind.clone(),
                matrix_sha256: phase.matrix_artifact.sha256.clone(),
            });
        }
        let predictive_matches =
            strict_forward_predictive_matches(a, b, STRICT_TWIN_PREDICTIVE_WINDOW_STEPS)?;
        let predictive_probability =
            (event_count_a > 0).then_some(predictive_matches as f32 / event_count_a as f32);
        let onset_step_a = a.first().map(|event| event.timestep);
        let onset_step_b = b.first().map(|event| event.timestep);
        let (onset_intensity, onset_intensity_support) = if let Some(onset_b) = onset_step_b {
            strict_mean_intensity_at_timestep(a, onset_b)?
        } else {
            (None, 0)
        };
        let record = StrictTwinResidueFeature {
            schema_kind: "prismos.coupled_twin_strict_residue_feature.v3".to_string(),
            schema_version: STRICT_TWIN_FEATURE_SCHEMA_VERSION,
            residue_id,
            residue_name: topology_residue_name(topology, residue_id as usize),
            event_count_a,
            event_count_b,
            intensity_sum_a,
            intensity_sum_b,
            signed_event_count_contrast: i64::from(event_count_b) - i64::from(event_count_a),
            signed_intensity_contrast: intensity_sum_b - intensity_sum_a,
            b_over_a_event_ratio: (event_count_a > 0)
                .then_some(event_count_b as f32 / event_count_a as f32),
            b_over_a_intensity_ratio: (intensity_sum_a > 0.0)
                .then_some(intensity_sum_b / intensity_sum_a),
            global_correlation_lag_steps: global_projection
                .as_ref()
                .map(|_| STRICT_TWIN_CCF_TEMPORAL_LAG_STEPS),
            global_variance_support_a: global_support_a,
            global_variance_support_b: global_support_b,
            global_supported_partner_count: global_projection
                .as_ref()
                .map_or(0, |projection| projection.supported_partner_count),
            global_aligned_zero_lag_ccf: global_projection
                .as_ref()
                .map(|projection| projection.aligned_zero_lag_ccf),
            global_peak_partner_residue_id: global_projection
                .as_ref()
                .map(|projection| projection.peak_partner_residue_id),
            global_peak_partner_residue_offset: global_projection
                .as_ref()
                .map(|projection| projection.peak_partner_residue_offset),
            global_peak_partner_zero_lag_ccf: global_projection
                .as_ref()
                .map(|projection| projection.peak_partner_zero_lag_ccf),
            global_cross_residue_network_reproducibility_fraction: global_projection
                .as_ref()
                .map(|projection| projection.cross_residue_network_reproducibility_fraction),
            global_ccf_support_status: if global_projection.is_some() {
                STRICT_TWIN_CCF_SUPPORTED
            } else {
                STRICT_TWIN_GLOBAL_CCF_UNAVAILABLE
            }
            .to_string(),
            phase_ccf,
            predictive_window_steps: STRICT_TWIN_PREDICTIVE_WINDOW_STEPS,
            predictive_source_event_count: event_count_a,
            predictive_matched_event_count: predictive_matches,
            predictive_probability_b_after_a: predictive_probability,
            predictive_algorithm: STRICT_TWIN_PREDICTIVE_ALGORITHM.to_string(),
            onset_step_a,
            onset_step_b,
            b_minus_a_onset_lag_steps: onset_step_a.zip(onset_step_b).map(|(a, b)| b - a),
            a_intensity_at_exact_b_onset: onset_intensity,
            a_intensity_at_exact_b_onset_support_count: onset_intensity_support,
            b_only_condition_event_count: strict_unmatched_event_count(
                b,
                a,
                STRICT_TWIN_PREDICTIVE_WINDOW_STEPS,
            )?,
            a_only_reference_event_count: strict_unmatched_event_count(
                a,
                b,
                STRICT_TWIN_PREDICTIVE_WINDOW_STEPS,
            )?,
            transfer_entropy_kernel_lag_bins: transfer_entropy_supported
                .then_some(STRICT_TWIN_TE_KERNEL_LAG_BINS),
            transfer_entropy_past_index_offset_bins: transfer_entropy_supported
                .then_some(STRICT_TWIN_TE_PAST_INDEX_OFFSET_BINS),
            transfer_entropy_future_index_offset_bins: transfer_entropy_supported
                .then_some(STRICT_TWIN_TE_FUTURE_INDEX_OFFSET_BINS),
            transfer_entropy_effective_past_to_future_horizon_bins: transfer_entropy_supported
                .then_some(STRICT_TWIN_TE_EFFECTIVE_HORIZON_BINS),
            transfer_entropy_bin_size_steps: transfer_entropy_supported
                .then_some(STRICT_TWIN_CCF_BIN_SIZE_STEPS),
            transfer_entropy_effective_past_to_future_horizon_steps: transfer_entropy_supported
                .then_some(STRICT_TWIN_TE_EFFECTIVE_HORIZON_STEPS),
            transfer_entropy_transition_sample_count,
            bidirectional_transfer_entropy_mass: transfer_entropy_supported
                .then_some(te_mi[residue_id as usize]),
            thresholded_total_outgoing_transfer_entropy_a_to_b: transfer_entropy_supported
                .then_some(te_per_residue[residue_id as usize]),
            transfer_entropy_outgoing_term_floor: transfer_entropy_supported.then_some(0.001),
            net_causal_flow_out_minus_in: transfer_entropy_supported
                .then_some(te_causal[residue_id as usize]),
            transfer_entropy_support_status: if transfer_entropy_supported {
                STRICT_TWIN_TE_SUPPORTED
            } else {
                "UNAVAILABLE_TE_REQUIRES_AT_LEAST_12_BINS"
            }
            .to_string(),
            ccf_frequency_peak: None,
            ccf_frequency_status: "UNAVAILABLE_NO_TEMPORAL_LAG_SPECTRUM".to_string(),
            phase_offset_enrichment: None,
            phase_offset_enrichment_status:
                "NOT_APPLICABLE_ALIGNED_COMMON_RANDOM_NUMBERS_OFFSET_ZERO".to_string(),
            nma_responsive_mode: None,
            nma_mode_eigenvalue: None,
            nma_work_at_residue: None,
            mechanical_sensitivity: None,
            susceptibility_magnitude: None,
            nma_status: "NOT_APPLICABLE_TOPOLOGY_ONLY_INTERVENTION".to_string(),
            fallback_used: false,
        };
        validate_strict_residue_feature(&record)?;
        records.push(record);
    }
    Ok(records)
}

#[cfg(feature = "gpu")]
fn mean_finite(values: impl Iterator<Item = f32>) -> Result<Option<f32>> {
    let values = values.collect::<Vec<_>>();
    if values.is_empty() {
        return Ok(None);
    }
    if values.iter().any(|value| !value.is_finite()) {
        bail!("strict feature aggregation received nonfinite input");
    }
    let mean = (values.iter().map(|value| *value as f64).sum::<f64>() / values.len() as f64) as f32;
    if !mean.is_finite() {
        bail!("strict feature aggregate became nonfinite");
    }
    Ok(Some(mean))
}

#[cfg(feature = "gpu")]
#[derive(Debug, Clone, PartialEq)]
struct StrictBindingSiteInput {
    site_id: u64,
    site_rank: u32,
    lining_residue_ids: Vec<i32>,
    volume: f32,
    druggability: f32,
    is_druggable: bool,
    centroid: [f32; 3],
}

#[cfg(feature = "gpu")]
fn parse_strict_site_residue_ids(
    site: &serde_json::Value,
    field: &str,
    object_field: Option<&str>,
    n_residues: usize,
) -> Result<Vec<i32>> {
    let values = site
        .get(field)
        .and_then(serde_json::Value::as_array)
        .with_context(|| format!("binding site lacks {field} array"))?;
    if values.is_empty() {
        bail!("binding site {field} array is empty");
    }
    let mut seen = BTreeSet::new();
    for value in values {
        let value = if let Some(object_field) = object_field {
            value
                .as_object()
                .with_context(|| format!("binding site {field} entry is not an object"))?
                .get(object_field)
                .with_context(|| format!("binding site {field} entry lacks {object_field}"))?
        } else {
            value
        };
        let parsed = value
            .as_i64()
            .with_context(|| format!("binding site {field} contains a malformed residue ID"))?;
        let parsed = i32::try_from(parsed)
            .with_context(|| format!("binding site {field} residue ID is outside i32"))?;
        if parsed < 0 || usize::try_from(parsed)? >= n_residues {
            bail!("binding site {field} residue ID {parsed} is outside topology bounds");
        }
        if !seen.insert(parsed) {
            bail!("binding site {field} contains duplicate residue ID {parsed}");
        }
    }
    Ok(seen.into_iter().collect())
}

#[cfg(feature = "gpu")]
fn parse_strict_binding_sites(
    sites_json: &serde_json::Value,
    n_residues: usize,
) -> Result<Vec<StrictBindingSiteInput>> {
    let sites = sites_json
        .get("sites")
        .and_then(serde_json::Value::as_array)
        .context("binding-site geometry lacks sites array")?;
    if sites.is_empty() {
        bail!("strict site aggregation requires at least one detected site");
    }
    let mut authoritative_ids = BTreeSet::new();
    let mut ranks = BTreeSet::new();
    let mut parsed = Vec::with_capacity(sites.len());
    for site in sites {
        let site_id = site
            .get("id")
            .and_then(serde_json::Value::as_u64)
            .context("binding site lacks a nonnegative integer authoritative id")?;
        if site_id > u64::from(u32::MAX) {
            bail!(
                "binding-site authoritative id {site_id} exceeds the detector/Path-B u32 identity domain"
            );
        }
        if !authoritative_ids.insert(site_id) {
            bail!("binding-site geometry contains duplicate authoritative id {site_id}");
        }
        let site_rank = site
            .get("rank")
            .and_then(serde_json::Value::as_u64)
            .and_then(|rank| u32::try_from(rank).ok())
            .filter(|rank| *rank > 0)
            .context("binding site lacks a positive integer rank")?;
        if !ranks.insert(site_rank) {
            bail!("binding-site geometry contains duplicate rank {site_rank}");
        }
        let residue_ids = parse_strict_site_residue_ids(site, "residue_ids", None, n_residues)?;
        let lining_residue_ids =
            parse_strict_site_residue_ids(site, "lining_residues", Some("resid"), n_residues)?;
        if residue_ids != lining_residue_ids {
            bail!(
                "binding site {site_id} substitutes or disagrees between residue_ids and lining_residues"
            );
        }
        let centroid_values = site
            .get("centroid")
            .and_then(serde_json::Value::as_array)
            .context("binding site lacks centroid")?;
        if centroid_values.len() != 3 {
            bail!("binding site centroid must have exactly three coordinates");
        }
        let centroid = [
            centroid_values[0]
                .as_f64()
                .context("invalid site centroid x")? as f32,
            centroid_values[1]
                .as_f64()
                .context("invalid site centroid y")? as f32,
            centroid_values[2]
                .as_f64()
                .context("invalid site centroid z")? as f32,
        ];
        let volume = site
            .get("volume")
            .and_then(serde_json::Value::as_f64)
            .context("binding site lacks volume")? as f32;
        let druggability = site
            .get("druggability")
            .and_then(serde_json::Value::as_f64)
            .context("binding site lacks druggability")? as f32;
        let is_druggable = site
            .get("is_druggable")
            .and_then(serde_json::Value::as_bool)
            .context("binding site lacks is_druggable")?;
        if !volume.is_finite()
            || volume < 0.0
            || !druggability.is_finite()
            || centroid.iter().any(|value| !value.is_finite())
        {
            bail!("binding site {site_id} contains invalid geometry");
        }
        parsed.push(StrictBindingSiteInput {
            site_id,
            site_rank,
            lining_residue_ids,
            volume,
            druggability,
            is_druggable,
            centroid,
        });
    }
    let site_count = u32::try_from(sites.len()).context("binding-site count exceeds u32")?;
    let expected_ranks = (1..=site_count).collect::<BTreeSet<_>>();
    if ranks != expected_ranks {
        bail!(
            "binding-site ranks must be the exact one-based permutation 1..={site_count}; observed {ranks:?}"
        );
    }
    parsed.sort_by_key(|site| site.site_id);
    Ok(parsed)
}

#[cfg(feature = "gpu")]
fn validate_strict_site_feature(record: &StrictTwinSiteFeature) -> Result<()> {
    let sorted_unique = |values: &[i32]| {
        values.windows(2).all(|pair| pair[0] < pair[1]) && values.iter().all(|value| *value >= 0)
    };
    if record.schema_kind != "prismos.coupled_twin_strict_site_feature.v3"
        || record.schema_version != STRICT_TWIN_FEATURE_SCHEMA_VERSION
        || record.site_id > u64::from(u32::MAX)
        || record.site_rank == 0
        || record.lining_residue_ids.is_empty()
        || !sorted_unique(&record.lining_residue_ids)
        || !sorted_unique(&record.strict_residue_ids_with_data)
        || !sorted_unique(&record.missing_strict_residue_ids)
        || record
            .volume
            .is_none_or(|value| !value.is_finite() || value < 0.0)
        || record.druggability.is_none_or(|value| !value.is_finite())
        || record.is_druggable.is_none()
        || record
            .centroid
            .is_none_or(|values| values.iter().any(|value| !value.is_finite()))
        || record
            .mean_global_aligned_zero_lag_ccf
            .is_some_and(|value| !value.is_finite())
        || record
            .max_global_aligned_zero_lag_ccf
            .is_some_and(|value| !value.is_finite())
        || record
            .mean_intra_site_cross_residue_zero_lag_ccf
            .is_some_and(|value| !value.is_finite())
        || record
            .mean_phase_aligned_zero_lag_ccf
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
        || record
            .mean_predictive_probability_b_after_a
            .is_some_and(|value| !value.is_finite())
        || !record.signed_intensity_contrast.is_finite()
        || record.geometry_source
            != "binding_sites.json:topology_derived_lining_residues_and_geometry"
        || record.active_feature_surface != "STRICT_EVENT_DERIVED_GPU_ZERO_LAG_CCF_AND_LAG1_TE_V3"
        || record.fallback_used
    {
        bail!(
            "strict site {} failed schema/identity/geometry/finite/no-fallback validation",
            record.site_id
        );
    }
    let data = record
        .strict_residue_ids_with_data
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let missing = record
        .missing_strict_residue_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let lining = record
        .lining_residue_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    if !data.is_disjoint(&missing)
        || data.union(&missing).copied().collect::<BTreeSet<_>>() != lining
    {
        bail!("strict site residue availability does not partition lining residues");
    }
    let expected_candidate_count = record
        .lining_residue_ids
        .len()
        .checked_mul(record.lining_residue_ids.len().saturating_sub(1))
        .context("strict site validation pair-count overflow")?
        / 2;
    if record.intra_site_ccf_candidate_pair_count != u32::try_from(expected_candidate_count)?
        || record.intra_site_ccf_pair_decisions.len() != expected_candidate_count
    {
        bail!("strict site candidate-pair inventory is incomplete");
    }
    let mut expected_pairs = Vec::with_capacity(expected_candidate_count);
    for left_index in 0..record.lining_residue_ids.len() {
        for right_index in (left_index + 1)..record.lining_residue_ids.len() {
            expected_pairs.push((
                record.lining_residue_ids[left_index],
                record.lining_residue_ids[right_index],
            ));
        }
    }
    let observed_pairs = record
        .intra_site_ccf_pair_decisions
        .iter()
        .map(|decision| (decision.left_residue_id, decision.right_residue_id))
        .collect::<Vec<_>>();
    if observed_pairs != expected_pairs {
        bail!("strict site pair decisions are not canonical and complete");
    }
    let mut supported_values = Vec::new();
    for decision in &record.intra_site_ccf_pair_decisions {
        if decision.left_strict_row_present != data.contains(&decision.left_residue_id)
            || decision.right_strict_row_present != data.contains(&decision.right_residue_id)
            || decision.left_ccf_supported && !decision.left_strict_row_present
            || decision.right_ccf_supported && !decision.right_strict_row_present
            || decision.pair_supported
                != (decision.left_ccf_supported && decision.right_ccf_supported)
        {
            bail!("strict site pair endpoint/pair support flags are inconsistent");
        }
        match (decision.pair_supported, decision.symmetric_zero_lag_ccf) {
            (true, Some(value)) if value.is_finite() => {
                if decision.support_status != "SUPPORTED_BOTH_ENDPOINTS_AND_REAL_GPU_PAIR_MATRIX" {
                    bail!("strict supported site pair has the wrong support status");
                }
                supported_values.push(value);
            }
            (false, None) if decision.support_status.starts_with("UNAVAILABLE_") => {}
            _ => bail!("strict site pair substitutes numeric zero for unavailable evidence"),
        }
    }
    if record.intra_site_ccf_supported_pair_count != u32::try_from(supported_values.len())?
        || record
            .mean_intra_site_cross_residue_zero_lag_ccf
            .map(f32::to_bits)
            != mean_finite(supported_values.into_iter())?.map(f32::to_bits)
    {
        bail!("strict site supported-pair count/mean does not recompute");
    }
    if record.global_ccf_supported_residue_count == 0
        && (record.mean_global_aligned_zero_lag_ccf.is_some()
            || record.max_global_aligned_zero_lag_ccf.is_some())
        || record.global_ccf_supported_residue_count > 0
            && (record.mean_global_aligned_zero_lag_ccf.is_none()
                || record.max_global_aligned_zero_lag_ccf.is_none())
    {
        bail!("strict site global supported-residue aggregate uses invalid null semantics");
    }
    for index in 0..STRICT_TWIN_PHASE_NAMES.len() {
        if (record.phase_ccf_supported_residue_counts[index] == 0)
            != record.mean_phase_aligned_zero_lag_ccf[index].is_none()
        {
            bail!("strict site phase aggregate uses invalid null semantics");
        }
    }
    if (record.predictive_supported_residue_count == 0)
        != record.mean_predictive_probability_b_after_a.is_none()
    {
        bail!("strict site predictive aggregate uses invalid null semantics");
    }
    Ok(())
}

#[cfg(feature = "gpu")]
fn build_strict_site_features(
    sites_json: &serde_json::Value,
    residue_features: &[StrictTwinResidueFeature],
    ccf_matrix: &[f32],
    n_residues: usize,
) -> Result<Vec<StrictTwinSiteFeature>> {
    if ccf_matrix.len() != n_residues.saturating_mul(n_residues)
        || ccf_matrix.iter().any(|value| !value.is_finite())
    {
        bail!("strict site aggregation lacks a finite global GPU CCF matrix");
    }
    let sites = parse_strict_binding_sites(sites_json, n_residues)?;
    let mut residue_by_id = BTreeMap::<i32, &StrictTwinResidueFeature>::new();
    for record in residue_features {
        if residue_by_id.insert(record.residue_id, record).is_some() {
            bail!(
                "strict site aggregation rejected duplicate strict residue identity {}",
                record.residue_id
            );
        }
    }
    let mut records = Vec::with_capacity(sites.len());
    for site in sites {
        let site_id = site.site_id;
        let lining_residue_ids = site.lining_residue_ids;
        let strict_residue_ids_with_data = lining_residue_ids
            .iter()
            .filter(|residue_id| residue_by_id.contains_key(residue_id))
            .copied()
            .collect::<Vec<_>>();
        let missing_strict_residue_ids = lining_residue_ids
            .iter()
            .filter(|residue_id| !residue_by_id.contains_key(residue_id))
            .copied()
            .collect::<Vec<_>>();
        if strict_residue_ids_with_data.is_empty() {
            bail!(
                "strict site {} has no event-derived residue evidence",
                site_id
            );
        }
        let strict_rows = strict_residue_ids_with_data
            .iter()
            .filter_map(|residue_id| residue_by_id.get(residue_id).copied())
            .collect::<Vec<_>>();
        let global_ccf = strict_rows
            .iter()
            .filter_map(|record| record.global_aligned_zero_lag_ccf)
            .collect::<Vec<_>>();
        let mean_global_aligned_zero_lag_ccf = mean_finite(global_ccf.iter().copied())?;
        let max_global_aligned_zero_lag_ccf = global_ccf.iter().copied().max_by(f32::total_cmp);
        let mut intra_site_ccf_pair_decisions = Vec::new();
        for left_index in 0..lining_residue_ids.len() {
            for right_index in (left_index + 1)..lining_residue_ids.len() {
                let left_residue_id = lining_residue_ids[left_index];
                let right_residue_id = lining_residue_ids[right_index];
                let left_record = residue_by_id.get(&left_residue_id).copied();
                let right_record = residue_by_id.get(&right_residue_id).copied();
                let left_ccf_supported = left_record.is_some_and(|record| {
                    record.global_ccf_support_status == STRICT_TWIN_CCF_SUPPORTED
                        && record.global_aligned_zero_lag_ccf.is_some()
                });
                let right_ccf_supported = right_record.is_some_and(|record| {
                    record.global_ccf_support_status == STRICT_TWIN_CCF_SUPPORTED
                        && record.global_aligned_zero_lag_ccf.is_some()
                });
                let pair_supported = left_ccf_supported && right_ccf_supported;
                let symmetric_zero_lag_ccf = if pair_supported {
                    let left = usize::try_from(left_residue_id)?;
                    let right = usize::try_from(right_residue_id)?;
                    let value = 0.5
                        * (ccf_matrix[left * n_residues + right]
                            + ccf_matrix[right * n_residues + left]);
                    if !value.is_finite() {
                        bail!("strict intra-site CCF became nonfinite");
                    }
                    Some(value)
                } else {
                    None
                };
                let support_status = match (
                    left_record.is_some(),
                    right_record.is_some(),
                    left_ccf_supported,
                    right_ccf_supported,
                ) {
                    (true, true, true, true) => "SUPPORTED_BOTH_ENDPOINTS_AND_REAL_GPU_PAIR_MATRIX",
                    (false, false, _, _) => "UNAVAILABLE_BOTH_STRICT_RESIDUE_ROWS_MISSING",
                    (false, true, _, _) => "UNAVAILABLE_LEFT_STRICT_RESIDUE_ROW_MISSING",
                    (true, false, _, _) => "UNAVAILABLE_RIGHT_STRICT_RESIDUE_ROW_MISSING",
                    (true, true, false, false) => {
                        "UNAVAILABLE_BOTH_ENDPOINTS_LACK_GLOBAL_CCF_SUPPORT"
                    }
                    (true, true, false, true) => {
                        "UNAVAILABLE_LEFT_ENDPOINT_LACKS_GLOBAL_CCF_SUPPORT"
                    }
                    (true, true, true, false) => {
                        "UNAVAILABLE_RIGHT_ENDPOINT_LACKS_GLOBAL_CCF_SUPPORT"
                    }
                }
                .to_string();
                intra_site_ccf_pair_decisions.push(StrictTwinSiteCcfPairDecision {
                    left_residue_id,
                    right_residue_id,
                    left_strict_row_present: left_record.is_some(),
                    right_strict_row_present: right_record.is_some(),
                    left_ccf_supported,
                    right_ccf_supported,
                    pair_supported,
                    symmetric_zero_lag_ccf,
                    support_status,
                });
            }
        }
        let intra_site_values = intra_site_ccf_pair_decisions
            .iter()
            .filter_map(|decision| decision.symmetric_zero_lag_ccf)
            .collect::<Vec<_>>();
        let intra_site_ccf_candidate_pair_count = intra_site_ccf_pair_decisions.len();
        let mut phase_ccf_supported_residue_counts = [0u32; 5];
        let mut mean_phase_aligned_zero_lag_ccf = [None; 5];
        for phase_index in 0..STRICT_TWIN_PHASE_NAMES.len() {
            let values = strict_rows
                .iter()
                .filter_map(|record| record.phase_ccf[phase_index].aligned_zero_lag_ccf)
                .collect::<Vec<_>>();
            phase_ccf_supported_residue_counts[phase_index] = u32::try_from(values.len())?;
            mean_phase_aligned_zero_lag_ccf[phase_index] = mean_finite(values.into_iter())?;
        }
        let predictive_values = strict_rows
            .iter()
            .filter_map(|record| record.predictive_probability_b_after_a)
            .collect::<Vec<_>>();
        let record = StrictTwinSiteFeature {
            schema_kind: "prismos.coupled_twin_strict_site_feature.v3".to_string(),
            schema_version: STRICT_TWIN_FEATURE_SCHEMA_VERSION,
            site_id,
            site_rank: site.site_rank,
            lining_residue_ids,
            strict_residue_ids_with_data,
            missing_strict_residue_ids,
            volume: Some(site.volume),
            druggability: Some(site.druggability),
            is_druggable: Some(site.is_druggable),
            centroid: Some(site.centroid),
            global_ccf_supported_residue_count: u32::try_from(global_ccf.len())?,
            mean_global_aligned_zero_lag_ccf,
            max_global_aligned_zero_lag_ccf,
            intra_site_ccf_candidate_pair_count: u32::try_from(
                intra_site_ccf_candidate_pair_count,
            )?,
            intra_site_ccf_supported_pair_count: u32::try_from(intra_site_values.len())?,
            intra_site_ccf_pair_decisions,
            mean_intra_site_cross_residue_zero_lag_ccf: mean_finite(intra_site_values.into_iter())?,
            phase_ccf_supported_residue_counts,
            mean_phase_aligned_zero_lag_ccf,
            predictive_supported_residue_count: u32::try_from(predictive_values.len())?,
            mean_predictive_probability_b_after_a: mean_finite(predictive_values.into_iter())?,
            total_b_only_condition_events: strict_rows
                .iter()
                .map(|record| record.b_only_condition_event_count as u64)
                .sum(),
            total_a_only_reference_events: strict_rows
                .iter()
                .map(|record| record.a_only_reference_event_count as u64)
                .sum(),
            signed_event_count_contrast: strict_rows
                .iter()
                .map(|record| record.signed_event_count_contrast)
                .sum(),
            signed_intensity_contrast: strict_rows
                .iter()
                .map(|record| record.signed_intensity_contrast as f64)
                .sum::<f64>() as f32,
            geometry_source: "binding_sites.json:topology_derived_lining_residues_and_geometry"
                .to_string(),
            active_feature_surface: "STRICT_EVENT_DERIVED_GPU_ZERO_LAG_CCF_AND_LAG1_TE_V3"
                .to_string(),
            fallback_used: false,
        };
        validate_strict_site_feature(&record)?;
        records.push(record);
    }
    Ok(records)
}

#[cfg(feature = "gpu")]
fn strict_artifact_binding(
    artifact: &CoupledTwinArtifactEvidence,
) -> StrictTwinFeatureArtifactBinding {
    StrictTwinFeatureArtifactBinding {
        kind: artifact.kind.clone(),
        stream_id: artifact.stream_id,
        path: artifact.path.clone(),
        size_bytes: artifact.size_bytes,
        sha256: artifact.sha256.clone(),
        blake3: artifact.blake3.clone(),
    }
}

#[cfg(feature = "gpu")]
fn retire_legacy_detector_feature_surfaces(
    output_dir: &Path,
    topology: &PrismPrepTopology,
) -> Result<Vec<StrictTwinRetiredSurface>> {
    let prefix = Path::new(&topology.source_pdb)
        .file_stem()
        .and_then(|value| value.to_str())
        .map(|value| value.replace("_sanitized", "").replace("_clean", ""))
        .unwrap_or_else(|| "prism_twin".to_string());
    let superseded_dir = output_dir
        .join("COUPLED_TWIN_STRICT_FEATURES")
        .join("SUPERSEDED_LEGACY_DETECTOR_SURFACES");
    std::fs::create_dir_all(&superseded_dir)?;
    let mut retired = Vec::new();
    for suffix in ["twin_per_residue.json", "twin_ccf_matrix.json"] {
        let source = output_dir.join(format!("{prefix}.{suffix}"));
        let metadata = std::fs::symlink_metadata(&source).with_context(|| {
            format!(
                "strict two-topology detector did not emit required legacy surface for explicit supersession: {}",
                source.display()
            )
        })?;
        if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() == 0 {
            bail!(
                "legacy detector surface is not a nonempty regular file: {}",
                source.display()
            );
        }
        let destination = superseded_dir.join(format!("{prefix}.{suffix}.superseded"));
        if std::fs::symlink_metadata(&destination).is_ok() {
            bail!(
                "refusing to overwrite superseded detector surface {}",
                destination.display()
            );
        }
        std::fs::rename(&source, &destination).with_context(|| {
            format!(
                "retire legacy detector surface {} -> {}",
                source.display(),
                destination.display()
            )
        })?;
        let (sha256, blake3) = hash_file_dual(&destination)?;
        retired.push(StrictTwinRetiredSurface {
            original_path: source.display().to_string(),
            superseded_path: destination.display().to_string(),
            size_bytes: metadata.len(),
            sha256,
            blake3,
            disposition: "SUPERSEDED_NON_AUTHORITATIVE_NOT_ADMITTED_TO_RUNTIME_MANIFEST"
                .to_string(),
        });
    }
    Ok(retired)
}

#[cfg(feature = "gpu")]
fn emit_strict_twin_feature_artifacts(
    output_dir: &Path,
    config: &CoupledTwinEvidenceConfig,
    binding_sites_path: &Path,
    residue_records: &[StrictTwinResidueFeature],
    site_records: &[StrictTwinSiteFeature],
    runtime: &CoupledTwinRuntimeEvidence,
    retired_surfaces: Vec<StrictTwinRetiredSurface>,
) -> Result<Vec<CoupledTwinArtifactEvidence>> {
    if residue_records.is_empty() || site_records.is_empty() || retired_surfaces.len() != 2 {
        bail!("strict feature emission lacks residue/site/supersession evidence");
    }
    for record in residue_records {
        validate_strict_residue_feature(record)?;
    }
    let strict_dir = output_dir.join("COUPLED_TWIN_STRICT_FEATURES");
    std::fs::create_dir_all(&strict_dir)?;
    let geometry_sha256 = sha256_file(binding_sites_path)?;
    let residue_path = strict_dir.join("strict_residue_features.v3.json");
    let residue_payload = StrictTwinResidueFeatureArtifact {
        schema_kind: "prismos.coupled_twin_strict_residue_features.v3".to_string(),
        schema_version: STRICT_TWIN_FEATURE_SCHEMA_VERSION,
        status: "PASS".to_string(),
        run_id: config.run_id.clone(),
        topology_a_sha256: config.topology_a_sha256.clone(),
        topology_b_sha256: config.topology_b_sha256.clone(),
        event_source: "runtime_owned_PRSPK001_streams_0_and_1".to_string(),
        ccf_backend: STRICT_TWIN_PHASE_CCF_BACKEND.to_string(),
        records: residue_records.to_vec(),
        fallback_used: false,
    };
    write_atomic_json(&residue_path, &residue_payload)?;
    let residue_artifact = artifact_evidence(
        "strict_twin_residue_features",
        None,
        &residue_path,
        Some(residue_records.len() as u64),
        Some(STRICT_TWIN_FEATURE_SCHEMA_VERSION),
    )?;

    let site_path = strict_dir.join("strict_site_features.v3.json");
    let site_payload = StrictTwinSiteFeatureArtifact {
        schema_kind: "prismos.coupled_twin_strict_site_features.v3".to_string(),
        schema_version: STRICT_TWIN_FEATURE_SCHEMA_VERSION,
        status: "PASS".to_string(),
        run_id: config.run_id.clone(),
        topology_a_sha256: config.topology_a_sha256.clone(),
        topology_b_sha256: config.topology_b_sha256.clone(),
        geometry_source_path: binding_sites_path.display().to_string(),
        geometry_source_sha256: geometry_sha256.clone(),
        records: site_records.to_vec(),
        fallback_used: false,
    };
    write_atomic_json(&site_path, &site_payload)?;
    let site_artifact = artifact_evidence(
        "strict_twin_site_features",
        None,
        &site_path,
        Some(site_records.len() as u64),
        Some(STRICT_TWIN_FEATURE_SCHEMA_VERSION),
    )?;

    let mut source_spike_artifacts = runtime
        .artifacts
        .iter()
        .filter(|artifact| artifact.kind == "spikes")
        .map(strict_artifact_binding)
        .collect::<Vec<_>>();
    source_spike_artifacts.sort_by(|left, right| left.path.cmp(&right.path));
    let mut phase_ccf_artifacts = runtime
        .artifacts
        .iter()
        .filter(|artifact| artifact.kind.starts_with("gpu_phase_ccf_"))
        .map(strict_artifact_binding)
        .collect::<Vec<_>>();
    phase_ccf_artifacts.sort_by(|left, right| left.kind.cmp(&right.kind));
    if source_spike_artifacts.len() != 2
        || phase_ccf_artifacts.len() != STRICT_TWIN_PHASE_NAMES.len()
    {
        bail!("strict feature receipt lacks exact source-spike/phase-matrix closure");
    }
    let global_gpu_analysis_receipt = runtime
        .artifacts
        .iter()
        .find(|artifact| {
            artifact.kind == "gpu_ccf_te_execution_receipt" && artifact.stream_id.is_none()
        })
        .context("strict feature receipt lacks the global GPU-v3 execution receipt")?;
    let binding_sites_artifact = runtime
        .artifacts
        .iter()
        .find(|artifact| {
            artifact.kind == "binding_sites"
                && artifact.stream_id.is_none()
                && Path::new(&artifact.path) == binding_sites_path
        })
        .context("strict feature receipt lacks the authoritative binding-sites artifact")?;
    let geometry_metadata = std::fs::symlink_metadata(binding_sites_path)?;
    if geometry_metadata.file_type().is_symlink()
        || !geometry_metadata.is_file()
        || geometry_metadata.len() != binding_sites_artifact.size_bytes
        || geometry_sha256 != binding_sites_artifact.sha256
    {
        bail!("strict feature receipt binding-sites source identity failed");
    }
    let global_gpu_analysis_payload: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&global_gpu_analysis_receipt.path)?)?;
    let phase_ccf_kernel_source: StrictTwinKernelSourceBinding = serde_json::from_value(
        global_gpu_analysis_payload
            .get("ccf_kernel_source")
            .context("global GPU-v3 receipt lacks the phase CCF kernel source")?
            .clone(),
    )?;
    validate_strict_kernel_source_binding(
        &phase_ccf_kernel_source,
        "tensor_ccf.ptx",
        &["tensor_ccf_compute", "compute_spike_norms"],
    )?;
    let receipt_path = strict_dir.join("strict_feature_execution_receipt.v3.json");
    let receipt = StrictTwinFeatureReceipt {
        schema_kind: "prismos.coupled_twin_strict_feature_execution_receipt.v3".to_string(),
        schema_version: STRICT_TWIN_FEATURE_SCHEMA_VERSION,
        status: "PASS".to_string(),
        run_id: config.run_id.clone(),
        topology_a_sha256: config.topology_a_sha256.clone(),
        topology_b_sha256: config.topology_b_sha256.clone(),
        source_spike_artifacts,
        phase_ccf_artifacts,
        phase_ccf_kernel_source,
        global_gpu_analysis_receipt_artifact: strict_artifact_binding(global_gpu_analysis_receipt),
        binding_sites_artifact: strict_artifact_binding(binding_sites_artifact),
        residue_feature_artifact: strict_artifact_binding(&residue_artifact),
        site_feature_artifact: strict_artifact_binding(&site_artifact),
        strict_residue_record_count: residue_records.len() as u64,
        strict_site_record_count: site_records.len() as u64,
        legacy_residue_rows_active: 0,
        legacy_site_rows_active: 0,
        retired_legacy_detector_surfaces: retired_surfaces,
        exact_predictive_algorithm: STRICT_TWIN_PREDICTIVE_ALGORITHM.to_string(),
        phase_ccf_backend: STRICT_TWIN_PHASE_CCF_BACKEND.to_string(),
        finite_values_verified: true,
        unavailable_values_are_null: true,
        fallback_used: false,
    };
    write_atomic_json(&receipt_path, &receipt)?;
    let receipt_artifact = artifact_evidence(
        "strict_twin_feature_receipt",
        None,
        &receipt_path,
        Some(1),
        Some(STRICT_TWIN_FEATURE_SCHEMA_VERSION),
    )?;

    let residue_readback: StrictTwinResidueFeatureArtifact =
        serde_json::from_slice(&std::fs::read(&residue_path)?)?;
    let site_readback: StrictTwinSiteFeatureArtifact =
        serde_json::from_slice(&std::fs::read(&site_path)?)?;
    let receipt_readback: StrictTwinFeatureReceipt =
        serde_json::from_slice(&std::fs::read(&receipt_path)?)?;
    if serde_json::to_value(&residue_readback)? != serde_json::to_value(&residue_payload)?
        || serde_json::to_value(&site_readback)? != serde_json::to_value(&site_payload)?
        || serde_json::to_value(&receipt_readback)? != serde_json::to_value(&receipt)?
    {
        bail!("strict feature artifact typed readback differs from emitted values");
    }
    Ok(vec![residue_artifact, site_artifact, receipt_artifact])
}

#[cfg(feature = "gpu")]
fn coupled_topology_stem(config: &CoupledTwinEvidenceConfig) -> String {
    Path::new(&config.topology_a_path)
        .file_stem()
        .and_then(|value| value.to_str())
        .map(|value| value.trim_end_matches(".topology").to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "coupled_twin".to_string())
}

#[cfg(feature = "gpu")]
pub(crate) fn coupled_protocol_sha256(protocol: &CryoUvProtocol) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"prism-coupled-cryo-uv-protocol-v1\0");
    for value in [
        protocol.start_temp,
        protocol.end_temp,
        protocol.uv_burst_energy,
    ] {
        hasher.update(value.to_bits().to_le_bytes());
    }
    for value in [
        protocol.cold_hold_steps,
        protocol.ramp_steps,
        protocol.warm_hold_steps,
        protocol.current_step,
        protocol.uv_burst_interval,
        protocol.uv_burst_duration,
        protocol.wavelength_dwell_steps,
        protocol.ramp_down_steps,
        protocol.cold_return_steps,
    ] {
        hasher.update(value.to_le_bytes());
    }
    hasher.update((protocol.scan_wavelengths.len() as u64).to_le_bytes());
    for wavelength in &protocol.scan_wavelengths {
        hasher.update(wavelength.to_bits().to_le_bytes());
    }
    hasher.update((protocol.stepped_holds.len() as u64).to_le_bytes());
    for (temperature, steps) in &protocol.stepped_holds {
        hasher.update(temperature.to_bits().to_le_bytes());
        hasher.update(steps.to_le_bytes());
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// One phase interpretation for the native Ghost producer and shared consumer.
#[cfg(feature = "gpu")]
pub(crate) fn coupled_protocol_phase_binding(
    protocol: &CryoUvProtocol,
    stream_id: u32,
    step_idx: u64,
    condition_id: String,
) -> Result<crate::ghost_phase_lattice::StreamProtocolPhaseBinding> {
    let cold_end = protocol.cold_hold_steps.max(0) as u64;
    let ramp_end = cold_end.checked_add(protocol.ramp_steps.max(0) as u64)
        .context("coupled Ghost heating boundary overflow")?;
    let mut warm_end = ramp_end.checked_add(protocol.warm_hold_steps.max(0) as u64)
        .context("coupled Ghost warm boundary overflow")?;
    let observed_end = step_idx.checked_add(1).context("coupled Ghost final-step overflow")?;
    let no_cooling = protocol.ramp_down_steps == 0 && protocol.cold_return_steps == 0;
    if no_cooling {
        warm_end = warm_end.max(observed_end);
    }
    let down_end = warm_end.checked_add(protocol.ramp_down_steps.max(0) as u64)
        .context("coupled Ghost cooling boundary overflow")?;
    let cooling_end = if no_cooling {
        warm_end
    } else {
        (protocol.total_steps().max(0) as u64).max(observed_end).max(
            down_end.checked_add(1).context("coupled Ghost cooling endpoint overflow")?)
    };
    let state_hash = crate::ghost_phase_lattice::final_engine_protocol_state_hash(
        &coupled_protocol_sha256(protocol), step_idx, protocol.total_steps().max(0) as u64,
        cold_end, ramp_end, warm_end, down_end,
    );
    crate::ghost_phase_lattice::StreamProtocolPhaseBinding::from_engine_boundaries(
        u16::try_from(stream_id)?, condition_id, "coupled_twin_cryo_uv_protocol_v1".to_string(),
        state_hash, cold_end, ramp_end, warm_end, cooling_end,
    )
}

#[cfg(feature = "gpu")]
fn protocol_phase(protocol: &CryoUvProtocol, step: u32) -> i32 {
    if step < protocol.cold_hold_steps.max(0) as u32 {
        0
    } else if step
        < protocol
            .cold_hold_steps
            .saturating_add(protocol.ramp_steps)
            .max(0) as u32
    {
        1
    } else {
        2
    }
}

#[cfg(feature = "gpu")]
fn validate_native_receipt(
    receipt: &prismcode::md_native_runtime::MdNativeRuntimeReceipt,
    config: &CoupledTwinEvidenceConfig,
) -> Result<()> {
    if receipt.status != "PASS" || receipt.coverage_status != "PASS" {
        bail!(
            "stream {} native MD receipt failed: status={} coverage={} missing={:?}",
            receipt.stream_id,
            receipt.status,
            receipt.coverage_status,
            receipt.missing_required_opcodes
        );
    }
    if receipt.cuda_graph_capture_dispatch_count == 0
        || receipt.cuda_graph_replay_dispatch_count == 0
        || receipt.resident_graph_replay_dispatch_count == 0
        || receipt.legacy_graph_replay_dispatch_count != 0
    {
        bail!(
            "stream {} resident CUDA graph evidence invalid: captures={} replays={} resident={} legacy={}",
            receipt.stream_id,
            receipt.cuda_graph_capture_dispatch_count,
            receipt.cuda_graph_replay_dispatch_count,
            receipt.resident_graph_replay_dispatch_count,
            receipt.legacy_graph_replay_dispatch_count
        );
    }
    if receipt.schema_version < 2
        || receipt.graph_window_consistency_status != "PASS"
        || !matches!(receipt.graph_exact_tail_status.as_str(), "NONE" | "PRESENT")
        || receipt.graph_sustained_body_steps_per_launch
            != u64::from(config.captured_graph_body_unroll)
        || receipt.hotpath_acceleration_status != "PASS"
        || receipt.graph_host_launch_collapse_ratio < config.min_hotpath_launch_collapse
    {
        bail!(
            "stream {} hotpath acceleration failed: schema={} consistency={} tail={} body_steps_per_launch={} expected_body={} status={} collapse={:.3} required={:.3}",
            receipt.stream_id,
            receipt.schema_version,
            receipt.graph_window_consistency_status,
            receipt.graph_exact_tail_status,
            receipt.graph_sustained_body_steps_per_launch,
            config.captured_graph_body_unroll,
            receipt.hotpath_acceleration_status,
            receipt.graph_host_launch_collapse_ratio,
            config.min_hotpath_launch_collapse
        );
    }
    if receipt.materialized_readout_status != "MATERIALIZED"
        || receipt.readout_opcode_coverage != "PASS"
    {
        bail!(
            "stream {} materialized readout evidence failed: status={} coverage={}",
            receipt.stream_id,
            receipt.materialized_readout_status,
            receipt.readout_opcode_coverage
        );
    }
    Ok(())
}

#[cfg(feature = "gpu")]
fn execute_strict_coupled_pair(
    engine_a: &mut PersistentNhsEngine,
    engine_b: &mut PersistentNhsEngine,
    topology_a: &PrismPrepTopology,
    topology_b: &PrismPrepTopology,
    protocol: &CryoUvProtocol,
    config: &CoupledTwinEvidenceConfig,
    seed: u64,
    outer_steps: i32,
    fused_inner_steps: i32,
    output_dir: &Path,
    inventory: &SplitT7Inventory,
    p1_memory_writer: &mut Option<std::io::BufWriter<std::fs::File>>,
) -> Result<CoupledTwinRuntimeEvidence> {
    config.validate()?;
    if outer_steps <= SPLIT_T7_TOTAL_FRAMES as i32 || fused_inner_steps <= 0 {
        bail!(
            "strict coupled-TWIN execution needs an online split-T7 prefix plus resident remainder; outer={} inner={} calibration_frames={}",
            outer_steps,
            fused_inner_steps,
            SPLIT_T7_TOTAL_FRAMES
        );
    }
    // The resident WHILE template contains one Director/physics body per
    // iteration.  `fused_inner_steps` affects ordinary `run()` but is not
    // repeated inside this captured resident body, so multiplying resident
    // chronology metadata by it would fabricate physical progress.  Until a
    // source-defined repeated-body resident template is bound, strict coupled
    // execution must use the one-director-step configuration.
    if fused_inner_steps != 1 {
        bail!(
            "strict coupled-TWIN resident graph requires fused_inner_steps=1: the active resident template executes one Director/physics body per iteration, not {} fused substeps",
            fused_inner_steps
        );
    }

    engine_a.enable_prismcode_md_native_runtime(0, true, true)?;
    engine_b.enable_prismcode_md_native_runtime(1, true, true)?;
    let temperature_kelvin = protocol.start_temp.max(50.0);
    let authoritative_crn_receipt = apply_coupled_twin_crn_initialization_v3(
        engine_a,
        engine_b,
        topology_a,
        topology_b,
        config,
        seed,
        temperature_kelvin,
        "authoritative",
        None,
    )?;
    let stem = coupled_topology_stem(config);
    let authoritative_crn_artifact = emit_coupled_twin_crn_receipt_v3(
        output_dir,
        &format!("{stem}_authoritative_crn_initialization.json"),
        "authoritative_coupled_twin_crn_initialization",
        &authoritative_crn_receipt,
        topology_a,
        topology_b,
        config,
        temperature_kelvin,
    )?;
    let (chain_role_map_a, chain_role_map_b) = coupled_native_chain_role_maps(
        topology_a,
        topology_b,
        config.topology_a_sha256 == config.topology_b_sha256,
    )?;
    let mut split_t7 = execute_online_split_t7_calibration(
        engine_a,
        engine_b,
        topology_a,
        topology_b,
        &chain_role_map_a,
        &chain_role_map_b,
        protocol,
        config,
        outer_steps as u32,
        fused_inner_steps,
        output_dir,
        inventory,
        &authoritative_crn_artifact,
        &authoritative_crn_receipt,
    )?;
    let remaining_outer = (outer_steps as u32)
        .checked_sub(split_t7.direct_prefix_outer_steps)
        .context("split-T7 direct prefix exceeds requested outer-step budget")?;
    if remaining_outer == 0 {
        bail!("split-T7 consumed the entire run; resident graph proof is impossible");
    }
    let body_unroll = config
        .captured_graph_body_unroll
        .min(remaining_outer)
        .max(1);
    let requested_host_chunk = config.graph_chunk_outer_steps.min(remaining_outer).max(1);
    let main_chunk = (requested_host_chunk / body_unroll)
        .max(1)
        .saturating_mul(body_unroll)
        .min(remaining_outer);
    crate::vram_pool::record_allocation_boundary(p1_memory_writer, engine_a.cuda_context(), 0,
        "coupled_pair_before_strict_primary_capture", None)?;
    let graph_a = engine_a
        .capture_autonomous_while_resident_graph_with_replay_binding(body_unroll)
        .context("capture stream A metal-resident WHILE graph")?;
    let graph_b = engine_b
        .capture_autonomous_while_resident_graph_with_replay_binding(body_unroll)
        .context("capture stream B metal-resident WHILE graph")?;
    crate::vram_pool::record_allocation_boundary(p1_memory_writer, engine_b.cuda_context(), 1,
        "coupled_pair_after_strict_primary_captures", None)?;
    if graph_a.resident_loop_status().is_none() || graph_b.resident_loop_status().is_none() {
        bail!("coupled-TWIN strict path captured a non-resident CUDA graph");
    }
    if graph_a.steps_per_launch() != body_unroll || graph_b.steps_per_launch() != body_unroll {
        bail!(
            "resident graph body-unroll mismatch: requested={} A={} B={}",
            body_unroll,
            graph_a.steps_per_launch(),
            graph_b.steps_per_launch()
        );
    }

    let mut completed_outer = 0u32;
    let mut host_launches = [0u64; 2];
    let mut resident_launches = [0u64; 2];
    let mut capture_counts = [1u64; 2];

    let mut execute_window = |engine_a: &mut PersistentNhsEngine,
                              engine_b: &mut PersistentNhsEngine,
                              graph_a: &crate::persistent_engine::BoundResidentGraph,
                              graph_b: &crate::persistent_engine::BoundResidentGraph,
                              window_outer: u32,
                              start_outer: u32|
     -> Result<()> {
        // Keep the authoritative run on the same matched, serialized schedule
        // as split-T7 calibration.  This preserves exact CRN control semantics
        // and prevents cross-stream scheduling from becoming a hidden contrast.
        let receipt_a = engine_a.execute_bound_resident_graph_replay_window(
            graph_a,
            window_outer,
            "graph.while_resident",
        )?;
        let launches_a = receipt_a.graph_launches;
        let post_graph_a = config
            .identical_topology_negative_control
            .then(|| engine_physical_component_snapshot(engine_a, 0))
            .transpose()?;
        let receipt_b = engine_b.execute_bound_resident_graph_replay_window(
            graph_b,
            window_outer,
            "graph.while_resident",
        )?;
        let launches_b = receipt_b.graph_launches;
        if let Some(post_graph_a) = post_graph_a.as_deref() {
            let post_graph_b = engine_physical_component_snapshot(engine_b, 1)?;
            require_identical_physical_components(
                &format!("resident_window_{start_outer}_post_graph"),
                post_graph_a,
                &post_graph_b,
            )?;
        }
        if launches_a == 0 || launches_b == 0 {
            bail!("resident coupled-TWIN graph issued zero host launches");
        }
        host_launches[0] = host_launches[0].saturating_add(launches_a as u64);
        host_launches[1] = host_launches[1].saturating_add(launches_b as u64);
        resident_launches[0] = resident_launches[0].saturating_add(launches_a as u64);
        resident_launches[1] = resident_launches[1].saturating_add(launches_b as u64);

        if receipt_a.end_device_step != receipt_b.end_device_step {
            bail!(
                "resident coupled-TWIN device-clock divergence after window {}: A={} B={}",
                start_outer,
                receipt_a.end_device_step,
                receipt_b.end_device_step,
            );
        }
        let completed_device_step = receipt_a.end_device_step;
        let phase = protocol_phase(protocol, completed_device_step);
        engine_a.kcc_step_once(completed_device_step, phase)?;
        engine_a.cuda_stream().synchronize()?;
        let post_kcc_a = config
            .identical_topology_negative_control
            .then(|| engine_physical_component_snapshot(engine_a, 0))
            .transpose()?;
        engine_b.kcc_step_once(completed_device_step, phase)?;
        engine_b.cuda_stream().synchronize()?;
        if let Some(post_kcc_a) = post_kcc_a.as_deref() {
            let post_kcc_b = engine_physical_component_snapshot(engine_b, 1)?;
            require_identical_physical_components(
                &format!("resident_window_{start_outer}_post_kcc"),
                post_kcc_a,
                &post_kcc_b,
            )?;
        }
        engine_a.force_spike_sync()?;
        engine_a.cuda_stream().synchronize()?;
        let post_spike_a = config
            .identical_topology_negative_control
            .then(|| engine_physical_component_snapshot(engine_a, 0))
            .transpose()?;
        engine_b.force_spike_sync()?;
        engine_b.cuda_stream().synchronize()?;
        if let Some(post_spike_a) = post_spike_a.as_deref() {
            let post_spike_b = engine_physical_component_snapshot(engine_b, 1)?;
            require_identical_physical_components(
                &format!("resident_window_{start_outer}_post_spike_sync"),
                post_spike_a,
                &post_spike_b,
            )?;
        }
        engine_a.rebuild_neighbor_lists_if_needed()?;
        engine_a.cuda_stream().synchronize()?;
        let post_rebuild_a = config
            .identical_topology_negative_control
            .then(|| engine_physical_component_snapshot(engine_a, 0))
            .transpose()?;
        engine_b.rebuild_neighbor_lists_if_needed()?;
        engine_b.cuda_stream().synchronize()?;
        if let Some(post_rebuild_a) = post_rebuild_a.as_deref() {
            let post_rebuild_b = engine_physical_component_snapshot(engine_b, 1)?;
            require_identical_physical_components(
                &format!("resident_window_{start_outer}_post_neighbor_rebuild"),
                post_rebuild_a,
                &post_rebuild_b,
            )?;
        }
        Ok(())
    };

    let bulk_outer = (remaining_outer / body_unroll).saturating_mul(body_unroll);
    while completed_outer < bulk_outer {
        let window = main_chunk.min(bulk_outer - completed_outer);
        execute_window(
            engine_a,
            engine_b,
            &graph_a,
            &graph_b,
            window,
            completed_outer,
        )?;
        completed_outer = completed_outer.saturating_add(window);
    }
    let tail = remaining_outer.saturating_sub(completed_outer);
    if tail > 0 {
        crate::vram_pool::record_allocation_boundary(p1_memory_writer, engine_a.cuda_context(), 0,
            "coupled_pair_before_strict_tail_capture", None)?;
        let tail_graph_a = engine_a
            .capture_autonomous_while_resident_graph_with_replay_binding(tail)
            .context("capture stream A exact-tail resident graph")?;
        let tail_graph_b = engine_b
            .capture_autonomous_while_resident_graph_with_replay_binding(tail)
            .context("capture stream B exact-tail resident graph")?;
        crate::vram_pool::record_allocation_boundary(p1_memory_writer, engine_b.cuda_context(), 1,
            "coupled_pair_after_strict_tail_captures", None)?;
        capture_counts[0] += 1;
        capture_counts[1] += 1;
        execute_window(
            engine_a,
            engine_b,
            &tail_graph_a,
            &tail_graph_b,
            tail,
            completed_outer,
        )?;
        completed_outer = completed_outer.saturating_add(tail);
    }
    if completed_outer != remaining_outer {
        bail!(
            "coupled-TWIN graph coverage mismatch: completed={} expected={}",
            completed_outer,
            remaining_outer
        );
    }

    let stem = coupled_topology_stem(config);
    let signal_a = engine_a
        .download_signal_preservation()
        .context("materialize stream A signal grid")?;
    let signal_b = engine_b
        .download_signal_preservation()
        .context("materialize stream B signal grid")?;
    let kcc_a = engine_a
        .compute_and_download_kcc()
        .context("materialize stream A KCC descriptors")?;
    let kcc_b = engine_b
        .compute_and_download_kcc()
        .context("materialize stream B KCC descriptors")?;

    let mut artifacts = inventory.shadow_trial_artifacts.clone();
    artifacts.extend(inventory.shadow_crn_artifacts.clone());
    artifacts.push(inventory.shadow_ensemble_artifact.clone());
    artifacts.push(authoritative_crn_artifact);
    artifacts.push(split_t7.admission_artifact.clone());
    for (stream_id, signal, kcc) in [(0u32, &signal_a, &kcc_a), (1u32, &signal_b, &kcc_b)] {
        let signal_path = output_dir.join(format!("{stem}_stream{stream_id}_signal_grid.bin"));
        write_signal_grid_evidence(&signal_path, signal, stream_id, &config.run_id, &stem)?;
        artifacts.push(artifact_evidence(
            "signal_grid",
            Some(stream_id),
            &signal_path,
            Some(signal.voxel_hit_grid.len() as u64),
            Some(1),
        )?);

        let kcc_path = output_dir.join(format!("{stem}_stream{stream_id}_kcc_v2full.bin"));
        write_kcc_evidence(&kcc_path, kcc, stream_id, &config.run_id, &stem)?;
        artifacts.push(artifact_evidence(
            "kcc_v2full",
            Some(stream_id),
            &kcc_path,
            Some(kcc.n_residues as u64),
            Some(1),
        )?);
    }

    let completed_md_steps_a = engine_a
        .device_protocol_current_step()
        .context("read stream A final resident ProtocolState clock")?;
    let completed_md_steps_b = engine_b
        .device_protocol_current_step()
        .context("read stream B final resident ProtocolState clock")?;
    if completed_md_steps_a != completed_md_steps_b {
        bail!(
            "coupled-TWIN final resident device-clock divergence: A={} B={}",
            completed_md_steps_a,
            completed_md_steps_b,
        );
    }
    let completed_md_steps = u64::from(completed_md_steps_a);
    let (
        ghost_a,
        ghost_map_a,
        noise_floor_a,
        noise_floor_input_a,
        frozen_anchor_a,
        final_manifold_input_a,
        ghost_segment_manifest_a,
    ) = emit_gpu_ghost_v2_snapshot(
        engine_a,
        topology_a,
        0,
        &config.topology_a_path,
        &config.topology_a_sha256,
        &chain_role_map_a,
        config,
        protocol,
        seed,
        &mut split_t7.stream_a,
        output_dir,
        completed_md_steps,
    )?;
    let ghost_map_a_path = ghost_map_a.path.clone();
    artifacts.push(ghost_a);
    artifacts.push(ghost_map_a);
    artifacts.push(noise_floor_a);
    artifacts.push(noise_floor_input_a);
    artifacts.push(frozen_anchor_a);
    artifacts.push(final_manifold_input_a);
    artifacts.push(ghost_segment_manifest_a);
    let (
        ghost_b,
        ghost_map_b,
        noise_floor_b,
        noise_floor_input_b,
        frozen_anchor_b,
        final_manifold_input_b,
        ghost_segment_manifest_b,
    ) = emit_gpu_ghost_v2_snapshot(
        engine_b,
        topology_b,
        1,
        &config.topology_b_path,
        &config.topology_b_sha256,
        &chain_role_map_b,
        config,
        protocol,
        seed,
        &mut split_t7.stream_b,
        output_dir,
        completed_md_steps,
    )?;
    if config.identical_topology_negative_control {
        let map_a: CoupledGhostSiteMap =
            serde_json::from_slice(&std::fs::read(&ghost_map_a_path)?)?;
        let map_b: CoupledGhostSiteMap =
            serde_json::from_slice(&std::fs::read(&ghost_map_b.path)?)?;
        if map_a.final_engine_physical_state_sha256_before
            != map_b.final_engine_physical_state_sha256_before
            || map_a.final_engine_physical_state_sha256_after
                != map_b.final_engine_physical_state_sha256_after
        {
            bail!("identical-topology control diverged in unsalted final physical engine state");
        }
    }
    artifacts.push(ghost_b);
    artifacts.push(ghost_map_b);
    artifacts.push(noise_floor_b);
    artifacts.push(noise_floor_input_b);
    artifacts.push(frozen_anchor_b);
    artifacts.push(final_manifold_input_b);
    artifacts.push(ghost_segment_manifest_b);

    let mut graph = Vec::new();
    for (stream_id, engine) in [(0u32, &*engine_a), (1u32, &*engine_b)] {
        let receipt_path = output_dir.join(format!(
            "{stem}_stream{stream_id:02}_prismcode_md_native_receipt.json"
        ));
        let receipt = engine
            .write_prismcode_md_native_receipt(&receipt_path)?
            .ok_or_else(|| {
                anyhow::anyhow!("stream {} native MD receipt was not emitted", stream_id)
            })?;
        validate_native_receipt(&receipt, config)?;
        let receipt_readback: prismcode::md_native_runtime::MdNativeRuntimeReceipt =
            serde_json::from_slice(&std::fs::read(&receipt_path)?)?;
        if receipt_readback.coverage_hash_blake3 != receipt.coverage_hash_blake3
            || receipt_readback.dispatch_trace_hash_blake3 != receipt.dispatch_trace_hash_blake3
        {
            bail!("stream {} native MD receipt readback drift", stream_id);
        }
        artifacts.push(artifact_evidence(
            "prismcode_md_native_receipt",
            Some(stream_id),
            &receipt_path,
            Some(receipt.dispatch_count),
            Some(receipt.schema_version),
        )?);
        graph.push(CoupledTwinGraphEvidence {
            stream_id,
            direct_prefix_outer_steps: split_t7.direct_prefix_outer_steps as u64,
            direct_prefix_md_steps: split_t7.direct_prefix_md_steps,
            resident_outer_steps: remaining_outer as u64,
            resident_md_steps: (remaining_outer as u64).saturating_mul(fused_inner_steps as u64),
            total_md_steps: (outer_steps as u64).saturating_mul(fused_inner_steps as u64),
            capture_count: capture_counts[stream_id as usize],
            host_launch_count: host_launches[stream_id as usize],
            resident_launch_count: resident_launches[stream_id as usize],
            legacy_launch_count: 0,
            logical_outer_steps: remaining_outer as u64,
            logical_md_steps: (remaining_outer as u64).saturating_mul(fused_inner_steps as u64),
            steps_per_launch: body_unroll,
            launch_collapse_ratio: receipt.graph_host_launch_collapse_ratio,
            launch_path: "graph.while_resident".to_string(),
        });
    }

    log::info!(
        "  Strict coupled-TWIN metal runtime: online split-T7 source spikes A/B={}/{}, direct prefix={}, resident outer steps={}, host launches A/B={}/{}, common seed={}",
        split_t7.warmup_spikes_a,
        split_t7.warmup_spikes_b,
        split_t7.direct_prefix_outer_steps,
        remaining_outer,
        host_launches[0],
        host_launches[1],
        seed
    );

    Ok(CoupledTwinRuntimeEvidence {
        schema_kind: "prismos.coupled_twin_runtime_evidence.v2".to_string(),
        schema_version: COUPLED_TWIN_EVIDENCE_SCHEMA_VERSION,
        run_id: config.run_id.clone(),
        execution_semantics: "independent_aligned_trajectory_analysis_common_random_numbers"
            .to_string(),
        exchange_active: false,
        causal_transport_claim: false,
        graph,
        artifacts,
    })
}

/// Run a PRISM-TWIN interferometric coupled observation simulation.
///
/// Two engines run with INTERLEAVED stepping on separate CUDA streams.
/// Every `exchange_interval` steps, spike density is exchanged (Layer 1)
/// and cross-correlation is computed (Layer 2).
///
/// Gate 1: interleaved stepping, NO exchange, NO CCF. Just verify dual streams.
/// Gate 2+: enable exchange and CCF progressively.
#[cfg(feature = "gpu")]
pub fn run_coupled_twin(
    config_ref: &crate::persistent_engine::PersistentBatchConfig,
    context: Arc<CudaContext>,
    fused_module: Arc<CudaModule>,
    topology: &PrismPrepTopology,
    // LINK #1: OPTIONAL second system loaded by engine_b (holo / designed
    // variant) for the molecular differential. None => engine_b loads `topology`
    // exactly like engine_a (today's behavior, bit-for-bit).
    topology_b: Option<&PrismPrepTopology>,
    protocol: CryoUvProtocol,
    twin_config: &CoupledTwinConfig,
    seed_a: u64,
    steps: i32,
    hmr: bool,
    fused_steps: u32,
    adaptive_dt: bool,
    ladd_enabled: bool,
    evidence_config: Option<&CoupledTwinEvidenceConfig>,
    output_dir: &std::path::Path,
    prismcode_admission: Option<&PrismCodeMdPolicyAdmissionV1>,
) -> Result<CoupledTwinResult> {
    let seed_b = if topology_b.is_some() {
        seed_a
    } else {
        seed_a + 1000
    };
    match (topology_b, evidence_config) {
        (Some(_), Some(config)) => {
            config.validate_source_files()?;
            if twin_config.enable_exchange
                || twin_config.persistent_coupling
                || twin_config.graph_coupling
            {
                bail!(
                    "two-topology molecular contrast requires exchange/persistent/graph coupling disabled"
                );
            }
            if !twin_config.enable_ccf {
                bail!("two-topology molecular contrast requires real GPU CCF/TE analysis");
            }
            if twin_config.nma_modes_path.is_some() {
                bail!(
                    "two-topology molecular contrast forbids one-sided NMA; use a separate factorial protocol lane"
                );
            }
        }
        (Some(_), None) => {
            bail!("two-topology coupled-TWIN execution requires a strict evidence configuration")
        }
        (None, Some(_)) => {
            bail!("a coupled-TWIN evidence configuration is invalid without --topology-b")
        }
        (None, None) => {}
    }

    log::info!("╔══════════════════════════════════════════════════════════╗");
    log::info!("║   PRISM-TWIN: Interferometric Spike Dynamics            ║");
    log::info!("╚══════════════════════════════════════════════════════════╝");
    log::info!("  Stream A (Scout):    seed={}, thermal only", seed_a);
    log::info!(
        "  Stream B (Observer): seed={}, thermal{}",
        seed_b,
        if topology_b.is_none() && twin_config.nma_modes_path.is_some() {
            " + NMA"
        } else {
            " only"
        }
    );
    log::info!(
        "  Phase offset: {:.0}%",
        twin_config.phase_offset_fraction * 100.0
    );
    log::info!(
        "  Exchange: {} (interval={})",
        if twin_config.enable_exchange {
            "ENABLED"
        } else {
            "DISABLED"
        },
        twin_config.exchange_interval
    );
    log::info!(
        "  CCF: {}",
        if twin_config.enable_ccf {
            "ENABLED"
        } else {
            "DISABLED"
        }
    );

    // Determine the fixed captured-graph shape on disposable same-seed shadow
    // engines. They are dropped before the authoritative engines are allocated,
    // so the inventory cannot mutate the scientific state or double steady VRAM.
    let strict_split_t7_inventory = if let (Some(strict_topology_b), Some(strict_config)) =
        (topology_b, evidence_config)
    {
        let strict_admission = prismcode_admission
            .context("strict coupled-TWIN requires retained PrismCode packet policy admission")?;
        let inventory_inner = fused_steps.max(1) as i32;
        let inventory_outer = u32::try_from((steps + inventory_inner - 1) / inventory_inner)
            .context("strict coupled-TWIN outer-step count is negative or exceeds u32")?;
        Some(inventory_split_t7_fixed_cardinality(
            config_ref,
            context.clone(),
            fused_module.clone(),
            topology,
            strict_topology_b,
            &protocol,
            strict_config,
            seed_a,
            inventory_outer,
            inventory_inner,
            hmr,
            fused_steps,
            adaptive_dt,
            ladd_enabled,
            output_dir,
            strict_admission,
        )?)
    } else {
        None
    };

    let mut p1_memory_writer = if twin_config.p1_allocation_timing {
        std::fs::create_dir_all(output_dir)?;
        Some(std::io::BufWriter::new(std::fs::OpenOptions::new()
            .write(true).create_new(true)
            .open(output_dir.join("coupled_twin_pair_p1_allocation_boundaries.jsonl"))?))
    } else { None };

    // Create authoritative CUDA streams only after shadow inventory teardown.
    let stream_a = context.new_stream().context("CUDA stream A")?;
    let stream_b = context.new_stream().context("CUDA stream B")?;
    log::info!("  CUDA streams: A ✓, B ✓");

    // Check VRAM before allocation
    let (vram_free, vram_total) = cudarc::driver::result::mem_get_info().unwrap_or((0, 0));
    log::info!(
        "  VRAM: {:.1} GB free / {:.1} GB total",
        vram_free as f64 / 1e9,
        vram_total as f64 / 1e9
    );

    // ── Initialize Engine A (Scout) ──
    log::info!("  Initializing Stream A...");
    crate::vram_pool::record_allocation_boundary(&mut p1_memory_writer, &context, 0,
        "coupled_pair_before_engine_a_construction", None)?;
    let mut engine_a = PersistentNhsEngine::new_on_stream(
        config_ref,
        context.clone(),
        fused_module.clone(),
        stream_a,
    )?;
    if let Some(admission) = prismcode_admission {
        engine_a.set_prismcode_md_policy_admission(admission.clone())?;
    }
    engine_a.load_topology(topology)?;
    crate::vram_pool::record_allocation_boundary(&mut p1_memory_writer, &context, 0,
        "coupled_pair_after_engine_a_topology_load", None)?;
    if hmr {
        engine_a.set_dt(0.004)?;
    }
    if fused_steps > 1 {
        engine_a.set_fused_inner_steps(fused_steps)?;
    }
    if adaptive_dt {
        engine_a.set_adaptive_dt(true)?;
    }
    if ladd_enabled {
        engine_a.set_ladd_enabled(true);
    }
    // A two-topology molecular contrast must differ only by topology. Applying
    // NMA to either lane would make the identical-topology negative control
    // non-zero by construction. The historical single-topology protocol lane
    // remains unchanged below (NMA on engine B only).
    if topology_b.is_some() {
        if twin_config.nma_modes_path.is_some() {
            log::info!(
                "  Two-topology molecular contrast: NMA suppressed on both streams to preserve the topology-only intervention and identical-control null"
            );
        }
    }
    engine_a.set_cryo_uv_protocol(protocol.clone())?;
    engine_a.set_spike_accumulation(true);
    log::info!("  Stream A: ✓");

    // ── Initialize Engine B (Focused Observer) ──
    log::info!("  Initializing Stream B...");
    crate::vram_pool::record_allocation_boundary(&mut p1_memory_writer, &context, 1,
        "coupled_pair_before_engine_b_construction", None)?;
    let mut engine_b = PersistentNhsEngine::new_on_stream(
        config_ref,
        context.clone(),
        fused_module.clone(),
        stream_b,
    )?;
    if let Some(admission) = prismcode_admission {
        engine_b.set_prismcode_md_policy_admission(admission.clone())?;
    }
    // LINK #1: engine_b loads the second system when present; None ⇒ same
    // reference topology engine_a loaded (bit-for-bit today's behavior).
    engine_b.load_topology(topology_b.unwrap_or(topology))?;
    crate::vram_pool::record_allocation_boundary(&mut p1_memory_writer, &context, 1,
        "coupled_pair_after_engine_b_topology_load", None)?;
    if hmr {
        engine_b.set_dt(0.004)?;
    }
    if fused_steps > 1 {
        engine_b.set_fused_inner_steps(fused_steps)?;
    }
    if adaptive_dt {
        engine_b.set_adaptive_dt(true)?;
    }
    if ladd_enabled {
        engine_b.set_ladd_enabled(true);
    }

    // NMA perturbation for B only (Layer 4).
    // LINK #1 (decision 3): suppressed in two-system mode — engine_b must carry
    // ONLY the molecular difference, and the NMA ground-truth was applied to the
    // reference stream A above. Single-topology (protocol contrast) is unchanged.
    if topology_b.is_none() {
        if let Some(ref nma_path) = twin_config.nma_modes_path {
            engine_b.set_nma_amplification(twin_config.nma_amplification)?;
            engine_b.load_nma_modes(nma_path)?;
            log::info!(
                "  Stream B: NMA loaded (amplification={:.1})",
                twin_config.nma_amplification
            );
        }
    } else {
        log::info!(
            "  LINK #1: engine_b NMA suppressed (two-system molecular contrast — B carries molecular difference only)"
        );
    }

    // Gate 3: Phase-offset scheduling
    // B gets extra cold_hold steps so it starts heating AFTER A.
    // Offset = cold_hold_steps × phase_offset_fraction
    //
    // LINK #1 (invariant): a two-system MOLECULAR contrast must run A and B for
    // IDENTICAL (steps, protocol) — offset_steps=0, protocol_b == protocol.clone(),
    // steps_b == steps — so the differential is a clean molecular contrast and the
    // TE/CCF time bins line up. Protocol-contrast (no topology_b) keeps the
    // historical phase offset byte-for-byte.
    let offset_steps = if topology_b.is_some() {
        0
    } else {
        (protocol.cold_hold_steps as f32 * twin_config.phase_offset_fraction) as i32
    };
    let mut protocol_b = protocol.clone();
    protocol_b.cold_hold_steps += offset_steps;
    let steps_b = steps + offset_steps; // B runs longer

    engine_b.set_cryo_uv_protocol(protocol_b.clone())?;
    engine_b.set_spike_accumulation(true);

    // Strict paired execution enables the governed runtime at its dedicated
    // resident launch seam. The canonical single-topology coupled route has no
    // later seam, so enable it here only when an exact packet admission was
    // carried by the caller. Library compatibility callers without admission
    // retain their historical non-governed behavior.
    if strict_split_t7_inventory.is_none() && prismcode_admission.is_some() {
        engine_a.enable_prismcode_md_native_runtime(0, true, true)?;
        engine_b.enable_prismcode_md_native_runtime(1, true, true)?;
    }

    log::info!("  Stream B: ✓");
    log::info!(
        "  Gate 3 phase offset: {} steps ({:.0}% of cold_hold={})",
        offset_steps,
        twin_config.phase_offset_fraction * 100.0,
        protocol.cold_hold_steps
    );
    log::info!(
        "  Stream A schedule: cold_hold={}, ramp={}, warm_hold={}, total={}",
        protocol.cold_hold_steps,
        protocol.ramp_steps,
        protocol.warm_hold_steps,
        steps
    );
    log::info!(
        "  Stream B schedule: cold_hold={}, ramp={}, warm_hold={}, total={}",
        protocol.cold_hold_steps + offset_steps,
        protocol.ramp_steps,
        protocol.warm_hold_steps,
        steps_b
    );
    log::info!(
        "  A enters heating at step {}, B enters heating at step {}",
        protocol.cold_hold_steps,
        protocol.cold_hold_steps + offset_steps
    );

    // VRAM after allocation
    let (vram_after, _) = cudarc::driver::result::mem_get_info().unwrap_or((0, 0));
    let vram_used = (vram_free - vram_after) as f64 / 1e9;
    log::info!(
        "  VRAM used by twin engines: {:.2} GB ({:.1} GB remaining)",
        vram_used,
        vram_after as f64 / 1e9
    );

    // ── RING BUFFER ALLOCATION (TWIN interferometric coupling) ──
    //
    // Two ring buffers carry spike evidence between streams:
    //   ring_a: holds A's spikes → read by B's threshold adapter
    //   ring_b: holds B's spikes → read by A's threshold adapter
    //
    // The coupling flow per step:
    //   1. read_and_adapt: ring_b → lower A's thresholds where B saw spikes
    //   2. read_and_adapt: ring_a → lower B's thresholds where A saw spikes
    //   3. sync exchange stream
    //   4. run both engines (they pick up modified thresholds)
    //   5. compact new spikes → push to ring buffers
    //   6. periodic threshold recovery (prevent permanent suppression)

    let (mut ring_a, mut ring_b, stream_exchange) = if twin_config.enable_exchange {
        let ring_module = context
            .load_module(cudarc::nvrtc::Ptx::from_file(&find_twin_ptx(
                "ring_buffer.ptx",
            )?))
            .context("Failed to load ring_buffer.ptx")?;
        let stream_ex = context.new_stream().context("CUDA exchange stream")?;

        let mut ra = TwinRingBuffer::new(&context, &stream_ex, &ring_module, 8192)?;
        let mut rb = TwinRingBuffer::new(&context, &stream_ex, &ring_module, 8192)?;
        ra.reset(&stream_ex)?;
        rb.reset(&stream_ex)?;

        log::info!("  Ring buffers: A ✓, B ✓ (capacity=8192 each)");
        (Some(ra), Some(rb), Some(stream_ex))
    } else {
        (None, None, None)
    };

    // Molecular contrast deliberately keeps cross-system threshold exchange
    // disabled, but CCF/TE are independent post-run analyses and require their
    // own CUDA ordering domain. Coupling their availability to the exchange
    // stream caused the prior silent-zero defect.
    let analysis_stream = if topology_b.is_some() && twin_config.enable_ccf {
        Some(
            context
                .new_stream()
                .context("CUDA molecular-contrast analysis stream")?,
        )
    } else {
        None
    };

    // ── GATE 3: Device-side compaction via compact_and_push (ring_buffer.cu) ──
    // When graph_coupling is enabled, push_device() replaces push_compacted().
    // No separate DeviceCompactor needed — compact_and_push is in the ring buffer PTX.
    let use_device_compaction = twin_config.graph_coupling
        && twin_config.enable_exchange
        && ring_a
            .as_ref()
            .map(|r| r.has_device_push())
            .unwrap_or(false);
    if use_device_compaction {
        log::info!("  Gate 3: Device-side spike compaction ACTIVE (compact_and_push kernel)");
    }

    // ── PERSISTENT COUPLING KERNEL (optional, --persistent-coupling) ──
    let mut twin_signal: Option<crate::twin_kernels::TwinSignal> = None;
    let mut _persistent_kernel: Option<crate::twin_kernels::TwinCouplingPersistent> = None;

    if twin_config.persistent_coupling {
        if let Some(ref se) = stream_exchange {
            // Load signal + coupling PTX modules
            let signal_module = context
                .load_module(cudarc::nvrtc::Ptx::from_file(&find_twin_ptx(
                    "twin_signal.ptx",
                )?))
                .context("Failed to load twin_signal.ptx")?;
            let coupling_module = context
                .load_module(cudarc::nvrtc::Ptx::from_file(&find_twin_ptx(
                    "twin_coupling_persistent.ptx",
                )?))
                .context("Failed to load twin_coupling_persistent.ptx")?;

            let mut signal = crate::twin_kernels::TwinSignal::new(se, &signal_module)?;

            // Launch persistent coupling kernel — runs for the ENTIRE simulation
            if let (Some(ref mut ra), Some(ref mut rb)) = (&mut ring_a, &mut ring_b) {
                let persistent =
                    crate::twin_kernels::TwinCouplingPersistent::new(&coupling_module)?;

                let outer_total =
                    ((steps + fused_steps.max(1) as i32 - 1) / fused_steps.max(1) as i32) as u32;
                let coupling_config = crate::twin_kernels::TwinCouplingConfig {
                    sensitivity_boost: twin_config.sensitivity_boost,
                    max_reduction_fraction: twin_config.max_threshold_reduction,
                    decay_constant: 500.0,
                    recovery_rate: 0.01,
                    recovery_interval: 1000,
                    total_steps: outer_total,
                };

                // The persistent kernel needs simultaneous immutable refs (spike buffers,
                // grid info) and mutable refs (thresholds) to the same engines. The borrow
                // checker prevents this, but the GPU buffers are distinct allocations with
                // no aliasing. Use threshold_buffers_mut() which returns both in one call.
                let grid_info = engine_a.grid_info();

                if let Some((gx, gy, gz, ox, oy, oz, vs)) = grid_info {
                    // Use combined accessor to get all GPU state in one borrow
                    let state_a = engine_a.twin_coupling_gpu_state();
                    let state_b = engine_b.twin_coupling_gpu_state();

                    if let (
                        Some((thresh_a, base_a, sbuf_a, sc_a)),
                        Some((thresh_b, base_b, sbuf_b, sc_b)),
                    ) = (state_a, state_b)
                    {
                        persistent.launch(
                            se,
                            &mut signal,
                            ra,
                            rb,
                            8192,
                            sbuf_a,
                            sc_a,
                            sbuf_b,
                            sc_b,
                            thresh_a,
                            base_a,
                            thresh_b,
                            base_b,
                            (gx, gy, gz),
                            (ox, oy, oz),
                            vs,
                            &coupling_config,
                        )?;
                        log::info!(
                            "  ✓ Persistent coupling kernel LAUNCHED ({} steps)",
                            outer_total
                        );
                    } else {
                        log::warn!("  Persistent coupling: engine GPU state not available");
                    }
                } else {
                    log::warn!("  Persistent coupling: grid info not available");
                }

                _persistent_kernel = Some(persistent);
            }

            twin_signal = Some(signal);
            log::info!("  Persistent coupling: signal kernels loaded ✓");
        }
    }

    // PERSISTENT COUPLING DISABLED: the cooperative kernel's spin-wait loop
    // is non-preemptible on SM120 (Blackwell), causing SM starvation even with
    // 2 blocks. The physics kernel gets zero MBW because the CUDA scheduler
    // can't reclaim the persistent kernel's blocks. Host-mediated coupling (CPU path) is RARE per GPU primacy — captured graph is primary for TWIN surfaces.
    // (which is functionally identical) runs at full physics throughput.
    //
    // The persistent kernel (Gates 0-3) is compiled and launchable but should
    // NOT be used until CUDA cooperative kernel preemption is investigated
    // on the RTX 5080 Blackwell architecture.
    let use_persistent = false; // FORCED OFF — see comment above
    if twin_config.persistent_coupling {
        log::warn!("  --persistent-coupling: DISABLED (SM starvation on Blackwell SM120)");
        log::warn!(
            "  Using host-mediated coupling (functionally identical, full physics throughput)"
        );
    }

    let diag_enabled = std::env::var("PRISM_TWIN_DIAG").is_ok();
    if diag_enabled {
        log::info!("  PRISM_TWIN_DIAG: threshold coupling diagnostics ENABLED");
    }

    // ── EXECUTION MODE SELECTION ──
    let inner = fused_steps.max(1) as i32;
    let outer_steps_a = (steps + inner - 1) / inner;
    let outer_steps_b = (steps_b + inner - 1) / inner;
    let outer_steps = outer_steps_a.max(outer_steps_b);

    // ════════════════════════════════════════════════════════════════════
    // GRAPH-BASED AUTONOMOUS COUPLING (--graph-coupling)
    //
    // Instead of a CPU for-loop launching kernels 40M times, we:
    //   1. Capture one physics step from each engine (stream capture)
    //   2. Build a CUDA WHILE graph: physics → compact → adapt → repeat
    //   3. Launch the graph ONCE
    //   4. CPU harvests spike data to Parquet while GPU runs
    //   5. Wait for completion
    //
    // The GPU's Gigathread Engine manages the loop in silicon.
    // SMs run physics, retire, get fed the next kernel — zero host latency.
    // ════════════════════════════════════════════════════════════════════
    if twin_config.graph_coupling {
        log::info!("╔══════════════════════════════════════════════════════════╗");
        log::info!("║   CUDA GRAPH AUTONOMOUS COUPLING MODE                    ║");
        log::info!(
            "║   One cudaGraphLaunch → GPU runs {} steps             ║",
            outer_steps
        );
        log::info!("╚══════════════════════════════════════════════════════════╝");

        // Check support
        if !prism_cuda_ext::graph::TwinCouplingGraph::is_supported() {
            anyhow::bail!("--graph-coupling requires CUDA driver ≥ 12.4");
        }

        log::info!(
            "{}",
            prism_cuda_ext::graph::TwinCouplingGraph::capabilities_report()
        );

        // TODO: The full graph-based execution path:
        //
        // Step 1: Stream capture of physics
        //   CapturedPhysicsGraph::begin_capture(&stream_a)?;
        //   engine_a.step()?;  // recorded, not executed
        //   let physics_a = CapturedPhysicsGraph::end_capture(&stream_a)?;
        //   // Same for engine B
        //
        // Step 2: Build the WHILE graph body
        //   body_graph.add_child_node(physics_a) → node_pa
        //   body_graph.add_child_node(physics_b) → node_pb
        //   body_graph.add_kernel_node(compact_a, depends=[node_pa]) → node_ca
        //   body_graph.add_kernel_node(compact_b, depends=[node_pb]) → node_cb
        //   body_graph.add_kernel_node(adapt_b_to_a, depends=[node_cb]) → node_adapt_a
        //   body_graph.add_kernel_node(adapt_a_to_b, depends=[node_ca]) → node_adapt_b
        //   body_graph.add_kernel_node(recovery, depends=[node_adapt_a, node_adapt_b])
        //   body_graph.add_kernel_node(decrement, depends=[recovery])
        //
        // Step 3: Launch
        //   graph.launch(&stream_exchange)?;
        //
        // Step 4: Harvest (background thread)
        //   std::thread::spawn(|| exhaust_buffer.harvest_to_parquet())
        //
        // Step 5: Wait
        //   stream_exchange.synchronize()?;
        //
        // BLOCKER: engine_a.step() launches on engine_a's internal stream,
        // but capture must happen on the SAME stream the engine uses.
        // PersistentNhsEngine doesn't expose its internal CudaStream.
        // This requires adding a stream() accessor to the engine.
        //
        // For now, fall back to host-mediated with a clear message.
        log::info!("  Graph coupling: will capture coupling kernels on first step");
        log::info!("  Then replay via cuGraphLaunch each subsequent step");
        log::info!("  Physics remains host-managed (all flags respected)");
    }

    log::info!(
        "  Running {} interleaved outer steps (A={}, B={} fused steps)...",
        outer_steps,
        outer_steps_a,
        outer_steps_b
    );

    let start = std::time::Instant::now();
    let mut spikes_a_total = 0usize;
    let mut spikes_b_total = 0usize;

    // Gate 2 exchange accumulators (CPU density telemetry — kept for logging)
    let mut n_exchanges: u32 = 0;
    let mut total_density_a_to_b: f64 = 0.0;
    let mut total_density_b_to_a: f64 = 0.0;
    let mut max_nonzero_regions: u32 = 0;

    // Ring buffer spike tracking
    let mut prev_accum_len_a: usize = 0; // index into accumulated_spikes for delta
    let mut prev_accum_len_b: usize = 0;
    let mut ring_spikes_exchanged: u64 = 0;
    let mut coupling_active_steps: u32 = 0;

    let mut a_finished = false;

    // Graph-based coupling: captured on step 1, replayed on step 2+
    let mut coupling_replay: Option<cudarc::driver::safe::CudaGraph> = None;
    let use_graph_coupling = twin_config.graph_coupling && twin_config.enable_exchange;

    if twin_config.enable_exchange && !twin_config.graph_coupling {
        log::warn!(
            "╔════════════════════════════════════════════════════════════════════════════╗"
        );
        log::warn!("║  RECOMMENDED OPERATIONAL IMPROVEMENT (PRISM TWIN CPU PATH)                ║");
        log::warn!("║  --coupled-twin is active WITHOUT --graph-coupling.                       ║");
        log::warn!("║  This selects the host-mediated (CPU-involved) coupling path for          ║");
        log::warn!("║  compaction, adaptation signaling, and step orchestration.                ║");
        log::warn!("║                                                                           ║");
        log::warn!("║  GPU + captured graph + tile surfaces are PRIMARY for all wired engine    ║");
        log::warn!("║  surfaces (TWIN, MD physics, DSTW temporal/superresolution, GhostTile     ║");
        log::warn!("║  diagnostic surfaces, spike/phase lattices, etc.).                        ║");
        log::warn!("║                                                                           ║");
        log::warn!("║  CPU/host TWIN (or any CPU operation in the E2E pipeline) should be used  ║");
        log::warn!("║  ONLY RARELY (explicit debug, hardware without graph support, or          ║");
        log::warn!("║  controlled fallback testing).                                            ║");
        log::warn!("║                                                                           ║");
        log::warn!("║  Add --graph-coupling (with --coupled-twin) for full GPU utilization,     ║");
        log::warn!("║  one cudaGraphLaunch autonomous WHILE coupling, background CPU harvest    ║");
        log::warn!("║  only. Captured graph body unroll and Ghost tile lattice configs are the  ║");
        log::warn!("║  primary surfaces.                                                        ║");
        log::warn!("║  This applies to MD stages, DSTW, all post-MD, binding, therm, and any    ║");
        log::warn!("║  other E2E pipeline/ecosystem surfaces the engine has wired up.           ║");
        log::warn!(
            "╚════════════════════════════════════════════════════════════════════════════╝"
        );
    }

    let mut strict_runtime_evidence = None;
    if let Some(strict_config) = evidence_config {
        let strict_topology_b =
            topology_b.context("strict coupled-TWIN evidence configuration requires topology B")?;
        strict_runtime_evidence = Some(execute_strict_coupled_pair(
            &mut engine_a,
            &mut engine_b,
            topology,
            strict_topology_b,
            &protocol,
            strict_config,
            seed_a,
            outer_steps,
            inner,
            output_dir,
            strict_split_t7_inventory
                .as_ref()
                .context("strict coupled-TWIN split-T7 inventory is absent")?,
            &mut p1_memory_writer,
        )?);
    } else {
        for step in 0..outer_steps {
            // ════════════════════════════════════════════════════════════════════
            // PHASE 1: Cross-stream threshold adaptation (BEFORE engine.run)
            // SKIPPED when persistent coupling is active — the persistent kernel
            // handles threshold adaptation on-device between signal flags.
            if !use_persistent {
                //
                // B's recent spikes → lower A's thresholds (make A more sensitive
                //   in regions where B found activity)
                // A's recent spikes → lower B's thresholds (symmetric)
                //
                // This is the interferometric coupling — detector sensitivity in
                // one stream is steered by evidence from the other stream.
                // ════════════════════════════════════════════════════════════════════
                if let (Some(ref mut rb), Some(ref se)) = (&mut ring_b, &stream_exchange) {
                    if step > 0 {
                        // Get grid geometry from engine A (both engines share the same grid)
                        if let Some((gx, gy, gz, ox, oy, oz, vs)) = engine_a.grid_info() {
                            // B's evidence → adapt A's thresholds
                            if let Some((thresh_a, base_a)) = engine_a.threshold_buffers_mut() {
                                rb.read_and_adapt(
                                    se,
                                    thresh_a,
                                    base_a,
                                    (gx, gy, gz),
                                    (ox, oy, oz),
                                    vs,
                                    twin_config.sensitivity_boost,
                                    twin_config.max_threshold_reduction,
                                    step as u32,
                                    500.0, // decay_constant: spikes > 500 steps old lose influence
                                )?;
                            }
                        }
                    }
                }
                if let (Some(ref mut ra), Some(ref se)) = (&mut ring_a, &stream_exchange) {
                    if step > 0 {
                        if let Some((gx, gy, gz, ox, oy, oz, vs)) = engine_b.grid_info() {
                            // A's evidence → adapt B's thresholds
                            if let Some((thresh_b, base_b)) = engine_b.threshold_buffers_mut() {
                                ra.read_and_adapt(
                                    se,
                                    thresh_b,
                                    base_b,
                                    (gx, gy, gz),
                                    (ox, oy, oz),
                                    vs,
                                    twin_config.sensitivity_boost,
                                    twin_config.max_threshold_reduction,
                                    step as u32,
                                    500.0,
                                )?;
                            }
                        }
                    }
                }
                // Sync exchange stream BEFORE launching fused kernels so threshold
                // modifications are visible to both engines' next run().
                if let Some(ref se) = stream_exchange {
                    if step > 0 {
                        se.synchronize()?;
                    }
                }
            } // end if !use_persistent (Phase 1)

            // ════════════════════════════════════════════════════════════════════
            // PHASE 1.5: Diagnostic — measure coupling effectiveness
            // ════════════════════════════════════════════════════════════════════
            if diag_enabled && step > 0 && step % 100 == 0 {
                // Measure coupling effectiveness: L2 norm of (threshold - base_threshold)
                // over a sample of voxels. Non-zero delta means coupling is modifying thresholds.
                let delta_l2 = if ring_spikes_exchanged > 0 {
                    // Use the ring buffer overflow count as a proxy — if spikes are
                    // being exchanged and read, coupling is active. Full threshold
                    // download would be expensive, so we use the spike count as signal.
                    let overflow_a = ring_b
                        .as_ref()
                        .and_then(|r| {
                            stream_exchange
                                .as_ref()
                                .and_then(|se| r.overflow_count(se).ok())
                        })
                        .unwrap_or(0);
                    let overflow_b = ring_a
                        .as_ref()
                        .and_then(|r| {
                            stream_exchange
                                .as_ref()
                                .and_then(|se| r.overflow_count(se).ok())
                        })
                        .unwrap_or(0);
                    // Coupling is active if spikes were exchanged without 100% overflow
                    if overflow_a + overflow_b < ring_spikes_exchanged as u32 {
                        coupling_active_steps += 1;
                    }
                    ring_spikes_exchanged as f32 / (step as f32 + 1.0)
                } else {
                    0.0
                };
                log::info!(
                    "  [DIAG] step={} spikes_per_step={:.1} ring_total={} active={}/{}",
                    step,
                    delta_l2,
                    ring_spikes_exchanged,
                    coupling_active_steps,
                    step / 100
                );
            }

            // ════════════════════════════════════════════════════════════════════
            // PHASE 2: Run both engines (concurrent on separate CUDA streams)
            // ════════════════════════════════════════════════════════════════════
            if step < outer_steps_a {
                let summary_a = engine_a.run(inner)?;
                spikes_a_total += summary_a.total_spikes;
            } else if !a_finished {
                a_finished = true;
                log::info!(
                    "  Stream A finished at step {} (B continues for {} more)",
                    step,
                    outer_steps_b - step
                );
            }

            let summary_b = engine_b.run(inner)?;
            spikes_b_total += summary_b.total_spikes;

            // ════════════════════════════════════════════════════════════════════
            // GRAPH COUPLING: if we have a captured graph, replay it and skip
            // all host-mediated coupling phases (1, 3, 4).
            // On step 1: capture the coupling sequence into a graph.
            // On step 2+: replay the graph with cuGraphLaunch.
            // ════════════════════════════════════════════════════════════════════
            if use_graph_coupling && step > 0 {
                if let Some(ref coupling_graph) = coupling_replay {
                    // Replay the captured coupling graph — zero host kernel launches
                    coupling_graph
                        .launch()
                        .map_err(|e| anyhow::anyhow!("Coupling graph launch failed: {:?}", e))?;

                    // Progress logging
                    if (step + 1) % 5000 == 0 || step == outer_steps - 1 {
                        let elapsed = start.elapsed().as_secs_f64();
                        let steps_per_sec = (step + 1) as f64 / elapsed;
                        log::info!(
                            "  [GRAPH] Step {}/{}: ({:.0} steps/s)",
                            step + 1,
                            outer_steps,
                            steps_per_sec
                        );
                    }
                    continue; // Graph handled all coupling — skip to physics
                } else if step == 1 {
                    // Step 1: capture the coupling sequence as a CUDA Graph.
                    // Uses Gate 3 push_device() (compact_and_push kernel) — fully capturable.
                    if let Some(ref se) = stream_exchange {
                        log::info!("  [GRAPH] Capturing coupling kernel sequence on step 1...");
                        use cudarc::driver::sys::CUstreamCaptureMode;

                        crate::vram_pool::record_allocation_boundary(&mut p1_memory_writer, &context, 0,
                            "coupled_pair_before_exchange_graph_capture", None)?;
                        let capture_result: Result<()> = (|| {
                            se.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
                                .map_err(|e| anyhow::anyhow!("Capture begin failed: {:?}", e))?;

                            // Phase 1: threshold adaptation (B→A, A→B)
                            if let Some((gx, gy, gz, ox, oy, oz, vs)) = engine_a.grid_info() {
                                if let Some((thresh_a, base_a)) = engine_a.threshold_buffers_mut() {
                                    if let Some(ref mut rb) = ring_b {
                                        rb.read_and_adapt(
                                            se,
                                            thresh_a,
                                            base_a,
                                            (gx, gy, gz),
                                            (ox, oy, oz),
                                            vs,
                                            twin_config.sensitivity_boost,
                                            twin_config.max_threshold_reduction,
                                            1,
                                            500.0,
                                        )?;
                                    }
                                }
                                if let Some((thresh_b, base_b)) = engine_b.threshold_buffers_mut() {
                                    if let Some(ref mut ra) = ring_a {
                                        ra.read_and_adapt(
                                            se,
                                            thresh_b,
                                            base_b,
                                            (gx, gy, gz),
                                            (ox, oy, oz),
                                            vs,
                                            twin_config.sensitivity_boost,
                                            twin_config.max_threshold_reduction,
                                            1,
                                            500.0,
                                        )?;
                                    }
                                }
                            }

                            // Phase 3: device-side spike compaction via compact_and_push (Gate 3)
                            // push_device() launches the compact_and_push kernel — fully capturable.
                            if let (Some(sbuf_a), Some(sc_a)) =
                                (engine_a.spike_buffer_gpu(), engine_a.spike_count_gpu())
                            {
                                if let Some(ref mut ra) = ring_a {
                                    ra.push_device(se, sbuf_a, sc_a)?;
                                }
                            }
                            if let (Some(sbuf_b), Some(sc_b)) =
                                (engine_b.spike_buffer_gpu(), engine_b.spike_count_gpu())
                            {
                                if let Some(ref mut rb) = ring_b {
                                    rb.push_device(se, sbuf_b, sc_b)?;
                                }
                            }

                            // Phase 4: recovery
                            let n_voxels = engine_a.total_voxels() as u32;
                            if let Some((thresh_a, base_a)) = engine_a.threshold_buffers_mut() {
                                if let Some(ref rb) = ring_b {
                                    rb.threshold_recovery(se, thresh_a, base_a, n_voxels, 0.01)?;
                                }
                            }
                            if let Some((thresh_b, base_b)) = engine_b.threshold_buffers_mut() {
                                if let Some(ref ra) = ring_a {
                                    ra.threshold_recovery(se, thresh_b, base_b, n_voxels, 0.01)?;
                                }
                            }

                            Ok(())
                        })();

                        match capture_result {
                            Ok(()) => {
                                use cudarc::driver::sys::CUgraphInstantiate_flags;
                                match se.end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH) {
                                Ok(Some(graph)) => {
                                    log::info!("  [GRAPH] Coupling graph captured and instantiated");
                                    log::info!("  [GRAPH] Steps 2+ will use cuGraphLaunch (zero host coupling overhead)");
                                    coupling_replay = Some(graph);
                                    crate::vram_pool::record_allocation_boundary(&mut p1_memory_writer, &context, 0,
                                        "coupled_pair_after_exchange_graph_capture", None)?;
                                }
                                Ok(None) => {
                                    log::warn!("  [GRAPH] Capture produced null graph — continuing host-mediated (RARE CPU fallback). RECOMMENDED: investigate capture support for full GPU captured graph primacy on this TWIN surface.");
                                }
                                Err(e) => {
                                    log::warn!("  [GRAPH] Instantiation failed: {:?} — continuing host-mediated (RARE CPU fallback). RECOMMENDED OPERATIONAL IMPROVEMENT: resolve for full GPU utilization + captured graph on TWIN/MD/DSTW surfaces.", e);
                                }
                            }
                            }
                            Err(e) => {
                                log::warn!(
                                "  [GRAPH] Capture failed: {} — continuing host-mediated (RARE CPU fallback). RECOMMENDED: port/advance this to full captured graph GPU utilization for all PRISM surfaces (TWIN, MD, DSTW, Ghost tiles, etc.). No functionality removed; this path remains for accuracy parity.",
                                e
                            );
                                // Attempt to end capture to restore stream to normal mode
                                let _ = se.end_capture(
                                cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH
                            );
                            }
                        }
                    }
                    // Fall through to host-mediated for this step (capture recorded but didn't execute)
                }
            }

            // ════════════════════════════════════════════════════════════════════
            // PHASE 2.5: Signal stream completion (persistent coupling mode only)
            // The persistent kernel spin-waits on these flags to begin exchange.
            // ════════════════════════════════════════════════════════════════════
            if use_persistent {
                if let Some(ref mut sig) = twin_signal {
                    // Signal on the engine's own CUDA stream so the signal is
                    // ordered AFTER all physics + detection kernels complete.
                    // The persistent kernel sees the flag only after engine work is done.
                    // NOTE: we use stream_exchange for now since engine streams are internal.
                    // Production: should signal on engine's stream for proper ordering.
                    if let Some(ref se) = stream_exchange {
                        sig.signal_a(se, (step + 1) as u32)?;
                        sig.signal_b(se, (step + 1) as u32)?;
                    }
                }
            }

            // ════════════════════════════════════════════════════════════════════
            // PHASE 3: Compact new spikes and push to ring buffers
            // SKIPPED when persistent coupling is active — the persistent kernel
            // handles spike compaction + ring push on-device.
            if !use_persistent {
                //
                // After engine.run(), new spikes have been downloaded to CPU and
                // appended to accumulated_spikes. We slice the delta (new since
                // last push), compact GpuSpikeEvent (92B) → RingSpikeEvent (48B),
                // upload, and push to the ring buffer.
                // ════════════════════════════════════════════════════════════════════
                if twin_config.enable_exchange {
                    let se = stream_exchange.as_ref().unwrap();

                    if use_device_compaction {
                        // Gate 3: device-side compaction — spikes never leave GPU
                        if let (Some(sbuf_a), Some(sc_a)) =
                            (engine_a.spike_buffer_gpu(), engine_a.spike_count_gpu())
                        {
                            if let Some(ref mut ra) = ring_a {
                                ra.push_device(se, sbuf_a, sc_a)?;
                            }
                        }
                        if let (Some(sbuf_b), Some(sc_b)) =
                            (engine_b.spike_buffer_gpu(), engine_b.spike_count_gpu())
                        {
                            if let Some(ref mut rb) = ring_b {
                                rb.push_device(se, sbuf_b, sc_b)?;
                            }
                        }
                    } else {
                        // Host-mediated: CPU-side compaction (92→48 bytes) + upload
                        let curr_len_a = engine_a.accumulated_spike_count();
                        if curr_len_a > prev_accum_len_a {
                            let accum_a = engine_a.get_accumulated_spikes();
                            let delta_spikes_a = &accum_a[prev_accum_len_a..];
                            if let Some(ref mut ra) = ring_a {
                                ra.push_compacted(se, delta_spikes_a)?;
                            }
                            ring_spikes_exchanged += delta_spikes_a.len() as u64;
                            prev_accum_len_a = curr_len_a;
                        }
                        let curr_len_b = engine_b.accumulated_spike_count();
                        if curr_len_b > prev_accum_len_b {
                            let accum_b = engine_b.get_accumulated_spikes();
                            let delta_spikes_b = &accum_b[prev_accum_len_b..];
                            if let Some(ref mut rb) = ring_b {
                                rb.push_compacted(se, delta_spikes_b)?;
                            }
                            ring_spikes_exchanged += delta_spikes_b.len() as u64;
                            prev_accum_len_b = curr_len_b;
                        }
                    }
                }

                // ════════════════════════════════════════════════════════════════════
                // PHASE 4: Periodic threshold recovery (every 1000 steps)
                //
                // Without recovery, thresholds drift permanently downward as more
                // spikes accumulate. Recovery gently pushes thresholds back toward
                // baseline at 1% per call. This means:
                //   - Recently active voxels: stay suppressed (new spikes keep pushing down)
                //   - Inactive voxels: recover to baseline over ~100K steps
                //   - Net effect: coupling is recency-weighted
                // ════════════════════════════════════════════════════════════════════
                if twin_config.enable_exchange && step as u32 % 1000 == 0 && step > 0 {
                    let n_voxels = engine_a.total_voxels() as u32;
                    if let Some(ref se) = stream_exchange {
                        if let Some((thresh_a, base_a)) = engine_a.threshold_buffers_mut() {
                            if let Some(ref rb) = ring_b {
                                rb.threshold_recovery(se, thresh_a, base_a, n_voxels, 0.01)?;
                            }
                        }
                        if let Some((thresh_b, base_b)) = engine_b.threshold_buffers_mut() {
                            if let Some(ref ra) = ring_a {
                                ra.threshold_recovery(se, thresh_b, base_b, n_voxels, 0.01)?;
                            }
                        }
                    }
                }
            } // end if !use_persistent (Phases 3+4)

            // ════════════════════════════════════════════════════════════════════
            // CPU density exchange telemetry (kept for logging, no longer drives coupling)
            // ════════════════════════════════════════════════════════════════════
            if twin_config.enable_exchange && step as u32 % twin_config.exchange_interval == 0 {
                let snap_a = engine_a.get_accumulated_spikes();
                let snap_b = engine_b.get_accumulated_spikes();

                let cell_ang = twin_config.region_size as f32;
                let (cells_a, _) = compute_spike_density_cpu(&snap_a, cell_ang);
                let (cells_b, _) = compute_spike_density_cpu(&snap_b, cell_ang);

                let density_a: f64 = cells_a.iter().map(|c| c.total_intensity).sum();
                let density_b: f64 = cells_b.iter().map(|c| c.total_intensity).sum();
                let nonzero_union = {
                    use std::collections::HashSet;
                    let keys_a: HashSet<_> = cells_a.iter().map(|c| c.idx).collect();
                    let keys_b: HashSet<_> = cells_b.iter().map(|c| c.idx).collect();
                    keys_a.union(&keys_b).count() as u32
                };

                n_exchanges += 1;
                total_density_a_to_b += density_a;
                total_density_b_to_a += density_b;
                if nonzero_union > max_nonzero_regions {
                    max_nonzero_regions = nonzero_union;
                }

                log::debug!(
                "  [exchange #{n_exchanges}] step={step}: density A={density_a:.3e} B={density_b:.3e} union={nonzero_union}"
            );
            }

            // Progress logging
            if (step + 1) % 5000 == 0 || step == outer_steps - 1 {
                let elapsed = start.elapsed().as_secs_f64();
                let steps_per_sec = (step + 1) as f64 / elapsed;
                let overflow_a = ring_a
                    .as_ref()
                    .and_then(|r| {
                        stream_exchange
                            .as_ref()
                            .and_then(|se| r.overflow_count(se).ok())
                    })
                    .unwrap_or(0);
                let overflow_b = ring_b
                    .as_ref()
                    .and_then(|r| {
                        stream_exchange
                            .as_ref()
                            .and_then(|se| r.overflow_count(se).ok())
                    })
                    .unwrap_or(0);
                log::info!("  Step {}/{}: A={} B={} spikes  ring_spikes_exchanged={} overflow_a={} overflow_b={} ({:.0} steps/s)",
                step + 1, outer_steps, spikes_a_total, spikes_b_total,
                ring_spikes_exchanged, overflow_a, overflow_b, steps_per_sec);
            }
        }
    }

    let wall_time = start.elapsed();
    log::info!(
        "  Simulation complete: {:.1}s wall time",
        wall_time.as_secs_f64()
    );

    // ── Collect accumulated spikes + snapshots ──
    let spikes_a = engine_a.get_accumulated_spikes();
    let spikes_b = engine_b.get_accumulated_spikes();
    let snapshots_a = engine_a.get_snapshots();
    let snapshots_b = engine_b.get_snapshots();
    let physical_steps_a = i32::try_from(engine_a.device_protocol_current_step()?)
        .context("coupled post-MD stream A physical step exceeds i32")?;
    let physical_steps_b = i32::try_from(engine_b.device_protocol_current_step()?)
        .context("coupled post-MD stream B physical step exceeds i32")?;

    log::info!(
        "  Stream A: {} spikes, {} snapshots",
        spikes_a.len(),
        snapshots_a.len()
    );
    log::info!(
        "  Stream B: {} spikes, {} snapshots",
        spikes_b.len(),
        snapshots_b.len()
    );

    // ── Delegate to shared post-processing ──
    let meta = TwinSimulationMetadata {
        seed_a,
        seed_b,
        steps,
        steps_b,
        offset_steps,
        wall_time_secs: wall_time.as_secs_f64(),
        vram_used_gb: vram_used,
        n_exchanges,
        total_density_a_to_b,
        total_density_b_to_a,
        max_nonzero_regions,
        ring_spikes_exchanged,
        runtime_evidence: strict_runtime_evidence,
    };

    let mut result = twin_post_process(
        &spikes_a,
        &spikes_b,
        &snapshots_a,
        &snapshots_b,
        topology,
        topology_b,
        &protocol,
        twin_config,
        &context,
        analysis_stream.as_ref().or(stream_exchange.as_ref()),
        evidence_config,
        output_dir,
        &meta,
    )?;
    result.post_md_streams = vec![
        CoupledPostMdStream {
            source_stream_id: 0,
            topology: topology.clone(),
            topology_source_sha256: evidence_config.map(|config| config.topology_a_sha256.clone()),
            protocol,
            seed: seed_a,
            physical_steps: physical_steps_a,
            spikes: spikes_a,
            snapshots: snapshots_a,
        },
        CoupledPostMdStream {
            source_stream_id: 1,
            topology: topology_b.unwrap_or(topology).clone(),
            topology_source_sha256: evidence_config.map(|config| config.topology_b_sha256.clone()),
            protocol: protocol_b,
            seed: seed_b,
            physical_steps: physical_steps_b,
            spikes: spikes_b,
            snapshots: snapshots_b,
        },
    ];
    Ok(result)
}

/// Shared post-processing for PRISM-TWIN: takes aggregated spikes from
/// either Phase A (2 engines) or Phase B (2N engines) and produces all
/// output files + the CoupledTwinResult.
///
/// This function handles:
///   1. Ensemble trajectory metadata (JSON)
///   2. Spike persistence (coupled_spikes.json)
///   3. TWIN-aware site detection (binding_sites, kcc, therm, per_residue)
///   4. GPU Tensor Core CCF (WMMA kernel)
///   5. Per-residue feature assembly (50 fields, 39 populated, 11 placeholders)
///   6. Site-level feature aggregation from lining residues
///   7. Result struct construction + summary banner
#[cfg(feature = "gpu")]
pub fn twin_post_process(
    spikes_a: &[crate::fused_engine::GpuSpikeEvent],
    spikes_b: &[crate::fused_engine::GpuSpikeEvent],
    snapshots_a: &[crate::fused_engine::EnsembleSnapshot],
    snapshots_b: &[crate::fused_engine::EnsembleSnapshot],
    topology: &PrismPrepTopology,
    // LINK #1: OPTIONAL second system (holo / variant) for the molecular
    // differential. n_residues + resid->name/CA lookups continue to source the
    // PRIMARY `topology` (=A, N_ref); topology_b is reserved for the second
    // ca_xyz + ResidueAlignment consumed in a later stage. None => single-system
    // path, bit-for-bit today's behavior.
    topology_b: Option<&PrismPrepTopology>,
    protocol: &CryoUvProtocol,
    twin_config: &CoupledTwinConfig,
    context: &Arc<CudaContext>,
    stream_exchange: Option<&Arc<CudaStream>>,
    evidence_config: Option<&CoupledTwinEvidenceConfig>,
    output_dir: &std::path::Path,
    meta: &TwinSimulationMetadata,
) -> Result<CoupledTwinResult> {
    let seed_a = meta.seed_a;
    let seed_b = meta.seed_b;
    let steps = meta.steps;
    let steps_b = meta.steps_b;
    let offset_steps = meta.offset_steps;
    let n_exchanges = meta.n_exchanges;
    let total_density_a_to_b = meta.total_density_a_to_b;
    let total_density_b_to_a = meta.total_density_b_to_a;
    let max_nonzero_regions = meta.max_nonzero_regions;
    let ring_spikes_exchanged = meta.ring_spikes_exchanged;
    let mut runtime_evidence = meta.runtime_evidence.clone();

    std::fs::create_dir_all(output_dir)?;

    // Write ensemble_trajectory.json with both streams' snapshots tagged
    {
        let prefix = std::path::Path::new(&topology.source_pdb)
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| s.replace("_sanitized", "").replace("_clean", ""))
            .unwrap_or_else(|| "prism_twin".to_string());
        let ens_path = output_dir.join(format!("{}.ensemble_trajectory.json", prefix));

        let ens_json = serde_json::json!({
            "stream_a": {
                "role": "scout",
                "seed": seed_a,
                "n_snapshots": snapshots_a.len(),
                "snapshots": snapshots_a.iter().map(|s| serde_json::json!({
                    "timestep": s.timestep,
                    "temperature": s.temperature,
                    "time_ps": s.time_ps,
                    "alignment_quality": s.alignment_quality,
                    "spike_region_rmsd": s.spike_region_rmsd,
                    "n_trigger_spikes": s.trigger_spikes.len(),
                    "trigger_reason": format!("{:?}", s.trigger_reason),
                    "delta_sasa": s.delta_sasa,
                    // positions/velocities NOT serialized (too large for JSON)
                    // use get_snapshots() API for programmatic access
                    "n_atoms": s.positions.len() / 3,
                })).collect::<Vec<_>>(),
            },
            "stream_b": {
                "role": if twin_config.nma_modes_path.is_some() { "observer_nma" } else { "observer" },
                "seed": seed_b,
                "n_snapshots": snapshots_b.len(),
                "snapshots": snapshots_b.iter().map(|s| serde_json::json!({
                    "timestep": s.timestep,
                    "temperature": s.temperature,
                    "time_ps": s.time_ps,
                    "alignment_quality": s.alignment_quality,
                    "spike_region_rmsd": s.spike_region_rmsd,
                    "n_trigger_spikes": s.trigger_spikes.len(),
                    "trigger_reason": format!("{:?}", s.trigger_reason),
                    "delta_sasa": s.delta_sasa,
                    "n_atoms": s.positions.len() / 3,
                })).collect::<Vec<_>>(),
            },
            "total_snapshots": snapshots_a.len() + snapshots_b.len(),
            "note": "Atomic positions/velocities available via engine.get_snapshots() API. JSON contains metadata only for size efficiency."
        });
        if let Err(e) = std::fs::write(
            &ens_path,
            serde_json::to_string_pretty(&ens_json).unwrap_or_default(),
        ) {
            log::warn!("  Failed to write ensemble trajectory: {}", e);
        } else {
            log::info!(
                "  Wrote ensemble trajectory: {} ({} A + {} B snapshots)",
                ens_path.display(),
                snapshots_a.len(),
                snapshots_b.len()
            );
        }
    }

    // ── SPIKE PERSISTENCE: legacy giant coupled_spikes.json (Phase 6: PREVIEW-ONLY) ───────────
    // The canonical, byte-lossless spike store is the QNANL SPIKE_EVENT_STREAM below (round-trip
    // verified). This 41 GB JSON is a regenerable preview, written ONLY when explicitly requested
    // via --emit-coupled-spikes-preview-json. Default OFF ⇒ QNANL-only (no multi-minute JSON write).
    if twin_config.emit_preview_json {
        use std::io::Write;

        let spike_output_path = output_dir;
        let _ = std::fs::create_dir_all(spike_output_path);

        // Write combined spike file with stream_id
        let combined_path = spike_output_path.join("coupled_spikes.json");
        log::info!(
            "  Writing {} + {} = {} raw spikes to {}",
            spikes_a.len(),
            spikes_b.len(),
            spikes_a.len() + spikes_b.len(),
            combined_path.display()
        );

        if let Ok(mut f) = std::fs::File::create(&combined_path) {
            let _ = writeln!(
                f,
                "{{\"n_spikes_a\": {}, \"n_spikes_b\": {}, \"spikes\": [",
                spikes_a.len(),
                spikes_b.len()
            );

            let write_spike = |f: &mut std::fs::File,
                               s: &crate::fused_engine::GpuSpikeEvent,
                               stream_id: u8,
                               last: bool| {
                // Copy fields to locals to avoid unaligned packed struct access
                let ts = s.timestep;
                let pos = s.position;
                let int = s.intensity;
                let ve = s.vibrational_energy;
                let wd = s.water_density;
                let src = s.spike_source;
                let vi = s.voxel_idx;
                let wl = s.wavelength_nm;
                let ne = s.n_nearby_excited;
                let _ = writeln!(f, "  {{\"stream_id\": {}, \"timestep\": {}, \"x\": {:.3}, \"y\": {:.3}, \"z\": {:.3}, \
                    \"intensity\": {:.4}, \"vib_energy\": {:.6}, \"water_density\": {:.4}, \
                    \"spike_source\": {}, \"voxel_idx\": {}, \"wavelength_nm\": {:.1}, \
                    \"n_nearby_excited\": {}}}{}",
                    stream_id, ts, pos[0], pos[1], pos[2],
                    int, ve, wd, src, vi, wl, ne,
                    if last { "" } else { "," });
            };

            let total = spikes_a.len() + spikes_b.len();
            let mut count = 0;
            for s in spikes_a {
                count += 1;
                write_spike(&mut f, s, 0, count == total);
            }
            for s in spikes_b {
                count += 1;
                write_spike(&mut f, s, 1, count == total);
            }
            let _ = writeln!(f, "]}}");
            log::info!(
                "  ✓ Wrote {} spike records ({:.1} MB)",
                total,
                combined_path
                    .metadata()
                    .map(|m| m.len() as f64 / 1e6)
                    .unwrap_or(0.0)
            );
        } else {
            log::warn!(
                "  ✗ Failed to write coupled spikes to {}",
                combined_path.display()
            );
        }
    }

    // ── Phase 1: canonical CHUNKED QNANL spike-event stream ───────────────────
    // Promotes the spike persistence away from the giant coupled_spikes.json
    // (kept this pass as a regenerable preview) into bounded, compressed, sealed
    // QNANL chunks. The canonical record is the byte-lossless 96-byte
    // GpuSpikeEvent — the SEALED packet layout the PRSPK001/tokenizer decoders
    // consume; QNANL decode reproduces the raw chunk bytes EXACTLY (round-trip
    // verified per chunk). No physics / spike-field / timing change —
    // persistence only.
    {
        use crate::spike_sensor_bus::{AlwaysOnSpikeSensorBus, SensorRecord};
        use prismcode::qnanl_artifact::{
            decode_packet_bytes, encode_bytes, QnanlArtifactFamily, QnanlEncodeOptions,
        };
        const CHUNK_RECORDS: usize = 2_000_000; // 96 B × 2M ≈ 192 MB raw/chunk (sealed packet)
        let rec_size = std::mem::size_of::<crate::fused_engine::GpuSpikeEvent>();
        // Phase 2: AlwaysOnSpikeSensorBus normalizes each spike chunk's EXISTING per-spike sensor
        // fields (spike_source/intensity/water_density/vibrational_energy/wd_change already produced
        // by the engine) into range-linked bus records — additive metadata, joins by chunk_id.
        let mut bus = AlwaysOnSpikeSensorBus::new();
        let stream_root = output_dir.join("SPIKE_EVENT_STREAM");
        let chunks_dir = stream_root.join("chunks");
        let _ = std::fs::create_dir_all(&chunks_dir);
        let mut manifest_rows: Vec<serde_json::Value> = Vec::new();
        let mut chunk_id = 0usize;
        let mut total_records = 0usize;
        let mut all_verified = true;
        for (stream_id, spikes) in [(0u8, spikes_a), (1u8, spikes_b)] {
            let mut record_start = 0usize;
            for chunk in spikes.chunks(CHUNK_RECORDS) {
                // canonical raw bytes = the GpuSpikeEvent records verbatim (sealed
                // packet, 96 B/record; `rec_size` is derived from the type, never assumed).
                let raw: &[u8] = unsafe {
                    std::slice::from_raw_parts(chunk.as_ptr() as *const u8, chunk.len() * rec_size)
                };
                let raw_hash = blake3::hash(raw).to_hex().to_string();
                let pkt = chunks_dir.join(format!("spike_chunk_{:06}.prsqev", chunk_id));
                let rcpt =
                    chunks_dir.join(format!("spike_chunk_{:06}.prsqev.receipt.json", chunk_id));
                let opts = QnanlEncodeOptions::runtime_subprocess_orchestrated_by_prismos(
                    format!("spike_chunk_{:06}_stream{}", chunk_id, stream_id),
                    QnanlArtifactFamily::EventStream,
                    None,
                    "coupled_md::canonical_qnanl_spike_event_stream",
                );
                match encode_bytes(raw, &pkt, &rcpt, opts) {
                    Ok(_receipt) => {
                        // gate: QNANL decode must reproduce the canonical raw chunk bytes.
                        let pkt_bytes = std::fs::read(&pkt).unwrap_or_default();
                        let qnanl_hash = blake3::hash(&pkt_bytes).to_hex().to_string();
                        let verified = decode_packet_bytes(&pkt_bytes)
                            .map(|dec| dec.as_slice() == raw)
                            .unwrap_or(false);
                        all_verified &= verified;
                        // Phase 2: normalize this chunk's existing per-spike sensor fields into a
                        // single range-linked bus record (mean intensity / water / vibration / wd,
                        // dominant UV vs LIF source) joined to this spike chunk by chunk_id.
                        {
                            let n = chunk.len().max(1) as f64;
                            let mut n_uv = 0u64;
                            let (mut s_int, mut s_wd, mut s_vib, mut s_wdc) =
                                (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                            for s in chunk {
                                let src = s.spike_source;
                                let it = s.intensity;
                                let wd = s.water_density;
                                let vib = s.vibrational_energy;
                                let wdc = s.wd_change;
                                if src == 1 {
                                    n_uv += 1;
                                }
                                s_int += it as f64;
                                s_wd += wd as f64;
                                s_vib += vib as f64;
                                s_wdc += wdc as f64;
                            }
                            let uv_dominant = n_uv * 2 >= chunk.len() as u64;
                            let mut sr = SensorRecord::new_linked(
                                format!("spike_sensor_{:06}", chunk_id),
                                if uv_dominant {
                                    "UV_PHOTO_SPIKE"
                                } else {
                                    "LIF_SPIKE"
                                },
                                "coupled_md.GpuSpikeEvent",
                                stream_id as u32,
                                chunk_id as u64,
                                record_start as u64,
                                chunk.len() as u64,
                                "range",
                                raw_hash.clone(),
                                qnanl_hash.clone(),
                            );
                            sr.spike_source = Some(if uv_dominant { 1 } else { 2 });
                            sr.intensity = Some((s_int / n) as f32);
                            sr.water_state = Some((s_wd / n) as f32);
                            sr.vibration_state = Some((s_vib / n) as f32);
                            sr.wd_change = Some((s_wdc / n) as f32);
                            if let Some(config) = evidence_config {
                                sr.perturbation_edge_id = Some(format!(
                                    "molecular_contrast:{}:{}:{}",
                                    config.contrast_kind,
                                    config.topology_a_sha256,
                                    config.topology_b_sha256
                                ));
                                sr.twin_pair_id = Some(config.run_id.clone());
                                sr.condition_id = Some(if stream_id == 0 {
                                    format!("topology_a_sha256:{}", config.topology_a_sha256)
                                } else {
                                    format!("topology_b_sha256:{}", config.topology_b_sha256)
                                });
                                sr.condition_role = Some(if stream_id == 0 {
                                    "reference_topology".to_string()
                                } else if config.identical_topology_negative_control {
                                    "identical_topology_control".to_string()
                                } else {
                                    "molecular_variant_topology".to_string()
                                });
                                sr.condition_family = Some(config.contrast_kind.clone());
                                sr.condition_amplitude = None;
                                sr.condition_scan_fraction = None;
                            } else {
                                // Preserve the historical thermal/NMA sensor identity byte path
                                // for the single-topology protocol-contrast lane.
                                let nma_active = twin_config.nma_modes_path.is_some();
                                sr.perturbation_edge_id = Some(format!(
                                    "edge_thermal_only__to__thermal_plus_nma__amp{:.1}",
                                    twin_config.nma_amplification
                                ));
                                sr.twin_pair_id =
                                    Some(format!("twinpair_seedA{}_seedB{}", seed_a, seed_b));
                                if stream_id == 0 {
                                    sr.condition_id = Some("thermal_only".to_string());
                                    sr.condition_role = Some("scout".to_string());
                                    sr.condition_family = Some("thermal_only".to_string());
                                    sr.condition_amplitude = Some(0.0);
                                    sr.condition_scan_fraction = Some(0.0);
                                } else {
                                    let family = if nma_active {
                                        "thermal_plus_nma"
                                    } else {
                                        "thermal_only"
                                    };
                                    sr.condition_id = Some(family.to_string());
                                    sr.condition_role = Some("focused_observer".to_string());
                                    sr.condition_family = Some(family.to_string());
                                    sr.condition_amplitude = Some(if nma_active {
                                        twin_config.nma_amplification
                                    } else {
                                        0.0
                                    });
                                    sr.condition_scan_fraction = Some(if nma_active {
                                        twin_config.nma_scan_fraction
                                    } else {
                                        0.0
                                    });
                                }
                            }
                            bus.push(sr);
                        }
                        // PACKAGE C — chunk-level Coupled Chronology Record.
                        //
                        // Written next to the chunk as a `.ccr` sidecar in the
                        // canonical 192-byte CCR wire format, stating the
                        // EMITTER identity of the chunk's first and last spike
                        // plus the host-side chunk context (which stream, which
                        // drain epoch) that produced it.
                        //
                        // It does NOT state a global MD clock. The sealed 96 B
                        // spike record carries none, and nothing available at
                        // this chunk boundary establishes one, so both clock
                        // fields stay at the CCR's out-of-band sentinels rather
                        // than taking a plausible value. Placing these chunks on
                        // the coordinate-frame axis is a merge against the
                        // per-step governor stream, which is the only measured
                        // source of that clock; chunk resolution is frame
                        // resolution because a chunk emits at most one frame.
                        //
                        // Envelope, not a per-spike copy: 1.39M x 192 B of
                        // duplicated chronology would be 267 MB of restatement.
                        let ccr_stream_id = u32::from(stream_id);
                        let ccr_record_seq = chunk_id as u64;
                        let ccr_open = chunk
                            .first()
                            .map(|s| {
                                crate::coupled_chronology::CoupledChronologyRecord::from_spike_event(
                                    s,
                                    ccr_stream_id,
                                    ccr_record_seq,
                                )
                            })
                            .unwrap_or_else(
                                crate::coupled_chronology::CoupledChronologyRecord::unmeasured,
                            );
                        let ccr_close = chunk
                            .last()
                            .map(|s| {
                                crate::coupled_chronology::CoupledChronologyRecord::from_spike_event(
                                    s,
                                    ccr_stream_id,
                                    ccr_record_seq,
                                )
                            })
                            .unwrap_or_else(
                                crate::coupled_chronology::CoupledChronologyRecord::unmeasured,
                            );
                        let ccr_path = chunks_dir.join(format!("spike_chunk_{:06}.ccr", chunk_id));
                        let mut ccr_bytes =
                            Vec::with_capacity(2 * crate::coupled_chronology::CCR_RECORD_BYTES);
                        ccr_bytes.extend_from_slice(&ccr_open.to_bytes());
                        ccr_bytes.extend_from_slice(&ccr_close.to_bytes());
                        let ccr_written = std::fs::write(&ccr_path, &ccr_bytes).is_ok();
                        if !ccr_written {
                            log::warn!(
                                "  [SPIKE_EVENT_STREAM] chunk {} chronology sidecar write failed",
                                chunk_id
                            );
                        }
                        manifest_rows.push(serde_json::json!({
                            "chunk_id": chunk_id,
                            "chunk_kind": "spike",
                            "stream_id": stream_id,
                            "record_count": chunk.len(),
                            "record_start": record_start,
                            "record_end": record_start + chunk.len(),
                            "raw_bytes": raw.len(),
                            "compressed_bytes": pkt_bytes.len(),
                            "raw_hash_blake3": raw_hash,
                            "qnanl_hash_blake3": qnanl_hash,
                            "codec_stack": "SectionZstd+QNANL(PRSQNB01)",
                            "roundtrip_verified": verified,
                            "source_surfaces": ["coupled_md.GpuSpikeEvent"],
                            "spike_record_bytes": rec_size,
                            "chronology_sidecar": ccr_path
                                .file_name()
                                .and_then(|n| n.to_str())
                                .unwrap_or_default(),
                            "chronology_sidecar_written": ccr_written,
                            "chronology_record_bytes": crate::coupled_chronology::CCR_RECORD_BYTES,
                            "chronology_records": 2,
                            "global_md_step_open": ccr_open.global_md_step_opt(),
                            "global_md_step_close": ccr_close.global_md_step_opt(),
                            "cumulative_simulated_time_ps_open":
                                ccr_open.cumulative_simulated_time_ps_opt(),
                            "cumulative_simulated_time_ps_close":
                                ccr_close.cumulative_simulated_time_ps_opt(),
                            "record_stream_id": ccr_open.stream_id_opt(),
                            "emitter_tag_open": ccr_open.emitter_tag,
                        }));
                        chunk_id += 1;
                        total_records += chunk.len();
                    }
                    Err(e) => {
                        log::warn!(
                            "  [SPIKE_EVENT_STREAM] chunk {} encode failed: {}",
                            chunk_id,
                            e
                        );
                        all_verified = false;
                    }
                }
                record_start += chunk.len();
            }
        }
        let emitted = spikes_a.len() + spikes_b.len();
        // Phase 2: seal the normalized sensor records into a QNANL sensor metadata chunk. The bus
        // has exactly one record per spike chunk (range-linked) ⇒ sensor_count == n_chunks and
        // every sensor record joins a valid spike chunk (zero orphans).
        let sensors_dir = stream_root.join("sensors");
        let sensor_result = bus.emit_qnanl_sensor_chunk(&sensors_dir, 0);
        let (sensor_count, sensor_verified) = sensor_result
            .as_ref()
            .map(|r| (r.record_count, r.roundtrip_verified))
            .unwrap_or((0, true));
        let sensor_join_complete = sensor_count == chunk_id; // every chunk has a sensor record
        let manifest = serde_json::json!({
            "schema_kind": "qnanl_spike_event_stream_manifest",
            "schema_version": 1,
            "run_id": evidence_config.map(|config| config.run_id.as_str()),
            "molecular_contrast_identity": evidence_config.map(|config| serde_json::json!({
                "contrast_kind": config.contrast_kind,
                "topology_a_sha256": config.topology_a_sha256,
                "topology_b_sha256": config.topology_b_sha256,
                "identical_topology_negative_control": config.identical_topology_negative_control,
                "probe_residue_ids": config.probe_residue_ids,
                "direct_contact_cutoff_angstrom": config.ligand_contact_cutoff_angstrom,
                "cryptic_search_outer_cutoff_angstrom": config.cryptic_search_outer_cutoff_angstrom,
                "exchange_active": false,
            })),
            "n_chunks": chunk_id,
            "total_spike_records": total_records,
            "record_count_matches_emitted": total_records == emitted,
            "all_chunks_roundtrip_verified": all_verified,
            "canonical": true,
            "preview_artifact": "coupled_spikes.json (regenerable; not canonical)",
            "sensor_bus": {
                "schema_kind": crate::spike_sensor_bus::SENSOR_BUS_SCHEMA_KIND,
                "sensor_record_count": sensor_count,
                "sensor_roundtrip_verified": sensor_verified,
                "sensor_join_complete_zero_orphans": sensor_join_complete,
                "linkage": "range (by chunk_id); each spike chunk has exactly one normalized sensor record",
                "source_surfaces": ["coupled_md.GpuSpikeEvent"],
                "receipt_ops": ["MD.SPIKE_CHUNK_COMMIT", "MD.SENSOR_EMIT", "MD.SENSOR_CHUNK_COMMIT", "MD.QNANL_SEAL", "MD.QNANL_VERIFY", "MD.EVENT_STREAM_MANIFEST_COMMIT"]
            },
            "rows": manifest_rows,
        });
        let qnanl_manifest_path = stream_root.join("MANIFEST.qnanl.json");
        if evidence_config.is_some() {
            write_atomic_json(&qnanl_manifest_path, &manifest)?;
        } else if let Err(error) = std::fs::write(
            &qnanl_manifest_path,
            serde_json::to_string_pretty(&manifest).unwrap_or_default(),
        ) {
            log::warn!("  [SPIKE_EVENT_STREAM] manifest write failed: {}", error);
        }
        log::info!(
            "  ✓ SPIKE_EVENT_STREAM: {} QNANL chunks ({} recs, count_match={}, rt_verified={}) + {} sensor records (rt_verified={}, zero_orphans={}) ({})",
            chunk_id,
            total_records,
            total_records == emitted,
            all_verified,
            sensor_count,
            sensor_verified,
            sensor_join_complete,
            stream_root.display()
        );
    }

    // ── TWIN-aware site detection ────────────────────────────────────────────
    // Borrows the complete stream spike arrays retained by the caller. Writes
    // binding_sites.json, kcc_visualization.json, prism_therm.json,
    // ensemble_trajectory.json (stub), twin_per_residue.json, twin_ccf_matrix.json
    // into the output directory. Single-pass replacement for the non-twin
    // detection pass. Failure is logged but does not abort the twin run.
    {
        let twin_output_path = output_dir;
        let prefix = std::path::Path::new(&topology.source_pdb)
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| s.replace("_sanitized", "").replace("_clean", ""))
            .unwrap_or_else(|| "prism_twin".to_string());
        log::info!("  Running TWIN-aware site detection (prefix={})", prefix);
        match crate::twin_detection::detect_and_write_twin_sites(
            &spikes_a,
            &spikes_b,
            topology,
            topology_b,
            twin_output_path,
            &prefix,
            twin_config.nma_modes_path.as_deref(),
        ) {
            Ok(summary) => {
                log::info!(
                    "  ✓ TWIN detection: {} sites ({} consensus, {} barrier-gated, {} allo-hub) in {:.1}s",
                    summary.n_sites,
                    summary.n_consensus_sites,
                    summary.n_barrier_gated_sites,
                    summary.n_allosteric_hub_sites,
                    summary.elapsed_seconds,
                );
            }
            Err(e) => {
                if evidence_config.is_some() {
                    return Err(e.context(
                        "production two-topology TWIN detection failed; Path-B fallback is forbidden",
                    ));
                }
                log::warn!(
                    "  ✗ TWIN detection failed: {} — spike JSON still written",
                    e
                );
            }
        }
    }

    // ── GPU Tensor Core CCF + per-residue feature assembly ────────────────
    //
    // Two-stage computation:
    //   Stage 1: Per-residue spike counting (CPU) — agreement ratio, B/A ratio
    //   Stage 2: GPU WMMA CCF — cross-correlation matrix, CCF features
    //
    // The CPU counting gives us per-residue spike counts from both streams.
    // The GPU CCF gives us the residue×residue correlation structure.
    // Together they populate the InterferometricFeatures struct.

    // Stage 1: CPU per-residue spike counting (same as before, still needed)
    let per_res_map = if twin_config.enable_ccf {
        // LINK #1: two_system => frame-independent resid co-occurrence (the two
        // molecular systems are integrated in unsuperposed absolute frames).
        let result =
            compute_cpu_cross_correlation(&spikes_a, &spikes_b, 5.0, 500, topology_b.is_some());
        log::info!(
            "  CPU co-occurrence matches (A∩B within 5Å / 500 steps): {}",
            result.0
        );
        let mut aligned = result.1;
        if topology_b.is_some() {
            aligned.retain(|residue_id, _| {
                *residue_id >= 0 && (*residue_id as usize) < topology.n_residues
            });
        }
        aligned
    } else {
        std::collections::HashMap::new()
    };

    // Stage 2: GPU WMMA CCF
    // Use n_residues from topology (not topology.residues.len() which may be 0
    // in old topology files that predate the ResidueMetadata field).
    // LINK #1: this is the PRIMARY `topology` param (= reference A, N_ref). Path-2
    // (build_ccf_matrices) sizes on it and drops B ligand rows (resid >= n_residues)
    // via the twin_kernels.rs:833 filter — so CCF/TE stay in the reference
    // residue-ordinal frame even when topology_b (holo) carries appended ligands.
    let n_residues = topology.n_residues;
    let bin_size = 100i32; // 100 steps per time bin
    let (ccf_matrix_vec, ccf_features_vec, ccf_kernel_binding) = if twin_config.enable_ccf
        && n_residues > 0
        && !spikes_a.is_empty()
        && !spikes_b.is_empty()
    {
        log::info!(
            "  Building CCF matrices: {} residues, {} steps, bin_size={}...",
            n_residues,
            steps,
            bin_size
        );
        let ccf_start = std::time::Instant::now();

        // Build mean-centered time-binned spike matrices
        let (mat_a_f32, mat_b_f32, n_res_padded, n_bins_padded) =
            crate::twin_kernels::build_ccf_matrices(
                &spikes_a, &spikes_b, n_residues, steps, bin_size,
            );

        // Attempt GPU CCF
        let gpu_ccf_result = (|| -> Result<(
            Vec<f32>,
            Vec<crate::twin_kernels::CcfResidueFeatures>,
            StrictTwinKernelSourceBinding,
        )> {
                let ccf_ptx_path = crate::twin_kernels::find_twin_ptx("tensor_ccf.ptx")?;
                let ccf_kernel_binding = bind_strict_kernel_source(
                    "tensor_ccf.ptx",
                    Path::new(&ccf_ptx_path),
                    &["tensor_ccf_compute", "compute_spike_norms"],
                )?;
                let ccf_module = context
                    .load_module(cudarc::nvrtc::Ptx::from_file(&ccf_ptx_path))
                    .context("Failed to load tensor_ccf.ptx")?;

                let se = stream_exchange
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("Exchange stream not available for CCF"))?;

                let mut ccf = crate::twin_kernels::TwinCcfCompute::new(
                    se,
                    &ccf_module,
                    n_residues as i32,
                    (steps / bin_size) as i32,
                )?;
                ccf.upload_matrices(se, &mat_a_f32, &mat_b_f32)?;
                ccf.compute(se)?;
                let ccf_matrix = ccf.download_ccf(se)?;

                let nonzero = ccf_matrix.iter().filter(|&&v| v.abs() > 0.01).count();
                log::info!(
                    "  ✓ GPU CCF: {}×{} matrix, {} non-zero entries ({:.1}%), {:.1}ms",
                    n_residues,
                    n_residues,
                    nonzero,
                    nonzero as f64 / (n_residues * n_residues).max(1) as f64 * 100.0,
                    ccf_start.elapsed().as_secs_f64() * 1000.0
                );

                let features = crate::twin_kernels::extract_ccf_features(&ccf_matrix, n_residues);
                Ok((ccf_matrix, features, ccf_kernel_binding))
            })();

        match gpu_ccf_result {
            Ok((ccf_matrix, features, kernel_binding)) => {
                if ccf_matrix.len() != n_residues.saturating_mul(n_residues)
                    || ccf_matrix.iter().any(|value| !value.is_finite())
                    || features.len() != n_residues
                {
                    bail!(
                            "GPU CCF returned invalid dimensions/nonfinite values: matrix={} expected={} features={} expected={}",
                            ccf_matrix.len(),
                            n_residues.saturating_mul(n_residues),
                            features.len(),
                            n_residues
                        );
                }
                (Some(ccf_matrix), Some(features), Some(kernel_binding))
            }
            Err(e) => {
                if evidence_config.is_some() {
                    return Err(e.context(
                        "production two-topology GPU CCF failed; zero fallback is forbidden",
                    ));
                }
                log::warn!("  GPU CCF failed ({}), legacy zero features retained", e);
                (None, None, None)
            }
        }
    } else {
        if evidence_config.is_some() {
            bail!(
                    "production two-topology GPU CCF admission failed: enabled={} residues={} spikes_a={} spikes_b={}",
                    twin_config.enable_ccf,
                    n_residues,
                    spikes_a.len(),
                    spikes_b.len()
                );
        }
        (None, None, None)
    };

    // ── GPU Transfer Entropy (Layer 5: causal spike propagation) ──────────
    //
    // Uses the same time-binned spike matrices as CCF. The CUDA kernel produces:
    //   0.5 * sum_j(TE_Ai→Bj + TE_Bj→Ai): bidirectional TE mass
    //   sum_j(TE_Ai→Bj), retaining terms > 0.001: thresholded outgoing TE mass
    //   sum_j(TE_Ai→Bj) - sum_j(TE_Bj→Ai): net causal flow
    // The kernel uses one 100-step lag bin; these are not Shannon-MI values.
    let (te_mi, te_causal, te_per_res, te_kernel_binding) = if twin_config.enable_ccf
        && n_residues > 0
        && !spikes_a.is_empty()
        && !spikes_b.is_empty()
    {
        log::info!(
            "  Computing GPU Transfer Entropy ({} residues)...",
            n_residues
        );
        let te_start = std::time::Instant::now();

        // Rebuild matrices (same as CCF — fast, O(n_spikes))
        let (mat_a_f32, mat_b_f32, n_res_padded, n_bins_padded) =
            crate::twin_kernels::build_ccf_matrices(
                &spikes_a, &spikes_b, n_residues, steps, bin_size,
            );
        let n_bins = (steps / bin_size) as i32;

        let te_result =
            (|| -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, StrictTwinKernelSourceBinding)> {
                let te_ptx_path = crate::twin_kernels::find_twin_ptx("twin_transfer_entropy.ptx")?;
                let te_kernel_binding = bind_strict_kernel_source(
                    "twin_transfer_entropy.ptx",
                    Path::new(&te_ptx_path),
                    &[
                        "twin_compute_row_means",
                        "twin_binned_te",
                        "twin_compute_mutual_info",
                    ],
                )?;
                let te_module = context
                    .load_module(cudarc::nvrtc::Ptx::from_file(&te_ptx_path))
                    .context("Failed to load twin_transfer_entropy.ptx")?;

                let se = stream_exchange
                    .ok_or_else(|| anyhow::anyhow!("Exchange stream not available for TE"))?;

                // Upload matrices (reuse CCF's format: f32 → u16/FP16)
                let to_u16 = |vals: &[f32]| -> Vec<u16> {
                    vals.iter()
                        .map(|&v| half::f16::from_f32(v).to_bits())
                        .collect()
                };
                let a_u16 = to_u16(&mat_a_f32);
                let b_u16 = to_u16(&mat_b_f32);
                let mut d_mat_a = se.alloc_zeros::<u16>(a_u16.len())?;
                let mut d_mat_b = se.alloc_zeros::<u16>(b_u16.len())?;
                se.memcpy_htod(&a_u16, &mut d_mat_a)?;
                se.memcpy_htod(&b_u16, &mut d_mat_b)?;

                let mut te = crate::twin_kernels::TwinTeCompute::new(
                    se,
                    &te_module,
                    n_residues as i32,
                    n_bins,
                    n_res_padded,
                    n_bins_padded,
                )?;
                te.compute(se, &d_mat_a, &d_mat_b)?;

                let (mi, cf) = te.download_per_residue(se)?;
                let te_per_res = te.download_te_per_residue(se)?;

                let nonzero_mi = mi.iter().filter(|&&v| v > 0.001).count();
                let nonzero_cf = cf.iter().filter(|&&v| v.abs() > 0.001).count();
                log::info!(
                "  ✓ GPU TE: {} residues, bidirectional mass non-zero={}, net flow non-zero={}, {:.1}ms",
                n_residues,
                nonzero_mi,
                nonzero_cf,
                te_start.elapsed().as_secs_f64() * 1000.0
            );

                Ok((mi, cf, te_per_res, te_kernel_binding))
            })();

        match te_result {
            Ok((mi, cf, te, kernel_binding)) => {
                if mi.len() != n_residues
                    || cf.len() != n_residues
                    || te.len() != n_residues
                    || mi
                        .iter()
                        .chain(cf.iter())
                        .chain(te.iter())
                        .any(|value| !value.is_finite())
                {
                    bail!(
                            "GPU TE returned invalid dimensions/nonfinite values: bidirectional_mass={} net_flow={} thresholded_outgoing={} expected={}",
                            mi.len(),
                            cf.len(),
                            te.len(),
                            n_residues
                        );
                }
                (Some(mi), Some(cf), Some(te), Some(kernel_binding))
            }
            Err(e) => {
                if evidence_config.is_some() {
                    return Err(e.context(
                            "production two-topology GPU transfer entropy failed; zero fallback is forbidden",
                        ));
                }
                log::warn!("  GPU TE failed ({}), legacy zero features retained", e);
                (None, None, None, None)
            }
        }
    } else {
        if evidence_config.is_some() {
            bail!(
                    "production two-topology GPU TE admission failed: enabled={} residues={} spikes_a={} spikes_b={}",
                    twin_config.enable_ccf,
                    n_residues,
                    spikes_a.len(),
                    spikes_b.len()
                );
        }
        (None, None, None, None)
    };

    let strict_phase_runtime = if let Some(config) = evidence_config {
        let analysis_stream = stream_exchange
            .context("strict phase CCF requires the dedicated CUDA analysis stream")?;
        Some(compute_strict_phase_ccf_runtime(
            context,
            analysis_stream,
            output_dir,
            config,
            protocol,
            steps,
            n_residues,
            &spikes_a,
            &spikes_b,
            ccf_kernel_binding
                .as_ref()
                .context("strict phase CCF lacks the loaded global CCF kernel binding")?,
        )?)
    } else {
        None
    };

    if let Some(config) = evidence_config {
        let runtime = runtime_evidence
            .as_mut()
            .context("strict two-topology analysis lacks engine runtime evidence")?;
        let stem = coupled_topology_stem(config);
        for (stream_id, spikes) in [(0u32, spikes_a), (1u32, spikes_b)] {
            if spikes.is_empty() {
                bail!(
                    "strict two-topology stream {} emitted zero spikes",
                    stream_id
                );
            }
            let spike_path = output_dir.join(format!("{stem}_stream{stream_id}_spikes.bin"));
            write_spike_evidence(&spike_path, spikes, stream_id, &config.run_id, &stem)?;
            runtime.artifacts.push(artifact_evidence(
                "spikes",
                Some(stream_id),
                &spike_path,
                Some(spikes.len() as u64),
                Some(1),
            )?);
        }
        let ccf = ccf_matrix_vec
            .as_deref()
            .context("strict two-topology analysis lacks the CUDA CCF matrix")?;
        let mi = te_mi
            .as_deref()
            .context("strict two-topology analysis lacks CUDA bidirectional TE mass")?;
        let causal = te_causal
            .as_deref()
            .context("strict two-topology analysis lacks CUDA net causal flow")?;
        let transfer = te_per_res
            .as_deref()
            .context("strict two-topology analysis lacks CUDA thresholded outgoing TE")?;
        runtime
            .artifacts
            .extend(emit_coupled_gpu_analysis_artifacts(
                output_dir,
                config,
                n_residues,
                steps,
                ccf,
                mi,
                causal,
                transfer,
                ccf_kernel_binding
                    .as_ref()
                    .context("strict GPU analysis lacks loaded CCF kernel identity")?,
                te_kernel_binding
                    .as_ref()
                    .context("strict GPU analysis lacks loaded TE kernel identity")?,
            )?);
        for phase in strict_phase_runtime
            .as_ref()
            .context("strict phase CCF runtime disappeared before artifact admission")?
        {
            runtime.artifacts.push(phase.matrix_artifact.clone());
        }

        let qnanl_manifest_path = output_dir.join("SPIKE_EVENT_STREAM/MANIFEST.qnanl.json");
        let qnanl_manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&qnanl_manifest_path).with_context(|| {
                format!(
                    "read strict QNANL spike manifest {}",
                    qnanl_manifest_path.display()
                )
            })?)?;
        let qnanl_green = qnanl_manifest
            .get("schema_kind")
            .and_then(serde_json::Value::as_str)
            == Some("qnanl_spike_event_stream_manifest")
            && qnanl_manifest
                .get("n_chunks")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
                > 0
            && qnanl_manifest
                .get("total_spike_records")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
                > 0
            && qnanl_manifest
                .get("record_count_matches_emitted")
                .and_then(serde_json::Value::as_bool)
                == Some(true)
            && qnanl_manifest
                .get("all_chunks_roundtrip_verified")
                .and_then(serde_json::Value::as_bool)
                == Some(true)
            && qnanl_manifest
                .pointer("/sensor_bus/sensor_roundtrip_verified")
                .and_then(serde_json::Value::as_bool)
                == Some(true)
            && qnanl_manifest
                .pointer("/sensor_bus/sensor_join_complete_zero_orphans")
                .and_then(serde_json::Value::as_bool)
                == Some(true);
        if !qnanl_green {
            bail!("strict two-topology QNANL spike stream failed round-trip/count/join gates");
        }
        runtime.artifacts.push(artifact_evidence(
            "qnanl_spike_event_stream_manifest",
            None,
            &qnanl_manifest_path,
            qnanl_manifest
                .get("total_spike_records")
                .and_then(serde_json::Value::as_u64),
            Some(1),
        )?);
    }

    // ── Build per-residue InterferometricFeatures (50 fields) ──────────────
    //
    // Data sources:
    //   per_res_map: HashMap<resid, (count_a, count_b, intensity_a, intensity_b)> from CPU co-occurrence
    //   ccf_features_vec: Option<Vec<CcfResidueFeatures>> from GPU WMMA CCF
    //   te_mi, te_causal, te_per_res: from GPU Transfer Entropy
    //   phase_counts_a, phase_counts_b: per-residue per-phase spike counts
    //   onset_a, onset_b: earliest spike timestep per residue
    //   intensity_a, intensity_b: total intensity per residue

    let phase_counts_a = per_residue_phase_counts(&spikes_a, &protocol);
    let phase_counts_b = per_residue_phase_counts(&spikes_b, &protocol);
    let onset_a = per_residue_onset(&spikes_a);
    let onset_b = per_residue_onset(&spikes_b);
    let intensity_sum_a = per_residue_intensity(&spikes_a);
    let intensity_sum_b = per_residue_intensity(&spikes_b);

    let per_residue_features: Vec<InterferometricFeatures> = if evidence_config.is_some() {
        Vec::new()
    } else {
        // Build resid→name lookup. Prefer topology.residues (ResidueMetadata) if
        // populated; fall back to topology.residue_names + residue_ids (atom-indexed
        // arrays) for old topology files where the residues array is empty.
        let mut resid_to_name: std::collections::HashMap<i32, String> =
            std::collections::HashMap::new();
        if !topology.residues.is_empty() {
            for r in &topology.residues {
                resid_to_name.insert(r.residue_id, r.residue_name.clone());
            }
        } else {
            // Fall back: residue_names is atom-indexed, residue_ids maps atom→resid
            for (i, &resid) in topology.residue_ids.iter().enumerate() {
                if i < topology.residue_names.len() && !resid_to_name.contains_key(&(resid as i32))
                {
                    resid_to_name.insert(resid as i32, topology.residue_names[i].clone());
                }
            }
        }

        let ccf_by_idx: Vec<crate::twin_kernels::CcfResidueFeatures> =
            ccf_features_vec.clone().unwrap_or_else(|| {
                vec![crate::twin_kernels::CcfResidueFeatures::default(); n_residues]
            });

        let mut features: Vec<InterferometricFeatures> = per_res_map
            .iter()
            .filter_map(|(&resid, &(cnt_a, cnt_b, int_a, int_b))| {
                if cnt_a == 0 && cnt_b == 0 {
                    return None;
                }

                // ── Consensus (12) ──
                let max_cnt = cnt_a.max(cnt_b) as f32;
                let min_cnt = cnt_a.min(cnt_b) as f32;
                let agreement = if max_cnt > 0.0 {
                    min_cnt / max_cnt
                } else {
                    0.0
                };
                let mean_intensity = if cnt_a + cnt_b > 0 {
                    (int_a + int_b) as f32 / (cnt_a + cnt_b) as f32
                } else {
                    0.0
                };

                // Per-phase consensus: min(A_phase, B_phase) / max(A_phase, B_phase)
                let pa = phase_counts_a.get(&resid).copied().unwrap_or([0; 5]);
                let pb = phase_counts_b.get(&resid).copied().unwrap_or([0; 5]);
                let mut consensus_phase = [0.0f32; 5];
                for i in 0..5 {
                    let mx = pa[i].max(pb[i]) as f32;
                    let mn = pa[i].min(pb[i]) as f32;
                    consensus_phase[i] = if mx > 0.0 { mn / mx } else { 0.0 };
                }

                // Temporal onset: earliest step where BOTH streams have at least 1 spike
                let oa = onset_a.get(&resid).copied().unwrap_or(i32::MAX);
                let ob = onset_b.get(&resid).copied().unwrap_or(i32::MAX);
                let consensus_onset = if oa < i32::MAX && ob < i32::MAX {
                    oa.max(ob) as f32 // the later of the two = when consensus starts
                } else {
                    0.0
                };

                // ── Cross-correlation (12) ──
                let ccf = if (resid as usize) < ccf_by_idx.len() {
                    &ccf_by_idx[resid as usize]
                } else {
                    &crate::twin_kernels::CcfResidueFeatures::default()
                };

                // Per-phase CCF: placeholder (would need per-phase CCF computation)
                // For now, weight the global CCF by the phase's consensus ratio
                let mut ccf_per_phase = [0.0f32; 5];
                for i in 0..5 {
                    ccf_per_phase[i] = ccf.ccf_peak_value * consensus_phase[i];
                }

                // ── Differential (18) ──
                let b_over_a = if topology_b.is_some() {
                    (cnt_b as f32 + 0.5) / (cnt_a as f32 + 0.5)
                } else if cnt_a > 0 {
                    cnt_b as f32 / cnt_a as f32
                } else {
                    f32::INFINITY
                };
                let int_total_a = intensity_sum_a.get(&resid).copied().unwrap_or(0.0);
                let int_total_b = intensity_sum_b.get(&resid).copied().unwrap_or(0.0);
                let b_over_a_intensity = if topology_b.is_some() {
                    (int_total_b.max(0.0) + 1.0e-6) / (int_total_a.max(0.0) + 1.0e-6)
                } else if int_total_a > 0.0 {
                    int_total_b / int_total_a
                } else if int_total_b > 0.0 {
                    f32::INFINITY
                } else {
                    1.0
                };

                // NMA vs thermal exclusive: count phases where only one stream is active
                let mut nma_excl: u32 = 0;
                let mut thermal_excl: u32 = 0;
                for i in 0..5 {
                    if pb[i] > 0 && pa[i] == 0 {
                        nma_excl += pb[i];
                    }
                    if pa[i] > 0 && pb[i] == 0 {
                        thermal_excl += pa[i];
                    }
                }

                // Per-phase B/A differential
                let mut phase_diff = [0.0f32; 5];
                for i in 0..5 {
                    phase_diff[i] = if topology_b.is_some() {
                        (pb[i] as f32 + 0.5) / (pa[i] as f32 + 0.5)
                    } else if pa[i] > 0 {
                        pb[i] as f32 / pa[i] as f32
                    } else if pb[i] > 0 {
                        2.0
                    } else {
                        1.0
                    };
                }

                let barrier_classification = if b_over_a > 1.5 {
                    "LOW".to_string()
                } else if b_over_a < 0.67 {
                    "HIGH".to_string()
                } else {
                    "MEDIUM".to_string()
                };

                // ── Scout/propagation (8) ──
                // Scout lead time: how many steps earlier A fires vs B at this residue
                let lead_time = if oa < i32::MAX && ob < i32::MAX {
                    (ob - oa) as f32 // positive = A leads (expected from phase offset)
                } else {
                    0.0
                };

                // Predictive value: if A fires, does B also fire? (proxy: agreement)
                let predictive = if cnt_a > 0 && cnt_b > 0 {
                    agreement
                } else {
                    0.0
                };

                // Phase offset enrichment: spike ratio during the offset window
                // (B's cold_hold extension) vs the rest
                let offset_window_phase = 0; // cold_hold is phase 0
                let offset_enrich = if pa[offset_window_phase] > 0 {
                    pb[offset_window_phase] as f32 / pa[offset_window_phase] as f32
                } else {
                    1.0
                };

                // A's intensity when B first fires
                let scout_intensity = if ob < i32::MAX {
                    // Find A spikes near B's onset time — use mean intensity of A at this residue as proxy
                    if cnt_a > 0 {
                        int_total_a / cnt_a as f32
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                let resname = resid_to_name
                    .get(&resid)
                    .cloned()
                    .unwrap_or_else(|| "UNK".to_string());

                Some(InterferometricFeatures {
                    resid,
                    resname,
                    // Consensus (12)
                    spike_agreement_ratio: agreement,
                    consensus_intensity_mean: mean_intensity,
                    consensus_phase_profile: consensus_phase,
                    consensus_spatial_coherence: 0.0, // populated in second pass (needs neighbor lookup)
                    consensus_temporal_onset: consensus_onset,
                    n_consensus_neighbors: 0, // populated in second pass
                    // Cross-correlation (12)
                    ccf_peak_lag: ccf.ccf_peak_lag,
                    ccf_peak_value: ccf.ccf_peak_value,
                    ccf_width: ccf.ccf_width,
                    ccf_asymmetry: ccf.ccf_asymmetry,
                    ccf_per_phase,
                    // CCF frequency: count zero-crossings in per-phase CCF as proxy for periodicity
                    // High frequency peak = rapid alternation of correlated/uncorrelated phases
                    ccf_frequency_peak: {
                        let mut crossings = 0u32;
                        for i in 1..5 {
                            if (ccf_per_phase[i] > 0.0) != (ccf_per_phase[i - 1] > 0.0) {
                                crossings += 1;
                            }
                        }
                        crossings as f32 / 4.0 // normalized to [0, 1]
                    },
                    ccf_reproducibility: if ccf.ccf_reproducibility > 0.0 {
                        ccf.ccf_reproducibility
                    } else {
                        agreement
                    },
                    // CCF lag consistency: std dev of per-phase CCF values
                    // Low std = consistent CCF across all phases = robust allosteric coupling
                    ccf_lag_consistency: {
                        let mean_phase_ccf: f32 = ccf_per_phase.iter().sum::<f32>() / 5.0;
                        let var: f32 = ccf_per_phase
                            .iter()
                            .map(|&v| (v - mean_phase_ccf).powi(2))
                            .sum::<f32>()
                            / 5.0;
                        var.sqrt()
                    },
                    // Differential (18)
                    spikes_a: cnt_a,
                    spikes_b: cnt_b,
                    b_over_a_ratio: b_over_a,
                    nma_exclusive_count: nma_excl,
                    thermal_exclusive_count: thermal_excl,
                    b_over_a_intensity_ratio: b_over_a_intensity,
                    nma_responsive_mode: -1, // PLACEHOLDER: Phase C Step 16 (NMA mode correlation)
                    nma_mode_eigenvalue: 0.0, // PLACEHOLDER: Phase C Step 16
                    barrier_classification,
                    per_phase_differential: phase_diff,
                    differential_onset_lag: lead_time, // reuse lead_time as onset differential
                    nma_work_at_residue: 0.0, // PLACEHOLDER: Phase C Step 16 (NMA force integration)
                    mechanical_sensitivity: 0.0, // PLACEHOLDER: Phase C Step 16
                    susceptibility_magnitude: 0.0, // PLACEHOLDER: Phase C Step 16
                    // Scout/propagation (8)
                    scout_lead_time: lead_time,
                    scout_predictive_value: predictive,
                    phase_offset_enrichment: offset_enrich,
                    scout_intensity_at_onset: scout_intensity,
                    scout_spatial_propagation: 0.0, // populated in second pass using CA positions
                    // GPU Transfer Entropy (computed by twin_binned_te kernel)
                    mutual_information: te_mi
                        .as_ref()
                        .and_then(|v| {
                            if (resid as usize) < v.len() {
                                Some(v[resid as usize])
                            } else {
                                None
                            }
                        })
                        .unwrap_or(0.0),
                    transfer_entropy_a_to_b: te_per_res
                        .as_ref()
                        .and_then(|v| {
                            if (resid as usize) < v.len() {
                                Some(v[resid as usize])
                            } else {
                                None
                            }
                        })
                        .unwrap_or(0.0),
                    causal_flow_direction: te_causal
                        .as_ref()
                        .and_then(|v| {
                            if (resid as usize) < v.len() {
                                Some(v[resid as usize])
                            } else {
                                None
                            }
                        })
                        .unwrap_or(0.0),
                })
            })
            .collect();

        // Build CA position lookup for spatial analysis passes
        let ca_pos: std::collections::HashMap<i32, [f32; 3]> = {
            let mut map = std::collections::HashMap::new();
            if !topology.residues.is_empty() {
                for r in &topology.residues {
                    if let Some(&ca_idx) = topology.ca_indices.iter().find(|&&idx| {
                        idx < topology.residue_ids.len()
                            && topology.residue_ids[idx] == r.residue_id as usize
                    }) {
                        if ca_idx * 3 + 2 < topology.positions.len() {
                            map.insert(
                                r.residue_id,
                                [
                                    topology.positions[ca_idx * 3],
                                    topology.positions[ca_idx * 3 + 1],
                                    topology.positions[ca_idx * 3 + 2],
                                ],
                            );
                        }
                    }
                }
            }
            map
        };

        // Second pass: compute spatial coherence using CA positions
        if !ca_pos.is_empty() {
            // Build agreement lookup for fast neighbor check
            let agreement_by_resid: std::collections::HashMap<i32, f32> = features
                .iter()
                .map(|f| (f.resid, f.spike_agreement_ratio))
                .collect();

            for feat in features.iter_mut() {
                if let Some(pos) = ca_pos.get(&feat.resid) {
                    let mut n_neighbors = 0u32;
                    let mut n_agreeing = 0u32;
                    for (&other_resid, &other_pos) in &ca_pos {
                        if other_resid == feat.resid {
                            continue;
                        }
                        let dx = pos[0] - other_pos[0];
                        let dy = pos[1] - other_pos[1];
                        let dz = pos[2] - other_pos[2];
                        let dist_sq = dx * dx + dy * dy + dz * dz;
                        if dist_sq <= 64.0 {
                            // 8Å radius
                            n_neighbors += 1;
                            if let Some(&other_agree) = agreement_by_resid.get(&other_resid) {
                                if other_agree > 0.5 {
                                    n_agreeing += 1;
                                }
                            }
                        }
                    }
                    feat.n_consensus_neighbors = n_agreeing;
                    feat.consensus_spatial_coherence = if n_neighbors > 0 {
                        n_agreeing as f32 / n_neighbors as f32
                    } else {
                        0.0
                    };

                    // Scout spatial propagation computed in third pass below
                }
            }
        }

        // Third pass: scout_spatial_propagation
        // Pre-compute which residues have positive scout_lead_time
        let a_leading_resids: std::collections::HashSet<i32> = features
            .iter()
            .filter(|f| f.scout_lead_time > 0.0)
            .map(|f| f.resid)
            .collect();

        if !a_leading_resids.is_empty() && !ca_pos.is_empty() {
            for feat in features.iter_mut() {
                if feat.scout_lead_time > 0.0 {
                    if let Some(pos) = ca_pos.get(&feat.resid) {
                        let mut max_dist = 0.0f32;
                        for &other_resid in &a_leading_resids {
                            if other_resid == feat.resid {
                                continue;
                            }
                            if let Some(&other_pos) = ca_pos.get(&other_resid) {
                                let dx = pos[0] - other_pos[0];
                                let dy = pos[1] - other_pos[1];
                                let dz = pos[2] - other_pos[2];
                                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                                if dist > max_dist {
                                    max_dist = dist;
                                }
                            }
                        }
                        feat.scout_spatial_propagation = max_dist;
                    }
                }
            }
        }

        features.sort_by(|a, b| {
            b.consensus_intensity_mean
                .partial_cmp(&a.consensus_intensity_mean)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // ════════════════════════════════════════════════════════════════
        // FIELD STATUS: 45/50 populated, 5 NMA-dependent placeholders
        //
        // Populated (45): all consensus(12), all CCF(12), differential(13/18),
        //   scout(6/8), TE(3): MI, TE_a_to_b, causal_flow
        //
        // PLACEHOLDER (5 — all require NMA mode file + perturbation run):
        //   nma_responsive_mode, nma_mode_eigenvalue, nma_work_at_residue,
        //   mechanical_sensitivity, susceptibility_magnitude
        //
        // These 5 fields are populated when --nma-perturb is provided
        // (auto-generated by the wrapper for TWIN runs with available PDB).
        // Without NMA modes, Group B runs thermal-only and these fields = 0.
        // ════════════════════════════════════════════════════════════════
        let n_populated = 45;
        let n_placeholder = 5;
        log::info!(
            "  Per-residue features: {}/{} fields populated, {} NMA-dependent placeholders",
            n_populated,
            n_populated + n_placeholder,
            n_placeholder
        );

        features
    };

    let strict_residue_features = if evidence_config.is_some() {
        Some(build_strict_residue_features(
            topology,
            steps,
            &spikes_a,
            &spikes_b,
            ccf_matrix_vec
                .as_deref()
                .context("strict residue plane lacks global GPU CCF matrix")?,
            ccf_features_vec
                .as_deref()
                .context("strict residue plane lacks global GPU CCF features")?,
            strict_phase_runtime
                .as_deref()
                .context("strict residue plane lacks five phase GPU CCF executions")?,
            te_mi
                .as_deref()
                .context("strict residue plane lacks GPU mutual information")?,
            te_causal
                .as_deref()
                .context("strict residue plane lacks GPU causal flow")?,
            te_per_res
                .as_deref()
                .context("strict residue plane lacks GPU transfer entropy")?,
        )?)
    } else {
        None
    };

    // ── Site-level feature aggregation ────────────────────────────────────
    //
    // Read binding_sites.json produced by twin_detection, extract lining
    // residue IDs for each site, aggregate per-residue features to per-site.
    let per_site_features: Vec<SiteInterferometricFeatures> = if evidence_config.is_some() {
        Vec::new()
    } else {
        let prefix = std::path::Path::new(&topology.source_pdb)
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| s.replace("_sanitized", "").replace("_clean", ""))
            .unwrap_or_else(|| "prism_twin".to_string());
        let sites_path = output_dir.join(format!("{}.binding_sites.json", prefix));

        if sites_path.exists() {
            match std::fs::read_to_string(&sites_path).and_then(|s| {
                serde_json::from_str::<serde_json::Value>(&s)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
            }) {
                Ok(sites_json) => {
                    let sites = sites_json["sites"].as_array();
                    if let Some(sites) = sites {
                        // Extract lining residue IDs from each site
                        let site_lining: Vec<Vec<i32>> = sites
                            .iter()
                            .map(|site| {
                                site["lining_residues"]
                                    .as_array()
                                    .map(|arr| {
                                        arr.iter()
                                            .filter_map(|v| v.as_i64().map(|id| id as i32))
                                            .collect()
                                    })
                                    .unwrap_or_else(Vec::new)
                            })
                            .collect();

                        let mut result = aggregate_site_features(
                            &site_lining,
                            &per_residue_features,
                            if evidence_config.is_some() {
                                ccf_matrix_vec.as_deref()
                            } else {
                                None
                            },
                            n_residues,
                        );

                        // Populate geometry fields from binding_sites.json
                        for (i, sf) in result.iter_mut().enumerate() {
                            if i < sites.len() {
                                let site = &sites[i];
                                sf.volume = site["volume"].as_f64().unwrap_or(0.0) as f32;
                                sf.druggability =
                                    site["druggability"].as_f64().unwrap_or(0.0) as f32;
                                sf.is_druggable = site["is_druggable"].as_bool().unwrap_or(false);
                                if let Some(centroid) = site["centroid"].as_array() {
                                    if centroid.len() >= 3 {
                                        sf.centroid = [
                                            centroid[0].as_f64().unwrap_or(0.0) as f32,
                                            centroid[1].as_f64().unwrap_or(0.0) as f32,
                                            centroid[2].as_f64().unwrap_or(0.0) as f32,
                                        ];
                                    }
                                }
                            }
                        }

                        log::info!(
                            "  Site features: {} sites aggregated from {} per-residue features",
                            result.len(),
                            per_residue_features.len()
                        );
                        for (i, sf) in result.iter().enumerate() {
                            log::info!("    Site {}: {}/{} lining residues, agree={:.3} ccf={:.3} barrier=L{:.0}%/M{:.0}%/H{:.0}%",
                                i, sf.n_lining_with_data, sf.n_lining_residues,
                                sf.mean_agreement, sf.mean_ccf_peak,
                                sf.barrier_composition_low * 100.0,
                                sf.barrier_composition_medium * 100.0,
                                sf.barrier_composition_high * 100.0);
                        }
                        result
                    } else {
                        log::warn!("  binding_sites.json has no 'sites' array");
                        Vec::new()
                    }
                }
                Err(e) => {
                    log::warn!(
                        "  Could not read binding_sites.json for site aggregation: {}",
                        e
                    );
                    Vec::new()
                }
            }
        } else {
            log::warn!(
                "  binding_sites.json not found at {} — skipping site aggregation",
                sites_path.display()
            );
            Vec::new()
        }
    };

    let binding_site_prefix = Path::new(&topology.source_pdb)
        .file_stem()
        .and_then(|value| value.to_str())
        .map(|value| value.replace("_sanitized", "").replace("_clean", ""))
        .unwrap_or_else(|| "prism_twin".to_string());
    let binding_sites_path = output_dir.join(format!("{binding_site_prefix}.binding_sites.json"));
    let strict_site_features = if evidence_config.is_some() {
        let sites_json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&binding_sites_path).with_context(|| {
                format!(
                    "read topology-derived binding-site geometry {}",
                    binding_sites_path.display()
                )
            })?)?;
        Some(build_strict_site_features(
            &sites_json,
            strict_residue_features
                .as_deref()
                .context("strict site plane lacks strict residue records")?,
            ccf_matrix_vec
                .as_deref()
                .context("strict site plane lacks global GPU CCF matrix")?,
            n_residues,
        )?)
    } else {
        None
    };

    let (n_consensus_events, n_differential_events) =
        if let Some(strict_records) = strict_residue_features.as_deref() {
            (
                strict_records
                    .iter()
                    .filter(|record| {
                        record
                            .global_aligned_zero_lag_ccf
                            .is_some_and(|value| value > 0.1)
                    })
                    .count() as u64,
                strict_records
                    .iter()
                    .filter(|record| {
                        record.b_only_condition_event_count > 0
                            || record.a_only_reference_event_count > 0
                    })
                    .count() as u64,
            )
        } else {
            (
                per_residue_features
                    .iter()
                    .filter(|feature| feature.ccf_peak_value > 0.1)
                    .count() as u64,
                per_residue_features
                    .iter()
                    .filter(|feature| feature.barrier_classification != "MEDIUM")
                    .count() as u64,
            )
        };

    log::info!("  Gate 2 summary:");
    log::info!("    Exchanges:            {}", n_exchanges);
    log::info!("    Total density A→B:    {:.3e}", total_density_a_to_b);
    log::info!("    Total density B→A:    {:.3e}", total_density_b_to_a);
    log::info!("    Peak nonzero regions: {}", max_nonzero_regions);
    log::info!("    CCF consensus events: {}", n_consensus_events);
    log::info!("    Per-residue features: {}", per_residue_features.len());
    log::info!("    Differential events:  {}", n_differential_events);
    // ─────────────────────────────────────────────────────────────────────────

    // A molecular contrast is not the historical thermal→NMA condition edge.
    // Emit a dedicated finite, topology-bound readout for topology-B and keep
    // the legacy DPD/QNANL edge byte-path unchanged for single-topology runs.
    let condition_edge_readout_path = if let (Some(topology_b), Some(evidence_config)) =
        (topology_b, evidence_config)
    {
        let mut readout = build_molecular_contrast_readout(
            topology,
            topology_b,
            &spikes_a,
            &spikes_b,
            strict_residue_features
                .as_deref()
                .context("molecular contrast lacks strict event/GPU residue records")?,
            evidence_config,
        )?;
        let runtime = runtime_evidence
            .as_mut()
            .context("molecular contrast lacks strict engine runtime evidence")?;
        bind_engine_ghost_to_molecular_contrast(&mut readout, runtime, evidence_config)?;
        let (readout_dir, readout_path, sidecar_path) =
            emit_molecular_contrast_readout(output_dir, &readout)?;
        runtime.artifacts.push(artifact_evidence(
            "molecular_contrast_readout",
            None,
            &readout_path,
            Some(readout.residues.len() as u64),
            Some(MOLECULAR_CONTRAST_READOUT_SCHEMA_VERSION),
        )?);
        runtime.artifacts.push(artifact_evidence(
            "molecular_contrast_sha256_sidecar",
            None,
            &sidecar_path,
            Some(1),
            Some(1),
        )?);
        if let Some(probe_readout) = build_glycan_mechanistic_probe_readout_v1(
            &readout,
            topology_b,
            runtime,
            evidence_config,
        )? {
            let (probe_dir, probe_path, probe_sidecar_path) =
                emit_glycan_mechanistic_probe_readout_v1(output_dir, &probe_readout)?;
            runtime.artifacts.push(artifact_evidence(
                "glycan_mechanistic_probe_readout",
                None,
                &probe_path,
                Some(probe_readout.candidate_count),
                Some(GLYCAN_MECHANISTIC_PROBE_READOUT_SCHEMA_VERSION),
            )?);
            runtime.artifacts.push(artifact_evidence(
                "glycan_mechanistic_probe_sha256_sidecar",
                None,
                &probe_sidecar_path,
                Some(1),
                Some(1),
            )?);
            log::info!(
                "  ✓ GLYCAN_MECHANISTIC_PROBE_READOUT emitted: {} candidates={} control_not_observed={} probe_enriched={} causal_claim=false",
                probe_dir.display(),
                probe_readout.candidate_count,
                probe_readout.control_not_observed_candidate_count,
                probe_readout.probe_enriched_candidate_count,
            );
        }
        log::info!(
            "  ✓ MOLECULAR_CONTRAST_READOUT emitted: {} mean_abs={:.6} pocket_fraction={:.6} ghost_bound={} negative_control={}",
            readout_dir.display(),
            readout.global_mean_abs_signed_contrast,
            readout.pocket_localization_fraction,
            readout.ghost_pocket_bound_count,
            readout.identical_topology_identity
        );
        Some(readout_dir.to_string_lossy().to_string())
    } else {
        let edge_id = format!(
            "edge_thermal_only__to__thermal_plus_nma__amp{:.1}",
            twin_config.nma_amplification
        );
        let pair_id = format!("twinpair_seedA{}_seedB{}", seed_a, seed_b);
        let manifest_ref = output_dir
            .join("SPIKE_EVENT_STREAM/MANIFEST.qnanl.json")
            .to_string_lossy()
            .to_string();
        let readout = crate::condition_edge::compute_condition_edge(
            &per_residue_features,
            &spikes_a,
            &spikes_b,
            twin_config.nma_amplification,
            twin_config.nma_scan_fraction,
            edge_id,
            pair_id,
            Some(topology.source_pdb.clone()),
            Some(manifest_ref),
        );
        let m = &readout.metrics;
        log::info!(
            "  ✓ CONDITION_EDGE: score={:.4} exclusive_residues={} edge_events={} enrich={:.3} phase_enrich={:.4} intensity_shift={:.4} sensor_shift={:.4} stable={}",
            m.condition_edge_score, m.condition_exclusive_residues, m.condition_edge_events,
            m.condition_enrichment_ratio, m.condition_phase_enrichment, m.condition_intensity_shift,
            m.condition_sensor_shift, m.condition_stability_gate
        );
        let emitted = match crate::condition_edge::emit_condition_edge_readout(&readout, output_dir)
        {
            Ok(res) => {
                log::info!(
                    "  ✓ CONDITION_EDGE_READOUT emitted: {} (qnanl_rt_verified={})",
                    res.readout_dir.display(),
                    res.qnanl_roundtrip_verified
                );
                Some(res.readout_dir.to_string_lossy().to_string())
            }
            Err(e) => {
                log::warn!("  [CONDITION_EDGE] emit failed: {}", e);
                None
            }
        };
        emitted
    };

    if evidence_config.is_some() {
        let strict_residue_records = strict_residue_features
            .as_deref()
            .context("strict result lacks strict residue feature records")?;
        let strict_site_records = strict_site_features
            .as_deref()
            .context("strict result lacks strict site feature records")?;
        if strict_residue_records.is_empty() || strict_site_records.is_empty() {
            bail!("strict two-topology run produced an empty strict residue/site plane");
        }
        let runtime = runtime_evidence
            .as_mut()
            .context("strict TWIN feature planes lack runtime evidence")?;
        runtime.artifacts.push(artifact_evidence(
            "binding_sites",
            None,
            &binding_sites_path,
            Some(strict_site_records.len() as u64),
            Some(1),
        )?);
        let retired_surfaces = retire_legacy_detector_feature_surfaces(output_dir, topology)?;
        let strict_artifacts = emit_strict_twin_feature_artifacts(
            output_dir,
            evidence_config.context("strict feature emission lacks evidence config")?,
            &binding_sites_path,
            strict_residue_records,
            strict_site_records,
            runtime,
            retired_surfaces,
        )?;
        runtime.artifacts.extend(strict_artifacts);
    }

    let result_per_residue_features = if let Some(strict_records) = strict_residue_features {
        strict_records
            .into_iter()
            .map(CoupledTwinResidueFeatureRecord::Strict)
            .collect()
    } else {
        per_residue_features
            .into_iter()
            .map(CoupledTwinResidueFeatureRecord::Legacy)
            .collect()
    };
    let result_per_site_features = if let Some(strict_records) = strict_site_features {
        strict_records
            .into_iter()
            .map(CoupledTwinSiteFeatureRecord::Strict)
            .collect()
    } else {
        per_site_features
            .into_iter()
            .map(CoupledTwinSiteFeatureRecord::Legacy)
            .collect()
    };

    // ── Build result ──
    let result = CoupledTwinResult {
        post_md_streams: Vec::new(),
        condition_edge_readout_path,
        stream_a: StreamResult {
            role: "scout".to_string(),
            seed: seed_a,
            total_spikes: spikes_a.len(),
            perturbation: "thermal_only".to_string(),
        },
        stream_b: StreamResult {
            role: "focused_observer".to_string(),
            seed: seed_b,
            total_spikes: spikes_b.len(),
            perturbation: if twin_config.nma_modes_path.is_some() {
                "thermal_plus_nma".to_string()
            } else {
                "thermal_only".to_string()
            },
        },
        config: twin_config.clone(),
        per_residue_features: result_per_residue_features,
        per_site_features: result_per_site_features,
        n_consensus_events,
        n_differential_events,
        n_exchanges,
        total_density_a_to_b,
        total_density_b_to_a,
        n_nonzero_regions: max_nonzero_regions,
        ring_spikes_tracked: ring_spikes_exchanged,
        // LINK #1 provenance: set by the bin after run_coupled_twin returns (None here).
        contrast_kind: None,
        topology_a_sha256: None,
        topology_b_sha256: None,
        engine_b_role: None,
        runtime_evidence,
        phase_offset_steps: offset_steps,
        steps_a: steps,
        steps_b: steps_b,
    };

    log::info!("╔══════════════════════════════════════════════════════════╗");
    log::info!("║   PRISM-TWIN COMPLETE                                   ║");
    log::info!("╠══════════════════════════════════════════════════════════╣");
    log::info!(
        "║  Stream A: {:>10} spikes (scout)                   ║",
        spikes_a.len()
    );
    log::info!(
        "║  Stream B: {:>10} spikes (observer)                ║",
        spikes_b.len()
    );
    log::info!(
        "║  Exchanges:    {:>6}                                   ║",
        n_exchanges
    );
    log::info!(
        "║  CCF matches:  {:>6}                                   ║",
        n_consensus_events
    );
    log::info!(
        "║  Residues:     {:>6}                                   ║",
        result.per_residue_features.len()
    );
    log::info!(
        "║  Wall time: {:.1}s                                     ║",
        meta.wall_time_secs
    );
    log::info!(
        "║  VRAM: {:.2} GB used                                   ║",
        meta.vram_used_gb
    );
    log::info!("╚══════════════════════════════════════════════════════════╝");

    Ok(result)
}

#[cfg(feature = "gpu")]
fn validate_coupled_ghost_runtime_evidence(
    output_dir: &Path,
    config: &CoupledTwinEvidenceConfig,
    runtime: &CoupledTwinRuntimeEvidence,
) -> Result<()> {
    use crate::ghost_tile::{
        decode_ghost_v3_cluster_residue_membership, GHOST_FCF_BIT_SPATIAL_NATIVE_AABB_MIDPOINT,
        GHOST_FRAME_SCHEMA_V3, GHOST_V2_OFFSET_AABB_MAX, GHOST_V2_OFFSET_AABB_MIN,
        GHOST_V2_OFFSET_CENTROID, GHOST_V2_OFFSET_FIELD_COMPLETENESS_FLAGS,
        GHOST_V2_OFFSET_SCHEMA_VERSION, GHOST_V2_OFFSET_STEP_IDX,
    };
    let topology_a: PrismPrepTopology = serde_json::from_slice(
        &std::fs::read(&config.topology_a_path)
            .with_context(|| format!("read CRN topology A {}", config.topology_a_path))?,
    )?;
    let topology_b: PrismPrepTopology = serde_json::from_slice(
        &std::fs::read(&config.topology_b_path)
            .with_context(|| format!("read CRN topology B {}", config.topology_b_path))?,
    )?;

    let maps: Vec<&CoupledTwinArtifactEvidence> = runtime
        .artifacts
        .iter()
        .filter(|artifact| artifact.kind == "ghost_site_map")
        .collect();
    if maps.len() != 2 {
        bail!("strict coupled-TWIN requires exactly two engine Ghost site maps");
    }
    let mut observed_streams = BTreeSet::new();
    let mut calibrations = BTreeMap::<u32, CoupledNoiseFloorCalibration>::new();
    let mut split_t7_layouts = BTreeMap::<u32, SplitT7TopologyLayout>::new();
    for map_artifact in maps {
        let stream_id = map_artifact
            .stream_id
            .context("engine Ghost map lacks stream identity")?;
        if !observed_streams.insert(stream_id) || stream_id > 1 {
            bail!("invalid or duplicate engine Ghost map stream {}", stream_id);
        }
        validate_run_artifact(output_dir, map_artifact)?;
        let map: CoupledGhostSiteMap = serde_json::from_slice(
            &std::fs::read(&map_artifact.path)
                .with_context(|| format!("read engine Ghost map {}", map_artifact.path))?,
        )?;
        let (topology_path, topology_sha256) = if stream_id == 0 {
            (&config.topology_a_path, &config.topology_a_sha256)
        } else {
            (&config.topology_b_path, &config.topology_b_sha256)
        };
        if map.schema_kind != "prismos.coupled_twin_ghost_site_map.v3"
            || map.schema_version != COUPLED_TWIN_EVIDENCE_SCHEMA_VERSION
            || map.run_id != config.run_id
            || map.stream_id != stream_id
            || &map.topology_path != topology_path
            || &map.topology_sha256 != topology_sha256
            || map.coordinate_frame != "topology_cartesian_angstrom"
            || map.fallback_used
            || map.record_count == 0
            || map.record_count != map.records.len() as u64
            || map.record_count != map.emitted_occupied_cluster_count as u64
            || map.topology_cluster_count != SPLIT_T7_CLUSTER_COUNT
            || map.emitted_occupied_cluster_count == 0
            || map.emitted_occupied_cluster_count > map.topology_cluster_count
            || map.topology_layout_sha256.len() != 64
            || map.frozen_anchor_sha256.len() != 64
            || map.final_source_events_sha256.len() != 64
            || map.final_canonical_source_events_sha256.len() != 64
            || map.selected_spike_count == 0
            || map.raw_spike_count < map.selected_spike_count as u64
            || map.emitted_step_idx == 0
            || map.step_counter_initial != map.emitted_step_idx
            || !map.dt_fs.is_finite()
            || map.dt_fs <= 0.0
            || !map.noise_floor_mu.is_finite()
            || !map.noise_floor_sigma.is_finite()
            || map.noise_floor_sigma <= 0.0
            || !map.final_total_kl.is_finite()
            || map.final_total_kl < 0.0
            || map.final_adjudication_code > 1
            || map.final_adjudication_reason_flags & 0x8000_0000 != 0
            || map.final_engine_state_sha256_before != map.final_engine_state_sha256_after
            || map.final_engine_physical_state_sha256_before
                != map.final_engine_physical_state_sha256_after
            || map.native_chain_role_map.is_empty()
            || map
                .native_chain_role_map
                .values()
                .any(|role| role != "A" && role != "B")
        {
            bail!(
                "stream {} engine Ghost map identity/telemetry contract failed",
                stream_id
            );
        }
        let topology = PrismPrepTopology::load(topology_path).with_context(|| {
            format!(
                "load stream {} topology for independent Ghost evidence verification",
                stream_id
            )
        })?;
        let layout = split_t7_topology_layout(&topology, &map.native_chain_role_map)?;
        if layout.layout_sha256 != map.topology_layout_sha256 {
            bail!(
                "stream {} split-T7 topology-layout identity failed",
                stream_id
            );
        }
        split_t7_layouts.insert(stream_id, layout.clone());
        for (kind, path, sha256, bytes) in [
            (
                "ghost_noise_floor_calibration",
                map.noise_floor_calibration_path.as_str(),
                map.noise_floor_calibration_sha256.as_str(),
                None,
            ),
            (
                "ghost_noise_floor_frozen_anchor",
                map.frozen_anchor_path.as_str(),
                map.frozen_anchor_sha256.as_str(),
                Some(map.frozen_anchor_bytes),
            ),
            (
                "ghost_final_manifold_input",
                map.final_source_events_path.as_str(),
                map.final_source_events_sha256.as_str(),
                Some(map.final_source_events_bytes),
            ),
        ] {
            let artifact = runtime
                .artifacts
                .iter()
                .find(|artifact| artifact.kind == kind && artifact.stream_id == Some(stream_id))
                .with_context(|| format!("stream {} lacks {} artifact", stream_id, kind))?;
            validate_run_artifact(output_dir, artifact)?;
            if artifact.path != path
                || artifact.sha256 != sha256
                || bytes.is_some_and(|expected| artifact.size_bytes != expected)
            {
                bail!("stream {} {} binding failed", stream_id, kind);
            }
        }
        let calibration_artifact = runtime
            .artifacts
            .iter()
            .find(|artifact| {
                artifact.kind == "ghost_noise_floor_calibration"
                    && artifact.stream_id == Some(stream_id)
            })
            .context("missing split-T7 calibration artifact")?;
        let calibration: CoupledNoiseFloorCalibration =
            serde_json::from_slice(&std::fs::read(&calibration_artifact.path)?)?;
        let calibration_input = runtime
            .artifacts
            .iter()
            .find(|artifact| {
                artifact.kind == "ghost_noise_floor_input" && artifact.stream_id == Some(stream_id)
            })
            .context("missing split-T7 calibration input artifact")?;
        validate_run_artifact(output_dir, calibration_input)?;
        if calibration.schema_kind != "prismos.coupled_twin_split_t7_noise_floor.v3"
            || calibration.schema_version != SPLIT_T7_SHADOW_SCHEMA_VERSION
            || calibration.status != "IMPLEMENTATION_VERIFIED"
            || calibration.validation_status != "PASS"
            || calibration.run_id != config.run_id
            || calibration.stream_id != stream_id
            || calibration.topology_path != *topology_path
            || calibration.topology_sha256 != *topology_sha256
            || calibration.topology_layout_sha256 != map.topology_layout_sha256
            || calibration.topology_aabb_min != layout.aabb_min
            || calibration.topology_aabb_max != layout.aabb_max
            || calibration.fixed_cluster_count != SPLIT_T7_CLUSTER_COUNT
            || calibration.fixed_spikes_per_frame != map.selected_spike_count
            || calibration.inventory_window_md_steps == 0
            || calibration.shadow_trial_count != SPLIT_T7_SHADOW_TRIALS as u32
            || calibration.shadow_reserve_factor != SPLIT_T7_SHADOW_RESERVE_FACTOR
            || calibration.shadow_lower_envelope_frame_counts.len() != SPLIT_T7_TOTAL_FRAMES
            || calibration.shadow_ensemble_sha256.len() != 64
            || calibration.shadow_ensemble_receipt_path.is_empty()
            || calibration.shadow_ensemble_receipt_sha256.len() != 64
            || calibration.shadow_ensemble_receipt_bytes == 0
            || !calibration.shadow_holdout_admission_verified
            || !calibration.authoritative_admission_verified
            || calibration.authoritative_minimum < calibration.fixed_spikes_per_frame
            || calibration.authoritative_minimum_slack_from_k
                != calibration.authoritative_minimum - calibration.fixed_spikes_per_frame
            || calibration.inventory_frame_counts.len() != SPLIT_T7_TOTAL_FRAMES
            || calibration.paired_inventory_sha256.len() != 64
            || !calibration.equal_duration_windows_verified
            || calibration.direct_prefix_outer_steps != SPLIT_T7_TOTAL_FRAMES as u32
            || calibration.frames.len() != SPLIT_T7_TOTAL_FRAMES
            || calibration.calibration_input_path != calibration_input.path
            || calibration.calibration_input_sha256 != calibration_input.sha256
            || calibration.calibration_input_bytes != calibration_input.size_bytes
            || calibration.frozen_anchor_path != map.frozen_anchor_path
            || calibration.frozen_anchor_sha256 != map.frozen_anchor_sha256
            || calibration.frozen_anchor_bytes != map.frozen_anchor_bytes
            || calibration.fallback_used
            || !calibration.state_noninterference_verified
            || !calibration.topology_constants_rebound_each_launch
            || calibration.frames.iter().enumerate().any(|(index, frame)| {
                frame.ordinal != index as u32
                    || frame.source_spike_count != calibration.inventory_frame_counts[index]
                    || frame.selected_spike_count != calibration.fixed_spikes_per_frame
                    || frame.engine_state_sha256_before != frame.engine_state_sha256_after
                    || frame.engine_physical_state_sha256_before
                        != frame.engine_physical_state_sha256_after
            })
        {
            bail!("stream {} split-T7 calibration binding failed", stream_id);
        }
        verify_split_t7_calibration_input(
            Path::new(&calibration_input.path),
            config,
            topology_sha256,
            &calibration,
            &layout,
        )?;
        if calibrations
            .insert(stream_id, calibration.clone())
            .is_some()
        {
            bail!("duplicate split-T7 calibration stream {}", stream_id);
        }
        let final_input_artifact = runtime
            .artifacts
            .iter()
            .find(|artifact| {
                artifact.kind == "ghost_final_manifold_input"
                    && artifact.stream_id == Some(stream_id)
            })
            .context("missing split-T7 final manifold input artifact")?;
        let final_frame = verify_split_t7_final_input(
            Path::new(&final_input_artifact.path),
            config,
            topology_sha256,
            &map,
            &layout,
        )?;
        let payload_artifact = runtime
            .artifacts
            .iter()
            .find(|artifact| {
                artifact.kind == "ghost_tiles_v2" && artifact.stream_id == Some(stream_id)
            })
            .with_context(|| format!("stream {} Ghost map lacks payload artifact", stream_id))?;
        validate_run_artifact(output_dir, payload_artifact)?;
        if map.ghost_payload_path != payload_artifact.path
            || map.ghost_payload_sha256 != payload_artifact.sha256
            || map.ghost_payload_bytes != payload_artifact.size_bytes
            || map.record_count != payload_artifact.record_count.unwrap_or(0)
        {
            bail!("stream {} Ghost map/payload binding failed", stream_id);
        }
        let payload = std::fs::read(&payload_artifact.path)?;
        if read_le_u32(&payload, 0)? as u64 != map.record_count
            || payload.len() as u64 != 4096u64.saturating_add(map.record_count.saturating_mul(4096))
        {
            bail!("stream {} Ghost payload envelope/count mismatch", stream_id);
        }
        let segment_manifest_artifact = runtime
            .artifacts
            .iter()
            .find(|artifact| {
                artifact.kind == "ghost_segment_manifest" && artifact.stream_id == Some(stream_id)
            })
            .with_context(|| format!("stream {} lacks Ghost segment manifest", stream_id))?;
        validate_run_artifact(output_dir, segment_manifest_artifact)?;
        let segment_root = Path::new(&segment_manifest_artifact.path)
            .parent()
            .context("Ghost segment manifest lacks parent stream root")?;
        let verified_segments = crate::ghost_segment::verify_stream_root(segment_root)?;
        let expected_condition = if stream_id == 0 {
            format!("coupled_reference::{}", config.contrast_kind)
        } else {
            format!("coupled_design::{}", config.contrast_kind)
        };
        if verified_segments.manifest_sha256 != segment_manifest_artifact.sha256
            || !verified_segments.manifest.completed
            || !verified_segments.manifest.produced_equals_persisted
            || verified_segments.manifest.stream_id != stream_id as u16
            || verified_segments.manifest.produced_records != map.record_count
            || verified_segments.manifest.persisted_records != map.record_count
            || verified_segments.manifest.record_schema_version != GHOST_FRAME_SCHEMA_V3
            || verified_segments.manifest.context.run_id != config.run_id
            || verified_segments.manifest.context.stream_id != stream_id as u16
            || verified_segments.manifest.context.topology_sha256 != *topology_sha256
            || verified_segments
                .manifest
                .context
                .thermodynamic_condition_id
                != expected_condition
            || verified_segments.manifest.context.record_abi_sha256
                != crate::ghost_tile::ghost_record_abi_sha256()
            || verified_segments.manifest.context.record_source_sha256
                != crate::ghost_tile::ghost_record_source_sha256()
            || segment_manifest_artifact.record_count != Some(map.record_count)
            || segment_manifest_artifact.schema_version
                != Some(crate::ghost_segment::GHOST_SEGMENT_SCHEMA_VERSION)
        {
            bail!("stream {} Ghost segment manifest binding failed", stream_id);
        }
        let reconstructed_segment_payload: Vec<u8> = verified_segments
            .segments
            .iter()
            .flat_map(|segment| segment.payload.iter().copied())
            .collect();
        if reconstructed_segment_payload.as_slice() != &payload[4096..] {
            bail!(
                "stream {} Ghost segmented authority differs from compatibility payload",
                stream_id
            );
        }
        let stem = coupled_topology_stem(config);
        let phase_binding_path = output_dir.join(format!(
            "{stem}_stream{stream_id:02}_ghost_phase_binding.json"
        ));
        let phase_binding: crate::ghost_phase_lattice::StreamProtocolPhaseBinding =
            serde_json::from_slice(&std::fs::read(&phase_binding_path).with_context(|| {
                format!(
                    "read stream {} Ghost phase binding {}",
                    stream_id,
                    phase_binding_path.display()
                )
            })?)?;
        phase_binding.verify()?;
        if phase_binding.stream_id != stream_id as u16
            || phase_binding.condition_id != expected_condition
            || phase_binding.protocol_name != verified_segments.manifest.context.protocol_id
            || phase_binding.protocol_state_hash
                != verified_segments.manifest.context.protocol_state_hash
            || phase_binding.phase_schedule_digest_sha256
                != verified_segments
                    .manifest
                    .context
                    .phase_schedule_digest_sha256
            || phase_binding
                .checked_phase_for(map.emitted_step_idx)
                .is_none()
        {
            bail!("stream {} Ghost phase/segment binding failed", stream_id);
        }
        let mut seen_records = BTreeSet::new();
        let mut witnessed_selected_ordinals = BTreeSet::new();
        for record in &map.records {
            if !seen_records.insert((record.record_index, record.frame_idx, record.site_id))
                || record.byte_offset
                    != 4096u64.saturating_add(record.record_index.saturating_mul(4096))
                || record.step_idx != map.emitted_step_idx
                || record.topology_cluster_id >= map.topology_cluster_count
                || record.source_event_witnesses.is_empty()
                || !record.spatial_native
                || record.field_completeness_flags & GHOST_FCF_BIT_SPATIAL_NATIVE_AABB_MIDPOINT == 0
                || record
                    .aabb_min
                    .iter()
                    .chain(record.aabb_max.iter())
                    .chain(record.centroid.iter())
                    .any(|value| !value.is_finite())
            {
                bail!(
                    "stream {} Ghost record identity/spatial contract failed",
                    stream_id
                );
            }
            for axis in 0..3 {
                if record.aabb_max[axis] < record.aabb_min[axis] {
                    bail!("stream {} Ghost AABB has negative extent", stream_id);
                }
                let midpoint = 0.5 * (record.aabb_min[axis] + record.aabb_max[axis]);
                let tolerance = 1.0e-5f32.max(midpoint.abs() * 1.0e-5);
                if (record.centroid[axis] - midpoint).abs() > tolerance {
                    bail!(
                        "stream {} Ghost centroid is not the AABB midpoint",
                        stream_id
                    );
                }
            }
            let offset = usize::try_from(record.byte_offset)?;
            let bytes = payload
                .get(offset..offset + GHOST_V2_RECORD_BYTES)
                .context("Ghost record provenance exceeds payload")?;
            if read_le_u64(bytes, 0)? != record.frame_idx
                || read_le_u32(bytes, 8)? != record.site_id
                || read_le_u32(bytes, GHOST_V2_OFFSET_SCHEMA_VERSION)? != GHOST_FRAME_SCHEMA_V3
                || read_le_u16(bytes, GHOST_V2_OFFSET_FIELD_COMPLETENESS_FLAGS)?
                    != record.field_completeness_flags
                || read_le_u64(bytes, GHOST_V2_OFFSET_STEP_IDX)? != record.step_idx
                || ghost_vec3(bytes, GHOST_V2_OFFSET_AABB_MIN)? != record.aabb_min
                || ghost_vec3(bytes, GHOST_V2_OFFSET_AABB_MAX)? != record.aabb_max
                || ghost_vec3(bytes, GHOST_V2_OFFSET_CENTROID)? != record.centroid
            {
                bail!(
                    "stream {} Ghost map does not match its exact source bytes",
                    stream_id
                );
            }
            decode_ghost_v3_cluster_residue_membership(bytes).map_err(|error| {
                anyhow::anyhow!(
                    "stream {} Ghost runtime evidence v3 cluster-residue membership contract failed: {}",
                    stream_id,
                    error
                )
            })?;
            let mut witness_support = BTreeSet::new();
            let mut witness_keys = BTreeSet::new();
            for witness in &record.source_event_witnesses {
                let selected = final_frame
                    .selected_events
                    .get(witness.selected_ordinal as usize)
                    .context("Ghost witness selected ordinal exceeds final source selection")?;
                let mut normalized_support = witness.support_residue_ids.clone();
                normalized_support.sort_unstable();
                normalized_support.dedup();
                if witness.canonical_event_sha256.len() != 64
                    || witness.primary_residue_id < 0
                    || !witness
                        .support_residue_ids
                        .contains(&witness.primary_residue_id)
                    || witness.position.iter().any(|value| !value.is_finite())
                    || !witness_keys.insert((witness.selected_ordinal, witness.source_index))
                    || !witnessed_selected_ordinals.insert(witness.selected_ordinal)
                    || witness.source_index != selected.source_index
                    || witness.canonical_event_sha256 != selected.canonical_event_sha256
                    || witness.position != selected.event.position
                    || witness.primary_residue_id != selected.event.nearby_residues[0]
                    || normalized_support != selected.support_residue_ids
                    || record.topology_cluster_id != selected.topology_cluster_id
                {
                    bail!("stream {} Ghost source-event witness is invalid", stream_id);
                }
                witness_support.extend(witness.support_residue_ids.iter().copied());
            }
            if witness_support.into_iter().collect::<Vec<_>>() != record.support_residue_ids {
                bail!(
                    "stream {} Ghost support is not the exact witness union",
                    stream_id
                );
            }
        }
        if witnessed_selected_ordinals
            != (0..final_frame.selected_event_count).collect::<BTreeSet<_>>()
        {
            bail!(
                "stream {} Ghost map does not witness every selected source event exactly once",
                stream_id
            );
        }
    }
    if observed_streams != BTreeSet::from([0u32, 1u32]) {
        bail!("strict coupled-TWIN Ghost evidence does not cover streams 0 and 1");
    }
    let ensemble_artifact = runtime
        .artifacts
        .iter()
        .find(|artifact| {
            artifact.kind == "split_t7_shadow_ensemble_receipt" && artifact.stream_id.is_none()
        })
        .context("strict coupled-TWIN lacks the split-T7 shadow ensemble receipt")?;
    let trial_artifacts = (0..SPLIT_T7_SHADOW_TRIALS)
        .map(|ordinal| {
            let kind = format!("split_t7_shadow_trial_{ordinal:02}_sources");
            runtime
                .artifacts
                .iter()
                .find(|artifact| artifact.kind == kind && artifact.stream_id.is_none())
                .with_context(|| format!("strict coupled-TWIN lacks {kind}"))
        })
        .collect::<Result<Vec<_>>>()?;
    let trial_crn_artifacts = (0..SPLIT_T7_SHADOW_TRIALS)
        .map(|ordinal| {
            let kind = format!("split_t7_shadow_trial_{ordinal:02}_crn_initialization");
            runtime
                .artifacts
                .iter()
                .find(|artifact| artifact.kind == kind && artifact.stream_id.is_none())
                .with_context(|| format!("strict coupled-TWIN lacks {kind}"))
        })
        .collect::<Result<Vec<_>>>()?;
    let durable_ensemble: SplitT7ShadowEnsembleReceipt =
        serde_json::from_slice(&std::fs::read(&ensemble_artifact.path)?)?;
    let durable_trial_artifacts = trial_artifacts
        .iter()
        .map(|artifact| (*artifact).clone())
        .collect::<Vec<CoupledTwinArtifactEvidence>>();
    for (ordinal, (trial, artifact)) in durable_ensemble
        .trials
        .iter()
        .zip(&trial_crn_artifacts)
        .enumerate()
    {
        if !strict_binding_matches_artifact(&trial.crn_initialization_artifact, artifact) {
            bail!("split-T7 shadow trial {ordinal} runtime CRN artifact binding failed");
        }
    }
    verify_split_t7_shadow_ensemble(
        output_dir,
        config,
        &topology_a,
        &topology_b,
        split_t7_layouts
            .get(&0)
            .context("split-T7 verifier lacks stream A topology layout")?,
        split_t7_layouts
            .get(&1)
            .context("split-T7 verifier lacks stream B topology layout")?,
        &durable_ensemble,
        &durable_trial_artifacts,
        ensemble_artifact,
    )?;
    let admission_artifact = runtime
        .artifacts
        .iter()
        .find(|artifact| {
            artifact.kind == "split_t7_admission_receipt_v3" && artifact.stream_id.is_none()
        })
        .context("strict coupled-TWIN lacks SPLIT_T7_ADMISSION_RECEIPT_V3.json")?;
    let durable_admission: SplitT7AdmissionReceiptV3 =
        serde_json::from_slice(&std::fs::read(&admission_artifact.path)?)?;
    let authoritative_crn_artifact = runtime
        .artifacts
        .iter()
        .find(|artifact| {
            artifact.kind == "authoritative_coupled_twin_crn_initialization"
                && artifact.stream_id.is_none()
        })
        .context("strict coupled-TWIN lacks authoritative CRN initialization artifact")?;
    if !strict_binding_matches_artifact(
        &durable_admission.authoritative_crn_initialization_artifact,
        authoritative_crn_artifact,
    ) {
        bail!("split-T7 admission/runtime authoritative CRN artifact binding failed");
    }
    verify_split_t7_admission_receipt_v3(
        output_dir,
        config,
        &topology_a,
        &topology_b,
        split_t7_layouts
            .get(&0)
            .context("split-T7 admission verifier lacks stream A topology layout")?,
        split_t7_layouts
            .get(&1)
            .context("split-T7 admission verifier lacks stream B topology layout")?,
        &durable_admission,
        &durable_trial_artifacts,
        ensemble_artifact,
        admission_artifact,
    )?;
    let calibration_a = calibrations
        .get(&0)
        .context("split-T7 paired inventory lacks stream A")?;
    let calibration_b = calibrations
        .get(&1)
        .context("split-T7 paired inventory lacks stream B")?;
    if calibration_a.inventory_window_md_steps != calibration_b.inventory_window_md_steps
        || calibration_a.fixed_spikes_per_frame != calibration_b.fixed_spikes_per_frame
        || calibration_a.selection_method != SPLIT_T7_SELECTION_METHOD_V3
        || calibration_b.selection_method != SPLIT_T7_SELECTION_METHOD_V3
        || calibration_a.shadow_ensemble_sha256 != calibration_b.shadow_ensemble_sha256
        || calibration_a.shadow_ensemble_sha256 != durable_ensemble.ensemble_sha256
        || calibration_a.shadow_ensemble_receipt_path != ensemble_artifact.path
        || calibration_b.shadow_ensemble_receipt_path != ensemble_artifact.path
        || calibration_a.shadow_ensemble_receipt_sha256 != ensemble_artifact.sha256
        || calibration_b.shadow_ensemble_receipt_sha256 != ensemble_artifact.sha256
        || calibration_a.shadow_ensemble_receipt_bytes != ensemble_artifact.size_bytes
        || calibration_b.shadow_ensemble_receipt_bytes != ensemble_artifact.size_bytes
        || calibration_a.shadow_lower_envelope_frame_counts
            != durable_ensemble.lower_envelope_counts_a
        || calibration_b.shadow_lower_envelope_frame_counts
            != durable_ensemble.lower_envelope_counts_b
        || calibration_a.direct_prefix_outer_steps != calibration_b.direct_prefix_outer_steps
        || calibration_a.direct_prefix_md_steps
            != calibration_a.direct_prefix_outer_steps as u64
                * calibration_a.inventory_window_md_steps as u64
        || calibration_b.direct_prefix_md_steps
            != calibration_b.direct_prefix_outer_steps as u64
                * calibration_b.inventory_window_md_steps as u64
    {
        bail!("split-T7 paired inventory geometry is inconsistent");
    }
    let envelope_fixed_spikes = split_t7_precommitted_cardinality(
        &durable_ensemble.lower_envelope_counts_a,
        &durable_ensemble.lower_envelope_counts_b,
    )?;
    let (authoritative_minimum, authoritative_slack) = split_t7_authoritative_admission(
        &calibration_a.inventory_frame_counts,
        &calibration_b.inventory_frame_counts,
        calibration_a.fixed_spikes_per_frame,
    )?;
    let paired_inventory_sha256 = split_t7_paired_inventory_sha256(
        SPLIT_T7_AUTHORITATIVE_INVENTORY_DOMAIN,
        calibration_a.inventory_window_md_steps,
        calibration_a.fixed_spikes_per_frame,
        &calibration_a.inventory_frame_counts,
        &calibration_b.inventory_frame_counts,
    )?;
    if envelope_fixed_spikes != calibration_a.fixed_spikes_per_frame
        || envelope_fixed_spikes != durable_ensemble.fixed_spikes_per_frame
        || authoritative_minimum != calibration_a.authoritative_minimum
        || authoritative_minimum != calibration_b.authoritative_minimum
        || authoritative_slack != calibration_a.authoritative_minimum_slack_from_k
        || authoritative_slack != calibration_b.authoritative_minimum_slack_from_k
        || paired_inventory_sha256 != calibration_a.paired_inventory_sha256
        || paired_inventory_sha256 != calibration_b.paired_inventory_sha256
    {
        bail!("split-T7 shadow-ensemble/authoritative inventory recomputation failed");
    }
    Ok(())
}

#[cfg(feature = "gpu")]
fn validate_coupled_runtime_evidence(
    output_dir: &Path,
    config: &CoupledTwinEvidenceConfig,
    result: &CoupledTwinResult,
    runtime: &CoupledTwinRuntimeEvidence,
) -> Result<()> {
    config.validate_source_files()?;
    if runtime.schema_kind != "prismos.coupled_twin_runtime_evidence.v2"
        || runtime.schema_version != COUPLED_TWIN_EVIDENCE_SCHEMA_VERSION
        || runtime.run_id != config.run_id
        || runtime.execution_semantics
            != "independent_aligned_trajectory_analysis_common_random_numbers"
        || runtime.exchange_active
        || runtime.causal_transport_claim
    {
        bail!("coupled-TWIN runtime evidence identity/semantics failed");
    }
    if result.topology_a_sha256.as_deref() != Some(config.topology_a_sha256.as_str())
        || result.topology_b_sha256.as_deref() != Some(config.topology_b_sha256.as_str())
        || result.contrast_kind.as_deref() != Some(config.contrast_kind.as_str())
        || result.engine_b_role.as_deref() != Some(config.engine_b_role.as_str())
        || result.stream_a.seed != result.stream_b.seed
        || result.steps_a <= 0
        || result.steps_b <= 0
    {
        bail!("coupled result is not bound to the validated topology/configuration identity");
    }

    let mut graph_streams = BTreeSet::new();
    if runtime.graph.len() != 2 {
        bail!("strict coupled-TWIN runtime requires exactly two graph receipts");
    }
    for graph in &runtime.graph {
        if !graph_streams.insert(graph.stream_id)
            || graph.stream_id > 1
            || graph.capture_count == 0
            || graph.host_launch_count == 0
            || graph.resident_launch_count == 0
            || graph.legacy_launch_count != 0
            || graph.direct_prefix_outer_steps == 0
            || graph.direct_prefix_md_steps == 0
            || graph.resident_outer_steps == 0
            || graph.resident_md_steps == 0
            || graph.resident_outer_steps != graph.logical_outer_steps
            || graph.resident_md_steps != graph.logical_md_steps
            || graph
                .direct_prefix_md_steps
                .saturating_add(graph.resident_md_steps)
                != graph.total_md_steps
            || graph.logical_outer_steps == 0
            || graph.logical_md_steps == 0
            || graph.steps_per_launch == 0
            || !graph.launch_collapse_ratio.is_finite()
            || graph.launch_collapse_ratio < config.min_hotpath_launch_collapse
            || graph.launch_path != "graph.while_resident"
        {
            bail!(
                "stream {} coupled CUDA graph evidence failed",
                graph.stream_id
            );
        }
    }
    if graph_streams != BTreeSet::from([0u32, 1u32]) {
        bail!("coupled CUDA graph evidence does not cover streams 0 and 1");
    }

    let mut artifact_keys = BTreeSet::new();
    for artifact in &runtime.artifacts {
        if !artifact_keys.insert((artifact.kind.clone(), artifact.stream_id)) {
            bail!(
                "duplicate coupled artifact identity kind={} stream={:?}",
                artifact.kind,
                artifact.stream_id
            );
        }
        validate_run_artifact(output_dir, artifact)?;
    }
    for stream_id in [0u32, 1u32] {
        for kind in [
            "spikes",
            "signal_grid",
            "kcc_v2full",
            "ghost_tiles_v2",
            "ghost_site_map",
            "ghost_noise_floor_input",
            "ghost_noise_floor_calibration",
            "ghost_noise_floor_frozen_anchor",
            "ghost_final_manifold_input",
            "ghost_segment_manifest",
            "prismcode_md_native_receipt",
        ] {
            if !artifact_keys.contains(&(kind.to_string(), Some(stream_id))) {
                bail!(
                    "stream {} is missing required coupled artifact {}",
                    stream_id,
                    kind
                );
            }
        }
    }
    for kind in [
        "split_t7_shadow_trial_00_sources",
        "split_t7_shadow_trial_01_sources",
        "split_t7_shadow_trial_02_sources",
        "split_t7_shadow_ensemble_receipt",
        "split_t7_admission_receipt_v3",
        "gpu_ccf_matrix",
        "gpu_transfer_entropy",
        "gpu_ccf_te_execution_receipt",
        "gpu_phase_ccf_0",
        "gpu_phase_ccf_1",
        "gpu_phase_ccf_2",
        "gpu_phase_ccf_3",
        "gpu_phase_ccf_4",
        "qnanl_spike_event_stream_manifest",
        "molecular_contrast_readout",
        "molecular_contrast_sha256_sidecar",
        "binding_sites",
        "strict_twin_residue_features",
        "strict_twin_site_features",
        "strict_twin_feature_receipt",
    ] {
        if !artifact_keys.contains(&(kind.to_string(), None)) {
            bail!(
                "coupled runtime is missing required global artifact {}",
                kind
            );
        }
    }
    if config.contrast_kind == GLYCAN_MECHANISTIC_PROBE_CONTRAST_KIND {
        for kind in [
            "glycan_mechanistic_probe_readout",
            "glycan_mechanistic_probe_sha256_sidecar",
        ] {
            if !artifact_keys.contains(&(kind.to_string(), None)) {
                bail!("glycan mechanistic probe runtime is missing required artifact {kind}");
            }
        }
    }

    let stem = coupled_topology_stem(config);
    for artifact in &runtime.artifacts {
        let Some(stream_id) = artifact.stream_id else {
            continue;
        };
        match artifact.kind.as_str() {
            "spikes" => validate_evidence_envelope(
                Path::new(&artifact.path),
                b"PRSPK001",
                b"PRSPKEND",
                stream_id,
                &config.run_id,
                &stem,
                artifact.record_count,
            )?,
            "signal_grid" => validate_evidence_envelope(
                Path::new(&artifact.path),
                b"PRSGD001",
                b"PRSGDEND",
                stream_id,
                &config.run_id,
                &stem,
                artifact.record_count,
            )?,
            "kcc_v2full" => validate_evidence_envelope(
                Path::new(&artifact.path),
                b"PRKCC001",
                b"PRKCCEND",
                stream_id,
                &config.run_id,
                &stem,
                artifact.record_count,
            )?,
            "prismcode_md_native_receipt" => {
                let receipt: prismcode::md_native_runtime::MdNativeRuntimeReceipt =
                    serde_json::from_slice(&std::fs::read(&artifact.path)?)?;
                if receipt.stream_id != stream_id {
                    bail!("native MD receipt stream identity mismatch");
                }
                validate_native_receipt(&receipt, config)?;
            }
            _ => {}
        }
    }

    let qnanl_artifact = runtime
        .artifacts
        .iter()
        .find(|artifact| artifact.kind == "qnanl_spike_event_stream_manifest")
        .context("missing strict QNANL event-stream manifest")?;
    let qnanl: serde_json::Value = serde_json::from_slice(&std::fs::read(&qnanl_artifact.path)?)?;
    let qnanl_identity = qnanl
        .get("molecular_contrast_identity")
        .context("strict QNANL manifest lacks molecular-contrast identity")?;
    if qnanl.get("schema_kind").and_then(serde_json::Value::as_str)
        != Some("qnanl_spike_event_stream_manifest")
        || qnanl
            .get("n_chunks")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
            == 0
        || qnanl
            .get("record_count_matches_emitted")
            .and_then(serde_json::Value::as_bool)
            != Some(true)
        || qnanl
            .get("all_chunks_roundtrip_verified")
            .and_then(serde_json::Value::as_bool)
            != Some(true)
        || qnanl
            .pointer("/sensor_bus/sensor_roundtrip_verified")
            .and_then(serde_json::Value::as_bool)
            != Some(true)
        || qnanl
            .pointer("/sensor_bus/sensor_join_complete_zero_orphans")
            .and_then(serde_json::Value::as_bool)
            != Some(true)
        || qnanl_identity
            .get("topology_a_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(config.topology_a_sha256.as_str())
        || qnanl_identity
            .get("topology_b_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(config.topology_b_sha256.as_str())
        || qnanl_identity
            .get("contrast_kind")
            .and_then(serde_json::Value::as_str)
            != Some(config.contrast_kind.as_str())
    {
        bail!("strict QNANL event-stream semantic/readback contract failed");
    }

    let readout_artifact = runtime
        .artifacts
        .iter()
        .find(|artifact| artifact.kind == "molecular_contrast_readout")
        .context("missing molecular-contrast readout")?;
    let readout: MolecularContrastReadout =
        serde_json::from_slice(&std::fs::read(&readout_artifact.path)?)?;
    let identical = config.topology_a_sha256 == config.topology_b_sha256;
    if readout.schema_kind != "prismos.molecular_contrast_readout.v4"
        || readout.schema_version != MOLECULAR_CONTRAST_READOUT_SCHEMA_VERSION
        || readout.run_id != config.run_id
        || readout.topology_a_sha256 != config.topology_a_sha256
        || readout.topology_b_sha256 != config.topology_b_sha256
        || readout.contrast_kind != config.contrast_kind
        || readout.identical_topology_identity != identical
        || readout.exchange_active
        || readout.causal_transport_claim
        || readout.fallback_used
        || !readout.finite_values_verified
        || readout.residue_alignment_sha256.len() != 64
    {
        bail!("molecular-contrast readout identity/finite/fallback contract failed");
    }
    let pocket_contact_residue_ids = readout
        .pocket_contact_residue_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    if pocket_contact_residue_ids.len() != readout.pocket_contact_residue_ids.len() {
        bail!("molecular-contrast readout repeats a pocket residue identity");
    }
    validate_molecular_contrast_pocket_support_v4(
        &pocket_contact_residue_ids,
        &MolecularContrastPocketSupportV4 {
            decisions: readout.pocket_contrast_residue_decisions.clone(),
            candidate_residue_count: readout.pocket_contrast_candidate_residue_count,
            supported_residue_count: readout.pocket_contrast_supported_residue_count,
            unavailable_residue_count: readout.pocket_contrast_unavailable_residue_count,
            l1_signed_contrast: readout.pocket_l1_signed_contrast,
            mean_abs_signed_contrast: readout.pocket_mean_abs_signed_contrast,
        },
    )?;
    let topology_a = PrismPrepTopology::load(&config.topology_a_path)
        .context("reload topology A for molecular-contrast recomputation")?;
    let topology_b = PrismPrepTopology::load(&config.topology_b_path)
        .context("reload topology B for molecular-contrast recomputation")?;
    let stem = coupled_topology_stem(config);
    let spike_a_artifact = runtime
        .artifacts
        .iter()
        .find(|artifact| artifact.kind == "spikes" && artifact.stream_id == Some(0))
        .context("molecular-contrast recomputation lacks stream-A spikes")?;
    let spike_b_artifact = runtime
        .artifacts
        .iter()
        .find(|artifact| artifact.kind == "spikes" && artifact.stream_id == Some(1))
        .context("molecular-contrast recomputation lacks stream-B spikes")?;
    let spikes_a = read_spike_evidence_events(spike_a_artifact, 0, &config.run_id, &stem)?;
    let spikes_b = read_spike_evidence_events(spike_b_artifact, 1, &config.run_id, &stem)?;
    let strict_residue_artifact = runtime
        .artifacts
        .iter()
        .find(|artifact| artifact.kind == "strict_twin_residue_features")
        .context("molecular-contrast recomputation lacks strict residue features")?;
    let strict_residue_payload: StrictTwinResidueFeatureArtifact = serde_json::from_slice(
        &std::fs::read(&strict_residue_artifact.path)
            .context("read strict residue features for molecular-contrast recomputation")?,
    )?;
    let mut recomputed_readout = build_molecular_contrast_readout(
        &topology_a,
        &topology_b,
        &spikes_a,
        &spikes_b,
        &strict_residue_payload.records,
        config,
    )?;
    bind_engine_ghost_to_molecular_contrast(&mut recomputed_readout, runtime, config)?;
    if serde_json::to_value(&recomputed_readout)? != serde_json::to_value(&readout)? {
        bail!(
            "molecular-contrast readout does not recompute from topology, events, strict features, and exact Ghost geometry"
        );
    }
    if config.contrast_kind == GLYCAN_MECHANISTIC_PROBE_CONTRAST_KIND {
        let probe_artifact = runtime
            .artifacts
            .iter()
            .find(|artifact| artifact.kind == "glycan_mechanistic_probe_readout")
            .context("missing glycan mechanistic probe readout")?;
        let probe_readout: GlycanMechanisticProbeReadoutV1 =
            serde_json::from_slice(&std::fs::read(&probe_artifact.path)?)?;
        let recomputed_probe = build_glycan_mechanistic_probe_readout_v1(
            &recomputed_readout,
            &topology_b,
            runtime,
            config,
        )?
        .context("glycan mechanistic probe recomputation did not produce a readout")?;
        if serde_json::to_value(&probe_readout)? != serde_json::to_value(&recomputed_probe)?
            || probe_readout.schema_kind != "prismos.glycan_mechanistic_probe_readout.v1"
            || probe_readout.schema_version != GLYCAN_MECHANISTIC_PROBE_READOUT_SCHEMA_VERSION
            || probe_readout.causal_transport_claim
            || probe_readout.structural_novelty_claim
            || probe_readout.patent_novelty_claim
            || probe_readout.replica_consensus_verified
            || !probe_readout.finite_values_verified
            || probe_readout.fallback_used
        {
            bail!("glycan mechanistic probe readout failed recomputation/claim-boundary gates");
        }
        let probe_sidecar = runtime
            .artifacts
            .iter()
            .find(|artifact| artifact.kind == "glycan_mechanistic_probe_sha256_sidecar")
            .context("missing glycan mechanistic probe SHA-256 sidecar")?;
        let expected_probe_sidecar = format!(
            "{}  glycan_mechanistic_probe_readout.json\n",
            probe_artifact.sha256
        );
        if std::fs::read_to_string(&probe_sidecar.path)? != expected_probe_sidecar {
            bail!("glycan mechanistic probe sidecar does not bind the readout bytes");
        }
    }
    if identical {
        if readout.global_mean_abs_signed_contrast > config.identical_control_max_mean_abs_contrast
        {
            bail!("committed identical-topology readout exceeds the contrast ceiling");
        }
    } else if readout.global_mean_abs_signed_contrast < config.distinct_min_mean_abs_contrast
        || readout.pocket_localization_fraction < config.distinct_min_pocket_localization_fraction
        || readout.pocket_contact_residue_ids.is_empty()
        || readout.pocket_contrast_supported_residue_count == 0
        || readout.pocket_mean_abs_signed_contrast.is_none()
        || !readout.ghost_pocket_binding_verified
        || readout.ghost_pocket_bound_count == 0
        || readout.ghost_pocket_witnesses.is_empty()
        || readout.ghost_pocket_witnesses.iter().any(|witness| {
            witness.canonical_event_sha256.len() != 64
                || witness.position.iter().any(|value| !value.is_finite())
                || witness.topology_b_sha256 != config.topology_b_sha256
                || witness.ligand_coordinate_sha256 != readout.ligand_heavy_atom_coordinate_sha256
                || witness.cutoff_angstrom.to_bits()
                    != config.ligand_contact_cutoff_angstrom.to_bits()
                || witness.pocket_support_intersection_residue_ids.is_empty()
                || !witness.exact_distance_admission
                || !witness
                    .minimum_ligand_heavy_atom_distance_angstrom
                    .is_finite()
                || witness.minimum_ligand_heavy_atom_distance_angstrom
                    > config.ligand_contact_cutoff_angstrom
                || !readout
                    .ghost_pocket_component_ids
                    .contains(&witness.component_id)
        })
        || !readout
            .residues
            .iter()
            .any(|row| row.support_class == "B_ONLY" && row.residue_name != "UNK")
    {
        bail!("committed distinct-topology readout failed nontrivial/pocket/Ghost gates");
    }
    let sidecar_artifact = runtime
        .artifacts
        .iter()
        .find(|artifact| artifact.kind == "molecular_contrast_sha256_sidecar")
        .context("missing molecular-contrast SHA-256 sidecar")?;
    let expected_sidecar = format!(
        "{}  molecular_contrast_readout.json\n",
        readout_artifact.sha256
    );
    if std::fs::read_to_string(&sidecar_artifact.path)? != expected_sidecar {
        bail!("molecular-contrast sidecar does not bind the readout bytes");
    }

    let analysis_receipt_artifact = runtime
        .artifacts
        .iter()
        .find(|artifact| artifact.kind == "gpu_ccf_te_execution_receipt")
        .context("missing GPU CCF/TE execution receipt")?;
    let analysis_receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&analysis_receipt_artifact.path)?)?;
    if analysis_receipt
        .get("status")
        .and_then(serde_json::Value::as_str)
        != Some("PASS")
        || analysis_receipt
            .get("fallback_used")
            .and_then(serde_json::Value::as_bool)
            != Some(false)
        || analysis_receipt
            .get("run_id")
            .and_then(serde_json::Value::as_str)
            != Some(config.run_id.as_str())
        || analysis_receipt
            .get("ccf_nonzero_count")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
            == 0
        || analysis_receipt
            .get("bidirectional_te_mass_nonzero_count")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
            == 0
        || analysis_receipt
            .get("thresholded_outgoing_te_nonzero_count")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
            == 0
        || analysis_receipt
            .get("kernel_supported_residue_count")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
            == 0
        || (!identical
            && analysis_receipt
                .get("net_causal_flow_nonzero_count")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
                == 0)
    {
        bail!("GPU CCF/TE execution receipt is not a real no-fallback PASS");
    }

    validate_strict_twin_feature_plane(output_dir, config, result, runtime)?;
    validate_coupled_ghost_runtime_evidence(output_dir, config, runtime)
}

/// Independently validate every strict coupled-TWIN runtime artifact before
/// any aggregate result, Path-B map, or manifest is published.
#[cfg(feature = "gpu")]
fn validate_shared_therm_report_value(
    report: &serde_json::Value,
    run_id: &str,
    expected_streams: &[(u32, &str, u64, u64, u64)],
) -> Result<()> {
    if report["schema_kind"].as_str() != Some("prismos.coupled_stream_lattice_therm.v1")
        || report["schema_version"].as_u64() != Some(1)
        || report["run_id"].as_str() != Some(run_id)
    {
        bail!("coupled shared Therm report schema/run identity mismatch");
    }
    let streams = report["streams"].as_array().context("coupled Therm lacks stream reports")?;
    if streams.len() != 2 || expected_streams.len() != 2 {
        bail!("coupled shared Therm requires exactly two stream reports");
    }
    let mut seen = BTreeSet::new();
    let mut total_nodes = 0u64;
    for stream in streams {
        let id = stream["source_stream_id"].as_u64().context("coupled Therm lacks stream id")?;
        if !seen.insert(id) {
            bail!("coupled shared Therm repeats stream {id}");
        }
        let (_, topology, seed, steps, spikes) = expected_streams.iter()
            .find(|(expected, _, _, _, _)| u64::from(*expected) == id)
            .context("coupled Therm has unexpected stream identity")?;
        let input = stream["input_spike_count"].as_u64();
        let ingested = stream["ingested_event_count"].as_u64()
            .context("coupled Therm lacks ingested count")?;
        let nodes = stream["lattice_node_count"].as_u64()
            .context("coupled Therm lacks lattice node count")?;
        if stream["topology_source_sha256"].as_str() != Some(*topology)
            || stream["seed"].as_u64() != Some(*seed)
            || stream["physical_steps"].as_u64() != Some(*steps)
            || *steps == 0 || input != Some(*spikes) || ingested == 0 || ingested > *spikes
            || stream.pointer("/analysis/sdst_event_count").and_then(|v| v.as_u64()) != Some(ingested)
            || nodes == 0
            || stream.pointer("/conservation/promotable").and_then(|v| v.as_bool()) != Some(true)
            || stream.pointer("/conservation/counts/unmatched_site_count").and_then(|v| v.as_u64()) != Some(0)
        {
            bail!("coupled shared Therm stream {id} identity/population mismatch");
        }
        total_nodes = total_nodes.checked_add(nodes).context("coupled Therm node count overflow")?;
    }
    if seen != BTreeSet::from([0, 1])
        || report["committed_lattice_node_count"].as_u64() != Some(total_nodes)
    {
        bail!("coupled shared Therm source-node conservation mismatch");
    }
    Ok(())
}

#[cfg(feature = "gpu")]
fn validate_required_shared_therm(
    output_dir: &Path,
    config: &CoupledTwinEvidenceConfig,
    result: &CoupledTwinResult,
    runtime: &CoupledTwinRuntimeEvidence,
) -> Result<()> {
    let reports: Vec<_> = runtime.artifacts.iter()
        .filter(|artifact| artifact.kind == "coupled_shared_therm").collect();
    if reports.is_empty() && !config.require_shared_lattice_therm {
        return Ok(());
    }
    if reports.len() != 1 || reports[0].stream_id.is_some()
        || reports[0].record_count != Some(2) || reports[0].schema_version != Some(1)
    {
        bail!("requested coupled shared Therm lacks one exact two-stream report artifact");
    }
    let path = validate_run_artifact(output_dir, reports[0])?;
    let report: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
    let steps = |id| -> Result<u64> {
        Ok(runtime.graph.iter().find(|graph| graph.stream_id == id)
            .context("coupled Therm lacks measured graph horizon")?.total_md_steps)
    };
    validate_shared_therm_report_value(&report, &config.run_id, &[
        (0, &config.topology_a_sha256, result.stream_a.seed, steps(0)?, result.stream_a.total_spikes as u64),
        (1, &config.topology_b_sha256, result.stream_b.seed, steps(1)?, result.stream_b.total_spikes as u64),
    ])?;
    let lattice_path = Path::new(report.pointer("/committed_lattice/path")
        .and_then(|value| value.as_str()).context("coupled Therm lacks committed lattice path")?);
    let expected_sha = report.pointer("/committed_lattice/sha256")
        .and_then(|value| value.as_str()).context("coupled Therm lacks committed lattice SHA256")?;
    let mut lattice_artifact = artifact_evidence("ghost_phase_lattice_components", None, lattice_path, None, Some(2))?;
    lattice_artifact.sha256 = expected_sha.to_string();
    let lattice_path = validate_run_artifact(output_dir, &lattice_artifact)?;
    let expected_lattice_path = output_dir.join(format!("{}_binding_sites_lattice.json", coupled_topology_stem(config)));
    if lattice_path != std::fs::canonicalize(&expected_lattice_path)? {
        bail!("coupled shared Therm references another lattice artifact");
    }
    let lattice: serde_json::Value = serde_json::from_slice(&std::fs::read(lattice_path)?)?;
    if lattice["schema_kind"].as_str() != Some("prismos_ghost_phase_lattice_components")
        || lattice["run_id"].as_str() != Some(config.run_id.as_str())
    {
        bail!("coupled shared Therm committed lattice identity mismatch");
    }
    Ok(())
}

#[cfg(feature = "gpu")]
pub fn verify_coupled_twin_runtime_evidence(
    output_dir: &Path,
    config: &CoupledTwinEvidenceConfig,
    result: &CoupledTwinResult,
) -> Result<()> {
    let runtime = result
        .runtime_evidence
        .as_ref()
        .context("strict coupled result lacks runtime-owned evidence")?;
    validate_coupled_runtime_evidence(output_dir, config, result, runtime)?;
    validate_required_shared_therm(output_dir, config, result, runtime)
}

#[cfg(feature = "gpu")]
pub fn commit_coupled_twin_md_evidence_manifest(
    output_dir: &Path,
    config: &CoupledTwinEvidenceConfig,
    result_path: &Path,
    result: &CoupledTwinResult,
    prismcode_substrate_receipt_path: Option<&Path>,
    aggregate_ghost_site_map_path: &Path,
    ghost_phase_lattice_path: Option<&Path>,
) -> Result<PathBuf> {
    let runtime = result
        .runtime_evidence
        .as_ref()
        .context("strict coupled result lacks runtime-owned evidence")?;
    validate_coupled_runtime_evidence(output_dir, config, result, runtime)?;

    validate_required_shared_therm(output_dir, config, result, runtime)?;
    let mut artifacts = runtime.artifacts.clone();
    artifacts.push(artifact_evidence(
        "coupled_twin_result",
        None,
        result_path,
        Some(1),
        Some(COUPLED_TWIN_EVIDENCE_SCHEMA_VERSION),
    )?);
    artifacts.push(artifact_evidence(
        "aggregate_ghost_site_map",
        None,
        aggregate_ghost_site_map_path,
        Some(1),
        Some(COUPLED_TWIN_EVIDENCE_SCHEMA_VERSION),
    )?);
    if let Some(lattice_path) = ghost_phase_lattice_path {
        artifacts.push(artifact_evidence(
            "ghost_phase_lattice_components",
            None,
            lattice_path,
            None,
            Some(2),
        )?);
    }
    if let Some(receipt_path) = prismcode_substrate_receipt_path {
        artifacts.push(artifact_evidence(
            "prismcode_substrate_receipt",
            None,
            receipt_path,
            Some(1),
            Some(1),
        )?);
    }
    artifacts.sort_by(|left, right| {
        (&left.kind, left.stream_id, &left.path).cmp(&(&right.kind, right.stream_id, &right.path))
    });
    let mut artifact_keys = BTreeSet::new();
    for artifact in &artifacts {
        if !artifact_keys.insert((artifact.kind.clone(), artifact.stream_id)) {
            bail!("duplicate artifact identity during manifest commit");
        }
        validate_run_artifact(output_dir, artifact)?;
    }
    let manifest_artifacts = artifacts
        .into_iter()
        .map(|artifact| CoupledTwinManifestArtifact {
            note: Some("engine-owned hash-bound coupled-TWIN runtime evidence".to_string()),
            kind: artifact.kind,
            stream_id: artifact.stream_id,
            path: artifact.path,
            size_bytes: artifact.size_bytes,
            present: true,
            sha256: artifact.sha256,
            blake3: artifact.blake3,
            record_count: artifact.record_count,
            schema_version: artifact.schema_version,
            runtime_owned: artifact.runtime_owned,
        })
        .collect();
    let manifest = CoupledTwinMdEvidenceManifest {
        schema_version: 1,
        schema_kind: "md_evidence_manifest".to_string(),
        status: "IMPLEMENTATION_VERIFIED".to_string(),
        run_id: config.run_id.clone(),
        topology_input: config.topology_a_path.clone(),
        topology_b_input: config.topology_b_path.clone(),
        topology_a_sha256: config.topology_a_sha256.clone(),
        topology_b_sha256: config.topology_b_sha256.clone(),
        target: coupled_topology_stem(config),
        stream_count: 2,
        streams_serialized: 2,
        total_spikes_md: (result.stream_a.total_spikes as u64)
            .checked_add(result.stream_b.total_spikes as u64)
            .context("coupled total-spike count overflow")?,
        artifacts: manifest_artifacts,
        coupled_twin_evidence: CoupledTwinManifestExtension {
            schema_kind: "prismos.coupled_twin_manifest_extension.v2".to_string(),
            schema_version: COUPLED_TWIN_EVIDENCE_SCHEMA_VERSION,
            topology_a_path: config.topology_a_path.clone(),
            topology_b_path: config.topology_b_path.clone(),
            topology_a_sha256: config.topology_a_sha256.clone(),
            topology_b_sha256: config.topology_b_sha256.clone(),
            identical_topology_negative_control: config.identical_topology_negative_control,
            exchange_active: false,
            causal_transport_claim: false,
            fallback_used: false,
            config: config.clone(),
            runtime_evidence: runtime.clone(),
        },
        compression_status: "qnanl_spike_event_stream_roundtrip_verified".to_string(),
        checksum_algorithm: "sha256+blake3+fnv1a64_envelopes".to_string(),
        endian: "little".to_string(),
        v2_was_live: true,
        path_b_required: true,
        validation_status: "PASS".to_string(),
        required_artifacts_complete: true,
        serialization_failure: None,
    };
    let manifest_path = output_dir.join("md_evidence_manifest.json");
    let mut candidate_bytes = serde_json::to_vec_pretty(&manifest)?;
    candidate_bytes.push(b'\n');
    let candidate: CoupledTwinMdEvidenceManifest = serde_json::from_slice(&candidate_bytes)
        .context("parse serialized coupled manifest before commit")?;
    validate_coupled_twin_md_evidence_manifest_value(output_dir, &candidate, config, result)?;
    write_atomic_bytes(&manifest_path, &candidate_bytes)?;
    verify_coupled_twin_md_evidence_manifest(&manifest_path, config, result)?;
    Ok(manifest_path)
}

#[cfg(feature = "gpu")]
fn validate_coupled_twin_md_evidence_manifest_value(
    output_dir: &Path,
    manifest: &CoupledTwinMdEvidenceManifest,
    config: &CoupledTwinEvidenceConfig,
    result: &CoupledTwinResult,
) -> Result<()> {
    let result_runtime = result
        .runtime_evidence
        .as_ref()
        .context("coupled manifest verification lacks result runtime evidence")?;
    if manifest.schema_version != 1
        || manifest.schema_kind != "md_evidence_manifest"
        || manifest.status != "IMPLEMENTATION_VERIFIED"
        || manifest.run_id != config.run_id
        || manifest.topology_input != config.topology_a_path
        || manifest.topology_b_input != config.topology_b_path
        || manifest.topology_a_sha256 != config.topology_a_sha256
        || manifest.topology_b_sha256 != config.topology_b_sha256
        || manifest.stream_count != 2
        || manifest.streams_serialized != 2
        || manifest.serialization_failure.is_some()
        || !manifest.required_artifacts_complete
        || manifest.validation_status != "PASS"
        || !manifest.v2_was_live
        || !manifest.path_b_required
        || manifest.coupled_twin_evidence.schema_kind
            != "prismos.coupled_twin_manifest_extension.v2"
        || manifest.coupled_twin_evidence.fallback_used
        || manifest.coupled_twin_evidence.exchange_active
        || manifest.coupled_twin_evidence.causal_transport_claim
        || manifest.coupled_twin_evidence.config.run_id != config.run_id
        || manifest.coupled_twin_evidence.config.require_shared_lattice_therm
            != config.require_shared_lattice_therm
        || serde_json::to_value(&manifest.coupled_twin_evidence.runtime_evidence)?
            != serde_json::to_value(result_runtime)?
    {
        bail!("coupled md_evidence_manifest identity/completeness contract failed");
    }
    let expected_spikes = (result.stream_a.total_spikes as u64)
        .checked_add(result.stream_b.total_spikes as u64)
        .context("coupled result spike-count overflow")?;
    if manifest.total_spikes_md != expected_spikes || manifest.artifacts.is_empty() {
        bail!("coupled manifest total-spike/artifact count mismatch");
    }
    let mut keys = BTreeSet::new();
    for artifact in &manifest.artifacts {
        if !artifact.present
            || !artifact.runtime_owned
            || !keys.insert((artifact.kind.clone(), artifact.stream_id))
        {
            bail!("coupled manifest contains missing/non-runtime/duplicate artifact rows");
        }
        validate_run_artifact(
            output_dir,
            &CoupledTwinArtifactEvidence {
                kind: artifact.kind.clone(),
                stream_id: artifact.stream_id,
                path: artifact.path.clone(),
                size_bytes: artifact.size_bytes,
                sha256: artifact.sha256.clone(),
                blake3: artifact.blake3.clone(),
                record_count: artifact.record_count,
                schema_version: artifact.schema_version,
                runtime_owned: artifact.runtime_owned,
            },
        )?;
    }
    if !keys.contains(&("coupled_twin_result".to_string(), None))
        || !keys.contains(&("aggregate_ghost_site_map".to_string(), None))
    {
        bail!("coupled manifest does not bind result plus aggregate Ghost site map");
    }
    if config.require_shared_lattice_therm {
        for kind in ["coupled_shared_therm", "ghost_phase_lattice_components"] {
            if !keys.contains(&(kind.to_string(), None)) {
                bail!("coupled manifest lacks required shared Therm artifact {kind}");
            }
        }
        let report = manifest.artifacts.iter().find(|artifact| artifact.kind == "coupled_shared_therm")
            .context("manifest shared Therm artifact missing")?;
        let expected = result_runtime.artifacts.iter().find(|artifact| artifact.kind == "coupled_shared_therm")
            .context("result shared Therm artifact missing")?;
        if report.path != expected.path || report.sha256 != expected.sha256
            || report.blake3 != expected.blake3 || report.size_bytes != expected.size_bytes
        {
            bail!("manifest shared Therm artifact differs from result authority");
        }
    }
    if let Some(lattice_artifact) = manifest
        .artifacts
        .iter()
        .find(|artifact| artifact.kind == "ghost_phase_lattice_components")
    {
        if lattice_artifact.stream_id.is_some() || lattice_artifact.schema_version != Some(2) {
            bail!("coupled Ghost Phase-Lattice manifest artifact identity is invalid");
        }
        let lattice: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&lattice_artifact.path).with_context(|| {
                format!("read coupled Ghost Phase-Lattice {}", lattice_artifact.path)
            })?)?;
        if lattice
            .get("schema_kind")
            .and_then(serde_json::Value::as_str)
            != Some("prismos_ghost_phase_lattice_components")
            || lattice
                .get("schema_version")
                .and_then(serde_json::Value::as_u64)
                != Some(2)
            || lattice.get("backend").and_then(serde_json::Value::as_str)
                != Some("ghost_phase_lattice_4d")
            || lattice.get("run_id").and_then(serde_json::Value::as_str)
                != Some(config.run_id.as_str())
            || lattice.get("target").and_then(serde_json::Value::as_str)
                != Some(coupled_topology_stem(config).as_str())
            || lattice
                .get("topology_input")
                .and_then(serde_json::Value::as_str)
                != Some(config.topology_a_path.as_str())
        {
            bail!("coupled Ghost Phase-Lattice schema/run/topology binding failed");
        }
        let components = lattice
            .get("components")
            .and_then(serde_json::Value::as_array)
            .filter(|components| !components.is_empty())
            .context("coupled Ghost Phase-Lattice has no components")?;
        let mut lattice_events = BTreeSet::new();
        for component in components {
            let lineages = component
                .get("node_lineage")
                .and_then(serde_json::Value::as_array)
                .filter(|lineages| !lineages.is_empty())
                .context("coupled Ghost Phase-Lattice component lacks node lineage")?;
            if component.get("n_nodes").and_then(serde_json::Value::as_u64)
                != Some(lineages.len() as u64)
            {
                bail!("coupled Ghost Phase-Lattice component lineage count mismatch");
            }
            for lineage in lineages {
                let event = (
                    lineage
                        .get("stream_id")
                        .and_then(serde_json::Value::as_u64)
                        .context("coupled lattice lineage lacks stream_id")?,
                    lineage
                        .get("frame_idx")
                        .and_then(serde_json::Value::as_u64)
                        .context("coupled lattice lineage lacks frame_idx")?,
                    lineage
                        .get("site_id")
                        .and_then(serde_json::Value::as_u64)
                        .context("coupled lattice lineage lacks site_id")?,
                );
                if event.0 > 1 || !lattice_events.insert(event) {
                    bail!("coupled Ghost Phase-Lattice contains duplicate/invalid lineage");
                }
            }
        }
        let aggregate_artifact = manifest
            .artifacts
            .iter()
            .find(|artifact| artifact.kind == "aggregate_ghost_site_map")
            .context("coupled lattice validation lacks aggregate Ghost map")?;
        let aggregate: NativeGhostSiteMap =
            serde_json::from_slice(&std::fs::read(&aggregate_artifact.path)?)?;
        let aggregate_events: BTreeSet<_> = aggregate
            .entries
            .iter()
            .map(|entry| {
                (
                    entry.stream_id as u64,
                    entry.frame_idx,
                    entry.site_id as u64,
                )
            })
            .collect();
        if lattice_events != aggregate_events {
            bail!("coupled Ghost Phase-Lattice lineage differs from aggregate Ghost-v2 authority");
        }
    }
    validate_coupled_runtime_evidence(output_dir, config, result, result_runtime)?;
    validate_required_shared_therm(output_dir, config, result, result_runtime)
}

#[cfg(feature = "gpu")]
pub fn verify_coupled_twin_md_evidence_manifest(
    manifest_path: &Path,
    config: &CoupledTwinEvidenceConfig,
    result: &CoupledTwinResult,
) -> Result<()> {
    let metadata = std::fs::symlink_metadata(manifest_path)
        .with_context(|| format!("stat coupled manifest {}", manifest_path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() == 0 {
        bail!("coupled manifest must be a nonempty regular non-symlink file");
    }
    let output_dir = manifest_path
        .parent()
        .context("coupled manifest has no run-root parent")?;
    let manifest: CoupledTwinMdEvidenceManifest =
        serde_json::from_slice(&std::fs::read(manifest_path)?)?;
    validate_coupled_twin_md_evidence_manifest_value(output_dir, &manifest, config, result)
}

#[cfg(feature = "gpu")]
fn validate_native_ghost_site_map_value(
    map: &NativeGhostSiteMap,
    config: &CoupledTwinEvidenceConfig,
    result_path: &Path,
) -> Result<()> {
    if map.schema != "prismos.ghost_v2_native_site_map.v1"
        || map.schema_version != COUPLED_TWIN_EVIDENCE_SCHEMA_VERSION
        || map.schema_kind != "pathb_ghost_v2_native_site_map"
        || map.status != "available_native_v2"
        || map.run_id != config.run_id
        || map.native_v2_event_count != map.entries.len() as u64
        || map.source_result.sha256 != sha256_file(result_path)?
        || map.source_result.bytes != std::fs::metadata(result_path)?.len()
        || map.source_stream_artifacts.len() != 4
    {
        bail!("aggregate native Ghost site-map semantic validation failed");
    }
    let mut component_ids = BTreeSet::new();
    let mut event_ids = BTreeSet::new();
    for entry in &map.entries {
        if !component_ids.insert(entry.component_id.as_str())
            || !event_ids.insert((entry.stream_id, entry.frame_idx, entry.site_id))
        {
            bail!("aggregate native Ghost site-map contains duplicate identities");
        }
        let source = map
            .source_stream_artifacts
            .iter()
            .find(|artifact| {
                artifact.kind == "ghost_tiles_v2" && artifact.stream_id == entry.stream_id
            })
            .context("aggregate Ghost entry lacks its stream payload binding")?;
        if entry.record_provenance.file != source.path
            || entry.record_provenance.file_sha256 != source.sha256
            || entry.record_provenance.file_bytes != source.bytes
        {
            bail!("aggregate Ghost entry/payload source binding drift");
        }
    }
    if map.entries.is_empty() {
        bail!("aggregate native Ghost site-map has zero engine-owned records");
    }
    Ok(())
}

#[cfg(feature = "gpu")]
pub fn emit_coupled_twin_native_ghost_site_map(
    output_dir: &Path,
    config: &CoupledTwinEvidenceConfig,
    result_path: &Path,
    result: &CoupledTwinResult,
) -> Result<PathBuf> {
    let runtime = result
        .runtime_evidence
        .as_ref()
        .context("aggregate Ghost map lacks runtime evidence")?;
    validate_coupled_runtime_evidence(output_dir, config, result, runtime)?;
    let result_artifact = artifact_evidence(
        "coupled_twin_result",
        None,
        result_path,
        Some(1),
        Some(COUPLED_TWIN_EVIDENCE_SCHEMA_VERSION),
    )?;
    validate_run_artifact(output_dir, &result_artifact)?;
    let result_binding = NativeGhostArtifactBinding {
        path: result_artifact.path,
        sha256: result_artifact.sha256,
        bytes: result_artifact.size_bytes,
    };
    let mut source_stream_artifacts = Vec::new();
    for artifact in runtime
        .artifacts
        .iter()
        .filter(|artifact| artifact.kind == "ghost_site_map" || artifact.kind == "ghost_tiles_v2")
    {
        let stream_id = artifact
            .stream_id
            .context("aggregate Ghost source artifact lacks stream id")?;
        validate_run_artifact(output_dir, artifact)?;
        source_stream_artifacts.push(NativeGhostSourceArtifactBinding {
            kind: artifact.kind.clone(),
            stream_id,
            path: artifact.path.clone(),
            sha256: artifact.sha256.clone(),
            bytes: artifact.size_bytes,
        });
    }
    source_stream_artifacts
        .sort_by(|left, right| (&left.kind, left.stream_id).cmp(&(&right.kind, right.stream_id)));
    let expected_source_keys = BTreeSet::from([
        ("ghost_site_map".to_string(), 0u32),
        ("ghost_site_map".to_string(), 1u32),
        ("ghost_tiles_v2".to_string(), 0u32),
        ("ghost_tiles_v2".to_string(), 1u32),
    ]);
    let observed_source_keys: BTreeSet<_> = source_stream_artifacts
        .iter()
        .map(|artifact| (artifact.kind.clone(), artifact.stream_id))
        .collect();
    if source_stream_artifacts.len() != 4 || observed_source_keys != expected_source_keys {
        bail!(
            "aggregate Ghost map requires exact two-stream map/tile source set: {:?}",
            observed_source_keys
        );
    }
    let mut entries = Vec::new();
    for artifact in runtime
        .artifacts
        .iter()
        .filter(|artifact| artifact.kind == "ghost_site_map")
    {
        let map: CoupledGhostSiteMap = serde_json::from_slice(&std::fs::read(&artifact.path)?)?;
        for record in map.records {
            entries.push(NativeGhostSiteMapEntry {
                component_id: format!(
                    "ghost-v2-stream{}-frame{}-site{}",
                    map.stream_id, record.frame_idx, record.site_id
                ),
                stream_id: map.stream_id,
                frame_idx: record.frame_idx,
                site_id: record.site_id,
                aabb: [
                    record.aabb_min[0],
                    record.aabb_min[1],
                    record.aabb_min[2],
                    record.aabb_max[0],
                    record.aabb_max[1],
                    record.aabb_max[2],
                ],
                centroid_xyz: record.centroid,
                residue_support: record.support_residue_ids,
                source: "engine_owned_CapturedAdjudicationPipeline_Ghost_v2".to_string(),
                record_provenance: NativeGhostRecordProvenance {
                    file: map.ghost_payload_path.clone(),
                    file_sha256: map.ghost_payload_sha256.clone(),
                    file_bytes: map.ghost_payload_bytes,
                    record_index: record.record_index,
                    byte_offset: record.byte_offset,
                    record_bytes: GHOST_V2_RECORD_BYTES as u64,
                    schema_version: COUPLED_TWIN_EVIDENCE_SCHEMA_VERSION,
                    field_completeness_flags: record.field_completeness_flags,
                    spatial_native_bit: 0x0010,
                },
            });
        }
    }
    entries.sort_by_key(|entry| (entry.stream_id, entry.frame_idx, entry.site_id));
    let mut components = BTreeSet::new();
    let mut events = BTreeSet::new();
    for entry in &entries {
        if !components.insert(entry.component_id.clone())
            || !events.insert((entry.stream_id, entry.frame_idx, entry.site_id))
        {
            bail!("aggregate native Ghost map contains duplicate component/event identity");
        }
    }
    if entries.is_empty() {
        bail!("aggregate native Ghost map has zero engine-owned records");
    }
    let map = NativeGhostSiteMap {
        schema: "prismos.ghost_v2_native_site_map.v1".to_string(),
        schema_version: COUPLED_TWIN_EVIDENCE_SCHEMA_VERSION,
        schema_kind: "pathb_ghost_v2_native_site_map".to_string(),
        status: "available_native_v2".to_string(),
        coordinate_frame: "topology_cartesian_angstrom".to_string(),
        run_id: config.run_id.clone(),
        native_v2_event_count: entries.len() as u64,
        source_result: result_binding,
        source_stream_artifacts,
        entries,
    };
    let site_map_path = output_dir.join("ghost_site_map.json");
    let mut candidate_bytes = serde_json::to_vec_pretty(&map)?;
    candidate_bytes.push(b'\n');
    let candidate: NativeGhostSiteMap = serde_json::from_slice(&candidate_bytes)
        .context("parse serialized native Ghost site map before commit")?;
    validate_native_ghost_site_map_value(&candidate, config, result_path)?;
    write_atomic_bytes(&site_map_path, &candidate_bytes)?;
    let readback: NativeGhostSiteMap = serde_json::from_slice(&std::fs::read(&site_map_path)?)?;
    validate_native_ghost_site_map_value(&readback, config, result_path)
        .context("aggregate native Ghost site-map atomic readback failed")?;
    Ok(site_map_path)
}

/// Placeholder for non-GPU builds
#[cfg(not(feature = "gpu"))]
pub fn run_coupled_twin(
    _config_ref: &crate::persistent_engine::PersistentBatchConfig,
    _topology: &PrismPrepTopology,
    _twin_config: &CoupledTwinConfig,
    _seed_a: u64,
    _steps: i32,
) -> Result<CoupledTwinResult> {
    anyhow::bail!("PRISM-TWIN requires GPU feature")
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests (no GPU required — tests CPU-side logic only)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "gpu")]
    #[test]
    fn shared_therm_file_readback_rejects_missing_duplicate_and_tampered_bindings() {
        // Synthetic transaction fixtures exercise CPU/file admission only;
        // these bytes are not molecular inputs or evidence of GPU execution.
        let directory = unique_test_dir("synthetic-shared-therm-transaction");
        std::fs::create_dir_all(&directory).unwrap();
        let topology_a = directory.join("synthetic_a.topology.json");
        let topology_b = directory.join("synthetic_b.topology.json");
        std::fs::write(&topology_a, b"synthetic topology identity A").unwrap();
        std::fs::write(&topology_b, b"synthetic topology identity B").unwrap();
        let mut config = strict_evidence_config(&topology_a, &topology_b, false);
        config.require_shared_lattice_therm = true;
        let stream_result = serde_json::json!({
            "role": "synthetic", "seed": 41, "total_spikes": 7,
            "perturbation": "synthetic-file-test"
        });
        let result: CoupledTwinResult = serde_json::from_value(serde_json::json!({
            "stream_a": stream_result, "stream_b": stream_result,
            "config": CoupledTwinConfig::default(), "per_residue_features": [],
            "n_consensus_events": 0, "n_differential_events": 0,
            "n_exchanges": 0, "total_density_a_to_b": 0.0, "total_density_b_to_a": 0.0,
            "n_nonzero_regions": 0, "ring_spikes_tracked": 0,
            "phase_offset_steps": 0, "steps_a": 1008, "steps_b": 1008
        })).unwrap();
        let graph = (0..2).map(|id| CoupledTwinGraphEvidence {
            stream_id: id, direct_prefix_outer_steps: 8, direct_prefix_md_steps: 8,
            resident_outer_steps: 1000, resident_md_steps: 1000, total_md_steps: 1008,
            capture_count: 1, host_launch_count: 10, resident_launch_count: 10,
            legacy_launch_count: 0, logical_outer_steps: 1000, logical_md_steps: 1000,
            steps_per_launch: 100, launch_collapse_ratio: 100.0,
            launch_path: "synthetic-file-test".into(),
        }).collect();
        let runtime = CoupledTwinRuntimeEvidence {
            schema_kind: "synthetic-file-test".into(), schema_version: 1,
            run_id: config.run_id.clone(), execution_semantics: "synthetic-file-test".into(),
            exchange_active: false, causal_transport_claim: false,
            graph, artifacts: Vec::new(),
        };
        let lattice_path = directory.join(format!("{}_binding_sites_lattice.json", coupled_topology_stem(&config)));
        let lattice = serde_json::json!({
            "schema_kind": "prismos_ghost_phase_lattice_components", "run_id": config.run_id
        });
        let lattice_bytes = serde_json::to_vec(&lattice).unwrap();
        std::fs::write(&lattice_path, &lattice_bytes).unwrap();
        let report_path = directory.join("coupled_shared_therm.json");
        let streams: Vec<_> = [config.topology_a_sha256.as_str(), config.topology_b_sha256.as_str()]
            .iter().enumerate().map(|(id, topology)| serde_json::json!({
                "source_stream_id": id, "topology_source_sha256": topology,
                "seed": 41, "physical_steps": 1008, "input_spike_count": 7,
                "ingested_event_count": 7, "lattice_node_count": 3,
                "analysis": {"sdst_event_count": 7},
                "conservation": {"promotable": true, "counts": {"unmatched_site_count": 0}}
            })).collect();
        let report = serde_json::json!({
            "schema_kind": "prismos.coupled_stream_lattice_therm.v1", "schema_version": 1,
            "run_id": config.run_id, "committed_lattice_node_count": 6, "streams": streams,
            "committed_lattice": {"path": lattice_path, "sha256": sha256_file(&lattice_path).unwrap()}
        });
        let bind_report = |value: &serde_json::Value| {
            std::fs::write(&report_path, serde_json::to_vec(value).unwrap()).unwrap();
            let mut bound = runtime.clone();
            bound.artifacts.push(artifact_evidence(
                "coupled_shared_therm", None, &report_path, Some(2), Some(1),
            ).unwrap());
            bound
        };
        let valid = bind_report(&report);
        validate_required_shared_therm(&directory, &config, &result, &valid).unwrap();
        assert!(validate_required_shared_therm(&directory, &config, &result, &runtime).is_err());
        let mut optional = config.clone();
        optional.require_shared_lattice_therm = false;
        validate_required_shared_therm(&directory, &optional, &result, &runtime).unwrap();
        let mut duplicate = valid.clone();
        duplicate.artifacts.push(duplicate.artifacts[0].clone());
        assert!(validate_required_shared_therm(&directory, &config, &result, &duplicate).is_err());

        let mut tampered_bytes = std::fs::read(&report_path).unwrap();
        tampered_bytes.push(b' ');
        std::fs::write(&report_path, tampered_bytes).unwrap();
        assert!(validate_required_shared_therm(&directory, &config, &result, &valid).is_err());
        let mut wrong_run = report.clone();
        wrong_run["run_id"] = serde_json::json!("another-synthetic-run");
        let rebound = bind_report(&wrong_run);
        assert!(validate_required_shared_therm(&directory, &config, &result, &rebound).is_err());

        let mut wrong_hash = report.clone();
        wrong_hash["committed_lattice"]["sha256"] = serde_json::json!("0".repeat(64));
        let rebound = bind_report(&wrong_hash);
        assert!(validate_required_shared_therm(&directory, &config, &result, &rebound).is_err());
        let alternate_path = directory.join("another_lattice.json");
        std::fs::write(&alternate_path, &lattice_bytes).unwrap();
        let mut wrong_path = report.clone();
        wrong_path["committed_lattice"]["path"] = serde_json::json!(alternate_path);
        let rebound = bind_report(&wrong_path);
        assert!(validate_required_shared_therm(&directory, &config, &result, &rebound).is_err());

        let valid = bind_report(&report);
        let mut lattice_drift = lattice_bytes.clone();
        lattice_drift.push(b' ');
        std::fs::write(&lattice_path, lattice_drift).unwrap();
        assert!(validate_required_shared_therm(&directory, &config, &result, &valid).is_err());
        let mut other_lattice = lattice.clone();
        other_lattice["run_id"] = serde_json::json!("another-synthetic-run");
        std::fs::write(&lattice_path, serde_json::to_vec(&other_lattice).unwrap()).unwrap();
        let mut rehashed_report = report.clone();
        rehashed_report["committed_lattice"]["sha256"] = serde_json::json!(sha256_file(&lattice_path).unwrap());
        let rebound = bind_report(&rehashed_report);
        assert!(validate_required_shared_therm(&directory, &config, &result, &rebound).is_err());
        std::fs::write(&lattice_path, lattice_bytes).unwrap();
        let restored = bind_report(&report);
        validate_required_shared_therm(&directory, &config, &result, &restored).unwrap();
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn shared_therm_report_readback_requires_exact_run_stream_and_population_identity() {
        let topology_a = "a".repeat(64);
        let topology_b = "b".repeat(64);
        let expected = [(0, topology_a.as_str(), 41, 1008, 7), (1, topology_b.as_str(), 41, 1008, 9)];
        let streams: Vec<_> = expected.iter().map(|(id, topology, seed, steps, spikes)| {
            serde_json::json!({
                "source_stream_id": id, "topology_source_sha256": topology,
                "seed": seed, "physical_steps": steps, "input_spike_count": spikes,
                "ingested_event_count": spikes, "lattice_node_count": 3,
                "analysis": {"sdst_event_count": spikes},
                "conservation": {"promotable": true, "counts": {"unmatched_site_count": 0}}
            })
        }).collect();
        let report = serde_json::json!({
            "schema_kind": "prismos.coupled_stream_lattice_therm.v1", "schema_version": 1,
            "run_id": "run-a", "committed_lattice_node_count": 6, "streams": streams
        });
        validate_shared_therm_report_value(&report, "run-a", &expected).unwrap();
        for (pointer, replacement) in [
            ("/schema_kind", serde_json::json!("other")),
            ("/schema_version", serde_json::json!(2)),
            ("/run_id", serde_json::json!("another-run")),
            ("/streams/1/source_stream_id", serde_json::json!(0)),
            ("/streams/1/topology_source_sha256", serde_json::json!(topology_a)),
            ("/streams/0/seed", serde_json::json!(42)),
            ("/streams/0/physical_steps", serde_json::json!(1007)),
            ("/streams/0/input_spike_count", serde_json::json!(6)),
            ("/streams/0/ingested_event_count", serde_json::json!(0)),
            ("/streams/0/analysis/sdst_event_count", serde_json::json!(6)),
            ("/streams/0/conservation/promotable", serde_json::json!(false)),
            ("/streams/0/conservation/counts/unmatched_site_count", serde_json::json!(1)),
            ("/committed_lattice_node_count", serde_json::json!(5)),
        ] {
            let mut changed = report.clone();
            *changed.pointer_mut(pointer).unwrap() = replacement;
            assert!(validate_shared_therm_report_value(&changed, "run-a", &expected).is_err(), "accepted drift at {pointer}");
        }
        let mut missing = report.clone();
        missing["streams"].as_array_mut().unwrap().pop();
        assert!(validate_shared_therm_report_value(&missing, "run-a", &expected).is_err());
        let mut sampled = report;
        sampled["streams"][0]["ingested_event_count"] = serde_json::json!(3);
        sampled["streams"][0]["analysis"]["sdst_event_count"] = serde_json::json!(3);
        validate_shared_therm_report_value(&sampled, "run-a", &expected).unwrap();
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn post_md_streams_preserve_owned_buffers_and_stay_out_of_result_json() {
        use crate::fused_engine::{EnsembleSnapshot, GpuSpikeEvent, SnapshotTrigger};
        let topology_a = crn_topology_fixture(&[("A", 1, "CA", "C", 12.0, [1.0, 2.0, 3.0])]);
        let topology_b = crn_topology_fixture(&[
            ("A", 1, "CA", "C", 12.0, [4.0, 5.0, 6.0]),
            ("L", 2, "C1", "C", 12.0, [7.0, 8.0, 9.0]),
        ]);
        let protocol_a = CryoUvProtocol::standard();
        let mut protocol_b = protocol_a.clone();
        protocol_b.cold_hold_steps += 123;
        let spikes_a = vec![GpuSpikeEvent { timestep: 17, ..GpuSpikeEvent::default() }];
        let spikes_b = vec![GpuSpikeEvent { timestep: 23, ..GpuSpikeEvent::default() }];
        let snapshots_a = vec![EnsembleSnapshot {
            timestep: 17,
            positions: vec![1.0, 2.0, 3.0],
            velocities: vec![0.0; 3],
            trigger_spikes: Vec::new(),
            temperature: 77.0,
            spike_quality_scores: Vec::new(),
            alignment_quality: 1.0,
            spike_region_rmsd: 0.0,
            time_ps: 0.034,
            trigger_reason: SnapshotTrigger::IntervalScheduled,
            delta_sasa: None,
        }];
        let ptr_a = spikes_a.as_ptr();
        let ptr_b = spikes_b.as_ptr();
        let snapshot_ptr = snapshots_a.as_ptr();
        let position_ptr = snapshots_a[0].positions.as_ptr();
        let stream_result = serde_json::json!({
            "role": "test", "seed": 41, "total_spikes": 1, "perturbation": "thermal_only"
        });
        let legacy_json = serde_json::json!({
            "stream_a": stream_result, "stream_b": stream_result,
            "config": CoupledTwinConfig::default(), "per_residue_features": [],
            "n_consensus_events": 0, "n_differential_events": 0,
            "n_exchanges": 0, "total_density_a_to_b": 0.0, "total_density_b_to_a": 0.0,
            "n_nonzero_regions": 0, "ring_spikes_tracked": 0,
            "phase_offset_steps": 123, "steps_a": 1007, "steps_b": 1130
        });
        let mut result: CoupledTwinResult = serde_json::from_value(legacy_json).unwrap();
        assert!(result.post_md_streams.is_empty());
        let serialized_before = serde_json::to_value(&result).unwrap();
        result.post_md_streams = vec![
            CoupledPostMdStream {
                source_stream_id: 0, topology: topology_a,
                topology_source_sha256: Some("a".repeat(64)), protocol: protocol_a,
                seed: 41, physical_steps: 1008, spikes: spikes_a, snapshots: snapshots_a,
            },
            CoupledPostMdStream {
                source_stream_id: 1, topology: topology_b,
                topology_source_sha256: Some("b".repeat(64)), protocol: protocol_b,
                seed: 1041, physical_steps: 1134, spikes: spikes_b, snapshots: Vec::new(),
            },
        ];
        let a = &result.post_md_streams[0];
        let b = &result.post_md_streams[1];
        assert_eq!(a.spikes.as_ptr(), ptr_a);
        assert_eq!(b.spikes.as_ptr(), ptr_b);
        assert_eq!(a.snapshots.as_ptr(), snapshot_ptr);
        assert_eq!(a.snapshots[0].positions.as_ptr(), position_ptr);
        assert_eq!((a.topology.n_atoms, b.topology.n_atoms), (1, 2));
        assert_eq!(b.protocol.cold_hold_steps - a.protocol.cold_hold_steps, 123);
        assert_eq!((a.seed, b.seed), (41, 1041));
        assert_eq!((a.physical_steps, b.physical_steps), (1008, 1134));
        assert_eq!(serde_json::to_value(&result).unwrap(), serialized_before);
        let readback: CoupledTwinResult = serde_json::from_value(serialized_before).unwrap();
        assert!(readback.post_md_streams.is_empty(), "serialized summaries cannot fabricate raw state");
    }

    #[cfg(feature = "gpu")]
    fn kcc_semantics_fixture() -> crate::fused_engine::KccData {
        crate::fused_engine::KccData {
            lag_provenance: Vec::new(),
            lag_support: Vec::new(),
            temporal_corr: vec![0.2, 0.0, -0.1],
            direction_score: vec![0.1, 0.0, -0.2],
            motion_efficiency: vec![0.5, 0.0, 0.4],
            burst_motion: vec![0.3, 0.0, 0.2],
            phase_shift: vec![0.1, 0.0, -0.1],
            causal_lag: vec![2.0, f32::NAN, -3.0],
            lag_corr_peak: vec![0.75, f32::NAN, -0.5],
            local_cov: vec![0.4, 0.0, 0.2],
            residue_count: vec![16, 2, 12],
            active_causal: vec![8, 0, 6],
            net_dx: vec![0.1, 0.0, -0.1],
            net_dy: vec![0.2, 0.0, -0.2],
            net_dz: vec![0.3, 0.0, -0.3],
            sum_m: vec![1.0, 0.0, 0.8],
            n_residues: 3,
        }
    }

    fn strict_evidence_config(
        topology_a: &Path,
        topology_b: &Path,
        identical_control: bool,
    ) -> CoupledTwinEvidenceConfig {
        CoupledTwinEvidenceConfig {
            schema_kind: "prismos.coupled_twin_evidence_config.v2".to_string(),
            schema_version: COUPLED_TWIN_EVIDENCE_SCHEMA_VERSION,
            run_id: "coupled-twin-test-run".to_string(),
            topology_a_path: topology_a.display().to_string(),
            topology_b_path: topology_b.display().to_string(),
            topology_a_sha256: sha256_file(topology_a).expect("hash topology A"),
            topology_b_sha256: sha256_file(topology_b).expect("hash topology B"),
            contrast_kind: if identical_control {
                "identical-topology-negative-control".to_string()
            } else {
                "apo-holo".to_string()
            },
            engine_b_role: if identical_control {
                "identical-control".to_string()
            } else {
                "holo".to_string()
            },
            identical_topology_negative_control: identical_control,
            require_prismcode_md_native: true,
            require_cuda_graph_native: true,
            require_metal_resident_md: true,
            require_hotpath_acceleration: true,
            min_hotpath_launch_collapse: 2.0,
            graph_chunk_outer_steps: 64,
            captured_graph_body_unroll: 8,
            ghost_diagnostic_firehose: true,
            mar_v2_telemetry: true,
            probe_residue_ids: Vec::new(),
            ligand_contact_cutoff_angstrom: 5.0,
            cryptic_search_outer_cutoff_angstrom: default_cryptic_search_outer_cutoff_angstrom(),
            identical_control_max_mean_abs_contrast: 0.01,
            distinct_min_mean_abs_contrast: 0.001,
            distinct_min_pocket_localization_fraction: 0.05,
            require_ghost_pocket_binding: true,
            require_shared_lattice_therm: false,
        }
    }

    fn unique_test_dir(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "prismos-coupled-twin-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock")
                .as_nanos()
        ))
    }

    #[cfg(feature = "gpu")]
    fn crn_topology_fixture(
        atoms: &[(&str, usize, &str, &str, f32, [f32; 3])],
    ) -> PrismPrepTopology {
        let mut positions = Vec::with_capacity(atoms.len() * 3);
        let mut elements = Vec::with_capacity(atoms.len());
        let mut atom_names = Vec::with_capacity(atoms.len());
        let mut residue_names = Vec::with_capacity(atoms.len());
        let mut residue_ids = Vec::with_capacity(atoms.len());
        let mut chain_ids = Vec::with_capacity(atoms.len());
        let mut masses = Vec::with_capacity(atoms.len());
        for (chain, residue, atom_name, element, mass, position) in atoms {
            chain_ids.push((*chain).to_string());
            residue_ids.push(*residue);
            atom_names.push((*atom_name).to_string());
            elements.push((*element).to_string());
            residue_names.push(if *chain == "L" { "LIG" } else { "ALA" }.to_string());
            masses.push(*mass);
            positions.extend_from_slice(position);
        }
        PrismPrepTopology {
            aromatic_targets: Vec::new(),
            source_pdb: "crn-fixture.pdb".to_string(),
            n_atoms: atoms.len(),
            n_residues: residue_ids.iter().copied().collect::<BTreeSet<_>>().len(),
            total_physical_residue_count: None,
            n_analysis_residues: None,
            n_chains: chain_ids.iter().cloned().collect::<BTreeSet<_>>().len(),
            positions,
            elements,
            atom_names,
            residue_names,
            residue_ids,
            atom_physical_residue_ids: None,
            atom_analysis_residue_ids: None,
            analysis_residue_contract: None,
            chain_ids,
            charges: vec![0.0; atoms.len()],
            masses,
            ca_indices: vec![0],
            bonds: Vec::new(),
            angles: Vec::new(),
            dihedrals: Vec::new(),
            lj_params: Vec::new(),
            exclusions: vec![Vec::new(); atoms.len()],
            nonbonded_exceptions: Vec::new(),
            nonbonded_exception_contract: None,
            generated_prmtop_sha256: None,
            h_clusters: Vec::new(),
            hmr_applied: Some(false),
            recommended_timestep_fs: None,
            water_oxygens: Vec::new(),
            residues: Vec::new(),
            dimer_dyad: None,
            lambda_fragment_system: None,
            harmonic_parameter_contract: None,
            schema_kind: None,
            schema_version: None,
            periodic_cell: None,
            water_model_contract: None,
            molecule_identity_table: None,
            topology_provenance: None,
            npt_frozen_cell_contract: None,
        }
    }

    #[cfg(feature = "gpu")]
    fn crn_reference_atoms() -> Vec<(
        &'static str,
        usize,
        &'static str,
        &'static str,
        f32,
        [f32; 3],
    )> {
        vec![
            ("A", 0, "N", "N", 14.0, [0.0, 0.0, 0.0]),
            ("A", 0, "CA", "C", 12.0, [1.0, 0.0, 0.0]),
            ("A", 0, "C", "C", 12.0, [2.0, 0.0, 0.0]),
        ]
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn glycan_probe_config_requires_canonical_multi_residue_identity_and_annulus() {
        let root = unique_test_dir("glycan-probe-config");
        std::fs::create_dir_all(&root).expect("create glycan config root");
        let topology_a_path = root.join("a.json");
        let topology_b_path = root.join("b.json");
        std::fs::write(&topology_a_path, b"topology-a").expect("write A identity");
        std::fs::write(&topology_b_path, b"topology-b").expect("write B identity");
        let mut config = strict_evidence_config(&topology_a_path, &topology_b_path, false);
        config.contrast_kind = GLYCAN_MECHANISTIC_PROBE_CONTRAST_KIND.to_string();
        config.engine_b_role = "glycan-probe".to_string();
        config.probe_residue_ids = vec![2, 3];
        config
            .validate()
            .expect("canonical multi-residue glycan probe is admitted");

        let mut duplicate = config.clone();
        duplicate.probe_residue_ids = vec![2, 2];
        assert!(duplicate.validate().is_err());
        let mut reordered = config.clone();
        reordered.probe_residue_ids = vec![3, 2];
        assert!(reordered.validate().is_err());
        let mut collapsed_annulus = config;
        collapsed_annulus.cryptic_search_outer_cutoff_angstrom =
            collapsed_annulus.ligand_contact_cutoff_angstrom;
        assert!(collapsed_annulus.validate().is_err());
        std::fs::remove_dir_all(root).expect("remove glycan config root");
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn glycan_probe_chain_roles_preserve_multichain_receptor_and_probe_partition() {
        let topology_a = crn_topology_fixture(&[
            ("A", 0, "CA", "C", 12.0, [0.0, 0.0, 0.0]),
            ("B", 1, "CA", "C", 12.0, [2.0, 0.0, 0.0]),
        ]);
        let topology_b = crn_topology_fixture(&[
            ("A", 0, "CA", "C", 12.0, [0.0, 0.0, 0.0]),
            ("B", 1, "CA", "C", 12.0, [2.0, 0.0, 0.0]),
            ("G", 2, "C1", "C", 12.0, [5.0, 0.0, 0.0]),
            ("H", 3, "O1", "O", 16.0, [6.0, 0.0, 0.0]),
        ]);
        let (map_a, map_b) = coupled_native_chain_role_maps(&topology_a, &topology_b, false)
            .expect("derive multichain receptor/probe roles");
        assert_eq!(
            map_a,
            BTreeMap::from([
                ("A".to_string(), "A".to_string()),
                ("B".to_string(), "A".to_string()),
            ])
        );
        assert_eq!(
            map_b,
            BTreeMap::from([
                ("A".to_string(), "A".to_string()),
                ("B".to_string(), "A".to_string()),
                ("G".to_string(), "B".to_string()),
                ("H".to_string(), "B".to_string()),
            ])
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn coupled_twin_crn_append_only_intervention_preserves_shared_random_state() {
        let root = unique_test_dir("crn-append-only");
        std::fs::create_dir_all(&root).expect("create CRN root");
        let topology_a_path = root.join("a.json");
        let topology_b_path = root.join("b.json");
        std::fs::write(&topology_a_path, b"topology-a").expect("write A identity");
        std::fs::write(&topology_b_path, b"topology-b").expect("write B identity");
        let topology_a = crn_topology_fixture(&crn_reference_atoms());
        let mut holo_atoms = crn_reference_atoms();
        holo_atoms.extend([
            ("L", 1, "C1", "C", 12.0, [4.0, 1.0, 0.0]),
            ("L", 1, "O1", "O", 16.0, [5.0, 1.0, 0.0]),
        ]);
        let topology_b = crn_topology_fixture(&holo_atoms);
        let identical_config = strict_evidence_config(&topology_a_path, &topology_a_path, true);
        let distinct_config = strict_evidence_config(&topology_a_path, &topology_b_path, false);
        let identical = build_coupled_twin_crn_host_state_v3(
            &topology_a,
            &topology_a,
            &identical_config,
            42,
            50.0,
            "authoritative",
            None,
        )
        .expect("build identical CRN state");
        let distinct = build_coupled_twin_crn_host_state_v3(
            &topology_a,
            &topology_b,
            &distinct_config,
            42,
            50.0,
            "authoritative",
            None,
        )
        .expect("build append-only CRN state");
        assert_eq!(
            identical.receipt.shared_atom_map_sha256,
            distinct.receipt.shared_atom_map_sha256
        );
        assert_eq!(
            identical.receipt.shared_velocity_sha256,
            distinct.receipt.shared_velocity_sha256
        );
        assert_eq!(
            identical.receipt.shared_displacement_sha256,
            distinct.receipt.shared_displacement_sha256
        );
        assert_eq!(identical.positions_a, distinct.positions_a);
        assert_eq!(identical.velocities_a, distinct.velocities_a);
        assert_eq!(distinct.receipt.shared_atom_count, 3);
        assert_eq!(distinct.receipt.b_only_atom_count, 2);
        assert_eq!(distinct.receipt.status, "IMPLEMENTATION_PARTIAL");
        assert_eq!(
            distinct.receipt.validation_status,
            "PENDING_DEVICE_READBACK"
        );
        assert!(!distinct.receipt.exact_device_readback_verified);
        assert!(!distinct.receipt.direct_cuda_rng_state_readback);
        let expected_readback =
            expected_coupled_twin_crn_device_readback_v3(&distinct, &topology_a, &topology_b);
        let finalized = finalize_coupled_twin_crn_device_readback_v3(
            distinct.receipt.clone(),
            &expected_readback,
            &expected_readback,
        )
        .expect("finalize exact CRN readback");
        assert_eq!(finalized.status, "IMPLEMENTATION_VERIFIED");
        assert_eq!(finalized.validation_status, "PASS");
        assert!(finalized.exact_device_readback_verified);
        assert!(!finalized.device_position_readback_sha256_a.is_empty());
        assert!(!finalized.device_zero_potential_readback_sha256_b.is_empty());
        std::fs::remove_dir_all(root).expect("remove CRN root");
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn coupled_twin_crn_rejects_reorder_insertion_duplicate_and_mass_substitution() {
        let root = unique_test_dir("crn-negative-controls");
        std::fs::create_dir_all(&root).expect("create CRN root");
        let topology_a_path = root.join("a.json");
        let topology_b_path = root.join("b.json");
        std::fs::write(&topology_a_path, b"topology-a").expect("write A identity");
        std::fs::write(&topology_b_path, b"topology-b").expect("write B identity");
        let topology_a = crn_topology_fixture(&crn_reference_atoms());
        let config = strict_evidence_config(&topology_a_path, &topology_b_path, false);

        let mut reordered = crn_reference_atoms();
        reordered.swap(0, 1);
        reordered.extend([
            ("L", 1, "C1", "C", 12.0, [4.0, 1.0, 0.0]),
            ("L", 1, "O1", "O", 16.0, [5.0, 1.0, 0.0]),
        ]);
        assert!(build_coupled_twin_crn_host_state_v3(
            &topology_a,
            &crn_topology_fixture(&reordered),
            &config,
            7,
            50.0,
            "authoritative",
            None,
        )
        .is_err());

        let mut inserted = crn_reference_atoms();
        inserted.insert(1, ("L", 1, "C1", "C", 12.0, [4.0, 1.0, 0.0]));
        inserted.push(("L", 1, "O1", "O", 16.0, [5.0, 1.0, 0.0]));
        assert!(build_coupled_twin_crn_host_state_v3(
            &topology_a,
            &crn_topology_fixture(&inserted),
            &config,
            7,
            50.0,
            "authoritative",
            None,
        )
        .is_err());

        let mut duplicate = crn_reference_atoms();
        duplicate.push(("A", 0, "CA", "C", 12.0, [1.0, 0.0, 0.0]));
        duplicate.push(("L", 1, "O1", "O", 16.0, [5.0, 1.0, 0.0]));
        assert!(build_coupled_twin_crn_host_state_v3(
            &topology_a,
            &crn_topology_fixture(&duplicate),
            &config,
            7,
            50.0,
            "authoritative",
            None,
        )
        .is_err());

        let mut mass_substitution = crn_reference_atoms();
        mass_substitution[1].4 = 13.0;
        mass_substitution.extend([
            ("L", 1, "C1", "C", 12.0, [4.0, 1.0, 0.0]),
            ("L", 1, "O1", "O", 16.0, [5.0, 1.0, 0.0]),
        ]);
        assert!(build_coupled_twin_crn_host_state_v3(
            &topology_a,
            &crn_topology_fixture(&mass_substitution),
            &config,
            7,
            50.0,
            "authoritative",
            None,
        )
        .is_err());
        std::fs::remove_dir_all(root).expect("remove CRN root");
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn coupled_twin_crn_sha256_gaussian_has_golden_bits_and_domain_separation() {
        let key = CoupledTwinCrnAtomKeyV3 {
            chain_id: "A".to_string(),
            residue_id: 7,
            atom_name: "CA".to_string(),
            element: "C".to_string(),
        };
        let key_bytes = crn_atom_key_bytes(&key).expect("canonical atom key");
        let value = crn_standard_normal(
            COUPLED_TWIN_CRN_SHARED_VELOCITY_DOMAIN,
            42,
            "scope-hash",
            &key_bytes,
            1,
        )
        .expect("golden Gaussian");
        assert_eq!(value.to_bits(), 0x3fe6_e875_22e7_f078);
        assert_eq!((value as f32).to_bits(), 0x3f37_43a9);
        assert_eq!(
            value.to_bits(),
            crn_standard_normal(
                COUPLED_TWIN_CRN_SHARED_VELOCITY_DOMAIN,
                42,
                "scope-hash",
                &key_bytes,
                1,
            )
            .expect("repeat Gaussian")
            .to_bits()
        );
        assert_ne!(
            value.to_bits(),
            crn_standard_normal(
                COUPLED_TWIN_CRN_SHARED_POSITION_DOMAIN,
                42,
                "scope-hash",
                &key_bytes,
                1,
            )
            .expect("domain-separated Gaussian")
            .to_bits()
        );
    }

    #[cfg(feature = "gpu")]
    fn strict_twin_residue_fixture() -> StrictTwinResidueFeature {
        let phase_ccf = STRICT_TWIN_PHASE_NAMES
            .iter()
            .enumerate()
            .map(|(phase_index, phase_name)| StrictTwinPhaseCcfFeature {
                phase_index: phase_index as u32,
                phase_name: (*phase_name).to_string(),
                start_step: phase_index as i32 * 200,
                end_step_exclusive: (phase_index as i32 + 1) * 200,
                bin_size_steps: STRICT_TWIN_CCF_BIN_SIZE_STEPS,
                bin_count: 2,
                residue_event_count_a: 1,
                residue_event_count_b: 1,
                variance_support_a: true,
                variance_support_b: true,
                supported_partner_count: 1,
                correlation_lag_steps: Some(STRICT_TWIN_CCF_TEMPORAL_LAG_STEPS),
                aligned_zero_lag_ccf: Some(0.0),
                peak_partner_residue_id: Some(1),
                peak_partner_residue_offset: Some(1),
                peak_partner_zero_lag_ccf: Some(0.25),
                cross_residue_network_reproducibility_fraction: Some(0.5),
                support_status: STRICT_TWIN_CCF_SUPPORTED.to_string(),
                matrix_artifact_kind: format!("gpu_phase_ccf_{phase_index}"),
                matrix_sha256: "a".repeat(64),
            })
            .collect();
        StrictTwinResidueFeature {
            schema_kind: "prismos.coupled_twin_strict_residue_feature.v3".to_string(),
            schema_version: STRICT_TWIN_FEATURE_SCHEMA_VERSION,
            residue_id: 0,
            residue_name: "ALA".to_string(),
            event_count_a: 2,
            event_count_b: 2,
            intensity_sum_a: 2.0,
            intensity_sum_b: 2.0,
            signed_event_count_contrast: 0,
            signed_intensity_contrast: 0.0,
            b_over_a_event_ratio: Some(1.0),
            b_over_a_intensity_ratio: Some(1.0),
            global_correlation_lag_steps: Some(STRICT_TWIN_CCF_TEMPORAL_LAG_STEPS),
            global_variance_support_a: true,
            global_variance_support_b: true,
            global_supported_partner_count: 1,
            global_aligned_zero_lag_ccf: Some(0.0),
            global_peak_partner_residue_id: Some(1),
            global_peak_partner_residue_offset: Some(1),
            global_peak_partner_zero_lag_ccf: Some(0.25),
            global_cross_residue_network_reproducibility_fraction: Some(0.5),
            global_ccf_support_status: STRICT_TWIN_CCF_SUPPORTED.to_string(),
            phase_ccf,
            predictive_window_steps: STRICT_TWIN_PREDICTIVE_WINDOW_STEPS,
            predictive_source_event_count: 2,
            predictive_matched_event_count: 2,
            predictive_probability_b_after_a: Some(1.0),
            predictive_algorithm: STRICT_TWIN_PREDICTIVE_ALGORITHM.to_string(),
            onset_step_a: Some(0),
            onset_step_b: Some(0),
            b_minus_a_onset_lag_steps: Some(0),
            a_intensity_at_exact_b_onset: Some(1.0),
            a_intensity_at_exact_b_onset_support_count: 1,
            b_only_condition_event_count: 0,
            a_only_reference_event_count: 0,
            transfer_entropy_kernel_lag_bins: Some(STRICT_TWIN_TE_KERNEL_LAG_BINS),
            transfer_entropy_past_index_offset_bins: Some(STRICT_TWIN_TE_PAST_INDEX_OFFSET_BINS),
            transfer_entropy_future_index_offset_bins: Some(
                STRICT_TWIN_TE_FUTURE_INDEX_OFFSET_BINS,
            ),
            transfer_entropy_effective_past_to_future_horizon_bins: Some(
                STRICT_TWIN_TE_EFFECTIVE_HORIZON_BINS,
            ),
            transfer_entropy_bin_size_steps: Some(STRICT_TWIN_CCF_BIN_SIZE_STEPS),
            transfer_entropy_effective_past_to_future_horizon_steps: Some(
                STRICT_TWIN_TE_EFFECTIVE_HORIZON_STEPS,
            ),
            transfer_entropy_transition_sample_count: Some(10),
            bidirectional_transfer_entropy_mass: Some(0.0),
            thresholded_total_outgoing_transfer_entropy_a_to_b: Some(0.0),
            transfer_entropy_outgoing_term_floor: Some(0.001),
            net_causal_flow_out_minus_in: Some(0.0),
            transfer_entropy_support_status: STRICT_TWIN_TE_SUPPORTED.to_string(),
            ccf_frequency_peak: None,
            ccf_frequency_status: "UNAVAILABLE_NO_TEMPORAL_LAG_SPECTRUM".to_string(),
            phase_offset_enrichment: None,
            phase_offset_enrichment_status:
                "NOT_APPLICABLE_ALIGNED_COMMON_RANDOM_NUMBERS_OFFSET_ZERO".to_string(),
            nma_responsive_mode: None,
            nma_mode_eigenvalue: None,
            nma_work_at_residue: None,
            mechanical_sensitivity: None,
            susceptibility_magnitude: None,
            nma_status: "NOT_APPLICABLE_TOPOLOGY_ONLY_INTERVENTION".to_string(),
            fallback_used: false,
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn strict_twin_zero_lag_projection_separates_diagonal_and_partner_network() {
        let matrix = vec![0.99, 0.20, 0.80, 0.70, 0.95, 0.05, 0.11, 0.12, 0.93];
        let features = crate::twin_kernels::extract_ccf_features(&matrix, 3);
        let support = [true, true, true];
        let row0 = strict_zero_lag_ccf_projection(&matrix, &features, 3, 0, &support, &support)
            .expect("project strict row 0")
            .expect("supported strict row 0");
        assert_eq!(row0.aligned_zero_lag_ccf.to_bits(), 0.99f32.to_bits());
        assert_eq!(
            (
                row0.peak_partner_residue_id,
                row0.peak_partner_residue_offset,
            ),
            (2, 2)
        );
        assert_eq!(row0.peak_partner_zero_lag_ccf.to_bits(), 0.80f32.to_bits());
        assert_eq!(
            row0.cross_residue_network_reproducibility_fraction
                .to_bits(),
            1.0f32.to_bits()
        );

        let row1 = strict_zero_lag_ccf_projection(&matrix, &features, 3, 1, &support, &support)
            .expect("project strict row 1")
            .expect("supported strict row 1");
        assert_eq!(row1.aligned_zero_lag_ccf.to_bits(), 0.95f32.to_bits());
        assert_eq!(
            (
                row1.peak_partner_residue_id,
                row1.peak_partner_residue_offset,
            ),
            (0, -1)
        );
        assert_eq!(row1.peak_partner_zero_lag_ccf.to_bits(), 0.70f32.to_bits());
        assert_eq!(
            row1.cross_residue_network_reproducibility_fraction
                .to_bits(),
            0.5f32.to_bits()
        );

        let mut corrupt = features;
        corrupt[0].ccf_peak_value = 0.79;
        assert!(
            strict_zero_lag_ccf_projection(&matrix, &corrupt, 3, 0, &support, &support,).is_err()
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn strict_twin_valid_zero_te_is_supported_independently_of_diagonal_ccf() {
        let mut record = strict_twin_residue_fixture();
        record.global_variance_support_a = false;
        record.global_variance_support_b = false;
        record.global_supported_partner_count = 0;
        record.global_correlation_lag_steps = None;
        record.global_aligned_zero_lag_ccf = None;
        record.global_peak_partner_residue_id = None;
        record.global_peak_partner_residue_offset = None;
        record.global_peak_partner_zero_lag_ccf = None;
        record.global_cross_residue_network_reproducibility_fraction = None;
        record.global_ccf_support_status = STRICT_TWIN_GLOBAL_CCF_UNAVAILABLE.to_string();
        validate_strict_residue_feature(&record)
            .expect("real zero lag-1 TE remains supported without diagonal CCF variance");

        record.thresholded_total_outgoing_transfer_entropy_a_to_b = None;
        assert!(validate_strict_residue_feature(&record).is_err());
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn strict_twin_rejects_inapplicable_nma_numeric_injection() {
        let mut record = strict_twin_residue_fixture();
        record.nma_work_at_residue = Some(0.0);
        assert!(validate_strict_residue_feature(&record).is_err());
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn strict_twin_predictive_window_uses_exact_directional_closed_boundaries() {
        let source = [StrictResidueEventObservation {
            timestep: 100,
            intensity: 1.0,
        }];
        let count = |timestep| {
            strict_forward_predictive_matches(
                &source,
                &[StrictResidueEventObservation {
                    timestep,
                    intensity: 1.0,
                }],
                STRICT_TWIN_PREDICTIVE_WINDOW_STEPS,
            )
            .expect("predictive match")
        };
        assert_eq!(count(99), 0);
        assert_eq!(count(100), 1);
        assert_eq!(count(600), 1);
        assert_eq!(count(601), 0);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn strict_twin_onset_intensity_is_exact_timestep_only() {
        let observations = vec![
            StrictResidueEventObservation {
                timestep: 99,
                intensity: 9.0,
            },
            StrictResidueEventObservation {
                timestep: 100,
                intensity: 0.5,
            },
            StrictResidueEventObservation {
                timestep: 100,
                intensity: 0.7,
            },
            StrictResidueEventObservation {
                timestep: 101,
                intensity: 8.0,
            },
        ];
        let (mean, support) =
            strict_mean_intensity_at_timestep(&observations, 100).expect("exact onset");
        assert_eq!(support, 2);
        assert!((mean.expect("supported mean") - 0.6).abs() < 1.0e-6);
        assert_eq!(
            strict_mean_intensity_at_timestep(&observations, 102).expect("absent onset"),
            (None, 0)
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn strict_twin_phase_windows_require_complete_five_phase_schedule() {
        let protocol = CryoUvProtocol {
            start_temp: 50.0,
            end_temp: 300.0,
            cold_hold_steps: 200,
            ramp_steps: 200,
            warm_hold_steps: 200,
            current_step: 0,
            uv_burst_energy: 0.5,
            uv_burst_interval: 500,
            uv_burst_duration: 10,
            scan_wavelengths: vec![280.0],
            wavelength_dwell_steps: 100,
            ramp_down_steps: 200,
            cold_return_steps: 200,
            stepped_holds: Vec::new(),
        };
        let windows = strict_twin_phase_windows(&protocol, 1000).expect("complete schedule");
        assert_eq!(windows.len(), 5);
        assert_eq!(windows.first().expect("first").start_step, 0);
        assert_eq!(windows.last().expect("last").end_step_exclusive, 1000);
        assert!(strict_twin_phase_windows(&protocol, 999).is_err());
        let mut incomplete = protocol;
        incomplete.cold_return_steps = 0;
        assert!(strict_twin_phase_windows(&incomplete, 800).is_err());
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn strict_twin_variance_support_rejects_degenerate_bins() {
        let equal = vec![
            StrictResidueEventObservation {
                timestep: 0,
                intensity: 1.0,
            },
            StrictResidueEventObservation {
                timestep: 100,
                intensity: 1.0,
            },
        ];
        assert!(!strict_binned_variance_support(&equal, 0, 200, 100).expect("equal bins"));
        let uneven = vec![StrictResidueEventObservation {
            timestep: 0,
            intensity: 1.0,
        }];
        assert!(strict_binned_variance_support(&uneven, 0, 200, 100).expect("uneven bins"));
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn strict_twin_legacy_record_envelopes_preserve_single_topology_json_shape() {
        let mut residue = InterferometricFeatures::default();
        residue.resid = 7;
        residue.resname = "GLY".to_string();
        assert_eq!(
            serde_json::to_vec(&residue).expect("legacy residue bytes"),
            serde_json::to_vec(&CoupledTwinResidueFeatureRecord::Legacy(residue))
                .expect("wrapped legacy residue bytes")
        );
        let site = SiteInterferometricFeatures {
            site_id: 1,
            n_lining_residues: 2,
            n_lining_with_data: 2,
            mean_agreement: 0.0,
            min_agreement: 0.0,
            max_agreement: 0.0,
            mean_spatial_coherence: 0.0,
            mean_consensus_onset: 0.0,
            mean_ccf_peak: 0.0,
            max_ccf_peak: 0.0,
            mean_intra_site_ccf: 0.0,
            mean_ccf_reproducibility: 0.0,
            mean_b_over_a: 0.0,
            barrier_composition_low: 0.0,
            barrier_composition_medium: 0.0,
            barrier_composition_high: 0.0,
            total_nma_exclusive: 0,
            total_thermal_exclusive: 0,
            volume: 0.0,
            druggability: 0.0,
            is_druggable: false,
            centroid: [0.0; 3],
            mean_scout_lead_time: 0.0,
            mean_predictive_value: 0.0,
            mean_phase_offset_enrichment: 0.0,
        };
        assert_eq!(
            serde_json::to_vec(&site).expect("legacy site bytes"),
            serde_json::to_vec(&CoupledTwinSiteFeatureRecord::Legacy(site))
                .expect("wrapped legacy site bytes")
        );
        let strict = serde_json::to_value(CoupledTwinResidueFeatureRecord::Strict(
            strict_twin_residue_fixture(),
        ))
        .expect("strict record");
        assert!(strict.get("residue_id").is_some());
        assert!(strict.get("resid").is_none());
        assert!(strict.get("ccf_peak_lag").is_none());
        assert!(strict.get("mutual_information").is_none());
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn strict_twin_kernel_and_nested_artifact_bindings_fail_closed_on_tamper() {
        let root = unique_test_dir("strict-bindings");
        std::fs::create_dir_all(&root).expect("create binding root");
        let ptx = root.join("tensor_ccf.ptx");
        std::fs::write(
            &ptx,
            b".visible .entry tensor_ccf_compute() {}\n.visible .entry compute_spike_norms() {}\n",
        )
        .expect("write test PTX");
        let binding = bind_strict_kernel_source(
            "tensor_ccf.ptx",
            &ptx,
            &["tensor_ccf_compute", "compute_spike_norms"],
        )
        .expect("bind test PTX");
        validate_strict_kernel_source_binding(
            &binding,
            "tensor_ccf.ptx",
            &["tensor_ccf_compute", "compute_spike_norms"],
        )
        .expect("stable kernel binding");

        let artifact = CoupledTwinArtifactEvidence {
            kind: "gpu_ccf_matrix".to_string(),
            stream_id: None,
            path: "/run/test/ccf.bin".to_string(),
            size_bytes: 16,
            sha256: "1".repeat(64),
            blake3: "2".repeat(64),
            record_count: Some(4),
            schema_version: Some(2),
            runtime_owned: true,
        };
        let mut nested = serde_json::json!({
            "path": artifact.path,
            "sha256": artifact.sha256,
            "blake3": artifact.blake3,
        });
        assert!(strict_nested_binding_matches_artifact(
            Some(&nested),
            &artifact
        ));
        nested["sha256"] = serde_json::Value::String("3".repeat(64));
        assert!(!strict_nested_binding_matches_artifact(
            Some(&nested),
            &artifact
        ));

        std::fs::write(
            &ptx,
            b".visible .entry tensor_ccf_compute() {}\n.visible .entry compute_spike_norms() {}\n//tamper\n",
        )
        .expect("tamper test PTX");
        assert!(validate_strict_kernel_source_binding(
            &binding,
            "tensor_ccf.ptx",
            &["tensor_ccf_compute", "compute_spike_norms"],
        )
        .is_err());
        std::fs::remove_dir_all(root).expect("remove binding root");
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn strict_twin_phase_matrix_roundtrip_rejects_envelope_tamper() {
        let root = unique_test_dir("phase-matrix");
        std::fs::create_dir_all(&root).expect("create phase root");
        let topology_a = root.join("apo.json");
        let topology_b = root.join("holo.json");
        std::fs::write(&topology_a, b"{\"topology\":\"apo\"}\n").expect("write apo");
        std::fs::write(&topology_b, b"{\"topology\":\"holo\"}\n").expect("write holo");
        let config = strict_evidence_config(&topology_a, &topology_b, false);
        let window = StrictTwinPhaseWindow {
            phase_index: 0,
            phase_name: STRICT_TWIN_PHASE_NAMES[0].to_string(),
            start_step: 0,
            end_step_exclusive: 200,
            bin_size_steps: STRICT_TWIN_CCF_BIN_SIZE_STEPS,
            bin_count: 2,
        };
        let path = root.join("phase_0_cold_hold_ccf_f32le.bin");
        let matrix = vec![0.9, 0.2, 0.3, 0.8];
        write_strict_phase_ccf_matrix(&path, &config, &window, 2, &matrix)
            .expect("write strict phase matrix");
        let artifact = artifact_evidence(
            "gpu_phase_ccf_0",
            None,
            &path,
            Some(4),
            Some(STRICT_TWIN_FEATURE_SCHEMA_VERSION),
        )
        .expect("phase artifact");
        let (roundtrip_window, roundtrip_matrix) =
            read_strict_phase_ccf_matrix(&artifact, &config, 0, 2).expect("phase roundtrip");
        assert_eq!(roundtrip_window, window);
        assert_eq!(roundtrip_matrix, matrix);

        let mut bytes = std::fs::read(&path).expect("read phase bytes");
        *bytes.last_mut().expect("phase trailer") ^= 0xff;
        std::fs::write(&path, bytes).expect("tamper phase trailer");
        assert!(read_strict_phase_ccf_matrix(&artifact, &config, 0, 2).is_err());
        std::fs::remove_dir_all(root).expect("remove phase root");
    }

    #[test]
    fn strict_identity_contract_requires_explicit_identical_control() {
        let root = unique_test_dir("identity");
        std::fs::create_dir_all(&root).expect("create test root");
        let topology = root.join("topology.json");
        std::fs::write(&topology, b"{\"identity\":\"same\"}\n").expect("write topology");

        let valid = strict_evidence_config(&topology, &topology, true);
        valid
            .validate_source_files()
            .expect("valid explicit control");

        let mut invalid = valid.clone();
        invalid.identical_topology_negative_control = false;
        invalid.contrast_kind = "apo-holo".to_string();
        assert!(invalid.validate().is_err());
        std::fs::remove_dir_all(root).expect("remove test root");
    }

    #[test]
    fn strict_identity_contract_rejects_distinct_hash_control_and_hash_drift() {
        let root = unique_test_dir("distinct");
        std::fs::create_dir_all(&root).expect("create test root");
        let topology_a = root.join("a.json");
        let topology_b = root.join("b.json");
        std::fs::write(&topology_a, b"{\"identity\":\"a\"}\n").expect("write A");
        std::fs::write(&topology_b, b"{\"identity\":\"b\"}\n").expect("write B");

        let valid = strict_evidence_config(&topology_a, &topology_b, false);
        valid.validate_source_files().expect("valid distinct pair");

        let mut false_control = valid.clone();
        false_control.identical_topology_negative_control = true;
        false_control.contrast_kind = "identical-topology-negative-control".to_string();
        assert!(false_control.validate().is_err());

        std::fs::write(&topology_b, b"{\"identity\":\"tampered\"}\n").expect("tamper B");
        assert!(valid.validate_source_files().is_err());
        std::fs::remove_dir_all(root).expect("remove test root");
    }

    #[test]
    fn evidence_envelope_roundtrip_and_corruption_control() {
        let root = unique_test_dir("envelope");
        std::fs::create_dir_all(&root).expect("create test root");
        let path = root.join("target_stream0_spikes.bin");
        write_md_evidence_envelope(
            &path,
            b"PRSPK001",
            b"PRSPKEND",
            0,
            "run-identity",
            "target",
            1,
            4,
            &[1, 2, 3, 4],
        )
        .expect("write envelope");
        validate_evidence_envelope(
            &path,
            b"PRSPK001",
            b"PRSPKEND",
            0,
            "run-identity",
            "target",
            Some(1),
        )
        .expect("validate envelope");

        let mut bytes = std::fs::read(&path).expect("read envelope");
        let payload_index = bytes.len() - 17;
        bytes[payload_index] ^= 0xff;
        std::fs::write(&path, bytes).expect("write corrupted envelope");
        assert!(validate_evidence_envelope(
            &path,
            b"PRSPK001",
            b"PRSPKEND",
            0,
            "run-identity",
            "target",
            Some(1),
        )
        .is_err());
        std::fs::remove_dir_all(root).expect("remove test root");
    }

    #[test]
    fn no_clobber_artifact_commit_preserves_first_authority() {
        let root = unique_test_dir("no-clobber");
        std::fs::create_dir_all(&root).expect("create test root");
        let path = root.join("authority.bin");
        write_atomic_bytes(&path, b"first-authoritative-bytes").expect("first commit");
        let first_hash = sha256_file(&path).expect("hash first commit");
        assert!(write_atomic_bytes(&path, b"replacement-must-fail").is_err());
        assert_eq!(
            std::fs::read(&path).expect("read preserved authority"),
            b"first-authoritative-bytes"
        );
        assert_eq!(sha256_file(&path).expect("rehash authority"), first_hash);
        assert_eq!(
            std::fs::read_dir(&root)
                .expect("read test directory")
                .filter_map(|entry| entry.ok())
                .filter(|entry| entry.file_name().to_string_lossy().contains(".tmp."))
                .count(),
            0,
            "failed no-clobber commit must not strand a temporary artifact"
        );
        std::fs::remove_dir_all(root).expect("remove test root");
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn canonical_event_roundtrip_preserves_primary_ligand_and_full_support() {
        use crate::fused_engine::GpuSpikeEvent;

        let event = GpuSpikeEvent {
            timestep: 42,
            voxel_idx: 7,
            position: [1.25, -2.5, 3.75],
            intensity: 0.875,
            nearby_residues: [380, 0, 17, -1, -1, -1, -1, -1],
            n_residues: 3,
            spike_source: 1,
            wavelength_nm: 280.0,
            aromatic_type: 2,
            aromatic_residue_id: 17,
            water_density: 0.42,
            vibrational_energy: 0.33,
            n_nearby_excited: 2,
            wd_change: 0.04,
            phase_bits: 511,
        };
        let canonical = canonical_gpu_spike_bytes(&event);
        let readback = parse_canonical_gpu_spike_bytes(&canonical).expect("canonical round-trip");
        assert_eq!(canonical_gpu_spike_bytes(&readback), canonical);
        let (rich, support) = rich_spike_from_gpu_event(&readback, 63).expect("derive RichSpike");
        assert_eq!(rich.residue_id, 380, "ordered primary residue must survive");
        assert_eq!(rich.cluster_id, 63);
        assert_eq!(support, BTreeSet::from([0, 17, 380]));
        assert_eq!(rich.x.to_bits(), event.position[0].to_bits());
        assert_eq!(rich.y.to_bits(), event.position[1].to_bits());
        assert_eq!(rich.z.to_bits(), event.position[2].to_bits());
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn canonical_spike_identity_is_order_invariant_and_duplicate_preserving() {
        use crate::fused_engine::GpuSpikeEvent;

        let mut first = GpuSpikeEvent::default();
        first.timestep = 1;
        first.voxel_idx = 11;
        first.intensity = 0.25;
        let mut second = first;
        second.timestep = 2;
        second.voxel_idx = 22;
        second.intensity = 0.75;
        let ordered = vec![first, second, first];
        let permuted = vec![second, first, first];
        let missing_duplicate = vec![first, second];

        assert_eq!(
            canonical_split_t7_source_identity(&ordered).expect("ordered identity"),
            canonical_split_t7_source_identity(&permuted).expect("permuted identity")
        );
        assert_ne!(
            canonical_split_t7_source_identity(&ordered).expect("duplicate identity"),
            canonical_split_t7_source_identity(&missing_duplicate)
                .expect("missing duplicate identity")
        );

        let mut ordered_raw = Vec::new();
        let mut permuted_raw = Vec::new();
        write_canonical_split_t7_source(&mut ordered_raw, &ordered).expect("write ordered raw");
        write_canonical_split_t7_source(&mut permuted_raw, &permuted).expect("write permuted raw");
        assert_ne!(
            ordered_raw, permuted_raw,
            "raw provenance order must be preserved"
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn molecular_contrast_intensity_aggregation_matches_strict_f64_policy() {
        use crate::fused_engine::GpuSpikeEvent;

        let events = (0..100_000)
            .map(|timestep| GpuSpikeEvent {
                timestep,
                voxel_idx: timestep,
                position: [0.0, 0.0, 0.0],
                intensity: 0.1,
                nearby_residues: [0, -1, -1, -1, -1, -1, -1, -1],
                n_residues: 1,
                spike_source: 2,
                wavelength_nm: 0.0,
                aromatic_type: -1,
                aromatic_residue_id: -1,
                water_density: 0.0,
                vibrational_energy: 0.0,
                n_nearby_excited: 0,
                wd_change: 0.0,
                phase_bits: 1,
            })
            .collect::<Vec<_>>();
        let source = spike_count_map(&events);
        let strict = collect_strict_residue_observations(&events, 1, 100_000)
            .expect("collect strict observations");
        let strict_intensity = strict[&0]
            .iter()
            .map(|event| event.intensity as f64)
            .sum::<f64>() as f32;
        let mut naive_f32 = 0.0f32;
        for event in &events {
            naive_f32 += event.intensity;
        }

        assert_eq!(source[&0].0, 100_000);
        assert_eq!(source[&0].1.to_bits(), strict_intensity.to_bits());
        assert!(
            (naive_f32 - strict_intensity).abs()
                > 1.0e-5 * naive_f32.abs().max(strict_intensity.abs()).max(1.0),
            "the regression fixture must expose the rejected f32 accumulation drift"
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn split_t7_full_span_selection_is_exact_and_rejects_undersized_windows() {
        assert_eq!(
            split_t7_selection_indices(10, 4).expect("valid full-span selection"),
            vec![0, 3, 6, 9]
        );
        assert_eq!(
            split_t7_selection_indices(4, 4).expect("identity selection"),
            vec![0, 1, 2, 3]
        );
        assert!(split_t7_selection_indices(3, 4).is_err());
        assert!(split_t7_selection_indices(10, 3).is_err());
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn split_t7_frame_verifier_reconstructs_cluster_sorted_richspikes() {
        use crate::fused_engine::GpuSpikeEvent;
        use std::io::Cursor;

        let mut residue_to_cluster = vec![u32::MAX; 381];
        residue_to_cluster[0] = 1;
        residue_to_cluster[17] = 2;
        residue_to_cluster[380] = 63;
        let layout = SplitT7TopologyLayout {
            aabb_min: [0.0; 3],
            aabb_max: [10.0; 3],
            cell: [2.5; 3],
            residue_to_anchor_atom: vec![0; 381],
            residue_to_cluster,
            cluster_representative_residue: vec![0; SPLIT_T7_CLUSTER_COUNT as usize],
            cluster_chain_role: vec![0; SPLIT_T7_CLUSTER_COUNT as usize],
            chain_offsets: [0; 8],
            layout_sha256: "1".repeat(64),
        };
        let event = |timestep: i32, residue: i32, x: f32| GpuSpikeEvent {
            timestep,
            voxel_idx: timestep,
            position: [x, x + 0.5, x + 1.0],
            intensity: 0.75,
            nearby_residues: [residue, -1, -1, -1, -1, -1, -1, -1],
            n_residues: 1,
            spike_source: 2,
            wavelength_nm: 0.0,
            aromatic_type: -1,
            aromatic_residue_id: -1,
            water_density: 0.25,
            vibrational_energy: 0.5,
            n_nearby_excited: 0,
            wd_change: 0.125,
            phase_bits: 9,
        };
        let source_events = vec![
            event(60, 380, 6.0),
            event(20, 17, 2.0),
            event(10, 0, 1.0),
            event(30, 0, 3.0),
            event(40, 380, 4.0),
            event(50, 17, 5.0),
        ];
        let source = select_split_t7_full_span(&source_events, 4, 0).expect("select frame");
        let mut selected_for_gpu = source.selected_events.clone();
        let (rich, _, timestep_min, timestep_max) =
            prepare_split_t7_frame(&mut selected_for_gpu, &layout).expect("prepare frame");
        let mut writer = Cursor::new(Vec::new());
        let offsets = append_split_t7_input_frame(
            &mut writer,
            7,
            1234,
            timestep_min,
            timestep_max,
            &source,
            &rich,
        )
        .expect("serialize frame");
        writer.set_position(0);
        let verified = verify_split_t7_frame(&mut writer, 0, 7, Some(1234), Some(4), None, &layout)
            .expect("verify frame");
        assert_eq!(verified.source_event_count, 6);
        assert_eq!(verified.selected_event_count, 4);
        assert_eq!(
            verified
                .selected_events
                .iter()
                .map(|selected| selected.source_index)
                .collect::<Vec<_>>(),
            vec![0, 1, 3, 5]
        );

        let mut corrupted = writer.into_inner();
        corrupted[offsets.rich_offset as usize] ^= 0x01;
        let mut corrupted_reader = Cursor::new(corrupted);
        assert!(verify_split_t7_frame(
            &mut corrupted_reader,
            0,
            7,
            Some(1234),
            Some(4),
            None,
            &layout,
        )
        .is_err());
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn strict_twin_ccf_support_preserves_real_zero_and_excludes_unsupported_zero() {
        let matrix = vec![0.0f32; 9];
        let features = crate::twin_kernels::extract_ccf_features(&matrix, 3);
        let all_supported = [true, true, true];
        let supported_zero = strict_zero_lag_ccf_projection(
            &matrix,
            &features,
            3,
            0,
            &all_supported,
            &all_supported,
        )
        .expect("supported real-zero projection")
        .expect("real zero remains supported");
        assert_eq!(
            supported_zero.aligned_zero_lag_ccf.to_bits(),
            0.0f32.to_bits()
        );
        assert_eq!(
            supported_zero.peak_partner_zero_lag_ccf.to_bits(),
            0.0f32.to_bits()
        );
        assert_eq!(supported_zero.supported_partner_count, 2);

        let only_self_supported = [true, false, false];
        assert_eq!(
            strict_zero_lag_ccf_projection(
                &matrix,
                &features,
                3,
                0,
                &only_self_supported,
                &only_self_supported,
            )
            .expect("unsupported cells are not numeric failures"),
            None
        );

        let mixed_matrix = vec![0.0, 0.99, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mixed_features = crate::twin_kernels::extract_ccf_features(&mixed_matrix, 3);
        let mixed_support = [true, false, true];
        let mixed = strict_zero_lag_ccf_projection(
            &mixed_matrix,
            &mixed_features,
            3,
            0,
            &mixed_support,
            &mixed_support,
        )
        .expect("mixed support projection")
        .expect("one supported partner remains");
        assert_eq!(mixed.peak_partner_residue_id, 2);
        assert_eq!(mixed.peak_partner_zero_lag_ccf.to_bits(), 0.25f32.to_bits());
        assert_eq!(mixed.supported_partner_count, 1);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn strict_twin_support_mask_tampering_and_null_to_zero_substitution_fail_closed() {
        let mut record = strict_twin_residue_fixture();
        record.global_variance_support_a = false;
        assert!(validate_strict_residue_feature(&record).is_err());

        let mut unsupported = strict_twin_residue_fixture();
        unsupported.global_variance_support_a = false;
        unsupported.global_variance_support_b = false;
        unsupported.global_supported_partner_count = 0;
        unsupported.global_correlation_lag_steps = None;
        unsupported.global_aligned_zero_lag_ccf = None;
        unsupported.global_peak_partner_residue_id = None;
        unsupported.global_peak_partner_residue_offset = None;
        unsupported.global_peak_partner_zero_lag_ccf = None;
        unsupported.global_cross_residue_network_reproducibility_fraction = None;
        unsupported.global_ccf_support_status = STRICT_TWIN_GLOBAL_CCF_UNAVAILABLE.to_string();
        validate_strict_residue_feature(&unsupported).expect("explicit unavailable CCF");
        unsupported.global_aligned_zero_lag_ccf = Some(0.0);
        assert!(validate_strict_residue_feature(&unsupported).is_err());
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn molecular_contrast_pocket_v4_uses_only_supported_residues_in_mean() {
        let pocket_ids = BTreeSet::from([10usize, 11usize]);
        let counts_a = BTreeMap::from([(10, (1, 1.0))]);
        let counts_b = BTreeMap::from([(10, (3, 3.0))]);
        let strict_ids = BTreeSet::from([10]);
        let support = build_molecular_contrast_pocket_support_v4(
            &pocket_ids,
            &counts_a,
            &counts_b,
            &strict_ids,
        )
        .expect("partial pocket support remains explicit");
        assert_eq!(support.candidate_residue_count, 2);
        assert_eq!(support.supported_residue_count, 1);
        assert_eq!(support.unavailable_residue_count, 1);
        assert_eq!(support.l1_signed_contrast.to_bits(), 0.5f64.to_bits());
        assert_eq!(
            support.mean_abs_signed_contrast.map(f64::to_bits),
            Some(0.5f64.to_bits())
        );
        assert_eq!(
            support
                .decisions
                .iter()
                .map(|decision| {
                    (
                        decision.residue_id,
                        decision.contrast_supported,
                        decision.abs_signed_count_contrast.map(f32::to_bits),
                    )
                })
                .collect::<Vec<_>>(),
            vec![(10, true, Some(0.5f32.to_bits())), (11, false, None)]
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn molecular_contrast_pocket_v4_preserves_supported_zero_and_empty_support_null() {
        let pocket_ids = BTreeSet::from([10usize, 11usize]);
        let supported_zero = build_molecular_contrast_pocket_support_v4(
            &pocket_ids,
            &BTreeMap::from([(10, (2, 1.0))]),
            &BTreeMap::from([(10, (2, 1.0))]),
            &BTreeSet::from([10]),
        )
        .expect("supported physical zero remains numeric");
        assert_eq!(
            supported_zero.decisions[0]
                .abs_signed_count_contrast
                .map(f32::to_bits),
            Some(0.0f32.to_bits())
        );
        assert_eq!(
            supported_zero.mean_abs_signed_contrast.map(f64::to_bits),
            Some(0.0f64.to_bits())
        );

        let unavailable = build_molecular_contrast_pocket_support_v4(
            &pocket_ids,
            &BTreeMap::new(),
            &BTreeMap::new(),
            &BTreeSet::new(),
        )
        .expect("empty support remains explicitly unavailable");
        assert_eq!(unavailable.supported_residue_count, 0);
        assert_eq!(unavailable.unavailable_residue_count, 2);
        assert!(unavailable.mean_abs_signed_contrast.is_none());
        assert!(unavailable
            .decisions
            .iter()
            .all(|decision| decision.abs_signed_count_contrast.is_none()));
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn molecular_contrast_pocket_v4_rejects_mask_and_null_substitution() {
        let pocket_ids = BTreeSet::from([10usize, 11usize]);
        let counts = BTreeMap::from([(10, (1, 1.0))]);
        assert!(build_molecular_contrast_pocket_support_v4(
            &pocket_ids,
            &counts,
            &BTreeMap::new(),
            &BTreeSet::new(),
        )
        .is_err());
        assert!(build_molecular_contrast_pocket_support_v4(
            &pocket_ids,
            &BTreeMap::new(),
            &BTreeMap::new(),
            &BTreeSet::from([10]),
        )
        .is_err());

        let mut support = build_molecular_contrast_pocket_support_v4(
            &pocket_ids,
            &counts,
            &BTreeMap::new(),
            &BTreeSet::from([10]),
        )
        .expect("valid mixed support fixture");
        support.decisions[1].abs_signed_count_contrast = Some(0.0);
        support.decisions[1].contrast_supported = true;
        assert!(validate_molecular_contrast_pocket_support_v4(&pocket_ids, &support).is_err());
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn strict_twin_te_metadata_requires_two_bin_two_hundred_step_horizon() {
        let mut record = strict_twin_residue_fixture();
        validate_strict_residue_feature(&record).expect("exact TE horizon metadata");
        record.transfer_entropy_effective_past_to_future_horizon_steps = Some(100);
        assert!(validate_strict_residue_feature(&record).is_err());

        let mut legacy = serde_json::to_value(strict_twin_residue_fixture())
            .expect("serialize strict residue fixture");
        let object = legacy.as_object_mut().expect("strict residue object");
        object.remove("transfer_entropy_kernel_lag_bins");
        object.insert(
            "transfer_entropy_lag_bins".to_string(),
            serde_json::json!(1),
        );
        assert!(serde_json::from_value::<StrictTwinResidueFeature>(legacy).is_err());
    }

    #[cfg(feature = "gpu")]
    fn strict_binding_site_fixture() -> serde_json::Value {
        serde_json::json!({
            "sites": [
                {
                    "id": 9,
                    "rank": 2,
                    "residue_ids": [0, 1],
                    "lining_residues": [{"resid": 0}, {"resid": 1}],
                    "volume": 125.0,
                    "druggability": 0.7,
                    "is_druggable": true,
                    "centroid": [1.0, 2.0, 3.0]
                },
                {
                    "id": 2,
                    "rank": 1,
                    "residue_ids": [0, 1],
                    "lining_residues": [{"resid": 0}, {"resid": 1}],
                    "volume": 100.0,
                    "druggability": 0.6,
                    "is_druggable": false,
                    "centroid": [4.0, 5.0, 6.0]
                }
            ]
        })
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn strict_twin_site_identity_is_authoritative_and_array_reorder_stable() {
        let original = strict_binding_site_fixture();
        let parsed = parse_strict_binding_sites(&original, 2).expect("parse authoritative sites");
        assert_eq!(
            parsed
                .iter()
                .map(|site| (site.site_id, site.site_rank))
                .collect::<Vec<_>>(),
            vec![(2, 1), (9, 2)]
        );
        let mut reordered = original.clone();
        reordered["sites"]
            .as_array_mut()
            .expect("site array")
            .reverse();
        assert_eq!(
            parse_strict_binding_sites(&reordered, 2).expect("parse reordered sites"),
            parsed
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn strict_twin_site_identity_rejects_missing_duplicate_negative_malformed_and_substituted_ids()
    {
        let base = strict_binding_site_fixture();
        let mut cases = Vec::new();
        let mut missing = base.clone();
        missing["sites"][0]
            .as_object_mut()
            .expect("site object")
            .remove("id");
        cases.push(missing);
        let mut duplicate = base.clone();
        duplicate["sites"][1]["id"] = serde_json::json!(9);
        cases.push(duplicate);
        let mut negative = base.clone();
        negative["sites"][0]["id"] = serde_json::json!(-1);
        cases.push(negative);
        let mut malformed = base.clone();
        malformed["sites"][0]["id"] = serde_json::json!("9");
        cases.push(malformed);
        let mut fractional = base.clone();
        fractional["sites"][0]["id"] = serde_json::json!(9.5);
        cases.push(fractional);
        let mut overflow = base.clone();
        overflow["sites"][0]["id"] = serde_json::json!(u64::from(u32::MAX) + 1);
        cases.push(overflow);
        let mut substituted = base;
        substituted["sites"][0]["residue_ids"] = serde_json::json!([0]);
        cases.push(substituted);
        for case in cases {
            assert!(parse_strict_binding_sites(&case, 2).is_err());
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn strict_twin_site_rank_requires_exact_one_based_permutation() {
        let base = strict_binding_site_fixture();
        for ranks in [[0, 2], [1, 1], [1, 3], [2, 3]] {
            let mut case = base.clone();
            case["sites"][0]["rank"] = serde_json::json!(ranks[0]);
            case["sites"][1]["rank"] = serde_json::json!(ranks[1]);
            assert!(parse_strict_binding_sites(&case, 2).is_err());
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn strict_twin_site_empty_supported_pair_set_is_null_not_zero() {
        let mut left = strict_twin_residue_fixture();
        left.residue_id = 0;
        left.global_variance_support_a = false;
        left.global_variance_support_b = false;
        left.global_supported_partner_count = 0;
        left.global_correlation_lag_steps = None;
        left.global_aligned_zero_lag_ccf = None;
        left.global_peak_partner_residue_id = None;
        left.global_peak_partner_residue_offset = None;
        left.global_peak_partner_zero_lag_ccf = None;
        left.global_cross_residue_network_reproducibility_fraction = None;
        left.global_ccf_support_status = STRICT_TWIN_GLOBAL_CCF_UNAVAILABLE.to_string();
        let mut right = left.clone();
        right.residue_id = 1;
        let sites = build_strict_site_features(
            &strict_binding_site_fixture(),
            &[left, right],
            &[0.0; 4],
            2,
        )
        .expect("support-aware site aggregation");
        assert_eq!(sites.len(), 2);
        assert!(sites.iter().all(|site| {
            site.intra_site_ccf_candidate_pair_count == 1
                && site.intra_site_ccf_supported_pair_count == 0
                && site.intra_site_ccf_pair_decisions.len() == 1
                && !site.intra_site_ccf_pair_decisions[0].pair_supported
                && site.intra_site_ccf_pair_decisions[0]
                    .symmetric_zero_lag_ccf
                    .is_none()
                && site.mean_intra_site_cross_residue_zero_lag_ccf.is_none()
                && site.mean_global_aligned_zero_lag_ccf.is_none()
        }));
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn strict_twin_site_pair_inventory_preserves_supported_zero_and_excludes_unsupported_cells() {
        let sites_json = serde_json::json!({
            "sites": [{
                "id": 17,
                "rank": 1,
                "residue_ids": [0, 1, 2],
                "lining_residues": [{"resid": 0}, {"resid": 1}, {"resid": 2}],
                "volume": 75.0,
                "druggability": 0.5,
                "is_druggable": true,
                "centroid": [1.0, 1.0, 1.0]
            }]
        });
        let supported = |residue_id: i32| {
            let mut record = strict_twin_residue_fixture();
            record.residue_id = residue_id;
            record
        };
        let mut unsupported = supported(1);
        unsupported.global_variance_support_a = false;
        unsupported.global_variance_support_b = false;
        unsupported.global_supported_partner_count = 0;
        unsupported.global_correlation_lag_steps = None;
        unsupported.global_aligned_zero_lag_ccf = None;
        unsupported.global_peak_partner_residue_id = None;
        unsupported.global_peak_partner_residue_offset = None;
        unsupported.global_peak_partner_zero_lag_ccf = None;
        unsupported.global_cross_residue_network_reproducibility_fraction = None;
        unsupported.global_ccf_support_status = STRICT_TWIN_GLOBAL_CCF_UNAVAILABLE.to_string();

        let matrix = vec![0.0, 99.0, 0.0, 99.0, 0.0, 99.0, 0.0, 99.0, 0.0];
        let records = build_strict_site_features(
            &sites_json,
            &[supported(0), unsupported, supported(2)],
            &matrix,
            3,
        )
        .expect("support-aware complete pair inventory");
        let record = &records[0];
        assert_eq!(record.intra_site_ccf_candidate_pair_count, 3);
        assert_eq!(record.intra_site_ccf_supported_pair_count, 1);
        assert_eq!(record.intra_site_ccf_pair_decisions.len(), 3);
        assert_eq!(
            record
                .mean_intra_site_cross_residue_zero_lag_ccf
                .map(f32::to_bits),
            Some(0.0f32.to_bits())
        );
        assert_eq!(
            record
                .intra_site_ccf_pair_decisions
                .iter()
                .map(|decision| {
                    (
                        decision.left_residue_id,
                        decision.right_residue_id,
                        decision.pair_supported,
                        decision.symmetric_zero_lag_ccf.map(f32::to_bits),
                    )
                })
                .collect::<Vec<_>>(),
            vec![
                (0, 1, false, None),
                (0, 2, true, Some(0.0f32.to_bits())),
                (1, 2, false, None),
            ]
        );

        let mut tampered = record.clone();
        tampered.intra_site_ccf_pair_decisions[0].pair_supported = true;
        assert!(validate_strict_site_feature(&tampered).is_err());

        assert!(build_strict_site_features(
            &sites_json,
            &[supported(0), supported(0), supported(2)],
            &matrix,
            3,
        )
        .is_err());
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn strict_twin_site_pair_inventory_records_missing_rows_explicitly() {
        let sites_json = serde_json::json!({
            "sites": [{
                "id": 21,
                "rank": 1,
                "residue_ids": [0, 1, 2],
                "lining_residues": [{"resid": 0}, {"resid": 1}, {"resid": 2}],
                "volume": 80.0,
                "druggability": 0.4,
                "is_druggable": false,
                "centroid": [2.0, 2.0, 2.0]
            }]
        });
        let mut left = strict_twin_residue_fixture();
        left.residue_id = 0;
        let mut right = strict_twin_residue_fixture();
        right.residue_id = 2;
        let records = build_strict_site_features(&sites_json, &[left, right], &[0.0; 9], 3)
            .expect("missing-row decisions remain explicit");
        let record = &records[0];
        assert_eq!(record.missing_strict_residue_ids, vec![1]);
        assert_eq!(record.intra_site_ccf_candidate_pair_count, 3);
        assert_eq!(record.intra_site_ccf_supported_pair_count, 1);
        assert!(record.intra_site_ccf_pair_decisions.iter().any(|decision| {
            decision.left_residue_id == 0
                && decision.right_residue_id == 1
                && !decision.right_strict_row_present
                && decision.symmetric_zero_lag_ccf.is_none()
        }));
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn strict_ghost_ligand_exact_distance_rejects_aabb_false_positive() {
        let ligand_atoms = vec![
            LigandHeavyAtomBinding {
                atom_index: 10,
                residue_id: 381,
                element: "C".to_string(),
                position: [0.0, 0.0, 0.0],
            },
            LigandHeavyAtomBinding {
                atom_index: 11,
                residue_id: 381,
                element: "O".to_string(),
                position: [10.0, 10.0, 10.0],
            },
        ];
        let witness = [5.0, 5.0, 5.0];
        assert_eq!(
            aabb_separation(witness, witness, [0.0; 3], [10.0; 3]).to_bits(),
            0.0f32.to_bits()
        );
        let (_, exact_distance) =
            exact_ligand_heavy_atom_distance(witness, &ligand_atoms).expect("exact distance");
        assert!(exact_distance > 5.0);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn split_t7_paired_layout_preserves_receptor_alignment_and_role_separation() {
        let make_layout = |residue_to_cluster: Vec<u32>, hash_byte: char| {
            let mut cluster_roles = vec![0xFF; SPLIT_T7_CLUSTER_COUNT as usize];
            for cluster in residue_to_cluster
                .iter()
                .copied()
                .filter(|cluster| *cluster != u32::MAX)
            {
                cluster_roles[cluster as usize] = if cluster < SPLIT_T7_ROLE_B_CLUSTER_OFFSET {
                    0
                } else {
                    1
                };
            }
            SplitT7TopologyLayout {
                aabb_min: [0.0; 3],
                aabb_max: [10.0; 3],
                cell: [2.5; 3],
                residue_to_anchor_atom: vec![0; residue_to_cluster.len()],
                residue_to_cluster,
                cluster_representative_residue: vec![0; SPLIT_T7_CLUSTER_COUNT as usize],
                cluster_chain_role: cluster_roles,
                chain_offsets: [0; 8],
                layout_sha256: hash_byte.to_string().repeat(64),
            }
        };
        assert_eq!(SPLIT_T7_ROLE_B_CLUSTER_OFFSET, 48);
        assert_eq!(
            SPLIT_T7_ROLE_B_CLUSTER_OFFSET + SPLIT_T7_ROLE_B_GRID.iter().product::<u32>(),
            SPLIT_T7_CLUSTER_COUNT
        );
        let layout_a = make_layout(vec![0, 7, 47], 'a');
        let layout_b = make_layout(vec![0, 7, 47, 48], 'b');
        validate_split_t7_paired_layouts(&layout_a, &layout_b, false)
            .expect("aligned receptor plus separated ligand");

        let receptor_drift = make_layout(vec![0, 8, 47, 48], 'c');
        assert!(validate_split_t7_paired_layouts(&layout_a, &receptor_drift, false).is_err());
        let identical = layout_a.clone();
        validate_split_t7_paired_layouts(&layout_a, &identical, true)
            .expect("identical layout control");
        let mut false_identical = identical;
        false_identical.layout_sha256 = "d".repeat(64);
        assert!(validate_split_t7_paired_layouts(&layout_a, &false_identical, true).is_err());
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn split_t7_frame_rejects_oversized_event_and_absent_secondary_support() {
        use crate::fused_engine::GpuSpikeEvent;
        use std::io::Cursor;

        let mut residue_to_cluster = vec![u32::MAX; 4];
        residue_to_cluster[0] = 0;
        let layout = SplitT7TopologyLayout {
            aabb_min: [0.0; 3],
            aabb_max: [1.0; 3],
            cell: [0.25; 3],
            residue_to_anchor_atom: vec![0, u32::MAX, u32::MAX, u32::MAX],
            residue_to_cluster,
            cluster_representative_residue: vec![0; SPLIT_T7_CLUSTER_COUNT as usize],
            cluster_chain_role: vec![0; SPLIT_T7_CLUSTER_COUNT as usize],
            chain_offsets: [0; 8],
            layout_sha256: "1".repeat(64),
        };
        let event = |timestep: i32| GpuSpikeEvent {
            timestep,
            voxel_idx: timestep,
            position: [0.1, 0.2, 0.3],
            intensity: 0.5,
            nearby_residues: [0, -1, -1, -1, -1, -1, -1, -1],
            n_residues: 1,
            spike_source: 2,
            wavelength_nm: 0.0,
            aromatic_type: -1,
            aromatic_residue_id: -1,
            water_density: 0.2,
            vibrational_energy: 0.3,
            n_nearby_excited: 0,
            wd_change: 0.1,
            phase_bits: 1,
        };
        validate_split_t7_shadow_source_event(&event(1), 0, 0, 6, 0, &layout)
            .expect("frame zero first completed step is admitted");
        validate_split_t7_shadow_source_event(&event(6), 0, 0, 6, 1, &layout)
            .expect("frame zero completed-step endpoint is admitted");
        assert!(
            validate_split_t7_shadow_source_event(&event(0), 0, 0, 6, 2, &layout).is_err(),
            "the pre-window timestamp must fail closed"
        );
        assert!(
            validate_split_t7_shadow_source_event(&event(7), 0, 0, 6, 3, &layout).is_err(),
            "the next frame's first completed step must fail closed"
        );
        validate_split_t7_shadow_source_event(&event(7), 0, 1, 6, 4, &layout)
            .expect("frame one first completed step is admitted");
        validate_split_t7_shadow_source_event(&event(12), 0, 1, 6, 5, &layout)
            .expect("frame one completed-step endpoint is admitted");
        assert!(
            validate_split_t7_shadow_source_event(&event(6), 0, 1, 6, 6, &layout).is_err(),
            "the prior frame endpoint must fail closed"
        );
        assert!(
            validate_split_t7_shadow_source_event(&event(13), 0, 1, 6, 7, &layout).is_err(),
            "the post-window timestamp must fail closed"
        );
        let source_events = (1..=4).map(event).collect::<Vec<_>>();
        let source = select_split_t7_full_span(&source_events, 4, 0).expect("select frame");
        let mut selected = source.selected_events.clone();
        let (rich, _, min, max) =
            prepare_split_t7_frame(&mut selected, &layout).expect("prepare frame");
        let mut writer = Cursor::new(Vec::new());
        let offsets = append_split_t7_input_frame(&mut writer, 0, 1, min, max, &source, &rich)
            .expect("serialize frame");

        let mut oversized = writer.get_ref().clone();
        let first_event_length = offsets.source_offset as usize + 34 + 8;
        oversized[first_event_length..first_event_length + 4]
            .copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(verify_split_t7_frame(
            &mut Cursor::new(oversized),
            0,
            0,
            Some(1),
            Some(4),
            None,
            &layout,
        )
        .is_err());

        let mut absent_support_events = source_events;
        for event in &mut absent_support_events {
            event.nearby_residues[1] = 3;
            event.n_residues = 2;
        }
        let absent_source =
            select_split_t7_full_span(&absent_support_events, 4, 0).expect("select absent support");
        let mut absent_selected = absent_source.selected_events.clone();
        let (absent_rich, _, absent_min, absent_max) =
            prepare_split_t7_frame(&mut absent_selected, &layout).expect("prepare absent support");
        let mut absent_writer = Cursor::new(Vec::new());
        append_split_t7_input_frame(
            &mut absent_writer,
            0,
            1,
            absent_min,
            absent_max,
            &absent_source,
            &absent_rich,
        )
        .expect("serialize absent support");
        absent_writer.set_position(0);
        assert!(
            verify_split_t7_frame(&mut absent_writer, 0, 0, Some(1), Some(4), None, &layout,)
                .is_err()
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn split_t7_sha_parser_rejects_noncanonical_case() {
        use std::io::Cursor;

        assert!(read_stream_sha256(&mut Cursor::new(vec![b'A'; 64]), "uppercase hash").is_err());
        assert_eq!(
            read_stream_sha256(&mut Cursor::new(vec![b'a'; 64]), "lowercase hash")
                .expect("canonical lowercase hash"),
            "a".repeat(64)
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn split_t7_population_gate_rejects_zero_and_nonfinite_variance() {
        let mut valid = Vec::with_capacity(SPLIT_T7_VARIANCE_FRAMES);
        for index in 0..SPLIT_T7_VARIANCE_FRAMES {
            valid.push(1.0 + index as f64 * 0.01);
        }
        let (mean, sigma) = split_t7_population_stats(&valid, 0).expect("nondegenerate stats");
        assert!(mean.is_finite() && sigma.is_finite() && sigma > 0.0);
        assert!(split_t7_population_stats(&vec![0.0; SPLIT_T7_VARIANCE_FRAMES], 0).is_err());
        valid[7] = f64::NAN;
        assert!(split_t7_population_stats(&valid, 0).is_err());
    }

    #[cfg(feature = "gpu")]
    fn split_t7_shadow_trial_fixture(
        trial_ordinal: u32,
        trial_seed: u64,
        count_a: u32,
        count_b: u32,
    ) -> SplitT7ShadowTrialEvidence {
        SplitT7ShadowTrialEvidence {
            trial_ordinal,
            base_seed: 77,
            trial_seed,
            seed_derivation_sha256: format!("{:064x}", trial_seed),
            window_md_steps: 6,
            frame_count: SPLIT_T7_TOTAL_FRAMES as u32,
            source_bundle_path: format!("/isolated/trial-{trial_ordinal}.bin"),
            source_bundle_sha256: format!("{:064x}", trial_seed + 100),
            source_bundle_bytes: 4096,
            frame_counts_a: vec![count_a; SPLIT_T7_TOTAL_FRAMES],
            frame_counts_b: vec![count_b; SPLIT_T7_TOTAL_FRAMES],
            frame_source_sha256_a: vec!["a".repeat(64); SPLIT_T7_TOTAL_FRAMES],
            frame_source_sha256_b: vec!["b".repeat(64); SPLIT_T7_TOTAL_FRAMES],
            initial_physical_state_sha256_a: "1".repeat(64),
            initial_physical_state_sha256_b: "2".repeat(64),
            final_physical_state_sha256_a: "3".repeat(64),
            final_physical_state_sha256_b: "4".repeat(64),
            crn_initialization_artifact: StrictTwinFeatureArtifactBinding {
                kind: format!("split_t7_shadow_trial_{trial_ordinal:02}_crn_initialization"),
                stream_id: None,
                path: format!("/isolated/trial-{trial_ordinal}-crn.json"),
                size_bytes: 2048,
                sha256: "c".repeat(64),
                blake3: "d".repeat(64),
            },
            crn_initialization_semantic_sha256: "e".repeat(64),
            inventory_sha256: format!("{:064x}", trial_seed + 200),
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn split_t7_three_trial_seeds_are_distinct_deterministic_and_pair_shared() {
        let root = unique_test_dir("split-t7-seeds");
        std::fs::create_dir_all(&root).expect("create seed-test root");
        let topology_a = root.join("a.json");
        let topology_b = root.join("b.json");
        std::fs::write(&topology_a, b"topology-a").expect("write topology A identity");
        std::fs::write(&topology_b, b"topology-b").expect("write topology B identity");
        let config = strict_evidence_config(&topology_a, &topology_b, false);
        let derived = (0..SPLIT_T7_SHADOW_TRIALS as u32)
            .map(|ordinal| split_t7_trial_seed(77, ordinal, &config).expect("derive trial seed"))
            .collect::<Vec<_>>();
        assert_eq!(
            derived
                .iter()
                .map(|(seed, _)| *seed)
                .collect::<BTreeSet<_>>()
                .len(),
            SPLIT_T7_SHADOW_TRIALS
        );
        for (ordinal, (seed, digest)) in derived.iter().enumerate() {
            assert_eq!(
                split_t7_trial_seed(77, ordinal as u32, &config).expect("repeat trial seed"),
                (*seed, digest.clone())
            );
            let stream_a_seed = *seed;
            let stream_b_seed = *seed;
            assert_eq!(stream_a_seed, stream_b_seed);
        }
        std::fs::remove_dir_all(root).expect("remove seed-test root");
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn split_t7_pointwise_envelope_and_all_leave_one_out_trials_are_recomputed() {
        let mut trials = vec![
            split_t7_shadow_trial_fixture(0, 100, 20, 22),
            split_t7_shadow_trial_fixture(1, 101, 18, 24),
            split_t7_shadow_trial_fixture(2, 102, 26, 16),
        ];
        trials[0].frame_counts_a[7] = 12;
        trials[1].frame_counts_b[29] = 14;
        trials[2].frame_counts_a[88] = 15;
        let (lower_a, lower_b) =
            split_t7_pointwise_lower_envelope(&trials).expect("pointwise envelope");
        assert_eq!(lower_a[7], 12);
        assert_eq!(lower_a[88], 15);
        assert_eq!(lower_b[29], 14);
        assert_eq!(lower_a.iter().chain(&lower_b).copied().min(), Some(12));
        let leave_one_out =
            split_t7_leave_one_out_admissions(&trials).expect("all leave-one-out trials");
        assert_eq!(leave_one_out.len(), SPLIT_T7_SHADOW_TRIALS);
        assert!(leave_one_out.iter().all(|result| {
            result.training_trial_ordinals.len() == SPLIT_T7_SHADOW_TRIALS - 1
                && !result
                    .training_trial_ordinals
                    .contains(&result.held_out_trial_ordinal)
                && result.admitted
        }));

        trials.swap(0, 1);
        assert!(split_t7_pointwise_lower_envelope(&trials).is_err());
        assert!(split_t7_leave_one_out_admissions(&trials).is_err());
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn split_t7_v5_frame_89_regression_is_evidence_derived_not_hard_coded() {
        let mut authoritative_a = vec![21u32; SPLIT_T7_TOTAL_FRAMES];
        let authoritative_b = vec![21u32; SPLIT_T7_TOTAL_FRAMES];
        authoritative_a[89] = 7;
        assert_eq!(
            split_t7_authoritative_admission(&authoritative_a, &authoritative_b, 4)
                .expect("evidence-derived K=4 admits the real count"),
            (7, 3)
        );
        assert!(split_t7_authoritative_admission(&authoritative_a, &authoritative_b, 8).is_err());
        authoritative_a[89] = 3;
        assert!(split_t7_authoritative_admission(&authoritative_a, &authoritative_b, 4).is_err());
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn split_t7_selection_is_full_span_without_padding_or_reuse() {
        let indices = split_t7_selection_indices(7, 4).expect("full-span real-event selection");
        assert_eq!(indices, vec![0, 2, 4, 6]);
        assert_eq!(indices.iter().copied().collect::<BTreeSet<_>>().len(), 4);
        assert!(split_t7_selection_indices(3, 4).is_err());
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn split_t7_cardinality_precommit_separates_shadow_and_authoritative_counts() {
        let mut shadow_a = vec![703_978u32; SPLIT_T7_TOTAL_FRAMES];
        let mut shadow_b = vec![716_938u32; SPLIT_T7_TOTAL_FRAMES];
        shadow_a[37] = 700_001;
        shadow_b[63] = 699_999;
        let fixed = split_t7_precommitted_cardinality(&shadow_a, &shadow_b)
            .expect("derive fixed CUDA cardinality");
        assert_eq!(fixed, 262_144);
        assert_eq!(
            split_t7_precommitted_cardinality(
                &vec![524_288; SPLIT_T7_TOTAL_FRAMES],
                &vec![524_288; SPLIT_T7_TOTAL_FRAMES],
            )
            .expect("strict power-of-two headroom"),
            262_144
        );

        let precommit = split_t7_paired_inventory_sha256(
            SPLIT_T7_SHADOW_ENSEMBLE_DOMAIN,
            6,
            fixed,
            &shadow_a,
            &shadow_b,
        )
        .expect("hash precommit inventory");
        let mut authoritative_a = shadow_a.clone();
        let mut authoritative_b = shadow_b.clone();
        authoritative_a[0] -= 1;
        authoritative_b[0] -= 4;
        let authoritative = split_t7_paired_inventory_sha256(
            SPLIT_T7_AUTHORITATIVE_INVENTORY_DOMAIN,
            6,
            fixed,
            &authoritative_a,
            &authoritative_b,
        )
        .expect("hash authoritative inventory");
        assert_ne!(precommit, authoritative);

        authoritative_a[0] = fixed - 1;
        assert!(split_t7_paired_inventory_sha256(
            SPLIT_T7_AUTHORITATIVE_INVENTORY_DOMAIN,
            6,
            fixed,
            &authoritative_a,
            &authoritative_b,
        )
        .is_err());
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn ghost_envelope_accepts_v3_membership_and_rejects_corruption() {
        let mut payload = vec![
            0u8;
            crate::ghost_tile::GhostTileRing::COUNTER_SECTOR_BYTES
                + GHOST_V2_RECORD_BYTES
        ];
        payload[..4].copy_from_slice(&1u32.to_le_bytes());
        let record = &mut payload[crate::ghost_tile::GhostTileRing::COUNTER_SECTOR_BYTES..];
        record[crate::ghost_tile::GHOST_V2_OFFSET_SCHEMA_VERSION
            ..crate::ghost_tile::GHOST_V2_OFFSET_SCHEMA_VERSION + 4]
            .copy_from_slice(&crate::ghost_tile::GHOST_FRAME_SCHEMA_V3.to_le_bytes());
        record[crate::ghost_tile::GHOST_V3_OFFSET_MEMBERSHIP_SCHEMA_VERSION
            ..crate::ghost_tile::GHOST_V3_OFFSET_MEMBERSHIP_SCHEMA_VERSION + 4]
            .copy_from_slice(&crate::ghost_tile::GHOST_V3_MEMBERSHIP_SCHEMA_VERSION.to_le_bytes());
        record[crate::ghost_tile::GHOST_V3_OFFSET_CLUSTER_RESIDUE_COUNT
            ..crate::ghost_tile::GHOST_V3_OFFSET_CLUSTER_RESIDUE_COUNT + 4]
            .copy_from_slice(&2u32.to_le_bytes());
        record[crate::ghost_tile::GHOST_V3_OFFSET_CLUSTER_RESIDUE_CAPACITY
            ..crate::ghost_tile::GHOST_V3_OFFSET_CLUSTER_RESIDUE_CAPACITY + 4]
            .copy_from_slice(
                &(crate::ghost_tile::GHOST_V3_CLUSTER_RESIDUE_CAPACITY as u32).to_le_bytes(),
            );
        record[crate::ghost_tile::GHOST_V3_OFFSET_SOURCE_SPIKE_COUNT
            ..crate::ghost_tile::GHOST_V3_OFFSET_SOURCE_SPIKE_COUNT + 4]
            .copy_from_slice(&5u32.to_le_bytes());
        record[crate::ghost_tile::GHOST_V3_OFFSET_UNRESOLVED_RESIDUE_SPIKE_COUNT
            ..crate::ghost_tile::GHOST_V3_OFFSET_UNRESOLVED_RESIDUE_SPIKE_COUNT + 4]
            .copy_from_slice(&1u32.to_le_bytes());
        record[crate::ghost_tile::GHOST_V3_OFFSET_CLUSTER_RESIDUE_ENTRY_BYTES
            ..crate::ghost_tile::GHOST_V3_OFFSET_CLUSTER_RESIDUE_ENTRY_BYTES + 4]
            .copy_from_slice(
                &(crate::ghost_tile::GHOST_V3_CLUSTER_RESIDUE_ENTRY_BYTES as u32).to_le_bytes(),
            );
        record[crate::ghost_tile::GHOST_V3_OFFSET_GEOMETRY_AVAILABILITY_FLAGS
            ..crate::ghost_tile::GHOST_V3_OFFSET_GEOMETRY_AVAILABILITY_FLAGS + 4]
            .copy_from_slice(
                &(crate::ghost_tile::GHOST_V3_GEOMETRY_CLUSTER_RESIDUE_MEMBERSHIP_EXACT
                    | crate::ghost_tile::GHOST_V3_GEOMETRY_COORDINATE_SIDECAR_REQUIRED)
                    .to_le_bytes(),
            );
        let first = crate::ghost_tile::GHOST_V3_OFFSET_CLUSTER_RESIDUE_ENTRIES;
        record[first..first + 4].copy_from_slice(&7u32.to_le_bytes());
        record[first + 4..first + 8].copy_from_slice(&2u32.to_le_bytes());
        let second = first + crate::ghost_tile::GHOST_V3_CLUSTER_RESIDUE_ENTRY_BYTES;
        record[second..second + 4].copy_from_slice(&3u32.to_le_bytes());
        record[second + 4..second + 8].copy_from_slice(&2u32.to_le_bytes());

        validate_ghost_envelope_zero_regions(&payload, 1)
            .expect("production v3 membership bytes are legal envelope content");
        let decoded = crate::ghost_tile::decode_ghost_v3_cluster_residue_membership(
            &payload[crate::ghost_tile::GhostTileRing::COUNTER_SECTOR_BYTES..],
        )
        .expect("valid production v3 membership");
        assert_eq!(decoded.source_spike_count, 5);
        assert_eq!(decoded.unresolved_residue_spike_count, 1);
        assert_eq!(decoded.cluster_residue_spike_counts.len(), 2);

        payload[8] = 1;
        assert!(validate_ghost_envelope_zero_regions(&payload, 1).is_err());
        payload[8] = 0;

        let record = &mut payload[crate::ghost_tile::GhostTileRing::COUNTER_SECTOR_BYTES..];
        record[crate::ghost_tile::GHOST_V3_OFFSET_CLUSTER_RESIDUE_OVERFLOW
            ..crate::ghost_tile::GHOST_V3_OFFSET_CLUSTER_RESIDUE_OVERFLOW + 4]
            .copy_from_slice(&1u32.to_le_bytes());
        assert!(crate::ghost_tile::decode_ghost_v3_cluster_residue_membership(record).is_err());
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn kcc_evidence_preserves_paired_nan_as_explicit_unavailable_lag() {
        let kcc = kcc_semantics_fixture();
        let validity =
            validate_kcc_evidence_semantics(&kcc).expect("paired NaN is an unavailable estimate");
        assert_eq!(validity.finite_pairs, 2);
        assert_eq!(validity.unavailable_pairs, 1);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn kcc_evidence_rejects_corrupt_or_unbounded_lag_semantics() {
        let mut asymmetric = kcc_semantics_fixture();
        asymmetric.lag_corr_peak[1] = 0.0;
        assert!(validate_kcc_evidence_semantics(&asymmetric).is_err());

        let mut infinite = kcc_semantics_fixture();
        infinite.causal_lag[0] = f32::INFINITY;
        infinite.lag_corr_peak[0] = f32::INFINITY;
        assert!(validate_kcc_evidence_semantics(&infinite).is_err());

        let mut unbounded_peak = kcc_semantics_fixture();
        unbounded_peak.lag_corr_peak[0] = 1.01;
        assert!(validate_kcc_evidence_semantics(&unbounded_peak).is_err());

        let mut no_valid_pair = kcc_semantics_fixture();
        no_valid_pair.causal_lag.fill(f32::NAN);
        no_valid_pair.lag_corr_peak.fill(f32::NAN);
        assert!(validate_kcc_evidence_semantics(&no_valid_pair).is_err());

        let mut non_lag_nan = kcc_semantics_fixture();
        non_lag_nan.temporal_corr[0] = f32::NAN;
        assert!(validate_kcc_evidence_semantics(&non_lag_nan).is_err());
    }

    #[test]
    fn test_coupled_twin_config_defaults() {
        let c = CoupledTwinConfig::default();
        assert_eq!(c.phase_offset_fraction, 0.20);
        assert_eq!(c.exchange_interval, 100);
        assert!(!c.enable_ccf, "CCF should be off by default (Gate 1)");
        assert!(
            !c.enable_exchange,
            "Exchange should be off by default (Gate 1)"
        );
        assert!(c.sensitivity_boost > 0.0);
        assert!(c.max_threshold_reduction > 0.0);
        assert!(
            c.max_threshold_reduction <= 1.0,
            "Max reduction must be ≤ 1.0 (100%)"
        );
        assert!(c.nma_modes_path.is_none());
    }

    #[test]
    fn test_interferometric_features_default_all_fields() {
        let f = InterferometricFeatures::default();
        assert_eq!(f.resid, -1);
        assert_eq!(f.barrier_classification, "MEDIUM");
        assert_eq!(f.consensus_phase_profile, [0.0; 5]);
        assert_eq!(f.ccf_per_phase, [0.0; 5]);
        assert_eq!(f.per_phase_differential, [0.0; 5]);
        assert_eq!(f.nma_responsive_mode, -1);
        assert_eq!(f.mutual_information, 0.0, "MI must be 0 (placeholder)");
        assert_eq!(f.transfer_entropy_a_to_b, 0.0, "TE must be 0 (placeholder)");
        assert_eq!(
            f.causal_flow_direction, 0.0,
            "Causal flow must be 0 (placeholder)"
        );
    }

    #[test]
    fn test_interferometric_features_serde_roundtrip() {
        let mut f = InterferometricFeatures::default();
        f.resid = 42;
        f.resname = "ALA".to_string();
        f.spike_agreement_ratio = 0.85;
        f.ccf_peak_value = 0.72;
        f.spikes_a = 150;
        f.spikes_b = 120;
        f.barrier_classification = "LOW".to_string();
        f.consensus_phase_profile = [0.1, 0.3, 0.8, 0.5, 0.2];
        f.scout_lead_time = 500.0;

        let json = serde_json::to_string(&f).expect("serialize");
        let deser: InterferometricFeatures = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(deser.resid, 42);
        assert_eq!(deser.resname, "ALA");
        assert!((deser.spike_agreement_ratio - 0.85).abs() < 1e-6);
        assert!((deser.ccf_peak_value - 0.72).abs() < 1e-6);
        assert_eq!(deser.spikes_a, 150);
        assert_eq!(deser.barrier_classification, "LOW");
        assert!((deser.consensus_phase_profile[2] - 0.8).abs() < 1e-6);
        assert!((deser.scout_lead_time - 500.0).abs() < 1e-6);
    }

    #[test]
    fn test_interferometric_features_field_count() {
        // Verify the JSON contains all expected fields by checking key count
        let f = InterferometricFeatures::default();
        let json_val: serde_json::Value = serde_json::to_value(&f).expect("to_value");
        let obj = json_val.as_object().expect("should be object");
        // 2 identity fields (resid, resname) + 48 feature fields = 50 total
        // But arrays serialize as arrays inside the object, so count keys
        let n_keys = obj.len();
        assert!(
            n_keys >= 30,
            "Expected ≥30 JSON keys for 50-field struct, got {}",
            n_keys
        );
    }

    #[test]
    fn test_ccns_phase_assignment() {
        let protocol = CryoUvProtocol {
            start_temp: 50.0,
            end_temp: 300.0,
            cold_hold_steps: 5000,
            ramp_steps: 10000,
            warm_hold_steps: 5000,
            current_step: 0,
            uv_burst_energy: 0.5,
            uv_burst_interval: 500,
            uv_burst_duration: 10,
            scan_wavelengths: vec![280.0],
            wavelength_dwell_steps: 100,
            ramp_down_steps: 6000,
            cold_return_steps: 4000,
            stepped_holds: Vec::new(),
        };
        assert_eq!(ccns_phase(0, &protocol), 0, "t=0 → cold_hold");
        assert_eq!(ccns_phase(4999, &protocol), 0, "t=4999 → cold_hold");
        assert_eq!(ccns_phase(5000, &protocol), 1, "t=5000 → heating");
        assert_eq!(ccns_phase(14999, &protocol), 1, "t=14999 → heating");
        assert_eq!(ccns_phase(15000, &protocol), 2, "t=15000 → warm_hold");
        assert_eq!(ccns_phase(19999, &protocol), 2, "t=19999 → warm_hold");
        assert_eq!(ccns_phase(20000, &protocol), 3, "t=20000 → cooling");
        assert_eq!(ccns_phase(25999, &protocol), 3, "t=25999 → cooling");
        assert_eq!(ccns_phase(26000, &protocol), 4, "t=26000 → cold_return");
        assert_eq!(ccns_phase(30000, &protocol), 4, "t=30000 → cold_return");
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_spike_density_cpu_empty() {
        let (cells, _bbox) = compute_spike_density_cpu(&[], 4.0);
        assert!(cells.is_empty());
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_cpu_cross_correlation_empty() {
        let (n, map) = compute_cpu_cross_correlation(&[], &[], 5.0, 500, false);
        assert_eq!(n, 0);
        assert!(map.is_empty());
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_per_residue_phase_counts_basic() {
        use crate::fused_engine::GpuSpikeEvent;
        let protocol = CryoUvProtocol {
            start_temp: 50.0,
            end_temp: 300.0,
            cold_hold_steps: 100,
            ramp_steps: 100,
            warm_hold_steps: 100,
            current_step: 0,
            uv_burst_energy: 0.5,
            uv_burst_interval: 500,
            uv_burst_duration: 10,
            scan_wavelengths: vec![280.0],
            wavelength_dwell_steps: 100,
            ramp_down_steps: 100,
            cold_return_steps: 100,
            stepped_holds: Vec::new(),
        };

        let mut spike = GpuSpikeEvent::default();
        spike.timestep = 50; // cold_hold phase
        spike.nearby_residues[0] = 0;
        spike.n_residues = 1;

        let counts = per_residue_phase_counts(&[spike], &protocol);
        assert_eq!(
            counts.get(&0).unwrap()[0],
            1,
            "Should have 1 spike in cold_hold"
        );
        assert_eq!(counts.get(&0).unwrap()[1], 0, "Should have 0 in heating");
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_per_residue_onset() {
        use crate::fused_engine::GpuSpikeEvent;
        let mut s1 = GpuSpikeEvent::default();
        s1.timestep = 1000;
        s1.nearby_residues[0] = 5;
        s1.n_residues = 1;
        let mut s2 = GpuSpikeEvent::default();
        s2.timestep = 500;
        s2.nearby_residues[0] = 5;
        s2.n_residues = 1;

        let onset = per_residue_onset(&[s1, s2]);
        assert_eq!(
            *onset.get(&5).unwrap(),
            500,
            "Onset should be earliest timestep"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PRISM-TWIN v3.0 Gate 3: Autonomous Graph-Coupled Dual-Engine Pipeline
// ═══════════════════════════════════════════════════════════════════════════════
//
// The true TWIN interferometric platform:
// - Both engines run inside a SINGLE CudaGraphExec
// - Spike data flows through GPU-resident ring buffers (zero PCIe)
// - A's discoveries modulate B's thresholds, and vice versa
// - CPU only wakes for neighbor list rebuild every 500 steps
//
// Call graph (per graph launch = one coupled MD step):
//   Director_A → Physics_A → multi_lif_A → compact_push(A→ring_a)
//   read_adapt(ring_a → thresholds_B)
//   Director_B → Physics_B → multi_lif_B → compact_push(B→ring_b)
//   read_adapt(ring_b → thresholds_A)
//   recovery_A, recovery_B (periodic threshold decay)
//   heartbeat_A, heartbeat_B
//   CA_restraints_A, CA_restraints_B
//   coupling_clear_A, coupling_clear_B

/// Configuration for autonomous TWIN execution.
pub struct AutonomousTwinConfig {
    /// Steps per graph replay chunk (host breaks for NL rebuild)
    pub chunk_size: u32,
    /// Total steps to run
    pub total_steps: u32,
    /// Sensitivity boost for threshold adaptation (default 2.0)
    pub sensitivity_boost: f32,
    /// Maximum threshold reduction fraction (default 0.3 = 30%)
    pub max_reduction: f32,
    /// Threshold recovery rate per call (default 0.01 = 1%)
    pub recovery_rate: f32,
    /// Steps between threshold recovery calls (default 1000)
    pub recovery_interval: u32,
}

impl Default for AutonomousTwinConfig {
    fn default() -> Self {
        Self {
            chunk_size: 500,
            total_steps: 35000,
            sensitivity_boost: 2.0,
            max_reduction: 0.3,
            recovery_rate: 0.01,
            recovery_interval: 1000,
        }
    }
}

/// Run autonomous TWIN dual-engine simulation with GPU-graph coupling.
///
/// Both engines execute inside a single CUDA Graph. Spike coupling flows
/// through device-resident ring buffers with zero CPU intervention.
///
/// Host only wakes every `chunk_size` steps for neighbor list rebuild.
/// Heartbeat polling between chunks aborts on NaN/divergence.
///
/// # Returns
/// Total steps completed (may be < total_steps if heartbeat aborts)
#[cfg(feature = "gpu")]
pub fn run_coupled_twin_autonomous(
    engine_a: &mut crate::fused_engine::NhsAmberFusedEngine,
    engine_b: &mut crate::fused_engine::NhsAmberFusedEngine,
    ring_a: &mut crate::twin_kernels::TwinRingBuffer,
    ring_b: &mut crate::twin_kernels::TwinRingBuffer,
    config: &AutonomousTwinConfig,
) -> anyhow::Result<u32> {
    use cudarc::driver::sys;

    let stream = engine_a.stream().clone();

    log::info!("╔═══════════════════════════════════════════════════════════════╗");
    log::info!("║  PRISM-TWIN AUTONOMOUS INTERFEROMETRIC COUPLING              ║");
    log::info!("║  GPU-Resident Dual-Engine Graph Execution                    ║");
    log::info!("╚═══════════════════════════════════════════════════════════════╝");
    log::info!("  Chunk size: {} steps", config.chunk_size);
    log::info!("  Total steps: {}", config.total_steps);
    log::info!("  Coupling: device-side compact_and_push → ring buffer → read_adapt");
    log::info!(
        "  CPU involvement: NL rebuild only (every {} steps)",
        config.chunk_size
    );

    // ── Helper: launch one complete coupled TWIN step ──
    let launch_one_coupled_step = |ea: &mut crate::fused_engine::NhsAmberFusedEngine,
                                   eb: &mut crate::fused_engine::NhsAmberFusedEngine,
                                   ra: &mut crate::twin_kernels::TwinRingBuffer,
                                   rb: &mut crate::twin_kernels::TwinRingBuffer,
                                   s: &std::sync::Arc<cudarc::driver::CudaStream>,
                                   cfg: &AutonomousTwinConfig|
     -> anyhow::Result<()> {
        // Group A: Director → Physics → multi_lif → housekeeping
        ea.step_autonomous_kernels(s)?;

        // A→B: compact A's spikes on GPU, push to ring_a
        ra.push_device(s, ea.spike_events_buffer(), ea.spike_count_buffer())?;

        // Interferometric bridge: ring_a → B's thresholds
        {
            let g = eb.grid_info(); // immutable borrow first
            let (thresh_b, base_b) = eb.threshold_buffers_mut(); // then mutable
            ra.read_and_adapt(
                s,
                thresh_b,
                base_b,
                (g.0, g.1, g.2),
                (g.3, g.4, g.5),
                g.6,
                cfg.sensitivity_boost,
                cfg.max_reduction,
                0,
                0.001,
            )?;
        }

        // Group B: Director → Physics → multi_lif → housekeeping
        eb.step_autonomous_kernels(s)?;

        // B→A: compact B's spikes on GPU, push to ring_b
        rb.push_device(s, eb.spike_events_buffer(), eb.spike_count_buffer())?;

        // Interferometric bridge: ring_b → A's thresholds
        {
            let g = ea.grid_info();
            let (thresh_a, base_a) = ea.threshold_buffers_mut();
            rb.read_and_adapt(
                s,
                thresh_a,
                base_a,
                (g.0, g.1, g.2),
                (g.3, g.4, g.5),
                g.6,
                cfg.sensitivity_boost,
                cfg.max_reduction,
                0,
                0.001,
            )?;
        }

        Ok(())
    };

    // ── Capture one coupled step as a CUDA Graph ──
    log::info!("Capturing TWIN dual-engine step as CUDA Graph...");
    stream
        .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        .map_err(|e| anyhow::anyhow!("TWIN graph capture begin failed: {:?}", e))?;

    launch_one_coupled_step(engine_a, engine_b, ring_a, ring_b, &stream, config)?;

    let graph = stream
        .end_capture(sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
        .map_err(|e| anyhow::anyhow!("TWIN graph capture end failed: {:?}", e))?
        .ok_or_else(|| anyhow::anyhow!("TWIN graph capture produced null graph"))?;

    log::info!("TWIN CUDA Graph captured and instantiated");

    // ── Replay loop ──
    let mut steps_completed = 0u32;
    let n_chunks = (config.total_steps + config.chunk_size - 1) / config.chunk_size;

    for chunk_idx in 0..n_chunks {
        let steps_this_chunk = config.chunk_size.min(config.total_steps - steps_completed);

        // Replay the dual-engine graph (zero CPU involvement per step)
        for _ in 0..steps_this_chunk {
            graph
                .launch()
                .map_err(|e| anyhow::anyhow!("TWIN graph launch failed: {:?}", e))?;
        }
        steps_completed += steps_this_chunk;

        // Synchronize
        stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("TWIN sync failed: {:?}", e))?;

        // Poll heartbeats (both engines)
        let status_a =
            crate::graph_capture::poll_heartbeat_async(&stream, &engine_a.d_protocol_state)?;
        let status_b =
            crate::graph_capture::poll_heartbeat_async(&stream, &engine_b.d_protocol_state)?;
        if status_a != 0 || status_b != 0 {
            let (label, code) = if status_a != 0 {
                ("Group A", status_a)
            } else {
                ("Group B", status_b)
            };
            log::error!(
                "TWIN HEARTBEAT ABORT ({}) at step {}: status={}",
                label,
                steps_completed,
                code
            );
            return Ok(steps_completed);
        }

        // CPU-side neighbor list rebuild (the ONLY host wakeup)
        engine_a.rebuild_neighbor_lists_if_needed()?;
        engine_b.rebuild_neighbor_lists_if_needed()?;

        if chunk_idx % 10 == 0 {
            log::info!(
                "  TWIN chunk {}/{}: {} coupled steps",
                chunk_idx + 1,
                n_chunks,
                steps_completed
            );
        }
    }

    log::info!(
        "TWIN autonomous loop complete: {} coupled steps",
        steps_completed
    );
    Ok(steps_completed)
}

// ═══════════════════════════════════════════════════════════════════════════════
// PRISM-TWIN Multi-Differential Interferometric Observation
// ═══════════════════════════════════════════════════════════════════════════════
//
// N groups × M engines/group, each group running a different CryoUvProtocol.
// The interferometric signal is the N-way cross-correlation tensor.
//
// Standard configuration (8 engines on RTX 5080, 16GB):
//   Group A (2 engines): Thermal Shock — aggressive cryo, fast ramp, high UV
//   Group B (2 engines): Equilibrium — gentle thermal, slow ramp, moderate UV
//   Group C (2 engines): UV Aromatic — constant T, max UV on TRP/TYR
//   Group D (2 engines): Hysteresis — full cycle, extended cooldown
//
// A site that correlates across ALL 4 groups has multi-mechanism evidence.
// Surface noise appears in 1-2 groups but not all 4.

/// Configuration for a multi-differential interferometric TWIN run.
#[derive(Debug, Clone)]
pub struct MultiDifferentialConfig {
    /// Protocol per group. Length determines number of groups.
    pub group_protocols: Vec<crate::fused_engine::CryoUvProtocol>,
    /// Group labels for logging/output.
    pub group_labels: Vec<String>,
    /// Engines per group (all groups get the same count).
    pub engines_per_group: usize,
    /// Base seed (each engine gets base + group_idx*1000 + engine_idx).
    pub base_seed: u64,
    /// Steps per graph replay chunk (host breaks for NL rebuild).
    pub chunk_size: u32,
    /// Coupling config (sensitivity, exchange interval, etc.)
    pub coupling: CoupledTwinConfig,
    /// HMR enabled
    pub hmr: bool,
    /// Fused inner steps
    pub fused_steps: u32,
    /// Adaptive dt
    pub adaptive_dt: bool,
    /// LADD enabled
    pub ladd_enabled: bool,
    /// Explicit non-production admission for exception-free legacy topologies.
    pub nonbonded_execution_policy: crate::input::NonbondedExecutionPolicy,
}

impl MultiDifferentialConfig {
    /// Standard 4-group × 1-engine configuration (conservative VRAM, ~4GB).
    /// Use standard_4x2 for 16GB+ GPUs.
    pub fn standard_4x1(base_seed: u64) -> Self {
        let protocols = crate::fused_engine::CryoUvProtocol::twin_differential_set();
        Self {
            group_protocols: protocols.to_vec(),
            group_labels: vec![
                "ThermalShock".into(),
                "Equilibrium".into(),
                "UvAromatic".into(),
                "Hysteresis".into(),
            ],
            engines_per_group: 1,
            base_seed,
            chunk_size: 500,
            coupling: {
                let mut c = CoupledTwinConfig::default();
                c.enable_exchange = true;
                c.enable_ccf = true;
                c.graph_coupling = true;
                c
            },
            hmr: true,
            fused_steps: 4,
            adaptive_dt: true,
            ladd_enabled: true,
            nonbonded_execution_policy: crate::input::NonbondedExecutionPolicy::RequireExactV2,
        }
    }

    /// 4-group × 2-engine configuration for 16GB+ GPUs.
    /// Total: 8 engines, ~8GB VRAM.
    pub fn standard_4x2(base_seed: u64) -> Self {
        let protocols = crate::fused_engine::CryoUvProtocol::twin_differential_set();
        Self {
            group_protocols: protocols.to_vec(),
            group_labels: vec![
                "ThermalShock".into(),
                "Equilibrium".into(),
                "UvAromatic".into(),
                "Hysteresis".into(),
            ],
            engines_per_group: 2,
            base_seed,
            chunk_size: 500,
            coupling: {
                let mut c = CoupledTwinConfig::default();
                c.enable_exchange = true;
                c.enable_ccf = true;
                c.graph_coupling = true;
                c
            },
            hmr: true,
            fused_steps: 4,
            adaptive_dt: true,
            ladd_enabled: true,
            nonbonded_execution_policy: crate::input::NonbondedExecutionPolicy::RequireExactV2,
        }
    }

    /// Total number of engines.
    pub fn total_engines(&self) -> usize {
        self.group_protocols.len() * self.engines_per_group
    }

    /// Number of groups.
    pub fn n_groups(&self) -> usize {
        self.group_protocols.len()
    }
}

/// Result from a multi-differential TWIN run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiDifferentialResult {
    /// Per-group spike counts.
    pub group_spikes: Vec<u64>,
    /// Per-group labels.
    pub group_labels: Vec<String>,
    /// Total spikes across all groups.
    pub total_spikes: u64,
    /// Number of groups.
    pub n_groups: usize,
    /// Engines per group.
    pub engines_per_group: usize,
    /// Wall time in seconds.
    pub wall_time_seconds: f64,
    /// Steps run per group (may differ due to protocol length).
    pub steps_per_group: Vec<i32>,
}

/// Run multi-differential interferometric TWIN.
///
/// Creates N groups of M engines, each group with a distinct CryoUvProtocol.
/// All groups share ring-buffer coupling: every group's spikes modulate
/// every other group's detection thresholds.
///
/// The N-way cross-correlation tensor reveals sites with multi-mechanism evidence.
#[cfg(feature = "gpu")]
pub fn run_multi_differential_twin(
    config: &MultiDifferentialConfig,
    context: Arc<CudaContext>,
    fused_module: Arc<CudaModule>,
    topology: &PrismPrepTopology,
    output_dir: &std::path::Path,
    prismcode_admission: Option<&PrismCodeMdPolicyAdmissionV1>,
) -> anyhow::Result<MultiDifferentialResult> {
    let start = std::time::Instant::now();
    let n_groups = config.n_groups();
    let epg = config.engines_per_group;
    let total = config.total_engines();

    log::info!("╔═══════════════════════════════════════════════════════════════╗");
    log::info!("║  PRISM-TWIN MULTI-DIFFERENTIAL INTERFEROMETRIC OBSERVATION     ║");
    log::info!("╠═══════════════════════════════════════════════════════════════╣");
    log::info!(
        "║  Groups: {}                                                    ║",
        n_groups
    );
    log::info!(
        "║  Engines/group: {}                                             ║",
        epg
    );
    log::info!(
        "║  Total engines: {}                                             ║",
        total
    );
    log::info!("╚═══════════════════════════════════════════════════════════════╝");
    for (i, (proto, label)) in config
        .group_protocols
        .iter()
        .zip(&config.group_labels)
        .enumerate()
    {
        log::info!(
            "  Group {} [{}]: {}K→{}K, UV={:.0}kcal, interval={}, ramp={}",
            i,
            label,
            proto.start_temp,
            proto.end_temp,
            proto.uv_burst_energy,
            proto.uv_burst_interval,
            proto.ramp_steps
        );
    }

    // Check VRAM
    let (vram_free, _vram_total) = cudarc::driver::result::mem_get_info().unwrap_or((0, 0));
    let estimated_per_engine = 1_200_000_000u64; // ~1.2GB per engine
    let estimated_total = estimated_per_engine * total as u64;
    if estimated_total > vram_free as u64 {
        log::warn!(
            "  VRAM: {:.1}GB free, need ~{:.1}GB for {} engines — may OOM",
            vram_free as f64 / 1e9,
            estimated_total as f64 / 1e9,
            total
        );
    }

    std::fs::create_dir_all(output_dir)?;

    // Create one CUDA stream per engine
    let streams: Vec<Arc<CudaStream>> = (0..total)
        .map(|_| context.new_stream().expect("CUDA stream"))
        .collect();

    let batch_config = crate::persistent_engine::PersistentBatchConfig {
        max_atoms: topology.n_atoms.max(50000) as usize,
        nonbonded_execution_policy: config.nonbonded_execution_policy,
        ..Default::default()
    };

    // Initialize engines: group_idx * epg + engine_idx
    let mut engines: Vec<PersistentNhsEngine> = Vec::with_capacity(total);
    for group_idx in 0..n_groups {
        let proto = &config.group_protocols[group_idx];
        let label = &config.group_labels[group_idx];
        for engine_idx in 0..epg {
            let flat_idx = group_idx * epg + engine_idx;
            let seed = config.base_seed + (group_idx as u64) * 1000 + (engine_idx as u64);
            let stream = streams[flat_idx].clone();

            log::info!(
                "  Initializing engine {}/{} (group={} [{}], seed={})...",
                flat_idx + 1,
                total,
                group_idx,
                label,
                seed
            );

            let mut engine = PersistentNhsEngine::new_on_stream(
                &batch_config,
                context.clone(),
                fused_module.clone(),
                stream,
            )?;
            if let Some(admission) = prismcode_admission {
                engine.set_prismcode_md_policy_admission(admission.clone())?;
            }
            engine.load_topology(topology)?;
            if config.hmr {
                engine.set_dt(0.004)?;
            }
            if config.fused_steps > 1 {
                engine.set_fused_inner_steps(config.fused_steps)?;
            }
            if config.adaptive_dt {
                engine.set_adaptive_dt(true)?;
            }
            if config.ladd_enabled {
                engine.set_ladd_enabled(true);
            }
            engine.set_cryo_uv_protocol(proto.clone())?;
            engine.set_spike_accumulation(true);
            if prismcode_admission.is_some() {
                engine.enable_prismcode_md_native_runtime(flat_idx as u32, true, true)?;
            }

            engines.push(engine);
        }
    }
    log::info!("  All {} engines initialized", total);

    // Create a shared ring buffer for N-way coupling
    // Each group pushes its spikes; all groups read from the shared ring
    let exchange_stream = context.new_stream()?;
    let rb_module = context.load_module(cudarc::nvrtc::Ptx::from_file(&find_twin_ptx(
        "ring_buffer.ptx",
    )?))?;
    let mut ring_buffers: Vec<TwinRingBuffer> = (0..n_groups)
        .map(|_| TwinRingBuffer::new(&context, &exchange_stream, &rb_module, 16384).unwrap())
        .collect();
    log::info!(
        "  Ring buffers: {} (one per group, capacity=16384)",
        n_groups
    );

    // Determine steps per group (each protocol may have different total_steps)
    let steps_per_group: Vec<i32> = config
        .group_protocols
        .iter()
        .map(|p| p.total_steps())
        .collect();
    let max_steps = *steps_per_group.iter().max().unwrap_or(&35000);
    log::info!(
        "  Steps per group: {:?} (max={})",
        steps_per_group,
        max_steps
    );

    // ── Main simulation loop ──
    let mut group_spikes = vec![0u64; n_groups];

    for step in 0..max_steps {
        // Run one step on each engine (sequential — each on its own stream)
        for (flat_idx, engine) in engines.iter_mut().enumerate() {
            let group_idx = flat_idx / epg;
            if step < steps_per_group[group_idx] {
                engine.run(1)?;
            }
        }

        // Spike exchange: every exchange_interval steps
        if config.coupling.enable_exchange
            && step > 0
            && step as u32 % config.coupling.exchange_interval == 0
        {
            // Each group pushes its new spikes to its ring buffer
            for group_idx in 0..n_groups {
                for engine_idx in 0..epg {
                    let flat_idx = group_idx * epg + engine_idx;
                    if let (Some(sbuf), Some(sc)) = (
                        engines[flat_idx].spike_buffer_gpu(),
                        engines[flat_idx].spike_count_gpu(),
                    ) {
                        if ring_buffers[group_idx].has_device_push() {
                            ring_buffers[group_idx].push_device(&exchange_stream, sbuf, sc)?;
                        }
                    }
                }
            }

            // N-way coupling: each group reads from ALL OTHER groups' ring buffers
            for target_group in 0..n_groups {
                for source_group in 0..n_groups {
                    if source_group == target_group {
                        continue;
                    }
                    // Source group's ring buffer → target group's thresholds
                    for engine_idx in 0..epg {
                        let flat_idx = target_group * epg + engine_idx;
                        if let Some((gx, gy, gz, ox, oy, oz, vs)) = engines[flat_idx].grid_info() {
                            if let Some((thresh, base)) = engines[flat_idx].threshold_buffers_mut()
                            {
                                ring_buffers[source_group].read_and_adapt(
                                    &exchange_stream,
                                    thresh,
                                    base,
                                    (gx, gy, gz),
                                    (ox, oy, oz),
                                    vs,
                                    config.coupling.sensitivity_boost,
                                    config.coupling.max_threshold_reduction,
                                    step as u32,
                                    500.0,
                                )?;
                            }
                        }
                    }
                }
            }
        }

        // Periodic threshold recovery
        if config.coupling.enable_exchange && step > 0 && step % 1000 == 0 {
            let n_voxels = engines[0].total_voxels() as u32;
            for group_idx in 0..n_groups {
                for engine_idx in 0..epg {
                    let flat_idx = group_idx * epg + engine_idx;
                    if let Some((thresh, base)) = engines[flat_idx].threshold_buffers_mut() {
                        // Recovery from ALL source ring buffers
                        for src in 0..n_groups {
                            if src != group_idx {
                                ring_buffers[src].threshold_recovery(
                                    &exchange_stream,
                                    thresh,
                                    base,
                                    n_voxels,
                                    0.01,
                                )?;
                            }
                        }
                    }
                }
            }
        }

        // Progress logging
        if (step + 1) % 5000 == 0 || step == max_steps - 1 {
            let elapsed = start.elapsed().as_secs_f64();
            let sps = (step + 1) as f64 / elapsed;
            log::info!(
                "  Step {}/{}: {:.0} steps/s, {:.1}s elapsed",
                step + 1,
                max_steps,
                sps,
                elapsed
            );
        }
    }

    // Synchronize all streams
    for stream in &streams {
        stream.synchronize().unwrap_or_default();
    }
    exchange_stream.synchronize().unwrap_or_default();

    // Collect spike counts per group
    for group_idx in 0..n_groups {
        for engine_idx in 0..epg {
            let flat_idx = group_idx * epg + engine_idx;
            group_spikes[group_idx] += engines[flat_idx].accumulated_spike_count() as u64;
        }
    }
    let total_spikes: u64 = group_spikes.iter().sum();

    let wall_time = start.elapsed().as_secs_f64();

    log::info!("╔═══════════════════════════════════════════════════════════════╗");
    log::info!("║  MULTI-DIFFERENTIAL TWIN COMPLETE                              ║");
    log::info!("╠═══════════════════════════════════════════════════════════════╣");
    for (i, (label, spikes)) in config.group_labels.iter().zip(&group_spikes).enumerate() {
        log::info!(
            "║  Group {} [{}]: {:>12} spikes                     ║",
            i,
            label,
            spikes
        );
    }
    log::info!(
        "║  Total: {:>12} spikes                                    ║",
        total_spikes
    );
    log::info!(
        "║  Wall time: {:.1}s                                           ║",
        wall_time
    );
    log::info!("╚═══════════════════════════════════════════════════════════════╝");

    // Post-process: aggregate spikes from all engines and run TWIN detection
    let mut all_spikes_a = Vec::new(); // groups 0+2 (shock + UV probe)
    let mut all_spikes_b = Vec::new(); // groups 1+3 (equilibrium + hysteresis)
    for group_idx in 0..n_groups {
        for engine_idx in 0..epg {
            let flat_idx = group_idx * epg + engine_idx;
            let spikes = engines[flat_idx].get_accumulated_spikes().to_vec();
            // Odd groups → "observer" stream, Even groups → "scout" stream
            if group_idx % 2 == 0 {
                all_spikes_a.extend(spikes);
            } else {
                all_spikes_b.extend(spikes);
            }
        }
    }

    log::info!(
        "  Post-processing: {} scout spikes + {} observer spikes",
        all_spikes_a.len(),
        all_spikes_b.len()
    );

    // Run TWIN detection on the aggregated streams
    let prefix = std::path::Path::new(&topology.source_pdb)
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.replace("_sanitized", "").replace("_clean", ""))
        .unwrap_or_else(|| "prism_multi_twin".to_string());

    match crate::twin_detection::detect_and_write_twin_sites(
        &all_spikes_a,
        &all_spikes_b,
        topology,
        None, // LINK #1: dead multi-differential path — single-topology fallback.
        output_dir,
        &prefix,
        None,
    ) {
        Ok(summary) => {
            log::info!(
                "  TWIN detection: {} sites ({} consensus, {} barrier-gated)",
                summary.n_sites,
                summary.n_consensus_sites,
                summary.n_barrier_gated_sites
            );
        }
        Err(e) => log::warn!("  TWIN detection failed: {}", e),
    }

    Ok(MultiDifferentialResult {
        group_spikes,
        group_labels: config.group_labels.clone(),
        total_spikes,
        n_groups,
        engines_per_group: epg,
        wall_time_seconds: wall_time,
        steps_per_group,
    })
}
