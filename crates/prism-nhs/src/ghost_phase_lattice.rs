//! GhostPhaseLattice4D — physically-constrained spatiotemporal connected
//! components for PRISM Ghost v2 records.
//!
//! Replaces the legacy O(N²) point-cloud DBSCAN on `_stream{:02}_ghost_tiles.bin`
//! records with a 4-D phase lattice: nodes are bucketed by spatial cell ×
//! protocol phase × step bucket, and edges are adjudicated by:
//!
//!   * **Boolean constraints (hard fails)** — arrow of time, temporal adjacency,
//!     AABB overlap, monotone protocol-phase transition.
//!   * **Continuous physics scoring** — SO(3) cosine similarity weighted across
//!     the 4 GhostTileFrame planes (geometry / causality / thermo / chemistry),
//!     causal-driver continuity, KL-divergence smoothness.
//!
//! Edges with a composite score above `so3_threshold` (default 0.75) trigger a
//! union-find merge on the GPU; components survive into a per-site
//! [`PhaseManifold`] / [`ThermCcnsLifecycle`] / [`So3Manifold`] block on the
//! [`crate::site_manifest::SiteManifest`] structure.
//!
//! ## Why this is not "just a faster DBSCAN"
//!
//! Legacy DBSCAN flattened ~12 M raw spikes into a static 3-D grid, averaging
//! across kinematic phases (open / closed / burst) into one dead centroid and
//! merging spatial lobes (e.g. Y > 10 vs Y < 0) that share a global thermo
//! transition — anchoring the centroid in empty solvent. The 4-D lattice
//! preserves the temporal trajectory and refuses to merge across illegal
//! protocol-phase transitions or across the empty-solvent void.
//!
//! ## Provenance — Ghost v2 record schema
//!
//! Source-of-truth for the raw bytes: [`crate::ghost_tile`] (offsets at
//! `GHOST_V2_OFFSET_*`). This module never mutates the ring; it consumes a
//! borrowed slice of [`GhostTileFrame`] records (or a borrowed `&[u8]` payload
//! buffer) and projects each one to a [`GhostPhaseLatticeNode`].

#![cfg(any(feature = "gpu", feature = "cpu-analysis"))]

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
#[cfg(feature = "gpu")]
use std::sync::Arc;

#[cfg(feature = "gpu")]
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};

use crate::ghost_segment::GhostOverflowConservation;
use crate::ghost_tile::{
    decode_ghost_v3_cluster_residue_membership, GhostTileFrame, GhostV3ClusterResidueMembership,
    GHOST_FCF_BIT_SPATIAL_NATIVE_AABB_MIDPOINT, GHOST_FRAME_SCHEMA_V2, GHOST_FRAME_SCHEMA_V3,
    GHOST_RECORD_BYTES, GHOST_TELEMETRY_CLASS_TAINTED, GHOST_V2_OFFSET_AABB_MAX,
    GHOST_V2_OFFSET_AABB_MIN, GHOST_V2_OFFSET_CENTROID, GHOST_V2_OFFSET_DT_FS,
    GHOST_V2_OFFSET_FIELD_COMPLETENESS_FLAGS, GHOST_V2_OFFSET_GEAR_ID,
    GHOST_V2_OFFSET_PERTURBATION_CHAN, GHOST_V2_OFFSET_SCHEMA_VERSION, GHOST_V2_OFFSET_STEP_IDX,
};

// ─── Sentinels (mandatory by Part V — never zero-fill missing physical data) ──

/// Sentinel for "this node has no resolved causal lead residue". Mirrors the
/// kernel-level u32 sentinel written by the Ghost v2 producer when the per-frame
/// KCC argmax has not yet been wired into the captured graph.
pub const NODE_CAUSAL_LEAD_NONE: u32 = u32::MAX;

/// Sentinel for "this node has no resolved gear id". Distinguishes
/// "Wave A default 0" (which is a real gear) from "missing/unknown" emission.
pub const NODE_GEAR_ID_NONE: u32 = u32::MAX;

/// Sentinel for "ccns phase bin not provided by the upstream telemetry plane".
pub const NODE_CCNS_PHASE_NONE: u16 = 0xFFFF;

/// SO(3) plane bit flags — Bit 0=Geometry, 1=Causality, 2=Thermo, 3=Chemistry.
pub const SO3_PLANE_GEOMETRY: u8 = 0b0001;
pub const SO3_PLANE_CAUSALITY: u8 = 0b0010;
pub const SO3_PLANE_THERMO: u8 = 0b0100;
pub const SO3_PLANE_CHEMISTRY: u8 = 0b1000;

/// Default acceptance threshold for the composite physics score (SO(3) ×
/// causal × thermo). Edges below this threshold are rejected pre-union.
pub const DEFAULT_SO3_THRESHOLD: f32 = 0.75;

/// Default spatial cell size in Å. Picked to match the legacy 5.0 Å DBSCAN
/// epsilon so the lattice's neighborhood radius (3³ cell box) covers the same
/// physical neighborhood that the legacy backend admitted.
pub const DEFAULT_SPATIAL_CELL_A: f32 = 5.0;

/// Default temporal-edge cap (in MD steps). Edges that span more than this
/// number of steps are rejected — beyond ~500 steps the kinematic continuity
/// assumption fails for cryptic-pocket lifecycles.
pub const DEFAULT_MAX_TEMPORAL_EDGE_STEPS: u64 = 500;

/// Default bucket size for `step_bucket = floor(step_idx / bucket_size)`.
/// Mirrors the temporal-edge cap so a single bucket spans one max-edge window.
pub const DEFAULT_STEP_BUCKET_SIZE: u64 = 500;

/// Number of protocol phases recognised by the kernel.
pub const N_PROTOCOL_PHASES: u8 = 4;

/// Version of the accepted-edge row shared by static CUDA, NVRTC, Rust and
/// durable lattice manifests.
pub const GHOST_LATTICE_EDGE_SCHEMA_VERSION: u32 = 1;

/// Durable accepted-edge envelope. Version 2 adds independent pass-1 count
/// conservation and consumer-supplied artifact/node-lineage hash binding.
pub const GHOST_LATTICE_EDGE_ARTIFACT_SCHEMA_VERSION: u32 = 2;

/// Version of one exact Ghost-record to lattice-node byte-lineage row.
pub const GHOST_RECORD_BYTE_LINEAGE_SCHEMA_VERSION: u32 = 1;

/// Byte width of one accepted edge emitted by the device kernel.
pub const GHOST_LATTICE_EDGE_BYTES: usize = 56;

/// Protocol-phase ordinals matching directive Part II.1.
pub const PHASE_COLD_HOLD: u8 = 0;
pub const PHASE_HEATING: u8 = 1;
pub const PHASE_WARM_HOLD: u8 = 2;
pub const PHASE_COOLING: u8 = 3;

/// One exact interval in an engine-owned stream protocol.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProtocolPhaseInterval {
    /// Inclusive MD step.
    pub start_step: u64,
    /// Exclusive MD step.
    pub end_step: u64,
    /// Lattice phase ordinal.
    pub phase_ordinal: u8,
    /// Stable human-readable phase name.
    pub phase_name: String,
}

/// Cryptographically bound per-stream condition and protocol schedule.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct StreamProtocolPhaseBinding {
    /// Schema discriminator.
    pub schema_kind: String,
    /// Schema version.
    pub schema_version: u32,
    /// Source stream identity.
    pub stream_id: u16,
    /// Thermodynamic/experimental condition identity.
    pub condition_id: String,
    /// Engine protocol name.
    pub protocol_name: String,
    /// Hash of the complete engine protocol state used for this stream.
    pub protocol_state_hash: String,
    /// Exact, contiguous phase intervals.
    pub intervals: Vec<ProtocolPhaseInterval>,
    /// SHA-256 over the canonical identity and interval tuple sequence.
    pub phase_schedule_digest_sha256: String,
    /// Set only by constructor validation; rechecked by every reader.
    pub phase_protocols_bound_by_stream: bool,
}

impl StreamProtocolPhaseBinding {
    /// Construct the production four-phase lattice view from exact engine
    /// boundaries. `cooling_end` includes the cold-return hold because the
    /// lattice ontology currently represents ramp-down plus cold-return as one
    /// cooling phase. An absent phase has a zero-length interval and never
    /// matches a record; phase ordinals remain stable across protocols.
    pub fn from_engine_boundaries(
        stream_id: u16,
        condition_id: String,
        protocol_name: String,
        protocol_state_hash: String,
        cold_hold_end: u64,
        ramp_end: u64,
        warm_hold_end: u64,
        cooling_end: u64,
    ) -> Result<Self> {
        let intervals = vec![
            ProtocolPhaseInterval {
                start_step: 0,
                end_step: cold_hold_end,
                phase_ordinal: PHASE_COLD_HOLD,
                phase_name: "cold_hold".to_string(),
            },
            ProtocolPhaseInterval {
                start_step: cold_hold_end,
                end_step: ramp_end,
                phase_ordinal: PHASE_HEATING,
                phase_name: "heating".to_string(),
            },
            ProtocolPhaseInterval {
                start_step: ramp_end,
                end_step: warm_hold_end,
                phase_ordinal: PHASE_WARM_HOLD,
                phase_name: "warm_hold".to_string(),
            },
            ProtocolPhaseInterval {
                start_step: warm_hold_end,
                end_step: cooling_end,
                phase_ordinal: PHASE_COOLING,
                phase_name: "cooling".to_string(),
            },
        ];
        let digest = phase_binding_digest(
            stream_id,
            &condition_id,
            &protocol_name,
            &protocol_state_hash,
            &intervals,
        );
        let binding = Self {
            schema_kind: "prism_ghost_stream_protocol_phase_binding".to_string(),
            schema_version: 1,
            stream_id,
            condition_id,
            protocol_name,
            protocol_state_hash,
            intervals,
            phase_schedule_digest_sha256: digest,
            phase_protocols_bound_by_stream: true,
        };
        binding.verify()?;
        Ok(binding)
    }

    /// Verify exact coverage, phase order, identity and digest.
    pub fn verify(&self) -> Result<()> {
        if self.schema_kind != "prism_ghost_stream_protocol_phase_binding"
            || self.schema_version != 1
            || self.condition_id.trim().is_empty()
            || self.protocol_name.trim().is_empty()
            || self.protocol_state_hash.len() != 64
            || !self
                .protocol_state_hash
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
            || !self.phase_protocols_bound_by_stream
        {
            bail!("invalid or unbound Ghost stream protocol identity");
        }
        if self.intervals.len() != N_PROTOCOL_PHASES as usize {
            bail!("Ghost phase binding must contain exactly four intervals");
        }
        let mut expected_start = 0u64;
        for (expected_phase, interval) in self.intervals.iter().enumerate() {
            if interval.start_step != expected_start
                || interval.end_step < interval.start_step
                || interval.phase_ordinal != expected_phase as u8
            {
                bail!(
                    "Ghost phase gap/overlap/order error at stream {} phase {}",
                    self.stream_id,
                    expected_phase
                );
            }
            expected_start = interval.end_step;
        }
        if expected_start == 0 {
            bail!("Ghost phase binding has no physical-step coverage");
        }
        let expected = phase_binding_digest(
            self.stream_id,
            &self.condition_id,
            &self.protocol_name,
            &self.protocol_state_hash,
            &self.intervals,
        );
        if expected != self.phase_schedule_digest_sha256 {
            bail!(
                "Ghost phase schedule digest mismatch for stream {}",
                self.stream_id
            );
        }
        Ok(())
    }

    /// Resolve a step only when it lies inside the exact engine schedule.
    pub fn checked_phase_for(&self, step_idx: u64) -> Option<u8> {
        self.intervals
            .iter()
            .find(|interval| step_idx >= interval.start_step && step_idx < interval.end_step)
            .map(|interval| interval.phase_ordinal)
    }
}

impl PhaseSchedule for StreamProtocolPhaseBinding {
    fn phase_for(&self, step_idx: u64, _gear_id: u32) -> u8 {
        self.checked_phase_for(step_idx).unwrap_or(u8::MAX)
    }
}

/// Per-stream conservation evidence emitted by the strict segment loader.
#[derive(Clone, Debug, Serialize)]
pub struct GhostStreamConservation {
    /// Stream identity.
    pub stream_id: u16,
    /// SHA-256 of the exact committed stream manifest bytes.
    pub segment_manifest_sha256: String,
    /// SHA-256 of the immutable producer context in manifest and headers.
    pub segment_context_digest_sha256: String,
    /// Engine-produced record count.
    pub produced_records: u64,
    /// Fsynced segment record count.
    pub persisted_records: u64,
    /// Structurally decoded record count.
    pub decoded_records: u64,
    /// Projected lattice node count.
    pub lattice_node_lineage_count: u64,
    /// Bound condition identity.
    pub condition_id: String,
    /// Bound protocol name.
    pub protocol_name: String,
    /// Bound protocol-state hash.
    pub protocol_state_hash: String,
    /// Bound schedule digest.
    pub phase_schedule_digest_sha256: String,
    /// Exact equality verdict.
    pub equality_pass: bool,
    /// Persisted engine-owned capacity/drain/terminal-residual authority.
    pub overflow_conservation: GhostOverflowConservation,
    /// True only because the v4 verifier reconstructed ordered headers,
    /// ordinals, manifest totals and every segment SHA before this receipt.
    pub ordinal_header_manifest_reconstruction_pass: bool,
    /// Number of records lost before persistence. Successful v4 admission can
    /// expose only zero; it is derived from the bound overflow authority.
    pub overflow_loss_count: u64,
    /// Complete per-stream overflow-conservation verdict.
    pub overflow_conservation_pass: bool,
}

/// Exact byte-level provenance for one decoded Ghost record and its one
/// lattice-node projection. This row is constructed while the authenticated
/// segment bytes are still in hand; downstream Path-B/fiber code must consume
/// it directly and must not reconstruct provenance from node fields.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct GhostRecordByteLineage {
    /// Lineage row schema version.
    pub schema_version: u32,
    /// Source stream identity.
    pub stream_id: u16,
    /// Immutable run and replica identities embedded in the segment chain.
    pub run_id: String,
    pub replica_id: String,
    pub replica_seed: u64,
    pub stream_seed: u64,
    /// Source topology and selected executable identities.
    pub topology_sha256: String,
    pub selected_engine_sha256: String,
    /// Record ABI and producer-source identities.
    pub record_abi_sha256: String,
    pub record_source_sha256: String,
    /// Digest of the complete immutable segment context.
    pub segment_context_digest_sha256: String,
    /// Per-stream record ordinal in the manifest's exactly-once chain.
    pub stream_record_ordinal: u64,
    /// Global index in the exact node vector passed to the lattice backend.
    pub global_lattice_index: u64,
    /// SHA-256 of the exact committed stream manifest bytes.
    pub segment_manifest_sha256: String,
    /// Segment ordinal within the stream.
    pub segment_ordinal: u64,
    /// Stable output-root-relative segment artifact path.
    pub segment_file: String,
    /// SHA-256 of the complete segment header plus payload.
    pub segment_file_sha256: String,
    /// Byte offset of this record in the segment file.
    pub payload_byte_offset: u64,
    /// Exact record byte length.
    pub byte_length: u32,
    /// SHA-256 of this record's exact bytes.
    pub record_sha256: String,
    /// Record identity fields decoded from the same bytes.
    pub frame_idx: u64,
    pub step_idx: u64,
    pub site_id: u32,
    /// Engine-owned thermodynamic condition identity.
    pub condition_id: String,
    /// Resolved engine protocol phase.
    pub protocol_phase: u8,
    /// Digest of the exact phase schedule used for resolution.
    pub phase_schedule_digest_sha256: String,
    /// Hash of the final engine protocol state.
    pub protocol_state_hash: String,
    /// Exact RichSpike cluster-residue membership decoded from the same record.
    pub cluster_residue_membership: GhostV3ClusterResidueMembership,
}

/// Strict loader result. Nodes cannot be obtained without conservation and
/// per-stream phase binding evidence.
#[derive(Clone, Debug, Serialize)]
pub struct GhostSegmentLatticeLoad {
    /// Projected nodes with global record ordinals in `tile_index`.
    #[serde(skip_serializing)]
    pub nodes: Vec<GhostPhaseLatticeNode>,
    /// One byte-lineage row for each node, in identical vector order.
    pub record_lineage: Vec<GhostRecordByteLineage>,
    /// Per-stream conservation receipts.
    pub streams: Vec<GhostStreamConservation>,
    /// Total engine-produced records.
    pub produced_records: u64,
    /// Total durable records.
    pub persisted_records: u64,
    /// Total decoded records.
    pub decoded_records: u64,
    /// Total projected lineage nodes.
    pub lattice_node_lineage_count: u64,
    /// True only after every node was mapped through an authenticated binding.
    pub phase_protocols_bound_by_stream: bool,
    /// Total equality verdict.
    pub equality_pass: bool,
    /// Total engine-owned staging overflow loss reconstructed across streams.
    pub overflow_loss_count: u64,
    /// True only when every v4 stream carries a complete capacity, drain,
    /// payload, membership, terminal-residual and reconstruction authority.
    pub overflow_conservation_pass: bool,
}

/// Bind the stream's declared protocol hash to the final engine-resident
/// phase boundaries. Wall-bound controllers may rewrite those boundaries, so
/// the pre-run protocol hash alone is not sufficient identity.
pub fn final_engine_protocol_state_hash(
    declared_protocol_hash: &str,
    _current_step: u64,
    total_steps: u64,
    cold_hold_end: u64,
    ramp_end: u64,
    warm_hold_end: u64,
    ramp_down_end: u64,
) -> String {
    let mut hasher = Sha256::new();
    // Runtime progress is deliberately excluded. It is completion evidence,
    // not schedule identity, and including it would make segments emitted
    // before teardown impossible to bind to their immutable phase schedule.
    hasher.update(b"prism-ghost-protocol-schedule-state-v2\0");
    hasher.update((declared_protocol_hash.len() as u64).to_le_bytes());
    hasher.update(declared_protocol_hash.as_bytes());
    for value in [
        total_steps,
        cold_hold_end,
        ramp_end,
        warm_hold_end,
        ramp_down_end,
    ] {
        hasher.update(value.to_le_bytes());
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

// ─── Node ontology ──────────────────────────────────────────────────────────

/// A single Ghost v2 record projected into the 4-D phase lattice.
///
/// This struct is `#[repr(C)]` so the host-side packing matches the on-device
/// view passed to the edge kernel. Total size on x86_64 / sm_120 is 208 B
/// (the directive comment-counts of 16 / 12 / 24 / 20 / 96 are *logical*
/// groupings; the actual byte counts include `#[repr(C)]` padding).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GhostPhaseLatticeNode {
    // ── Identity & provenance ──
    /// Global record ordinal in the authenticated per-stream segment chain.
    pub tile_index: u32,
    /// Source stream id (one stream per `--multi-stream` slot).
    pub stream_id: u16,
    /// `_pad0`: alignment slack so `site_id` lands 4-byte aligned.
    pub _pad0: u16,
    /// Cluster index within the producing frame. Mirrors
    /// [`GhostTileFrame::site_id`].
    pub site_id: u32,
    /// `_pad1`: alignment slack so `frame_idx` lands 8-byte aligned.
    pub _pad1: u32,
    /// Producer-side monotonic frame counter
    /// ([`GhostTileFrame::frame_idx`]).
    pub frame_idx: u64,
    /// Producer-side MD step index — Ghost v2 `step_idx` at offset 152.
    pub step_idx: u64,

    // ── Phase & state ──
    /// Protocol phase ordinal. 0=cold_hold, 1=heating, 2=warm_hold, 3=cooling.
    /// Resolved by the host from `step_idx` via the supplied
    /// [`PhaseSchedule`].
    pub protocol_phase: u8,
    /// `_pad2`: alignment slack so `step_bucket` lands 4-byte aligned.
    pub _pad2: [u8; 3],
    /// `floor(step_idx / step_bucket_size)`. Reused as a temporal coordinate
    /// in the lattice key.
    pub step_bucket: u32,
    /// CCNS phase bin (16-bit ordinal). [`NODE_CCNS_PHASE_NONE`] when the
    /// ccns telemetry plane has not yet been populated for this record.
    pub ccns_phase_bin: u16,
    /// `_pad3`: alignment slack so `gear_id` lands 4-byte aligned.
    pub _pad3: u16,
    /// Active gearbox gear at write time. [`NODE_GEAR_ID_NONE`] when the
    /// `gear_id` field of the v2 record is the Wave A default and the
    /// completeness flag bit is clear.
    pub gear_id: u32,

    // ── Spatial extent (Å) ──
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    /// AABB midpoint by default — labelled `centroid_xyz` here only as an
    /// alias. The phase-manifold-complete centroid is materialised offline
    /// by [`crate::site_manifest::PhaseManifold`] using AABB-volume-weighted
    /// aggregation across the constituent nodes.
    pub centroid_xyz: [f32; 3],

    // ── Thermodynamic & causal state ──
    /// `Σ_planes Δ_AB` total weighted KL divergence at this frame.
    /// Mirrors [`GhostTileFrame::kl_divergence`].
    pub kl_divergence: f32,
    /// `[wd_change, vib_energy]`. Either component may be `NaN` when the
    /// upstream manifold has not yet populated it; the kernel treats `NaN`
    /// as "unavailable_neutral" and defaults the multiplier to 1.0.
    pub thermo_flux: [f32; 2],
    /// Local water density. `NaN` when the prism-therm sidecar has not been
    /// written for this frame.
    pub water_density: f32,
    /// Driver residue id. [`NODE_CAUSAL_LEAD_NONE`] when unresolved.
    pub causal_lead_residue: u32,

    // ── SO(3) spherical manifold ──
    /// 4 planes × 6 bands of rotationally invariant SO(3) power spectrum
    /// `C_l[0..6]`. Plane order: 0=Geometry, 1=Causality, 2=Thermo,
    /// 3=Chemistry.
    pub so3_power_spectrum: [[f32; 6]; 4],
    /// Bitmask: which planes are populated (vs sentinel-zero).
    pub so3_plane_status: u8,
    /// `_pad4`: trailing alignment slack so the struct's total alignment
    /// matches its largest field (`u64`, 8-byte) and downstream
    /// `static_assert` size checks land on a stable footprint.
    pub _pad4: [u8; 7],
}

impl GhostPhaseLatticeNode {
    /// Construct an all-sentinel node — used by the host as a starting state
    /// before populating from a Ghost v2 record.
    #[inline]
    pub const fn empty() -> Self {
        Self {
            tile_index: 0,
            stream_id: 0,
            _pad0: 0,
            site_id: 0,
            _pad1: 0,
            frame_idx: 0,
            step_idx: 0,
            protocol_phase: PHASE_COLD_HOLD,
            _pad2: [0; 3],
            step_bucket: 0,
            ccns_phase_bin: NODE_CCNS_PHASE_NONE,
            _pad3: 0,
            gear_id: NODE_GEAR_ID_NONE,
            aabb_min: [0.0; 3],
            aabb_max: [0.0; 3],
            centroid_xyz: [0.0; 3],
            kl_divergence: f32::NAN,
            thermo_flux: [f32::NAN, f32::NAN],
            water_density: f32::NAN,
            causal_lead_residue: NODE_CAUSAL_LEAD_NONE,
            so3_power_spectrum: [[0.0; 6]; 4],
            so3_plane_status: 0,
            _pad4: [0; 7],
        }
    }

    /// True when the node has *any* SO(3) plane populated (geometry plane is
    /// the minimum guarantee from the v2 producer).
    #[inline]
    pub fn has_any_so3_plane(&self) -> bool {
        self.so3_plane_status != 0
    }

    /// True when the AABB is non-degenerate (every component max >= min).
    #[inline]
    pub fn has_valid_aabb(&self) -> bool {
        self.aabb_max[0] >= self.aabb_min[0]
            && self.aabb_max[1] >= self.aabb_min[1]
            && self.aabb_max[2] >= self.aabb_min[2]
    }

    /// AABB volume in Å³.
    #[inline]
    pub fn aabb_volume(&self) -> f32 {
        let dx = (self.aabb_max[0] - self.aabb_min[0]).max(0.0);
        let dy = (self.aabb_max[1] - self.aabb_min[1]).max(0.0);
        let dz = (self.aabb_max[2] - self.aabb_min[2]).max(0.0);
        dx * dy * dz
    }
}

/// Exact accepted edge emitted at the scoring predicate, before union-find
/// can collapse its endpoints. Rows are canonicalized by endpoint identity on
/// the host before component reconstruction and persistence.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GhostPhaseLatticeAcceptedEdge {
    /// Earlier endpoint in the kernel's `(step_idx, tile_index)` ordering.
    pub from_node_index: u32,
    /// Later endpoint in the kernel's `(step_idx, tile_index)` ordering.
    pub to_node_index: u32,
    pub from_step_idx: u64,
    pub to_step_idx: u64,
    pub temporal_delta_steps: u64,
    /// Composite acceptance score `so3 * causal * thermo`.
    pub total_score: f32,
    /// Active-plane-normalized SO(3) cosine score.
    pub so3_score: f32,
    pub causal_score: f32,
    pub thermo_score: f32,
    pub from_protocol_phase: u8,
    pub to_protocol_phase: u8,
    pub common_so3_plane_mask: u8,
    /// Bit 0 phase-legal, bit 1 temporal-legal, bit 2 AABB-overlap, bit 3
    /// score-threshold accepted. Every persisted row must equal `0x0f`.
    pub predicate_flags: u8,
    pub schema_version: u32,
}

const _: () = {
    use std::mem::{align_of, size_of};
    assert!(size_of::<GhostPhaseLatticeAcceptedEdge>() == GHOST_LATTICE_EDGE_BYTES);
    assert!(align_of::<GhostPhaseLatticeAcceptedEdge>() == 8);
};

/// Exact kernel-to-host edge conservation before durable manifest commit.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GhostLatticeEdgeConservation {
    pub schema_version: u32,
    /// Pass-1 accepted count. This is the exact pass-2 allocation size, not a
    /// candidate-pair upper bound.
    pub counted_edges_pass1: u64,
    /// Rows emitted by the independent pass-2 predicate evaluation.
    pub produced_edges_pass2: u64,
    pub decoded_edges: u64,
    pub indexed_edges: u64,
    pub overflow_count: u64,
    pub counted_equals_produced_equals_decoded_equals_indexed: bool,
    pub canonical_payload_sha256: String,
}

impl Default for GhostLatticeEdgeConservation {
    fn default() -> Self {
        Self {
            schema_version: GHOST_LATTICE_EDGE_SCHEMA_VERSION,
            counted_edges_pass1: 0,
            produced_edges_pass2: 0,
            decoded_edges: 0,
            indexed_edges: 0,
            overflow_count: 0,
            counted_equals_produced_equals_decoded_equals_indexed: true,
            canonical_payload_sha256: ghost_record_sha256(&[]),
        }
    }
}

// Compile-time layout pin — protects the on-device kernel against silent host
// repacking. The C++ mirror in `cuda/ghost_lattice_kernel.cuh` carries an
// identical `static_assert` on its own struct.
const _: () = {
    use std::mem::{align_of, size_of};
    assert!(size_of::<GhostPhaseLatticeNode>() == 208);
    assert!(align_of::<GhostPhaseLatticeNode>() == 8);
};

/// 4-D lattice key used to bucket nodes for the edge kernel.
///
/// `(spatial_cell_x, spatial_cell_y, spatial_cell_z, protocol_phase,
/// step_bucket)` — the cartesian-product of the 3-D spatial cell and the 2-D
/// temporal coordinates (phase + step bucket). Two nodes can only share an
/// edge if their keys are within the configured neighborhood radius (default:
/// 1 spatial cell in each direction, ±1 step bucket, ±1 phase).
#[repr(C)]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct GhostPhaseLatticeKey {
    pub spatial_cell_x: i32,
    pub spatial_cell_y: i32,
    pub spatial_cell_z: i32,
    pub protocol_phase: u8,
    pub _pad: [u8; 3],
    pub step_bucket: u32,
}

// ─── Phase schedule ─────────────────────────────────────────────────────────

/// Protocol-phase resolution policy. The kernel needs a `protocol_phase: u8`
/// per record; this trait is how the host computes it from the v2 record's
/// `step_idx` (and optionally `gear_id`).
pub trait PhaseSchedule: Send + Sync {
    /// Resolve `(step_idx, gear_id)` to a phase ordinal in `0..N_PROTOCOL_PHASES`.
    fn phase_for(&self, step_idx: u64, gear_id: u32) -> u8;
}

/// Quartile-based phase schedule — partitions the observed step-range into
/// four equal quartiles and labels them cold_hold / heating / warm_hold /
/// cooling in that order. Default fallback when the operator has not supplied
/// an explicit protocol schedule.
#[derive(Clone, Copy, Debug)]
pub struct QuartilePhaseSchedule {
    pub min_step: u64,
    pub max_step: u64,
}

impl PhaseSchedule for QuartilePhaseSchedule {
    #[inline]
    fn phase_for(&self, step_idx: u64, _gear_id: u32) -> u8 {
        if self.max_step <= self.min_step {
            return PHASE_COLD_HOLD;
        }
        let span = self.max_step - self.min_step;
        let pos = step_idx.saturating_sub(self.min_step);
        let q = ((pos as u128 * 4u128) / span as u128).min(3) as u8;
        q
    }
}

/// Explicit interval schedule — pass `[(start_step, end_step, phase_ordinal)]`
/// and any `step_idx` inside `[start, end)` resolves to the matching phase.
#[derive(Clone, Debug)]
pub struct IntervalPhaseSchedule {
    pub intervals: Vec<(u64, u64, u8)>,
    pub default_phase: u8,
}

impl PhaseSchedule for IntervalPhaseSchedule {
    fn phase_for(&self, step_idx: u64, _gear_id: u32) -> u8 {
        for &(s, e, p) in &self.intervals {
            if step_idx >= s && step_idx < e {
                return p.min(N_PROTOCOL_PHASES - 1);
            }
        }
        self.default_phase.min(N_PROTOCOL_PHASES - 1)
    }
}

// ─── Ghost v2 → lattice node projection ─────────────────────────────────────

/// Build a [`GhostPhaseLatticeNode`] from a single 4096-byte Ghost v2 record.
///
/// The `record` slice must be exactly [`GHOST_RECORD_BYTES`] long. Returns
/// `None` if the record is `schema_version != 2` (legacy v1 records do not
/// carry the AABB / step_idx fields the lattice needs and must be filtered
/// out upstream).
pub fn project_ghost_v2_record(
    record: &[u8],
    tile_index: u32,
    stream_id: u16,
    schedule: &dyn PhaseSchedule,
    step_bucket_size: u64,
) -> Option<GhostPhaseLatticeNode> {
    if record.len() != GHOST_RECORD_BYTES {
        return None;
    }

    // Schema gate — only v2 records carry AABB + step_idx.
    let schema_version = u32::from_le_bytes(
        record[GHOST_V2_OFFSET_SCHEMA_VERSION..GHOST_V2_OFFSET_SCHEMA_VERSION + 4]
            .try_into()
            .ok()?,
    );
    if schema_version != GHOST_FRAME_SCHEMA_V2 && schema_version != GHOST_FRAME_SCHEMA_V3 {
        return None;
    }

    let mut n = GhostPhaseLatticeNode::empty();

    // ── v1 header fields ──
    n.tile_index = tile_index;
    n.stream_id = stream_id;
    n.frame_idx = u64::from_le_bytes(record[0..8].try_into().ok()?);
    n.site_id = u32::from_le_bytes(record[8..12].try_into().ok()?);
    let telemetry_flags = u16::from_le_bytes(record[14..16].try_into().ok()?);
    n.kl_divergence = f32::from_le_bytes(record[16..20].try_into().ok()?);

    // 4-plane SO(3) power spectrum at offset 20: 24 floats × 4 B each.
    let mut spectrum = [[0.0f32; 6]; 4];
    for plane in 0..4 {
        for band in 0..6 {
            let off = 20 + (plane * 6 + band) * 4;
            spectrum[plane][band] = f32::from_le_bytes(record[off..off + 4].try_into().ok()?);
        }
    }
    n.so3_power_spectrum = spectrum;
    // The saved graph remains immutable. Newly constructed graphs exclude
    // exact producer padding; legacy occupancy remains explicitly unknown in
    // ghost_plane_evidence, rather than being invented from small numbers.
    n.so3_plane_status = crate::ghost_tile::ghost_plane_evidence(record)
        .map(|e| e.cosine_evidence_mask).unwrap_or(0);

    // thermo_flux at offset 116. NaN-tainted records keep the NaN through —
    // the kernel handles missing thermo as "unavailable_neutral" (multiplier
    // 1.0) per directive Part V.
    let tf0 = f32::from_le_bytes(record[116..120].try_into().ok()?);
    let tf1 = f32::from_le_bytes(record[120..124].try_into().ok()?);
    let class_tainted = (telemetry_flags & GHOST_TELEMETRY_CLASS_TAINTED) != 0;
    n.thermo_flux = if class_tainted {
        [f32::NAN, f32::NAN]
    } else {
        [tf0, tf1]
    };
    n.water_density = f32::NAN; // populated downstream from the prism-therm
                                // sidecar; v2 records don't carry it.

    n.causal_lead_residue = u32::from_le_bytes(record[124..128].try_into().ok()?);
    // Native residue indices are zero-based. Only u32::MAX is unavailable;
    // residue zero must survive projection. Historical zero-filled records
    // require producer provenance and are not retroactively reclassified here.

    // ── v2 fields ──
    let dt_fs = f32::from_le_bytes(
        record[GHOST_V2_OFFSET_DT_FS..GHOST_V2_OFFSET_DT_FS + 4]
            .try_into()
            .ok()?,
    );
    let _ = dt_fs; // available but not currently used in adjudication

    let gear_id = u32::from_le_bytes(
        record[GHOST_V2_OFFSET_GEAR_ID..GHOST_V2_OFFSET_GEAR_ID + 4]
            .try_into()
            .ok()?,
    );
    n.gear_id = if gear_id == 0 {
        NODE_GEAR_ID_NONE
    } else {
        gear_id
    };

    let perturbation = record[GHOST_V2_OFFSET_PERTURBATION_CHAN];
    let _ = perturbation;

    n.step_idx = u64::from_le_bytes(
        record[GHOST_V2_OFFSET_STEP_IDX..GHOST_V2_OFFSET_STEP_IDX + 8]
            .try_into()
            .ok()?,
    );

    // AABB only valid when the producer set the SPATIAL_NATIVE bit. Sentinel-
    // zero AABBs collapse to a degenerate point AABB at the origin which the
    // kernel's overlap test will then never accept — exactly the directive's
    // "must not zero-fill missing physical data" rule.
    let fcf = u16::from_le_bytes(
        record[GHOST_V2_OFFSET_FIELD_COMPLETENESS_FLAGS
            ..GHOST_V2_OFFSET_FIELD_COMPLETENESS_FLAGS + 2]
            .try_into()
            .ok()?,
    );
    if (fcf & GHOST_FCF_BIT_SPATIAL_NATIVE_AABB_MIDPOINT) != 0 {
        for i in 0..3 {
            let off_min = GHOST_V2_OFFSET_AABB_MIN + i * 4;
            let off_max = GHOST_V2_OFFSET_AABB_MAX + i * 4;
            let off_cen = GHOST_V2_OFFSET_CENTROID + i * 4;
            n.aabb_min[i] = f32::from_le_bytes(record[off_min..off_min + 4].try_into().ok()?);
            n.aabb_max[i] = f32::from_le_bytes(record[off_max..off_max + 4].try_into().ok()?);
            n.centroid_xyz[i] = f32::from_le_bytes(record[off_cen..off_cen + 4].try_into().ok()?);
        }
    } else {
        // No spatial — the lattice cannot place this record. Caller must
        // discard nodes for which `has_valid_aabb` returns false (the
        // empty AABB will collapse to a zero-volume point that fails every
        // overlap test, but pre-filtering on the host avoids burning a
        // cell key on an unplaceable record).
        return None;
    }

    // ── Phase resolution ──
    n.protocol_phase = schedule.phase_for(n.step_idx, n.gear_id);
    n.step_bucket = (n.step_idx / step_bucket_size.max(1)) as u32;

    Some(n)
}

/// Project an entire `_stream{:02}_ghost_tiles.bin` payload (excluding the
/// 4096 B counter sector) into a vector of lattice nodes. Records that fail
/// projection (legacy v1, sentinel AABB, malformed) are skipped — the
/// returned vector contains only placeable nodes.
pub fn project_ghost_v2_payload(
    payload: &[u8],
    stream_id: u16,
    schedule: &dyn PhaseSchedule,
    step_bucket_size: u64,
) -> Vec<GhostPhaseLatticeNode> {
    let n_records = payload.len() / GHOST_RECORD_BYTES;
    let mut out = Vec::with_capacity(n_records);
    for i in 0..n_records {
        let off = i * GHOST_RECORD_BYTES;
        let record = &payload[off..off + GHOST_RECORD_BYTES];
        if let Some(node) =
            project_ghost_v2_record(record, i as u32, stream_id, schedule, step_bucket_size)
        {
            out.push(node);
        }
    }
    out
}

// ─── Backend ────────────────────────────────────────────────────────────────

/// Configuration for [`GhostPhaseLattice4D`]. All fields are physical
/// parameters that the directive Part III specifies as backend defaults; the
/// operator overrides them via CLI flags `--ghost-phase-lattice-*`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GhostPhaseLatticeConfig {
    pub spatial_cell_size_a: f32,
    pub max_temporal_edge_steps: u64,
    pub step_bucket_size: u64,
    pub so3_threshold: f32,
}

impl Default for GhostPhaseLatticeConfig {
    fn default() -> Self {
        Self {
            spatial_cell_size_a: DEFAULT_SPATIAL_CELL_A,
            max_temporal_edge_steps: DEFAULT_MAX_TEMPORAL_EDGE_STEPS,
            step_bucket_size: DEFAULT_STEP_BUCKET_SIZE,
            so3_threshold: DEFAULT_SO3_THRESHOLD,
        }
    }
}

/// One connected component returned by the backend, carrying the indices of
/// the constituent [`GhostPhaseLatticeNode`]s plus aggregate edge telemetry
/// for the [`So3Manifold`] block.
#[derive(Clone, Debug)]
pub struct GhostPhaseLatticeComponent {
    /// Indices into the input slice — every node in this group passed the
    /// boolean constraints + scoring threshold against at least one other
    /// node in the group (or is a singleton).
    pub node_indices: Vec<u32>,
    /// Number of accepted edges that fell *inside* this component
    /// (intra-component edge count). Used by the [`So3Manifold`] block to
    /// compute the mean-cosine summary.
    pub n_intra_edges: u32,
    /// Sum of `total_edge_score` over accepted intra-component edges. The
    /// mean cosine emitted into `so3_manifold.intra_component_mean_cosine`
    /// is `intra_edge_score_sum / max(1, n_intra_edges)`.
    pub intra_edge_score_sum: f64,
    /// True only when edge endpoints and scores were retained exactly for
    /// this component. False forbids phase-resolved or component-resolved
    /// cosine claims from aggregate global counters.
    pub edge_lineage_exact: bool,
}

/// Backend-level summary aggregating edge-construction telemetry. Surfaced
/// in the per-site `ghost_phase_lattice` provenance block (directive 4.1).
#[derive(Clone, Debug, Default)]
pub struct GhostPhaseLatticeRunStats {
    pub n_nodes: u32,
    pub n_lattice_cells: u32,
    pub n_pairs_evaluated: u64,
    pub n_pairs_phase_legal: u64,
    pub n_pairs_aabb_overlap: u64,
    pub n_directed_edges: u64,
    pub kernel_elapsed_ms: f64,
    pub host_elapsed_ms: f64,
    pub min_step_idx: u64,
    pub max_step_idx: u64,
    pub phases_present: u8, // bitmask of phases that appeared in the input
}

/// Outcome of a [`GhostPhaseLattice4D::cluster`] call.
#[derive(Clone, Debug)]
pub struct GhostPhaseLatticeOutcome {
    pub components: Vec<GhostPhaseLatticeComponent>,
    /// Canonically sorted, exact accepted-edge rows. This is the sole
    /// connectivity authority for component and Path-B reconstruction.
    pub accepted_edges: Vec<GhostPhaseLatticeAcceptedEdge>,
    pub edge_conservation: GhostLatticeEdgeConservation,
    pub stats: GhostPhaseLatticeRunStats,
    pub config: GhostPhaseLatticeConfig,
}

/// Durable accepted-edge artifact consumed by Native Lattice, Path-B and
/// fiber materialization. It binds exact edge rows to the authenticated node
/// byte-lineage sidecar rather than permitting connectivity reconstruction
/// from component aggregates.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GhostLatticeAcceptedEdgeArtifact {
    pub schema_kind: String,
    pub schema_version: u32,
    pub run_id: String,
    pub target: String,
    pub node_lineage_artifact_sha256: String,
    pub config: GhostPhaseLatticeConfig,
    pub counted_edges_pass1: u64,
    pub produced_edges: u64,
    pub persisted_edges: u64,
    pub decoded_edges: u64,
    pub indexed_edges: u64,
    pub counted_equals_produced_equals_persisted_equals_decoded_equals_indexed: bool,
    pub produced_equals_persisted_equals_decoded_equals_indexed: bool,
    pub canonical_payload_sha256: String,
    pub edges: Vec<GhostPhaseLatticeAcceptedEdge>,
}

impl GhostLatticeAcceptedEdgeArtifact {
    pub fn from_outcome(
        run_id: &str,
        target: &str,
        node_lineage_artifact_sha256: &str,
        nodes: &[GhostPhaseLatticeNode],
        outcome: &GhostPhaseLatticeOutcome,
    ) -> Result<Self> {
        if run_id.trim().is_empty() || target.trim().is_empty() {
            bail!("accepted lattice-edge artifact requires run and target identity");
        }
        require_sha256(node_lineage_artifact_sha256, "node lineage artifact")?;
        validate_accepted_edges(
            nodes,
            &outcome.accepted_edges,
            outcome.config.so3_threshold,
            outcome.config.max_temporal_edge_steps,
        )?;
        let count = outcome.accepted_edges.len() as u64;
        if !outcome
            .edge_conservation
            .counted_equals_produced_equals_decoded_equals_indexed
            || outcome.stats.n_directed_edges != count
            || outcome.edge_conservation.counted_edges_pass1 != count
            || outcome.edge_conservation.produced_edges_pass2 != count
            || outcome.edge_conservation.decoded_edges != count
            || outcome.edge_conservation.indexed_edges != count
        {
            bail!("accepted lattice-edge outcome conservation is not closed");
        }
        let artifact = Self {
            schema_kind: "prism_ghost_lattice_accepted_edges".into(),
            schema_version: GHOST_LATTICE_EDGE_ARTIFACT_SCHEMA_VERSION,
            run_id: run_id.into(),
            target: target.into(),
            node_lineage_artifact_sha256: node_lineage_artifact_sha256.into(),
            config: outcome.config,
            counted_edges_pass1: count,
            produced_edges: count,
            persisted_edges: count,
            decoded_edges: count,
            indexed_edges: count,
            counted_equals_produced_equals_persisted_equals_decoded_equals_indexed: true,
            produced_equals_persisted_equals_decoded_equals_indexed: true,
            canonical_payload_sha256: ghost_record_sha256(&encode_accepted_edges(
                &outcome.accepted_edges,
            )),
            edges: outcome.accepted_edges.clone(),
        };
        artifact.verify(nodes)?;
        Ok(artifact)
    }

    pub fn verify(&self, nodes: &[GhostPhaseLatticeNode]) -> Result<()> {
        require_sha256(&self.node_lineage_artifact_sha256, "node lineage artifact")?;
        require_sha256(&self.canonical_payload_sha256, "accepted edge payload")?;
        validate_accepted_edges(
            nodes,
            &self.edges,
            self.config.so3_threshold,
            self.config.max_temporal_edge_steps,
        )?;
        let count = self.edges.len() as u64;
        let unique = self
            .edges
            .iter()
            .map(|edge| (edge.from_node_index, edge.to_node_index))
            .collect::<BTreeSet<_>>()
            .len() as u64;
        if self.schema_kind != "prism_ghost_lattice_accepted_edges"
            || self.schema_version != GHOST_LATTICE_EDGE_ARTIFACT_SCHEMA_VERSION
            || self.run_id.trim().is_empty()
            || self.target.trim().is_empty()
            || !self.counted_equals_produced_equals_persisted_equals_decoded_equals_indexed
            || !self.produced_equals_persisted_equals_decoded_equals_indexed
            || self.counted_edges_pass1 != count
            || self.produced_edges != count
            || self.persisted_edges != count
            || self.decoded_edges != count
            || self.indexed_edges != count
            || unique != count
            || self.canonical_payload_sha256
                != ghost_record_sha256(&encode_accepted_edges(&self.edges))
        {
            bail!("accepted lattice-edge durable conservation/hash verification failed");
        }
        Ok(())
    }
}

/// Load and verify a durable accepted-edge artifact against the exact lattice
/// node vector. No component or transition consumer may bypass this check.
pub fn load_verified_ghost_lattice_edges(
    path: &Path,
    nodes: &[GhostPhaseLatticeNode],
    expected_artifact_sha256: &str,
    expected_artifact_bytes: u64,
    expected_node_lineage_artifact_sha256: &str,
) -> Result<GhostLatticeAcceptedEdgeArtifact> {
    require_sha256(expected_artifact_sha256, "accepted edge artifact")?;
    require_sha256(
        expected_node_lineage_artifact_sha256,
        "expected node lineage artifact",
    )?;
    let metadata = std::fs::symlink_metadata(path).with_context(|| {
        format!(
            "stat accepted lattice-edge artifact without following symlink {}",
            path.display()
        )
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        bail!(
            "accepted lattice-edge artifact must be a regular non-symlink file: {}",
            path.display()
        );
    }
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)
        .with_context(|| {
            format!(
                "open accepted lattice-edge artifact with O_NOFOLLOW {}",
                path.display()
            )
        })?;
    let opened_metadata = file.metadata().with_context(|| {
        format!(
            "fstat opened accepted lattice-edge artifact {}",
            path.display()
        )
    })?;
    if !opened_metadata.is_file() || opened_metadata.len() != expected_artifact_bytes {
        bail!(
            "accepted lattice-edge opened-file identity mismatch: expected_regular=true expected_bytes={} actual_regular={} actual_bytes={}",
            expected_artifact_bytes,
            opened_metadata.is_file(),
            opened_metadata.len(),
        );
    }
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .with_context(|| format!("read accepted lattice-edge artifact {}", path.display()))?;
    if bytes.len() as u64 != expected_artifact_bytes {
        bail!(
            "accepted lattice-edge read length mismatch: expected={} actual={}",
            expected_artifact_bytes,
            bytes.len(),
        );
    }
    let actual_artifact_sha256 = ghost_record_sha256(&bytes);
    if actual_artifact_sha256 != expected_artifact_sha256 {
        bail!(
            "accepted lattice-edge artifact SHA-256 mismatch: expected={} actual={}",
            expected_artifact_sha256,
            actual_artifact_sha256,
        );
    }
    let artifact: GhostLatticeAcceptedEdgeArtifact = serde_json::from_slice(&bytes)
        .with_context(|| format!("decode accepted lattice-edge artifact {}", path.display()))?;
    if artifact.node_lineage_artifact_sha256 != expected_node_lineage_artifact_sha256 {
        bail!(
            "accepted lattice-edge node-lineage SHA-256 mismatch: expected={} declared={}",
            expected_node_lineage_artifact_sha256,
            artifact.node_lineage_artifact_sha256,
        );
    }
    artifact.verify(nodes)?;
    let mut canonical = serde_json::to_vec_pretty(&artifact)
        .context("canonicalize accepted lattice-edge artifact")?;
    canonical.push(b'\n');
    if bytes != canonical {
        bail!(
            "accepted lattice-edge artifact is not the canonical writer serialization: {}",
            path.display()
        );
    }
    Ok(artifact)
}

/// Public, hash-bound component reconstruction consumed by Native Path-B.
/// Components are derived solely from verified exact edge rows; aggregate GPU
/// counters are never treated as connectivity authority.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GhostExactLatticeComponent {
    pub schema_version: u32,
    pub component_digest_sha256: String,
    pub node_lineage_artifact_sha256: String,
    pub accepted_edge_payload_sha256: String,
    pub node_indices: Vec<u32>,
    pub edge_indices: Vec<u64>,
    pub n_intra_edges: u64,
    pub intra_edge_score_sum: f64,
}

/// Reconstruct canonical component memberships and stable component digests
/// from a verified accepted-edge artifact.
pub fn reconstruct_verified_lattice_components(
    nodes: &[GhostPhaseLatticeNode],
    artifact: &GhostLatticeAcceptedEdgeArtifact,
) -> Result<Vec<GhostExactLatticeComponent>> {
    artifact.verify(nodes)?;
    let mut raw = components_from_exact_edges(nodes.len(), &artifact.edges)?;
    raw.sort_by_key(|component| component.node_indices.first().copied().unwrap_or(u32::MAX));
    raw.into_iter()
        .map(|component| {
            let members = component
                .node_indices
                .iter()
                .copied()
                .collect::<BTreeSet<_>>();
            let edge_indices = artifact
                .edges
                .iter()
                .enumerate()
                .filter(|(_, edge)| {
                    members.contains(&edge.from_node_index) && members.contains(&edge.to_node_index)
                })
                .map(|(index, _)| index as u64)
                .collect::<Vec<_>>();
            if edge_indices.len() as u32 != component.n_intra_edges {
                bail!("exact lattice component edge-index conservation failed");
            }
            let digest_payload = serde_json::json!({
                "schema_kind": "prism_ghost_exact_lattice_component",
                "schema_version": GHOST_LATTICE_EDGE_ARTIFACT_SCHEMA_VERSION,
                "node_lineage_artifact_sha256": artifact.node_lineage_artifact_sha256,
                "accepted_edge_payload_sha256": artifact.canonical_payload_sha256,
                "node_indices": component.node_indices,
                "edge_indices": edge_indices,
            });
            Ok(GhostExactLatticeComponent {
                schema_version: GHOST_LATTICE_EDGE_ARTIFACT_SCHEMA_VERSION,
                component_digest_sha256: ghost_record_sha256(&serde_json::to_vec(&digest_payload)?),
                node_lineage_artifact_sha256: artifact.node_lineage_artifact_sha256.clone(),
                accepted_edge_payload_sha256: artifact.canonical_payload_sha256.clone(),
                node_indices: component.node_indices,
                edge_indices,
                n_intra_edges: u64::from(component.n_intra_edges),
                intra_edge_score_sum: component.intra_edge_score_sum,
            })
        })
        .collect()
}

// FFI surface to the static-archived two-pass host orchestrator defined in
// `cuda/ghost_lattice_kernel.cu`. Production has no runtime NVRTC
// substitution path: static launch failure is terminal unless a future,
// independently hash-bound equivalence contract is added.
#[cfg(feature = "gpu")]
extern "C" {
    /// Run the full edge-adjudication + union-find + path-compression
    /// pipeline on the device. Returns 0 on success, non-zero on a kernel
    /// launch / sync failure (the integer value mirrors a `cudaError_t`).
    ///
    /// All pointers are device pointers obtained via cudarc allocations.
    /// `nodes_dev` points to `n_nodes * sizeof(GhostPhaseLatticeNode)`
    /// bytes; the kernel reads it but does not mutate it.
    /// `cell_first_dev` / `cell_count_dev` / `permutation_dev` are the
    /// host-built lattice index tables (one entry per non-empty cell;
    /// permutation maps cell-sorted slot → original node index).
    /// `parent_dev` is the union-find parent array (n_nodes ints).
    /// `edge_score_sum_dev` and `edge_count_dev` are 1-element buffers
    /// for atomicAdd-accumulated intra-component edge telemetry.
    pub fn prism_ghost_phase_lattice_count(
        nodes_dev: u64,
        n_nodes: u32,
        permutation_dev: u64,
        cell_first_dev: u64,
        cell_count_dev: u64,
        n_cells: u32,
        cell_table_dev: u64, // packed [cx,cy,cz,phase,step_bucket] per cell
        parent_dev: u64,
        edge_score_sum_dev: u64,
        edge_count_dev: u64,
        pair_count_dev: u64,
        phase_legal_count_dev: u64,
        aabb_overlap_count_dev: u64,
        spatial_cell_size_a: f32,
        max_temporal_edge_steps: u64,
        so3_threshold: f32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Re-evaluate the identical edge predicate and write into the exact
    /// pass-1-sized row allocation. This pass does not mutate union-find.
    pub fn prism_ghost_phase_lattice_write(
        nodes_dev: u64,
        n_nodes: u32,
        permutation_dev: u64,
        cell_first_dev: u64,
        cell_count_dev: u64,
        n_cells: u32,
        cell_table_dev: u64,
        edge_score_sum_dev: u64,
        edge_count_dev: u64,
        accepted_edges_dev: u64,
        accepted_edge_capacity: u64,
        accepted_edge_overflow_dev: u64,
        spatial_cell_size_a: f32,
        max_temporal_edge_steps: u64,
        so3_threshold: f32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
}

/// GPU-resident edge-adjudication backend. Implements the directive's Part II
/// boolean + continuous physics scoring against the static archived kernel.
#[cfg(feature = "gpu")]
pub struct GhostPhaseLattice4D {
    _context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    config: GhostPhaseLatticeConfig,
}

#[cfg(feature = "gpu")]
impl GhostPhaseLattice4D {
    /// Construct the static-archive backend.
    pub fn new(
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        config: GhostPhaseLatticeConfig,
    ) -> Self {
        Self {
            _context: context,
            stream,
            config,
        }
    }

    /// Admit only the explicit additional payload at this allocator boundary.
    /// CUDA reports the current context/client view (including an MPS quota
    /// when present), not the supervisor's independent physical NVML free.
    /// This is not an exclusive reservation or an allocator-overhead bound.
    fn admit_additional_device_payload(
        &self,
        boundary: &str,
        allocations: &[(&str, u64)],
    ) -> Result<()> {
        const CLIENT_RESERVE_BYTES: u64 = 3 * 1024 * 1024 * 1024;
        let additional = allocations.iter().try_fold(0u64, |sum, (_, bytes)| {
            sum.checked_add(*bytes).context("Ghost lattice additional payload sum overflow")
        })?;
        let required = additional.checked_add(CLIENT_RESERVE_BYTES)
            .context("Ghost lattice payload plus client reserve overflow")?;
        self._context.bind_to_thread()
            .context("bind Ghost lattice allocation-admission context")?;
        let (free, total) = cudarc::driver::result::mem_get_info()
            .context("query Ghost lattice current-context/client-visible CUDA memory")?;
        let free = u64::try_from(free).context("CUDA free bytes exceed u64")?;
        let total = u64::try_from(total).context("CUDA total bytes exceed u64")?;
        let admitted = free >= required;
        log::info!("GHOST_LATTICE_ALLOCATION_ADMISSION {}", serde_json::json!({
            "schema": "prism.ghost_lattice_allocation_admission.v1",
            "boundary": boundary,
            "pid": std::process::id(),
            "device_ordinal": self._context.ordinal(),
            "memory_view": "CUDA_current_context_client_visible_not_physical_NVML",
            "MPS_quota_visibility": "client quota view when MPS is active; no physical-free inference",
            "current_free_bytes": free,
            "current_reported_total_bytes": total,
            "additional_payload_ledger": allocations,
            "additional_requested_payload_bytes": additional,
            "client_reserve_bytes": CLIENT_RESERVE_BYTES,
            "required_current_free_bytes": required,
            "admitted": admitted,
            "already_live_allocations": "included in observed usage; not charged again",
            "unknowns": ["allocator rounding and driver workspaces", "concurrent external allocation", "later SDST and host materialization"],
            "exclusive_reservation": false,
            "whole_flow_memory_fit_qualified": false
        }));
        anyhow::ensure!(admitted,
            "Ghost lattice native allocation admission refused at {}: current-context/client-visible free={} additional_payload={} reserve={} required={}; physical NVML free is independently supervised; no truncation, alternate clustering, or durable MD data deletion",
            boundary, free, additional, CLIENT_RESERVE_BYTES, required);
        Ok(())
    }

    /// Run the full pipeline against `nodes`. Empty input returns an empty
    /// outcome (no kernel launches).
    pub fn cluster(&mut self, nodes: &[GhostPhaseLatticeNode]) -> Result<GhostPhaseLatticeOutcome> {
        let t_total = std::time::Instant::now();
        if nodes.is_empty() {
            return Ok(GhostPhaseLatticeOutcome {
                components: Vec::new(),
                accepted_edges: Vec::new(),
                edge_conservation: GhostLatticeEdgeConservation::default(),
                stats: GhostPhaseLatticeRunStats::default(),
                config: self.config,
            });
        }

        // ── Step 1. Build host-side lattice cell tables ──────────────
        let t_host = std::time::Instant::now();
        let (permutation, cell_first, cell_count, cell_table, span) =
            self.build_lattice_cells(nodes);
        let n_nodes = u32::try_from(nodes.len())
            .context("Ghost lattice node population exceeds native u32 ABI; no truncation")?;
        let n_cells = u32::try_from(cell_first.len())
            .context("Ghost lattice cell population exceeds native u32 ABI; no truncation")?;
        log::info!(
            "  [GHOST-LATTICE-4D] lattice build: nodes={} cells={} step_span=[{}, {}] | {} ms",
            n_nodes,
            n_cells,
            span.min_step,
            span.max_step,
            t_host.elapsed().as_millis()
        );

        // ── Step 2. Upload to device ─────────────────────────────────
        let t_upload = std::time::Instant::now();
        let payload_bytes = |elements: usize, element_bytes: usize| -> Result<u64> {
            u64::try_from(elements).context("Ghost lattice element count exceeds u64")?
                .checked_mul(u64::try_from(element_bytes).context("Ghost lattice element width exceeds u64")?)
                .context("Ghost lattice requested payload byte count overflow")
        };
        let initial_allocations = [
            ("nodes", payload_bytes(nodes.len(), std::mem::size_of::<GhostPhaseLatticeNode>())?),
            ("permutation", payload_bytes(permutation.len(), std::mem::size_of::<u32>())?),
            ("cell_first", payload_bytes(cell_first.len(), std::mem::size_of::<u32>())?),
            ("cell_count", payload_bytes(cell_count.len(), std::mem::size_of::<u32>())?),
            ("cell_table", payload_bytes(cell_table.len(), std::mem::size_of::<i32>())?),
            ("parent", payload_bytes(nodes.len(), std::mem::size_of::<i32>())?),
            ("pass1_edge_score_sum", std::mem::size_of::<u64>() as u64),
            ("pass1_edge_count", std::mem::size_of::<u64>() as u64),
            ("pass1_pair_count", std::mem::size_of::<u64>() as u64),
            ("pass1_phase_legal", std::mem::size_of::<u64>() as u64),
            ("pass1_aabb_overlap", std::mem::size_of::<u64>() as u64),
        ];
        self.admit_additional_device_payload("before_initial_upload", &initial_allocations)?;
        let nodes_bytes_len = usize::try_from(initial_allocations[0].1)
            .context("Ghost lattice node upload exceeds host address space")?;
        let nodes_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                nodes.as_ptr() as *const u8,
                nodes_bytes_len,
            )
        };
        let d_nodes: CudaSlice<u8> = self
            .stream
            .memcpy_stod(nodes_bytes)
            .context("upload lattice nodes")?;
        let d_permutation: CudaSlice<u32> = self
            .stream
            .memcpy_stod(&permutation)
            .context("upload permutation")?;
        let d_cell_first: CudaSlice<u32> = self
            .stream
            .memcpy_stod(&cell_first)
            .context("upload cell_first")?;
        let d_cell_count: CudaSlice<u32> = self
            .stream
            .memcpy_stod(&cell_count)
            .context("upload cell_count")?;
        let d_cell_table: CudaSlice<i32> = self
            .stream
            .memcpy_stod(&cell_table)
            .context("upload cell_table")?;
        let mut d_parent: CudaSlice<i32> = unsafe {
            self.stream
                .alloc::<i32>(nodes.len())
                .context("alloc parent")?
        };
        let mut d_edge_score_sum: CudaSlice<u64> = self
            .stream
            .memcpy_stod(&[0u64; 1])
            .context("alloc score sum")?;
        let mut d_edge_count: CudaSlice<u64> = self
            .stream
            .memcpy_stod(&[0u64; 1])
            .context("alloc edge count")?;
        let mut d_pair_count: CudaSlice<u64> = self
            .stream
            .memcpy_stod(&[0u64; 1])
            .context("alloc pair count")?;
        let mut d_phase_legal: CudaSlice<u64> = self
            .stream
            .memcpy_stod(&[0u64; 1])
            .context("alloc phase legal count")?;
        let mut d_aabb_overlap: CudaSlice<u64> = self
            .stream
            .memcpy_stod(&[0u64; 1])
            .context("alloc aabb overlap count")?;
        log::info!(
            "  [GHOST-LATTICE-4D] device upload: {} ms ({} KB nodes + {} KB cells)",
            t_upload.elapsed().as_millis(),
            nodes_bytes.len() / 1024,
            (cell_first.len() + cell_count.len() + cell_table.len() / 5) * 4 / 1024
        );

        // ── Step 3. Launch kernels (try static-archive FFI first; fall
        //            back to NVRTC if the symbol isn't linked) ────────
        let t_kernel = std::time::Instant::now();
        let static_rc = {
            // Each `.device_ptr(stream)` call returns a (raw_ptr, guard)
            // tuple where the guard pins the slice's lifetime against the
            // stream's outstanding work. Hold every guard in scope until
            // after the FFI call returns so cudarc doesn't reclaim the
            // allocation while the kernel is mid-launch.
            let (p_nodes, _g_nodes) = d_nodes.device_ptr(&self.stream);
            let (p_perm, _g_perm) = d_permutation.device_ptr(&self.stream);
            let (p_cf, _g_cf) = d_cell_first.device_ptr(&self.stream);
            let (p_cc, _g_cc) = d_cell_count.device_ptr(&self.stream);
            let (p_ct, _g_ct) = d_cell_table.device_ptr(&self.stream);
            let (p_par, _g_par) = d_parent.device_ptr_mut(&self.stream);
            let (p_score, _g_score) = d_edge_score_sum.device_ptr_mut(&self.stream);
            let (p_ec, _g_ec) = d_edge_count.device_ptr_mut(&self.stream);
            let (p_pc, _g_pc) = d_pair_count.device_ptr_mut(&self.stream);
            let (p_pl, _g_pl) = d_phase_legal.device_ptr_mut(&self.stream);
            let (p_ao, _g_ao) = d_aabb_overlap.device_ptr_mut(&self.stream);
            let raw_stream = self.stream.cu_stream() as *mut std::ffi::c_void;
            unsafe {
                prism_ghost_phase_lattice_count(
                    p_nodes,
                    n_nodes,
                    p_perm,
                    p_cf,
                    p_cc,
                    n_cells,
                    p_ct,
                    p_par,
                    p_score,
                    p_ec,
                    p_pc,
                    p_pl,
                    p_ao,
                    self.config.spatial_cell_size_a,
                    self.config.max_temporal_edge_steps,
                    self.config.so3_threshold,
                    raw_stream,
                )
            }
        };

        if static_rc != 0 {
            bail!(
                "production static Ghost lattice pass-1 count kernel failed rc={}; NVRTC substitution is forbidden without a hash-bound equivalence receipt",
                static_rc,
            );
        }

        self.stream
            .synchronize()
            .context("sync lattice pass-1 count kernel")?;
        let mut pass1_count_host = [0u64; 1];
        let mut pass1_score_sum_host = [0u64; 1];
        self.stream
            .memcpy_dtoh(&d_edge_count, &mut pass1_count_host)
            .context("download pass-1 accepted-edge count")?;
        self.stream
            .memcpy_dtoh(&d_edge_score_sum, &mut pass1_score_sum_host)
            .context("download pass-1 accepted-edge score sum")?;
        self.stream
            .synchronize()
            .context("complete lattice pass-1 count/score readbacks before exact edge allocation")?;

        // Pass 2 allocates exactly the pass-1 accepted count, then repeats
        // the identical predicate to emit rows. This prevents saturated
        // candidate neighborhoods from forcing candidate-pair-sized VRAM.
        let accepted_edge_capacity = pass1_count_host[0];
        let accepted_edge_bytes = accepted_edge_capacity
            .checked_mul(GHOST_LATTICE_EDGE_BYTES as u64)
            .context("accepted lattice-edge exact byte count overflow")?;
        let accepted_edge_allocation_bytes: usize = accepted_edge_bytes
            .max(GHOST_LATTICE_EDGE_BYTES as u64)
            .try_into()
            .context("accepted lattice-edge exact allocation exceeds host address space")?;
        // Count/score readback above already supplies exact E. Existing N/C
        // allocations remain live and are represented in current free bytes.
        // Charge only the new exact row buffer and three pass-2 counters.
        self.admit_additional_device_payload("after_pass1_before_exact_edge_allocation", &[
            ("accepted_edges", u64::try_from(accepted_edge_allocation_bytes)
                .context("Ghost lattice edge allocation exceeds u64")?),
            ("pass2_edge_count", std::mem::size_of::<u64>() as u64),
            ("pass2_edge_score_sum", std::mem::size_of::<u64>() as u64),
            ("pass2_accepted_edge_overflow", std::mem::size_of::<u64>() as u64),
        ])?;
        let mut d_accepted_edges: CudaSlice<u8> = self
            .stream
            .alloc_zeros(accepted_edge_allocation_bytes)
            .with_context(|| {
                format!(
                    "allocate exact pass-1 accepted-edge rows={} bytes={}",
                    accepted_edge_capacity, accepted_edge_bytes,
                )
            })?;
        let mut d_pass2_edge_count: CudaSlice<u64> = self
            .stream
            .memcpy_stod(&[0u64; 1])
            .context("alloc pass-2 edge count")?;
        let mut d_pass2_edge_score_sum: CudaSlice<u64> = self
            .stream
            .memcpy_stod(&[0u64; 1])
            .context("alloc pass-2 edge score sum")?;
        let mut d_accepted_edge_overflow: CudaSlice<u64> = self
            .stream
            .memcpy_stod(&[0u64; 1])
            .context("alloc pass-2 accepted-edge overflow counter")?;
        let static_write_rc = {
            let (p_nodes, _g_nodes) = d_nodes.device_ptr(&self.stream);
            let (p_perm, _g_perm) = d_permutation.device_ptr(&self.stream);
            let (p_cf, _g_cf) = d_cell_first.device_ptr(&self.stream);
            let (p_cc, _g_cc) = d_cell_count.device_ptr(&self.stream);
            let (p_ct, _g_ct) = d_cell_table.device_ptr(&self.stream);
            let (p_score, _g_score) = d_pass2_edge_score_sum.device_ptr_mut(&self.stream);
            let (p_ec, _g_ec) = d_pass2_edge_count.device_ptr_mut(&self.stream);
            let (p_edges, _g_edges) = d_accepted_edges.device_ptr_mut(&self.stream);
            let (p_overflow, _g_overflow) = d_accepted_edge_overflow.device_ptr_mut(&self.stream);
            let raw_stream = self.stream.cu_stream() as *mut std::ffi::c_void;
            unsafe {
                prism_ghost_phase_lattice_write(
                    p_nodes,
                    n_nodes,
                    p_perm,
                    p_cf,
                    p_cc,
                    n_cells,
                    p_ct,
                    p_score,
                    p_ec,
                    p_edges,
                    accepted_edge_capacity,
                    p_overflow,
                    self.config.spatial_cell_size_a,
                    self.config.max_temporal_edge_steps,
                    self.config.so3_threshold,
                    raw_stream,
                )
            }
        };
        if static_write_rc != 0 {
            bail!(
                "production static Ghost lattice pass-2 exact-write kernel failed rc={}; NVRTC substitution is forbidden without a hash-bound equivalence receipt",
                static_write_rc,
            );
        }
        self.stream
            .synchronize()
            .context("sync lattice pass-2 exact-write kernel")?;
        let kernel_ms = t_kernel.elapsed().as_secs_f64() * 1000.0;
        log::info!("  [GHOST-LATTICE-4D] kernel: {:.2} ms", kernel_ms);

        // ── Step 4. Download parent + edge telemetry ────────────────
        let t_dl = std::time::Instant::now();
        let mut parent_host: Vec<i32> = vec![0; nodes.len()];
        self.stream
            .memcpy_dtoh(&d_parent, &mut parent_host)
            .context("download parent")?;
        let mut edge_score_sum_host = [0u64; 1];
        let mut edge_count_host = [0u64; 1];
        let mut pass2_edge_score_sum_host = [0u64; 1];
        let mut pass2_edge_count_host = [0u64; 1];
        let mut accepted_edge_overflow_host = [0u64; 1];
        let mut pair_count_host = [0u64; 1];
        let mut phase_legal_host = [0u64; 1];
        let mut aabb_overlap_host = [0u64; 1];
        self.stream
            .memcpy_dtoh(&d_edge_score_sum, &mut edge_score_sum_host)?;
        self.stream
            .memcpy_dtoh(&d_edge_count, &mut edge_count_host)?;
        self.stream
            .memcpy_dtoh(&d_pass2_edge_score_sum, &mut pass2_edge_score_sum_host)?;
        self.stream
            .memcpy_dtoh(&d_pass2_edge_count, &mut pass2_edge_count_host)?;
        self.stream
            .memcpy_dtoh(&d_accepted_edge_overflow, &mut accepted_edge_overflow_host)?;
        self.stream
            .memcpy_dtoh(&d_pair_count, &mut pair_count_host)?;
        self.stream
            .memcpy_dtoh(&d_phase_legal, &mut phase_legal_host)?;
        self.stream
            .memcpy_dtoh(&d_aabb_overlap, &mut aabb_overlap_host)?;
        self.stream
            .synchronize()
            .context("complete lattice pass-2 counter readbacks before conservation checks")?;
        if accepted_edge_overflow_host[0] != 0
            || edge_count_host[0] != pass1_count_host[0]
            || pass2_edge_count_host[0] != accepted_edge_capacity
            || edge_score_sum_host[0] != pass1_score_sum_host[0]
            || pass2_edge_score_sum_host[0] != pass1_score_sum_host[0]
        {
            bail!(
                "accepted Ghost lattice two-pass conservation failed: pass1={} pass2={} capacity={} pass1_score={} pass2_score={} overflow={}",
                edge_count_host[0],
                pass2_edge_count_host[0],
                accepted_edge_capacity,
                edge_score_sum_host[0],
                pass2_edge_score_sum_host[0],
                accepted_edge_overflow_host[0],
            );
        }
        let mut accepted_edge_bytes_host = vec![0u8; accepted_edge_allocation_bytes];
        self.stream
            .memcpy_dtoh(&d_accepted_edges, &mut accepted_edge_bytes_host)
            .context("download exact accepted lattice-edge rows")?;
        self.stream
            .synchronize()
            .context("complete lattice accepted-edge readback before host decoding")?;
        let accepted_payload_len: usize = pass2_edge_count_host[0]
            .checked_mul(GHOST_LATTICE_EDGE_BYTES as u64)
            .context("accepted edge payload length overflow")?
            .try_into()
            .context("accepted edge payload exceeds host address space")?;
        let mut accepted_edges =
            decode_accepted_edges(&accepted_edge_bytes_host[..accepted_payload_len])?;
        accepted_edges.sort_by_key(|edge| {
            (
                edge.from_step_idx,
                edge.from_node_index,
                edge.to_step_idx,
                edge.to_node_index,
            )
        });
        validate_accepted_edges(
            nodes,
            &accepted_edges,
            self.config.so3_threshold,
            self.config.max_temporal_edge_steps,
        )?;
        let canonical_edge_payload = encode_accepted_edges(&accepted_edges);
        let edge_conservation = GhostLatticeEdgeConservation {
            schema_version: GHOST_LATTICE_EDGE_SCHEMA_VERSION,
            counted_edges_pass1: edge_count_host[0],
            produced_edges_pass2: pass2_edge_count_host[0],
            decoded_edges: accepted_edges.len() as u64,
            indexed_edges: accepted_edges
                .iter()
                .map(|edge| (edge.from_node_index, edge.to_node_index))
                .collect::<BTreeSet<_>>()
                .len() as u64,
            overflow_count: accepted_edge_overflow_host[0],
            counted_equals_produced_equals_decoded_equals_indexed: false,
            canonical_payload_sha256: ghost_record_sha256(&canonical_edge_payload),
        };
        let mut edge_conservation = edge_conservation;
        edge_conservation.counted_equals_produced_equals_decoded_equals_indexed =
            edge_conservation.overflow_count == 0
                && edge_conservation.counted_edges_pass1 == edge_conservation.produced_edges_pass2
                && edge_conservation.produced_edges_pass2 == edge_conservation.decoded_edges
                && edge_conservation.decoded_edges == edge_conservation.indexed_edges;
        if !edge_conservation.counted_equals_produced_equals_decoded_equals_indexed {
            bail!(
                "accepted Ghost lattice edge conservation failed: {:?}",
                edge_conservation,
            );
        }
        let decoded_score_sum_q = accepted_edges.iter().try_fold(0u64, |sum, edge| {
            sum.checked_add((edge.total_score * 1_000_000.0) as u64)
                .context("accepted edge score accumulator overflow")
        })?;
        if decoded_score_sum_q != pass2_edge_score_sum_host[0] {
            bail!(
                "accepted Ghost lattice edge score conservation failed: kernel={} decoded={}",
                pass2_edge_score_sum_host[0],
                decoded_score_sum_q,
            );
        }
        log::info!(
            "  [GHOST-LATTICE-4D] download: {} ms",
            t_dl.elapsed().as_millis()
        );

        // ── Step 5. Component extraction from exact edge rows ──────
        let t_extract = std::time::Instant::now();
        let total_edges = pass2_edge_count_host[0];
        let mut components = components_from_exact_edges(nodes.len(), &accepted_edges)?;
        verify_device_parent_components(&parent_host, &components)?;
        components.sort_by(|a, b| b.node_indices.len().cmp(&a.node_indices.len()));
        log::info!(
            "  [GHOST-LATTICE-4D] components: {} (largest={}) | extract {} ms",
            components.len(),
            components
                .first()
                .map(|c| c.node_indices.len())
                .unwrap_or(0),
            t_extract.elapsed().as_millis()
        );

        let host_ms = t_total.elapsed().as_secs_f64() * 1000.0;
        let mut phases_present: u8 = 0;
        for n in nodes {
            phases_present |= 1u8 << n.protocol_phase.min(7);
        }

        let stats = GhostPhaseLatticeRunStats {
            n_nodes,
            n_lattice_cells: n_cells,
            n_pairs_evaluated: pair_count_host[0],
            n_pairs_phase_legal: phase_legal_host[0],
            n_pairs_aabb_overlap: aabb_overlap_host[0],
            n_directed_edges: total_edges,
            kernel_elapsed_ms: kernel_ms,
            host_elapsed_ms: host_ms,
            min_step_idx: span.min_step,
            max_step_idx: span.max_step,
            phases_present,
        };

        log::info!(
            "POST_MD_CLUSTER_GHOST_LATTICE_4D_DONE components={} pairs={} edges={} | {:.2} ms total",
            components.len(),
            stats.n_pairs_evaluated,
            stats.n_directed_edges,
            host_ms
        );

        Ok(GhostPhaseLatticeOutcome {
            components,
            accepted_edges,
            edge_conservation,
            stats,
            config: self.config,
        })
    }

    /// Per-cell index table. Sorts nodes by lattice key so the kernel can
    /// iterate one cell's nodes contiguously, then builds a 27-cell × 4-phase
    /// × 3-step-bucket neighbour list per query.
    fn build_lattice_cells(
        &self,
        nodes: &[GhostPhaseLatticeNode],
    ) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<i32>, StepSpan) {
        let cell_size = self.config.spatial_cell_size_a.max(1e-3);

        // Step span (also used for stats).
        let mut min_step = u64::MAX;
        let mut max_step = 0u64;
        for n in nodes {
            min_step = min_step.min(n.step_idx);
            max_step = max_step.max(n.step_idx);
        }
        if min_step == u64::MAX {
            min_step = 0;
        }
        let span = StepSpan { min_step, max_step };

        // Project each node to its lattice key. The kernel's neighborhood is
        // ±1 cell in (x, y, z), ±1 phase, ±1 step bucket. Spatial cells use
        // a signed integer floor; the host packs them into a stable 64-bit
        // composite so the sort drops contiguous cells next to each other.
        let mut keys: Vec<(u64, u32)> = Vec::with_capacity(nodes.len());
        for (i, n) in nodes.iter().enumerate() {
            let cx = (n.centroid_xyz[0] / cell_size).floor() as i32;
            let cy = (n.centroid_xyz[1] / cell_size).floor() as i32;
            let cz = (n.centroid_xyz[2] / cell_size).floor() as i32;
            // Pack i32 cell coords into u16 ranges biased by 32_768 so the
            // composite key is monotone under sort. For a 1000 Å spatial
            // box at 5 Å cells, |cx| < 200 — well inside the 16-bit range.
            let bias = 32_768i32;
            let kx = ((cx + bias).clamp(0, 65_535)) as u64;
            let ky = ((cy + bias).clamp(0, 65_535)) as u64;
            let kz = ((cz + bias).clamp(0, 65_535)) as u64;
            // Composite: phase (2 bits) | step_bucket (16 bits) | kx (16) |
            // ky (16) | kz (14). Fits in 64 bits and is monotone in the
            // tuple order (phase, step_bucket, x, y, z) which is what the
            // kernel iterates.
            let phase = (n.protocol_phase & 0b11) as u64;
            let bucket = (n.step_bucket & 0xFFFF) as u64;
            let composite =
                (phase << 62) | (bucket << 46) | (kx << 30) | (ky << 14) | (kz & 0x3FFF);
            keys.push((composite, i as u32));
        }
        keys.sort_by_key(|&(k, _)| k);

        // Build cell tables: one entry per *unique* composite key.
        let mut cell_first: Vec<u32> = Vec::new();
        let mut cell_count: Vec<u32> = Vec::new();
        let mut cell_table: Vec<i32> = Vec::new();
        let mut permutation: Vec<u32> = Vec::with_capacity(nodes.len());

        let mut cur_key: Option<u64> = None;
        let mut run_start: u32 = 0;
        for (slot, &(k, idx)) in keys.iter().enumerate() {
            permutation.push(idx);
            match cur_key {
                Some(ck) if ck == k => {}
                _ => {
                    if let Some(_ck) = cur_key {
                        cell_count.push(slot as u32 - run_start);
                    }
                    // Decode the composite back into (cx, cy, cz, phase, bucket)
                    // for the kernel's neighbour walk.
                    let phase = ((k >> 62) & 0b11) as i32;
                    let bucket = ((k >> 46) & 0xFFFF) as i32;
                    let kx = ((k >> 30) & 0xFFFF) as i32;
                    let ky = ((k >> 14) & 0xFFFF) as i32;
                    let kz = (k & 0x3FFF) as i32;
                    let cx = kx - 32_768;
                    let cy = ky - 32_768;
                    let cz = kz - 32_768;
                    cell_table.extend_from_slice(&[cx, cy, cz, phase, bucket]);
                    cell_first.push(slot as u32);
                    cur_key = Some(k);
                    run_start = slot as u32;
                }
            }
        }
        if cur_key.is_some() {
            cell_count.push(keys.len() as u32 - run_start);
        }

        (permutation, cell_first, cell_count, cell_table, span)
    }
}

fn encode_accepted_edges(edges: &[GhostPhaseLatticeAcceptedEdge]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(edges.len() * GHOST_LATTICE_EDGE_BYTES);
    for edge in edges {
        bytes.extend_from_slice(&edge.from_node_index.to_le_bytes());
        bytes.extend_from_slice(&edge.to_node_index.to_le_bytes());
        bytes.extend_from_slice(&edge.from_step_idx.to_le_bytes());
        bytes.extend_from_slice(&edge.to_step_idx.to_le_bytes());
        bytes.extend_from_slice(&edge.temporal_delta_steps.to_le_bytes());
        bytes.extend_from_slice(&edge.total_score.to_le_bytes());
        bytes.extend_from_slice(&edge.so3_score.to_le_bytes());
        bytes.extend_from_slice(&edge.causal_score.to_le_bytes());
        bytes.extend_from_slice(&edge.thermo_score.to_le_bytes());
        bytes.push(edge.from_protocol_phase);
        bytes.push(edge.to_protocol_phase);
        bytes.push(edge.common_so3_plane_mask);
        bytes.push(edge.predicate_flags);
        bytes.extend_from_slice(&edge.schema_version.to_le_bytes());
    }
    bytes
}

fn decode_accepted_edges(bytes: &[u8]) -> Result<Vec<GhostPhaseLatticeAcceptedEdge>> {
    if bytes.len() % GHOST_LATTICE_EDGE_BYTES != 0 {
        bail!("accepted lattice-edge payload is truncated");
    }
    bytes
        .chunks_exact(GHOST_LATTICE_EDGE_BYTES)
        .map(|row| {
            Ok(GhostPhaseLatticeAcceptedEdge {
                from_node_index: u32::from_le_bytes(row[0..4].try_into()?),
                to_node_index: u32::from_le_bytes(row[4..8].try_into()?),
                from_step_idx: u64::from_le_bytes(row[8..16].try_into()?),
                to_step_idx: u64::from_le_bytes(row[16..24].try_into()?),
                temporal_delta_steps: u64::from_le_bytes(row[24..32].try_into()?),
                total_score: f32::from_le_bytes(row[32..36].try_into()?),
                so3_score: f32::from_le_bytes(row[36..40].try_into()?),
                causal_score: f32::from_le_bytes(row[40..44].try_into()?),
                thermo_score: f32::from_le_bytes(row[44..48].try_into()?),
                from_protocol_phase: row[48],
                to_protocol_phase: row[49],
                common_so3_plane_mask: row[50],
                predicate_flags: row[51],
                schema_version: u32::from_le_bytes(row[52..56].try_into()?),
            })
        })
        .collect()
}

fn phase_transition_is_legal(from: u8, to: u8) -> bool {
    matches!(
        (from, to),
        (PHASE_COLD_HOLD, PHASE_COLD_HOLD)
            | (PHASE_COLD_HOLD, PHASE_HEATING)
            | (PHASE_HEATING, PHASE_HEATING)
            | (PHASE_HEATING, PHASE_WARM_HOLD)
            | (PHASE_WARM_HOLD, PHASE_WARM_HOLD)
            | (PHASE_WARM_HOLD, PHASE_COOLING)
            | (PHASE_COOLING, PHASE_COOLING)
    )
}

fn validate_accepted_edges(
    nodes: &[GhostPhaseLatticeNode],
    edges: &[GhostPhaseLatticeAcceptedEdge],
    score_threshold: f32,
    max_temporal_edge_steps: u64,
) -> Result<()> {
    let mut identities = BTreeSet::new();
    let mut previous_sort_key: Option<(u64, u32, u64, u32)> = None;
    for edge in edges {
        let from = nodes
            .get(edge.from_node_index as usize)
            .context("accepted lattice-edge from endpoint is out of range")?;
        let to = nodes
            .get(edge.to_node_index as usize)
            .context("accepted lattice-edge to endpoint is out of range")?;
        let scores = [
            edge.total_score,
            edge.so3_score,
            edge.causal_score,
            edge.thermo_score,
        ];
        let product = edge.so3_score * edge.causal_score * edge.thermo_score;
        let sort_key = (
            edge.from_step_idx,
            edge.from_node_index,
            edge.to_step_idx,
            edge.to_node_index,
        );
        let overlap = (from.aabb_max[0] >= to.aabb_min[0] && from.aabb_min[0] <= to.aabb_max[0])
            && (from.aabb_max[1] >= to.aabb_min[1] && from.aabb_min[1] <= to.aabb_max[1])
            && (from.aabb_max[2] >= to.aabb_min[2] && from.aabb_min[2] <= to.aabb_max[2]);
        if edge.schema_version != GHOST_LATTICE_EDGE_SCHEMA_VERSION
            || edge.from_node_index == edge.to_node_index
            || previous_sort_key.is_some_and(|previous| sort_key <= previous)
            || !identities.insert((edge.from_node_index, edge.to_node_index))
            || from.step_idx != edge.from_step_idx
            || to.step_idx != edge.to_step_idx
            || edge.to_step_idx < edge.from_step_idx
            || edge.temporal_delta_steps != edge.to_step_idx - edge.from_step_idx
            || edge.temporal_delta_steps > max_temporal_edge_steps
            || from.protocol_phase != edge.from_protocol_phase
            || to.protocol_phase != edge.to_protocol_phase
            || !phase_transition_is_legal(edge.from_protocol_phase, edge.to_protocol_phase)
            || edge.common_so3_plane_mask != (from.so3_plane_status & to.so3_plane_status)
            || edge.predicate_flags != 0x0f
            || scores.iter().any(|value| !value.is_finite())
            || edge.total_score < score_threshold
            || (edge.total_score - product).abs() > 1.0e-5 * edge.total_score.abs().max(1.0)
            || !overlap
            || (edge.from_step_idx, from.tile_index) > (edge.to_step_idx, to.tile_index)
        {
            bail!(
                "accepted Ghost lattice-edge row failed exact endpoint/physics validation: {:?}",
                edge
            );
        }
        previous_sort_key = Some(sort_key);
    }
    Ok(())
}

fn components_from_exact_edges(
    n_nodes: usize,
    edges: &[GhostPhaseLatticeAcceptedEdge],
) -> Result<Vec<GhostPhaseLatticeComponent>> {
    let mut parent = (0..n_nodes).collect::<Vec<_>>();
    fn root(parent: &mut [usize], mut index: usize) -> usize {
        while parent[index] != index {
            let grandparent = parent[parent[index]];
            parent[index] = grandparent;
            index = grandparent;
        }
        index
    }
    for edge in edges {
        let a = root(&mut parent, edge.from_node_index as usize);
        let b = root(&mut parent, edge.to_node_index as usize);
        if a != b {
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            parent[hi] = lo;
        }
    }
    let mut nodes_by_root = BTreeMap::<usize, Vec<u32>>::new();
    for node in 0..n_nodes {
        let canonical = root(&mut parent, node);
        nodes_by_root
            .entry(canonical)
            .or_default()
            .push(node as u32);
    }
    let mut edge_aggregates = BTreeMap::<usize, (u32, f64)>::new();
    for edge in edges {
        let canonical = root(&mut parent, edge.from_node_index as usize);
        let aggregate = edge_aggregates.entry(canonical).or_default();
        aggregate.0 = aggregate
            .0
            .checked_add(1)
            .context("component edge-count overflow")?;
        aggregate.1 += f64::from(edge.total_score);
    }
    Ok(nodes_by_root
        .into_iter()
        .map(|(canonical, node_indices)| {
            let (n_intra_edges, intra_edge_score_sum) =
                edge_aggregates.get(&canonical).copied().unwrap_or_default();
            GhostPhaseLatticeComponent {
                node_indices,
                n_intra_edges,
                intra_edge_score_sum,
                edge_lineage_exact: true,
            }
        })
        .collect())
}

fn verify_device_parent_components(
    parent: &[i32],
    exact_components: &[GhostPhaseLatticeComponent],
) -> Result<()> {
    if parent
        .iter()
        .any(|value| *value < 0 || *value as usize >= parent.len())
    {
        bail!("device lattice parent array contains an out-of-range root");
    }
    let mut device = BTreeMap::<i32, Vec<u32>>::new();
    for (index, &initial) in parent.iter().enumerate() {
        let mut root = initial;
        let mut traversed = 0usize;
        while parent[root as usize] != root {
            root = parent[root as usize];
            traversed += 1;
            if traversed > parent.len() {
                bail!("device lattice parent array contains a cycle");
            }
        }
        device.entry(root).or_default().push(index as u32);
    }
    let mut device_sets = device.into_values().collect::<Vec<_>>();
    let mut exact_sets = exact_components
        .iter()
        .map(|component| component.node_indices.clone())
        .collect::<Vec<_>>();
    for set in device_sets.iter_mut().chain(exact_sets.iter_mut()) {
        set.sort_unstable();
    }
    device_sets.sort();
    exact_sets.sort();
    if device_sets != exact_sets {
        bail!("device union-find components differ from exact accepted-edge reconstruction");
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct StepSpan {
    min_step: u64,
    max_step: u64,
}

/// Convenience: load every `_stream{:02}_ghost_tiles.bin` in `output_dir`
/// matching `topo_stem`, project their v2 records to lattice nodes, and
/// concatenate them into a single vector. Skips files that don't open or
/// don't carry a valid GhostTileRing counter sector.
#[deprecated(
    note = "legacy flat Ghost payloads are not admissible for production lattice/Path-B evidence; use load_verified_ghost_segments_from_dir"
)]
#[cfg(test)]
pub fn load_ghost_v2_payloads_from_dir(
    output_dir: &std::path::Path,
    topo_stem: &str,
    schedule: &dyn PhaseSchedule,
    step_bucket_size: u64,
) -> Result<Vec<GhostPhaseLatticeNode>> {
    let mut out: Vec<GhostPhaseLatticeNode> = Vec::new();
    let mut total_records_seen: usize = 0;
    let mut total_v2_projected: usize = 0;
    for entry in std::fs::read_dir(output_dir)
        .with_context(|| format!("read_dir({})", output_dir.display()))?
    {
        let entry = entry?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        let prefix = format!("{}_stream", topo_stem);
        if !name_str.starts_with(&prefix) || !name_str.ends_with("_ghost_tiles.bin") {
            continue;
        }
        // Pull the stream id from `<prefix>NN_ghost_tiles.bin`.
        let mid = &name_str[prefix.len()..];
        let stream_id: u16 = mid.split('_').next().unwrap_or("0").parse().unwrap_or(0);

        let path = entry.path();
        let bytes = std::fs::read(&path).with_context(|| format!("read({})", path.display()))?;
        if bytes.len() <= 4096 {
            log::warn!(
                "  [GHOST-LATTICE-4D] {} too small ({} B) — skipping",
                path.display(),
                bytes.len()
            );
            continue;
        }
        // First 4 bytes hold the GPU-written n_frames_written counter; the
        // first 4096 B are the counter sector.
        let n_frames = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
        let payload = &bytes[4096..];
        let n_in_payload = payload.len() / GHOST_RECORD_BYTES;
        let n_actual = n_frames.min(n_in_payload);
        let actual_payload = &payload[..n_actual * GHOST_RECORD_BYTES];

        let projected =
            project_ghost_v2_payload(actual_payload, stream_id, schedule, step_bucket_size);
        total_records_seen += n_actual;
        total_v2_projected += projected.len();
        log::info!(
            "  [GHOST-LATTICE-4D] {}: stream={} n_records={} v2_projected={}",
            path.file_name().unwrap().to_string_lossy(),
            stream_id,
            n_actual,
            projected.len()
        );
        out.extend(projected);
    }

    if total_records_seen == 0 {
        bail!(
            "no ghost_tiles.bin records found in {} for stem {}",
            output_dir.display(),
            topo_stem
        );
    }
    log::info!(
        "  [GHOST-LATTICE-4D] total: {} records seen, {} v2-projected ({}% v2 yield)",
        total_records_seen,
        total_v2_projected,
        if total_records_seen > 0 {
            (total_v2_projected * 100) / total_records_seen
        } else {
            0
        }
    );
    Ok(out)
}

/// Persist one verified stream phase binding with file and directory fsync.
pub fn write_stream_phase_binding_atomic(
    output_dir: &Path,
    topo_stem: &str,
    binding: &StreamProtocolPhaseBinding,
) -> Result<PathBuf> {
    binding.verify()?;
    std::fs::create_dir_all(output_dir)?;
    let path = output_dir.join(format!(
        "{}_stream{:02}_ghost_phase_binding.json",
        topo_stem, binding.stream_id
    ));
    let partial = output_dir.join(format!(
        ".{}_stream{:02}_ghost_phase_binding.json.partial",
        topo_stem, binding.stream_id
    ));
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&partial)?;
    let bytes = serde_json::to_vec_pretty(binding)?;
    file.write_all(&bytes)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(&partial, &path)?;
    std::fs::File::open(output_dir)?.sync_all()?;
    Ok(path)
}

/// Load every phase binding for a topology stem and reject duplicate stream
/// identities or invalid digests.
pub fn load_stream_phase_bindings(
    output_dir: &Path,
    topo_stem: &str,
) -> Result<BTreeMap<u16, StreamProtocolPhaseBinding>> {
    let prefix = format!("{topo_stem}_stream");
    let suffix = "_ghost_phase_binding.json";
    let mut out = BTreeMap::new();
    for entry in std::fs::read_dir(output_dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with(&prefix) || !name.ends_with(suffix) {
            continue;
        }
        let binding: StreamProtocolPhaseBinding =
            serde_json::from_slice(&std::fs::read(entry.path())?)?;
        binding.verify()?;
        if out.insert(binding.stream_id, binding).is_some() {
            bail!("duplicate Ghost phase binding stream identity");
        }
    }
    if out.is_empty() {
        bail!(
            "no per-stream Ghost phase bindings found in {} for {}",
            output_dir.display(),
            topo_stem
        );
    }
    Ok(out)
}

/// Production loader for versioned Ghost segments.
///
/// Unlike the test-only legacy flat-payload loader, this function never
/// consumes a flat `*_ghost_tiles.bin` artifact and never accepts a quartile
/// phase fallback. Every stream must be completed, SHA-chain verified, and
/// bound to an engine-owned protocol schedule. Any malformed or
/// non-projectable record fails the complete load, enforcing `produced =
/// persisted = decoded = lattice nodes` per stream and in aggregate.
/// Reconstruct the only admissible staging-loss count from the v4 writer
/// authority and verified manifest totals. Every term is engine-owned: replay
/// admission, capacity violations, payload-count mismatches, membership
/// overflow, terminal residual and producer/persistence/drain equality. A
/// successful Layer-05 load is impossible unless this deterministic sum is
/// exactly zero.
fn reconstruct_overflow_loss_count(
    manifest: &crate::ghost_segment::GhostSegmentManifest,
) -> Result<u64> {
    let authority = &manifest.overflow_conservation;
    let mut loss = 0_u64;
    for term in [
        authority.drain_capacity_violation_count,
        authority.writer_payload_record_count_mismatch_count,
        authority.membership_overflow_record_count,
        authority.terminal_residual_records.unwrap_or(0),
        authority
            .drain_produced_records
            .abs_diff(manifest.produced_records),
        manifest
            .produced_records
            .abs_diff(manifest.persisted_records),
        u64::from(!authority.replay_upper_bound_admission_pass),
    ] {
        loss = loss
            .checked_add(term)
            .context("Ghost overflow-loss reconstruction overflow")?;
    }
    Ok(loss)
}

pub fn load_verified_ghost_segments_from_dir(
    output_dir: &Path,
    topo_stem: &str,
    step_bucket_size: u64,
) -> Result<GhostSegmentLatticeLoad> {
    let bindings = load_stream_phase_bindings(output_dir, topo_stem)?;
    let verified_streams =
        crate::ghost_segment::discover_verified_ghost_streams(output_dir, topo_stem)?;
    let segment_stream_ids = verified_streams
        .iter()
        .map(|stream| stream.manifest.stream_id)
        .collect::<BTreeSet<_>>();
    let binding_stream_ids = bindings.keys().copied().collect::<BTreeSet<_>>();
    if segment_stream_ids != binding_stream_ids {
        bail!(
            "Ghost segment/binding stream identity sets differ: segments={:?} bindings={:?}",
            segment_stream_ids,
            binding_stream_ids
        );
    }
    let mut nodes = Vec::new();
    let mut record_lineage = Vec::new();
    let mut receipts = Vec::new();
    let mut total_produced = 0u64;
    let mut total_persisted = 0u64;
    let mut total_decoded = 0u64;
    let mut total_lattice = 0u64;
    let mut total_overflow_loss = 0u64;

    for stream in verified_streams {
        if !stream.manifest.completed {
            bail!(
                "Ghost production loader rejects incomplete stream {}",
                stream.manifest.stream_id
            );
        }
        let stream_id = stream.manifest.stream_id;
        if stream.manifest.record_schema_version
            != crate::ghost_segment::GHOST_SEGMENT_RECORD_SCHEMA_VERSION
        {
            bail!(
                "Ghost lattice rejects segment-only record schema {} on stream {}; exact MAR-v2 schema {} is required",
                stream.manifest.record_schema_version,
                stream_id,
                crate::ghost_segment::GHOST_SEGMENT_RECORD_SCHEMA_VERSION,
            );
        }
        let binding = bindings
            .get(&stream_id)
            .with_context(|| format!("missing Ghost phase binding for stream {stream_id}"))?;
        binding.verify()?;
        let context = &stream.manifest.context;
        let overflow_authority = &stream.manifest.overflow_conservation;
        let stream_overflow_loss = reconstruct_overflow_loss_count(&stream.manifest)?;
        if stream_overflow_loss != 0 || !overflow_authority.acceptance_pass() {
            bail!(
                "Ghost stream {} lacks complete v4 engine-owned overflow conservation: reconstructed_loss={}",
                stream_id,
                stream_overflow_loss,
            );
        }
        if context.stream_id != binding.stream_id
            || context.thermodynamic_condition_id != binding.condition_id
            || context.protocol_id != binding.protocol_name
            || context.protocol_state_hash != binding.protocol_state_hash
            || context.phase_schedule_digest_sha256 != binding.phase_schedule_digest_sha256
        {
            bail!(
                "Ghost segment context/phase binding mismatch for stream {}",
                stream_id
            );
        }
        let mut decoded = 0u64;
        let lattice_before = nodes.len() as u64;
        for segment in &stream.segments {
            for local_index in 0..segment.entry.record_count {
                let start = local_index as usize * GHOST_RECORD_BYTES;
                let end = start + GHOST_RECORD_BYTES;
                let record = segment
                    .payload
                    .get(start..end)
                    .context("verified Ghost segment lost a complete record")?;
                let global_ordinal = segment
                    .entry
                    .first_record_ordinal
                    .checked_add(local_index)
                    .context("Ghost global record ordinal overflow")?;
                let global_lattice_index = nodes.len() as u64;
                let tile_index: u32 = global_lattice_index.try_into().map_err(|_| {
                    anyhow!(
                        "Ghost global lineage exceeds u32 lattice identity for stream {}",
                        stream_id
                    )
                })?;
                let cluster_residue_membership = decode_ghost_v3_cluster_residue_membership(record)
                    .map_err(|error| {
                        anyhow!(
                            "Ghost v3 cluster-residue membership failed stream={} record={}: {}",
                            stream_id,
                            global_ordinal,
                            error
                        )
                    })?;
                decoded = decoded
                    .checked_add(1)
                    .context("Ghost decoded record count overflow")?;
                let node = project_ghost_v2_record(
                    record,
                    tile_index,
                    stream_id,
                    binding,
                    step_bucket_size,
                )
                .with_context(|| {
                    format!(
                        "Ghost record {} stream {} failed schema/spatial/phase projection",
                        global_ordinal, stream_id
                    )
                })?;
                if binding.checked_phase_for(node.step_idx) != Some(node.protocol_phase) {
                    bail!(
                        "Ghost node phase mismatch stream={} record={} step={}",
                        stream_id,
                        global_ordinal,
                        node.step_idx
                    );
                }
                let payload_byte_offset = segment
                    .entry
                    .payload_offset
                    .checked_add(
                        local_index
                            .checked_mul(GHOST_RECORD_BYTES as u64)
                            .context("Ghost record byte offset overflow")?,
                    )
                    .context("Ghost segment payload byte offset overflow")?;
                record_lineage.push(GhostRecordByteLineage {
                    schema_version: GHOST_RECORD_BYTE_LINEAGE_SCHEMA_VERSION,
                    stream_id,
                    run_id: context.run_id.clone(),
                    replica_id: context.replica_id.clone(),
                    replica_seed: context.replica_seed,
                    stream_seed: context.stream_seed,
                    topology_sha256: context.topology_sha256.clone(),
                    selected_engine_sha256: context.selected_engine_sha256.clone(),
                    record_abi_sha256: context.record_abi_sha256.clone(),
                    record_source_sha256: context.record_source_sha256.clone(),
                    segment_context_digest_sha256: stream.manifest.context_digest_sha256.clone(),
                    stream_record_ordinal: global_ordinal,
                    global_lattice_index,
                    segment_manifest_sha256: stream.manifest_sha256.clone(),
                    segment_ordinal: segment.entry.segment_ordinal,
                    segment_file: format!(
                        "{}_stream{:02}_ghost_segments/{}",
                        topo_stem, stream_id, segment.entry.file
                    ),
                    segment_file_sha256: segment.entry.file_sha256.clone(),
                    payload_byte_offset,
                    byte_length: GHOST_RECORD_BYTES as u32,
                    record_sha256: ghost_record_sha256(record),
                    frame_idx: node.frame_idx,
                    step_idx: node.step_idx,
                    site_id: node.site_id,
                    condition_id: binding.condition_id.clone(),
                    protocol_phase: node.protocol_phase,
                    phase_schedule_digest_sha256: binding.phase_schedule_digest_sha256.clone(),
                    protocol_state_hash: binding.protocol_state_hash.clone(),
                    cluster_residue_membership,
                });
                nodes.push(node);
            }
        }
        let lattice_count = (nodes.len() as u64)
            .checked_sub(lattice_before)
            .context("Ghost lattice count underflow")?;
        let produced = stream.manifest.produced_records;
        let persisted = stream.manifest.persisted_records;
        let equality = produced == persisted && persisted == decoded && decoded == lattice_count;
        if !equality {
            bail!(
                "Ghost conservation failed stream={}: produced={} persisted={} decoded={} lattice={}",
                stream_id,
                produced,
                persisted,
                decoded,
                lattice_count
            );
        }
        receipts.push(GhostStreamConservation {
            stream_id,
            segment_manifest_sha256: stream.manifest_sha256.clone(),
            segment_context_digest_sha256: stream.manifest.context_digest_sha256.clone(),
            produced_records: produced,
            persisted_records: persisted,
            decoded_records: decoded,
            lattice_node_lineage_count: lattice_count,
            condition_id: binding.condition_id.clone(),
            protocol_name: binding.protocol_name.clone(),
            protocol_state_hash: binding.protocol_state_hash.clone(),
            phase_schedule_digest_sha256: binding.phase_schedule_digest_sha256.clone(),
            equality_pass: true,
            overflow_conservation: overflow_authority.clone(),
            ordinal_header_manifest_reconstruction_pass: true,
            overflow_loss_count: stream_overflow_loss,
            overflow_conservation_pass: true,
        });
        total_produced = total_produced
            .checked_add(produced)
            .context("Ghost total produced count overflow")?;
        total_persisted = total_persisted
            .checked_add(persisted)
            .context("Ghost total persisted count overflow")?;
        total_decoded = total_decoded
            .checked_add(decoded)
            .context("Ghost total decoded count overflow")?;
        total_lattice = total_lattice
            .checked_add(lattice_count)
            .context("Ghost total lattice count overflow")?;
        total_overflow_loss = total_overflow_loss
            .checked_add(stream_overflow_loss)
            .context("Ghost total overflow-loss count overflow")?;
    }
    if bindings.len() != receipts.len() {
        bail!(
            "Ghost phase/segment stream-set mismatch: bindings={} segments={}",
            bindings.len(),
            receipts.len()
        );
    }
    let equality = total_produced == total_persisted
        && total_persisted == total_decoded
        && total_decoded == total_lattice
        && total_lattice == nodes.len() as u64
        && nodes.len() == record_lineage.len()
        && record_lineage
            .iter()
            .enumerate()
            .all(|(index, lineage)| lineage.global_lattice_index == index as u64);
    if !equality {
        bail!(
            "Ghost total conservation failed: produced={} persisted={} decoded={} lattice={} nodes={} byte_lineage={}",
            total_produced,
            total_persisted,
            total_decoded,
            total_lattice,
            nodes.len(),
            record_lineage.len(),
        );
    }
    let overflow_conservation_pass = receipts
        .iter()
        .all(|receipt| receipt.overflow_conservation_pass);
    Ok(GhostSegmentLatticeLoad {
        nodes,
        record_lineage,
        streams: receipts,
        produced_records: total_produced,
        persisted_records: total_persisted,
        decoded_records: total_decoded,
        lattice_node_lineage_count: total_lattice,
        phase_protocols_bound_by_stream: true,
        equality_pass: true,
        overflow_loss_count: total_overflow_loss,
        overflow_conservation_pass,
    })
}

fn ghost_record_sha256(record: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(record);
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn require_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != 64
        || !value.bytes().all(|byte| byte.is_ascii_hexdigit())
        || value.bytes().any(|byte| byte.is_ascii_uppercase())
    {
        bail!("{label} must be canonical lowercase SHA256");
    }
    Ok(())
}

fn phase_binding_digest(
    stream_id: u16,
    condition_id: &str,
    protocol_name: &str,
    protocol_state_hash: &str,
    intervals: &[ProtocolPhaseInterval],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"prism-ghost-phase-binding-v1\0");
    hasher.update(stream_id.to_le_bytes());
    for text in [condition_id, protocol_name, protocol_state_hash] {
        hasher.update((text.len() as u64).to_le_bytes());
        hasher.update(text.as_bytes());
    }
    for interval in intervals {
        hasher.update(interval.start_step.to_le_bytes());
        hasher.update(interval.end_step.to_le_bytes());
        hasher.update([interval.phase_ordinal]);
        hasher.update((interval.phase_name.len() as u64).to_le_bytes());
        hasher.update(interval.phase_name.as_bytes());
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn exact_edge_fixture() -> (
        Vec<GhostPhaseLatticeNode>,
        Vec<GhostPhaseLatticeAcceptedEdge>,
    ) {
        let mut nodes = (0..3)
            .map(|index| {
                let mut node = GhostPhaseLatticeNode::empty();
                node.tile_index = index;
                node.stream_id = 2;
                node.site_id = 9;
                node.step_idx = 100 + u64::from(index) * 10;
                node.protocol_phase = PHASE_COLD_HOLD;
                node.aabb_min = [0.0, 0.0, 0.0];
                node.aabb_max = [2.0, 2.0, 2.0];
                node.centroid_xyz = [1.0, 1.0, 1.0];
                node.so3_plane_status = SO3_PLANE_GEOMETRY;
                node.so3_power_spectrum[0] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
                node
            })
            .collect::<Vec<_>>();
        nodes[2].aabb_min = [10.0, 10.0, 10.0];
        nodes[2].aabb_max = [12.0, 12.0, 12.0];
        let edge = GhostPhaseLatticeAcceptedEdge {
            from_node_index: 0,
            to_node_index: 1,
            from_step_idx: 100,
            to_step_idx: 110,
            temporal_delta_steps: 10,
            total_score: 0.8,
            so3_score: 0.8,
            causal_score: 1.0,
            thermo_score: 1.0,
            from_protocol_phase: PHASE_COLD_HOLD,
            to_protocol_phase: PHASE_COLD_HOLD,
            common_so3_plane_mask: SO3_PLANE_GEOMETRY,
            predicate_flags: 0x0f,
            schema_version: GHOST_LATTICE_EDGE_SCHEMA_VERSION,
        };
        (nodes, vec![edge])
    }

    fn exact_edge_artifact(
        nodes: &[GhostPhaseLatticeNode],
        edges: Vec<GhostPhaseLatticeAcceptedEdge>,
    ) -> GhostLatticeAcceptedEdgeArtifact {
        let count = edges.len() as u64;
        let outcome = GhostPhaseLatticeOutcome {
            components: components_from_exact_edges(nodes.len(), &edges).unwrap(),
            accepted_edges: edges.clone(),
            edge_conservation: GhostLatticeEdgeConservation {
                schema_version: GHOST_LATTICE_EDGE_SCHEMA_VERSION,
                counted_edges_pass1: count,
                produced_edges_pass2: count,
                decoded_edges: count,
                indexed_edges: count,
                overflow_count: 0,
                counted_equals_produced_equals_decoded_equals_indexed: true,
                canonical_payload_sha256: ghost_record_sha256(&encode_accepted_edges(&edges)),
            },
            stats: GhostPhaseLatticeRunStats {
                n_nodes: nodes.len() as u32,
                n_directed_edges: count,
                ..GhostPhaseLatticeRunStats::default()
            },
            config: GhostPhaseLatticeConfig::default(),
        };
        GhostLatticeAcceptedEdgeArtifact::from_outcome(
            "run_test",
            "mycmax",
            &"11".repeat(32),
            nodes,
            &outcome,
        )
        .unwrap()
    }

    fn synth_record(
        step_idx: u64,
        x: f32,
        y: f32,
        z: f32,
        kl: f32,
        plane_geo: [f32; 6],
        plane_caus: Option<[f32; 6]>,
        causal_lead: u32,
    ) -> [u8; GHOST_RECORD_BYTES] {
        let mut r = [0u8; GHOST_RECORD_BYTES];
        r[0..8].copy_from_slice(&1u64.to_le_bytes()); // frame_idx
        r[8..12].copy_from_slice(&7u32.to_le_bytes()); // site_id
        r[12] = 0x41; // chain_id
        r[13] = 1; // adjudication_code
        r[14..16].copy_from_slice(&0u16.to_le_bytes());
        r[16..20].copy_from_slice(&kl.to_le_bytes());
        // plane geometry @ offset 20
        for (i, v) in plane_geo.iter().enumerate() {
            let off = 20 + i * 4;
            r[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        if let Some(pc) = plane_caus {
            for (i, v) in pc.iter().enumerate() {
                let off = 20 + (6 + i) * 4;
                r[off..off + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        // thermo_flux @ 116
        r[116..120].copy_from_slice(&f32::NAN.to_le_bytes());
        r[120..124].copy_from_slice(&f32::NAN.to_le_bytes());
        r[124..128].copy_from_slice(&causal_lead.to_le_bytes());
        // schema_version @ 128
        r[128..132].copy_from_slice(&GHOST_FRAME_SCHEMA_V3.to_le_bytes());
        r[GHOST_V2_OFFSET_GEAR_ID..GHOST_V2_OFFSET_GEAR_ID + 4]
            .copy_from_slice(&0u32.to_le_bytes());
        r[GHOST_V2_OFFSET_DT_FS..GHOST_V2_OFFSET_DT_FS + 4].copy_from_slice(&2.0f32.to_le_bytes());
        r[GHOST_V2_OFFSET_STEP_IDX..GHOST_V2_OFFSET_STEP_IDX + 8]
            .copy_from_slice(&step_idx.to_le_bytes());
        // Spatial native bit + AABB
        let fcf = GHOST_FCF_BIT_SPATIAL_NATIVE_AABB_MIDPOINT;
        r[GHOST_V2_OFFSET_FIELD_COMPLETENESS_FLAGS..GHOST_V2_OFFSET_FIELD_COMPLETENESS_FLAGS + 2]
            .copy_from_slice(&fcf.to_le_bytes());
        for (i, v) in [x - 0.5, y - 0.5, z - 0.5].iter().enumerate() {
            let off = GHOST_V2_OFFSET_AABB_MIN + i * 4;
            r[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        for (i, v) in [x + 0.5, y + 0.5, z + 0.5].iter().enumerate() {
            let off = GHOST_V2_OFFSET_AABB_MAX + i * 4;
            r[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        for (i, v) in [x, y, z].iter().enumerate() {
            let off = GHOST_V2_OFFSET_CENTROID + i * 4;
            r[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        let write_u32 = |record: &mut [u8; GHOST_RECORD_BYTES], offset: usize, value: u32| {
            record[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        };
        write_u32(
            &mut r,
            crate::ghost_tile::GHOST_V3_OFFSET_MEMBERSHIP_SCHEMA_VERSION,
            crate::ghost_tile::GHOST_V3_MEMBERSHIP_SCHEMA_VERSION,
        );
        write_u32(
            &mut r,
            crate::ghost_tile::GHOST_V3_OFFSET_CLUSTER_RESIDUE_COUNT,
            1,
        );
        write_u32(
            &mut r,
            crate::ghost_tile::GHOST_V3_OFFSET_CLUSTER_RESIDUE_CAPACITY,
            crate::ghost_tile::GHOST_V3_CLUSTER_RESIDUE_CAPACITY as u32,
        );
        write_u32(
            &mut r,
            crate::ghost_tile::GHOST_V3_OFFSET_SOURCE_SPIKE_COUNT,
            1,
        );
        write_u32(
            &mut r,
            crate::ghost_tile::GHOST_V3_OFFSET_CLUSTER_RESIDUE_ENTRY_BYTES,
            crate::ghost_tile::GHOST_V3_CLUSTER_RESIDUE_ENTRY_BYTES as u32,
        );
        write_u32(
            &mut r,
            crate::ghost_tile::GHOST_V3_OFFSET_GEOMETRY_AVAILABILITY_FLAGS,
            crate::ghost_tile::GHOST_V3_GEOMETRY_CLUSTER_RESIDUE_MEMBERSHIP_EXACT
                | crate::ghost_tile::GHOST_V3_GEOMETRY_COORDINATE_SIDECAR_REQUIRED,
        );
        write_u32(
            &mut r,
            crate::ghost_tile::GHOST_V3_OFFSET_CLUSTER_RESIDUE_ENTRIES,
            7,
        );
        write_u32(
            &mut r,
            crate::ghost_tile::GHOST_V3_OFFSET_CLUSTER_RESIDUE_ENTRIES + 4,
            1,
        );
        r
    }

    #[test]
    fn projects_v2_record_with_native_aabb() {
        let r = synth_record(
            1000,
            10.0,
            20.0,
            30.0,
            0.123,
            [0.4, 0.3, 0.15, 0.08, 0.04, 0.03],
            None,
            42,
        );
        let schedule = QuartilePhaseSchedule {
            min_step: 0,
            max_step: 4000,
        };
        let n = project_ghost_v2_record(&r, 17, 3, &schedule, 500).unwrap();
        assert_eq!(n.tile_index, 17);
        assert_eq!(n.stream_id, 3);
        assert_eq!(n.site_id, 7);
        assert_eq!(n.frame_idx, 1);
        assert_eq!(n.step_idx, 1000);
        assert_eq!(n.protocol_phase, PHASE_HEATING);
        assert_eq!(n.step_bucket, 2);
        assert_eq!(n.causal_lead_residue, 42);
        assert!(n.has_valid_aabb());
        assert!(n.has_any_so3_plane());
        assert_eq!(n.so3_plane_status & SO3_PLANE_GEOMETRY, SO3_PLANE_GEOMETRY);
        assert_eq!(n.so3_plane_status & SO3_PLANE_CAUSALITY, 0);
        assert!((n.centroid_xyz[0] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn rejects_v1_record() {
        let mut r = [0u8; GHOST_RECORD_BYTES];
        r[128..132].copy_from_slice(&0u32.to_le_bytes()); // v1 / legacy
        let schedule = QuartilePhaseSchedule {
            min_step: 0,
            max_step: 1000,
        };
        let n = project_ghost_v2_record(&r, 0, 0, &schedule, 500);
        assert!(n.is_none());
    }

    #[test]
    fn quartile_schedule_partitions_correctly() {
        let s = QuartilePhaseSchedule {
            min_step: 0,
            max_step: 4000,
        };
        assert_eq!(s.phase_for(0, 0), PHASE_COLD_HOLD);
        assert_eq!(s.phase_for(999, 0), PHASE_COLD_HOLD);
        assert_eq!(s.phase_for(1000, 0), PHASE_HEATING);
        assert_eq!(s.phase_for(2000, 0), PHASE_WARM_HOLD);
        assert_eq!(s.phase_for(3500, 0), PHASE_COOLING);
        assert_eq!(s.phase_for(5000, 0), PHASE_COOLING); // saturate
    }

    #[test]
    fn aabb_volume_handles_degenerate_box() {
        let mut n = GhostPhaseLatticeNode::empty();
        n.aabb_min = [1.0, 2.0, 3.0];
        n.aabb_max = [2.0, 4.0, 6.0];
        assert!((n.aabb_volume() - 6.0).abs() < 1e-6);
        n.aabb_max = [1.0, 2.0, 3.0]; // collapsed
        assert!(n.aabb_volume() == 0.0);
    }

    #[test]
    fn node_struct_size_pinned() {
        // Hard-pin against silent repacking that would break the kernel.
        assert_eq!(std::mem::size_of::<GhostPhaseLatticeNode>(), 208);
    }

    #[test]
    fn accepted_edge_binary_roundtrip_truncation_and_tamper_are_fail_closed() {
        let (nodes, edges) = exact_edge_fixture();
        let bytes = encode_accepted_edges(&edges);
        assert_eq!(bytes.len(), GHOST_LATTICE_EDGE_BYTES);
        assert_eq!(decode_accepted_edges(&bytes).unwrap(), edges);
        assert!(decode_accepted_edges(&bytes[..bytes.len() - 1]).is_err());

        let mut tampered = bytes;
        tampered[51] = 0x07;
        let decoded = decode_accepted_edges(&tampered).unwrap();
        assert!(validate_accepted_edges(
            &nodes,
            &decoded,
            DEFAULT_SO3_THRESHOLD,
            DEFAULT_MAX_TEMPORAL_EDGE_STEPS,
        )
        .is_err());
    }

    #[test]
    fn accepted_edge_duplicate_endpoint_order_phase_and_score_are_rejected() {
        let (nodes, edges) = exact_edge_fixture();
        let valid = edges[0];

        let mut duplicate = vec![valid, valid];
        assert!(validate_accepted_edges(&nodes, &duplicate, 0.75, 500).is_err());

        let mut invalid_endpoint = valid;
        invalid_endpoint.to_node_index = 99;
        assert!(validate_accepted_edges(&nodes, &[invalid_endpoint], 0.75, 500).is_err());

        let mut invalid_order = valid;
        invalid_order.from_node_index = 1;
        invalid_order.to_node_index = 0;
        invalid_order.from_step_idx = 110;
        invalid_order.to_step_idx = 100;
        assert!(validate_accepted_edges(&nodes, &[invalid_order], 0.75, 500).is_err());

        let mut invalid_phase = valid;
        invalid_phase.to_protocol_phase = PHASE_COOLING;
        assert!(validate_accepted_edges(&nodes, &[invalid_phase], 0.75, 500).is_err());

        let mut invalid_score = valid;
        invalid_score.total_score = f32::NAN;
        assert!(validate_accepted_edges(&nodes, &[invalid_score], 0.75, 500).is_err());

        duplicate.reverse();
        assert!(validate_accepted_edges(&nodes, &duplicate, 0.75, 500).is_err());
    }

    #[test]
    fn durable_edge_loader_binds_raw_sha_node_lineage_canonical_bytes_and_nofollow() {
        let (nodes, edges) = exact_edge_fixture();
        let artifact = exact_edge_artifact(&nodes, edges);
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("edges.json");
        let mut canonical = serde_json::to_vec_pretty(&artifact).unwrap();
        canonical.push(b'\n');
        std::fs::write(&path, &canonical).unwrap();
        let file_sha = ghost_record_sha256(&canonical);
        let loaded = load_verified_ghost_lattice_edges(
            &path,
            &nodes,
            &file_sha,
            canonical.len() as u64,
            &artifact.node_lineage_artifact_sha256,
        )
        .unwrap();
        assert_eq!(loaded, artifact);
        assert!(load_verified_ghost_lattice_edges(
            &path,
            &nodes,
            &"22".repeat(32),
            canonical.len() as u64,
            &artifact.node_lineage_artifact_sha256,
        )
        .is_err());
        assert!(load_verified_ghost_lattice_edges(
            &path,
            &nodes,
            &file_sha,
            canonical.len() as u64,
            &"33".repeat(32),
        )
        .is_err());
        assert!(load_verified_ghost_lattice_edges(
            &path,
            &nodes,
            &file_sha,
            canonical.len() as u64 + 1,
            &artifact.node_lineage_artifact_sha256,
        )
        .is_err());

        let noncanonical_path = tmp.path().join("edges_noncanonical.json");
        let mut noncanonical = canonical.clone();
        noncanonical.extend_from_slice(b" \n");
        std::fs::write(&noncanonical_path, &noncanonical).unwrap();
        assert!(load_verified_ghost_lattice_edges(
            &noncanonical_path,
            &nodes,
            &ghost_record_sha256(&noncanonical),
            noncanonical.len() as u64,
            &artifact.node_lineage_artifact_sha256,
        )
        .is_err());

        let symlink_path = tmp.path().join("edges_symlink.json");
        std::os::unix::fs::symlink(&path, &symlink_path).unwrap();
        assert!(load_verified_ghost_lattice_edges(
            &symlink_path,
            &nodes,
            &file_sha,
            canonical.len() as u64,
            &artifact.node_lineage_artifact_sha256,
        )
        .is_err());
    }

    #[test]
    fn durable_edge_conservation_and_component_reconstruction_are_exact() {
        let (nodes, edges) = exact_edge_fixture();
        let artifact = exact_edge_artifact(&nodes, edges);
        let components = reconstruct_verified_lattice_components(&nodes, &artifact).unwrap();
        assert_eq!(components.len(), 2);
        assert_eq!(components[0].node_indices, vec![0, 1]);
        assert_eq!(components[0].edge_indices, vec![0]);
        assert_eq!(components[0].n_intra_edges, 1);
        assert_eq!(components[1].node_indices, vec![2]);
        assert!(components[1].edge_indices.is_empty());
        require_sha256(&components[0].component_digest_sha256, "component digest").unwrap();
        assert_eq!(
            reconstruct_verified_lattice_components(&nodes, &artifact).unwrap(),
            components
        );

        let raw_components = components_from_exact_edges(nodes.len(), &artifact.edges).unwrap();
        assert!(verify_device_parent_components(&[0, 0, 2], &raw_components).is_ok());
        assert!(verify_device_parent_components(&[0, 1, 2], &raw_components).is_err());

        let mut mismatch = artifact;
        mismatch.persisted_edges += 1;
        assert!(mismatch.verify(&nodes).is_err());
    }

    #[test]
    fn static_and_nvrtc_edge_abi_and_two_pass_entrypoints_are_pinned() {
        assert_eq!(std::mem::size_of::<GhostPhaseLatticeAcceptedEdge>(), 56);
        assert_eq!(std::mem::align_of::<GhostPhaseLatticeAcceptedEdge>(), 8);
        let nvrtc = include_str!("cuda/ghost_lattice_kernel_nvrtc.cu");
        let header = include_str!("cuda/ghost_lattice_kernel.cuh");
        let wrapper = include_str!("cuda/ghost_lattice_kernel.cu");
        for source in [nvrtc, header, wrapper] {
            assert!(source.contains("GhostPhaseLatticeAcceptedEdge"));
            assert!(source.contains("schema_version"));
        }
        assert!(nvrtc.contains("const unsigned int pass_kind"));
        assert!(nvrtc.contains("pass_kind == 0u"));
        assert!(nvrtc.contains("pass_kind == 1u"));
        assert!(header.contains("prism_ghost_phase_lattice_count"));
        assert!(header.contains("prism_ghost_phase_lattice_write"));
        assert!(wrapper.contains("prism_ghost_phase_lattice_count"));
        assert!(wrapper.contains("prism_ghost_phase_lattice_write"));
        assert!(wrapper.contains("static_assert(sizeof(GhostPhaseLatticeAcceptedEdgeHost) == 56"));
    }

    #[test]
    fn absent_cooling_phase_does_not_relabel_standard_warm_tail() {
        let binding = StreamProtocolPhaseBinding::from_engine_boundaries(
            0, "standard_0".into(), "standard".into(), "aa".repeat(32),
            50_000, 200_000, 1_234_568, 1_234_568,
        ).unwrap();
        assert_eq!(binding.checked_phase_for(1_234_567), Some(PHASE_WARM_HOLD));
        assert_eq!(binding.checked_phase_for(1_234_568), None);
        assert_eq!(binding.intervals[3].start_step, binding.intervals[3].end_step);
        assert!(StreamProtocolPhaseBinding::from_engine_boundaries(
            0, "standard_0".into(), "standard".into(), "aa".repeat(32), 0, 0, 0, 0,
        ).is_err());
        assert!(StreamProtocolPhaseBinding::from_engine_boundaries(
            0, "standard_0".into(), "standard".into(), "aa".repeat(32), 20, 10, 30, 30,
        ).is_err());
    }

    #[test]
    fn exact_stream_phase_binding_is_gap_free_and_digest_bound() {
        let binding = StreamProtocolPhaseBinding::from_engine_boundaries(
            3,
            "TWIN16_0011".to_string(),
            "UvAromatic".to_string(),
            "aa".repeat(32),
            100,
            300,
            600,
            1000,
        )
        .unwrap();
        assert_eq!(binding.checked_phase_for(0), Some(PHASE_COLD_HOLD));
        assert_eq!(binding.checked_phase_for(100), Some(PHASE_HEATING));
        assert_eq!(binding.checked_phase_for(300), Some(PHASE_WARM_HOLD));
        assert_eq!(binding.checked_phase_for(999), Some(PHASE_COOLING));
        assert_eq!(binding.checked_phase_for(1000), None);
        let mut corrupted = binding.clone();
        corrupted.intervals[1].start_step += 1;
        assert!(corrupted.verify().is_err());
    }

    #[test]
    fn verified_segments_enforce_full_conservation_and_phase_binding() {
        let tmp = tempfile::tempdir().unwrap();
        let binding = StreamProtocolPhaseBinding::from_engine_boundaries(
            0,
            "TWIN16_0000".to_string(),
            "ThermalShock".to_string(),
            "bb".repeat(32),
            100,
            200,
            300,
            400,
        )
        .unwrap();
        let context = crate::ghost_segment::GhostSegmentContext {
            schema_kind: "prism_ghost_segment_context".to_string(),
            schema_version: 1,
            run_id: "run_test".to_string(),
            replica_id: "rep_00".to_string(),
            replica_seed: 42,
            stream_seed: 42,
            stream_id: 0,
            topology_sha256: "11".repeat(32),
            selected_engine_sha256: "22".repeat(32),
            record_abi_sha256: crate::ghost_tile::ghost_record_abi_sha256(),
            record_source_sha256: "44".repeat(32),
            thermodynamic_condition_id: binding.condition_id.clone(),
            protocol_id: binding.protocol_name.clone(),
            protocol_state_hash: binding.protocol_state_hash.clone(),
            phase_schedule_digest_sha256: binding.phase_schedule_digest_sha256.clone(),
        };
        let mut writer =
            crate::ghost_segment::GhostSegmentWriter::create(tmp.path(), "mycmax", context)
                .unwrap();
        writer.bind_overflow_admission(8, 2).unwrap();
        let r0 = synth_record(
            50,
            1.0,
            2.0,
            3.0,
            0.2,
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            None,
            1,
        );
        let r1 = synth_record(
            250,
            1.2,
            2.2,
            3.2,
            0.3,
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            None,
            1,
        );
        let mut payload = Vec::with_capacity(GHOST_RECORD_BYTES * 2);
        payload.extend_from_slice(&r0);
        payload.extend_from_slice(&r1);
        writer.append_payload(&payload, 2).unwrap();
        writer.complete_with_terminal_residual(0).unwrap();

        write_stream_phase_binding_atomic(tmp.path(), "mycmax", &binding).unwrap();
        let loaded = load_verified_ghost_segments_from_dir(tmp.path(), "mycmax", 50).unwrap();
        assert_eq!(loaded.produced_records, 2);
        assert_eq!(loaded.persisted_records, 2);
        assert_eq!(loaded.decoded_records, 2);
        assert_eq!(loaded.lattice_node_lineage_count, 2);
        assert!(loaded.equality_pass);
        assert!(loaded.phase_protocols_bound_by_stream);
        assert_eq!(loaded.overflow_loss_count, 0);
        assert!(loaded.overflow_conservation_pass);
        assert!(loaded.streams[0].ordinal_header_manifest_reconstruction_pass);
        assert_eq!(loaded.nodes[0].protocol_phase, PHASE_COLD_HOLD);
        assert_eq!(loaded.nodes[1].protocol_phase, PHASE_WARM_HOLD);
    }
}
