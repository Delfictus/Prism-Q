//! Durable Ghost telemetry segments.
//!
//! This module is the sole on-disk authority for Channel-B Ghost records.
//! GPU rings are bounded staging buffers, not artifacts.  At a synchronized
//! chunk boundary the engine drains the live payload through
//! [`GhostSegmentWriter`], which commits a versioned segment and an atomic
//! manifest before the ring counter may be reset.  Readers verify the complete
//! SHA-256 chain and ordinal coverage before exposing a single record.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};

/// Binary segment magic (`PRISM Ghost SEGment v4`). Version 4 adds the
/// manifest-bound engine overflow authority and binds every stream to one
/// explicit record contract: legacy segment-only schema 0 or exact MAR-v2
/// RichSpike schema 3.
pub const GHOST_SEGMENT_MAGIC: [u8; 8] = *b"PGHSEG04";
/// Durable segment schema version.
pub const GHOST_SEGMENT_SCHEMA_VERSION: u32 = 4;
/// Versioned overflow-conservation authority embedded in every v4 manifest.
pub const GHOST_OVERFLOW_CONSERVATION_SCHEMA_VERSION: u32 = 1;
/// One sector-aligned segment header.
pub const GHOST_SEGMENT_HEADER_BYTES: usize = 4096;
/// One fixed-width Ghost record under either admitted record schema.
pub const GHOST_SEGMENT_RECORD_BYTES: usize = 4096;
/// Production Ghost record schema carried by every segment record.
pub const GHOST_SEGMENT_RECORD_SCHEMA_VERSION: u32 = 3;
/// Legacy MAR-v1 record schema. These records retain the base Ghost frame but
/// do not claim a device-authenticated step, spatial projection, exact
/// cluster-residue membership, coordinate sidecar, or lattice eligibility.
pub const GHOST_SEGMENT_LEGACY_RECORD_SCHEMA_VERSION: u32 = 0;
const GHOST_RECORD_SCHEMA_OFFSET: usize = 128;
/// Canonical ABI identity derived by `ghost_record_abi_sha256`. This module is
/// available in feature-minimal verifier builds where `ghost_tile` is not, so
/// the digest is pinned here and parity-tested against the producer derivation
/// whenever the producer module is compiled.
const GHOST_V3_RECORD_ABI_SHA256: &str =
    "1d85fcd99fab9e8a68acb6101c8de013c04f793b8090f0385d87440ba368232f";
/// Canonical ABI identity derived by `ghost_legacy_record_abi_sha256`.
const GHOST_LEGACY_RECORD_ABI_SHA256: &str =
    "051592f4f525ad1176ad9d1fc43b2f9e5b9b2f589243da1e7c111aed2ecae0a1";

fn supported_record_schema(record_schema_version: u32) -> bool {
    matches!(
        record_schema_version,
        GHOST_SEGMENT_LEGACY_RECORD_SCHEMA_VERSION | GHOST_SEGMENT_RECORD_SCHEMA_VERSION
    )
}

fn expected_record_abi_sha256(record_schema_version: u32) -> Result<&'static str> {
    match record_schema_version {
        GHOST_SEGMENT_LEGACY_RECORD_SCHEMA_VERSION => Ok(GHOST_LEGACY_RECORD_ABI_SHA256),
        GHOST_SEGMENT_RECORD_SCHEMA_VERSION => Ok(GHOST_V3_RECORD_ABI_SHA256),
        _ => bail!("unsupported Ghost record schema {record_schema_version}"),
    }
}

/// Bind the manifest/header record schema to the only ABI that may carry it.
/// A syntactically valid SHA-256 is not sufficient: allowing schema 0 to name
/// the v3 ABI (or vice versa) would make a self-consistent artifact lie about
/// the semantics of its 4096-byte records.
fn verify_record_contract(context: &GhostSegmentContext, record_schema_version: u32) -> Result<()> {
    context.verify()?;
    let expected_record_abi_sha256 = expected_record_abi_sha256(record_schema_version)?;
    if context.record_abi_sha256 != expected_record_abi_sha256 {
        bail!(
            "Ghost record schema/ABI mismatch: schema={} expected_abi={} observed_abi={}",
            record_schema_version,
            expected_record_abi_sha256,
            context.record_abi_sha256,
        );
    }
    Ok(())
}

// WP5 wire-ABI pin: the independent segment verifier re-derives payload
// offsets from these exact values. A release build must fail to compile if a
// future edit changes one value without changing the governed wire contract.
const _: () = assert!(
    GHOST_SEGMENT_HEADER_BYTES == 4096
        && GHOST_SEGMENT_RECORD_BYTES == 4096
        && GHOST_SEGMENT_MAGIC[0] == b'P'
        && GHOST_SEGMENT_MAGIC[1] == b'G'
        && GHOST_SEGMENT_MAGIC[2] == b'H'
        && GHOST_SEGMENT_MAGIC[3] == b'S'
        && GHOST_SEGMENT_MAGIC[4] == b'E'
        && GHOST_SEGMENT_MAGIC[5] == b'G'
        && GHOST_SEGMENT_MAGIC[6] == b'0'
        && GHOST_SEGMENT_MAGIC[7] == b'4'
);

/// Engine-owned proof that bounded GPU staging never discarded a Ghost
/// record before the durable segment chain. An unbound authority is retained
/// for legacy/non-acceptance writers, but the production lattice loader
/// rejects it for Layer-05 conservation admission.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct GhostOverflowConservation {
    pub schema_kind: String,
    pub schema_version: u32,
    pub admission_status: String,
    pub ring_capacity_records: u64,
    pub replay_upper_bound_records: u64,
    pub replay_upper_bound_admission_pass: bool,
    pub drain_count: u64,
    pub empty_drain_count: u64,
    pub drain_produced_records: u64,
    pub drain_capacity_violation_count: u64,
    pub writer_payload_record_count_mismatch_count: u64,
    pub membership_overflow_record_count: u64,
    pub terminal_residual_records: Option<u64>,
    pub terminal_residual_zero_pass: bool,
}

impl GhostOverflowConservation {
    fn unbound() -> Self {
        Self {
            schema_kind: "prism_ghost_overflow_conservation".to_string(),
            schema_version: GHOST_OVERFLOW_CONSERVATION_SCHEMA_VERSION,
            admission_status: "EVIDENCE_NOT_FOUND".to_string(),
            ring_capacity_records: 0,
            replay_upper_bound_records: 0,
            replay_upper_bound_admission_pass: false,
            drain_count: 0,
            empty_drain_count: 0,
            drain_produced_records: 0,
            drain_capacity_violation_count: 0,
            writer_payload_record_count_mismatch_count: 0,
            membership_overflow_record_count: 0,
            terminal_residual_records: None,
            terminal_residual_zero_pass: false,
        }
    }

    fn verify_shape(&self) -> Result<()> {
        if self.schema_kind != "prism_ghost_overflow_conservation"
            || self.schema_version != GHOST_OVERFLOW_CONSERVATION_SCHEMA_VERSION
        {
            bail!("unsupported Ghost overflow-conservation schema");
        }
        match self.admission_status.as_str() {
            "EVIDENCE_NOT_FOUND" => {
                if self.ring_capacity_records != 0
                    || self.replay_upper_bound_records != 0
                    || self.replay_upper_bound_admission_pass
                    || self.drain_count != 0
                    || self.empty_drain_count != 0
                    || self.drain_produced_records != 0
                    || self.drain_capacity_violation_count != 0
                    || self.writer_payload_record_count_mismatch_count != 0
                    || self.membership_overflow_record_count != 0
                    || self.terminal_residual_records.is_some()
                    || self.terminal_residual_zero_pass
                {
                    bail!("unbound Ghost overflow authority contains claimed evidence");
                }
            }
            "BOUND_ENGINE_RUNTIME" => {
                if self.ring_capacity_records == 0
                    || self.replay_upper_bound_records == 0
                    || self.replay_upper_bound_records > self.ring_capacity_records
                    || !self.replay_upper_bound_admission_pass
                    || self.empty_drain_count > self.drain_count
                    || self.drain_capacity_violation_count != 0
                    || self.writer_payload_record_count_mismatch_count != 0
                    || self.membership_overflow_record_count != 0
                {
                    bail!("bound Ghost overflow authority violates admission or drain invariants");
                }
                if self.terminal_residual_zero_pass != (self.terminal_residual_records == Some(0)) {
                    bail!("Ghost terminal residual verdict is not mechanically bound");
                }
            }
            status => bail!("unknown Ghost overflow admission status: {status}"),
        }
        Ok(())
    }

    /// Complete Layer-05 authority predicate. This cannot be true for a
    /// legacy/unbound manifest or before an explicit terminal-zero closeout.
    pub fn acceptance_pass(&self) -> bool {
        self.verify_shape().is_ok()
            && self.admission_status == "BOUND_ENGINE_RUNTIME"
            && self.replay_upper_bound_admission_pass
            && self.drain_capacity_violation_count == 0
            && self.writer_payload_record_count_mismatch_count == 0
            && self.membership_overflow_record_count == 0
            && self.terminal_residual_records == Some(0)
            && self.terminal_residual_zero_pass
    }
}
const GHOST_V3_MEMBERSHIP_SCHEMA_VERSION_OFFSET: usize = 256;
const GHOST_V3_CLUSTER_RESIDUE_COUNT_OFFSET: usize = 260;
const GHOST_V3_CLUSTER_RESIDUE_CAPACITY_OFFSET: usize = 264;
const GHOST_V3_CLUSTER_RESIDUE_OVERFLOW_OFFSET: usize = 268;
const GHOST_V3_SOURCE_SPIKE_COUNT_OFFSET: usize = 272;
const GHOST_V3_UNRESOLVED_RESIDUE_SPIKE_COUNT_OFFSET: usize = 276;
const GHOST_V3_CLUSTER_RESIDUE_ENTRY_BYTES_OFFSET: usize = 280;
const GHOST_V3_GEOMETRY_FLAGS_OFFSET: usize = 284;
const GHOST_V3_COORDINATE_FRAME_SHA256_OFFSET: usize = 288;
const GHOST_V3_CLUSTER_RESIDUE_ENTRIES_OFFSET: usize = 320;
const GHOST_V3_MEMBERSHIP_SCHEMA_VERSION: u32 = 1;
const GHOST_V3_CLUSTER_RESIDUE_CAPACITY: u32 = 472;
const GHOST_V3_CLUSTER_RESIDUE_ENTRY_BYTES: u32 = 8;
const GHOST_V3_GEOMETRY_CLUSTER_RESIDUE_MEMBERSHIP_EXACT: u32 = 0x0001;
const GHOST_V3_GEOMETRY_COORDINATE_SIDECAR_REQUIRED: u32 = 0x0002;
const GHOST_SEGMENT_CONTEXT_DIGEST_OFFSET: usize = 132;
const GHOST_SEGMENT_CONTEXT_JSON_LENGTH_OFFSET: usize = 164;
const GHOST_SEGMENT_CONTEXT_JSON_OFFSET: usize = 168;
const GHOST_SEGMENT_CONTEXT_JSON_MAX_BYTES: usize = 2048;
const ZERO_SHA256: &str = "0000000000000000000000000000000000000000000000000000000000000000";

/// Immutable producer identity required before a Ghost segment can be
/// created, resumed, reconstructed, or exposed to a lattice consumer.
///
/// The context is embedded in full in every segment header and duplicated in
/// the atomic manifest. The digest therefore binds each record chain to the
/// run, replica, topology, executable, record implementation, thermodynamic
/// condition, protocol and exact phase schedule that produced it.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct GhostSegmentContext {
    pub schema_kind: String,
    pub schema_version: u32,
    pub run_id: String,
    pub replica_id: String,
    pub replica_seed: u64,
    pub stream_seed: u64,
    pub stream_id: u16,
    pub topology_sha256: String,
    pub selected_engine_sha256: String,
    pub record_abi_sha256: String,
    pub record_source_sha256: String,
    pub thermodynamic_condition_id: String,
    pub protocol_id: String,
    pub protocol_state_hash: String,
    pub phase_schedule_digest_sha256: String,
}

impl GhostSegmentContext {
    pub fn verify(&self) -> Result<()> {
        if self.schema_kind != "prism_ghost_segment_context" || self.schema_version != 1 {
            bail!("unsupported Ghost segment context schema");
        }
        if self.run_id.trim().is_empty()
            || self.replica_id.trim().is_empty()
            || self.thermodynamic_condition_id.trim().is_empty()
            || self.protocol_id.trim().is_empty()
            || self.protocol_state_hash.trim().is_empty()
        {
            bail!("Ghost segment context contains an empty identity field");
        }
        validate_sha256_field("topology_sha256", &self.topology_sha256)?;
        validate_sha256_field("selected_engine_sha256", &self.selected_engine_sha256)?;
        validate_sha256_field("record_abi_sha256", &self.record_abi_sha256)?;
        validate_sha256_field("record_source_sha256", &self.record_source_sha256)?;
        validate_sha256_field("protocol_state_hash", &self.protocol_state_hash)?;
        validate_sha256_field(
            "phase_schedule_digest_sha256",
            &self.phase_schedule_digest_sha256,
        )?;
        let context_json = serde_json::to_vec(self)?;
        if context_json.len() > GHOST_SEGMENT_CONTEXT_JSON_MAX_BYTES {
            bail!(
                "Ghost segment context is {} bytes; maximum is {}",
                context_json.len(),
                GHOST_SEGMENT_CONTEXT_JSON_MAX_BYTES
            );
        }
        Ok(())
    }

    pub fn digest_sha256(&self) -> Result<String> {
        self.verify()?;
        Ok(sha256_bytes(&serde_json::to_vec(self)?))
    }
}

/// One committed segment in the authoritative stream manifest.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct GhostSegmentEntry {
    /// Contiguous zero-based segment ordinal.
    pub segment_ordinal: u64,
    /// Global zero-based ordinal of the first record.
    pub first_record_ordinal: u64,
    /// Number of complete records in this segment.
    pub record_count: u64,
    /// Synchronized producer counter captured for this exact drain. This is
    /// persisted independently of file length for restart evidence.
    pub producer_drain_record_count: u64,
    /// Uniform record schema version verified for this segment.
    pub record_schema_version: u32,
    /// Byte offset of the payload in the segment file.
    pub payload_offset: u64,
    /// Payload byte count.
    pub payload_bytes: u64,
    /// Basename only; traversal is forbidden by the reader.
    pub file: String,
    /// SHA-256 of payload bytes.
    pub payload_sha256: String,
    /// SHA-256 of header plus payload.
    pub file_sha256: String,
    /// Previous committed segment's full-file SHA-256, or all-zero for root.
    pub previous_file_sha256: String,
    /// SHA-256 of the immutable producer context embedded in this header.
    pub context_digest_sha256: String,
    /// `true` only after `File::sync_all` succeeded before rename.
    pub fsync_pass: bool,
}

/// Atomic stream-level index for all Ghost segments.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct GhostSegmentManifest {
    /// Schema discriminator.
    pub schema_kind: String,
    /// Manifest schema version.
    pub schema_version: u32,
    /// Source stream identity.
    pub stream_id: u16,
    /// Immutable producer identity, duplicated in every segment header.
    pub context: GhostSegmentContext,
    /// SHA-256 of canonical serialized `context`.
    pub context_digest_sha256: String,
    /// Fixed record width.
    pub record_bytes: u32,
    /// Uniform production record schema required for every segment.
    pub record_schema_version: u32,
    /// Records reported by synchronized producer drains.
    pub produced_records: u64,
    /// Records durably committed in segments.
    pub persisted_records: u64,
    /// Total payload bytes durably committed.
    pub persisted_payload_bytes: u64,
    /// Ordered segment index.
    pub segments: Vec<GhostSegmentEntry>,
    /// Set only after a conservation check and final directory fsync.
    pub completed: bool,
    /// Manifest-level conservation verdict.
    pub produced_equals_persisted: bool,
    /// Versioned engine-owned staging-overflow conservation authority.
    pub overflow_conservation: GhostOverflowConservation,
}

impl GhostSegmentManifest {
    fn empty(context: GhostSegmentContext, record_schema_version: u32) -> Result<Self> {
        verify_record_contract(&context, record_schema_version)?;
        let context_digest_sha256 = context.digest_sha256()?;
        Ok(Self {
            schema_kind: "prism_ghost_segment_manifest".to_string(),
            schema_version: GHOST_SEGMENT_SCHEMA_VERSION,
            stream_id: context.stream_id,
            context,
            context_digest_sha256,
            record_bytes: GHOST_SEGMENT_RECORD_BYTES as u32,
            record_schema_version,
            produced_records: 0,
            persisted_records: 0,
            persisted_payload_bytes: 0,
            segments: Vec::new(),
            completed: false,
            produced_equals_persisted: true,
            overflow_conservation: GhostOverflowConservation::unbound(),
        })
    }
}

/// A fully verified segment payload.
#[derive(Clone, Debug)]
pub struct VerifiedGhostSegment {
    /// Manifest entry authenticated against the file.
    pub entry: GhostSegmentEntry,
    /// Exact record payload bytes (header excluded).
    pub payload: Vec<u8>,
}

/// A fully reconstructed, gap-free stream.
#[derive(Clone, Debug)]
pub struct VerifiedGhostStream {
    /// Verified manifest.
    pub manifest: GhostSegmentManifest,
    /// SHA-256 of the exact committed manifest bytes read from disk.
    pub manifest_sha256: String,
    /// Verified segment payloads in ordinal order.
    pub segments: Vec<VerifiedGhostSegment>,
}

/// Validate the semantic contract of every complete record in a payload.
///
/// This is deliberately shared by the writer, independent verifier, and
/// crash-reconstruction path. Hash agreement proves byte identity; it does not
/// prove that those bytes actually carry the record schema named by the
/// manifest/header or that a v3 membership population is internally complete.
fn validate_record_payload_semantics(payload: &[u8], expected_record_schema: u32) -> Result<()> {
    if !supported_record_schema(expected_record_schema) {
        bail!("unsupported Ghost record schema {expected_record_schema}");
    }
    if payload.len() % GHOST_SEGMENT_RECORD_BYTES != 0 {
        bail!(
            "partial Ghost payload: {} bytes is not divisible by {}",
            payload.len(),
            GHOST_SEGMENT_RECORD_BYTES,
        );
    }

    for (index, record) in payload.chunks_exact(GHOST_SEGMENT_RECORD_BYTES).enumerate() {
        let record_schema = u32::from_le_bytes(
            record[GHOST_RECORD_SCHEMA_OFFSET..GHOST_RECORD_SCHEMA_OFFSET + 4]
                .try_into()
                .expect("fixed Ghost schema slice"),
        );
        if record_schema != expected_record_schema {
            bail!(
                "Ghost segment rejects record schema drift at local record {}: expected={} observed={}",
                index,
                expected_record_schema,
                record_schema,
            );
        }
        if expected_record_schema != GHOST_SEGMENT_RECORD_SCHEMA_VERSION {
            continue;
        }

        let read_u32 = |offset: usize| -> u32 {
            u32::from_le_bytes(
                record[offset..offset + 4]
                    .try_into()
                    .expect("fixed Ghost v3 field"),
            )
        };
        let membership_schema = read_u32(GHOST_V3_MEMBERSHIP_SCHEMA_VERSION_OFFSET);
        let membership_count = read_u32(GHOST_V3_CLUSTER_RESIDUE_COUNT_OFFSET);
        let membership_capacity = read_u32(GHOST_V3_CLUSTER_RESIDUE_CAPACITY_OFFSET);
        let membership_overflow = read_u32(GHOST_V3_CLUSTER_RESIDUE_OVERFLOW_OFFSET);
        let source_spike_count = read_u32(GHOST_V3_SOURCE_SPIKE_COUNT_OFFSET);
        let unresolved_spike_count = read_u32(GHOST_V3_UNRESOLVED_RESIDUE_SPIKE_COUNT_OFFSET);
        let membership_entry_bytes = read_u32(GHOST_V3_CLUSTER_RESIDUE_ENTRY_BYTES_OFFSET);
        let geometry_flags = read_u32(GHOST_V3_GEOMETRY_FLAGS_OFFSET);
        if membership_schema != GHOST_V3_MEMBERSHIP_SCHEMA_VERSION
            || membership_overflow != 0
            || membership_capacity != GHOST_V3_CLUSTER_RESIDUE_CAPACITY
            || membership_count == 0
            || membership_count > membership_capacity
            || membership_entry_bytes != GHOST_V3_CLUSTER_RESIDUE_ENTRY_BYTES
            || geometry_flags & GHOST_V3_GEOMETRY_CLUSTER_RESIDUE_MEMBERSHIP_EXACT == 0
            || geometry_flags & GHOST_V3_GEOMETRY_COORDINATE_SIDECAR_REQUIRED == 0
        {
            bail!(
                "Ghost v3 cluster-residue membership rejected at local record {}: count={} capacity={} entry_bytes={} overflow={} flags=0x{:08x}",
                index,
                membership_count,
                membership_capacity,
                membership_entry_bytes,
                membership_overflow,
                geometry_flags,
            );
        }
        if record
            [GHOST_V3_COORDINATE_FRAME_SHA256_OFFSET..GHOST_V3_COORDINATE_FRAME_SHA256_OFFSET + 32]
            .iter()
            .any(|byte| *byte != 0)
        {
            bail!(
                "Ghost v3 record {} carries an unauthenticated inline coordinate digest",
                index,
            );
        }
        let mut residue_ids = BTreeSet::new();
        let mut resolved_spike_count = 0u32;
        for membership_index in 0..membership_count as usize {
            let offset = GHOST_V3_CLUSTER_RESIDUE_ENTRIES_OFFSET
                + membership_index * GHOST_V3_CLUSTER_RESIDUE_ENTRY_BYTES as usize;
            let residue_id = read_u32(offset);
            let spike_count = read_u32(offset + 4);
            if spike_count == 0 || !residue_ids.insert(residue_id) {
                bail!(
                    "Ghost v3 cluster-residue membership is zero or duplicate at local record {} row {}",
                    index,
                    membership_index,
                );
            }
            resolved_spike_count = resolved_spike_count
                .checked_add(spike_count)
                .context("Ghost v3 resolved spike-count overflow")?;
        }
        if resolved_spike_count.checked_add(unresolved_spike_count) != Some(source_spike_count) {
            bail!(
                "Ghost v3 source/membership conservation failed at local record {}: source={} resolved={} unresolved={}",
                index,
                source_spike_count,
                resolved_spike_count,
                unresolved_spike_count,
            );
        }
    }
    Ok(())
}

/// Single-writer durable sink for one Ghost stream.
pub struct GhostSegmentWriter {
    root: PathBuf,
    manifest_path: PathBuf,
    manifest: GhostSegmentManifest,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GhostSegmentContinuationV1 {
    pub output: crate::checkpoint_output::OutputDirectoryBoundary,
    pub manifest: GhostSegmentManifest,
}
impl GhostSegmentWriter {
    pub fn checkpoint(&self, run_root: &Path) -> Result<GhostSegmentContinuationV1> {
        let verified = verify_stream_root(&self.root)?.manifest;
        if verified != self.manifest {
            bail!("Ghost live manifest differs from durable checkpoint boundary");
        }
        Ok(GhostSegmentContinuationV1 {
            output: crate::checkpoint_output::capture_directory(
                run_root,
                &self.root,
                &["manifest.json"],
            )?,
            manifest: self.manifest.clone(),
        })
    }
    pub fn reopen_checkpoint(
        run_root: &Path,
        state: &GhostSegmentContinuationV1,
        expected: &GhostSegmentContext,
    ) -> Result<Self> {
        ensure_context_equal(&state.manifest.context, expected)?;
        let root = crate::checkpoint_output::restore_directory(run_root, &state.output)?;
        let manifest = verify_stream_root(&root)?.manifest;
        if manifest != state.manifest {
            bail!("Ghost restored manifest differs from checkpoint");
        }
        Ok(Self {
            manifest_path: root.join("manifest.json"),
            root,
            manifest,
        })
    }

    /// Create a new empty stream. Existing manifests fail closed; use
    /// [`resume_or_reconstruct`](Self::resume_or_reconstruct) for recovery.
    pub fn create(
        output_dir: &Path,
        topo_stem: &str,
        context: GhostSegmentContext,
    ) -> Result<Self> {
        Self::create_with_record_schema(
            output_dir,
            topo_stem,
            context,
            GHOST_SEGMENT_RECORD_SCHEMA_VERSION,
        )
    }

    /// Create a stream with an explicit producer record contract. Schema 3 is
    /// the exact MAR-v2 RichSpike product. Schema 0 is the segment-only MAR-v1
    /// compatibility product and is never eligible for exact lattice claims.
    pub fn create_with_record_schema(
        output_dir: &Path,
        topo_stem: &str,
        context: GhostSegmentContext,
        record_schema_version: u32,
    ) -> Result<Self> {
        validate_topology_stem(topo_stem)?;
        verify_record_contract(&context, record_schema_version)?;
        let stream_id = context.stream_id;
        let root = segment_root(output_dir, topo_stem, stream_id);
        std::fs::create_dir_all(&root)
            .with_context(|| format!("create Ghost segment root {}", root.display()))?;
        let manifest_path = root.join("manifest.json");
        if manifest_path.exists() {
            bail!(
                "Ghost segment manifest already exists for stream {}: {}",
                stream_id,
                manifest_path.display()
            );
        }
        let mut writer = Self {
            root,
            manifest_path,
            manifest: GhostSegmentManifest::empty(context, record_schema_version)?,
        };
        writer.commit_manifest()?;
        Ok(writer)
    }

    /// Resume a verified stream, or reconstruct its manifest from committed
    /// segment files if the manifest is absent. A completed stream cannot be
    /// resumed because appending would violate its terminal hash identity.
    pub fn resume_or_reconstruct(
        output_dir: &Path,
        topo_stem: &str,
        context: GhostSegmentContext,
    ) -> Result<Self> {
        Self::resume_or_reconstruct_with_record_schema(
            output_dir,
            topo_stem,
            context,
            GHOST_SEGMENT_RECORD_SCHEMA_VERSION,
        )
    }

    /// Resume or reconstruct a stream while requiring the same explicit
    /// producer record contract that was admitted at creation.
    pub fn resume_or_reconstruct_with_record_schema(
        output_dir: &Path,
        topo_stem: &str,
        context: GhostSegmentContext,
        record_schema_version: u32,
    ) -> Result<Self> {
        validate_topology_stem(topo_stem)?;
        verify_record_contract(&context, record_schema_version)?;
        let stream_id = context.stream_id;
        let root = segment_root(output_dir, topo_stem, stream_id);
        std::fs::create_dir_all(&root)
            .with_context(|| format!("create Ghost segment root {}", root.display()))?;
        let manifest_path = root.join("manifest.json");
        let manifest = if manifest_path.exists() {
            let verified = verify_stream_root(&root)?.manifest;
            ensure_context_equal(&verified.context, &context)?;
            if verified.record_schema_version != record_schema_version {
                bail!(
                    "Ghost segment record schema differs from requested resume contract: observed={} requested={}",
                    verified.record_schema_version,
                    record_schema_version,
                );
            }
            let committed_segment_count = std::fs::read_dir(&root)?
                .filter_map(|entry| entry.ok())
                .filter(|entry| {
                    entry
                        .file_name()
                        .to_str()
                        .map(|name| name.starts_with("segment_") && name.ends_with(".ghostbin"))
                        .unwrap_or(false)
                })
                .count();
            if committed_segment_count != verified.segments.len() {
                if verified.completed {
                    bail!(
                        "completed Ghost stream {} has unindexed committed segments and is immutable",
                        stream_id
                    );
                }
                // Crash window: a segment was fsynced+renamed but the atomic
                // manifest update did not commit. Reconstruct from explicit
                // per-segment producer watermarks and the SHA chain.
                let reconstructed = reconstruct_manifest(&root, &context, record_schema_version)?;
                write_manifest_atomic(&root, &manifest_path, &reconstructed)?;
                reconstructed
            } else {
                verified
            }
        } else {
            let reconstructed = reconstruct_manifest(&root, &context, record_schema_version)?;
            write_manifest_atomic(&root, &manifest_path, &reconstructed)?;
            reconstructed
        };
        if manifest.completed {
            bail!("cannot append to completed Ghost stream {}", stream_id);
        }
        ensure_context_equal(&manifest.context, &context)?;
        Ok(Self {
            root,
            manifest_path,
            manifest,
        })
    }

    /// Bind the staging capacity and worst-case replay emission before any
    /// producer drain. Rebinding is byte-equivalent/idempotent only; a
    /// reconstructed or legacy stream with existing records cannot be
    /// retroactively promoted because empty historical drains are not
    /// recoverable from segment headers.
    pub fn bind_overflow_admission(
        &mut self,
        ring_capacity_records: u64,
        replay_upper_bound_records: u64,
    ) -> Result<()> {
        if self.manifest.completed {
            bail!("cannot bind overflow admission on a completed Ghost stream");
        }
        if ring_capacity_records == 0
            || replay_upper_bound_records == 0
            || replay_upper_bound_records > ring_capacity_records
        {
            bail!(
                "Ghost replay upper-bound admission failed: replay_upper={} capacity={}",
                replay_upper_bound_records,
                ring_capacity_records
            );
        }
        let authority = &mut self.manifest.overflow_conservation;
        if authority.admission_status == "BOUND_ENGINE_RUNTIME" {
            if authority.ring_capacity_records != ring_capacity_records
                || authority.replay_upper_bound_records != replay_upper_bound_records
            {
                bail!("Ghost overflow admission rebinding differs from durable authority");
            }
            return Ok(());
        }
        authority.verify_shape()?;
        if !self.manifest.segments.is_empty()
            || self.manifest.produced_records != 0
            || self.manifest.persisted_records != 0
        {
            bail!("Ghost overflow admission cannot be reconstructed after record persistence");
        }
        *authority = GhostOverflowConservation {
            schema_kind: "prism_ghost_overflow_conservation".to_string(),
            schema_version: GHOST_OVERFLOW_CONSERVATION_SCHEMA_VERSION,
            admission_status: "BOUND_ENGINE_RUNTIME".to_string(),
            ring_capacity_records,
            replay_upper_bound_records,
            replay_upper_bound_admission_pass: true,
            drain_count: 0,
            empty_drain_count: 0,
            drain_produced_records: 0,
            drain_capacity_violation_count: 0,
            writer_payload_record_count_mismatch_count: 0,
            membership_overflow_record_count: 0,
            terminal_residual_records: None,
            terminal_residual_zero_pass: false,
        };
        self.commit_manifest()
    }

    /// Append one synchronized ring drain as a durable segment.
    ///
    /// `produced_records` is the raw producer counter for this drain. It must
    /// equal the number of complete payload records; clamped or partial drains
    /// are rejected before artifact mutation.
    pub fn append_payload(&mut self, payload: &[u8], produced_records: u64) -> Result<()> {
        if self.manifest.completed {
            bail!("cannot append to a completed Ghost stream");
        }
        if payload.len() % GHOST_SEGMENT_RECORD_BYTES != 0 {
            bail!(
                "partial Ghost payload: {} bytes is not divisible by {}",
                payload.len(),
                GHOST_SEGMENT_RECORD_BYTES
            );
        }
        let record_count = (payload.len() / GHOST_SEGMENT_RECORD_BYTES) as u64;
        if produced_records != record_count {
            bail!(
                "Ghost producer/payload mismatch: produced={} payload_records={}",
                produced_records,
                record_count
            );
        }
        let overflow_bound =
            self.manifest.overflow_conservation.admission_status == "BOUND_ENGINE_RUNTIME";
        if overflow_bound {
            self.manifest.overflow_conservation.verify_shape()?;
            if produced_records > self.manifest.overflow_conservation.ring_capacity_records {
                bail!(
                    "Ghost drain exceeds admitted staging capacity: produced={} capacity={}",
                    produced_records,
                    self.manifest.overflow_conservation.ring_capacity_records
                );
            }
        }
        if record_count == 0 {
            if overflow_bound {
                let authority = &mut self.manifest.overflow_conservation;
                authority.drain_count = authority
                    .drain_count
                    .checked_add(1)
                    .context("Ghost drain count overflow")?;
                authority.empty_drain_count = authority
                    .empty_drain_count
                    .checked_add(1)
                    .context("Ghost empty drain count overflow")?;
                self.commit_manifest()?;
            }
            return Ok(());
        }
        validate_record_payload_semantics(payload, self.manifest.record_schema_version)?;

        let segment_ordinal = self.manifest.segments.len() as u64;
        let first_record_ordinal = self.manifest.persisted_records;
        let previous_file_sha256 = self
            .manifest
            .segments
            .last()
            .map(|s| s.file_sha256.clone())
            .unwrap_or_else(|| ZERO_SHA256.to_string());
        let payload_sha256 = sha256_bytes(payload);
        let header = encode_header(
            self.manifest.stream_id,
            segment_ordinal,
            first_record_ordinal,
            record_count,
            produced_records,
            self.manifest.record_schema_version,
            &payload_sha256,
            &previous_file_sha256,
            &self.manifest.context,
        )?;
        let file_name = format!("segment_{segment_ordinal:08}.ghostbin");
        let final_path = self.root.join(&file_name);
        let partial_path = self.root.join(format!(".{file_name}.partial"));
        if final_path.exists() || partial_path.exists() {
            bail!("duplicate Ghost segment target: {}", final_path.display());
        }

        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&partial_path)
            .with_context(|| format!("create {}", partial_path.display()))?;
        file.write_all(&header)?;
        file.write_all(payload)?;
        file.sync_all()
            .with_context(|| format!("fsync {}", partial_path.display()))?;
        drop(file);
        std::fs::rename(&partial_path, &final_path).with_context(|| {
            format!(
                "commit Ghost segment {} -> {}",
                partial_path.display(),
                final_path.display()
            )
        })?;
        fsync_directory(&self.root)?;

        let file_sha256 = sha256_file(&final_path)?;
        let entry = GhostSegmentEntry {
            segment_ordinal,
            first_record_ordinal,
            record_count,
            producer_drain_record_count: produced_records,
            record_schema_version: self.manifest.record_schema_version,
            payload_offset: GHOST_SEGMENT_HEADER_BYTES as u64,
            payload_bytes: payload.len() as u64,
            file: file_name,
            payload_sha256,
            file_sha256,
            previous_file_sha256,
            context_digest_sha256: self.manifest.context_digest_sha256.clone(),
            fsync_pass: true,
        };
        self.manifest.produced_records = self
            .manifest
            .produced_records
            .checked_add(produced_records)
            .context("Ghost produced_records overflow")?;
        self.manifest.persisted_records = self
            .manifest
            .persisted_records
            .checked_add(record_count)
            .context("Ghost persisted_records overflow")?;
        self.manifest.persisted_payload_bytes = self
            .manifest
            .persisted_payload_bytes
            .checked_add(payload.len() as u64)
            .context("Ghost persisted_payload_bytes overflow")?;
        self.manifest.segments.push(entry);
        self.manifest.produced_equals_persisted =
            self.manifest.produced_records == self.manifest.persisted_records;
        if overflow_bound {
            let authority = &mut self.manifest.overflow_conservation;
            authority.drain_count = authority
                .drain_count
                .checked_add(1)
                .context("Ghost drain count overflow")?;
            authority.drain_produced_records = authority
                .drain_produced_records
                .checked_add(produced_records)
                .context("Ghost drain produced-record count overflow")?;
        }
        self.commit_manifest()
    }

    /// Seal the stream after enforcing producer/persistence conservation.
    pub fn complete(&mut self) -> Result<()> {
        if self.manifest.overflow_conservation.admission_status == "BOUND_ENGINE_RUNTIME" {
            bail!("bound Ghost overflow authority requires complete_with_terminal_residual");
        }
        if self.manifest.produced_records != self.manifest.persisted_records {
            bail!(
                "Ghost conservation failure at completion: produced={} persisted={}",
                self.manifest.produced_records,
                self.manifest.persisted_records
            );
        }
        self.manifest.completed = true;
        self.manifest.produced_equals_persisted = true;
        self.commit_manifest()?;
        fsync_directory(&self.root)
    }

    /// Seal a production stream only after the synchronized terminal staging
    /// counter has been observed. A nonzero residual is never normalized or
    /// drained into a PASS receipt.
    pub fn complete_with_terminal_residual(
        &mut self,
        terminal_residual_records: u64,
    ) -> Result<()> {
        if self.manifest.produced_records != self.manifest.persisted_records {
            bail!(
                "Ghost conservation failure at completion: produced={} persisted={}",
                self.manifest.produced_records,
                self.manifest.persisted_records
            );
        }
        let authority = &mut self.manifest.overflow_conservation;
        authority.verify_shape()?;
        if authority.admission_status != "BOUND_ENGINE_RUNTIME" {
            bail!("production Ghost closeout lacks bound overflow admission");
        }
        if terminal_residual_records != 0 {
            bail!(
                "Ghost terminal staging residual is nonzero: {}",
                terminal_residual_records
            );
        }
        if authority.drain_produced_records != self.manifest.produced_records
            || authority.drain_count
                != self.manifest.segments.len() as u64 + authority.empty_drain_count
        {
            bail!("Ghost drain population does not reconstruct durable segments");
        }
        authority.terminal_residual_records = Some(terminal_residual_records);
        authority.terminal_residual_zero_pass = true;
        authority.verify_shape()?;
        self.manifest.completed = true;
        self.manifest.produced_equals_persisted = true;
        self.commit_manifest()?;
        fsync_directory(&self.root)
    }

    /// Current durable manifest.
    pub fn manifest(&self) -> &GhostSegmentManifest {
        &self.manifest
    }

    fn commit_manifest(&mut self) -> Result<()> {
        write_manifest_atomic(&self.root, &self.manifest_path, &self.manifest)
    }
}

/// Locate the segment root for one stream.
pub fn segment_root(output_dir: &Path, topo_stem: &str, stream_id: u16) -> PathBuf {
    output_dir.join(format!(
        "{}_stream{:02}_ghost_segments",
        topo_stem, stream_id
    ))
}

/// Discover and verify every segment stream for a topology stem.
pub fn discover_verified_ghost_streams(
    output_dir: &Path,
    topo_stem: &str,
) -> Result<Vec<VerifiedGhostStream>> {
    validate_topology_stem(topo_stem)?;
    let prefix = format!("{topo_stem}_stream");
    let suffix = "_ghost_segments";
    let mut roots: Vec<(u16, PathBuf)> = Vec::new();
    for entry in std::fs::read_dir(output_dir)
        .with_context(|| format!("read_dir({})", output_dir.display()))?
    {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with(&prefix) || !name.ends_with(suffix) {
            continue;
        }
        let stream_text = &name[prefix.len()..name.len() - suffix.len()];
        let stream_id: u16 = stream_text
            .parse()
            .with_context(|| format!("invalid Ghost stream directory {name}"))?;
        roots.push((stream_id, entry.path()));
    }
    roots.sort_by_key(|(stream_id, _)| *stream_id);
    if roots.is_empty() {
        bail!(
            "no versioned Ghost segment streams found in {} for stem {}",
            output_dir.display(),
            topo_stem
        );
    }
    roots
        .into_iter()
        .map(|(stream_id, root)| {
            let verified = verify_stream_root(&root)?;
            if verified.manifest.stream_id != stream_id {
                bail!(
                    "Ghost stream directory/manifest mismatch: directory={} manifest={}",
                    stream_id,
                    verified.manifest.stream_id
                );
            }
            let committed_file_count = std::fs::read_dir(&root)?
                .filter_map(|entry| entry.ok())
                .filter(|entry| {
                    entry
                        .file_name()
                        .to_str()
                        .map(|name| name.starts_with("segment_") && name.ends_with(".ghostbin"))
                        .unwrap_or(false)
                })
                .count();
            if committed_file_count != verified.manifest.segments.len() {
                bail!(
                    "Ghost completed manifest leaves committed segments unindexed: files={} manifest_entries={} stream={}",
                    committed_file_count,
                    verified.manifest.segments.len(),
                    stream_id
                );
            }
            Ok(verified)
        })
        .collect()
}

/// Verify a manifest plus all of its segments, including chain continuity.
pub fn verify_stream_root(root: &Path) -> Result<VerifiedGhostStream> {
    let manifest_path = root.join("manifest.json");
    let bytes = std::fs::read(&manifest_path)
        .with_context(|| format!("read {}", manifest_path.display()))?;
    let manifest_sha256 = sha256_bytes(&bytes);
    let manifest: GhostSegmentManifest = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse {}", manifest_path.display()))?;
    verify_manifest_shape(&manifest)?;
    let mut segments = Vec::with_capacity(manifest.segments.len());
    let mut expected_segment = 0u64;
    let mut expected_record = 0u64;
    let mut reconstructed_produced = 0u64;
    let mut previous_sha = ZERO_SHA256.to_string();
    let mut persisted_bytes = 0u64;
    for entry in &manifest.segments {
        if entry.segment_ordinal != expected_segment {
            bail!(
                "Ghost segment gap/duplicate: expected ordinal {} observed {}",
                expected_segment,
                entry.segment_ordinal
            );
        }
        if entry.first_record_ordinal != expected_record {
            bail!(
                "Ghost record gap/duplicate: expected first {} observed {}",
                expected_record,
                entry.first_record_ordinal
            );
        }
        if entry.previous_file_sha256 != previous_sha {
            bail!(
                "Ghost SHA chain mismatch at segment {}",
                entry.segment_ordinal
            );
        }
        let expected_file = format!("segment_{:08}.ghostbin", entry.segment_ordinal);
        if entry.file != expected_file || !entry.fsync_pass {
            bail!(
                "Ghost segment name/fsync contract failed at ordinal {}",
                entry.segment_ordinal
            );
        }
        validate_basename(&entry.file)?;
        let path = root.join(&entry.file);
        let file_sha = sha256_file(&path)?;
        if file_sha != entry.file_sha256 {
            bail!("Ghost file corruption at {}", path.display());
        }
        let mut bytes = std::fs::read(&path)
            .with_context(|| format!("read Ghost segment {}", path.display()))?;
        if bytes.len() != GHOST_SEGMENT_HEADER_BYTES + entry.payload_bytes as usize {
            bail!("Ghost partial/trailing segment {}", path.display());
        }
        let header = decode_header(&bytes[..GHOST_SEGMENT_HEADER_BYTES])?;
        if header.stream_id != manifest.stream_id
            || header.segment_ordinal != entry.segment_ordinal
            || header.first_record_ordinal != entry.first_record_ordinal
            || header.record_count != entry.record_count
            || header.producer_drain_record_count != entry.producer_drain_record_count
            || header.record_schema_version != entry.record_schema_version
            || entry.record_schema_version != manifest.record_schema_version
            || entry.producer_drain_record_count != entry.record_count
            || header.payload_sha256 != entry.payload_sha256
            || header.previous_file_sha256 != entry.previous_file_sha256
            || header.context_digest_sha256 != entry.context_digest_sha256
            || entry.context_digest_sha256 != manifest.context_digest_sha256
            || header.context != manifest.context
        {
            bail!(
                "Ghost segment header/manifest mismatch at {}",
                path.display()
            );
        }
        let payload = bytes.split_off(GHOST_SEGMENT_HEADER_BYTES);
        if payload.len() as u64 != entry.record_count * GHOST_SEGMENT_RECORD_BYTES as u64
            || sha256_bytes(&payload) != entry.payload_sha256
        {
            bail!("Ghost payload corruption at {}", path.display());
        }
        validate_record_payload_semantics(&payload, manifest.record_schema_version)
            .with_context(|| format!("validate Ghost payload semantics at {}", path.display()))?;
        expected_segment += 1;
        expected_record += entry.record_count;
        reconstructed_produced += entry.producer_drain_record_count;
        persisted_bytes += entry.payload_bytes;
        previous_sha = entry.file_sha256.clone();
        segments.push(VerifiedGhostSegment {
            entry: entry.clone(),
            payload,
        });
    }
    if expected_record != manifest.persisted_records
        || reconstructed_produced != manifest.produced_records
        || persisted_bytes != manifest.persisted_payload_bytes
        || manifest.produced_records != manifest.persisted_records
        || !manifest.produced_equals_persisted
    {
        bail!(
            "Ghost manifest conservation failure: produced={} persisted={} reconstructed={} bytes={} reconstructed_bytes={}",
            manifest.produced_records,
            manifest.persisted_records,
            expected_record,
            manifest.persisted_payload_bytes,
            persisted_bytes
        );
    }
    manifest.overflow_conservation.verify_shape()?;
    if manifest.overflow_conservation.admission_status == "BOUND_ENGINE_RUNTIME" {
        if manifest.overflow_conservation.drain_produced_records != reconstructed_produced
            || manifest.overflow_conservation.drain_count
                != manifest.segments.len() as u64 + manifest.overflow_conservation.empty_drain_count
            || (manifest.completed && !manifest.overflow_conservation.acceptance_pass())
            || (!manifest.completed
                && (manifest
                    .overflow_conservation
                    .terminal_residual_records
                    .is_some()
                    || manifest.overflow_conservation.terminal_residual_zero_pass))
        {
            bail!("Ghost overflow authority does not reconstruct the verified stream");
        }
    }
    Ok(VerifiedGhostStream {
        manifest,
        manifest_sha256,
        segments,
    })
}

#[derive(Debug)]
struct DecodedHeader {
    stream_id: u16,
    segment_ordinal: u64,
    first_record_ordinal: u64,
    record_count: u64,
    producer_drain_record_count: u64,
    record_schema_version: u32,
    payload_sha256: String,
    previous_file_sha256: String,
    context: GhostSegmentContext,
    context_digest_sha256: String,
}

fn encode_header(
    stream_id: u16,
    segment_ordinal: u64,
    first_record_ordinal: u64,
    record_count: u64,
    producer_drain_record_count: u64,
    record_schema_version: u32,
    payload_sha256: &str,
    previous_file_sha256: &str,
    context: &GhostSegmentContext,
) -> Result<[u8; GHOST_SEGMENT_HEADER_BYTES]> {
    verify_record_contract(context, record_schema_version)?;
    if context.stream_id != stream_id {
        bail!(
            "Ghost segment header stream/context mismatch: header={} context={}",
            stream_id,
            context.stream_id
        );
    }
    let payload_digest = decode_sha256(payload_sha256)?;
    let previous_digest = decode_sha256(previous_file_sha256)?;
    let context_json = serde_json::to_vec(context)?;
    if context_json.len() > GHOST_SEGMENT_CONTEXT_JSON_MAX_BYTES {
        bail!("Ghost segment context exceeds header capacity");
    }
    let context_digest_sha256 = context.digest_sha256()?;
    let context_digest = decode_sha256(&context_digest_sha256)?;
    let mut h = [0u8; GHOST_SEGMENT_HEADER_BYTES];
    h[0..8].copy_from_slice(&GHOST_SEGMENT_MAGIC);
    h[8..12].copy_from_slice(&GHOST_SEGMENT_SCHEMA_VERSION.to_le_bytes());
    h[12..14].copy_from_slice(&stream_id.to_le_bytes());
    h[16..24].copy_from_slice(&segment_ordinal.to_le_bytes());
    h[24..32].copy_from_slice(&first_record_ordinal.to_le_bytes());
    h[32..40].copy_from_slice(&record_count.to_le_bytes());
    h[40..44].copy_from_slice(&(GHOST_SEGMENT_RECORD_BYTES as u32).to_le_bytes());
    h[44..48].copy_from_slice(&(GHOST_SEGMENT_HEADER_BYTES as u32).to_le_bytes());
    h[48..56].copy_from_slice(&(record_count * GHOST_SEGMENT_RECORD_BYTES as u64).to_le_bytes());
    // Explicit producer-drain watermark: reconstructors must use this field,
    // never infer the produced count from the file length.
    h[56..64].copy_from_slice(&producer_drain_record_count.to_le_bytes());
    h[64..96].copy_from_slice(&payload_digest);
    h[96..128].copy_from_slice(&previous_digest);
    h[128..132].copy_from_slice(&record_schema_version.to_le_bytes());
    h[GHOST_SEGMENT_CONTEXT_DIGEST_OFFSET..GHOST_SEGMENT_CONTEXT_DIGEST_OFFSET + 32]
        .copy_from_slice(&context_digest);
    h[GHOST_SEGMENT_CONTEXT_JSON_LENGTH_OFFSET..GHOST_SEGMENT_CONTEXT_JSON_LENGTH_OFFSET + 4]
        .copy_from_slice(&(context_json.len() as u32).to_le_bytes());
    h[GHOST_SEGMENT_CONTEXT_JSON_OFFSET..GHOST_SEGMENT_CONTEXT_JSON_OFFSET + context_json.len()]
        .copy_from_slice(&context_json);
    Ok(h)
}

fn decode_header(bytes: &[u8]) -> Result<DecodedHeader> {
    if bytes.len() != GHOST_SEGMENT_HEADER_BYTES {
        bail!("invalid Ghost segment header length {}", bytes.len());
    }
    if bytes[0..8] != GHOST_SEGMENT_MAGIC {
        bail!("invalid Ghost segment magic");
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let record_bytes = u32::from_le_bytes(bytes[40..44].try_into().unwrap());
    let header_bytes = u32::from_le_bytes(bytes[44..48].try_into().unwrap());
    if version != GHOST_SEGMENT_SCHEMA_VERSION
        || record_bytes != GHOST_SEGMENT_RECORD_BYTES as u32
        || header_bytes != GHOST_SEGMENT_HEADER_BYTES as u32
    {
        bail!("unsupported Ghost segment schema/layout");
    }
    let record_count = u64::from_le_bytes(bytes[32..40].try_into().unwrap());
    let payload_bytes = u64::from_le_bytes(bytes[48..56].try_into().unwrap());
    let producer_drain_record_count = u64::from_le_bytes(bytes[56..64].try_into().unwrap());
    let record_schema_version = u32::from_le_bytes(bytes[128..132].try_into().unwrap());
    let context_json_len = u32::from_le_bytes(
        bytes[GHOST_SEGMENT_CONTEXT_JSON_LENGTH_OFFSET
            ..GHOST_SEGMENT_CONTEXT_JSON_LENGTH_OFFSET + 4]
            .try_into()
            .unwrap(),
    ) as usize;
    if payload_bytes != record_count * GHOST_SEGMENT_RECORD_BYTES as u64 {
        bail!("Ghost segment header payload length mismatch");
    }
    if producer_drain_record_count != record_count {
        bail!("Ghost synchronized producer watermark mismatch");
    }
    if !supported_record_schema(record_schema_version) {
        bail!("unsupported Ghost record schema {record_schema_version}");
    }
    if context_json_len == 0 || context_json_len > GHOST_SEGMENT_CONTEXT_JSON_MAX_BYTES {
        bail!("invalid Ghost segment context length {context_json_len}");
    }
    let context_end = GHOST_SEGMENT_CONTEXT_JSON_OFFSET
        .checked_add(context_json_len)
        .context("Ghost segment context length overflow")?;
    if context_end > GHOST_SEGMENT_HEADER_BYTES {
        bail!("Ghost segment context exceeds header");
    }
    if bytes[context_end..].iter().any(|byte| *byte != 0) {
        bail!("Ghost segment header has nonzero bytes after embedded context");
    }
    let context: GhostSegmentContext =
        serde_json::from_slice(&bytes[GHOST_SEGMENT_CONTEXT_JSON_OFFSET..context_end])
            .context("decode embedded Ghost segment context")?;
    verify_record_contract(&context, record_schema_version)?;
    let context_digest_sha256 = encode_hex(
        &bytes[GHOST_SEGMENT_CONTEXT_DIGEST_OFFSET..GHOST_SEGMENT_CONTEXT_DIGEST_OFFSET + 32],
    );
    if context.digest_sha256()? != context_digest_sha256 {
        bail!("Ghost segment embedded context digest mismatch");
    }
    let stream_id = u16::from_le_bytes(bytes[12..14].try_into().unwrap());
    if context.stream_id != stream_id {
        bail!("Ghost segment embedded context stream mismatch");
    }
    Ok(DecodedHeader {
        stream_id,
        segment_ordinal: u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
        first_record_ordinal: u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
        record_count,
        producer_drain_record_count,
        record_schema_version,
        payload_sha256: encode_hex(&bytes[64..96]),
        previous_file_sha256: encode_hex(&bytes[96..128]),
        context,
        context_digest_sha256,
    })
}

fn reconstruct_manifest(
    root: &Path,
    expected_context: &GhostSegmentContext,
    expected_record_schema_version: u32,
) -> Result<GhostSegmentManifest> {
    verify_record_contract(expected_context, expected_record_schema_version)?;
    let stream_id = expected_context.stream_id;
    let mut paths: Vec<PathBuf> = std::fs::read_dir(root)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.starts_with("segment_") && name.ends_with(".ghostbin"))
                .unwrap_or(false)
        })
        .collect();
    paths.sort();
    let mut manifest =
        GhostSegmentManifest::empty(expected_context.clone(), expected_record_schema_version)?;
    let mut expected_record = 0u64;
    let mut previous_sha = ZERO_SHA256.to_string();
    for (ordinal, path) in paths.iter().enumerate() {
        let bytes = std::fs::read(path)?;
        if bytes.len() < GHOST_SEGMENT_HEADER_BYTES {
            bail!(
                "partial Ghost segment during reconstruction: {}",
                path.display()
            );
        }
        let header = decode_header(&bytes[..GHOST_SEGMENT_HEADER_BYTES])?;
        if header.stream_id != stream_id
            || header.segment_ordinal != ordinal as u64
            || header.first_record_ordinal != expected_record
            || header.previous_file_sha256 != previous_sha
            || header.producer_drain_record_count != header.record_count
            || header.record_schema_version != expected_record_schema_version
            || header.context_digest_sha256 != manifest.context_digest_sha256
            || header.context != manifest.context
        {
            bail!("Ghost reconstruction gap/duplicate at {}", path.display());
        }
        let payload = &bytes[GHOST_SEGMENT_HEADER_BYTES..];
        if payload.len() as u64 != header.record_count * GHOST_SEGMENT_RECORD_BYTES as u64
            || sha256_bytes(payload) != header.payload_sha256
        {
            bail!("Ghost reconstruction corruption at {}", path.display());
        }
        validate_record_payload_semantics(payload, expected_record_schema_version).with_context(
            || format!("validate reconstructed Ghost payload at {}", path.display()),
        )?;
        let file_sha256 = sha256_bytes(&bytes);
        let file = path.file_name().unwrap().to_string_lossy().to_string();
        manifest.segments.push(GhostSegmentEntry {
            segment_ordinal: ordinal as u64,
            first_record_ordinal: expected_record,
            record_count: header.record_count,
            producer_drain_record_count: header.producer_drain_record_count,
            record_schema_version: header.record_schema_version,
            payload_offset: GHOST_SEGMENT_HEADER_BYTES as u64,
            payload_bytes: payload.len() as u64,
            file,
            payload_sha256: header.payload_sha256,
            file_sha256: file_sha256.clone(),
            previous_file_sha256: previous_sha,
            context_digest_sha256: header.context_digest_sha256,
            fsync_pass: true,
        });
        expected_record += header.producer_drain_record_count;
        previous_sha = file_sha256;
    }
    manifest.produced_records = expected_record;
    manifest.persisted_records = expected_record;
    manifest.persisted_payload_bytes = expected_record * GHOST_SEGMENT_RECORD_BYTES as u64;
    manifest.produced_equals_persisted = true;
    Ok(manifest)
}

fn verify_manifest_shape(manifest: &GhostSegmentManifest) -> Result<()> {
    if manifest.schema_kind != "prism_ghost_segment_manifest"
        || manifest.schema_version != GHOST_SEGMENT_SCHEMA_VERSION
        || manifest.record_bytes != GHOST_SEGMENT_RECORD_BYTES as u32
        || !supported_record_schema(manifest.record_schema_version)
    {
        bail!("unsupported Ghost manifest schema/layout");
    }
    verify_record_contract(&manifest.context, manifest.record_schema_version)?;
    if manifest.context.stream_id != manifest.stream_id
        || manifest.context.digest_sha256()? != manifest.context_digest_sha256
    {
        bail!("Ghost manifest context identity/digest mismatch");
    }
    manifest.overflow_conservation.verify_shape()?;
    Ok(())
}

fn ensure_context_equal(
    observed: &GhostSegmentContext,
    expected: &GhostSegmentContext,
) -> Result<()> {
    observed.verify()?;
    expected.verify()?;
    if observed != expected || observed.digest_sha256()? != expected.digest_sha256()? {
        bail!("Ghost segment producer context differs from requested resume context");
    }
    Ok(())
}

fn write_manifest_atomic(
    root: &Path,
    manifest_path: &Path,
    manifest: &GhostSegmentManifest,
) -> Result<()> {
    let partial = root.join(".manifest.json.partial");
    let bytes = serde_json::to_vec_pretty(manifest)?;
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&partial)?;
    file.write_all(&bytes)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(&partial, manifest_path)?;
    fsync_directory(root)
}

fn validate_topology_stem(stem: &str) -> Result<()> {
    if stem.is_empty()
        || Path::new(stem).components().count() != 1
        || stem.contains('/')
        || stem.contains('\\')
    {
        bail!("invalid topology stem for Ghost artifact: {stem:?}");
    }
    Ok(())
}

fn validate_basename(name: &str) -> Result<()> {
    let mut components = Path::new(name).components();
    if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
        bail!("Ghost manifest file is not a basename: {name:?}");
    }
    Ok(())
}

fn fsync_directory(path: &Path) -> Result<()> {
    let dir = File::open(path).with_context(|| format!("open dir {}", path.display()))?;
    dir.sync_all()
        .with_context(|| format!("fsync dir {}", path.display()))
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(encode_hex(&hasher.finalize()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    encode_hex(&Sha256::digest(bytes))
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

fn decode_sha256(text: &str) -> Result<[u8; 32]> {
    if text.len() != 64 {
        bail!("invalid SHA-256 length {}", text.len());
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        let hi = hex_value(text.as_bytes()[i * 2])?;
        let lo = hex_value(text.as_bytes()[i * 2 + 1])?;
        *byte = (hi << 4) | lo;
    }
    Ok(out)
}

fn validate_sha256_field(field: &str, value: &str) -> Result<()> {
    let decoded = decode_sha256(value).with_context(|| format!("invalid Ghost {field}"))?;
    if encode_hex(&decoded) != value {
        bail!("Ghost {field} must be lowercase canonical SHA-256 hex");
    }
    Ok(())
}

fn hex_value(byte: u8) -> Result<u8> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => bail!("invalid hexadecimal byte"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_context(stream_id: u16) -> GhostSegmentContext {
        GhostSegmentContext {
            schema_kind: "prism_ghost_segment_context".to_string(),
            schema_version: 1,
            run_id: "run_test".to_string(),
            replica_id: "rep_00".to_string(),
            replica_seed: 42,
            stream_seed: 42u64.wrapping_add(stream_id as u64),
            stream_id,
            topology_sha256: "11".repeat(32),
            selected_engine_sha256: "22".repeat(32),
            record_abi_sha256: GHOST_V3_RECORD_ABI_SHA256.to_string(),
            record_source_sha256: "44".repeat(32),
            thermodynamic_condition_id: format!("condition_{stream_id}"),
            protocol_id: "protocol_test".to_string(),
            protocol_state_hash: "66".repeat(32),
            phase_schedule_digest_sha256: "55".repeat(32),
        }
    }

    fn legacy_test_context(stream_id: u16) -> GhostSegmentContext {
        let mut context = test_context(stream_id);
        context.record_abi_sha256 = GHOST_LEGACY_RECORD_ABI_SHA256.to_string();
        context
    }

    #[cfg(any(feature = "gpu", feature = "cpu-analysis"))]
    #[test]
    fn pinned_record_abi_contracts_match_the_producer_derivations() {
        validate_sha256_field("v3_record_abi", GHOST_V3_RECORD_ABI_SHA256).unwrap();
        validate_sha256_field("legacy_record_abi", GHOST_LEGACY_RECORD_ABI_SHA256).unwrap();
        assert_eq!(
            crate::ghost_tile::ghost_record_abi_sha256(),
            GHOST_V3_RECORD_ABI_SHA256,
        );
        assert_eq!(
            crate::ghost_tile::ghost_legacy_record_abi_sha256(),
            GHOST_LEGACY_RECORD_ABI_SHA256,
        );
        assert_eq!(
            GHOST_RECORD_SCHEMA_OFFSET,
            crate::ghost_tile::GHOST_V2_OFFSET_SCHEMA_VERSION,
        );
        assert_eq!(
            GHOST_V3_MEMBERSHIP_SCHEMA_VERSION_OFFSET,
            crate::ghost_tile::GHOST_V3_OFFSET_MEMBERSHIP_SCHEMA_VERSION,
        );
        assert_eq!(
            GHOST_V3_CLUSTER_RESIDUE_COUNT_OFFSET,
            crate::ghost_tile::GHOST_V3_OFFSET_CLUSTER_RESIDUE_COUNT,
        );
        assert_eq!(
            GHOST_V3_CLUSTER_RESIDUE_CAPACITY_OFFSET,
            crate::ghost_tile::GHOST_V3_OFFSET_CLUSTER_RESIDUE_CAPACITY,
        );
        assert_eq!(
            GHOST_V3_CLUSTER_RESIDUE_OVERFLOW_OFFSET,
            crate::ghost_tile::GHOST_V3_OFFSET_CLUSTER_RESIDUE_OVERFLOW,
        );
        assert_eq!(
            GHOST_V3_SOURCE_SPIKE_COUNT_OFFSET,
            crate::ghost_tile::GHOST_V3_OFFSET_SOURCE_SPIKE_COUNT,
        );
        assert_eq!(
            GHOST_V3_UNRESOLVED_RESIDUE_SPIKE_COUNT_OFFSET,
            crate::ghost_tile::GHOST_V3_OFFSET_UNRESOLVED_RESIDUE_SPIKE_COUNT,
        );
        assert_eq!(
            GHOST_V3_CLUSTER_RESIDUE_ENTRY_BYTES_OFFSET,
            crate::ghost_tile::GHOST_V3_OFFSET_CLUSTER_RESIDUE_ENTRY_BYTES,
        );
        assert_eq!(
            GHOST_V3_GEOMETRY_FLAGS_OFFSET,
            crate::ghost_tile::GHOST_V3_OFFSET_GEOMETRY_AVAILABILITY_FLAGS,
        );
        assert_eq!(
            GHOST_V3_COORDINATE_FRAME_SHA256_OFFSET,
            crate::ghost_tile::GHOST_V3_OFFSET_COORDINATE_FRAME_SHA256,
        );
        assert_eq!(
            GHOST_V3_CLUSTER_RESIDUE_ENTRIES_OFFSET,
            crate::ghost_tile::GHOST_V3_OFFSET_CLUSTER_RESIDUE_ENTRIES,
        );
        assert_eq!(
            GHOST_V3_MEMBERSHIP_SCHEMA_VERSION,
            crate::ghost_tile::GHOST_V3_MEMBERSHIP_SCHEMA_VERSION,
        );
        assert_eq!(
            GHOST_V3_CLUSTER_RESIDUE_ENTRY_BYTES as usize,
            crate::ghost_tile::GHOST_V3_CLUSTER_RESIDUE_ENTRY_BYTES,
        );
        assert_eq!(
            GHOST_V3_CLUSTER_RESIDUE_CAPACITY as usize,
            crate::ghost_tile::GHOST_V3_CLUSTER_RESIDUE_CAPACITY,
        );
        assert_eq!(
            GHOST_V3_GEOMETRY_CLUSTER_RESIDUE_MEMBERSHIP_EXACT,
            crate::ghost_tile::GHOST_V3_GEOMETRY_CLUSTER_RESIDUE_MEMBERSHIP_EXACT,
        );
        assert_eq!(
            GHOST_V3_GEOMETRY_COORDINATE_SIDECAR_REQUIRED,
            crate::ghost_tile::GHOST_V3_GEOMETRY_COORDINATE_SIDECAR_REQUIRED,
        );
    }

    fn records(n: usize, base: u64) -> Vec<u8> {
        let mut bytes = vec![0u8; n * GHOST_SEGMENT_RECORD_BYTES];
        for i in 0..n {
            let off = i * GHOST_SEGMENT_RECORD_BYTES;
            bytes[off..off + 8].copy_from_slice(&(base + i as u64).to_le_bytes());
            bytes[off + GHOST_RECORD_SCHEMA_OFFSET..off + GHOST_RECORD_SCHEMA_OFFSET + 4]
                .copy_from_slice(&GHOST_SEGMENT_RECORD_SCHEMA_VERSION.to_le_bytes());
            bytes[off + GHOST_V3_MEMBERSHIP_SCHEMA_VERSION_OFFSET
                ..off + GHOST_V3_MEMBERSHIP_SCHEMA_VERSION_OFFSET + 4]
                .copy_from_slice(&GHOST_V3_MEMBERSHIP_SCHEMA_VERSION.to_le_bytes());
            bytes[off + GHOST_V3_CLUSTER_RESIDUE_COUNT_OFFSET
                ..off + GHOST_V3_CLUSTER_RESIDUE_COUNT_OFFSET + 4]
                .copy_from_slice(&1u32.to_le_bytes());
            bytes[off + GHOST_V3_CLUSTER_RESIDUE_CAPACITY_OFFSET
                ..off + GHOST_V3_CLUSTER_RESIDUE_CAPACITY_OFFSET + 4]
                .copy_from_slice(&GHOST_V3_CLUSTER_RESIDUE_CAPACITY.to_le_bytes());
            bytes[off + GHOST_V3_SOURCE_SPIKE_COUNT_OFFSET
                ..off + GHOST_V3_SOURCE_SPIKE_COUNT_OFFSET + 4]
                .copy_from_slice(&1u32.to_le_bytes());
            bytes[off + GHOST_V3_CLUSTER_RESIDUE_ENTRY_BYTES_OFFSET
                ..off + GHOST_V3_CLUSTER_RESIDUE_ENTRY_BYTES_OFFSET + 4]
                .copy_from_slice(&GHOST_V3_CLUSTER_RESIDUE_ENTRY_BYTES.to_le_bytes());
            bytes[off + GHOST_V3_GEOMETRY_FLAGS_OFFSET..off + GHOST_V3_GEOMETRY_FLAGS_OFFSET + 4]
                .copy_from_slice(
                    &(GHOST_V3_GEOMETRY_CLUSTER_RESIDUE_MEMBERSHIP_EXACT
                        | GHOST_V3_GEOMETRY_COORDINATE_SIDECAR_REQUIRED)
                        .to_le_bytes(),
                );
            bytes[off + GHOST_V3_CLUSTER_RESIDUE_ENTRIES_OFFSET
                ..off + GHOST_V3_CLUSTER_RESIDUE_ENTRIES_OFFSET + 4]
                .copy_from_slice(&7u32.to_le_bytes());
            bytes[off + GHOST_V3_CLUSTER_RESIDUE_ENTRIES_OFFSET + 4
                ..off + GHOST_V3_CLUSTER_RESIDUE_ENTRIES_OFFSET + 8]
                .copy_from_slice(&1u32.to_le_bytes());
        }
        bytes
    }

    fn legacy_records(n: usize, base: u64) -> Vec<u8> {
        let mut bytes = vec![0u8; n * GHOST_SEGMENT_RECORD_BYTES];
        for i in 0..n {
            let off = i * GHOST_SEGMENT_RECORD_BYTES;
            bytes[off..off + 8].copy_from_slice(&(base + i as u64).to_le_bytes());
            bytes[off + GHOST_RECORD_SCHEMA_OFFSET..off + GHOST_RECORD_SCHEMA_OFFSET + 4]
                .copy_from_slice(&GHOST_SEGMENT_LEGACY_RECORD_SCHEMA_VERSION.to_le_bytes());
        }
        bytes
    }

    fn rewrite_single_segment_payload_and_reseal(root: &Path, mutate: impl FnOnce(&mut [u8])) {
        let manifest_path = root.join("manifest.json");
        let mut manifest: GhostSegmentManifest =
            serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
        assert_eq!(manifest.segments.len(), 1);
        let segment_path = root.join(&manifest.segments[0].file);
        let mut segment_bytes = std::fs::read(&segment_path).unwrap();
        mutate(&mut segment_bytes[GHOST_SEGMENT_HEADER_BYTES..]);

        let payload_sha256 = sha256_bytes(&segment_bytes[GHOST_SEGMENT_HEADER_BYTES..]);
        segment_bytes[64..96].copy_from_slice(&decode_sha256(&payload_sha256).unwrap());
        let file_sha256 = sha256_bytes(&segment_bytes);
        std::fs::write(&segment_path, &segment_bytes).unwrap();
        manifest.segments[0].payload_sha256 = payload_sha256;
        manifest.segments[0].file_sha256 = file_sha256;
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn record_schema_requires_its_exact_abi_identity() {
        let schema0_with_v3_abi = tempfile::tempdir().unwrap();
        let rejected_legacy_root = segment_root(schema0_with_v3_abi.path(), "legacy", 12);
        let error = GhostSegmentWriter::create_with_record_schema(
            schema0_with_v3_abi.path(),
            "legacy",
            test_context(12),
            GHOST_SEGMENT_LEGACY_RECORD_SCHEMA_VERSION,
        )
        .err()
        .expect("schema 0 must reject the v3 ABI identity");
        assert!(error
            .to_string()
            .contains("Ghost record schema/ABI mismatch"));
        assert!(!rejected_legacy_root.exists());

        let schema3_with_legacy_abi = tempfile::tempdir().unwrap();
        let rejected_exact_root = segment_root(schema3_with_legacy_abi.path(), "exact", 13);
        let error = GhostSegmentWriter::create_with_record_schema(
            schema3_with_legacy_abi.path(),
            "exact",
            legacy_test_context(13),
            GHOST_SEGMENT_RECORD_SCHEMA_VERSION,
        )
        .err()
        .expect("schema 3 must reject the legacy ABI identity");
        assert!(error
            .to_string()
            .contains("Ghost record schema/ABI mismatch"));
        assert!(!rejected_exact_root.exists());
    }

    #[test]
    fn legacy_segment_only_records_are_persisted_without_v3_claims() {
        let tmp = tempfile::tempdir().unwrap();
        let mut writer = GhostSegmentWriter::create_with_record_schema(
            tmp.path(),
            "mycmax",
            legacy_test_context(8),
            GHOST_SEGMENT_LEGACY_RECORD_SCHEMA_VERSION,
        )
        .unwrap();
        writer.bind_overflow_admission(8, 2).unwrap();
        writer.append_payload(&legacy_records(2, 6500), 2).unwrap();
        writer.append_payload(&[], 0).unwrap();
        writer.complete_with_terminal_residual(0).unwrap();

        let verified = verify_stream_root(&segment_root(tmp.path(), "mycmax", 8)).unwrap();
        assert_eq!(
            verified.manifest.record_schema_version,
            GHOST_SEGMENT_LEGACY_RECORD_SCHEMA_VERSION
        );
        assert_eq!(verified.manifest.persisted_records, 2);
        assert!(verified.manifest.overflow_conservation.acceptance_pass());
        assert!(
            verified.segments[0].payload[GHOST_V3_MEMBERSHIP_SCHEMA_VERSION_OFFSET
                ..GHOST_V3_MEMBERSHIP_SCHEMA_VERSION_OFFSET + 4]
                .iter()
                .all(|byte| *byte == 0)
        );
    }

    #[test]
    fn record_schema_cross_promotion_is_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let mut legacy = GhostSegmentWriter::create_with_record_schema(
            tmp.path(),
            "legacy",
            legacy_test_context(9),
            GHOST_SEGMENT_LEGACY_RECORD_SCHEMA_VERSION,
        )
        .unwrap();
        legacy.bind_overflow_admission(8, 1).unwrap();
        assert!(legacy.append_payload(&records(1, 0), 1).is_err());
        assert_eq!(legacy.manifest().persisted_records, 0);

        let mut exact = GhostSegmentWriter::create(tmp.path(), "exact", test_context(10)).unwrap();
        exact.bind_overflow_admission(8, 1).unwrap();
        assert!(exact.append_payload(&legacy_records(1, 0), 1).is_err());
        assert_eq!(exact.manifest().persisted_records, 0);
    }

    #[test]
    fn resume_requires_the_original_record_schema_contract() {
        let tmp = tempfile::tempdir().unwrap();
        let context = legacy_test_context(11);
        let mut writer = GhostSegmentWriter::create_with_record_schema(
            tmp.path(),
            "mycmax",
            context.clone(),
            GHOST_SEGMENT_LEGACY_RECORD_SCHEMA_VERSION,
        )
        .unwrap();
        writer.bind_overflow_admission(8, 1).unwrap();
        writer.append_payload(&legacy_records(1, 0), 1).unwrap();

        assert!(
            GhostSegmentWriter::resume_or_reconstruct_with_record_schema(
                tmp.path(),
                "mycmax",
                context,
                GHOST_SEGMENT_RECORD_SCHEMA_VERSION,
            )
            .is_err()
        );
    }

    #[test]
    fn verifier_rejects_hash_consistent_record_schema_drift() {
        let tmp = tempfile::tempdir().unwrap();
        let mut writer =
            GhostSegmentWriter::create(tmp.path(), "mycmax", test_context(14)).unwrap();
        writer.append_payload(&records(1, 0), 1).unwrap();
        writer.complete().unwrap();
        let root = segment_root(tmp.path(), "mycmax", 14);
        rewrite_single_segment_payload_and_reseal(&root, |payload| {
            payload[GHOST_RECORD_SCHEMA_OFFSET..GHOST_RECORD_SCHEMA_OFFSET + 4]
                .copy_from_slice(&GHOST_SEGMENT_LEGACY_RECORD_SCHEMA_VERSION.to_le_bytes());
        });

        let error = verify_stream_root(&root)
            .err()
            .expect("semantic verification must reject a hash-consistent schema drift");
        assert!(format!("{error:#}").contains("record schema drift"));
    }

    #[test]
    fn verifier_rejects_hash_consistent_v3_membership_drift() {
        let tmp = tempfile::tempdir().unwrap();
        let mut writer =
            GhostSegmentWriter::create(tmp.path(), "mycmax", test_context(15)).unwrap();
        writer.append_payload(&records(1, 0), 1).unwrap();
        writer.complete().unwrap();
        let root = segment_root(tmp.path(), "mycmax", 15);
        rewrite_single_segment_payload_and_reseal(&root, |payload| {
            payload[GHOST_V3_SOURCE_SPIKE_COUNT_OFFSET..GHOST_V3_SOURCE_SPIKE_COUNT_OFFSET + 4]
                .copy_from_slice(&2u32.to_le_bytes());
        });

        let error = verify_stream_root(&root)
            .err()
            .expect("semantic verification must reject hash-consistent membership drift");
        assert!(format!("{error:#}").contains("source/membership conservation"));
    }

    #[test]
    fn verifier_rejects_manifest_schema_abi_cross_binding() {
        let tmp = tempfile::tempdir().unwrap();
        let mut writer =
            GhostSegmentWriter::create(tmp.path(), "mycmax", test_context(16)).unwrap();
        writer.append_payload(&records(1, 0), 1).unwrap();
        writer.complete().unwrap();
        let root = segment_root(tmp.path(), "mycmax", 16);
        let manifest_path = root.join("manifest.json");
        let mut manifest: GhostSegmentManifest =
            serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
        manifest.record_schema_version = GHOST_SEGMENT_LEGACY_RECORD_SCHEMA_VERSION;
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();

        let error = verify_stream_root(&root)
            .err()
            .expect("manifest schema cannot cross-bind the embedded v3 ABI");
        assert!(error
            .to_string()
            .contains("Ghost record schema/ABI mismatch"));
    }

    #[test]
    fn reconstruction_rejects_semantically_invalid_resealed_payload() {
        let tmp = tempfile::tempdir().unwrap();
        let context = test_context(17);
        let mut writer = GhostSegmentWriter::create(tmp.path(), "mycmax", context.clone()).unwrap();
        writer.append_payload(&records(1, 0), 1).unwrap();
        let root = segment_root(tmp.path(), "mycmax", 17);
        rewrite_single_segment_payload_and_reseal(&root, |payload| {
            payload[GHOST_V3_SOURCE_SPIKE_COUNT_OFFSET..GHOST_V3_SOURCE_SPIKE_COUNT_OFFSET + 4]
                .copy_from_slice(&2u32.to_le_bytes());
        });
        let manifest_path = root.join("manifest.json");
        std::fs::remove_file(&manifest_path).unwrap();

        let error = GhostSegmentWriter::resume_or_reconstruct(tmp.path(), "mycmax", context)
            .err()
            .expect("reconstruction must rerun semantic record validation");
        assert!(format!("{error:#}").contains("source/membership conservation"));
        assert!(!manifest_path.exists());
    }

    #[test]
    fn forced_rollover_and_reconstruction_preserve_exact_lineage() {
        let tmp = tempfile::tempdir().unwrap();
        let mut writer = GhostSegmentWriter::create(tmp.path(), "mycmax", test_context(3)).unwrap();
        for i in 0..5 {
            writer.append_payload(&records(2, i * 2), 2).unwrap();
        }
        assert_eq!(writer.manifest().segments.len(), 5);
        assert_eq!(writer.manifest().produced_records, 10);
        let root = segment_root(tmp.path(), "mycmax", 3);
        std::fs::remove_file(root.join("manifest.json")).unwrap();
        let mut resumed =
            GhostSegmentWriter::resume_or_reconstruct(tmp.path(), "mycmax", test_context(3))
                .unwrap();
        resumed.append_payload(&records(1, 10), 1).unwrap();
        resumed.complete().unwrap();
        let verified = verify_stream_root(&root).unwrap();
        assert_eq!(verified.manifest.persisted_records, 11);
        assert_eq!(verified.segments.len(), 6);
    }

    #[test]
    fn partial_payload_is_rejected_without_manifest_mutation() {
        let tmp = tempfile::tempdir().unwrap();
        let mut writer = GhostSegmentWriter::create(tmp.path(), "mycmax", test_context(0)).unwrap();
        assert!(writer.append_payload(&[0u8; 4095], 1).is_err());
        assert_eq!(writer.manifest().persisted_records, 0);
    }

    #[test]
    fn v3_cluster_residue_overflow_is_rejected_before_persistence() {
        let tmp = tempfile::tempdir().unwrap();
        let mut writer = GhostSegmentWriter::create(tmp.path(), "mycmax", test_context(0)).unwrap();
        let mut payload = records(1, 0);
        payload[GHOST_V3_CLUSTER_RESIDUE_OVERFLOW_OFFSET
            ..GHOST_V3_CLUSTER_RESIDUE_OVERFLOW_OFFSET + 4]
            .copy_from_slice(&1u32.to_le_bytes());
        assert!(writer.append_payload(&payload, 1).is_err());
        assert_eq!(writer.manifest().persisted_records, 0);
        assert!(writer.manifest().segments.is_empty());
    }

    #[test]
    fn v3_membership_conservation_is_rejected_before_persistence() {
        let tmp = tempfile::tempdir().unwrap();
        let mut writer = GhostSegmentWriter::create(tmp.path(), "mycmax", test_context(0)).unwrap();
        let mut payload = records(1, 0);
        payload[GHOST_V3_SOURCE_SPIKE_COUNT_OFFSET..GHOST_V3_SOURCE_SPIKE_COUNT_OFFSET + 4]
            .copy_from_slice(&2u32.to_le_bytes());
        assert!(writer.append_payload(&payload, 1).is_err());
        assert_eq!(writer.manifest().persisted_records, 0);
        assert!(writer.manifest().segments.is_empty());
    }

    #[test]
    fn resume_rejects_changed_immutable_context() {
        let tmp = tempfile::tempdir().unwrap();
        let mut writer = GhostSegmentWriter::create(tmp.path(), "mycmax", test_context(0)).unwrap();
        writer.append_payload(&records(1, 0), 1).unwrap();
        let mut changed = test_context(0);
        changed.selected_engine_sha256 = "aa".repeat(32);
        assert!(GhostSegmentWriter::resume_or_reconstruct(tmp.path(), "mycmax", changed).is_err());
        assert_eq!(writer.manifest().persisted_records, 1);
    }

    #[test]
    fn reconstruction_requires_header_context_to_match_requested_context() {
        let tmp = tempfile::tempdir().unwrap();
        let mut writer = GhostSegmentWriter::create(tmp.path(), "mycmax", test_context(0)).unwrap();
        writer.append_payload(&records(1, 0), 1).unwrap();
        let root = segment_root(tmp.path(), "mycmax", 0);
        std::fs::remove_file(root.join("manifest.json")).unwrap();
        let mut changed = test_context(0);
        changed.phase_schedule_digest_sha256 = "aa".repeat(32);
        assert!(GhostSegmentWriter::resume_or_reconstruct(tmp.path(), "mycmax", changed).is_err());
    }

    #[test]
    fn corruption_is_detected() {
        let tmp = tempfile::tempdir().unwrap();
        let mut writer = GhostSegmentWriter::create(tmp.path(), "mycmax", test_context(0)).unwrap();
        writer.append_payload(&records(2, 0), 2).unwrap();
        writer.complete().unwrap();
        let root = segment_root(tmp.path(), "mycmax", 0);
        let path = root.join("segment_00000000.ghostbin");
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[GHOST_SEGMENT_HEADER_BYTES + 7] ^= 0xff;
        std::fs::write(&path, bytes).unwrap();
        assert!(verify_stream_root(&root).is_err());
    }

    #[test]
    fn manifest_gap_and_duplicate_are_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let mut writer = GhostSegmentWriter::create(tmp.path(), "mycmax", test_context(0)).unwrap();
        writer.append_payload(&records(1, 0), 1).unwrap();
        writer.append_payload(&records(1, 1), 1).unwrap();
        writer.complete().unwrap();
        let root = segment_root(tmp.path(), "mycmax", 0);
        let manifest_path = root.join("manifest.json");
        let mut m: GhostSegmentManifest =
            serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
        m.segments[1].segment_ordinal = 0;
        std::fs::write(&manifest_path, serde_json::to_vec(&m).unwrap()).unwrap();
        assert!(verify_stream_root(&root).is_err());
    }

    #[test]
    fn manifest_cannot_claim_an_unfsynced_segment() {
        let tmp = tempfile::tempdir().unwrap();
        let mut writer = GhostSegmentWriter::create(tmp.path(), "mycmax", test_context(0)).unwrap();
        writer.append_payload(&records(1, 0), 1).unwrap();
        writer.complete().unwrap();
        let root = segment_root(tmp.path(), "mycmax", 0);
        let manifest_path = root.join("manifest.json");
        let mut manifest: GhostSegmentManifest =
            serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
        manifest.segments[0].fsync_pass = false;
        std::fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        assert!(verify_stream_root(&root).is_err());
    }

    #[test]
    fn orphan_partial_segment_is_not_reconstructed() {
        let tmp = tempfile::tempdir().unwrap();
        let root = segment_root(tmp.path(), "mycmax", 0);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join(".segment_00000000.ghostbin.partial"), [7u8; 20]).unwrap();
        let writer =
            GhostSegmentWriter::resume_or_reconstruct(tmp.path(), "mycmax", test_context(0))
                .unwrap();
        assert_eq!(writer.manifest().persisted_records, 0);
    }

    #[test]
    fn committed_orphan_after_stale_manifest_is_reconstructed_once() {
        let tmp = tempfile::tempdir().unwrap();
        let mut writer = GhostSegmentWriter::create(tmp.path(), "mycmax", test_context(1)).unwrap();
        writer.append_payload(&records(2, 0), 2).unwrap();
        let root = segment_root(tmp.path(), "mycmax", 1);
        let manifest_path = root.join("manifest.json");
        let stale_manifest = std::fs::read(&manifest_path).unwrap();

        // Model the fsync+rename/manifest-update crash window by restoring the
        // previously committed manifest after segment 1 has reached its final
        // name. Recovery must discover that segment from its header watermark
        // and SHA chain exactly once.
        writer.append_payload(&records(3, 2), 3).unwrap();
        std::fs::write(&manifest_path, stale_manifest).unwrap();
        std::fs::File::open(&manifest_path)
            .unwrap()
            .sync_all()
            .unwrap();

        let mut resumed =
            GhostSegmentWriter::resume_or_reconstruct(tmp.path(), "mycmax", test_context(1))
                .unwrap();
        assert_eq!(resumed.manifest().segments.len(), 2);
        assert_eq!(resumed.manifest().produced_records, 5);
        assert_eq!(resumed.manifest().persisted_records, 5);
        resumed.complete().unwrap();
        let verified = verify_stream_root(&root).unwrap();
        assert_eq!(verified.manifest.segments.len(), 2);
        assert_eq!(verified.manifest.produced_records, 5);
    }

    #[test]
    fn completed_stream_rejects_unindexed_committed_file() {
        let tmp = tempfile::tempdir().unwrap();
        let mut writer = GhostSegmentWriter::create(tmp.path(), "mycmax", test_context(2)).unwrap();
        writer.append_payload(&records(1, 0), 1).unwrap();
        writer.complete().unwrap();
        let root = segment_root(tmp.path(), "mycmax", 2);
        std::fs::copy(
            root.join("segment_00000000.ghostbin"),
            root.join("segment_00000001.ghostbin"),
        )
        .unwrap();
        assert!(
            GhostSegmentWriter::resume_or_reconstruct(tmp.path(), "mycmax", test_context(2),)
                .is_err()
        );
    }

    #[test]
    fn bound_overflow_authority_reconstructs_every_drain_and_terminal_zero() {
        let tmp = tempfile::tempdir().unwrap();
        let mut writer = GhostSegmentWriter::create(tmp.path(), "mycmax", test_context(4)).unwrap();
        writer.bind_overflow_admission(8, 2).unwrap();
        writer.append_payload(&records(2, 0), 2).unwrap();
        writer.append_payload(&[], 0).unwrap();
        writer.complete_with_terminal_residual(0).unwrap();
        let verified = verify_stream_root(&segment_root(tmp.path(), "mycmax", 4)).unwrap();
        let authority = &verified.manifest.overflow_conservation;
        assert!(authority.acceptance_pass());
        assert_eq!(authority.drain_count, 2);
        assert_eq!(authority.empty_drain_count, 1);
        assert_eq!(authority.drain_produced_records, 2);
        assert_eq!(authority.terminal_residual_records, Some(0));
    }

    #[test]
    fn overflow_admission_capacity_and_terminal_residual_fail_closed() {
        let tmp = tempfile::tempdir().unwrap();
        let mut writer = GhostSegmentWriter::create(tmp.path(), "mycmax", test_context(5)).unwrap();
        assert!(writer.bind_overflow_admission(1, 2).is_err());
        writer.bind_overflow_admission(1, 1).unwrap();
        assert!(writer.append_payload(&records(2, 0), 2).is_err());
        writer.append_payload(&records(1, 0), 1).unwrap();
        assert!(writer.complete().is_err());
        assert!(writer.complete_with_terminal_residual(1).is_err());
        assert!(!writer.manifest().completed);
        assert!(
            !writer
                .manifest()
                .overflow_conservation
                .terminal_residual_zero_pass
        );
    }

    #[test]
    fn overflow_authority_omission_corruption_and_old_schema_are_rejected() {
        for mutation in ["omit", "capacity", "old_schema"] {
            let tmp = tempfile::tempdir().unwrap();
            let mut writer =
                GhostSegmentWriter::create(tmp.path(), "mycmax", test_context(6)).unwrap();
            writer.bind_overflow_admission(8, 2).unwrap();
            writer.append_payload(&records(1, 0), 1).unwrap();
            writer.complete_with_terminal_residual(0).unwrap();
            let root = segment_root(tmp.path(), "mycmax", 6);
            let manifest_path = root.join("manifest.json");
            let mut value: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
            match mutation {
                "omit" => {
                    value
                        .as_object_mut()
                        .unwrap()
                        .remove("overflow_conservation");
                }
                "capacity" => {
                    value["overflow_conservation"]["ring_capacity_records"] = serde_json::json!(1);
                }
                "old_schema" => value["schema_version"] = serde_json::json!(3),
                _ => unreachable!(),
            }
            std::fs::write(&manifest_path, serde_json::to_vec(&value).unwrap()).unwrap();
            assert!(verify_stream_root(&root).is_err(), "mutation={mutation}");
        }
    }

    #[test]
    fn stale_v3_binary_segment_magic_is_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let mut writer = GhostSegmentWriter::create(tmp.path(), "mycmax", test_context(7)).unwrap();
        writer.bind_overflow_admission(8, 1).unwrap();
        writer.append_payload(&records(1, 0), 1).unwrap();
        let path = segment_root(tmp.path(), "mycmax", 7).join("segment_00000000.ghostbin");
        let mut bytes = std::fs::read(path).unwrap();
        bytes[..8].copy_from_slice(b"PGHSEG03");
        assert!(decode_header(&bytes[..GHOST_SEGMENT_HEADER_BYTES]).is_err());
    }

    #[test]
    fn checkpoint_reopen_excludes_later_segments_without_discarding_them() {
        let tmp = tempfile::tempdir().unwrap();
        let context = test_context(0);
        let mut writer = GhostSegmentWriter::create(tmp.path(), "target", context.clone()).unwrap();
        writer.append_payload(&records(1, 0), 1).unwrap();
        let boundary = writer.checkpoint(tmp.path()).unwrap();
        writer.append_payload(&records(1, 1), 1).unwrap();
        drop(writer);
        let mut restored =
            GhostSegmentWriter::reopen_checkpoint(tmp.path(), &boundary, &context).unwrap();
        assert_eq!(restored.manifest.segments.len(), 1);
        assert!(restored.root.join(".continuation_tails").is_dir());
        restored.append_payload(&records(1, 1), 1).unwrap();
        assert_eq!(
            verify_stream_root(&restored.root)
                .unwrap()
                .manifest
                .segments
                .len(),
            2
        );
    }
}
