//! F2 async I/O evidence-plane schema.
//!
//! This module is data-only. It defines JSON sidecar records for ZSTR/Ghost
//! ring health, write commits, artifact completeness, and cuda-error-drain
//! visibility. It does not perform runtime emission or touch CUDA graph
//! topology.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Current F2 evidence-plane schema version.
pub const F2_EVIDENCE_SCHEMA_VERSION: u32 = 1;

/// TIER 8 cuda-error-drain caveat reference carried in completeness reports.
pub const TIER8_CUDA_ERROR_DRAIN_CAVEAT_DOC: &str = "docs/TIER8_GRAPH_TOPOLOGY.md:99-107";

/// Evidence channel represented in F2 sidecars.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum F2ChannelKind {
    /// ZSTR Channel A positions/forces stream.
    Zstr,
    /// Ghost tile Channel B evidence stream.
    Ghost,
}

/// Consumer or commit exit status captured for audit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ZstrExitStatus {
    /// Consumer was not spawned or the status is not yet known.
    NotStarted,
    /// Stop signal was observed and the consumer exited cleanly.
    NormalStop,
    /// Backwards-compatible clean status for non-ZSTR commit records.
    Clean,
    /// Consumer aborted because the G21 alignment gate failed.
    AbortedG21Alignment,
    /// Channel open failed before any write could be committed.
    FileOpenFailure,
    /// io_uring initialization failed before any write could be committed.
    IoUringInitFailure,
    /// Consumer was intentionally skipped by caller policy.
    Skipped,
    /// Consumer or commit completed without a final fsync.
    NoFsync,
    /// Final fsync completed successfully.
    FsyncOk,
    /// Final fsync returned an error.
    FsyncError,
    /// Consumer joined with a typed error.
    Error,
    /// Consumer thread panicked.
    ConsumerPanic,
}

/// ZSTR channel accounting captured after the consumer joins.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZstrChannelAccounting {
    /// Number of ring slots.
    pub n_slots: u64,
    /// Bytes per ZSTR frame slot, including sector padding.
    pub frame_size_bytes: u64,
    /// Atom count represented by each frame.
    pub n_atoms: u64,
    /// Offset from slot base to the position payload.
    pub position_payload_offset_bytes: u64,
    /// Offset from slot base to the force payload.
    pub force_payload_offset_bytes: u64,
    /// Frames written by the consumer.
    pub frames_written: u64,
    /// Frames dropped by fence timeout or validation failure.
    pub frames_dropped: u64,
    /// Frames lost because the submission queue or ring overflowed.
    pub frames_overflow: u64,
    /// Bytes written by the consumer.
    pub bytes_written: u64,
    /// Channel-A fsync result: 0 means success, nonzero is errno, None means unknown.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fsync_status_a: Option<i32>,
    /// Channel-B fsync result: -1 means disabled, 0 means success, nonzero is errno.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fsync_status_b: Option<i32>,
}

impl ZstrChannelAccounting {
    /// Construct ZSTR accounting with stable zero defaults.
    pub fn new(n_slots: u64, frame_size_bytes: u64, n_atoms: u64) -> Self {
        Self {
            n_slots,
            frame_size_bytes,
            n_atoms,
            position_payload_offset_bytes: 4096,
            force_payload_offset_bytes: 0,
            frames_written: 0,
            frames_dropped: 0,
            frames_overflow: 0,
            bytes_written: 0,
            fsync_status_a: None,
            fsync_status_b: None,
        }
    }
}

/// Ghost evidence-channel accounting captured after the consumer joins.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GhostChannelAccounting {
    /// Bytes per ghost record.
    pub record_bytes: u64,
    /// Counter sector size in bytes.
    pub counter_sector_bytes: u64,
    /// Maximum record capacity.
    pub max_records: u64,
    /// Total mapped channel bytes.
    pub total_bytes: u64,
    /// Records observed by the host consumer.
    pub records_observed: u64,
    /// Records actually written, when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub records_written: Option<u64>,
    /// Bytes actually written, when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes_written: Option<u64>,
    /// Whether the GPU-written counter saturated at `max_records`.
    pub counter_saturated: bool,
    /// Channel-B output mode, such as `io_uring_odirect` or `legacy_pwrite`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channel_b_path: Option<String>,
}

impl GhostChannelAccounting {
    /// Construct Ghost accounting with stable zero defaults.
    pub fn new(record_bytes: u64, counter_sector_bytes: u64, max_records: u64) -> Self {
        Self {
            record_bytes,
            counter_sector_bytes,
            max_records,
            total_bytes: counter_sector_bytes + record_bytes.saturating_mul(max_records),
            records_observed: 0,
            records_written: None,
            bytes_written: None,
            counter_saturated: false,
            channel_b_path: None,
        }
    }
}

/// Ring health and alignment status for one F2 stream/channel.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct F2RingStatus {
    /// Schema version for this record.
    pub schema_version: u32,
    /// Stable sidecar ring identifier.
    pub ring_id: String,
    /// PRISM stream identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_id: Option<u32>,
    /// Evidence channel represented by this record.
    pub channel: F2ChannelKind,
    /// Optional artifact path associated with this ring.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_path: Option<PathBuf>,
    /// Required alignment in bytes for the usable pointer.
    pub required_alignment_bytes: u64,
    /// Whether the working pointer satisfied the required alignment.
    pub alignment_ok: bool,
    /// Raw pinned allocation address, when captured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_pointer_addr: Option<u64>,
    /// Aligned usable pointer address, when captured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aligned_pointer_addr: Option<u64>,
    /// Raw pinned allocation bytes, including alignment padding.
    pub raw_alloc_bytes: u64,
    /// Usable bytes exposed as records.
    pub usable_bytes: u64,
    /// Offset from raw pinned pointer to usable aligned pointer.
    pub alignment_offset: u64,
    /// Record stride in bytes.
    pub record_stride_bytes: u64,
    /// Number of frames or records the ring can hold.
    pub capacity_frames: u64,
    /// Producer sequence at snapshot time.
    pub head_seq: u64,
    /// Consumer sequence at snapshot time.
    pub tail_seq: u64,
    /// Frames produced by the device side.
    pub frames_produced: u64,
    /// Frames consumed by the host side.
    pub frames_consumed: u64,
    /// Frames dropped by the host side, when the channel tracks drops.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frames_dropped: Option<u64>,
    /// Frames lost to overflow.
    pub frames_overflow: u64,
    /// Backpressure events, when the channel tracks them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backpressure_events: Option<u64>,
    /// Whether a consumer thread was spawned for this ring.
    pub consumer_spawned: bool,
    /// Consumer exit status.
    pub consumer_exit_status: ZstrExitStatus,
    /// Human-readable status detail, when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub consumer_exit_reason: Option<String>,
    /// Whether the last observed completion fence was ready.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fence_was_ready: Option<bool>,
    /// Whether pinned memory allocation was attested.
    pub pinned_attest: bool,
    /// Human-readable pinned-memory attestation detail.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pinned_attest_detail: Option<String>,
    /// Open flags such as O_DIRECT/O_CREAT/O_TRUNC.
    pub open_flags: Vec<String>,
    /// io_uring submission queue depth, when applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub iouring_sq_depth: Option<u32>,
    /// CUDA error drain count associated with this stream/channel.
    pub cuda_error_drain_count: u64,
    /// Whether a CUDA error drain preceded the observed launch or commit.
    pub cuda_error_drain_preceded: bool,
    /// CUDA error drain call sites associated with this stream/channel.
    pub cuda_error_drain_sites: Vec<String>,
    /// Last cuda-error-drain call site, when captured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cuda_error_drain_call_site: Option<String>,
    /// Whether the cuda-error-drain context matched this stream.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cuda_error_drain_matches_stream: Option<bool>,
    /// Director stream id attached to the cuda-error-drain context.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub director_stream_id: Option<u32>,
    /// CUDA result code associated with the CUDA error drain.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cuda_error_drain_rc: Option<i32>,
    /// CUDA result name associated with the CUDA error drain.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cuda_error_drain_name: Option<String>,
    /// CUDA result text associated with the CUDA error drain.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cuda_error_drain_string: Option<String>,
    /// ZSTR-specific accounting when this is a ZSTR channel.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub zstr: Option<ZstrChannelAccounting>,
    /// Ghost-specific accounting when this is a Ghost channel.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ghost: Option<GhostChannelAccounting>,
}

impl F2RingStatus {
    /// Construct a minimal ring-status record with stable defaults.
    pub fn new(stream_id: u32, channel: F2ChannelKind, alignment_ok: bool) -> Self {
        Self {
            schema_version: F2_EVIDENCE_SCHEMA_VERSION,
            ring_id: format!("stream-{stream_id}-{}", channel.as_str()),
            stream_id: Some(stream_id),
            channel,
            artifact_path: None,
            required_alignment_bytes: 4096,
            alignment_ok,
            raw_pointer_addr: None,
            aligned_pointer_addr: None,
            raw_alloc_bytes: 0,
            usable_bytes: 0,
            alignment_offset: 0,
            record_stride_bytes: 0,
            capacity_frames: 0,
            head_seq: 0,
            tail_seq: 0,
            frames_produced: 0,
            frames_consumed: 0,
            frames_dropped: None,
            frames_overflow: 0,
            backpressure_events: None,
            consumer_spawned: false,
            consumer_exit_status: ZstrExitStatus::NotStarted,
            consumer_exit_reason: None,
            fence_was_ready: None,
            pinned_attest: false,
            pinned_attest_detail: None,
            open_flags: Vec::new(),
            iouring_sq_depth: None,
            cuda_error_drain_count: 0,
            cuda_error_drain_preceded: false,
            cuda_error_drain_sites: Vec::new(),
            cuda_error_drain_call_site: None,
            cuda_error_drain_matches_stream: None,
            director_stream_id: None,
            cuda_error_drain_rc: None,
            cuda_error_drain_name: None,
            cuda_error_drain_string: None,
            zstr: None,
            ghost: None,
        }
    }
}

impl F2ChannelKind {
    /// Stable lowercase channel label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Zstr => "zstr",
            Self::Ghost => "ghost",
        }
    }
}

/// Durable write-commit record for one emitted F2 artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct F2WriteCommit {
    /// Schema version for this record.
    pub schema_version: u32,
    /// PRISM stream identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_id: Option<u32>,
    /// Evidence channel represented by this commit.
    pub channel: F2ChannelKind,
    /// Ring identifier associated with this commit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ring_id: Option<String>,
    /// Artifact path committed by the consumer.
    pub artifact_path: PathBuf,
    /// Bytes written to the artifact.
    pub bytes_written: u64,
    /// Frames written to the artifact.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frames_written: Option<u64>,
    /// First sequence number committed into the artifact.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_seq: Option<u64>,
    /// Last sequence number committed into the artifact.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_seq: Option<u64>,
    /// Open flags such as O_DIRECT/O_CREAT/O_TRUNC.
    pub open_flags: Vec<String>,
    /// Consumer close/join status.
    pub close_status: ZstrExitStatus,
    /// Human-readable close status detail, when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub close_status_reason: Option<String>,
    /// Whether truncation/close semantics were enforced.
    pub truncation_guarantee: bool,
    /// Artifact hash captured after close.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hash: Option<String>,
    /// Channel-A fsync result: 0 means success, nonzero is errno, None means unknown.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fsync_status_a: Option<i32>,
    /// Channel-B fsync result: -1 means disabled, 0 means success, nonzero is errno.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fsync_status_b: Option<i32>,
    /// CUDA error drain count visible before this commit.
    pub cuda_error_drain_count: u64,
    /// Whether a CUDA error drain preceded this commit.
    pub cuda_error_drain_preceded: bool,
    /// CUDA error drain call sites visible before this commit.
    pub cuda_error_drain_sites: Vec<String>,
}

impl F2WriteCommit {
    /// Construct a write-commit record with stable defaults.
    pub fn new(
        stream_id: u32,
        channel: F2ChannelKind,
        artifact_path: impl Into<PathBuf>,
        bytes_written: u64,
        frames_written: u64,
        hash: impl Into<String>,
    ) -> Self {
        Self {
            schema_version: F2_EVIDENCE_SCHEMA_VERSION,
            stream_id: Some(stream_id),
            channel,
            ring_id: Some(format!("stream-{stream_id}-{}", channel.as_str())),
            artifact_path: artifact_path.into(),
            bytes_written,
            frames_written: Some(frames_written),
            first_seq: None,
            last_seq: None,
            open_flags: Vec::new(),
            close_status: ZstrExitStatus::NotStarted,
            close_status_reason: None,
            truncation_guarantee: false,
            hash: Some(hash.into()),
            fsync_status_a: None,
            fsync_status_b: None,
            cuda_error_drain_count: 0,
            cuda_error_drain_preceded: false,
            cuda_error_drain_sites: Vec::new(),
        }
    }
}

/// Expected or observed F2 artifact entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct F2ArtifactEntry {
    /// Artifact path.
    pub path: PathBuf,
    /// Optional channel classification.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channel: Option<F2ChannelKind>,
    /// Whether the artifact exists.
    pub exists: bool,
    /// Artifact size in bytes, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
    /// Record count, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub record_count: Option<u64>,
    /// Expected bytes per record, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_record_bytes: Option<u64>,
}

impl F2ArtifactEntry {
    /// Return whether the artifact exists and has nonzero size.
    pub fn is_complete(&self) -> bool {
        self.exists && self.size_bytes.unwrap_or(0) > 0
    }
}

/// Partial-artifact explanation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct F2PartialArtifact {
    /// Artifact path.
    pub path: PathBuf,
    /// Human-readable reason why the artifact is partial.
    pub reason: String,
}

/// Overall F2 completeness state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum F2CompletenessStatus {
    /// Every expected artifact exists and passes fences.
    Complete,
    /// One or more expected artifacts or fences failed.
    Partial,
    /// Backwards-compatible name for a partial result.
    Incomplete,
    /// Completeness has not been evaluated.
    Unknown,
}

/// Whole-run F2 artifact completeness report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct F2ArtifactCompleteness {
    /// Schema version for this record.
    pub schema_version: u32,
    /// Run identifier shared with the transform DAG and summary artifacts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    /// Artifacts expected by the F2 evidence plane.
    pub expected_artifacts: Vec<F2ArtifactEntry>,
    /// Paths that were emitted.
    pub emitted_artifacts: Vec<PathBuf>,
    /// Missing expected artifact paths.
    pub missing: Vec<PathBuf>,
    /// Partial artifacts and reasons.
    pub partial: Vec<F2PartialArtifact>,
    /// Whether the final fence passed.
    pub fence_pass: bool,
    /// Whether all alignment checks passed.
    pub alignment_pass: bool,
    /// Whether drain checks passed.
    pub drain_pass: bool,
    /// Whole-run TIER8 cuda-error-drain count.
    pub tier8_cuda_error_drain_count: u64,
    /// Whether the non-fatal TIER8 cuda-error-drain caveat was present.
    pub tier8_cuda_error_drain_caveat: bool,
    /// Whole-run cuda-error-drain call sites.
    pub cuda_error_drain_sites: Vec<String>,
    /// Human-readable completeness reasons and caveats.
    pub reasons: Vec<String>,
    /// Overall completeness status.
    pub overall_status: F2CompletenessStatus,
}

impl F2ArtifactCompleteness {
    /// Build a completeness report from expected artifact observations.
    pub fn from_expected(
        expected_artifacts: Vec<F2ArtifactEntry>,
        fence_pass: bool,
        alignment_pass: bool,
        drain_pass: bool,
        tier8_cuda_error_drain_count: u64,
    ) -> Self {
        let mut emitted_artifacts = Vec::new();
        let mut missing = Vec::new();
        let mut partial = Vec::new();
        let mut reasons = Vec::new();

        for artifact in &expected_artifacts {
            if artifact.exists {
                emitted_artifacts.push(artifact.path.clone());
                if !artifact.is_complete() {
                    partial.push(F2PartialArtifact {
                        path: artifact.path.clone(),
                        reason: "artifact exists but is empty or size is unknown".to_string(),
                    });
                }
            } else {
                missing.push(artifact.path.clone());
            }
        }

        if !missing.is_empty() {
            reasons.push(format!("missing_artifacts={}", missing.len()));
        }
        if !partial.is_empty() {
            reasons.push(format!("partial_artifacts={}", partial.len()));
        }
        if !fence_pass {
            reasons.push("fence_pass=false".to_string());
        }
        if !alignment_pass {
            reasons.push("alignment_pass=false".to_string());
        }
        if !drain_pass {
            reasons.push("drain_pass=false".to_string());
        }

        let tier8_cuda_error_drain_caveat = tier8_cuda_error_drain_count > 0;
        if tier8_cuda_error_drain_caveat {
            reasons.push(format!(
                "tier8_cuda_error_drain_count={tier8_cuda_error_drain_count} - see {TIER8_CUDA_ERROR_DRAIN_CAVEAT_DOC}"
            ));
        }

        let overall_status = if missing.is_empty()
            && partial.is_empty()
            && fence_pass
            && alignment_pass
            && drain_pass
        {
            F2CompletenessStatus::Complete
        } else {
            F2CompletenessStatus::Partial
        };

        Self {
            schema_version: F2_EVIDENCE_SCHEMA_VERSION,
            run_id: None,
            expected_artifacts,
            emitted_artifacts,
            missing,
            partial,
            fence_pass,
            alignment_pass,
            drain_pass,
            tier8_cuda_error_drain_count,
            tier8_cuda_error_drain_caveat,
            cuda_error_drain_sites: Vec::new(),
            reasons,
            overall_status,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact(
        path: &str,
        channel: Option<F2ChannelKind>,
        exists: bool,
        size_bytes: Option<u64>,
    ) -> F2ArtifactEntry {
        F2ArtifactEntry {
            path: PathBuf::from(path),
            channel,
            exists,
            size_bytes,
            record_count: None,
            expected_record_bytes: None,
        }
    }

    #[test]
    fn ring_status_serializes_alignment_and_cuda_error_drain_fields() {
        let mut status = F2RingStatus::new(7, F2ChannelKind::Zstr, true);
        status.artifact_path = Some(PathBuf::from("run/f2_ring_status.json"));
        status.raw_alloc_bytes = 8191;
        status.usable_bytes = 4096;
        status.alignment_offset = 128;
        status.record_stride_bytes = 4096;
        status.capacity_frames = 64;
        status.frames_produced = 12;
        status.frames_consumed = 12;
        status.frames_dropped = Some(0);
        status.consumer_spawned = true;
        status.consumer_exit_status = ZstrExitStatus::NormalStop;
        status.pinned_attest = true;
        status.pinned_attest_detail = Some("cuda_host_alloc_aligned_4096".to_string());
        status.open_flags = vec!["O_DIRECT".to_string(), "O_TRUNC".to_string()];
        status.iouring_sq_depth = Some(32);
        status.cuda_error_drain_count = 2;
        status.cuda_error_drain_preceded = true;
        status.cuda_error_drain_sites = vec!["post_t7_sync".to_string()];
        status.zstr = Some(ZstrChannelAccounting {
            n_slots: 64,
            frame_size_bytes: 4096,
            n_atoms: 4682,
            position_payload_offset_bytes: 4096,
            force_payload_offset_bytes: 60_304,
            frames_written: 12,
            frames_dropped: 0,
            frames_overflow: 0,
            bytes_written: 49152,
            fsync_status_a: Some(0),
            fsync_status_b: Some(-1),
        });

        let json = serde_json::to_string(&status).expect("serialize f2 ring status");
        assert!(json.contains("\"alignment_ok\":true"));
        assert!(json.contains("\"alignment_offset\":128"));
        assert!(json.contains("\"cuda_error_drain_count\":2"));
        assert!(json.contains("\"cuda_error_drain_preceded\":true"));
        assert!(json.contains("\"post_t7_sync\""));
        assert!(json.contains("\"iouring_sq_depth\":32"));

        let roundtrip: F2RingStatus =
            serde_json::from_str(&json).expect("roundtrip f2 ring status");
        assert_eq!(roundtrip, status);
    }

    #[test]
    fn write_commit_serializes_path_and_ghost_accounting() {
        let mut commit = F2WriteCommit::new(
            3,
            F2ChannelKind::Ghost,
            "run/f2_write_commit_log.json",
            16384,
            4,
            "sha256:abc",
        );
        commit.close_status = ZstrExitStatus::FsyncOk;
        commit.truncation_guarantee = true;
        commit.open_flags = vec!["O_DIRECT".to_string(), "O_DSYNC".to_string()];
        commit.fsync_status_b = Some(0);
        commit.cuda_error_drain_preceded = true;
        commit.cuda_error_drain_sites = vec!["ctx.check_err".to_string()];

        let json = serde_json::to_string(&commit).expect("serialize f2 write commit");
        assert!(json.contains("\"channel\":\"ghost\""));
        assert!(json.contains("f2_write_commit_log.json"));
        assert!(json.contains("\"close_status\":\"fsync_ok\""));
        assert!(json.contains("\"truncation_guarantee\":true"));

        let roundtrip: F2WriteCommit =
            serde_json::from_str(&json).expect("roundtrip f2 write commit");
        assert_eq!(roundtrip, commit);
    }

    #[test]
    fn completeness_reports_missing_partial_and_cuda_error_drain_counts() {
        let report = F2ArtifactCompleteness::from_expected(
            vec![
                F2ArtifactEntry {
                    path: PathBuf::from("run/zstr.bin"),
                    channel: Some(F2ChannelKind::Zstr),
                    exists: true,
                    size_bytes: Some(8192),
                    record_count: Some(2),
                    expected_record_bytes: Some(4096),
                },
                F2ArtifactEntry {
                    path: PathBuf::from("run/ghost.bin"),
                    channel: Some(F2ChannelKind::Ghost),
                    exists: true,
                    size_bytes: Some(0),
                    record_count: Some(0),
                    expected_record_bytes: Some(4096),
                },
                F2ArtifactEntry {
                    path: PathBuf::from("run/missing.bin"),
                    channel: None,
                    exists: false,
                    size_bytes: None,
                    record_count: None,
                    expected_record_bytes: None,
                },
            ],
            true,
            true,
            true,
            2,
        );

        assert_eq!(report.emitted_artifacts.len(), 2);
        assert_eq!(report.missing, vec![PathBuf::from("run/missing.bin")]);
        assert_eq!(report.partial.len(), 1);
        assert_eq!(report.tier8_cuda_error_drain_count, 2);
        assert!(report.tier8_cuda_error_drain_caveat);
        assert!(report
            .reasons
            .iter()
            .any(|reason| reason.contains(TIER8_CUDA_ERROR_DRAIN_CAVEAT_DOC)));
        assert_eq!(report.overall_status, F2CompletenessStatus::Partial);

        let json = serde_json::to_string(&report).expect("serialize f2 completeness");
        assert!(json.contains("\"overall_status\":\"partial\""));
        assert!(json.contains("\"tier8_cuda_error_drain_count\":2"));
        assert!(json.contains("\"tier8_cuda_error_drain_caveat\":true"));
    }

    #[test]
    fn completeness_roundtrips_complete_verdict() {
        let report = F2ArtifactCompleteness::from_expected(
            vec![
                artifact("run/zstr.bin", Some(F2ChannelKind::Zstr), true, Some(8192)),
                artifact(
                    "run/ghost.bin",
                    Some(F2ChannelKind::Ghost),
                    true,
                    Some(4096),
                ),
            ],
            true,
            true,
            true,
            0,
        );

        assert_eq!(report.schema_version, F2_EVIDENCE_SCHEMA_VERSION);
        assert_eq!(report.overall_status, F2CompletenessStatus::Complete);
        assert_eq!(
            report.emitted_artifacts,
            vec![
                PathBuf::from("run/zstr.bin"),
                PathBuf::from("run/ghost.bin")
            ]
        );
        assert!(report.missing.is_empty());
        assert!(report.partial.is_empty());
        assert!(report.reasons.is_empty());
        assert!(!report.tier8_cuda_error_drain_caveat);

        let json = serde_json::to_string(&report).expect("serialize complete f2 completeness");
        assert!(json.contains("\"overall_status\":\"complete\""));
        let roundtrip: F2ArtifactCompleteness =
            serde_json::from_str(&json).expect("roundtrip complete f2 completeness");
        assert_eq!(roundtrip, report);
    }

    #[test]
    fn completeness_distinguishes_missing_from_partial_artifacts() {
        let report = F2ArtifactCompleteness::from_expected(
            vec![
                artifact(
                    "run/zstr-present.bin",
                    Some(F2ChannelKind::Zstr),
                    true,
                    Some(4096),
                ),
                artifact(
                    "run/zstr-empty.bin",
                    Some(F2ChannelKind::Zstr),
                    true,
                    Some(0),
                ),
                artifact(
                    "run/ghost-unknown.bin",
                    Some(F2ChannelKind::Ghost),
                    true,
                    None,
                ),
                artifact("run/missing.bin", Some(F2ChannelKind::Ghost), false, None),
            ],
            true,
            true,
            true,
            0,
        );

        assert_eq!(report.overall_status, F2CompletenessStatus::Partial);
        assert_eq!(report.missing, vec![PathBuf::from("run/missing.bin")]);
        assert_eq!(
            report.emitted_artifacts,
            vec![
                PathBuf::from("run/zstr-present.bin"),
                PathBuf::from("run/zstr-empty.bin"),
                PathBuf::from("run/ghost-unknown.bin")
            ]
        );
        assert_eq!(
            report
                .partial
                .iter()
                .map(|partial| partial.path.clone())
                .collect::<Vec<_>>(),
            vec![
                PathBuf::from("run/zstr-empty.bin"),
                PathBuf::from("run/ghost-unknown.bin")
            ]
        );
        assert!(report
            .partial
            .iter()
            .all(|partial| partial.reason == "artifact exists but is empty or size is unknown"));
        assert!(report
            .reasons
            .iter()
            .any(|reason| reason == "missing_artifacts=1"));
        assert!(report
            .reasons
            .iter()
            .any(|reason| reason == "partial_artifacts=2"));
    }
}
