//! Durable, hash-linked evidence journaling for Split-T7 calibration chunks.
//!
//! The journal deliberately records aggregate spike counts and, when a chunk
//! produces one, the single normalized six-component `C_l` vector. It does not
//! persist the underlying spike cloud and therefore cannot be used to claim raw
//! spike-event reconstructability.

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// JSON schema kind emitted for every T7 chunk record.
pub const T7_CHUNK_RECORD_SCHEMA_KIND: &str = "prismos.t7_calibration_chunk.v1";
/// JSON schema kind emitted for a terminal or partial journal closeout.
pub const T7_JOURNAL_SUMMARY_SCHEMA_KIND: &str = "prismos.t7_calibration_summary.v1";
/// Current T7 journal schema version.
pub const T7_JOURNAL_SCHEMA_VERSION: u32 = 1;
/// The largest supported durability batch, bounding evidence buffered between syncs.
pub const MAX_RECORDS_PER_DURABLE_FLUSH: usize = 256;
/// Maximum encoded JSONL size accepted for one chunk record during verification.
pub const MAX_T7_RECORD_BYTES: usize = 32 * 1024;
/// Maximum persisted summary size accepted during no-follow readback.
pub const MAX_T7_SUMMARY_BYTES: usize = 64 * 1024;
/// Maximum encoded length of an optional GPU count-read diagnostic code.
pub const MAX_T7_GPU_COUNT_DIAGNOSTIC_CODE_BYTES: usize = 128;
/// Exact maximum number of GPU spike events admitted for one T7 chunk.
pub const T7_SPIKE_EVENT_CAP: u64 = 8_192;

const JOURNAL_BUFFER_BYTES: usize = 64 * 1024;
const MAX_RUN_ID_BYTES: usize = 256;
const MAX_PHASE_BYTES: usize = 128;
const MAX_PARTIAL_REASON_BYTES: usize = 1024;
static TEMP_FILE_NONCE: AtomicU64 = AtomicU64::new(0);

/// Immutable execution identity repeated in records and the journal summary.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct T7JournalIdentity {
    /// Operator/run identity owning the calibration evidence.
    pub run_id: String,
    /// Zero-based CUDA stream index for this per-stream journal.
    pub stream_index: u32,
    /// SHA-256 of the exact executing engine ELF, without a `sha256:` prefix.
    pub engine_sha256: String,
    /// SHA-256 of the exact topology input, without a `sha256:` prefix.
    pub topology_sha256: String,
}

impl T7JournalIdentity {
    fn validate(&self) -> Result<()> {
        require_bounded_text("run_id", &self.run_id, MAX_RUN_ID_BYTES)?;
        require_sha256("engine_sha256", &self.engine_sha256)?;
        require_sha256("topology_sha256", &self.topology_sha256)?;
        Ok(())
    }
}

/// Durability policy for a journal writer.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct T7JournalConfig {
    /// Maximum accepted records between `flush` plus `sync_data` operations.
    pub records_per_durable_flush: usize,
}

impl Default for T7JournalConfig {
    fn default() -> Self {
        Self {
            records_per_durable_flush: 8,
        }
    }
}

impl T7JournalConfig {
    fn validate(self) -> Result<()> {
        if self.records_per_durable_flush == 0
            || self.records_per_durable_flush > MAX_RECORDS_PER_DURABLE_FLUSH
        {
            bail!("records_per_durable_flush must be in 1..={MAX_RECORDS_PER_DURABLE_FLUSH}");
        }
        Ok(())
    }
}

/// Result of transferring the capped T7 spike-event cloud from the GPU.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum T7CopyStatus {
    /// No host copy was attempted for this chunk.
    NotAttempted,
    /// The requested capped event cloud was copied successfully.
    Complete,
    /// The host copy failed; no sample may be emitted from this chunk.
    Failed,
}

/// Result of reading the per-chunk GPU spike counter before event download.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum T7GpuCountReadStatus {
    /// The GPU counter was read successfully; zero is a valid observed count.
    Complete,
    /// The GPU counter read was attempted and failed.
    Failed,
    /// The counter or event-buffer authority was unavailable for this chunk.
    Unavailable,
}

/// Result of converting a copied event cloud into one normalized `C_l` vector.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum T7NormalizationStatus {
    /// Normalization was not attempted.
    NotAttempted,
    /// Exactly one finite normalized `C_l` vector was produced.
    Normalized,
    /// The copied cloud had no usable normalization mass.
    NoUsableSignal,
    /// A non-finite intermediate or result prevented sample emission.
    RejectedNonFinite,
    /// Normalization failed for another explicitly handled reason.
    Failed,
}

/// One caller-supplied T7 chunk observation before hash-chain fields are added.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct T7ChunkObservation {
    /// Monotonic chunk index within the owning stream.
    pub chunk_index: u64,
    /// Inclusive first MD step represented by the chunk.
    pub step_start: u64,
    /// Inclusive final MD step represented by the chunk.
    pub step_end: u64,
    /// Runtime phase in which the GPU observation was made.
    pub phase: String,
    /// Spike count reported by the GPU counter before applying the host cap.
    pub gpu_spike_count_observed: u64,
    /// Whether the GPU spike counter was read, failed, or unavailable.
    pub gpu_count_read_status: T7GpuCountReadStatus,
    /// Optional bounded machine-readable diagnostic for a failed/unavailable count read.
    pub gpu_count_diagnostic_code: Option<String>,
    /// Maximum number of spike events eligible for host download.
    pub spike_event_cap: u64,
    /// Number of capped spike events actually downloaded to the host.
    pub spike_events_downloaded: u64,
    /// Number of downloaded spike events consumed by the `C_l` calculation.
    pub spike_events_used: u64,
    /// Status of the GPU-to-host event-cloud copy.
    pub copy_status: T7CopyStatus,
    /// Status of the `C_l` normalization calculation.
    pub normalization_status: T7NormalizationStatus,
    /// Whether this chunk appended exactly one normalized `C_l` sample.
    pub sample_emitted: bool,
    /// The one normalized six-component `C_l` vector, only when emitted.
    pub cl_normalized: Option<[f32; 6]>,
}

impl T7ChunkObservation {
    fn validate(&self) -> Result<()> {
        if self.step_end < self.step_start {
            bail!("step_end precedes step_start");
        }
        require_bounded_text("phase", &self.phase, MAX_PHASE_BYTES)?;
        if self.spike_event_cap != T7_SPIKE_EVENT_CAP {
            bail!("spike_event_cap must equal the governed T7 cap of {T7_SPIKE_EVENT_CAP}");
        }
        if let Some(code) = self.gpu_count_diagnostic_code.as_deref() {
            require_diagnostic_code("gpu_count_diagnostic_code", code)?;
        }
        match self.gpu_count_read_status {
            T7GpuCountReadStatus::Complete => {
                if self.gpu_count_diagnostic_code.is_some() {
                    bail!("a completed GPU count read cannot carry a diagnostic code");
                }
            }
            T7GpuCountReadStatus::Failed | T7GpuCountReadStatus::Unavailable => {
                if self.gpu_spike_count_observed != 0
                    || self.spike_events_downloaded != 0
                    || self.spike_events_used != 0
                    || self.copy_status != T7CopyStatus::NotAttempted
                    || self.normalization_status != T7NormalizationStatus::NotAttempted
                    || self.sample_emitted
                    || self.cl_normalized.is_some()
                {
                    bail!(
                        "a failed or unavailable GPU count read cannot report downstream capture evidence"
                    );
                }
            }
        }
        if self.spike_events_downloaded > self.spike_event_cap {
            bail!("spike_events_downloaded exceeds spike_event_cap");
        }
        if self.spike_events_downloaded > self.gpu_spike_count_observed {
            bail!("spike_events_downloaded exceeds gpu_spike_count_observed");
        }
        if self.spike_events_used > self.spike_events_downloaded {
            bail!("spike_events_used exceeds spike_events_downloaded");
        }
        if self.copy_status != T7CopyStatus::Complete
            && (self.spike_events_downloaded != 0 || self.spike_events_used != 0)
        {
            bail!("an incomplete copy cannot report downloaded or used spike events");
        }

        match (
            self.sample_emitted,
            self.normalization_status,
            self.cl_normalized,
        ) {
            (true, T7NormalizationStatus::Normalized, Some(vector)) => {
                if self.copy_status != T7CopyStatus::Complete {
                    bail!("sample emission requires a completed GPU-to-host copy");
                }
                if self.spike_events_used == 0 {
                    bail!("sample emission requires at least one used spike event");
                }
                if vector.iter().any(|value| !value.is_finite()) {
                    bail!("cl_normalized contains a non-finite component");
                }
            }
            (true, _, _) => {
                bail!("sample_emitted requires NORMALIZED status and cl_normalized");
            }
            (false, T7NormalizationStatus::Normalized, _) => {
                bail!("NORMALIZED status requires sample_emitted=true");
            }
            (false, _, Some(_)) => {
                bail!("cl_normalized must be absent when sample_emitted=false");
            }
            (false, _, None) => {}
        }
        Ok(())
    }
}

/// Persisted T7 chunk evidence with an explicit SHA-256 predecessor link.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct T7ChunkRecord {
    /// Record schema identifier.
    pub schema_kind: String,
    /// Record schema version.
    pub schema_version: u32,
    /// Immutable run identity.
    pub run_id: String,
    /// Zero-based stream identity.
    pub stream_index: u32,
    /// SHA-256 of the exact engine ELF.
    pub engine_sha256: String,
    /// SHA-256 of the exact topology input.
    pub topology_sha256: String,
    /// Zero-based position of this record in the journal.
    pub record_index: u64,
    /// Monotonic chunk index within the stream.
    pub chunk_index: u64,
    /// Inclusive first MD step represented by the chunk.
    pub step_start: u64,
    /// Inclusive final MD step represented by the chunk.
    pub step_end: u64,
    /// Runtime phase in which the observation was made.
    pub phase: String,
    /// Spike count read from the GPU before applying the event cap.
    pub gpu_spike_count_observed: u64,
    /// Whether the GPU spike counter was read, failed, or unavailable.
    pub gpu_count_read_status: T7GpuCountReadStatus,
    /// Optional bounded machine-readable diagnostic for the GPU count read.
    pub gpu_count_diagnostic_code: Option<String>,
    /// Maximum number of spike events eligible for host download.
    pub spike_event_cap: u64,
    /// Number of capped spike events downloaded to the host.
    pub spike_events_downloaded: u64,
    /// Number of downloaded spike events used for the `C_l` calculation.
    pub spike_events_used: u64,
    /// Status of the GPU-to-host copy.
    pub copy_status: T7CopyStatus,
    /// Status of normalization.
    pub normalization_status: T7NormalizationStatus,
    /// Whether this record represents exactly one emitted `C_l` sample.
    pub sample_emitted: bool,
    /// Optional normalized six-component `C_l` vector.
    pub cl_normalized: Option<[f32; 6]>,
    /// Explicit statement that raw spike-event coordinates are not in this journal.
    pub raw_spike_cloud_persisted: bool,
    /// SHA-256 of the previous record, or `null` for the first record.
    pub previous_record_sha256: Option<String>,
    /// SHA-256 over this record with this field omitted.
    pub record_sha256: String,
}

impl T7ChunkRecord {
    fn from_observation(
        identity: &T7JournalIdentity,
        record_index: u64,
        observation: T7ChunkObservation,
        previous_record_sha256: Option<String>,
    ) -> Result<Self> {
        let mut record = Self {
            schema_kind: T7_CHUNK_RECORD_SCHEMA_KIND.to_string(),
            schema_version: T7_JOURNAL_SCHEMA_VERSION,
            run_id: identity.run_id.clone(),
            stream_index: identity.stream_index,
            engine_sha256: identity.engine_sha256.clone(),
            topology_sha256: identity.topology_sha256.clone(),
            record_index,
            chunk_index: observation.chunk_index,
            step_start: observation.step_start,
            step_end: observation.step_end,
            phase: observation.phase,
            gpu_spike_count_observed: observation.gpu_spike_count_observed,
            gpu_count_read_status: observation.gpu_count_read_status,
            gpu_count_diagnostic_code: observation.gpu_count_diagnostic_code,
            spike_event_cap: observation.spike_event_cap,
            spike_events_downloaded: observation.spike_events_downloaded,
            spike_events_used: observation.spike_events_used,
            copy_status: observation.copy_status,
            normalization_status: observation.normalization_status,
            sample_emitted: observation.sample_emitted,
            cl_normalized: observation.cl_normalized,
            raw_spike_cloud_persisted: false,
            previous_record_sha256,
            record_sha256: String::new(),
        };
        record.record_sha256 = record.calculated_sha256()?;
        Ok(record)
    }

    /// Recalculate the record digest with `record_sha256` excluded.
    pub fn calculated_sha256(&self) -> Result<String> {
        let mut value = serde_json::to_value(self).context("serialize T7 record for hashing")?;
        value
            .as_object_mut()
            .ok_or_else(|| anyhow!("T7 record did not serialize as an object"))?
            .remove("record_sha256")
            .ok_or_else(|| anyhow!("T7 record lacks record_sha256"))?;
        let bytes = serde_json::to_vec(&value).context("canonicalize T7 record for hashing")?;
        Ok(sha256_bytes(&bytes))
    }

    /// Verify the embedded digest and fail closed on any record mutation.
    pub fn verify_sha256(&self) -> Result<()> {
        require_sha256("record_sha256", &self.record_sha256)?;
        let calculated = self.calculated_sha256()?;
        if calculated != self.record_sha256 {
            bail!(
                "T7 record SHA-256 mismatch: expected {}, observed {}",
                self.record_sha256,
                calculated
            );
        }
        Ok(())
    }
}

/// Terminal state of a T7 calibration journal.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum T7JournalStatus {
    /// The caller explicitly completed and durably synchronized the journal.
    Complete,
    /// The journal closed before normal completion and remains partial evidence.
    Partial,
}

/// No-clobber closeout binding one journal to its unambiguous sample count.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct T7CalibrationSummary {
    /// Summary schema identifier.
    pub schema_kind: String,
    /// Summary schema version.
    pub schema_version: u32,
    /// Immutable execution identity.
    pub identity: T7JournalIdentity,
    /// Whether this is a normal complete or partial closeout.
    pub status: T7JournalStatus,
    /// Reason for a partial closeout; absent for normal completion.
    pub partial_reason: Option<String>,
    /// Number of hash-linked chunk records in the journal.
    pub n_records: u64,
    /// Number of emitted normalized `C_l` vectors, derived only from `sample_emitted`.
    pub n_samples: u64,
    /// Sum of GPU spike counters across recorded chunks; never a sample count.
    pub total_gpu_spike_count_observed: u64,
    /// Sum of capped spike events downloaded across recorded chunks.
    pub total_spike_events_downloaded: u64,
    /// Sum of downloaded spike events used across recorded chunks.
    pub total_spike_events_used: u64,
    /// Explicit statement that the journal cannot reconstruct raw spike clouds.
    pub raw_spike_cloud_persisted: bool,
    /// Configured maximum records between durable flushes.
    pub records_per_durable_flush: usize,
    /// SHA-256 of the first chunk record, or `null` for an empty journal.
    pub first_record_sha256: Option<String>,
    /// SHA-256 of the terminal chunk record, or `null` for an empty journal.
    pub last_record_sha256: Option<String>,
    /// SHA-256 of the exact synchronized JSONL journal bytes.
    pub journal_sha256: String,
    /// Exact synchronized JSONL journal size in bytes.
    pub journal_size_bytes: u64,
}

/// Exact byte identity of a persisted T7 artifact read from a regular file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct T7ArtifactIdentity {
    /// SHA-256 of the exact artifact bytes, without a `sha256:` prefix.
    pub sha256: String,
    /// Exact artifact length in bytes.
    pub size_bytes: u64,
}

/// Create-new writer for one stream's hash-linked T7 calibration evidence.
pub struct T7CalibrationJournal {
    identity: T7JournalIdentity,
    config: T7JournalConfig,
    journal_path: PathBuf,
    summary_path: PathBuf,
    writer: BufWriter<File>,
    pending_since_sync: usize,
    n_records: u64,
    n_samples: u64,
    total_gpu_spike_count_observed: u64,
    total_spike_events_downloaded: u64,
    total_spike_events_used: u64,
    first_record_sha256: Option<String>,
    last_record_sha256: Option<String>,
    last_chunk_index: Option<u64>,
    poisoned: Option<String>,
    close_attempted: bool,
    closed: bool,
}

/// Complete writable T7 host state plus committed journal prefix.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct T7ContinuationV1 {
    pub schema_version: u32,
    pub journal: crate::scientific_checkpoint::AppendCursor,
    pub summary_relative_path: PathBuf,
    pub summary: T7CalibrationSummary,
    pub last_chunk_index: Option<u64>,
}
impl T7CalibrationJournal {
    /// Capture only an open, healthy writer after flushing every buffered record.
    pub fn checkpoint(&mut self, run_root: &Path) -> Result<T7ContinuationV1> {
        self.flush_durable()?;
        let relative = self
            .journal_path
            .strip_prefix(run_root)?
            .to_str()
            .context("non-UTF8 T7 journal path")?;
        let journal =
            crate::scientific_checkpoint::seal_append_file(self.writer.get_ref(), relative)?;
        let summary = T7CalibrationSummary {
            schema_kind: T7_JOURNAL_SUMMARY_SCHEMA_KIND.into(),
            schema_version: T7_JOURNAL_SCHEMA_VERSION,
            identity: self.identity.clone(),
            status: T7JournalStatus::Partial,
            partial_reason: Some(
                "native scientific checkpoint boundary; writer remains open".into(),
            ),
            n_records: self.n_records,
            n_samples: self.n_samples,
            total_gpu_spike_count_observed: self.total_gpu_spike_count_observed,
            total_spike_events_downloaded: self.total_spike_events_downloaded,
            total_spike_events_used: self.total_spike_events_used,
            raw_spike_cloud_persisted: false,
            records_per_durable_flush: self.config.records_per_durable_flush,
            first_record_sha256: self.first_record_sha256.clone(),
            last_record_sha256: self.last_record_sha256.clone(),
            journal_sha256: journal.committed_sha256.clone(),
            journal_size_bytes: journal.committed_bytes,
        };
        verify_t7_calibration_journal(&self.journal_path, &summary)?;
        Ok(T7ContinuationV1 {
            schema_version: 1,
            journal,
            summary_relative_path: self.summary_path.strip_prefix(run_root)?.to_path_buf(),
            summary,
            last_chunk_index: self.last_chunk_index,
        })
    }
    /// Restore a checkpoint prefix; later output and terminal summaries are
    /// retained before restoring the live journal namespace. No new run ID.
    pub fn reopen_checkpoint(
        run_root: &Path,
        state: &T7ContinuationV1,
        expected: &T7JournalIdentity,
    ) -> Result<Self> {
        if state.schema_version != 1
            || &state.summary.identity != expected
            || state.summary.journal_sha256 != state.journal.committed_sha256
            || state.summary.journal_size_bytes != state.journal.committed_bytes
        {
            bail!("T7 continuation identity/prefix mismatch");
        }
        if state.summary_relative_path.as_os_str().is_empty()
            || !state
                .summary_relative_path
                .components()
                .all(|c| matches!(c, std::path::Component::Normal(_)))
        {
            bail!("invalid T7 summary locator");
        }
        let summary_path = run_root.join(&state.summary_relative_path);
        if summary_path.exists() {
            crate::checkpoint_output::retain_later_file(run_root, &summary_path)?;
        }
        let file = crate::scientific_checkpoint::reopen_append_file(run_root, &state.journal)?;
        let journal_path = run_root.join(&state.journal.relative_path);
        verify_t7_calibration_journal(&journal_path, &state.summary)?;
        let mut last = None;
        for line in BufReader::new(File::open(&journal_path)?).lines() {
            let r: T7ChunkRecord = serde_json::from_str(&line?)?;
            last = Some(r.chunk_index);
        }
        if last != state.last_chunk_index {
            bail!("T7 continuation last chunk differs from verified records");
        }
        let config = T7JournalConfig {
            records_per_durable_flush: state.summary.records_per_durable_flush,
        };
        config.validate()?;
        Ok(Self {
            identity: expected.clone(),
            config,
            journal_path,
            summary_path,
            writer: BufWriter::with_capacity(JOURNAL_BUFFER_BYTES, file),
            pending_since_sync: 0,
            n_records: state.summary.n_records,
            n_samples: state.summary.n_samples,
            total_gpu_spike_count_observed: state.summary.total_gpu_spike_count_observed,
            total_spike_events_downloaded: state.summary.total_spike_events_downloaded,
            total_spike_events_used: state.summary.total_spike_events_used,
            first_record_sha256: state.summary.first_record_sha256.clone(),
            last_record_sha256: state.summary.last_record_sha256.clone(),
            last_chunk_index: last,
            poisoned: None,
            close_attempted: false,
            closed: false,
        })
    }

    /// Create a journal without overwriting either its JSONL or summary path.
    ///
    /// Both parent directories must already exist. The journal is opened with
    /// `O_NOFOLLOW`, mode `0600`, and `create_new`; a pre-existing file or
    /// symlink is rejected.
    pub fn create(
        journal_path: impl AsRef<Path>,
        summary_path: impl AsRef<Path>,
        identity: T7JournalIdentity,
        config: T7JournalConfig,
    ) -> Result<Self> {
        identity.validate()?;
        config.validate()?;
        let journal_path = journal_path.as_ref().to_path_buf();
        let summary_path = summary_path.as_ref().to_path_buf();
        if journal_path == summary_path {
            bail!("journal_path and summary_path must differ");
        }
        require_existing_parent("journal_path", &journal_path)?;
        require_existing_parent("summary_path", &summary_path)?;
        require_absent_path("summary path", &summary_path)?;

        let file = OpenOptions::new()
            .read(true)
            .append(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
            .open(&journal_path)
            .with_context(|| format!("create-new T7 journal {}", journal_path.display()))?;

        Ok(Self {
            identity,
            config,
            journal_path,
            summary_path,
            writer: BufWriter::with_capacity(JOURNAL_BUFFER_BYTES, file),
            pending_since_sync: 0,
            n_records: 0,
            n_samples: 0,
            total_gpu_spike_count_observed: 0,
            total_spike_events_downloaded: 0,
            total_spike_events_used: 0,
            first_record_sha256: None,
            last_record_sha256: None,
            last_chunk_index: None,
            poisoned: None,
            close_attempted: false,
            closed: false,
        })
    }

    /// Append one validated chunk record and return its hash-bound representation.
    pub fn record(&mut self, observation: T7ChunkObservation) -> Result<T7ChunkRecord> {
        self.require_writable()?;
        observation.validate()?;
        if let Some(previous_chunk) = self.last_chunk_index {
            if observation.chunk_index <= previous_chunk {
                bail!(
                    "chunk_index must increase monotonically: previous={}, next={}",
                    previous_chunk,
                    observation.chunk_index
                );
            }
        }

        let record = T7ChunkRecord::from_observation(
            &self.identity,
            self.n_records,
            observation,
            self.last_record_sha256.clone(),
        )?;
        let next_n_records = checked_add("n_records", self.n_records, 1)?;
        let next_n_samples = checked_add(
            "n_samples",
            self.n_samples,
            u64::from(record.sample_emitted),
        )?;
        let next_total_observed = checked_add(
            "total_gpu_spike_count_observed",
            self.total_gpu_spike_count_observed,
            record.gpu_spike_count_observed,
        )?;
        let next_total_downloaded = checked_add(
            "total_spike_events_downloaded",
            self.total_spike_events_downloaded,
            record.spike_events_downloaded,
        )?;
        let next_total_used = checked_add(
            "total_spike_events_used",
            self.total_spike_events_used,
            record.spike_events_used,
        )?;
        let mut bytes = serde_json::to_vec(&record).context("serialize T7 chunk record")?;
        if bytes.len() + 1 > MAX_T7_RECORD_BYTES {
            bail!("encoded T7 chunk record exceeds {MAX_T7_RECORD_BYTES} bytes");
        }
        bytes.push(b'\n');
        if let Err(error) = self.writer.write_all(&bytes) {
            return Err(self.poison_io("write T7 chunk record", error));
        }

        self.n_records = next_n_records;
        self.n_samples = next_n_samples;
        self.total_gpu_spike_count_observed = next_total_observed;
        self.total_spike_events_downloaded = next_total_downloaded;
        self.total_spike_events_used = next_total_used;
        self.pending_since_sync += 1;
        self.first_record_sha256
            .get_or_insert_with(|| record.record_sha256.clone());
        self.last_record_sha256 = Some(record.record_sha256.clone());
        self.last_chunk_index = Some(record.chunk_index);

        if self.pending_since_sync >= self.config.records_per_durable_flush {
            self.flush_durable()?;
        }
        Ok(record)
    }

    /// Flush buffered records and request durable data synchronization.
    pub fn flush_durable(&mut self) -> Result<()> {
        self.require_writable()?;
        if let Err(error) = self.writer.flush() {
            return Err(self.poison_io("flush T7 journal", error));
        }
        if let Err(error) = self.writer.get_ref().sync_data() {
            return Err(self.poison_io("sync T7 journal data", error));
        }
        self.pending_since_sync = 0;
        Ok(())
    }

    /// Number of accepted chunk records, independent of GPU spike counts.
    pub fn n_records(&self) -> u64 {
        self.n_records
    }

    /// Number of emitted normalized `C_l` vectors.
    pub fn n_samples(&self) -> u64 {
        self.n_samples
    }

    /// Path of the create-new JSONL journal.
    pub fn journal_path(&self) -> &Path {
        &self.journal_path
    }

    /// Path reserved for the no-replace closeout summary.
    pub fn summary_path(&self) -> &Path {
        &self.summary_path
    }

    /// Durably close the journal as complete and atomically publish its summary.
    pub fn close_complete(mut self) -> Result<T7CalibrationSummary> {
        self.close_attempted = true;
        if let Some(error) = &self.poisoned {
            bail!("cannot complete poisoned T7 journal: {error}");
        }
        let summary = self.finish(T7JournalStatus::Complete, None)?;
        self.closed = true;
        Ok(summary)
    }

    /// Durably close the journal as partial and atomically publish its reason.
    pub fn close_partial(mut self, reason: impl Into<String>) -> Result<T7CalibrationSummary> {
        self.close_attempted = true;
        let reason = reason.into();
        require_bounded_text("partial_reason", &reason, MAX_PARTIAL_REASON_BYTES)?;
        let summary = self.finish(T7JournalStatus::Partial, Some(reason))?;
        self.closed = true;
        Ok(summary)
    }

    fn require_writable(&self) -> Result<()> {
        if self.close_attempted || self.closed {
            bail!("T7 journal is closing or closed");
        }
        if let Some(error) = &self.poisoned {
            bail!("T7 journal is poisoned by an earlier I/O failure: {error}");
        }
        Ok(())
    }

    fn poison_io(&mut self, operation: &str, error: std::io::Error) -> anyhow::Error {
        let message = format!("{operation}: {error}");
        self.poisoned = Some(message.clone());
        anyhow!(message)
    }

    fn finish(
        &mut self,
        status: T7JournalStatus,
        mut partial_reason: Option<String>,
    ) -> Result<T7CalibrationSummary> {
        if status == T7JournalStatus::Partial {
            if let Some(poisoned) = &self.poisoned {
                partial_reason = Some(match partial_reason {
                    Some(reason) => format!("{reason}; prior_io_error={poisoned}"),
                    None => format!("prior_io_error={poisoned}"),
                });
            }
        }

        self.writer
            .flush()
            .context("flush T7 journal during closeout")?;
        self.writer
            .get_ref()
            .sync_all()
            .context("sync T7 journal during closeout")?;
        self.pending_since_sync = 0;

        let (journal_sha256, journal_size_bytes) = hash_open_journal(self.writer.get_mut())?;
        let summary = T7CalibrationSummary {
            schema_kind: T7_JOURNAL_SUMMARY_SCHEMA_KIND.to_string(),
            schema_version: T7_JOURNAL_SCHEMA_VERSION,
            identity: self.identity.clone(),
            status,
            partial_reason,
            n_records: self.n_records,
            n_samples: self.n_samples,
            total_gpu_spike_count_observed: self.total_gpu_spike_count_observed,
            total_spike_events_downloaded: self.total_spike_events_downloaded,
            total_spike_events_used: self.total_spike_events_used,
            raw_spike_cloud_persisted: false,
            records_per_durable_flush: self.config.records_per_durable_flush,
            first_record_sha256: self.first_record_sha256.clone(),
            last_record_sha256: self.last_record_sha256.clone(),
            journal_sha256,
            journal_size_bytes,
        };
        write_json_atomic_no_replace(&self.summary_path, &summary)?;
        Ok(summary)
    }
}

/// Stream-verify a synchronized journal against its terminal or partial summary.
///
/// Verification checks every record digest, predecessor link, execution identity,
/// monotonic index, counter invariant, derived `n_samples`, and exact journal byte
/// digest. Raw spike-cloud reconstruction is intentionally outside this contract.
pub fn verify_t7_calibration_journal(
    journal_path: impl AsRef<Path>,
    summary: &T7CalibrationSummary,
) -> Result<()> {
    summary.identity.validate()?;
    if summary.schema_kind != T7_JOURNAL_SUMMARY_SCHEMA_KIND
        || summary.schema_version != T7_JOURNAL_SCHEMA_VERSION
    {
        bail!("unsupported T7 calibration summary schema");
    }
    if summary.raw_spike_cloud_persisted {
        bail!("T7 summary falsely claims raw spike-cloud persistence");
    }
    match (summary.status, summary.partial_reason.as_deref()) {
        (T7JournalStatus::Complete, None) => {}
        (T7JournalStatus::Complete, Some(_)) => {
            bail!("complete T7 summary must not carry a partial reason");
        }
        (T7JournalStatus::Partial, Some(reason)) => {
            require_bounded_text("partial_reason", reason, MAX_PARTIAL_REASON_BYTES)?;
        }
        (T7JournalStatus::Partial, None) => {
            bail!("partial T7 summary requires a reason");
        }
    }
    if summary.records_per_durable_flush == 0
        || summary.records_per_durable_flush > MAX_RECORDS_PER_DURABLE_FLUSH
    {
        bail!("summary carries an invalid durability interval");
    }
    require_sha256("journal_sha256", &summary.journal_sha256)?;

    let journal_path = journal_path.as_ref();
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(journal_path)
        .with_context(|| {
            format!(
                "open T7 journal for verification {}",
                journal_path.display()
            )
        })?;
    if !file
        .metadata()
        .context("read T7 journal metadata")?
        .is_file()
    {
        bail!(
            "T7 journal is not a regular file: {}",
            journal_path.display()
        );
    }

    let mut reader = BufReader::with_capacity(JOURNAL_BUFFER_BYTES, file);
    let mut digest = Sha256::new();
    let mut exact_size = 0_u64;
    let mut n_records = 0_u64;
    let mut n_samples = 0_u64;
    let mut total_observed = 0_u64;
    let mut total_downloaded = 0_u64;
    let mut total_used = 0_u64;
    let mut first_record_sha256: Option<String> = None;
    let mut previous_record_sha256: Option<String> = None;
    let mut previous_chunk_index: Option<u64> = None;
    let mut encoded = Vec::with_capacity(2048);

    loop {
        encoded.clear();
        let count = reader
            .read_until(b'\n', &mut encoded)
            .context("read T7 journal record")?;
        if count == 0 {
            break;
        }
        if count > MAX_T7_RECORD_BYTES {
            bail!("T7 journal record exceeds {MAX_T7_RECORD_BYTES} bytes");
        }
        if encoded.last() != Some(&b'\n') {
            bail!("T7 journal ends with an unterminated record");
        }
        digest.update(&encoded);
        exact_size = checked_add("journal_size_bytes", exact_size, count as u64)?;
        let record: T7ChunkRecord =
            serde_json::from_slice(&encoded).context("decode T7 journal record")?;
        record.verify_sha256()?;
        if record.schema_kind != T7_CHUNK_RECORD_SCHEMA_KIND
            || record.schema_version != T7_JOURNAL_SCHEMA_VERSION
        {
            bail!("unsupported T7 chunk record schema at index {n_records}");
        }
        if record.run_id != summary.identity.run_id
            || record.stream_index != summary.identity.stream_index
            || record.engine_sha256 != summary.identity.engine_sha256
            || record.topology_sha256 != summary.identity.topology_sha256
        {
            bail!("T7 record identity mismatch at index {n_records}");
        }
        if record.record_index != n_records {
            bail!(
                "T7 record index mismatch: expected {}, observed {}",
                n_records,
                record.record_index
            );
        }
        if record.previous_record_sha256 != previous_record_sha256 {
            bail!("T7 predecessor hash mismatch at index {n_records}");
        }
        if record.raw_spike_cloud_persisted {
            bail!("T7 record falsely claims raw spike-cloud persistence");
        }
        if let Some(previous_chunk) = previous_chunk_index {
            if record.chunk_index <= previous_chunk {
                bail!("T7 chunk indices are not strictly monotonic");
            }
        }

        T7ChunkObservation {
            chunk_index: record.chunk_index,
            step_start: record.step_start,
            step_end: record.step_end,
            phase: record.phase.clone(),
            gpu_spike_count_observed: record.gpu_spike_count_observed,
            gpu_count_read_status: record.gpu_count_read_status,
            gpu_count_diagnostic_code: record.gpu_count_diagnostic_code.clone(),
            spike_event_cap: record.spike_event_cap,
            spike_events_downloaded: record.spike_events_downloaded,
            spike_events_used: record.spike_events_used,
            copy_status: record.copy_status,
            normalization_status: record.normalization_status,
            sample_emitted: record.sample_emitted,
            cl_normalized: record.cl_normalized,
        }
        .validate()?;

        n_records = checked_add("n_records", n_records, 1)?;
        n_samples = checked_add("n_samples", n_samples, u64::from(record.sample_emitted))?;
        total_observed = checked_add(
            "total_gpu_spike_count_observed",
            total_observed,
            record.gpu_spike_count_observed,
        )?;
        total_downloaded = checked_add(
            "total_spike_events_downloaded",
            total_downloaded,
            record.spike_events_downloaded,
        )?;
        total_used = checked_add(
            "total_spike_events_used",
            total_used,
            record.spike_events_used,
        )?;
        first_record_sha256.get_or_insert_with(|| record.record_sha256.clone());
        previous_record_sha256 = Some(record.record_sha256);
        previous_chunk_index = Some(record.chunk_index);
    }

    let observed_journal_sha256 = format!("{:x}", digest.finalize());
    if observed_journal_sha256 != summary.journal_sha256
        || exact_size != summary.journal_size_bytes
        || n_records != summary.n_records
        || n_samples != summary.n_samples
        || total_observed != summary.total_gpu_spike_count_observed
        || total_downloaded != summary.total_spike_events_downloaded
        || total_used != summary.total_spike_events_used
        || first_record_sha256 != summary.first_record_sha256
        || previous_record_sha256 != summary.last_record_sha256
    {
        bail!("T7 journal does not match its closeout summary");
    }
    Ok(())
}

/// Read back and verify a persisted T7 summary and its bound journal.
///
/// The summary is opened with `O_NOFOLLOW`, must be a bounded regular file,
/// and is hashed over its exact bytes before deserialization. The returned
/// identity therefore binds the artifact that was actually persisted, rather
/// than the in-memory value supplied to the closeout writer.
pub fn read_and_verify_t7_calibration_summary(
    summary_path: impl AsRef<Path>,
    journal_path: impl AsRef<Path>,
) -> Result<(T7CalibrationSummary, T7ArtifactIdentity)> {
    let summary_path = summary_path.as_ref();
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(summary_path)
        .with_context(|| {
            format!(
                "open T7 summary for verification {}",
                summary_path.display()
            )
        })?;
    let metadata = file
        .metadata()
        .with_context(|| format!("read T7 summary metadata {}", summary_path.display()))?;
    if !metadata.is_file() {
        bail!(
            "T7 summary is not a regular file: {}",
            summary_path.display()
        );
    }
    if metadata.len() > MAX_T7_SUMMARY_BYTES as u64 {
        bail!("T7 summary exceeds {MAX_T7_SUMMARY_BYTES} bytes");
    }

    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    let mut bounded = file.take(MAX_T7_SUMMARY_BYTES as u64 + 1);
    bounded
        .read_to_end(&mut bytes)
        .with_context(|| format!("read T7 summary bytes {}", summary_path.display()))?;
    if bytes.len() > MAX_T7_SUMMARY_BYTES {
        bail!("T7 summary exceeds {MAX_T7_SUMMARY_BYTES} bytes");
    }
    let final_size = bounded
        .get_ref()
        .metadata()
        .with_context(|| format!("re-read T7 summary metadata {}", summary_path.display()))?
        .len();
    if final_size != bytes.len() as u64 || metadata.len() != final_size {
        bail!("T7 summary size changed during readback");
    }

    let identity = T7ArtifactIdentity {
        sha256: sha256_bytes(&bytes),
        size_bytes: bytes.len() as u64,
    };
    let summary: T7CalibrationSummary = serde_json::from_slice(&bytes)
        .with_context(|| format!("decode T7 summary {}", summary_path.display()))?;
    verify_t7_calibration_journal(journal_path, &summary)?;
    Ok((summary, identity))
}

impl Drop for T7CalibrationJournal {
    fn drop(&mut self) {
        if self.closed || self.close_attempted {
            return;
        }
        self.close_attempted = true;
        let reason = if std::thread::panicking() {
            "journal scope unwound during panic"
        } else {
            "journal dropped without explicit close"
        };
        if let Err(error) = self.finish(T7JournalStatus::Partial, Some(reason.to_string())) {
            log::error!(
                "failed to publish partial T7 journal closeout {}: {error:#}",
                self.summary_path.display()
            );
        } else {
            self.closed = true;
        }
    }
}

fn require_existing_parent(label: &str, path: &Path) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("{label} has no parent: {}", path.display()))?;
    let metadata = fs::symlink_metadata(parent)
        .with_context(|| format!("read {label} parent metadata {}", parent.display()))?;
    if metadata.file_type().is_symlink() {
        bail!("{label} parent must not be a symlink: {}", parent.display());
    }
    if !metadata.is_dir() {
        bail!("{label} parent is not a directory: {}", parent.display());
    }
    Ok(())
}

fn require_absent_path(label: &str, path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(_) => bail!("{label} already exists: {}", path.display()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| format!("inspect {label} {}", path.display())),
    }
}

fn require_bounded_text(label: &str, value: &str, max_bytes: usize) -> Result<()> {
    if value.trim().is_empty() {
        bail!("{label} must not be empty");
    }
    if value.len() > max_bytes {
        bail!("{label} exceeds {max_bytes} bytes");
    }
    if value
        .chars()
        .any(|character| matches!(character, '\n' | '\r' | '\0'))
    {
        bail!("{label} contains a forbidden control character");
    }
    Ok(())
}

fn require_diagnostic_code(label: &str, value: &str) -> Result<()> {
    require_bounded_text(label, value, MAX_T7_GPU_COUNT_DIAGNOSTIC_CODE_BYTES)?;
    if !value.bytes().all(|byte| {
        byte.is_ascii_uppercase()
            || byte.is_ascii_digit()
            || matches!(byte, b'_' | b'-' | b'.' | b':')
    }) {
        bail!(
            "{label} must contain only uppercase ASCII letters, digits, underscore, hyphen, dot, or colon"
        );
    }
    Ok(())
}

fn require_sha256(label: &str, value: &str) -> Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{label} must be exactly 64 hexadecimal characters");
    }
    Ok(())
}

fn checked_add(label: &str, lhs: u64, rhs: u64) -> Result<u64> {
    lhs.checked_add(rhs)
        .ok_or_else(|| anyhow!("{label} overflow"))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn hash_open_journal(file: &mut File) -> Result<(String, u64)> {
    file.seek(SeekFrom::Start(0))
        .context("seek T7 journal for closeout hashing")?;
    let mut digest = Sha256::new();
    let mut size = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .context("read T7 journal for closeout hashing")?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
        size = checked_add("journal_size_bytes", size, count as u64)?;
    }
    Ok((format!("{:x}", digest.finalize()), size))
}

fn write_json_atomic_no_replace<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    require_existing_parent("summary_path", path)?;
    require_absent_path("summary path", path)?;
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("summary path has no parent: {}", path.display()))?;
    let file_name = path
        .file_name()
        .ok_or_else(|| anyhow!("summary path has no file name: {}", path.display()))?
        .to_string_lossy();
    let (temporary_path, mut temporary_file) = (0..128_u64)
        .find_map(|_| {
            let nonce = TEMP_FILE_NONCE.fetch_add(1, Ordering::Relaxed);
            let candidate =
                parent.join(format!(".{file_name}.tmp.{}.{}", std::process::id(), nonce));
            match OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
                .open(&candidate)
            {
                Ok(file) => Some(Ok((candidate, file))),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(error) => Some(Err(error)),
            }
        })
        .transpose()
        .context("create temporary T7 summary")?
        .ok_or_else(|| anyhow!("could not allocate a unique temporary T7 summary"))?;

    let result = (|| -> Result<()> {
        serde_json::to_writer_pretty(&mut temporary_file, value)
            .context("serialize temporary T7 summary")?;
        temporary_file
            .write_all(b"\n")
            .context("terminate temporary T7 summary")?;
        temporary_file
            .sync_all()
            .context("sync temporary T7 summary")?;
        fs::hard_link(&temporary_path, path).with_context(|| {
            format!("publish T7 summary without replacement {}", path.display())
        })?;
        fs::remove_file(&temporary_path)
            .with_context(|| format!("unlink temporary T7 summary {}", temporary_path.display()))?;
        File::open(parent)
            .with_context(|| format!("open T7 summary parent {}", parent.display()))?
            .sync_all()
            .with_context(|| format!("sync T7 summary parent {}", parent.display()))?;
        Ok(())
    })();

    if result.is_err() {
        let _ = fs::remove_file(&temporary_path);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_NONCE: AtomicU64 = AtomicU64::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn create() -> Self {
            let nonce = TEST_NONCE.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "prismos-t7-journal-test-{}-{}",
                std::process::id(),
                nonce
            ));
            fs::create_dir(&path).expect("create isolated test directory");
            Self(path)
        }

        fn join(&self, name: &str) -> PathBuf {
            self.0.join(name)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn identity(stream_index: u32) -> T7JournalIdentity {
        T7JournalIdentity {
            run_id: "WP2_TEST_RUN".to_string(),
            stream_index,
            engine_sha256: "a".repeat(64),
            topology_sha256: "b".repeat(64),
        }
    }

    fn emitted_observation(chunk_index: u64) -> T7ChunkObservation {
        T7ChunkObservation {
            chunk_index,
            step_start: chunk_index * 250,
            step_end: chunk_index * 250 + 249,
            phase: "cold_hold".to_string(),
            gpu_spike_count_observed: 21_131,
            gpu_count_read_status: T7GpuCountReadStatus::Complete,
            gpu_count_diagnostic_code: None,
            spike_event_cap: 8_192,
            spike_events_downloaded: 8_192,
            spike_events_used: 8_192,
            copy_status: T7CopyStatus::Complete,
            normalization_status: T7NormalizationStatus::Normalized,
            sample_emitted: true,
            cl_normalized: Some([0.1, 0.2, 0.3, 0.15, 0.1, 0.15]),
        }
    }

    #[test]
    fn gpu_count_21131_emits_exactly_one_sample() {
        let directory = TestDirectory::create();
        let journal_path = directory.join("stream-0.jsonl");
        let summary_path = directory.join("stream-0.summary.json");
        let mut journal = T7CalibrationJournal::create(
            &journal_path,
            &summary_path,
            identity(0),
            T7JournalConfig {
                records_per_durable_flush: 1,
            },
        )
        .unwrap();

        journal.record(emitted_observation(0)).unwrap();
        assert_eq!(journal.n_records(), 1);
        assert_eq!(journal.n_samples(), 1);
        let summary = journal.close_complete().unwrap();

        assert_eq!(summary.status, T7JournalStatus::Complete);
        assert_eq!(summary.n_records, 1);
        assert_eq!(summary.n_samples, 1);
        assert_eq!(summary.total_gpu_spike_count_observed, 21_131);
        assert_eq!(summary.total_spike_events_downloaded, 8_192);
        assert!(!summary.raw_spike_cloud_persisted);
    }

    #[test]
    fn verifier_rejects_a_non_governed_spike_event_cap() {
        let directory = TestDirectory::create();
        let journal_path = directory.join("invalid-cap.jsonl");
        let summary_path = directory.join("invalid-cap.summary.json");
        let mut journal = T7CalibrationJournal::create(
            &journal_path,
            &summary_path,
            identity(13),
            T7JournalConfig::default(),
        )
        .unwrap();
        journal.record(emitted_observation(0)).unwrap();
        let mut summary = journal.close_complete().unwrap();

        let mut record: T7ChunkRecord =
            serde_json::from_slice(&fs::read(&journal_path).unwrap()).unwrap();
        record.spike_event_cap = T7_SPIKE_EVENT_CAP + 1;
        record.record_sha256 = record.calculated_sha256().unwrap();
        let mut bytes = serde_json::to_vec(&record).unwrap();
        bytes.push(b'\n');
        fs::write(&journal_path, &bytes).unwrap();
        summary.first_record_sha256 = Some(record.record_sha256.clone());
        summary.last_record_sha256 = Some(record.record_sha256);
        summary.journal_sha256 = sha256_bytes(&bytes);
        summary.journal_size_bytes = bytes.len() as u64;

        assert!(verify_t7_calibration_journal(&journal_path, &summary).is_err());
    }

    #[test]
    fn records_form_a_verifiable_predecessor_chain() {
        let directory = TestDirectory::create();
        let journal_path = directory.join("stream-1.jsonl");
        let summary_path = directory.join("stream-1.summary.json");
        let mut journal = T7CalibrationJournal::create(
            &journal_path,
            &summary_path,
            identity(1),
            T7JournalConfig::default(),
        )
        .unwrap();
        let first = journal.record(emitted_observation(4)).unwrap();
        let second = journal.record(emitted_observation(5)).unwrap();
        assert_eq!(first.previous_record_sha256, None);
        assert_eq!(
            second.previous_record_sha256,
            Some(first.record_sha256.clone())
        );
        first.verify_sha256().unwrap();
        second.verify_sha256().unwrap();

        let summary = journal.close_complete().unwrap();
        assert_eq!(summary.first_record_sha256, Some(first.record_sha256));
        assert_eq!(summary.last_record_sha256, Some(second.record_sha256));
        assert_eq!(summary.n_samples, 2);

        let bytes = fs::read(&journal_path).unwrap();
        assert_eq!(summary.journal_sha256, sha256_bytes(&bytes));
        assert_eq!(summary.journal_size_bytes, bytes.len() as u64);
        verify_t7_calibration_journal(&journal_path, &summary).unwrap();
        let summary_bytes = fs::read(&summary_path).unwrap();
        let (persisted_summary, summary_identity) =
            read_and_verify_t7_calibration_summary(&summary_path, &journal_path).unwrap();
        assert_eq!(persisted_summary, summary);
        assert_eq!(summary_identity.sha256, sha256_bytes(&summary_bytes));
        assert_eq!(summary_identity.size_bytes, summary_bytes.len() as u64);
    }

    #[test]
    fn failed_count_read_is_distinct_from_a_completed_zero_count() {
        let directory = TestDirectory::create();
        let journal_path = directory.join("stream-count-failed.jsonl");
        let summary_path = directory.join("stream-count-failed.summary.json");
        let mut journal = T7CalibrationJournal::create(
            &journal_path,
            &summary_path,
            identity(8),
            T7JournalConfig::default(),
        )
        .unwrap();
        let mut failed = emitted_observation(0);
        failed.gpu_spike_count_observed = 0;
        failed.gpu_count_read_status = T7GpuCountReadStatus::Failed;
        failed.gpu_count_diagnostic_code = Some("CUDA_ERROR_INVALID_VALUE".to_string());
        failed.spike_events_downloaded = 0;
        failed.spike_events_used = 0;
        failed.copy_status = T7CopyStatus::NotAttempted;
        failed.normalization_status = T7NormalizationStatus::NotAttempted;
        failed.sample_emitted = false;
        failed.cl_normalized = None;
        let failed_record = journal.record(failed).unwrap();
        assert_eq!(
            failed_record.gpu_count_read_status,
            T7GpuCountReadStatus::Failed
        );
        assert_eq!(
            failed_record.gpu_count_diagnostic_code.as_deref(),
            Some("CUDA_ERROR_INVALID_VALUE")
        );

        let mut zero = emitted_observation(1);
        zero.gpu_spike_count_observed = 0;
        zero.spike_events_downloaded = 0;
        zero.spike_events_used = 0;
        zero.copy_status = T7CopyStatus::NotAttempted;
        zero.normalization_status = T7NormalizationStatus::NotAttempted;
        zero.sample_emitted = false;
        zero.cl_normalized = None;
        let zero_record = journal.record(zero).unwrap();
        assert_eq!(
            zero_record.gpu_count_read_status,
            T7GpuCountReadStatus::Complete
        );
        assert_eq!(zero_record.gpu_count_diagnostic_code, None);
        let summary = journal.close_complete().unwrap();
        assert_eq!(summary.n_records, 2);
        assert_eq!(summary.n_samples, 0);
        read_and_verify_t7_calibration_summary(&summary_path, &journal_path).unwrap();
    }

    #[test]
    fn failed_count_read_cannot_claim_downstream_capture() {
        let directory = TestDirectory::create();
        let mut journal = T7CalibrationJournal::create(
            directory.join("invalid-count.jsonl"),
            directory.join("invalid-count.summary.json"),
            identity(9),
            T7JournalConfig::default(),
        )
        .unwrap();
        let mut invalid = emitted_observation(0);
        invalid.gpu_count_read_status = T7GpuCountReadStatus::Unavailable;
        invalid.gpu_count_diagnostic_code = Some("GPU_BUFFERS_UNAVAILABLE".to_string());
        assert!(journal.record(invalid).is_err());
        assert_eq!(journal.n_records(), 0);
    }

    #[test]
    fn diagnostic_code_is_bounded_and_machine_readable() {
        let directory = TestDirectory::create();
        let mut journal = T7CalibrationJournal::create(
            directory.join("invalid-diagnostic.jsonl"),
            directory.join("invalid-diagnostic.summary.json"),
            identity(10),
            T7JournalConfig::default(),
        )
        .unwrap();
        let mut invalid = emitted_observation(0);
        invalid.gpu_count_read_status = T7GpuCountReadStatus::Failed;
        invalid.gpu_count_diagnostic_code = Some("lowercase diagnostic".to_string());
        invalid.gpu_spike_count_observed = 0;
        invalid.spike_events_downloaded = 0;
        invalid.spike_events_used = 0;
        invalid.copy_status = T7CopyStatus::NotAttempted;
        invalid.normalization_status = T7NormalizationStatus::NotAttempted;
        invalid.sample_emitted = false;
        invalid.cl_normalized = None;
        assert!(journal.record(invalid.clone()).is_err());
        invalid.gpu_count_diagnostic_code =
            Some("X".repeat(MAX_T7_GPU_COUNT_DIAGNOSTIC_CODE_BYTES + 1));
        assert!(journal.record(invalid).is_err());
        assert_eq!(journal.n_records(), 0);
    }

    #[test]
    fn create_new_rejects_journal_and_summary_collisions() {
        let directory = TestDirectory::create();
        let journal_path = directory.join("stream-2.jsonl");
        let summary_path = directory.join("stream-2.summary.json");
        let journal = T7CalibrationJournal::create(
            &journal_path,
            &summary_path,
            identity(2),
            T7JournalConfig::default(),
        )
        .unwrap();

        let second = T7CalibrationJournal::create(
            &journal_path,
            directory.join("different.summary.json"),
            identity(2),
            T7JournalConfig::default(),
        );
        assert!(second.is_err());
        journal.close_complete().unwrap();

        let summary_collision = T7CalibrationJournal::create(
            directory.join("different.jsonl"),
            &summary_path,
            identity(2),
            T7JournalConfig::default(),
        );
        assert!(summary_collision.is_err());
    }

    #[test]
    fn invalid_sample_semantics_do_not_mutate_the_journal() {
        let directory = TestDirectory::create();
        let journal_path = directory.join("stream-3.jsonl");
        let summary_path = directory.join("stream-3.summary.json");
        let mut journal = T7CalibrationJournal::create(
            &journal_path,
            &summary_path,
            identity(3),
            T7JournalConfig::default(),
        )
        .unwrap();
        let mut invalid = emitted_observation(0);
        invalid.sample_emitted = false;
        assert!(journal.record(invalid).is_err());
        assert_eq!(journal.n_records(), 0);
        assert_eq!(journal.n_samples(), 0);
        let summary = journal.close_complete().unwrap();
        assert_eq!(summary.n_records, 0);
        assert_eq!(summary.n_samples, 0);
    }

    #[test]
    fn drop_publishes_a_partial_closeout() {
        let directory = TestDirectory::create();
        let journal_path = directory.join("stream-4.jsonl");
        let summary_path = directory.join("stream-4.summary.json");
        {
            let mut journal = T7CalibrationJournal::create(
                &journal_path,
                &summary_path,
                identity(4),
                T7JournalConfig::default(),
            )
            .unwrap();
            journal.record(emitted_observation(0)).unwrap();
        }

        let summary: T7CalibrationSummary =
            serde_json::from_slice(&fs::read(summary_path).unwrap()).unwrap();
        assert_eq!(summary.status, T7JournalStatus::Partial);
        assert_eq!(summary.n_samples, 1);
        assert_eq!(
            summary.partial_reason.as_deref(),
            Some("journal dropped without explicit close")
        );
    }

    #[test]
    fn complete_closeout_never_replaces_an_existing_summary() {
        let directory = TestDirectory::create();
        let journal_path = directory.join("stream-5.jsonl");
        let summary_path = directory.join("stream-5.summary.json");
        let mut journal = T7CalibrationJournal::create(
            &journal_path,
            &summary_path,
            identity(5),
            T7JournalConfig::default(),
        )
        .unwrap();
        journal.record(emitted_observation(0)).unwrap();
        fs::write(&summary_path, b"preexisting").unwrap();
        assert!(journal.close_complete().is_err());
        assert_eq!(fs::read(&summary_path).unwrap(), b"preexisting");
    }

    #[test]
    fn verifier_rejects_a_mutated_journal() {
        let directory = TestDirectory::create();
        let journal_path = directory.join("stream-6.jsonl");
        let summary_path = directory.join("stream-6.summary.json");
        let mut journal = T7CalibrationJournal::create(
            &journal_path,
            &summary_path,
            identity(6),
            T7JournalConfig::default(),
        )
        .unwrap();
        journal.record(emitted_observation(0)).unwrap();
        let summary = journal.close_complete().unwrap();
        verify_t7_calibration_journal(&journal_path, &summary).unwrap();

        let mut bytes = fs::read(&journal_path).unwrap();
        let position = bytes
            .windows(b"21131".len())
            .position(|window| window == b"21131")
            .expect("encoded GPU count");
        bytes[position] = b'3';
        fs::write(&journal_path, bytes).unwrap();
        assert!(verify_t7_calibration_journal(&journal_path, &summary).is_err());
    }

    #[test]
    fn persisted_summary_readback_rejects_mutated_counts() {
        let directory = TestDirectory::create();
        let journal_path = directory.join("stream-summary-mutation.jsonl");
        let summary_path = directory.join("stream-summary-mutation.summary.json");
        let mut journal = T7CalibrationJournal::create(
            &journal_path,
            &summary_path,
            identity(11),
            T7JournalConfig::default(),
        )
        .unwrap();
        journal.record(emitted_observation(0)).unwrap();
        journal.close_complete().unwrap();

        let mut summary: serde_json::Value =
            serde_json::from_slice(&fs::read(&summary_path).unwrap()).unwrap();
        summary["n_samples"] = serde_json::json!(21_131);
        fs::write(&summary_path, serde_json::to_vec_pretty(&summary).unwrap()).unwrap();
        assert!(read_and_verify_t7_calibration_summary(&summary_path, &journal_path).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn persisted_summary_readback_rejects_a_symlink() {
        use std::os::unix::fs::symlink;

        let directory = TestDirectory::create();
        let journal_path = directory.join("stream-summary-symlink.jsonl");
        let summary_path = directory.join("stream-summary-symlink.summary.json");
        let linked_summary_path = directory.join("linked.summary.json");
        let journal = T7CalibrationJournal::create(
            &journal_path,
            &summary_path,
            identity(12),
            T7JournalConfig::default(),
        )
        .unwrap();
        journal.close_complete().unwrap();
        symlink(&summary_path, &linked_summary_path).unwrap();
        assert!(
            read_and_verify_t7_calibration_summary(&linked_summary_path, &journal_path).is_err()
        );
    }

    #[cfg(unix)]
    #[test]
    fn create_rejects_a_symlink_parent() {
        use std::os::unix::fs::symlink;

        let directory = TestDirectory::create();
        let real_parent = directory.join("real-parent");
        let linked_parent = directory.join("linked-parent");
        fs::create_dir(&real_parent).unwrap();
        symlink(&real_parent, &linked_parent).unwrap();
        let result = T7CalibrationJournal::create(
            linked_parent.join("stream.jsonl"),
            linked_parent.join("stream.summary.json"),
            identity(7),
            T7JournalConfig::default(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn checkpoint_reopen_retains_later_records_and_restores_hash_chain() {
        let dir = TestDirectory::create();
        let id = identity(0);
        let mut journal = T7CalibrationJournal::create(
            dir.join("records.jsonl"),
            dir.join("summary.json"),
            id.clone(),
            T7JournalConfig::default(),
        )
        .unwrap();
        journal.record(emitted_observation(0)).unwrap();
        let boundary = journal.checkpoint(&dir.0).unwrap();
        journal.record(emitted_observation(1)).unwrap();
        journal
            .close_partial("interrupted after checkpoint")
            .unwrap();
        let original = fs::read(dir.join("records.jsonl")).unwrap();
        let mut restored = T7CalibrationJournal::reopen_checkpoint(&dir.0, &boundary, &id).unwrap();
        assert_eq!(restored.n_records(), 1);
        restored.record(emitted_observation(1)).unwrap();
        let complete = restored.close_complete().unwrap();
        assert_eq!(fs::read(dir.join("records.jsonl")).unwrap(), original);
        assert_eq!(complete.n_records, 2);
        verify_t7_calibration_journal(dir.join("records.jsonl"), &complete).unwrap();
        assert!(dir.join(".continuation_tails").is_dir());
    }
}
