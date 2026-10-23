//! Same-runtime scientific continuation storage.
//!
//! The capsule commits a caller-declared *complete* owner registry at a common
//! synchronized boundary. It deliberately does not infer closure from positions,
//! a device allocation list, an engine log, or a successful file write. Actual
//! engine owners must provide every field in the admitted registry; missing or
//! extra fields are errors. Handles and pointers must be reconstructed by their
//! owners, never loaded as persisted CUDA addresses.
//!
//! A local flock serializes writers. Immutable content objects and manifests are
//! fsynced before an atomic, directory-synced HEAD publication. A process crash
//! before HEAD publication leaves the previous committed generation readable;
//! unreachable objects may be collected separately. This is a local POSIX
//! filesystem contract, not a claim about untested network filesystems or power
//! loss. Restore verifies the entire committed population before returning any
//! bytes to a device consumer.

use anyhow::{ensure, Context, Result};
#[cfg(feature = "gpu")]
use anyhow::bail;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

pub mod chronicle;
pub mod restart_transaction;
pub mod lossless_object;
pub mod native_spike_archive;

pub const SCHEMA: u32 = 1;
const MAX_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;
const MAX_FIELDS: usize = 100_000;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointIdentity {
    pub run_id: String,
    pub topology_sha256: String,
    pub engine_sha256: String,
    pub kernel_set_sha256: String,
    pub protocol_sha256: String,
    pub execution_request_sha256: String,
    /// Same-build ABI and precision contract; migration is a different operation.
    pub state_abi_sha256: String,
    pub provider: String,
    pub stream_count: u32,
}
impl CheckpointIdentity {
    fn validate(&self) -> Result<()> {
        ensure!(!self.run_id.is_empty() && self.run_id.len() <= 1024, "invalid run identity");
        ensure!(!self.provider.is_empty() && self.provider.len() <= 1024, "missing provider identity");
        ensure!(self.stream_count > 0 && self.stream_count <= 65536, "invalid stream count");
        for digest in [&self.topology_sha256, &self.engine_sha256, &self.kernel_set_sha256,
            &self.protocol_sha256, &self.execution_request_sha256, &self.state_abi_sha256] {
            validate_digest(digest)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamBoundary {
    pub stream_id: u32,
    pub replica_epoch: u32,
    pub replica_seed: u64,
    pub completed_step: u64,
    /// Exact native f64 bits, not a wall-clock estimate or nominal dt product.
    pub physical_time_ps_bits: u64,
    pub next_integrator_phase: String,
    pub coupling_epoch: u64,
    pub coupling_parity: u8,
    pub committed_output_sequence: u64,
}
impl StreamBoundary {
    fn validate(&self) -> Result<()> {
        let t = f64::from_bits(self.physical_time_ps_bits);
        ensure!(t.is_finite() && t >= 0.0, "invalid exact physical clock");
        ensure!(!self.next_integrator_phase.is_empty(), "missing integrator phase");
        ensure!(self.coupling_parity < 2, "invalid coupling parity");
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FieldContract {
    /// Canonical owner name (for example stream_0/fused or stream_0/governor).
    pub owner: String,
    pub name: String,
    /// Typed wire format, including layout version, precision and endianness.
    pub encoding: String,
    /// None permits a variable-size history; Some binds the exact allocation.
    pub expected_bytes: Option<u64>,
    /// Optional allocation absence is represented explicitly, never omitted.
    pub may_be_absent: bool,
}
impl FieldContract {
    pub fn key(&self) -> String { format!("{}/{}", self.owner, self.name) }
    pub(crate) fn validate(&self) -> Result<()> {
        validate_label(&self.owner)?;
        validate_label(&self.name)?;
        ensure!(!self.encoding.is_empty() && self.encoding.len() <= 1024,
            "missing typed encoding for {}", self.key());
        Ok(())
    }
}

/// Constructed from source-owned registries. A caller cannot replace missing
/// owner state by marking an arbitrary field optional at capture time: this
/// exact registry is a pinned operand of the experiment/release contract.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClosureContract {
    pub schema_version: u32,
    pub fields: Vec<FieldContract>,
    pub required_owners: BTreeSet<String>,
    /// Nonempty while any native state owner lacks a complete implementation.
    pub unresolved_owners: BTreeSet<String>,
}
impl ClosureContract {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.schema_version == SCHEMA, "unsupported closure schema");
        ensure!(self.unresolved_owners.is_empty(),
            "scientific continuation closure is incomplete: {:?}", self.unresolved_owners);
        ensure!(!self.fields.is_empty() && self.fields.len() <= MAX_FIELDS,
            "invalid checkpoint field population");
        let mut keys = BTreeSet::new();
        let mut owners = BTreeSet::new();
        for field in &self.fields {
            field.validate()?;
            ensure!(keys.insert(field.key()), "duplicate field {}", field.key());
            owners.insert(field.owner.clone());
        }
        ensure!(owners == self.required_owners, "owner/field coverage differs from closure contract");
        Ok(())
    }
    pub fn digest(&self) -> Result<String> {
        self.validate()?;
        // Contract field ordering is significant and is included in identity.
        Ok(hash(&serde_json::to_vec(self)?))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StoredField {
    contract: FieldContract,
    present: bool,
    byte_count: u64,
    object_sha256: Option<String>,
    #[serde(default,skip_serializing_if="Option::is_none")]
    representation:Option<lossless_object::LosslessObjectEnvelope>,
    #[serde(default,skip_serializing_if="Option::is_none")]
    host_allocation_budget:Option<u64>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Head {
    pub generation: u64,
    pub manifest_sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: u32,
    generation: u64,
    previous_head: Option<Head>,
    identity: CheckpointIdentity,
    closure_contract_sha256: String,
    boundaries: Vec<StreamBoundary>,
    fields: Vec<StoredField>,
}

/// Host bytes are explicit typed state. CUDA pointers and live handles are not
/// accepted as a restart contract merely because they fit in a byte vector.
#[derive(Clone, Debug,Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapturedField {
    pub contract: FieldContract,
    #[serde(with="crate::checkpoint_codec::optional_bytes")]
    pub bytes: Option<Vec<u8>>,
}

pub struct CheckpointStore {
    root:PathBuf,max_total_bytes:u64,
    parallel_staging:std::sync::Mutex<Option<std::sync::Arc<ParallelScientificStagingLease>>>,
    reservation_updates:std::sync::Mutex<()>,
    lossless_publication_stripes:[std::sync::Mutex<()>;64],
    // Operational immutable-object witnesses only. A new process starts empty;
    // actual restored payloads are still hash-checked as they are decoded.
    verified_lossless:std::sync::Mutex<BTreeMap<String,(lossless_object::LosslessObjectEnvelope,lossless_object::LosslessFileStamp)>>,
    verified_native_archive_sources:std::sync::Mutex<BTreeMap<String,Vec<(String,native_spike_archive::RawStamp)>>> ,
}
struct WriterLock(File);
// Operational same-process pause lease only, never serialized scientific state.
struct ParallelScientificStagingLease {_writer:WriterLock}
enum ScientificStagingGuard {Shared(std::sync::Arc<ParallelScientificStagingLease>),Exclusive(WriterLock)}
impl Drop for WriterLock {
    fn drop(&mut self) { unsafe { libc::flock(self.0.as_raw_fd(), libc::LOCK_UN); } }
}

impl CheckpointStore {
    /// The parent directory must already exist and every ancestor must be a
    /// real directory. Run directories are operator-controlled, mode 0700.
    pub fn create(root: &Path, max_total_bytes: u64) -> Result<Self> {
        ensure!(max_total_bytes > 0, "checkpoint byte budget is zero");
        ensure!(root.is_absolute(), "checkpoint root must be absolute");
        let parent = root.parent().context("checkpoint root has no parent")?;
        verify_directory_chain(parent)?;
        if !root.exists() {
            fs::create_dir(root)?;
            fs::set_permissions(root, fs::Permissions::from_mode(0o700))?;
            sync_dir(parent)?;
        }
        verify_directory_chain(root)?;
        for name in ["objects", "manifests", "encoded_objects"] {
            let dir = root.join(name);
            if !dir.exists() { fs::create_dir(&dir)?; }
            verify_directory_chain(&dir)?;
        }
        sync_dir(root)?;
        Ok(Self { root: root.to_owned(), max_total_bytes,parallel_staging:std::sync::Mutex::new(None),reservation_updates:std::sync::Mutex::new(()),lossless_publication_stripes:std::array::from_fn(|_|std::sync::Mutex::new(())),verified_lossless:std::sync::Mutex::new(BTreeMap::new()),verified_native_archive_sources:std::sync::Mutex::new(BTreeMap::new()) })
    }
    fn lock(&self) -> Result<WriterLock> {
        let file = OpenOptions::new().read(true).write(true).create(true)
            .mode(0o600).custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(self.root.join("LOCK"))?;
        ensure!(file.metadata()?.is_file(), "checkpoint LOCK is not a file");
        let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        ensure!(rc == 0, "checkpoint writer is already active: {}", std::io::Error::last_os_error());
        Ok(WriterLock(file))
    }
    /// Begin only after all paused owners have admitted their full projection.
    /// One original cross-process LOCK stays held while at most two staging
    /// lanes operate. Ordinary SCI/raw writers still use the original lock().
    pub(crate) fn begin_parallel_scientific_staging(&self)->Result<()> {
        let mut session=self.parallel_staging.lock().map_err(|_|anyhow::anyhow!("scientific staging lease poisoned"))?;
        ensure!(session.is_none(),"scientific staging lease already active");
        let writer=self.lock()?;self.ensure_scientific_producer_open()?;
        ensure!(!self.root.join(".scientific_branch_copy_reservation").try_exists()?,"parallel scientific capture cannot overlap historical branch copying");
        *session=Some(std::sync::Arc::new(ParallelScientificStagingLease{_writer:writer}));Ok(())
    }
    pub(crate) fn parallel_scientific_staging_active(&self)->Result<bool>{
        Ok(self.parallel_staging.lock().map_err(|_|anyhow::anyhow!("scientific staging lease poisoned"))?.is_some())
    }
    pub(crate) fn finish_parallel_scientific_staging(&self)->Result<()> {
        let mut session=self.parallel_staging.lock().map_err(|_|anyhow::anyhow!("scientific staging lease poisoned"))?;
        let active=session.as_ref().context("scientific staging lease absent at completion")?;
        ensure!(std::sync::Arc::strong_count(active)==1,"scientific staging completion still has live field writers");
        // Every field producer returned. The existing atomic commit reacquires
        // LOCK and checks exact prior HEAD, just as between old serial stages.
        let lease=session.take();drop(session);drop(lease);Ok(())
    }
    pub(crate) fn abort_parallel_scientific_staging(&self) {
        if let Ok(mut session)=self.parallel_staging.lock(){
            // Active guards retain the original fd until their last write ends.
            let lease=session.take();drop(session);drop(lease);
        }
    }
    fn scientific_staging_guard(&self)->Result<ScientificStagingGuard> {
        let session=self.parallel_staging.lock().map_err(|_|anyhow::anyhow!("scientific staging lease poisoned"))?;
        if let Some(active)=session.as_ref(){return Ok(ScientificStagingGuard::Shared(active.clone()));}
        drop(session);Ok(ScientificStagingGuard::Exclusive(self.lock()?))
    }
    fn lossless_publication_guard(&self,digest:&str)->Result<std::sync::MutexGuard<'_,()>> {
        validate_digest(digest)?;let stripe=usize::from(u8::from_str_radix(&digest[..2],16)?)%self.lossless_publication_stripes.len();
        self.lossless_publication_stripes[stripe].lock().map_err(|_|anyhow::anyhow!("scientific object publication stripe poisoned"))
    }
    pub fn head(&self) -> Result<Option<Head>> { self.read_head_named("HEAD") }
    fn read_head_named(&self, name: &str) -> Result<Option<Head>> {
        ensure!(matches!(name, "HEAD" | "CHRONICLE_HEAD"), "unknown store head namespace");
        match read_bounded(&self.root.join(name), 4096) {
            Ok(bytes) => {
                let head: Head = serde_json::from_slice(&bytes)?;
                validate_digest(&head.manifest_sha256)?;
                ensure!(head.generation > 0, "invalid checkpoint generation");
                Ok(Some(head))
            }
            Err(e) if e.downcast_ref::<std::io::Error>()
                .is_some_and(|e| e.kind() == std::io::ErrorKind::NotFound) => Ok(None),
            Err(e) => Err(e),
        }
    }
    pub fn commit(&self, expected_head: Option<&Head>, identity: &CheckpointIdentity,
        closure: &ClosureContract, boundaries: &[StreamBoundary], fields: &[CapturedField]) -> Result<Head> {
        self.commit_with_hook(expected_head, identity, closure, boundaries, fields, |_| Ok(()))
    }
    fn commit_with_hook(&self, expected_head: Option<&Head>, identity: &CheckpointIdentity,
        closure: &ClosureContract, boundaries: &[StreamBoundary], fields: &[CapturedField],
        mut hook: impl FnMut(&str) -> Result<()>) -> Result<Head> {
        identity.validate()?;
        closure.validate()?;
        validate_boundaries(identity, boundaries)?;
        validate_population(closure, fields, self.max_total_bytes)?;
        let _lock = self.lock()?;
        self.ensure_scientific_producer_open()?;
        let current = self.head()?;
        ensure!(current.as_ref() == expected_head, "stale checkpoint HEAD; caller must revalidate");
        let generation = current.as_ref().map(|x| x.generation).unwrap_or(0)
            .checked_add(1).context("checkpoint generation overflow")?;
        if current.is_some() {
            let previous = self.read_manifest(current.as_ref().unwrap())?;
            ensure!(previous.identity == *identity, "checkpoint identity changed within run");
            ensure!(previous.closure_contract_sha256 == closure.digest()?, "checkpoint closure changed within run");
            for (prior, next) in previous.boundaries.iter().zip(boundaries) {
                ensure!(next.replica_epoch == prior.replica_epoch && next.replica_seed == prior.replica_seed,
                    "replica identity changed during continuation");
                ensure!(next.completed_step > prior.completed_step, "checkpoint scientific step did not advance");
                ensure!(f64::from_bits(next.physical_time_ps_bits) > f64::from_bits(prior.physical_time_ps_bits),
                    "checkpoint physical clock did not advance");
                ensure!(next.coupling_epoch >= prior.coupling_epoch &&
                    next.committed_output_sequence >= prior.committed_output_sequence,
                    "checkpoint coupling/output chronology regressed");
            }
        }
        let by_key: BTreeMap<_, _> = fields.iter().map(|f| (f.contract.key(), f)).collect();
        let mut stored = Vec::with_capacity(fields.len());
        for contract in &closure.fields {
            let field = by_key[&contract.key()];
            let object = if let Some(bytes) = &field.bytes {
                let digest = hash(bytes);
                self.write_immutable("objects", &digest, bytes)?;
                Some(digest)
            } else { None };
            stored.push(StoredField { contract: contract.clone(), present: field.bytes.is_some(),
                byte_count: field.bytes.as_ref().map(|x| x.len() as u64).unwrap_or(0), object_sha256: object, representation:None,host_allocation_budget:None });
        }
        sync_dir(&self.root.join("objects"))?;
        hook("objects_synced")?;
        let manifest = Manifest { schema_version: SCHEMA, generation, previous_head: current,
            identity: identity.clone(), closure_contract_sha256: closure.digest()?,
            boundaries: boundaries.to_owned(), fields: stored };
        let bytes = serde_json::to_vec(&manifest)?;
        ensure!(bytes.len() as u64 <= MAX_MANIFEST_BYTES, "checkpoint manifest exceeds limit");
        let digest = hash(&bytes);
        self.write_immutable("manifests", &digest, &bytes)?;
        sync_dir(&self.root.join("manifests"))?;
        hook("manifest_synced")?;
        let head = Head { generation, manifest_sha256: digest };
        self.publish_head_named("HEAD", &head, &mut hook)?;
        Ok(head)
    }
    fn publish_head_named(&self, name: &str, head: &Head, hook: &mut impl FnMut(&str) -> Result<()>) -> Result<()> {
        ensure!(matches!(name, "HEAD" | "CHRONICLE_HEAD"), "unknown store head namespace");
        let head_bytes = serde_json::to_vec(head)?;
        // A previous crash may leave HEAD.next; only the current lock owner may
        // retire it. It was never an advertised committed generation.
        let staged = self.root.join(format!("{name}.next"));
        match fs::symlink_metadata(&staged) {
            Ok(meta) => { ensure!(meta.is_file() && !meta.file_type().is_symlink(), "invalid staged HEAD"); fs::remove_file(&staged)?; },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {},
            Err(e) => return Err(e.into()),
        }
        write_new(&staged, &head_bytes)?;
        hook("head_staged_synced")?;
        fs::rename(&staged, self.root.join(name))?;
        hook("head_published")?;
        sync_dir(&self.root)?;
        hook("head_directory_synced")?;
        Ok(())
    }
    fn write_immutable(&self, area: &str, digest: &str, bytes: &[u8]) -> Result<()> {
        validate_digest(digest)?;
        let dest = self.root.join(area).join(digest);
        if dest.exists() {
            if area=="objects" {self.verify_object_streaming(digest,bytes.len() as u64)?;}
            else {ensure!(read_bounded(&dest,bytes.len() as u64)?==bytes,"immutable object collision/corruption");}
            return Ok(());
        }
        // Write privately first. A partial object at its content address must
        // never be mistaken for committed state after process interruption.
        let temporary = self.root.join(area).join(format!(".{digest}.next"));
        match fs::symlink_metadata(&temporary) {
            Ok(meta) => { ensure!(meta.is_file() && !meta.file_type().is_symlink(), "invalid staged object"); fs::remove_file(&temporary)?; },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {},
            Err(e) => return Err(e.into()),
        }
        self.spend_reserved_capacity(bytes.len() as u64)?;
        write_new(&temporary, bytes)?;
        fs::rename(&temporary, &dest)?;
        Ok(())
    }
    fn read_manifest(&self, head: &Head) -> Result<Manifest> {
        validate_digest(&head.manifest_sha256)?;
        let bytes = read_bounded(&self.root.join("manifests").join(&head.manifest_sha256), MAX_MANIFEST_BYTES)?;
        ensure!(hash(&bytes) == head.manifest_sha256, "checkpoint manifest digest mismatch");
        let manifest: Manifest = serde_json::from_slice(&bytes)?;
        ensure!(manifest.schema_version == SCHEMA && manifest.generation == head.generation,
            "checkpoint schema/generation mismatch");
        Ok(manifest)
    }
    /// Inspect only the hash-verified HEAD identity to recover its run ID.
    /// The caller must still construct and compare every current identity operand
    /// through restore() before using any scientific state.
    pub fn identity_at_head(&self)->Result<Option<(Head,CheckpointIdentity)>> {
        let _lock=self.lock()?;let Some(head)=self.head()? else{return Ok(None)};
        let manifest=self.read_manifest(&head)?;manifest.identity.validate()?;
        Ok(Some((head,manifest.identity)))
    }
    pub fn restore(&self, expected_head: &Head, identity: &CheckpointIdentity,
        closure: &ClosureContract) -> Result<RestoredCheckpoint> {
        identity.validate()?;
        closure.validate()?;
        let lock = self.lock()?;
        self.restore_under_writer_lock(&lock, expected_head, identity, closure)
    }
    fn restore_under_writer_lock(&self, _lock: &WriterLock, expected_head: &Head,
        identity: &CheckpointIdentity, closure: &ClosureContract) -> Result<RestoredCheckpoint> {
        identity.validate()?;
        closure.validate()?;
        // Refuse accidental rollback. Reading historical generations is a
        // distinct inspection operation, not automatic scientific continuation.
        ensure!(self.head()?.as_ref() == Some(expected_head), "stale restore generation");
        let manifest = self.read_manifest(expected_head)?;
        ensure!(manifest.identity == *identity, "checkpoint execution identity mismatch");
        ensure!(manifest.closure_contract_sha256 == closure.digest()?, "checkpoint state closure mismatch");
        validate_boundaries(identity, &manifest.boundaries)?;
        ensure!(manifest.fields.len() == closure.fields.len(), "checkpoint field population mismatch");
        let mut total = 0u64;
        let mut fields = Vec::with_capacity(manifest.fields.len());
        for (stored, contract) in manifest.fields.iter().zip(&closure.fields) {
            ensure!(stored.contract == *contract, "checkpoint field contract/order mismatch");
            total = total.checked_add(stored.byte_count).context("checkpoint byte count overflow")?;
            ensure!(total <= self.max_total_bytes, "checkpoint exceeds admitted byte budget");
            let bytes = if stored.present {
                let digest = stored.object_sha256.as_ref().context("present field has no object digest")?;
                validate_digest(digest)?;
                let bytes = if let Some(envelope)=&stored.representation {
                    ensure!(envelope.decoded_sha256==*digest && envelope.decoded_bytes==stored.byte_count,"lossless field identity mismatch");
                    self.read_lossless_object(envelope,stored.byte_count)?
                }else{read_bounded(&self.root.join("objects").join(digest), stored.byte_count)?};
                ensure!(bytes.len() as u64 == stored.byte_count && hash(&bytes) == *digest,
                    "checkpoint field {} is corrupt or truncated", contract.key());
                Some(bytes)
            } else {
                ensure!(contract.may_be_absent && stored.byte_count == 0 && stored.object_sha256.is_none(),
                    "required checkpoint field {} absent", contract.key());
                None
            };
            fields.push(CapturedField { contract: contract.clone(), bytes });
        }
        validate_population(closure, &fields, self.max_total_bytes)?;
        Ok(RestoredCheckpoint { head: expected_head.clone(), boundaries: manifest.boundaries, fields })
    }
}

pub struct RestoredCheckpoint {
    pub head: Head,
    pub boundaries: Vec<StreamBoundary>,
    pub fields: Vec<CapturedField>,
}

fn validate_population(closure: &ClosureContract, fields: &[CapturedField], limit: u64) -> Result<()> {
    ensure!(fields.len() == closure.fields.len(), "missing or extra checkpoint fields");
    let expected: BTreeMap<_, _> = closure.fields.iter().map(|x| (x.key(), x)).collect();
    let mut seen = BTreeSet::new();
    let mut total = 0u64;
    for field in fields {
        let key = field.contract.key();
        ensure!(seen.insert(key.clone()), "duplicate checkpoint field {key}");
        ensure!(expected.get(&key).copied() == Some(&field.contract), "unadmitted field contract {key}");
        if let Some(bytes) = &field.bytes {
            ensure!(field.contract.expected_bytes.is_none_or(|n| n == bytes.len() as u64),
                "checkpoint allocation length mismatch: {key}");
            total = total.checked_add(bytes.len() as u64).context("checkpoint byte count overflow")?;
            ensure!(total <= limit, "checkpoint exceeds admitted byte budget");
        } else { ensure!(field.contract.may_be_absent, "required checkpoint field {key} absent"); }
    }
    Ok(())
}
fn validate_boundaries(identity: &CheckpointIdentity, boundaries: &[StreamBoundary]) -> Result<()> {
    ensure!(boundaries.len() == identity.stream_count as usize, "missing/duplicate stream boundary");
    let first = boundaries.first().context("empty stream population")?;
    for (id, boundary) in boundaries.iter().enumerate() {
        boundary.validate()?;
        ensure!(boundary.stream_id as usize == id, "stream boundary order/identity mismatch");
        ensure!(boundary.coupling_epoch == first.coupling_epoch,
            "streams are not at one synchronized coupling boundary");
        if crate::native_run_continuation::own_end_phase(&first.next_integrator_phase) {
            ensure!(identity.stream_count==8 && boundary.next_integrator_phase==first.next_integrator_phase,
                "own-end capsule requires its complete eight-worker phase population");
            // Exact logical/physical/startup/prefix joins are mandatory in
            // validate_own_end_publication; header arithmetic is insufficient.
        }else{
            ensure!(boundary.completed_step==first.completed_step && boundary.coupling_parity==first.coupling_parity && boundary.next_integrator_phase==first.next_integrator_phase,
                "ordinary streams are not at one physical/integrator boundary");
        }
        // Physical times may differ when native, governed dt differs between
        // streams. Their actual bit patterns remain independently committed.
    }
    Ok(())
}
fn validate_label(label: &str) -> Result<()> {
    ensure!(!label.is_empty() && label.len() <= 1024 && !label.starts_with('/') &&
        label.split('/').all(|p| !p.is_empty() && p != "." && p != ".." &&
            p.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-' || c == b'.')),
        "invalid checkpoint state label");
    Ok(())
}
fn validate_digest(digest: &str) -> Result<()> {
    ensure!(digest.len() == 64 && digest.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)),
        "invalid lowercase SHA-256 digest"); Ok(())
}
fn hash(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }
fn verify_directory_chain(path: &Path) -> Result<()> {
    ensure!(path.is_absolute(), "non-absolute checkpoint path");
    let mut current = PathBuf::new();
    for component in path.components() {
        ensure!(!matches!(component, std::path::Component::ParentDir), "parent traversal in checkpoint path");
        current.push(component);
        let metadata = fs::symlink_metadata(&current)?;
        ensure!(metadata.is_dir() && !metadata.file_type().is_symlink(), "checkpoint path is not a real directory");
    }
    Ok(())
}
fn sync_dir(path: &Path) -> Result<()> {
    let directory = OpenOptions::new().read(true).custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC).open(path)?;
    directory.sync_all()?; Ok(())
}
fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC).open(path)?;
    file.write_all(bytes)?; file.sync_all()?; Ok(())
}
fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let file = OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC).open(path)?;
    let metadata = file.metadata()?;
    ensure!(metadata.is_file() && metadata.len() <= limit, "checkpoint file type/size outside admission");
    let mut bytes = Vec::with_capacity(usize::try_from(metadata.len())?);
    file.take(limit.checked_add(1).context("read limit overflow")?).read_to_end(&mut bytes)?;
    ensure!(bytes.len() as u64 <= limit, "checkpoint grew beyond admitted size");
    Ok(bytes)
}

/// Host serialization uses an explicit typed codec, never Rust object memory.
/// Its float words preserve original bits, including nonfinite observer values.
pub fn capture_host<T: Serialize + ?Sized>(owner:&str,name:&str,value:&T)->Result<CapturedField> {
    Ok(CapturedField {contract:FieldContract{owner:owner.into(),name:name.into(),
        encoding:"prism.typed-host-state.v1".into(),expected_bytes:None,may_be_absent:false},
        bytes:Some(crate::checkpoint_codec::to_bytes(value)? )})
}
pub fn capture_host_bounded<T:Serialize+?Sized>(owner:&str,name:&str,value:&T,maximum:u64)->Result<CapturedField> {
    Ok(CapturedField{contract:FieldContract{owner:owner.into(),name:name.into(),encoding:"prism.typed-host-state.v1".into(),expected_bytes:None,may_be_absent:false},bytes:Some(crate::checkpoint_codec::to_bytes_bounded(value,maximum)?)})
}
pub fn restore_host<T:serde::de::DeserializeOwned>(field:&CapturedField)->Result<T> {
    ensure!(field.contract.encoding=="prism.typed-host-state.v1", "host-state encoding mismatch");
    let bytes=field.bytes.as_deref().context("required typed host state absent")?;
    Ok(crate::checkpoint_codec::from_bytes(bytes)?)
}

#[cfg(feature = "gpu")]
pub mod device {
    //! Typed byte-exact device transfers outside CUDA graph capture. The caller
    //! must hold the all-stream pause through capture/restore; synchronizing one
    //! stream does not establish cross-stream checkpoint closure.
    use super::*;
    use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr};
    use std::sync::Arc;
    mod sealed { pub trait Sealed {} }
    pub trait PlaneElement: sealed::Sealed + DeviceRepr + Copy + Default {
        const ENCODING: &'static str;
        const WIDTH: usize;
        fn append(self, out: &mut Vec<u8>);
        fn decode(bytes: &[u8]) -> Self;
    }
    macro_rules! integer {
        ($kind:ty,$tag:literal,$width:expr) => {
            impl sealed::Sealed for $kind {}
            impl PlaneElement for $kind {
                const ENCODING: &'static str=$tag; const WIDTH: usize=$width;
                fn append(self,out:&mut Vec<u8>) { out.extend_from_slice(&self.to_le_bytes()); }
                fn decode(bytes:&[u8])->Self { <$kind>::from_le_bytes(bytes.try_into().expect("validated plane width")) }
            }
        }
    }
    integer!(u16,"u16.le.v1",2); integer!(u8,"u8.le.v1",1); integer!(i32,"i32.le.v1",4);
    integer!(u32,"u32.le.v1",4); integer!(u64,"u64.le.v1",8);
    integer!(f32,"f32.ieee754.le.v1",4); integer!(f64,"f64.ieee754.le.v1",8);

    /// Caller proves allocation ownership/extent and freezes all producers.
    /// Used for existing native FFI allocations with explicit ABI layouts.
    pub unsafe fn read_raw(stream:&Arc<CudaStream>,pointer:u64,bytes:usize)->Result<Vec<u8>> {
        ensure!(pointer!=0 || bytes==0,"null native checkpoint allocation");
        stream.synchronize()?;
        let rc=unsafe{cudarc::driver::sys::cuCtxSetCurrent(stream.context().cu_ctx())};
        ensure!(rc==cudarc::driver::sys::CUresult::CUDA_SUCCESS,"CUDA checkpoint context: {rc:?}");
        let mut out=vec![0u8;bytes];
        if bytes>0 {
            let rc=unsafe{cudarc::driver::sys::cuMemcpyDtoH_v2(out.as_mut_ptr().cast(),pointer,bytes)};
            ensure!(rc==cudarc::driver::sys::CUresult::CUDA_SUCCESS,"CUDA checkpoint read: {rc:?}");
        }
        Ok(out)
    }
    /// Destination is an already rebuilt and verified allocation. Pointers in
    /// payloads must have been rebound by the owning ABI implementation.
    pub unsafe fn write_raw(stream:&Arc<CudaStream>,pointer:u64,bytes:&[u8])->Result<()> {
        ensure!(pointer!=0 || bytes.is_empty(),"null native restore allocation");
        stream.synchronize()?;
        let rc=unsafe{cudarc::driver::sys::cuCtxSetCurrent(stream.context().cu_ctx())};
        ensure!(rc==cudarc::driver::sys::CUresult::CUDA_SUCCESS,"CUDA restore context: {rc:?}");
        if !bytes.is_empty() {
            let rc=unsafe{cudarc::driver::sys::cuMemcpyHtoD_v2(pointer,bytes.as_ptr().cast(),bytes.len())};
            ensure!(rc==cudarc::driver::sys::CUresult::CUDA_SUCCESS,"CUDA checkpoint write: {rc:?}");
        }
        Ok(())
    }
    /// Byte-exact streaming of an existing sealed primitive CUDA plane. CUDA
    /// storage is little endian; this route explicitly refuses another host ABI
    /// rather than reinterpreting floating values. No full-plane host allocation.
    pub fn write_plane<T:PlaneElement>(stream:&Arc<CudaStream>,owner:&str,name:&str,
        plane:Option<&CudaSlice<T>>,may_be_absent:bool,out:&mut dyn Write)->Result<FieldContract> {
        use cudarc::driver::DevicePtr;
        ensure!(cfg!(target_endian="little") && std::mem::size_of::<T>()==T::WIDTH,
            "streamed native plane requires exact little-endian sealed primitive ABI");
        let expected=plane.map(|p|p.len().checked_mul(T::WIDTH).context("device extent overflow")).transpose()?;
        if let Some(plane)=plane {
            stream.synchronize()?;
            let rc=unsafe{cudarc::driver::sys::cuCtxSetCurrent(stream.context().cu_ctx())};
            ensure!(rc==cudarc::driver::sys::CUresult::CUDA_SUCCESS,"CUDA stream projection context: {rc:?}");
            let (pointer,guard)=plane.device_ptr(stream);
            let count=expected.context("present native plane lost its extent")?;
            // Same bounded four-MiB granularity as the scientific CAS leaf.
            // Preserve the owner thread/context, exact bytes and final tail.
            let slab_bytes=count.min(4*1024*1024).max(1);
            let mut slab=Vec::new();slab.try_reserve_exact(slab_bytes)?;slab.resize(slab_bytes,0u8);
            let projection_start=std::time::Instant::now();
            let mut transfer_ns=0u128;let mut downstream_ns=0u128;let mut transfers=0u64;
            for offset in (0..count).step_by(slab.len()) {
                let n=(count-offset).min(slab.len());
                let source=pointer.checked_add(offset as u64).context("device source pointer overflow")?;
                let transfer_start=std::time::Instant::now();
                let rc=unsafe{cudarc::driver::sys::cuMemcpyDtoH_v2(slab.as_mut_ptr().cast(),source,n)};
                transfer_ns+=transfer_start.elapsed().as_nanos();transfers+=1;
                ensure!(rc==cudarc::driver::sys::CUresult::CUDA_SUCCESS,"CUDA streamed checkpoint read: {rc:?}");
                let downstream_start=std::time::Instant::now();out.write_all(&slab[..n])?;downstream_ns+=downstream_start.elapsed().as_nanos();
            }
            log::info!("SCIENTIFIC_PLANE_PROJECTION owner={} field={} decoded_bytes={} slab_bytes={} transfers={} device_transfer_ns={} downstream_write_ns={} total_ns={}",owner,name,count,slab_bytes,transfers,transfer_ns,downstream_ns,projection_start.elapsed().as_nanos());
            drop(guard);
        }else{ensure!(may_be_absent,"required streamed device plane {owner}/{name} absent");}
        Ok(FieldContract{owner:owner.into(),name:name.into(),encoding:T::ENCODING.into(),
            expected_bytes:expected.map(|n|n as u64),may_be_absent})
    }
    pub fn capture<T:PlaneElement>(stream:&Arc<CudaStream>,owner:&str,name:&str,
        plane:Option<&CudaSlice<T>>, may_be_absent:bool)->Result<CapturedField> {
        let bytes=match plane {
            Some(plane)=>{
                let mut values=vec![T::default();plane.len()];
                stream.memcpy_dtoh(plane,&mut values)?;
                stream.synchronize()?;
                let mut bytes=Vec::with_capacity(values.len().checked_mul(T::WIDTH).context("plane byte overflow")?);
                for value in values { value.append(&mut bytes); }
                Some(bytes)
            },
            None=>{ ensure!(may_be_absent,"required device plane {owner}/{name} absent");None }
        };
        Ok(CapturedField { contract:FieldContract { owner:owner.into(),name:name.into(),
            encoding:T::ENCODING.into(),expected_bytes:bytes.as_ref().map(|v|v.len() as u64),may_be_absent },bytes })
    }
    pub fn validate<T:PlaneElement>(field:&CapturedField,plane:Option<&CudaSlice<T>>)->Result<()> {
        ensure!(field.contract.encoding==T::ENCODING,"device plane type/precision mismatch");
        match (&field.bytes,plane) {
            (Some(bytes),Some(plane))=>ensure!(bytes.len()==plane.len().checked_mul(T::WIDTH)
                .context("plane size overflow")?,"device plane allocation differs on restore"),
            (None,None)=>ensure!(field.contract.may_be_absent,"required device plane absent"),
            _=>bail!("device plane presence differs on restore"),
        }
        Ok(())
    }
    pub fn restore<T:PlaneElement>(stream:&Arc<CudaStream>,field:&CapturedField,
        plane:Option<&mut CudaSlice<T>>)->Result<()> {
        validate(field,plane.as_deref())?;
        if let (Some(bytes),Some(plane))=(&field.bytes,plane) {
            let values:Vec<T>=bytes.chunks_exact(T::WIDTH).map(T::decode).collect();
            stream.memcpy_htod(&values,plane)?;
            // CUDA can still reference the host temporary until completion.
            stream.synchronize()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(step:u64)->(CheckpointIdentity,ClosureContract,Vec<StreamBoundary>,Vec<CapturedField>) {
        let identity=CheckpointIdentity { run_id:"native-store-test".into(),topology_sha256:"1".repeat(64),
            engine_sha256:"2".repeat(64),kernel_set_sha256:"3".repeat(64),protocol_sha256:"4".repeat(64),
            execution_request_sha256:"5".repeat(64),state_abi_sha256:"6".repeat(64),provider:"test-storage-owner".into(),stream_count:2 };
        let field=FieldContract { owner:"stream_0/owner".into(),name:"state".into(),encoding:"u64.le.v1".into(),expected_bytes:Some(8),may_be_absent:false };
        let closure=ClosureContract { schema_version:SCHEMA,fields:vec![field.clone()],
            required_owners:BTreeSet::from([field.owner.clone()]),unresolved_owners:BTreeSet::new() };
        let boundaries=(0..2).map(|stream_id|StreamBoundary{stream_id,replica_epoch:1,replica_seed:42+stream_id as u64,
            completed_step:step,physical_time_ps_bits:(step as f64*0.002).to_bits(),next_integrator_phase:"begin_step".into(),
            coupling_epoch:step,coupling_parity:(step%2)as u8,committed_output_sequence:step}).collect();
        let fields=vec![CapturedField{contract:field,bytes:Some(step.to_le_bytes().to_vec())}];
        (identity,closure,boundaries,fields)
    }
    #[test]
    fn exact_population_and_stale_identity_refusals() {
        let dir=tempfile::tempdir().unwrap();let store=CheckpointStore::create(&dir.path().join("store"),1024).unwrap();
        let (id,c,b,f)=fixture(64);let head=store.commit(None,&id,&c,&b,&f).unwrap();
        let restored=store.restore(&head,&id,&c).unwrap();assert_eq!(restored.fields[0].bytes,f[0].bytes);
        assert!(store.commit(None,&id,&c,&b,&f).is_err());
        let mut bad=id.clone();bad.engine_sha256="7".repeat(64);assert!(store.restore(&head,&bad,&c).is_err());
        let mut missing=c.clone();missing.unresolved_owners.insert("governor".into());assert!(store.restore(&head,&id,&missing).is_err());
        let mut unaligned=b.clone();unaligned[1].completed_step+=1;assert!(store.commit(Some(&head),&id,&c,&unaligned,&f).is_err());
        assert!(store.commit(Some(&head),&id,&c,&b,&[]).is_err());
        let mut duplicate=f.clone();duplicate.push(f[0].clone());assert!(store.commit(Some(&head),&id,&c,&b,&duplicate).is_err());
        assert!(store.commit(Some(&head),&id,&c,&b,&f).is_err());
    }
    #[test]
    fn corrupt_payload_and_truncated_manifest_refused() {
        let dir=tempfile::tempdir().unwrap();let store=CheckpointStore::create(&dir.path().join("store"),1024).unwrap();
        let(id,c,b,f)=fixture(64);let h=store.commit(None,&id,&c,&b,&f).unwrap();
        let object=store.root.join("objects").join(hash(f[0].bytes.as_ref().unwrap()));fs::write(&object,b"corrupt!").unwrap();
        assert!(store.restore(&h,&id,&c).is_err());
        fs::write(store.root.join("manifests").join(&h.manifest_sha256),b"{").unwrap();assert!(store.restore(&h,&id,&c).is_err());
    }
    #[test]
    fn no_silent_budget_or_optional_allocation_changes() {
        let dir=tempfile::tempdir().unwrap();let store=CheckpointStore::create(&dir.path().join("store"),4).unwrap();
        let(id,c,b,f)=fixture(64);assert!(store.commit(None,&id,&c,&b,&f).is_err());assert!(store.head().unwrap().is_none());
        let mut absent=f.clone();absent[0].bytes=None;assert!(store.commit(None,&id,&c,&b,&absent).is_err());
    }
    #[test]
    fn rejects_symlink_root_and_pointer_file() {
        use std::os::unix::fs::symlink;
        let dir=tempfile::tempdir().unwrap();let target=dir.path().join("target");fs::create_dir(&target).unwrap();
        symlink(&target,dir.path().join("link")).unwrap();assert!(CheckpointStore::create(&dir.path().join("link"),1024).is_err());
        let store=CheckpointStore::create(&target.join("store"),1024).unwrap();
        symlink(dir.path().join("unrelated"),store.root.join("HEAD")).unwrap();assert!(store.head().is_err());
    }
    #[test]
    #[ignore = "child process entry; invoked only by process-crash acceptance"]
    fn checkpoint_child() {
        let root=PathBuf::from(std::env::var_os("PRISM_CHECKPOINT_TEST_ROOT").unwrap());
        let step=std::env::var("PRISM_CHECKPOINT_TEST_STEP").unwrap().parse().unwrap();
        let point=std::env::var("PRISM_CHECKPOINT_TEST_CRASH").unwrap_or_default();
        let store=CheckpointStore::create(&root,1024).unwrap();let (id,c,b,f)=fixture(step);
        let expected=std::env::var("PRISM_CHECKPOINT_TEST_EXPECTED").ok().map(|x|serde_json::from_str::<Head>(&x).unwrap());
        let result=store.commit_with_hook(expected.as_ref(),&id,&c,&b,&f,|phase|{
            if phase==point { unsafe {libc::_exit(73)} };Ok(())});
        match result { Ok(_)=>(), Err(e)=>{eprintln!("{e:#}");std::process::exit(74)} }
    }
    fn child(root:&Path,head:&Head,point:&str)->std::process::Command {
        let mut cmd=std::process::Command::new(std::env::current_exe().unwrap());
        cmd.arg("--exact").arg("scientific_checkpoint::tests::checkpoint_child").arg("--ignored").arg("--nocapture")
            .env("PRISM_CHECKPOINT_TEST_ROOT",root).env("PRISM_CHECKPOINT_TEST_STEP","128")
            .env("PRISM_CHECKPOINT_TEST_EXPECTED",serde_json::to_string(head).unwrap()).env("PRISM_CHECKPOINT_TEST_CRASH",point);
        cmd
    }
    #[test]
    fn actual_process_death_preserves_a_verified_commit_prefix() {
        for point in ["objects_synced","manifest_synced","head_staged_synced","head_published","head_directory_synced"] {
            let dir=tempfile::tempdir().unwrap();let store=CheckpointStore::create(&dir.path().join("store"),1024).unwrap();
            let(id,c,b,f)=fixture(64);let first=store.commit(None,&id,&c,&b,&f).unwrap();
            let status=child(&store.root,&first,point).status().unwrap();assert_eq!(status.code(),Some(73),"{point}");
            let reopened=CheckpointStore::create(&store.root,1024).unwrap();let current=reopened.head().unwrap().unwrap();
            let actual=reopened.restore(&current,&id,&c).unwrap();
            let expected=if ["head_published","head_directory_synced"].contains(&point){128u64}else{64u64};
            assert_eq!(actual.fields[0].bytes.as_ref().unwrap(),&expected.to_le_bytes());
            if expected==64 { let(_,_,b,f)=fixture(128);let next=reopened.commit(Some(&current),&id,&c,&b,&f).unwrap();
                assert_eq!(next.generation,2);assert!(reopened.restore(&next,&id,&c).is_ok()); }
        }
    }
    #[test]
    fn independent_writer_race_has_one_committed_generation() {
        let dir=tempfile::tempdir().unwrap();let store=CheckpointStore::create(&dir.path().join("store"),1024).unwrap();
        let(id,c,b,f)=fixture(64);let first=store.commit(None,&id,&c,&b,&f).unwrap();
        let mut a=child(&store.root,&first,"").spawn().unwrap();let mut b=child(&store.root,&first,"").spawn().unwrap();
        let codes=[a.wait().unwrap().code().unwrap(),b.wait().unwrap().code().unwrap()];
        assert!(codes==[0,74]||codes==[74,0],"{codes:?}");
        let current=store.head().unwrap().unwrap();assert_eq!(current.generation,2);assert!(store.restore(&current,&id,&c).is_ok());
    }
}

#[derive(Clone,Debug,Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AppendCursor {
    pub relative_path:String,
    pub committed_bytes:u64,
    pub committed_sha256:String,
}
/// Seal a buffered writer only after its owner has flushed the buffer. Hashing
/// uses positional reads and cannot change an active writer's seek position.
pub fn seal_append_file(file:&File,relative_path:&str)->Result<AppendCursor> {
    validate_label(relative_path)?;
    file.sync_all()?;
    let committed_bytes=file.metadata()?.len();
    let committed_sha256=hash_file_prefix(file,committed_bytes)?;
    Ok(AppendCursor{relative_path:relative_path.into(),committed_bytes,committed_sha256})
}
fn hash_file_prefix(file:&File,length:u64)->Result<String> {
    use std::os::unix::fs::FileExt;
    let mut hash=Sha256::new();let mut offset=0u64;let mut bytes=vec![0u8;1024*1024];
    while offset<length {
        let amount=usize::try_from((length-offset).min(bytes.len()as u64))?;
        let n=file.read_at(&mut bytes[..amount],offset)?;
        ensure!(n>0,"committed output prefix is truncated");hash.update(&bytes[..n]);offset+=n as u64;
    }
    Ok(format!("{:x}",hash.finalize()))
}
/// Open the committed prefix for append. Bytes produced after this checkpoint
/// are retained in a hash-bound recovery tail before truncation, so a retry
/// neither duplicates publication nor silently destroys observed evidence.
/// The run coordinator must hold its exclusive experiment lease throughout.
pub fn reopen_append_file(root:&Path,cursor:&AppendCursor)->Result<File> {
    use std::io::{Seek,SeekFrom};
    validate_label(&cursor.relative_path)?;validate_digest(&cursor.committed_sha256)?;
    verify_directory_chain(root)?;
    let path=root.join(&cursor.relative_path);verify_directory_chain(path.parent().context("output parent absent")?)?;
    let mut file=OpenOptions::new().read(true).write(true).custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC).open(&path)?;
    ensure!(file.metadata()?.is_file(),"append output is not a regular file");
    let size=file.metadata()?.len();ensure!(size>=cursor.committed_bytes,"committed append output is truncated");
    ensure!(hash_file_prefix(&file,cursor.committed_bytes)?==cursor.committed_sha256,"committed append prefix changed");
    if size>cursor.committed_bytes {
        let tail_bytes=size-cursor.committed_bytes;
        let mut tail_hash=Sha256::new();
        let mut tail_source=file.try_clone()?;tail_source.seek(SeekFrom::Start(cursor.committed_bytes))?;
        let mut tail_source=tail_source.take(tail_bytes);let mut tail_buffer=vec![0u8;1024*1024];
        loop {let n=tail_source.read(&mut tail_buffer)?;if n==0{break;}tail_hash.update(&tail_buffer[..n]);}
        let tail_sha256=format!("{:x}",tail_hash.finalize());
        let tail=path.with_file_name(format!("{}.uncommitted.{}.{}.{}",path.file_name().context("output filename absent")?.to_string_lossy(),
            cursor.committed_sha256,size,tail_sha256));
        let mut reader=file.try_clone()?;reader.seek(SeekFrom::Start(cursor.committed_bytes))?;
        if !tail.exists() {
            let staged=tail.with_extension("next");
            if staged.exists(){ensure!(fs::symlink_metadata(&staged)?.is_file(),"invalid recovery-tail staging file");fs::remove_file(&staged)?;}
            let mut output=OpenOptions::new().write(true).create_new(true).mode(0o600)
                .custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC).open(&staged)?;
            ensure!(std::io::copy(&mut reader.take(tail_bytes),&mut output)?==tail_bytes,"incomplete recovery-tail copy");
            output.sync_all()?;fs::rename(&staged,&tail)?;sync_dir(tail.parent().unwrap())?;
        }
        // Verify the retained tail against the still-present source before
        // modifying the active append file, including retries after a crash.
        let tail_file=OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC).open(&tail)?;
        ensure!(tail_file.metadata()?.len()==tail_bytes,"recovery-tail size differs");
        let mut source_hash=Sha256::new();
        let mut source=file.try_clone()?;source.seek(SeekFrom::Start(cursor.committed_bytes))?;
        let mut source=source.take(tail_bytes);let mut buffer=vec![0u8;1024*1024];
        loop {let n=source.read(&mut buffer)?;if n==0{break;}source_hash.update(&buffer[..n]);}
        ensure!(format!("{:x}",source_hash.finalize())==hash_file_prefix(&tail_file,tail_bytes)?,"retained recovery-tail differs from source");
        file.set_len(cursor.committed_bytes)?;file.sync_all()?;
    }
    file.seek(SeekFrom::Start(cursor.committed_bytes))?;
    Ok(file)
}

#[cfg(test)]
mod append_cursor_tests {
    use super::*;use std::io::{Seek,SeekFrom};
    #[test]fn rollback_preserves_actual_displaced_tail_and_retries(){
        let root=tempfile::tempdir().unwrap();let path=root.path().join("rows.bin");
        let mut writer=OpenOptions::new().read(true).write(true).create_new(true).open(&path).unwrap();
        writer.write_all(b"committed-native-records").unwrap();let cursor=seal_append_file(&writer,"rows.bin").unwrap();
        writer.write_all(b"uncommitted-after-checkpoint").unwrap();writer.sync_all().unwrap();drop(writer);
        let mut reopened=reopen_append_file(root.path(),&cursor).unwrap();assert_eq!(reopened.stream_position().unwrap(),cursor.committed_bytes);
        assert_eq!(fs::read(&path).unwrap(),b"committed-native-records");
        // Use the actual retained filename population, not an assumed tail size.
        let tails=fs::read_dir(root.path()).unwrap().map(|x|x.unwrap().path()).filter(|p|p.file_name().unwrap().to_string_lossy().contains(".uncommitted.")).collect::<Vec<_>>();
        assert_eq!(tails.len(),1);assert_eq!(fs::read(&tails[0]).unwrap(),b"uncommitted-after-checkpoint");
        reopened.write_all(b"continued").unwrap();reopened.sync_all().unwrap();drop(reopened);
        let mut second=reopen_append_file(root.path(),&cursor).unwrap();second.seek(SeekFrom::Start(0)).unwrap();let mut bytes=Vec::new();second.read_to_end(&mut bytes).unwrap();assert_eq!(bytes,b"committed-native-records");
    }
    #[test]fn corrupt_or_truncated_committed_prefix_is_never_repaired(){
        let root=tempfile::tempdir().unwrap();let path=root.path().join("rows.bin");let mut writer=OpenOptions::new().read(true).write(true).create_new(true).open(&path).unwrap();
        writer.write_all(b"actual committed bytes").unwrap();let cursor=seal_append_file(&writer,"rows.bin").unwrap();
        writer.seek(SeekFrom::Start(0)).unwrap();writer.write_all(b"X").unwrap();writer.sync_all().unwrap();assert!(reopen_append_file(root.path(),&cursor).is_err());
        writer.set_len(3).unwrap();assert!(reopen_append_file(root.path(),&cursor).is_err());
    }
}
