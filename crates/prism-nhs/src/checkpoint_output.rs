//! Evidence-directory boundaries bound by the native scientific checkpoint.
//! Call only while the run lease and all producer/consumer barriers are held.
//! Later output is retained before restoring the active writer namespace.
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::os::fd::AsRawFd;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Component, Path, PathBuf},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputFileBoundary {
    pub relative_path: PathBuf,
    pub bytes: u64,
    pub sha256: String,
    /// Only mutable small metadata is retained in the capsule. Scientific data
    /// remains in its original files and must still verify on reopen.
    pub metadata_bytes: Option<Vec<u8>>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputDirectoryBoundary {
    pub schema_version: u32,
    pub relative_root: PathBuf,
    pub files: Vec<OutputFileBoundary>,
}
fn relative(p: &Path) -> Result<()> {
    ensure!(
        !p.as_os_str().is_empty() && p.components().all(|c| matches!(c, Component::Normal(_))),
        "invalid checkpoint output relative path"
    );
    Ok(())
}
fn real_dir(p: &Path) -> Result<()> {
    let mut prefix = PathBuf::new();
    for c in p.components() {
        ensure!(
            !matches!(c, Component::ParentDir | Component::CurDir),
            "output path traversal"
        );
        prefix.push(c.as_os_str());
        let m = fs::symlink_metadata(&prefix)?;
        ensure!(
            m.is_dir() && !m.file_type().is_symlink(),
            "output ancestor is not a real directory"
        );
    }
    Ok(())
}
fn pin(p: &Path) -> Result<(u64, String)> {
    real_dir(p.parent().context("output file parent")?)?;
    let mut f = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(p)?;
    let before = f.metadata()?;
    ensure!(before.is_file(), "output is not a regular file");
    let mut h = Sha256::new();
    let mut n = 0u64;
    let mut b = vec![0; 1024 * 1024];
    loop {
        let k = f.read(&mut b)?;
        if k == 0 {
            break;
        }
        h.update(&b[..k]);
        n += k as u64;
    }
    ensure!(
        n == before.len() && f.metadata()?.len() == n,
        "output changed during boundary capture"
    );
    Ok((n, format!("{:x}", h.finalize())))
}
fn files(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    real_dir(dir)?;
    for item in fs::read_dir(dir)? {
        let path = item?.path();
        let rel = path.strip_prefix(root)?.to_path_buf();
        if rel
            .components()
            .next()
            .is_some_and(|c| c.as_os_str() == ".continuation_tails")
        {
            continue;
        }
        let m = fs::symlink_metadata(&path)?;
        ensure!(!m.file_type().is_symlink(), "symlink in output boundary");
        if m.is_dir() {
            files(root, &path, out)?;
        } else {
            ensure!(m.is_file(), "special output file");
            out.push(rel);
        }
    }
    Ok(())
}
/// Capture a recursively complete directory, including immutable index journals.
/// Mutable metadata names must be explicitly named by their owning writer.
pub fn capture_directory(
    run_root: &Path,
    root: &Path,
    metadata: &[&str],
) -> Result<OutputDirectoryBoundary> {
    real_dir(run_root)?;
    real_dir(root)?;
    let relative_root = root.strip_prefix(run_root)?.to_path_buf();
    relative(&relative_root)?;
    let mutable: BTreeSet<PathBuf> = metadata.iter().map(PathBuf::from).collect();
    for p in &mutable {
        relative(p)?;
    }
    let mut paths = Vec::new();
    files(root, root, &mut paths)?;
    paths.sort();
    let mut result = Vec::new();
    for p in paths {
        let path = root.join(&p);
        File::open(&path)?.sync_all()?;
        let (bytes, sha256) = pin(&path)?;
        let metadata_bytes = if mutable.contains(&p) {
            let b = fs::read(&path)?;
            ensure!(
                b.len() as u64 == bytes && format!("{:x}", Sha256::digest(&b)) == sha256,
                "metadata changed during capture"
            );
            Some(b)
        } else {
            None
        };
        result.push(OutputFileBoundary {
            relative_path: p,
            bytes,
            sha256,
            metadata_bytes,
        });
    }
    ensure!(
        mutable
            .iter()
            .all(|p| result.iter().any(|r| &r.relative_path == p)),
        "checkpoint metadata absent"
    );
    File::open(root)?.sync_all()?;
    Ok(OutputDirectoryBoundary {
        schema_version: 1,
        relative_root,
        files: result,
    })
}
/// Serialize only output-retention bookkeeping. This is separate from the
/// scientific ROOT lock and grants no producer/checkpoint mutation authority.
/// The caller still holds the native run lease and its writer/pause barriers.
struct RetentionTransaction {
    root: PathBuf,
    history: PathBuf,
    directory: File,
    lock: File,
}
impl RetentionTransaction {
    fn acquire(root: &Path) -> Result<Self> {
        real_dir(root)?;
        let history = root.join(".continuation_tails");
        match fs::create_dir(&history) {
            Ok(()) => {},
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {},
            Err(e) => return Err(e).with_context(||format!("create output-retention directory {}",history.display())),
        }
        // Another restoring stream may have created the name. Accept it only
        // after full real-directory validation, then durably publish that name.
        real_dir(&history)?;
        let directory = OpenOptions::new().read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC).open(&history)?;
        File::open(root)?.sync_all()?;
        let lock_path = history.join("LOCK");
        let lock = OpenOptions::new().read(true).write(true).create(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC).open(&lock_path)?;
        ensure!(lock.metadata()?.is_file() && lock.metadata()?.nlink()==1,
            "output-retention lock is not a private regular inode");
        loop {
            if unsafe {libc::flock(lock.as_raw_fd(),libc::LOCK_EX)} == 0 {break;}
            let e=std::io::Error::last_os_error();
            if e.kind()!=std::io::ErrorKind::Interrupted {return Err(e).context("lock output-retention transaction");}
        }
        let transaction=Self{root:root.to_path_buf(),history,directory,lock};
        transaction.validate()?;
        transaction.lock.sync_all()?;
        transaction.directory.sync_all()?;
        Ok(transaction)
    }
    fn validate(&self)->Result<()> {
        real_dir(&self.root)?;real_dir(&self.history)?;
        let held=self.directory.metadata()?;let named=fs::symlink_metadata(&self.history)?;
        ensure!(held.is_dir() && named.is_dir() && !named.file_type().is_symlink()
            && held.dev()==named.dev() && held.ino()==named.ino(),"output-retention directory inode changed");
        let held=self.lock.metadata()?;let named=fs::symlink_metadata(self.history.join("LOCK"))?;
        ensure!(held.is_file() && named.is_file() && !named.file_type().is_symlink()
            && held.dev()==named.dev() && held.ino()==named.ino() && held.nlink()==1 && named.nlink()==1,
            "output-retention lock inode changed");
        Ok(())
    }
}
impl Drop for RetentionTransaction {
    fn drop(&mut self) {unsafe{libc::flock(self.lock.as_raw_fd(),libc::LOCK_UN);}}
}
fn same_source_inode(path:&Path,expected:&fs::Metadata)->Result<()> {
    let actual=fs::symlink_metadata(path)?;
    ensure!(actual.is_file() && !actual.file_type().is_symlink()
        && actual.dev()==expected.dev() && actual.ino()==expected.ino(),
        "retained source locator now names another inode: {}",path.display());Ok(())
}
/// A stage is fsynced before its no-clobber link becomes visible. Existing
/// receipt/stage names must match exact bytes; an incomplete crash stage is
/// retained and rejected, never overwritten or interpreted as a valid receipt.
fn publish_retention_bytes(transaction:&RetentionTransaction,pending_name:&str,
    path:&Path,bytes:&[u8],allow_exact_existing:bool)->Result<()> {
    transaction.validate()?;
    let hash=format!("{:x}",Sha256::digest(bytes));
    let pending=transaction.history.join(pending_name);
    real_dir(path.parent().context("output-retention target parent")?)?;
    let stage=match OpenOptions::new().read(true).write(true).create_new(true)
        .custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC).open(&pending) {
        Ok(mut file)=>{file.write_all(bytes)?;file.sync_all()?;file},
        Err(e)if e.kind()==std::io::ErrorKind::AlreadyExists=>{
            ensure!(pin(&pending)?==(bytes.len()as u64,hash.clone()),"output-retention pending bytes differ: {}",pending.display());
            let file=OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC).open(&pending)?;
            file.sync_all()?;file
        },Err(e)=>return Err(e).with_context(||format!("create output-retention pending {}",pending.display())),
    };
    // Verify the entire stage through the no-follow hash path before publication.
    ensure!(pin(&pending)?==(bytes.len()as u64,hash.clone()),"output-retention stage differs");
    let expected=stage.metadata()?;
    same_source_inode(&pending,&expected)?;
    match fs::hard_link(&pending,path) {
        Ok(())=>{},
        Err(e)if allow_exact_existing && e.kind()==std::io::ErrorKind::AlreadyExists=>{
            ensure!(pin(path)?==(bytes.len()as u64,hash.clone()),"output-retention receipt collision differs: {}",path.display());
            let mut file=OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC).open(path)?;
            ensure!(file.metadata()?.len()==bytes.len()as u64,"output-retention receipt extent differs");
            let mut existing=Vec::with_capacity(bytes.len());file.read_to_end(&mut existing)?;
            ensure!(existing==bytes,"output-retention receipt exact bytes differ");file.sync_all()?;
        },Err(e)=>return Err(e).with_context(||format!("no-clobber output-retention publication {}",path.display())),
    }
    ensure!(pin(path)?==(bytes.len()as u64,hash),"published output-retention bytes differ");
    OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC).open(path)?.sync_all()?;
    File::open(path.parent().context("output-retention target parent")?)?.sync_all()?;
    transaction.validate()?;same_source_inode(&pending,&expected)?;
    fs::remove_file(&pending)?;transaction.directory.sync_all()?;Ok(())
}
fn archive_later_file_locked(transaction:&RetentionTransaction,path:&Path)->Result<(u64,String,fs::Metadata)> {
    transaction.validate()?;
    let rel=path.strip_prefix(&transaction.root)?;relative(rel)?;
    ensure!(!rel.components().next().is_some_and(|c|c.as_os_str()==".continuation_tails"),
        "retention bookkeeping cannot be archived into itself");
    let source=fs::symlink_metadata(path)?;
    ensure!(source.is_file() && !source.file_type().is_symlink(),"retained source is not a regular file");
    let (bytes,sha)=pin(path)?;same_source_inode(path,&source)?;
    let object=transaction.history.join(format!("{sha}.bytes"));
    match fs::hard_link(path,&object) {
        Ok(())=>{},Err(e)if e.kind()==std::io::ErrorKind::AlreadyExists=>{},
        Err(e)=>return Err(e).with_context(||format!("retain output body {}",path.display())),
    }
    // An existing hash name has no authority until its complete body verifies.
    ensure!(pin(&object)?==(bytes,sha.clone()),"retained output tail differs");
    OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC).open(&object)?.sync_all()?;transaction.directory.sync_all()?;
    let record=serde_json::to_vec(&serde_json::json!({"source_relative_path":rel,"bytes":bytes,"sha256":sha}))?;
    let name=format!("{:x}",Sha256::digest(&record));
    publish_retention_bytes(transaction,&format!(".record-{name}.pending"),
        &transaction.history.join(format!("{name}.json")),&record,true)?;
    // Preserve byte and inode identity before any caller can unlink the active
    // source. This check also covers pre-existing deduplicated object inodes.
    ensure!(pin(path)?==(bytes,sha.clone()),"source changed while retaining output tail");
    same_source_inode(path,&source)?;transaction.validate()?;Ok((bytes,sha,source))
}
/// Retain an unwanted later file in a content-addressed history, including its
/// original relative locator. The whole shared transaction is serialized.
pub fn archive_later_file(root:&Path,path:&Path)->Result<()> {
    let transaction=RetentionTransaction::acquire(root)?;
    archive_later_file_locked(&transaction,path).with_context(||format!("archive later native output {}",path.display()))?;Ok(())
}
pub fn retain_later_file(root:&Path,path:&Path)->Result<()> {
    let transaction=RetentionTransaction::acquire(root)?;
    let (_,_,source)=archive_later_file_locked(&transaction,path)
        .with_context(||format!("retain later native output {}",path.display()))?;
    same_source_inode(path,&source)?;fs::remove_file(path)?;
    File::open(path.parent().context("retained source parent")?)?.sync_all()?;Ok(())
}
fn replace_metadata(root:&Path,path:&Path,bytes:&[u8])->Result<()> {
    // Share the retention lock through the complete stage/link/unlink lifecycle.
    // Target identity is part of the pending name; equal bytes at distinct
    // metadata owners never contend for or remove each other's pending file.
    let transaction=RetentionTransaction::acquire(root)?;
    let rel=path.strip_prefix(root)?;relative(rel)?;
    let owner=format!("{:x}",Sha256::digest(rel.as_os_str().as_encoded_bytes()));
    let hash=format!("{:x}",Sha256::digest(bytes));
    publish_retention_bytes(&transaction,&format!(".restore-{owner}-{hash}.pending"),path,bytes,false)
}
pub fn restore_directory(run_root: &Path, boundary: &OutputDirectoryBoundary) -> Result<PathBuf> {
    ensure!(boundary.schema_version == 1, "unsupported output boundary");
    relative(&boundary.relative_root)?;
    real_dir(run_root)?;
    let root = run_root.join(&boundary.relative_root);
    real_dir(&root)?;
    let mut expected = BTreeMap::new();
    for file in &boundary.files {
        relative(&file.relative_path)?;
        ensure!(
            expected.insert(file.relative_path.clone(), file).is_none(),
            "duplicate output boundary member"
        );
        if let Some(bytes) = &file.metadata_bytes {
            ensure!(
                bytes.len() as u64 == file.bytes
                    && format!("{:x}", Sha256::digest(bytes)) == file.sha256,
                "checkpoint metadata identity differs"
            );
        } else {
            ensure!(
                pin(&root.join(&file.relative_path))? == (file.bytes, file.sha256.clone()),
                "committed scientific output changed or missing"
            );
        }
    }
    let mut current = Vec::new();
    files(&root, &root, &mut current)?;
    for rel in current {
        let path = root.join(&rel);
        match expected.get(&rel) {
            Some(bound) if pin(&path)? == (bound.bytes, bound.sha256.clone()) => {}
            Some(bound) => {
                ensure!(bound.metadata_bytes.is_some(), "immutable output changed");
                retain_later_file(&root, &path)?;
            }
            None => retain_later_file(&root, &path)?,
        }
    }
    for (rel, bound) in expected {
        let path = root.join(rel);
        if !path.exists() {
            let b = bound
                .metadata_bytes
                .as_ref()
                .context("committed output missing")?;
            replace_metadata(&root, &path, b)?;
        }
        ensure!(
            pin(&path)? == (bound.bytes, bound.sha256.clone()),
            "restored output differs"
        );
    }
    File::open(&root)?.sync_all()?;
    Ok(root)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn directory_reopen_retains_tail_and_rejects_committed_corruption() -> Result<()> {
        let t = tempfile::tempdir()?;
        let r = t.path().join("writer");
        fs::create_dir(&r)?;
        fs::write(r.join("manifest.json"), b"old")?;
        fs::write(r.join("segment_0"), b"measured")?;
        let b = capture_directory(t.path(), &r, &["manifest.json"])?;
        fs::write(r.join("manifest.json"), b"later")?;
        fs::write(r.join("segment_1"), b"later measured")?;
        restore_directory(t.path(), &b)?;
        ensure!(
            fs::read(r.join("manifest.json"))? == b"old",
            "metadata restoration"
        );
        ensure!(!r.join("segment_1").exists(), "tail stayed active");
        ensure!(r.join(".continuation_tails").is_dir(), "tail missing");
        restore_directory(t.path(), &b)?;
        fs::write(r.join("segment_0"), b"corrupt")?;
        ensure!(
            restore_directory(t.path(), &b).is_err(),
            "corruption accepted"
        );
        Ok(())
    }
}

/// Read a small regular metadata file without following path or file symlinks.
pub fn read_metadata(path: &Path) -> Result<Vec<u8>> {
    real_dir(path.parent().context("metadata parent")?)?;
    let mut f = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)?;
    ensure!(
        f.metadata()?.is_file() && f.metadata()?.len() <= 16 * 1024 * 1024,
        "invalid metadata file"
    );
    let mut bytes = Vec::new();
    f.read_to_end(&mut bytes)?;
    Ok(bytes)
}
/// The V2 trajectory count header is patched only at final closeout. Restore
/// its checkpoint header separately from its immutable measured payload prefix.
/// Full displaced bytes are archived before the atomic active-file replacement.
pub fn reopen_append_with_header(
    run_root: &Path,
    cursor: &crate::scientific_checkpoint::AppendCursor,
    header: &[u8],
) -> Result<File> {
    use std::io::{Seek, SeekFrom};
    relative(Path::new(&cursor.relative_path))?;
    real_dir(run_root)?;
    let path = run_root.join(&cursor.relative_path);
    real_dir(path.parent().context("append parent")?)?;
    let mut source = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(&path)?;
    ensure!(
        source.metadata()?.is_file()
            && source.metadata()?.len() >= cursor.committed_bytes
            && cursor.committed_bytes >= header.len() as u64,
        "invalid mutable-header output length"
    );
    let mut current = vec![0; header.len()];
    source.read_exact(&mut current)?;
    if current == header {
        return crate::scientific_checkpoint::reopen_append_file(run_root, cursor);
    }
    let mut digest = Sha256::new();
    digest.update(header);
    let mut remaining = cursor.committed_bytes - header.len() as u64;
    let mut buffer = vec![0u8; 1024 * 1024];
    while remaining > 0 {
        let n = usize::try_from(remaining.min(buffer.len() as u64))?;
        source.read_exact(&mut buffer[..n])?;
        digest.update(&buffer[..n]);
        remaining -= n as u64;
    }
    ensure!(
        format!("{:x}", digest.finalize()) == cursor.committed_sha256,
        "mutable-header normalized prefix differs from checkpoint"
    );
    // Distinct writer locators with identical payload hashes still own
    // distinct stages; a peer must never retain/remove our in-progress stage.
    let owner=format!("{:x}",Sha256::digest(cursor.relative_path.as_bytes()));
    let staged = path.with_file_name(format!(
        ".restore-prefix-{}-{}.pending",
        owner,cursor.committed_sha256
    ));
    if staged.exists() {
        retain_later_file(run_root, &staged)?;
    }
    let mut output = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(&staged)?;
    output.write_all(header)?;
    source.seek(SeekFrom::Start(header.len() as u64))?;
    ensure!(
        std::io::copy(
            &mut source.take(cursor.committed_bytes - header.len() as u64),
            &mut output
        )? == cursor.committed_bytes - header.len() as u64,
        "incomplete restored prefix"
    );
    output.sync_all()?;
    ensure!(
        pin(&staged)? == (cursor.committed_bytes, cursor.committed_sha256.clone()),
        "restored staged prefix mismatch"
    );
    archive_later_file(run_root, &path)?;
    fs::rename(staged, &path)?;
    File::open(path.parent().unwrap())?.sync_all()?;
    crate::scientific_checkpoint::reopen_append_file(run_root, cursor)
}

/// Relocate a complete native output boundary into a disjoint historical branch.
/// Scientific bytes are independently copied and hash-verified. Mutable metadata
/// comes from its committed exact image, even if the source advanced later.
impl OutputDirectoryBoundary {
    pub fn branch_copy_allocation_bound(&self)->Result<u64> {
        ensure!(self.schema_version==1,"unknown output allocation contract");
        self.files.iter().try_fold(0u64,|sum,item|sum.checked_add(crate::scientific_checkpoint::restart_transaction::branch_copy_member_bound(item.bytes)?).context("output directory branch bound overflow"))
    }
    pub fn fork_outputs(&self,source:&Path,destination:&Path)->Result<()> {
        ensure!(self.schema_version==1,"unsupported historical output boundary");
        relative(&self.relative_root)?;
        let mut members=std::collections::BTreeSet::new();
        for item in &self.files {
            relative(&item.relative_path)?;
            ensure!(members.insert(item.relative_path.clone()),"duplicate historical output member");
            let relative_path=self.relative_root.join(&item.relative_path);
            if let Some(bytes)=&item.metadata_bytes {
                ensure!(bytes.len() as u64==item.bytes && format!("{:x}",Sha256::digest(bytes))==item.sha256,
                    "saved historical metadata identity differs");
                crate::scientific_checkpoint::restart_transaction::publish_branch_metadata(destination,&relative_path,bytes)?;
            }else{
                crate::scientific_checkpoint::restart_transaction::copy_committed_append_prefix(source,destination,
                    &crate::scientific_checkpoint::AppendCursor{relative_path:relative_path.to_str().context("nonUTF8 output branch member")?.into(),
                        committed_bytes:item.bytes,committed_sha256:item.sha256.clone()})?;
            }
        }
        restore_directory(destination,self)?;Ok(())
    }
}
