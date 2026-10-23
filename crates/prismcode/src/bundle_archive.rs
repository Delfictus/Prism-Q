use crate::binary_runtime::{PcError, PcResult};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};

const MAGIC: &[u8] = b"PRISM_BUNDLE_ARCHIVE_V0_1\n";

pub fn unpack_archive(archive: &Path, destination: &Path) -> PcResult<Value> {
    if destination.exists() {
        fs::remove_dir_all(destination)?;
    }
    fs::create_dir_all(destination)?;
    let dest = fs::canonicalize(destination)?;
    let file = fs::File::open(archive)?;
    let mut decoder = zstd::Decoder::new(file)
        .map_err(|err| PcError::Format(format!("archive zstd decode failed: {err}")))?;
    let mut magic = vec![0u8; MAGIC.len()];
    decoder.read_exact(&mut magic)?;
    if magic != MAGIC {
        return Err(PcError::Format(
            "archive magic mismatch; expected PRISM bundle archive".to_string(),
        ));
    }
    let count = read_u64_from(&mut decoder)?;
    let mut out = Vec::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    for _ in 0..count {
        let path_len = read_u32_from(&mut decoder)? as usize;
        let mut path = vec![0u8; path_len];
        decoder.read_exact(&mut path)?;
        let rel_text = String::from_utf8(path)
            .map_err(|err| PcError::Format(format!("archive path is not UTF-8: {err}")))?;
        let mode = read_u32_from(&mut decoder)?;
        let len = read_u64_from(&mut decoder)?;
        let rel = sanitize_rel_path(&rel_text)?;
        let path = dest.join(&rel);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut output = fs::File::create(&path)?;
        let mut remaining = len;
        let mut hasher = blake3::Hasher::new();
        while remaining > 0 {
            let take = remaining.min(buffer.len() as u64) as usize;
            decoder.read_exact(&mut buffer[..take])?;
            output.write_all(&buffer[..take])?;
            hasher.update(&buffer[..take]);
            remaining -= take as u64;
        }
        output.flush()?;
        set_mode(&path, mode)?;
        let real = fs::canonicalize(&path)?;
        if !real.starts_with(&dest) {
            return Err(PcError::Security(format!(
                "PC_ERR_HERMETICITY_PATH_ESCAPE:{}",
                real.display()
            )));
        }
        out.push(json!({
            "path": rel.display().to_string(),
            "byte_size": len,
            "hash_blake3": hasher.finalize().to_hex().to_string(),
        }));
    }
    Ok(json!({
        "status": "PASS",
        "archive": archive.display().to_string(),
        "destination": dest.display().to_string(),
        "entry_count": out.len(),
        "entries": out,
    }))
}

pub fn sha256_file_hex(path: &Path) -> PcResult<String> {
    let mut hasher = Sha256::new();
    let mut file = fs::File::open(path)?;
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn verify_sha256_sidecar(archive: &Path, sidecar: &Path) -> PcResult<Value> {
    let expected_text = fs::read_to_string(sidecar)?;
    let expected = expected_text
        .split_whitespace()
        .next()
        .ok_or_else(|| PcError::Format(format!("empty SHA-256 sidecar: {}", sidecar.display())))?;
    let actual = sha256_file_hex(archive)?;
    if expected != actual {
        return Err(PcError::Hash(format!(
            "archive SHA-256 mismatch: expected {expected} actual {actual}"
        )));
    }
    Ok(json!({
        "status": "PASS",
        "archive": archive.display().to_string(),
        "sidecar": sidecar.display().to_string(),
        "sha256": actual,
    }))
}

fn sanitize_rel_path(rel: &str) -> PcResult<PathBuf> {
    let path = PathBuf::from(rel);
    if path.is_absolute() || rel.contains("..") {
        return Err(PcError::Security(format!(
            "PC_ERR_HERMETICITY_PATH_ESCAPE:{rel}"
        )));
    }
    Ok(path)
}

fn read_u32_from<R: Read>(reader: &mut R) -> PcResult<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64_from<R: Read>(reader: &mut R) -> PcResult<u64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

pub fn encode_entries_for_test(entries: &[(String, Vec<u8>)]) -> PcResult<Vec<u8>> {
    let mut raw = Vec::new();
    raw.write_all(MAGIC)?;
    raw.write_all(&(entries.len() as u64).to_le_bytes())?;
    for (rel, bytes) in entries {
        raw.write_all(&(rel.len() as u32).to_le_bytes())?;
        raw.write_all(rel.as_bytes())?;
        raw.write_all(&0o644u32.to_le_bytes())?;
        raw.write_all(&(bytes.len() as u64).to_le_bytes())?;
        raw.write_all(bytes)?;
    }
    zstd::stream::encode_all(Cursor::new(raw), 10)
        .map_err(|err| PcError::Format(format!("archive zstd encode failed: {err}")))
}

#[cfg(unix)]
fn set_mode(path: &Path, mode: u32) -> PcResult<()> {
    use std::os::unix::fs::PermissionsExt;
    let mut perms = fs::metadata(path)?.permissions();
    perms.set_mode(mode);
    fs::set_permissions(path, perms)?;
    Ok(())
}

#[cfg(not(unix))]
fn set_mode(_path: &Path, _mode: u32) -> PcResult<()> {
    Ok(())
}
