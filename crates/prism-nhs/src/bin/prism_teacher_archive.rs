//! Archive executor for PRISM Twin teacher consensus bundles.
//!
//! This consumes `teacher_archive_manifest.json`, validates local object size
//! and SHA-256, and optionally uploads each object to Cloudflare R2 through
//! rclone.  Dry-run is the default so production can inventory before moving
//! bytes.

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    process::Command,
};

#[derive(Parser, Debug)]
#[command(name = "prism-teacher-archive")]
#[command(about = "Validate and optionally upload PRISM Twin teacher archive manifests")]
struct Args {
    #[arg(long)]
    manifest: PathBuf,

    /// Actually upload objects with rclone copyto. Default is validation-only dry-run.
    #[arg(long, default_value_t = false)]
    execute: bool,

    #[arg(long, default_value = "rclone")]
    rclone_bin: String,

    #[arg(long, default_value = "r2")]
    rclone_remote: String,
}

#[derive(Debug, Deserialize)]
struct ArchiveManifest {
    schema_kind: String,
    schema_version: String,
    archive_profile: String,
    bucket: String,
    r2_prefix: String,
    target_id: String,
    n_replicas: usize,
    objects: Vec<ArchiveObject>,
}

#[derive(Debug, Deserialize)]
struct ArchiveObject {
    kind: String,
    relative_path: String,
    object_key: String,
    local_path: String,
    sha256: String,
    size_bytes: u64,
}

#[derive(Debug, Serialize)]
struct ArchiveReport {
    schema_kind: &'static str,
    manifest_path: String,
    archive_profile: String,
    bucket: String,
    r2_prefix: String,
    target_id: String,
    n_replicas: usize,
    execute: bool,
    object_count: usize,
    uploaded_count: usize,
    objects: Vec<ObjectReport>,
}

#[derive(Debug, Serialize)]
struct ObjectReport {
    kind: String,
    relative_path: String,
    object_key: String,
    destination: String,
    size_bytes: u64,
    sha256: String,
    local_validated: bool,
    uploaded: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let manifest = read_manifest(&args.manifest)?;
    validate_manifest(&manifest)?;

    let mut objects = Vec::new();
    let mut uploaded_count = 0usize;
    for object in &manifest.objects {
        validate_object(&manifest, object)?;
        let destination = rclone_destination(&args.rclone_remote, &manifest.bucket, object);
        let uploaded = if args.execute {
            upload_object(&args.rclone_bin, object, &destination)?;
            uploaded_count += 1;
            true
        } else {
            false
        };
        objects.push(ObjectReport {
            kind: object.kind.clone(),
            relative_path: object.relative_path.clone(),
            object_key: object.object_key.clone(),
            destination,
            size_bytes: object.size_bytes,
            sha256: object.sha256.clone(),
            local_validated: true,
            uploaded,
        });
    }

    let report = ArchiveReport {
        schema_kind: "prism_twin_teacher_archive_report",
        manifest_path: args.manifest.to_string_lossy().into_owned(),
        archive_profile: manifest.archive_profile,
        bucket: manifest.bucket,
        r2_prefix: manifest.r2_prefix,
        target_id: manifest.target_id,
        n_replicas: manifest.n_replicas,
        execute: args.execute,
        object_count: objects.len(),
        uploaded_count,
        objects,
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn read_manifest(path: &Path) -> Result<ArchiveManifest> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
}

fn validate_manifest(manifest: &ArchiveManifest) -> Result<()> {
    if manifest.schema_kind != "prism_twin_teacher_archive_manifest" {
        bail!(
            "unsupported archive schema_kind {}; expected prism_twin_teacher_archive_manifest",
            manifest.schema_kind
        );
    }
    if manifest.schema_version != "1.0.0" {
        bail!(
            "unsupported archive schema_version {}",
            manifest.schema_version
        );
    }
    if manifest.archive_profile != "prism_twin_teacher_v1" {
        bail!(
            "unsupported archive_profile {}; expected prism_twin_teacher_v1",
            manifest.archive_profile
        );
    }
    if manifest.bucket != "prism-archive" {
        bail!(
            "teacher archive manifest must target prism-archive, got {}",
            manifest.bucket
        );
    }
    if manifest.objects.is_empty() {
        bail!("archive manifest contains no objects");
    }
    Ok(())
}

fn validate_object(manifest: &ArchiveManifest, object: &ArchiveObject) -> Result<()> {
    if !object
        .object_key
        .starts_with(manifest.r2_prefix.trim_matches('/'))
    {
        bail!(
            "object {} key {} is outside manifest prefix {}",
            object.kind,
            object.object_key,
            manifest.r2_prefix
        );
    }
    let path = Path::new(&object.local_path);
    let meta = path
        .metadata()
        .with_context(|| format!("missing local archive object {}", path.display()))?;
    if !meta.is_file() {
        bail!("{} is not a file", path.display());
    }
    if meta.len() != object.size_bytes {
        bail!(
            "{} size {} != manifest size {}",
            path.display(),
            meta.len(),
            object.size_bytes
        );
    }
    let observed = sha256_file(path)?;
    if observed != object.sha256.to_ascii_lowercase() {
        bail!(
            "{} sha256 {} != manifest sha256 {}",
            path.display(),
            observed,
            object.sha256
        );
    }
    Ok(())
}

fn rclone_destination(remote: &str, bucket: &str, object: &ArchiveObject) -> String {
    format!(
        "{}:{}/{}",
        remote.trim_end_matches(':'),
        bucket.trim_matches('/'),
        object.object_key.trim_start_matches('/')
    )
}

fn upload_object(rclone_bin: &str, object: &ArchiveObject, destination: &str) -> Result<()> {
    let status = Command::new(rclone_bin)
        .arg("copyto")
        .arg(&object.local_path)
        .arg(destination)
        .status()
        .with_context(|| format!("spawn {rclone_bin} copyto {}", destination))?;
    if !status.success() {
        bail!(
            "rclone copyto failed for {} -> {} with status {:?}",
            object.local_path,
            destination,
            status.code()
        );
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1024 * 1024];
    loop {
        let n = file
            .read(&mut buf)
            .with_context(|| format!("read {}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_manifest_object_hash_and_destination() {
        let dir = tempfile::tempdir().unwrap();
        let object_path = dir.path().join("bundle.parquet");
        std::fs::write(&object_path, b"teacher-bundle").unwrap();
        let sha = sha256_file(&object_path).unwrap();
        let object = ArchiveObject {
            kind: "student_label_bundle".to_string(),
            relative_path: "stage_c/student_label_bundle.parquet".to_string(),
            object_key: "teacher-consensus/t/n5/stage_c/student_label_bundle.parquet".to_string(),
            local_path: object_path.to_string_lossy().into_owned(),
            sha256: sha,
            size_bytes: object_path.metadata().unwrap().len(),
        };
        let manifest = ArchiveManifest {
            schema_kind: "prism_twin_teacher_archive_manifest".to_string(),
            schema_version: "1.0.0".to_string(),
            archive_profile: "prism_twin_teacher_v1".to_string(),
            bucket: "prism-archive".to_string(),
            r2_prefix: "teacher-consensus/t/n5".to_string(),
            target_id: "t".to_string(),
            n_replicas: 5,
            objects: vec![object],
        };
        validate_manifest(&manifest).unwrap();
        validate_object(&manifest, &manifest.objects[0]).unwrap();
        assert_eq!(
            rclone_destination("r2", &manifest.bucket, &manifest.objects[0]),
            "r2:prism-archive/teacher-consensus/t/n5/stage_c/student_label_bundle.parquet"
        );
    }

    #[test]
    fn rejects_objects_outside_archive_prefix() {
        let dir = tempfile::tempdir().unwrap();
        let object_path = dir.path().join("artifact.json");
        std::fs::write(&object_path, b"artifact").unwrap();
        let manifest = ArchiveManifest {
            schema_kind: "prism_twin_teacher_archive_manifest".to_string(),
            schema_version: "1.0.0".to_string(),
            archive_profile: "prism_twin_teacher_v1".to_string(),
            bucket: "prism-archive".to_string(),
            r2_prefix: "teacher-consensus/t/n5".to_string(),
            target_id: "t".to_string(),
            n_replicas: 5,
            objects: vec![ArchiveObject {
                kind: "bad".to_string(),
                relative_path: "artifact.json".to_string(),
                object_key: "other-prefix/artifact.json".to_string(),
                local_path: object_path.to_string_lossy().into_owned(),
                sha256: sha256_file(&object_path).unwrap(),
                size_bytes: object_path.metadata().unwrap().len(),
            }],
        };
        assert!(validate_object(&manifest, &manifest.objects[0]).is_err());
    }
}
