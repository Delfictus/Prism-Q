use std::{
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context, Result};
use arrow_array::{ArrayRef, Float64Array, RecordBatch, StringArray, UInt32Array};
use arrow_schema::{DataType, Field, Schema};
use parquet::{
    arrow::ArrowWriter,
    basic::{Compression, ZstdLevel},
    file::properties::WriterProperties,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

use super::edge_attenuator::PerturbedEdgeStats;

pub const CHEM_PERTURBED_DTSG_SCHEMA_TAG: &str = "Chem_Perturbed_DTSG.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ChemPerturbedDtsgSidecar {
    pub schema: &'static str,
    pub parquet_path: String,
    pub parquet_sha256: String,
    pub parquet_bytes: u64,
    pub row_count: usize,
}

/// Write the Phase-0 Chem_Perturbed_DTSG edge contract without changing the
/// dispatcher response schema or invoking CUDA propagation.
pub fn write_chem_perturbed_dtsg_parquet(
    rows: &[PerturbedEdgeStats],
    path: impl AsRef<Path>,
) -> Result<ChemPerturbedDtsgSidecar> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }

    let tmp = temp_path(path)?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("schema", DataType::Utf8, false),
        Field::new("edge_index", DataType::UInt32, false),
        Field::new("from_residue", DataType::UInt32, false),
        Field::new("to_residue", DataType::UInt32, false),
        Field::new("te_out_mean", DataType::Float64, false),
        Field::new("te_in_mean", DataType::Float64, false),
        Field::new("delta_hc_mean", DataType::Float64, false),
        Field::new("sigma_hyd_perturbed", DataType::Float64, false),
        Field::new("u_pose_te", DataType::Float64, false),
        Field::new("u_pose_hc", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(vec![
                CHEM_PERTURBED_DTSG_SCHEMA_TAG;
                rows.len()
            ])) as ArrayRef,
            Arc::new(UInt32Array::from(
                (0..rows.len()).map(|idx| idx as u32).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(UInt32Array::from(
                rows.iter()
                    .map(|row| row.edge.from.get())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(UInt32Array::from(
                rows.iter().map(|row| row.edge.to.get()).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Float64Array::from(
                rows.iter()
                    .map(|row| row.te_out_mean.get())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Float64Array::from(
                rows.iter()
                    .map(|row| row.te_in_mean.get())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Float64Array::from(
                rows.iter()
                    .map(|row| row.delta_hc_mean.get())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Float64Array::from(
                rows.iter()
                    .map(|row| row.sigma_hyd_perturbed.get())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Float64Array::from(
                rows.iter()
                    .map(|row| row.u_pose_te.get())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Float64Array::from(
                rows.iter()
                    .map(|row| row.u_pose_hc.get())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
        ],
    )
    .context("build Chem_Perturbed_DTSG record batch")?;

    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
        .set_dictionary_enabled(true)
        .build();
    let file = File::create(&tmp).with_context(|| format!("create {}", tmp.display()))?;
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    fs::rename(&tmp, path)
        .with_context(|| format!("rename {} to {}", tmp.display(), path.display()))?;

    let parquet_sha256 = sha256_file(path)?;
    let parquet_bytes = fs::metadata(path)
        .with_context(|| format!("stat {}", path.display()))?
        .len();
    let sidecar = ChemPerturbedDtsgSidecar {
        schema: CHEM_PERTURBED_DTSG_SCHEMA_TAG,
        parquet_path: path.display().to_string(),
        parquet_sha256,
        parquet_bytes,
        row_count: rows.len(),
    };
    write_sidecar_atomic(&sidecar_path(path), &sidecar)?;
    sync_parent(path)?;
    Ok(sidecar)
}

pub fn sidecar_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .map(|name| name.to_string_lossy())
        .unwrap_or_else(|| "Chem_Perturbed_DTSG.parquet".into());
    path.with_file_name(format!("{file_name}.sha256.json"))
}

fn write_sidecar_atomic(path: &Path, sidecar: &ChemPerturbedDtsgSidecar) -> Result<()> {
    let tmp = temp_path(path)?;
    let mut payload = serde_json::to_vec_pretty(sidecar)?;
    payload.push(b'\n');
    {
        let mut file = File::create(&tmp).with_context(|| format!("create {}", tmp.display()))?;
        file.write_all(&payload)
            .with_context(|| format!("write {}", tmp.display()))?;
        file.sync_all()
            .with_context(|| format!("sync {}", tmp.display()))?;
    }
    fs::rename(&tmp, path)
        .with_context(|| format!("rename {} to {}", tmp.display(), path.display()))?;
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("read {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn temp_path(path: &Path) -> Result<PathBuf> {
    let file_name = path
        .file_name()
        .context("output path must include a file name")?
        .to_string_lossy();
    Ok(path.with_file_name(format!(".{file_name}.{}.tmp", std::process::id())))
}

fn sync_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if let Ok(dir) = File::open(parent) {
            dir.sync_all()
                .with_context(|| format!("sync {}", parent.display()))?;
        }
    }
    Ok(())
}
