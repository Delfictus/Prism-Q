use std::{collections::BTreeMap, path::PathBuf};

use anyhow::{Context, Result};
use arrow_schema::DataType;
use clap::Parser;
use prism_nhs::io::provenance::{
    f32s, filter_streams, i32s, mechanical_dot, read_f32_file, record_batch, schema, strings, u8s,
    AscVector, AtomIdx, ForceVector, ProvenanceOptions, ProvenanceParquetWriter, StreamPath, Vec3,
    CAMPAIGN_ID, DEFAULT_OUTPUT_DIR,
};
use rayon::prelude::*;
use serde_json::json;

#[derive(Parser, Debug)]
struct Args {
    /// Raw campaign root (required; there is no compiled-in host default).
    #[arg(long)]
    raw_root: PathBuf,
    #[arg(long, default_value = DEFAULT_OUTPUT_DIR)]
    out_dir: PathBuf,
    #[arg(long)]
    condition_id: Option<String>,
    #[arg(long)]
    replica_id: Option<u16>,
    #[arg(long)]
    max_streams: Option<usize>,
}

#[derive(Debug, Clone)]
struct ForceAscPair {
    force: StreamPath,
    asc: StreamPath,
}

fn pair_streams(forces: Vec<StreamPath>, asc: Vec<StreamPath>) -> Vec<ForceAscPair> {
    let asc_by_key = asc
        .into_iter()
        .map(|item| {
            (
                (
                    item.condition_id.clone(),
                    item.replica_id.0,
                    item.stream_id.0,
                ),
                item,
            )
        })
        .collect::<BTreeMap<_, _>>();
    forces
        .into_iter()
        .filter_map(|force| {
            asc_by_key
                .get(&(
                    force.condition_id.clone(),
                    force.replica_id.0,
                    force.stream_id.0,
                ))
                .cloned()
                .map(|asc| ForceAscPair { force, asc })
        })
        .collect()
}

fn batch_for_pair(
    pair: &ForceAscPair,
    schema_ref: std::sync::Arc<arrow_schema::Schema>,
) -> Result<arrow_array::RecordBatch> {
    let forces = read_f32_file(&pair.force.path)?;
    let asc = read_f32_file(&pair.asc.path)?;
    if forces.len() != asc.len() {
        anyhow::bail!(
            "shape mismatch force={} asc={} for stream {}",
            forces.len(),
            asc.len(),
            pair.force.path.display()
        );
    }
    if forces.len() % 3 != 0 {
        anyhow::bail!(
            "{} force vector length is not xyz triples",
            pair.force.path.display()
        );
    }
    let n_atoms = forces.len() / 3;
    let mut atom_idx = Vec::with_capacity(n_atoms);
    let mut force_x = Vec::with_capacity(n_atoms);
    let mut force_y = Vec::with_capacity(n_atoms);
    let mut force_z = Vec::with_capacity(n_atoms);
    let mut asc_x = Vec::with_capacity(n_atoms);
    let mut asc_y = Vec::with_capacity(n_atoms);
    let mut asc_z = Vec::with_capacity(n_atoms);
    let mut load = Vec::with_capacity(n_atoms);
    for idx in 0..n_atoms {
        let atom = AtomIdx(idx as u32);
        let base = idx * 3;
        let force = ForceVector(Vec3::new(forces[base], forces[base + 1], forces[base + 2]));
        let asc_vec = AscVector(Vec3::new(asc[base], asc[base + 1], asc[base + 2]));
        let dot = mechanical_dot(force, asc_vec);
        atom_idx.push(atom.0 as i32);
        force_x.push(force.0.x);
        force_y.push(force.0.y);
        force_z.push(force.0.z);
        asc_x.push(asc_vec.0.x);
        asc_y.push(asc_vec.0.y);
        asc_z.push(asc_vec.0.z);
        load.push(dot.0);
    }
    record_batch(
        schema_ref,
        vec![
            strings(vec![CAMPAIGN_ID.to_string(); n_atoms]),
            strings(vec![pair.force.condition_id.clone(); n_atoms]),
            u8s(vec![pair.force.replica_id.0 as u8; n_atoms]),
            u8s(vec![pair.force.stream_id.0; n_atoms]),
            i32s(atom_idx),
            f32s(force_x),
            f32s(force_y),
            f32s(force_z),
            f32s(asc_x),
            f32s(asc_y),
            f32s(asc_z),
            f32s(load),
        ],
    )
}

fn main() -> Result<()> {
    let args = Args::parse();
    let force_files =
        prism_nhs::io::provenance::discover_stream_files(&args.raw_root, "forces_final.bin")?;
    let asc_files =
        prism_nhs::io::provenance::discover_stream_files(&args.raw_root, "asc_vectors.bin")?;
    let selected_forces = filter_streams(
        &force_files,
        args.condition_id.as_deref(),
        args.replica_id,
        args.max_streams,
    );
    let selected_asc = filter_streams(
        &asc_files,
        args.condition_id.as_deref(),
        args.replica_id,
        None,
    );
    let pairs = pair_streams(selected_forces, selected_asc);
    let out = args.out_dir.join("mechanical_load_network.parquet");
    let schema_ref = schema(vec![
        ("campaign_id", DataType::Utf8, false),
        ("condition_id", DataType::Utf8, false),
        ("replica_id", DataType::UInt8, false),
        ("stream_id", DataType::UInt8, false),
        ("atom_idx", DataType::Int32, false),
        ("force_x", DataType::Float32, false),
        ("force_y", DataType::Float32, false),
        ("force_z", DataType::Float32, false),
        ("asc_x", DataType::Float32, false),
        ("asc_y", DataType::Float32, false),
        ("asc_z", DataType::Float32, false),
        ("mechanical_load", DataType::Float32, false),
    ]);
    let input_paths = pairs
        .iter()
        .flat_map(|pair| [pair.force.path.clone(), pair.asc.path.clone()])
        .collect::<Vec<_>>();
    let schema_for_batches = schema_ref.clone();
    let mut writer = ProvenanceParquetWriter::try_new(
        &out,
        schema_ref.clone(),
        ProvenanceOptions {
            module: "phase4_mechanical_load".to_string(),
            schema_version: "prism.mechanical_load_network.v1".to_string(),
            producer: "crates/prism-nhs/src/bin/mechanical_load.rs".to_string(),
            input_paths,
            source_parquets: Vec::new(),
            partition_keys: vec![
                "condition_id".to_string(),
                "replica_id".to_string(),
                "stream_id".to_string(),
                "atom_idx".to_string(),
            ],
            parameters: json!({
                "math": "mechanical_load = force_vector dot asc_vector",
                "layout": "raw f32 AoS n_atoms x 3 for forces_final.bin and asc_vectors.bin",
                "row_group_stream_pair_chunk_size": 8,
                "selected_stream_pair_count": pairs.len(),
                "max_streams": args.max_streams
            }),
        },
    )?;
    for chunk in pairs.chunks(8) {
        let batches = chunk
            .par_iter()
            .map(|pair| batch_for_pair(pair, schema_for_batches.clone()))
            .collect::<Result<Vec<_>>>()
            .context("mechanical load batch computation")?;
        for batch in &batches {
            writer.write(batch)?;
        }
    }
    writer.close()?;
    println!("WROTE {}", out.display());
    Ok(())
}
