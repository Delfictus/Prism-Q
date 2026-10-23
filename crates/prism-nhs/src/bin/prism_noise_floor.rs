//! Native same-seed noise-floor analyzer for CUDA epsilon-budget decisions.
//!
//! This binary is intentionally downstream-invariance oriented: frame-count
//! mismatches fail, exact frame-hash drift is reported, and the acceptance
//! decision is based on binding-site stability plus per-residue feature drift.

use anyhow::{Context, Result};
use arrow_array::{
    Array, BooleanArray, FixedSizeListArray, Float32Array, Float64Array, Int16Array, Int32Array,
    Int64Array, RecordBatch, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow_ipc::reader::{FileReader as ArrowFileReader, StreamReader as ArrowStreamReader};
use arrow_schema::DataType;
use chrono::Utc;
use clap::Parser;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

const RESIDUE_COLUMNS: &[&str] = &[
    "residue_id",
    "res_id",
    "residue_idx",
    "residue_index",
    "rid",
];
const SPIKE_RESIDUE_COLUMNS: &[&str] =
    &["residue_id", "res_id", "residue_idx", "nearest_residue_id"];
const SPIKE_INTENSITY_COLUMNS: &[&str] = &["intensity", "spike_intensity", "amplitude", "weight"];
const SITE_LIST_KEYS: &[&str] = &[
    "sites",
    "binding_sites",
    "ranked_sites",
    "pockets",
    "candidate_sites",
];
const SITE_SCORE_KEYS: &[&str] = &[
    "rank_score",
    "score",
    "phase_manifold_score",
    "max_phase_manifold_score",
    "kcc_score",
    "confidence",
    "druggability",
];
const THERM_KEYS: &[&str] = &[
    "therm_class",
    "classification",
    "pocket_class",
    "site_classification",
];

#[derive(Parser, Debug)]
#[command(name = "prism-noise-floor")]
#[command(about = "Native downstream-invariance noise-floor analyzer")]
struct Args {
    #[arg(required = true)]
    runs: Vec<String>,
    #[arg(long, default_value = "noise_floor_report.json")]
    output_json: PathBuf,
    #[arg(long)]
    read_spike_arrow: bool,
    #[arg(long, default_value_t = 5)]
    top_k: usize,
    #[arg(long, default_value_t = 5)]
    min_trials: usize,
    #[arg(long, default_value_t = 0.001)]
    max_spike_count_rel_drift: f64,
    #[arg(long, default_value_t = 0.001)]
    max_feature_rel_drift_p99: f64,
    #[arg(long, default_value_t = 1.0)]
    min_top1_agreement: f64,
    #[arg(long, default_value_t = 1.0)]
    min_topk_jaccard: f64,
    #[arg(long, default_value_t = 1.0)]
    min_therm_agreement: f64,
    #[arg(long, default_value_t = 1024)]
    parquet_batch_size: usize,
}

#[derive(Clone, Debug, Serialize)]
struct Site {
    key: String,
    rank: i64,
    score: f64,
    residues: Vec<i32>,
    centroid: Option<[f64; 3]>,
    therm_class: String,
}

#[derive(Debug, Serialize)]
struct RunRecord {
    name: String,
    path: String,
    frame_counts: Vec<u64>,
    frame_hashes: Vec<String>,
    all_hashes_match: Option<bool>,
    sites: Vec<Site>,
    #[serde(skip_serializing)]
    residue_features: HashMap<String, BTreeMap<i32, f64>>,
    artifacts: BTreeMap<String, String>,
    warnings: Vec<String>,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    let paths = expand_paths(&args.runs)?;
    let mut runs = Vec::with_capacity(paths.len());
    for path in paths {
        runs.push(load_run(&path, &args)?);
    }

    let frames = summarize_frames(&runs);
    let sites = summarize_sites(&runs, args.top_k);
    let features = summarize_features(&runs, args.max_spike_count_rel_drift);
    let decision = make_decision(runs.len(), &frames, &sites, &features, &args);

    let report = json!({
        "schema_version": "1.0.0",
        "computed_at": Utc::now().to_rfc3339(),
        "computed_by": "prism-noise-floor",
        "n_runs": runs.len(),
        "runs": runs.iter().map(|r| json!({
            "name": r.name,
            "path": r.path,
            "artifacts": r.artifacts,
            "warnings": r.warnings,
            "n_sites": r.sites.len(),
            "n_feature_metrics": r.residue_features.len(),
        })).collect::<Vec<_>>(),
        "frames": frames,
        "sites": sites,
        "features": features,
        "decision": decision,
    });
    if let Some(parent) = args
        .output_json
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.output_json, serde_json::to_vec_pretty(&report)?)?;

    let accepted = report["decision"]["accepted_for_epsilon_budget"]
        .as_bool()
        .unwrap_or(false);
    println!(
        "noise-floor decision: {}",
        if accepted { "ACCEPT" } else { "REJECT" }
    );
    println!("runs: {}", runs.len());
    println!(
        "frame count match: {:.3}",
        report["frames"]["count_match_fraction_vs_baseline"]
            .as_f64()
            .unwrap_or(0.0)
    );
    println!(
        "top{} jaccard min: {:.3}",
        args.top_k,
        report["sites"]["topk_jaccard_min"].as_f64().unwrap_or(0.0)
    );
    println!(
        "feature rel drift p99: {}",
        scalar_json_display(&report["features"]["all_feature_rel_drift_p99"])
    );
    println!("report: {}", args.output_json.display());
    std::process::exit(if accepted { 0 } else { 2 });
}

fn expand_paths(items: &[String]) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for item in items {
        let mut matched = false;
        for hit in glob::glob(item).with_context(|| format!("glob {}", item))? {
            out.push(hit?);
            matched = true;
        }
        if !matched {
            out.push(PathBuf::from(item));
        }
    }
    out.sort();
    out.dedup();
    Ok(out)
}

fn load_run(path: &Path, args: &Args) -> Result<RunRecord> {
    let root = if path.is_dir() {
        path.to_path_buf()
    } else {
        path.parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf()
    };
    let name = if path.is_file() {
        path.file_stem()
    } else {
        path.file_name()
    }
    .and_then(|v| v.to_str())
    .unwrap_or("run")
    .to_string();
    let mut rec = RunRecord {
        name,
        path: path.display().to_string(),
        frame_counts: Vec::new(),
        frame_hashes: Vec::new(),
        all_hashes_match: None,
        sites: Vec::new(),
        residue_features: HashMap::new(),
        artifacts: BTreeMap::new(),
        warnings: Vec::new(),
    };

    let manifest_path = if path.is_file() {
        Some(path.to_path_buf())
    } else {
        let mut candidates = vec![root.join("ensemble_manifest.json")];
        candidates.extend(glob_paths(&root, "ensemble_replica_*.json")?);
        first_existing(candidates)
    };
    let mut manifest = Value::Null;
    let mut replica = Value::Null;
    if let Some(p) = manifest_path {
        match load_json(&p) {
            Ok(obj) => {
                replica = first_replica(&obj);
                manifest = obj;
                rec.artifacts
                    .insert("manifest_or_record".to_string(), p.display().to_string());
            }
            Err(e) => rec
                .warnings
                .push(format!("manifest_or_record_unreadable:{e}")),
        }
    }

    let audit_src = if replica.is_object() {
        &replica
    } else {
        &manifest
    };
    let (counts, hashes, all_match) = extract_frame_audit(audit_src);
    rec.frame_counts = counts;
    rec.frame_hashes = hashes;
    rec.all_hashes_match = all_match;
    if rec.frame_counts.is_empty() {
        let (counts, hashes) = extract_pdb_frame_audit(&root, &mut rec.warnings)?;
        if !counts.is_empty() {
            rec.frame_counts = counts;
            rec.frame_hashes = hashes;
            rec.warnings
                .push("frame_audit_fallback_from_stream_ensemble_pdb".to_string());
        }
    }

    let outputs = replica.get("outputs").or_else(|| manifest.get("outputs"));
    let binding_path = outputs
        .and_then(|o| o.get("binding_sites_json_relative"))
        .and_then(Value::as_str)
        .map(|v| resolve_relative(&root, v))
        .filter(|p| p.exists())
        .or_else(|| {
            find_one(
                &root,
                &[
                    "binding_sites.json",
                    "**/binding_sites.json",
                    "*.binding_sites.json",
                    "**/*.binding_sites.json",
                ],
            )
            .ok()
            .flatten()
        });
    if let Some(p) = binding_path {
        rec.artifacts
            .insert("binding_sites_json".to_string(), p.display().to_string());
        rec.sites = parse_sites(&p, &mut rec.warnings);
    } else {
        rec.warnings.push("binding_sites_json_missing".to_string());
    }

    if let Some(p) = find_one(
        &root,
        &[
            "*_v5.parquet",
            "**/*_v5.parquet",
            "*features*.parquet",
            "**/*features*.parquet",
        ],
    )? {
        rec.artifacts
            .insert("feature_parquet".to_string(), p.display().to_string());
        merge_feature_maps(
            &mut rec.residue_features,
            read_feature_parquet(&p, args.parquet_batch_size, &mut rec.warnings)?,
        );
    }

    if args.read_spike_arrow {
        let arrow_path = outputs
            .and_then(|o| o.get("trajectory_arrow_relative"))
            .and_then(Value::as_str)
            .map(|v| resolve_relative(&root, v))
            .filter(|p| p.exists())
            .or_else(|| {
                find_one(&root, &["*.spike_events.arrow", "**/*.spike_events.arrow"])
                    .ok()
                    .flatten()
            });
        if let Some(p) = arrow_path {
            rec.artifacts
                .insert("spike_arrow".to_string(), p.display().to_string());
            merge_feature_maps(
                &mut rec.residue_features,
                read_spike_arrow(&p, &mut rec.warnings)?,
            );
        } else {
            rec.warnings.push("spike_arrow_missing".to_string());
        }
    }
    Ok(rec)
}

fn glob_paths(root: &Path, pattern: &str) -> Result<Vec<PathBuf>> {
    let mut paths = glob::glob(&root.join(pattern).to_string_lossy())?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    paths.sort();
    Ok(paths)
}

fn first_existing<I>(paths: I) -> Option<PathBuf>
where
    I: IntoIterator<Item = PathBuf>,
{
    paths.into_iter().find(|p| p.exists())
}

fn find_one(root: &Path, patterns: &[&str]) -> Result<Option<PathBuf>> {
    for pattern in patterns {
        let paths = glob_paths(root, pattern)?;
        if let Some(p) = paths.into_iter().next() {
            return Ok(Some(p));
        }
    }
    Ok(None)
}

fn resolve_relative(base: &Path, value: &str) -> PathBuf {
    let path = PathBuf::from(value);
    if path.is_absolute() {
        path
    } else {
        base.join(path)
    }
}

fn load_json(path: &Path) -> Result<Value> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
}

fn first_replica(obj: &Value) -> Value {
    if let Some(rep) = obj
        .get("replicas")
        .and_then(Value::as_array)
        .and_then(|v| v.first())
    {
        return rep.clone();
    }
    obj.get("replica").cloned().unwrap_or(Value::Null)
}

fn extract_frame_audit(obj: &Value) -> (Vec<u64>, Vec<String>, Option<bool>) {
    let audit = obj
        .get("frame_audit")
        .or_else(|| obj.get("replica").and_then(|r| r.get("frame_audit")))
        .unwrap_or(&Value::Null);
    let counts = first_array(
        audit,
        &[
            "disk_count_per_stream",
            "writer_count_per_stream",
            "producer_count_per_stream",
        ],
    )
    .into_iter()
    .filter_map(Value::as_u64)
    .collect::<Vec<_>>();
    let hashes = first_array(
        audit,
        &[
            "disk_hash_per_stream",
            "writer_hash_per_stream",
            "producer_hash_per_stream",
        ],
    )
    .into_iter()
    .filter_map(Value::as_str)
    .map(str::to_string)
    .collect::<Vec<_>>();
    let all_hashes_match = audit.get("all_hashes_match").and_then(Value::as_bool);
    (counts, hashes, all_hashes_match)
}

fn extract_pdb_frame_audit(
    root: &Path,
    warnings: &mut Vec<String>,
) -> Result<(Vec<u64>, Vec<String>)> {
    let mut paths = glob_paths(root, "*_stream*.ensemble_trajectory.pdb")?;
    if paths.is_empty() {
        paths = glob_paths(root, "**/*_stream*.ensemble_trajectory.pdb")?;
    }
    paths.sort();

    let mut counts = Vec::with_capacity(paths.len());
    let mut hashes = Vec::with_capacity(paths.len());
    for path in paths {
        let file = match File::open(&path) {
            Ok(file) => file,
            Err(e) => {
                warnings.push(format!("trajectory_pdb_unreadable:{}:{e}", path.display()));
                continue;
            }
        };
        let mut n_models = 0_u64;
        for line in BufReader::new(file).lines() {
            let line = line?;
            if line.starts_with("MODEL") {
                n_models += 1;
            }
        }
        if n_models == 0 {
            warnings.push(format!(
                "trajectory_pdb_no_model_records:{}",
                path.display()
            ));
        }
        counts.push(n_models);
        hashes.push(prism_nhs::ensemble::manifest::sha256_file_or_absent(&path));
    }
    Ok((counts, hashes))
}

fn first_array<'a>(obj: &'a Value, keys: &[&str]) -> Vec<&'a Value> {
    keys.iter()
        .find_map(|k| obj.get(*k).and_then(Value::as_array))
        .map(|v| v.iter().collect())
        .unwrap_or_default()
}

fn parse_sites(path: &Path, warnings: &mut Vec<String>) -> Vec<Site> {
    let obj = match load_json(path) {
        Ok(v) => v,
        Err(e) => {
            warnings.push(format!("binding_sites_json_unreadable:{e}"));
            return Vec::new();
        }
    };
    let raw_sites = find_site_list(&obj);
    let mut sites = raw_sites
        .iter()
        .enumerate()
        .map(|(idx, site)| {
            let residues = normalize_residue_list(
                site.get("residue_ids")
                    .or_else(|| site.get("residues"))
                    .or_else(|| site.get("lining_residues"))
                    .or_else(|| site.get("hot_residues")),
            );
            let site_id = site
                .get("site_id")
                .or_else(|| site.get("id"))
                .or_else(|| site.get("rank"))
                .map(Value::to_string)
                .unwrap_or_else(|| idx.to_string());
            let rank = site
                .get("rank")
                .and_then(Value::as_i64)
                .unwrap_or(idx as i64 + 1);
            let score = SITE_SCORE_KEYS
                .iter()
                .find_map(|k| site.get(*k).and_then(numeric))
                .unwrap_or(0.0);
            let therm_class = THERM_KEYS
                .iter()
                .find_map(|k| site.get(*k))
                .and_then(value_to_string)
                .unwrap_or_default();
            let centroid = extract_centroid(site);
            let key = if residues.is_empty() {
                format!("id:{site_id}")
            } else {
                residues
                    .iter()
                    .map(i32::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            };
            Site {
                key,
                rank,
                score,
                residues,
                centroid,
                therm_class,
            }
        })
        .collect::<Vec<_>>();
    sites.sort_by(|a, b| {
        a.rank
            .cmp(&b.rank)
            .then_with(|| b.score.total_cmp(&a.score))
            .then_with(|| a.key.cmp(&b.key))
    });
    sites
}

fn find_site_list(obj: &Value) -> Vec<Value> {
    match obj {
        Value::Array(items) => items.iter().filter(|v| v.is_object()).cloned().collect(),
        Value::Object(map) => {
            for key in SITE_LIST_KEYS {
                if let Some(Value::Array(items)) = map.get(*key) {
                    return items.iter().filter(|v| v.is_object()).cloned().collect();
                }
            }
            for val in map.values() {
                let nested = find_site_list(val);
                if !nested.is_empty() {
                    return nested;
                }
            }
            Vec::new()
        }
        _ => Vec::new(),
    }
}

fn normalize_residue_list(value: Option<&Value>) -> Vec<i32> {
    let Some(value) = value else {
        return Vec::new();
    };
    let mut out = Vec::new();
    match value {
        Value::Array(items) => {
            for item in items {
                if let Some(v) = item
                    .as_i64()
                    .or_else(|| item.get("residue_id").and_then(Value::as_i64))
                    .or_else(|| item.get("res_id").and_then(Value::as_i64))
                    .or_else(|| item.get("id").and_then(Value::as_i64))
                {
                    out.push(v as i32);
                }
            }
        }
        Value::String(s) => {
            for item in s.replace(',', " ").split_whitespace() {
                if let Ok(v) = item.parse::<i32>() {
                    out.push(v);
                }
            }
        }
        Value::Object(map) => {
            out = normalize_residue_list(
                map.get("residue_ids")
                    .or_else(|| map.get("residues"))
                    .or_else(|| map.get("ids")),
            );
        }
        _ => {}
    }
    out.sort();
    out.dedup();
    out
}

fn extract_centroid(site: &Value) -> Option<[f64; 3]> {
    for key in ["centroid", "center", "coordinates", "xyz", "position"] {
        let Some(v) = site.get(key) else {
            continue;
        };
        if let Some(xyz) = xyz_from_value(v) {
            return Some(xyz);
        }
    }
    let x = site.get("x").and_then(numeric)?;
    let y = site.get("y").and_then(numeric)?;
    let z = site.get("z").and_then(numeric)?;
    Some([x, y, z])
}

fn xyz_from_value(v: &Value) -> Option<[f64; 3]> {
    match v {
        Value::Array(items) if items.len() >= 3 => Some([
            numeric(&items[0])?,
            numeric(&items[1])?,
            numeric(&items[2])?,
        ]),
        Value::Object(map) => Some([
            map.get("x").or_else(|| map.get("0")).and_then(numeric)?,
            map.get("y").or_else(|| map.get("1")).and_then(numeric)?,
            map.get("z").or_else(|| map.get("2")).and_then(numeric)?,
        ]),
        _ => None,
    }
}

fn numeric(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64().filter(|v| v.is_finite()),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::String(s) => s.parse::<f64>().ok().filter(|v| v.is_finite()),
        _ => None,
    }
}

fn value_to_string(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(_) | Value::Bool(_) => Some(v.to_string()),
        _ => None,
    }
}

fn read_feature_parquet(
    path: &Path,
    batch_size: usize,
    warnings: &mut Vec<String>,
) -> Result<HashMap<String, BTreeMap<i32, f64>>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("open parquet reader {}", path.display()))?
        .with_batch_size(batch_size);
    let mut reader = builder.build()?;
    let mut out = HashMap::<String, BTreeMap<i32, f64>>::new();
    let mut row_offset = 0usize;
    while let Some(batch) = reader.next().transpose()? {
        let residue_ids = batch_residue_ids(&batch, row_offset, RESIDUE_COLUMNS)?;
        merge_numeric_batch(&batch, &residue_ids, &mut out);
        row_offset += batch.num_rows();
    }
    if out.is_empty() {
        warnings.push(format!(
            "feature_parquet_no_numeric_metrics:{}",
            path.display()
        ));
    }
    Ok(out)
}

fn read_spike_arrow(
    path: &Path,
    warnings: &mut Vec<String>,
) -> Result<HashMap<String, BTreeMap<i32, f64>>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let batches = match ArrowFileReader::try_new(file, None) {
        Ok(reader) => reader.collect::<std::result::Result<Vec<_>, _>>()?,
        Err(_) => {
            let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
            ArrowStreamReader::try_new(file, None)?.collect::<std::result::Result<Vec<_>, _>>()?
        }
    };
    let mut counts = BTreeMap::<i32, f64>::new();
    let mut sums = BTreeMap::<i32, f64>::new();
    for batch in batches {
        let intensity_idx = SPIKE_INTENSITY_COLUMNS
            .iter()
            .find_map(|c| batch.schema().index_of(c).ok());
        for row in 0..batch.num_rows() {
            let residues = spike_residue_weights(&batch, row);
            if residues.is_empty() {
                continue;
            }
            if let Some(idx) = intensity_idx {
                if let Some(intensity) = array_value_f64(batch.column(idx).as_ref(), row) {
                    for (rid, weight) in residues {
                        *counts.entry(rid).or_insert(0.0) += weight;
                        *sums.entry(rid).or_insert(0.0) += intensity * weight;
                    }
                    continue;
                }
            }
            for (rid, weight) in residues {
                *counts.entry(rid).or_insert(0.0) += weight;
            }
        }
    }
    if counts.is_empty() {
        warnings.push(format!("spike_arrow_no_residue_signal:{}", path.display()));
    }
    let mut out = HashMap::new();
    if !counts.is_empty() {
        out.insert("spike_count".to_string(), counts.clone());
    }
    if !sums.is_empty() {
        let means = sums
            .iter()
            .filter_map(|(rid, sum)| counts.get(rid).map(|count| (*rid, sum / count.max(1.0))))
            .collect::<BTreeMap<_, _>>();
        out.insert("spike_intensity_sum".to_string(), sums);
        out.insert("spike_intensity_mean".to_string(), means);
    }
    Ok(out)
}

fn spike_residue_weights(batch: &RecordBatch, row: usize) -> Vec<(i32, f64)> {
    for name in SPIKE_RESIDUE_COLUMNS {
        if let Ok(idx) = batch.schema().index_of(name) {
            if let Some(rid) = array_value_i64(batch.column(idx).as_ref(), row) {
                if rid >= 0 {
                    return vec![(rid as i32, 1.0)];
                }
            }
        }
    }

    if let Ok(idx) = batch.schema().index_of("nearby_residues") {
        if let Some(list) = batch
            .column(idx)
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
        {
            if let Some(values) = list.values().as_any().downcast_ref::<Int32Array>() {
                let width = list.value_length().max(0) as usize;
                let declared = batch
                    .schema()
                    .index_of("n_residues")
                    .ok()
                    .and_then(|n_idx| array_value_i64(batch.column(n_idx).as_ref(), row))
                    .map(|n| n.max(0) as usize)
                    .unwrap_or(width)
                    .min(width);
                let base = row * width;
                let mut residues = Vec::with_capacity(declared);
                for j in 0..declared {
                    let rid = values.value(base + j);
                    if rid >= 0 {
                        residues.push(rid);
                    }
                }
                residues.sort_unstable();
                residues.dedup();
                if !residues.is_empty() {
                    let weight = 1.0 / residues.len() as f64;
                    return residues.into_iter().map(|rid| (rid, weight)).collect();
                }
            }
        }
    }

    if let Ok(idx) = batch.schema().index_of("aromatic_residue_id") {
        if let Some(rid) = array_value_i64(batch.column(idx).as_ref(), row) {
            if rid >= 0 {
                return vec![(rid as i32, 1.0)];
            }
        }
    }

    Vec::new()
}

fn batch_residue_ids(
    batch: &RecordBatch,
    row_offset: usize,
    candidates: &[&str],
) -> Result<Vec<i32>> {
    for name in candidates {
        if let Ok(idx) = batch.schema().index_of(name) {
            let array = batch.column(idx);
            let mut out = Vec::with_capacity(batch.num_rows());
            for row in 0..batch.num_rows() {
                out.push(
                    array_value_i64(array.as_ref(), row)
                        .with_context(|| format!("invalid residue id in column {name}"))?
                        as i32,
                );
            }
            return Ok(out);
        }
    }
    Ok((0..batch.num_rows())
        .map(|i| (row_offset + i) as i32)
        .collect())
}

fn merge_numeric_batch(
    batch: &RecordBatch,
    residue_ids: &[i32],
    out: &mut HashMap<String, BTreeMap<i32, f64>>,
) {
    for (idx, field) in batch.schema().fields().iter().enumerate() {
        let name = field.name();
        if RESIDUE_COLUMNS.contains(&name.as_str()) || !is_numeric(field.data_type()) {
            continue;
        }
        let metric = out.entry(name.clone()).or_default();
        let array = batch.column(idx);
        for row in 0..batch.num_rows() {
            if let Some(v) = array_value_f64(array.as_ref(), row) {
                metric.insert(residue_ids[row], v);
            }
        }
    }
}

fn is_numeric(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Float32
            | DataType::Float64
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Boolean
    )
}

fn array_value_f64(array: &dyn Array, row: usize) -> Option<f64> {
    if array.is_null(row) {
        return None;
    }
    let v = match array.data_type() {
        DataType::Float32 => array.as_any().downcast_ref::<Float32Array>()?.value(row) as f64,
        DataType::Float64 => array.as_any().downcast_ref::<Float64Array>()?.value(row),
        DataType::Int16 => array.as_any().downcast_ref::<Int16Array>()?.value(row) as f64,
        DataType::Int32 => array.as_any().downcast_ref::<Int32Array>()?.value(row) as f64,
        DataType::Int64 => array.as_any().downcast_ref::<Int64Array>()?.value(row) as f64,
        DataType::UInt8 => array.as_any().downcast_ref::<UInt8Array>()?.value(row) as f64,
        DataType::UInt16 => array.as_any().downcast_ref::<UInt16Array>()?.value(row) as f64,
        DataType::UInt32 => array.as_any().downcast_ref::<UInt32Array>()?.value(row) as f64,
        DataType::UInt64 => array.as_any().downcast_ref::<UInt64Array>()?.value(row) as f64,
        DataType::Boolean => {
            if array.as_any().downcast_ref::<BooleanArray>()?.value(row) {
                1.0
            } else {
                0.0
            }
        }
        _ => return None,
    };
    v.is_finite().then_some(v)
}

fn array_value_i64(array: &dyn Array, row: usize) -> Option<i64> {
    if array.is_null(row) {
        return None;
    }
    match array.data_type() {
        DataType::Int16 => Some(array.as_any().downcast_ref::<Int16Array>()?.value(row) as i64),
        DataType::Int32 => Some(array.as_any().downcast_ref::<Int32Array>()?.value(row) as i64),
        DataType::Int64 => Some(array.as_any().downcast_ref::<Int64Array>()?.value(row)),
        DataType::UInt8 => Some(array.as_any().downcast_ref::<UInt8Array>()?.value(row) as i64),
        DataType::UInt16 => Some(array.as_any().downcast_ref::<UInt16Array>()?.value(row) as i64),
        DataType::UInt32 => Some(array.as_any().downcast_ref::<UInt32Array>()?.value(row) as i64),
        DataType::UInt64 => Some(array.as_any().downcast_ref::<UInt64Array>()?.value(row) as i64),
        _ => None,
    }
}

fn merge_feature_maps(
    dst: &mut HashMap<String, BTreeMap<i32, f64>>,
    src: HashMap<String, BTreeMap<i32, f64>>,
) {
    for (name, values) in src {
        dst.entry(name).or_default().extend(values);
    }
}

fn summarize_frames(runs: &[RunRecord]) -> Value {
    let baseline_counts = runs
        .first()
        .map(|r| r.frame_counts.clone())
        .unwrap_or_default();
    let baseline_hashes = runs
        .first()
        .map(|r| r.frame_hashes.clone())
        .unwrap_or_default();
    let count_matches = runs
        .iter()
        .map(|r| !r.frame_counts.is_empty() && r.frame_counts == baseline_counts)
        .collect::<Vec<_>>();
    let hash_matches = runs
        .iter()
        .map(|r| !r.frame_hashes.is_empty() && r.frame_hashes == baseline_hashes)
        .collect::<Vec<_>>();
    let all_flags = runs
        .iter()
        .filter_map(|r| r.all_hashes_match)
        .collect::<Vec<_>>();
    json!({
        "streams": baseline_counts.len(),
        "baseline_total_frames": baseline_counts.iter().sum::<u64>(),
        "count_match_fraction_vs_baseline": bool_mean(&count_matches),
        "hash_exact_match_fraction_vs_baseline": bool_mean(&hash_matches),
        "all_hashes_match_fraction_internal": optional_f64((!all_flags.is_empty()).then(|| bool_mean(&all_flags))),
        "count_mismatched_runs": runs.iter().zip(count_matches.iter()).filter_map(|(r, ok)| (!*ok).then_some(r.name.clone())).collect::<Vec<_>>(),
        "hash_mismatched_runs": runs.iter().zip(hash_matches.iter()).filter_map(|(r, ok)| (!*ok).then_some(r.name.clone())).collect::<Vec<_>>(),
    })
}

fn summarize_sites(runs: &[RunRecord], top_k: usize) -> Value {
    let baseline = runs
        .first()
        .map(|r| r.sites.iter().take(top_k).cloned().collect::<Vec<_>>())
        .unwrap_or_default();
    let baseline_keys = baseline.iter().map(|s| s.key.clone()).collect::<Vec<_>>();
    let mut top1_matches = Vec::new();
    let mut topk_jaccards = Vec::new();
    let mut therm_matches = Vec::new();
    for run in runs {
        let top = run.sites.iter().take(top_k).collect::<Vec<_>>();
        top1_matches.push(
            baseline
                .first()
                .zip(top.first())
                .map(|(base, site)| site_matches(base, site))
                .unwrap_or(false),
        );
        let matched = match_sites(&baseline, &top);
        let union = baseline.len() + top.len() - matched.len();
        topk_jaccards.push(if union == 0 {
            0.0
        } else {
            matched.len() as f64 / union as f64
        });
        let shared = matched
            .iter()
            .filter_map(|(bi, ti)| {
                let base = &baseline[*bi];
                let site = top[*ti];
                (!base.therm_class.is_empty() && !site.therm_class.is_empty())
                    .then_some(base.therm_class == site.therm_class)
            })
            .collect::<Vec<_>>();
        if !shared.is_empty() {
            therm_matches.push(bool_mean(&shared));
        }
    }
    json!({
        "top_k": top_k,
        "baseline_top_keys": baseline_keys,
        "top1_agreement_fraction": bool_mean(&top1_matches),
        "topk_jaccard_mean": mean(&topk_jaccards),
        "topk_jaccard_min": topk_jaccards.iter().copied().reduce(f64::min).unwrap_or(0.0),
        "therm_class_agreement_fraction": optional_f64((!therm_matches.is_empty()).then(|| mean(&therm_matches))),
        "runs_without_sites": runs.iter().filter(|r| r.sites.is_empty()).map(|r| r.name.clone()).collect::<Vec<_>>(),
    })
}

fn match_sites(baseline: &[Site], top: &[&Site]) -> Vec<(usize, usize)> {
    let mut used = vec![false; top.len()];
    let mut matched = Vec::new();
    for (bi, base) in baseline.iter().enumerate() {
        let mut best: Option<(usize, f64)> = None;
        for (ti, site) in top.iter().enumerate() {
            if used[ti] {
                continue;
            }
            let Some(score) = site_match_score(base, site) else {
                continue;
            };
            if best.map(|(_, s)| score > s).unwrap_or(true) {
                best = Some((ti, score));
            }
        }
        if let Some((ti, _)) = best {
            used[ti] = true;
            matched.push((bi, ti));
        }
    }
    matched
}

fn site_matches(a: &Site, b: &Site) -> bool {
    site_match_score(a, b).is_some()
}

fn site_match_score(a: &Site, b: &Site) -> Option<f64> {
    if let (Some(ac), Some(bc)) = (a.centroid, b.centroid) {
        let d2 = ac
            .iter()
            .zip(bc.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>();
        let d = d2.sqrt();
        if d <= 2.5 {
            return Some(10.0 - d);
        }
    }
    let residue_j = residue_jaccard(&a.residues, &b.residues);
    if residue_j >= 0.85 {
        return Some(residue_j);
    }
    (a.key == b.key).then_some(1.0)
}

fn residue_jaccard(a: &[i32], b: &[i32]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let a = a.iter().copied().collect::<BTreeSet<_>>();
    let b = b.iter().copied().collect::<BTreeSet<_>>();
    let inter = a.intersection(&b).count();
    let union = a.union(&b).count();
    if union == 0 {
        0.0
    } else {
        inter as f64 / union as f64
    }
}

fn summarize_features(runs: &[RunRecord], spike_count_threshold: f64) -> Value {
    let metric_names = runs
        .iter()
        .flat_map(|r| r.residue_features.keys().cloned())
        .collect::<BTreeSet<_>>();
    let mut per_metric = serde_json::Map::new();
    let mut all_rel_drifts = Vec::new();
    let mut spike_rel_drifts = Vec::new();
    for metric in &metric_names {
        let residue_ids = runs
            .iter()
            .flat_map(|r| {
                r.residue_features
                    .get(metric)
                    .into_iter()
                    .flat_map(|m| m.keys().copied())
            })
            .collect::<BTreeSet<_>>();
        let mut abs_drifts = Vec::new();
        let mut rel_drifts = Vec::new();
        for rid in residue_ids {
            let vals = runs
                .iter()
                .map(|r| {
                    r.residue_features
                        .get(metric)
                        .and_then(|m| m.get(&rid))
                        .copied()
                })
                .collect::<Vec<_>>();
            let Some(Some(baseline)) = vals.first() else {
                continue;
            };
            let finite = vals.iter().flatten().copied().collect::<Vec<_>>();
            if finite.len() < 2 {
                continue;
            }
            let max_abs = finite
                .iter()
                .map(|v| (v - baseline).abs())
                .fold(0.0, f64::max);
            let denom = baseline.abs().max(if metric.contains("count") {
                1.0
            } else {
                1.0e-6
            });
            let rel = max_abs / denom;
            abs_drifts.push(max_abs);
            rel_drifts.push(rel);
        }
        if metric.contains("spike_count") || metric == "spike_count" {
            spike_rel_drifts.extend(rel_drifts.iter().copied());
        }
        all_rel_drifts.extend(rel_drifts.iter().copied());
        per_metric.insert(
            metric.clone(),
            json!({
                "n_residues_compared": rel_drifts.len(),
                "abs_drift_max": optional_f64(quantile(&abs_drifts, 1.0)),
                "rel_drift_p95": optional_f64(quantile(&rel_drifts, 0.95)),
                "rel_drift_p99": optional_f64(quantile(&rel_drifts, 0.99)),
                "rel_drift_max": optional_f64(quantile(&rel_drifts, 1.0)),
            }),
        );
    }
    json!({
        "n_metrics": metric_names.len(),
        "metrics": per_metric,
        "all_feature_rel_drift_p95": optional_f64(quantile(&all_rel_drifts, 0.95)),
        "all_feature_rel_drift_p99": optional_f64(quantile(&all_rel_drifts, 0.99)),
        "all_feature_rel_drift_max": optional_f64(quantile(&all_rel_drifts, 1.0)),
        "spike_count_rel_drift_max": optional_f64(quantile(&spike_rel_drifts, 1.0)),
        "spike_count_threshold": spike_count_threshold,
        "has_spike_count_signal": !spike_rel_drifts.is_empty(),
    })
}

fn make_decision(
    n_runs: usize,
    frames: &Value,
    sites: &Value,
    features: &Value,
    args: &Args,
) -> Value {
    let checks = BTreeMap::from([
        ("min_trials", n_runs >= args.min_trials),
        (
            "frame_counts_match",
            frames["count_match_fraction_vs_baseline"].as_f64() == Some(1.0),
        ),
        (
            "site_signal_present",
            sites["runs_without_sites"]
                .as_array()
                .map(|v| v.is_empty())
                .unwrap_or(false),
        ),
        (
            "feature_signal_present",
            features["n_metrics"].as_u64().unwrap_or(0) > 0,
        ),
        (
            "top1_site_agreement",
            sites["top1_agreement_fraction"].as_f64().unwrap_or(0.0) >= args.min_top1_agreement,
        ),
        (
            "topk_jaccard",
            sites["topk_jaccard_min"].as_f64().unwrap_or(0.0) >= args.min_topk_jaccard,
        ),
        (
            "therm_class_agreement",
            value_or_null_at_least(
                sites,
                "therm_class_agreement_fraction",
                args.min_therm_agreement,
            ),
        ),
        (
            "feature_rel_drift",
            features_or_null_ok(
                features,
                "all_feature_rel_drift_p99",
                args.max_feature_rel_drift_p99,
            ),
        ),
        (
            "spike_count_rel_drift",
            !features["has_spike_count_signal"]
                .as_bool()
                .unwrap_or(false)
                || features_or_null_ok(
                    features,
                    "spike_count_rel_drift_max",
                    args.max_spike_count_rel_drift,
                ),
        ),
    ]);
    json!({
        "accepted_for_epsilon_budget": checks.values().all(|v| *v),
        "checks": checks,
        "note": "Downstream-invariance decision: exact f32 frame-hash drift may be acceptable under an epsilon budget, but frame-count drift is fatal.",
    })
}

fn features_or_null_ok(obj: &Value, key: &str, threshold: f64) -> bool {
    obj.get(key)
        .and_then(Value::as_f64)
        .map(|v| v <= threshold)
        .unwrap_or(true)
}

fn value_or_null_at_least(obj: &Value, key: &str, threshold: f64) -> bool {
    obj.get(key)
        .and_then(Value::as_f64)
        .map(|v| v >= threshold)
        .unwrap_or(true)
}

fn bool_mean(values: &[bool]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().filter(|v| **v).count() as f64 / values.len() as f64
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn jaccard(a: &[String], b: &[String]) -> f64 {
    let aa = a.iter().collect::<HashSet<_>>();
    let bb = b.iter().collect::<HashSet<_>>();
    if aa.is_empty() && bb.is_empty() {
        return 1.0;
    }
    if aa.is_empty() || bb.is_empty() {
        return 0.0;
    }
    aa.intersection(&bb).count() as f64 / aa.union(&bb).count() as f64
}

fn quantile(values: &[f64], q: f64) -> Option<f64> {
    let mut x = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect::<Vec<_>>();
    if x.is_empty() {
        return None;
    }
    x.sort_by(|a, b| a.total_cmp(b));
    if x.len() == 1 {
        return Some(x[0]);
    }
    let pos = q.clamp(0.0, 1.0) * (x.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        Some(x[lo])
    } else {
        Some(x[lo] * (hi as f64 - pos) + x[hi] * (pos - lo as f64))
    }
}

fn optional_f64(v: Option<f64>) -> Value {
    v.filter(|v| v.is_finite())
        .map_or(Value::Null, |v| json!(v))
}

fn scalar_json_display(v: &Value) -> String {
    if v.is_null() {
        "null".to_string()
    } else {
        v.to_string()
    }
}
