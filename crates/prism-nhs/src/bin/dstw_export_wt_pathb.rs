//! PRISM Path-B MD-only materialization -> DSTW WT physics export.
//!
//! This exporter is deliberately separate from `dstw-export-wt`, which consumes
//! the legacy `prism_therm` payload in `<base>.binding_sites.json`. The GLP-1R
//! WT prime runs in this campaign were executed with `--md-only-evidence`, then
//! materialized offline through `prism-materialize-sites`, producing
//! `binding_sites.materialized.json`.
//!
//! The materializer reports residue identifiers in topology-index space. This
//! binary uses the PRISM-PREP residue-map sidecar to convert them back to the
//! receptor residue index before exporting to DSTW. When canonical
//! PRISMOS_BINDING_SITE_PHYSIOLOGY_V1 / PRISMOS_GEOMETRIC_POCKET_MORPHOLOGY_V1
//! blocks are present, their TIDE/PRISM-Therm-compatible fields seed the WT
//! physics profile. Contact edges remain conservative co-membership edges among
//! materialized site lining residues.

use anyhow::{bail, Context, Result};
use arrow_array::{
    builder::{Float64Builder, Int32Builder, StringBuilder},
    ArrayRef, RecordBatch,
};
use arrow_schema::{DataType, Field, Schema};
use clap::Parser;
use parquet::{
    arrow::ArrowWriter,
    basic::{Compression, ZstdLevel},
    file::properties::WriterProperties,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

const PHYSICS_SCHEMA_TAG: &str = "dstw_wt_physical_profile_v1";

#[derive(Parser, Debug)]
#[command(
    name = "dstw-export-wt-pathb",
    about = "Export Path-B materialized MD-only evidence into DSTW WT physics parquets.",
    after_help = "Native SDST saved-evidence mode (GPU build): --native-md-evidence-manifest PATH --help. This mode emits complete native measurements and an SDST checkpoint; it bypasses legacy physics proxies."
)]
struct Args {
    /// Path to binding_sites.materialized.json from prism-materialize-sites.
    #[arg(long)]
    materialized_sites: PathBuf,

    /// PRISM-PREP residue_map.json sidecar for converting topology indices to
    /// receptor residue indices.
    #[arg(long)]
    residue_map: PathBuf,

    /// Optional topology JSON used to compute CA-CA distances for contact
    /// graph edges.
    #[arg(long)]
    topology_json: Option<PathBuf>,

    #[arg(long)]
    target: String,

    #[arg(long)]
    uniprot_accession: String,

    #[arg(long)]
    structure_anchor_id: String,

    #[arg(long)]
    receptor_chain_id: String,

    #[arg(long)]
    prism_run_id: String,

    #[arg(long)]
    pdb_anchor_inactive: Option<String>,

    #[arg(long)]
    pdb_anchor_active: Option<String>,

    #[arg(long)]
    out_json: PathBuf,

    #[arg(long)]
    out_parquet: PathBuf,

    #[arg(long)]
    out_contact_parquet: PathBuf,
}

#[derive(Debug, Clone, Deserialize)]
struct ResidueMapEnvelope {
    residues: Vec<ResidueMapRow>,
}

#[derive(Debug, Clone, Deserialize)]
struct ResidueMapRow {
    topology_index: i32,
    chain: String,
    pdb_resid: i32,
    resname: String,
}

#[derive(Debug, Clone)]
struct MappedResidue {
    residue_index: i32,
    residue_name: String,
    chain_id: String,
}

#[derive(Debug, Default, Clone, Copy)]
struct ResidueAccumulator {
    te_out: f64,
    te_in: f64,
    delta_hc: f64,
    sigma_hydration_sq: f64,
    n_site_appearances: u32,
}

impl ResidueAccumulator {
    fn fold(&mut self, te_out: f64, te_in: f64, delta_hc: f64, sigma: f64) {
        if te_out.is_finite() && te_out > self.te_out {
            self.te_out = te_out;
        }
        if te_in.is_finite() && te_in > self.te_in {
            self.te_in = te_in;
        }
        if delta_hc.is_finite() && delta_hc > self.delta_hc {
            self.delta_hc = delta_hc;
        }
        if sigma.is_finite() && sigma > self.sigma_hydration_sq {
            self.sigma_hydration_sq = sigma;
        }
        self.n_site_appearances = self.n_site_appearances.saturating_add(1);
    }
}

#[derive(Debug, Clone, Serialize)]
struct WTResiduePhysics {
    uniprot_residue_index: i32,
    residue_name: String,
    te_out: f64,
    te_in: f64,
    delta_hc: f64,
    sigma_hydration_sq: f64,
}

#[derive(Debug, Clone, Serialize)]
struct WTDynamicContactEdge {
    source_uniprot_residue_index: i32,
    target_uniprot_residue_index: i32,
    contact_probability: f64,
    mean_distance_angstrom: Option<f64>,
    dynamic_correlation: Option<f64>,
    replicate_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct WTPhysicalProfileResponse {
    target: String,
    uniprot_accession: String,
    pdb_anchor_inactive: Option<String>,
    pdb_anchor_active: Option<String>,
    prism_run_id: String,
    completed_at_utc: String,
    residues: Vec<WTResiduePhysics>,
    dynamic_contact_edges: Vec<WTDynamicContactEdge>,
    physics_schema: &'static str,
}

fn v_f64(v: &Value, path: &[&str]) -> f64 {
    let mut cur = v;
    for key in path {
        cur = match cur.get(*key) {
            Some(next) => next,
            None => return 0.0,
        };
    }
    cur.as_f64().unwrap_or(0.0)
}

fn v_i64(v: &Value, path: &[&str]) -> Option<i64> {
    let mut cur = v;
    for key in path {
        cur = cur.get(*key)?;
    }
    cur.as_i64()
}

fn v_i32_vec(v: &Value, key: &str) -> Vec<i32> {
    v.get(key)
        .and_then(|x| x.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_i64())
                .filter_map(|x| i32::try_from(x).ok())
                .collect()
        })
        .unwrap_or_default()
}

fn load_residue_map(path: &Path, receptor_chain_id: &str) -> Result<BTreeMap<i32, MappedResidue>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read residue map {}", path.display()))?;
    let env: ResidueMapEnvelope = serde_json::from_str(&text)
        .with_context(|| format!("parse residue map {}", path.display()))?;
    let mut out = BTreeMap::new();
    for row in env.residues {
        if row.chain != receptor_chain_id {
            continue;
        }
        if row.pdb_resid < 1 {
            bail!(
                "{}: residue map row has non-positive receptor residue index: {:?}",
                path.display(),
                row.pdb_resid
            );
        }
        out.insert(
            row.topology_index,
            MappedResidue {
                residue_index: row.pdb_resid,
                residue_name: row.resname,
                chain_id: row.chain,
            },
        );
    }
    if out.is_empty() {
        bail!(
            "{}: residue map contains no residues for receptor chain {:?}",
            path.display(),
            receptor_chain_id
        );
    }
    Ok(out)
}

fn load_ca_positions(
    topology_path: Option<&Path>,
    residue_map: &BTreeMap<i32, MappedResidue>,
) -> Result<BTreeMap<i32, [f64; 3]>> {
    let Some(path) = topology_path else {
        return Ok(BTreeMap::new());
    };
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read topology {}", path.display()))?;
    let root: Value = serde_json::from_str(&text)
        .with_context(|| format!("parse topology {}", path.display()))?;
    let positions = root
        .get("positions")
        .and_then(|v| v.as_array())
        .context("topology positions must be an array")?;
    let ca_indices = root
        .get("ca_indices")
        .and_then(|v| v.as_array())
        .context("topology ca_indices must be an array")?;

    let mut out = BTreeMap::new();
    for (topo_idx, atom_idx_value) in ca_indices.iter().enumerate() {
        let Some(mapped) = residue_map.get(&(topo_idx as i32)) else {
            continue;
        };
        let Some(atom_idx) = atom_idx_value.as_u64().map(|x| x as usize) else {
            continue;
        };
        let base = atom_idx * 3;
        if base + 2 >= positions.len() {
            continue;
        }
        let xyz = [
            positions[base].as_f64().unwrap_or(0.0),
            positions[base + 1].as_f64().unwrap_or(0.0),
            positions[base + 2].as_f64().unwrap_or(0.0),
        ];
        out.insert(mapped.residue_index, xyz);
    }
    Ok(out)
}

fn dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn convert_topology_indices(
    topology_indices: Vec<i32>,
    residue_map: &BTreeMap<i32, MappedResidue>,
) -> Vec<i32> {
    let mut seen = BTreeSet::new();
    topology_indices
        .into_iter()
        .filter_map(|idx| residue_map.get(&idx).map(|r| r.residue_index))
        .filter(|idx| seen.insert(*idx))
        .collect()
}

fn load_materialized(path: &Path) -> Result<Value> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read materialized sites {}", path.display()))?;
    let root: Value = serde_json::from_str(&text)
        .with_context(|| format!("parse materialized sites {}", path.display()))?;
    if root.get("schema_kind").and_then(|v| v.as_str()) != Some("pathb_binding_sites_materialized")
    {
        bail!(
            "{}: expected schema_kind pathb_binding_sites_materialized",
            path.display()
        );
    }
    Ok(root)
}

fn aggregate_pathb(
    root: &Value,
    residue_map: &BTreeMap<i32, MappedResidue>,
) -> Result<(
    BTreeMap<i32, ResidueAccumulator>,
    BTreeMap<(i32, i32), (u32, f64)>,
    usize,
)> {
    let mut per_residue: BTreeMap<i32, ResidueAccumulator> = BTreeMap::new();
    for mapped in residue_map.values() {
        per_residue
            .entry(mapped.residue_index)
            .or_insert_with(ResidueAccumulator::default);
    }

    let sites = root
        .get("binding_sites")
        .and_then(|v| v.as_array())
        .context("binding_sites must be an array")?;
    if sites.is_empty() {
        bail!("materialized file has zero binding_sites");
    }

    let mut pair_counts: BTreeMap<(i32, i32), (u32, f64)> = BTreeMap::new();

    for site in sites {
        let lining = convert_topology_indices(v_i32_vec(site, "lining_residues"), residue_map);
        if lining.is_empty() {
            continue;
        }
        let drivers: BTreeSet<i32> =
            convert_topology_indices(v_i32_vec(site, "driver_residues"), residue_map)
                .into_iter()
                .collect();

        let density = v_f64(site, &["score_components", "density_score"]);
        let temporal = v_f64(site, &["score_components", "temporal_persistence_factor"]);
        let stream_balance = v_f64(site, &["score_components", "stream_balance_factor"]);
        let residue_shell = v_f64(
            site,
            &["score_components", "residue_shell_plausibility_factor"],
        );
        let manifold = v_f64(
            site,
            &["score_components", "centroid_manifold_consistency_factor"],
        )
        .max(v_f64(
            site,
            &["manifold_consistency", "manifold_consistency_score"],
        ));
        let kcc_factor = v_f64(site, &["score_components", "kcc_driver_factor"]).max(1.0);
        let kcc_score = v_f64(site, &["kcc_driver", "kcc_driver_score"]);
        let rayleigh = v_f64(site, &["phase_support", "rayleigh_r_stat"]);
        let n_spikes = v_i64(site, &["n_spikes"]).unwrap_or(0).max(0) as f64;
        let spread = v_f64(site, &["centroid_spread_a"]);
        let physiology_te_out = v_f64(
            site,
            &["binding_site_physiology_surface", "tide_te_out_proxy"],
        );
        let physiology_te_in = v_f64(
            site,
            &["binding_site_physiology_surface", "tide_te_in_proxy"],
        );
        let physiology_delta_hc = v_f64(
            site,
            &["binding_site_physiology_surface", "tide_delta_hc_proxy"],
        );
        let physiology_signal = v_f64(
            site,
            &[
                "binding_site_physiology_surface",
                "signal_preservation_factor",
            ],
        );
        let physiology_scale = v_f64(
            site,
            &[
                "binding_site_physiology_surface",
                "scale_normalized_dynamic_support",
            ],
        );
        let morphology_rank =
            v_f64(site, &["pocket_morphology_surface", "morphology_rank"]) / 1000.0;
        let morphology_enclosure = v_f64(
            site,
            &["pocket_morphology_surface", "enclosure_wall_blocking_score"],
        );
        let morphology_access = v_f64(
            site,
            &["pocket_morphology_surface", "mouth_accessibility_proxy"],
        );

        // These are conservative, documented channel lifts from materialized MD
        // evidence. Canonical physiology/morphology fields take precedence when
        // present; legacy-compatible fallbacks remain for older materializations:
        //   te_out   := density × temporal persistence × stream balance,
        //               scaled by log spike support and KCC driver support.
        //   te_in    := residue-shell plausibility × manifold consistency,
        //               i.e. local sensitivity/enclosure support.
        //   delta_hc := phase coherence × manifold consistency.
        //   sigma    := centroid spread variance proxy; downstream EiV paths
        //               can down-weight high-spread sites.
        let spike_scale = n_spikes.ln_1p().max(1.0);
        let fallback_te_out =
            density.max(0.0) * temporal.max(0.0) * stream_balance.max(0.0) * spike_scale;
        let fallback_te_in = residue_shell.max(0.0)
            * manifold.max(0.0)
            * stream_balance.max(0.0)
            * morphology_rank.max(0.0).max(0.5);
        let fallback_delta_hc = rayleigh.max(0.0) * manifold.max(0.0);
        let physiology_scale_factor = if physiology_scale > 0.0 {
            physiology_scale.clamp(0.5, 1.25)
        } else {
            1.0
        };
        let morphology_enclosure_factor = if morphology_enclosure > 0.0 {
            morphology_enclosure.clamp(0.5, 1.25)
        } else {
            1.0
        };
        let morphology_access_factor = if morphology_access > 0.0 {
            (1.0 + (1.0 - morphology_access.clamp(0.0, 1.0))).clamp(1.0, 2.0)
        } else {
            1.0
        };
        let base_te_out = physiology_te_out.max(fallback_te_out) * physiology_scale_factor;
        let base_te_in = physiology_te_in.max(fallback_te_in) * morphology_enclosure_factor;
        let site_delta_hc = physiology_delta_hc
            .max(fallback_delta_hc)
            .max(physiology_signal.max(0.0) * rayleigh.max(0.0));
        let site_sigma = (spread.max(0.0) / 10.0).powi(2) * morphology_access_factor;

        for &rid in &lining {
            let is_driver = drivers.contains(&rid);
            let driver_boost = if is_driver {
                kcc_factor.max(1.0) + kcc_score.max(0.0).ln_1p()
            } else {
                1.0
            };
            let acc = per_residue.entry(rid).or_default();
            acc.fold(
                base_te_out * driver_boost,
                base_te_in * if is_driver { 1.2 } else { 1.0 },
                site_delta_hc,
                site_sigma,
            );
        }

        for i in 0..lining.len() {
            for j in (i + 1)..lining.len() {
                let a = lining[i];
                let b = lining[j];
                if a == b {
                    continue;
                }
                let key = if a < b { (a, b) } else { (b, a) };
                let entry = pair_counts.entry(key).or_insert((0, 0.0));
                entry.0 = entry.0.saturating_add(1);
                entry.1 += rayleigh.clamp(-1.0, 1.0);
            }
        }
    }

    Ok((per_residue, pair_counts, sites.len()))
}

fn build_edges(
    pair_counts: BTreeMap<(i32, i32), (u32, f64)>,
    ca_positions: &BTreeMap<i32, [f64; 3]>,
    replicate_id: String,
) -> Vec<WTDynamicContactEdge> {
    let max_count = pair_counts.values().map(|(n, _)| *n).max().unwrap_or(1) as f64;
    pair_counts
        .into_iter()
        .filter_map(|((a, b), (count, corr_sum))| {
            if a == b || count == 0 {
                return None;
            }
            let mean_distance = match (ca_positions.get(&a), ca_positions.get(&b)) {
                (Some(x), Some(y)) => Some(dist3(*x, *y)),
                _ => None,
            };
            Some(WTDynamicContactEdge {
                source_uniprot_residue_index: a,
                target_uniprot_residue_index: b,
                contact_probability: (count as f64 / max_count).clamp(0.0, 1.0),
                mean_distance_angstrom: mean_distance,
                dynamic_correlation: Some((corr_sum / count as f64).clamp(-1.0, 1.0)),
                replicate_id: Some(replicate_id.clone()),
            })
        })
        .collect()
}

fn iso_8601_utc_now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default();
    let (year, month, day, hour, minute, second) = epoch_to_civil(secs);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

fn epoch_to_civil(secs: u64) -> (i64, u32, u32, u32, u32, u32) {
    let secs_in_day: u64 = 86400;
    let z = (secs / secs_in_day) as i64 + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 {
        (mp + 3) as u32
    } else {
        (mp - 9) as u32
    };
    let year = y + (if m <= 2 { 1 } else { 0 });
    let time = secs % secs_in_day;
    let hour = (time / 3600) as u32;
    let minute = ((time % 3600) / 60) as u32;
    let second = (time % 60) as u32;
    (year, m, d, hour, minute, second)
}

fn build_response(
    args: &Args,
    residue_map: &BTreeMap<i32, MappedResidue>,
    per_residue: &BTreeMap<i32, ResidueAccumulator>,
    edges: Vec<WTDynamicContactEdge>,
) -> WTPhysicalProfileResponse {
    let mut residue_name_by_index = BTreeMap::new();
    for mapped in residue_map.values() {
        residue_name_by_index.insert(mapped.residue_index, mapped.residue_name.clone());
    }
    let residues = per_residue
        .iter()
        .map(|(&idx, acc)| WTResiduePhysics {
            uniprot_residue_index: idx,
            residue_name: residue_name_by_index
                .get(&idx)
                .cloned()
                .unwrap_or_else(|| "UNK".to_string()),
            te_out: acc.te_out,
            te_in: acc.te_in,
            delta_hc: acc.delta_hc,
            sigma_hydration_sq: acc.sigma_hydration_sq,
        })
        .collect();
    WTPhysicalProfileResponse {
        target: args.target.clone(),
        uniprot_accession: args.uniprot_accession.clone(),
        pdb_anchor_inactive: args.pdb_anchor_inactive.clone(),
        pdb_anchor_active: args.pdb_anchor_active.clone(),
        prism_run_id: args.prism_run_id.clone(),
        completed_at_utc: iso_8601_utc_now(),
        residues,
        dynamic_contact_edges: edges,
        physics_schema: PHYSICS_SCHEMA_TAG,
    }
}

fn write_json(response: &WTPhysicalProfileResponse, out: &Path) -> Result<()> {
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let f = File::create(out).with_context(|| format!("create {}", out.display()))?;
    serde_json::to_writer_pretty(f, response)
        .with_context(|| format!("write {}", out.display()))?;
    Ok(())
}

fn write_physics_parquet(
    response: &WTPhysicalProfileResponse,
    residue_map: &BTreeMap<i32, MappedResidue>,
    structure_anchor_id: &str,
    out: &Path,
) -> Result<()> {
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("target", DataType::Utf8, false),
        Field::new("uniprot_accession", DataType::Utf8, false),
        Field::new("uniprot_residue_index", DataType::Int32, false),
        Field::new("residue_name", DataType::Utf8, false),
        Field::new("receptor_chain_id", DataType::Utf8, false),
        Field::new("structure_anchor_id", DataType::Utf8, false),
        Field::new("te_out", DataType::Float64, false),
        Field::new("te_in", DataType::Float64, false),
        Field::new("delta_hc", DataType::Float64, false),
        Field::new("sigma_hydration_sq", DataType::Float64, false),
        Field::new("replicate_id", DataType::Utf8, false),
    ]));
    let mut target_b = StringBuilder::new();
    let mut acc_b = StringBuilder::new();
    let mut idx_b = Int32Builder::with_capacity(response.residues.len());
    let mut name_b = StringBuilder::new();
    let mut chain_b = StringBuilder::new();
    let mut anchor_b = StringBuilder::new();
    let mut te_out_b = Float64Builder::with_capacity(response.residues.len());
    let mut te_in_b = Float64Builder::with_capacity(response.residues.len());
    let mut dh_b = Float64Builder::with_capacity(response.residues.len());
    let mut sigma_b = Float64Builder::with_capacity(response.residues.len());
    let mut rep_b = StringBuilder::new();

    let chain_by_index: BTreeMap<i32, String> = residue_map
        .values()
        .map(|r| (r.residue_index, r.chain_id.clone()))
        .collect();
    for r in &response.residues {
        target_b.append_value(&response.target);
        acc_b.append_value(&response.uniprot_accession);
        idx_b.append_value(r.uniprot_residue_index);
        name_b.append_value(&r.residue_name);
        chain_b.append_value(
            chain_by_index
                .get(&r.uniprot_residue_index)
                .map(String::as_str)
                .unwrap_or(""),
        );
        anchor_b.append_value(structure_anchor_id);
        te_out_b.append_value(r.te_out);
        te_in_b.append_value(r.te_in);
        dh_b.append_value(r.delta_hc);
        sigma_b.append_value(r.sigma_hydration_sq);
        rep_b.append_value(&response.prism_run_id);
    }

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(target_b.finish()) as ArrayRef,
            Arc::new(acc_b.finish()),
            Arc::new(idx_b.finish()),
            Arc::new(name_b.finish()),
            Arc::new(chain_b.finish()),
            Arc::new(anchor_b.finish()),
            Arc::new(te_out_b.finish()),
            Arc::new(te_in_b.finish()),
            Arc::new(dh_b.finish()),
            Arc::new(sigma_b.finish()),
            Arc::new(rep_b.finish()),
        ],
    )?;
    write_one_batch(out, schema, batch)
}

fn write_contact_parquet(
    response: &WTPhysicalProfileResponse,
    structure_anchor_id: &str,
    out: &Path,
) -> Result<()> {
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("structure_anchor_id", DataType::Utf8, false),
        Field::new("source_uniprot_residue_index", DataType::Int32, false),
        Field::new("target_uniprot_residue_index", DataType::Int32, false),
        Field::new("contact_probability", DataType::Float64, false),
        Field::new("mean_distance_angstrom", DataType::Float64, true),
        Field::new("dynamic_correlation", DataType::Float64, true),
        Field::new("replicate_id", DataType::Utf8, true),
    ]));
    let mut anchor_b = StringBuilder::new();
    let mut src_b = Int32Builder::with_capacity(response.dynamic_contact_edges.len());
    let mut dst_b = Int32Builder::with_capacity(response.dynamic_contact_edges.len());
    let mut prob_b = Float64Builder::with_capacity(response.dynamic_contact_edges.len());
    let mut dist_b = Float64Builder::with_capacity(response.dynamic_contact_edges.len());
    let mut corr_b = Float64Builder::with_capacity(response.dynamic_contact_edges.len());
    let mut rep_b = StringBuilder::new();
    for e in &response.dynamic_contact_edges {
        anchor_b.append_value(structure_anchor_id);
        src_b.append_value(e.source_uniprot_residue_index);
        dst_b.append_value(e.target_uniprot_residue_index);
        prob_b.append_value(e.contact_probability);
        match e.mean_distance_angstrom {
            Some(v) => dist_b.append_value(v),
            None => dist_b.append_null(),
        }
        match e.dynamic_correlation {
            Some(v) => corr_b.append_value(v),
            None => corr_b.append_null(),
        }
        match e.replicate_id.as_deref() {
            Some(v) => rep_b.append_value(v),
            None => rep_b.append_null(),
        }
    }
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(anchor_b.finish()) as ArrayRef,
            Arc::new(src_b.finish()),
            Arc::new(dst_b.finish()),
            Arc::new(prob_b.finish()),
            Arc::new(dist_b.finish()),
            Arc::new(corr_b.finish()),
            Arc::new(rep_b.finish()),
        ],
    )?;
    write_one_batch(out, schema, batch)
}

fn write_one_batch(out: &Path, schema: Arc<Schema>, batch: RecordBatch) -> Result<()> {
    let file = File::create(out).with_context(|| format!("create {}", out.display()))?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn main() -> Result<()> {
    if std::env::args().any(|arg| arg == "--native-kcc-winning-pair-time-manifest"
        || arg.starts_with("--native-kcc-winning-pair-time-manifest=")) {
        #[derive(Parser)]
        struct KccPhysicalTimeArgs {
            #[arg(long)] native_kcc_winning_pair_time_manifest: PathBuf,
            #[arg(long)] output_dir: PathBuf,
        }
        let args=KccPhysicalTimeArgs::parse();
        let receipt=prism_nhs::chemical_state_fiber::mechanistic::kcc_time::materialize_recorded_pair_times(
            &args.native_kcc_winning_pair_time_manifest,&args.output_dir)?;
        println!("{}",serde_json::to_string_pretty(&receipt)?);
        return Ok(());
    }

    if std::env::args().any(|arg| arg == "--recorded-measurement-fiber-manifest"
        || arg.starts_with("--recorded-measurement-fiber-manifest=")) {
        #[cfg(any(feature = "gpu", feature = "cpu-analysis"))]
        {
            #[derive(Parser)]
            struct RecordedMeasurementFiberArgs {
                #[arg(long)] recorded_measurement_fiber_manifest: PathBuf,
                #[arg(long)] output_dir: PathBuf,
            }
            let args = RecordedMeasurementFiberArgs::parse();
            let result = prism_nhs::chemical_state_fiber::mechanistic::measurement_fibers::materialize_recorded_measurement_fibers(
                &args.recorded_measurement_fiber_manifest, &args.output_dir)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
            return Ok(());
        }
        #[cfg(not(any(feature = "gpu", feature = "cpu-analysis")))]
        bail!("native recorded measurement fibers require gpu or cpu-analysis feature");
    }
    if std::env::args().any(|arg| arg == "--asc-observed-support-manifest"
        || arg.starts_with("--asc-observed-support-manifest=")) {
        #[derive(Parser)]
        struct ObservedSupportArgs {
            #[arg(long)] asc_observed_support_manifest: PathBuf,
            #[arg(long)] output_dir: PathBuf,
        }
        let args = ObservedSupportArgs::parse();
        let receipt = prism_nhs::asc_support::observed::build_observed_support(
            &args.asc_observed_support_manifest, &args.output_dir)?;
        println!("{}", serde_json::to_string_pretty(&receipt)?);
        return Ok(());
    }
    if std::env::args().any(|arg| arg == "--native-fiber-readout-manifest"
        || arg.starts_with("--native-fiber-readout-manifest=")) {
        #[derive(Parser)]
        struct NativeFiberReadoutArgs {
            #[arg(long)] native_fiber_readout_manifest: PathBuf,
            #[arg(long)] output_dir: PathBuf,
        }
        let args = NativeFiberReadoutArgs::parse();
        let receipt = prism_nhs::chemical_state_fiber::mechanistic::fiber_readout::integrate_native_fiber_readout(
            &args.native_fiber_readout_manifest, &args.output_dir)?;
        println!("{}", serde_json::to_string_pretty(&receipt)?);
        return Ok(());
    }
    // Reuse completed full measurements and geometry without re-evaluation.
    if std::env::args().any(|arg| arg == "--native-mechanistic-readout-artifact"
        || arg.starts_with("--native-mechanistic-readout-artifact=")) {
        #[derive(Parser)]
        struct NativeReadoutArgs {
            #[arg(long)] native_mechanistic_readout_artifact: PathBuf,
            #[arg(long)] output_dir: PathBuf,
        }
        let args = NativeReadoutArgs::parse();
        let receipt = prism_nhs::chemical_state_fiber::mechanistic::readout::evaluate_native_readout(
            &args.native_mechanistic_readout_artifact, &args.output_dir)?;
        println!("{}", serde_json::to_string_pretty(&receipt)?);
        return Ok(());
    }
    // Native measured-event input is consumed by the same chemical-state
    // evaluator used by the observer plane. This route preserves full tensors
    // and exact event identities instead of entering legacy scalar aggregation.
    if std::env::args().any(|arg| arg == "--native-mechanistic-geometry-manifest"
        || arg.starts_with("--native-mechanistic-geometry-manifest=")) {
        #[derive(Parser)]
        struct GeometrySupplementArgs {
            #[arg(long)] native_mechanistic_geometry_manifest: PathBuf,
            #[arg(long)] output_dir: PathBuf,
            /// Parallel full-region validators; scientific checks and output order unchanged.
            #[arg(long, default_value_t=prism_nhs::chemical_state_fiber::mechanistic::geometry::default_geometry_workers())]
            geometry_workers: usize,
        }
        let args = GeometrySupplementArgs::parse();
        let receipt = prism_nhs::chemical_state_fiber::mechanistic::geometry::evaluate_geometry_supplement_with_workers(
            &args.native_mechanistic_geometry_manifest, &args.output_dir,args.geometry_workers)?;
        println!("{}", serde_json::to_string_pretty(&receipt)?);
        return Ok(());
    }
    if std::env::args().any(|arg| arg == "--native-mechanistic-pair-manifest"
        || arg.starts_with("--native-mechanistic-pair-manifest=")) {
        #[derive(Parser)]
        struct MeasuredEventArgs {
            #[arg(long)] native_mechanistic_pair_manifest: PathBuf,
            #[arg(long)] output_dir: PathBuf,
        }
        let args = MeasuredEventArgs::parse();
        let receipt = prism_nhs::chemical_state_fiber::mechanistic::evaluate_mechanistic_pair(
            &args.native_mechanistic_pair_manifest, &args.output_dir)?;
        println!("{}", serde_json::to_string_pretty(&receipt)?);
        return Ok(());
    }
    if std::env::args().any(|arg| arg == "--native-directed-evidence" || arg.starts_with("--native-directed-evidence=")) {
        #[cfg(feature = "gpu")]
        { return native_saved_evidence::run_directed_evidence(); }
        #[cfg(not(feature = "gpu"))]
        { bail!("native directed-evidence export requires the existing GPU-feature exporter build"); }
    }
    // The native saved-evidence mode is an alternate input contract of this
    // existing exporter. It never enters the legacy proxy aggregation branch.
    if std::env::args().any(|arg| {
        arg == "--native-md-evidence-manifest" || arg.starts_with("--native-md-evidence-manifest=")
    }) {
        #[cfg(feature = "gpu")]
        {
            return native_saved_evidence::run();
        }
        #[cfg(not(feature = "gpu"))]
        {
            bail!("native saved-evidence replay requires this existing exporter built with --features gpu");
        }
    }
    let args = Args::parse();
    let residue_map = load_residue_map(&args.residue_map, &args.receptor_chain_id)?;
    let ca_positions = load_ca_positions(args.topology_json.as_deref(), &residue_map)?;
    let materialized = load_materialized(&args.materialized_sites)?;
    let (per_residue, pair_counts, sites_consumed) = aggregate_pathb(&materialized, &residue_map)?;
    let edges = build_edges(
        pair_counts,
        &ca_positions,
        format!("pathb_lining_comembership:{}", args.structure_anchor_id),
    );
    if edges.is_empty() {
        bail!("Path-B export produced no dynamic contact edges");
    }
    let response = build_response(&args, &residue_map, &per_residue, edges);
    write_json(&response, &args.out_json)?;
    write_physics_parquet(
        &response,
        &residue_map,
        &args.structure_anchor_id,
        &args.out_parquet,
    )?;
    write_contact_parquet(
        &response,
        &args.structure_anchor_id,
        &args.out_contact_parquet,
    )?;
    eprintln!(
        "[dstw_export_wt_pathb] target={} anchor={} sites={} residues={} edges={}",
        response.target,
        args.structure_anchor_id,
        sites_consumed,
        response.residues.len(),
        response.dynamic_contact_edges.len(),
    );
    eprintln!("[dstw_export_wt_pathb] wrote {}", args.out_json.display());
    eprintln!(
        "[dstw_export_wt_pathb] wrote {}",
        args.out_parquet.display()
    );
    eprintln!(
        "[dstw_export_wt_pathb] wrote {}",
        args.out_contact_parquet.display()
    );
    Ok(())
}

/// Saved native evidence adapter: no MD, no substitute physical scoring.
#[cfg(feature = "gpu")]
mod native_saved_evidence {
    use super::*;
    use anyhow::ensure;
    use prism_nhs::fused_engine::{CryoUvProtocol, GpuSpikeEvent};
    use prism_nhs::input::PrismPrepTopology;
    use prism_nhs::sdst_bridge::{
        select_native_percentile_indices, LatticeComponentId, MaterializedSiteId, SdstBridge,
        SdstRegionRequest, SdstGridGeometry,
    };
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use std::io::{BufReader, Read, Seek, SeekFrom, Write};

    #[derive(Parser)]
    #[command(
        about = "Replay complete recorded PRSPK streams through native SDST/TIDE; no MD is executed."
    )]
    struct NativeArgs {
        #[arg(long)]
        native_md_evidence_manifest: PathBuf,
        /// raw preserves every recorded event; recorded-native-policy reuses the live per-channel filter.
        #[arg(long, default_value = "raw", value_parser = ["raw", "recorded-native-policy"])]
        native_analysis_population: String,
        /// Explicit WCT parent window in MD steps; omitted preserves native default5. Never auto-tuned.
        #[arg(long)]
        native_wavefront_max_gap_md_steps: Option<u32>,
        #[arg(long)]
        materialized_sites: PathBuf,
        #[arg(long)]
        native_lattice: PathBuf,
        /// Original native Therm selection receipt, if the interrupted run emitted one.
        #[arg(long, conflicts_with = "recover_all_saved_lattice_components")]
        native_therm_status: Option<PathBuf>,
        /// New post-MD analysis request for every saved native lattice component.
        /// This does not reconstruct or assert an original Therm selection/status.
        #[arg(long)]
        recover_all_saved_lattice_components: bool,
        #[arg(long)]
        topology_json: PathBuf,
        /// New output JSON; native checkpoints are retained alongside this file.
        #[arg(long)]
        out_json: PathBuf,
    }

    /// Reuse authenticated native replay artifacts; no GPU graph recomputation or MD.
    #[derive(Parser)]
    struct DirectedArgs {
        #[arg(long)] native_directed_evidence: PathBuf,
        #[arg(long)] condition_edge_manifest: Option<PathBuf>,
        #[arg(long)] out_json: PathBuf,
    }

    #[derive(Clone, Copy)]
    struct DirectedEndpoint {
        ordinal: u64, step: i32, source: i32, residue: i32,
        nearby_count: i32, nearby: [i32;8], wavelength: f32, physical_time_ps: Option<f64>,
    }

    fn authenticated_path(receipt: &Value) -> Result<PathBuf> {
        let path=PathBuf::from(required_str(receipt,"path")?);
        let actual=file_receipt(&path)?;
        ensure!(actual["sha256"]==receipt["sha256"] && actual["size_bytes"]==receipt["size_bytes"],
            "directed evidence source hash/size mismatch: {}",path.display());
        Ok(path)
    }

    fn checked_endpoint_pair(events:&[DirectedEndpoint], source:u32,target:u32,source_step:u32,target_step:u32) -> Result<(DirectedEndpoint,DirectedEndpoint)> {
        let a=*events.get(source as usize).context("source endpoint outside population")?;
        let b=*events.get(target as usize).context("target endpoint outside population")?;
        ensure!(a.step>=0 && b.step>a.step && a.step as u32==source_step && b.step as u32==target_step,
            "directed endpoint identity/time mismatch");
        Ok((a,b))
    }

    pub fn run_directed_evidence() -> Result<()> {
        use arrow_array::{Array, UInt32Array,UInt64Array,Int32Array,Float32Array,Float64Array};
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        let args=DirectedArgs::parse();
        ensure!(!args.out_json.exists(),"refusing to replace native directed output");
        let output_dir=args.out_json.parent().context("output parent required")?;
        std::fs::create_dir_all(output_dir)?;
        let source_receipt=file_receipt(&args.native_directed_evidence)?;
        let mut native=read_json(&args.native_directed_evidence)?;
        let run=required_str(&native,"run_id")?.to_string();
        let target=required_str(&native,"target")?.to_string();
        let controls=if let Some(path)=&args.condition_edge_manifest {
            let v=read_json(path)?;
            ensure!(v["run_id"]==run && v["target"]==target,"condition evidence run/target mismatch");
            json!({"artifact":file_receipt(path)?,"recorded_condition_manifest":v,
                "effect_estimate":null,"matched_control_effect_available":false,
                "interpretation":"condition population/readout is preserved; no matched trajectory/action contrast computed by endpoint join"})
        } else {json!({"availability":"not_supplied","effect_estimate":null,"matched_control_effect_available":false})};
        let streams=native.get_mut("streams").and_then(Value::as_array_mut).context("native stream array missing")?;
        for st in streams.iter_mut() {
            let stream=u32_value(required_u64(st,"stream_id")?)?;
            let expected=required_u64(st,"sdst_ingested_event_count")? as usize;
            let context_path=authenticated_path(&st["native_event_context"]["artifact"])?;
            let edge_path=authenticated_path(&st["native_parent_edges"]["artifact"])?;
            authenticated_path(&st["native_checkpoint"]["artifact"])?;
            authenticated_path(&st["native_region_edge_incidence"]["artifact"])?;
            authenticated_path(&st["spike_receipt"])?;
            let mut endpoints=Vec::with_capacity(expected);
            let context_reader=ParquetRecordBatchReaderBuilder::try_new(File::open(&context_path)?)?.with_batch_size(65536).build()?;
            for batch in context_reader {
                let b=batch?;
                macro_rules! col {($n:expr,$ty:ty)=>{b.column_by_name($n).context(format!("missing {}",$n))?.as_any().downcast_ref::<$ty>().context("endpoint column type")?};}
                let indices=col!("native_event_index",UInt32Array);let streams=col!("stream_id",UInt32Array);
                let ordinals=col!("original_prspk_record_ordinal",UInt64Array);let steps=col!("md_step",Int32Array);
                let sources=col!("spike_source",Int32Array);let residues=col!("aromatic_residue_id",Int32Array);
                let counts=col!("nearby_residue_count",Int32Array);let waves=col!("recorded_wavelength_nm",Float32Array);
                let physical=col!("physical_time_ps",Float64Array);
                for i in 0..b.num_rows() {
                    ensure!(indices.value(i) as usize==endpoints.len() && streams.value(i)==stream,"noncontiguous or cross-stream endpoint identity");
                    let mut nearby=[0;8];for slot in 0..8 {nearby[slot]=col!(&format!("nearby_residue_{slot}"),Int32Array).value(i);}
                    ensure!((0..=8).contains(&counts.value(i)),"invalid endpoint nearby-residue count");
                    endpoints.push(DirectedEndpoint {ordinal:ordinals.value(i),step:steps.value(i),source:sources.value(i),residue:residues.value(i),nearby_count:counts.value(i),nearby,wavelength:waves.value(i),physical_time_ps:(!physical.is_null(i)).then(||physical.value(i))});
                }
            }
            ensure!(endpoints.len()==expected,"directed endpoint population mismatch");
            let clock=&st["native_event_context"]["physical_clock"];
            let exact_clock = clock["mapping"] == "exact_native_per_physical_step_dt_sum";
            if exact_clock {
                use prism_nhs::chemical_state_fiber::mechanistic::physical_time::{PhysicalTimeJournalAuthority,load_physical_time_endpoints};
                let native_manifest_path=authenticated_path(&clock["native_journal_manifest"])?;
                let authority:PhysicalTimeJournalAuthority=serde_json::from_value(clock["native_journal_authority"].clone())?;
                let declared:PhysicalTimeJournalAuthority=serde_json::from_value(read_json(&native_manifest_path)?)?;
                ensure!(authority==declared,"directed native clock authority differs from its pinned manifest");
                ensure!(authority.source_stream_id==stream as u64,"directed native clock stream mismatch");
                let mut requested=endpoints.iter().map(|e|u64::try_from(e.step).context("negative directed step"))
                    .collect::<Result<std::collections::BTreeSet<_>>>()?;
                requested.remove(&0); if requested.is_empty(){requested.insert(1);}
                let times=load_physical_time_endpoints(&authority,&requested)?;
                for e in &endpoints {
                    let t=if e.step==0 {0.0}else{times[&(e.step as u64)].native_record.cumulative_simulated_time_ps};
                    ensure!(e.physical_time_ps==Some(t),"directed context differs from exact native journal time");
                }
            }
            let dt=if clock["physical_time_available"]==true && !exact_clock {
                let d=clock["dt_ps"].as_f64().context("physical dt missing")?;
                ensure!(d.is_finite() && d>0.0,"invalid physical clock");
                authenticated_path(&clock["runtime_log"])?;Some(d)
            } else {None};
            let builder=ParquetRecordBatchReaderBuilder::try_new(File::open(&edge_path)?)?;
            let mut fields=builder.schema().fields().to_vec();
            for side in ["source","target"] {
                for (name,ty) in [("original_prspk_record_ordinal",DataType::UInt64),("spike_source",DataType::Int32),("raw_producer_residue_id",DataType::Int32),("nearby_residue_count",DataType::Int32),("recorded_wavelength_nm",DataType::Float32)] {
                    fields.push(Arc::new(Field::new(format!("{side}_{name}"),ty,false)));
                }
                for slot in 0..8 {fields.push(Arc::new(Field::new(format!("{side}_nearby_residue_{slot}"),DataType::Int32,false)));}
            }
            fields.push(Arc::new(Field::new("lag_physical_ps",DataType::Float64,true)));
            let metadata=std::collections::HashMap::from([
                ("schema_kind".into(),"prism.native_directed_endpoint_correspondence.v1".into()),
                ("native_input_receipt".into(),serde_json::to_string(&source_receipt)?),
                ("source_context_receipt".into(),serde_json::to_string(&st["native_event_context"])?),
                ("parent_edge_receipt".into(),serde_json::to_string(&st["native_parent_edges"])?),
                ("control_support".into(),serde_json::to_string(&controls)?),
                ("direction_semantics".into(),"exact stored algorithmic parent -> child, not intervention attribution or interresidue causal effect".into()),
                ("residue_semantics".into(),"source_1_2_3 raw aromatic residue; source_4_5 displacement residue; nearby slots are neighborhood membership, not directed residue pairs".into())]);
            let schema=Arc::new(Schema::new_with_metadata(fields,metadata));
            let path=output_dir.join(format!("native_sdst_stream{stream}.directed_endpoints.parquet"));
            let (file,partial)=partial_output(&path)?;
            let props=WriterProperties::builder().set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?)).build();
            let mut writer=ArrowWriter::try_new(file,schema.clone(),Some(props))?;
            let mut count=0u64;let mut channel_pairs=BTreeMap::<String,u64>::new();
            let mut lag_counts=BTreeMap::<u32,u64>::new();
            for batch in builder.with_batch_size(65536).build()? {
                let b=batch?;
                macro_rules! u {($n:expr)=>{b.column_by_name($n).context($n)?.as_any().downcast_ref::<UInt32Array>().context("edge column type")?};}
                let mut source_rows=Vec::with_capacity(b.num_rows());let mut target_rows=Vec::with_capacity(b.num_rows());let mut times=Vec::with_capacity(b.num_rows());
                for i in 0..b.num_rows() {
                    ensure!(u!("stream_id").value(i)==stream,"edge stream mismatch");
                    let (a,z)=checked_endpoint_pair(&endpoints,u!("source_event_index").value(i),u!("target_event_index").value(i),u!("source_step").value(i),u!("target_step").value(i))?;
                    let lag=(z.step-a.step) as u32;ensure!(lag==u!("lag_md_steps").value(i),"lag mismatch");
                    *channel_pairs.entry(format!("{}->{}",a.source,z.source)).or_default()+=1;
                    *lag_counts.entry(lag).or_default()+=1;times.push(if exact_clock {
                        let elapsed=z.physical_time_ps.context("target physical time missing")?-a.physical_time_ps.context("source physical time missing")?;
                        ensure!(elapsed.is_finite()&&elapsed>0.0,"directed native physical time order");Some(elapsed)
                    }else{dt.map(|d|lag as f64*d)});source_rows.push(a);target_rows.push(z);
                }
                let mut arrays=b.columns().to_vec();
                for rows in [&source_rows,&target_rows] {
                    arrays.push(Arc::new(UInt64Array::from_iter_values(rows.iter().map(|r|r.ordinal))));
                    arrays.push(Arc::new(Int32Array::from_iter_values(rows.iter().map(|r|r.source))));
                    arrays.push(Arc::new(Int32Array::from_iter_values(rows.iter().map(|r|r.residue))));
                    arrays.push(Arc::new(Int32Array::from_iter_values(rows.iter().map(|r|r.nearby_count))));
                    arrays.push(Arc::new(Float32Array::from_iter_values(rows.iter().map(|r|r.wavelength))));
                    for slot in 0..8 {arrays.push(Arc::new(Int32Array::from_iter_values(rows.iter().map(|r|r.nearby[slot]))));}
                }
                arrays.push(Arc::new(Float64Array::from(times)));count+=b.num_rows() as u64;
                writer.write(&RecordBatch::try_new(schema.clone(),arrays)?)?;
            }
            writer.close()?;
            ensure!(count==required_u64(&st["native_parent_edges"],"parent_edge_count")?,"directed edge population loss");
            publish_partial(&partial,&path)?;
            st["native_directed_mechanism"]=json!({"schema_kind":"prism.native_directed_endpoint_correspondence.v1","artifact":file_receipt(&path)?,"edge_count":count,"event_count":endpoints.len(),"channel_pair_counts":channel_pairs,"lag_histogram_md_steps":lag_counts,"physical_clock":clock,"control_support":controls,"exact_endpoint_context_join":true,"causal_effect_estimate":null,"causal_effect_availability":"not_established_by_algorithmic_parent_link","region_membership":st["native_region_edge_incidence"],"source_record_binding":st["spike_receipt"]});
            eprintln!("[native_directed] stream={stream} exact_endpoints={} exact_edges={count}",endpoints.len());
        }
        native["directed_mechanism_augmentation"]=json!({"producer":"dstw_export_wt_pathb --native-directed-evidence","original_native_payload":source_receipt,"md_executed":false,"native_graph_recomputed":false,"original_numerical_analysis_preserved":true});
        save_json_new(&args.out_json,&native)?;
        Ok(())
    }

    fn read_json(path: &Path) -> Result<Value> {
        serde_json::from_reader(BufReader::new(File::open(path)?))
            .with_context(|| format!("read {}", path.display()))
    }
    fn required_u64(value: &Value, key: &str) -> Result<u64> {
        value
            .get(key)
            .and_then(Value::as_u64)
            .with_context(|| format!("missing unsigned {key}"))
    }
    fn required_str<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
        value
            .get(key)
            .and_then(Value::as_str)
            .with_context(|| format!("missing string {key}"))
    }
    fn u32_value(value: u64) -> Result<u32> {
        Ok(u32::try_from(value)?)
    }
    fn array3(value: &Value) -> Result<[f32; 3]> {
        let a = value.as_array().context("expected three-vector")?;
        ensure!(a.len() == 3, "three-vector length mismatch");
        let mut out = [0.0; 3];
        for i in 0..3 {
            out[i] = a[i].as_f64().context("non-numeric coordinate")? as f32;
            ensure!(out[i].is_finite(), "nonfinite coordinate");
        }
        Ok(out)
    }
    fn file_receipt(path: &Path) -> Result<Value> {
        let mut reader = BufReader::new(File::open(path)?);
        let mut hash = Sha256::new();
        let mut buf = vec![0u8; 1 << 20];
        let mut size = 0u64;
        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hash.update(&buf[..n]);
            size += n as u64;
        }
        Ok(json!({"path":path,"size_bytes":size,"sha256":format!("{:x}",hash.finalize())}))
    }
    fn partial_output(path: &Path) -> Result<(File, PathBuf)> {
        ensure!(!path.exists(), "refusing to replace published artifact {}", path.display());
        let mut name = path.as_os_str().to_os_string(); name.push(".partial");
        let partial = PathBuf::from(name);
        let file = std::fs::OpenOptions::new().write(true).create_new(true).open(&partial)?;
        Ok((file, partial))
    }
    fn publish_partial(partial: &Path, path: &Path) -> Result<()> {
        File::open(partial)?.sync_all()?;
        // Atomic no-clobber publication on the same filesystem. A failed write
        // retains its .partial evidence and never occupies the completed name.
        std::fs::hard_link(partial, path)?;
        std::fs::remove_file(partial)?;
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) { File::open(parent)?.sync_all()?; }
        Ok(())
    }
    fn save_json_new(path: &Path, value: &Value) -> Result<()> {
        let (mut out, partial) = partial_output(path)?;
        serde_json::to_writer_pretty(&mut out, value)?;
        out.write_all(b"\n")?; out.sync_all()?; drop(out);
        publish_partial(&partial, path)
    }

    // Same population Welford accumulation as dstw_export_wt.rs::WaterDensityStats.
    // Unavailable observations remain null, and rejected nonfinite inputs are counted.
    #[derive(Default, Clone)]
    struct WaterStats {
        count: u64,
        nonfinite_count: u64,
        mean: f64,
        m2: f64,
    }
    impl WaterStats {
        fn push(&mut self, value: f32) {
            if !value.is_finite() {
                self.nonfinite_count += 1;
                return;
            }
            self.count += 1;
            let d = value as f64 - self.mean;
            self.mean += d / self.count as f64;
            self.m2 += d * (value as f64 - self.mean);
        }
        fn value(&self) -> Value {
            json!({"count":self.count,"nonfinite_count":self.nonfinite_count,
                "mean":if self.count>0 {Some(self.mean)} else {None},
                "variance":if self.count>0 {Some((self.m2/self.count as f64).max(0.0))} else {None},
                "status":if self.count>0 {"observed"} else {"not_observed"},
                "definition":"population variance of recorded native spike water_density for unique valid nearby topology residues; no explicit-water claim"})
        }
    }
    #[derive(Default, Clone)]
    struct FiniteMax {
        value: Option<f32>,
        count: u64,
        nonfinite_count: u64,
    }
    impl FiniteMax {
        fn push(&mut self, x: f32) {
            if !x.is_finite() {
                self.nonfinite_count += 1;
                return;
            }
            self.count += 1;
            self.value = Some(self.value.map_or(x, |v| v.max(x)));
        }
        fn json(&self) -> Value {
            json!({"value":self.value,"finite_observation_count":self.count,
            "nonfinite_observation_count":self.nonfinite_count,
            "status":if self.value.is_some(){"observed"}else{"not_observed"}})
        }
    }
    #[derive(Default, Clone)]
    struct NativeResidueSummary {
        te: FiniteMax,
        te_variance: FiniteMax,
        asymmetry: FiniteMax,
        supported_regions: u64,
        strict_roi_regions: u64,
    }
    fn f32_record(x: f32) -> Value {
        json!({"value":if x.is_finite(){Some(x)}else{None},"raw_f32_bits":x.to_bits(),
            "status":if x.is_nan(){"nan"}else if x==f32::INFINITY{"positive_infinity"}else if x==f32::NEG_INFINITY{"negative_infinity"}else{"finite"}})
    }

    fn exact<const N: usize>(reader: &mut impl Read) -> Result<[u8; N]> {
        let mut a = [0; N];
        reader.read_exact(&mut a)?;
        Ok(a)
    }
    fn u32_read(reader: &mut impl Read) -> Result<u32> {
        Ok(u32::from_le_bytes(exact(reader)?))
    }
    fn u64_read(reader: &mut impl Read) -> Result<u64> {
        Ok(u64::from_le_bytes(exact(reader)?))
    }
    fn bounded_text(reader: &mut impl Read) -> Result<String> {
        let n = u64_read(reader)?;
        ensure!(n <= 65536, "unbounded PRSPK identity");
        let mut bytes = vec![0; n as usize];
        reader.read_exact(&mut bytes)?;
        Ok(String::from_utf8(bytes)?)
    }
    /// Strict PRSPK001 envelope contract from the existing materializer;
    /// the record decoder is the canonical GpuSpikeEvent decoder, not a mirror.
    fn saved_grid(artifact:&Value,run_id:&str,target:&str,topology:&PrismPrepTopology)->Result<(SdstGridGeometry,Value)> {
        let path=Path::new(required_str(artifact,"path")?);let raw=std::fs::read(path)?;
        ensure!(artifact["present"]==true && raw.len() as u64==required_u64(artifact,"size_bytes")?,"saved grid absent/size mismatch");
        let mut cursor=std::io::Cursor::new(&raw);
        ensure!(exact::<8>(&mut cursor)?==*b"PRSGD001" && u32_read(&mut cursor)?==1 && u32_read(&mut cursor)?==0x01020304,"saved grid header ABI mismatch");
        let sid=u32_read(&mut cursor)?;ensure!(sid as u64==required_u64(artifact,"stream_id")?,"saved grid stream mismatch");
        ensure!(bounded_text(&mut cursor)?==run_id && bounded_text(&mut cursor)?==target,"saved grid run/target mismatch");
        let count=u64_read(&mut cursor)?;let stride=u64_read(&mut cursor)?;let payload=u64_read(&mut cursor)?;let offset=cursor.position() as usize;
        ensure!(raw.len() as u64==offset as u64+payload+16 && payload==16+count*16 && stride==4,"saved grid payload layout mismatch");
        let dim=u64_read(&mut cursor)?;let voxels=u64_read(&mut cursor)?;
        ensure!(dim>0 && dim<=128 && dim*dim*dim==count && voxels==count && count==required_u64(artifact,"record_count")?,"saved grid cube population mismatch");
        let mut fnv=0xcbf29ce484222325u64;
        for byte in &raw[offset..offset+payload as usize] {fnv=(fnv^*byte as u64).wrapping_mul(0x100000001b3);}
        let mut tail=std::io::Cursor::new(&raw[offset+payload as usize..]);
        ensure!(u64_read(&mut tail)?==fnv && exact::<8>(&mut tail)?==*b"PRSGDEND","saved grid checksum/trailer mismatch");
        ensure!(format!("{fnv:016x}")==required_str(artifact,"checksum_fnv1a_64_hex")?.to_ascii_lowercase(),"saved grid manifest checksum mismatch");
        let packet=PathBuf::from(format!("{}.prismqnanl",path.display()));
        let packet_bytes=std::fs::read(&packet).context("canonical saved signal-grid packet required")?;
        let decoded=prismcode::qnanl_artifact::decode_packet_bytes(&packet_bytes).map_err(|e|anyhow::anyhow!("saved grid canonical packet: {e}"))?;
        ensure!(decoded==raw,"saved grid raw differs from canonical QNANL payload");
        let geometry=SdstGridGeometry::from_recorded_dimension(topology,dim as u32)?;
        Ok((geometry,json!({"source_signal_grid":file_receipt(path)?,"source_canonical_packet":file_receipt(&packet)?,
            "raw_matches_canonical_payload":true,"geometry":geometry,
            "geometry_producer":"PrismPrepTopology::grid_spacing_for_coverage shared with nhs_rt_full; grid_origin min-5A",
            "source_position_validation":"every original PRSPK event must bit-match native FMA voxel center before CUDA ingestion",
            "residue_map_convention":"existing grid-node/corner distance rule preserved; no half-cell scientific change"})))
    }

    fn read_spikes(
        artifact: &Value,
        run_id: &str,
        target: &str,
        nres: usize,
        geometry:&SdstGridGeometry,
    ) -> Result<(Vec<GpuSpikeEvent>, Vec<WaterStats>, Value)> {
        let path = Path::new(required_str(artifact, "path")?);
        ensure!(
            artifact.get("present") == Some(&Value::Bool(true)),
            "spike manifest marks absent"
        );
        let sid = u32_value(required_u64(artifact, "stream_id")?)?;
        let count = required_u64(artifact, "record_count")?;
        ensure!(count>0 && count<=42_000_000,"native replay requires complete population within current SDST capacity; no stride sampling");
        let mut r = BufReader::with_capacity(1 << 20, File::open(path)?);
        ensure!(exact::<8>(&mut r)? == *b"PRSPK001", "invalid PRSPK magic");
        ensure!(u32_read(&mut r)? == 1, "unsupported PRSPK schema");
        ensure!(u32_read(&mut r)? == 0x01020304, "invalid PRSPK endian");
        ensure!(u32_read(&mut r)? == sid, "stream identity mismatch");
        let actual_run = bounded_text(&mut r)?;
        let actual_target = bounded_text(&mut r)?;
        ensure!(
            actual_run == run_id && actual_target == target,
            "PRSPK run/target mismatch"
        );
        ensure!(u64_read(&mut r)? == count, "PRSPK count mismatch");
        let stride = u64_read(&mut r)?;
        ensure!(
            stride == std::mem::size_of::<GpuSpikeEvent>() as u64 && stride == 96,
            "PRSPK ABI mismatch"
        );
        let payload = u64_read(&mut r)?;
        ensure!(
            payload == count.checked_mul(stride).context("payload overflow")?,
            "PRSPK payload size mismatch"
        );
        let total = 60 + actual_run.len() as u64 + actual_target.len() as u64 + payload + 16;
        ensure!(
            std::fs::metadata(path)?.len() == total
                && required_u64(artifact, "size_bytes")? == total,
            "PRSPK file size mismatch"
        );
        let mut fnv = 0xcbf29ce484222325u64;
        let mut spikes = Vec::with_capacity(count as usize);
        let mut water = vec![WaterStats::default(); nres];
        let mut step_min = i32::MAX;
        let mut step_max = i32::MIN;
        for _ in 0..count {
            let bytes = exact::<96>(&mut r)?;
            for b in bytes {
                fnv = (fnv ^ b as u64).wrapping_mul(0x100000001b3);
            }
            let spike = GpuSpikeEvent::from_le_bytes_with_stride(&bytes, 96)?;
            ensure!(
                spike.timestep >= 0 && (0..=8).contains(&spike.n_residues),
                "invalid native spike time/residue cardinality"
            );
            geometry.validate_recorded_center(&spike)?;
            for i in 0..spike.n_residues as usize {
                let resid = spike.nearby_residues[i];
                ensure!(
                    resid >= 0 && (resid as usize) < nres,
                    "native nearby residue out of topology domain"
                );
                if !spike.nearby_residues[..i].contains(&resid) {
                    water[resid as usize].push(spike.water_density);
                }
            }
            step_min = step_min.min(spike.timestep);
            step_max = step_max.max(spike.timestep);
            spikes.push(spike);
        }
        ensure!(u64_read(&mut r)? == fnv, "PRSPK payload checksum mismatch");
        ensure!(
            format!("{fnv:016x}")
                == required_str(artifact, "checksum_fnv1a_64_hex")?.to_ascii_lowercase(),
            "manifest checksum mismatch"
        );
        ensure!(exact::<8>(&mut r)? == *b"PRSPKEND", "invalid PRSPK trailer");
        let mut eof = [0; 1];
        ensure!(r.read(&mut eof)? == 0, "PRSPK trailing bytes");
        let mut receipt = file_receipt(path)?;
        receipt["stream_id"] = json!(sid);
        receipt["record_count"] = json!(count);
        receipt["payload_fnv1a_64"] = json!(format!("{fnv:016x}"));
        receipt["step_range"] = json!([step_min, step_max]);
        receipt["record_decoder"] = json!("GpuSpikeEvent::from_le_bytes_with_stride");
        Ok((spikes, water, receipt))
    }

    /// Direct transcription of SDST's existing Morton voxel encoding. This is
    /// coordinate decoding, not an inferred edge or a new dynamics calculation.
    fn morton_coordinates(code: u32) -> [u32; 3] {
        let mut xyz = [0u32; 3];
        for bit in 0..10 {
            for axis in 0..3 {
                xyz[axis] |= ((code >> (3 * bit + axis)) & 1) << bit;
            }
        }
        xyz
    }
    fn native_event(reader: &mut impl Read) -> Result<sdst::SpikeEvent> {
        ensure!(
            std::mem::size_of::<sdst::SpikeEvent>() == 36,
            "native SDST event ABI changed"
        );
        let bytes = exact::<36>(reader)?;
        // SpikeEvent is the existing repr(C) FFI record, contains only integer
        // fields, and all bit patterns are valid. File bytes need no alignment.
        Ok(unsafe { std::ptr::read_unaligned(bytes.as_ptr().cast::<sdst::SpikeEvent>()) })
    }
    fn edge_batch(
        writer: &mut ArrowWriter<File>,
        schema: &Arc<Schema>,
        columns: &mut [Vec<u32>],
    ) -> Result<()> {
        if columns[0].is_empty() {
            return Ok(());
        }
        let arrays = columns
            .iter_mut()
            .map(|c| Arc::new(arrow_array::UInt32Array::from(std::mem::take(c))) as ArrayRef)
            .collect();
        writer.write(&RecordBatch::try_new(schema.clone(), arrays)?)?;
        Ok(())
    }
    /// Extract every stored nonzero parent reference without reconstructing a
    /// graph or calling the defective legacy subgraph local-index API.
    fn export_native_parent_edges(
        checkpoint: &Path,
        metadata: &Value,
        output: &Path,
        stream: u32,
    ) -> Result<Value> {
        ensure!(
            cfg!(target_endian = "little"),
            "native checkpoint reader requires witnessed little-endian host ABI"
        );
        let mut reader = BufReader::with_capacity(1 << 20, File::open(checkpoint)?);
        ensure!(
            u32_read(&mut reader)? == 0x54534453 && u32_read(&mut reader)? == 1,
            "SDST checkpoint header mismatch"
        );
        let config_size = std::mem::size_of::<sdst::SdstConfig>();
        ensure!(
            required_u64(metadata, "config_size_bytes")? == config_size as u64,
            "SDST config ABI mismatch"
        );
        let mut config_bytes = vec![0u8; config_size];
        reader.read_exact(&mut config_bytes)?;
        // Existing C config consists solely of 32-bit integer/float fields.
        let config =
            unsafe { std::ptr::read_unaligned(config_bytes.as_ptr().cast::<sdst::SdstConfig>()) };
        ensure!(config.grid_nx>0 && config.grid_nx<=128 && config.grid_ny==config.grid_nx && config.grid_nz==config.grid_nx,
            "native checkpoint cubic grid mismatch");
        ensure!(metadata["configuration"]["grid"]==json!([config.grid_nx,config.grid_ny,config.grid_nz])
            && metadata["configuration"]["grid_spacing_a"].as_f64()==Some(config.grid_spacing as f64),"native checkpoint geometry differs from producer receipt");
        let count = u32_read(&mut reader)?;
        ensure!(
            count as u64 == required_u64(metadata, "event_count")?,
            "native checkpoint event count mismatch"
        );
        let offset = 12 + config_size as u64;
        ensure!(
            offset == required_u64(metadata, "event_buffer_offset")?,
            "native checkpoint offset mismatch"
        );
        ensure!(
            std::fs::metadata(checkpoint)?.len() >= offset + count as u64 * 36,
            "truncated native event buffer"
        );
        let mut times = Vec::with_capacity(count as usize);
        let mut voxels = Vec::with_capacity(count as usize);
        let mut phases = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let e = native_event(&mut reader)?;
            times.push(e.timestamp);
            voxels.push(e.voxel);
            phases.push(e.phase_id);
        }
        reader.seek(SeekFrom::Start(offset))?;
        let fields = [
            "stream_id",
            "source_event_index",
            "target_event_index",
            "target_parent_spike_encoded",
            "source_step",
            "target_step",
            "lag_md_steps",
            "source_morton_voxel",
            "target_morton_voxel",
            "source_voxel_x",
            "source_voxel_y",
            "source_voxel_z",
            "target_voxel_x",
            "target_voxel_y",
            "target_voxel_z",
            "source_phase_id",
            "target_phase_id",
            "target_avalanche_id",
            "target_wavefront_id",
            "target_tcl_flags",
            "target_amplitude_raw_u16",
            "target_local_temp_raw_u16",
            "target_energy_gradient_raw_u16",
            "target_solvent_exposure_raw_u16",
            "target_wavefront_velocity_raw_u16",
            "target_wavefront_coherence_raw_u16",
        ];
        let mut schema_metadata = std::collections::HashMap::new();
        schema_metadata.insert(
            "prism.schema_kind".to_string(),
            "prism.native_sdst_parent_edges.v1".to_string(),
        );
        schema_metadata.insert(
            "prism.source_checkpoint".to_string(),
            checkpoint.display().to_string(),
        );
        schema_metadata.insert(
            "prism.source_checkpoint_sha256".to_string(),
            metadata
                .pointer("/artifact/sha256")
                .and_then(Value::as_str)
                .context("checkpoint SHA256 missing")?
                .to_string(),
        );
        schema_metadata.insert("prism.edge_semantics".to_string(),"exact GPU SDST stored parent reference; source is parent and target is child; global zero-based event-buffer identities".to_string());
        let schema = Arc::new(Schema::new_with_metadata(
            fields
                .iter()
                .map(|name| Field::new(*name, DataType::UInt32, false))
                .collect::<Vec<_>>(),
            schema_metadata,
        ));
        let partial = output.with_extension("parquet.partial");
        ensure!(!output.exists(), "refusing to replace parent-edge output");
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&partial)?;
        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
            .set_max_row_group_size(65536)
            .build();
        let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;
        let mut columns: Vec<Vec<u32>> = (0..fields.len())
            .map(|_| Vec::with_capacity(65536))
            .collect();
        let mut edges = 0u64;
        let mut roots = 0u64;
        let mut max_lag = 0u32;
        let mut max_distance_sq = 0u32;
        let mut effective_spatial_rule_violations = 0u64;
        let mut configured_time_rule_violations = 0u64;
        let mut lag_histogram = std::collections::BTreeMap::<u32,u64>::new();
        let mut wavefront_time_eligible = 0u64;
        let mut wavefront_time_and_space_eligible = 0u64;
        for child in 0..count {
            let e = native_event(&mut reader)?;
            if e.parent_spike == 0 {
                roots += 1;
                continue;
            }
            ensure!(
                e.parent_spike <= count,
                "parent reference {} outside native event population at child {child}",
                e.parent_spike
            );
            let parent = e.parent_spike - 1;
            let parent_time = times[parent as usize];
            ensure!(parent_time<e.timestamp,"native parent not strictly earlier: parent={parent},child={child},times={parent_time}/{}",e.timestamp);
            let lag = e.timestamp - parent_time;
            max_lag = max_lag.max(lag);
            let source = morton_coordinates(voxels[parent as usize]);
            let target = morton_coordinates(e.voxel);
            let distance_sq = (0..3)
                .map(|a| {
                    let d = source[a] as i64 - target[a] as i64;
                    (d * d) as u32
                })
                .sum::<u32>();
            max_distance_sq = max_distance_sq.max(distance_sq);
            *lag_histogram.entry(lag).or_default() += 1;
            if lag <= config.wavefront_max_dt as u32 {
                wavefront_time_eligible += 1;
                if (distance_sq as f32).sqrt() * config.grid_spacing <= config.wavefront_merge_dist {
                    wavefront_time_and_space_eligible += 1;
                }
            }
            if distance_sq == 0 || distance_sq > 9 {
                effective_spatial_rule_violations += 1;
            }
            if lag > config.avalanche_max_gap {
                configured_time_rule_violations += 1;
            }
            let values = [
                stream,
                parent,
                child,
                e.parent_spike,
                parent_time,
                e.timestamp,
                lag,
                voxels[parent as usize],
                e.voxel,
                source[0],
                source[1],
                source[2],
                target[0],
                target[1],
                target[2],
                phases[parent as usize] as u32,
                e.phase_id as u32,
                e.avalanche_id,
                e.wavefront_id,
                e.tcl_flags as u32,
                e.amplitude as u32,
                e.local_temp as u32,
                e.energy_gradient as u32,
                e.solvent_exposure as u32,
                e.wavefront_velocity as u32,
                e.wavefront_coherence as u32,
            ];
            for (column, value) in columns.iter_mut().zip(values) {
                column.push(value);
            }
            edges += 1;
            if columns[0].len() == 65536 {
                edge_batch(&mut writer, &schema, &mut columns)?;
            }
        }
        edge_batch(&mut writer, &schema, &mut columns)?;
        writer.close()?;
        ensure!(
            edges + roots == count as u64,
            "native graph population conservation failed"
        );
        publish_partial(&partial, output)?;
        Ok(
            json!({"schema_kind":"prism.native_sdst_parent_edges.v1","artifact":file_receipt(output)?,
            "checkpoint_node_count":count,"parent_edge_count":edges,"root_node_count":roots,
            "parent_lag_histogram_md_steps":lag_histogram,
            "wavefront_eligibility":{"max_dt_md_steps":config.wavefront_max_dt,"merge_distance_a":config.wavefront_merge_dist,
                "time_eligible_parent_edges":wavefront_time_eligible,"time_and_space_eligible_parent_edges":wavefront_time_and_space_eligible,
                "time_eligible_fraction":if edges>0{Some(wavefront_time_eligible as f64/edges as f64)}else{None},
                "all_parent_links_preserved_independently":true,"automatic_window_tuning":false},
            "all_nodes_preserved_in_checkpoint":true,"all_nonzero_parent_references_exported":true,
            "parent_reference_bounds_pass":true,"strict_parent_time_order_pass":true,
            "max_observed_lag_md_steps":max_lag,"max_observed_distance_squared_voxels":max_distance_sq,
            "effective_spatial_rule_violations":effective_spatial_rule_violations,"configured_time_rule_violations":configured_time_rule_violations,
            "event_identity":"unfiltered native checkpoint row index, zero based; stored parent encoding retained separately",
            "lag_semantics":"target timestamp minus stored parent timestamp in native MD steps; no assumed uniform physical-time conversion",
            "raw_u16_semantics":"original SDST packed event fields; no invented physical calibration"}),
        )
    }

    struct PhysicalEventClock {
        fixed_dt: Option<f64>,
        exact: Option<BTreeMap<u64, f64>>,
        exact_fs: Option<BTreeMap<u64,u64>>,
    }
    impl PhysicalEventClock {
        fn time(&self, step: u64) -> Result<Option<f64>> {
            if let Some(axis)=&self.exact_fs {
                let fs=if step==0 {0} else {*axis.get(&step).context("event step missing from native integer-fs clock")?};
                ensure!(fs<=1u64<<53,"integer-fs clock exceeds exact f64 integer representation");
                return Ok(Some(fs as f64*0.001));
            }
            if let Some(axis) = &self.exact {
                if step == 0 { return Ok(Some(0.0)); }
                return Ok(Some(*axis.get(&step).context("event step missing from exact native physical clock")?));
            }
            Ok(self.fixed_dt.map(|dt| step as f64 * dt))
        }
        fn lag(&self, source: u64, target: u64) -> Result<Option<f64>> {
            ensure!(target > source, "physical edge endpoints not ordered");
            if let Some(axis)=&self.exact_fs {
                let a=if source==0 {0} else {*axis.get(&source).context("source endpoint missing from integer-fs clock")?};
                let b=*axis.get(&target).context("target endpoint missing from integer-fs clock")?;
                let fs=b.checked_sub(a).context("integer-fs endpoints reversed")?;
                ensure!(fs>0 && fs<=1u64<<53,"invalid integer-fs edge duration");
                return Ok(Some(fs as f64*0.001));
            }
            if self.exact.is_some() {
                let dt = self.time(target)?.unwrap() - self.time(source)?.unwrap();
                ensure!(dt.is_finite() && dt > 0.0, "native physical edge time not ordered");
                Ok(Some(dt))
            } else { Ok(self.fixed_dt.map(|dt| (target-source) as f64 * dt)) }
        }
    }
    fn integer_fs_axis(raw:&[u8], requested:&BTreeSet<u64>) -> Result<Option<BTreeMap<u64,u64>>> {
        let mut axis=BTreeMap::new();let mut previous=0u64;let mut rows=0u64;let mut absent=0u64;
        for line in raw.split(|b|*b==b'\n').filter(|l|!l.is_empty()) {
            let row:Value=serde_json::from_slice(line)?;rows+=1;
            match (row.get("cumulative_physical_time_fs"),row.get("applied_dt_fs")) {
                (None,None)=>{absent+=1;},
                (Some(total),Some(dt))=>{
                    ensure!(row["time_authority"]=="cumulative_physical_time_fs","integer-fs fields lack recorded authority");
                    let step=required_u64(&row,"step")?;let fs=total.as_u64().context("invalid cumulative integer fs")?;
                    let dt=dt.as_u64().context("invalid applied integer fs")?;
                    ensure!(step==rows && dt>0 && previous.checked_add(dt)==Some(fs),"native integer-fs population/time conservation failed");
                    previous=fs;if requested.contains(&step) {axis.insert(step,fs);}
                },
                _=>bail!("partial native integer-fs field pair"),
            }
        }
        if absent==rows {return Ok(None);}
        ensure!(absent==0 && axis.len()==requested.len(),"missing integer-fs event endpoints");
        Ok(Some(axis))
    }

    fn native_time_mapping(manifest_path: &Path, manifest: &Value, protocol: &Value,
        stream: u32, spikes: &[GpuSpikeEvent]) -> Result<(PhysicalEventClock, Value)> {
        use prism_nhs::chemical_state_fiber::mechanistic::physical_time::{PhysicalTimeJournalAuthority, load_physical_time_endpoints};
        let journals: Vec<_> = manifest["artifacts"].as_array().context("native evidence artifact array")?.iter()
            .filter(|a| a["kind"] == "physical_time_journal_manifest" && a["stream_id"] == stream).collect();
        ensure!(journals.len() <= 1, "duplicate native physical time journal authority");
        if let Some(artifact) = journals.first() {
            let (saved, receipt) = verified_protocol(artifact)?;
            let authority: PhysicalTimeJournalAuthority = serde_json::from_value(saved.clone())?;
            ensure!(authority.source_stream_id == stream as u64, "native clock stream differs");
            let mut requested = spikes.iter().map(|s| u64::try_from(s.timestep)
                .context("negative native event time")).collect::<Result<std::collections::BTreeSet<_>>>()?;
            requested.remove(&0);
            if requested.is_empty() { requested.insert(1); }
            let endpoints = load_physical_time_endpoints(&authority, &requested)?;
            let exact = endpoints.iter().map(|(step,e)| (*step,e.native_record.cumulative_simulated_time_ps)).collect();
            let raw=std::fs::read(&authority.source.path)?;
            ensure!(format!("{:x}",Sha256::digest(&raw))==authority.source.sha256,"physical-time bytes changed after native verification");
            let exact_fs=integer_fs_axis(&raw,&requested)?;let has_integer_fs=exact_fs.is_some();
            return Ok((PhysicalEventClock{fixed_dt:None,exact:Some(exact),exact_fs},json!({
                "physical_time_available":true,"dt_ps":null,"origin":"MD step zero",
                "mapping":if has_integer_fs {"source_pinned_cumulative_integer_fs; subtract endpoints before ps conversion"}else{"recorded_legacy_cumulative_simulated_time_ps; authoritative integer-fs fields absent"},
                "integer_fs_authority_consumed":has_integer_fs,
                "legacy_ps_presentation_retained_in_source_journal":true,
                "native_journal_authority":authority,"native_journal_manifest":receipt,
                "retained_event_step_count":endpoints.len(),"journal_full_population_verified":true,
                "lag_mapping":if has_integer_fs {"exact integer-fs target minus source, then scale0.001 to ps"}else{"recorded legacy cumulative ps target minus source; no exact integer-fs claim"}})));
        }
        let (fixed_dt,receipt) = fixed_time_mapping(manifest_path,manifest,protocol,stream)?;
        Ok((PhysicalEventClock{fixed_dt,exact:None,exact_fs:None},receipt))
    }

    fn fixed_time_mapping(manifest_path: &Path, manifest: &Value, protocol: &Value, stream: u32) -> Result<(Option<f64>, Value)> {
        let dt = protocol.get("dt_ps").and_then(Value::as_f64).filter(|v| v.is_finite() && *v > 0.0);
        let root = manifest_path.parent().and_then(Path::parent).context("manifest run root")?;
        let log_path = root.join("run.log");
        let text = std::fs::read_to_string(&log_path).unwrap_or_default();
        let fixed = manifest.pointer("/run_config/adaptive_dt") == Some(&Value::Bool(false))
            && text.lines().any(|l| l.contains(&format!("[PHYSGOV stream {stream}] DISARMED (--gearbox off)")))
            && text.lines().any(|l| l.contains(&format!("PHYSICAL_GUARD_CHRONOMETRY source={stream} ")) && l.contains("director_adaptive_dt=0 host_legacy_dt_authorized=false"));
        let receipt = json!({"physical_time_available":fixed && dt.is_some(),
            "dt_ps":if fixed {dt} else {None},"origin":"MD step zero",
            "mapping":"signed_md_step_delta_times_verified_fixed_dt",
            "protocol_dt_ps":dt,"manifest_adaptive_dt":manifest.pointer("/run_config/adaptive_dt"),
            "runtime_fixed_step_guard_witness":fixed,
            "runtime_log":if log_path.exists(){file_receipt(&log_path)?}else{Value::Null}});
        Ok((if fixed {dt} else {None},receipt))
    }

    /// Emits the same ordered PRSPK records uploaded to CUDA. The native C sort
    /// operates on auxiliary keys, so event-buffer row identity is unchanged.
    fn export_event_context(path: &Path, ordered: &[GpuSpikeEvent], ordinals: &[u64], stream: u32,
                            protocol: &CryoUvProtocol, time_axis: &PhysicalEventClock, clock: &Value,
                            source_receipt: &Value, source_contract: &Value) -> Result<Value> {
        use arrow_array::{BinaryArray, BooleanArray, Float32Array, Float64Array, Int32Array, UInt32Array, UInt64Array};
        ensure!(!path.exists() && ordered.len()==ordinals.len(),"event-context identity collision");
        ensure!(cfg!(target_endian="little") && std::mem::size_of::<GpuSpikeEvent>()==96,"PRSPK event ABI mismatch");
        let channel_semantics = json!({
            "raw_record_bytes":"GpuSpikeEvent::to_le_bytes; exact source record values, no native padding read",
            "source_codes":{"1":"UV","2":"LIF","3":"EFP","4":"LADD","5":"COFIRE"},
            "source_4_5":{"aromatic_residue_id":"max_displacement_residue_not_aromatic_assignment",
                "water_density":"heavy_atom_density","vibrational_energy":"displacement_coherence",
                "n_nearby_excited":"adjacent_depleted_voxel_count","wd_change":"departed_atom_count",
                "phase_bits_origin":if source_contract["schema_kind"]=="prism.spike_source_fields.v2"
                    && source_contract["source_4_5_phase_bits"]=="device_ProtocolState.current_phase_bits" {
                    "declared_device_ProtocolState.current_phase_bits"
                } else { "legacy_unassigned_bytes; producer_did_not_assign_phase; raw_value_preserved" }},
            "source_1_2_3_phase_bits":"device_ProtocolState.current_phase_bits",
            "producer_field_contract":source_contract});
        let mut fields=vec![Field::new("native_event_index",DataType::UInt32,false),Field::new("original_prspk_record_ordinal",DataType::UInt64,false),Field::new("stream_id",DataType::UInt32,false),Field::new("md_step",DataType::Int32,false),Field::new("voxel_linear_index",DataType::Int32,false),Field::new("spike_source",DataType::Int32,false),Field::new("aromatic_type",DataType::Int32,false),Field::new("aromatic_residue_id",DataType::Int32,false),Field::new("nearby_residue_count",DataType::Int32,false),Field::new("phase_bits",DataType::UInt32,false),Field::new("recorded_wavelength_nm",DataType::Float32,false),Field::new("recorded_vibrational_energy",DataType::Float32,false),Field::new("recorded_intensity",DataType::Float32,false),Field::new("recorded_water_density",DataType::Float32,false),Field::new("recorded_nearby_excited_count",DataType::Int32,false),Field::new("scheduled_uv_burst_active",DataType::Boolean,false),Field::new("scheduled_wavelength_nm",DataType::Float32,false),Field::new("physical_time_ps",DataType::Float64,true),Field::new("original_prspk_record_le_bytes",DataType::Binary,false)];
        for i in 0..8 {fields.push(Field::new(format!("nearby_residue_{i}"),DataType::Int32,false));}
        let schema=Arc::new(Schema::new(fields).with_metadata(std::collections::HashMap::from([
            ("schema_kind".into(),"prism.native_event_source_context.v1".into()),
            ("physical_clock".into(),serde_json::to_string(clock)?),
            ("source_receipt".into(),serde_json::to_string(source_receipt)?),
            ("source_channel_semantics".into(),serde_json::to_string(&channel_semantics)?),
            ("identity".into(),"exact CUDA ingestion-vector row; original PRSPK zero-based ordinal retained through channel selection and the single actual permutation".into())])));
        let props=WriterProperties::builder().set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?)).build();
        let (file, partial) = partial_output(path)?;
        let mut writer=ArrowWriter::try_new(file,schema.clone(),Some(props))?;
        for (chunk_index,rows) in ordered.chunks(65536).enumerate() {
            let offset=chunk_index*65536;
            let mut scheduled=protocol.clone();
            let mut burst=Vec::with_capacity(rows.len());let mut wave=Vec::with_capacity(rows.len());
            for row in rows {scheduled.current_step=row.timestep;burst.push(scheduled.is_uv_burst_active());wave.push(scheduled.current_wavelength());}
            let mut arrays:Vec<ArrayRef>=vec![
                Arc::new(UInt32Array::from_iter_values((offset..offset+rows.len()).map(|i|i as u32))),
                Arc::new(UInt64Array::from(ordinals[offset..offset+rows.len()].to_vec())),
                Arc::new(UInt32Array::from(vec![stream;rows.len()])),
                Arc::new(Int32Array::from_iter_values(rows.iter().map(|s|s.timestep))),
                Arc::new(Int32Array::from_iter_values(rows.iter().map(|s|s.voxel_idx))),
                Arc::new(Int32Array::from_iter_values(rows.iter().map(|s|s.spike_source))),
                Arc::new(Int32Array::from_iter_values(rows.iter().map(|s|s.aromatic_type))),
                Arc::new(Int32Array::from_iter_values(rows.iter().map(|s|s.aromatic_residue_id))),
                Arc::new(Int32Array::from_iter_values(rows.iter().map(|s|s.n_residues))),
                Arc::new(UInt32Array::from_iter_values(rows.iter().map(|s|s.phase_bits))),
                Arc::new(Float32Array::from_iter_values(rows.iter().map(|s|s.wavelength_nm))),
                Arc::new(Float32Array::from_iter_values(rows.iter().map(|s|s.vibrational_energy))),
                Arc::new(Float32Array::from_iter_values(rows.iter().map(|s|s.intensity))),
                Arc::new(Float32Array::from_iter_values(rows.iter().map(|s|s.water_density))),
                Arc::new(Int32Array::from_iter_values(rows.iter().map(|s|s.n_nearby_excited))),
                Arc::new(BooleanArray::from(burst)),Arc::new(Float32Array::from(wave)),
                Arc::new(Float64Array::from(rows.iter().map(|s|time_axis.time(u64::try_from(s.timestep).context("negative event physical step")?)).collect::<Result<Vec<_>>>()?)),
                Arc::new(BinaryArray::from_iter_values(rows.iter().map(GpuSpikeEvent::to_le_bytes))),
            ];
            for i in 0..8 {arrays.push(Arc::new(Int32Array::from_iter_values(rows.iter().map(|s|s.nearby_residues[i]))));}
            writer.write(&RecordBatch::try_new(schema.clone(),arrays)?)?;
        }
        writer.close()?;
        publish_partial(&partial,path)?;
        Ok(json!({"source_channel_semantics":channel_semantics,"schema_kind":"prism.native_event_source_context.v1","artifact":file_receipt(path)?,"event_count":ordered.len(),"physical_clock":clock,
            "raw_record_fields_preserved":true,"source_identity":"original_prspk_record_ordinal indexes manifest-bound PRSPK payload, not envelope bytes",
            "scheduled_context_is_measured_action":false,"aromatic_residue_sentinel":-1,"nearby_residue_contract":"use first nearby_residue_count entries only; exact eight raw slots preserved",
            "causal_scope":"native source metadata and algorithmic parent links; intervention-specific response requires matched controller/action records"}))
    }

    fn export_region_edge_incidence(checkpoint:&Path,metadata:&Value,requests:&[(SdstRegionRequest,Value)],path:&Path,stream:u32,time_axis:&PhysicalEventClock,clock:&Value)->Result<Value> {
        use arrow_array::{builder::{ListBuilder,UInt32Builder},Float64Array,Int64Array,UInt32Array};
        ensure!(!path.exists(),"region edge incidence output exists");
        let count=required_u64(metadata,"event_count")? as usize;
        let mut reader=BufReader::with_capacity(1<<20,File::open(checkpoint)?);
        reader.seek(SeekFrom::Start(required_u64(metadata,"event_buffer_offset")?))?;
        let mut events=Vec::with_capacity(count);
        let mut membership=std::collections::HashMap::<u32,Vec<u32>>::new();
        for _ in 0..count {let e=native_event(&mut reader)?;membership.entry(e.voxel).or_insert_with(||{
            let p=morton_coordinates(e.voxel);
            requests.iter().enumerate().filter_map(|(i,(request,_))|{let r=&request.region;
                (p[0]>=r.x_min&&p[0]<=r.x_max&&p[1]>=r.y_min&&p[1]<=r.y_max&&p[2]>=r.z_min&&p[2]<=r.z_max).then_some(i as u32)
            }).collect()});events.push(e);}
        let item=Arc::new(Field::new("item",DataType::UInt32,true));
        let schema=Arc::new(Schema::new(vec![Field::new("stream_id",DataType::UInt32,false),Field::new("source_event_index",DataType::UInt32,false),Field::new("target_event_index",DataType::UInt32,false),Field::new("lag_md_steps_signed",DataType::Int64,false),Field::new("lag_ps_signed",DataType::Float64,true),Field::new("source_region_ordinals",DataType::List(item.clone()),false),Field::new("target_region_ordinals",DataType::List(item),false)]).with_metadata(std::collections::HashMap::from([("schema_kind".into(),"prism.native_region_edge_incidence.v1".into()),("physical_clock".into(),serde_json::to_string(clock)?)])));
        let props=WriterProperties::builder().set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?)).build();
        let (file, partial) = partial_output(path)?;
        let mut writer=ArrowWriter::try_new(file,schema.clone(),Some(props))?;
        let mut total=0u64;let mut incident=0u64;let mut region_counts=vec![0u64;requests.len()];
        for (batch_index,batch) in events.chunks(65536).enumerate() {
            let mut sources=Vec::new();let mut targets=Vec::new();let mut lags=Vec::new();
            let mut source_membership=ListBuilder::new(UInt32Builder::new());let mut target_membership=ListBuilder::new(UInt32Builder::new());
            for (i,e) in batch.iter().enumerate() {if e.parent_spike==0 {continue;}let src=(e.parent_spike-1)as usize;let target=batch_index*65536+i;
                ensure!(src<count,"native source row out of bounds");let lag=e.timestamp as i64-events[src].timestamp as i64;ensure!(lag>0,"parent chronology invalid");
                let a=&membership[&events[src].voxel];let b=&membership[&e.voxel];
                for &r in a {source_membership.values().append_value(r);}source_membership.append(true);
                for &r in b {target_membership.values().append_value(r);}target_membership.append(true);
                if !a.is_empty()||!b.is_empty(){incident+=1;}
                for &r in a {region_counts[r as usize]+=1;}
                for &r in b {if a.binary_search(&r).is_err(){region_counts[r as usize]+=1;}}
                sources.push(src as u32);targets.push(target as u32);lags.push(lag);total+=1;
            }
            if sources.is_empty(){continue;}
            let physical=sources.iter().zip(&targets).map(|(&source,&target)|time_axis.lag(events[source as usize].timestamp as u64, events[target as usize].timestamp as u64)).collect::<Result<Vec<_>>>()?;
            writer.write(&RecordBatch::try_new(schema.clone(),vec![Arc::new(UInt32Array::from(vec![stream;sources.len()])),Arc::new(UInt32Array::from(sources)),Arc::new(UInt32Array::from(targets)),Arc::new(Int64Array::from(lags)),Arc::new(Float64Array::from(physical)),Arc::new(source_membership.finish()),Arc::new(target_membership.finish())])?)?;
        }
        writer.close()?;
        publish_partial(&partial,path)?;
        Ok(json!({"schema_kind":"prism.native_region_edge_incidence.v1","artifact":file_receipt(path)?,"edge_count":total,"edges_incident_to_any_analyzed_region":incident,"physical_clock":clock,
            "membership":"inclusive exact native analysis voxel ROI; source and target memberships separate; overlap retained; no exclusive ownership or invented episode attribution",
            "regions":requests.iter().enumerate().map(|(i,(r,lineage))|json!({"region_ordinal":i,"analyzed_site_identity":r.analyzed_site_identity,"materialized_site_id":r.materialized_site_id,"source_lineage":lineage,"incident_edge_count":region_counts[i],"voxel_region":[r.region.x_min,r.region.x_max,r.region.y_min,r.region.y_max,r.region.z_min,r.region.z_max]})).collect::<Vec<_>>()}))
    }

    fn verified_protocol(artifact: &Value) -> Result<(Value, Value)> {
        let path = Path::new(required_str(artifact, "path")?);
        ensure!(
            artifact.get("present") == Some(&Value::Bool(true)),
            "protocol manifest marks absent"
        );
        let raw = std::fs::read(path)?;
        ensure!(
            raw.len() as u64 == required_u64(artifact, "size_bytes")?,
            "protocol manifest byte count mismatch"
        );
        let mut fnv = 0xcbf29ce484222325u64;
        for byte in &raw {
            fnv = (fnv ^ *byte as u64).wrapping_mul(0x100000001b3);
        }
        let manifest_hash = artifact
            .get("checksum_fnv1a_64_hex")
            .and_then(Value::as_str);
        if let Some(expected) = manifest_hash {
            ensure!(
                format!("{fnv:016x}") == expected.to_ascii_lowercase(),
                "protocol manifest checksum mismatch"
            );
        }
        let packet_path = PathBuf::from(format!("{}.prismqnanl", path.display()));
        let packet_receipt = if packet_path.exists() {
            let packet = std::fs::read(&packet_path)?;
            let decoded = prismcode::qnanl_artifact::decode_packet_bytes(&packet)
                .map_err(|e| anyhow::anyhow!("protocol QNANL validation failed: {e}"))?;
            ensure!(
                decoded == raw,
                "raw protocol differs from canonical sealed QNANL payload"
            );
            Some(file_receipt(&packet_path)?)
        } else {
            None
        };
        ensure!(
            manifest_hash.is_some() || packet_receipt.is_some(),
            "protocol requires manifest checksum or canonical QNANL integrity binding"
        );
        let receipt = json!({"path":path,"size_bytes":raw.len(),"sha256":format!("{:x}",Sha256::digest(&raw)),
            "payload_fnv1a_64":format!("{fnv:016x}"),"manifest_checksum_present":manifest_hash.is_some(),
            "canonical_qnanl_packet":packet_receipt,"raw_matches_canonical_payload":packet_receipt.is_some()});
        Ok((serde_json::from_slice(&raw)?, receipt))
    }

    fn protocol_from_saved(value: &Value, stream: u32) -> Result<CryoUvProtocol> {
        ensure!(
            required_u64(value, "stream_index")? == stream as u64,
            "protocol stream mismatch"
        );
        let b = [
            0,
            required_u64(value, "cold_hold_end")?,
            required_u64(value, "ramp_end")?,
            required_u64(value, "warm_hold_end")?,
            required_u64(value, "ramp_down_end")?,
            required_u64(value, "total_steps")?,
        ];
        ensure!(
            b.windows(2).all(|x| x[1] >= x[0]),
            "non-monotone saved phase boundaries"
        );
        let duration = |i: usize| -> Result<i32> { Ok(i32::try_from(b[i + 1] - b[i])?) };
        let float = |key: &str| -> Result<f32> {
            let x = value
                .get(key)
                .and_then(Value::as_f64)
                .with_context(|| format!("missing protocol {key}"))? as f32;
            ensure!(x.is_finite(), "nonfinite protocol {key}");
            Ok(x)
        };
        let integer = |key: &str| -> Result<i32> { Ok(i32::try_from(required_u64(value, key)?)?) };
        let wavelengths = value
            .get("scan_wavelengths_nm")
            .and_then(Value::as_array)
            .context("missing saved UV channels")?
            .iter()
            .map(|x| {
                let f = x.as_f64().context("invalid wavelength")? as f32;
                ensure!(f.is_finite() && f > 0.0, "invalid wavelength");
                Ok(f)
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            wavelengths.len() == required_u64(value, "n_wavelengths")? as usize,
            "UV channel count mismatch"
        );
        // ProtocolState ABI records piecewise-linear boundaries. A future stepped
        // schedule must supply its own complete contract, never silently flatten it.
        ensure!(
            value
                .get("stepped_holds")
                .map_or(true, |v| v.as_array().is_some_and(|a| a.is_empty())),
            "stepped protocol requires explicit native schedule adapter"
        );
        Ok(CryoUvProtocol {
            start_temp: float("start_temp_K")?,
            end_temp: float("end_temp_K")?,
            cold_hold_steps: duration(0)?,
            ramp_steps: duration(1)?,
            warm_hold_steps: duration(2)?,
            ramp_down_steps: duration(3)?,
            cold_return_steps: duration(4)?,
            current_step: integer("current_step")?,
            uv_burst_energy: float("uv_burst_energy_kcal_mol")?,
            uv_burst_interval: integer("uv_burst_interval_steps")?,
            uv_burst_duration: integer("uv_burst_duration_steps")?,
            scan_wavelengths: wavelengths,
            wavelength_dwell_steps: integer("wavelength_dwell_steps")?,
            stepped_holds: Vec::new(),
        })
    }

    fn lattice_selection_ids(lattice: &Value, status: Option<&Value>) -> Result<Vec<(u64,Option<u32>)>> {
        let components=lattice["components"].as_array().context("missing lattice components")?;
        let mut available=BTreeSet::new();
        for c in components {ensure!(available.insert(required_u64(c,"component_id")?),"duplicate source lattice component");}
        ensure!(!available.is_empty(),"empty saved lattice population");
        let Some(status)=status else {return Ok(available.into_iter().map(|id|(id,None)).collect());};
        let mappings=status["site_mappings"].as_array().context("missing native lattice selection mapping")?;
        ensure!(mappings.len() as u64==required_u64(status,"selected_site_count")?,"native lattice mapping count mismatch");
        let mut seen=BTreeSet::new();let mut out=Vec::new();
        for pair in mappings {
            let pair=pair.as_array().context("invalid site mapping")?;
            ensure!(pair.len()==2 && pair[0]["domain"]=="lattice_component","non-lattice original therm identity");
            let id=required_u64(&pair[0],"id")?;
            ensure!(available.contains(&id) && seen.insert(id),"absent or duplicate selected lattice component");
            out.push((id,Some(u32_value(pair[1].as_u64().context("invalid materialized lattice id")?)?)));
        }
        Ok(out)
    }

    fn requests(
        bridge: &SdstBridge,
        materialized: &Value,
        lattice: &Value,
        status: Option<&Value>,
        stream: u32,
    ) -> Result<Vec<(SdstRegionRequest, Value)>> {
        let mut out = Vec::new();
        let mut ids = BTreeSet::new();
        for bucket in ["binding_sites", "spatiotemporal_partial_candidates"] {
            for site in materialized
                .get(bucket)
                .and_then(Value::as_array)
                .context("missing materialization population")?
            {
                let support = site
                    .pointer("/stream_support/per_stream_spike_counts")
                    .and_then(Value::as_object)
                    .context("missing region stream support")?;
                if support
                    .get(&stream.to_string())
                    .and_then(Value::as_u64)
                    .unwrap_or(0)
                    == 0
                {
                    continue;
                }
                let id = u32_value(required_u64(
                    site.get("provenance").context("missing provenance")?,
                    "region_id",
                )?)?;
                ensure!(ids.insert(id), "duplicate materializer region ID {id}");
                let geometry = site.get("spike_support_geometry").context("materializer must emit attributed spike support geometry before native SDST replay")?;
                let attributed = required_u64(geometry, "attributed_record_count")?;
                ensure!(
                    attributed > 0 && attributed == required_u64(site, "n_spikes")?,
                    "materializer geometry population mismatch"
                );
                ensure!(
                    required_u64(geometry, "nonfinite_position_count")? == 0,
                    "incomplete finite spike support bounds"
                );
                let min = array3(
                    geometry
                        .get("aabb_min")
                        .context("missing observed spike AABB min")?,
                )?;
                let max = array3(
                    geometry
                        .get("aabb_max")
                        .context("missing observed spike AABB max")?,
                )?;
                let request = bridge.materializer_region(id, min, max)?;
                out.push((request,json!({"source_bucket":bucket,"source_site_id":site.get("site_id"),
                    "source_region_id":id,"promoted":site.get("promoted"),
                    "spike_support_geometry":geometry,"source_region_radius_a":site.get("region_radius_a"),
                    "geometry_definition":"observed AABB of every attributed spike position; legacy region_radius_a is centroid spread and is not used as support extent"})));
            }
        }
        let components = lattice
            .get("components")
            .and_then(Value::as_array)
            .context("missing lattice components")?;
        let components: BTreeMap<u64, &Value> = components
            .iter()
            .map(|c| Ok((required_u64(c, "component_id")?, c)))
            .collect::<Result<_>>()?;
        let selected = lattice_selection_ids(lattice, status)?;
        for (id, materialized_id) in selected {
            let c = components
                .get(&id)
                .context("selected lattice component not found")?;
            let nodes = c
                .get("node_lineage")
                .and_then(Value::as_array)
                .context("missing lattice stream lineage")?;
            if !nodes
                .iter()
                .any(|n| n.get("stream_id").and_then(Value::as_u64) == Some(stream as u64))
            {
                continue;
            }
            let request = bridge.lattice_region(
                LatticeComponentId(id),
                materialized_id.map(MaterializedSiteId),
                array3(
                    c.get("component_aabb_min")
                        .context("missing lattice AABB")?,
                )?,
                array3(
                    c.get("component_aabb_max")
                        .context("missing lattice AABB")?,
                )?,
            )?;
            out.push((request,json!({"source_component_id":id,"source_materialized_site_id":materialized_id,
                "source_selection":if status.is_some(){"original native therm_status.site_mappings"}else{"explicit new saved-lattice recovery analysis; all original component identities; no historical selection or binding qualification asserted"},
                "region_geometry_scope":"original lattice bounds used only as the native SDST query ROI; not measured molecular cavity geometry",
                "n_nodes":c.get("n_nodes")})));
        }
        ensure!(!out.is_empty(), "no native regions for stream");
        Ok(out)
    }

    pub fn run() -> Result<()> {
        let args = NativeArgs::parse();
        ensure!(
            !args.out_json.exists(),
            "refusing to replace existing native output"
        );
        let output_dir = args.out_json.parent().context("output parent required")?;
        std::fs::create_dir_all(output_dir)?;
        let manifest = read_json(&args.native_md_evidence_manifest)?;
        let mut policy_receipt = json!({"population":args.native_analysis_population});
        let percentile = if args.native_analysis_population == "recorded-native-policy" {
            let config = manifest
                .get("run_config")
                .context("missing saved run_config")?;
            let percentile = u32_value(required_u64(config, "spike_percentile")?)?;
            let otsu = if let Some(flag) = config.get("filter_otsu").and_then(Value::as_bool) {
                policy_receipt["filter_mode_evidence"] = json!("manifest.run_config.filter_otsu");
                flag
            } else {
                // Historical manifests omitted this boolean. Resolve only from the
                // complete recorded launch command, never the partial replica flag list.
                let launch_path = args
                    .native_md_evidence_manifest
                    .parent()
                    .and_then(Path::parent)
                    .context("manifest run parent required for historical launch")?
                    .join("launch.json");
                let launch = read_json(&launch_path)?;
                let command = launch
                    .get("command")
                    .and_then(Value::as_array)
                    .context("recorded launch command absent")?;
                let argv: Vec<&str> = command
                    .iter()
                    .map(|v| v.as_str().context("invalid launch argv"))
                    .collect::<Result<_>>()?;
                let topology_path = required_str(&manifest, "topology_input")?;
                ensure!(
                    argv.contains(&topology_path),
                    "launch topology identity mismatch"
                );
                let output_path = args
                    .native_md_evidence_manifest
                    .parent()
                    .context("manifest parent")?
                    .to_str()
                    .context("non-UTF8 output path")?;
                ensure!(
                    argv.contains(&output_path),
                    "launch output identity mismatch"
                );
                policy_receipt["historical_launch_receipt"] = file_receipt(&launch_path)?;
                policy_receipt["filter_mode_evidence"] =
                    json!("complete launch argv; filter_otsu is a false-default opt-in flag");
                argv.iter()
                    .any(|arg| *arg == "--filter-otsu" || arg.starts_with("--filter-otsu="))
            };
            ensure!(!otsu, "recorded Otsu policy is not supported by this offline selector; no substitute applied");
            policy_receipt["spike_percentile"] = json!(percentile);
            policy_receipt["filter_otsu"] = json!(false);
            policy_receipt["implementation"] =
                json!("sdst_bridge::select_native_percentile_spikes (shared with nhs_rt_full)");
            Some(percentile)
        } else {
            None
        };
        let materialized = read_json(&args.materialized_sites)?;
        let lattice = read_json(&args.native_lattice)?;
        ensure!(args.native_therm_status.is_some() ^ args.recover_all_saved_lattice_components,
            "supply original --native-therm-status or explicit --recover-all-saved-lattice-components");
        let status=args.native_therm_status.as_ref().map(|p|read_json(p)).transpose()?;
        if let Some(status)=&status {
            ensure!(status.get("schema_kind").and_then(Value::as_str)==Some("prismos.therm_surface_status.v1"),
                "wrong native therm status schema");
        }
        let run_id = required_str(&manifest, "run_id")?;
        let target = required_str(&manifest, "target")?;
        for item in [&materialized, &lattice] {
            ensure!(
                required_str(item, "run_id")? == run_id && required_str(item, "target")? == target,
                "input run/target identity mismatch"
            );
        }
        ensure!(
            std::fs::canonicalize(required_str(&manifest, "topology_input")?)?
                == std::fs::canonicalize(&args.topology_json)?,
            "topology path does not match manifest"
        );
        if let Some(status_path)=&args.native_therm_status {
            let expected_status=args.native_md_evidence_manifest.parent().context("manifest parent")?
                .join(format!("{target}.prism_therm_status.json"));
            ensure!(std::fs::canonicalize(expected_status)?==std::fs::canonicalize(status_path)?,
                "therm status must belong to this manifest run root");
        }
        let lattice_binding = manifest
            .get("ghost_phase_lattice_binding")
            .context("missing native lattice binding")?;
        ensure!(
            std::fs::canonicalize(required_str(lattice_binding, "path")?)?
                == std::fs::canonicalize(&args.native_lattice)?,
            "native lattice path mismatch"
        );
        let lattice_receipt = file_receipt(&args.native_lattice)?;
        ensure!(
            lattice_receipt.get("sha256") == lattice_binding.get("sha256"),
            "native lattice SHA256 mismatch"
        );
        let topology = PrismPrepTopology::load(&args.topology_json)?;
        let mut sources = Vec::new();
        for path in [
            &args.native_md_evidence_manifest,
            &args.materialized_sites,
            &args.native_lattice,
            &args.topology_json,
        ] {
            sources.push(file_receipt(path)?);
        }
        if let Some(path)=&args.native_therm_status {sources.push(file_receipt(path)?);}
        let lattice_selection_provenance=json!({
            "mode":if status.is_some(){"original_native_therm_selection"}else{"new_post_md_all_saved_lattice_components"},
            "historical_therm_status_reconstructed":false,
            "original_therm_status":args.native_therm_status,
            "selected_original_component_count":lattice_selection_ids(&lattice,status.as_ref())?.len(),
            "source_geometry_role":"native query ROI only; coordinate cavity measurements and molecular binding qualification remain separate"});
        let artifacts = manifest
            .get("artifacts")
            .and_then(Value::as_array)
            .context("missing manifest artifacts")?;
        let mut streams = Vec::new();
        let mut stream_ids = BTreeSet::new();
        let mut total_records = 0u64;
        let mut selected_records = 0u64;
        for artifact in artifacts
            .iter()
            .filter(|a| a.get("kind").and_then(Value::as_str) == Some("spikes"))
        {
            let stream = u32_value(required_u64(artifact, "stream_id")?)?;
            ensure!(stream_ids.insert(stream), "duplicate spike stream artifact");
            let protocol_artifacts: Vec<_> = artifacts
                .iter()
                .filter(|a| {
                    a.get("kind").and_then(Value::as_str) == Some("protocol_state")
                        && a.get("stream_id").and_then(Value::as_u64) == Some(stream as u64)
                })
                .collect();
            ensure!(
                protocol_artifacts.len() == 1,
                "each spike stream requires exactly one protocol artifact"
            );
            let (saved_protocol, protocol_receipt) = verified_protocol(protocol_artifacts[0])?;
            let protocol = protocol_from_saved(&saved_protocol, stream)?;
            eprintln!("[native_sdst] decoding complete stream {stream}");
            let grids:Vec<_>=artifacts.iter().filter(|a|a["kind"]=="signal_grid" && a["stream_id"]==stream).collect();
            ensure!(grids.len()==1,"each replay stream requires its exact saved signal-grid envelope");
            let (geometry,grid_receipt)=saved_grid(grids[0],run_id,target,&topology)?;
            let (spikes, water, spike_receipt) =
                read_spikes(artifact, run_id, target, topology.n_residues,&geometry)?;
            let raw_count = spikes.len() as u64;
            total_records += raw_count;
            ensure!(
                spikes.iter().all(|s| s.timestep as u64
                    <= required_u64(&saved_protocol, "total_steps").unwrap_or(0)
                        .max(required_u64(&saved_protocol,"current_step").unwrap_or(0))),
                "spike step exceeds both saved nominal and actual protocol extent"
            );
            let (selected_indices, channel_receipts) = if let Some(percentile) = percentile {
                let (indices, receipts) = select_native_percentile_indices(&spikes, percentile);
                (indices, serde_json::to_value(receipts)?)
            } else {((0..spikes.len()).collect::<Vec<_>>(), Value::Null)};
            let source_ordinals: Vec<u64> = selected_indices.iter().map(|&i|i as u64).collect();
            let selected_spikes: Vec<GpuSpikeEvent> = selected_indices.into_iter().map(|i|spikes[i]).collect();
            drop(spikes);
            let spikes=selected_spikes;
            let count = spikes.len() as u64;
            selected_records += count;
            let mut steps_by_source = std::collections::BTreeMap::<i32,std::collections::BTreeSet<i32>>::new();
            for event in &spikes { steps_by_source.entry(event.spike_source).or_default().insert(event.timestep); }
            let observed_cadence: Vec<_> = steps_by_source.iter().map(|(source,steps)| {
                let sorted: Vec<_> = steps.iter().copied().collect();
                let mut deltas = std::collections::BTreeMap::<i32,u64>::new();
                for pair in sorted.windows(2) { *deltas.entry(pair[1]-pair[0]).or_default() += 1; }
                json!({"spike_source":source,"distinct_observed_steps":sorted.len(),"first_step":sorted.first(),"last_step":sorted.last(),"positive_step_delta_histogram":deltas})
            }).collect();
            let population_receipt = json!({"policy":policy_receipt,"raw_record_count":raw_count,
                "observed_native_event_cadence":observed_cadence,
                "wavefront_window_configuration":{"requested_max_gap_md_steps":args.native_wavefront_max_gap_md_steps,
                    "origin":if args.native_wavefront_max_gap_md_steps.is_some(){"explicit_operator_option"}else{"native_sdst_default_config"},
                    "native_default_literal_md_steps":5,"automatic_window_tuning":false},
                "selected_record_count":count,"excluded_by_recorded_policy_count":raw_count-count,
                "per_channel":channel_receipts,"spike_water_density_population":"raw_prspk_recorded_nearby_residues",
                "native_sdst_tide_ccns_population":args.native_analysis_population,
                "region_geometry_population":"saved materializer full attributed spike AABB and original lattice bounds"});
            save_json_new(
                &output_dir.join(format!("native_sdst_stream{stream}.population.json")),
                &population_receipt,
            )?;
            let checkpoint = output_dir.join(format!("native_sdst_stream{stream}.sdst"));
            ensure!(
                !checkpoint.exists(),
                "refusing to replace native checkpoint"
            );
            let mut bridge = SdstBridge::new_for_spikes_with_recorded_grid(&topology, &protocol, &spikes, args.native_wavefront_max_gap_md_steps,geometry)?;
            bridge.preserve_full_tide_output();
            let region_requests = requests(&bridge, &materialized, &lattice, status.as_ref(), stream)?;
            eprintln!("[native_sdst] stream {stream}: ingesting {count} spikes; {} domain-preserved regions",region_requests.len());
            let (time_axis,clock)=native_time_mapping(&args.native_md_evidence_manifest,&manifest,&saved_protocol,stream,&spikes)?;
            let mut event_context=Value::Null;
            let ingested = bridge.ingest_spikes_with_lineage(&spikes,&source_ordinals,|ordered,ordinals| {
                event_context=export_event_context(&output_dir.join(format!("native_sdst_stream{stream}.event_context.parquet")),ordered,ordinals,stream,&protocol,&time_axis,&clock,&spike_receipt,&manifest["spike_source_field_contract"])?;
                Ok(())
            })?;
            ensure!(
                ingested as u64 == count,
                "native SDST population loss: expected {count}, got {ingested}"
            );
            drop(spikes);
            let mut checkpoint_metadata = bridge.save_native_state(&checkpoint)?;
            checkpoint_metadata["artifact"] = file_receipt(&checkpoint)?;
            save_json_new(
                &checkpoint.with_extension("receipt.json"),
                &checkpoint_metadata,
            )?;
            let parent_edges = export_native_parent_edges(
                &checkpoint,
                &checkpoint_metadata,
                &output_dir.join(format!("native_sdst_stream{stream}.parent_edges.parquet")),
                stream,
            )?;
            let region_edge_incidence=export_region_edge_incidence(&checkpoint,&checkpoint_metadata,&region_requests,
                &output_dir.join(format!("native_sdst_stream{stream}.region_edge_incidence.parquet")),stream,&time_axis,&clock)?;
            ensure!(region_edge_incidence["edge_count"]==parent_edges["parent_edge_count"],"region incidence edge population mismatch");
            eprintln!(
                "[native_sdst] full stored parent-edge table emitted: {} edges",
                parent_edges["parent_edge_count"]
            );
            eprintln!(
                "[native_sdst] native checkpoint preserved; computing CCNS/hysteresis/full TIDE"
            );
            let request_values: Vec<_> = region_requests.iter().map(|(r, _)| *r).collect();
            let analysis = bridge.analyze_regions(&request_values)?;
            ensure!(
                analysis.sites.len() == region_requests.len(),
                "native region result coverage mismatch"
            );
            let mut summaries = vec![NativeResidueSummary::default(); topology.n_residues];
            let mut regions = Vec::new();
            for (request, lineage) in &region_requests {
                let matches: Vec<_> = analysis
                    .sites
                    .iter()
                    .filter(|s| s.analyzed_site_identity == request.analyzed_site_identity)
                    .collect();
                ensure!(matches.len() == 1, "native result identity mismatch");
                let site = matches[0];
                let mut tide = Vec::new();
                let mut observed_ids = BTreeSet::new();
                for row in &site.tide_decomposition {
                    let resid = row.residue_id as usize;
                    ensure!(
                        resid < topology.n_residues && observed_ids.insert(resid),
                        "duplicate/out-of-domain TIDE residue"
                    );
                    let supported = row.n_causal_spikes > 0
                        || (row.transfer_entropy.is_finite() && row.transfer_entropy > 0.0);
                    let s = &mut summaries[resid];
                    // Each estimator carries its own availability. A finite
                    // windowed result from the native two-voxel source margin
                    // must not be discarded by a strict-ROI/whole-window gate.
                    s.te.push(row.transfer_entropy);
                    s.te_variance.push(row.fisher_info);
                    s.strict_roi_regions += u64::from(row.n_causal_spikes > 0);
                    if supported {
                        s.supported_regions += 1;
                        s.asymmetry.push(site.asymmetry_score);
                    }
                    tide.push(json!({"engine_topology_residue_id":row.residue_id,"n_causal_spikes":row.n_causal_spikes,
                        "native_transfer_entropy":f32_record(row.transfer_entropy),
                        "native_windowed_transfer_entropy_variance":f32_record(row.fisher_info),
                        "native_causal_dg_proxy":f32_record(row.causal_dg),"native_kl_divergence":f32_record(row.kl_divergence),
                        "native_support_present":supported,"measurement_support":row.measurement_support}));
                }
                regions.push(json!({"analyzed_site_identity":request.analyzed_site_identity,
                    "materialized_site_id":request.materialized_site_id,"source_lineage":lineage,
                    "voxel_region":[request.region.x_min,request.region.x_max,request.region.y_min,request.region.y_max,request.region.z_min,request.region.z_max],
                    "native_analysis":site,"full_tide_decomposition":tide,"native_tide_rows_returned":tide.len(),
                    "native_tide_topology_residue_count":topology.n_residues,"tide_output_truncated":false,
                    "native_tide_not_returned_topology_residue_ids":(0..topology.n_residues).filter(|i|!observed_ids.contains(i)).collect::<Vec<_>>(),
                    "native_tide_producer_admission":"sdst_tide.cu skips residue_spike_counts==0; no offline values are fabricated for absent rows",
                    "asymmetry_measurement":f32_record(site.asymmetry_score)}));
            }
            let observations:Vec<_>=water.iter().zip(summaries.iter()).enumerate().map(|(id,(w,s))|json!({
                "engine_topology_residue_id":id,"spike_water_density":w.value(),
                "native_transfer_entropy_max":s.te.json(),"native_windowed_transfer_entropy_variance_max":s.te_variance.json(),
                "native_heating_cooling_asymmetry_max":s.asymmetry.json(),"native_supported_region_count":s.supported_regions,
                "native_strict_roi_region_count":s.strict_roi_regions,
                "estimator_aggregation_scope":"all native admitted source-residue rows including two-voxel margin; metric availability independent of strict ROI membership"})).collect();
            let stream_value = json!({"stream_id":stream,"protocol":protocol,"saved_protocol_state":saved_protocol,
                "protocol_receipt":protocol_receipt,"spike_receipt":spike_receipt,"analysis_population_receipt":population_receipt,
                "saved_native_grid_geometry":grid_receipt,
                "sdst_ingested_event_count":ingested,"native_checkpoint":checkpoint_metadata,"native_parent_edges":parent_edges,"native_event_context":event_context,"native_region_edge_incidence":region_edge_incidence,
                "native_analysis_outcome":analysis.outcome,
                "native_analysis_summary":{"hysteresis_threshold":analysis.hysteresis_threshold,
                    "hysteretic_site_count":analysis.hysteretic_site_count,"tide_residues_mapped":analysis.tide_residues_mapped,
                    "requested_region_count":region_requests.len(),"relative_asymmetry_normalization_population":"all requested region identities in this stream replay"},
                "global_pockets":analysis.global_pockets,
                "total_avalanches":analysis.total_avalanches,"regions":regions,"residue_observations":observations});
            save_json_new(
                &output_dir.join(format!("native_sdst_stream{stream}.json")),
                &stream_value,
            )?;
            streams.push(stream_value);
        }
        ensure!(
            stream_ids.len() as u64 == required_u64(&manifest, "streams_serialized")?,
            "serialized stream population mismatch"
        );
        ensure!(
            total_records == required_u64(&manifest, "total_spikes_md")?,
            "manifest total spike population mismatch"
        );
        let output = json!({"schema_kind":"prism.native_sdst_saved_evidence.v1","run_id":run_id,"target":target,
            "producer":"dstw_export_wt_pathb native saved-evidence branch -> SdstBridge -> libsdst CUDA",
            "md_executed":false,"native_spike_count":selected_records,"raw_spike_count":total_records,
            "analysis_population":policy_receipt,"source_receipts":sources,"streams":streams,
            "lattice_selection_provenance":lattice_selection_provenance,
            "excluded_density_only_candidates":materialized.get("density_only_candidates"),
            "density_only_exclusion_reason":"native replay input is attributed spatiotemporal support; density-only records are retained verbatim here and not silently promoted to measured regions",
            "measurement_contract":{
                "residue_identity":"zero-based engine topology index; no implicit UniProt conversion",
                "native_transfer_entropy":"existing native TIDE residue-to-pocket binned transfer entropy; estimator output, not intervention proof",
                "native_tide_estimator_source_criteria":{"source":".archive/crates/sdst/src/sdst_tide.cu",
                    "TE_BIN_WIDTH_md_steps":100,"TE_MAX_BINS":1024,"source_and_target_occupancy_open_interval":[0.05,0.95],
                    "windowed_te_variance_window_bins":10,"causal_dg_RT_kcal_mol":0.596,
                    "residue_admission":"residue_spike_counts > 0","n_causal_spikes_definition":"native events from mapped residue lying within queried pocket voxel bounds"},
                "native_windowed_transfer_entropy_variance":"existing SDST compute_fisher_info output, upstream field fisher_info; windowed TE variance, not incoming TE or a Fisher information matrix",
                "native_heating_cooling_asymmetry":"native SDST heating/cooling spike-rate asymmetry; projected only onto residues with native TIDE support",
                "spike_water_density_variance":"population Welford variance over recorded nearby-residue spike water_density; PRISM native field, not explicit-water trajectories",
                "aggregation":"TE and windowed TE variance: finite maxima across all native admitted source-region rows, including native two-voxel margin; each metric retains finite/nonfinite counts. Asymmetry retains legacy strict-ROI-or-positive-TE support. Unobserved remains null",
                "causal_edges":"native checkpoint retains complete GPU-produced parent_spike references and timestamps; legacy subgraph parent_indices API intentionally not invoked",
                "selection":"all binding_sites and spatiotemporal_partial_candidates with recorded support for each stream plus the explicitly recorded lattice_selection_provenance; density_only_candidates retained separately; identity domains never conflated",
                "tide_residue_assignment":"existing initial-topology nearest-atom voxel paint, native RESMAP_PAINT_RADIUS=2; differs from per-spike nearby_residues attribution used for water-density variance"}});
        save_json_new(&args.out_json, &output)?;
        eprintln!("[native_sdst] wrote {}", args.out_json.display());
        Ok(())
    }
    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn saved_lattice_recovery_preserves_population_and_historical_identity() {
            let lattice=json!({"components":[{"component_id":9},{"component_id":2}]});
            assert_eq!(lattice_selection_ids(&lattice,None).unwrap(),vec![(2,None),(9,None)]);
            let status=json!({"selected_site_count":1,"site_mappings":[[{"domain":"lattice_component","id":9},3]]});
            assert_eq!(lattice_selection_ids(&lattice,Some(&status)).unwrap(),vec![(9,Some(3))]);
            assert!(lattice_selection_ids(&json!({"components":[{"component_id":2},{"component_id":2}]}),None).is_err());
            assert!(lattice_selection_ids(&lattice,Some(&json!({"selected_site_count":1,"site_mappings":[[{"domain":"materializer_region","id":9},3]]}))).is_err());
            assert!(lattice_selection_ids(&lattice,Some(&json!({"selected_site_count":1,"site_mappings":[[{"domain":"lattice_component","id":99},3]]}))).is_err());
        }
        #[test]
        fn native_integer_fs_overrides_legacy_ps_and_rejects_partial_authority() {
            let raw=b"{\"step\":1,\"applied_dt_fs\":2,\"cumulative_physical_time_fs\":2,\"time_authority\":\"cumulative_physical_time_fs\"}\n{\"step\":2,\"applied_dt_fs\":3,\"cumulative_physical_time_fs\":5,\"time_authority\":\"cumulative_physical_time_fs\"}\n";
            let axis=integer_fs_axis(raw,&BTreeSet::from([1,2])).unwrap().unwrap();
            let clock=PhysicalEventClock{fixed_dt:None,exact:Some(BTreeMap::from([(1,0.0020001),(2,0.0050002)])),exact_fs:Some(axis)};
            assert_eq!(clock.time(2).unwrap(),Some(0.005));assert_eq!(clock.lag(1,2).unwrap(),Some(0.003));
            assert!(clock.time(3).is_err());assert!(integer_fs_axis(b"{\"step\":1,\"applied_dt_fs\":2}\n",&BTreeSet::from([1])).is_err());
            assert!(integer_fs_axis(b"{\"step\":1}\n",&BTreeSet::from([1])).unwrap().is_none());
        }
        #[test]
        fn exact_native_edge_time_subtracts_endpoints() {
            let axis=PhysicalEventClock{fixed_dt:None,exact:Some(BTreeMap::from([(1,0.002),(3,0.007)])),exact_fs:None};
            assert_eq!(axis.time(1).unwrap(),Some(0.002));
            assert_eq!(axis.lag(1,3).unwrap(),Some(0.007-0.002));
            assert!(axis.time(2).is_err());
            assert!(axis.lag(3,1).is_err());
            assert_eq!(PhysicalEventClock{fixed_dt:None,exact:None,exact_fs:None}.lag(1,3).unwrap(),None);
        }
        #[test]
        fn directed_endpoint_identity_and_order_are_required() {
            let a=DirectedEndpoint{ordinal:12,step:6,source:1,residue:3,nearby_count:0,nearby:[-1;8],wavelength:280.0,physical_time_ps:None};
            let mut b=a;b.ordinal=19;b.step=12;b.source=2;
            let events=[a,b];
            let (x,y)=checked_endpoint_pair(&events,0,1,6,12).unwrap();
            assert_eq!((x.ordinal,y.ordinal),(12,19));
            assert!(checked_endpoint_pair(&events,1,0,12,6).is_err());
            assert!(checked_endpoint_pair(&events,0,2,6,12).is_err());
            assert!(checked_endpoint_pair(&events,0,1,7,12).is_err());
            assert!(checked_endpoint_pair(&events,0,0,6,6).is_err());
        }

        #[test]
        fn atomic_publication_never_replaces_completed_output() {
            let dir=std::env::temp_dir().join(format!("prism_native_atomic_{}",std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let path=dir.join("record.json");
            save_json_new(&path,&json!({"source":1})).unwrap();
            assert!(save_json_new(&path,&json!({"source":2})).is_err());
            assert_eq!(read_json(&path).unwrap()["source"],1);
            let unpublished=dir.join("incomplete.json");
            let (mut file,partial)=partial_output(&unpublished).unwrap();
            file.write_all(b"incomplete").unwrap(); drop(file);
            assert!(!unpublished.exists()); assert!(partial.exists());
            std::fs::remove_dir_all(dir).unwrap();
        }

        #[test]
        fn wavefront_override_is_optional_and_never_auto_selected() {
            let args=["native","--native-md-evidence-manifest","m","--materialized-sites","s","--native-lattice","l","--native-therm-status","t","--topology-json","p","--out-json","o"];
            assert_eq!(NativeArgs::try_parse_from(args).unwrap().native_wavefront_max_gap_md_steps,None);
            let mut explicit=args.to_vec(); explicit.extend(["--native-wavefront-max-gap-md-steps","6"]);
            assert_eq!(NativeArgs::try_parse_from(explicit).unwrap().native_wavefront_max_gap_md_steps,Some(6));
            assert_eq!(unsafe{sdst::sdst_default_config()}.wavefront_max_dt,5.0);
        }

        #[test]
        fn variance_distinguishes_missing_and_measured_zero() {
            let mut w = WaterStats::default();
            assert!(w.value()["variance"].is_null());
            w.push(2.0);
            w.push(2.0);
            w.push(f32::NAN);
            assert_eq!(w.value()["variance"], json!(0.0));
            assert_eq!(w.count, 2);
            assert_eq!(w.nonfinite_count, 1);
            let mut m = FiniteMax::default();
            m.push(f32::NAN);
            assert!(m.json()["value"].is_null());
            m.push(0.0);
            assert_eq!(m.json()["value"], json!(0.0));
        }
        #[test]
        fn saved_protocol_boundaries_are_reconstructed_without_defaults() {
            let p = json!({"stream_index":7,"cold_hold_end":14,"ramp_end":20,"warm_hold_end":35,
                "ramp_down_end":41,"total_steps":55,"current_step":55,"start_temp_K":50.0,"end_temp_K":300.0,
                "scan_wavelengths_nm":[280.0,211.0],"n_wavelengths":2,"uv_burst_energy_kcal_mol":42.0,
                "uv_burst_interval_steps":250,"uv_burst_duration_steps":50,"wavelength_dwell_steps":300});
            let x = protocol_from_saved(&p, 7).unwrap();
            assert_eq!(
                [
                    x.cold_hold_steps,
                    x.ramp_steps,
                    x.warm_hold_steps,
                    x.ramp_down_steps,
                    x.cold_return_steps
                ],
                [14, 6, 15, 6, 14]
            );
            assert_eq!(x.start_temp, 50.0);
            assert_eq!(x.uv_burst_energy, 42.0);
            assert!(protocol_from_saved(&p, 0).is_err());
            let mut invalid = p.clone();
            invalid.as_object_mut().unwrap().remove("ramp_down_end");
            assert!(protocol_from_saved(&invalid, 7).is_err());
        }
        #[test]
        fn checkpoint_edges_keep_global_ids_and_check_time_and_reference_domains() {
            let dir = std::env::temp_dir().join(format!(
                "prism_native_edges_{}_{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir(&dir).unwrap();
            let path = dir.join("state.sdst");
            let mut cfg: sdst::SdstConfig = unsafe { std::mem::zeroed() };
            cfg.grid_nx = 64;
            cfg.grid_ny = 64;
            cfg.grid_nz = 64;
            cfg.grid_spacing = 0.8641069531440735;
            cfg.avalanche_max_gap = 50;
            let mut events = [sdst::SpikeEvent::default(); 3];
            events[0].timestamp = 10;
            events[1].timestamp = 30;
            events[1].parent_spike = 3;
            events[1].voxel = 8;
            events[1].local_temp = 1234;
            events[2].timestamp = 20;
            events[2].parent_spike = 1;
            events[2].voxel = 1;
            let write_fixture = |events: &[sdst::SpikeEvent]| {
                let mut bytes = Vec::new();
                bytes.extend(0x54534453u32.to_le_bytes());
                bytes.extend(1u32.to_le_bytes());
                let cfg_words = [
                    cfg.grid_nx,
                    cfg.grid_ny,
                    cfg.grid_nz,
                    cfg.grid_spacing.to_bits(),
                    cfg.hash_table_capacity,
                    cfg.max_spike_events,
                    cfg.max_wavefronts,
                    cfg.wavefront_merge_dist.to_bits(),
                    cfg.wavefront_max_dt.to_bits(),
                    cfg.avalanche_spatial_cutoff.to_bits(),
                    cfg.avalanche_max_gap,
                    cfg.phase_boundaries[0],
                    cfg.phase_boundaries[1],
                    cfg.phase_boundaries[2],
                    cfg.phase_boundaries[3],
                    cfg.phase_boundaries[4],
                    cfg.phase_boundaries[5],
                    cfg.ccns_soc_threshold.to_bits(),
                    cfg.ccns_barrier_threshold.to_bits(),
                    cfg.num_streams,
                    cfg.device_id as u32,
                ];
                assert_eq!(cfg_words.len() * 4, std::mem::size_of::<sdst::SdstConfig>());
                for word in cfg_words {
                    bytes.extend(word.to_le_bytes());
                }
                bytes.extend((events.len() as u32).to_le_bytes());
                for event in events {
                    let mut record = [0u8; 36];
                    for (offset, word) in [
                        (0, event.voxel),
                        (4, event.timestamp),
                        (8, event.parent_spike),
                        (12, event.avalanche_id),
                        (16, event.wavefront_id),
                    ] {
                        record[offset..offset + 4].copy_from_slice(&word.to_le_bytes());
                    }
                    for (offset, word) in [
                        (20, event.amplitude),
                        (22, event.local_temp),
                        (24, event.energy_gradient),
                        (26, event.solvent_exposure),
                        (28, event.wavefront_velocity),
                        (30, event.wavefront_coherence),
                    ] {
                        record[offset..offset + 2].copy_from_slice(&word.to_le_bytes());
                    }
                    record[32] = event.phase_id;
                    record[33] = event.tcl_flags;
                    bytes.extend(record);
                }
                std::fs::write(&path, bytes).unwrap();
            };
            write_fixture(&events);
            let mut metadata = json!({"config_size_bytes":std::mem::size_of::<sdst::SdstConfig>(),
                "event_buffer_offset":12+std::mem::size_of::<sdst::SdstConfig>(),"event_count":3,"artifact":file_receipt(&path).unwrap(),
                "configuration":{"grid":[cfg.grid_nx,cfg.grid_ny,cfg.grid_nz],"grid_spacing_a":cfg.grid_spacing}});
            let output = dir.join("edges.parquet");
            let receipt = export_native_parent_edges(&path, &metadata, &output, 7).unwrap();
            assert_eq!(receipt["parent_edge_count"], json!(2));
            assert_eq!(receipt["root_node_count"], json!(1));
            assert_eq!(receipt["strict_parent_time_order_pass"], json!(true));
            let batch = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
                File::open(output).unwrap(),
            )
            .unwrap()
            .build()
            .unwrap()
            .next()
            .unwrap()
            .unwrap();
            let source = batch
                .column_by_name("source_event_index")
                .unwrap()
                .as_any()
                .downcast_ref::<arrow_array::UInt32Array>()
                .unwrap();
            let target = batch
                .column_by_name("target_event_index")
                .unwrap()
                .as_any()
                .downcast_ref::<arrow_array::UInt32Array>()
                .unwrap();
            assert_eq!(source.values().as_ref(), &[2, 0]);
            assert_eq!(target.values().as_ref(), &[1, 2]);
            events[1].parent_spike = 4;
            write_fixture(&events);
            metadata["artifact"] = file_receipt(&path).unwrap();
            assert!(
                export_native_parent_edges(&path, &metadata, &dir.join("bad_ref.parquet"), 7)
                    .is_err()
            );
            events[1].parent_spike = 3;
            events[1].timestamp = 20;
            write_fixture(&events);
            metadata["artifact"] = file_receipt(&path).unwrap();
            assert!(
                export_native_parent_edges(&path, &metadata, &dir.join("bad_time.parquet"), 7)
                    .is_err()
            );
            std::fs::remove_dir_all(dir).unwrap();
        }

        #[test]
        fn native_envelope_checks_lineage_integrity_and_water_residue_zero() {
            let dir = std::env::temp_dir().join(format!(
                "prism_native_replay_{}_{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir(&dir).unwrap();
            let path = dir.join("spikes.bin");
            let mut bytes = Vec::new();
            bytes.extend(b"PRSPK001");
            for x in [1u32, 0x01020304, 7] {
                bytes.extend(x.to_le_bytes());
            }
            for x in ["recorded_run", "target"] {
                bytes.extend((x.len() as u64).to_le_bytes());
                bytes.extend(x.as_bytes());
            }
            for x in [2u64, 96, 192] {
                bytes.extend(x.to_le_bytes());
            }
            let mut fnv = 0xcbf29ce484222325u64;
            for (step, wd) in [(10, 1.0), (20, 3.0)] {
                let mut spike = GpuSpikeEvent::default();
                spike.timestep = step;
                spike.water_density = wd;
                spike.n_residues = 2;
                spike.nearby_residues[0] = 0;
                spike.nearby_residues[1] = 0;
                let raw = spike.to_le_bytes();
                for b in raw {
                    fnv = (fnv ^ b as u64).wrapping_mul(0x100000001b3);
                }
                bytes.extend(raw);
            }
            bytes.extend(fnv.to_le_bytes());
            bytes.extend(b"PRSPKEND");
            std::fs::write(&path, &bytes).unwrap();
            let artifact = json!({"path":path,"present":true,"stream_id":7,"record_count":2,
                "size_bytes":bytes.len(),"checksum_fnv1a_64_hex":format!("{fnv:016x}")});
            let geometry=SdstGridGeometry{dimension:64,spacing_a:1.0,origin_a:[-0.5;3]};
            let (spikes, stats, _) = read_spikes(&artifact, "recorded_run", "target", 2,&geometry).unwrap();
            assert_eq!(spikes.len(), 2);
            assert_eq!(stats[0].count, 2);
            assert_eq!(stats[0].value()["variance"], json!(1.0));
            assert!(stats[1].value()["variance"].is_null());
            assert!(read_spikes(&artifact, "different_run", "target", 2,&geometry).is_err());
            let checksum = bytes.len() - 16;
            bytes[checksum] ^= 1;
            std::fs::write(&path, &bytes).unwrap();
            assert!(read_spikes(&artifact, "recorded_run", "target", 2,&geometry).is_err());
            std::fs::remove_dir_all(dir).unwrap();
        }
    }
}
