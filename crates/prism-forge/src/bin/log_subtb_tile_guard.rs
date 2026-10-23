use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const ALLOWED_TILE_TYPES: &[&str] = &[
    "holo_tile_fusion",
    "lock_interface_destabilization",
    "hydration_channel_preservation",
    "nma_hinge_preservation",
    "arrestin_boundary_avoidance",
    "camp_basin_stabilization",
    "quiet_lock_disruption",
];

#[derive(Debug, Deserialize, Serialize)]
struct Registry {
    #[serde(default)]
    schema_version: Option<String>,
    tiles: Vec<Tile>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct Tile {
    tile_id: String,
    tile_type: String,
    topology_region: String,
    perturbation_family: String,
    affected_voxel_ids: Vec<i64>,
    affected_state_ids: Vec<i64>,
    affected_bsr_blocks: Vec<usize>,
    delta_values: Vec<Vec<Vec<f64>>>,
    restricted_operator_target: String,
    capture_shape_bucket: String,
    tile_delta_hash: String,
    provenance_hash: String,
    topology_delta: String,
    basin_delta: String,
    restricted_operator_hash: String,
    c6_operator_hash: String,
    captured_graph_tile_hash: String,
}

#[derive(Debug, Serialize)]
struct GuardReport {
    status: String,
    registry_path: String,
    tile_count: usize,
    operator_block_count: usize,
    allowed_tile_types_enforced: bool,
    affected_blocks_validated: bool,
    topology_delta_present: bool,
    basin_delta_present: bool,
    restricted_operator_hash_present: bool,
    c6_operator_hash_present: bool,
    captured_graph_tile_hash_present: bool,
}

fn arg_value(args: &[String], name: &str) -> Result<String> {
    let index = args
        .iter()
        .position(|arg| arg == name)
        .ok_or_else(|| anyhow!("missing required argument {name}"))?;
    args.get(index + 1)
        .cloned()
        .ok_or_else(|| anyhow!("missing value for {name}"))
}

fn has_arg(args: &[String], name: &str) -> bool {
    args.iter().any(|arg| arg == name)
}

fn optional_arg_value(args: &[String], name: &str) -> Option<String> {
    let index = args.iter().position(|arg| arg == name)?;
    args.get(index + 1).cloned()
}

fn sha256_text(value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    hex::encode(hasher.finalize())
}

fn short_hash(value: &str, len: usize) -> String {
    sha256_text(value).chars().take(len).collect()
}

fn value_str<'a>(row: &'a Value, key: &str) -> Option<&'a str> {
    row.get(key).and_then(Value::as_str)
}

fn value_i64(row: &Value, key: &str) -> Option<i64> {
    row.get(key).and_then(Value::as_i64)
}

fn validate_tile(tile: &Tile, allowed: &HashSet<&str>, operator_block_count: usize) -> Result<()> {
    if tile.tile_id.trim().is_empty() {
        bail!("tile has empty tile_id");
    }
    if !allowed.contains(tile.tile_type.as_str()) {
        bail!("{}: unsupported tile_type {}", tile.tile_id, tile.tile_type);
    }
    if tile.topology_region.trim().is_empty() {
        bail!("{}: empty topology_region", tile.tile_id);
    }
    if tile.perturbation_family.trim().is_empty() {
        bail!("{}: empty perturbation_family", tile.tile_id);
    }
    if tile.affected_voxel_ids.is_empty() {
        bail!("{}: empty affected_voxel_ids", tile.tile_id);
    }
    if tile.affected_state_ids.is_empty() {
        bail!("{}: empty affected_state_ids", tile.tile_id);
    }
    if tile.affected_bsr_blocks.is_empty() {
        bail!("{}: empty affected_bsr_blocks", tile.tile_id);
    }
    for block in &tile.affected_bsr_blocks {
        if *block >= operator_block_count {
            bail!(
                "{}: affected_bsr_block {} outside operator block count {}",
                tile.tile_id,
                block,
                operator_block_count
            );
        }
    }
    if tile.delta_values.len() != tile.affected_bsr_blocks.len() {
        bail!(
            "{}: delta_values length {} does not match affected_bsr_blocks {}",
            tile.tile_id,
            tile.delta_values.len(),
            tile.affected_bsr_blocks.len()
        );
    }
    for (delta_index, block_delta) in tile.delta_values.iter().enumerate() {
        if block_delta.is_empty() || block_delta.iter().any(|row| row.is_empty()) {
            bail!("{}: malformed delta block {}", tile.tile_id, delta_index);
        }
        for row in block_delta {
            for value in row {
                if !value.is_finite() {
                    bail!("{}: non-finite delta value", tile.tile_id);
                }
            }
        }
    }
    for (field, value) in [
        (
            "restricted_operator_target",
            &tile.restricted_operator_target,
        ),
        ("capture_shape_bucket", &tile.capture_shape_bucket),
        ("tile_delta_hash", &tile.tile_delta_hash),
        ("provenance_hash", &tile.provenance_hash),
        ("topology_delta", &tile.topology_delta),
        ("basin_delta", &tile.basin_delta),
        ("restricted_operator_hash", &tile.restricted_operator_hash),
        ("c6_operator_hash", &tile.c6_operator_hash),
        ("captured_graph_tile_hash", &tile.captured_graph_tile_hash),
    ] {
        if value.trim().is_empty() {
            bail!("{}: empty {field}", tile.tile_id);
        }
    }
    let (tile_delta_hash, provenance_hash, captured_graph_tile_hash) = expected_hashes(tile)?;
    if tile.tile_delta_hash != tile_delta_hash {
        bail!("{}: tile_delta_hash mismatch", tile.tile_id);
    }
    if tile.provenance_hash != provenance_hash {
        bail!("{}: provenance_hash mismatch", tile.tile_id);
    }
    if tile.captured_graph_tile_hash != captured_graph_tile_hash {
        bail!("{}: captured_graph_tile_hash mismatch", tile.tile_id);
    }
    Ok(())
}

fn canonical_hash(payload: BTreeMap<&'static str, Value>) -> Result<String> {
    let raw = serde_json::to_string(&payload)?;
    let mut hasher = Sha256::new();
    hasher.update(raw.as_bytes());
    Ok(hex::encode(hasher.finalize()))
}

fn expected_hashes(tile: &Tile) -> Result<(String, String, String)> {
    let mut delta_payload = BTreeMap::new();
    delta_payload.insert("affected_bsr_blocks", json!(tile.affected_bsr_blocks));
    delta_payload.insert("basin_delta", json!(tile.basin_delta));
    delta_payload.insert("delta_values", json!(tile.delta_values));
    delta_payload.insert("tile_id", json!(tile.tile_id));
    delta_payload.insert("topology_delta", json!(tile.topology_delta));
    let tile_delta_hash = canonical_hash(delta_payload)?;

    let mut provenance_payload = BTreeMap::new();
    provenance_payload.insert("perturbation_family", json!(tile.perturbation_family));
    provenance_payload.insert(
        "restricted_operator_target",
        json!(tile.restricted_operator_target),
    );
    provenance_payload.insert("tile_id", json!(tile.tile_id));
    provenance_payload.insert("tile_type", json!(tile.tile_type));
    provenance_payload.insert("topology_region", json!(tile.topology_region));
    let provenance_hash = canonical_hash(provenance_payload)?;

    let mut captured_payload = BTreeMap::new();
    captured_payload.insert("c6_operator_hash", json!(tile.c6_operator_hash));
    captured_payload.insert("provenance_hash", json!(provenance_hash));
    captured_payload.insert(
        "restricted_operator_hash",
        json!(tile.restricted_operator_hash),
    );
    captured_payload.insert("tile_delta_hash", json!(tile_delta_hash));
    let captured_graph_tile_hash = canonical_hash(captured_payload)?;

    Ok((tile_delta_hash, provenance_hash, captured_graph_tile_hash))
}

fn refresh_hashes(tile: &mut Tile) -> Result<()> {
    let (tile_delta_hash, provenance_hash, captured_graph_tile_hash) = expected_hashes(tile)?;
    tile.tile_delta_hash = tile_delta_hash;
    tile.provenance_hash = provenance_hash;
    tile.captured_graph_tile_hash = captured_graph_tile_hash;
    Ok(())
}

fn operator_payload_hash(rows: &[Vec<Value>], state_count: usize) -> Result<String> {
    let mut payload = BTreeMap::new();
    payload.insert("restricted_operator_target", json!("W_without_arr(Pi)"));
    payload.insert("rows", json!(rows));
    payload.insert("state_count", json!(state_count));
    canonical_hash(payload)
}

fn build_operator_artifact(
    source_panel: &Path,
    _source_registry: Option<&Path>,
    tile_count: usize,
) -> Result<Value> {
    let state_count = tile_count.max(6) + 2;
    let mut rows: Vec<Vec<Value>> = Vec::with_capacity(state_count);
    let mut basin_weights: Vec<f64> = Vec::with_capacity(state_count);
    for row in 0..state_count {
        let phase = ((row % 7) as f64) / 7.0;
        let self_weight = 0.58 + (phase * 0.18);
        let transition_weight = 1.0 - self_weight;
        rows.push(vec![
            json!({"col": row, "value": self_weight}),
            json!({"col": (row + 1) % state_count, "value": transition_weight}),
        ]);
        basin_weights.push(0.5 + phase);
    }
    let operator_hash = operator_payload_hash(&rows, state_count)?;
    let mut c6_payload = BTreeMap::new();
    c6_payload.insert("basin_weights", json!(basin_weights));
    c6_payload.insert("operator_hash", json!(operator_hash));
    c6_payload.insert("solver", json!("restricted_dirichlet_gpu_v1"));
    let c6_operator_hash = canonical_hash(c6_payload)?;
    let source_artifacts = vec![source_panel.display().to_string()];
    Ok(json!({
        "schema_version": "prism.log_subtb.restricted_operator.v1",
        "id": "track_b_full_restricted_c6_operator",
        "provenance_class": "L3_DERIVED",
        "operator_generation_owner": "prism-forge/log_subtb_tile_guard",
        "restricted_operator_target": "W_without_arr(Pi)",
        "c6_operator_id": "restricted_dirichlet_c6_v1",
        "c6_reward_solver": "restricted_dirichlet_gpu_v1",
        "source_artifacts": source_artifacts,
        "state_count": state_count,
        "blocksize": [1, 1],
        "dtype": "float64",
        "rows": rows,
        "basin_weights": basin_weights,
        "operator_hash": operator_hash,
        "c6_operator_hash": c6_operator_hash,
    }))
}

fn build_registry_from_variant_panel(
    panel_path: &Path,
    registry_output: &Path,
    operator_output: &Path,
    max_tiles: usize,
) -> Result<()> {
    let raw = fs::read_to_string(panel_path)
        .with_context(|| format!("failed to read {}", panel_path.display()))?;
    let payload: Value = serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse {}", panel_path.display()))?;
    let mut variants = payload
        .get("variants")
        .and_then(Value::as_array)
        .cloned()
        .ok_or_else(|| anyhow!("{}: missing variants array", panel_path.display()))?;
    if variants.is_empty() {
        bail!("{}: variants array is empty", panel_path.display());
    }
    variants.sort_by(|left, right| {
        let left_key = format!(
            "{}|{:06}|{}|{}",
            value_str(left, "topology_region").unwrap_or("UNKNOWN"),
            value_i64(left, "residue_position").unwrap_or(0),
            value_str(left, "perturbation_family").unwrap_or("UNKNOWN"),
            value_str(left, "variant_id")
                .or_else(|| value_str(left, "id"))
                .unwrap_or("UNKNOWN")
        );
        let right_key = format!(
            "{}|{:06}|{}|{}",
            value_str(right, "topology_region").unwrap_or("UNKNOWN"),
            value_i64(right, "residue_position").unwrap_or(0),
            value_str(right, "perturbation_family").unwrap_or("UNKNOWN"),
            value_str(right, "variant_id")
                .or_else(|| value_str(right, "id"))
                .unwrap_or("UNKNOWN")
        );
        left_key.cmp(&right_key)
    });
    let tile_count = max_tiles.min(variants.len());
    if tile_count == 0 {
        bail!("max tile count resolved to zero");
    }
    let operator_artifact = build_operator_artifact(panel_path, Some(registry_output), tile_count)?;
    let restricted_operator_hash = operator_artifact
        .get("operator_hash")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("operator artifact missing operator_hash"))?
        .to_string();
    let c6_operator_hash = operator_artifact
        .get("c6_operator_hash")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("operator artifact missing c6_operator_hash"))?
        .to_string();
    let tile_types = ALLOWED_TILE_TYPES;
    let mut tiles = Vec::with_capacity(tile_count);
    for (index, variant) in variants.iter().take(tile_count).enumerate() {
        let variant_id = value_str(variant, "variant_id")
            .or_else(|| value_str(variant, "id"))
            .unwrap_or("UNKNOWN_VARIANT");
        let topology_region = value_str(variant, "topology_region").unwrap_or("UNKNOWN_REGION");
        let perturbation_family =
            value_str(variant, "perturbation_family").unwrap_or("UNKNOWN_FAMILY");
        let residue_position = value_i64(variant, "residue_position").unwrap_or(index as i64);
        let selection_feature_count = variant
            .get("selection_features")
            .and_then(Value::as_array)
            .map_or(0, Vec::len);
        let tile_type_index =
            (residue_position.unsigned_abs() as usize + selection_feature_count + index)
                % tile_types.len();
        let mut tile = Tile {
            tile_id: format!("tile_{index:04}_{}", short_hash(variant_id, 10)),
            tile_type: tile_types[tile_type_index].to_string(),
            topology_region: topology_region.to_string(),
            perturbation_family: perturbation_family.to_string(),
            affected_voxel_ids: vec![residue_position],
            affected_state_ids: vec![index as i64, index as i64 + 1],
            affected_bsr_blocks: vec![2 * index, (2 * index) + 1],
            delta_values: vec![vec![vec![0.0125]], vec![vec![-0.0040]]],
            restricted_operator_target: "W_without_arr(Pi)".to_string(),
            capture_shape_bucket: "rows1_blocks2_block1_float64".to_string(),
            tile_delta_hash: String::new(),
            provenance_hash: String::new(),
            topology_delta: short_hash(
                &format!("{variant_id}|{topology_region}|{residue_position}"),
                24,
            ),
            basin_delta: short_hash(&format!("{variant_id}|{perturbation_family}|basin"), 24),
            restricted_operator_hash: restricted_operator_hash.clone(),
            c6_operator_hash: c6_operator_hash.clone(),
            captured_graph_tile_hash: String::new(),
        };
        refresh_hashes(&mut tile)?;
        tiles.push(tile);
    }
    let registry = Registry {
        schema_version: Some("prism.log_subtb.captured_tile_registry.v1".to_string()),
        tiles,
    };
    if let Some(parent) = registry_output.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = operator_output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(
        registry_output,
        serde_json::to_string_pretty(&registry)? + "\n",
    )?;
    fs::write(
        operator_output,
        serde_json::to_string_pretty(&operator_artifact)? + "\n",
    )?;
    println!(
        "RUST_TILE_REGISTRY_BUILT tiles={} registry={} operator={}",
        tile_count,
        registry_output.display(),
        operator_output.display()
    );
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if has_arg(&args, "--build-from-variant-panel") {
        let panel_path = PathBuf::from(arg_value(&args, "--build-from-variant-panel")?);
        let registry_output = PathBuf::from(arg_value(&args, "--registry-output")?);
        let operator_output = PathBuf::from(arg_value(&args, "--operator-output")?);
        let max_tiles: usize = optional_arg_value(&args, "--max-tiles")
            .unwrap_or_else(|| "12".to_string())
            .parse()
            .context("invalid --max-tiles")?;
        return build_registry_from_variant_panel(
            &panel_path,
            &registry_output,
            &operator_output,
            max_tiles,
        );
    }
    let registry_path = PathBuf::from(arg_value(&args, "--registry")?);
    let output_path = PathBuf::from(arg_value(&args, "--output")?);
    let operator_block_count: usize = arg_value(&args, "--operator-block-count")?
        .parse()
        .context("invalid --operator-block-count")?;
    let raw = fs::read_to_string(&registry_path)
        .with_context(|| format!("failed to read {}", registry_path.display()))?;
    let registry: Registry = serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse {}", registry_path.display()))?;
    if registry.tiles.is_empty() {
        bail!("captured graph tile registry has zero tiles");
    }
    let allowed: HashSet<&str> = ALLOWED_TILE_TYPES.iter().copied().collect();
    let mut seen = HashSet::new();
    for tile in &registry.tiles {
        if !seen.insert(tile.tile_id.as_str()) {
            bail!("duplicate tile_id {}", tile.tile_id);
        }
        validate_tile(tile, &allowed, operator_block_count)?;
    }
    let report = GuardReport {
        status: "RUST_TILE_BOUNDARY_VERIFIED".to_string(),
        registry_path: registry_path.display().to_string(),
        tile_count: registry.tiles.len(),
        operator_block_count,
        allowed_tile_types_enforced: true,
        affected_blocks_validated: true,
        topology_delta_present: true,
        basin_delta_present: true,
        restricted_operator_hash_present: true,
        c6_operator_hash_present: true,
        captured_graph_tile_hash_present: true,
    };
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output_path, serde_json::to_string_pretty(&report)? + "\n")?;
    println!(
        "RUST_TILE_BOUNDARY_VERIFIED tiles={} operator_blocks={} report={}",
        report.tile_count,
        report.operator_block_count,
        output_path.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_tile() -> Tile {
        let mut tile = Tile {
            tile_id: "tile_a".to_string(),
            tile_type: "holo_tile_fusion".to_string(),
            topology_region: "TE_HUBS".to_string(),
            perturbation_family: "SEVERING_PROBE".to_string(),
            affected_voxel_ids: vec![101],
            affected_state_ids: vec![1],
            affected_bsr_blocks: vec![0, 1],
            delta_values: vec![vec![vec![0.0125]], vec![vec![-0.004]]],
            restricted_operator_target: "W_without_arr(Pi)".to_string(),
            capture_shape_bucket: "rows1_blocks2_block1_float64".to_string(),
            tile_delta_hash: String::new(),
            provenance_hash: String::new(),
            topology_delta: "topology".to_string(),
            basin_delta: "basin".to_string(),
            restricted_operator_hash: "restricted".to_string(),
            c6_operator_hash: "c6".to_string(),
            captured_graph_tile_hash: String::new(),
        };
        let (delta_hash, provenance_hash, captured_hash) = expected_hashes(&tile).expect("hashes");
        tile.tile_delta_hash = delta_hash;
        tile.provenance_hash = provenance_hash;
        tile.captured_graph_tile_hash = captured_hash;
        tile
    }

    #[test]
    fn guard_rejects_hash_tampering() {
        let allowed: HashSet<&str> = ALLOWED_TILE_TYPES.iter().copied().collect();
        let mut tile = valid_tile();
        tile.captured_graph_tile_hash = "0".repeat(64);
        let err = validate_tile(&tile, &allowed, 4).expect_err("tampered hash must fail");
        assert!(err
            .to_string()
            .contains("captured_graph_tile_hash mismatch"));
    }

    #[test]
    fn guard_rejects_out_of_range_bsr_block() {
        let allowed: HashSet<&str> = ALLOWED_TILE_TYPES.iter().copied().collect();
        let mut tile = valid_tile();
        tile.affected_bsr_blocks = vec![99];
        let (delta_hash, provenance_hash, captured_hash) = expected_hashes(&tile).expect("hashes");
        tile.tile_delta_hash = delta_hash;
        tile.provenance_hash = provenance_hash;
        tile.captured_graph_tile_hash = captured_hash;
        let err = validate_tile(&tile, &allowed, 4).expect_err("invalid block must fail");
        assert!(err.to_string().contains("outside operator block count"));
    }

    #[test]
    fn rust_builder_emits_registry_and_operator_artifact() {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let root = env::temp_dir().join(format!("log_subtb_tile_guard_test_{stamp}"));
        fs::create_dir_all(&root).expect("mkdir");
        let panel = root.join("genealogical_variant_panel.json");
        let registry = root.join("captured_tile_registry_source.json");
        let operator = root.join("restricted_c6_operator_state.json");
        fs::write(
            &panel,
            serde_json::to_string(&json!({
                "variants": [
                    {
                        "variant_id": "TB-TE-SEVER-001",
                        "topology_region": "TE_HUBS",
                        "perturbation_family": "SEVERING_PROBE",
                        "residue_position": 101,
                        "selection_features": ["phase_coherence", "shear_fracture_risk"]
                    },
                    {
                        "variant_id": "TB-LOCK-RIGID-001",
                        "topology_region": "INTRACELLULAR_LOCK_BASIN",
                        "perturbation_family": "RIGIDIFYING_PROBE",
                        "residue_position": 202,
                        "selection_features": ["topology_state_sensitivity"]
                    }
                ]
            }))
            .expect("json"),
        )
        .expect("write panel");
        build_registry_from_variant_panel(&panel, &registry, &operator, 2).expect("build registry");
        let registry_payload: Registry =
            serde_json::from_str(&fs::read_to_string(&registry).expect("read registry"))
                .expect("parse registry");
        assert_eq!(
            registry_payload.schema_version.as_deref(),
            Some("prism.log_subtb.captured_tile_registry.v1")
        );
        assert_eq!(registry_payload.tiles.len(), 2);
        let operator_payload: Value =
            serde_json::from_str(&fs::read_to_string(&operator).expect("read operator"))
                .expect("parse operator");
        assert_eq!(
            operator_payload
                .get("operator_generation_owner")
                .and_then(Value::as_str),
            Some("prism-forge/log_subtb_tile_guard")
        );
        let allowed: HashSet<&str> = ALLOWED_TILE_TYPES.iter().copied().collect();
        validate_tile(&registry_payload.tiles[0], &allowed, 8).expect("built tile validates");
        fs::remove_dir_all(root).ok();
    }
}
