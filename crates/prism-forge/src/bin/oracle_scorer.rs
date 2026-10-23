use anyhow::{anyhow, Context, Result};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int32Array, Int64Array, LargeStringArray,
    RecordBatch, StringArray, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::{arrow_reader::ParquetRecordBatchReaderBuilder, ArrowWriter};
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use prism_forge::scoring::{
    load_continuity_maps, load_shear_field, phase_resolved_lock_profile, score_molecule,
    score_molecule_with_continuity_and_pose, LoadedContinuityMaps, LoadedSignalGrid, MoleculeScore,
};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::{create_dir_all, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

const DEFAULT_SURVIVORS: &str =
    "campaigns/glp1r_aleniglipron/track_a_generative/vspace_survivors_full_scale.parquet";
const DEFAULT_LOCK_MASK: &str =
    "campaigns/glp1r_aleniglipron/track_a_generative/lock_region_mask.json";
const DEFAULT_TRANSLATION_PATHWAY: &str =
    "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/translation_pathway_nodes.parquet";
const FRAGMENT_CONTEXT_EXCLUSION_A: f64 = 2.32;

#[derive(Debug, Clone)]
struct Config {
    batch: PathBuf,
    rewards: PathBuf,
    survivors: PathBuf,
    lock_mask: Option<PathBuf>,
    live_scoring: bool,
    signal_grid: PathBuf,
    grid_config: PathBuf,
    shear_stress: Option<PathBuf>,
    translation_pathway: Option<PathBuf>,
    nma_continuity_map: Option<PathBuf>,
    hydration_continuity_map: Option<PathBuf>,
    thermodynamic_continuity_map: Option<PathBuf>,
    continuity_admissibility: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            batch: default_scratch_path("oracle_batch.parquet"),
            rewards: default_scratch_path("oracle_rewards.parquet"),
            survivors: PathBuf::from(DEFAULT_SURVIVORS),
            lock_mask: Some(PathBuf::from(DEFAULT_LOCK_MASK)),
            live_scoring: false,
            signal_grid: PathBuf::from(
                "campaigns/glp1r_aleniglipron/track_a_generative/signal_grid_population_consensus.parquet",
            ),
            grid_config: PathBuf::from(
                "campaigns/glp1r_aleniglipron/track_0_manual_emulation/grid_coordinate_mapping.json",
            ),
            shear_stress: Some(PathBuf::from(
                "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/shear_stress_field.parquet",
            )),
            translation_pathway: Some(PathBuf::from(DEFAULT_TRANSLATION_PATHWAY)),
            nma_continuity_map: None,
            hydration_continuity_map: None,
            thermodynamic_continuity_map: None,
            continuity_admissibility: false,
        }
    }
}

/// Scratch root from `PRISM_SCRATCH_ROOT`. There is no compiled-in host
/// default: when it is unset the default path is empty and `parse_args`
/// requires an explicit `--batch` / `--rewards`.
fn default_scratch_root() -> Option<PathBuf> {
    env::var_os("PRISM_SCRATCH_ROOT")
        .map(PathBuf::from)
        .filter(|root| root.is_absolute())
}

fn default_scratch_path(name: &str) -> PathBuf {
    default_scratch_root()
        .map(|root| root.join("oracle_bridge").join(name))
        .unwrap_or_default()
}

#[derive(Debug, Clone)]
struct SurvivorReward {
    anchor_id: String,
    canonical_smiles: String,
    reward: f64,
    pi_complement: f64,
    adjusted_pi_clash: f64,
    pi_clash_pocket: f64,
    pi_clash_lock: f64,
    pi_clash_lock_per_phase: [f64; 5],
    lock_geometry_score: f64,
    lock_geometry_atom_count: u64,
    lock_voxel_indices_json: String,
    lock_occupancy_per_phase: [f64; 5],
    intracellular_penetration_depth_angstrom: f64,
    lock_steric_volume_angstrom3: f64,
    cryptic_bonus: f64,
    consensus_complement_bonus: f64,
    survival_tier: String,
    selected_dihedral_deg: f64,
    lock_proxy_method: String,
    lock_phase_provenance: String,
}

#[derive(Debug, Clone)]
struct Proposal {
    trajectory_id: String,
    anchor_id: String,
    canonical_smiles: String,
    coordinates_json: String,
    score_atom_offset: usize,
    u_pose: f64,
}

#[derive(Debug, Clone)]
struct RewardRow {
    trajectory_id: String,
    anchor_id: String,
    canonical_smiles: String,
    reward: f64,
    pi_complement: f64,
    adjusted_pi_clash: f64,
    pi_clash_pocket: f64,
    pi_clash_lock: f64,
    sigma_shear: f64,
    pi_clash_lock_per_phase: [f64; 5],
    lock_geometry_score: f64,
    lock_geometry_atom_count: u64,
    lock_voxel_indices_json: String,
    lock_occupancy_per_phase: [f64; 5],
    intracellular_penetration_depth_angstrom: f64,
    lock_steric_volume_angstrom3: f64,
    cryptic_bonus: f64,
    consensus_complement_bonus: f64,
    pathway_voxels: u64,
    void_atom_count: u64,
    survival_tier: String,
    selected_dihedral_deg: f64,
    reward_components_json: String,
    lock_phase_provenance: String,
    nma_disruption_penalty: f64,
    hydration_blockade_penalty: f64,
    thermodynamic_trap_penalty: f64,
    pathway_bonus: f64,
    u_pose: f64,
    continuity_admissibility: bool,
    continuity_reward_v1: f64,
    continuity_provenance: String,
    oracle_valid: bool,
    // WT-REROUTE (D1): per-candidate physical-feasibility verdict + machine-readable
    // reason codes, surfaced to the Python bridge so an infeasible candidate ROUTES to a
    // shaped-low reward (rust_reward_oracle.py) instead of crashing the batch. `feasible`
    // is distinct from `oracle_valid`: `oracle_valid` is "the oracle could score this row
    // at all"; `feasible` is "the scored chemistry is physically admissible for
    // promotion". `feasibility_reason_codes` is a JSON array of FeasibilityReason strings.
    feasible: bool,
    feasibility_reason_codes: String,
}

fn main() -> Result<()> {
    let config = parse_args()?;
    let start = Instant::now();
    let lock_mask = match &config.lock_mask {
        Some(path) => Some(
            LockRegionMask::from_path(path)
                .with_context(|| format!("load lock mask {}", path.display()))?,
        ),
        None => None,
    };
    if config.live_scoring {
        let mut grid = LoadedSignalGrid::from_parquet(
            &config.signal_grid.display().to_string(),
            &config.grid_config.display().to_string(),
        )
        .with_context(|| {
            format!(
                "load live signal grid {} with config {}",
                config.signal_grid.display(),
                config.grid_config.display()
            )
        })?;
        let pathway_voxels = match &config.translation_pathway {
            Some(path) if path.exists() => grid
                .add_activation_pathway_from_parquet(path)
                .with_context(|| format!("load translation pathway {}", path.display()))?,
            _ => 0,
        };
        let shear_field = match &config.shear_stress {
            Some(path) => load_shear_field(path, &grid.condition_id)
                .with_context(|| format!("load shear stress {}", path.display()))?,
            None => HashMap::new(),
        };
        let continuity_maps = if config.continuity_admissibility {
            Some(load_required_continuity_maps(&config)?)
        } else {
            None
        };
        let proposals = load_batch(&config.batch)
            .with_context(|| format!("load batch {}", config.batch.display()))?;
        let lock_voxels = lock_mask
            .as_ref()
            .map(|mask| mask.lock_voxels.clone())
            .unwrap_or_default();
        let mut reward_rows = Vec::with_capacity(proposals.len());
        for proposal in proposals {
            let positions =
                parse_coordinates_json(&proposal.coordinates_json).with_context(|| {
                    format!(
                        "parse coordinates_json for trajectory={} smiles={}",
                        proposal.trajectory_id, proposal.canonical_smiles
                    )
                })?;
            let score_positions = live_score_positions(&positions, proposal.score_atom_offset)
                .with_context(|| {
                    format!(
                        "select live score atoms for trajectory={} smiles={}",
                        proposal.trajectory_id, proposal.canonical_smiles
                    )
                })?;
            let score = match &continuity_maps {
                Some(maps) => score_molecule_with_continuity_and_pose(
                    &score_positions,
                    &grid,
                    &lock_voxels,
                    &shear_field,
                    maps,
                    proposal.u_pose,
                ),
                None => score_molecule(&score_positions, &grid, &lock_voxels, &shear_field),
            };
            reward_rows.push(reward_row_from_live_score(proposal, score));
        }
        write_rewards(&config.rewards, &reward_rows)?;
        let mean_reward = if reward_rows.is_empty() {
            0.0
        } else {
            reward_rows.iter().map(|row| row.reward).sum::<f64>() / reward_rows.len() as f64
        };
        println!(
            "oracle_scorer mode=live_signal_grid batch={} valid={} invalid=0 reward_mean={:.6} grid_condition={} field_voxels={} pathway_voxels={} shear_voxels={} continuity_admissibility={} elapsed_ms={:.3} rewards={}",
            reward_rows.len(),
            reward_rows.len(),
            mean_reward,
            grid.condition_id,
            grid.field.len(),
            pathway_voxels,
            shear_field.len(),
            config.continuity_admissibility,
            start.elapsed().as_secs_f64() * 1000.0,
            config.rewards.display()
        );
        return Ok(());
    }
    let phase_grid = LoadedSignalGrid::from_parquet(
        &config.signal_grid.display().to_string(),
        &config.grid_config.display().to_string(),
    )
    .with_context(|| {
        format!(
            "load lock phase signal grid {} with config {}",
            config.signal_grid.display(),
            config.grid_config.display()
        )
    })?;
    let survivors = load_survivors(&config.survivors, lock_mask.as_ref(), Some(&phase_grid))
        .with_context(|| format!("load survivor corpus {}", config.survivors.display()))?;
    let proposals = load_batch(&config.batch)
        .with_context(|| format!("load batch {}", config.batch.display()))?;
    let survivor_by_smiles = best_survivors_by_smiles(survivors);
    let mut reward_rows = Vec::with_capacity(proposals.len());
    for proposal in proposals {
        let Some(reward) = survivor_by_smiles.get(&proposal.canonical_smiles) else {
            reward_rows.push(RewardRow {
                trajectory_id: proposal.trajectory_id,
                anchor_id: proposal.anchor_id,
                canonical_smiles: proposal.canonical_smiles,
                reward: 0.0,
                pi_complement: 0.0,
                adjusted_pi_clash: 0.0,
                pi_clash_pocket: 0.0,
                pi_clash_lock: 0.0,
                sigma_shear: 0.0,
                pi_clash_lock_per_phase: [0.0; 5],
                lock_geometry_score: 0.0,
                lock_geometry_atom_count: 0,
                lock_voxel_indices_json: "[]".to_owned(),
                lock_occupancy_per_phase: [0.0; 5],
                intracellular_penetration_depth_angstrom: 0.0,
                lock_steric_volume_angstrom3: 0.0,
                cryptic_bonus: 0.0,
                consensus_complement_bonus: 0.0,
                pathway_voxels: 0,
                void_atom_count: 0,
                survival_tier: "invalid_missing_survivor".to_owned(),
                selected_dihedral_deg: f64::NAN,
                reward_components_json: json!({
                    "oracle_mode": "validated_o3a_zmatrix_corpus_lookup",
                    "lock_phase_provenance": "REPLICATED_AGGREGATE",
                    "error": "proposal not present in immutable survivor corpus"
                })
                .to_string(),
                lock_phase_provenance: "REPLICATED_AGGREGATE".to_owned(),
                nma_disruption_penalty: 0.0,
                hydration_blockade_penalty: 0.0,
                thermodynamic_trap_penalty: 0.0,
                pathway_bonus: 0.0,
                u_pose: 0.0,
                continuity_admissibility: true,
                continuity_reward_v1: 0.0,
                continuity_provenance: "NOT_REQUESTED".to_owned(),
                oracle_valid: false,
                // A proposal absent from the immutable survivor corpus has no scored
                // chemistry to promote: mark it infeasible with an explicit reason code so
                // the Python bridge routes it to a shaped-low reward instead of crashing.
                feasible: false,
                feasibility_reason_codes: json!(["missing_survivor_corpus_entry"]).to_string(),
            });
            continue;
        };
        reward_rows.push(RewardRow {
            trajectory_id: proposal.trajectory_id,
            anchor_id: reward.anchor_id.clone(),
            canonical_smiles: reward.canonical_smiles.clone(),
            reward: reward.reward.max(1.0e-8),
            pi_complement: reward.pi_complement,
            adjusted_pi_clash: reward.adjusted_pi_clash,
            pi_clash_pocket: reward.pi_clash_pocket,
            pi_clash_lock: reward.pi_clash_lock,
            sigma_shear: 0.0,
            pi_clash_lock_per_phase: reward.pi_clash_lock_per_phase,
            lock_geometry_score: reward.lock_geometry_score,
            lock_geometry_atom_count: reward.lock_geometry_atom_count,
            lock_voxel_indices_json: reward.lock_voxel_indices_json.clone(),
            lock_occupancy_per_phase: reward.lock_occupancy_per_phase,
            intracellular_penetration_depth_angstrom: reward.intracellular_penetration_depth_angstrom,
            lock_steric_volume_angstrom3: reward.lock_steric_volume_angstrom3,
            cryptic_bonus: reward.cryptic_bonus,
            consensus_complement_bonus: reward.consensus_complement_bonus,
            pathway_voxels: 0,
            void_atom_count: 0,
            survival_tier: reward.survival_tier.clone(),
            selected_dihedral_deg: reward.selected_dihedral_deg,
            reward_components_json: json!({
                "oracle_mode": "validated_o3a_zmatrix_corpus_lookup",
                "rust_reward_authority": true,
                "fragment_pi_complement": reward.pi_complement,
                "adjusted_pi_clash": reward.adjusted_pi_clash,
                "pi_clash_pocket": reward.pi_clash_pocket,
                "pi_clash_lock": reward.pi_clash_lock,
                "pi_clash_lock_cold_hold": reward.pi_clash_lock_per_phase[0],
                "pi_clash_lock_ramp_up": reward.pi_clash_lock_per_phase[1],
                "pi_clash_lock_warm_hold": reward.pi_clash_lock_per_phase[2],
                "pi_clash_lock_ramp_down": reward.pi_clash_lock_per_phase[3],
                "pi_clash_lock_cold_return": reward.pi_clash_lock_per_phase[4],
                "lock_geometry_score": reward.lock_geometry_score,
                "lock_geometry_atom_count": reward.lock_geometry_atom_count,
                "lock_voxel_indices": reward.lock_voxel_indices_json,
                "lock_occupancy_cold_hold": reward.lock_occupancy_per_phase[0],
                "lock_occupancy_ramp_up": reward.lock_occupancy_per_phase[1],
                "lock_occupancy_warm_hold": reward.lock_occupancy_per_phase[2],
                "lock_occupancy_ramp_down": reward.lock_occupancy_per_phase[3],
                "lock_occupancy_cold_return": reward.lock_occupancy_per_phase[4],
                "intracellular_penetration_depth_angstrom": reward.intracellular_penetration_depth_angstrom,
                "lock_steric_volume_angstrom3": reward.lock_steric_volume_angstrom3,
                "lock_proxy_method": reward.lock_proxy_method,
                "lock_phase_provenance": reward.lock_phase_provenance,
                "cryptic_bonus": reward.cryptic_bonus,
                "consensus_complement_bonus": reward.consensus_complement_bonus,
                "survival_tier": reward.survival_tier,
                "selected_dihedral_deg": reward.selected_dihedral_deg
            })
            .to_string(),
            lock_phase_provenance: reward.lock_phase_provenance.clone(),
            nma_disruption_penalty: 0.0,
            hydration_blockade_penalty: 0.0,
            thermodynamic_trap_penalty: 0.0,
            pathway_bonus: 0.0,
            u_pose: 0.0,
            continuity_admissibility: true,
            continuity_reward_v1: reward.reward.max(1.0e-8),
            continuity_provenance: "NOT_REQUESTED".to_owned(),
            oracle_valid: true,
            // A survivor-corpus hit was vetted into the immutable corpus, so the Rust
            // oracle reports it feasible with no reason codes. The authoritative ligand
            // feasibility gate (ligand_feasibility_gate.py) runs Python-side and may add
            // reason codes / flip this verdict before reward shaping.
            feasible: true,
            feasibility_reason_codes: "[]".to_owned(),
        });
    }
    write_rewards(&config.rewards, &reward_rows)?;
    let invalid = reward_rows.iter().filter(|row| !row.oracle_valid).count();
    let mean_reward = if reward_rows.is_empty() {
        0.0
    } else {
        reward_rows.iter().map(|row| row.reward).sum::<f64>() / reward_rows.len() as f64
    };
    println!(
        "oracle_scorer batch={} valid={} invalid={} reward_mean={:.6} elapsed_ms={:.3} rewards={}",
        reward_rows.len(),
        reward_rows.len().saturating_sub(invalid),
        invalid,
        mean_reward,
        start.elapsed().as_secs_f64() * 1000.0,
        config.rewards.display()
    );
    Ok(())
}

fn parse_args() -> Result<Config> {
    let mut config = Config::default();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--live-scoring" => config.live_scoring = true,
            "--batch" | "--input" => config.batch = PathBuf::from(value_after(&mut args, &arg)?),
            "--rewards" | "--output" => {
                config.rewards = PathBuf::from(value_after(&mut args, &arg)?)
            }
            "--survivors" => {
                config.survivors = PathBuf::from(value_after(&mut args, "--survivors")?)
            }
            "--signal-grid" => {
                config.signal_grid = PathBuf::from(value_after(&mut args, "--signal-grid")?)
            }
            "--grid-config" => {
                config.grid_config = PathBuf::from(value_after(&mut args, "--grid-config")?)
            }
            "--shear-stress" => {
                config.shear_stress = Some(PathBuf::from(value_after(&mut args, "--shear-stress")?))
            }
            "--no-shear-stress" => {
                config.shear_stress = None;
            }
            "--translation-pathway" => {
                config.translation_pathway = Some(PathBuf::from(value_after(
                    &mut args,
                    "--translation-pathway",
                )?))
            }
            "--no-translation-pathway" => {
                config.translation_pathway = None;
            }
            "--nma-continuity-map" => {
                config.nma_continuity_map = Some(PathBuf::from(value_after(
                    &mut args,
                    "--nma-continuity-map",
                )?))
            }
            "--hydration-continuity-map" => {
                config.hydration_continuity_map = Some(PathBuf::from(value_after(
                    &mut args,
                    "--hydration-continuity-map",
                )?))
            }
            "--thermodynamic-continuity-map" => {
                config.thermodynamic_continuity_map = Some(PathBuf::from(value_after(
                    &mut args,
                    "--thermodynamic-continuity-map",
                )?))
            }
            "--continuity-admissibility" => {
                config.continuity_admissibility = true;
            }
            "--lock-mask" => {
                config.lock_mask = Some(PathBuf::from(value_after(&mut args, "--lock-mask")?))
            }
            "--no-lock-mask" => {
                config.lock_mask = None;
            }
            "--help" | "-h" => {
                println!(
                    "oracle_scorer [--live-scoring --signal-grid <parquet> --grid-config <json> --shear-stress <parquet>|--no-shear-stress --translation-pathway <parquet>|--no-translation-pathway --continuity-admissibility --nma-continuity-map <parquet> --hydration-continuity-map <parquet> --thermodynamic-continuity-map <parquet>] --batch|--input <parquet> --rewards|--output <parquet> [--survivors <parquet>] [--lock-mask <json>|--no-lock-mask]"
                );
                std::process::exit(0);
            }
            _ => return Err(anyhow!("unknown argument {arg}")),
        }
    }
    if config.batch.as_os_str().is_empty() || config.rewards.as_os_str().is_empty() {
        return Err(anyhow!(
            "--batch and --rewards are required when PRISM_SCRATCH_ROOT is unset or not an absolute path (no compiled-in host default)"
        ));
    }
    Ok(config)
}

fn load_required_continuity_maps(config: &Config) -> Result<LoadedContinuityMaps> {
    let nma = config
        .nma_continuity_map
        .as_ref()
        .ok_or_else(|| anyhow!("--continuity-admissibility requires --nma-continuity-map"))?;
    let hydration = config
        .hydration_continuity_map
        .as_ref()
        .ok_or_else(|| anyhow!("--continuity-admissibility requires --hydration-continuity-map"))?;
    let thermodynamic = config
        .thermodynamic_continuity_map
        .as_ref()
        .ok_or_else(|| {
            anyhow!("--continuity-admissibility requires --thermodynamic-continuity-map")
        })?;
    load_continuity_maps(nma, hydration, thermodynamic)
}

fn value_after(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    args.next()
        .ok_or_else(|| anyhow!("{flag} requires a value"))
}

fn best_survivors_by_smiles(survivors: Vec<SurvivorReward>) -> HashMap<String, SurvivorReward> {
    let mut by_smiles: HashMap<String, SurvivorReward> = HashMap::new();
    for survivor in survivors {
        by_smiles
            .entry(survivor.canonical_smiles.clone())
            .and_modify(|existing| {
                if survivor.reward > existing.reward {
                    *existing = survivor.clone();
                }
            })
            .or_insert(survivor);
    }
    by_smiles
}

fn load_batch(path: &Path) -> Result<Vec<Proposal>> {
    let mut proposals = Vec::new();
    for batch in read_batches(path)? {
        let trajectory = string_column(&batch, "trajectory_id")?;
        let anchors = string_column(&batch, "anchor_id")?;
        let smiles = string_column(&batch, "canonical_smiles")?;
        let coordinates = optional_string_column(&batch, "coordinates_json");
        let offsets = optional_u64_column(&batch, "score_atom_offset")?;
        let pose_penalties = optional_f64_column(&batch, "u_pose");
        for row_idx in 0..batch.num_rows() {
            proposals.push(Proposal {
                trajectory_id: string_value(trajectory, row_idx)?,
                anchor_id: string_value(anchors, row_idx)?,
                canonical_smiles: string_value(smiles, row_idx)?,
                coordinates_json: coordinates
                    .as_ref()
                    .and_then(|array| string_value(*array, row_idx).ok())
                    .unwrap_or_default(),
                score_atom_offset: match offsets {
                    Some(array) => score_atom_offset_value(array, row_idx)?,
                    None => 0,
                },
                u_pose: optional_f64_value(pose_penalties, row_idx, 0.0),
            });
        }
    }
    Ok(proposals)
}

fn load_survivors(
    path: &Path,
    lock_mask: Option<&LockRegionMask>,
    phase_grid: Option<&LoadedSignalGrid>,
) -> Result<Vec<SurvivorReward>> {
    let mut survivors = Vec::new();
    for batch in read_batches(path)? {
        let anchors = string_column(&batch, "anchor_id")?;
        let smiles = string_column(&batch, "canonical_smiles")?;
        let complements = f64_column(&batch, "fragment_pi_complement")?;
        let clashes = f64_column(&batch, "fragment_pi_clash_adjusted")?;
        let cryptic = f64_column(&batch, "cryptic_bonus")?;
        let consensus_bonus = optional_f64_column(&batch, "scaffold_consensus_bonus")
            .or_else(|| optional_f64_column(&batch, "consensus_complement_bonus"))
            .or_else(|| optional_f64_column(&batch, "population_consensus_bonus"))
            .or_else(|| optional_f64_column(&batch, "population_consensus_bonus_scaled"));
        let tiers = string_column(&batch, "survival_tier")?;
        let dihedrals = f64_column(&batch, "selected_dihedral_deg")?;
        let coordinates = optional_string_column(&batch, "coordinates_json");
        for row_idx in 0..batch.num_rows() {
            let adjusted_pi_clash = f64_value(clashes, row_idx)?;
            let coordinates_json = coordinates
                .as_ref()
                .and_then(|array| string_value(*array, row_idx).ok())
                .unwrap_or_default();
            let lock_proxy =
                bifurcate_clash(&coordinates_json, adjusted_pi_clash, lock_mask, phase_grid);
            survivors.push(SurvivorReward {
                anchor_id: string_value(anchors, row_idx)?,
                canonical_smiles: string_value(smiles, row_idx)?,
                reward: bifurcated_reward(
                    f64_value(complements, row_idx)?,
                    lock_proxy.pi_clash_pocket,
                    lock_proxy.pi_clash_lock,
                    f64_value(cryptic, row_idx)?,
                ),
                pi_complement: f64_value(complements, row_idx)?,
                adjusted_pi_clash,
                pi_clash_pocket: lock_proxy.pi_clash_pocket,
                pi_clash_lock: lock_proxy.pi_clash_lock,
                pi_clash_lock_per_phase: lock_proxy.pi_clash_lock_per_phase,
                lock_geometry_score: lock_proxy.lock_geometry_score,
                lock_geometry_atom_count: lock_proxy.lock_geometry_atom_count,
                lock_voxel_indices_json: lock_proxy.lock_voxel_indices_json,
                lock_occupancy_per_phase: lock_proxy.lock_occupancy_per_phase,
                intracellular_penetration_depth_angstrom: lock_proxy
                    .intracellular_penetration_depth_angstrom,
                lock_steric_volume_angstrom3: lock_proxy.lock_steric_volume_angstrom3,
                cryptic_bonus: f64_value(cryptic, row_idx)?,
                consensus_complement_bonus: optional_f64_value(consensus_bonus, row_idx, 0.0),
                survival_tier: string_value(tiers, row_idx)?,
                selected_dihedral_deg: f64_value(dihedrals, row_idx)?,
                lock_proxy_method: lock_proxy.method,
                lock_phase_provenance: lock_proxy.lock_phase_provenance,
            });
        }
    }
    Ok(survivors)
}

#[derive(Debug, Clone)]
struct ClashBifurcation {
    pi_clash_pocket: f64,
    pi_clash_lock: f64,
    pi_clash_lock_per_phase: [f64; 5],
    lock_geometry_score: f64,
    lock_geometry_atom_count: u64,
    lock_voxel_indices_json: String,
    lock_occupancy_per_phase: [f64; 5],
    intracellular_penetration_depth_angstrom: f64,
    lock_steric_volume_angstrom3: f64,
    method: String,
    lock_phase_provenance: String,
}

#[derive(Debug, Clone)]
struct LockRegionMask {
    lock_voxels: HashSet<u64>,
    origin: [f64; 3],
    spacing: f64,
    dims: [u64; 3],
    lock_centroid_z: f64,
}

impl LockRegionMask {
    fn from_path(path: &Path) -> Result<Self> {
        let decoded: serde_json::Value = serde_json::from_reader(File::open(path)?)?;
        let lock_voxels = decoded
            .get("lock_voxel_indices")
            .and_then(|value| value.as_array())
            .ok_or_else(|| anyhow!("lock mask missing lock_voxel_indices"))?
            .iter()
            .filter_map(|value| value.as_u64())
            .collect::<HashSet<_>>();
        let grid = decoded
            .get("grid")
            .ok_or_else(|| anyhow!("lock mask missing grid object"))?;
        let origin_values = grid
            .get("origin_xyz_angstrom")
            .and_then(|value| value.as_array())
            .ok_or_else(|| anyhow!("lock mask grid missing origin_xyz_angstrom"))?;
        let origin = [
            origin_values
                .first()
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0),
            origin_values
                .get(1)
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0),
            origin_values
                .get(2)
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0),
        ];
        let dims = [
            grid.get("nx")
                .and_then(|value| value.as_u64())
                .unwrap_or(96),
            grid.get("ny")
                .and_then(|value| value.as_u64())
                .unwrap_or(96),
            grid.get("nz")
                .and_then(|value| value.as_u64())
                .unwrap_or(96),
        ];
        let spacing = grid
            .get("spacing_angstrom")
            .and_then(|value| value.as_f64())
            .ok_or_else(|| anyhow!("lock mask grid missing spacing_angstrom"))?;
        let lock_centroid_z = if lock_voxels.is_empty() {
            origin[2]
        } else {
            lock_voxels
                .iter()
                .map(|voxel_idx| voxel_center_z(*voxel_idx, origin, spacing, dims))
                .sum::<f64>()
                / lock_voxels.len() as f64
        };
        Ok(Self {
            lock_voxels,
            origin,
            spacing,
            dims,
            lock_centroid_z,
        })
    }

    fn voxel_idx_for_xyz(&self, xyz: &[f64]) -> Option<u64> {
        if xyz.len() < 3 || self.spacing <= 0.0 {
            return None;
        }
        let ix = ((xyz[0] - self.origin[0]) / self.spacing).floor() as i64;
        let iy = ((xyz[1] - self.origin[1]) / self.spacing).floor() as i64;
        let iz = ((xyz[2] - self.origin[2]) / self.spacing).floor() as i64;
        if ix < 0
            || iy < 0
            || iz < 0
            || ix >= self.dims[0] as i64
            || iy >= self.dims[1] as i64
            || iz >= self.dims[2] as i64
        {
            return None;
        }
        Some(iz as u64 * (self.dims[0] * self.dims[1]) + iy as u64 * self.dims[0] + ix as u64)
    }

    fn contains_voxel(&self, voxel_idx: u64) -> bool {
        self.lock_voxels.contains(&voxel_idx)
    }
}

fn voxel_center_z(voxel_idx: u64, origin: [f64; 3], spacing: f64, dims: [u64; 3]) -> f64 {
    let plane = dims[0] * dims[1];
    let iz = voxel_idx / plane;
    origin[2] + (iz as f64 + 0.5) * spacing
}

fn bifurcated_reward(
    pi_complement: f64,
    pi_clash_pocket: f64,
    pi_clash_lock: f64,
    cryptic_bonus: f64,
) -> f64 {
    (pi_complement + cryptic_bonus + pi_clash_lock - pi_clash_pocket).max(1.0e-8)
}

fn parse_coordinates_json(raw: &str) -> Result<Vec<(f64, f64, f64)>> {
    if raw.trim().is_empty() {
        return Err(anyhow!("coordinates_json is empty"));
    }
    let value: Value = serde_json::from_str(raw)?;
    if let Some(rows) = value.as_array() {
        if rows.is_empty() {
            return Err(anyhow!("coordinates_json contains zero atom coordinates"));
        }
        if rows.iter().all(|row| row.as_array().is_some()) {
            let mut positions = Vec::with_capacity(rows.len());
            for row in rows {
                let coords = row
                    .as_array()
                    .ok_or_else(|| anyhow!("coordinate row is not an array"))?;
                if coords.len() != 3 {
                    return Err(anyhow!("coordinate row must contain exactly 3 values"));
                }
                let xyz = (
                    finite_coordinate(&coords[0], "x")?,
                    finite_coordinate(&coords[1], "y")?,
                    finite_coordinate(&coords[2], "z")?,
                );
                positions.push(xyz);
            }
            return Ok(positions);
        }
        if rows.len() % 3 == 0 {
            let values = rows
                .iter()
                .map(|entry| finite_coordinate(entry, "flat"))
                .collect::<Result<Vec<_>>>()?;
            let positions = values
                .chunks_exact(3)
                .map(|xyz| (xyz[0], xyz[1], xyz[2]))
                .collect::<Vec<_>>();
            return Ok(positions);
        }
    }
    Err(anyhow!(
        "coordinates_json must be [[x,y,z],...] or a flat xyz array"
    ))
}

fn finite_coordinate(value: &Value, label: &str) -> Result<f64> {
    let coordinate = value
        .as_f64()
        .ok_or_else(|| anyhow!("{label} coordinate is not numeric"))?;
    if !coordinate.is_finite() {
        return Err(anyhow!("{label} coordinate is not finite"));
    }
    Ok(coordinate)
}

fn live_score_positions(
    positions: &[(f64, f64, f64)],
    score_atom_offset: usize,
) -> Result<Vec<(f64, f64, f64)>> {
    if positions.is_empty() {
        return Err(anyhow!(
            "live scoring requires at least one atom coordinate"
        ));
    }
    if score_atom_offset == 0 {
        return Ok(positions.to_vec());
    }
    if score_atom_offset >= positions.len() {
        return Err(anyhow!(
            "score_atom_offset {score_atom_offset} leaves no atoms to score in {} coordinates",
            positions.len()
        ));
    }
    let fragment_positions = live_fragment_positions(positions, score_atom_offset);
    if fragment_positions.is_empty() {
        return Err(anyhow!(
            "score_atom_offset {score_atom_offset} produced zero fragment atoms after scaffold-context exclusion"
        ));
    }
    Ok(fragment_positions)
}

fn live_fragment_positions(
    positions: &[(f64, f64, f64)],
    score_atom_offset: usize,
) -> Vec<(f64, f64, f64)> {
    let scaffold_positions = &positions[..score_atom_offset];
    positions
        .iter()
        .skip(score_atom_offset)
        .copied()
        .filter(|xyz| {
            min_distance_to_positions(*xyz, scaffold_positions) > FRAGMENT_CONTEXT_EXCLUSION_A
        })
        .collect()
}

fn min_distance_to_positions(xyz: (f64, f64, f64), positions: &[(f64, f64, f64)]) -> f64 {
    positions
        .iter()
        .map(|other| {
            let dx = xyz.0 - other.0;
            let dy = xyz.1 - other.1;
            let dz = xyz.2 - other.2;
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .fold(f64::INFINITY, f64::min)
}

fn reward_row_from_live_score(proposal: Proposal, score: MoleculeScore) -> RewardRow {
    let lock_phase = score.lock_occupancy_per_phase;
    let lock_voxel_indices_json = json!(score.occupied_lock_voxels).to_string();
    let lock_volume = f64::from(score.lock_atom_count) * 20.0;
    let phase_provenance = lock_phase_provenance(&lock_phase);
    RewardRow {
        trajectory_id: proposal.trajectory_id,
        anchor_id: proposal.anchor_id,
        canonical_smiles: proposal.canonical_smiles,
        reward: score.reward.max(1.0e-8),
        pi_complement: score.pi_complement,
        adjusted_pi_clash: score.pi_clash_pocket + score.pi_clash_lock,
        pi_clash_pocket: score.pi_clash_pocket,
        pi_clash_lock: score.pi_clash_lock,
        sigma_shear: score.sigma_shear,
        pi_clash_lock_per_phase: lock_phase,
        lock_geometry_score: score.pi_clash_lock,
        lock_geometry_atom_count: u64::from(score.lock_atom_count),
        lock_voxel_indices_json,
        lock_occupancy_per_phase: lock_phase,
        intracellular_penetration_depth_angstrom: 0.0,
        lock_steric_volume_angstrom3: lock_volume,
        cryptic_bonus: score.cryptic_bonus,
        consensus_complement_bonus: score.consensus_bonus,
        pathway_voxels: u64::from(score.pathway_voxels),
        void_atom_count: u64::from(score.void_atom_count),
        survival_tier: "live_signal_grid".to_owned(),
        selected_dihedral_deg: 0.0,
        reward_components_json: json!({
            "oracle_mode": "live_signal_grid",
            "rust_reward_authority": true,
            "fragment_pi_complement": score.pi_complement,
            "adjusted_pi_clash": score.pi_clash_pocket + score.pi_clash_lock,
            "pi_clash_pocket": score.pi_clash_pocket,
            "pi_clash_lock": score.pi_clash_lock,
            "sigma_shear": score.sigma_shear,
            "nma_disruption_penalty": score.nma_disruption_penalty,
            "hydration_blockade_penalty": score.hydration_blockade_penalty,
            "thermodynamic_trap_penalty": score.thermodynamic_trap_penalty,
            "pathway_bonus": score.pathway_bonus,
            "u_pose": score.u_pose,
            "continuity_admissibility": score.continuity_admissibility,
            "continuity_reward_v1": score.continuity_reward_v1,
            "continuity_provenance": score.continuity_provenance,
            "consensus_complement_bonus": score.consensus_bonus,
            "cryptic_bonus": score.cryptic_bonus,
            "lock_phase_provenance": phase_provenance,
            "pathway_voxels": score.pathway_voxels,
            "void_atom_count": score.void_atom_count
        })
        .to_string(),
        lock_phase_provenance: phase_provenance,
        nma_disruption_penalty: score.nma_disruption_penalty,
        hydration_blockade_penalty: score.hydration_blockade_penalty,
        thermodynamic_trap_penalty: score.thermodynamic_trap_penalty,
        pathway_bonus: score.pathway_bonus,
        u_pose: score.u_pose,
        continuity_admissibility: score.continuity_admissibility,
        continuity_reward_v1: score.continuity_reward_v1,
        continuity_provenance: score.continuity_provenance,
        oracle_valid: true,
        // Carry the live MoleculeScore feasibility verdict straight through to the parquet
        // row so the Python bridge can route an infeasible candidate to a shaped-low
        // reward (D1) rather than treating it as promotable.
        feasible: score.feasible,
        feasibility_reason_codes: score.feasibility_reason_codes,
    }
}

fn lock_phase_provenance(phases: &[f64; 5]) -> String {
    let distinct = phases
        .iter()
        .map(|value| (value * 1.0e12).round() as i128)
        .collect::<HashSet<_>>()
        .len();
    if distinct >= 4 {
        "PHASE_RESOLVED".to_owned()
    } else {
        "REPLICATED_AGGREGATE".to_owned()
    }
}

fn bifurcate_clash(
    coordinates_json: &str,
    adjusted_pi_clash: f64,
    lock_mask: Option<&LockRegionMask>,
    phase_grid: Option<&LoadedSignalGrid>,
) -> ClashBifurcation {
    let parsed = serde_json::from_str::<Vec<Vec<f64>>>(coordinates_json);
    let Ok(coords) = parsed else {
        return ClashBifurcation {
            pi_clash_pocket: adjusted_pi_clash,
            pi_clash_lock: 0.0,
            pi_clash_lock_per_phase: [0.0; 5],
            lock_geometry_score: 0.0,
            lock_geometry_atom_count: 0,
            lock_voxel_indices_json: "[]".to_owned(),
            lock_occupancy_per_phase: [0.0; 5],
            intracellular_penetration_depth_angstrom: 0.0,
            lock_steric_volume_angstrom3: 0.0,
            method: "missing_coordinates_fallback_all_pocket".to_owned(),
            lock_phase_provenance: "UNKNOWN".to_owned(),
        };
    };
    if let Some(mask) = lock_mask {
        let mut occupied_lock_voxels = HashSet::new();
        let mut lock_atom_count: u64 = 0;
        let mut min_lock_atom_z = f64::INFINITY;
        let mut lock_cold = 0.0;
        let mut lock_warm = 0.0;
        let mut lock_delta = 0.0;
        let mut lock_phase_samples = 0_u64;
        for coord in &coords {
            let Some(voxel_idx) = mask.voxel_idx_for_xyz(coord) else {
                continue;
            };
            if mask.contains_voxel(voxel_idx) {
                lock_atom_count += 1;
                occupied_lock_voxels.insert(voxel_idx);
                if let Some(field) = phase_grid.and_then(|grid| grid.field.get(&voxel_idx)) {
                    lock_cold += field.cold_mean;
                    lock_warm += field.warm_mean;
                    lock_delta += field.delta;
                    lock_phase_samples += 1;
                }
                if coord.len() >= 3 {
                    min_lock_atom_z = min_lock_atom_z.min(coord[2]);
                }
            }
        }
        let atom_count = coords.len().max(1);
        let lock_fraction = lock_atom_count as f64 / atom_count as f64;
        let pi_clash_lock = adjusted_pi_clash * lock_atom_count as f64;
        let mut lock_voxels_sorted = occupied_lock_voxels.into_iter().collect::<Vec<_>>();
        lock_voxels_sorted.sort_unstable();
        let lock_voxel_indices_json = json!(lock_voxels_sorted).to_string();
        let penetration_depth = if lock_atom_count == 0 || !min_lock_atom_z.is_finite() {
            0.0
        } else {
            (mask.lock_centroid_z - min_lock_atom_z).max(0.0)
        };
        let steric_volume = lock_atom_count as f64 * 20.0;
        let (lock_occupancy_per_phase, lock_phase_provenance) = if lock_phase_samples > 0 {
            let phase_profile = phase_resolved_lock_profile(lock_cold, lock_warm, lock_delta);
            (phase_profile, lock_phase_provenance(&phase_profile))
        } else if lock_atom_count > 0 {
            ([pi_clash_lock; 5], "REPLICATED_AGGREGATE".to_owned())
        } else {
            ([0.0; 5], "PHASE_RESOLVED".to_owned())
        };
        return ClashBifurcation {
            pi_clash_pocket: adjusted_pi_clash * (1.0 - lock_fraction),
            pi_clash_lock,
            pi_clash_lock_per_phase: lock_occupancy_per_phase,
            lock_geometry_score: pi_clash_lock,
            lock_geometry_atom_count: lock_atom_count,
            lock_voxel_indices_json,
            lock_occupancy_per_phase,
            intracellular_penetration_depth_angstrom: penetration_depth,
            lock_steric_volume_angstrom3: steric_volume,
            method: format!(
                "residue_lock_region_mask_v2:lock_atoms={lock_atom_count}:total_atoms={atom_count}:phase_lock={lock_phase_provenance}:steric_volume_proxy=20A3_per_atom"
            ),
            lock_phase_provenance,
        };
    }
    ClashBifurcation {
        pi_clash_pocket: adjusted_pi_clash,
        pi_clash_lock: 0.0,
        pi_clash_lock_per_phase: [0.0; 5],
        lock_geometry_score: 0.0,
        lock_geometry_atom_count: 0,
        lock_voxel_indices_json: "[]".to_owned(),
        lock_occupancy_per_phase: [0.0; 5],
        intracellular_penetration_depth_angstrom: 0.0,
        lock_steric_volume_angstrom3: 0.0,
        method: "lock_mask_missing_legacy_z_proxy_invalidated_all_pocket".to_owned(),
        lock_phase_provenance: "UNKNOWN".to_owned(),
    }
}

fn read_batches(path: &Path) -> Result<Vec<RecordBatch>> {
    let file = File::open(path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
    reader
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(Into::into)
}

fn column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a dyn Array> {
    let idx = batch
        .schema()
        .index_of(name)
        .with_context(|| format!("missing column {name}"))?;
    Ok(batch.column(idx).as_ref())
}

fn string_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a dyn Array> {
    let array = column(batch, name)?;
    if array.as_any().downcast_ref::<StringArray>().is_some()
        || array.as_any().downcast_ref::<LargeStringArray>().is_some()
    {
        Ok(array)
    } else {
        Err(anyhow!("column {name} is not string"))
    }
}

fn optional_string_column<'a>(batch: &'a RecordBatch, name: &str) -> Option<&'a dyn Array> {
    let idx = batch.schema().index_of(name).ok()?;
    let array = batch.column(idx).as_ref();
    if array.as_any().downcast_ref::<StringArray>().is_some()
        || array.as_any().downcast_ref::<LargeStringArray>().is_some()
    {
        Some(array)
    } else {
        None
    }
}

fn optional_f64_column<'a>(batch: &'a RecordBatch, name: &str) -> Option<&'a Float64Array> {
    let idx = batch.schema().index_of(name).ok()?;
    batch.column(idx).as_any().downcast_ref::<Float64Array>()
}

fn optional_u64_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<Option<&'a dyn Array>> {
    let Ok(idx) = batch.schema().index_of(name) else {
        return Ok(None);
    };
    let array = batch.column(idx).as_ref();
    if array.as_any().downcast_ref::<UInt64Array>().is_some()
        || array.as_any().downcast_ref::<UInt32Array>().is_some()
        || array.as_any().downcast_ref::<Int64Array>().is_some()
        || array.as_any().downcast_ref::<Int32Array>().is_some()
        || array.as_any().downcast_ref::<Float64Array>().is_some()
    {
        return Ok(Some(array));
    }
    Err(anyhow!(
        "column {name} has unsupported u64-compatible type {:?}",
        array.data_type()
    ))
}

fn f64_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Float64Array> {
    column(batch, name)?
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| anyhow!("column {name} is not Float64"))
}

fn string_value(array: &dyn Array, row_idx: usize) -> Result<String> {
    if array.is_null(row_idx) {
        return Err(anyhow!("null string at row {row_idx}"));
    }
    if let Some(values) = array.as_any().downcast_ref::<StringArray>() {
        return Ok(values.value(row_idx).to_owned());
    }
    if let Some(values) = array.as_any().downcast_ref::<LargeStringArray>() {
        return Ok(values.value(row_idx).to_owned());
    }
    Err(anyhow!("array is not string"))
}

fn f64_value(array: &Float64Array, row_idx: usize) -> Result<f64> {
    if array.is_null(row_idx) {
        return Err(anyhow!("null f64 at row {row_idx}"));
    }
    Ok(array.value(row_idx))
}

fn optional_f64_value(array: Option<&Float64Array>, row_idx: usize, default: f64) -> f64 {
    match array {
        Some(values) if !values.is_null(row_idx) => values.value(row_idx),
        _ => default,
    }
}

fn optional_u64_value(array: &dyn Array, row_idx: usize) -> Result<Option<u64>> {
    if array.is_null(row_idx) {
        return Ok(None);
    }
    if let Some(values) = array.as_any().downcast_ref::<UInt64Array>() {
        return Ok(Some(values.value(row_idx)));
    }
    if let Some(values) = array.as_any().downcast_ref::<UInt32Array>() {
        return Ok(Some(u64::from(values.value(row_idx))));
    }
    if let Some(values) = array.as_any().downcast_ref::<Int64Array>() {
        let value = values.value(row_idx);
        return u64::try_from(value)
            .map(Some)
            .map_err(|_| anyhow!("negative u64-compatible value at row {row_idx}: {value}"));
    }
    if let Some(values) = array.as_any().downcast_ref::<Int32Array>() {
        let value = values.value(row_idx);
        return u64::try_from(value)
            .map(Some)
            .map_err(|_| anyhow!("negative u64-compatible value at row {row_idx}: {value}"));
    }
    if let Some(values) = array.as_any().downcast_ref::<Float64Array>() {
        let value = values.value(row_idx);
        if !value.is_finite() {
            return Err(anyhow!(
                "non-finite u64-compatible value at row {row_idx}: {value}"
            ));
        }
        if value < 0.0 {
            return Err(anyhow!(
                "negative u64-compatible value at row {row_idx}: {value}"
            ));
        }
        if value.fract() != 0.0 {
            return Err(anyhow!(
                "fractional u64-compatible value at row {row_idx}: {value}"
            ));
        }
        return Ok(Some(value as u64));
    }
    Ok(None)
}

fn score_atom_offset_value(array: &dyn Array, row_idx: usize) -> Result<usize> {
    let Some(offset) = optional_u64_value(array, row_idx)? else {
        return Ok(0);
    };
    usize::try_from(offset).map_err(|_| anyhow!("score_atom_offset is too large: {offset}"))
}

fn write_rewards(path: &Path, rows: &[RewardRow]) -> Result<()> {
    if let Some(parent) = path.parent() {
        create_dir_all(parent)?;
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("trajectory_id", DataType::Utf8, false),
        Field::new("anchor_id", DataType::Utf8, false),
        Field::new("canonical_smiles", DataType::Utf8, false),
        Field::new("reward", DataType::Float64, false),
        Field::new("pi_complement", DataType::Float64, false),
        Field::new("adjusted_pi_clash", DataType::Float64, false),
        Field::new("pi_clash_pocket", DataType::Float64, false),
        Field::new("pi_clash_lock", DataType::Float64, false),
        Field::new("sigma_shear", DataType::Float64, false),
        Field::new("pi_clash_lock_cold_hold", DataType::Float64, false),
        Field::new("pi_clash_lock_ramp_up", DataType::Float64, false),
        Field::new("pi_clash_lock_warm_hold", DataType::Float64, false),
        Field::new("pi_clash_lock_ramp_down", DataType::Float64, false),
        Field::new("pi_clash_lock_cold_return", DataType::Float64, false),
        Field::new("lock_geometry_score", DataType::Float64, false),
        Field::new("lock_geometry_atom_count", DataType::Float64, false),
        Field::new("lock_voxel_indices_json", DataType::Utf8, false),
        Field::new("lock_occupancy_cold_hold", DataType::Float64, false),
        Field::new("lock_occupancy_ramp_up", DataType::Float64, false),
        Field::new("lock_occupancy_warm_hold", DataType::Float64, false),
        Field::new("lock_occupancy_ramp_down", DataType::Float64, false),
        Field::new("lock_occupancy_cold_return", DataType::Float64, false),
        Field::new(
            "intracellular_penetration_depth_angstrom",
            DataType::Float64,
            false,
        ),
        Field::new("lock_steric_volume_angstrom3", DataType::Float64, false),
        Field::new("cryptic_bonus", DataType::Float64, false),
        Field::new("consensus_complement_bonus", DataType::Float64, false),
        Field::new("pathway_voxels", DataType::UInt64, false),
        Field::new("void_atom_count", DataType::UInt64, false),
        Field::new("survival_tier", DataType::Utf8, false),
        Field::new("selected_dihedral_deg", DataType::Float64, false),
        Field::new("reward_components_json", DataType::Utf8, false),
        Field::new("lock_phase_provenance", DataType::Utf8, false),
        Field::new("nma_disruption_penalty", DataType::Float64, false),
        Field::new("hydration_blockade_penalty", DataType::Float64, false),
        Field::new("thermodynamic_trap_penalty", DataType::Float64, false),
        Field::new("pathway_bonus", DataType::Float64, false),
        Field::new("u_pose", DataType::Float64, false),
        Field::new("continuity_admissibility", DataType::Boolean, false),
        Field::new("continuity_reward_v1", DataType::Float64, false),
        Field::new("continuity_provenance", DataType::Utf8, false),
        Field::new("oracle_valid", DataType::Boolean, false),
        // WT-REROUTE (D1): feasibility verdict + JSON reason-code array for the
        // reroute-before-promotion loop.
        Field::new("feasible", DataType::Boolean, false),
        Field::new("feasibility_reason_codes", DataType::Utf8, false),
    ]));
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.trajectory_id.as_str())
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.anchor_id.as_str())
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.canonical_smiles.as_str())
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.reward).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.pi_complement).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.adjusted_pi_clash)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.pi_clash_pocket)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.pi_clash_lock).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.sigma_shear).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.pi_clash_lock_per_phase[0])
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.pi_clash_lock_per_phase[1])
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.pi_clash_lock_per_phase[2])
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.pi_clash_lock_per_phase[3])
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.pi_clash_lock_per_phase[4])
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.lock_geometry_score)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.lock_geometry_atom_count as f64)
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.lock_voxel_indices_json.as_str())
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.lock_occupancy_per_phase[0])
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.lock_occupancy_per_phase[1])
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.lock_occupancy_per_phase[2])
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.lock_occupancy_per_phase[3])
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.lock_occupancy_per_phase[4])
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.intracellular_penetration_depth_angstrom)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.lock_steric_volume_angstrom3)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.cryptic_bonus).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.consensus_complement_bonus)
                .collect::<Vec<_>>(),
        )),
        Arc::new(UInt64Array::from(
            rows.iter()
                .map(|row| row.pathway_voxels)
                .collect::<Vec<_>>(),
        )),
        Arc::new(UInt64Array::from(
            rows.iter()
                .map(|row| row.void_atom_count)
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.survival_tier.as_str())
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.selected_dihedral_deg)
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.reward_components_json.as_str())
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.lock_phase_provenance.as_str())
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.nma_disruption_penalty)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.hydration_blockade_penalty)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.thermodynamic_trap_penalty)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.pathway_bonus).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.u_pose).collect::<Vec<_>>(),
        )),
        Arc::new(BooleanArray::from(
            rows.iter()
                .map(|row| row.continuity_admissibility)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.continuity_reward_v1)
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.continuity_provenance.as_str())
                .collect::<Vec<_>>(),
        )),
        Arc::new(BooleanArray::from(
            rows.iter().map(|row| row.oracle_valid).collect::<Vec<_>>(),
        )),
        Arc::new(BooleanArray::from(
            rows.iter().map(|row| row.feasible).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.feasibility_reason_codes.as_str())
                .collect::<Vec<_>>(),
        )),
    ];
    let batch = RecordBatch::try_new(schema.clone(), arrays)?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();
    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_coordinates_rejects_empty_arrays() {
        let error = parse_coordinates_json("[]").unwrap_err().to_string();
        assert!(error.contains("zero atom coordinates"));
    }

    #[test]
    fn parse_coordinates_rejects_short_nested_rows() {
        let error = parse_coordinates_json("[[1.0, 2.0]]")
            .unwrap_err()
            .to_string();
        assert!(error.contains("exactly 3 values"));
    }

    #[test]
    fn live_score_positions_rejects_offsets_without_score_atoms() {
        let positions = vec![(0.0, 0.0, 0.0), (5.0, 0.0, 0.0)];

        let equal_error = live_score_positions(&positions, positions.len())
            .unwrap_err()
            .to_string();
        assert!(equal_error.contains("leaves no atoms to score"));

        let greater_error = live_score_positions(&positions, positions.len() + 1)
            .unwrap_err()
            .to_string();
        assert!(greater_error.contains("leaves no atoms to score"));
    }

    #[test]
    fn score_atom_offset_rejects_negative_values() {
        let offsets = Int64Array::from(vec![-1_i64]);

        let error = score_atom_offset_value(&offsets, 0)
            .unwrap_err()
            .to_string();

        assert!(error.contains("negative u64-compatible value"));

        let float_offsets = Float64Array::from(vec![-1.0_f64]);
        let float_error = score_atom_offset_value(&float_offsets, 0)
            .unwrap_err()
            .to_string();
        assert!(float_error.contains("negative u64-compatible value"));

        let fractional_offsets = Float64Array::from(vec![1.5_f64]);
        let fractional_error = score_atom_offset_value(&fractional_offsets, 0)
            .unwrap_err()
            .to_string();
        assert!(fractional_error.contains("fractional u64-compatible value"));
    }

    #[test]
    fn optional_u64_column_rejects_unsupported_present_columns() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "score_atom_offset",
            DataType::Utf8,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(vec!["not-an-offset"]))],
        )
        .expect("test record batch should be valid");

        let error = optional_u64_column(&batch, "score_atom_offset")
            .unwrap_err()
            .to_string();
        assert!(error.contains("unsupported u64-compatible type"));
    }

    #[test]
    fn live_reward_row_preserves_phase_resolved_lock_occupancy() {
        let proposal = Proposal {
            trajectory_id: "t0".to_owned(),
            anchor_id: "a0".to_owned(),
            canonical_smiles: "CC".to_owned(),
            coordinates_json: "[[0.0,0.0,0.0]]".to_owned(),
            score_atom_offset: 0,
            u_pose: 0.0,
        };
        let mut score = MoleculeScore::default();
        score.reward = 1.0;
        score.pi_clash_lock = 3.0;
        score.lock_atom_count = 2;
        score.lock_occupancy_per_phase = [3.0, 1.5, 0.0, 1.5, 3.1500000000000004];
        score.occupied_lock_voxels = vec![1, 2];

        let row = reward_row_from_live_score(proposal, score);

        assert_eq!(
            row.lock_occupancy_per_phase,
            [3.0, 1.5, 0.0, 1.5, 3.1500000000000004]
        );
        assert_eq!(row.pi_clash_lock_per_phase, row.lock_occupancy_per_phase);
        assert_eq!(row.lock_phase_provenance, "PHASE_RESOLVED");
    }

    #[test]
    fn live_reward_row_carries_feasibility_verdict_default() {
        // WT-REROUTE (D1): a default MoleculeScore is feasible with no reason codes.
        let proposal = Proposal {
            trajectory_id: "t0".to_owned(),
            anchor_id: "a0".to_owned(),
            canonical_smiles: "CC".to_owned(),
            coordinates_json: "[[0.0,0.0,0.0]]".to_owned(),
            score_atom_offset: 0,
            u_pose: 0.0,
        };
        let row = reward_row_from_live_score(proposal, MoleculeScore::default());
        assert!(row.feasible);
        assert_eq!(row.feasibility_reason_codes, "[]");
    }

    #[test]
    fn live_reward_row_propagates_infeasible_verdict_and_reason_codes() {
        // WT-REROUTE (D1): an infeasible MoleculeScore routes its verdict + reason
        // codes straight to the parquet row for shaped-low reward handling.
        let proposal = Proposal {
            trajectory_id: "t1".to_owned(),
            anchor_id: "a1".to_owned(),
            canonical_smiles: "BAD".to_owned(),
            coordinates_json: "[[0.0,0.0,0.0]]".to_owned(),
            score_atom_offset: 0,
            u_pose: 0.0,
        };
        let mut score = MoleculeScore::default();
        score.feasible = false;
        score.feasibility_reason_codes = json!(["bond_out_of_bounds"]).to_string();
        let row = reward_row_from_live_score(proposal, score);
        assert!(!row.feasible);
        assert_eq!(row.feasibility_reason_codes, "[\"bond_out_of_bounds\"]");
        // oracle_valid stays true: the oracle COULD score it; feasibility is separate.
        assert!(row.oracle_valid);
    }

    #[test]
    fn reward_parquet_schema_includes_feasibility_columns() {
        // WT-REROUTE (D1): the emitted parquet carries `feasible` + reason codes so the
        // Python bridge can route without crashing.
        let dir = std::env::temp_dir().join(format!(
            "prism_reroute_schema_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let path = dir.join("rewards.parquet");
        let proposal = Proposal {
            trajectory_id: "t0".to_owned(),
            anchor_id: "a0".to_owned(),
            canonical_smiles: "CC".to_owned(),
            coordinates_json: "[[0.0,0.0,0.0]]".to_owned(),
            score_atom_offset: 0,
            u_pose: 0.0,
        };
        let mut score = MoleculeScore::default();
        score.feasible = false;
        score.feasibility_reason_codes = json!(["lj_out_of_bounds"]).to_string();
        let row = reward_row_from_live_score(proposal, score);
        write_rewards(&path, std::slice::from_ref(&row)).expect("write rewards parquet");

        let file = File::open(&path).expect("open rewards parquet");
        let builder = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
            .expect("parquet reader builder");
        let schema = builder.schema();
        assert!(schema.field_with_name("feasible").is_ok());
        assert!(schema.field_with_name("feasibility_reason_codes").is_ok());
        let mut reader = builder.build().expect("parquet reader");
        let batch = reader.next().expect("one batch").expect("valid batch");
        let feasible = batch
            .column_by_name("feasible")
            .expect("feasible column")
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("boolean feasible");
        assert!(!feasible.value(0));
        let codes = batch
            .column_by_name("feasibility_reason_codes")
            .expect("reason codes column")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("string reason codes");
        assert_eq!(codes.value(0), "[\"lj_out_of_bounds\"]");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
