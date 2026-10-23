#![recursion_limit = "256"]
#![allow(
    dead_code,
    clippy::collapsible_if,
    clippy::collapsible_str_replace,
    clippy::too_many_arguments
)]

use anyhow::{anyhow, Context, Result};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int32Array, Int64Array, LargeStringArray,
    RecordBatch, StringArray, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::{arrow_reader::ParquetRecordBatchReaderBuilder, ArrowWriter};
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use prism_forge::core::synthon::{AttachmentPoint, ScaffoldState3D};
use prism_forge::reactions::kinematic_assembly::{
    execute_smarts_zmatrix_reaction, execute_zmatrix_reaction, load_synthon_library_from_parquet,
    AssemblyReactionRule, LoadedSynthon, Product3D, SynthonLibrary, ZMatrixAssemblyRule,
};
use prism_forge::reactions::kinematics::{
    execute_3d_reaction, DihedralConstraint, ReactionKinematicRule,
};
use prism_forge::reactions::reaction_registry::{AssemblyPlan, ReactionRegistry};
use prism_forge::scoring::{score_positions_with_field, VoxelField};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::env;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_SYNTHONS: &str =
    "campaigns/glp1r_aleniglipron/track_a_generative/enamine_115k_synthons_3d.parquet";
const DEFAULT_SCAFFOLD_SDF: &str =
    "campaigns/glp1r_aleniglipron/track_a_generative/ALENI-PARENT_6XOX_frame_o3a_relaxed.sdf";
const DEFAULT_FRAGMENT_REGISTRY: &str =
    "campaigns/glp1r_aleniglipron/track_0_manual_emulation/aleniglipron_brics_fragment_registry.json";
const DEFAULT_SIGNAL_GRID: &str =
    "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/signal_grid_variance_channel.parquet";
const DEFAULT_RESIDUE_PHASE_TENSOR: &str =
    "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/residue_phase_tensor.parquet";
const DEFAULT_PHASE_COHERENCE: &str =
    "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/phase_manifold_coherence.parquet";
const DEFAULT_VOXEL_THRESHOLDS: &str =
    "campaigns/glp1r_aleniglipron/track_a_generative/voxel_thresholds.json";
const DEFAULT_REACTION_REGISTRY: &str = "00_registry/chemistry/reaction_rules.v1.yml";
const DEFAULT_FULLSCALE_DRY_RUN_PLAN: &str =
    "campaigns/glp1r_aleniglipron/track_a_generative/vspace_fullscale_dry_run_plan.json";
const DEFAULT_DENDRITIC_PLAN: &str =
    "campaigns/glp1r_aleniglipron/track_a_generative/vspace_38b_dendritic_plan.json";
const DEFAULT_SHARD_PLAN: &str =
    "campaigns/glp1r_aleniglipron/track_a_generative/shard_plan.parquet";
const DEFAULT_RESUME_LEDGER: &str =
    "campaigns/glp1r_aleniglipron/track_a_generative/resume_ledger.json";
const DEFAULT_FULLSCALE_SHARD_DIR: &str =
    "campaigns/glp1r_aleniglipron/track_a_generative/fullscale_shards";
const DEFAULT_SELECTED_LIGAND_POSE: &str =
    "campaigns/glp1r_aleniglipron/track_a_generative/selected_ligand_pose.json";
const DEFAULT_GRID_MAPPING: &str =
    "campaigns/glp1r_aleniglipron/track_0_manual_emulation/grid_coordinate_mapping.json";
const DEFAULT_PATHWAY_NODES: &str =
    "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/translation_pathway_nodes.parquet";
const DEFAULT_OUTPUT: &str =
    "campaigns/glp1r_aleniglipron/track_a_generative/vspace_survivors_full_scale.parquet";
const NORMAL_CLASH_LIMIT: f64 = 2.5;
const HARD_CLASH_LIMIT: f64 = 3.5;
const NORMAL_COMPLEMENT_FLOOR: f64 = 1.0;
const RESCUE_COMPLEMENT_FLOOR: f64 = 0.5;
const CRYPTIC_RESCUE_FLOOR: f64 = 2.0;
const FRAGMENT_CONTEXT_EXCLUSION_A: f64 = 2.32;
const DEFAULT_SURVIVOR_LIMIT: usize = 9_750;
const DEFAULT_STREAM_ROW_GROUP_SIZE: usize = 10_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AssemblyMode {
    Rigid,
    ZMatrix,
    SmartsZMatrix,
}

impl AssemblyMode {
    fn label(self) -> &'static str {
        match self {
            AssemblyMode::Rigid => "rigid",
            AssemblyMode::ZMatrix => "zmatrix",
            AssemblyMode::SmartsZMatrix => "smarts_zmatrix",
        }
    }

    fn parse(value: &str) -> Result<Self> {
        match value {
            "rigid" => Ok(Self::Rigid),
            "zmatrix" => Ok(Self::ZMatrix),
            "smarts_zmatrix" => Ok(Self::SmartsZMatrix),
            _ => Err(anyhow!(
                "invalid --assembly-mode {value}; expected rigid, zmatrix, or smarts_zmatrix"
            )),
        }
    }
}

#[derive(Debug, Clone)]
struct Config {
    synthons: PathBuf,
    scaffold_sdf: PathBuf,
    fragment_registry: PathBuf,
    signal_grid: PathBuf,
    residue_phase_tensor: PathBuf,
    phase_coherence: PathBuf,
    voxel_thresholds: PathBuf,
    reaction_registry: PathBuf,
    fullscale_dry_run_plan: PathBuf,
    dendritic_plan: PathBuf,
    shard_plan: PathBuf,
    resume_ledger: PathBuf,
    fullscale_shard_dir: PathBuf,
    selected_ligand_pose: PathBuf,
    grid_mapping: PathBuf,
    pathway_nodes: PathBuf,
    output: PathBuf,
    condition_id: String,
    max_pairs: usize,
    shard_index: usize,
    shard_count: usize,
    dendritic_shard: Option<usize>,
    row_group_size: usize,
    grid_dim: i64,
    grid_spacing_a: f32,
    scaffold_bond_length_a: f32,
    bridge_radius_a: f32,
    bridge_anchors: Vec<String>,
    origin: [f32; 3],
    removed_fragment_id: String,
    survivor_limit: usize,
    assembly_mode: AssemblyMode,
    full_scale: bool,
    dry_run_plan: bool,
    force: bool,
    dihedral_samples: usize,
    dihedral_grid_deg: Vec<f32>,
    real_anchors_only: bool,
    allow_ligand_override: bool,
    telemetry_json: Option<PathBuf>,
    release_delta_threshold: Option<f64>,
    gain_delta_threshold: Option<f64>,
    stability_epsilon: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            synthons: PathBuf::from(DEFAULT_SYNTHONS),
            scaffold_sdf: PathBuf::from(DEFAULT_SCAFFOLD_SDF),
            fragment_registry: PathBuf::from(DEFAULT_FRAGMENT_REGISTRY),
            signal_grid: PathBuf::from(DEFAULT_SIGNAL_GRID),
            residue_phase_tensor: PathBuf::from(DEFAULT_RESIDUE_PHASE_TENSOR),
            phase_coherence: PathBuf::from(DEFAULT_PHASE_COHERENCE),
            voxel_thresholds: PathBuf::from(DEFAULT_VOXEL_THRESHOLDS),
            reaction_registry: PathBuf::from(DEFAULT_REACTION_REGISTRY),
            fullscale_dry_run_plan: PathBuf::from(DEFAULT_FULLSCALE_DRY_RUN_PLAN),
            dendritic_plan: PathBuf::from(DEFAULT_DENDRITIC_PLAN),
            shard_plan: PathBuf::from(DEFAULT_SHARD_PLAN),
            resume_ledger: PathBuf::from(DEFAULT_RESUME_LEDGER),
            fullscale_shard_dir: PathBuf::from(DEFAULT_FULLSCALE_SHARD_DIR),
            selected_ligand_pose: PathBuf::from(DEFAULT_SELECTED_LIGAND_POSE),
            grid_mapping: PathBuf::from(DEFAULT_GRID_MAPPING),
            pathway_nodes: PathBuf::from(DEFAULT_PATHWAY_NODES),
            output: PathBuf::from(DEFAULT_OUTPUT),
            condition_id: "glp1r_6XOX_WT".to_owned(),
            max_pairs: 100_000,
            shard_index: 0,
            shard_count: 1,
            dendritic_shard: None,
            row_group_size: DEFAULT_STREAM_ROW_GROUP_SIZE,
            grid_dim: 96,
            grid_spacing_a: 1.056_475_9,
            scaffold_bond_length_a: 1.50,
            bridge_radius_a: 4.0,
            bridge_anchors: vec!["ASN182".to_owned()],
            origin: [-20.234_434, -61.996_574, -79.476_11],
            removed_fragment_id: "FRAG-A".to_owned(),
            survivor_limit: DEFAULT_SURVIVOR_LIMIT,
            assembly_mode: AssemblyMode::SmartsZMatrix,
            full_scale: true,
            dry_run_plan: false,
            force: false,
            dihedral_samples: 6,
            dihedral_grid_deg: vec![0.0, 60.0, 120.0, 180.0, 240.0, 300.0],
            real_anchors_only: false,
            allow_ligand_override: false,
            telemetry_json: None,
            release_delta_threshold: None,
            gain_delta_threshold: None,
            stability_epsilon: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VarianceClass {
    StableOccupied,
    ThermallyActivated,
    ThermallyDestabilized,
}

impl VarianceClass {
    fn label(self) -> &'static str {
        match self {
            VarianceClass::StableOccupied => "stable_occupied",
            VarianceClass::ThermallyActivated => "thermally_activated",
            VarianceClass::ThermallyDestabilized => "thermally_destabilized",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct GridGeometry {
    nx: i64,
    ny: i64,
    nz: i64,
    spacing_a: f32,
    origin: [f32; 3],
}

#[derive(Debug, Clone)]
struct PathwayContext {
    voxel_to_residue: HashMap<u64, i64>,
    kinetic_burst_residues: HashSet<i64>,
}

#[derive(Debug, Clone)]
struct FieldAudit {
    columns: Vec<String>,
    condition_counts: BTreeMap<String, u64>,
    raw_variance_counts: BTreeMap<String, u64>,
    boundary_class_counts: Option<BTreeMap<String, u64>>,
    voxel_context_counts: Option<BTreeMap<String, u64>>,
    effective_variance_counts: BTreeMap<String, u64>,
    hit_count_delta_count: u64,
    hit_count_delta_min: f64,
    hit_count_delta_max: f64,
    hit_count_delta_mean: f64,
    hit_count_delta_p90: f64,
    release_delta_count: u64,
    release_delta_min: f64,
    release_delta_max: f64,
    release_delta_mean: f64,
    release_delta_positive_p90: f64,
    release_delta_threshold: f64,
    gain_delta_positive_p90: f64,
    gain_delta_threshold: f64,
    stability_epsilon: f64,
    fallback_enabled: bool,
    has_voxel_activated_fraction: bool,
}

#[derive(Debug, Clone)]
struct ProtocolResidueContext {
    release_delta: f64,
    gain_delta: f64,
    stable_occupied: bool,
}

#[derive(Debug, Clone)]
struct ProtocolContext {
    residues: HashMap<i64, ProtocolResidueContext>,
    release_delta_threshold: f64,
    gain_delta_threshold: f64,
}

#[derive(Debug, Clone, Copy)]
struct VoxelThresholds {
    cold_p80: f64,
    warm_p80: f64,
    gain_p90: f64,
    release_p90: f64,
    cold_p95: f64,
    warm_p95: f64,
    release_p95: f64,
    nonvoid_voxel_count: u64,
}

#[derive(Debug, Clone, Copy)]
struct DynamicVoxelClass {
    stable_occupied: bool,
    thermally_destabilized: bool,
    thermally_activated: bool,
    thermally_released: bool,
    gain_delta: f64,
    release_delta: f64,
}

#[derive(Debug, Clone, Copy)]
struct ResidueCoordinate {
    residue_idx: i64,
    xyz: [f64; 3],
}

#[derive(Debug, Clone)]
struct Survivor {
    anchor_id: String,
    canonical_smiles: String,
    product_id: String,
    smiles: String,
    synthon_a_id: String,
    synthon_b_id: String,
    score: f64,
    fragment_pi_complement: f64,
    fragment_pi_clash_adjusted: f64,
    pi_complement: f64,
    pi_clash: f64,
    cryptic_bonus: f64,
    cryptic_bonus_atoms: u64,
    survival_tier: &'static str,
    selected_dihedral_deg: f64,
    assembly_mode: &'static str,
    z_matrix_active: bool,
    rotamers_evaluated: u64,
    best_rotamer_rank: u64,
    ligand_sdf_path: String,
    pose_reconciliation_method: String,
    coordinates_json: String,
}

#[derive(Debug, Clone)]
struct SdfMol {
    coordinates: Vec<[f32; 3]>,
    elements: Vec<String>,
    bonds: Vec<(usize, usize)>,
}

#[derive(Debug, Clone, Copy)]
struct LigandGridDiagnostics {
    centroid: [f64; 3],
    inside_grid: bool,
    min_distance_to_grid_a: f64,
    max_distance_to_grid_a: f64,
}

#[derive(Debug, Clone)]
struct SelectedLigandPose {
    selected_ligand_sdf: String,
    selected_pose_method: String,
    guard_level_passed: String,
    o3a_score: f64,
}

#[derive(Debug, Clone, Copy)]
struct FieldCounts {
    stable_occupied: usize,
    thermally_destabilized: usize,
    thermally_activated: usize,
    thermally_released: usize,
    complement_voxels: usize,
    cryptic_bonus_voxels: usize,
}

#[derive(Debug, Clone)]
struct ScaffoldAnchor {
    original_scaffold_atom_idx: usize,
    removed_fragment_atom_idx: usize,
    scaffold_ref1_atom_idx: usize,
    scaffold_ref2_atom_idx: usize,
}

#[derive(Debug, Clone)]
struct ScaffoldTemplate {
    state: ScaffoldState3D,
    anchors: Vec<ScaffoldAnchor>,
}

#[derive(Debug, Clone)]
struct AnchoredProduct {
    product: Product3D,
    synthon_a_id: String,
    synthon_b_id: String,
    score_atom_offset: usize,
    score_atom_atomic_numbers: Vec<u8>,
    synthon_index_map: Vec<Option<usize>>,
    selected_dihedral_deg: f64,
    assembly_mode: &'static str,
    z_matrix_active: bool,
    rotamers_evaluated: u64,
    best_rotamer_rank: u64,
}

#[derive(Debug, Clone)]
struct DendriticShard {
    requested_shard_id: usize,
    plan_shard_id: usize,
    pathway_id: String,
    first_reaction_id: String,
    second_reaction_id: String,
    pathway_start_pair_idx: usize,
    pathway_end_pair_idx_exclusive: usize,
    pair_count: usize,
    output: PathBuf,
}

#[derive(Debug, Clone)]
struct DendriticTelemetry {
    attempted_a: usize,
    intermediate_a_dropped: usize,
    attempted_b: usize,
    final_survivors: usize,
}

#[derive(Debug)]
struct FieldDiagnostics {
    complement_centers: Vec<[f64; 3]>,
    cryptic_bonus_centers: Vec<[f64; 3]>,
    clash_centers: Vec<[f64; 3]>,
    no_fly_centers: Vec<[f64; 3]>,
    complement_voxels: HashSet<u64>,
    cryptic_bonus_voxels: HashSet<u64>,
    nearest_complement_cache: Mutex<HashMap<u64, Option<u64>>>,
    nearest_cryptic_cache: Mutex<HashMap<u64, Option<u64>>>,
}

#[derive(Debug, Clone, Copy)]
struct PlacementMetrics {
    min_atom_to_complement_a: f64,
    min_atom_to_cryptic_bonus_a: f64,
    atoms_in_complement_voxels: usize,
    atoms_within_1_voxel_of_complement: usize,
    atoms_within_2_voxels_of_complement: usize,
    coherence_missing_residue_hits: usize,
    coherence_factor_sum_for_clashes: f64,
    coherence_factor_count_for_clashes: usize,
    min_coherence_factor_for_clashes: f64,
    max_coherence_factor_for_clashes: f64,
    raw_pi_clash_sum: f64,
    adjusted_pi_clash_sum: f64,
    scaffold_context_pi_complement: f64,
    scaffold_context_pi_clash: f64,
    fragment_pi_complement: f64,
    fragment_pi_clash_adjusted: f64,
    fragment_cryptic_bonus: f64,
}

fn main() -> Result<()> {
    let config = parse_args()?;
    let geometry = load_grid_geometry(&config)?;
    let selected_pose =
        load_selected_ligand_pose(&config.selected_ligand_pose).with_context(|| {
            format!(
                "load selected ligand pose {}",
                config.selected_ligand_pose.display()
            )
        })?;
    validate_selected_ligand_pose(&selected_pose, &config)?;
    let ligand_grid = ligand_grid_diagnostics(&config.scaffold_sdf, geometry)
        .with_context(|| format!("diagnose ligand SDF {}", config.scaffold_sdf.display()))?;
    print_ligand_grid_diagnostics(&config.scaffold_sdf, geometry, &ligand_grid);
    println!(
        "selected_pose selected_ligand_sdf={} selected_pose_method={} o3a_score={:.6} guard_level_passed={} ligand_inside_grid={} ligand_centroid_xyz=[{:.6},{:.6},{:.6}] grid_min_xyz=[{:.6},{:.6},{:.6}] grid_max_xyz=[{:.6},{:.6},{:.6}]",
        selected_pose.selected_ligand_sdf,
        selected_pose.selected_pose_method,
        selected_pose.o3a_score,
        selected_pose.guard_level_passed,
        ligand_grid.inside_grid,
        ligand_grid.centroid[0],
        ligand_grid.centroid[1],
        ligand_grid.centroid[2],
        grid_min_xyz(geometry)[0],
        grid_min_xyz(geometry)[1],
        grid_min_xyz(geometry)[2],
        grid_max_xyz(geometry)[0],
        grid_max_xyz(geometry)[1],
        grid_max_xyz(geometry)[2],
    );
    if !ligand_grid.inside_grid {
        return Err(anyhow!(
            "ligand_inside_grid == false for {}",
            config.scaffold_sdf.display()
        ));
    }
    let library = load_synthon_library_from_parquet(&config.synthons)
        .with_context(|| format!("load synthons {}", config.synthons.display()))?;
    if !config.full_scale && config.max_pairs == 0 {
        return Err(anyhow!("--max-pairs is required when --full-scale false"));
    }
    if config.shard_count == 0 {
        return Err(anyhow!("--shard-count must be greater than 0"));
    }
    if config.shard_index >= config.shard_count {
        return Err(anyhow!(
            "--shard-index {} must be less than --shard-count {}",
            config.shard_index,
            config.shard_count
        ));
    }
    if config.row_group_size == 0 {
        return Err(anyhow!("--row-group-size must be greater than 0"));
    }
    let registry = if matches!(config.assembly_mode, AssemblyMode::SmartsZMatrix) {
        Some(
            ReactionRegistry::load(&config.reaction_registry).with_context(|| {
                format!(
                    "load reaction registry {}",
                    config.reaction_registry.display()
                )
            })?,
        )
    } else {
        None
    };
    let mock_anchor_count = library
        .ids()
        .filter(|id| id.to_ascii_uppercase().contains("MOCK"))
        .count();
    println!(
        "anchor_library path={} real_anchor_count_loaded={} mock_anchor_count={} real_anchors_only={}",
        config.synthons.display(),
        library.len(),
        mock_anchor_count,
        config.real_anchors_only
    );
    if matches!(config.assembly_mode, AssemblyMode::SmartsZMatrix) {
        let registry = registry
            .as_ref()
            .expect("registry loaded for smarts_zmatrix");
        let enabled = registry
            .enabled_reactions()
            .map(|reaction| reaction.reaction_id.as_str())
            .collect::<Vec<_>>()
            .join(",");
        println!(
            "reaction_registry loaded path={} enabled_reactions={}",
            config.reaction_registry.display(),
            enabled
        );
        if mock_anchor_count > 0 {
            return Err(anyhow!(
                "SMARTS V-space path refuses mock synthons: mock_anchor_count={mock_anchor_count}"
            ));
        }
        if config.full_scale && config.dry_run_plan {
            write_fullscale_dry_run_plan(&config, &library, registry)?;
            println!(
                "fullscale_dry_run_plan_written path={} full_scale_execution=false",
                config.fullscale_dry_run_plan.display()
            );
            return Ok(());
        }
    }
    if config.real_anchors_only && mock_anchor_count > 0 {
        return Err(anyhow!(
            "--real-anchors-only true but {mock_anchor_count} mock anchors were loaded"
        ));
    }
    let scaffold = load_scaffold_template(&config)?;
    let coherence = load_phase_coherence_map(&config.phase_coherence, &config.condition_id)?;
    let pathway_context = load_pathway_context(&config.pathway_nodes, &config.condition_id)?;
    let protocol_context =
        load_protocol_context(&config.residue_phase_tensor, &config.condition_id, &config)?;
    let voxel_thresholds = load_voxel_thresholds(&config.voxel_thresholds, &config.condition_id)?;
    let residue_coordinates = load_residue_coordinates(&config)?;
    println!(
        "phase_manifold_coherence loaded path={} condition={} residues={} voxel_residue_links={} kinetic_burst_residues={}",
        config.phase_coherence.display(),
        config.condition_id,
        coherence.len(),
        pathway_context.voxel_to_residue.len(),
        pathway_context.kinetic_burst_residues.len()
    );
    println!(
        "protocol_context loaded path={} condition={} residues={} release_delta_threshold={:.6} gain_delta_threshold={:.6} topology_residues={}",
        config.residue_phase_tensor.display(),
        config.condition_id,
        protocol_context.residues.len(),
        protocol_context.release_delta_threshold,
        protocol_context.gain_delta_threshold,
        residue_coordinates.len()
    );
    println!(
        "voxel_thresholds loaded path={} condition={} nonvoid_voxel_count={} cold_p80={:.6} warm_p80={:.6} gain_p90={:.6} release_p90={:.6} cold_p95={:.6} warm_p95={:.6} release_p95={:.6}",
        config.voxel_thresholds.display(),
        config.condition_id,
        voxel_thresholds.nonvoid_voxel_count,
        voxel_thresholds.cold_p80,
        voxel_thresholds.warm_p80,
        voxel_thresholds.gain_p90,
        voxel_thresholds.release_p90,
        voxel_thresholds.cold_p95,
        voxel_thresholds.warm_p95,
        voxel_thresholds.release_p95
    );
    let field = Arc::new(load_signal_field(
        &config.signal_grid,
        &config.condition_id,
        &coherence,
        &pathway_context,
        &voxel_thresholds,
        &residue_coordinates,
        geometry,
        &config,
    )?);
    let no_fly = Arc::new(load_bridge_no_fly_voxels(
        &config.pathway_nodes,
        &config.condition_id,
        geometry,
        config.bridge_radius_a,
        &config.bridge_anchors,
    )?);
    let field_counts = field_counts(&field);
    let complement_voxels = field_counts.complement_voxels;
    let cryptic_bonus_voxels = field_counts.cryptic_bonus_voxels;
    let coherence_adjusted_voxels = field
        .values()
        .filter(|voxel| voxel.primary_residue_idx.is_some() && voxel.coherence_score < 1.0)
        .count();
    println!(
        "grid condition={} origin=[{:.6},{:.6},{:.6}] spacing_A={:.6} dims={}x{}x{} field_voxels={} complement_voxels={} cryptic_bonus_voxels={} coherence_adjusted_voxels={} no_fly_voxels={}",
        config.condition_id,
        geometry.origin[0],
        geometry.origin[1],
        geometry.origin[2],
        geometry.spacing_a,
        geometry.nx,
        geometry.ny,
        geometry.nz,
        field.len(),
        complement_voxels,
        cryptic_bonus_voxels,
        coherence_adjusted_voxels,
        no_fly.len()
    );
    if complement_voxels == 0 {
        return Err(anyhow!(
            "complement_voxels == 0 after release-direction fallback for {}",
            config.condition_id
        ));
    }
    if field_counts.stable_occupied == 0 {
        return Err(anyhow!("stable_occupied == 0 for {}", config.condition_id));
    }
    let field_diagnostics = Arc::new(FieldDiagnostics::new(&field, &no_fly, geometry));
    print_anchor_field_diagnostics(&scaffold, &field_diagnostics, geometry);
    let rule = if matches!(config.assembly_mode, AssemblyMode::SmartsZMatrix) {
        smarts_primary_rule(
            registry
                .as_ref()
                .expect("registry loaded for smarts_zmatrix"),
            &library,
        )?
    } else {
        AssemblyReactionRule::amide_coupling()
    };
    let mut dendritic_telemetry: Option<DendriticTelemetry> = None;
    let mut effective_output = config.output.clone();
    let (survivor_count, telemetry) = if let Some(requested_shard_id) = config.dendritic_shard {
        let registry_ref = registry
            .as_ref()
            .ok_or_else(|| anyhow!("--shard requires --assembly-mode smarts_zmatrix"))?;
        let shard = load_dendritic_shard(&config, requested_shard_id)?;
        effective_output = shard.output.clone();
        let (survivor_count, telemetry, dendritic) = prune_dendritic_shard_streaming(
            &library,
            &scaffold,
            registry_ref,
            &field,
            &no_fly,
            &field_diagnostics,
            &config,
            geometry,
            &shard,
        )?;
        update_resume_ledger(&config.resume_ledger, &shard, &dendritic)?;
        dendritic_telemetry = Some(dendritic);
        (survivor_count, telemetry)
    } else if config.full_scale && !config.dry_run_plan {
        prune_rule_streaming(
            &library,
            &scaffold,
            &rule,
            registry.as_ref(),
            &field,
            &no_fly,
            &field_diagnostics,
            &config,
            geometry,
        )?
    } else {
        let (survivors, telemetry) = prune_rule(
            &library,
            &scaffold,
            &rule,
            registry.as_ref(),
            &field,
            &no_fly,
            &field_diagnostics,
            &config,
            geometry,
        )?;
        let survivor_count = survivors.len();
        write_survivors(&config.output, &survivors)?;
        (survivor_count, telemetry)
    };
    if library.len() < 512 {
        return Err(anyhow!(
            "real_anchor_count_loaded {} is below 512",
            library.len()
        ));
    }
    if telemetry.z_matrix_active_count == 0 {
        return Err(anyhow!("z_matrix_active_count == 0"));
    }
    if telemetry.rigid_fallback_count > 0 {
        return Err(anyhow!(
            "rigid_fallback_count {} is greater than 0",
            telemetry.rigid_fallback_count
        ));
    }
    if no_fly.is_empty() {
        return Err(anyhow!("no_fly_voxels == 0"));
    }
    if let Some(path) = &config.telemetry_json {
        write_telemetry_json(
            path,
            &telemetry,
            &config,
            &selected_pose,
            &ligand_grid,
            geometry,
            field_counts,
            no_fly.len(),
            survivor_count,
            library.len(),
            mock_anchor_count,
        )?;
    }
    println!(
        "telemetry attempted={} dropped_assembly={} dropped_bounds={} dropped_no_fly={} dropped_hard_clash={} dropped_insufficient_complement={} survived_normal={} survived_cryptic_rescue={} survivors={} rank_pruned_survivors={} cryptic_bonus_atoms={} complement_voxels={} cryptic_bonus_voxels={} no_fly_voxels={} ligand_inside_grid={} exit_vector_flipped_count=0",
        telemetry.attempted,
        telemetry.dropped_assembly,
        telemetry.dropped_bounds,
        telemetry.dropped_no_fly,
        telemetry.dropped_hard_clash,
        telemetry.dropped_insufficient_complement,
        telemetry.survived_normal,
        telemetry.survived_cryptic_rescue,
        survivor_count,
        telemetry
            .survived_normal
            .saturating_add(telemetry.survived_cryptic_rescue)
            .saturating_sub(survivor_count),
        telemetry.cryptic_bonus_atoms,
        complement_voxels,
        cryptic_bonus_voxels,
        no_fly.len(),
        ligand_grid.inside_grid
    );
    println!(
        "assembly_telemetry assembly_mode={} dihedral_samples={} rotamers_evaluated={} rotamers_dropped_bounds={} rotamers_dropped_no_fly={} rotamers_dropped_hard_clash={} candidates_with_at_least_one_valid_rotamer={} candidate_survivors={} mean_best_dihedral_deg={:.6} top_dihedral_bins={} z_matrix_active_count={} rigid_fallback_count={}",
        config.assembly_mode.label(),
        config.dihedral_samples,
        telemetry.rotamers_evaluated,
        telemetry.rotamers_dropped_bounds,
        telemetry.rotamers_dropped_no_fly,
        telemetry.rotamers_dropped_hard_clash,
        telemetry.candidates_with_at_least_one_valid_rotamer,
        survivor_count,
        telemetry.mean_best_dihedral_deg,
        telemetry.top_dihedral_bins,
        telemetry.z_matrix_active_count,
        telemetry.rigid_fallback_count
    );
    println!(
        "coherence_telemetry coherence_loaded_residues={} coherence_missing_residue_hits={} mean_coherence_factor_for_clashes={:.6} min_coherence_factor_for_clashes={:.6} max_coherence_factor_for_clashes={:.6} raw_pi_clash_sum={:.6} adjusted_pi_clash_sum={:.6}",
        coherence.len(),
        telemetry.coherence_missing_residue_hits,
        telemetry.mean_coherence_factor_for_clashes,
        telemetry.min_coherence_factor_for_clashes,
        telemetry.max_coherence_factor_for_clashes,
        telemetry.raw_pi_clash_sum,
        telemetry.adjusted_pi_clash_sum
    );
    println!(
        "fragment_only_telemetry mean_scaffold_context_pi_complement={:.6} mean_fragment_pi_complement={:.6} mean_fragment_pi_clash_adjusted={:.6} mean_fragment_cryptic_bonus={:.6} candidates_with_fragment_complement={} candidates_with_fragment_cryptic_bonus={}",
        telemetry.mean_scaffold_context_pi_complement,
        telemetry.mean_fragment_pi_complement,
        telemetry.mean_fragment_pi_clash_adjusted,
        telemetry.mean_fragment_cryptic_bonus,
        telemetry.candidates_with_fragment_complement,
        telemetry.candidates_with_fragment_cryptic_bonus
    );
    println!(
        "placement_telemetry mean_min_atom_to_complement_A={:.6} p10_min_atom_to_complement_A={:.6} min_seen_atom_to_complement_A={:.6} candidates_with_atoms_within_1_voxel_complement={} candidates_with_atoms_within_2_voxels_complement={}",
        telemetry.mean_min_atom_to_complement_a,
        telemetry.p10_min_atom_to_complement_a,
        telemetry.min_seen_atom_to_complement_a,
        telemetry.candidates_with_atoms_within_1_voxel_complement,
        telemetry.candidates_with_atoms_within_2_voxels_complement
    );
    if let Some(dendritic) = dendritic_telemetry {
        println!(
            "dendritic_shard_telemetry attempted_a={} intermediate_a_dropped={} attempted_b={} final_survivors={}",
            dendritic.attempted_a,
            dendritic.intermediate_a_dropped,
            dendritic.attempted_b,
            dendritic.final_survivors
        );
    }
    println!(
        "wrote {} survivors={} max_pairs={} scaffold_anchors={}",
        effective_output.display(),
        survivor_count,
        config.max_pairs,
        scaffold.anchors.len()
    );
    Ok(())
}

fn parse_args() -> Result<Config> {
    let mut config = Config::default();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        let value = match arg.as_str() {
            "--synthons" => args.next().map(PathBuf::from).map(|value| {
                config.synthons = value;
            }),
            "--anchor-parquet" => args.next().map(PathBuf::from).map(|value| {
                config.synthons = value;
            }),
            "--synthon-parquet" => args.next().map(PathBuf::from).map(|value| {
                config.synthons = value;
            }),
            "--scaffold-sdf" | "--ligand-sdf" => args.next().map(PathBuf::from).map(|value| {
                config.scaffold_sdf = value;
            }),
            "--fragment-registry" => args.next().map(PathBuf::from).map(|value| {
                config.fragment_registry = value;
            }),
            "--signal-grid" => args.next().map(PathBuf::from).map(|value| {
                config.signal_grid = value;
            }),
            "--residue-phase-tensor" => args.next().map(PathBuf::from).map(|value| {
                config.residue_phase_tensor = value;
            }),
            "--phase-coherence" => args.next().map(PathBuf::from).map(|value| {
                config.phase_coherence = value;
            }),
            "--voxel-thresholds" => args.next().map(PathBuf::from).map(|value| {
                config.voxel_thresholds = value;
            }),
            "--reaction-registry" => args.next().map(PathBuf::from).map(|value| {
                config.reaction_registry = value;
            }),
            "--fullscale-dry-run-plan" => args.next().map(PathBuf::from).map(|value| {
                config.fullscale_dry_run_plan = value;
            }),
            "--dendritic-plan" => args.next().map(PathBuf::from).map(|value| {
                config.dendritic_plan = value;
            }),
            "--shard-plan" => args.next().map(PathBuf::from).map(|value| {
                config.shard_plan = value;
            }),
            "--resume-ledger" => args.next().map(PathBuf::from).map(|value| {
                config.resume_ledger = value;
            }),
            "--fullscale-shard-dir" => args.next().map(PathBuf::from).map(|value| {
                config.fullscale_shard_dir = value;
            }),
            "--selected-ligand-pose" => args.next().map(PathBuf::from).map(|value| {
                config.selected_ligand_pose = value;
            }),
            "--grid-mapping" => args.next().map(PathBuf::from).map(|value| {
                config.grid_mapping = value;
            }),
            "--pathway-nodes" => args.next().map(PathBuf::from).map(|value| {
                config.pathway_nodes = value;
            }),
            "--output" => args.next().map(PathBuf::from).map(|value| {
                config.output = value;
            }),
            "--telemetry-json" => args.next().map(PathBuf::from).map(|value| {
                config.telemetry_json = Some(value);
            }),
            "--condition-id" => args.next().map(|value| {
                config.condition_id = value;
            }),
            "--max-pairs" => args.next().map(|value| {
                config.max_pairs = value.parse().unwrap_or(config.max_pairs);
            }),
            "--shard-index" => args.next().map(|value| {
                config.shard_index = value
                    .parse()
                    .unwrap_or_else(|_| panic!("invalid --shard-index {value}"));
            }),
            "--shard-count" => args.next().map(|value| {
                config.shard_count = value
                    .parse()
                    .unwrap_or_else(|_| panic!("invalid --shard-count {value}"));
            }),
            "--shard" => args.next().map(|value| {
                config.dendritic_shard = Some(
                    value
                        .parse()
                        .unwrap_or_else(|_| panic!("invalid --shard {value}")),
                );
            }),
            "--row-group-size" => args.next().map(|value| {
                config.row_group_size = value
                    .parse()
                    .unwrap_or_else(|_| panic!("invalid --row-group-size {value}"));
            }),
            "--threshold" => args.next().map(|_| ()),
            "--grid-dim" => args.next().map(|value| {
                config.grid_dim = value.parse().unwrap_or(config.grid_dim);
            }),
            "--grid-spacing-a" => args.next().map(|value| {
                config.grid_spacing_a = value.parse().unwrap_or(config.grid_spacing_a);
            }),
            "--scaffold-bond-length-a" => args.next().map(|value| {
                config.scaffold_bond_length_a =
                    value.parse().unwrap_or(config.scaffold_bond_length_a);
            }),
            "--bridge-radius-a" => args.next().map(|value| {
                config.bridge_radius_a = value.parse().unwrap_or(config.bridge_radius_a);
            }),
            "--removed-fragment-id" => args.next().map(|value| {
                config.removed_fragment_id = value;
            }),
            "--survivor-limit" => args.next().map(|value| {
                config.survivor_limit = value
                    .parse()
                    .unwrap_or_else(|_| panic!("invalid --survivor-limit {value}"));
            }),
            "--assembly-mode" => args.next().map(|value| {
                config.assembly_mode =
                    AssemblyMode::parse(&value).unwrap_or_else(|err| panic!("{err}"));
            }),
            "--dihedral-samples" => args.next().map(|value| {
                config.dihedral_samples = value
                    .parse()
                    .unwrap_or_else(|_| panic!("invalid --dihedral-samples {value}"));
            }),
            "--dihedral-grid-deg" => args.next().map(|value| {
                config.dihedral_grid_deg = value
                    .split(',')
                    .filter(|item| !item.is_empty())
                    .map(|item| {
                        item.parse::<f32>()
                            .unwrap_or_else(|_| panic!("invalid dihedral degree {item}"))
                    })
                    .collect();
            }),
            "--real-anchors-only" => args.next().map(|value| {
                config.real_anchors_only = parse_bool(&value)
                    .unwrap_or_else(|| panic!("invalid --real-anchors-only {value}"));
            }),
            "--full-scale" => args.next().map(|value| {
                config.full_scale =
                    parse_bool(&value).unwrap_or_else(|| panic!("invalid --full-scale {value}"));
            }),
            "--dry-run-plan" => args.next().map(|value| {
                config.dry_run_plan =
                    parse_bool(&value).unwrap_or_else(|| panic!("invalid --dry-run-plan {value}"));
            }),
            "--force" => args.next().map(|value| {
                config.force =
                    parse_bool(&value).unwrap_or_else(|| panic!("invalid --force {value}"));
            }),
            "--allow-ligand-override" => args.next().map(|value| {
                config.allow_ligand_override = parse_bool(&value)
                    .unwrap_or_else(|| panic!("invalid --allow-ligand-override {value}"));
            }),
            "--release-delta-threshold" => args.next().map(|value| {
                config.release_delta_threshold = Some(
                    value
                        .parse()
                        .unwrap_or_else(|_| panic!("invalid --release-delta-threshold {value}")),
                );
            }),
            "--gain-delta-threshold" => args.next().map(|value| {
                config.gain_delta_threshold = Some(
                    value
                        .parse()
                        .unwrap_or_else(|_| panic!("invalid --gain-delta-threshold {value}")),
                );
            }),
            "--stability-epsilon" => args.next().map(|value| {
                config.stability_epsilon = value
                    .parse()
                    .unwrap_or_else(|_| panic!("invalid --stability-epsilon {value}"));
            }),
            "--bridge-anchors" => args.next().map(|value| {
                config.bridge_anchors = if value == "all" {
                    Vec::new()
                } else {
                    value
                        .split(',')
                        .filter(|item| !item.is_empty())
                        .map(ToOwned::to_owned)
                        .collect()
                };
            }),
            "--origin-x" => args.next().map(|value| {
                config.origin[0] = value.parse().unwrap_or(config.origin[0]);
            }),
            "--origin-y" => args.next().map(|value| {
                config.origin[1] = value.parse().unwrap_or(config.origin[1]);
            }),
            "--origin-z" => args.next().map(|value| {
                config.origin[2] = value.parse().unwrap_or(config.origin[2]);
            }),
            _ => return Err(anyhow!("unknown argument: {arg}")),
        };
        if value.is_none() {
            return Err(anyhow!("missing value for {arg}"));
        }
    }
    Ok(config)
}

fn parse_bool(value: &str) -> Option<bool> {
    match value {
        "true" | "1" | "yes" | "y" => Some(true),
        "false" | "0" | "no" | "n" => Some(false),
        _ => None,
    }
}

fn load_dendritic_shard(config: &Config, requested_shard_id: usize) -> Result<DendriticShard> {
    let plan_text = std::fs::read_to_string(&config.dendritic_plan)
        .with_context(|| format!("read {}", config.dendritic_plan.display()))?;
    let plan: Value = serde_json::from_str(&plan_text)
        .with_context(|| format!("parse {}", config.dendritic_plan.display()))?;
    let master_total_pairs = plan
        .get("total_valid_pairs")
        .and_then(Value::as_u64)
        .unwrap_or_default();
    let mut reader = parquet_reader(&config.shard_plan)
        .with_context(|| format!("load shard plan {}", config.shard_plan.display()))?;
    let mut two_step_rows = Vec::new();
    for batch in &mut reader {
        let batch = batch?;
        for row in 0..batch.num_rows() {
            if string_value(&batch, "pathway_type", row)? != "2-step" {
                continue;
            }
            let pathway_id = string_value(&batch, "pathway_id", row)?;
            let parts = pathway_id.split("__").collect::<Vec<_>>();
            if parts.len() != 3 || parts[0] != "TWO_STEP" {
                return Err(anyhow!("invalid two-step pathway_id {pathway_id}"));
            }
            let first_reaction_id = parts[1].to_owned();
            let second_reaction_id = parts[2].to_owned();
            two_step_rows.push(DendriticShard {
                requested_shard_id,
                plan_shard_id: usize::try_from(i64_value(&batch, "shard_id", row)?)?,
                pathway_id,
                first_reaction_id,
                second_reaction_id,
                pathway_start_pair_idx: usize::try_from(i64_value(
                    &batch,
                    "pathway_start_pair_idx",
                    row,
                )?)?,
                pathway_end_pair_idx_exclusive: usize::try_from(i64_value(
                    &batch,
                    "pathway_end_pair_idx_exclusive",
                    row,
                )?)?,
                pair_count: usize::try_from(i64_value(&batch, "pair_count", row)?)?,
                output: config
                    .fullscale_shard_dir
                    .join(format!("shard_{requested_shard_id:04}.parquet")),
            });
        }
    }
    two_step_rows.sort_by_key(|row| row.plan_shard_id);
    let shard = two_step_rows
        .get(requested_shard_id)
        .cloned()
        .ok_or_else(|| {
            anyhow!(
                "requested dendritic shard {} unavailable; two_step_shards={}",
                requested_shard_id,
                two_step_rows.len()
            )
        })?;
    println!(
        "shard_router requested_shard={} selected_plan_shard={} pathway_id={} plan_total_valid_pairs={} shard_plan_path={} output={}",
        requested_shard_id,
        shard.plan_shard_id,
        shard.pathway_id,
        master_total_pairs,
        config.shard_plan.display(),
        shard.output.display()
    );
    Ok(shard)
}

fn update_resume_ledger(
    path: &Path,
    shard: &DendriticShard,
    telemetry: &DendriticTelemetry,
) -> Result<()> {
    let mut payload = if path.exists() {
        serde_json::from_str::<Value>(&std::fs::read_to_string(path)?)?
    } else {
        json!({})
    };
    if !payload.is_object() {
        payload = json!({});
    }
    let completed = payload
        .as_object_mut()
        .expect("resume ledger object")
        .entry("completed_shards")
        .or_insert_with(|| json!([]));
    let completed_array = completed
        .as_array_mut()
        .ok_or_else(|| anyhow!("resume_ledger completed_shards was not an array"))?;
    completed_array.push(json!({
        "requested_shard_id": shard.requested_shard_id,
        "plan_shard_id": shard.plan_shard_id,
        "pathway_id": shard.pathway_id,
        "status": "COMPLETED",
        "completed_at_unix_s": SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        "output": shard.output.display().to_string(),
        "attempted_a": telemetry.attempted_a,
        "intermediate_a_dropped": telemetry.intermediate_a_dropped,
        "attempted_b": telemetry.attempted_b,
        "final_survivors": telemetry.final_survivors,
    }));
    let object = payload.as_object_mut().expect("resume ledger object");
    object.insert("status".to_owned(), json!("running"));
    object.insert(
        "next_pending_shard_id".to_owned(),
        json!(shard.requested_shard_id.saturating_add(1)),
    );
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(&payload)?)?;
    println!(
        "resume_ledger_updated path={} completed_requested_shard={} completed_plan_shard={}",
        path.display(),
        shard.requested_shard_id,
        shard.plan_shard_id
    );
    Ok(())
}

fn write_fullscale_dry_run_plan(
    config: &Config,
    library: &SynthonLibrary,
    registry: &ReactionRegistry,
) -> Result<()> {
    if config.output.exists() && !config.force {
        return Err(anyhow!(
            "output path {} already exists; use --force true for explicit overwrite planning",
            config.output.display()
        ));
    }
    let mut reaction_pair_counts = serde_json::Map::new();
    let mut compatible_synthons = HashSet::new();
    let mut estimated_pairs: u128 = 0;
    for reaction in registry.enabled_reactions() {
        let synthon_count = library
            .compatible_ids(&reaction.reaction_id, "synthon")
            .len();
        let acid_legacy_count = library.compatible_ids(&reaction.reaction_id, "acid").len();
        let role_count = synthon_count.max(acid_legacy_count);
        for synthon_id in library.compatible_ids(&reaction.reaction_id, "synthon") {
            compatible_synthons.insert(synthon_id.to_owned());
        }
        for synthon_id in library.compatible_ids(&reaction.reaction_id, "acid") {
            compatible_synthons.insert(synthon_id.to_owned());
        }
        reaction_pair_counts.insert(
            reaction.reaction_id.clone(),
            json!({
                "compatible_synthons": role_count,
                "estimated_scaffold_pairs": role_count,
            }),
        );
        estimated_pairs += role_count as u128;
    }
    let estimated_rotamers = estimated_pairs
        .checked_mul(config.dihedral_grid_deg.len().max(1) as u128)
        .ok_or_else(|| anyhow!("estimated rotamer count overflowed"))?;
    let rotamer_threshold: u128 = 250_000_000;
    if estimated_rotamers > rotamer_threshold {
        return Err(anyhow!(
            "estimated_rotamers {estimated_rotamers} exceeds safety threshold {rotamer_threshold}; sharding plan required"
        ));
    }
    if compatible_synthons.is_empty() {
        return Err(anyhow!("no compatible reaction pairs for dry-run plan"));
    }
    let payload = json!({
        "total_synthons": library.len(),
        "compatible_synthons": compatible_synthons.len(),
        "reaction_pair_counts": reaction_pair_counts,
        "estimated_combinatorial_pairs": estimated_pairs,
        "estimated_rotamers": estimated_rotamers,
        "expected_output_path": config.output.display().to_string(),
        "estimated_disk_bytes": estimated_pairs.saturating_mul(512),
        "enabled_reactions": registry.enabled_reactions().map(|reaction| reaction.reaction_id.clone()).collect::<Vec<_>>(),
        "top_reaction_classes_by_count": registry.enabled_reactions().map(|reaction| reaction.reaction_class.clone()).collect::<Vec<_>>(),
        "safety_status": "dry_run_plan_only_full_scale_not_executed"
    });
    if let Some(parent) = config.fullscale_dry_run_plan.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(
        &config.fullscale_dry_run_plan,
        serde_json::to_string_pretty(&payload)?,
    )?;
    Ok(())
}

fn load_selected_ligand_pose(path: &Path) -> Result<SelectedLigandPose> {
    let text = std::fs::read_to_string(path)?;
    let payload: Value = serde_json::from_str(&text)?;
    Ok(SelectedLigandPose {
        selected_ligand_sdf: json_string_required(&payload, "selected_ligand_sdf")?,
        selected_pose_method: json_string_required(&payload, "selected_pose_method")?,
        guard_level_passed: json_string_required(&payload, "guard_level_passed")?,
        o3a_score: json_f64_required(&payload, "o3a_score")?,
    })
}

fn validate_selected_ligand_pose(
    selected_pose: &SelectedLigandPose,
    config: &Config,
) -> Result<()> {
    if selected_pose.selected_pose_method != "compact_pseudo_ligand_o3a" {
        return Err(anyhow!(
            "selected_pose_method must be compact_pseudo_ligand_o3a, got {}",
            selected_pose.selected_pose_method
        ));
    }
    if !config.allow_ligand_override {
        let configured = std::fs::canonicalize(&config.scaffold_sdf)
            .with_context(|| format!("canonicalize ligand {}", config.scaffold_sdf.display()))?;
        let selected = std::fs::canonicalize(Path::new(&selected_pose.selected_ligand_sdf))
            .with_context(|| {
                format!(
                    "canonicalize selected ligand {}",
                    selected_pose.selected_ligand_sdf
                )
            })?;
        if configured != selected {
            return Err(anyhow!(
                "ligand_sdf_path {} does not match selected_ligand_pose.selected_ligand_sdf {}",
                config.scaffold_sdf.display(),
                selected_pose.selected_ligand_sdf
            ));
        }
    }
    if !matches!(
        config.assembly_mode,
        AssemblyMode::ZMatrix | AssemblyMode::SmartsZMatrix
    ) {
        return Err(anyhow!("assembly_mode must be zmatrix or smarts_zmatrix"));
    }
    Ok(())
}

fn smarts_primary_rule(
    registry: &ReactionRegistry,
    library: &SynthonLibrary,
) -> Result<AssemblyReactionRule> {
    let reaction = registry
        .enabled_reactions()
        .filter(|reaction| {
            !library
                .compatible_ids(&reaction.reaction_id, "synthon")
                .is_empty()
        })
        .min_by_key(|reaction| {
            if reaction.reaction_id == "RXN_AMIDE_COUPLING" {
                0
            } else {
                1
            }
        })
        .ok_or_else(|| anyhow!("no compatible reaction pairs for SMARTS registry synthons"))?;
    Ok(AssemblyReactionRule {
        rule_id: reaction.reaction_id.clone(),
        synthon_a_role: "synthon".to_owned(),
        synthon_b_role: "scaffold".to_owned(),
        bond_length_a: reaction.product_bond.ideal_bond_length_a,
        dihedral_omega_rad: DihedralConstraint::Fixed(core::f32::consts::PI),
    })
}

fn prune_rule(
    library: &SynthonLibrary,
    scaffold: &ScaffoldTemplate,
    rule: &AssemblyReactionRule,
    registry: Option<&ReactionRegistry>,
    field: &HashMap<u64, VoxelField>,
    no_fly: &HashSet<u64>,
    diagnostics: &FieldDiagnostics,
    config: &Config,
    geometry: GridGeometry,
) -> Result<(Vec<Survivor>, PruneTelemetry)> {
    let synthon_a_ids = library.compatible_ids(&rule.rule_id, &rule.synthon_a_role);
    if synthon_a_ids.is_empty() || scaffold.anchors.is_empty() {
        return Ok((Vec::new(), PruneCounters::default().snapshot()));
    }
    let total_pairs = synthon_a_ids
        .len()
        .checked_mul(scaffold.anchors.len())
        .ok_or_else(|| anyhow!("synthon pair space overflowed usize"))?;
    let pair_limit = pair_limit(total_pairs, config);
    let counters = PruneCounters::default();
    let mut survivors: Vec<Survivor> = (0..pair_limit)
        .into_par_iter()
        .filter_map(|pair_idx| {
            if !pair_in_shard(pair_idx, config) {
                return None;
            }
            counters.attempted.fetch_add(1, Ordering::Relaxed);
            let a_idx = pair_idx / scaffold.anchors.len();
            let anchor_idx = pair_idx % scaffold.anchors.len();
            let a_id = &synthon_a_ids[a_idx];
            let synthon = library.get(a_id)?;
            let (anchored, score) = match best_rotamer_for_candidate(
                scaffold,
                synthon,
                anchor_idx,
                rule,
                registry,
                field,
                no_fly,
                diagnostics,
                config,
                geometry,
                &counters,
            ) {
                Some(value) => value,
                None => return None,
            };
            counters
                .candidates_with_at_least_one_valid_rotamer
                .fetch_add(1, Ordering::Relaxed);
            counters
                .cryptic_bonus_atoms
                .fetch_add(score.cryptic_bonus_atoms as usize, Ordering::Relaxed);
            counters.inc_survival(score.survival_tier);
            counters.record_selected_dihedral(anchored.selected_dihedral_deg);
            Some(make_survivor(anchored, synthon, score, config))
        })
        .collect();
    survivors.sort_by(|lhs, rhs| {
        rhs.score
            .total_cmp(&lhs.score)
            .then_with(|| lhs.product_id.cmp(&rhs.product_id))
    });
    if survivors.len() > config.survivor_limit {
        survivors.truncate(config.survivor_limit);
    }
    Ok((survivors, counters.snapshot()))
}

fn prune_rule_streaming(
    library: &SynthonLibrary,
    scaffold: &ScaffoldTemplate,
    rule: &AssemblyReactionRule,
    registry: Option<&ReactionRegistry>,
    field: &HashMap<u64, VoxelField>,
    no_fly: &HashSet<u64>,
    diagnostics: &FieldDiagnostics,
    config: &Config,
    geometry: GridGeometry,
) -> Result<(usize, PruneTelemetry)> {
    let synthon_a_ids = library.compatible_ids(&rule.rule_id, &rule.synthon_a_role);
    if synthon_a_ids.is_empty() || scaffold.anchors.is_empty() {
        write_survivors_streaming(&config.output, mpsc::channel().1, config.row_group_size)?;
        return Ok((0, PruneCounters::default().snapshot()));
    }
    let total_pairs = synthon_a_ids
        .len()
        .checked_mul(scaffold.anchors.len())
        .ok_or_else(|| anyhow!("synthon pair space overflowed usize"))?;
    let pair_limit = pair_limit(total_pairs, config);
    println!(
        "mpsc_stream_start output={} row_group_size={} total_pairs={} pair_limit={} shard_index={} shard_count={}",
        config.output.display(),
        config.row_group_size,
        total_pairs,
        pair_limit,
        config.shard_index,
        config.shard_count
    );
    let counters = PruneCounters::default();
    let (tx, rx) = mpsc::channel::<Survivor>();
    let output = config.output.clone();
    let row_group_size = config.row_group_size;
    let writer = thread::spawn(move || write_survivors_streaming(&output, rx, row_group_size));

    (0..pair_limit)
        .into_par_iter()
        .try_for_each_with(tx, |sender, pair_idx| -> Result<()> {
            if !pair_in_shard(pair_idx, config) {
                return Ok(());
            }
            counters.attempted.fetch_add(1, Ordering::Relaxed);
            let a_idx = pair_idx / scaffold.anchors.len();
            let anchor_idx = pair_idx % scaffold.anchors.len();
            let a_id = &synthon_a_ids[a_idx];
            let Some(synthon) = library.get(a_id) else {
                return Ok(());
            };
            let Some((anchored, score)) = best_rotamer_for_candidate(
                scaffold,
                synthon,
                anchor_idx,
                rule,
                registry,
                field,
                no_fly,
                diagnostics,
                config,
                geometry,
                &counters,
            ) else {
                return Ok(());
            };
            counters
                .candidates_with_at_least_one_valid_rotamer
                .fetch_add(1, Ordering::Relaxed);
            counters
                .cryptic_bonus_atoms
                .fetch_add(score.cryptic_bonus_atoms as usize, Ordering::Relaxed);
            counters.inc_survival(score.survival_tier);
            counters.record_selected_dihedral(anchored.selected_dihedral_deg);
            sender
                .send(make_survivor(anchored, synthon, score, config))
                .map_err(|err| anyhow!("MPSC survivor send failed: {err}"))?;
            Ok(())
        })?;

    let survivor_count = writer
        .join()
        .map_err(|_| anyhow!("MPSC survivor writer thread panicked"))??;
    println!(
        "mpsc_stream_complete output={} survivors={} row_group_size={}",
        config.output.display(),
        survivor_count,
        config.row_group_size
    );
    Ok((survivor_count, counters.snapshot()))
}

#[allow(clippy::too_many_arguments)]
fn prune_dendritic_shard_streaming(
    library: &SynthonLibrary,
    scaffold: &ScaffoldTemplate,
    registry: &ReactionRegistry,
    field: &HashMap<u64, VoxelField>,
    no_fly: &HashSet<u64>,
    diagnostics: &FieldDiagnostics,
    config: &Config,
    geometry: GridGeometry,
    shard: &DendriticShard,
) -> Result<(usize, PruneTelemetry, DendriticTelemetry)> {
    let first_rule = registry.get(&shard.first_reaction_id)?;
    let second_rule = registry.get(&shard.second_reaction_id)?;
    let first_assembly_rule = AssemblyReactionRule {
        rule_id: shard.first_reaction_id.clone(),
        synthon_a_role: "synthon".to_owned(),
        synthon_b_role: "scaffold".to_owned(),
        bond_length_a: first_rule.product_bond.ideal_bond_length_a,
        dihedral_omega_rad: DihedralConstraint::Fixed(core::f32::consts::PI),
    };
    let bridge_ids = library
        .compatible_ids(&shard.first_reaction_id, "synthon")
        .iter()
        .filter(|synthon_id| {
            library
                .get(synthon_id)
                .is_some_and(|synthon| synthon.has_role(&shard.second_reaction_id, "scaffold"))
        })
        .cloned()
        .collect::<Vec<_>>();
    let terminal_ids = library
        .compatible_ids(&shard.second_reaction_id, "synthon")
        .to_vec();
    if bridge_ids.is_empty() {
        return Err(anyhow!(
            "no bridge synthons for {} -> {}",
            shard.first_reaction_id,
            shard.second_reaction_id
        ));
    }
    if terminal_ids.is_empty() {
        return Err(anyhow!(
            "no terminal synthons for {}",
            shard.second_reaction_id
        ));
    }
    if scaffold.anchors.is_empty() {
        return Err(anyhow!("no scaffold anchors available for dendritic shard"));
    }
    let terminal_count = terminal_ids.len();
    let a_assignment_count = bridge_ids
        .len()
        .checked_mul(scaffold.anchors.len())
        .ok_or_else(|| anyhow!("A-assignment count overflowed"))?;
    let total_pairs = a_assignment_count
        .checked_mul(terminal_count)
        .ok_or_else(|| anyhow!("dendritic pair count overflowed"))?;
    if shard.pathway_end_pair_idx_exclusive > total_pairs {
        return Err(anyhow!(
            "shard end {} exceeds computed pathway total {} for {}",
            shard.pathway_end_pair_idx_exclusive,
            total_pairs,
            shard.pathway_id
        ));
    }
    let start_a_assignment = shard.pathway_start_pair_idx / terminal_count;
    let end_a_assignment = shard.pathway_end_pair_idx_exclusive.saturating_sub(1) / terminal_count;
    println!(
        "dendritic_stream_start requested_shard={} plan_shard={} pathway_id={} output={} row_group_size={} first_reaction={} second_reaction={} bridge_synthons={} terminal_synthons={} scaffold_anchors={} shard_pair_start={} shard_pair_end={} shard_pairs={} a_assignment_start={} a_assignment_end={} total_pairs={}",
        shard.requested_shard_id,
        shard.plan_shard_id,
        shard.pathway_id,
        shard.output.display(),
        config.row_group_size,
        shard.first_reaction_id,
        shard.second_reaction_id,
        bridge_ids.len(),
        terminal_ids.len(),
        scaffold.anchors.len(),
        shard.pathway_start_pair_idx,
        shard.pathway_end_pair_idx_exclusive,
        shard.pair_count,
        start_a_assignment,
        end_a_assignment,
        total_pairs
    );
    let counters = PruneCounters::default();
    let attempted_a = Arc::new(AtomicUsize::new(0));
    let intermediate_a_dropped = Arc::new(AtomicUsize::new(0));
    let attempted_b = Arc::new(AtomicUsize::new(0));
    let (tx, rx) = mpsc::channel::<Survivor>();
    let output = shard.output.clone();
    let row_group_size = config.row_group_size;
    let writer = thread::spawn(move || write_survivors_streaming(&output, rx, row_group_size));

    (start_a_assignment..=end_a_assignment)
        .into_par_iter()
        .try_for_each_with(tx, |sender, a_assignment_idx| -> Result<()> {
            attempted_a.fetch_add(1, Ordering::Relaxed);
            counters.attempted.fetch_add(1, Ordering::Relaxed);
            let bridge_idx = a_assignment_idx / scaffold.anchors.len();
            let anchor_idx = a_assignment_idx % scaffold.anchors.len();
            let Some(bridge_id) = bridge_ids.get(bridge_idx) else {
                return Ok(());
            };
            let Some(bridge_synthon) = library.get(bridge_id) else {
                return Ok(());
            };
            let Some((intermediate, intermediate_score)) = best_rotamer_for_candidate(
                scaffold,
                bridge_synthon,
                anchor_idx,
                &first_assembly_rule,
                Some(registry),
                field,
                no_fly,
                diagnostics,
                config,
                geometry,
                &counters,
            ) else {
                intermediate_a_dropped.fetch_add(1, Ordering::Relaxed);
                return Ok(());
            };
            counters.cryptic_bonus_atoms.fetch_add(
                intermediate_score.cryptic_bonus_atoms as usize,
                Ordering::Relaxed,
            );

            let block_start = a_assignment_idx
                .checked_mul(terminal_count)
                .ok_or_else(|| anyhow!("A-assignment block start overflowed"))?;
            let block_end = block_start
                .checked_add(terminal_count)
                .ok_or_else(|| anyhow!("A-assignment block end overflowed"))?;
            let b_start = shard.pathway_start_pair_idx.max(block_start) - block_start;
            let b_end = shard
                .pathway_end_pair_idx_exclusive
                .min(block_end)
                .saturating_sub(block_start);
            for b_idx in b_start..b_end {
                attempted_b.fetch_add(1, Ordering::Relaxed);
                let Some(terminal_id) = terminal_ids.get(b_idx) else {
                    continue;
                };
                let Some(terminal_synthon) = library.get(terminal_id) else {
                    continue;
                };
                let Some((anchored, score)) = best_second_stage_rotamer_for_candidate(
                    &intermediate,
                    bridge_synthon,
                    terminal_synthon,
                    second_rule,
                    field,
                    no_fly,
                    diagnostics,
                    config,
                    geometry,
                    &counters,
                ) else {
                    continue;
                };
                counters
                    .candidates_with_at_least_one_valid_rotamer
                    .fetch_add(1, Ordering::Relaxed);
                counters
                    .cryptic_bonus_atoms
                    .fetch_add(score.cryptic_bonus_atoms as usize, Ordering::Relaxed);
                counters.inc_survival(score.survival_tier);
                counters.record_selected_dihedral(anchored.selected_dihedral_deg);
                sender
                    .send(make_dendritic_survivor(
                        anchored,
                        bridge_synthon,
                        terminal_synthon,
                        score,
                        config,
                    ))
                    .map_err(|err| anyhow!("MPSC dendritic survivor send failed: {err}"))?;
            }
            Ok(())
        })?;

    let survivor_count = writer
        .join()
        .map_err(|_| anyhow!("MPSC dendritic survivor writer thread panicked"))??;
    println!(
        "dendritic_stream_complete requested_shard={} plan_shard={} output={} survivors={} row_group_size={}",
        shard.requested_shard_id,
        shard.plan_shard_id,
        shard.output.display(),
        survivor_count,
        config.row_group_size
    );
    let dendritic = DendriticTelemetry {
        attempted_a: attempted_a.load(Ordering::Relaxed),
        intermediate_a_dropped: intermediate_a_dropped.load(Ordering::Relaxed),
        attempted_b: attempted_b.load(Ordering::Relaxed),
        final_survivors: survivor_count,
    };
    Ok((survivor_count, counters.snapshot(), dendritic))
}

fn pair_limit(total_pairs: usize, config: &Config) -> usize {
    if config.full_scale && !config.dry_run_plan {
        total_pairs
    } else {
        total_pairs.min(config.max_pairs)
    }
}

fn pair_in_shard(pair_idx: usize, config: &Config) -> bool {
    config.shard_count <= 1 || pair_idx % config.shard_count == config.shard_index
}

fn make_survivor(
    anchored: AnchoredProduct,
    synthon: &LoadedSynthon,
    score: ProductScore,
    config: &Config,
) -> Survivor {
    Survivor {
        anchor_id: anchored.synthon_a_id.clone(),
        canonical_smiles: synthon.canonical_smiles.clone(),
        product_id: anchored.product.product_id,
        smiles: anchored.product.smiles,
        synthon_a_id: anchored.synthon_a_id,
        synthon_b_id: anchored.synthon_b_id,
        score: score.score,
        fragment_pi_complement: score.pi_complement,
        fragment_pi_clash_adjusted: score.pi_clash,
        pi_complement: score.pi_complement,
        pi_clash: score.pi_clash,
        cryptic_bonus: score.cryptic_bonus,
        cryptic_bonus_atoms: score.cryptic_bonus_atoms,
        survival_tier: score.survival_tier.label(),
        selected_dihedral_deg: anchored.selected_dihedral_deg,
        assembly_mode: anchored.assembly_mode,
        z_matrix_active: anchored.z_matrix_active,
        rotamers_evaluated: anchored.rotamers_evaluated,
        best_rotamer_rank: anchored.best_rotamer_rank,
        ligand_sdf_path: config.scaffold_sdf.display().to_string(),
        pose_reconciliation_method: infer_pose_reconciliation_method(&config.scaffold_sdf),
        coordinates_json: coordinates_json(&anchored.product.coordinates),
    }
}

fn make_dendritic_survivor(
    anchored: AnchoredProduct,
    _bridge_synthon: &LoadedSynthon,
    _terminal_synthon: &LoadedSynthon,
    score: ProductScore,
    config: &Config,
) -> Survivor {
    Survivor {
        anchor_id: anchored.synthon_a_id.clone(),
        canonical_smiles: anchored.product.smiles.clone(),
        product_id: anchored.product.product_id,
        smiles: anchored.product.smiles,
        synthon_a_id: anchored.synthon_a_id,
        synthon_b_id: anchored.synthon_b_id,
        score: score.score,
        fragment_pi_complement: score.pi_complement,
        fragment_pi_clash_adjusted: score.pi_clash,
        pi_complement: score.pi_complement,
        pi_clash: score.pi_clash,
        cryptic_bonus: score.cryptic_bonus,
        cryptic_bonus_atoms: score.cryptic_bonus_atoms,
        survival_tier: score.survival_tier.label(),
        selected_dihedral_deg: anchored.selected_dihedral_deg,
        assembly_mode: anchored.assembly_mode,
        z_matrix_active: anchored.z_matrix_active,
        rotamers_evaluated: anchored.rotamers_evaluated,
        best_rotamer_rank: anchored.best_rotamer_rank,
        ligand_sdf_path: config.scaffold_sdf.display().to_string(),
        pose_reconciliation_method: infer_pose_reconciliation_method(&config.scaffold_sdf),
        coordinates_json: coordinates_json(&anchored.product.coordinates),
    }
}

#[allow(clippy::too_many_arguments)]
fn best_rotamer_for_candidate(
    scaffold: &ScaffoldTemplate,
    synthon: &LoadedSynthon,
    anchor_idx: usize,
    rule: &AssemblyReactionRule,
    registry: Option<&ReactionRegistry>,
    field: &HashMap<u64, VoxelField>,
    no_fly: &HashSet<u64>,
    diagnostics: &FieldDiagnostics,
    config: &Config,
    geometry: GridGeometry,
    counters: &PruneCounters,
) -> Option<(AnchoredProduct, ProductScore)> {
    let dihedrals = active_dihedral_grid(config);
    let mut best: Option<(AnchoredProduct, ProductScore)> = None;
    for (rank, dihedral_deg) in dihedrals.iter().copied().enumerate() {
        counters.rotamers_evaluated.fetch_add(1, Ordering::Relaxed);
        let mut anchored = match assemble_scaffold_synthon(
            scaffold,
            synthon,
            anchor_idx,
            rule,
            registry,
            config,
            f64::from(dihedral_deg),
            u64::try_from(dihedrals.len()).unwrap_or(u64::MAX),
            u64::try_from(rank + 1).unwrap_or(u64::MAX),
        ) {
            Ok(value) => value,
            Err(_) => {
                counters.dropped_assembly.fetch_add(1, Ordering::Relaxed);
                continue;
            }
        };
        if anchored.z_matrix_active {
            counters
                .z_matrix_active_count
                .fetch_add(1, Ordering::Relaxed);
        }
        let score = match score_product(
            &anchored.product,
            anchored.score_atom_offset,
            &anchored.score_atom_atomic_numbers,
            field,
            no_fly,
            diagnostics,
            geometry,
        ) {
            ScoreDecision::Keep(value) => {
                counters.record_placement(value.placement_metrics);
                value
            }
            ScoreDecision::Drop(reason, metrics) => {
                if let Some(metrics) = metrics {
                    counters.record_placement(metrics);
                }
                counters.inc_drop(reason);
                counters.inc_rotamer_drop(reason);
                continue;
            }
        };
        anchored.best_rotamer_rank = u64::try_from(rank + 1).unwrap_or(u64::MAX);
        if best
            .as_ref()
            .is_none_or(|(_best_product, best_score)| score.score > best_score.score)
        {
            best = Some((anchored, score));
        }
    }
    best
}

fn assemble_scaffold_synthon(
    scaffold: &ScaffoldTemplate,
    synthon: &LoadedSynthon,
    anchor_idx: usize,
    rule: &AssemblyReactionRule,
    registry: Option<&ReactionRegistry>,
    config: &Config,
    dihedral_deg: f64,
    rotamers_evaluated: u64,
    best_rotamer_rank: u64,
) -> Result<AnchoredProduct> {
    if !synthon.has_role(&rule.rule_id, &rule.synthon_a_role) {
        return Err(anyhow!(
            "synthon {} is not compatible with {}:{}",
            synthon.synthon_id,
            rule.rule_id,
            rule.synthon_a_role
        ));
    }
    let anchor = scaffold
        .anchors
        .get(anchor_idx)
        .ok_or_else(|| anyhow!("scaffold anchor index {anchor_idx} out of range"))?;
    let mut score_atom_atomic_numbers = retained_synthon_atomic_numbers(synthon);
    let (product_state, z_matrix_active, synthon_index_map) = match config.assembly_mode {
        AssemblyMode::Rigid => (
            execute_3d_reaction(
                &scaffold.state,
                &synthon.synthon,
                anchor_idx,
                0,
                ReactionKinematicRule {
                    bond_length_a: config.scaffold_bond_length_a,
                    dihedral_omega_rad: DihedralConstraint::Fixed(dihedral_deg.to_radians() as f32),
                },
            )?,
            false,
            Vec::new(),
        ),
        AssemblyMode::ZMatrix => (
            execute_zmatrix_reaction(
                &scaffold.state,
                &synthon.synthon,
                anchor_idx,
                0,
                &ZMatrixAssemblyRule {
                    scaffold_reference_atom_1: anchor.scaffold_ref1_atom_idx,
                    scaffold_reference_atom_2: anchor.scaffold_ref2_atom_idx,
                    bond_length_a: config.scaffold_bond_length_a,
                    bond_angle_deg: 109.5,
                    dihedral_deg: dihedral_deg as f32,
                    hybridization_model: config.assembly_mode.label().to_owned(),
                },
            )?,
            true,
            Vec::new(),
        ),
        AssemblyMode::SmartsZMatrix => {
            let reaction_rule = registry
                .ok_or_else(|| anyhow!("reaction registry is required for smarts_zmatrix"))?
                .get(&rule.rule_id)?;
            let scaffold_attachment = scaffold
                .state
                .attachment_points
                .get(anchor_idx)
                .ok_or_else(|| anyhow!("scaffold attachment index {anchor_idx} out of range"))?;
            let synthon_site = synthon
                .reaction_site(&rule.rule_id, &rule.synthon_a_role)
                .ok_or_else(|| {
                    anyhow!(
                        "synthon {} lacks SMARTS site for {}:{}",
                        synthon.synthon_id,
                        rule.rule_id,
                        rule.synthon_a_role
                    )
                })?;
            let synthon_leaving_group_atom_indices =
                synthon.leaving_groups_for(&rule.rule_id, &rule.synthon_a_role);
            let synthon_reference_atom_idx = synthon_site
                .reference_atom_idx
                .filter(|atom_idx| *atom_idx != synthon_site.reactive_atom_idx)
                .unwrap_or_else(|| {
                    fallback_synthon_reference_atom(synthon, synthon_site.reactive_atom_idx)
                });
            let plan = AssemblyPlan {
                reaction_id: rule.rule_id.clone(),
                scaffold_role: "scaffold".to_owned(),
                synthon_role: rule.synthon_a_role.clone(),
                scaffold_reactive_atom_idx: scaffold_attachment.atom_index,
                synthon_reactive_atom_idx: synthon_site.reactive_atom_idx,
                scaffold_reference_atom_1: anchor.scaffold_ref1_atom_idx,
                scaffold_reference_atom_2: anchor.scaffold_ref2_atom_idx,
                synthon_reference_atom_idx,
                scaffold_leaving_group_atom_indices: Vec::new(),
                synthon_leaving_group_atom_indices: synthon_leaving_group_atom_indices.clone(),
                selected_dihedral_deg: dihedral_deg as f32,
            };
            score_atom_atomic_numbers = retained_synthon_atomic_numbers_for_site(
                synthon,
                synthon_site.reactive_atom_idx,
                &synthon_leaving_group_atom_indices,
            );
            let product = execute_smarts_zmatrix_reaction(
                &scaffold.state,
                &synthon.synthon,
                &plan,
                reaction_rule,
            )?;
            (
                product.state,
                product.metadata.z_matrix_active,
                product.synthon_index_map,
            )
        }
    };
    let product_id = format!(
        "SCAFFOLD_ANCHORED__{}__EXIT_{}__CUT_{}__PHI_{:.0}",
        synthon.synthon_id,
        anchor.original_scaffold_atom_idx,
        anchor.removed_fragment_atom_idx,
        dihedral_deg
    );
    Ok(AnchoredProduct {
        product: Product3D {
            product_id,
            smiles: format!(
                "PROJECTED_RULE_PRODUCT::{}::{}",
                rule.rule_id, synthon.synthon_id
            ),
            coordinates: product_state.coordinates,
            charges: product_state.charges,
            bonds: product_state.bonds,
        },
        synthon_a_id: synthon.synthon_id.clone(),
        synthon_b_id: format!(
            "ALENI_CORE_EXIT_{}_CUT_{}",
            anchor.original_scaffold_atom_idx, anchor.removed_fragment_atom_idx
        ),
        score_atom_offset: scaffold.state.atom_count(),
        score_atom_atomic_numbers,
        synthon_index_map,
        selected_dihedral_deg: dihedral_deg,
        assembly_mode: config.assembly_mode.label(),
        z_matrix_active,
        rotamers_evaluated,
        best_rotamer_rank,
    })
}

#[allow(clippy::too_many_arguments)]
fn best_second_stage_rotamer_for_candidate(
    intermediate: &AnchoredProduct,
    bridge_synthon: &LoadedSynthon,
    terminal_synthon: &LoadedSynthon,
    second_rule: &prism_forge::reactions::reaction_registry::ReactionRule,
    field: &HashMap<u64, VoxelField>,
    no_fly: &HashSet<u64>,
    diagnostics: &FieldDiagnostics,
    config: &Config,
    geometry: GridGeometry,
    counters: &PruneCounters,
) -> Option<(AnchoredProduct, ProductScore)> {
    let dihedrals = active_dihedral_grid(config);
    let mut best: Option<(AnchoredProduct, ProductScore)> = None;
    for (rank, dihedral_deg) in dihedrals.iter().copied().enumerate() {
        counters.rotamers_evaluated.fetch_add(1, Ordering::Relaxed);
        let mut anchored = match assemble_intermediate_synthon(
            intermediate,
            bridge_synthon,
            terminal_synthon,
            second_rule,
            f64::from(dihedral_deg),
            u64::try_from(dihedrals.len()).unwrap_or(u64::MAX),
            u64::try_from(rank + 1).unwrap_or(u64::MAX),
        ) {
            Ok(value) => value,
            Err(_) => {
                counters.dropped_assembly.fetch_add(1, Ordering::Relaxed);
                continue;
            }
        };
        counters
            .z_matrix_active_count
            .fetch_add(1, Ordering::Relaxed);
        let score = match score_product(
            &anchored.product,
            anchored.score_atom_offset,
            &anchored.score_atom_atomic_numbers,
            field,
            no_fly,
            diagnostics,
            geometry,
        ) {
            ScoreDecision::Keep(value) => {
                counters.record_placement(value.placement_metrics);
                value
            }
            ScoreDecision::Drop(reason, metrics) => {
                if let Some(metrics) = metrics {
                    counters.record_placement(metrics);
                }
                counters.inc_drop(reason);
                counters.inc_rotamer_drop(reason);
                continue;
            }
        };
        anchored.best_rotamer_rank = u64::try_from(rank + 1).unwrap_or(u64::MAX);
        if best
            .as_ref()
            .is_none_or(|(_best_product, best_score)| score.score > best_score.score)
        {
            best = Some((anchored, score));
        }
    }
    best
}

fn assemble_intermediate_synthon(
    intermediate: &AnchoredProduct,
    bridge_synthon: &LoadedSynthon,
    terminal_synthon: &LoadedSynthon,
    second_rule: &prism_forge::reactions::reaction_registry::ReactionRule,
    dihedral_deg: f64,
    rotamers_evaluated: u64,
    best_rotamer_rank: u64,
) -> Result<AnchoredProduct> {
    let bridge_site = bridge_synthon
        .reaction_site(&second_rule.reaction_id, "scaffold")
        .ok_or_else(|| {
            anyhow!(
                "bridge synthon {} lacks {}:scaffold site",
                bridge_synthon.synthon_id,
                second_rule.reaction_id
            )
        })?;
    let terminal_site = terminal_synthon
        .reaction_site(&second_rule.reaction_id, "synthon")
        .ok_or_else(|| {
            anyhow!(
                "terminal synthon {} lacks {}:synthon site",
                terminal_synthon.synthon_id,
                second_rule.reaction_id
            )
        })?;
    let scaffold_state = ScaffoldState3D::new_with_bonds(
        intermediate.product.coordinates.clone(),
        intermediate.product.charges.clone(),
        Vec::new(),
        intermediate.product.bonds.clone(),
    )?;
    let scaffold_reactive_atom_idx = map_intermediate_synthon_atom(
        &intermediate.synthon_index_map,
        bridge_site.reactive_atom_idx,
        "bridge reactive atom",
    )?;
    let scaffold_reference_atom_1 = bridge_site
        .reference_atom_idx
        .and_then(|atom_idx| {
            intermediate
                .synthon_index_map
                .get(atom_idx)
                .copied()
                .flatten()
        })
        .filter(|atom_idx| *atom_idx != scaffold_reactive_atom_idx)
        .unwrap_or_else(|| {
            fallback_product_reference_atom(&scaffold_state, &[scaffold_reactive_atom_idx])
        });
    let scaffold_reference_atom_2 = fallback_product_reference_atom(
        &scaffold_state,
        &[scaffold_reactive_atom_idx, scaffold_reference_atom_1],
    );
    let scaffold_leaving_group_atom_indices = bridge_synthon
        .leaving_groups_for(&second_rule.reaction_id, "scaffold")
        .into_iter()
        .filter_map(|atom_idx| {
            intermediate
                .synthon_index_map
                .get(atom_idx)
                .copied()
                .flatten()
        })
        .collect::<Vec<_>>();
    let synthon_reference_atom_idx = terminal_site
        .reference_atom_idx
        .filter(|atom_idx| *atom_idx != terminal_site.reactive_atom_idx)
        .unwrap_or_else(|| {
            fallback_synthon_reference_atom(terminal_synthon, terminal_site.reactive_atom_idx)
        });
    let synthon_leaving_group_atom_indices =
        terminal_synthon.leaving_groups_for(&second_rule.reaction_id, "synthon");
    let plan = AssemblyPlan {
        reaction_id: second_rule.reaction_id.clone(),
        scaffold_role: "scaffold".to_owned(),
        synthon_role: "synthon".to_owned(),
        scaffold_reactive_atom_idx,
        synthon_reactive_atom_idx: terminal_site.reactive_atom_idx,
        scaffold_reference_atom_1,
        scaffold_reference_atom_2,
        synthon_reference_atom_idx,
        scaffold_leaving_group_atom_indices,
        synthon_leaving_group_atom_indices: synthon_leaving_group_atom_indices.clone(),
        selected_dihedral_deg: dihedral_deg as f32,
    };
    let product = execute_smarts_zmatrix_reaction(
        &scaffold_state,
        &terminal_synthon.synthon,
        &plan,
        second_rule,
    )?;
    let score_atom_offset = scaffold_state.atom_count();
    let product_id = format!(
        "DENDRITIC__{}__{}__{}__PHI_{:.0}",
        second_rule.reaction_id,
        bridge_synthon.synthon_id,
        terminal_synthon.synthon_id,
        dihedral_deg
    );
    Ok(AnchoredProduct {
        product: Product3D {
            product_id,
            smiles: format!(
                "PROJECTED_DENDRITIC_RULE_PRODUCT::{}::{}::{}",
                second_rule.reaction_id, bridge_synthon.synthon_id, terminal_synthon.synthon_id
            ),
            coordinates: product.state.coordinates,
            charges: product.state.charges,
            bonds: product.state.bonds,
        },
        synthon_a_id: bridge_synthon.synthon_id.clone(),
        synthon_b_id: terminal_synthon.synthon_id.clone(),
        score_atom_offset,
        score_atom_atomic_numbers: retained_synthon_atomic_numbers_for_site(
            terminal_synthon,
            terminal_site.reactive_atom_idx,
            &synthon_leaving_group_atom_indices,
        ),
        synthon_index_map: product.synthon_index_map,
        selected_dihedral_deg: dihedral_deg,
        assembly_mode: "smarts_zmatrix",
        z_matrix_active: true,
        rotamers_evaluated,
        best_rotamer_rank,
    })
}

fn active_dihedral_grid(config: &Config) -> Vec<f32> {
    let mut grid = if config.dihedral_grid_deg.is_empty() {
        vec![0.0]
    } else {
        config.dihedral_grid_deg.clone()
    };
    if config.dihedral_samples > 0 && grid.len() > config.dihedral_samples {
        grid.truncate(config.dihedral_samples);
    }
    grid
}

fn infer_pose_reconciliation_method(path: &Path) -> String {
    if let Some(parent) = path.parent() {
        let manifest_path = parent.join("ALENI-PARENT_6XOX_pose_reconciliation_manifest.json");
        if let Ok(text) = std::fs::read_to_string(&manifest_path) {
            if let Ok(payload) = serde_json::from_str::<Value>(&text) {
                if let Some(method) = payload
                    .get("selected_alignment_method")
                    .and_then(Value::as_str)
                {
                    return method.to_owned();
                }
            }
        }
    }
    let filename = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or_default();
    if filename.contains("o3a_relaxed") || filename.contains("o3a") {
        "o3a_or_documented_fallback".to_owned()
    } else if filename.contains("kabsch") || filename.contains("minimized") {
        "kabsch_fallback".to_owned()
    } else {
        "unknown".to_owned()
    }
}

#[derive(Debug, Clone, Copy)]
struct ProductScore {
    score: f64,
    pi_complement: f64,
    pi_clash: f64,
    cryptic_bonus: f64,
    cryptic_bonus_atoms: u64,
    survival_tier: SurvivalTier,
    placement_metrics: PlacementMetrics,
}

#[derive(Debug, Clone, Copy)]
enum DropReason {
    Bounds,
    NoFly,
    HardClash,
    InsufficientComplement,
}

#[derive(Debug, Clone, Copy)]
enum SurvivalTier {
    Normal,
    CrypticRescue,
}

impl SurvivalTier {
    fn label(self) -> &'static str {
        match self {
            SurvivalTier::Normal => "normal",
            SurvivalTier::CrypticRescue => "cryptic_rescue",
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum ScoreDecision {
    Keep(ProductScore),
    Drop(DropReason, Option<PlacementMetrics>),
}

#[derive(Debug, Default)]
struct PruneCounters {
    attempted: AtomicUsize,
    dropped_assembly: AtomicUsize,
    dropped_bounds: AtomicUsize,
    dropped_no_fly: AtomicUsize,
    dropped_hard_clash: AtomicUsize,
    dropped_insufficient_complement: AtomicUsize,
    rotamers_evaluated: AtomicUsize,
    rotamers_dropped_bounds: AtomicUsize,
    rotamers_dropped_no_fly: AtomicUsize,
    rotamers_dropped_hard_clash: AtomicUsize,
    candidates_with_at_least_one_valid_rotamer: AtomicUsize,
    z_matrix_active_count: AtomicUsize,
    rigid_fallback_count: AtomicUsize,
    survived_normal: AtomicUsize,
    survived_cryptic_rescue: AtomicUsize,
    cryptic_bonus_atoms: AtomicUsize,
    candidates_with_atoms_within_1_voxel_complement: AtomicUsize,
    candidates_with_atoms_within_2_voxels_complement: AtomicUsize,
    placement_min_complement_distances: Mutex<Vec<f64>>,
    selected_dihedral_degs: Mutex<Vec<f64>>,
    coherence_metrics: Mutex<CoherenceMetricsAccumulator>,
    fragment_score_metrics: Mutex<FragmentScoreMetricsAccumulator>,
}

#[derive(Debug, Clone)]
struct PruneTelemetry {
    attempted: usize,
    dropped_assembly: usize,
    dropped_bounds: usize,
    dropped_no_fly: usize,
    dropped_hard_clash: usize,
    dropped_insufficient_complement: usize,
    rotamers_evaluated: usize,
    rotamers_dropped_bounds: usize,
    rotamers_dropped_no_fly: usize,
    rotamers_dropped_hard_clash: usize,
    candidates_with_at_least_one_valid_rotamer: usize,
    mean_best_dihedral_deg: f64,
    top_dihedral_bins: String,
    z_matrix_active_count: usize,
    rigid_fallback_count: usize,
    survived_normal: usize,
    survived_cryptic_rescue: usize,
    cryptic_bonus_atoms: usize,
    mean_min_atom_to_complement_a: f64,
    p10_min_atom_to_complement_a: f64,
    min_seen_atom_to_complement_a: f64,
    candidates_with_atoms_within_1_voxel_complement: usize,
    candidates_with_atoms_within_2_voxels_complement: usize,
    coherence_missing_residue_hits: usize,
    mean_coherence_factor_for_clashes: f64,
    min_coherence_factor_for_clashes: f64,
    max_coherence_factor_for_clashes: f64,
    raw_pi_clash_sum: f64,
    adjusted_pi_clash_sum: f64,
    mean_scaffold_context_pi_complement: f64,
    mean_fragment_pi_complement: f64,
    mean_fragment_pi_clash_adjusted: f64,
    mean_fragment_cryptic_bonus: f64,
    candidates_with_fragment_complement: usize,
    candidates_with_fragment_cryptic_bonus: usize,
}

#[derive(Debug, Clone, Copy)]
struct CoherenceMetricsAccumulator {
    missing_residue_hits: usize,
    factor_sum: f64,
    factor_count: usize,
    min_factor: f64,
    max_factor: f64,
    raw_pi_clash_sum: f64,
    adjusted_pi_clash_sum: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct FragmentScoreMetricsAccumulator {
    candidate_count: usize,
    scaffold_context_pi_complement_sum: f64,
    fragment_pi_complement_sum: f64,
    fragment_pi_clash_adjusted_sum: f64,
    fragment_cryptic_bonus_sum: f64,
    candidates_with_fragment_complement: usize,
    candidates_with_fragment_cryptic_bonus: usize,
}

impl Default for CoherenceMetricsAccumulator {
    fn default() -> Self {
        Self {
            missing_residue_hits: 0,
            factor_sum: 0.0,
            factor_count: 0,
            min_factor: f64::INFINITY,
            max_factor: f64::NEG_INFINITY,
            raw_pi_clash_sum: 0.0,
            adjusted_pi_clash_sum: 0.0,
        }
    }
}

impl PruneCounters {
    fn record_placement(&self, metrics: PlacementMetrics) {
        let _ = metrics.min_atom_to_cryptic_bonus_a;
        let _ = metrics.atoms_in_complement_voxels;
        if metrics.atoms_within_1_voxel_of_complement > 0 {
            self.candidates_with_atoms_within_1_voxel_complement
                .fetch_add(1, Ordering::Relaxed);
        }
        if metrics.atoms_within_2_voxels_of_complement > 0 {
            self.candidates_with_atoms_within_2_voxels_complement
                .fetch_add(1, Ordering::Relaxed);
        }
        if metrics.min_atom_to_complement_a.is_finite() {
            let mut distances = self
                .placement_min_complement_distances
                .lock()
                .expect("placement distance mutex poisoned");
            distances.push(metrics.min_atom_to_complement_a);
        }
        if metrics.coherence_factor_count_for_clashes > 0
            || metrics.coherence_missing_residue_hits > 0
            || metrics.raw_pi_clash_sum > 0.0
        {
            let mut coherence = self
                .coherence_metrics
                .lock()
                .expect("coherence metrics mutex poisoned");
            coherence.missing_residue_hits += metrics.coherence_missing_residue_hits;
            coherence.factor_sum += metrics.coherence_factor_sum_for_clashes;
            coherence.factor_count += metrics.coherence_factor_count_for_clashes;
            coherence.min_factor = coherence
                .min_factor
                .min(metrics.min_coherence_factor_for_clashes);
            coherence.max_factor = coherence
                .max_factor
                .max(metrics.max_coherence_factor_for_clashes);
            coherence.raw_pi_clash_sum += metrics.raw_pi_clash_sum;
            coherence.adjusted_pi_clash_sum += metrics.adjusted_pi_clash_sum;
        }
        {
            let mut fragment = self
                .fragment_score_metrics
                .lock()
                .expect("fragment score metrics mutex poisoned");
            fragment.candidate_count += 1;
            fragment.scaffold_context_pi_complement_sum += metrics.scaffold_context_pi_complement;
            fragment.fragment_pi_complement_sum += metrics.fragment_pi_complement;
            fragment.fragment_pi_clash_adjusted_sum += metrics.fragment_pi_clash_adjusted;
            fragment.fragment_cryptic_bonus_sum += metrics.fragment_cryptic_bonus;
            if metrics.fragment_pi_complement > 0.0 {
                fragment.candidates_with_fragment_complement += 1;
            }
            if metrics.fragment_cryptic_bonus > 0.0 {
                fragment.candidates_with_fragment_cryptic_bonus += 1;
            }
        }
    }

    fn inc_drop(&self, reason: DropReason) {
        match reason {
            DropReason::Bounds => &self.dropped_bounds,
            DropReason::NoFly => &self.dropped_no_fly,
            DropReason::HardClash => &self.dropped_hard_clash,
            DropReason::InsufficientComplement => &self.dropped_insufficient_complement,
        }
        .fetch_add(1, Ordering::Relaxed);
    }

    fn inc_rotamer_drop(&self, reason: DropReason) {
        match reason {
            DropReason::Bounds => &self.rotamers_dropped_bounds,
            DropReason::NoFly => &self.rotamers_dropped_no_fly,
            DropReason::HardClash => &self.rotamers_dropped_hard_clash,
            DropReason::InsufficientComplement => &self.dropped_insufficient_complement,
        }
        .fetch_add(1, Ordering::Relaxed);
    }

    fn inc_survival(&self, survival_tier: SurvivalTier) {
        match survival_tier {
            SurvivalTier::Normal => &self.survived_normal,
            SurvivalTier::CrypticRescue => &self.survived_cryptic_rescue,
        }
        .fetch_add(1, Ordering::Relaxed);
    }

    fn record_selected_dihedral(&self, value: f64) {
        self.selected_dihedral_degs
            .lock()
            .expect("selected dihedral mutex poisoned")
            .push(value);
    }

    fn snapshot(&self) -> PruneTelemetry {
        let mut distances = self
            .placement_min_complement_distances
            .lock()
            .expect("placement distance mutex poisoned")
            .clone();
        let distance_count = distances.len();
        let mean = if distance_count == 0 {
            f64::INFINITY
        } else {
            distances.iter().sum::<f64>() / distance_count as f64
        };
        let min_seen = distances.iter().copied().fold(f64::INFINITY, f64::min);
        let p10 = quantile_nearest_optional(&mut distances, 0.10).unwrap_or(f64::INFINITY);
        let coherence = *self
            .coherence_metrics
            .lock()
            .expect("coherence metrics mutex poisoned");
        let fragment = *self
            .fragment_score_metrics
            .lock()
            .expect("fragment score metrics mutex poisoned");
        let selected_dihedrals = self
            .selected_dihedral_degs
            .lock()
            .expect("selected dihedral mutex poisoned")
            .clone();
        let mean_dihedral = if selected_dihedrals.is_empty() {
            f64::NAN
        } else {
            selected_dihedrals.iter().sum::<f64>() / selected_dihedrals.len() as f64
        };
        let top_dihedral_bins = top_dihedral_bins(&selected_dihedrals);
        let mean_coherence = if coherence.factor_count == 0 {
            f64::NAN
        } else {
            coherence.factor_sum / coherence.factor_count as f64
        };
        let mean_scaffold_context_pi_complement = if fragment.candidate_count == 0 {
            f64::NAN
        } else {
            fragment.scaffold_context_pi_complement_sum / fragment.candidate_count as f64
        };
        let mean_fragment_pi_complement = if fragment.candidate_count == 0 {
            f64::NAN
        } else {
            fragment.fragment_pi_complement_sum / fragment.candidate_count as f64
        };
        let mean_fragment_pi_clash_adjusted = if fragment.candidate_count == 0 {
            f64::NAN
        } else {
            fragment.fragment_pi_clash_adjusted_sum / fragment.candidate_count as f64
        };
        let mean_fragment_cryptic_bonus = if fragment.candidate_count == 0 {
            f64::NAN
        } else {
            fragment.fragment_cryptic_bonus_sum / fragment.candidate_count as f64
        };
        PruneTelemetry {
            attempted: self.attempted.load(Ordering::Relaxed),
            dropped_assembly: self.dropped_assembly.load(Ordering::Relaxed),
            dropped_bounds: self.dropped_bounds.load(Ordering::Relaxed),
            dropped_no_fly: self.dropped_no_fly.load(Ordering::Relaxed),
            dropped_hard_clash: self.dropped_hard_clash.load(Ordering::Relaxed),
            dropped_insufficient_complement: self
                .dropped_insufficient_complement
                .load(Ordering::Relaxed),
            rotamers_evaluated: self.rotamers_evaluated.load(Ordering::Relaxed),
            rotamers_dropped_bounds: self.rotamers_dropped_bounds.load(Ordering::Relaxed),
            rotamers_dropped_no_fly: self.rotamers_dropped_no_fly.load(Ordering::Relaxed),
            rotamers_dropped_hard_clash: self.rotamers_dropped_hard_clash.load(Ordering::Relaxed),
            candidates_with_at_least_one_valid_rotamer: self
                .candidates_with_at_least_one_valid_rotamer
                .load(Ordering::Relaxed),
            mean_best_dihedral_deg: mean_dihedral,
            top_dihedral_bins,
            z_matrix_active_count: self.z_matrix_active_count.load(Ordering::Relaxed),
            rigid_fallback_count: self.rigid_fallback_count.load(Ordering::Relaxed),
            survived_normal: self.survived_normal.load(Ordering::Relaxed),
            survived_cryptic_rescue: self.survived_cryptic_rescue.load(Ordering::Relaxed),
            cryptic_bonus_atoms: self.cryptic_bonus_atoms.load(Ordering::Relaxed),
            mean_min_atom_to_complement_a: mean,
            p10_min_atom_to_complement_a: p10,
            min_seen_atom_to_complement_a: min_seen,
            candidates_with_atoms_within_1_voxel_complement: self
                .candidates_with_atoms_within_1_voxel_complement
                .load(Ordering::Relaxed),
            candidates_with_atoms_within_2_voxels_complement: self
                .candidates_with_atoms_within_2_voxels_complement
                .load(Ordering::Relaxed),
            coherence_missing_residue_hits: coherence.missing_residue_hits,
            mean_coherence_factor_for_clashes: mean_coherence,
            min_coherence_factor_for_clashes: if coherence.factor_count == 0 {
                f64::NAN
            } else {
                coherence.min_factor
            },
            max_coherence_factor_for_clashes: if coherence.factor_count == 0 {
                f64::NAN
            } else {
                coherence.max_factor
            },
            raw_pi_clash_sum: coherence.raw_pi_clash_sum,
            adjusted_pi_clash_sum: coherence.adjusted_pi_clash_sum,
            mean_scaffold_context_pi_complement,
            mean_fragment_pi_complement,
            mean_fragment_pi_clash_adjusted,
            mean_fragment_cryptic_bonus,
            candidates_with_fragment_complement: fragment.candidates_with_fragment_complement,
            candidates_with_fragment_cryptic_bonus: fragment.candidates_with_fragment_cryptic_bonus,
        }
    }
}

fn score_product(
    product: &Product3D,
    score_atom_offset: usize,
    score_atom_atomic_numbers: &[u8],
    field: &HashMap<u64, VoxelField>,
    no_fly: &HashSet<u64>,
    diagnostics: &FieldDiagnostics,
    geometry: GridGeometry,
) -> ScoreDecision {
    let mut scaffold_context_pi_complement = 0.0;
    let mut scaffold_context_pi_clash = 0.0;
    let mut mapped_atoms = 0_u64;
    let mut scaffold_context_complement_voxels = HashSet::new();
    let mut fragment_complement_unique_voxels = HashSet::new();
    let mut min_atom_to_complement_a = f64::INFINITY;
    let mut min_atom_to_cryptic_bonus_a = f64::INFINITY;
    let mut atoms_in_complement_voxels = 0_usize;
    let mut atoms_within_1_voxel_of_complement = 0_usize;
    let mut atoms_within_2_voxels_of_complement = 0_usize;
    let mut coherence_missing_residue_hits = 0_usize;
    let mut coherence_factor_sum_for_clashes = 0.0_f64;
    let mut coherence_factor_count_for_clashes = 0_usize;
    let mut min_coherence_factor_for_clashes = f64::INFINITY;
    let mut max_coherence_factor_for_clashes = f64::NEG_INFINITY;
    let mut raw_pi_clash_sum = 0.0_f64;
    let mut adjusted_pi_clash_sum = 0.0_f64;
    let scaffold_points: Vec<[f32; 3]> = product
        .coordinates
        .chunks_exact(3)
        .take(score_atom_offset)
        .map(|xyz| [xyz[0], xyz[1], xyz[2]])
        .collect();
    let mut scored_atom_positions: Vec<(f64, f64, f64)> = Vec::new();
    for (atom_idx, xyz) in product.coordinates.chunks_exact(3).enumerate() {
        let is_fragment_atom = atom_idx >= score_atom_offset;
        if !is_fragment_atom {
            if let Some(voxel_idx) = coordinate_to_voxel([xyz[0], xyz[1], xyz[2]], geometry) {
                if let Some(voxel) = field.get(&voxel_idx) {
                    scaffold_context_pi_complement += voxel.complement;
                    scaffold_context_pi_clash += voxel.clash;
                    if voxel.complement > 0.0 {
                        scaffold_context_complement_voxels.insert(voxel_idx);
                    }
                }
            }
            continue;
        }
        if atom_idx < score_atom_offset {
            continue;
        }
        let score_atom_idx = atom_idx - score_atom_offset;
        if score_atom_atomic_numbers
            .get(score_atom_idx)
            .is_some_and(|atomic_number| *atomic_number <= 1)
        {
            continue;
        }
        if min_distance_to_points([xyz[0], xyz[1], xyz[2]], &scaffold_points)
            <= FRAGMENT_CONTEXT_EXCLUSION_A
        {
            continue;
        }
        let voxel_idx = match coordinate_to_voxel([xyz[0], xyz[1], xyz[2]], geometry) {
            Some(value) => value,
            None => return ScoreDecision::Drop(DropReason::Bounds, None),
        };
        if no_fly.contains(&voxel_idx) {
            return ScoreDecision::Drop(DropReason::NoFly, None);
        }
        mapped_atoms += 1;
        scored_atom_positions.push((f64::from(xyz[0]), f64::from(xyz[1]), f64::from(xyz[2])));
        if diagnostics.complement_voxels.contains(&voxel_idx) {
            atoms_in_complement_voxels += 1;
            if !has_target_within_radius_voxels(
                voxel_idx,
                &scaffold_context_complement_voxels,
                geometry,
                0,
            ) {
                fragment_complement_unique_voxels.insert(voxel_idx);
            }
        }
        if has_target_within_radius_voxels(voxel_idx, &diagnostics.complement_voxels, geometry, 1) {
            atoms_within_1_voxel_of_complement += 1;
        }
        if has_target_within_radius_voxels(voxel_idx, &diagnostics.complement_voxels, geometry, 2) {
            atoms_within_2_voxels_of_complement += 1;
        }
        min_atom_to_complement_a = min_atom_to_complement_a.min(
            diagnostics.nearest_complement_distance([xyz[0], xyz[1], xyz[2]], voxel_idx, geometry),
        );
        min_atom_to_cryptic_bonus_a = min_atom_to_cryptic_bonus_a.min(
            diagnostics.nearest_cryptic_distance([xyz[0], xyz[1], xyz[2]], voxel_idx, geometry),
        );
        if let Some(voxel) = field.get(&voxel_idx) {
            if voxel.complement > 0.0 {
                if !has_target_within_radius_voxels(
                    voxel_idx,
                    &scaffold_context_complement_voxels,
                    geometry,
                    0,
                ) {
                    fragment_complement_unique_voxels.insert(voxel_idx);
                }
            }
            if voxel.raw_clash > 0.0 {
                raw_pi_clash_sum += voxel.raw_clash;
                adjusted_pi_clash_sum += voxel.clash;
                coherence_factor_sum_for_clashes += voxel.coherence_factor;
                coherence_factor_count_for_clashes += 1;
                min_coherence_factor_for_clashes =
                    min_coherence_factor_for_clashes.min(voxel.coherence_factor);
                max_coherence_factor_for_clashes =
                    max_coherence_factor_for_clashes.max(voxel.coherence_factor);
                if voxel.coherence_missing {
                    coherence_missing_residue_hits += 1;
                }
            }
        }
    }
    let shared_score = score_positions_with_field(
        &scored_atom_positions,
        field,
        |x, y, z| coordinate_to_voxel([x as f32, y as f32, z as f32], geometry),
        &HashSet::new(),
        &HashMap::new(),
    );
    let fragment_pi_complement = shared_score.pi_complement;
    let fragment_pi_clash = shared_score.pi_clash_pocket + shared_score.pi_clash_lock;
    let fragment_cryptic_bonus = shared_score.cryptic_bonus;
    let cryptic_bonus_atoms = shared_score.cryptic_bonus_atoms;
    let placement_metrics = PlacementMetrics {
        min_atom_to_complement_a,
        min_atom_to_cryptic_bonus_a,
        atoms_in_complement_voxels,
        atoms_within_1_voxel_of_complement,
        atoms_within_2_voxels_of_complement,
        coherence_missing_residue_hits,
        coherence_factor_sum_for_clashes,
        coherence_factor_count_for_clashes,
        min_coherence_factor_for_clashes,
        max_coherence_factor_for_clashes,
        raw_pi_clash_sum,
        adjusted_pi_clash_sum,
        scaffold_context_pi_complement,
        scaffold_context_pi_clash,
        fragment_pi_complement,
        fragment_pi_clash_adjusted: fragment_pi_clash,
        fragment_cryptic_bonus,
    };
    if fragment_pi_clash > HARD_CLASH_LIMIT {
        return ScoreDecision::Drop(DropReason::HardClash, Some(placement_metrics));
    }
    if mapped_atoms == 0 {
        return ScoreDecision::Drop(DropReason::InsufficientComplement, Some(placement_metrics));
    }
    let fragment_pi_complement_unique_voxels = fragment_complement_unique_voxels.len();
    let survival_tier = if fragment_pi_clash <= NORMAL_CLASH_LIMIT
        && fragment_pi_complement_unique_voxels >= 2
    {
        SurvivalTier::Normal
    } else if fragment_pi_clash <= HARD_CLASH_LIMIT
        && fragment_pi_complement_unique_voxels >= 1
        && fragment_cryptic_bonus >= CRYPTIC_RESCUE_FLOOR
    {
        SurvivalTier::CrypticRescue
    } else {
        return ScoreDecision::Drop(DropReason::InsufficientComplement, Some(placement_metrics));
    };
    let normalization = mapped_atoms as f64;
    let score =
        (fragment_pi_complement + fragment_cryptic_bonus - fragment_pi_clash) / normalization;
    ScoreDecision::Keep(ProductScore {
        score,
        pi_complement: fragment_pi_complement,
        pi_clash: fragment_pi_clash,
        cryptic_bonus: fragment_cryptic_bonus,
        cryptic_bonus_atoms,
        survival_tier,
        placement_metrics,
    })
}

fn retained_synthon_atomic_numbers(synthon: &LoadedSynthon) -> Vec<u8> {
    let attachment_atom = synthon.attachment().atom_index;
    let skip_atom = synthon.attachment().leaving_group_atom_index;
    synthon
        .atomic_numbers
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(atom_idx, atomic_number)| {
            if skip_atom == Some(atom_idx) {
                None
            } else if atom_idx == attachment_atom {
                Some(0)
            } else {
                Some(atomic_number)
            }
        })
        .collect()
}

fn retained_synthon_atomic_numbers_for_site(
    synthon: &LoadedSynthon,
    reactive_atom_idx: usize,
    leaving_group_atom_indices: &[usize],
) -> Vec<u8> {
    let leaving_groups = leaving_group_atom_indices
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    synthon
        .atomic_numbers
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(atom_idx, atomic_number)| {
            if leaving_groups.contains(&atom_idx) {
                None
            } else if atom_idx == reactive_atom_idx {
                Some(0)
            } else {
                Some(atomic_number)
            }
        })
        .collect()
}

fn map_intermediate_synthon_atom(
    index_map: &[Option<usize>],
    synthon_atom_idx: usize,
    label: &str,
) -> Result<usize> {
    index_map
        .get(synthon_atom_idx)
        .copied()
        .flatten()
        .ok_or_else(|| anyhow!("{label} {synthon_atom_idx} was removed or unmapped"))
}

fn fallback_synthon_reference_atom(synthon: &LoadedSynthon, reactive_atom_idx: usize) -> usize {
    synthon
        .atomic_numbers
        .iter()
        .enumerate()
        .find_map(|(atom_idx, atomic_number)| {
            (*atomic_number > 1 && atom_idx != reactive_atom_idx).then_some(atom_idx)
        })
        .unwrap_or(reactive_atom_idx)
}

fn fallback_product_reference_atom(state: &ScaffoldState3D, excluded: &[usize]) -> usize {
    (0..state.atom_count())
        .find(|atom_idx| !excluded.contains(atom_idx))
        .unwrap_or_else(|| excluded.first().copied().unwrap_or(0))
}

fn synthon_reference_atom_index(synthon: &LoadedSynthon) -> Result<usize> {
    if let Some(atom_idx) = synthon.attachment().dihedral_reference_atom_index {
        return Ok(atom_idx);
    }
    let attachment_atom = synthon.attachment().atom_index;
    synthon
        .atomic_numbers
        .iter()
        .copied()
        .enumerate()
        .find_map(|(atom_idx, atomic_number)| {
            if atom_idx != attachment_atom && atomic_number > 1 {
                Some(atom_idx)
            } else {
                None
            }
        })
        .ok_or_else(|| anyhow!("synthon {} lacks a reference atom", synthon.synthon_id))
}

fn coordinate_to_voxel(coordinate: [f32; 3], geometry: GridGeometry) -> Option<u64> {
    let mut index = [0_i64; 3];
    let dims = [geometry.nx, geometry.ny, geometry.nz];
    for axis in 0..3 {
        let raw = ((coordinate[axis] - geometry.origin[axis]) / geometry.spacing_a).floor();
        if !raw.is_finite() || raw < 0.0 || raw >= dims[axis] as f32 {
            return None;
        }
        index[axis] = raw as i64;
    }
    Some((index[2] * geometry.nx * geometry.ny + index[1] * geometry.nx + index[0]) as u64)
}

fn field_counts(field: &HashMap<u64, VoxelField>) -> FieldCounts {
    FieldCounts {
        stable_occupied: field.values().filter(|voxel| voxel.stable_occupied).count(),
        thermally_destabilized: field
            .values()
            .filter(|voxel| voxel.thermally_destabilized)
            .count(),
        thermally_activated: field
            .values()
            .filter(|voxel| voxel.thermally_activated)
            .count(),
        thermally_released: field
            .values()
            .filter(|voxel| voxel.thermally_released)
            .count(),
        complement_voxels: field
            .values()
            .filter(|voxel| voxel.complement > 0.0)
            .count(),
        cryptic_bonus_voxels: field
            .values()
            .filter(|voxel| voxel.cryptic_bonus > 0.0)
            .count(),
    }
}

fn voxel_to_indices(voxel_idx: u64, geometry: GridGeometry) -> Option<[i64; 3]> {
    let plane = u64::try_from(geometry.nx.checked_mul(geometry.ny)?).ok()?;
    let nx = u64::try_from(geometry.nx).ok()?;
    let z = voxel_idx / plane;
    let rem = voxel_idx % plane;
    let y = rem / nx;
    let x = rem % nx;
    Some([
        i64::try_from(x).ok()?,
        i64::try_from(y).ok()?,
        i64::try_from(z).ok()?,
    ])
}

fn indices_to_voxel(x: i64, y: i64, z: i64, geometry: GridGeometry) -> Option<u64> {
    if !(0..geometry.nx).contains(&x)
        || !(0..geometry.ny).contains(&y)
        || !(0..geometry.nz).contains(&z)
    {
        return None;
    }
    Some((z * geometry.nx * geometry.ny + y * geometry.nx + x) as u64)
}

fn voxel_center(voxel_idx: u64, geometry: GridGeometry) -> Option<[f64; 3]> {
    let [x, y, z] = voxel_to_indices(voxel_idx, geometry)?;
    Some([
        f64::from(geometry.origin[0]) + (x as f64 + 0.5) * f64::from(geometry.spacing_a),
        f64::from(geometry.origin[1]) + (y as f64 + 0.5) * f64::from(geometry.spacing_a),
        f64::from(geometry.origin[2]) + (z as f64 + 0.5) * f64::from(geometry.spacing_a),
    ])
}

impl FieldDiagnostics {
    fn new(
        field: &HashMap<u64, VoxelField>,
        no_fly: &HashSet<u64>,
        geometry: GridGeometry,
    ) -> Self {
        let mut complement_centers = Vec::new();
        let mut cryptic_bonus_centers = Vec::new();
        let mut clash_centers = Vec::new();
        let mut no_fly_centers = Vec::new();
        let mut complement_voxels = HashSet::new();
        let mut cryptic_bonus_voxels = HashSet::new();

        for (voxel_idx, voxel) in field {
            if voxel.complement > 0.0 {
                complement_voxels.insert(*voxel_idx);
                if let Some(center) = voxel_center(*voxel_idx, geometry) {
                    complement_centers.push(center);
                }
            }
            if voxel.cryptic_bonus > 0.0 {
                cryptic_bonus_voxels.insert(*voxel_idx);
                if let Some(center) = voxel_center(*voxel_idx, geometry) {
                    cryptic_bonus_centers.push(center);
                }
            }
            if voxel.clash > 0.0 {
                if let Some(center) = voxel_center(*voxel_idx, geometry) {
                    clash_centers.push(center);
                }
            }
        }
        for voxel_idx in no_fly {
            if let Some(center) = voxel_center(*voxel_idx, geometry) {
                no_fly_centers.push(center);
            }
        }

        Self {
            complement_centers,
            cryptic_bonus_centers,
            clash_centers,
            no_fly_centers,
            complement_voxels,
            cryptic_bonus_voxels,
            nearest_complement_cache: Mutex::new(HashMap::new()),
            nearest_cryptic_cache: Mutex::new(HashMap::new()),
        }
    }

    fn nearest_complement_distance(
        &self,
        coordinate: [f32; 3],
        voxel_idx: u64,
        geometry: GridGeometry,
    ) -> f64 {
        self.nearest_distance_to_cached_voxel(
            coordinate,
            voxel_idx,
            geometry,
            &self.complement_voxels,
            &self.nearest_complement_cache,
        )
    }

    fn nearest_cryptic_distance(
        &self,
        coordinate: [f32; 3],
        voxel_idx: u64,
        geometry: GridGeometry,
    ) -> f64 {
        self.nearest_distance_to_cached_voxel(
            coordinate,
            voxel_idx,
            geometry,
            &self.cryptic_bonus_voxels,
            &self.nearest_cryptic_cache,
        )
    }

    fn nearest_distance_to_cached_voxel(
        &self,
        coordinate: [f32; 3],
        voxel_idx: u64,
        geometry: GridGeometry,
        targets: &HashSet<u64>,
        cache: &Mutex<HashMap<u64, Option<u64>>>,
    ) -> f64 {
        let target_idx = {
            let cached = cache
                .lock()
                .expect("nearest voxel cache mutex poisoned")
                .get(&voxel_idx)
                .copied();
            if let Some(value) = cached {
                value
            } else {
                let value = nearest_target_voxel_by_center(voxel_idx, targets, geometry);
                cache
                    .lock()
                    .expect("nearest voxel cache mutex poisoned")
                    .insert(voxel_idx, value);
                value
            }
        };
        target_idx
            .and_then(|target| voxel_center(target, geometry))
            .map(|center| {
                distance_sq(
                    [
                        f64::from(coordinate[0]),
                        f64::from(coordinate[1]),
                        f64::from(coordinate[2]),
                    ],
                    center,
                )
                .sqrt()
            })
            .unwrap_or(f64::INFINITY)
    }
}

fn nearest_target_voxel_by_center(
    source_idx: u64,
    targets: &HashSet<u64>,
    geometry: GridGeometry,
) -> Option<u64> {
    if targets.is_empty() {
        return None;
    }
    let [sx, sy, sz] = voxel_to_indices(source_idx, geometry)?;
    let source_center = voxel_center(source_idx, geometry)?;
    let mut best_idx = None;
    let mut best_dist_sq = f64::INFINITY;
    let max_radius = geometry.nx.max(geometry.ny).max(geometry.nz);
    for radius in 0..=max_radius {
        for dz in -radius..=radius {
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    if dx.abs().max(dy.abs()).max(dz.abs()) != radius {
                        continue;
                    }
                    let Some(candidate_idx) = indices_to_voxel(sx + dx, sy + dy, sz + dz, geometry)
                    else {
                        continue;
                    };
                    if !targets.contains(&candidate_idx) {
                        continue;
                    }
                    if let Some(center) = voxel_center(candidate_idx, geometry) {
                        let dist_sq = distance_sq(source_center, center);
                        if dist_sq < best_dist_sq {
                            best_dist_sq = dist_sq;
                            best_idx = Some(candidate_idx);
                        }
                    }
                }
            }
        }
        let next_shell_lower_bound = ((radius + 1) as f64 * f64::from(geometry.spacing_a)).powi(2);
        if best_idx.is_some() && best_dist_sq <= next_shell_lower_bound {
            break;
        }
    }
    best_idx
}

fn has_target_within_radius_voxels(
    source_idx: u64,
    targets: &HashSet<u64>,
    geometry: GridGeometry,
    radius: i64,
) -> bool {
    let Some([sx, sy, sz]) = voxel_to_indices(source_idx, geometry) else {
        return false;
    };
    for dz in -radius..=radius {
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if let Some(candidate_idx) = indices_to_voxel(sx + dx, sy + dy, sz + dz, geometry) {
                    if targets.contains(&candidate_idx) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

fn print_anchor_field_diagnostics(
    scaffold: &ScaffoldTemplate,
    diagnostics: &FieldDiagnostics,
    geometry: GridGeometry,
) {
    println!(
        "placement_field_summary complement_centers={} cryptic_bonus_centers={} clash_centers={} no_fly_centers={}",
        diagnostics.complement_centers.len(),
        diagnostics.cryptic_bonus_centers.len(),
        diagnostics.clash_centers.len(),
        diagnostics.no_fly_centers.len()
    );
    for (anchor_idx, anchor) in scaffold.anchors.iter().enumerate() {
        let Some(attachment) = scaffold.state.attachment_points.get(anchor_idx) else {
            continue;
        };
        let exit_xyz = scaffold.state.atom_xyz(attachment.atom_index);
        let exit_vector = normalize_vec3(attachment.attachment_vector, "diagnostic exit vector")
            .unwrap_or([0.0, 0.0, 0.0]);
        println!(
            "anchor_diagnostic anchor_idx={} original_scaffold_atom_idx={} removed_fragment_atom_idx={} scaffold_exit_xyz=[{:.6},{:.6},{:.6}] exit_vector_final=[{:.6},{:.6},{:.6}]",
            anchor_idx,
            anchor.original_scaffold_atom_idx,
            anchor.removed_fragment_atom_idx,
            exit_xyz[0],
            exit_xyz[1],
            exit_xyz[2],
            exit_vector[0],
            exit_vector[1],
            exit_vector[2]
        );
        for (label, direction) in diagnostic_ray_directions(exit_vector) {
            let metrics = ray_metrics(
                [
                    f64::from(exit_xyz[0]),
                    f64::from(exit_xyz[1]),
                    f64::from(exit_xyz[2]),
                ],
                direction,
                diagnostics,
            );
            println!(
                "ray_diagnostic anchor_idx={} ray={} nearest_complement_distance_A={:.6} complement_hits_within_3A={} complement_hits_within_5A={} complement_hits_within_8A={} nearest_cryptic_bonus_distance_A={:.6} cryptic_bonus_hits_within_5A={} clash_hits_within_3A={} no_fly_hits_within_4A={}",
                anchor_idx,
                label,
                metrics.nearest_complement_distance_a,
                metrics.complement_hits_within_3a,
                metrics.complement_hits_within_5a,
                metrics.complement_hits_within_8a,
                metrics.nearest_cryptic_bonus_distance_a,
                metrics.cryptic_bonus_hits_within_5a,
                metrics.clash_hits_within_3a,
                metrics.no_fly_hits_within_4a
            );
        }
    }
    let _ = geometry;
}

#[derive(Debug, Clone, Copy)]
struct RayMetrics {
    nearest_complement_distance_a: f64,
    complement_hits_within_3a: usize,
    complement_hits_within_5a: usize,
    complement_hits_within_8a: usize,
    nearest_cryptic_bonus_distance_a: f64,
    cryptic_bonus_hits_within_5a: usize,
    clash_hits_within_3a: usize,
    no_fly_hits_within_4a: usize,
}

fn ray_metrics(
    origin: [f64; 3],
    direction: [f64; 3],
    diagnostics: &FieldDiagnostics,
) -> RayMetrics {
    let complement = ray_center_metrics(origin, direction, &diagnostics.complement_centers);
    let cryptic = ray_center_metrics(origin, direction, &diagnostics.cryptic_bonus_centers);
    let clash = ray_center_metrics(origin, direction, &diagnostics.clash_centers);
    let no_fly = ray_center_metrics(origin, direction, &diagnostics.no_fly_centers);
    RayMetrics {
        nearest_complement_distance_a: complement.0,
        complement_hits_within_3a: complement.1,
        complement_hits_within_5a: complement.2,
        complement_hits_within_8a: complement.3,
        nearest_cryptic_bonus_distance_a: cryptic.0,
        cryptic_bonus_hits_within_5a: cryptic.2,
        clash_hits_within_3a: clash.1,
        no_fly_hits_within_4a: no_fly.4,
    }
}

fn ray_center_metrics(
    origin: [f64; 3],
    direction: [f64; 3],
    centers: &[[f64; 3]],
) -> (f64, usize, usize, usize, usize) {
    let mut nearest = f64::INFINITY;
    let mut within_3 = 0_usize;
    let mut within_4 = 0_usize;
    let mut within_5 = 0_usize;
    let mut within_8 = 0_usize;
    for center in centers {
        let distance = distance_to_ray_segment(origin, direction, *center, 8.0);
        nearest = nearest.min(distance);
        if distance <= 3.0 {
            within_3 += 1;
        }
        if distance <= 4.0 {
            within_4 += 1;
        }
        if distance <= 5.0 {
            within_5 += 1;
        }
        if distance <= 8.0 {
            within_8 += 1;
        }
    }
    (nearest, within_3, within_5, within_8, within_4)
}

fn distance_to_ray_segment(
    origin: [f64; 3],
    direction: [f64; 3],
    point: [f64; 3],
    max_t: f64,
) -> f64 {
    let relative = [
        point[0] - origin[0],
        point[1] - origin[1],
        point[2] - origin[2],
    ];
    let t = dot_f64(relative, direction).clamp(0.0, max_t);
    let closest = [
        origin[0] + direction[0] * t,
        origin[1] + direction[1] * t,
        origin[2] + direction[2] * t,
    ];
    distance_sq(point, closest).sqrt()
}

fn diagnostic_ray_directions(exit_vector: [f32; 3]) -> Vec<(&'static str, [f64; 3])> {
    let exit = normalize_f64([
        f64::from(exit_vector[0]),
        f64::from(exit_vector[1]),
        f64::from(exit_vector[2]),
    ]);
    let fallback = if exit[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let u = normalize_f64(cross_f64(exit, fallback));
    let v = normalize_f64(cross_f64(exit, u));
    vec![
        ("+exit", exit),
        ("-exit", mul_f64(exit, -1.0)),
        ("jitter+u", u),
        ("jitter-u", mul_f64(u, -1.0)),
        ("jitter+v", v),
        ("jitter-v", mul_f64(v, -1.0)),
        ("jitter+uv", normalize_f64(add_f64(u, v))),
        ("jitter-uv", normalize_f64(mul_f64(add_f64(u, v), -1.0))),
    ]
}

fn load_signal_field(
    path: &Path,
    condition_id: &str,
    coherence: &HashMap<i64, f64>,
    pathway_context: &PathwayContext,
    voxel_thresholds: &VoxelThresholds,
    residue_coordinates: &[ResidueCoordinate],
    geometry: GridGeometry,
    config: &Config,
) -> Result<HashMap<u64, VoxelField>> {
    let mut audit = collect_field_audit(path, condition_id, config)?;
    audit.release_delta_threshold = voxel_thresholds.release_p90;
    audit.gain_delta_threshold = voxel_thresholds.gain_p90;
    print_field_audit(path, condition_id, &audit);
    let mut reader = parquet_reader(path)?;
    let mut field = HashMap::new();
    for batch in &mut reader {
        let batch = batch?;
        for row in 0..batch.num_rows() {
            if string_value(&batch, "condition_id", row)? != condition_id {
                continue;
            }
            let cold = f64_value(&batch, "hit_count_cold_mean", row)?;
            let warm = f64_value(&batch, "hit_count_warm_mean", row)?;
            let voxel_idx = u64_value(&batch, "voxel_idx", row)?;
            let x_idx = i64_value(&batch, "x_idx", row)?;
            let y_idx = i64_value(&batch, "y_idx", row)?;
            let z_idx = i64_value(&batch, "z_idx", row)?;
            let spatial_residue_idx =
                nearest_residue_for_voxel(x_idx, y_idx, z_idx, geometry, residue_coordinates);
            let dynamic_class = classify_dynamic_voxel(cold, warm, voxel_thresholds);
            if !dynamic_class.stable_occupied
                && !dynamic_class.thermally_destabilized
                && !dynamic_class.thermally_activated
                && !dynamic_class.thermally_released
            {
                *audit
                    .effective_variance_counts
                    .entry("void".to_owned())
                    .or_default() += 1;
                continue;
            }
            if dynamic_class.stable_occupied {
                *audit
                    .effective_variance_counts
                    .entry("stable_occupied".to_owned())
                    .or_default() += 1;
            }
            if dynamic_class.thermally_destabilized {
                *audit
                    .effective_variance_counts
                    .entry("thermally_destabilized".to_owned())
                    .or_default() += 1;
            }
            if dynamic_class.thermally_activated {
                *audit
                    .effective_variance_counts
                    .entry("thermally_activated".to_owned())
                    .or_default() += 1;
            }
            if dynamic_class.thermally_released {
                *audit
                    .effective_variance_counts
                    .entry("thermally_released".to_owned())
                    .or_default() += 1;
            }
            let residue_idx = signal_row_residue_idx(&batch, row)?
                .or_else(|| pathway_context.voxel_to_residue.get(&voxel_idx).copied())
                .or(spatial_residue_idx);
            let coherence_lookup = residue_idx.and_then(|value| coherence.get(&value).copied());
            let coherence_missing = coherence_lookup.is_none();
            let coherence_score = coherence_lookup.unwrap_or(1.0).clamp(0.0, 1.0);
            let coherence_factor = coherence_score.max(0.35);
            let raw_clash = if dynamic_class.stable_occupied {
                1.0
            } else {
                0.0
            } + if dynamic_class.thermally_destabilized {
                0.5
            } else {
                0.0
            };
            let mut complement = if dynamic_class.thermally_activated {
                1.0
            } else {
                0.0
            } + if dynamic_class.thermally_released {
                1.0
            } else {
                0.0
            };
            let clash = raw_clash * coherence_factor;
            if complement > 0.0
                && residue_idx
                    .is_some_and(|value| pathway_context.kinetic_burst_residues.contains(&value))
            {
                complement *= 1.5;
            }
            let activated_cryptic = dynamic_class.thermally_activated
                && voxel_thresholds.gain_p90 > 0.0
                && dynamic_class.gain_delta >= voxel_thresholds.gain_p90;
            let released_cryptic = dynamic_class.thermally_released
                && dynamic_class.release_delta >= voxel_thresholds.release_p95;
            let cryptic_bonus = if activated_cryptic || released_cryptic {
                2.0
            } else {
                0.0
            };
            if complement == 0.0 && clash == 0.0 {
                continue;
            }
            field.insert(
                voxel_idx,
                VoxelField {
                    complement,
                    clash,
                    raw_clash,
                    cryptic_bonus,
                    stable_occupied: dynamic_class.stable_occupied,
                    thermally_destabilized: dynamic_class.thermally_destabilized,
                    thermally_activated: dynamic_class.thermally_activated,
                    thermally_released: dynamic_class.thermally_released,
                    coherence_score,
                    coherence_factor,
                    coherence_missing,
                    primary_residue_idx: residue_idx,
                    cold_mean: cold,
                    warm_mean: warm,
                    delta: dynamic_class.gain_delta,
                    consensus_complement_bonus: 0.0,
                    on_activation_pathway: residue_idx.is_some_and(|value| {
                        pathway_context.kinetic_burst_residues.contains(&value)
                    }),
                },
            );
        }
    }
    println!(
        "field_classification_histogram effective condition={} stable_occupied={} thermally_destabilized={} thermally_activated={} thermally_released={} void={}",
        condition_id,
        audit.effective_count("stable_occupied"),
        audit.effective_count("thermally_destabilized"),
        audit.effective_count("thermally_activated"),
        audit.effective_count("thermally_released"),
        audit.effective_count("void")
    );
    Ok(field)
}

impl FieldAudit {
    fn raw_count(&self, label: &str) -> u64 {
        self.raw_variance_counts.get(label).copied().unwrap_or(0)
    }

    fn effective_count(&self, label: &str) -> u64 {
        self.effective_variance_counts
            .get(label)
            .copied()
            .unwrap_or(0)
    }
}

fn collect_field_audit(path: &Path, condition_id: &str, config: &Config) -> Result<FieldAudit> {
    let mut reader = parquet_reader(path)?;
    let mut columns = Vec::new();
    let mut condition_counts = BTreeMap::new();
    let mut raw_variance_counts = BTreeMap::new();
    let mut boundary_class_counts: Option<BTreeMap<String, u64>> = None;
    let mut voxel_context_counts: Option<BTreeMap<String, u64>> = None;
    let mut hit_count_deltas = Vec::new();
    let mut positive_release_deltas = Vec::new();
    let mut positive_gain_deltas = Vec::new();
    let mut delta_min = f64::INFINITY;
    let mut delta_max = f64::NEG_INFINITY;
    let mut delta_sum = 0.0_f64;
    let mut delta_count = 0_u64;
    let mut release_min = f64::INFINITY;
    let mut release_max = f64::NEG_INFINITY;
    let mut release_sum = 0.0_f64;
    let mut release_count = 0_u64;
    let mut has_voxel_activated_fraction = false;

    for batch in &mut reader {
        let batch = batch?;
        if columns.is_empty() {
            let schema = batch.schema();
            columns = schema
                .fields()
                .iter()
                .map(|field| field.name().to_owned())
                .collect();
            if batch.column_by_name("boundary_class").is_some() {
                boundary_class_counts = Some(BTreeMap::new());
            }
            if batch.column_by_name("voxel_context").is_some() {
                voxel_context_counts = Some(BTreeMap::new());
            }
            has_voxel_activated_fraction =
                batch.column_by_name("voxel_activated_fraction").is_some();
        }
        for row in 0..batch.num_rows() {
            let row_condition = string_value(&batch, "condition_id", row)?;
            *condition_counts.entry(row_condition.clone()).or_default() += 1;
            if row_condition != condition_id {
                continue;
            }
            let variance_class = optional_variance_class_value(&batch, row)?
                .unwrap_or_else(|| "<missing>".to_owned());
            *raw_variance_counts.entry(variance_class).or_default() += 1;

            if let Some(counts) = boundary_class_counts.as_mut() {
                if let Some(value) = optional_string_value(&batch, "boundary_class", row)? {
                    *counts.entry(value).or_default() += 1;
                }
            }
            if let Some(counts) = voxel_context_counts.as_mut() {
                if let Some(value) = optional_string_value(&batch, "voxel_context", row)? {
                    *counts.entry(value).or_default() += 1;
                }
            }

            let cold = f64_value(&batch, "hit_count_cold_mean", row)?;
            let warm = f64_value(&batch, "hit_count_warm_mean", row)?;
            let delta = warm - cold;
            if delta.is_finite() {
                hit_count_deltas.push(delta);
                delta_min = delta_min.min(delta);
                delta_max = delta_max.max(delta);
                delta_sum += delta;
                delta_count += 1;
            }
            let release_delta = cold - warm;
            if release_delta.is_finite() {
                release_min = release_min.min(release_delta);
                release_max = release_max.max(release_delta);
                release_sum += release_delta;
                release_count += 1;
                if release_delta > 0.0 {
                    positive_release_deltas.push(release_delta);
                }
            }
            let gain_delta = warm - cold;
            if gain_delta.is_finite() && gain_delta > 0.0 {
                positive_gain_deltas.push(gain_delta);
            }
        }
    }

    let p90 = quantile_nearest(&mut hit_count_deltas, 0.90);
    let release_p90 =
        quantile_nearest_optional(&mut positive_release_deltas, 0.90).unwrap_or(f64::INFINITY);
    let gain_p90 =
        quantile_nearest_optional(&mut positive_gain_deltas, 0.90).unwrap_or(f64::INFINITY);
    let mean = if delta_count == 0 {
        0.0
    } else {
        delta_sum / delta_count as f64
    };
    let release_mean = if release_count == 0 {
        0.0
    } else {
        release_sum / release_count as f64
    };
    let fallback_enabled = raw_variance_counts
        .get("thermally_activated")
        .copied()
        .unwrap_or(0)
        == 0;
    let release_delta_threshold = config.release_delta_threshold.unwrap_or(release_p90);
    let gain_delta_threshold = config.gain_delta_threshold.unwrap_or(gain_p90);
    Ok(FieldAudit {
        columns,
        condition_counts,
        raw_variance_counts,
        boundary_class_counts,
        voxel_context_counts,
        effective_variance_counts: BTreeMap::new(),
        hit_count_delta_count: delta_count,
        hit_count_delta_min: if delta_count == 0 { 0.0 } else { delta_min },
        hit_count_delta_max: if delta_count == 0 { 0.0 } else { delta_max },
        hit_count_delta_mean: mean,
        hit_count_delta_p90: p90,
        release_delta_count: release_count,
        release_delta_min: if release_count == 0 { 0.0 } else { release_min },
        release_delta_max: if release_count == 0 { 0.0 } else { release_max },
        release_delta_mean: release_mean,
        release_delta_positive_p90: release_p90,
        release_delta_threshold,
        gain_delta_positive_p90: gain_p90,
        gain_delta_threshold,
        stability_epsilon: config.stability_epsilon,
        fallback_enabled,
        has_voxel_activated_fraction,
    })
}

fn print_field_audit(path: &Path, condition_id: &str, audit: &FieldAudit) {
    println!(
        "field_columns path={} columns=[{}]",
        path.display(),
        audit.columns.join(",")
    );
    println!(
        "condition_distribution {}",
        count_map_json(&audit.condition_counts)
    );
    println!(
        "field_classification_histogram raw condition={} stable_occupied={} thermally_destabilized={} thermally_activated={} void={}",
        condition_id,
        audit.raw_count("stable_occupied"),
        audit.raw_count("thermally_destabilized"),
        audit.raw_count("thermally_activated"),
        audit.raw_count("void")
    );
    println!(
        "hit_count_delta_stats condition={} count={} min={:.6} max={:.6} mean={:.6} p90={:.6}",
        condition_id,
        audit.hit_count_delta_count,
        audit.hit_count_delta_min,
        audit.hit_count_delta_max,
        audit.hit_count_delta_mean,
        audit.hit_count_delta_p90
    );
    println!(
        "release_delta_stats condition={} count={} min={:.6} max={:.6} mean={:.6} positive_p90={:.6} release_delta_threshold={:.6} gain_delta_positive_p90={:.6} gain_delta_threshold={:.6} stability_epsilon={:.6}",
        condition_id,
        audit.release_delta_count,
        audit.release_delta_min,
        audit.release_delta_max,
        audit.release_delta_mean,
        audit.release_delta_positive_p90,
        audit.release_delta_threshold,
        audit.gain_delta_positive_p90,
        audit.gain_delta_threshold,
        audit.stability_epsilon
    );
    match &audit.boundary_class_counts {
        Some(counts) => println!("boundary_class_distribution {}", count_map_json(counts)),
        None => println!("boundary_class_distribution <absent>"),
    }
    match &audit.voxel_context_counts {
        Some(counts) => println!("voxel_context_distribution {}", count_map_json(counts)),
        None => println!("voxel_context_distribution <absent>"),
    }
    if audit.fallback_enabled {
        println!(
            "activation_fallback enabled reason=raw_thermally_activated_zero strategies=boundary_class_or_voxel_context_labels,voxel_activated_fraction_gt_0,release_delta_gt_threshold release_delta_threshold={:.6} voxel_activated_fraction_present={}",
            audit.release_delta_threshold,
            audit.has_voxel_activated_fraction
        );
    } else {
        println!("activation_fallback disabled reason=raw_thermally_activated_present");
    }
}

fn load_voxel_thresholds(path: &Path, condition_id: &str) -> Result<VoxelThresholds> {
    let payload: Value = serde_json::from_reader(File::open(path)?)
        .with_context(|| format!("parse voxel thresholds {}", path.display()))?;
    let payload_condition = payload
        .get("condition_id")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("voxel thresholds missing condition_id"))?;
    if payload_condition != condition_id {
        return Err(anyhow!(
            "voxel thresholds condition_id mismatch: expected {}, got {}",
            condition_id,
            payload_condition
        ));
    }
    Ok(VoxelThresholds {
        cold_p80: json_f64_required(&payload, "cold_p80")?,
        warm_p80: json_f64_required(&payload, "warm_p80")?,
        gain_p90: json_f64_required(&payload, "gain_p90")?,
        release_p90: json_f64_required(&payload, "release_p90")?,
        cold_p95: json_f64_required(&payload, "cold_p95")?,
        warm_p95: json_f64_required(&payload, "warm_p95")?,
        release_p95: json_f64_required(&payload, "release_p95")?,
        nonvoid_voxel_count: payload
            .get("nonvoid_voxel_count")
            .and_then(Value::as_u64)
            .ok_or_else(|| anyhow!("voxel thresholds missing nonvoid_voxel_count"))?,
    })
}

fn json_f64_required(payload: &Value, key: &str) -> Result<f64> {
    payload
        .get(key)
        .map(|value| json_f64(value, key))
        .transpose()?
        .ok_or_else(|| anyhow!("voxel thresholds missing {key}"))
}

fn json_string_required(payload: &Value, key: &str) -> Result<String> {
    payload
        .get(key)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .ok_or_else(|| anyhow!("JSON payload missing string {key}"))
}

fn classify_dynamic_voxel(cold: f64, warm: f64, thresholds: &VoxelThresholds) -> DynamicVoxelClass {
    let gain_delta = warm - cold;
    let release_delta = cold - warm;
    if cold <= 0.0 && warm <= 0.0 {
        return DynamicVoxelClass {
            stable_occupied: false,
            thermally_destabilized: false,
            thermally_activated: false,
            thermally_released: false,
            gain_delta,
            release_delta,
        };
    }
    let warm_activation_gate = if thresholds.warm_p80 <= 0.0 {
        warm > 0.0
    } else {
        warm >= thresholds.warm_p80
    };
    DynamicVoxelClass {
        stable_occupied: cold >= thresholds.cold_p80 && warm >= thresholds.warm_p80,
        thermally_destabilized: cold >= thresholds.cold_p80 && warm < thresholds.warm_p80,
        thermally_activated: warm_activation_gate && cold < thresholds.cold_p80,
        thermally_released: release_delta >= thresholds.release_p90,
        gain_delta,
        release_delta,
    }
}

fn classify_signal_row(
    batch: &RecordBatch,
    row: usize,
    variance_class: &str,
    audit: &FieldAudit,
    protocol_residue: Option<&ProtocolResidueContext>,
) -> Result<Option<VarianceClass>> {
    if audit.fallback_enabled {
        if row_matches_activation_fallback(batch, row, audit, protocol_residue)? {
            return Ok(Some(VarianceClass::ThermallyActivated));
        }
        if protocol_residue.is_some_and(|context| context.gain_delta > audit.gain_delta_threshold) {
            return Ok(Some(VarianceClass::ThermallyDestabilized));
        }
        if protocol_residue.is_some_and(|context| context.stable_occupied) {
            return Ok(Some(VarianceClass::StableOccupied));
        }
        return Ok(None);
    }
    if variance_class == "thermally_activated" {
        return Ok(Some(VarianceClass::ThermallyActivated));
    }
    match variance_class {
        "stable_occupied" => Ok(Some(VarianceClass::StableOccupied)),
        "thermally_destabilized" => Ok(Some(VarianceClass::ThermallyDestabilized)),
        "void" => Ok(None),
        _ => Ok(None),
    }
}

fn row_matches_activation_fallback(
    batch: &RecordBatch,
    row: usize,
    audit: &FieldAudit,
    protocol_residue: Option<&ProtocolResidueContext>,
) -> Result<bool> {
    if optional_string_value(batch, "boundary_class", row)?
        .is_some_and(|value| is_activation_label(&value))
    {
        return Ok(true);
    }
    if optional_string_value(batch, "voxel_context", row)?
        .is_some_and(|value| is_activation_label(&value))
    {
        return Ok(true);
    }
    if optional_f64_value(batch, "voxel_activated_fraction", row)?.is_some_and(|value| value > 0.0)
    {
        return Ok(true);
    }
    Ok(protocol_residue
        .is_some_and(|context| context.release_delta > audit.release_delta_threshold))
}

fn is_activation_label(value: &str) -> bool {
    matches!(
        normalized_label(value).as_str(),
        "thermally_activated" | "activated" | "cryptic_bridge"
    )
}

fn normalized_label(value: &str) -> String {
    value
        .trim()
        .to_ascii_lowercase()
        .replace(' ', "_")
        .replace('-', "_")
}

fn count_map_json(map: &BTreeMap<String, u64>) -> String {
    serde_json::to_string(map).unwrap_or_else(|_| "{}".to_owned())
}

fn quantile_nearest(values: &mut [f64], quantile: f64) -> f64 {
    if values.is_empty() {
        return 1.0;
    }
    values.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
    let index = ((values.len() - 1) as f64 * quantile).round() as usize;
    values[index]
}

fn quantile_nearest_optional(values: &mut [f64], quantile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
    let index = ((values.len() - 1) as f64 * quantile).round() as usize;
    Some(values[index])
}

fn load_protocol_context(
    path: &Path,
    condition_id: &str,
    config: &Config,
) -> Result<ProtocolContext> {
    let mut reader = parquet_reader(path)?;
    let mut residues = HashMap::new();
    let mut positive_release = Vec::new();
    let mut positive_gain = Vec::new();
    for batch in &mut reader {
        let batch = batch?;
        for row in 0..batch.num_rows() {
            if string_value(&batch, "condition_id", row)? != condition_id {
                continue;
            }
            let residue_idx = if batch.column_by_name("primary_residue_idx").is_some() {
                i64_value(&batch, "primary_residue_idx", row)?
            } else {
                i64_value(&batch, "residue_idx", row)?
            };
            let mut release_delta = 0.0_f64;
            let mut gain_delta = 0.0_f64;
            let mut stable_occupied = false;
            for group in ["A", "B", "C", "D"] {
                let cold = protocol_phase_value(&batch, row, group, "cold_hold")?;
                let warm = protocol_phase_value(&batch, row, group, "warm_hold")?;
                let release = cold - warm;
                let gain = warm - cold;
                release_delta = release_delta.max(release);
                gain_delta = gain_delta.max(gain);
                stable_occupied |=
                    cold > 0.0 && warm > 0.0 && (cold - warm).abs() <= config.stability_epsilon;
            }
            if release_delta > 0.0 {
                positive_release.push(release_delta);
            }
            if gain_delta > 0.0 {
                positive_gain.push(gain_delta);
            }
            residues.insert(
                residue_idx,
                ProtocolResidueContext {
                    release_delta,
                    gain_delta,
                    stable_occupied,
                },
            );
        }
    }
    let release_p90 =
        quantile_nearest_optional(&mut positive_release, 0.90).unwrap_or(f64::INFINITY);
    let gain_p90 = quantile_nearest_optional(&mut positive_gain, 0.90).unwrap_or(f64::INFINITY);
    Ok(ProtocolContext {
        residues,
        release_delta_threshold: config.release_delta_threshold.unwrap_or(release_p90),
        gain_delta_threshold: config.gain_delta_threshold.unwrap_or(gain_p90),
    })
}

fn protocol_phase_value(batch: &RecordBatch, row: usize, group: &str, phase: &str) -> Result<f64> {
    let with_suffix = format!("{group}_{phase}_spikes");
    if batch.column_by_name(&with_suffix).is_some() {
        return numeric_f64_value(batch, &with_suffix, row);
    }
    let plain = format!("{group}_{phase}");
    numeric_f64_value(batch, &plain, row)
}

fn load_residue_coordinates(config: &Config) -> Result<Vec<ResidueCoordinate>> {
    let payload: Value = serde_json::from_reader(File::open(&config.grid_mapping)?)
        .with_context(|| format!("parse grid mapping {}", config.grid_mapping.display()))?;
    let condition = payload
        .get("conditions")
        .and_then(|conditions| conditions.get(&config.condition_id))
        .ok_or_else(|| {
            anyhow!(
                "condition {} was not present in {}",
                config.condition_id,
                config.grid_mapping.display()
            )
        })?;
    let topology_path = condition
        .get("topology_path")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing topology_path for {}", config.condition_id))?;
    let topology: Value = serde_json::from_reader(File::open(topology_path)?)
        .with_context(|| format!("parse topology {topology_path}"))?;
    let positions = topology
        .get("positions")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("{topology_path} missing positions"))?;
    let ca_indices = topology
        .get("ca_indices")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("{topology_path} missing ca_indices"))?;
    let mut rows = Vec::with_capacity(ca_indices.len());
    for (residue_idx, raw_ca_idx) in ca_indices.iter().enumerate() {
        let ca_idx = raw_ca_idx
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| anyhow!("invalid ca_indices entry in {topology_path}"))?;
        let base = ca_idx
            .checked_mul(3)
            .ok_or_else(|| anyhow!("ca index overflow in {topology_path}"))?;
        if base + 2 >= positions.len() {
            return Err(anyhow!(
                "ca index {ca_idx} out of bounds in {topology_path}"
            ));
        }
        rows.push(ResidueCoordinate {
            residue_idx: i64::try_from(residue_idx)?,
            xyz: [
                json_f64(&positions[base], "ca_x")?,
                json_f64(&positions[base + 1], "ca_y")?,
                json_f64(&positions[base + 2], "ca_z")?,
            ],
        });
    }
    Ok(rows)
}

fn nearest_residue_for_voxel(
    x_idx: i64,
    y_idx: i64,
    z_idx: i64,
    geometry: GridGeometry,
    residues: &[ResidueCoordinate],
) -> Option<i64> {
    let center = [
        f64::from(geometry.origin[0]) + (x_idx as f64 + 0.5) * f64::from(geometry.spacing_a),
        f64::from(geometry.origin[1]) + (y_idx as f64 + 0.5) * f64::from(geometry.spacing_a),
        f64::from(geometry.origin[2]) + (z_idx as f64 + 0.5) * f64::from(geometry.spacing_a),
    ];
    residues
        .iter()
        .min_by(|lhs, rhs| distance_sq(center, lhs.xyz).total_cmp(&distance_sq(center, rhs.xyz)))
        .map(|row| row.residue_idx)
}

fn distance_sq(lhs: [f64; 3], rhs: [f64; 3]) -> f64 {
    let dx = lhs[0] - rhs[0];
    let dy = lhs[1] - rhs[1];
    let dz = lhs[2] - rhs[2];
    dx * dx + dy * dy + dz * dz
}

fn min_distance_to_points(point: [f32; 3], points: &[[f32; 3]]) -> f64 {
    points
        .iter()
        .map(|candidate| {
            let dx = f64::from(point[0] - candidate[0]);
            let dy = f64::from(point[1] - candidate[1]);
            let dz = f64::from(point[2] - candidate[2]);
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .fold(f64::INFINITY, f64::min)
}

fn load_phase_coherence_map(path: &Path, condition_id: &str) -> Result<HashMap<i64, f64>> {
    let mut reader = parquet_reader(path)?;
    let mut coherence = HashMap::new();
    for batch in &mut reader {
        let batch = batch?;
        for row in 0..batch.num_rows() {
            if string_value(&batch, "condition_id", row)? != condition_id {
                continue;
            }
            let residue_idx = i64_value(&batch, "residue_idx", row)?;
            let coherence_score = f64_value(&batch, "coherence_score", row)?.clamp(0.0, 1.0);
            coherence.insert(residue_idx, coherence_score);
        }
    }
    Ok(coherence)
}

fn load_pathway_context(path: &Path, condition_id: &str) -> Result<PathwayContext> {
    let mut reader = parquet_reader(path)?;
    let mut voxel_to_residue = HashMap::new();
    let mut kinetic_burst_residues = HashSet::new();
    for batch in &mut reader {
        let batch = batch?;
        for row in 0..batch.num_rows() {
            if string_value(&batch, "condition_id", row)? != condition_id {
                continue;
            }
            let residue_idx = i64_value(&batch, "residue_idx", row)?;
            if bool_value(&batch, "violent_kinetic_node", row).unwrap_or(false) {
                kinetic_burst_residues.insert(residue_idx);
            }
            let voxel_idx = i64_value(&batch, "voxel_idx", row)?;
            if voxel_idx >= 0 {
                voxel_to_residue.insert(u64::try_from(voxel_idx)?, residue_idx);
            }
        }
    }
    Ok(PathwayContext {
        voxel_to_residue,
        kinetic_burst_residues,
    })
}

fn load_bridge_no_fly_voxels(
    path: &Path,
    condition_id: &str,
    geometry: GridGeometry,
    radius_a: f32,
    bridge_anchors: &[String],
) -> Result<HashSet<u64>> {
    let radius_voxels = (radius_a / geometry.spacing_a).ceil() as i64;
    let radius_sq = radius_voxels * radius_voxels;
    let mut reader = parquet_reader(path)?;
    let mut no_fly = HashSet::new();
    for batch in &mut reader {
        let batch = batch?;
        for row in 0..batch.num_rows() {
            if string_value(&batch, "condition_id", row)? != condition_id {
                continue;
            }
            let residue_name = string_value(&batch, "residue_name", row)?;
            let residue_idx = i64_value(&batch, "residue_idx", row)?;
            if !bridge_anchors.is_empty()
                && !bridge_anchors
                    .iter()
                    .any(|anchor| residue_matches(anchor, &residue_name, residue_idx))
            {
                continue;
            }
            let x0 = i64_value(&batch, "x_idx", row)?;
            let y0 = i64_value(&batch, "y_idx", row)?;
            let z0 = i64_value(&batch, "z_idx", row)?;
            let center = [
                geometry.origin[0] + (x0 as f32 + 0.5) * geometry.spacing_a,
                geometry.origin[1] + (y0 as f32 + 0.5) * geometry.spacing_a,
                geometry.origin[2] + (z0 as f32 + 0.5) * geometry.spacing_a,
            ];
            let Some(center_voxel) = coordinate_to_voxel(center, geometry) else {
                continue;
            };
            let center_z = i64::try_from(center_voxel / (geometry.nx as u64 * geometry.ny as u64))?;
            let center_y = i64::try_from(
                (center_voxel % (geometry.nx as u64 * geometry.ny as u64)) / geometry.nx as u64,
            )?;
            let center_x = i64::try_from(center_voxel % geometry.nx as u64)?;
            for dz in -radius_voxels..=radius_voxels {
                for dy in -radius_voxels..=radius_voxels {
                    for dx in -radius_voxels..=radius_voxels {
                        if dx * dx + dy * dy + dz * dz > radius_sq {
                            continue;
                        }
                        let x = center_x + dx;
                        let y = center_y + dy;
                        let z = center_z + dz;
                        if (0..geometry.nx).contains(&x)
                            && (0..geometry.ny).contains(&y)
                            && (0..geometry.nz).contains(&z)
                        {
                            no_fly.insert(
                                (z * geometry.nx * geometry.ny + y * geometry.nx + x) as u64,
                            );
                        }
                    }
                }
            }
        }
    }
    Ok(no_fly)
}

fn residue_matches(anchor: &str, residue_name: &str, residue_idx: i64) -> bool {
    let anchor = anchor.trim();
    if anchor == residue_name || anchor == residue_idx.to_string() {
        return true;
    }
    if let Some(prefix) = residue_name.get(0..3) {
        return anchor == format!("{prefix}{residue_idx}");
    }
    false
}

fn load_grid_geometry(config: &Config) -> Result<GridGeometry> {
    if config.grid_mapping.exists() {
        let payload: Value = serde_json::from_reader(File::open(&config.grid_mapping)?)
            .with_context(|| format!("parse grid mapping {}", config.grid_mapping.display()))?;
        let condition = payload
            .get("conditions")
            .and_then(|conditions| conditions.get(&config.condition_id))
            .ok_or_else(|| {
                anyhow!(
                    "condition {} was not present in {}",
                    config.condition_id,
                    config.grid_mapping.display()
                )
            })?;
        let origin_value = condition
            .get("origin_xyz_angstrom")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow!("missing origin_xyz_angstrom for {}", config.condition_id))?;
        if origin_value.len() != 3 {
            return Err(anyhow!(
                "origin_xyz_angstrom for {} must contain exactly three coordinates",
                config.condition_id
            ));
        }
        let origin = [
            json_f32(&origin_value[0], "origin_x")?,
            json_f32(&origin_value[1], "origin_y")?,
            json_f32(&origin_value[2], "origin_z")?,
        ];
        let grid_dim = condition
            .get("grid_dim")
            .and_then(Value::as_i64)
            .unwrap_or(config.grid_dim);
        let nx = condition
            .get("nx")
            .and_then(Value::as_i64)
            .unwrap_or(grid_dim);
        let ny = condition
            .get("ny")
            .and_then(Value::as_i64)
            .unwrap_or(grid_dim);
        let nz = condition
            .get("nz")
            .and_then(Value::as_i64)
            .unwrap_or(grid_dim);
        let spacing_a = condition
            .get("spacing_angstrom")
            .map(|value| json_f32(value, "spacing_angstrom"))
            .transpose()?
            .unwrap_or(config.grid_spacing_a);
        return validate_grid_geometry(GridGeometry {
            nx,
            ny,
            nz,
            spacing_a,
            origin,
        });
    }
    validate_grid_geometry(GridGeometry {
        nx: config.grid_dim,
        ny: config.grid_dim,
        nz: config.grid_dim,
        spacing_a: config.grid_spacing_a,
        origin: config.origin,
    })
}

fn json_f32(value: &Value, label: &str) -> Result<f32> {
    let number = value
        .as_f64()
        .ok_or_else(|| anyhow!("{label} was not a JSON number"))?;
    if !number.is_finite() {
        return Err(anyhow!("{label} was not finite"));
    }
    Ok(number as f32)
}

fn json_f64(value: &Value, label: &str) -> Result<f64> {
    let number = value
        .as_f64()
        .ok_or_else(|| anyhow!("{label} was not a JSON number"))?;
    if !number.is_finite() {
        return Err(anyhow!("{label} was not finite"));
    }
    Ok(number)
}

fn validate_grid_geometry(geometry: GridGeometry) -> Result<GridGeometry> {
    if geometry.nx <= 0 || geometry.ny <= 0 || geometry.nz <= 0 {
        return Err(anyhow!(
            "grid dimensions must be positive, got {}x{}x{}",
            geometry.nx,
            geometry.ny,
            geometry.nz
        ));
    }
    if !geometry.spacing_a.is_finite() || geometry.spacing_a <= 0.0 {
        return Err(anyhow!("grid spacing must be positive and finite"));
    }
    if geometry.origin.iter().any(|value| !value.is_finite()) {
        return Err(anyhow!("grid origin must be finite"));
    }
    Ok(geometry)
}

fn load_scaffold_template(config: &Config) -> Result<ScaffoldTemplate> {
    let mol = load_v2000_sdf(&config.scaffold_sdf)
        .with_context(|| format!("load scaffold SDF {}", config.scaffold_sdf.display()))?;
    let fragment_atoms =
        load_fragment_atom_indices(&config.fragment_registry, &config.removed_fragment_id)
            .with_context(|| {
                format!(
                    "load {} from {}",
                    config.removed_fragment_id,
                    config.fragment_registry.display()
                )
            })?;
    let removed_atoms = removed_fragment_atom_set(&mol, &fragment_atoms);
    let mut index_map = HashMap::new();
    let mut coordinates = Vec::new();
    let mut charges = Vec::new();
    for atom_idx in 0..mol.coordinates.len() {
        if removed_atoms.contains(&atom_idx) {
            continue;
        }
        index_map.insert(atom_idx, charges.len());
        coordinates.extend_from_slice(&mol.coordinates[atom_idx]);
        charges.push(0.0);
    }

    let mut anchors = Vec::new();
    let mut attachment_points = Vec::new();
    for (lhs, rhs) in &mol.bonds {
        let lhs_fragment = fragment_atoms.contains(lhs);
        let rhs_fragment = fragment_atoms.contains(rhs);
        if lhs_fragment == rhs_fragment {
            continue;
        }
        let removed_fragment_atom_idx = if lhs_fragment { *lhs } else { *rhs };
        let original_scaffold_atom_idx = if lhs_fragment { *rhs } else { *lhs };
        if !is_heavy_atom(&mol, removed_fragment_atom_idx)
            || !is_heavy_atom(&mol, original_scaffold_atom_idx)
            || removed_atoms.contains(&original_scaffold_atom_idx)
        {
            continue;
        }
        let scaffold_atom_idx = *index_map.get(&original_scaffold_atom_idx).ok_or_else(|| {
            anyhow!("scaffold atom {original_scaffold_atom_idx} was not retained")
        })?;
        let (scaffold_ref1_atom_idx, scaffold_ref2_atom_idx) =
            scaffold_reference_atoms(&mol, original_scaffold_atom_idx, &removed_atoms, &index_map)?;
        let exit_vector = normalize_vec3(
            sub_vec3(
                mol.coordinates[removed_fragment_atom_idx],
                mol.coordinates[original_scaffold_atom_idx],
            ),
            "scaffold exit vector",
        )?;
        attachment_points.push(AttachmentPoint::new(
            scaffold_atom_idx,
            None,
            exit_vector,
            None,
        ));
        anchors.push(ScaffoldAnchor {
            original_scaffold_atom_idx,
            removed_fragment_atom_idx,
            scaffold_ref1_atom_idx,
            scaffold_ref2_atom_idx,
        });
    }
    if anchors.is_empty() {
        return Err(anyhow!(
            "no heavy-atom boundary bonds found for removed fragment {}",
            config.removed_fragment_id
        ));
    }
    let retained_bonds = mol
        .bonds
        .iter()
        .filter_map(|(lhs, rhs)| {
            let new_lhs = index_map.get(lhs).copied()?;
            let new_rhs = index_map.get(rhs).copied()?;
            (new_lhs != new_rhs).then_some((new_lhs, new_rhs))
        })
        .collect();
    let state =
        ScaffoldState3D::new_with_bonds(coordinates, charges, attachment_points, retained_bonds)?;
    Ok(ScaffoldTemplate { state, anchors })
}

fn scaffold_reference_atoms(
    mol: &SdfMol,
    original_scaffold_atom_idx: usize,
    removed_atoms: &HashSet<usize>,
    index_map: &HashMap<usize, usize>,
) -> Result<(usize, usize)> {
    let mut original_refs = Vec::new();
    for (lhs, rhs) in &mol.bonds {
        let neighbor = if *lhs == original_scaffold_atom_idx {
            *rhs
        } else if *rhs == original_scaffold_atom_idx {
            *lhs
        } else {
            continue;
        };
        if removed_atoms.contains(&neighbor) || !is_heavy_atom(mol, neighbor) {
            continue;
        }
        if index_map.contains_key(&neighbor) {
            original_refs.push(neighbor);
        }
    }
    if original_refs.len() < 2 {
        let mut fallback: Vec<(f32, usize)> = index_map
            .keys()
            .copied()
            .filter(|atom_idx| {
                *atom_idx != original_scaffold_atom_idx
                    && !removed_atoms.contains(atom_idx)
                    && is_heavy_atom(mol, *atom_idx)
                    && !original_refs.contains(atom_idx)
            })
            .map(|atom_idx| {
                (
                    distance3(
                        mol.coordinates[original_scaffold_atom_idx],
                        mol.coordinates[atom_idx],
                    ),
                    atom_idx,
                )
            })
            .collect();
        fallback.sort_by(|lhs, rhs| lhs.0.total_cmp(&rhs.0));
        for (_distance, atom_idx) in fallback {
            original_refs.push(atom_idx);
            if original_refs.len() >= 2 {
                break;
            }
        }
    }
    if original_refs.len() < 2 {
        return Err(anyhow!(
            "could not find two retained scaffold reference atoms for atom {original_scaffold_atom_idx}"
        ));
    }
    Ok((
        *index_map
            .get(&original_refs[0])
            .ok_or_else(|| anyhow!("reference atom {} was not retained", original_refs[0]))?,
        *index_map
            .get(&original_refs[1])
            .ok_or_else(|| anyhow!("reference atom {} was not retained", original_refs[1]))?,
    ))
}

fn distance3(lhs: [f32; 3], rhs: [f32; 3]) -> f32 {
    let dx = lhs[0] - rhs[0];
    let dy = lhs[1] - rhs[1];
    let dz = lhs[2] - rhs[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn load_v2000_sdf(path: &Path) -> Result<SdfMol> {
    let mut text = String::new();
    File::open(path)?.read_to_string(&mut text)?;
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() < 4 {
        return Err(anyhow!("{} is too short to be an SDF", path.display()));
    }
    let counts = lines[3];
    let atom_count = counts
        .get(0..3)
        .ok_or_else(|| anyhow!("missing V2000 atom count"))?
        .trim()
        .parse::<usize>()?;
    let bond_count = counts
        .get(3..6)
        .ok_or_else(|| anyhow!("missing V2000 bond count"))?
        .trim()
        .parse::<usize>()?;
    let mut coordinates = Vec::with_capacity(atom_count);
    let mut elements = Vec::with_capacity(atom_count);
    for atom_line in lines.iter().skip(4).take(atom_count) {
        let fields: Vec<&str> = atom_line.split_whitespace().collect();
        if fields.len() < 4 {
            return Err(anyhow!("invalid V2000 atom line: {atom_line}"));
        }
        coordinates.push([
            fields[0].parse::<f32>()?,
            fields[1].parse::<f32>()?,
            fields[2].parse::<f32>()?,
        ]);
        elements.push(fields[3].to_owned());
    }
    let mut bonds = Vec::with_capacity(bond_count);
    for bond_line in lines.iter().skip(4 + atom_count).take(bond_count) {
        let lhs = bond_line
            .get(0..3)
            .ok_or_else(|| anyhow!("invalid V2000 bond line: {bond_line}"))?
            .trim()
            .parse::<usize>()?;
        let rhs = bond_line
            .get(3..6)
            .ok_or_else(|| anyhow!("invalid V2000 bond line: {bond_line}"))?
            .trim()
            .parse::<usize>()?;
        bonds.push((lhs - 1, rhs - 1));
    }
    Ok(SdfMol {
        coordinates,
        elements,
        bonds,
    })
}

fn ligand_grid_diagnostics(path: &Path, geometry: GridGeometry) -> Result<LigandGridDiagnostics> {
    let mol = load_v2000_sdf(path)?;
    let heavy: Vec<[f32; 3]> = mol
        .coordinates
        .iter()
        .zip(&mol.elements)
        .filter_map(|(xyz, element)| is_heavy_element(element).then_some(*xyz))
        .collect();
    if heavy.is_empty() {
        return Err(anyhow!("{} contained no heavy atoms", path.display()));
    }
    let mut centroid = [0.0_f64; 3];
    let grid_min = grid_min_xyz(geometry);
    let grid_max = grid_max_xyz(geometry);
    let mut inside_grid = true;
    let mut min_distance = f64::INFINITY;
    let mut max_distance = f64::NEG_INFINITY;
    for xyz in &heavy {
        for axis in 0..3 {
            centroid[axis] += f64::from(xyz[axis]);
            if f64::from(xyz[axis]) < grid_min[axis] || f64::from(xyz[axis]) > grid_max[axis] {
                inside_grid = false;
            }
        }
        let boundary_distance = distance_to_grid_boundary(*xyz, geometry);
        min_distance = min_distance.min(boundary_distance);
        max_distance = max_distance.max(boundary_distance);
    }
    for value in &mut centroid {
        *value /= heavy.len() as f64;
    }
    Ok(LigandGridDiagnostics {
        centroid,
        inside_grid,
        min_distance_to_grid_a: min_distance,
        max_distance_to_grid_a: max_distance,
    })
}

fn print_ligand_grid_diagnostics(
    ligand_sdf_path: &Path,
    geometry: GridGeometry,
    diagnostics: &LigandGridDiagnostics,
) {
    let grid_min = grid_min_xyz(geometry);
    let grid_max = grid_max_xyz(geometry);
    println!(
        "ligand_grid_diagnostic ligand_sdf_path={} ligand_centroid_xyz=[{:.6},{:.6},{:.6}] grid_min_xyz=[{:.6},{:.6},{:.6}] grid_max_xyz=[{:.6},{:.6},{:.6}] ligand_inside_grid={} ligand_min_distance_to_grid_A={:.6} ligand_max_distance_to_grid_A={:.6}",
        ligand_sdf_path.display(),
        diagnostics.centroid[0],
        diagnostics.centroid[1],
        diagnostics.centroid[2],
        grid_min[0],
        grid_min[1],
        grid_min[2],
        grid_max[0],
        grid_max[1],
        grid_max[2],
        diagnostics.inside_grid,
        diagnostics.min_distance_to_grid_a,
        diagnostics.max_distance_to_grid_a,
    );
}

fn is_heavy_element(element: &str) -> bool {
    let normalized = element.trim().to_ascii_uppercase();
    !matches!(normalized.as_str(), "H" | "D" | "*")
}

fn grid_min_xyz(geometry: GridGeometry) -> [f64; 3] {
    [
        f64::from(geometry.origin[0]),
        f64::from(geometry.origin[1]),
        f64::from(geometry.origin[2]),
    ]
}

fn grid_max_xyz(geometry: GridGeometry) -> [f64; 3] {
    [
        f64::from(geometry.origin[0]) + geometry.nx as f64 * f64::from(geometry.spacing_a),
        f64::from(geometry.origin[1]) + geometry.ny as f64 * f64::from(geometry.spacing_a),
        f64::from(geometry.origin[2]) + geometry.nz as f64 * f64::from(geometry.spacing_a),
    ]
}

fn distance_to_grid_boundary(xyz: [f32; 3], geometry: GridGeometry) -> f64 {
    let grid_min = grid_min_xyz(geometry);
    let grid_max = grid_max_xyz(geometry);
    let mut distances = [0.0_f64; 6];
    for axis in 0..3 {
        let value = f64::from(xyz[axis]);
        distances[2 * axis] = value - grid_min[axis];
        distances[2 * axis + 1] = grid_max[axis] - value;
    }
    distances.into_iter().fold(f64::INFINITY, f64::min)
}

fn load_fragment_atom_indices(path: &Path, fragment_id: &str) -> Result<HashSet<usize>> {
    let payload: Value = serde_json::from_reader(File::open(path)?)?;
    let fragments = payload
        .get("fragments")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("fragment registry has no fragments array"))?;
    for fragment in fragments {
        if fragment.get("fragment_id").and_then(Value::as_str) != Some(fragment_id) {
            continue;
        }
        let indices = fragment
            .get("parent_atom_indices")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow!("{fragment_id} has no parent_atom_indices"))?;
        return indices
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|item| usize::try_from(item).ok())
                    .ok_or_else(|| anyhow!("invalid parent atom index in {fragment_id}"))
            })
            .collect();
    }
    Err(anyhow!(
        "{fragment_id} was not present in fragment registry"
    ))
}

fn removed_fragment_atom_set(mol: &SdfMol, fragment_atoms: &HashSet<usize>) -> HashSet<usize> {
    let mut removed = fragment_atoms.clone();
    for (lhs, rhs) in &mol.bonds {
        let lhs_fragment = fragment_atoms.contains(lhs);
        let rhs_fragment = fragment_atoms.contains(rhs);
        if lhs_fragment && mol.elements[*rhs] == "H" {
            removed.insert(*rhs);
        }
        if rhs_fragment && mol.elements[*lhs] == "H" {
            removed.insert(*lhs);
        }
    }
    removed
}

fn is_heavy_atom(mol: &SdfMol, atom_idx: usize) -> bool {
    mol.elements
        .get(atom_idx)
        .is_some_and(|element| element != "H")
}

fn sub_vec3(lhs: [f32; 3], rhs: [f32; 3]) -> [f32; 3] {
    [lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]]
}

fn add_f64(lhs: [f64; 3], rhs: [f64; 3]) -> [f64; 3] {
    [lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]]
}

fn mul_f64(value: [f64; 3], scale: f64) -> [f64; 3] {
    [value[0] * scale, value[1] * scale, value[2] * scale]
}

fn dot_f64(lhs: [f64; 3], rhs: [f64; 3]) -> f64 {
    lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]
}

fn cross_f64(lhs: [f64; 3], rhs: [f64; 3]) -> [f64; 3] {
    [
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0],
    ]
}

fn normalize_f64(value: [f64; 3]) -> [f64; 3] {
    let norm = dot_f64(value, value).sqrt();
    if !norm.is_finite() || norm <= 1.0e-12 {
        [0.0, 0.0, 0.0]
    } else {
        mul_f64(value, 1.0 / norm)
    }
}

fn normalize_vec3(value: [f32; 3], label: &'static str) -> Result<[f32; 3]> {
    let norm = (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt();
    if !norm.is_finite() || norm <= 1.0e-6 {
        return Err(anyhow!("{label} is zero length"));
    }
    Ok([value[0] / norm, value[1] / norm, value[2] / norm])
}

fn parquet_reader(path: &Path) -> Result<parquet::arrow::arrow_reader::ParquetRecordBatchReader> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("read parquet metadata {}", path.display()))?
        .build()
        .with_context(|| format!("build parquet reader {}", path.display()))
}

fn string_value(batch: &RecordBatch, column: &str, row: usize) -> Result<String> {
    let array = batch
        .column_by_name(column)
        .ok_or_else(|| anyhow!("missing column {column}"))?;
    if let Some(values) = array.as_any().downcast_ref::<StringArray>() {
        return Ok(values.value(row).to_owned());
    }
    if let Some(values) = array.as_any().downcast_ref::<LargeStringArray>() {
        return Ok(values.value(row).to_owned());
    }
    Err(anyhow!("column {column} was not a string array"))
}

fn optional_string_value(batch: &RecordBatch, column: &str, row: usize) -> Result<Option<String>> {
    if batch.column_by_name(column).is_none() {
        return Ok(None);
    }
    string_value(batch, column, row).map(Some)
}

fn optional_variance_class_value(batch: &RecordBatch, row: usize) -> Result<Option<String>> {
    if batch.column_by_name("variance_classification").is_some()
        || batch.column_by_name("variance_class").is_some()
    {
        return variance_class_value(batch, row).map(Some);
    }
    Ok(None)
}

fn variance_class_value(batch: &RecordBatch, row: usize) -> Result<String> {
    if batch.column_by_name("variance_classification").is_some() {
        return string_value(batch, "variance_classification", row);
    }
    string_value(batch, "variance_class", row)
}

fn signal_row_residue_idx(batch: &RecordBatch, row: usize) -> Result<Option<i64>> {
    if batch.column_by_name("primary_residue_idx").is_some() {
        return Ok(Some(i64_value(batch, "primary_residue_idx", row)?));
    }
    if batch.column_by_name("residue_idx").is_some() {
        return Ok(Some(i64_value(batch, "residue_idx", row)?));
    }
    Ok(None)
}

fn u64_value(batch: &RecordBatch, column: &str, row: usize) -> Result<u64> {
    let array = batch
        .column_by_name(column)
        .ok_or_else(|| anyhow!("missing column {column}"))?;
    if let Some(values) = array.as_any().downcast_ref::<UInt64Array>() {
        return Ok(values.value(row));
    }
    if let Some(values) = array.as_any().downcast_ref::<UInt32Array>() {
        return Ok(u64::from(values.value(row)));
    }
    if let Some(values) = array.as_any().downcast_ref::<Int64Array>() {
        return Ok(u64::try_from(values.value(row))?);
    }
    Err(anyhow!("column {column} was not a u64-compatible array"))
}

fn i64_value(batch: &RecordBatch, column: &str, row: usize) -> Result<i64> {
    let array = batch
        .column_by_name(column)
        .ok_or_else(|| anyhow!("missing column {column}"))?;
    if let Some(values) = array.as_any().downcast_ref::<Int64Array>() {
        return Ok(values.value(row));
    }
    if let Some(values) = array.as_any().downcast_ref::<Int32Array>() {
        return Ok(i64::from(values.value(row)));
    }
    Err(anyhow!("column {column} was not an i64-compatible array"))
}

fn bool_value(batch: &RecordBatch, column: &str, row: usize) -> Result<bool> {
    let array = batch
        .column_by_name(column)
        .ok_or_else(|| anyhow!("missing column {column}"))?;
    if let Some(values) = array.as_any().downcast_ref::<BooleanArray>() {
        return Ok(values.value(row));
    }
    Err(anyhow!("column {column} was not a boolean array"))
}

fn optional_f64_value(batch: &RecordBatch, column: &str, row: usize) -> Result<Option<f64>> {
    if batch.column_by_name(column).is_none() {
        return Ok(None);
    }
    f64_value(batch, column, row).map(Some)
}

fn f64_value(batch: &RecordBatch, column: &str, row: usize) -> Result<f64> {
    let array = batch
        .column_by_name(column)
        .ok_or_else(|| anyhow!("missing column {column}"))?;
    if let Some(values) = array.as_any().downcast_ref::<Float64Array>() {
        return Ok(values.value(row));
    }
    Err(anyhow!("column {column} was not a f64 array"))
}

fn numeric_f64_value(batch: &RecordBatch, column: &str, row: usize) -> Result<f64> {
    let array = batch
        .column_by_name(column)
        .ok_or_else(|| anyhow!("missing column {column}"))?;
    if let Some(values) = array.as_any().downcast_ref::<Float64Array>() {
        return Ok(values.value(row));
    }
    if let Some(values) = array.as_any().downcast_ref::<UInt64Array>() {
        return Ok(values.value(row) as f64);
    }
    if let Some(values) = array.as_any().downcast_ref::<UInt32Array>() {
        return Ok(f64::from(values.value(row)));
    }
    if let Some(values) = array.as_any().downcast_ref::<Int64Array>() {
        return Ok(values.value(row) as f64);
    }
    if let Some(values) = array.as_any().downcast_ref::<Int32Array>() {
        return Ok(f64::from(values.value(row)));
    }
    Err(anyhow!("column {column} was not numeric"))
}

fn coordinates_json(coordinates: &[f32]) -> String {
    let rows: Vec<[f32; 3]> = coordinates
        .chunks_exact(3)
        .map(|xyz| [xyz[0], xyz[1], xyz[2]])
        .collect();
    serde_json::to_string(&rows).expect("coordinate serialization cannot fail")
}

fn write_survivors(path: &Path, survivors: &[Survivor]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("anchor_id", DataType::Utf8, false),
        Field::new("canonical_smiles", DataType::Utf8, false),
        Field::new("product_id", DataType::Utf8, false),
        Field::new("smiles", DataType::Utf8, false),
        Field::new("synthon_a_id", DataType::Utf8, false),
        Field::new("synthon_b_id", DataType::Utf8, false),
        Field::new("score", DataType::Float64, false),
        Field::new("fragment_pi_complement", DataType::Float64, false),
        Field::new("fragment_pi_clash_adjusted", DataType::Float64, false),
        Field::new("pi_complement", DataType::Float64, false),
        Field::new("pi_clash", DataType::Float64, false),
        Field::new("cryptic_bonus", DataType::Float64, false),
        Field::new("cryptic_bonus_atoms", DataType::UInt64, false),
        Field::new("survival_tier", DataType::Utf8, false),
        Field::new("selected_dihedral_deg", DataType::Float64, false),
        Field::new("assembly_mode", DataType::Utf8, false),
        Field::new("z_matrix_active", DataType::Boolean, false),
        Field::new("rotamers_evaluated", DataType::UInt64, false),
        Field::new("best_rotamer_rank", DataType::UInt64, false),
        Field::new("ligand_sdf_path", DataType::Utf8, false),
        Field::new("pose_reconciliation_method", DataType::Utf8, false),
        Field::new("coordinates_json", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(
                survivors
                    .iter()
                    .map(|row| row.anchor_id.clone())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                survivors
                    .iter()
                    .map(|row| row.canonical_smiles.clone())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                survivors
                    .iter()
                    .map(|row| row.product_id.clone())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                survivors
                    .iter()
                    .map(|row| row.smiles.clone())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                survivors
                    .iter()
                    .map(|row| row.synthon_a_id.clone())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                survivors
                    .iter()
                    .map(|row| row.synthon_b_id.clone())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Float64Array::from(
                survivors.iter().map(|row| row.score).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Float64Array::from(
                survivors
                    .iter()
                    .map(|row| row.fragment_pi_complement)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Float64Array::from(
                survivors
                    .iter()
                    .map(|row| row.fragment_pi_clash_adjusted)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Float64Array::from(
                survivors
                    .iter()
                    .map(|row| row.pi_complement)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Float64Array::from(
                survivors.iter().map(|row| row.pi_clash).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Float64Array::from(
                survivors
                    .iter()
                    .map(|row| row.cryptic_bonus)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(UInt64Array::from(
                survivors
                    .iter()
                    .map(|row| row.cryptic_bonus_atoms)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                survivors
                    .iter()
                    .map(|row| row.survival_tier)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Float64Array::from(
                survivors
                    .iter()
                    .map(|row| row.selected_dihedral_deg)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                survivors
                    .iter()
                    .map(|row| row.assembly_mode)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(BooleanArray::from(
                survivors
                    .iter()
                    .map(|row| row.z_matrix_active)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(UInt64Array::from(
                survivors
                    .iter()
                    .map(|row| row.rotamers_evaluated)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(UInt64Array::from(
                survivors
                    .iter()
                    .map(|row| row.best_rotamer_rank)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                survivors
                    .iter()
                    .map(|row| row.ligand_sdf_path.clone())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                survivors
                    .iter()
                    .map(|row| row.pose_reconciliation_method.clone())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                survivors
                    .iter()
                    .map(|row| row.coordinates_json.clone())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
        ],
    )?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();
    let file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn write_survivors_streaming(
    path: &Path,
    rx: mpsc::Receiver<Survivor>,
    row_group_size: usize,
) -> Result<usize> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let schema = survivor_schema();
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();
    let tmp_path = path.with_extension(format!(
        "{}.tmp",
        path.extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or("parquet")
    ));
    let file = File::create(&tmp_path).with_context(|| format!("create {}", tmp_path.display()))?;
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;
    let mut buffer = Vec::with_capacity(row_group_size);
    let mut total_rows = 0_usize;
    let mut row_group_idx = 0_usize;
    for survivor in rx {
        buffer.push(survivor);
        if buffer.len() >= row_group_size {
            flush_survivor_row_group(
                &mut writer,
                schema.clone(),
                &mut buffer,
                path,
                &mut row_group_idx,
                &mut total_rows,
            )?;
        }
    }
    if !buffer.is_empty() {
        flush_survivor_row_group(
            &mut writer,
            schema,
            &mut buffer,
            path,
            &mut row_group_idx,
            &mut total_rows,
        )?;
    }
    writer.close()?;
    std::fs::rename(&tmp_path, path)
        .with_context(|| format!("atomic rename {} -> {}", tmp_path.display(), path.display()))?;
    Ok(total_rows)
}

fn flush_survivor_row_group(
    writer: &mut ArrowWriter<File>,
    schema: Arc<Schema>,
    buffer: &mut Vec<Survivor>,
    path: &Path,
    row_group_idx: &mut usize,
    total_rows: &mut usize,
) -> Result<()> {
    let rows = buffer.len();
    let batch = survivor_record_batch(schema, buffer)?;
    writer.write(&batch)?;
    *total_rows += rows;
    *row_group_idx += 1;
    println!(
        "mpsc_stream_flush output={} row_group={} rows={} total_rows={}",
        path.display(),
        *row_group_idx,
        rows,
        *total_rows
    );
    buffer.clear();
    Ok(())
}

fn survivor_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("anchor_id", DataType::Utf8, false),
        Field::new("canonical_smiles", DataType::Utf8, false),
        Field::new("product_id", DataType::Utf8, false),
        Field::new("smiles", DataType::Utf8, false),
        Field::new("synthon_a_id", DataType::Utf8, false),
        Field::new("synthon_b_id", DataType::Utf8, false),
        Field::new("score", DataType::Float64, false),
        Field::new("fragment_pi_complement", DataType::Float64, false),
        Field::new("fragment_pi_clash_adjusted", DataType::Float64, false),
        Field::new("pi_complement", DataType::Float64, false),
        Field::new("pi_clash", DataType::Float64, false),
        Field::new("cryptic_bonus", DataType::Float64, false),
        Field::new("cryptic_bonus_atoms", DataType::UInt64, false),
        Field::new("survival_tier", DataType::Utf8, false),
        Field::new("selected_dihedral_deg", DataType::Float64, false),
        Field::new("assembly_mode", DataType::Utf8, false),
        Field::new("z_matrix_active", DataType::Boolean, false),
        Field::new("rotamers_evaluated", DataType::UInt64, false),
        Field::new("best_rotamer_rank", DataType::UInt64, false),
        Field::new("ligand_sdf_path", DataType::Utf8, false),
        Field::new("pose_reconciliation_method", DataType::Utf8, false),
        Field::new("coordinates_json", DataType::Utf8, false),
    ]))
}

fn survivor_record_batch(schema: Arc<Schema>, survivors: &[Survivor]) -> Result<RecordBatch> {
    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(
                survivors
                    .iter()
                    .map(|row| row.anchor_id.clone())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                survivors
                    .iter()
                    .map(|row| row.canonical_smiles.clone())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                survivors
                    .iter()
                    .map(|row| row.product_id.clone())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                survivors
                    .iter()
                    .map(|row| row.smiles.clone())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                survivors
                    .iter()
                    .map(|row| row.synthon_a_id.clone())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                survivors
                    .iter()
                    .map(|row| row.synthon_b_id.clone())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Float64Array::from(
                survivors.iter().map(|row| row.score).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Float64Array::from(
                survivors
                    .iter()
                    .map(|row| row.fragment_pi_complement)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Float64Array::from(
                survivors
                    .iter()
                    .map(|row| row.fragment_pi_clash_adjusted)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Float64Array::from(
                survivors
                    .iter()
                    .map(|row| row.pi_complement)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Float64Array::from(
                survivors.iter().map(|row| row.pi_clash).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Float64Array::from(
                survivors
                    .iter()
                    .map(|row| row.cryptic_bonus)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(UInt64Array::from(
                survivors
                    .iter()
                    .map(|row| row.cryptic_bonus_atoms)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                survivors
                    .iter()
                    .map(|row| row.survival_tier)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Float64Array::from(
                survivors
                    .iter()
                    .map(|row| row.selected_dihedral_deg)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                survivors
                    .iter()
                    .map(|row| row.assembly_mode)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(BooleanArray::from(
                survivors
                    .iter()
                    .map(|row| row.z_matrix_active)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(UInt64Array::from(
                survivors
                    .iter()
                    .map(|row| row.rotamers_evaluated)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(UInt64Array::from(
                survivors
                    .iter()
                    .map(|row| row.best_rotamer_rank)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                survivors
                    .iter()
                    .map(|row| row.ligand_sdf_path.clone())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                survivors
                    .iter()
                    .map(|row| row.pose_reconciliation_method.clone())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                survivors
                    .iter()
                    .map(|row| row.coordinates_json.clone())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
        ],
    )?)
}

#[allow(clippy::too_many_arguments)]
fn write_telemetry_json(
    path: &Path,
    telemetry: &PruneTelemetry,
    config: &Config,
    selected_pose: &SelectedLigandPose,
    ligand_grid: &LigandGridDiagnostics,
    geometry: GridGeometry,
    field_counts: FieldCounts,
    no_fly_voxels: usize,
    survivors: usize,
    real_anchor_count_loaded: usize,
    mock_anchor_count: usize,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let payload = json!({
        "selected_ligand_sdf": selected_pose.selected_ligand_sdf,
        "selected_pose_method": selected_pose.selected_pose_method,
        "o3a_score": selected_pose.o3a_score,
        "guard_level_passed": selected_pose.guard_level_passed,
        "ligand_inside_grid": ligand_grid.inside_grid,
        "ligand_centroid_xyz": ligand_grid.centroid,
        "grid_min_xyz": grid_min_xyz(geometry),
        "grid_max_xyz": grid_max_xyz(geometry),
        "assembly_mode": config.assembly_mode.label(),
        "full_scale": config.full_scale,
        "shard_index": config.shard_index,
        "shard_count": config.shard_count,
        "row_group_size": config.row_group_size,
        "dihedral_samples": config.dihedral_samples,
        "dihedral_grid_deg": config.dihedral_grid_deg,
        "real_anchor_count_loaded": real_anchor_count_loaded,
        "mock_anchor_count": mock_anchor_count,
        "unique_anchor_count_attempted": real_anchor_count_loaded.min(telemetry.attempted),
        "attempted_pairs": telemetry.attempted,
        "attempted": telemetry.attempted,
        "rotamers_evaluated": telemetry.rotamers_evaluated,
        "rotamers_dropped_bounds": telemetry.rotamers_dropped_bounds,
        "rotamers_dropped_no_fly": telemetry.rotamers_dropped_no_fly,
        "rotamers_dropped_hard_clash": telemetry.rotamers_dropped_hard_clash,
        "candidates_with_at_least_one_valid_rotamer": telemetry.candidates_with_at_least_one_valid_rotamer,
        "candidate_survivors": survivors,
        "dropped_assembly": telemetry.dropped_assembly,
        "dropped_bounds": telemetry.dropped_bounds,
        "dropped_no_fly": telemetry.dropped_no_fly,
        "dropped_hard_clash": telemetry.dropped_hard_clash,
        "dropped_insufficient_complement": telemetry.dropped_insufficient_complement,
        "survived_normal": telemetry.survived_normal,
        "survived_cryptic_rescue": telemetry.survived_cryptic_rescue,
        "cryptic_bonus_atoms": telemetry.cryptic_bonus_atoms,
        "stable_occupied": field_counts.stable_occupied,
        "thermally_destabilized": field_counts.thermally_destabilized,
        "thermally_activated": field_counts.thermally_activated,
        "thermally_released": field_counts.thermally_released,
        "complement_voxels": field_counts.complement_voxels,
        "cryptic_bonus_voxels": field_counts.cryptic_bonus_voxels,
        "no_fly_voxels": no_fly_voxels,
        "mean_best_dihedral_deg": finite_json_number(telemetry.mean_best_dihedral_deg),
        "top_dihedral_bins": telemetry.top_dihedral_bins,
        "z_matrix_active_count": telemetry.z_matrix_active_count,
        "rigid_fallback_count": telemetry.rigid_fallback_count,
        "mean_min_atom_to_complement_A": finite_json_number(telemetry.mean_min_atom_to_complement_a),
        "p10_min_atom_to_complement_A": finite_json_number(telemetry.p10_min_atom_to_complement_a),
        "min_seen_atom_to_complement_A": finite_json_number(telemetry.min_seen_atom_to_complement_a),
        "mean_fragment_pi_complement": finite_json_number(telemetry.mean_fragment_pi_complement),
        "mean_fragment_pi_clash_adjusted": finite_json_number(telemetry.mean_fragment_pi_clash_adjusted),
        "mean_fragment_cryptic_bonus": finite_json_number(telemetry.mean_fragment_cryptic_bonus),
        "candidates_with_fragment_complement": telemetry.candidates_with_fragment_complement,
        "candidates_with_fragment_cryptic_bonus": telemetry.candidates_with_fragment_cryptic_bonus,
        "raw_pi_clash_sum": finite_json_number(telemetry.raw_pi_clash_sum),
        "adjusted_pi_clash_sum": finite_json_number(telemetry.adjusted_pi_clash_sum),
        "ligand_sdf_path": config.scaffold_sdf.display().to_string(),
        "output": config.output.display().to_string(),
    });
    std::fs::write(path, serde_json::to_string_pretty(&payload)? + "\n")?;
    Ok(())
}

fn finite_json_number(value: f64) -> Value {
    if value.is_finite() {
        json!(value)
    } else {
        Value::Null
    }
}

fn top_dihedral_bins(values: &[f64]) -> String {
    let mut counts: BTreeMap<i64, usize> = BTreeMap::new();
    for value in values {
        if value.is_finite() {
            *counts.entry(value.round() as i64).or_default() += 1;
        }
    }
    let mut rows = counts.into_iter().collect::<Vec<_>>();
    rows.sort_by(|lhs, rhs| rhs.1.cmp(&lhs.1).then_with(|| lhs.0.cmp(&rhs.0)));
    rows.into_iter()
        .take(6)
        .map(|(bin, count)| format!("{bin}:{count}"))
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod streaming_writer_guard_tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn survivor(index: usize) -> Survivor {
        Survivor {
            anchor_id: format!("ANCHOR_{index:06}"),
            canonical_smiles: format!("CCO{index}"),
            product_id: format!("PRODUCT_{index:06}"),
            smiles: format!("CCO{index}"),
            synthon_a_id: format!("A_{index:06}"),
            synthon_b_id: "B".to_owned(),
            score: index as f64,
            fragment_pi_complement: 1.0,
            fragment_pi_clash_adjusted: 0.1,
            pi_complement: 1.0,
            pi_clash: 0.1,
            cryptic_bonus: 0.0,
            cryptic_bonus_atoms: 0,
            survival_tier: "normal",
            selected_dihedral_deg: 60.0,
            assembly_mode: "smarts_zmatrix",
            z_matrix_active: true,
            rotamers_evaluated: 1,
            best_rotamer_rank: 1,
            ligand_sdf_path: "unit.sdf".to_owned(),
            pose_reconciliation_method: "unit".to_owned(),
            coordinates_json: "[]".to_owned(),
        }
    }

    #[test]
    fn guard_mpsc_streaming_writer_flushes_and_atomic_renames() -> Result<()> {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let path = std::env::temp_dir().join(format!("vspace_streaming_guard_{nonce}.parquet"));
        let tmp_path = path.with_extension("parquet.tmp");
        let (tx, rx) = mpsc::channel();
        for index in 0..3 {
            tx.send(survivor(index))?;
        }
        drop(tx);
        let rows = write_survivors_streaming(&path, rx, 2)?;
        assert_eq!(rows, 3);
        assert!(path.exists());
        assert!(!tmp_path.exists());
        std::fs::remove_file(path)?;
        Ok(())
    }
}
