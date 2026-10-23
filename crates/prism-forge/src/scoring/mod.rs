use anyhow::{anyhow, Context, Result};
use arrow_array::{
    Array, BooleanArray, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray,
    UInt32Array, UInt64Array,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::Value;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs::File;
use std::path::Path;

pub const W_COMPLEMENT: f64 = 1.0;
pub const W_CLASH_POCKET: f64 = 1.0;
pub const W_LOCK_GEO: f64 = 1.0;
pub const W_SHEAR: f64 = 0.25;
pub const W_CONSENSUS: f64 = 1.0;
pub const W_NMA: f64 = 0.35;
pub const W_HYDRATION: f64 = 0.35;
pub const W_CONTINUITY: f64 = 0.50;
pub const W_PATHWAY: f64 = 0.25;
pub const COULOMBIC_INEFFICIENCY: f64 = 0.05;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoxelClassification {
    StableOccupied,
    ThermallyDestabilized,
    ThermallyActivated,
    ThermallyReleased,
    Void,
}

#[derive(Debug, Clone)]
pub struct VoxelField {
    pub complement: f64,
    pub clash: f64,
    pub raw_clash: f64,
    pub cryptic_bonus: f64,
    pub stable_occupied: bool,
    pub thermally_destabilized: bool,
    pub thermally_activated: bool,
    pub thermally_released: bool,
    pub coherence_score: f64,
    pub coherence_factor: f64,
    pub coherence_missing: bool,
    pub primary_residue_idx: Option<i64>,
    pub cold_mean: f64,
    pub warm_mean: f64,
    pub delta: f64,
    pub consensus_complement_bonus: f64,
    pub on_activation_pathway: bool,
}

impl VoxelField {
    fn pathway_void() -> Self {
        Self {
            complement: 0.0,
            clash: 0.0,
            raw_clash: 0.0,
            cryptic_bonus: 0.0,
            stable_occupied: false,
            thermally_destabilized: false,
            thermally_activated: false,
            thermally_released: false,
            coherence_score: 1.0,
            coherence_factor: 1.0,
            coherence_missing: false,
            primary_residue_idx: None,
            cold_mean: 0.0,
            warm_mean: 0.0,
            delta: 0.0,
            consensus_complement_bonus: 0.0,
            on_activation_pathway: true,
        }
    }

    pub fn classification(&self) -> VoxelClassification {
        if self.stable_occupied {
            VoxelClassification::StableOccupied
        } else if self.thermally_destabilized {
            VoxelClassification::ThermallyDestabilized
        } else if self.thermally_activated {
            VoxelClassification::ThermallyActivated
        } else if self.thermally_released {
            VoxelClassification::ThermallyReleased
        } else {
            VoxelClassification::Void
        }
    }
}

#[derive(Debug, Clone)]
pub struct LoadedSignalGrid {
    pub field: HashMap<u64, VoxelField>,
    pub origin: [f64; 3],
    pub spacing: f64,
    pub dims: [usize; 3],
    pub condition_id: String,
}

#[derive(Debug, Clone, Default)]
pub struct ContinuityVoxelProfile {
    pub nma_disruption_penalty: f64,
    pub hydration_blockade_penalty: f64,
    pub thermodynamic_trap_penalty: f64,
}

#[derive(Debug, Clone, Default)]
pub struct LoadedContinuityMaps {
    pub by_voxel: HashMap<u64, ContinuityVoxelProfile>,
    pub global_nma_penalty: f64,
    pub global_hydration_penalty: f64,
    pub global_thermodynamic_penalty: f64,
    pub provenance: String,
}

#[derive(Debug, Clone)]
struct SignalGridRow {
    voxel_idx: u64,
    class_raw: String,
    cold_mean: f64,
    warm_mean: f64,
    delta: f64,
    consensus_bonus: f64,
    on_activation_pathway: bool,
}

#[derive(Debug, Clone, Copy)]
struct DynamicThresholds {
    cold_p80: f64,
    warm_p80: f64,
    gain_p90: f64,
    release_p90: f64,
    release_p95: f64,
    nonvoid_count: usize,
}

impl DynamicThresholds {
    fn from_rows(rows: &[SignalGridRow]) -> Self {
        let mut cold = Vec::new();
        let mut warm = Vec::new();
        let mut gain = Vec::new();
        let mut release = Vec::new();
        for row in rows {
            if row.cold_mean > 0.0 || row.warm_mean > 0.0 {
                cold.push(row.cold_mean);
                warm.push(row.warm_mean);
            }
            let gain_delta = row.warm_mean - row.cold_mean;
            if gain_delta.is_finite() && gain_delta > 0.0 {
                gain.push(gain_delta);
            }
            let release_delta = row.cold_mean - row.warm_mean;
            if release_delta.is_finite() && release_delta > 0.0 {
                release.push(release_delta);
            }
        }
        Self {
            cold_p80: quantile_nearest_optional(&mut cold, 0.80).unwrap_or(f64::INFINITY),
            warm_p80: quantile_nearest_optional(&mut warm, 0.80).unwrap_or(f64::INFINITY),
            gain_p90: quantile_nearest_optional(&mut gain, 0.90).unwrap_or(f64::INFINITY),
            release_p90: quantile_nearest_optional(&mut release, 0.90).unwrap_or(f64::INFINITY),
            release_p95: quantile_nearest_optional(&mut release, 0.95).unwrap_or(f64::INFINITY),
            nonvoid_count: cold.len().max(warm.len()),
        }
    }
}

impl LoadedSignalGrid {
    pub fn from_parquet(path: &str, config_path: &str) -> Result<Self> {
        let conditions = collect_condition_ids(Path::new(path))?;
        let condition_id = select_condition_id(&conditions);
        let (origin, spacing, dims) = load_grid_geometry(Path::new(config_path), &condition_id)?;
        let mut reader = parquet_reader(Path::new(path))?;
        let mut raw_rows = Vec::new();
        for batch in &mut reader {
            let batch = batch?;
            for row in 0..batch.num_rows() {
                if let Some(row_condition) = optional_string_value(&batch, "condition_id", row)? {
                    if row_condition != condition_id {
                        continue;
                    }
                }
                let voxel_idx = required_u64_value(&batch, "voxel_idx", row)?;
                let class_raw = optional_string_value(&batch, "variance_class", row)?
                    .or(optional_string_value(
                        &batch,
                        "variance_classification",
                        row,
                    )?)
                    .or(optional_string_value(
                        &batch,
                        "consensus_raw_variance_class",
                        row,
                    )?)
                    .unwrap_or_else(|| "void".to_owned());
                let cold_mean =
                    optional_f64_value(&batch, "hit_count_cold_mean", row)?.unwrap_or(0.0);
                let warm_mean =
                    optional_f64_value(&batch, "hit_count_warm_mean", row)?.unwrap_or(0.0);
                let delta = optional_f64_value(&batch, "hit_count_delta", row)?
                    .unwrap_or(warm_mean - cold_mean);
                let consensus_bonus = optional_f64_value(&batch, "scaffold_consensus_bonus", row)?
                    .or(optional_f64_value(
                        &batch,
                        "consensus_complement_bonus",
                        row,
                    )?)
                    .unwrap_or(0.0);
                let on_activation_pathway =
                    optional_bool_value(&batch, "on_activation_pathway", row)?
                        .or(optional_bool_value(&batch, "activation_pathway", row)?)
                        .or(optional_bool_value(&batch, "pathway_voxel", row)?)
                        .unwrap_or(false);
                raw_rows.push(SignalGridRow {
                    voxel_idx,
                    class_raw,
                    cold_mean,
                    warm_mean,
                    delta,
                    consensus_bonus,
                    on_activation_pathway,
                });
            }
        }

        let thresholds = DynamicThresholds::from_rows(&raw_rows);
        let use_dynamic = thresholds.nonvoid_count > 0;
        let mut field = HashMap::new();
        for row in raw_rows {
            let explicit_classification = parse_classification(&row.class_raw);
            let classification = if explicit_classification != VoxelClassification::Void {
                explicit_classification
            } else if use_dynamic {
                classify_dynamic(row.cold_mean, row.warm_mean, thresholds)
            } else {
                explicit_classification
            };
            let gain_delta = row.warm_mean - row.cold_mean;
            let release_delta = row.cold_mean - row.warm_mean;
            let complement = match classification {
                VoxelClassification::ThermallyActivated
                | VoxelClassification::ThermallyReleased => 1.0,
                _ => 0.0,
            };
            let raw_clash = match classification {
                VoxelClassification::StableOccupied => 1.0,
                VoxelClassification::ThermallyDestabilized => 0.5,
                _ => 0.0,
            };
            let cryptic_bonus = if row.consensus_bonus > 0.0 && complement > 0.0 {
                row.consensus_bonus
            } else if use_dynamic
                && ((classification == VoxelClassification::ThermallyActivated
                    && gain_delta >= thresholds.gain_p90)
                    || (classification == VoxelClassification::ThermallyReleased
                        && release_delta >= thresholds.release_p95))
            {
                2.0
            } else {
                0.0
            };
            if complement == 0.0
                && raw_clash == 0.0
                && cryptic_bonus == 0.0
                && row.cold_mean == 0.0
                && row.warm_mean == 0.0
                && !row.on_activation_pathway
            {
                continue;
            }
            field.insert(
                row.voxel_idx,
                VoxelField {
                    complement,
                    clash: raw_clash,
                    raw_clash,
                    cryptic_bonus,
                    stable_occupied: classification == VoxelClassification::StableOccupied,
                    thermally_destabilized: classification
                        == VoxelClassification::ThermallyDestabilized,
                    thermally_activated: classification == VoxelClassification::ThermallyActivated,
                    thermally_released: classification == VoxelClassification::ThermallyReleased,
                    coherence_score: 1.0,
                    coherence_factor: 1.0,
                    coherence_missing: false,
                    primary_residue_idx: None,
                    cold_mean: row.cold_mean,
                    warm_mean: row.warm_mean,
                    delta: if row.delta.is_finite() {
                        row.delta
                    } else {
                        row.warm_mean - row.cold_mean
                    },
                    consensus_complement_bonus: row.consensus_bonus,
                    on_activation_pathway: row.on_activation_pathway,
                },
            );
        }
        Ok(Self {
            field,
            origin,
            spacing,
            dims,
            condition_id,
        })
    }

    pub fn map_to_voxel(&self, x: f64, y: f64, z: f64) -> Option<u64> {
        if self.spacing <= 0.0 {
            return None;
        }
        let ix = ((x - self.origin[0]) / self.spacing).floor() as i64;
        let iy = ((y - self.origin[1]) / self.spacing).floor() as i64;
        let iz = ((z - self.origin[2]) / self.spacing).floor() as i64;
        let (nx, ny, nz) = (
            self.dims[0] as i64,
            self.dims[1] as i64,
            self.dims[2] as i64,
        );
        if ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz {
            return None;
        }
        Some((ix + iy * nx + iz * nx * ny) as u64)
    }

    pub fn add_activation_pathway_from_parquet(&mut self, path: &Path) -> Result<usize> {
        let pathway_voxels = load_pathway_voxels(path, &self.condition_id)?;
        let count = pathway_voxels.len();
        for voxel_idx in pathway_voxels {
            self.field
                .entry(voxel_idx)
                .and_modify(|field| field.on_activation_pathway = true)
                .or_insert_with(VoxelField::pathway_void);
        }
        Ok(count)
    }
}

#[derive(Debug, Clone)]
pub struct MoleculeScore {
    pub reward: f64,
    pub pi_complement: f64,
    pub pi_clash_pocket: f64,
    pub pi_clash_lock: f64,
    pub sigma_shear: f64,
    pub consensus_bonus: f64,
    pub cryptic_bonus: f64,
    pub cryptic_bonus_atoms: u64,
    pub lock_atom_count: u16,
    pub lock_cold: f64,
    pub lock_warm: f64,
    pub lock_delta: f64,
    pub lock_occupancy_per_phase: [f64; 5],
    pub pathway_voxels: u16,
    pub void_atom_count: u16,
    pub occupied_lock_voxels: Vec<u64>,
    pub nma_disruption_penalty: f64,
    pub hydration_blockade_penalty: f64,
    pub thermodynamic_trap_penalty: f64,
    pub pathway_bonus: f64,
    pub u_pose: f64,
    pub continuity_admissibility: bool,
    pub continuity_reward_v1: f64,
    pub continuity_provenance: String,
    // WT-REROUTE (D1 bounded-positive shaped-low): physical-feasibility verdict for
    // the generated chemistry. `feasible == false` is NOT a terminal reject and NOT a
    // mask — it routes the candidate to a shaped-low (but strictly positive) reward so
    // the GFlowNet learns to avoid the region while Trajectory-Balance math stays valid
    // (`terminal_rewards > 0`, trajectory_balance.py:49). `feasibility_reason_codes` is a
    // JSON array of `FeasibilityReason` strings (ligand_feasibility_gate.py:93-141), the
    // policy's machine-readable steering vocabulary. FIREWALL: this verdict lives only in
    // the generative reward lane and never enters the canonical lexicographic site_ranker.
    pub feasible: bool,
    pub feasibility_reason_codes: String,
}

impl Default for MoleculeScore {
    fn default() -> Self {
        Self {
            reward: 0.0,
            pi_complement: 0.0,
            pi_clash_pocket: 0.0,
            pi_clash_lock: 0.0,
            sigma_shear: 0.0,
            consensus_bonus: 0.0,
            cryptic_bonus: 0.0,
            cryptic_bonus_atoms: 0,
            lock_atom_count: 0,
            lock_cold: 0.0,
            lock_warm: 0.0,
            lock_delta: 0.0,
            lock_occupancy_per_phase: [0.0; 5],
            pathway_voxels: 0,
            void_atom_count: 0,
            occupied_lock_voxels: Vec::new(),
            nma_disruption_penalty: 0.0,
            hydration_blockade_penalty: 0.0,
            thermodynamic_trap_penalty: 0.0,
            pathway_bonus: 0.0,
            u_pose: 0.0,
            continuity_admissibility: true,
            continuity_reward_v1: 0.0,
            continuity_provenance: "NOT_REQUESTED".to_owned(),
            // Default to feasible/no-violations: a freshly scored molecule is presumed
            // physically admissible until the feasibility gate says otherwise. The empty
            // JSON array is the "no reason codes" sentinel.
            feasible: true,
            feasibility_reason_codes: "[]".to_owned(),
        }
    }
}

pub fn phase_resolved_lock_profile(cold: f64, warm: f64, delta: f64) -> [f64; 5] {
    let thermal_delta = if delta.is_finite() {
        delta
    } else {
        warm - cold
    };
    [
        cold,
        cold + 0.5 * thermal_delta,
        warm,
        warm - 0.5 * thermal_delta,
        cold * (1.0 + COULOMBIC_INEFFICIENCY),
    ]
}

pub fn score_molecule(
    atom_positions: &[(f64, f64, f64)],
    grid: &LoadedSignalGrid,
    lock_mask: &HashSet<u64>,
    shear_field: &HashMap<u64, f64>,
) -> MoleculeScore {
    score_positions_with_field(
        atom_positions,
        &grid.field,
        |x, y, z| grid.map_to_voxel(x, y, z),
        lock_mask,
        shear_field,
    )
}

pub fn score_molecule_with_continuity(
    atom_positions: &[(f64, f64, f64)],
    grid: &LoadedSignalGrid,
    lock_mask: &HashSet<u64>,
    shear_field: &HashMap<u64, f64>,
    continuity_maps: &LoadedContinuityMaps,
) -> MoleculeScore {
    score_molecule_with_continuity_and_pose(
        atom_positions,
        grid,
        lock_mask,
        shear_field,
        continuity_maps,
        0.0,
    )
}

pub fn score_molecule_with_continuity_and_pose(
    atom_positions: &[(f64, f64, f64)],
    grid: &LoadedSignalGrid,
    lock_mask: &HashSet<u64>,
    shear_field: &HashMap<u64, f64>,
    continuity_maps: &LoadedContinuityMaps,
    u_pose: f64,
) -> MoleculeScore {
    let mut result = score_molecule(atom_positions, grid, lock_mask, shear_field);
    result.u_pose = if u_pose.is_finite() {
        u_pose.max(0.0)
    } else {
        0.0
    };
    let mut hits = 0_u64;
    for &(x, y, z) in atom_positions {
        let Some(voxel_idx) = grid.map_to_voxel(x, y, z) else {
            continue;
        };
        if let Some(profile) = continuity_maps.by_voxel.get(&voxel_idx) {
            result.nma_disruption_penalty += profile.nma_disruption_penalty;
            result.hydration_blockade_penalty += profile.hydration_blockade_penalty;
            result.thermodynamic_trap_penalty += profile.thermodynamic_trap_penalty;
            hits = hits.saturating_add(1);
        }
    }
    if hits == 0 {
        result.nma_disruption_penalty = continuity_maps.global_nma_penalty;
        result.hydration_blockade_penalty = continuity_maps.global_hydration_penalty;
        result.thermodynamic_trap_penalty = continuity_maps.global_thermodynamic_penalty;
    } else {
        let denom = hits as f64;
        result.nma_disruption_penalty /= denom;
        result.hydration_blockade_penalty /= denom;
        result.thermodynamic_trap_penalty /= denom;
    }
    let total_penalty = W_NMA * result.nma_disruption_penalty
        + W_HYDRATION * result.hydration_blockade_penalty
        + W_CONTINUITY * result.thermodynamic_trap_penalty
        + result.u_pose;
    result.pathway_bonus = W_PATHWAY * f64::from(result.pathway_voxels);
    result.continuity_reward_v1 =
        (result.reward - total_penalty + result.pathway_bonus).max(1.0e-8);
    result.continuity_admissibility = result.thermodynamic_trap_penalty < 0.85
        && result.hydration_blockade_penalty < 0.85
        && result.nma_disruption_penalty < 10.0;
    result.continuity_provenance = continuity_maps.provenance.clone();
    result.reward = result.continuity_reward_v1;
    result
}

pub fn score_positions_with_field<F>(
    atom_positions: &[(f64, f64, f64)],
    field: &HashMap<u64, VoxelField>,
    mut map_to_voxel: F,
    lock_mask: &HashSet<u64>,
    shear_field: &HashMap<u64, f64>,
) -> MoleculeScore
where
    F: FnMut(f64, f64, f64) -> Option<u64>,
{
    let mut result = MoleculeScore::default();
    let mut occupied_lock_voxels = HashSet::new();
    for &(x, y, z) in atom_positions {
        let voxel_idx = match map_to_voxel(x, y, z) {
            Some(value) => value,
            None => {
                result.void_atom_count = result.void_atom_count.saturating_add(1);
                continue;
            }
        };
        if let Some(field) = field.get(&voxel_idx) {
            let classification = field.classification();
            let in_lock_region = lock_mask.contains(&voxel_idx);
            if in_lock_region && !matches!(classification, VoxelClassification::Void) {
                result.lock_atom_count = result.lock_atom_count.saturating_add(1);
                occupied_lock_voxels.insert(voxel_idx);
            }
            match classification {
                VoxelClassification::ThermallyActivated
                | VoxelClassification::ThermallyReleased => {
                    result.pi_complement += field.complement;
                    result.consensus_bonus += field.consensus_complement_bonus;
                }
                VoxelClassification::StableOccupied => {
                    if in_lock_region {
                        result.pi_clash_lock += field.clash;
                    } else {
                        result.pi_clash_pocket += field.clash;
                    }
                }
                VoxelClassification::ThermallyDestabilized => {
                    if in_lock_region {
                        result.pi_clash_lock += field.clash;
                    } else {
                        result.pi_clash_pocket += field.clash;
                    }
                }
                VoxelClassification::Void => {
                    result.void_atom_count = result.void_atom_count.saturating_add(1);
                }
            }
            result.cryptic_bonus += field.cryptic_bonus;
            if field.cryptic_bonus > 0.0 {
                result.cryptic_bonus_atoms = result.cryptic_bonus_atoms.saturating_add(1);
            }
            if in_lock_region {
                result.lock_cold += field.cold_mean;
                result.lock_warm += field.warm_mean;
                result.lock_delta += field.delta;
                let phase_profile =
                    phase_resolved_lock_profile(field.cold_mean, field.warm_mean, field.delta);
                for (slot, value) in result
                    .lock_occupancy_per_phase
                    .iter_mut()
                    .zip(phase_profile)
                {
                    *slot += value;
                }
            }
            if field.on_activation_pathway {
                result.pathway_voxels = result.pathway_voxels.saturating_add(1);
            }
        } else {
            result.void_atom_count = result.void_atom_count.saturating_add(1);
        }
        if let Some(shear) = shear_field.get(&voxel_idx) {
            result.sigma_shear += *shear;
        }
    }
    result.occupied_lock_voxels = occupied_lock_voxels.into_iter().collect();
    result.occupied_lock_voxels.sort_unstable();
    result.reward = W_COMPLEMENT * result.pi_complement - W_CLASH_POCKET * result.pi_clash_pocket
        + W_LOCK_GEO * result.pi_clash_lock
        - W_SHEAR * (1.0 + result.sigma_shear).ln()
        + W_CONSENSUS * result.consensus_bonus
        + result.cryptic_bonus;
    if result.reward < 1.0e-8 {
        result.reward = 1.0e-8;
    }
    result
}

pub fn load_shear_field(path: &Path, condition_hint: &str) -> Result<HashMap<u64, f64>> {
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let mut reader = parquet_reader(path)?;
    let mut shear = HashMap::new();
    for batch in &mut reader {
        let batch = batch?;
        for row in 0..batch.num_rows() {
            if let Some(condition) = optional_string_value(&batch, "condition_id", row)? {
                if !condition_matches_hint(&condition, condition_hint) {
                    continue;
                }
            }
            let voxel_idx = required_u64_value(&batch, "voxel_idx", row)?;
            let value = optional_f64_value(&batch, "shear_stress", row)?
                .or(optional_f64_value(&batch, "frobenius_norm", row)?)
                .or(optional_f64_value(&batch, "shear_stress_max", row)?)
                .unwrap_or(0.0);
            if value > 0.0 {
                shear.insert(voxel_idx, value);
            }
        }
    }
    Ok(shear)
}

pub fn load_pathway_voxels(path: &Path, condition_hint: &str) -> Result<HashSet<u64>> {
    if !path.exists() {
        return Ok(HashSet::new());
    }
    let mut reader = parquet_reader(path)?;
    let mut voxels = HashSet::new();
    for batch in &mut reader {
        let batch = batch?;
        for row in 0..batch.num_rows() {
            if let Some(condition) = optional_string_value(&batch, "condition_id", row)? {
                if !condition_matches_hint(&condition, condition_hint) {
                    continue;
                }
            }
            if optional_bool_value(&batch, "voxel_in_bounds", row)?.is_some_and(|value| !value) {
                continue;
            }
            voxels.insert(required_u64_value(&batch, "voxel_idx", row)?);
        }
    }
    Ok(voxels)
}

pub fn load_continuity_maps(
    nma_path: &Path,
    hydration_path: &Path,
    thermodynamic_path: &Path,
) -> Result<LoadedContinuityMaps> {
    let mut maps = LoadedContinuityMaps {
        provenance: "L3_DERIVED".to_owned(),
        ..LoadedContinuityMaps::default()
    };
    maps.global_nma_penalty = load_continuity_penalty(
        nma_path,
        "voxel_idx",
        "hinge_disruption_risk",
        ContinuityKind::Nma,
        &mut maps.by_voxel,
    )
    .with_context(|| format!("load NMA continuity map {}", nma_path.display()))?;
    maps.global_hydration_penalty = load_continuity_penalty(
        hydration_path,
        "voxel_idx",
        "occlusion_risk",
        ContinuityKind::Hydration,
        &mut maps.by_voxel,
    )
    .with_context(|| format!("load hydration continuity map {}", hydration_path.display()))?;
    maps.global_thermodynamic_penalty = load_continuity_penalty(
        thermodynamic_path,
        "voxel_idx",
        "trap_risk",
        ContinuityKind::Thermodynamic,
        &mut maps.by_voxel,
    )
    .with_context(|| {
        format!(
            "load thermodynamic continuity map {}",
            thermodynamic_path.display()
        )
    })?;
    if maps.global_hydration_penalty == 0.0 {
        maps.provenance = "L3_DERIVED_WITH_L0_HYDRATION_BLOCKER".to_owned();
    }
    Ok(maps)
}

#[derive(Debug, Clone, Copy)]
enum ContinuityKind {
    Nma,
    Hydration,
    Thermodynamic,
}

fn load_continuity_penalty(
    path: &Path,
    voxel_column: &str,
    value_column: &str,
    kind: ContinuityKind,
    by_voxel: &mut HashMap<u64, ContinuityVoxelProfile>,
) -> Result<f64> {
    let mut reader = parquet_reader(path)?;
    let mut total = 0.0_f64;
    let mut count = 0_u64;
    for batch in &mut reader {
        let batch = batch?;
        for row in 0..batch.num_rows() {
            let value = optional_f64_value(&batch, value_column, row)?.unwrap_or(0.0);
            if !value.is_finite() || value < 0.0 {
                continue;
            }
            total += value;
            count = count.saturating_add(1);
            if let Some(voxel_idx) = optional_u64_value(&batch, voxel_column, row)? {
                let entry = by_voxel.entry(voxel_idx).or_default();
                match kind {
                    ContinuityKind::Nma => entry.nma_disruption_penalty = value,
                    ContinuityKind::Hydration => entry.hydration_blockade_penalty = value,
                    ContinuityKind::Thermodynamic => entry.thermodynamic_trap_penalty = value,
                }
            }
        }
    }
    if count == 0 {
        Ok(0.0)
    } else {
        Ok(total / count as f64)
    }
}

fn collect_condition_ids(path: &Path) -> Result<BTreeSet<String>> {
    let mut conditions = BTreeSet::new();
    let mut reader = parquet_reader(path)?;
    for batch in &mut reader {
        let batch = batch?;
        for row in 0..batch.num_rows() {
            if let Some(condition) = optional_string_value(&batch, "condition_id", row)? {
                conditions.insert(condition);
            }
        }
    }
    Ok(conditions)
}

fn select_condition_id(conditions: &BTreeSet<String>) -> String {
    for preferred in [
        "glp1r_6XOX_CONSENSUS_GLOBAL",
        "glp1r_6XOX_WT",
        "glp1r_6XOX_A316T",
        "glp1r_6XOX_T149M",
    ] {
        if conditions.contains(preferred) {
            return preferred.to_owned();
        }
    }
    conditions
        .iter()
        .find(|condition| condition.contains("6XOX"))
        .cloned()
        .or_else(|| conditions.iter().next().cloned())
        .unwrap_or_else(|| "glp1r_6XOX_WT".to_owned())
}

fn condition_matches_hint(condition: &str, hint: &str) -> bool {
    condition == hint
        || (hint.contains("6XOX") && condition.contains("6XOX"))
        || (hint.contains("6LN2") && condition.contains("6LN2"))
        || (hint.contains("5VEX") && condition.contains("5VEX"))
}

fn load_grid_geometry(path: &Path, condition_id: &str) -> Result<([f64; 3], f64, [usize; 3])> {
    let payload: Value = serde_json::from_reader(File::open(path)?)
        .with_context(|| format!("parse {}", path.display()))?;
    let conditions = payload
        .get("conditions")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow!("grid mapping missing conditions object"))?;
    let geometry_key = if conditions.contains_key(condition_id) {
        condition_id.to_owned()
    } else if condition_id.contains("6XOX") && conditions.contains_key("glp1r_6XOX_WT") {
        "glp1r_6XOX_WT".to_owned()
    } else if condition_id.contains("6LN2") && conditions.contains_key("glp1r_6LN2_WT") {
        "glp1r_6LN2_WT".to_owned()
    } else if condition_id.contains("5VEX") && conditions.contains_key("glp1r_5VEX_WT") {
        "glp1r_5VEX_WT".to_owned()
    } else {
        conditions
            .keys()
            .next()
            .ok_or_else(|| anyhow!("grid mapping has no condition entries"))?
            .to_owned()
    };
    let condition = conditions
        .get(&geometry_key)
        .ok_or_else(|| anyhow!("missing geometry for {geometry_key}"))?;
    let origin_values = condition
        .get("origin_xyz_angstrom")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("missing origin_xyz_angstrom for {geometry_key}"))?;
    if origin_values.len() != 3 {
        return Err(anyhow!(
            "origin_xyz_angstrom for {geometry_key} must have 3 values"
        ));
    }
    let origin = [
        json_f64(&origin_values[0], "origin_x")?,
        json_f64(&origin_values[1], "origin_y")?,
        json_f64(&origin_values[2], "origin_z")?,
    ];
    let spacing = condition
        .get("spacing_angstrom")
        .and_then(Value::as_f64)
        .ok_or_else(|| anyhow!("missing spacing_angstrom for {geometry_key}"))?;
    let dim = condition
        .get("grid_dim")
        .and_then(Value::as_u64)
        .unwrap_or(96) as usize;
    let dims = [
        condition
            .get("nx")
            .and_then(Value::as_u64)
            .unwrap_or(dim as u64) as usize,
        condition
            .get("ny")
            .and_then(Value::as_u64)
            .unwrap_or(dim as u64) as usize,
        condition
            .get("nz")
            .and_then(Value::as_u64)
            .unwrap_or(dim as u64) as usize,
    ];
    Ok((origin, spacing, dims))
}

fn json_f64(value: &Value, name: &str) -> Result<f64> {
    value
        .as_f64()
        .ok_or_else(|| anyhow!("{name} must be a number"))
}

fn parse_classification(value: &str) -> VoxelClassification {
    let normalized = value.to_ascii_lowercase();
    if normalized.contains("stable_occupied") {
        VoxelClassification::StableOccupied
    } else if normalized.contains("thermally_destabilized") {
        VoxelClassification::ThermallyDestabilized
    } else if normalized.contains("thermally_activated") {
        VoxelClassification::ThermallyActivated
    } else if normalized.contains("thermally_released") {
        VoxelClassification::ThermallyReleased
    } else {
        VoxelClassification::Void
    }
}

fn classify_dynamic(cold: f64, warm: f64, thresholds: DynamicThresholds) -> VoxelClassification {
    if cold <= 0.0 && warm <= 0.0 {
        return VoxelClassification::Void;
    }
    let warm_activation_gate = if thresholds.warm_p80 <= 0.0 {
        warm > 0.0
    } else {
        warm >= thresholds.warm_p80
    };
    let release_delta = cold - warm;
    if cold >= thresholds.cold_p80 && warm >= thresholds.warm_p80 {
        VoxelClassification::StableOccupied
    } else if cold >= thresholds.cold_p80 && warm < thresholds.warm_p80 {
        VoxelClassification::ThermallyDestabilized
    } else if warm_activation_gate && cold < thresholds.cold_p80 {
        VoxelClassification::ThermallyActivated
    } else if release_delta >= thresholds.release_p90 {
        VoxelClassification::ThermallyReleased
    } else {
        VoxelClassification::Void
    }
}

fn quantile_nearest_optional(values: &mut [f64], q: f64) -> Option<f64> {
    let mut finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return None;
    }
    finite.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
    let idx = ((finite.len() - 1) as f64 * q).round() as usize;
    finite.get(idx).copied()
}

fn parquet_reader(
    path: &Path,
) -> Result<impl Iterator<Item = std::result::Result<RecordBatch, arrow_schema::ArrowError>>> {
    let file = File::open(path)?;
    Ok(ParquetRecordBatchReaderBuilder::try_new(file)?.build()?)
}

fn optional_string_value(batch: &RecordBatch, name: &str, row: usize) -> Result<Option<String>> {
    let Ok(idx) = batch.schema().index_of(name) else {
        return Ok(None);
    };
    let array = batch.column(idx);
    if array.is_null(row) {
        return Ok(None);
    }
    if let Some(values) = array.as_any().downcast_ref::<StringArray>() {
        return Ok(Some(values.value(row).to_owned()));
    }
    if let Some(values) = array
        .as_any()
        .downcast_ref::<arrow_array::LargeStringArray>()
    {
        return Ok(Some(values.value(row).to_owned()));
    }
    Ok(None)
}

fn optional_f64_value(batch: &RecordBatch, name: &str, row: usize) -> Result<Option<f64>> {
    let Ok(idx) = batch.schema().index_of(name) else {
        return Ok(None);
    };
    let array = batch.column(idx);
    if array.is_null(row) {
        return Ok(None);
    }
    if let Some(values) = array.as_any().downcast_ref::<Float64Array>() {
        return Ok(Some(values.value(row)));
    }
    Ok(None)
}

fn optional_u64_value(batch: &RecordBatch, name: &str, row: usize) -> Result<Option<u64>> {
    let Ok(idx) = batch.schema().index_of(name) else {
        return Ok(None);
    };
    let array = batch.column(idx);
    if array.is_null(row) {
        return Ok(None);
    }
    if let Some(values) = array.as_any().downcast_ref::<UInt64Array>() {
        return Ok(Some(values.value(row)));
    }
    if let Some(values) = array.as_any().downcast_ref::<UInt32Array>() {
        return Ok(Some(u64::from(values.value(row))));
    }
    if let Some(values) = array.as_any().downcast_ref::<Int64Array>() {
        let value = values.value(row);
        return Ok(Some(
            u64::try_from(value).map_err(|_| anyhow!("negative {name} at row {row}: {value}"))?,
        ));
    }
    if let Some(values) = array.as_any().downcast_ref::<Int32Array>() {
        let value = values.value(row);
        return Ok(Some(
            u64::try_from(value).map_err(|_| anyhow!("negative {name} at row {row}: {value}"))?,
        ));
    }
    Ok(None)
}

fn optional_bool_value(batch: &RecordBatch, name: &str, row: usize) -> Result<Option<bool>> {
    let Ok(idx) = batch.schema().index_of(name) else {
        return Ok(None);
    };
    let array = batch.column(idx);
    if array.is_null(row) {
        return Ok(None);
    }
    if let Some(values) = array.as_any().downcast_ref::<BooleanArray>() {
        return Ok(Some(values.value(row)));
    }
    if let Some(values) = array.as_any().downcast_ref::<Float64Array>() {
        let value = values.value(row);
        return Ok(Some(value.is_finite() && value != 0.0));
    }
    if let Some(values) = array.as_any().downcast_ref::<UInt64Array>() {
        return Ok(Some(values.value(row) != 0));
    }
    if let Some(values) = array.as_any().downcast_ref::<UInt32Array>() {
        return Ok(Some(values.value(row) != 0));
    }
    if let Some(values) = array.as_any().downcast_ref::<Int64Array>() {
        return Ok(Some(values.value(row) != 0));
    }
    if let Some(values) = array.as_any().downcast_ref::<Int32Array>() {
        return Ok(Some(values.value(row) != 0));
    }
    Ok(None)
}

fn required_u64_value(batch: &RecordBatch, name: &str, row: usize) -> Result<u64> {
    let idx = batch
        .schema()
        .index_of(name)
        .with_context(|| format!("missing column {name}"))?;
    let array = batch.column(idx);
    if array.is_null(row) {
        return Err(anyhow!("null {name} at row {row}"));
    }
    if let Some(values) = array.as_any().downcast_ref::<UInt64Array>() {
        return Ok(values.value(row));
    }
    if let Some(values) = array.as_any().downcast_ref::<UInt32Array>() {
        return Ok(u64::from(values.value(row)));
    }
    if let Some(values) = array.as_any().downcast_ref::<Int64Array>() {
        let value = values.value(row);
        return u64::try_from(value).map_err(|_| anyhow!("negative {name} at row {row}: {value}"));
    }
    if let Some(values) = array.as_any().downcast_ref::<Int32Array>() {
        let value = values.value(row);
        return u64::try_from(value).map_err(|_| anyhow!("negative {name} at row {row}: {value}"));
    }
    Err(anyhow!("column {name} is not an integer type"))
}
