//! PRISM-Zero v3.1 Intrinsic Reward System
//!
//! This module calculates rewards using only geometric/physical metrics,
//! enabling completely unsupervised training.
//!
//! ## Key Principle: JSON → Math Connection
//! The `RewardWeighting` struct from the manifest is directly plugged into the
//! reward calculation, allowing you to tune the scoring function without recompiling.
//!
//! ## Reward Formula
//! ```text
//! reward = (exposure_gain × exposure_weight)
//!        + (target_bonus × [target_achieved])
//!        - (stability_penalty)
//!        - (clash_penalty × num_clashes)
//!
//! where stability_penalty = max(0, core_rmsd - threshold) × rmsd_weight
//! ```

use crate::buffers::SimulationBuffers;
use crate::manifest::RewardWeighting;
use anyhow::Result;
use std::collections::{HashMap, HashSet};

// ============================================================================================
// DEFAULT CONFIGURATION (Used when no manifest is provided)
// ============================================================================================
const DEFAULT_CUTOFF_RADIUS: f32 = 8.0;

// ============================================================================================
// PUBLIC INTERFACE
// ============================================================================================

/// Detailed simulation evaluation results
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    /// Total exposure gain (proxy for SASA increase)
    pub total_sasa_gain: f32,
    /// Core RMSD from initial structure (Angstroms)
    pub core_rmsd: f32,
    /// Final computed reward value
    pub intrinsic_reward: f32,
    /// Number of atomic clashes detected
    pub clash_count: u32,
    /// Whether target exposure threshold was achieved
    pub target_achieved: bool,
    /// Breakdown of reward components for debugging
    pub reward_breakdown: RewardBreakdown,
}

/// Detailed breakdown of reward components
#[derive(Debug, Clone, Default)]
pub struct RewardBreakdown {
    pub exposure_component: f32,
    pub stability_penalty: f32,
    pub clash_penalty: f32,
    pub target_bonus: f32,
    /// v3.1.1: Bonus for glycan-correlated discovery
    pub glycan_discovery_bonus: f32,
    /// v3.1.1: Bonus for adjacent residue exposure
    pub adjacent_bonus: f32,
    /// v3.1.1: Site value multiplier applied
    pub site_value_multiplier: f32,
}

/// Main entry point - evaluates simulation with configurable weights from manifest
///
/// This function connects the JSON Configuration (Manifest) directly to the Math (Rewards),
/// allowing you to tune the reward function from JSON without recompiling.
pub fn evaluate_simulation_weighted(
    initial: &SimulationBuffers,
    final_state: &SimulationBuffers,
    target_residues: &[usize],
    core_residues: &[usize],
    weights: &RewardWeighting,
    expected_sasa_gain: f32,
) -> Result<EvaluationResult> {
    // 1. Identify Target vs Core Atoms
    let target_res_set: HashSet<usize> = target_residues.iter().cloned().collect();
    let core_res_set: HashSet<usize> = core_residues.iter().cloned().collect();

    let mut target_atom_indices = Vec::new();
    let mut core_atom_indices = Vec::new();

    for i in 0..initial.num_atoms {
        if let Some(res_idx) = initial.atom_to_res.get(i) {
            let res = *res_idx as usize;
            if target_res_set.contains(&res) {
                target_atom_indices.push(i);
            }
            if core_res_set.contains(&res) {
                core_atom_indices.push(i);
            }
        }
    }

    // Fall back to non-target atoms as core if no explicit core residues
    if core_atom_indices.is_empty() {
        for i in 0..initial.num_atoms {
            if let Some(res_idx) = initial.atom_to_res.get(i) {
                if !target_res_set.contains(&(*res_idx as usize)) {
                    core_atom_indices.push(i);
                }
            }
        }
    }

    // 2. Calculate Exposure Gain (The "Carrot")
    let cutoff = weights.clash_distance.max(DEFAULT_CUTOFF_RADIUS);
    let initial_exposure = calculate_exposure_score(initial, &target_atom_indices, cutoff);
    let final_exposure = calculate_exposure_score(final_state, &target_atom_indices, cutoff);
    let exposure_gain = final_exposure - initial_exposure;

    // 3. Calculate Core RMSD (The "Stick")
    let core_rmsd = calculate_core_rmsd(initial, final_state, &core_atom_indices);

    // 4. Calculate Clashes
    let clash_count = count_clashes(final_state, weights.clash_distance);

    // 5. Check Target Achievement
    let target_achieved = exposure_gain >= expected_sasa_gain;

    // 6. Compute Final Reward with Weights from Manifest
    let mut breakdown = RewardBreakdown::default();

    // Exposure component (positive reward for opening cryptic sites)
    breakdown.exposure_component = exposure_gain * weights.exposure_weight;

    // Stability penalty (only kicks in above threshold)
    let rmsd_excess = (core_rmsd - weights.stability_threshold).max(0.0);
    breakdown.stability_penalty = rmsd_excess * weights.rmsd_weight;

    // Clash penalty
    breakdown.clash_penalty = clash_count as f32 * weights.clash_penalty;

    // Target bonus (big reward for achieving the goal)
    breakdown.target_bonus = if target_achieved {
        weights.target_bonus
    } else {
        0.0
    };

    // Final reward calculation
    let intrinsic_reward = breakdown.exposure_component + breakdown.target_bonus
        - breakdown.stability_penalty
        - breakdown.clash_penalty;

    Ok(EvaluationResult {
        total_sasa_gain: exposure_gain,
        core_rmsd,
        intrinsic_reward,
        clash_count,
        target_achieved,
        reward_breakdown: breakdown,
    })
}

/// Legacy entry point for backward compatibility (uses default weights)
pub fn evaluate_simulation(
    initial: &SimulationBuffers,
    final_state: &SimulationBuffers,
    target_residues: &[usize],
) -> Result<EvaluationResult> {
    evaluate_simulation_weighted(
        initial,
        final_state,
        target_residues,
        &[], // No explicit core residues, will use non-target atoms
        &RewardWeighting::default(),
        100.0, // Default expected SASA gain
    )
}

// ============================================================================================
// SPATIAL HASHING (O(N) Neighbor Search)
// ============================================================================================

struct SpatialGrid {
    cell_size: f32,
    cutoff_sq: f32,
    cells: HashMap<(i32, i32, i32), Vec<usize>>,
}

impl SpatialGrid {
    fn new(buffers: &SimulationBuffers, cell_size: f32) -> Self {
        let mut cells = HashMap::new();
        for i in 0..buffers.num_atoms {
            let x = buffers.positions[i * 4];
            let y = buffers.positions[i * 4 + 1];
            let z = buffers.positions[i * 4 + 2];

            let key = (
                (x / cell_size).floor() as i32,
                (y / cell_size).floor() as i32,
                (z / cell_size).floor() as i32,
            );

            cells.entry(key).or_insert_with(Vec::new).push(i);
        }
        Self {
            cell_size,
            cutoff_sq: cell_size * cell_size,
            cells,
        }
    }

    fn count_neighbors(&self, buffers: &SimulationBuffers, atom_idx: usize) -> f32 {
        let x = buffers.positions[atom_idx * 4];
        let y = buffers.positions[atom_idx * 4 + 1];
        let z = buffers.positions[atom_idx * 4 + 2];

        let cx = (x / self.cell_size).floor() as i32;
        let cy = (y / self.cell_size).floor() as i32;
        let cz = (z / self.cell_size).floor() as i32;

        let mut count = 0.0;

        // Check 3x3x3 block of cells
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(indices) = self.cells.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &j in indices {
                            if atom_idx == j {
                                continue;
                            }

                            let x2 = buffers.positions[j * 4];
                            let y2 = buffers.positions[j * 4 + 1];
                            let z2 = buffers.positions[j * 4 + 2];

                            let dist_sq =
                                (x - x2) * (x - x2) + (y - y2) * (y - y2) + (z - z2) * (z - z2);
                            if dist_sq < self.cutoff_sq {
                                count += 1.0;
                            }
                        }
                    }
                }
            }
        }
        count
    }

    /// Count pairs of atoms closer than a given distance (clash detection)
    fn count_close_pairs(&self, buffers: &SimulationBuffers, dist_threshold: f32) -> u32 {
        let threshold_sq = dist_threshold * dist_threshold;
        let mut clash_count = 0u32;

        for i in 0..buffers.num_atoms {
            let x = buffers.positions[i * 4];
            let y = buffers.positions[i * 4 + 1];
            let z = buffers.positions[i * 4 + 2];

            let cx = (x / self.cell_size).floor() as i32;
            let cy = (y / self.cell_size).floor() as i32;
            let cz = (z / self.cell_size).floor() as i32;

            // Only check forward to avoid double counting
            for dx in 0..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if let Some(indices) = self.cells.get(&(cx + dx, cy + dy, cz + dz)) {
                            for &j in indices {
                                if j <= i {
                                    continue;
                                }

                                let x2 = buffers.positions[j * 4];
                                let y2 = buffers.positions[j * 4 + 1];
                                let z2 = buffers.positions[j * 4 + 2];

                                let dist_sq =
                                    (x - x2) * (x - x2) + (y - y2) * (y - y2) + (z - z2) * (z - z2);
                                if dist_sq < threshold_sq {
                                    clash_count += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        clash_count
    }
}

fn calculate_exposure_score(
    buffers: &SimulationBuffers,
    target_indices: &[usize],
    cutoff: f32,
) -> f32 {
    if target_indices.is_empty() {
        return 0.0;
    }

    let grid = SpatialGrid::new(buffers, cutoff);
    let mut total_exposure = 0.0;

    for &idx in target_indices {
        let neighbors = grid.count_neighbors(buffers, idx);
        // Exposure Proxy: Inverse of neighbor count.
        // 0 neighbors = 1.0 (Exposed)
        // Many neighbors -> 0.0 (Buried)
        total_exposure += 1.0 / (1.0 + neighbors);
    }

    // Normalize by number of target atoms
    total_exposure / (target_indices.len() as f32)
}

fn count_clashes(buffers: &SimulationBuffers, clash_distance: f32) -> u32 {
    let grid = SpatialGrid::new(buffers, clash_distance * 2.0);
    grid.count_close_pairs(buffers, clash_distance)
}

// ============================================================================================
// STABILITY METRICS (Translation-Invariant RMSD)
// ============================================================================================

fn calculate_core_rmsd(
    initial: &SimulationBuffers,
    final_state: &SimulationBuffers,
    core_indices: &[usize],
) -> f32 {
    if core_indices.is_empty() {
        return 0.0;
    }

    // 1. Calculate Center of Mass (COM) for Core
    let com_initial = calculate_com(initial, core_indices);
    let com_final = calculate_com(final_state, core_indices);

    // 2. Calculate RMSD with translation correction
    let mut sum_sq_diff = 0.0;

    for &idx in core_indices {
        let base = idx * 4;

        // Shifted positions (translation-invariant)
        let ix = initial.positions[base] - com_initial.0;
        let iy = initial.positions[base + 1] - com_initial.1;
        let iz = initial.positions[base + 2] - com_initial.2;

        let fx = final_state.positions[base] - com_final.0;
        let fy = final_state.positions[base + 1] - com_final.1;
        let fz = final_state.positions[base + 2] - com_final.2;

        let dist_sq = (fx - ix) * (fx - ix) + (fy - iy) * (fy - iy) + (fz - iz) * (fz - iz);
        sum_sq_diff += dist_sq;
    }

    (sum_sq_diff / core_indices.len() as f32).sqrt()
}

fn calculate_com(buffers: &SimulationBuffers, indices: &[usize]) -> (f32, f32, f32) {
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_z = 0.0;

    for &idx in indices {
        sum_x += buffers.positions[idx * 4];
        sum_y += buffers.positions[idx * 4 + 1];
        sum_z += buffers.positions[idx * 4 + 2];
    }

    let n = indices.len() as f32;
    (sum_x / n, sum_y / n, sum_z / n)
}

// ============================================================================================
// MACRO-STEP REWARD UTILITIES
// ============================================================================================

/// Calculate intermediate reward for macro-step training
/// This gives partial credit during the simulation, not just at the end
///
/// UPDATED v3.1.1: Now includes stability penalty and clash penalty for
/// more robust learning that doesn't reward protein destabilization.
pub fn calculate_macro_step_reward(
    initial: &SimulationBuffers,
    current: &SimulationBuffers,
    target_residues: &[usize],
    core_residues: &[usize],
    weights: &RewardWeighting,
    macro_step: usize,
    total_macro_steps: usize,
) -> f32 {
    let target_res_set: HashSet<usize> = target_residues.iter().cloned().collect();
    let core_res_set: HashSet<usize> = core_residues.iter().cloned().collect();

    let mut target_atom_indices = Vec::new();
    let mut core_atom_indices = Vec::new();

    for i in 0..initial.num_atoms {
        if let Some(res_idx) = initial.atom_to_res.get(i) {
            let res = *res_idx as usize;
            if target_res_set.contains(&res) {
                target_atom_indices.push(i);
            }
            if core_res_set.contains(&res) {
                core_atom_indices.push(i);
            }
        }
    }

    // Fallback: if no core residues specified, use non-target atoms
    if core_atom_indices.is_empty() {
        for i in 0..initial.num_atoms {
            if let Some(res_idx) = initial.atom_to_res.get(i) {
                if !target_res_set.contains(&(*res_idx as usize)) {
                    core_atom_indices.push(i);
                }
            }
        }
    }

    // 1. Calculate exposure progress (The "Carrot")
    let initial_exposure =
        calculate_exposure_score(initial, &target_atom_indices, DEFAULT_CUTOFF_RADIUS);
    let current_exposure =
        calculate_exposure_score(current, &target_atom_indices, DEFAULT_CUTOFF_RADIUS);
    let exposure_gain = current_exposure - initial_exposure;

    // 2. Calculate stability penalty (The "Stick")
    let core_rmsd = calculate_core_rmsd(initial, current, &core_atom_indices);
    let rmsd_excess = (core_rmsd - weights.stability_threshold).max(0.0);
    let stability_penalty = rmsd_excess * weights.rmsd_weight;

    // 3. Calculate clash penalty
    let clash_count = count_clashes(current, weights.clash_distance);
    let clash_penalty = clash_count as f32 * weights.clash_penalty;

    // 4. Scale reward based on progress through episode
    // Early macro-steps get smaller rewards, later ones get larger
    // This encourages gradual opening rather than immediate disruption
    let progress_factor = (macro_step as f32 + 1.0) / (total_macro_steps as f32);

    // Final reward: exposure gain minus penalties, scaled by progress
    let reward = (exposure_gain * weights.exposure_weight - stability_penalty - clash_penalty)
        * progress_factor;

    reward
}

// ============================================================================================
// v3.1.1 META-LEARNING REWARD UTILITIES
// ============================================================================================

/// Enhanced macro-step reward with glycan awareness and site value weighting
///
/// Key insight: When glycan displacement correlates with target exposure,
/// it indicates biologically meaningful cryptic site opening (like the 6VXX discovery).
/// This function rewards such correlated discovery with a bonus.
///
/// # Arguments
/// * `initial` - Initial simulation state
/// * `current` - Current simulation state
/// * `target_residues` - Primary cryptic site residues
/// * `core_residues` - Structurally stable reference residues
/// * `glycan_residues` - Glycan-bearing residues (shield positions)
/// * `adjacent_residues` - Secondary residues near primary targets
/// * `site_value` - Priority multiplier (1.0 = standard, 5.0 = high-value epitope)
/// * `weights` - Reward configuration from manifest
/// * `macro_step` - Current macro-step in episode
/// * `total_macro_steps` - Total macro-steps per episode
pub fn calculate_enhanced_macro_step_reward(
    initial: &SimulationBuffers,
    current: &SimulationBuffers,
    target_residues: &[usize],
    core_residues: &[usize],
    glycan_residues: &[usize],
    adjacent_residues: &[usize],
    site_value: f32,
    weights: &RewardWeighting,
    macro_step: usize,
    total_macro_steps: usize,
) -> (f32, RewardBreakdown) {
    let target_res_set: HashSet<usize> = target_residues.iter().cloned().collect();
    let core_res_set: HashSet<usize> = core_residues.iter().cloned().collect();
    let glycan_res_set: HashSet<usize> = glycan_residues.iter().cloned().collect();
    let adjacent_res_set: HashSet<usize> = adjacent_residues.iter().cloned().collect();

    let mut target_atom_indices = Vec::new();
    let mut core_atom_indices = Vec::new();
    let mut glycan_atom_indices = Vec::new();
    let mut adjacent_atom_indices = Vec::new();

    for i in 0..initial.num_atoms {
        if let Some(res_idx) = initial.atom_to_res.get(i) {
            let res = *res_idx as usize;
            if target_res_set.contains(&res) {
                target_atom_indices.push(i);
            }
            if core_res_set.contains(&res) {
                core_atom_indices.push(i);
            }
            if glycan_res_set.contains(&res) {
                glycan_atom_indices.push(i);
            }
            if adjacent_res_set.contains(&res) {
                adjacent_atom_indices.push(i);
            }
        }
    }

    // Fallback: if no core residues specified, use non-target atoms
    if core_atom_indices.is_empty() {
        for i in 0..initial.num_atoms {
            if let Some(res_idx) = initial.atom_to_res.get(i) {
                if !target_res_set.contains(&(*res_idx as usize)) {
                    core_atom_indices.push(i);
                }
            }
        }
    }

    let mut breakdown = RewardBreakdown::default();
    breakdown.site_value_multiplier = site_value * weights.site_value_weight;

    // 1. Calculate exposure progress (Primary target)
    let initial_exposure =
        calculate_exposure_score(initial, &target_atom_indices, DEFAULT_CUTOFF_RADIUS);
    let current_exposure =
        calculate_exposure_score(current, &target_atom_indices, DEFAULT_CUTOFF_RADIUS);
    let exposure_gain = current_exposure - initial_exposure;
    breakdown.exposure_component = exposure_gain * weights.exposure_weight;

    // 2. Calculate stability penalty
    let core_rmsd = calculate_core_rmsd(initial, current, &core_atom_indices);
    let rmsd_excess = (core_rmsd - weights.stability_threshold).max(0.0);
    breakdown.stability_penalty = rmsd_excess * weights.rmsd_weight;

    // 3. Calculate clash penalty
    let clash_count = count_clashes(current, weights.clash_distance);
    breakdown.clash_penalty = clash_count as f32 * weights.clash_penalty;

    // 4. Calculate glycan discovery bonus (NEW in v3.1.1)
    if !glycan_atom_indices.is_empty() && exposure_gain > 0.0 {
        let glycan_displacement =
            calculate_glycan_displacement(initial, current, &glycan_atom_indices);

        // If glycan displacement exceeds threshold AND we have exposure gain,
        // this indicates biologically meaningful opening (the 6VXX insight!)
        if glycan_displacement >= weights.glycan_displacement_threshold {
            // Correlation-based bonus: more displacement + more exposure = bigger bonus
            let correlation_factor = (glycan_displacement / 10.0).min(1.0) * exposure_gain;
            breakdown.glycan_discovery_bonus = correlation_factor * weights.glycan_discovery_bonus;
        }
    }

    // 5. Calculate adjacent residue bonus (partial discovery indicator)
    if !adjacent_atom_indices.is_empty() {
        let initial_adj_exposure =
            calculate_exposure_score(initial, &adjacent_atom_indices, DEFAULT_CUTOFF_RADIUS);
        let current_adj_exposure =
            calculate_exposure_score(current, &adjacent_atom_indices, DEFAULT_CUTOFF_RADIUS);
        let adj_exposure_gain = current_adj_exposure - initial_adj_exposure;

        if adj_exposure_gain > 0.0 {
            breakdown.adjacent_bonus = adj_exposure_gain * weights.adjacent_bonus;
        }
    }

    // 6. Scale reward based on progress through episode
    let progress_factor = (macro_step as f32 + 1.0) / (total_macro_steps as f32);

    // Final reward calculation with all components
    let base_reward =
        breakdown.exposure_component + breakdown.glycan_discovery_bonus + breakdown.adjacent_bonus
            - breakdown.stability_penalty
            - breakdown.clash_penalty;

    // Apply site value multiplier and progress scaling
    let reward = base_reward * breakdown.site_value_multiplier * progress_factor;

    (reward, breakdown)
}

/// Calculate average glycan displacement from initial positions
fn calculate_glycan_displacement(
    initial: &SimulationBuffers,
    current: &SimulationBuffers,
    glycan_indices: &[usize],
) -> f32 {
    if glycan_indices.is_empty() {
        return 0.0;
    }

    let mut total_displacement = 0.0;

    for &idx in glycan_indices {
        let base = idx * 4;
        if base + 2 < initial.positions.len() && base + 2 < current.positions.len() {
            let dx = current.positions[base] - initial.positions[base];
            let dy = current.positions[base + 1] - initial.positions[base + 1];
            let dz = current.positions[base + 2] - initial.positions[base + 2];
            total_displacement += (dx * dx + dy * dy + dz * dz).sqrt();
        }
    }

    total_displacement / glycan_indices.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_weights() {
        let weights = RewardWeighting::default();
        assert_eq!(weights.exposure_weight, 10.0);
        assert_eq!(weights.rmsd_weight, 5.0);
        assert_eq!(weights.stability_threshold, 2.5);
    }

    #[test]
    fn test_reward_breakdown() {
        let breakdown = RewardBreakdown::default();
        assert_eq!(breakdown.exposure_component, 0.0);
        assert_eq!(breakdown.stability_penalty, 0.0);
    }
}
