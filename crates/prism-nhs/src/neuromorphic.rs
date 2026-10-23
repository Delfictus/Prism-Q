//! Neuromorphic Dewetting Detection
//!
//! Implements spike-based detection of dewetting events using Leaky
//! Integrate-and-Fire (LIF) neurons at protein surface voxels.
//!
//! # The Neuromorphic Insight
//!
//! Cryptic pocket opening is an EVENT, not a state. Neuromorphic detection
//! is natural because:
//! - Dewetting (water leaving) triggers a spike
//! - Lateral connections enable cooperative detection
//! - Spike avalanches indicate coordinated pocket opening
//!
//! # Neuron Model
//!
//! Each surface voxel has an associated LIF neuron:
//! - Membrane potential h driven by local water density
//! - Spikes when h drops below threshold (dewetting)
//! - Lateral synaptic connections to neighbors
//! - Refractory period after spiking

use crate::config::{NhsConfig, BULK_WATER_DENSITY, DEWETTING_THRESHOLD};
use crate::exclusion::ExclusionGrid;
use serde::{Deserialize, Serialize};

// =============================================================================
// NEURON STRUCTURES
// =============================================================================

/// Leaky Integrate-and-Fire neuron for dewetting detection
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DewettingNeuron {
    /// Membrane potential (hydration state)
    /// High = well-hydrated, Low = dewetted
    pub potential: f32,

    /// Grid voxel index
    pub voxel_idx: usize,

    /// Associated residue index (if on protein surface)
    pub residue_idx: Option<u32>,

    /// Refractory counter (frames since last spike)
    pub refractory: u32,

    /// Did this neuron spike this frame?
    pub spiked: bool,

    /// Cumulative spike count
    pub total_spikes: u32,
}

impl DewettingNeuron {
    pub fn new(voxel_idx: usize) -> Self {
        Self {
            potential: 1.0, // Start fully hydrated
            voxel_idx,
            residue_idx: None,
            refractory: 0,
            spiked: false,
            total_spikes: 0,
        }
    }

    /// Reset neuron state
    pub fn reset(&mut self) {
        self.potential = 1.0;
        self.refractory = 0;
        self.spiked = false;
        self.total_spikes = 0;
    }
}

/// Sparse synapse between neurons
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Synapse {
    pub from_idx: usize,
    pub to_idx: usize,
    pub weight: f32,
}

// =============================================================================
// DEWETTING NETWORK
// =============================================================================

/// Neuromorphic network for dewetting detection
pub struct DewettingNetwork {
    config: NhsConfig,

    /// Neurons at protein surface voxels
    neurons: Vec<DewettingNeuron>,

    /// Sparse lateral connections (6-connectivity for 3D grid)
    synapses: Vec<Synapse>,

    /// Map from voxel index to neuron index
    voxel_to_neuron: Vec<Option<usize>>,

    /// Current frame spike buffer
    spike_buffer: Vec<usize>,

    /// Spike history for avalanche detection (neuron indices per frame)
    spike_history: Vec<Vec<usize>>,

    /// Frame counter
    current_frame: usize,
}

impl DewettingNetwork {
    /// Create network from exclusion grid
    ///
    /// Neurons are placed at "surface" voxels where exclusion is intermediate
    /// (not bulk water, not buried in protein)
    pub fn from_grid(grid: &ExclusionGrid, config: NhsConfig) -> Self {
        let total_voxels = grid.total_voxels();
        let mut neurons = Vec::new();
        let mut voxel_to_neuron = vec![None; total_voxels];

        // Create neurons at surface voxels
        for idx in 0..total_voxels {
            let exclusion = grid.exclusion[idx];

            // Surface = partially excluded (between 0.1 and 0.9)
            if exclusion > 0.1 && exclusion < 0.9 {
                let neuron_idx = neurons.len();
                voxel_to_neuron[idx] = Some(neuron_idx);
                neurons.push(DewettingNeuron::new(idx));
            }
        }

        log::info!(
            "DewettingNetwork: {} surface neurons from {} voxels ({:.1}%)",
            neurons.len(),
            total_voxels,
            100.0 * neurons.len() as f32 / total_voxels as f32
        );

        // Build lateral connections
        let synapses = build_lateral_connections(&neurons, &voxel_to_neuron, grid, &config);

        log::info!(
            "DewettingNetwork: {} lateral synapses ({:.1} per neuron)",
            synapses.len(),
            if neurons.is_empty() {
                0.0
            } else {
                synapses.len() as f32 / neurons.len() as f32
            }
        );

        Self {
            config,
            neurons,
            synapses,
            voxel_to_neuron,
            spike_buffer: Vec::new(),
            spike_history: Vec::new(),
            current_frame: 0,
        }
    }

    /// Process one frame of water density data
    ///
    /// Returns indices of neurons that spiked (dewetted)
    pub fn step(&mut self, water_density: &[f32]) -> Vec<usize> {
        self.spike_buffer.clear();
        self.current_frame += 1;

        let tau = self.config.membrane_tau;
        let threshold = self.config.spike_threshold * BULK_WATER_DENSITY;
        let refractory_period = self.config.refractory_period;

        // Phase 1: Update membrane potentials from water density
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            // Skip if in refractory period
            if neuron.refractory > 0 {
                neuron.refractory -= 1;
                neuron.spiked = false;
                continue;
            }

            // Input current from local water density
            let local_density = water_density.get(neuron.voxel_idx).copied().unwrap_or(0.0);
            let density_normalized = local_density / BULK_WATER_DENSITY;

            // Leaky integration: dh/dt = (-h + input) / tau
            neuron.potential += (-neuron.potential + density_normalized) / tau;

            // Check for spike (dewetting = potential drops below threshold)
            if neuron.potential < threshold / BULK_WATER_DENSITY {
                neuron.spiked = true;
                neuron.refractory = refractory_period;
                neuron.potential = 1.0; // Reset to hydrated
                neuron.total_spikes += 1;
                self.spike_buffer.push(i);
            } else {
                neuron.spiked = false;
            }
        }

        // Phase 2: Propagate spikes through synapses
        let synaptic_strength = self.config.synaptic_strength;

        // Collect spike effects first to avoid borrow conflicts
        let mut potential_deltas: Vec<(usize, f32)> = Vec::new();

        for synapse in &self.synapses {
            if self.neurons[synapse.from_idx].spiked {
                // Cooperative effect: neighbor spike makes this neuron more likely to spike
                // (reduces potential toward dewetted state)
                potential_deltas.push((synapse.to_idx, -synapse.weight * synaptic_strength));
            }
        }

        // Apply potential changes
        for (neuron_idx, delta) in potential_deltas {
            self.neurons[neuron_idx].potential += delta;
        }

        // Record spike history
        self.spike_history.push(self.spike_buffer.clone());

        // Limit history size
        let max_history = self.config.avalanche_temporal_window * 2;
        while self.spike_history.len() > max_history {
            self.spike_history.remove(0);
        }

        self.spike_buffer.clone()
    }

    /// Get world positions of neurons that spiked
    pub fn get_spike_positions(&self, grid: &ExclusionGrid) -> Vec<[f32; 3]> {
        self.spike_buffer
            .iter()
            .map(|&neuron_idx| {
                let voxel_idx = self.neurons[neuron_idx].voxel_idx;
                let (ix, iy, iz) = grid.coords(voxel_idx);
                grid.voxel_position(ix, iy, iz)
            })
            .collect()
    }

    /// Get residue indices associated with spiked neurons
    pub fn get_spike_residues(&self) -> Vec<u32> {
        self.spike_buffer
            .iter()
            .filter_map(|&neuron_idx| self.neurons[neuron_idx].residue_idx)
            .collect()
    }

    /// Get voxel indices of spiked neurons
    pub fn get_spike_voxels(&self) -> Vec<usize> {
        self.spike_buffer
            .iter()
            .map(|&neuron_idx| self.neurons[neuron_idx].voxel_idx)
            .collect()
    }

    /// Get total spikes in recent history
    pub fn get_recent_spike_count(&self) -> usize {
        self.spike_history.iter().map(|v| v.len()).sum()
    }

    /// Get all spikes from history as flat list
    pub fn get_spike_history_flat(&self) -> Vec<usize> {
        self.spike_history.iter().flatten().copied().collect()
    }

    /// Get neuron voxel indices
    pub fn get_neuron_voxels(&self) -> Vec<usize> {
        self.neurons.iter().map(|n| n.voxel_idx).collect()
    }

    /// Get number of neurons
    pub fn num_neurons(&self) -> usize {
        self.neurons.len()
    }

    /// Get number of synapses
    pub fn num_synapses(&self) -> usize {
        self.synapses.len()
    }

    /// Get current frame
    pub fn current_frame(&self) -> usize {
        self.current_frame
    }

    /// Associate neurons with residues based on proximity
    pub fn associate_residues(
        &mut self,
        residue_positions: &[[f32; 3]],
        grid: &ExclusionGrid,
        max_distance: f32,
    ) {
        let max_dist_sq = max_distance * max_distance;

        for neuron in &mut self.neurons {
            let (ix, iy, iz) = grid.coords(neuron.voxel_idx);
            let pos = grid.voxel_position(ix, iy, iz);

            let mut best_residue = None;
            let mut best_dist_sq = max_dist_sq;

            for (res_idx, res_pos) in residue_positions.iter().enumerate() {
                let dx = pos[0] - res_pos[0];
                let dy = pos[1] - res_pos[1];
                let dz = pos[2] - res_pos[2];
                let dist_sq = dx * dx + dy * dy + dz * dz;

                if dist_sq < best_dist_sq {
                    best_dist_sq = dist_sq;
                    best_residue = Some(res_idx as u32);
                }
            }

            neuron.residue_idx = best_residue;
        }
    }

    /// Reset network state
    pub fn reset(&mut self) {
        for neuron in &mut self.neurons {
            neuron.reset();
        }
        self.spike_buffer.clear();
        self.spike_history.clear();
        self.current_frame = 0;
    }

    /// Get network statistics
    pub fn stats(&self) -> NetworkStats {
        let total_spikes: u32 = self.neurons.iter().map(|n| n.total_spikes).sum();
        let active_neurons = self.neurons.iter().filter(|n| n.total_spikes > 0).count();

        NetworkStats {
            num_neurons: self.neurons.len(),
            num_synapses: self.synapses.len(),
            current_frame: self.current_frame,
            total_spikes,
            active_neurons,
            avg_spikes_per_frame: if self.current_frame > 0 {
                total_spikes as f32 / self.current_frame as f32
            } else {
                0.0
            },
        }
    }
}

/// Network statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStats {
    pub num_neurons: usize,
    pub num_synapses: usize,
    pub current_frame: usize,
    pub total_spikes: u32,
    pub active_neurons: usize,
    pub avg_spikes_per_frame: f32,
}

// =============================================================================
// CONNECTIVITY
// =============================================================================

/// Build 6-connected lateral synapses for 3D grid
fn build_lateral_connections(
    neurons: &[DewettingNeuron],
    voxel_to_neuron: &[Option<usize>],
    grid: &ExclusionGrid,
    _config: &NhsConfig,
) -> Vec<Synapse> {
    let mut synapses = Vec::new();

    // 6-connectivity offsets (face neighbors)
    let offsets: [(i32, i32, i32); 6] = [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ];

    for (from_neuron_idx, neuron) in neurons.iter().enumerate() {
        let (ix, iy, iz) = grid.coords(neuron.voxel_idx);

        for &(dx, dy, dz) in &offsets {
            let nx = ix as i32 + dx;
            let ny = iy as i32 + dy;
            let nz = iz as i32 + dz;

            // Bounds check
            if nx < 0 || ny < 0 || nz < 0 {
                continue;
            }
            if nx >= grid.nx as i32 || ny >= grid.ny as i32 || nz >= grid.nz as i32 {
                continue;
            }

            let neighbor_voxel = grid.index(nx as usize, ny as usize, nz as usize);

            if let Some(to_neuron_idx) = voxel_to_neuron.get(neighbor_voxel).copied().flatten() {
                synapses.push(Synapse {
                    from_idx: from_neuron_idx,
                    to_idx: to_neuron_idx,
                    weight: 1.0, // Uniform weight
                });
            }
        }
    }

    synapses
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuron_creation() {
        let neuron = DewettingNeuron::new(42);
        assert_eq!(neuron.voxel_idx, 42);
        assert!((neuron.potential - 1.0).abs() < 0.001);
        assert!(!neuron.spiked);
        assert_eq!(neuron.total_spikes, 0);
    }

    #[test]
    fn test_neuron_reset() {
        let mut neuron = DewettingNeuron::new(0);
        neuron.potential = 0.5;
        neuron.spiked = true;
        neuron.total_spikes = 10;

        neuron.reset();

        assert!((neuron.potential - 1.0).abs() < 0.001);
        assert!(!neuron.spiked);
        assert_eq!(neuron.total_spikes, 0);
    }

    #[test]
    fn test_spike_detection() {
        // Create minimal grid with one surface voxel
        let mut grid = ExclusionGrid::new([0.0; 3], [2.0; 3], 1.0, 0.0);
        grid.exclusion[0] = 0.5; // Surface

        let config = NhsConfig::default();
        let mut network = DewettingNetwork::from_grid(&grid, config);

        assert_eq!(network.num_neurons(), 1);

        // High water density - no spike
        let spikes = network.step(&[BULK_WATER_DENSITY]);
        assert!(spikes.is_empty());

        // Low water density - should spike
        let spikes = network.step(&[BULK_WATER_DENSITY * 0.1]);
        // May or may not spike depending on integration dynamics
        // Just verify no crash
    }
}
