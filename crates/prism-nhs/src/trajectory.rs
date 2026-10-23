//! Trajectory I/O for NHS ensemble generation and analysis
//!
//! Supports:
//! - Multi-model PDB output (ensemble snapshots)
//! - Regular interval trajectory saving
//! - Spike-triggered snapshot capture
//! - Metadata embedding (temperature, timestep, spike info)

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use crate::input::PrismPrepTopology;

/// Configuration for trajectory output during MD
#[derive(Debug, Clone)]
pub struct TrajectoryConfig {
    /// Save snapshot every N steps (0 = disabled)
    pub save_interval: i32,
    /// Save snapshot on spike detection
    pub save_on_spike: bool,
    /// Maximum snapshots to keep in memory before flushing
    pub max_memory_snapshots: usize,
    /// Output directory
    pub output_dir: String,
    /// Base name for output files
    pub base_name: String,
}

impl Default for TrajectoryConfig {
    fn default() -> Self {
        Self {
            save_interval: 1000, // Every 1000 steps = 2ps
            save_on_spike: true,
            max_memory_snapshots: 1000,
            output_dir: ".".to_string(),
            base_name: "trajectory".to_string(),
        }
    }
}

/// A single trajectory frame with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryFrame {
    /// Frame index (0-based)
    pub frame_idx: usize,
    /// MD timestep when captured
    pub timestep: i32,
    /// Temperature at capture (K)
    pub temperature: f32,
    /// Simulation time (ps)
    pub time_ps: f32,
    /// Atom positions [x0, y0, z0, x1, y1, z1, ...]
    pub positions: Vec<f32>,
    /// Was this frame triggered by a spike?
    pub spike_triggered: bool,
    /// Number of spikes at this frame (if spike-triggered)
    pub spike_count: Option<usize>,
    /// Spike voxel indices (if spike-triggered)
    pub spike_voxels: Option<Vec<usize>>,
    /// UV wavelength (nm) when frame was captured (spectroscopy mode only)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wavelength_nm: Option<f32>,
}

/// Trajectory writer for ensemble output
pub struct TrajectoryWriter {
    config: TrajectoryConfig,
    frames: Vec<TrajectoryFrame>,
    frames_written: usize,
    current_file_idx: usize,
}

impl TrajectoryWriter {
    /// Create new trajectory writer
    pub fn new(config: TrajectoryConfig) -> Result<Self> {
        // Create output directory
        fs::create_dir_all(&config.output_dir)
            .context("Failed to create trajectory output directory")?;

        Ok(Self {
            config,
            frames: Vec::new(),
            frames_written: 0,
            current_file_idx: 0,
        })
    }

    /// Add a frame to the trajectory
    pub fn add_frame(&mut self, frame: TrajectoryFrame) {
        self.frames.push(frame);

        // Flush if we've accumulated too many frames
        if self.frames.len() >= self.config.max_memory_snapshots {
            if let Err(e) = self.flush() {
                log::warn!("Failed to flush trajectory: {}", e);
            }
        }
    }

    /// Check if we should save at this timestep
    pub fn should_save(&self, timestep: i32) -> bool {
        if self.config.save_interval <= 0 {
            return false;
        }
        timestep % self.config.save_interval == 0
    }

    /// Flush frames to disk
    pub fn flush(&mut self) -> Result<()> {
        if self.frames.is_empty() {
            return Ok(());
        }

        let filename = format!(
            "{}/{}_part{:04}.frames.json",
            self.config.output_dir, self.config.base_name, self.current_file_idx
        );

        let file = File::create(&filename).context("Failed to create trajectory file")?;
        serde_json::to_writer_pretty(file, &self.frames)
            .context("Failed to write trajectory JSON")?;

        log::info!("Flushed {} frames to {}", self.frames.len(), filename);

        self.frames_written += self.frames.len();
        self.frames.clear();
        self.current_file_idx += 1;

        Ok(())
    }

    /// Finalize trajectory and write ensemble PDB
    pub fn finalize(&mut self, topology: &PrismPrepTopology) -> Result<TrajectoryStats> {
        // Flush remaining frames
        self.flush()?;

        // Write combined ensemble PDB
        let pdb_path = format!(
            "{}/{}_ensemble.pdb",
            self.config.output_dir, self.config.base_name
        );

        // Load all frames back and write PDB
        let all_frames = self.load_all_frames()?;
        write_ensemble_pdb(Path::new(&pdb_path), &all_frames, topology)?;

        // Write metadata JSON
        let meta_path = format!(
            "{}/{}_metadata.json",
            self.config.output_dir, self.config.base_name
        );
        let stats = TrajectoryStats {
            total_frames: all_frames.len(),
            spike_triggered_frames: all_frames.iter().filter(|f| f.spike_triggered).count(),
            interval_frames: all_frames.iter().filter(|f| !f.spike_triggered).count(),
            time_range_ps: (
                all_frames.first().map(|f| f.time_ps).unwrap_or(0.0),
                all_frames.last().map(|f| f.time_ps).unwrap_or(0.0),
            ),
            temperature_range: (
                all_frames
                    .iter()
                    .map(|f| f.temperature)
                    .fold(f32::INFINITY, f32::min),
                all_frames
                    .iter()
                    .map(|f| f.temperature)
                    .fold(f32::NEG_INFINITY, f32::max),
            ),
            ensemble_pdb: pdb_path.clone(),
        };

        let meta_file = File::create(&meta_path)?;
        serde_json::to_writer_pretty(meta_file, &stats)?;

        log::info!(
            "Trajectory finalized: {} frames, PDB: {}",
            stats.total_frames,
            pdb_path
        );

        Ok(stats)
    }

    /// Load all frames from disk
    fn load_all_frames(&self) -> Result<Vec<TrajectoryFrame>> {
        let mut all_frames = Vec::new();

        for idx in 0..self.current_file_idx {
            let filename = format!(
                "{}/{}_part{:04}.frames.json",
                self.config.output_dir, self.config.base_name, idx
            );

            if Path::new(&filename).exists() {
                let file = File::open(&filename)?;
                let frames: Vec<TrajectoryFrame> = serde_json::from_reader(file)?;
                all_frames.extend(frames);
            }
        }

        // Also include any unflushed frames
        all_frames.extend(self.frames.clone());

        Ok(all_frames)
    }

    /// Get number of frames collected
    pub fn frame_count(&self) -> usize {
        self.frames_written + self.frames.len()
    }
}

/// Statistics about a trajectory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryStats {
    pub total_frames: usize,
    pub spike_triggered_frames: usize,
    pub interval_frames: usize,
    pub time_range_ps: (f32, f32),
    pub temperature_range: (f32, f32),
    pub ensemble_pdb: String,
}

/// Write frames as multi-model PDB
pub fn write_ensemble_pdb(
    path: &Path,
    frames: &[TrajectoryFrame],
    topology: &PrismPrepTopology,
) -> Result<()> {
    let mut file = File::create(path).context("Failed to create ensemble PDB")?;

    // Write header
    writeln!(file, "REMARK   PRISM-CryoUV Ensemble Trajectory")?;
    writeln!(file, "REMARK   Frames: {}", frames.len())?;
    if let Some(first) = frames.first() {
        writeln!(
            file,
            "REMARK   Time range: {:.2} - {:.2} ps",
            first.time_ps,
            frames.last().map(|f| f.time_ps).unwrap_or(first.time_ps)
        )?;
    }

    for (model_idx, frame) in frames.iter().enumerate() {
        writeln!(file, "MODEL     {:>4}", model_idx + 1)?;
        writeln!(
            file,
            "REMARK   Timestep: {} Time: {:.2}ps Temp: {:.1}K Spike: {}",
            frame.timestep, frame.time_ps, frame.temperature, frame.spike_triggered
        )?;

        let positions = &frame.positions;
        let n_atoms = positions.len() / 3;

        for i in 0..n_atoms.min(topology.n_atoms) {
            let x = positions[i * 3];
            let y = positions[i * 3 + 1];
            let z = positions[i * 3 + 2];

            let atom_name = topology
                .atom_names
                .get(i)
                .map(|s| s.as_str())
                .unwrap_or("CA");
            let residue_name = topology
                .residue_names
                .get(i)
                .map(|s| s.as_str())
                .unwrap_or("UNK");
            let residue_id = topology.residue_ids.get(i).copied().unwrap_or(1);
            let chain_id = topology
                .chain_ids
                .get(i)
                .map(|s| s.chars().next().unwrap_or('A'))
                .unwrap_or('A');
            let element_sym = topology.elements.get(i).map(|s| s.as_str()).unwrap_or("C");

            // B-factor encodes temperature, occupancy encodes spike status
            let b_factor = frame.temperature;
            let occupancy = if frame.spike_triggered { 1.0 } else { 0.5 };

            writeln!(file,
                "ATOM  {:>5} {:>4} {:>3} {}{:>4}    {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}          {:>2}",
                (i + 1) % 100000,
                format_atom_name(atom_name),
                residue_name,
                chain_id,
                residue_id % 10000,
                x, y, z,
                occupancy,
                b_factor,
                element_sym
            )?;
        }

        writeln!(file, "ENDMDL")?;
    }

    writeln!(file, "END")?;

    Ok(())
}

/// Load trajectory frames from ensemble PDB
pub fn load_ensemble_pdb(path: &Path) -> Result<Vec<TrajectoryFrame>> {
    let file = File::open(path).context("Failed to open ensemble PDB")?;
    let reader = BufReader::new(file);

    let mut frames = Vec::new();
    let mut current_positions: Vec<f32> = Vec::new();
    let mut current_frame_idx = 0;
    let mut current_timestep = 0;
    let mut current_time = 0.0f32;
    let mut current_temp = 300.0f32;
    let mut current_spike = false;

    for line in reader.lines() {
        let line = line?;

        if line.starts_with("MODEL") {
            current_positions.clear();
        } else if line.starts_with("REMARK") && line.contains("Timestep:") {
            // Parse: "REMARK   Timestep: 1000 Time: 2.00ps Temp: 150.0K Spike: false"
            if let Some(ts_part) = line.split("Timestep:").nth(1) {
                if let Some(ts_str) = ts_part.split_whitespace().next() {
                    current_timestep = ts_str.parse().unwrap_or(0);
                }
            }
            if let Some(time_part) = line.split("Time:").nth(1) {
                if let Some(time_str) = time_part.split("ps").next() {
                    current_time = time_str.trim().parse().unwrap_or(0.0);
                }
            }
            if let Some(temp_part) = line.split("Temp:").nth(1) {
                if let Some(temp_str) = temp_part.split("K").next() {
                    current_temp = temp_str.trim().parse().unwrap_or(300.0);
                }
            }
            current_spike = line.contains("Spike: true");
        } else if line.starts_with("ATOM") || line.starts_with("HETATM") {
            // Parse coordinates
            if line.len() >= 54 {
                let x: f32 = line[30..38].trim().parse().unwrap_or(0.0);
                let y: f32 = line[38..46].trim().parse().unwrap_or(0.0);
                let z: f32 = line[46..54].trim().parse().unwrap_or(0.0);
                current_positions.push(x);
                current_positions.push(y);
                current_positions.push(z);
            }
        } else if line.starts_with("ENDMDL") {
            if !current_positions.is_empty() {
                frames.push(TrajectoryFrame {
                    frame_idx: current_frame_idx,
                    timestep: current_timestep,
                    temperature: current_temp,
                    time_ps: current_time,
                    positions: current_positions.clone(),
                    spike_triggered: current_spike,
                    spike_count: None,
                    spike_voxels: None,
                    wavelength_nm: None, // Not available when loading from PDB
                });
                current_frame_idx += 1;
            }
        }
    }

    log::info!("Loaded {} frames from {}", frames.len(), path.display());
    Ok(frames)
}

/// Format atom name for PDB (4 characters, special alignment rules)
fn format_atom_name(name: &str) -> String {
    if name.len() >= 4 {
        name[..4].to_string()
    } else if name.len() == 1 {
        format!(" {}  ", name)
    } else if name.len() == 2 {
        format!(" {} ", name)
    } else {
        format!(" {}", name)
    }
}

/// Get element symbol from atomic number
fn element_symbol(atomic_num: u8) -> &'static str {
    match atomic_num {
        1 => "H",
        6 => "C",
        7 => "N",
        8 => "O",
        16 => "S",
        15 => "P",
        _ => "X",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_atom_name() {
        assert_eq!(format_atom_name("CA"), " CA ");
        assert_eq!(format_atom_name("N"), " N  ");
        assert_eq!(format_atom_name("OXT"), " OXT");
    }

    #[test]
    fn test_element_symbol() {
        assert_eq!(element_symbol(1), "H");
        assert_eq!(element_symbol(6), "C");
        assert_eq!(element_symbol(7), "N");
        assert_eq!(element_symbol(8), "O");
    }

    #[test]
    fn test_trajectory_config_default() {
        let config = TrajectoryConfig::default();
        assert_eq!(config.save_interval, 1000);
        assert!(config.save_on_spike);
    }
}
