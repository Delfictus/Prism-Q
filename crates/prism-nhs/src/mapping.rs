//! NHS Site Mapping Module
//!
//! Provides all-atom AMBER correlation for NHS spike hotspots.
//! Enables comparative evaluation between:
//! - Cryo vs non-cryo conditions
//! - Different structures
//! - Different simulation protocols
//!
//! ## Naming Conventions
//!
//! | Field | Format | Example |
//! |-------|--------|---------|
//! | site_id | `NHS_{pdb}_{chain}_{resnum}` | `NHS_6OIM_A_12` |
//! | hotspot_id | `HS_{voxel_idx}` | `HS_20213` |
//! | condition | `{temp_k}K_{protocol}` | `80K_cryo`, `300K_standard` |
//! | comparison_id | `{site_id}_{condition}` | `NHS_6OIM_A_12_80K_cryo` |

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Standardized site identification following AMBER conventions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NhsSiteId {
    /// PDB ID (e.g., "6OIM")
    pub pdb_id: String,
    /// Chain identifier (e.g., "A")
    pub chain_id: String,
    /// Residue number
    pub residue_number: i32,
    /// Residue name (e.g., "GLY", "ALA")
    pub residue_name: String,
    /// Full site identifier
    pub full_id: String,
}

impl NhsSiteId {
    pub fn new(pdb_id: &str, chain_id: &str, residue_number: i32, residue_name: &str) -> Self {
        let full_id = format!(
            "NHS_{}_{}_{}_{}",
            pdb_id, chain_id, residue_number, residue_name
        );
        Self {
            pdb_id: pdb_id.to_string(),
            chain_id: chain_id.to_string(),
            residue_number,
            residue_name: residue_name.to_string(),
            full_id,
        }
    }
}

/// Experimental condition for comparative analysis
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExperimentalCondition {
    /// Temperature in Kelvin
    pub temperature_k: f32,
    /// Protocol type
    pub protocol: ProtocolType,
    /// Formatted condition string (e.g., "80K_cryo")
    pub condition_id: String,
}

/// Protocol type for NHS simulation
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ProtocolType {
    /// Standard constant temperature
    Standard,
    /// Cryogenic freeze-probe-warm
    Cryo,
    /// Cryogenic with UV probing
    CryoUvProbe,
    /// Thermal annealing
    Annealing,
}

impl ProtocolType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Cryo => "cryo",
            Self::CryoUvProbe => "cryo_uv",
            Self::Annealing => "anneal",
        }
    }
}

impl ExperimentalCondition {
    pub fn new(temperature_k: f32, protocol: ProtocolType) -> Self {
        let condition_id = format!("{:.0}K_{}", temperature_k, protocol.as_str());
        Self {
            temperature_k,
            protocol,
            condition_id,
        }
    }

    pub fn cryo(start_temp_k: f32) -> Self {
        Self::new(start_temp_k, ProtocolType::Cryo)
    }

    pub fn standard(temp_k: f32) -> Self {
        Self::new(temp_k, ProtocolType::Standard)
    }
}

/// Spike hotspot with AMBER residue mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MappedHotspot {
    /// Voxel index in NHS grid
    pub voxel_idx: i32,
    /// Position in Angstroms
    pub position_angstrom: [f32; 3],
    /// Total spike count
    pub spike_count: usize,
    /// Nearest residues (within mapping_radius)
    pub nearby_residues: Vec<NearbyResidue>,
    /// Primary residue (closest or most significant)
    pub primary_residue: Option<NhsSiteId>,
    /// Experimental condition
    pub condition: ExperimentalCondition,
    /// Hotspot ID for cross-reference
    pub hotspot_id: String,
    /// Full comparison ID
    pub comparison_id: String,
}

/// Residue near a spike hotspot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NearbyResidue {
    /// Site identifier
    pub site_id: NhsSiteId,
    /// Distance from hotspot center (Angstroms)
    pub distance_angstrom: f32,
    /// Number of atoms within radius
    pub atoms_in_radius: usize,
    /// Is this an aromatic residue (UV target)?
    pub is_aromatic: bool,
    /// Is this a known binding site residue?
    pub is_known_binding: bool,
}

/// NHS Site Mapper - correlates voxels with AMBER topology
pub struct NhsSiteMapper {
    /// PDB identifier
    pdb_id: String,
    /// Atom positions (flattened x,y,z)
    atom_positions: Vec<f32>,
    /// Atom to residue mapping
    atom_residue_ids: Vec<i32>,
    /// Residue names
    residue_names: Vec<String>,
    /// Chain IDs per residue
    chain_ids: Vec<String>,
    /// Number of atoms
    n_atoms: usize,
    /// Number of residues
    n_residues: usize,
    /// Mapping radius (Angstroms)
    mapping_radius: f32,
    /// Known aromatic residue types
    aromatic_types: Vec<&'static str>,
}

impl NhsSiteMapper {
    /// Create mapper from PRISM-PREP topology
    pub fn from_topology(
        pdb_id: &str,
        positions: &[f32],
        residue_ids: &[i32],
        residue_names: &[String],
        chain_ids: &[String],
        mapping_radius: f32,
    ) -> Self {
        let n_atoms = positions.len() / 3;
        let n_residues = *residue_ids.iter().max().unwrap_or(&0) as usize + 1;

        Self {
            pdb_id: pdb_id.to_string(),
            atom_positions: positions.to_vec(),
            atom_residue_ids: residue_ids.to_vec(),
            residue_names: residue_names.to_vec(),
            chain_ids: chain_ids.to_vec(),
            n_atoms,
            n_residues,
            mapping_radius,
            aromatic_types: vec!["TRP", "TYR", "PHE", "HIS"],
        }
    }

    /// Map a spike hotspot to nearby residues
    pub fn map_hotspot(
        &self,
        voxel_idx: i32,
        position: [f32; 3],
        spike_count: usize,
        condition: ExperimentalCondition,
    ) -> MappedHotspot {
        let mut nearby_residues = Vec::new();
        let mut residue_min_dist: HashMap<i32, (f32, usize)> = HashMap::new();

        // Find all atoms within mapping radius
        for atom_idx in 0..self.n_atoms {
            let ax = self.atom_positions[atom_idx * 3];
            let ay = self.atom_positions[atom_idx * 3 + 1];
            let az = self.atom_positions[atom_idx * 3 + 2];

            let dx = position[0] - ax;
            let dy = position[1] - ay;
            let dz = position[2] - az;
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();

            if dist <= self.mapping_radius {
                let res_id = self.atom_residue_ids[atom_idx];
                let entry = residue_min_dist.entry(res_id).or_insert((dist, 0));
                if dist < entry.0 {
                    entry.0 = dist;
                }
                entry.1 += 1;
            }
        }

        // Build nearby residue list
        for (res_id, (min_dist, atom_count)) in &residue_min_dist {
            let res_idx = *res_id as usize;
            if res_idx >= self.residue_names.len() {
                continue;
            }

            let res_name = &self.residue_names[res_idx];
            let chain_id = if res_idx < self.chain_ids.len() {
                &self.chain_ids[res_idx]
            } else {
                "A"
            };

            let site_id = NhsSiteId::new(&self.pdb_id, chain_id, *res_id, res_name);
            let is_aromatic = self.aromatic_types.contains(&res_name.as_str());

            nearby_residues.push(NearbyResidue {
                site_id,
                distance_angstrom: *min_dist,
                atoms_in_radius: *atom_count,
                is_aromatic,
                is_known_binding: false, // Can be set later with binding site info
            });
        }

        // Sort by distance
        nearby_residues.sort_by(|a, b| {
            a.distance_angstrom
                .partial_cmp(&b.distance_angstrom)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Get primary residue (closest)
        let primary_residue = nearby_residues.first().map(|r| r.site_id.clone());

        // Build IDs
        let hotspot_id = format!("HS_{}", voxel_idx);
        let comparison_id = if let Some(ref primary) = primary_residue {
            format!("{}_{}", primary.full_id, condition.condition_id)
        } else {
            format!("{}_{}", hotspot_id, condition.condition_id)
        };

        MappedHotspot {
            voxel_idx,
            position_angstrom: position,
            spike_count,
            nearby_residues,
            primary_residue,
            condition,
            hotspot_id,
            comparison_id,
        }
    }

    /// Map all spike hotspots from accumulator
    pub fn map_all_hotspots(
        &self,
        spike_accumulator: &HashMap<i32, (usize, [f32; 3])>,
        condition: ExperimentalCondition,
        min_spikes: usize,
    ) -> Vec<MappedHotspot> {
        spike_accumulator
            .iter()
            .filter(|(_, (count, _))| *count >= min_spikes)
            .map(|(voxel_idx, (count, pos))| {
                self.map_hotspot(*voxel_idx, *pos, *count, condition.clone())
            })
            .collect()
    }
}

/// Comparative analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparativeAnalysis {
    /// Structure identifier
    pub pdb_id: String,
    /// Conditions compared
    pub conditions: Vec<ExperimentalCondition>,
    /// Hotspots by condition
    pub hotspots_by_condition: HashMap<String, Vec<MappedHotspot>>,
    /// Sites that appear in multiple conditions (robust)
    pub robust_sites: Vec<RobustSite>,
    /// Sites unique to cryo (cryo-specific)
    pub cryo_specific: Vec<MappedHotspot>,
    /// Sites unique to non-cryo (thermal-activated)
    pub thermal_activated: Vec<MappedHotspot>,
    /// Calibration metrics
    pub calibration: CalibrationMetrics,
}

/// Site that appears across multiple conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustSite {
    /// Primary site ID
    pub site_id: NhsSiteId,
    /// Position (averaged across conditions)
    pub position_angstrom: [f32; 3],
    /// Spike counts per condition
    pub spike_counts: HashMap<String, usize>,
    /// Conditions where this site appears
    pub present_in: Vec<String>,
    /// Robustness score (0-1)
    pub robustness_score: f32,
}

/// Calibration metrics for cryo vs non-cryo
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CalibrationMetrics {
    /// Number of sites detected in cryo
    pub n_cryo_sites: usize,
    /// Number of sites detected at room temp
    pub n_room_sites: usize,
    /// Overlap ratio (sites in both / total unique)
    pub overlap_ratio: f32,
    /// Cryo specificity (cryo-only / cryo-total)
    pub cryo_specificity: f32,
    /// Thermal activation ratio (room-only / room-total)
    pub thermal_activation: f32,
    /// Average spike count ratio (cryo / room)
    pub spike_ratio: f32,
}

/// Compare hotspots across conditions
pub fn compare_conditions(
    pdb_id: &str,
    hotspots_by_condition: HashMap<String, Vec<MappedHotspot>>,
    distance_threshold: f32,
) -> ComparativeAnalysis {
    let conditions: Vec<_> = hotspots_by_condition
        .values()
        .filter_map(|v| v.first().map(|h| h.condition.clone()))
        .collect();

    // Find sites present in multiple conditions
    let mut site_appearances: HashMap<String, Vec<(String, MappedHotspot)>> = HashMap::new();

    for (cond_id, hotspots) in &hotspots_by_condition {
        for hs in hotspots {
            if let Some(ref primary) = hs.primary_residue {
                site_appearances
                    .entry(primary.full_id.clone())
                    .or_default()
                    .push((cond_id.clone(), hs.clone()));
            }
        }
    }

    // Build robust sites (appear in 2+ conditions)
    let robust_sites: Vec<_> = site_appearances
        .iter()
        .filter(|(_, appearances)| appearances.len() > 1)
        .map(|(site_id, appearances)| {
            let first_hs = &appearances[0].1;
            let spike_counts: HashMap<_, _> = appearances
                .iter()
                .map(|(cond, hs)| (cond.clone(), hs.spike_count))
                .collect();
            let present_in: Vec<_> = appearances.iter().map(|(cond, _)| cond.clone()).collect();
            let robustness = present_in.len() as f32 / conditions.len() as f32;

            RobustSite {
                site_id: first_hs.primary_residue.clone().unwrap(),
                position_angstrom: first_hs.position_angstrom,
                spike_counts,
                present_in,
                robustness_score: robustness,
            }
        })
        .collect();

    // Find cryo-specific and thermal-activated sites
    let cryo_condition = conditions.iter().find(|c| c.protocol == ProtocolType::Cryo);
    let room_condition = conditions
        .iter()
        .find(|c| c.protocol == ProtocolType::Standard && c.temperature_k >= 290.0);

    let cryo_specific = if let Some(cryo_cond) = cryo_condition {
        hotspots_by_condition
            .get(&cryo_cond.condition_id)
            .map(|hs| {
                hs.iter()
                    .filter(|h| {
                        h.primary_residue.as_ref().map_or(true, |p| {
                            !robust_sites.iter().any(|r| r.site_id.full_id == p.full_id)
                        })
                    })
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    } else {
        Vec::new()
    };

    let thermal_activated = if let Some(room_cond) = room_condition {
        hotspots_by_condition
            .get(&room_cond.condition_id)
            .map(|hs| {
                hs.iter()
                    .filter(|h| {
                        h.primary_residue.as_ref().map_or(true, |p| {
                            !robust_sites.iter().any(|r| r.site_id.full_id == p.full_id)
                        })
                    })
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    } else {
        Vec::new()
    };

    // Calculate calibration metrics
    let n_cryo = cryo_condition
        .and_then(|c| hotspots_by_condition.get(&c.condition_id))
        .map(|v| v.len())
        .unwrap_or(0);
    let n_room = room_condition
        .and_then(|c| hotspots_by_condition.get(&c.condition_id))
        .map(|v| v.len())
        .unwrap_or(0);
    let n_robust = robust_sites.len();
    let total_unique = n_cryo + n_room - n_robust;

    let calibration = CalibrationMetrics {
        n_cryo_sites: n_cryo,
        n_room_sites: n_room,
        overlap_ratio: if total_unique > 0 {
            n_robust as f32 / total_unique as f32
        } else {
            0.0
        },
        cryo_specificity: if n_cryo > 0 {
            cryo_specific.len() as f32 / n_cryo as f32
        } else {
            0.0
        },
        thermal_activation: if n_room > 0 {
            thermal_activated.len() as f32 / n_room as f32
        } else {
            0.0
        },
        spike_ratio: 1.0, // TODO: Calculate from actual data
    };

    ComparativeAnalysis {
        pdb_id: pdb_id.to_string(),
        conditions,
        hotspots_by_condition,
        robust_sites,
        cryo_specific,
        thermal_activated,
        calibration,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_site_id_format() {
        let site = NhsSiteId::new("6OIM", "A", 12, "GLY");
        assert_eq!(site.full_id, "NHS_6OIM_A_12_GLY");
    }

    #[test]
    fn test_condition_format() {
        let cryo = ExperimentalCondition::cryo(80.0);
        assert_eq!(cryo.condition_id, "80K_cryo");

        let standard = ExperimentalCondition::standard(300.0);
        assert_eq!(standard.condition_id, "300K_standard");
    }
}
