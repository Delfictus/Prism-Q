//! Stage 1B: Structure Composition Analysis
//!
//! Analyzes protein structure composition for:
//! - Unambiguous chain/residue identification across pipeline
//! - Memory tier classification for batch scheduling
//! - Computational profile estimation for optimal GPU utilization
//!
//! This stage runs after PRISM-PREP (Stage 1) and before equilibration (Stage 2A).
//!
//! ## Benefits
//! - Deterministic batch scheduling based on structure size and complexity
//! - Unambiguous residue identification for reproducible results
//! - Optimal GPU memory utilization by grouping similar structures

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::input::PrismPrepTopology;

/// Existing CA contact-map distance predicate, in Angstroms (strictly less).
pub const CA_CONTACT_THRESHOLD_ANGSTROM: f64 = 8.0;

// ═══════════════════════════════════════════════════════════════════════════════
// Unambiguous Residue Identification
// ═══════════════════════════════════════════════════════════════════════════════

/// Unambiguous residue identifier
///
/// Handles PDB quirks like insertion codes, non-sequential numbering,
/// and alternate conformations. Two ResidueKeys are equal iff they refer
/// to the same physical residue instance.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResidueKey {
    /// Chain identifier (A, B, C, etc.)
    pub chain_id: String,
    /// Residue sequence number (from PDB, may have gaps, can be negative)
    pub residue_number: i32,
    /// Insertion code for non-standard numbering (A, B, etc.)
    pub insertion_code: Option<char>,
    /// Alternate location indicator (A/B conformers in crystal structures)
    /// None = single conformer or primary conformer selected
    pub alt_loc: Option<char>,
}

impl ResidueKey {
    /// Create a new residue key with full specification
    pub fn new(
        chain_id: impl Into<String>,
        residue_number: i32,
        insertion_code: Option<char>,
        alt_loc: Option<char>,
    ) -> Self {
        Self {
            chain_id: chain_id.into(),
            residue_number,
            insertion_code,
            alt_loc,
        }
    }

    /// Create from chain and residue number (no insertion code or alt_loc)
    pub fn simple(chain_id: impl Into<String>, residue_number: i32) -> Self {
        Self::new(chain_id, residue_number, None, None)
    }

    /// Create with insertion code but no alt_loc
    pub fn with_insertion(
        chain_id: impl Into<String>,
        residue_number: i32,
        insertion_code: char,
    ) -> Self {
        Self::new(chain_id, residue_number, Some(insertion_code), None)
    }

    /// Create with alt_loc for alternate conformers
    pub fn with_alt_loc(chain_id: impl Into<String>, residue_number: i32, alt_loc: char) -> Self {
        Self::new(chain_id, residue_number, None, Some(alt_loc))
    }

    /// Format as string for display (PDB-style)
    pub fn to_pdb_string(&self) -> String {
        let mut s = format!("{}:{}", self.chain_id, self.residue_number);
        if let Some(ins) = self.insertion_code {
            s.push(ins);
        }
        if let Some(alt) = self.alt_loc {
            s.push_str(&format!("({})", alt));
        }
        s
    }

    /// Check if this is an alternate conformer
    pub fn is_alternate_conformer(&self) -> bool {
        self.alt_loc.is_some()
    }
}

impl std::fmt::Display for ResidueKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_pdb_string())
    }
}

/// Information about a residue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidueInfo {
    /// Residue name (ALA, GLY, etc.)
    pub name: String,
    /// Sequential index in topology (0-based)
    pub topology_index: usize,
    /// Number of atoms in this residue
    pub atom_count: usize,
    /// First atom index in topology
    pub first_atom_index: usize,
    /// Last atom index (exclusive) in topology
    pub last_atom_index: usize,
    /// Is this an aromatic residue (TRP, TYR, PHE, HIS)?
    pub is_aromatic: bool,
    /// Is this a catalytic residue (GLU, ASP, HIS, SER, CYS, LYS)?
    pub is_catalytic: bool,
    /// Is this a hydrophobic residue?
    pub is_hydrophobic: bool,
}

// ═══════════════════════════════════════════════════════════════════════════════
// GPU Memory Profile - Detailed Allocation Tracking
// ═══════════════════════════════════════════════════════════════════════════════

/// Detailed GPU memory allocation profile
///
/// Provides accurate memory estimation based on actual GPU buffer requirements
/// rather than naive atom count heuristics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryProfile {
    /// Coordinate buffers: 3 coords × 4 bytes × n_atoms
    pub coordinates_bytes: usize,
    /// Velocity buffers: 3 velocities × 4 bytes × n_atoms
    pub velocities_bytes: usize,
    /// Force buffers: 3 forces × 4 bytes × n_atoms
    pub forces_bytes: usize,
    /// Neighbor list storage (varies with density)
    pub neighbor_list_bytes: usize,
    /// Spike buffer for detection (dynamic based on aromatic content)
    pub spike_buffer_bytes: usize,
    /// RT-core clustering overhead (BVH + metadata)
    pub clustering_overhead_bytes: usize,
    /// Topology data (bonds, angles, dihedrals)
    pub topology_bytes: usize,
    /// Miscellaneous overhead (scratch buffers, etc.)
    pub overhead_bytes: usize,
}

impl MemoryProfile {
    /// Calculate memory profile from structure metrics
    pub fn calculate(
        n_atoms: usize,
        n_bonds: usize,
        n_angles: usize,
        n_dihedrals: usize,
        n_aromatics: usize,
        estimated_density: f32, // atoms per Å³
    ) -> Self {
        // Core MD buffers: positions, velocities, forces
        let coordinates_bytes = n_atoms * 3 * 4; // float3
        let velocities_bytes = n_atoms * 3 * 4;
        let forces_bytes = n_atoms * 3 * 4;

        // Neighbor list: depends on density
        // Typical: 50-200 neighbors per atom for 10Å cutoff
        let avg_neighbors = (estimated_density * 4188.79).min(200.0) as usize; // 4/3 π r³ for r=10Å
        let avg_neighbors = avg_neighbors.max(50); // Floor for sparse structures
        let neighbor_list_bytes = n_atoms * avg_neighbors * 4; // u32 indices

        // Spike buffer: ~100 spikes per aromatic per 1000 steps, 16 bytes each
        let estimated_spikes = n_aromatics * 100;
        let spike_buffer_bytes = estimated_spikes * 16;

        // Clustering overhead: BVH nodes + cluster metadata
        // BVH is roughly 2× point count, each node ~32 bytes
        let clustering_overhead_bytes = estimated_spikes * 2 * 32;

        // Topology: bonds(3×4), angles(4×4), dihedrals(6×4)
        let topology_bytes = n_bonds * 12 + n_angles * 16 + n_dihedrals * 24;

        // Overhead: ~10% of total for scratch buffers, alignment, etc.
        let subtotal = coordinates_bytes
            + velocities_bytes
            + forces_bytes
            + neighbor_list_bytes
            + spike_buffer_bytes
            + clustering_overhead_bytes
            + topology_bytes;
        let overhead_bytes = subtotal / 10;

        Self {
            coordinates_bytes,
            velocities_bytes,
            forces_bytes,
            neighbor_list_bytes,
            spike_buffer_bytes,
            clustering_overhead_bytes,
            topology_bytes,
            overhead_bytes,
        }
    }

    /// Total GPU memory in bytes
    pub fn total_bytes(&self) -> usize {
        self.coordinates_bytes
            + self.velocities_bytes
            + self.forces_bytes
            + self.neighbor_list_bytes
            + self.spike_buffer_bytes
            + self.clustering_overhead_bytes
            + self.topology_bytes
            + self.overhead_bytes
    }

    /// Total GPU memory in MB
    pub fn total_mb(&self) -> usize {
        self.total_bytes() / (1024 * 1024)
    }

    /// Simple estimate from atom count (fallback)
    pub fn estimate_from_atoms(n_atoms: usize) -> Self {
        // Use typical ratios
        Self::calculate(
            n_atoms,
            n_atoms,      // ~1 bond per atom
            n_atoms * 2,  // ~2 angles per atom
            n_atoms * 4,  // ~4 dihedrals per atom
            n_atoms / 20, // ~5% aromatics
            0.01,         // typical protein density
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Memory Tiers for Batch Scheduling
// ═══════════════════════════════════════════════════════════════════════════════

/// Memory tier for GPU batch scheduling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryTier {
    /// < 500 MB - can run 4+ structures concurrently
    Small,
    /// 500-1500 MB - can run 2-3 structures concurrently
    Medium,
    /// > 1500 MB - run 1 at a time to avoid OOM
    Large,
}

impl MemoryTier {
    /// Classify based on atom count (simple heuristic)
    pub fn from_atom_count(n_atoms: usize) -> Self {
        let profile = MemoryProfile::estimate_from_atoms(n_atoms);
        Self::from_memory_mb(profile.total_mb())
    }

    /// Classify based on actual memory profile
    pub fn from_memory_profile(profile: &MemoryProfile) -> Self {
        Self::from_memory_mb(profile.total_mb())
    }

    /// Classify based on memory in MB
    ///
    /// Thresholds based on detailed memory profiling of real structures:
    /// - Small: <30 MB (~35K atoms) - can run 4 concurrently on 16GB GPU
    /// - Medium: 30-80 MB (~35K-100K atoms) - can run 2-3 concurrently
    /// - Large: >80 MB (>100K atoms) - run sequentially
    pub fn from_memory_mb(memory_mb: usize) -> Self {
        if memory_mb < 30 {
            MemoryTier::Small
        } else if memory_mb < 80 {
            MemoryTier::Medium
        } else {
            MemoryTier::Large
        }
    }

    /// Maximum concurrent structures for this tier (assuming 16GB GPU)
    pub fn max_concurrent(&self) -> u8 {
        match self {
            MemoryTier::Small => 4,
            MemoryTier::Medium => 2,
            MemoryTier::Large => 1,
        }
    }

    /// Maximum concurrent for given total GPU memory (MB)
    pub fn max_concurrent_for_memory(&self, total_gpu_mb: usize) -> u8 {
        let per_structure = self.estimated_memory_mb();
        ((total_gpu_mb / per_structure) as u8).max(1)
    }

    /// Estimated GPU memory per structure (MB)
    /// Typical memory usage for this tier (conservative estimate for scheduling)
    pub fn estimated_memory_mb(&self) -> usize {
        match self {
            MemoryTier::Small => 20,  // ~20K atoms typical
            MemoryTier::Medium => 50, // ~60K atoms typical
            MemoryTier::Large => 150, // ~150K atoms typical
        }
    }
}

impl std::fmt::Display for MemoryTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryTier::Small => write!(f, "Small (<15K atoms)"),
            MemoryTier::Medium => write!(f, "Medium (15-30K atoms)"),
            MemoryTier::Large => write!(f, "Large (>30K atoms)"),
        }
    }
}

/// Complexity tier based on structural features
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComplexityTier {
    /// Simple structure: single domain, few aromatics
    Simple,
    /// Moderate complexity: multi-domain or moderate aromatics
    Moderate,
    /// Complex: many domains, high aromatic density, membrane protein
    Complex,
}

impl ComplexityTier {
    /// Classify based on structure features
    pub fn from_features(
        n_chains: usize,
        aromatic_density: f32,
        has_membrane_like_features: bool,
    ) -> Self {
        if has_membrane_like_features {
            return ComplexityTier::Complex;
        }

        if n_chains >= 4 || aromatic_density > 0.15 {
            ComplexityTier::Complex
        } else if n_chains >= 2 || aromatic_density > 0.08 {
            ComplexityTier::Moderate
        } else {
            ComplexityTier::Simple
        }
    }
}

/// Spike density tier (analysis workload)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpikeDensityTier {
    /// Low spike density: <500 spikes expected
    Low,
    /// Medium spike density: 500-5000 spikes expected
    Medium,
    /// High spike density: >5000 spikes expected
    High,
}

impl SpikeDensityTier {
    /// Classify based on aromatic count and simulation steps
    pub fn from_aromatics(n_aromatics: usize, steps: usize) -> Self {
        // Empirical: ~10-20 spikes per aromatic per 1000 steps
        let estimated_spikes = n_aromatics * steps / 100;
        Self::from_spike_count(estimated_spikes)
    }

    /// Classify based on expected spike count
    pub fn from_spike_count(spikes: usize) -> Self {
        if spikes < 500 {
            SpikeDensityTier::Low
        } else if spikes < 5000 {
            SpikeDensityTier::Medium
        } else {
            SpikeDensityTier::High
        }
    }

    /// Clustering complexity multiplier
    pub fn clustering_complexity(&self) -> f32 {
        match self {
            SpikeDensityTier::Low => 1.0,
            SpikeDensityTier::Medium => 2.0,
            SpikeDensityTier::High => 4.0,
        }
    }
}

/// Multi-dimensional batch compatibility
///
/// Structures are compatible for batching if they have similar profiles
/// across all dimensions, not just memory size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BatchCompatibility {
    /// Memory tier (GPU memory pressure)
    pub memory_tier: MemoryTier,
    /// Dynamics complexity (simulation workload)
    pub dynamics_complexity: ComplexityTier,
    /// Spike density (analysis workload)
    pub spike_density_tier: SpikeDensityTier,
}

impl BatchCompatibility {
    /// Create new batch compatibility profile
    pub fn new(
        memory_tier: MemoryTier,
        dynamics_complexity: ComplexityTier,
        spike_density_tier: SpikeDensityTier,
    ) -> Self {
        Self {
            memory_tier,
            dynamics_complexity,
            spike_density_tier,
        }
    }

    /// Generate batch group ID (structures with same ID can be batched)
    pub fn batch_group_id(&self) -> u32 {
        // Encode as: memory(0-2) * 100 + complexity(0-2) * 10 + spike_density(0-2)
        let mem = match self.memory_tier {
            MemoryTier::Small => 0,
            MemoryTier::Medium => 1,
            MemoryTier::Large => 2,
        };
        let comp = match self.dynamics_complexity {
            ComplexityTier::Simple => 0,
            ComplexityTier::Moderate => 1,
            ComplexityTier::Complex => 2,
        };
        let spike = match self.spike_density_tier {
            SpikeDensityTier::Low => 0,
            SpikeDensityTier::Medium => 1,
            SpikeDensityTier::High => 2,
        };
        mem * 100 + comp * 10 + spike
    }

    /// Check if two profiles are compatible for batching
    pub fn is_compatible(&self, other: &BatchCompatibility) -> bool {
        // Must match on memory tier (hard constraint)
        // Complexity and spike density can differ by one tier
        self.memory_tier == other.memory_tier
            && ((self.dynamics_complexity as i32) - (other.dynamics_complexity as i32)).abs() <= 1
            && ((self.spike_density_tier as i32) - (other.spike_density_tier as i32)).abs() <= 1
    }

    /// Estimated relative workload (for load balancing)
    pub fn estimated_workload(&self) -> f32 {
        let mem_factor = match self.memory_tier {
            MemoryTier::Small => 1.0,
            MemoryTier::Medium => 2.0,
            MemoryTier::Large => 4.0,
        };
        let comp_factor = match self.dynamics_complexity {
            ComplexityTier::Simple => 1.0,
            ComplexityTier::Moderate => 1.5,
            ComplexityTier::Complex => 2.5,
        };
        let spike_factor = self.spike_density_tier.clustering_complexity();

        mem_factor * comp_factor * spike_factor
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Structure Composition
// ═══════════════════════════════════════════════════════════════════════════════

/// Chain boundary information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainInfo {
    /// Chain identifier
    pub chain_id: String,
    /// First residue index (topology)
    pub first_residue: usize,
    /// Last residue index (topology, exclusive)
    pub last_residue: usize,
    /// Number of residues
    pub residue_count: usize,
    /// Number of atoms
    pub atom_count: usize,
}

/// Complete structure composition analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructureComposition {
    /// Structure name/identifier
    pub structure_name: String,

    /// Source file path
    pub source_path: String,

    // ─────────────────────────────────────────────────────────────────────────
    // Unambiguous Identification
    // ─────────────────────────────────────────────────────────────────────────
    /// Residue registry: maps ResidueKey -> ResidueInfo
    pub residue_registry: HashMap<ResidueKey, ResidueInfo>,

    /// Chain information
    pub chains: Vec<ChainInfo>,

    // ─────────────────────────────────────────────────────────────────────────
    // Basic Metrics
    // ─────────────────────────────────────────────────────────────────────────
    /// Total atoms
    pub n_atoms: usize,

    /// Total residues
    pub n_residues: usize,

    /// Number of chains
    pub n_chains: usize,

    // ─────────────────────────────────────────────────────────────────────────
    // Aromatic Analysis
    // ─────────────────────────────────────────────────────────────────────────
    /// Number of aromatic residues (TRP, TYR, PHE, HIS)
    pub n_aromatic_residues: usize,

    /// Aromatic density (aromatic_residues / total_residues)
    pub aromatic_density: f32,

    /// Aromatic residue keys for UV perturbation
    pub aromatic_residue_keys: Vec<ResidueKey>,

    // ─────────────────────────────────────────────────────────────────────────
    // Catalytic Analysis
    // ─────────────────────────────────────────────────────────────────────────
    /// Number of catalytic residues (GLU, ASP, HIS, SER, CYS, LYS)
    pub n_catalytic_residues: usize,

    /// Catalytic residue density
    pub catalytic_density: f32,

    // ─────────────────────────────────────────────────────────────────────────
    // Hydrophobic Analysis
    // ─────────────────────────────────────────────────────────────────────────
    /// Number of hydrophobic residues
    pub n_hydrophobic_residues: usize,

    /// Hydrophobic surface area estimate (Å²)
    pub estimated_hydrophobic_area: f32,

    /// Computed hydrophobic surface ratio (from Kyte-Doolittle + surface exposure)
    pub hydrophobic_surface_ratio_computed: f32,

    // ─────────────────────────────────────────────────────────────────────────
    // Structural Analysis (COMPUTED from 3D geometry)
    // ─────────────────────────────────────────────────────────────────────────
    /// Secondary structure classification (COMPUTED from phi/psi dihedrals)
    pub secondary_structure_class: SecondaryStructureClass,

    /// Number of structural domains (COMPUTED from CA contact map)
    pub domain_count: usize,

    /// Number of ALL ring-containing residues (TRP/TYR/PHE/HIS/PRO for UV)
    pub ring_residue_count: usize,

    // ─────────────────────────────────────────────────────────────────────────
    // Computational Profile
    // ─────────────────────────────────────────────────────────────────────────
    /// Detailed GPU memory profile
    pub memory_profile: MemoryProfile,

    /// Memory tier classification
    pub memory_tier: MemoryTier,

    /// Complexity tier classification
    pub complexity_tier: ComplexityTier,

    /// Spike density tier
    pub spike_density_tier: SpikeDensityTier,

    /// Predicted spike density (spikes per 1000 steps, estimated)
    pub predicted_spike_density: f32,

    // ─────────────────────────────────────────────────────────────────────────
    // Batch Scheduling
    // ─────────────────────────────────────────────────────────────────────────
    /// Multi-dimensional batch compatibility profile
    pub batch_compatibility: BatchCompatibility,

    /// Batch group ID (structures with same ID can be batched together)
    pub batch_group_id: u32,

    /// Maximum concurrent structures
    pub max_concurrent: u8,

    /// Recommended simulation parameters
    pub recommended_steps: usize,

    // ─────────────────────────────────────────────────────────────────────────
    // Structure Quality / Edge Cases
    // ─────────────────────────────────────────────────────────────────────────
    /// Number of chain breaks detected (non-sequential residue numbers)
    pub chain_breaks: usize,

    /// Missing residue ranges (gaps in sequence)
    pub missing_residue_ranges: Vec<(String, i32, i32)>, // (chain_id, start, end)

    /// Number of alternate conformers detected
    pub alternate_conformers: usize,

    /// Is this a multi-model structure (NMR ensemble)?
    pub is_multi_model: bool,

    /// Model number if multi-model (None for single model)
    pub model_number: Option<usize>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Structural Geometry Computations
// ═══════════════════════════════════════════════════════════════════════════════

/// Secondary structure classification from backbone geometry
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecondaryStructureClass {
    /// >50% alpha helix
    Alpha,
    /// >50% beta sheet
    Beta,
    /// No single type dominates
    Mixed,
    /// >60% coil/loop
    Coil,
}

/// Compute secondary structure from phi/psi dihedrals
fn compute_secondary_structure(topology: &PrismPrepTopology) -> SecondaryStructureClass {
    // Find backbone atoms (N, CA, C) for each residue
    let mut residue_backbone: HashMap<usize, (Option<usize>, Option<usize>, Option<usize>)> =
        HashMap::new();

    for i in 0..topology.n_atoms {
        let res_id = topology.residue_ids[i];
        let atom_name = topology.atom_names.get(i).map(|s| s.trim()).unwrap_or("");

        let entry = residue_backbone.entry(res_id).or_insert((None, None, None));

        match atom_name {
            "N" => entry.0 = Some(i),
            "CA" => entry.1 = Some(i),
            "C" => entry.2 = Some(i),
            _ => {}
        }
    }

    // Compute phi/psi for each residue
    let mut helix_count = 0;
    let mut sheet_count = 0;
    let mut coil_count = 0;
    let mut total_classified = 0;

    let residue_count = residue_backbone.len();
    let residue_ids: Vec<usize> = residue_backbone.keys().copied().collect();

    for i in 1..residue_ids.len() - 1 {
        let prev_res = residue_ids.get(i.wrapping_sub(1));
        let curr_res = Some(&residue_ids[i]);
        let next_res = residue_ids.get(i + 1);

        if let (Some(&prev), Some(&curr), Some(&next)) = (prev_res, curr_res, next_res) {
            // Get backbone atoms for phi (C_{i-1}, N_i, CA_i, C_i)
            let prev_c = residue_backbone.get(&prev).and_then(|bb| bb.2);
            let curr_n = residue_backbone.get(&curr).and_then(|bb| bb.0);
            let curr_ca = residue_backbone.get(&curr).and_then(|bb| bb.1);
            let curr_c = residue_backbone.get(&curr).and_then(|bb| bb.2);

            // Get backbone atoms for psi (N_i, CA_i, C_i, N_{i+1})
            let next_n = residue_backbone.get(&next).and_then(|bb| bb.0);

            if let (Some(c_prev), Some(n), Some(ca), Some(c), Some(n_next)) =
                (prev_c, curr_n, curr_ca, curr_c, next_n)
            {
                // Compute phi dihedral
                let phi = compute_dihedral(topology, c_prev, n, ca, c);
                // Compute psi dihedral
                let psi = compute_dihedral(topology, n, ca, c, n_next);

                // Classify based on Ramachandran regions
                let ss_type = classify_ramachandran(phi, psi);
                match ss_type {
                    0 => helix_count += 1, // Alpha helix
                    1 => sheet_count += 1, // Beta sheet
                    _ => coil_count += 1,  // Coil
                }
                total_classified += 1;
            }
        }
    }

    if total_classified == 0 {
        return SecondaryStructureClass::Coil;
    }

    let helix_frac = helix_count as f32 / total_classified as f32;
    let sheet_frac = sheet_count as f32 / total_classified as f32;
    let coil_frac = coil_count as f32 / total_classified as f32;

    if helix_frac > 0.5 {
        SecondaryStructureClass::Alpha
    } else if sheet_frac > 0.5 {
        SecondaryStructureClass::Beta
    } else if coil_frac > 0.6 {
        SecondaryStructureClass::Coil
    } else {
        SecondaryStructureClass::Mixed
    }
}

/// Compute dihedral angle from 4 atom indices (in degrees)
fn compute_dihedral(topology: &PrismPrepTopology, i: usize, j: usize, k: usize, l: usize) -> f32 {
    let pos = &topology.positions;

    let p1 = [pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]];
    let p2 = [pos[j * 3], pos[j * 3 + 1], pos[j * 3 + 2]];
    let p3 = [pos[k * 3], pos[k * 3 + 1], pos[k * 3 + 2]];
    let p4 = [pos[l * 3], pos[l * 3 + 1], pos[l * 3 + 2]];

    // Vectors
    let b1 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
    let b2 = [p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]];
    let b3 = [p4[0] - p3[0], p4[1] - p3[1], p4[2] - p3[2]];

    // Normal vectors
    let n1 = cross_product(b1, b2);
    let n2 = cross_product(b2, b3);

    // Dihedral angle
    let dot = n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2];
    let n1_mag = (n1[0] * n1[0] + n1[1] * n1[1] + n1[2] * n1[2]).sqrt();
    let n2_mag = (n2[0] * n2[0] + n2[1] * n2[1] + n2[2] * n2[2]).sqrt();

    if n1_mag < 1e-6 || n2_mag < 1e-6 {
        return 0.0;
    }

    let cos_angle = (dot / (n1_mag * n2_mag)).clamp(-1.0, 1.0);
    cos_angle.acos().to_degrees()
}

fn cross_product(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Classify phi/psi into secondary structure type
fn classify_ramachandran(phi: f32, psi: f32) -> u8 {
    // Alpha helix region: phi ~ -60, psi ~ -45
    if phi >= -100.0 && phi <= -20.0 && psi >= -80.0 && psi <= -10.0 {
        return 0; // Helix
    }

    // Beta sheet region: phi ~ -120, psi ~ +130
    if phi >= -180.0 && phi <= -90.0 && psi >= 90.0 && psi <= 180.0 {
        return 1; // Sheet
    }

    2 // Coil
}

/// Detect structural domains from CA contact map
fn detect_domains(topology: &PrismPrepTopology) -> usize {
    // Extract CA atom indices
    let ca_indices: Vec<(usize, usize)> = topology
        .atom_names
        .iter()
        .enumerate()
        .filter(|(_, name)| name.trim() == "CA")
        .map(|(idx, _)| (idx, topology.residue_ids[idx]))
        .collect();

    if ca_indices.len() < 10 {
        return 1; // Too small for domain detection
    }

    // Build distance matrix
    let n = ca_indices.len();
    let mut contact_matrix = vec![vec![false; n]; n];
    let contact_threshold = CA_CONTACT_THRESHOLD_ANGSTROM as f32; // Angstroms

    for i in 0..n {
        for j in (i + 1)..n {
            let idx_i = ca_indices[i].0;
            let idx_j = ca_indices[j].0;

            let dx = topology.positions[idx_i * 3] - topology.positions[idx_j * 3];
            let dy = topology.positions[idx_i * 3 + 1] - topology.positions[idx_j * 3 + 1];
            let dz = topology.positions[idx_i * 3 + 2] - topology.positions[idx_j * 3 + 2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();

            if dist < contact_threshold {
                contact_matrix[i][j] = true;
                contact_matrix[j][i] = true;
            }
        }
    }

    // Find domains using contact density along diagonal
    // Look for blocks of high contact density separated by low-contact regions
    let window_size = 10.min(n / 4);
    let mut domain_boundaries = vec![0];

    for i in window_size..(n - window_size) {
        // Count contacts before and after this position
        let before_contacts: usize = (i.saturating_sub(window_size)..i)
            .map(|row| contact_matrix[row].iter().take(i).filter(|&&c| c).count())
            .sum();

        let after_contacts: usize = (i..(i + window_size).min(n))
            .map(|row| contact_matrix[row].iter().skip(i).filter(|&&c| c).count())
            .sum();

        // Low connectivity region = domain boundary
        let total_contacts = before_contacts + after_contacts;
        if total_contacts < window_size * 2 {
            domain_boundaries.push(i);
        }
    }

    domain_boundaries.push(n);

    // Merge adjacent boundaries
    domain_boundaries.dedup_by(|a, b| *b - *a < window_size);

    (domain_boundaries.len() - 1).max(1)
}

/// Compute hydrophobic surface ratio using Kyte-Doolittle scale + surface exposure
fn compute_hydrophobic_surface_ratio(
    topology: &PrismPrepTopology,
    residue_registry: &HashMap<ResidueKey, ResidueInfo>,
) -> f32 {
    // Kyte-Doolittle hydrophobicity scale
    let hydrophobicity_scale: HashMap<&str, f32> = [
        ("ALA", 1.8),
        ("ARG", -4.5),
        ("ASN", -3.5),
        ("ASP", -3.5),
        ("CYS", 2.5),
        ("GLN", -3.5),
        ("GLU", -3.5),
        ("GLY", -0.4),
        ("HIS", -3.2),
        ("ILE", 4.5),
        ("LEU", 3.8),
        ("LYS", -3.9),
        ("MET", 1.9),
        ("PHE", 2.8),
        ("PRO", -1.6),
        ("SER", -0.8),
        ("THR", -0.7),
        ("TRP", -0.9),
        ("TYR", -1.3),
        ("VAL", 4.2),
    ]
    .iter()
    .copied()
    .collect();

    // Find surface-exposed residues (CA atoms near convex hull)
    // Simplified: residues with CA > 4Å from any other CA = surface
    let ca_positions: Vec<(usize, [f32; 3])> = topology
        .atom_names
        .iter()
        .enumerate()
        .filter(|(_, name)| name.trim() == "CA")
        .map(|(idx, _)| {
            let pos = [
                topology.positions[idx * 3],
                topology.positions[idx * 3 + 1],
                topology.positions[idx * 3 + 2],
            ];
            (topology.residue_ids[idx], pos)
        })
        .collect();

    let mut surface_residues = Vec::new();
    let surface_threshold = 10.0; // Å from geometric center

    // Compute geometric center
    let n = ca_positions.len() as f32;
    let center = [
        ca_positions.iter().map(|(_, p)| p[0]).sum::<f32>() / n,
        ca_positions.iter().map(|(_, p)| p[1]).sum::<f32>() / n,
        ca_positions.iter().map(|(_, p)| p[2]).sum::<f32>() / n,
    ];

    for (res_id, pos) in &ca_positions {
        let dx = pos[0] - center[0];
        let dy = pos[1] - center[1];
        let dz = pos[2] - center[2];
        let dist_from_center = (dx * dx + dy * dy + dz * dz).sqrt();

        // Surface = far from center
        if dist_from_center > surface_threshold {
            surface_residues.push(*res_id);
        }
    }

    if surface_residues.is_empty() {
        return 0.0;
    }

    // Count hydrophobic surface residues
    let mut hydrophobic_surface = 0;
    for res_id in &surface_residues {
        if let Some(res_name) = topology.residue_names.get(*res_id) {
            if let Some(&score) = hydrophobicity_scale.get(res_name.trim()) {
                if score > 0.0 {
                    hydrophobic_surface += 1;
                }
            }
        }
    }

    hydrophobic_surface as f32 / surface_residues.len() as f32
}

/// Detect all ring-containing residues for UV perturbation
fn count_ring_residues(topology: &PrismPrepTopology) -> usize {
    // ALL ring-containing residues: TRP, TYR, PHE, HIS (all states), PRO
    let ring_names = ["TRP", "TYR", "PHE", "HIS", "HID", "HIE", "HIP", "PRO"];

    // residue_names is per-atom, so we need to count unique residue IDs
    let mut ring_residue_ids = std::collections::HashSet::new();

    for (atom_idx, res_name) in topology.residue_names.iter().enumerate() {
        if atom_idx < topology.residue_ids.len() && ring_names.contains(&res_name.trim()) {
            ring_residue_ids.insert(topology.residue_ids[atom_idx]);
        }
    }

    ring_residue_ids.len()
}

impl StructureComposition {
    /// Analyze a topology and compute its composition
    pub fn analyze(topology: &PrismPrepTopology) -> Result<Self> {
        let structure_name = std::path::Path::new(&topology.source_pdb)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        log::info!(
            "Analyzing structure composition: {} ({} atoms, {} residues)",
            structure_name,
            topology.n_atoms,
            topology.n_residues
        );

        // Build residue registry
        let mut residue_registry = HashMap::new();
        let mut chains: HashMap<String, ChainInfo> = HashMap::new();

        // Aromatic and catalytic residue sets
        let aromatic_names = ["TRP", "TYR", "PHE", "HIS", "HID", "HIE", "HIP"];
        let catalytic_names = [
            "GLU", "ASP", "HIS", "HID", "HIE", "HIP", "SER", "CYS", "LYS",
        ];
        let hydrophobic_names = ["ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "PRO"];

        let mut aromatic_residue_keys = Vec::new();
        let mut n_aromatic = 0usize;
        let mut n_catalytic = 0usize;
        let mut n_hydrophobic = 0usize;

        // Track residues by their first atom
        let mut current_res_id = usize::MAX;
        let mut current_chain = String::new();
        let mut res_first_atom = 0usize;

        for i in 0..topology.n_atoms {
            let res_id = topology.residue_ids[i];
            let chain_id = &topology.chain_ids[i];

            // New residue?
            if res_id != current_res_id || chain_id != &current_chain {
                // Finalize previous residue
                if current_res_id != usize::MAX {
                    let res_name = &topology.residue_names[current_res_id];
                    let is_aromatic = aromatic_names.contains(&res_name.as_str());
                    let is_catalytic = catalytic_names.contains(&res_name.as_str());
                    let is_hydrophobic = hydrophobic_names.contains(&res_name.as_str());

                    // Create residue key (use sequential numbering for now)
                    let key = ResidueKey::simple(current_chain.clone(), current_res_id as i32);

                    let info = ResidueInfo {
                        name: res_name.clone(),
                        topology_index: current_res_id,
                        atom_count: i - res_first_atom,
                        first_atom_index: res_first_atom,
                        last_atom_index: i,
                        is_aromatic,
                        is_catalytic,
                        is_hydrophobic,
                    };

                    if is_aromatic {
                        aromatic_residue_keys.push(key.clone());
                        n_aromatic += 1;
                    }
                    if is_catalytic {
                        n_catalytic += 1;
                    }
                    if is_hydrophobic {
                        n_hydrophobic += 1;
                    }

                    residue_registry.insert(key, info);
                }

                // Start new residue
                current_res_id = res_id;
                current_chain = chain_id.clone();
                res_first_atom = i;

                // Update chain info
                let chain_entry = chains.entry(chain_id.clone()).or_insert(ChainInfo {
                    chain_id: chain_id.clone(),
                    first_residue: res_id,
                    last_residue: res_id + 1,
                    residue_count: 0,
                    atom_count: 0,
                });
                chain_entry.last_residue = res_id + 1;
                chain_entry.residue_count += 1;
            }

            // Update chain atom count
            if let Some(chain) = chains.get_mut(chain_id) {
                chain.atom_count += 1;
            }
        }

        // Finalize last residue
        if current_res_id != usize::MAX && current_res_id < topology.residue_names.len() {
            let res_name = &topology.residue_names[current_res_id];
            let is_aromatic = aromatic_names.contains(&res_name.as_str());
            let is_catalytic = catalytic_names.contains(&res_name.as_str());
            let is_hydrophobic = hydrophobic_names.contains(&res_name.as_str());

            let key = ResidueKey::simple(current_chain.clone(), current_res_id as i32);
            let info = ResidueInfo {
                name: res_name.clone(),
                topology_index: current_res_id,
                atom_count: topology.n_atoms - res_first_atom,
                first_atom_index: res_first_atom,
                last_atom_index: topology.n_atoms,
                is_aromatic,
                is_catalytic,
                is_hydrophobic,
            };

            if is_aromatic {
                aromatic_residue_keys.push(key.clone());
                n_aromatic += 1;
            }
            if is_catalytic {
                n_catalytic += 1;
            }
            if is_hydrophobic {
                n_hydrophobic += 1;
            }

            residue_registry.insert(key, info);
        }

        // Convert chains to sorted vec
        let mut chains_vec: Vec<ChainInfo> = chains.into_values().collect();
        chains_vec.sort_by(|a, b| a.chain_id.cmp(&b.chain_id));

        // Compute densities
        let aromatic_density = if topology.n_residues > 0 {
            n_aromatic as f32 / topology.n_residues as f32
        } else {
            0.0
        };

        let catalytic_density = if topology.n_residues > 0 {
            n_catalytic as f32 / topology.n_residues as f32
        } else {
            0.0
        };

        // Estimate hydrophobic surface area (rough: 40Å² per hydrophobic residue)
        let estimated_hydrophobic_area = n_hydrophobic as f32 * 40.0;

        // Calculate detailed memory profile
        let memory_profile = MemoryProfile::calculate(
            topology.n_atoms,
            topology.bonds.len(),
            topology.angles.len(),
            topology.dihedrals.len(),
            n_aromatic,
            0.01, // typical protein density
        );

        // Classify tiers
        let memory_tier = MemoryTier::from_memory_profile(&memory_profile);
        let complexity_tier = ComplexityTier::from_features(
            chains_vec.len(),
            aromatic_density,
            false, // TODO: detect membrane-like features
        );

        // Predict spike density (empirical: ~5-20 spikes per aromatic per 1000 steps)
        let predicted_spike_density = n_aromatic as f32 * 12.0;
        let spike_density_tier = SpikeDensityTier::from_spike_count(
            (predicted_spike_density * 50.0) as usize, // estimate for 50K steps
        );

        // Multi-dimensional batch compatibility
        let batch_compatibility =
            BatchCompatibility::new(memory_tier, complexity_tier, spike_density_tier);
        let batch_group_id = batch_compatibility.batch_group_id();

        // Recommended steps based on complexity
        let recommended_steps = match complexity_tier {
            ComplexityTier::Simple => 30_000,
            ComplexityTier::Moderate => 50_000,
            ComplexityTier::Complex => 100_000,
        };

        // Detect chain breaks (gaps > 1 in residue numbering)
        let mut chain_breaks = 0usize;
        let mut missing_residue_ranges = Vec::new();

        for chain in &chains_vec {
            let chain_residues: Vec<_> = residue_registry
                .iter()
                .filter(|(k, _)| k.chain_id == chain.chain_id)
                .collect();

            let mut sorted_residues: Vec<i32> = chain_residues
                .iter()
                .map(|(k, _)| k.residue_number)
                .collect();
            sorted_residues.sort();

            for window in sorted_residues.windows(2) {
                let gap = window[1] - window[0];
                if gap > 1 {
                    chain_breaks += 1;
                    missing_residue_ranges.push((
                        chain.chain_id.clone(),
                        window[0] + 1,
                        window[1] - 1,
                    ));
                }
            }
        }

        // Count alternate conformers
        let alternate_conformers = residue_registry
            .keys()
            .filter(|k| k.alt_loc.is_some())
            .count();

        // ─────────────────────────────────────────────────────────────────────────
        // COMPUTE structural features from 3D geometry
        // ─────────────────────────────────────────────────────────────────────────

        // Secondary structure from phi/psi dihedrals
        let secondary_structure_class = compute_secondary_structure(topology);

        // Domain count from CA contact map
        let domain_count = detect_domains(topology);

        // Hydrophobic surface ratio (Kyte-Doolittle + surface exposure)
        let hydrophobic_surface_ratio_computed =
            compute_hydrophobic_surface_ratio(topology, &residue_registry);

        // ALL ring-containing residues for UV perturbation
        let ring_residue_count = count_ring_residues(topology);

        // Update spike density prediction to use ALL ring residues, not just aromatics
        let predicted_spike_density = ring_residue_count as f32 * 12.0;
        let spike_density_tier = SpikeDensityTier::from_spike_count(
            (predicted_spike_density * 50.0) as usize, // estimate for 50K steps
        );

        // Re-compute batch compatibility with updated spike density
        let batch_compatibility =
            BatchCompatibility::new(memory_tier, complexity_tier, spike_density_tier);
        let batch_group_id = batch_compatibility.batch_group_id();

        log::info!(
            "  Chains: {}",
            chains_vec
                .iter()
                .map(|c| &c.chain_id)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        );
        log::info!(
            "  Aromatics: {} ({:.1}%)",
            n_aromatic,
            aromatic_density * 100.0
        );
        log::info!("  Ring residues (UV-active): {}", ring_residue_count);
        log::info!(
            "  Catalytic: {} ({:.1}%)",
            n_catalytic,
            catalytic_density * 100.0
        );
        log::info!("  Secondary structure: {:?}", secondary_structure_class);
        log::info!("  Domains: {}", domain_count);
        log::info!(
            "  Hydrophobic surface: {:.1}%",
            hydrophobic_surface_ratio_computed * 100.0
        );
        log::info!(
            "  Memory: {} MB ({:?})",
            memory_profile.total_mb(),
            memory_tier
        );
        log::info!(
            "  Complexity: {:?}, Spike density: {:?}",
            complexity_tier,
            spike_density_tier
        );
        log::info!(
            "  Batch group: {} (workload: {:.1})",
            batch_group_id,
            batch_compatibility.estimated_workload()
        );
        if chain_breaks > 0 {
            log::info!(
                "  Chain breaks: {} (missing ranges: {:?})",
                chain_breaks,
                missing_residue_ranges
            );
        }

        Ok(Self {
            structure_name,
            source_path: topology.source_pdb.clone(),
            residue_registry,
            chains: chains_vec,
            n_atoms: topology.n_atoms,
            n_residues: topology.n_residues,
            n_chains: topology.n_chains,
            n_aromatic_residues: n_aromatic,
            aromatic_density,
            aromatic_residue_keys,
            n_catalytic_residues: n_catalytic,
            catalytic_density,
            n_hydrophobic_residues: n_hydrophobic,
            estimated_hydrophobic_area,
            hydrophobic_surface_ratio_computed,
            secondary_structure_class,
            domain_count,
            ring_residue_count,
            memory_profile,
            memory_tier,
            complexity_tier,
            spike_density_tier,
            predicted_spike_density,
            batch_compatibility,
            batch_group_id,
            max_concurrent: memory_tier.max_concurrent(),
            recommended_steps,
            chain_breaks,
            missing_residue_ranges,
            alternate_conformers,
            is_multi_model: false, // TODO: detect from topology
            model_number: None,
        })
    }

    /// Get residue info by key
    pub fn get_residue(&self, key: &ResidueKey) -> Option<&ResidueInfo> {
        self.residue_registry.get(key)
    }

    /// Get all residues for a chain
    pub fn get_chain_residues(&self, chain_id: &str) -> Vec<(&ResidueKey, &ResidueInfo)> {
        self.residue_registry
            .iter()
            .filter(|(k, _)| k.chain_id == chain_id)
            .collect()
    }

    /// Check if two compositions are compatible for batching
    /// Uses multi-dimensional compatibility check
    pub fn is_batch_compatible(&self, other: &StructureComposition) -> bool {
        self.batch_compatibility
            .is_compatible(&other.batch_compatibility)
    }

    /// Get estimated workload for load balancing
    pub fn estimated_workload(&self) -> f32 {
        self.batch_compatibility.estimated_workload()
    }

    /// Check if this structure has quality issues (chain breaks, missing residues)
    pub fn has_quality_issues(&self) -> bool {
        self.chain_breaks > 0 || !self.missing_residue_ranges.is_empty()
    }

    /// Export composition to JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).context("Failed to serialize composition to JSON")
    }

    /// Save composition to file
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let json = self.to_json()?;
        std::fs::write(&path, json).with_context(|| {
            format!(
                "Failed to write composition to: {}",
                path.as_ref().display()
            )
        })?;
        Ok(())
    }

    /// Load composition from file
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let content = std::fs::read_to_string(&path).with_context(|| {
            format!(
                "Failed to read composition from: {}",
                path.as_ref().display()
            )
        })?;
        serde_json::from_str(&content).context("Failed to parse composition JSON")
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Batch Grouping
// ═══════════════════════════════════════════════════════════════════════════════

/// Group structures for optimal batch processing
pub struct BatchGroup {
    /// Group identifier
    pub group_id: u32,
    /// Memory tier for this group
    pub memory_tier: MemoryTier,
    /// Structures in this group
    pub structures: Vec<StructureComposition>,
}

impl BatchGroup {
    /// Create empty batch group
    pub fn new(group_id: u32, memory_tier: MemoryTier) -> Self {
        Self {
            group_id,
            memory_tier,
            structures: Vec::new(),
        }
    }

    /// Add structure to group
    pub fn add(&mut self, composition: StructureComposition) {
        self.structures.push(composition);
    }

    /// Number of structures
    pub fn len(&self) -> usize {
        self.structures.len()
    }

    /// Is empty
    pub fn is_empty(&self) -> bool {
        self.structures.is_empty()
    }

    /// Maximum concurrent structures for this group
    pub fn max_concurrent(&self) -> u8 {
        self.memory_tier.max_concurrent()
    }
}

/// Group multiple structures by batch compatibility
pub fn group_for_batch(compositions: Vec<StructureComposition>) -> Vec<BatchGroup> {
    let mut groups: HashMap<u32, BatchGroup> = HashMap::new();

    for comp in compositions {
        let group_id = comp.batch_group_id;
        let memory_tier = comp.memory_tier;

        groups
            .entry(group_id)
            .or_insert_with(|| BatchGroup::new(group_id, memory_tier))
            .add(comp);
    }

    // Sort groups by: Large first (serialize), then Medium, then Small (parallelize)
    let mut groups_vec: Vec<BatchGroup> = groups.into_values().collect();
    groups_vec.sort_by_key(|g| match g.memory_tier {
        MemoryTier::Large => 0,
        MemoryTier::Medium => 1,
        MemoryTier::Small => 2,
    });

    groups_vec
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_residue_key() {
        let key1 = ResidueKey::simple("A", 42);
        let key2 = ResidueKey::with_insertion("A", 42, 'A');
        let key3 = ResidueKey::with_alt_loc("A", 42, 'B');
        let key4 = ResidueKey::new("A", 42, Some('A'), Some('B'));

        assert_eq!(key1.to_pdb_string(), "A:42");
        assert_eq!(key2.to_pdb_string(), "A:42A");
        assert_eq!(key3.to_pdb_string(), "A:42(B)");
        assert_eq!(key4.to_pdb_string(), "A:42A(B)");
        assert_ne!(key1, key2);
        assert_ne!(key2, key3);
        assert_ne!(key3, key4);
    }

    #[test]
    fn test_memory_tier() {
        // Based on detailed memory profiling:
        // 10K atoms → ~8 MB (Small)
        // 50K atoms → ~40 MB (Medium)
        // 150K atoms → ~120 MB (Large)
        assert_eq!(MemoryTier::from_atom_count(10_000), MemoryTier::Small); // ~8 MB
        assert_eq!(MemoryTier::from_atom_count(50_000), MemoryTier::Medium); // ~40 MB
        assert_eq!(MemoryTier::from_atom_count(150_000), MemoryTier::Large); // ~120 MB

        // Concurrency limits for 16GB GPU
        assert_eq!(MemoryTier::Small.max_concurrent(), 4);
        assert_eq!(MemoryTier::Medium.max_concurrent(), 2);
        assert_eq!(MemoryTier::Large.max_concurrent(), 1);
    }
}
