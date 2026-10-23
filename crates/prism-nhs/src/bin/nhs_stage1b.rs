//! Stage 1B: Structure Composition Analysis CLI
//!
//! Analyzes topology files from prism-prep and generates a batch manifest
//! for optimal GPU scheduling in Stage 2A (nhs-rt-full).
//!
//! Usage:
//!   nhs-stage1b --topology-dir prep/ --output batch_manifest.json
//!   nhs-stage1b --topology-dir prep/ --output batch_manifest.json --gpu-memory 24000

use anyhow::{Context, Result};
use chrono::Utc;
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use prism_nhs::{
    group_for_batch, ComplexityTier, PrismPrepTopology, SpikeDensityTier, StructureComposition,
};

#[derive(Parser)]
#[command(name = "nhs-stage1b")]
#[command(about = "Stage 1B: Structure Composition Analysis & Batch Scheduling")]
struct Args {
    /// Directory containing .topology.json files from prism-prep
    #[arg(long)]
    topology_dir: PathBuf,

    /// Output path for batch manifest JSON
    #[arg(long, short, default_value = "batch_manifest.json")]
    output: PathBuf,

    /// GPU memory in MB (for concurrency calculation)
    #[arg(long, default_value = "16000")]
    gpu_memory: usize,

    /// Number of replicas (affects total GPU memory per structure)
    #[arg(long, default_value = "3")]
    replicas: usize,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Manifest Data Structures
// ═══════════════════════════════════════════════════════════════════════════════

/// Tracks whether factors are computed from topology data or estimated via heuristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputationFlags {
    /// Chain count: computed from topology
    pub chain_count_computed: bool,
    /// Aromatic density: computed from residue names in topology
    pub aromatic_density_computed: bool,
    /// Spike density: estimated from aromatic count (not measured from actual simulation)
    pub spike_density_estimated: bool,
    /// Hydrophobic surface ratio: estimated from residue types (NOT computed from 3D geometry)
    pub hydrophobic_surface_estimated: bool,
    /// Secondary structure: NOT COMPUTED (would require DSSP or dihedral analysis)
    pub secondary_structure_computed: bool,
    /// Domain count: NOT COMPUTED (would require structural/sequence domain analysis)
    pub domain_count_computed: bool,
    /// Chain breaks: computed from residue number gaps
    pub chain_breaks_computed: bool,
}

impl Default for ComputationFlags {
    fn default() -> Self {
        Self {
            chain_count_computed: true,
            aromatic_density_computed: true,
            spike_density_estimated: true, // Estimated from ring count, not measured from simulation
            hydrophobic_surface_estimated: false, // NOW COMPUTED from Kyte-Doolittle + surface exposure
            secondary_structure_computed: true,   // NOW COMPUTED from phi/psi dihedrals
            domain_count_computed: true,          // NOW COMPUTED from CA contact map
            chain_breaks_computed: true,
        }
    }
}

/// Complexity factors for a structure (for manifest output)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityFactors {
    /// Number of chains (computed from topology)
    pub chain_count: usize,
    /// Estimated spike density tier (estimated from ring residue count)
    pub estimated_spike_density: String,
    /// Complexity classification (computed from chains + aromatic density)
    pub dynamics_complexity: String,
    /// Hydrophobic surface ratio (COMPUTED from Kyte-Doolittle + surface exposure)
    pub hydrophobic_surface_ratio: f32,
    /// Number of aromatic residues (computed from topology)
    pub aromatic_count: usize,
    /// Number of ALL ring-containing residues for UV (TRP/TYR/PHE/HIS/PRO)
    pub ring_residue_count: usize,
    /// Aromatic density (computed: aromatics / total residues)
    pub aromatic_density: f32,
    /// Predicted spikes per 1000 steps (estimated from ring count, NOT measured)
    pub predicted_spike_density: f32,
    /// Secondary structure class (COMPUTED from phi/psi dihedrals)
    pub secondary_structure_class: String,
    /// Domain count (COMPUTED from CA contact map)
    pub domain_count: usize,
    /// Has chain breaks or missing residues (computed from residue numbering)
    pub has_quality_issues: bool,
    /// Batch group ID (computed from compatibility profile)
    pub batch_group_id: u32,
    /// Estimated relative workload (computed from tier factors)
    pub estimated_workload: f32,
    /// Tracks which factors are computed vs estimated
    pub computation_flags: ComputationFlags,
}

/// Structure entry in manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestStructure {
    /// Structure name
    pub name: String,
    /// Full path to topology file
    pub topology_path: String,
    /// Total atoms
    pub atoms: usize,
    /// Total residues
    pub residues: usize,
    /// Chain IDs
    pub chains: Vec<String>,
    /// Memory tier classification
    pub memory_tier: String,
    /// Estimated GPU memory in MB
    pub estimated_gpu_mb: usize,
    /// Complexity factors
    pub complexity_factors: ComplexityFactors,
}

/// Batch entry in manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestBatch {
    /// Batch ID
    pub batch_id: usize,
    /// Structures in this batch
    pub structures: Vec<ManifestStructure>,
    /// Maximum concurrent structures
    pub concurrency: usize,
    /// Memory tier for this batch
    pub memory_tier: String,
    /// Total estimated GPU memory for concurrent execution
    pub estimated_total_gpu_mb: usize,
    /// Batch group ID
    pub batch_group_id: u32,
    /// Number of replicas per structure for this batch (GPU-informed)
    pub replicas_per_structure: usize,
}

/// Complete batch manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchManifest {
    /// Generation timestamp
    pub generated_at: String,
    /// Pipeline version
    pub pipeline_version: String,
    /// GPU memory available (MB)
    pub gpu_memory_mb: usize,
    /// Number of replicas per structure
    pub replicas: usize,
    /// Total structures
    pub total_structures: usize,
    /// Total batches
    pub total_batches: usize,
    /// Batches in execution order
    pub batches: Vec<ManifestBatch>,
    /// Flat execution order (structure names)
    pub execution_order: Vec<String>,
    /// Summary statistics
    pub statistics: ManifestStatistics,
}

/// Manifest statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestStatistics {
    /// Structures by memory tier
    pub by_memory_tier: TierCounts,
    /// Structures by complexity
    pub by_complexity: TierCounts,
    /// Structures by spike density
    pub by_spike_density: TierCounts,
    /// Total estimated GPU memory (MB)
    pub total_estimated_gpu_mb: usize,
    /// Maximum batch concurrency
    pub max_concurrency: usize,
    /// Structures with quality issues
    pub structures_with_issues: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierCounts {
    pub small: usize,
    pub medium: usize,
    pub large: usize,
}

impl Default for TierCounts {
    fn default() -> Self {
        Self {
            small: 0,
            medium: 0,
            large: 0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════════

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     PRISM4D Stage 1B: Structure Composition Analysis          ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    // Find topology files
    let topology_files = find_topology_files(&args.topology_dir)?;
    println!(
        "Found {} topology files in {}",
        topology_files.len(),
        args.topology_dir.display()
    );

    if topology_files.is_empty() {
        anyhow::bail!(
            "No .topology.json files found in {}",
            args.topology_dir.display()
        );
    }

    // Analyze each topology
    println!();
    println!("=== Analyzing Structures ===");
    let mut compositions = Vec::new();
    let mut statistics = ManifestStatistics {
        by_memory_tier: TierCounts::default(),
        by_complexity: TierCounts::default(),
        by_spike_density: TierCounts::default(),
        total_estimated_gpu_mb: 0,
        max_concurrency: 0,
        structures_with_issues: 0,
    };

    for topo_path in &topology_files {
        let topology = PrismPrepTopology::load(topo_path)
            .with_context(|| format!("Failed to load: {}", topo_path.display()))?;

        let comp = StructureComposition::analyze(&topology)?;

        // Update statistics
        match comp.memory_tier {
            MemoryTier::Small => statistics.by_memory_tier.small += 1,
            MemoryTier::Medium => statistics.by_memory_tier.medium += 1,
            MemoryTier::Large => statistics.by_memory_tier.large += 1,
        }
        match comp.complexity_tier {
            ComplexityTier::Simple => statistics.by_complexity.small += 1,
            ComplexityTier::Moderate => statistics.by_complexity.medium += 1,
            ComplexityTier::Complex => statistics.by_complexity.large += 1,
        }
        match comp.spike_density_tier {
            SpikeDensityTier::Low => statistics.by_spike_density.small += 1,
            SpikeDensityTier::Medium => statistics.by_spike_density.medium += 1,
            SpikeDensityTier::High => statistics.by_spike_density.large += 1,
        }
        statistics.total_estimated_gpu_mb += comp.memory_profile.total_mb() * args.replicas;
        if comp.has_quality_issues() {
            statistics.structures_with_issues += 1;
        }

        // Store with path
        compositions.push((topo_path.clone(), comp));
    }

    // Group into batches using multi-dimensional compatibility
    println!();
    println!("=== Grouping into Batches ===");
    let compositions_only: Vec<_> = compositions.iter().map(|(_, c)| c.clone()).collect();
    let fine_groups = group_for_batch(compositions_only);

    // Merge groups within same memory tier to reduce single-structure batches
    // group_for_batch() creates too many groups - we want to maximize concurrency
    use prism_nhs::MemoryTier;
    use std::collections::HashMap;
    let mut merged_groups: HashMap<MemoryTier, Vec<prism_nhs::StructureComposition>> =
        HashMap::new();

    for group in fine_groups {
        merged_groups
            .entry(group.memory_tier)
            .or_insert_with(Vec::new)
            .extend(group.structures);
    }

    // Convert back to batch groups
    let mut groups = Vec::new();
    for (tier, structures) in merged_groups {
        let group = prism_nhs::composition::BatchGroup {
            group_id: 0, // Not used after merge
            memory_tier: tier,
            structures,
        };
        groups.push(group);
    }

    // Sort by memory tier: Large first (sequential), then Medium, then Small (parallel)
    groups.sort_by_key(|g| match g.memory_tier {
        MemoryTier::Large => 0,
        MemoryTier::Medium => 1,
        MemoryTier::Small => 2,
    });

    // Build manifest batches
    let mut batches = Vec::new();
    let mut execution_order = Vec::new();
    let mut batch_id = 0;

    for group in &groups {
        // Initial concurrency calculation with minimum replicas (3)
        let per_structure_mb_base = group.memory_tier.estimated_memory_mb();
        let initial_replicas = 3; // Minimum for statistical validity
        let per_structure_mb = per_structure_mb_base * initial_replicas;
        let memory_limited_concurrency = args.gpu_memory / per_structure_mb.max(1);
        let concurrency = group.max_concurrent().min(memory_limited_concurrency as u8) as usize;
        statistics.max_concurrency = statistics.max_concurrency.max(concurrency);

        // Create sub-batches based on concurrency
        for chunk in group.structures.chunks(concurrency.max(1)) {
            let mut manifest_structures = Vec::new();

            for comp in chunk {
                // Find the original path
                let topo_path = compositions
                    .iter()
                    .find(|(_, c)| c.structure_name == comp.structure_name)
                    .map(|(p, _)| p.to_string_lossy().to_string())
                    .unwrap_or_default();

                let structure = ManifestStructure {
                    name: comp.structure_name.clone(),
                    topology_path: topo_path,
                    atoms: comp.n_atoms,
                    residues: comp.n_residues,
                    chains: comp.chains.iter().map(|c| c.chain_id.clone()).collect(),
                    memory_tier: format!("{:?}", comp.memory_tier),
                    estimated_gpu_mb: comp.memory_profile.total_mb(), // Per-structure, SINGLE replica
                    complexity_factors: ComplexityFactors {
                        chain_count: comp.n_chains,
                        estimated_spike_density: format!("{:?}", comp.spike_density_tier),
                        dynamics_complexity: format!("{:?}", comp.complexity_tier),
                        hydrophobic_surface_ratio: comp.hydrophobic_surface_ratio_computed,
                        aromatic_count: comp.n_aromatic_residues,
                        ring_residue_count: comp.ring_residue_count,
                        aromatic_density: comp.aromatic_density,
                        predicted_spike_density: comp.predicted_spike_density,
                        secondary_structure_class: format!("{:?}", comp.secondary_structure_class),
                        domain_count: comp.domain_count,
                        has_quality_issues: comp.has_quality_issues(),
                        batch_group_id: comp.batch_group_id,
                        estimated_workload: comp.estimated_workload(),
                        computation_flags: ComputationFlags::default(),
                    },
                };

                execution_order.push(comp.structure_name.clone());
                manifest_structures.push(structure);
            }

            let total_gpu_mb_base: usize =
                manifest_structures.iter().map(|s| s.estimated_gpu_mb).sum();

            // Calculate GPU-informed replica count for this batch
            // Target 80-85% GPU utilization, floor of 3 (statistical validity), ceiling of 15 (diminishing returns)
            let per_structure_mb_avg = if chunk.len() > 0 {
                total_gpu_mb_base / chunk.len()
            } else {
                per_structure_mb_base
            };

            let replicas_per_structure = if per_structure_mb_avg > 0 {
                let max_replicas_float = (args.gpu_memory as f32 * 0.85)
                    / (chunk.len() as f32 * per_structure_mb_avg as f32);
                let max_replicas = max_replicas_float.floor() as usize;

                // Adaptive ceiling based on structure size (max 15 overall):
                // Smaller structures = higher ceiling (more sampling, less overhead)
                // Larger structures = lower ceiling (memory-constrained)
                let ceiling = if per_structure_mb_avg < 5 {
                    15 // <5MB: maximum sampling (was 30, reduced to 15 max)
                } else if per_structure_mb_avg < 20 {
                    15 // 5-20MB: standard sampling
                } else if per_structure_mb_avg < 50 {
                    10 // 20-50MB: moderate sampling
                } else {
                    5 // >50MB: memory-constrained, minimal sampling
                };

                max_replicas
                    .max(3) // Floor: minimum 3 for statistical validity
                    .min(ceiling) // Adaptive ceiling based on structure size
            } else {
                3
            };

            // Recalculate total GPU memory with optimized replica count
            let total_gpu_mb = total_gpu_mb_base * replicas_per_structure;

            batches.push(ManifestBatch {
                batch_id,
                structures: manifest_structures,
                concurrency: chunk.len(),
                memory_tier: format!("{:?}", group.memory_tier),
                estimated_total_gpu_mb: total_gpu_mb,
                batch_group_id: group.group_id,
                replicas_per_structure,
            });

            batch_id += 1;
        }
    }

    // Print batch summary
    for batch in &batches {
        let names: Vec<_> = batch.structures.iter().map(|s| s.name.as_str()).collect();
        let gpu_utilization =
            (batch.estimated_total_gpu_mb as f32 / args.gpu_memory as f32) * 100.0;
        println!(
            "  Batch {}: [{}] ({} concurrent × {} replicas, {} tier, ~{}MB, {:.1}% GPU)",
            batch.batch_id,
            names.join(", "),
            batch.concurrency,
            batch.replicas_per_structure,
            batch.memory_tier,
            batch.estimated_total_gpu_mb,
            gpu_utilization
        );
    }

    // Build final manifest
    let manifest = BatchManifest {
        generated_at: Utc::now().to_rfc3339(),
        pipeline_version: format!("prism4d-{}", env!("CARGO_PKG_VERSION")),
        gpu_memory_mb: args.gpu_memory,
        replicas: args.replicas,
        total_structures: compositions.len(),
        total_batches: batches.len(),
        batches,
        execution_order,
        statistics,
    };

    // Write manifest
    let manifest_json =
        serde_json::to_string_pretty(&manifest).context("Failed to serialize manifest")?;
    std::fs::write(&args.output, &manifest_json)
        .with_context(|| format!("Failed to write manifest to: {}", args.output.display()))?;

    // Print summary
    println!();
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║                    STAGE 1B COMPLETE                          ║");
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!(
        "║ Structures analyzed: {:>4}                                    ║",
        manifest.total_structures
    );
    println!(
        "║ Batches created:     {:>4}                                    ║",
        manifest.total_batches
    );
    println!(
        "║ Max concurrency:     {:>4}                                    ║",
        manifest.statistics.max_concurrency
    );
    println!(
        "║ GPU memory budget:   {:>4} MB                                 ║",
        manifest.gpu_memory_mb
    );
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!(
        "║ Memory tiers:  Small={:<3} Medium={:<3} Large={:<3}              ║",
        manifest.statistics.by_memory_tier.small,
        manifest.statistics.by_memory_tier.medium,
        manifest.statistics.by_memory_tier.large
    );
    println!(
        "║ Complexity:    Simple={:<2} Moderate={:<2} Complex={:<2}            ║",
        manifest.statistics.by_complexity.small,
        manifest.statistics.by_complexity.medium,
        manifest.statistics.by_complexity.large
    );
    println!(
        "║ Spike density: Low={:<3} Medium={:<3} High={:<3}                 ║",
        manifest.statistics.by_spike_density.small,
        manifest.statistics.by_spike_density.medium,
        manifest.statistics.by_spike_density.large
    );
    if manifest.statistics.structures_with_issues > 0 {
        println!(
            "║ ⚠ Structures with quality issues: {:>3}                        ║",
            manifest.statistics.structures_with_issues
        );
    }
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Manifest written to: {}", args.output.display());
    println!();
    println!("Next step:");
    println!(
        "  nhs-rt-full --manifest {} --fast --replicas {}",
        args.output.display(),
        args.replicas
    );

    Ok(())
}

/// Find all .topology.json files in directory
fn find_topology_files(dir: &PathBuf) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    for entry in std::fs::read_dir(dir)
        .with_context(|| format!("Failed to read directory: {}", dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            if let Some(name) = path.file_name() {
                let name_str = name.to_string_lossy();
                if name_str.ends_with(".topology.json") {
                    files.push(path);
                }
            }
        }
    }

    // Sort for deterministic ordering
    files.sort();
    Ok(files)
}
