//! Batch Scheduler for Optimal GPU Utilization
//!
//! Uses structure composition analysis to intelligently schedule
//! batch processing for maximum throughput.
//!
//! ## Scheduling Strategy
//!
//! 1. **Large structures** (>30K atoms): Run sequentially to avoid OOM
//! 2. **Medium structures** (15-30K atoms): Run 2-3 concurrently
//! 3. **Small structures** (<15K atoms): Run 4+ concurrently
//!
//! ## Benefits
//!
//! - Optimal GPU memory utilization
//! - Predictable batch completion times
//! - Reduced total wall-clock time for multi-structure benchmarks

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::PathBuf;

use crate::composition::{group_for_batch, BatchGroup, MemoryTier, StructureComposition};
use crate::input::PrismPrepTopology;

// ═══════════════════════════════════════════════════════════════════════════════
// Scheduling Types
// ═══════════════════════════════════════════════════════════════════════════════

/// A scheduled batch of structures
#[derive(Debug, Clone)]
pub struct ScheduledBatch {
    /// Batch index (execution order)
    pub batch_id: usize,
    /// Structure compositions in this batch
    pub structures: Vec<StructureComposition>,
    /// Topology file paths
    pub topology_paths: Vec<PathBuf>,
    /// Number of concurrent structures to run
    pub concurrency: usize,
    /// Estimated GPU memory usage (MB)
    pub estimated_memory_mb: usize,
    /// Memory tier of this batch
    pub memory_tier: MemoryTier,
}

impl ScheduledBatch {
    /// Total structures in batch
    pub fn len(&self) -> usize {
        self.structures.len()
    }

    /// Is empty
    pub fn is_empty(&self) -> bool {
        self.structures.is_empty()
    }

    /// Structure names
    pub fn structure_names(&self) -> Vec<String> {
        self.structures
            .iter()
            .map(|s| s.structure_name.clone())
            .collect()
    }
}

/// Complete execution schedule
#[derive(Debug)]
pub struct ExecutionSchedule {
    /// Ordered batches for execution
    pub batches: Vec<ScheduledBatch>,
    /// Total structures
    pub total_structures: usize,
    /// Estimated total time (assuming average per-structure time)
    pub estimated_total_batches: usize,
    /// Statistics
    pub stats: ScheduleStats,
}

/// Schedule statistics
#[derive(Debug, Default)]
pub struct ScheduleStats {
    /// Number of small structures
    pub small_count: usize,
    /// Number of medium structures
    pub medium_count: usize,
    /// Number of large structures
    pub large_count: usize,
    /// Maximum concurrency achieved
    pub max_concurrency: usize,
    /// Number of sequential batches (concurrency=1)
    pub sequential_batches: usize,
    /// Number of parallel batches (concurrency>1)
    pub parallel_batches: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Batch Scheduler
// ═══════════════════════════════════════════════════════════════════════════════

/// Batch scheduler configuration
#[derive(Debug, Clone)]
pub struct BatchSchedulerConfig {
    /// Maximum GPU memory to use (MB)
    pub max_gpu_memory_mb: usize,
    /// Override max concurrent (0 = auto)
    pub max_concurrent_override: usize,
    /// Prefer throughput over latency
    pub prefer_throughput: bool,
}

impl Default for BatchSchedulerConfig {
    fn default() -> Self {
        Self {
            max_gpu_memory_mb: 16_000, // 16 GB
            max_concurrent_override: 0,
            prefer_throughput: true,
        }
    }
}

/// Batch scheduler for optimal GPU utilization
pub struct BatchScheduler {
    config: BatchSchedulerConfig,
    compositions: Vec<StructureComposition>,
    topology_paths: HashMap<String, PathBuf>,
}

impl BatchScheduler {
    /// Create a new batch scheduler
    pub fn new(config: BatchSchedulerConfig) -> Self {
        Self {
            config,
            compositions: Vec::new(),
            topology_paths: HashMap::new(),
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(BatchSchedulerConfig::default())
    }

    /// Analyze and add a topology file
    pub fn add_topology(&mut self, path: impl Into<PathBuf>) -> Result<&StructureComposition> {
        let path = path.into();
        let topology = PrismPrepTopology::load(&path)
            .with_context(|| format!("Failed to load topology: {}", path.display()))?;

        let composition = StructureComposition::analyze(&topology)?;
        let name = composition.structure_name.clone();

        self.topology_paths.insert(name.clone(), path);
        self.compositions.push(composition);

        Ok(self.compositions.last().unwrap())
    }

    /// Add multiple topology files
    pub fn add_topologies(&mut self, paths: impl IntoIterator<Item = PathBuf>) -> Result<usize> {
        let mut count = 0;
        for path in paths {
            self.add_topology(path)?;
            count += 1;
        }
        Ok(count)
    }

    /// Add a pre-analyzed composition (with path)
    pub fn add_composition(&mut self, composition: StructureComposition, path: PathBuf) {
        let name = composition.structure_name.clone();
        self.topology_paths.insert(name, path);
        self.compositions.push(composition);
    }

    /// Number of structures added
    pub fn len(&self) -> usize {
        self.compositions.len()
    }

    /// Is empty
    pub fn is_empty(&self) -> bool {
        self.compositions.is_empty()
    }

    /// Generate execution schedule
    pub fn schedule(&self) -> ExecutionSchedule {
        if self.compositions.is_empty() {
            return ExecutionSchedule {
                batches: Vec::new(),
                total_structures: 0,
                estimated_total_batches: 0,
                stats: ScheduleStats::default(),
            };
        }

        // Group structures by batch compatibility
        let groups = group_for_batch(self.compositions.clone());

        let mut batches = Vec::new();
        let mut stats = ScheduleStats::default();
        let mut batch_id = 0;

        for group in groups {
            // Count by tier
            match group.memory_tier {
                MemoryTier::Small => stats.small_count += group.len(),
                MemoryTier::Medium => stats.medium_count += group.len(),
                MemoryTier::Large => stats.large_count += group.len(),
            }

            // Determine concurrency for this group
            let max_concurrent = if self.config.max_concurrent_override > 0 {
                self.config.max_concurrent_override
            } else {
                group.max_concurrent() as usize
            };

            // Check memory constraints
            let memory_per_structure = group.memory_tier.estimated_memory_mb();
            let memory_limited_concurrent = self.config.max_gpu_memory_mb / memory_per_structure;
            let actual_concurrent = max_concurrent.min(memory_limited_concurrent).max(1);

            // Create batches from this group
            let mut current_batch = Vec::new();
            let mut current_paths = Vec::new();

            for structure in group.structures {
                let path = self
                    .topology_paths
                    .get(&structure.structure_name)
                    .cloned()
                    .unwrap_or_default();

                current_batch.push(structure);
                current_paths.push(path);

                // Batch is full
                if current_batch.len() >= actual_concurrent {
                    let is_parallel = actual_concurrent > 1;

                    batches.push(ScheduledBatch {
                        batch_id,
                        structures: std::mem::take(&mut current_batch),
                        topology_paths: std::mem::take(&mut current_paths),
                        concurrency: actual_concurrent,
                        estimated_memory_mb: actual_concurrent * memory_per_structure,
                        memory_tier: group.memory_tier,
                    });

                    if is_parallel {
                        stats.parallel_batches += 1;
                    } else {
                        stats.sequential_batches += 1;
                    }

                    stats.max_concurrency = stats.max_concurrency.max(actual_concurrent);
                    batch_id += 1;
                }
            }

            // Remaining structures in partial batch
            if !current_batch.is_empty() {
                let batch_size = current_batch.len();
                let is_parallel = batch_size > 1;

                batches.push(ScheduledBatch {
                    batch_id,
                    structures: current_batch,
                    topology_paths: current_paths,
                    concurrency: batch_size,
                    estimated_memory_mb: batch_size * memory_per_structure,
                    memory_tier: group.memory_tier,
                });

                if is_parallel {
                    stats.parallel_batches += 1;
                } else {
                    stats.sequential_batches += 1;
                }

                stats.max_concurrency = stats.max_concurrency.max(batch_size);
                batch_id += 1;
            }
        }

        ExecutionSchedule {
            total_structures: self.compositions.len(),
            estimated_total_batches: batches.len(),
            batches,
            stats,
        }
    }

    /// Get all compositions
    pub fn compositions(&self) -> &[StructureComposition] {
        &self.compositions
    }

    /// Print schedule summary
    pub fn print_schedule_summary(&self) {
        let schedule = self.schedule();

        println!("\n=== Batch Execution Schedule ===");
        println!("Total structures: {}", schedule.total_structures);
        println!("Total batches: {}", schedule.estimated_total_batches);
        println!();

        println!("Structure breakdown:");
        println!("  Small (<15K):  {}", schedule.stats.small_count);
        println!("  Medium (15-30K): {}", schedule.stats.medium_count);
        println!("  Large (>30K):  {}", schedule.stats.large_count);
        println!();

        println!("Batch statistics:");
        println!(
            "  Sequential batches: {}",
            schedule.stats.sequential_batches
        );
        println!("  Parallel batches: {}", schedule.stats.parallel_batches);
        println!("  Max concurrency: {}", schedule.stats.max_concurrency);
        println!();

        println!("Execution order:");
        for batch in &schedule.batches {
            let names = batch.structure_names().join(", ");
            println!(
                "  Batch {}: [{}] ({} concurrent, {} tier, ~{}MB)",
                batch.batch_id,
                names,
                batch.concurrency,
                match batch.memory_tier {
                    MemoryTier::Small => "Small",
                    MemoryTier::Medium => "Medium",
                    MemoryTier::Large => "Large",
                },
                batch.estimated_memory_mb
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Async Batch Executor (placeholder for future async implementation)
// ═══════════════════════════════════════════════════════════════════════════════

/// Result from batch execution
#[derive(Debug)]
pub struct BatchExecutionResult {
    /// Structure name
    pub structure_name: String,
    /// Success or failure
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
    /// Execution time in seconds
    pub elapsed_seconds: f64,
    /// Number of binding sites found
    pub sites_found: usize,
    /// Number of druggable sites
    pub druggable_sites: usize,
}

/// Batch execution summary
#[derive(Debug)]
pub struct BatchExecutionSummary {
    /// Results for each structure
    pub results: Vec<BatchExecutionResult>,
    /// Total execution time
    pub total_elapsed_seconds: f64,
    /// Successful structures
    pub successes: usize,
    /// Failed structures
    pub failures: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schedule_empty() {
        let scheduler = BatchScheduler::default_config();
        let schedule = scheduler.schedule();
        assert_eq!(schedule.total_structures, 0);
        assert_eq!(schedule.batches.len(), 0);
    }

    #[test]
    fn test_scheduler_config() {
        let config = BatchSchedulerConfig {
            max_gpu_memory_mb: 8000,
            max_concurrent_override: 2,
            prefer_throughput: true,
        };
        let scheduler = BatchScheduler::new(config);
        assert!(scheduler.is_empty());
    }
}
