//! # prism-gpu
//!
//! GPU acceleration layer for PRISM v2.
//! Optimized for PRISM-VE Benchmark.

pub mod active_inference;
pub mod bio_chemistry_gpu;
pub mod context;
pub mod dendritic_reservoir;
pub mod dendritic_snn;
pub mod feature_merge;
pub mod global_context;
pub mod lbs;
pub mod mega_fused;
pub mod mega_fused_batch;
pub mod molecular;
pub mod pimc;
pub mod polycentric_immunity;
pub mod prism_nova;
pub mod reservoir_construction;
pub mod tda;
pub mod thermodynamic;
pub mod ve_swarm;
pub mod viral_evolution_fitness;

// AMBER ff14SB bonded force calculator
pub mod amber_forces;

// Mega-fused AMBER HMC dynamics (full GPU acceleration)
pub mod amber_mega_fused;

// PME (Particle Mesh Ewald) for long-range electrostatics
pub mod cufft_sys;
pub mod pme;

// SETTLE constraint solver for rigid water
pub mod settle;

// H-bond constraint solver for protein X-H bonds
pub mod h_constraints;

// SIMD Batched AMBER MD (Tier 1: 10-50x throughput, identical physics)
pub mod amber_simd_batch;

// SOTA Performance Optimizations
// Verlet neighbor lists: 2-3× speedup for non-bonded force computation
pub mod verlet_list;
// Tensor Core (WMMA) accelerated force computation: 2-4× speedup
pub mod tensor_core_forces;
// Async CUDA streams: 1.1-1.3× speedup from latency hiding
pub mod async_md_pipeline;

// Revolutionary Ensemble Warp MD - True parallel clone processing
// Each WARP (32 threads) processes ONE CLONE independently
// Expected: N× speedup for N clones (theoretical limit)
pub mod ensemble_warp_md;

// Adaptive Cryo-Thermal Protocol for NHS-UV Cryptic Site Detection
// Three-phase protocol: CRYO BURST → THERMAL RAMP → FOCUSED DIG
// UV absorption → thermal signatures in frozen landscape
pub mod adaptive_protocol;

// Optimized Replica-Parallel MD with 2D Grid
// Grid: (ceil(n_atoms/256), n_replicas, 1), blockIdx.y = replica
// ~30% faster than work-pool 1D grid due to 95%+ cache efficiency
pub mod amber_replica_parallel;

// Essential exports
pub use active_inference::{ActiveInferenceGpu, ActiveInferencePolicy};
pub use adaptive_protocol::{
    AdaptiveProtocolConfig, AdaptiveProtocolState, ExplorationPhase, HotSpotCandidate,
};
pub use amber_forces::{
    AmberBondedForces, Angle, AngleParam, Bond, BondParam, Dihedral, DihedralParam,
    EnergyComponents, NB14Param, Pair14, TopologyBuilder,
};
pub use amber_mega_fused::{
    build_exclusion_lists as build_amber_exclusions,
    f16_bits_to_f32,
    f32_to_f16_bits,
    AmberMegaFusedHmc,
    ConstraintInfo,
    EnergyRecord,
    HmcRunResult,
    MixedPrecisionBuffers,
    // Phase 7: Mixed precision exports
    MixedPrecisionConfig,
    KB_KCAL_MOL_K,
    MAX_EXCLUSIONS as AMBER_MAX_EXCLUSIONS,
};
pub use amber_replica_parallel::{
    ReplicaDiagnostics, ReplicaFrameData, ReplicaParallelConfig, ReplicaParallelMD,
    ReplicaStepResult, SharedTopology, KB_KCAL_MOL_K as REPLICA_KB,
};
pub use amber_simd_batch::{
    compute_convergence_confidence, compute_rmsf_per_residue, merge_cv_cross_replica,
    merge_replica_frames, merge_rmsf_cross_replica, AmberSimdBatch, BatchMdResult,
    OptimizationConfig, ReplicaConfig, ReplicaFrame, ReplicaMergedResult, SotaStats,
    StructureTopology, BATCH_SPATIAL_OFFSET, MAX_BATCH_SIZE, NB_CUTOFF,
};
pub use async_md_pipeline::{
    AsyncMdPipeline, AsyncPipelineConfig, MdPhase, PipelineExecutor, PipelineStats, SyncPoint,
};
pub use bio_chemistry_gpu::{BiochemistryGpu, GpuAtomicMetadata, MAX_ATOMS as BIO_MAX_ATOMS};
pub use context::{GpuContext, GpuInfo, GpuSecurityConfig};
pub use dendritic_reservoir::DendriticReservoirGpu;
pub use dendritic_snn::{
    DendriticSNNReservoir, DEFAULT_RESERVOIR_SIZE, EXPANDED_INPUT_DIM as SNN_EXPANDED_INPUT_DIM,
    INPUT_DIM as SNN_INPUT_DIM,
};
pub use ensemble_warp_md::{
    topology_from_prism_prep, EnsembleResult, EnsembleTopology, EnsembleWarpMd, MAX_ATOMS_WARP,
    WARP_SIZE,
};
pub use feature_merge::{FeatureMergeConfig, FeatureMergeGpu, FeatureMergeResult};
pub use global_context::{GlobalGpuContext, GlobalGpuError};
pub use h_constraints::{build_h_clusters, ClusterType, HConstraintCluster, HConstraints};
pub use lbs::LbsGpu;
pub use lcpo_sasa::{
    elements_to_atom_types, elements_to_radii, AtomType as LcpoAtomType, BatchedSasaResult,
    LcpoSasaGpu, SasaResult,
};
pub use mega_fused::{
    confidence, signals, GpuProvenanceData, GpuTelemetry, KernelTelemetryEvent, MegaFusedConfig,
    MegaFusedGpu, MegaFusedMode, MegaFusedOutput, MegaFusedParams,
};
pub use mega_fused_batch::{
    BatchOutput, BatchStructureDesc, BatchStructureOutput, CountryImmunityTimeSeriesV2,
    ImmunityMetadataV2, MegaFusedBatchGpu, PackedBatch, PkParams, StructureInput,
    StructureMetadata, TrainingOutput,
};
pub use memory::{global_vram_guard, init_global_vram_guard, VramGuard, VramGuardError, VramInfo};
pub use molecular::{MDParams, MDResults, MolecularDynamicsGpu, Particle};
pub use pme::{compute_ewald_beta, DEFAULT_PME_TOLERANCE, PME};
pub use polycentric_immunity::{
    PolycentricImmunityGpu, DEFAULT_CROSS_REACTIVITY, N_EPITOPE_CENTERS, N_PK_SCENARIOS,
    POLYCENTRIC_OUTPUT_DIM,
};
pub use prism_nova::{
    NovaConfig, NovaStepResult, PrismNova, RESERVOIR_SIZE as NOVA_RESERVOIR_SIZE,
};
pub use reservoir_construction::{compute_readout_weights, BioReservoir, SparseConnection};
pub use settle::Settle;
pub use tensor_core_forces::{TensorCoreForces, TC_BLOCK_SIZE, TC_TILE_SIZE};
pub use verlet_list::{
    VerletList, MAX_NEIGHBORS_PER_ATOM, VERLET_LIST_CUTOFF, VERLET_SKIN, VERLET_SKIN_HALF,
};
pub use viral_evolution_fitness::FitnessParams;
pub use whcr::{RepairResult as WhcrRepairResult, WhcrGpu};

// Commented out unused modules to isolate benchmark requirements
// pub mod aatgs;
// pub mod aatgs_integration;
// pub mod cma;
// pub mod cma_es;
// pub mod cryptic_gpu;
// pub mod glycan_gpu;
// pub mod dendritic_whcr;
// pub mod floyd_warshall;
// pub mod lbs;
// pub mod readout_training;
// pub mod molecular;
// pub mod multi_device_pool;
// pub mod multi_gpu;
// pub mod multi_gpu_integration;
// pub mod pimc;
// pub mod quantum;
pub mod stream_integration;
pub mod stream_manager;

// LCPO SASA - GPU-accelerated solvent accessible surface area for cryptic site detection
pub mod lcpo_sasa;

// VRAM Safety Guard - Battle-tested memory management
pub mod memory;
// pub mod tda;
// pub mod thermodynamic;
// pub mod transfer_entropy;
// pub mod ultra_kernel;
pub mod whcr; // Re-enabled for prism-whcr dependency
              // pub mod batch_tda;
              // pub mod mega_fused_integrated;
              // pub mod training;
