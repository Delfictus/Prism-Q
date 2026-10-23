/**
 * SDST - Spike-Driven Sparse Temporal Hash
 * 
 * Complete spatial-temporal data structure for PRISM-Therm.
 * Replaces OptiX BVH with a neuromorphic-native event-sourced
 * hash table that encodes spike causality, thermodynamic context,
 * and wavefront coherence directly in the data structure.
 *
 * Designed for:
 *   - 128³ NHS grid at 0.75Å spacing
 *   - 5-phase cryo-thermal hysteresis (50K→300K→50K)
 *   - 55K total simulation steps
 *   - RTX 5080 (SM120, 16GB GDDR7)
 *   - 4-stream CUDA concurrency
 *
 * Memory footprint: ~16-50MB typical (vs ~4GB for naive SVH)
 *
 * Copyright (c) 2026 Delfictus IO LLC. All rights reserved.
 */

#ifndef SDST_API_H
#define SDST_API_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * Error handling
 * ============================================================ */

typedef enum {
    SDST_SUCCESS = 0,
    SDST_ERROR_CUDA = -1,
    SDST_ERROR_OOM = -2,
    SDST_ERROR_INVALID_PARAM = -3,
    SDST_ERROR_TABLE_FULL = -4,
    SDST_ERROR_NOT_FOUND = -5,
    SDST_ERROR_WAVEFRONT_OVERFLOW = -6,
    SDST_ERROR_STREAM_INVALID = -7,
} SdstError;

const char* sdst_error_string(SdstError err);

/* ============================================================
 * Core types
 * ============================================================ */

/** Morton-encoded voxel coordinate (21 bits for 128³) */
typedef uint32_t MortonCode;

/** Unique spike event identifier */
typedef uint32_t SpikeId;

/** Wavefront identifier */
typedef uint32_t WavefrontId;

/** Avalanche cluster identifier */
typedef uint32_t AvalancheId;

/** Hysteresis phase (0-4 for the 5-phase cycle) */
typedef uint8_t PhaseId;

/**
 * Complete spike event - the fundamental unit of PRISM-Therm data.
 * 32 bytes, cache-line aligned for coalesced GPU access.
 */
/* Repacked layout: u32s first, then u16s, then u8s.
 * Eliminates internal padding. sizeof = 36 bytes (34 fields + 2 trailing).
 * aligned(4) lets aligned(32) be applied at the allocation site (cudaMalloc
 * returns 256-byte aligned memory, so arrays are automatically cache-line safe).
 * GATE 0 verified: aligned(32) inflates sizeof to 64; aligned(4) gives 36. */
typedef struct __attribute__((aligned(4))) {
    /* --- u32 group (20 bytes, offsets 0-19) --- */
    MortonCode  voxel;              /** Morton-encoded (x,y,z) */
    uint32_t    timestamp;          /** Simulation step */
    SpikeId     parent_spike;       /** Causal parent (0 = spontaneous) */
    AvalancheId avalanche_id;       /** Avalanche membership */
    WavefrontId wavefront_id;       /** Coherent wavefront membership */

    /* --- u16 group (12 bytes, offsets 20-31) --- */
    uint16_t    amplitude;          /** Spike amplitude (f16 stored as u16) */
    uint16_t    local_temp;         /** Local effective temperature (f16) */
    uint16_t    energy_gradient;    /** |∇E| at this voxel (f16) */
    uint16_t    solvent_exposure;   /** SASA proxy from NHS density (f16) */
    uint16_t    wavefront_velocity; /** Local propagation speed (f16) */
    uint16_t    wavefront_coherence;/** Spatial correlation with neighbors (f16) */

    /* --- u8 group (2 bytes, offsets 32-33) + 2 implicit trailing pad → 36 --- */
    PhaseId     phase_id;           /** Hysteresis phase 0-4 */
    uint8_t     tcl_flags;          /** Bit flags: is_transition, is_boundary, etc */
} SpikeEvent;

/* C++17 static_assert — _Static_assert is C11 only, doesn't compile under -std=c++17 */
static_assert(sizeof(SpikeEvent) == 36, "SpikeEvent must be 36 bytes (verified by GATE 0)");

/** Hash table entry: Morton key + index into event buffer */
typedef struct {
    uint32_t key;       /** Morton code (upper bit = occupied flag) */
    uint32_t head_idx;  /** Index of most recent spike in this voxel's chain */
} HashEntry;

/** Avalanche statistics for CCNS */
typedef struct {
    AvalancheId id;
    uint32_t    size;           /** Number of spikes in avalanche */
    uint32_t    duration;       /** Timesteps from first to last spike */
    float       spatial_extent; /** Bounding radius in Angstroms */
    MortonCode  seed_voxel;     /** Where the avalanche started */
    PhaseId     phase;          /** Which hysteresis phase */
    float       tau_local;      /** Local criticality exponent */
} AvalancheStats;

/** Wavefront descriptor */
typedef struct {
    WavefrontId id;
    MortonCode  origin;             /** Where wavefront started */
    uint32_t    birth_time;         /** When it started */
    uint32_t    death_time;         /** When it dissipated (0 = still active) */
    uint32_t    spike_count;        /** How many spikes in this wavefront */
    float       mean_velocity;      /** Average propagation speed */
    float       mean_coherence;     /** Average spatial coherence */
    float       spatial_extent;     /** Max radius reached */
    PhaseId     phase;              /** Hysteresis phase */
} WavefrontStats;

/** Hysteresis asymmetry result for a spatial region */
typedef struct {
    float       heating_spike_rate;     /** Spikes/step during heating phases */
    float       cooling_spike_rate;     /** Spikes/step during cooling phases */
    float       asymmetry_score;        /** |heating - cooling| / (heating + cooling) */
    float       avalanche_size_ratio;   /** Mean avalanche size heating / cooling */
    float       wavefront_coherence_ratio;
    uint32_t    heating_spike_count;
    uint32_t    cooling_spike_count;
    bool        is_hysteretic;          /** asymmetry_score > threshold */
} HysteresisResult;

/** CCNS classification */
typedef enum {
    CCNS_SOC = 0,           /** tau < 1.5: Self-Organized Critical */
    CCNS_NEAR_CRITICAL = 1, /** 1.5 <= tau < 2.0 */
    CCNS_BARRIER = 2,       /** tau >= 2.0 */
} CcnsClass;

/** Per-region CCNS result */
typedef struct {
    float       tau;            /** Power-law exponent */
    CcnsClass   classification;
    float       tau_stderr;     /** Standard error on tau fit */
    uint32_t    n_avalanches;   /** Sample size */
    float       druggability;   /** Composite score */
} CcnsResult;

/** Spatial query region (axis-aligned bounding box in grid coords) */
typedef struct {
    uint32_t x_min, x_max;
    uint32_t y_min, y_max;
    uint32_t z_min, z_max;
} SpatialRegion;

/** Causal subgraph extraction result */
typedef struct {
    SpikeEvent* events;         /** Array of events in subgraph */
    uint32_t    count;          /** Number of events */
    uint32_t*   parent_indices; /** Index into events[] for each event's parent */
} CausalSubgraph;

/** Plan C (TIDE) export: per-residue causal energy decomposition */
typedef struct {
    uint32_t    residue_id;
    float       causal_dG;          /** Transfer entropy-weighted ΔG contribution */
    float       transfer_entropy;   /** TE from this residue to pocket */
    float       fisher_info;        /** Fisher information (leverage) */
    float       kl_divergence;      /** Conformational reorganization cost */
    uint32_t    n_causal_spikes;    /** Residue-attributed events inside the strict ROI; not parent-edge count */
} TideDecomposition;

/** Companion v1 receipt for the existing estimator; no TideDecomposition ABI change.
 * unavailable_flags: 1 insufficient bins, 2/4 source <=.05/>=.95 occupancy,
 * 8/16 target <=.05/>=.95 occupancy, 32 nonfinite computed result.
 * Joint counts order is [target_next][target_current][source_current]. */
typedef struct {
    uint32_t residue_id;
    uint32_t bin_width_steps;
    uint32_t max_bins;
    uint32_t n_bins;
    uint32_t source_active_bins;
    uint32_t target_active_bins;
    uint32_t source_binned_spikes;
    uint32_t target_binned_spikes;
    uint32_t source_outside_bin_range;
    uint32_t target_outside_bin_range;
    uint32_t unavailable_flags;
    uint32_t variance_window_bins;
    uint32_t variance_total_windows;
    uint32_t variance_valid_windows;
    uint32_t joint_counts[8];
} TideMeasurementSupportV1;


/* ============================================================
 * Configuration
 * ============================================================ */

typedef struct {
    /* Grid dimensions */
    uint32_t grid_nx, grid_ny, grid_nz;  /** Default: 128, 128, 128 */
    float    grid_spacing;                /** Default: 0.75 Å */

    /* Hash table sizing */
    uint32_t hash_table_capacity;   /** Slots. Default: 4194304 (2^22, ~2x grid) */
    uint32_t max_spike_events;      /** Max events before flush. Default: 2000000 */

    /* Wavefront tracking */
    uint32_t max_wavefronts;        /** Default: 65536 */
    float    wavefront_merge_dist;  /** Spatial proximity for wavefront propagation (Å) */
    float    wavefront_max_dt;      /** Max timestep gap for parent→child wavefront */

    /* Avalanche detection */
    float    avalanche_spatial_cutoff;   /** Max distance for avalanche membership (Å) */
    uint32_t avalanche_max_gap;          /** Max timestep gap within avalanche */

    /* Hysteresis phases: step ranges for each phase */
    uint32_t phase_boundaries[6];   /** [0, p1_end, p2_end, p3_end, p4_end, p5_end] */

    /* CCNS */
    float    ccns_soc_threshold;         /** Default: 1.5 */
    float    ccns_barrier_threshold;     /** Default: 2.0 */

    /* CUDA */
    uint32_t num_streams;           /** Default: 4 */
    int      device_id;             /** Default: 0 */
} SdstConfig;

/** Get default config for 128³ PRISM-Therm */
SdstConfig sdst_default_config(void);

/* ============================================================
 * Lifecycle
 * ============================================================ */

/** Opaque handle to SDST instance */
typedef struct SdstContext* SdstHandle;

/** Initialize SDST on GPU. Allocates all memory. */
SdstError sdst_create(const SdstConfig* config, SdstHandle* out_handle);

/** Destroy SDST and free all GPU memory. */
SdstError sdst_destroy(SdstHandle handle);

/** Reset all data (keep allocations). For new simulation run. */
SdstError sdst_reset(SdstHandle handle);

/** Get current event count */
SdstError sdst_event_count(SdstHandle handle, uint32_t* out_count);

/** Get GPU memory usage in bytes */
SdstError sdst_memory_usage(SdstHandle handle, size_t* out_bytes);

/* ============================================================
 * Spike Insertion (called from NHS engine per timestep)
 * ============================================================ */

/**
 * Insert a batch of spike events from the current timestep.
 * This is the primary ingestion path - called once per simulation step.
 *
 * @param handle        SDST instance
 * @param events        Device pointer to array of SpikeEvent (caller fills
 *                      voxel, timestamp, amplitude, local_temp, energy_gradient,
 *                      solvent_exposure, phase_id; SDST fills the rest)
 * @param count         Number of events in this batch
 * @param stream        CUDA stream for this operation
 *
 * SDST will:
 *   1. Compute parent_spike via spatial-temporal proximity lookup
 *   2. Assign avalanche_id via GPU union-find
 *   3. Propagate or create wavefront_id
 *   4. Compute wavefront_velocity and wavefront_coherence
 *   5. Insert into hash table with Morton-keyed chaining
 */
SdstError sdst_insert_spikes(
    SdstHandle handle,
    SpikeEvent* d_events,   /* device pointer */
    uint32_t count,
    void* stream            /* cudaStream_t */
);

/**
 * Lightweight insertion: caller provides minimal data, SDST computes everything.
 * Use when integrating with existing NHS engine that doesn't construct full SpikeEvents.
 */
typedef struct {
    uint32_t voxel_x, voxel_y, voxel_z;
    uint32_t timestamp;
    float    amplitude;
    float    local_temp;
    float    energy_gradient;
    float    solvent_exposure;
    uint8_t  phase_id;
} SpikeInput;

SdstError sdst_insert_raw(
    SdstHandle handle,
    const SpikeInput* d_inputs,  /* device pointer */
    uint32_t count,
    void* stream
);

/**
 * GPU-native insertion from NHS raw spike buffer.
 *
 * Eliminates the CPU conversion round-trip. Takes a HOST pointer to
 * the accumulated GpuSpikeEvent[] array (sorted by timestep on CPU),
 * uploads in one bulk transfer, converts on GPU, and inserts in
 * temporal batches of avalanche_max_gap timesteps for efficient
 * parent detection.
 *
 * @param h_nhs_events  HOST pointer to sorted GpuSpikeEvent[]
 * @param count         Number of events
 * @param nhs_stride    sizeof(GpuSpikeEvent) = 92
 * @param start_temp    Protocol cold temperature (K)
 * @param end_temp      Protocol warm temperature (K)
 * @param cold_hold     Protocol: cold_hold_steps
 * @param ramp_up       Protocol: ramp_steps
 * @param warm_hold     Protocol: warm_hold_steps
 * @param ramp_down     Protocol: ramp_down_steps
 */
SdstError sdst_insert_from_nhs_buffer(
    SdstHandle handle,
    const void* h_nhs_events,
    uint32_t count,
    uint32_t nhs_stride,
    float start_temp,
    float end_temp,
    uint32_t cold_hold,
    uint32_t ramp_up,
    uint32_t warm_hold,
    uint32_t ramp_down,
    void* stream
);

/* ============================================================
 * Spatial Queries
 * ============================================================ */

/**
 * Get all spike events within a spatial region.
 * Returns device pointer to matching events (valid until next query).
 */
SdstError sdst_query_region(
    SdstHandle handle,
    const SpatialRegion* region,
    SpikeEvent** out_events,    /* device pointer to results */
    uint32_t* out_count,
    void* stream
);

/**
 * Get all spike events for a specific voxel (by grid coords).
 * Returns temporal chain from most recent to oldest.
 */
SdstError sdst_query_voxel(
    SdstHandle handle,
    uint32_t x, uint32_t y, uint32_t z,
    SpikeEvent** out_events,
    uint32_t* out_count,
    void* stream
);

/**
 * Get all spike events within a time window.
 */
SdstError sdst_query_timerange(
    SdstHandle handle,
    uint32_t t_start, uint32_t t_end,
    SpikeEvent** out_events,
    uint32_t* out_count,
    void* stream
);

/**
 * Combined spatial + temporal query.
 */
SdstError sdst_query_region_timerange(
    SdstHandle handle,
    const SpatialRegion* region,
    uint32_t t_start, uint32_t t_end,
    SpikeEvent** out_events,
    uint32_t* out_count,
    void* stream
);

/* ============================================================
 * Causal Graph Queries
 * ============================================================ */

/**
 * Extract the full causal subgraph rooted at a specific spike.
 * Follows parent_spike chains to reconstruct the causal tree.
 */
SdstError sdst_causal_subgraph(
    SdstHandle handle,
    SpikeId root_spike,
    uint32_t max_depth,     /** 0 = unlimited */
    CausalSubgraph* out_graph,
    void* stream
);

/**
 * Extract causal subgraph for all spikes in a spatial region.
 * This answers: "what caused the activity in this pocket?"
 */
SdstError sdst_causal_subgraph_region(
    SdstHandle handle,
    const SpatialRegion* region,
    uint32_t max_depth,
    CausalSubgraph* out_graph,
    void* stream
);

/**
 * Free a causal subgraph allocated by the above functions.
 */
SdstError sdst_free_subgraph(CausalSubgraph* graph);

/* ============================================================
 * Avalanche / CCNS Analysis
 * ============================================================ */

/**
 * Get statistics for all avalanches (or filtered by phase).
 * @param phase_filter  -1 = all phases, 0-4 = specific phase
 */
SdstError sdst_avalanche_stats(
    SdstHandle handle,
    int phase_filter,
    AvalancheStats** out_stats,  /* host pointer, caller frees */
    uint32_t* out_count,
    void* stream
);

/**
 * Compute CCNS tau exponent for a spatial region.
 * Uses avalanche size distribution within the region.
 */
SdstError sdst_ccns_region(
    SdstHandle handle,
    const SpatialRegion* region,
    CcnsResult* out_result,
    void* stream
);

/**
 * Compute CCNS for all detected pocket regions (LEGACY — host-side, O(N) per tile).
 * Pockets are identified as spatially contiguous clusters of high spike density.
 * WARNING: Downloads all events to host. Use sdst_ccns_all_pockets_gpu() instead.
 */
SdstError sdst_ccns_all_pockets(
    SdstHandle handle,
    CcnsResult** out_results,   /* host pointer, caller frees */
    SpatialRegion** out_regions,
    uint32_t* out_count,
    void* stream
);

/**
 * GPU-native CCNS for all spatial tiles (replaces sdst_ccns_all_pockets).
 *
 * Sort-reduce architecture: CUB radix sort → tile segmentation → fused
 * per-tile CSN estimator (Clauset-Shalizi-Newman truncated power law).
 * Zero host-side event downloads. All 13M+ events stay on GPU.
 *
 * @param out_results   Host pointer to CcnsResult array (caller frees)
 * @param out_regions   Host pointer to SpatialRegion array (caller frees)
 * @param out_count     Number of valid tiles returned
 */
SdstError sdst_ccns_all_pockets_gpu(
    SdstHandle handle,
    CcnsResult** out_results,
    SpatialRegion** out_regions,
    uint32_t* out_count,
    void* stream
);

/**
 * GPU-native CCNS for caller-specified spatial regions (batched).
 *
 * Reuses Stage 1 (avalanche sizes) from the global pipeline, then
 * sorts events by region membership and runs the fused CSN estimator
 * per region. One GPU call for all NHS pocket regions.
 *
 * @param regions       Host pointer to array of SpatialRegion (pocket bounding boxes)
 * @param n_regions     Number of regions
 * @param out_results   Caller-allocated array of n_regions CcnsResult
 */
SdstError sdst_ccns_for_regions(
    SdstHandle handle,
    const SpatialRegion* regions,
    uint32_t n_regions,
    CcnsResult* out_results,
    void* stream
);

/* ============================================================
 * Wavefront Analysis
 * ============================================================ */

/**
 * Get statistics for all wavefronts (or filtered by phase).
 */
SdstError sdst_wavefront_stats(
    SdstHandle handle,
    int phase_filter,
    WavefrontStats** out_stats,
    uint32_t* out_count,
    void* stream
);

/**
 * Get the wavefront propagation path: ordered sequence of voxels
 * the wavefront traversed, with timing.
 */
SdstError sdst_wavefront_path(
    SdstHandle handle,
    WavefrontId wavefront,
    SpikeEvent** out_events,
    uint32_t* out_count,
    void* stream
);

/**
 * Find wavefronts that passed through a specific region.
 * Answers: "what coherent events hit this pocket?"
 */
SdstError sdst_wavefronts_through_region(
    SdstHandle handle,
    const SpatialRegion* region,
    WavefrontStats** out_stats,
    uint32_t* out_count,
    void* stream
);

/* ============================================================
 * Hysteresis Analysis (PRISM-Therm core)
 * ============================================================ */

/**
 * Compute hysteresis asymmetry for a spatial region.
 * Compares spike activity between heating and cooling phases.
 */
SdstError sdst_hysteresis_region(
    SdstHandle handle,
    const SpatialRegion* region,
    float asymmetry_threshold,  /** Default: 0.2 */
    HysteresisResult* out_result,
    void* stream
);

/**
 * Scan entire grid for hysteretic regions.
 * Returns all regions where asymmetry_score > threshold.
 * These are candidate cryptic binding sites.
 */
SdstError sdst_hysteresis_scan(
    SdstHandle handle,
    float asymmetry_threshold,
    HysteresisResult** out_results,
    SpatialRegion** out_regions,
    uint32_t* out_count,
    void* stream
);

/* ============================================================
 * Plan C (PRISM-TIDE) Export
 * ============================================================ */

/**
 * Compute transfer entropy-based causal ΔG decomposition
 * for a target pocket region.
 *
 * For each residue that has causal spike connections to the pocket,
 * computes the transfer entropy, Fisher information, and KL divergence.
 *
 * @param pocket_region     The target binding site region
 * @param h_residue_map     Host pointer: dense linear-voxel-indexed → residue ID.
 *                          Size = grid_nx × grid_ny × grid_nz.
 *                          Index = x + y * grid_nx + z * grid_nx * grid_ny.
 *                          Use UINT32_MAX for empty (no-residue) voxels.
 * @param n_residues        Total number of residues
 */
SdstError sdst_tide_decomposition(
    SdstHandle handle,
    const SpatialRegion* pocket_region,
    const uint32_t* h_residue_map,  /* host pointer, linear-voxel-indexed */
    uint32_t n_residues,
    TideDecomposition** out_decomp,  /* host pointer, caller frees */
    uint32_t* out_count,
    void* stream
);

/** Same computation and row population as sdst_tide_decomposition, with
 * one same-order diagnostic receipt per returned row. Caller frees both arrays. */
SdstError sdst_tide_decomposition_with_support_v1(
    SdstHandle handle, const SpatialRegion* pocket_region,
    const uint32_t* h_residue_map, uint32_t n_residues,
    TideDecomposition** out_decomp, TideMeasurementSupportV1** out_support,
    uint32_t* out_count, void* stream
);

/* ============================================================
 * DCC (Distance to Closest Contact) Integration
 * ============================================================ */

/**
 * Compute DCC for detected pocket centroids against known sites.
 *
 * @param known_sites       Array of (x,y,z) in Angstroms for known binding sites
 * @param n_known           Number of known sites
 * @param out_dcc           DCC values per detected pocket (host, caller frees)
 * @param out_centroids     Detected pocket centroids in Angstroms (host, caller frees)
 * @param out_n_detected    Number of detected pockets
 */
SdstError sdst_compute_dcc(
    SdstHandle handle,
    const float* known_sites,   /* [n_known][3] */
    uint32_t n_known,
    float** out_dcc,
    float** out_centroids,
    uint32_t* out_n_detected,
    void* stream
);

/* ============================================================
 * Serialization (checkpoint/resume)
 * ============================================================ */

/** Save complete SDST state to file */
SdstError sdst_save(SdstHandle handle, const char* filepath);

/** Load SDST state from file */
SdstError sdst_load(const char* filepath, SdstHandle* out_handle);

/* ============================================================
 * Debug / Diagnostics
 * ============================================================ */

/** Print hash table occupancy and event buffer statistics */
SdstError sdst_print_stats(SdstHandle handle);

/** Validate internal consistency (debug builds) */
SdstError sdst_validate(SdstHandle handle);

#ifdef __cplusplus
}
#endif

#endif /* SDST_API_H */
