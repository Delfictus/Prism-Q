//=============================================================================
// PRISM-LBS MEGA-FUSED BATCH KERNEL
// Processes MULTIPLE structures in a SINGLE kernel launch
//
// Key Optimizations:
// - One thread block per structure (up to 512 structures per launch)
// - L1 cache preference for read-only global data (__ldg)
// - Maximum register usage for intermediate computations
// - Minimal global memory traffic
// - Persistent shared memory reuse
//
// Performance Target: 221 structures in < 100ms (vs 2+ seconds sequential)
//
//=============================================================================
// 136-DIMENSIONAL COMBINED FEATURE INDEX MAP (P0 EXTENDED)
//=============================================================================
// Index Range   | Count | Name              | Description
// --------------|-------|-------------------|----------------------------------
//   0 -  47     |  48   | TDA               | Topological Data Analysis features
//  48 -  79     |  32   | Reservoir         | Dendritic reservoir state
//  80 -  91     |  12   | Physics           | Electrostatics, hydrophobicity
//  92 -  95     |   4   | Fitness           | ddG_bind(92), ddG_stab(93),
//               |       |                   | expression(94), transmit(95)
//  96 - 100     |   5   | Cycle             | phase(96), emergence_prob(97),
//               |       |                   | time_to_peak(98), freq(99), vel(100)
// 101 - 108     |   8   | Spike             | LIF neuron density outputs
// 109 - 124     |  16   | Immunity          | 10 epitope values (109-118) +
//               |       |                   | 6 derived: gamma(119), fold_red(120),
//               |       |                   | pop_imm_avg(121), days_since(122),
//               |       |                   | ab_titer_norm(123), escape_press(124)
// 125 - 135     |  11   | Epi               | Competition(125-127): freq_rank_norm,
//               |       |                   |   gamma_deficit, suppression_pressure
//               |       |                   | Momentum(128-130): log_slope_7d,
//               |       |                   |   log_slope_28d, acceleration
//               |       |                   | Immunity(131-134): days_since_vaccine,
//               |       |                   |   days_since_wave, immunity_derivative,
//               |       |                   |   immunity_source_ratio
//               |       |                   | Country(135): country_id_norm
// --------------|-------|-------------------|----------------------------------
// Total         | 136   |                   | TOTAL_COMBINED_FEATURES
//=============================================================================

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "prism_numerics.cuh"  // P0 PATCH: Compensated accumulation for precision

namespace cg = cooperative_groups;

//=============================================================================
// BATCH CONFIGURATION
//=============================================================================

#define TILE_SIZE 32
#define BLOCK_SIZE 256
#define MAX_RESIDUES 2048
#define MAX_STRUCTURES 512
#define WARP_SIZE 32

// P0 PATCH: Escape score computation constants
#define MAX_EPITOPE_CLASSES 10
#define MAX_ANTIBODIES 836
#define RBD_SITES 201
#define MAX_MUTATIONS_PER_VARIANT 50

// Reservoir configuration - Enhanced with multi-compartment dendritic dynamics
// Based on dendritic_whcr.cu: 4 compartments with varied time constants
#define RESERVOIR_DIM 32       // 32 reservoir neurons (balanced for register pressure)
#define N_BRANCHES 4           // Dendritic branches: local, neighbor, global, recurrent
#define N_COMPARTMENTS 4       // Multi-compartment: proximal, distal1, distal2, spine
#define N_INPUT_FEATURES 12    // Input features per residue (8 base + 4 geometry)
#define RESERVOIR_OUTPUT_DIM 32  // Output dimension for training (geometry+dendritic+reservoir)
#define KEMPE_CHAIN_MAX 128

// PRISM>4D Feature Dimensions
#define TDA_FEATURE_COUNT 48           // TDA topology features
#define PHYSICS_FEATURE_COUNT 12       // Physics features (electrostatics, hydrophobicity)
#define FITNESS_FEATURE_COUNT 4        // Fitness: ddG_bind, ddG_stab, expression, transmit
#define CYCLE_FEATURE_COUNT 5          // Cycle: phase, emergence_prob, time_to_peak, freq, vel
#define SPIKE_FEATURE_COUNT 8          // Spike: LIF neuron densities from Stage 8.5
#define IMMUNITY_FEAT_OUT 16           // Immunity: 10 epitopes + 6 derived (gamma, fold_red, etc.)
#define EPI_FEATURE_COUNT 11           // P0: Competition(3) + Momentum(3) + ImmRecency(4) + Country(1)
#define TOTAL_COMBINED_FEATURES 136    // 48 + 32 + 12 + 4 + 5 + 8 + 16 + 11 = 136

// Multi-compartment time constants (from dendritic_whcr.cu)
// Each compartment has different temporal dynamics for binding site detection
#define TAU_PROXIMAL 0.1f      // Fast response (current conflicts)
#define TAU_DISTAL_1 0.5f      // Medium memory (recent patterns)
#define TAU_DISTAL_2 0.85f     // Slow integration (structural features)
#define TAU_SPINE 0.95f        // Long-term memory (binding site persistence)

//=============================================================================
// RUNTIME PARAMETERS (Same as single-structure kernel)
//=============================================================================

struct __align__(16) MegaFusedParams {
    float contact_cutoff;
    float contact_sigma;
    int power_iterations;
    int kempe_iterations;
    float thresh_geometric;
    float thresh_conservation;
    float thresh_centrality;
    float thresh_flexibility;
    int min_signals;
    float consensus_threshold;
    float branch_weight_local;
    float branch_weight_neighbor;
    float branch_weight_global;
    float branch_weight_recurrent;
    float recurrent_decay;
    float consensus_weight_geometric;
    float consensus_weight_conservation;
    float consensus_weight_centrality;
    float consensus_weight_flexibility;
    float signal_bonus_0;
    float signal_bonus_1;
    float signal_bonus_2;
    float signal_bonus_3;
    float confidence_high_score;
    float confidence_medium_score;
    int confidence_high_signals;
    int confidence_medium_signals;
    float kempe_contact_threshold;
    float kempe_swap_threshold;
    float centrality_degree_weight;
    float centrality_eigenvector_weight;

    //=========================================================================
    // QUALITY CONTROL (QC) GATE PARAMETERS
    // These enforce scientifically validated thresholds from hyper-tuning.
    // They act as QC gates to filter noise and ensure druggable binding sites.
    //=========================================================================
    float min_pocket_volume;      // Minimum pocket volume in Å³ (default: 160.0)
    float max_pocket_volume;      // Maximum pocket volume in Å³ (default: 4800.0)
    float min_druggability;       // Minimum druggability score (default: 0.60)
    int max_pocket_residues;      // Maximum residues per pocket (default: 80)
    int max_pockets;              // Maximum pockets to return (default: 10)
};

//=============================================================================
// BATCH STRUCTURE DESCRIPTOR
// Describes one structure within the packed batch
//=============================================================================

struct __align__(16) BatchStructureDesc {
    int atom_offset;      // Start index in packed atoms array
    int residue_offset;   // Start index in packed residue arrays
    int n_atoms;          // Number of atoms in this structure
    int n_residues;       // Number of residues in this structure
};

//=============================================================================
// CONSTANT MEMORY (Pre-loaded neural network weights - 64 reservoir neurons)
//=============================================================================

// Multi-compartment time constants in constant memory for fast access
__constant__ float c_tau_decay[N_COMPARTMENTS] = {
    TAU_PROXIMAL, TAU_DISTAL_1, TAU_DISTAL_2, TAU_SPINE
};

// Compartment integration weights (proximal strongest, spine weakest)
__constant__ float c_compartment_weights[N_COMPARTMENTS] = {
    1.0f, 0.6f, 0.3f, 0.1f
};

// P0 PATCH: Escape score computation
// NOTE: Large escape matrices must be passed as global memory arguments
// due to constant memory limit (64KB max, matrix needs ~672KB)
// Use launch_epi_features() from prism_epi_features.cuh for escape computation

/*
// P0 PATCH: ATOMIC-FREE ESCAPE SCORE REDUCTION
// DISABLED: Requires global memory for escape_matrix (672KB > 64KB constant limit)
// Use launch_epi_features() from prism_epi_features.cuh instead
__device__ void batch_compute_escape_scores_v2(...) { ... }
*/

//=============================================================================
// P0 PATCH: COMPENSATED TEMPORAL INTEGRATION FOR GAMMA SERIES
//=============================================================================

__device__ float integrate_gamma_window_compensated(
    const float* __restrict__ gamma_window,
    const float* __restrict__ freq_window,
    int n_samples,
    float dt
) {
    prism::CompensatedAccumulator S;

    #pragma unroll 4
    for (int t = 0; t < n_samples; t++) {
        const float g = gamma_window[t];
        const float f = freq_window[t];

        // Logistic growth: dS = γ × F × (1-F) × dt
        const float dS = g * f * (1.0f - f) * dt;
        S.add(dS);
    }

    return S.finalize();
}

//=============================================================================
// STAGE 9-10: IMMUNITY DYNAMICS + 600-DAY INTEGRAL (VASIL GPU Implementation)
// GPU-accelerated pharmacokinetic model and immunity integral computation
//=============================================================================

// Immunity model dimensions
#define N_EPITOPES 10              // A, B, C, D1, D2, E12, E3, F1, F2, F3
#define N_COUNTRIES 12             // VASIL countries
#define N_PK_SCENARIOS 3           // Fast, Medium, Slow antibody decay
#define IMMUNITY_WINDOW_DAYS 600   // 600-day lookback window
#define IMMUNITY_FEATURE_COUNT 16  // 10 epitopes + 6 derived features
#define WEEKLY_SAMPLES 86          // ~600 days / 7 days = 86 weekly samples

// Antibody Pharmacokinetic parameters (t_half, t_max in days)
// Scenarios: 0=Fast (young), 1=Medium (default), 2=Slow (elderly)
__constant__ float c_pk_t_half[N_PK_SCENARIOS] = { 25.0f, 45.0f, 69.0f };
__constant__ float c_pk_t_max[N_PK_SCENARIOS] = { 14.0f, 21.0f, 28.0f };

// VASIL gamma weights (escape vs transmissibility)
__constant__ float c_alpha_escape = 0.65f;
__constant__ float c_beta_transmit = 0.35f;

// Cross-reactivity matrix: prior[i] → target[j] (10x10 for variant families)
// Rows: Wuhan, Alpha, Beta, Gamma, Delta, BA.1, BA.2, BA.45, BQ.1, XBB
__constant__ float c_cross_reactivity[10][10] = {
    // Wuhan immunity vs all
    {1.00f, 0.85f, 0.40f, 0.50f, 0.70f, 0.15f, 0.12f, 0.08f, 0.05f, 0.03f},
    // Alpha immunity
    {0.80f, 1.00f, 0.35f, 0.45f, 0.65f, 0.12f, 0.10f, 0.07f, 0.04f, 0.02f},
    // Beta immunity
    {0.35f, 0.30f, 1.00f, 0.70f, 0.40f, 0.25f, 0.22f, 0.15f, 0.10f, 0.08f},
    // Gamma immunity
    {0.45f, 0.40f, 0.65f, 1.00f, 0.45f, 0.22f, 0.20f, 0.12f, 0.08f, 0.06f},
    // Delta immunity
    {0.60f, 0.55f, 0.35f, 0.40f, 1.00f, 0.18f, 0.15f, 0.10f, 0.06f, 0.04f},
    // BA.1 immunity
    {0.20f, 0.18f, 0.30f, 0.28f, 0.22f, 1.00f, 0.75f, 0.45f, 0.30f, 0.20f},
    // BA.2 immunity
    {0.18f, 0.15f, 0.28f, 0.25f, 0.20f, 0.70f, 1.00f, 0.55f, 0.35f, 0.25f},
    // BA.4/5 immunity
    {0.12f, 0.10f, 0.20f, 0.18f, 0.15f, 0.50f, 0.60f, 1.00f, 0.60f, 0.40f},
    // BQ.1 immunity
    {0.08f, 0.07f, 0.15f, 0.12f, 0.10f, 0.35f, 0.40f, 0.65f, 1.00f, 0.55f},
    // XBB immunity
    {0.05f, 0.04f, 0.12f, 0.10f, 0.08f, 0.25f, 0.30f, 0.45f, 0.60f, 1.00f}
};

// Country population weights (for proper aggregation)
__constant__ float c_country_population[N_COUNTRIES] = {
    83.0f,  // Germany
    331.0f, // USA
    67.0f,  // UK
    125.0f, // Japan
    214.0f, // Brazil
    67.0f,  // France
    38.0f,  // Canada
    5.8f,   // Denmark
    26.0f,  // Australia
    10.0f,  // Sweden
    128.0f, // Mexico
    60.0f   // South Africa
};

//=============================================================================
// GPU ANTIBODY PHARMACOKINETICS MODEL
// Computes antibody level at time t since vaccination/infection
//=============================================================================

__device__ __forceinline__ float gpu_antibody_pk(
    float days_since_activation,
    int pk_scenario  // 0=fast, 1=medium, 2=slow
) {
    if (days_since_activation < 0.0f) return 0.0f;

    float t_max = c_pk_t_max[pk_scenario];
    float t_half = c_pk_t_half[pk_scenario];

    if (days_since_activation <= t_max) {
        // Rise phase: Linear rise to peak
        return days_since_activation / t_max;
    } else {
        // Decay phase: Exponential decay from peak
        float days_since_peak = days_since_activation - t_max;
        float decay_constant = 0.693147f / t_half;  // ln(2) / t_half
        return expf(-decay_constant * days_since_peak);
    }
}

//=============================================================================
// GPU FOLD-REDUCTION (Cross-neutralization)
// VASIL formula: fold_reduction = exp(Σ escape[epitope] × immunity[epitope])
//=============================================================================

__device__ __forceinline__ float gpu_fold_reduction(
    const float* epitope_escape,     // [N_EPITOPES]
    const float* epitope_immunity,   // [N_EPITOPES]
    int n_epitopes
) {
    float weighted_escape = 0.0f;

    #pragma unroll
    for (int e = 0; e < N_EPITOPES; e++) {
        weighted_escape += epitope_escape[e] * epitope_immunity[e];
    }

    return expf(weighted_escape);
}

// Deterministic reservoir input weights (sin/cos basis for feature mixing)
// 32 reservoir neurons x 12 input features = 384 weights
// Generated using deterministic sin/cos basis functions for reproducibility
__device__ __forceinline__ float get_input_weight(int neuron, int feature) {
    // Deterministic weight generation using sin/cos basis
    // This is equivalent to pre-computed weights but saves constant memory
    float phase = (float)(neuron * 13 + feature * 7) * 0.1f;
    float base = sinf(phase) * 0.4f + cosf(phase * 0.7f) * 0.3f;

    // Emphasize certain feature-neuron combinations
    if (feature == 1 && neuron % 4 == 0) base += 0.2f;  // Conservation emphasis
    if (feature == 2 && neuron % 4 == 1) base += 0.2f;  // Centrality emphasis
    if (feature == 4 && neuron % 4 == 2) base += 0.2f;  // Burial emphasis
    if (feature == 3 && neuron % 4 == 3) base += 0.15f; // Flexibility emphasis

    // GEOMETRY FEATURES - STRONG EMPHASIS (these are CRITICAL for discrimination)
    if (feature == 8 && neuron % 4 == 0) base += 0.35f;   // HSE up (high weight)
    if (feature == 9 && neuron % 4 == 1) base += 0.35f;   // HSE down (high weight)
    if (feature == 10 && neuron % 4 == 2) base += 0.40f;  // Concavity (highest weight)
    if (feature == 11 && neuron % 4 == 3) base += 0.45f;  // Pocket depth (CRITICAL)

    // Additional cross-neuron geometry emphasis
    if (feature >= 8 && neuron < 16) base += 0.15f;  // Lower neurons focus on geometry

    return base;
}

// Branch integration weights (64 neurons x 4 branches)
__device__ __forceinline__ float get_branch_weight(int branch, int neuron) {
    float phase = (float)(branch * 17 + neuron * 11) * 0.15f;
    float base = 0.3f + fabsf(sinf(phase)) * 0.4f;

    // Different branches emphasize different neuron groups
    if (branch == 0) base *= (neuron < 32) ? 1.2f : 0.8f;  // Local: lower neurons
    if (branch == 1) base *= (neuron >= 16 && neuron < 48) ? 1.2f : 0.9f;  // Neighbor: middle
    if (branch == 2) base *= (neuron >= 32) ? 1.2f : 0.8f;  // Global: upper neurons
    if (branch == 3) base *= 0.5f;  // Recurrent: dampened

    return base;
}

// Readout weights for final scoring (64 neurons -> binding site probability)
// Higher neurons have less influence (spectral scaling)
__device__ __forceinline__ float get_readout_weight(int neuron) {
    return 0.5f / (1.0f + 0.05f * (float)neuron);  // Decaying influence
}

// Batch default params in constant memory (fast access)
// TUNED for high-precision pocket detection (publication quality)
__device__ __constant__ MegaFusedParams d_default_params = {
    .contact_cutoff = 8.0f,          // Tightened: 12→8Å (typical binding site)
    .contact_sigma = 4.0f,           // Tightened: 6→4Å (sharper decay)
    .power_iterations = 15,
    .kempe_iterations = 10,
    .thresh_geometric = 0.45f,       // Raised: 0.40→0.45
    .thresh_conservation = 0.55f,    // Raised: 0.50→0.55
    .thresh_centrality = 0.35f,      // Raised: 0.30→0.35
    .thresh_flexibility = 0.50f,     // Raised: 0.45→0.50
    .min_signals = 2,
    .consensus_threshold = 0.50f,    // Raised: 0.35→0.50 (critical for precision)
    .branch_weight_local = 0.40f,
    .branch_weight_neighbor = 0.30f,
    .branch_weight_global = 0.20f,
    .branch_weight_recurrent = 0.10f,
    .recurrent_decay = 0.90f,
    .consensus_weight_geometric = 0.30f,
    .consensus_weight_conservation = 0.25f,
    .consensus_weight_centrality = 0.25f,
    .consensus_weight_flexibility = 0.20f,
    .signal_bonus_0 = 0.70f,
    .signal_bonus_1 = 1.00f,
    .signal_bonus_2 = 1.15f,
    .signal_bonus_3 = 1.30f,
    .confidence_high_score = 0.70f,
    .confidence_medium_score = 0.40f,
    .confidence_high_signals = 3,
    .confidence_medium_signals = 2,
    .kempe_contact_threshold = 0.20f,
    .kempe_swap_threshold = 1.10f,
    .centrality_degree_weight = 0.60f,
    .centrality_eigenvector_weight = 0.40f,
    // QC gate parameters (high-confidence threshold)
    .min_pocket_volume = 160.0f,     // Å³ - minimum for real binding sites
    .max_pocket_volume = 4800.0f,    // Å³ - prevents mega-pockets
    .min_druggability = 0.60f,       // High-confidence threshold for druggable pockets
    .max_pocket_residues = 80,       // Hard limit on pocket size
    .max_pockets = 10                // Top-N limit
};

//=============================================================================
// SHARED MEMORY (Per-block, reused for each structure)
//=============================================================================

// Global clustering arrays (shared across entire block, up to 512 residues)
// For larger structures, we use a secondary pass
#define GLOBAL_CLUSTER_MAX 512

struct __align__(16) BatchSharedMem {
    // Stage 1: Distance/Contact tiles (tiled computation)
    float distance_tile[TILE_SIZE][TILE_SIZE + 1];  // +1 for bank conflict avoidance
    float contact_tile[TILE_SIZE][TILE_SIZE + 1];

    // Stage 2: Per-residue features (single tile at a time)
    float3 ca_coords[TILE_SIZE];
    float conservation[TILE_SIZE];
    float bfactor[TILE_SIZE];
    float burial[TILE_SIZE];

    // Stage 2b: Geometry features (binding site discriminators)
    float hse_up[TILE_SIZE];           // Half-sphere exposure upper
    float hse_down[TILE_SIZE];         // Half-sphere exposure lower
    float local_concavity[TILE_SIZE];  // Local surface curvature
    float pocket_depth[TILE_SIZE];     // Pocket burial depth proxy

    // Stage 2c: TDA features (topologically distinct from geometry)
    // Multi-scale Betti numbers: β₀ (components), β₁ (loops) at 4 scales
    float tda_b0_scale1[TILE_SIZE];    // β₀ at 4Å
    float tda_b0_scale2[TILE_SIZE];    // β₀ at 6Å
    float tda_b0_scale3[TILE_SIZE];    // β₀ at 8Å
    float tda_b0_scale4[TILE_SIZE];    // β₀ at 10Å
    float tda_b1_scale1[TILE_SIZE];    // β₁ at 4Å
    float tda_b1_scale2[TILE_SIZE];    // β₁ at 6Å
    float void_boundary[TILE_SIZE];    // Bridge-point score (pocket boundary)
    float persistence_score[TILE_SIZE]; // Persistence of topological features

    // Stage 3: Network centrality
    float degree[TILE_SIZE];
    float centrality[TILE_SIZE];
    float eigenvector[TILE_SIZE];
    float eigenvector_new[TILE_SIZE];

    // Stage 4: Enhanced Multi-Compartment Dendritic Reservoir
    // 4 compartments per residue with different time constants
    float compartment_proximal[TILE_SIZE];     // Fast response (tau=0.1)
    float compartment_distal1[TILE_SIZE];      // Medium memory (tau=0.5)
    float compartment_distal2[TILE_SIZE];      // Slow integration (tau=0.85)
    float compartment_spine[TILE_SIZE];        // Long-term memory (tau=0.95)
    float soma_potential[TILE_SIZE];           // Integrated soma potential
    float calcium_accumulator[TILE_SIZE];      // LTP-like binding site memory

    // Reservoir neuron activations (32 neurons per residue, compressed to 8 for shared memory)
    // Full 32-dim computed in registers, 8 key activations stored for readout
    float reservoir_activations[TILE_SIZE][8];

    // Stage 5: Consensus
    float geometric_score[TILE_SIZE];
    float consensus_score[TILE_SIZE];
    int signal_mask[TILE_SIZE];
    int confidence[TILE_SIZE];

    // Stage 6: Kempe
    int pocket_assignment[TILE_SIZE];
    int chain_label[TILE_SIZE];
    float assignment_score[TILE_SIZE];

    // Stage 7: Global Union-Find clustering (for up to GLOBAL_CLUSTER_MAX residues)
    int uf_parent[GLOBAL_CLUSTER_MAX];         // Union-Find parent pointers
    int uf_pocket_id[GLOBAL_CLUSTER_MAX];      // Final cluster IDs
    float uf_consensus[GLOBAL_CLUSTER_MAX];    // Consensus scores for ranking
    int uf_cluster_size[32];                    // Size of each cluster (max 32 pockets)

    // PRISM>4D: Fitness + Cycle features
    float fitness_features[TILE_SIZE][4];      // ddG_bind, ddG_stab, expression, transmit
    float cycle_features[TILE_SIZE][5];        // phase, emergence_prob, time_to_peak, freq, vel

    // Stage 8.5: Synaptic Spike features (LIF neuron outputs)
    float spike_features[TILE_SIZE][8];        // 8 spike features (F101-F108)

    // Stage 9-10: Immunity Dynamics + 600-day Integral
    float epitope_immunity[TILE_SIZE][N_EPITOPES];    // Per-epitope immunity levels (0-1)
    float epitope_escape[TILE_SIZE][N_EPITOPES];      // Per-epitope escape scores (0-1)
    float immunity_features[TILE_SIZE][IMMUNITY_FEAT_OUT]; // 16 immunity output features

    // Shared tiles for 600-day immunity integral (reused across time blocks)
    // freq_window[t] = variant frequency at time t (86 weekly samples)
    // p_neut_window[t] = neutralization probability at time t
    float freq_window[WEEKLY_SAMPLES];
    float p_neut_window[WEEKLY_SAMPLES];
    float immunity_integral_S[TILE_SIZE];             // Accumulated immunity integral per residue
    float gamma_vasil[TILE_SIZE];                     // Final VASIL gamma score

    // Stage 11: Epidemiological features (P0 priority for FALL prediction)
    // [0-2]: Competition (freq_rank_norm, gamma_deficit, suppression_pressure)
    // [3-5]: Momentum (log_slope_7d, log_slope_28d, acceleration)
    // [6-9]: Immunity recency (vaccine_norm, wave_norm, derivative, source_ratio)
    // [10]: Country ID normalized
    float epi_features[TILE_SIZE][EPI_FEATURE_COUNT];
};

//=============================================================================
// DEVICE HELPER FUNCTIONS (Optimized with __forceinline__)
//=============================================================================

__device__ __forceinline__ float fast_tanh(float x) {
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

__device__ __forceinline__ float fast_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float gaussian_weight(float dist, float sigma) {
    return expf(-dist * dist / (2.0f * sigma * sigma));
}

__device__ __forceinline__ int popcount_signals(int mask) {
    return __popc(mask);
}

//=============================================================================
// PRISM>4D: Amino Acid Properties (Constant Memory)
//=============================================================================

// Kyte-Doolittle hydrophobicity scale (normalized 0-1)
__constant__ float c_hydrophobicity[20] = {
    0.700f, 0.000f, 0.111f, 0.111f, 0.778f,  // A,R,N,D,C
    0.111f, 0.111f, 0.000f, 0.144f, 1.000f,  // Q,E,G,H,I
    0.922f, 0.067f, 0.711f, 0.811f, 0.000f,  // L,K,M,F,P
    0.411f, 0.333f, 0.400f, 0.356f, 0.967f   // S,T,W,Y,V
};

// Residue volumes (Å³, normalized 0-1)
__constant__ float c_residue_volume[20] = {
    0.152f, 0.476f, 0.243f, 0.220f, 0.190f,  // A,R,N,D,C
    0.302f, 0.280f, 0.100f, 0.333f, 0.341f,  // Q,E,G,H,I
    0.341f, 0.373f, 0.324f, 0.402f, 0.220f,  // L,K,M,F,P
    0.165f, 0.220f, 0.476f, 0.422f, 0.275f   // S,T,W,Y,V
};

//=============================================================================
// STAGE 8.5: SYNAPTIC SPIKE CONSTANT MEMORY
//=============================================================================

// Varied thresholds for population diversity (8 LIF neurons)
__constant__ float c_spike_thresholds[8] = {
    0.15f, 0.25f, 0.35f, 0.45f, 0.55f, 0.65f, 0.75f, 0.90f
};

// Input weights: [8 neurons × 5 cycle features]
// Each row: [velocity_w, frequency_w, emergence_w, time_to_peak_w, phase_w]
__constant__ float c_spike_input_weights[40] = {
    0.6f, 0.1f, 0.1f, 0.1f, 0.1f,   // N0: Velocity-dominant
    0.1f, 0.6f, 0.1f, 0.1f, 0.1f,   // N1: Frequency-dominant
    0.1f, 0.1f, 0.6f, 0.1f, 0.1f,   // N2: Emergence-dominant
    0.4f, 0.1f, 0.1f, 0.3f, 0.1f,   // N3: Phase-sensitive
    0.2f, 0.2f, 0.2f, 0.2f, 0.2f,   // N4: Balanced
    0.35f, 0.35f, 0.1f, 0.1f, 0.1f, // N5: Vel+Freq
    0.1f, 0.1f, 0.4f, 0.3f, 0.1f,   // N6: Emerg+Phase
    0.8f, 0.05f, 0.05f, 0.05f, 0.05f // N7: High-pass velocity
};

// LIF Neuron Parameters
#define LIF_TAU_MEMBRANE 0.9f
#define LIF_V_RESET 0.0f
#define LIF_V_REST 0.0f
#define LIF_REFRACTORY_STEPS 1
#define SPIKE_NEURON_COUNT 8
#define SPIKE_TIMESTEPS 4

//=============================================================================
// BATCH STAGE 1: Distance + Contact (with L1 cache hints)
//=============================================================================

__device__ void batch_stage1_distance_contact(
    const float* __restrict__ atoms,       // All structures packed
    const int* __restrict__ ca_indices,    // All structures packed
    int atom_offset,                        // This structure's offset in atoms
    int residue_offset,                     // This structure's offset in ca_indices
    int n_residues,
    int tile_row,
    int tile_col,
    BatchSharedMem* smem,
    const MegaFusedParams* params
) {
    int local_row = threadIdx.x % TILE_SIZE;
    int local_col = threadIdx.x / TILE_SIZE;
    int global_row = tile_row * TILE_SIZE + local_row;
    int global_col = tile_col * TILE_SIZE + local_col;

    // Load CA coordinates with L1 cache hint (__ldg = read-only cache)
    // NOTE: ca_indices are already global (offset added during packing in Rust)
    if (threadIdx.x < TILE_SIZE && global_row < n_residues) {
        int ca_local_idx = residue_offset + global_row;
        int ca_idx = __ldg(&ca_indices[ca_local_idx]);  // L1 cached, already global

        if (ca_idx >= 0) {
            // ca_idx is already global (includes atom_offset from Rust packing)
            smem->ca_coords[threadIdx.x] = make_float3(
                __ldg(&atoms[ca_idx * 3 + 0]),  // L1 cached
                __ldg(&atoms[ca_idx * 3 + 1]),
                __ldg(&atoms[ca_idx * 3 + 2])
            );
        } else {
            smem->ca_coords[threadIdx.x] = make_float3(0.0f, 0.0f, 0.0f);
        }
    }
    __syncthreads();

    // Compute distance and contact (fused, using registers)
    if (global_row < n_residues && global_col < n_residues && local_col < TILE_SIZE) {
        float3 ci = smem->ca_coords[local_row];
        float3 cj;

        if (tile_row == tile_col) {
            cj = smem->ca_coords[local_col];
        } else {
            int ca_local_idx_j = residue_offset + global_col;
            int ca_idx_j = __ldg(&ca_indices[ca_local_idx_j]);  // Already global
            if (ca_idx_j >= 0) {
                // ca_idx_j is already global (includes atom_offset from Rust packing)
                cj = make_float3(
                    __ldg(&atoms[ca_idx_j * 3 + 0]),
                    __ldg(&atoms[ca_idx_j * 3 + 1]),
                    __ldg(&atoms[ca_idx_j * 3 + 2])
                );
            } else {
                cj = make_float3(0.0f, 0.0f, 0.0f);
            }
        }

        // Distance in registers
        float dx = ci.x - cj.x;
        float dy = ci.y - cj.y;
        float dz = ci.z - cj.z;
        float dist = sqrtf(dx*dx + dy*dy + dz*dz);

        smem->distance_tile[local_row][local_col] = dist;

        // Contact weight (fused)
        float contact = 0.0f;
        if (dist > 0.0f && dist < params->contact_cutoff) {
            contact = gaussian_weight(dist, params->contact_sigma);
        }
        smem->contact_tile[local_row][local_col] = contact;
    }
    __syncthreads();
}

//=============================================================================
// BATCH STAGE 2: Local Features (with L1 cache)
//=============================================================================

__device__ void batch_stage2_local_features(
    const float* __restrict__ conservation_input,
    const float* __restrict__ bfactor_input,
    const float* __restrict__ burial_input,
    int residue_offset,
    int n_residues,
    int tile_idx,
    BatchSharedMem* smem
) {
    int local_idx = threadIdx.x;
    int global_idx = tile_idx * TILE_SIZE + local_idx;

    if (local_idx < TILE_SIZE && global_idx < n_residues) {
        int packed_idx = residue_offset + global_idx;

        // Load with L1 cache hint
        smem->conservation[local_idx] = __ldg(&conservation_input[packed_idx]);
        smem->bfactor[local_idx] = __ldg(&bfactor_input[packed_idx]);
        smem->burial[local_idx] = __ldg(&burial_input[packed_idx]);

        // Compute degree from contact tile (in registers)
        float deg = 0.0f;
        #pragma unroll 8
        for (int j = 0; j < TILE_SIZE; j++) {
            deg += smem->contact_tile[local_idx][j];
        }
        smem->degree[local_idx] = deg;
    }
    __syncthreads();
}

//=============================================================================
// BATCH STAGE 2b: Geometry Features (HSE, Concavity, Pocket Depth)
// These are CRITICAL for binding site discrimination
//=============================================================================

__device__ void batch_stage2b_geometry_features(
    int n_residues,
    int tile_idx,
    BatchSharedMem* smem,
    const MegaFusedParams* params
) {
    int local_idx = threadIdx.x;
    int global_idx = tile_idx * TILE_SIZE + local_idx;
    bool active = (local_idx < TILE_SIZE && global_idx < n_residues);

    if (active) {
        float3 my_coord = smem->ca_coords[local_idx];

        // ═══════════════════════════════════════════════════════════════════
        // HALF-SPHERE EXPOSURE (HSE)
        // Count neighbors in upper vs lower hemisphere relative to local normal
        // Binding sites tend to have asymmetric HSE (exposed from one side)
        // ═══════════════════════════════════════════════════════════════════

        // Estimate local normal from neighbor positions
        float3 centroid = make_float3(0.0f, 0.0f, 0.0f);
        int n_near = 0;

        #pragma unroll 8
        for (int j = 0; j < TILE_SIZE; j++) {
            float d = smem->distance_tile[local_idx][j];
            if (j != local_idx && d > 0.1f && d < 12.0f) {  // 12Å radius
                centroid.x += smem->ca_coords[j].x;
                centroid.y += smem->ca_coords[j].y;
                centroid.z += smem->ca_coords[j].z;
                n_near++;
            }
        }

        // Local normal points from centroid to this residue
        float3 normal;
        if (n_near > 0) {
            centroid.x /= n_near;
            centroid.y /= n_near;
            centroid.z /= n_near;
            normal.x = my_coord.x - centroid.x;
            normal.y = my_coord.y - centroid.y;
            normal.z = my_coord.z - centroid.z;
            float norm_len = sqrtf(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);
            if (norm_len > 0.1f) {
                normal.x /= norm_len;
                normal.y /= norm_len;
                normal.z /= norm_len;
            }
        } else {
            normal = make_float3(0.0f, 1.0f, 0.0f);  // Default up
        }

        // Count atoms in upper vs lower hemisphere
        float hse_up_count = 0.0f;
        float hse_down_count = 0.0f;

        #pragma unroll 8
        for (int j = 0; j < TILE_SIZE; j++) {
            float d = smem->distance_tile[local_idx][j];
            if (j != local_idx && d > 0.1f && d < 13.0f) {  // 13Å hemisphere
                // Vector from this residue to neighbor
                float3 vec;
                vec.x = smem->ca_coords[j].x - my_coord.x;
                vec.y = smem->ca_coords[j].y - my_coord.y;
                vec.z = smem->ca_coords[j].z - my_coord.z;

                // Dot product with normal
                float dot = vec.x * normal.x + vec.y * normal.y + vec.z * normal.z;

                // Distance-weighted count
                float weight = 1.0f / (1.0f + d * 0.1f);
                if (dot > 0.0f) {
                    hse_up_count += weight;
                } else {
                    hse_down_count += weight;
                }
            }
        }

        // Normalize HSE values
        float total_hse = hse_up_count + hse_down_count + 0.1f;
        smem->hse_up[local_idx] = hse_up_count / total_hse;
        smem->hse_down[local_idx] = hse_down_count / total_hse;

        // ═══════════════════════════════════════════════════════════════════
        // LOCAL CONCAVITY
        // Measures how concave the local surface is around this residue
        // Binding sites are typically in concave regions (pockets)
        // ═══════════════════════════════════════════════════════════════════

        float dist_sum = 0.0f;
        float neighbor_dist_sum = 0.0f;
        int n_counted = 0;

        #pragma unroll 8
        for (int j = 0; j < TILE_SIZE; j++) {
            float d_ij = smem->distance_tile[local_idx][j];
            if (j != local_idx && d_ij > 0.1f && d_ij < 10.0f) {
                dist_sum += d_ij;

                // Average distance from j to other neighbors
                float j_dist_sum = 0.0f;
                int j_count = 0;
                #pragma unroll 4
                for (int k = 0; k < TILE_SIZE; k++) {
                    if (k != j && k != local_idx) {
                        float d_jk = smem->distance_tile[j][k];
                        if (d_jk > 0.1f && d_jk < 10.0f) {
                            j_dist_sum += d_jk;
                            j_count++;
                        }
                    }
                }
                if (j_count > 0) {
                    neighbor_dist_sum += j_dist_sum / j_count;
                }
                n_counted++;
            }
        }

        // Concavity: if I'm closer to my neighbors than they are to each other
        // = positive concavity (inside a pocket)
        float avg_my_dist = (n_counted > 0) ? dist_sum / n_counted : 6.0f;
        float avg_neighbor_dist = (n_counted > 0) ? neighbor_dist_sum / n_counted : 6.0f;

        // Normalize to [0,1] range with sigmoid
        float concavity_raw = (avg_neighbor_dist - avg_my_dist) / 3.0f;
        smem->local_concavity[local_idx] = 1.0f / (1.0f + expf(-concavity_raw));

        // ═══════════════════════════════════════════════════════════════════
        // POCKET DEPTH PROXY
        // Combines burial with local density and HSE asymmetry
        // Deep pockets have: high burial + asymmetric HSE + high local density
        // ═══════════════════════════════════════════════════════════════════

        float burial = smem->burial[local_idx];
        float hse_asymmetry = fabsf(hse_up_count - hse_down_count) / (total_hse + 0.1f);
        float local_density = (float)n_near / 12.0f;  // Normalize by expected neighbors

        // Pocket depth: binding sites have high burial + asymmetric HSE + medium density
        float pocket_depth = burial * 0.4f +
                            hse_asymmetry * 0.3f +
                            smem->local_concavity[local_idx] * 0.3f;

        // Apply sigmoid for normalization
        smem->pocket_depth[local_idx] = 1.0f / (1.0f + expf(-(pocket_depth - 0.5f) * 3.0f));
    }
    __syncthreads();
}

//=============================================================================
// BATCH STAGE 2c: TDA Features (Topologically Distinct from Geometry)
// These features capture VOID topology - binding pockets are persistent H₂ voids
//
// Key insight from user analysis:
// - β₂ (voids) directly identify binding pockets
// - void_boundary_score identifies pocket boundaries (articulation points)
// - Multi-scale β₀ captures hierarchical structure
// - These are topologically orthogonal to geometry features
//=============================================================================

__device__ void batch_stage2c_tda_features(
    int n_residues,
    int tile_idx,
    BatchSharedMem* smem,
    const MegaFusedParams* params
) {
    int local_idx = threadIdx.x;
    int global_idx = tile_idx * TILE_SIZE + local_idx;
    bool active = (local_idx < TILE_SIZE && global_idx < n_residues);

    if (active) {
        // ═══════════════════════════════════════════════════════════════════
        // MULTI-SCALE β₀ (Connected Components)
        // Count connected components at different distance thresholds
        // Binding sites show characteristic scale-dependent connectivity
        // ═══════════════════════════════════════════════════════════════════

        const float scales[4] = {4.0f, 6.0f, 8.0f, 10.0f};

        #pragma unroll 4
        for (int s = 0; s < 4; s++) {
            float threshold = scales[s];

            // Count how many distinct clusters this residue connects to
            // Using a simplified union-find based on distance
            int connected_count = 0;
            int cluster_ids[TILE_SIZE];

            // Initialize each residue as its own cluster
            #pragma unroll 8
            for (int j = 0; j < TILE_SIZE; j++) {
                cluster_ids[j] = j;
            }

            // Simple single-pass union (not full union-find, but fast)
            #pragma unroll 8
            for (int j = 0; j < TILE_SIZE; j++) {
                float d = smem->distance_tile[local_idx][j];
                if (d > 0.1f && d < threshold && j < local_idx) {
                    // Union: propagate smaller id
                    int id_j = cluster_ids[j];
                    int id_i = cluster_ids[local_idx];
                    int min_id = min(id_j, id_i);
                    cluster_ids[local_idx] = min_id;
                    cluster_ids[j] = min_id;
                }
            }

            // Count unique clusters in neighborhood
            int unique_clusters = 0;
            bool seen[TILE_SIZE];
            #pragma unroll 8
            for (int j = 0; j < TILE_SIZE; j++) {
                seen[j] = false;
            }

            #pragma unroll 8
            for (int j = 0; j < TILE_SIZE; j++) {
                float d = smem->distance_tile[local_idx][j];
                if (j != local_idx && d > 0.1f && d < threshold + 2.0f) {
                    int cid = cluster_ids[j];
                    if (cid >= 0 && cid < TILE_SIZE && !seen[cid]) {
                        seen[cid] = true;
                        unique_clusters++;
                    }
                }
            }

            // Store normalized β₀ for this scale
            float b0_normalized = fminf((float)unique_clusters / 8.0f, 1.0f);

            if (s == 0) smem->tda_b0_scale1[local_idx] = b0_normalized;
            else if (s == 1) smem->tda_b0_scale2[local_idx] = b0_normalized;
            else if (s == 2) smem->tda_b0_scale3[local_idx] = b0_normalized;
            else smem->tda_b0_scale4[local_idx] = b0_normalized;
        }

        // ═══════════════════════════════════════════════════════════════════
        // β₁ ESTIMATION (Loops/Tunnels)
        // Uses Euler characteristic: χ = V - E + F, and β₁ = 1 - χ + β₀ - β₂
        // For local neighborhood graph: β₁ ≈ E - V + β₀
        // Tunnels and channels have high β₁ (access paths to binding sites)
        // ═══════════════════════════════════════════════════════════════════

        int local_vertices = 0;
        int local_edges = 0;

        #pragma unroll 8
        for (int j = 0; j < TILE_SIZE; j++) {
            float d_ij = smem->distance_tile[local_idx][j];
            if (d_ij > 0.1f && d_ij < 8.0f) {
                local_vertices++;
                // Count edges in local neighborhood
                #pragma unroll 4
                for (int k = j + 1; k < TILE_SIZE; k++) {
                    float d_jk = smem->distance_tile[j][k];
                    if (d_jk > 0.1f && d_jk < 8.0f) {
                        local_edges++;
                    }
                }
            }
        }

        // β₁ ≈ E - V + β₀ (simplified Euler formula for graphs)
        float b1_raw = (float)local_edges - (float)local_vertices + smem->tda_b0_scale3[local_idx] * 8.0f;
        smem->tda_b1_scale1[local_idx] = fmaxf(0.0f, fminf(b1_raw / 20.0f, 1.0f));

        // β₁ at smaller scale (more sensitive to local loops)
        local_edges = 0;
        local_vertices = 0;
        #pragma unroll 8
        for (int j = 0; j < TILE_SIZE; j++) {
            float d_ij = smem->distance_tile[local_idx][j];
            if (d_ij > 0.1f && d_ij < 6.0f) {
                local_vertices++;
                #pragma unroll 4
                for (int k = j + 1; k < TILE_SIZE; k++) {
                    float d_jk = smem->distance_tile[j][k];
                    if (d_jk > 0.1f && d_jk < 6.0f) {
                        local_edges++;
                    }
                }
            }
        }
        float b1_small = (float)local_edges - (float)local_vertices + smem->tda_b0_scale2[local_idx] * 8.0f;
        smem->tda_b1_scale2[local_idx] = fmaxf(0.0f, fminf(b1_small / 15.0f, 1.0f));

        // ═══════════════════════════════════════════════════════════════════
        // VOID BOUNDARY SCORE (Bridge-Point Analysis)
        // Measures if this residue is an articulation point for the local graph
        // High score = removing this residue would disconnect neighbors = pocket boundary
        // This is the HIGHEST VALUE TDA feature for binding site prediction
        // ═══════════════════════════════════════════════════════════════════

        // Count neighbors at threshold
        int n_neighbors = 0;
        int neighbor_ids[TILE_SIZE];

        #pragma unroll 8
        for (int j = 0; j < TILE_SIZE; j++) {
            float d = smem->distance_tile[local_idx][j];
            if (j != local_idx && d > 0.1f && d < 8.0f) {
                neighbor_ids[n_neighbors++] = j;
            }
        }

        // Check connectivity between neighbors WITHOUT this residue
        // If neighbors can't reach each other without going through us, we're a bridge
        int disconnected_pairs = 0;
        int total_pairs = 0;

        for (int i = 0; i < n_neighbors && i < 16; i++) {
            for (int j = i + 1; j < n_neighbors && j < 16; j++) {
                int ni = neighbor_ids[i];
                int nj = neighbor_ids[j];
                total_pairs++;

                // Check if ni and nj are directly connected
                float d_direct = smem->distance_tile[ni][nj];
                if (d_direct > 8.0f || d_direct < 0.1f) {
                    // Not directly connected - check if any other neighbor bridges them
                    bool found_bridge = false;
                    for (int k = 0; k < n_neighbors && k < 16; k++) {
                        if (k != i && k != j) {
                            int nk = neighbor_ids[k];
                            float d_ik = smem->distance_tile[ni][nk];
                            float d_kj = smem->distance_tile[nk][nj];
                            if (d_ik > 0.1f && d_ik < 8.0f && d_kj > 0.1f && d_kj < 8.0f) {
                                found_bridge = true;
                                break;
                            }
                        }
                    }
                    if (!found_bridge) {
                        disconnected_pairs++;
                    }
                }
            }
        }

        // void_boundary_score: high if removing this residue disconnects the neighborhood
        float void_boundary = (total_pairs > 0) ?
            (float)disconnected_pairs / (float)total_pairs : 0.0f;

        // Weight by number of neighbors (articulation points have many neighbors)
        float neighbor_weight = fminf((float)n_neighbors / 10.0f, 1.0f);
        smem->void_boundary[local_idx] = void_boundary * (0.5f + 0.5f * neighbor_weight);

        // ═══════════════════════════════════════════════════════════════════
        // PERSISTENCE SCORE
        // Measures how "stable" the topological features are across scales
        // Binding sites have HIGH persistence (stable voids that persist)
        // ═══════════════════════════════════════════════════════════════════

        // Persistence = consistency of β₀ across scales
        // Low variance = high persistence = stable structure
        float b0_mean = (smem->tda_b0_scale1[local_idx] +
                        smem->tda_b0_scale2[local_idx] +
                        smem->tda_b0_scale3[local_idx] +
                        smem->tda_b0_scale4[local_idx]) / 4.0f;

        float b0_var = 0.0f;
        b0_var += (smem->tda_b0_scale1[local_idx] - b0_mean) * (smem->tda_b0_scale1[local_idx] - b0_mean);
        b0_var += (smem->tda_b0_scale2[local_idx] - b0_mean) * (smem->tda_b0_scale2[local_idx] - b0_mean);
        b0_var += (smem->tda_b0_scale3[local_idx] - b0_mean) * (smem->tda_b0_scale3[local_idx] - b0_mean);
        b0_var += (smem->tda_b0_scale4[local_idx] - b0_mean) * (smem->tda_b0_scale4[local_idx] - b0_mean);
        b0_var /= 4.0f;

        // High persistence = low variance + moderate mean
        // Binding sites: stable moderate connectivity, not highly variable
        float stability = 1.0f / (1.0f + b0_var * 10.0f);
        float activity = b0_mean * (1.0f - b0_mean) * 4.0f;  // Peak at 0.5

        // Combine with void_boundary for pocket persistence
        smem->persistence_score[local_idx] = stability * 0.4f +
                                             activity * 0.3f +
                                             smem->void_boundary[local_idx] * 0.3f;
    }
    __syncthreads();
}

//=============================================================================
// BATCH STAGE 3: Network Centrality (Power Iteration)
//=============================================================================

__device__ void batch_stage3_network_centrality(
    int n_residues,
    int tile_idx,
    BatchSharedMem* smem,
    const MegaFusedParams* params
) {
    int local_idx = threadIdx.x;
    int global_idx = tile_idx * TILE_SIZE + local_idx;
    bool active = (local_idx < TILE_SIZE && global_idx < n_residues);

    // Initialize eigenvector
    if (active) {
        smem->eigenvector[local_idx] = rsqrtf((float)TILE_SIZE);
    }
    __syncthreads();

    // Power iteration with runtime configurable count
    // Use registers for accumulation
    for (int iter = 0; iter < params->power_iterations; iter++) {
        float new_val = 0.0f;
        if (active) {
            #pragma unroll 8
            for (int j = 0; j < TILE_SIZE; j++) {
                new_val += smem->contact_tile[local_idx][j] * smem->eigenvector[j];
            }
            smem->eigenvector_new[local_idx] = new_val;
        }
        __syncthreads();

        // Normalize
        if (active) {
            float norm_sq = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < TILE_SIZE; j++) {
                norm_sq += smem->eigenvector_new[j] * smem->eigenvector_new[j];
            }
            float norm = rsqrtf(norm_sq + 1e-10f);
            smem->eigenvector[local_idx] = smem->eigenvector_new[local_idx] * norm;
        }
        __syncthreads();
    }

    // Compute centrality
    if (active) {
        float max_degree = 0.0f;
        #pragma unroll 8
        for (int j = 0; j < TILE_SIZE; j++) {
            max_degree = fmaxf(max_degree, smem->degree[j]);
        }

        float normalized_degree = smem->degree[local_idx] / (max_degree + 1e-10f);
        float eigenvector_cent = fabsf(smem->eigenvector[local_idx]);

        smem->centrality[local_idx] = params->centrality_degree_weight * normalized_degree +
                                      params->centrality_eigenvector_weight * eigenvector_cent;
    }
    __syncthreads();
}

//=============================================================================
// BATCH STAGE 4: Enhanced Multi-Compartment Dendritic Reservoir
// Implements 64-neuron reservoir with 4 compartments and varied time constants
// Based on dendritic_whcr.cu multi-compartment architecture
//=============================================================================

__device__ void batch_stage4_dendritic_reservoir(
    int n_residues,
    int tile_idx,
    BatchSharedMem* smem,
    const MegaFusedParams* params
) {
    int local_idx = threadIdx.x;
    int global_idx = tile_idx * TILE_SIZE + local_idx;
    bool active = (local_idx < TILE_SIZE && global_idx < n_residues);

    if (active) {
        // ═══════════════════════════════════════════════════════════════════
        // STEP 1: Gather input features into registers (12 features total)
        // First 8: network/sequence features | Last 4: GEOMETRY features (CRITICAL!)
        // ═══════════════════════════════════════════════════════════════════
        float features[N_INPUT_FEATURES];

        // Network/sequence features (8)
        features[0] = smem->degree[local_idx] / 20.0f;           // Degree centrality
        features[1] = smem->conservation[local_idx];              // Conservation/hydrophobicity
        features[2] = smem->centrality[local_idx];                // Network centrality
        features[3] = smem->bfactor[local_idx];                   // Flexibility
        features[4] = smem->burial[local_idx];                    // Burial depth
        features[5] = smem->eigenvector[local_idx];               // Eigenvector centrality
        features[6] = smem->distance_tile[local_idx][0] / 50.0f;  // Reference distance
        features[7] = (float)local_idx / TILE_SIZE;               // Position encoding

        // GEOMETRY features (4) - These are CRITICAL for binding site discrimination
        features[8] = smem->hse_up[local_idx];                    // HSE upper hemisphere
        features[9] = smem->hse_down[local_idx];                  // HSE lower hemisphere
        features[10] = smem->local_concavity[local_idx];          // Local surface concavity
        features[11] = smem->pocket_depth[local_idx];             // Pocket depth proxy

        // ═══════════════════════════════════════════════════════════════════
        // STEP 2: Compute neighbor aggregation (for spatial context)
        // ═══════════════════════════════════════════════════════════════════
        float neighbor_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float neighbor_burial = 0.0f;
        int n_neighbors = 0;

        #pragma unroll 8
        for (int j = 0; j < TILE_SIZE; j++) {
            float contact = smem->contact_tile[local_idx][j];
            if (j != local_idx && contact > 0.1f) {
                neighbor_sum[0] += smem->degree[j] * contact;
                neighbor_sum[1] += smem->conservation[j] * contact;
                neighbor_sum[2] += smem->centrality[j] * contact;
                neighbor_sum[3] += smem->bfactor[j] * contact;
                neighbor_burial += smem->burial[j] * contact;
                n_neighbors++;
            }
        }

        float inv_neighbors = (n_neighbors > 0) ? 1.0f / n_neighbors : 0.0f;
        for (int i = 0; i < 4; i++) neighbor_sum[i] *= inv_neighbors;
        neighbor_burial *= inv_neighbors;

        // ═══════════════════════════════════════════════════════════════════
        // STEP 3: Compute global context statistics
        // ═══════════════════════════════════════════════════════════════════
        float global_cons = 0.0f, global_cent = 0.0f, global_burial = 0.0f;
        #pragma unroll 8
        for (int j = 0; j < TILE_SIZE; j++) {
            global_cons += smem->conservation[j];
            global_cent += smem->centrality[j];
            global_burial += smem->burial[j];
        }
        global_cons /= TILE_SIZE;
        global_cent /= TILE_SIZE;
        global_burial /= TILE_SIZE;

        // ═══════════════════════════════════════════════════════════════════
        // STEP 4: Multi-Compartment Dendritic Processing
        // Each compartment has different time constants for temporal integration
        // ═══════════════════════════════════════════════════════════════════

        // Load previous compartment states (with decay)
        float proximal = smem->compartment_proximal[local_idx] * TAU_PROXIMAL;
        float distal1 = smem->compartment_distal1[local_idx] * TAU_DISTAL_1;
        float distal2 = smem->compartment_distal2[local_idx] * TAU_DISTAL_2;
        float spine = smem->compartment_spine[local_idx] * TAU_SPINE;
        float calcium = smem->calcium_accumulator[local_idx] * 0.99f;  // Very slow decay

        // PROXIMAL COMPARTMENT: Fast response to current binding site signals
        // Strong weight on conservation, centrality, burial (binding site features)
        float proximal_input = features[1] * 1.5f +   // Conservation
                               features[2] * 1.2f +   // Centrality
                               features[4] * 1.3f;    // Burial
        proximal += proximal_input * (1.0f - TAU_PROXIMAL);

        // DISTAL-1 COMPARTMENT: Integrates neighbor context (spatial patterns)
        float distal1_input = neighbor_sum[1] * 1.2f +  // Neighbor conservation
                              neighbor_sum[2] * 1.0f +  // Neighbor centrality
                              neighbor_burial * 1.1f;    // Neighbor burial
        distal1 += distal1_input * (1.0f - TAU_DISTAL_1);

        // DISTAL-2 COMPARTMENT: Slow integration of structural features
        float distal2_input = global_cons * 0.8f +
                              global_cent * 0.6f +
                              features[3] * 0.5f;   // Flexibility
        distal2 += distal2_input * (1.0f - TAU_DISTAL_2);

        // SPINE COMPARTMENT: Long-term binding site memory
        // Accumulates when strong binding signals present
        float binding_signal = proximal + distal1 * 0.5f;
        if (binding_signal > 0.5f) {
            spine += binding_signal * 0.1f;
            calcium += binding_signal * 0.02f;  // LTP-like accumulation
        }
        calcium = fminf(calcium, 1.0f);  // Cap at 1.0

        // Store compartment states
        smem->compartment_proximal[local_idx] = proximal;
        smem->compartment_distal1[local_idx] = distal1;
        smem->compartment_distal2[local_idx] = distal2;
        smem->compartment_spine[local_idx] = spine;
        smem->calcium_accumulator[local_idx] = calcium;

        // ═══════════════════════════════════════════════════════════════════
        // STEP 5: 64-Neuron Reservoir Computation (in registers)
        // Projects features through 64 reservoir neurons with nonlinear mixing
        // ═══════════════════════════════════════════════════════════════════

        float reservoir_neurons[RESERVOIR_DIM];  // 64 neurons in registers
        float readout_sum = 0.0f;

        // Compute all 32 reservoir neurons
        #pragma unroll 8
        for (int n = 0; n < RESERVOIR_DIM; n++) {
            // Input projection: weighted sum of features
            float neuron_input = 0.0f;
            for (int f = 0; f < N_INPUT_FEATURES; f++) {
                neuron_input += features[f] * get_input_weight(n, f);
            }

            // Add compartment contributions
            neuron_input += proximal * get_branch_weight(0, n);
            neuron_input += distal1 * get_branch_weight(1, n);
            neuron_input += distal2 * get_branch_weight(2, n);
            neuron_input += spine * get_branch_weight(3, n) * 2.0f;  // Boost spine influence

            // Calcium modulation (LTP effect)
            neuron_input += calcium * 0.5f * (float)(n % 4 == 0);

            // Nonlinear activation
            reservoir_neurons[n] = fast_tanh(neuron_input);

            // Accumulate for readout
            readout_sum += reservoir_neurons[n] * get_readout_weight(n);
        }

        // Store top 8 activations for feature extraction (every 4th of 32 neurons)
        #pragma unroll 8
        for (int k = 0; k < 8; k++) {
            smem->reservoir_activations[local_idx][k] = reservoir_neurons[k * 4];
        }

        // ═══════════════════════════════════════════════════════════════════
        // STEP 6: Soma Integration and Geometric Score
        // Combines all compartments and reservoir output
        // ═══════════════════════════════════════════════════════════════════

        // Soma integration: weighted sum of compartments
        float soma = c_compartment_weights[0] * proximal +
                     c_compartment_weights[1] * distal1 +
                     c_compartment_weights[2] * distal2 +
                     c_compartment_weights[3] * spine;

        // Add reservoir readout
        soma += readout_sum * 0.3f;

        // Add calcium memory boost
        soma += calcium * 0.5f;

        smem->soma_potential[local_idx] = soma;

        // Final geometric score with sigmoid normalization
        smem->geometric_score[local_idx] = fast_sigmoid(soma * 2.0f);
    }
    __syncthreads();
}

//=============================================================================
// BATCH STAGE 5: Consensus Scoring
//=============================================================================

__device__ void batch_stage5_consensus(
    int n_residues,
    int tile_idx,
    BatchSharedMem* smem,
    const MegaFusedParams* params
) {
    int local_idx = threadIdx.x;
    int global_idx = tile_idx * TILE_SIZE + local_idx;
    bool active = (local_idx < TILE_SIZE && global_idx < n_residues);

    if (active) {
        // Gather evidence (in registers)
        float geometric = smem->geometric_score[local_idx];
        float conservation = smem->conservation[local_idx];
        float centrality = smem->centrality[local_idx];
        float flexibility = smem->bfactor[local_idx];

        // Signal detection
        int signals = 0;
        if (geometric > params->thresh_geometric) signals |= 0x01;
        if (conservation > params->thresh_conservation) signals |= 0x02;
        if (centrality > params->thresh_centrality) signals |= 0x04;
        if (flexibility > params->thresh_flexibility) signals |= 0x08;

        smem->signal_mask[local_idx] = signals;

        int signal_count = popcount_signals(signals);

        // Weighted consensus
        float consensus = params->consensus_weight_geometric * geometric +
                          params->consensus_weight_conservation * conservation +
                          params->consensus_weight_centrality * centrality +
                          params->consensus_weight_flexibility * flexibility;

        // Signal bonus
        float bonus;
        switch (min(signal_count, 3)) {
            case 0: bonus = params->signal_bonus_0; break;
            case 1: bonus = params->signal_bonus_1; break;
            case 2: bonus = params->signal_bonus_2; break;
            default: bonus = params->signal_bonus_3; break;
        }
        consensus = fminf(consensus * bonus, 1.0f);

        smem->consensus_score[local_idx] = consensus;

        // Confidence
        int confidence;
        if (consensus >= params->confidence_high_score && signal_count >= params->confidence_high_signals) {
            confidence = 2;
        } else if (consensus >= params->confidence_medium_score && signal_count >= params->confidence_medium_signals) {
            confidence = 1;
        } else {
            confidence = 0;
        }

        smem->confidence[local_idx] = confidence;
        smem->pocket_assignment[local_idx] = (consensus > params->consensus_threshold) ? 1 : 0;
    }
    __syncthreads();
}

//=============================================================================
// BATCH STAGE 6: Kempe Chain Refinement
//=============================================================================

__device__ void batch_stage6_kempe_refinement(
    int n_residues,
    int tile_idx,
    BatchSharedMem* smem,
    const MegaFusedParams* params
) {
    int local_idx = threadIdx.x;
    int global_idx = tile_idx * TILE_SIZE + local_idx;
    bool active = (local_idx < TILE_SIZE && global_idx < n_residues);

    int my_assignment = 0;

    // Find connected components
    if (active) {
        my_assignment = smem->pocket_assignment[local_idx];

        int root = local_idx;
        #pragma unroll 8
        for (int j = 0; j < local_idx && j < TILE_SIZE; j++) {
            if (smem->contact_tile[local_idx][j] > params->kempe_contact_threshold &&
                smem->pocket_assignment[j] == my_assignment) {
                root = min(root, j);
            }
        }
        smem->chain_label[local_idx] = root;
    }
    __syncthreads();

    // Kempe iterations (configurable)
    for (int iter = 0; iter < params->kempe_iterations; iter++) {
        if (active) {
            bool is_boundary = false;
            int other_pocket = -1;

            #pragma unroll 8
            for (int j = 0; j < TILE_SIZE; j++) {
                if (smem->contact_tile[local_idx][j] > params->kempe_contact_threshold &&
                    smem->pocket_assignment[j] != my_assignment) {
                    is_boundary = true;
                    other_pocket = smem->pocket_assignment[j];
                    break;
                }
            }

            if (is_boundary && other_pocket >= 0) {
                float current_score = 0.0f;
                float swapped_score = 0.0f;

                #pragma unroll 8
                for (int j = 0; j < TILE_SIZE; j++) {
                    float contact = smem->contact_tile[local_idx][j];
                    if (smem->pocket_assignment[j] == my_assignment) {
                        current_score += contact;
                    }
                    if (smem->pocket_assignment[j] == other_pocket) {
                        swapped_score += contact;
                    }
                }

                current_score += smem->consensus_score[local_idx] * 2.0f;

                if (swapped_score > current_score * params->kempe_swap_threshold) {
                    smem->pocket_assignment[local_idx] = other_pocket;
                    my_assignment = other_pocket;
                }
            }
        }
        __syncthreads();
    }

    // Final assignment score
    if (active) {
        float final_score = 0.0f;
        #pragma unroll 8
        for (int j = 0; j < TILE_SIZE; j++) {
            if (smem->pocket_assignment[j] == smem->pocket_assignment[local_idx]) {
                final_score += smem->contact_tile[local_idx][j];
            }
        }
        smem->assignment_score[local_idx] = final_score;
    }
    __syncthreads();
}

//=============================================================================
// PRISM>4D STAGE 7: FITNESS FEATURES (Viral Evolution)
// Computes biochemical fitness: ΔΔG_binding, ΔΔG_stability, expression, transmit
//=============================================================================

__device__ void batch_stage7_fitness_features(
    int n_residues,
    int tile_idx,
    const float* __restrict__ bfactor_input,
    const int* __restrict__ residue_types_input,
    int residue_offset,
    BatchSharedMem* smem,
    const MegaFusedParams* params
) {
    int local_idx = threadIdx.x;
    int global_idx = tile_idx * TILE_SIZE + local_idx;
    bool active = (local_idx < TILE_SIZE && global_idx < n_residues);

    if (active) {
        int res_type = residue_types_input[residue_offset + global_idx];
        // Clamp res_type to valid range [0, 19] to prevent constant memory OOB
        res_type = max(0, min(res_type, 19));
        float bfactor = smem->bfactor[local_idx];
        float burial = smem->burial[local_idx];
        float centrality = smem->centrality[local_idx];

        // Get amino acid properties from constant memory
        float hydro = c_hydrophobicity[res_type];
        float volume = c_residue_volume[res_type];

        // Feature 92: ΔΔG_binding
        float interface_penalty = centrality;
        float ddg_binding = (hydro - 0.5f) * interface_penalty * (1.0f - burial);

        // Feature 93: ΔΔG_stability
        float core_burial = (burial > 0.5f) ? burial : 0.0f;
        float ddg_stability = core_burial * (volume - 0.5f) * (1.0f - bfactor);

        // Feature 94: Expression fitness
        float expression_fitness = 0.3f + 0.5f * (1.0f - burial) + 0.2f * bfactor;

        // Feature 95: Structural transmissibility (RAW - NO escape weighting!)
        float transmit = (1.0f / (1.0f + expf(ddg_binding))) *
                         (1.0f / (1.0f + expf(ddg_stability))) *
                         expression_fitness;

        // Store REAL fitness features (NO hardcoded values!)
        smem->fitness_features[local_idx][0] = ddg_binding;
        smem->fitness_features[local_idx][1] = ddg_stability;
        smem->fitness_features[local_idx][2] = expression_fitness;
        smem->fitness_features[local_idx][3] = transmit;  // RAW transmit, NOT gamma!
    }
    __syncthreads();
}

//=============================================================================
// PRISM>4D STAGE 8: CYCLE FEATURES (Temporal Dynamics)
// Predicts variant emergence based on GISAID frequency/velocity data
//=============================================================================

// FIX #1: Updated Stage 8 with per-structure frequency/velocity (scalar values)
__device__ void batch_stage8_cycle_features_v2(
    int n_residues,
    int tile_idx,
    float struct_frequency,   // Per-structure (scalar)
    float struct_velocity,    // Per-structure (scalar)
    BatchSharedMem* smem
) {
    int local_idx = threadIdx.x;
    int global_idx = tile_idx * TILE_SIZE + local_idx;
    bool active = (local_idx < TILE_SIZE && global_idx < n_residues);

    if (active) {
        float current_freq = struct_frequency;
        float velocity = struct_velocity;

        float escape_score = smem->consensus_score[local_idx];
        float transmit = smem->fitness_features[local_idx][3];

        // Feature 96: Cycle phase (0-5)
        int phase = 0;  // Default: NAIVE
        if (current_freq < 0.01f && velocity < 0.01f && escape_score < 0.5f) {
            phase = 0;  // NAIVE
        } else if (velocity > 0.05f && current_freq < 0.50f) {
            phase = 1;  // EXPLORING
        } else if (current_freq > 0.50f && velocity >= -0.02f) {
            phase = 2;  // ESCAPED
        } else if (current_freq > 0.20f && velocity < -0.02f) {
            phase = 3;  // COSTLY
        } else if (velocity < -0.05f) {
            phase = 4;  // REVERTING
        } else if (current_freq > 0.80f && fabsf(velocity) < 0.02f) {
            phase = 5;  // FIXED
        } else {
            phase = 1;  // Default EXPLORING
        }

        // Feature 97: Emergence probability
        float cycle_mult = 1.0f;
        if (phase == 0) cycle_mult = 0.3f;
        if (phase == 1) cycle_mult = 1.0f;
        if (phase == 2) cycle_mult = 0.1f;
        if (phase == 3) cycle_mult = 0.4f;
        if (phase == 4) cycle_mult = 0.2f;
        if (phase == 5) cycle_mult = 0.05f;
        float emergence_prob = escape_score * transmit * cycle_mult;

        // Feature 98: Time to peak
        float time_to_peak = 0.0f;
        if (velocity > 0.001f) {
            time_to_peak = (0.50f - current_freq) / velocity;
            time_to_peak = fmaxf(0.0f, fminf(time_to_peak, 24.0f));
        }

        // Features 99-100: Frequency and velocity
        float freq_normalized = fminf(current_freq, 1.0f);
        float velocity_normalized = fmaxf(-0.5f, fminf(velocity, 0.5f));

        // Store cycle features
        smem->cycle_features[local_idx][0] = (float)phase;
        smem->cycle_features[local_idx][1] = emergence_prob;
        smem->cycle_features[local_idx][2] = time_to_peak;
        smem->cycle_features[local_idx][3] = freq_normalized;
        smem->cycle_features[local_idx][4] = velocity_normalized;
    }
    __syncthreads();
}

//=============================================================================
// PRISM>4D STAGE 8.5: Synaptic Spike Phase (LIF Neurons)
// Transforms cycle features into sparse spike density signals.
// Creates "temporal sharpening": only consistent, strong signals generate spikes.
//=============================================================================

__device__ void batch_stage8_5_synaptic_spike_phase(
    int n_residues,
    int tile_idx,
    BatchSharedMem* smem
) {
    int local_idx = threadIdx.x;
    int global_idx = tile_idx * TILE_SIZE + local_idx;
    bool active = (local_idx < TILE_SIZE && global_idx < n_residues);

    // Initialize spike features to 0 for ALL threads (including inactive ones)
    // This prevents garbage/NaN values in shared memory
    if (local_idx < TILE_SIZE) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            smem->spike_features[local_idx][i] = 0.0f;
        }
    }
    __syncthreads();

    if (active) {
        // =====================================================================
        // STEP 1: Gather and normalize cycle features
        // =====================================================================
        float cycle_inputs[5];

        // Velocity: -0.5 to +0.5 → 0 to 1
        float raw_velocity = smem->cycle_features[local_idx][4];
        cycle_inputs[0] = (raw_velocity + 0.5f);

        // Frequency: already 0-1
        cycle_inputs[1] = smem->cycle_features[local_idx][3];

        // Emergence probability: already 0-1
        cycle_inputs[2] = smem->cycle_features[local_idx][1];

        // Time to peak: 0-24 months → 0-1
        float raw_ttp = smem->cycle_features[local_idx][2];
        cycle_inputs[3] = fminf(raw_ttp / 24.0f, 1.0f);

        // Phase: 0-5 → 0-1
        float raw_phase = smem->cycle_features[local_idx][0];
        cycle_inputs[4] = raw_phase / 5.0f;

        // =====================================================================
        // STEP 2: LIF Simulation (8 neurons × 4 timesteps)
        // =====================================================================
        float V[SPIKE_NEURON_COUNT];           // Membrane potentials
        int spikes[SPIKE_NEURON_COUNT];        // Spike counts per neuron
        int refrac[SPIKE_NEURON_COUNT];        // Refractory timers

        // Initialize
        #pragma unroll
        for (int n = 0; n < SPIKE_NEURON_COUNT; n++) {
            V[n] = LIF_V_REST;
            spikes[n] = 0;
            refrac[n] = 0;
        }

        // Accumulators for derived features
        int total_spikes = 0;
        int burst_steps = 0;        // Steps with 3+ concurrent spikes
        int threshold_crossings = 0;
        int refractory_total = 0;

        // Simulate
        #pragma unroll
        for (int t = 0; t < SPIKE_TIMESTEPS; t++) {
            int step_spikes = 0;

            #pragma unroll
            for (int n = 0; n < SPIKE_NEURON_COUNT; n++) {
                // Check refractory
                if (refrac[n] > 0) {
                    refrac[n]--;
                    refractory_total++;
                    continue;
                }

                // Compute input current
                float I = 0.0f;
                #pragma unroll
                for (int f = 0; f < 5; f++) {
                    I += cycle_inputs[f] * c_spike_input_weights[n * 5 + f];
                }

                // Leaky integration
                V[n] = LIF_TAU_MEMBRANE * V[n] + I;

                // Threshold check
                if (V[n] >= c_spike_thresholds[n]) {
                    spikes[n]++;
                    total_spikes++;
                    step_spikes++;
                    threshold_crossings++;
                    V[n] = LIF_V_RESET;
                    refrac[n] = LIF_REFRACTORY_STEPS;
                }
            }

            // Burst detection (3+ concurrent spikes)
            if (step_spikes >= 3) {
                burst_steps++;
            }
        }

        // =====================================================================
        // STEP 3: Compute 8 spike features
        // =====================================================================
        const float inv_max = 1.0f / 4.0f;  // 4 timesteps max per neuron

        // F101: Velocity-sensitive spike density (neurons 0, 5, 7)
        smem->spike_features[local_idx][0] =
            (float)(spikes[0] + spikes[5] + spikes[7]) / 12.0f;

        // F102: Frequency-sensitive spike density (neurons 1, 4)
        smem->spike_features[local_idx][1] =
            (float)(spikes[1] + spikes[4]) / 8.0f;

        // F103: Emergence-sensitive spike density (neurons 2, 6)
        smem->spike_features[local_idx][2] =
            (float)(spikes[2] + spikes[6]) / 8.0f;

        // F104: Burst ratio
        smem->spike_features[local_idx][3] =
            (float)burst_steps / 4.0f;

        // F105: Phase-spike coherence (neuron 3 modulated by phase)
        smem->spike_features[local_idx][4] =
            (float)spikes[3] * inv_max * cycle_inputs[4];

        // F106: Spike momentum (late vs early activity)
        float early = (spikes[0] + spikes[1] > 0) ? 1.0f : 0.0f;
        float late = (spikes[6] + spikes[7] > 0) ? 1.0f : 0.0f;
        smem->spike_features[local_idx][5] =
            (late - early + 1.0f) * 0.5f;

        // F107: Threshold crossings (normalized)
        smem->spike_features[local_idx][6] =
            (float)threshold_crossings / 32.0f;  // 8 neurons × 4 timesteps

        // F108: Refractory fraction
        smem->spike_features[local_idx][7] =
            (float)refractory_total / 32.0f;
    }
    __syncthreads();
}

//=============================================================================
// PRISM>4D STAGE 9: IMMUNITY DYNAMICS (GPU Pharmacokinetics + Cross-Neutralization)
// Computes per-epitope immunity and cross-neutralization on GPU
//
// Implements VASIL model:
// - Antibody rise (0 → t_max) and decay (t_max → ∞) phases
// - Multi-epitope immunity (10 epitope classes)
// - Cross-reactivity matrix for variant families
// - Fold-reduction computation
//=============================================================================

__device__ void batch_stage9_immunity_dynamics(
    int n_residues,
    int tile_idx,
    const float* __restrict__ epitope_escape_packed,   // [n_residues * N_EPITOPES] or nullptr
    const float* __restrict__ immunity_events_packed,  // [n_events * event_stride] or nullptr
    int n_events,                                      // Number of immunity events
    int current_day,                                   // Current day for PK computation
    int variant_family_idx,                            // Variant family index (0-9)
    int residue_offset,
    BatchSharedMem* smem,
    const MegaFusedParams* params
) {
    int local_idx = threadIdx.x;
    int global_idx = tile_idx * TILE_SIZE + local_idx;
    bool active = (local_idx < TILE_SIZE && global_idx < n_residues);

    // Initialize immunity features to 0
    if (local_idx < TILE_SIZE) {
        #pragma unroll
        for (int e = 0; e < N_EPITOPES; e++) {
            smem->epitope_immunity[local_idx][e] = 0.0f;
            smem->epitope_escape[local_idx][e] = 0.0f;
        }
        #pragma unroll
        for (int f = 0; f < IMMUNITY_FEAT_OUT; f++) {
            smem->immunity_features[local_idx][f] = 0.0f;
        }
        smem->immunity_integral_S[local_idx] = 0.0f;
        smem->gamma_vasil[local_idx] = 0.0f;
    }
    __syncthreads();

    if (active) {
        // =====================================================================
        // STEP 1: Load per-epitope escape scores for this residue
        // =====================================================================
        if (epitope_escape_packed != nullptr) {
            int escape_base = (residue_offset + global_idx) * N_EPITOPES;
            #pragma unroll
            for (int e = 0; e < N_EPITOPES; e++) {
                smem->epitope_escape[local_idx][e] = __ldg(&epitope_escape_packed[escape_base + e]);
            }
        } else {
            // Default: derive from consensus score (fallback)
            float base_escape = smem->consensus_score[local_idx] * 0.8f;
            #pragma unroll
            for (int e = 0; e < N_EPITOPES; e++) {
                smem->epitope_escape[local_idx][e] = base_escape * (0.8f + 0.4f * ((e * 7) % 10) / 10.0f);
            }
        }

        // =====================================================================
        // STEP 2: Compute per-epitope immunity from immunity events
        // Implements PK model: rise to t_max, then exponential decay
        // =====================================================================
        if (immunity_events_packed != nullptr && n_events > 0) {
            // Event layout: [event_day, magnitude, pk_scenario, source_family, epitope_profile[10]]
            const int EVENT_STRIDE = 14;  // 4 + 10 epitope values

            for (int ev = 0; ev < n_events && ev < 64; ev++) {  // Cap at 64 events
                int ev_base = ev * EVENT_STRIDE;

                float event_day = __ldg(&immunity_events_packed[ev_base + 0]);
                float magnitude = __ldg(&immunity_events_packed[ev_base + 1]);
                int pk_scenario = (int)__ldg(&immunity_events_packed[ev_base + 2]);
                int source_family = (int)__ldg(&immunity_events_packed[ev_base + 3]);

                float days_since = (float)current_day - event_day;
                if (days_since < 0.0f) continue;  // Event hasn't happened yet

                // Compute antibody level using PK model
                float ab_level = gpu_antibody_pk(days_since, pk_scenario);

                // Get cross-reactivity from source → target variant
                float cross_react = c_cross_reactivity[source_family][variant_family_idx];

                // Add contribution to each epitope
                #pragma unroll
                for (int e = 0; e < N_EPITOPES; e++) {
                    float epitope_contrib = __ldg(&immunity_events_packed[ev_base + 4 + e]);
                    smem->epitope_immunity[local_idx][e] += magnitude * ab_level * epitope_contrib * cross_react;
                }
            }

            // Cap immunity at 1.0 per epitope
            #pragma unroll
            for (int e = 0; e < N_EPITOPES; e++) {
                smem->epitope_immunity[local_idx][e] = fminf(smem->epitope_immunity[local_idx][e], 1.0f);
            }
        } else {
            // Default immunity profile (Germany mid-2022 baseline)
            float base_immunity = 0.55f;  // ~55% population immunity
            #pragma unroll
            for (int e = 0; e < N_EPITOPES; e++) {
                smem->epitope_immunity[local_idx][e] = base_immunity * (0.7f + 0.6f * ((e * 11) % 10) / 10.0f);
            }
        }

        // =====================================================================
        // STEP 3: Compute fold-reduction (cross-neutralization)
        // VASIL formula: fold_red = exp(Σ escape[e] × immunity[e])
        // =====================================================================
        float fold_reduction = gpu_fold_reduction(
            smem->epitope_escape[local_idx],
            smem->epitope_immunity[local_idx],
            N_EPITOPES
        );

        // =====================================================================
        // STEP 4: Compute effective escape (immunity-weighted)
        // =====================================================================
        float effective_escape = 0.0f;
        #pragma unroll
        for (int e = 0; e < N_EPITOPES; e++) {
            effective_escape += smem->epitope_escape[local_idx][e] * (1.0f - smem->epitope_immunity[local_idx][e]);
        }
        effective_escape /= N_EPITOPES;

        // =====================================================================
        // STEP 5: Compute VASIL gamma
        // gamma = alpha × (-log(fold_red)) + beta × transmit
        // =====================================================================
        float structural_transmit = smem->fitness_features[local_idx][3];  // From Stage 7
        float escape_component = -logf(fmaxf(fold_reduction, 0.01f));
        float gamma = c_alpha_escape * escape_component + c_beta_transmit * structural_transmit;

        // =====================================================================
        // STEP 6: Store immunity output features (F109-F124)
        // =====================================================================
        // Features 109-118: Per-epitope immunity levels
        #pragma unroll
        for (int e = 0; e < N_EPITOPES; e++) {
            smem->immunity_features[local_idx][e] = smem->epitope_immunity[local_idx][e];
        }

        // Features 119-124: Derived immunity metrics
        smem->immunity_features[local_idx][10] = fold_reduction;        // F119: Fold reduction
        smem->immunity_features[local_idx][11] = effective_escape;      // F120: Effective escape
        smem->immunity_features[local_idx][12] = gamma;                 // F121: VASIL gamma
        smem->immunity_features[local_idx][13] = escape_component;      // F122: Escape component
        smem->immunity_features[local_idx][14] = structural_transmit;   // F123: Transmit component

        // F124: Overall immunity (mean across epitopes)
        float overall_immunity = 0.0f;
        #pragma unroll
        for (int e = 0; e < N_EPITOPES; e++) {
            overall_immunity += smem->epitope_immunity[local_idx][e];
        }
        smem->immunity_features[local_idx][15] = overall_immunity / N_EPITOPES;

        // Store VASIL gamma for Stage 10
        smem->gamma_vasil[local_idx] = gamma;
    }
    __syncthreads();
}

//=============================================================================
// PRISM>4D STAGE 10: 600-DAY IMMUNITY INTEGRAL (VASIL Growth Rate)
// Computes immunity integral over 600-day window for accurate growth prediction
//
// VASIL formula:
//   S(c,v,d) = ∫[d-600 to d] freq(t,v) × p_neut(t,v) × Ab_decay(d-t) dt
//   gamma(c,v,d) = S(c,target_v,d) / mean_v(S(c,v,d)) - 1
//
// Grid: Each thread block handles one (country, variant, date) combination
// Shared tiles for freq[t] and p_neut[t] with double-buffering
//=============================================================================

__device__ void batch_stage10_immunity_integral(
    int n_residues,
    int tile_idx,
    const float* __restrict__ freq_time_series,     // [WEEKLY_SAMPLES] or nullptr
    const float* __restrict__ p_neut_time_series,   // [WEEKLY_SAMPLES] or nullptr
    int n_time_samples,                              // Actual number of samples (up to 86)
    float current_immunity_level,                    // Current overall immunity
    int residue_offset,
    BatchSharedMem* smem
) {
    int local_idx = threadIdx.x;
    int global_idx = tile_idx * TILE_SIZE + local_idx;
    bool active = (local_idx < TILE_SIZE && global_idx < n_residues);

    // =========================================================================
    // STEP 1: Cooperatively load frequency/neutralization time series into shared
    // Each thread loads some samples (86 samples / 32 threads ≈ 3 per thread)
    // =========================================================================
    int samples_per_thread = (WEEKLY_SAMPLES + TILE_SIZE - 1) / TILE_SIZE;

    for (int s = 0; s < samples_per_thread; s++) {
        int sample_idx = local_idx + s * TILE_SIZE;
        if (sample_idx < WEEKLY_SAMPLES) {
            if (freq_time_series != nullptr && sample_idx < n_time_samples) {
                smem->freq_window[sample_idx] = __ldg(&freq_time_series[sample_idx]);
            } else {
                // Default: exponential growth from low baseline
                float t_norm = (float)sample_idx / WEEKLY_SAMPLES;
                smem->freq_window[sample_idx] = 0.001f * expf(4.0f * t_norm);
            }

            if (p_neut_time_series != nullptr && sample_idx < n_time_samples) {
                smem->p_neut_window[sample_idx] = __ldg(&p_neut_time_series[sample_idx]);
            } else {
                // Default: decreasing neutralization as immunity wanes
                float t_norm = (float)sample_idx / WEEKLY_SAMPLES;
                smem->p_neut_window[sample_idx] = 0.8f * expf(-0.5f * t_norm);
            }
        }
    }
    __syncthreads();

    // Store n_time_samples in shared memory for all threads to access
    __shared__ int actual_samples;
    if (threadIdx.x == 0) {
        actual_samples = (n_time_samples > 0 && n_time_samples <= WEEKLY_SAMPLES)
                        ? n_time_samples : WEEKLY_SAMPLES;
    }
    __syncthreads();

    if (active) {
        // =====================================================================
        // STEP 2: Compute immunity integral S over 600-day window
        // S = Σ_t freq(t) × p_neut(t) × Ab_decay(current_day - t)
        // Uses fused multiply-add for numerical accuracy
        // Only integrates over actual_samples (not full WEEKLY_SAMPLES buffer)
        // =====================================================================
        float S_integral = 0.0f;
        float weight_sum = 0.0f;

        // Default PK scenario (medium = 1)
        int pk_scenario = 1;

        // Integrate only over actual data samples (not default-filled padding)
        int n_samples = actual_samples;

        #pragma unroll 8
        for (int t = 0; t < n_samples; t++) {
            // Days since this time point (7 days per sample)
            float days_ago = (float)(n_samples - t) * 7.0f;

            // Antibody decay factor from PK model
            float ab_decay = gpu_antibody_pk(days_ago, pk_scenario);

            // Contribution to integral
            float freq = smem->freq_window[t];
            float p_neut = smem->p_neut_window[t];

            // Fused multiply-add: S += freq * p_neut * ab_decay
            S_integral = fmaf(freq * p_neut, ab_decay, S_integral);
            weight_sum += ab_decay;
        }

        // Normalize by total weight
        if (weight_sum > 0.001f) {
            S_integral /= weight_sum;
        }

        smem->immunity_integral_S[local_idx] = S_integral;
    }
    __syncthreads();

    // =========================================================================
    // STEP 3: Warp reduction to compute mean S across residues
    // gamma = S_target / mean(S) - 1
    // NOTE: Warp shuffle must be executed by ALL threads in first warp (0-31)
    //       to avoid deadlock on partial tiles. Inactive threads contribute 0.
    // =========================================================================

    // All threads in first warp participate in shuffle, inactive ones contribute 0
    float my_S = 0.0f;
    if (active) {
        my_S = smem->immunity_integral_S[local_idx];
    }

    // Only threads 0-31 (first warp) participate in shuffle reduction
    float warp_sum = my_S;
    if (local_idx < WARP_SIZE) {
        // Warp-level reduction using shuffle - all 32 threads must execute together
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
    }
    __syncthreads();  // Ensure all warps are synchronized

    // Store result in shared memory for non-warp-0 threads to read
    __shared__ float mean_S_shared;
    if (local_idx == 0) {
        // Divide by actual active residue count, not full TILE_SIZE
        int active_count = min(TILE_SIZE, n_residues - tile_idx * TILE_SIZE);
        if (active_count < 1) active_count = 1;  // Prevent divide by zero
        mean_S_shared = warp_sum / (float)active_count;
    }
    __syncthreads();

    // =====================================================================
    // STEP 4: Compute final VASIL gamma with integral
    // gamma_integral = S_target / mean(S) - 1
    // Combined gamma = 0.6 × gamma_static + 0.4 × gamma_integral
    // =====================================================================
    if (active) {
        float mean_S = mean_S_shared;
        float gamma_integral = 0.0f;
        if (mean_S > 0.001f) {
            gamma_integral = (my_S / mean_S) - 1.0f;
        }

        // Combine with static gamma from Stage 9
        float gamma_static = smem->gamma_vasil[local_idx];
        float gamma_combined = 0.6f * gamma_static + 0.4f * gamma_integral;

        // Update gamma in shared memory
        smem->gamma_vasil[local_idx] = gamma_combined;

        // Update immunity features with integral-based values
        smem->immunity_features[local_idx][12] = gamma_combined;  // F121: Updated gamma
        smem->immunity_features[local_idx][15] = fminf(fmaxf(current_immunity_level + my_S * 0.1f, 0.0f), 1.0f);
    }
    __syncthreads();
}

//=============================================================================
// BATCH STAGE 10: Immunity Integral with 75-PK Support (FIX #2 CORRECTED)
// Computes immunity envelope from 75 PK parameter combinations
//=============================================================================
__device__ void batch_stage10_immunity_integral_75pk(
    int n_residues,
    int tile_idx,
    int structure_idx,
    const float* __restrict__ p_neut_75pk,     // [n_countries × 75 × 86] or nullptr
    const float* __restrict__ immunity_75,     // [n_structures × 75] or nullptr
    const float* __restrict__ pk_params,       // [75 × 4] or nullptr
    int n_time_samples,
    int residue_offset,
    BatchSharedMem* smem
) {
    int local_idx = threadIdx.x;
    int global_idx = tile_idx * TILE_SIZE + local_idx;
    bool active = (local_idx < TILE_SIZE && global_idx < n_residues);

    // Simplified: Use first PK value (pk_idx=0) for now
    // Full implementation would compute min/max/mean across all 75
    if (active && immunity_75 != nullptr) {
        // Read immunity values for this structure (75 values)
        int struct_base = structure_idx * 75;

        float imm_min = 1.0f;
        float imm_max = 0.0f;
        float imm_sum = 0.0f;

        #pragma unroll 8
        for (int pk = 0; pk < 75; pk++) {
            float imm = immunity_75[struct_base + pk];
            imm_min = fminf(imm_min, imm);
            imm_max = fmaxf(imm_max, imm);
            imm_sum += imm;
        }

        float imm_mean = imm_sum / 75.0f;
        float imm_range = imm_max - imm_min;

        // Store in immunity features
        smem->immunity_features[local_idx][12] = imm_min;    // F121: immunity_min
        smem->immunity_features[local_idx][13] = imm_max;    // F122: immunity_max
        smem->immunity_features[local_idx][14] = imm_mean;   // F123: immunity_mean
        smem->immunity_features[local_idx][15] = imm_range;  // F124: immunity_range
        smem->gamma_vasil[local_idx] = imm_mean * 0.5f;      // Placeholder gamma
    }
    __syncthreads();
}

//=============================================================================
// BATCH STAGE 11: Epidemiological Features (P0 Priority)
// Competition, momentum, immunity recency, and country context
//=============================================================================
__device__ void batch_stage11_epi_features(
    int n_residues,
    int tile_idx,
    // Competition context (all variants in batch)
    const float* __restrict__ all_frequencies,    // [n_variants] frequencies
    const float* __restrict__ all_gammas,         // [n_variants] fitness scores
    int my_variant_idx,                           // Which variant this structure represents
    int n_variants,                               // Total variants in competition
    // Momentum (historical frequency)
    const float* __restrict__ freq_history,       // [HISTORY_DAYS * n_variants] column-major
    // Immunity recency (uniform per country-date)
    float days_since_vaccine_norm,
    float days_since_wave_norm,
    float immunity_derivative,
    float immunity_source_ratio,
    // Country context
    float country_id_norm,
    BatchSharedMem* smem
) {
    int local_idx = threadIdx.x;
    int global_idx = tile_idx * TILE_SIZE + local_idx;
    bool active = (local_idx < TILE_SIZE && global_idx < n_residues);

    // Default zero epi features
    float epi[EPI_FEATURE_COUNT] = {0.0f};

    if (active && n_variants > 0 && all_frequencies != nullptr && all_gammas != nullptr) {
        float my_freq = all_frequencies[my_variant_idx];
        float my_gamma = all_gammas[my_variant_idx];

        // ===================== COMPETITION FEATURES [0-2] =====================
        int rank = 0;
        float max_competitor_gamma = -1e9f;
        float suppression_pressure = 0.0f;

        for (int v = 0; v < n_variants; v++) {
            if (v == my_variant_idx) continue;
            float v_freq = all_frequencies[v];
            float v_gamma = all_gammas[v];

            // Rank: count variants with higher frequency
            if (v_freq > my_freq) rank++;
            // Max competitor fitness
            if (v_gamma > max_competitor_gamma) max_competitor_gamma = v_gamma;
            // Suppression from fitter variants
            if (v_gamma > my_gamma) suppression_pressure += v_freq;
        }

        epi[0] = (float)rank / fmaxf((float)(n_variants - 1), 1.0f);  // freq_rank_norm
        epi[1] = fmaxf(-2.0f, fminf(2.0f, my_gamma - max_competitor_gamma));  // gamma_deficit
        epi[2] = fminf(suppression_pressure, 1.0f);  // suppression_pressure

        // ===================== MOMENTUM FEATURES [3-5] =====================
        if (freq_history != nullptr) {
            float f_t0  = freq_history[0 * n_variants + my_variant_idx];
            float f_t7  = freq_history[7 * n_variants + my_variant_idx];
            float f_t14 = freq_history[14 * n_variants + my_variant_idx];
            float f_t28 = freq_history[28 * n_variants + my_variant_idx];

            float slope_7d = (f_t0 - f_t7) / 7.0f;
            float slope_7d_prev = (f_t7 - f_t14) / 7.0f;
            float slope_28d = (f_t0 - f_t28) / 28.0f;

            // Log-transform for scale invariance
            float log_slope_7d = copysignf(log1pf(fabsf(slope_7d) * 100.0f), slope_7d);
            float log_slope_28d = copysignf(log1pf(fabsf(slope_28d) * 100.0f), slope_28d);
            float acceleration = fmaxf(-1.0f, fminf(1.0f, (slope_7d - slope_7d_prev) / 7.0f * 100.0f));

            epi[3] = fmaxf(-5.0f, fminf(5.0f, log_slope_7d));
            epi[4] = fmaxf(-5.0f, fminf(5.0f, log_slope_28d));
            epi[5] = acceleration;
        }

        // ===================== IMMUNITY RECENCY FEATURES [6-9] =====================
        epi[6] = days_since_vaccine_norm;
        epi[7] = days_since_wave_norm;
        epi[8] = fmaxf(-0.1f, fminf(0.1f, immunity_derivative));
        epi[9] = immunity_source_ratio;

        // ===================== COUNTRY CONTEXT [10] =====================
        epi[10] = country_id_norm;
    }

    // Store to shared memory
    if (active) {
        #pragma unroll
        for (int i = 0; i < EPI_FEATURE_COUNT; i++) {
            smem->epi_features[local_idx][i] = epi[i];
        }
    }
    __syncthreads();
}

//=============================================================================
// PRISM>4D: Write Combined Features (136-dim per residue)
// Simplified version for batch kernel - writes fitness+cycle+epi features
// (TDA and base features already computed in earlier stages)
//=============================================================================

// Inline PTX helpers to force writes (prevent dead code elimination)
__device__ __forceinline__ void ptx_store_global_f32(float* addr, float val) {
    asm volatile("st.global.f32 [%0], %1;" :: "l"(addr), "f"(val) : "memory");
}

__device__ void batch_write_combined_features(
    int n_residues,
    int tile_idx,
    float* __restrict__ combined_features_out,
    int residue_offset,
    BatchSharedMem* smem
) {
    int local_idx = threadIdx.x;
    int global_idx = tile_idx * TILE_SIZE + local_idx;
    bool active = (local_idx < TILE_SIZE && global_idx < n_residues);

    if (active) {
        int out_res_idx = residue_offset + global_idx;
        float* out_base = combined_features_out + out_res_idx * TOTAL_COMBINED_FEATURES;

        // Use inline PTX to force writes (prevent compiler optimization)
        // Features 92-95: Fitness
        ptx_store_global_f32(out_base + 92, smem->fitness_features[local_idx][0]);
        ptx_store_global_f32(out_base + 93, smem->fitness_features[local_idx][1]);
        ptx_store_global_f32(out_base + 94, smem->fitness_features[local_idx][2]);
        ptx_store_global_f32(out_base + 95, smem->fitness_features[local_idx][3]);

        // Features 96-100: Cycle
        ptx_store_global_f32(out_base + 96, smem->cycle_features[local_idx][0]);
        ptx_store_global_f32(out_base + 97, smem->cycle_features[local_idx][1]);
        ptx_store_global_f32(out_base + 98, smem->cycle_features[local_idx][2]);
        ptx_store_global_f32(out_base + 99, smem->cycle_features[local_idx][3]);
        ptx_store_global_f32(out_base + 100, smem->cycle_features[local_idx][4]);

        // Features 101-108: Stage 8.5 Spike (LIF neuron outputs)
        ptx_store_global_f32(out_base + 101, smem->spike_features[local_idx][0]);  // Velocity spike
        ptx_store_global_f32(out_base + 102, smem->spike_features[local_idx][1]);  // Freq spike
        ptx_store_global_f32(out_base + 103, smem->spike_features[local_idx][2]);  // Emergence spike
        ptx_store_global_f32(out_base + 104, smem->spike_features[local_idx][3]);  // Burst ratio
        ptx_store_global_f32(out_base + 105, smem->spike_features[local_idx][4]);  // Phase coherence
        ptx_store_global_f32(out_base + 106, smem->spike_features[local_idx][5]);  // Spike momentum
        ptx_store_global_f32(out_base + 107, smem->spike_features[local_idx][6]);  // Threshold crossings
        ptx_store_global_f32(out_base + 108, smem->spike_features[local_idx][7]);  // Refractory fraction

        // Features 109-124: Stage 9-10 Immunity Dynamics (16 features)
        // 109-118: Per-epitope immunity levels (10 epitopes)
        #pragma unroll
        for (int e = 0; e < 10; e++) {
            ptx_store_global_f32(out_base + 109 + e, smem->immunity_features[local_idx][e]);
        }

        // 119-124: Derived immunity metrics
        ptx_store_global_f32(out_base + 119, smem->immunity_features[local_idx][10]);  // Fold reduction
        ptx_store_global_f32(out_base + 120, smem->immunity_features[local_idx][11]);  // Effective escape
        ptx_store_global_f32(out_base + 121, smem->gamma_vasil[local_idx]);            // VASIL gamma (final)
        ptx_store_global_f32(out_base + 122, smem->immunity_features[local_idx][13]);  // Escape component
        ptx_store_global_f32(out_base + 123, smem->immunity_features[local_idx][14]);  // Transmit component
        ptx_store_global_f32(out_base + 124, smem->immunity_features[local_idx][15]);  // Overall immunity

        // Features 125-135: Stage 11 Epidemiological Features (P0 priority)
        // Competition: freq_rank_norm, gamma_deficit, suppression_pressure
        ptx_store_global_f32(out_base + 125, smem->epi_features[local_idx][0]);   // freq_rank_norm
        ptx_store_global_f32(out_base + 126, smem->epi_features[local_idx][1]);   // gamma_deficit
        ptx_store_global_f32(out_base + 127, smem->epi_features[local_idx][2]);   // suppression_pressure
        // Momentum: log_slope_7d, log_slope_28d, acceleration
        ptx_store_global_f32(out_base + 128, smem->epi_features[local_idx][3]);   // log_slope_7d
        ptx_store_global_f32(out_base + 129, smem->epi_features[local_idx][4]);   // log_slope_28d
        ptx_store_global_f32(out_base + 130, smem->epi_features[local_idx][5]);   // acceleration
        // Immunity recency: days_since_vaccine, days_since_wave, derivative, source_ratio
        ptx_store_global_f32(out_base + 131, smem->epi_features[local_idx][6]);   // days_since_vaccine_norm
        ptx_store_global_f32(out_base + 132, smem->epi_features[local_idx][7]);   // days_since_wave_norm
        ptx_store_global_f32(out_base + 133, smem->epi_features[local_idx][8]);   // immunity_derivative
        ptx_store_global_f32(out_base + 134, smem->epi_features[local_idx][9]);   // immunity_source_ratio
        // Country ID normalized
        ptx_store_global_f32(out_base + 135, smem->epi_features[local_idx][10]);  // country_id_norm
    }
    __threadfence();  // Ensure all writes visible before kernel exit
}

//=============================================================================
// BATCH STAGE 7: Global Union-Find Clustering (Publication Quality)
// Runs AFTER all tiles are processed to cluster ALL residues globally
// Uses shared memory for Union-Find with path compression
//=============================================================================

// Union-Find helpers (in shared memory)
__device__ __forceinline__ int uf_find(int* parent, int x) {
    // Path compression: flatten tree as we find
    int root = x;
    while (parent[root] != root) {
        root = parent[root];
    }
    // Compress path
    while (parent[x] != root) {
        int next = parent[x];
        parent[x] = root;
        x = next;
    }
    return root;
}

__device__ __forceinline__ void uf_union(int* parent, int x, int y) {
    int rx = uf_find(parent, x);
    int ry = uf_find(parent, y);
    if (rx != ry) {
        // Union by smaller root (simple but effective)
        if (rx < ry) {
            parent[ry] = rx;
        } else {
            parent[rx] = ry;
        }
    }
}

__device__ void batch_stage7_global_clustering(
    // Global memory arrays (already written by tile loop)
    float* __restrict__ consensus_out,
    int* __restrict__ pocket_assignment_out,
    // Structure info
    const float* __restrict__ atoms_packed,
    const int* __restrict__ ca_indices_packed,
    int atom_offset,
    int residue_offset,
    int n_residues,
    BatchSharedMem* smem,
    const MegaFusedParams* params
) {
    int tid = threadIdx.x;

    // Step 1: Load consensus scores and initial pocket flags into shared memory
    // Process in chunks if n_residues > GLOBAL_CLUSTER_MAX
    int n_to_process = min(n_residues, GLOBAL_CLUSTER_MAX);

    // Initialize Union-Find parent pointers (each node is its own parent initially)
    for (int i = tid; i < n_to_process; i += BLOCK_SIZE) {
        int out_idx = residue_offset + i;
        smem->uf_parent[i] = i;  // Self-parent (initial state)
        smem->uf_consensus[i] = __ldg(&consensus_out[out_idx]);
        smem->uf_pocket_id[i] = __ldg(&pocket_assignment_out[out_idx]);
    }
    __syncthreads();

    // Step 2: Build Union-Find structure based on 8Å CA-CA distance
    // Only connect residues that BOTH pass the consensus threshold
    const float CLUSTER_DISTANCE = 8.0f;
    const float CLUSTER_DISTANCE_SQ = CLUSTER_DISTANCE * CLUSTER_DISTANCE;

    // SERIALIZED to thread 0 to avoid race conditions in Union-Find
    // (uf_find does path compression which causes write conflicts when called concurrently)
    // Performance impact minimal: n_to_process capped at 512, O(n²) = 262k ops on single thread
    if (tid == 0) {
        for (int i = 0; i < n_to_process; i++) {
            // Skip residues that didn't pass threshold
            if (smem->uf_pocket_id[i] <= 0 || smem->uf_consensus[i] <= params->consensus_threshold) {
                continue;
            }

            // Load CA coordinate for residue i
            int ca_local_idx_i = residue_offset + i;
            int ca_idx_i = __ldg(&ca_indices_packed[ca_local_idx_i]);  // Already global
            if (ca_idx_i < 0) continue;

            // ca_idx_i is already global (includes atom_offset from Rust packing)
            float3 pos_i = make_float3(
                __ldg(&atoms_packed[ca_idx_i * 3 + 0]),
                __ldg(&atoms_packed[ca_idx_i * 3 + 1]),
                __ldg(&atoms_packed[ca_idx_i * 3 + 2])
            );

            // Check against all other pocket residues
            for (int j = i + 1; j < n_to_process; j++) {
                if (smem->uf_pocket_id[j] <= 0 || smem->uf_consensus[j] <= params->consensus_threshold) {
                    continue;
                }

                // Load CA coordinate for residue j
                int ca_local_idx_j = residue_offset + j;
                int ca_idx_j = __ldg(&ca_indices_packed[ca_local_idx_j]);  // Already global
                if (ca_idx_j < 0) continue;

                // ca_idx_j is already global (includes atom_offset from Rust packing)
                float3 pos_j = make_float3(
                    __ldg(&atoms_packed[ca_idx_j * 3 + 0]),
                    __ldg(&atoms_packed[ca_idx_j * 3 + 1]),
                    __ldg(&atoms_packed[ca_idx_j * 3 + 2])
                );

                // Check distance
                float dx = pos_i.x - pos_j.x;
                float dy = pos_i.y - pos_j.y;
                float dz = pos_i.z - pos_j.z;
                float dist_sq = dx*dx + dy*dy + dz*dz;

                if (dist_sq <= CLUSTER_DISTANCE_SQ) {
                    // These residues are in contact - union them
                    uf_union(smem->uf_parent, i, j);
                }
            }
        }
    }
    __syncthreads();

    // Step 3: Flatten all parent pointers to root
    for (int i = tid; i < n_to_process; i += BLOCK_SIZE) {
        if (smem->uf_pocket_id[i] > 0) {
            smem->uf_parent[i] = uf_find(smem->uf_parent, i);
        }
    }
    __syncthreads();

    // Step 4: Count cluster sizes and find unique clusters
    // Initialize cluster size counters
    if (tid < 32) {
        smem->uf_cluster_size[tid] = 0;
    }
    __syncthreads();

    // Count members per cluster root (atomic add to shared memory)
    for (int i = tid; i < n_to_process; i += BLOCK_SIZE) {
        if (smem->uf_pocket_id[i] > 0 && smem->uf_consensus[i] > params->consensus_threshold) {
            int root = smem->uf_parent[i];
            // Atomic increment cluster size (use mod 32 for cluster slot)
            atomicAdd(&smem->uf_cluster_size[root % 32], 1);
        }
    }
    __syncthreads();

    //=========================================================================
    // Step 5: QC Gate Filtering and Final Pocket Assignment
    //
    // QUALITY CONTROL GATES (from MegaFusedParams):
    // - max_pocket_residues: Pockets exceeding this are split by consensus tier
    // - min_druggability (consensus_threshold): Already applied in Step 1-4
    // - Volume filtering happens post-kernel (requires residue coordinates)
    //
    // The max_pocket_residues QC gate prevents mega-pockets:
    // - Kinase ATP sites: ~40-60 residues
    // - max_pocket_residues=80 gives margin for binding site variation
    // - Clusters >80 residues are artifacts of aggressive merging
    //=========================================================================
    const int MIN_POCKET_SIZE = 5;                       // Hard minimum (noise filter)
    const int MAX_POCKET_SIZE = params->max_pocket_residues;  // QC gate from params (default: 80)

    for (int i = tid; i < n_to_process; i += BLOCK_SIZE) {
        int out_idx = residue_offset + i;

        // QC Gate 1: Druggability (consensus score threshold)
        // Residues below min_druggability are marked as noise
        if (smem->uf_pocket_id[i] <= 0 || smem->uf_consensus[i] <= params->min_druggability) {
            // Not a pocket residue - failed druggability QC gate
            pocket_assignment_out[out_idx] = -1;
        } else {
            int root = smem->uf_parent[i];
            int cluster_size = smem->uf_cluster_size[root % 32];

            if (cluster_size < MIN_POCKET_SIZE) {
                // QC Gate 2: Minimum size (too small = noise)
                pocket_assignment_out[out_idx] = -1;
            } else if (cluster_size > MAX_POCKET_SIZE) {
                // QC Gate 3: Maximum pocket residues (prevents mega-pockets)
                // Split by consensus score tier to create sub-pockets
                // This preserves the high-confidence core while separating extensions
                float my_score = smem->uf_consensus[i];
                if (my_score > 0.70f) {
                    pocket_assignment_out[out_idx] = root + 1;  // Core tier (highest druggability)
                } else if (my_score > 0.55f) {
                    pocket_assignment_out[out_idx] = root + 2;  // Extension tier
                } else {
                    pocket_assignment_out[out_idx] = root + 3;  // Peripheral tier
                }
            } else {
                // Good-sized cluster within QC gates - assign pocket ID
                pocket_assignment_out[out_idx] = root + 1;  // +1 so pocket IDs are 1-based
            }
        }
    }
    __syncthreads();
}

//=============================================================================
// MAIN BATCH KERNEL
// Each block processes one structure independently
// Grid: (n_structures, 1, 1), Block: (256, 1, 1)
//=============================================================================

extern "C" __global__ void __launch_bounds__(256, 2)  // 2 blocks/SM for max registers
mega_fused_batch_detection(
    // Packed input data (all structures concatenated)
    const float* __restrict__ atoms_packed,           // [total_atoms * 3]
    const int* __restrict__ ca_indices_packed,        // [total_residues]
    const float* __restrict__ conservation_packed,    // [total_residues]
    const float* __restrict__ bfactor_packed,         // [total_residues]
    const float* __restrict__ burial_packed,          // [total_residues]
    const int* __restrict__ residue_types_packed,     // [total_residues] - PRISM>4D

    // Structure descriptors (one per structure)
    const BatchStructureDesc* __restrict__ descriptors,  // [n_structures]
    int n_structures,

    // Packed output data
    float* __restrict__ consensus_out,                // [total_residues]
    int* __restrict__ confidence_out,                 // [total_residues]
    int* __restrict__ signal_mask_out,                // [total_residues]
    int* __restrict__ pocket_assignment_out,          // [total_residues]
    float* __restrict__ centrality_out,               // [total_residues]
    float* __restrict__ combined_features_out,        // [total_residues * 136] - PRISM>4D + Immunity + Epi

    // Stage 8: Cycle feature inputs (per-structure, FIX #1)
    const float* __restrict__ frequencies_packed,     // [n_structures] or nullptr
    const float* __restrict__ velocities_packed,      // [n_structures] or nullptr

    // Stage 9-10: Immunity inputs with 75-PK support (FIX #2 CORRECTED)
    const float* __restrict__ epitope_escape_packed,       // [total_residues * N_EPITOPES] or nullptr
    const float* __restrict__ immunity_events_packed,      // [n_events * 14] or nullptr
    int n_immunity_events,                                 // Number of immunity events
    int current_day,                                       // Current day index
    int variant_family_idx,                                // Variant family index (0-9)
    const float* __restrict__ p_neut_time_series_75pk,     // [n_countries × 75 × 86] or nullptr
    const float* __restrict__ current_immunity_levels_75,  // [n_structures × 75] or nullptr
    const float* __restrict__ pk_params_packed,            // [75 × 4] or nullptr
    int n_time_samples,                                    // Number of time samples (up to 86)

    // Stage 11: Epidemiological feature inputs (P0 priority for FALL prediction)
    const float* __restrict__ all_variant_frequencies,  // [n_variants] frequencies for competition
    const float* __restrict__ all_variant_gammas,       // [n_variants] fitness scores for competition
    const float* __restrict__ freq_history,             // [35 * n_variants] historical freq (column-major)
    int my_variant_idx,                                 // This structure's variant index
    int n_variants,                                     // Total variants in competition context
    float days_since_vaccine_norm,                      // Immunity recency: days/365, capped at 1
    float days_since_wave_norm,                         // Immunity recency: days/180, capped at 1
    float immunity_derivative,                          // (I_t - I_{t-30}) / 30
    float immunity_source_ratio,                        // vaccine_immunity / total_immunity
    float country_id_norm,                              // Country ID / 11

    // Runtime parameters
    const MegaFusedParams* __restrict__ params
) {
    // Each block handles one structure
    int structure_idx = blockIdx.x;
    if (structure_idx >= n_structures) return;

    // Load structure descriptor with L1 cache
    BatchStructureDesc desc;
    desc.atom_offset = __ldg(&descriptors[structure_idx].atom_offset);
    desc.residue_offset = __ldg(&descriptors[structure_idx].residue_offset);
    desc.n_atoms = __ldg(&descriptors[structure_idx].n_atoms);
    desc.n_residues = __ldg(&descriptors[structure_idx].n_residues);

    if (desc.n_residues == 0) return;

    // Allocate shared memory
    __shared__ BatchSharedMem smem;

    // Initialize shared memory with multi-compartment states
    int local_idx = threadIdx.x;
    if (local_idx < TILE_SIZE) {
        smem.compartment_proximal[local_idx] = 0.0f;
        smem.compartment_distal1[local_idx] = 0.0f;
        smem.compartment_distal2[local_idx] = 0.0f;
        smem.compartment_spine[local_idx] = 0.0f;
        smem.soma_potential[local_idx] = 0.0f;
        smem.calcium_accumulator[local_idx] = 0.0f;
        for (int k = 0; k < 8; k++) {
            smem.reservoir_activations[local_idx][k] = 0.0f;
        }
        smem.pocket_assignment[local_idx] = 0;
        // Initialize epi features (Stage 11)
        for (int k = 0; k < EPI_FEATURE_COUNT; k++) {
            smem.epi_features[local_idx][k] = 0.0f;
        }
    }
    __syncthreads();

    // Process tiles for this structure
    int n_tiles = (desc.n_residues + TILE_SIZE - 1) / TILE_SIZE;

    // Process tiles for this structure
    for (int tile = 0; tile < n_tiles; tile++) {
        int global_idx = tile * TILE_SIZE + local_idx;

        // Stage 1-6: Core pocket detection
        batch_stage1_distance_contact(atoms_packed, ca_indices_packed,
            desc.atom_offset, desc.residue_offset, desc.n_residues, tile, tile, &smem, params);
        batch_stage2_local_features(conservation_packed, bfactor_packed, burial_packed,
            desc.residue_offset, desc.n_residues, tile, &smem);
        batch_stage2b_geometry_features(desc.n_residues, tile, &smem, params);
        batch_stage3_network_centrality(desc.n_residues, tile, &smem, params);
        batch_stage4_dendritic_reservoir(desc.n_residues, tile, &smem, params);
        batch_stage5_consensus(desc.n_residues, tile, &smem, params);
        batch_stage6_kempe_refinement(desc.n_residues, tile, &smem, params);

        // Stage 7-10: PRISM>4D viral evolution features
        batch_stage7_fitness_features(desc.n_residues, tile, bfactor_packed, residue_types_packed, desc.residue_offset, &smem, params);

        // Stage 8: Cycle features with per-structure freq/velocity (FIX #1)
        float struct_freq = (frequencies_packed != nullptr) ? frequencies_packed[structure_idx] : 0.0f;
        float struct_vel = (velocities_packed != nullptr) ? velocities_packed[structure_idx] : 0.0f;
        batch_stage8_cycle_features_v2(desc.n_residues, tile, struct_freq, struct_vel, &smem);

        batch_stage8_5_synaptic_spike_phase(desc.n_residues, tile, &smem);
        batch_stage9_immunity_dynamics(desc.n_residues, tile, epitope_escape_packed, immunity_events_packed,
            n_immunity_events, current_day, variant_family_idx, desc.residue_offset, &smem, params);
        batch_stage10_immunity_integral_75pk(desc.n_residues, tile, structure_idx,
            p_neut_time_series_75pk, current_immunity_levels_75, pk_params_packed,
            n_time_samples, desc.residue_offset, &smem);

        // Stage 11: Epidemiological features (P0 priority for FALL prediction)
        batch_stage11_epi_features(desc.n_residues, tile,
            all_variant_frequencies, all_variant_gammas, my_variant_idx, n_variants,
            freq_history,
            days_since_vaccine_norm, days_since_wave_norm, immunity_derivative, immunity_source_ratio,
            country_id_norm, &smem);

        // Write combined 136-dim features and outputs (incl. 11 epi features)
        batch_write_combined_features(desc.n_residues, tile, combined_features_out, desc.residue_offset, &smem);

        if (local_idx < TILE_SIZE && global_idx < desc.n_residues) {
            int out_idx = desc.residue_offset + global_idx;
            consensus_out[out_idx] = smem.consensus_score[local_idx];
            confidence_out[out_idx] = smem.confidence[local_idx];
            signal_mask_out[out_idx] = smem.signal_mask[local_idx];
            pocket_assignment_out[out_idx] = smem.pocket_assignment[local_idx];
            centrality_out[out_idx] = smem.centrality[local_idx];
        }
        __syncthreads();
    }

    // Global Union-Find Clustering (after all tiles)
    batch_stage7_global_clustering(consensus_out, pocket_assignment_out, atoms_packed, ca_indices_packed,
        desc.atom_offset, desc.residue_offset, desc.n_residues, &smem, params);
}

//=============================================================================
// SCREENING MODE BATCH KERNEL (Minimal iterations for speed)
//=============================================================================

extern "C" __global__ void __launch_bounds__(256, 4)  // Higher occupancy, fewer registers
mega_fused_batch_screening(
    const float* __restrict__ atoms_packed,
    const int* __restrict__ ca_indices_packed,
    const float* __restrict__ conservation_packed,
    const float* __restrict__ bfactor_packed,
    const float* __restrict__ burial_packed,
    const BatchStructureDesc* __restrict__ descriptors,
    int n_structures,
    float* __restrict__ consensus_out,
    int* __restrict__ confidence_out,
    int* __restrict__ signal_mask_out,
    int* __restrict__ pocket_assignment_out,
    float* __restrict__ centrality_out,
    const MegaFusedParams* __restrict__ params
) {
    // Same as full kernel but with hardcoded minimal iterations
    // (Params should already have power_iterations=5, kempe_iterations=3 for screening)

    int structure_idx = blockIdx.x;
    if (structure_idx >= n_structures) return;

    BatchStructureDesc desc;
    desc.atom_offset = __ldg(&descriptors[structure_idx].atom_offset);
    desc.residue_offset = __ldg(&descriptors[structure_idx].residue_offset);
    desc.n_atoms = __ldg(&descriptors[structure_idx].n_atoms);
    desc.n_residues = __ldg(&descriptors[structure_idx].n_residues);

    if (desc.n_residues == 0) return;

    __shared__ BatchSharedMem smem;

    int local_idx = threadIdx.x;
    if (local_idx < TILE_SIZE) {
        smem.compartment_proximal[local_idx] = 0.0f;
        smem.compartment_distal1[local_idx] = 0.0f;
        smem.compartment_distal2[local_idx] = 0.0f;
        smem.compartment_spine[local_idx] = 0.0f;
        smem.soma_potential[local_idx] = 0.0f;
        smem.calcium_accumulator[local_idx] = 0.0f;
        for (int k = 0; k < 8; k++) {
            smem.reservoir_activations[local_idx][k] = 0.0f;
        }
        smem.pocket_assignment[local_idx] = 0;
    }
    __syncthreads();

    int n_tiles = (desc.n_residues + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < n_tiles; tile++) {
        int global_idx = tile * TILE_SIZE + local_idx;

        batch_stage1_distance_contact(atoms_packed, ca_indices_packed,
            desc.atom_offset, desc.residue_offset, desc.n_residues, tile, tile, &smem, params);

        batch_stage2_local_features(conservation_packed, bfactor_packed, burial_packed,
            desc.residue_offset, desc.n_residues, tile, &smem);

        batch_stage2b_geometry_features(desc.n_residues, tile, &smem, params);
        batch_stage3_network_centrality(desc.n_residues, tile, &smem, params);
        batch_stage4_dendritic_reservoir(desc.n_residues, tile, &smem, params);
        batch_stage5_consensus(desc.n_residues, tile, &smem, params);
        batch_stage6_kempe_refinement(desc.n_residues, tile, &smem, params);

        if (local_idx < TILE_SIZE && global_idx < desc.n_residues) {
            int out_idx = desc.residue_offset + global_idx;
            consensus_out[out_idx] = smem.consensus_score[local_idx];
            confidence_out[out_idx] = smem.confidence[local_idx];
            signal_mask_out[out_idx] = smem.signal_mask[local_idx];
            pocket_assignment_out[out_idx] = smem.pocket_assignment[local_idx];
            centrality_out[out_idx] = smem.centrality[local_idx];
        }
        __syncthreads();
    }

    //=========================================================================
    // STAGE 7: Global Union-Find Clustering (Screening Mode)
    //=========================================================================
    batch_stage7_global_clustering(
        consensus_out,
        pocket_assignment_out,
        atoms_packed,
        ca_indices_packed,
        desc.atom_offset,
        desc.residue_offset,
        desc.n_residues,
        &smem,
        params
    );
}

//=============================================================================
// HOST HELPER FUNCTIONS
//=============================================================================

extern "C" {

cudaError_t launch_mega_fused_batch(
    const float* d_atoms_packed,
    const int* d_ca_indices_packed,
    const float* d_conservation_packed,
    const float* d_bfactor_packed,
    const float* d_burial_packed,
    const int* d_residue_types_packed,
    const BatchStructureDesc* d_descriptors,
    int n_structures,
    float* d_consensus_out,
    int* d_confidence_out,
    int* d_signal_mask_out,
    int* d_pocket_assignment_out,
    float* d_centrality_out,
    float* d_combined_features_out,
    // Stage 8: Cycle feature inputs (FIX #1)
    const float* d_frequencies_packed,
    const float* d_velocities_packed,
    // Stage 9-10: Immunity inputs with 75-PK support (FIX #2 CORRECTED)
    const float* d_epitope_escape_packed,
    const float* d_immunity_events_packed,
    int n_immunity_events,
    int current_day,
    int variant_family_idx,
    const float* d_p_neut_time_series_75pk,      // 75-PK P_neut
    const float* d_current_immunity_levels_75,   // 75-PK immunity
    const float* d_pk_params_packed,              // 75 PK params
    int n_time_samples,
    // Stage 11: Epi feature inputs
    const float* d_all_variant_frequencies,
    const float* d_all_variant_gammas,
    const float* d_freq_history,
    int my_variant_idx,
    int n_variants,
    float days_since_vaccine_norm,
    float days_since_wave_norm,
    float immunity_derivative,
    float immunity_source_ratio,
    float country_id_norm,
    const MegaFusedParams* d_params,
    cudaStream_t stream
) {
    // One block per structure
    dim3 grid(n_structures);
    dim3 block(BLOCK_SIZE);

    // Prefer L1 cache over shared memory
    cudaFuncSetCacheConfig(mega_fused_batch_detection, cudaFuncCachePreferL1);

    mega_fused_batch_detection<<<grid, block, 0, stream>>>(
        d_atoms_packed, d_ca_indices_packed, d_conservation_packed,
        d_bfactor_packed, d_burial_packed, d_residue_types_packed,
        d_descriptors, n_structures,
        d_consensus_out, d_confidence_out, d_signal_mask_out,
        d_pocket_assignment_out, d_centrality_out, d_combined_features_out,
        // Stage 8: Cycle inputs (FIX #1)
        d_frequencies_packed, d_velocities_packed,
        // Stage 9-10: Immunity inputs with 75-PK (FIX #2 CORRECTED)
        d_epitope_escape_packed, d_immunity_events_packed,
        n_immunity_events, current_day, variant_family_idx,
        d_p_neut_time_series_75pk, d_current_immunity_levels_75, d_pk_params_packed,
        n_time_samples,
        // Epi feature inputs
        d_all_variant_frequencies, d_all_variant_gammas, d_freq_history,
        my_variant_idx, n_variants,
        days_since_vaccine_norm, days_since_wave_norm,
        immunity_derivative, immunity_source_ratio, country_id_norm,
        d_params
    );

    return cudaGetLastError();
}

cudaError_t launch_mega_fused_batch_screening(
    const float* d_atoms_packed,
    const int* d_ca_indices_packed,
    const float* d_conservation_packed,
    const float* d_bfactor_packed,
    const float* d_burial_packed,
    const BatchStructureDesc* d_descriptors,
    int n_structures,
    float* d_consensus_out,
    int* d_confidence_out,
    int* d_signal_mask_out,
    int* d_pocket_assignment_out,
    float* d_centrality_out,
    const MegaFusedParams* d_params,
    cudaStream_t stream
) {
    dim3 grid(n_structures);
    dim3 block(BLOCK_SIZE);

    // Maximum L1 for screening
    cudaFuncSetCacheConfig(mega_fused_batch_screening, cudaFuncCachePreferL1);

    mega_fused_batch_screening<<<grid, block, 0, stream>>>(
        d_atoms_packed, d_ca_indices_packed, d_conservation_packed,
        d_bfactor_packed, d_burial_packed, d_descriptors, n_structures,
        d_consensus_out, d_confidence_out, d_signal_mask_out,
        d_pocket_assignment_out, d_centrality_out, d_params
    );

    return cudaGetLastError();
}

//=============================================================================
// TRAINING MODE: Extract Reservoir States for Readout Training
// Same as screening kernel but also outputs 4 reservoir state values per residue
//=============================================================================

// Output: 8 TDA + 8 input + 4 geometry + 6 dendritic + 2 calcium/soma + 8 reservoir + 6 combined = 40 dims
// TDA features [0-7]: b0_scale1-4, b1_scale1-2, void_boundary, persistence
// NOTE: Must match RESERVOIR_DIM in mega_fused_batch.rs and RESERVOIR_STATE_DIM in readout_training.rs
#undef RESERVOIR_OUTPUT_DIM
#define RESERVOIR_OUTPUT_DIM 40

__global__ void __launch_bounds__(256, 4)
mega_fused_batch_training(
    const float* __restrict__ atoms_packed,
    const int* __restrict__ ca_indices_packed,
    const float* __restrict__ conservation_packed,
    const float* __restrict__ bfactor_packed,
    const float* __restrict__ burial_packed,
    const BatchStructureDesc* __restrict__ descriptors,
    int n_structures,
    float* __restrict__ consensus_out,
    int* __restrict__ confidence_out,
    int* __restrict__ signal_mask_out,
    int* __restrict__ pocket_assignment_out,
    float* __restrict__ centrality_out,
    float* __restrict__ reservoir_states_out,  // [total_residues * RESERVOIR_OUTPUT_DIM]
    const MegaFusedParams* __restrict__ params
) {
    int structure_idx = blockIdx.x;
    if (structure_idx >= n_structures) return;

    BatchStructureDesc desc;
    desc.atom_offset = __ldg(&descriptors[structure_idx].atom_offset);
    desc.residue_offset = __ldg(&descriptors[structure_idx].residue_offset);
    desc.n_atoms = __ldg(&descriptors[structure_idx].n_atoms);
    desc.n_residues = __ldg(&descriptors[structure_idx].n_residues);

    if (desc.n_residues == 0) return;

    __shared__ BatchSharedMem smem;

    int local_idx = threadIdx.x;
    if (local_idx < TILE_SIZE) {
        // Initialize multi-compartment dendritic states
        smem.compartment_proximal[local_idx] = 0.0f;
        smem.compartment_distal1[local_idx] = 0.0f;
        smem.compartment_distal2[local_idx] = 0.0f;
        smem.compartment_spine[local_idx] = 0.0f;
        smem.soma_potential[local_idx] = 0.0f;
        smem.calcium_accumulator[local_idx] = 0.0f;
        // Initialize reservoir activations (8 values)
        for (int k = 0; k < 8; k++) {
            smem.reservoir_activations[local_idx][k] = 0.0f;
        }
        smem.pocket_assignment[local_idx] = 0;
    }
    __syncthreads();

    int n_tiles = (desc.n_residues + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < n_tiles; tile++) {
        int global_idx = tile * TILE_SIZE + local_idx;

        batch_stage1_distance_contact(atoms_packed, ca_indices_packed,
            desc.atom_offset, desc.residue_offset,
            desc.n_residues, tile, tile, &smem, params);

        batch_stage2_local_features(conservation_packed, bfactor_packed, burial_packed,
            desc.residue_offset, desc.n_residues, tile, &smem);

        batch_stage2b_geometry_features(desc.n_residues, tile, &smem, params);

        // Stage 2c: TDA Features (topological binding site features)
        batch_stage2c_tda_features(desc.n_residues, tile, &smem, params);

        batch_stage3_network_centrality(desc.n_residues, tile, &smem, params);

        batch_stage4_dendritic_reservoir(desc.n_residues, tile, &smem, params);

        batch_stage5_consensus(desc.n_residues, tile, &smem, params);

        batch_stage6_kempe_refinement(desc.n_residues, tile, &smem, params);

        // Write outputs including enhanced reservoir states
        if (local_idx < TILE_SIZE && global_idx < desc.n_residues) {
            int out_idx = desc.residue_offset + global_idx;
            consensus_out[out_idx] = smem.consensus_score[local_idx];
            confidence_out[out_idx] = smem.confidence[local_idx];
            signal_mask_out[out_idx] = smem.signal_mask[local_idx];
            pocket_assignment_out[out_idx] = smem.pocket_assignment[local_idx];
            centrality_out[out_idx] = smem.centrality[local_idx];

            // Export enhanced reservoir states for training (40 values per residue)
            // Layout:
            // [0-7]:   TDA FEATURES (topological binding site features)
            // [8-15]:  Raw input features (degree, conservation, centrality, bfactor, burial, eigenvector, geometric, consensus)
            // [16-19]: GEOMETRY FEATURES (CRITICAL for binding site discrimination)
            // [20-23]: Multi-compartment dendritic states (proximal, distal1, distal2, spine)
            // [24]:    Calcium accumulator (long-term binding site memory)
            // [25]:    Soma potential (integrated dendritic output)
            // [26-33]: Top 8 reservoir neuron activations
            // [34-39]: Combined geometry+network signals
            int res_out_idx = out_idx * RESERVOIR_OUTPUT_DIM;

            // TDA FEATURES (topological - highest value for binding site prediction)
            reservoir_states_out[res_out_idx + 0] = smem.tda_b0_scale1[local_idx];           // β₀ at 4Å
            reservoir_states_out[res_out_idx + 1] = smem.tda_b0_scale2[local_idx];           // β₀ at 6Å
            reservoir_states_out[res_out_idx + 2] = smem.tda_b0_scale3[local_idx];           // β₀ at 8Å
            reservoir_states_out[res_out_idx + 3] = smem.tda_b0_scale4[local_idx];           // β₀ at 10Å
            reservoir_states_out[res_out_idx + 4] = smem.tda_b1_scale1[local_idx];           // β₁ at 4Å
            reservoir_states_out[res_out_idx + 5] = smem.tda_b1_scale2[local_idx];           // β₁ at 6Å
            reservoir_states_out[res_out_idx + 6] = smem.void_boundary[local_idx];           // Bridge-point score (HIGHEST VALUE)
            reservoir_states_out[res_out_idx + 7] = smem.persistence_score[local_idx];       // Persistence score

            // Input features (normalized to reasonable ranges)
            reservoir_states_out[res_out_idx + 8] = smem.degree[local_idx] / 20.0f;          // degree
            reservoir_states_out[res_out_idx + 9] = smem.conservation[local_idx];            // conservation
            reservoir_states_out[res_out_idx + 10] = smem.centrality[local_idx];             // centrality
            reservoir_states_out[res_out_idx + 11] = smem.bfactor[local_idx];                // flexibility
            reservoir_states_out[res_out_idx + 12] = smem.burial[local_idx];                 // burial
            reservoir_states_out[res_out_idx + 13] = smem.eigenvector[local_idx];            // eigenvector
            reservoir_states_out[res_out_idx + 14] = smem.geometric_score[local_idx];        // geometric score
            reservoir_states_out[res_out_idx + 15] = smem.consensus_score[local_idx];        // consensus score

            // GEOMETRY FEATURES (CRITICAL for binding site discrimination)
            reservoir_states_out[res_out_idx + 16] = smem.hse_up[local_idx];                 // HSE upper hemisphere
            reservoir_states_out[res_out_idx + 17] = smem.hse_down[local_idx];               // HSE lower hemisphere
            reservoir_states_out[res_out_idx + 18] = smem.local_concavity[local_idx];        // Local surface concavity
            reservoir_states_out[res_out_idx + 19] = smem.pocket_depth[local_idx];           // Pocket depth proxy

            // Multi-compartment dendritic states
            reservoir_states_out[res_out_idx + 20] = smem.compartment_proximal[local_idx];   // proximal (fast)
            reservoir_states_out[res_out_idx + 21] = smem.compartment_distal1[local_idx];    // distal1 (medium)
            reservoir_states_out[res_out_idx + 22] = smem.compartment_distal2[local_idx];    // distal2 (slow)
            reservoir_states_out[res_out_idx + 23] = smem.compartment_spine[local_idx];      // spine (long-term)

            // Calcium and soma
            reservoir_states_out[res_out_idx + 24] = smem.calcium_accumulator[local_idx];    // calcium (LTP memory)
            reservoir_states_out[res_out_idx + 25] = smem.soma_potential[local_idx];         // soma potential

            // Top 8 reservoir neuron activations (every 4th of 32 neurons)
            #pragma unroll 8
            for (int k = 0; k < 8; k++) {
                reservoir_states_out[res_out_idx + 26 + k] = smem.reservoir_activations[local_idx][k];
            }

            // Combined geometry+network signals
            reservoir_states_out[res_out_idx + 34] = smem.hse_up[local_idx] * smem.centrality[local_idx];
            reservoir_states_out[res_out_idx + 35] = smem.local_concavity[local_idx] * smem.burial[local_idx];
            reservoir_states_out[res_out_idx + 36] = smem.pocket_depth[local_idx] * smem.conservation[local_idx];
            reservoir_states_out[res_out_idx + 37] = (smem.hse_up[local_idx] - smem.hse_down[local_idx]) * smem.degree[local_idx] / 20.0f;
            reservoir_states_out[res_out_idx + 38] = smem.local_concavity[local_idx] * smem.geometric_score[local_idx];
            reservoir_states_out[res_out_idx + 39] = smem.pocket_depth[local_idx] * smem.soma_potential[local_idx];
        }
        __syncthreads();
    }

    // Skip global clustering for training (only need reservoir states)
}

// Host wrapper for training kernel
cudaError_t launch_mega_fused_batch_training(
    const float* d_atoms_packed,
    const int* d_ca_indices_packed,
    const float* d_conservation_packed,
    const float* d_bfactor_packed,
    const float* d_burial_packed,
    const BatchStructureDesc* d_descriptors,
    int n_structures,
    float* d_consensus_out,
    int* d_confidence_out,
    int* d_signal_mask_out,
    int* d_pocket_assignment_out,
    float* d_centrality_out,
    float* d_reservoir_states_out,  // [total_residues * 4]
    const MegaFusedParams* d_params,
    cudaStream_t stream
) {
    dim3 grid(n_structures);
    dim3 block(BLOCK_SIZE);

    cudaFuncSetCacheConfig(mega_fused_batch_training, cudaFuncCachePreferL1);

    mega_fused_batch_training<<<grid, block, 0, stream>>>(
        d_atoms_packed, d_ca_indices_packed, d_conservation_packed,
        d_bfactor_packed, d_burial_packed, d_descriptors, n_structures,
        d_consensus_out, d_confidence_out, d_signal_mask_out,
        d_pocket_assignment_out, d_centrality_out, d_reservoir_states_out, d_params
    );

    return cudaGetLastError();
}

}  // extern "C"

//=============================================================================
// PERFORMANCE ANALYSIS
//=============================================================================
/*
BATCH MODE BENEFITS:
- Single kernel launch for ALL structures (vs N launches)
- L1 cache preference reduces global memory latency
- __ldg() intrinsics use read-only texture cache
- Register-heavy computation minimizes shared memory pressure
- No inter-structure synchronization needed

MEMORY LAYOUT (PACKED):
atoms_packed:       [struct0_atoms | struct1_atoms | ... | structN_atoms]
ca_indices_packed:  [struct0_ca    | struct1_ca    | ... | structN_ca]
descriptors:        [desc0, desc1, ..., descN] with offsets

EXPECTED PERFORMANCE (RTX 3060 Laptop):
- Sequential (221 structures): ~2-3 seconds (9-13ms per structure)
- Batch kernel: ~50-100ms total (0.2-0.5ms per structure)
- Speedup: ~20-50x

THROUGHPUT:
- Batch mode: ~2000-4000 structures/second
- Screening mode: ~4000-8000 structures/second (3 kempe, 5 power iters)
*/
