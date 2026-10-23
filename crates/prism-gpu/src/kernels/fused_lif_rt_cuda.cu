// [FUSED-LIF-RT] Post-Traversal CUDA Kernels
//
// These kernels run AFTER the OptiX BVH traversal to:
// 1. Classify and emit spike events based on V + neighbor context
// 2. Update STDP thresholds (GPU-resident, persists across timesteps)
//
// Confidence classification (2D discriminant):
//   early + synchronous → HIGH_CONFIDENCE
//   early + isolated    → TEMPORAL_ARTIFACT
//   late  + synchronous → STRUCTURAL_NOISE
//   late  + isolated    → THERMAL_NOISE (discard)

// GPU spike event struct (must match GpuSpikeEvent in fused_engine.rs)
struct SpikeEventGpu {
    int timestep;
    int voxel_idx;
    float position[3];
    float intensity;
    int nearby_residues[8];
    int n_residues;
    int spike_source;
    float wavelength_nm;
    int aromatic_type;
    int aromatic_residue_id;
    float water_density;
    float vibrational_energy;
    int n_nearby_excited;
    int ramp_phase;
};

extern "C" {

// ═══════════════════════════════════════════════════════════════════════════════
// KERNEL 1: Spike Classification and Emission
//
// For each voxel:
//   if V > threshold_hi AND neighbor_count >= K → emit HIGH_CONFIDENCE spike
//   if V > threshold_hi AND neighbor_count == 0 → mark TEMPORAL_ARTIFACT
//   if V <= threshold_hi AND neighbor_count >= K → mark STRUCTURAL_NOISE
//   if V <= threshold_hi AND neighbor_count == 0 → THERMAL_NOISE (discard)
//
// Only HIGH_CONFIDENCE and TEMPORAL_ARTIFACT spikes are emitted to output buffer.
// ═══════════════════════════════════════════════════════════════════════════════

__global__ void spike_classify_emit(
    const float* __restrict__ water_density,        // [n_voxels] density field
    const float* __restrict__ voxel_positions,      // [n_voxels * 3] xyz
    const float* __restrict__ threshold_adaptation, // [n_voxels] STDP thresholds
    const unsigned int* __restrict__ neighbor_counts,// [n_voxels] from OptiX pass
    const float* __restrict__ neighbor_density_sum,  // [n_voxels] from OptiX pass
    SpikeEventGpu* __restrict__ spike_output,        // output buffer
    unsigned int* __restrict__ spike_count,          // atomic counter
    float threshold_lo,
    float threshold_hi,
    float baseline_density,
    unsigned int min_neighbors,
    unsigned int n_voxels,
    unsigned int current_timestep,
    float current_temperature,
    int ramp_phase,
    unsigned int max_spikes
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_voxels) return;

    float density = water_density[idx];
    float V = baseline_density - density;
    float adapted_lo = threshold_lo + threshold_adaptation[idx];
    float adapted_hi = threshold_hi + threshold_adaptation[idx];

    // Not even a pre-spike candidate
    if (V <= adapted_lo) return;

    unsigned int nbr_count = neighbor_counts[idx];
    bool above_hi = V > adapted_hi;
    bool synchronous = nbr_count >= min_neighbors;

    // Classification
    // Only emit HIGH_CONFIDENCE and TEMPORAL_ARTIFACT (early spikes)
    // STRUCTURAL_NOISE and THERMAL_NOISE are suppressed
    if (!above_hi) return;  // Only emit when V exceeds high threshold

    // Claim a slot in the output buffer
    unsigned int slot = atomicAdd(spike_count, 1);
    if (slot >= max_spikes) return;  // Buffer full — safety check

    // Compute intensity: V scaled by neighbor cooperation
    float coop_factor = synchronous ? (1.0f + 0.5f * min((float)nbr_count, 10.0f)) : 1.0f;
    float intensity = V * coop_factor;

    // Write spike event
    SpikeEventGpu spike;
    spike.timestep = (int)current_timestep;
    spike.voxel_idx = (int)idx;
    spike.position[0] = voxel_positions[idx * 3];
    spike.position[1] = voxel_positions[idx * 3 + 1];
    spike.position[2] = voxel_positions[idx * 3 + 2];
    spike.intensity = intensity;
    // nearby_residues populated by caller (needs topology context)
    for (int r = 0; r < 8; r++) spike.nearby_residues[r] = -1;
    spike.n_residues = 0;
    spike.spike_source = 2;  // LIF
    spike.wavelength_nm = 0.0f;
    spike.aromatic_type = -1;
    spike.aromatic_residue_id = -1;
    spike.water_density = density;
    spike.vibrational_energy = 0.0f;
    spike.n_nearby_excited = synchronous ? (int)nbr_count : 0;
    spike.ramp_phase = ramp_phase;

    spike_output[slot] = spike;
}

// ═══════════════════════════════════════════════════════════════════════════════
// KERNEL 2: STDP Threshold Adaptation (GPU-Resident)
//
// Updates per-voxel adaptive thresholds based on spike+neighbor context:
//   if spiked && isolated: threshold += rate  (harder to spike = less noise)
//   if spiked && synchronous: threshold -= rate*0.5  (easier to spike = encourage)
//   clamp to [0.1 * base, 5.0 * base]
//
// This kernel runs AFTER spike_classify_emit and uses the same neighbor_counts.
// The threshold_adaptation buffer persists across timesteps (GPU-resident).
// ═══════════════════════════════════════════════════════════════════════════════

__global__ void stdp_update_thresholds(
    float* __restrict__ threshold_adaptation,       // [n_voxels] in/out
    const float* __restrict__ water_density,        // [n_voxels]
    const unsigned int* __restrict__ neighbor_counts,// [n_voxels]
    float threshold_lo,
    float baseline_density,
    float adaptation_rate,
    unsigned int min_neighbors,
    unsigned int n_voxels,
    float threshold_lo_base                         // original unmodified threshold_lo
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_voxels) return;

    float density = water_density[idx];
    float V = baseline_density - density;
    float adapted_lo = threshold_lo + threshold_adaptation[idx];

    // Only update for voxels that were pre-spike candidates
    if (V <= adapted_lo) return;

    unsigned int nbr_count = neighbor_counts[idx];
    float current_adapt = threshold_adaptation[idx];

    if (nbr_count == 0) {
        // Isolated spike → raise threshold (suppress noise)
        current_adapt += adaptation_rate;
    } else if (nbr_count >= min_neighbors) {
        // Synchronous spike → lower threshold (encourage cooperative detection)
        current_adapt -= adaptation_rate * 0.5f;
    }

    // Clamp to [0.1 * base, 5.0 * base]
    float lo_clamp = -0.9f * threshold_lo_base;  // allows going 90% below base
    float hi_clamp = 4.0f * threshold_lo_base;    // allows going 4x above base
    current_adapt = fminf(fmaxf(current_adapt, lo_clamp), hi_clamp);

    threshold_adaptation[idx] = current_adapt;
}

// ═══════════════════════════════════════════════════════════════════════════════
// KERNEL 3: Reset neighbor counts/density for next timestep
// ═══════════════════════════════════════════════════════════════════════════════

__global__ void reset_fused_buffers(
    unsigned int* __restrict__ neighbor_counts,
    float* __restrict__ neighbor_density_sum,
    unsigned int n_voxels
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_voxels) return;

    neighbor_counts[idx] = 0;
    neighbor_density_sum[idx] = 0.0f;
}

} // extern "C"
