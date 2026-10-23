// [POPULATION-IZHIKEVICH] 4-Type Population Neuron Model with Dendritic Compartments
//
// Each voxel has 1000 Izhikevich neurons: 250 per type (FS, RS, IB, CH).
// Each neuron has a two-compartment dendritic model.
//
// Population spike rule:
//   FS firing rate > 20% AND RS firing rate > 15% → voxel spikes
//
// Memory layout (SoA for coalesced access):
//   v_soma[n_voxels * N_POP]  — membrane potential (persistent)
//   u_recov[n_voxels * N_POP] — recovery variable (persistent)
//   v_dend[n_voxels * N_POP]  — dendritic potential (persistent)
//   threshold_adapt[n_voxels * 4] — STDP per type (persistent)
//   pop_spike_flags[n_voxels]  — output: 1 if population spike this step
//
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

// Population size per voxel
#define N_POP       1000
#define N_PER_TYPE  250

// Type indices
#define TYPE_FS  0
#define TYPE_RS  1
#define TYPE_IB  2
#define TYPE_CH  3

// ═══════════════════════════════════════════════════════════════════════════════
// Izhikevich base parameters per type
// ═══════════════════════════════════════════════════════════════════════════════

struct IzhParams {
    float a, b, c, d;
    float g_c;      // dendritic coupling
    float g_leak;   // dendritic leak conductance
};

__device__ __constant__ IzhParams BASE_PARAMS[4] = {
    // FS:  fast-spiking — EDL/EFP channel, short dendrite
    { 0.1f,  0.2f,  -65.0f, 2.0f,   0.3f, 0.05f },
    // RS:  regular-spiking — water density/dewetting, long dendrite
    { 0.02f, 0.2f,  -65.0f, 8.0f,   0.1f, 0.05f },
    // IB:  intrinsic bursting — benzene cosolvent, medium dendrite
    { 0.02f, 0.2f,  -55.0f, 4.0f,   0.2f, 0.05f },
    // CH:  chattering — aromatics/UV channel, medium dendrite
    { 0.02f, 0.25f, -55.0f, 0.05f,  0.2f, 0.05f },
};

// ═══════════════════════════════════════════════════════════════════════════════
// Deterministic per-neuron parameter jitter (±10%)
// Uses a simple hash of (voxel_idx, neuron_local_idx) for reproducibility
// ═══════════════════════════════════════════════════════════════════════════════

__device__ __forceinline__ float hash_jitter(unsigned int voxel_idx, unsigned int local_idx, unsigned int param_id) {
    // Wang hash variant — deterministic, uniform in [0,1)
    unsigned int seed = voxel_idx * 1000u + local_idx;
    seed ^= param_id * 2654435761u;
    seed ^= seed >> 16;
    seed *= 0x45d9f3bu;
    seed ^= seed >> 16;
    // Map to [-0.1, +0.1]
    return ((seed & 0xFFFFu) / 65535.0f - 0.5f) * 0.2f;
}

__device__ __forceinline__ float jitter_param(float base, unsigned int voxel, unsigned int local, unsigned int pid) {
    return base * (1.0f + hash_jitter(voxel, local, pid));
}

// ═══════════════════════════════════════════════════════════════════════════════
// KERNEL 1: Population Izhikevich Step — Multi-Substep
//
// One thread per neuron (n_voxels × N_POP).
// Each thread runs n_substeps Izhikevich iterations in registers before
// writing back to global memory. This amortizes the global load/store cost
// across many FLOPs, increasing arithmetic intensity ~n_substeps×.
//
// Per substep:
//   1. Update dendritic compartment from held input
//   2. Compute somatic current from dendrite + STDP
//   3. Two half-step Euler Izhikevich integration
//   4. Spike check with local counter accumulation
//
// After all substeps: single global store + single atomicAdd per counter.
// ═══════════════════════════════════════════════════════════════════════════════

__global__ void population_izhikevich_step(
    float* __restrict__ v_soma,             // [n_voxels * N_POP] persistent
    float* __restrict__ u_recov,            // [n_voxels * N_POP] persistent
    float* __restrict__ v_dend,             // [n_voxels * N_POP] persistent
    float* __restrict__ threshold_adapt,    // [n_voxels * 4] per-type STDP
    const float* __restrict__ water_density,// [n_voxels] from MD
    const float* __restrict__ uv_input,     // [n_voxels] aromatic/UV excitation
    const float* __restrict__ efp_input,    // [n_voxels] EDL/EFP field
    const float* __restrict__ benz_input,   // [n_voxels] benzene cosolvent probe
    unsigned int* __restrict__ pop_spike_flags, // [n_voxels] (unused, reserved)
    float* __restrict__ pop_fs_rate,        // [n_voxels] output: FS spike count
    float* __restrict__ pop_rs_rate,        // [n_voxels] output: RS spike count
    float baseline_density,
    unsigned int n_voxels,
    unsigned int total_neurons,             // n_voxels * N_POP
    unsigned int n_substeps                 // number of Izhikevich iterations per launch
) {
    unsigned int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id >= total_neurons) return;

    unsigned int voxel_idx = global_id / N_POP;
    unsigned int local_idx = global_id % N_POP;
    unsigned int type_idx  = local_idx / N_PER_TYPE;  // 0=FS, 1=RS, 2=IB, 3=CH

    if (voxel_idx >= n_voxels) return;

    // ─── Load parameters ONCE (constant across substeps) ───
    IzhParams bp = BASE_PARAMS[type_idx];
    float a     = jitter_param(bp.a, voxel_idx, local_idx, 0);
    float b     = jitter_param(bp.b, voxel_idx, local_idx, 1);
    float c     = jitter_param(bp.c, voxel_idx, local_idx, 2);
    float d     = jitter_param(bp.d, voxel_idx, local_idx, 3);
    float g_c   = bp.g_c;
    float g_leak = bp.g_leak;

    // ─── Load persistent state ONCE into registers ───
    float v  = v_soma[global_id];
    float u  = u_recov[global_id];
    float vd = v_dend[global_id];
    float ta = threshold_adapt[voxel_idx * 4 + type_idx];

    // ─── Load input ONCE (held constant across substeps — input changes at MD sync rate) ───
    float I_input = 0.0f;
    float wd = water_density[voxel_idx];
    float dewetting_signal = (baseline_density - wd) * 50.0f;

    switch (type_idx) {
        case TYPE_FS:  I_input = efp_input[voxel_idx] * 40.0f;   break;
        case TYPE_RS:  I_input = dewetting_signal;                 break;
        case TYPE_IB:  I_input = benz_input[voxel_idx] * 40.0f;  break;
        case TYPE_CH:  I_input = uv_input[voxel_idx] * 40.0f;    break;
    }

    // ─── Substep loop: all computation in registers ───
    unsigned int local_fs_spikes = 0;
    unsigned int local_rs_spikes = 0;
    unsigned int local_total_spikes = 0;

    for (unsigned int step = 0; step < n_substeps; step++) {
        // Dendritic compartment
        vd += -g_leak * vd + I_input + g_c * (v - vd);

        // Somatic current from dendrite + STDP adaptation
        float I_soma = g_c * (vd - v) + ta;

        // Izhikevich dynamics — two half-step Euler for stability
        float v1 = v + 0.5f * (0.04f * v * v + 5.0f * v + 140.0f - u + I_soma);
        v1 = v1 + 0.5f * (0.04f * v1 * v1 + 5.0f * v1 + 140.0f - u + I_soma);
        u = u + a * (b * v1 - u);

        // Spike check
        if (v1 >= 30.0f) {
            v = c;
            u = u + d;
            local_total_spikes++;
            if (type_idx == TYPE_FS) local_fs_spikes++;
            else if (type_idx == TYPE_RS) local_rs_spikes++;
        } else {
            v = v1;
        }
    }

    // ─── Store state ONCE back to global memory ───
    v_soma[global_id] = v;
    u_recov[global_id] = u;
    v_dend[global_id] = vd;

    // ─── Spike counts: single atomicAdd per counter (not per substep) ───
    if (local_fs_spikes > 0) {
        atomicAdd(&pop_fs_rate[voxel_idx], (float)local_fs_spikes);
    }
    if (local_rs_spikes > 0) {
        atomicAdd(&pop_rs_rate[voxel_idx], (float)local_rs_spikes);
    }

    // ─── STDP: accumulated over all substeps ───
    float adaptation_rate = 0.001f;
    float target_rate = 0.1f;
    float spike_f = (float)local_total_spikes;
    float target_total = target_rate * (float)n_substeps;
    float delta = adaptation_rate * (spike_f - target_total) / (float)N_PER_TYPE;
    atomicAdd(&threshold_adapt[voxel_idx * 4 + type_idx], delta);
}

// ═══════════════════════════════════════════════════════════════════════════════
// KERNEL 2: Population Spike Decision + Emit
//
// One thread per voxel. Reads accumulated FS/RS rates, applies quorum rule,
// emits spike events. Resets rate accumulators for next step.
// ═══════════════════════════════════════════════════════════════════════════════

__global__ void population_spike_emit(
    const float* __restrict__ pop_fs_rate,       // [n_voxels] accumulated FS spike count
    const float* __restrict__ pop_rs_rate,       // [n_voxels] accumulated RS spike count
    const float* __restrict__ water_density,     // [n_voxels]
    const float* __restrict__ voxel_positions,   // [n_voxels * 3]
    const unsigned int* __restrict__ neighbor_counts, // [n_voxels] from OptiX
    SpikeEventGpu* __restrict__ spike_output,
    unsigned int* __restrict__ spike_count,
    float baseline_density,
    unsigned int n_voxels,
    unsigned int current_timestep,
    int ramp_phase,
    unsigned int max_spikes,
    unsigned int n_substeps                      // for firing rate normalization
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_voxels) return;

    // Normalize: each neuron can spike once per substep, so max = N_PER_TYPE * n_substeps
    float fs_count = pop_fs_rate[idx];
    float rs_count = pop_rs_rate[idx];
    float max_possible = (float)(N_PER_TYPE * n_substeps);
    float fs_rate = fs_count / max_possible;
    float rs_rate = rs_count / max_possible;

    // Population spike rule: FS > 20% AND RS > 15%
    if (fs_rate <= 0.20f || rs_rate <= 0.15f) return;

    // Claim output slot
    unsigned int slot = atomicAdd(spike_count, 1);
    if (slot >= max_spikes) return;

    float density = water_density[idx];
    unsigned int nbr_count = neighbor_counts[idx];
    bool synchronous = nbr_count >= 3;
    float coop_factor = synchronous ? (1.0f + 0.5f * fminf((float)nbr_count, 10.0f)) : 1.0f;

    // Intensity combines population activation strength and dewetting magnitude
    float V = baseline_density - density;
    float pop_strength = (fs_rate + rs_rate) * 0.5f;  // mean population activation
    float intensity = V * coop_factor * pop_strength * 10.0f;

    SpikeEventGpu spike;
    spike.timestep = (int)current_timestep;
    spike.voxel_idx = (int)idx;
    spike.position[0] = voxel_positions[idx * 3];
    spike.position[1] = voxel_positions[idx * 3 + 1];
    spike.position[2] = voxel_positions[idx * 3 + 2];
    spike.intensity = intensity;
    for (int r = 0; r < 8; r++) spike.nearby_residues[r] = -1;
    spike.n_residues = 0;
    spike.spike_source = 2;  // LIF population
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
// KERNEL 3: Reset population rate accumulators
// ═══════════════════════════════════════════════════════════════════════════════

__global__ void reset_population_rates(
    float* __restrict__ pop_fs_rate,
    float* __restrict__ pop_rs_rate,
    unsigned int n_voxels
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_voxels) return;
    pop_fs_rate[idx] = 0.0f;
    pop_rs_rate[idx] = 0.0f;
}

// ═══════════════════════════════════════════════════════════════════════════════
// KERNEL 4: Initialize Izhikevich state
//
// Sets v_soma = c (resting), u_recov = b*c, v_dend = 0
// Must be called once before first step.
// ═══════════════════════════════════════════════════════════════════════════════

__global__ void init_population_state(
    float* __restrict__ v_soma,
    float* __restrict__ u_recov,
    float* __restrict__ v_dend,
    float* __restrict__ threshold_adapt,
    unsigned int n_voxels,
    unsigned int total_neurons
) {
    unsigned int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize neuron state
    if (global_id < total_neurons) {
        unsigned int local_idx = global_id % N_POP;
        unsigned int type_idx = local_idx / N_PER_TYPE;
        IzhParams bp = BASE_PARAMS[type_idx];
        float c_jit = bp.c * (1.0f + hash_jitter(global_id / N_POP, local_idx, 2));
        float b_jit = bp.b * (1.0f + hash_jitter(global_id / N_POP, local_idx, 1));
        v_soma[global_id] = c_jit;
        u_recov[global_id] = b_jit * c_jit;
        v_dend[global_id] = 0.0f;
    }

    // Initialize per-type threshold adaptation (4 per voxel)
    if (global_id < n_voxels * 4) {
        threshold_adapt[global_id] = 0.0f;
    }
}

} // extern "C"
