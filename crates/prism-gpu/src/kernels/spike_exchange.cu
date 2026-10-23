// ═══════════════════════════════════════════════════════════════════════
// PRISM-TWIN: Spike Density Exchange Kernels
//
// Layer 1: Compute spike density per coarse region, exchange between streams
// Layer 2: Cross-correlation computation on paired spike histories
//
// These kernels run on a dedicated exchange CUDA stream, separate from
// the simulation streams A and B.
// ═══════════════════════════════════════════════════════════════════════

// Spike record must match the GpuSpikeEvent struct in fused_engine
struct SpikeRecord {
    int spike_source;
    int voxel_idx;
    float x, y, z;
    float intensity;
    float vibrational_energy;
    float water_density;
    int n_nearby_excited;
    int timestep;
    int ccns_phase;
    int pad;  // alignment
};

// ─────────────────────────────────────────────────────────────────────
// KERNEL 1: Compute spike density per coarse region
// ─────────────────────────────────────────────────────────────────────

extern "C" __global__ void compute_spike_density(
    const SpikeRecord* __restrict__ spike_buffer,
    const int spike_count,
    float* __restrict__ density_map,      // output: [n_regions]
    float grid_origin_x, float grid_origin_y, float grid_origin_z,
    float region_size_x, float region_size_y, float region_size_z,
    int n_regions_x, int n_regions_y, int n_regions_z,
    int last_exchange_step,
    int current_step
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= spike_count) return;

    SpikeRecord spike = spike_buffer[idx];

    // Only count spikes since last exchange
    if (spike.timestep <= last_exchange_step) return;

    // Map spike position to region index
    int rx = (int)((spike.x - grid_origin_x) / region_size_x);
    int ry = (int)((spike.y - grid_origin_y) / region_size_y);
    int rz = (int)((spike.z - grid_origin_z) / region_size_z);

    // Bounds check
    if (rx < 0 || rx >= n_regions_x) return;
    if (ry < 0 || ry >= n_regions_y) return;
    if (rz < 0 || rz >= n_regions_z) return;

    int region_idx = rx + ry * n_regions_x + rz * n_regions_x * n_regions_y;

    // Atomic add intensity (benign races OK — approximate density)
    atomicAdd(&density_map[region_idx], spike.intensity);
}


// ─────────────────────────────────────────────────────────────────────
// KERNEL 2: Apply threshold modifiers from other stream's density
// ─────────────────────────────────────────────────────────────────────

extern "C" __global__ void apply_threshold_modifiers(
    float* __restrict__ oscillator_thresholds,  // in/out: per-voxel thresholds
    const float* __restrict__ base_thresholds,  // original thresholds (read-only)
    const float* __restrict__ density_map_other, // other stream's spike density
    int n_voxels,
    int n_voxels_x, int n_voxels_y, int n_voxels_z,
    float voxel_size,
    float region_size,
    int n_regions_x, int n_regions_y, int n_regions_z,
    float sensitivity_boost,
    float max_reduction
) {
    int voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (voxel_idx >= n_voxels) return;

    // Decompose linear index to 3D
    int vz = voxel_idx / (n_voxels_x * n_voxels_y);
    int vy = (voxel_idx - vz * n_voxels_x * n_voxels_y) / n_voxels_x;
    int vx = voxel_idx - vz * n_voxels_x * n_voxels_y - vy * n_voxels_x;

    // Map voxel to coarse region
    int rx = (int)(vx * voxel_size / region_size);
    int ry = (int)(vy * voxel_size / region_size);
    int rz = (int)(vz * voxel_size / region_size);

    rx = min(rx, n_regions_x - 1);
    ry = min(ry, n_regions_y - 1);
    rz = min(rz, n_regions_z - 1);

    int region_idx = rx + ry * n_regions_x + rz * n_regions_x * n_regions_y;

    float other_density = density_map_other[region_idx];

    // Sigmoid mapping: density → threshold reduction
    // Higher density in other stream → lower threshold in this stream
    float boost = max_reduction * (1.0f - 1.0f / (1.0f + other_density * sensitivity_boost));

    // Apply: reduce threshold but never below 20% of base (safety floor)
    float base = base_thresholds[voxel_idx];
    float new_threshold = fmaxf(base * 0.2f, base - boost);

    oscillator_thresholds[voxel_idx] = new_threshold;
}


// ─────────────────────────────────────────────────────────────────────
// KERNEL 3: Cross-correlation on paired spike ring buffers (Layer 2)
//
// For each voxel region, computes CCF between A's and B's recent
// spike history at multiple lags.
// ─────────────────────────────────────────────────────────────────────

extern "C" __global__ void compute_cross_correlation(
    const float* __restrict__ ring_a,    // [n_regions * ring_size] A's spike history
    const float* __restrict__ ring_b,    // [n_regions * ring_size] B's spike history
    float* __restrict__ ccf_map,         // output: [n_regions * n_lags]
    int n_regions,
    int ring_size,
    int n_lags,                          // total lags (e.g. 64 for τ = -32..+31)
    int ring_head                        // current write position in ring
) {
    int region = blockIdx.x * blockDim.x + threadIdx.x;
    if (region >= n_regions) return;

    int half_lags = n_lags / 2;
    int base_a = region * ring_size;
    int base_b = region * ring_size;
    int base_ccf = region * n_lags;

    // Compute mean of A and B for this region
    float mean_a = 0.0f, mean_b = 0.0f;
    for (int i = 0; i < ring_size; i++) {
        mean_a += ring_a[base_a + i];
        mean_b += ring_b[base_b + i];
    }
    mean_a /= ring_size;
    mean_b /= ring_size;

    // Compute variance for normalization
    float var_a = 0.0f, var_b = 0.0f;
    for (int i = 0; i < ring_size; i++) {
        float da = ring_a[base_a + i] - mean_a;
        float db = ring_b[base_b + i] - mean_b;
        var_a += da * da;
        var_b += db * db;
    }
    float norm = sqrtf(var_a * var_b);
    if (norm < 1e-10f) norm = 1e-10f;  // avoid division by zero

    // Compute CCF at each lag
    for (int lag_idx = 0; lag_idx < n_lags; lag_idx++) {
        int tau = lag_idx - half_lags;  // τ from -half_lags to +half_lags-1

        float ccf = 0.0f;
        int count = 0;

        for (int t = 0; t < ring_size; t++) {
            int t_b = t + tau;
            if (t_b >= 0 && t_b < ring_size) {
                // Use ring buffer with proper wrapping
                int idx_a = (ring_head + t) % ring_size;
                int idx_b = (ring_head + t_b) % ring_size;

                float da = ring_a[base_a + idx_a] - mean_a;
                float db = ring_b[base_b + idx_b] - mean_b;
                ccf += da * db;
                count++;
            }
        }

        // Normalize
        ccf_map[base_ccf + lag_idx] = (count > 0) ? ccf / norm : 0.0f;
    }
}


// ─────────────────────────────────────────────────────────────────────
// KERNEL 4: Update ring buffers with current spike density
// ─────────────────────────────────────────────────────────────────────

extern "C" __global__ void update_ring_buffers(
    float* __restrict__ ring_a,          // [n_regions * ring_size]
    float* __restrict__ ring_b,          // [n_regions * ring_size]
    const float* __restrict__ density_a, // [n_regions] current A density
    const float* __restrict__ density_b, // [n_regions] current B density
    int n_regions,
    int ring_size,
    int ring_head                        // current write position
) {
    int region = blockIdx.x * blockDim.x + threadIdx.x;
    if (region >= n_regions) return;

    int write_idx = region * ring_size + ring_head;
    ring_a[write_idx] = density_a[region];
    ring_b[write_idx] = density_b[region];
}
