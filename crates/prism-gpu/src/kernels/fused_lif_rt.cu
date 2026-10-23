// [FUSED-LIF-RT] Fused Neuromorphic-Spatial OptiX Pipeline
//
// Collapses the 2-pass sequence (CPU LIF step → GPU RT clustering) into a
// single OptiX launch:
//   raygen: read water_density → compute membrane potential → fire rays
//   closesthit: read neighbor density → atomic increment neighbor_count
//   miss: mark voxel as isolated candidate
//
// Post-traversal CUDA kernel (fused_lif_rt_cuda.cu) handles:
//   spike emission + confidence classification + STDP threshold update
//
// Target: NVIDIA RTX 5080 (84 RT cores, sm_120)

#include <optix.h>
#include <cuda_runtime.h>

// ═══════════════════════════════════════════════════════════════════════════════
// GPU Spike Event (must match GpuSpikeEvent in fused_engine.rs exactly)
// ═══════════════════════════════════════════════════════════════════════════════

struct SpikeEventGpu {
    int timestep;
    int voxel_idx;
    float position[3];
    float intensity;
    int nearby_residues[8];
    int n_residues;
    int spike_source;           // 1=UV, 2=LIF
    float wavelength_nm;
    int aromatic_type;          // 0=TRP, 1=TYR, 2=PHE, 3=SS, -1=none
    int aromatic_residue_id;
    float water_density;
    float vibrational_energy;
    int n_nearby_excited;
    int ramp_phase;             // 1-5
};

// ═══════════════════════════════════════════════════════════════════════════════
// Launch Parameters (must match FusedLifRtParams in rt_clustering.rs exactly)
// ═══════════════════════════════════════════════════════════════════════════════

struct FusedLifRtParams {
    OptixTraversableHandle traversable;     // BVH handle (8 bytes)

    // Input fields
    float* water_density;                   // voxel density field from MD (8 bytes)
    float* voxel_positions;                 // x,y,z per voxel (8 bytes)
    float* threshold_adaptation;            // per-voxel adaptive threshold (STDP) (8 bytes)

    // Output fields
    unsigned int* neighbor_counts;          // output: neighbors near threshold (8 bytes)
    float* neighbor_density_sum;            // output: sum of neighbor densities (8 bytes)
    SpikeEventGpu* spike_output;            // output: emitted spikes (8 bytes)
    unsigned int* spike_count;              // atomic counter (8 bytes)

    // Scalar parameters
    float threshold_lo;                     // pre-spike threshold (4 bytes)
    float threshold_hi;                     // spike emission threshold (4 bytes)
    float epsilon;                          // neighborhood radius Å (4 bytes)
    float baseline_density;                 // bulk water density (4 bytes)
    unsigned int min_neighbors;             // K for synchrony (4 bytes)
    unsigned int n_voxels;                  // (4 bytes)
    unsigned int current_timestep;          // (4 bytes)
    float current_temperature;              // (4 bytes)
    int ramp_phase;                         // 1-5 (4 bytes)
    unsigned int max_spikes;                // max spike buffer size (4 bytes)
    unsigned int rays_per_voxel;            // rays to cast per voxel (4 bytes)
    int pad0;                               // alignment padding (4 bytes)
};

extern "C" {
    __constant__ FusedLifRtParams params;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Utility: Fibonacci Sphere Direction (reused from rt_clustering.cu)
// ═══════════════════════════════════════════════════════════════════════════════

__device__ __forceinline__ float3 fibonacci_sphere_direction(unsigned int idx, unsigned int total) {
    const float golden_ratio = 1.618033988749895f;
    const float pi = 3.14159265358979f;

    float theta = 2.0f * pi * idx / golden_ratio;
    float phi = acosf(1.0f - 2.0f * (idx + 0.5f) / total);

    float sin_phi = sinf(phi);
    return make_float3(
        sin_phi * cosf(theta),
        sin_phi * sinf(theta),
        cosf(phi)
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// RAYGEN: One thread per voxel
//
// Read water_density → compute membrane potential V
// If V > threshold_lo: cast rays to find near-threshold neighbors
// ═══════════════════════════════════════════════════════════════════════════════

extern "C" __global__ void __raygen__fused_lif() {
    const uint3 idx = optixGetLaunchIndex();
    unsigned int voxel_id = idx.x;
    unsigned int ray_id = idx.y;

    if (voxel_id >= params.n_voxels) return;
    if (ray_id >= params.rays_per_voxel) return;

    // Read water density for this voxel
    float density = params.water_density[voxel_id];

    // Compute membrane potential: deviation from baseline
    // Low density = dewetting = high potential for spike
    float V = params.baseline_density - density;

    // Apply per-voxel STDP threshold adaptation
    float adapted_threshold = params.threshold_lo + params.threshold_adaptation[voxel_id];

    // Only cast rays if this voxel is a pre-spike candidate
    if (V <= adapted_threshold) return;

    // Get voxel position
    float3 origin = make_float3(
        params.voxel_positions[voxel_id * 3],
        params.voxel_positions[voxel_id * 3 + 1],
        params.voxel_positions[voxel_id * 3 + 2]
    );

    // Generate ray direction (Fibonacci sphere for uniform coverage)
    float3 direction = fibonacci_sphere_direction(ray_id, params.rays_per_voxel);

    // Payload:
    //   p0 = voxel_id (origin voxel for closesthit to reference)
    //   p1 = unused (reserved)
    unsigned int p0 = voxel_id;
    unsigned int p1 = 0;

    optixTrace(
        params.traversable,
        origin,
        direction,
        0.001f,                         // tmin (avoid self-intersection)
        params.epsilon,                 // tmax = neighborhood radius
        0.0f,                           // rayTime
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0,                              // SBT offset
        1,                              // SBT stride
        0,                              // missSBTIndex
        p0, p1
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// CLOSESTHIT: Fires when ray hits a neighbor voxel within epsilon
//
// Check if neighbor is also near threshold → lateral cooperation
// Atomically increment neighbor_count + accumulate neighbor density
// ═══════════════════════════════════════════════════════════════════════════════

extern "C" __global__ void __closesthit__fused_lif() {
    unsigned int hit_voxel_id = optixGetPrimitiveIndex();
    unsigned int origin_voxel = optixGetPayload_0();

    // Skip self-hits
    if (hit_voxel_id == origin_voxel) return;
    if (hit_voxel_id >= params.n_voxels) return;

    // Read neighbor's water density
    float neighbor_density = params.water_density[hit_voxel_id];

    // Check if neighbor is also near dewetting threshold
    float neighbor_V = params.baseline_density - neighbor_density;
    float neighbor_adapted = params.threshold_lo + params.threshold_adaptation[hit_voxel_id];

    if (neighbor_V > neighbor_adapted * 0.5f) {
        // Neighbor is also approaching threshold — cooperative event
        // This IS lateral inhibition: spike decision has neighbor context
        atomicAdd(&params.neighbor_counts[origin_voxel], 1);
        atomicAdd(&params.neighbor_density_sum[origin_voxel], neighbor_V);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MISS: Voxel ray found no near-threshold neighbors in this direction
//
// No action needed — neighbor_counts stays at 0 for isolated voxels
// ═══════════════════════════════════════════════════════════════════════════════

extern "C" __global__ void __miss__fused_lif() {
    // No action needed for miss
    // An isolated voxel will have neighbor_counts[voxel_id] == 0
    // → classified as TEMPORAL_ARTIFACT or THERMAL_NOISE by post-traversal kernel
}
