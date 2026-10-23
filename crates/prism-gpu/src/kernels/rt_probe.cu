// [STAGE-2-RT-PROBE] OptiX Ray Tracing Kernel for Cryptic Site Detection
//
// PRISM4D RT Probe Engine - Spatial sensing via hardware ray tracing
// Target: NVIDIA RTX 5080 (84 RT cores, sm_120)
//
// Uses OptiX 9.1 built-in sphere primitives for optimal performance.
// Detects void formation, solvation disruption, and aromatic LIF proximity.

#include <optix.h>
#include <cuda_runtime.h>

// ═══════════════════════════════════════════════════════════════════════════════
// Launch Parameters (passed from host via optixLaunch)
// ═══════════════════════════════════════════════════════════════════════════════

struct RtProbeLaunchParams {
    // BVH traversable handle
    OptixTraversableHandle traversable;

    // Input: Probe configuration
    float3* probe_origins;          // [attention_points] probe positions
    unsigned int num_probes;        // Number of probe points
    unsigned int rays_per_probe;    // Rays per probe point (e.g., 256)
    float max_distance;             // Maximum ray travel distance (Å)

    // Input: Aromatic center positions for LIF tracking
    float3* aromatic_centers;       // [num_aromatics] ring centers
    unsigned int num_aromatics;     // Number of aromatic rings
    float aromatic_lif_radius;      // LIF interaction radius (Å)

    // Output buffers
    float* hit_distances;           // [num_probes * rays_per_probe] hit distances
    int* hit_atom_ids;              // [num_probes * rays_per_probe] hit atom IDs (-1 for miss)
    unsigned int* void_flags;       // [num_probes] 1 if void detected
    float* solvation_variance;      // [num_probes] variance in hit distances
    unsigned int* aromatic_counts;  // [num_probes] aromatics within LIF radius

    // Current simulation state
    int timestep;
    float temperature;
};

extern "C" {
    __constant__ RtProbeLaunchParams params;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Ray Payload - Per-ray state passed between programs
// ═══════════════════════════════════════════════════════════════════════════════

struct RayPayload {
    float hit_distance;     // Distance to closest hit (-1 if miss)
    int hit_atom_id;        // Index of hit atom (-1 if miss)
    float3 hit_normal;      // Surface normal at hit point
};

// ═══════════════════════════════════════════════════════════════════════════════
// Utility Functions
// ═══════════════════════════════════════════════════════════════════════════════

// Generate uniform sphere direction using golden spiral
__device__ __forceinline__ float3 sphere_direction(unsigned int idx, unsigned int total) {
    // Golden ratio for uniform distribution
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

// Compute variance of hit distances for a probe
__device__ float compute_variance(const float* distances, int count, float mean) {
    float sum_sq_diff = 0.0f;
    int valid_count = 0;

    for (int i = 0; i < count; i++) {
        if (distances[i] > 0.0f) {
            float diff = distances[i] - mean;
            sum_sq_diff += diff * diff;
            valid_count++;
        }
    }

    return (valid_count > 0) ? sum_sq_diff / valid_count : 0.0f;
}

// Count aromatics within LIF radius of a point
__device__ unsigned int count_nearby_aromatics(float3 point) {
    unsigned int count = 0;
    float radius_sq = params.aromatic_lif_radius * params.aromatic_lif_radius;

    for (unsigned int i = 0; i < params.num_aromatics; i++) {
        float3 center = params.aromatic_centers[i];
        float dx = point.x - center.x;
        float dy = point.y - center.y;
        float dz = point.z - center.z;
        float dist_sq = dx*dx + dy*dy + dz*dz;

        if (dist_sq <= radius_sq) {
            count++;
        }
    }

    return count;
}

// ═══════════════════════════════════════════════════════════════════════════════
// RAY GENERATION PROGRAM
// Entry point: One thread per ray
// ═══════════════════════════════════════════════════════════════════════════════

extern "C" __global__ void __raygen__rt_probe() {
    // Get launch indices
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // Calculate probe and ray indices
    // Layout: dim.x = rays_per_probe, dim.y = num_probes
    unsigned int ray_idx = idx.x;
    unsigned int probe_idx = idx.y;

    if (probe_idx >= params.num_probes || ray_idx >= params.rays_per_probe) {
        return;
    }

    // Get probe origin
    float3 origin = params.probe_origins[probe_idx];

    // Generate ray direction (uniform sphere sampling)
    float3 direction = sphere_direction(ray_idx, params.rays_per_probe);

    // Initialize payload
    RayPayload payload;
    payload.hit_distance = -1.0f;
    payload.hit_atom_id = -1;
    payload.hit_normal = make_float3(0.0f, 0.0f, 0.0f);

    // Pack payload into registers (2 x uint32 for simple payload)
    unsigned int p0, p1;
    p0 = __float_as_uint(payload.hit_distance);
    p1 = payload.hit_atom_id;

    // Trace ray
    optixTrace(
        params.traversable,
        origin,
        direction,
        0.001f,                          // tmin (avoid self-intersection)
        params.max_distance,             // tmax
        0.0f,                            // rayTime (no motion blur)
        OptixVisibilityMask(255),        // visibilityMask (all visible)
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,   // rayFlags (closest hit only)
        0,                               // SBT offset
        1,                               // SBT stride
        0,                               // missSBTIndex
        p0, p1                           // payload registers
    );

    // Unpack payload
    payload.hit_distance = __uint_as_float(p0);
    payload.hit_atom_id = (int)p1;

    // Store results
    unsigned int result_idx = probe_idx * params.rays_per_probe + ray_idx;
    params.hit_distances[result_idx] = payload.hit_distance;
    params.hit_atom_ids[result_idx] = payload.hit_atom_id;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CLOSEST HIT PROGRAM
// Called when ray intersects a sphere (atom)
// ═══════════════════════════════════════════════════════════════════════════════

extern "C" __global__ void __closesthit__rt_probe() {
    // Get hit distance
    float t_hit = optixGetRayTmax();

    // Get primitive (atom) index
    unsigned int prim_idx = optixGetPrimitiveIndex();

    // Compute hit point and normal for sphere
    float3 ray_origin = optixGetWorldRayOrigin();
    float3 ray_dir = optixGetWorldRayDirection();
    float3 hit_point = make_float3(
        ray_origin.x + t_hit * ray_dir.x,
        ray_origin.y + t_hit * ray_dir.y,
        ray_origin.z + t_hit * ray_dir.z
    );

    // For sphere primitive, normal points from center to hit point
    // (OptiX provides this via attributes for built-in spheres)
    float3 normal = make_float3(
        optixGetAttribute_0(),
        optixGetAttribute_1(),
        optixGetAttribute_2()
    );

    // Pack payload
    optixSetPayload_0(__float_as_uint(t_hit));
    optixSetPayload_1(prim_idx);
}

// ═══════════════════════════════════════════════════════════════════════════════
// MISS PROGRAM
// Called when ray doesn't hit anything (void detection!)
// ═══════════════════════════════════════════════════════════════════════════════

extern "C" __global__ void __miss__rt_probe() {
    // Set payload to indicate miss
    // hit_distance = -1.0 indicates void (no atom hit)
    optixSetPayload_0(__float_as_uint(-1.0f));
    optixSetPayload_1((unsigned int)(-1));  // No atom hit
}

// ═══════════════════════════════════════════════════════════════════════════════
// POST-PROCESSING KERNEL
// Compute per-probe statistics after all rays complete
// ═══════════════════════════════════════════════════════════════════════════════

extern "C" __global__ void rt_probe_postprocess(
    float* hit_distances,           // [num_probes * rays_per_probe]
    int* hit_atom_ids,              // [num_probes * rays_per_probe]
    float3* probe_origins,          // [num_probes]
    float3* aromatic_centers,       // [num_aromatics]
    unsigned int num_probes,
    unsigned int rays_per_probe,
    unsigned int num_aromatics,
    float aromatic_lif_radius,
    float void_threshold,           // Fraction of misses to declare void
    // Outputs
    unsigned int* void_flags,       // [num_probes]
    float* solvation_variance,      // [num_probes]
    unsigned int* aromatic_counts,  // [num_probes]
    float* avg_hit_distances        // [num_probes]
) {
    unsigned int probe_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (probe_idx >= num_probes) return;

    // Compute statistics for this probe
    unsigned int base_idx = probe_idx * rays_per_probe;

    float sum_distance = 0.0f;
    unsigned int hit_count = 0;
    unsigned int miss_count = 0;

    // First pass: compute mean and count hits/misses
    for (unsigned int i = 0; i < rays_per_probe; i++) {
        float dist = hit_distances[base_idx + i];
        if (dist > 0.0f) {
            sum_distance += dist;
            hit_count++;
        } else {
            miss_count++;
        }
    }

    float mean_distance = (hit_count > 0) ? sum_distance / hit_count : 0.0f;
    avg_hit_distances[probe_idx] = mean_distance;

    // Void detection: high fraction of misses indicates void
    float miss_fraction = (float)miss_count / (float)rays_per_probe;
    void_flags[probe_idx] = (miss_fraction >= void_threshold) ? 1 : 0;

    // Second pass: compute variance (solvation disruption indicator)
    float sum_sq_diff = 0.0f;
    for (unsigned int i = 0; i < rays_per_probe; i++) {
        float dist = hit_distances[base_idx + i];
        if (dist > 0.0f) {
            float diff = dist - mean_distance;
            sum_sq_diff += diff * diff;
        }
    }
    solvation_variance[probe_idx] = (hit_count > 1) ? sum_sq_diff / (hit_count - 1) : 0.0f;

    // Count aromatics near probe (for LIF correlation)
    float3 probe_pos = probe_origins[probe_idx];
    float radius_sq = aromatic_lif_radius * aromatic_lif_radius;
    unsigned int aromatic_count = 0;

    for (unsigned int i = 0; i < num_aromatics; i++) {
        float3 center = aromatic_centers[i];
        float dx = probe_pos.x - center.x;
        float dy = probe_pos.y - center.y;
        float dz = probe_pos.z - center.z;
        float dist_sq = dx*dx + dy*dy + dz*dz;

        if (dist_sq <= radius_sq) {
            aromatic_count++;
        }
    }
    aromatic_counts[probe_idx] = aromatic_count;
}

// ═══════════════════════════════════════════════════════════════════════════════
// ATTENTION POINT SELECTION KERNEL
// Select probe positions based on aromatic centers + random sampling
// ═══════════════════════════════════════════════════════════════════════════════

extern "C" __global__ void select_attention_points(
    float3* atom_positions,         // [num_atoms] all atom positions
    float3* aromatic_centers,       // [num_aromatics] aromatic ring centers
    unsigned int num_atoms,
    unsigned int num_aromatics,
    unsigned int num_probes,        // Total probes to select
    unsigned int seed,              // Random seed
    // Output
    float3* probe_origins           // [num_probes] selected probe positions
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_probes) return;

    // Strategy:
    // - First 50%: Near aromatic centers (cryptic site indicators)
    // - Next 30%: Near protein surface (random atoms)
    // - Last 20%: Grid sampling for coverage

    unsigned int aromatic_probes = num_probes / 2;
    unsigned int surface_probes = num_probes * 3 / 10;

    float3 result;

    if (idx < aromatic_probes && num_aromatics > 0) {
        // Near aromatic center
        unsigned int aromatic_idx = idx % num_aromatics;
        result = aromatic_centers[aromatic_idx];

        // Add small random offset (1-3 Å)
        unsigned int hash = (idx * 2654435761u) ^ seed;
        float offset_x = ((hash & 0xFF) / 255.0f - 0.5f) * 3.0f;
        float offset_y = (((hash >> 8) & 0xFF) / 255.0f - 0.5f) * 3.0f;
        float offset_z = (((hash >> 16) & 0xFF) / 255.0f - 0.5f) * 3.0f;

        result.x += offset_x;
        result.y += offset_y;
        result.z += offset_z;
    }
    else if (idx < aromatic_probes + surface_probes) {
        // Near random atom (surface sampling)
        unsigned int hash = ((idx - aromatic_probes) * 1664525u + 1013904223u) ^ seed;
        unsigned int atom_idx = hash % num_atoms;
        result = atom_positions[atom_idx];

        // Offset outward from protein (5-8 Å)
        float offset = 5.0f + (((hash >> 8) & 0xFF) / 255.0f) * 3.0f;
        float norm = sqrtf(result.x*result.x + result.y*result.y + result.z*result.z);
        if (norm > 0.001f) {
            result.x += (result.x / norm) * offset;
            result.y += (result.y / norm) * offset;
            result.z += (result.z / norm) * offset;
        }
    }
    else {
        // Grid sampling for coverage
        unsigned int grid_idx = idx - aromatic_probes - surface_probes;

        // Simple grid based on protein bounds (computed elsewhere)
        // For now, use random position near center
        unsigned int hash = (grid_idx * 3141592653u) ^ seed;
        result.x = ((hash & 0xFFFF) / 65535.0f - 0.5f) * 100.0f;
        result.y = (((hash >> 16) & 0xFFFF) / 65535.0f - 0.5f) * 100.0f;
        result.z = ((hash * 2654435761u) / 4294967295.0f - 0.5f) * 100.0f;
    }

    probe_origins[idx] = result;
}
