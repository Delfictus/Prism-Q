#include <optix.h>
#include <optix_device.h>

struct RtClusteringParams {
    OptixTraversableHandle traversable;
    float3* event_positions;
    unsigned int num_events;
    float epsilon_sq;
    float ray_tmax;
    unsigned int rays_per_event;
    unsigned int max_neighbors;
    unsigned int _pad;
    unsigned int* neighbor_list;
    unsigned int* neighbor_count;
    int* parent;
    unsigned int* num_clusters;
};

extern "C" { __constant__ RtClusteringParams params; }

extern "C" __global__ void __raygen__find_neighbors() {
    const uint3 idx = optixGetLaunchIndex();
    const unsigned int source_event = idx.x;
    if (source_event >= params.num_events) return;

    float3 origin = params.event_positions[source_event];
    const float golden_ratio = 1.6180340f;
    const float pi = 3.1415927f;

    for (unsigned int i = 0; i < params.rays_per_event; i++) {
        float t = (float)i / (float)(params.rays_per_event - 1);
        float z = 1.0f - (2.0f * t);
        float radius = sqrtf(fmaxf(0.0f, 1.0f - z * z));
        float theta = 2.0f * pi * (float)i / golden_ratio;
        float3 direction = make_float3(radius * cosf(theta), radius * sinf(theta), z);

        unsigned int p0 = source_event;
        unsigned int p1 = 0;

        optixTrace(
            params.traversable,
            origin, direction,
            0.0f, params.ray_tmax, 0.0f,
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_NONE,
            0, 1, 0,
            p0, p1
        );
    }
}

extern "C" __global__ void __closesthit__find_neighbors() {
    unsigned int source_event = optixGetPayload_0();
    unsigned int target_event = optixGetInstanceId();

    if (source_event == target_event) return;

    float3 p1 = params.event_positions[source_event];
    float3 p2 = params.event_positions[target_event];
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    float dz = p1.z - p2.z;
    float dist_sq = dx*dx + dy*dy + dz*dz;

    if (dist_sq < params.epsilon_sq) {
        unsigned int slot = atomicAdd(&params.neighbor_count[source_event], 1);
        if (slot < params.max_neighbors) {
            params.neighbor_list[source_event * params.max_neighbors + slot] = target_event;
        }
    }
}

extern "C" __global__ void __miss__find_neighbors() {}
