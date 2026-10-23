#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>

// ============================================================================================
// CONSTANTS & TYPES
// ============================================================================================
#define KB 0.001987f
typedef curandStatePhilox4_32_10_t RngState;

// AUDIT REQUIREMENT: Enforce size contract
static_assert(sizeof(RngState) == 64, "RngState size mismatch: Host expects 64 bytes");

// ============================================================================================
// VECTOR HELPERS
// ============================================================================================
__device__ inline float3 add(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ inline float3 sub(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ inline float3 scale(float3 a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }

// ============================================================================================
// KERNEL 1: RNG INITIALIZATION (Run Once)
// ============================================================================================
extern "C" __global__ void init_rng_kernel(
    unsigned long long seed,
    RngState* __restrict__ states,
    int num_atoms
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_atoms) return;

    curand_init(seed, idx, 0, &states[idx]);
}

// ============================================================================================
// KERNEL 2: HOLOGRAPHIC STEP (Euler-Maruyama)
// ============================================================================================
extern "C" __global__ void holographic_step_kernel(
    float* __restrict__ atoms_ptr,                // 0: [x,y,z,w]
    const float* __restrict__ anchors_ptr,        // 1: [x,y,z,w]
    float* __restrict__ velocities_ptr,           // 2: [vx,vy,vz,w]
    const float* __restrict__ bias_ptr,           // 3: [bx,by,bz,w]
    int num_atoms,                                // 4
    float dt,                                     // 5
    float friction,                               // 6
    float temp_start,                             // 7
    float temp_end,                               // 8
    float bias_strength,                          // 9
    float spring_k,                               // 10
    RngState* __restrict__ rng_states,            // 11
    int step_idx,                                 // 12
    int annealing_steps                           // 13
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_atoms) return;

    // 1) FLOAT4 MEMORY ACCESS (ABI/stride contract)
    float4* atoms_f4 = (float4*)atoms_ptr;
    const float4* anchors_f4 = (const float4*)anchors_ptr;
    float4* vels_f4 = (float4*)velocities_ptr;
    const float4* bias_f4 = (const float4*)bias_ptr;

    float4 pos_4 = atoms_f4[idx];
    float4 anc_4 = anchors_f4[idx];
    float4 vel_4 = vels_f4[idx];
    float4 bias_4 = bias_f4[idx];

    float3 pos  = make_float3(pos_4.x, pos_4.y, pos_4.z);
    float3 anc  = make_float3(anc_4.x, anc_4.y, anc_4.z);
    float3 vel  = make_float3(vel_4.x, vel_4.y, vel_4.z);
    float3 bias = make_float3(bias_4.x, bias_4.y, bias_4.z);

    // 2) ANNEALING
    float current_temp = temp_end;
    if (annealing_steps > 0) {
        float progress = (float)step_idx / (float)annealing_steps;
        if (progress > 1.0f) progress = 1.0f;
        current_temp = temp_start + (temp_end - temp_start) * progress;
    }

    // 3) DETERMINISTIC FORCES
    float3 diff     = sub(pos, anc);
    float3 f_spring = scale(diff, -spring_k);
    float3 f_bias   = scale(bias, bias_strength);
    float3 f_drag   = scale(vel, -friction);
    float3 f_det    = add(add(f_spring, f_bias), f_drag);

    // 4) STOCHASTIC IMPULSE (Euler–Maruyama)
    RngState local_state = rng_states[idx];

    // Vectorized RNG: 4 normals at once (discard w)
    float4 n4   = curand_normal4(&local_state);
    float3 noise = make_float3(n4.x, n4.y, n4.z);

    rng_states[idx] = local_state;

    // Scaling: sqrt(2 * kB * T * gamma * dt)
    float impulse_scale = sqrtf(2.0f * KB * current_temp * friction * dt);
    float3 noise_impulse = scale(noise, impulse_scale);

    // 5) INTEGRATION
    float3 vel_change = add(scale(f_det, dt), noise_impulse);
    float3 vel_new    = add(vel, vel_change);
    float3 pos_new    = add(pos, scale(vel_new, dt));

    // 6) STORE
    atoms_f4[idx] = make_float4(pos_new.x, pos_new.y, pos_new.z, pos_4.w);
    vels_f4[idx]  = make_float4(vel_new.x, vel_new.y, vel_new.z, vel_4.w);
}