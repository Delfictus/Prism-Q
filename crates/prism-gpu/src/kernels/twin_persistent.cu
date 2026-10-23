// ═══════════════════════════════════════════════════════════════════════
// PRISM-TWIN PERSISTENT KERNEL — Level 4 Cooperative Groups
//
// Gate L4-1: Skeleton kernel that verifies grid.sync() works with
// 168 blocks on RTX 5080 (84 SMs × 2 blocks/SM).
// No physics. Just cooperative synchronization proof.
//
// Compiled with: nvcc -rdc=true -arch=sm_120
// Launched with: cudaLaunchCooperativeKernel()
// ═══════════════════════════════════════════════════════════════════════

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Block roles (84 blocks per stream = 168 total)
#define NUM_BLOCKS_PER_STREAM 84
#define ROLE_FORCES      0  // blocks 0-41
#define ROLE_INTEGRATION 1  // blocks 42-55
#define ROLE_OSCILLATORS 2  // blocks 56-69
#define ROLE_SPIKE_MGMT  3  // blocks 70-79
#define ROLE_PHASE_CTRL  4  // blocks 80-83

__device__ int get_block_role(int stream_block) {
    if (stream_block < 42)  return ROLE_FORCES;
    if (stream_block < 56)  return ROLE_INTEGRATION;
    if (stream_block < 70)  return ROLE_OSCILLATORS;
    if (stream_block < 80)  return ROLE_SPIKE_MGMT;
    return ROLE_PHASE_CTRL;
}


// ─────────────────────────────────────────────────────────────────────
// L4-1: SKELETON KERNEL — verify cooperative groups grid.sync()
// ─────────────────────────────────────────────────────────────────────

extern "C" __global__ void __launch_bounds__(256, 2)
twin_persistent_skeleton(
    volatile unsigned int* global_step,     // [1] — host reads this to monitor progress
    unsigned int total_steps,               // how many steps to run
    unsigned int* block_reached,            // [n_blocks] — each block writes its max step
    unsigned int* block_roles,              // [n_blocks] — each block writes its role
    float* dummy_output                     // [n_blocks] — prevent compiler optimization
) {
    cg::grid_group grid = cg::this_grid();

    const bool is_stream_a = (blockIdx.x < NUM_BLOCKS_PER_STREAM);
    const int stream_block = is_stream_a ?
        blockIdx.x : (blockIdx.x - NUM_BLOCKS_PER_STREAM);
    const int role = get_block_role(stream_block);

    // Record role (once)
    if (threadIdx.x == 0) {
        block_roles[blockIdx.x] = (unsigned int)role;
    }

    float accumulator = 0.0f;

    // ═══════════════════════════════════════════════════
    // MAIN LOOP: simulates the persistent kernel structure
    // 3 grid.sync() per step (matching the real kernel)
    // ═══════════════════════════════════════════════════
    for (unsigned int step = 0; step < total_steps; step++) {

        // ─── Stage 1: "Force computation" (simulated work) ───
        // Each role does different amount of work (realistic load balance)
        if (role == ROLE_FORCES) {
            accumulator += sinf((float)(step + blockIdx.x * 256 + threadIdx.x));
        } else if (role == ROLE_INTEGRATION) {
            accumulator += cosf((float)(step + blockIdx.x * 256 + threadIdx.x));
        }

        // ─── GRID SYNC 1: forces complete ───
        grid.sync();

        // ─── Stage 2: "Integration" (simulated) ───
        if (role == ROLE_OSCILLATORS || role == ROLE_SPIKE_MGMT) {
            accumulator += (float)step * 0.001f;
        }

        // ─── GRID SYNC 2: integration complete ───
        grid.sync();

        // ─── Stage 3: "Oscillator update" + housekeeping ───
        if (threadIdx.x == 0) {
            block_reached[blockIdx.x] = step;
        }

        if (blockIdx.x == 0 && threadIdx.x == 0) {
            *global_step = step;
        }

        // ─── GRID SYNC 3: end of timestep ───
        grid.sync();
    }

    // Write accumulator to prevent compiler from optimizing away the work
    if (threadIdx.x == 0) {
        dummy_output[blockIdx.x] = accumulator;
    }
}


// ─────────────────────────────────────────────────────────────────────
// L4-1 helper: query maximum cooperative blocks
// Called from host to determine launch configuration
// ─────────────────────────────────────────────────────────────────────

// NOTE: The host-side query uses cudaOccupancyMaxActiveBlocksPerMultiprocessor
// This is called from Rust, not from device code.
// The kernel above is compiled with __launch_bounds__(256, 2) which hints
// 256 threads per block, 2 blocks per SM = 168 blocks total on 84 SMs.
