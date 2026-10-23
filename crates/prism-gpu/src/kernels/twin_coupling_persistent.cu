// ═══════════════════════════════════════════════════════════════════════
// PRISM-TWIN: Persistent Cooperative Coupling Kernel
//
// This kernel runs for the ENTIRE simulation on a dedicated CUDA stream.
// It handles all inter-stream coupling (ring buffer exchange, threshold
// adaptation, recovery) WITHOUT returning to host between steps.
//
// The physics kernels (nhs_amber_fused.cu) are FROZEN and unchanged.
// This kernel synchronizes with them via atomic signal flags:
//
//   Physics stream A:  fused_step → multi_lif → ladd_* → signal(flag_a)
//   Physics stream B:  fused_step → multi_lif → ladd_* → signal(flag_b)
//   Coupling stream:   twin_coupling_persistent() {
//                        for step in 0..total_steps {
//                          spin-wait(flag_a == step, flag_b == step)
//                          device-side spike compaction + ring push
//                          grid.sync()
//                          ring buffer read + threshold adaptation
//                          grid.sync()
//                          periodic threshold recovery
//                          grid.sync()
//                          clear flags
//                          grid.sync()
//                        }
//                      }
//
// Compiled with: nvcc -rdc=true -arch=sm_120 (cooperative groups)
// Launched with: cudaLaunchCooperativeKernel (cudarc: launch_cooperative())
//
// Grid: 168 blocks (84 SMs × 2 blocks/SM on RTX 5080)
//   Blocks 0-83:   handle Stream A coupling
//   Blocks 84-167: handle Stream B coupling
//
// Performance: eliminates ~65μs/step of host-mediated coupling overhead
// (4 kernel launches + 2 H2D memcpy + stream sync per step)
// ═══════════════════════════════════════════════════════════════════════

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// IMPORTANT: Only 2 blocks launched (1 per stream direction).
// This leaves 82+ SMs free for the physics kernels which run concurrently.
// The original design used 168 blocks (full SM occupancy) which STARVED
// the physics kernels — 100% SM util but 1% MBW because all SMs were
// spinning in the coupling kernel's wait loop, not doing physics.
#define COUPLING_BLOCK_SIZE 256
#define NUM_BLOCKS_TOTAL 2
#define NUM_BLOCKS_PER_STREAM 1

// ─────────────────────────────────────────────────────────────────────
// Inline ring buffer operations (from ring_buffer.cu, inlined to avoid
// multi-file cooperative kernel linking complexity)
// ─────────────────────────────────────────────────────────────────────

struct RingSpikeEvent {
    int timestep;
    int voxel_idx;
    float x, y, z;
    float intensity;
    float vibrational_energy;
    float water_density;
    int n_nearby_excited;
    int spike_source;
    float wavelength_nm;
    int pad;
};

// Device-side spike compaction: read from the sealed GpuSpikeEvent
// (96-byte stride),
// extract 10 fields into RingSpikeEvent (48-byte stride).
// Byte offsets verified against fused_engine.rs GpuSpikeEvent struct:
//   timestep=0, voxel_idx=4, position=8, intensity=20,
//   nearby_residues=24(skip), n_residues=56(skip), spike_source=60,
//   wavelength_nm=64, aromatic_type=68(skip), aromatic_residue_id=72(skip),
//   water_density=76, vibrational_energy=80, n_nearby_excited=84,
//   wd_change=88(skip), phase_bits=92(skip)
constexpr int GPU_SPIKE_RECORD_BYTES = 96;
static_assert(GPU_SPIKE_RECORD_BYTES == 96,
              "persistent coupling must consume the sealed 96-byte GpuSpikeEvent packet");
__device__ void compact_spike(
    const unsigned char* __restrict__ gpu_spike_buf,
    int spike_idx,
    RingSpikeEvent* out
) {
    int offset = spike_idx * GPU_SPIKE_RECORD_BYTES;
    const unsigned char* src = &gpu_spike_buf[offset];

    // Use memcpy for type-punning safety (compiler optimizes to loads)
    memcpy(&out->timestep,           src + 0,  4);
    memcpy(&out->voxel_idx,          src + 4,  4);
    memcpy(&out->x,                  src + 8,  4);
    memcpy(&out->y,                  src + 12, 4);
    memcpy(&out->z,                  src + 16, 4);
    memcpy(&out->intensity,          src + 20, 4);
    memcpy(&out->vibrational_energy, src + 80, 4);
    memcpy(&out->water_density,      src + 76, 4);
    memcpy(&out->n_nearby_excited,   src + 84, 4);
    memcpy(&out->spike_source,       src + 60, 4);
    memcpy(&out->wavelength_nm,      src + 64, 4);
    out->pad = 0;
}

// ─────────────────────────────────────────────────────────────────────
// PERSISTENT COUPLING KERNEL
// ─────────────────────────────────────────────────────────────────────

extern "C" __global__ void __launch_bounds__(COUPLING_BLOCK_SIZE, 2)
twin_coupling_persistent(
    // ── Signal flags ──
    volatile unsigned int* __restrict__ flag_a,  // [1] — set by physics stream A
    volatile unsigned int* __restrict__ flag_b,  // [1] — set by physics stream B

    // ── Ring buffer A (A's spikes → read by B's adapter) ──
    RingSpikeEvent* __restrict__ ring_a,          // [ring_capacity]
    unsigned int* __restrict__ ring_a_head,
    unsigned int* __restrict__ ring_a_tail,
    unsigned int* __restrict__ ring_a_overflow,

    // ── Ring buffer B (B's spikes → read by A's adapter) ──
    RingSpikeEvent* __restrict__ ring_b,
    unsigned int* __restrict__ ring_b_head,
    unsigned int* __restrict__ ring_b_tail,
    unsigned int* __restrict__ ring_b_overflow,

    unsigned int ring_capacity,

    // ── Spike buffers (read-only, from physics engines) ──
    const unsigned char* __restrict__ spike_buf_a,   // GpuSpikeEvent[], 96-byte stride
    const int* __restrict__ spike_count_a,            // [1] — current spike count
    const unsigned char* __restrict__ spike_buf_b,
    const int* __restrict__ spike_count_b,

    // ── Threshold buffers (A's thresholds modified by B's evidence, and vice versa) ──
    float* __restrict__ thresh_a,      // [n_voxels_total] — A's neuron thresholds
    const float* __restrict__ base_a,  // [n_voxels_total] — A's initial thresholds
    float* __restrict__ thresh_b,
    const float* __restrict__ base_b,

    // ── Grid geometry (for threshold voxel mapping) ──
    int n_voxels_x, int n_voxels_y, int n_voxels_z,
    float grid_origin_x, float grid_origin_y, float grid_origin_z,
    float voxel_size,

    // ── Coupling parameters ──
    float sensitivity_boost,
    float max_reduction_fraction,
    float decay_constant,
    float recovery_rate,
    unsigned int recovery_interval,   // apply recovery every N steps

    // ── Simulation control ──
    unsigned int total_steps
) {
    cg::grid_group grid = cg::this_grid();

    // Block 0 = stream A coupling, Block 1 = stream B coupling
    const bool is_stream_b_block = (blockIdx.x >= NUM_BLOCKS_PER_STREAM);
    const int stream_block = 0;  // Only 1 block per stream
    const int n_voxels_total = n_voxels_x * n_voxels_y * n_voxels_z;

    // Track previous spike counts to compute deltas
    int prev_spike_count_a = 0;
    int prev_spike_count_b = 0;

    // ═══════════════════════════════════════════════════════════════════
    // MAIN PERSISTENT LOOP — runs for entire simulation
    // ═══════════════════════════════════════════════════════════════════
    for (unsigned int step = 1; step <= total_steps; step++) {

        // ─── PHASE 1: Spin-wait for both physics streams to complete ───
        // Block 0 spins; all others wait at grid.sync()
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            // Spin-wait with nanosleep to avoid power waste.
            // Typical wait: ~2ms (one fused step time)
            while (*flag_a < step || *flag_b < step) {
                __nanosleep(100);  // 100ns between polls
            }
        }
        grid.sync();  // All blocks now know both streams completed this step

        // ─── PHASE 2: Device-side spike compaction + ring buffer push ───
        // Each block handles a chunk of new spikes from its assigned stream.
        // Stream A blocks (0-83) compact A's spikes → ring_a
        // Stream B blocks (84-167) compact B's spikes → ring_b
        {
            int curr_count, prev_count;
            RingSpikeEvent* ring;
            unsigned int* ring_head;
            unsigned int* ring_overflow;
            const unsigned char* spike_buf;

            if (!is_stream_b_block) {
                curr_count = *spike_count_a;
                prev_count = prev_spike_count_a;
                ring = ring_a;
                ring_head = ring_a_head;
                ring_overflow = ring_a_overflow;
                spike_buf = spike_buf_a;
            } else {
                curr_count = *spike_count_b;
                prev_count = prev_spike_count_b;
                ring = ring_b;
                ring_head = ring_b_head;
                ring_overflow = ring_b_overflow;
                spike_buf = spike_buf_b;
            }

            int n_new = curr_count - prev_count;
            // Clamp to ring capacity to avoid overwhelming the ring
            int start_idx = (n_new > (int)ring_capacity) ? (curr_count - (int)ring_capacity) : prev_count;
            n_new = curr_count - start_idx;

            // Single block handles all new spikes (256 threads stride across them)
            for (int i = (int)threadIdx.x; i < n_new; i += COUPLING_BLOCK_SIZE) {
                RingSpikeEvent compacted;
                compact_spike(spike_buf, start_idx + i, &compacted);

                unsigned int write_pos = atomicAdd(ring_head, 1);
                unsigned int current_tail = *ring_a_tail;  // approximate
                if (write_pos - current_tail >= ring_capacity) {
                    atomicAdd(ring_overflow, 1);
                }
                ring[write_pos % ring_capacity] = compacted;
            }
        }

        // Update prev counts (block 0 does this, visible after next grid.sync)
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            prev_spike_count_a = *spike_count_a;
        }
        if (blockIdx.x == NUM_BLOCKS_PER_STREAM && threadIdx.x == 0) {
            prev_spike_count_b = *spike_count_b;
        }

        grid.sync();  // Ring buffer push complete, all spikes visible

        // ─── PHASE 3: Read ring buffer + adapt thresholds ───
        // Stream A blocks read ring_b (B's evidence) → adapt A's thresholds
        // Stream B blocks read ring_a (A's evidence) → adapt B's thresholds
        //
        // Each block handles a chunk of voxels for threshold adaptation.
        // This is the INTERFEROMETRIC COUPLING — detector sensitivity in
        // one stream steered by evidence from the other stream.
        {
            // Determine which ring to read and which thresholds to write
            RingSpikeEvent* src_ring;
            unsigned int* src_head;
            unsigned int* src_tail;
            float* dst_thresh;
            const float* dst_base;

            if (!is_stream_b_block) {
                // A reads B's ring → adapt A's thresholds
                src_ring = ring_b;
                src_head = ring_b_head;
                src_tail = ring_b_tail;
                dst_thresh = thresh_a;
                dst_base = base_a;
            } else {
                // B reads A's ring → adapt B's thresholds
                src_ring = ring_a;
                src_head = ring_a_head;
                src_tail = ring_a_tail;
                dst_thresh = thresh_b;
                dst_base = base_b;
            }

            // Block 0 of each stream group processes the ring buffer
            // (same single-threaded read pattern as ring_buffer_read_and_adapt)
            if (stream_block == 0 && threadIdx.x == 0) {
                unsigned int head = *src_head;
                unsigned int tail = *src_tail;
                unsigned int n_to_process = head - tail;
                if (n_to_process > ring_capacity) {
                    tail = head - ring_capacity;
                    n_to_process = ring_capacity;
                }

                for (unsigned int i = 0; i < n_to_process; i++) {
                    unsigned int ring_idx = (tail + i) % ring_capacity;
                    RingSpikeEvent spike = src_ring[ring_idx];

                    int vx = (int)((spike.x - grid_origin_x) / voxel_size);
                    int vy = (int)((spike.y - grid_origin_y) / voxel_size);
                    int vz = (int)((spike.z - grid_origin_z) / voxel_size);

                    if (vx < 0 || vx >= n_voxels_x) continue;
                    if (vy < 0 || vy >= n_voxels_y) continue;
                    if (vz < 0 || vz >= n_voxels_z) continue;

                    int voxel_idx = vx + vy * n_voxels_x + vz * n_voxels_x * n_voxels_y;

                    float age = (float)((int)step - spike.timestep);
                    if (age < 0.0f) age = 0.0f;
                    float time_weight = expf(-age / decay_constant);
                    float boost = sensitivity_boost * spike.intensity * time_weight;

                    float base_val = dst_base[voxel_idx];
                    float floor_val = base_val * (1.0f - max_reduction_fraction);
                    float current = dst_thresh[voxel_idx];
                    float new_val = fmaxf(floor_val, current - boost);
                    dst_thresh[voxel_idx] = new_val;
                }

                *src_tail = head;  // Mark all as read
            }
        }

        grid.sync();  // Threshold adaptation complete

        // ─── PHASE 4: Periodic threshold recovery ───
        // Every recovery_interval steps, nudge thresholds back toward baseline
        if (step % recovery_interval == 0) {
            float* my_thresh = is_stream_b_block ? thresh_b : thresh_a;
            const float* my_base = is_stream_b_block ? base_b : base_a;

            // Single block strides across all voxels
            for (int v = (int)threadIdx.x; v < n_voxels_total; v += COUPLING_BLOCK_SIZE) {
                float current = my_thresh[v];
                float base = my_base[v];
                my_thresh[v] = current + recovery_rate * (base - current);
            }
        }

        grid.sync();  // Recovery complete

        // ─── PHASE 5: Clear signal flags for next step ───
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            atomicExch((unsigned int*)flag_a, 0);
            atomicExch((unsigned int*)flag_b, 0);
            __threadfence_system();
        }

        grid.sync();  // Flags cleared, ready for next step
    }
}
