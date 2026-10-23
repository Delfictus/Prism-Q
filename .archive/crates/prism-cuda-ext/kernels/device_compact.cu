// ═══════════════════════════════════════════════════════════════════════
// PRISM-TWIN: Device-Side Spike Compaction + Dual-Destination Write
//
// Eliminates the CPU round-trip that was the #1 coupling latency source:
//   OLD: GPU spike_events → CPU download → CPU repack → CPU upload → GPU ring
//   NEW: GPU spike_events → GPU compact → GPU ring + mapped host RAM (zero PCIe CPU involvement)
//
// DUAL DESTINATION:
//   1. Ring buffer (VRAM) — for real-time threshold coupling between twins
//   2. Exhaust buffer (mapped pinned host RAM) — for training data preservation
//
// The exhaust write goes through PCIe Gen 5 DMA (32 GB/s) without using
// SM ALUs. The ring write stays in VRAM. Both happen in the same kernel.
//
// Every spike is preserved with its channel tag (spike_source):
//   0=LIF, 1=UV, 2=RAF, 3=EFP, 4=LADD, 5=COFIRE
// This is the proprietary training signal for the 109-model ensemble teacher.
//
// Compiled with: nvcc -arch=sm_120 -O3 --use_fast_math
// ═══════════════════════════════════════════════════════════════════════

#include <cstdint>

// Match the ring buffer struct layout from ring_buffer.cu
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

// GpuSpikeEvent byte offsets (verified against fused_engine.rs:225-241)
#define GPU_SPIKE_SIZE 92
#define OFF_TIMESTEP           0
#define OFF_VOXEL_IDX          4
#define OFF_POSITION_X         8
#define OFF_POSITION_Y        12
#define OFF_POSITION_Z        16
#define OFF_INTENSITY         20
#define OFF_SPIKE_SOURCE      60
#define OFF_WAVELENGTH_NM     64
#define OFF_WATER_DENSITY     76
#define OFF_VIB_ENERGY        80
#define OFF_N_NEARBY_EXCITED  84

// ─────────────────────────────────────────────────────────────────────
// HELPER: Extract RingSpikeEvent from raw GpuSpikeEvent bytes
// ─────────────────────────────────────────────────────────────────────

__device__ __forceinline__ void extract_spike(
    const unsigned char* __restrict__ src,
    int spike_idx,
    RingSpikeEvent* __restrict__ out
) {
    int offset = spike_idx * GPU_SPIKE_SIZE;
    const unsigned char* s = &src[offset];
    memcpy(&out->timestep,           s + OFF_TIMESTEP,          4);
    memcpy(&out->voxel_idx,          s + OFF_VOXEL_IDX,         4);
    memcpy(&out->x,                  s + OFF_POSITION_X,        4);
    memcpy(&out->y,                  s + OFF_POSITION_Y,        4);
    memcpy(&out->z,                  s + OFF_POSITION_Z,        4);
    memcpy(&out->intensity,          s + OFF_INTENSITY,          4);
    memcpy(&out->vibrational_energy, s + OFF_VIB_ENERGY,        4);
    memcpy(&out->water_density,      s + OFF_WATER_DENSITY,     4);
    memcpy(&out->n_nearby_excited,   s + OFF_N_NEARBY_EXCITED,  4);
    memcpy(&out->spike_source,       s + OFF_SPIKE_SOURCE,      4);
    memcpy(&out->wavelength_nm,      s + OFF_WAVELENGTH_NM,     4);
    out->pad = 0;
}

// ─────────────────────────────────────────────────────────────────────
// KERNEL: Compact + Dual Push (ring buffer + exhaust)
//
// Each thread handles one spike:
//   1. Reads from physics engine's raw spike buffer (VRAM)
//   2. Extracts 10 fields into compact 48-byte format
//   3. Atomically pushes into VRAM ring buffer (for coupling)
//   4. Atomically pushes into mapped host RAM exhaust buffer (for training)
//
// The exhaust write goes through PCIe DMA automatically —
// the GPU writes to a device pointer that's physically backed
// by system RAM. No cudaMemcpy, no stream sync, no CPU involvement.
// ─────────────────────────────────────────────────────────────────────

extern "C" __global__ void device_compact_and_push(
    // Source: physics engine spike buffer (read-only, VRAM)
    const unsigned char* __restrict__ gpu_spike_buf,
    const int* __restrict__ d_prev_count,
    const int* __restrict__ d_curr_count,

    // Destination 1: VRAM ring buffer (for real-time coupling)
    RingSpikeEvent* __restrict__ ring_buffer,
    unsigned int* __restrict__ ring_head,
    unsigned int* __restrict__ ring_overflow,
    unsigned int ring_capacity,

    // Destination 2: mapped host RAM exhaust buffer (for training data)
    // These pointers are device-accessible but physically in system DDR5.
    // Writes go through PCIe Gen 5 DMA engine (32 GB/s, zero SM cost).
    RingSpikeEvent* __restrict__ exhaust_buffer,   // mapped host RAM
    unsigned int* __restrict__ exhaust_head,         // mapped host RAM (atomicAdd)
    unsigned int exhaust_capacity,                   // total spike slots
    int exhaust_enabled                              // 0 = skip exhaust writes
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int prev = *d_prev_count;
    int curr = *d_curr_count;
    int n_new = curr - prev;

    // Clamp to ring capacity (take most recent spikes if too many)
    int start_idx = (n_new > (int)ring_capacity) ? (curr - (int)ring_capacity) : prev;
    n_new = curr - start_idx;

    if (tid >= n_new) return;

    // Extract spike into compact format (one per thread)
    RingSpikeEvent compacted;
    extract_spike(gpu_spike_buf, start_idx + tid, &compacted);

    // ── Write 1: VRAM ring buffer (for coupling) ──
    {
        unsigned int write_pos = atomicAdd(ring_head, 1);
        if (write_pos - *ring_head >= ring_capacity) {
            atomicAdd(ring_overflow, 1);
        }
        ring_buffer[write_pos % ring_capacity] = compacted;
    }

    // ── Write 2: Mapped host RAM exhaust buffer (for training data) ──
    // This write goes through PCIe DMA. The GPU's DMA controller handles
    // the transfer asynchronously. SM ALUs are NOT involved in the data
    // movement — they just issue the store and move on.
    if (exhaust_enabled) {
        unsigned int exhaust_pos = atomicAdd(exhaust_head, 1);
        // Circular write — if buffer is full, overwrite oldest data.
        // The CPU harvester thread should read fast enough to keep up.
        exhaust_buffer[exhaust_pos % exhaust_capacity] = compacted;
    }
}

// ─────────────────────────────────────────────────────────────────────
// KERNEL: Update prev_count after compaction
// ─────────────────────────────────────────────────────────────────────

extern "C" __global__ void update_prev_spike_count(
    int* __restrict__ d_prev_count,
    const int* __restrict__ d_curr_count
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *d_prev_count = *d_curr_count;
    }
}

// ─────────────────────────────────────────────────────────────────────
// KERNEL: Compute grid size for compact kernel (device-side sizing)
// ─────────────────────────────────────────────────────────────────────

extern "C" __global__ void compute_compact_grid_size(
    const int* __restrict__ d_curr_count,
    const int* __restrict__ d_prev_count,
    unsigned int* __restrict__ d_grid_dim_x,
    unsigned int ring_capacity
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int n_new = *d_curr_count - *d_prev_count;
        if (n_new <= 0) {
            *d_grid_dim_x = 1;
            return;
        }
        if (n_new > (int)ring_capacity) n_new = (int)ring_capacity;
        *d_grid_dim_x = ((unsigned int)n_new + 255) / 256;
    }
}

// ─────────────────────────────────────────────────────────────────────
// KERNEL: Decrement step counter (for CUDA Graph WHILE condition)
//
// This is the LAST kernel in the graph's WHILE body. After all coupling
// work is done for this step, decrement the counter. When it reaches 0,
// the GPU's hardware command processor exits the WHILE loop.
// ─────────────────────────────────────────────────────────────────────

extern "C" __global__ void decrement_step_counter(
    unsigned int* __restrict__ counter
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned int val = *counter;
        if (val > 0) *counter = val - 1;
    }
}
