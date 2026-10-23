/**
 * @file nhs_spike_detect.cu
 * @brief GPU-accelerated neuromorphic spike detection for the NHS pipeline
 *
 * Replaces the CPU detect_spikes_from_positions() function in nhs_rt_full.rs
 * with a fully parallel CUDA implementation. One thread per aromatic atom
 * computes displacement between current and previous positions, then counts
 * nearby atoms to identify isolated (pocket-like) aromatic movements as
 * spike events.
 *
 * ALGORITHM (from nhs_rt_full.rs):
 *   1. For each aromatic atom, compute displacement from previous frame
 *   2. If displacement > threshold (0.5 A):
 *      a. Count all atoms within proximity_threshold (6.0 A)
 *      b. If nearby_count < max_nearby (20), this is a pocket-like region
 *      c. intensity = displacement * (max_nearby - nearby_count) / max_nearby
 *      d. Emit spike event
 *
 * OUTPUT STRUCT: GpuSpikeEvent (60 bytes, packed, matches Rust repr(C, packed))
 *   i32  timestep
 *   i32  voxel_idx          (always 0 for position-based detection)
 *   f32  position[3]        (current atom position)
 *   f32  intensity           (displacement-weighted pocket isolation score)
 *   i32  nearby_residues[8] (zeroed - not tracked here)
 *   i32  n_residues          (0 - not tracked here)
 *
 * TARGET: SM86+ (Ampere/Ada/Blackwell SM120)
 *
 * COMPILATION:
 *   nvcc -ptx -arch=sm_86 -O3 --use_fast_math -o nhs_spike_detect.ptx nhs_spike_detect.cu
 *   nvcc -ptx -arch=sm_120 -O3 --use_fast_math -o nhs_spike_detect_sm120.ptx nhs_spike_detect.cu
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// ============================================================================
// CONSTANTS
// ============================================================================

// Block size for spike detection kernels (SM120 optimized)
#define SPIKE_BLOCK_SIZE    256
#define SPIKE_MIN_BLOCKS    4   // 1024 threads/SM for memory-bound workload

// Default detection parameters (can be overridden via kernel arguments)
#define DEFAULT_DISPLACEMENT_THRESHOLD  0.5f    // Angstroms
#define DEFAULT_PROXIMITY_THRESHOLD     6.0f    // Angstroms
#define DEFAULT_MAX_NEARBY              20      // Atom count for isolation test

// Shared memory tile size for aromatic atom positions
// Each aromatic loads 3 floats (12 bytes). 256 threads * 12 = 3072 bytes
#define SMEM_TILE_FLOATS    (SPIKE_BLOCK_SIZE * 3)

// Maximum spikes per kernel launch (safety bound for output buffer)
#define MAX_SPIKES_PER_LAUNCH   65536

// ============================================================================
// OUTPUT STRUCT - must match Rust GpuSpikeEvent exactly (repr(C, packed))
// ============================================================================

/**
 * Packed spike event structure - 60 bytes total.
 *
 * Layout (byte offsets):
 *   [0..4)    timestep         : int32
 *   [4..8)    voxel_idx        : int32
 *   [8..20)   position[3]      : float32 x 3
 *   [20..24)  intensity        : float32
 *   [24..56)  nearby_residues[8]: int32 x 8
 *   [56..60)  n_residues       : int32
 *
 * We write this as raw bytes to avoid compiler padding issues with float3.
 * The struct is attribute-packed to match Rust's #[repr(C, packed)].
 */
struct __attribute__((packed)) GpuSpikeEvent {
    int   timestep;
    int   voxel_idx;
    float position_x;
    float position_y;
    float position_z;
    float intensity;
    int   nearby_residues[8];
    int   n_residues;
};

// Compile-time size assertion: GpuSpikeEvent must be exactly 60 bytes
// (4 + 4 + 4 + 4 + 4 + 4 + 32 + 4 = 60)
// This is enforced in Rust as well, but we verify here for safety.
static_assert(sizeof(GpuSpikeEvent) == 60, "GpuSpikeEvent must be 60 bytes (packed)");

// ============================================================================
// DEVICE HELPERS
// ============================================================================

/**
 * Compute 3D Euclidean distance between two positions stored as flat arrays.
 */
__device__ __forceinline__ float compute_displacement(
    const float* __restrict__ curr,
    const float* __restrict__ prev,
    int idx3   // base index (atom_index * 3)
) {
    float dx = curr[idx3]     - prev[idx3];
    float dy = curr[idx3 + 1] - prev[idx3 + 1];
    float dz = curr[idx3 + 2] - prev[idx3 + 2];
    return sqrtf(dx * dx + dy * dy + dz * dz);
}

/**
 * Count atoms within a distance cutoff from a reference point.
 *
 * Uses the squared distance comparison to avoid unnecessary sqrtf.
 * Skips the source atom (self_atom_idx) in the count.
 *
 * @param positions       All atom positions [n_atoms * 3]
 * @param ref_x/y/z       Reference point coordinates
 * @param n_atoms         Total number of atoms
 * @param self_atom_idx   The aromatic atom index to skip in counting
 * @param cutoff_sq       Squared proximity threshold (proximity_threshold^2)
 * @return number of atoms within cutoff (excluding self)
 */
__device__ int count_nearby_atoms(
    const float* __restrict__ positions,
    float ref_x, float ref_y, float ref_z,
    int n_atoms,
    int self_atom_idx,
    float cutoff_sq
) {
    int count = 0;

    for (int i = 0; i < n_atoms; i++) {
        if (i == self_atom_idx) continue;

        int i3 = i * 3;
        float dx = positions[i3]     - ref_x;
        float dy = positions[i3 + 1] - ref_y;
        float dz = positions[i3 + 2] - ref_z;
        float dist_sq = dx * dx + dy * dy + dz * dz;

        if (dist_sq < cutoff_sq) {
            count++;
        }
    }

    return count;
}

// ============================================================================
// SPIKE COUNTING KERNEL (for pre-allocation)
// ============================================================================

/**
 * Count how many spike events will be generated without writing them.
 *
 * Used for a two-pass approach: first count, allocate output buffer on host,
 * then run the detection kernel. This avoids over-allocating GPU memory.
 *
 * PARALLELISM: One thread per aromatic atom.
 *
 * @param positions              Current atom positions [n_atoms * 3]
 * @param prev_positions         Previous frame atom positions [n_atoms * 3]
 * @param aromatic_indices       Indices of aromatic atoms [n_aromatics]
 * @param n_aromatics            Number of aromatic atoms
 * @param n_atoms                Total number of atoms in the system
 * @param displacement_threshold Minimum displacement to trigger detection (A)
 * @param proximity_threshold    Distance cutoff for nearby atom count (A)
 * @param max_nearby             Maximum nearby count for pocket classification
 * @param spike_count            OUTPUT: atomic counter, must be zeroed before launch
 */
extern "C" __global__
__launch_bounds__(SPIKE_BLOCK_SIZE, SPIKE_MIN_BLOCKS)
void nhs_spike_count(
    const float* __restrict__ positions,
    const float* __restrict__ prev_positions,
    const int*   __restrict__ aromatic_indices,
    const int    n_aromatics,
    const int    n_atoms,
    const float  displacement_threshold,
    const float  proximity_threshold,
    const int    max_nearby,
    int*         spike_count
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_aromatics) return;

    const int atom_idx = aromatic_indices[tid];

    // Bounds check: ensure atom_idx is valid
    if (atom_idx < 0 || atom_idx >= n_atoms) return;

    const int idx3 = atom_idx * 3;

    // Step 1: Compute displacement between frames
    const float displacement = compute_displacement(positions, prev_positions, idx3);

    // Step 2: Only process atoms with significant movement
    if (displacement <= displacement_threshold) return;

    // Step 3: Count nearby atoms to determine pocket isolation
    const float cutoff_sq = proximity_threshold * proximity_threshold;
    const float ref_x = positions[idx3];
    const float ref_y = positions[idx3 + 1];
    const float ref_z = positions[idx3 + 2];

    const int nearby_count = count_nearby_atoms(
        positions, ref_x, ref_y, ref_z,
        n_atoms, atom_idx, cutoff_sq
    );

    // Step 4: Only count if sufficiently isolated (pocket-like environment)
    if (nearby_count < max_nearby) {
        atomicAdd(spike_count, 1);
    }
}

// ============================================================================
// MAIN SPIKE DETECTION KERNEL
// ============================================================================

/**
 * GPU-accelerated neuromorphic spike detection from atom position changes.
 *
 * Direct port of detect_spikes_from_positions() from nhs_rt_full.rs.
 * One thread per aromatic atom:
 *   1. Compute displacement = |curr_pos - prev_pos|
 *   2. If displacement > threshold, count nearby atoms within proximity_threshold
 *   3. If nearby_count < max_nearby (isolated = potential pocket):
 *        intensity = displacement * (max_nearby - nearby_count) / max_nearby
 *        Emit GpuSpikeEvent
 *
 * Uses atomicAdd on spike_count for thread-safe output compaction.
 *
 * SHARED MEMORY USAGE:
 *   Aromatic atom positions are loaded cooperatively into shared memory to
 *   reduce redundant global memory reads when multiple aromatics are in the
 *   same spatial neighborhood. Each block loads its tile of aromatic positions.
 *
 * PARALLELISM: One thread per aromatic atom.
 *
 * @param positions              Current atom positions [n_atoms * 3]
 * @param prev_positions         Previous frame atom positions [n_atoms * 3]
 * @param aromatic_indices       Indices of aromatic atoms [n_aromatics]
 * @param n_aromatics            Number of aromatic atoms
 * @param n_atoms                Total number of atoms in the system
 * @param timestep               Current simulation timestep
 * @param displacement_threshold Minimum displacement to trigger spike (A)
 * @param proximity_threshold    Distance cutoff for nearby atom counting (A)
 * @param max_nearby             Max nearby count for pocket isolation test
 * @param spike_events           OUTPUT: spike event buffer [max_spikes]
 * @param spike_count            OUTPUT: atomic counter for emitted spikes
 *                               Must be zeroed before launch.
 * @param max_spikes             Maximum number of spikes the output buffer can hold
 */
extern "C" __global__
__launch_bounds__(SPIKE_BLOCK_SIZE, SPIKE_MIN_BLOCKS)
void nhs_spike_detect(
    const float* __restrict__ positions,
    const float* __restrict__ prev_positions,
    const int*   __restrict__ aromatic_indices,
    const int    n_aromatics,
    const int    n_atoms,
    const int    timestep,
    const float  displacement_threshold,
    const float  proximity_threshold,
    const int    max_nearby,
    GpuSpikeEvent* __restrict__ spike_events,
    int*         spike_count,
    const int    max_spikes
) {
    // ----------------------------------------------------------------
    // Shared memory: cooperative load of this block's aromatic positions
    // ----------------------------------------------------------------
    // Each thread loads the current position for its assigned aromatic atom.
    // This gives fast reads when computing inter-aromatic distances later,
    // but the main proximity search still reads all atoms from global memory
    // (since n_atoms >> n_aromatics typically).
    __shared__ float s_aromatic_pos[SMEM_TILE_FLOATS];  // [blockDim.x * 3]
    __shared__ int   s_aromatic_idx[SPIKE_BLOCK_SIZE];   // atom indices for this tile

    const int tid     = blockIdx.x * blockDim.x + threadIdx.x;
    const int local   = threadIdx.x;

    // Cooperatively load aromatic positions into shared memory
    if (tid < n_aromatics) {
        const int atom_idx = aromatic_indices[tid];
        s_aromatic_idx[local] = atom_idx;

        if (atom_idx >= 0 && atom_idx < n_atoms) {
            const int idx3 = atom_idx * 3;
            s_aromatic_pos[local * 3]     = positions[idx3];
            s_aromatic_pos[local * 3 + 1] = positions[idx3 + 1];
            s_aromatic_pos[local * 3 + 2] = positions[idx3 + 2];
        } else {
            s_aromatic_pos[local * 3]     = 0.0f;
            s_aromatic_pos[local * 3 + 1] = 0.0f;
            s_aromatic_pos[local * 3 + 2] = 0.0f;
        }
    }
    __syncthreads();

    // Early exit for out-of-range threads
    if (tid >= n_aromatics) return;

    const int atom_idx = s_aromatic_idx[local];

    // Bounds check
    if (atom_idx < 0 || atom_idx >= n_atoms) return;

    const int idx3 = atom_idx * 3;

    // ----------------------------------------------------------------
    // Step 1: Compute displacement between current and previous frame
    // ----------------------------------------------------------------
    const float dx_disp = positions[idx3]     - prev_positions[idx3];
    const float dy_disp = positions[idx3 + 1] - prev_positions[idx3 + 1];
    const float dz_disp = positions[idx3 + 2] - prev_positions[idx3 + 2];
    const float displacement = sqrtf(dx_disp * dx_disp + dy_disp * dy_disp + dz_disp * dz_disp);

    // Only process atoms with displacement above threshold
    if (displacement <= displacement_threshold) return;

    // ----------------------------------------------------------------
    // Step 2: Count nearby atoms within proximity_threshold
    // ----------------------------------------------------------------
    // Read this atom's current position from shared memory (fast)
    const float ref_x = s_aromatic_pos[local * 3];
    const float ref_y = s_aromatic_pos[local * 3 + 1];
    const float ref_z = s_aromatic_pos[local * 3 + 2];

    const float cutoff_sq = proximity_threshold * proximity_threshold;
    int nearby_count = 0;

    // Full scan over all atoms - this is the O(n_atoms) hot loop.
    // For proteins < 50K atoms this completes in < 1ms on modern GPUs.
    // For larger systems, a cell-list or neighbor-list approach could be
    // used, but for NHS pipeline proteins this direct approach is sufficient.
    for (int i = 0; i < n_atoms; i++) {
        if (i == atom_idx) continue;

        const int i3 = i * 3;
        const float dx = positions[i3]     - ref_x;
        const float dy = positions[i3 + 1] - ref_y;
        const float dz = positions[i3 + 2] - ref_z;
        const float dist_sq = dx * dx + dy * dy + dz * dz;

        if (dist_sq < cutoff_sq) {
            nearby_count++;
        }

        // Early termination: once we've found max_nearby neighbors,
        // this atom is NOT in a pocket, so skip the rest
        if (nearby_count >= max_nearby) break;
    }

    // ----------------------------------------------------------------
    // Step 3: Emit spike if sufficiently isolated (pocket-like)
    // ----------------------------------------------------------------
    if (nearby_count >= max_nearby) return;

    // Compute spike intensity: displacement weighted by isolation score
    // Exactly matches: displacement * (20.0 - nearby_count) / 20.0
    const float isolation = (float)(max_nearby - nearby_count) / (float)max_nearby;
    const float intensity = displacement * isolation;

    // Atomic increment to get output slot
    const int out_idx = atomicAdd(spike_count, 1);

    // Bounds check on output buffer
    if (out_idx >= max_spikes) return;

    // ----------------------------------------------------------------
    // Step 4: Write packed GpuSpikeEvent to output buffer
    // ----------------------------------------------------------------
    GpuSpikeEvent evt;
    evt.timestep   = timestep;
    evt.voxel_idx  = 0;    // Not used in position-based detection
    evt.position_x = ref_x;
    evt.position_y = ref_y;
    evt.position_z = ref_z;
    evt.intensity  = intensity;

    // Zero out residue tracking (not computed in this kernel)
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        evt.nearby_residues[r] = 0;
    }
    evt.n_residues = 0;

    // Write the full struct. Since GpuSpikeEvent is packed (60 bytes, not
    // aligned to any power of two), we write field by field to ensure
    // correct byte layout on the GPU side matching Rust repr(C, packed).
    spike_events[out_idx] = evt;
}

// ============================================================================
// BATCHED SPIKE DETECTION KERNEL (multiple frames at once)
// ============================================================================

/**
 * Detect spikes across multiple frames in a single kernel launch.
 *
 * This amortizes launch overhead when processing many frames.
 * Grid dimensions: (ceil(n_aromatics / BLOCK_SIZE), n_frames, 1)
 *
 * @param positions_batch        Positions for all frames [n_frames * n_atoms * 3]
 * @param prev_positions_batch   Previous positions [n_frames * n_atoms * 3]
 * @param aromatic_indices       Aromatic atom indices [n_aromatics]
 * @param n_aromatics            Number of aromatic atoms
 * @param n_atoms                Atoms per frame
 * @param base_timestep          Timestep of first frame
 * @param frame_stride           Timestep increment between frames
 * @param displacement_threshold Displacement threshold (A)
 * @param proximity_threshold    Proximity cutoff (A)
 * @param max_nearby             Max nearby count for pocket test
 * @param spike_events           OUTPUT: spike event buffer [max_spikes]
 * @param spike_count            OUTPUT: atomic counter (shared across all frames)
 * @param max_spikes             Output buffer capacity
 */
extern "C" __global__
__launch_bounds__(SPIKE_BLOCK_SIZE, SPIKE_MIN_BLOCKS)
void nhs_spike_detect_batch(
    const float* __restrict__ positions_batch,
    const float* __restrict__ prev_positions_batch,
    const int*   __restrict__ aromatic_indices,
    const int    n_aromatics,
    const int    n_atoms,
    const int    base_timestep,
    const int    frame_stride,
    const float  displacement_threshold,
    const float  proximity_threshold,
    const int    max_nearby,
    GpuSpikeEvent* __restrict__ spike_events,
    int*         spike_count,
    const int    max_spikes
) {
    const int arom_tid  = blockIdx.x * blockDim.x + threadIdx.x;
    const int frame_idx = blockIdx.y;

    if (arom_tid >= n_aromatics) return;

    const int atom_idx = aromatic_indices[arom_tid];
    if (atom_idx < 0 || atom_idx >= n_atoms) return;

    // Offset into this frame's position data
    const int frame_offset = frame_idx * n_atoms * 3;
    const float* positions      = positions_batch      + frame_offset;
    const float* prev_positions = prev_positions_batch + frame_offset;

    const int idx3 = atom_idx * 3;

    // Step 1: Displacement
    const float dx_d = positions[idx3]     - prev_positions[idx3];
    const float dy_d = positions[idx3 + 1] - prev_positions[idx3 + 1];
    const float dz_d = positions[idx3 + 2] - prev_positions[idx3 + 2];
    const float displacement = sqrtf(dx_d * dx_d + dy_d * dy_d + dz_d * dz_d);

    if (displacement <= displacement_threshold) return;

    // Step 2: Nearby atom count
    const float ref_x = positions[idx3];
    const float ref_y = positions[idx3 + 1];
    const float ref_z = positions[idx3 + 2];
    const float cutoff_sq = proximity_threshold * proximity_threshold;
    int nearby_count = 0;

    for (int i = 0; i < n_atoms; i++) {
        if (i == atom_idx) continue;

        const int i3 = i * 3;
        const float dx = positions[i3]     - ref_x;
        const float dy = positions[i3 + 1] - ref_y;
        const float dz = positions[i3 + 2] - ref_z;
        const float dist_sq = dx * dx + dy * dy + dz * dz;

        if (dist_sq < cutoff_sq) {
            nearby_count++;
        }

        if (nearby_count >= max_nearby) break;
    }

    // Step 3: Emit spike
    if (nearby_count >= max_nearby) return;

    const float isolation = (float)(max_nearby - nearby_count) / (float)max_nearby;
    const float intensity = displacement * isolation;

    const int out_idx = atomicAdd(spike_count, 1);
    if (out_idx >= max_spikes) return;

    const int timestep = base_timestep + frame_idx * frame_stride;

    GpuSpikeEvent evt;
    evt.timestep   = timestep;
    evt.voxel_idx  = 0;
    evt.position_x = ref_x;
    evt.position_y = ref_y;
    evt.position_z = ref_z;
    evt.intensity  = intensity;

    #pragma unroll
    for (int r = 0; r < 8; r++) {
        evt.nearby_residues[r] = 0;
    }
    evt.n_residues = 0;

    spike_events[out_idx] = evt;
}

// ============================================================================
// UTILITY: Zero the spike counter
// ============================================================================

/**
 * Zero the spike counter. Launch with <<<1, 1>>>.
 * Useful for resetting between frames without a host-device round trip.
 */
extern "C" __global__ void nhs_spike_reset_counter(int* spike_count) {
    spike_count[0] = 0;
}
