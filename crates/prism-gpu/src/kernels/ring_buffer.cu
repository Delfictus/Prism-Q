// ═══════════════════════════════════════════════════════════════════════
// PRISM-TWIN: Spike Ring Buffer Module
//
// Standalone circular buffer for continuous spike exchange between
// twin MD streams. Operates BETWEEN kernel launches — does NOT
// modify nhs_amber_fused.cu.
//
// Architecture:
//   AFTER fused kernel: ring_buffer_push_batch (copy new spikes to ring)
//   BEFORE fused kernel: ring_buffer_read_and_adapt (apply to thresholds)
//   PERIODICALLY: ring_buffer_threshold_recovery (prevent permanent reduction)
// ═══════════════════════════════════════════════════════════════════════

#ifndef RING_BUFFER_CU
#define RING_BUFFER_CU

#include <stddef.h>

#include "protocol_state.cuh"

#define RING_DEFAULT_CAPACITY 8192

// Must match GpuSpikeEvent in fused_engine.rs
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
    int primary_residue_id;  // lossless residue attribution (was: pad)
};

struct RingBufferState {
    unsigned int head;           // total writes (monotonic, never reset)
    unsigned int tail;           // total reads (monotonic, never reset)
    unsigned int capacity;       // ring size
    unsigned int overflow_count; // times head overtook tail+capacity
};


// ─────────────────────────────────────────────────────────────────────
// KERNEL 1: Push batch of new spikes into ring buffer
// Called AFTER each fused kernel launch completes.
// Only copies spikes [prev_count .. curr_count) from the fused spike buffer.
// ─────────────────────────────────────────────────────────────────────

extern "C" __global__ void ring_buffer_push_batch(
    RingSpikeEvent* __restrict__ ring_buffer,    // [capacity]
    unsigned int* __restrict__ head_ptr,          // single uint: write pointer
    unsigned int* __restrict__ tail_ptr,          // single uint: read pointer
    unsigned int* __restrict__ overflow_ptr,      // single uint: overflow counter
    unsigned int capacity,
    const RingSpikeEvent* __restrict__ spike_buffer, // fused kernel output
    unsigned int prev_count,
    unsigned int curr_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_new = (int)(curr_count - prev_count);
    if (idx >= n_new) return;

    // Reserve a write position
    unsigned int write_pos = atomicAdd(head_ptr, 1);

    // Check overflow
    unsigned int current_tail = *tail_ptr;
    if (write_pos - current_tail >= capacity) {
        atomicAdd(overflow_ptr, 1);
    }

    // Write spike to ring (circular)
    ring_buffer[write_pos % capacity] = spike_buffer[prev_count + idx];
}


// ─────────────────────────────────────────────────────────────────────
// KERNEL 2: Read spikes from ring buffer and adapt thresholds
// Called BEFORE each fused kernel launch.
// Single-threaded (ring buffer typically has <1000 entries per step).
// ─────────────────────────────────────────────────────────────────────

// One physical endpoint/source-group window, even when its events need several
// bounded publication batches. Max/hotspot ties must span the entire window.
struct RingAdaptationWindow {
    float max_reduction;
    int hotspot_residue;
    unsigned int step;
    unsigned int active;
    unsigned int failure;
};
static_assert(sizeof(RingAdaptationWindow) == 20, "window ABI");

__device__ __forceinline__ void ring_buffer_read_and_adapt_impl(
    const RingSpikeEvent* __restrict__ ring_buffer,
    unsigned int* __restrict__ head_ptr,
    unsigned int* __restrict__ tail_ptr,
    unsigned int capacity,
    float* __restrict__ osc_thresholds,
    const float* __restrict__ base_thresholds,
    int n_voxels_x, int n_voxels_y, int n_voxels_z,
    float grid_origin_x, float grid_origin_y, float grid_origin_z,
    float voxel_size,
    float sensitivity_boost,
    float max_reduction_fraction,
    unsigned int current_step,
    float decay_constant,
    // ASC steering: coupling kernel writes when it detects hotspots
    ProtocolState* __restrict__ d_protocol, // nullable: pass nullptr to skip ASC
    RingAdaptationWindow* window = nullptr,
    bool finalize_window = true
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    unsigned int head = *head_ptr;
    unsigned int tail = *tail_ptr;
    unsigned int n_to_process = head - tail;

    if (n_to_process > capacity) {
        tail = head - capacity;
        n_to_process = capacity;
    }

    // ASC: track max threshold reduction for hotspot detection
    float max_reduction = window ? window->max_reduction : 0.0f;
    int hotspot_residue = window ? window->hotspot_residue : -1;

    // ── Stage 2: Closed-loop ASC steering ──
    //
    // Snapshot the steering focus list once per kernel launch (it's small —
    // up to 64 entries — and the rest of the loop only reads it). Reading
    // through the device pointer once amortizes the global-memory load.
    //
    // The autonomous CUDA Graph captures `d_protocol` as a device pointer;
    // the contents change between graph replays via the host's
    // cuMemcpyHtoDAsync writes. This kernel re-loads the focus list every
    // launch (every chunk), so it always sees the latest decisions from
    // the ASC controller without any graph re-capture.
    int n_focus = 0;
    if (d_protocol != nullptr) {
        n_focus = d_protocol->steering_focus_count;
        if (n_focus < 0) n_focus = 0;
        if (n_focus > 64) n_focus = 64;
        // Stage 2 calibration: snapshot the first focus residue id every launch
        // so the host can verify the kernel sees the same data the upload wrote.
        d_protocol->last_seen_focus_id = (n_focus > 0)
            ? d_protocol->steering_focus_residues[0].residue_id
            : -3;
    }

    for (unsigned int i = 0; i < n_to_process; i++) {
        unsigned int ring_idx = (tail + i) % capacity;
        RingSpikeEvent spike = ring_buffer[ring_idx];

        // Map spike position to voxel
        int vx = (int)((spike.x - grid_origin_x) / voxel_size);
        int vy = (int)((spike.y - grid_origin_y) / voxel_size);
        int vz = (int)((spike.z - grid_origin_z) / voxel_size);

        if (vx < 0 || vx >= n_voxels_x) continue;
        if (vy < 0 || vy >= n_voxels_y) continue;
        if (vz < 0 || vz >= n_voxels_z) continue;

        int voxel_idx = vx + vy * n_voxels_x + vz * n_voxels_x * n_voxels_y;

        float age = (float)((int)current_step - spike.timestep);
        if (age < 0.0f) age = 0.0f;
        float time_weight = expf(-age / decay_constant);
        float boost = sensitivity_boost * spike.intensity * time_weight;

        // ── Stage 2 steering boost ──
        //
        // Extract the spike's primary residue id from the bit-packed
        // primary_residue_id field (low 16 bits hold the residue id; bits
        // 16-25 hold the 10-bit phase angle from compact_and_push). Then
        // walk the focus list and check if this spike's residue is in it.
        // If it is, multiply the threshold-reduction boost by
        // `(1 + steering_gain × weight)`.
        //
        // STEERING_GAIN = 2.0 means a synergy_fraction of 1.0 produces a
        // 3× boost on the spike's threshold reduction. This is moderate
        // enough not to overwhelm the natural physics signal but strong
        // enough to actually shift the cross-group divergence. Documented
        // as a tunable in the kernel comment so future calibration work
        // can swap it without searching.
        float steer_weight = 0.0f;
        const int residue_bits = spike.primary_residue_id & 0xFFFF;
        const int spike_residue = (residue_bits == 0xFFFF) ? -1 : residue_bits;
        // Stage 2 calibration: count every spike processed and record its
        // residue id (raw, before & 0xFFFF) so the host can sanity-check the
        // residue id space the spike pipeline produces.
        if (d_protocol != nullptr) {
            atomicAdd(&d_protocol->processed_spike_count, 1u);
            d_protocol->last_seen_spike_residue = spike_residue;
        }
        if (n_focus > 0 && spike_residue >= 0) {
            #pragma unroll 8
            for (int k = 0; k < 64; k++) {
                if (k >= n_focus) break;
                if (d_protocol->steering_focus_residues[k].residue_id == spike_residue) {
                    steer_weight = d_protocol->steering_focus_residues[k].weight;
                    // Stage 2 calibration: count this as a successful focus
                    // match. atomicAdd because multiple read_and_adapt launches
                    // (across chunks) and multiple matched spikes inside one
                    // launch all increment the same counter. The kernel itself
                    // is single-threaded per launch, so the atomic is only
                    // necessary across launches — but the cost of an atomic on
                    // a single-threaded path is negligible.
                    atomicAdd(&d_protocol->focus_match_count, 1u);
                    break;
                }
            }
        }
        const float STEERING_GAIN = 2.0f;
        boost *= (1.0f + STEERING_GAIN * steer_weight);

        float base = base_thresholds[voxel_idx];
        float floor_val = base * (1.0f - max_reduction_fraction);
        float current = osc_thresholds[voxel_idx];
        float new_val = fmaxf(floor_val, current - boost);
        osc_thresholds[voxel_idx] = new_val;

        // ASC: track largest threshold reduction (hotspot candidate)
        float reduction = (base - new_val) / fmaxf(base, 1e-6f);
        if (reduction > max_reduction) {
            max_reduction = reduction;
            hotspot_residue = spike_residue;
        }
    }

    *tail_ptr = head;

    // ── ASC Steering Hook ──
    // When coupling detects a hotspot (>20% threshold reduction at a voxel),
    // write steering fields. Director reads these next step and adjusts UV/temp.
    if (window != nullptr) {
        window->max_reduction = max_reduction;
        window->hotspot_residue = hotspot_residue;
    }
    if (finalize_window && d_protocol != nullptr && max_reduction > 0.20f) {
        d_protocol->steering_uv_boost = 1.5f;          // 50% UV boost at hotspot
        d_protocol->steering_focus_residue = hotspot_residue;
        d_protocol->steering_flags |= 0x1;              // phase_lock flag
    }
}


// Existing single-consumer ABI retains its original arithmetic and cursor.
extern "C" __global__ void ring_buffer_read_and_adapt(
    const RingSpikeEvent* __restrict__ ring_buffer,
    unsigned int* __restrict__ head_ptr,
    unsigned int* __restrict__ tail_ptr,
    unsigned int capacity,
    float* __restrict__ osc_thresholds,
    const float* __restrict__ base_thresholds,
    int n_voxels_x, int n_voxels_y, int n_voxels_z,
    float grid_origin_x, float grid_origin_y, float grid_origin_z,
    float voxel_size,
    float sensitivity_boost,
    float max_reduction_fraction,
    unsigned int current_step,
    float decay_constant,
    // ASC steering: coupling kernel writes when it detects hotspots
    ProtocolState* __restrict__ d_protocol  // nullable: pass nullptr to skip ASC
) {
    ring_buffer_read_and_adapt_impl(ring_buffer, head_ptr, tail_ptr, capacity, osc_thresholds, base_thresholds, n_voxels_x, n_voxels_y, n_voxels_z, grid_origin_x, grid_origin_y, grid_origin_z, voxel_size, sensitivity_boost, max_reduction_fraction, current_step, decay_constant, d_protocol);
}

// One receipt per physical recipient; owned for the entire captured graph lifetime.
struct RingRecipientReceipt {
    unsigned int cursor;
    unsigned int last_head;
    unsigned int last_completed_step;
    unsigned int last_available;
    unsigned long long total_inspected;
    unsigned long long total_overwritten;
    unsigned long long read_windows;
};
static_assert(sizeof(RingRecipientReceipt) == 40, "Rust/CUDA ring receipt ABI");

// Same adaptation body, independent cursor and replay-time physical clock.
extern "C" __global__ void ring_buffer_read_and_adapt_recipient(
    const RingSpikeEvent* __restrict__ ring_buffer,
    unsigned int* __restrict__ head_ptr,
    RingRecipientReceipt* __restrict__ receipt,
    unsigned int capacity,
    float* __restrict__ osc_thresholds,
    const float* __restrict__ base_thresholds,
    int n_voxels_x, int n_voxels_y, int n_voxels_z,
    float grid_origin_x, float grid_origin_y, float grid_origin_z,
    float voxel_size,
    float sensitivity_boost,
    float max_reduction_fraction,
    float decay_constant,
    // ASC steering: coupling kernel writes when it detects hotspots
    ProtocolState* __restrict__ d_protocol  // nullable: pass nullptr to skip ASC
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (receipt == nullptr || d_protocol == nullptr || capacity == 0u) return;
    const unsigned int head = *head_ptr;
    const unsigned int available = head - receipt->cursor;
    const unsigned int inspected = min(available, capacity);
    const unsigned int current_step = d_protocol->current_step;
    ring_buffer_read_and_adapt_impl(ring_buffer, head_ptr, &receipt->cursor, capacity, osc_thresholds, base_thresholds, n_voxels_x, n_voxels_y, n_voxels_z, grid_origin_x, grid_origin_y, grid_origin_z, voxel_size, sensitivity_boost, max_reduction_fraction, current_step, decay_constant, d_protocol);
    receipt->last_head = head;
    receipt->last_completed_step = current_step;
    receipt->last_available = available;
    receipt->total_inspected += inspected;
    receipt->total_overwritten += available - inspected;
    receipt->read_windows += 1ULL;
}

// Window operations: 0 begin, 1 accumulate one bounded batch, 2 finalize.
// The original arithmetic above remains the sole adaptation implementation.
// No threshold recovery, physical advance, or steering publication between batches.
extern "C" __global__ void ring_buffer_adapt_window(
    const RingSpikeEvent* ring_buffer, unsigned int* head_ptr,
    RingRecipientReceipt* receipt, unsigned int capacity,
    float* osc_thresholds, const float* base_thresholds,
    int nx, int ny, int nz, float ox, float oy, float oz, float voxel_size,
    float sensitivity_boost, float max_reduction_fraction, float decay_constant,
    ProtocolState* protocol, RingAdaptationWindow* window, unsigned int operation
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (!protocol || !window || !receipt || !capacity) return;
    if (window->failure) return;
    if (operation == 0u) {
        if (window->active) { window->failure |= 1u; return; }
        window->max_reduction = 0.0f; window->hotspot_residue = -1;
        window->step = protocol->current_step; window->active = 1u;
        return;
    }
    if (!window->active || window->step != protocol->current_step || operation > 2u) {
        window->failure |= 2u; return;
    }
    const unsigned int head = *head_ptr;
    const unsigned int available = head - receipt->cursor;
    if (available > capacity || (operation == 2u && available != 0u)) {
        window->failure |= 4u; return;
    }
    ring_buffer_read_and_adapt_impl(ring_buffer, head_ptr, &receipt->cursor,
        capacity, osc_thresholds, base_thresholds, nx, ny, nz, ox, oy, oz,
        voxel_size, sensitivity_boost, max_reduction_fraction, window->step,
        decay_constant, protocol, window, operation == 2u);
    receipt->last_head = head; receipt->last_completed_step = window->step;
    receipt->last_available = available; receipt->total_inspected += available;
    if (operation == 2u) { receipt->read_windows += 1ULL; window->active = 0u; }
}

// ─────────────────────────────────────────────────────────────────────
// KERNEL 3: Threshold recovery (gradual restore toward baseline)
// Called periodically to prevent permanent threshold suppression.
// ─────────────────────────────────────────────────────────────────────

extern "C" __global__ void ring_buffer_threshold_recovery(
    float* __restrict__ osc_thresholds,
    const float* __restrict__ base_thresholds,
    unsigned int n_voxels_total,
    float recovery_rate
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int)n_voxels_total) return;

    float current = osc_thresholds[idx];
    float base = base_thresholds[idx];
    osc_thresholds[idx] = current + recovery_rate * (base - current);
}


// ─────────────────────────────────────────────────────────────────────
// KERNEL 4: Reset ring buffer state (for initialization)
// ─────────────────────────────────────────────────────────────────────

// ═══════════════════════════════════════════════════════════════════════
// GATE 3: Device-side spike compaction + push
// ═══════════════════════════════════════════════════════════════════════
//
// Reads GpuSpikeEvent (96 bytes) directly from d_spike_events on GPU,
// extracts the 12 fields that RingSpikeEvent needs (48 bytes), and pushes
// directly into the ring buffer — ALL on GPU, zero CPU memcpy.
//
// This kernel replaces the CPU-side push_compacted() path:
//   Old: GPU spikes → memcpy_dtoh → CPU field extraction → memcpy_htod → push kernel
//   New: GPU spikes → this kernel → ring buffer (never leaves VRAM)

// GpuSpikeEvent layout — must match the sealed 96-byte SpikeEvent packet in
// nhs_amber_fused.cu and fused_engine.rs.  This direct device consumer is
// intentionally kept separate from the 48-byte RingSpikeEvent payload below.
struct GpuSpikeEvent {
    int timestep;               // 0
    int voxel_idx;              // 4
    float pos_x, pos_y, pos_z;  // 8, 12, 16 (float3 = 3 contiguous floats)
    float intensity;            // 20
    int nearby_residues[8];     // 24-55
    int n_residues;             // 56
    int spike_source;           // 60
    float wavelength_nm;        // 64
    int aromatic_type;          // 68
    int aromatic_residue_id;    // 72
    float water_density;        // 76
    float vibrational_energy;   // 80
    int n_nearby_excited;       // 84
    float wd_change;            // 88
    unsigned int phase_bits;    // 92
};  // total = 96 bytes

static_assert(sizeof(GpuSpikeEvent) == 96,
              "compact_and_push must consume the sealed 96-byte GpuSpikeEvent packet");
static_assert(offsetof(GpuSpikeEvent, phase_bits) == 92,
              "GpuSpikeEvent.phase_bits must remain at byte 92");

extern "C" __global__ void compact_and_push(
    const GpuSpikeEvent* __restrict__ spike_events, // [max_spikes] source (96 bytes each)
    const int* __restrict__ spike_count,              // [1] number of valid spikes
    unsigned char* __restrict__ ring_buffer,           // ring buffer storage
    unsigned int* __restrict__ head_ptr,               // monotonic write counter
    unsigned int* __restrict__ overflow_ptr,            // overflow counter
    unsigned int capacity,                             // ring buffer capacity
    unsigned int source_capacity,                      // d_spike_events record capacity
    const ProtocolState* __restrict__ d_protocol       // for phase angle computation
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_spikes = *spike_count;
    // The source buffer is fixed-capacity. An overrun is rejected by the
    // authoritative host readout; this feedback path must never index beyond
    // the records that physically exist in d_spike_events.
    if (capacity == 0 || n_spikes <= 0 || (unsigned int)n_spikes > source_capacity) return;

    const unsigned int valid_spikes = (unsigned int)n_spikes;
    // The ring is a bounded recent-evidence exchange. Match push_compacted's
    // established tail semantics: retain the newest source window that fits
    // the ring rather than the first fixed launch-sized prefix.
    const unsigned int retained_spikes =
        (valid_spikes < capacity) ? valid_spikes : capacity;
    if ((unsigned int)tid >= retained_spikes) return;
    const unsigned int source_start = valid_spikes - retained_spikes;

    // Read from the fat GpuSpikeEvent
    const GpuSpikeEvent* src = &spike_events[source_start + (unsigned int)tid];

    // Compact into RingSpikeEvent (48 bytes)
    RingSpikeEvent compact;
    compact.timestep          = src->timestep;
    compact.voxel_idx         = src->voxel_idx;
    compact.x                 = src->pos_x;
    compact.y                 = src->pos_y;
    compact.z                 = src->pos_z;
    compact.intensity         = src->intensity;
    compact.vibrational_energy = src->vibrational_energy;
    compact.water_density     = src->water_density;
    compact.n_nearby_excited  = src->n_nearby_excited;
    compact.spike_source      = src->spike_source;
    compact.wavelength_nm     = src->wavelength_nm;

    // ── BIT-PACKED PHASE ANGLE + RESIDUE ID ──
    // Bits 0-15:  primary residue ID (0-65535)
    // Bits 16-25: quantized CCNS phase angle (0-1023 → 0-2π)
    // Bits 26-31: reserved
    //
    // The native producer stamped this 10-bit protocol phase on the event.
    // Preserve that per-event value through exchange; do not restamp at push.
    int residue_id = (src->n_residues > 0) ? src->nearby_residues[0] : -1;
    int residue_bits = residue_id & 0xFFFF; // 16-bit residue ID

    // Preserve the native event's recorded phase. An endpoint ProtocolState
    // cannot reconstruct the phase of every event accumulated in this chunk.
    const int phase_bits = (int)(src->phase_bits & 0x3FFu);

    // Pack: residue_id[15:0] | phase_angle[25:16]
    compact.primary_residue_id = residue_bits | (phase_bits << 16);

    // Atomic increment head to claim a slot
    unsigned int slot = atomicAdd(head_ptr, 1);
    unsigned int ring_idx = slot % capacity;

    // Write compact event to ring buffer
    RingSpikeEvent* dst = (RingSpikeEvent*)ring_buffer + ring_idx;
    *dst = compact;

    // Track overflow (informational — ring overwrites old data, which is correct)
    if (slot >= capacity) {
        atomicAdd(overflow_ptr, 1);
    }
}

// ─────────────────────────────────────────────────────────────────────

extern "C" __global__ void ring_buffer_reset(
    unsigned int* __restrict__ head_ptr,
    unsigned int* __restrict__ tail_ptr,
    unsigned int* __restrict__ overflow_ptr
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *head_ptr = 0;
        *tail_ptr = 0;
        *overflow_ptr = 0;
    }
}

#endif // RING_BUFFER_CU


// Lossless delta policy, ABI isolated from compact_and_push's recent-window policy.
// Reuses the original PRISM bounded reservation algorithm. Producers must complete
// before recipient readers; retirement must complete before the next producer wave.
// failure bits: 1 invalid source count, 2 invalid source cursor, 4 invalid ring state,
// 8 invalid recipient cursor. overflow remains a sticky PRESSURE flag, not event loss.
__device__ __forceinline__ void compact_and_push_delta_lossless_body(
    const GpuSpikeEvent* __restrict__ spike_events,
    const int* __restrict__ spike_count,
    unsigned int source_capacity,
    int* __restrict__ push_cursor,
    unsigned char* __restrict__ ring_buffer,
    unsigned int* __restrict__ head_ptr,
    const unsigned int* __restrict__ tail_ptr,
    unsigned int* __restrict__ overflow_ptr,
    unsigned int capacity,
    unsigned int* __restrict__ failure_ptr,
    unsigned int* __restrict__ continuation_pending
) {
    if (blockIdx.x != 0) return;
    __shared__ int cursor;
    __shared__ unsigned int pending, start, reserved;
    if (threadIdx.x == 0) {
        cursor = *push_cursor;
        const int count = *spike_count;
        pending = start = reserved = 0u;
        if (count < 0 || (unsigned int)count > source_capacity) {
            atomicOr(failure_ptr, 1u);
        } else if (cursor < 0 || cursor > count) {
            atomicOr(failure_ptr, 2u);
        } else if (capacity == 0u || (capacity & (capacity - 1u)) != 0u || capacity >= 0x80000000u) {
            atomicOr(failure_ptr, 4u);
        } else if (atomicAdd(failure_ptr, 0u) == 0u) {
            pending = (unsigned int)(count - cursor);
            while (pending > 0u) {
                const unsigned int head = atomicAdd(head_ptr, 0u);
                const unsigned int tail = *tail_ptr;
                const unsigned int used = head - tail;
                if (used > capacity) { atomicOr(failure_ptr, 4u); break; }
                const unsigned int take = min(pending, capacity - used);
                if (take == 0u) break;
                if (atomicCAS(head_ptr, head, head + take) == head) {
                    start = head; reserved = take; break;
                }
            }
        }
    }
    __syncthreads();
    for (unsigned int i = threadIdx.x; i < reserved; i += blockDim.x) {
        const GpuSpikeEvent* src = &spike_events[cursor + (int)i];
        RingSpikeEvent out;
        out.timestep = src->timestep;
        out.voxel_idx = src->voxel_idx;
        out.x = src->pos_x; out.y = src->pos_y; out.z = src->pos_z;
        out.intensity = src->intensity;
        out.vibrational_energy = src->vibrational_energy;
        out.water_density = src->water_density;
        out.n_nearby_excited = src->n_nearby_excited;
        out.spike_source = src->spike_source;
        out.wavelength_nm = src->wavelength_nm;
        const int residue = src->n_residues > 0 ? src->nearby_residues[0] : -1;
        // Zero is a valid recorded phase. Never restamp from endpoint protocol.
        out.primary_residue_id = (residue & 0xffff) | ((src->phase_bits & 0x3ffu) << 16);
        ((RingSpikeEvent*)ring_buffer)[(start + i) % capacity] = out;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        *push_cursor = cursor + (int)reserved;
        if (reserved < pending) atomicExch(overflow_ptr, 1u);
        if (continuation_pending != nullptr) {
            *continuation_pending = (*failure_ptr == 0u && *push_cursor < *spike_count) ? 1u : 0u;
        }
    }
}

extern "C" __global__ void compact_and_push_delta_lossless(
    const GpuSpikeEvent* __restrict__ spike_events,
    const int* __restrict__ spike_count, unsigned int source_capacity,
    int* __restrict__ push_cursor, unsigned char* __restrict__ ring_buffer,
    unsigned int* __restrict__ head_ptr, const unsigned int* __restrict__ tail_ptr,
    unsigned int* __restrict__ overflow_ptr, unsigned int capacity,
    unsigned int* __restrict__ failure_ptr
) {
    compact_and_push_delta_lossless_body(spike_events, spike_count, source_capacity, push_cursor, ring_buffer, head_ptr, tail_ptr, overflow_ptr, capacity, failure_ptr, nullptr);
}

// Distinct ABI: the same native publication additionally exports actual remaining work.
extern "C" __global__ void compact_and_push_delta_lossless_continue(
    const GpuSpikeEvent* __restrict__ spike_events,
    const int* __restrict__ spike_count, unsigned int source_capacity,
    int* __restrict__ push_cursor, unsigned char* __restrict__ ring_buffer,
    unsigned int* __restrict__ head_ptr, const unsigned int* __restrict__ tail_ptr,
    unsigned int* __restrict__ overflow_ptr, unsigned int capacity,
    unsigned int* __restrict__ failure_ptr,
    unsigned int* __restrict__ continuation_pending
) {
    compact_and_push_delta_lossless_body(spike_events, spike_count, source_capacity, push_cursor, ring_buffer, head_ptr, tail_ptr, overflow_ptr, capacity, failure_ptr, continuation_pending);
}

// The pointer table contains ONLY actual readers of this source group. Same-group
// receipt slots are intentionally excluded. Readers and producers cannot overlap
// this retirement node. Capacity < 2^31 bounds modular head/cursor differences.
extern "C" __global__ void ring_buffer_retire_participating_recipients(
    const RingRecipientReceipt* const* __restrict__ recipients,
    unsigned int recipient_count,
    const unsigned int* __restrict__ head_ptr,
    unsigned int* __restrict__ tail_ptr,
    unsigned int capacity,
    unsigned int* __restrict__ failure_ptr
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const unsigned int head = *head_ptr;
    const unsigned int used = head - *tail_ptr;
    if (recipient_count == 0u || capacity == 0u || (capacity & (capacity - 1u)) != 0u || capacity >= 0x80000000u || used > capacity) {
        atomicOr(failure_ptr, 4u); return;
    }
    if (*failure_ptr != 0u) return;
    unsigned int retained = 0u;
    for (unsigned int i = 0; i < recipient_count; ++i) {
        if (recipients[i] == nullptr) { atomicOr(failure_ptr, 8u); return; }
        const unsigned int behind = head - recipients[i]->cursor;
        if (behind > used) { atomicOr(failure_ptr, 8u); return; }
        retained = max(retained, behind);
    }
    *tail_ptr = head - retained;
}

// Exact ownership handoff between a previous archived endpoint and the next
// resident physical window. This changes no adaptation law or event packet.
// Sources: [count*, publication_cursor*, failure*, ProtocolState*, capacity].
// Ring states: [head*, tail*, failure*]. Window pointers include every real
// source-group recipient exactly once. Expected: [step, epoch_lo, epoch_hi,
// reset_sources, archived_count[physical_worker]]. Receipt rows (8 u32) retain
// [expected_step, actual_step, archived, resident, published, failures, epoch_lo,
// epoch_hi], followed by one aggregate failure word. All pointer arrays and
// receipts belong to the owning coordinator and survive every parent replay.
extern "C" __global__ void ring_buffer_verify_archived_epoch_and_reset(
    const unsigned long long* __restrict__ sources,
    unsigned int n_sources,
    const unsigned long long* __restrict__ rings,
    unsigned int n_rings,
    const unsigned long long* __restrict__ windows,
    unsigned int n_windows,
    const unsigned int* __restrict__ expected,
    unsigned int* __restrict__ receipts
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const unsigned int step = expected[0];
    unsigned int failures = expected[3] <= 1u ? 0u : 128u;
    for (unsigned int i = 0; i < n_sources; ++i) {
        const unsigned long long* s = sources + 5u * i;
        const int count = *reinterpret_cast<const int*>(s[0]);
        const int cursor = *reinterpret_cast<const int*>(s[1]);
        const unsigned int failure = *reinterpret_cast<const unsigned int*>(s[2]);
        const unsigned int actual_step = reinterpret_cast<const ProtocolState*>(s[3])->current_step;
        unsigned int bad = 0u;
        if (count < 0 || (unsigned long long)count > s[4]) bad |= 1u;
        if (cursor < 0 || cursor != count) bad |= 2u;
        if (failure != 0u) bad |= 4u;
        if ((unsigned int)count != expected[4u + i]) bad |= 8u;
        if (actual_step != step) bad |= 16u;
        unsigned int* out = receipts + 8u * i;
        out[0] = step; out[1] = actual_step; out[2] = expected[4u + i];
        out[3] = (unsigned int)count; out[4] = (unsigned int)cursor;
        out[5] = bad; out[6] = expected[1]; out[7] = expected[2];
        failures |= bad;
    }
    for (unsigned int i = 0; i < n_rings; ++i) {
        const unsigned long long* r = rings + 3u * i;
        if (*reinterpret_cast<const unsigned int*>(r[0]) != *reinterpret_cast<const unsigned int*>(r[1]) ||
            *reinterpret_cast<const unsigned int*>(r[2]) != 0u) failures |= 32u;
    }
    for (unsigned int i = 0; i < n_windows; ++i) {
        const RingAdaptationWindow* w = reinterpret_cast<const RingAdaptationWindow*>(windows[i]);
        if (w->failure != 0u || w->active != 0u || w->step != step) failures |= 64u;
    }
    receipts[8u * n_sources] = failures;
    __threadfence_system();
    // A failed ownership handoff must stop every downstream physics/observer
    // node; merely writing a failure flag would allow unvalidated MD to advance.
    // No source counter/cursor is changed on this path. CUDA launch failure is
    // propagated by the existing checked fence and owner quarantine.
    if (failures != 0u) { asm volatile("trap;"); return; }
    if (expected[3] != 0u) {
        for (unsigned int i = 0; i < n_sources; ++i) {
            const unsigned long long* s = sources + 5u * i;
            *reinterpret_cast<int*>(s[0]) = 0;
            *reinterpret_cast<int*>(s[1]) = 0;
        }
    }
}
