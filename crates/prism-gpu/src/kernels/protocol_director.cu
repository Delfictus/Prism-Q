// ═══════════════════════════════════════════════════════════════════════════════
// PRISM-TWIN v3.0 — Protocol Director Kernel
// ═══════════════════════════════════════════════════════════════════════════════
//
// Purpose: GPU-resident protocol state machine for CUDA Graph execution.
//
// The Director kernel runs BEFORE physics kernels on the SAME stream.
// CUDA stream ordering guarantees the Director's writes to ProtocolState
// are visible to the physics kernels — no explicit synchronization needed.
//
// Gate 0: ProtocolState struct + Director kernel (standalone validation)
// Gate 1: Physics kernel reads from ProtocolState (replaces scalar params)
// Gate 2: Full CUDA Graph capture (Director -> Physics -> Compact -> Adapt)
//
// ═══════════════════════════════════════════════════════════════════════════════

#include "protocol_state.cuh"
#include "nhs_biprism_shared_step_abi.cuh"

// CUDA graph conditional handles are a captured-graph optimization surface.
// They first appear in CUDA 12.4. PrismOS beta hardware acceptance can build
// the complete PTX bundle with an older CUDA 12 toolkit so the laptop driver
// receives compatible PTX. In that case keep the protocol_director_graph symbol
// and the director state-machine semantics loadable, but compile the
// conditional-handle writes as no-ops; callers that require conditional graph
// execution already gate that path at runtime.
#if !defined(CUDART_VERSION) || CUDART_VERSION < 12040
typedef unsigned long long cudaGraphConditionalHandle;
static __device__ __forceinline__ void prism_protocol_director_set_conditional_compat(
    cudaGraphConditionalHandle,
    unsigned int
) {}
#define cudaGraphSetConditional prism_protocol_director_set_conditional_compat
#endif

// Reference temperature for cryogenic friction scaling (K)
#define T_REF 300.0f
// Minimum temperature clamp to prevent division by zero
#define T_MIN 10.0f
// Must match protocol_state.rs; no field or device ABI offset changes.
#define PRISM_WARM_TAIL_WITHOUT_COOLING_FLAG 0x10

// ─────────────────────────────────────────────────────────────────────────────
// protocol_director — single-thread state machine, runs BEFORE physics each step
// ─────────────────────────────────────────────────────────────────────────────
//
// Launch: <<<1, 1>>> — one thread, one block.
// Cost: ~20 ns per invocation (negligible vs physics kernel).
//
// Updates: current_step, current_temperature, effective_gamma, dt,
//          uv_burst_active, uv_wavelength_nm, uv_target_idx,
//          fused_step_counter, coupling_phase.

// Shared ordinary Director evaluation at one supplied physical step. Stateful
// observation counters and steering hints advance exactly once per call.
// Factored without reordering the ordinary baseline or live-modifier arithmetic.
// Existing exports call both parts once at their original step.
static __device__ __forceinline__ void prism_protocol_baseline_v1(
    ProtocolState* state,unsigned int step
) {
    int s = (int)step;

    // ══════════════════════════════════════════════════════════════════════════
    // CCNS 5-Phase Temperature State Machine
    // ══════════════════════════════════════════════════════════════════════════
    //
    // Phase 1: cold_hold    [0, cold_hold_end)             → start_temp
    // Phase 2: ramp_up      [cold_hold_end, ramp_end)      → start_temp → end_temp
    // Phase 3: warm_hold    [ramp_end, warm_hold_end)      → end_temp
    // Phase 4: ramp_down    [warm_hold_end, ramp_down_end) → end_temp → start_temp
    // Phase 5: cold_return  [ramp_down_end, ...)           → start_temp
    // Without cooling phases, the requested warm temperature persists.

    int is_hold_phase;  // needed for adaptive dt

    if (s < state->cold_hold_end) {
        state->current_temperature = state->start_temp;
        is_hold_phase = 1;
    } else if (s < state->ramp_end) {
        int ramp_length = state->ramp_end - state->cold_hold_end;
        float progress = (float)(s - state->cold_hold_end) / (float)ramp_length;
        state->current_temperature = state->start_temp
            + progress * (state->end_temp - state->start_temp);
        is_hold_phase = 0;
    } else if (s < state->warm_hold_end) {
        state->current_temperature = state->end_temp;
        is_hold_phase = 1;
    } else if (s < state->ramp_down_end) {
        int ramp_length = state->ramp_down_end - state->warm_hold_end;
        float progress = (float)(s - state->warm_hold_end) / (float)ramp_length;
        state->current_temperature = state->end_temp
            - progress * (state->end_temp - state->start_temp);
        is_hold_phase = 0;
    } else {
        state->current_temperature =
            (state->steering_flags & PRISM_WARM_TAIL_WITHOUT_COOLING_FLAG)
            ? state->end_temp : state->start_temp;
        is_hold_phase = 1;
    }

    float current_temp = state->current_temperature;

    // ══════════════════════════════════════════════════════════════════════════
    // Effective Gamma (Langevin friction) — matches compute_cryo_friction()
    // ══════════════════════════════════════════════════════════════════════════
    //
    // Two components:
    // 1. Equilibration boost: exponential decay from equilibration_gamma to
    //    gamma_base over the first equilibration_steps (typically 10000).
    //    Critical for structures that haven't been energy-minimized.
    // 2. Cryogenic scaling: at T < T_REF, scale by sqrt(T_REF / T) to
    //    slow dynamics proportionally.

    float base_gamma;
    if (s < state->equilibration_steps) {
        float progress = (float)s / (float)state->equilibration_steps;
        float decay = expf(-3.0f * progress);
        base_gamma = state->equilibration_gamma * decay
                   + state->gamma_base * (1.0f - decay);
    } else {
        base_gamma = state->gamma_base;
    }

    if (state->cryo_enabled && current_temp < T_REF) {
        float t_clamped = fmaxf(current_temp, T_MIN);
        float scale = T_REF / t_clamped;
        state->effective_gamma = base_gamma * sqrtf(scale);
    } else {
        state->effective_gamma = base_gamma;
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Adaptive Timestep — matches adaptive_dt logic in fused_engine.rs
    // ══════════════════════════════════════════════════════════════════════════
    //
    // Never exceed the topology-admitted base timestep. During ramps use a
    // conservative half-step; during holds return to base_dt.

    if (state->adaptive_dt_enabled) {
        state->dt = state->adaptive_dt_enabled == 2
            ? (is_hold_phase ? state->base_dt * 1.5f : state->base_dt)
            : (is_hold_phase ? state->base_dt : state->base_dt * 0.5f);
    }

    // ══════════════════════════════════════════════════════════════════════════
    // UV Burst Scheduling
    // ══════════════════════════════════════════════════════════════════════════

    if (state->uv_burst_interval > 0) {
        int burst_cycle = s % state->uv_burst_interval;
        state->uv_burst_active = (burst_cycle < state->uv_burst_duration) ? 1 : 0;

        int nwl = state->n_wavelengths;
        if (nwl > 8) nwl = 8;
        if (nwl < 0) nwl = 0;
        if (nwl > 0 && state->wavelength_dwell_steps > 0) {
            int wl_idx = (s / state->wavelength_dwell_steps) % nwl;
            state->uv_wavelength_nm = state->scan_wavelengths[wl_idx];
            state->uv_target_idx = wl_idx;
        }
    } else {
        state->uv_burst_active = 0;
    }

}
static __device__ __forceinline__ void prism_protocol_live_modifiers_v1(
    ProtocolState* state,unsigned int step,unsigned int control_step
) {
    // ══════════════════════════════════════════════════════════════════════════
    // Fused Step Counter
    // ══════════════════════════════════════════════════════════════════════════

    state->fused_step_counter = prism_protocol_next_observation_counter(
        state->fused_step_counter, state->fused_inner_steps);

    // ══════════════════════════════════════════════════════════════════════════
    // ASC Steering — read coupling-written hints and apply adjustments
    // ══════════════════════════════════════════════════════════════════════════
    //
    // The coupling kernel (read_and_adapt) writes steering_uv_boost and
    // steering_focus_residue when it detects a high-value signal pattern.
    // We apply the modulation here and then decay toward neutral.

    if (state->steering_uv_boost != 1.0f) {
        // Apply UV boost: scale the burst energy via the protocol state.
        // Physics kernels read uv_burst_active from ProtocolState —
        // we can't modify uv_burst_energy directly (it's a protocol param),
        // but we can extend burst duration for the focused residue.
        // Decay: multiplicatively toward 1.0 with 0.5% per step.
        float decay = 0.005f;
        state->steering_uv_boost = 1.0f +
            (state->steering_uv_boost - 1.0f) * (1.0f - decay);
        if (fabsf(state->steering_uv_boost - 1.0f) < 0.01f) {
            state->steering_uv_boost = 1.0f;
        }
    }

    if (state->steering_temp_bias != 0.0f) {
        // Apply temperature bias: additive on current_temperature.
        state->current_temperature += state->steering_temp_bias;
        // Decay toward 0 with 1% per step.
        state->steering_temp_bias *= 0.99f;
        if (fabsf(state->steering_temp_bias) < 0.1f) {
            state->steering_temp_bias = 0.0f;
        }
    }

    // Clear phase_lock flag after 500 steps of steering
    if (state->steering_flags & 0x1) {
        // Phase lock decays after ~500 steps (cleared by fused step counter wrap)
        if (step % 500 == 0) {
            state->steering_flags &= ~0x1;
            state->steering_focus_residue = -1;
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Phase-Coherence Metrology — 10-bit CCNS phase angle
    // ══════════════════════════════════════════════════════════════════════════
    // Maps current_step within the full CCNS cycle to 0-1023.
    // Downstream phasor math uses atan2(sin_sum, cos_sum) to compute S_pc.
    // Spikes at different thermal phases get different phase_bits → real pocket
    // signals show coherent lag; temperature artifacts show uniform phase.

    state->current_phase_bits = (unsigned int)((float)control_step / (float)state->total_steps * 1024.0f) & 0x3FF;

    // ══════════════════════════════════════════════════════════════════════════
    // Coupling Phase Toggle (multi-LIF double-buffer swap)
    // ══════════════════════════════════════════════════════════════════════════

    if (prism_protocol_observation_due(state)) {
        state->coupling_phase = 1 - state->coupling_phase;
    }
}

static __device__ __forceinline__ void prism_protocol_evaluate_step_v1(
    ProtocolState* state,unsigned int step
) {
    prism_protocol_baseline_v1(state,step);
    prism_protocol_live_modifiers_v1(state,step,step);
}

// Baselines are sampled before endogenous ASC effects. The original fixed
// request is owned separately, so common-min publication cannot feed back into
// the next mode0 request. Terminal continuation never snapshots steering hints.
static __device__ __forceinline__ void prism_protocol_capture_baseline_v1(
    const ProtocolState* s,uint32_t leaf,uint32_t control_step,
    BiprismTerminalControlsFfi* out
) {
    *out={BIPRISM_SHARED_STEP_MAGIC,leaf,s->total_steps,control_step,
        s->current_temperature,s->effective_gamma,s->dt,s->uv_wavelength_nm,
        s->uv_burst_active,s->uv_target_idx,
        (uint32_t)((float)control_step/(float)s->total_steps*1024.0f)&0x3ffu,0u};
}
static __device__ __forceinline__ bool prism_protocol_biprism_request_v1(
    ProtocolState* state,uint32_t step,BiprismTerminalControlsFfi* terminal,
    BiprismTerminalControlsFfi* baseline,float original_fixed_request_ps,
    uint32_t leaf,BiprismSharedStepFfi* shared
) {
    if(!state||!terminal||!baseline||!shared||leaf>=8u||!step||step==UINT32_MAX
        ||!state->total_steps||state->total_steps>0x7fffffffu
        ||shared->abi_magic!=BIPRISM_SHARED_STEP_MAGIC||!shared->generation||shared->sticky_fault
        ||shared->policy!=BIPRISM_SHARED_STEP_COMMON_MIN
        ||(shared->state!=0u&&shared->state!=BIPRISM_SHARED_STEP_ACCEPTED)
        ||shared->last_accepted_step+1u!=step||state->current_step+1u!=step
        ||terminal->abi_magic!=BIPRISM_SHARED_STEP_MAGIC||terminal->leaf_id!=leaf
        ||terminal->terminal_step!=state->total_steps||terminal->reserved
        ||!isfinite(original_fixed_request_ps)||original_fixed_request_ps<=0.f
        ||(state->adaptive_dt_enabled!=0&&state->adaptive_dt_enabled!=1))return false;
    const uint32_t control_step=step<state->total_steps?step:state->total_steps;
    if(step<=state->total_steps) {
        if(terminal->captured_step)return false;
        if(state->adaptive_dt_enabled==0)state->dt=original_fixed_request_ps;
        prism_protocol_baseline_v1(state,step);
        prism_protocol_capture_baseline_v1(state,leaf,control_step,baseline);
        if(step==state->total_steps)*terminal=*baseline;
    } else {
        if(terminal->captured_step!=state->total_steps)return false;
        state->current_temperature=terminal->temperature;
        state->effective_gamma=terminal->effective_gamma;
        state->dt=terminal->requested_dt_ps;
        state->uv_wavelength_nm=terminal->uv_wavelength_nm;
        state->uv_burst_active=terminal->uv_burst_active;
        state->uv_target_idx=terminal->uv_target_idx;
        *baseline=*terminal;
    }
    if(!isfinite(state->current_temperature)||state->current_temperature<=0.f
        ||!isfinite(state->effective_gamma)||state->effective_gamma<0.f
        ||!isfinite(state->dt)||state->dt<=0.f||!isfinite(state->uv_wavelength_nm)
        ||(state->uv_burst_active!=0&&state->uv_burst_active!=1))return false;
    state->current_step=step;
    prism_protocol_live_modifiers_v1(state,step,control_step);
    return true;
}

extern "C" __global__ void protocol_director_biprism_selected_v1(
    ProtocolState* state,const uint32_t* scheduled_step,
    BiprismTerminalControlsFfi* terminal,BiprismTerminalControlsFfi* baseline,
    float original_fixed_request_ps,uint32_t leaf,BiprismSharedStepFfi* shared
) {
    if(blockIdx.x||threadIdx.x)return;
    if(!scheduled_step||!prism_protocol_biprism_request_v1(state,*scheduled_step,
        terminal,baseline,original_fixed_request_ps,leaf,shared)) {
        if(shared){shared->sticky_fault|=1u;shared->state=3u;}
    }
}

extern "C" __global__ void protocol_director(ProtocolState* __restrict__ state) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    state->current_step += 1;
    prism_protocol_evaluate_step_v1(state, state->current_step);
}

// Explicit Biprism owner: the collective word is a scheduled step, not a second
// native completed-step clock. The caller supplies stream ordering and lifetime.
// No restrict qualifier: the word may alias state->current_step.
extern "C" __global__ void protocol_director_biprism_v1(
    ProtocolState* state, const unsigned int* collective_step
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (collective_step == nullptr) { asm volatile("trap;"); return; }
    const unsigned int step = *collective_step;
    state->current_step = step;
    prism_protocol_evaluate_step_v1(state, step);
}

// ─────────────────────────────────────────────────────────────────────────────
// protocol_director_graph — Director variant for CUDA Graph conditional nodes
// ─────────────────────────────────────────────────────────────────────────────
//
// Identical to protocol_director but additionally writes conditional handles
// for COM removal and heartbeat triggers. Used inside captured graphs.
//
// The handles are CUgraphConditionalHandle values created by the host.
// cudaGraphSetConditional(handle, 1) triggers the conditional body graph;
// cudaGraphSetConditional(handle, 0) skips it.
//
// Launch: <<<1, 1>>>

extern "C" __global__ void protocol_director_graph(
    ProtocolState* __restrict__ state,
    cudaGraphConditionalHandle com_handle,
    cudaGraphConditionalHandle heartbeat_handle
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // ── All standard Director logic (duplicated for graph variant) ──
    state->current_step += 1;
    unsigned int step = state->current_step;
    int s = (int)step;

    // Temperature state machine
    int is_hold_phase;
    if (s < state->cold_hold_end) {
        state->current_temperature = state->start_temp;
        is_hold_phase = 1;
    } else if (s < state->ramp_end) {
        int ramp_length = state->ramp_end - state->cold_hold_end;
        float progress = (float)(s - state->cold_hold_end) / (float)ramp_length;
        state->current_temperature = state->start_temp
            + progress * (state->end_temp - state->start_temp);
        is_hold_phase = 0;
    } else if (s < state->warm_hold_end) {
        state->current_temperature = state->end_temp;
        is_hold_phase = 1;
    } else if (s < state->ramp_down_end) {
        int ramp_length = state->ramp_down_end - state->warm_hold_end;
        float progress = (float)(s - state->warm_hold_end) / (float)ramp_length;
        state->current_temperature = state->end_temp
            - progress * (state->end_temp - state->start_temp);
        is_hold_phase = 0;
    } else {
        state->current_temperature =
            (state->steering_flags & PRISM_WARM_TAIL_WITHOUT_COOLING_FLAG)
            ? state->end_temp : state->start_temp;
        is_hold_phase = 1;
    }

    float current_temp = state->current_temperature;

    // Effective gamma
    float base_gamma;
    if (s < state->equilibration_steps) {
        float progress = (float)s / (float)state->equilibration_steps;
        float decay = expf(-3.0f * progress);
        base_gamma = state->equilibration_gamma * decay
                   + state->gamma_base * (1.0f - decay);
    } else {
        base_gamma = state->gamma_base;
    }
    if (state->cryo_enabled && current_temp < T_REF) {
        float t_clamped = fmaxf(current_temp, T_MIN);
        state->effective_gamma = base_gamma * sqrtf(T_REF / t_clamped);
    } else {
        state->effective_gamma = base_gamma;
    }

    // Adaptive dt
    if (state->adaptive_dt_enabled) {
        state->dt = state->adaptive_dt_enabled == 2
            ? (is_hold_phase ? state->base_dt * 1.5f : state->base_dt)
            : (is_hold_phase ? state->base_dt : state->base_dt * 0.5f);
    }

    // UV burst scheduling
    if (state->uv_burst_interval > 0) {
        int burst_cycle = s % state->uv_burst_interval;
        state->uv_burst_active = (burst_cycle < state->uv_burst_duration) ? 1 : 0;
        int nwl = state->n_wavelengths;
        if (nwl > 8) nwl = 8;
        if (nwl < 0) nwl = 0;
        if (nwl > 0 && state->wavelength_dwell_steps > 0) {
            int wl_idx = (s / state->wavelength_dwell_steps) % nwl;
            state->uv_wavelength_nm = state->scan_wavelengths[wl_idx];
            state->uv_target_idx = wl_idx;
        }
    } else {
        state->uv_burst_active = 0;
    }

    // Fused step counter
    state->fused_step_counter = prism_protocol_next_observation_counter(
        state->fused_step_counter, state->fused_inner_steps);

    // Phase-coherence metrology (same as standard Director)
    state->current_phase_bits = (unsigned int)((float)step / (float)state->total_steps * 1024.0f) & 0x3FF;

    // Coupling phase toggle
    if (prism_protocol_observation_due(state)) {
        state->coupling_phase = 1 - state->coupling_phase;
    }

    // ── Conditional handle triggers ──
    // COM removal: fire every com_removal_interval steps
    if (state->com_removal_interval > 0) {
        cudaGraphSetConditional(com_handle,
            (s > 0 && (s % state->com_removal_interval) == 0) ? 1 : 0);
    } else {
        cudaGraphSetConditional(com_handle, 0);
    }

    // Heartbeat: check every 1000 steps (lightweight sampling)
    cudaGraphSetConditional(heartbeat_handle,
        (s > 0 && (s % 1000) == 0) ? 1 : 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// init_protocol_state — initialize from host-provided values
// ─────────────────────────────────────────────────────────────────────────────
//
// Launch: <<<1, 1>>>

extern "C" __global__ void init_protocol_state(
    ProtocolState* __restrict__ state,
    unsigned int total_steps,
    float start_temp,
    float end_temp,
    int cold_hold_end,
    int ramp_end,
    int warm_hold_end,
    int ramp_down_end,
    float uv_burst_energy,
    int uv_burst_interval,
    int uv_burst_duration,
    float wl0, float wl1, float wl2, float wl3,
    float wl4, float wl5, float wl6, float wl7,
    int n_wavelengths,
    int wavelength_dwell_steps,
    float dt,
    float gamma,
    float cutoff,
    int fused_inner_steps,
    // Gate 1 expansion params
    float gamma_base,
    int cryo_enabled,
    int equilibration_steps,
    float equilibration_gamma,
    int adaptive_dt_enabled,
    float base_dt,
    int coupling_phase
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    state->current_step = 0;
    state->total_steps = total_steps;
    state->current_temperature = start_temp;
    state->start_temp = start_temp;
    state->end_temp = end_temp;

    state->cold_hold_end = cold_hold_end;
    state->ramp_end = ramp_end;
    state->warm_hold_end = warm_hold_end;
    state->ramp_down_end = ramp_down_end;

    state->uv_burst_active = 0;
    state->uv_burst_energy = uv_burst_energy;
    state->uv_wavelength_nm = wl0;
    state->uv_burst_interval = uv_burst_interval;
    state->uv_burst_duration = uv_burst_duration;
    state->uv_target_idx = 0;

    state->scan_wavelengths[0] = wl0;
    state->scan_wavelengths[1] = wl1;
    state->scan_wavelengths[2] = wl2;
    state->scan_wavelengths[3] = wl3;
    state->scan_wavelengths[4] = wl4;
    state->scan_wavelengths[5] = wl5;
    state->scan_wavelengths[6] = wl6;
    state->scan_wavelengths[7] = wl7;
    state->n_wavelengths = n_wavelengths > 8 ? 8 : (n_wavelengths < 0 ? 0 : n_wavelengths);
    state->wavelength_dwell_steps = wavelength_dwell_steps;

    state->dt = dt;
    state->gamma = gamma;
    state->cutoff = cutoff;

    state->fused_inner_steps = fused_inner_steps;
    state->fused_step_counter = 0;

    // Gate 1 expansion
    state->gamma_base = gamma_base;
    state->cryo_enabled = cryo_enabled;
    state->equilibration_steps = equilibration_steps;
    state->equilibration_gamma = equilibration_gamma;
    state->effective_gamma = gamma_base;  // initial value before first Director step
    state->adaptive_dt_enabled = adaptive_dt_enabled;
    state->base_dt = base_dt;
    state->coupling_phase = coupling_phase;
}

// ─────────────────────────────────────────────────────────────────────────────
// read_protocol_state — diagnostic snapshot
// ─────────────────────────────────────────────────────────────────────────────
//
// Launch: <<<1, 1>>>

extern "C" __global__ void read_protocol_state(
    const ProtocolState* __restrict__ state,
    unsigned int* out_current_step,
    float* out_current_temperature,
    int* out_uv_burst_active,
    float* out_uv_wavelength_nm
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    *out_current_step = state->current_step;
    *out_current_temperature = state->current_temperature;
    *out_uv_burst_active = state->uv_burst_active;
    *out_uv_wavelength_nm = state->uv_wavelength_nm;
}

// Capability witness only; loaded (never launched) for explicit policy2.
extern "C" __global__ void protocol_director_historical_may_policy_v1() {}
