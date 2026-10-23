// ═══════════════════════════════════════════════════════════════════════════════
// PRISM-TWIN v3.0 — ProtocolState shared header
// ═══════════════════════════════════════════════════════════════════════════════
//
// Included by protocol_director.cu AND nhs_amber_fused.cu.
// Rust side must match with #[repr(C)] — see protocol_state.rs.
//
// Layout: 716 bytes, all fields naturally 4-byte aligned.
// ═══════════════════════════════════════════════════════════════════════════════

#ifndef PROTOCOL_STATE_CUH
#define PROTOCOL_STATE_CUH

struct ProtocolState {
    // ── Step tracking ──
    unsigned int current_step;       // incremented by Director each iteration
    unsigned int total_steps;        // simulation length (immutable after init)

    // ── Temperature (CCNS 5-phase) ──
    float current_temperature;       // computed by Director based on phase
    float start_temp;                // cold temperature (K), e.g. 50
    float end_temp;                  // warm temperature (K), e.g. 300

    // ── Phase boundaries (in AMBER steps, set once at init) ──
    int cold_hold_end;               // = cold_hold_steps
    int ramp_end;                    // = cold_hold_end + ramp_steps
    int warm_hold_end;               // = ramp_end + warm_hold_steps
    int ramp_down_end;               // = warm_hold_end + ramp_down_steps
    // cold_return: everything after ramp_down_end until total_steps

    // ── UV burst state ──
    int uv_burst_active;             // 0 or 1 (written by Director)
    float uv_burst_energy;           // kcal/mol (base energy, immutable after init)
    float uv_wavelength_nm;          // current wavelength (written by Director)
    int uv_burst_interval;           // steps between bursts (immutable)
    int uv_burst_duration;           // steps per burst (immutable)
    int uv_target_idx;               // current aromatic target index (written by Director)

    // ── Wavelength scan table ──
    float scan_wavelengths[8];       // up to 8 wavelengths (keeps HIS/BNZ channels)
    int n_wavelengths;               // number of active wavelengths (1-8)
    int wavelength_dwell_steps;      // steps per wavelength before hopping

    // ── Langevin parameters ──
    float dt;                        // timestep (ps) — updated by Director if adaptive_dt
    float gamma;                     // STATIC base friction coefficient (ps^-1)
    float cutoff;                    // nonbonded cutoff (A) — immutable

    // ── Fused step tracking ──
    int fused_inner_steps;           // --fused-steps value
    int fused_step_counter;          // 0..fused_inner_steps (written by Director)

    // ════════════════════════════════════════════════════════════════════════
    // Gate 1 expansion: dynamic thermodynamic state (32 bytes)
    // ════════════════════════════════════════════════════════════════════════

    // ── Friction computation (Director computes effective_gamma each step) ──
    float gamma_base;                // base friction coefficient (immutable)
    int cryo_enabled;                // 0 or 1 (immutable)
    int equilibration_steps;         // steps for initial high-friction equilibration (immutable)
    float equilibration_gamma;       // extreme friction for equilibration (immutable)
    float effective_gamma;           // COMPUTED by Director: cryo-scaled + equilibration blend

    // ── Adaptive timestep ──
    int adaptive_dt_enabled;         // immutable:0 off,1 bounded,2 explicit historicalMay holds1.5base/rampsbase
    float base_dt;                   // base timestep before adaptive scaling (immutable)

    // ── Multi-LIF coupling double-buffer phase ──
    int coupling_phase;              // 0 or 1, toggled at each multi-LIF observation

    // ════════════════════════════════════════════════════════════════════════
    // Gate 2 expansion: autonomous housekeeping + heartbeat (12 bytes)
    // ════════════════════════════════════════════════════════════════════════

    int nl_rebuild_interval;         // steps between neighbor list rebuilds (immutable, default 20)
    int com_removal_interval;        // steps between COM velocity removal (immutable, default 100)
    int status_code;                 // 0=OK, 1=NaN detected, 2=system diverged (written by heartbeat)

    // ════════════════════════════════════════════════════════════════════════
    // ASC Fusion Controller hooks (16 bytes)
    // ════════════════════════════════════════════════════════════════════════

    // Steering command block — written by the interferometric bridge (read_and_adapt)
    // when the coupling kernel detects a high-value signal pattern.
    // Read by the Director to modulate UV energy, temperature bias, or dt.
    float steering_uv_boost;         // multiplicative UV energy adjustment (1.0 = neutral)
    float steering_temp_bias;        // additive temperature offset (K, 0.0 = neutral)
    int steering_focus_residue;      // residue ID to focus UV on (-1 = no focus)
    int steering_flags;              // bit flags: 0x1=phase_lock, 0x2=cryo_stabilize, 0x4=adaptive_slow,
                                     // 0x8=uv_burst_suppressed_declared (P2; host-written at upload,
                                     // never written or cleared by any kernel; see protocol_state.rs)

    // ════════════════════════════════════════════════════════════════════════
    // Phase-coherence metrology (4 bytes)
    // ════════════════════════════════════════════════════════════════════════

    unsigned int current_phase_bits; // 10-bit CCNS phase angle (0-1023), updated by Director each step

    // ════════════════════════════════════════════════════════════════════════
    // Stage 2: Closed-loop ASC steering — focus residues with weights (516 B)
    // ════════════════════════════════════════════════════════════════════════
    //
    // The ASC controller writes the top-K residues by GC-PID synergy_fraction
    // (from Stage 1B-3 — see crates/prism-nhs/src/gcpid.rs and the per-run
    // {prefix}.gcpid_synergy.json export) into this fixed-size array each
    // chunk via cuMemcpyHtoDAsync on the engine stream. The
    // ring_buffer_read_and_adapt kernel reads it on its next launch and
    // applies a multiplicative threshold-reduction boost to spikes whose
    // primary residue id is in the focus list.
    //
    // The captured CUDA Graph holds a POINTER to this ProtocolState, not a
    // value. The contents change between graph replays without requiring
    // graph re-capture — this is the canonical "captured pointer with
    // mutable contents" pattern documented at length in
    // graph_capture.rs and persistent_engine.rs::kcc_step_once.
    //
    // ## Why we steer toward synergy_fraction residues, NOT pocket residues
    //
    // Stage 1B-3 validation on 4LPK showed that the SII signature pocket
    // residues (the canonical detection targets) have HIGH total mutual
    // information about future spike rate but LOW synergy_fraction —
    // their information is mostly REDUNDANT across the four TWIN groups
    // (every group sees the same pocket the same way). The high-
    // synergy_fraction residues are DIFFERENT residues (152, 45, 65,
    // 105, ...) where the joint distribution of (scout, observer)
    // groups carries information that no single group has alone.
    // These are the cross-group COORDINATION points — the leverage
    // points for steering. Steering pocket residues that every group
    // already agrees on produces no divergence. Steering coordination
    // points forces groups into states they wouldn't reach
    // independently — which is precisely what the ACL contrast metric
    // is supposed to measure but currently can't because the loop is
    // open. See Stage 1B-2+1B-3 commit b8aeff61 for the full
    // theoretical justification.
    //
    // ## Layout
    //
    // The capacity of 64 is enough to cover the cryptic-pocket-relevant
    // surface of any reasonable target (the largest benchmark protein has
    // ~500 residues; even if a quarter are coordination centers, 64 is
    // a generous slice). Inactive slots have residue_id = -1.

    int steering_focus_count;       // number of active entries in the array (0..64)
    struct SteerEntry {
        int residue_id;             // -1 = inactive slot
        float weight;               // GC-PID synergy_fraction in [0,1]; 0 = inactive
    } steering_focus_residues[64];  // 64 × 8 bytes = 512 bytes

    // ── Stage 2 calibration counters (12 bytes) ──
    // atomicAdd'd by ring_buffer_read_and_adapt every time a spike's
    // primary residue id matches an active entry in steering_focus_residues.
    unsigned int focus_match_count;
    // atomicAdd'd for every spike processed in the inner loop. Distinguishes
    // "kernel never runs" from "kernel runs but matches fail."
    unsigned int processed_spike_count;
    // Kernel writes steering_focus_residues[0].residue_id here every launch.
    // Catches a struct layout mismatch (host writes one offset, kernel reads
    // another).
    int last_seen_focus_id;
    // Kernel writes the last spike's primary_residue_id & 0xFFFF here every
    // launch. Catches an encoding mismatch between the residue ID space the
    // ASC controller uses and the one the spike pipeline produces.
    int last_seen_spike_residue;
};

// Struct size:
//   164 (Gates 0-2 + expanded scan table) + 16 (legacy ASC hooks) + 4 (phase) = 184
//   + 4 (steering_focus_count) + 512 (steering_focus_residues[64]) = 700
//   + 4 (focus_match_count) + 4 (processed_spike_count)
//   + 4 (last_seen_focus_id) + 4 (last_seen_spike_residue) = 716 bytes

// Physical steps and observation cadence share the existing device counter;
// graph launch/chunk boundaries are not observation boundaries.
#if defined(__CUDACC__)
__host__ __device__
#endif
inline int prism_protocol_next_observation_counter(int counter, int inner_steps) {
    return inner_steps > 1 ? (counter + 1) % inner_steps : 0;
}

#if defined(__CUDACC__)
__host__ __device__
#endif
inline bool prism_protocol_observation_due(const ProtocolState* state) {
    return state->fused_step_counter == 0;
}

#endif // PROTOCOL_STATE_CUH
