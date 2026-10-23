//! PRISM-NHS Adaptive Exploration Protocol
//!
//! Three-phase cryptic site detection:
//! 
//! Phase 1: CRYO BURST (aggressive exploration)
//!   - Frozen landscape (80-100K)
//!   - High-intensity UV sweep across all aromatics
//!   - Map ALL potential hot spots with clean signal
//!   - No thermal noise to mask weak resonances
//!
//! Phase 2: THERMAL RAMP (validation)
//!   - Gradual warming (100K → 300K)
//!   - Focus probes on Phase 1 candidates
//!   - Watch which hot spots "wake up"
//!   - Filter artifacts from real pockets
//!
//! Phase 3: FOCUSED DIG (exploitation)
//!   - Physiological temperature (300K)
//!   - Concentrate on validated candidates
//!   - Deep resonance analysis
//!   - Confirm pocket opening dynamics
//!
//! This mimics experimental biophysics:
//! - Cryo-EM gives static snapshots
//! - Temperature-jump reveals dynamics
//! - Focused NMR/HDX confirms mechanism

#ifndef PRISM_ADAPTIVE_PROTOCOL_CUH
#define PRISM_ADAPTIVE_PROTOCOL_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>

// ============================================================================
// PROTOCOL CONFIGURATION
// ============================================================================

// Phase definitions
#define PHASE_CRYO_BURST    0
#define PHASE_THERMAL_RAMP  1
#define PHASE_FOCUSED_DIG   2

// Default phase durations (steps)
#define DEFAULT_CRYO_STEPS      100
#define DEFAULT_RAMP_STEPS      400
#define DEFAULT_DIG_STEPS       9500  // Remainder of 10000

// Temperature settings
#define CRYO_TEMP           80.0f   // Kelvin - liquid nitrogen range
#define RAMP_START_TEMP     80.0f
#define RAMP_END_TEMP       300.0f
#define PHYSIOLOGICAL_TEMP  300.0f

// UV probe intensities by phase
#define CRYO_UV_INTENSITY   10.0f   // High - blast everything
#define RAMP_UV_INTENSITY   5.0f    // Moderate - track response
#define DIG_UV_INTENSITY    2.0f    // Low - gentle probing

// Exploration parameters
#define CRYO_EXPLORATION_RADIUS     50.0f  // Angstroms - search everywhere
#define RAMP_EXPLORATION_RADIUS     20.0f  // Focus on candidates
#define DIG_EXPLORATION_RADIUS      8.0f   // Tight focus

// Confidence thresholds for phase transitions
#define MIN_CANDIDATES_FOR_RAMP     3      // Need at least 3 hot spots
#define MIN_CONFIDENCE_FOR_DIG      0.5f   // 50% confidence to proceed
#define HOT_SPOT_THRESHOLD          0.3f   // Spike response threshold

// ============================================================================
// DATA STRUCTURES
// ============================================================================

// Hot spot candidate from cryo phase
struct HotSpotCandidate {
    float3 center;              // Spatial center
    float confidence;           // 0-1 confidence score
    float resonance_frequency;  // Dominant soft mode (THz)
    float spike_density;        // Spikes per probe per volume
    int aromatic_count;         // Nearby aromatics
    int phase_discovered;       // Which phase found this
    int validation_count;       // How many times validated
    bool is_active;             // Still being tracked
    
    // Aromatic contributors
    int contributing_aromatics[8];
    int n_contributors;
    
    // History for temporal tracking
    float confidence_history[16];
    int history_idx;
};

// Adaptive protocol state
struct AdaptiveProtocolState {
    // Current phase
    int current_phase;
    int phase_step;             // Steps within current phase
    int total_step;             // Total simulation steps
    
    // Phase durations (configurable)
    int cryo_steps;
    int ramp_steps;
    int dig_steps;
    
    // Temperature control
    float current_temp;
    float target_temp;
    float temp_ramp_rate;       // K per step
    
    // UV probe control
    float current_uv_intensity;
    int probe_sweep_idx;        // Which aromatic being probed
    int probes_this_phase;      // Probes executed in current phase
    
    // Exploration control
    float exploration_radius;
    float3 exploration_center;  // Focus point for later phases
    
    // Candidate tracking
    HotSpotCandidate candidates[32];
    int n_candidates;
    int best_candidate_idx;
    
    // Phase transition flags
    bool cryo_complete;
    bool ramp_complete;
    bool dig_complete;
    bool site_confirmed;
    
    // Statistics
    int total_spikes;
    int spikes_this_phase;
    float peak_confidence;
    float3 best_site_location;
};

// Per-probe response for scoring
struct ProbeResponse {
    int aromatic_idx;
    float3 probe_center;
    float spike_count;
    float spike_intensity_sum;
    float spatial_spread;       // How spread out are the spikes
    float temporal_coherence;   // Are spikes synchronized
    float resonance_score;      // Low-frequency response
};

// ============================================================================
// DEVICE FUNCTIONS: PHASE MANAGEMENT
// ============================================================================

__device__ __forceinline__ void init_protocol_state(
    AdaptiveProtocolState* state,
    int total_steps
) {
    state->current_phase = PHASE_CRYO_BURST;
    state->phase_step = 0;
    state->total_step = 0;
    
    // Set phase durations proportionally
    state->cryo_steps = DEFAULT_CRYO_STEPS;
    state->ramp_steps = DEFAULT_RAMP_STEPS;
    state->dig_steps = total_steps - state->cryo_steps - state->ramp_steps;
    if (state->dig_steps < 0) state->dig_steps = 0;
    
    // Start cryo
    state->current_temp = CRYO_TEMP;
    state->target_temp = CRYO_TEMP;
    state->temp_ramp_rate = 0.0f;
    
    state->current_uv_intensity = CRYO_UV_INTENSITY;
    state->exploration_radius = CRYO_EXPLORATION_RADIUS;
    
    state->n_candidates = 0;
    state->best_candidate_idx = -1;
    
    state->cryo_complete = false;
    state->ramp_complete = false;
    state->dig_complete = false;
    state->site_confirmed = false;
    
    state->total_spikes = 0;
    state->spikes_this_phase = 0;
    state->peak_confidence = 0.0f;
}

__device__ __forceinline__ void advance_protocol_step(
    AdaptiveProtocolState* state
) {
    state->total_step++;
    state->phase_step++;
    
    // Check phase transitions
    if (state->current_phase == PHASE_CRYO_BURST) {
        if (state->phase_step >= state->cryo_steps) {
            // Transition to ramp if we have candidates
            if (state->n_candidates >= MIN_CANDIDATES_FOR_RAMP) {
                state->current_phase = PHASE_THERMAL_RAMP;
                state->phase_step = 0;
                state->spikes_this_phase = 0;
                state->cryo_complete = true;
                
                // Set up ramp
                state->temp_ramp_rate = (RAMP_END_TEMP - RAMP_START_TEMP) / state->ramp_steps;
                state->target_temp = RAMP_END_TEMP;
                state->current_uv_intensity = RAMP_UV_INTENSITY;
                state->exploration_radius = RAMP_EXPLORATION_RADIUS;
                
                // Focus on best candidate
                if (state->best_candidate_idx >= 0) {
                    state->exploration_center = state->candidates[state->best_candidate_idx].center;
                }
            } else {
                // Not enough candidates - extend cryo phase
                state->cryo_steps += 50;
            }
        }
    }
    else if (state->current_phase == PHASE_THERMAL_RAMP) {
        // Update temperature
        state->current_temp += state->temp_ramp_rate;
        if (state->current_temp > state->target_temp) {
            state->current_temp = state->target_temp;
        }
        
        if (state->phase_step >= state->ramp_steps) {
            // Transition to dig if we have validated candidates
            bool has_validated = false;
            for (int i = 0; i < state->n_candidates; i++) {
                if (state->candidates[i].confidence >= MIN_CONFIDENCE_FOR_DIG &&
                    state->candidates[i].validation_count >= 2) {
                    has_validated = true;
                    break;
                }
            }
            
            if (has_validated) {
                state->current_phase = PHASE_FOCUSED_DIG;
                state->phase_step = 0;
                state->spikes_this_phase = 0;
                state->ramp_complete = true;
                
                // Full physiological
                state->current_temp = PHYSIOLOGICAL_TEMP;
                state->temp_ramp_rate = 0.0f;
                state->current_uv_intensity = DIG_UV_INTENSITY;
                state->exploration_radius = DIG_EXPLORATION_RADIUS;
            } else {
                // No validated candidates - this might not be a good target
                // Continue anyway but note the concern
                state->current_phase = PHASE_FOCUSED_DIG;
                state->phase_step = 0;
                state->ramp_complete = true;
            }
        }
    }
    else if (state->current_phase == PHASE_FOCUSED_DIG) {
        // Check for site confirmation
        if (state->peak_confidence > 0.8f) {
            state->site_confirmed = true;
        }
        
        if (state->phase_step >= state->dig_steps) {
            state->dig_complete = true;
        }
    }
}

// ============================================================================
// DEVICE FUNCTIONS: CRYO BURST PHASE
// ============================================================================

__device__ __forceinline__ float get_cryo_probe_intensity(
    const AdaptiveProtocolState* state,
    int aromatic_idx,
    int n_aromatics
) {
    // In cryo phase: cycle through all aromatics with maximum intensity
    // Use high intensity to detect ALL possible resonances
    
    if (state->current_phase != PHASE_CRYO_BURST) {
        return 0.0f;
    }
    
    // Sweep pattern: hit each aromatic in sequence with bursts
    int sweep_period = max(1, state->cryo_steps / n_aromatics);
    int current_target = (state->phase_step / sweep_period) % n_aromatics;
    
    if (aromatic_idx == current_target) {
        return state->current_uv_intensity;
    }
    
    // Also probe neighbors at lower intensity for cross-talk detection
    int prev_target = (current_target - 1 + n_aromatics) % n_aromatics;
    int next_target = (current_target + 1) % n_aromatics;
    
    if (aromatic_idx == prev_target || aromatic_idx == next_target) {
        return state->current_uv_intensity * 0.3f;
    }
    
    return 0.0f;
}

__device__ __forceinline__ void record_cryo_response(
    AdaptiveProtocolState* state,
    int aromatic_idx,
    float3 spike_position,
    float spike_intensity,
    float resonance_freq
) {
    // Record spike response during cryo phase
    // Build up candidate hot spots
    
    if (spike_intensity < HOT_SPOT_THRESHOLD) return;
    
    state->total_spikes++;
    state->spikes_this_phase++;
    
    // Check if this spike is near an existing candidate
    bool found_match = false;
    for (int i = 0; i < state->n_candidates; i++) {
        HotSpotCandidate* cand = &state->candidates[i];
        if (!cand->is_active) continue;
        
        float3 diff = make_float3(
            spike_position.x - cand->center.x,
            spike_position.y - cand->center.y,
            spike_position.z - cand->center.z
        );
        float dist = sqrtf(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
        
        if (dist < 5.0f) {  // Within 5 Angstroms
            // Update existing candidate
            cand->spike_density += spike_intensity;
            cand->confidence = fminf(1.0f, cand->confidence + 0.05f);
            
            // Update center (weighted average)
            float w = spike_intensity / (cand->spike_density + 1e-6f);
            cand->center.x = cand->center.x * (1.0f - w) + spike_position.x * w;
            cand->center.y = cand->center.y * (1.0f - w) + spike_position.y * w;
            cand->center.z = cand->center.z * (1.0f - w) + spike_position.z * w;
            
            // Track resonance
            if (resonance_freq > 0.0f && resonance_freq < cand->resonance_frequency) {
                cand->resonance_frequency = resonance_freq;  // Keep lowest (softest mode)
            }
            
            // Track contributing aromatic
            bool aromatic_known = false;
            for (int j = 0; j < cand->n_contributors; j++) {
                if (cand->contributing_aromatics[j] == aromatic_idx) {
                    aromatic_known = true;
                    break;
                }
            }
            if (!aromatic_known && cand->n_contributors < 8) {
                cand->contributing_aromatics[cand->n_contributors++] = aromatic_idx;
            }
            
            found_match = true;
            
            // Update best candidate
            if (cand->confidence > state->peak_confidence) {
                state->peak_confidence = cand->confidence;
                state->best_candidate_idx = i;
            }
            
            break;
        }
    }
    
    // Create new candidate if no match and we have room
    if (!found_match && state->n_candidates < 32) {
        HotSpotCandidate* cand = &state->candidates[state->n_candidates];
        cand->center = spike_position;
        cand->confidence = 0.1f;  // Initial low confidence
        cand->resonance_frequency = resonance_freq > 0.0f ? resonance_freq : 10.0f;
        cand->spike_density = spike_intensity;
        cand->aromatic_count = 1;
        cand->phase_discovered = PHASE_CRYO_BURST;
        cand->validation_count = 0;
        cand->is_active = true;
        cand->contributing_aromatics[0] = aromatic_idx;
        cand->n_contributors = 1;
        cand->history_idx = 0;
        
        state->n_candidates++;
    }
}

// ============================================================================
// DEVICE FUNCTIONS: THERMAL RAMP PHASE
// ============================================================================

__device__ __forceinline__ float get_ramp_probe_intensity(
    const AdaptiveProtocolState* state,
    int aromatic_idx,
    const HotSpotCandidate* candidates,
    int n_candidates,
    float3 aromatic_center
) {
    // In ramp phase: focus on aromatics near candidates
    
    if (state->current_phase != PHASE_THERMAL_RAMP) {
        return 0.0f;
    }
    
    float intensity = 0.0f;
    
    // Check if this aromatic is near any candidate
    for (int i = 0; i < n_candidates; i++) {
        if (!candidates[i].is_active) continue;
        
        float3 diff = make_float3(
            aromatic_center.x - candidates[i].center.x,
            aromatic_center.y - candidates[i].center.y,
            aromatic_center.z - candidates[i].center.z
        );
        float dist = sqrtf(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
        
        if (dist < state->exploration_radius) {
            // Weight by candidate confidence and proximity
            float proximity_weight = 1.0f - dist / state->exploration_radius;
            float confidence_weight = candidates[i].confidence;
            intensity = fmaxf(intensity, 
                state->current_uv_intensity * proximity_weight * confidence_weight);
        }
    }
    
    return intensity;
}

__device__ __forceinline__ void validate_candidate_during_ramp(
    AdaptiveProtocolState* state,
    int candidate_idx,
    float spike_intensity,
    float thermal_factor  // How much thermal motion affects response
) {
    if (candidate_idx < 0 || candidate_idx >= state->n_candidates) return;
    
    HotSpotCandidate* cand = &state->candidates[candidate_idx];
    
    // Key insight: REAL cryptic sites become MORE active during warming
    // because thermal motion helps open the pocket
    // ARTIFACTS become LESS active because they were noise
    
    if (spike_intensity > HOT_SPOT_THRESHOLD) {
        // Site is responding to thermal activation
        cand->validation_count++;
        cand->confidence += 0.1f * thermal_factor;
        cand->confidence = fminf(1.0f, cand->confidence);
        
        // Record in history
        if (cand->history_idx < 16) {
            cand->confidence_history[cand->history_idx++] = cand->confidence;
        }
    } else {
        // Site is going quiet - might be artifact
        cand->confidence -= 0.05f;
        cand->confidence = fmaxf(0.0f, cand->confidence);
        
        // Deactivate if confidence drops too low
        if (cand->confidence < 0.1f) {
            cand->is_active = false;
        }
    }
    
    // Update best candidate
    if (cand->confidence > state->peak_confidence && cand->is_active) {
        state->peak_confidence = cand->confidence;
        state->best_candidate_idx = candidate_idx;
        state->best_site_location = cand->center;
    }
}

// ============================================================================
// DEVICE FUNCTIONS: FOCUSED DIG PHASE
// ============================================================================

__device__ __forceinline__ float get_dig_probe_intensity(
    const AdaptiveProtocolState* state,
    int aromatic_idx,
    float3 aromatic_center
) {
    // In dig phase: concentrate ALL probe energy on best candidate region
    
    if (state->current_phase != PHASE_FOCUSED_DIG) {
        return 0.0f;
    }
    
    if (state->best_candidate_idx < 0) {
        return 0.0f;
    }
    
    const HotSpotCandidate* best = &state->candidates[state->best_candidate_idx];
    
    float3 diff = make_float3(
        aromatic_center.x - best->center.x,
        aromatic_center.y - best->center.y,
        aromatic_center.z - best->center.z
    );
    float dist = sqrtf(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
    
    if (dist < state->exploration_radius) {
        // Gaussian focus - strongest at center
        float sigma = state->exploration_radius / 2.0f;
        float weight = expf(-0.5f * (dist * dist) / (sigma * sigma));
        return state->current_uv_intensity * weight;
    }
    
    return 0.0f;
}

__device__ __forceinline__ void deep_resonance_analysis(
    AdaptiveProtocolState* state,
    float* resonance_spectrum,  // Pre-allocated array for frequencies
    int n_frequencies,
    float spike_intensity,
    float3 spike_position
) {
    // During dig phase: build up detailed resonance spectrum
    // Looking for the characteristic soft mode of pocket opening
    
    if (state->current_phase != PHASE_FOCUSED_DIG) return;
    if (state->best_candidate_idx < 0) return;
    
    HotSpotCandidate* best = &state->candidates[state->best_candidate_idx];
    
    // Check if spike is in region of interest
    float3 diff = make_float3(
        spike_position.x - best->center.x,
        spike_position.y - best->center.y,
        spike_position.z - best->center.z
    );
    float dist = sqrtf(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
    
    if (dist > state->exploration_radius) return;
    
    // Accumulate spike contribution to spectrum
    // Weight by intensity and proximity
    float weight = spike_intensity * (1.0f - dist / state->exploration_radius);
    
    // Spectral contribution would be computed from spike timing
    // For now, just accumulate intensity
    best->spike_density += weight;
    
    // Update confidence based on consistent activity
    if (spike_intensity > HOT_SPOT_THRESHOLD) {
        best->confidence += 0.02f;
        best->confidence = fminf(1.0f, best->confidence);
        best->validation_count++;
    }
    
    // Check for site confirmation
    if (best->confidence > 0.8f && best->validation_count > 10) {
        state->site_confirmed = true;
        state->best_site_location = best->center;
    }
    
    // Update global peak
    if (best->confidence > state->peak_confidence) {
        state->peak_confidence = best->confidence;
    }
}

// ============================================================================
// UNIFIED PROBE INTENSITY FUNCTION
// ============================================================================

__device__ __forceinline__ float get_adaptive_probe_intensity(
    const AdaptiveProtocolState* state,
    int aromatic_idx,
    int n_aromatics,
    float3 aromatic_center
) {
    // Dispatch to appropriate phase function
    
    switch (state->current_phase) {
        case PHASE_CRYO_BURST:
            return get_cryo_probe_intensity(state, aromatic_idx, n_aromatics);
            
        case PHASE_THERMAL_RAMP:
            return get_ramp_probe_intensity(state, aromatic_idx, 
                state->candidates, state->n_candidates, aromatic_center);
            
        case PHASE_FOCUSED_DIG:
            return get_dig_probe_intensity(state, aromatic_idx, aromatic_center);
            
        default:
            return 0.0f;
    }
}

// ============================================================================
// TEMPERATURE CONTROL
// ============================================================================

__device__ __forceinline__ float get_adaptive_temperature(
    const AdaptiveProtocolState* state
) {
    return state->current_temp;
}

__device__ __forceinline__ float get_langevin_gamma(
    const AdaptiveProtocolState* state
) {
    // Damping coefficient varies by phase
    // High damping in cryo (frozen), low in dig (natural dynamics)
    
    switch (state->current_phase) {
        case PHASE_CRYO_BURST:
            return 50.0f;   // Heavy damping - minimize thermal motion
            
        case PHASE_THERMAL_RAMP:
            // Gradually reduce damping as temperature increases
            {
                float progress = (float)state->phase_step / (float)state->ramp_steps;
                return 50.0f * (1.0f - progress) + 5.0f * progress;
            }
            
        case PHASE_FOCUSED_DIG:
            return 5.0f;    // Normal physiological damping
            
        default:
            return 5.0f;
    }
}

// ============================================================================
// EXPLORATION CONTROL
// ============================================================================

__device__ __forceinline__ bool is_in_exploration_region(
    const AdaptiveProtocolState* state,
    float3 position
) {
    // In cryo phase: everywhere is valid
    if (state->current_phase == PHASE_CRYO_BURST) {
        return true;
    }
    
    // In later phases: only near exploration center
    float3 diff = make_float3(
        position.x - state->exploration_center.x,
        position.y - state->exploration_center.y,
        position.z - state->exploration_center.z
    );
    float dist = sqrtf(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
    
    return dist <= state->exploration_radius;
}

// ============================================================================
// STATUS REPORTING
// ============================================================================

__device__ __forceinline__ void get_protocol_status(
    const AdaptiveProtocolState* state,
    int* phase,
    float* temperature,
    float* best_confidence,
    float3* best_location,
    int* n_active_candidates
) {
    *phase = state->current_phase;
    *temperature = state->current_temp;
    *best_confidence = state->peak_confidence;
    *best_location = state->best_site_location;
    
    int active = 0;
    for (int i = 0; i < state->n_candidates; i++) {
        if (state->candidates[i].is_active) active++;
    }
    *n_active_candidates = active;
}

#endif // PRISM_ADAPTIVE_PROTOCOL_CUH
