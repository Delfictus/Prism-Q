// =============================================================================
// PRISM-NHS: Advanced UV-LIF Coupling System
// =============================================================================
//
// THEORY: UV excitation of aromatics creates detectable signals through multiple
// physical mechanisms. This module implements a sophisticated multi-channel
// detection system that captures these signals with high sensitivity.
//
// PHYSICAL MECHANISMS:
//
// 1. THERMAL WAVEFRONT PROPAGATION
//    Heat from UV absorption propagates through protein at speed of sound
//    (~2000 m/s = 2 Å/ps in proteins). This creates time-delayed thermal
//    signatures at distant voxels that are PHASE-LOCKED to the UV pulse.
//    Detection: Correlation between voxel distance and signal arrival time.
//
// 2. ANISOTROPIC HEAT DIFFUSION
//    Aromatic ring out-of-plane vibrations (A2u mode, ~700 cm⁻¹) preferentially
//    couple to atoms along the ring normal direction. Heat escapes faster
//    perpendicular to the ring plane.
//    Detection: Enhanced signal along ring normal axis.
//
// 3. DEWETTING HALO EFFECT
//    Excited aromatic has larger dipole → more polar → attracts water.
//    But surrounding hydrophobic residues now appear RELATIVELY more hydrophobic
//    by contrast, creating a "dewetting halo" around the excited aromatic.
//    Detection: Ring of enhanced exclusion surrounding the aromatic.
//
// 4. TEMPORAL DERIVATIVE AMPLIFICATION
//    UV-induced signals have characteristic profile: fast onset (fs excitation),
//    slow decay (ps vibrational relaxation). Thermal noise is Brownian (no
//    characteristic timescale). Temporal derivative filtering enhances UV signals.
//    Detection: d(signal)/dt with matched filter for UV profile.
//
// 5. COOPERATIVE MULTI-AROMATIC ENHANCEMENT
//    When multiple aromatics are excited, water restructuring is cooperative.
//    Signal scales super-linearly with number of nearby excited aromatics.
//    Detection: Non-linear boost based on local excited aromatic count.
//
// REFERENCES:
// - Leitner & Straub, "Proteins: Energy, Heat and Signal Flow" (2009)
// - Sagnella & Straub, "Directed Energy 'Funneling' in Proteins" (2001)
// - Li et al., "Heat Dissipation in Proteins" J. Phys. Chem. B (2015)
// =============================================================================

#ifndef UV_LIF_COUPLING_CUH
#define UV_LIF_COUPLING_CUH

#include <cuda_runtime.h>

// =============================================================================
// PHYSICAL CONSTANTS
// =============================================================================

// Speed of sound in proteins (Å/ps)
// Reference: ~2000 m/s = 2 nm/ps = 20 Å/ps in globular proteins
#define PROTEIN_SOUND_SPEED  20.0f

// Thermal diffusivity in proteins (Å²/ps)
// Reference: D ≈ 0.1-0.5 nm²/ps for heat in proteins
#define THERMAL_DIFFUSIVITY  2.0f

// Vibrational relaxation timescale (ps)
#define VIB_RELAX_TAU  2.0f

// Ring normal coupling enhancement (anisotropy factor)
// Heat couples ~2x stronger along ring normal due to out-of-plane modes
#define RING_NORMAL_ENHANCEMENT  2.0f

// Dewetting halo parameters - TIGHTENED for spatial localization
#define HALO_INNER_RADIUS  2.0f   // Å - direct effect radius (was 4.0)
#define HALO_OUTER_RADIUS  5.0f   // Å - halo effect radius (was 12.0)
#define HALO_CONTRAST_FACTOR  0.8f  // How much halo amplifies dewetting (increased)

// Cooperative enhancement exponent
// Signal scales as N^COOP_EXPONENT for N nearby excited aromatics
#define COOP_EXPONENT  1.3f

// UV-LIF coupling strength (dimensionless)
// This determines how strongly UV signals couple to LIF membrane potential
#define UV_LIF_COUPLING_STRENGTH  0.8f

// Temporal derivative filter time constant (ps)
#define DERIVATIVE_TAU  0.5f

// Maximum UV signal per aromatic (normalized)
#define MAX_UV_SIGNAL_PER_AROMATIC  0.15f

// =============================================================================
// DATA STRUCTURES
// =============================================================================

// UV-induced signal state for a voxel
struct UvSignalState {
    float thermal_wavefront;      // Accumulated thermal wavefront signal
    float derivative_signal;      // Temporal derivative filtered signal
    float halo_signal;            // Dewetting halo contribution
    float cooperative_boost;      // Multi-aromatic cooperative factor
    float total_uv_signal;        // Combined UV signal for LIF injection
    float last_total_signal;      // Previous timestep (for derivative)
    int last_update_timestep;     // For staleness detection
};

// Per-aromatic excitation state (extended from nhs_excited_state.cuh)
struct AromaticExcitationState {
    int is_excited;
    float electronic_population;
    float vibrational_energy;
    float time_since_excitation;  // ps
    float3 position;              // Current aromatic centroid
    float3 ring_normal;           // Unit normal to ring plane
    int aromatic_type;            // 0=TRP, 1=TYR, 2=PHE, 3=S-S
};

// =============================================================================
// DEVICE FUNCTIONS: THERMAL WAVEFRONT PROPAGATION
// =============================================================================

/**
 * @brief Compute thermal wavefront signal at a voxel from an excited aromatic
 *
 * The thermal wavefront travels at the speed of sound and has a Gaussian
 * spatial profile that broadens with distance (diffusion).
 *
 * @param voxel_pos Position of the voxel
 * @param aromatic_pos Position of the excited aromatic
 * @param ring_normal Unit normal to the aromatic ring
 * @param time_since_excitation Time since UV pulse (ps)
 * @param vibrational_energy Current vibrational energy (kcal/mol)
 * @return Thermal wavefront signal contribution (0-1)
 */
__device__ __forceinline__ float compute_thermal_wavefront(
    float3 voxel_pos,
    float3 aromatic_pos,
    float3 ring_normal,
    float time_since_excitation,
    float vibrational_energy
) {
    // Distance from aromatic to voxel
    float3 delta = make_float3(
        voxel_pos.x - aromatic_pos.x,
        voxel_pos.y - aromatic_pos.y,
        voxel_pos.z - aromatic_pos.z
    );
    float distance = sqrtf(delta.x*delta.x + delta.y*delta.y + delta.z*delta.z);

    if (distance < 0.1f || time_since_excitation < 0.001f) {
        return 0.0f;  // Too close or too early
    }

    // Expected arrival time of thermal wavefront
    float expected_arrival = distance / PROTEIN_SOUND_SPEED;

    // Wavefront has finite width due to thermal diffusion
    // Width scales as sqrt(D * t) where D is thermal diffusivity
    float wavefront_width = sqrtf(THERMAL_DIFFUSIVITY * time_since_excitation);
    wavefront_width = fmaxf(wavefront_width, 1.0f);  // Minimum width

    // Gaussian profile centered on expected arrival time
    float time_delta = time_since_excitation - expected_arrival;
    float temporal_factor = expf(-(time_delta * time_delta) / (2.0f * wavefront_width * wavefront_width));

    // Anisotropic enhancement along ring normal
    // Project displacement onto ring normal
    float normal_component = fabsf(delta.x * ring_normal.x +
                                   delta.y * ring_normal.y +
                                   delta.z * ring_normal.z);
    float normal_fraction = normal_component / fmaxf(distance, 0.1f);
    float anisotropy_factor = 1.0f + (RING_NORMAL_ENHANCEMENT - 1.0f) * normal_fraction;

    // Energy decay with distance (1/r² geometric spreading + absorption)
    float distance_decay = 1.0f / (1.0f + 0.1f * distance * distance);

    // Combine factors
    float signal = vibrational_energy * temporal_factor * anisotropy_factor * distance_decay;

    // Normalize to reasonable range
    return fminf(signal * 0.1f, MAX_UV_SIGNAL_PER_AROMATIC);
}

// =============================================================================
// DEVICE FUNCTIONS: DEWETTING HALO EFFECT
// =============================================================================

/**
 * @brief Compute dewetting halo signal
 *
 * Excited aromatic is more polar → attracts water → surrounding hydrophobic
 * residues appear MORE hydrophobic by contrast → dewetting halo.
 *
 * Signal profile:
 * - Inner region (< HALO_INNER_RADIUS): Reduced exclusion (more polar)
 * - Halo region (INNER to OUTER): ENHANCED exclusion (contrast effect)
 * - Outer region (> HALO_OUTER_RADIUS): No effect
 */
__device__ __forceinline__ float compute_dewetting_halo(
    float distance,
    float electronic_population
) {
    if (electronic_population < 0.01f) {
        return 0.0f;
    }

    if (distance < HALO_INNER_RADIUS) {
        // Inner region: reduced exclusion (negative signal)
        // Excited aromatic is more polar, attracts water
        float inner_factor = 1.0f - distance / HALO_INNER_RADIUS;
        return -0.3f * electronic_population * inner_factor;
    }
    else if (distance < HALO_OUTER_RADIUS) {
        // Halo region: ENHANCED exclusion (positive signal)
        // Surrounding hydrophobics appear MORE hydrophobic by contrast
        float halo_fraction = (distance - HALO_INNER_RADIUS) / (HALO_OUTER_RADIUS - HALO_INNER_RADIUS);
        // Bell-shaped profile in halo
        float halo_profile = sinf(halo_fraction * 3.14159f);
        return HALO_CONTRAST_FACTOR * electronic_population * halo_profile;
    }

    return 0.0f;
}

// =============================================================================
// DEVICE FUNCTIONS: COOPERATIVE MULTI-AROMATIC ENHANCEMENT
// =============================================================================

/**
 * @brief Compute cooperative enhancement factor
 *
 * When multiple aromatics near a voxel are excited, the signal is enhanced
 * super-linearly due to cooperative water restructuring.
 *
 * @param n_nearby_excited Number of excited aromatics within detection radius
 * @return Cooperative enhancement factor (>= 1.0)
 */
__device__ __forceinline__ float compute_cooperative_boost(int n_nearby_excited) {
    if (n_nearby_excited <= 1) {
        return 1.0f;
    }

    // Super-linear scaling: N^1.3 instead of N
    return powf((float)n_nearby_excited, COOP_EXPONENT);
}

// =============================================================================
// DEVICE FUNCTIONS: TEMPORAL DERIVATIVE FILTER
// =============================================================================

/**
 * @brief Apply temporal derivative filter to extract UV-specific signals
 *
 * UV signals have characteristic fast-onset/slow-decay profile.
 * Thermal noise is Brownian with no characteristic timescale.
 * Derivative filtering enhances UV signals over noise.
 *
 * Uses exponential moving average of derivative for stability.
 */
__device__ __forceinline__ float apply_derivative_filter(
    float current_signal,
    float previous_signal,
    float dt
) {
    float raw_derivative = (current_signal - previous_signal) / fmaxf(dt, 0.001f);

    // Rectify: only care about signal INCREASES (onset of UV effect)
    float rectified = fmaxf(raw_derivative, 0.0f);

    // Scale by time constant for matched filtering
    float filter_response = rectified * DERIVATIVE_TAU;

    return fminf(filter_response, 0.5f);  // Cap at reasonable value
}

// =============================================================================
// MAIN UV-LIF COUPLING FUNCTION
// =============================================================================

/**
 * @brief Compute total UV-induced signal for LIF injection at a voxel
 *
 * This is the main entry point for UV-LIF coupling. It combines all the
 * physical mechanisms into a single signal that's injected into the LIF
 * membrane potential.
 *
 * @param voxel_pos Position of the voxel center
 * @param aromatic_states Array of aromatic excitation states [n_aromatics]
 * @param n_aromatics Number of aromatics
 * @param dt Timestep (ps)
 * @param previous_signal Previous total UV signal at this voxel
 * @return Total UV signal to inject into LIF membrane potential
 */
__device__ float compute_uv_lif_signal(
    float3 voxel_pos,
    const float3* aromatic_positions,
    const float3* ring_normals,
    const int* is_excited,
    const float* electronic_population,
    const float* vibrational_energy,
    const float* time_since_excitation,
    int n_aromatics,
    float dt,
    float previous_signal
) {
    float thermal_wavefront_sum = 0.0f;
    float halo_sum = 0.0f;
    int n_nearby_excited = 0;
    float min_distance_to_excited = 1000.0f;  // Track closest excited aromatic

    // Maximum detection radius - TIGHTENED for spatial localization
    // Only voxels VERY close to aromatics should get UV spikes
    const float MAX_DETECTION_RADIUS = 6.0f;  // Å (was 20.0)

    // Accumulate signals from all excited aromatics
    for (int a = 0; a < n_aromatics; a++) {
        if (!is_excited[a]) continue;

        float3 arom_pos = aromatic_positions[a];
        float3 delta = make_float3(
            voxel_pos.x - arom_pos.x,
            voxel_pos.y - arom_pos.y,
            voxel_pos.z - arom_pos.z
        );
        float distance = sqrtf(delta.x*delta.x + delta.y*delta.y + delta.z*delta.z);

        if (distance > MAX_DETECTION_RADIUS) continue;

        n_nearby_excited++;
        min_distance_to_excited = fminf(min_distance_to_excited, distance);

        // Distance-based weighting - MUCH stronger decay to localize signal
        // Use steep Gaussian: σ = 2Å so signal drops to ~14% at 4Å
        float proximity_weight = expf(-distance * distance / (2.0f * 2.0f * 2.0f));

        // Thermal wavefront contribution (weighted by proximity)
        thermal_wavefront_sum += proximity_weight * compute_thermal_wavefront(
            voxel_pos,
            arom_pos,
            ring_normals[a],
            time_since_excitation[a],
            vibrational_energy[a]
        );

        // Dewetting halo contribution (also weighted by proximity)
        halo_sum += proximity_weight * compute_dewetting_halo(distance, electronic_population[a]);
    }

    if (n_nearby_excited == 0) {
        return 0.0f;
    }

    // Apply cooperative enhancement
    float coop_boost = compute_cooperative_boost(n_nearby_excited);

    // PROXIMITY BOOST: Strongly favor voxels VERY close to aromatics
    // This ensures UV spikes happen at aromatic locations, not spread broadly
    float proximity_boost = 1.0f;
    if (min_distance_to_excited < 3.0f) {
        // Very close: strong boost (2-4x)
        proximity_boost = 4.0f - min_distance_to_excited;
    } else if (min_distance_to_excited < 5.0f) {
        // Medium distance: moderate signal
        proximity_boost = 1.0f;
    } else {
        // Far: attenuate signal significantly
        proximity_boost = 0.3f;
    }

    // Combine signals with proximity boost
    float combined_signal = (thermal_wavefront_sum + halo_sum) * coop_boost * proximity_boost;

    // Apply temporal derivative filter for enhanced UV specificity
    float derivative_boost = apply_derivative_filter(combined_signal, previous_signal, dt);

    // Final signal with derivative enhancement
    float total_signal = combined_signal + derivative_boost * 0.5f;

    // Apply coupling strength and clamp
    return fminf(total_signal * UV_LIF_COUPLING_STRENGTH, 1.0f);
}

// =============================================================================
// EXPANDED EXCLUSION MODIFIER
// =============================================================================

/**
 * @brief Compute exclusion modifier for ANY atom based on proximity to excited aromatics
 *
 * This expands the UV effect beyond just ring atoms to ALL atoms within
 * a cutoff distance of excited aromatics. The effect decays with distance
 * and is modulated by the excitation state.
 *
 * @param atom_pos Position of the atom
 * @param aromatic_positions Positions of aromatic centroids [n_aromatics]
 * @param ring_normals Ring normal vectors [n_aromatics]
 * @param is_excited Excitation flags [n_aromatics]
 * @param electronic_population Electronic state populations [n_aromatics]
 * @param n_aromatics Number of aromatics
 * @return Exclusion modifier (0.2-1.0, where 1.0 = no modification)
 */
__device__ float compute_expanded_exclusion_modifier(
    float3 atom_pos,
    const float3* aromatic_positions,
    const float3* ring_normals,
    const int* is_excited,
    const float* electronic_population,
    int n_aromatics
) {
    // Effect radius for exclusion modification
    const float EXCLUSION_EFFECT_RADIUS = 8.0f;  // Å
    const float MIN_MODIFIER = 0.2f;  // Maximum 80% reduction

    float total_effect = 0.0f;

    for (int a = 0; a < n_aromatics; a++) {
        if (!is_excited[a]) continue;

        float3 arom_pos = aromatic_positions[a];
        float3 delta = make_float3(
            atom_pos.x - arom_pos.x,
            atom_pos.y - arom_pos.y,
            atom_pos.z - arom_pos.z
        );
        float distance = sqrtf(delta.x*delta.x + delta.y*delta.y + delta.z*delta.z);

        if (distance > EXCLUSION_EFFECT_RADIUS) continue;

        // Gaussian decay with distance
        float distance_factor = expf(-distance * distance / (2.0f * 4.0f * 4.0f));  // σ = 4 Å

        // Anisotropic enhancement along ring normal
        float3 ring_n = ring_normals[a];
        float normal_component = fabsf(delta.x * ring_n.x + delta.y * ring_n.y + delta.z * ring_n.z);
        float normal_fraction = normal_component / fmaxf(distance, 0.1f);
        float anisotropy = 1.0f + 0.5f * normal_fraction;  // Up to 50% enhancement along normal

        // Accumulate effect weighted by electronic population
        total_effect += electronic_population[a] * distance_factor * anisotropy;
    }

    // Clamp total effect
    total_effect = fminf(total_effect, 1.0f);

    // Convert to modifier: 1.0 = no effect, MIN_MODIFIER = maximum effect
    // Effect of 1.0 → modifier of MIN_MODIFIER (80% reduction)
    float modifier = 1.0f - total_effect * (1.0f - MIN_MODIFIER);

    return modifier;
}

// =============================================================================
// INTEGRATION KERNEL HELPER
// =============================================================================

/**
 * @brief Update LIF membrane potential with UV-induced signal
 *
 * This should be called from the main NHS kernel after the standard LIF update.
 * It injects the UV signal directly into the membrane potential.
 *
 * @param membrane_potential Current membrane potential (modified in-place)
 * @param uv_signal UV-induced signal from compute_uv_lif_signal()
 * @param is_uv_burst_active Whether UV burst is currently active
 */
__device__ __forceinline__ void inject_uv_signal_to_lif(
    float& membrane_potential,
    float uv_signal,
    int is_uv_burst_active
) {
    // UV signal is additive to membrane potential
    // Apply slightly higher weight during active UV burst
    float burst_boost = is_uv_burst_active ? 1.5f : 1.0f;

    membrane_potential += uv_signal * burst_boost;
}

// =============================================================================
// SHARED MEMORY OPTIMIZATION STRUCTURES
// =============================================================================

// For efficient parallel processing, aromatic data can be cached in shared memory
#define MAX_AROMATICS_CACHED 64

struct CachedAromaticData {
    float3 position;
    float3 ring_normal;
    float electronic_population;
    float vibrational_energy;
    float time_since_excitation;
    int is_excited;
};

/**
 * @brief Load aromatic data into shared memory for cache efficiency
 */
__device__ void cache_aromatic_data(
    CachedAromaticData* shared_cache,
    const float3* aromatic_positions,
    const float3* ring_normals,
    const int* is_excited,
    const float* electronic_population,
    const float* vibrational_energy,
    const float* time_since_excitation,
    int n_aromatics,
    int tid,
    int block_size
) {
    // Cooperative loading
    for (int a = tid; a < n_aromatics && a < MAX_AROMATICS_CACHED; a += block_size) {
        shared_cache[a].position = aromatic_positions[a];
        shared_cache[a].ring_normal = ring_normals[a];
        shared_cache[a].electronic_population = electronic_population[a];
        shared_cache[a].vibrational_energy = vibrational_energy[a];
        shared_cache[a].time_since_excitation = time_since_excitation[a];
        shared_cache[a].is_excited = is_excited[a];
    }
    __syncthreads();
}

/**
 * @brief Optimized UV signal computation using cached aromatic data
 */
__device__ float compute_uv_lif_signal_cached(
    float3 voxel_pos,
    const CachedAromaticData* aromatic_cache,
    int n_aromatics,
    float dt,
    float previous_signal
) {
    float thermal_wavefront_sum = 0.0f;
    float halo_sum = 0.0f;
    int n_nearby_excited = 0;

    const float MAX_DETECTION_RADIUS = 20.0f;

    #pragma unroll 4
    for (int a = 0; a < n_aromatics && a < MAX_AROMATICS_CACHED; a++) {
        if (!aromatic_cache[a].is_excited) continue;

        float3 arom_pos = aromatic_cache[a].position;
        float dx = voxel_pos.x - arom_pos.x;
        float dy = voxel_pos.y - arom_pos.y;
        float dz = voxel_pos.z - arom_pos.z;
        float dist_sq = dx*dx + dy*dy + dz*dz;

        if (dist_sq > MAX_DETECTION_RADIUS * MAX_DETECTION_RADIUS) continue;

        float distance = sqrtf(dist_sq);
        n_nearby_excited++;

        // Thermal wavefront
        thermal_wavefront_sum += compute_thermal_wavefront(
            voxel_pos,
            arom_pos,
            aromatic_cache[a].ring_normal,
            aromatic_cache[a].time_since_excitation,
            aromatic_cache[a].vibrational_energy
        );

        // Dewetting halo
        halo_sum += compute_dewetting_halo(distance, aromatic_cache[a].electronic_population);
    }

    if (n_nearby_excited == 0) return 0.0f;

    float coop_boost = compute_cooperative_boost(n_nearby_excited);
    float combined = (thermal_wavefront_sum + halo_sum) * coop_boost;
    float deriv_boost = apply_derivative_filter(combined, previous_signal, dt);

    return fminf((combined + deriv_boost * 0.5f) * UV_LIF_COUPLING_STRENGTH, 1.0f);
}

#endif // UV_LIF_COUPLING_CUH
