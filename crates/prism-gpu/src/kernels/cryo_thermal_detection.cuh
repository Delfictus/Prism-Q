//! PRISM-NHS Cryo Thermal Detection System
//!
//! Ultra-sensitive detection of UV absorption signatures in frozen landscapes.
//!
//! KEY PHYSICS:
//! - UV at 280nm is absorbed by aromatic residues (Trp >> Tyr > Phe)
//! - Absorption causes LOCAL HEATING even in frozen state
//! - Water does NOT absorb 280nm UV efficiently
//! - Therefore: thermal spike = binding site signature
//!
//! DETECTION MODES:
//! 1. Direct thermal spike at probe target
//! 2. Thermal gradient spreading from hot spot
//! 3. Phase transition (ice→water melt wave)
//! 4. Correlated multi-aromatic response (pocket signature)
//!
//! The neuromorphic detector converts these subtle thermal changes
//! into spike trains for pattern recognition.

#ifndef PRISM_CRYO_THERMAL_DETECTION_CUH
#define PRISM_CRYO_THERMAL_DETECTION_CUH

#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// PHYSICAL CONSTANTS (PHYSICS-CORRECTED UV PUMP-PROBE CALIBRATION)
// ============================================================================

// ε → σ conversion: σ(Å²) = ε(M⁻¹cm⁻¹) × 3.823×10⁻⁵
#define EPSILON_TO_SIGMA_FACTOR    3.823e-5f

// Extinction coefficients at λmax (M⁻¹cm⁻¹)
#define EPSILON_TRP_280     5600.0f   // Tryptophan at 280nm
#define EPSILON_TYR_274     1490.0f   // Tyrosine at 274nm
#define EPSILON_PHE_258     200.0f    // Phenylalanine at 258nm
#define EPSILON_SS_250      300.0f    // Disulfide at 250nm

// Calibrated absorption cross-sections at peak wavelengths (Å², per molecule)
// σ = ε × 3.823×10⁻⁵
#define UV_SIGMA_TRP        0.21409f  // TRP at 280nm: 5600 × 3.823e-5
#define UV_SIGMA_TYR        0.05696f  // TYR at 274nm: 1490 × 3.823e-5
#define UV_SIGMA_PHE        0.00765f  // PHE at 258nm: 200 × 3.823e-5
#define UV_SIGMA_SS         0.01147f  // S-S at 250nm: 300 × 3.823e-5
#define UV_SIGMA_WATER      0.0f      // Water - transparent at UV

// Relative absorption (Trp=1.0, for backward compatibility)
#define UV_ABSORPTION_TRP   1.0f
#define UV_ABSORPTION_TYR   (UV_SIGMA_TYR / UV_SIGMA_TRP)   // ~0.266
#define UV_ABSORPTION_PHE   (UV_SIGMA_PHE / UV_SIGMA_TRP)   // ~0.036
#define UV_ABSORPTION_HIS   0.01f     // Histidine (negligible)
#define UV_ABSORPTION_WATER 0.0f      // Water - TRANSPARENT at 280nm

// Calibrated photon fluence (photons/Å² per pulse)
// Anchored to ΔT = 20K for TRP @ 280nm with η = 1.0
#define CALIBRATED_PHOTON_FLUENCE   0.024f

// Heat yield η (fraction of photon energy → heat)
#define DEFAULT_HEAT_YIELD          1.0f

// Effective degrees of freedom for local heating (N_eff)
#define NEFF_TRP            9.0f      // Indole ring
#define NEFF_TYR            10.0f     // Phenol ring + OH
#define NEFF_PHE            9.0f      // Benzene ring + side chain
#define NEFF_SS             2.0f      // S-S bond

// Boltzmann constant in eV/K
#define KB_EV_K             8.617e-5f

// Photon energies at key wavelengths (eV) - E = 1239.84 / λ(nm)
#define PHOTON_ENERGY_250NM 4.959f
#define PHOTON_ENERGY_258NM 4.806f
#define PHOTON_ENERGY_274NM 4.525f
#define PHOTON_ENERGY_280NM 4.428f
#define PHOTON_ENERGY_290NM 4.275f

// Thermal properties
#define HEAT_CAPACITY_PROTEIN  1.5f   // J/(g·K) approximate
#define HEAT_CAPACITY_ICE      2.09f  // J/(g·K)
#define HEAT_CAPACITY_WATER    4.18f  // J/(g·K)
#define LATENT_HEAT_FUSION     334.0f // J/g for ice→water

// Cryo physics
#define CRYO_BATH_TEMPERATURE  100.0f  // K - defensible classical MD range
#define AMBIENT_TEMPERATURE    300.0f  // K
#define MELTING_POINT          273.15f // K
#define SUPERCOOL_MARGIN       10.0f   // K below melting for metastable ice

// Detection thresholds (calibrated for 20K TRP signal)
#define THERMAL_SPIKE_THRESHOLD     0.5f    // K - detect ~2.5% of TRP signal
#define THERMAL_GRADIENT_THRESHOLD  0.01f   // K/Å - minimum gradient
#define MELT_FRACTION_THRESHOLD     0.001f  // 0.1% melting detectable
#define CORRELATION_THRESHOLD       0.3f    // Cross-correlation for pocket

// Neuromorphic parameters
#define THERMAL_LIF_TAU            10.0f    // ps - integration time constant
#define THERMAL_LIF_THRESHOLD      0.05f   // Normalized spike threshold
#define THERMAL_LIF_RESET          0.0f    // Reset potential
#define THERMAL_REFRACTORY         5.0f    // ps - refractory period

// Grid resolution for thermal field
#define THERMAL_GRID_SPACING       1.0f    // Å - fine resolution for gradients

// ============================================================================
// DATA STRUCTURES
// ============================================================================

// Thermal state of a single voxel
struct ThermalVoxel {
    float temperature;          // Current temperature (K)
    float baseline_temp;        // Reference temperature before probe
    float delta_temp;           // Change from baseline
    float temp_gradient_mag;    // Magnitude of local gradient
    float3 temp_gradient_dir;   // Direction of steepest gradient
    
    float ice_fraction;         // 0=water, 1=ice
    float melt_rate;            // Rate of ice→water transition
    
    float heat_capacity;        // Effective heat capacity (ice/water mix)
    float thermal_conductivity; // Local conductivity
    
    int last_spike_time;        // When this voxel last spiked
    float lif_potential;        // Neuromorphic potential
    bool in_refractory;         // Refractory period flag
};

// UV probe event with absorption tracking
struct UVProbeEvent {
    float3 target_position;     // Where probe was aimed
    int target_aromatic_idx;    // Which aromatic (if any)
    float probe_energy;         // Energy delivered (arbitrary units)
    float absorption_fraction;  // How much was absorbed (0-1)
    
    float temp_rise;            // Resulting temperature rise
    float affected_volume;      // Volume heated (Å³)
    
    int response_spikes;        // Neuromorphic spikes triggered
    float response_latency;     // Time to first spike (ps)
    
    bool hit_aromatic;          // Did we hit an absorbing residue?
    bool caused_melting;        // Did we trigger ice→water?
    bool pocket_signature;      // Correlated multi-aromatic response?
};

// Pocket detection signature
struct ThermalPocketSignature {
    float3 center;              // Estimated pocket center
    float radius;               // Estimated pocket radius
    
    float peak_temp_rise;       // Maximum temperature rise observed
    float integrated_heat;      // Total heat deposited
    float melt_volume;          // Volume of ice melted
    
    int n_responding_aromatics; // How many aromatics responded
    int responding_aromatics[8];// Which ones
    float response_correlation; // How correlated are the responses
    
    float confidence;           // Overall pocket confidence
    int detection_count;        // How many times detected
};

// Complete thermal detection state
struct CryoThermalState {
    // Grid dimensions
    int3 grid_dim;
    float3 grid_origin;
    float grid_spacing;
    
    // Thermal field (flattened 3D array)
    ThermalVoxel* voxels;       // Device pointer
    int n_voxels;
    
    // Global temperature
    float ambient_temp;         // Background temperature
    float cryo_temp;            // Target cryo temperature
    
    // Probe tracking
    UVProbeEvent* recent_probes;// Circular buffer of recent probes
    int n_recent_probes;
    int probe_buffer_idx;
    
    // Detected signatures
    ThermalPocketSignature* signatures;
    int n_signatures;
    int max_signatures;
    
    // Neuromorphic output
    int* spike_voxels;          // Voxels that spiked this step
    int n_spikes_this_step;
    int total_thermal_spikes;
    
    // Statistics
    float max_temp_observed;
    float total_heat_absorbed;
    float total_ice_melted;
};

// ============================================================================
// CHROMOPHORE TYPE ENUM - EXTENDED VERSION
// ============================================================================
// This header uses an EXTENDED enum for comprehensive residue typing.
//
// WARNING: nhs_excited_state.cuh and CPU fused_engine.rs use SIMPLE enum:
//   SIMPLE:   0=TRP, 1=TYR, 2=PHE, 3=S-S (4 types only)
//   EXTENDED: 0=TRP, 1=TYR, 2=PHE, 3=HIS, 4=OTHER, 5=WATER, 6=S-S
//
// If calling functions from this header, translate SS: SIMPLE 3 → EXTENDED 6
// ============================================================================
// EXTENDED residue_type encoding for UV pump-probe:
//   0 = TRP (Tryptophan)
//   1 = TYR (Tyrosine)
//   2 = PHE (Phenylalanine)
//   3 = HIS (Histidine - weak/negligible)
//   4 = other protein (non-absorbing)
//   5 = water (model choice: not heated by UV in this implementation)
//   6 = S-S (Disulfide bond - only valid at ~250nm)
// ============================================================================

#define CHROMOPHORE_TRP     0
#define CHROMOPHORE_TYR     1
#define CHROMOPHORE_PHE     2
#define CHROMOPHORE_HIS     3
#define CHROMOPHORE_OTHER   4
#define CHROMOPHORE_WATER   5
#define CHROMOPHORE_SS      6

// Helper: Convert SIMPLE enum (from CPU/nhs_excited_state) to EXTENDED
// SIMPLE:   0=TRP, 1=TYR, 2=PHE, 3=S-S
// EXTENDED: 0=TRP, 1=TYR, 2=PHE, 6=S-S (HIS/OTHER/WATER in between)
__device__ __forceinline__ int simple_to_extended_chromophore(int simple_type) {
    // 0,1,2 map directly; 3 (S-S) maps to 6
    return (simple_type == 3) ? CHROMOPHORE_SS : simple_type;
}

// Gaussian band parameters: λmax (nm), εmax (M⁻¹cm⁻¹), band 1/e half-width w (nm)
// P6: w is NOT the FWHM. σ = w / sqrt(2); FWHM = 1.665109 * w.
// Values mirror crate::config (TRP 15, TYR 12, PHE 10, S-S 20).
#define CRYO_BAND_W_TO_SIGMA      0.70710678f   // 1/sqrt(2)

#define CRYO_TRP_LAMBDA_MAX      280.0f
#define CRYO_TRP_EPSILON_MAX     5600.0f
#define CRYO_TRP_BANDWIDTH       15.0f

#define CRYO_TYR_LAMBDA_MAX      274.0f
#define CRYO_TYR_EPSILON_MAX     1490.0f
#define CRYO_TYR_BANDWIDTH       12.0f

#define CRYO_PHE_LAMBDA_MAX      258.0f
#define CRYO_PHE_EPSILON_MAX     200.0f
#define CRYO_PHE_BANDWIDTH       10.0f

#define CRYO_SS_LAMBDA_MAX       250.0f
#define CRYO_SS_EPSILON_MAX      300.0f
#define CRYO_SS_BANDWIDTH        20.0f

// ============================================================================
// DEVICE FUNCTIONS: THERMAL PHYSICS (PHYSICS-CORRECTED)
// ============================================================================

// Get photon energy at wavelength (eV)
// E = hc/λ = 1239.84 / λ(nm)
__device__ __forceinline__ float get_photon_energy(float wavelength_nm) {
    return 1239.84f / wavelength_nm;
}

// Compute Gaussian extinction ε(λ) for a chromophore
// ε(λ) = ε_max × exp[-(λ - λ_max)² / (2σ²)]
// where σ = w / sqrt(2)  (P6 canonical; w is the 1/e half-width, not the FWHM)
__device__ __forceinline__ float compute_extinction_gaussian(
    float wavelength_nm,
    float lambda_max,
    float epsilon_max,
    float band_w_nm
) {
    float sigma = band_w_nm * CRYO_BAND_W_TO_SIGMA;
    float delta = wavelength_nm - lambda_max;
    return epsilon_max * expf(-0.5f * (delta * delta) / (sigma * sigma));
}

// Get wavelength-dependent extinction coefficient ε(λ) for chromophore type
__device__ __forceinline__ float get_extinction_at_wavelength(int residue_type, float wavelength_nm) {
    switch (residue_type) {
        case CHROMOPHORE_TRP:
            return compute_extinction_gaussian(wavelength_nm, CRYO_TRP_LAMBDA_MAX, CRYO_TRP_EPSILON_MAX, CRYO_TRP_BANDWIDTH);
        case CHROMOPHORE_TYR:
            return compute_extinction_gaussian(wavelength_nm, CRYO_TYR_LAMBDA_MAX, CRYO_TYR_EPSILON_MAX, CRYO_TYR_BANDWIDTH);
        case CHROMOPHORE_PHE:
            return compute_extinction_gaussian(wavelength_nm, CRYO_PHE_LAMBDA_MAX, CRYO_PHE_EPSILON_MAX, CRYO_PHE_BANDWIDTH);
        case CHROMOPHORE_SS:
            return compute_extinction_gaussian(wavelength_nm, CRYO_SS_LAMBDA_MAX, CRYO_SS_EPSILON_MAX, CRYO_SS_BANDWIDTH);
        case CHROMOPHORE_HIS:
            // HIS has negligible absorption at these wavelengths
            return compute_extinction_gaussian(wavelength_nm, CRYO_PHE_LAMBDA_MAX, 20.0f, CRYO_PHE_BANDWIDTH);
        case CHROMOPHORE_WATER:
        case CHROMOPHORE_OTHER:
        default:
            return 0.0f;  // Model choice: water/other not heated by UV
    }
}

// Convert extinction to cross-section: σ(Å²) = ε × 3.823×10⁻⁵
__device__ __forceinline__ float extinction_to_sigma(float epsilon) {
    return epsilon * EPSILON_TO_SIGMA_FACTOR;
}

// Get wavelength-dependent absorption cross-section σ(λ) (Å², per molecule)
__device__ __forceinline__ float get_uv_cross_section_at_wavelength(int residue_type, float wavelength_nm) {
    float epsilon = get_extinction_at_wavelength(residue_type, wavelength_nm);
    return extinction_to_sigma(epsilon);
}

// Get absorption cross-section at peak wavelength (for backward compatibility)
__device__ __forceinline__ float get_uv_cross_section(int residue_type) {
    switch (residue_type) {
        case CHROMOPHORE_TRP:   return UV_SIGMA_TRP;    // 0.21409 Å² at 280nm
        case CHROMOPHORE_TYR:   return UV_SIGMA_TYR;    // 0.05696 Å² at 274nm
        case CHROMOPHORE_PHE:   return UV_SIGMA_PHE;    // 0.00765 Å² at 258nm
        case CHROMOPHORE_SS:    return UV_SIGMA_SS;     // 0.01147 Å² at 250nm
        case CHROMOPHORE_HIS:   return UV_SIGMA_PHE * 0.1f;
        case CHROMOPHORE_WATER:
        case CHROMOPHORE_OTHER:
        default:
            return 0.0f;
    }
}

// Get N_eff (effective DOF proxy) for residue type
// NOTE: If nhs_excited_state.cuh is included first (NHS pipeline), its get_neff() takes precedence
// The cryo version uses CHROMOPHORE_* enum, NHS version uses CANONICAL 0=TRP ordering
#ifndef NHS_EXCITED_STATE_FUSED_CU
__device__ __forceinline__ float get_neff_cryo(int residue_type) {
    switch (residue_type) {
        case CHROMOPHORE_TRP:   return NEFF_TRP;   // 9.0
        case CHROMOPHORE_TYR:   return NEFF_TYR;   // 10.0
        case CHROMOPHORE_PHE:   return NEFF_PHE;   // 9.0
        case CHROMOPHORE_HIS:   return NEFF_PHE;   // ~9.0
        case CHROMOPHORE_SS:    return NEFF_SS;    // 2.0
        default:                return 9.0f;
    }
}
#define get_neff get_neff_cryo
#endif

// Get UV absorption coefficient (backward compatible, relative to TRP=1.0)
__device__ __forceinline__ float get_uv_absorption(int atom_type, int residue_type) {
    switch (residue_type) {
        case CHROMOPHORE_TRP:   return UV_ABSORPTION_TRP;
        case CHROMOPHORE_TYR:   return UV_ABSORPTION_TYR;
        case CHROMOPHORE_PHE:   return UV_ABSORPTION_PHE;
        case CHROMOPHORE_HIS:   return UV_ABSORPTION_HIS;
        case CHROMOPHORE_WATER: return UV_ABSORPTION_WATER;
        case CHROMOPHORE_OTHER:
        default:
            return 0.0f;
    }
}

// PHYSICS-CORRECTED: Calculate temperature rise from UV absorption
// Formula: ΔT = (E_γ × p × η) / (3/2 × k_B × N_eff)
// where p = σ(λ) × F (absorption probability, wavelength-dependent)
//
// This is the PRIMARY UV heating function - use this for all UV pump-probe.
// Legacy calc_temp_rise() should NOT be used for UV heating.
__device__ __forceinline__ float calc_temp_rise_physics(
    int residue_type,           // CHROMOPHORE_TRP, CHROMOPHORE_TYR, etc.
    float wavelength_nm,        // Pump wavelength (nm)
    float photon_fluence,       // Photons/Å²
    float heat_yield            // η: fraction of absorbed energy → heat
) {
    // Get wavelength-dependent cross-section σ(λ)
    float sigma = get_uv_cross_section_at_wavelength(residue_type, wavelength_nm);  // Å²

    // Get photon energy E_γ = hc/λ
    float E_photon = get_photon_energy(wavelength_nm);  // eV

    // Absorption probability (single-photon regime: p << 0.01)
    float p_absorb = sigma * photon_fluence;

    // Energy deposited: E_dep = E_γ × p × η
    float E_dep = E_photon * p_absorb * heat_yield;  // eV

    // Get effective DOF for this chromophore
    float n_eff = get_neff(residue_type);

    // Temperature increase via equipartition
    // ΔT = E_dep / (3/2 × k_B × N_eff)
    return E_dep / (1.5f * KB_EV_K * n_eff);  // Kelvin
}

// LEGACY: Generic thermal calculation - DO NOT USE FOR UV PUMP-PROBE
// Use calc_temp_rise_physics() for UV heating calculations.
// This function is retained only for non-UV thermal features (if any).
__device__ __forceinline__ float calc_temp_rise_legacy(
    float absorbed_energy,      // Energy absorbed (arbitrary units)
    float mass,                 // Mass of absorbing region (amu)
    float heat_capacity         // Effective heat capacity
) {
    // Q = m * c * ΔT  →  ΔT = Q / (m * c)
    // Convert mass from amu to grams: 1 amu = 1.66e-24 g
    float mass_grams = mass * 1.66e-24f;

    // Absorbed energy in joules (scale factor for arbitrary units)
    float energy_joules = absorbed_energy * 1e-21f;  // Tunable scale

    if (mass_grams < 1e-30f || heat_capacity < 1e-6f) return 0.0f;

    return energy_joules / (mass_grams * heat_capacity);
}

// Backward-compatible alias - prefer calc_temp_rise_physics() for UV
#define calc_temp_rise calc_temp_rise_legacy

// Calculate thermal diffusion (heat spreading)
__device__ __forceinline__ float calc_thermal_diffusion(
    float center_temp,
    float neighbor_temps[6],    // +x, -x, +y, -y, +z, -z
    float diffusivity,          // Thermal diffusivity (Å²/ps)
    float dt,                   // Timestep (ps)
    float grid_spacing          // Grid spacing (Å)
) {
    // 3D Laplacian: ∇²T = (T_+x + T_-x + T_+y + T_-y + T_+z + T_-z - 6*T_center) / h²
    float laplacian = 0.0f;
    for (int i = 0; i < 6; i++) {
        laplacian += neighbor_temps[i];
    }
    laplacian -= 6.0f * center_temp;
    laplacian /= (grid_spacing * grid_spacing);
    
    // Heat equation: dT/dt = α * ∇²T
    return diffusivity * laplacian * dt;
}

// Check for phase transition (melting)
__device__ __forceinline__ float calc_melt_rate(
    float temperature,
    float ice_fraction,
    float dt
) {
    if (ice_fraction <= 0.0f) return 0.0f;
    
    // Melting occurs above melting point
    if (temperature > MELTING_POINT) {
        // Rate proportional to superheat
        float superheat = temperature - MELTING_POINT;
        float rate = superheat * 0.01f;  // Tunable rate constant
        return fminf(rate * dt, ice_fraction);  // Can't melt more than exists
    }
    
    return 0.0f;
}

// Calculate temperature gradient at a voxel
__device__ __forceinline__ void calc_temp_gradient(
    float neighbor_temps[6],
    float grid_spacing,
    float3* gradient_dir,
    float* gradient_mag
) {
    // Central difference gradient
    gradient_dir->x = (neighbor_temps[0] - neighbor_temps[1]) / (2.0f * grid_spacing);
    gradient_dir->y = (neighbor_temps[2] - neighbor_temps[3]) / (2.0f * grid_spacing);
    gradient_dir->z = (neighbor_temps[4] - neighbor_temps[5]) / (2.0f * grid_spacing);
    
    *gradient_mag = sqrtf(
        gradient_dir->x * gradient_dir->x +
        gradient_dir->y * gradient_dir->y +
        gradient_dir->z * gradient_dir->z
    );
    
    // Normalize direction
    if (*gradient_mag > 1e-10f) {
        gradient_dir->x /= *gradient_mag;
        gradient_dir->y /= *gradient_mag;
        gradient_dir->z /= *gradient_mag;
    }
}

// ============================================================================
// DEVICE FUNCTIONS: NEUROMORPHIC THERMAL DETECTION
// ============================================================================

// Update LIF neuron for thermal detection
__device__ __forceinline__ bool update_thermal_lif(
    ThermalVoxel* voxel,
    float dt,
    int current_time
) {
    // Check refractory period
    if (voxel->in_refractory) {
        if (current_time - voxel->last_spike_time > THERMAL_REFRACTORY) {
            voxel->in_refractory = false;
        } else {
            return false;
        }
    }
    
    // Input current is the temperature change from baseline
    // VERY sensitive - we want to detect tiny changes
    float input = fabsf(voxel->delta_temp) / THERMAL_SPIKE_THRESHOLD;
    
    // Also respond to gradient (heat flow)
    input += voxel->temp_gradient_mag / THERMAL_GRADIENT_THRESHOLD;
    
    // Also respond to melting
    if (voxel->melt_rate > MELT_FRACTION_THRESHOLD) {
        input += voxel->melt_rate * 10.0f;  // Melting is a strong signal
    }
    
    // Leaky integration
    float decay = expf(-dt / THERMAL_LIF_TAU);
    voxel->lif_potential = voxel->lif_potential * decay + input * (1.0f - decay);
    
    // Check for spike
    if (voxel->lif_potential >= THERMAL_LIF_THRESHOLD) {
        voxel->lif_potential = THERMAL_LIF_RESET;
        voxel->last_spike_time = current_time;
        voxel->in_refractory = true;
        return true;  // SPIKE!
    }
    
    return false;
}

// Detect correlated response across multiple aromatics (pocket signature)
__device__ __forceinline__ float calc_aromatic_correlation(
    float* aromatic_responses,  // Temperature rise at each aromatic
    int n_aromatics,
    float time_window           // ps - window for correlation
) {
    if (n_aromatics < 2) return 0.0f;
    
    // Count how many aromatics responded significantly
    int n_responding = 0;
    float mean_response = 0.0f;
    
    for (int i = 0; i < n_aromatics; i++) {
        if (aromatic_responses[i] > THERMAL_SPIKE_THRESHOLD) {
            n_responding++;
            mean_response += aromatic_responses[i];
        }
    }
    
    if (n_responding < 2) return 0.0f;
    mean_response /= n_responding;
    
    // Calculate correlation as ratio of responding aromatics
    // times similarity of response magnitude
    float response_ratio = (float)n_responding / (float)n_aromatics;
    
    // Variance of responses (lower = more correlated)
    float variance = 0.0f;
    for (int i = 0; i < n_aromatics; i++) {
        if (aromatic_responses[i] > THERMAL_SPIKE_THRESHOLD) {
            float diff = aromatic_responses[i] - mean_response;
            variance += diff * diff;
        }
    }
    variance /= n_responding;
    
    // Correlation score: high when many aromatics respond similarly
    float similarity = 1.0f / (1.0f + variance / (mean_response * mean_response + 1e-6f));
    
    return response_ratio * similarity;
}

// ============================================================================
// KERNEL: APPLY UV PROBE AND TRACK THERMAL RESPONSE (PHYSICS-CORRECTED)
// ============================================================================

__global__ void apply_uv_probe_thermal(
    // Probe parameters (PHYSICS-CORRECTED)
    float3 probe_position,
    float wavelength_nm,        // Pump wavelength (nm) - NEW: enables multi-wavelength
    float photon_fluence,       // Photons/Å² - NEW: calibrated fluence (default: 0.024)
    float heat_yield,           // η: fraction of energy → heat (default: 1.0)
    float probe_radius,         // Effective radius of probe

    // Atom data
    const float3* __restrict__ positions,
    const int* __restrict__ residue_types,  // CHROMOPHORE_TRP, CHROMOPHORE_TYR, etc.
    const float* __restrict__ masses,
    int n_atoms,

    // Thermal grid
    ThermalVoxel* thermal_voxels,
    int3 grid_dim,
    float3 grid_origin,
    float grid_spacing,

    // Output
    UVProbeEvent* probe_result,
    int* spike_count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory for reduction
    __shared__ float s_total_energy_deposited;
    __shared__ float s_max_temp_rise;
    __shared__ int s_spike_count;
    __shared__ int s_hit_aromatic;

    if (threadIdx.x == 0) {
        s_total_energy_deposited = 0.0f;
        s_max_temp_rise = 0.0f;
        s_spike_count = 0;
        s_hit_aromatic = 0;
    }
    __syncthreads();

    // Each thread handles one atom
    if (tid < n_atoms) {
        float3 pos = positions[tid];
        int res_type = residue_types[tid];

        // Distance from probe center
        float3 diff = make_float3(
            pos.x - probe_position.x,
            pos.y - probe_position.y,
            pos.z - probe_position.z
        );
        float dist = sqrtf(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);

        // Within probe radius?
        if (dist < probe_radius) {
            // Gaussian spatial profile for fluence
            float gaussian = expf(-0.5f * (dist * dist) / (probe_radius * probe_radius * 0.25f));
            float local_fluence = photon_fluence * gaussian;

            // PHYSICS-CORRECTED: Calculate temperature rise using proper formula
            // ΔT = (E_γ × p × η) / (3/2 × k_B × N_eff)
            // where p = σ(λ) × F (wavelength-dependent!)
            float temp_rise = calc_temp_rise_physics(
                res_type,
                wavelength_nm,
                local_fluence,
                heat_yield
            );

            if (temp_rise > 0.0f) {
                // Get cross-section for energy accounting
                float sigma = get_uv_cross_section_at_wavelength(res_type, wavelength_nm);
                float E_photon = get_photon_energy(wavelength_nm);
                float p_absorb = sigma * local_fluence;
                float E_dep = E_photon * p_absorb * heat_yield;

                atomicAdd(&s_total_energy_deposited, E_dep);

                // Mark if we hit a chromophore (not water/other)
                if (res_type <= CHROMOPHORE_PHE || res_type == CHROMOPHORE_SS) {
                    atomicExch(&s_hit_aromatic, 1);
                }

                // Update thermal grid
                int vx = (int)((pos.x - grid_origin.x) / grid_spacing);
                int vy = (int)((pos.y - grid_origin.y) / grid_spacing);
                int vz = (int)((pos.z - grid_origin.z) / grid_spacing);

                if (vx >= 0 && vx < grid_dim.x &&
                    vy >= 0 && vy < grid_dim.y &&
                    vz >= 0 && vz < grid_dim.z) {

                    int voxel_idx = vx + vy * grid_dim.x + vz * grid_dim.x * grid_dim.y;

                    // Update voxel temperature
                    atomicAdd(&thermal_voxels[voxel_idx].temperature, temp_rise);
                    atomicAdd(&thermal_voxels[voxel_idx].delta_temp, temp_rise);

                    // Track max temperature rise (atomic max for float)
                    float old_max = s_max_temp_rise;
                    while (temp_rise > old_max) {
                        old_max = atomicMax((int*)&s_max_temp_rise, __float_as_int(temp_rise));
                        old_max = __int_as_float((int)old_max);
                    }
                }
            }
        }
    }
    __syncthreads();

    // Thread 0 writes probe result
    if (threadIdx.x == 0 && probe_result != nullptr) {
        probe_result->target_position = probe_position;
        // Store calibrated fluence as "probe_energy" for compatibility
        probe_result->probe_energy = photon_fluence;
        // Total energy deposited (eV) rather than absorption fraction
        probe_result->absorption_fraction = s_total_energy_deposited;
        probe_result->temp_rise = s_max_temp_rise;
        probe_result->hit_aromatic = (s_hit_aromatic > 0);
        probe_result->response_spikes = s_spike_count;
    }
}

// ============================================================================
// KERNEL: UPDATE THERMAL FIELD (DIFFUSION + LIF)
// ============================================================================

__global__ void update_thermal_field(
    ThermalVoxel* voxels,
    int3 grid_dim,
    float grid_spacing,
    float dt,
    int current_time,
    float thermal_diffusivity,
    
    // Output
    int* spike_voxels,
    int* n_spikes
) {
    int vx = blockIdx.x * blockDim.x + threadIdx.x;
    int vy = blockIdx.y * blockDim.y + threadIdx.y;
    int vz = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (vx >= grid_dim.x || vy >= grid_dim.y || vz >= grid_dim.z) return;
    
    int idx = vx + vy * grid_dim.x + vz * grid_dim.x * grid_dim.y;
    ThermalVoxel* voxel = &voxels[idx];
    
    // Get neighbor temperatures
    float neighbor_temps[6];
    
    // +x
    if (vx < grid_dim.x - 1)
        neighbor_temps[0] = voxels[idx + 1].temperature;
    else
        neighbor_temps[0] = voxel->temperature;
    
    // -x
    if (vx > 0)
        neighbor_temps[1] = voxels[idx - 1].temperature;
    else
        neighbor_temps[1] = voxel->temperature;
    
    // +y
    if (vy < grid_dim.y - 1)
        neighbor_temps[2] = voxels[idx + grid_dim.x].temperature;
    else
        neighbor_temps[2] = voxel->temperature;
    
    // -y
    if (vy > 0)
        neighbor_temps[3] = voxels[idx - grid_dim.x].temperature;
    else
        neighbor_temps[3] = voxel->temperature;
    
    // +z
    int z_stride = grid_dim.x * grid_dim.y;
    if (vz < grid_dim.z - 1)
        neighbor_temps[4] = voxels[idx + z_stride].temperature;
    else
        neighbor_temps[4] = voxel->temperature;
    
    // -z
    if (vz > 0)
        neighbor_temps[5] = voxels[idx - z_stride].temperature;
    else
        neighbor_temps[5] = voxel->temperature;
    
    // Calculate thermal diffusion
    float dT = calc_thermal_diffusion(
        voxel->temperature,
        neighbor_temps,
        thermal_diffusivity,
        dt,
        grid_spacing
    );
    
    // Update temperature
    voxel->temperature += dT;
    voxel->delta_temp = voxel->temperature - voxel->baseline_temp;
    
    // Calculate gradient
    calc_temp_gradient(
        neighbor_temps,
        grid_spacing,
        &voxel->temp_gradient_dir,
        &voxel->temp_gradient_mag
    );
    
    // Check for melting
    voxel->melt_rate = calc_melt_rate(voxel->temperature, voxel->ice_fraction, dt);
    voxel->ice_fraction -= voxel->melt_rate;
    if (voxel->ice_fraction < 0.0f) voxel->ice_fraction = 0.0f;
    
    // Neuromorphic LIF update
    bool spiked = update_thermal_lif(voxel, dt, current_time);
    
    if (spiked) {
        int spike_idx = atomicAdd(n_spikes, 1);
        if (spike_idx < grid_dim.x * grid_dim.y * grid_dim.z) {
            spike_voxels[spike_idx] = idx;
        }
    }
}

// ============================================================================
// KERNEL: DETECT POCKET SIGNATURES
// ============================================================================

__global__ void detect_pocket_signatures(
    // Aromatic data
    const float3* __restrict__ aromatic_centers,
    const float* __restrict__ aromatic_responses,  // Temp rise at each aromatic
    int n_aromatics,
    
    // Thermal grid for context
    const ThermalVoxel* __restrict__ voxels,
    int3 grid_dim,
    float3 grid_origin,
    float grid_spacing,
    
    // Output
    ThermalPocketSignature* signatures,
    int* n_signatures,
    int max_signatures,
    
    // Parameters
    float min_correlation,
    float min_temp_rise
) {
    // Single thread kernel for now (could parallelize over aromatic pairs)
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    // Find clusters of responding aromatics
    // Simple approach: for each aromatic with significant response,
    // check for nearby aromatics that also responded
    
    for (int i = 0; i < n_aromatics && *n_signatures < max_signatures; i++) {
        if (aromatic_responses[i] < min_temp_rise) continue;
        
        // Count nearby responding aromatics
        float3 center = aromatic_centers[i];
        float total_response = aromatic_responses[i];
        int responding[8];
        float responses[8];
        int n_responding = 1;
        responding[0] = i;
        responses[0] = aromatic_responses[i];
        
        for (int j = i + 1; j < n_aromatics && n_responding < 8; j++) {
            if (aromatic_responses[j] < min_temp_rise * 0.5f) continue;
            
            float3 other = aromatic_centers[j];
            float3 diff = make_float3(
                other.x - center.x,
                other.y - center.y,
                other.z - center.z
            );
            float dist = sqrtf(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
            
            // Within pocket-like distance (< 15 Å)
            if (dist < 15.0f) {
                responding[n_responding] = j;
                responses[n_responding] = aromatic_responses[j];
                total_response += aromatic_responses[j];
                
                // Update center (weighted average)
                float w = aromatic_responses[j] / total_response;
                center.x = center.x * (1.0f - w) + other.x * w;
                center.y = center.y * (1.0f - w) + other.y * w;
                center.z = center.z * (1.0f - w) + other.z * w;
                
                n_responding++;
            }
        }
        
        // Need at least 2 aromatics for a pocket signature
        if (n_responding >= 2) {
            float correlation = calc_aromatic_correlation(responses, n_responding, 10.0f);
            
            if (correlation >= min_correlation) {
                // Found a pocket signature!
                int sig_idx = atomicAdd(n_signatures, 1);
                if (sig_idx < max_signatures) {
                    ThermalPocketSignature* sig = &signatures[sig_idx];
                    sig->center = center;
                    sig->peak_temp_rise = total_response / n_responding;
                    sig->n_responding_aromatics = n_responding;
                    for (int k = 0; k < n_responding; k++) {
                        sig->responding_aromatics[k] = responding[k];
                    }
                    sig->response_correlation = correlation;
                    sig->confidence = correlation * (float)n_responding / 8.0f;
                    sig->detection_count = 1;
                    
                    // Estimate radius from aromatic spread
                    float max_dist = 0.0f;
                    for (int k = 0; k < n_responding; k++) {
                        float3 pos = aromatic_centers[responding[k]];
                        float3 diff = make_float3(
                            pos.x - center.x,
                            pos.y - center.y,
                            pos.z - center.z
                        );
                        float dist = sqrtf(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
                        if (dist > max_dist) max_dist = dist;
                    }
                    sig->radius = max_dist + 3.0f;  // Add buffer
                }
            }
        }
    }
}

// ============================================================================
// HOST HELPER: INITIALIZE THERMAL GRID
// ============================================================================

__host__ void init_cryo_thermal_grid(
    CryoThermalState* state,
    float3 min_bounds,
    float3 max_bounds,
    float ambient_temp,
    float grid_spacing
) {
    // Calculate grid dimensions
    state->grid_spacing = grid_spacing;
    state->grid_origin = min_bounds;
    
    state->grid_dim.x = (int)ceilf((max_bounds.x - min_bounds.x) / grid_spacing) + 1;
    state->grid_dim.y = (int)ceilf((max_bounds.y - min_bounds.y) / grid_spacing) + 1;
    state->grid_dim.z = (int)ceilf((max_bounds.z - min_bounds.z) / grid_spacing) + 1;
    
    state->n_voxels = state->grid_dim.x * state->grid_dim.y * state->grid_dim.z;
    
    // Allocate voxels
    cudaMalloc(&state->voxels, state->n_voxels * sizeof(ThermalVoxel));
    
    // Initialize to cryo temperature
    state->ambient_temp = ambient_temp;
    state->cryo_temp = ambient_temp;
    
    // Would need a kernel to initialize voxels...
}

#endif // PRISM_CRYO_THERMAL_DETECTION_CUH
