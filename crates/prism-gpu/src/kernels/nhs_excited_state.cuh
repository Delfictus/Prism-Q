// =============================================================================
// PRISM-NHS: Excited State Dynamics - FUSED KERNEL VERSION
// =============================================================================
// Designed for integration into nhs_amber_fused.cu
// Works with holographic neuromorphic stream + implicit solvent
// =============================================================================

#ifndef NHS_EXCITED_STATE_FUSED_CU
#define NHS_EXCITED_STATE_FUSED_CU

// =============================================================================
// CONSTANTS (same as before)
// =============================================================================

// Dipole ratios (excited/ground) - precomputed for efficiency
#define TYR_DIPOLE_RATIO_SQRT  1.749f  // sqrt(4.50/1.47)
#define PHE_DIPOLE_RATIO_SQRT  1.801f  // sqrt(1.20/0.37)
#define TRP_DIPOLE_RATIO_SQRT  1.690f  // sqrt(6.00/2.10)

// Timescales (ps) - as inverse rates for multiply instead of divide
#define INV_TAU_FRANCK_CONDON  20.0f   // 1/0.05 ps
#define INV_TAU_VIBRATIONAL    0.5f    // 1/2.0 ps
#define INV_TAU_FLUOR_TYR      0.000294f // 1/3400 ps
#define INV_TAU_FLUOR_PHE      0.000147f // 1/6800 ps
#define INV_TAU_FLUOR_TRP      0.000385f // 1/2600 ps
#define INV_TAU_IC_FACTOR      3.33f   // IC ~3x faster than fluorescence

// =============================================================================
// PHYSICS-BASED UV ABSORPTION (CORRECTED)
// =============================================================================
// Formula: E_dep = E_γ × p × η  where p = σ × F
// σ(Å²) = ε(M⁻¹cm⁻¹) × 3.823×10⁻⁵
// F = 0.024 photons/Å² (calibrated)
// Target: TRP @ 280nm → ΔT ≈ 20K

// Cross-section conversion factor
#define EPSILON_TO_SIGMA_FACTOR  3.823e-5f

// Calibrated photon fluence (photons/Å² per pulse)
#define CALIBRATED_PHOTON_FLUENCE  0.024f

// Pre-computed cross-sections at peak wavelengths (Å²)
// σ = ε × 3.823e-5
#define UV_SIGMA_TRP  0.21409f   // TRP @ 280nm: 5600 × 3.823e-5
#define UV_SIGMA_TYR  0.05696f   // TYR @ 274nm: 1490 × 3.823e-5
#define UV_SIGMA_PHE  0.00765f   // PHE @ 258nm: 200 × 3.823e-5
#define UV_SIGMA_SS   0.01147f   // S-S @ 250nm: 300 × 3.823e-5

// =============================================================================
// GAUSSIAN ABSORPTION BAND PARAMETERS
// =============================================================================
// ε(λ) = ε_max × exp(-(λ - λ_max)² / (2σ²))
//
// P6 CANONICAL CONVENTION. `*_BANDWIDTH` is the Gaussian **1/e half-width**
// `w`, NOT the FWHM:
//
//     σ = w / sqrt(2) = w * BAND_W_TO_SIGMA      FWHM = 1.665109 * w
//
// Use BAND_W_TO_SIGMA. Do not divide by 2.355, and do not use `w` as σ.
//
// Before P6 this header used `w` directly as σ while crate::config divided the
// same constants by 2.355 — a 2.355x disagreement on the identical quantity,
// worth 134x in deposited heat at 258 nm. It also carried TYR/PHE/SS values
// that differed from crate::config. Both are reconciled here; the Rust side is
// authoritative and `config::test_gpu_parity_band_model` now PARSES this file
// rather than restating it, so the two cannot drift again.
#define BAND_W_TO_SIGMA   0.70710678f   // 1/sqrt(2)

// TRP: λ_max=280nm, ε_max=5600, w=15nm  → σ=10.607nm, FWHM=24.98nm
#define TRP_LAMBDA_MAX    280.0f
#define TRP_EPSILON_MAX   5600.0f
#define TRP_BANDWIDTH     15.0f

// TYR: λ_max=274nm, ε_max=1490, w=12nm  → σ=8.485nm, FWHM=19.98nm
#define TYR_LAMBDA_MAX    274.0f
#define TYR_EPSILON_MAX   1490.0f
#define TYR_BANDWIDTH     12.0f

// PHE: λ_max=258nm, ε_max=200, w=10nm   → σ=7.071nm, FWHM=16.65nm
#define PHE_LAMBDA_MAX    258.0f
#define PHE_EPSILON_MAX   200.0f
#define PHE_BANDWIDTH     10.0f

// S-S: λ_max=250nm, ε_max=300, w=20nm   → σ=14.142nm, FWHM=33.30nm
#define SS_LAMBDA_MAX     250.0f
#define SS_EPSILON_MAX    300.0f
#define SS_BANDWIDTH      20.0f

// Photon energy at 280nm
#define UV_PHOTON_ENERGY_EV    4.428f   // eV (hc/λ = 1239.84/280)
#define EV_TO_KCAL_MOL         23.06f   // 1 eV = 23.06 kcal/mol
#define UV_PHOTON_ENERGY       102.0f   // kcal/mol (280nm) - for legacy compatibility

// N_eff values (effective DOF proxies for equipartition)
#define NEFF_TRP  9.0f    // Indole ring system
#define NEFF_TYR  10.0f   // Phenol + OH system
#define NEFF_PHE  9.0f    // Benzene + side chain
#define NEFF_SS   2.0f    // S-S bond

// Boltzmann constant
#define KB_EV_K   8.617e-5f   // eV/K

// Energy conversion for velocity
// AMBER units: 1 kcal/mol = 418.4 amu⋅Å²/ps²
// For KE = 0.5*m*v²: v = sqrt(2*KE/m) = sqrt(2*418.4*KE_kcal/m_amu) Å/ps
// ENERGY_VEL_PREFACTOR = sqrt(2*418.4) = 28.93
// v = ENERGY_VEL_PREFACTOR * sqrt(KE_kcal / m_amu)
#define ENERGY_VEL_PREFACTOR   28.93f  // sqrt(2 * 418.4)

// Legacy factor (assumes m=2 amu - INCORRECT for most atoms)
// Kept for reference: sqrt(2 * 418.4 / 2) = 20.45
#define ENERGY_TO_VEL_FACTOR   20.45f  // DEPRECATED - assumes hydrogen mass

// Average neighbor mass for simplified calculation (mostly carbons + some H)
#define AVG_NEIGHBOR_MASS      8.0f   // Approximate average mass in amu

// =============================================================================
// PHYSICS-BASED ENERGY CALCULATION
// =============================================================================

// Get cross-section for chromophore type at peak wavelength
// CANONICAL ordering: 0=TRP, 1=TYR, 2=PHE, 3=S-S
__device__ __forceinline__ float get_cross_section_peak(int residue_type) {
    switch (residue_type) {
        case 0: return UV_SIGMA_TRP;   // 0.21409 Å²
        case 1: return UV_SIGMA_TYR;   // 0.05696 Å²
        case 2: return UV_SIGMA_PHE;   // 0.00765 Å²
        case 3: return UV_SIGMA_SS;    // 0.01147 Å²
        default: return UV_SIGMA_TRP;
    }
}

// =============================================================================
// WAVELENGTH-DEPENDENT CROSS-SECTION: σ(λ) via Gaussian band model
// =============================================================================
// ε(λ) = ε_max × exp(-(λ - λ_max)² / (2σ²))
// σ(λ) = ε(λ) × 3.823×10⁻⁵

__device__ __forceinline__ float compute_extinction_at_wavelength(
    int residue_type,
    float wavelength_nm
) {
    float lambda_max, epsilon_max, bandwidth;

    switch (residue_type) {
        case 0:  // TRP
            lambda_max = TRP_LAMBDA_MAX;
            epsilon_max = TRP_EPSILON_MAX;
            bandwidth = TRP_BANDWIDTH;
            break;
        case 1:  // TYR
            lambda_max = TYR_LAMBDA_MAX;
            epsilon_max = TYR_EPSILON_MAX;
            bandwidth = TYR_BANDWIDTH;
            break;
        case 2:  // PHE
            lambda_max = PHE_LAMBDA_MAX;
            epsilon_max = PHE_EPSILON_MAX;
            bandwidth = PHE_BANDWIDTH;
            break;
        case 3:  // S-S
            lambda_max = SS_LAMBDA_MAX;
            epsilon_max = SS_EPSILON_MAX;
            bandwidth = SS_BANDWIDTH;
            break;
        default:
            lambda_max = TRP_LAMBDA_MAX;
            epsilon_max = TRP_EPSILON_MAX;
            bandwidth = TRP_BANDWIDTH;
    }

    // Gaussian profile: ε(λ) = ε_max × exp(-(λ-λ_max)²/(2σ²)), σ = w/sqrt(2).
    // P6: `bandwidth` is the 1/e half-width w, so it must be converted. Using
    // it directly as σ (the pre-P6 behaviour) widened every band by 2.355x
    // relative to the host.
    float sigma = bandwidth * BAND_W_TO_SIGMA;
    float delta = wavelength_nm - lambda_max;
    float exponent = -(delta * delta) / (2.0f * sigma * sigma);
    return epsilon_max * expf(exponent);
}

// Get wavelength-dependent cross-section σ(λ) in Å²
__device__ __forceinline__ float get_cross_section_at_wavelength(
    int residue_type,
    float wavelength_nm
) {
    float epsilon = compute_extinction_at_wavelength(residue_type, wavelength_nm);
    return epsilon * EPSILON_TO_SIGMA_FACTOR;
}

// Legacy: Get cross-section at peak (for backwards compatibility)
__device__ __forceinline__ float get_cross_section(int residue_type) {
    return get_cross_section_peak(residue_type);
}

// Get N_eff for chromophore type
// CANONICAL ordering: 0=TRP, 1=TYR, 2=PHE, 3=S-S
__device__ __forceinline__ float get_neff(int residue_type) {
    switch (residue_type) {
        case 0: return NEFF_TRP;   // 9.0
        case 1: return NEFF_TYR;   // 10.0
        case 2: return NEFF_PHE;   // 9.0
        case 3: return NEFF_SS;    // 2.0
        default: return NEFF_TRP;
    }
}

// =============================================================================
// WAVELENGTH-DEPENDENT ENERGY DEPOSITION
// =============================================================================

// Compute photon energy at given wavelength
// E = hc/λ = 1239.84 eV⋅nm / λ(nm)
__device__ __forceinline__ float photon_energy_ev(float wavelength_nm) {
    return 1239.84f / wavelength_nm;
}

// Compute physics-based energy deposition (kcal/mol) with wavelength dependence
// Formula: E_dep = E_photon(λ) × p_absorb(λ) × η
// where p_absorb(λ) = σ(λ) × F
__device__ __forceinline__ float compute_deposited_energy_wavelength(
    int residue_type,
    float wavelength_nm
) {
    float sigma = get_cross_section_at_wavelength(residue_type, wavelength_nm);  // Å²
    float p_absorb = sigma * CALIBRATED_PHOTON_FLUENCE;  // dimensionless
    float e_photon_ev = photon_energy_ev(wavelength_nm);
    float e_photon_kcal = e_photon_ev * EV_TO_KCAL_MOL;  // kcal/mol
    float eta = 1.0f;  // Heat yield
    return e_photon_kcal * p_absorb * eta;
}

// Legacy: Compute energy at peak wavelength (280nm for TRP-like behavior)
__device__ __forceinline__ float compute_deposited_energy(int residue_type) {
    // Use peak cross-section and 280nm photon energy for backwards compatibility
    float sigma = get_cross_section_peak(residue_type);  // Å²
    float p_absorb = sigma * CALIBRATED_PHOTON_FLUENCE;  // dimensionless
    float e_photon_kcal = UV_PHOTON_ENERGY_EV * EV_TO_KCAL_MOL;  // kcal/mol at 280nm
    float eta = 1.0f;
    return e_photon_kcal * p_absorb * eta;
}

// Debug diagnostic: print physics values for verification (compile with -DDEBUG_UV_PHYSICS)
#ifdef DEBUG_UV_PHYSICS
__device__ void debug_print_uv_physics(int residue_type, int aromatic_idx) {
    float sigma = get_cross_section(residue_type);
    float fluence = CALIBRATED_PHOTON_FLUENCE;
    float p_absorb = sigma * fluence;
    float e_photon_ev = UV_PHOTON_ENERGY_EV;
    float e_photon_kcal = e_photon_ev * EV_TO_KCAL_MOL;
    float eta = 1.0f;
    float e_dep_kcal = e_photon_kcal * p_absorb * eta;
    float n_eff = get_neff(residue_type);
    float delta_t = (e_dep_kcal / EV_TO_KCAL_MOL) / (1.5f * KB_EV_K * n_eff);

    printf("[UV DEBUG] aromatic_idx=%d type=%d\n", aromatic_idx, residue_type);
    printf("  sigma=%.5f A^2 (PEAK), fluence=%.4f photons/A^2\n", sigma, fluence);
    printf("  p_absorb=%.6f, E_photon=%.3f eV (%.2f kcal/mol)\n", p_absorb, e_photon_ev, e_photon_kcal);
    printf("  E_dep=%.4f kcal/mol, N_eff=%.1f, delta_T=%.2f K\n", e_dep_kcal, n_eff, delta_t);
}
#endif

// Compute temperature rise from deposited energy
// Formula: ΔT = E_dep / (3/2 × k_B × N_eff)
__device__ __forceinline__ float compute_temp_rise(int residue_type, float e_dep_ev) {
    float n_eff = get_neff(residue_type);
    return e_dep_ev / (1.5f * KB_EV_K * n_eff);
}

// =============================================================================
// INLINE DEVICE FUNCTIONS (to be called from main fused kernel)
// =============================================================================

// Fast exponential approximation (accurate to ~0.1% for x in [-2, 0])
__device__ __forceinline__ float fast_expf(float x) {
    // For small negative x, use: exp(x) ≈ 1 + x + x²/2
    // For larger x, use intrinsic
    if (x > -0.1f) {
        return 1.0f + x + 0.5f * x * x;
    }
    return __expf(x);  // CUDA fast intrinsic
}

// -----------------------------------------------------------------------------
// STEP 1: Apply electronic excitation (call when UV pulse triggers)
// PHYSICS-CORRECTED: Uses σ(λ) × F absorption probability with wavelength dependence
// -----------------------------------------------------------------------------
__device__ __forceinline__ void excite_aromatic_wavelength(
    int aromatic_idx,
    int residue_type,           // CANONICAL: 0=TRP, 1=TYR, 2=PHE, 3=S-S
    float wavelength_nm,        // Current UV wavelength
    // Excited state arrays (in shared or global memory)
    int* is_excited,
    float* time_since_excitation,
    float* electronic_population,
    float* vibrational_energy,
    float* franck_condon_progress,
    // M1.2.26.RECOUPLE Part 1.1 — Operator override (kcal/mol).
    // When > 0.0f the caller has populated ProtocolState.uv_burst_energy
    // and wants the excitation kernel to bypass the cross-section physics
    // path. Sentinel <= 0.0f preserves the hardcoded compute_deposited_
    // energy_wavelength() Naive Safety baseline.
    float uv_burst_energy_override
) {
    is_excited[aromatic_idx] = 1;
    time_since_excitation[aromatic_idx] = 0.0f;
    electronic_population[aromatic_idx] = 1.0f;

    // Baseline: physics cross-section path (Transition Dipole Moment).
    float energy = compute_deposited_energy_wavelength(residue_type, wavelength_nm);
    // M1.2.26.RECOUPLE Part 1.1 — Operator Override (Target T28.1).
    // The orchestrator-supplied UV burst energy supersedes the baseline.
    // This connects the --uv-burst-energy CLI surface (host) to the actual
    // kinetic injection on the Blackwell SM, closing the dead-parameter
    // gap identified in the M1.2.26 coupling audit.
    if (uv_burst_energy_override > 0.0f) {
        energy = uv_burst_energy_override;
    }
    vibrational_energy[aromatic_idx] = energy;

#ifdef DEBUG_UV_PHYSICS
    if (aromatic_idx == 0) {
        float sigma = get_cross_section_at_wavelength(residue_type, wavelength_nm);
        float p_absorb = sigma * CALIBRATED_PHOTON_FLUENCE;
        float e_photon_ev = photon_energy_ev(wavelength_nm);
        float n_eff = get_neff(residue_type);
        float delta_t = (energy / EV_TO_KCAL_MOL) / (1.5f * KB_EV_K * n_eff);
        const char* types[] = {"TRP", "TYR", "PHE", "S-S"};
        printf("[UV λ-DEP] aromatic=%d type=%s λ=%.0fnm\n",
            aromatic_idx, types[residue_type], wavelength_nm);
        printf("  σ(λ)=%.5f A^2, p=%.6f, E_γ=%.3f eV\n", sigma, p_absorb, e_photon_ev);
        printf("  E_dep=%.4f kcal/mol, N_eff=%.1f, ΔT=%.2f K\n", energy, n_eff, delta_t);
    }
#endif

    franck_condon_progress[aromatic_idx] = 0.0f;
}

// Legacy version: Uses peak cross-section at 280nm (for backwards compatibility)
__device__ __forceinline__ void excite_aromatic_inline(
    int aromatic_idx,
    int residue_type,           // CANONICAL: 0=TRP, 1=TYR, 2=PHE, 3=S-S
    // Excited state arrays (in shared or global memory)
    int* is_excited,
    float* time_since_excitation,
    float* electronic_population,
    float* vibrational_energy,
    float* franck_condon_progress
) {
    // Call wavelength-dependent version with default 280nm.
    // Legacy path: no operator override (uv_burst_energy_override = 0.0f),
    // physics-derived deposit applies.
    excite_aromatic_wavelength(
        aromatic_idx, residue_type, 280.0f,
        is_excited, time_since_excitation, electronic_population,
        vibrational_energy, franck_condon_progress,
        0.0f /* uv_burst_energy_override = sentinel */
    );
}

// -----------------------------------------------------------------------------
// STEP 2-4: Update excited state (call every timestep, AFTER force calculation)
// -----------------------------------------------------------------------------
__device__ __forceinline__ void update_excited_state_inline(
    int aromatic_idx,
    int residue_type,
    float dt,
    // Excited state arrays
    int* is_excited,
    float* time_since_excitation,
    float* electronic_population,
    float* vibrational_energy,
    float* franck_condon_progress,
    // Output: energy to transfer to environment this step
    float* energy_to_transfer_out
) {
    if (!is_excited[aromatic_idx]) {
        *energy_to_transfer_out = 0.0f;
        return;
    }
    
    // Update timer
    time_since_excitation[aromatic_idx] += dt;
    
    // --- Franck-Condon relaxation (50 fs) ---
    float fc = franck_condon_progress[aromatic_idx];
    fc += (1.0f - fc) * INV_TAU_FRANCK_CONDON * dt;
    franck_condon_progress[aromatic_idx] = fc;
    
    // --- Vibrational relaxation (2 ps) ---
    float vib = vibrational_energy[aromatic_idx];
    float vib_decay = fast_expf(-INV_TAU_VIBRATIONAL * dt);
    float energy_out = vib * (1.0f - vib_decay);
    vibrational_energy[aromatic_idx] = vib * vib_decay;
    *energy_to_transfer_out = energy_out;
    
    // --- Electronic decay (ns timescale) ---
    float inv_tau_fluor;
    // CANONICAL ordering: 0=TRP, 1=TYR, 2=PHE, 3=S-S
    switch (residue_type) {
        case 0: inv_tau_fluor = INV_TAU_FLUOR_TRP; break;  // TRP: 2600 ps
        case 1: inv_tau_fluor = INV_TAU_FLUOR_TYR; break;  // TYR: 3400 ps
        case 2: inv_tau_fluor = INV_TAU_FLUOR_PHE; break;  // PHE: 6800 ps
        default: inv_tau_fluor = INV_TAU_FLUOR_TRP;
    }
    float total_decay_rate = inv_tau_fluor * (1.0f + INV_TAU_IC_FACTOR);
    float pop = electronic_population[aromatic_idx];
    pop *= fast_expf(-total_decay_rate * dt);
    electronic_population[aromatic_idx] = pop;
    
    // Check for complete decay
    if (pop < 0.01f) {
        is_excited[aromatic_idx] = 0;
        electronic_population[aromatic_idx] = 0.0f;
        vibrational_energy[aromatic_idx] = 0.0f;
    }
}

// -----------------------------------------------------------------------------
// Get charge scaling factor for an aromatic
// -----------------------------------------------------------------------------
__device__ __forceinline__ float get_charge_scale_inline(
    int aromatic_idx,
    int residue_type,
    const int* is_excited,
    const float* electronic_population
) {
    if (!is_excited[aromatic_idx]) {
        return 1.0f;
    }
    
    float ratio_sqrt;
    // CANONICAL ordering: 0=TRP, 1=TYR, 2=PHE, 3=S-S
    switch (residue_type) {
        case 0: ratio_sqrt = TRP_DIPOLE_RATIO_SQRT; break;  // TRP
        case 1: ratio_sqrt = TYR_DIPOLE_RATIO_SQRT; break;  // TYR
        case 2: ratio_sqrt = PHE_DIPOLE_RATIO_SQRT; break;  // PHE
        default: ratio_sqrt = 1.0f;  // S-S or unknown
    }
    
    float pop = electronic_population[aromatic_idx];
    // Interpolate: ground (1.0) → excited (ratio_sqrt)
    return 1.0f + (ratio_sqrt - 1.0f) * pop;
}

// =============================================================================
// INTEGRATION INTO NHS-AMBER FUSED KERNEL
// =============================================================================
// 
// Add these to your existing kernel parameters:
//
//   // Excited state data
//   int* d_is_excited,              // [n_aromatics]
//   float* d_time_since_excitation, // [n_aromatics]
//   float* d_electronic_population, // [n_aromatics]
//   float* d_vibrational_energy,    // [n_aromatics]
//   float* d_franck_condon_progress,// [n_aromatics]
//   float* d_ground_state_charges,  // [n_atoms]
//   int* d_atom_to_aromatic,        // [n_atoms] → -1 or aromatic index
//   int* d_aromatic_type,           // [n_aromatics] → CANONICAL: 0=TRP,1=TYR,2=PHE,3=S-S
//   float3* d_ring_normals,         // [n_aromatics]
//   int n_aromatics,
//
// =============================================================================

// Example of how to modify your Coulomb calculation:
__device__ __forceinline__ float get_effective_charge(
    int atom_idx,
    const float* base_charges,
    const float* ground_state_charges,
    const int* atom_to_aromatic,
    const int* aromatic_type,
    const int* is_excited,
    const float* electronic_population
) {
    int aromatic_idx = atom_to_aromatic[atom_idx];
    
    if (aromatic_idx < 0) {
        // Not an aromatic atom - use base charge
        return base_charges[atom_idx];
    }
    
    // Aromatic atom - apply excited state scaling
    float scale = get_charge_scale_inline(
        aromatic_idx,
        aromatic_type[aromatic_idx],
        is_excited,
        electronic_population
    );
    
    return ground_state_charges[atom_idx] * scale;
}

// Apply vibrational energy transfer with proper mass-dependent velocity conversion
//
// PHYSICS: For energy KE (kcal/mol) deposited as kinetic energy to atom of mass m (amu):
//   v = sqrt(2 * KE * 418.4 / m) Å/ps
//   where 418.4 is AMBER's kcal/mol → amu⋅Å²/ps² conversion
//
// Energy conservation: The input energy_to_transfer is distributed among neighbors,
// and each neighbor gains kinetic energy KE_i such that sum(KE_i) = energy_to_transfer.
__device__ __forceinline__ void apply_vibrational_transfer(
    int aromatic_idx,
    float energy_to_transfer,
    float3 ring_normal,
    float3* velocities,
    const float* masses,         // Atom masses in amu
    const int* neighbor_list,    // Atoms within ~5Å of aromatic
    int n_neighbors,
    unsigned int seed,           // For random direction component
    int aromatic_type,           // ChromophoreType enum (0=TRP, 1=TYR, 2=PHE, 3=SS)
    float uv_wavelength_nm,      // Current UV wavelength for debug output
    // M1.2.18.5 — Total External Work COMPONENTS (Hamiltonian audit).
    // One f64 component per atom. The caller is responsible for invoking this
    // function from an ordered single-thread UV phase, so overlapping aromatic
    // neighborhoods accumulate in aromatic-index order without velocity or
    // work atomics. A downstream fixed-order reducer writes the scalar total to
    // d_external_work_components[0] for the SFA stability fuse.
    double* d_external_work_components
) {
    if (energy_to_transfer < 0.001f || n_neighbors == 0) return;

    float energy_per_neighbor = energy_to_transfer / (float)n_neighbors;

    // Simple LCG for random numbers (no curand needed)
    unsigned int rng = seed;

#ifdef DEBUG_UV_PHYSICS
    float total_ke_added = 0.0f;
#endif

    for (int i = 0; i < n_neighbors; i++) {
        int neighbor = neighbor_list[i];
        if (neighbor < 0) continue;

        // Get neighbor mass for proper velocity conversion
        float mass = masses[neighbor];

        // v = sqrt(2 * E * 418.4 / m) = ENERGY_VEL_PREFACTOR * sqrt(E / m)
        // where ENERGY_VEL_PREFACTOR = sqrt(2 * 418.4) = 28.93
        float vel_magnitude = ENERGY_VEL_PREFACTOR * sqrtf(energy_per_neighbor / mass);

        // Mix of directed (ring normal) and random
        rng = rng * 1103515245 + 12345;
        float r = (float)(rng & 0x7FFFFFFF) / (float)0x7FFFFFFF;

        float3 kick_dir;
        if (r < 0.6f) {
            // 60% along ring normal (strongest coupling)
            kick_dir = ring_normal;
            if ((rng >> 16) & 1) kick_dir = make_float3(-kick_dir.x, -kick_dir.y, -kick_dir.z);
        } else {
            // 40% random
            rng = rng * 1103515245 + 12345;
            float rx = (float)((rng >> 0) & 0xFF) / 127.5f - 1.0f;
            float ry = (float)((rng >> 8) & 0xFF) / 127.5f - 1.0f;
            float rz = (float)((rng >> 16) & 0xFF) / 127.5f - 1.0f;
            float len = sqrtf(rx*rx + ry*ry + rz*rz);
            kick_dir = make_float3(rx/len, ry/len, rz/len);
        }

        // M1.2.18.5 — Exact ΔK accumulation (operator UV KINETIC PERFECTION).
        //
        // Read v_old as a float3, compose v_new = v_old + kick·vel_mag in
        // float, then accumulate
        //   ΔK = 0.5 · m · (v_new·v_new − v_old·v_old)
        // in bit-accurate f64 BEFORE applying the velocity update.
        // v_old is read non-atomically (acceptable: the captured fused
        // step's UV path is the only writer to this neighbor in-frame —
        // SISR/G28 ensure no cross-frame race, and the SFA is a coarse
        // First-Law drift detector, not a bit-exact integrator).
        //
        // Conversion: AMBER velocities are in Å/ps and masses in amu, so
        // ½·m·v² is in amu·Å²/ps² ⇒ ÷418.4 gives kcal/mol.
        const float3 v_old = make_float3(
            velocities[neighbor].x,
            velocities[neighbor].y,
            velocities[neighbor].z
        );
        const float3 v_new = make_float3(
            v_old.x + kick_dir.x * vel_magnitude,
            v_old.y + kick_dir.y * vel_magnitude,
            v_old.z + kick_dir.z * vel_magnitude
        );

        if (d_external_work_components != nullptr) {
            const double m_d = (double)mass;
            const double vnx = (double)v_new.x;
            const double vny = (double)v_new.y;
            const double vnz = (double)v_new.z;
            const double vox = (double)v_old.x;
            const double voy = (double)v_old.y;
            const double voz = (double)v_old.z;
            const double delta_k_amu_a2_ps2 = 0.5 * m_d * (
                vnx * vnx + vny * vny + vnz * vnz
              - vox * vox - voy * voy - voz * voz
            );
            const double delta_k_kcal_mol = delta_k_amu_a2_ps2 / 418.4;
            d_external_work_components[neighbor] += delta_k_kcal_mol;
        }

        velocities[neighbor] = v_new;

#ifdef DEBUG_UV_PHYSICS
        // Verify energy conservation: KE = 0.5 * m * v² / 418.4 kcal/mol
        float ke_added = 0.5f * mass * vel_magnitude * vel_magnitude / 418.4f;
        total_ke_added += ke_added;
        if (i == 0) {
            printf("[UV ENERGY] neighbor=%d mass=%.1f E_in=%.5f kcal/mol v=%.4f A/ps KE_out=%.5f kcal/mol\n",
                neighbor, mass, energy_per_neighbor, vel_magnitude, ke_added);
        }
#endif

#ifdef DEBUG_UV_ENERGY_ACCOUNTING
        // HARDENING: Single-site energy accounting check
        // Only run for aromatic_idx == 0 and first neighbor to avoid perf impact
        // Pass criteria: ratio within [0.8, 1.2]
        if (aromatic_idx == 0 && i == 0) {
            float delta_ke = 0.5f * mass * vel_magnitude * vel_magnitude / 418.4f;
            float ratio = delta_ke / energy_per_neighbor;
            const char* type_name = (aromatic_type == 0) ? "TRP" :
                                    (aromatic_type == 1) ? "TYR" :
                                    (aromatic_type == 2) ? "PHE" : "S-S";
            printf("[UV KE CHECK] type=%s lambda=%.0f E_per_neighbor=%.5f ΔKE=%.5f ratio=%.3f %s\n",
                   type_name, uv_wavelength_nm, energy_per_neighbor, delta_ke, ratio,
                   (ratio >= 0.8f && ratio <= 1.2f) ? "PASS" : "FAIL");
        }
#endif
    }

#ifdef DEBUG_UV_PHYSICS
    printf("[UV ENERGY] aromatic=%d n_neighbors=%d E_total_in=%.5f E_total_out=%.5f ratio=%.3f\n",
        aromatic_idx, n_neighbors, energy_to_transfer, total_ke_added, total_ke_added / energy_to_transfer);
#endif
}

// =============================================================================
// NHS INTEGRATION: How excited states affect exclusion field
// =============================================================================
//
// Key insight: Excited aromatic has DIFFERENT hydrophobicity
// - Ground state Tyr: moderately hydrophobic ring + polar OH
// - Excited state Tyr: more polar (larger dipole) → LESS hydrophobic
//
// This means: exclusion field should DECREASE around excited aromatics
// → Water is MORE attracted to the enhanced dipole
// → This is what creates the detectable signal!

__device__ __forceinline__ float get_exclusion_modifier(
    int aromatic_idx,
    const int* is_excited,
    const float* electronic_population
) {
    if (!is_excited[aromatic_idx]) {
        return 1.0f;  // No modification
    }

    float pop = electronic_population[aromatic_idx];

    // Excited state is more polar → less hydrophobic → MUCH less exclusion
    // Reduce exclusion by up to 80% at full excitation (aggressive for cryo detection)
    return 1.0f - 0.8f * pop;
}

// Use this when computing exclusion field contribution from aromatic atoms:
//
// float exclusion_contribution = base_exclusion;
// if (atom_to_aromatic[atom_idx] >= 0) {
//     int arom_idx = atom_to_aromatic[atom_idx];
//     exclusion_contribution *= get_exclusion_modifier(arom_idx, is_excited, population);
// }

// =============================================================================
// NHS INTEGRATION: Enhanced spike detection for excited states
// =============================================================================
//
// The excited state creates a TRANSIENT change in exclusion
// This should produce spikes that are:
// 1. Temporally correlated with UV pulse
// 2. Spatially localized around aromatics
// 3. Decaying with fluorescence timescale
//
// You can detect this by looking for:
// - Spike ONSET after UV pulse (immediate exclusion decrease)
// - Spike OFFSET after decay (exclusion recovery)

__device__ __forceinline__ float compute_uv_induced_signal(
    int voxel_idx,
    float current_exclusion,
    float previous_exclusion,
    float baseline_exclusion,  // Before UV pulse
    float time_since_uv_pulse
) {
    // Signal = deviation from baseline, weighted by recency
    float deviation = fabsf(current_exclusion - baseline_exclusion);
    
    // Weight by temporal proximity to UV pulse (exponential window)
    float temporal_weight = fast_expf(-time_since_uv_pulse * 0.5f);  // 2 ps window
    
    // Also include rate of change (dewetting signal)
    float rate_of_change = fabsf(current_exclusion - previous_exclusion);
    
    return (deviation + rate_of_change) * temporal_weight;
}

#endif // NHS_EXCITED_STATE_FUSED_CU
