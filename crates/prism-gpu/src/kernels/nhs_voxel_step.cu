// ==========================================================================
// nhs_voxel_step.cu — Voxel-parallel kernel (split from nhs_amber_fused.cu)
// Launch with: ceil(total_voxels / 256) blocks × 256 threads
// Fuses Phase 4 (exclusion) + Phase 5 (LIF/UV) + EFP into ONE voxel pass
// ==========================================================================

#include "nhs_amber_fused.cu"  // Get all structs, device functions, constants

extern "C" __global__ void nhs_voxel_step(
    // Positions (read-only, updated by atom kernel)
    const float3* __restrict__ positions,
    int n_atoms,
    // Voxel grid params
    int grid_dim,
    float grid_spacing,
    float grid_origin_x,
    float grid_origin_y,
    float grid_origin_z,
    // Exclusion field arrays
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    // Warp matrix
    const WarpEntry* warp_matrix,
    // Atom metadata
    const int* __restrict__ atom_types,
    const float* __restrict__ charges,
    const int* __restrict__ residue_ids,
    // Aromatic data
    const float3* __restrict__ d_aromatic_centroids,
    const float3* __restrict__ d_ring_normals,
    const int* __restrict__ d_is_excited,
    const float* __restrict__ d_electronic_population,
    const float* __restrict__ d_vibrational_energy,
    const float* __restrict__ d_time_since_excitation,
    const int* __restrict__ d_aromatic_type,
    const int* __restrict__ d_atom_to_aromatic,
    int n_aromatics,
    // UV params
    float uv_wavelength_nm,
    int uv_burst_active,
    float* d_uv_signal_prev,
    // Spike output
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,
    // Temperature
    float target_temp,
    float dt,
    int timestep,
    // EFP arrays
    float* efp_potential,
    float* efp_potential_prev,
    float* efp_lif_potential,
    // Aromatic neighbors (for expanded exclusion)
    const AromaticNeighbors* __restrict__ d_aromatic_neighbors,
    const float* __restrict__ d_franck_condon_progress,
    int* spike_grid_efp,             // independent EFP refractory grid
    // Signal preservation buffers (accumulated across all timesteps)
    unsigned int* voxel_hit_grid,          // [grid_dim³] spatial recurrence counter
    int* last_uv_step,                     // [grid_dim³] timestep of last UV event per voxel
    unsigned int* coupled_spike_grid,      // [grid_dim³] UV→LIF causal spike counter
    int* primary_residue_id,               // [grid_dim³] dominant driver residue ID (-1 = none)
    unsigned int* primary_residue_count,   // [grid_dim³] count for dominant driver
    int* residue_step_causal               // [n_residues] KCC per-step causal counter
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_voxels = grid_dim * grid_dim * grid_dim;
    if (tid >= total_voxels) return;

    int v = tid;  // One thread per voxel — no grid-stride loop needed

    float3 grid_origin = make_float3(grid_origin_x, grid_origin_y, grid_origin_z);

    // Voxel coordinates (computed ONCE, shared across all phases)
    int vz = v / (grid_dim * grid_dim);
    int vy = (v / grid_dim) % grid_dim;
    int vx = v % grid_dim;
    float3 voxel_center = make_float3(
        grid_origin.x + (vx + 0.5f) * grid_spacing,
        grid_origin.y + (vy + 0.5f) * grid_spacing,
        grid_origin.z + (vz + 0.5f) * grid_spacing
    );

    // ====================================================================
    // FUSED PHASE 4: EXCLUSION FIELD + WATER DENSITY
    // ====================================================================
    water_density_prev[v] = water_density[v];

    WarpEntry entry = warp_matrix[v];
    float total_exclusion = 0.0f;
    float polar_field = 0.0f;

    for (int i = 0; i < entry.n_atoms; i++) {
        int a = entry.atom_indices[i];
        if (a < 0 || a >= n_atoms) continue;

        float contrib = compute_exclusion_contribution(
            positions[a], voxel_center,
            atom_types[a], charges[a]
        );

        if (n_aromatics > 0 && d_aromatic_centroids != nullptr) {
            float expanded_modifier = compute_expanded_exclusion_modifier(
                positions[a],
                d_aromatic_centroids,
                d_ring_normals,
                d_is_excited,
                d_electronic_population,
                n_aromatics
            );
            contrib *= expanded_modifier;
        } else {
            int aromatic_idx = d_atom_to_aromatic[a];
            if (aromatic_idx >= 0 && aromatic_idx < n_aromatics) {
                float excitation_modifier = get_exclusion_modifier(
                    aromatic_idx,
                    d_is_excited,
                    d_electronic_population
                );
                contrib *= excitation_modifier;
            }
        }

        total_exclusion += contrib * entry.atom_weights[i] * 4.0f;

        if (atom_types[a] == 1 || atom_types[a] == 2 || atom_types[a] == 3) {
            polar_field += contrib * 0.5f;
        }
    }

    total_exclusion = fminf(1.0f, total_exclusion);
    exclusion_field[v] = total_exclusion;
    water_density[v] = infer_water_density(total_exclusion, polar_field, target_temp);

    // ====================================================================
    // FUSED PHASE 5: NEUROMORPHIC LIF + UV-LIF COUPLING
    // ====================================================================

    // Refractory check — if counting down, decrement and skip LIF+EFP
    if (spike_grid[v] > 0) {
        spike_grid[v]--;
        // Still do EFP below (it has its own refractory check)
        goto efp_phase;
    }

    {
        float spike_intensity = 0.0f;
        float tau_mem = 0.1f;

        float uv_signal = 0.0f;
        int n_nearby_excited = 0;
        float min_distance_to_excited = 1000.0f;
        int closest_excited_idx = -1;

        if (n_aromatics > 0) {
            const float UV_DETECTION_RADIUS = 4.0f;
            const float UV_DIRECT_STRENGTH = 0.8f;

            float total_vib_energy = 0.0f;

            for (int a = 0; a < n_aromatics; a++) {
                if (!d_is_excited[a]) continue;
                if (d_aromatic_centroids == nullptr) continue;

                float3 arom_pos = d_aromatic_centroids[a];
                float dx = voxel_center.x - arom_pos.x;
                float dy = voxel_center.y - arom_pos.y;
                float dz = voxel_center.z - arom_pos.z;
                float dist = sqrtf(dx*dx + dy*dy + dz*dz);

                if (dist < UV_DETECTION_RADIUS) {
                    n_nearby_excited++;
                    total_vib_energy += d_vibrational_energy[a];
                    if (dist < min_distance_to_excited) {
                        min_distance_to_excited = dist;
                        closest_excited_idx = a;
                    }
                }
            }

            if (n_nearby_excited > 0) {
                float energy_factor = fminf(total_vib_energy / (n_nearby_excited * 3.0f), 1.0f);
                float coop_boost = 1.0f + 0.3f * (n_nearby_excited - 1);
                uv_signal = UV_DIRECT_STRENGTH * energy_factor * coop_boost;
                if (uv_burst_active) {
                    uv_signal *= 2.0f;
                }
            }

            if (d_aromatic_centroids != nullptr && d_uv_signal_prev != nullptr) {
                float prev_signal = d_uv_signal_prev[v];
                float advanced_signal = compute_uv_lif_signal(
                    voxel_center,
                    d_aromatic_centroids,
                    d_ring_normals,
                    d_is_excited,
                    d_electronic_population,
                    d_vibrational_energy,
                    d_time_since_excitation,
                    n_aromatics,
                    dt,
                    prev_signal
                );
                uv_signal += advanced_signal;
                d_uv_signal_prev[v] = uv_signal;
            }

            if (uv_signal > 0.0f) {
                lif_potential[v] += uv_signal;
            }

            // Record UV event timestamp for causal tracking
            if (uv_signal > 0.1f) {
                last_uv_step[v] = timestep;
            }

            // Direct UV spike trigger
            const float DIRECT_UV_SPIKE_THRESHOLD = 0.3f;
            const float MAX_SPIKE_DISTANCE = 4.0f;

            const int voxel_n_atoms = warp_matrix[v].n_atoms;
            bool voxel_has_aromatic_atom = false;
            for (int wi = 0; wi < voxel_n_atoms && !voxel_has_aromatic_atom; wi++) {
                int atom_idx = warp_matrix[v].atom_indices[wi];
                if (atom_idx >= 0 && atom_idx < n_atoms) {
                    if (d_atom_to_aromatic[atom_idx] >= 0) {
                        voxel_has_aromatic_atom = true;
                    }
                }
            }

            if (n_nearby_excited > 0 &&
                uv_signal > DIRECT_UV_SPIKE_THRESHOLD &&
                voxel_n_atoms > 0 &&
                min_distance_to_excited < MAX_SPIKE_DISTANCE) {
                spike_grid[v] = REFRACTORY_STEPS;
                spike_intensity = uv_signal;

                // Signal preservation (legacy UV spike)
                update_signal_preservation(v, timestep, 1,
                    voxel_hit_grid, last_uv_step, coupled_spike_grid,
                    primary_residue_id, primary_residue_count,
                    warp_matrix[v], residue_ids, n_atoms, residue_step_causal);

                int _arom_type = (closest_excited_idx >= 0) ? d_aromatic_type[closest_excited_idx] : -1;
                int _arom_res = -1;
                if (closest_excited_idx >= 0) {
                    for (int wi = 0; wi < warp_matrix[v].n_atoms && _arom_res < 0; wi++) {
                        int ai = warp_matrix[v].atom_indices[wi];
                        if (ai >= 0 && ai < n_atoms && d_atom_to_aromatic[ai] == closest_excited_idx) {
                            _arom_res = residue_ids[ai];
                        }
                    }
                }
                float _vib_e = (closest_excited_idx >= 0) ? d_vibrational_energy[closest_excited_idx] : 0.0f;

                int spike_idx = atomicAdd(spike_count, 1);
                if (spike_idx < max_spikes) {
                    capture_spike_event(
                        spike_events[spike_idx], timestep, v, voxel_center,
                        spike_intensity, warp_matrix[v], residue_ids,
                        1, uv_wavelength_nm, _arom_type, _arom_res,
                        water_density[v], _vib_e, n_nearby_excited,
                        fabsf(water_density[v] - water_density_prev[v])
                    );
                }
                lif_potential[v] = LIF_RESET;
            }
        }

        // Standard LIF update (non-UV)
        if (spike_grid[v] == 0 && !uv_burst_active) {
            bool spike = lif_neuron_update(
                lif_potential[v], water_density[v], water_density_prev[v],
                tau_mem, dt, LIF_THRESHOLD, spike_intensity
            );

            if (spike) {
                spike_grid[v] = REFRACTORY_STEPS;

                // Signal preservation (legacy LIF spike)
                int lif_src = (n_nearby_excited > 0) ? 1 : 2;
                update_signal_preservation(v, timestep, lif_src,
                    voxel_hit_grid, last_uv_step, coupled_spike_grid,
                    primary_residue_id, primary_residue_count,
                    warp_matrix[v], residue_ids, n_atoms, residue_step_causal);

                int spike_idx = atomicAdd(spike_count, 1);
                if (spike_idx < max_spikes) {
                    int lif_atype = -1, lif_ares = -1;
                    float lif_wl = 0.0f, lif_vibe = 0.0f;
                    if (n_nearby_excited > 0 && closest_excited_idx >= 0) {
                        lif_atype = d_aromatic_type[closest_excited_idx];
                        lif_wl = uv_wavelength_nm;
                        lif_vibe = d_vibrational_energy[closest_excited_idx];
                        for (int wi = 0; wi < warp_matrix[v].n_atoms && lif_ares < 0; wi++) {
                            int ai = warp_matrix[v].atom_indices[wi];
                            if (ai >= 0 && ai < n_atoms && d_atom_to_aromatic[ai] == closest_excited_idx) {
                                lif_ares = residue_ids[ai];
                            }
                        }
                    }
                    capture_spike_event(
                        spike_events[spike_idx], timestep, v, voxel_center,
                        spike_intensity, warp_matrix[v], residue_ids,
                        lif_src, lif_wl,
                        lif_atype, lif_ares, water_density[v],
                        lif_vibe, n_nearby_excited,
                        fabsf(water_density[v] - water_density_prev[v])
                    );
                }
            }
        }
    }

    // ====================================================================
    // FUSED EFP: ELECTROSTATIC FLUX PROBE
    // ====================================================================
efp_phase:

    if (warp_matrix[v].n_atoms > 0) {
        float phi = 0.0f;
        int n_charged_nearby = 0;

        for (int wi = 0; wi < warp_matrix[v].n_atoms; wi++) {
            int ai = warp_matrix[v].atom_indices[wi];
            if (ai < 0 || ai >= n_atoms) continue;
            float q = charges[ai];
            if (fabsf(q) < 0.15f) continue;

            float3 ap = positions[ai];
            float dx = voxel_center.x - ap.x;
            float dy = voxel_center.y - ap.y;
            float dz = voxel_center.z - ap.z;
            float dist = sqrtf(dx*dx + dy*dy + dz*dz);

            if (dist > 0.5f && dist < 8.0f) {
                float eps_r = fmaxf(4.0f * dist, 4.0f);
                phi += q / (eps_r * dist);
                n_charged_nearby++;
            }
        }

        float wd_change = fabsf(water_density[v] - water_density_prev[v]);
        float polar_water_signal = fabsf(phi) * wd_change * 40.0f;

        float phi_prev = efp_potential_prev[v];
        efp_potential[v] = phi;

        float flux = fabsf(phi - phi_prev);
        float polar_signal = flux * 150.0f + polar_water_signal;

        if (n_charged_nearby >= 1 && spike_grid_efp[v] == 0) {
            const float EFP_TAU = 0.5f;
            const float EFP_THRESHOLD = 0.15f;
            float efp_decay = expf(-dt / EFP_TAU);
            efp_lif_potential[v] = efp_decay * efp_lif_potential[v] + polar_signal;

            if (efp_lif_potential[v] > EFP_THRESHOLD) {
                spike_grid_efp[v] = REFRACTORY_STEPS;
                float polar_intensity = efp_lif_potential[v];
                efp_lif_potential[v] = LIF_RESET;

                // Signal preservation (legacy EFP spike)
                update_signal_preservation(v, timestep, 3,
                    voxel_hit_grid, last_uv_step, coupled_spike_grid,
                    primary_residue_id, primary_residue_count,
                    warp_matrix[v], residue_ids, n_atoms, residue_step_causal);

                int si = atomicAdd(spike_count, 1);
                if (si < max_spikes) {
                    int polar_type = (phi > 0.0f) ? 5 : 6;
                    capture_spike_event(
                        spike_events[si], timestep, v, voxel_center,
                        polar_intensity, warp_matrix[v], residue_ids,
                        3, 0.0f, polar_type, -1,
                        water_density[v], flux, n_charged_nearby,
                        wd_change
                    );
                }
            }
        }
        efp_potential_prev[v] = phi;
    }
}
