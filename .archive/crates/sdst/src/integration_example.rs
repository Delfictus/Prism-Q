/**
 * NHS → SDST Integration Example
 *
 * Shows the insertion point in the nhs_rt_full simulation loop
 * where spike events feed into SDST, and how to run analysis
 * after simulation completes.
 *
 * This is pseudocode/reference - adapt to your actual nhs_rt_full.rs structure.
 */

// In nhs_rt_full.rs, after spike detection in each simulation step:

mod sdst_ffi; // The Rust FFI module from sdst_ffi.rs

use sdst_ffi::*;

/// Integration point: call this after each NHS simulation step
/// where spikes have been detected.
fn feed_spikes_to_sdst(
    sdst: &Sdst,
    active_voxels: &[(u32, u32, u32)],   // Grid coordinates of firing voxels
    amplitudes: &[f32],                    // Spike amplitudes
    temperatures: &[f32],                  // Local temperatures
    energy_gradients: &[f32],              // |∇E| per voxel
    solvent_exposures: &[f32],             // SASA proxy
    timestep: u32,
    phase: u8,
) -> Result<(), SdstError> {
    let inputs: Vec<SpikeInput> = active_voxels.iter().enumerate().map(|(i, &(x, y, z))| {
        SpikeInput {
            voxel_x: x,
            voxel_y: y,
            voxel_z: z,
            timestamp: timestep,
            amplitude: amplitudes[i],
            local_temp: temperatures[i],
            energy_gradient: energy_gradients[i],
            solvent_exposure: solvent_exposures[i],
            phase_id: phase,
        }
    }).collect();

    sdst.insert_raw(&inputs)
}

/// Determine hysteresis phase from timestep
fn get_phase(timestep: u32, config: &SdstConfig) -> u8 {
    for p in (0..5).rev() {
        if timestep >= config.phase_boundaries[p] {
            return p as u8;
        }
    }
    0
}

/// Full simulation integration outline
fn run_prism_therm_with_sdst() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize SDST
    let mut config = unsafe { sdst_default_config() };
    // Customize if needed:
    // config.max_spike_events = 5_000_000;
    // config.phase_boundaries = [0, 14000, 20000, 35000, 41000, 55000];

    let sdst = Sdst::new(&config)?;
    println!("SDST initialized");
    sdst.print_stats()?;

    // 2. Run NHS simulation loop
    let total_steps = config.phase_boundaries[5];
    for step in 0..total_steps {
        let phase = get_phase(step, &config);

        // --- Your existing NHS simulation step ---
        // let (active_voxels, amplitudes, temps, grads, sasas) = nhs_step(step);

        // --- Feed spikes to SDST ---
        // feed_spikes_to_sdst(
        //     &sdst, &active_voxels, &amplitudes,
        //     &temps, &grads, &sasas, step, phase
        // )?;

        // Checkpoint every 10K steps
        if step > 0 && step % 10000 == 0 {
            let path = format!("checkpoints/sdst_step_{}.bin", step);
            sdst.save(&path)?;
            println!("Checkpoint saved: {}", path);
        }
    }

    // 3. Post-simulation analysis
    println!("\n=== PRISM-Therm Analysis ===");
    let count = sdst.event_count()?;
    let mem = sdst.memory_usage()?;
    println!("Total spike events: {}", count);
    println!("GPU memory used: {:.1} MB", mem as f64 / 1024.0 / 1024.0);

    // 3a. Hysteresis scan: find all cryptic site candidates
    // (Called via unsafe FFI - safe wrapper could be added)
    println!("\n--- Hysteresis Scan ---");
    // Results show regions with heating/cooling asymmetry
    // These are cryptic binding site candidates

    // 3b. CCNS for each candidate
    // For each hysteretic region, compute tau exponent
    // tau < 1.5 = SOC (most druggable)
    // tau 1.5-2.0 = NearCritical
    // tau >= 2.0 = Barrier

    // Example: analyze a specific region
    let test_region = SpatialRegion {
        x_min: 40, x_max: 55,
        y_min: 50, y_max: 65,
        z_min: 30, z_max: 45,
    };

    let ccns = sdst.ccns_region(&test_region)?;
    println!("Region CCNS: tau={:.3} ± {:.3}, class={:?}, druggability={:.3}",
             ccns.tau, ccns.tau_stderr, ccns.classification, ccns.druggability);

    let hyst = sdst.hysteresis_region(&test_region, 0.2)?;
    println!("Hysteresis: asymmetry={:.3}, heating_rate={:.4}, cooling_rate={:.4}, is_hysteretic={}",
             hyst.asymmetry_score, hyst.heating_spike_rate, hyst.cooling_spike_rate,
             hyst.is_hysteretic);

    // 3c. TIDE decomposition for Plan C
    // For each confirmed pocket, compute per-residue causal ΔG
    // This tells drug designers which residues to target

    // 3d. Causal subgraph extraction
    // For a specific spike event, trace the entire causal chain
    // This shows the mechanism of pocket opening

    // 4. Save final state
    sdst.save("results/sdst_final.bin")?;
    println!("\nFinal state saved.");

    Ok(())
}

// ============================================================
// Output format for drug designer consumption
// ============================================================

/// Generate a report from SDST analysis results
fn generate_pocket_report(
    region: &SpatialRegion,
    ccns: &CcnsResult,
    hyst: &HysteresisResult,
    // tide: &[TideDecomposition],
    // wavefronts: &[WavefrontStats],
    grid_spacing: f32,
) {
    let cx = (region.x_min + region.x_max) as f32 / 2.0 * grid_spacing;
    let cy = (region.y_min + region.y_max) as f32 / 2.0 * grid_spacing;
    let cz = (region.z_min + region.z_max) as f32 / 2.0 * grid_spacing;

    println!("╔══════════════════════════════════════════════╗");
    println!("║     PRISM-Therm Cryptic Site Report          ║");
    println!("╠══════════════════════════════════════════════╣");
    println!("║ Centroid:     ({:.1}, {:.1}, {:.1}) Å", cx, cy, cz);
    println!("║ CCNS τ:       {:.3} ± {:.3}", ccns.tau, ccns.tau_stderr);
    println!("║ Class:        {:?}", ccns.classification);
    println!("║ Druggability: {:.3}", ccns.druggability);
    println!("║ Hysteresis:   {:.3} ({})",
             hyst.asymmetry_score,
             if hyst.is_hysteretic { "CRYPTIC" } else { "static" });
    println!("║ Avalanches:   {}", ccns.n_avalanches);
    println!("╠══════════════════════════════════════════════╣");
    println!("║ Spike rates (spikes/step):                   ║");
    println!("║   Heating:  {:.4}", hyst.heating_spike_rate);
    println!("║   Cooling:  {:.4}", hyst.cooling_spike_rate);
    println!("╚══════════════════════════════════════════════╝");

    /*
    // TIDE decomposition (when available):
    println!("Top causal residues (by transfer entropy):");
    for td in tide.iter().take(10) {
        println!("  Residue {:4}: TE={:.4}  ΔG={:+.2}  FI={:.4}  KL={:.4}",
                 td.residue_id, td.transfer_entropy, td.causal_dg,
                 td.fisher_info, td.kl_divergence);
    }
    */
}
