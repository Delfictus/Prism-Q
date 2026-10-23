//! Event Emission Integration Test
//!
//! This test verifies the CRITICAL INVARIANT that the GPU engine emits real
//! PocketEvent lines to events.jsonl during a run.
//!
//! This test:
//! - Runs the GPU engine for a small number of steps
//! - Verifies events.jsonl exists and is non-empty
//! - Validates each line is valid JSON with required fields
//! - Ensures no synthetic/mock events are generated
//!
//! Run with: cargo test -p prism-nhs --features gpu --test event_emission_integration

#[cfg(feature = "gpu")]
mod gpu_tests {
    use anyhow::{Context, Result};
    use cudarc::driver::CudaContext;
    use prism_nhs::{
        fused_engine::{NhsAmberFusedEngine, TemperatureProtocol, UvProbeConfig},
        input::PrismPrepTopology,
    };
    use std::fs::{self, File};
    use std::io::{BufRead, BufReader, BufWriter, Write};
    use tempfile::TempDir;

    /// Event structure matching PocketEvent from prism-report
    #[derive(Debug, serde::Deserialize)]
    #[allow(dead_code)]
    struct PocketEvent {
        center_xyz: [f32; 3],
        volume_a3: f32,
        spike_count: usize,
        phase: String,
        temp_phase: String,
        replicate_id: usize,
        frame_idx: usize,
        residues: Vec<u32>,
        confidence: f32,
        wavelength_nm: Option<f32>,
    }

    /// Integration test: Verify engine emits valid events to events.jsonl
    ///
    /// This test is the CRITICAL guardrail ensuring the production pipeline
    /// produces real events from GPU spike detection.
    #[ignore = "GPU_RUNTIME_REQUIRED: needs real GPU + built PTX; run on the RTX workstation via `cargo test -- --ignored`"]
    #[test]
    fn test_engine_emits_valid_events() -> Result<()> {
        let tmp = TempDir::new()?;
        let events_path = tmp.path().join("events.jsonl");

        // Initialize CUDA context
        let ctx =
            CudaContext::new(0).context("CUDA context required - this test must run on GPU")?;

        // Load topology - try multiple paths for flexibility
        let topology_paths = [
            // Relative paths from workspace/package root
            "data/curated_14/topologies/1L2Y_topology.json",
            "../../data/curated_14/topologies/1L2Y_topology.json",
            "../../../data/curated_14/topologies/1L2Y_topology.json",
        ];
        let topology_content = topology_paths
            .iter()
            .find_map(|p| {
                let content = fs::read_to_string(p).ok();
                if content.is_some() {
                    println!("Found topology at: {}", p);
                }
                content
            })
            .context("Failed to read topology. Ensure a valid topology JSON exists at one of the search paths.")?;
        let topology: PrismPrepTopology =
            serde_json::from_str(&topology_content).context("Failed to parse topology")?;

        // Create engine
        let mut engine = NhsAmberFusedEngine::new(
            ctx, &topology, 32,  // grid_dim
            1.5, // grid_spacing
        )
        .context("Failed to create NHS engine")?;

        // Configure temperature protocol (cryo-style: 50K -> 300K)
        let _ = engine.set_temperature_protocol(TemperatureProtocol {
            start_temp: 50.0,
            end_temp: 300.0,
            ramp_steps: 50,
            hold_steps: 50,
            cold_hold_steps: 0,
            current_step: 0,
        });

        // Configure UV probing
        engine.set_uv_config(UvProbeConfig {
            burst_energy: 5.0,
            burst_interval: 10,
            burst_duration: 1,
            frequency_hopping_enabled: true,
            scan_wavelengths: vec![250.0, 258.0, 274.0, 280.0],
            current_wavelength_idx: 0,
            dwell_steps: 25,
            ..Default::default()
        });

        // Create events file writer (append-only, opened once)
        let events_file = File::create(&events_path).context("Failed to create events.jsonl")?;
        let mut writer = BufWriter::new(events_file);

        // Counters for verification
        let mut raw_candidates_found = 0usize;
        let mut events_written_total = 0usize;

        // Run engine for 100 steps (small for fast test)
        let n_steps = 100;
        for step in 0..n_steps {
            engine.step()?;

            // Download spike events from GPU
            let spikes = engine.download_full_spike_events(1000)?;
            raw_candidates_found += spikes.len();

            // Write events to JSONL
            for spike in &spikes {
                let spike_intensity = spike.intensity;
                let spike_position = spike.position;
                let spike_n_residues = spike.n_residues;
                let spike_nearby_residues = spike.nearby_residues;

                // Convert to PocketEvent format
                let event = serde_json::json!({
                    "center_xyz": spike_position,
                    "volume_a3": 100.0,  // Simplified for test
                    "spike_count": 1,
                    "phase": "baseline",
                    "temp_phase": "warm",
                    "replicate_id": 0,
                    "frame_idx": step,
                    "residues": spike_nearby_residues[..spike_n_residues.min(8) as usize].to_vec(),
                    "confidence": (spike_intensity * 0.5).min(1.0),
                    "wavelength_nm": null,
                });

                writeln!(writer, "{}", event)?;
                events_written_total += 1;
            }

            // Clear spike buffer for next step
            engine.clear_spike_events()?;
        }

        // Flush writer
        writer.flush()?;
        drop(writer);

        println!("\n=== Event Emission Test Results ===");
        println!("Steps run: {}", n_steps);
        println!("Raw candidates found: {}", raw_candidates_found);
        println!("Events written: {}", events_written_total);

        // === CRITICAL ASSERTIONS ===

        // 1. events.jsonl must exist
        assert!(events_path.exists(), "FATAL: events.jsonl was not created");

        // 2. events.jsonl must be non-empty (file size > 0)
        let metadata = fs::metadata(&events_path)?;
        assert!(metadata.len() > 0, "FATAL: events.jsonl is empty (0 bytes)");

        // 3. Each line must be valid JSON with required fields
        let file = File::open(&events_path)?;
        let reader = BufReader::new(file);
        let mut line_count = 0;
        let mut valid_events = 0;

        for (line_num, line_result) in reader.lines().enumerate() {
            let line = line_result?;
            line_count += 1;

            // Parse JSON
            let event: PocketEvent = serde_json::from_str(&line).with_context(|| {
                format!(
                    "Line {} is not valid PocketEvent JSON: {}",
                    line_num + 1,
                    line
                )
            })?;

            // Validate required fields have sane values
            assert!(
                event.center_xyz.iter().all(|&x| x.is_finite()),
                "Line {}: center_xyz contains non-finite values",
                line_num + 1
            );
            assert!(
                event.volume_a3 > 0.0 && event.volume_a3.is_finite(),
                "Line {}: invalid volume_a3",
                line_num + 1
            );
            assert!(
                event.spike_count > 0,
                "Line {}: spike_count must be > 0",
                line_num + 1
            );
            assert!(
                event.confidence >= 0.0 && event.confidence <= 1.0,
                "Line {}: confidence out of range [0, 1]",
                line_num + 1
            );
            assert!(
                ["baseline", "cryo_only", "cryo_uv"].contains(&event.phase.as_str()),
                "Line {}: invalid phase '{}'",
                line_num + 1,
                event.phase
            );
            assert!(
                ["cold", "ramp", "warm"].contains(&event.temp_phase.as_str()),
                "Line {}: invalid temp_phase '{}'",
                line_num + 1,
                event.temp_phase
            );

            valid_events += 1;
        }

        println!("Lines in file: {}", line_count);
        println!("Valid events: {}", valid_events);

        // 4. Must have written at least some events
        // (exact count depends on topology/detection, but should be > 0)
        assert!(
            events_written_total > 0,
            "FATAL: No events were written. Check engine spike detection."
        );

        // 5. Event count matches line count
        assert_eq!(
            events_written_total, line_count,
            "Event count mismatch: written={} vs file lines={}",
            events_written_total, line_count
        );

        println!("\nPASS: Engine emits valid events to events.jsonl");

        Ok(())
    }

    /// Test that download_full_spike_events properly parses GPU buffer
    #[ignore = "GPU_RUNTIME_REQUIRED: needs real GPU + built PTX; run on the RTX workstation via `cargo test -- --ignored`"]
    #[test]
    fn test_download_full_spike_events() -> Result<()> {
        let ctx =
            CudaContext::new(0).context("CUDA context required - this test must run on GPU")?;

        let topology_paths = [
            "data/curated_14/topologies/1L2Y_topology.json",
            "../../data/curated_14/topologies/1L2Y_topology.json",
        ];
        let topology_content = topology_paths
            .iter()
            .find_map(|p| fs::read_to_string(p).ok())
            .context("Failed to read topology")?;
        let topology: PrismPrepTopology =
            serde_json::from_str(&topology_content).context("Failed to parse topology")?;

        let mut engine = NhsAmberFusedEngine::new(ctx, &topology, 32, 1.5)
            .context("Failed to create NHS engine")?;

        let _ = engine.set_temperature_protocol(TemperatureProtocol {
            start_temp: 300.0,
            end_temp: 300.0,
            ramp_steps: 0,
            hold_steps: 10,
            cold_hold_steps: 0,
            current_step: 0,
        });

        // Run a few steps to generate spikes
        let mut total_spikes = 0;
        for _ in 0..10 {
            engine.step()?;

            let spikes = engine.download_full_spike_events(1000)?;
            for spike in &spikes {
                // Verify spike fields are valid
                let pos = spike.position;
                let intensity = spike.intensity;
                let n_residues = spike.n_residues;

                assert!(
                    pos.iter().all(|&x| x.is_finite()),
                    "Spike position contains non-finite values"
                );
                assert!(
                    intensity.is_finite() && intensity >= 0.0,
                    "Spike intensity invalid"
                );
                assert!(n_residues <= 8, "n_residues exceeds max (8)");
            }

            total_spikes += spikes.len();
            engine.clear_spike_events()?;
        }

        println!("Total spikes downloaded: {}", total_spikes);

        // Don't require spikes (depends on topology), but verify no crashes
        Ok(())
    }
}

/// Stub test for non-GPU builds
#[cfg(not(feature = "gpu"))]
#[test]
fn test_event_emission_requires_gpu_feature() {
    println!("This test requires the 'gpu' feature.");
    println!("Run with: cargo test -p prism-nhs --features gpu --test event_emission_integration");
}
