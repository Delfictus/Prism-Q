//! Strict real-GPU regression for native pre-replay reference initialization.
//! This builds the captured pipeline but never launches its graph or MD.
#![cfg(all(feature = "gpu", feature = "v2_ignition"))]

use std::ffi::c_void;
use cudarc::driver::{CudaContext, DevicePtr};
use cudarc::driver::sys::{cuMemcpyDtoH_v2, CUdeviceptr, CUresult};
use prism_nhs::captured_pipeline::{CapturedAdjudicationPipeline, PipelineConfig, MarV2Config};
use prism_nhs::rich_spike::RichSpike;
use prism_nhs::so3_project::{ContactShellTile, SiteManifestFfi, So3ProjectInput, So3ProjectTransform};
use prism_nhs::transform::AuditedTransform;
use sha2::{Digest, Sha256};

#[test]
#[ignore = "GPU_RUNTIME_REQUIRED: explicit standalone process only; no MD/graph replay"]
fn native_relaxed_reference_is_projected_and_frozen_before_any_replay() {
    let ctx = CudaContext::new(0).expect("real CUDA is required; absence is not a passing test");
    let stream = ctx.new_stream().expect("dedicated regression stream");
    let raw_stream = stream.cu_stream() as *mut c_void;
    prism_nhs::sh_basis::init_device_basis(raw_stream).expect("native basis initialization");
    stream.synchronize().expect("basis initialized");
    let basis = prism_nhs::sh_basis::k_lm_device_ptr().expect("native basis pointer");
    const ATOMS: usize = 16;
    const CLUSTERS: u32 = 2;
    const FRAME: u32 = 17;
    let spikes: Vec<RichSpike> = (0..ATOMS).map(|i| {
        let mut s = RichSpike::zero();
        let theta = 0.3 + i as f32 * 0.2;
        let phi = 0.4 + i as f32 * 0.3;
        s.x = 4.0 * theta.sin() * phi.cos();
        s.y = 4.0 * theta.sin() * phi.sin();
        s.z = 4.0 * theta.cos();
        s.cluster_id = (i / 8) as i32;
        s
    }).collect();
    let spike_bytes = unsafe { std::slice::from_raw_parts(
        spikes.as_ptr() as *const u8, spikes.len() * std::mem::size_of::<RichSpike>()) }.to_vec();
    let offsets = vec![0u32, 8, ATOMS as u32];
    let mut d_spikes = stream.alloc_zeros::<u8>(spike_bytes.len()).expect("spikes");
    let mut d_offsets = stream.alloc_zeros::<u32>(offsets.len()).expect("CSR");
    stream.memcpy_htod(&spike_bytes, &mut d_spikes).expect("upload spikes");
    stream.memcpy_htod(&offsets, &mut d_offsets).expect("upload CSR");
    // All mandatory capture inputs are allocated and kept alive until after
    // explicit pipeline drop. No null mandatory argument or zero-count escape.
    let d_forces = stream.alloc_zeros::<f32>(ATOMS * 3).expect("forces");
    let mut d_masses = stream.alloc_zeros::<f32>(ATOMS).expect("masses");
    let d_velocities = stream.alloc_zeros::<f32>(ATOMS * 3).expect("velocities");
    let mut d_dt = stream.alloc_zeros::<f32>(1).expect("dt");
    let d_pe = stream.alloc_zeros::<f64>(ATOMS).expect("per-atom PE");
    let d_work = stream.alloc_zeros::<f64>(1).expect("external work");
    let mut d_clock = stream.clone_htod(&[17u32]).expect("owned fixture clock");
    let d_ghost_clock = stream.clone_htod(&[999u64]).expect("owned Ghost counter");
    let ghost_ring = prism_nhs::ghost_tile::GhostTileRing::allocate(CLUSTERS).expect("real mapped Ghost ring");
    stream.memcpy_htod(&vec![12.0f32; ATOMS], &mut d_masses).expect("upload masses");
    stream.memcpy_htod(&[0.002f32], &mut d_dt).expect("upload dt");
    macro_rules! address { ($buffer:expr) => {{ let (p, _guard) = $buffer.device_ptr(&stream); p }}; }
    let cfg = PipelineConfig {
            seed_layout: prism_nhs::rich_spike::NativeV2SeedLayout::default(),
        d_spikes: address!(d_spikes) as *const RichSpike,
        d_cluster_offsets: address!(d_offsets) as *const u32,
        n_clusters: CLUSTERS,
        d_k_lm: basis,
        initial_frame_id: FRAME,
        source_stream_id: 0,
        diagnostic_stream_id: Some(0),
        asc: None,
        d_force_norm_mirror: 0,
        governor_device: None,
        zstr: None,
        sisr: None,
        noise_floor_override: None,
        mar_v2: Some(MarV2Config { d_step_counter: address!(d_ghost_clock), ..MarV2Config::default() }),
        d_dt: address!(d_dt) as *mut f32,
        d_velocities: address!(d_velocities) as *mut f32,
        g26_parent_cond_handle: 0,
        f1_parent_cond_handle: 0,
        d_pe_components: address!(d_pe) as *const f64,
        n_atoms_for_pe: ATOMS as u32,
        d_external_work: address!(d_work) as *mut f64,
        ghost_tile_ring_dev: ghost_ring.device_base,
        ghost_tile_max_records: CLUSTERS,
        firehose_enable: 0,
        body_unroll: 1,
        d_kcc_lead: 0,
        d_forces_anchor: address!(d_forces),
        d_masses: address!(d_masses),
        force_burst_step: None,
        n_spikes: ATOMS as u32,
        d_current_md_step: address!(d_clock) as *const u32,
        branch_trace_dev: 0,
    };
    stream.synchronize().expect("all actual input buffers uploaded");
    let pipeline = CapturedAdjudicationPipeline::build(&ctx, &stream, &cfg)
        .expect("strict native captured build must not skip failures");
    stream.memcpy_htod(&[23u32], &mut d_clock).unwrap();
    pipeline.bind_exact_ghost_step(23).expect("validation orders pending source update");
    stream.memcpy_htod(&[17u32], &mut d_clock).unwrap();
    pipeline.bind_exact_ghost_step(17).expect("compatibility API verifies actual physical endpoint");
    assert!(pipeline.bind_exact_ghost_step(18).is_err(), "host must not override physical time");
    assert!(pipeline.bind_exact_ghost_step(0).is_err());
    assert!(pipeline.bind_exact_ghost_step(u32::MAX as u64 + 1).is_err());
    assert_eq!(stream.clone_dtoh(&d_ghost_clock).unwrap(), vec![999u64], "no counter writes before replay");
    assert_eq!(ghost_ring.n_frames_written(), 0);

    let d_expected = stream.alloc_zeros::<u8>(pipeline.tiles_bytes as usize).expect("independent expected tiles");
    let mut expected_manifest = SiteManifestFfi {
        total_sites: CLUSTERS, _pad0: 0,
        tiles_dev_ptr: address!(d_expected) as *mut ContactShellTile,
        vram_high_water_mark: pipeline.tiles_bytes,
        adjudication_trigger_ptr: std::ptr::null_mut(),
    };
    let (projected, violations) = So3ProjectTransform::new().apply(So3ProjectInput {
        pool_handle: 0, // unused by the existing transform; fixture owns CUDA allocation
        stream_handle: stream.cu_stream() as usize,
        d_spikes: cfg.d_spikes, d_cluster_offsets: cfg.d_cluster_offsets,
        n_clusters: CLUSTERS, d_k_lm: basis, frame_id: FRAME, source_stream_id: 0,
        manifest: &mut expected_manifest,
    }).into_result().expect("independent existing-native-projector invocation");
    assert!(violations.is_empty());
    assert_eq!(projected.cuda_error, 0);
    stream.synchronize().expect("independent projection completed");
    let mut expected = vec![0u8; pipeline.tiles_bytes as usize];
    stream.memcpy_dtoh(&d_expected, &mut expected).expect("read expected native bytes");
    let mut spectra = Vec::new();
    for tile in expected.chunks_exact(std::mem::size_of::<ContactShellTile>()) {
        let read_f32s = |offset: usize, count: usize| -> Vec<f32> {
            tile[offset..offset + count * 4].chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
        };
        let coefficients = read_f32s(std::mem::offset_of!(ContactShellTile, geo_alm), 64);
        let power = read_f32s(std::mem::offset_of!(ContactShellTile, geo_power_spectrum), 8);
        assert!(coefficients.iter().all(|v| v.is_finite()));
        assert!(coefficients.iter().any(|v| *v != 0.0), "geometry must contain computed coefficients");
        assert!(power.iter().all(|v| v.is_finite()));
        assert!(power[..6].iter().any(|v| *v > 1e-7), "geometry must exceed native empty-cluster KL_EPS_FP floor");
        spectra.push(serde_json::json!({
            "geo_power_spectrum": power,
            "geo_alm_squared_norm": coefficients.iter().map(|v| (*v as f64).powi(2)).sum::<f64>(),
        }));
    }
    let expected_sha = format!("{:x}", Sha256::digest(&expected));
    let before = pipeline.frozen_baseline_readback().expect("before-freeze device receipt");
    assert_eq!(before["relaxed_sha256"], expected_sha);
    assert_ne!(before["relaxed_sha256"], before["frozen_sha256"], "zero storage is not a native reference");
    assert_eq!(before["adjudicator_relaxed_binding_is_frozen"], false);
    for (cluster, row) in before["relaxed_headers"].as_array().unwrap().iter().enumerate() {
        assert_eq!(row["spike_count"], 8);
        assert_eq!(row["captured_frame_id"], FRAME);
        assert_eq!(row["stream_id"], 0);
        assert_eq!(row["cluster_id_bits"], cluster);
    }
    pipeline.freeze_baseline().expect("freeze without any graph launch");
    let after = pipeline.frozen_baseline_readback().expect("after-freeze device receipt");
    assert_eq!(after["relaxed_sha256"], expected_sha);
    assert_eq!(after["frozen_sha256"], expected_sha);
    assert_eq!(after["byte_identical"], true);
    assert_eq!(after["adjudicator_relaxed_binding_is_frozen"], true);
    // Actual input and integrator buffers remain bit-identical. This test
    // does not infer non-advancement merely from a graph-launch counter.
    let mut current_spikes = vec![0u8; spike_bytes.len()];
    stream.memcpy_dtoh(&d_spikes, &mut current_spikes).expect("spike input readback");
    assert_eq!(current_spikes, spike_bytes);
    for (buffer, expected) in [(&d_forces, vec![0.0f32; ATOMS * 3]),
                              (&d_masses, vec![12.0f32; ATOMS]),
                              (&d_velocities, vec![0.0f32; ATOMS * 3]),
                              (&d_dt, vec![0.002f32])] {
        let mut actual = vec![0.0f32; expected.len()];
        stream.memcpy_dtoh(buffer, &mut actual).expect("integrator buffer readback");
        assert_eq!(actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                   expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    }
    for (buffer, count) in [(&d_work, 1), (&d_pe, ATOMS)] {
        let mut actual = vec![0.0f64; count];
        stream.memcpy_dtoh(buffer, &mut actual).expect("energy/work readback");
        assert!(actual.iter().all(|v| v.to_bits() == 0));
    }
    let csr_bytes: Vec<u8> = offsets.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut basis_bytes = vec![0u8; 36 * 4];
    let rc = unsafe { cuMemcpyDtoH_v2(basis_bytes.as_mut_ptr() as *mut c_void, basis as CUdeviceptr, basis_bytes.len()) };
    assert!(matches!(rc, CUresult::CUDA_SUCCESS));
    println!("{}", serde_json::json!({
        "schema_kind": "prism_native_baseline_regression_receipt.v1",
        "scope": "actual GPU projection/freeze lifecycle on explicit fixture bytes; no MD trajectory or mechanistic inference",
        "graph_replays": 0, "MD_launched": false, "native_ghost_clock_bridge_captured":true,
        "host_physical_clock_override_rejected":true,
        "fixture_atom_count": ATOMS, "fixture_spike_count": spikes.len(), "fixture_csr_offsets": offsets,
        "raw_spikes_sha256": format!("{:x}", Sha256::digest(&spike_bytes)),
        "csr_offsets_sha256": format!("{:x}", Sha256::digest(&csr_bytes)),
        "sh_basis_sha256": format!("{:x}", Sha256::digest(&basis_bytes)),
        "raw_spikes_and_integrator_buffers_unchanged": true,
        "independent_native_projection_sha256": expected_sha,
        "independent_native_geometry_spectra": spectra,
        "before_freeze": before, "after_freeze": after,
    }));
    drop(pipeline); // all cfg-backed CUDA allocations remain alive through drop
}
