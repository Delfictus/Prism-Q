//! Exact native full-record/compact-storage equivalence and CUDA-event timing.
//! These are explicit kernel fixtures, not a molecular campaign or MD speed claim.
#![cfg(feature = "gpu")]
use cudarc::driver::{sys, CudaContext, CudaSlice, CudaStream, DevicePtr};
use prism_nhs::{
    interferometric_adjudicator::InterferometricAdjudicatorFfi as Adj,
    protocol_state::ProtocolState, rich_spike::RichSpike, so3_project::ContactShellTile,
};
use sha2::{Digest, Sha256};
use std::{ffi::c_void, ptr, sync::Arc};
extern "C" {
    fn prism_apply_gradient_gasp_clock_launch(
        raw: *const c_void,
        out: *mut c_void,
        forces: *const c_void,
        masses: *const c_void,
        adj: *const c_void,
        shift: *mut c_void,
        mass: *mut c_void,
        clock: *const u32,
        n: u32,
        atoms: u32,
        stream: *mut c_void,
    ) -> i32;
    fn prism_apply_gradient_gasp_positions_clock_launch(
        raw: *const c_void,
        out: *mut c_void,
        forces: *const c_void,
        masses: *const c_void,
        adj: *const c_void,
        shift: *mut c_void,
        mass: *mut c_void,
        clock: *const u32,
        n: u32,
        atoms: u32,
        stream: *mut c_void,
    ) -> i32;
    fn prism_momentum_guard_check_launch(
        shift: *const c_void,
        mass: *const c_void,
        correction: *mut c_void,
        adj: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
    fn prism_apply_com_correction_launch(
        out: *mut c_void,
        correction: *const c_void,
        n: u32,
        stream: *mut c_void,
    ) -> i32;
    fn prism_apply_com_positions_correction_launch(
        out: *mut c_void,
        correction: *const c_void,
        n: u32,
        stream: *mut c_void,
    ) -> i32;
    fn prism_so3_project_run(
        raw: *const c_void,
        offsets: *const u32,
        clusters: u32,
        basis: *const f32,
        tiles: *mut c_void,
        frame: u32,
        stream_id: u32,
        stream: *mut c_void,
    ) -> i32;
    fn prism_so3_project_positions_run(
        raw: *const c_void,
        xyz: *const f32,
        offsets: *const u32,
        clusters: u32,
        basis: *const f32,
        tiles: *mut c_void,
        frame: u32,
        stream_id: u32,
        stream: *mut c_void,
    ) -> i32;
}
fn ok(r: sys::CUresult) {
    assert_eq!(r, sys::CUresult::CUDA_SUCCESS);
}
fn address<T>(b: &CudaSlice<T>, s: &Arc<CudaStream>) -> u64 {
    let (p, _g) = b.device_ptr(s);
    p
}
fn bytes<T>(data: &[T]) -> Vec<u8> {
    unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)).to_vec()
    }
}
fn hash(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}
const ATOMS: usize = 512;
fn fixture(extra_records: usize) -> (Vec<RichSpike>, Vec<u32>) {
    let mut records = Vec::new();
    let mut offsets = vec![0];
    // Empty, small, tile/warp boundary and multi-iteration reduction cases.
    for (cluster, count) in [0, 1, 15, 16, 17, 31, 32, 33, 257, 3]
        .into_iter()
        .enumerate()
    {
        for local in 0..count {
            let i = records.len();
            let mut s = RichSpike::zero();
            s.x = (local as f32 * 0.37).sin() * 2.0;
            s.y = (local as f32 * 0.71).cos() * 1.7;
            s.z = (local as f32 * 0.19).sin() * 0.8;
            if cluster == 9 {
                s.x = 0.25;
                s.y = 0.25;
                s.z = 0.25;
            } // centroid-coincident, all unmapped
            s.cluster_id = cluster as i32;
            s.t_frame = (i + 3) as u32;
            s.residue_id = if cluster == 9 {
                -1
            } else {
                match i % 11 {
                    0 => -1,
                    1 => 1024,
                    2 => 1023,
                    3 => 900,
                    _ => (i % ATOMS) as i32,
                }
            };
            s.vib_energy = match i % 4 {
                0 => 0.0,
                1 => 0.01,
                2 => 1.0,
                _ => 2.0,
            };
            s.intensity_packed = ((i % 4) as u32) << 30 | (i as u32 & 0x3fff);
            s.water_density = 0.125 + (i % 13) as f32 * 0.125;
            s.wd_change = -0.25;
            s.causal_lag = if i % 2 == 0 { -0.75 } else { 1.25 };
            s.n_excited = (i % 7) as u32;
            s.spike_source = 1u32 << (i % 31);
            s.origin_phase = 1u32 << ((i + 7) % 31);
            s.chem_flags = 1u32 << ((i + 13) % 31);
            s.kinetic_delta = -0.5;
            records.push(s);
        }
        offsets.push(records.len() as u32);
    }
    if extra_records > records.len() {
        let pattern = records.clone();
        let cluster = (offsets.len() - 1) as i32;
        while records.len() < extra_records {
            let mut s = pattern[records.len() % pattern.len()];
            s.cluster_id = cluster;
            records.push(s);
        }
        offsets.push(records.len() as u32);
    }
    (records, offsets)
}
struct GraphCase {
    stream: Arc<CudaStream>,
    graph: sys::CUgraph,
    exec: sys::CUgraphExec,
    compact: bool,
    com: bool,
    raw: CudaSlice<u8>,
    scratch: CudaSlice<u8>,
    tiles: CudaSlice<u8>,
    clock: CudaSlice<u8>,
    adj: CudaSlice<u8>,
    shift: CudaSlice<f32>,
    mass: CudaSlice<f32>,
    correction: CudaSlice<f32>,
    _offsets: CudaSlice<u32>,
    _forces: CudaSlice<f32>,
    _masses: CudaSlice<f32>,
    _dt: CudaSlice<f32>,
    count: usize,
    clusters: usize,
    raw_bytes: Vec<u8>,
    nodes: usize,
}
struct Observation {
    xyz: Vec<u8>,
    tiles: Vec<u8>,
    shift: Vec<u8>,
    mass: Vec<u8>,
    correction: Vec<u8>,
    reason: u32,
    momentum: u32,
}
impl GraphCase {
    fn new(
        ctx: &Arc<CudaContext>,
        records: &[RichSpike],
        offsets: &[u32],
        compact: bool,
        com: bool,
    ) -> Self {
        let stream = ctx.new_stream().expect("real CUDA stream");
        let rs = stream.cu_stream() as *mut c_void;
        prism_nhs::sh_basis::init_device_basis(rs).unwrap();
        stream.synchronize().unwrap();
        let basis = prism_nhs::sh_basis::k_lm_device_ptr().unwrap();
        let count = records.len();
        let clusters = offsets.len() - 1;
        let raw_bytes = bytes(records);
        let mut raw = stream.alloc_zeros::<u8>(raw_bytes.len()).unwrap();
        let scratch = stream
            .alloc_zeros::<u8>(count * if compact { 12 } else { 64 })
            .unwrap();
        let tiles = stream
            .alloc_zeros::<u8>(clusters * std::mem::size_of::<ContactShellTile>())
            .unwrap();
        let clock = stream
            .alloc_zeros::<u8>(std::mem::size_of::<ProtocolState>())
            .unwrap();
        let mut d_offsets = stream.alloc_zeros::<u32>(offsets.len()).unwrap();
        let mut forces = stream.alloc_zeros::<f32>(ATOMS * 3).unwrap();
        let mut masses = stream.alloc_zeros::<f32>(ATOMS).unwrap();
        let mut dt = stream.alloc_zeros::<f32>(1).unwrap();
        let mut adj = stream
            .alloc_zeros::<u8>(std::mem::size_of::<Adj>())
            .unwrap();
        let shift = stream.alloc_zeros::<f32>(3).unwrap();
        let mass = stream.alloc_zeros::<f32>(1).unwrap();
        let correction = stream.alloc_zeros::<f32>(3).unwrap();
        stream.memcpy_htod(&raw_bytes, &mut raw).unwrap();
        stream.memcpy_htod(offsets, &mut d_offsets).unwrap();
        let host_forces: Vec<f32> = (0..ATOMS * 3)
            .map(|i| (i as f32 * 0.3).sin() * 0.7)
            .collect();
        let host_masses: Vec<f32> = (0..ATOMS)
            .map(|i| if i == 5 { 0.0 } else { 1.0 + (i % 12) as f32 })
            .collect();
        stream.memcpy_htod(&host_forces, &mut forces).unwrap();
        stream.memcpy_htod(&host_masses, &mut masses).unwrap();
        stream.memcpy_htod(&[0.002f32], &mut dt).unwrap();
        let mut ab = vec![0u8; std::mem::size_of::<Adj>()];
        let o = std::mem::offset_of!(Adj, d_dt);
        ab[o..o + 8].copy_from_slice(&address(&dt, &stream).to_ne_bytes());
        let o = std::mem::offset_of!(Adj, gasp_gain_eta);
        ab[o..o + 4].copy_from_slice(&1.0f32.to_ne_bytes());
        let o = std::mem::offset_of!(Adj, force_burst_step);
        ab[o..o + 4].copy_from_slice(&11u32.to_ne_bytes());
        stream.memcpy_htod(&ab, &mut adj).unwrap();
        let mut anchors = [u32::MAX; 1024];
        for (i, a) in anchors[..ATOMS].iter_mut().enumerate() {
            *a = i as u32;
        }
        anchors[900] = ATOMS as u32;
        assert_eq!(
            unsafe { prism_nhs::so3_project::set_residue_to_calpha(anchors.as_ptr(), 1024, rs) },
            0
        );
        stream.synchronize().unwrap();
        let p = [
            address(&raw, &stream),
            address(&scratch, &stream),
            address(&forces, &stream),
            address(&masses, &stream),
            address(&adj, &stream),
            address(&shift, &stream),
            address(&mass, &stream),
            address(&correction, &stream),
            address(&d_offsets, &stream),
            address(&tiles, &stream),
            address(&clock, &stream) + std::mem::offset_of!(ProtocolState, current_step) as u64,
        ];
        if compact {
            // Reject a populated zero-atom request before capture or any write.
            assert_eq!(unsafe {
                prism_apply_gradient_gasp_positions_clock_launch(
                    p[0] as _, p[1] as _, p[2] as _, p[3] as _, p[4] as _,
                    ptr::null_mut(), ptr::null_mut(), p[10] as _,
                    count as u32, 0, rs,
                )
            }, 1);
        }
        let mut graph = ptr::null_mut();
        let mut exec = ptr::null_mut();
        let mut nodes = 0;
        unsafe {
            cudarc::driver::set_capture_mode_active(true);
            ok(sys::cuStreamBeginCapture_v2(
                stream.cu_stream(),
                sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            ));
            let launch = if compact {
                prism_apply_gradient_gasp_positions_clock_launch
            } else {
                prism_apply_gradient_gasp_clock_launch
            };
            assert_eq!(
                launch(
                    p[0] as _,
                    p[1] as _,
                    p[2] as _,
                    p[3] as _,
                    p[4] as _,
                    if com { p[5] as _ } else { ptr::null_mut() },
                    if com { p[6] as _ } else { ptr::null_mut() },
                    p[10] as _,
                    count as u32,
                    ATOMS as u32,
                    rs
                ),
                0
            );
            if com {
                assert_eq!(
                    prism_momentum_guard_check_launch(
                        p[5] as _, p[6] as _, p[7] as _, p[4] as _, rs
                    ),
                    0
                );
                let correct = if compact {
                    prism_apply_com_positions_correction_launch
                } else {
                    prism_apply_com_correction_launch
                };
                assert_eq!(correct(p[1] as _, p[7] as _, count as u32, rs), 0);
            }
            let rc = if compact {
                prism_so3_project_positions_run(
                    p[0] as _,
                    p[1] as _,
                    p[8] as _,
                    clusters as u32,
                    basis,
                    p[9] as _,
                    53,
                    7,
                    rs,
                )
            } else {
                prism_so3_project_run(
                    p[1] as _,
                    p[8] as _,
                    clusters as u32,
                    basis,
                    p[9] as _,
                    53,
                    7,
                    rs,
                )
            };
            assert_eq!(rc, 0);
            ok(sys::cuStreamEndCapture(stream.cu_stream(), &mut graph));
            cudarc::driver::set_capture_mode_active(false);
            ok(sys::cuGraphGetNodes(graph, ptr::null_mut(), &mut nodes));
            assert_eq!(nodes, if com { 4 } else { 2 });
            ok(sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0));
        }
        Self {
            stream,
            graph,
            exec,
            compact,
            com,
            raw,
            scratch,
            tiles,
            clock,
            adj,
            shift,
            mass,
            correction,
            _offsets: d_offsets,
            _forces: forces,
            _masses: masses,
            _dt: dt,
            count,
            clusters,
            raw_bytes,
            nodes,
        }
    }
    fn clock_address(&self) -> u64 {
        address(&self.clock, &self.stream)
            + std::mem::offset_of!(ProtocolState, current_step) as u64
    }
    fn update_clock(&self, step: u32) {
        unsafe {
            ok(sys::cuMemcpyHtoDAsync_v2(
                self.clock_address(),
                &step as *const u32 as _,
                4,
                self.stream.cu_stream(),
            ));
        }
        // Preserve stream-ordered pool ownership and the host scalar lifetime.
        self.stream.synchronize().unwrap();
    }
    fn launch(&self) {
        unsafe {
            ok(sys::cuGraphLaunch(self.exec, self.stream.cu_stream()));
        }
    }
    fn read(&self, step: u32) -> Observation {
        self.stream.synchronize().unwrap();
        let mut observed = 0u32;
        unsafe {
            ok(sys::cuMemcpyDtoHAsync_v2(
                &mut observed as *mut u32 as _,
                self.clock_address(),
                4,
                self.stream.cu_stream(),
            ));
        }
        self.stream.synchronize().unwrap();
        assert_eq!(observed, step);
        let mut raw = vec![0u8; self.raw_bytes.len()];
        self.stream.memcpy_dtoh(&self.raw, &mut raw).unwrap();
        assert_eq!(raw, self.raw_bytes);
        let mut scratch = vec![0u8; self.count * if self.compact { 12 } else { 64 }];
        self.stream
            .memcpy_dtoh(&self.scratch, &mut scratch)
            .unwrap();
        let xyz = if self.compact {
            scratch
        } else {
            let mut xyz = Vec::with_capacity(self.count * 12);
            for (i, s) in scratch.chunks_exact(64).enumerate() {
                assert_eq!(&s[12..], &raw[i * 64 + 12..(i + 1) * 64]);
                xyz.extend_from_slice(&s[..12]);
            }
            xyz
        };
        let mut tiles = vec![0u8; self.clusters * std::mem::size_of::<ContactShellTile>()];
        self.stream.memcpy_dtoh(&self.tiles, &mut tiles).unwrap();
        let mut a = vec![0u8; std::mem::size_of::<Adj>()];
        self.stream.memcpy_dtoh(&self.adj, &mut a).unwrap();
        let read_field = |o| u32::from_ne_bytes(a[o..o + 4].try_into().unwrap());
        let mut shift = [0.0f32; 3];
        let mut mass = [0.0f32];
        let mut correction = [0.0f32; 3];
        self.stream.memcpy_dtoh(&self.shift, &mut shift).unwrap();
        self.stream.memcpy_dtoh(&self.mass, &mut mass).unwrap();
        self.stream
            .memcpy_dtoh(&self.correction, &mut correction)
            .unwrap();
        Observation {
            xyz,
            tiles,
            shift: bytes(&shift),
            mass: bytes(&mass),
            correction: bytes(&correction),
            reason: read_field(std::mem::offset_of!(Adj, adjudication_reason_flags)),
            momentum: read_field(std::mem::offset_of!(Adj, momentum_violation_flag)),
        }
    }
    fn timed_ms(&self, repeats: usize) -> f32 {
        let mut start = ptr::null_mut();
        let mut stop = ptr::null_mut();
        let mut ms = 0.0;
        unsafe {
            ok(sys::cuEventCreate(&mut start, 0));
            ok(sys::cuEventCreate(&mut stop, 0));
            ok(sys::cuEventRecord(start, self.stream.cu_stream()));
            for _ in 0..repeats {
                self.launch();
            }
            ok(sys::cuEventRecord(stop, self.stream.cu_stream()));
            ok(sys::cuEventSynchronize(stop));
            ok(sys::cuEventElapsedTime(&mut ms, start, stop));
            ok(sys::cuEventDestroy_v2(start));
            ok(sys::cuEventDestroy_v2(stop));
        }
        assert!(ms.is_finite() && ms > 0.0);
        ms / repeats as f32
    }
}
impl Drop for GraphCase {
    fn drop(&mut self) {
        self.stream.synchronize().unwrap();
        unsafe {
            ok(sys::cuGraphExecDestroy(self.exec));
            ok(sys::cuGraphDestroy(self.graph));
        }
    }
}
fn assert_bytes(label: &str, a: &[u8], b: &[u8]) {
    assert_eq!(a.len(), b.len());
    if let Some(i) = a.iter().zip(b).position(|(x, y)| x != y) {
        panic!("{label} byte mismatch at{i}:{} !={}", a[i], b[i]);
    }
}
#[test]
#[ignore = "GPU_RUNTIME_REQUIRED: real captured kernels; no MD; optional large kernel fixture"]
fn compact_positions_preserve_native_gasp_com_and_spectral_tiles() {
    assert_eq!(std::mem::size_of::<RichSpike>(), 64);
    assert_eq!(std::mem::size_of::<[f32; 3]>(), 12);
    let requested = std::env::var("PRISM_COMPACT_BENCH_RECORDS")
        .ok()
        .map(|s| s.parse::<usize>().expect("integer record count"))
        .unwrap_or(0);
    assert!(requested <= u32::MAX as usize, "CSR index limit");
    let (records, offsets) = fixture(requested);
    let ctx = CudaContext::new(0).expect("actual CUDA required, never skipped success");
    let mut receipts = Vec::new();
    for com in [false, true] {
        let full = GraphCase::new(&ctx, &records, &offsets, false, com);
        let compact = GraphCase::new(&ctx, &records, &offsets, true, com);
        assert_ne!(full.clock_address(), compact.clock_address());
        let mut rounds = Vec::new();
        let mut ordinary_xyz = None;
        for (step_index, step) in [10u32, 11, 12, 11].into_iter().enumerate() {
            full.update_clock(step);
            compact.update_clock(step);
            if step_index % 2 == 0 {
                full.launch();
                compact.launch();
            } else {
                compact.launch();
                full.launch();
            }
            let a = full.read(step);
            let b = compact.read(step);
            assert_bytes("xyz", &a.xyz, &b.xyz);
            assert_bytes("all ContactShellTile bytes", &a.tiles, &b.tiles);
            assert_bytes("COM numerator", &a.shift, &b.shift);
            assert_bytes("COM mass", &a.mass, &b.mass);
            assert_bytes("COM correction", &a.correction, &b.correction);
            assert_eq!(a.reason, b.reason);
            assert_eq!(a.momentum, b.momentum);
            assert_ne!(
                a.reason & 8,
                0,
                "saturation fixture must exercise native reason bit"
            );
            if step == 10 {
                ordinary_xyz = Some(a.xyz.clone());
            }
            if step == 11 {
                assert_ne!(
                    ordinary_xyz.as_ref().unwrap(),
                    &a.xyz,
                    "burst must actually change positions"
                );
            }
            // Complete-byte equality above includes all four spectral planes and
            // every tag; also require the fixture to excite each plane.
            for plane in [
                std::mem::offset_of!(ContactShellTile, geo_alm),
                std::mem::offset_of!(ContactShellTile, caus_alm),
                std::mem::offset_of!(ContactShellTile, therm_alm),
                std::mem::offset_of!(ContactShellTile, chem_alm),
            ] {
                assert!(a
                    .tiles
                    .chunks_exact(std::mem::size_of::<ContactShellTile>())
                    .any(|t| t[plane..plane + 64 * 4]
                        .chunks_exact(4)
                        .any(|v| f32::from_ne_bytes(v.try_into().unwrap()) != 0.0)));
            }
            for (cluster, tile) in a
                .tiles
                .chunks_exact(std::mem::size_of::<ContactShellTile>())
                .enumerate()
            {
                let population = &records[offsets[cluster] as usize..offsets[cluster + 1] as usize];
                for (offset, expected) in [
                    (
                        std::mem::offset_of!(ContactShellTile, spike_count),
                        population.len() as u32,
                    ),
                    (
                        std::mem::offset_of!(ContactShellTile, agg_spike_source),
                        population.iter().fold(0, |v, r| v | r.spike_source),
                    ),
                    (
                        std::mem::offset_of!(ContactShellTile, agg_origin_phase),
                        population.iter().fold(0, |v, r| v | r.origin_phase),
                    ),
                    (
                        std::mem::offset_of!(ContactShellTile, agg_chem_flags),
                        population.iter().fold(0, |v, r| v | r.chem_flags),
                    ),
                ] {
                    assert_eq!(
                        u32::from_ne_bytes(tile[offset..offset + 4].try_into().unwrap()),
                        expected
                    );
                }
                for plane in [
                    std::mem::offset_of!(ContactShellTile, geo_alm),
                    std::mem::offset_of!(ContactShellTile, caus_alm),
                    std::mem::offset_of!(ContactShellTile, therm_alm),
                    std::mem::offset_of!(ContactShellTile, chem_alm),
                ] {
                    assert!(tile[plane..plane + 64 * 4]
                        .chunks_exact(4)
                        .all(|v| f32::from_ne_bytes(v.try_into().unwrap()).is_finite()));
                }
            }
            rounds.push(serde_json::json!({"step":step,"xyz_sha256":hash(&a.xyz),"tiles_sha256":hash(&a.tiles),
                "reason_flags":a.reason,"momentum_flag":a.momentum}));
        }
        // Previous equivalence replays warm both graphs. Alternate timing order;
        // event durations cover device graph execution only, no upload/readback.
        let repeats = if records.len() > 100000 { 2 } else { 25 };
        let mut full_ms = Vec::new();
        let mut compact_ms = Vec::new();
        for round in 0..5 {
            if round % 2 == 0 {
                full_ms.push(full.timed_ms(repeats));
                compact_ms.push(compact.timed_ms(repeats));
            } else {
                compact_ms.push(compact.timed_ms(repeats));
                full_ms.push(full.timed_ms(repeats));
            }
        }
        receipts.push(serde_json::json!({"com_enabled":com,"graph_nodes":full.nodes,"compact_graph_nodes":compact.nodes,
            "live_clock_rounds":rounds,"timing_scope":"kernel_fixture_only_not_molecular_campaign_or_end_to_end_MD",
            "gpu_event_ms_per_replay":{"full":full_ms,"compact":compact_ms},"timed_replays_per_sample":repeats}));
    }
    println!(
        "{}",
        serde_json::json!({"schema":"prism.compact_positions_native_equivalence.v1",
        "input_scope":"explicit_deterministic_edge_case_kernel_fixture_not_recorded_molecular_input",
        "n_spikes":records.len(),"clusters":offsets.len()-1,"cluster_offsets":offsets,"raw_sha256":hash(&bytes(&records)),
        "requested_allocation_bytes":{"raw":records.len()*64,"full_scratch":records.len()*64,"compact_scratch":records.len()*12,"saved":records.len()*52},
        "allocator_reserved_bytes":"not_measured","all_comparisons":"bitwise_no_tolerance","md_steps_advanced":0,"results":receipts})
    );
}
