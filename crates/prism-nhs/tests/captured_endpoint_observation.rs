//! Native captured KCC -> physical clock -> Ghost/ZSTR endpoint contract.
//! Explicit kernel fixture; no MD, molecular mechanism or full pipeline execution.
#![cfg(all(feature = "gpu", feature = "v2_ignition"))]
use cudarc::driver::{sys, CudaContext, CudaSlice, CudaStream, DevicePtr};
use prism_nhs::{captured_pipeline::{KccCausalCandidates, measured_kcc_causal_leads,
    prism_bind_ghost_physical_clock_launch}, fused_engine::KccData,
    ghost_tile::GhostTileRing, protocol_state::ProtocolState,
    interferometric_adjudicator::InterferometricAdjudicatorFfi as Adj,
    rich_spike::RichSpike, so3_project::ContactShellTile};
use std::{ffi::c_void, ptr, sync::Arc};

extern "C" {
    fn prism_ghost_pipe_stage_launch_v2(ring: u64, tiles: *const c_void,
        spikes: *const c_void, offsets: *const c_void, adj: *const c_void,
        leads: *const c_void, frame: u64, clusters: u32, capacity: u32,
        stream: *mut c_void, firehose: u32, obs: f32, disc: f32, gear: u32,
        dt_fs: f32, step: u64, counter: u64, prune: f32) -> i32;
}
fn ok(rc: sys::CUresult) { assert_eq!(rc, sys::CUresult::CUDA_SUCCESS); }
fn address<T>(buffer: &CudaSlice<T>, stream: &Arc<CudaStream>) -> u64 {
    let (ptr, _guard) = buffer.device_ptr(stream); ptr
}
fn bytes<T>(value: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(value.as_ptr() as _, std::mem::size_of_val(value)) }
}
fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset+4].try_into().unwrap())
}
fn u64_at(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset+8].try_into().unwrap())
}
struct Endpoint {
    stream: Arc<CudaStream>, graph: sys::CUgraph, exec: sys::CUgraphExec,
    protocol: CudaSlice<u8>, ghost_clock: CudaSlice<u64>,
    corr: CudaSlice<f32>, count: CudaSlice<u32>, active: CudaSlice<u32>, leads: CudaSlice<u32>,
    _csr: KccCausalCandidates, _tiles: CudaSlice<u8>, _spikes: CudaSlice<u8>,
    _offsets: CudaSlice<u32>, _adj: CudaSlice<u8>,
    ring: GhostTileRing, zstr: CudaSlice<u8>, slot: CudaSlice<u32>,
    candidates: Vec<Vec<u32>>, stream_id: u32,
}
impl Endpoint {
    fn new(ctx: &Arc<CudaContext>, stream_id: u32) -> Self {
        let stream = ctx.new_stream().unwrap();
        let candidates = vec![vec![2, 1, 0, 3, 4, 5], vec![], vec![7, 6]];
        assert!(KccCausalCandidates::new(&stream, &[vec![8]], 8).is_err());
        let csr = KccCausalCandidates::new(&stream, &candidates, 8).unwrap();
        let protocol = stream.alloc_zeros::<u8>(std::mem::size_of::<ProtocolState>()).unwrap();
        let ghost_clock = stream.clone_htod(&[0xFEDCBA9876543210u64]).unwrap();
        let corr = stream.alloc_zeros::<f32>(8).unwrap();
        let count = stream.clone_htod(&[3u32; 8]).unwrap();
        let active = stream.clone_htod(&[1u32; 8]).unwrap();
        let mut leads = stream.clone_htod(&[u32::MAX; 3]).unwrap();
        let tiles = stream.clone_htod(bytes(&[ContactShellTile::zero(); 3])).unwrap();
        let spikes = stream.clone_htod(bytes(&[RichSpike::zero(); 3])).unwrap();
        let offsets = stream.clone_htod(&[0u32, 1, 2, 3]).unwrap();
        let mut adj_bytes = vec![0u8; std::mem::size_of::<Adj>()];
        adj_bytes[std::mem::offset_of!(Adj, adjudication_code)] = 1;
        let off = std::mem::offset_of!(Adj, current_divergence);
        adj_bytes[off..off+4].copy_from_slice(&1.0f32.to_ne_bytes());
        let adj = stream.clone_htod(&adj_bytes).unwrap();
        let ring = GhostTileRing::allocate(15).unwrap();
        let zstr = stream.alloc_zeros::<u8>(8 * 4096).unwrap();
        let slot = stream.alloc_zeros::<u32>(1).unwrap();
        stream.synchronize().unwrap();
        let raw = stream.cu_stream() as *mut c_void;
        let clock = address(&protocol, &stream) + std::mem::offset_of!(ProtocolState, current_step) as u64;
        let gclock = address(&ghost_clock, &stream);
        unsafe {
            assert_ne!(prism_bind_ghost_physical_clock_launch(ptr::null(), gclock as _, raw), 0);
            assert_ne!(prism_bind_ghost_physical_clock_launch((clock+1) as _, gclock as _, raw), 0);
            assert_ne!(prism_bind_ghost_physical_clock_launch(clock as _, (gclock+4) as _, raw), 0);
        }
        let p = [address(&tiles,&stream),address(&spikes,&stream),address(&offsets,&stream),
            address(&adj,&stream),address(&leads,&stream),address(&zstr,&stream),address(&slot,&stream)];
        let selection=csr.bind(&stream,&corr,&count,&active,&mut leads).unwrap();
        let mut graph = ptr::null_mut(); let mut exec = ptr::null_mut(); let mut nodes = 0;
        unsafe {
            cudarc::driver::set_capture_mode_active(true);
            ok(sys::cuStreamBeginCapture_v2(stream.cu_stream(), sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
            selection.launch().unwrap();
            assert_eq!(prism_bind_ghost_physical_clock_launch(clock as _, gclock as _, raw), 0);
            assert_eq!(prism_ghost_pipe_stage_launch_v2(ring.device_base,
                p[0] as _,p[1] as _,p[2] as _,p[3] as _,p[4] as _, 999, 3, 15,
                raw, 0, 3.0, 12.0, u32::MAX, 2.0, 123456, gclock, 0.01), 0);
            assert_eq!(prism_nhs::zstr::ffi::prism_zstr_device_slot_update_launch(gclock as _,
                8, p[5] as _, 4096, p[6] as _,
                1, 0.002, u32::MAX, stream_id, 1, 0, ptr::null_mut(), raw), 0);
            ok(sys::cuStreamEndCapture(stream.cu_stream(), &mut graph));
            cudarc::driver::set_capture_mode_active(false);
            ok(sys::cuGraphGetNodes(graph, ptr::null_mut(), &mut nodes));
            assert_eq!(nodes, 4);
            ok(sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0));
        }
        drop(selection); // event guards publish only after capture ends
        Self { stream, graph, exec, protocol, ghost_clock, corr, count, active, leads,
            _csr:csr, _tiles:tiles, _spikes:spikes, _offsets:offsets, _adj:adj,
            ring, zstr, slot, candidates, stream_id }
    }
    fn update(&mut self, step: u32, pass: usize) -> KccData {
        let mut kcc = KccData::empty(); kcc.n_residues = 8;
        kcc.temporal_corr = vec![0.25,-0.75,0.75,f32::NAN,f32::INFINITY,-0.0,
            f32::from_bits(1),f32::from_bits(0x80000001)];
        kcc.residue_count = vec![3;8]; kcc.active_causal = vec![1;8];
        if pass % 2 == 1 { kcc.temporal_corr[2] = -0.9; kcc.residue_count[6] = 1; }
        if pass == 2 { kcc.active_causal[1] = 0; kcc.active_causal[2] = 0; }
        if pass == 3 { kcc.temporal_corr[0] = 0.0; kcc.temporal_corr[1] = f32::NEG_INFINITY; kcc.temporal_corr[2] = -0.0; }
        let mut p = vec![0u8; std::mem::size_of::<ProtocolState>()];
        let off = std::mem::offset_of!(ProtocolState, current_step); p[off..off+4].copy_from_slice(&step.to_ne_bytes());
        self.stream.memcpy_htod(&p, &mut self.protocol).unwrap();
        self.stream.memcpy_htod(&kcc.temporal_corr, &mut self.corr).unwrap();
        self.stream.memcpy_htod(&kcc.residue_count, &mut self.count).unwrap();
        self.stream.memcpy_htod(&kcc.active_causal, &mut self.active).unwrap();
        self.stream.synchronize().unwrap(); // host fixture source lifetime
        kcc
    }
    fn launch(&self) { unsafe { ok(sys::cuGraphLaunch(self.exec, self.stream.cu_stream())); } }
    fn check(&self, step: u32, pass: usize, kcc: &KccData) -> serde_json::Value {
        self.stream.synchronize().unwrap();
        let expected: Vec<u32> = measured_kcc_causal_leads(&self.candidates, kcc, step as u64).unwrap()
            .iter().map(|r| r.causal_lead_residue.unwrap_or(u32::MAX)).collect();
        assert_eq!(self.stream.clone_dtoh(&self.leads).unwrap(), expected);
        assert_eq!(self.stream.clone_dtoh(&self.ghost_clock).unwrap(), vec![step as u64]);
        assert_eq!(self.ring.n_frames_written(), (pass as u32+1)*3);
        let payload = self.ring.payload_bytes();
        for record in payload[pass*3*4096..(pass+1)*3*4096].chunks_exact(4096) {
            assert_eq!(u64_at(record,152), step as u64);
            assert_eq!(u32_at(record,124), expected[u32_at(record,8) as usize]);
            assert_eq!(u64_at(record,0), 999); // capture identity unchanged; step_idx is physical time
        }
        let slot = self.stream.clone_dtoh(&self.slot).unwrap()[0]; assert_eq!(slot,step%8);
        let zstr = self.stream.clone_dtoh(&self.zstr).unwrap();
        assert_eq!(u64_at(&zstr,slot as usize*4096), step as u64);
        serde_json::json!({"stream":self.stream_id,"step":step,"leads":expected,"zstr_slot":slot,"records":3})
    }
}
impl Drop for Endpoint {
    fn drop(&mut self) { self.stream.synchronize().unwrap(); unsafe {
        ok(sys::cuGraphExecDestroy(self.exec)); ok(sys::cuGraphDestroy(self.graph));
    } }
}
fn check_all_empty_csr(ctx: &Arc<CudaContext>) {
    let s = ctx.new_stream().unwrap();
    // 130 clusters cross a CUDA block boundary; no candidate reads are legal.
    let csr = KccCausalCandidates::new(&s, &vec![vec![]; 130], 1).unwrap();
    let corr = s.clone_htod(&[1.0f32]).unwrap();
    let count = s.clone_htod(&[3u32]).unwrap();
    let active = s.clone_htod(&[1u32]).unwrap();
    let mut leads = s.alloc_zeros::<u32>(130).unwrap();
    let mut wrong = s.alloc_zeros::<u32>(129).unwrap();
    assert!(csr.select(&s,&corr,&count,&active,&mut wrong).is_err());
    s.synchronize().unwrap();
    let selection=csr.bind(&s,&corr,&count,&active,&mut leads).unwrap();
    let mut graph=ptr::null_mut(); let mut exec=ptr::null_mut();
    unsafe {
        cudarc::driver::set_capture_mode_active(true);
        ok(sys::cuStreamBeginCapture_v2(s.cu_stream(),sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
        selection.launch().unwrap();
        ok(sys::cuStreamEndCapture(s.cu_stream(),&mut graph));
        cudarc::driver::set_capture_mode_active(false);
        drop(selection);
        ok(sys::cuGraphInstantiateWithFlags(&mut exec,graph,0));
        ok(sys::cuGraphLaunch(exec,s.cu_stream()));
        s.synchronize().unwrap();
        assert_eq!(s.clone_dtoh(&leads).unwrap(),vec![u32::MAX;130]);
        ok(sys::cuGraphExecDestroy(exec)); ok(sys::cuGraphDestroy(graph));
    }
}
#[test]
#[ignore = "requires actual CUDA and explicit invocation; no silent skip"]
fn captured_endpoints_preserve_kcc_selection_and_physical_ghost_zstr_time() {
    let ctx = CudaContext::new(0).expect("actual CUDA required");
    check_all_empty_csr(&ctx);
    let mut a = Endpoint::new(&ctx,0); let mut b = Endpoint::new(&ctx,1);
    let mut rows=Vec::new();
    for (pass,(sa,sb)) in [(10,110),(20,207),(20,207),(55000,44000),(u32::MAX,u32::MAX-1)].into_iter().enumerate() {
        let ka=a.update(sa,pass); let kb=b.update(sb,pass);
        // Both streams are enqueued before synchronization/readback.
        a.launch(); b.launch(); rows.push(a.check(sa,pass,&ka)); rows.push(b.check(sb,pass,&kb));
    }
    println!("{}",serde_json::json!({"schema":"prism.captured_endpoint_observation.v1",
        "scope":"native captured fixture; no molecular MD or global WHILE composition",
        "physical_md_steps":0,"streams":2,"graph_replays":10,"recaptures":0,
        "ghost_records_checked":30,"zstr_headers_checked":10,"host_ghost_counter_writes_after_capture":0,
        "nan_inf_signed_zero_subnormal_ties_checked":true,"empty_cluster_and_all_empty_csr_checked":true,"rows":rows}));
}
