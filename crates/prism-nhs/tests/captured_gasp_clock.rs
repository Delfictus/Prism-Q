//! Actual two-stream captured native GASP regression. No MD is advanced.
#![cfg(feature = "gpu")]
use cudarc::driver::{sys, CudaContext, CudaSlice, CudaStream, DevicePtr};
use prism_nhs::{
    interferometric_adjudicator::InterferometricAdjudicatorFfi as Adj,
    protocol_state::ProtocolState, rich_spike::RichSpike,
};
use std::{ffi::c_void, ptr, sync::Arc};

extern "C" {
    fn prism_apply_gradient_gasp_clock_launch(
        input: *const c_void,
        output: *mut c_void,
        forces: *const c_void,
        masses: *const c_void,
        adj: *const c_void,
        shift: *mut c_void,
        mass: *mut c_void,
        clock: *const u32,
        spikes: u32,
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
        spikes: *mut c_void,
        correction: *const c_void,
        count: u32,
        stream: *mut c_void,
    ) -> i32;
}
fn ok(result: sys::CUresult) {
    assert_eq!(result, sys::CUresult::CUDA_SUCCESS);
}
fn address<T>(buffer: &CudaSlice<T>, stream: &Arc<CudaStream>) -> u64 {
    let (p, _guard) = buffer.device_ptr(stream);
    p
}
struct CapturedGasp {
    stream: Arc<CudaStream>,
    graph: sys::CUgraph,
    exec: sys::CUgraphExec,
    input: CudaSlice<u8>,
    output: CudaSlice<u8>,
    protocol: CudaSlice<u8>,
    _forces: CudaSlice<f32>,
    _masses: CudaSlice<f32>,
    _dt: CudaSlice<f32>,
    adj: CudaSlice<u8>,
    _shift: CudaSlice<f32>,
    _mass: CudaSlice<f32>,
    _correction: CudaSlice<f32>,
    input_bytes: Vec<u8>,
    nodes: usize,
}
impl CapturedGasp {
    fn new(ctx: &Arc<CudaContext>) -> Self {
        let stream = ctx.new_stream().expect("real independent CUDA stream");
        let mut spikes = [RichSpike::zero(); 2];
        for (i, s) in spikes.iter_mut().enumerate() {
            s.x = i as f32;
            s.residue_id = i as i32;
            s.vib_energy = 1.0;
            s.intensity_packed = 1u32 << 30; // native Q_s[1] = 1.0
        }
        let input_bytes = unsafe {
            std::slice::from_raw_parts(spikes.as_ptr() as *const u8, std::mem::size_of_val(&spikes))
                .to_vec()
        };
        let mut input = stream.alloc_zeros::<u8>(input_bytes.len()).unwrap();
        let output = stream.alloc_zeros::<u8>(input_bytes.len()).unwrap();
        let protocol = stream
            .alloc_zeros::<u8>(std::mem::size_of::<ProtocolState>())
            .unwrap();
        let mut forces = stream.alloc_zeros::<f32>(6).unwrap();
        let mut masses = stream.alloc_zeros::<f32>(2).unwrap();
        let mut dt = stream.alloc_zeros::<f32>(1).unwrap();
        let mut adj = stream
            .alloc_zeros::<u8>(std::mem::size_of::<Adj>())
            .unwrap();
        let shift = stream.alloc_zeros::<f32>(3).unwrap();
        let mass = stream.alloc_zeros::<f32>(1).unwrap();
        let correction = stream.alloc_zeros::<f32>(3).unwrap();
        stream.memcpy_htod(&input_bytes, &mut input).unwrap();
        stream
            .memcpy_htod(&[2.0f32, 0.0, 0.0, 0.0, 0.0, 0.0], &mut forces)
            .unwrap();
        stream.memcpy_htod(&[2.0f32, 2.0], &mut masses).unwrap();
        stream.memcpy_htod(&[0.1f32], &mut dt).unwrap();
        let mut adj_bytes = vec![0u8; std::mem::size_of::<Adj>()];
        let dt_offset = std::mem::offset_of!(Adj, d_dt);
        adj_bytes[dt_offset..dt_offset + 8].copy_from_slice(&address(&dt, &stream).to_ne_bytes());
        let gain = std::mem::offset_of!(Adj, gasp_gain_eta);
        adj_bytes[gain..gain + 4].copy_from_slice(&1.0f32.to_ne_bytes());
        let burst = std::mem::offset_of!(Adj, force_burst_step);
        adj_bytes[burst..burst + 4].copy_from_slice(&11u32.to_ne_bytes());
        stream.memcpy_htod(&adj_bytes, &mut adj).unwrap();
        let raw = stream.cu_stream() as *mut c_void;
        // The same immutable topology map is shared; clocks are not.
        let mut anchors = [u32::MAX; 1024];
        anchors[0] = 0;
        anchors[1] = 1;
        assert_eq!(
            unsafe {
                prism_nhs::so3_project::set_residue_to_calpha(
                    anchors.as_ptr(),
                    anchors.len() as u32,
                    raw,
                )
            },
            0
        );
        stream.synchronize().unwrap();
        let p = [
            address(&input, &stream),
            address(&output, &stream),
            address(&forces, &stream),
            address(&masses, &stream),
            address(&adj, &stream),
            address(&shift, &stream),
            address(&mass, &stream),
            address(&correction, &stream),
        ];
        let clock =
            address(&protocol, &stream) + std::mem::offset_of!(ProtocolState, current_step) as u64;
        let launch = |clock: *const u32| unsafe {
            prism_apply_gradient_gasp_clock_launch(
                p[0] as _, p[1] as _, p[2] as _, p[3] as _, p[4] as _, p[5] as _, p[6] as _, clock,
                2, 2, raw,
            )
        };
        assert_ne!(
            launch(ptr::null()),
            0,
            "missing clock must fail, never use global state"
        );
        assert_ne!(
            launch((clock + 1) as *const u32),
            0,
            "unaligned clock must fail"
        );
        let mut graph = ptr::null_mut();
        let mut exec = ptr::null_mut();
        let mut nodes = 0;
        unsafe {
            cudarc::driver::set_capture_mode_active(true);
            ok(sys::cuStreamBeginCapture_v2(
                stream.cu_stream(),
                sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            ));
            assert_eq!(launch(clock as *const u32), 0);
            assert_eq!(
                prism_momentum_guard_check_launch(p[5] as _, p[6] as _, p[7] as _, p[4] as _, raw),
                0
            );
            assert_eq!(
                prism_apply_com_correction_launch(p[1] as _, p[7] as _, 2, raw),
                0
            );
            ok(sys::cuStreamEndCapture(stream.cu_stream(), &mut graph));
            cudarc::driver::set_capture_mode_active(false);
            ok(sys::cuGraphGetNodes(graph, ptr::null_mut(), &mut nodes));
            assert_eq!(
                nodes, 3,
                "capture must contain actual GASP, momentum guard and correction"
            );
            ok(sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0));
        }
        Self {
            stream,
            graph,
            exec,
            input,
            output,
            protocol,
            _forces: forces,
            _masses: masses,
            _dt: dt,
            adj,
            _shift: shift,
            _mass: mass,
            _correction: correction,
            input_bytes,
            nodes,
        }
    }
    fn clock_address(&self) -> u64 {
        address(&self.protocol, &self.stream)
            + std::mem::offset_of!(ProtocolState, current_step) as u64
    }
    fn write_clock(&self, step: u32) {
        // Synchronous fixture upload avoids host-source lifetime ambiguity. The
        // production director updates this device-owned field from its own graph.
        unsafe {
            ok(sys::cuMemcpyHtoD_v2(
                self.clock_address(),
                &step as *const u32 as _,
                4,
            ));
        }
    }
    fn verify(&self, step: u32) -> serde_json::Value {
        self.stream.synchronize().unwrap();
        let mut bytes = vec![0u8; self.input_bytes.len()];
        self.stream.memcpy_dtoh(&self.output, &mut bytes).unwrap();
        let output: Vec<RichSpike> = bytes
            .chunks_exact(std::mem::size_of::<RichSpike>())
            .map(|b| unsafe { ptr::read_unaligned(b.as_ptr() as *const RichSpike) })
            .collect();
        let expected = if step == 11 { 0.025f32 } else { 0.0025f32 };
        assert!(
            (output[0].x - expected).abs() < 3e-7,
            "stream clock{step}: {} vs{expected}",
            output[0].x
        );
        assert!((output[1].x - (1.0 - expected)).abs() < 3e-7);
        let weighted_com = 2.0 * output[0].x + 2.0 * (output[1].x - 1.0);
        assert!(
            weighted_com.abs() < 3e-7,
            "COM must remain locked after both ordinary and burst replays"
        );
        for (i, chunk) in bytes
            .chunks_exact(std::mem::size_of::<RichSpike>())
            .enumerate()
        {
            // Only positions may change; all native spike identity/metadata is retained.
            assert_eq!(&chunk[12..], &self.input_bytes[i * 64 + 12..(i + 1) * 64]);
            assert_eq!(output[i].y.to_bits(), 0);
            assert_eq!(output[i].z.to_bits(), 0);
        }
        let mut raw = vec![0u8; self.input_bytes.len()];
        self.stream.memcpy_dtoh(&self.input, &mut raw).unwrap();
        assert_eq!(raw, self.input_bytes);
        let mut adj = vec![0u8; std::mem::size_of::<Adj>()];
        self.stream.memcpy_dtoh(&self.adj, &mut adj).unwrap();
        let flag = std::mem::offset_of!(Adj, momentum_violation_flag);
        assert_eq!(
            u32::from_ne_bytes(adj[flag..flag + 4].try_into().unwrap()),
            0
        );
        let mut observed = 0u32;
        unsafe {
            ok(sys::cuMemcpyDtoH_v2(
                &mut observed as *mut u32 as _,
                self.clock_address(),
                4,
            ));
        }
        assert_eq!(observed, step, "measurement must not advance either clock");
        serde_json::json!({"step":step,"clock_ptr":self.clock_address(),"x0":output[0].x,
            "x1":output[1].x,"weighted_com_displacement":weighted_com,"nodes":self.nodes})
    }
}
impl Drop for CapturedGasp {
    fn drop(&mut self) {
        self.stream
            .synchronize()
            .expect("retire captured kernels before owners drop");
        unsafe {
            ok(sys::cuGraphExecDestroy(self.exec));
            ok(sys::cuGraphDestroy(self.graph));
        }
    }
}
#[test]
#[ignore = "GPU_RUNTIME_REQUIRED: actual native captured kernels in a standalone process"]
fn two_captured_streams_keep_live_protocol_clocks_and_com_independent() {
    let ctx = CudaContext::new(0).expect("real CUDA required; no skipped success");
    let a = CapturedGasp::new(&ctx);
    let b = CapturedGasp::new(&ctx);
    assert_ne!(a.clock_address(), b.clock_address());
    let mut receipts = Vec::new();
    for (round, (sa, sb)) in [(5, 11), (11, 5), (11, 11), (5, 5), (7, 11)]
        .into_iter()
        .enumerate()
    {
        // Both clocks are updated before either graph executes. A global clock
        // implementation cannot satisfy the differing expectations in both arms.
        a.write_clock(sa);
        b.write_clock(sb);
        unsafe {
            if round % 2 == 0 {
                ok(sys::cuGraphLaunch(a.exec, a.stream.cu_stream()));
                ok(sys::cuGraphLaunch(b.exec, b.stream.cu_stream()));
            } else {
                ok(sys::cuGraphLaunch(b.exec, b.stream.cu_stream()));
                ok(sys::cuGraphLaunch(a.exec, a.stream.cu_stream()));
            }
        }
        receipts.push(serde_json::json!({"a":a.verify(sa),"b":b.verify(sb)}));
    }
    println!(
        "{}",
        serde_json::json!({"schema":"prism.gasp_stream_clock_gpu.v1",
        "captures":2,"replays":10,"recaptures":0,"md_steps_advanced":0,"rounds":receipts})
    );
}
