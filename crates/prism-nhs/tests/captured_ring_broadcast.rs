//! Native coupled ring delivery and cancellation; no molecular dynamics.
#![cfg(feature = "gpu")]

#[cfg(test)]
mod coupled_window_boundary_tests {
    use prism_nhs::rescue_controller::CoupledWindowBarrier;
    #[test]
    fn coupled_boundary_reuses_generations_and_releases_on_exit() {
        let b = CoupledWindowBarrier::new(3);
        std::thread::scope(|s| {
            for id in 0..3 {
                let b = b.clone();
                s.spawn(move || {
                    let _participant = b.participant(id);
                    for _ in 0..30 { b.wait(|| false).unwrap(); }
                });
            }
        });
        assert!(b.is_aborted());
        assert!(b.wait(|| false).is_err());
    }
    #[test]
    fn coupled_boundary_failed_participant_releases_waiter() {
        let b = CoupledWindowBarrier::new(2);
        std::thread::scope(|s| {
            let peer = b.clone();
            let h = s.spawn(move || peer.wait(|| false));
            drop(b.participant(1));
            assert!(h.join().unwrap().is_err());
        });
    }
    #[test]
    fn coupled_boundary_observes_external_cancel() {
        let b = CoupledWindowBarrier::new(2);
        let cancelled = std::sync::atomic::AtomicBool::new(false);
        std::thread::scope(|s| {
            let h = s.spawn(|| b.wait(|| cancelled.load(std::sync::atomic::Ordering::Acquire)));
            cancelled.store(true, std::sync::atomic::Ordering::Release);
            assert!(h.join().unwrap().is_err());
        });
    }
}

use cudarc::driver::{sys, CudaContext, CudaSlice, CudaStream};
use prism_nhs::{fused_engine::GpuSpikeEvent, protocol_state::ProtocolState,
    twin_kernels::TwinRingBuffer};
use std::{ptr, sync::Arc};
fn ok(r: sys::CUresult) { assert_eq!(r,sys::CUresult::CUDA_SUCCESS); }
struct Graph { graph: sys::CUgraph, exec: sys::CUgraphExec, stream: Arc<CudaStream> }
impl Graph {
    fn capture(stream: &Arc<CudaStream>, f: impl FnOnce()) -> Self {
        stream.synchronize().unwrap();
        let mut graph=ptr::null_mut();let mut exec=ptr::null_mut();
        unsafe {
            cudarc::driver::set_capture_mode_active(true);
            ok(sys::cuStreamBeginCapture_v2(stream.cu_stream(),sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
            f();
            ok(sys::cuStreamEndCapture(stream.cu_stream(),&mut graph));
            cudarc::driver::set_capture_mode_active(false);
            let mut nodes=0;ok(sys::cuGraphGetNodes(graph,ptr::null_mut(),&mut nodes));assert_eq!(nodes,1);
            ok(sys::cuGraphInstantiateWithFlags(&mut exec,graph,0));
        }
        Self{graph,exec,stream:stream.clone()}
    }
    fn launch(&self){unsafe{ok(sys::cuGraphLaunch(self.exec,self.stream.cu_stream()));}}
}
impl Drop for Graph {fn drop(&mut self){self.stream.synchronize().unwrap();unsafe{ok(sys::cuGraphExecDestroy(self.exec));ok(sys::cuGraphDestroy(self.graph));}}}
struct Target { stream: Arc<CudaStream>, thresholds:CudaSlice<f32>, base:CudaSlice<f32>, protocol:CudaSlice<u8> }
impl Target {
    fn new(ctx:&Arc<CudaContext>)->Self{
        let stream=ctx.new_stream().unwrap();
        let mut thresholds=stream.alloc_zeros::<f32>(2).unwrap();let mut base=stream.alloc_zeros::<f32>(2).unwrap();
        stream.memcpy_htod(&[1.0f32;2],&mut thresholds).unwrap();stream.memcpy_htod(&[1.0f32;2],&mut base).unwrap();
        let protocol=stream.alloc_zeros::<u8>(std::mem::size_of::<ProtocolState>()).unwrap();
        stream.synchronize().unwrap();Self{stream,thresholds,base,protocol}
    }
    fn clock(&mut self,step:u32){
        let mut bytes=vec![0u8;std::mem::size_of::<ProtocolState>()];
        let o=std::mem::offset_of!(ProtocolState,current_step);bytes[o..o+4].copy_from_slice(&step.to_ne_bytes());
        self.stream.memcpy_htod(&bytes,&mut self.protocol).unwrap();self.stream.synchronize().unwrap();
    }
    fn values(&self)->Vec<f32>{let mut v=vec![0f32;2];self.stream.memcpy_dtoh(&self.thresholds,&mut v).unwrap();self.stream.synchronize().unwrap();v}
}
fn source_bytes(step:i32)->Vec<u8>{
    let mut events=vec![GpuSpikeEvent::default();6];
    for i in 0..2 {events[i].timestep=step;events[i].position=[i as f32+0.25,0.25,0.25];events[i].intensity=0.1*(i+1) as f32;events[i].n_residues=1;events[i].nearby_residues[0]=i as i32;}
    unsafe{std::slice::from_raw_parts(events.as_ptr().cast::<u8>(),events.len()*std::mem::size_of::<GpuSpikeEvent>()).to_vec()}
}
#[test]
#[ignore="requires real CUDA and source-built PRISM_RING_TEST_PTX"]
fn captured_ring_delivers_each_window_to_all_recipients_with_live_clocks(){
    let ctx=CudaContext::new(0).expect("real GPU");let source=ctx.new_stream().unwrap();
    let ptx=std::env::var("PRISM_RING_TEST_PTX").expect("exact source-built PTX");
    let module=ctx.load_module(cudarc::nvrtc::Ptx::from_file(&ptx)).unwrap();
    assert_recorded_phase_and_residue(&ctx, &source, &module);
    assert_lossless_publication_and_retirement(&ctx, &source, &module);
    let mut ring=TwinRingBuffer::new_broadcast(&ctx,&source,&module,4,3).unwrap();
    let mut bytes=source.alloc_zeros::<u8>(6*std::mem::size_of::<GpuSpikeEvent>()).unwrap();
    let mut count=source.alloc_zeros::<i32>(1).unwrap();source.memcpy_htod(&source_bytes(5),&mut bytes).unwrap();source.memcpy_htod(&[2],&mut count).unwrap();source.synchronize().unwrap();
    let producer=Graph::capture(&source,||ring.push_device(&source,&bytes,&count).unwrap());
    let mut targets:Vec<_>=(0..3).map(|_|Target::new(&ctx)).collect();
    let mut readers=Vec::new();
    for (id,t) in targets.iter_mut().enumerate(){t.clock(10+10*id as u32);let stream=t.stream.clone();
        readers.push(Graph::capture(&stream,||ring.read_and_adapt_for_recipient(&stream,id,&mut t.thresholds,&t.base,(2,1,1),(0.0,0.0,0.0),1.0,0.3,0.2,500.0,&mut t.protocol).unwrap()));}
    producer.launch();source.synchronize().unwrap();
    let mut expected=vec![[1f32;2];3];let mut rows=Vec::new();
    // Three windows with new events; intervening replay is deliberately empty.
    for window in 0..3 {
        if window>0 {
            source.memcpy_htod(&source_bytes(45),&mut bytes).unwrap();source.synchronize().unwrap();
            for (id,t) in targets.iter_mut().enumerate(){t.clock(40+10*id as u32);}
            for _ in 0..if window==2 {5}else{1}{producer.launch();}
            source.synchronize().unwrap();
        }
        for id in [2usize,0,1]{readers[id].launch();}
        for t in &targets{t.stream.synchronize().unwrap();}
        for id in 0..3 {
            let step=if window==0 {10+10*id as u32}else{40+10*id as u32};
            let event_step=if window==0{5}else{45};
            let age=(step as i32-event_step).max(0) as f32;
            for _ in 0..if window==2{2}else{1}{for voxel in 0..2{expected[id][voxel]=(expected[id][voxel]-0.3*(0.1*(voxel+1) as f32)*(-age/500.0).exp()).max(0.8);}}
            let actual=targets[id].values();for j in 0..2{assert!((actual[j]-expected[id][j]).abs()<1e-6,"native aging/threshold mismatch");}
            let receipt=ring.recipient_receipt(&source,id).unwrap();assert_eq!(receipt.last_completed_step,step);
            assert_eq!(receipt.cursor,if window==0{2}else if window==1{4}else{14});
            assert_eq!(receipt.total_inspected,if window==0{2}else if window==1{4}else{8});
            assert_eq!(receipt.total_overwritten,if window==2{6}else{0});
            rows.push(serde_json::json!({"window":window,"recipient":id,"thresholds":actual,"receipt":receipt}));
        }
        if window==0 {
            let mut legacy=Target::new(&ctx);
            ring.read_and_adapt(&legacy.stream,&mut legacy.thresholds,&legacy.base,(2,1,1),(0.0,0.0,0.0),1.0,0.3,0.2,10,500.0).unwrap();
            assert_eq!(legacy.values(),targets[0].values(),"existing single-consumer numerical oracle");
            for reader in &readers{reader.launch();}for t in &targets{t.stream.synchronize().unwrap();}
            for id in 0..3 {assert_eq!(ring.recipient_receipt(&source,id).unwrap().total_inspected,2);}
        }
    }
    // Empty and invalid source counts cannot invent entries or read beyond allocation.
    for n in [0,7] {source.memcpy_htod(&[n],&mut count).unwrap();producer.launch();source.synchronize().unwrap();}
    for reader in &readers{reader.launch();}for t in &targets{t.stream.synchronize().unwrap();}
    for id in 0..3 {let r=ring.recipient_receipt(&source,id).unwrap();assert_eq!(r.cursor,14);assert_eq!(r.total_inspected,8);assert_eq!(r.read_windows,5);}
    println!("{}",serde_json::json!({"schema":"prism.native_captured_ring_broadcast.v1","physical_md_steps":0,"recipient_count":3,"captured_graph_count":4,"recaptures":0,"legacy_oracle_exact":true,"native_phase_codes_preserved":[0,1,512,1023],"hotspot_residue_decoded":true,"source_capacity":6,"ring_capacity":4,"rows":rows,"scope":"native GPU event-exchange fixture; not molecular TWIN16 execution"}));
}

fn assert_recorded_phase_and_residue(ctx:&Arc<CudaContext>,stream:&Arc<CudaStream>,module:&Arc<cudarc::driver::CudaModule>){
    let mut gpu=TwinRingBuffer::new_broadcast(ctx,stream,module,8,1).unwrap();
    let mut cpu=TwinRingBuffer::new(ctx,stream,module,8).unwrap();
    let mut events=vec![GpuSpikeEvent::default();4];
    for (i,phase) in [0u32,1,512,1023].into_iter().enumerate(){
        events[i].phase_bits=phase;events[i].timestep=300;events[i].position=[0.25;3];
        events[i].n_residues=if i==2 {0}else{1};events[i].nearby_residues[0]=if i==3 {19}else{17+i as i32};
        events[i].intensity=if i==3{0.9}else{0.01};
    }
    let raw=unsafe{std::slice::from_raw_parts(events.as_ptr().cast::<u8>(),events.len()*96)};
    let mut db=stream.alloc_zeros::<u8>(raw.len()).unwrap();let mut count=stream.alloc_zeros::<i32>(1).unwrap();
    stream.memcpy_htod(raw,&mut db).unwrap();stream.memcpy_htod(&[4],&mut count).unwrap();
    let mut target=Target::new(ctx);target.clock(300); // Endpoint has zero phase fields; events disagree deliberately.
    gpu.push_device_with_phase(stream,&db,&count,Some(&target.protocol)).unwrap();
    cpu.push_compacted(stream,&events).unwrap();stream.synchronize().unwrap();
    let mut records=Vec::new();
    for ring in [&gpu,&cpu] {
        let mut b=vec![0u8;8*48];stream.memcpy_dtoh(&ring.buffer,&mut b).unwrap();stream.synchronize().unwrap();
        let mut rows:Vec<Vec<u8>>=b[..4*48].chunks_exact(48).map(|b|b.to_vec()).collect();rows.sort();records.push(rows);
    }
    assert_eq!(records[0],records[1],"CPU/GPU compactors must preserve all48 event bytes");
    let mut packed:Vec<_>=records[0].iter().map(|r|u32::from_ne_bytes(r[44..48].try_into().unwrap())).collect();packed.sort();
    assert_eq!(packed,vec![17,(1<<16)|18,(512<<16)|0xffff,(1023<<16)|19]);
    // Exercise the existing hotspot branch (>20%) with fixture inputs only.
    gpu.read_and_adapt_for_recipient(&target.stream,0,&mut target.thresholds,&target.base,(2,1,1),(0.0,0.0,0.0),1.0,1.0,0.5,500.0,&mut target.protocol).unwrap();
    let mut ps=vec![0u8;std::mem::size_of::<ProtocolState>()];target.stream.memcpy_dtoh(&target.protocol,&mut ps).unwrap();target.stream.synchronize().unwrap();
    let o=std::mem::offset_of!(ProtocolState,steering_focus_residue);assert_eq!(i32::from_ne_bytes(ps[o..o+4].try_into().unwrap()),19,"plain residue ID, never packed phase word");
    // An unresolved residue remains -1 in native observation telemetry.
    gpu.reset(stream).unwrap();let raw=unsafe{std::slice::from_raw_parts((&events[2] as *const GpuSpikeEvent).cast::<u8>(),96)};
    let mut absent=stream.alloc_zeros::<u8>(96).unwrap();stream.memcpy_htod(raw,&mut absent).unwrap();stream.memcpy_htod(&[1],&mut count).unwrap();
    gpu.push_device(stream,&absent,&count).unwrap();stream.synchronize().unwrap();target.clock(300);
    gpu.read_and_adapt_for_recipient(&target.stream,0,&mut target.thresholds,&target.base,(2,1,1),(0.0,0.0,0.0),1.0,0.3,0.2,500.0,&mut target.protocol).unwrap();
    target.stream.memcpy_dtoh(&target.protocol,&mut ps).unwrap();target.stream.synchronize().unwrap();
    let o=std::mem::offset_of!(ProtocolState,last_seen_spike_residue);assert_eq!(i32::from_ne_bytes(ps[o..o+4].try_into().unwrap()),-1);
}

// Primitive-only validation. Captured publication/read/retirement do not advance MD.
// Keeping this inside the existing entrypoint makes the recorded ring GPU runner
// exercise both policies without claiming that the production caller is integrated.
fn assert_lossless_publication_and_retirement(
    ctx: &Arc<CudaContext>, source: &Arc<CudaStream>, module: &Arc<cudarc::driver::CudaModule>,
) {
    use prism_nhs::twin_kernels::TwinSpikePublication;
    use cudarc::driver::DevicePtr;
    let mut ring = TwinRingBuffer::new_lossless_broadcast(ctx,source,module,2,4,&[1,3]).unwrap();
    assert!(TwinRingBuffer::new_lossless_broadcast(ctx,source,module,3,4,&[1,3]).is_err());
    assert!(TwinRingBuffer::new_lossless_broadcast(ctx,source,module,2,4,&[1,1]).is_err());
    assert!(TwinRingBuffer::new_lossless_broadcast(ctx,source,module,2,4,&[4]).is_err());
    assert!(TwinRingBuffer::new_lossless_broadcast(ctx,source,module,2,4,&[]).is_err());
    let mut events = vec![GpuSpikeEvent::default();6];
    for (i,e) in events.iter_mut().enumerate() {
        e.timestep=5; e.position=[0.25;3]; e.intensity=0.01;
        e.n_residues=1; e.nearby_residues[0]=i as i32;
        e.phase_bits=[0,1,512,1023,37,71][i];
    }
    let raw=unsafe{std::slice::from_raw_parts(events.as_ptr().cast::<u8>(),events.len()*96)};
    let mut bytes=source.alloc_zeros::<u8>(raw.len()).unwrap();
    let mut count=source.alloc_zeros::<i32>(1).unwrap();
    source.memcpy_htod(raw,&mut bytes).unwrap(); source.memcpy_htod(&[6],&mut count).unwrap();
    let mut publication=TwinSpikePublication::new(source,&bytes,&count).unwrap();
    source.synchronize().unwrap();
    let producer=Graph::capture(source,||ring.push_device_delta(source,&bytes,&count,&mut publication).unwrap());
    let mut targets:Vec<_>=(0..2).map(|_|Target::new(ctx)).collect();
    let mut readers=Vec::new();
    for (j,t) in targets.iter_mut().enumerate() {
        t.clock(10+10*j as u32); let stream=t.stream.clone(); let id=[1usize,3][j];
        readers.push(Graph::capture(&stream,||ring.read_and_adapt_for_recipient(&stream,id,&mut t.thresholds,&t.base,(2,1,1),(0.0,0.0,0.0),1.0,0.3,0.2,500.0,&mut t.protocol).unwrap()));
    }
    let retirement=Graph::capture(source,||ring.retire_participating_recipients(source).unwrap());
    assert!(ring.push_device(source,&bytes,&count).is_err());
    assert!(ring.push_compacted(source,&events).is_err());
    assert!(ring.reset(source).is_err());
    {
        let t=&mut targets[0];
        assert!(ring.read_and_adapt_for_recipient(&t.stream,0,&mut t.thresholds,&t.base,(2,1,1),(0.0,0.0,0.0),1.0,0.3,0.2,500.0,&mut t.protocol).is_err());
        assert!(ring.read_and_adapt(&t.stream,&mut t.thresholds,&t.base,(2,1,1),(0.0,0.0,0.0),1.0,0.3,0.2,10,500.0).is_err());
    }
    let mut other_ring=TwinRingBuffer::new_lossless_broadcast(ctx,source,module,2,4,&[1,3]).unwrap();
    assert!(other_ring.push_device_delta(source,&bytes,&count,&mut publication).is_err());
    let alien=source.alloc_zeros::<u8>(raw.len()).unwrap();
    assert!(ring.push_device_delta(source,&alien,&count,&mut publication).is_err());
    producer.launch();source.synchronize().unwrap();
    assert_eq!(publication.published_count(source).unwrap(),2);
    assert!(publication.archive_and_reset(source,&bytes,&count,||panic!("unpublished source must not reach archive callback")).is_err());
    let mut packed=Vec::new();
    for wave in 0..3 {
        if wave>0 {producer.launch();source.synchronize().unwrap();}
        let mut storage=vec![0u8;96];source.memcpy_dtoh(&ring.buffer,&mut storage).unwrap();source.synchronize().unwrap();
        for b in storage.chunks_exact(48) {packed.push(u32::from_ne_bytes(b[44..48].try_into().unwrap()));}
        readers[0].launch();targets[0].stream.synchronize().unwrap();
        retirement.launch();source.synchronize().unwrap();
        assert_eq!(read_u32(source,&ring.tail),2*wave,"late physical recipient must retain every unread record");
        producer.launch();source.synchronize().unwrap();
        assert_eq!(publication.published_count(source).unwrap(),2*(wave as i32+1),"pressure must not consume deferred source suffix");
        readers[1].launch();targets[1].stream.synchronize().unwrap();
        retirement.launch();source.synchronize().unwrap();
        assert_eq!(read_u32(source,&ring.tail),2*(wave+1));
    }
    packed.sort_unstable();let mut expected:Vec<u32>=events.iter().enumerate().map(|(i,e)|(e.phase_bits<<16)|i as u32).collect();expected.sort_unstable();assert_eq!(packed,expected,"all events including recorded phase zero delivered without duplicate or restamp");
    for (j,id) in [1usize,3].into_iter().enumerate() {
        let r=ring.recipient_receipt(source,id).unwrap(); assert_eq!(r.cursor,6); assert_eq!(r.total_inspected,6); assert_eq!(r.total_overwritten,0);assert_eq!(r.last_completed_step,10+10*j as u32);
        // Every delivered event independently applies the existing native formula.
        let age=(10+10*j as i32-5) as f32;
        let mut expected=1f32;for _ in 0..6{expected=(expected-0.3*0.01*(-age/500.0).exp()).max(0.8);}
        assert!((targets[j].values()[0]-expected).abs()<1e-6);
    }
    for id in [0usize,2]{assert_eq!(ring.recipient_receipt(source,id).unwrap().cursor,0,"unused same-group slot remains outside retirement population");}
    assert_eq!(read_u32(source,&ring.overflow),1,"pressure is observable without event loss");
    ring.validate_lossless_retirement(source).unwrap();
    producer.launch();source.synchronize().unwrap(); assert_eq!(publication.published_count(source).unwrap(),6);
    assert!(publication.archive_and_reset(source,&bytes,&count,||anyhow::bail!("archive failure fixture")).is_err());
    assert_eq!(publication.published_count(source).unwrap(),6);
    assert!(publication.archive_and_reset(source,&bytes,&count,||Ok(())).is_err(),"callback must reset the archived source");
    // Fixture stands in for the production archive callback only here; no claim of
    // durable molecular archival. Production source integration remains separate.
    publication.archive_and_reset(source,&bytes,&count,||{
        let (ptr,_guard)=count.device_ptr(source);
        unsafe{ok(sys::cuMemsetD32Async(ptr,0,1,source.cu_stream()));}
        source.synchronize()?;Ok(())
    }).unwrap();assert_eq!(publication.published_count(source).unwrap(),0);
    source.memcpy_htod(&[2],&mut count).unwrap();producer.launch();source.synchronize().unwrap();
    assert_eq!(publication.published_count(source).unwrap(),2,"new source epoch republishes from zero");
    // An unannounced source reset must not silently reuse the old cursor.
    source.memcpy_htod(&[0],&mut count).unwrap();
    ring.push_device_delta(source,&bytes,&count,&mut publication).unwrap();source.synchronize().unwrap();
    assert!(publication.published_count(source).is_err());
    // Invalid device count is an explicit device error, not truncated success.
    let mut bad=TwinSpikePublication::new(source,&bytes,&count).unwrap();
    source.memcpy_htod(&[7],&mut count).unwrap();let before=read_u32(source,&ring.head);
    ring.push_device_delta(source,&bytes,&count,&mut bad).unwrap();source.synchronize().unwrap();
    assert!(bad.published_count(source).is_err());assert_eq!(read_u32(source,&ring.head),before);
    source.memcpy_htod(&[-1],&mut count).unwrap();
    let mut negative=TwinSpikePublication::new(source,&bytes,&count).unwrap();
    ring.push_device_delta(source,&bytes,&count,&mut negative).unwrap();source.synchronize().unwrap();
    assert!(negative.published_count(source).is_err());assert_eq!(read_u32(source,&ring.head),before);
    assert_lossless_wrap_retirement(ctx,source,module);
    assert_distinct_source_cursors(ctx,source,module);
    println!("{}",serde_json::json!({"schema":"prism.native_lossless_delta_primitive.v1","physical_md_steps":0,"source_records":6,"participating_recipients":[1,3],"unused_receipts":[0,2],"ring_capacity":2,"delivered_per_recipient":6,"overwritten":0,"phase_zero_preserved":true,"source_identity_rejection":true,"archive_failure_retains_cursor":true,"invalid_counts_rejected":true,"wrap_exercised":true,"scope":"captured native primitive fixture; production coupled graph integration not executed"}));
}
fn read_u32(stream:&Arc<CudaStream>,v:&CudaSlice<u32>)->u32{
    let mut b=[0u32];stream.memcpy_dtoh(v,&mut b).unwrap();stream.synchronize().unwrap();b[0]
}
fn assert_lossless_wrap_retirement(ctx:&Arc<CudaContext>,stream:&Arc<CudaStream>,module:&Arc<cudarc::driver::CudaModule>){
    use cudarc::driver::{DevicePtr,LaunchConfig,PushKernelArg};
    use prism_nhs::twin_kernels::TwinSpikePublication;
    let mut ring=TwinRingBuffer::new_lossless_broadcast(ctx,stream,module,4,2,&[0,1]).unwrap();
    stream.memcpy_htod(&[u32::MAX-1],&mut ring.head).unwrap();stream.memcpy_htod(&[u32::MAX-1],&mut ring.tail).unwrap();
    let raw=source_bytes(5);let mut bytes=stream.alloc_zeros::<u8>(raw.len()).unwrap();let mut count=stream.alloc_zeros::<i32>(1).unwrap();
    stream.memcpy_htod(&raw,&mut bytes).unwrap();stream.memcpy_htod(&[4],&mut count).unwrap();
    let mut publication=TwinSpikePublication::new(stream,&bytes,&count).unwrap();
    ring.push_device_delta(stream,&bytes,&count,&mut publication).unwrap();stream.synchronize().unwrap();
    assert_eq!(read_u32(stream,&ring.head),2);assert_eq!(publication.published_count(stream).unwrap(),4);
    // Direct receipt allocations let this fixture seed counters near wrap without
    // introducing a production API that can mutate registered reader chronology.
    let mut a=stream.alloc_zeros::<u8>(40).unwrap();let mut b=stream.alloc_zeros::<u8>(40).unwrap();
    let mut seed=vec![0u8;40];seed[..4].copy_from_slice(&u32::MAX.to_ne_bytes());stream.memcpy_htod(&seed,&mut a).unwrap();seed[..4].copy_from_slice(&0u32.to_ne_bytes());stream.memcpy_htod(&seed,&mut b).unwrap();
    let pointers=[a.device_ptr(stream).0,b.device_ptr(stream).0];let mut table=stream.alloc_zeros::<u64>(2).unwrap();stream.memcpy_htod(&pointers,&mut table).unwrap();
    let mut failure=stream.alloc_zeros::<u32>(1).unwrap();let kernel=module.load_function("ring_buffer_retire_participating_recipients").unwrap();
    let mut retire=||unsafe{stream.launch_builder(&kernel).arg(&table).arg(&2u32).arg(&ring.head).arg(&mut ring.tail).arg(&4u32).arg(&mut failure).launch(LaunchConfig{grid_dim:(1,1,1),block_dim:(1,1,1),shared_mem_bytes:0}).unwrap();};
    retire();stream.synchronize().unwrap();drop(retire);
    assert_eq!(read_u32(stream,&ring.tail),u32::MAX,"modular oldest cursor differs from numerical minimum");assert_eq!(read_u32(stream,&failure),0);
    seed[..4].copy_from_slice(&2u32.to_ne_bytes());stream.memcpy_htod(&seed,&mut a).unwrap();stream.memcpy_htod(&seed,&mut b).unwrap();
    unsafe{stream.launch_builder(&kernel).arg(&table).arg(&2u32).arg(&ring.head).arg(&mut ring.tail).arg(&4u32).arg(&mut failure).launch(LaunchConfig{grid_dim:(1,1,1),block_dim:(1,1,1),shared_mem_bytes:0}).unwrap();}
    stream.synchronize().unwrap();assert_eq!(read_u32(stream,&ring.tail),2);assert_eq!(read_u32(stream,&failure),0);
    seed[..4].copy_from_slice(&3u32.to_ne_bytes());stream.memcpy_htod(&seed,&mut a).unwrap();
    unsafe{stream.launch_builder(&kernel).arg(&table).arg(&2u32).arg(&ring.head).arg(&mut ring.tail).arg(&4u32).arg(&mut failure).launch(LaunchConfig{grid_dim:(1,1,1),block_dim:(1,1,1),shared_mem_bytes:0}).unwrap();}
    stream.synchronize().unwrap();assert_eq!(read_u32(stream,&failure),8,"future recipient cursor fails explicitly");assert_eq!(read_u32(stream,&ring.tail),2);
}

fn assert_distinct_source_cursors(ctx:&Arc<CudaContext>,stream:&Arc<CudaStream>,module:&Arc<cudarc::driver::CudaModule>){
    use prism_nhs::twin_kernels::TwinSpikePublication;
    let mut ring=TwinRingBuffer::new_lossless_broadcast(ctx,stream,module,4,1,&[0]).unwrap();
    let peers=[ctx.new_stream().unwrap(),ctx.new_stream().unwrap()];
    let mut sources=Vec::new();
    for (id,peer) in peers.iter().enumerate(){
        let mut bytes=peer.alloc_zeros::<u8>(6*96).unwrap();let mut count=peer.alloc_zeros::<i32>(1).unwrap();
        peer.memcpy_htod(&source_bytes(100+id as i32),&mut bytes).unwrap();peer.memcpy_htod(&[2],&mut count).unwrap();
        let publication=TwinSpikePublication::new(peer,&bytes,&count).unwrap();peer.synchronize().unwrap();
        sources.push((bytes,count,publication));
    }
    stream.synchronize().unwrap();
    let mut graphs=Vec::new();
    for (id,(bytes,count,publication)) in sources.iter_mut().enumerate(){
        graphs.push(Graph::capture(&peers[id],||ring.push_device_delta(&peers[id],bytes,count,publication).unwrap()));
    }
    // Both captured source graphs target one group ring; neither is synchronized
    // between launches. Readers are withheld until both completion fences.
    for g in &graphs{g.launch();}for peer in &peers{peer.synchronize().unwrap();}
    assert_eq!(read_u32(stream,&ring.head),4);
    for (id,(_,_,p)) in sources.iter().enumerate(){assert_eq!(p.published_count(&peers[id]).unwrap(),2);}
    let mut b=vec![0u8;4*48];stream.memcpy_dtoh(&ring.buffer,&mut b).unwrap();stream.synchronize().unwrap();
    let mut steps:Vec<_>=b.chunks_exact(48).map(|r|i32::from_ne_bytes(r[..4].try_into().unwrap())).collect();steps.sort_unstable();
    assert_eq!(steps,vec![100,100,101,101],"group publication retains distinct physical sources");
    let mut target=Target::new(ctx);target.clock(110);
    ring.read_and_adapt_for_recipient(&target.stream,0,&mut target.thresholds,&target.base,(2,1,1),(0.0,0.0,0.0),1.0,0.3,0.2,500.0,&mut target.protocol).unwrap();target.stream.synchronize().unwrap();
    ring.retire_participating_recipients(stream).unwrap();stream.synchronize().unwrap();ring.validate_lossless_retirement(stream).unwrap();
    assert_eq!(read_u32(stream,&ring.tail),4);assert_eq!(ring.recipient_receipt(stream,0).unwrap().total_inspected,4);
}

/// Batching must preserve the complete one-window nonlinear update, including
/// the FIRST strongest hotspot when subsequent records tie at the floor.
#[test]
#[ignore="requires real CUDA and source-built PRISM_RING_TEST_PTX"]
fn lossless_batches_match_one_window_nonlinear_adaptation() {
    use prism_nhs::twin_kernels::TwinSpikePublication;
    let ctx=CudaContext::new(0).unwrap(); let source=ctx.new_stream().unwrap();
    let module=ctx.load_module(cudarc::nvrtc::Ptx::from_file(std::env::var("PRISM_RING_TEST_PTX").unwrap())).unwrap();
    let mut events=vec![GpuSpikeEvent::default();6];
    for (i,e) in events.iter_mut().enumerate() {
        e.timestep=10; e.position=[0.25;3]; e.intensity=1.0;
        e.n_residues=1; e.nearby_residues[0]=7+i as i32; e.phase_bits=0;
    }
    let raw=unsafe {std::slice::from_raw_parts(events.as_ptr().cast::<u8>(),events.len()*96)};
    let mut bytes=source.alloc_zeros::<u8>(raw.len()).unwrap(); source.memcpy_htod(raw,&mut bytes).unwrap();
    let mut count=source.alloc_zeros::<i32>(1).unwrap(); source.memcpy_htod(&[6],&mut count).unwrap();
    let mut whole=TwinRingBuffer::new_lossless_broadcast(&ctx,&source,&module,8,3,&[0,2]).unwrap();
    let mut bounded=TwinRingBuffer::new_lossless_broadcast(&ctx,&source,&module,2,3,&[0,2]).unwrap();
    let mut wp=TwinSpikePublication::new(&source,&bytes,&count).unwrap();
    let mut bp=TwinSpikePublication::new(&source,&bytes,&count).unwrap();
    let mut reference:Vec<_>=(0..2).map(|_|Target::new(&ctx)).collect();
    let mut batched:Vec<_>=(0..2).map(|_|Target::new(&ctx)).collect();
    for idx in 0..2 {reference[idx].clock(10+idx as u32*10);batched[idx].clock(10+idx as u32*10);}
    source.synchronize().unwrap();
    let push=Graph::capture(&source,||whole.push_device_delta(&source,&bytes,&count,&mut wp).unwrap());
    push.launch();source.synchronize().unwrap();
    for (idx,target) in reference.iter_mut().enumerate() {
        whole.read_and_adapt_for_recipient(&target.stream,idx*2,&mut target.thresholds,&target.base,
            (2,1,1),(0.,0.,0.),1.,0.3,0.5,500.,&mut target.protocol).unwrap();
        target.stream.synchronize().unwrap();
    }
    for (idx,target) in batched.iter_mut().enumerate() {
        let stream=target.stream.clone();
        let graph=Graph::capture(&stream,||bounded.adapt_lossless_window(&stream,idx*2,0,
            &mut target.thresholds,&target.base,(2,1,1),(0.,0.,0.),1.,0.3,0.5,500.,&mut target.protocol).unwrap());
        graph.launch();stream.synchronize().unwrap();
    }
    let push=Graph::capture(&source,||bounded.push_device_delta(&source,&bytes,&count,&mut bp).unwrap());
    for _ in 0..3 {
        push.launch();source.synchronize().unwrap();
        for (idx,target) in batched.iter_mut().enumerate() {
            let stream=target.stream.clone();
            let graph=Graph::capture(&stream,||bounded.adapt_lossless_window(&stream,idx*2,1,
                &mut target.thresholds,&target.base,(2,1,1),(0.,0.,0.),1.,0.3,0.5,500.,&mut target.protocol).unwrap());
            graph.launch();stream.synchronize().unwrap();
        }
        bounded.retire_participating_recipients(&source).unwrap();source.synchronize().unwrap();
    }
    for (idx,target) in batched.iter_mut().enumerate() {
        let stream=target.stream.clone();
        let graph=Graph::capture(&stream,||bounded.adapt_lossless_window(&stream,idx*2,2,
            &mut target.thresholds,&target.base,(2,1,1),(0.,0.,0.),1.,0.3,0.5,500.,&mut target.protocol).unwrap());
        graph.launch();stream.synchronize().unwrap();
        bounded.validate_lossless_window(&stream,idx*2).unwrap();
        assert_eq!(target.values().iter().map(|x|x.to_bits()).collect::<Vec<_>>(),
            reference[idx].values().iter().map(|x|x.to_bits()).collect::<Vec<_>>());
        let mut actual=vec![0u8;std::mem::size_of::<ProtocolState>()];let mut expected=actual.clone();
        stream.memcpy_dtoh(&target.protocol,&mut actual).unwrap();stream.synchronize().unwrap();
        reference[idx].stream.memcpy_dtoh(&reference[idx].protocol,&mut expected).unwrap();reference[idx].stream.synchronize().unwrap();
        assert_eq!(actual,expected,"nonlinear protocol/steering state must match one full window");
        let offset=std::mem::offset_of!(ProtocolState,steering_focus_residue);
        assert_eq!(i32::from_ne_bytes(actual[offset..offset+4].try_into().unwrap()),8);
        let receipt=bounded.recipient_receipt(&stream,idx*2).unwrap();
        assert_eq!(receipt.total_inspected,6);assert_eq!(receipt.total_overwritten,0);assert_eq!(receipt.read_windows,1);
    }
    bounded.validate_lossless_retirement(&source).unwrap();
    bp.validate_archived_population(&source,&bytes,&count,6).unwrap();
    assert!(bp.validate_archived_population(&source,&bytes,&count,5).is_err());
    assert!(bp.acknowledge_source_reset(&source,&bytes,&count).is_err());
    source.memcpy_htod(&[0],&mut count).unwrap();
    bp.acknowledge_source_reset(&source,&bytes,&count).unwrap();
    assert_eq!(bp.published_count(&source).unwrap(),0);
}

#[path = "captured_ring_broadcast/endpoint_capture.rs"]
mod captured_endpoint_tests;
