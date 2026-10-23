//! Native GPU equivalence for the physical caller's actual endpoint coordinator.
//! No molecular MD is executed. All arrays are actual device allocations.
use super::*;
use anyhow::Result;
use prism_nhs::governor_device::CudaOwned;
use prism_nhs::twin_kernels::{
    endpoint_capture::{CapturedEndpointCoordinator, EndpointBuffers},
    TwinSpikePublication,
};
use std::sync::{Barrier, Mutex};

struct WorkerArrays {
    target: Target,
    events: CudaSlice<u8>,
    count: CudaSlice<i32>,
}
impl WorkerArrays {
    fn new(context: &Arc<CudaContext>) -> Self {
        let target = Target::new(context);
        let events = target.stream.alloc_zeros::<u8>(6 * 96).unwrap();
        let count = target.stream.alloc_zeros::<i32>(1).unwrap();
        target.stream.synchronize().unwrap();
        Self {
            target,
            events,
            count,
        }
    }
    fn set_endpoint(&mut self, id: usize, epoch: usize, count: usize) {
        let step = 10 + (epoch as u32) * 10;
        self.target.clock(step);
        let mut events = vec![GpuSpikeEvent::default(); 6];
        for (k, event) in events.iter_mut().enumerate() {
            event.timestep = step as i32 - (k % 3) as i32;
            event.position = [(k % 2) as f32 + 0.25, 0.25, 0.25];
            event.intensity = 0.1 + (id + k) as f32 * 0.03125;
            event.n_residues = 1;
            event.nearby_residues[0] = (id * 10 + k) as i32;
            event.phase_bits = if k % 2 == 0 { 0 } else { 512 };
        }
        let raw =
            unsafe { std::slice::from_raw_parts(events.as_ptr().cast::<u8>(), events.len() * 96) };
        self.target
            .stream
            .memcpy_htod(raw, &mut self.events)
            .unwrap();
        self.target
            .stream
            .memcpy_htod(&[count as i32], &mut self.count)
            .unwrap();
        self.target.stream.synchronize().unwrap();
    }
    fn observation(&self) -> serde_json::Value {
        let mut protocol = vec![0u8; std::mem::size_of::<ProtocolState>()];
        self.target
            .stream
            .memcpy_dtoh(&self.target.protocol, &mut protocol)
            .unwrap();
        self.target.stream.synchronize().unwrap();
        serde_json::json!({"threshold_bits":self.target.values().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),
            "protocol_bytes":protocol})
    }
}
impl EndpointBuffers for WorkerArrays {
    fn endpoint_stream(&self) -> Arc<CudaStream> {
        self.target.stream.clone()
    }
    fn endpoint_source(&self) -> Result<(&CudaSlice<u8>, &CudaSlice<i32>)> {
        Ok((&self.events, &self.count))
    }
    fn endpoint_target(
        &mut self,
    ) -> Result<(&mut CudaSlice<f32>, &CudaSlice<f32>, &mut CudaSlice<u8>)> {
        Ok((
            &mut self.target.thresholds,
            &self.target.base,
            &mut self.target.protocol,
        ))
    }
    fn endpoint_grid(&self) -> Result<((i32, i32, i32), (f32, f32, f32), f32)> {
        Ok(((2, 1, 1), (0., 0., 0.), 1.))
    }
}
fn rings(
    context: &Arc<CudaContext>,
    module: &Arc<cudarc::driver::CudaModule>,
    n: usize,
) -> Vec<Arc<Mutex<TwinRingBuffer>>> {
    let stream = context.new_stream().unwrap();
    let result = (0..4)
        .map(|group| {
            let recipients: Vec<_> = (0..n).filter(|id| id / (n / 4) != group).collect();
            Arc::new(Mutex::new(
                TwinRingBuffer::new_lossless_broadcast(context, &stream, module, 2, n, &recipients)
                    .unwrap(),
            ))
        })
        .collect();
    stream.synchronize().unwrap();
    result
}
const COUNTS: [[usize; 8]; 3] = [
    [0, 1, 2, 3, 4, 5, 6, 1],
    [6, 0, 5, 2, 1, 4, 3, 0],
    [1, 6, 0, 5, 2, 3, 4, 0],
];

fn reference_endpoints(
    context: &Arc<CudaContext>,
    module: &Arc<cudarc::driver::CudaModule>,
) -> Vec<Vec<serde_json::Value>> {
    let mut workers: Vec<_> = (0..8).map(|_| WorkerArrays::new(context)).collect();
    let mut publications: Vec<_> = workers
        .iter()
        .map(|worker| {
            TwinSpikePublication::new(&worker.target.stream, &worker.events, &worker.count).unwrap()
        })
        .collect();
    let rings = rings(context, module, 8);
    let mut output = vec![Vec::new(); 8];
    for (epoch, counts) in COUNTS.iter().enumerate() {
        for (id, worker) in workers.iter_mut().enumerate() {
            worker.set_endpoint(id, epoch, counts[id]);
        }
        for group in 0..4 {
            let mut ring = rings[group].lock().unwrap();
            for (id, worker) in workers
                .iter_mut()
                .enumerate()
                .filter(|(id, _)| id / 2 != group)
            {
                let stream = worker.target.stream.clone();
                let (t, b, p) = worker.endpoint_target().unwrap();
                ring.adapt_lossless_window(
                    &stream,
                    id,
                    0,
                    t,
                    b,
                    (2, 1, 1),
                    (0., 0., 0.),
                    1.,
                    0.3,
                    0.2,
                    500.,
                    p,
                )
                .unwrap();
                stream.synchronize().unwrap();
            }
            for id in group * 2..group * 2 + 2 {
                for _ in 0..counts[id].div_ceil(2) {
                    let worker = &workers[id];
                    let stream = worker.target.stream.clone();
                    ring.push_device_delta(
                        &stream,
                        &worker.events,
                        &worker.count,
                        &mut publications[id],
                    )
                    .unwrap();
                    stream.synchronize().unwrap();
                    for (recipient, worker) in workers
                        .iter_mut()
                        .enumerate()
                        .filter(|(r, _)| r / 2 != group)
                    {
                        let stream = worker.target.stream.clone();
                        let (t, b, p) = worker.endpoint_target().unwrap();
                        ring.adapt_lossless_window(
                            &stream,
                            recipient,
                            1,
                            t,
                            b,
                            (2, 1, 1),
                            (0., 0., 0.),
                            1.,
                            0.3,
                            0.2,
                            500.,
                            p,
                        )
                        .unwrap();
                        stream.synchronize().unwrap();
                    }
                    let stream = workers[0].target.stream.clone();
                    ring.retire_participating_recipients(&stream).unwrap();
                    stream.synchronize().unwrap();
                }
            }
            for (id, worker) in workers
                .iter_mut()
                .enumerate()
                .filter(|(id, _)| id / 2 != group)
            {
                let stream = worker.target.stream.clone();
                let (t, b, p) = worker.endpoint_target().unwrap();
                ring.adapt_lossless_window(
                    &stream,
                    id,
                    2,
                    t,
                    b,
                    (2, 1, 1),
                    (0., 0., 0.),
                    1.,
                    0.3,
                    0.2,
                    500.,
                    p,
                )
                .unwrap();
                stream.synchronize().unwrap();
                ring.validate_lossless_window(&stream, id).unwrap();
            }
        }
        for (id, worker) in workers.iter_mut().enumerate() {
            publications[id]
                .validate_archived_population(
                    &worker.target.stream,
                    &worker.events,
                    &worker.count,
                    counts[id],
                )
                .unwrap();
            output[id].push(worker.observation());
            worker
                .target
                .stream
                .memset_zeros(&mut worker.count)
                .unwrap();
            worker.target.stream.synchronize().unwrap();
            publications[id]
                .acknowledge_source_reset(&worker.target.stream, &worker.events, &worker.count)
                .unwrap();
        }
    }
    output
}

#[test]
#[ignore = "requires real CUDA conditional graphs and exact source-built PRISM_RING_TEST_PTX"]
fn captured_endpoint_matches_physical_ordered_host_batches_and_reuses_graph() {
    let context = CudaContext::new(0).unwrap();
    let module = context
        .load_module(cudarc::nvrtc::Ptx::from_file(
            std::env::var("PRISM_RING_TEST_PTX").unwrap(),
        ))
        .unwrap();
    let expected = reference_endpoints(&context, &module);
    let rings = rings(&context, &module, 8);
    let coordinator = CapturedEndpointCoordinator::new(&context, rings.clone(), 8).unwrap();
    let actual = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..8)
            .map(|id| {
                let context = context.clone();
                let coordinator = coordinator.clone();
                let rings = rings.clone();
                scope.spawn(move || {
                    context.bind_to_thread().unwrap();
                    let arrays = WorkerArrays::new(&context);
                    let stream = arrays.target.stream.clone();
                    let mut owner = CudaOwned::new(arrays, stream.clone());
                    coordinator.bind_owner(&mut owner);
                    let publication =
                        TwinSpikePublication::new(&stream, &owner.events, &owner.count).unwrap();
                    let mut publication = CudaOwned::new(Some(publication), stream.clone());
                    coordinator.bind_owner(&mut publication);
                    let mut output = Vec::new();
                    for (epoch, counts) in COUNTS.iter().enumerate() {
                        owner.set_endpoint(id, epoch, counts[id]);
                        coordinator
                            .prepare_worker(id, &mut owner, &mut publication)
                            .unwrap();
                        assert_eq!(
                            coordinator
                                .exchange(id, 10 + epoch as u32 * 10, || false)
                                .unwrap(),
                            epoch as u64 + 1
                        );
                        publication
                            .as_ref()
                            .unwrap()
                            .validate_archived_population(
                                &stream,
                                &owner.events,
                                &owner.count,
                                counts[id],
                            )
                            .unwrap();
                        output.push(owner.observation());
                        let expected_count: usize = COUNTS[..=epoch]
                            .iter()
                            .map(|counts| {
                                counts
                                    .iter()
                                    .enumerate()
                                    .filter(|(producer, _)| producer / 2 != id / 2)
                                    .map(|(_, count)| count)
                                    .sum::<usize>()
                            })
                            .sum();
                        let mut actual_count = 0;
                        for group in 0..4 {
                            if group == id / 2 {
                                continue;
                            }
                            let receipt = rings[group]
                                .lock()
                                .unwrap()
                                .recipient_receipt(&stream, id)
                                .unwrap();
                            assert_eq!(receipt.total_overwritten, 0);
                            assert_eq!(receipt.read_windows, epoch as u64 + 1);
                            actual_count += receipt.total_inspected;
                        }
                        assert_eq!(actual_count, expected_count as u64);
                        stream.memset_zeros(&mut owner.count).unwrap();
                        stream.synchronize().unwrap();
                        publication
                            .as_mut()
                            .unwrap()
                            .acknowledge_source_reset(&stream, &owner.events, &owner.count)
                            .unwrap();
                    }
                    // A completed endpoint remains successful if an earlier-returning
                    // peer retires its owners before this peer leaves the rendezvous.
                    output
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert_eq!(actual,expected,"all target threshold bits and complete protocol/steering bytes must match native host sequence");
    let telemetry = coordinator.execution_telemetry();
    assert_eq!(telemetry["parent_graph_launches"], 3);
    assert_eq!(telemetry["completed_endpoints"], 3);
    println!(
        "{}",
        serde_json::json!({"schema_kind":"prism.captured_endpoint_equivalence.v1",
        "physical_md_steps":0,"streams":8,"source_groups":4,"physical_producers_per_group":2,
        "variable_source_counts":COUNTS,"ring_capacity":2,"endpoints":3,
        "threshold_bits_and_complete_protocol_bytes_equal":true,"zero_overwritten":true,
        "normal_owner_retirement_executed":true,"telemetry":telemetry})
    );
}

#[test]
#[ignore = "requires real CUDA and exact source-built PRISM_RING_TEST_PTX"]
fn captured_endpoint_owner_panic_releases_waiters_before_array_destruction() {
    let context = CudaContext::new(0).unwrap();
    let module = context
        .load_module(cudarc::nvrtc::Ptx::from_file(
            std::env::var("PRISM_RING_TEST_PTX").unwrap(),
        ))
        .unwrap();
    let coordinator =
        CapturedEndpointCoordinator::new(&context, rings(&context, &module, 4), 4).unwrap();
    let ready = Arc::new(Barrier::new(4));
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|id| {
                let context = context.clone();
                let coordinator = coordinator.clone();
                let ready = ready.clone();
                scope.spawn(move || {
                    context.bind_to_thread().unwrap();
                    let arrays = WorkerArrays::new(&context);
                    let stream = arrays.target.stream.clone();
                    let mut owner = CudaOwned::new(arrays, stream.clone());
                    coordinator.bind_owner(&mut owner);
                    owner.set_endpoint(id, 0, 6);
                    let publication =
                        TwinSpikePublication::new(&stream, &owner.events, &owner.count).unwrap();
                    let mut publication = CudaOwned::new(Some(publication), stream);
                    coordinator.bind_owner(&mut publication);
                    coordinator
                        .prepare_worker(id, &mut owner, &mut publication)
                        .unwrap();
                    ready.wait();
                    if id == 3 {
                        panic!("deliberate endpoint owner unwind before launch");
                    }
                    assert!(coordinator.exchange(id, 10, || false).is_err());
                })
            })
            .collect();
        for (id, handle) in handles.into_iter().enumerate() {
            assert_eq!(handle.join().is_err(), id == 3);
        }
    });
    assert_eq!(
        coordinator.execution_telemetry()["parent_graph_launches"],
        0
    );
}
